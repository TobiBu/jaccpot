"""Reverse-mode gradients of the fused multi-GPU force, end to end (WP-G, G4).

``make_mesh_grad_fn`` (``FusedMeshForce.differentiable_shard`` in the force's
``shard_map``) with the real cross seams: G2's merged M2L reverse (seam A) and G3's
cross near reverse (seam B), both sending OUTPUT cotangents forward along the frozen
cross layout. The loss is built OUTSIDE the ``shard_map``. Gates, on CPU (the S9
interpret set, the GPU's target-sorted symmetric walk interpreted, the ``buf``
exchange), N = 2048 uniform cube with unequal masses, leaf 64, p4, theta 0.8, fp64,
with non-empty cross far AND near pairs on every device:

* MG4: central differences of the full pipeline (rebuild included) at h = 1e-6 in a
  random direction of the positions, of the masses, and in ``G`` and ``eps2``, with
  every device's cross pair hashes identical at +-h, agree with the reverse to ~1e-9;
* MG5/MG6: the gradients in x and m, matched by global id, are FMM-class against the
  direct sum (AD of a dense fp64 sum with the same pair kernel), as accurate as the
  single card's (error ratio in [0.67, 1.5]) and close to it;
* MG8: a corrupted pair hash on one device makes every live gradient row NaN;
* MG9: ``G * G_bar == sum(m * m_bar)`` (the force is linear in G and in the source
  masses) to round-off, and ``eps2_bar`` equals the single card's (only near pairs,
  evaluated exactly, see the softening of a compact kernel).

Each case compiles for minutes in interpret mode (``slow``, ``distributed_grad``).

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_mesh_grad_gates.py -q -n 0 -m distributed_grad
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yggdrax.distributed.comm as ygg_comm

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from jaccpot import FastMultipoleMethod, FusedForceProvider  # noqa: E402
from jaccpot.config import (  # noqa: E402
    FarFieldConfig,
    FMMAdvancedConfig,
    NearFieldConfig,
    TreeConfig,
)
from jaccpot.distributed.cross import CrossCapacities  # noqa: E402
from jaccpot.distributed.rollout import decompose  # noqa: E402

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the fused multi-GPU lane needs >= 2 devices"
    ),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="fp64 gates"),
    pytest.mark.slow,
    pytest.mark.distributed_grad,
]

AXIS = "gpus"
N = 2048
# p3, not p4: the interpret-mode reverse program's compile peaks at ~7.5 GB at p3 and
# ~15 GB at p4 (N does not matter), and CI's ubuntu-latest runner has 16 GB. The
# gates check the gradient's correctness, which does not depend on the order; the
# p4 numbers are in the WP-G record (PR #510).
LEAF, LEAF_CAP, ORDER, THETA, SOFT, EPS = 64, 128, 3, 0.8, 1e-2, 1e-4
H = 1e-6

_ENV = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "65536",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "32",
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_WALK_PALLAS_INTERPRET": "1",
}
_ENV_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_M2L_CSR",
    "JACCPOT_STATIC_STRICT_FUSED_WALK",
    "JACCPOT_CROSS_WALK",
)


@pytest.fixture(scope="module")
def lane_on_cpu():
    """The S9 lane on CPU for the module (module-scoped: the cases share programs)."""
    from jaccpot.runtime import _level_shapes

    mp = pytest.MonkeyPatch()
    for key in _ENV_UNSET:
        mp.delenv(key, raising=False)
    for key, value in _ENV.items():
        mp.setenv(key, value)
    mp.setattr(jax, "default_backend", lambda: "gpu")
    resolve = ygg_comm.resolve_ragged_method
    mp.setattr(
        ygg_comm,
        "resolve_ragged_method",
        lambda method="auto": "buf" if method == "auto" else resolve(method),
    )
    mp.setattr(_level_shapes, "_WIDTHS", {})
    mp.setattr(_level_shapes, "_LEVELS", {})
    yield
    mp.undo()


def _solver(mac):
    extra = (
        dict(
            adaptive_eps=EPS,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if mac == "dehnen_error"
        else {}
    )
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFT,
        working_dtype=jnp.float64,
        use_pallas=True,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=LEAF_CAP,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            farfield=FarFieldConfig(m2l_chunk_size=512),
            mac_type=mac,
        ),
        fixed_order=ORDER,
        **extra,
    )


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


class _Case:
    """One mesh, its force, the loss's weights, the reverse at the base point."""

    def __init__(self, ndev, mac, seed=0):
        from jaccpot.distributed import grad as GR
        from jaccpot.distributed.template_force import FusedMeshForce
        from jaccpot.runtime.fused_topology import ForceParams

        self.ndev, self.mac = ndev, mac
        rng = np.random.default_rng(seed)
        self.pos = rng.uniform(-1.0, 1.0, (N, 3))
        vel = 0.1 * rng.normal(size=(N, 3))
        self.mass = rng.uniform(0.5, 1.5, N) / N
        self.mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))
        cap = int(N / ndev * 1.3)
        self.shard, _ = decompose(
            self.mesh, self.pos, vel, self.mass, cap=cap, num_samples=64
        )
        force = FusedMeshForce(
            _solver(mac),
            self.mesh,
            leaf_size=LEAF,
            max_order=ORDER,
            theta=THETA,
            cross_caps=CrossCapacities(max_cells=LEAF_CAP, leaf_width=LEAF),
        )
        force.prepare(self.shard.positions, self.shard.masses, self.shard.count)
        force.with_grad_plan(self.shard.positions, self.shard.masses, self.shard.count)
        assert force.grad_statics.far_pairs_symmetric
        self.force = force
        self.scale = force.scale_carry
        # the criterion's force accepts only the engine's own parameters
        self.use_params = not self.scale
        ids = np.asarray(self.shard.ids)
        count = np.asarray(self.shard.count)
        self.rows = np.concatenate(
            [np.arange(d * cap, d * cap + count[d]) for d in range(ndev)]
        )
        self.gid = ids[self.rows]
        assert sorted(self.gid.tolist()) == list(range(N))
        self.w = rng.normal(size=(N, 3))
        w_mesh = np.zeros((ndev * cap, 3))
        w_mesh[self.rows] = self.w[self.gid]
        sh = NamedSharding(self.mesh, P(AXIS))
        self.sharding = sh
        self.w_mesh = jax.device_put(jnp.asarray(w_mesh), sh)
        self.G0 = jnp.asarray(1.0, jnp.float64)
        self.E0 = jnp.asarray(SOFT**2, jnp.float64)

        captured: dict = {}
        self.corrupt = False
        real = GR.CrossGradPlan.from_sink.__func__
        case = self

        def spy(cls, sink):
            plan = real(cls, sink)
            if case.corrupt:
                me = jax.lax.axis_index(AXIS)
                bump = jnp.where(me == 0, jnp.uint64(1), jnp.uint64(0))
                fields = {f: getattr(plan, f) for f in plan.__dataclass_fields__}
                fields["pair_hashes"] = plan.pair_hashes.at[:, 0, 0].add(bump)
                plan = cls(**fields)
            captured["plan"] = plan
            return plan

        self._spy = classmethod(spy)
        self._GR = GR
        fn = GR.make_mesh_grad_fn(force)
        seed_fb = force.force_scale_seed if self.scale else None
        spec = P(AXIS)

        def probe_body(x, m, n, i, g, e2, *fb):
            # the production entry point's body plus the trace's cross counts and
            # pair hashes (the spied plan of THIS trace)
            params = ForceParams(G=g, eps2=e2) if self.use_params else None
            out = force.differentiable_shard(
                x, m, jnp.minimum(n[0], x.shape[0]), i, *fb, params=params
            )
            plan = captured["plan"]
            far = jnp.sum(plan.far_sources >= 0)
            live = jnp.arange(plan.near_targets.shape[0]) < plan.near_count
            near = jnp.sum(live & (plan.near_targets >= 0))
            return (
                out[0],
                out[1].reshape(1),
                far.reshape(1),
                near.reshape(1),
                plan.pair_hashes.reshape(1, -1),
                out[2].reshape(1, -1),
            )

        fb_args = () if seed_fb is None else (seed_fb,)
        probe = jax.shard_map(
            probe_body,
            mesh=self.mesh,
            in_specs=(spec,) * 4 + (P(), P()) + (spec,) * len(fb_args),
            out_specs=(spec,) * 6,
            check_vma=False,
        )

        def fwd(x, m, g, e2):
            return probe(x, m, self.shard.count, self.shard.ids, g, e2, *fb_args)

        def grad_fwd(x, m, g, e2):
            # the user-facing program (make_mesh_grad_fn)
            params = ForceParams(G=g, eps2=e2) if self.use_params else None
            return fn(x, m, self.shard.count, self.shard.ids, params, seed_fb)[0]

        self.grad_fwd = grad_fwd
        self.fwd = fwd

    def loss(self, x, m, g, e2):
        return jnp.sum(self.grad_fwd(x, m, g, e2) * self.w_mesh)

    def run(self, mp):
        mp.setattr(self._GR.CrossGradPlan, "from_sink", self._spy)
        x0, m0 = self.shard.positions, self.shard.masses
        force = self.force
        with force.context():
            force.before_trace()
            out = jax.jit(self.fwd)(x0, m0, self.G0, self.E0)
            self.bits = np.asarray(out[1])
            self.far, self.near = np.asarray(out[2]), np.asarray(out[3])
            # the cross pairs' hashes and the local lists' needs (their counts)
            self.hashes = np.asarray(out[4])
            self.needs = np.asarray(out[5])
            force.before_trace()
            self.L, grads = jax.jit(
                jax.value_and_grad(self.loss, argnums=(0, 1, 2, 3))
            )(x0, m0, self.G0, self.E0)
        self.gx, self.gm = np.asarray(grads[0]), np.asarray(grads[1])
        self.gG, self.ge2 = float(grads[2]), float(grads[3])
        return self

    def by_id(self, g):
        out = np.zeros((N,) + g.shape[1:])
        out[self.gid] = g[self.rows]
        return out


_CASES: dict = {}


def _case(ndev, mac):
    if jax.device_count() < ndev:
        pytest.skip(f"needs {ndev} devices")
    key = (ndev, mac)
    if key not in _CASES:
        mp = pytest.MonkeyPatch()
        try:
            _CASES[key] = _Case(ndev, mac).run(mp)
        finally:
            mp.undo()
    return _CASES[key]


def _direct_grads(case):
    from jaccpot.softening import masked_pair_factors, softening_params_from_sq

    kernel = getattr(case.force.engine, "softening_kernel", None)
    w = jnp.asarray(case.w)

    def loss(x, m, g, e2):
        d = x[None, :, :] - x[:, None, :]
        r2 = jnp.sum(d * d, -1)
        params = softening_params_from_sq(kernel, e2, x.dtype)
        inv = masked_pair_factors(r2, r2 > 0, params, kernel)[0]
        acc = g * jnp.einsum(
            "ij,ijk->ik", m[None, :] * inv, d, precision=jax.lax.Precision.HIGHEST
        )
        return jnp.sum(acc * w)

    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(
        jnp.asarray(case.pos), jnp.asarray(case.mass), case.G0, case.E0
    )
    return [np.asarray(g) for g in grads]


def _single_grads(case):
    from jaccpot.distributed.fused import host_mesh_bounds
    from jaccpot.runtime.fused_topology import ForceParams

    xs, ms = jnp.asarray(case.pos), jnp.asarray(case.mass)
    box = host_mesh_bounds(
        [case.pos], [N], cubic=bool(case.force.engine._tree_cubic_bounds)
    )
    single, _ = FusedForceProvider.prepare(
        _solver(case.mac),
        xs,
        ms,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        bounds=box,
        differentiable=True,
    )
    w = jnp.asarray(case.w)

    def loss(x, m, g, e2):
        stop = jax.lax.stop_gradient
        topo = single.rebuild(stop(x), stop(m), bounds=box, keep_far_list=True)
        plan, _ = single.grad_plan(topo)
        acc = single.differentiable_accelerations(
            topo, x, m, ForceParams(G=g, eps2=e2), plan=plan
        )
        return jnp.sum(acc * w)

    grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(xs, ms, case.G0, case.E0)
    return [np.asarray(g) for g in grads]


def _assert_case_sane(case):
    print(f"ndev={case.ndev} {case.mac}: cross far {case.far}, near {case.near}")
    assert np.all(case.far > 0) and np.all(case.near > 0), (case.far, case.near)
    assert int(np.bitwise_or.reduce(case.bits)) == 0, case.bits
    assert np.all(np.isfinite(case.gx)) and np.all(np.isfinite(case.gm))
    pad = np.setdiff1d(np.arange(case.gx.shape[0]), case.rows)
    assert np.all(case.gx[pad] == 0.0) and np.all(case.gm[pad] == 0.0)
    assert float(np.max(np.abs(case.gm))) > 0.0


def _grid():
    cases = [(2, "dehnen"), (2, "dehnen_error")]
    if jax.device_count() >= 4:
        cases.append((4, "dehnen"))
    return cases


@pytest.mark.parametrize("ndev, mac", _grid())
def test_gradient_matches_direct_sum_and_single_card(lane_on_cpu, ndev, mac):
    """MG5/MG6 (+ MG9's Euler identity and eps2_bar where params are traced)."""
    case = _case(ndev, mac)
    _assert_case_sane(case)
    dgx, dgm, _, _ = _direct_grads(case)
    gx, gm = case.by_id(case.gx), case.by_id(case.gm)
    ex, em = _rel(gx, dgx), _rel(gm, dgm)
    print(f"vs direct: x {ex:.3e} m {em:.3e}")
    # FMM-class at p4 / theta 0.8 (measured ~7e-4 / 4e-3); a dropped cross term is O(1)
    assert ex < 3e-3 and em < 1.5e-2, (ex, em)
    if case.use_params:
        sgx, sgm, _, sge2 = _single_grads(case)
        sx, sm = _rel(sgx, dgx), _rel(sgm, dgm)
        print(
            f"single vs direct: x {sx:.3e} m {sm:.3e}; mesh vs single x "
            f"{_rel(gx, sgx):.3e} m {_rel(gm, sgm):.3e}"
        )
        assert 0.67 <= ex / sx <= 1.5 and 0.67 <= em / sm <= 1.5, (ex, sx, em, sm)
        assert _rel(gx, sgx) <= ex and _rel(gm, sgm) <= em
        # MG9
        lhs = float(case.G0) * case.gG
        rhs = float(
            np.sum(np.asarray(case.shard.masses)[case.rows] * case.gm[case.rows])
        )
        assert abs(lhs - rhs) <= 1e-12 * abs(lhs), (lhs, rhs)
        assert abs(case.gG - float(case.L)) <= 1e-12 * abs(case.gG)
        np.testing.assert_allclose(case.ge2, float(sge2), rtol=1e-10)


@pytest.mark.parametrize("ndev, mac", [(2, "dehnen"), (2, "dehnen_error")])
def test_gradient_matches_central_differences(lane_on_cpu, ndev, mac):
    """MG4: the full pipeline's central differences, topology unchanged at +-h."""
    case = _case(ndev, mac)
    rng = np.random.default_rng(100)
    nrow = case.gx.shape[0]
    v = np.zeros((nrow, 3))
    v[case.rows] = rng.normal(size=(case.rows.size, 3))
    u = np.zeros(nrow)
    u[case.rows] = rng.normal(size=case.rows.size)
    vj = jax.device_put(jnp.asarray(v), case.sharding)
    uj = jax.device_put(jnp.asarray(u), case.sharding)
    x0, m0, G0, E0 = case.shard.positions, case.shard.masses, case.G0, case.E0

    def value(x, m, g, e2):
        acc, _, _, _, hashes, needs = case.fwd(x, m, g, e2)
        return jnp.sum(acc * case.w_mesh), (hashes, needs)

    mp = pytest.MonkeyPatch()
    mp.setattr(case._GR.CrossGradPlan, "from_sink", case._spy)
    try:
        with case.force.context():
            case.force.before_trace()
            value_j = jax.jit(value)

            def central(f):
                (lp, (hp, np_)), (lm, (hm, nm)) = f(H), f(-H)
                # the topology is the base point's at +-h: the cross pairs (order-free
                # hashes) and the local lists' counts
                assert np.array_equal(np.asarray(hp), case.hashes)
                assert np.array_equal(np.asarray(hm), case.hashes)
                assert np.array_equal(np.asarray(np_), case.needs)
                assert np.array_equal(np.asarray(nm), case.needs)
                return (float(lp) - float(lm)) / (2 * H)

            checks = {
                "x": (
                    central(lambda s: value_j(x0 + s * vj, m0, G0, E0)),
                    float(np.sum(case.gx * v)),
                ),
                "m": (
                    central(lambda s: value_j(x0, m0 * (1 + s * uj), G0, E0)),
                    float(np.sum(case.gm * np.asarray(m0) * u)),
                ),
            }
            if case.use_params:
                checks["G"] = (
                    central(lambda s: value_j(x0, m0, G0 * (1 + s), E0)),
                    case.gG,
                )
                checks["eps2"] = (
                    central(lambda s: value_j(x0, m0, G0, E0 * (1 + s))),
                    case.ge2 * float(E0),
                )
    finally:
        mp.undo()
    for name, (fd, an) in checks.items():
        print(f"MG4 {name}: fd {fd:.12e} reverse {an:.12e}")
        assert abs(fd - an) <= 1e-8 * abs(an), (name, fd, an)


def test_a_corrupted_pair_hash_gives_a_nan_gradient(lane_on_cpu):
    """MG8: one device's far hash off by one -> the mesh verdict fails -> NaN."""
    case = _case(2, "dehnen")
    mp = pytest.MonkeyPatch()
    mp.setattr(case._GR.CrossGradPlan, "from_sink", case._spy)
    case.corrupt = True
    try:
        with case.force.context():
            case.force.before_trace()
            grads = jax.jit(jax.grad(case.loss, argnums=(0, 1)))(
                case.shard.positions, case.shard.masses, case.G0, case.E0
            )
    finally:
        case.corrupt = False
        mp.undo()
    gx = np.asarray(grads[0])
    assert np.all(np.isnan(gx[case.rows]))
