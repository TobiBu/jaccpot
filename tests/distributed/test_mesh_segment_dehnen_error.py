"""Dehnen's error criterion on the fused mesh lane's segment runner (S12, gates a-c).

``mac_type='dehnen_error'`` (eq 16a at ``adaptive_eps = 1e-4``, the galaxy preset's
MAC) on the template route: eq (16b)'s per-particle ``f_b`` rides the segment carry,
each device publishes its nodes' eq (16a) table rows with its summary, and the
symmetric exchange's cross walks accept by the criterion the local walk runs.

The CPU end-to-end pattern of ``test_mesh_segment_runner.py`` (the large-N profile
reached on CPU with ``jax.default_backend`` reporting ``"gpu"``, the all-gather
``"buf"`` exchange, the Pallas kernels interpreted), at its toy size: N = 2048
uniform cube, leaf 64, p4, eps 1e-2 (ferrers3), with the CSR near-field lane the
criterion needs for its near f_b (interpret mode).

* (a) a one-device mesh segment equals a user-written single-card scan over the
  provider carrying f_b, from the same seed in the same box, bitwise: state,
  acceleration AND f_b.
* (b) two devices: both derive the identical cross pair sets from the identical
  gathered table; every cross far pair passes eq (16a) evaluated independently
  (float64) on that table, every leaf-leaf near pair fails it; the mesh's f_b is
  the single card's to its estimator's spread and holds the cross terms (dropped,
  it falls to the local-only sum); the lanes-kernel route (the GPU's) sums the
  cross far terms as the separate pass does; rel-L2 against the direct sum in class
  with the single card; 2 + 2 steps equal 4, f_b carried across.
* (c) a segment that outgrows its capacities under the criterion is re-planned
  and equals a fresh run with the final capacities, bitwise (f_b included).

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_mesh_segment_dehnen_error.py -q
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yggdrax.distributed.comm as ygg_comm

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from jaccpot import FastMultipoleMethod  # noqa: E402
from jaccpot.config import (  # noqa: E402
    FarFieldConfig,
    FMMAdvancedConfig,
    NearFieldConfig,
    TreeConfig,
)
from jaccpot.distributed.rollout import (  # noqa: E402
    decompose,
    make_reference_direct_force,
)

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the fused multi-GPU lane needs >= 2 devices"
    ),
    pytest.mark.slow,
    pytest.mark.distributed_segment,
]

AXIS = "gpus"
NDEV = 2
N = 2048
LEAF = 64
ORDER = 4
THETA = 0.8
SOFT = 1e-2
EPS = 1e-4
CAP = int(N / NDEV * 1.3)
DT = 1e-3
#: the flat M2L's chunk switch (test_mesh_segment_runner.py's M2L_CHUNK)
M2L_CHUNK = 512


@pytest.fixture(scope="module")
def cpu_fused_lane():
    """The segment tests' CPU switches, plus the CSR near-field lane (eq 16b's near
    half is its force-scale lane; interpret mode, the prepacked payload)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax, "default_backend", lambda: "gpu")
        resolve = ygg_comm.resolve_ragged_method
        mp.setattr(
            ygg_comm,
            "resolve_ragged_method",
            lambda method="auto": "buf" if method == "auto" else resolve(method),
        )
        mp.setenv("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1")
        mp.setenv("JACCPOT_NEARFIELD_LEAFPAIR_CSR", "1")
        mp.setenv("JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB", "0")
        mp.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "0")
        mp.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")
        mp.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")
        yield


@pytest.fixture
def fresh_level_registry(monkeypatch):
    """An empty level-shape registry (process-global, never lowered)."""
    from jaccpot.runtime import _level_shapes

    monkeypatch.setattr(_level_shapes, "_WIDTHS", {})
    monkeypatch.setattr(_level_shapes, "_LEVELS", {})


def _solver(leaf_capacity=64):
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFT,
        working_dtype=jnp.float32,
        use_pallas=True,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=leaf_capacity,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            farfield=FarFieldConfig(m2l_chunk_size=M2L_CHUNK),
            mac_type="dehnen_error",
        ),
        fixed_order=ORDER,
        adaptive_eps=EPS,
        adaptive_error_model="dehnen_paper",
        mac_force_scale_mode="paper_fb",
    )


def _mesh(ndev=NDEV):
    return jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))


def _ic(seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, (N, 3)).astype(np.float32)
    vel = (0.1 * rng.normal(size=(N, 3))).astype(np.float32)
    mass = np.full(N, 1.0 / N, np.float32)
    return pos, vel, mass


def _mesh_state(mesh, pos, vel, mass, cap=CAP):
    from jaccpot.distributed.segment import MeshState

    shard, _ = decompose(mesh, pos, vel, mass, cap=cap, num_samples=64)
    return MeshState.from_shard_state(shard, mesh)


def _live_rows(count, cap=CAP):
    count = np.asarray(count)
    return np.concatenate(
        [np.arange(d * cap, d * cap + int(count[d])) for d in range(count.shape[0])]
    )


def _by_id(values, ids, rows, n=N):
    values = np.asarray(values)[rows]
    out = np.empty((n,) + values.shape[1:], values.dtype)
    out[np.asarray(ids)[rows]] = values
    return out


def _assert_bitwise(got, want, what):
    got, want = np.asarray(got), np.asarray(want)
    differ = int(np.sum(got != want))
    assert differ == 0, (
        f"{what}: {differ} of {got.size} elements differ, max "
        f"{float(np.nanmax(np.abs(got.astype(np.float64) - want))):.3e}"
    )


def _seed_in_input_order(solver, topology, n):
    """The single card's eq (16b) seed (its caps-only prepare's) in input order, as
    ``strict_run_v2``'s particle carry reads it."""
    perm = np.asarray(topology.prepared.stub.tree.particle_indices)
    seed = np.zeros(n, np.float32)
    seed[perm] = np.asarray(solver._impl._last_force_scale_particles)
    return seed


# --------------------------------------------------------------------------
# Gate (a): one device, the single card's particle carry under the criterion.
# --------------------------------------------------------------------------


def test_a_one_device_segment_is_the_single_card_particle_carry_with_its_force_scale(
    cpu_fused_lane, fresh_level_registry
):
    """GATE (a): a one-device mesh segment equals a user-written single-card scan
    over the provider that carries f_b (each force's thresholds from the previous
    force's), started from the single card's caps-only seed, in the same box:
    state, acceleration and f_b bitwise. The mesh's seed is that seed."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from jaccpot import FusedForceProvider
    from jaccpot.distributed.fused import cube_bounds, host_mesh_bounds
    from jaccpot.distributed.segment import MeshSegmentRunner, MeshState

    n_one, steps = N // 2, 3
    mesh = _mesh(1)
    rng = np.random.default_rng(1)
    pos = rng.uniform(-1.0, 1.0, (n_one, 3)).astype(np.float32)
    vel = (0.1 * rng.normal(size=(n_one, 3))).astype(np.float32)
    mass = np.full(n_one, 1.0 / n_one, np.float32)
    packed = np.stack([pos, vel], axis=1)
    sharding = NamedSharding(mesh, P(AXIS))
    state = MeshState(
        state=jax.device_put(packed, sharding),
        masses=jax.device_put(mass, sharding),
        ids=jax.device_put(np.arange(n_one, dtype=np.int32), sharding),
        count=jax.device_put(np.asarray([n_one], np.int32), sharding),
    )
    runner = MeshSegmentRunner.fused(
        _solver(), mesh, state, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    assert runner.scale_carry
    mesh_seed = np.asarray(runner.force.force_scale_seed)
    report = runner.run(steps, dt=DT)
    assert report.bits == 0 and report.retries == 0, report
    got_state = np.asarray(runner.state.state)
    got_acc = np.asarray(runner.state.acceleration)
    got_fb = np.asarray(runner.state.force_scale)

    # the single card: its own solver and caps-only prepare, in the same box
    single = _solver()
    box0 = host_mesh_bounds([pos], [n_one])
    prov, topology = FusedForceProvider.prepare(
        single,
        jnp.asarray(pos),
        jnp.asarray(mass),
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        bounds=box0,
        caps_only=True,
    )
    assert prov.scale_carry
    seed = _seed_in_input_order(single, topology, n_one)
    _assert_bitwise(mesh_seed, seed, "the prepare's f_b seed")

    def force(x, m, f_b):
        box = cube_bounds(jnp.min(x, axis=0), jnp.max(x, axis=0), pad=1e-6)
        topo = prov.rebuild(x, m, bounds=box, force_scale=f_b)
        acc, f_b_new = prov.accelerations(topo, force_scale=True)
        return acc, f_b_new, prov.capacity_ok(topo)

    @jax.jit
    def start(s, m, f_b):
        acc, f_b_new, ok = force(s[:, 0], m, f_b)
        return jnp.asarray(acc, s.dtype), jnp.asarray(f_b_new, s.dtype), ok

    @jax.jit
    def segment(s, a, f_b, m, dt):
        def step(carry, _):
            s_now, a_now, f_b_now, ok = carry
            x = s_now[:, 0] + s_now[:, 1] * dt + 0.5 * a_now * dt**2
            s_now = s_now.at[:, 0].set(x)
            a_new, f_b_new, ok_new = force(x, m, f_b_now)
            a_new = jnp.asarray(a_new, s_now.dtype)
            v = s_now[:, 1] + 0.5 * (a_now + a_new) * dt
            carry = (s_now.at[:, 1].set(v), a_new, jnp.asarray(f_b_new, s.dtype))
            return carry + (ok & ok_new,), None

        init = (s, a, f_b, jnp.asarray(True))
        return jax.lax.scan(step, init, None, length=steps)[0]

    s0, m0 = jnp.asarray(packed), jnp.asarray(mass)
    a0, f_b0, ok0 = start(s0, m0, jnp.asarray(seed))
    want_state, want_acc, want_fb, ok = segment(
        s0, a0, f_b0, m0, jnp.asarray(DT, s0.dtype)
    )
    assert bool(ok0) and bool(ok)
    assert not np.array_equal(np.asarray(want_state), packed), "nothing moved"
    assert np.all(np.asarray(want_fb) > 0)
    _assert_bitwise(got_state, want_state, "state")
    _assert_bitwise(got_acc, want_acc, "acceleration")
    _assert_bitwise(got_fb, want_fb, "f_b")


# --------------------------------------------------------------------------
# Gate (b): two devices.
# --------------------------------------------------------------------------

#: what the probe reads off the cross hook's record, per device
_RECORD = (
    "gathered",
    "walk_far_a",
    "walk_far_b",
    "walk_far_count",
    "walk_near_a",
    "walk_near_b",
    "walk_near_count",
    "imp_far_rows",
    "imp_far_sizes",
    "cross_masses",
)


def _probe(force, state, seed):
    """The start force at ``state`` from ``seed``, with the hook's record per device.

    A program of its own (a fresh trace on every call): ``force.shard`` under the
    criterion, its ``(acceleration, f_b, bits)`` and what the symmetric exchange's
    walk read and emitted on each device.
    """
    from jax.sharding import PartitionSpec as P

    record = force.cross_record

    def body(x, m, count, ids, f_b):
        live = jnp.minimum(count[0], x.shape[0])
        acc, bits, _, f_b_new = force.shard(x, m, live, ids, force_scale=f_b)
        out = {k: jnp.asarray(record[k])[None] for k in _RECORD}
        return acc, f_b_new, jnp.asarray(bits)[None], out

    spec = P(AXIS)
    program = jax.jit(
        jax.shard_map(
            body,
            mesh=force.mesh,
            in_specs=(spec,) * 5,
            out_specs=(spec, spec, spec, {k: spec for k in _RECORD}),
            check_vma=False,
        )
    )
    with force.context():
        force.before_trace()
        acc, f_b, bits, out = program(
            state.state[:, 0], state.masses, state.count, state.ids, seed
        )
    return acc, f_b, np.asarray(bits), {k: np.asarray(v) for k, v in out.items()}


def _pair_set(out, device, kind):
    a = out[f"walk_{kind}_a"][device]
    b = out[f"walk_{kind}_b"][device]
    live = (np.arange(a.shape[0]) < int(out[f"walk_{kind}_count"][device])) & (
        (a >= 0) & (b >= 0)
    )
    return set(zip(np.minimum(a, b)[live].tolist(), np.maximum(a, b)[live].tolist()))


def _eq16a(gathered, pairs, *, order, theta_max, floor):
    """Eq (15)/(16a), symmetrised, in float64 from the gathered summary rows alone.

    Written out here, independent of ``dehnen_pair_accept``: with ``a = rho_A / r``,
    ``b = rho_B / r``, ``E_{A->B} = 8 max(rho) / (rho_A + rho_B) sum_n C(p, n)
    s_n(A) a^n b^(p-n)`` (``s_0 = 1``), accepted when ``G M_A E_{A->B} / r^2 <
    thr_B`` both ways, ``rho_A + rho_B < theta_max r`` and the gap clears the
    separation floor. Returns the worst direction's ratio to its threshold, the
    convergence ratio and the relative gap past the floor.
    """
    rows = np.asarray(gathered, np.float64).reshape(-1, gathered.shape[-1])
    a = np.array([p for p, _ in pairs])
    b = np.array([q for _, q in pairs])
    centre, radius = rows[:, 0:3], rows[:, 3]
    gm, thr, s = rows[:, 8], rows[:, 9], rows[:, 10 : 10 + order]
    ra, rb = radius[a], radius[b]
    r = np.linalg.norm(centre[a] - centre[b], axis=1)
    x, y = ra / r, rb / r
    e_ab, e_ba = y**order, x**order
    for n in range(1, order + 1):
        c = math.comb(order, n)
        e_ab = e_ab + c * s[a, n - 1] * x**n * y ** (order - n)
        e_ba = e_ba + c * s[b, n - 1] * y**n * x ** (order - n)
    scale = 8.0 * np.maximum(ra, rb) / np.maximum(ra + rb, 1e-24) / r**2
    ratio = np.maximum(gm[a] * e_ab * scale / thr[b], gm[b] * e_ba * scale / thr[a])
    return ratio, (ra + rb) / (theta_max * r), (r - ra - rb - floor) / r


def _spread(x, y):
    rel = np.abs(np.asarray(x, np.float64) / np.asarray(y, np.float64) - 1.0)
    return float(np.median(rel)), float(np.max(rel))


def test_b_two_devices_decide_every_cross_pair_alike_and_carry_the_cross_force_scale(
    cpu_fused_lane, monkeypatch
):
    """GATE (b), the start force on two devices (one solver, one prepare).

    * Both devices hold the identical gathered summary (geometry + eq (16a) table
      rows) and derive the identical cross far and near pair sets from it.
    * Every cross far pair passes eq (16a) evaluated in float64 on that table
      (convergent, clear of the separation floor); every leaf-leaf near pair fails
      it (the control: the evaluation can tell).
    * f_b: the mesh's against the single card's on the same particles (different
      trees, the same estimator: near pairs exact, far ones as monopoles) and
      against the exact sum; with the cross terms dropped (the cross near lane and
      the cross far own terms zeroed) it falls to the local-only sum's spread.
    * The lanes-kernel route (CSR M2L + Pallas cascade, the GPU's; interpret
      mode) sums the cross far terms into f_b as the separate pass does.
    * rel-L2 against the direct sum in class with the single card's at the same
      eps.
    """
    import jaccpot.distributed.template_force as template_force
    import jaccpot.runtime.fmm_prepare as fmm_prepare
    from jaccpot import FusedForceProvider
    from jaccpot.distributed.segment import MeshSegmentRunner
    from jaccpot.distributed.template_force import FusedMeshForce

    mesh = _mesh()
    pos, vel, mass = _ic()
    solver = _solver()
    engine = solver._impl
    force = FusedMeshForce(
        solver,
        mesh,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        cross_record={},
    )
    assert force.scale_carry
    state = _mesh_state(mesh, pos, vel, mass)
    # the library's cross capacities, sized by the runner's start force
    runner = MeshSegmentRunner(mesh, state, force)
    caps = force.cross_caps
    seed = force.force_scale_seed
    acc, f_b, bits, out = _probe(force, runner.state, seed)
    assert not np.any(bits), bits

    # identical tables, identical pair sets, on both devices
    gathered = out["gathered"]
    assert gathered.shape[:2] == (NDEV, NDEV) and gathered.shape[-1] == 8 + ORDER + 2
    _assert_bitwise(gathered[1], gathered[0], "the gathered summary")
    far = [_pair_set(out, d, "far") for d in range(NDEV)]
    near = [_pair_set(out, d, "near") for d in range(NDEV)]
    assert far[0] == far[1] and near[0] == near[1]
    assert len(far[0]) > 50 and len(near[0]) > 50, (len(far[0]), len(near[0]))

    # eq (16a) on the gathered table, independently, in float64
    floor = engine._walk_separation_floor()
    assert floor > 0.0
    kw = dict(order=ORDER, theta_max=float(engine.mac_theta_max), floor=floor)
    ratio, convergence, gap = _eq16a(gathered[0], sorted(far[0]), **kw)
    # float32 decisions near the threshold are allowed a few ulps either way
    assert float(ratio.max()) < 1.0 + 1e-5, float(ratio.max())
    assert float(convergence.max()) < 1.0 and float(gap.min()) >= -1e-6
    ratio_n, convergence_n, gap_n = _eq16a(gathered[0], sorted(near[0]), **kw)
    fails = (ratio_n > 1.0 - 1e-5) | (convergence_n >= 1.0) | (gap_n < 1e-6)
    assert bool(np.all(fails)), int(np.sum(~fails))

    # every imported far row carries its sender node's own mass moment (eq (16b)'s
    # far terms read it): the published table's G M of that node, G = 1 here
    for d in range(NDEV):
        sizes = out["imp_far_sizes"][d]
        live = int(sizes.sum())
        assert live > 0
        peer = np.searchsorted(np.cumsum(sizes), np.arange(live), side="right")
        node = out["imp_far_rows"][d][:live]
        assert np.all(peer != d) and np.all(node >= 0)
        _assert_bitwise(
            out["cross_masses"][d][:live],
            gathered[d][peer, node, 8],
            f"device {d}: the imported far rows' masses",
        )

    # f_b and the force by id
    rows = _live_rows(runner.state.count)
    ids = np.asarray(runner.state.ids)
    fb_mesh = _by_id(f_b, ids, rows)
    acc_mesh = _by_id(acc, ids, rows)
    owner = _by_id(np.repeat(np.arange(NDEV), CAP), ids, rows)
    p64 = pos.astype(np.float64)
    w = mass[None, :] / (np.sum((p64[:, None] - p64[None, :]) ** 2, -1) + SOFT**2)
    np.fill_diagonal(w, 0.0)
    fb_exact = w.sum(axis=1)
    fb_local = np.where(owner[None, :] == owner[:, None], w, 0.0).sum(axis=1)

    single = _solver()
    prov, topology = FusedForceProvider.prepare(
        single,
        jnp.asarray(pos),
        jnp.asarray(mass),
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        caps_only=True,
    )
    single_seed = jnp.asarray(_seed_in_input_order(single, topology, N))
    acc_single, (fb_single, bits_single, _) = jax.jit(prov.step)(
        jnp.asarray(pos), jnp.asarray(mass), None, (single_seed, None, None)
    )
    assert int(bits_single) == 0
    mesh_vs_single = _spread(fb_mesh, fb_single)
    mesh_vs_exact = _spread(fb_mesh, fb_exact)
    single_vs_exact = _spread(fb_single, fb_exact)
    local_vs_exact = _spread(fb_local, fb_exact)
    # measured: 3.7e-3 / 3.9e-2, 4.3e-3 / 3.1e-2, 6.6e-3 / 4.2e-2, 0.21 / 0.61
    assert mesh_vs_single[0] < 1e-2 and mesh_vs_single[1] < 0.1, mesh_vs_single
    assert mesh_vs_exact[0] < 2.0 * single_vs_exact[0] + 1e-3, (
        mesh_vs_exact,
        single_vs_exact,
    )
    assert local_vs_exact[0] > 10.0 * mesh_vs_exact[0], (local_vs_exact, mesh_vs_exact)

    # the force: rel-L2 against the direct sum, the mesh's in class with one card's
    ref = make_reference_direct_force(mesh, G=1.0, softening=SOFT)
    s = runner.state
    want = _by_id(ref(s.state[:, 0], s.masses, s.count, s.ids)[0], ids, rows)
    err_mesh = float(np.linalg.norm(acc_mesh - want) / np.linalg.norm(want))
    err_single = float(
        np.linalg.norm(np.asarray(acc_single) - want) / np.linalg.norm(want)
    )
    # measured: 2.63e-5 (mesh) and 3.30e-5 (one card); theta 0.8 gives 3.6e-3
    assert err_mesh < 1.5 * err_single and err_mesh < 1e-4, (err_mesh, err_single)

    # MUTATION: the cross terms dropped from f_b (the cross near lane's share and
    # the cross far pairs' own terms); the force itself is untouched
    real_near = template_force._add_cross_near_term

    def near_without_scale(engine_, refreshed, acceleration, sink, **kwargs):
        scale = kwargs.get("force_scale")
        if scale is None:
            return real_near(engine_, refreshed, acceleration, sink, **kwargs)
        return real_near(engine_, refreshed, acceleration, sink, **kwargs)[0], scale

    with monkeypatch.context() as mp:
        mp.setattr(template_force, "_add_cross_near_term", near_without_scale)
        mp.setattr(
            fmm_prepare,
            "cross_far_force_scale_own",
            lambda cross_far, masses, **kw: jnp.zeros_like(kw["node_mass"]),
        )
        acc_drop, fb_drop, _, _ = _probe(force, runner.state, seed)
    _assert_bitwise(acc_drop, acc, "the force without the cross f_b terms")
    fb_drop = _by_id(fb_drop, ids, rows)
    drop_vs_exact = _spread(fb_drop, fb_exact)
    assert int(np.sum(fb_drop != fb_mesh)) > N // 2
    assert drop_vs_exact[0] > 10.0 * mesh_vs_exact[0], (drop_vs_exact, mesh_vs_exact)

    # the lanes-kernel route: the cross far terms through FarForceScaleInputs over
    # the local and imported nodes, summed in the CSR M2L launch and pushed down by
    # the Pallas cascade -- the same f_b as the separate pass, from the same seed
    import jaccpot.pallas.m2l_real_csr_lanes as lanes

    calls = {"lane": 0}
    lane_fn = lanes.m2l_real_csr_lanes_fb_pallas_cvjp

    def lane_spy(*args, **kwargs):
        calls["lane"] += 1
        return lane_fn(*args, **kwargs)

    with monkeypatch.context() as mp:
        mp.delenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR")
        mp.setenv("JACCPOT_M2L_CSR_INTERPRET", "1")
        mp.setenv("JACCPOT_CASCADE_PALLAS_INTERPRET", "1")
        mp.setattr(lanes, "m2l_real_csr_lanes_fb_pallas_cvjp", lane_spy)
        lanes_force = FusedMeshForce(
            _solver(),
            mesh,
            leaf_size=LEAF,
            max_order=ORDER,
            theta=THETA,
            cross_caps=caps,
            cross_record={},
        )
        lanes_force.prepare(s.state[:, 0], s.masses, s.count)
        _assert_bitwise(lanes_force.force_scale_seed, seed, "the seed on both routes")
        acc_lanes, fb_lanes, bits_lanes, _ = _probe(lanes_force, s, seed)
    assert calls["lane"] >= 1 and not np.any(bits_lanes)
    np.testing.assert_allclose(
        _by_id(fb_lanes, ids, rows), fb_mesh, rtol=2e-6, atol=0.0
    )
    np.testing.assert_allclose(
        _by_id(acc_lanes, ids, rows), acc_mesh, rtol=0.0, atol=1e-6 * np.abs(want).max()
    )


def test_b_segments_chain_with_the_force_scale_carried(cpu_fused_lane, monkeypatch):
    """GATE (b), the carry: 2 + 2 steps, f_b and the total acceleration carried
    across the segment boundary in :class:`MeshState`, equal one segment of 4,
    bitwise by id (state, acceleration, f_b); flags clean, no retry."""
    from jaccpot.distributed.segment import MeshSegmentRunner

    # The local far pairs grow from ~20 at the start to 132 by step 3. The eager
    # walk measured 176 while the shards' padding rows were leaves of their own
    # (their far pairs, not the particles'); since the eager prepare cuts them the
    # cap is the particles' 128, and the second segment re-plans. Retries are gate
    # (c)'s subject: name the far cap at what the start used to get.
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP", "512")

    mesh = _mesh()
    pos, vel, mass = _ic(seed=3)
    solver = _solver()
    kwargs = dict(leaf_size=LEAF, max_order=ORDER, theta=THETA)
    chained = MeshSegmentRunner.fused(
        solver, mesh, _mesh_state(mesh, pos, vel, mass), **kwargs
    )
    caps = chained.force.cross_caps
    first, second = chained.run(2, dt=DT), chained.run(2, dt=DT)
    assert first.start_force and not second.start_force
    whole = MeshSegmentRunner.fused(
        solver, mesh, _mesh_state(mesh, pos, vel, mass), cross_caps=caps, **kwargs
    )
    report = whole.run(4, dt=DT)
    for rep in (first, second, report):
        assert rep.bits == 0 and rep.retries == 0, rep
    a, b = chained.gather(), whole.gather()
    for name in ("positions", "velocities", "acceleration", "force_scale"):
        _assert_bitwise(a[name], b[name], name)
    assert np.all(a["force_scale"] > 0)


# --------------------------------------------------------------------------
# Gate (c): a retry under the criterion equals a fresh run with the final caps.
# --------------------------------------------------------------------------


def _collapse_into_a_plummer_sphere(steps):
    """``test_mesh_segment_runner.py``'s A3 IC: a uniform cube whose free drift remaps
    every particle radially onto a clipped Plummer sphere over ``steps`` steps,
    masses so small (1e-9) that gravity moves nothing measurably (eq (16a) is
    invariant under a mass scale: G M on its left, f_b on its right)."""
    rng = np.random.default_rng(0)
    start = rng.uniform(-1.0, 1.0, (N, 3))
    radius = np.linalg.norm(start, axis=1)
    rank = (np.argsort(np.argsort(radius)) + 0.5) / N
    radius_end = np.minimum(0.7 / np.sqrt(rank ** (-2.0 / 3.0) - 1.0), 1.5)
    end = start / radius[:, None] * radius_end[:, None]
    vel = (end - start) / (steps * DT)
    return (
        start.astype(np.float32),
        vel.astype(np.float32),
        np.full(N, 1e-9, np.float32),
    )


def test_c_an_outgrown_segment_under_the_criterion_equals_a_fresh_run(
    cpu_fused_lane, fresh_level_registry, monkeypatch
):
    """GATE (c): prepared at leaf capacity 44 on the uniform start, the segment's
    trees and walks outgrow it; the runner re-plans (leaf capacity, walk caps, cross
    capacities), re-prepares every device at the segment's start -- a fresh eq (16b)
    seed there -- and re-runs. The result equals a fresh run started there with the
    final capacities (which needs no retry), bitwise by id: state, acceleration and
    f_b."""
    from jaccpot.distributed.segment import MeshSegmentRunner
    from jaccpot.runtime.capacity_plan import install_walk_caps

    # the library's own walk caps: a NAMED near cap is never widened
    monkeypatch.delenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP")
    steps = 2
    mesh = _mesh()
    pos, vel, mass = _collapse_into_a_plummer_sphere(steps)
    solver = _solver(leaf_capacity=44)
    engine = solver._impl
    kwargs = dict(leaf_size=LEAF, max_order=ORDER, theta=THETA)
    retried = MeshSegmentRunner.fused(
        solver, mesh, _mesh_state(mesh, pos, vel, mass), **kwargs
    )
    report = retried.run(steps, dt=DT)
    assert report.retries >= 1, report
    assert report.bits == 0, report
    assert engine._tree_leaf_capacity > 44

    fresh_solver = _solver(leaf_capacity=engine._tree_leaf_capacity)
    install_walk_caps(fresh_solver, dict(engine._strict_fused_validated_caps))
    fresh = MeshSegmentRunner.fused(
        fresh_solver,
        mesh,
        _mesh_state(mesh, pos, vel, mass),
        cross_caps=retried.force.cross_caps,
        level_floor=retried.force.level_floor,
        **kwargs,
    )
    fresh_report = fresh.run(steps, dt=DT)
    assert fresh_report.retries == 0, fresh_report
    assert fresh.force.setup.plan == retried.force.setup.plan
    a, b = retried.gather(), fresh.gather()
    for name in ("positions", "velocities", "acceleration", "force_scale"):
        _assert_bitwise(a[name], b[name], name)
    assert np.all(a["force_scale"] > 0)
