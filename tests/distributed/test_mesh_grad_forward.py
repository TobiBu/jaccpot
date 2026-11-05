"""The multi-GPU differentiable force's forward, and its gradient on one device (WP-G, G1).

``FusedMeshForce.differentiable_shard`` (``jaccpot.distributed.grad``): the topology
rebuilt from the inputs' stopped gradients with a cross hook that records the frozen
cross layout (``grad_sink`` -> ``CrossGradPlan``), S8's differentiable evaluation at
the live inputs with the cross far sources in its far chain (seam A) and the cross
near term (seam B). The seams are G1's placeholders here: the production launches
forward, reverses that raise. Gates, on CPU (the S9 interpret set, the GPU's
target-sorted symmetric walk in interpret mode, the ``buf`` exchange), N = 2048
uniform cube, leaf 64 (capacity 128), p4, fp32:

* MG1: the forward equals ``FusedMeshForce.shard``'s BITWISE on the live rows at 2
  and 4 devices (padding rows zero, bits clean), with non-empty cross far AND near
  pairs on every device; under ``mac_type='dehnen_error'`` at 2 devices the
  acceleration and the carried ``f_b`` too;
* MG2: at one device (no hook, no seam reached) the mesh gradient -- ``jax.grad`` of
  a loss OUTSIDE the ``shard_map`` -- equals the single card's differentiable path in
  the same box, bitwise, in the positions and the masses;
* at two devices ``jax.grad`` reaches the placeholder seams and raises.

Each program is minutes of interpret-mode compilation on a loaded host (``slow``,
``distributed_segment``).

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_mesh_grad_forward.py -q -n 0
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

from jaccpot import FastMultipoleMethod  # noqa: E402
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
    pytest.mark.slow,
    pytest.mark.distributed_segment,
]

AXIS = "gpus"
N = 2048
LEAF, LEAF_CAP, ORDER, THETA, SOFT, EPS = 64, 128, 4, 0.8, 1e-2, 1e-4

#: the S9 differentiable set: every lane kernel interpreted, the prepacked near
#: layout, the CSR M2L lanes kernel, and the GPU's walk (its far list is the
#: target-sorted symmetric CSR the sort-free reverses read)
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


@pytest.fixture(autouse=True)
def _lane_on_cpu(monkeypatch):
    from jaccpot.runtime import _level_shapes

    for key in _ENV_UNSET:
        monkeypatch.delenv(key, raising=False)
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    resolve = ygg_comm.resolve_ragged_method
    monkeypatch.setattr(
        ygg_comm,
        "resolve_ragged_method",
        lambda method="auto": "buf" if method == "auto" else resolve(method),
    )
    monkeypatch.setattr(_level_shapes, "_WIDTHS", {})
    monkeypatch.setattr(_level_shapes, "_LEVELS", {})


def _solver(mac="dehnen"):
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
        working_dtype=jnp.float32,
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


def _mesh(ndev):
    return jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))


def _cube(n, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32),
        (0.1 * rng.normal(size=(n, 3))).astype(np.float32),
        np.full(n, 1.0 / n, np.float32),
    )


def _assert_bitwise(got, want, what):
    got, want = np.asarray(got), np.asarray(want)
    differ = int(np.sum(got != want))
    assert differ == 0, (
        f"{what}: {differ} of {got.size} elements differ, max "
        f"{float(np.nanmax(np.abs(got.astype(np.float64) - want))):.3e}"
    )


def _mesh_force(mesh, mac, shard):
    from jaccpot.distributed.template_force import FusedMeshForce

    force = FusedMeshForce(
        _solver(mac),
        mesh,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        cross_caps=CrossCapacities(max_cells=LEAF_CAP, leaf_width=LEAF),
    )
    force.prepare(shard.positions, shard.masses, shard.count)
    force.with_grad_plan(shard.positions, shard.masses, shard.count)
    return force


def _forward_cases():
    cases = [(2, "dehnen"), (2, "dehnen_error")]
    if jax.device_count() >= 4:
        cases.append((4, "dehnen"))
    return cases


@pytest.mark.parametrize("ndev, mac", _forward_cases())
def test_the_differentiable_forward_is_the_mesh_forces_bitwise(ndev, mac, monkeypatch):
    """MG1: ``differentiable_shard`` == ``shard`` bitwise on the live rows (and
    ``f_b`` under the criterion), padding rows zero, bits clean, and the cross
    far and near lists non-empty on every device."""
    from jaccpot.distributed import grad as G

    mesh = _mesh(ndev)
    cap = int(N / ndev * 1.3)
    pos, vel, mass = _cube(N, 0)
    shard, _ = decompose(mesh, pos, vel, mass, cap=cap, num_samples=64)
    force = _mesh_force(mesh, mac, shard)
    assert force.grad_statics.far_pairs_target_sorted
    assert force.grad_statics.far_pairs_symmetric
    seed = (force.force_scale_seed,) if force.scale_carry else ()
    k = len(seed)

    captured = {}
    real = G.CrossGradPlan.from_sink.__func__

    def spy(cls, sink):
        plan = real(cls, sink)
        captured["plan"] = plan
        return plan

    monkeypatch.setattr(G.CrossGradPlan, "from_sink", classmethod(spy))

    def diff_body(x, m, n, i, *fb):
        out = force.differentiable_shard(x, m, jnp.minimum(n[0], x.shape[0]), i, *fb)
        plan = captured["plan"]
        far = jnp.sum(plan.far_sources >= 0)
        live = jnp.arange(plan.near_targets.shape[0]) < plan.near_count
        near = jnp.sum(live & (plan.near_targets >= 0))
        return (
            out[0],
            out[1].reshape(1),
            out[2].reshape(1, -1),
            *out[3:],
            far.reshape(1),
            near.reshape(1),
        )

    def prod_body(x, m, n, i, *fb):
        out = force.shard(x, m, jnp.minimum(n[0], x.shape[0]), i, *fb)
        return (out[0], out[1].reshape(1), out[2].reshape(1, -1), *out[3:])

    spec = P(AXIS)
    args = (shard.positions, shard.masses, shard.count, shard.ids, *seed)
    with force.context():
        force.before_trace()
        got = jax.jit(
            jax.shard_map(
                diff_body,
                mesh=mesh,
                in_specs=(spec,) * (4 + k),
                out_specs=(spec,) * (5 + k),
                check_vma=False,
            )
        )(*args)
        got = [np.asarray(o) for o in got]
        force.before_trace()
        want = jax.jit(
            jax.shard_map(
                prod_body,
                mesh=mesh,
                in_specs=(spec,) * (4 + k),
                out_specs=(spec,) * (3 + k),
                check_vma=False,
            )
        )(*args)
        want = [np.asarray(o) for o in want]

    count = np.asarray(shard.count)
    rows = np.concatenate([np.arange(d * cap, d * cap + count[d]) for d in range(ndev)])
    pad = np.setdiff1d(np.arange(ndev * cap), rows)
    far, near = got[-2], got[-1]
    print(f"ndev={ndev} {mac}: cross far pairs {far}, near pairs {near}")
    assert np.all(far > 0) and np.all(near > 0), (far, near)
    assert int(np.bitwise_or.reduce(want[1])) == 0, want[1]
    assert int(np.bitwise_or.reduce(got[1])) == 0, got[1]
    _assert_bitwise(got[0][rows], want[0][rows], "differentiable vs mesh force")
    assert np.all(got[0][pad] == 0.0)
    _assert_bitwise(got[2], want[2], "needs")
    if k:
        _assert_bitwise(got[3][rows], want[3][rows], "f_b")
        assert np.all(got[3][pad] == 0.0)


def _one_device(seed):
    mesh = _mesh(1)
    n = N // 2
    pos, _, mass = _cube(n, seed)
    sharding = NamedSharding(mesh, P(AXIS))
    put = lambda a: jax.device_put(a, sharding)  # noqa: E731
    return mesh, n, pos, mass, put(pos), put(mass), put(np.asarray([n], np.int32))


def test_one_device_mesh_gradient_is_the_single_cards():
    """MG2: one device (no hook: the placeholder seams are never reached), the loss
    outside the ``shard_map``: the mesh's forward and its gradient in x and m equal
    the single card's differentiable path (rebuild on stopped inputs in the same
    box, the eager plan's statics, ``differentiable_accelerations``), bitwise."""
    from jaccpot import FusedForceProvider
    from jaccpot.distributed.fused import host_mesh_bounds

    mesh, n, pos, mass, x, m, count = _one_device(1)
    force = _mesh_force(mesh, "dehnen", _OneShard(x, m, count))
    assert force.hook is None
    w = jnp.asarray(np.random.default_rng(7).normal(size=(n, 3)).astype(np.float32))
    spec = P(AXIS)

    def body(xx, mm, nn):
        acc, bits, _ = force.differentiable_shard(
            xx, mm, jnp.minimum(nn[0], xx.shape[0]), nn
        )
        return acc, bits.reshape(1)

    program = jax.shard_map(
        body, mesh=mesh, in_specs=(spec,) * 3, out_specs=(spec,) * 2, check_vma=False
    )

    def loss(xx, mm):
        acc, _ = program(xx, mm, count)
        return jnp.sum(acc * w)

    with force.context():
        force.before_trace()
        acc_mesh, bits = (np.asarray(o) for o in jax.jit(program)(x, m, count))
        force.before_trace()
        gx, gm = (np.asarray(g) for g in jax.jit(jax.grad(loss, argnums=(0, 1)))(x, m))
    assert int(bits[0]) == 0, bits

    box = host_mesh_bounds([pos], [n], cubic=bool(force.engine._tree_cubic_bounds))
    single, _ = FusedForceProvider.prepare(
        _solver(),
        jnp.asarray(pos),
        jnp.asarray(mass),
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        bounds=box,
        differentiable=True,
    )
    assert single.grad_statics == force.grad_statics, (
        single.grad_statics,
        force.grad_statics,
    )

    def forward(xx, mm):
        stop = jax.lax.stop_gradient
        topo = single.rebuild(stop(xx), stop(mm), bounds=box, keep_far_list=True)
        plan, _ = single.grad_plan(topo)
        return single.differentiable_accelerations(topo, xx, mm, plan=plan)

    acc_one = np.asarray(jax.jit(forward)(jnp.asarray(pos), jnp.asarray(mass)))
    sgx, sgm = (
        np.asarray(g)
        for g in jax.jit(
            jax.grad(lambda a, b: jnp.sum(forward(a, b) * w), argnums=(0, 1))
        )(jnp.asarray(pos), jnp.asarray(mass))
    )
    assert np.all(np.isfinite(gx)) and np.all(np.isfinite(gm))
    assert float(np.max(np.abs(gm))) > 0.0
    _assert_bitwise(acc_mesh, acc_one, "one-device forward vs the single card's")
    _assert_bitwise(gx, sgx, "d/dx vs the single card's")
    _assert_bitwise(gm, sgm, "d/dm vs the single card's")


class _OneShard:
    """``decompose``'s fields :func:`_mesh_force` reads, for one device."""

    def __init__(self, x, m, count):
        self.positions, self.masses, self.count = x, m, count


def test_the_gradient_reaches_the_placeholder_seams_and_raises():
    """Two devices: the reverse needs seams A and B, which G1 leaves unwired --
    ``jax.grad`` raises instead of returning a gradient without the cross terms."""
    mesh = _mesh(2)
    cap = int(N / 2 * 1.3)
    pos, vel, mass = _cube(N, 0)
    shard, _ = decompose(mesh, pos, vel, mass, cap=cap, num_samples=64)
    force = _mesh_force(mesh, "dehnen", shard)
    spec = P(AXIS)

    def loss(x, m):
        def body(xx, mm, nn, ii):
            return force.differentiable_shard(
                xx, mm, jnp.minimum(nn[0], xx.shape[0]), ii
            )[0]

        acc = jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 4, out_specs=spec, check_vma=False
        )(x, m, shard.count, shard.ids)
        return jnp.sum(acc)

    with force.context():
        force.before_trace()
        with pytest.raises(NotImplementedError, match="cross seam [AB] not wired yet"):
            jax.eval_shape(jax.grad(loss), shard.positions, shard.masses)
