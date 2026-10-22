"""The fused multi-GPU lane's segment runner on two forced CPU devices (S11, gates A1-A3).

The CPU end-to-end pattern of ``test_fused_multigpu_cpu_e2e.py`` (the large-N profile
reached on CPU with ``jax.default_backend`` reporting ``"gpu"``, the all-gather
``"buf"`` exchange for XLA:CPU's missing ``ragged_all_to_all``, the Pallas kernels
interpreted), at its toy size: N = 2048 uniform cube, leaf 64, p4, theta 0.8,
eps 1e-2 (ferrers3), two devices.

* A1: the template route's force (caps-only prepares, ONE shape template) equals the
  stacked route's (``setup_fused_force``: full eager prepares, stacked states)
  BITWISE with the same caps.
* A2: a one-device mesh segment equals a user-written single-card scan over the
  provider in the same box, bitwise; two segments of 2 steps equal one of 4,
  bitwise by id.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_mesh_segment_runner.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yggdrax.distributed.comm as ygg_comm

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from jaccpot import FastMultipoleMethod  # noqa: E402
from jaccpot.config import (  # noqa: E402
    FMMAdvancedConfig,
    NearFieldConfig,
    TreeConfig,
)
from jaccpot.distributed.cross import CrossCapacities  # noqa: E402
from jaccpot.distributed.rollout import (  # noqa: E402
    decompose,
    make_reference_direct_force,
    setup_fused_force,
)
from jaccpot.distributed.template_force import setup_template_force  # noqa: E402

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the fused multi-GPU lane needs >= 2 devices"
    ),
    pytest.mark.slow,
]

AXIS = "gpus"
NDEV = 2
N = 2048
LEAF = 64
LEAF_CAP = 64
ORDER = 4
THETA = 0.8
SOFT = 1e-2
CAP = int(N / NDEV * 1.3)


@pytest.fixture(scope="module")
def cpu_fused_lane():
    """The CPU switches of the end-to-end test, for the whole module."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax, "default_backend", lambda: "gpu")
        resolve = ygg_comm.resolve_ragged_method
        mp.setattr(
            ygg_comm,
            "resolve_ragged_method",
            lambda method="auto": "buf" if method == "auto" else resolve(method),
        )
        mp.setenv("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1")
        mp.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "0")
        mp.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")
        mp.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")
        yield


def _solver(leaf_capacity=LEAF_CAP):
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFT,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=leaf_capacity,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
    )


def _mesh(ndev=NDEV):
    return jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))


def _ic(seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, (N, 3)).astype(np.float32)
    vel = (0.1 * rng.normal(size=(N, 3))).astype(np.float32)
    mass = np.full(N, 1.0 / N, np.float32)
    return pos, vel, mass


def _cross_caps():
    return CrossCapacities(max_cells=LEAF_CAP, leaf_width=LEAF)


def _live_rows(count, cap=CAP):
    count = np.asarray(count)
    return np.concatenate(
        [np.arange(d * cap, d * cap + int(count[d])) for d in range(count.shape[0])]
    )


def _assert_bitwise(got, want, what):
    got, want = np.asarray(got), np.asarray(want)
    differ = int(np.sum(got != want))
    assert differ == 0, (
        f"{what}: {differ} of {got.size} elements differ, max "
        f"{float(np.nanmax(np.abs(got.astype(np.float64) - want))):.3e}"
    )


def test_a1_the_template_force_is_the_stacked_force_bitwise(cpu_fused_lane):
    """GATE A1: same caps, same force, bit for bit; nothing reads the template."""
    mesh = _mesh()
    pos, vel, mass = _ic()
    state, _ = decompose(mesh, pos, vel, mass, cap=CAP, num_samples=64)
    solver = _solver()
    stacked, plan_s, walk_s = setup_fused_force(
        solver,
        mesh,
        state,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        cross_caps=_cross_caps(),
    )
    acc_s, flag_s = stacked(state.positions, state.masses, state.count, state.ids)
    assert not bool(np.asarray(flag_s))
    # the template route on the same solver: its prepares start from the walk caps
    # the stacked setup validated, so the two compile the same caps
    template, plan_t, walk_t = setup_template_force(
        solver,
        mesh,
        state,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        cross_caps=_cross_caps(),
    )
    assert plan_t == plan_s
    assert walk_t == walk_s
    acc_t, flag_t = template(state.positions, state.masses, state.count, state.ids)
    assert not bool(np.asarray(flag_t))
    rows = _live_rows(state.count)
    got = np.asarray(acc_t)[rows]
    assert np.all(np.isfinite(got)), "the template route read a template value"
    _assert_bitwise(got, np.asarray(acc_s)[rows], "template vs stacked force")
    # and it is the force: the single-device fused lane's own error on these
    # particles is 3.578e-3 (test_fused_multigpu_cpu_e2e.py)
    ref = make_reference_direct_force(mesh, G=1.0, softening=SOFT)
    want = np.asarray(ref(state.positions, state.masses, state.count, state.ids)[0])
    want = want[rows].astype(np.float64)
    err = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    assert err < 5e-3, err


# --------------------------------------------------------------------------
# Gate A2: the segment program is the single card's particle carry, and segments
# chain bitwise.
# --------------------------------------------------------------------------

DT = 1e-3


def _mesh_state(mesh, pos, vel, mass, cap=CAP, num_samples=64):
    from jaccpot.distributed.segment import MeshState

    shard, _ = decompose(mesh, pos, vel, mass, cap=cap, num_samples=num_samples)
    return MeshState.from_shard_state(shard, mesh)


def _gathered_equal(a, b, names=("positions", "velocities", "acceleration")):
    ga, gb = a.gather(), b.gather()
    for name in names:
        _assert_bitwise(ga[name], gb[name], name)
    return ga


@pytest.fixture
def fresh_level_registry(monkeypatch):
    """An empty level-shape registry: process-global, never lowered, so a width
    another test registered for the same tree shape would size this one's loops."""
    from jaccpot.runtime import _level_shapes

    monkeypatch.setattr(_level_shapes, "_WIDTHS", {})
    monkeypatch.setattr(_level_shapes, "_LEVELS", {})


def test_a2_one_device_segment_is_the_single_card_particle_carry(
    cpu_fused_lane, fresh_level_registry
):
    """GATE A2 (1): a one-device mesh segment equals a user-written single-card
    scan over the provider, bitwise, when that scan builds its trees in the same
    box (the live particles' min/max, padded as ``global_mesh_bounds`` pads it).

    One device, no padding rows (cap = N), no cross hook: what differs is the
    mesh plumbing alone -- the ``shard_map``, the box's all-reduce, the template
    of the merged caps-only prepare, ``num_valid``, the installed plan."""
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
    runner.run(steps, dt=DT)
    got_state = np.asarray(runner.state.state)
    got_acc = np.asarray(runner.state.acceleration)

    # the single card: its own solver and caps-only prepare, in the same box
    single = _solver()
    box0 = host_mesh_bounds([pos], [n_one])
    prov, _ = FusedForceProvider.prepare(
        single,
        jnp.asarray(pos),
        jnp.asarray(mass),
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        bounds=box0,
        caps_only=True,
    )

    def force(x, m):
        box = cube_bounds(jnp.min(x, axis=0), jnp.max(x, axis=0), pad=1e-6)
        topology = prov.rebuild(x, m, bounds=box)
        return prov.accelerations(topology), prov.capacity_ok(topology)

    @jax.jit
    def start(s, m):
        acc, ok = force(s[:, 0], m)
        return jnp.asarray(acc, s.dtype), ok

    @jax.jit
    def segment(s, a, m, dt):
        def step(carry, _):
            s_now, a_now, ok = carry
            x = s_now[:, 0] + s_now[:, 1] * dt + 0.5 * a_now * dt**2
            s_now = s_now.at[:, 0].set(x)
            a_new, ok_new = force(x, m)
            a_new = jnp.asarray(a_new, s_now.dtype)
            v = s_now[:, 1] + 0.5 * (a_now + a_new) * dt
            return (s_now.at[:, 1].set(v), a_new, ok & ok_new), None

        return jax.lax.scan(step, (s, a, jnp.asarray(True)), None, length=steps)[0]

    s0, m0 = jnp.asarray(packed), jnp.asarray(mass)
    a0, ok0 = start(s0, m0)
    want_state, want_acc, ok = segment(s0, a0, m0, jnp.asarray(DT, s0.dtype))
    assert bool(ok0) and bool(ok)
    assert not np.array_equal(np.asarray(want_state), packed), "nothing moved"
    _assert_bitwise(got_state, want_state, "state")
    _assert_bitwise(got_acc, want_acc, "acceleration")


def test_a2_two_segments_are_one_segment_bitwise(cpu_fused_lane):
    """GATE A2 (2): 2 + 2 steps (the total acceleration carried across) equal one
    segment of 4, bitwise by id, on two devices with the cross field -- and the
    second segment starts from the carried acceleration, not a start force."""
    from jaccpot.distributed.segment import MeshSegmentRunner

    mesh = _mesh()
    pos, vel, mass = _ic()
    solver = _solver()
    kwargs = dict(
        leaf_size=LEAF, max_order=ORDER, theta=THETA, cross_caps=_cross_caps()
    )
    chained = MeshSegmentRunner.fused(
        solver, mesh, _mesh_state(mesh, pos, vel, mass), **kwargs
    )
    first, second = chained.run(2, dt=DT), chained.run(2, dt=DT)
    assert first.start_force and not second.start_force
    whole = MeshSegmentRunner.fused(
        solver, mesh, _mesh_state(mesh, pos, vel, mass), **kwargs
    )
    whole.run(4, dt=DT)
    _gathered_equal(chained, whole)
