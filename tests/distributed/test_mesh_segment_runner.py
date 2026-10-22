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
  bitwise by id (the repartition half of A2, with a partition-independent force:
  ``test_mesh_segment_repartition.py``).
* A3: a segment that outgrows its leaf capacity, walk caps and cross summary is
  re-planned at its start, and equals a fresh run with the final capacities,
  bitwise by id.

Each case is ~1.5-4 min on a loaded 64-core host, nearly all compilation, so the
file has a CI job of its own (marker ``distributed_segment``).

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
    FarFieldConfig,
    FMMAdvancedConfig,
    NearFieldConfig,
    TreeConfig,
)
from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS, CrossCapacities  # noqa: E402
from jaccpot.distributed.rollout import (  # noqa: E402
    decompose,
    make_reference_direct_force,
    setup_fused_force,
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
LEAF_CAP = 64
ORDER = 4
THETA = 0.8
SOFT = 1e-2
CAP = int(N / NDEV * 1.3)
#: the flat M2L (the CPU lane) runs ONE fullbatch segment sum when the merged pair
#: list -- the local far list plus the cross far list's CAPACITY -- fits a chunk,
#: and a chunked scan otherwise; the two round 1 ulp apart (fe3618fd). 512-pair
#: chunks keep every capacity here in the chunked regime, so capacities that differ
#: in width (the library's against explicit ones) compute the same sums.
M2L_CHUNK = 512


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
            farfield=FarFieldConfig(m2l_chunk_size=M2L_CHUNK),
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


def test_a1_a2_the_template_force_and_chained_segments_bitwise(cpu_fused_lane):
    """GATE A1, and A2 (2), on one solver (their compiled programs shared).

    A1: with the same caps, the runner's start force (the template route:
    caps-only prepares, ONE shape template materialised in the body) equals the
    stacked route's force (``setup_fused_force``: full eager prepares, stacked
    states) bitwise on every live row, all finite -- no template value reaches
    the force -- and it is the force (direct sum).

    Library cross capacities: a runner on the default ``cross_caps="auto"`` (sized
    from the start force's measured needs: other widths) computes the same start
    force bitwise.

    A2 (2): 2 + 2 steps, the total acceleration carried across, equal one segment
    of 4, bitwise by id, with the cross field; the second segment starts from the
    carried acceleration, not a start force."""
    from jaccpot.distributed.segment import MeshSegmentRunner

    mesh = _mesh()
    pos, vel, mass = _ic()
    shard, _ = decompose(mesh, pos, vel, mass, cap=CAP, num_samples=64)
    solver = _solver()
    stacked, plan_s, walk_s = setup_fused_force(
        solver,
        mesh,
        shard,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        cross_caps=_cross_caps(),
    )
    acc_s, flag_s = stacked(shard.positions, shard.masses, shard.count, shard.ids)
    assert not bool(np.asarray(flag_s))

    # the template route on the same solver: its prepares start from the walk caps
    # the stacked setup validated, so the two compile the same caps
    kwargs = dict(
        leaf_size=LEAF, max_order=ORDER, theta=THETA, cross_caps=_cross_caps()
    )
    chained = MeshSegmentRunner.fused(
        solver, mesh, _mesh_state(mesh, pos, vel, mass), **kwargs
    )
    assert chained.force.setup.plan == plan_s
    assert chained.force.setup.walk_caps == walk_s
    start = chained.start()
    assert start.start_force and start.retries == 0
    rows = _live_rows(chained.state.count)
    np.testing.assert_array_equal(
        np.asarray(chained.state.ids)[rows], np.asarray(shard.ids)[rows]
    )
    got = np.asarray(chained.state.acceleration)[rows]
    assert np.all(np.isfinite(got)), "the template route read a template value"
    _assert_bitwise(got, np.asarray(acc_s)[rows], "template vs stacked force")
    # and it is the force: the single-device fused lane's own error on these
    # particles is 3.578e-3 (test_fused_multigpu_cpu_e2e.py)
    ref = make_reference_direct_force(mesh, G=1.0, softening=SOFT)
    want = np.asarray(ref(shard.positions, shard.masses, shard.count, shard.ids)[0])
    want = want[rows].astype(np.float64)
    err = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    assert err < 5e-3, err

    # the library's cross capacities (the default): sized from the start force's
    # needs, other widths than the explicit ones, the same force bitwise
    library = MeshSegmentRunner.fused(
        solver,
        mesh,
        _mesh_state(mesh, pos, vel, mass),
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
    )
    lib_caps = library.force.cross_caps
    assert library.force.cross_needs_measured is not None
    widths = [
        k for k in vars(lib_caps) if getattr(lib_caps, k) != getattr(_cross_caps(), k)
    ]
    assert "export_far_cap" in widths and "recv_particle_cap" in widths, widths
    library.start()
    got_lib = np.asarray(library.state.acceleration)[rows]
    _assert_bitwise(got_lib, np.asarray(acc_s)[rows], "library vs explicit cross caps")
    del library

    first, second = chained.run(2, dt=DT), chained.run(2, dt=DT)
    assert not first.start_force and not second.start_force  # start() ran it
    whole = MeshSegmentRunner.fused(
        solver, mesh, _mesh_state(mesh, pos, vel, mass), **kwargs
    )
    assert whole.run(4, dt=DT).start_force
    _gathered_equal(chained, whole)


# --------------------------------------------------------------------------
# Gate A3: a segment that outgrows its capacities is re-planned at its start, and
# the re-run equals a fresh run with the final capacities, bitwise.
# --------------------------------------------------------------------------


def _collapse_into_a_plummer_sphere(steps):
    """``(positions, velocities, masses)``: a uniform cube whose free drift remaps
    every particle radially -- direction kept, radius rank kept -- onto a clipped
    Plummer sphere (scale 0.7) over ``steps`` steps of ``DT``, with masses so small
    (1e-9) that gravity moves nothing measurably. Keeping the directions keeps each
    device's Morton domain a half-space, so the cross volume stays a boundary's.
    The start fits what it is prepared with; the steps need more cell leaves (per
    device ~33 -> 64 at leaf 64) and longer walk lists (far 1230 and near 1294
    directed pairs against the start's validated 1024 each)."""
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


def test_a3_an_outgrown_segment_is_replanned_and_equals_a_fresh_run(
    cpu_fused_lane, fresh_level_registry, monkeypatch
):
    """GATE A3, on the library's capacities throughout: prepared at leaf capacity
    44 on the uniform start (it fits; the cross capacities sized there from the
    start force's needs), the segment's trees need more leaves than that and its
    walks longer lists: the flags fire, the needs are reduced over the mesh, and the
    segment is re-planned at its start -- the leaf capacity to ``auto_leaf_capacity``
    of what the trees needed, the walk caps to what the walks needed, the cross
    capacities by ``CrossCapacities.replanned`` with the summary following the grown
    leaf capacity -- every device re-prepared caps-only there, and the segment
    re-run. The result equals a fresh run started there with the final capacities
    (leaf capacity, walk caps, level plan, cross caps), bitwise by id, and that run
    needs no retry."""
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
    sized = retried.force.cross_caps
    # the summary is clamped to what a tree of 44 leaves can hold
    assert sized.max_cells <= 44 and sized.max_summary_nodes <= 2 * 44 - 1, vars(sized)
    caps_before = dict(engine._strict_fused_validated_caps)
    report = retried.run(steps, dt=DT)
    caps_after = dict(engine._strict_fused_validated_caps)
    assert report.retries >= 1, report
    assert engine._tree_leaf_capacity == 1024  # auto_leaf_capacity(64 leaves)
    for key in ("compact_far_pair_capacity", "near_edge_capacity"):
        assert int(caps_after[key]) > int(caps_before[key]), key
    final = retried.force.cross_caps
    # the summary followed the leaf capacity (64 live leaves x 1.5, in 1024s)
    assert 64 <= final.max_cells <= 1024, vars(final)
    assert final.max_summary_nodes <= 2 * 1024 - 1, vars(final)
    assert (
        report.needs[-len(CROSS_NEEDS_FIELDS) :][CROSS_NEEDS_FIELDS.index("overflow")]
        == 0
    ), report.needs

    fresh_solver = _solver(leaf_capacity=engine._tree_leaf_capacity)
    install_walk_caps(fresh_solver, caps_after)
    fresh = MeshSegmentRunner.fused(
        fresh_solver,
        mesh,
        _mesh_state(mesh, pos, vel, mass),
        cross_caps=final,
        level_floor=retried.force.level_floor,
        **kwargs,
    )
    fresh_report = fresh.run(steps, dt=DT)
    assert fresh_report.retries == 0, fresh_report
    assert fresh.force.setup.plan == retried.force.setup.plan
    _gathered_equal(retried, fresh)


def test_an_undersized_cross_capacity_is_replanned_and_equals_a_fresh_run(
    cpu_fused_lane,
):
    """A cross capacity too small for the start (two buffers: the near rows a
    device receives and the particles it sends) fires the cross flag on the start
    force; the runner re-plans the cross capacities from the measured needs
    (``CrossCapacities.replanned``, exact past their overflow: one retry) and runs
    the segment. It equals a run started with the re-planned capacities, bitwise
    by id, which needs no retry -- on the same solver, so the two share their
    compiled programs and what is compared is the retry: the start state it
    restarted from and the start force it recomputed."""
    from jaccpot.distributed.segment import MeshSegmentRunner

    mesh = _mesh()
    pos, vel, mass = _ic()
    solver = _solver()
    small = _cross_caps()._with(recv_node_cap=64, send_particle_cap=512)
    kwargs = dict(leaf_size=LEAF, max_order=ORDER, theta=THETA)
    retried = MeshSegmentRunner.fused(
        solver, mesh, _mesh_state(mesh, pos, vel, mass), cross_caps=small, **kwargs
    )
    report = retried.run(2, dt=DT)
    final = retried.force.cross_caps
    assert report.retries == 1, report
    assert final.recv_node_cap > 64 and final.send_particle_cap > 512, vars(final)
    # nothing else moved: the re-plan grew what overflowed, monotone elsewhere
    for name, value in vars(small).items():
        if value is not None and name not in ("recv_node_cap", "send_particle_cap"):
            assert getattr(final, name) >= value, name

    fresh = MeshSegmentRunner.fused(
        solver, mesh, _mesh_state(mesh, pos, vel, mass), cross_caps=final, **kwargs
    )
    assert fresh.run(2, dt=DT).retries == 0
    _gathered_equal(retried, fresh)
