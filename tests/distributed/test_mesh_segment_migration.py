"""The segment runner's migration to fixed cuts, with a partition-independent force.

``migrate=True`` (:mod:`jaccpot.distributed.migration`) moves, after every drift and
before the force, each particle whose Morton key in the force's box lies outside its
device's splitter range to the owner. It changes ownership only: with
:class:`~jaccpot.distributed.segment.DirectMeshForce` (sources summed in global-id
order) a migrating run and one that never migrates nor repartitions agree BITWISE
by id -- any mis-route of a position, velocity, carried acceleration, mass, id or
``f_b`` breaks that; a mutated arm (the acceleration left in the old rows) proves
the comparison sees it. And every force sees exact Morton domains: at each segment
edge the devices' codes in the box of the current positions are disjoint ranges.

The exchange itself (:func:`~jaccpot.distributed.migration.migrate_shard`) on
crafted rows: what moves where, the padding, and the two declines (the buffer, the
shard), which move nothing and say so.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_mesh_segment_migration.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

import jaccpot.distributed.migration as migration  # noqa: E402
import jaccpot.distributed.rollout as rollout  # noqa: E402
from jaccpot.distributed.segment import (  # noqa: E402
    DirectMeshForce,
    MeshSegmentRunner,
    MeshState,
)

pytestmark = pytest.mark.skipif(
    jax.device_count() < 2, reason="the mesh runner's migration needs >= 2 devices"
)

AXIS = "gpus"
NDEV = 2
N = 256
CAP = int(np.ceil(1.6 * N / NDEV))
DT = 0.02


def _mesh():
    return jax.sharding.Mesh(
        np.asarray(jax.devices()[:NDEV]),
        (AXIS,),
        axis_types=(jax.sharding.AxisType.Auto,),
    )


def _state(mesh, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 1.0, size=(N, 3))
    # fast relative motion: particles cross the cuts every step (a bulk drift
    # alone would not -- the force's box moves with it)
    vel = 0.5 * rng.normal(size=(N, 3)) + np.array([0.6, 0.0, 0.0])
    mass = rng.uniform(0.5, 1.5, size=N) / N
    shard, _ = rollout.decompose(
        mesh, pos, vel, mass, cap=CAP, num_samples=64, axis_name=AXIS
    )
    return MeshState.from_shard_state(shard, mesh)


def _run(migrate, every, segments, *, steps=2, force=None, **kwargs):
    mesh = _mesh()
    state = _state(mesh)
    runner = MeshSegmentRunner(
        mesh,
        state,
        (
            force(state)
            if force
            else DirectMeshForce(G=1e-3, softening=0.05, axis_name=AXIS)
        ),
        repartition_every=every,
        num_samples=64,
        migrate=migrate,
        **kwargs,
    )
    reports = [runner.run(steps, dt=DT) for _ in range(segments)]
    return runner, reports


def _assert_equal_by_id(a, b, names=("positions", "velocities", "acceleration")):
    ga, gb = a.gather(), b.gather()
    for name in (*names, "masses"):
        differ = int(np.sum(ga[name] != gb[name]))
        assert differ == 0, f"{name}: {differ} of {ga[name].size} elements differ"
    return ga, gb


def _exact_domains(runner):
    """Every device's live codes, in the box of the current positions (the box the
    next force builds its trees in), are a range below the next device's."""
    from yggdrax.morton import morton_encode

    from jaccpot.distributed.fused import cube_bounds
    from jaccpot.distributed.segment import _force_box_cubic

    s, cap = runner.state, runner.cap
    count = np.asarray(s.count)
    pos = np.asarray(s.state)[:, 0]
    live = [pos[d * cap : d * cap + int(count[d])] for d in range(runner.ndev)]
    lo = np.min([x.min(0) for x in live], 0)
    hi = np.max([x.max(0) for x in live], 0)
    box = cube_bounds(
        jnp.asarray(lo), jnp.asarray(hi), pad=1e-6, cubic=_force_box_cubic(runner.force)
    )
    codes = [np.asarray(morton_encode(jnp.asarray(x), box)) for x in live]
    splitters = np.asarray(runner.splitters)
    for d in range(runner.ndev):
        if d:
            assert codes[d].min() >= splitters[d - 1], d
        if d + 1 < runner.ndev:
            assert codes[d].max() < splitters[d], d


def test_migrating_changes_nothing_but_ownership():
    """10 segments of 2 steps, migrating and repartitioning before every segment,
    equal a run that neither migrates nor repartitions, bitwise by id; rows moved
    every segment, the ownership differs, and every segment ends on exact Morton
    domains under the cuts it migrated to."""
    a, rep_a = _run(True, 1, 10)
    b, rep_b = _run(False, 0, 10)
    ga, gb = _assert_equal_by_id(a, b)
    assert all(r.migrated > 0 for r in rep_a), [r.migrated for r in rep_a]
    assert all(r.migrated == 0 for r in rep_b)
    assert not np.array_equal(ga["owner"], gb["owner"]), "ownership should differ"
    assert [r.repartitioned for r in rep_a] == [False] + [True] * 9
    for r in rep_a:
        assert r.bits == 0 and r.retries == 0, r
        assert 0 < r.migration_need <= a.migration_cap
        assert sum(r.counts) == N and max(r.counts) <= CAP
    _exact_domains(a)
    # without a repartition the cuts stay and the counts drift; still exact domains
    c, rep_c = _run(True, 0, 6)
    _assert_equal_by_id(c, _run(False, 0, 6)[0])
    assert len({tuple(r.counts) for r in rep_c}) > 1, "the counts never moved"
    _exact_domains(c)


def test_an_acceleration_left_in_the_old_rows_breaks_it(monkeypatch):
    """MUTATION CONTROL: the carried acceleration not moved by the migration."""
    real = migration.migrate_shard

    def unrouted(state, acceleration, *args, **kwargs):
        out = real(state, acceleration, *args, **kwargs)
        return out[:1] + (acceleration,) + out[2:]

    monkeypatch.setattr(migration, "migrate_shard", unrouted)
    a, rep_a = _run(True, 0, 4)
    monkeypatch.setattr(migration, "migrate_shard", real)
    b, _ = _run(False, 0, 4)
    assert sum(r.migrated for r in rep_a) > 0
    assert not np.array_equal(a.gather()["positions"], b.gather()["positions"])


def test_an_undersized_buffer_is_replanned_and_equals_a_fresh_run():
    """A one-row migration buffer: the first migration that moves two rows from a
    device is declined (``OVERFLOW_MIGRATION``), the buffer re-sized from the
    exact need and the segment re-run from its start; the run equals one started
    with the final buffer, and the never-migrating run, bitwise by id."""
    a, rep_a = _run(True, 1, 4, migration_cap=1)
    failed = [f for r in rep_a for f in r.failures]
    assert failed and all(f["bits"] & migration.OVERFLOW_MIGRATION for f in failed)
    assert sum(r.retries for r in rep_a) == len(failed)
    assert a.migration_cap > 1
    assert a.migration_cap >= max(f["migration_need"] for f in failed)
    b, rep_b = _run(True, 1, 4, migration_cap=a.migration_cap)
    assert all(r.retries == 0 for r in rep_b)
    _assert_equal_by_id(a, b)
    _assert_equal_by_id(a, _run(False, 0, 4)[0])


class _CountingScale(DirectMeshForce):
    """The direct force with a stand-in f_b carry (as in
    ``test_mesh_segment_repartition.py``): a particle's f_b after ``k`` forces is
    ``id + k`` wherever it lives, only if every migration moves it along."""

    scale_carry = True

    def __init__(self, ids, **kwargs):
        super().__init__(**kwargs)
        self.force_scale_seed = jnp.where(ids >= 0, ids.astype(jnp.float64), 0.0)

    def shard(self, positions, masses, live, ids, force_scale=None):
        acc, bits, needs = super().shard(positions, masses, live, ids)
        rows = jnp.arange(positions.shape[0]) < live
        return acc, bits, needs, jnp.where(rows, force_scale + 1.0, 0.0)


def test_the_force_scale_rides_the_migration_with_its_particle():
    """S12's f_b carry: after a start force and 10 x 2 migrating steps every
    particle's stand-in f_b is ``id + 21``, and the run equals one that never
    migrates, bitwise by id."""

    def force(state):
        return _CountingScale(state.ids, G=1e-3, softening=0.05, axis_name=AXIS)

    a, rep_a = _run(True, 1, 10, force=force)
    b, _ = _run(False, 0, 10, force=force)
    ga, _ = _assert_equal_by_id(
        a, b, names=("positions", "velocities", "acceleration", "force_scale")
    )
    assert sum(r.migrated for r in rep_a) > 0
    np.testing.assert_array_equal(ga["force_scale"], np.arange(N) + 21.0)


# --------------------------------------------------------------------------
# The exchange on crafted rows.
# --------------------------------------------------------------------------

ROWS = 8


def _crafted(migration_cap, cap, n1):
    """Two devices, rows on a line along x (y = z = 0.5: the Morton order is the x
    order) cut at x = 0.5; device 0 holds five live rows, three of them past the
    cut (rows 1, 3, 4: device 1's), device 1 holds ``n1``, all its own.
    Returns the migrated rows per device, and the input."""
    mesh = _mesh()
    x = np.full((NDEV, cap, 3), 0.5)
    x[0, :5, 0] = [0.1, 0.9, 0.2, 0.7, 0.8]
    x[1, :n1, 0] = np.linspace(0.55, 0.95, n1)
    x[0, 5:, 0] = 0.1
    x[1, n1:, 0] = 0.55
    count = np.array([5, n1])
    state = np.zeros((NDEV, cap, 2, 3))
    state[:, :, 0] = x
    state[:, :, 1] = 1.0 + np.arange(NDEV * cap * 3).reshape(NDEV, cap, 3)
    acc = -1.0 - np.arange(NDEV * cap * 3, dtype=float).reshape(NDEV, cap, 3)
    masses = np.where(np.arange(cap)[None] < count[:, None], 1.0 + np.arange(cap), 0.0)
    ids = np.where(
        np.arange(cap)[None] < count[:, None],
        np.arange(NDEV * cap).reshape(NDEV, cap),
        -1,
    ).astype(np.int32)
    flat = [a.reshape((NDEV * cap,) + a.shape[2:]) for a in (state, acc, masses, ids)]
    spec = P(AXIS)

    def body(state, acc, masses, ids, count):
        # cut between device 0's own rows and device 1's: every row with x > 0.5
        # is device 1's -- the splitter is the lowest code among those
        from yggdrax.morton import morton_encode_impl

        from jaccpot.distributed.fused import global_mesh_bounds

        n = jnp.minimum(count[0], state.shape[0])
        bounds = global_mesh_bounds(state[:, 0], num_valid=n, axis_name=AXIS)
        codes = morton_encode_impl(state[:, 0], bounds)
        theirs = (jnp.arange(state.shape[0]) < n) & (state[:, 0, 0] > 0.5)
        low = jnp.min(jnp.where(theirs, codes, jnp.uint64(2**64 - 1)))
        ours = (jnp.arange(state.shape[0]) < n) & (state[:, 0, 0] < 0.5)
        high = jnp.max(jnp.where(ours, codes, jnp.uint64(0)))
        cut = jax.lax.pmin(low, AXIS)
        ok = jax.lax.pmax(high, AXIS) < cut
        out = migration.migrate_shard(
            state,
            acc,
            masses,
            ids,
            count,
            cut[None],
            migration_cap=migration_cap,
            cubic=False,
            axis_name=AXIS,
        )
        return out[:5] + (out[6]["need"], out[6]["moved"], out[6]["bits"], ok)

    program = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * 5,
            out_specs=(spec,) * 5 + (P(),) * 4,
            check_vma=False,
        )
    )
    out = program(*(jnp.asarray(a) for a in flat), jnp.asarray(count, jnp.int32))
    assert bool(out[-1]), "the crafted rows are not separable by one Morton cut"
    rows = [
        np.asarray(a).reshape((NDEV, cap) + np.asarray(a).shape[1:]) for a in out[:4]
    ]
    return dict(
        state=rows[0],
        acc=rows[1],
        masses=rows[2],
        ids=rows[3],
        count=np.asarray(out[4]),
        need=int(out[5]),
        moved=int(out[6]),
        bits=int(out[7]),
        before=dict(state=state, acc=acc, masses=masses, ids=ids, count=count),
    )


def test_the_exchange_moves_the_strangers_and_keeps_the_order():
    """Device 0's three rows past the cut go to device 1, appended after its own in
    their row order, every array with them; device 0 keeps its two in order; the
    padding sits on each device's first row with zero velocity, acceleration and
    mass and id -1."""
    cap, n1 = ROWS + 3, ROWS
    got = _crafted(migration_cap=4, cap=cap, n1=n1)
    before = got["before"]
    assert got["bits"] == 0 and got["need"] == 3 and got["moved"] == 3
    np.testing.assert_array_equal(got["count"], [2, cap])
    np.testing.assert_array_equal(got["ids"][0, :2], before["ids"][0, [0, 2]])
    want1 = np.concatenate([before["ids"][1, :n1], before["ids"][0, [1, 3, 4]]])
    np.testing.assert_array_equal(got["ids"][1, :cap], want1)
    for name in ("state", "acc", "masses"):
        np.testing.assert_array_equal(got[name][0, :2], before[name][0, [0, 2]])
        np.testing.assert_array_equal(
            got[name][1, :cap],
            np.concatenate([before[name][1, :n1], before[name][0, [1, 3, 4]]]),
        )
    for d, n in enumerate(got["count"]):
        assert np.all(got["ids"][d, n:] == -1)
        assert np.all(got["masses"][d, n:] == 0) and np.all(got["acc"][d, n:] == 0)
        assert np.all(got["state"][d, n:, 1] == 0)
        np.testing.assert_array_equal(
            got["state"][d, n:, 0], np.broadcast_to(got["state"][d, 0, 0], (cap - n, 3))
        )


@pytest.mark.parametrize(
    "migration_cap, n1, bit",
    [
        (2, ROWS - 3, migration.OVERFLOW_MIGRATION),
        (4, ROWS - 2, migration.OVERFLOW_SHARD),
    ],
    ids=["buffer", "shard"],
)
def test_a_migration_that_does_not_fit_is_declined_everywhere(migration_cap, n1, bit):
    """Three rows to move through a two-row buffer, or into a shard with room for
    two: declined on every device -- nothing moved, the need still exact -- and
    the bit set."""
    got = _crafted(migration_cap=migration_cap, cap=ROWS, n1=n1)
    before = got["before"]
    assert got["bits"] == bit and got["need"] == 3 and got["moved"] == 0
    np.testing.assert_array_equal(got["count"], before["count"])
    for name in ("ids", "masses", "acc"):
        for d, n in enumerate(before["count"]):
            np.testing.assert_array_equal(got[name][d, :n], before[name][d, :n])


def test_the_cut_from_the_lowest_codes_reproduces_a_repartition():
    """``splitters_shard`` after a repartition (in the box the force builds its
    trees in) assigns every live row to the device that holds it."""
    mesh = _mesh()
    state = _state(mesh)

    def body(x, count):
        from yggdrax.morton import morton_encode_impl

        from jaccpot.distributed.fused import global_mesh_bounds

        cut = migration.splitters_shard(x, count, cubic=True, axis_name=AXIS)
        n = jnp.minimum(count[0], x.shape[0])
        bounds = global_mesh_bounds(x, num_valid=n, axis_name=AXIS, cubic=True)
        dest = jnp.searchsorted(cut, morton_encode_impl(x, bounds), side="right")
        me = jax.lax.axis_index(AXIS)
        live = jnp.arange(x.shape[0]) < n
        return jax.lax.pmax(jnp.sum(live & (dest != me)), AXIS)

    def repartition(x, v, m, i, c):
        return rollout._repartition_shard(
            x,
            v,
            m,
            i,
            c,
            ndev=NDEV,
            cap=CAP,
            num_samples=64,
            axis_name=AXIS,
            cubic=True,
        )[:5]

    spec = P(AXIS)
    routed = jax.jit(
        jax.shard_map(
            repartition,
            mesh=mesh,
            in_specs=(spec,) * 5,
            out_specs=(spec,) * 5,
            check_vma=False,
        )
    )(state.state[:, 0], state.state[:, 1], state.masses, state.ids, state.count)
    strangers = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(spec, spec), out_specs=P(), check_vma=False
        )
    )(routed[0], routed[4])
    assert int(strangers) == 0


# --------------------------------------------------------------------------
# Through the production runner's multi-GPU mode.
# --------------------------------------------------------------------------


def _production(migrate, every=0, **config):
    from jaccpot.production import MeshRunConfig, ProductionRunner, SegmentConfig
    from tests.distributed import test_production_runner_mesh as prm

    return ProductionRunner(
        prm._solver(),
        segment=SegmentConfig(**{"steps": 2, **config}),
        external_acceleration_fn=prm.harmonic,
        mesh=MeshRunConfig(
            _mesh(),
            cap=prm.CAP,
            repartition_every=every,
            num_samples=prm.SAMPLES,
            force=DirectMeshForce(G=prm.G, softening=prm.SOFT, axis_name=AXIS),
            migrate=migrate,
        ),
    )


def test_the_production_mode_migrates_and_restores_its_cuts(tmp_path):
    """``MeshRunConfig(migrate=True)``: the run equals the non-migrating mode's,
    bitwise by id, and its plan records say what moved. Its restart file holds
    the cuts and the buffer; a run continued from it on the same mesh keeps them
    (no repartition at the restore) and ends on the straight run's layout -- the
    same id in every row -- and cuts, bit for bit. Without repartitions after the
    first (``repartition_every=0``), so the cuts the file holds are the ones every
    later migration uses: a restore that chose its own would end elsewhere."""
    from jaccpot.production import load_restart
    from tests.distributed import test_production_runner_mesh as prm

    state, masses = prm._ic()
    kwargs = dict(dt=prm.DT, num_steps=8, external_params=prm.PARAMS)
    still = _production(False).run(state, masses, **kwargs)
    path = str(tmp_path / "restart_{segment}.npz")
    moving = _production(True, restart_every=1, restart_path=path).run(
        state, masses, **kwargs
    )
    np.testing.assert_array_equal(moving.state, still.state)
    plans = [e.plan["migration"] for e in moving.edges if e.step > 0]
    assert all(p["migrated"] > 0 and p["cap"] >= p["need"] for p in plans), plans
    assert all(e.plan["migration"] is None for e in still.edges)

    _, meta = load_restart(tmp_path / "restart_2.npz")
    saved = meta["mesh"]["migration"]
    assert len(saved["splitters"]) == NDEV - 1 and saved["migration_cap"] >= 1
    resumed = _production(True).run(
        dt=prm.DT,
        num_steps=8,
        restart_from=tmp_path / "restart_2.npz",
        external_params=prm.PARAMS,
    )
    assert resumed.restored["bitwise"], resumed.restored
    np.testing.assert_array_equal(resumed.state, moving.state)
    np.testing.assert_array_equal(
        np.asarray(resumed.handle.state.ids), np.asarray(moving.handle.state.ids)
    )
    np.testing.assert_array_equal(
        np.asarray(resumed.handle.splitters), np.asarray(moving.handle.splitters)
    )
    # a migrating restart continued without migration is told it is not the run
    with pytest.warns(UserWarning, match="not bitwise"):
        other = _production(False).run(
            dt=prm.DT,
            num_steps=8,
            restart_from=tmp_path / "restart_2.npz",
            external_params=prm.PARAMS,
        )
    assert "migration" in other.restored["mismatches"]
