"""The escaper path in ``ProductionRunner``'s multi-GPU mode (wave 3, MESC), forced CPU devices.

ESC's synthetic system (``tests/integration/strict/test_escapers.py``: a Plummer
sphere, N = 1024 clipped at r = 3, eight fliers leaving ``r_escape = 5`` at 6-12
times the escape speed and one particle outside it falling back inside
``r_return = 4.5``), 6 segments of 10 steps at dt 1e-3, the energy at every edge,
on a 2-device mesh:

* with :class:`~jaccpot.distributed.segment.DirectMeshForce` (seconds of compile),
  migrating and not: the escaper counts per edge are the single card's; the
  ledgered energy stays within the no-split control's noise and the momentum is
  conserved; the split changes at several edges and compiles ONE escaper segment
  program (the tail's counts are traced); a restart from the run's mid-run file and
  a chained run equal the run in one go, bitwise; the result and the edges'
  gathers are in id order; escapers on but nobody escaping IS the run without them
  (the default programs), bitwise; a tail that outgrows its capacity is grown;
* with the fused mesh force under both MACs: the same counts, ledger and restart.

Marker ``distributed_segment`` (job ``test-distributed-segment-escapers``).

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        JAX_ENABLE_X64=1 pytest tests/distributed/test_production_runner_mesh_escapers.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from jaccpot import FastMultipoleMethod  # noqa: E402
from jaccpot.distributed import escapers as mesh_escapers  # noqa: E402
from jaccpot.distributed.segment import DirectMeshForce  # noqa: E402
from jaccpot.escapers import EscaperPolicy  # noqa: E402
from jaccpot.production import (  # noqa: E402
    MeshRunConfig,
    ProductionRunner,
    SegmentConfig,
)

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the escaper mesh tier needs >= 2 devices"
    ),
    pytest.mark.slow,
    pytest.mark.distributed_segment,
]

AXIS = "gpus"
NDEV = 2
N, K = 1024, 8
R_ESCAPE, R_RETURN = 5.0, 4.5
DT, STEPS, SEGMENTS = 1e-3, 10, 6
G, SOFT = 1.0, 1e-2
#: the single card's escapers per edge on this system (ESC's gate 1, both MACs)
COUNTS = [1, 1, 2, 4, 9, 9, 8]
#: escaper fields a run without escapers does not report
ESCAPER_FIELDS = (
    "n_escapers",
    "escaper_transfers",
    "escaper_jump",
    "escaper_ledger",
    "potential_couple",
    "kinetic_escapers",
    "energy_ledgered",
    "energy_ledgered_drift",
)


def _system(seed: int = 0, *, fliers: bool = True) -> tuple:
    """ESC's system (``test_escapers._system``) as host float32 arrays."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, N)
    r = np.minimum(1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0), 3.0)
    d = rng.normal(size=(N, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pos = r[:, None] * d
    pos -= pos.mean(axis=0)
    vel = 0.3 * rng.normal(size=(N, 3))
    if fliers:
        de = rng.normal(size=(K + 1, 3))
        de /= np.linalg.norm(de, axis=1, keepdims=True)
        re = rng.uniform(4.8, 4.9, K)
        ve = np.linspace(6.0, 12.0, K) * np.sqrt(2.0 / re)
        pos = np.concatenate([pos, re[:, None] * de[:K], 5.3 * de[K:]])
        vel = np.concatenate([vel, ve[:, None] * de[:K], -15.0 * de[K:]])
    masses = np.full(pos.shape[0], 1.0 / pos.shape[0])
    vel -= np.sum(masses[:, None] * vel, axis=0) / masses.sum()
    return (
        np.stack([pos, vel], axis=1).astype(np.float32),
        masses.astype(np.float32),
    )


def _mesh():
    return jax.sharding.Mesh(np.asarray(jax.devices()[:NDEV]), (AXIS,))


def _direct_runner(
    escapers=EscaperPolicy(R_ESCAPE, R_RETURN),
    *,
    migrate=False,
    force=None,
    migration_cap="auto",
    **seg,
):
    config = dict(steps=STEPS, diagnostics_every_energy=1)
    config.update(seg)
    return ProductionRunner(
        FastMultipoleMethod(preset="large_n_gpu", G=G, softening=SOFT),
        segment=SegmentConfig(**config),
        escapers=escapers,
        mesh=MeshRunConfig(
            _mesh(),
            cap=int(np.ceil(1.5 * (N + K + 1) / NDEV)),
            num_samples=64,
            migrate=migrate,
            migration_cap=migration_cap,
            force=force or DirectMeshForce(G=G, softening=SOFT, axis_name=AXIS),
        ),
    )


def _escaper_programs(force) -> dict:
    """The force's cached mesh programs that carry the tail, by kind."""
    cache = getattr(force.cache_owner, "_mesh_segment_programs", {})
    kinds: dict = {}
    for key in cache:
        if "escapers" in key:
            kinds[key[1]] = kinds.get(key[1], 0) + 1
    return kinds


@pytest.fixture(scope="module", params=[False, True], ids=["fixed", "migrate"])
def direct_run(request, tmp_path_factory):
    migrate = request.param
    state, masses = _system()
    path = str(
        tmp_path_factory.mktemp(f"mesc_{int(migrate)}") / "restart_{segment}.npz"
    )
    force = DirectMeshForce(G=G, softening=SOFT, axis_name=AXIS)
    runner = _direct_runner(
        migrate=migrate,
        force=force,
        restart_every=SEGMENTS // 2,
        restart_path=path,
    )
    seen: list = []
    runner.callbacks = (lambda edge: seen.append((edge.step, edge.state_host())),)
    res = runner.run(state, masses, dt=DT, num_steps=STEPS * SEGMENTS)
    control = _direct_runner(escapers=None, migrate=migrate).run(
        state, masses, dt=DT, num_steps=STEPS * SEGMENTS
    )
    return dict(
        migrate=migrate,
        state=state,
        masses=masses,
        force=force,
        res=res,
        control=control,
        seen=seen,
        path=path,
    )


def test_the_counts_are_the_single_cards_and_one_program_runs_every_split(
    direct_run,
):
    res = direct_run["res"]
    rows = [e.diagnostics for e in res.edges]
    assert [r["n_escapers"] for r in rows] == COUNTS
    moved = [r["escaper_transfers"] for r in rows]
    assert sum(1 for m in moved[1:] if m) >= 3, moved
    assert res.escapers["n_bound"] == N + 1
    assert res.escapers["transfers"] == 1 + sum(moved[1:])
    # the tail's counts are traced: one start and one segment program for every
    # split (the default programs for the bound rows are not built at all)
    assert _escaper_programs(direct_run["force"]) == {"start": 1, "segment": 1}
    # the plan record says where the escapers are
    counts = res.edges[-1].plan["escapers"]["counts"]
    assert sum(counts) == K


def test_the_ledger_conserves_the_energy_and_the_momentum(direct_run):
    res, control = direct_run["res"], direct_run["control"]
    rows = [e.diagnostics for e in res.edges]
    e0 = res.baselines["energy"]
    jumps = [r["escaper_jump"] for r in rows if r["escaper_jump"] is not None]
    assert jumps and all(abs(j) / abs(e0) < 1e-4 for j in jumps)
    worst = max(abs(r["energy_ledgered_drift"]) for r in rows)
    raw = max(abs(r["energy_drift"]) for r in rows)
    noise = max(abs(e.diagnostics["energy_drift"]) for e in control.edges)
    assert worst < max(5.0 * noise, 3e-6), (worst, noise)
    assert raw > 10.0 * worst  # the jumps are what the ledger takes out
    for row in rows:
        assert row["momentum_drift"] < 1e-7, row["momentum_drift"]
    last = rows[-1]
    assert last["kinetic_escapers"] > 0.0 and last["potential_couple"] < 0.0
    assert res.escapers["ledger"] == pytest.approx(sum(jumps), rel=1e-12)


def test_the_result_and_the_edges_are_in_id_order(direct_run):
    res, seen = direct_run["res"], direct_run["seen"]
    assert np.array_equal(res.masses, direct_run["masses"])
    r = np.linalg.norm(res.state[:, 0], axis=1)
    assert np.all(r[N : N + K] > R_ESCAPE) and np.all(r[:N] < R_ESCAPE)
    assert [s for s, _ in seen] == [STEPS * i for i in range(SEGMENTS + 1)]
    assert np.array_equal(seen[0][1], direct_run["state"])
    assert np.array_equal(seen[-1][1], res.state)
    got = res.handle.gather()
    assert np.array_equal(got["positions"], res.state[:, 0])
    # the escapers are on the devices they left from (their tail), by id
    assert set(np.unique(got["owner"][N:])) <= set(range(NDEV))


def test_restart_and_chained_equal_one_go(direct_run):
    res, path = direct_run["res"], direct_run["path"]
    mid = path.format(segment=SEGMENTS // 2)
    assert res.restarts == [mid, path.format(segment=SEGMENTS)]
    migrate = direct_run["migrate"]
    resumed = _direct_runner(migrate=migrate).run(
        dt=DT, num_steps=STEPS * SEGMENTS, restart_from=mid
    )
    assert resumed.restored["bitwise"], resumed.restored
    assert np.array_equal(resumed.state, res.state)
    assert resumed.escapers["ledger"] == res.escapers["ledger"]
    assert resumed.escapers["n_bound"] == res.escapers["n_bound"]
    for a, b in zip(resumed.edges, res.edges[SEGMENTS // 2 + 1 :]):
        assert a.diagnostics == b.diagnostics
    runner = _direct_runner(migrate=migrate)
    state, masses = direct_run["state"], direct_run["masses"]
    part = runner.run(state, masses, dt=DT, num_steps=STEPS * SEGMENTS // 2)
    chained = runner.run(dt=DT, num_steps=STEPS * SEGMENTS, restart_from=part)
    assert np.array_equal(chained.state, res.state)
    assert chained.escapers == res.escapers


def test_nobody_escaping_is_the_run_without_escapers():
    state, masses = _system(fliers=False)
    on = _direct_runner(EscaperPolicy(R_ESCAPE, R_RETURN)).run(
        state, masses, dt=DT, num_steps=2 * STEPS
    )
    off = _direct_runner(escapers=None).run(state, masses, dt=DT, num_steps=2 * STEPS)
    assert on.escapers["n_bound"] == N and on.escapers["transfers"] == 0
    assert np.array_equal(on.state, off.state)
    assert np.array_equal(
        on.handle.gather()["acceleration"], off.handle.gather()["acceleration"]
    )
    for a, b in zip(on.edges, off.edges):
        da = dict(a.diagnostics)
        assert da["n_escapers"] == 0 and da["escaper_transfers"] == 0
        for name in ESCAPER_FIELDS:
            da.pop(name, None)
        assert da == b.diagnostics
        plan = dict(a.plan)
        assert plan.pop("escapers")["counts"] == [0] * NDEV
        assert plan == b.plan


def _trap(state_rows, params):
    """A weak harmonic trap toward the origin (module level: one object)."""
    return -params["k"] * state_rows[:, 0]


def _trap_potential(state_rows, params):
    """Its potential per unit mass."""
    return 0.5 * params["k"] * jnp.sum(state_rows[:, 0] ** 2, axis=1)


def test_the_external_field_acts_on_the_escapers():
    """With an external field on every row (the tail too) and its potential in the
    energy (``W_ext`` of the escapers included), the ledgered energy stays within
    the no-split control's noise."""
    state, masses = _system()
    params = {"k": 0.02}

    def run(escapers):
        runner = _direct_runner(escapers)
        runner.external_acceleration_fn = _trap
        runner.external_potential_fn = _trap_potential
        return runner.run(
            state, masses, dt=DT, num_steps=STEPS * SEGMENTS, external_params=params
        )

    res, control = run(EscaperPolicy(R_ESCAPE, R_RETURN)), run(None)
    rows = [e.diagnostics for e in res.edges]
    assert rows[-1]["n_escapers"] >= 4 and res.escapers["transfers"] > 1
    worst = max(abs(r["energy_ledgered_drift"]) for r in rows)
    raw = max(abs(r["energy_drift"]) for r in rows)
    noise = max(abs(e.diagnostics["energy_drift"]) for e in control.edges)
    assert worst < max(5.0 * noise, 3e-6), (worst, noise)
    assert raw > 10.0 * worst


def test_a_tail_that_outgrows_its_capacity_is_grown(direct_run, monkeypatch):
    if direct_run["migrate"]:
        pytest.skip("one layout is enough")
    monkeypatch.setattr(mesh_escapers, "TAIL_FLOOR", 1)
    monkeypatch.setattr(mesh_escapers, "TAIL_ALIGN", 1)
    state, masses = direct_run["state"], direct_run["masses"]
    res = _direct_runner().run(state, masses, dt=DT, num_steps=STEPS * SEGMENTS)
    assert res.escapers["tail_grown"] >= 1, res.escapers
    assert [e.diagnostics["n_escapers"] for e in res.edges] == COUNTS
    want = direct_run["res"]
    np.testing.assert_allclose(res.state, want.state, rtol=0, atol=1e-5)


def test_a_retried_segment_restores_the_tail():
    """A migration buffer of one row overflows in the first segment (the tail
    already holds the returner): the segment is re-run from its start, the tail's
    state from its host copy -- bitwise the run whose buffer fitted."""
    state, masses = _system()
    state = state.copy()
    state[:, 1] *= 3.0  # rows cross the cuts within the first segment

    def run(migration_cap):
        runner = _direct_runner(
            diagnostics_every_energy=0, migrate=True, migration_cap=migration_cap
        )
        return runner.run(state, masses, dt=DT, num_steps=4 * STEPS)

    small, ample = run(1), run(4096)
    assert small.edges[1].retries >= 1, [e.retries for e in small.edges]
    assert all(e.retries == 0 for e in ample.edges)
    assert [e.diagnostics["n_escapers"] for e in small.edges][0] == 1
    assert np.array_equal(small.state, ample.state)


def test_the_tail_programs_are_equal_and_opposite():
    """The coupling over the mesh: ``M_b a_u = -sum m_e a_e`` from the folded sums."""
    mesh = _mesh()
    from jax.sharding import PartitionSpec as P

    rng = np.random.default_rng(3)
    pos = rng.normal(size=(NDEV * 32, 3))
    tail = rng.normal(size=(NDEV * 8, 3)) * 30.0
    m = rng.uniform(0.5, 1.5, NDEV * 32)
    me = rng.uniform(0.5, 1.5, NDEV * 8)

    def body(p, mm, t, tm):
        live = jnp.ones(p.shape[0], bool)
        m_b, x_b = mesh_escapers.bound_frame_shard(p, mm, live, AXIS)
        a_u, a_e, _ = mesh_escapers.coupling_shard(
            m_b, x_b, t, tm, jnp.ones(t.shape[0], bool), 1.0, AXIS
        )
        return m_b, a_u, a_e

    m_b, a_u, a_e = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS),) * 4,
            out_specs=(P(), P(), P(AXIS)),
            check_vma=False,
        )
    )(pos, m, tail, me)
    total = float(m_b) * np.asarray(a_u) + np.sum(me[:, None] * np.asarray(a_e), 0)
    np.testing.assert_allclose(total, 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# The fused mesh force, both MACs.
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cpu_fused_lane():
    """The fused mesh lane's CPU switches, with ESC's CSR near field."""
    import yggdrax.distributed.comm as ygg_comm

    from tests.integration.strict import test_production_runner as tpr
    from tests.integration.strict import test_strict_runners_on_provider as r6

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax, "default_backend", lambda: "gpu")
        resolve = ygg_comm.resolve_ragged_method
        mp.setattr(
            ygg_comm,
            "resolve_ragged_method",
            lambda method="auto": "buf" if method == "auto" else resolve(method),
        )
        for key in sorted(set(r6._ENV_UNSET) | set(tpr._ENV_UNSET)):
            mp.delenv(key, raising=False)
        for key, val in tpr._CSR_ON_CPU.items():
            mp.setenv(key, val)
        mp.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "0")
        mp.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")
        mp.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")
        yield


@pytest.mark.parametrize("mac", ["dehnen", "dehnen_error"])
def test_the_fused_force_under_both_macs(cpu_fused_lane, mac, tmp_path):
    """ESC's single-card solver and settings on the mesh: the single card's counts,
    the ledger, and a restart from the mid-run file equal to the run, bitwise."""
    from tests.integration.strict import test_escapers as te

    def runner(**seg):
        return ProductionRunner(
            te._solver(mac),
            settings=te._settings(mac),
            segment=SegmentConfig(steps=STEPS, diagnostics_every_energy=1, **seg),
            escapers=EscaperPolicy(R_ESCAPE, R_RETURN),
            mesh=MeshRunConfig(
                _mesh(), cap=int(np.ceil(1.5 * (N + K + 1) / NDEV)), num_samples=64
            ),
        )

    state, masses = _system()
    path = str(tmp_path / "restart_{segment}.npz")
    res = runner(restart_every=SEGMENTS // 2, restart_path=path).run(
        state, masses, dt=DT, num_steps=STEPS * SEGMENTS
    )
    rows = [e.diagnostics for e in res.edges]
    assert [r["n_escapers"] for r in rows] == COUNTS
    assert all(r["energy_ok"] for r in rows)
    worst = max(abs(r["energy_ledgered_drift"]) for r in rows)
    assert worst < 5e-5, worst
    assert max(r["momentum_drift"] for r in rows) < 1e-6
    resumed = runner().run(
        dt=DT,
        num_steps=STEPS * SEGMENTS,
        restart_from=path.format(segment=SEGMENTS // 2),
    )
    assert resumed.restored["bitwise"], resumed.restored
    assert np.array_equal(resumed.state, res.state)
    assert resumed.escapers["ledger"] == res.escapers["ledger"]
