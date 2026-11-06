"""The escaper path in ``ProductionRunner(escapers=...)`` (wave 3, ESC).

A bound Plummer sphere (N = 1024, clipped at r = 3) with eight fliers just inside
``r_escape = 5`` leaving at 6 to 12 times the escape speed (at different edges)
and one particle outside it falling back inside ``r_return = 4.5`` (the
hysteresis), on CPU (the CSR near field in interpret mode, cell leaves), 6
segments of 10 steps at dt 1e-3 with the energy at every edge:

* the split changes at several edges and compiles ONE split scan program (the
  tree's row count is traced);
* the ledger: ``energy_ledgered = E - sum(jumps)`` stays within the run's own
  noise (the no-split control's, the FMM and fp32 floor) and the momentum is
  conserved to the FMM's accuracy (the coupling's pair forces are equal and
  opposite), under ``dehnen`` and ``dehnen_error``;
* a restart mid-run (written by the run itself) and a chained run equal the run in
  one go, bitwise;
* the result and the callbacks see the caller's row order;
* with escapers on and nobody escaping, the run IS the run without them (the
  default programs), bitwise.
"""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.escapers import EscaperPolicy, coupling, coupling_energy
from jaccpot.production import GalaxySettings, ProductionRunner, SegmentConfig
from tests.integration.strict import test_production_runner as tpr
from tests.integration.strict import test_strict_runners_on_provider as r6

N, K = 1024, 8
R_ESCAPE, R_RETURN = 5.0, 4.5
DT, STEPS, SEGMENTS = 1e-3, 10, 6
LEAF, ORDER, THETA = 16, 4, 0.6


@pytest.fixture(scope="module", autouse=True)
def _lane():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jax, "default_backend", lambda: "gpu")
        for key in sorted(set(r6._ENV_UNSET) | set(tpr._ENV_UNSET)):
            patch.delenv(key, raising=False)
        for key, val in tpr._CSR_ON_CPU.items():
            patch.setenv(key, val)
        yield


def _system(seed: int = 0, *, fliers: bool = True):
    """Plummer + K fliers (6x..12x escape speed) + one returner, caller order."""
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
        # the returner: at 5.3, outside r_escape, falling in at 15
        pos = np.concatenate([pos, re[:, None] * de[:K], 5.3 * de[K:]])
        vel = np.concatenate([vel, ve[:, None] * de[:K], -15.0 * de[K:]])
    masses = np.full(pos.shape[0], 1.0 / pos.shape[0])
    vel -= np.sum(masses[:, None] * vel, axis=0) / masses.sum()
    return (
        jnp.asarray(np.stack([pos, vel], axis=1), jnp.float32),
        jnp.asarray(masses, jnp.float32),
    )


def _solver(mac: str):
    if mac == "dehnen_error":
        return tpr._restart_solver("dehnen_error")
    return r6._solver(leaf_capacity=1024, use_pallas=True)


def _settings(mac: str) -> GalaxySettings:
    return GalaxySettings(
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        mac_type=mac,
        adaptive_eps=1e-5 if mac == "dehnen_error" else None,
    )


def _runner(mac: str, escapers=EscaperPolicy(R_ESCAPE, R_RETURN), **segment):
    config = dict(steps=STEPS, diagnostics_every_energy=1)
    config.update(segment)
    return ProductionRunner(
        _solver(mac),
        settings=_settings(mac),
        segment=SegmentConfig(**config),
        escapers=escapers,
    )


def _scan_keys(runner) -> set:
    cache = getattr(runner.engine, "_strict_fused_jit_function_cache", {})
    return {k for k in cache if k and k[0] == "strict_velocity_verlet_particles"}


#: one worker per MAC (``--dist loadgroup``): the module fixture runs once each
_MACS = [
    pytest.param(mac, marks=pytest.mark.xdist_group(f"escapers_{mac}"))
    for mac in ("dehnen", "dehnen_error")
]


@pytest.fixture(scope="module", params=_MACS)
def run(request, tmp_path_factory):
    mac = request.param
    state, masses = _system()
    seen: list = []

    def look(edge):
        seen.append(
            (np.asarray(edge.state).copy(), np.asarray(edge.masses).copy(), edge.step)
        )

    # the run writes its mid-run restart (segment 3) for the restart test
    path = str(tmp_path_factory.mktemp(f"esc_{mac}") / "restart_{segment}.npz")
    runner = _runner(mac, restart_every=SEGMENTS // 2, restart_path=path)
    runner.callbacks = (look,)
    res = runner.run(state, masses, dt=DT, num_steps=STEPS * SEGMENTS)
    control = _runner(mac, escapers=None).run(
        state, masses, dt=DT, num_steps=STEPS * SEGMENTS
    )
    return mac, state, masses, runner, res, control, seen, path


def test_the_split_changes_and_compiles_one_scan(run):
    _, _, _, runner, res, _, _, _ = run
    rows = [e.diagnostics for e in res.edges]
    counts = [r["n_escapers"] for r in rows]
    moved = [r["escaper_transfers"] for r in rows]
    # the returner is an escaper from the start, then comes back; the fliers
    # leave at several edges
    assert counts[0] == 1
    assert counts[-1] == K
    assert sum(1 for m in moved[1:] if m) >= 2
    assert all(r["energy_ok"] is not False for r in rows)
    split_keys = [k for k in _scan_keys(runner) if "live" in repr(k)]
    # one split scan per donation flag at most: the row count is traced
    assert 1 <= len(split_keys) <= 2, split_keys
    assert res.escapers["n_bound"] == N + 1  # the bulk and the returner


def test_the_ledger_conserves_the_energy_and_the_momentum(run):
    _, state, masses, _, res, control, _, _ = run
    rows = [e.diagnostics for e in res.edges if e.diagnostics.get("energy")]
    e0 = res.baselines["energy"]
    worst = max(abs(r["energy_ledgered_drift"]) for r in rows)
    jumps = [r["escaper_jump"] for r in rows if r["escaper_jump"] is not None]
    assert jumps and all(abs(j) / abs(e0) < 1e-4 for j in jumps)
    noise = max(
        abs(e.diagnostics["energy_drift"])
        for e in control.edges
        if e.diagnostics.get("energy_drift") is not None
    )
    # within a few times the control's own noise (and the fp32 floor)
    assert worst < max(5.0 * noise, 3e-5), (worst, noise)
    for r in res.edges:
        assert r.diagnostics["momentum_drift"] < 1e-6
    last = res.edges[-1].diagnostics
    assert last["kinetic_escapers"] > 0.0 and last["potential_couple"] < 0.0


def test_the_caller_sees_its_own_rows(run):
    _, state, masses, _, res, _, seen, _ = run
    assert np.array_equal(np.asarray(res.masses), np.asarray(masses))
    r = np.linalg.norm(np.asarray(res.state)[:, 0], axis=1)
    assert np.all(r[N : N + K] > R_ESCAPE) and np.all(r[:N] < R_ESCAPE)
    for s, m, _ in seen:
        assert np.array_equal(m, np.asarray(masses))
    # the callback's last state is the result's
    assert np.array_equal(seen[-1][0], np.asarray(res.state))


def test_restart_and_chained_equal_one_go(run):
    mac, state, masses, _, res, _, _, path = run
    half = STEPS * SEGMENTS // 2
    mid = path.format(segment=SEGMENTS // 2)
    assert res.restarts == [mid, path.format(segment=SEGMENTS)]
    resumed = _runner(mac).run(dt=DT, num_steps=STEPS * SEGMENTS, restart_from=mid)
    assert resumed.restored["bitwise"], resumed.restored
    assert np.array_equal(np.asarray(resumed.state), np.asarray(res.state))
    assert resumed.escapers["ledger"] == res.escapers["ledger"]
    runner = _runner(mac)
    part = runner.run(state, masses, dt=DT, num_steps=half)
    chained = runner.run(dt=DT, num_steps=STEPS * SEGMENTS, restart_from=part)
    assert np.array_equal(np.asarray(chained.state), np.asarray(res.state))
    assert chained.escapers["n_bound"] == res.escapers["n_bound"]


@pytest.mark.xdist_group("escapers_dehnen")
def test_nobody_escaping_is_the_run_without_escapers():
    state, masses = _system(fliers=False)
    on = _runner("dehnen", escapers=EscaperPolicy(100.0)).run(
        state, masses, dt=DT, num_steps=2 * STEPS
    )
    off = _runner("dehnen", escapers=None).run(
        state, masses, dt=DT, num_steps=2 * STEPS
    )
    assert on.escapers["order"] is None and on.escapers["n_bound"] == N
    assert np.array_equal(np.asarray(on.state), np.asarray(off.state))
    for a, b in zip(on.edges, off.edges):
        da, db = dict(a.diagnostics), b.diagnostics
        assert da.pop("n_escapers") == 0 and da.pop("escaper_transfers") == 0
        for name in ("escaper_jump", "escaper_ledger", "potential_couple"):
            da.pop(name, None)
        for name in ("kinetic_escapers", "energy_ledgered", "energy_ledgered_drift"):
            da.pop(name, None)
        assert da == db


def test_the_coupling_is_equal_and_opposite():
    rng = np.random.default_rng(3)
    pos = jnp.asarray(rng.normal(size=(64, 3)) * np.r_[[1.0] * 60, [30.0] * 4][:, None])
    m = jnp.asarray(rng.uniform(0.5, 1.5, 64))
    acc, scale, live = coupling(pos, m, 60, 1.0)
    acc = np.asarray(acc)
    force = np.asarray(m)[:, None] * acc
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-12)
    assert np.all(np.asarray(scale)[:60] == 0) and np.all(np.asarray(scale)[60:] > 0)
    # two bodies: W_c = -G M_b m_e / r
    state = jnp.asarray([[[0.0, 0, 0], [0.0, 1, 0]], [[2.0, 0, 0], [0.0, 0, 3]]])
    energy = coupling_energy(state, jnp.asarray([2.0, 1.0]), 1, 1.0)
    assert float(energy["couple"]) == pytest.approx(-1.0)
    assert float(energy["kinetic_escapers"]) == pytest.approx(4.5)


def test_the_escaper_path_refuses_what_it_does_not_do():
    with pytest.raises(TypeError):
        ProductionRunner(_solver("dehnen"), settings=_settings("dehnen"), escapers=5.0)
    runner = _runner("dehnen")
    state, masses = _system()
    with pytest.raises(ValueError, match="escaper"):
        runner.gradient_plan(state, masses, dt=DT, num_steps=STEPS)
    with pytest.raises(ValueError, match="r_return"):
        EscaperPolicy(5.0, 6.0)
    assert dataclasses.asdict(EscaperPolicy(5.0)) == {
        "r_escape": 5.0,
        "r_return": None,
        "every": 1,
    }
    assert os.environ.get("JACCPOT_NEARFIELD_LEAFPAIR_CSR") == "1"
