"""``jaccpot.production.ProductionRunner`` (S3): a run of segments is one long call.

The runner is a host loop of ``strict_run_v2(carry="particles")`` segments chained
through the handle. Its gates, on CPU at N = 512 (four blobs, the production leaf
partition, the pure-JAX near field), each bitwise:

* R1: 3 segments of 2 steps are one call of 6; 2 + 2 + 1 are one call of 5 (and
  the remainder warns);
* R2: an external field across the segment edges -- where chained calls without
  ``carry_total_acceleration`` evaluate the field again outside the scan and differ
  from one call (1 of 3072 components, 1.9e-6;
  ``test_strict_runners_on_provider.py::test_chained_calls_with_an_external_field``)
  -- with a one-argument field and with traced field parameters;
* R3: the runner with traced parameters is ``strict_run_v2`` with the same traced
  parameters in one call;
* R4: the diagnostics leave the trajectory alone; their values are the moments;
* R9: donating the input changes no number, and consumes the input only when asked;
* a callback stops the run after its edge, and a run continued from the
  :class:`~jaccpot.production.RunResult` finishes it as the straight run does;
* R5: 2 segments, a restart file, 4 more segments in a FRESH process (a child that
  imports this module) are the 6 segments of the straight run -- geometric MAC,
  ``dehnen_error`` (CSR near field in interpret mode), an external field with traced
  parameters, traced force parameters -- and a restore whose configuration differs
  says so;
* R6 through the runner: a tree that outgrows its leaf capacity in the SECOND
  segment (``test_strict_runners_on_provider.py``'s drift into four blobs, the CSR
  near field in interpret mode) is re-planned at that segment's start -- from the
  handle, the donated state's host copy -- and the run equals ONE
  ``strict_run_v2`` call that re-planned at its own start.

``jax.default_backend`` reports ``"gpu"``, as
``tests/characterization/test_lane_goldens.py`` does.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.production import (
    DiagnosticsLog,
    GalaxySettings,
    ProductionRunner,
    RunResult,
    SegmentConfig,
    SnapshotWriter,
    load_restart,
    segment_lengths,
)
from tests.integration.strict import test_strict_runners_on_provider as r6
from tests.integration.strict.test_strict_runners_on_provider import (  # noqa: F401
    _fresh_level_registry,
)

N = 512
LEAF, ORDER, THETA = 16, 4, 0.6
SOFTENING, DT = 1e-2, 1e-3
#: the runner's run-time half at this operating point, with the geometric MAC of
#: ``_solver`` (the restart fingerprint records the settings; ``preset="galaxy"``'s
#: MAC is ``dehnen_error`` since 2026-10-21)
SETTINGS = GalaxySettings(
    leaf_size=LEAF, max_order=ORDER, theta=THETA, mac_type="dehnen", adaptive_eps=None
)
#: the halo of the external-field gates (provider_hlo_identity.py's NFW)
HALO_M, HALO_RS = 10.0, 1.0

_ENV_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_STRICT_SEGMENT_RETRY",
    "JACCPOT_STRICT_CARRY",
    "JACCPOT_STRICT_CARRY_ORDER",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB",
)


@pytest.fixture(scope="module", autouse=True)
def _lane():
    # module scope: the shared reference below is computed under it
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jax, "default_backend", lambda: "gpu")
        for key in _ENV_UNSET:
            patch.delenv(key, raising=False)
        yield


def _solver():
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFTENING,
        working_dtype=jnp.float32,
        use_pallas=False,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
    )


def _system():
    """Four tight blobs (provider_hlo_identity.py's CPU draw), with velocities."""
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    pos = np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )
    vel = 0.1 * rng.normal(size=(N, 3))
    masses = np.abs(rng.normal(size=N)) + 0.5
    state = jnp.asarray(np.stack([pos, vel], axis=1), jnp.float32)
    return state, jnp.asarray(masses, jnp.float32)


def nfw_halo(state_rows):
    """An analytic NFW halo at the packed state's positions (one-argument form)."""
    return nfw_halo_params(state_rows, {"m": HALO_M, "rs": HALO_RS})


def nfw_halo_params(state_rows, params):
    """The same halo with its mass and scale radius as parameters."""
    pos = state_rows[:, 0, :]
    r2 = jnp.sum(pos * pos, axis=1)
    r = jnp.sqrt(jnp.maximum(r2, 1e-30))
    x = r / params["rs"]
    big = jnp.log1p(x) / jnp.maximum(r2 * r, 1e-30) - 1.0 / jnp.maximum(
        r2 * (params["rs"] + r), 1e-30
    )
    ser = (1.0 / (2.0 * params["rs"] ** 3)) * (1.0 - (4.0 / 3.0) * x)
    coeff = jnp.where(x < 1e-3, ser, big)
    return -params["m"] * coeff[:, None] * pos


def nfw_potential_params(state_rows, params):
    """The halo's potential per unit mass, ``-m log(1 + r / rs) / r`` (Odisseo's
    ``potentials.NFW``), the field's partner for the energy diagnostic."""
    pos = state_rows[:, 0, :]
    r = jnp.sqrt(jnp.maximum(jnp.sum(pos * pos, axis=1), 1e-30))
    x = r / params["rs"]
    log1p_over_x = jnp.where(
        x < 1e-3, 1.0 - 0.5 * x + x * x / 3.0, jnp.log1p(x) / jnp.maximum(x, 1e-30)
    )
    return -params["m"] / params["rs"] * log1p_over_x


def _direct_energy(state, masses, *, halo=None):
    """fp64 ``(K, 1/2 sum m phi, sum m Phi_halo)`` of a packed state, with the
    test solver's pair kernel (the library default)."""
    from jaccpot.softening import resolve_softening_kernel
    from tests.integration.strict.test_fused_potential import direct_potential

    s = np.asarray(state, np.float64)
    m = np.asarray(masses, np.float64)
    kinetic = 0.5 * np.sum(m * np.sum(s[:, 1] ** 2, axis=1))
    kernel = resolve_softening_kernel(None)
    w_self = 0.5 * np.sum(
        m * direct_potential(s[:, 0], m, kernel=kernel, softening=SOFTENING)
    )
    w_ext = 0.0
    if halo is not None:
        r = np.linalg.norm(s[:, 0], axis=1)
        w_ext = np.sum(m * -halo["m"] * np.log1p(r / halo["rs"]) / r)
    return kinetic, w_self, w_ext


def _one_call(steps, *, external=None, external_params=None, traced=False):
    """``strict_run_v2`` once for all ``steps``: the reference of every gate."""
    from jaccpot import ForceParams

    state, masses = _system()
    kw = {}
    dt = DT
    if traced:
        dt = jnp.asarray(DT, jnp.float32)
        kw["force_params"] = ForceParams(
            G=jnp.asarray(1.0, jnp.float32),
            eps2=jnp.asarray(SOFTENING**2, jnp.float32),
        )
    out, handle, _ = _solver().strict_run_v2(
        state=state,
        masses=masses,
        dt=dt,
        num_steps=steps,
        refresh_every=1,
        leaf_size=LEAF,
        max_order=ORDER,
        carry="particles",
        add_external=external is not None,
        external_acceleration_fn=external,
        external_params=external_params,
        **kw,
    )
    return np.asarray(out), np.asarray(handle.self_acceleration)


@pytest.fixture(scope="module")
def six_steps():
    """``strict_run_v2`` once for 6 steps, static parameters: the reference the
    runner's 3 x 2 segments meet in several gates (computed once per worker)."""
    return _one_call(6)


def _runner(segment, **kw):
    return ProductionRunner(_solver(), settings=SETTINGS, segment=segment, **kw)


def _particle_runners(runner):
    """The runner's scan programs (one per segment length and donation)."""
    cache = runner.engine._strict_fused_jit_function_cache
    return [k for k in cache if k[0] == "strict_velocity_verlet_particles"]


def _static(**kw):
    """A segment config without traced parameters or diagnostics."""
    return SegmentConfig(**{"traced_params": False, "diagnostics_every": 0, **kw})


@pytest.mark.parametrize(
    "steps, num_steps", [(2, 6), (2, 5)], ids=["3x2_vs_6", "2+3_vs_5"]
)
def test_segments_are_one_long_call(steps, num_steps, six_steps):
    """R1: the runner's segments compute what one ``strict_run_v2`` call of all the
    steps computes, bit for bit, and its last handle is that call's. Without
    ``donate_input`` the first segment's scan and the later ones' are two programs
    (the second donates its input); a last segment of another length adds one.
    A 1-step remainder joins the last segment (``segment_lengths``), so 5 steps of 2
    run as 2 + 3."""
    ref_state, ref_force = six_steps if num_steps == 6 else _one_call(num_steps)
    state, masses = _system()
    runner = _runner(_static(steps=steps))
    if num_steps % steps:
        with pytest.warns(UserWarning, match="compiles its own scan"):
            res = runner.run(state, masses, dt=DT, num_steps=num_steps)
    else:
        res = runner.run(state, masses, dt=DT, num_steps=num_steps)
    assert np.array_equal(np.asarray(res.state), ref_state)
    assert np.array_equal(np.asarray(res.handle.self_acceleration), ref_force)
    plan = segment_lengths(num_steps, steps)
    assert [e.step for e in res.edges] == [0, *np.cumsum(plan).tolist()]
    assert [e.segment for e in res.edges] == list(range(len(res.edges)))
    assert res.edges[-1].time == pytest.approx(num_steps * DT, rel=1e-14)
    assert res.step == num_steps and not res.stopped
    assert all(e.retries == 0 for e in res.edges)
    # the edges kept in the result hold no arrays
    assert all(e.state is None and e.self_acceleration is None for e in res.edges)
    # one scan per (length, donation): the first segment keeps the caller's state,
    # every later one donates the runner's own
    programs = {(plan[0], False)} | {(n, True) for n in plan[1:]}
    assert len(_particle_runners(runner)) == len(programs)
    assert not state.is_deleted()  # the caller's input is kept


@pytest.mark.parametrize("form", ["closure", "traced_params"])
def test_an_external_field_crosses_the_segment_edges(form):
    """R2: the field at each segment's start is the scan's own total from the
    previous segment (the handle's ``acceleration``), not an eager evaluation
    outside the scan, so 3 x 2 steps with the halo are one call of 6."""
    if form == "closure":
        field, params, ref_params = nfw_halo, None, None
    else:
        field = nfw_halo_params
        params = {"m": HALO_M, "rs": HALO_RS}  # the runner traces the floats
        ref_params = {k: jnp.asarray(v, jnp.float32) for k, v in params.items()}
    ref_state, _ = _one_call(
        6,
        external=field,
        external_params=ref_params,
        traced=form == "traced_params",
    )
    state, masses = _system()
    traced = form == "traced_params"
    # the traced form with the energy at every edge, the halo's potential included
    # (S3 R4 with a field): the trajectory stays the one call's
    res = _runner(
        SegmentConfig(
            steps=2,
            traced_params=traced,
            diagnostics_every=int(traced),
            diagnostics_every_energy=1,
        ),
        external_acceleration_fn=field,
        external_potential_fn=nfw_potential_params if traced else None,
    ).run(state, masses, dt=DT, num_steps=6, external_params=params)
    assert np.array_equal(np.asarray(res.state), ref_state)
    assert res.handle.acceleration is not None
    if traced:
        last = res.edges[-1].diagnostics
        kinetic, w_self, w_ext = _direct_energy(res.state, masses, halo=params)
        assert last["energy_ok"]
        assert last["potential_external"] == pytest.approx(w_ext, rel=1e-6)
        assert last["potential_self"] == pytest.approx(w_self, rel=1e-3)
        e0 = res.baselines["energy"]
        assert e0 == pytest.approx(
            sum(_direct_energy(state, masses, halo=params)), rel=1e-3
        )
        assert last["energy_drift"] == (last["energy"] - e0) / abs(e0)


def test_the_traced_runner_is_the_traced_call():
    """R3: ``traced_params=True`` (the runner's default) passes dt, G and eps2 as
    arrays, and computes what ``strict_run_v2`` with those arrays computes."""
    ref_state, ref_force = _one_call(6, traced=True)
    state, masses = _system()
    res = _runner(SegmentConfig(steps=2, diagnostics_every=0)).run(
        state, masses, dt=DT, num_steps=6
    )
    assert np.array_equal(np.asarray(res.state), ref_state)
    assert np.array_equal(np.asarray(res.handle.self_acceleration), ref_force)
    # one compile for any value: the arrays are arguments, the handle records them
    assert res.handle.force_key == (1.0, float(np.float32(SOFTENING**2)))


def test_the_diagnostics_leave_the_trajectory_alone(six_steps):
    """R4: with diagnostics -- the energy too -- at every edge the states are those
    without (the long call's, as R1's runner without them), with the same two scan
    programs, and their numbers are the fp64 moments of the edge's state against
    step 0's and the energy ``K + 1/2 sum m phi`` against the direct sum's."""
    state, masses = _system()
    runner = _runner(_static(steps=2, diagnostics_every=1, diagnostics_every_energy=1))
    on = runner.run(state, masses, dt=DT, num_steps=6)
    assert np.array_equal(np.asarray(on.state), six_steps[0])
    assert len(_particle_runners(runner)) == 2  # as R1's 3 x 2 without diagnostics
    first, last = on.edges[0].diagnostics, on.edges[-1].diagnostics
    x = np.asarray(state, np.float64)
    m = np.asarray(masses, np.float64)
    p0 = (m[:, None] * x[:, 1]).sum(0)
    np.testing.assert_allclose(on.baselines["momentum"], p0, rtol=1e-12)
    assert on.baselines["mass"] == pytest.approx(m.sum(), rel=1e-14)
    assert first["momentum_drift"] == 0.0 and first["kinetic_change"] == 0.0
    assert first["com_drift"] == 0.0 and first["angular_momentum_drift"] == 0.0
    # the self-gravity of an FMM conserves momentum to its accuracy, not exactly
    xf = np.asarray(on.state, np.float64)
    pf = (m[:, None] * xf[:, 1]).sum(0)
    np.testing.assert_allclose(last["momentum"], pf, rtol=1e-12, atol=1e-12)
    assert 0.0 < last["momentum_drift"] < 1e-3
    # step 0's energy needs the first segment's provider: the baseline, not its row
    assert first["energy"] is None
    k0, w0, _ = _direct_energy(state, masses)
    assert on.baselines["kinetic"] == pytest.approx(k0, rel=1e-12)
    assert on.baselines["potential"] == pytest.approx(w0, rel=1e-3)
    kinetic, w_self, _ = _direct_energy(on.state, masses)
    assert last["energy_ok"] and last["potential_external"] == 0.0
    assert last["potential_self"] == pytest.approx(w_self, rel=1e-3)
    assert last["energy"] == pytest.approx(kinetic + w_self, rel=1e-3)
    e0 = on.baselines["energy"]
    assert last["energy_drift"] == (last["energy"] - e0) / abs(e0)
    # every edge after the first segment has its energy
    assert all(e.diagnostics["energy"] is not None for e in on.edges[1:])
    # step 0's energy and the segments' run ONE executable (step 0's state arrives
    # committed, the segments' do not: two compiles of ~50 s at 25M before)
    assert [p._cache_size() for p in runner._energy_programs.values()] == [1]


def test_donating_the_input_changes_no_number(six_steps):
    """R9: ``donate_input`` changes no number (segments 2 and later always donate)
    and gives every segment ONE scan program. (On a card the donated input is
    deleted; the CPU backend may keep it.)"""
    state, masses = _system()
    given = jnp.array(state)
    runner = _runner(_static(steps=2, donate_input=True))
    donated = runner.run(given, masses, dt=DT, num_steps=6)
    assert np.array_equal(np.asarray(donated.state), six_steps[0])
    assert len(_particle_runners(runner)) == 1
    if jax.devices()[0].platform == "gpu":
        assert given.is_deleted()


def test_a_callback_stops_the_run_and_a_continuation_finishes_it(six_steps):
    """A callback returning True stops after its edge; ``restart_from`` the
    result runs the missing steps, and the two calls are the straight run."""
    state, masses = _system()
    log, snaps = DiagnosticsLog(), SnapshotWriter(every=2)
    runner = _runner(
        _static(steps=2), callbacks=(log, snaps, lambda edge: edge.segment == 2)
    )
    first = runner.run(state, masses, dt=DT, num_steps=6)
    assert first.stopped and first.step == 4 and first.segment == 2
    rest = runner.run(dt=DT, num_steps=6, restart_from=first)
    assert first.state is None  # consumed
    assert np.array_equal(np.asarray(rest.state), six_steps[0])
    assert [e.step for e in rest.edges] == [6]
    assert rest.time == pytest.approx(6 * DT, rel=1e-14)
    # the log saw every edge of both calls; the snapshots every second segment
    assert [row["step"] for row in log.rows] == [0, 2, 4, 6]
    assert [s["step"] for s in snaps.snapshots] == [0, 4]
    assert np.array_equal(snaps.snapshots[0]["state"], np.asarray(state))


def test_a_run_without_segments_keeps_the_callers_state():
    """A run with nothing to do returns the caller's own array, marked as not the
    runner's, so a continuation from it does not consume it (``donate_input``
    off): only a segment's output is the runner's to donate."""
    state, masses = _system()
    res = _runner(_static(steps=2)).run(state, masses, dt=DT, num_steps=0)
    assert res.state is state and not res.state_owned
    assert [e.step for e in res.edges] == [0] and res.handle is None
    given = _runner(_static(steps=2, donate_input=True)).run(
        state, masses, dt=DT, num_steps=0
    )
    assert given.state_owned  # the caller handed it over


def test_segment_lengths_put_the_remainder_last():
    assert segment_lengths(6, 2) == (2, 2, 2)
    assert segment_lengths(8, 3) == (3, 3, 2)
    assert segment_lengths(1, 4) == (1,)
    assert segment_lengths(0, 3) == ()


def test_a_one_step_remainder_joins_the_last_segment():
    """A scan of length one compiles to other code than a longer scan's loop body."""
    assert segment_lengths(5, 2) == (2, 3)
    assert segment_lengths(7, 3) == (3, 4)
    assert segment_lengths(3, 2) == (3,)
    assert segment_lengths(4, 1) == (1, 1, 1, 1)  # steps=1 is the caller's choice


def test_segment_config_is_checked():
    with pytest.raises(ValueError, match="steps"):
        SegmentConfig(steps=0)
    with pytest.raises(ValueError, match="unknown diagnostics"):
        SegmentConfig(diagnostics=("momentum", "virial"))
    with pytest.raises(ValueError, match="restart_path"):
        SegmentConfig(restart_every=2)
    with pytest.raises(ValueError, match="particles"):
        ProductionRunner(None, settings=dataclasses.replace(SETTINGS, carry="state"))
    with pytest.raises(TypeError, match="RunResult"):
        ProductionRunner(None, settings=SETTINGS).run(
            dt=DT, num_steps=1, restart_from=object()
        )
    assert isinstance(RunResult, type)


# --------------------------------------------------------------------------
# R5: a restart in a fresh process continues the run bitwise.
# --------------------------------------------------------------------------

#: the CSR near field in interpret mode: dehnen_error's force scale needs it
_CSR_ON_CPU = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}
_RESTART_VARIANTS = ("geometric", "dehnen_error", "external", "traced")


def _restart_solver(variant):
    if variant != "dehnen_error":
        return _solver()
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFTENING,
        working_dtype=jnp.float32,
        use_pallas=True,
        adaptive_eps=1e-5,
        adaptive_error_model="dehnen_paper",
        mac_force_scale_mode="paper_fb",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen_error",
        ),
        fixed_order=ORDER,
    )


def _restart_runner(variant, **segment):
    """The variant's runner: 2-step segments, diagnostics every edge; the energy
    (step 0's after the first segment, then every third edge) in the external
    variant only, with the halo's potential (each energy program is a compile)."""
    external = variant == "external"
    config = SegmentConfig(
        steps=2,
        traced_params=variant in ("external", "traced"),
        diagnostics_every_energy=3 if external else 0,
        **segment,
    )
    return ProductionRunner(
        _restart_solver(variant),
        settings=SETTINGS,
        segment=config,
        external_acceleration_fn=nfw_halo_params if external else None,
        external_potential_fn=nfw_potential_params if external else None,
    )


def _restart_params(variant):
    return {"m": HALO_M, "rs": HALO_RS} if variant == "external" else None


def _resume(variant, path, out):
    """The fresh process: continue ``path`` to step 12; save what it computed."""
    jax.config.update("jax_enable_x64", True)
    jax.default_backend = lambda: "gpu"  # as the _lane fixture
    if variant == "dehnen_error":
        os.environ.update(_CSR_ON_CPU)
    runner = _restart_runner(variant)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = runner.run(
            dt=DT,
            num_steps=12,
            restart_from=path,
            external_params=_restart_params(variant),
        )
    np.savez(
        out,
        state=np.asarray(res.state),
        self_acceleration=np.asarray(res.handle.self_acceleration),
        steps=np.asarray([e.step for e in res.edges]),
        diagnostics=np.asarray(json.dumps(res.edges[-1].diagnostics)),
        restored=np.asarray(json.dumps(res.restored)),
        warned=np.asarray(json.dumps([str(w.message) for w in caught])),
    )


@pytest.mark.parametrize("variant", _RESTART_VARIANTS)
def test_a_restart_in_a_fresh_process_continues_bitwise(variant, tmp_path, monkeypatch):
    """R5: 6 segments straight (writing a restart after each) against 2 of them, the
    restart file, and 4 more segments in a fresh process: the same state and
    self-gravity, bit for bit, with the restore's digests matching."""
    if variant == "dehnen_error":
        for key, val in _CSR_ON_CPU.items():
            monkeypatch.setenv(key, val)
    state, masses = _system()
    runner = _restart_runner(
        variant, restart_every=1, restart_path=str(tmp_path / "restart_{segment}.npz")
    )
    straight = runner.run(
        state, masses, dt=DT, num_steps=12, external_params=_restart_params(variant)
    )
    assert len(straight.restarts) == 6
    path = tmp_path / "restart_2.npz"
    arrays, meta = load_restart(path)
    expected = {"state", "masses", "self_acceleration"}
    expected |= {"force_scale"} if variant == "dehnen_error" else set()
    expected |= {"acceleration"} if variant == "external" else set()
    assert set(arrays) == expected
    assert meta["counters"]["step"] == 4 and meta["counters"]["segment"] == 2
    assert meta["baselines"] == straight.baselines
    out = tmp_path / "resumed.npz"
    # the module by its own name, so the field is the same function by name
    child = f"import {__name__} as m; m._resume(*{[variant, str(path), str(out)]!r})"
    proc = subprocess.run(
        [sys.executable, "-c", child],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    with np.load(out) as got:
        restored = json.loads(str(got["restored"]))
        assert restored["bitwise"], (restored, str(got["warned"]))
        assert json.loads(str(got["warned"])) == []
        assert got["steps"].tolist() == [6, 8, 10, 12]
        assert np.array_equal(got["state"], np.asarray(straight.state))
        assert np.array_equal(
            got["self_acceleration"], np.asarray(straight.handle.self_acceleration)
        )
        if variant == "external":
            # the energy too: E0 from the file, E from the restored provider
            last = json.loads(str(got["diagnostics"]))
            ref = straight.edges[-1].diagnostics
            assert last["energy"] is not None and last["energy_ok"]
            assert last["energy"] == ref["energy"]
            assert last["energy_drift"] == ref["energy_drift"]


def test_a_restore_says_when_it_cannot_be_bitwise(tmp_path):
    """A restore in this process with the run's own configuration matches its
    digests; with another dt it warns and records why (and still continues)."""
    state, masses = _system()
    path = tmp_path / "restart.npz"
    first = _runner(_static(steps=2, restart_every=2, restart_path=str(path))).run(
        state, masses, dt=DT, num_steps=4
    )
    assert first.restarts == [str(path)]
    # num_steps reached already: the restore alone, no segment
    same = _runner(_static(steps=2)).run(dt=DT, num_steps=4, restart_from=path)
    assert same.restored["bitwise"] and same.restored["mismatches"] == []
    assert np.array_equal(np.asarray(same.state), np.asarray(first.state))
    with pytest.warns(UserWarning, match="not bitwise"):
        other = _runner(_static(steps=2)).run(dt=2 * DT, num_steps=4, restart_from=path)
    assert other.restored["mismatches"] == ["dt"]
    with pytest.raises(RuntimeError, match="has done a segment"):
        _runner(_static(steps=2)).save_restart(tmp_path / "nothing.npz")


# --------------------------------------------------------------------------
# R6 through the runner: a segment whose tree outgrows its bounds is re-planned
# at its start, and the run finishes as strict_run_v2's re-planned run does.
# --------------------------------------------------------------------------


def test_a_segment_that_outgrows_its_tree_is_replanned_and_the_run_finishes(
    monkeypatch, _fresh_level_registry
):
    """``test_strict_runners_on_provider.py``'s R6 drift at a quarter of its speed:
    from a uniform draw prepared at leaf capacity 96 (64 live leaves) the particles
    reach the four blobs at step 8, and steps 1-4 cut 64, 90, 110 and 109 live
    leaves. The first segment (steps 1-2) fits; the second (steps 3-4) outgrows 96
    leaves. ``strict_run_v2`` re-plans it at ITS start -- the runner's handle, the
    host copy of the state the segment donated, capacity ``auto_leaf_capacity(110)
    = 1024`` -- and runs it again; the edge reports the retry. The walk caps are
    R6's named ones (never overflowed), so only the tree is re-planned.

    The run equals one ``strict_run_v2`` call of the 4 steps, bitwise: that call
    re-plans at step 0 and runs every step at capacity 1024, the runner's first
    segment ran at 96 -- the capacity a tree is padded to changes no force."""
    for key, val in (r6._CSR_ON_CPU | r6._WALK_CAPS["named"]).items():
        monkeypatch.setenv(key, val)
    blobs, _, _ = r6._systems()
    uniform = np.random.default_rng(2).uniform(-1.0, 1.0, (N, 3))
    # a quarter of the way to the blobs after R6's 2 steps: the blobs at step 8
    state, masses = r6._drifting(
        uniform, uniform + 0.25 * (np.asarray(blobs) - uniform)
    )

    solver = r6._solver(leaf_capacity=96, use_pallas=True)
    res = ProductionRunner(solver, settings=SETTINGS, segment=_static(steps=2)).run(
        state, masses, dt=DT, num_steps=4
    )
    assert res.step == 4 and not res.stopped
    assert [(e.step, e.retries) for e in res.edges] == [(0, 0), (2, 0), (4, 1)]
    assert [e.plan["leaf_capacity"] for e in res.edges] == [96, 96, 1024]
    diag = solver.get_runtime_diagnostics()
    assert diag["strict_fused_last_fallback_reason"] == "capacity_segment_retry"

    reference = r6._solver(leaf_capacity=96, use_pallas=True)
    want, handle, _ = r6._run(reference, state, masses, carry="particles", steps=4)
    assert r6._fallbacks(reference) == 1  # re-planned at step 0
    assert reference._impl._tree_leaf_capacity == 1024
    r6._final_states_equal(res.state, want)
    r6._final_states_equal(res.handle.self_acceleration, handle.self_acceleration)
