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
  :class:`~jaccpot.production.RunResult` finishes it as the straight run does.
"""

from __future__ import annotations

import dataclasses

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
    segment_lengths,
)

N = 512
LEAF, ORDER, THETA = 16, 4, 0.6
SOFTENING, DT = 1e-2, 1e-3
#: the runner's run-time half at this operating point
SETTINGS = GalaxySettings(leaf_size=LEAF, max_order=ORDER, theta=THETA)
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
    "steps, num_steps", [(2, 6), (2, 5)], ids=["3x2_vs_6", "2+2+1_vs_5"]
)
def test_segments_are_one_long_call(steps, num_steps, six_steps):
    """R1: the runner's segments compute what one ``strict_run_v2`` call of all the
    steps computes, bit for bit, and its last handle is that call's. Without
    ``donate_input`` the first segment's scan and the later ones' are two programs
    (the second donates its input); the remainder segment is a third."""
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
    assert [e.step for e in res.edges] == [0, *range(steps, num_steps, steps)] + [
        num_steps
    ]
    assert [e.segment for e in res.edges] == list(range(len(res.edges)))
    assert res.edges[-1].time == pytest.approx(num_steps * DT, rel=1e-14)
    assert res.step == num_steps and not res.stopped
    assert all(e.retries == 0 for e in res.edges)
    # the edges kept in the result hold no arrays
    assert all(e.state is None and e.self_acceleration is None for e in res.edges)
    assert len(_particle_runners(runner)) == 2 + bool(num_steps % steps)
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
    res = _runner(
        SegmentConfig(
            steps=2, traced_params=form == "traced_params", diagnostics_every=0
        ),
        external_acceleration_fn=field,
    ).run(state, masses, dt=DT, num_steps=6, external_params=params)
    assert np.array_equal(np.asarray(res.state), ref_state)
    assert res.handle.acceleration is not None


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
    """R4: with diagnostics at every edge the states are those without (the long
    call's, as R1's runner without them), and their numbers are the fp64 moments
    of the edge's state against step 0's."""
    state, masses = _system()
    on = _runner(_static(steps=2, diagnostics_every=1)).run(
        state, masses, dt=DT, num_steps=6
    )
    assert np.array_equal(np.asarray(on.state), six_steps[0])
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
    # energy needs the fast-lane potential (S3 session 2)
    assert first["energy"] is None


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


def test_segment_lengths_put_the_remainder_last():
    assert segment_lengths(6, 2) == (2, 2, 2)
    assert segment_lengths(5, 2) == (2, 2, 1)
    assert segment_lengths(1, 4) == (1,)
    assert segment_lengths(0, 3) == ()


def test_segment_config_is_checked():
    with pytest.raises(ValueError, match="steps"):
        SegmentConfig(steps=0)
    with pytest.raises(ValueError, match="unknown diagnostics"):
        SegmentConfig(diagnostics=("momentum", "virial"))
    with pytest.raises(ValueError, match="particles"):
        ProductionRunner(None, settings=dataclasses.replace(SETTINGS, carry="state"))
    with pytest.raises(TypeError, match="RunResult"):
        ProductionRunner(None, settings=SETTINGS).run(
            dt=DT, num_steps=1, restart_from=object()
        )
    assert isinstance(RunResult, type)
