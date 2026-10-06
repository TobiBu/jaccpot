"""``strict_run_v2(carry="particles", donate_state=...)`` and the capacity retry.

``donate_state`` hands the input state's buffer to the scan; a segment whose
refresh overflows a list is re-run once from its start with re-planned caps, so
the call keeps a host copy of the start state when donating. The scenario forces
the overflow MID-RUN: the particles drift ballistically (negligible masses) from
a uniform cube, where the eager prepare sizes the caps, into a contracted version
of it inside the same box. The first three steps fit; the fourth refresh does
not. The re-run must equal a run whose caps never overflow (a large cap
headroom), its history included; the stream stops at the failed step, so the
steps before it are streamed once per attempt and never a failed step's state.
"""

from __future__ import annotations

import collections

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        jax.default_backend() != "gpu", reason="fused strict lane is GPU-only"
    ),
]

_N = 20_000
_STEPS = 6
_LEAF, _ORDER, _THETA = 64, 4, 0.8
_FUSED_ENV = {
    "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS": "1",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE": "4",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "64",
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_LARGE_N_COMPILED_STATE_MODE": "on",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET": str(_N),
    "JACCPOT_STRICT_CARRY": "particles",
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key, val in _FUSED_ENV.items():
        monkeypatch.setenv(key, val)
    for key in (
        "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
        "JACCPOT_FLAT_WALK_CAP_HEADROOM",
        "JACCPOT_STRICT_SEGMENT_RETRY",
    ):
        monkeypatch.delenv(key, raising=False)


def _solver():
    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    return FastMultipoleMethod(
        preset="large_n_gpu",
        runtime_path="large_n",
        basis="real",
        theta=_THETA,
        G=1.0,
        softening=1e-4,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=_LEAF,
                leaf_partition="cells",
                leaf_capacity=2048,
            ),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=_ORDER,
    )


def _drifting_system():
    """A uniform cube contracting inside its own box, with nearly no gravity.

    Each coordinate drifts from ``u`` towards ``sign(u) u**2`` (the eight corners
    stay put, so the box does not change); after the 6 steps it is 0.2 of the
    way. The directed far count rises 21510 -> 32632 by s = 0.12, past the cap
    the eager prepare sizes on the cube (32768 = 1.5 x 21510, rounded up to a
    power of two) near s = 0.125: steps 1-3 (s <= 0.1) fit, step 4 (s = 0.133)
    overflows.
    """
    rng = np.random.default_rng(0)
    cube = rng.uniform(-1.0, 1.0, (_N, 3))
    cube[:8] = [[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    target = np.sign(cube) * cube**2
    dt = 1e-2
    vel = 0.2 * (target - cube) / (_STEPS * dt)
    state = jnp.stack(
        [jnp.asarray(cube, jnp.float32), jnp.asarray(vel, jnp.float32)], axis=1
    )
    masses = jnp.full((_N,), 1e-12 / _N, jnp.float32)
    return state, masses, dt


def _run(solver, state, masses, dt, *, donate_state=False, seen=None):
    def callback(step, _state):
        jax.debug.callback(lambda s: seen.append(int(s)), step)

    out, _, history = solver.strict_run_v2(
        state=state,
        masses=masses,
        dt=dt,
        num_steps=_STEPS,
        refresh_every=1,
        leaf_size=_LEAF,
        max_order=_ORDER,
        theta=_THETA,
        return_prepared_state=True,
        return_history=True,
        step_callback=callback if seen is not None else None,
        donate_state=donate_state,
    )
    jax.effects_barrier()
    return (
        np.asarray(jax.block_until_ready(out), np.float64),
        np.asarray(history, np.float64),
    )


def _fallbacks(solver):
    diag = dict(solver.get_runtime_diagnostics() or {})
    return int(diag.get("strict_fused_fallback_count") or 0), diag.get(
        "strict_fused_last_fallback_reason"
    )


@pytest.fixture(scope="module")
def reference():
    """The same rollout with caps that never overflow."""
    mp = pytest.MonkeyPatch()
    for key, val in _FUSED_ENV.items():
        mp.setenv(key, val)
    mp.setenv("JACCPOT_FLAT_WALK_CAP_HEADROOM", "16")
    try:
        state, masses, dt = _drifting_system()
        solver = _solver()
        out, history = _run(solver, state, masses, dt)
        assert _fallbacks(solver)[0] == 0, "the reference must fit throughout"
        return out, history
    finally:
        mp.undo()


@pytest.mark.parametrize("donate", [False, True])
def test_a_mid_run_overflow_is_rerun_from_the_start(reference, donate):
    ref_out, ref_history = reference
    state, masses, dt = _drifting_system()
    solver = _solver()
    seen: list = []
    before = _fallbacks(solver)[0]
    out, history = _run(solver, state, masses, dt, donate_state=donate, seen=seen)
    count, reason = _fallbacks(solver)
    assert count == before + 1 and reason == "capacity_segment_retry"
    failed = solver._impl._strict_particle_failed_step
    assert 0 < failed < _STEPS, f"the overflow should come mid-run, came at {failed}"
    # the lists do not depend on their caps: the re-run is the same run
    assert np.array_equal(out, ref_out)
    assert history.shape == ref_history.shape
    assert np.array_equal(history, ref_history)
    # the first attempt streams the steps before the failure, the re-run all
    want = {s: (2 if s < failed else 1) for s in range(_STEPS)}
    assert dict(collections.Counter(seen)) == want
    assert state.is_deleted() == donate


def test_without_donation_the_input_state_is_kept():
    state, masses, dt = _drifting_system()
    before = np.asarray(state)
    _run(_solver(), state, masses, dt)
    assert not state.is_deleted()
    assert np.array_equal(np.asarray(state), before)


def test_donate_state_needs_the_particle_carry(monkeypatch):
    monkeypatch.setenv("JACCPOT_STRICT_CARRY", "state")
    state, masses, dt = _drifting_system()
    with pytest.raises(ValueError, match="donate_state"):
        _run(_solver(), state, masses, dt, donate_state=True)
