"""A potential request must not fall off the large-N near-field fast lane.

``evaluate_large_n_nearfield_fast_lane`` used to admit the fused Pallas potential
lane only when the prepared state carried neither an overflow payload nor a
target-block payload. The target-block half of that guard was stale: the
prepacked Pallas lane reads the potential out of the fused kernel's fourth
output component alongside the three acceleration components, so it accumulates a
potential for target blocks just as it does for plain leaf pairs.

Target blocks are on by default (``target_owned_block_size`` resolves to 32), so
the stale guard fired on essentially every large-N configuration and delegated to
the generic bucketed path. Measured cost of that fallback, A100, ``large_n_gpu``,
leaf 128, p=3, theta=0.77: a potential at N=881744 took 2.32 s against 240 ms for
the acceleration that contains it, and scaled as N^1.32 against the
acceleration's N^0.64 (bench/results/scaling/wallclock_vs_n.json).

The overflow half of the guard is *not* stale and is retained: on the
acceleration branch the overflow payload's contribution is added by a separate
``compute_leaf_p2p_accelerations_radix_payload_pairs_only`` call, and the
potential branch returns before reaching it, so admitting an overflow payload
here would silently return a potential missing those pairs.

These run on CPU by forcing the Pallas kernel's interpret mode, the same way
``tests/unit/operators/test_pallas_nearfield_fused.py`` does; the GPU gate on the
prepare path is stubbed the way the sibling large-N tests stub it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.config import FarFieldConfig
from jaccpot.pallas.nearfield_fused_leaf import pallas_nearfield_fused_supported
from jaccpot.runtime import _large_n_nearfield
from jaccpot.runtime._large_n_nearfield import evaluate_large_n_nearfield_fast_lane
from jaccpot.runtime._large_n_types import LargeNPreparedState
from jaccpot.runtime.fmm import FMMEngine

# Compile-bound: a full large-N prepare plus several near-field evaluations.
#
# Gated on the fused kernel's hardware because the lane this module pins is
# unreachable without it: off Ampere, `pallas_nearfield_fused_supported()` is
# False, the potential branch delegates to the generic path regardless of the
# guards, and the premise these tests assert (a target-block payload present and
# the fused lane taken) cannot be established. Interpret mode reaches the kernel
# but not through `_large_n_nearfield`, which gates on hardware support rather
# than on interpret the way `_fast_lane.py` does. Running these on CPU would
# assert against the wrong lane, so they skip rather than pass vacuously.
# See docs/potential_falls_off_the_fast_lane.md.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not pallas_nearfield_fused_supported(),
        reason=(
            "the large-N near-field potential lane requires an Ampere+ (sm_80+) "
            "GPU; see docs/potential_falls_off_the_fast_lane.md"
        ),
    ),
]

N = 256
LEAF = 32
ORDER = 2


@pytest.fixture(scope="module")
def engine_and_state(request: pytest.FixtureRequest):
    """Build a real large-N prepared state carrying a target-block payload.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Used to register the backend stub's undo. Module-scoped, so the
        function-scoped ``monkeypatch`` fixture is unavailable and one is driven
        by hand.

    Returns
    -------
    tuple[FMMEngine, LargeNPreparedState]
        The engine and a prepared state from the real large-N path, not a double.
    """
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    rng = np.random.default_rng(0)
    positions = jnp.asarray(rng.uniform(-1.0, 1.0, size=(N, 3)), dtype=jnp.float32)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(N,)), dtype=jnp.float32)

    engine = FMMEngine(
        preset="large_n_gpu",
        runtime_path="large_n",
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        fixed_order=ORDER,
    )
    state = engine.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)
    assert isinstance(state, LargeNPreparedState), (
        "the large-N prepare path declined; this test would otherwise assert "
        "against a different lane than the one it means to pin"
    )
    return engine, state


def _has_target_blocks(state: LargeNPreparedState) -> bool:
    """Report whether the state carries a target-block payload.

    Parameters
    ----------
    state : LargeNPreparedState
        Prepared large-N state to inspect.

    Returns
    -------
    bool
        True when the target-block source-leaf ids are present and non-empty.
    """
    ids = state.nearfield_target_block_source_leaf_ids
    return ids is not None and int(ids.size) > 0


def test_the_state_under_test_actually_carries_target_blocks(engine_and_state):
    """Guard the premise: without target blocks the lane test is vacuous.

    Parameters
    ----------
    engine_and_state : tuple[FMMEngine, LargeNPreparedState]
        Engine and prepared state fixture.
    """
    _, state = engine_and_state
    assert _has_target_blocks(state), (
        "this state has no target-block payload, so it cannot exercise the guard "
        "this module exists to pin -- raise N or lower the leaf size"
    )
    assert getattr(state, "radix_overflow_payload", None) is None, (
        "this state carries an overflow payload, whose potential genuinely is "
        "not accumulated on the fast lane; the guard is correct to refuse it"
    )


def test_potential_matches_the_generic_path(engine_and_state, monkeypatch):
    """The fast-lane potential agrees with the generic near-field potential.

    Parameters
    ----------
    engine_and_state : tuple[FMMEngine, LargeNPreparedState]
        Engine and prepared state fixture.
    monkeypatch : pytest.MonkeyPatch
        Used to force the Pallas interpret mode on CPU.
    """
    engine, state = engine_and_state

    ref_acc, ref_pot = evaluate_large_n_nearfield_fast_lane(
        engine, state, return_potential=True
    )

    monkeypatch.setenv("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1")
    monkeypatch.setattr(engine, "use_pallas", True, raising=False)
    acc, pot = evaluate_large_n_nearfield_fast_lane(
        engine, state, return_potential=True
    )

    np.testing.assert_allclose(
        np.asarray(acc), np.asarray(ref_acc), rtol=2e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(pot), np.asarray(ref_pot), rtol=2e-5, atol=1e-6
    )


def test_potential_does_not_delegate_to_the_generic_path(engine_and_state, monkeypatch):
    """With target blocks present, the potential stays on the fused lane.

    Asserting the lane rather than only the value matters: the delegation target
    *is* the reference, so a value-only test passes vacuously while the fallback
    is still being taken.

    Parameters
    ----------
    engine_and_state : tuple[FMMEngine, LargeNPreparedState]
        Engine and prepared state fixture.
    monkeypatch : pytest.MonkeyPatch
        Used to force interpret mode and to trip-wire the generic path.
    """
    engine, state = engine_and_state
    assert _has_target_blocks(state)

    monkeypatch.setenv("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1")
    monkeypatch.setattr(engine, "use_pallas", True, raising=False)

    def _tripwire(*args, **kwargs):
        raise AssertionError(
            "the potential request delegated to the generic near-field path "
            "even though the fused prepacked lane accumulates a potential for "
            "target blocks"
        )

    monkeypatch.setattr(_large_n_nearfield, "compute_leaf_p2p_accelerations", _tripwire)

    acc, pot = evaluate_large_n_nearfield_fast_lane(
        engine, state, return_potential=True
    )
    assert acc.shape == (N, 3)
    assert pot.shape == (N,)
    assert bool(jnp.all(jnp.isfinite(pot)))
    # A near-field potential from attracting masses is negative where it is
    # nonzero at all, which a transposed or mis-scattered output would break.
    assert float(jnp.min(pot)) < 0.0
