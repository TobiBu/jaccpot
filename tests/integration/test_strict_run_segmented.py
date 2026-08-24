"""``strict_run_segmented`` -- the integrator-agnostic runner. Audit item **F33**.

Lines 487-550 of ``fmm_strict_run.py`` were unexercised: 64 statements, the
second-largest untouched block in the file after the refresh paths. #193
established that a GPU does not reach them either -- an A100 full-suite run
measured 226 missed statements against CPU's 227.

The method owns only the refresh *cadence*: ``num_steps`` is cut into segments of
``refresh_every`` plus a tail, the prepared state is refreshed at each boundary,
and a caller-supplied ``segment_runner`` advances the caller's own state. That
makes it unusually testable for this lane -- the "integrator" can be a spy that
records the segment lengths it was handed, so the cadence can be asserted
directly rather than inferred from a physical result.

The two validation checks run before anything touches a profile, so they need no
GPU-shaped engine; everything else does, via the same setup the sibling strict
tests use.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaccpot.runtime._fmm_impl import FMMEngine

N_PARTICLES = 512
LEAF_SIZE = 64
MAX_ORDER = 2


@pytest.fixture
def strict_env(monkeypatch):
    """Make the large-N production profile reachable on CPU."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")
    monkeypatch.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")


def _engine():
    return FMMEngine(
        preset="large_n_gpu",
        runtime_path="large_n",
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        theta=0.6,
        working_dtype=jnp.float32,
        tree_build_mode="static_radix",
        fixed_order=MAX_ORDER,
    )


def _particles(seed: int = 5):
    key = jax.random.PRNGKey(seed)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (N_PARTICLES, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
    )
    masses = jax.random.uniform(
        key_mass, (N_PARTICLES,), minval=0.1, maxval=1.1, dtype=jnp.float32
    )
    return positions, masses


class _SegmentSpy:
    """A stand-in integrator that records the segment lengths it is given."""

    def __init__(self):
        self.lengths: list[int] = []
        self.acceleration_shapes: list[tuple[int, ...]] = []

    def __call__(self, state, accelerations, num_steps):
        self.lengths.append(int(num_steps))
        self.acceleration_shapes.append(tuple(accelerations.shape))
        # drift a little so each segment sees different positions
        return state + 1e-4, ("segment", int(num_steps))


@pytest.mark.parametrize(
    "num_steps,refresh_every,expected",
    [
        (7, 3, [3, 3, 1]),  # full segments plus a tail
        (6, 3, [3, 3]),  # exact divisor -- no tail segment
        (2, 5, [2]),  # fewer steps than one segment -- tail only
        (1, 1, [1]),  # the smallest legal call
    ],
)
@pytest.mark.slow
def test_steps_are_cut_into_segments_plus_a_tail(
    strict_env, num_steps, refresh_every, expected
):
    """The cadence is the whole contract, so assert it exactly.

    Each case is a distinct branch: with a tail, without one, tail-only when the
    run is shorter than a segment, and the minimal call. A single divisible case
    would leave the tail block -- half the method -- unexercised.
    """
    fmm = _engine()
    positions, masses = _particles()
    spy = _SegmentSpy()

    state_out, prepared_out, history = fmm.strict_run_segmented(
        state=positions,
        masses=masses,
        num_steps=num_steps,
        refresh_every=refresh_every,
        segment_runner=spy,
        positions_getter=lambda s: s,
        leaf_size=LEAF_SIZE,
        max_order=MAX_ORDER,
    )

    assert spy.lengths == expected
    assert prepared_out is not None, "the prepared state must come back for reuse"
    assert state_out.shape == positions.shape
    assert history is None, "history is collected only under collect_history"
    for shape in spy.acceleration_shapes:
        assert shape == (N_PARTICLES, 3)


@pytest.mark.slow
def test_history_is_collected_only_when_asked(strict_env):
    """One entry per segment, in order, and ``None`` otherwise."""
    fmm = _engine()
    positions, masses = _particles()

    _, _, history = fmm.strict_run_segmented(
        state=positions,
        masses=masses,
        num_steps=5,
        refresh_every=2,
        segment_runner=_SegmentSpy(),
        positions_getter=lambda s: s,
        leaf_size=LEAF_SIZE,
        max_order=MAX_ORDER,
        collect_history=True,
    )
    assert history == [("segment", 2), ("segment", 2), ("segment", 1)]


@pytest.mark.slow
def test_rematerialize_fn_runs_once_per_segment(strict_env):
    """The hook exists for donated-buffer re-materialisation between segments."""
    fmm = _engine()
    positions, masses = _particles()
    calls: list[int] = []

    def _remat(state):
        calls.append(1)
        return state

    fmm.strict_run_segmented(
        state=positions,
        masses=masses,
        num_steps=5,
        refresh_every=2,
        segment_runner=_SegmentSpy(),
        positions_getter=lambda s: s,
        leaf_size=LEAF_SIZE,
        max_order=MAX_ORDER,
        rematerialize_fn=_remat,
    )
    assert len(calls) == 3, "two full segments and the tail"


@pytest.mark.slow
def test_positions_getter_sees_the_state_the_runner_returned(strict_env):
    """The refresh must follow the caller's state, not the initial positions.

    If the getter were called once up front, every segment after the first would
    refresh against stale positions -- correct-looking and silently wrong.
    """
    fmm = _engine()
    positions, masses = _particles()
    seen: list[float] = []

    def _getter(state):
        seen.append(float(jnp.asarray(state).sum()))
        return state

    fmm.strict_run_segmented(
        state=positions,
        masses=masses,
        num_steps=4,
        refresh_every=2,
        segment_runner=_SegmentSpy(),
        positions_getter=_getter,
        leaf_size=LEAF_SIZE,
        max_order=MAX_ORDER,
    )
    assert len(seen) == 2
    assert seen[1] > seen[0], "the second segment refreshed against stale positions"


@pytest.mark.parametrize(
    "num_steps,refresh_every,message",
    [
        (0, 1, "num_steps must be positive"),
        (-1, 1, "num_steps must be positive"),
        (1, 0, "refresh_every must be positive"),
        (1, -3, "refresh_every must be positive"),
    ],
)
def test_cadence_arguments_are_validated(num_steps, refresh_every, message):
    """Both checks run before any profile work, so a plain engine reaches them."""
    fmm = FMMEngine(theta=0.6, working_dtype=jnp.float32, fixed_order=MAX_ORDER)
    positions, masses = _particles()
    with pytest.raises(ValueError, match=message):
        fmm.strict_run_segmented(
            state=positions,
            masses=masses,
            num_steps=num_steps,
            refresh_every=refresh_every,
            segment_runner=_SegmentSpy(),
            positions_getter=lambda s: s,
        )
