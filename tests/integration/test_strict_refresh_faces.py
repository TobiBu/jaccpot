"""The three refresh faces of the strict lane -- audit item **F33**.

``fmm_strict_run.py`` sat at 55% coverage, and #193 established that a GPU does
not change that: an A100 full-suite run measured 226 missed statements against
CPU's 227. The lane is *reachable* on hardware and still not *exercised*, which is
the same conflation F27 turned out to be -- and F27 went from 0% to 91% on a test,
not a card. This file is that test for the largest untouched block.

It covers ``refresh_prepared_state`` (148 statements missing),
``update_multipoles_only`` (26) and ``rebuild_topology_in_place`` (11). The three
are one implementation behind three names: same signature, same profile guard,
differing only in which diagnostic counter they bump and whether they accept
``bounds``. Their docstrings say exactly that, so the tests assert it rather than
paraphrasing it.

Running on CPU needs the same setup the sibling strict tests use: the large-N
production profile is gated on a GPU backend, so ``jax.default_backend`` is
patched, and the fused static caps are pinned so the first build cannot size an
undersized cap from leaked cross-test state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.config import FarFieldConfig, TreeConfig
from jaccpot.runtime._fmm_impl import FMMEngine
from jaccpot.runtime._large_n_types import LargeNPreparedState

N_PARTICLES = 1024
LEAF_SIZE = 64
MAX_ORDER = 2

# `refresh` reproduces a fresh `prepare_state` to 0.0 relative L2 as measured on
# CPU -- the bound is a tolerance rather than an equality because ARCHITECTURE
# section 10 records that exact-equality assertions need
# `--xla_gpu_deterministic_ops=true` before they mean anything on a GPU, and this
# file is meant to run in the gate too.
RELATIVE_TOLERANCE = 1e-6

FACES = (
    "refresh_prepared_state",
    "update_multipoles_only",
    "rebuild_topology_in_place",
)
COUNTERS = {
    "refresh_prepared_state": "_compiled_profile_refresh_calls",
    "update_multipoles_only": "_compiled_profile_multipoles_only_calls",
    "rebuild_topology_in_place": "_compiled_profile_topology_rebuild_calls",
}


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
        farfield=FarFieldConfig(rotation="solidfmm"),
        theta=0.6,
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=MAX_ORDER,
    )


def _particles(seed: int = 11):
    key = jax.random.PRNGKey(seed)
    key_pos, key_mass, key_move = jax.random.split(key, 3)
    positions = jax.random.uniform(
        key_pos, (N_PARTICLES, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
    )
    masses = jax.random.uniform(
        key_mass, (N_PARTICLES,), minval=0.1, maxval=1.1, dtype=jnp.float32
    )
    moved = positions + 1e-3 * jax.random.normal(
        key_move, positions.shape, dtype=positions.dtype
    )
    return positions, masses, moved


@pytest.mark.slow
@pytest.mark.parametrize("face", FACES)
def test_refresh_face_reproduces_a_fresh_prepare(strict_env, face):
    """Rebinding to moved particles must give what preparing them again gives.

    This is the property the whole lane exists for: the refresh skips the tree
    build, so it is only sound if the state it returns is the state a full
    ``prepare_state`` would have produced.
    """
    fmm = _engine()
    positions, masses, moved = _particles()
    prepared = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    assert isinstance(prepared, LargeNPreparedState)

    refreshed = getattr(fmm, face)(
        prepared, moved, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    fresh = fmm.prepare_state(moved, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER)

    from_refresh = np.asarray(fmm.evaluate_prepared_state(refreshed))
    from_fresh = np.asarray(fmm.evaluate_prepared_state(fresh))
    relative = np.linalg.norm(from_refresh - from_fresh) / np.linalg.norm(from_fresh)
    assert (
        relative <= RELATIVE_TOLERANCE
    ), f"{face} disagrees with a fresh prepare by {relative:.3e} relative L2"


@pytest.mark.slow
@pytest.mark.parametrize("face", FACES)
def test_each_face_bumps_its_own_counter_and_the_refresh_total(strict_env, face):
    """The counters nest; they are not disjoint, and that is worth pinning.

    Measured rather than assumed -- the first version of this test asserted the
    three were mutually exclusive and failed for two of them. They are not:
    ``update_multipoles_only`` and ``rebuild_topology_in_place`` *delegate to*
    ``refresh_prepared_state``, exactly as their docstrings say ("a
    :meth:`refresh_prepared_state` under a different name"), so each bumps its
    own counter **and** the refresh counter, which is therefore a total over all
    three rather than the count of direct calls.

    Anyone reading these diagnostics needs to know that, or they will read
    ``_compiled_profile_refresh_calls`` as the number of plain refreshes and
    over-count it by however many of the other two faces ran.
    """
    fmm = _engine()
    positions, masses, moved = _particles()
    prepared = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )

    before = {name: int(getattr(fmm, attr, 0)) for name, attr in COUNTERS.items()}
    getattr(fmm, face)(
        prepared, moved, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    after = {name: int(getattr(fmm, attr, 0)) for name, attr in COUNTERS.items()}
    delta = {name: after[name] - before[name] for name in COUNTERS}

    assert delta[face] == 1, f"{face} did not bump {COUNTERS[face]}"
    # the refresh counter totals all three, because the other two delegate to it
    assert (
        delta["refresh_prepared_state"] == 1
    ), f"{face} should also register on the refresh total; got {delta}"
    for other in FACES:
        if other not in (face, "refresh_prepared_state"):
            assert (
                delta[other] == 0
            ), f"{face} bumped {COUNTERS[other]}, which belongs to {other}: {delta}"


@pytest.mark.slow
def test_rebuild_topology_in_place_does_not_mutate_its_input(strict_env):
    """ "In place" names the intent, not the mechanism.

    The docstring is explicit that a new state is returned and the input is
    untouched, because the name invites the opposite reading.
    """
    fmm = _engine()
    positions, masses, moved = _particles()
    prepared = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    before = np.asarray(prepared.positions_sorted).copy()

    rebuilt = fmm.rebuild_topology_in_place(
        prepared, moved, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )

    assert rebuilt is not prepared
    np.testing.assert_array_equal(np.asarray(prepared.positions_sorted), before)


@pytest.mark.parametrize("face", FACES)
def test_refresh_faces_reject_a_non_large_n_profile(face):
    """All three are large-N-production only, and say so rather than degrading.

    No ``strict_env`` here on purpose: a default engine is exactly the
    configuration these must refuse. The match is on ``large_n_gpu`` alone
    because these three word the rejection differently from ``strict_run_v2``
    ("supported only for preset='large_n_gpu', tree_type='radix',
    expansion_basis='solidfmm'"), and the test should pin the profile they
    demand, not one phrasing of it.
    """
    fmm = FMMEngine(theta=0.6, working_dtype=jnp.float32, fixed_order=MAX_ORDER)
    positions, masses, moved = _particles()
    prepared = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )

    with pytest.raises(RuntimeError, match="large_n_gpu"):
        getattr(fmm, face)(
            prepared, moved, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
        )


@pytest.mark.slow
def test_refresh_falls_back_to_a_full_prepare_when_topology_cannot_be_reused(
    strict_env,
):
    """The branch a science run hits when the tree stops being reusable.

    ``refresh_prepared_state`` tries ``_refresh_large_n_same_topology`` first and
    calls ``prepare_state`` when it declines. That fallback was unexercised: the
    suite only ever refreshed under conditions the fast path accepts.

    Finding the trigger took measurement. The same-topology key is derived from
    *configuration* and inferred bounds, not from the particle arrangement, so
    none of the obvious candidates decline -- not an affine relocation (which
    preserves the Morton structure), not an independent uniform draw, not a tight
    Gaussian cluster, not explicit wider bounds. Changing ``leaf_size`` does,
    because the leaf size is part of the key.
    """
    fmm = _engine()
    positions, masses, moved = _particles()
    prepared = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )

    misses_before = int(fmm._large_n_same_topology_refresh_misses)
    refreshed = fmm.refresh_prepared_state(
        prepared, moved, masses, leaf_size=LEAF_SIZE // 2, max_order=MAX_ORDER
    )
    misses_after = int(fmm._large_n_same_topology_refresh_misses)

    assert misses_after == misses_before + 1, (
        "the fast path should have declined on the changed leaf size; without "
        "that this test silently exercises the fast path instead of the fallback"
    )

    fresh = fmm.prepare_state(
        moved, masses, leaf_size=LEAF_SIZE // 2, max_order=MAX_ORDER
    )
    from_fallback = np.asarray(fmm.evaluate_prepared_state(refreshed))
    from_fresh = np.asarray(fmm.evaluate_prepared_state(fresh))
    relative = np.linalg.norm(from_fallback - from_fresh) / np.linalg.norm(from_fresh)
    assert (
        relative <= RELATIVE_TOLERANCE
    ), f"the fallback disagrees with a fresh prepare by {relative:.3e} relative L2"


@pytest.mark.slow
def test_refresh_timing_path_returns_the_same_state(strict_env, monkeypatch):
    """The instrumented variant must not be a second implementation.

    ``refresh_prepared_state`` has two bodies: a fast one, and a duplicate under
    ``JACCPOT_REFRESH_TIMING_ENABLE`` that wraps the same calls in
    ``perf_counter`` accounting. The timed body is ~100 statements and was
    entirely unexercised, so nothing checked that it still returns what the
    untimed one returns.
    """
    monkeypatch.setenv("JACCPOT_REFRESH_TIMING_ENABLE", "1")
    timed = _engine()
    assert timed._refresh_timing_enabled, "the env flag did not reach the engine"

    positions, masses, moved = _particles()
    prepared = timed.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    refreshed = timed.refresh_prepared_state(
        prepared, moved, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )

    assert int(timed._refresh_timing_calls) == 1, "the timed body did not run"

    fresh = timed.prepare_state(moved, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER)
    from_timed = np.asarray(timed.evaluate_prepared_state(refreshed))
    from_fresh = np.asarray(timed.evaluate_prepared_state(fresh))
    relative = np.linalg.norm(from_timed - from_fresh) / np.linalg.norm(from_fresh)
    assert (
        relative <= RELATIVE_TOLERANCE
    ), f"the timed refresh disagrees with a fresh prepare by {relative:.3e}"


@pytest.mark.slow
def test_refresh_rejects_a_prepared_state_that_is_not_large_n(strict_env):
    """A large-N engine still refuses a state built by a non-large-N one.

    The profile guard and this one are separate checks: passing the first says
    the *engine* is large-N, not that the *state* is.
    """
    plain = FMMEngine(theta=0.6, working_dtype=jnp.float32, fixed_order=MAX_ORDER)
    positions, masses, moved = _particles()
    plain_state = plain.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    assert not isinstance(plain_state, LargeNPreparedState)

    with pytest.raises(NotImplementedError, match="LargeNPreparedState"):
        _engine().refresh_prepared_state(
            plain_state, moved, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
        )
