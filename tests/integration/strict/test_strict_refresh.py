"""The refresh face of the strict lane -- audit item **F33**.

``fmm_strict_run.py`` sat at 55% coverage, and #193 established that a GPU does
not change that: an A100 full-suite run measured 226 missed statements against
CPU's 227. The lane is *reachable* on hardware and still not *exercised*, which is
the same conflation F27 turned out to be -- and F27 went from 0% to 91% on a test,
not a card. This file is that test for the largest untouched block.

It covers ``refresh_prepared_state`` (148 statements missing). It used to cover
``update_multipoles_only`` (26) and ``rebuild_topology_in_place`` (11) too: one
implementation behind three names, differing only in which diagnostic counter
they bumped and whether they accepted ``bounds``. Those two went in the 2026-10
cleanup (X4), and with them their cases here; the tuple below keeps the shape
so the remaining face is parametrised as before.

Running on CPU needs the same setup the sibling strict tests use: the large-N
production profile is gated on a GPU backend, so ``jax.default_backend`` is
patched, and the fused static caps are pinned so the first build cannot size an
undersized cap from leaked cross-test state.

Was ``tests/integration/test_strict_refresh_faces.py``; since the 2026-10 cleanup
(Z-a, rule R5) it also holds the upward-only refresh diagnostic (from
``test_same_topology_refresh_modes.py``) and the static-radix refresh tests from
``tests/integration/test_fmm.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.config import FarFieldConfig, NearFieldConfig, TreeConfig
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

FACES = ("refresh_prepared_state",)
COUNTERS = {
    "refresh_prepared_state": "_compiled_profile_refresh_calls",
}


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

    strict_plans_before = int(fmm._refresh_strict_mode_active_count)
    refreshed = getattr(fmm, face)(
        prepared, moved, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    # the refresh rebuilt its lists through the strict plan, not the general one
    assert int(fmm._refresh_strict_mode_active_count) == strict_plans_before + 1
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
    """One refresh bumps the refresh counter exactly once.

    Until the 2026-10 cleanup (X4) the counters nested: ``update_multipoles_only``
    and ``rebuild_topology_in_place`` delegated to ``refresh_prepared_state``, so
    each bumped its own counter **and** the refresh counter. With those two gone,
    ``_compiled_profile_refresh_calls`` counts plain refreshes again.
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


@pytest.mark.parametrize("face", FACES)
def test_refresh_faces_reject_a_non_large_n_profile(face):
    """The refresh is large-N-production only, and says so rather than degrading.

    No ``strict_env`` here on purpose: a default engine is exactly the
    configuration it must refuse. The match is on ``large_n_gpu`` alone because
    the refresh words the rejection differently from ``strict_run_v2``
    ("supported only for preset='large_n_gpu', tree_type='radix',
    expansion_basis='solidfmm'"), and the test should pin the profile it demands,
    not one phrasing of it.
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


# The upward-only diagnostic short-circuit inside `_refresh_large_n_same_topology`
# (audit F33; was tests/integration/test_same_topology_refresh_modes.py, 2026-10
# cleanup, Z-a, rule R5). It returns after the upward sweep so a profiler can
# attribute stage cost, behind a flag that defaults off; it carries a `dep` term
# built from sums multiplied by 0.0, a data dependency that keeps the compiler from
# eliding the measured work while contributing nothing numerically. (The file's
# compact far-pair reuse cases went in X6; the opt-in now raises,
# test_strict_lane_removed_switches.py.)
UPWARD_ONLY_N_PARTICLES = 512


def _upward_only_particles(seed: int = 3):
    key = jax.random.PRNGKey(seed)
    key_pos, key_mass, key_move = jax.random.split(key, 3)
    positions = jax.random.uniform(
        key_pos,
        (UPWARD_ONLY_N_PARTICLES, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass, (UPWARD_ONLY_N_PARTICLES,), minval=0.1, maxval=1.1, dtype=jnp.float32
    )
    moved = positions + 1e-4 * jax.random.normal(
        key_move, positions.shape, dtype=positions.dtype
    )
    return positions, masses, moved


@pytest.mark.slow
def test_upward_only_diagnostic_returns_after_the_upward_sweep(strict_env, monkeypatch):
    """The stage-attribution mode still returns a usable state.

    ``fused_device_mode=True`` is passed explicitly because
    ``static_fused_refresh`` requires it, and ``refresh_prepared_state`` defaults
    it to ``False``.
    """
    monkeypatch.setenv("JACCPOT_STRICT_REFRESH_DIAG_MODE", "upward_only")
    fmm = _engine()
    assert fmm._strict_refresh_diag_mode == "upward_only"
    assert fmm._strict_refresh_diag_upward_active
    assert not fmm._strict_refresh_diag_downward_active, (
        "upward_only must switch the downward stage off, or it is not the mode "
        "under test"
    )

    positions, masses, moved = _upward_only_particles()
    prepared = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    hits_before = int(fmm._large_n_same_topology_refresh_hits)

    refreshed = fmm.refresh_prepared_state(
        prepared,
        moved,
        masses,
        leaf_size=LEAF_SIZE,
        max_order=MAX_ORDER,
        fused_device_mode=True,
    )

    assert isinstance(refreshed, LargeNPreparedState)
    assert int(fmm._large_n_same_topology_refresh_hits) == hits_before + 1
    assert refreshed.tree.positions_sorted.shape == (UPWARD_ONLY_N_PARTICLES, 3)


# ---------------------------------------------------------------------------
# From tests/integration/test_fmm.py (2026-10 cleanup, Z-a, rule R5): the static-radix refresh
# ---------------------------------------------------------------------------


def test_prepare_refresh_static_radix_tree_preserves_static_shape(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    # This test profiles the strict cap on the fly (there is no pre-recorded
    # profile for its key), so do not require an exact cap-profile match --
    # otherwise it depends on another test having recorded one first (order
    # dependence). Matches the sibling strict-lane tests below.
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")

    key = jax.random.PRNGKey(123)
    core = 0.01 * jax.random.normal(key, (160, 3), dtype=jnp.float32)
    halo = jax.random.uniform(
        key,
        (64, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    positions = jnp.concatenate([core, halo], axis=0)
    masses = jnp.ones((positions.shape[0],), dtype=jnp.float32)

    fmm = FMMEngine(
        preset="large_n_gpu",
        runtime_path="large_n",
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
        softening_kernel="plummer",
    )
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=32,
        max_order=2,
    )
    moved = positions.at[:8].add(jnp.array([1.0e-5, 0.0, 0.0], dtype=jnp.float32))
    refreshed = fmm.refresh_prepared_state(
        state,
        moved,
        masses,
        leaf_size=32,
        max_order=2,
    )
    diagnostics = fmm.get_runtime_diagnostics()

    assert state.tree.build_mode == "static_radix"
    assert refreshed.tree.build_mode == "static_radix"
    assert state.max_leaf_size <= 32
    assert refreshed.max_leaf_size <= 32
    assert refreshed.tree.num_leaves == state.tree.num_leaves
    assert state.tree.parent.shape == refreshed.tree.parent.shape
    assert (
        state.neighbor_list.neighbors.shape == refreshed.neighbor_list.neighbors.shape
    )
    assert diagnostics["large_n_same_topology_refresh_hits"] >= 1
    assert diagnostics["static_radix_refresh_hits"] >= 1


def test_static_radix_refresh_rebuilds_current_large_n_payloads(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    # Profile the strict cap on the fly rather than requiring a pre-recorded
    # profile match (which would make this test depend on run order).
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "4")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "16")

    key = jax.random.PRNGKey(20260507)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (2048, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (2048,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )
    displacement = 0.02 * jnp.stack(
        [
            jnp.sin(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.13),
            jnp.cos(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.17),
            jnp.sin(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.19),
        ],
        axis=1,
    )
    moved = positions + displacement

    kwargs = dict(
        preset="large_n_gpu",
        runtime_path="large_n",
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm", grouped_interactions=False),
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
    )

    fmm = FMMEngine(**kwargs, softening_kernel="plummer")
    state = fmm.prepare_state(positions, masses, leaf_size=128, max_order=2)
    refreshed = fmm.refresh_prepared_state(
        state,
        moved,
        masses,
        leaf_size=128,
        max_order=2,
    )
    diagnostics = fmm.get_runtime_diagnostics()

    fresh_fmm = FMMEngine(**kwargs, softening_kernel="plummer")
    fresh = fresh_fmm.prepare_state(moved, masses, leaf_size=128, max_order=2)

    assert diagnostics["large_n_same_topology_refresh_hits"] >= 1
    assert diagnostics["static_radix_refresh_hits"] >= 1
    assert refreshed.tree.build_mode == "static_radix"
    assert fresh.tree.build_mode == "static_radix"

    refreshed_acc = np.asarray(fmm.evaluate_prepared_state(refreshed))
    fresh_acc = np.asarray(fresh_fmm.evaluate_prepared_state(fresh))
    assert np.allclose(refreshed_acc, fresh_acc, rtol=1e-5, atol=1e-5)

    def assert_array_equal(left, right):
        if left is None or right is None:
            assert left is None and right is None
            return
        assert np.array_equal(np.asarray(left), np.asarray(right))

    assert_array_equal(
        refreshed.nearfield_leaf_particle_indices,
        fresh.nearfield_leaf_particle_indices,
    )
    assert_array_equal(
        refreshed.nearfield_leaf_particle_mask,
        fresh.nearfield_leaf_particle_mask,
    )
    assert_array_equal(
        refreshed.nearfield_target_block_source_leaf_ids_padded,
        fresh.nearfield_target_block_source_leaf_ids_padded,
    )
    assert_array_equal(
        refreshed.nearfield_target_block_valid_mask_padded,
        fresh.nearfield_target_block_valid_mask_padded,
    )
    assert_array_equal(
        refreshed.nearfield_target_block_source_leaf_ids,
        fresh.nearfield_target_block_source_leaf_ids,
    )
    assert_array_equal(
        refreshed.nearfield_target_block_valid_mask,
        fresh.nearfield_target_block_valid_mask,
    )
    assert_array_equal(
        refreshed.nearfield_target_block_offsets,
        fresh.nearfield_target_block_offsets,
    )

    assert refreshed.radix_fast_payload is not None
    assert fresh.radix_fast_payload is not None
    for attr in (
        "target_particle_ids",
        "target_particle_mask",
        "source_leaf_ids",
        "source_leaf_valid_mask",
        "source_particle_ids",
        "source_particle_mask",
    ):
        assert_array_equal(
            getattr(refreshed.radix_fast_payload, attr),
            getattr(fresh.radix_fast_payload, attr),
        )


def test_strict_prepare_refresh_and_evaluate_api_and_diagnostics(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")

    key = jax.random.PRNGKey(20260513)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (1024, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (1024,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )
    moved = positions + 0.01 * jnp.stack(
        [
            jnp.sin(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.11),
            jnp.cos(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.07),
            jnp.sin(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.05),
        ],
        axis=1,
    )

    fmm = FMMEngine(
        preset="large_n_gpu",
        runtime_path="large_n",
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm", grouped_interactions=False),
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
        softening_kernel="plummer",
    )
    state0, acc0 = fmm.strict_prepare_refresh_and_evaluate(
        None,
        positions,
        masses,
        leaf_size=128,
        max_order=2,
        theta=0.6,
    )
    state1, acc1 = fmm.strict_prepare_refresh_and_evaluate(
        state0,
        moved,
        masses,
        leaf_size=128,
        max_order=2,
        theta=0.6,
    )

    assert state0.tree.build_mode == "static_radix"
    assert state1.tree.build_mode == "static_radix"
    assert np.asarray(acc0).shape == (positions.shape[0], 3)
    assert np.asarray(acc1).shape == (positions.shape[0], 3)

    diagnostics = fmm.get_runtime_diagnostics()
    # Two strict_prepare_refresh_and_evaluate calls -> two strict-runner
    # executions (the increment is +1 per call at _fmm_impl.py:3732).
    assert diagnostics["strict_runner_execute_count"] >= 2
    assert diagnostics["strict_runner_compile_count"] >= 1
    assert diagnostics["strict_runner_profile_key_misses"] >= 1
    assert diagnostics["strict_runner_profile_key_hits"] >= 1
