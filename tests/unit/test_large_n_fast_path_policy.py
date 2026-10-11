"""Lock-policy tests for the large-N radix fast path."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

import jaccpot.runtime._fmm_impl as fmm_impl_private
import jaccpot.runtime._large_n_nearfield as large_n_nearfield_private
from jaccpot import (
    FastMultipoleMethod,
    FMMAdvancedConfig,
    FMMPreset,
    RuntimePolicyConfig,
)
from jaccpot.config import FarFieldConfig, NearFieldConfig, TreeConfig
from jaccpot.runtime._large_n_nearfield import resolve_large_n_execution_config
from jaccpot.runtime.fmm import FMMEngine


def _make_large_n_fmm():
    return FMMEngine(
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        tree=TreeConfig(tree_type="radix"),
        nearfield=NearFieldConfig(mode="bucketed"),
        working_dtype=jnp.float32,
    )


def test_large_n_fast_lane_defaults_on(monkeypatch):
    monkeypatch.delenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", raising=False)

    cfg = resolve_large_n_execution_config(_make_large_n_fmm(), num_particles=2048)
    assert bool(cfg.radix_fast_lane)
    assert str(cfg.nearfield_mode) == "bucketed"
    assert bool(cfg.retain_leaf_groups)
    assert bool(cfg.precompute_scatter) is False
    # The fast-lane default target block size: 8 -> 32 (static radix refresh path),
    # then 32 -> 4 in the 2026-10 cleanup (D2: Odisseo's value became the default).
    assert int(cfg.target_owned_block_size) == 4
    assert bool(cfg.speed_prepared_layout)


def test_a_zero_target_block_size_means_the_default(monkeypatch):
    """``JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=0`` means "use the default", now 4.

    Was ``test_large_n_fast_lane_legacy_opt_out_env_is_noop``; its other half, the
    legacy ``JACCPOT_LARGE_N_RADIX_FAST_LANE=0`` opt-out that nothing reads, went in
    the 2026-10 cleanup (Z-b).
    """
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "0")

    cfg = resolve_large_n_execution_config(_make_large_n_fmm(), num_particles=2048)
    assert bool(cfg.radix_fast_lane)
    assert int(cfg.target_owned_block_size) == 4


def test_large_n_accel_eval_requires_fast_lane_state(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "8")

    key = jax.random.PRNGKey(7)
    pos_key, mass_key = jax.random.split(key)
    positions = jax.random.uniform(
        pos_key,
        (1024, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        mass_key,
        (1024,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )

    fmm = _make_large_n_fmm()
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=256,
        max_order=4,
    )
    state_no_fast = replace(state, radix_fast_lane=False, radix_fast_payload=None)
    assert bool(getattr(state, "radix_fast_lane", False))
    assert getattr(state.neighbor_list, "neighbor_leaf_positions", None) is None

    with pytest.raises(RuntimeError, match="requires radix fast-lane state"):
        _ = fmm.evaluate_prepared_state(state_no_fast)

    acc = fmm.evaluate_prepared_state(state)
    assert tuple(acc.shape) == (1024, 3)
    state_bad = replace(state, nearfield_mode="baseline")
    with pytest.raises(RuntimeError, match="nearfield_mode='bucketed'"):
        _ = fmm.evaluate_prepared_state(state_bad)


def test_the_large_n_lane_is_gated_on_the_preset_alone(monkeypatch):
    """``preset="large_n_gpu"`` opens the large-N prepare lane; another preset does not.

    The live half of ``test_runtime_path_is_accepted_and_no_longer_selects_the_lane``
    (2026-10 cleanup, Z-a; that test, of the ``runtime_path`` shim, went with the
    shim in Z-b). The GPU predicate is stubbed, as in the lane's own tests.
    """
    from jaccpot import FastMultipoleMethod, FMMPreset
    from jaccpot.runtime import _large_n_pipeline

    monkeypatch.setattr(_large_n_pipeline.jax, "default_backend", lambda: "gpu")
    key_pos, key_mass = jax.random.split(jax.random.PRNGKey(11))
    positions = jax.random.uniform(key_pos, (64, 3), minval=-1.0, maxval=1.0)
    masses = jax.random.uniform(key_mass, (64,), minval=0.5, maxval=1.5)

    fast = FastMultipoleMethod(preset=FMMPreset.FAST, basis="solidfmm")
    assert not _large_n_pipeline.can_use_large_n_prepare_path(
        fast._impl,
        positions_arr=positions,
        masses_arr=masses,
        allow_stateful_cache=True,
    )
    large_n = FastMultipoleMethod(preset=FMMPreset.LARGE_N_GPU, basis="solidfmm")
    assert _large_n_pipeline.can_use_large_n_prepare_path(
        large_n._impl,
        positions_arr=positions,
        masses_arr=masses,
        allow_stateful_cache=True,
    )


# ---------------------------------------------------------------------------
# From test_solver_api.py (2026-10 cleanup, Z-a): the large_n_gpu preset
# ---------------------------------------------------------------------------


def _sample_problem(n: int = 64):
    key = jax.random.PRNGKey(11)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (n,),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.float32,
    )
    return positions, masses


def test_large_n_gpu_preset_applies_memory_safe_gpu_defaults():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.LARGE_N_GPU,
        basis="solidfmm",
    )
    assert fmm._impl.tree_type == "radix"
    assert fmm._impl.tree_build_mode == "static_radix"
    assert fmm._impl.nearfield_mode == "bucketed"
    assert fmm._impl.precompute_nearfield_scatter_schedules is False
    assert fmm._impl.streamed_far_pairs is True
    assert fmm._impl.mixed_order_farfield is False
    assert fmm._impl.m2l_chunk_size is None
    # The large_n_gpu production contract keeps the topology-derived interaction
    # scaffold resident (enable_interaction_cache=True) so fixed-shape refreshes
    # can reuse it. This is memory-neutral on the production path: the stateful
    # interaction cache is gated off for static_radix tree mode, so True here does
    # not add device-memory pressure. (The memory-heavy retain_* knobs stay off.)
    assert fmm._impl.enable_interaction_cache is True
    assert fmm._impl.retain_traversal_result is False
    assert fmm._impl.retain_interactions is False
    assert fmm._impl.memory_objective == "minimum_memory"
    assert fmm._impl.upward_leaf_batch_size == 2048
    assert fmm._impl.mac_type == "dehnen"


def test_large_n_gpu_preset_accepts_string_alias():
    fmm = FastMultipoleMethod(
        preset="large_n_gpu",
        basis="solidfmm",
    )
    assert fmm.preset is FMMPreset.LARGE_N_GPU


def test_large_n_gpu_profile_coerces_conflicting_runtime_knobs():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.LARGE_N_GPU,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            nearfield=NearFieldConfig(
                mode="baseline", precompute_scatter_schedules=True
            ),
            farfield=FarFieldConfig(
                streamed_far_pairs=False,
                mixed_order=True,
                mixed_order_min_order=2,
            ),
            runtime=RuntimePolicyConfig(
                memory_objective="throughput",
                enable_interaction_cache=True,
                retain_traversal_result=True,
                retain_interactions=True,
            ),
        ),
    )
    impl = fmm._impl
    assert impl.memory_objective == "minimum_memory"
    assert impl.nearfield_mode == "bucketed"
    assert impl.precompute_nearfield_scatter_schedules is False
    assert impl.streamed_far_pairs is True
    assert impl.mixed_order_farfield is False
    assert impl.mixed_order_min_order is None
    # Contract keeps the interaction scaffold resident (enable_interaction_cache
    # =True); it is memory-neutral on the static_radix production path (the
    # stateful cache is gated off there). The requested True is honoured, not
    # coerced off, while the memory-heavy retain_* knobs are coerced off below.
    assert impl.enable_interaction_cache is True
    assert impl.retain_traversal_result is False
    assert impl.retain_interactions is False


def test_large_n_gpu_profile_emits_deprecation_warnings_for_conflicting_knobs():
    with pytest.warns(FutureWarning):
        _ = FastMultipoleMethod(
            preset=FMMPreset.LARGE_N_GPU,
            basis="solidfmm",
            advanced=FMMAdvancedConfig(
                nearfield=NearFieldConfig(mode="baseline"),
                farfield=FarFieldConfig(streamed_far_pairs=False),
                runtime=RuntimePolicyConfig(memory_objective="throughput"),
            ),
        )


def test_large_n_compiled_eval_uses_specialized_nearfield(monkeypatch):
    positions, masses = _sample_problem(n=64)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.LARGE_N_GPU,
        basis="solidfmm",
    )
    monkeypatch.setattr(fmm_impl_private.jax, "default_backend", lambda: "gpu")

    state = fmm.prepare_state(positions, masses, leaf_size=16, max_order=3)

    assert state.execution_backend == "large_n"
    assert state.nearfield_mode == "bucketed"
    assert int(state.nearfield_leaf_particle_indices.shape[0]) > 0
    assert state.nearfield_chunk_sort_indices is None

    called = {"specialized": 0}
    original = large_n_nearfield_private.compute_leaf_p2p_accelerations_radix_fast_lane

    def _spy_specialized(*args, **kwargs):
        called["specialized"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        large_n_nearfield_private,
        "compute_leaf_p2p_accelerations_radix_fast_lane",
        _spy_specialized,
    )

    acc = fmm.evaluate_prepared_state(state)

    assert acc.shape == positions.shape
    assert called["specialized"] == 1
