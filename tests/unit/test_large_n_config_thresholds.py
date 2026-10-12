"""Regressions for the radix differentiable-FMM path at large-N config thresholds.

**This module does not exercise ``LargeNPreparedState``.** It was called
``test_large_n_grad_path.py``, which read as though it did. It does not: the large-N
prepare path requires a GPU backend (``can_use_large_n_prepare_path`` returns False when
``jax.default_backend() != "gpu"``), so on CPU every test here runs the ordinary *radix*
grad path at the *configuration thresholds* large N would cross. Measured:
``runtime/_large_n_grad.py`` sits at 0% coverage with these 20 tests passing and none
skipping. The genuinely uncovered large-N reverse path is tracked as F27 in
``docs/refactor_audit_2026-08.md`` and needs a GPU CI leg, not a rename.

Three distinct failures were reachable on the DEFAULT configuration once
``n_particles >= _GPU_LARGE_PARTICLE_THRESHOLD`` (65536) auto-enabled the
grouped/class-major M2L. Two of them -- ``farfield_mode`` left unresolved at
``"auto"`` by the static-fixed-sizing branch of the override resolver, and the
grouped M2L building its classes on the host, which broke every reverse pass --
went with the grouped far field in the 2026-10 cleanup (X3), and the
``farfield_mode`` field itself in Z-b; the resolver check below still pins that the
far field is expanded about COM centres at every N. The third is fp32 far-field
gradients: see
``test_fp32_farfield_gradient_is_finite``.

The resolver assertions are pure host-side policy checks (no tree build), so they
are instant; the gradient tests use a deliberately deep tree at small N to get a
non-empty M2L list cheaply.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.interactions import DualTreeTraversalConfig

import jaccpot.runtime._fmm_impl as fmm_impl_private
import jaccpot.runtime._interaction_cache as interaction_cache_private
import jaccpot.runtime.fmm_constants as fmm_constants_private
from jaccpot import (
    FarFieldConfig,
    FastMultipoleMethod,
    FMMAdvancedConfig,
    FMMPreset,
    NearFieldConfig,
    RuntimePolicyConfig,
)
from jaccpot.runtime.fmm_constants import _GPU_LARGE_PARTICLE_THRESHOLD


def _solver(**kwargs):
    return FastMultipoleMethod(
        basis="real", use_pallas=False, theta=0.6, G=1.0, softening=1e-2, **kwargs
    )


def _clustered(n, dtype=jnp.float64, seed=3):
    """Clustered, flattened (galaxy-like) distribution."""
    rng = np.random.default_rng(seed)
    r = rng.gamma(shape=2.0, scale=1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    z = 0.1 * rng.standard_normal(n)
    pos = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    mass = (np.abs(rng.standard_normal(n)) + 0.5) / n
    return jnp.asarray(pos, dtype), jnp.asarray(mass, dtype)


def _num_far_pairs(state):
    inter = state.interactions
    return int(jnp.sum(inter.counts)) if inter is not None else 0


# --------------------------------------------------------------------------
# 1. Override-resolver policy (host-only, no tree build)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n", [_GPU_LARGE_PARTICLE_THRESHOLD, 4 * _GPU_LARGE_PARTICLE_THRESHOLD]
)
@pytest.mark.parametrize("backend", ["gpu", "cpu"])
def test_the_far_field_is_expanded_about_com_centres_at_large_n(n, backend):
    """The resolver hands the kernels COM expansion centres at every N.

    Was ``test_farfield_mode_never_resolves_to_auto_at_large_n``. Regression: the
    static-fixed-sizing branch (the production default) returned ``farfield_mode=
    "auto"`` once the grouped M2L auto-enabled at ``n >= 65536`` on GPU; the grouped
    far field went in X3 and the always-``"pair_grouped"`` field in Z-b. (This also
    carries the centre assertion of
    ``test_static_sizing_does_not_inherit_the_adaptive_grouped_rewrite``, merged
    here when adaptive sizing went in the 2026-10 cleanup, X4.)
    """
    fmm = _solver()
    overrides = fmm._impl._resolve_runtime_execution_overrides(
        num_particles=n, backend=backend
    )
    assert overrides.center_mode == "com", f"n={n} backend={backend}"


# --------------------------------------------------------------------------
# 2. fp32 gradients (the precision the large-N production path runs in)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize(
    "delta,label",
    [
        ([0.0, 0.0, 0.0], "exact-zero displacement (single-child COM L2L)"),
        ([0.0, 0.0, 0.7], "z-axis aligned (rho == 0)"),
        ([0.3, -0.2, 0.7], "generic control"),
    ],
)
def test_translation_reverse_is_finite_at_degenerate_displacements(dtype, delta, label):
    """The squared-radius floor must survive the working dtype.

    ``jnp.maximum(r2, 1e-60)`` is what makes ``d sqrt/dr2 = 1/(2r)`` finite at the
    degenerate displacements a fixed-topology FMM genuinely hits -- when the floor
    wins, ``jnp.maximum`` routes a zero cotangent. But ``1e-60`` underflows to
    exactly ``0.0`` in float32, which silently disabled the guard: this exact probe
    returned ``[nan, nan, nan]`` in float32 (float64 stayed finite), which is why
    fp32 far-field gradients were all-NaN.

    Probed through the real L2L since the complex basis was removed in the 2026-10
    cleanup (C); it was ``l2l_complex`` before. Both take ``r = sqrt(max(r2,
    floor))`` through :func:`~jaccpot.operators.dtypes.floor_squared_radius`.
    Mutation-checked on the real L2L: with the legacy ``1e-60`` floor in every dtype,
    the float32 exact-zero case returns a non-finite gradient (the z-axis case stays
    finite there), so the retargeted probe still catches the regression.
    """
    from jaccpot.operators.real_harmonics import l2l_real

    order = 4
    ncoeff = (order + 1) ** 2
    local = jnp.asarray(np.arange(1, ncoeff + 1) * 0.1, dtype=dtype)
    d = jnp.asarray(delta, dtype)

    grad = jax.grad(lambda dd: jnp.sum(l2l_real(local, dd, order=order) ** 2))(d)
    assert jnp.all(jnp.isfinite(grad)), f"non-finite L2L reverse at {label}"


def test_squared_radius_floor_preserves_float64_and_is_normal_in_float32():
    """float64 keeps the historical constant (bit-identical); float32 gets a normal."""
    from jaccpot.operators.dtypes import squared_radius_floor

    assert squared_radius_floor(jnp.float64) == 1e-60
    # The bug in one line: the legacy constant is not representable in float32.
    assert np.float32(1e-60) == 0.0
    f32_floor = squared_radius_floor(jnp.float32)
    assert f32_floor >= float(np.finfo(np.float32).tiny)
    assert np.float32(f32_floor) > 0.0


def test_fp32_farfield_gradient_is_finite():
    """fp32 gradients must be finite when the far field is active.

    Regression: with a non-empty M2L list the fp32 reverse pass returned all-NaN
    (float64 was fine, and fp32 with an empty M2L list was fine), so gradients
    were unusable at the precision the large-N production path uses.
    """
    positions, masses = _clustered(512, dtype=jnp.float32)
    fmm = FastMultipoleMethod(
        basis="real",
        use_pallas=False,
        theta=0.6,
        G=1.0,
        softening=1e-2,
        working_dtype=jnp.float32,
    )
    state = fmm.prepare_state(positions, masses, max_order=4, leaf_size=4)
    assert _num_far_pairs(state) > 0, "config must exercise the far field"

    def loss(p):
        return jnp.sum(fmm.differentiable_accelerations(state, p, masses) ** 2)

    grad = jax.grad(loss)(positions)
    assert jnp.all(jnp.isfinite(grad)), "fp32 far-field gradient is non-finite"
    assert float(jnp.linalg.norm(grad)) > 0.0


# ---------------------------------------------------------------------------
# From test_solver_api.py (2026-10 cleanup, Z-a): GPU memory and traversal caps
# ---------------------------------------------------------------------------


def test_large_gpu_minimum_memory_streamed_path_keeps_small_explicit_traversal():
    impl = fmm_impl_private.FMMEngine(
        preset=FMMPreset.LARGE_N_GPU,
        expansion_basis="solidfmm",
        mac_type="engblom",
        farfield=FarFieldConfig(streamed_far_pairs=True),
        runtime_policy=RuntimePolicyConfig(
            memory_objective="minimum_memory",
            fail_fast=True,
            traversal_config=DualTreeTraversalConfig(
                max_pair_queue=32768,
                process_block=64,
                max_interactions_per_node=1024,
                max_neighbors_per_leaf=256,
            ),
        ),
    )

    overrides = impl._resolve_runtime_execution_overrides(
        num_particles=2_097_152,
        backend="gpu",
    )

    assert overrides.traversal_config is not None
    assert int(overrides.traversal_config.max_pair_queue) == 32_768
    assert int(overrides.traversal_config.process_block) == 64
    assert int(overrides.traversal_config.max_interactions_per_node) == 1_024
    assert int(overrides.traversal_config.max_neighbors_per_leaf) == 256


def test_large_gpu_minimum_memory_streamed_tree_guard_caps_overflowing_seed():
    cfg = DualTreeTraversalConfig(
        max_pair_queue=1_048_576,
        process_block=256,
        max_interactions_per_node=32_768,
        max_neighbors_per_leaf=16_384,
    )

    capped = fmm_constants_private._cap_minimum_memory_streamed_gpu_traversal_config_for_tree(
        traversal_config=cfg,
        total_nodes=262_143,
        num_leaves=131_072,
        num_particles=16_777_216,
    )

    assert capped is not None
    assert int(capped.max_pair_queue) == 524_288
    assert int(capped.process_block) == 256
    assert int(capped.max_interactions_per_node) == 8_192
    assert int(capped.max_neighbors_per_leaf) == 4_096


def test_large_gpu_minimum_memory_streamed_tree_guard_keeps_safe_seed():
    cfg = DualTreeTraversalConfig(
        max_pair_queue=1_048_576,
        process_block=256,
        max_interactions_per_node=16_384,
        max_neighbors_per_leaf=8_192,
    )

    kept = fmm_constants_private._cap_minimum_memory_streamed_gpu_traversal_config_for_tree(
        traversal_config=cfg,
        total_nodes=65_535,
        num_leaves=32_768,
        num_particles=2_097_152,
    )

    assert kept == cfg


def test_large_gpu_minimum_memory_streamed_path_clamps_auto_traversal_seed():
    impl = fmm_impl_private.FMMEngine(
        preset=FMMPreset.LARGE_N_GPU,
        expansion_basis="solidfmm",
        mac_type="engblom",
        farfield=FarFieldConfig(streamed_far_pairs=True),
        runtime_policy=RuntimePolicyConfig(
            memory_objective="minimum_memory", fail_fast=True
        ),
    )
    overrides = impl._resolve_runtime_execution_overrides(
        num_particles=2_097_152,
        backend="gpu",
    )

    assert overrides.traversal_config is not None
    assert int(overrides.traversal_config.max_pair_queue) == 262_144
    assert int(overrides.traversal_config.process_block) == 256
    assert int(overrides.traversal_config.max_interactions_per_node) == 8_192
    assert int(overrides.traversal_config.max_neighbors_per_leaf) == 4_096


def test_large_gpu_minimum_memory_streamed_seed_scales_for_xl_particle_counts():
    impl = fmm_impl_private.FMMEngine(
        preset=FMMPreset.LARGE_N_GPU,
        expansion_basis="solidfmm",
        runtime_policy=RuntimePolicyConfig(
            memory_objective="minimum_memory", fail_fast=True
        ),
    )
    overrides = impl._resolve_runtime_execution_overrides(
        num_particles=4_194_304,
        backend="gpu",
    )

    assert overrides.traversal_config is not None
    assert int(overrides.traversal_config.max_pair_queue) == 524_288
    assert int(overrides.traversal_config.process_block) == 256
    assert int(overrides.traversal_config.max_interactions_per_node) == 8_192
    assert int(overrides.traversal_config.max_neighbors_per_leaf) == 4_096


def test_prepare_bucketed_scatter_schedules_skips_int32_overflow_shape():
    impl = fmm_impl_private.FMMEngine(
        preset=FMMPreset.LARGE_N_GPU,
        expansion_basis="solidfmm",
        runtime_policy=RuntimePolicyConfig(memory_objective="minimum_memory"),
    )
    nearfield_interop = fmm_impl_private.NearfieldInteropData(
        leaf_nodes=jnp.zeros((0,), dtype=jnp.int32),
        node_ranges=jnp.zeros((0, 2), dtype=jnp.int32),
        offsets=jnp.zeros((0,), dtype=jnp.int32),
        neighbors=jnp.zeros((0,), dtype=jnp.int32),
        counts=jnp.zeros((0,), dtype=jnp.int32),
        particle_order_node_ranges=jnp.zeros((0, 2), dtype=jnp.int32),
        particle_order_leaf_indices=jnp.zeros((0,), dtype=jnp.int32),
        particle_order_to_native_leaf=jnp.zeros((0,), dtype=jnp.int32),
    )

    target_leaf_ids = jnp.zeros((67_108_608,), dtype=jnp.int32)
    valid_pairs = jnp.ones((67_108_608,), dtype=bool)

    out = impl._prepare_bucketed_scatter_schedules_safe(
        nearfield_interop=nearfield_interop,
        target_leaf_ids=target_leaf_ids,
        valid_pairs=valid_pairs,
        leaf_cap=128,
        edge_chunk_size=128,
    )

    assert out == (None, None, None)


def test_large_gpu_minimum_memory_nearfield_prepare_skips_pair_vector_precompute():
    impl = fmm_impl_private.FMMEngine(
        preset=FMMPreset.LARGE_N_GPU,
        expansion_basis="solidfmm",
        runtime_policy=RuntimePolicyConfig(memory_objective="minimum_memory"),
        nearfield=NearFieldConfig(mode="bucketed", precompute_scatter_schedules=False),
    )
    nearfield_interop = fmm_impl_private.NearfieldInteropData(
        leaf_nodes=jnp.zeros((1,), dtype=jnp.int32),
        node_ranges=jnp.zeros((1, 2), dtype=jnp.int32),
        offsets=jnp.zeros((2,), dtype=jnp.int32),
        neighbors=jnp.zeros((0,), dtype=jnp.int32),
        counts=jnp.zeros((1,), dtype=jnp.int32),
        particle_order_node_ranges=jnp.zeros((1, 2), dtype=jnp.int32),
        particle_order_leaf_indices=jnp.zeros((1,), dtype=jnp.int32),
        particle_order_to_native_leaf=jnp.zeros((1,), dtype=jnp.int32),
    )

    # `neighbor_list=None` used to be passed here with the comment "unused on the
    # short-circuit path". It was unused on EVERY path -- 0 references in the
    # method's 112 lines -- so the parameter is gone rather than widened to
    # `Optional[...]` to accommodate a value nothing read (F40).
    out = impl._prepare_nearfield_precompute_artifacts(
        nearfield_interop=nearfield_interop,
        leaf_cap=128,
        num_particles=2_097_152,
        nearfield_mode="bucketed",
        nearfield_edge_chunk_size=128,
        retain_pair_vectors=False,
    )

    assert out.target_leaf_ids is None
    assert out.source_leaf_ids is None
    assert out.valid_pairs is None
    assert out.chunk_sort_indices is None
    assert out.chunk_group_ids is None
    assert out.chunk_unique_indices is None


def test_gpu_runtime_overrides_cap_traversal_capacities_for_large_n():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    overrides = fmm._impl._resolve_runtime_execution_overrides(
        num_particles=131072,
        backend="gpu",
    )
    cfg = overrides.traversal_config
    assert cfg is not None
    assert int(cfg.max_neighbors_per_leaf) == 2048
    assert int(cfg.max_interactions_per_node) == 8192
    assert int(cfg.max_pair_queue) >= 131072


def test_large_gpu_runs_skip_nearfield_scatter_precompute(monkeypatch):
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    monkeypatch.setattr(fmm_impl_private.jax, "default_backend", lambda: "gpu")
    assert (
        fmm._impl._should_precompute_nearfield_scatter_schedules(num_particles=131072)
        is False
    )


def test_minimum_memory_gpu_runtime_starts_with_smaller_traversal_capacities():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.LARGE_N_GPU,
        basis="solidfmm",
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax, "default_backend", lambda: "gpu")
        overrides = fmm._impl._resolve_runtime_execution_overrides(
            num_particles=524288,
        )

    assert overrides.traversal_config is not None
    # A *bounded* pair queue is the load-bearing assertion, and it now scales with
    # the particle count rather than being one constant for every N below a
    # million. At N=524288 that is 131072 (N/4 rounded to a power of two); the
    # historical flat 32768 remains the floor and is what N <= 131072 still gets,
    # asserted in tests/unit/runtime/test_capacity_replanning.py. The constant was
    # a hard ceiling in the middle of the measured range: N=262144 raised
    # "Pair queue capacity exceeded" on an A100 with 29.62 GiB free and the
    # largest static buffer at 0.31 GiB.
    assert int(overrides.traversal_config.max_pair_queue) == 131072
    assert int(overrides.traversal_config.max_pair_queue) >= 32768
    # process_block is floored to the streamed minimum-memory ceiling (256, i.e.
    # _GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_PROCESS_BLOCK) to avoid underfilled
    # count-pass kernels -- matching the sibling auto-seed tests above. (The
    # original 64 predates that process-block floor.) process_block is a compute
    # chunk size, not a buffer dimension, so 256 is not a memory regression.
    assert int(overrides.traversal_config.process_block) == 256
    assert int(overrides.traversal_config.max_interactions_per_node) == 1024
    assert int(overrides.traversal_config.max_neighbors_per_leaf) == 256


def test_tracing_traversal_config_caps_queue_and_process_block():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )

    capped = fmm._impl._resolve_tracing_traversal_config(
        traversal_config=DualTreeTraversalConfig(
            max_pair_queue=524288,
            process_block=512,
            max_interactions_per_node=8192,
            max_neighbors_per_leaf=2048,
        )
    )

    assert capped is not None
    assert int(capped.max_pair_queue) == 65536
    assert int(capped.process_block) == 128
    assert int(capped.max_interactions_per_node) == 512
    assert int(capped.max_neighbors_per_leaf) == 512


def test_capacity_retry_traversal_settings_clamp_interaction_growth():
    next_cfg, next_queue, next_block = (
        interaction_cache_private._next_retry_traversal_settings(
            traversal_config=DualTreeTraversalConfig(
                max_pair_queue=2_097_152,
                process_block=2048,
                max_interactions_per_node=16384,
                max_neighbors_per_leaf=4096,
            ),
            max_pair_queue=None,
            pair_process_block=None,
        )
    )

    assert next_queue == 4_194_304
    assert next_block == 4096
    assert int(next_cfg.max_interactions_per_node) == 16_384


def test_minimum_memory_objective_reduces_default_nearfield_schedule_cap():
    balanced = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(memory_objective="balanced"),
        ),
    )
    minimum = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(memory_objective="minimum_memory"),
        ),
    )

    balanced_cap = balanced._impl._resolve_nearfield_schedule_item_cap(
        edge_count=4096,
        leaf_cap=32,
        edge_chunk_size=256,
    )
    minimum_cap = minimum._impl._resolve_nearfield_schedule_item_cap(
        edge_count=4096,
        leaf_cap=32,
        edge_chunk_size=256,
    )

    assert minimum_cap < balanced_cap


def test_memory_budget_limits_default_nearfield_schedule_cap():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(memory_budget_bytes=96),
        ),
    )

    cap = fmm._impl._resolve_nearfield_schedule_item_cap(
        edge_count=1024,
        leaf_cap=32,
        edge_chunk_size=256,
    )

    assert cap == 8
