"""
Fast Multipole Method (FMM) for computing gravitational accelerations.

This implementation uses multipole and local expansions to compute
gravitational forces in O(N) time instead of O(N^2) for direct summation.
"""

import hashlib
import json
import math
import os
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from math import comb
from typing import Any, Literal, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Callable, Tuple
from jaxtyping import Array, DTypeLike, jaxtyped
from yggdrax import build_tree
from yggdrax.dense_interactions import DenseInteractionBuffers
from yggdrax.geometry import compute_tree_geometry
from yggdrax.grouped_interactions import GroupedInteractionBuffers
from yggdrax.interactions import (
    CompactTaggedFarPairs,
    CompactTaggedOctreeFarPairs,
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    DualTreeWalkResult,
    MACType,
    NodeInteractionList,
    NodeNeighborList,
    OctreeNativeNeighborList,
    build_interactions_and_neighbors,
    build_octree_native_far_pairs,
    build_octree_native_neighbor_lists,
    build_well_separated_interactions,
)
from yggdrax.morton import morton_encode
from yggdrax.tree import (
    RadixTree,
    Tree,
    TreeType,
    available_tree_types,
    get_node_levels,
    rebuild_static_radix_tree_from_template,
    reorder_particles_by_indices,
)
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.basis.real_sh import complex_to_real_coeffs
from jaccpot.config import FMMExecutionBackend, MemoryObjective
from jaccpot.downward.local_expansions import (
    LocalExpansionData,
    TreeDownwardData,
    initialize_local_expansions,
)
from jaccpot.downward.local_expansions import (
    prepare_downward_sweep as prepare_tree_downward_sweep,
)
from jaccpot.downward.local_expansions import (
    run_downward_sweep as run_tree_downward_sweep,
)
from jaccpot.downward.local_expansions import translate_local_expansion
from jaccpot.nearfield.near_field import (
    compute_leaf_p2p_accelerations,
    compute_leaf_p2p_accelerations_large_n_accel_only,
    prepare_bucketed_scatter_schedules,
    prepare_bucketed_scatter_schedules_from_groups,
    prepare_leaf_neighbor_pairs,
)
from jaccpot.operators.complex_ops import (
    complex_rotation_blocks_from_z_solidfmm_batch,
    complex_rotation_blocks_to_z_solidfmm_batch,
    enforce_conjugate_symmetry_batch,
    evaluate_local_complex_derivative_tower_batch,
    evaluate_local_complex_grad_analytic,
    evaluate_local_complex_grad_analytic_batch,
    evaluate_local_complex_grad_analytic_preserve_dtype,
    evaluate_local_complex_grad_order4_unrolled,
    evaluate_local_complex_with_grad,
    evaluate_local_complex_with_grad_analytic_batch,
    evaluate_local_complex_with_grad_batch,
    l2l_complex,
    l2l_complex_batch,
    m2l_complex_reference,
    m2l_complex_reference_batch,
    m2l_complex_reference_batch_cached_blocks,
)
from jaccpot.operators.m2l_real_rot_scale import (
    m2l_rot_scale_real_batch,
    m2l_rot_scale_real_batch_cached_blocks,
    real_rotation_blocks_from_z_local_batch,
    real_rotation_blocks_to_z_multipole_batch,
)
from jaccpot.operators.multipole_utils import (
    LOCAL_LEVEL_COMBOS,
    MAX_MULTIPOLE_ORDER,
    level_offset,
    total_coefficients,
)
from jaccpot.operators.real_harmonics import (
    evaluate_local_real_derivative_tower_batch,
    evaluate_local_real_with_grad,
    l2l_real,
    sh_size,
)
from jaccpot.operators.symmetric_tensors import (
    component_lift_index_map_3d,
    contract_symmetric_one_axis_3d,
)
from jaccpot.upward.real_tree_expansions import (
    prepare_real_upward_sweep,
)
from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_source_motion_multipoles,
    prepare_solidfmm_complex_upward_sweep,
)
from jaccpot.upward.tree_expansions import (
    NodeMultipoleData,
    TreeUpwardData,
)
from jaccpot.upward.tree_expansions import (
    prepare_upward_sweep as prepare_tree_upward_sweep,
)

from ._adaptive_policy import (
    adaptive_pair_policy,
    adaptive_policy_tolerance,
    bucket_far_pairs_by_tag,
    build_adaptive_policy_state,
    compute_node_force_scale_from_sorted_acc,
    source_error_proxy_by_order_from_multipoles,
)
from ._interaction_cache import (
    _build_dual_tree_artifacts,
    _compiled_refresh_dual_planner_route,
    _DualTreeArtifacts,
    _interaction_cache_key,
    _InteractionCacheEntry,
    _RefreshDualPlannerHint,
)
from ._large_n_pipeline import (
    can_use_large_n_prepare_path,
    evaluate_large_n_state,
    prepare_large_n_state,
)
from ._large_n_types import LargeNPreparedState
from ._nearfield_cache import (
    NearfieldPrecomputeArtifacts,
    nearfield_cache_matches,
    nearfield_from_cache,
    with_nearfield_cache_artifacts,
)
from ._octree_adapter import (
    OctreeExecutionData,
    build_octree_execution_data,
    build_octree_execution_data_with_status,
)
from ._octree_fmm import (
    OctreeSolidFMMComplexMultipoles,
    OctreeSolidFMMDownwardPlan,
    accumulate_octree_solidfmm_m2l,
    build_octree_downward_plan,
    build_octree_interaction_plan,
    build_octree_interaction_plan_from_native_pairs,
    build_octree_upward_plan,
    prepare_octree_solidfmm_complex_multipoles,
    propagate_octree_solidfmm_l2l,
)
from .dtypes import INDEX_DTYPE, as_index, complex_dtype_for_real
from .fmm_autotune import AutotuneMixin
from .fmm_caches import (
    _GPU_M2L_AUTOTUNE_LARGE_CANDIDATES,
    _GPU_M2L_AUTOTUNE_MAX_SAMPLE_NODES,
    _GPU_M2L_AUTOTUNE_MAX_SAMPLE_PAIRS,
    _GPU_M2L_AUTOTUNE_MEDIUM_CANDIDATES,
    _GPU_M2L_AUTOTUNE_PAIR_BINS,
    _GPU_M2L_AUTOTUNE_SMALL_CANDIDATES,
    _GPU_M2L_AUTOTUNE_XL_CANDIDATES,
    _M2L_FULLBATCH_MAX_PAIRS,
    _clear_global_runtime_caches,
    _contains_tracer,
    _estimate_payload_nbytes,
    _format_nbytes,
    _grouped_operator_cache_get,
    _grouped_operator_cache_key,
    _grouped_operator_cache_put,
    _grouped_segment_cache_get,
    _grouped_segment_cache_key,
    _grouped_segment_cache_put,
    _m2l_autotune_lookup,
    _m2l_autotune_payload,
    _m2l_autotune_store,
    _restore_m2l_autotune_payload,
)
from .fmm_constants import (
    _CLASS_MAJOR_CPU_PARTICLE_THRESHOLD,
    _GPU_LARGE_PARTICLE_THRESHOLD,
    _GPU_MAX_INTERACTIONS_PER_NODE,
    _GPU_MAX_NEIGHBORS_PER_LEAF,
    _GPU_MIN_INTERACTIONS_PER_NODE,
    _GPU_MIN_NEIGHBORS_PER_LEAF,
    _GPU_MIN_PAIR_QUEUE_LARGE,
    _GPU_MIN_PAIR_QUEUE_MEDIUM,
    _GPU_MIN_PAIR_QUEUE_XL,
    _GPU_MINIMUM_MEMORY_INTERACTIONS_PER_NODE,
    _GPU_MINIMUM_MEMORY_NEIGHBORS_PER_LEAF,
    _GPU_MINIMUM_MEMORY_PAIR_QUEUE,
    _GPU_MINIMUM_MEMORY_PROCESS_BLOCK,
    _GROUPED_SCHEDULE_BUDGET_DEFAULT,
    _KDTREE_DEFAULT_TRAVERSAL_CONFIG,
    _LARGE_CPU_M2L_CHUNK_SIZE,
    _LARGE_CPU_PARTICLE_THRESHOLD,
    _LARGE_CPU_TRAVERSAL_CONFIG,
    _MINIMUM_MEMORY_CPU_M2L_CHUNK_SIZE,
    _MINIMUM_MEMORY_GPU_M2L_CHUNK_SIZE,
    _NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_LARGE,
    _NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_MEDIUM,
    _NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_XL,
    _NEARFIELD_BUCKETED_CPU_PARTICLE_THRESHOLD,
    _NEARFIELD_GPU_PRECOMPUTE_MAX_PARTICLES,
    _NEARFIELD_SCATTER_SCHEDULE_INT32_ITEM_LIMIT,
    _NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP,
    _NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP_GPU,
    _TRACING_MAX_INTERACTIONS_PER_NODE,
    _TRACING_MAX_NEIGHBORS_PER_LEAF,
    _TRACING_MAX_PAIR_QUEUE,
    _TRACING_MAX_PROCESS_BLOCK,
    _cap_minimum_memory_streamed_gpu_traversal_config_for_tree,
    _env_int,
    _minimum_memory_streamed_gpu_traversal_ceiling,
    _minimum_memory_streamed_gpu_traversal_seed,
    _prepare_diag,
)
from .fmm_derivatives import DerivativesMixin
from .fmm_diagnostics import DiagnosticsMixin
from .fmm_overrides import OverridesMixin
from .fmm_policy import PolicyMixin
from .fmm_presets import FMMPreset, FMMPresetConfig, get_preset_config
from .fmm_state import (
    FMMPreparedState,
    TreeBuilderConfig,
    _bucket_far_pairs_by_level_split,
    _build_octree_downward_artifacts,
    _build_octree_upward_artifacts,
    _build_tree_with_config,
    _empty_interaction_storage_like,
    _finalize_octree_downward_artifacts,
    _GeometryReuseEntry,
    _normalize_strict_refresh_diag_mode,
    _octree_farfield_eval_inputs,
    _prepared_state_octree_upward_payload,
    _prepared_state_upward_payload,
    _PrepareStateDualDownwardArtifacts,
    _PrepareStateFarPairPlan,
    _PrepareStateTreeUpwardArtifacts,
    _resolve_fmm_config,
    _RuntimeExecutionOverrides,
    _strict_refresh_diag_stage_flags,
    _TopologyReuseCandidate,
    _TopologyReuseEntry,
    _TreeBuildArtifacts,
    _velocity_verlet_state_update,
)
from .fmm_strict_cap_profile import StrictCapProfileMixin
from .fmm_sweeps import SweepsMixin
from .kernels.core import (
    ExpansionBasis,
    NearfieldInteropData,
    PackedAccelerationDerivatives,
    _accumulate_real_m2l_chunked_scan,
    _accumulate_real_m2l_chunked_scan_pallas,
    _accumulate_solidfmm_m2l_chunked_scan,
    _build_nearfield_interop_data,
    _build_target_nearfield_source_index_matrix,
    _compute_targeted_nearfield,
    _empty_interaction_storage_for_tree,
    _evaluate_local_expansions_for_particles,
    _evaluate_local_expansions_for_target_particles,
    _evaluate_prepared_tree,
    _evaluate_prepared_tree_targets,
    _evaluate_tree_compiled_impl,
    _FarPairCOO,
    _infer_bounds,
    _infer_order_from_coeff_count,
    _map_targets_to_leaf_positions,
    _max_leaf_size_from_tree,
    _normalize_strict_refresh_detail_diag_mode,
    _prepare_solidfmm_downward_sweep,
    _prepare_tree_evaluation_inputs,
)
from .reference import MultipoleExpansion
from .reference import compute_expansion as reference_compute_expansion
from .reference import compute_gravitational_potential as reference_compute_potential
from .reference import direct_sum as reference_direct_sum
from .reference import evaluate_expansion as reference_evaluate_expansion

FarFieldMode = Literal["auto", "pair_grouped", "class_major"]
NearFieldMode = Literal["auto", "baseline", "bucketed"]
JerkMode = Literal["fast_approx", "accurate"]
PreparedStateLike = Union["FMMPreparedState", LargeNPreparedState]


class FastMultipoleMethod(
    SweepsMixin,
    OverridesMixin,
    AutotuneMixin,
    PolicyMixin,
    DerivativesMixin,
    StrictCapProfileMixin,
    DiagnosticsMixin,
):
    """
    Fast Multipole Method for gravitational N-body simulations.

    Args:
        theta: Opening angle criterion (typically 0.5-1.0)
        G: Gravitational constant (default: 1.0)
        softening: Softening length to avoid singularities (default: 0.0)
        tree_build_mode:
            Choose between "lbvh" and "fixed_depth" builders.
        tree_type:
            Yggdrax tree family selector (e.g. ``"radix"`` or ``"kdtree"``).
        target_leaf_particles:
            Desired particle count per leaf for fixed-depth trees.
        refine_local:
            Enable host-side refinement of elongated leaves.
        max_refine_levels:
            Maximum refinement depth for the local refinement pass.
        aspect_threshold:
            Aspect ratio threshold that triggers extra splits.
    """

    def __init__(
        self: "FastMultipoleMethod",
        theta: float = 0.5,
        G: float = 1.0,
        softening: float = 1e-12,
        working_dtype: Optional[DTypeLike] = None,
        *,
        expansion_basis: ExpansionBasis = "cartesian",
        basis_impl: Optional[Any] = None,
        m2l_impl: Optional[str] = None,
        adaptive_order: bool = False,
        p_gears: Optional[tuple[int, ...]] = None,
        use_pallas: Optional[bool] = None,
        reuse_topology: bool = False,
        rebuild_every: int = 1,
        mac_force_scale_mode: str = "prev",
        adaptive_error_model: str = "tail_proxy",
        adaptive_eps: Optional[float] = None,
        dehnen_geometry_mode: str = "tree",
        mac_type: MACType = "bh",
        complex_rotation: str = "solidfmm",  # "cached",
        dehnen_radius_scale: float = 1.0,
        m2l_chunk_size: Optional[int] = None,
        l2l_chunk_size: Optional[int] = None,
        max_pair_queue: Optional[int] = None,
        pair_process_block: Optional[int] = None,
        traversal_config: Optional[DualTreeTraversalConfig] = None,
        tree_build_mode: Optional[str] = None,
        tree_type: str = "radix",
        target_leaf_particles: Optional[int] = None,
        refine_local: Optional[bool] = None,
        max_refine_levels: Optional[int] = None,
        aspect_threshold: Optional[float] = None,
        interaction_retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
        use_dense_interactions: Optional[bool] = None,
        grouped_interactions: Optional[bool] = None,
        farfield_mode: FarFieldMode = "auto",
        streamed_far_pairs: Optional[bool] = None,
        mixed_order_farfield: bool = False,
        mixed_order_min_order: Optional[int] = None,
        nearfield_mode: NearFieldMode = "auto",
        runtime_path: Literal["auto", "legacy", "large_n"] = "auto",
        execution_backend: FMMExecutionBackend = "auto",
        nearfield_edge_chunk_size: int = 256,
        precompute_nearfield_scatter_schedules: bool = True,
        memory_objective: MemoryObjective = "balanced",
        memory_budget_bytes: Optional[int] = None,
        enable_interaction_cache: bool = True,
        retain_traversal_result: bool = True,
        retain_interactions: bool = True,
        prepare_stage_memory_split_enabled: Optional[bool] = None,
        autotune_m2l_chunk: bool = False,
        precompute_grouped_class_segments: Optional[bool] = None,
        grouped_schedule_budget_bytes: Optional[int] = None,
        nearfield_schedule_item_cap: Optional[int] = None,
        upward_leaf_batch_size: Optional[int] = None,
        host_refine_mode: str = "auto",
        fail_fast: bool = False,
        preset: Optional[Union[str, FMMPreset]] = None,
        fixed_order: Optional[int] = None,
        fixed_max_leaf_size: Optional[int] = None,
    ):
        basis_norm = str(expansion_basis).strip().lower()
        if basis_norm == "complex":
            basis_norm = "solidfmm"
        if basis_norm not in ("cartesian", "solidfmm"):
            raise ValueError(
                "expansion_basis must be 'cartesian', 'solidfmm', or 'complex'",
            )
        self.expansion_basis = basis_norm  # type: ignore[assignment]
        self.basis_impl = basis_impl
        self.m2l_impl = None if m2l_impl is None else str(m2l_impl).strip().lower()
        if self.m2l_impl is None and self._solidfmm_basis_mode() == "real":
            self.m2l_impl = "rot_scale"
        self.adaptive_order = bool(adaptive_order)
        self.p_gears = tuple(int(v) for v in (p_gears or ()))
        # Default the Pallas near-field ON wherever it can run (Ampere sm_80+),
        # and fall back to the pure-JAX near-field ONLY on hardware that cannot
        # run Pallas (e.g. RTX 2080 / sm_75) or CPU. Leaving it off by default
        # silently ran the ~10x-slower launch-bound jnp near-field on capable
        # GPUs; the non-Pallas path is retained solely as the sm_75/CPU lane.
        # An explicit use_pallas=True/False still overrides this resolution.
        if use_pallas is None:
            try:
                from jaccpot.pallas.nearfield_fused_leaf import (
                    pallas_nearfield_fused_supported,
                )

                resolved_use_pallas = bool(pallas_nearfield_fused_supported())
            except Exception:
                resolved_use_pallas = False
        else:
            resolved_use_pallas = bool(use_pallas)
        self.use_pallas = resolved_use_pallas
        self.reuse_topology = bool(reuse_topology)
        if int(rebuild_every) <= 0:
            raise ValueError("rebuild_every must be positive")
        self.rebuild_every = int(rebuild_every)
        self._recent_far_pairs_by_gear_counts: tuple[int, ...] = tuple()
        self._recent_dual_node_count: int = 0
        self._recent_dual_leaf_count: int = 0
        self._recent_dual_neighbor_count: int = 0
        self._recent_dual_far_pair_count: int = 0
        self._recent_dual_m2l_chunk_size: int = 0
        self._static_radix_tree_leaf_count: int = 0
        self._static_radix_tree_node_count: int = 0
        # Concrete tree depth (unpadded), stashed at build so the traced refresh
        # can pass it as a static arg to the upward M2M level loop. Radix trees
        # pad level_offsets to full Morton depth; using the padded shape makes the
        # M2M loop iterate many empty levels. See _resolve_upward_num_levels.
        self._static_upward_num_levels: Optional[int] = None
        self._static_radix_far_pair_count: int = 0
        self._static_radix_m2l_chunk_count: int = 0
        self._static_radix_l2l_edge_count: int = 0
        force_scale_mode_norm = str(mac_force_scale_mode).strip().lower()
        if force_scale_mode_norm not in ("prev", "prepass", "paper"):
            raise ValueError(
                "mac_force_scale_mode must be 'prev', 'prepass', or 'paper'"
            )
        self.mac_force_scale_mode = force_scale_mode_norm
        adaptive_error_model_norm = str(adaptive_error_model).strip().lower()
        if adaptive_error_model_norm not in (
            "tail_proxy",
            "dehnen_degree",
            "dehnen_paper",
        ):
            raise ValueError(
                "adaptive_error_model must be 'tail_proxy', 'dehnen_degree', or 'dehnen_paper'"
            )
        self.adaptive_error_model = adaptive_error_model_norm
        dehnen_geometry_mode_norm = str(dehnen_geometry_mode).strip().lower()
        if dehnen_geometry_mode_norm not in ("exact", "tree", "tree_approx", "runtime"):
            raise ValueError(
                "dehnen_geometry_mode must be 'exact', 'tree', 'tree_approx', or 'runtime'"
            )
        self.dehnen_geometry_mode = dehnen_geometry_mode_norm
        self.adaptive_eps = None if adaptive_eps is None else float(adaptive_eps)
        if self.adaptive_eps is not None and self.adaptive_eps <= 0.0:
            raise ValueError("adaptive_eps must be > 0 when provided")
        self._last_force_scale_nodes: Optional[Array] = None
        self._in_force_scale_prepass = False

        rotation_norm = str(complex_rotation).strip().lower()
        if rotation_norm != "solidfmm":
            raise ValueError("complex_rotation must be 'solidfmm'")
        self.complex_rotation = rotation_norm
        farfield_mode_norm = str(farfield_mode).strip().lower()
        if farfield_mode_norm not in ("auto", "pair_grouped", "class_major"):
            raise ValueError(
                "farfield_mode must be 'auto', 'pair_grouped', or 'class_major'"
            )
        self.farfield_mode = farfield_mode_norm
        self._explicit_streamed_far_pairs = streamed_far_pairs is not None
        self.streamed_far_pairs = bool(streamed_far_pairs)
        self.mixed_order_farfield = bool(mixed_order_farfield)
        self.mixed_order_min_order = (
            None if mixed_order_min_order is None else int(mixed_order_min_order)
        )
        if (
            self.mixed_order_min_order is not None
            and int(self.mixed_order_min_order) < 0
        ):
            raise ValueError("mixed_order_min_order must be >= 0")
        nearfield_mode_norm = str(nearfield_mode).strip().lower()
        if nearfield_mode_norm not in ("auto", "baseline", "bucketed"):
            raise ValueError("nearfield_mode must be 'auto', 'baseline', or 'bucketed'")
        runtime_path_norm = str(runtime_path).strip().lower()
        if runtime_path_norm not in ("auto", "legacy", "large_n"):
            raise ValueError("runtime_path must be 'auto', 'legacy', or 'large_n'")
        if runtime_path_norm == "legacy":
            warnings.warn(
                "runtime_path='legacy' is deprecated and will be removed; "
                "use runtime_path='large_n' or runtime_path='auto'.",
                FutureWarning,
                stacklevel=2,
            )
        execution_backend_norm = str(execution_backend).strip().lower()
        if execution_backend_norm not in ("auto", "radix", "octree"):
            raise ValueError("execution_backend must be 'auto', 'radix', or 'octree'")
        if int(nearfield_edge_chunk_size) <= 0:
            raise ValueError("nearfield_edge_chunk_size must be positive")
        self.nearfield_mode = nearfield_mode_norm
        self._explicit_nearfield_mode = nearfield_mode_norm != "auto"
        self.runtime_path = runtime_path_norm
        self.execution_backend = execution_backend_norm
        self.nearfield_edge_chunk_size = int(nearfield_edge_chunk_size)
        self.precompute_nearfield_scatter_schedules = bool(
            precompute_nearfield_scatter_schedules
        )
        objective_norm = str(memory_objective).strip().lower()
        if objective_norm not in ("balanced", "throughput", "minimum_memory"):
            raise ValueError(
                "memory_objective must be 'balanced', 'throughput', or 'minimum_memory'"
            )
        self.memory_objective: MemoryObjective = objective_norm  # type: ignore[assignment]
        self._explicit_memory_objective = objective_norm != "balanced"
        self.memory_budget_bytes = (
            None if memory_budget_bytes is None else int(memory_budget_bytes)
        )
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be > 0 when provided")
        self.enable_interaction_cache = bool(enable_interaction_cache)
        self.retain_traversal_result = bool(retain_traversal_result)
        self.retain_interactions = bool(retain_interactions)
        self.prepare_stage_memory_split_enabled = (
            None
            if prepare_stage_memory_split_enabled is None
            else bool(prepare_stage_memory_split_enabled)
        )
        self.fail_fast = bool(fail_fast)
        self.autotune_m2l_chunk = bool(autotune_m2l_chunk) and not self.fail_fast
        self.precompute_grouped_class_segments = (
            None
            if precompute_grouped_class_segments is None
            else bool(precompute_grouped_class_segments)
        )
        self.grouped_schedule_budget_bytes = (
            _GROUPED_SCHEDULE_BUDGET_DEFAULT
            if grouped_schedule_budget_bytes is None
            else int(grouped_schedule_budget_bytes)
        )
        if self.grouped_schedule_budget_bytes <= 0:
            raise ValueError("grouped_schedule_budget_bytes must be positive")
        self.nearfield_schedule_item_cap = (
            None
            if nearfield_schedule_item_cap is None
            else int(nearfield_schedule_item_cap)
        )
        if (
            self.nearfield_schedule_item_cap is not None
            and self.nearfield_schedule_item_cap <= 0
        ):
            raise ValueError("nearfield_schedule_item_cap must be > 0 when provided")
        self.upward_leaf_batch_size = (
            None if upward_leaf_batch_size is None else int(upward_leaf_batch_size)
        )
        if self.upward_leaf_batch_size is not None and self.upward_leaf_batch_size <= 0:
            raise ValueError("upward_leaf_batch_size must be > 0 when provided")
        dehnen_scale_val = float(dehnen_radius_scale)
        if dehnen_scale_val <= 0.0:
            raise ValueError("dehnen_radius_scale must be > 0")
        self.dehnen_radius_scale = dehnen_scale_val

        refine_mode_norm = str(host_refine_mode).strip().lower()
        if refine_mode_norm not in ("auto", "on", "off"):
            raise ValueError("host_refine_mode must be 'auto', 'on', or 'off'")
        if self.fail_fast:
            refine_mode_norm = "off"
        self.host_refine_mode = refine_mode_norm
        tree_type_norm = str(tree_type).strip().lower()
        supported_tree_types = set(available_tree_types())
        if tree_type_norm not in supported_tree_types:
            supported_txt = ", ".join(sorted(supported_tree_types))
            raise ValueError(
                f"tree_type must be one of ({supported_txt}), got '{tree_type}'"
            )
        self.tree_type: TreeType = tree_type_norm  # type: ignore[assignment]

        preset_config = get_preset_config(preset) if preset is not None else None

        resolved = _resolve_fmm_config(
            theta=theta,
            G=G,
            softening=softening,
            working_dtype=working_dtype,
            tree_build_mode=tree_build_mode,
            target_leaf_particles=target_leaf_particles,
            refine_local=refine_local,
            max_refine_levels=max_refine_levels,
            aspect_threshold=aspect_threshold,
            m2l_chunk_size=m2l_chunk_size,
            l2l_chunk_size=l2l_chunk_size,
            max_pair_queue=max_pair_queue,
            pair_process_block=pair_process_block,
            traversal_config=traversal_config,
            use_dense_interactions=use_dense_interactions,
            preset_config=preset_config,
        )

        self.config = resolved
        self.preset = resolved.preset
        self.theta = resolved.theta
        self.mac_type = mac_type
        if self._uses_dehnen_error_policy():
            if self.adaptive_error_model == "tail_proxy":
                self.adaptive_error_model = "dehnen_paper"
            if self.mac_force_scale_mode == "prev":
                self.mac_force_scale_mode = "paper"
        self.G = resolved.G
        self.softening = resolved.softening
        self.working_dtype = resolved.working_dtype
        self._preset_config = preset_config
        self._jit_tree_default = resolved.traversal.jit_tree
        self._jit_traversal_default = resolved.traversal.jit_traversal
        self.m2l_chunk_size = resolved.traversal.m2l_chunk_size
        self.l2l_chunk_size = resolved.traversal.l2l_chunk_size
        self.max_pair_queue = resolved.traversal.max_pair_queue
        self.pair_process_block = resolved.traversal.pair_process_block
        self.traversal_config = resolved.traversal.traversal_config
        self.tree_build_mode = resolved.tree.mode
        self.target_leaf_particles = resolved.tree.target_leaf_particles
        self.refine_local = resolved.tree.refine_local
        self.max_refine_levels = resolved.tree.max_refine_levels
        self.aspect_threshold = resolved.tree.aspect_threshold
        self.interaction_retry_logger = interaction_retry_logger
        self.use_dense_interactions = resolved.traversal.use_dense_interactions
        self._tree_workspace: Optional[object] = None
        self._locals_template: Optional[LocalExpansionData] = None
        self._interaction_cache: Optional[_InteractionCacheEntry] = None
        self._interaction_cache_hits: int = 0
        self._interaction_cache_misses: int = 0
        self._prepared_state_cache_key: Optional[tuple[Any, ...]] = None
        self._prepared_state_cache_value: Optional[PreparedStateLike] = None
        self._prepared_state_cache_positions: Optional[Array] = None
        self._prepared_state_cache_masses: Optional[Array] = None
        self._topology_reuse_entry: Optional[_TopologyReuseEntry] = None
        self._geometry_reuse_entry: Optional[_GeometryReuseEntry] = None
        self._recent_topology_reused: bool = False
        self._recent_retry_events: Tuple[DualTreeRetryEvent, ...] = tuple()
        self._compiled_profile_fingerprint_last: Optional[str] = None
        self._compiled_profile_transitions: int = 0
        self._large_n_eval_leaf_nodes_shape: tuple[int, ...] = ()
        self._large_n_eval_local_coefficients_shape: tuple[int, ...] = ()
        self._large_n_eval_local_centers_shape: tuple[int, ...] = ()
        self._large_n_eval_active_leaf_count: int = 0
        self._large_n_eval_max_leaf_size: int = 0
        self._large_n_eval_leaf_particle_slots: int = 0
        self._large_n_radix_payload_present: bool = False
        self._large_n_radix_payload_source_particle_shape: tuple[int, ...] = ()
        self._large_n_radix_payload_source_particle_slots: int = 0
        self._large_n_radix_payload_source_leaf_shape: tuple[int, ...] = ()
        self._large_n_radix_payload_source_leaf_slots: int = 0
        self._large_n_target_block_source_leaf_padded_shape: tuple[int, ...] = ()
        self._compiled_profile_refresh_calls: int = 0
        self._compiled_profile_refresh_reuse_tier_full: int = 0
        self._compiled_profile_refresh_reuse_tier_topology: int = 0
        self._compiled_profile_refresh_reuse_tier_overflow: int = 0
        self._large_n_same_topology_refresh_attempts: int = 0
        self._large_n_same_topology_refresh_hits: int = 0
        self._large_n_same_topology_refresh_misses: int = 0
        self._large_n_same_topology_refresh_miss_no_key: int = 0
        self._large_n_same_topology_refresh_miss_topology: int = 0
        self._large_n_same_topology_refresh_miss_neighbor: int = 0
        self._large_n_same_topology_refresh_miss_traced: int = 0
        self._large_n_same_topology_refresh_last_error: str = ""
        self._static_radix_refresh_hits: int = 0
        self._static_radix_refresh_misses: int = 0
        self._static_radix_profile_overflows: int = 0
        self._static_radix_compact_pair_reuse_hits: int = 0
        self._static_radix_compact_pair_reuse_misses: int = 0
        self._compiled_profile_multipoles_only_calls: int = 0
        self._compiled_profile_topology_rebuild_calls: int = 0
        self._large_n_overflow_profile_cap: int = 0
        self._large_n_overflow_profile_reprofiles: int = 0
        self._large_n_neighbor_edges_profile_cap: int = 0
        self._large_n_neighbor_edges_profile_reprofiles: int = 0
        self._refresh_timing_total_seconds: float = 0.0
        self._refresh_timing_input_seconds: float = 0.0
        self._refresh_timing_tree_upward_seconds: float = 0.0
        self._refresh_timing_tree_build_seconds: float = 0.0
        self._refresh_timing_upward_compute_seconds: float = 0.0
        self._refresh_timing_upward_geometry_seconds: float = 0.0
        self._refresh_timing_upward_mass_moments_seconds: float = 0.0
        self._refresh_timing_upward_p2m_seconds: float = 0.0
        self._refresh_timing_upward_m2m_seconds: float = 0.0
        self._refresh_timing_upward_source_motion_seconds: float = 0.0
        self._refresh_timing_dual_downward_seconds: float = 0.0
        self._refresh_timing_nearfield_seconds: float = 0.0
        self._refresh_timing_profile_accounting_seconds: float = 0.0
        self._refresh_timing_compile_or_sync_suspect_seconds: float = 0.0
        self._refresh_timing_dual_setup_seconds: float = 0.0
        self._refresh_timing_dual_artifact_build_seconds: float = 0.0
        self._refresh_timing_dual_split_shared_far_near_seconds: float = 0.0
        self._refresh_timing_dual_split_shared_count_seconds: float = 0.0
        self._refresh_timing_dual_split_shared_combined_fill_seconds: float = 0.0
        self._refresh_timing_dual_split_shared_far_fill_seconds: float = 0.0
        self._refresh_timing_dual_split_shared_near_fill_seconds: float = 0.0
        self._refresh_timing_dual_split_far_pairs_seconds: float = 0.0
        self._refresh_timing_dual_split_leaf_neighbors_seconds: float = 0.0
        self._refresh_timing_dual_split_combined_seconds: float = 0.0
        self._refresh_timing_dual_raw_combined_seconds: float = 0.0
        self._refresh_timing_dual_split_dense_buffers_seconds: float = 0.0
        self._refresh_timing_dual_far_pair_plan_seconds: float = 0.0
        self._refresh_timing_dual_m2l_autotune_seconds: float = 0.0
        self._refresh_timing_dual_select_interactions_seconds: float = 0.0
        self._refresh_timing_dual_downward_compute_seconds: float = 0.0
        self._refresh_timing_dual_m2l_compute_seconds: float = 0.0
        self._refresh_timing_dual_l2l_compute_seconds: float = 0.0
        self._refresh_timing_dual_final_symmetry_seconds: float = 0.0
        self._refresh_timing_dual_source_motion_seconds: float = 0.0
        self._refresh_timing_dual_finalize_seconds: float = 0.0
        self._refresh_timing_dual_residual_seconds: float = 0.0
        self._refresh_timing_nearfield_leaf_groups_seconds: float = 0.0
        self._refresh_timing_nearfield_precompute_seconds: float = 0.0
        self._refresh_timing_nearfield_target_blocks_seconds: float = 0.0
        self._refresh_timing_nearfield_block_sort_seconds: float = 0.0
        self._refresh_timing_nearfield_speed_layout_seconds: float = 0.0
        self._refresh_timing_nearfield_overflow_profile_seconds: float = 0.0
        self._refresh_timing_nearfield_radix_payload_seconds: float = 0.0
        self._refresh_timing_nearfield_neighbor_padding_seconds: float = 0.0
        self._refresh_timing_nearfield_state_pack_seconds: float = 0.0
        self._refresh_timing_nearfield_residual_seconds: float = 0.0
        self._refresh_timing_calls: int = 0
        self._refresh_timing_active: bool = False
        self._refresh_timing_enabled: bool = str(
            os.environ.get("JACCPOT_REFRESH_TIMING_ENABLE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._refresh_dual_planner_mode: str = (
            str(os.environ.get("JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_MODE", "auto"))
            .strip()
            .lower()
        )
        self._refresh_dual_planner_mode_on: bool = (
            self._refresh_dual_planner_mode == "on"
        )
        self._refresh_dual_planner_mode_auto: bool = (
            self._refresh_dual_planner_mode == "auto"
        )
        self._strict_gpu_mode: str = (
            str(os.environ.get("JACCPOT_STATIC_STRICT_GPU_MODE", "auto"))
            .strip()
            .lower()
        )
        self._strict_gpu_mode_on: bool = self._strict_gpu_mode == "on"
        self._strict_gpu_mode_auto: bool = self._strict_gpu_mode == "auto"
        self._strict_cap_record_enabled: bool = str(
            os.environ.get("JACCPOT_STATIC_STRICT_CAP_RECORD", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_cap_require_exact_profile_match: bool = str(
            os.environ.get("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        split_build_env_raw = os.environ.get(
            "JACCPOT_PREPARE_STAGE_MEMORY_SPLIT_ENABLED"
        )
        self._prepare_stage_memory_split_env_override: Optional[bool] = (
            None
            if split_build_env_raw is None
            else str(split_build_env_raw).strip().lower() in {"1", "true", "yes", "on"}
        )
        self._planner_steady_timing_bypass_enabled: bool = str(
            os.environ.get(
                "JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_STEADY_NO_SUBSTAGE_TIMING",
                "1",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_shared_env_applied: bool = False
        self._refresh_dual_planner_cache: dict[str, _RefreshDualPlannerHint] = {}
        self._refresh_dual_planner_cache_hits: int = 0
        self._refresh_dual_planner_cache_misses: int = 0
        self._refresh_dual_planner_compile_count: int = 0
        self._refresh_dual_planner_execute_count: int = 0
        self._refresh_dual_planner_steady_timing_bypass_count: int = 0
        self._refresh_dual_planner_compiled_route_count: int = 0
        self._refresh_strict_mode_active_count: int = 0
        self._strict_runner_compile_count: int = 0
        self._strict_runner_execute_count: int = 0
        self._strict_runner_profile_key_hits: int = 0
        self._strict_runner_profile_key_misses: int = 0
        self._strict_runner_fail_fast_reject_count: int = 0
        self._strict_runner_seen_profile_keys: set[str] = set()
        self._strict_v2_compile_count: int = 0
        self._strict_v2_execute_count: int = 0
        self._strict_v2_profile_key_hits: int = 0
        self._strict_v2_profile_key_misses: int = 0
        self._strict_v2_fail_fast_reject_count: int = 0
        self._strict_v2_seen_profile_keys: set[str] = set()
        self._strict_fused_mode_raw: str = (
            str(os.environ.get("JACCPOT_STATIC_STRICT_FUSED_MODE", "off"))
            .strip()
            .lower()
        )
        self._strict_fused_mode_enabled: bool = self._strict_fused_mode_raw in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._strict_fused_profile_set_raw: str = str(
            os.environ.get("JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET", "")
        ).strip()
        self._strict_fused_disable_hot_timing: bool = str(
            os.environ.get("JACCPOT_STATIC_STRICT_FUSED_DISABLE_HOT_TIMING", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_fused_disable_rematerialize: bool = str(
            os.environ.get("JACCPOT_STATIC_STRICT_FUSED_DISABLE_REMATERIALIZE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_fused_disallow_host_segment_fallback: bool = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK",
                "0",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Default ON: the device-only fused hot path enables the streamed
        # fast-lane (_prepare_state_dual_and_downward_strict_streamed_fast),
        # which is ~10x faster than the host-routed path for the strict fused
        # static-radix lane (200k particles: ~1224 -> ~119 ms/step on an A100)
        # with bit-identical energy / angular-momentum conservation
        # (max|dE/E0| = 8.415e-04 either way, verified over 400 steps). Set the
        # env var to "0" to opt back into the slower host-routed path, which is
        # retained only as a fallback.
        self._strict_fused_device_only: bool = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY",
                "1",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_fused_compiled_segment_loop: bool = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_COMPILED_SEGMENT_LOOP",
                "1",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_fused_jit_refresh_eval: bool = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_JIT_REFRESH_EVAL",
                "1",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._large_n_eval_diag_mode: str = (
            str(os.environ.get("JACCPOT_LARGE_N_EVAL_DIAG_MODE", "full"))
            .strip()
            .lower()
        )
        if self._large_n_eval_diag_mode not in {
            "full",
            "near_only",
            "far_only",
            "local_only",
            "near_zero",
            "far_zero",
            "permutation_only",
            "zero",
        }:
            self._large_n_eval_diag_mode = "full"
        self._large_n_nearfield_diag_mode: str = (
            str(os.environ.get("JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE", "full"))
            .strip()
            .lower()
        )
        if self._large_n_nearfield_diag_mode not in {
            "full",
            "self_only",
            "pairs_only",
            "overflow_only",
            "zero",
        }:
            self._large_n_nearfield_diag_mode = "full"
        self._strict_refresh_diag_mode: str = _normalize_strict_refresh_diag_mode(
            os.environ.get("JACCPOT_STRICT_REFRESH_DIAG_MODE", "full")
        )
        self._strict_refresh_detail_diag_mode: str = (
            _normalize_strict_refresh_detail_diag_mode(
                os.environ.get("JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE", "full")
            )
        )
        (
            self._strict_refresh_diag_tree_active,
            self._strict_refresh_diag_upward_active,
            self._strict_refresh_diag_downward_active,
            self._strict_refresh_diag_eval_active,
        ) = _strict_refresh_diag_stage_flags(self._strict_refresh_diag_mode)
        self._strict_fused_mode_active: bool = False
        self._strict_fused_compile_count: int = 0
        self._strict_fused_execute_count: int = 0
        self._strict_fused_profile_key_hits: int = 0
        self._strict_fused_profile_key_misses: int = 0
        self._strict_fused_fallback_count: int = 0
        self._strict_fused_last_fallback_reason: str = ""
        self._strict_fused_device_refresh_route_count: int = 0
        self._strict_fused_planner_bypassed_count: int = 0
        self._strict_velocity_verlet_acceleration_carry_active: bool = False
        self._strict_self_force_bootstrap_evaluations: int = 0
        self._strict_self_force_endpoint_evaluations: int = 0
        self._strict_external_bootstrap_evaluations: int = 0
        self._strict_external_endpoint_evaluations: int = 0
        self._strict_static_target_block_capacity_ok: bool = True
        self._large_n_radix_fast_occupancy_sort: bool = str(
            os.environ.get("JACCPOT_LARGE_N_RADIX_FAST_OCCUPANCY_SORT", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._large_n_radix_fast_skip_empty_tiles: bool = str(
            os.environ.get("JACCPOT_LARGE_N_RADIX_FAST_SKIP_EMPTY_TILES", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_fused_seen_profile_keys: set[str] = set()
        self._strict_fused_fastlane_diag_enabled: bool = str(
            os.environ.get("JACCPOT_STATIC_STRICT_FUSED_FASTLANE_DIAG", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_fused_fastlane_attempts: int = 0
        self._strict_fused_fastlane_hits: int = 0
        self._strict_fused_fastlane_misses: int = 0
        self._strict_fused_fastlane_last_blockers: tuple[str, ...] = tuple()
        self._strict_fused_fastlane_block_counts: dict[str, int] = {}
        self._strict_fused_jit_function_cache: dict[
            tuple[Any, ...], tuple[Any, ...]
        ] = {}
        self._strict_profiled_max_pair_queue: int = 0
        self._strict_profiled_pair_process_block: int = 0
        self._strict_profiled_context_key: str = ""
        self._strict_profile_catalog: dict[str, dict[str, int]] = {}
        self._strict_profile_loaded_once: bool = False
        self.fixed_order = fixed_order
        self.fixed_max_leaf_size = fixed_max_leaf_size
        self._explicit_m2l_chunk_size = m2l_chunk_size is not None
        self._explicit_l2l_chunk_size = l2l_chunk_size is not None
        self._explicit_traversal_config = traversal_config is not None
        self._explicit_max_pair_queue = max_pair_queue is not None
        self._explicit_pair_process_block = pair_process_block is not None
        self._explicit_grouped_interactions = grouped_interactions is not None
        self.grouped_interactions = grouped_interactions
        self._streamed_minimum_memory_gpu_default_split_build: bool = bool(
            self.memory_objective == "minimum_memory"
            and jax.default_backend() == "gpu"
            and self.tree_type == "radix"
            and self.expansion_basis == "solidfmm"
            and bool(self.streamed_far_pairs)
        )
        self._large_n_gpu_production_profile_cached: bool = (
            str(self.preset).strip().lower() == "large_n_gpu"
            and str(self.tree_type).strip().lower() == "radix"
            and str(self.expansion_basis).strip().lower() == "solidfmm"
            and str(self.execution_backend).strip().lower() != "octree"
        )
        self._static_runtime_fixed_sizing: bool = str(
            os.environ.get("JACCPOT_STATIC_RUNTIME_FIXED_SIZING", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._apply_large_n_gpu_production_contract()

    def _is_large_n_gpu_production_profile(self) -> bool:
        """Whether this solver should run the canonical large-N GPU contract."""
        return bool(
            getattr(
                self,
                "_large_n_gpu_production_profile_cached",
                (
                    str(self.preset).strip().lower() == "large_n_gpu"
                    and str(self.tree_type).strip().lower() == "radix"
                    and str(self.expansion_basis).strip().lower() == "solidfmm"
                    and str(self.execution_backend).strip().lower() != "octree"
                ),
            )
        )

    def _apply_large_n_gpu_production_contract(self) -> None:
        """Normalize large-N GPU runtime knobs to the canonical fast memory path."""

        if not self._is_large_n_gpu_production_profile():
            return

        if (
            self._explicit_memory_objective
            and self.memory_objective != "minimum_memory"
        ):
            warnings.warn(
                "large_n_gpu production profile coerces memory_objective to "
                "'minimum_memory' for memory-stable performance.",
                FutureWarning,
                stacklevel=2,
            )
        if (
            self._explicit_nearfield_mode
            and str(self.nearfield_mode).strip().lower() != "bucketed"
        ):
            warnings.warn(
                "large_n_gpu production profile coerces nearfield_mode to "
                "'bucketed' to keep radix fast-lane active.",
                FutureWarning,
                stacklevel=2,
            )
        if bool(self.grouped_interactions):
            warnings.warn(
                "large_n_gpu production profile disables grouped_interactions "
                "to keep streamed pair_grouped execution.",
                FutureWarning,
                stacklevel=2,
            )
        if self._explicit_streamed_far_pairs and not bool(self.streamed_far_pairs):
            warnings.warn(
                "large_n_gpu production profile enables streamed_far_pairs "
                "for low-memory execution.",
                FutureWarning,
                stacklevel=2,
            )

        # Keep large-N production on one stable runtime lane.
        self.runtime_path = "large_n"
        self.memory_objective = "minimum_memory"  # type: ignore[assignment]
        self.streamed_far_pairs = True
        self.grouped_interactions = False
        self._explicit_grouped_interactions = False
        self.farfield_mode = "pair_grouped"

        # Keep near-field on the radix fast-lane compatible bucketed path.
        self.nearfield_mode = "bucketed"
        self.precompute_nearfield_scatter_schedules = False
        self.mixed_order_farfield = False
        self.mixed_order_min_order = None

        # Keep the topology-derived interaction scaffold resident so fixed-shape
        # refreshes can reuse it instead of rebuilding the dual-tree artifacts.
        self.enable_interaction_cache = True
        self.retain_traversal_result = False
        self.retain_interactions = False
        self.precompute_grouped_class_segments = False
        if self.upward_leaf_batch_size is None:
            self.upward_leaf_batch_size = 2048

    def _resolve_execution_backend(self) -> str:
        """Resolve the active FMM execution backend without altering tree choice."""
        if self.execution_backend == "auto":
            return "radix"
        return self.execution_backend

    def _ensure_execution_backend_supported(
        self, *, tree: Optional[Tree] = None
    ) -> str:
        """Validate execution backends that are available for the current tree."""
        backend = self._resolve_execution_backend()
        if backend != "octree":
            return backend

        tree_type = getattr(tree, "tree_type", self.tree_type)
        if str(tree_type).strip().lower() != "octree":
            raise ValueError("execution_backend='octree' requires an octree tree_type")
        if self.expansion_basis != "solidfmm":
            raise NotImplementedError(
                "execution_backend='octree' currently supports basis='solidfmm' only"
            )
        return backend

    @property
    def recent_retry_events(
        self: "FastMultipoleMethod",
    ) -> Tuple[DualTreeRetryEvent, ...]:
        """Return retry telemetry collected during the latest build."""

        return self._recent_retry_events

    @property
    def recent_topology_reused(self: "FastMultipoleMethod") -> bool:
        """Whether the most recent prepare/evaluate path reused cached topology."""

        return bool(self._recent_topology_reused)

    def clear_prepared_state_cache(self: "FastMultipoleMethod") -> None:
        """Clear cached prepared-state payloads used by reuse mode."""

        self._prepared_state_cache_key = None
        self._prepared_state_cache_value = None
        self._prepared_state_cache_positions = None
        self._prepared_state_cache_masses = None
        self._topology_reuse_entry = None
        self._recent_topology_reused = False

    def clear_runtime_caches(
        self: "FastMultipoleMethod", *, clear_jax_compilation: bool = False
    ) -> None:
        """Release solver/runtime caches to reduce memory pressure."""

        self.clear_prepared_state_cache()
        self._locals_template = None
        self._interaction_cache = None
        self._interaction_cache_hits = 0
        self._interaction_cache_misses = 0
        self._tree_workspace = None
        self._last_force_scale_nodes = None
        self._recent_retry_events = tuple()
        self._recent_far_pairs_by_gear_counts = tuple()
        self._recent_dual_node_count = 0
        self._recent_dual_leaf_count = 0
        self._recent_dual_neighbor_count = 0
        self._recent_dual_far_pair_count = 0
        self._recent_dual_m2l_chunk_size = 0
        self._compiled_profile_fingerprint_last = None
        self._compiled_profile_transitions = 0
        self._large_n_eval_leaf_nodes_shape = ()
        self._large_n_eval_local_coefficients_shape = ()
        self._large_n_eval_local_centers_shape = ()
        self._large_n_eval_active_leaf_count = 0
        self._large_n_eval_max_leaf_size = 0
        self._large_n_eval_leaf_particle_slots = 0
        self._large_n_radix_payload_present = False
        self._large_n_radix_payload_source_particle_shape = ()
        self._large_n_radix_payload_source_particle_slots = 0
        self._large_n_radix_payload_source_leaf_shape = ()
        self._large_n_radix_payload_source_leaf_slots = 0
        self._large_n_target_block_source_leaf_padded_shape = ()
        self._compiled_profile_refresh_calls = 0
        self._compiled_profile_refresh_reuse_tier_full = 0
        self._compiled_profile_refresh_reuse_tier_topology = 0
        self._compiled_profile_refresh_reuse_tier_overflow = 0
        self._large_n_same_topology_refresh_attempts = 0
        self._large_n_same_topology_refresh_hits = 0
        self._large_n_same_topology_refresh_misses = 0
        self._large_n_same_topology_refresh_miss_no_key = 0
        self._large_n_same_topology_refresh_miss_topology = 0
        self._large_n_same_topology_refresh_miss_neighbor = 0
        self._large_n_same_topology_refresh_miss_traced = 0
        self._large_n_same_topology_refresh_last_error = ""
        self._static_radix_refresh_hits = 0
        self._static_radix_refresh_misses = 0
        self._static_radix_profile_overflows = 0
        self._static_radix_compact_pair_reuse_hits = 0
        self._static_radix_compact_pair_reuse_misses = 0
        self._compiled_profile_multipoles_only_calls = 0
        self._compiled_profile_topology_rebuild_calls = 0
        self._large_n_overflow_profile_cap = 0
        self._large_n_overflow_profile_reprofiles = 0
        self._large_n_neighbor_edges_profile_cap = 0
        self._large_n_neighbor_edges_profile_reprofiles = 0
        self._refresh_timing_total_seconds = 0.0
        self._refresh_timing_input_seconds = 0.0
        self._refresh_timing_tree_upward_seconds = 0.0
        self._refresh_timing_tree_build_seconds = 0.0
        self._refresh_timing_upward_compute_seconds = 0.0
        self._refresh_timing_upward_geometry_seconds = 0.0
        self._refresh_timing_upward_mass_moments_seconds = 0.0
        self._refresh_timing_upward_p2m_seconds = 0.0
        self._refresh_timing_upward_m2m_seconds = 0.0
        self._refresh_timing_upward_source_motion_seconds = 0.0
        self._refresh_timing_dual_downward_seconds = 0.0
        self._refresh_timing_nearfield_seconds = 0.0
        self._refresh_timing_profile_accounting_seconds = 0.0
        self._refresh_timing_compile_or_sync_suspect_seconds = 0.0
        self._refresh_timing_dual_setup_seconds = 0.0
        self._refresh_timing_dual_artifact_build_seconds = 0.0
        self._refresh_timing_dual_split_shared_far_near_seconds = 0.0
        self._refresh_timing_dual_split_shared_count_seconds = 0.0
        self._refresh_timing_dual_split_shared_combined_fill_seconds = 0.0
        self._refresh_timing_dual_split_shared_far_fill_seconds = 0.0
        self._refresh_timing_dual_split_shared_near_fill_seconds = 0.0
        self._refresh_timing_dual_split_far_pairs_seconds = 0.0
        self._refresh_timing_dual_split_leaf_neighbors_seconds = 0.0
        self._refresh_timing_dual_split_combined_seconds = 0.0
        self._refresh_timing_dual_raw_combined_seconds = 0.0
        self._refresh_timing_dual_split_dense_buffers_seconds = 0.0
        self._refresh_timing_dual_far_pair_plan_seconds = 0.0
        self._refresh_timing_dual_m2l_autotune_seconds = 0.0
        self._refresh_timing_dual_select_interactions_seconds = 0.0
        self._refresh_timing_dual_downward_compute_seconds = 0.0
        self._refresh_timing_dual_m2l_compute_seconds = 0.0
        self._refresh_timing_dual_l2l_compute_seconds = 0.0
        self._refresh_timing_dual_final_symmetry_seconds = 0.0
        self._refresh_timing_dual_source_motion_seconds = 0.0
        self._refresh_timing_dual_finalize_seconds = 0.0
        self._refresh_timing_dual_residual_seconds = 0.0
        self._refresh_timing_nearfield_leaf_groups_seconds = 0.0
        self._refresh_timing_nearfield_precompute_seconds = 0.0
        self._refresh_timing_nearfield_target_blocks_seconds = 0.0
        self._refresh_timing_nearfield_block_sort_seconds = 0.0
        self._refresh_timing_nearfield_speed_layout_seconds = 0.0
        self._refresh_timing_nearfield_overflow_profile_seconds = 0.0
        self._refresh_timing_nearfield_radix_payload_seconds = 0.0
        self._refresh_timing_nearfield_neighbor_padding_seconds = 0.0
        self._refresh_timing_nearfield_state_pack_seconds = 0.0
        self._refresh_timing_nearfield_residual_seconds = 0.0
        self._refresh_timing_calls = 0
        self._refresh_timing_active = False
        self._refresh_dual_planner_cache = {}
        self._refresh_dual_planner_cache_hits = 0
        self._refresh_dual_planner_cache_misses = 0
        self._refresh_dual_planner_compile_count = 0
        self._refresh_dual_planner_execute_count = 0
        self._refresh_dual_planner_steady_timing_bypass_count = 0
        self._refresh_dual_planner_compiled_route_count = 0
        self._refresh_strict_mode_active_count = 0
        self._strict_runner_compile_count = 0
        self._strict_runner_execute_count = 0
        self._strict_runner_profile_key_hits = 0
        self._strict_runner_profile_key_misses = 0
        self._strict_runner_fail_fast_reject_count = 0
        self._strict_runner_seen_profile_keys = set()
        self._strict_v2_compile_count = 0
        self._strict_v2_execute_count = 0
        self._strict_v2_profile_key_hits = 0
        self._strict_v2_profile_key_misses = 0
        self._strict_v2_fail_fast_reject_count = 0
        self._strict_v2_seen_profile_keys = set()
        self._strict_fused_mode_active = False
        self._strict_fused_compile_count = 0
        self._strict_fused_execute_count = 0
        self._strict_fused_profile_key_hits = 0
        self._strict_fused_profile_key_misses = 0
        self._strict_fused_fallback_count = 0
        self._strict_fused_last_fallback_reason = ""
        self._strict_fused_device_refresh_route_count = 0
        self._strict_fused_planner_bypassed_count = 0
        self._strict_velocity_verlet_acceleration_carry_active = False
        self._strict_self_force_bootstrap_evaluations = 0
        self._strict_self_force_endpoint_evaluations = 0
        self._strict_external_bootstrap_evaluations = 0
        self._strict_external_endpoint_evaluations = 0
        self._strict_static_target_block_capacity_ok = True
        self._strict_fused_seen_profile_keys = set()
        self._strict_fused_fastlane_attempts = 0
        self._strict_fused_fastlane_hits = 0
        self._strict_fused_fastlane_misses = 0
        self._strict_fused_fastlane_last_blockers = tuple()
        self._strict_fused_fastlane_block_counts = {}
        self._strict_fused_jit_function_cache = {}
        self._strict_profiled_max_pair_queue = 0
        self._strict_profiled_pair_process_block = 0
        self._strict_profiled_context_key = ""
        self._strict_profile_catalog = {}
        self._strict_profile_loaded_once = False
        _clear_global_runtime_caches(clear_jax_compilation=bool(clear_jax_compilation))

    def refresh_prepared_state(
        self: "FastMultipoleMethod",
        prepared_state: PreparedStateLike,
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
        fused_device_mode: bool = False,
    ) -> PreparedStateLike:
        """Refresh prepared state under large-N/radix profile constraints."""
        if not self._is_large_n_gpu_production_profile():
            raise NotImplementedError(
                "refresh_prepared_state is currently supported only for "
                "preset='large_n_gpu', tree_type='radix', expansion_basis='solidfmm'."
            )
        if not isinstance(prepared_state, LargeNPreparedState):
            raise NotImplementedError(
                "refresh_prepared_state currently supports LargeNPreparedState only."
            )

        self._compiled_profile_refresh_calls += 1
        refresh_timing_enabled = bool(getattr(self, "_refresh_timing_enabled", False))
        if not refresh_timing_enabled:
            next_state = self._refresh_large_n_same_topology(
                prepared_state,
                positions,
                masses,
                bounds=bounds,
                leaf_size=int(
                    prepared_state.max_leaf_size if leaf_size is None else leaf_size
                ),
                max_order=(
                    int(prepared_state.local_data.order)
                    if max_order is None
                    else int(max_order)
                ),
                theta=theta,
                fused_device_mode=bool(fused_device_mode),
            )
            if next_state is None:
                next_state = self.prepare_state(
                    positions,
                    masses,
                    bounds=bounds,
                    leaf_size=int(
                        prepared_state.max_leaf_size if leaf_size is None else leaf_size
                    ),
                    max_order=(
                        int(prepared_state.local_data.order)
                        if max_order is None
                        else int(max_order)
                    ),
                    theta=theta,
                    fused_device_mode=bool(fused_device_mode),
                )
            return next_state

        refresh_t0 = time.perf_counter()
        input_before = float(getattr(self, "_refresh_timing_input_seconds", 0.0))
        tree_before = float(getattr(self, "_refresh_timing_tree_upward_seconds", 0.0))
        dual_before = float(getattr(self, "_refresh_timing_dual_downward_seconds", 0.0))
        nearfield_before = float(
            getattr(self, "_refresh_timing_nearfield_seconds", 0.0)
        )
        profile_t0 = time.perf_counter()
        prev_profile = self._compiled_profile_from_prepared_state(prepared_state)
        prev_fingerprint = self._compiled_profile_fingerprint(prev_profile)
        profile_seconds = time.perf_counter() - profile_t0

        was_refresh_timing_active = bool(getattr(self, "_refresh_timing_active", False))
        self._refresh_timing_active = True
        try:
            next_state = self._refresh_large_n_same_topology(
                prepared_state,
                positions,
                masses,
                bounds=bounds,
                leaf_size=int(
                    prepared_state.max_leaf_size if leaf_size is None else leaf_size
                ),
                max_order=(
                    int(prepared_state.local_data.order)
                    if max_order is None
                    else int(max_order)
                ),
                theta=theta,
                fused_device_mode=bool(fused_device_mode),
            )
            if next_state is None:
                next_state = self.prepare_state(
                    positions,
                    masses,
                    bounds=bounds,
                    leaf_size=int(
                        prepared_state.max_leaf_size if leaf_size is None else leaf_size
                    ),
                    max_order=(
                        int(prepared_state.local_data.order)
                        if max_order is None
                        else int(max_order)
                    ),
                    theta=theta,
                    fused_device_mode=bool(fused_device_mode),
                )
        finally:
            self._refresh_timing_active = was_refresh_timing_active
        prepare_elapsed = time.perf_counter() - refresh_t0

        profile_t0 = time.perf_counter()
        next_profile = self._compiled_profile_from_prepared_state(next_state)
        next_fingerprint = self._compiled_profile_fingerprint(next_profile)
        self._compiled_profile_record_transition(next_fingerprint)

        if next_fingerprint == prev_fingerprint:
            self._compiled_profile_refresh_reuse_tier_full += 1
        elif self._compiled_profile_capacity_compatible(prev_profile, next_profile):
            self._compiled_profile_refresh_reuse_tier_topology += 1
        else:
            self._compiled_profile_refresh_reuse_tier_overflow += 1
        profile_seconds += time.perf_counter() - profile_t0
        total_elapsed = time.perf_counter() - refresh_t0
        input_delta = (
            float(getattr(self, "_refresh_timing_input_seconds", 0.0)) - input_before
        )
        tree_delta = (
            float(getattr(self, "_refresh_timing_tree_upward_seconds", 0.0))
            - tree_before
        )
        dual_delta = (
            float(getattr(self, "_refresh_timing_dual_downward_seconds", 0.0))
            - dual_before
        )
        nearfield_delta = (
            float(getattr(self, "_refresh_timing_nearfield_seconds", 0.0))
            - nearfield_before
        )
        stage_sum = (
            input_delta
            + tree_delta
            + dual_delta
            + nearfield_delta
            + float(profile_seconds)
        )
        # prepare_large_n_state records cumulative stage timings directly on
        # the solver. Attribute the unaccounted part of this refresh to Python
        # overhead, sync, compilation, or other work outside the explicit
        # large-N stage timers.
        self._refresh_timing_profile_accounting_seconds += float(profile_seconds)
        self._refresh_timing_total_seconds += float(total_elapsed)
        self._refresh_timing_compile_or_sync_suspect_seconds += max(
            0.0,
            float(total_elapsed) - float(stage_sum),
        )
        self._refresh_timing_calls += 1
        return next_state

    def strict_prepare_refresh_and_evaluate(
        self: "FastMultipoleMethod",
        prepared_state: Optional[PreparedStateLike],
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        theta: Optional[float] = None,
        jit_traversal: Optional[bool] = True,
        runtime_overrides: Optional[_RuntimeExecutionOverrides] = None,
        fused_device_mode: Optional[bool] = None,
    ) -> tuple[PreparedStateLike, Array]:
        """Strict static-radix helper: prepare/refresh once, then evaluate."""
        if not self._is_large_n_gpu_production_profile():
            self._strict_runner_fail_fast_reject_count += 1
            raise RuntimeError(
                "strict_prepare_refresh_and_evaluate requires large_n_gpu production profile."
            )

        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        profile_key = (
            f"n={int(positions_arr.shape[0])}|"
            f"leaf={int(leaf_size)}|"
            f"order={int(max_order)}|"
            f"theta={float(self.theta if theta is None else theta):.12g}"
        )
        if profile_key in self._strict_runner_seen_profile_keys:
            self._strict_runner_profile_key_hits += 1
        else:
            self._strict_runner_profile_key_misses += 1
            self._strict_runner_compile_count += 1
            self._strict_runner_seen_profile_keys.add(profile_key)
        self._strict_runner_execute_count += 1

        if prepared_state is None:
            next_state = self.prepare_state(
                positions_arr,
                masses_arr,
                bounds=bounds,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_tree=self._jit_tree_default,
                runtime_overrides_override=runtime_overrides,
                fused_device_mode=bool(
                    self._strict_fused_mode_active
                    if fused_device_mode is None
                    else fused_device_mode
                ),
            )
        else:
            if not isinstance(prepared_state, LargeNPreparedState):
                self._strict_runner_fail_fast_reject_count += 1
                raise RuntimeError(
                    "strict_prepare_refresh_and_evaluate requires LargeNPreparedState input."
                )
            next_state_try = self._refresh_large_n_same_topology(
                prepared_state,
                positions_arr,
                masses_arr,
                bounds=bounds,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                runtime_overrides_override=runtime_overrides,
                fused_device_mode=bool(
                    self._strict_fused_mode_active
                    if fused_device_mode is None
                    else fused_device_mode
                ),
            )
            if next_state_try is None:
                self._strict_runner_fail_fast_reject_count += 1
                raise RuntimeError(
                    "strict_prepare_refresh_and_evaluate fail-fast: "
                    "refresh miss (profile/topology mismatch)."
                )
            next_state = next_state_try

        acc = self.evaluate_prepared_state(
            next_state,
            target_indices=None,
            return_potential=False,
            jit_traversal=(
                self._jit_traversal_default
                if jit_traversal is None
                else bool(jit_traversal)
            ),
            max_acc_derivative_order=0,
        )
        return next_state, jnp.asarray(acc)

    def strict_run_segmented(
        self: "FastMultipoleMethod",
        *,
        state: Any,
        masses: Array,
        num_steps: int,
        refresh_every: int,
        segment_runner: Callable[[Any, Array, int], tuple[Any, Any]],
        positions_getter: Callable[[Any], Array],
        prepared_state: Optional[PreparedStateLike] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        theta: Optional[float] = None,
        jit_traversal: Optional[bool] = True,
        rematerialize_fn: Optional[Callable[[Any], Any]] = None,
        collect_history: bool = False,
    ) -> tuple[Any, PreparedStateLike, Optional[list[Any]]]:
        """Run strict refresh/evaluate cadence with caller-provided segment runner."""
        if int(num_steps) <= 0:
            raise ValueError("num_steps must be positive")
        if int(refresh_every) <= 0:
            raise ValueError("refresh_every must be positive")

        num_steps_i = int(num_steps)
        refresh_every_i = int(refresh_every)
        full_segments = num_steps_i // refresh_every_i
        tail_segment = num_steps_i % refresh_every_i

        state_curr = state
        prepared_curr = prepared_state
        history: Optional[list[Any]] = [] if collect_history else None
        runtime_overrides_cached = self._resolve_runtime_execution_overrides(
            num_particles=int(jnp.asarray(masses).shape[0]),
        )

        for _ in range(full_segments):
            positions_curr = positions_getter(state_curr)
            prepared_curr, acc_self = self.strict_prepare_refresh_and_evaluate(
                prepared_curr,
                positions_curr,
                masses,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_traversal=jit_traversal,
                runtime_overrides=runtime_overrides_cached,
                fused_device_mode=bool(self._strict_fused_mode_active),
            )
            state_curr, seg_hist = segment_runner(
                state_curr,
                jnp.asarray(acc_self),
                int(refresh_every_i),
            )
            if rematerialize_fn is not None:
                state_curr = rematerialize_fn(state_curr)
            if history is not None:
                history.append(seg_hist)

        if tail_segment > 0:
            positions_curr = positions_getter(state_curr)
            prepared_curr, acc_self = self.strict_prepare_refresh_and_evaluate(
                prepared_curr,
                positions_curr,
                masses,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_traversal=jit_traversal,
                runtime_overrides=runtime_overrides_cached,
                fused_device_mode=bool(self._strict_fused_mode_active),
            )
            state_curr, seg_hist = segment_runner(
                state_curr,
                jnp.asarray(acc_self),
                int(tail_segment),
            )
            if rematerialize_fn is not None:
                state_curr = rematerialize_fn(state_curr)
            if history is not None:
                history.append(seg_hist)

        return state_curr, prepared_curr, history

    def strict_run_v2(
        self: "FastMultipoleMethod",
        *,
        state: Array,
        masses: Array,
        dt: float,
        num_steps: int,
        refresh_every: int,
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
        prepared_state: Optional[PreparedStateLike] = None,
        initial_self_acceleration: Optional[Array] = None,
        jit_traversal: Optional[bool] = True,
        add_external: bool = False,
        external_acceleration_fn: Optional[Callable[[Array], Array]] = None,
        rematerialize_between_refresh: bool = True,
        return_history: bool = False,
        return_prepared_state: bool = True,
        step_callback: Optional[Callable[[Array, Array], None]] = None,
        step_callback_stride: int = 1,
    ) -> tuple[Array, Optional[PreparedStateLike], Optional[Array]]:
        """Run endpoint-correct velocity Verlet with strict prepared-state refresh.

        ``step_callback`` is an optional traced, side-effecting hook called inside
        the device-resident scan as ``step_callback(step_index, state)`` every
        ``step_callback_stride`` steps (``step_index`` and ``state`` are traced
        device values). It must be fire-and-forget (return nothing) and should use
        ``jax.debug.callback`` internally to ship only small, on-device-reduced
        data to the host (e.g. a projected density grid), so the GPU is not
        stalled. It does not touch the scan carry and is independent of
        ``return_history``."""
        state_arr = jnp.asarray(state)
        masses_arr = jnp.asarray(masses)
        dt_arr = jnp.asarray(float(dt), dtype=state_arr.dtype)
        num_steps_i = int(num_steps)

        if not self._is_large_n_gpu_production_profile():
            self._strict_v2_fail_fast_reject_count += 1
            raise RuntimeError("strict_run_v2 requires large_n_gpu production profile.")
        if num_steps_i <= 0:
            raise ValueError("num_steps must be positive")
        if int(refresh_every) != 1:
            self._strict_v2_fail_fast_reject_count += 1
            raise ValueError(
                "strict_run_v2 requires refresh_every=1 for endpoint-correct "
                "velocity-Verlet self gravity"
            )

        profile_key = (
            f"n={int(state_arr.shape[0])}|leaf={int(leaf_size)}|"
            f"order={int(max_order)}|refresh=1|"
            f"dt={float(dt):.12g}|external={int(bool(add_external))}|"
            f"theta={float(self.theta if theta is None else theta):.12g}"
        )
        if profile_key in self._strict_v2_seen_profile_keys:
            self._strict_v2_profile_key_hits += 1
        else:
            self._strict_v2_profile_key_misses += 1
            self._strict_v2_compile_count += 1
            self._strict_v2_seen_profile_keys.add(profile_key)
        self._strict_v2_execute_count += 1

        fused_mode_requested = bool(getattr(self, "_strict_fused_mode_enabled", False))
        fused_mode_allowed = self._strict_fused_profile_allows_n(
            int(state_arr.shape[0])
        )
        self._strict_fused_mode_active = bool(
            fused_mode_requested and fused_mode_allowed
        )
        if self._strict_fused_mode_active:
            if profile_key in self._strict_fused_seen_profile_keys:
                self._strict_fused_profile_key_hits += 1
            else:
                self._strict_fused_profile_key_misses += 1
                self._strict_fused_compile_count += 1
                self._strict_fused_seen_profile_keys.add(profile_key)
            self._strict_fused_execute_count += 1
            self._strict_fused_device_refresh_route_count += num_steps_i
            self._strict_fused_planner_bypassed_count += num_steps_i
        elif fused_mode_requested and not fused_mode_allowed:
            self._strict_fused_fallback_count += 1
            self._strict_fused_last_fallback_reason = (
                "particle_count_not_in_JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET"
            )
            # Fused mode was requested but this particle count is not in the
            # configured profile set. Refuse to silently disable the fused fast
            # lane and run a slower non-fused path -- raise so the profile set is
            # fixed (or cleared to allow all N) instead.
            raise RuntimeError(
                "strict fused mode requested but particle count "
                f"N={int(state_arr.shape[0])} is not in "
                "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET="
                f"{os.environ.get('JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET', '')!r}; "
                "refusing to silently fall back to a slower non-fused path. Add "
                "this N to the profile set, or leave it empty to allow all N."
            )
        else:
            self._strict_fused_last_fallback_reason = ""

        self._strict_velocity_verlet_acceleration_carry_active = True
        diag_mode = str(getattr(self, "_strict_refresh_diag_mode", "full"))
        eval_diag_mode = str(getattr(self, "_large_n_eval_diag_mode", "full"))
        detail_diag_mode = str(
            getattr(self, "_strict_refresh_detail_diag_mode", "full")
        )
        self_eval_active = (
            bool(getattr(self, "_strict_refresh_diag_eval_active", True))
            and detail_diag_mode == "full"
            and eval_diag_mode != "zero"
        )
        self._strict_self_force_bootstrap_evaluations = int(self_eval_active)
        self._strict_self_force_endpoint_evaluations = (
            num_steps_i if self_eval_active else 0
        )
        self._strict_external_bootstrap_evaluations = int(
            bool(add_external) and external_acceleration_fn is not None
        )
        self._strict_external_endpoint_evaluations = (
            num_steps_i
            if bool(add_external) and external_acceleration_fn is not None
            else 0
        )

        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(state_arr.shape[0])
        )
        prepared_curr = prepared_state
        if prepared_curr is None:
            prepared_curr = self.prepare_state(
                state_arr[:, 0, :],
                masses_arr,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_tree=self._jit_tree_default,
                runtime_overrides_override=runtime_overrides,
                fused_device_mode=bool(self._strict_fused_mode_active),
            )
        if self._strict_fused_mode_active and not isinstance(
            prepared_curr, LargeNPreparedState
        ):
            self._strict_runner_fail_fast_reject_count += 1
            raise RuntimeError(
                "strict fused velocity-Verlet requires LargeNPreparedState input."
            )
        if isinstance(prepared_curr, LargeNPreparedState):
            self._record_large_n_eval_shape_diagnostics(prepared_curr)

        def _evaluate_self(prepared_in: PreparedStateLike, state_in: Array) -> Array:
            if not self_eval_active:
                return jnp.zeros_like(state_in[:, 0, :])
            if eval_diag_mode == "permutation_only":
                return jnp.asarray(prepared_in.positions_sorted)[
                    prepared_in.inverse_permutation
                ] * jnp.asarray(0.0, dtype=state_in.dtype)
            return jnp.asarray(
                evaluate_large_n_state(
                    self,
                    prepared_in,
                    target_indices=None,
                    return_potential=False,
                    max_acc_derivative_order=0,
                ),
                dtype=state_in.dtype,
            )

        if not self_eval_active:
            acceleration_self_current = jnp.zeros_like(state_arr[:, 0, :])
        elif initial_self_acceleration is None:
            acceleration_self_current = _evaluate_self(prepared_curr, state_arr)
        else:
            acceleration_self_current = jnp.asarray(
                initial_self_acceleration, dtype=state_arr.dtype
            )
        if add_external and external_acceleration_fn is not None:
            acceleration_current = acceleration_self_current + jnp.asarray(
                external_acceleration_fn(state_arr), dtype=state_arr.dtype
            )
        else:
            acceleration_current = acceleration_self_current

        def _static_target_block_capacity_ok(
            prepared_in: PreparedStateLike,
        ) -> Array:
            padded = getattr(
                prepared_in,
                "nearfield_target_block_source_leaf_ids_padded",
                None,
            )
            if padded is None:
                return jnp.asarray(True)
            padded_arr = jnp.asarray(padded)
            if padded_arr.ndim != 3 or int(padded_arr.shape[1]) == 0:
                return jnp.asarray(True)
            offsets = jnp.asarray(prepared_in.neighbor_list.offsets)
            counts = offsets[1:] - offsets[:-1]
            capacity = int(padded_arr.shape[1]) * int(padded_arr.shape[2])
            return jnp.all(counts <= jnp.asarray(capacity, dtype=counts.dtype))

        def _refresh_and_evaluate_endpoint(
            prepared_in: PreparedStateLike,
            state_position: Array,
        ) -> tuple[PreparedStateLike, Array]:
            if diag_mode in {"integrator_only", "eval_only"}:
                prepared_new = prepared_in
            else:
                prepared_new = self._refresh_large_n_same_topology(
                    prepared_in,
                    state_position[:, 0, :],
                    masses_arr,
                    bounds=None,
                    leaf_size=int(leaf_size),
                    max_order=int(max_order),
                    theta=theta,
                    runtime_overrides_override=None,
                    fused_device_mode=bool(self._strict_fused_mode_active),
                )
                if prepared_new is None:
                    raise RuntimeError(
                        "strict velocity-Verlet refresh failed: topology/profile mismatch"
                    )
            return prepared_new, _evaluate_self(prepared_new, state_position)

        if self._strict_fused_mode_active:
            # Stash the concrete tree depth now, while prepared_curr is concrete,
            # so the traced refresh inside the compiled runner passes it as the
            # M2M level-loop static arg. Keyed into cache_key so a topology with a
            # different depth compiles its own runner.
            static_upward_num_levels = self._resolve_upward_num_levels(
                getattr(prepared_curr, "tree", None)
            )
            cache_key = (
                "strict_velocity_verlet",
                tuple(int(v) for v in state_arr.shape),
                str(state_arr.dtype),
                tuple(int(v) for v in masses_arr.shape),
                str(masses_arr.dtype),
                float(dt),
                num_steps_i,
                int(leaf_size),
                int(max_order),
                float(self.theta if theta is None else theta),
                bool(add_external),
                (
                    id(external_acceleration_fn)
                    if external_acceleration_fn is not None
                    else 0
                ),
                bool(rematerialize_between_refresh),
                bool(return_history),
                diag_mode,
                detail_diag_mode,
                eval_diag_mode,
                str(getattr(self, "_large_n_nearfield_diag_mode", "full")),
                static_upward_num_levels,
                id(step_callback) if step_callback is not None else 0,
                int(step_callback_stride),
            )
            jit_cache = getattr(self, "_strict_fused_jit_function_cache", {})
            compiled_runner = jit_cache.get(cache_key)
            if compiled_runner is None:

                @jax.jit
                def _compiled_runner(
                    prepared_initial: LargeNPreparedState,
                    state_initial: Array,
                    acceleration_initial: Array,
                ) -> tuple[
                    tuple[LargeNPreparedState, Array, Array, Array], Optional[Array]
                ]:
                    def _step(carry, scan_x):
                        (
                            prepared_now,
                            state_now,
                            acceleration_now,
                            capacity_ok_now,
                        ) = carry
                        position_new = (
                            state_now[:, 0]
                            + state_now[:, 1] * dt_arr
                            + 0.5 * acceleration_now * dt_arr**2
                        )
                        state_position = state_now.at[:, 0].set(position_new)
                        prepared_new, acceleration_self_new = (
                            _refresh_and_evaluate_endpoint(prepared_now, state_position)
                        )
                        if add_external and external_acceleration_fn is not None:
                            acceleration_new = acceleration_self_new + jnp.asarray(
                                external_acceleration_fn(state_position),
                                dtype=state_now.dtype,
                            )
                        else:
                            acceleration_new = acceleration_self_new
                        state_new = _velocity_verlet_state_update(
                            state_now,
                            acceleration_now,
                            acceleration_new,
                            dt_arr,
                        )
                        if rematerialize_between_refresh:
                            state_new = jnp.asarray(state_new, dtype=state_now.dtype)
                        if step_callback is not None:
                            # Fire-and-forget streaming hook (e.g. render). Gated by
                            # stride via lax.cond so it only fires + only computes its
                            # on-device reduction on emit steps. Returns a dummy int so
                            # both cond branches match; the result is discarded and the
                            # scan carry is untouched.
                            def _emit(_):
                                step_callback(scan_x, state_new)
                                return jnp.int32(0)

                            def _skip(_):
                                return jnp.int32(0)

                            jax.lax.cond(
                                (scan_x % jnp.int32(step_callback_stride))
                                == jnp.int32(0),
                                _emit,
                                _skip,
                                operand=None,
                            )
                        capacity_ok_new = capacity_ok_now & (
                            _static_target_block_capacity_ok(prepared_new)
                        )
                        return (
                            prepared_new,
                            state_new,
                            acceleration_new,
                            capacity_ok_new,
                        ), (state_new if return_history else None)

                    # Feed a per-step index only when a streaming callback needs it
                    # (keeps the no-callback path byte-for-byte unchanged).
                    scan_xs = (
                        jnp.arange(num_steps_i, dtype=jnp.int32)
                        if step_callback is not None
                        else None
                    )
                    return jax.lax.scan(
                        _step,
                        (
                            prepared_initial,
                            state_initial,
                            acceleration_initial,
                            _static_target_block_capacity_ok(prepared_initial),
                        ),
                        xs=scan_xs,
                        length=num_steps_i,
                    )

                compiled_runner = _compiled_runner
                jit_cache[cache_key] = compiled_runner
                self._strict_fused_jit_function_cache = jit_cache

            try:
                (
                    prepared_curr,
                    state_curr,
                    _,
                    capacity_ok_all,
                ), history_out = compiled_runner(
                    prepared_curr,
                    state_arr,
                    jnp.asarray(acceleration_current, dtype=state_arr.dtype),
                )
                self._strict_static_target_block_capacity_ok = bool(
                    np.asarray(jax.device_get(capacity_ok_all))
                )
                if not self._strict_static_target_block_capacity_ok:
                    max_blocks = os.environ.get(
                        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
                        "32",
                    )
                    raise RuntimeError(
                        "fused payload static target-block cap exceeded during "
                        "compiled velocity-Verlet scan: max_blocks_per_leaf="
                        f"{max_blocks}. Increase "
                        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF."
                    )
            except Exception as exc:
                if bool(
                    getattr(self, "_strict_fused_disallow_host_segment_fallback", False)
                ):
                    raise RuntimeError(
                        "strict fused velocity-Verlet scan failed while host fallback "
                        "is disallowed"
                    ) from exc
                raise
        else:
            state_curr = state_arr
            history_parts: list[Array] = []
            acceleration_now = jnp.asarray(acceleration_current, dtype=state_arr.dtype)
            for _ in range(num_steps_i):
                position_new = (
                    state_curr[:, 0]
                    + state_curr[:, 1] * dt_arr
                    + 0.5 * acceleration_now * dt_arr**2
                )
                state_position = state_curr.at[:, 0].set(position_new)
                prepared_curr, acceleration_self_new = _refresh_and_evaluate_endpoint(
                    prepared_curr, state_position
                )
                if add_external and external_acceleration_fn is not None:
                    acceleration_new = acceleration_self_new + jnp.asarray(
                        external_acceleration_fn(state_position),
                        dtype=state_curr.dtype,
                    )
                else:
                    acceleration_new = acceleration_self_new
                state_curr = _velocity_verlet_state_update(
                    state_curr, acceleration_now, acceleration_new, dt_arr
                )
                acceleration_now = acceleration_new
                if return_history:
                    history_parts.append(state_curr)
            history_out = jnp.stack(history_parts, axis=0) if return_history else None

        self._strict_runner_execute_count += num_steps_i
        if profile_key in self._strict_runner_seen_profile_keys:
            self._strict_runner_profile_key_hits += num_steps_i
        else:
            self._strict_runner_seen_profile_keys.add(profile_key)
            self._strict_runner_compile_count += 1
            self._strict_runner_profile_key_misses += 1
            self._strict_runner_profile_key_hits += max(0, num_steps_i - 1)
        prepared_out = prepared_curr if return_prepared_state else None
        return state_curr, prepared_out, history_out

    def strict_fused_prepared_eval_fn(
        self: "FastMultipoleMethod",
        *,
        positions: Array,
        masses: Array,
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
    ) -> tuple[PreparedStateLike, Callable[[PreparedStateLike], Array]]:
        """Build a fused-lane prepared state and return a jitted eval-only closure.

        Isolates the *evaluate* cost of the strict fused static-radix lane for
        apples-to-apples benchmarking against functional FMM eval APIs (e.g.
        jaxfmm ``eval_potential``): the prepared state is built eagerly with the
        fused device-mode layout (optimized flat compact far-pairs + static
        target-block near-field), exactly as ``strict_run_v2`` bootstraps it, and
        the returned closure runs the same self-force evaluation the fused step
        runs per endpoint (``evaluate_large_n_state``) with **no refresh and no
        velocity-Verlet update**.

        Returns ``(prepared_state, eval_fn)``; time ``eval_fn(prepared_state)``.
        """
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        if not self._is_large_n_gpu_production_profile():
            raise RuntimeError(
                "strict_fused_prepared_eval_fn requires large_n_gpu production profile."
            )
        fused_mode_requested = bool(getattr(self, "_strict_fused_mode_enabled", False))
        fused_mode_allowed = self._strict_fused_profile_allows_n(
            int(positions_arr.shape[0])
        )
        self._strict_fused_mode_active = bool(
            fused_mode_requested and fused_mode_allowed
        )
        if not self._strict_fused_mode_active:
            raise RuntimeError(
                "strict fused mode is not active for this particle count/config; "
                "enable JACCPOT_STATIC_STRICT_FUSED_MODE and include N in "
                "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET."
            )
        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(positions_arr.shape[0])
        )
        prepared = self.prepare_state(
            positions_arr,
            masses_arr,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            theta=theta,
            jit_tree=self._jit_tree_default,
            runtime_overrides_override=runtime_overrides,
            fused_device_mode=True,
        )
        if not isinstance(prepared, LargeNPreparedState):
            raise RuntimeError("strict fused eval-only requires a LargeNPreparedState.")
        self._record_large_n_eval_shape_diagnostics(prepared)

        @jax.jit
        def _eval(prepared_in: PreparedStateLike) -> Array:
            return jnp.asarray(
                evaluate_large_n_state(
                    self,
                    prepared_in,
                    target_indices=None,
                    return_potential=False,
                    max_acc_derivative_order=0,
                )
            )

        return prepared, _eval

    def _refresh_large_n_same_topology(
        self: "FastMultipoleMethod",
        prepared_state: LargeNPreparedState,
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]],
        leaf_size: int,
        max_order: int,
        theta: Optional[float],
        runtime_overrides_override: Optional[_RuntimeExecutionOverrides] = None,
        fused_device_mode: bool = False,
    ) -> Optional[LargeNPreparedState]:
        """Refresh large-N numeric payloads when the radix topology is unchanged."""

        self._large_n_same_topology_refresh_attempts += 1
        if not isinstance(prepared_state.tree, RadixTree):
            self._large_n_same_topology_refresh_misses += 1
            self._large_n_same_topology_refresh_miss_no_key += 1
            return None

        refresh_timing_active = bool(
            getattr(self, "_refresh_timing_active", False)
        ) and not (
            bool(fused_device_mode)
            and bool(getattr(self, "_strict_fused_disable_hot_timing", False))
        )

        input_t0 = time.perf_counter() if refresh_timing_active else 0.0
        positions_arr, masses_arr, input_dtype = self._prepare_state_input_arrays(
            positions,
            masses,
        )
        if refresh_timing_active:
            self._refresh_timing_input_seconds += time.perf_counter() - input_t0
        traced_refresh = bool(_contains_tracer((positions_arr, masses_arr)))
        allow_stateful_cache = bool(fused_device_mode) or (not traced_refresh)
        if (not allow_stateful_cache) and (not bool(fused_device_mode)):
            self._large_n_same_topology_refresh_misses += 1
            self._large_n_same_topology_refresh_miss_traced += 1
            return None

        self._validate_prepare_state_request(
            leaf_size=int(leaf_size),
            max_order=int(max_order),
        )
        runtime_overrides = runtime_overrides_override
        if runtime_overrides is None:
            runtime_overrides = self._resolve_runtime_execution_overrides(
                num_particles=int(positions_arr.shape[0]),
            )
        runtime_traversal_config = runtime_overrides.traversal_config
        runtime_m2l_chunk_size = runtime_overrides.m2l_chunk_size
        runtime_l2l_chunk_size = runtime_overrides.l2l_chunk_size
        upward_center_mode = runtime_overrides.center_mode
        refine_local_val = self.refine_local
        if runtime_overrides.refine_local_override is not None:
            refine_local_val = bool(runtime_overrides.refine_local_override)
        max_refine_levels_val = self.max_refine_levels
        aspect_threshold_val = self.aspect_threshold
        theta_val = float(self.theta if theta is None else theta)
        mac_type_val = self._base_mac_type()

        tree_config = self.config.tree
        if self.tree_type != "radix" and tree_config.mode in (
            "fixed_depth",
            "static_radix",
        ):
            tree_config = TreeBuilderConfig(
                mode="lbvh",
                target_leaf_particles=tree_config.target_leaf_particles,
                refine_local=tree_config.refine_local,
                max_refine_levels=tree_config.max_refine_levels,
                aspect_threshold=tree_config.aspect_threshold,
            )
        static_fused_refresh = bool(fused_device_mode) and (
            str(tree_config.mode).strip().lower() == "static_radix"
        )
        inferred_bounds = self._resolve_prepare_state_bounds(
            positions=positions_arr,
            bounds=bounds,
        )

        tree_t0 = time.perf_counter() if refresh_timing_active else 0.0
        refresh_topology_key = getattr(prepared_state, "topology_key", None)
        topology_candidate = None

        if static_fused_refresh:
            build_artifacts = self._rebuild_tree_artifacts_from_static_template(
                template_tree=prepared_state.tree,
                positions=positions_arr,
                masses=masses_arr,
                bounds=inferred_bounds,
                max_leaf_size=int(prepared_state.max_leaf_size),
                cache_leaf_parameter=int(leaf_size),
            )
            if refresh_topology_key is None:
                refresh_topology_key = "static_fused_template"
        else:
            previous_topology_key = refresh_topology_key
            if previous_topology_key is None:
                if tree_config.mode == "static_radix" and isinstance(
                    prepared_state.tree, RadixTree
                ):
                    previous_topology_key = self._static_radix_topology_key_from_tree(
                        prepared_state.tree,
                        leaf_size=int(leaf_size),
                    )
                else:
                    previous_codes = getattr(prepared_state.tree, "morton_codes", None)
                    if previous_codes is not None:
                        previous_topology_key = (
                            self._topology_reuse_key_from_sorted_codes(
                                sorted_codes=jnp.asarray(previous_codes),
                                tree_config=tree_config,
                                leaf_size=int(leaf_size),
                                refine_local=refine_local_val,
                                max_refine_levels=max_refine_levels_val,
                                aspect_threshold=aspect_threshold_val,
                            )
                        )
            if previous_topology_key is None:
                self._large_n_same_topology_refresh_misses += 1
                self._large_n_same_topology_refresh_miss_no_key += 1
                if tree_config.mode == "static_radix":
                    self._static_radix_refresh_misses += 1
                return None

            topology_candidate = self._topology_reuse_candidate(
                positions=positions_arr,
                bounds=inferred_bounds,
                tree_config=tree_config,
                leaf_size=int(leaf_size),
                refine_local=refine_local_val,
                max_refine_levels=max_refine_levels_val,
                aspect_threshold=aspect_threshold_val,
                allow_stateful_cache=allow_stateful_cache,
            )
            if (
                topology_candidate is None
                or topology_candidate.key != previous_topology_key
            ):
                self._large_n_same_topology_refresh_misses += 1
                self._large_n_same_topology_refresh_miss_topology += 1
                if tree_config.mode == "static_radix":
                    self._static_radix_refresh_misses += 1
                return None

            topology_entry = _TopologyReuseEntry(
                key=str(previous_topology_key),
                tree=prepared_state.tree,
                max_leaf_size=int(prepared_state.max_leaf_size),
                cache_leaf_parameter=int(leaf_size),
                reuse_count=0,
            )
            build_artifacts = self._rebuild_tree_artifacts_from_topology(
                candidate=topology_candidate,
                entry=topology_entry,
                positions=positions_arr,
                masses=masses_arr,
            )
            refresh_topology_key = topology_candidate.key

        strict_refresh_diag_mode = str(
            getattr(self, "_strict_refresh_diag_mode", "full")
        )
        strict_refresh_detail_diag_mode = str(
            getattr(self, "_strict_refresh_detail_diag_mode", "full")
        )
        strict_refresh_tree_detail_only = strict_refresh_detail_diag_mode in {
            "tree_sort_only",
            "tree_metadata_only",
        }
        strict_refresh_upward_detail_only = strict_refresh_detail_diag_mode in {
            "p2m_only",
            "m2m_only",
        }
        if bool(static_fused_refresh) and (
            strict_refresh_diag_mode == "tree_only" or strict_refresh_tree_detail_only
        ):
            self._large_n_same_topology_refresh_hits += 1
            if tree_config.mode == "static_radix":
                self._static_radix_refresh_hits += 1
            return replace(
                prepared_state,
                tree=build_artifacts.tree,
                topology_key=refresh_topology_key,
            )

        defer_geometry = False
        upward = self.prepare_upward_sweep(
            build_artifacts.tree,
            build_artifacts.positions_sorted,
            build_artifacts.masses_sorted,
            max_order=int(max_order),
            center_mode=upward_center_mode,
            max_leaf_size=int(build_artifacts.max_leaf_size),
            defer_geometry=defer_geometry,
        )
        locals_template = self._build_locals_template_for_prepare_state(
            tree=build_artifacts.tree,
            upward=upward,
            max_order=int(max_order),
            pos_sorted=build_artifacts.positions_sorted,
        )
        tree_artifacts = _PrepareStateTreeUpwardArtifacts(
            tree_mode=tree_config.mode,
            tree=build_artifacts.tree,
            positions_sorted=build_artifacts.positions_sorted,
            masses_sorted=build_artifacts.masses_sorted,
            inverse_permutation=build_artifacts.inverse_permutation,
            leaf_cap=int(build_artifacts.max_leaf_size),
            leaf_parameter=int(build_artifacts.cache_leaf_parameter),
            topology_key=refresh_topology_key,
            upward=upward,
            locals_template=locals_template,
        )
        if refresh_timing_active:
            self._refresh_timing_tree_upward_seconds += time.perf_counter() - tree_t0

        if bool(static_fused_refresh) and (
            strict_refresh_diag_mode == "upward_only"
            or strict_refresh_upward_detail_only
        ):
            dep = jnp.asarray(0.0, dtype=tree_artifacts.positions_sorted.dtype)
            multipoles = getattr(tree_artifacts.upward, "multipoles", None)
            packed = getattr(multipoles, "packed", None)
            centers = getattr(multipoles, "centers", None)
            if packed is not None:
                dep = dep + jnp.asarray(
                    jnp.real(jnp.sum(jnp.asarray(packed))),
                    dtype=dep.dtype,
                ) * jnp.asarray(0.0, dtype=dep.dtype)
            if centers is not None:
                dep = dep + jnp.asarray(
                    jnp.sum(jnp.asarray(centers)),
                    dtype=dep.dtype,
                ) * jnp.asarray(0.0, dtype=dep.dtype)
            diag_tree = replace(
                tree_artifacts.tree,
                positions_sorted=tree_artifacts.positions_sorted + dep,
            )
            self._large_n_same_topology_refresh_hits += 1
            if tree_config.mode == "static_radix":
                self._static_radix_refresh_hits += 1
            return replace(
                prepared_state,
                tree=diag_tree,
                topology_key=refresh_topology_key,
            )

        collected_retries: list[DualTreeRetryEvent] = []

        def record_retry(event: DualTreeRetryEvent) -> None:
            collected_retries.append(event)
            if self.interaction_retry_logger is not None:
                self.interaction_retry_logger(event)

        dual_t0 = time.perf_counter() if refresh_timing_active else 0.0
        strict_fused_traced_hot_path = bool(fused_device_mode) and bool(
            getattr(self, "_strict_fused_mode_active", False)
        )
        cached_compact_far_pairs = getattr(prepared_state, "compact_far_pairs", None)
        compact_far_pairs_carry_placeholder = cached_compact_far_pairs
        reuse_static_compact_pairs_enabled = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_REUSE_COMPACT_PAIRS",
                "1",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        allow_unsafe_compact_pair_reuse = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE",
                "0",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        safe_fresh_compact_pair_rebuild = (
            bool(strict_fused_traced_hot_path)
            and str(tree_config.mode).strip().lower() == "static_radix"
            and str(
                os.environ.get(
                    "JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD",
                    "1",
                )
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            and not bool(allow_unsafe_compact_pair_reuse)
        )
        reuse_static_compact_pairs = (
            bool(strict_fused_traced_hot_path)
            and bool(cached_compact_far_pairs is not None)
            and str(tree_config.mode).strip().lower() == "static_radix"
            and bool(reuse_static_compact_pairs_enabled)
            and bool(allow_unsafe_compact_pair_reuse)
        )
        if (
            bool(strict_fused_traced_hot_path)
            and bool(cached_compact_far_pairs is not None)
            and str(tree_config.mode).strip().lower() == "static_radix"
            and bool(reuse_static_compact_pairs_enabled)
            and not bool(allow_unsafe_compact_pair_reuse)
            and not bool(safe_fresh_compact_pair_rebuild)
        ):
            raise RuntimeError(
                "strict fused compact far-pair reuse is unsafe for moved "
                "static-radix positions: cached M2L pairs can change after "
                "the drift and corrupt endpoint forces. A production fix needs "
                "fresh fixed-cap compact pairs with an active mask/count, or a "
                "proven far-pair validity key. Set "
                "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE=1 "
                "only for legacy performance experiments."
            )
        if bool(safe_fresh_compact_pair_rebuild):
            cached_compact_far_pairs = None
        if bool(strict_fused_traced_hot_path) and (
            str(tree_config.mode).strip().lower() == "static_radix"
        ):
            if reuse_static_compact_pairs:
                self._static_radix_compact_pair_reuse_hits += 1
            else:
                self._static_radix_compact_pair_reuse_misses += 1
        if reuse_static_compact_pairs:
            src_far = jnp.asarray(cached_compact_far_pairs.sources, dtype=INDEX_DTYPE)
            tgt_far = jnp.asarray(cached_compact_far_pairs.targets, dtype=INDEX_DTYPE)
            far_pairs_by_gear = ((src_far, tgt_far),)
            downward = self._prepare_downward_with_artifacts(
                tree=tree_artifacts.tree,
                upward=tree_artifacts.upward,
                theta_val=theta_val,
                locals_template=tree_artifacts.locals_template,
                interactions=None,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                runtime_traversal_config=runtime_traversal_config,
                record_retry=record_retry,
                dense_buffers=None,
                grouped_interactions=False,
                grouped_buffers=None,
                grouped_segment_starts=None,
                grouped_segment_lengths=None,
                grouped_segment_class_ids=None,
                grouped_segment_sort_permutation=None,
                grouped_segment_group_ids=None,
                grouped_segment_unique_targets=None,
                farfield_mode="pair_grouped",
                far_pairs_coo=_FarPairCOO(
                    sources=src_far,
                    targets=tgt_far,
                    active_count=getattr(
                        cached_compact_far_pairs, "far_pair_count", None
                    ),
                ),
                far_pairs_by_gear=far_pairs_by_gear,
                adaptive_order=True,
                p_gears=(int(tree_artifacts.upward.multipoles.order),),
            )
            downward = downward._replace(
                interactions=_empty_interaction_storage_for_tree(tree_artifacts.tree)
            )
            dual_downward_artifacts = _PrepareStateDualDownwardArtifacts(
                interactions=None,
                neighbor_list=prepared_state.neighbor_list,
                traversal_result=None,
                compact_far_pairs=cached_compact_far_pairs,
                downward=downward,
                cache_entry=None,
            )
        else:
            dual_downward_artifacts = self._prepare_state_dual_and_downward(
                tree_artifacts=tree_artifacts,
                force_scale_nodes=prepared_state.force_scale_nodes,
                upward_center_mode=upward_center_mode,
                theta_val=theta_val,
                mac_type_val=mac_type_val,
                dehnen_radius_scale=self.dehnen_radius_scale,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                grouped_interactions=False,
                farfield_mode="pair_grouped",
                record_retry=record_retry,
                refine_local_val=refine_local_val,
                max_refine_levels_val=max_refine_levels_val,
                aspect_threshold_val=aspect_threshold_val,
                allow_stateful_cache=True,
                suppress_host_side_effects=strict_fused_traced_hot_path,
            )
        if str(tree_config.mode).strip().lower() == "static_radix":
            tree_now = build_artifacts.tree
            leaf_codes = getattr(tree_now, "leaf_codes", None)
            parent = getattr(tree_now, "parent", None)
            left_child = getattr(tree_now, "left_child", None)
            compact_pairs = getattr(dual_downward_artifacts, "compact_far_pairs", None)
            compact_sources = (
                getattr(compact_pairs, "sources", None)
                if compact_pairs is not None
                else None
            )
            far_pair_count = (
                int(getattr(compact_sources, "shape", (0,))[0])
                if compact_sources is not None
                else int(getattr(self, "_recent_dual_far_pair_count", 0))
            )
            chunk_size = (
                4096 if runtime_m2l_chunk_size is None else int(runtime_m2l_chunk_size)
            )
            self._static_radix_tree_leaf_count = (
                int(getattr(leaf_codes, "shape", (0,))[0])
                if leaf_codes is not None
                else int(getattr(self, "_recent_dual_leaf_count", 0))
            )
            self._static_radix_tree_node_count = (
                int(getattr(parent, "shape", (0,))[0])
                if parent is not None
                else int(getattr(self, "_recent_dual_node_count", 0))
            )
            self._static_radix_far_pair_count = int(far_pair_count)
            self._static_radix_m2l_chunk_count = (
                0
                if chunk_size <= 0 or far_pair_count <= 0
                else int((far_pair_count + chunk_size - 1) // chunk_size)
            )
            self._static_radix_l2l_edge_count = (
                2 * int(getattr(left_child, "shape", (0,))[0])
                if left_child is not None
                else 0
            )

        if refresh_timing_active:
            elapsed = time.perf_counter() - dual_t0
            recorded = float(
                getattr(self, "_refresh_timing_dual_downward_seconds", 0.0)
            )
            # _prepare_state_dual_and_downward records detailed timing itself.
            # Keep this branch intentionally empty except to make the elapsed
            # value visible while avoiding double accounting.
            _ = (elapsed, recorded)

        if (
            tree_config.mode != "static_radix"
            and not self._large_n_neighbor_list_matches(
                prepared_state.neighbor_list,
                dual_downward_artifacts.neighbor_list,
            )
        ):
            self._large_n_same_topology_refresh_misses += 1
            self._large_n_same_topology_refresh_miss_neighbor += 1
            return None

        self._large_n_same_topology_refresh_hits += 1
        if tree_config.mode == "static_radix":
            self._static_radix_refresh_hits += 1

        if allow_stateful_cache and (not traced_refresh):
            self._update_locals_template_cache_after_prepare(
                locals_template=tree_artifacts.locals_template,
                upward=tree_artifacts.upward,
                max_order=int(max_order),
            )
            self._recent_retry_events = tuple(collected_retries)
            self._record_strict_cap_profile_from_retries(
                self._recent_retry_events,
                context_key=self._strict_cap_profile_context_key(
                    tree_mode=str(tree_artifacts.tree_mode),
                    leaf_parameter=int(tree_artifacts.leaf_parameter),
                    particle_count=int(
                        jnp.asarray(tree_artifacts.positions_sorted).shape[0]
                    ),
                ),
            )
            self._topology_reuse_entry = _TopologyReuseEntry(
                key=str(refresh_topology_key),
                tree=tree_artifacts.tree,
                max_leaf_size=int(tree_artifacts.leaf_cap),
                cache_leaf_parameter=int(tree_artifacts.leaf_parameter),
                reuse_count=0,
            )

        refreshed_state = prepare_large_n_state(
            self,
            positions_arr=positions_arr,
            masses_arr=masses_arr,
            input_dtype=input_dtype,
            bounds=bounds,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            theta_val=theta_val,
            mac_type_val=mac_type_val,
            refine_local_val=refine_local_val,
            max_refine_levels_val=max_refine_levels_val,
            aspect_threshold_val=aspect_threshold_val,
            jit_tree_override=None,
            allow_stateful_cache=allow_stateful_cache,
            runtime_traversal_config=runtime_traversal_config,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            upward_center_mode=upward_center_mode,
            record_retry=record_retry,
            collected_retries=collected_retries,
            tree_artifacts=tree_artifacts,
            dual_downward_artifacts=dual_downward_artifacts,
            fused_device_mode=bool(fused_device_mode),
        )
        if bool(safe_fresh_compact_pair_rebuild):
            return replace(
                refreshed_state,
                compact_far_pairs=compact_far_pairs_carry_placeholder,
            )
        return refreshed_state

    def _large_n_neighbor_list_matches(
        self: "FastMultipoleMethod",
        previous: NodeNeighborList,
        current: NodeNeighborList,
    ) -> bool:
        """Return True when current active neighbor edges match previous state."""

        try:
            prev_offsets = np.asarray(jax.device_get(previous.offsets))
            cur_offsets = np.asarray(jax.device_get(current.offsets))
            prev_counts = np.asarray(jax.device_get(previous.counts))
            cur_counts = np.asarray(jax.device_get(current.counts))
            prev_leaf = np.asarray(jax.device_get(previous.leaf_indices))
            cur_leaf = np.asarray(jax.device_get(current.leaf_indices))
            if (
                prev_offsets.shape != cur_offsets.shape
                or prev_counts.shape != cur_counts.shape
                or prev_leaf.shape != cur_leaf.shape
            ):
                return False
            if (
                not np.array_equal(prev_offsets, cur_offsets)
                or not np.array_equal(prev_counts, cur_counts)
                or not np.array_equal(prev_leaf, cur_leaf)
            ):
                return False
            active_edges = int(cur_offsets[-1]) if cur_offsets.size > 0 else 0
            prev_neighbors = np.asarray(jax.device_get(previous.neighbors))
            cur_neighbors = np.asarray(jax.device_get(current.neighbors))
            if int(prev_neighbors.shape[0]) < active_edges:
                return False
            if int(cur_neighbors.shape[0]) < active_edges:
                return False
            return bool(
                np.array_equal(
                    prev_neighbors[:active_edges],
                    cur_neighbors[:active_edges],
                )
            )
        except Exception:
            return False

    def update_multipoles_only(
        self: "FastMultipoleMethod",
        prepared_state: PreparedStateLike,
        positions: Array,
        masses: Array,
        *,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
    ) -> PreparedStateLike:
        """Refresh multipole/local payloads when topology key remains unchanged."""
        if not self._is_large_n_gpu_production_profile():
            raise NotImplementedError(
                "update_multipoles_only is currently supported only for "
                "preset='large_n_gpu', tree_type='radix', expansion_basis='solidfmm'."
            )
        if not isinstance(prepared_state, LargeNPreparedState):
            raise NotImplementedError(
                "update_multipoles_only currently supports LargeNPreparedState only."
            )
        self._compiled_profile_multipoles_only_calls += 1
        refreshed = self.refresh_prepared_state(
            prepared_state,
            positions,
            masses,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
        )
        if getattr(refreshed, "topology_key", None) != getattr(
            prepared_state, "topology_key", None
        ):
            raise RuntimeError(
                "Topology changed during update_multipoles_only; "
                "use rebuild_topology_in_place for topology updates."
            )
        return refreshed

    def rebuild_topology_in_place(
        self: "FastMultipoleMethod",
        prepared_state: PreparedStateLike,
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
    ) -> PreparedStateLike:
        """Rebuild topology while attempting to remain profile-capacity compatible."""
        if not self._is_large_n_gpu_production_profile():
            raise NotImplementedError(
                "rebuild_topology_in_place is currently supported only for "
                "preset='large_n_gpu', tree_type='radix', expansion_basis='solidfmm'."
            )
        if not isinstance(prepared_state, LargeNPreparedState):
            raise NotImplementedError(
                "rebuild_topology_in_place currently supports LargeNPreparedState only."
            )
        self._compiled_profile_topology_rebuild_calls += 1
        return self.refresh_prepared_state(
            prepared_state,
            positions,
            masses,
            bounds=bounds,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
        )

    def export_m2l_autotune_cache(self: "FastMultipoleMethod") -> list[dict[str, Any]]:
        """Return a JSON-serializable snapshot of global M2L autotune results."""

        return _m2l_autotune_payload()

    def import_m2l_autotune_cache(
        self: "FastMultipoleMethod",
        payload: list[dict[str, Any]],
        *,
        merge: bool = True,
    ) -> int:
        """Restore global M2L autotune results from serialized payload."""

        return _restore_m2l_autotune_payload(payload, merge=bool(merge))

    def save_m2l_autotune_cache(self: "FastMultipoleMethod", path: str) -> int:
        """Write global M2L autotune cache to a JSON file."""

        payload = self.export_m2l_autotune_cache()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return int(len(payload))

    def load_m2l_autotune_cache(
        self: "FastMultipoleMethod",
        path: str,
        *,
        merge: bool = True,
    ) -> int:
        """Load global M2L autotune cache from a JSON file."""

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("autotune cache JSON must contain a list payload")
        return self.import_m2l_autotune_cache(payload, merge=bool(merge))

    def _validate_prepare_state_request(
        self,
        *,
        leaf_size: int,
        max_order: int,
    ) -> None:
        """Validate order/leaf-size constraints for state preparation."""
        if leaf_size < 1:
            raise ValueError("leaf_size must be >= 1")
        if self.fixed_order is not None and int(self.fixed_order) != int(max_order):
            raise ValueError("fixed_order must match max_order")
        if max_order > MAX_MULTIPOLE_ORDER and self.expansion_basis != "solidfmm":
            raise NotImplementedError(
                "orders above 4 require expansion_basis='solidfmm'",
            )

    def _prepare_state_input_arrays(
        self,
        positions: Array,
        masses: Array,
    ) -> tuple[Array, Array, Any]:
        """Validate prepare-state inputs and cast them to the working dtype."""
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        input_dtype = positions_arr.dtype

        if positions_arr.ndim != 2 or positions_arr.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if masses_arr.shape != (positions_arr.shape[0],):
            raise ValueError("masses must have shape (N,)")
        if positions_arr.shape[0] == 0:
            raise ValueError("need at least one particle")

        working_dtype = self.working_dtype
        if working_dtype is not None and positions_arr.dtype != working_dtype:
            positions_arr = positions_arr.astype(working_dtype)
        if working_dtype is not None and masses_arr.dtype != working_dtype:
            masses_arr = masses_arr.astype(working_dtype)
        return positions_arr, masses_arr, input_dtype

    def _resolve_prepare_state_bounds(
        self,
        *,
        positions: Array,
        bounds: Optional[Tuple[Array, Array]],
    ) -> tuple[Array, Array]:
        """Return bounds converted to the working dtype or infer them."""
        if bounds is None:
            min_corner, max_corner = _infer_bounds(positions)
            return min_corner, max_corner
        min_corner, max_corner = bounds
        return (
            jnp.asarray(min_corner, dtype=positions.dtype),
            jnp.asarray(max_corner, dtype=positions.dtype),
        )

    def _topology_reuse_candidate(
        self,
        *,
        positions: Array,
        bounds: Tuple[Array, Array],
        tree_config: TreeBuilderConfig,
        leaf_size: int,
        refine_local: bool,
        max_refine_levels: int,
        aspect_threshold: float,
        allow_stateful_cache: bool,
    ) -> Optional[_TopologyReuseCandidate]:
        """Return a radix-topology reuse signature when host-side caching is safe."""

        if (
            (not self.reuse_topology and tree_config.mode != "static_radix")
            or not allow_stateful_cache
            or self.tree_type != "radix"
        ):
            return None
        try:
            morton_codes = morton_encode(positions, bounds)
            orig_idx = jnp.arange(positions.shape[0], dtype=INDEX_DTYPE)
            sorted_indices = jnp.lexsort((orig_idx, morton_codes))
            sorted_codes = morton_codes[sorted_indices]
            if tree_config.mode == "static_radix":
                num_leaves = (int(positions.shape[0]) + int(leaf_size) - 1) // int(
                    leaf_size
                )
                hasher = hashlib.sha256()
                hasher.update(b"static_radix_topology_v1")
                hasher.update(
                    np.asarray(int(positions.shape[0]), dtype=np.int64).tobytes()
                )
                hasher.update(
                    np.asarray(max(2 * num_leaves - 1, 1), dtype=np.int64).tobytes()
                )
                hasher.update(
                    np.asarray(max(num_leaves - 1, 0), dtype=np.int64).tobytes()
                )
                hasher.update(np.asarray(int(leaf_size), dtype=np.int64).tobytes())
                key = hasher.hexdigest()
            else:
                key = self._topology_reuse_key_from_sorted_codes(
                    sorted_codes=sorted_codes,
                    tree_config=tree_config,
                    leaf_size=leaf_size,
                    refine_local=refine_local,
                    max_refine_levels=max_refine_levels,
                    aspect_threshold=aspect_threshold,
                )
        except Exception:
            return None
        if key is None:
            return None
        return _TopologyReuseCandidate(
            key=key,
            sorted_indices=jnp.asarray(sorted_indices, dtype=INDEX_DTYPE),
            sorted_codes=jnp.asarray(sorted_codes),
            bounds=bounds,
        )

    def _topology_reuse_key_from_sorted_codes(
        self,
        *,
        sorted_codes: Array,
        tree_config: TreeBuilderConfig,
        leaf_size: int,
        refine_local: bool,
        max_refine_levels: int,
        aspect_threshold: float,
    ) -> Optional[str]:
        """Build a stable topology key from sorted Morton codes and tree options."""

        try:
            hasher = hashlib.sha256()
            hasher.update(
                np.asarray(jax.device_get(sorted_codes), dtype=np.uint64).tobytes()
            )
            hasher.update(str(tree_config.mode).encode("utf8"))
            leaf_param = (
                int(leaf_size)
                if tree_config.mode == "lbvh"
                else int(tree_config.target_leaf_particles)
            )
            hasher.update(np.asarray(leaf_param, dtype=np.int64).tobytes())
            hasher.update(np.asarray(int(bool(refine_local)), dtype=np.int64).tobytes())
            hasher.update(np.asarray(int(max_refine_levels), dtype=np.int64).tobytes())
            hasher.update(
                np.asarray(float(aspect_threshold), dtype=np.float64).tobytes()
            )
        except Exception:
            return None
        return hasher.hexdigest()

    def _static_radix_topology_key_from_tree(
        self,
        tree: RadixTree,
        *,
        leaf_size: int,
    ) -> Optional[str]:
        """Return a stable key for a static-radix data-structure shape."""

        try:
            hasher = hashlib.sha256()
            hasher.update(b"static_radix_topology_v1")
            hasher.update(np.asarray(int(tree.num_particles), dtype=np.int64).tobytes())
            hasher.update(
                np.asarray(int(tree.parent.shape[0]), dtype=np.int64).tobytes()
            )
            hasher.update(
                np.asarray(int(tree.num_internal_nodes), dtype=np.int64).tobytes()
            )
            hasher.update(np.asarray(int(leaf_size), dtype=np.int64).tobytes())
        except Exception:
            return None
        return hasher.hexdigest()

    def _rebuild_tree_artifacts_from_topology(
        self,
        *,
        candidate: _TopologyReuseCandidate,
        entry: _TopologyReuseEntry,
        positions: Array,
        masses: Array,
    ) -> _TreeBuildArtifacts:
        """Reorder particles and attach them to a cached radix topology."""

        positions_sorted, masses_sorted, inverse = reorder_particles_by_indices(
            positions,
            masses,
            candidate.sorted_indices,
        )
        cached_tree = entry.tree
        if not isinstance(cached_tree, RadixTree):
            raise ValueError("topology reuse currently supports radix trees only")
        topology = cached_tree.topology
        if (
            cached_tree.build_mode == "static_radix"
            and candidate.sorted_codes is not None
            and candidate.bounds is not None
        ):
            num_internal = int(cached_tree.num_internal_nodes)
            leaf_starts = jnp.asarray(
                cached_tree.node_ranges[num_internal:, 0],
                dtype=INDEX_DTYPE,
            )
            topology = topology._replace(
                particle_indices=jnp.asarray(
                    candidate.sorted_indices,
                    dtype=INDEX_DTYPE,
                ),
                morton_codes=jnp.asarray(candidate.sorted_codes),
                bounds_min=jnp.asarray(candidate.bounds[0], dtype=positions.dtype),
                bounds_max=jnp.asarray(candidate.bounds[1], dtype=positions.dtype),
                leaf_codes=jnp.asarray(candidate.sorted_codes)[leaf_starts],
                leaf_depths=jnp.full(
                    (leaf_starts.shape[0],),
                    -1,
                    dtype=INDEX_DTYPE,
                ),
                use_morton_geometry=jnp.asarray(False, dtype=jnp.bool_),
            )
        rebuilt_tree = RadixTree(
            topology=topology,
            build_mode=cached_tree.build_mode,
            positions_sorted=positions_sorted,
            masses_sorted=masses_sorted,
            inverse_permutation=inverse,
            workspace=cached_tree.workspace,
        )
        return _TreeBuildArtifacts(
            tree=rebuilt_tree,
            positions_sorted=positions_sorted,
            masses_sorted=masses_sorted,
            inverse_permutation=inverse,
            workspace=cached_tree.workspace,
            max_leaf_size=int(entry.max_leaf_size),
            cache_leaf_parameter=int(entry.cache_leaf_parameter),
        )

    def _rebuild_tree_artifacts_from_static_template(
        self,
        *,
        template_tree: RadixTree,
        positions: Array,
        masses: Array,
        bounds: Optional[Tuple[Array, Array]],
        max_leaf_size: int,
        cache_leaf_parameter: int,
    ) -> _TreeBuildArtifacts:
        """Refresh static-radix tree artifacts from a fixed template topology."""

        rebuilt_result = rebuild_static_radix_tree_from_template(
            positions,
            masses,
            template_tree,
            bounds=bounds,
            return_reordered=True,
        )
        if not isinstance(rebuilt_result, tuple) or len(rebuilt_result) != 4:
            raise RuntimeError(
                "static radix template rebuild must return tree and reordered arrays"
            )
        rebuilt_tree, positions_sorted, masses_sorted, inverse = rebuilt_result
        if not isinstance(rebuilt_tree, RadixTree):
            raise ValueError("static radix template rebuild returned non-radix tree")
        return _TreeBuildArtifacts(
            tree=rebuilt_tree,
            positions_sorted=jnp.asarray(positions_sorted, dtype=positions.dtype),
            masses_sorted=jnp.asarray(masses_sorted, dtype=masses.dtype),
            inverse_permutation=jnp.asarray(inverse, dtype=INDEX_DTYPE),
            workspace=template_tree.workspace,
            max_leaf_size=int(max_leaf_size),
            cache_leaf_parameter=int(cache_leaf_parameter),
        )

    def _build_locals_template_for_prepare_state(
        self,
        *,
        tree: Tree,
        upward: TreeUpwardData,
        max_order: int,
        pos_sorted: Array,
    ) -> Optional[LocalExpansionData]:
        """Build initial local-expansion buffers matching the active basis."""
        if self.expansion_basis == "solidfmm":
            # Solid-FMM does not reuse a cached locals template across prepare calls.
            # Let the downward pass allocate its accumulator on demand so we do not
            # hold a redundant zero buffer alive across dual-tree staging.
            return None

        template = self._locals_template
        total_nodes = int(tree.parent.shape[0])
        coeff_count = total_coefficients(max_order)
        reuse_template = (
            template is not None
            and int(template.order) == max_order
            and template.coefficients.shape == (total_nodes, coeff_count)
            and template.coefficients.dtype == pos_sorted.dtype
        )

        if reuse_template:
            return LocalExpansionData(
                order=max_order,
                centers=upward.multipoles.centers,
                coefficients=jnp.zeros_like(template.coefficients),
            )
        return initialize_local_expansions(
            tree,
            upward.multipoles.centers,
            max_order=max_order,
        )

    def _prepare_state_tree_and_upward(
        self,
        *,
        positions_arr: Array,
        masses_arr: Array,
        bounds: Optional[Tuple[Array, Array]],
        leaf_size: int,
        max_order: int,
        refine_local_val: bool,
        max_refine_levels_val: int,
        aspect_threshold_val: float,
        jit_tree_override: Optional[bool],
        upward_center_mode: str,
        allow_stateful_cache: bool,
    ) -> _PrepareStateTreeUpwardArtifacts:
        """Build tree artifacts and run upward preparation for prepare_state."""
        tree_config = self.config.tree
        if self.tree_type != "radix" and tree_config.mode in (
            "fixed_depth",
            "static_radix",
        ):
            tree_config = TreeBuilderConfig(
                mode="lbvh",
                target_leaf_particles=tree_config.target_leaf_particles,
                refine_local=tree_config.refine_local,
                max_refine_levels=tree_config.max_refine_levels,
                aspect_threshold=tree_config.aspect_threshold,
            )

        inferred_bounds = self._resolve_prepare_state_bounds(
            positions=positions_arr,
            bounds=bounds,
        )
        jit_tree_flag = self._resolve_jit_tree_flag(
            positions_arr,
            jit_tree_override=jit_tree_override,
        )

        self._recent_topology_reused = False
        cached_topology = self._topology_reuse_entry
        topology_candidate = None
        if cached_topology is not None:
            topology_candidate = self._topology_reuse_candidate(
                positions=positions_arr,
                bounds=inferred_bounds,
                tree_config=tree_config,
                leaf_size=int(leaf_size),
                refine_local=refine_local_val,
                max_refine_levels=max_refine_levels_val,
                aspect_threshold=aspect_threshold_val,
                allow_stateful_cache=allow_stateful_cache,
            )
        can_reuse_cached_topology = (
            topology_candidate is not None
            and cached_topology is not None
            and topology_candidate.key == cached_topology.key
            and cached_topology.reuse_count < (self.rebuild_every - 1)
        )
        topology_key_for_state: Optional[str] = None

        tree_build_t0 = time.perf_counter()
        if can_reuse_cached_topology:
            build_artifacts = self._rebuild_tree_artifacts_from_topology(
                candidate=topology_candidate,
                entry=cached_topology,
                positions=positions_arr,
                masses=masses_arr,
            )
            self._recent_topology_reused = True
            topology_key_for_state = topology_candidate.key
            self._topology_reuse_entry = _TopologyReuseEntry(
                key=cached_topology.key,
                tree=build_artifacts.tree,
                max_leaf_size=int(cached_topology.max_leaf_size),
                cache_leaf_parameter=int(cached_topology.cache_leaf_parameter),
                reuse_count=int(cached_topology.reuse_count) + 1,
            )
        else:
            build_artifacts = _build_tree_with_config(
                positions_arr,
                masses_arr,
                inferred_bounds,
                tree_type=self.tree_type,
                tree_config=tree_config,
                leaf_size=int(leaf_size),
                workspace=(self._tree_workspace if allow_stateful_cache else None),
                jit_tree=jit_tree_flag,
                refine_local=refine_local_val,
                max_refine_levels=max_refine_levels_val,
                aspect_threshold=aspect_threshold_val,
            )
            if allow_stateful_cache:
                self._tree_workspace = build_artifacts.workspace
                if tree_config.mode == "static_radix":
                    topology_key_for_state = self._static_radix_topology_key_from_tree(
                        build_artifacts.tree,
                        leaf_size=int(leaf_size),
                    )
                if self.reuse_topology:
                    if topology_candidate is not None:
                        topology_key_for_state = topology_candidate.key
                    elif topology_key_for_state is None:
                        topology_key_for_state = (
                            self._topology_reuse_key_from_sorted_codes(
                                sorted_codes=build_artifacts.tree.morton_codes,
                                tree_config=tree_config,
                                leaf_size=int(leaf_size),
                                refine_local=refine_local_val,
                                max_refine_levels=max_refine_levels_val,
                                aspect_threshold=aspect_threshold_val,
                            )
                        )
                if topology_key_for_state is not None:
                    self._topology_reuse_entry = _TopologyReuseEntry(
                        key=topology_key_for_state,
                        tree=build_artifacts.tree,
                        max_leaf_size=int(build_artifacts.max_leaf_size),
                        cache_leaf_parameter=int(build_artifacts.cache_leaf_parameter),
                        reuse_count=0,
                    )
                elif self.reuse_topology:
                    self._topology_reuse_entry = None
        if bool(getattr(self, "_refresh_timing_active", False)):
            self._refresh_timing_tree_build_seconds += (
                time.perf_counter() - tree_build_t0
            )

        tree = build_artifacts.tree
        pos_sorted = build_artifacts.positions_sorted
        mass_sorted = build_artifacts.masses_sorted
        total_nodes = int(tree.parent.shape[0])
        num_internal = int(jnp.asarray(tree.left_child).shape[0])
        num_leaves = int(total_nodes - num_internal)
        _prepare_diag(
            "tree built "
            f"particles={int(pos_sorted.shape[0])} leaf_size={int(leaf_size)} "
            f"total_nodes={total_nodes} num_internal={num_internal} num_leaves={num_leaves} "
            f"tree_type={self.tree_type} tree_mode={tree_config.mode} jit_tree={bool(jit_tree_flag)}"
        )
        # Keep one leaf-size contract in eager and traced paths.
        leaf_cap_hint = int(build_artifacts.max_leaf_size)
        upward_leaf_batch = (
            int(self.upward_leaf_batch_size)
            if self.upward_leaf_batch_size is not None
            else None
        )
        _prepare_diag(
            "upward start "
            f"max_order={int(max_order)} center_mode={upward_center_mode} "
            f"leaf_cap={leaf_cap_hint} upward_leaf_batch_size={upward_leaf_batch}"
        )
        geometry_cache_key: Optional[tuple[Any, ...]] = None
        cached_geometry = None
        if allow_stateful_cache:
            if topology_key_for_state is not None:
                geometry_key_base: Any = topology_key_for_state
            else:
                bounds_key = (
                    None if bounds is None else (int(id(bounds[0])), int(id(bounds[1])))
                )
                geometry_key_base = (
                    self.tree_type,
                    tree_config.mode,
                    int(leaf_size),
                    bool(refine_local_val),
                    int(max_refine_levels_val),
                    float(aspect_threshold_val),
                    bounds_key,
                )
            geometry_cache_key = (
                geometry_key_base,
                int(id(positions_arr)),
                int(positions_arr.shape[0]),
                str(positions_arr.dtype),
            )
            geometry_entry = self._geometry_reuse_entry
            if geometry_entry is not None and geometry_entry.key == geometry_cache_key:
                cached_geometry = geometry_entry.geometry

        upward_t0 = time.perf_counter()
        defer_geometry = (
            bool(getattr(self, "_refresh_timing_active", False))
            and tree_config.mode == "static_radix"
            and str(upward_center_mode).strip().lower() == "com"
            and self._interaction_cache is not None
        )
        upward = self.prepare_upward_sweep(
            tree,
            pos_sorted,
            mass_sorted,
            max_order=max_order,
            center_mode=upward_center_mode,
            max_leaf_size=leaf_cap_hint,
            precomputed_geometry=cached_geometry,
            defer_geometry=defer_geometry,
        )
        if bool(getattr(self, "_refresh_timing_active", False)):
            self._refresh_timing_upward_compute_seconds += (
                time.perf_counter() - upward_t0
            )
        if (
            allow_stateful_cache
            and geometry_cache_key is not None
            and upward.geometry is not None
        ):
            self._geometry_reuse_entry = _GeometryReuseEntry(
                key=geometry_cache_key,
                geometry=upward.geometry,
            )
        _prepare_diag(
            "upward done "
            f"multipole_order={int(upward.multipoles.order)} "
            f"multipole_shape={tuple(int(v) for v in upward.multipoles.packed.shape)}"
        )
        locals_template = self._build_locals_template_for_prepare_state(
            tree=tree,
            upward=upward,
            max_order=max_order,
            pos_sorted=pos_sorted,
        )

        return _PrepareStateTreeUpwardArtifacts(
            tree_mode=tree_config.mode,
            tree=tree,
            positions_sorted=pos_sorted,
            masses_sorted=mass_sorted,
            inverse_permutation=build_artifacts.inverse_permutation,
            leaf_cap=leaf_cap_hint,
            leaf_parameter=build_artifacts.cache_leaf_parameter,
            topology_key=topology_key_for_state,
            upward=upward,
            locals_template=locals_template,
        )

    def _prepare_state_tree_upward_and_dual_downward(
        self,
        *,
        positions_arr: Array,
        masses_arr: Array,
        bounds: Optional[Tuple[Array, Array]],
        leaf_size: int,
        max_order: int,
        refine_local_val: bool,
        max_refine_levels_val: int,
        aspect_threshold_val: float,
        jit_tree_override: Optional[bool],
        upward_center_mode: str,
        allow_stateful_cache: bool,
        theta_val: float,
        mac_type_val: MACType,
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        record_retry: Callable[[DualTreeRetryEvent], None],
    ) -> tuple[_PrepareStateTreeUpwardArtifacts, _PrepareStateDualDownwardArtifacts]:
        """Build tree/upward and dual/downward artifacts in one helper call."""

        tree_artifacts = self._prepare_state_tree_and_upward(
            positions_arr=positions_arr,
            masses_arr=masses_arr,
            bounds=bounds,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            refine_local_val=bool(refine_local_val),
            max_refine_levels_val=int(max_refine_levels_val),
            aspect_threshold_val=float(aspect_threshold_val),
            jit_tree_override=jit_tree_override,
            upward_center_mode=upward_center_mode,
            allow_stateful_cache=bool(allow_stateful_cache),
        )
        dual_downward_artifacts = self._prepare_state_dual_and_downward(
            tree_artifacts=tree_artifacts,
            force_scale_nodes=None,
            upward_center_mode=upward_center_mode,
            theta_val=float(theta_val),
            mac_type_val=mac_type_val,
            dehnen_radius_scale=self.dehnen_radius_scale,
            runtime_traversal_config=runtime_traversal_config,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            grouped_interactions=False,
            farfield_mode=self.farfield_mode,
            record_retry=record_retry,
            refine_local_val=bool(refine_local_val),
            max_refine_levels_val=int(max_refine_levels_val),
            aspect_threshold_val=float(aspect_threshold_val),
            allow_stateful_cache=bool(allow_stateful_cache),
        )
        return tree_artifacts, dual_downward_artifacts

    def _prepare_state_dual_and_downward(
        self,
        *,
        tree_artifacts: _PrepareStateTreeUpwardArtifacts,
        force_scale_nodes: Optional[Array],
        upward_center_mode: str,
        theta_val: float,
        mac_type_val: MACType,
        dehnen_radius_scale: float,
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        grouped_interactions: bool,
        farfield_mode: str,
        record_retry: Callable[[DualTreeRetryEvent], None],
        refine_local_val: bool,
        max_refine_levels_val: int,
        aspect_threshold_val: float,
        allow_stateful_cache: bool,
        suppress_host_side_effects: bool = False,
    ) -> _PrepareStateDualDownwardArtifacts:
        """Build/reuse interactions and prepare downward artifacts.

        Memory note:
        The dominant warm prepare peak is usually the far-field payload that
        exists long enough to feed M2L: raw interactions or streamed COO pairs,
        optional grouped layouts, and the downward locals buffer. Only the
        returned ``TreeDownwardData`` is meant to survive into prepared state;
        grouped schedules and other M2L feed artifacts are transient and should
        stay scoped to this helper.
        """
        suppress_host_side_effects = bool(suppress_host_side_effects)
        refresh_timing_active = bool(
            getattr(self, "_refresh_timing_active", False)
        ) and (not suppress_host_side_effects)
        dual_total_t0 = time.perf_counter() if refresh_timing_active else 0.0
        dual_stage_sum = 0.0

        def _stage_now() -> float:
            return time.perf_counter() if refresh_timing_active else 0.0

        def _record_dual_stage(attr: str, start: float) -> None:
            nonlocal dual_stage_sum
            if not refresh_timing_active:
                return
            elapsed = float(time.perf_counter() - start)
            dual_stage_sum += elapsed
            setattr(
                self,
                attr,
                float(getattr(self, attr, 0.0)) + elapsed,
            )

        def _record_dual_artifact_substage(name: str, elapsed: float) -> None:
            if not refresh_timing_active:
                return
            attr_by_name = {
                "dual_split_shared_far_pairs_leaf_neighbors": "_refresh_timing_dual_split_shared_far_near_seconds",
                "dual_split_shared_count": "_refresh_timing_dual_split_shared_count_seconds",
                "dual_split_shared_combined_fill": "_refresh_timing_dual_split_shared_combined_fill_seconds",
                "dual_split_shared_far_fill": "_refresh_timing_dual_split_shared_far_fill_seconds",
                "dual_split_shared_near_fill": "_refresh_timing_dual_split_shared_near_fill_seconds",
                "dual_split_far_pairs": "_refresh_timing_dual_split_far_pairs_seconds",
                "dual_split_leaf_neighbors": "_refresh_timing_dual_split_leaf_neighbors_seconds",
                "dual_split_interactions_and_neighbors": "_refresh_timing_dual_split_combined_seconds",
                "dual_raw_interactions_and_neighbors": "_refresh_timing_dual_raw_combined_seconds",
                "dual_split_dense_buffers": "_refresh_timing_dual_split_dense_buffers_seconds",
            }
            attr = attr_by_name.get(str(name))
            if attr is None:
                return
            setattr(self, attr, float(getattr(self, attr, 0.0)) + float(elapsed))

        stage_t0 = _stage_now()
        pair_policy = None
        policy_state = None
        cache_key = None
        use_paper_fixed_policy = (not self.adaptive_order) and (
            self._uses_paper_style_traversal_policy()
        )
        if (
            bool(suppress_host_side_effects)
            and bool(getattr(self, "_strict_fused_mode_active", False))
            and bool(getattr(self, "_strict_fused_device_only", False))
        ):
            use_paper_fixed_policy = False

        # A static-radix topology key describes the capacity-fixed tree shape,
        # not the current leaf membership, geometry, or MAC decisions. Reusing
        # cached dual-tree artifacts across evolved positions can therefore
        # attach stale neighbor/far-field payloads to freshly sorted particles.
        stateful_cache_enabled = (
            bool(allow_stateful_cache)
            and bool(self.enable_interaction_cache)
            and str(tree_artifacts.tree_mode) != "static_radix"
            and (not suppress_host_side_effects)
        )
        grouped_interactions_active = bool(grouped_interactions)
        strict_fused_device_only_hot_path = (
            bool(suppress_host_side_effects)
            and bool(getattr(self, "_strict_fused_mode_active", False))
            and bool(getattr(self, "_strict_fused_device_only", False))
        )
        adaptive_order_active = bool(self.adaptive_order) and not bool(
            strict_fused_device_only_hot_path
        )
        mixed_order_farfield_active = bool(self.mixed_order_farfield) and not bool(
            strict_fused_device_only_hot_path
        )
        retain_interactions_active = bool(self.retain_interactions) and not bool(
            strict_fused_device_only_hot_path
        )
        need_traversal_result = (
            bool(self.retain_traversal_result)
            and not bool(strict_fused_device_only_hot_path)
        ) or bool(use_paper_fixed_policy)
        traced_prepare_inputs = bool(
            _contains_tracer(
                (tree_artifacts.positions_sorted, tree_artifacts.masses_sorted)
            )
        )
        strict_fused_node_interactions_safe_path = (
            bool(strict_fused_device_only_hot_path)
            and str(
                os.environ.get(
                    "JACCPOT_STATIC_STRICT_FUSED_NODE_INTERACTIONS_SAFE_PATH",
                    "0",
                )
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            and not (
                str(
                    os.environ.get(
                        "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE",
                        "0",
                    )
                )
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            )
        )
        use_compact_streamed_pairs = (
            (bool(self.streamed_far_pairs) or bool(strict_fused_device_only_hot_path))
            and not bool(strict_fused_node_interactions_safe_path)
            and not adaptive_order_active
            and not grouped_interactions_active
            and not mixed_order_farfield_active
            and not retain_interactions_active
            and not bool(need_traversal_result)
            and (
                not bool(traced_prepare_inputs)
                or bool(strict_fused_device_only_hot_path)
            )
        )
        need_compact_far_pairs = (
            bool(adaptive_order_active) and not bool(need_traversal_result)
        ) or bool(use_compact_streamed_pairs)
        need_node_interactions = not bool(use_compact_streamed_pairs)
        use_dense_interactions_for_prepare = bool(self.use_dense_interactions) and (
            self.expansion_basis != "solidfmm"
        )
        if not suppress_host_side_effects:
            _prepare_diag(
                "dual-tree start "
                f"theta={theta_val:.3f} mac_type={mac_type_val} "
                f"streamed={bool(self.streamed_far_pairs)} grouped={grouped_interactions_active} "
                f"farfield_mode={farfield_mode} memory_objective={self.memory_objective} "
                f"traversal_config={runtime_traversal_config} "
                f"need_compact_far_pairs={bool(need_compact_far_pairs)} "
                f"need_node_interactions={bool(need_node_interactions)} "
                f"dense_buffers={bool(use_dense_interactions_for_prepare)}"
            )
        if (
            runtime_traversal_config is not None
            and self.memory_objective == "minimum_memory"
            and jax.default_backend() == "gpu"
            and self.tree_type == "radix"
            and self.expansion_basis == "solidfmm"
            and bool(self.streamed_far_pairs)
            and not grouped_interactions_active
        ):
            total_nodes = int(tree_artifacts.tree.parent.shape[0])
            num_internal = int(jnp.asarray(tree_artifacts.tree.left_child).shape[0])
            num_leaves = max(1, total_nodes - num_internal)
            sanitized_traversal_config = (
                _cap_minimum_memory_streamed_gpu_traversal_config_for_tree(
                    traversal_config=runtime_traversal_config,
                    total_nodes=total_nodes,
                    num_leaves=num_leaves,
                    num_particles=int(tree_artifacts.positions_sorted.shape[0]),
                )
            )
            if sanitized_traversal_config != runtime_traversal_config:
                far_slots_before = total_nodes * int(
                    runtime_traversal_config.max_interactions_per_node
                )
                near_slots_before = num_leaves * int(
                    runtime_traversal_config.max_neighbors_per_leaf
                )
                if not suppress_host_side_effects:
                    _prepare_diag(
                        "capped explicit traversal_config for legacy streamed GPU walk "
                        f"total_nodes={total_nodes} num_leaves={num_leaves} "
                        f"far_slots={far_slots_before} near_slots={near_slots_before} "
                        f"from={runtime_traversal_config} "
                        f"to={sanitized_traversal_config}"
                    )
                runtime_traversal_config = sanitized_traversal_config
        jit_traversal_for_prepare = bool(
            self._jit_traversal_default
        ) and not _contains_tracer(
            (tree_artifacts.positions_sorted, tree_artifacts.masses_sorted)
        )
        if self.prepare_stage_memory_split_enabled is not None:
            allow_split_build = bool(self.prepare_stage_memory_split_enabled)
        elif self._prepare_stage_memory_split_env_override is None:
            # Default to the lower-peak split traversal build in the production
            # minimum-memory streamed GPU path; keep env opt-out for debugging.
            allow_split_build = bool(
                self._streamed_minimum_memory_gpu_default_split_build
                and not grouped_interactions_active
            )
        else:
            allow_split_build = bool(self._prepare_stage_memory_split_env_override)
        if bool(strict_fused_device_only_hot_path):
            allow_split_build = True
        if not suppress_host_side_effects:
            _prepare_diag(f"allow_split_build={bool(allow_split_build)}")
        tree_mode_static_radix = (
            str(tree_artifacts.tree_mode).strip().lower() == "static_radix"
        )
        strict_mode_active = bool(
            (
                self._strict_gpu_mode_on
                or (
                    self._strict_gpu_mode_auto
                    and self._large_n_gpu_production_profile_cached
                    and tree_mode_static_radix
                )
            )
        )
        if strict_mode_active:
            strict_context_key = self._strict_cap_profile_context_key(
                tree_mode=str(tree_artifacts.tree_mode),
                leaf_parameter=int(tree_artifacts.leaf_parameter),
                particle_count=int(
                    jnp.asarray(tree_artifacts.positions_sorted).shape[0]
                ),
            )
            strict_fused_hot_path = bool(
                getattr(self, "_strict_fused_mode_active", False)
            )
            strict_profile_key_stable = str(self._strict_profiled_context_key) == str(
                strict_context_key
            )
            if not (strict_fused_hot_path and strict_profile_key_stable):
                self._maybe_load_strict_cap_profile(context_key=strict_context_key)
            if bool(self._strict_cap_require_exact_profile_match):
                if str(self._strict_profiled_context_key) != str(strict_context_key):
                    self._strict_runner_fail_fast_reject_count += 1
                    raise RuntimeError(
                        "strict static lane requires exact cap profile key match: "
                        f"requested={strict_context_key} "
                        f"resolved={self._strict_profiled_context_key or 'none'}"
                    )
            profiled_q = int(self._strict_profiled_max_pair_queue)
            profiled_b = int(self._strict_profiled_pair_process_block)
            if bool(self._strict_cap_require_exact_profile_match) and profiled_q <= 0:
                self._strict_runner_fail_fast_reject_count += 1
                raise RuntimeError(
                    "strict static lane requires non-zero profiled max_pair_queue "
                    f"for key {strict_context_key}"
                )
            if profiled_q > 0:
                if runtime_traversal_config is not None:
                    runtime_traversal_config = DualTreeTraversalConfig(
                        max_pair_queue=max(
                            int(runtime_traversal_config.max_pair_queue),
                            int(profiled_q),
                        ),
                        process_block=(
                            int(profiled_b)
                            if profiled_b > 0
                            else int(runtime_traversal_config.process_block)
                        ),
                        max_interactions_per_node=int(
                            runtime_traversal_config.max_interactions_per_node
                        ),
                        max_neighbors_per_leaf=int(
                            runtime_traversal_config.max_neighbors_per_leaf
                        ),
                    )
                else:
                    runtime_traversal_config = DualTreeTraversalConfig(
                        max_pair_queue=int(profiled_q),
                        process_block=(
                            int(profiled_b)
                            if profiled_b > 0
                            else int(self.pair_process_block or 1024)
                        ),
                        max_interactions_per_node=8192,
                        max_neighbors_per_leaf=4096,
                    )
            if bool(traced_prepare_inputs) and not bool(
                strict_fused_device_only_hot_path
            ):
                allow_split_build = False
            if not suppress_host_side_effects:
                self._refresh_strict_mode_active_count += 1
                # Strict mode contract: one-shot shared count->fill and single queue.
                if not bool(getattr(self, "_strict_shared_env_applied", False)):
                    os.environ["YGGDRAX_DUAL_TREE_SHARED_COUNT_FILL_ONE_SHOT"] = "1"
                    os.environ[
                        "YGGDRAX_DUAL_TREE_SHARED_COUNT_FILL_STEADY_SINGLE_QUEUE"
                    ] = "1"
                    self._strict_shared_env_applied = True
        strict_streamed_fast_path = bool(
            strict_mode_active
            and bool(allow_split_build)
            and bool(use_compact_streamed_pairs)
            and not grouped_interactions_active
            and not bool(need_traversal_result)
            and not adaptive_order_active
            and not mixed_order_farfield_active
            and (
                not bool(traced_prepare_inputs)
                or bool(strict_fused_device_only_hot_path)
            )
        )
        if bool(strict_fused_device_only_hot_path):
            blockers: list[str] = []
            if not bool(strict_mode_active):
                blockers.append("strict_mode_inactive")
            if not bool(allow_split_build):
                blockers.append("split_build_disabled")
            if not bool(use_compact_streamed_pairs):
                blockers.append("compact_streamed_pairs_disabled")
            if bool(grouped_interactions_active):
                blockers.append("grouped_interactions_active")
            if bool(need_traversal_result):
                blockers.append("traversal_result_required")
            if bool(adaptive_order_active):
                blockers.append("adaptive_order_active")
            if bool(mixed_order_farfield_active):
                blockers.append("mixed_order_farfield_active")
            if bool(traced_prepare_inputs) and not bool(strict_streamed_fast_path):
                blockers.append("compact_streamed_tracer_unsupported")
            if bool(getattr(self, "_strict_fused_fastlane_diag_enabled", True)):
                self._strict_fused_fastlane_attempts += 1
                if bool(strict_streamed_fast_path):
                    self._strict_fused_fastlane_hits += 1
                    self._strict_fused_fastlane_last_blockers = tuple()
                else:
                    self._strict_fused_fastlane_misses += 1
                    self._strict_fused_fastlane_last_blockers = tuple(blockers)
                    counts = dict(
                        getattr(self, "_strict_fused_fastlane_block_counts", {})
                    )
                    for key in blockers:
                        counts[str(key)] = int(counts.get(str(key), 0)) + 1
                    self._strict_fused_fastlane_block_counts = counts
            if not bool(strict_streamed_fast_path):
                # Production fused device-only hot path MUST engage the radix
                # streamed fast lane. Never silently fall back to the slower
                # generic build -- raise loudly with the blocking reasons so a
                # misconfiguration is caught instead of quietly running the
                # ~10x-slower generic path.
                self._strict_fused_fallback_count += 1
                self._strict_fused_last_fallback_reason = (
                    "fused_device_only_hot_path_fastlane_blocked:" + ",".join(blockers)
                )
                raise RuntimeError(
                    "strict fused device-only production lane could not engage the "
                    "radix streamed fast lane and must not silently fall back to a "
                    f"slower path (blockers={blockers}). This indicates a "
                    "misconfiguration of the large-N GPU production profile."
                )
        if strict_streamed_fast_path:
            if not suppress_host_side_effects:
                self._refresh_dual_planner_cache_hits += 1
                self._refresh_dual_planner_execute_count += 1
                self._refresh_dual_planner_steady_timing_bypass_count += 1
            return self._prepare_state_dual_and_downward_strict_streamed_fast(
                tree_artifacts=tree_artifacts,
                theta_val=theta_val,
                mac_type_val=mac_type_val,
                dehnen_radius_scale=dehnen_radius_scale,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                record_retry=record_retry,
                farfield_mode=farfield_mode,
                retain_interactions=bool(retain_interactions_active),
                suppress_host_side_effects=suppress_host_side_effects,
            )

        if self.adaptive_order or use_paper_fixed_policy:
            policy_orders = self.p_gears
            if use_paper_fixed_policy:
                policy_orders = (int(tree_artifacts.upward.multipoles.order),)
            if len(policy_orders) == 0:
                raise ValueError("adaptive traversal policy requires non-empty orders")
            policy_state = self._build_adaptive_policy_state(
                upward=tree_artifacts.upward,
                tree=tree_artifacts.tree,
                positions_sorted=tree_artifacts.positions_sorted,
                p_gears=policy_orders,
                force_scale_nodes=force_scale_nodes,
                eps=jnp.asarray(
                    (
                        self.adaptive_eps
                        if self.adaptive_eps is not None
                        else adaptive_policy_tolerance(
                            theta=theta_val,
                            p_gears=policy_orders,
                            dtype=tree_artifacts.upward.multipoles.packed.real.dtype,
                        )
                    ),
                    dtype=tree_artifacts.upward.multipoles.packed.real.dtype,
                ),
                theta=jnp.asarray(
                    theta_val,
                    dtype=tree_artifacts.upward.multipoles.packed.real.dtype,
                ),
                error_model_code=jnp.asarray(
                    self._traversal_policy_error_model_code(),
                    dtype=jnp.int32,
                ),
                dehnen_geometry_mode=self.dehnen_geometry_mode,
            )
            pair_policy = adaptive_pair_policy
        else:
            cache_key = _interaction_cache_key(
                tree_artifacts.tree,
                topology_key=tree_artifacts.topology_key,
                tree_mode=tree_artifacts.tree_mode,
                leaf_parameter=tree_artifacts.leaf_parameter,
                theta=theta_val,
                mac_type=mac_type_val,
                dehnen_radius_scale=dehnen_radius_scale,
                expansion_basis=self.expansion_basis,
                center_mode=upward_center_mode,
                max_pair_queue=self.max_pair_queue,
                pair_process_block=self.pair_process_block,
                traversal_config=runtime_traversal_config,
                refine_local=refine_local_val,
                max_refine_levels=max_refine_levels_val,
                aspect_threshold=aspect_threshold_val,
            )
        has_pair_policy = pair_policy is not None
        has_policy_state = policy_state is not None
        planner_enabled = bool(
            (
                self._refresh_dual_planner_mode_on
                or (
                    self._refresh_dual_planner_mode_auto
                    and self._large_n_gpu_production_profile_cached
                    and tree_mode_static_radix
                )
            )
        )
        planner_hint: Optional[_RefreshDualPlannerHint] = None
        planner_cache_hit = False
        if planner_enabled:
            strict_split_fastlane = bool(
                (
                    bool(suppress_host_side_effects)
                    and not has_pair_policy
                    and not has_policy_state
                )
                or (
                    strict_mode_active
                    and bool(allow_split_build)
                    and not grouped_interactions_active
                    and not bool(need_traversal_result)
                    and not has_pair_policy
                    and not has_policy_state
                )
            )
            if strict_split_fastlane:
                # Strict/static production lane: keep routing fully on a
                # fixed host-side decision and skip compiled route probing +
                # device_get round-trips in the refresh hot path.
                planner_hint = _RefreshDualPlannerHint(
                    use_split_build=bool(allow_split_build),
                    suppress_substage_timing=True,
                )
                planner_cache_hit = True
                if not suppress_host_side_effects:
                    self._refresh_dual_planner_cache_hits += 1
                    self._refresh_dual_planner_execute_count += 1
            else:
                traversal_key = (
                    "none"
                    if runtime_traversal_config is None
                    else (
                        f"{int(runtime_traversal_config.max_pair_queue)}:"
                        f"{int(runtime_traversal_config.process_block)}:"
                        f"{int(runtime_traversal_config.max_interactions_per_node)}:"
                        f"{int(runtime_traversal_config.max_neighbors_per_leaf)}"
                    )
                )
                planner_key = "|".join(
                    (
                        str(tree_artifacts.topology_key),
                        str(tree_artifacts.tree_mode),
                        str(int(tree_artifacts.leaf_parameter)),
                        f"{float(theta_val):.12g}",
                        str(mac_type_val),
                        str(grouped_interactions_active),
                        str(bool(need_traversal_result)),
                        str(bool(need_compact_far_pairs)),
                        str(bool(need_node_interactions)),
                        str(bool(allow_split_build)),
                        str(traversal_key),
                    )
                )
                planner_hint = self._refresh_dual_planner_cache.get(planner_key)
                if planner_hint is None:
                    if not suppress_host_side_effects:
                        self._refresh_dual_planner_cache_misses += 1
                    total_nodes_planner = int(tree_artifacts.tree.parent.shape[0])
                    internal_nodes_planner = int(
                        jnp.asarray(tree_artifacts.tree.left_child).shape[0]
                    )
                    leaf_count_planner = max(
                        0, total_nodes_planner - internal_nodes_planner
                    )
                    (
                        use_split_build_compiled,
                        _use_compact_shared_far_near_compiled,
                        suppress_substage_timing_compiled,
                    ) = _compiled_refresh_dual_planner_route(
                        allow_split_build_flag=jnp.asarray(
                            bool(allow_split_build), dtype=jnp.bool_
                        ),
                        grouped_interactions_flag=jnp.asarray(
                            grouped_interactions_active, dtype=jnp.bool_
                        ),
                        need_traversal_result_flag=jnp.asarray(
                            bool(need_traversal_result), dtype=jnp.bool_
                        ),
                        has_pair_policy_flag=jnp.asarray(
                            has_pair_policy, dtype=jnp.bool_
                        ),
                        has_policy_state_flag=jnp.asarray(
                            has_policy_state, dtype=jnp.bool_
                        ),
                        leaf_count=jnp.asarray(leaf_count_planner, dtype=jnp.int32),
                        need_node_interactions_flag=jnp.asarray(
                            bool(need_node_interactions), dtype=jnp.bool_
                        ),
                        need_compact_far_pairs_flag=jnp.asarray(
                            bool(need_compact_far_pairs), dtype=jnp.bool_
                        ),
                        use_dense_interactions_flag=jnp.asarray(
                            bool(use_dense_interactions_for_prepare), dtype=jnp.bool_
                        ),
                    )
                    if suppress_host_side_effects:
                        use_split_build_compiled_bool = bool(allow_split_build)
                        suppress_substage_timing_compiled_bool = True
                    else:
                        use_split_build_compiled_bool = bool(use_split_build_compiled)
                        suppress_substage_timing_compiled_bool = bool(
                            suppress_substage_timing_compiled
                        )
                    if not suppress_host_side_effects:
                        self._refresh_dual_planner_compiled_route_count += 1
                    planner_hint = _RefreshDualPlannerHint(
                        use_split_build=use_split_build_compiled_bool,
                        suppress_substage_timing=suppress_substage_timing_compiled_bool,
                    )
                    if not suppress_host_side_effects:
                        self._refresh_dual_planner_cache[planner_key] = planner_hint
                        self._refresh_dual_planner_compile_count += 1
                else:
                    planner_cache_hit = True
                    if not suppress_host_side_effects:
                        self._refresh_dual_planner_cache_hits += 1
                if not suppress_host_side_effects:
                    self._refresh_dual_planner_execute_count += 1
        planner_allow_steady_timing_bypass = bool(
            self._planner_steady_timing_bypass_enabled
        )
        dual_artifact_timing_callback = _record_dual_artifact_substage
        if (
            planner_hint is not None
            and planner_cache_hit
            and bool(getattr(planner_hint, "suppress_substage_timing", False))
            and planner_allow_steady_timing_bypass
        ):
            dual_artifact_timing_callback = None
            if not suppress_host_side_effects:
                self._refresh_dual_planner_steady_timing_bypass_count += 1
        _record_dual_stage("_refresh_timing_dual_setup_seconds", stage_t0)

        stage_t0 = time.perf_counter()
        geometry_factory = (
            None
            if tree_artifacts.upward.geometry is not None
            else lambda: compute_tree_geometry(
                tree_artifacts.tree,
                tree_artifacts.positions_sorted,
                max_leaf_size=int(tree_artifacts.leaf_cap),
            )
        )
        dual_artifacts, cache_entry = _build_dual_tree_artifacts(
            tree_artifacts.tree,
            tree_artifacts.upward.geometry,
            geometry_factory=geometry_factory,
            theta=theta_val,
            mac_type=mac_type_val,
            dehnen_radius_scale=dehnen_radius_scale,
            cache_key=cache_key,
            cache_entry=(self._interaction_cache if stateful_cache_enabled else None),
            max_pair_queue=self.max_pair_queue,
            pair_process_block=self.pair_process_block,
            traversal_config=runtime_traversal_config,
            retry_logger=(
                None
                if strict_mode_active
                else (
                    record_retry
                    if bool(getattr(self, "_strict_cap_record_enabled", True))
                    else (None if jit_traversal_for_prepare else record_retry)
                )
            ),
            fail_fast=(self.fail_fast or strict_mode_active),
            use_dense_interactions=use_dense_interactions_for_prepare,
            grouped_interactions=grouped_interactions,
            grouped_chunk_size=runtime_m2l_chunk_size,
            need_traversal_result=need_traversal_result,
            need_compact_far_pairs=need_compact_far_pairs,
            need_node_interactions=need_node_interactions,
            precompute_grouped_class_segments=self._should_precompute_grouped_class_segments(
                grouped_chunk_size=runtime_m2l_chunk_size,
                farfield_mode=farfield_mode,
            ),
            grouped_schedule_budget_bytes=self._grouped_schedule_item_budget(),
            allow_split_build=allow_split_build,
            pair_policy=pair_policy,
            policy_state=policy_state,
            jit_traversal=jit_traversal_for_prepare,
            timing_callback=dual_artifact_timing_callback,
            planner_hint=planner_hint,
        )
        if stateful_cache_enabled:
            if bool(getattr(dual_artifacts, "cache_hit", False)):
                if not suppress_host_side_effects:
                    self._interaction_cache_hits += 1
            else:
                if not suppress_host_side_effects:
                    self._interaction_cache_misses += 1
        _record_dual_stage("_refresh_timing_dual_artifact_build_seconds", stage_t0)
        if stateful_cache_enabled:
            self._interaction_cache = cache_entry

        stage_t0 = _stage_now()
        (
            interactions,
            neighbor_list,
            traversal_result,
            compact_far_pairs,
            dense_buffers,
            grouped_buffers,
            grouped_segment_starts,
            grouped_segment_lengths,
            grouped_segment_class_ids,
            grouped_segment_sort_permutation,
            grouped_segment_group_ids,
            grouped_segment_unique_targets,
        ) = self._unpack_dual_tree_artifacts(dual_artifacts)
        if not suppress_host_side_effects:
            far_pair_count_diag = None
            if compact_far_pairs is not None:
                far_pair_count_diag = int(compact_far_pairs.sources.shape[0])
            elif interactions is not None:
                far_pair_count_diag = int(interactions.sources.shape[0])
            total_nodes_diag = int(tree_artifacts.tree.parent.shape[0])
            internal_nodes_diag = int(
                jnp.asarray(tree_artifacts.tree.left_child).shape[0]
            )
            leaf_count_diag = max(0, total_nodes_diag - internal_nodes_diag)
            self._recent_dual_node_count = int(total_nodes_diag)
            self._recent_dual_leaf_count = int(leaf_count_diag)
            self._recent_dual_neighbor_count = int(neighbor_list.neighbors.shape[0])
            self._recent_dual_far_pair_count = (
                0 if far_pair_count_diag is None else int(far_pair_count_diag)
            )
            _prepare_diag(
                "dual-tree done "
                f"neighbor_count={int(neighbor_list.neighbors.shape[0])} "
                f"far_pair_count={far_pair_count_diag} "
                f"compact_far_pairs={compact_far_pairs is not None} "
                f"interactions_present={interactions is not None}"
            )
            _prepare_diag(
                "dual-tree bytes "
                f"neighbors={_format_nbytes(_estimate_payload_nbytes(neighbor_list))} "
                f"compact_far_pairs={_format_nbytes(_estimate_payload_nbytes(compact_far_pairs))} "
                f"interactions={_format_nbytes(_estimate_payload_nbytes(interactions))} "
                f"dense_buffers={_format_nbytes(_estimate_payload_nbytes(dense_buffers))} "
                f"grouped_buffers={_format_nbytes(_estimate_payload_nbytes(grouped_buffers))}"
            )

        strict_streamed_direct_far_pairs = bool(
            strict_mode_active
            and bool(use_compact_streamed_pairs)
            and compact_far_pairs is not None
            and not bool(adaptive_order_active)
            and not bool(mixed_order_farfield_active)
        )
        if strict_streamed_direct_far_pairs:
            src_far = jnp.asarray(compact_far_pairs.sources, dtype=INDEX_DTYPE)
            tgt_far = jnp.asarray(compact_far_pairs.targets, dtype=INDEX_DTYPE)
            far_pairs_coo = _FarPairCOO(
                sources=src_far,
                targets=tgt_far,
                active_count=getattr(compact_far_pairs, "far_pair_count", None),
            )
            far_pairs_by_gear = ((src_far, tgt_far),)
            adaptive_order_for_downward = True
            p_gears_for_downward = (int(tree_artifacts.upward.multipoles.order),)
            if not suppress_host_side_effects:
                self._recent_far_pairs_by_gear_counts = (int(src_far.shape[0]),)
        else:
            far_pair_plan = self._prepare_state_plan_far_pairs_for_downward(
                interactions=interactions,
                traversal_result=traversal_result,
                compact_far_pairs=compact_far_pairs,
                upward=tree_artifacts.upward,
            )
            far_pairs_by_gear = far_pair_plan.far_pairs_by_gear
            far_pairs_coo = far_pair_plan.far_pairs_coo
            adaptive_order_for_downward = far_pair_plan.adaptive_order_for_downward
            p_gears_for_downward = far_pair_plan.p_gears_for_downward
            if not suppress_host_side_effects:
                self._recent_far_pairs_by_gear_counts = (
                    far_pair_plan.recent_far_pairs_by_gear_counts
                )
        _record_dual_stage("_refresh_timing_dual_far_pair_plan_seconds", stage_t0)

        stage_t0 = _stage_now()
        if suppress_host_side_effects and bool(
            getattr(self, "_static_runtime_fixed_sizing", True)
        ):
            runtime_m2l_chunk_size = runtime_m2l_chunk_size
        else:
            runtime_m2l_chunk_size = self._prepare_state_autotune_downward_chunk_size(
                upward=tree_artifacts.upward,
                far_pairs_by_gear=far_pairs_by_gear,
                p_gears_for_downward=p_gears_for_downward,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            )
        if not suppress_host_side_effects:
            self._recent_dual_m2l_chunk_size = (
                0 if runtime_m2l_chunk_size is None else int(runtime_m2l_chunk_size)
            )
        _record_dual_stage("_refresh_timing_dual_m2l_autotune_seconds", stage_t0)

        stage_t0 = _stage_now()
        interactions_for_downward = (
            None
            if strict_streamed_direct_far_pairs and not bool(retain_interactions_active)
            else self._prepare_state_select_interactions_for_downward(
                interactions=interactions,
                far_pairs_coo=far_pairs_coo,
            )
        )
        _record_dual_stage(
            "_refresh_timing_dual_select_interactions_seconds",
            stage_t0,
        )

        stage_t0 = _stage_now()
        downward = self._prepare_downward_with_artifacts(
            tree=tree_artifacts.tree,
            upward=tree_artifacts.upward,
            theta_val=theta_val,
            locals_template=tree_artifacts.locals_template,
            interactions=interactions_for_downward,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            runtime_traversal_config=runtime_traversal_config,
            record_retry=record_retry,
            dense_buffers=dense_buffers,
            grouped_interactions=grouped_interactions,
            grouped_buffers=grouped_buffers,
            grouped_segment_starts=grouped_segment_starts,
            grouped_segment_lengths=grouped_segment_lengths,
            grouped_segment_class_ids=grouped_segment_class_ids,
            grouped_segment_sort_permutation=grouped_segment_sort_permutation,
            grouped_segment_group_ids=grouped_segment_group_ids,
            grouped_segment_unique_targets=grouped_segment_unique_targets,
            farfield_mode=farfield_mode,
            far_pairs_coo=far_pairs_coo,
            far_pairs_by_gear=far_pairs_by_gear,
            adaptive_order=adaptive_order_for_downward,
            p_gears=p_gears_for_downward,
        )
        _record_dual_stage("_refresh_timing_dual_downward_compute_seconds", stage_t0)

        stage_t0 = _stage_now()
        if not suppress_host_side_effects:
            _prepare_diag(
                "downward done "
                f"locals_shape={tuple(int(v) for v in downward.locals.coefficients.shape)} "
                f"interactions_shape={tuple(int(v) for v in downward.interactions.sources.shape)}"
            )
            _prepare_diag(
                "downward bytes "
                f"locals={_format_nbytes(_estimate_payload_nbytes(downward.locals))} "
                f"stored_interactions={_format_nbytes(_estimate_payload_nbytes(downward.interactions))}"
            )
        if not bool(retain_interactions_active):
            # Prepared state only needs the locals; keep a shape-compatible
            # placeholder so downstream code does not accidentally pin the full
            # far-field pair payload after prepare_state completes.
            downward = downward._replace(
                interactions=(
                    _empty_interaction_storage_like(interactions)
                    if interactions is not None
                    else _empty_interaction_storage_for_tree(tree_artifacts.tree)
                )
            )
        interactions_out: Optional[NodeInteractionList]
        if bool(retain_interactions_active):
            interactions_out = interactions
        else:
            interactions_out = None
        _record_dual_stage("_refresh_timing_dual_finalize_seconds", stage_t0)
        if refresh_timing_active:
            residual = max(
                0.0,
                float(time.perf_counter() - dual_total_t0) - float(dual_stage_sum),
            )
            self._refresh_timing_dual_residual_seconds += residual
        return _PrepareStateDualDownwardArtifacts(
            interactions=interactions_out,
            neighbor_list=neighbor_list,
            traversal_result=(
                traversal_result if bool(need_traversal_result) else None
            ),
            compact_far_pairs=(
                compact_far_pairs
                if (
                    bool(adaptive_order_active)
                    or bool(strict_streamed_direct_far_pairs)
                )
                else None
            ),
            downward=downward,
            cache_entry=cache_entry,
        )

    def _prepare_state_extract_adaptive_far_pairs(
        self,
        *,
        traversal_result: Optional[DualTreeWalkResult],
        compact_far_pairs: Optional[CompactTaggedFarPairs],
    ) -> tuple[Array, Array, Array]:
        """Extract tagged far pairs for adaptive-order downward planning."""

        if len(self.p_gears) == 0:
            raise ValueError("adaptive_order=True requires non-empty p_gears")
        if traversal_result is not None:
            far_total = int(traversal_result.far_pair_count)
            far_sources = jnp.asarray(
                traversal_result.interaction_sources[:far_total], dtype=INDEX_DTYPE
            )
            far_targets = jnp.asarray(
                traversal_result.interaction_targets[:far_total], dtype=INDEX_DTYPE
            )
            far_tags = jnp.asarray(
                traversal_result.interaction_tags[:far_total], dtype=INDEX_DTYPE
            )
            return far_sources, far_targets, far_tags
        if compact_far_pairs is not None:
            return (
                jnp.asarray(compact_far_pairs.sources, dtype=INDEX_DTYPE),
                jnp.asarray(compact_far_pairs.targets, dtype=INDEX_DTYPE),
                jnp.asarray(compact_far_pairs.tags, dtype=INDEX_DTYPE),
            )
        raise RuntimeError("adaptive-order traversal requires tagged far-pair payload")

    def _prepare_state_build_streamed_far_pair_plan(
        self,
        *,
        interactions: Optional[NodeInteractionList],
        compact_far_pairs: Optional[CompactTaggedFarPairs],
        upward: TreeUpwardData,
    ) -> tuple[_FarPairCOO, tuple[tuple[Array, Array], ...], tuple[int, ...]]:
        """Build streamed far-pair payloads and per-gear buckets."""

        if compact_far_pairs is not None:
            src_far = jnp.asarray(compact_far_pairs.sources, dtype=INDEX_DTYPE)
            tgt_far = jnp.asarray(compact_far_pairs.targets, dtype=INDEX_DTYPE)
        else:
            if interactions is None:
                raise RuntimeError(
                    "streamed far-pair execution requires interactions or compact far pairs"
                )
            src_far = jnp.asarray(interactions.sources, dtype=INDEX_DTYPE)
            tgt_far = jnp.asarray(interactions.targets, dtype=INDEX_DTYPE)
        active_count = (
            getattr(compact_far_pairs, "far_pair_count", None)
            if compact_far_pairs is not None
            else None
        )
        far_pairs_coo = _FarPairCOO(
            sources=src_far,
            targets=tgt_far,
            active_count=active_count,
        )
        max_order_int = int(upward.multipoles.order)
        strict_fused_device_only_active = bool(
            getattr(self, "_strict_fused_mode_active", False)
        ) and bool(getattr(self, "_strict_fused_device_only", False))
        if (
            self.mixed_order_farfield
            and max_order_int >= 1
            and (not strict_fused_device_only_active)
        ):
            if interactions is None:
                raise RuntimeError(
                    "mixed-order streamed farfield requires node interaction lists"
                )
            min_order_candidate = (
                max_order_int - 1
                if self.mixed_order_min_order is None
                else int(self.mixed_order_min_order)
            )
            min_order_candidate = max(0, min(min_order_candidate, max_order_int))
            p_gears_for_downward, far_pairs_by_gear = _bucket_far_pairs_by_level_split(
                interactions=interactions,
                src_far=src_far,
                tgt_far=tgt_far,
                max_order=max_order_int,
                min_order=min_order_candidate,
            )
        else:
            p_gears_for_downward = (max_order_int,)
            far_pairs_by_gear = ((src_far, tgt_far),)
        return far_pairs_coo, far_pairs_by_gear, p_gears_for_downward

    def _prepare_state_plan_far_pairs_for_downward(
        self,
        *,
        interactions: Optional[NodeInteractionList],
        traversal_result: Optional[DualTreeWalkResult],
        compact_far_pairs: Optional[CompactTaggedFarPairs],
        upward: TreeUpwardData,
    ) -> _PrepareStateFarPairPlan:
        """Prepare far-pair payloads consumed by the downward sweep."""

        far_pairs_by_gear = None
        far_pairs_coo: Optional[_FarPairCOO] = None
        adaptive_order_for_downward = bool(self.adaptive_order)
        p_gears_for_downward = self.p_gears
        recent_counts: tuple[int, ...] = tuple()

        if self.adaptive_order:
            far_sources, far_targets, far_tags = (
                self._prepare_state_extract_adaptive_far_pairs(
                    traversal_result=traversal_result,
                    compact_far_pairs=compact_far_pairs,
                )
            )
            far_pairs_by_gear = bucket_far_pairs_by_tag(
                far_sources,
                far_targets,
                far_tags,
                num_tags=len(self.p_gears),
            )
            recent_counts = tuple(
                int(bucket_src.shape[0]) for bucket_src, _ in far_pairs_by_gear
            )
        elif self.streamed_far_pairs:
            (
                far_pairs_coo,
                far_pairs_by_gear,
                p_gears_for_downward,
            ) = self._prepare_state_build_streamed_far_pair_plan(
                interactions=interactions,
                compact_far_pairs=compact_far_pairs,
                upward=upward,
            )
            adaptive_order_for_downward = True
            recent_counts = tuple(
                int(bucket_src.shape[0]) for bucket_src, _ in far_pairs_by_gear
            )

        return _PrepareStateFarPairPlan(
            far_pairs_by_gear=far_pairs_by_gear,
            far_pairs_coo=far_pairs_coo,
            adaptive_order_for_downward=adaptive_order_for_downward,
            p_gears_for_downward=p_gears_for_downward,
            recent_far_pairs_by_gear_counts=recent_counts,
        )

    def _prepare_state_autotune_downward_chunk_size(
        self,
        *,
        upward: TreeUpwardData,
        far_pairs_by_gear: Optional[tuple[tuple[Array, Array], ...]],
        p_gears_for_downward: tuple[int, ...],
        runtime_m2l_chunk_size: Optional[int],
    ) -> Optional[int]:
        """Choose runtime M2L chunk size for the downward pass."""

        static_runtime_fixed_sizing = bool(
            getattr(self, "_static_runtime_fixed_sizing", True)
        )
        if static_runtime_fixed_sizing:
            return runtime_m2l_chunk_size

        if bool(getattr(self, "_strict_fused_mode_active", False)):
            return runtime_m2l_chunk_size

        if (
            runtime_m2l_chunk_size is not None
            or not bool(self.autotune_m2l_chunk)
            or self.expansion_basis != "solidfmm"
            or jax.default_backend() != "gpu"
            or far_pairs_by_gear is None
            or len(far_pairs_by_gear) == 0
        ):
            return runtime_m2l_chunk_size

        tune_idx = 0
        tune_pair_count = -1
        for idx, (src_bucket, _) in enumerate(far_pairs_by_gear):
            count_i = int(src_bucket.shape[0])
            if count_i > tune_pair_count:
                tune_idx = idx
                tune_pair_count = count_i
        if tune_pair_count <= 0:
            return runtime_m2l_chunk_size

        tune_src, tune_tgt = far_pairs_by_gear[tune_idx]
        tune_order = int(
            upward.multipoles.order
            if tune_idx >= len(p_gears_for_downward)
            else p_gears_for_downward[tune_idx]
        )
        tuned_chunk = self._autotune_runtime_m2l_chunk_size(
            upward=upward,
            src=tune_src,
            tgt=tune_tgt,
            order=tune_order,
            pair_count=tune_pair_count,
        )
        if tuned_chunk is not None:
            return int(tuned_chunk)
        return runtime_m2l_chunk_size

    def _prepare_state_select_interactions_for_downward(
        self,
        *,
        interactions: Optional[NodeInteractionList],
        far_pairs_coo: Optional[_FarPairCOO],
    ) -> Optional[NodeInteractionList]:
        """Choose whether downward should consume node interactions or COO pairs."""

        if (
            self.streamed_far_pairs
            and far_pairs_coo is not None
            and not bool(self.retain_interactions)
        ):
            # Streamed far-pair execution can feed the downward pass directly
            # from COO pairs, so avoid keeping a second node-structured
            # interaction list alive unless the caller asked to retain it.
            return None
        return interactions

    def _prepare_state_nearfield_artifacts(
        self,
        *,
        neighbor_list: NodeNeighborList,
        nearfield_interop: NearfieldInteropData,
        leaf_cap: int,
        num_particles: int,
        cache_entry: Optional[_InteractionCacheEntry],
        allow_stateful_cache: bool,
    ) -> NearfieldPrecomputeArtifacts:
        """Build/reuse near-field precompute artifacts for prepare_state."""
        effective_leaf_cap = (
            int(nearfield_interop.leaf_particle_indices.shape[1])
            if nearfield_interop.leaf_particle_indices is not None
            else int(leaf_cap)
        )
        nearfield_mode_resolved = self._resolve_nearfield_mode(
            num_particles=num_particles
        )
        retain_pair_vectors = (
            nearfield_mode_resolved == "bucketed"
            and self._should_retain_nearfield_pair_vectors(
                num_particles=int(num_particles)
            )
        )
        nearfield_edge_chunk_size_resolved = self._resolve_nearfield_edge_chunk_size(
            num_particles=num_particles,
            nearfield_mode=nearfield_mode_resolved,
        )
        if nearfield_cache_matches(
            cache_entry,
            nearfield_mode=nearfield_mode_resolved,
            nearfield_edge_chunk_size=nearfield_edge_chunk_size_resolved,
            leaf_cap=int(leaf_cap),
            require_pair_vectors=bool(retain_pair_vectors),
        ):
            return nearfield_from_cache(cache_entry)

        nearfield_artifacts = self._prepare_nearfield_precompute_artifacts(
            neighbor_list=neighbor_list,
            nearfield_interop=nearfield_interop,
            leaf_cap=effective_leaf_cap,
            num_particles=num_particles,
            nearfield_mode=nearfield_mode_resolved,
            nearfield_edge_chunk_size=nearfield_edge_chunk_size_resolved,
            retain_pair_vectors=retain_pair_vectors,
        )
        if allow_stateful_cache and cache_entry is not None:
            self._interaction_cache = with_nearfield_cache_artifacts(
                cache_entry,
                artifacts=nearfield_artifacts,
                nearfield_mode=nearfield_mode_resolved,
                nearfield_edge_chunk_size=nearfield_edge_chunk_size_resolved,
                leaf_cap=int(leaf_cap),
            )
        return nearfield_artifacts

    def _update_locals_template_cache_after_prepare(
        self,
        *,
        locals_template: Optional[LocalExpansionData],
        upward: TreeUpwardData,
        max_order: int,
    ) -> None:
        """Update reusable cartesian local-expansion template after prepare_state."""
        if locals_template is not None and self.expansion_basis == "cartesian":
            self._locals_template = LocalExpansionData(
                order=max_order,
                centers=upward.multipoles.centers,
                coefficients=jnp.zeros_like(locals_template.coefficients),
            )
            return
        self._locals_template = None

    def _prepare_nearfield_precompute_artifacts(
        self,
        *,
        neighbor_list: NodeNeighborList,
        nearfield_interop: NearfieldInteropData,
        leaf_cap: int,
        num_particles: int,
        nearfield_mode: Optional[str] = None,
        nearfield_edge_chunk_size: Optional[int] = None,
        retain_pair_vectors: Optional[bool] = None,
    ) -> NearfieldPrecomputeArtifacts:
        """Best-effort precompute of nearfield leaf-pair and scatter artifacts."""
        nearfield_chunk_sort_indices = None
        nearfield_chunk_group_ids = None
        nearfield_chunk_unique_indices = None

        resolved_nearfield_mode = (
            self._resolve_nearfield_mode(num_particles=num_particles)
            if nearfield_mode is None
            else str(nearfield_mode).strip().lower()
        )
        resolved_nearfield_edge_chunk_size = (
            self._resolve_nearfield_edge_chunk_size(
                num_particles=num_particles,
                nearfield_mode=resolved_nearfield_mode,
            )
            if nearfield_edge_chunk_size is None
            else int(nearfield_edge_chunk_size)
        )
        retain_pair_vectors_resolved = (
            self._should_retain_nearfield_pair_vectors(num_particles=int(num_particles))
            if retain_pair_vectors is None
            else bool(retain_pair_vectors)
        )

        should_precompute_scatter = self._should_precompute_nearfield_scatter_schedules(
            num_particles=int(num_particles)
        )
        if not bool(retain_pair_vectors_resolved) and not bool(
            should_precompute_scatter
        ):
            # In large-N minimum-memory GPU runs, prepared evaluation can derive
            # near-field pair vectors on demand from the neighbor list. Avoid
            # materializing enormous edge-index buffers during prepare_state
            # when we are neither retaining them nor building scatter schedules.
            return NearfieldPrecomputeArtifacts(
                target_leaf_ids=None,
                source_leaf_ids=None,
                valid_pairs=None,
                chunk_sort_indices=None,
                chunk_group_ids=None,
                chunk_unique_indices=None,
            )
        if resolved_nearfield_mode != "bucketed":
            return NearfieldPrecomputeArtifacts(
                target_leaf_ids=None,
                source_leaf_ids=None,
                valid_pairs=None,
                chunk_sort_indices=None,
                chunk_group_ids=None,
                chunk_unique_indices=None,
            )

        nearfield_target_leaf_ids = None
        nearfield_valid_pairs = None

        (
            nearfield_target_leaf_ids,
            nearfield_source_leaf_ids,
            nearfield_valid_pairs,
        ) = self._prepare_leaf_neighbor_pairs_safe(
            nearfield_interop=nearfield_interop,
        )

        traced_nearfield_pairs = False
        if nearfield_target_leaf_ids is not None and nearfield_valid_pairs is not None:
            traced_nearfield_pairs = _contains_tracer(
                (nearfield_target_leaf_ids, nearfield_valid_pairs)
            )

        if (
            nearfield_target_leaf_ids is not None
            and nearfield_valid_pairs is not None
            and not traced_nearfield_pairs
            and bool(should_precompute_scatter)
        ):
            (
                nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices,
            ) = self._prepare_bucketed_scatter_schedules_safe(
                nearfield_interop=nearfield_interop,
                target_leaf_ids=nearfield_target_leaf_ids,
                valid_pairs=nearfield_valid_pairs,
                leaf_cap=int(leaf_cap),
                edge_chunk_size=resolved_nearfield_edge_chunk_size,
            )

        return NearfieldPrecomputeArtifacts(
            target_leaf_ids=(
                nearfield_target_leaf_ids if retain_pair_vectors_resolved else None
            ),
            source_leaf_ids=(
                nearfield_source_leaf_ids if retain_pair_vectors_resolved else None
            ),
            valid_pairs=(
                nearfield_valid_pairs if retain_pair_vectors_resolved else None
            ),
            chunk_sort_indices=nearfield_chunk_sort_indices,
            chunk_group_ids=nearfield_chunk_group_ids,
            chunk_unique_indices=nearfield_chunk_unique_indices,
        )

    def _should_retain_nearfield_pair_vectors(
        self,
        *,
        num_particles: int,
    ) -> bool:
        """Decide whether prepared-state near-field pair vectors should be retained."""
        if self.memory_objective == "minimum_memory":
            return False
        if jax.default_backend() != "gpu":
            return True
        return int(num_particles) <= int(_NEARFIELD_GPU_PRECOMPUTE_MAX_PARTICLES)

    def _should_precompute_nearfield_scatter_schedules(
        self,
        *,
        num_particles: int,
    ) -> bool:
        """Decide whether to materialize near-field scatter schedules."""
        if not bool(self.precompute_nearfield_scatter_schedules):
            return False
        if jax.default_backend() != "gpu":
            return True
        # For large GPU runs, schedule materialization is often memory-dominant;
        # keep execution streamed/chunked instead.
        return int(num_particles) <= int(_NEARFIELD_GPU_PRECOMPUTE_MAX_PARTICLES)

    def _prepare_leaf_neighbor_pairs_safe(
        self,
        *,
        nearfield_interop: NearfieldInteropData,
    ) -> tuple[Optional[Array], Optional[Array], Optional[Array]]:
        """Best-effort leaf neighbor pair generation."""
        try:
            return prepare_leaf_neighbor_pairs(
                jnp.asarray(nearfield_interop.node_ranges, dtype=INDEX_DTYPE),
                jnp.asarray(nearfield_interop.leaf_nodes, dtype=INDEX_DTYPE),
                jnp.asarray(nearfield_interop.offsets, dtype=INDEX_DTYPE),
                jnp.asarray(nearfield_interop.neighbors, dtype=INDEX_DTYPE),
                # Keep edge order aligned with neighbor_list so source leaf ids can
                # be derived on demand without storing a second index vector.
                sort_by_source=False,
            )
        except Exception:
            return None, None, None

    def _prepare_bucketed_scatter_schedules_safe(
        self,
        *,
        nearfield_interop: NearfieldInteropData,
        target_leaf_ids: Array,
        valid_pairs: Array,
        leaf_cap: int,
        edge_chunk_size: int,
    ) -> tuple[Optional[Array], Optional[Array], Optional[Array]]:
        """Best-effort bucketed scatter schedule generation."""
        try:
            edge_count = int(target_leaf_ids.shape[0])
            chunk = int(edge_chunk_size)
            chunk_count = (edge_count + chunk - 1) // chunk if edge_count > 0 else 0
            schedule_items = int(chunk_count * chunk * int(leaf_cap))
            if schedule_items > int(_NEARFIELD_SCATTER_SCHEDULE_INT32_ITEM_LIMIT):
                return None, None, None
            schedule_item_cap = self._resolve_nearfield_schedule_item_cap(
                edge_count=edge_count,
                leaf_cap=int(leaf_cap),
                edge_chunk_size=chunk,
            )
            if schedule_items > int(schedule_item_cap):
                return None, None, None
            if nearfield_interop.leaf_particle_indices is not None:
                return prepare_bucketed_scatter_schedules_from_groups(
                    jnp.asarray(
                        nearfield_interop.leaf_particle_indices,
                        dtype=INDEX_DTYPE,
                    ),
                    jnp.asarray(nearfield_interop.leaf_particle_mask, dtype=bool),
                    target_leaf_ids,
                    valid_pairs,
                    edge_chunk_size=chunk,
                )
            return prepare_bucketed_scatter_schedules(
                jnp.asarray(nearfield_interop.node_ranges, dtype=INDEX_DTYPE),
                jnp.asarray(nearfield_interop.leaf_nodes, dtype=INDEX_DTYPE),
                target_leaf_ids,
                valid_pairs,
                max_leaf_size=int(leaf_cap),
                edge_chunk_size=chunk,
            )
        except Exception:
            return None, None, None

    def _resolve_nearfield_schedule_item_cap(
        self,
        *,
        edge_count: int,
        leaf_cap: int,
        edge_chunk_size: int,
    ) -> int:
        """Return the max near-field schedule items allowed for this workload."""
        del edge_count, leaf_cap, edge_chunk_size
        if self.nearfield_schedule_item_cap is not None:
            return int(self.nearfield_schedule_item_cap)

        backend_name = jax.default_backend()
        base_cap = (
            _NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP_GPU
            if backend_name == "gpu"
            else _NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP
        )
        if self.memory_objective == "minimum_memory":
            base_cap = max(1, base_cap // 4)
        elif self.memory_objective == "throughput":
            base_cap = base_cap * 2

        if self.memory_budget_bytes is not None:
            bytes_per_item = 3 * np.dtype(np.int32).itemsize
            budget_limited_cap = max(1, int(self.memory_budget_bytes) // bytes_per_item)
            base_cap = min(base_cap, budget_limited_cap)
        return int(base_cap)

    def _should_precompute_grouped_class_segments(
        self,
        *,
        grouped_chunk_size: Optional[int],
        farfield_mode: str,
    ) -> bool:
        """Decide whether grouped class-major schedules should be materialized."""
        if grouped_chunk_size is None:
            return False
        if str(farfield_mode).strip().lower() != "class_major":
            return False
        if self.precompute_grouped_class_segments is not None:
            return bool(self.precompute_grouped_class_segments)
        return self.memory_objective != "minimum_memory"

    def _grouped_schedule_item_budget(self) -> int:
        """Return max bytes allowed for cached grouped schedule matrices."""
        return int(self.grouped_schedule_budget_bytes)

    def _resolve_target_indices(
        self,
        *,
        target_indices: Optional[Array],
        num_particles: int,
    ) -> Optional[Array]:
        """Validate and normalize optional target particle indices."""
        if target_indices is None:
            return None
        indices = jnp.asarray(target_indices)
        if indices.ndim != 1:
            raise ValueError("target_indices must be a 1D array")
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise ValueError("target_indices must contain integer values")
        if indices.shape[0] == 0:
            return indices.astype(INDEX_DTYPE)
        # Under JAX tracing we cannot materialize min/max as Python ints.
        if isinstance(indices, jax.core.Tracer):
            return indices.astype(INDEX_DTYPE)
        min_idx = int(jnp.min(indices))
        max_idx = int(jnp.max(indices))
        if min_idx < 0 or max_idx >= int(num_particles):
            raise ValueError("target_indices contains out-of-range values")
        return indices.astype(INDEX_DTYPE)

    def _unpack_dual_tree_artifacts(
        self,
        dual_artifacts: _DualTreeArtifacts,
    ) -> tuple[
        NodeInteractionList,
        NodeNeighborList,
        Optional[DualTreeWalkResult],
        Optional[CompactTaggedFarPairs],
        Optional[DenseInteractionBuffers],
        Optional[GroupedInteractionBuffers],
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
    ]:
        """Unpack dual-tree artifacts for downward preparation and state export."""
        return (
            dual_artifacts.interactions,
            dual_artifacts.neighbor_list,
            dual_artifacts.traversal_result,
            dual_artifacts.compact_far_pairs,
            dual_artifacts.dense_buffers,
            dual_artifacts.grouped_buffers,
            dual_artifacts.grouped_segment_starts,
            dual_artifacts.grouped_segment_lengths,
            dual_artifacts.grouped_segment_class_ids,
            dual_artifacts.grouped_segment_sort_permutation,
            dual_artifacts.grouped_segment_group_ids,
            dual_artifacts.grouped_segment_unique_targets,
        )

    def _prepare_downward_with_artifacts(
        self,
        *,
        tree: Tree,
        upward: TreeUpwardData,
        theta_val: float,
        locals_template: Optional[LocalExpansionData],
        interactions: Optional[NodeInteractionList],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        record_retry: Callable[[DualTreeRetryEvent], None],
        dense_buffers: Optional[DenseInteractionBuffers],
        grouped_interactions: bool,
        grouped_buffers: Optional[GroupedInteractionBuffers],
        grouped_segment_starts: Optional[Array],
        grouped_segment_lengths: Optional[Array],
        grouped_segment_class_ids: Optional[Array],
        grouped_segment_sort_permutation: Optional[Array],
        grouped_segment_group_ids: Optional[Array],
        grouped_segment_unique_targets: Optional[Array],
        farfield_mode: str,
        far_pairs_coo: Optional[_FarPairCOO] = None,
        far_pairs_by_gear: Optional[tuple[tuple[Array, Array], ...]] = None,
        adaptive_order: bool = False,
        p_gears: tuple[int, ...] = tuple(),
    ) -> TreeDownwardData:
        """Prepare downward sweep using precomputed interaction artifacts."""
        return self.prepare_downward_sweep(
            tree,
            upward,
            theta=theta_val,
            initial_locals=locals_template,
            interactions=interactions,
            m2l_chunk_size=runtime_m2l_chunk_size,
            l2l_chunk_size=runtime_l2l_chunk_size,
            traversal_config=runtime_traversal_config,
            retry_logger=record_retry,
            dense_buffers=dense_buffers,
            grouped_interactions=grouped_interactions,
            grouped_buffers=grouped_buffers,
            grouped_segment_starts=grouped_segment_starts,
            grouped_segment_lengths=grouped_segment_lengths,
            grouped_segment_class_ids=grouped_segment_class_ids,
            grouped_segment_sort_permutation=grouped_segment_sort_permutation,
            grouped_segment_group_ids=grouped_segment_group_ids,
            grouped_segment_unique_targets=grouped_segment_unique_targets,
            farfield_mode=farfield_mode,
            far_pairs_coo=far_pairs_coo,
            far_pairs_by_gear=far_pairs_by_gear,
            adaptive_order=adaptive_order,
            p_gears=p_gears,
        )

    def _prepare_state_dual_and_downward_strict_streamed_fast(
        self,
        *,
        tree_artifacts: _PrepareStateTreeUpwardArtifacts,
        theta_val: float,
        mac_type_val: MACType,
        dehnen_radius_scale: float,
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        record_retry: Callable[[DualTreeRetryEvent], None],
        farfield_mode: str,
        retain_interactions: bool = False,
        suppress_host_side_effects: bool = False,
    ) -> _PrepareStateDualDownwardArtifacts:
        """Strict static fast path with compact streamed far-pairs only."""

        geometry_factory = (
            None
            if tree_artifacts.upward.geometry is not None
            else lambda: compute_tree_geometry(
                tree_artifacts.tree,
                tree_artifacts.positions_sorted,
                max_leaf_size=int(tree_artifacts.leaf_cap),
            )
        )
        dual_artifacts, cache_entry = _build_dual_tree_artifacts(
            tree_artifacts.tree,
            tree_artifacts.upward.geometry,
            geometry_factory=geometry_factory,
            theta=theta_val,
            mac_type=mac_type_val,
            dehnen_radius_scale=dehnen_radius_scale,
            cache_key=None,
            cache_entry=None,
            max_pair_queue=self.max_pair_queue,
            pair_process_block=self.pair_process_block,
            traversal_config=runtime_traversal_config,
            retry_logger=None,
            fail_fast=True,
            use_dense_interactions=False,
            grouped_interactions=False,
            grouped_chunk_size=runtime_m2l_chunk_size,
            need_traversal_result=False,
            need_compact_far_pairs=True,
            need_node_interactions=False,
            precompute_grouped_class_segments=False,
            grouped_schedule_budget_bytes=self._grouped_schedule_item_budget(),
            allow_split_build=True,
            pair_policy=None,
            policy_state=None,
            jit_traversal=True,
            timing_callback=None,
            planner_hint=_RefreshDualPlannerHint(
                use_split_build=True,
                suppress_substage_timing=True,
            ),
        )
        (
            interactions,
            neighbor_list,
            traversal_result,
            compact_far_pairs,
            dense_buffers,
            grouped_buffers,
            grouped_segment_starts,
            grouped_segment_lengths,
            grouped_segment_class_ids,
            grouped_segment_sort_permutation,
            grouped_segment_group_ids,
            grouped_segment_unique_targets,
        ) = self._unpack_dual_tree_artifacts(dual_artifacts)
        del (
            interactions,
            traversal_result,
            dense_buffers,
            grouped_buffers,
            grouped_segment_starts,
            grouped_segment_lengths,
            grouped_segment_class_ids,
            grouped_segment_sort_permutation,
            grouped_segment_group_ids,
            grouped_segment_unique_targets,
        )
        if compact_far_pairs is None:
            raise RuntimeError(
                "strict streamed fast path requires compact far-pair artifacts"
            )
        if not suppress_host_side_effects:
            total_nodes_diag = int(tree_artifacts.tree.parent.shape[0])
            internal_nodes_diag = int(
                jnp.asarray(tree_artifacts.tree.left_child).shape[0]
            )
            leaf_count_diag = max(0, total_nodes_diag - internal_nodes_diag)
            self._recent_dual_node_count = int(total_nodes_diag)
            self._recent_dual_leaf_count = int(leaf_count_diag)
            self._recent_dual_neighbor_count = int(neighbor_list.neighbors.shape[0])
            self._recent_dual_far_pair_count = int(compact_far_pairs.sources.shape[0])

        src_far = jnp.asarray(compact_far_pairs.sources, dtype=INDEX_DTYPE)
        tgt_far = jnp.asarray(compact_far_pairs.targets, dtype=INDEX_DTYPE)
        far_pairs_coo = _FarPairCOO(
            sources=src_far,
            targets=tgt_far,
            active_count=getattr(compact_far_pairs, "far_pair_count", None),
        )
        far_pairs_by_gear: tuple[tuple[Array, Array], ...] = ((src_far, tgt_far),)
        p_gears_for_downward = (int(tree_artifacts.upward.multipoles.order),)
        if not suppress_host_side_effects:
            self._recent_far_pairs_by_gear_counts = (int(src_far.shape[0]),)

        if suppress_host_side_effects and bool(
            getattr(self, "_static_runtime_fixed_sizing", True)
        ):
            runtime_m2l_chunk_size = runtime_m2l_chunk_size
        else:
            runtime_m2l_chunk_size = self._prepare_state_autotune_downward_chunk_size(
                upward=tree_artifacts.upward,
                far_pairs_by_gear=far_pairs_by_gear,
                p_gears_for_downward=p_gears_for_downward,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            )
        if not suppress_host_side_effects:
            self._recent_dual_m2l_chunk_size = (
                0 if runtime_m2l_chunk_size is None else int(runtime_m2l_chunk_size)
            )
        downward = self._prepare_downward_with_artifacts(
            tree=tree_artifacts.tree,
            upward=tree_artifacts.upward,
            theta_val=theta_val,
            locals_template=tree_artifacts.locals_template,
            interactions=None,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            runtime_traversal_config=runtime_traversal_config,
            record_retry=record_retry,
            dense_buffers=None,
            grouped_interactions=False,
            grouped_buffers=None,
            grouped_segment_starts=None,
            grouped_segment_lengths=None,
            grouped_segment_class_ids=None,
            grouped_segment_sort_permutation=None,
            grouped_segment_group_ids=None,
            grouped_segment_unique_targets=None,
            farfield_mode=farfield_mode,
            far_pairs_coo=far_pairs_coo,
            far_pairs_by_gear=far_pairs_by_gear,
            adaptive_order=True,
            p_gears=p_gears_for_downward,
        )
        if not bool(retain_interactions):
            downward = downward._replace(
                interactions=_empty_interaction_storage_for_tree(tree_artifacts.tree)
            )
        return _PrepareStateDualDownwardArtifacts(
            interactions=None,
            neighbor_list=neighbor_list,
            traversal_result=None,
            compact_far_pairs=compact_far_pairs,
            downward=downward,
            cache_entry=cache_entry,
        )

    # ------------------------------------------------------------------
    # Expansion construction up to a given order
    # order=0: monopole, order=1: +dipole, order=2: +quadrupole
    # order=3: +octupole, order=4: +hexadecapole
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Expansion evaluation up to a given order
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Direct summation fallback (for validation / small N)
    # ------------------------------------------------------------------

    @jaxtyped(typechecker=beartype)
    def compute_accelerations(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        *,
        target_indices: Optional[Array] = None,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        return_potential: bool = False,
        theta: Optional[float] = None,
        jit_tree: Optional[bool] = None,
        refine_local: Optional[bool] = None,
        max_refine_levels: Optional[int] = None,
        aspect_threshold: Optional[float] = None,
        jit_traversal: Optional[bool] = None,
        reuse_prepared_state: bool = False,
        max_acc_derivative_order: int = 0,
    ) -> Union[
        Array,
        Tuple[Array, Array],
        Tuple[Array, PackedAccelerationDerivatives],
        Tuple[Array, Array, PackedAccelerationDerivatives],
    ]:
        """Run the full FMM pipeline for particle accelerations.

        Parameters
        ----------
        positions : Array
            Source and target particle positions.
        masses : Array
            Particle masses aligned with ``positions``.
        target_indices : Optional[Array]
            Optional 1D index array selecting which target-particle outputs to
            return. All particles are still used as source masses.
        bounds : Optional[Tuple[Array, Array]]
            Optional explicit domain bounds used during tree construction.
        leaf_size : int
            Target maximum particle count per leaf for the prepared tree.
        max_order : int
            Multipole/local expansion order used for the upward and downward
            passes.
        return_potential : bool
            When ``True``, return a tuple ``(accelerations, potentials)``.
        theta : Optional[float]
            Optional per-call MAC opening angle override.
        jit_tree : Optional[bool]
            When ``True``, specialise tree construction via JIT to amortise
            repeated builds for consistent tree sizes.
        refine_local : Optional[bool]
            Override the fixed-depth builder's local refinement toggle when
            ``tree_build_mode`` is ``"fixed_depth"``.
        max_refine_levels : Optional[int]
            Maximum local refinement iterations passed to the builder.
        aspect_threshold : Optional[float]
            Aspect ratio threshold that triggers additional splits in the
            refinement pass.
        jit_traversal : Optional[bool]
            When ``True``, evaluate the traversal/evaluation path with the
            compiled implementation for improved throughput.
        reuse_prepared_state : bool
            Reuse the most recent prepared state when identical array objects
            and preparation parameters are provided.

        Returns
        -------
        Union[Array, Tuple[Array, Array]]
            Accelerations for all particles or selected targets. When
            ``return_potential`` is ``True``, also returns the potential.
        """

        cache_key: Optional[tuple[Any, ...]] = None
        state: Optional[FMMPreparedState] = None
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        if _contains_tracer((positions_arr, masses_arr)):
            if bool(return_potential):
                raise NotImplementedError(
                    "traced compute_accelerations fallback does not support return_potential=True"
                )
            if int(max_acc_derivative_order) != 0:
                raise NotImplementedError(
                    "traced compute_accelerations fallback does not support acceleration derivatives"
                )
            target_positions = (
                positions_arr
                if target_indices is None
                else jnp.asarray(
                    positions_arr[jnp.asarray(target_indices, dtype=INDEX_DTYPE),],
                    dtype=positions_arr.dtype,
                )
            )
            return jax.vmap(
                lambda eval_point: reference_direct_sum(
                    positions_arr,
                    masses_arr,
                    eval_point,
                    G=self.G,
                    softening=self.softening,
                )
            )(target_positions)
        if reuse_prepared_state:
            if bounds is None:
                bounds_key: tuple[Any, ...] = ("none",)
            else:
                bounds_key = ("set", id(bounds[0]), id(bounds[1]))
            cache_key = (
                positions_arr.shape,
                str(positions_arr.dtype),
                masses_arr.shape,
                str(masses_arr.dtype),
                bounds_key,
                int(leaf_size),
                int(max_order),
                None if theta is None else float(theta),
                None if jit_tree is None else bool(jit_tree),
                None if refine_local is None else bool(refine_local),
                None if max_refine_levels is None else int(max_refine_levels),
                None if aspect_threshold is None else float(aspect_threshold),
            )
            state = self._prepared_state_cache_lookup(
                key=cache_key,
                positions=positions_arr,
                masses=masses_arr,
            )

        if state is None:
            state = self.prepare_state(
                positions,
                masses,
                bounds=bounds,
                leaf_size=leaf_size,
                max_order=max_order,
                theta=theta,
                jit_tree=jit_tree,
                refine_local=refine_local,
                max_refine_levels=max_refine_levels,
                aspect_threshold=aspect_threshold,
            )
            if reuse_prepared_state and cache_key is not None:
                self._prepared_state_cache_store(
                    key=cache_key,
                    positions=positions_arr,
                    masses=masses_arr,
                    state=state,
                )

        jit_traversal_flag = (
            self._jit_traversal_default
            if jit_traversal is None
            else bool(jit_traversal)
        )

        evaluation = self.evaluate_prepared_state(
            state,
            target_indices=target_indices,
            return_potential=return_potential,
            jit_traversal=jit_traversal_flag,
            max_acc_derivative_order=max_acc_derivative_order,
        )
        return evaluation

    @jaxtyped(typechecker=beartype)
    def prepare_state(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        theta: Optional[float] = None,
        jit_tree: Optional[bool] = None,
        refine_local: Optional[bool] = None,
        max_refine_levels: Optional[int] = None,
        aspect_threshold: Optional[float] = None,
        runtime_overrides_override: Optional[_RuntimeExecutionOverrides] = None,
        fused_device_mode: bool = False,
    ) -> PreparedStateLike:
        """Precompute tree and interaction data for repeated evaluations.

        When ``tree_build_mode`` is ``"fixed_depth"`` the optional
        ``refine_local``, ``max_refine_levels``, and ``aspect_threshold``
        arguments control the host-side leaf refinement pass.
        """

        self._validate_prepare_state_request(
            leaf_size=int(leaf_size),
            max_order=int(max_order),
        )

        refine_local_val = (
            self.refine_local if refine_local is None else bool(refine_local)
        )
        max_refine_levels_val = (
            self.max_refine_levels
            if max_refine_levels is None
            else int(max_refine_levels)
        )
        aspect_threshold_val = (
            self.aspect_threshold
            if aspect_threshold is None
            else float(aspect_threshold)
        )

        collected_retries: list[DualTreeRetryEvent] = []

        def record_retry(event: DualTreeRetryEvent) -> None:
            collected_retries.append(event)
            if self.interaction_retry_logger is not None:
                self.interaction_retry_logger(event)

        input_t0 = time.perf_counter()
        positions_arr, masses_arr, input_dtype = self._prepare_state_input_arrays(
            positions,
            masses,
        )
        if bool(getattr(self, "_refresh_timing_active", False)):
            self._refresh_timing_input_seconds += time.perf_counter() - input_t0
        allow_stateful_cache = not _contains_tracer((positions_arr, masses_arr))

        runtime_overrides = runtime_overrides_override
        if runtime_overrides is None:
            runtime_overrides = self._resolve_runtime_execution_overrides(
                num_particles=int(positions_arr.shape[0]),
            )
        runtime_traversal_config = runtime_overrides.traversal_config
        runtime_m2l_chunk_size = runtime_overrides.m2l_chunk_size
        runtime_l2l_chunk_size = runtime_overrides.l2l_chunk_size
        grouped_interactions = runtime_overrides.grouped_interactions
        farfield_mode = runtime_overrides.farfield_mode
        upward_center_mode = runtime_overrides.center_mode
        if refine_local is None and runtime_overrides.refine_local_override is not None:
            refine_local_val = bool(runtime_overrides.refine_local_override)
        if not allow_stateful_cache:
            runtime_traversal_config = self._resolve_tracing_traversal_config(
                traversal_config=runtime_traversal_config,
            )

        theta_val = float(self.theta if theta is None else theta)
        mac_type_val = self._base_mac_type()

        if can_use_large_n_prepare_path(
            self,
            positions_arr=positions_arr,
            masses_arr=masses_arr,
            allow_stateful_cache=allow_stateful_cache,
        ):
            return prepare_large_n_state(
                self,
                positions_arr=positions_arr,
                masses_arr=masses_arr,
                input_dtype=input_dtype,
                bounds=bounds,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta_val=theta_val,
                mac_type_val=mac_type_val,
                refine_local_val=refine_local_val,
                max_refine_levels_val=max_refine_levels_val,
                aspect_threshold_val=aspect_threshold_val,
                jit_tree_override=jit_tree,
                allow_stateful_cache=allow_stateful_cache,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                upward_center_mode=upward_center_mode,
                record_retry=record_retry,
                collected_retries=collected_retries,
                fused_device_mode=bool(fused_device_mode),
            )

        tree_artifacts = self._prepare_state_tree_and_upward(
            positions_arr=positions_arr,
            masses_arr=masses_arr,
            bounds=bounds,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            refine_local_val=refine_local_val,
            max_refine_levels_val=max_refine_levels_val,
            aspect_threshold_val=aspect_threshold_val,
            jit_tree_override=jit_tree,
            upward_center_mode=upward_center_mode,
            allow_stateful_cache=allow_stateful_cache,
        )
        force_scale_nodes = None
        use_paper_force_scale = self._uses_paper_style_force_scale()
        if use_paper_force_scale:
            node_count = int(tree_artifacts.tree.parent.shape[0])
            previous_force_scale = self._last_force_scale_nodes
            reduction_mode = self._force_scale_reduction_mode()
            need_prepass = False
            policy_orders = self._policy_orders_for_prepare_state(
                max_order=int(max_order)
            )
            if self.mac_force_scale_mode == "paper":
                need_prepass = True
            elif self.mac_force_scale_mode == "prev" or self._in_force_scale_prepass:
                if (
                    previous_force_scale is not None
                    and int(previous_force_scale.shape[0]) == node_count
                ):
                    force_scale_nodes = jnp.asarray(
                        previous_force_scale,
                        dtype=positions_arr.dtype,
                    )
                elif self._uses_paper_style_traversal_policy() and (
                    not self._in_force_scale_prepass
                ):
                    need_prepass = True
                else:
                    force_scale_nodes = jnp.ones(
                        (node_count,),
                        dtype=positions_arr.dtype,
                    )
            else:
                need_prepass = True
            if need_prepass:
                if len(policy_orders) == 0:
                    raise ValueError(
                        "mac_force_scale_mode='prepass' requires non-empty orders"
                    )
                low_order = int(min(policy_orders))
                if self.mac_force_scale_mode == "paper" and (
                    self._uses_paper_style_traversal_policy()
                ):
                    low_order = 1 if int(max_order) >= 1 else 0
                if self.mac_force_scale_mode == "paper" and (
                    self._uses_paper_style_traversal_policy()
                ):
                    prepass_sorted = (
                        self._compute_force_scale_paper_prepass_from_tree_artifacts(
                            tree_artifacts=tree_artifacts,
                            low_order=low_order,
                            theta_val=theta_val,
                            upward_center_mode=upward_center_mode,
                            runtime_traversal_config=runtime_traversal_config,
                            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                            grouped_interactions=grouped_interactions,
                            farfield_mode=farfield_mode,
                            record_retry=record_retry,
                            refine_local_val=refine_local_val,
                            max_refine_levels_val=max_refine_levels_val,
                            aspect_threshold_val=aspect_threshold_val,
                        )
                    )
                else:
                    self._in_force_scale_prepass = True
                    saved_p_gears = self.p_gears
                    saved_adaptive_order = self.adaptive_order
                    saved_adaptive_error_model = self.adaptive_error_model
                    saved_mac_type = self.mac_type
                    try:
                        self.p_gears = (low_order,)
                        self.adaptive_order = bool(saved_adaptive_order)
                        prepass_acc = self.compute_accelerations(
                            positions_arr,
                            masses_arr,
                            bounds=bounds,
                            leaf_size=int(leaf_size),
                            max_order=low_order,
                            return_potential=False,
                            theta=theta_val,
                            reuse_prepared_state=False,
                            jit_tree=jit_tree,
                            jit_traversal=False,
                        )
                    finally:
                        self.p_gears = saved_p_gears
                        self.adaptive_order = saved_adaptive_order
                        self.adaptive_error_model = saved_adaptive_error_model
                        self.mac_type = saved_mac_type
                        self._in_force_scale_prepass = False
                    prepass_sorted = jnp.asarray(prepass_acc)[
                        jnp.argsort(tree_artifacts.inverse_permutation)
                    ]
                force_scale_nodes = self._compute_node_force_scale_from_sorted_acc(
                    tree=tree_artifacts.tree,
                    accelerations_sorted=prepass_sorted,
                    reduction=reduction_mode,
                ).astype(positions_arr.dtype)
                self._last_force_scale_nodes = force_scale_nodes
        dual_downward_artifacts = self._prepare_state_dual_and_downward(
            tree_artifacts=tree_artifacts,
            force_scale_nodes=force_scale_nodes,
            upward_center_mode=upward_center_mode,
            theta_val=theta_val,
            mac_type_val=mac_type_val,
            dehnen_radius_scale=self.dehnen_radius_scale,
            runtime_traversal_config=runtime_traversal_config,
            grouped_interactions=grouped_interactions,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            farfield_mode=farfield_mode,
            record_retry=record_retry,
            refine_local_val=refine_local_val,
            max_refine_levels_val=max_refine_levels_val,
            aspect_threshold_val=aspect_threshold_val,
            allow_stateful_cache=allow_stateful_cache,
        )

        if allow_stateful_cache:
            self._update_locals_template_cache_after_prepare(
                locals_template=tree_artifacts.locals_template,
                upward=tree_artifacts.upward,
                max_order=int(max_order),
            )

        retry_events_tuple = tuple(collected_retries)
        if allow_stateful_cache:
            self._recent_retry_events = retry_events_tuple
            self._record_strict_cap_profile_from_retries(
                retry_events_tuple,
                context_key=self._strict_cap_profile_context_key(
                    tree_mode=str(tree_artifacts.tree_mode),
                    leaf_parameter=int(tree_artifacts.leaf_parameter),
                    particle_count=int(
                        jnp.asarray(tree_artifacts.positions_sorted).shape[0]
                    ),
                ),
            )

        execution_backend = self._resolve_execution_backend()
        tree_type_norm = (
            str(getattr(tree_artifacts.tree, "tree_type", "")).strip().lower()
        )
        build_octree_payload = (
            execution_backend == "octree" or tree_type_norm == "octree"
        )
        if build_octree_payload:
            octree, octree_native = build_octree_execution_data_with_status(
                tree_artifacts.tree
            )
        else:
            octree, octree_native = None, False
        # Only build the native-octree interaction lists when the octree view is
        # actually native (non-degenerate). On a degenerate octree (build_octree_
        # execution_data fell back to the binary tree), the native walk would produce
        # far pairs in a node space inconsistent with `octree`/the near list -> gaps +
        # double-counts; leaving native_far_pairs=None routes far through the compat
        # interaction list on the same (fallback) tree, matching the near field.
        octree_native_neighbors = None
        if execution_backend == "octree" and octree is not None and octree_native:
            octree_native_neighbors = build_octree_native_neighbor_lists(
                tree_artifacts.tree,
                tree_artifacts.upward.geometry,
                theta=theta_val,
                mac_type=mac_type_val,
                dehnen_radius_scale=self.dehnen_radius_scale,
                max_pair_queue=self.max_pair_queue,
                process_block=self.pair_process_block,
                traversal_config=runtime_traversal_config,
            )
        nearfield_interop = _build_nearfield_interop_data(
            tree_artifacts.tree,
            dual_downward_artifacts.neighbor_list,
            octree=None,
            native_neighbors=None,
        )
        if (
            execution_backend == "octree"
            and nearfield_interop.leaf_particle_indices is None
        ):
            leaf_nodes_nf = jnp.asarray(nearfield_interop.leaf_nodes, dtype=INDEX_DTYPE)
            node_ranges_nf = jnp.asarray(
                nearfield_interop.particle_order_node_ranges,
                dtype=INDEX_DTYPE,
            )
            leaf_ranges_nf = node_ranges_nf[leaf_nodes_nf]
            counts_nf = leaf_ranges_nf[:, 1] - leaf_ranges_nf[:, 0] + 1
            width_nf = int(jnp.max(counts_nf)) if int(leaf_nodes_nf.shape[0]) > 0 else 0
            if width_nf > 0:
                offsets_nf = jnp.arange(width_nf, dtype=INDEX_DTYPE)
                leaf_particle_indices_nf = (
                    leaf_ranges_nf[:, 0][:, None] + offsets_nf[None, :]
                )
                leaf_particle_mask_nf = offsets_nf[None, :] < counts_nf[:, None]
                particle_to_leaf_position_nf = jnp.zeros(
                    (int(positions_arr.shape[0]),),
                    dtype=INDEX_DTYPE,
                )
                particle_to_leaf_position_nf = particle_to_leaf_position_nf.at[
                    leaf_particle_indices_nf[leaf_particle_mask_nf]
                ].set(
                    jnp.repeat(
                        jnp.arange(int(leaf_nodes_nf.shape[0]), dtype=INDEX_DTYPE),
                        counts_nf.astype(INDEX_DTYPE),
                    )
                )
            else:
                leaf_particle_indices_nf = jnp.zeros(
                    (int(leaf_nodes_nf.shape[0]), 0), dtype=INDEX_DTYPE
                )
                leaf_particle_mask_nf = jnp.zeros(
                    (int(leaf_nodes_nf.shape[0]), 0), dtype=bool
                )
                particle_to_leaf_position_nf = jnp.zeros(
                    (int(positions_arr.shape[0]),),
                    dtype=INDEX_DTYPE,
                )
            nearfield_interop = nearfield_interop._replace(
                leaf_particle_indices=leaf_particle_indices_nf,
                leaf_particle_mask=leaf_particle_mask_nf,
                particle_to_leaf_position=particle_to_leaf_position_nf,
            )
        nearfield_artifacts = self._prepare_state_nearfield_artifacts(
            neighbor_list=dual_downward_artifacts.neighbor_list,
            nearfield_interop=nearfield_interop,
            leaf_cap=tree_artifacts.leaf_cap,
            num_particles=int(positions_arr.shape[0]),
            cache_entry=dual_downward_artifacts.cache_entry,
            allow_stateful_cache=allow_stateful_cache,
        )
        _prepare_diag(
            "nearfield bytes "
            f"target_leaf_ids={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.target_leaf_ids))} "
            f"source_leaf_ids={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.source_leaf_ids))} "
            f"valid_pairs={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.valid_pairs))} "
            f"chunk_sort_indices={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.chunk_sort_indices))} "
            f"chunk_group_ids={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.chunk_group_ids))} "
            f"chunk_unique_indices={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.chunk_unique_indices))}"
        )
        octree_upward = _build_octree_upward_artifacts(
            octree=octree,
            positions_sorted=tree_artifacts.positions_sorted,
            masses_sorted=tree_artifacts.masses_sorted,
            expansion_basis=self.expansion_basis,
            max_order=int(max_order),
        )
        octree_native_far_pairs = None
        if execution_backend == "octree" and octree is not None and octree_native:
            octree_native_far_pairs = build_octree_native_far_pairs(
                tree_artifacts.tree,
                tree_artifacts.upward.geometry,
                theta=theta_val,
                mac_type=mac_type_val,
                dehnen_radius_scale=self.dehnen_radius_scale,
                max_pair_queue=self.max_pair_queue,
                process_block=self.pair_process_block,
                traversal_config=runtime_traversal_config,
            )
        octree_downward = _build_octree_downward_artifacts(
            octree=octree,
            octree_upward=octree_upward,
            interactions=dual_downward_artifacts.interactions,
            native_far_pairs=octree_native_far_pairs,
            execution_backend=execution_backend,
        )

        return FMMPreparedState(
            tree=tree_artifacts.tree,
            upward=_prepared_state_upward_payload(
                upward=tree_artifacts.upward,
                memory_objective=self.memory_objective,
            ),
            downward=dual_downward_artifacts.downward,
            neighbor_list=dual_downward_artifacts.neighbor_list,
            max_leaf_size=tree_artifacts.leaf_cap,
            input_dtype=input_dtype,
            working_dtype=positions_arr.dtype,
            expansion_basis=self.expansion_basis,
            theta=theta_val,
            topology_key=tree_artifacts.topology_key,
            interactions=dual_downward_artifacts.interactions,
            dual_tree_result=dual_downward_artifacts.traversal_result,
            retry_events=retry_events_tuple,
            nearfield_interop=nearfield_interop,
            nearfield_target_leaf_ids=nearfield_artifacts.target_leaf_ids,
            nearfield_source_leaf_ids=nearfield_artifacts.source_leaf_ids,
            nearfield_valid_pairs=nearfield_artifacts.valid_pairs,
            nearfield_chunk_sort_indices=nearfield_artifacts.chunk_sort_indices,
            nearfield_chunk_group_ids=nearfield_artifacts.chunk_group_ids,
            nearfield_chunk_unique_indices=nearfield_artifacts.chunk_unique_indices,
            force_scale_nodes=force_scale_nodes,
            execution_backend=execution_backend,
            octree=octree,
            octree_upward=_prepared_state_octree_upward_payload(
                octree_upward=octree_upward,
                memory_objective=self.memory_objective,
            ),
            octree_downward=_finalize_octree_downward_artifacts(
                octree=octree,
                octree_upward=octree_upward,
                octree_downward=octree_downward,
                expansion_basis=self.expansion_basis,
                execution_backend=execution_backend,
                m2l_chunk_size=runtime_m2l_chunk_size,
            ),
        )

    @jaxtyped(typechecker=beartype)
    def evaluate_prepared_state(
        self: "FastMultipoleMethod",
        state: PreparedStateLike,
        *,
        target_indices: Optional[Array] = None,
        return_potential: bool = False,
        jit_traversal: bool = True,
        max_acc_derivative_order: int = 0,
    ) -> Union[
        Array,
        Tuple[Array, Array],
        Tuple[Array, PackedAccelerationDerivatives],
        Tuple[Array, Array, PackedAccelerationDerivatives],
    ]:
        """Evaluate accelerations/potentials for all particles or targets."""

        if isinstance(state, LargeNPreparedState):
            return evaluate_large_n_state(
                self,
                state,
                target_indices=target_indices,
                return_potential=return_potential,
                max_acc_derivative_order=max_acc_derivative_order,
            )

        resolved_target_indices = self._resolve_target_indices(
            target_indices=target_indices,
            num_particles=int(state.inverse_permutation.shape[0]),
        )
        tracing_targets = isinstance(
            state.positions_sorted, jax.core.Tracer
        ) or isinstance(resolved_target_indices, jax.core.Tracer)
        derivative_order = int(max_acc_derivative_order)
        if derivative_order < 0:
            raise ValueError("max_acc_derivative_order must be non-negative")
        if derivative_order > 0 and state.expansion_basis != "solidfmm":
            raise NotImplementedError(
                "max_acc_derivative_order > 0 currently requires expansion_basis='solidfmm'"
            )

        use_full_eval_for_targets = bool(return_potential) and (
            resolved_target_indices is not None
        )
        # Octree backend: evaluate the octree-native far-field locals (the near-field is
        # already octree-native). Only the full-particle path honours these overrides.
        (
            octree_farfield_local_data,
            octree_farfield_leaf_nodes,
            octree_farfield_node_ranges,
        ) = _octree_farfield_eval_inputs(state)
        if (
            resolved_target_indices is None
            or tracing_targets
            or use_full_eval_for_targets
        ):
            evaluation = _evaluate_prepared_tree(
                fmm=self,
                tree=state.tree,
                positions_sorted=state.positions_sorted,
                masses_sorted=state.masses_sorted,
                downward=state.downward,
                neighbor_list=state.neighbor_list,
                nearfield_interop=state.nearfield_interop,
                farfield_local_data=octree_farfield_local_data,
                farfield_leaf_nodes=octree_farfield_leaf_nodes,
                farfield_node_ranges=octree_farfield_node_ranges,
                nearfield_target_leaf_ids=state.nearfield_target_leaf_ids,
                nearfield_source_leaf_ids=state.nearfield_source_leaf_ids,
                nearfield_valid_pairs=state.nearfield_valid_pairs,
                nearfield_chunk_sort_indices=state.nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids=state.nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices=state.nearfield_chunk_unique_indices,
                max_leaf_size=state.max_leaf_size,
                return_potential=return_potential,
                jit_traversal=jit_traversal,
                max_acc_derivative_order=derivative_order,
            )
        else:
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[resolved_target_indices],
                dtype=INDEX_DTYPE,
            )
            evaluation = _evaluate_prepared_tree_targets(
                fmm=self,
                tree=state.tree,
                positions_sorted=state.positions_sorted,
                masses_sorted=state.masses_sorted,
                downward=state.downward,
                neighbor_list=state.neighbor_list,
                nearfield_interop=state.nearfield_interop,
                farfield_local_data=None,
                farfield_leaf_nodes=None,
                farfield_node_ranges=None,
                target_sorted_indices=target_sorted_indices,
                return_potential=return_potential,
                max_acc_derivative_order=derivative_order,
            )

        if jnp.issubdtype(state.input_dtype, jnp.floating):
            output_dtype = state.input_dtype
        else:
            output_dtype = state.working_dtype

        if derivative_order > 0:
            if return_potential:
                acc_sorted, pot_sorted, deriv_sorted = evaluation
            else:
                acc_sorted, deriv_sorted = evaluation
            if resolved_target_indices is None:
                accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation]
                derivatives = tuple(
                    jnp.asarray(level)[state.inverse_permutation]
                    for level in deriv_sorted
                )
                if return_potential:
                    potentials = jnp.asarray(pot_sorted)[state.inverse_permutation]
            elif tracing_targets or use_full_eval_for_targets:
                accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation][
                    resolved_target_indices
                ]
                derivatives = tuple(
                    jnp.asarray(level)[state.inverse_permutation][
                        resolved_target_indices
                    ]
                    for level in deriv_sorted
                )
                if return_potential:
                    potentials = jnp.asarray(pot_sorted)[state.inverse_permutation][
                        resolved_target_indices
                    ]
            else:
                accelerations = jnp.asarray(acc_sorted)
                derivatives = tuple(jnp.asarray(level) for level in deriv_sorted)
                if return_potential:
                    potentials = jnp.asarray(pot_sorted)
            accelerations = accelerations.astype(output_dtype)
            derivatives = tuple(level.astype(output_dtype) for level in derivatives)
            if return_potential:
                return accelerations, potentials.astype(output_dtype), derivatives
            return accelerations, derivatives

        if return_potential:
            acc_sorted, pot_sorted = evaluation
            if resolved_target_indices is None:
                accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation]
                potentials = jnp.asarray(pot_sorted)[state.inverse_permutation]
            elif tracing_targets or use_full_eval_for_targets:
                accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation][
                    resolved_target_indices
                ]
                potentials = jnp.asarray(pot_sorted)[state.inverse_permutation][
                    resolved_target_indices
                ]
            else:
                accelerations = jnp.asarray(acc_sorted)
                potentials = jnp.asarray(pot_sorted)
            accelerations = accelerations.astype(output_dtype)
            potentials = potentials.astype(output_dtype)
            return accelerations, potentials

        if resolved_target_indices is None:
            accelerations = jnp.asarray(evaluation)[state.inverse_permutation]
        elif tracing_targets:
            accelerations = jnp.asarray(evaluation)[state.inverse_permutation][
                resolved_target_indices
            ]
        else:
            accelerations = jnp.asarray(evaluation)
        accelerations = accelerations.astype(output_dtype)
        return accelerations

    @jaxtyped(typechecker=beartype)
    def _evaluate_prepared_state_at_positions_sorted(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        positions_sorted: Array,
        *,
        target_indices: Optional[Array] = None,
        jit_traversal: bool = True,
    ) -> Array:
        """Evaluate accelerations for updated sorted positions on a fixed topology."""
        positions_sorted_arr = jnp.asarray(positions_sorted, dtype=state.working_dtype)
        if positions_sorted_arr.shape != state.positions_sorted.shape:
            raise ValueError(
                "positions_sorted must have shape "
                f"{tuple(state.positions_sorted.shape)}, got {tuple(positions_sorted_arr.shape)}"
            )

        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(positions_sorted_arr.shape[0]),
        )
        upward = self.prepare_upward_sweep(
            state.tree,
            positions_sorted_arr,
            state.masses_sorted,
            max_order=int(state.downward.locals.order),
            center_mode=runtime_overrides.center_mode,
            max_leaf_size=int(state.max_leaf_size),
        )
        downward = self.prepare_downward_sweep(
            state.tree,
            upward,
            theta=float(state.theta),
            mac_type=self.mac_type,
            initial_locals=None,
            interactions=state.interactions,
            m2l_chunk_size=runtime_overrides.m2l_chunk_size,
            l2l_chunk_size=runtime_overrides.l2l_chunk_size,
            grouped_interactions=runtime_overrides.grouped_interactions,
            farfield_mode=runtime_overrides.farfield_mode,
            dehnen_radius_scale=self.dehnen_radius_scale,
        )
        resolved_target_indices = self._resolve_target_indices(
            target_indices=target_indices,
            num_particles=int(state.inverse_permutation.shape[0]),
        )
        tracing_targets = isinstance(
            positions_sorted_arr, jax.core.Tracer
        ) or isinstance(resolved_target_indices, jax.core.Tracer)
        # Octree backend: evaluate octree-native far-field locals (full path only).
        (
            octree_farfield_local_data,
            octree_farfield_leaf_nodes,
            octree_farfield_node_ranges,
        ) = _octree_farfield_eval_inputs(state)
        if resolved_target_indices is None or tracing_targets:
            evaluation = _evaluate_prepared_tree(
                fmm=self,
                tree=state.tree,
                positions_sorted=positions_sorted_arr,
                masses_sorted=state.masses_sorted,
                downward=downward,
                neighbor_list=state.neighbor_list,
                nearfield_interop=state.nearfield_interop,
                farfield_local_data=octree_farfield_local_data,
                farfield_leaf_nodes=octree_farfield_leaf_nodes,
                farfield_node_ranges=octree_farfield_node_ranges,
                nearfield_target_leaf_ids=state.nearfield_target_leaf_ids,
                nearfield_source_leaf_ids=state.nearfield_source_leaf_ids,
                nearfield_valid_pairs=state.nearfield_valid_pairs,
                nearfield_chunk_sort_indices=state.nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids=state.nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices=state.nearfield_chunk_unique_indices,
                max_leaf_size=state.max_leaf_size,
                return_potential=False,
                jit_traversal=jit_traversal,
                max_acc_derivative_order=0,
            )
        else:
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[resolved_target_indices],
                dtype=INDEX_DTYPE,
            )
            evaluation = _evaluate_prepared_tree_targets(
                fmm=self,
                tree=state.tree,
                positions_sorted=positions_sorted_arr,
                masses_sorted=state.masses_sorted,
                downward=downward,
                neighbor_list=state.neighbor_list,
                nearfield_interop=state.nearfield_interop,
                farfield_local_data=None,
                farfield_leaf_nodes=None,
                farfield_node_ranges=None,
                target_sorted_indices=target_sorted_indices,
                return_potential=False,
                max_acc_derivative_order=0,
            )

        if jnp.issubdtype(state.input_dtype, jnp.floating):
            output_dtype = state.input_dtype
        else:
            output_dtype = state.working_dtype
        if resolved_target_indices is None:
            accelerations = jnp.asarray(evaluation)[state.inverse_permutation]
        elif tracing_targets:
            accelerations = jnp.asarray(evaluation)[state.inverse_permutation][
                resolved_target_indices
            ]
        else:
            accelerations = jnp.asarray(evaluation)
        return accelerations.astype(output_dtype)

    @jaxtyped(typechecker=beartype)
    def evaluate_tree(
        self: "FastMultipoleMethod",
        tree: Tree,
        positions_sorted: Array,
        masses_sorted: Array,
        locals_or_downward: Union[LocalExpansionData, TreeDownwardData],
        neighbor_list: NodeNeighborList,
        *,
        nearfield_interop: Optional[NearfieldInteropData] = None,
        farfield_local_data: Optional[LocalExpansionData] = None,
        farfield_leaf_nodes: Optional[Array] = None,
        farfield_node_ranges: Optional[Array] = None,
        precomputed_target_leaf_ids: Optional[Array] = None,
        precomputed_source_leaf_ids: Optional[Array] = None,
        precomputed_valid_pairs: Optional[Array] = None,
        precomputed_chunk_sort_indices: Optional[Array] = None,
        precomputed_chunk_group_ids: Optional[Array] = None,
        precomputed_chunk_unique_indices: Optional[Array] = None,
        max_leaf_size: Optional[int] = None,
        return_potential: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Combine far- and near-field effects for leaf particles."""

        setup = _prepare_tree_evaluation_inputs(
            tree,
            positions_sorted,
            masses_sorted,
            locals_or_downward,
            neighbor_list,
            farfield_local_data=farfield_local_data,
            farfield_leaf_nodes=farfield_leaf_nodes,
            farfield_node_ranges=farfield_node_ranges,
            max_leaf_size=max_leaf_size,
            return_potential=return_potential,
        )

        if setup.empty_output is not None:
            return setup.empty_output

        locals_data = setup.locals_data
        positions = setup.positions
        masses = setup.masses
        leaf_nodes = setup.leaf_nodes
        node_ranges = setup.node_ranges
        resolved_max_leaf = setup.max_leaf_size

        order = int(locals_data.order)
        nearfield_mode = self._resolve_nearfield_mode(
            num_particles=int(positions.shape[0])
        )
        nearfield_edge_chunk_size = self._resolve_nearfield_edge_chunk_size(
            num_particles=int(positions.shape[0]),
            nearfield_mode=nearfield_mode,
        )
        nearfield_view = (
            _build_nearfield_interop_data(tree, neighbor_list)
            if nearfield_interop is None
            else nearfield_interop
        )

        near = compute_leaf_p2p_accelerations(
            tree,
            neighbor_list,
            positions,
            masses,
            G=self.G,
            softening=self.softening,
            max_leaf_size=resolved_max_leaf,
            return_potential=return_potential,
            nearfield_mode=nearfield_mode,
            edge_chunk_size=nearfield_edge_chunk_size,
            precomputed_target_leaf_ids=precomputed_target_leaf_ids,
            precomputed_source_leaf_ids=precomputed_source_leaf_ids,
            precomputed_valid_pairs=precomputed_valid_pairs,
            precomputed_chunk_sort_indices=precomputed_chunk_sort_indices,
            precomputed_chunk_group_ids=precomputed_chunk_group_ids,
            precomputed_chunk_unique_indices=precomputed_chunk_unique_indices,
            node_ranges_override=nearfield_view.node_ranges,
            leaf_nodes_override=nearfield_view.leaf_nodes,
            neighbor_offsets_override=nearfield_view.offsets,
            neighbor_indices_override=nearfield_view.neighbors,
            neighbor_counts_override=nearfield_view.counts,
            leaf_particle_indices_override=nearfield_view.leaf_particle_indices,
            leaf_particle_mask_override=nearfield_view.leaf_particle_mask,
        )

        far_grad, far_potential_pre, _ = _evaluate_local_expansions_for_particles(
            locals_data,
            positions,
            leaf_nodes=leaf_nodes,
            node_ranges=node_ranges,
            max_leaf_size=resolved_max_leaf,
            order=order,
            expansion_basis=self.expansion_basis,
            return_potential=return_potential,
            max_acc_derivative_order=0,
        )

        # far_grad is d/d(delta) of +1/r with delta = center - eval_point.
        # Physical acceleration is d/d(eval_point)(+1/r) * G = -d/d(delta)(+1/r) * G.
        far_acc = -self.G * far_grad

        if return_potential:
            near_acc, near_pot = near
            far_pot = (
                -self.G * far_potential_pre
                if far_potential_pre is not None
                else jnp.zeros((positions.shape[0],), dtype=positions.dtype)
            )
            accelerations = near_acc + far_acc
            potentials = near_pot + far_pot
            return accelerations, potentials

        accelerations = near + far_acc
        return accelerations

    @jaxtyped(typechecker=beartype)
    def evaluate_tree_compiled(
        self: "FastMultipoleMethod",
        tree: Tree,
        positions_sorted: Array,
        masses_sorted: Array,
        locals_or_downward: Union[LocalExpansionData, TreeDownwardData],
        neighbor_list: NodeNeighborList,
        *,
        nearfield_interop: Optional[NearfieldInteropData] = None,
        farfield_local_data: Optional[LocalExpansionData] = None,
        farfield_leaf_nodes: Optional[Array] = None,
        farfield_node_ranges: Optional[Array] = None,
        precomputed_target_leaf_ids: Optional[Array] = None,
        precomputed_source_leaf_ids: Optional[Array] = None,
        precomputed_valid_pairs: Optional[Array] = None,
        precomputed_chunk_sort_indices: Optional[Array] = None,
        precomputed_chunk_group_ids: Optional[Array] = None,
        precomputed_chunk_unique_indices: Optional[Array] = None,
        max_leaf_size: Optional[int] = None,
        return_potential: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        """JIT-compiled variant of :meth:`evaluate_tree`."""

        resolved_max_leaf = (
            self.fixed_max_leaf_size
            if self.fixed_max_leaf_size is not None
            else max_leaf_size
        )

        setup = _prepare_tree_evaluation_inputs(
            tree,
            positions_sorted,
            masses_sorted,
            locals_or_downward,
            neighbor_list,
            farfield_local_data=farfield_local_data,
            farfield_leaf_nodes=farfield_leaf_nodes,
            farfield_node_ranges=farfield_node_ranges,
            max_leaf_size=resolved_max_leaf,
            return_potential=return_potential,
        )

        if setup.empty_output is not None:
            return setup.empty_output

        if self.fixed_order is not None:
            order = int(self.fixed_order)
        else:
            coeff_count = int(setup.locals_data.coefficients.shape[-1])
            order = _infer_order_from_coeff_count(
                coeff_count=coeff_count,
                expansion_basis=self.expansion_basis,
            )

        if self.fixed_max_leaf_size is not None and setup.max_leaf_size > int(
            self.fixed_max_leaf_size
        ):
            raise ValueError("fixed_max_leaf_size too small for prepared tree")
        nearfield_mode = self._resolve_nearfield_mode(
            num_particles=int(setup.positions.shape[0])
        )
        nearfield_edge_chunk_size = self._resolve_nearfield_edge_chunk_size(
            num_particles=int(setup.positions.shape[0]),
            nearfield_mode=nearfield_mode,
        )
        nearfield_view = (
            _build_nearfield_interop_data(tree, neighbor_list)
            if nearfield_interop is None
            else nearfield_interop
        )

        return _evaluate_tree_compiled_impl(
            tree,
            setup.positions,
            setup.masses,
            setup.locals_data,
            neighbor_list,
            jnp.asarray(nearfield_view.leaf_nodes, dtype=INDEX_DTYPE),
            jnp.asarray(nearfield_view.node_ranges, dtype=INDEX_DTYPE),
            jnp.asarray(nearfield_view.offsets, dtype=INDEX_DTYPE),
            jnp.asarray(nearfield_view.neighbors, dtype=INDEX_DTYPE),
            jnp.asarray(nearfield_view.counts, dtype=INDEX_DTYPE),
            (
                jnp.asarray(nearfield_view.leaf_particle_indices, dtype=INDEX_DTYPE)
                if nearfield_view.leaf_particle_indices is not None
                else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(nearfield_view.leaf_particle_mask, dtype=bool)
                if nearfield_view.leaf_particle_mask is not None
                else jnp.zeros((0, 0), dtype=bool)
            ),
            setup.leaf_nodes,
            setup.node_ranges,
            (
                jnp.asarray(precomputed_target_leaf_ids, dtype=INDEX_DTYPE)
                if precomputed_target_leaf_ids is not None
                else jnp.zeros((0,), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(precomputed_source_leaf_ids, dtype=INDEX_DTYPE)
                if precomputed_source_leaf_ids is not None
                else jnp.zeros((0,), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(precomputed_valid_pairs, dtype=bool)
                if precomputed_valid_pairs is not None
                else jnp.zeros((0,), dtype=bool)
            ),
            (
                jnp.asarray(precomputed_chunk_sort_indices, dtype=INDEX_DTYPE)
                if precomputed_chunk_sort_indices is not None
                else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(precomputed_chunk_group_ids, dtype=INDEX_DTYPE)
                if precomputed_chunk_group_ids is not None
                else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(precomputed_chunk_unique_indices, dtype=INDEX_DTYPE)
                if precomputed_chunk_unique_indices is not None
                else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
            ),
            jnp.zeros((setup.leaf_nodes.shape[0] + 1,), dtype=INDEX_DTYPE),
            jnp.zeros((0,), dtype=INDEX_DTYPE),
            jnp.zeros((0, 0), dtype=INDEX_DTYPE),
            jnp.zeros((0, 0), dtype=bool),
            jnp.zeros((setup.leaf_nodes.shape[0], 0, 0), dtype=INDEX_DTYPE),
            jnp.zeros((setup.leaf_nodes.shape[0], 0, 0), dtype=bool),
            G=self.G,
            softening=self.softening,
            order=order,
            expansion_basis=self.expansion_basis,
            max_leaf_size=setup.max_leaf_size,
            return_potential=return_potential,
            nearfield_mode=nearfield_mode,
            nearfield_edge_chunk_size=nearfield_edge_chunk_size,
        )


@jaxtyped(typechecker=beartype)
def compute_gravitational_acceleration(
    positions: Array,
    masses: Array,
    theta: float = 0.5,
    G: Union[float, Array] = 1.0,
    softening: Union[float, Array] = 0.0,
    *,
    bounds: Optional[Tuple[Array, Array]] = None,
    leaf_size: int = 16,
    max_order: int = 2,
    return_potential: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """Compute gravitational accelerations via the Fast Multipole Method."""

    fmm = FastMultipoleMethod(
        theta=theta,
        G=G,
        softening=softening,
    )
    return fmm.compute_accelerations(
        positions,
        masses,
        bounds=bounds,
        leaf_size=leaf_size,
        max_order=max_order,
        return_potential=return_potential,
    )


@jaxtyped(typechecker=beartype)
def compute_gravitational_potential(
    positions: Array,
    masses: Array,
    eval_points: Array,
    G: Union[float, Array] = 1.0,
    softening: Union[float, Array] = 0.0,
) -> Array:
    """Compute gravitational potential at evaluation points."""

    return reference_compute_potential(
        positions,
        masses,
        eval_points,
        G=G,
        softening=softening,
    )
