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
from jaccpot.config import FMMExecutionBackend, FMMPreset, MemoryObjective
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
from .fmm_evaluate import EvaluateMixin
from .fmm_overrides import OverridesMixin
from .fmm_policy import PolicyMixin
from .fmm_prepare import PrepareMixin
from .fmm_presets import get_preset_config
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
from .fmm_strict_run import StrictRunMixin
from .fmm_sweeps import SweepsMixin
from .kernels.core import (
    ExpansionBasis,
    NearfieldInteropData,
    PackedAccelerationDerivatives,
    _accumulate_real_m2l_chunked_scan_pallas,
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
    PrepareMixin,
    EvaluateMixin,
    StrictRunMixin,
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
        runtime_path: Literal["auto", "large_n"] = "auto",
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
        if runtime_path_norm not in ("auto", "large_n"):
            raise ValueError("runtime_path must be 'auto' or 'large_n'")
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
