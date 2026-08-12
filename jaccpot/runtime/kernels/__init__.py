"""Reusable FMM numerical kernel library (leaf package).

The numerical core was one module extracted from ``_fmm_impl``; Tier 1.6
subdivided it into ``_shared`` / ``_downward_prep`` / ``_m2l`` / ``_l2l`` /
``_evaluate``, which is the follow-up this docstring used to promise.
:mod:`jaccpot.runtime.kernels.core` is now the aggregator over those five, and
this package re-exports its symbols so consumers import from
``jaccpot.runtime.kernels``.

Leaf contract: this package and ``core`` must NOT import the engine
(``runtime.fmm.engine`` / the prepare pipeline), so ``distributed`` and
``experimental`` can use the kernels without dragging in the orchestrator.
"""

from __future__ import annotations

from .core import (
    _STRICT_REFRESH_DETAIL_DIAG_MODES,
    ExpansionBasis,
    PackedAccelerationDerivatives,
    _accumulate_m2l_chunked_scan,
    _accumulate_m2l_fullbatch,
    _accumulate_solidfmm_m2l_class_major_chunked_scan,
    _accumulate_solidfmm_m2l_grouped,
    _accumulate_solidfmm_m2l_grouped_chunked_scan,
    _accumulate_solidfmm_m2l_grouped_class_major,
    _accumulate_solidfmm_m2l_grouped_fullbatch,
    _apply_complex_m2l,
    _apply_real_m2l,
    _build_grouped_class_segments,
    _build_nearfield_interop_data,
    _build_target_nearfield_source_index_matrix,
    _chunk_segment_scatter_add,
    _compute_targeted_nearfield,
    _empty_interaction_storage_for_tree,
    _evaluate_local_cartesian_with_grad_batch,
    _evaluate_local_expansions_for_particles,
    _evaluate_local_expansions_for_target_particles,
    _evaluate_prepared_tree,
    _evaluate_prepared_tree_targets,
    _evaluate_tree_compiled_impl,
    _EvaluationNodeViews,
    _FarPairCOO,
    _fused_complex_m2l_pallas_active,
    _infer_bounds,
    _infer_order_from_coeff_count,
    _l2l_complex_batch_kernel,
    _l2l_real_batch_kernel,
    _m2l_cached_kernel_dispatch,
    _m2l_complex_batch_cached_kernel,
    _m2l_complex_batch_kernel,
    _m2l_complex_batch_kernel_fused_pallas,
    _m2l_real_batch_kernel,
    _m2l_real_batch_kernel_fused_pallas,
    _map_targets_to_leaf_positions,
    _max_leaf_size_from_tree,
    _normalize_strict_refresh_detail_diag_mode,
    _prepare_solidfmm_downward_child_inputs,
    _prepare_solidfmm_downward_init,
    _prepare_solidfmm_downward_interaction_inputs,
    _prepare_solidfmm_downward_multipole_inputs,
    _prepare_solidfmm_downward_sweep,
    _prepare_tree_evaluation_inputs,
    _propagate_real_locals_to_children,
    _propagate_solidfmm_locals_by_level,
    _propagate_solidfmm_locals_to_children,
    _real_m2l_pallas_active,
    _resolve_evaluation_node_views,
    _rotation_blocks_for_grouped_classes,
    _scatter_rank3,
    _scatter_scalars,
    _scatter_vectors,
    _SolidFMMDownwardChildInputs,
    _SolidFMMDownwardInit,
    _SolidFMMDownwardInteractionInputs,
    _SolidFMMDownwardMultipoleInputs,
    _TreeEvaluationSetup,
)
