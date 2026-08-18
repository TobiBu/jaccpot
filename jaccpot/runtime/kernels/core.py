"""Reusable FMM numerical kernel library (aggregator).

The numerical core was one 3966-line module extracted verbatim from
``_fmm_impl.py`` (Phase 2c). Tier 1.6 subdivided it along the four families
ARCHITECTURE §5 already documents -- which is the follow-up
``kernels/__init__.py`` promised:

    _shared.py          basis tag, output bundles, the strict-refresh diag gate
    _downward_prep.py   the four downward-sweep input bundles + accumulate
    _m2l.py             the M2L apply/accumulate seam, whole (see its docstring)
    _l2l.py             L2L propagation + the downward-sweep driver
    _evaluate.py        L2P, the targeted near field, the evaluation entry points

This module re-exports all of them, so ``jaccpot.runtime.kernels.core`` remains a
working import path and ``kernels/__init__.py``'s curated surface is unchanged.

Leaf contract, unchanged and load-bearing: this package must NOT import the engine
(``runtime.fmm.engine`` / the prepare pipeline), so ``distributed/`` and
``experimental/`` can use the kernels without dragging in the orchestrator. The two
engine annotations in ``_evaluate`` are deliberately left dangling for that reason;
see the comment at the site.
"""

from __future__ import annotations

# Part of this module's surface, not an incidental import: the gate is now
# *consulted* in `_m2l`, but `tests/unit/test_grad_config.py` asserts it is bound
# on `core` -- a guard written after a NameError that only fired at call time.
# Keeping it here means the seam split changes no test.
from jaccpot.runtime.grad_options import fused_m2l_pallas_enabled  # noqa: F401

from ._downward_prep import (  # noqa: F401
    _empty_interaction_storage_for_tree,
    _FarPairCOO,
    _prepare_solidfmm_downward_child_inputs,
    _prepare_solidfmm_downward_init,
    _prepare_solidfmm_downward_interaction_inputs,
    _prepare_solidfmm_downward_multipole_inputs,
    _solidfmm_downward_accumulate_from_multipoles,
    _SolidFMMDownwardChildInputs,
    _SolidFMMDownwardInit,
    _SolidFMMDownwardInteractionInputs,
    _SolidFMMDownwardMultipoleInputs,
)
from ._evaluate import (  # noqa: F401
    _build_nearfield_interop_data,
    _build_target_nearfield_source_index_matrix,
    _compute_targeted_nearfield,
    _evaluate_local_cartesian_with_grad_batch,
    _evaluate_local_expansions_for_particles,
    _evaluate_local_expansions_for_target_particles,
    _evaluate_prepared_tree,
    _evaluate_prepared_tree_targets,
    _evaluate_tree_compiled_impl,
    _EvaluationNodeViews,
    _infer_bounds,
    _infer_order_from_coeff_count,
    _map_targets_to_leaf_positions,
    _max_leaf_size_from_tree,
    _prepare_tree_evaluation_inputs,
    _resolve_evaluation_node_views,
    _scatter_rank3,
    _scatter_scalars,
    _scatter_vectors,
    _TreeEvaluationSetup,
)
from ._l2l import (  # noqa: F401
    _l2l_complex_batch_kernel,
    _l2l_real_batch_kernel,
    _prepare_solidfmm_downward_sweep,
    _propagate_real_locals_to_children,
    _propagate_solidfmm_locals_by_level,
    _propagate_solidfmm_locals_to_children,
)
from ._m2l import (  # noqa: F401
    _accumulate_m2l_chunked_scan,
    _accumulate_m2l_fullbatch,
    _accumulate_solidfmm_m2l_class_major_chunked_scan,
    _accumulate_solidfmm_m2l_grouped,
    _accumulate_solidfmm_m2l_grouped_chunked_scan,
    _accumulate_solidfmm_m2l_grouped_class_major,
    _accumulate_solidfmm_m2l_grouped_fullbatch,
    _apply_complex_m2l,
    _apply_m2l,
    _apply_real_m2l,
    _build_grouped_class_segments,
    _chunk_segment_scatter_add,
    _fused_complex_m2l_pallas_active,
    _m2l_cached_kernel_dispatch,
    _m2l_chunk_contributions,
    _m2l_complex_batch_cached_kernel,
    _m2l_complex_batch_kernel,
    _m2l_complex_batch_kernel_fused_pallas,
    _m2l_real_batch_kernel,
    _m2l_real_batch_kernel_fused_pallas,
    _pair_class_ids_from_offsets,
    _real_m2l_pallas_active,
    _rotation_blocks_for_grouped_classes,
)
from ._shared import (  # noqa: F401
    _STRICT_REFRESH_DETAIL_DIAG_MODES,
    ExpansionBasis,
    NearfieldInteropData,
    PackedAccelerationDerivatives,
    _normalize_strict_refresh_detail_diag_mode,
)

__all__ = [
    "ExpansionBasis",
    "NearfieldInteropData",
    "PackedAccelerationDerivatives",
    "fused_m2l_pallas_enabled",
]

# RE-EXPORTS. The names below are imported for other modules to reach through
# this one, and are unused *here*. Holding references makes that a fact the
# interpreter can see: `pyflakes` counts them as used, so whatever it still
# reports for this module is a real dead import rather than noise. A tuple of
# STRINGS would not do it, and neither does `# noqa` -- pyflakes ignores noqa,
# and isort hoists a trailing comment onto the whole import group, so a name that
# later goes unused inside that group is never flagged. Verified, 2026-08-18.
_REEXPORTS = (
    _EvaluationNodeViews,
    _FarPairCOO,
    _STRICT_REFRESH_DETAIL_DIAG_MODES,
    _SolidFMMDownwardChildInputs,
    _SolidFMMDownwardInit,
    _SolidFMMDownwardInteractionInputs,
    _SolidFMMDownwardMultipoleInputs,
    _TreeEvaluationSetup,
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
    _pair_class_ids_from_offsets,
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
)
