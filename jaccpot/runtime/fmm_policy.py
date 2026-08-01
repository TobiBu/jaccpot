"""PolicyMixin: fmm_policy methods extracted from the FastMultipoleMethod
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from beartype.typing import Callable
from jaxtyping import Array
from yggdrax.interactions import (
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    MACType,
    build_octree_native_far_pairs,
    build_octree_native_neighbor_lists,
)
from yggdrax.tree import Tree

from jaccpot.upward.tree_expansions import TreeUpwardData

from ._adaptive_policy import (
    build_adaptive_policy_state,
    compute_node_force_scale_from_sorted_acc,
    source_error_proxy_by_order_from_multipoles,
)
from ._octree_adapter import build_octree_execution_data_with_status
from .fmm_state import (
    FMMPreparedState,
    _build_octree_downward_artifacts,
    _build_octree_upward_artifacts,
    _finalize_octree_downward_artifacts,
    _prepared_state_octree_upward_payload,
    _prepared_state_upward_payload,
    _PrepareStateTreeUpwardArtifacts,
)
from .kernels.core import _build_nearfield_interop_data


class PolicyMixin:
    def _solidfmm_basis_mode(self: "FastMultipoleMethod") -> str:
        """Return active solidfmm coefficient family ('complex' or 'real')."""
        basis_obj = self.basis_impl
        name = str(getattr(basis_obj, "name", "")).strip().lower()
        if name == "real":
            return "real"
        return "complex"

    def _compute_node_force_scale_from_sorted_acc(
        self: "FastMultipoleMethod",
        *,
        tree: Tree,
        accelerations_sorted: Array,
        reduction: str = "max",
    ) -> Array:
        """Estimate per-node force scales from sorted per-particle accelerations."""

        return compute_node_force_scale_from_sorted_acc(
            tree=tree,
            accelerations_sorted=accelerations_sorted,
            reduction=reduction,
        )

    def _source_error_proxy_by_order_from_multipoles(
        self: "FastMultipoleMethod",
        *,
        multipole_packed: Array,
        p_gears: tuple[int, ...],
    ) -> Array:
        """Compute a conservative per-node residual proxy for each candidate order."""

        return source_error_proxy_by_order_from_multipoles(
            multipole_packed=multipole_packed,
            p_gears=p_gears,
        )

    def _adaptive_error_model_code(self: "FastMultipoleMethod") -> int:
        """Return the integer policy code for the active adaptive error model."""

        if self.adaptive_error_model == "dehnen_paper":
            return 2
        if self.adaptive_error_model == "dehnen_degree":
            return 1
        return 0

    def _uses_dehnen_error_policy(self: "FastMultipoleMethod") -> bool:
        """Return whether traversal should use the Dehnen paper policy hook."""

        return str(self.mac_type) == "dehnen_error"

    def _uses_dehnen_paper_error_model(self: "FastMultipoleMethod") -> bool:
        """Return whether the active adaptive error model is the paper estimator."""

        return self.adaptive_error_model == "dehnen_paper"

    def _uses_paper_style_traversal_policy(self: "FastMultipoleMethod") -> bool:
        """Return whether traversal should use the paper-style error policy."""

        return self._uses_dehnen_paper_error_model() or self._uses_dehnen_error_policy()

    def _traversal_policy_error_model_code(self: "FastMultipoleMethod") -> int:
        """Return the policy error model code used during traversal."""

        if self._uses_dehnen_error_policy():
            return 2
        return self._adaptive_error_model_code()

    def _force_scale_reduction_mode(self: "FastMultipoleMethod") -> str:
        """Return the node reduction mode used for adaptive force scales."""

        return "min" if self._uses_dehnen_paper_error_model() else "max"

    def _uses_paper_style_force_scale(self: "FastMultipoleMethod") -> bool:
        """Return whether prepare_state needs paper-style force-scale handling."""

        return self.adaptive_order or self._uses_paper_style_traversal_policy()

    def _base_mac_type(self: "FastMultipoleMethod") -> MACType:
        """Return the Yggdrax-facing geometric MAC for the active solver mode."""

        return "dehnen" if self._uses_dehnen_error_policy() else self.mac_type

    def _policy_orders_for_prepare_state(
        self: "FastMultipoleMethod", *, max_order: int
    ) -> tuple[int, ...]:
        """Return candidate orders used to build adaptive traversal policy state."""

        if (not self.adaptive_order) and self._uses_dehnen_paper_error_model():
            return (int(max_order),)
        return self.p_gears

    def _build_adaptive_policy_state(
        self: "FastMultipoleMethod",
        *,
        upward: TreeUpwardData,
        tree: Tree,
        positions_sorted: Array,
        p_gears: tuple[int, ...],
        force_scale_nodes: Optional[Array],
        eps: Array,
        theta: Array,
        error_model_code: Array,
        dehnen_geometry_mode: str,
    ):
        """Build the solver-owned adaptive policy state from upward data."""

        return build_adaptive_policy_state(
            upward=upward,
            tree=tree,
            positions_sorted=positions_sorted,
            p_gears=p_gears,
            force_scale_nodes=force_scale_nodes,
            eps=eps,
            theta=theta,
            error_model_code=error_model_code,
            dehnen_geometry_mode=dehnen_geometry_mode,
        )

    def _compute_force_scale_paper_prepass_from_tree_artifacts(
        self: "FastMultipoleMethod",
        *,
        tree_artifacts: _PrepareStateTreeUpwardArtifacts,
        low_order: int,
        theta_val: float,
        upward_center_mode: str,
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        grouped_interactions: bool,
        farfield_mode: str,
        record_retry: Callable[[DualTreeRetryEvent], None],
        refine_local_val: bool,
        max_refine_levels_val: int,
        aspect_threshold_val: float,
    ) -> Array:
        """Compute paper-mode force scales via a low-order prepass on the current tree."""

        low_upward = self.prepare_upward_sweep(
            tree_artifacts.tree,
            tree_artifacts.positions_sorted,
            tree_artifacts.masses_sorted,
            max_order=int(low_order),
            center_mode=upward_center_mode,
            max_leaf_size=tree_artifacts.leaf_cap,
        )
        low_locals_template = self._build_locals_template_for_prepare_state(
            tree=tree_artifacts.tree,
            upward=low_upward,
            max_order=int(low_order),
            pos_sorted=tree_artifacts.positions_sorted,
        )
        low_tree_artifacts = _PrepareStateTreeUpwardArtifacts(
            tree_mode=tree_artifacts.tree_mode,
            tree=tree_artifacts.tree,
            positions_sorted=tree_artifacts.positions_sorted,
            masses_sorted=tree_artifacts.masses_sorted,
            inverse_permutation=tree_artifacts.inverse_permutation,
            leaf_cap=tree_artifacts.leaf_cap,
            leaf_parameter=tree_artifacts.leaf_parameter,
            topology_key=tree_artifacts.topology_key,
            upward=low_upward,
            locals_template=low_locals_template,
        )

        saved_p_gears = self.p_gears
        saved_adaptive_order = self.adaptive_order
        saved_adaptive_error_model = self.adaptive_error_model
        saved_mac_type = self.mac_type
        saved_recent_counts = self._recent_far_pairs_by_gear_counts
        self._in_force_scale_prepass = True
        try:
            self.p_gears = (int(low_order),)
            self.adaptive_order = False
            self.adaptive_error_model = "tail_proxy"
            self.mac_type = "dehnen"
            dual_downward_artifacts = self._prepare_state_dual_and_downward(
                tree_artifacts=low_tree_artifacts,
                force_scale_nodes=None,
                upward_center_mode=upward_center_mode,
                theta_val=theta_val,
                mac_type_val=self.mac_type,
                dehnen_radius_scale=self.dehnen_radius_scale,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                grouped_interactions=grouped_interactions,
                farfield_mode=farfield_mode,
                record_retry=record_retry,
                refine_local_val=refine_local_val,
                max_refine_levels_val=max_refine_levels_val,
                aspect_threshold_val=aspect_threshold_val,
                allow_stateful_cache=False,
            )
            prepass_execution_backend = self._resolve_execution_backend()
            if prepass_execution_backend == "octree":
                prepass_octree, prepass_octree_native = (
                    build_octree_execution_data_with_status(low_tree_artifacts.tree)
                )
            else:
                prepass_octree, prepass_octree_native = None, False
            # See the main prepared-state path: only build native-octree interaction
            # lists when the octree view is non-degenerate; otherwise far/near come from
            # the consistent compat lists on the fallback (binary) tree.
            prepass_octree_native_neighbors = None
            if (
                prepass_execution_backend == "octree"
                and prepass_octree is not None
                and prepass_octree_native
            ):
                prepass_octree_native_neighbors = build_octree_native_neighbor_lists(
                    low_tree_artifacts.tree,
                    low_tree_artifacts.upward.geometry,
                    theta=theta_val,
                    mac_type=self.mac_type,
                    dehnen_radius_scale=self.dehnen_radius_scale,
                    max_pair_queue=self.max_pair_queue,
                    process_block=self.pair_process_block,
                    traversal_config=runtime_traversal_config,
                )
            prepass_nearfield_interop = _build_nearfield_interop_data(
                low_tree_artifacts.tree,
                dual_downward_artifacts.neighbor_list,
                octree=prepass_octree,
                native_neighbors=prepass_octree_native_neighbors,
            )
            nearfield_artifacts = self._prepare_state_nearfield_artifacts(
                neighbor_list=dual_downward_artifacts.neighbor_list,
                nearfield_interop=prepass_nearfield_interop,
                leaf_cap=low_tree_artifacts.leaf_cap,
                num_particles=int(low_tree_artifacts.positions_sorted.shape[0]),
                cache_entry=dual_downward_artifacts.cache_entry,
                allow_stateful_cache=False,
            )
            prepass_octree_upward = _build_octree_upward_artifacts(
                octree=prepass_octree,
                positions_sorted=low_tree_artifacts.positions_sorted,
                masses_sorted=low_tree_artifacts.masses_sorted,
                expansion_basis=self.expansion_basis,
                max_order=int(low_order),
            )
            prepass_octree_native_far_pairs = None
            if (
                prepass_execution_backend == "octree"
                and prepass_octree is not None
                and prepass_octree_native
            ):
                prepass_octree_native_far_pairs = build_octree_native_far_pairs(
                    low_tree_artifacts.tree,
                    low_tree_artifacts.upward.geometry,
                    theta=theta_val,
                    mac_type=self.mac_type,
                    dehnen_radius_scale=self.dehnen_radius_scale,
                    max_pair_queue=self.max_pair_queue,
                    process_block=self.pair_process_block,
                    traversal_config=runtime_traversal_config,
                )
            prepass_octree_downward = _build_octree_downward_artifacts(
                octree=prepass_octree,
                octree_upward=prepass_octree_upward,
                interactions=dual_downward_artifacts.interactions,
                native_far_pairs=prepass_octree_native_far_pairs,
                execution_backend=prepass_execution_backend,
            )
            prepass_state = FMMPreparedState(
                tree=low_tree_artifacts.tree,
                upward=_prepared_state_upward_payload(
                    upward=low_tree_artifacts.upward,
                    memory_objective=self.memory_objective,
                ),
                downward=dual_downward_artifacts.downward,
                neighbor_list=dual_downward_artifacts.neighbor_list,
                max_leaf_size=low_tree_artifacts.leaf_cap,
                input_dtype=low_tree_artifacts.positions_sorted.dtype,
                working_dtype=low_tree_artifacts.positions_sorted.dtype,
                expansion_basis=self.expansion_basis,
                theta=theta_val,
                topology_key=low_tree_artifacts.topology_key,
                interactions=dual_downward_artifacts.interactions,
                dual_tree_result=dual_downward_artifacts.traversal_result,
                retry_events=tuple(),
                nearfield_interop=prepass_nearfield_interop,
                nearfield_target_leaf_ids=nearfield_artifacts.target_leaf_ids,
                nearfield_source_leaf_ids=nearfield_artifacts.source_leaf_ids,
                nearfield_valid_pairs=nearfield_artifacts.valid_pairs,
                nearfield_chunk_sort_indices=nearfield_artifacts.chunk_sort_indices,
                nearfield_chunk_group_ids=nearfield_artifacts.chunk_group_ids,
                nearfield_chunk_unique_indices=nearfield_artifacts.chunk_unique_indices,
                force_scale_nodes=None,
                execution_backend=prepass_execution_backend,
                octree=prepass_octree,
                octree_upward=_prepared_state_octree_upward_payload(
                    octree_upward=prepass_octree_upward,
                    memory_objective=self.memory_objective,
                ),
                octree_downward=_finalize_octree_downward_artifacts(
                    octree=prepass_octree,
                    octree_upward=prepass_octree_upward,
                    octree_downward=prepass_octree_downward,
                    expansion_basis=self.expansion_basis,
                    execution_backend=prepass_execution_backend,
                    m2l_chunk_size=runtime_m2l_chunk_size,
                ),
            )
            prepass_acc = self.evaluate_prepared_state(
                prepass_state,
                return_potential=False,
                jit_traversal=False,
            )
        finally:
            self.p_gears = saved_p_gears
            self.adaptive_order = saved_adaptive_order
            self.adaptive_error_model = saved_adaptive_error_model
            self.mac_type = saved_mac_type
            self._recent_far_pairs_by_gear_counts = saved_recent_counts
            self._in_force_scale_prepass = False

        sorted_idx = jnp.argsort(low_tree_artifacts.inverse_permutation)
        return jnp.asarray(prepass_acc)[sorted_idx]
