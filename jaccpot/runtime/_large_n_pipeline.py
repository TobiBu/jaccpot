"""Dedicated prepare/evaluate pipeline for large-N GPU radix solidfmm runs."""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jaxtyping import Array
from yggdrax.interactions import DualTreeRetryEvent, DualTreeTraversalConfig, MACType

from .dtypes import INDEX_DTYPE
from ._large_n_nearfield import (
    build_large_n_leaf_particle_groups,
    build_large_n_nearfield_precompute,
    resolve_large_n_execution_config,
)
from ._large_n_types import LargeNPreparedState


def prepare_large_n_state(
    fmm: object,
    *,
    positions_arr: Array,
    masses_arr: Array,
    input_dtype: jnp.dtype,
    bounds: Optional[Tuple[Array, Array]],
    leaf_size: int,
    max_order: int,
    theta_val: float,
    mac_type_val: MACType,
    refine_local_val: bool,
    max_refine_levels_val: int,
    aspect_threshold_val: float,
    jit_tree_override: Optional[bool],
    allow_stateful_cache: bool,
    runtime_traversal_config: Optional[DualTreeTraversalConfig],
    runtime_m2l_chunk_size: Optional[int],
    runtime_l2l_chunk_size: Optional[int],
    upward_center_mode: str,
    record_retry: Callable[[DualTreeRetryEvent], None],
    collected_retries: list[DualTreeRetryEvent],
) -> LargeNPreparedState:
    """Prepare the slim large-N state using the dedicated narrow path."""

    tree_artifacts = fmm._prepare_state_tree_and_upward(
        positions_arr=positions_arr,
        masses_arr=masses_arr,
        bounds=bounds,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        refine_local_val=refine_local_val,
        max_refine_levels_val=max_refine_levels_val,
        aspect_threshold_val=aspect_threshold_val,
        jit_tree_override=jit_tree_override,
        upward_center_mode=upward_center_mode,
        allow_stateful_cache=allow_stateful_cache,
    )

    dual_downward_artifacts = fmm._prepare_state_dual_and_downward(
        tree_artifacts=tree_artifacts,
        force_scale_nodes=None,
        upward_center_mode=upward_center_mode,
        theta_val=theta_val,
        mac_type_val=mac_type_val,
        dehnen_radius_scale=fmm.dehnen_radius_scale,
        runtime_traversal_config=runtime_traversal_config,
        runtime_m2l_chunk_size=runtime_m2l_chunk_size,
        runtime_l2l_chunk_size=runtime_l2l_chunk_size,
        grouped_interactions=False,
        farfield_mode="pair_grouped",
        record_retry=record_retry,
        refine_local_val=refine_local_val,
        max_refine_levels_val=max_refine_levels_val,
        aspect_threshold_val=aspect_threshold_val,
        allow_stateful_cache=allow_stateful_cache,
    )

    if allow_stateful_cache:
        fmm._update_locals_template_cache_after_prepare(
            locals_template=tree_artifacts.locals_template,
            upward=tree_artifacts.upward,
            max_order=int(max_order),
        )

    retry_events_tuple = tuple(collected_retries)
    if allow_stateful_cache:
        fmm._recent_retry_events = retry_events_tuple

    execution_config = resolve_large_n_execution_config(
        fmm,
        num_particles=int(positions_arr.shape[0]),
    )
    if bool(execution_config.retain_leaf_groups):
        leaf_particle_indices, leaf_particle_mask = build_large_n_leaf_particle_groups(
            tree_artifacts.tree,
            dual_downward_artifacts.neighbor_list,
            max_leaf_size=int(tree_artifacts.leaf_cap),
        )
    else:
        leaf_particle_indices = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        leaf_particle_mask = jnp.zeros((0, 0), dtype=bool)
    nearfield_artifacts = build_large_n_nearfield_precompute(
        tree=tree_artifacts.tree,
        neighbor_list=dual_downward_artifacts.neighbor_list,
        leaf_particle_indices=leaf_particle_indices,
        leaf_particle_mask=leaf_particle_mask,
        execution_config=execution_config,
    )

    return LargeNPreparedState(
        tree=tree_artifacts.tree,
        local_data=dual_downward_artifacts.downward.locals,
        neighbor_list=dual_downward_artifacts.neighbor_list,
        nearfield_leaf_particle_indices=leaf_particle_indices,
        nearfield_leaf_particle_mask=leaf_particle_mask,
        nearfield_target_leaf_ids=nearfield_artifacts.target_leaf_ids,
        nearfield_source_leaf_ids=nearfield_artifacts.source_leaf_ids,
        nearfield_valid_pairs=nearfield_artifacts.valid_pairs,
        nearfield_chunk_sort_indices=nearfield_artifacts.chunk_sort_indices,
        nearfield_chunk_group_ids=nearfield_artifacts.chunk_group_ids,
        nearfield_chunk_unique_indices=nearfield_artifacts.chunk_unique_indices,
        max_leaf_size=int(tree_artifacts.leaf_cap),
        input_dtype=jnp.dtype(input_dtype),
        working_dtype=jnp.dtype(positions_arr.dtype),
        theta=float(theta_val),
        topology_key=tree_artifacts.topology_key,
        retry_events=retry_events_tuple,
        execution_backend="large_n",
        expansion_basis="solidfmm",
        nearfield_mode=str(execution_config.nearfield_mode),
        nearfield_edge_chunk_size=int(execution_config.nearfield_edge_chunk_size),
    )


def evaluate_large_n_state(
    fmm: object,
    state: LargeNPreparedState,
    *,
    target_indices: Optional[Array],
    return_potential: bool,
    max_acc_derivative_order: int,
):
    """Evaluate the slim large-N state for the full particle set."""

    from ._fmm_impl import _evaluate_tree_compiled_impl

    if target_indices is not None:
        raise NotImplementedError(
            "Large-N runtime path currently supports full-particle evaluation only."
        )
    if int(max_acc_derivative_order) != 0:
        raise NotImplementedError(
            "Large-N runtime path currently does not support acceleration derivatives."
        )

    leaf_nodes = jnp.asarray(state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    nearfield_mode = str(state.nearfield_mode)
    nearfield_edge_chunk_size = int(state.nearfield_edge_chunk_size)
    eval_out = _evaluate_tree_compiled_impl(
        state.tree,
        state.positions_sorted,
        state.masses_sorted,
        state.local_data,
        state.neighbor_list,
        leaf_nodes,
        jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE),
        jnp.asarray(state.neighbor_list.offsets, dtype=INDEX_DTYPE),
        jnp.asarray(state.neighbor_list.neighbors, dtype=INDEX_DTYPE),
        jnp.asarray(state.neighbor_list.counts, dtype=INDEX_DTYPE),
        (
            jnp.asarray(state.nearfield_leaf_particle_indices, dtype=INDEX_DTYPE)
            if int(state.nearfield_leaf_particle_indices.size) > 0
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_leaf_particle_mask, dtype=bool)
            if int(state.nearfield_leaf_particle_mask.size) > 0
            else jnp.zeros((0, 0), dtype=bool)
        ),
        leaf_nodes,
        jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE),
        (
            jnp.asarray(state.nearfield_target_leaf_ids, dtype=INDEX_DTYPE)
            if state.nearfield_target_leaf_ids is not None
            else jnp.zeros((0,), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_source_leaf_ids, dtype=INDEX_DTYPE)
            if state.nearfield_source_leaf_ids is not None
            else jnp.zeros((0,), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_valid_pairs, dtype=bool)
            if state.nearfield_valid_pairs is not None
            else jnp.zeros((0,), dtype=bool)
        ),
        (
            jnp.asarray(state.nearfield_chunk_sort_indices, dtype=INDEX_DTYPE)
            if state.nearfield_chunk_sort_indices is not None
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_chunk_group_ids, dtype=INDEX_DTYPE)
            if state.nearfield_chunk_group_ids is not None
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_chunk_unique_indices, dtype=INDEX_DTYPE)
            if state.nearfield_chunk_unique_indices is not None
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        G=float(getattr(fmm, "G")),
        softening=float(getattr(fmm, "softening")),
        order=int(state.local_data.order),
        expansion_basis="solidfmm",
        max_leaf_size=int(state.max_leaf_size),
        return_potential=bool(return_potential),
        nearfield_mode=nearfield_mode,
        nearfield_edge_chunk_size=nearfield_edge_chunk_size,
    )

    if jnp.issubdtype(state.input_dtype, jnp.floating):
        output_dtype = state.input_dtype
    else:
        output_dtype = state.working_dtype

    if return_potential:
        accelerations_sorted, potentials_sorted = eval_out
    else:
        accelerations_sorted = eval_out

    if not return_potential:
        return jnp.asarray(accelerations_sorted)[state.inverse_permutation].astype(
            output_dtype
        )

    accelerations = jnp.asarray(accelerations_sorted)[state.inverse_permutation].astype(
        output_dtype
    )
    potentials = jnp.asarray(potentials_sorted)[state.inverse_permutation].astype(
        output_dtype
    )
    return accelerations, potentials


def can_use_large_n_prepare_path(
    fmm: object,
    *,
    positions_arr: Array,
    masses_arr: Array,
    allow_stateful_cache: bool,
) -> bool:
    """Decide whether prepare_state should dispatch to the large-N path."""

    runtime_path = str(getattr(fmm, "runtime_path", "auto")).strip().lower()
    if runtime_path == "legacy":
        return False
    if runtime_path == "auto" and str(getattr(fmm, "preset", "")).strip().lower() != "large_n_gpu":
        return False
    if not allow_stateful_cache:
        return False
    if jax.default_backend() != "gpu":
        return False
    if str(getattr(fmm, "tree_type", "")).strip().lower() != "radix":
        return False
    if str(getattr(fmm, "expansion_basis", "")).strip().lower() != "solidfmm":
        return False
    if str(getattr(fmm, "execution_backend", "auto")).strip().lower() == "octree":
        return False
    if bool(getattr(fmm, "adaptive_order", False)):
        return False
    if bool(getattr(fmm, "mixed_order_farfield", False)):
        return False
    if str(getattr(fmm, "complex_rotation", "")).strip().lower() != "solidfmm":
        return False
    if bool(getattr(fmm, "_uses_paper_style_force_scale")) and fmm._uses_paper_style_force_scale():
        return False
    if int(positions_arr.shape[0]) != int(masses_arr.shape[0]):
        return False
    return True
