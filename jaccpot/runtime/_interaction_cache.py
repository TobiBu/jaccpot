"""Dual-tree interaction cache helpers for the runtime FMM implementation."""

import hashlib
from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import numpy as np
from beartype.typing import Callable
from jaxtyping import Array
from yggdrax.dense_interactions import DenseInteractionBuffers, densify_interactions
from yggdrax.grouped_interactions import GroupedInteractionBuffers
from yggdrax.interactions import (
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    DualTreeWalkResult,
    MACType,
    NodeInteractionList,
    NodeNeighborList,
)
from yggdrax.tree import Tree


@dataclass(frozen=True)
class _DualTreeArtifacts:
    """Artifacts emitted by the dual-tree traversal builder."""

    interactions: NodeInteractionList
    neighbor_list: NodeNeighborList
    traversal_result: DualTreeWalkResult
    dense_buffers: Optional[DenseInteractionBuffers]
    grouped_buffers: Optional[GroupedInteractionBuffers]
    grouped_segment_starts: Optional[Array]
    grouped_segment_lengths: Optional[Array]
    grouped_segment_class_ids: Optional[Array]
    grouped_segment_sort_permutation: Optional[Array]
    grouped_segment_group_ids: Optional[Array]
    grouped_segment_unique_targets: Optional[Array]
    grouped_chunk_size: Optional[int]


class _InteractionCacheEntry(NamedTuple):
    """Cache entry for dual-tree interaction artifacts keyed by build options."""

    key: str
    interactions: NodeInteractionList
    neighbor_list: NodeNeighborList
    dual_tree_result: DualTreeWalkResult
    grouped_buffers: Optional[GroupedInteractionBuffers]
    grouped_segment_starts: Optional[Array]
    grouped_segment_lengths: Optional[Array]
    grouped_segment_class_ids: Optional[Array]
    grouped_segment_sort_permutation: Optional[Array]
    grouped_segment_group_ids: Optional[Array]
    grouped_segment_unique_targets: Optional[Array]
    grouped_chunk_size: Optional[int]
    nearfield_target_leaf_ids: Optional[Array]
    nearfield_source_leaf_ids: Optional[Array]
    nearfield_valid_pairs: Optional[Array]
    nearfield_chunk_sort_indices: Optional[Array]
    nearfield_chunk_group_ids: Optional[Array]
    nearfield_chunk_unique_indices: Optional[Array]
    nearfield_mode: Optional[str]
    nearfield_edge_chunk_size: Optional[int]
    nearfield_leaf_cap: Optional[int]


def _interaction_cache_key(
    tree: Tree,
    *,
    tree_mode: str,
    leaf_parameter: int,
    theta: float,
    mac_type: MACType,
    dehnen_radius_scale: float,
    expansion_basis: str,
    center_mode: str,
    max_pair_queue: Optional[int],
    pair_process_block: Optional[int],
    traversal_config: Optional[DualTreeTraversalConfig],
    refine_local: Optional[bool],
    max_refine_levels: Optional[int],
    aspect_threshold: Optional[float],
) -> Optional[str]:
    """Return a hash for the interaction list of a tree/theta configuration.

    If any tree arrays are tracers (for example under grad/jit), return ``None``
    to disable caching and avoid host round-trips on traced values.
    """

    hasher = hashlib.sha256()

    try:
        morton_codes = np.asarray(
            jax.device_get(tree.morton_codes),
            dtype=np.uint64,
        )
        node_ranges = np.asarray(
            jax.device_get(tree.node_ranges),
            dtype=np.int64,
        )
        bounds_min = np.asarray(
            jax.device_get(tree.bounds_min),
            dtype=np.float64,
        )
        bounds_max = np.asarray(
            jax.device_get(tree.bounds_max),
            dtype=np.float64,
        )
    except Exception:
        return None

    hasher.update(morton_codes.tobytes())
    hasher.update(node_ranges.tobytes())
    hasher.update(bounds_min.tobytes())
    hasher.update(bounds_max.tobytes())

    mode_bytes = tree_mode.encode("utf8")
    leaf_bytes = np.asarray(int(leaf_parameter), dtype=np.int64).tobytes()
    theta_bytes = np.asarray(float(theta), dtype=np.float64).tobytes()
    mac_bytes = str(mac_type).encode("utf8")
    dehnen_scale_bytes = np.asarray(
        float(dehnen_radius_scale), dtype=np.float64
    ).tobytes()
    basis_bytes = str(expansion_basis).encode("utf8")
    center_mode_bytes = str(center_mode).encode("utf8")
    if traversal_config is not None:
        queue_val = int(traversal_config.max_pair_queue)
        block_val = int(traversal_config.process_block)
        interaction_val = int(traversal_config.max_interactions_per_node)
        neighbor_val = int(traversal_config.max_neighbors_per_leaf)
    else:
        queue_val = -1 if max_pair_queue is None else int(max_pair_queue)
        block_val = -1 if pair_process_block is None else int(pair_process_block)
        interaction_val = -1
        neighbor_val = -1
    refine_val = -1 if refine_local is None else int(bool(refine_local))
    max_refine_val = -1 if max_refine_levels is None else int(max_refine_levels)
    aspect_val = -1.0 if aspect_threshold is None else float(aspect_threshold)
    hasher.update(mode_bytes)
    hasher.update(leaf_bytes)
    hasher.update(theta_bytes)
    hasher.update(mac_bytes)
    hasher.update(dehnen_scale_bytes)
    hasher.update(basis_bytes)
    hasher.update(center_mode_bytes)
    hasher.update(np.asarray(queue_val, dtype=np.int64).tobytes())
    hasher.update(np.asarray(block_val, dtype=np.int64).tobytes())
    hasher.update(np.asarray(interaction_val, dtype=np.int64).tobytes())
    hasher.update(np.asarray(neighbor_val, dtype=np.int64).tobytes())
    hasher.update(np.asarray(refine_val, dtype=np.int64).tobytes())
    hasher.update(np.asarray(max_refine_val, dtype=np.int64).tobytes())
    hasher.update(np.asarray(aspect_val, dtype=np.float64).tobytes())
    return hasher.hexdigest()


def _build_dual_tree_artifacts(
    tree: Tree,
    geometry,
    *,
    theta: float,
    mac_type: MACType,
    dehnen_radius_scale: float,
    cache_key: Optional[str],
    cache_entry: Optional[_InteractionCacheEntry],
    max_pair_queue: Optional[int],
    pair_process_block: Optional[int],
    traversal_config: Optional[DualTreeTraversalConfig],
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]],
    use_dense_interactions: bool,
    grouped_interactions: bool,
    grouped_chunk_size: Optional[int],
) -> tuple[_DualTreeArtifacts, Optional[_InteractionCacheEntry]]:
    """Construct or reuse dual-tree traversal products for a tree."""

    cache_out = cache_entry
    if (
        cache_key is not None
        and cache_entry is not None
        and cache_entry.key == cache_key
    ):
        interactions = cache_entry.interactions
        neighbor_list = cache_entry.neighbor_list
        traversal_result = cache_entry.dual_tree_result
        grouped_buffers = cache_entry.grouped_buffers
        grouped_segment_starts = cache_entry.grouped_segment_starts
        grouped_segment_lengths = cache_entry.grouped_segment_lengths
        grouped_segment_class_ids = cache_entry.grouped_segment_class_ids
        grouped_segment_sort_permutation = cache_entry.grouped_segment_sort_permutation
        grouped_segment_group_ids = cache_entry.grouped_segment_group_ids
        grouped_segment_unique_targets = cache_entry.grouped_segment_unique_targets
        grouped_chunk_size_cached = cache_entry.grouped_chunk_size
    else:
        from . import fmm as _runtime_fmm

        build_out = _runtime_fmm.build_interactions_and_neighbors(
            tree,
            geometry,
            theta=theta,
            mac_type=mac_type,
            dehnen_radius_scale=dehnen_radius_scale,
            max_pair_queue=max_pair_queue,
            process_block=pair_process_block,
            traversal_config=traversal_config,
            retry_logger=retry_logger,
            return_result=True,
            return_grouped=grouped_interactions,
        )
        if grouped_interactions:
            (
                interactions,
                neighbor_list,
                traversal_result,
                grouped_buffers,
            ) = build_out
        else:
            (
                interactions,
                neighbor_list,
                traversal_result,
            ) = build_out
        cache_out = (
            _InteractionCacheEntry(
                key=cache_key,
                interactions=interactions,
                neighbor_list=neighbor_list,
                dual_tree_result=traversal_result,
                grouped_buffers=grouped_buffers if grouped_interactions else None,
                grouped_segment_starts=None,
                grouped_segment_lengths=None,
                grouped_segment_class_ids=None,
                grouped_segment_sort_permutation=None,
                grouped_segment_group_ids=None,
                grouped_segment_unique_targets=None,
                grouped_chunk_size=None,
                nearfield_target_leaf_ids=None,
                nearfield_source_leaf_ids=None,
                nearfield_valid_pairs=None,
                nearfield_chunk_sort_indices=None,
                nearfield_chunk_group_ids=None,
                nearfield_chunk_unique_indices=None,
                nearfield_mode=None,
                nearfield_edge_chunk_size=None,
                nearfield_leaf_cap=None,
            )
            if cache_key is not None
            else None
        )
        grouped_segment_starts = None
        grouped_segment_lengths = None
        grouped_segment_class_ids = None
        grouped_segment_sort_permutation = None
        grouped_segment_group_ids = None
        grouped_segment_unique_targets = None
        grouped_chunk_size_cached = None
        if not grouped_interactions:
            grouped_buffers = None

    if grouped_interactions and grouped_buffers is None:
        from yggdrax import interactions as _yggdrax_interactions

        grouped_buffers = _yggdrax_interactions.build_grouped_interactions_from_pairs(
            tree,
            geometry,
            interactions.sources,
            interactions.targets,
            level_offsets=getattr(interactions, "level_offsets", None),
        )
        if cache_out is not None:
            cache_out = _InteractionCacheEntry(
                key=cache_out.key,
                interactions=cache_out.interactions,
                neighbor_list=cache_out.neighbor_list,
                dual_tree_result=cache_out.dual_tree_result,
                grouped_buffers=grouped_buffers,
                grouped_segment_starts=cache_out.grouped_segment_starts,
                grouped_segment_lengths=cache_out.grouped_segment_lengths,
                grouped_segment_class_ids=cache_out.grouped_segment_class_ids,
                grouped_segment_sort_permutation=cache_out.grouped_segment_sort_permutation,
                grouped_segment_group_ids=cache_out.grouped_segment_group_ids,
                grouped_segment_unique_targets=cache_out.grouped_segment_unique_targets,
                grouped_chunk_size=cache_out.grouped_chunk_size,
                nearfield_target_leaf_ids=cache_out.nearfield_target_leaf_ids,
                nearfield_source_leaf_ids=cache_out.nearfield_source_leaf_ids,
                nearfield_valid_pairs=cache_out.nearfield_valid_pairs,
                nearfield_chunk_sort_indices=cache_out.nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids=cache_out.nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices=cache_out.nearfield_chunk_unique_indices,
                nearfield_mode=cache_out.nearfield_mode,
                nearfield_edge_chunk_size=cache_out.nearfield_edge_chunk_size,
                nearfield_leaf_cap=cache_out.nearfield_leaf_cap,
            )

    if (
        grouped_interactions
        and grouped_buffers is not None
        and grouped_chunk_size is not None
    ):
        needs_schedule = (
            grouped_segment_starts is None
            or grouped_segment_lengths is None
            or grouped_segment_class_ids is None
            or grouped_segment_sort_permutation is None
            or grouped_segment_group_ids is None
            or grouped_segment_unique_targets is None
            or grouped_chunk_size_cached != int(grouped_chunk_size)
        )
        if needs_schedule:
            from . import fmm as _runtime_fmm

            (
                grouped_segment_starts,
                grouped_segment_lengths,
                grouped_segment_class_ids,
                grouped_segment_sort_permutation,
                grouped_segment_group_ids,
                grouped_segment_unique_targets,
            ) = _runtime_fmm._build_grouped_class_segments(
                grouped_buffers,
                chunk_size=int(grouped_chunk_size),
            )
            grouped_chunk_size_cached = int(grouped_chunk_size)
            if cache_out is not None:
                cache_out = _InteractionCacheEntry(
                    key=cache_out.key,
                    interactions=cache_out.interactions,
                    neighbor_list=cache_out.neighbor_list,
                    dual_tree_result=cache_out.dual_tree_result,
                    grouped_buffers=grouped_buffers,
                    grouped_segment_starts=grouped_segment_starts,
                    grouped_segment_lengths=grouped_segment_lengths,
                    grouped_segment_class_ids=grouped_segment_class_ids,
                    grouped_segment_sort_permutation=grouped_segment_sort_permutation,
                    grouped_segment_group_ids=grouped_segment_group_ids,
                    grouped_segment_unique_targets=grouped_segment_unique_targets,
                    grouped_chunk_size=grouped_chunk_size_cached,
                    nearfield_target_leaf_ids=cache_out.nearfield_target_leaf_ids,
                    nearfield_source_leaf_ids=cache_out.nearfield_source_leaf_ids,
                    nearfield_valid_pairs=cache_out.nearfield_valid_pairs,
                    nearfield_chunk_sort_indices=cache_out.nearfield_chunk_sort_indices,
                    nearfield_chunk_group_ids=cache_out.nearfield_chunk_group_ids,
                    nearfield_chunk_unique_indices=cache_out.nearfield_chunk_unique_indices,
                    nearfield_mode=cache_out.nearfield_mode,
                    nearfield_edge_chunk_size=cache_out.nearfield_edge_chunk_size,
                    nearfield_leaf_cap=cache_out.nearfield_leaf_cap,
                )

    dense_buffers = (
        densify_interactions(tree, geometry, interactions)
        if use_dense_interactions
        else None
    )

    artifacts = _DualTreeArtifacts(
        interactions=interactions,
        neighbor_list=neighbor_list,
        traversal_result=traversal_result,
        dense_buffers=dense_buffers,
        grouped_buffers=grouped_buffers,
        grouped_segment_starts=grouped_segment_starts,
        grouped_segment_lengths=grouped_segment_lengths,
        grouped_segment_class_ids=grouped_segment_class_ids,
        grouped_segment_sort_permutation=grouped_segment_sort_permutation,
        grouped_segment_group_ids=grouped_segment_group_ids,
        grouped_segment_unique_targets=grouped_segment_unique_targets,
        grouped_chunk_size=grouped_chunk_size_cached,
    )
    return artifacts, cache_out
