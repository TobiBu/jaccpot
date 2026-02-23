"""Near-field cache helpers for prepared-state artifacts."""

from typing import NamedTuple, Optional

from jaxtyping import Array

from ._interaction_cache import _InteractionCacheEntry


class NearfieldPrecomputeArtifacts(NamedTuple):
    """Precomputed near-field pair lists and optional bucket schedules."""

    target_leaf_ids: Optional[Array]
    source_leaf_ids: Optional[Array]
    valid_pairs: Optional[Array]
    chunk_sort_indices: Optional[Array]
    chunk_group_ids: Optional[Array]
    chunk_unique_indices: Optional[Array]


def nearfield_cache_matches(
    cache_entry: Optional[_InteractionCacheEntry],
    *,
    nearfield_mode: str,
    nearfield_edge_chunk_size: int,
    leaf_cap: int,
) -> bool:
    """Return whether cache entry contains reusable near-field artifacts."""

    return bool(
        cache_entry is not None
        and cache_entry.nearfield_mode == nearfield_mode
        and cache_entry.nearfield_edge_chunk_size == nearfield_edge_chunk_size
        and cache_entry.nearfield_leaf_cap == int(leaf_cap)
        and cache_entry.nearfield_target_leaf_ids is not None
        and cache_entry.nearfield_source_leaf_ids is not None
        and cache_entry.nearfield_valid_pairs is not None
    )


def nearfield_from_cache(
    cache_entry: _InteractionCacheEntry,
) -> NearfieldPrecomputeArtifacts:
    """Extract near-field artifacts from a cache entry."""

    return NearfieldPrecomputeArtifacts(
        target_leaf_ids=cache_entry.nearfield_target_leaf_ids,
        source_leaf_ids=cache_entry.nearfield_source_leaf_ids,
        valid_pairs=cache_entry.nearfield_valid_pairs,
        chunk_sort_indices=cache_entry.nearfield_chunk_sort_indices,
        chunk_group_ids=cache_entry.nearfield_chunk_group_ids,
        chunk_unique_indices=cache_entry.nearfield_chunk_unique_indices,
    )


def with_nearfield_cache_artifacts(
    cache_entry: _InteractionCacheEntry,
    *,
    artifacts: NearfieldPrecomputeArtifacts,
    nearfield_mode: str,
    nearfield_edge_chunk_size: int,
    leaf_cap: int,
) -> _InteractionCacheEntry:
    """Return cache entry updated with near-field artifacts and policy metadata."""

    return _InteractionCacheEntry(
        key=cache_entry.key,
        interactions=cache_entry.interactions,
        neighbor_list=cache_entry.neighbor_list,
        dual_tree_result=cache_entry.dual_tree_result,
        grouped_buffers=cache_entry.grouped_buffers,
        grouped_segment_starts=cache_entry.grouped_segment_starts,
        grouped_segment_lengths=cache_entry.grouped_segment_lengths,
        grouped_segment_class_ids=cache_entry.grouped_segment_class_ids,
        grouped_segment_sort_permutation=cache_entry.grouped_segment_sort_permutation,
        grouped_segment_group_ids=cache_entry.grouped_segment_group_ids,
        grouped_segment_unique_targets=cache_entry.grouped_segment_unique_targets,
        grouped_chunk_size=cache_entry.grouped_chunk_size,
        nearfield_target_leaf_ids=artifacts.target_leaf_ids,
        nearfield_source_leaf_ids=artifacts.source_leaf_ids,
        nearfield_valid_pairs=artifacts.valid_pairs,
        nearfield_chunk_sort_indices=artifacts.chunk_sort_indices,
        nearfield_chunk_group_ids=artifacts.chunk_group_ids,
        nearfield_chunk_unique_indices=artifacts.chunk_unique_indices,
        nearfield_mode=nearfield_mode,
        nearfield_edge_chunk_size=nearfield_edge_chunk_size,
        nearfield_leaf_cap=int(leaf_cap),
    )
