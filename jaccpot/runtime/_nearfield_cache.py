"""Near-field cache helpers for prepared-state artifacts."""

from __future__ import annotations

from typing import NamedTuple, Optional

from jaxtyping import Array

from ._interaction_cache import _InteractionCacheEntry


class NearfieldPrecomputeArtifacts(NamedTuple):
    """Precomputed near-field pair lists and optional bucket schedules.

    Every field is ``Optional``: which ones are populated depends on the
    near-field mode and on whether pair vectors were retained, so a consumer must
    check rather than assume. The three ``chunk_*`` fields are the bucketed
    scatter schedule and are only meaningful as a set -- the ``*_with_schedule``
    scatter helpers take all three -- but nothing here enforces that, and several
    modules populate the underlying cache fields, so check the one you use.

    Attributes
    ----------
    target_leaf_ids : Optional[Array]
        Target leaf of each near pair.
    source_leaf_ids : Optional[Array]
        Source leaf of each near pair, positionally aligned with the targets.
    valid_pairs : Optional[Array]
        Validity mask over the pair list.
    chunk_sort_indices : Optional[Array]
        Scatter-schedule permutation.
    chunk_group_ids : Optional[Array]
        Scatter-schedule segment ids.
    chunk_unique_indices : Optional[Array]
        Scatter-schedule target per segment.
    """

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
    require_pair_vectors: bool = False,
) -> bool:
    """Return whether cache entry contains reusable near-field artifacts.

    Reuse requires the POLICY to match, not just the arrays to exist: a schedule
    built for a different chunk size or leaf cap has the wrong shapes, and one
    built without pair vectors cannot serve a caller that needs them.

    Parameters
    ----------
    cache_entry : Optional[_InteractionCacheEntry]
        Candidate entry; ``None`` never matches.
    nearfield_mode : str
        Mode the caller needs.
    nearfield_edge_chunk_size : int
        Chunk size the caller needs; part of the schedule's shape.
    leaf_cap : int
        Leaf capacity the caller needs.
    require_pair_vectors : bool
        Whether the caller needs the pair lists as well as the schedule. Checks
        ``target_leaf_ids`` and ``valid_pairs`` only -- ``source_leaf_ids`` is not
        tested, so a caller that needs it must check separately.

    Returns
    -------
    bool
        ``True`` when the entry's artifacts are reusable as-is.
    """

    return bool(
        cache_entry is not None
        and cache_entry.nearfield_mode == nearfield_mode
        and cache_entry.nearfield_edge_chunk_size == nearfield_edge_chunk_size
        and cache_entry.nearfield_leaf_cap == int(leaf_cap)
        and (
            not bool(require_pair_vectors)
            or (
                cache_entry.nearfield_target_leaf_ids is not None
                and cache_entry.nearfield_valid_pairs is not None
            )
        )
    )


def nearfield_from_cache(
    cache_entry: _InteractionCacheEntry,
) -> NearfieldPrecomputeArtifacts:
    """Extract near-field artifacts from a cache entry.

    Assumes the entry has already been vetted by
    :func:`nearfield_cache_matches`; it reads fields without re-checking policy.

    Parameters
    ----------
    cache_entry : _InteractionCacheEntry
        Entry to read.

    Returns
    -------
    NearfieldPrecomputeArtifacts
        The artifacts held by that entry. Fields absent from the entry come back
        ``None`` rather than raising.
    """

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
    """Return cache entry updated with near-field artifacts and policy metadata.

    Returns a NEW entry rather than mutating: cache entries are shared, so an
    in-place update would rewrite policy metadata under another holder.

    Parameters
    ----------
    cache_entry : _InteractionCacheEntry
        Entry to copy.
    artifacts : NearfieldPrecomputeArtifacts
        Artifacts to attach.
    nearfield_mode : str
        Mode the artifacts were built for; recorded so a later
        :func:`nearfield_cache_matches` can reject a mismatched caller.
    nearfield_edge_chunk_size : int
        Chunk size they were built for.
    leaf_cap : int
        Leaf capacity they were built for.

    Returns
    -------
    _InteractionCacheEntry
        A copy carrying the artifacts and the policy they are valid for.
    """

    return _InteractionCacheEntry(
        key=cache_entry.key,
        interactions=cache_entry.interactions,
        neighbor_list=cache_entry.neighbor_list,
        dual_tree_result=cache_entry.dual_tree_result,
        compact_far_pairs=cache_entry.compact_far_pairs,
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
