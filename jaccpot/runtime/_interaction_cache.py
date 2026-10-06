"""Dual-tree interaction cache helpers for the runtime FMM implementation."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Callable
from jaxtyping import Array
from yggdrax.dense_interactions import DenseInteractionBuffers, densify_interactions
from yggdrax.geometry import TreeGeometry
from yggdrax.grouped_interactions import GroupedInteractionBuffers
from yggdrax.interactions import (
    CompactTaggedFarPairs,
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    DualTreeWalkResult,
    MACType,
    NodeInteractionList,
    NodeNeighborList,
    PairPolicy,
    build_compact_far_pairs_and_leaf_neighbor_lists,
    build_interactions_and_neighbors_split,
    build_leaf_neighbor_lists,
)
from yggdrax.tree import Tree

from jaccpot._env import env_flag, env_float
from jaccpot._jax_compat import Tracer
from jaccpot._searchsorted import searchsorted_method

# `_adaptive_policy` reaches only `fmm_caches` and `fmm_constants`, both UPSTREAM of
# this module in ARCHITECTURE §8's DAG (`fmm_constants -> fmm_caches -> kernels ->
# {_interaction_cache, ...}`), so this import runs with the layering rather than
# against it. Verified acyclic by walking the relative-import graph, not by eye.
#
# It exists so `policy_state` can be annotated at all. Until it, the four dual-tree
# builders took the Dehnen policy state as an untyped parameter, which is what made
# this file undocumentable: pydoclint refuses a Parameters section for a signature
# with missing hints (DOC106/107), so 70 violations sat behind one missing import.
from ._adaptive_policy import AdaptivePolicyState
from .dtypes import require_index_capacity

__all__ = [
    "POLICY_IDENTITY_UNCACHEABLE",
    "flat_walk_cap_headroom",
    "pair_policy_cache_identity",
    "strict_walk_backend",
    "strict_walk_deterministic_rows",
    "TargetSortedFarPairs",
    "far_pair_targets",
]


@dataclass(frozen=True)
class _DualTreeArtifacts:
    """Artifacts emitted by the dual-tree traversal builder.

    Attributes
    ----------
    interactions : Optional[NodeInteractionList]
        Accepted far-field node pairs, or ``None`` on the streamed path.
    neighbor_list : NodeNeighborList
        Near-neighbour list produced by the same traversal.
    traversal_result : Optional[DualTreeWalkResult]
        Full walk result, or ``None`` when it was not retained.
    compact_far_pairs : Optional[CompactTaggedFarPairs]
        Compact tagged far pairs, or ``None`` when not requested.
    dense_buffers : Optional[DenseInteractionBuffers]
        Dense interaction buffers, or ``None`` when unused.
    grouped_buffers : Optional[GroupedInteractionBuffers]
        Grouped (class-major) interaction buffers, or ``None`` when unused.
    grouped_segment_starts : Optional[Array]
        Start offset of each grouped segment.
    grouped_segment_lengths : Optional[Array]
        Length of each grouped segment.
    grouped_segment_class_ids : Optional[Array]
        Class id of each grouped segment.
    grouped_segment_sort_permutation : Optional[Array]
        Permutation sorting segments into class-major order.
    grouped_segment_group_ids : Optional[Array]
        Group id of each grouped segment.
    grouped_segment_unique_targets : Optional[Array]
        Unique target nodes per grouped segment.
    grouped_chunk_size : Optional[int]
        Pairs per grouped chunk, or ``None`` for the default.
    cache_hit : bool
        Whether these artifacts came from the cache rather than a fresh build.
    """

    interactions: Optional[NodeInteractionList]
    neighbor_list: NodeNeighborList
    traversal_result: Optional[DualTreeWalkResult]
    compact_far_pairs: Optional[CompactTaggedFarPairs]
    dense_buffers: Optional[DenseInteractionBuffers]
    grouped_buffers: Optional[GroupedInteractionBuffers]
    grouped_segment_starts: Optional[Array]
    grouped_segment_lengths: Optional[Array]
    grouped_segment_class_ids: Optional[Array]
    grouped_segment_sort_permutation: Optional[Array]
    grouped_segment_group_ids: Optional[Array]
    grouped_segment_unique_targets: Optional[Array]
    grouped_chunk_size: Optional[int]
    cache_hit: bool = False


class _InteractionCacheEntry(NamedTuple):
    """Cache entry for dual-tree interaction artifacts keyed by build options.

    Attributes
    ----------
    key : str
        Hash of the build options this entry was keyed on.
    interactions : Optional[NodeInteractionList]
        Accepted far-field node pairs, or ``None`` on the streamed path.
    neighbor_list : NodeNeighborList
        Near-neighbour list produced by the same traversal.
    dual_tree_result : Optional[DualTreeWalkResult]
        Full walk result, or ``None`` when it was not retained.
    compact_far_pairs : Optional[CompactTaggedFarPairs]
        Compact tagged far pairs, or ``None`` when not requested.
    grouped_buffers : Optional[GroupedInteractionBuffers]
        Grouped (class-major) interaction buffers, or ``None`` when unused.
    grouped_segment_starts : Optional[Array]
        Start offset of each grouped segment.
    grouped_segment_lengths : Optional[Array]
        Length of each grouped segment.
    grouped_segment_class_ids : Optional[Array]
        Class id of each grouped segment.
    grouped_segment_sort_permutation : Optional[Array]
        Permutation sorting segments into class-major order.
    grouped_segment_group_ids : Optional[Array]
        Group id of each grouped segment.
    grouped_segment_unique_targets : Optional[Array]
        Unique target nodes per grouped segment.
    grouped_chunk_size : Optional[int]
        Pairs per grouped chunk, or ``None`` for the default.
    nearfield_target_leaf_ids : Optional[Array]
        Target leaf id per near-field pair.
    nearfield_source_leaf_ids : Optional[Array]
        Source leaf id per near-field pair.
    nearfield_valid_pairs : Optional[Array]
        Mask marking which near-field pair slots are real.
    nearfield_chunk_sort_indices : Optional[Array]
        Permutation sorting near-field pairs into chunks.
    nearfield_chunk_group_ids : Optional[Array]
        Chunk id per near-field pair.
    nearfield_chunk_unique_indices : Optional[Array]
        Unique target index per near-field chunk.
    nearfield_mode : Optional[str]
        Near-field traversal mode the entry was built for.
    nearfield_edge_chunk_size : Optional[int]
        Edge chunk size the entry was built for.
    nearfield_leaf_cap : Optional[int]
        Leaf capacity the entry was built for.
    """

    key: str
    interactions: Optional[NodeInteractionList]
    neighbor_list: NodeNeighborList
    dual_tree_result: Optional[DualTreeWalkResult]
    compact_far_pairs: Optional[CompactTaggedFarPairs]
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


class _DualTreeCacheHit(NamedTuple):
    """Resolved cached dual-tree payload reused for a build request.

    Attributes
    ----------
    interactions : Optional[NodeInteractionList]
        Accepted far-field node pairs, or ``None`` on the streamed path.
    neighbor_list : NodeNeighborList
        Near-neighbour list produced by the same traversal.
    traversal_result : Optional[DualTreeWalkResult]
        Full walk result, or ``None`` when it was not retained.
    compact_far_pairs : Optional[CompactTaggedFarPairs]
        Compact tagged far pairs, or ``None`` when not requested.
    grouped_buffers : Optional[GroupedInteractionBuffers]
        Grouped (class-major) interaction buffers, or ``None`` when unused.
    grouped_segment_starts : Optional[Array]
        Start offset of each grouped segment.
    grouped_segment_lengths : Optional[Array]
        Length of each grouped segment.
    grouped_segment_class_ids : Optional[Array]
        Class id of each grouped segment.
    grouped_segment_sort_permutation : Optional[Array]
        Permutation sorting segments into class-major order.
    grouped_segment_group_ids : Optional[Array]
        Group id of each grouped segment.
    grouped_segment_unique_targets : Optional[Array]
        Unique target nodes per grouped segment.
    grouped_chunk_size_cached : Optional[int]
        The chunk size the cached payload was built with.
    cache_out : Optional['_InteractionCacheEntry']
        Entry to write back, or ``None`` when nothing needs storing.
    """

    interactions: Optional[NodeInteractionList]
    neighbor_list: NodeNeighborList
    traversal_result: Optional[DualTreeWalkResult]
    compact_far_pairs: Optional[CompactTaggedFarPairs]
    grouped_buffers: Optional[GroupedInteractionBuffers]
    grouped_segment_starts: Optional[Array]
    grouped_segment_lengths: Optional[Array]
    grouped_segment_class_ids: Optional[Array]
    grouped_segment_sort_permutation: Optional[Array]
    grouped_segment_group_ids: Optional[Array]
    grouped_segment_unique_targets: Optional[Array]
    grouped_chunk_size_cached: Optional[int]
    cache_out: Optional["_InteractionCacheEntry"]


class _RefreshDualPlannerHint(NamedTuple):
    """Cached refresh planner decision for dual artifact build routing.

    Attributes
    ----------
    use_split_build : bool
        Whether to build far and near traversal in separate passes.
    suppress_substage_timing : bool
        Whether to skip the per-substage timing callbacks.
    """

    use_split_build: bool
    suppress_substage_timing: bool = False


@partial(jax.jit, static_argnames=())
def _compiled_refresh_dual_planner_route(
    *,
    allow_split_build_flag: Array,
    grouped_interactions_flag: Array,
    need_traversal_result_flag: Array,
    leaf_count: Array,
    need_node_interactions_flag: Array,
    need_compact_far_pairs_flag: Array,
    use_dense_interactions_flag: Array,
) -> tuple[Array, Array, Array]:
    """Return compiled routing decisions for refresh dual-artifact planning.

    This keeps steady-state route/plan branching in JAX control flow so the
    refresh hot path avoids repeated Python-side conditional orchestration.

    A pair policy no longer routes away from the split build; it is threaded into
    it instead. That also closes a hole: the caller's planner cache key never
    included the policy flags, so a routing decision cached by a no-policy call
    could be replayed for a policy call, sending it down a split build that
    dropped the policy silently.

    Parameters
    ----------
    allow_split_build_flag : Array
        Traced flag: whether a split build is permitted.
    grouped_interactions_flag : Array
        Traced flag: whether grouped interactions are active.
    need_traversal_result_flag : Array
        See the module docstring.
    leaf_count : Array
        See the module docstring.
    need_node_interactions_flag : Array
        See the module docstring.
    need_compact_far_pairs_flag : Array
        See the module docstring.
    use_dense_interactions_flag : Array
        See the module docstring.

    Returns
    -------
    tuple[Array, Array, Array]
        The routing decisions as traced values, so the refresh path can branch without a host sync.
    """

    use_split_build = (
        allow_split_build_flag
        & (~grouped_interactions_flag)
        & (~need_traversal_result_flag)
    )
    need_far_payload = (
        need_node_interactions_flag
        | need_compact_far_pairs_flag
        | use_dense_interactions_flag
    )
    use_compact_shared_far_near = (
        use_split_build & need_far_payload & (~need_node_interactions_flag)
    )
    suppress_substage_timing = use_split_build & (leaf_count >= jnp.int32(1))
    return use_split_build, use_compact_shared_far_near, suppress_substage_timing


def _without_grouped_class_segments(
    entry: _InteractionCacheEntry,
) -> _InteractionCacheEntry:
    """Drop cached class-major schedule arrays from an interaction cache entry.

    Parameters
    ----------
    entry : _InteractionCacheEntry
        Cache entry being transformed.

    Returns
    -------
    _InteractionCacheEntry
        The entry with its class-major schedule arrays dropped.
    """
    return _InteractionCacheEntry(
        key=entry.key,
        interactions=entry.interactions,
        neighbor_list=entry.neighbor_list,
        dual_tree_result=entry.dual_tree_result,
        compact_far_pairs=entry.compact_far_pairs,
        grouped_buffers=entry.grouped_buffers,
        grouped_segment_starts=None,
        grouped_segment_lengths=None,
        grouped_segment_class_ids=None,
        grouped_segment_sort_permutation=None,
        grouped_segment_group_ids=None,
        grouped_segment_unique_targets=None,
        grouped_chunk_size=None,
        nearfield_target_leaf_ids=entry.nearfield_target_leaf_ids,
        nearfield_source_leaf_ids=entry.nearfield_source_leaf_ids,
        nearfield_valid_pairs=entry.nearfield_valid_pairs,
        nearfield_chunk_sort_indices=entry.nearfield_chunk_sort_indices,
        nearfield_chunk_group_ids=entry.nearfield_chunk_group_ids,
        nearfield_chunk_unique_indices=entry.nearfield_chunk_unique_indices,
        nearfield_mode=entry.nearfield_mode,
        nearfield_edge_chunk_size=entry.nearfield_edge_chunk_size,
        nearfield_leaf_cap=entry.nearfield_leaf_cap,
    )


def _dual_tree_cache_lookup(
    *,
    cache_key: Optional[str],
    cache_entry: Optional[_InteractionCacheEntry],
    need_traversal_result: bool,
    need_compact_far_pairs: bool,
    need_node_interactions: bool,
    precompute_grouped_class_segments: bool,
) -> Optional[_DualTreeCacheHit]:
    """Return reusable cached dual-tree artifacts when available.

    Parameters
    ----------
    cache_key : Optional[str]
        Key for the interaction cache, or ``None`` to bypass it.
    cache_entry : Optional[_InteractionCacheEntry]
        Existing cache entry, or ``None`` on a miss.
    need_traversal_result : bool
        Whether the full walk result must be retained.
    need_compact_far_pairs : bool
        Whether the compact tagged far-pair payload is required.
    need_node_interactions : bool
        Whether a node interaction list must be emitted.
    precompute_grouped_class_segments : bool
        Whether class-major schedules are materialised now.

    Returns
    -------
    Optional[_DualTreeCacheHit]
        The reusable cached payload, or ``None`` when nothing matches the request.
    """

    if not (
        cache_key is not None
        and cache_entry is not None
        and cache_entry.key == cache_key
        and (not need_traversal_result or cache_entry.dual_tree_result is not None)
        and (not need_compact_far_pairs or cache_entry.compact_far_pairs is not None)
        and (not need_node_interactions or cache_entry.interactions is not None)
    ):
        return None

    grouped_segment_starts = cache_entry.grouped_segment_starts
    grouped_segment_lengths = cache_entry.grouped_segment_lengths
    grouped_segment_class_ids = cache_entry.grouped_segment_class_ids
    grouped_segment_sort_permutation = cache_entry.grouped_segment_sort_permutation
    grouped_segment_group_ids = cache_entry.grouped_segment_group_ids
    grouped_segment_unique_targets = cache_entry.grouped_segment_unique_targets
    grouped_chunk_size_cached = cache_entry.grouped_chunk_size
    cache_out: Optional[_InteractionCacheEntry] = cache_entry
    if not precompute_grouped_class_segments and (
        grouped_segment_starts is not None
        or grouped_segment_lengths is not None
        or grouped_segment_class_ids is not None
        or grouped_segment_sort_permutation is not None
        or grouped_segment_group_ids is not None
        or grouped_segment_unique_targets is not None
    ):
        cache_out = _without_grouped_class_segments(cache_entry)
        grouped_segment_starts = None
        grouped_segment_lengths = None
        grouped_segment_class_ids = None
        grouped_segment_sort_permutation = None
        grouped_segment_group_ids = None
        grouped_segment_unique_targets = None
        grouped_chunk_size_cached = None
    return _DualTreeCacheHit(
        interactions=cache_entry.interactions,
        neighbor_list=cache_entry.neighbor_list,
        traversal_result=cache_entry.dual_tree_result,
        compact_far_pairs=cache_entry.compact_far_pairs,
        grouped_buffers=cache_entry.grouped_buffers,
        grouped_segment_starts=grouped_segment_starts,
        grouped_segment_lengths=grouped_segment_lengths,
        grouped_segment_class_ids=grouped_segment_class_ids,
        grouped_segment_sort_permutation=grouped_segment_sort_permutation,
        grouped_segment_group_ids=grouped_segment_group_ids,
        grouped_segment_unique_targets=grouped_segment_unique_targets,
        grouped_chunk_size_cached=grouped_chunk_size_cached,
        cache_out=cache_out,
    )


def _dual_tree_build_raw(
    *,
    tree: Tree,
    geometry: TreeGeometry,
    theta: float,
    mac_type: MACType,
    dehnen_radius_scale: float,
    max_pair_queue: Optional[int],
    pair_process_block: Optional[int],
    traversal_config: Optional[DualTreeTraversalConfig],
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]],
    fail_fast: bool,
    need_traversal_result: bool,
    need_compact_far_pairs: bool,
    need_node_interactions: bool,
    grouped_interactions: bool,
    pair_policy: Optional[PairPolicy],
    policy_state: Optional[AdaptivePolicyState],
    jit_traversal: bool,
) -> tuple[Any, Optional[DualTreeTraversalConfig], Optional[int], Optional[int]]:
    """Run the raw dual-tree traversal builder with retry growth.

    The single point where jaccpot calls yggdrax's traversal. Everything above
    it -- caching, splitting, buffer construction -- is arrangement around this
    one call.

    Parameters
    ----------
    tree : Tree
        Built tree.
    geometry : TreeGeometry
        Node centres and radii the MAC is evaluated against.
    theta : float
        Opening angle.
    mac_type : MACType
        Geometric criterion the traversal evaluates. jaccpot's Dehnen policies
        must already be mapped to a yggdrax literal -- see
        ``PolicyMixin._mac_type_for_traversal``.
    dehnen_radius_scale : float
        Radius inflation for the Dehnen MAC.
    max_pair_queue : Optional[int]
        Cap on the pending-pair queue; ``None`` lets the traversal size it.
    pair_process_block : Optional[int]
        Pairs drained per step; ``None`` as above.
    traversal_config : Optional[DualTreeTraversalConfig]
        Full traversal capacities, overriding the two above when given.
    retry_logger : Optional[Callable[[DualTreeRetryEvent], None]]
        Sink for capacity-retry events, so a retry stays visible in the caller's
        diagnostics rather than being absorbed here.
    fail_fast : bool
        Raise on a capacity overflow instead of retrying with more room.
    need_traversal_result : bool
        Retain the full walk result.
    need_compact_far_pairs : bool
        Ask for far pairs in compact tagged form.
    need_node_interactions : bool
        Ask for a node interaction list. The streamed lane sets this ``False``
        and reads compact pairs instead.
    grouped_interactions : bool
        Group interactions by displacement class during the walk.
    pair_policy : Optional[PairPolicy]
        Solver-owned per-pair acceptance callable. ``None`` leaves the traversal
        running the geometric MAC alone -- that is the difference between the
        Dehnen mass-dependent criterion being active and not.
    policy_state : Optional[AdaptivePolicyState]
        State the pair policy reads. Meaningless without ``pair_policy``.
    jit_traversal : bool
        Run the compiled traversal.

    Returns
    -------
    tuple[Any, Optional[DualTreeTraversalConfig], Optional[int], Optional[int]]
        ``(walk_result, resolved_config, resolved_pair_queue,
        resolved_process_block)``. The three resolved capacities come back so a
        caller can record what the traversal ACTUALLY used after any retry
        growth, not what was requested.

    Raises
    ------
    RuntimeError
        On a capacity overflow when ``fail_fast`` is set.
    """

    from . import fmm as _runtime_fmm

    current_traversal_config = traversal_config
    current_max_pair_queue = max_pair_queue
    current_pair_process_block = pair_process_block

    if fail_fast:
        # Strict/static lane: avoid Python retry-orchestration entirely.
        try:
            build_out = _runtime_fmm.build_interactions_and_neighbors(
                tree,
                geometry,
                theta=theta,
                mac_type=mac_type,
                dehnen_radius_scale=dehnen_radius_scale,
                max_pair_queue=current_max_pair_queue,
                process_block=current_pair_process_block,
                traversal_config=current_traversal_config,
                retry_logger=retry_logger,
                return_result=need_traversal_result,
                return_compact_far_pairs=need_compact_far_pairs,
                return_interactions=(
                    bool(need_node_interactions) or bool(grouped_interactions)
                ),
                return_grouped=grouped_interactions,
                pair_policy=pair_policy,
                policy_state=policy_state,
            )
        except RuntimeError as exc:
            if _looks_like_capacity_error(exc):
                raise RuntimeError(
                    _format_capacity_error_hint(
                        exc,
                        traversal_config=current_traversal_config,
                        max_pair_queue=current_max_pair_queue,
                        pair_process_block=current_pair_process_block,
                    )
                ) from exc
            raise
        return (
            build_out,
            current_traversal_config,
            current_max_pair_queue,
            current_pair_process_block,
        )

    last_exc: Optional[BaseException] = None
    build_out = None
    for attempt in range(_CAPACITY_RETRY_MAX_ATTEMPTS + 1):
        try:
            build_out = _runtime_fmm.build_interactions_and_neighbors(
                tree,
                geometry,
                theta=theta,
                mac_type=mac_type,
                dehnen_radius_scale=dehnen_radius_scale,
                max_pair_queue=current_max_pair_queue,
                process_block=current_pair_process_block,
                traversal_config=current_traversal_config,
                retry_logger=retry_logger,
                return_result=need_traversal_result,
                return_compact_far_pairs=need_compact_far_pairs,
                return_interactions=(
                    bool(need_node_interactions) or bool(grouped_interactions)
                ),
                return_grouped=grouped_interactions,
                pair_policy=pair_policy,
                policy_state=policy_state,
            )
            break
        except RuntimeError as exc:
            last_exc = exc
            if fail_fast and _looks_like_capacity_error(exc):
                raise RuntimeError(
                    _format_capacity_error_hint(
                        exc,
                        traversal_config=current_traversal_config,
                        max_pair_queue=current_max_pair_queue,
                        pair_process_block=current_pair_process_block,
                    )
                ) from exc
            if (
                attempt >= _CAPACITY_RETRY_MAX_ATTEMPTS
                or not _looks_like_capacity_error(exc)
            ):
                raise
            (
                current_traversal_config,
                current_max_pair_queue,
                current_pair_process_block,
            ) = _next_retry_traversal_settings(
                traversal_config=current_traversal_config,
                max_pair_queue=current_max_pair_queue,
                pair_process_block=current_pair_process_block,
            )
    if build_out is None:
        if last_exc is not None:
            raise RuntimeError(str(last_exc)) from last_exc
        raise RuntimeError(
            "dual-tree traversal build failed without producing artifacts"
        )
    return (
        build_out,
        current_traversal_config,
        current_max_pair_queue,
        current_pair_process_block,
    )


def _dual_tree_unpack_build_output(
    *,
    build_out: Any,
    grouped_interactions: bool,
    need_traversal_result: bool,
    need_compact_far_pairs: bool,
) -> tuple[
    Optional[NodeInteractionList],
    NodeNeighborList,
    Optional[DualTreeWalkResult],
    Optional[CompactTaggedFarPairs],
    Optional[GroupedInteractionBuffers],
]:
    """Normalize raw builder outputs into a fixed tuple.

    Parameters
    ----------
    build_out : Any
        Raw tuple returned by the yggdrax builder.
    grouped_interactions : bool
        Whether the grouped class-major layout is in use.
    need_traversal_result : bool
        Whether the full walk result must be retained.
    need_compact_far_pairs : bool
        Whether the compact tagged far-pair payload is required.

    Returns
    -------
    tuple[Optional[NodeInteractionList], NodeNeighborList, Optional[DualTreeWalkResult], Optional[CompactTaggedFarPairs], Optional[GroupedInteractionBuffers]]
        The raw builder output normalised to a fixed five-tuple, whichever optional payloads were requested.
    """

    if grouped_interactions:
        if need_traversal_result and need_compact_far_pairs:
            (
                interactions,
                neighbor_list,
                traversal_result,
                compact_far_pairs,
                grouped_buffers,
            ) = build_out
        elif need_traversal_result:
            interactions, neighbor_list, traversal_result, grouped_buffers = build_out
            compact_far_pairs = None
        elif need_compact_far_pairs:
            interactions, neighbor_list, compact_far_pairs, grouped_buffers = build_out
            traversal_result = None
        else:
            interactions, neighbor_list, grouped_buffers = build_out
            traversal_result = None
            compact_far_pairs = None
        return (
            interactions,
            neighbor_list,
            traversal_result,
            compact_far_pairs,
            grouped_buffers,
        )

    if need_traversal_result and need_compact_far_pairs:
        interactions, neighbor_list, traversal_result, compact_far_pairs = build_out
    elif need_traversal_result:
        interactions, neighbor_list, traversal_result = build_out
        compact_far_pairs = None
    elif need_compact_far_pairs:
        interactions, neighbor_list, compact_far_pairs = build_out
        traversal_result = None
    else:
        interactions, neighbor_list = build_out
        traversal_result = None
        compact_far_pairs = None
    return interactions, neighbor_list, traversal_result, compact_far_pairs, None


def _can_split_dual_tree_build(
    *,
    split_enabled: bool,
    grouped_interactions: bool,
    need_traversal_result: bool,
) -> bool:
    """Return whether far/near traversal can be built in separate passes.

    This path is meant for the minimum-memory streamed GPU regime where we do
    not need traversal tags/results and can trade extra prepare work for a lower
    peak by never materializing far and near traversal buffers in the same
    kernel.

    It used to decline whenever a ``pair_policy`` or ``policy_state`` was
    installed, which shut the Dehnen mass-dependent MAC out of the lane
    entirely. The three yggdrax entry points the split build calls all take a
    ``pair_policy``; jaccpot simply never passed it. They do now, so the policy
    is carried rather than routed around -- which matters at N >= 10^7, where the
    node-interaction buffers the monolithic build materializes
    (``num_nodes x max_interactions_per_node``) are the binding memory
    constraint.

    Parameters
    ----------
    split_enabled : bool
        Whether the caller permits a split build at all.
    grouped_interactions : bool
        Grouping needs the single combined walk, so it disqualifies the split.
    need_traversal_result : bool
        The full walk result likewise only exists on the combined path.

    Returns
    -------
    bool
        ``True`` only when all three allow it.
    """

    return (
        bool(split_enabled)
        and not bool(grouped_interactions)
        and not bool(need_traversal_result)
    )


def _build_dual_tree_artifacts_split(
    *,
    tree: Tree,
    geometry: TreeGeometry,
    theta: float,
    mac_type: MACType,
    dehnen_radius_scale: float,
    max_pair_queue: Optional[int],
    pair_process_block: Optional[int],
    traversal_config: Optional[DualTreeTraversalConfig],
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]],
    need_node_interactions: bool,
    need_compact_far_pairs: bool,
    use_dense_interactions: bool,
    pair_policy: Optional[PairPolicy],
    policy_state: Optional[AdaptivePolicyState],
    timing_callback: Optional[Callable[[str, float], None]] = None,
) -> _DualTreeArtifacts:
    """Build far and near traversal products in separate Yggdrax calls.

    ``pair_policy``/``policy_state`` have no default: every branch below runs a
    separate traversal, and a branch that forgot to forward the policy would
    silently fall back to the geometric MAC underneath it -- the exact
    "faster and wronger" failure this criterion keeps producing.

    Parameters
    ----------
    tree : Tree
        Built tree.
    geometry : TreeGeometry
        Node centres and radii the MAC is evaluated against.
    theta : float
        Opening angle.
    mac_type : MACType
        Geometric criterion, already mapped to a yggdrax literal.
    dehnen_radius_scale : float
        Radius inflation for the Dehnen MAC.
    max_pair_queue : Optional[int]
        Cap on the pending-pair queue; ``None`` lets the traversal size it.
    pair_process_block : Optional[int]
        Pairs drained per step; ``None`` as above.
    traversal_config : Optional[DualTreeTraversalConfig]
        Full traversal capacities, overriding the two above when given.
    retry_logger : Optional[Callable[[DualTreeRetryEvent], None]]
        Sink for capacity-retry events from either pass.
    need_node_interactions : bool
        Produce a node interaction list.
    need_compact_far_pairs : bool
        Produce compact tagged far pairs.
    use_dense_interactions : bool
        Also materialise dense buffers.
    pair_policy : Optional[PairPolicy]
        Solver-owned per-pair acceptance callable. ``None`` runs the geometric
        MAC alone.
    policy_state : Optional[AdaptivePolicyState]
        State the pair policy reads. Meaningless without ``pair_policy``.
    timing_callback : Optional[Callable[[str, float], None]]
        Per-phase timing sink; the two passes report separately.

    Returns
    -------
    _DualTreeArtifacts
        The same artifact bundle the combined path produces.
    """
    timing_enabled = timing_callback is not None

    def _record(name: str, start: Optional[float]) -> None:
        if timing_enabled and start is not None:
            timing_callback(name, float(time.perf_counter() - start))

    need_far_payload = bool(
        need_node_interactions or need_compact_far_pairs or use_dense_interactions
    )
    interactions: Optional[NodeInteractionList]
    compact_far_pairs: Optional[CompactTaggedFarPairs]
    if need_far_payload and not bool(need_node_interactions or use_dense_interactions):
        stage_t0 = time.perf_counter() if timing_enabled else None
        interactions = None
        compact_far_pairs, neighbor_list = (
            build_compact_far_pairs_and_leaf_neighbor_lists(
                tree,
                geometry,
                theta=theta,
                mac_type=mac_type,
                dehnen_radius_scale=dehnen_radius_scale,
                max_pair_queue=max_pair_queue,
                process_block=pair_process_block,
                traversal_config=traversal_config,
                retry_logger=retry_logger,
                timing_callback=timing_callback,
                pair_policy=pair_policy,
                policy_state=policy_state,
            )
        )
        _record("dual_split_shared_far_pairs_leaf_neighbors", stage_t0)
    elif need_far_payload:
        stage_t0 = time.perf_counter() if timing_enabled else None
        interactions, neighbor_list = build_interactions_and_neighbors_split(
            tree,
            geometry,
            theta=theta,
            max_interactions_per_node=None,
            max_neighbors_per_leaf=(
                int(traversal_config.max_neighbors_per_leaf)
                if traversal_config is not None
                else 2048
            ),
            mac_type=mac_type,
            dehnen_radius_scale=dehnen_radius_scale,
            max_pair_queue=max_pair_queue,
            process_block=pair_process_block,
            traversal_config=traversal_config,
            retry_logger=retry_logger,
            pair_policy=pair_policy,
            policy_state=policy_state,
        )
        _record("dual_split_interactions_and_neighbors", stage_t0)
        compact_far_pairs = None
    else:
        interactions = None
        compact_far_pairs = None
        stage_t0 = time.perf_counter() if timing_enabled else None
        neighbor_list = build_leaf_neighbor_lists(
            tree,
            geometry,
            theta=theta,
            max_neighbors_per_leaf=(
                int(traversal_config.max_neighbors_per_leaf)
                if traversal_config is not None
                else 2048
            ),
            mac_type=mac_type,
            dehnen_radius_scale=dehnen_radius_scale,
            max_pair_queue=max_pair_queue,
            process_block=pair_process_block,
            traversal_config=traversal_config,
            retry_logger=retry_logger,
            pair_policy=pair_policy,
            policy_state=policy_state,
        )
        _record("dual_split_leaf_neighbors", stage_t0)
    stage_t0 = time.perf_counter() if timing_enabled else None
    dense_buffers = _dual_tree_build_dense_buffers(
        tree=tree,
        geometry=geometry,
        interactions=interactions,
        use_dense_interactions=use_dense_interactions,
    )
    _record("dual_split_dense_buffers", stage_t0)
    return _DualTreeArtifacts(
        interactions=interactions,
        neighbor_list=neighbor_list,
        traversal_result=None,
        compact_far_pairs=compact_far_pairs,
        dense_buffers=dense_buffers,
        grouped_buffers=None,
        grouped_segment_starts=None,
        grouped_segment_lengths=None,
        grouped_segment_class_ids=None,
        grouped_segment_sort_permutation=None,
        grouped_segment_group_ids=None,
        grouped_segment_unique_targets=None,
        grouped_chunk_size=None,
    )


# Bounded capacity re-planning for the strict streamed walk. Floors are the
# smallest values the policy ever hands out; the ceiling is where a capacity
# stops being the plausible explanation (2^25 slots is ~268 MB of pair indices)
# and the caller should be told rather than retried at.
_STRICT_STREAMED_QUEUE_FLOOR = 32_768
_STRICT_STREAMED_FAR_PAIR_FLOOR = 131_072
_STRICT_STREAMED_RETRY_LIMIT = 1 << 25
_STRICT_STREAMED_RETRY_ATTEMPTS = 12
# Flat-walk lane: eager floors and ceilings for the two capacities the caller did
# NOT name (a named cap is never widened -- #333's rule). Directed pair counts. The
# ladder width is only transient -- an unnamed list is built at
# `_tight_list_capacity` of the count the eager walk measured -- so the ceilings
# only guard the int32 index range; memory is the real bound.
_FLAT_WALK_NEAR_EDGE_FLOOR = 1 << 21
_FLAT_WALK_NEAR_EDGE_LIMIT = 1 << 30
_FLAT_WALK_FAR_PAIR_LIMIT = 1 << 30
#: The flat walk's queue ceiling. The dual walk's 2^25 stopped one card at 3.2e7
#: Plummer particles, whose peak wavefront is past it (2.2e7 at 2.4e7); 2^28 slots
#: are 4 GiB of queue, allocated only if a walk needs them.
_FLAT_WALK_QUEUE_LIMIT = 1 << 28
#: Default headroom of an unnamed flat-walk list over the count it is sized from
#: (``JACCPOT_FLAT_WALK_CAP_HEADROOM``).
_FLAT_WALK_CAP_HEADROOM = 1.5
#: An unnamed list cap is rounded up to a multiple of this (a smaller one to the
#: next power of two), so a re-prepare whose counts moved a little keeps its width
#: and the compiled step its shapes.
_FLAT_WALK_CAP_GRANULE = 1 << 20


def flat_walk_cap_headroom() -> float:
    """Headroom of an unnamed flat-walk list cap over the eager count.

    ``JACCPOT_FLAT_WALK_CAP_HEADROOM`` (at least 1). The traced refresh inside the
    compiled step cannot grow a list, so the headroom is what the counts may grow
    by along a rollout before the step's capacity flag fires.

    Returns
    -------
    float
        The factor.
    """
    return env_float(
        "JACCPOT_FLAT_WALK_CAP_HEADROOM", _FLAT_WALK_CAP_HEADROOM, minimum=1.0
    )


def _tight_list_capacity(count: int, *, headroom: float, floor: int = 0) -> int:
    """Directed list capacity for ``count`` live entries.

    ``headroom x count`` rounded up to a multiple of ``_FLAT_WALK_CAP_GRANULE``
    (below one granule, to the next power of two), never below ``floor`` -- the
    width an earlier eager pass validated, so a re-prepare never shrinks a width
    the compiled step was built for -- and even.

    Parameters
    ----------
    count : int
        Directed entries the eager walk produced.
    headroom : float
        Factor over ``count``.
    floor : int
        Smallest capacity to return.

    Returns
    -------
    int
        The capacity.
    """
    want = max(2, int(np.ceil(float(headroom) * max(int(count), 1))))
    granule = (
        _FLAT_WALK_CAP_GRANULE
        if want >= _FLAT_WALK_CAP_GRANULE
        else 1 << (want - 1).bit_length()
    )
    cap = max(-(-want // granule) * granule, int(floor))
    return cap + (cap & 1)


def _strict_streamed_retry_diag(grew: list[str]) -> None:
    """Report a successful re-plan, so a silent 2x memory bump is visible.

    Opt-in via JACCPOT_PREPARE_DIAGNOSTICS, matching the other prepare-time
    diagnostics. Silence here would trade one invisible failure for another.

    Parameters
    ----------
    grew : list[str]
        See the module docstring.

    Returns
    -------
    None
        Nothing; it emits the diagnostic as a side effect.
    """

    if not env_flag("JACCPOT_PREPARE_DIAGNOSTICS", False):
        return
    print(
        "[jaccpot.prepare] strict streamed dual-tree walk re-planned: "
        + "; ".join(grew),
        flush=True,
    )


def _build_dual_tree_artifacts_split_strict_streamed(
    *,
    tree: Tree,
    geometry: TreeGeometry,
    theta: float,
    mac_type: MACType,
    dehnen_radius_scale: float,
    max_pair_queue: Optional[int],
    pair_process_block: Optional[int],
    traversal_config: Optional[DualTreeTraversalConfig],
    pair_policy: Optional[PairPolicy],
    policy_state: Optional[AdaptivePolicyState],
    capacity_report: Optional[Callable[[dict], None]] = None,
    max_neighbors_per_leaf_override: Optional[int] = None,
    flat_walk_capacity_floor: Optional[dict] = None,
    extra_overflow: Optional[Array] = None,
) -> _DualTreeArtifacts:
    """Strict static fast-lane: single compact shared far+near build call.

    ``capacity_report`` receives, after a successful build, the capacities the
    walk actually ran with plus (eager only) what it observed: the queue the
    yggdrax ladder settled on, the compact far-pair cap, the far-pair count, the
    longest neighbour row and the total edge count.  The fused traced refresh
    cannot grow capacities -- under ``jit`` the overflow flags are tracers and
    yggdrax returns the truncated result -- so it must be told what the eager
    prepare needed.  ``max_neighbors_per_leaf_override`` is how it is told, and it
    only ever RAISES the cap.  (Found 2026-09-06: at N=200k / leaf 256 the preset
    cap of 256 neighbours per leaf against rows of 781 cut 85 % of the near field
    out of every step after the first, with no diagnostic firing.)

    Only the neighbour cap is carried over, because it is the only one that fails
    SILENTLY: the compact far-pair cap is checked under tracing too
    (``_raw_to_compact_far_pairs`` raises through a debug callback), so a
    too-small one is already loud, and raising a cap the caller named in
    ``JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP`` would turn a deliberate
    memory bound into a no-op.

    This path intentionally avoids generic split-builder host branching and
    callback plumbing. It is valid only for streamed compact far-pairs with no
    dense/grouped/interactions payload requests.

    ``pair_policy``/``policy_state`` are forwarded rather than assumed absent:
    this branch is selected on payload shape (``fail_fast`` + compact far pairs),
    not on whether a criterion is installed, so dropping them here would run the
    geometric MAC under a caller that asked for the Dehnen one.

    Parameters
    ----------
    tree : Tree
        Built tree.
    geometry : TreeGeometry
        Node centres and radii the MAC is evaluated against.
    theta : float
        Opening angle.
    mac_type : MACType
        Geometric criterion, already mapped to a yggdrax literal.
    dehnen_radius_scale : float
        Radius inflation for the Dehnen MAC.
    max_pair_queue : Optional[int]
        Cap on the pending-pair queue; ``None`` lets the traversal size it.
    pair_process_block : Optional[int]
        Pairs drained per step; ``None`` as above.
    traversal_config : Optional[DualTreeTraversalConfig]
        Full traversal capacities, overriding the two above when given.
    pair_policy : Optional[PairPolicy]
        Solver-owned per-pair acceptance callable. ``None`` runs the geometric
        MAC alone.
    policy_state : Optional[AdaptivePolicyState]
        State the pair policy reads. Meaningless without ``pair_policy``.
    capacity_report : Optional[Callable[[dict], None]]
        Called once after a successful build with the capacities the walk ran
        with (queue, far-pair cap, neighbour cap) and, eagerly, what it observed
        (far-pair count, longest neighbour row, total edges). ``None`` skips it.
    max_neighbors_per_leaf_override : Optional[int]
        Floor for the per-leaf neighbour cap; only ever raises it.
    flat_walk_capacity_floor : Optional[dict]
        Floors for the flat-walk lane's directed far and near capacities
        (``compact_far_pair_capacity`` / ``near_edge_capacity``), carried over from
        the eager pass so the traced refresh builds the same widths.  Applies only
        to a capacity the caller did not name in the environment.
    extra_overflow : Optional[Array]
        An upstream capacity flag treated like the walk's own -- today the
        cell-leaf partition's ``leaf_capacity``.

    Returns
    -------
    _DualTreeArtifacts
        Compact far pairs and leaf neighbour lists; no node interaction list and
        no walk result, which is what makes this lane cheap.

    Raises
    ------
    ValueError
        If the request is inconsistent with what this lane can produce.
    RuntimeError
        On a capacity overflow this lane cannot grow out of, and when the
        treecode walk is asked to carry a solver-owned pair policy it cannot.
    Exception
        Re-raised unchanged when the retry loop sees a failure it does not
        recognise as a capacity overflow -- it grows capacities only for the two
        messages it knows, and refuses to guess at anything else.
    """

    if traversal_config is not None:
        max_interactions_per_node = int(traversal_config.max_interactions_per_node)
        max_neighbors_per_leaf = int(traversal_config.max_neighbors_per_leaf)
        max_pair_queue_resolved = int(traversal_config.max_pair_queue)
        process_block_resolved = int(traversal_config.process_block)
    else:
        max_interactions_per_node = 8192
        max_neighbors_per_leaf = 4096
        max_pair_queue_resolved = (
            None if max_pair_queue is None else int(max_pair_queue)
        )
        process_block_resolved = (
            None if pair_process_block is None else int(pair_process_block)
        )
    if max_neighbors_per_leaf_override is not None:
        max_neighbors_per_leaf = max(
            int(max_neighbors_per_leaf), int(max_neighbors_per_leaf_override)
        )

    flat_compact_enabled = os.environ.get(
        "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS", "1"
    ) not in ("0", "false", "False", "off", "OFF")
    compact_far_pair_capacity = None
    if flat_compact_enabled:
        compact_far_pair_capacity = int(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
                str(_STRICT_STREAMED_FAR_PAIR_FLOOR),
            )
        )
        if compact_far_pair_capacity <= 0:
            raise ValueError(
                "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP must be positive"
            )

    # Opt-in: build far/near from the device-resident per-leaf treecode walk
    # instead of the host-iterated yggdrax dual-tree walk (kills the walk launch
    # storm). Default off -> no behaviour change. See _build_treecode_artifacts.
    treecode_enabled = os.environ.get(
        "JACCPOT_STATIC_STRICT_FUSED_TREECODE_WALK", "0"
    ) not in ("0", "false", "False", "off", "OFF")
    # DEFAULT since 2026-09-10: yggdrax's flat-emission wavefront walk with the
    # dual walk's own MAC extents -- the same lists as sets at a fraction of the
    # per-step cost (leaf-64 step at N=200k: 177 -> 63 ms). Set
    # JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK=0 for the traced dual walk. A
    # configuration the flat walk cannot carry (the treecode walk requested, a
    # solver-owned pair policy, a MAC other than bh/dehnen, the non-flat far-pair
    # layout) falls back to the dual walk quietly while the flag is merely
    # defaulted; an EXPLICIT "1" against such a configuration raises, because then
    # the caller asked for a walk it cannot have and silence would hand it the
    # wrong one. See _build_flat_walk_artifacts_strict_streamed.
    flat_walk_raw = os.environ.get("JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK")
    flat_walk_explicit = flat_walk_raw is not None
    flat_walk_enabled = (flat_walk_raw if flat_walk_explicit else "1") not in (
        "0",
        "false",
        "False",
        "off",
        "OFF",
    )
    flat_walk_blocker: Optional[str] = None
    if flat_walk_enabled:
        if treecode_enabled:
            flat_walk_blocker = (
                "JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK and "
                "JACCPOT_STATIC_STRICT_FUSED_TREECODE_WALK are both set; pick one walk."
            )
        elif pair_policy is not None or policy_state is not None:
            flat_walk_blocker = (
                "JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK cannot carry a solver-owned "
                "pair policy (mac_type='dehnen_error' / adaptive_error_model="
                "'dehnen_paper'): the flat walk's acceptance test is the geometric "
                "MAC on per-node extents with no policy seam. Unset the env flag, "
                "or use mac_type='dehnen'."
            )
        elif str(mac_type) not in ("bh", "dehnen"):
            flat_walk_blocker = (
                "JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK supports mac_type 'bh' and "
                f"'dehnen' only, got {mac_type!r}."
            )
        elif compact_far_pair_capacity is None:
            flat_walk_blocker = (
                "JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK needs the flat compact "
                "far-pair layout (JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS=1)."
            )
    if flat_walk_blocker is not None:
        if flat_walk_explicit:
            raise RuntimeError(flat_walk_blocker)
        flat_walk_enabled = False
    if flat_walk_enabled:
        assert compact_far_pair_capacity is not None
        floor = dict(flat_walk_capacity_floor or {})
        # A cap the caller NAMED is a deliberate memory bound: honoured exactly,
        # never widened (#333). An unnamed one starts at its floor -- raised to
        # what an earlier eager pass needed -- and the eager ladder grows it.
        # A validated width is used EXACTLY: it is what the traced refresh must
        # build (the carry's shapes) and where an eager re-prepare starts, and it
        # can sit below the ladder's floor (an unnamed list is sized from its
        # count).
        near_edge_env = os.environ.get(
            "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP"
        )
        near_edge_named = near_edge_env is not None
        near_floor = int(floor.get("near_edge_capacity") or 0)
        if near_edge_named:
            near_edge_capacity = int(near_edge_env)
        else:
            near_edge_capacity = near_floor or _FLAT_WALK_NEAR_EDGE_FLOOR
        far_named = "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP" in os.environ
        far_floor = int(floor.get("compact_far_pair_capacity") or 0)
        far_capacity = int(compact_far_pair_capacity)
        if not far_named and far_floor:
            far_capacity = far_floor
        return _build_flat_walk_artifacts_strict_streamed(
            tree=tree,
            geometry=geometry,
            theta=theta,
            mac_type=mac_type,
            dehnen_radius_scale=dehnen_radius_scale,
            compact_far_pair_capacity=far_capacity,
            near_edge_capacity=near_edge_capacity,
            max_pair_queue=max_pair_queue_resolved,
            capacity_report=capacity_report,
            far_named=far_named,
            near_edge_named=near_edge_named,
            extra_overflow=extra_overflow,
            far_floor=0 if far_named else far_floor,
            near_floor=0 if near_edge_named else near_floor,
        )
    if treecode_enabled:
        if pair_policy is not None or policy_state is not None:
            # The treecode walk evaluates its own device-resident `_mac_ok` from
            # per-node extents; there is no seam for a solver-owned pair policy.
            # Running it anyway would answer the geometric MAC while the caller
            # asked for the Dehnen mass-dependent one, and cost nothing visible.
            raise RuntimeError(
                "JACCPOT_STATIC_STRICT_FUSED_TREECODE_WALK cannot carry a "
                "solver-owned pair policy (mac_type='dehnen_error' / "
                "adaptive_error_model='dehnen_paper'): its acceptance test is a "
                "per-node geometric extent comparison with no policy seam, so "
                "the criterion would be silently replaced by the geometric MAC. "
                "Unset the env flag, or use mac_type='dehnen'."
            )
        return _build_treecode_artifacts_strict_streamed(
            tree=tree,
            geometry=geometry,
            theta=theta,
            mac_type=mac_type,
            dehnen_radius_scale=dehnen_radius_scale,
            compact_far_pair_capacity=compact_far_pair_capacity,
        )

    # Re-plan on a capacity overflow instead of surfacing it. The generic
    # `_build_dual_tree_artifacts` has retried since it was written; this strict
    # streamed path -- the one `large_n_gpu`/static_radix actually takes -- never
    # did, so a capacity ceiling reached the caller as a hard wall. Measured on an
    # A100, large_n_gpu/static_radix/leaf 256/order 4/theta 0.6: N=262144 raised
    # "Pair queue capacity exceeded" with the queue resolved to 65536, while the
    # device had 29.62 GiB free and the largest static buffer was 0.31 GiB. It was
    # never a memory limit. Growing the queue then hit the NEXT flat constant, the
    # compact far-pair cap of 131072 -- same shape of bug, one layer down.
    #
    # Both are grown by doubling rather than predicted, because predicting either
    # needs a model of the interaction-list length that theta, the distribution
    # and the leaf size all move. Each retry is one extra traversal; the queue
    # costs ~8 B a slot and a far pair ~8 B, so even the last rung is megabytes.
    attempt_queue = max_pair_queue_resolved
    attempt_far_cap = compact_far_pair_capacity
    grew: list[str] = []
    # The eager yggdrax ladder may settle on a LARGER queue than requested; the
    # "success" event is the only place that number is reported, and the traced
    # refresh needs it (see ``capacity_report``).
    ladder_success: dict = {}

    def _capture_ladder(event: DualTreeRetryEvent) -> None:
        if str(event.status) == "success":
            ladder_success.update(
                queue_capacity=int(event.queue_capacity),
                interaction_capacity=int(event.interaction_capacity),
                far_pair_count=int(event.far_pair_count),
                near_pair_count=int(event.near_pair_count),
            )

    for attempt in range(_STRICT_STREAMED_RETRY_ATTEMPTS):
        try:
            (
                compact_far_pairs,
                neighbor_list,
            ) = build_compact_far_pairs_and_leaf_neighbor_lists(
                tree,
                geometry,
                theta=theta,
                mac_type=mac_type,
                dehnen_radius_scale=dehnen_radius_scale,
                max_interactions_per_node=max_interactions_per_node,
                max_neighbors_per_leaf=max_neighbors_per_leaf,
                max_pair_queue=attempt_queue,
                process_block=process_block_resolved,
                traversal_config=None,
                retry_logger=_capture_ladder,
                timing_callback=None,
                compact_far_pair_capacity=attempt_far_cap,
                pair_policy=pair_policy,
                policy_state=policy_state,
            )
            if grew:
                _strict_streamed_retry_diag(grew)
            break
        except Exception as exc:
            text = str(exc).lower()
            if "queue capacity exceeded" in text:
                which, current = "max_pair_queue", (
                    _STRICT_STREAMED_QUEUE_FLOOR
                    if attempt_queue is None
                    else int(attempt_queue)
                )
            elif "compact far-pair cap exceeded" in text:
                which, current = "compact_far_pair_capacity", (
                    _STRICT_STREAMED_FAR_PAIR_FLOOR
                    if attempt_far_cap is None
                    else int(attempt_far_cap)
                )
            else:
                raise
            grown = current * 2
            if (
                attempt == _STRICT_STREAMED_RETRY_ATTEMPTS - 1
                or grown > _STRICT_STREAMED_RETRY_LIMIT
            ):
                raise RuntimeError(
                    f"{which} overflowed on the strict streamed dual-tree walk and "
                    f"re-planning did not fit it: grew to {current} (ceiling "
                    f"{_STRICT_STREAMED_RETRY_LIMIT}) over {attempt + 1} attempts"
                    + (f" after also growing {', '.join(grew)}" if grew else "")
                    + ". Raise leaf_size to reduce the leaf count, or set "
                    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP / pass "
                    "jaccpot.TraversalOverrides(max_pair_queue=...) explicitly. "
                    "A TraversalOverrides merges onto the preset's other tuning "
                    "rather than replacing it."
                ) from exc
            grew.append(f"{which} {current}->{grown}")
            if which == "max_pair_queue":
                attempt_queue = grown
            else:
                attempt_far_cap = grown
    else:  # pragma: no cover - the loop always breaks or raises
        raise RuntimeError("strict streamed dual-tree walk did not run")
    if capacity_report is not None:
        counts_arr = getattr(neighbor_list, "counts", None)
        offsets_arr = getattr(neighbor_list, "offsets", None)
        fp_count = getattr(compact_far_pairs, "far_pair_count", None)
        traced = isinstance(counts_arr, Tracer) or isinstance(fp_count, Tracer)
        report = dict(
            traced=bool(traced),
            max_pair_queue_requested=(
                None if attempt_queue is None else int(attempt_queue)
            ),
            queue_capacity=int(
                ladder_success.get(
                    "queue_capacity",
                    (
                        _STRICT_STREAMED_QUEUE_FLOOR
                        if attempt_queue is None
                        else int(attempt_queue)
                    ),
                )
            ),
            compact_far_pair_capacity=(
                None if attempt_far_cap is None else int(attempt_far_cap)
            ),
            max_neighbors_per_leaf_used=int(max_neighbors_per_leaf),
            grew=list(grew),
        )
        if not traced:
            try:
                report["far_pair_count"] = (
                    int(fp_count)
                    if fp_count is not None
                    else int(compact_far_pairs.sources.shape[0])
                )
                report["max_neighbors_observed"] = (
                    int(jnp.max(counts_arr)) if int(counts_arr.shape[0]) else 0
                )
                report["total_neighbors"] = (
                    int(offsets_arr[-1]) if int(offsets_arr.shape[0]) else 0
                )
            except Exception:  # pragma: no cover - diagnostics only
                pass
        capacity_report(report)
    return _DualTreeArtifacts(
        interactions=None,
        neighbor_list=neighbor_list,
        traversal_result=None,
        compact_far_pairs=compact_far_pairs,
        dense_buffers=None,
        grouped_buffers=None,
        grouped_segment_starts=None,
        grouped_segment_lengths=None,
        grouped_segment_class_ids=None,
        grouped_segment_sort_permutation=None,
        grouped_segment_group_ids=None,
        grouped_segment_unique_targets=None,
        grouped_chunk_size=None,
    )


def _treecode_neighbor_list(
    prod: Any,
    *,
    num_leaves: int,
    num_internal: int,
    idx_dtype: Any,
) -> NodeNeighborList:
    """Full yggdrax ``NodeNeighborList`` from the treecode producer's near CSR.

    The radix fast lane reads only ``leaf_indices``/``offsets``/``neighbors``/
    ``counts`` and rebuilds target-owned blocks + particle-order maps itself, so the
    remaining fields are cheap valid placeholders (``target_block_size=0``, i.e. no
    prebuilt blocks; ``neighbor_leaf_positions`` empty). This is validated end-to-end
    against direct N-body by tests/experimental/test_treecode_graft_solidfmm.py.

    Parameters
    ----------
    prod : Any
        Treecode walk product. ``Any`` because it is the walk's own result type,
        which this module does not otherwise name.
    num_leaves : int
        Leaf count.
    num_internal : int
        Internal-node count; leaf ids start after these.
    idx_dtype : Any
        Index dtype for the rebuilt arrays.

    Returns
    -------
    NodeNeighborList
        A complete list, with the placeholder fields described above.
    """
    return NodeNeighborList(
        offsets=prod.near_offsets,
        neighbors=prod.near_neighbors,
        leaf_indices=prod.near_leaf_indices,
        counts=prod.near_counts,
        particle_order_leaf_indices=prod.near_leaf_indices,
        particle_order_to_native_leaf=jnp.arange(num_leaves, dtype=idx_dtype),
        neighbor_leaf_positions=jnp.zeros((num_leaves, 0), dtype=idx_dtype),
        target_block_leaf_ids=jnp.zeros((0,), dtype=idx_dtype),
        target_block_source_leaf_ids=jnp.zeros((0, 0), dtype=idx_dtype),
        target_block_valid_mask=jnp.zeros((0, 0), dtype=bool),
        target_block_offsets=jnp.zeros((num_leaves + 1,), dtype=idx_dtype),
        target_block_size=0,
    )


def _raise_if_true(flag: Any, message: str) -> None:
    """Raise ``message`` if ``flag`` is true, under jit (via callback) or eagerly.

    Under a trace the check cannot be a Python ``if``, so it goes through
    ``jax.debug.callback`` -- which means the raise happens at EXECUTION time,
    not trace time, and does not abort tracing.

    Parameters
    ----------
    flag : Any
        Condition. Typed ``Any`` because whether it is a tracer is precisely
        what this function branches on.
    message : str
        Text of the ``RuntimeError``.

    Raises
    ------
    RuntimeError
        When ``flag`` is true.
    """
    if isinstance(flag, Tracer):

        def _callback(value):
            if bool(value):
                raise RuntimeError(message)

        jax.debug.callback(_callback, flag)
    elif bool(flag):
        raise RuntimeError(message)


def _treecode_mac_extents(
    geometry: TreeGeometry,
    parent: Array,
    num_internal: int,
    mac_type: MACType,
    dehnen_radius_scale: float,
    dtype: Any,
) -> Array:
    """Per-node MAC extents matching the yggdrax dual-tree exactly.

    Same recipe as ``_interactions_impl`` (bh: box ``max_extent``; dehnen/engblom:
    bounding-sphere ``radius``), propagated to effective far/leaf extents and, for
    dehnen, scaled by ``dehnen_radius_scale``. The treecode ``_mac_ok`` uses the same
    ``(r_t + r_s)^2 <= theta^2 d^2`` sum form as bh/dehnen, so feeding these extents
    reproduces the dual-tree's far/near acceptance (accuracy-profile parity).

    STABILITY NOTE: the box ``max_extent`` (bh) systematically UNDER-bounds the true
    source radius; the bounding-sphere ``radius`` (dehnen) is the correct (upper) bound
    (the sphere circumscribes the box). Feeding the smaller box extent makes the MAC
    over-accept far pairs -> bh runs at an effectively coarser opening angle than the
    requested theta -> a coherent non-conservative force bias that accumulates into
    secular heating over a multi-step integration (even though geometry is recomputed
    fresh each step). Prefer the sphere (dehnen) extents for multi-step integration; see
    :func:`_build_treecode_artifacts_strict_streamed` and docs/treecode_mac_stability.md.

    Parameters
    ----------
    geometry : TreeGeometry
        Source of box extents and bounding-sphere radii.
    parent : Array
        Parent index per node, for propagating extents upward.
    num_internal : int
        Internal-node count.
    mac_type : MACType
        Selects the extent recipe -- box ``max_extent`` for bh, bounding-sphere
        ``radius`` for dehnen/engblom. See the stability note above: these are
        not interchangeable.
    dehnen_radius_scale : float
        Applied to the dehnen radii only.
    dtype : Any
        Output dtype.

    Returns
    -------
    Array
        Per-node effective extents the treecode ``_mac_ok`` consumes.
    """
    from yggdrax._interactions_impl import (
        _compute_effective_extents,
        _compute_leaf_effective_extents,
    )

    use_sphere = str(mac_type) in ("dehnen", "engblom")
    base = jnp.asarray(
        geometry.radius if use_sphere else geometry.max_extent, dtype=dtype
    )
    eff_far = _compute_effective_extents(parent, base)
    eff_leaf = _compute_leaf_effective_extents(parent, base, int(num_internal))
    if str(mac_type) == "dehnen":
        scale = jnp.asarray(dehnen_radius_scale, dtype=dtype)
        eff_far = scale * eff_far
        eff_leaf = scale * eff_leaf
    node_idx = jnp.arange(base.shape[0])
    return jnp.where(node_idx >= int(num_internal), eff_leaf, eff_far)


def _build_treecode_artifacts_strict_streamed(
    *,
    tree: Tree,
    geometry: TreeGeometry,
    theta: float,
    mac_type: MACType,
    dehnen_radius_scale: float,
    compact_far_pair_capacity: Optional[int],
    near_cap: Optional[int] = None,
) -> _DualTreeArtifacts:
    """Strict fast-lane far/near build from the per-leaf treecode walk.

    Device-resident replacement for
    :func:`_build_dual_tree_artifacts_split_strict_streamed`'s yggdrax walk call,
    gated by ``JACCPOT_STATIC_STRICT_FUSED_TREECODE_WALK`` (default off). The treecode
    yields a different-but-equally-valid interaction set (per-leaf, split-source): far
    targets are always leaves, so the downstream solidfmm L2L cascade acts as a no-op
    (internal locals stay zero -> no double-count). See
    :mod:`jaccpot.experimental.treecode_far_near` and ``benchmark_a100/WALK_SPEC.md``.

    ``mac_extents`` uses the treecode's OWN MAC (env
    ``JACCPOT_STATIC_STRICT_FUSED_TREECODE_MAC``, default ``dual``), selected via
    :func:`_treecode_mac_extents`:

      * ``dual`` (DEFAULT): reproduce the configured dual-tree ``mac_type`` extents
        exactly (for the large-N preset that is ``dehnen`` -> per-node bounding-SPHERE
        radius, ``dehnen_radius_scale``-scaled). This is the physically correct
        multipole-radius bound and gives ACCURACY-PROFILE PARITY with the validated
        dual-tree walk.
      * ``bh``: the treecode's own Barnes-Hut MAC using the axis-aligned box
        ``max_extent`` (box half-width).
      * ``dehnen`` / ``engblom``: force those sphere extents regardless of ``mac_type``.

    WHY ``dual``/dehnen IS THE DEFAULT (dynamic-stability finding, 2026-07-14):
      NOTE this is NOT a stale-geometry bug. Every refresh re-Morton-sorts the particles
      and recomputes ALL node quantities -- centers, bounding-sphere radii, box extents,
      multipoles, far/near lists -- from the CURRENT positions (see
      ``rebuild_static_radix_tree_from_template``, ``use_morton_geometry=False``). Only
      the tree SHAPE is frozen (node index-ranges, leaf count, buffer capacities) to keep
      array shapes constant / avoid recompilation. The bug is a bound-TIGHTNESS issue,
      present on every (freshly recomputed) step:

      The box ``bh`` extent is CHEAPER (smaller extent -> MAC passes more readily ->
      fewer far/M2L pairs -> faster) and STATICALLY looks as accurate as dehnen (t=0 force
      parity vs the dual-tree/direct N-body ~0.03%). But the box ``max_extent`` (max axis
      half-width) is a systematic UNDER-bound of the true source multipole radius: the
      bounding sphere always circumscribes the box (~sqrt(3)x larger for an isotropic
      cloud, more when anisotropic). Feeding the smaller box extent into
      ``(r_t + r_s)^2 <= theta^2 d^2`` makes the MAC accept pairs at smaller ``d`` than the
      sphere would -> bh effectively runs at a COARSER opening angle than the requested
      ``theta`` -> the far field is systematically under-resolved. As an instantaneous
      magnitude that is tiny, but it is a COHERENT, NON-GRADIENT force bias, and
      velocity-Verlet does not conserve energy under a non-conservative force, so it
      ACCUMULATES into secular heating over steps (200k/order-4: max|v| 7 -> 20 -> 142 ->
      >1000 over 300 steps; total energy diverges). The dehnen bounding-SPHERE radius is
      the correct bound, keeps every accepted pair inside the ``theta`` budget, and
      reproduces the dual-tree acceptance -> stable: max|v| and dKE/dLz track the dual-tree
      baseline to <1% over 300 steps. Cost: a modest slowdown (deeper acceptance -> more
      M2L pairs). Set the env knob to ``bh`` ONLY for single-shot / static force
      evaluations where per-step accumulation cannot occur. See
      ``benchmark_a100/WALK_SPEC.md`` and ``docs/treecode_mac_stability.md``.

    Overflow of any per-leaf / flat capacity is surfaced as a ``RuntimeError`` in the
    EAGER prepare pass; inside the traced velocity-Verlet scan the check is skipped (a
    per-step ``jax.debug.callback`` would serialize the device-resident scan), so the
    auto-sized per-leaf caps (>= total node count) plus generous flat caps must stay
    overflow-proof for the whole trajectory.

    Parameters
    ----------
    tree : Tree
        Built tree.
    geometry : TreeGeometry
        Node centres and radii the MAC is evaluated against.
    theta : float
        Opening angle.
    mac_type : MACType
        Geometric criterion, already mapped to a yggdrax literal.
    dehnen_radius_scale : float
        Radius inflation for the Dehnen MAC.
    compact_far_pair_capacity : Optional[int]
        Fixed capacity for the compact far-pair arrays; ``None`` auto-sizes from
        the tree.
    near_cap : Optional[int]
        Per-leaf near-list capacity; ``None`` auto-sizes likewise.

    Returns
    -------
    _DualTreeArtifacts
        Compact far pairs plus a neighbour list rebuilt by
        :func:`_treecode_neighbor_list`.
    """
    from jaccpot.experimental.treecode_far_near import (
        build_treecode_far_pairs_and_neighbors,
    )

    topo = tree.topology
    # Derive counts from STATIC shapes (scan-compatible): inside the strict
    # velocity-Verlet scan the tree is re-fed as a tracer, so int(topo.num_internal_nodes)
    # (a traced value) raises ConcretizationTypeError. left_child has shape
    # [num_internal] and parent has shape [total_nodes]; both are static.
    num_internal = int(topo.left_child.shape[0])
    total_nodes = int(topo.parent.shape[0])
    num_leaves = total_nodes - num_internal
    idx = topo.parent.dtype

    left_full = jnp.concatenate(
        [jnp.asarray(topo.left_child, idx), jnp.full((num_leaves,), -1, idx)]
    )
    right_full = jnp.concatenate(
        [jnp.asarray(topo.right_child, idx), jnp.full((num_leaves,), -1, idx)]
    )
    leaf_nodes = jnp.arange(num_internal, total_nodes, dtype=idx)
    root_idx = jnp.argmin(topo.parent).astype(idx)

    centers = jnp.asarray(geometry.center)
    # Default ``dual`` (reproduce the configured dual-tree MAC extents, i.e. the dehnen
    # bounding-SPHERE radius for the large-N preset). ``bh`` (box max_extent) is faster
    # but DYNAMICALLY UNSTABLE: the box half-width systematically UNDER-bounds the true
    # source multipole radius (sphere circumscribes box) -> the MAC runs at an effectively
    # coarser opening angle than the requested theta -> a coherent non-conservative force
    # bias that velocity-Verlet accumulates into secular heating (blows up over steps).
    # Geometry is recomputed fresh every step; this is a bound-tightness bug, not stale
    # geometry (see the docstring above + docs/treecode_mac_stability.md). ``bh`` is safe
    # only for single-shot/static force evaluations.
    tc_mac = os.environ.get("JACCPOT_STATIC_STRICT_FUSED_TREECODE_MAC", "dual").strip()
    walk_mac_type = mac_type if tc_mac == "dual" else tc_mac
    mac_extents = _treecode_mac_extents(
        geometry,
        topo.parent,
        num_internal,
        walk_mac_type,
        dehnen_radius_scale,
        centers.dtype,
    )

    def _env_int(name, default):
        return int(os.environ.get(name, str(default)))

    # Auto-size the treecode caps from the tree so the walk runs zero-config
    # (explicit env overrides still win). A leaf's far/near list can never exceed
    # the node count, so 4*num_leaves (>= total_nodes ~ 2*num_leaves for a binary
    # radix tree) makes per-leaf overflow impossible while staying modest memory.
    # near_cap defaults to the downstream neighbor-edge capacity (the treecode's
    # bh-MAC near split feeds the same near-field buffer, so this is the correct
    # bound); far_cap is the compact far-pair capacity. Defaults chosen so the
    # 200k/order4 fast lane runs without hand-tuning (old fixed 2048/256/1<<20
    # overflowed on the bh-MAC near split).
    auto_per_leaf = max(4096, 4 * int(num_leaves))
    neighbor_edge_cap = _env_int(
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", 1 << 21
    )
    max_far = _env_int(
        "JACCPOT_STATIC_STRICT_FUSED_TREECODE_FAR_PER_LEAF", auto_per_leaf
    )
    max_near = _env_int(
        "JACCPOT_STATIC_STRICT_FUSED_TREECODE_NEAR_PER_LEAF", auto_per_leaf
    )
    max_stack = _env_int("JACCPOT_STATIC_STRICT_FUSED_TREECODE_STACK", 512)
    # An explicit ``near_cap`` (right-sized by the caller, e.g. the distributed driver
    # to ~max_neighbors_per_leaf * num_leaves) overrides the env/1<<21 default. The
    # 1<<21 default keeps the single-GPU fast lane byte-identical when no cap is passed;
    # the fixed 2M buffer both wastes the downstream neighbour build and can SILENTLY
    # truncate at very large N (the overflow guard below is eager-only). Callers that
    # trace this (shard_map) should pass an explicit, validated ``near_cap``.
    if near_cap is not None:
        near_cap = int(near_cap)
    else:
        near_cap = _env_int(
            "JACCPOT_STATIC_STRICT_FUSED_TREECODE_NEAR_CAP", neighbor_edge_cap
        )
    far_cap = int(compact_far_pair_capacity) if compact_far_pair_capacity else 131072

    prod = build_treecode_far_pairs_and_neighbors(
        leaf_nodes,
        centers,
        mac_extents,
        left_full,
        right_full,
        jnp.asarray(float(theta) * float(theta), centers.dtype),
        root_idx,
        num_internal=num_internal,
        max_far=max_far,
        max_near=max_near,
        max_stack=max_stack,
        max_iters=total_nodes + 1,
        far_pair_capacity=far_cap,
        near_capacity=near_cap,
        idx_dtype=idx,
    )
    # Overflow guard: EAGER-ONLY. Inside the device-resident velocity-Verlet
    # scan `prod.overflow` is a tracer, and _raise_if_true would emit a
    # jax.debug.callback -- an ordered per-step host round-trip that serializes
    # the scan and starves the GPU (the whole fused runner is otherwise
    # device-resident). We skip it in the traced path: the auto-sized per-leaf
    # caps (max_far/max_near = 4*num_leaves >= total_nodes) make per-leaf
    # overflow structurally impossible, and the eager prepare pass (concrete
    # `prod.overflow`) validates the flat far/near caps once before the scan is
    # compiled. See docs/phase5_m2l_a100_findings_and_padding_plan.md.
    if not isinstance(prod.overflow, Tracer):
        _raise_if_true(
            prod.overflow,
            "treecode walk overflowed a capacity (far/near per-leaf, stack, or "
            "flat far/near cap). Raise JACCPOT_STATIC_STRICT_FUSED_TREECODE_FAR_PER_LEAF"
            " / NEAR_PER_LEAF / STACK / NEAR_CAP or COMPACT_FAR_PAIR_CAP.",
        )

    compact_far_pairs = CompactTaggedFarPairs(
        sources=prod.far_sources,
        targets=prod.far_targets,
        tags=prod.far_tags,
        far_pair_count=prod.far_pair_count,
    )
    neighbor_list = _treecode_neighbor_list(
        prod, num_leaves=num_leaves, num_internal=num_internal, idx_dtype=idx
    )
    return _DualTreeArtifacts(
        interactions=None,
        neighbor_list=neighbor_list,
        traversal_result=None,
        compact_far_pairs=compact_far_pairs,
        dense_buffers=None,
        grouped_buffers=None,
        grouped_segment_starts=None,
        grouped_segment_lengths=None,
        grouped_segment_class_ids=None,
        grouped_segment_sort_permutation=None,
        grouped_segment_group_ids=None,
        grouped_segment_unique_targets=None,
        grouped_chunk_size=None,
    )


def strict_walk_backend() -> str:
    """Walk implementation of the strict flat lane (plan sub-10ms Phase 2).

    ``JACCPOT_STATIC_STRICT_FUSED_WALK``: ``"pallas"`` (one Pallas launch per
    round, :func:`jaccpot.pallas.mutual_walk_pallas.mutual_walk_pallas`) or
    ``"flat"`` (yggdrax ``dual_tree_walk_mutual``). Unset: ``"pallas"`` wherever
    the kernel lowers (an Ampere+ GPU; plan sub-10ms Phase 6, 2026-09-11),
    ``"flat"`` elsewhere.

    Returns
    -------
    str
        ``"flat"`` or ``"pallas"``.

    Raises
    ------
    ValueError
        On any other value.
    """
    raw = os.environ.get("JACCPOT_STATIC_STRICT_FUSED_WALK")
    if raw is None:
        if env_flag("JACCPOT_WALK_PALLAS_INTERPRET", False):
            return "pallas"
        from jaccpot.pallas.m2l_real_csr import pallas_m2l_real_csr_supported

        return "pallas" if pallas_m2l_real_csr_supported() else "flat"
    raw = raw.strip().lower()
    if raw not in ("flat", "pallas"):
        raise ValueError(
            f"JACCPOT_STATIC_STRICT_FUSED_WALK={raw!r}; expected 'flat' or 'pallas'"
        )
    return raw


def strict_walk_deterministic_rows() -> bool:
    """Whether the Pallas walk's rows are sorted by (target, source).

    ``JACCPOT_STATIC_STRICT_FUSED_WALK_DETERMINISTIC`` (default on): the Pallas
    walk emits in atomic order, so without the sort the fp32 summation order of
    every M2L and near-field row -- hence the forces at ~1e-7 -- would vary
    between runs. Off saves one sort of the far list.

    Returns
    -------
    bool
        ``True`` unless the flag is ``0``/``off``/``false``.
    """
    raw = os.environ.get("JACCPOT_STATIC_STRICT_FUSED_WALK_DETERMINISTIC", "1")
    return raw.strip().lower() not in ("0", "off", "false")


def _lex_sorted(
    primary: Array, secondary: Array, *, primary_bound: int, secondary_bound: int
) -> tuple[Array, Array]:
    """``(primary, secondary)`` sorted lexicographically, ascending.

    One keys-only sort of the composite ``primary * secondary_bound + secondary``,
    decoded afterwards: int32 when the bounds' product fits, int64 otherwise (x64
    is on wherever yggdrax is imported). At the 8M flat walk's sizes this is HALF
    the time of two stable int32 argsorts (6.4 against 13.3 ms for 28.8M near
    entries, identical output), and needs neither a permutation nor the gathers
    that apply it. A two-key ``lax.sort`` is 5x slower again: XLA hands only
    single-key sorts to CUB.

    Parameters
    ----------
    primary : Array
        Non-negative primary keys, all below ``primary_bound``.
    secondary : Array
        Non-negative secondary keys, all below ``secondary_bound``.
    primary_bound : int
        Static exclusive bound on ``primary``.
    secondary_bound : int
        Static exclusive bound on ``secondary``.

    Returns
    -------
    tuple[Array, Array]
        The sorted keys, in the dtypes of ``primary`` and ``secondary``.
    """
    pdt, sdt = primary.dtype, secondary.dtype
    if int(primary_bound) * int(secondary_bound) < 2**31:
        wide = jnp.int32
    elif jax.config.jax_enable_x64:
        wide = jnp.int64
    else:  # pragma: no cover - the lane always runs with x64 on
        p1 = jnp.argsort(secondary, stable=True, dtype=jnp.int32)
        p2 = jnp.argsort(primary[p1], stable=True, dtype=jnp.int32)
        perm = p1[p2]
        return primary[perm], secondary[perm]
    bound = jnp.asarray(int(secondary_bound), wide)
    composite = jnp.sort(primary.astype(wide) * bound + secondary.astype(wide))
    return (composite // bound).astype(pdt), (composite % bound).astype(sdt)


class TargetSortedFarPairs(CompactTaggedFarPairs):
    """A far list in CSR order whose ``targets`` field holds the ROW OFFSETS.

    The deterministic flat walk emits its far list in CSR order: every target's
    sources contiguous and ascending (:func:`_directed_csr_from_canonical`), the
    live entries a prefix. So the list is ``sources`` plus ``total_nodes + 1`` row
    offsets, and ``targets`` holds those offsets rather than one target per entry
    (which was 2W int32 live from the build through the M2L). The M2L reads it as
    the CSR it is (``csr_by_target(presorted=True)``); anything wanting one target
    per entry calls :func:`far_pair_targets`. A field-less subclass: the same
    pytree fields, kept through ``jit`` and ``scan`` as the node type.
    """

    __slots__ = ()


def far_pair_targets(far_pairs: CompactTaggedFarPairs) -> Array:
    """One target per entry of a compact far list (``-1`` on padding).

    A :class:`TargetSortedFarPairs` list stores its row offsets in ``targets``;
    they are expanded here. Any other list returns its ``targets`` as they are.

    Parameters
    ----------
    far_pairs : CompactTaggedFarPairs
        A compact far list.

    Returns
    -------
    Array
        ``[len(far_pairs.sources)]`` target node ids.
    """
    if isinstance(far_pairs, TargetSortedFarPairs):
        from jaccpot.pallas.m2l_real_csr import targets_from_csr_offsets

        return targets_from_csr_offsets(
            far_pairs.targets, int(jnp.asarray(far_pairs.sources).shape[0])
        )
    return jnp.asarray(far_pairs.targets)


def _list_csr_kernel() -> str:
    """How the deterministic walk builds its directed CSR lists.

    ``JACCPOT_LIST_CSR_KERNEL``: ``pallas`` (counts, atomic placement and in-row
    ranks, :mod:`jaccpot.pallas.csr_place`: 16 B per canonical slot), ``xla``
    (two sorts over the pairs: 24 B per slot at the placement), ``interpret``
    (the Pallas route in interpret mode, for CPU tests) or ``auto`` (the default:
    ``pallas`` where it lowers, else ``xla``). Read at trace time.

    Returns
    -------
    str
        ``"pallas"``, ``"xla"`` or ``"interpret"``.
    """
    from jaccpot._env import env_choice
    from jaccpot.pallas.csr_place import pallas_directed_csr_supported

    choice = env_choice(
        "JACCPOT_LIST_CSR_KERNEL", "auto", ("auto", "pallas", "xla", "interpret")
    )
    if choice == "auto":
        return "pallas" if pallas_directed_csr_supported() else "xla"
    return choice


def _directed_csr_from_canonical(
    a: Array,
    b: Array,
    live: Array,
    *,
    row_offset: int,
    num_rows: int,
    idx: Any,
    pad_source: int,
    with_targets: bool,
) -> tuple[Array, Optional[Array], Array, Array]:
    """The directed list of CANONICAL pairs ``(a, b)``, ``a < b``, sorted by (target, source).

    Each canonical pair is the two directed entries ``b -> a`` and ``a -> b``. Sorted
    by ``(target, source)``, row ``t`` holds first the ``a`` of every pair ``(a, t)``
    (``a < t``), then the ``b`` of every pair ``(t, b)`` (``b > t``), each ascending.
    Where Pallas lowers (:func:`_list_csr_kernel`) no sort at all: counts, atomic
    placement and in-row ranks (:mod:`jaccpot.pallas.csr_place`). Otherwise,
    instead of one composite sort over the ``2W`` directed entries: one composite
    sort of the ``W`` canonical pairs by ``(a, b)`` (the rows' high halves, in
    order), one stable int32 key-value re-sort of them by ``b`` (the low halves,
    ``a`` ascending by stability), counts and offsets from both, and one
    unique-index placement. The same arrays to the bit as the directed sort.

    Parameters
    ----------
    a : Array
        ``(W,)`` lower node of each canonical pair (dead slots arbitrary).
    b : Array
        ``(W,)`` upper node.
    live : Array
        ``(W,)`` live slots, a prefix.
    row_offset : int
        Node id of row 0 (``num_internal`` for leaf rows, 0 for node rows). Static.
    num_rows : int
        Rows. Static.
    idx : Any
        Index dtype. Static.
    pad_source : int
        Source value past the live entries (0 for the near CSR, -1 for far pairs).
    with_targets : bool
        Also return the target node of every entry (``-1`` past the live ones).

    Returns
    -------
    tuple[Array, Optional[Array], Array, Array]
        ``(sources (2W,), targets (2W,) or None, offsets (R + 1,), counts (R,))``,
        node ids, live entries a prefix.

    Raises
    ------
    ValueError
        If ``3 W`` (the dropped lanes' out-of-range slots) overflows ``idx``.
    """
    width = int(a.shape[0])
    R = int(num_rows)
    if 3 * width >= int(jnp.iinfo(idx).max):
        raise ValueError(f"pair width {width} overflows {idx}")
    route = _list_csr_kernel()
    if route != "xla" and width > 0 and R > 0:
        # no sort over the pairs: counts, atomic placement, in-row ranks (the
        # same arrays to the bit; jaccpot.pallas.csr_place)
        from jaccpot.pallas.csr_place import directed_csr_pallas
        from jaccpot.pallas.m2l_real_csr import targets_from_csr_offsets

        sources, offsets, counts = directed_csr_pallas(
            a,
            b,
            jnp.sum(jnp.asarray(live, idx)),
            num_rows=R,
            row_offset=int(row_offset),
            pad_source=int(pad_source),
            idx=idx,
            interpret=route == "interpret",
        )
        targets = None
        if with_targets:
            rows = targets_from_csr_offsets(offsets, 2 * width)
            targets = jnp.where(rows >= 0, rows + jnp.asarray(row_offset, idx), -1)
        return sources, targets, offsets, counts
    dead = jnp.asarray(R, idx)
    ra = jnp.where(live, a - row_offset, dead).astype(idx)
    rb = jnp.where(live, b - row_offset, dead).astype(idx)
    # rows' high halves: canonical pairs by (a, b), dead (R, R) last
    a_s, b_s = _lex_sorted(ra, rb, primary_bound=R + 1, secondary_bound=R + 1)
    # rows' low halves: the same pairs stably by b (a stays ascending per b)
    b_t, a_t = jax.lax.sort((b_s, a_s), num_keys=1, is_stable=True)
    rows = jnp.arange(R + 1, dtype=idx)
    start_hi = jnp.searchsorted(
        a_s, rows, side="left", method=searchsorted_method()
    ).astype(idx)
    start_lo = jnp.searchsorted(
        b_t, rows, side="left", method=searchsorted_method()
    ).astype(idx)
    n_hi = start_hi[1:] - start_hi[:-1]
    n_lo = start_lo[1:] - start_lo[:-1]
    counts = (n_lo + n_hi).astype(idx)
    offsets = jnp.concatenate([jnp.zeros((1,), idx), jnp.cumsum(counts).astype(idx)])
    i = jnp.arange(width, dtype=idx)
    # dead entries point past the list at distinct slots and are dropped
    past = jnp.asarray(2 * width, idx) + i
    t_lo = jnp.minimum(b_t, R - 1)
    slot_lo = jnp.where(b_t < R, offsets[t_lo] + (i - start_lo[t_lo]), past)
    t_hi = jnp.minimum(a_s, R - 1)
    slot_hi = jnp.where(
        a_s < R, offsets[t_hi] + n_lo[t_hi] + (i - start_hi[t_hi]), past
    )

    def _place(lo_values: Array, hi_values: Array, fill: int) -> Array:
        out = jnp.full((2 * width,), fill, idx)
        out = out.at[slot_lo].set(
            (lo_values + row_offset).astype(idx), mode="drop", unique_indices=True
        )
        return out.at[slot_hi].set(
            (hi_values + row_offset).astype(idx), mode="drop", unique_indices=True
        )

    sources = _place(a_t, b_s, pad_source)
    targets = _place(b_t, a_s, -1) if with_targets else None
    return sources, targets, offsets, counts


def _near_csr_from_canonical(
    near_a: Array,
    near_b: Array,
    live: Array,
    *,
    num_internal: int,
    num_leaves: int,
    idx: Any,
) -> tuple[Array, Array, Array]:
    """The deterministic leaf-neighbour CSR from the CANONICAL near pairs.

    :func:`_directed_csr_from_canonical` over leaf rows: the walk emits each
    unordered leaf pair once as ``(min, max)`` and never ``(t, t)``. Same
    ``neighbors``, ``offsets`` and ``counts`` to the bit as one composite sort of
    the ``2W`` directed slots, at about half the sort bytes (the near-list build
    was the step's peak on dense draws).

    Parameters
    ----------
    near_a : Array
        ``(W,)`` lower node of each canonical near pair (dead slots arbitrary).
    near_b : Array
        ``(W,)`` upper node of each canonical near pair.
    live : Array
        ``(W,)`` live slots.
    num_internal : int
        Internal nodes (leaves are the last ``num_leaves`` nodes). Static.
    num_leaves : int
        Leaves. Static.
    idx : Any
        Index dtype. Static.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(neighbors (2W,), offsets (L + 1,), counts (L,))``: source NODE ids per
        row, ascending, padded with 0 past the live entries.
    """
    neighbors, _, offsets, counts = _directed_csr_from_canonical(
        near_a,
        near_b,
        live,
        row_offset=int(num_internal),
        num_rows=int(num_leaves),
        idx=idx,
        pad_source=0,
        with_targets=False,
    )
    return neighbors, offsets, counts


def _fit_width(x: Array, width: int, fill: int) -> Array:
    """``x`` cut or ``fill``-padded to ``width`` entries (its live part is a prefix).

    Parameters
    ----------
    x : Array
        ``[w]`` list.
    width : int
        Target width. Static.
    fill : int
        Padding value.

    Returns
    -------
    Array
        ``[width]``.
    """
    w = int(x.shape[0])
    if w >= int(width):
        return x[: int(width)]
    return jnp.concatenate([x, jnp.full((int(width) - w,), fill, x.dtype)])


@partial(
    jax.jit,
    static_argnames=(
        "far_width",
        "near_width",
        "num_internal",
        "total_nodes",
        "deterministic",
        "idx",
    ),
)
@jax.named_scope("fmm_lists")
def _flat_walk_lists(
    far_a: Array,
    far_b: Array,
    far_count: Array,
    near_a: Array,
    near_b: Array,
    near_count: Array,
    any_overflow: Array,
    *,
    far_width: int,
    near_width: int,
    num_internal: int,
    total_nodes: int,
    deterministic: bool,
    idx: Any,
) -> tuple[Array, ...]:
    """The flat walk's far list and leaf-neighbour CSR at the given widths.

    One jit, so the EAGER build frees its capacity-width temporaries as XLA goes
    instead of keeping every op-by-op local alive until the builder returns (the
    16M prepare ran out of memory here). Under a trace it inlines.

    Parameters
    ----------
    far_a : Array
        Walk output, lower node of each canonical far pair (live prefix).
    far_b : Array
        Upper node of each canonical far pair.
    far_count : Array
        Live canonical far pairs.
    near_a : Array
        Lower leaf of each canonical near pair (live prefix).
    near_b : Array
        Upper leaf of each canonical near pair.
    near_count : Array
        Live canonical near pairs.
    any_overflow : Array
        Any capacity flag; saturates the far-pair count.
    far_width : int
        Canonical far width (the directed list is twice that). Static.
    near_width : int
        Canonical near width. Static.
    num_internal : int
        Internal nodes (leaves are the last ``total_nodes - num_internal``). Static.
    total_nodes : int
        All nodes. Static.
    deterministic : bool
        Sort the rows by (target, source) (the Pallas walk emits in atomic order).
    idx : Any
        Index dtype. Static.

    Returns
    -------
    tuple[Array, ...]
        ``(far_sources, far_targets, far_tags, far_pair_count, neighbors, offsets,
        counts)``. With ``deterministic``, ``far_targets`` are the far list's
        ``total_nodes + 1`` ROW OFFSETS (a :class:`TargetSortedFarPairs` list);
        otherwise one target per entry.
    """
    num_leaves = total_nodes - num_internal
    # --- far pairs: directed, interleaved, prefix-live, capacity-width ---
    far_live = jnp.arange(far_width, dtype=idx) < far_count
    fa = jnp.where(far_live, _fit_width(far_a, far_width, -1), -1).astype(idx)
    fb = jnp.where(far_live, _fit_width(far_b, far_width, -1), -1).astype(idx)
    if deterministic and total_nodes > 0:
        # the directed list already in CSR order (target, then source ascending):
        # the M2L reads it without sorting it again (TargetSortedFarPairs), and
        # every row lists its sources exactly as the M2L's stable by-target sort of
        # the (a, b)-ordered interleaved list did -- the same sums, bit for bit.
        # The targets travel as the ROW OFFSETS: one target per entry was a 2W
        # array live from this build through the M2L (8 B per canonical slot, at
        # the step's peak); ``far_pair_targets`` expands it where one is needed.
        far_sources, _, far_targets, _ = _directed_csr_from_canonical(
            fa,
            fb,
            far_live,
            row_offset=0,
            num_rows=int(total_nodes),
            idx=idx,
            pad_source=-1,
            with_targets=False,
        )
    else:
        far_sources = jnp.stack([fb, fa], axis=1).reshape((2 * far_width,))
        far_targets = jnp.stack([fa, fb], axis=1).reshape((2 * far_width,))
    # The flat walk assigns no order tags (they were all -1, 2W int32 of them:
    # 1.6 GB at 6.4e7, held by every eager state); a zero-length array says so and
    # the adaptive-order extractor synthesises the -1s if a caller wants them.
    far_tags = jnp.zeros((0,), dtype=idx)
    # Saturate on ANY overflow: the strict runner's guard tests
    # ``far_pair_count < compact_far_pair_capacity`` and this is how the near and
    # queue flags reach it under trace.
    far_pair_count = jnp.where(
        any_overflow,
        jnp.asarray(2 * far_width, idx),
        (2 * far_count).astype(idx),
    )

    # --- near pairs: directed, one stable sort by target leaf, CSR ---
    near_live = jnp.arange(near_width, dtype=idx) < near_count
    na = jnp.where(near_live, _fit_width(near_a, near_width, 0), 0).astype(idx)
    nb = jnp.where(near_live, _fit_width(near_b, near_width, 0), 0).astype(idx)
    if deterministic and num_leaves > 0:
        neighbors, offsets, counts = _near_csr_from_canonical(
            na,
            nb,
            near_live,
            num_internal=num_internal,
            num_leaves=num_leaves,
            idx=idx,
        )
        return (
            far_sources,
            far_targets,
            far_tags,
            far_pair_count,
            neighbors,
            offsets,
            counts,
        )
    tgt = jnp.concatenate([na, nb])
    src = jnp.concatenate([nb, na])
    valid = jnp.concatenate([near_live, near_live])
    # static radix: leaves are the last ``num_leaves`` nodes
    tgt_leaf = tgt - jnp.asarray(num_internal, idx)
    key = jnp.where(valid, tgt_leaf, jnp.asarray(num_leaves, idx))
    src = jnp.where(valid, src, jnp.asarray(0, idx))
    if deterministic:
        sorted_key, neighbors = _lex_sorted(
            key, src, primary_bound=num_leaves + 1, secondary_bound=total_nodes
        )
    else:
        # one key-value sort by target (stable): no permutation, no gathers
        sorted_key, neighbors = jax.lax.sort((key, src), num_keys=1, is_stable=True)
    # the padding sorted to the end with source 0, as the CSR's consumers expect
    # unrolled: the default is a while loop with one small kernel per bisection
    # step (~23 here), launch-bound inside the fused step; unrolled, XLA fuses it
    offsets = jnp.searchsorted(
        sorted_key,
        jnp.arange(num_leaves + 1, dtype=idx),
        side="left",
        method=searchsorted_method(),
    ).astype(idx)
    counts = offsets[1:] - offsets[:-1]
    return (
        far_sources,
        far_targets,
        far_tags,
        far_pair_count,
        neighbors,
        offsets,
        counts,
    )


def _build_flat_walk_artifacts_strict_streamed(
    *,
    tree: Tree,
    geometry: TreeGeometry,
    theta: float,
    mac_type: MACType,
    dehnen_radius_scale: float,
    compact_far_pair_capacity: int,
    near_edge_capacity: int,
    max_pair_queue: Optional[int],
    capacity_report: Optional[Callable[[dict], None]] = None,
    far_named: bool = True,
    near_edge_named: bool = True,
    extra_overflow: Optional[Array] = None,
    far_floor: int = 0,
    near_floor: int = 0,
) -> _DualTreeArtifacts:
    """Far pairs and leaf neighbours from yggdrax's flat-emission wavefront walk.

    The strict fused lane's alternative to ``build_compact_far_pairs_and_leaf_
    neighbor_lists`` (plan "tree walk", 2026-09-10). Measured in isolation on an
    A100 at N=200k / leaf 64 / queue 2^20 the traced dual-tree walk costs 503 ms
    per call and ``dual_tree_walk_mutual`` 40 ms (26 ms with int32 indices) for
    IDENTICAL far and near pair counts: the dual walk carries dense per-node
    output rows (``total_nodes x max_interactions_per_node`` -- 820 MB at leaf 64
    -- plus ``num_leaves x max_neighbors_per_leaf``) through the loop, sorts every
    round to place each pair in its rows, and flattens ``total_nodes x K`` slots
    afterwards; the mutual walk appends each unordered pair once to a flat list
    with a cumsum. Fed the dual walk's own ``mac_extents`` and ``mac_type`` it
    produces the same lists as SETS (``tests/unit/test_dual_tree_walk_mutual.py``
    in yggdrax pins that), so only the fp32 summation order of a force changes.

    What this builder adds on top of the walk:

    * far pairs un-mutualised and INTERLEAVED -- ``[b->a, a->b]`` per canonical
      pair -- so the live pairs are a prefix of the buffer. Every M2L consumer
      masks ``idx < far_pair_count`` (``kernels/_m2l.py``, the CSR Pallas lane,
      ``_large_n_grad``); concatenating the two directions would put the second
      half beyond that prefix and silently drop it. Capacity is
      ``compact_far_pair_capacity`` (``2 x far_cap``), eager and traced alike, as
      ``_compact_prefix_with_fixed_capacity`` does for the dual walk.
    * the leaf neighbour CSR from the directed near pairs with one stable argsort
      by target leaf and ``searchsorted`` offsets -- no per-node row buffer, no
      duplicate-index scatter. Width is ``near_edge_capacity`` (``2 x near_cap``),
      which is exactly the fixed edge cap ``_trim_radix_fast_lane_neighbor_list``
      pads to, so eager and traced carries match without a pad.
    * overflow: eager, the queue is doubled on ``queue_overflow``; a far or near
      overflow doubles the capacity when the caller did not name it
      (``far_named`` / ``near_edge_named`` False) and raises naming the cap when
      it did (a named cap is never widened -- #333's rule). Traced,
      the three flags saturate ``far_pair_count`` to the capacity, which trips
      the strict runner's existing ``far_pair_count < capacity`` arm of the
      saturation guard (``fmm_strict_run.py``), so a truncated refresh is fatal
      rather than silent. The capacity report is ALWAYS emitted (the treecode
      graft's early return left that guard dark) and carries ``peak_wavefront``
      so the traced queue can be sized from data.
    * width: an UNNAMED list is built, eagerly, at ``_tight_list_capacity`` of the
      count the walk measured (``flat_walk_cap_headroom`` x count, granule-rounded,
      never below ``far_floor`` / ``near_floor``), cut from the walk's live prefix,
      and that width is what the report validates -- so the traced refresh, the
      M2L and near-field kernels and their sorts run at the pairs' size, not at a
      ladder rung or a cap fitted at another N (8M on one card: caps 6-9x the
      pairs). The Pallas walk reports what an overflowed list needed, so the
      eager ladder sizes such a cap in one retry.

    Parameters
    ----------
    tree : Tree
        Built static-radix tree (leaves are the last ``num_leaves`` nodes).
    geometry : TreeGeometry
        Node centres and extents the MAC is evaluated against.
    theta : float
        Opening angle.
    mac_type : MACType
        ``bh`` or ``dehnen``; ``engblom`` is supported by the walk but not by the
        strict lane's callers and is refused at the seam.
    dehnen_radius_scale : float
        Radius inflation for the Dehnen MAC.
    compact_far_pair_capacity : int
        Directed far-pair capacity (even).
    near_edge_capacity : int
        Directed near-pair capacity, i.e. the neighbour-edge cap.
    max_pair_queue : Optional[int]
        Wavefront capacity; ``None`` starts the eager ladder at the floor.
    capacity_report : Optional[Callable[[dict], None]]
        Receives the capacities used (and, eager only, what was observed).
    far_named : bool
        Whether ``compact_far_pair_capacity`` was named by the caller
        (``JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP``); a named cap raises
        on overflow, an unnamed one is doubled eagerly up to
        ``_FLAT_WALK_FAR_PAIR_LIMIT``.
    near_edge_named : bool
        Same for ``near_edge_capacity``
        (``JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP``; ceiling
        ``_FLAT_WALK_NEAR_EDGE_LIMIT``).
    extra_overflow : Optional[Array]
        An upstream capacity flag treated like the walk's own -- today the
        cell-leaf partition's ``leaf_capacity``.
    far_floor : int
        Directed far width an earlier eager pass validated; an unnamed far list
        is never built narrower (widths never shrink, so the compiled step keeps
        its shapes).
    near_floor : int
        Same for the near list.

    Returns
    -------
    _DualTreeArtifacts
        Compact far pairs and the leaf neighbour list; no interactions payload.

    Raises
    ------
    ValueError
        If a capacity is not even / positive, or (int32 indices) reaches 2^31.
    RuntimeError
        Eager far or near overflow of a named cap (or of an unnamed one past its
        ceiling), or a queue that will not fit within the retry ceiling.
    """
    from yggdrax._interactions_impl import _build_mac_extents
    from yggdrax.interactions import dual_tree_walk_mutual

    require_index_capacity(
        far_pairs=int(compact_far_pair_capacity),
        near_edges=int(near_edge_capacity),
        pair_queue=int(max_pair_queue or 0),
    )

    walk_backend = strict_walk_backend()
    deterministic_rows = strict_walk_deterministic_rows()
    if walk_backend == "pallas" and str(mac_type) not in ("bh", "dehnen"):
        raise RuntimeError(
            "JACCPOT_STATIC_STRICT_FUSED_WALK=pallas implements the dual-tree "
            f"sphere MAC (bh/dehnen) only; mac_type={mac_type!r}."
        )

    if int(compact_far_pair_capacity) <= 0 or int(compact_far_pair_capacity) % 2:
        raise ValueError(
            "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP must be a positive "
            f"even number on the flat-walk lane, got {compact_far_pair_capacity}"
        )
    if int(near_edge_capacity) <= 0 or int(near_edge_capacity) % 2:
        raise ValueError(
            "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP must be a positive even "
            f"number on the flat-walk lane, got {near_edge_capacity}"
        )
    far_cap = int(compact_far_pair_capacity) // 2
    near_cap = int(near_edge_capacity) // 2

    topo = tree.topology
    num_internal = int(topo.left_child.shape[0])
    total_nodes = int(topo.parent.shape[0])
    num_leaves = total_nodes - num_internal
    idx = topo.parent.dtype
    left_full = jnp.concatenate(
        [jnp.asarray(topo.left_child, idx), jnp.full((num_leaves,), -1, idx)]
    )
    right_full = jnp.concatenate(
        [jnp.asarray(topo.right_child, idx), jnp.full((num_leaves,), -1, idx)]
    )
    root_idx = jnp.argmin(topo.parent).astype(idx)
    centers = jnp.asarray(geometry.center)
    with jax.named_scope("fmm_mac_extents"):
        mac_extents, _leaf_extents = _build_mac_extents(
            topo.parent,
            geometry,
            num_internal,
            str(mac_type),
            float(dehnen_radius_scale),
        )
    mac_extents = jnp.asarray(mac_extents, dtype=centers.dtype)
    # A capacity-padded leaf partition (cell leaves) carries EMPTY nodes: one
    # centre, radius zero. Left in the walk they fail the MAC against each
    # other and flood the near list (30M edges at N=2e5), so they are dead
    # nodes for the walk. Bucket trees have no empty node and the mask is all
    # true -- the walk's result is then identical to the unmasked walk.
    node_ranges_all = jnp.asarray(topo.node_ranges)
    node_active = node_ranges_all[:, 1] >= node_ranges_all[:, 0]

    queue = (
        _STRICT_STREAMED_QUEUE_FLOOR if max_pair_queue is None else int(max_pair_queue)
    )
    headroom = flat_walk_cap_headroom()
    grew: list[str] = []
    walk = None
    traced = False
    # Cap growth (unnamed caps only) is bounded by the ceilings; queue growth by
    # the retry budget. Only queue doublings count against that budget, so a tiny
    # unnamed cap cannot exhaust it.
    queue_attempts = 0
    while True:
        # drop the previous attempt before the next one allocates its buffers
        walk = None
        if walk_backend == "pallas":
            # plan sub-10ms Phase 2: one Pallas launch per wavefront round with
            # atomic slot counters; same pair SETS as the flat walk (pinned by
            # tests/unit/runtime/test_mutual_walk_pallas.py), emission order
            # nondeterministic -> the rows below are sorted by (target, source).
            from jaccpot.pallas.mutual_walk_pallas import mutual_walk_pallas

            walk = mutual_walk_pallas(
                left_full,
                right_full,
                centers,
                mac_extents,
                float(theta),
                root_idx,
                max_pair_queue=int(queue),
                far_cap=far_cap,
                near_cap=near_cap,
                node_active=node_active,
                interpret=env_flag("JACCPOT_WALK_PALLAS_INTERPRET", False),
            )
        else:
            walk = dual_tree_walk_mutual(
                left_full,
                right_full,
                centers,
                mac_extents,
                float(theta),
                root_idx,
                max_pair_queue=int(queue),
                far_cap=far_cap,
                near_cap=near_cap,
                mac_type=str(mac_type),
                node_active=node_active,
            )
        traced = isinstance(walk.queue_overflow, Tracer)
        if traced:
            break
        far_ovf = bool(walk.far_overflow)
        near_ovf = bool(walk.near_overflow)
        queue_ovf = bool(walk.queue_overflow)
        # The Pallas walk keeps walking past a full far or near list and reports
        # what the list needed, so an unnamed cap is sized in ONE retry; the flat
        # walk stops at the first overflow (a lower bound), and so does any walk
        # whose queue overflowed -- those caps double.
        exact = (not queue_ovf) and hasattr(walk, "far_needed")
        # A walk that stopped on its queue counted only part of the pairs. The
        # Pallas walk sizes its lists exactly once the queue fits, so it grows only
        # the queue now; the flat walk's counts are never exact, so its lists double.
        defer_lists = queue_ovf and hasattr(walk, "far_needed")
        grown_far = (
            _tight_list_capacity(2 * int(walk.far_needed), headroom=headroom) // 2
            if exact
            else 2 * far_cap
        )
        grown_near = (
            _tight_list_capacity(2 * int(walk.near_needed), headroom=headroom) // 2
            if exact
            else 2 * near_cap
        )
        if far_ovf and (
            far_named or (not defer_lists and 2 * grown_far > _FLAT_WALK_FAR_PAIR_LIMIT)
        ):
            raise RuntimeError(
                "flat-walk far pairs overflowed: capacity "
                f"{2 * far_cap} directed pairs"
                + (f", {2 * int(walk.far_needed)} needed" if exact else "")
                + (
                    " (the caller named it, so it is not widened here). Raise "
                    if far_named
                    else f" (the eager ceiling is {_FLAT_WALK_FAR_PAIR_LIMIT}). Set "
                )
                + "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP."
            )
        if near_ovf and (
            near_edge_named
            or (not defer_lists and 2 * grown_near > _FLAT_WALK_NEAR_EDGE_LIMIT)
        ):
            raise RuntimeError(
                "flat-walk near pairs overflowed: capacity "
                f"{2 * near_cap} directed pairs"
                + (f", {2 * int(walk.near_needed)} needed" if exact else "")
                + (
                    " (the caller named it, so it is not widened here). Raise "
                    if near_edge_named
                    else f" (the eager ceiling is {_FLAT_WALK_NEAR_EDGE_LIMIT}). Set "
                )
                + "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP."
            )
        if far_ovf and not defer_lists:
            grew.append(f"compact_far_pair_capacity {2 * far_cap}->{2 * grown_far}")
            far_cap = grown_far
        if near_ovf and not defer_lists:
            grew.append(f"near_edge_capacity {2 * near_cap}->{2 * grown_near}")
            near_cap = grown_near
        if queue_ovf:
            grown = int(queue) * 2
            queue_attempts += 1
            if (
                queue_attempts >= _STRICT_STREAMED_RETRY_ATTEMPTS
                or grown > _FLAT_WALK_QUEUE_LIMIT
            ):
                raise RuntimeError(
                    "max_pair_queue overflowed on the flat wavefront walk and "
                    f"re-planning did not fit it: grew to {queue} (ceiling "
                    f"{_FLAT_WALK_QUEUE_LIMIT}) over {queue_attempts} attempts; "
                    "the walk needed a peak wavefront of "
                    f"{int(walk.peak_wavefront)}. Pass jaccpot.TraversalOverrides("
                    "max_pair_queue=...) explicitly."
                )
            grew.append(f"max_pair_queue {queue}->{grown}")
            queue = grown
        if not (far_ovf or near_ovf or queue_ovf):
            if grew:
                _strict_streamed_retry_diag(grew)
            break
    assert walk is not None

    # The widths the lists are BUILT at. A named cap is exact. An unnamed one is
    # sized from the count the eager walk measured -- headroom x count, never below
    # the width an earlier pass validated (`far_floor` / `near_floor`) -- and that
    # width is reported, so the traced refresh walks and builds at the same width
    # (the carry's shapes must match) and the step's per-pair work and memory scale
    # with the pairs, not with the ladder rung the walk happened to fit at.
    # Traced, the widths are the capacities as passed.
    far_width, near_width = far_cap, near_cap
    if not traced:
        if not far_named:
            far_width = (
                _tight_list_capacity(
                    2 * int(walk.far_count), headroom=headroom, floor=int(far_floor)
                )
                // 2
            )
        if not near_edge_named:
            near_width = (
                _tight_list_capacity(
                    2 * int(walk.near_count), headroom=headroom, floor=int(near_floor)
                )
                // 2
            )
    compact_far_pair_capacity = 2 * far_width
    near_edge_capacity = 2 * near_width

    any_overflow = walk.far_overflow | walk.near_overflow | walk.queue_overflow
    if extra_overflow is not None:
        # an upstream capacity the caller wants treated like the walk's own --
        # today the cell-leaf partition's leaf_capacity (plan sub-10ms 1.2)
        if not traced and not isinstance(extra_overflow, Tracer):
            if bool(extra_overflow):
                raise RuntimeError(
                    "static_radix cell leaves overflowed TreeConfig.leaf_capacity; "
                    "raise it."
                )
        else:
            any_overflow = any_overflow | jnp.asarray(extra_overflow, dtype=bool)
    # Eager overflow raised above, so the saturation inside only bites in the
    # compiled scan.
    (
        far_sources,
        far_targets,
        far_tags,
        far_pair_count,
        neighbors,
        offsets,
        counts,
    ) = _flat_walk_lists(
        walk.far_a,
        walk.far_b,
        walk.far_count,
        walk.near_a,
        walk.near_b,
        walk.near_count,
        any_overflow,
        far_width=int(far_width),
        near_width=int(near_width),
        num_internal=num_internal,
        total_nodes=total_nodes,
        deterministic=bool(walk_backend == "pallas" and deterministic_rows),
        idx=np.dtype(idx),
    )
    far_cls = (
        TargetSortedFarPairs
        if walk_backend == "pallas" and deterministic_rows
        else CompactTaggedFarPairs
    )
    compact_far_pairs = far_cls(
        sources=far_sources,
        targets=far_targets,
        tags=far_tags,
        far_pair_count=far_pair_count,
    )
    leaf_nodes = jnp.arange(num_internal, total_nodes, dtype=idx)
    neighbor_list = NodeNeighborList(
        offsets=offsets,
        neighbors=neighbors,
        leaf_indices=leaf_nodes,
        counts=counts,
        particle_order_leaf_indices=leaf_nodes,
        particle_order_to_native_leaf=jnp.arange(num_leaves, dtype=idx),
        neighbor_leaf_positions=jnp.zeros((num_leaves, 0), dtype=idx),
        target_block_leaf_ids=jnp.zeros((0,), dtype=idx),
        target_block_source_leaf_ids=jnp.zeros((0, 0), dtype=idx),
        target_block_valid_mask=jnp.zeros((0, 0), dtype=bool),
        target_block_offsets=jnp.zeros((num_leaves + 1,), dtype=idx),
        target_block_size=0,
    )

    if capacity_report is not None:
        report = dict(
            traced=bool(traced),
            flat_walk=True,
            max_pair_queue_requested=int(queue),
            queue_capacity=int(queue),
            compact_far_pair_capacity=int(compact_far_pair_capacity),
            near_edge_capacity=int(near_edge_capacity),
            far_named=bool(far_named),
            near_edge_named=bool(near_edge_named),
            # no per-leaf row cap on this lane; None switches the guard's row arm off
            max_neighbors_per_leaf_used=None,
            grew=list(grew),
            headroom=float(headroom),
        )
        if not traced:
            report["far_pair_count"] = 2 * int(walk.far_count)
            report["total_neighbors"] = 2 * int(walk.near_count)
            report["max_neighbors_observed"] = int(jnp.max(counts)) if num_leaves else 0
            report["peak_wavefront"] = int(walk.peak_wavefront)
            report["rounds"] = int(walk.rounds)
        else:
            # What this traced walk needed, as TRACERS: the receiver keeps them on
            # the engine for the scan to fold into its carry (they must never
            # reach host-side state). `strict_run_v2` sizes a segment retry from
            # them -- the Pallas walk's counts are exact past a list overflow; the
            # flat walk's are lower bounds, marked by the last entry.
            report["walk_needs"] = jnp.stack(
                [
                    2 * jnp.asarray(getattr(walk, "far_needed", walk.far_count)),
                    2 * jnp.asarray(getattr(walk, "near_needed", walk.near_count)),
                    jnp.asarray(walk.peak_wavefront),
                    jnp.asarray(walk.far_overflow),
                    jnp.asarray(walk.near_overflow),
                    jnp.asarray(walk.queue_overflow),
                    jnp.asarray(not hasattr(walk, "far_needed")),
                ]
            ).astype(jnp.int32)
        capacity_report(report)

    return _DualTreeArtifacts(
        interactions=None,
        neighbor_list=neighbor_list,
        traversal_result=None,
        compact_far_pairs=compact_far_pairs,
        dense_buffers=None,
        grouped_buffers=None,
        grouped_segment_starts=None,
        grouped_segment_lengths=None,
        grouped_segment_class_ids=None,
        grouped_segment_sort_permutation=None,
        grouped_segment_group_ids=None,
        grouped_segment_unique_targets=None,
        grouped_chunk_size=None,
    )


def _dual_tree_build_grouped_buffers(
    *,
    tree: Tree,
    geometry: TreeGeometry,
    interactions: Optional[NodeInteractionList],
) -> GroupedInteractionBuffers:
    """Materialize grouped interaction buffers from node interaction pairs.

    Parameters
    ----------
    tree : Tree
        Built tree.
    geometry : TreeGeometry
        Node centres and radii; the displacement classes are formed from these.
    interactions : Optional[NodeInteractionList]
        Far-field pairs to group.

    Returns
    -------
    GroupedInteractionBuffers
        Class-sorted sources and targets, class offsets, keys and representative
        displacements.

    Raises
    ------
    RuntimeError
        If the interaction list is absent or cannot be grouped -- grouping is
        requested explicitly, so failing is better than silently returning the
        ungrouped form.
    """

    from yggdrax import interactions as _yggdrax_interactions

    if interactions is None:
        raise RuntimeError(
            "grouped interaction preparation requires node interaction lists"
        )
    return _yggdrax_interactions.build_grouped_interactions_from_pairs(
        tree,
        geometry,
        interactions.sources,
        interactions.targets,
        level_offsets=getattr(interactions, "level_offsets", None),
    )


def _dual_tree_build_grouped_class_segments(
    *,
    grouped_buffers: GroupedInteractionBuffers,
    grouped_chunk_size: int,
) -> tuple[Array, Array, Array, int]:
    """Materialize class-major grouped schedule arrays.

    Parameters
    ----------
    grouped_buffers : GroupedInteractionBuffers
        See the module docstring.
    grouped_chunk_size : int
        Pairs per grouped chunk, or ``None`` for the default.

    Returns
    -------
    tuple[Array, Array, Array, int]
        ``(starts, lengths, class_ids, chunk_size)``: the class-major segment
        schedule and the chunk size it was built for. Three arrays and an int, not
        the six arrays ``_DualTreeArtifacts`` stores -- the caller derives the
        remaining two from these.
    """

    from . import fmm as _runtime_fmm

    grouped_segment_starts, grouped_segment_lengths, grouped_segment_class_ids = (
        _runtime_fmm._build_grouped_class_segments(
            grouped_buffers,
            chunk_size=int(grouped_chunk_size),
        )
    )
    return (
        grouped_segment_starts,
        grouped_segment_lengths,
        grouped_segment_class_ids,
        int(grouped_chunk_size),
    )


def _dual_tree_build_dense_buffers(
    *,
    tree: Tree,
    geometry: TreeGeometry,
    interactions: Optional[NodeInteractionList],
    use_dense_interactions: bool,
) -> Optional[DenseInteractionBuffers]:
    """Materialize dense interaction buffers when requested.

    Parameters
    ----------
    tree : Tree
        Built tree.
    geometry : TreeGeometry
        Node centres and radii.
    interactions : Optional[NodeInteractionList]
        Far-field pairs to densify.
    use_dense_interactions : bool
        Whether to build them at all.

    Returns
    -------
    Optional[DenseInteractionBuffers]
        The dense buffers, or ``None`` when not requested or when there is no
        interaction list to densify. Dense storage is quadratic in node count,
        which is why it is opt-in rather than the default.
    """

    if not use_dense_interactions:
        return None
    return densify_interactions(tree, geometry, interactions)


_CAPACITY_RETRY_MAX_ATTEMPTS = 2
_CAPACITY_RETRY_QUEUE_BASE = 262_144
_CAPACITY_RETRY_PROCESS_BLOCK_BASE = 256
_CAPACITY_RETRY_INTERACTIONS_BASE = 8192
_CAPACITY_RETRY_NEIGHBORS_BASE = 4096
_CAPACITY_RETRY_QUEUE_MAX = 4_194_304
_CAPACITY_RETRY_PROCESS_BLOCK_MAX = 4096
_CAPACITY_RETRY_INTERACTIONS_MAX = 16_384
_CAPACITY_RETRY_NEIGHBORS_MAX = 65_536


def _looks_like_capacity_error(exc: BaseException) -> bool:
    """Return whether an exception likely indicates traversal-capacity overflow.

    Parameters
    ----------
    exc : BaseException
        The exception under inspection.

    Returns
    -------
    bool
        Whether the exception looks like a traversal capacity overflow rather than an unrelated failure.
    """
    msg = str(exc).lower()
    needles = (
        "capacity exceeded",
        "pair queue",
        "neighbor list",
        "interactions per node",
        "max_pair_queue",
        "max_neighbors_per_leaf",
        "max_interactions_per_node",
    )
    return any(token in msg for token in needles)


def _next_retry_traversal_settings(
    *,
    traversal_config: Optional[DualTreeTraversalConfig],
    max_pair_queue: Optional[int],
    pair_process_block: Optional[int],
) -> tuple[DualTreeTraversalConfig, Optional[int], Optional[int]]:
    """Scale traversal capacities for one retry attempt.

    Parameters
    ----------
    traversal_config : Optional[DualTreeTraversalConfig]
        Traversal capacities, or ``None`` to take the template.
    max_pair_queue : Optional[int]
        Traversal pair-queue capacity, or ``None`` for the default.
    pair_process_block : Optional[int]
        Pairs processed per traversal block, or ``None`` for the default.

    Returns
    -------
    tuple[DualTreeTraversalConfig, Optional[int], Optional[int]]
        The scaled traversal config and the queue/block sizes for the next attempt.
    """
    if traversal_config is None:
        queue = (
            _CAPACITY_RETRY_QUEUE_BASE
            if max_pair_queue is None
            else max(int(max_pair_queue), _CAPACITY_RETRY_QUEUE_BASE)
        )
        block = (
            _CAPACITY_RETRY_PROCESS_BLOCK_BASE
            if pair_process_block is None
            else max(int(pair_process_block), _CAPACITY_RETRY_PROCESS_BLOCK_BASE)
        )
        interactions = _CAPACITY_RETRY_INTERACTIONS_BASE
        neighbors = _CAPACITY_RETRY_NEIGHBORS_BASE
    else:
        # Small explicit traversal configs are useful as the initial
        # minimum-memory seed, but after a real capacity overflow we should jump
        # to the established retry floor rather than spending retries on
        # intermediate capacities that are known to be below the normal host
        # retry baseline.
        queue = max(
            int(traversal_config.max_pair_queue) * 2,
            _CAPACITY_RETRY_QUEUE_BASE,
            1,
        )
        block = max(
            int(traversal_config.process_block) * 2,
            _CAPACITY_RETRY_PROCESS_BLOCK_BASE,
            1,
        )
        interactions = max(
            int(traversal_config.max_interactions_per_node) * 2,
            _CAPACITY_RETRY_INTERACTIONS_BASE,
            1,
        )
        neighbors = max(
            int(traversal_config.max_neighbors_per_leaf) * 2,
            _CAPACITY_RETRY_NEIGHBORS_BASE,
            1,
        )

    queue = min(queue, _CAPACITY_RETRY_QUEUE_MAX)
    block = min(block, _CAPACITY_RETRY_PROCESS_BLOCK_MAX)
    interactions = min(interactions, _CAPACITY_RETRY_INTERACTIONS_MAX)
    neighbors = min(neighbors, _CAPACITY_RETRY_NEIGHBORS_MAX)
    next_config = DualTreeTraversalConfig(
        max_pair_queue=int(queue),
        process_block=int(block),
        max_interactions_per_node=int(interactions),
        max_neighbors_per_leaf=int(neighbors),
    )
    return next_config, int(queue), int(block)


def _format_capacity_error_hint(
    exc: RuntimeError,
    *,
    traversal_config: Optional[DualTreeTraversalConfig],
    max_pair_queue: Optional[int],
    pair_process_block: Optional[int],
) -> str:
    """Augment traversal capacity failures with actionable tuning hints.

    Parameters
    ----------
    exc : RuntimeError
        The exception under inspection.
    traversal_config : Optional[DualTreeTraversalConfig]
        Traversal capacities, or ``None`` to take the template.
    max_pair_queue : Optional[int]
        Traversal pair-queue capacity, or ``None`` for the default.
    pair_process_block : Optional[int]
        Pairs processed per traversal block, or ``None`` for the default.

    Returns
    -------
    str
        The original message with the capacity knobs and a workable configuration appended.
    """
    msg = str(exc).strip()
    if traversal_config is None:
        queue = None if max_pair_queue is None else int(max_pair_queue)
        block = None if pair_process_block is None else int(pair_process_block)
        interactions = None
        neighbors = None
    else:
        queue = int(traversal_config.max_pair_queue)
        block = int(traversal_config.process_block)
        interactions = int(traversal_config.max_interactions_per_node)
        neighbors = int(traversal_config.max_neighbors_per_leaf)

    details = [
        "Traversal capacity overflow with fail_fast enabled.",
        msg,
        "Increase one or more traversal capacities and rerun.",
    ]
    details.append(
        "Current capacities: "
        f"max_pair_queue={queue}, "
        f"process_block={block}, "
        f"max_interactions_per_node={interactions}, "
        f"max_neighbors_per_leaf={neighbors}."
    )
    details.append(
        "Suggested knobs: "
        "`RuntimePolicyConfig(traversal_config=DualTreeTraversalConfig(...))`, "
        "`RuntimePolicyConfig(max_pair_queue=..., pair_process_block=...)`, "
        "or a larger preset/runtime traversal seed."
    )
    details.append(
        "For exploratory runs, disable `fail_fast` to re-enable host-side retry growth."
    )
    return " ".join(details)


#: Identity meaning "this request must not be served from, or written to, the
#: interaction cache at all". See :func:`pair_policy_cache_identity`.
POLICY_IDENTITY_UNCACHEABLE = "pair_policy_identity_uncacheable_v1"


def _hash_array_or_none(hasher: Any, label: bytes, value: Optional[Array]) -> bool:
    """Fold ``value`` into ``hasher``; return False if it cannot be read on host.

    A tracer cannot be hashed, and a cache key built from one would be wrong
    rather than merely absent -- so the caller must treat ``False`` as "do not
    cache this", not as "hashed nothing".

    Parameters
    ----------
    hasher : Any
        Incremental hash object, mutated in place.
    label : bytes
        Field tag folded in before the value, so two fields cannot collide by
        carrying the same bytes.
    value : Optional[Array]
        Array to fold. ``None`` folds the label alone and still succeeds.

    Returns
    -------
    bool
        ``True`` when the value was folded in, ``False`` when it could not be
        brought to the host.
    """

    hasher.update(label)
    if value is None:
        hasher.update(b"none")
        return True
    try:
        arr = np.asarray(jax.device_get(value))
    except Exception:
        return False
    hasher.update(str(arr.dtype).encode("utf8"))
    hasher.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    hasher.update(np.ascontiguousarray(arr).tobytes())
    return True


def pair_policy_cache_identity(
    *,
    pair_policy: Any,
    policy_state: Any,
    eps: Optional[float],
    force_scale_mode: Optional[str],
    geometry_mode: Optional[str],
    theta_max: Optional[float],
    error_model_code: Optional[int],
    force_scale_nodes: Optional[Array],
    mac_geometry_radius: Optional[Array],
) -> str:
    """Return an identity for the *acceptance criterion* behind a build request.

    :func:`_interaction_cache_key` describes geometry: topology, ``theta``, the
    base ``mac_type``, ``dehnen_radius_scale``, basis, centre mode, caps and
    refinement. Under the Dehnen mass-dependent MAC none of that describes what
    is actually accepted. ``mac_type="dehnen_error"`` reports the geometric base
    MAC ``"dehnen"`` (see ``_base_mac_type``) and paper mode pins ``theta`` at
    1.0 because it does not gate acceptance -- so two solvers at different
    ``adaptive_eps`` hash identically while answering different criteria.

    Serving one criterion's interaction list to another request makes the solver
    *cheaper* and silently wrong, which no cost measurement can detect (trap 6
    in ``docs/dehnen_mass_mac_status_and_plan.md``). Measured before this
    existed, at N=2048 / leaf 8 / eps=1e-3 with ``mac_type="dehnen_theta"``:
    injected per-node force scales of 1e-3, 1.0 and 1e+3 -- the whole right-hand
    side of eq (16a), across six orders of magnitude -- all returned the same
    17520 far pairs, because the second prepare hit the cache.

    Three outcomes:

    * ``""`` -- nothing criterion-shaped is in play, so the geometric key is
      already complete and keys stay exactly what they were.
    * :data:`POLICY_IDENTITY_UNCACHEABLE` -- a solver-owned ``pair_policy`` or
      ``policy_state`` is installed, or an input is a tracer. **Refuse to
      cache.** A pair policy is evaluated against ``policy_state``, which is
      built from the multipole power (hence the *masses*) and the per-particle
      positions; the geometric key hashes neither, so no entry can honestly be
      shown to match. Caching is a perf optimisation, so it yields -- and the
      large-N production profile does not depend on it, because the interaction
      cache is already off for ``static_radix`` trees.
    * a hex digest -- the criterion is carried by data this function *can* see,
      which is the ``dehnen_theta`` case: the criterion is folded into
      ``geometry.radius``, so hashing those radii pins acceptance regardless of
      what produced them.

    Parameters
    ----------
    pair_policy : Any
        The policy callable, or ``None``. Its presence alone changes the
        criterion, so it participates in the identity.
    policy_state : Any
        Policy state, or ``None``.
    eps : Optional[float]
        Relative force-accuracy target of eq (16a).
    force_scale_mode : Optional[str]
        How the per-node force scale was obtained.
    geometry_mode : Optional[str]
        Dehnen geometry mode.
    theta_max : Optional[float]
        Upper clamp on the adaptive opening angle.
    error_model_code : Optional[int]
        Which error estimator is active.
    force_scale_nodes : Optional[Array]
        Injected per-node force scale, folded in by value.
    mac_geometry_radius : Optional[Array]
        MAC radii, folded in likewise -- this is what pins the
        ``dehnen_theta`` case, where the criterion lives in the radii.

    Returns
    -------
    str
        One of the three outcomes above: empty, the uncacheable sentinel, or a
        hex digest.
    """

    if pair_policy is not None or policy_state is not None:
        return POLICY_IDENTITY_UNCACHEABLE
    if (
        eps is None
        and force_scale_mode is None
        and geometry_mode is None
        and theta_max is None
        and error_model_code is None
        and force_scale_nodes is None
        and mac_geometry_radius is None
    ):
        return ""

    hasher = hashlib.sha256()
    hasher.update(b"pair_policy_identity_v1")
    hasher.update(b"eps")
    hasher.update(
        b"none" if eps is None else np.asarray(float(eps), dtype=np.float64).tobytes()
    )
    hasher.update(b"force_scale_mode")
    hasher.update(str(force_scale_mode).encode("utf8"))
    hasher.update(b"geometry_mode")
    hasher.update(str(geometry_mode).encode("utf8"))
    hasher.update(b"theta_max")
    hasher.update(
        b"none"
        if theta_max is None
        else np.asarray(float(theta_max), dtype=np.float64).tobytes()
    )
    hasher.update(b"error_model_code")
    hasher.update(
        b"none"
        if error_model_code is None
        else np.asarray(int(error_model_code), dtype=np.int64).tobytes()
    )
    if not _hash_array_or_none(hasher, b"force_scale_nodes", force_scale_nodes):
        return POLICY_IDENTITY_UNCACHEABLE
    if not _hash_array_or_none(hasher, b"mac_geometry_radius", mac_geometry_radius):
        return POLICY_IDENTITY_UNCACHEABLE
    return hasher.hexdigest()


def _interaction_cache_key(
    tree: Tree,
    *,
    topology_key: Optional[str],
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
    pair_policy_identity: str,
) -> Optional[str]:
    """Return a hash for the interaction list of a tree/theta configuration.

    If any tree arrays are tracers (for example under grad/jit), return ``None``
    to disable caching and avoid host round-trips on traced values.

    ``pair_policy_identity`` carries everything about the *acceptance criterion*
    that the geometric fields below cannot see; it has no default on purpose,
    because a silent default is how the criterion came to be missing from this
    key in the first place. See :func:`pair_policy_cache_identity`.

    Parameters
    ----------
    tree : Tree
        The tree being traversed.
    topology_key : Optional[str]
        See the module docstring.
    tree_mode : str
        See the module docstring.
    leaf_parameter : int
        See the module docstring.
    theta : float
        Opening angle for the acceptance test.
    mac_type : MACType
        Geometric MAC family, already translated for the traversal.
    dehnen_radius_scale : float
        Multiplier on the Dehnen acceptance radius.
    expansion_basis : str
        See the module docstring.
    center_mode : str
        See the module docstring.
    max_pair_queue : Optional[int]
        Traversal pair-queue capacity, or ``None`` for the default.
    pair_process_block : Optional[int]
        Pairs processed per traversal block, or ``None`` for the default.
    traversal_config : Optional[DualTreeTraversalConfig]
        Traversal capacities, or ``None`` to take the template.
    refine_local : Optional[bool]
        See the module docstring.
    max_refine_levels : Optional[int]
        See the module docstring.
    aspect_threshold : Optional[float]
        See the module docstring.
    pair_policy_identity : str
        Identity of the *acceptance criterion* in force for this build. No
        default on purpose: a silent default is how the criterion came to be
        missing from this key. See :func:`pair_policy_cache_identity`.

    Returns
    -------
    Optional[str]
        A stable key for this build request, or ``None`` when caching does not apply.
    """

    if pair_policy_identity == POLICY_IDENTITY_UNCACHEABLE:
        return None

    hasher = hashlib.sha256()

    if topology_key is not None:
        hasher.update(b"topology_key_v1")
        hasher.update(str(topology_key).encode("utf8"))
    else:
        try:
            morton_codes, node_ranges, bounds_min, bounds_max = jax.device_get(
                (tree.morton_codes, tree.node_ranges, tree.bounds_min, tree.bounds_max)
            )
            morton_codes = np.asarray(morton_codes, dtype=np.uint64)
            node_ranges = np.asarray(node_ranges, dtype=np.int64)
            bounds_min = np.asarray(bounds_min, dtype=np.float64)
            bounds_max = np.asarray(bounds_max, dtype=np.float64)
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
    hasher.update(b"pair_policy_identity")
    hasher.update(str(pair_policy_identity).encode("utf8"))
    return hasher.hexdigest()


def _build_dual_tree_artifacts(
    tree: Tree,
    geometry: TreeGeometry,
    *,
    geometry_factory: Optional[Callable[[], Any]] = None,
    theta: float,
    mac_type: MACType,
    dehnen_radius_scale: float,
    cache_key: Optional[str],
    cache_entry: Optional[_InteractionCacheEntry],
    max_pair_queue: Optional[int],
    pair_process_block: Optional[int],
    traversal_config: Optional[DualTreeTraversalConfig],
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]],
    fail_fast: bool,
    use_dense_interactions: bool,
    grouped_interactions: bool,
    grouped_chunk_size: Optional[int],
    need_traversal_result: bool,
    need_compact_far_pairs: bool,
    need_node_interactions: bool,
    precompute_grouped_class_segments: bool,
    grouped_schedule_budget_bytes: Optional[int],
    allow_split_build: bool = False,
    pair_policy: Optional[PairPolicy] = None,
    policy_state: Optional[AdaptivePolicyState] = None,
    jit_traversal: bool = True,
    timing_callback: Optional[Callable[[str, float], None]] = None,
    planner_hint: Optional[_RefreshDualPlannerHint] = None,
    strict_capacity_report: Optional[Callable[[dict], None]] = None,
    strict_max_neighbors_per_leaf_override: Optional[int] = None,
    strict_flat_walk_capacity_floor: Optional[dict] = None,
    strict_extra_overflow: Optional[Array] = None,
) -> tuple[_DualTreeArtifacts, Optional[_InteractionCacheEntry]]:
    """Construct or reuse dual-tree traversal products for a tree.

    The top of this module: resolve the cache, choose a build strategy (split,
    strict-streamed, treecode, or the single traversal), construct whichever
    buffers the caller asked for, and hand back an entry to store.

    Parameters
    ----------
    tree : Tree
        Built tree.
    geometry : TreeGeometry
        Node centres and radii the MAC is evaluated against.
    geometry_factory : Optional[Callable[[], Any]]
        Deferred geometry builder, for lanes that avoid materialising geometry
        until the cache has been consulted.
    theta : float
        Opening angle.
    mac_type : MACType
        Geometric criterion the traversal evaluates. jaccpot's Dehnen policies
        must already be mapped to a yggdrax literal -- see
        ``PolicyMixin._mac_type_for_traversal``.
    dehnen_radius_scale : float
        Radius inflation for the Dehnen MAC.
    cache_key : Optional[str]
        Interaction-cache key; ``None`` disables both lookup and store.
    cache_entry : Optional[_InteractionCacheEntry]
        A previously stored entry to reuse rather than rebuild.
    max_pair_queue : Optional[int]
        Cap on the pending-pair queue; ``None`` lets the traversal size it.
    pair_process_block : Optional[int]
        Pairs drained per step; ``None`` as above.
    traversal_config : Optional[DualTreeTraversalConfig]
        Full traversal capacities, overriding the two above when given.
    retry_logger : Optional[Callable[[DualTreeRetryEvent], None]]
        Sink for capacity-retry events, so a retry stays visible in the caller's
        diagnostics rather than being absorbed here.
    fail_fast : bool
        Raise on a capacity overflow instead of retrying with more room.
    use_dense_interactions : bool
        Materialise the interaction list densely.
    grouped_interactions : bool
        Group interactions by displacement class.
    grouped_chunk_size : Optional[int]
        Pairs per grouped segment; ``None`` lets the builder choose.
    need_traversal_result : bool
        Retain the full walk result.
    need_compact_far_pairs : bool
        Produce compact tagged far pairs.
    need_node_interactions : bool
        Produce a node interaction list.
    precompute_grouped_class_segments : bool
        Build the class-major segment table up front.
    grouped_schedule_budget_bytes : Optional[int]
        Memory ceiling for that schedule; ``None`` is unbounded.
    allow_split_build : bool
        Permit the two-phase split build when
        :func:`_can_split_dual_tree_build` agrees.
    pair_policy : Optional[PairPolicy]
        Solver-owned per-pair acceptance callable. ``None`` leaves the traversal
        running the geometric MAC alone -- that is the difference between the
        Dehnen mass-dependent criterion being active and not.
    policy_state : Optional[AdaptivePolicyState]
        State the pair policy reads. Meaningless without ``pair_policy``.
    jit_traversal : bool
        Run the compiled traversal.
    timing_callback : Optional[Callable[[str, float], None]]
        Per-phase timing sink, named so a caller can attribute prepare cost.
    planner_hint : Optional[_RefreshDualPlannerHint]
        Hint from a previous refresh, letting the planner skip work whose answer
        is already known.
    strict_capacity_report : Optional[Callable[[dict], None]]
        Forwarded to the strict streamed builder as ``capacity_report``.
    strict_max_neighbors_per_leaf_override : Optional[int]
        Forwarded to the strict streamed builder; only ever raises the cap.
    strict_flat_walk_capacity_floor : Optional[dict]
        Forwarded to the strict streamed builder as ``flat_walk_capacity_floor``.
    strict_extra_overflow : Optional[Array]
        An upstream capacity flag treated like the walk's own; forwarded to the
        strict streamed builder as ``extra_overflow``.

    Returns
    -------
    tuple[_DualTreeArtifacts, Optional[_InteractionCacheEntry]]
        The artifacts, and an entry to store when a ``cache_key`` was supplied --
        ``None`` when there is nothing worth caching.

    Raises
    ------
    ValueError
        If the requested combination of buffers and lanes is inconsistent.
    """

    cache_out = cache_entry
    cache_hit = _dual_tree_cache_lookup(
        cache_key=cache_key,
        cache_entry=cache_entry,
        need_traversal_result=need_traversal_result,
        need_compact_far_pairs=need_compact_far_pairs,
        need_node_interactions=need_node_interactions,
        precompute_grouped_class_segments=precompute_grouped_class_segments,
    )
    if cache_hit is not None:
        dual_tree_cache_hit = True
        interactions = cache_hit.interactions
        neighbor_list = cache_hit.neighbor_list
        traversal_result = cache_hit.traversal_result
        compact_far_pairs = cache_hit.compact_far_pairs
        grouped_buffers = cache_hit.grouped_buffers
        grouped_segment_starts = cache_hit.grouped_segment_starts
        grouped_segment_lengths = cache_hit.grouped_segment_lengths
        grouped_segment_class_ids = cache_hit.grouped_segment_class_ids
        grouped_segment_sort_permutation = cache_hit.grouped_segment_sort_permutation
        grouped_segment_group_ids = cache_hit.grouped_segment_group_ids
        grouped_segment_unique_targets = cache_hit.grouped_segment_unique_targets
        grouped_chunk_size_cached = cache_hit.grouped_chunk_size_cached
        cache_out = cache_hit.cache_out
    else:
        dual_tree_cache_hit = False
        if geometry is None:
            if geometry_factory is None:
                raise ValueError(
                    "geometry must be provided when dual-tree cache lookup misses"
                )
            geometry = geometry_factory()
        if planner_hint is not None:
            # Fast refresh path: reuse prior routing decision and avoid
            # re-evaluating split-eligibility branching on host each call.
            use_split_build = bool(planner_hint.use_split_build)
        else:
            use_split_build = _can_split_dual_tree_build(
                split_enabled=bool(allow_split_build),
                grouped_interactions=grouped_interactions,
                need_traversal_result=need_traversal_result,
            )
        if use_split_build:
            strict_streamed_split = bool(
                fail_fast
                and bool(need_compact_far_pairs)
                and not bool(need_node_interactions)
                and not bool(use_dense_interactions)
                and not bool(grouped_interactions)
                and not bool(need_traversal_result)
            )
            split_artifacts = (
                _build_dual_tree_artifacts_split_strict_streamed(
                    tree=tree,
                    geometry=geometry,
                    theta=theta,
                    mac_type=mac_type,
                    dehnen_radius_scale=dehnen_radius_scale,
                    max_pair_queue=max_pair_queue,
                    pair_process_block=pair_process_block,
                    traversal_config=traversal_config,
                    pair_policy=pair_policy,
                    policy_state=policy_state,
                    capacity_report=strict_capacity_report,
                    max_neighbors_per_leaf_override=(
                        strict_max_neighbors_per_leaf_override
                    ),
                    flat_walk_capacity_floor=strict_flat_walk_capacity_floor,
                    extra_overflow=strict_extra_overflow,
                )
                if strict_streamed_split
                else _build_dual_tree_artifacts_split(
                    tree=tree,
                    geometry=geometry,
                    theta=theta,
                    mac_type=mac_type,
                    dehnen_radius_scale=dehnen_radius_scale,
                    max_pair_queue=max_pair_queue,
                    pair_process_block=pair_process_block,
                    traversal_config=traversal_config,
                    retry_logger=retry_logger,
                    need_node_interactions=need_node_interactions,
                    need_compact_far_pairs=need_compact_far_pairs,
                    use_dense_interactions=use_dense_interactions,
                    pair_policy=pair_policy,
                    policy_state=policy_state,
                    timing_callback=timing_callback,
                )
            )
            interactions = split_artifacts.interactions
            neighbor_list = split_artifacts.neighbor_list
            traversal_result = split_artifacts.traversal_result
            compact_far_pairs = split_artifacts.compact_far_pairs
            grouped_buffers = split_artifacts.grouped_buffers
            grouped_segment_starts = None
            grouped_segment_lengths = None
            grouped_segment_class_ids = None
            grouped_segment_sort_permutation = None
            grouped_segment_group_ids = None
            grouped_segment_unique_targets = None
            grouped_chunk_size_cached = None
            cache_out = (
                _InteractionCacheEntry(
                    key=cache_key,
                    interactions=interactions,
                    neighbor_list=neighbor_list,
                    dual_tree_result=traversal_result,
                    compact_far_pairs=compact_far_pairs,
                    grouped_buffers=None,
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
        else:
            stage_t0 = time.perf_counter() if timing_callback is not None else None
            build_out, _, _, _ = _dual_tree_build_raw(
                tree=tree,
                geometry=geometry,
                theta=theta,
                mac_type=mac_type,
                dehnen_radius_scale=dehnen_radius_scale,
                max_pair_queue=max_pair_queue,
                pair_process_block=pair_process_block,
                traversal_config=traversal_config,
                retry_logger=retry_logger,
                fail_fast=fail_fast,
                need_traversal_result=need_traversal_result,
                need_compact_far_pairs=need_compact_far_pairs,
                need_node_interactions=need_node_interactions,
                grouped_interactions=grouped_interactions,
                pair_policy=pair_policy,
                policy_state=policy_state,
                jit_traversal=jit_traversal,
            )
            if timing_callback is not None and stage_t0 is not None:
                timing_callback(
                    "dual_raw_interactions_and_neighbors",
                    float(time.perf_counter() - stage_t0),
                )
            (
                interactions,
                neighbor_list,
                traversal_result,
                compact_far_pairs,
                grouped_buffers,
            ) = _dual_tree_unpack_build_output(
                build_out=build_out,
                grouped_interactions=grouped_interactions,
                need_traversal_result=need_traversal_result,
                need_compact_far_pairs=need_compact_far_pairs,
            )
            cache_out = (
                _InteractionCacheEntry(
                    key=cache_key,
                    interactions=interactions,
                    neighbor_list=neighbor_list,
                    dual_tree_result=traversal_result,
                    compact_far_pairs=compact_far_pairs,
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
        grouped_buffers = _dual_tree_build_grouped_buffers(
            tree=tree,
            geometry=geometry,
            interactions=interactions,
        )
        if cache_out is not None:
            cache_out = _InteractionCacheEntry(
                key=cache_out.key,
                interactions=cache_out.interactions,
                neighbor_list=cache_out.neighbor_list,
                dual_tree_result=cache_out.dual_tree_result,
                compact_far_pairs=cache_out.compact_far_pairs,
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
        precompute_grouped_class_segments
        and (
            grouped_schedule_budget_bytes is None
            or int(grouped_chunk_size or 0) <= 0
            or (
                grouped_buffers is not None
                and (
                    int(grouped_buffers.class_targets.shape[0])
                    * 3
                    * int(grouped_chunk_size or 1)
                    * np.dtype(np.int32).itemsize
                )
                <= int(grouped_schedule_budget_bytes)
            )
        )
        and (
            grouped_interactions
            and grouped_buffers is not None
            and grouped_chunk_size is not None
        )
    ):
        # These segment arrays are a pure execution aid for class-major grouped
        # M2L. They are worth caching only when the schedule itself stays within
        # budget; otherwise the raw grouped buffers are already the smaller
        # resident representation.
        needs_schedule = (
            grouped_segment_starts is None
            or grouped_segment_lengths is None
            or grouped_segment_class_ids is None
            or grouped_chunk_size_cached != int(grouped_chunk_size)
        )
        if needs_schedule:
            (
                grouped_segment_starts,
                grouped_segment_lengths,
                grouped_segment_class_ids,
                grouped_chunk_size_cached,
            ) = _dual_tree_build_grouped_class_segments(
                grouped_buffers=grouped_buffers,
                grouped_chunk_size=int(grouped_chunk_size),
            )
            grouped_segment_sort_permutation = None
            grouped_segment_group_ids = None
            grouped_segment_unique_targets = None
            if cache_out is not None:
                cache_out = _InteractionCacheEntry(
                    key=cache_out.key,
                    interactions=cache_out.interactions,
                    neighbor_list=cache_out.neighbor_list,
                    dual_tree_result=cache_out.dual_tree_result,
                    compact_far_pairs=cache_out.compact_far_pairs,
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

    dense_buffers = _dual_tree_build_dense_buffers(
        tree=tree,
        geometry=geometry,
        interactions=interactions,
        use_dense_interactions=use_dense_interactions,
    )

    artifacts = _DualTreeArtifacts(
        interactions=interactions,
        neighbor_list=neighbor_list,
        traversal_result=traversal_result,
        compact_far_pairs=compact_far_pairs,
        dense_buffers=dense_buffers,
        grouped_buffers=grouped_buffers,
        grouped_segment_starts=grouped_segment_starts,
        grouped_segment_lengths=grouped_segment_lengths,
        grouped_segment_class_ids=grouped_segment_class_ids,
        grouped_segment_sort_permutation=grouped_segment_sort_permutation,
        grouped_segment_group_ids=grouped_segment_group_ids,
        grouped_segment_unique_targets=grouped_segment_unique_targets,
        grouped_chunk_size=grouped_chunk_size_cached,
        cache_hit=bool(dual_tree_cache_hit),
    )
    return artifacts, cache_out
