"""Dual-tree interaction cache helpers for the runtime FMM implementation."""

import hashlib
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
from yggdrax.grouped_interactions import GroupedInteractionBuffers
from yggdrax.interactions import (
    CompactTaggedFarPairs,
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    DualTreeWalkResult,
    MACType,
    NodeInteractionList,
    NodeNeighborList,
    build_compact_far_pairs,
    build_compact_far_pairs_and_leaf_neighbor_lists,
    build_interactions_and_neighbors_split,
    build_leaf_neighbor_lists,
)
from yggdrax.tree import Tree


@dataclass(frozen=True)
class _DualTreeArtifacts:
    """Artifacts emitted by the dual-tree traversal builder."""

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
    """Cache entry for dual-tree interaction artifacts keyed by build options."""

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
    """Resolved cached dual-tree payload reused for a build request."""

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
    """Cached refresh planner decision for dual artifact build routing."""

    use_split_build: bool
    suppress_substage_timing: bool = False


@partial(jax.jit, static_argnames=())
def _compiled_refresh_dual_planner_route(
    *,
    allow_split_build_flag: Array,
    grouped_interactions_flag: Array,
    need_traversal_result_flag: Array,
    has_pair_policy_flag: Array,
    has_policy_state_flag: Array,
    leaf_count: Array,
    need_node_interactions_flag: Array,
    need_compact_far_pairs_flag: Array,
    use_dense_interactions_flag: Array,
) -> tuple[Array, Array, Array]:
    """Return compiled routing decisions for refresh dual-artifact planning.

    This keeps steady-state route/plan branching in JAX control flow so the
    refresh hot path avoids repeated Python-side conditional orchestration.
    """

    use_split_build = (
        allow_split_build_flag
        & (~grouped_interactions_flag)
        & (~need_traversal_result_flag)
        & (~has_pair_policy_flag)
        & (~has_policy_state_flag)
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
    """Drop cached class-major schedule arrays from an interaction cache entry."""
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
    """Return reusable cached dual-tree artifacts when available."""

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
    geometry,
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
    pair_policy,
    policy_state,
    jit_traversal: bool,
) -> tuple[Any, Optional[DualTreeTraversalConfig], Optional[int], Optional[int]]:
    """Run the raw dual-tree traversal builder with retry growth."""

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
    """Normalize raw builder outputs into a fixed tuple."""

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
    pair_policy,
    policy_state,
) -> bool:
    """Return whether far/near traversal can be built in separate passes.

    This path is intentionally narrow. It is meant for the minimum-memory
    streamed GPU regime where we do not need traversal tags/results and can
    trade extra prepare work for a lower peak by never materializing far and
    near traversal buffers in the same kernel.
    """

    return (
        bool(split_enabled)
        and not bool(grouped_interactions)
        and not bool(need_traversal_result)
        and pair_policy is None
        and policy_state is None
    )


def _build_dual_tree_artifacts_split(
    *,
    tree: Tree,
    geometry,
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
    timing_callback: Optional[Callable[[str, float], None]] = None,
) -> _DualTreeArtifacts:
    """Build far and near traversal products in separate Yggdrax calls."""
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


def _dual_tree_build_grouped_buffers(
    *,
    tree: Tree,
    geometry,
    interactions: Optional[NodeInteractionList],
) -> GroupedInteractionBuffers:
    """Materialize grouped interaction buffers from node interaction pairs."""

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
    """Materialize class-major grouped schedule arrays."""

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
    geometry,
    interactions: Optional[NodeInteractionList],
    use_dense_interactions: bool,
) -> Optional[DenseInteractionBuffers]:
    """Materialize dense interaction buffers when requested."""

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
    """Return whether an exception likely indicates traversal-capacity overflow."""
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
    """Scale traversal capacities for one retry attempt."""
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
    """Augment traversal capacity failures with actionable tuning hints."""
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
) -> Optional[str]:
    """Return a hash for the interaction list of a tree/theta configuration.

    If any tree arrays are tracers (for example under grad/jit), return ``None``
    to disable caching and avoid host round-trips on traced values.
    """

    hasher = hashlib.sha256()

    if topology_key is not None:
        hasher.update(b"topology_key_v1")
        hasher.update(str(topology_key).encode("utf8"))
    else:
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
    pair_policy=None,
    policy_state=None,
    jit_traversal: bool = True,
    timing_callback: Optional[Callable[[str, float], None]] = None,
    planner_hint: Optional[_RefreshDualPlannerHint] = None,
) -> tuple[_DualTreeArtifacts, Optional[_InteractionCacheEntry]]:
    """Construct or reuse dual-tree traversal products for a tree."""

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
                pair_policy=pair_policy,
                policy_state=policy_state,
            )
        if use_split_build:
            split_artifacts = _build_dual_tree_artifacts_split(
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
                timing_callback=timing_callback,
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
