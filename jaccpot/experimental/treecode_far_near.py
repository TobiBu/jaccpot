"""Device-resident treecode far-pair + near-neighbor producer (2B step 3, isolated).

Bridges the per-leaf treecode walk to the structures the large-N fused fast-lane
consumes, so the walk can eventually replace the yggdrax dual-tree walk at
``_interaction_cache._build_dual_tree_artifacts_split_strict_streamed``.

The walk (Pallas on Ampere, pure-JAX fallback) yields, per target leaf, a padded
list of accepted far source node ids + near source leaf ids. This module compacts
those into:

* FAR: a flat ``(sources, targets)`` COO where every ``target`` is a LEAF node id
  (per-leaf treecode) -> feeds ``_FarPairCOO`` / the solidfmm M2L unchanged. Because
  targets are leaves only, every internal node's local expansion stays zero, so the
  downstream L2L cascade propagates zero and is a mathematical no-op (no L2L skip
  needed, no double-count). ``tags`` is emitted as ``-1`` to match the yggdrax
  ``CompactTaggedFarPairs`` convention, though the fused far path never reads it.
* NEAR: a CSR ``(leaf_indices, offsets, neighbors, counts)`` with the target leaf's
  OWN id removed from each row (the treecode marks self near; the fused near-field
  adds the self block once, so the neighbor list must EXCLUDE self, matching the
  yggdrax ``NodeNeighborList`` convention).

Everything here is plain jitted JAX (masked scatters/prefix sums); only the walk
is a Pallas kernel. Correctness target: the assembled treecode force (far M2L into
leaf locals + L2P + near P2P) matches direct N-body to theta tolerance.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array

from jaccpot.experimental.treecode_walk import TreecodeLeafLists, treecode_leaf_walk
from jaccpot.pallas.treecode_walk_pallas import (
    pallas_treecode_walk_supported,
    treecode_leaf_walk_pallas,
)


class TreecodeFarNearLists(NamedTuple):
    """Compacted far COO + near CSR produced from the per-leaf treecode walk.

    far_sources / far_targets: (far_capacity,) flat far pairs; ``target`` is always
        a leaf node id. Valid prefix length is ``far_pair_count``; the tail is 0-padded.
    far_tags: (far_capacity,) all ``-1`` (yggdrax CompactTaggedFarPairs compat).
    far_pair_count: () number of valid far pairs (<= far_capacity).
    near_leaf_indices: (num_leaves,) leaf node id of each CSR row.
    near_offsets: (num_leaves + 1,) CSR row offsets into ``near_neighbors``.
    near_neighbors: (near_capacity,) flat source-leaf node ids, self excluded.
    near_counts: (num_leaves,) per-row neighbor count (self excluded).
    overflow: () bool, walk or a compaction capacity overflowed.
    """

    far_sources: Array
    far_targets: Array
    far_tags: Array
    far_pair_count: Array
    near_leaf_indices: Array
    near_offsets: Array
    near_neighbors: Array
    near_counts: Array
    overflow: Array


def treecode_walk_dispatch(
    leaf_nodes: Array,
    centers: Array,
    mac_extents: Array,
    left_child_full: Array,
    right_child_full: Array,
    theta_sq: Array,
    root_idx: Array,
    *,
    num_internal: int,
    max_far: int,
    max_near: int,
    max_stack: int,
    max_iters: int,
    use_pallas: Optional[bool] = None,
    interpret: bool = False,
) -> TreecodeLeafLists:
    """Run the treecode walk on the best backend (dual-path, mirrors use_pallas).

    ``use_pallas=None`` -> Pallas where supported (Ampere+), else the pure-JAX walk.
    ``interpret=True`` forces the Pallas kernel in interpret mode (CPU parity path).
    """

    if use_pallas is None:
        use_pallas = interpret or pallas_treecode_walk_supported()
    if use_pallas:
        return treecode_leaf_walk_pallas(
            leaf_nodes,
            centers,
            mac_extents,
            left_child_full,
            right_child_full,
            theta_sq,
            root_idx,
            num_internal=num_internal,
            max_far=max_far,
            max_near=max_near,
            max_stack=max_stack,
            max_iters=max_iters,
            interpret=interpret,
        )
    return treecode_leaf_walk(
        leaf_nodes,
        centers,
        mac_extents,
        left_child_full,
        right_child_full,
        theta_sq,
        root_idx,
        num_internal=num_internal,
        max_far=max_far,
        max_near=max_near,
        max_stack=max_stack,
        max_iters=max_iters,
    )


def _exclusive_cumsum(x: Array) -> Array:
    return jnp.cumsum(x) - x


def _compact_far(far_nodes, far_count, leaf_nodes, capacity, idx_dtype):
    """Ragged (num_leaves, max_far) far list -> flat COO (sources, targets)."""
    num_leaves, max_far = far_nodes.shape
    j = jnp.arange(max_far, dtype=idx_dtype)
    valid = j[None, :] < far_count[:, None].astype(idx_dtype)
    start = _exclusive_cumsum(far_count.astype(idx_dtype))
    pos = start[:, None] + j[None, :]
    in_cap = valid & (pos < capacity)
    scatter_pos = jnp.where(in_cap, pos, capacity).reshape(-1)  # OOB -> dropped

    sources = (
        jnp.zeros((capacity,), idx_dtype)
        .at[scatter_pos]
        .set(far_nodes.reshape(-1).astype(idx_dtype), mode="drop")
    )
    tgt_vals = jnp.broadcast_to(leaf_nodes[:, None], far_nodes.shape)
    targets = (
        jnp.zeros((capacity,), idx_dtype)
        .at[scatter_pos]
        .set(tgt_vals.reshape(-1).astype(idx_dtype), mode="drop")
    )
    total = jnp.sum(far_count.astype(idx_dtype))
    far_pair_count = jnp.minimum(total, jnp.asarray(capacity, idx_dtype))
    overflow = total > capacity
    tags = jnp.full((capacity,), -1, dtype=idx_dtype)
    return sources, targets, tags, far_pair_count, overflow


def _compact_near(near_leaves, near_count, leaf_nodes, capacity, idx_dtype):
    """Ragged near list -> CSR (offsets, neighbors, counts), self excluded."""
    num_leaves, max_near = near_leaves.shape
    j = jnp.arange(max_near, dtype=idx_dtype)
    within = j[None, :] < near_count[:, None].astype(idx_dtype)
    not_self = near_leaves != leaf_nodes[:, None]
    valid = within & not_self
    counts = jnp.sum(valid, axis=1).astype(idx_dtype)
    offsets = jnp.concatenate([jnp.zeros((1,), idx_dtype), jnp.cumsum(counts)])
    start = offsets[:-1]
    rank = jnp.cumsum(valid.astype(idx_dtype), axis=1) - 1  # within-row rank of valids
    pos = start[:, None] + rank
    in_cap = valid & (pos < capacity)
    scatter_pos = jnp.where(in_cap, pos, capacity).reshape(-1)
    neighbors = (
        jnp.zeros((capacity,), idx_dtype)
        .at[scatter_pos]
        .set(near_leaves.reshape(-1).astype(idx_dtype), mode="drop")
    )
    total = jnp.sum(counts)
    overflow = total > capacity
    return offsets, neighbors, counts, overflow


def build_treecode_far_pairs_and_neighbors(
    leaf_nodes: Array,
    centers: Array,
    mac_extents: Array,
    left_child_full: Array,
    right_child_full: Array,
    theta_sq: Array,
    root_idx: Array,
    *,
    num_internal: int,
    max_far: int,
    max_near: int,
    max_stack: int,
    max_iters: int,
    far_pair_capacity: int,
    near_capacity: int,
    idx_dtype=None,
    use_pallas: Optional[bool] = None,
    interpret: bool = False,
) -> TreecodeFarNearLists:
    """Run the treecode walk and compact it into fused-lane far COO + near CSR.

    ``far_pair_capacity`` / ``near_capacity`` are the flat output widths (static);
    size them so ``overflow`` stays False (num_leaves * max_far / max_near is always
    sufficient). ``idx_dtype`` defaults to ``left_child_full.dtype``.
    """

    left_child_full = jnp.asarray(left_child_full)
    if idx_dtype is None:
        idx_dtype = left_child_full.dtype
    leaf_nodes = jnp.asarray(leaf_nodes, dtype=idx_dtype)

    walk = treecode_walk_dispatch(
        leaf_nodes,
        centers,
        mac_extents,
        left_child_full,
        jnp.asarray(right_child_full, dtype=idx_dtype),
        theta_sq,
        root_idx,
        num_internal=num_internal,
        max_far=max_far,
        max_near=max_near,
        max_stack=max_stack,
        max_iters=max_iters,
        use_pallas=use_pallas,
        interpret=interpret,
    )

    far_sources, far_targets, far_tags, far_pair_count, far_ovf = _compact_far(
        jnp.asarray(walk.far_nodes, idx_dtype),
        walk.far_count,
        leaf_nodes,
        int(far_pair_capacity),
        idx_dtype,
    )
    near_offsets, near_neighbors, near_counts, near_ovf = _compact_near(
        jnp.asarray(walk.near_leaves, idx_dtype),
        walk.near_count,
        leaf_nodes,
        int(near_capacity),
        idx_dtype,
    )
    overflow = jnp.logical_or(
        jnp.asarray(walk.overflow), jnp.logical_or(far_ovf, near_ovf)
    )
    return TreecodeFarNearLists(
        far_sources=far_sources,
        far_targets=far_targets,
        far_tags=far_tags,
        far_pair_count=far_pair_count,
        near_leaf_indices=leaf_nodes,
        near_offsets=near_offsets,
        near_neighbors=near_neighbors,
        near_counts=near_counts,
        overflow=overflow,
    )


__all__ = [
    "TreecodeFarNearLists",
    "build_treecode_far_pairs_and_neighbors",
    "treecode_walk_dispatch",
]
