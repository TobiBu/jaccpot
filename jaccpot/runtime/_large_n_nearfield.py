"""Canonical near-field helpers for the large-N runtime path."""

from __future__ import annotations

import os
from typing import Any, Optional

import jax.numpy as jnp
from jaxtyping import Array
from yggdrax.interactions import NodeNeighborList
from yggdrax.tree import Tree

from jaccpot.nearfield.near_field import (
    compute_leaf_p2p_accelerations,
    compute_leaf_p2p_accelerations_radix_payload_pairs_only,
    compute_leaf_p2p_accelerations_target_block_pairs_only,
    compute_leaf_p2p_accelerations_radix_fast_lane,
    prepare_bucketed_scatter_schedules_from_groups,
    prepare_leaf_neighbor_pairs,
)

from ._large_n_types import LargeNExecutionConfig, LargeNPreparedState
from ._nearfield_cache import NearfieldPrecomputeArtifacts
from .dtypes import INDEX_DTYPE, as_index

_RADIX_FAST_LANE_DEFAULT_TARGET_BLOCK_SIZE = 32


def build_large_n_leaf_particle_groups(
    tree: Tree,
    neighbor_list: NodeNeighborList,
    *,
    max_leaf_size: Optional[int] = None,
) -> tuple[Array, Array]:
    """Return explicit per-leaf particle membership for radix full evaluation."""

    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    if leaf_nodes.size == 0:
        width = 0 if max_leaf_size is None else int(max_leaf_size)
        return (
            jnp.zeros((0, width), dtype=INDEX_DTYPE),
            jnp.zeros((0, width), dtype=bool),
        )

    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    width = int(jnp.max(counts)) if max_leaf_size is None else int(max_leaf_size)
    if width <= 0:
        return (
            jnp.zeros((leaf_nodes.shape[0], 0), dtype=INDEX_DTYPE),
            jnp.zeros((leaf_nodes.shape[0], 0), dtype=bool),
        )

    offsets = jnp.arange(width, dtype=INDEX_DTYPE)
    particle_idx = leaf_ranges[:, 0][:, None] + offsets[None, :]
    particle_mask = offsets[None, :] < counts[:, None]
    return jnp.asarray(particle_idx, dtype=INDEX_DTYPE), jnp.asarray(
        particle_mask, dtype=bool
    )


def resolve_large_n_execution_config(
    fmm: object,
    *,
    num_particles: int,
) -> LargeNExecutionConfig:
    """Resolve near-field policy for the locked large-N radix fast path.

    The production large-N GPU radix/solidfmm path is locked to radix fast-lane
    execution. If no valid explicit target block size is provided via
    ``JACCPOT_LARGE_N_TARGET_BLOCK_SIZE``, the fast lane defaults to block size 32.
    """

    nearfield_mode = "bucketed"
    edge_chunk_size = int(
        fmm._resolve_nearfield_edge_chunk_size(
            num_particles=int(num_particles),
            nearfield_mode=nearfield_mode,
        )
    )
    retain_pair_vectors = (
        str(getattr(fmm, "memory_objective", "")).strip().lower() != "minimum_memory"
    )
    target_owned_block_size = int(
        os.environ.get("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "0")
    )
    if target_owned_block_size < 0:
        target_owned_block_size = 0

    tree_type = str(getattr(fmm, "tree_type", "")).strip().lower()
    preset = str(getattr(fmm, "preset", "")).strip().lower()
    basis = str(getattr(fmm, "expansion_basis", "")).strip().lower()
    grouped = bool(getattr(fmm, "grouped_interactions", False))
    working_dtype = jnp.dtype(getattr(fmm, "working_dtype", jnp.float32))

    if tree_type != "radix":
        raise ValueError("radix_fast_lane requires tree_type='radix'")
    if preset != "large_n_gpu":
        raise ValueError("radix_fast_lane requires preset='large_n_gpu'")
    if basis != "solidfmm":
        raise ValueError("radix_fast_lane requires expansion_basis='solidfmm'")
    if working_dtype != jnp.float32:
        raise ValueError("radix_fast_lane requires working_dtype=float32")
    if grouped:
        raise ValueError("radix_fast_lane requires grouped_interactions=False")
    if int(target_owned_block_size) <= 0:
        target_owned_block_size = int(_RADIX_FAST_LANE_DEFAULT_TARGET_BLOCK_SIZE)

    radix_fast_lane = True
    # Fast lane is a fixed-shape TONB path; force prepare-time canonical layout.
    speed_prepared_layout = True
    precompute_scatter = False

    return LargeNExecutionConfig(
        nearfield_mode=nearfield_mode,
        nearfield_edge_chunk_size=edge_chunk_size,
        retain_leaf_groups=True,
        retain_pair_vectors=retain_pair_vectors,
        precompute_scatter=precompute_scatter,
        target_owned_block_size=target_owned_block_size,
        speed_prepared_layout=speed_prepared_layout,
        radix_fast_lane=radix_fast_lane,
    )


def build_large_n_target_owned_blocks(
    *,
    tree: Tree,
    neighbor_list: NodeNeighborList,
    block_size: int,
) -> tuple[Array, Array, Array, Array]:
    """Precompute target-owned source-leaf blocks for the large-N nearfield path."""

    k = int(block_size)
    if k <= 0:
        num_leaves = int(
            jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE).shape[0]
        )
        return (
            jnp.zeros((0,), dtype=INDEX_DTYPE),
            jnp.zeros((0, 0), dtype=INDEX_DTYPE),
            jnp.zeros((0, 0), dtype=bool),
            jnp.zeros((num_leaves + 1,), dtype=INDEX_DTYPE),
        )

    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    offsets = jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE)
    neighbors = jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE)
    if int(leaf_nodes.shape[0]) == 0 or int(neighbors.shape[0]) == 0:
        return (
            jnp.zeros((0,), dtype=INDEX_DTYPE),
            jnp.zeros((0, k), dtype=INDEX_DTYPE),
            jnp.zeros((0, k), dtype=bool),
            jnp.zeros((int(leaf_nodes.shape[0]) + 1,), dtype=INDEX_DTYPE),
        )

    _, source_leaf_ids, valid_pairs = prepare_leaf_neighbor_pairs(
        node_ranges,
        leaf_nodes,
        offsets,
        neighbors,
        sort_by_source=False,
    )
    counts = offsets[1:] - offsets[:-1]
    blocks_per_leaf = (counts + as_index(k - 1)) // as_index(k)
    block_offsets = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=INDEX_DTYPE),
            jnp.cumsum(blocks_per_leaf, dtype=INDEX_DTYPE),
        ]
    )
    total_blocks = int(block_offsets[-1])
    if total_blocks == 0:
        return (
            jnp.zeros((0,), dtype=INDEX_DTYPE),
            jnp.zeros((0, k), dtype=INDEX_DTYPE),
            jnp.zeros((0, k), dtype=bool),
            jnp.asarray(block_offsets, dtype=INDEX_DTYPE),
        )

    block_ids = jnp.arange(total_blocks, dtype=INDEX_DTYPE)
    block_target_leaf_ids = jnp.searchsorted(
        block_offsets[1:],
        block_ids,
        side="right",
    )
    local_block_idx = block_ids - block_offsets[block_target_leaf_ids]
    edge_start = offsets[block_target_leaf_ids] + local_block_idx * as_index(k)
    edge_stop = offsets[block_target_leaf_ids + as_index(1)]
    edge_offsets = jnp.arange(k, dtype=INDEX_DTYPE)
    edge_idx = edge_start[:, None] + edge_offsets[None, :]
    in_leaf = edge_idx < edge_stop[:, None]
    safe_edge_idx = jnp.where(in_leaf, edge_idx, 0)
    edge_valid = in_leaf & valid_pairs[safe_edge_idx]
    block_source_leaf_ids = jnp.where(
        edge_valid,
        source_leaf_ids[safe_edge_idx],
        0,
    )
    return (
        jnp.asarray(block_target_leaf_ids, dtype=INDEX_DTYPE),
        jnp.asarray(block_source_leaf_ids, dtype=INDEX_DTYPE),
        jnp.asarray(edge_valid, dtype=bool),
        jnp.asarray(block_offsets, dtype=INDEX_DTYPE),
    )


def build_large_n_target_owned_blocks_static(
    *,
    tree: Tree,
    neighbor_list: NodeNeighborList,
    block_size: int,
    max_blocks_per_leaf: int,
) -> tuple[Array, Array, bool]:
    """Build fixed-capacity target-owned source-leaf blocks.

    Returns ``(source_leaf_ids_padded, valid_mask_padded, capacity_ok)`` where
    the first two tensors have shape ``(num_leaves, max_blocks_per_leaf,
    block_size)``. If any leaf needs more than ``max_blocks_per_leaf`` blocks,
    ``capacity_ok`` is false and callers should fall back to the dynamic
    builder.
    """

    k = int(block_size)
    max_blocks = int(max_blocks_per_leaf)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    num_leaves = int(leaf_nodes.shape[0])
    if k <= 0 or max_blocks <= 0:
        return (
            jnp.zeros((num_leaves, 0, 0), dtype=INDEX_DTYPE),
            jnp.zeros((num_leaves, 0, 0), dtype=bool),
            False,
        )

    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    offsets = jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE)
    neighbors = jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE)
    if num_leaves == 0:
        return (
            jnp.zeros((0, max_blocks, k), dtype=INDEX_DTYPE),
            jnp.zeros((0, max_blocks, k), dtype=bool),
            True,
        )

    counts = offsets[1:] - offsets[:-1]
    max_count = int(jnp.max(counts)) if int(counts.shape[0]) > 0 else 0
    if max_count > max_blocks * k:
        return (
            jnp.zeros((num_leaves, max_blocks, k), dtype=INDEX_DTYPE),
            jnp.zeros((num_leaves, max_blocks, k), dtype=bool),
            False,
        )

    total_nodes = int(node_ranges.shape[0])
    leaf_lookup = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
    leaf_lookup = leaf_lookup.at[leaf_nodes].set(
        jnp.arange(num_leaves, dtype=INDEX_DTYPE)
    )

    block_offsets = jnp.arange(max_blocks, dtype=INDEX_DTYPE)
    slot_offsets = jnp.arange(k, dtype=INDEX_DTYPE)
    edge_idx = (
        offsets[:-1, None, None]
        + block_offsets[None, :, None] * as_index(k)
        + slot_offsets[None, None, :]
    )
    edge_stop = offsets[1:, None, None]
    in_leaf = edge_idx < edge_stop
    safe_edge_idx = jnp.where(in_leaf, edge_idx, 0)
    source_leaf_ids = leaf_lookup[neighbors[safe_edge_idx]]
    valid_mask = in_leaf & (source_leaf_ids >= 0)
    return (
        jnp.where(valid_mask, source_leaf_ids, 0).astype(INDEX_DTYPE),
        jnp.asarray(valid_mask, dtype=bool),
        True,
    )


def build_large_n_nearfield_precompute(
    *,
    tree: Tree,
    neighbor_list: NodeNeighborList,
    leaf_particle_indices: Array,
    leaf_particle_mask: Array,
    execution_config: LargeNExecutionConfig,
) -> NearfieldPrecomputeArtifacts:
    """Build only the near-field artifacts needed by the large-N path."""

    if str(execution_config.nearfield_mode).strip().lower() != "bucketed":
        raise RuntimeError(
            "large_n nearfield precompute requires nearfield_mode='bucketed'"
        )

    if (not bool(execution_config.retain_pair_vectors)) and (
        not bool(execution_config.precompute_scatter)
    ):
        # Minimum-memory large-N runs do not benefit from retaining explicit
        # leaf-pair vectors when we are also skipping scatter-schedule
        # precomputation; the kernel can derive the edge order on demand.
        return NearfieldPrecomputeArtifacts(
            target_leaf_ids=None,
            source_leaf_ids=None,
            valid_pairs=None,
            chunk_sort_indices=None,
            chunk_group_ids=None,
            chunk_unique_indices=None,
        )

    target_leaf_ids, source_leaf_ids, valid_pairs = prepare_leaf_neighbor_pairs(
        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE),
        jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE),
        jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE),
        sort_by_source=False,
    )

    chunk_sort_indices = None
    chunk_group_ids = None
    chunk_unique_indices = None
    if bool(execution_config.precompute_scatter):
        (
            chunk_sort_indices,
            chunk_group_ids,
            chunk_unique_indices,
        ) = prepare_bucketed_scatter_schedules_from_groups(
            leaf_particle_indices,
            leaf_particle_mask,
            target_leaf_ids,
            valid_pairs,
            edge_chunk_size=int(execution_config.nearfield_edge_chunk_size),
        )

    return NearfieldPrecomputeArtifacts(
        target_leaf_ids=(
            target_leaf_ids if bool(execution_config.retain_pair_vectors) else None
        ),
        source_leaf_ids=(
            source_leaf_ids if bool(execution_config.retain_pair_vectors) else None
        ),
        valid_pairs=(
            valid_pairs if bool(execution_config.retain_pair_vectors) else None
        ),
        chunk_sort_indices=chunk_sort_indices,
        chunk_group_ids=chunk_group_ids,
        chunk_unique_indices=chunk_unique_indices,
    )


def evaluate_large_n_nearfield_fast_lane(
    fmm: object,
    state: LargeNPreparedState,
    *,
    return_potential: bool,
) -> Any:
    """Evaluate the radix fast-lane nearfield path (payload-driven)."""

    # Keep potential compatibility by delegating to canonical generic nearfield
    # until dedicated fast-lane potential accumulation is implemented.
    if bool(return_potential):
        has_leaf_groups = int(state.nearfield_leaf_particle_indices.size) > 0
        leaf_particle_indices_override = (
            state.nearfield_leaf_particle_indices if has_leaf_groups else None
        )
        leaf_particle_mask_override = (
            state.nearfield_leaf_particle_mask if has_leaf_groups else None
        )
        return compute_leaf_p2p_accelerations(
            state.tree,
            state.neighbor_list,
            state.positions_sorted,
            state.masses_sorted,
            G=getattr(fmm, "G"),
            softening=float(getattr(fmm, "softening")),
            max_leaf_size=int(state.max_leaf_size),
            return_potential=True,
            nearfield_mode=str(state.nearfield_mode),
            edge_chunk_size=int(state.nearfield_edge_chunk_size),
            precomputed_target_leaf_ids=state.nearfield_target_leaf_ids,
            precomputed_source_leaf_ids=state.nearfield_source_leaf_ids,
            precomputed_valid_pairs=state.nearfield_valid_pairs,
            precomputed_chunk_sort_indices=state.nearfield_chunk_sort_indices,
            precomputed_chunk_group_ids=state.nearfield_chunk_group_ids,
            precomputed_chunk_unique_indices=state.nearfield_chunk_unique_indices,
            leaf_particle_indices_override=leaf_particle_indices_override,
            leaf_particle_mask_override=leaf_particle_mask_override,
        )

    if state.radix_fast_payload is None:
        raise RuntimeError(
            "radix fast-lane evaluate requires radix_fast_payload to be present"
        )

    near_acc = compute_leaf_p2p_accelerations_radix_fast_lane(
        positions_sorted=state.positions_sorted,
        masses_sorted=state.masses_sorted,
        payload=state.radix_fast_payload,
        G=getattr(fmm, "G"),
        softening=float(getattr(fmm, "softening")),
        return_potential=False,
    )
    overflow_payload = getattr(state, "radix_overflow_payload", None)
    if overflow_payload is not None:
        return near_acc + compute_leaf_p2p_accelerations_radix_payload_pairs_only(
            positions_sorted=state.positions_sorted,
            masses_sorted=state.masses_sorted,
            payload=overflow_payload,
            G=getattr(fmm, "G"),
            softening=float(getattr(fmm, "softening")),
        )

    if (
        state.nearfield_target_block_offsets is None
        or state.nearfield_target_block_leaf_ids is None
        or state.nearfield_target_block_source_leaf_ids is None
        or state.nearfield_target_block_valid_mask is None
        or int(state.nearfield_target_block_source_leaf_ids.size) == 0
    ):
        return near_acc

    overflow_acc = compute_leaf_p2p_accelerations_target_block_pairs_only(
        state.positions_sorted,
        state.masses_sorted,
        state.nearfield_leaf_particle_indices,
        state.nearfield_leaf_particle_mask,
        state.nearfield_target_block_offsets,
        state.nearfield_target_block_leaf_ids,
        state.nearfield_target_block_source_leaf_ids,
        state.nearfield_target_block_valid_mask,
        G=getattr(fmm, "G"),
        softening=float(getattr(fmm, "softening")),
        target_leaf_batch_size=int(state.nearfield_target_leaf_batch_size),
        target_block_tile_size=int(state.nearfield_target_block_tile_size),
        target_block_tile_scan_unroll=int(
            state.nearfield_target_block_tile_scan_unroll
        ),
        target_block_batch_scan_unroll=int(
            state.nearfield_target_block_batch_scan_unroll
        ),
        target_block_overflow_fast_max_blocks=int(
            state.nearfield_target_block_overflow_fast_max_blocks
        ),
    )
    return near_acc + overflow_acc
