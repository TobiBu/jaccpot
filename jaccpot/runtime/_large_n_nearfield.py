"""Canonical near-field helpers for the large-N runtime path."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array
from yggdrax.interactions import NodeNeighborList
from yggdrax.tree import Tree

from jaccpot.nearfield.near_field import (
    compute_leaf_p2p_accelerations,
    prepare_bucketed_scatter_schedules_from_groups,
    prepare_leaf_neighbor_pairs,
)

from ._nearfield_cache import NearfieldPrecomputeArtifacts
from ._large_n_types import LargeNExecutionConfig, LargeNPreparedState
from .dtypes import INDEX_DTYPE

_NEARFIELD_GPU_PRECOMPUTE_MAX_PARTICLES = 131072


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
    """Resolve the reduced near-field policy for the large-N runtime path."""

    nearfield_mode = str(fmm._resolve_nearfield_mode(num_particles=int(num_particles)))
    edge_chunk_size = int(
        fmm._resolve_nearfield_edge_chunk_size(
            num_particles=int(num_particles),
            nearfield_mode=nearfield_mode,
        )
    )
    retain_pair_vectors = (
        str(getattr(fmm, "memory_objective", "")).strip().lower()
        != "minimum_memory"
    )
    precompute_scatter = bool(getattr(fmm, "precompute_nearfield_scatter_schedules"))
    if jax.default_backend() == "gpu":
        precompute_scatter = precompute_scatter and (
            int(num_particles) <= _NEARFIELD_GPU_PRECOMPUTE_MAX_PARTICLES
        )
    return LargeNExecutionConfig(
        nearfield_mode=nearfield_mode,
        nearfield_edge_chunk_size=edge_chunk_size,
        retain_leaf_groups=(nearfield_mode == "bucketed"),
        retain_pair_vectors=retain_pair_vectors,
        precompute_scatter=precompute_scatter,
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
        return NearfieldPrecomputeArtifacts(
            target_leaf_ids=None,
            source_leaf_ids=None,
            valid_pairs=None,
            chunk_sort_indices=None,
            chunk_group_ids=None,
            chunk_unique_indices=None,
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
        valid_pairs=(valid_pairs if bool(execution_config.retain_pair_vectors) else None),
        chunk_sort_indices=chunk_sort_indices,
        chunk_group_ids=chunk_group_ids,
        chunk_unique_indices=chunk_unique_indices,
    )


def evaluate_large_n_nearfield(
    fmm: object,
    state: LargeNPreparedState,
    *,
    return_potential: bool,
):
    """Evaluate near-field contributions for the slim large-N state."""

    if int(state.nearfield_leaf_particle_indices.size) > 0:
        leaf_particle_indices_override = state.nearfield_leaf_particle_indices
        leaf_particle_mask_override = state.nearfield_leaf_particle_mask
    else:
        leaf_particle_indices_override = None
        leaf_particle_mask_override = None

    return compute_leaf_p2p_accelerations(
        state.tree,
        state.neighbor_list,
        state.positions_sorted,
        state.masses_sorted,
        G=getattr(fmm, "G"),
        softening=float(getattr(fmm, "softening")),
        max_leaf_size=int(state.max_leaf_size),
        return_potential=bool(return_potential),
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
