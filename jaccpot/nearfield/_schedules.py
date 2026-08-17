"""Leaf-data padding and the precomputed near-field scatter schedules.

Everything the near-field kernels need *shaped* before they run, and nothing that
computes a force. Two groups:

* **leaf-data padding** (``_prepare_leaf_data``, ``_prepare_leaf_data_from_groups``)
  turns the tree's per-leaf particle spans -- or an explicit membership table on
  the large-N path -- into the uniform ``[num_leaves, max_leaf_size]`` arrays every
  kernel here consumes, with the out-of-range slots clipped to a valid index and
  masked to zero so padding contributes exactly nothing (NUMERICS_AND_JAX §2).
* **the scatter schedules** (``prepare_leaf_neighbor_pairs``,
  ``prepare_bucketed_scatter_schedules*``), which are the topology-fixed part of
  the bucketed traversal and therefore cacheable across a same-topology refresh.

READ THE WARNING on :func:`prepare_leaf_neighbor_pairs`. Its default
``sort_by_source=True`` is the *unsafe* value for anything that will be handed
back as one of ``compute_leaf_p2p_accelerations``'s ``precomputed_*`` arrays:
source-sorted and unsorted vectors have identical shapes, so a mismatch produces
wrong forces with nothing downstream able to detect it.

Split out of ``near_field.py`` (Tier 1.5, A.9 seam 1); every function body is
unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jaxtyping import Array
from yggdrax.dtypes import INDEX_DTYPE

from ._scatter import _build_scatter_schedule

__all__ = [
    "prepare_leaf_neighbor_pairs",
    "prepare_bucketed_scatter_schedules",
    "prepare_bucketed_scatter_schedules_from_groups",
]


def prepare_leaf_neighbor_pairs(
    node_ranges: Array,
    leaf_nodes: Array,
    offsets: Array,
    neighbors: Array,
    *,
    sort_by_source: bool = True,
) -> Tuple[Array, Array, Array]:
    """Precompute neighbor-edge leaf mappings and reorder for source locality.

    WARNING: the default ``sort_by_source=True`` is the **unsafe** value if the
    result is going to be stored and handed back as one of
    :func:`compute_leaf_p2p_accelerations`'s ``precomputed_*`` arrays. That
    contract requires positional alignment with ``neighbors``, because a consumer
    that is given ``target_leaf_ids`` but not ``source_leaf_ids`` re-derives the
    latter as ``leaf_lookup[neighbors]`` -- unsorted. Source-sorted and unsorted
    vectors have identical shapes, so nothing downstream can detect the mismatch;
    it produces wrong forces silently. Every producer in this repository that
    feeds the precomputed path passes ``sort_by_source=False`` explicitly.

    The default stays ``True`` because the non-bucketed path uses it live for
    gather locality and does not persist the result.

    Parameters
    ----------
    node_ranges : Array
        Per-node ``[start, stop)`` span into the sorted particle order.
    leaf_nodes : Array
        Node index of each leaf.
    offsets : Array
        Per-leaf offsets into ``neighbors``.
    neighbors : Array
        Flat neighbour-node list.
    sort_by_source : bool
        Reorder the edges by source leaf for gather locality. **Pass ``False``
        whenever the result will be stored and fed back as a ``precomputed_*``
        array** -- see the warning above.

    Returns
    -------
    Tuple[Array, Array, Array]
        ``(target_leaf_ids, source_leaf_ids, valid_pairs)``, positionally
        aligned with ``neighbors`` only when ``sort_by_source`` is ``False``.
    """
    total_nodes = node_ranges.shape[0]
    leaf_lookup = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
    leaf_lookup = leaf_lookup.at[leaf_nodes].set(
        jnp.arange(leaf_nodes.shape[0], dtype=INDEX_DTYPE)
    )
    edge_indices = jnp.arange(neighbors.shape[0], dtype=INDEX_DTYPE)
    target_leaf_ids = jnp.searchsorted(offsets[1:], edge_indices, side="right")
    # `neighbors` may carry -1 padding when the neighbour list is not compacted
    # (e.g. the traced/jax.shard_map branch of _result_to_neighbors keeps the
    # full [num_leaves * max_neighbors] buffer). A raw leaf_lookup[-1] would wrap
    # to a real leaf row; clip + an explicit >=0 mask drops padding edges so both
    # near-field kernels are correct regardless of compaction.
    valid_neighbor = neighbors >= 0
    source_leaf_ids = leaf_lookup[jnp.clip(neighbors, 0, total_nodes - 1)]
    valid_pairs = valid_neighbor & (source_leaf_ids >= 0)

    if not sort_by_source:
        return target_leaf_ids, source_leaf_ids, valid_pairs

    # Reorder once by source leaf to improve repeated gather locality.
    sort_idx = jnp.argsort(source_leaf_ids, stable=True)
    return (
        target_leaf_ids[sort_idx],
        source_leaf_ids[sort_idx],
        valid_pairs[sort_idx],
    )


def prepare_bucketed_scatter_schedules(
    node_ranges: Array,
    leaf_nodes: Array,
    target_leaf_ids: Array,
    valid_pairs: Array,
    *,
    max_leaf_size: int,
    edge_chunk_size: int,
) -> Tuple[Array, Array, Array]:
    """Precompute per-chunk scatter schedules for bucketed near-field scans.
    Built once at prepare time and reused across evaluations, so it is only
    valid while the topology and ``edge_chunk_size`` are unchanged.

    Parameters
    ----------
    node_ranges : Array
        Per-node ``[start, stop)`` span into the sorted particle order.
    leaf_nodes : Array
        Node index of each leaf.
    target_leaf_ids : Array
        Target leaf of each edge. Must be positionally aligned with the edge
        list the schedule will be applied to.
    valid_pairs : Array
        Validity mask over the edges.
    max_leaf_size : int
        Slot capacity per leaf.
    edge_chunk_size : int
        Edges per chunk; part of what the schedule is specialised to.

    Returns
    -------
    Tuple[Array, Array, Array]
        One ``(sort_idx, group_ids, unique_indices)`` schedule per chunk, as
        consumed by :mod:`jaccpot.nearfield._scatter`'s ``*_with_schedule``
        helpers.

    Raises
    ------
    ValueError
        If ``edge_chunk_size`` or ``max_leaf_size`` is not positive.
    """
    chunk = int(edge_chunk_size)
    if chunk <= 0:
        raise ValueError("edge_chunk_size must be positive")
    if int(max_leaf_size) <= 0:
        raise ValueError("max_leaf_size must be positive")

    node_ranges = jnp.asarray(node_ranges, dtype=INDEX_DTYPE)
    leaf_nodes = jnp.asarray(leaf_nodes, dtype=INDEX_DTYPE)
    target_leaf_ids = jnp.asarray(target_leaf_ids, dtype=INDEX_DTYPE)
    valid_pairs = jnp.asarray(valid_pairs, dtype=bool)

    n_edges = int(target_leaf_ids.shape[0])
    flat_size = int(max_leaf_size) * chunk
    if n_edges == 0:
        empty = jnp.zeros((0, flat_size), dtype=INDEX_DTYPE)
        return empty, empty, empty

    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    leaf_offsets = jnp.arange(int(max_leaf_size), dtype=INDEX_DTYPE)
    leaf_particle_idx = leaf_ranges[:, 0][:, None] + leaf_offsets[None, :]
    leaf_mask = leaf_offsets[None, :] < counts[:, None]

    chunk_starts = jnp.arange(0, n_edges, chunk, dtype=INDEX_DTYPE)
    chunk_offsets = jnp.arange(chunk, dtype=INDEX_DTYPE)
    edge_idx = chunk_starts[:, None] + chunk_offsets[None, :]
    in_range = edge_idx < n_edges
    safe_edge_idx = jnp.where(in_range, edge_idx, 0)
    valid_edge = in_range & valid_pairs[safe_edge_idx]

    tgt_leaf = target_leaf_ids[safe_edge_idx]
    tgt_leaf = jnp.where(valid_edge, tgt_leaf, 0)
    tgt_ids = leaf_particle_idx[tgt_leaf]
    tgt_mask = leaf_mask[tgt_leaf] & valid_edge[..., None]

    flat_tgt_ids = tgt_ids.reshape(chunk_starts.shape[0], flat_size)
    flat_tgt_mask = tgt_mask.reshape(chunk_starts.shape[0], flat_size)
    return jax.vmap(
        _build_scatter_schedule,
        in_axes=(0, 0),
        out_axes=0,
    )(flat_tgt_ids, flat_tgt_mask)


def prepare_bucketed_scatter_schedules_from_groups(
    leaf_particle_indices: Array,
    leaf_particle_mask: Array,
    target_leaf_ids: Array,
    valid_pairs: Array,
    *,
    edge_chunk_size: int,
) -> Tuple[Array, Array, Array]:
    """Precompute per-chunk scatter schedules for explicit leaf-particle groups.
    The membership-driven twin of :func:`prepare_bucketed_scatter_schedules`:
    it takes explicit per-leaf particle groups rather than deriving them from
    ``node_ranges``, which is what the large-N lane already has on hand.

    Parameters
    ----------
    leaf_particle_indices : Array
        ``(num_leaves, S)`` particle indices per leaf.
    leaf_particle_mask : Array
        Occupancy mask of the same shape.
    target_leaf_ids : Array
        Target leaf of each edge, positionally aligned with the edge list.
    valid_pairs : Array
        Validity mask over the edges.
    edge_chunk_size : int
        Edges per chunk.

    Returns
    -------
    Tuple[Array, Array, Array]
        One ``(sort_idx, group_ids, unique_indices)`` schedule per chunk.

    Raises
    ------
    ValueError
        If ``edge_chunk_size`` is not positive.
    """
    chunk = int(edge_chunk_size)
    if chunk <= 0:
        raise ValueError("edge_chunk_size must be positive")

    leaf_particle_indices = jnp.asarray(leaf_particle_indices, dtype=INDEX_DTYPE)
    leaf_particle_mask = jnp.asarray(leaf_particle_mask, dtype=bool)
    target_leaf_ids = jnp.asarray(target_leaf_ids, dtype=INDEX_DTYPE)
    valid_pairs = jnp.asarray(valid_pairs, dtype=bool)

    n_edges = int(target_leaf_ids.shape[0])
    flat_size = int(leaf_particle_indices.shape[1]) * chunk
    if n_edges == 0:
        empty = jnp.zeros((0, flat_size), dtype=INDEX_DTYPE)
        return empty, empty, empty

    chunk_starts = jnp.arange(0, n_edges, chunk, dtype=INDEX_DTYPE)
    chunk_offsets = jnp.arange(chunk, dtype=INDEX_DTYPE)
    edge_idx = chunk_starts[:, None] + chunk_offsets[None, :]
    in_range = edge_idx < n_edges
    safe_edge_idx = jnp.where(in_range, edge_idx, 0)
    valid_edge = in_range & valid_pairs[safe_edge_idx]

    tgt_leaf = target_leaf_ids[safe_edge_idx]
    tgt_leaf = jnp.where(valid_edge, tgt_leaf, 0)
    tgt_ids = leaf_particle_indices[tgt_leaf]
    tgt_mask = leaf_particle_mask[tgt_leaf] & valid_edge[..., None]

    flat_tgt_ids = tgt_ids.reshape(chunk_starts.shape[0], flat_size)
    flat_tgt_mask = tgt_mask.reshape(chunk_starts.shape[0], flat_size)
    return jax.vmap(
        _build_scatter_schedule,
        in_axes=(0, 0),
        out_axes=0,
    )(flat_tgt_ids, flat_tgt_mask)


def _prepare_leaf_data(
    node_ranges: Array,
    leaf_nodes: Array,
    positions: Array,
    masses: Array,
    *,
    max_leaf_size: int,
) -> Tuple[Array, Array, Array, Array]:
    """Pad per-leaf particle data to a uniform shape.
    Every leaf is padded to ``max_leaf_size`` so the kernels see one static
    shape. Padding slots get zero mass, which makes them inert in the force sum
    without a branch.

    Parameters
    ----------
    node_ranges : Array
        Per-node ``[start, stop)`` span into the sorted particle order.
    leaf_nodes : Array
        Node index of each leaf.
    positions : Array
        All particle positions, in sorted order.
    masses : Array
        All particle masses, same order.
    max_leaf_size : int
        Slot capacity per leaf.

    Returns
    -------
    Tuple[Array, Array, Array, Array]
        ``(particle_indices, positions, masses, mask)``, the last three padded
        to ``(num_leaves, max_leaf_size, ...)``.
    """

    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1

    idx = jnp.arange(max_leaf_size, dtype=INDEX_DTYPE)
    starts = leaf_ranges[:, 0][:, None]
    particle_idx = starts + idx
    valid = idx[None, :] < counts[:, None]

    safe_idx = jnp.clip(
        particle_idx,
        min=0,
        max=positions.shape[0] - 1,
    )
    leaf_positions = positions[safe_idx]
    leaf_masses = masses[safe_idx]

    leaf_positions = jnp.where(valid[..., None], leaf_positions, 0.0)
    leaf_masses = jnp.where(valid, leaf_masses, 0.0)

    return leaf_positions, leaf_masses, valid, safe_idx


def _prepare_leaf_data_from_groups(
    leaf_particle_indices: Array,
    leaf_particle_mask: Array,
    positions: Array,
    masses: Array,
) -> Tuple[Array, Array, Array, Array]:
    """Gather per-leaf particle data from explicit particle-membership groups.
    Same output contract as :func:`_prepare_leaf_data`, but the membership is
    given rather than derived, so the slot width comes from the caller's groups.

    Parameters
    ----------
    leaf_particle_indices : Array
        ``(num_leaves, S)`` particle indices per leaf.
    leaf_particle_mask : Array
        Occupancy mask of the same shape.
    positions : Array
        All particle positions, in sorted order.
    masses : Array
        All particle masses, same order.

    Returns
    -------
    Tuple[Array, Array, Array, Array]
        ``(particle_indices, positions, masses, mask)``, with masked slots
        carrying zero mass.
    """
    leaf_particle_indices = jnp.asarray(leaf_particle_indices, dtype=INDEX_DTYPE)
    leaf_particle_mask = jnp.asarray(leaf_particle_mask, dtype=bool)
    if leaf_particle_indices.size == 0:
        empty_pos = jnp.zeros(
            (leaf_particle_indices.shape[0], 0, positions.shape[-1]),
            dtype=positions.dtype,
        )
        empty_mass = jnp.zeros((leaf_particle_indices.shape[0], 0), dtype=masses.dtype)
        return empty_pos, empty_mass, leaf_particle_mask, leaf_particle_indices

    safe_idx = jnp.clip(
        leaf_particle_indices,
        min=0,
        max=positions.shape[0] - 1,
    )
    leaf_positions = positions[safe_idx]
    leaf_masses = masses[safe_idx]
    leaf_positions = jnp.where(leaf_particle_mask[..., None], leaf_positions, 0.0)
    leaf_masses = jnp.where(leaf_particle_mask, leaf_masses, 0.0)
    return leaf_positions, leaf_masses, leaf_particle_mask, safe_idx
