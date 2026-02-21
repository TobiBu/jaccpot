"""Near-field evaluation helpers for the Fast Multipole Method."""

from __future__ import annotations

from functools import partial
from typing import Optional, Union

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jax import lax
from jaxtyping import Array, jaxtyped

from yggdrasil.dtypes import INDEX_DTYPE, as_index
from yggdrasil.tree import RadixTree
from yggdrasil.interactions import NodeNeighborList


def prepare_leaf_neighbor_pairs(
    node_ranges: Array,
    leaf_nodes: Array,
    offsets: Array,
    neighbors: Array,
    *,
    sort_by_source: bool = True,
) -> Tuple[Array, Array, Array]:
    """Precompute neighbor-edge leaf mappings and reorder for source locality."""
    total_nodes = node_ranges.shape[0]
    leaf_lookup = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
    leaf_lookup = leaf_lookup.at[leaf_nodes].set(
        jnp.arange(leaf_nodes.shape[0], dtype=INDEX_DTYPE)
    )
    edge_indices = jnp.arange(neighbors.shape[0], dtype=INDEX_DTYPE)
    target_leaf_ids = jnp.searchsorted(offsets[1:], edge_indices, side="right")
    source_leaf_ids = leaf_lookup[neighbors]
    valid_pairs = source_leaf_ids >= 0

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
    """Precompute per-chunk scatter schedules for bucketed near-field scans."""
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


def _prepare_leaf_data(
    node_ranges: Array,
    leaf_nodes: Array,
    positions: Array,
    masses: Array,
    *,
    max_leaf_size: int,
) -> Tuple[Array, Array, Array, Array]:
    """Pad per-leaf particle data to a uniform shape."""

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


def _self_contributions(
    leaf_positions: Array,
    leaf_masses: Array,
    mask: Array,
    *,
    softening_sq: float,
    G: Array,
    compute_potential: bool,
) -> Tuple[Array, Optional[Array]]:
    dtype = leaf_positions.dtype
    leaf_size = leaf_positions.shape[1]
    identity = jnp.eye(leaf_size, dtype=bool)
    eps = jnp.finfo(dtype).eps

    def compute_single(args):
        positions_leaf, masses_leaf, mask_leaf = args
        diff = positions_leaf[:, None, :] - positions_leaf[None, :, :]
        dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq

        pair_mask = mask_leaf[:, None] & mask_leaf[None, :] & (~identity)
        inv_r = jnp.where(pair_mask, 1.0 / (jnp.sqrt(dist_sq) + eps), 0.0)
        inv_dist3 = jnp.where(pair_mask, inv_r / dist_sq, 0.0)

        weighted = inv_dist3[..., None] * masses_leaf[None, :, None]
        accel_leaf = -G * jnp.sum(weighted * diff, axis=1)
        accel_leaf = jnp.where(mask_leaf[:, None], accel_leaf, 0.0)

        if compute_potential:
            pot_leaf = -G * jnp.sum(inv_r * masses_leaf[None, :], axis=1)
            pot_leaf = jnp.where(mask_leaf, pot_leaf, 0.0)
        else:
            pot_leaf = jnp.zeros((leaf_size,), dtype=dtype)

        return accel_leaf, pot_leaf

    def scan_step(carry, args):
        accel_leaf, pot_leaf = compute_single(args)
        return carry, (accel_leaf, pot_leaf)

    _, (accels, potentials) = lax.scan(
        scan_step,
        None,
        (leaf_positions, leaf_masses, mask),
    )

    if compute_potential:
        return accels, potentials
    return accels, None


def _pair_contributions(
    target_positions: Array,
    target_mask: Array,
    source_positions: Array,
    source_masses: Array,
    source_mask: Array,
    *,
    softening_sq: float,
    G: Array,
    compute_potential: bool,
) -> Tuple[Array, Optional[Array]]:
    dtype = target_positions.dtype
    eps = jnp.finfo(dtype).eps

    source_pos = source_positions
    source_mass = source_masses
    source_active = source_mask
    mass_effective = jnp.where(source_active, source_mass, 0.0)

    soft = softening_sq

    def when_valid(pos):
        diff = pos - source_pos
        dist_sq = jnp.sum(diff * diff, axis=1) + soft
        mask_src = source_active

        inv_r = jnp.where(mask_src, 1.0 / (jnp.sqrt(dist_sq) + eps), 0.0)
        inv_dist3 = jnp.where(mask_src, inv_r / dist_sq, 0.0)

        weighted = inv_dist3[:, None] * mass_effective[:, None]
        accel = -G * jnp.sum(weighted * diff, axis=0)

        if compute_potential:
            pot = -G * jnp.sum(inv_r * mass_effective)
        else:
            pot = jnp.zeros((), dtype=dtype)

        return accel, pot

    def scan_step(carry, data):
        pos, valid = data

        accel, pot = lax.cond(
            valid,
            when_valid,
            lambda _: (
                jnp.zeros((3,), dtype=dtype),
                jnp.zeros((), dtype=dtype),
            ),
            pos,
        )
        return carry, (accel, pot)

    _, (accels, potentials) = lax.scan(
        scan_step,
        None,
        (target_positions, target_mask),
    )

    if compute_potential:
        potentials = jnp.where(target_mask, potentials, 0.0)
        return accels, potentials

    return accels, None


@partial(jax.jit, static_argnames=("compute_potential",))
def _pair_contributions_batched(
    target_positions: Array,
    target_mask: Array,
    source_positions: Array,
    source_masses: Array,
    source_mask: Array,
    *,
    softening_sq: float,
    G: Array,
    compute_potential: bool,
) -> Tuple[Array, Optional[Array]]:
    """Vectorized pair contributions for a batch of target/source leaf pairs."""
    dtype = target_positions.dtype
    eps = jnp.finfo(dtype).eps

    diff = target_positions[:, :, None, :] - source_positions[:, None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
    pair_mask = target_mask[:, :, None] & source_mask[:, None, :]

    inv_r = jnp.where(pair_mask, 1.0 / (jnp.sqrt(dist_sq) + eps), 0.0)
    inv_dist3 = jnp.where(pair_mask, inv_r / dist_sq, 0.0)

    weighted = inv_dist3 * source_masses[:, None, :]
    accels = -G * jnp.sum(weighted[..., None] * diff, axis=2)
    accels = jnp.where(target_mask[..., None], accels, 0.0)

    if compute_potential:
        potentials = -G * jnp.sum(inv_r * source_masses[:, None, :], axis=2)
        potentials = jnp.where(target_mask, potentials, 0.0)
        return accels, potentials

    return accels, None


def _scatter_contributions(
    base_acc: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    if values.size == 0:
        return base_acc
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)
    masked_values = jnp.where(flat_mask[:, None], flat_values, 0.0)
    return base_acc.at[flat_indices].add(masked_values)


def _scatter_contributions_two_stage(
    base_acc: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Two-stage scatter: in-batch contiguous reduction then compact add."""
    if values.size == 0:
        return base_acc
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)
    safe_indices = jnp.where(flat_mask, flat_indices, 0)
    safe_values = jnp.where(flat_mask[:, None], flat_values, 0.0)

    sort_idx = jnp.argsort(safe_indices)
    idx_sorted = safe_indices[sort_idx]
    values_sorted = safe_values[sort_idx]
    item_count = idx_sorted.shape[0]

    is_new = jnp.concatenate(
        [
            jnp.array([True]),
            idx_sorted[1:] != idx_sorted[:-1],
        ]
    )
    group_ids = jnp.cumsum(is_new.astype(INDEX_DTYPE)) - as_index(1)
    reduced = jax.ops.segment_sum(values_sorted, group_ids, item_count)
    unique_indices = jnp.zeros((item_count,), dtype=INDEX_DTYPE).at[group_ids].set(
        idx_sorted
    )
    return base_acc.at[unique_indices].add(reduced)


def _build_scatter_schedule(
    indices: Array,
    mask: Array,
) -> Tuple[Array, Array, Array]:
    """Build a reusable two-stage scatter schedule for a fixed index/mask layout."""
    flat_indices = indices.reshape(-1)
    flat_mask = mask.reshape(-1)
    safe_indices = jnp.where(flat_mask, flat_indices, 0)

    sort_idx = jnp.argsort(safe_indices)
    idx_sorted = safe_indices[sort_idx]
    item_count = idx_sorted.shape[0]
    is_new = jnp.concatenate(
        [
            jnp.array([True]),
            idx_sorted[1:] != idx_sorted[:-1],
        ]
    )
    group_ids = jnp.cumsum(is_new.astype(INDEX_DTYPE)) - as_index(1)
    unique_indices = jnp.zeros((item_count,), dtype=INDEX_DTYPE).at[group_ids].set(
        idx_sorted
    )
    return sort_idx, group_ids, unique_indices


def _scatter_vectors_with_schedule(
    base_acc: Array,
    values: Array,
    mask: Array,
    sort_idx: Array,
    group_ids: Array,
    unique_indices: Array,
) -> Array:
    """Apply a precomputed scatter schedule to vector contributions."""
    if values.size == 0:
        return base_acc
    flat_values = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)
    safe_values = jnp.where(flat_mask[:, None], flat_values, 0.0)
    values_sorted = safe_values[sort_idx]
    item_count = values_sorted.shape[0]
    reduced = jax.ops.segment_sum(values_sorted, group_ids, item_count)
    return base_acc.at[unique_indices].add(reduced)


def _scatter_scalar_contributions(
    base: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    if values.size == 0:
        return base
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1)
    flat_mask = mask.reshape(-1)
    masked = jnp.where(flat_mask, flat_values, 0.0)
    return base.at[flat_indices].add(masked)


def _scatter_scalar_contributions_two_stage(
    base: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Two-stage scalar scatter: in-batch contiguous reduction then compact add."""
    if values.size == 0:
        return base
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1)
    flat_mask = mask.reshape(-1)
    safe_indices = jnp.where(flat_mask, flat_indices, 0)
    safe_values = jnp.where(flat_mask, flat_values, 0.0)

    sort_idx = jnp.argsort(safe_indices)
    idx_sorted = safe_indices[sort_idx]
    values_sorted = safe_values[sort_idx]
    item_count = idx_sorted.shape[0]

    is_new = jnp.concatenate(
        [
            jnp.array([True]),
            idx_sorted[1:] != idx_sorted[:-1],
        ]
    )
    group_ids = jnp.cumsum(is_new.astype(INDEX_DTYPE)) - as_index(1)
    reduced = jax.ops.segment_sum(values_sorted, group_ids, item_count)
    unique_indices = jnp.zeros((item_count,), dtype=INDEX_DTYPE).at[group_ids].set(
        idx_sorted
    )
    return base.at[unique_indices].add(reduced)


def _scatter_scalars_with_schedule(
    base: Array,
    values: Array,
    mask: Array,
    sort_idx: Array,
    group_ids: Array,
    unique_indices: Array,
) -> Array:
    """Apply a precomputed scatter schedule to scalar contributions."""
    if values.size == 0:
        return base
    flat_values = values.reshape(-1)
    flat_mask = mask.reshape(-1)
    safe_values = jnp.where(flat_mask, flat_values, 0.0)
    values_sorted = safe_values[sort_idx]
    item_count = values_sorted.shape[0]
    reduced = jax.ops.segment_sum(values_sorted, group_ids, item_count)
    return base.at[unique_indices].add(reduced)


@partial(
    jax.jit,
    static_argnums=(12,),
    static_argnames=(
        "return_potential",
        "collect_neighbor_pairs",
        "nearfield_mode",
        "edge_chunk_size",
        "use_precomputed_scatter",
    ),
)
def _compute_leaf_p2p_impl(
    node_ranges: Array,
    leaf_nodes: Array,
    offsets: Array,
    neighbors: Array,
    positions: Array,
    masses: Array,
    target_leaf_ids: Array,
    source_leaf_ids: Array,
    valid_pairs: Array,
    precomputed_chunk_sort_indices: Array,
    precomputed_chunk_group_ids: Array,
    precomputed_chunk_unique_indices: Array,
    max_leaf_size: int,
    *,
    G: float,
    softening_sq: Array,
    return_potential: bool,
    collect_neighbor_pairs: bool,
    nearfield_mode: str,
    edge_chunk_size: int,
    use_precomputed_scatter: bool,
) -> Union[
    Array,
    Tuple[Array, Array],
    Tuple[Array, Array, Array],
    Tuple[Array, Array, Array, Array],
]:
    (
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
    ) = _prepare_leaf_data(
        node_ranges,
        leaf_nodes,
        positions,
        masses,
        max_leaf_size=max_leaf_size,
    )

    dtype = positions.dtype
    g_const = jnp.asarray(G, dtype=dtype)
    soft_sq = softening_sq

    accelerations = jnp.zeros_like(positions)
    if return_potential:
        potentials = jnp.zeros((positions.shape[0],), dtype=dtype)
    else:
        potentials = None

    # Self interactions within each leaf
    self_accel, self_potential = _self_contributions(
        leaf_positions,
        leaf_masses,
        leaf_mask,
        softening_sq=soft_sq,
        G=g_const,
        compute_potential=return_potential,
    )
    accelerations = _scatter_contributions(
        accelerations,
        leaf_particle_idx,
        self_accel,
        leaf_mask,
    )
    if return_potential and self_potential is not None and potentials is not None:
        potentials = _scatter_scalar_contributions(
            potentials,
            leaf_particle_idx,
            self_potential,
            leaf_mask,
        )

    inputs = (target_leaf_ids, source_leaf_ids, valid_pairs)

    if neighbors.shape[0] > 0:
        mode = str(nearfield_mode).strip().lower()
        if mode not in ("baseline", "bucketed"):
            raise ValueError("nearfield_mode must be 'baseline' or 'bucketed'")
        if mode == "bucketed":
            chunk = int(edge_chunk_size)
            if chunk <= 0:
                raise ValueError("edge_chunk_size must be positive")
            starts = jnp.arange(0, neighbors.shape[0], chunk, dtype=INDEX_DTYPE)
            chunk_offsets = jnp.arange(chunk, dtype=INDEX_DTYPE)
            chunk_flat_size = int(chunk * max_leaf_size)

            if return_potential and potentials is not None:
                if use_precomputed_scatter:

                    def _chunk_body(carry, data):
                        acc, pot = carry
                        start, sort_idx, group_ids, unique_indices = data
                        edge_idx = start + chunk_offsets
                        in_range = edge_idx < neighbors.shape[0]
                        safe_edge_idx = jnp.where(in_range, edge_idx, 0)
                        valid_edge = in_range & valid_pairs[safe_edge_idx]

                        tgt_leaf = target_leaf_ids[safe_edge_idx]
                        src_leaf = source_leaf_ids[safe_edge_idx]
                        tgt_leaf = jnp.where(valid_edge, tgt_leaf, 0)
                        src_leaf = jnp.where(valid_edge, src_leaf, 0)

                        tgt_pos = leaf_positions[tgt_leaf]
                        tgt_mask = leaf_mask[tgt_leaf] & valid_edge[:, None]
                        src_pos = leaf_positions[src_leaf]
                        src_mass = leaf_masses[src_leaf]
                        src_mask = leaf_mask[src_leaf] & valid_edge[:, None]

                        pair_acc, pair_pot = _pair_contributions_batched(
                            tgt_pos,
                            tgt_mask,
                            src_pos,
                            src_mass,
                            src_mask,
                            softening_sq=soft_sq,
                            G=g_const,
                            compute_potential=True,
                        )
                        acc = _scatter_vectors_with_schedule(
                            acc,
                            pair_acc,
                            tgt_mask,
                            sort_idx,
                            group_ids,
                            unique_indices,
                        )
                        pot = _scatter_scalars_with_schedule(  # type: ignore[arg-type]
                            pot,
                            pair_pot,
                            tgt_mask,
                            sort_idx,
                            group_ids,
                            unique_indices,
                        )
                        return (acc, pot), None

                    (accelerations, potentials), _ = lax.scan(
                        _chunk_body,
                        (accelerations, potentials),
                        (
                            starts,
                            precomputed_chunk_sort_indices[:, :chunk_flat_size],
                            precomputed_chunk_group_ids[:, :chunk_flat_size],
                            precomputed_chunk_unique_indices[:, :chunk_flat_size],
                        ),
                    )
                else:

                    def _chunk_body(carry, start):
                        acc, pot = carry
                        edge_idx = start + chunk_offsets
                        in_range = edge_idx < neighbors.shape[0]
                        safe_edge_idx = jnp.where(in_range, edge_idx, 0)
                        valid_edge = in_range & valid_pairs[safe_edge_idx]

                        tgt_leaf = target_leaf_ids[safe_edge_idx]
                        src_leaf = source_leaf_ids[safe_edge_idx]
                        tgt_leaf = jnp.where(valid_edge, tgt_leaf, 0)
                        src_leaf = jnp.where(valid_edge, src_leaf, 0)

                        tgt_pos = leaf_positions[tgt_leaf]
                        tgt_mask = leaf_mask[tgt_leaf] & valid_edge[:, None]
                        tgt_ids = leaf_particle_idx[tgt_leaf]
                        src_pos = leaf_positions[src_leaf]
                        src_mass = leaf_masses[src_leaf]
                        src_mask = leaf_mask[src_leaf] & valid_edge[:, None]
                        sort_idx, group_ids, unique_indices = _build_scatter_schedule(
                            tgt_ids,
                            tgt_mask,
                        )

                        pair_acc, pair_pot = _pair_contributions_batched(
                            tgt_pos,
                            tgt_mask,
                            src_pos,
                            src_mass,
                            src_mask,
                            softening_sq=soft_sq,
                            G=g_const,
                            compute_potential=True,
                        )
                        acc = _scatter_vectors_with_schedule(
                            acc,
                            pair_acc,
                            tgt_mask,
                            sort_idx,
                            group_ids,
                            unique_indices,
                        )
                        pot = _scatter_scalars_with_schedule(  # type: ignore[arg-type]
                            pot,
                            pair_pot,
                            tgt_mask,
                            sort_idx,
                            group_ids,
                            unique_indices,
                        )
                        return (acc, pot), None

                    (accelerations, potentials), _ = lax.scan(
                        _chunk_body,
                        (accelerations, potentials),
                        starts,
                    )
            else:
                if use_precomputed_scatter:

                    def _chunk_body(acc, data):
                        start, sort_idx, group_ids, unique_indices = data
                        edge_idx = start + chunk_offsets
                        in_range = edge_idx < neighbors.shape[0]
                        safe_edge_idx = jnp.where(in_range, edge_idx, 0)
                        valid_edge = in_range & valid_pairs[safe_edge_idx]

                        tgt_leaf = target_leaf_ids[safe_edge_idx]
                        src_leaf = source_leaf_ids[safe_edge_idx]
                        tgt_leaf = jnp.where(valid_edge, tgt_leaf, 0)
                        src_leaf = jnp.where(valid_edge, src_leaf, 0)

                        tgt_pos = leaf_positions[tgt_leaf]
                        tgt_mask = leaf_mask[tgt_leaf] & valid_edge[:, None]
                        src_pos = leaf_positions[src_leaf]
                        src_mass = leaf_masses[src_leaf]
                        src_mask = leaf_mask[src_leaf] & valid_edge[:, None]

                        pair_acc, _ = _pair_contributions_batched(
                            tgt_pos,
                            tgt_mask,
                            src_pos,
                            src_mass,
                            src_mask,
                            softening_sq=soft_sq,
                            G=g_const,
                            compute_potential=False,
                        )
                        acc = _scatter_vectors_with_schedule(
                            acc,
                            pair_acc,
                            tgt_mask,
                            sort_idx,
                            group_ids,
                            unique_indices,
                        )
                        return acc, None

                    accelerations, _ = lax.scan(
                        _chunk_body,
                        accelerations,
                        (
                            starts,
                            precomputed_chunk_sort_indices[:, :chunk_flat_size],
                            precomputed_chunk_group_ids[:, :chunk_flat_size],
                            precomputed_chunk_unique_indices[:, :chunk_flat_size],
                        ),
                    )
                else:

                    def _chunk_body(acc, start):
                        edge_idx = start + chunk_offsets
                        in_range = edge_idx < neighbors.shape[0]
                        safe_edge_idx = jnp.where(in_range, edge_idx, 0)
                        valid_edge = in_range & valid_pairs[safe_edge_idx]

                        tgt_leaf = target_leaf_ids[safe_edge_idx]
                        src_leaf = source_leaf_ids[safe_edge_idx]
                        tgt_leaf = jnp.where(valid_edge, tgt_leaf, 0)
                        src_leaf = jnp.where(valid_edge, src_leaf, 0)

                        tgt_pos = leaf_positions[tgt_leaf]
                        tgt_mask = leaf_mask[tgt_leaf] & valid_edge[:, None]
                        tgt_ids = leaf_particle_idx[tgt_leaf]
                        src_pos = leaf_positions[src_leaf]
                        src_mass = leaf_masses[src_leaf]
                        src_mask = leaf_mask[src_leaf] & valid_edge[:, None]
                        sort_idx, group_ids, unique_indices = _build_scatter_schedule(
                            tgt_ids,
                            tgt_mask,
                        )

                        pair_acc, _ = _pair_contributions_batched(
                            tgt_pos,
                            tgt_mask,
                            src_pos,
                            src_mass,
                            src_mask,
                            softening_sq=soft_sq,
                            G=g_const,
                            compute_potential=False,
                        )
                        acc = _scatter_vectors_with_schedule(
                            acc,
                            pair_acc,
                            tgt_mask,
                            sort_idx,
                            group_ids,
                            unique_indices,
                        )
                        return acc, None

                    accelerations, _ = lax.scan(
                        _chunk_body,
                        accelerations,
                        starts,
                    )
        elif return_potential and potentials is not None:

            def _edge_body(carry, data):
                acc, pot = carry
                tgt_idx, src_idx, is_valid = data

                def true_branch(args):
                    acc_state, pot_state, tgt, src = args
                    target_pos = leaf_positions[tgt]
                    target_mask = leaf_mask[tgt]
                    target_ids = leaf_particle_idx[tgt]

                    source_pos = leaf_positions[src]
                    source_mass = leaf_masses[src]
                    source_mask = leaf_mask[src]

                    pair_accel, pair_pot = _pair_contributions(
                        target_pos,
                        target_mask,
                        source_pos,
                        source_mass,
                        source_mask,
                        softening_sq=soft_sq,
                        G=g_const,
                        compute_potential=True,
                    )

                    masked_acc = jnp.where(
                        target_mask[:, None],
                        pair_accel,
                        0.0,
                    )
                    masked_pot = jnp.where(target_mask, pair_pot, 0.0)

                    acc_state = acc_state.at[target_ids].add(masked_acc)
                    pot_state = pot_state.at[target_ids].add(masked_pot)
                    return acc_state, pot_state

                def false_branch(args):
                    acc_state, pot_state, *_ = args
                    return acc_state, pot_state

                updated = lax.cond(
                    is_valid,
                    true_branch,
                    false_branch,
                    (acc, pot, tgt_idx, src_idx),
                )
                return updated, None

            (accelerations, potentials), _ = lax.scan(
                _edge_body,
                (accelerations, potentials),
                inputs,
            )
        else:

            def _edge_body(acc, data):
                tgt_idx, src_idx, is_valid = data

                def true_branch(args):
                    acc_state, tgt, src = args
                    target_pos = leaf_positions[tgt]
                    target_mask = leaf_mask[tgt]
                    target_ids = leaf_particle_idx[tgt]

                    source_pos = leaf_positions[src]
                    source_mass = leaf_masses[src]
                    source_mask = leaf_mask[src]

                    pair_accel, _ = _pair_contributions(
                        target_pos,
                        target_mask,
                        source_pos,
                        source_mass,
                        source_mask,
                        softening_sq=soft_sq,
                        G=g_const,
                        compute_potential=False,
                    )

                    masked_acc = jnp.where(
                        target_mask[:, None],
                        pair_accel,
                        0.0,
                    )
                    acc_state = acc_state.at[target_ids].add(masked_acc)
                    return acc_state

                def false_branch(args):
                    acc_state, *_ = args
                    return acc_state

                updated_acc = lax.cond(
                    is_valid,
                    true_branch,
                    false_branch,
                    (acc, tgt_idx, src_idx),
                )
                return updated_acc, None

            accelerations, _ = lax.scan(
                _edge_body,
                accelerations,
                inputs,
            )

    neighbor_pairs = jnp.zeros((0, 2), dtype=INDEX_DTYPE)
    pair_count = as_index(0)
    if collect_neighbor_pairs:
        max_pairs = neighbors.shape[0]
        pair_buffer = jnp.zeros((max_pairs, 2), dtype=INDEX_DTYPE)

        def _pair_body(idx, state):
            ptr, buf = state

            def _add_pair(args):
                ptr_val, buf_val = args
                pair = jnp.stack(
                    [target_leaf_ids[idx], source_leaf_ids[idx]],
                    axis=0,
                )
                buf_val = buf_val.at[ptr_val].set(pair)
                return ptr_val + as_index(1), buf_val

            return lax.cond(
                valid_pairs[idx],
                _add_pair,
                lambda args: args,
                (ptr, buf),
            )

        pair_count, pair_buffer = lax.fori_loop(
            0,
            max_pairs,
            _pair_body,
            (as_index(0), pair_buffer),
        )
        neighbor_pairs = pair_buffer

    outputs = (accelerations,)
    if return_potential and potentials is not None:
        outputs += (potentials,)
    if collect_neighbor_pairs:
        outputs += (
            neighbor_pairs,
            pair_count,
        )

    if len(outputs) == 1:
        return outputs[0]
    return outputs


@jaxtyped(typechecker=beartype)
def compute_leaf_p2p_accelerations(
    tree: RadixTree,
    neighbor_list: NodeNeighborList,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    G: float = 1.0,
    softening: float = 0.0,
    max_leaf_size: Optional[int] = None,
    return_potential: bool = False,
    collect_neighbor_pairs: bool = False,
    nearfield_mode: str = "baseline",
    edge_chunk_size: int = 256,
    precomputed_target_leaf_ids: Optional[Array] = None,
    precomputed_source_leaf_ids: Optional[Array] = None,
    precomputed_valid_pairs: Optional[Array] = None,
    precomputed_chunk_sort_indices: Optional[Array] = None,
    precomputed_chunk_group_ids: Optional[Array] = None,
    precomputed_chunk_unique_indices: Optional[Array] = None,
) -> Union[
    Array,
    Tuple[Array, Array],
    Tuple[Array, Array, Array],
    Tuple[Array, Array, Array, Array],
]:
    """Compute near-field contributions for all leaf particle pairs.

    Parameters
    ----------
    tree, neighbor_list:
        Tree structure and precomputed neighbor metadata.
    positions_sorted, masses_sorted:
        Particle data in Morton order.
    G, softening:
        Gravitational constant and Plummer softening length.
    max_leaf_size:
        Optional static bound for per-leaf particle counts. When evaluating
        under ``jax.jit`` this must be supplied to avoid tracing-time shape
        inference.
    return_potential:
        When ``True`` also accumulate gravitational potentials in addition to
        accelerations.
    collect_neighbor_pairs:
        When ``True`` also return the (target, source) leaf pair indices
        that were processed in the near-field evaluation.

    Returns
    -------
    Array or tuple of Arrays
        Accelerations in Morton order and optionally potentials. When
        ``collect_neighbor_pairs`` is enabled the tuple is extended with the
        full (target, source) neighbor pair buffer followed by the scalar
        count of valid entries (use ``neighbor_pairs[:neighbor_pair_count]``
        to inspect only the processed pairs).
    """

    positions = jnp.asarray(positions_sorted)
    masses = jnp.asarray(masses_sorted)
    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)

    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    offsets = jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE)
    neighbors = jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE)

    if leaf_nodes.size == 0:
        zeros = jnp.zeros_like(positions)
        if return_potential:
            pot_zeros = jnp.zeros((positions.shape[0],), dtype=zeros.dtype)
            return zeros, pot_zeros
        return zeros

    if max_leaf_size is None:
        leaf_ranges = node_ranges[leaf_nodes]
        counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
        try:
            max_leaf_size = int(jnp.max(counts).item())
        except TypeError as exc:
            raise ValueError(
                "max_leaf_size must be provided when tracing or JIT-compiling"
            ) from exc

    softening_sq = jnp.asarray(float(softening) ** 2, dtype=positions.dtype)

    if (
        precomputed_target_leaf_ids is None
        or precomputed_source_leaf_ids is None
        or precomputed_valid_pairs is None
    ):
        target_leaf_ids, source_leaf_ids, valid_pairs = prepare_leaf_neighbor_pairs(
            node_ranges,
            leaf_nodes,
            offsets,
            neighbors,
            sort_by_source=not bool(collect_neighbor_pairs),
        )
    else:
        target_leaf_ids = jnp.asarray(precomputed_target_leaf_ids, dtype=INDEX_DTYPE)
        source_leaf_ids = jnp.asarray(precomputed_source_leaf_ids, dtype=INDEX_DTYPE)
        valid_pairs = jnp.asarray(precomputed_valid_pairs, dtype=bool)

    use_precomputed_scatter = (
        precomputed_chunk_sort_indices is not None
        and precomputed_chunk_group_ids is not None
        and precomputed_chunk_unique_indices is not None
    )
    if use_precomputed_scatter:
        chunk_sort_indices = jnp.asarray(precomputed_chunk_sort_indices, dtype=INDEX_DTYPE)
        chunk_group_ids = jnp.asarray(precomputed_chunk_group_ids, dtype=INDEX_DTYPE)
        chunk_unique_indices = jnp.asarray(
            precomputed_chunk_unique_indices,
            dtype=INDEX_DTYPE,
        )
    else:
        chunk_sort_indices = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        chunk_group_ids = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        chunk_unique_indices = jnp.zeros((0, 0), dtype=INDEX_DTYPE)

    return _compute_leaf_p2p_impl(
        node_ranges,
        leaf_nodes,
        offsets,
        neighbors,
        positions,
        masses,
        target_leaf_ids,
        source_leaf_ids,
        valid_pairs,
        chunk_sort_indices,
        chunk_group_ids,
        chunk_unique_indices,
        int(max_leaf_size),
        G=G,
        softening_sq=softening_sq,
        return_potential=return_potential,
        collect_neighbor_pairs=collect_neighbor_pairs,
        nearfield_mode=nearfield_mode,
        edge_chunk_size=int(edge_chunk_size),
        use_precomputed_scatter=use_precomputed_scatter,
    )
