"""Near-field evaluation helpers for the Fast Multipole Method."""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jax import lax
from jaxtyping import Array, jaxtyped
from yggdrax.dtypes import INDEX_DTYPE, as_index
from yggdrax.interactions import NodeNeighborList
from yggdrax.tree import Tree

from jaccpot._env import env_flag, env_float, env_int
from jaccpot.runtime.grad_options import (  # noqa: F401
    LeafPairReverseOptions,
    analytic_p2p_vjp_enabled,
)

# Several of these are unused *in this module* -- some were already, and four more
# became so when the radix fast lane moved to `_fast_lane.py` (Tier 1.4). They are
# kept because they are a re-export surface, not dead code (F16): consumers reach
# them both by `from ...near_field import X` (`tests/unit/test_custom_vjp_parity.py`,
# `bench/audit_nearfield_padding.py`) and as attributes of the module
# (`nf._leafpair_accel_analytic_vjp` in `tests/unit/test_nearfield_fastlane_grad_path.py`).
# Removing them keeps every import in the package working and breaks callers at
# attribute-access time instead, which is how F16 was originally mis-measured. Do
# not let a pyflakes sweep take them.
from .grad import (  # noqa: F401
    _check_float_id_range,
    _leafpair_accel_analytic_vjp,
    _leafpair_reverse_tiers_cached,
    _pair_accel_cvjp,
    _pair_accel_masked_accels,
    build_leafpair_reverse_tiers,
    clear_leafpair_reverse_tier_cache,
)

_LARGE_N_NEARFIELD_DIAG_MODES = frozenset(("full", "self_only", "pairs_only", "zero"))


def _large_n_nearfield_diag_mode() -> str:
    mode = (
        str(os.environ.get("JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE", "full"))
        .strip()
        .lower()
    )
    return mode if mode in _LARGE_N_NEARFIELD_DIAG_MODES else "full"


# Unclamped by design: several near-field knobs use 0 as "unset, pick a default"
# (JACCPOT_NEARFIELD_PALLAS_NUM_WARPS, ..._TARGET_SUBTILE), so a minimum of 1
# would turn "auto" into a real, wrong value.
_env_flag = env_flag
_env_int = env_int


@dataclass(frozen=True)
class RadixFastLanePerfCounters:
    """Static-shape payload counters for radix fast-lane nearfield diagnostics."""

    gather_bytes: int
    scatter_bytes: int
    scatter_ops: int
    target_batches: int
    source_slot_tiles: int


def collect_radix_fast_lane_counters(
    *,
    payload: Any,
    positions_dtype: jnp.dtype,
    masses_dtype: jnp.dtype,
    accelerations_dtype: Optional[jnp.dtype] = None,
) -> RadixFastLanePerfCounters:
    """Estimate deterministic payload gather/scatter costs for one evaluation."""

    if accelerations_dtype is None:
        accelerations_dtype = positions_dtype

    target_particle_ids = jnp.asarray(payload.target_particle_ids, dtype=INDEX_DTYPE)
    source_particle_ids = jnp.asarray(payload.source_particle_ids, dtype=INDEX_DTYPE)

    target_slot_count = int(target_particle_ids.size)
    source_slot_count = int(source_particle_ids.size)

    pos_itemsize = int(jnp.dtype(positions_dtype).itemsize)
    mass_itemsize = int(jnp.dtype(masses_dtype).itemsize)
    accel_itemsize = int(jnp.dtype(accelerations_dtype).itemsize)

    gather_bytes = int(
        target_slot_count * (3 * pos_itemsize + mass_itemsize)
        + source_slot_count * (3 * pos_itemsize + mass_itemsize)
    )
    scatter_bytes = int(target_slot_count * 3 * accel_itemsize)
    scatter_ops = int(target_slot_count)

    num_target_leaves = (
        int(target_particle_ids.shape[0]) if target_particle_ids.ndim >= 1 else 0
    )
    num_source_slots = (
        int(source_particle_ids.shape[1]) if source_particle_ids.ndim >= 2 else 0
    )
    target_batch_size = max(1, int(getattr(payload, "batch_tile_t", 1)))
    source_slot_tile_size = max(1, int(getattr(payload, "batch_tile_s", 1)))
    target_batches = (num_target_leaves + target_batch_size - 1) // target_batch_size
    source_slot_tiles = (
        num_source_slots + source_slot_tile_size - 1
    ) // source_slot_tile_size

    return RadixFastLanePerfCounters(
        gather_bytes=gather_bytes,
        scatter_bytes=scatter_bytes,
        scatter_ops=scatter_ops,
        target_batches=int(target_batches),
        source_slot_tiles=int(source_slot_tiles),
    )


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


def prepare_bucketed_scatter_schedules_from_groups(
    leaf_particle_indices: Array,
    leaf_particle_mask: Array,
    target_leaf_ids: Array,
    valid_pairs: Array,
    *,
    edge_chunk_size: int,
) -> Tuple[Array, Array, Array]:
    """Precompute per-chunk scatter schedules for explicit leaf-particle groups."""
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


def _prepare_leaf_data_from_groups(
    leaf_particle_indices: Array,
    leaf_particle_mask: Array,
    positions: Array,
    masses: Array,
) -> Tuple[Array, Array, Array, Array]:
    """Gather per-leaf particle data from explicit particle-membership groups."""
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


def _self_contributions(
    leaf_positions: Array,
    leaf_masses: Array,
    mask: Array,
    *,
    softening_sq: Union[float, Array],
    G: Array,
    compute_potential: bool,
) -> Tuple[Array, Optional[Array]]:
    """Compute intra-leaf particle-particle contributions."""
    dtype = leaf_positions.dtype
    leaf_size = leaf_positions.shape[1]
    identity = jnp.eye(leaf_size, dtype=bool)

    def compute_single(args: tuple[Array, Array, Array]) -> tuple[Array, Array]:
        positions_leaf, masses_leaf, mask_leaf = args
        diff = positions_leaf[:, None, :] - positions_leaf[None, :, :]
        dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq

        pair_mask = mask_leaf[:, None] & mask_leaf[None, :] & (~identity)
        safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
        inv_r = lax.rsqrt(safe_dist_sq)
        inv_r = jnp.where(pair_mask, inv_r, 0.0)
        inv_dist3 = jnp.where(pair_mask, inv_r * inv_r * inv_r, 0.0)

        weighted = inv_dist3[:, :, None] * masses_leaf[None, :, None]
        accel_leaf = -G * jnp.sum(weighted * diff, axis=1)
        accel_leaf = jnp.where(mask_leaf[:, None], accel_leaf, 0.0)

        if compute_potential:
            pot_leaf = -G * jnp.sum(inv_r * masses_leaf[None, :], axis=1)
            pot_leaf = jnp.where(mask_leaf, pot_leaf, 0.0)
        else:
            pot_leaf = jnp.zeros((leaf_size,), dtype=dtype)

        return accel_leaf, pot_leaf

    # Rematerialize the per-leaf block. ``compute_single`` builds (W, W, 3) and
    # (W, W) intermediates, and the scan retains them for EVERY leaf, so the
    # residual is O(leaves * W^2) -- N*W in practice, i.e. ~1.4 GB at N=200000 and
    # ~7.5 GB at N=1048576 on the canonical leaf-256 config. Its inputs are only
    # the (W, 3) / (W,) slices of one leaf, so remat trades one extra intra-leaf
    # pass for a W-fold reduction.
    #
    # Note this term is NOT covered by ``_pair_accel_cvjp``: that rule handles
    # cross-leaf pair blocks, while intra-leaf self interaction is computed here.
    _compute_single_remat = jax.checkpoint(compute_single)

    def scan_step(
        carry: Any, args: tuple[Array, Array, Array]
    ) -> tuple[Any, tuple[Array, Array]]:
        accel_leaf, pot_leaf = _compute_single_remat(args)
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
    softening_sq: Union[float, Array],
    G: Array,
    compute_potential: bool,
) -> Tuple[Array, Optional[Array]]:
    """Compute one target-leaf vs source-leaf contribution block."""
    dtype = target_positions.dtype

    source_pos = source_positions
    source_mass = source_masses
    source_active = source_mask
    mass_effective = jnp.where(source_active, source_mass, 0.0)

    soft = softening_sq

    def when_valid(pos: Array) -> tuple[Array, Array]:
        diff = pos - source_pos
        dist_sq = jnp.sum(diff * diff, axis=1) + soft
        mask_src = source_active

        safe_dist_sq = jnp.where(mask_src, dist_sq, jnp.ones_like(dist_sq))
        inv_r = lax.rsqrt(safe_dist_sq)
        inv_r = jnp.where(mask_src, inv_r, 0.0)
        inv_dist3 = jnp.where(mask_src, inv_r * inv_r * inv_r, 0.0)

        weighted = inv_dist3[:, None] * mass_effective[:, None]
        accel = -G * jnp.sum(weighted * diff, axis=0)

        if compute_potential:
            pot = -G * jnp.sum(inv_r * mass_effective)
        else:
            pot = jnp.zeros((), dtype=dtype)

        return accel, pot

    def scan_step(
        carry: Any, data: tuple[Array, Array]
    ) -> tuple[Any, tuple[Array, Array]]:
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
    softening_sq: Union[float, Array],
    G: Array,
    compute_potential: bool,
) -> Tuple[Array, Optional[Array]]:
    """Vectorized pair contributions for a batch of target/source leaf pairs."""
    if analytic_p2p_vjp_enabled() and not compute_potential:
        # Accel-only path (the differentiable path): route through the analytic
        # symmetric-tidal-tensor custom_vjp. Forward is byte-identical; masks are
        # passed as 0/1 floats and softening/G as arrays so the reverse needs no
        # closure over tracers.
        dtype = target_positions.dtype
        accels = _pair_accel_cvjp(
            target_positions,
            source_positions,
            source_masses,
            target_mask.astype(dtype),
            source_mask.astype(dtype),
            jnp.asarray(softening_sq, dtype=dtype),
            jnp.asarray(G, dtype=dtype),
        )
        return accels, None
    diff = target_positions[:, :, None, :] - source_positions[:, None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
    pair_mask = target_mask[:, :, None] & source_mask[:, None, :]

    safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
    inv_r = lax.rsqrt(safe_dist_sq)
    inv_r = jnp.where(pair_mask, inv_r, 0.0)
    inv_dist3 = jnp.where(pair_mask, inv_r * inv_r * inv_r, 0.0)

    weighted = inv_dist3 * source_masses[:, None, :]
    accels = -G * jnp.sum(weighted[..., None] * diff, axis=2)
    accels = jnp.where(target_mask[..., None], accels, 0.0)

    if compute_potential:
        potentials = -G * jnp.sum(inv_r * source_masses[:, None, :], axis=2)
        potentials = jnp.where(target_mask, potentials, 0.0)
        return accels, potentials

    return accels, None


@partial(jax.jit, static_argnames=("compute_potential",))
def _pair_contributions_batched_componentwise(
    target_positions: Array,
    target_mask: Array,
    source_positions: Array,
    source_masses: Array,
    source_mask: Array,
    *,
    softening_sq: Union[float, Array],
    G: Array,
    compute_potential: bool,
) -> Tuple[Array, Optional[Array]]:
    """Vectorized pair contributions with explicit Cartesian components."""
    dx = target_positions[:, :, None, 0] - source_positions[:, None, :, 0]
    dy = target_positions[:, :, None, 1] - source_positions[:, None, :, 1]
    dz = target_positions[:, :, None, 2] - source_positions[:, None, :, 2]
    dist_sq = dx * dx + dy * dy + dz * dz + softening_sq
    pair_mask = target_mask[:, :, None] & source_mask[:, None, :]

    safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
    inv_r = jnp.where(pair_mask, lax.rsqrt(safe_dist_sq), 0.0)
    weighted = inv_r * inv_r * inv_r * source_masses[:, None, :]
    accel_x = -G * jnp.sum(weighted * dx, axis=2)
    accel_y = -G * jnp.sum(weighted * dy, axis=2)
    accel_z = -G * jnp.sum(weighted * dz, axis=2)
    accels = jnp.stack((accel_x, accel_y, accel_z), axis=-1)
    accels = jnp.where(target_mask[..., None], accels, 0.0)

    if compute_potential:
        potentials = -G * jnp.sum(inv_r * source_masses[:, None, :], axis=2)
        potentials = jnp.where(target_mask, potentials, 0.0)
        return accels, potentials

    return accels, None


def _bucketed_chunk_pair_accels(
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    target_leaf_local: Array,
    source_leaf_local: Array,
    valid_edge: Array,
    softening_sq: Array,
    G: Array,
) -> Tuple[Array, Array]:
    """Gather one edge chunk's leaf tensors and evaluate its near-field pair block.

    Deliberately takes the **leaf-major buffers plus leaf-id index vectors** rather
    than pre-gathered positions, so callers can wrap it in ``jax.checkpoint``:
    ``lax.scan``'s partial-eval hoists the scan-invariant leaf buffers out and
    counts them once, leaving only two integer leaf-id vectors and a mask stacked
    per chunk.

    That is the single largest reverse-pass term at galaxy scale. The gather sits
    *outside* :func:`_pair_accel_cvjp`, which explicitly saves its own inputs, so
    rematerializing only the gather would achieve nothing -- the consumer would
    save the gather's outputs anyway. Rematerializing the composite (gather **and**
    pair evaluation) is what collapses it: measured **77.8 B per (edge x
    max_leaf_size)** by ``bench/audit_reverse_residuals.py``, i.e. 8.7 GB at
    N=200000 and 124 GB at N=1048576 on the canonical leaf-256 config, versus
    ~14 B per *edge* once rematerialized.

    Returns ``(pair_accelerations, target_mask)``; the caller applies the scatter,
    which is linear and therefore needs only indices and the mask in reverse.
    """
    target_positions = leaf_positions[target_leaf_local]
    target_mask = leaf_mask[target_leaf_local] & valid_edge[:, None]
    source_positions = leaf_positions[source_leaf_local]
    source_masses = leaf_masses[source_leaf_local]
    source_mask = leaf_mask[source_leaf_local] & valid_edge[:, None]
    pair_acc, _ = _pair_contributions_batched(
        target_positions,
        target_mask,
        source_positions,
        source_masses,
        source_mask,
        softening_sq=softening_sq,
        G=G,
        compute_potential=False,
    )
    return pair_acc, target_mask


_bucketed_chunk_pair_accels_remat = jax.checkpoint(_bucketed_chunk_pair_accels)


def _scatter_contributions(
    base_acc: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Scatter-add vector contributions into particle-ordered output."""
    if values.size == 0:
        return base_acc
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)
    masked_values = jnp.where(flat_mask[:, None], flat_values, 0.0)
    return base_acc.at[flat_indices].add(masked_values)


def _scatter_contributions_sorted_hint(
    base_acc: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Scatter-add vector contributions assuming index order is nondecreasing."""
    if values.size == 0:
        return base_acc
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)
    masked_values = jnp.where(flat_mask[:, None], flat_values, 0.0)
    return base_acc.at[flat_indices].add(masked_values, indices_are_sorted=True)


def _scatter_contributions_grouped_sorted(
    base_acc: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Scatter-add for sorted indices with grouped reduction before add."""
    if values.size == 0:
        return base_acc
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)
    masked_values = jnp.where(flat_mask[:, None], flat_values, 0.0)

    item_count = flat_indices.shape[0]
    is_new = jnp.concatenate(
        [
            jnp.array([True], dtype=bool),
            flat_indices[1:] != flat_indices[:-1],
        ]
    )
    group_ids = jnp.cumsum(is_new.astype(INDEX_DTYPE)) - as_index(1)
    reduced = jax.ops.segment_sum(masked_values, group_ids, item_count)
    unique_indices = (
        jnp.zeros((item_count,), dtype=INDEX_DTYPE).at[group_ids].set(flat_indices)
    )
    return base_acc.at[unique_indices].add(reduced, indices_are_sorted=True)


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
    unique_indices = (
        jnp.zeros((item_count,), dtype=INDEX_DTYPE).at[group_ids].set(idx_sorted)
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
    """Scatter-add scalar contributions into particle-ordered output."""
    if values.size == 0:
        return base
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1)
    flat_mask = mask.reshape(-1)
    masked = jnp.where(flat_mask, flat_values, 0.0)
    return base.at[flat_indices].add(masked)


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


def _reduce_pair_bucket_by_target_leaf(
    target_leaf_ids: Array,
    valid_edge: Array,
    pair_acc: Array,
) -> Tuple[Array, Array, Array]:
    """Collapse target-local bucket rows before direct scatter.

    The minimum-memory large-N path preserves neighbor-list edge order, so
    edges for the same target leaf arrive in contiguous runs. We can exploit
    that to reduce repeated target-leaf updates inside a chunk without
    rebuilding a full per-particle scatter schedule.
    """

    chunk = int(target_leaf_ids.shape[0])
    if chunk == 0:
        empty_leaf = jnp.zeros((0,), dtype=INDEX_DTYPE)
        empty_pair = jnp.zeros_like(pair_acc)
        empty_valid = jnp.zeros((0,), dtype=bool)
        return empty_leaf, empty_pair, empty_valid

    invalid_leaf = jnp.asarray(-1, dtype=INDEX_DTYPE)
    grouped_leaf_ids = jnp.where(valid_edge, target_leaf_ids, invalid_leaf)
    is_new_group = jnp.concatenate(
        [
            jnp.array([True], dtype=bool),
            grouped_leaf_ids[1:] != grouped_leaf_ids[:-1],
        ]
    )
    group_ids = jnp.cumsum(is_new_group.astype(INDEX_DTYPE)) - as_index(1)
    masked_pair_acc = jnp.where(valid_edge[:, None, None], pair_acc, 0.0)
    reduced_pair_acc = jax.ops.segment_sum(
        masked_pair_acc,
        group_ids,
        chunk,
    )
    reduced_target_leaf_ids = (
        jnp.zeros((chunk,), dtype=INDEX_DTYPE)
        .at[group_ids]
        .set(jnp.where(valid_edge, target_leaf_ids, 0))
    )
    reduced_valid = jnp.zeros((chunk,), dtype=bool).at[group_ids].set(valid_edge)
    return reduced_target_leaf_ids, reduced_pair_acc, reduced_valid


@jax.jit
def _compute_leaf_p2p_prepared_large_n_self_only_impl(
    positions: Array,
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    *,
    G: Union[float, Array],
    softening_sq: Array,
) -> Array:
    """Self-leaf portion of the specialized large-N accel-only kernel."""
    dtype = positions.dtype
    g_const = jnp.asarray(G, dtype=dtype)
    accelerations = jnp.zeros_like(positions)
    self_accel, _ = _self_contributions(
        leaf_positions,
        leaf_masses,
        leaf_mask,
        softening_sq=softening_sq,
        G=g_const,
        compute_potential=False,
    )
    return _scatter_contributions(
        accelerations,
        leaf_particle_idx,
        self_accel,
        leaf_mask,
    )


@partial(
    jax.jit,
    static_argnames=(
        "edge_chunk_size",
        "chunks_per_superchunk",
        "chunk_scan_batch_size",
        "chunk_scan_unroll",
        "superchunk_scan_unroll",
        "sorted_scatter_hint",
        "grouped_sorted_scatter",
        "superchunk_target_reduce",
        "disable_chunk_cond",
    ),
)
def _compute_leaf_p2p_prepared_large_n_pairs_only_impl(
    positions: Array,
    target_leaf_ids: Array,
    source_leaf_ids: Array,
    valid_pairs: Array,
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    *,
    G: Union[float, Array],
    softening_sq: Array,
    edge_chunk_size: int,
    chunks_per_superchunk: int,
    chunk_scan_batch_size: int = 1,
    chunk_scan_unroll: int = 1,
    superchunk_scan_unroll: int = 1,
    sorted_scatter_hint: bool,
    grouped_sorted_scatter: bool,
    superchunk_target_reduce: bool,
    disable_chunk_cond: bool,
) -> Array:
    """Cross-leaf pair-bucket portion of the specialized large-N kernel."""
    dtype = positions.dtype
    g_const = jnp.asarray(G, dtype=dtype)
    accelerations = jnp.zeros_like(positions)
    edge_count = target_leaf_ids.shape[0]
    if edge_count == 0:
        return accelerations

    chunk = int(edge_chunk_size)
    if chunk <= 0:
        raise ValueError("edge_chunk_size must be positive")
    superchunk = int(chunks_per_superchunk)
    if superchunk <= 0:
        raise ValueError("chunks_per_superchunk must be positive")
    scan_batch = int(chunk_scan_batch_size)
    if scan_batch <= 0:
        raise ValueError("chunk_scan_batch_size must be positive")
    chunk_unroll = int(chunk_scan_unroll)
    if chunk_unroll <= 0:
        raise ValueError("chunk_scan_unroll must be positive")
    super_unroll = int(superchunk_scan_unroll)
    if super_unroll <= 0:
        raise ValueError("superchunk_scan_unroll must be positive")

    chunk_offsets = jnp.arange(chunk, dtype=INDEX_DTYPE)
    starts = jnp.arange(0, edge_count, chunk, dtype=INDEX_DTYPE)

    def _chunk_probe_from_start(
        start: Array, active: Array
    ) -> tuple[Array, Array, Array, Array]:
        edge_idx = start + chunk_offsets
        in_range = active & (edge_idx < edge_count)
        safe_edge_idx = jnp.where(in_range, edge_idx, 0)
        valid_edge = in_range & valid_pairs[safe_edge_idx]

        tgt_leaf = target_leaf_ids[safe_edge_idx]
        src_leaf = source_leaf_ids[safe_edge_idx]
        tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
        src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

        tgt_pos = leaf_positions[tgt_leaf_local]
        tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
        src_pos = leaf_positions[src_leaf_local]
        src_mass = leaf_masses[src_leaf_local]
        src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

        pair_acc, _ = _pair_contributions_batched(
            tgt_pos,
            tgt_mask,
            src_pos,
            src_mass,
            src_mask,
            softening_sq=softening_sq,
            G=g_const,
            compute_potential=False,
        )
        reduced_tgt_leaf_local, reduced_pair_acc, reduced_valid = (
            _reduce_pair_bucket_by_target_leaf(
                tgt_leaf_local,
                valid_edge,
                pair_acc,
            )
        )
        reduced_tgt_ids = leaf_particle_idx[reduced_tgt_leaf_local]
        reduced_tgt_mask = leaf_mask[reduced_tgt_leaf_local] & reduced_valid[:, None]
        return (
            reduced_tgt_leaf_local,
            reduced_tgt_ids,
            reduced_pair_acc,
            reduced_tgt_mask,
        )

    if superchunk == 1 and scan_batch == 1:
        if sorted_scatter_hint:
            if grouped_sorted_scatter:
                scatter_fn_single = _scatter_contributions_grouped_sorted
            else:
                scatter_fn_single = _scatter_contributions_sorted_hint
        else:
            scatter_fn_single = _scatter_contributions

        def _chunk_body(acc, start):
            _, tgt_ids, pair_acc, tgt_mask = _chunk_probe_from_start(
                start,
                jnp.array(True, dtype=bool),
            )
            if disable_chunk_cond:
                return scatter_fn_single(acc, tgt_ids, pair_acc, tgt_mask), None

            def _apply_scatter(acc_in: Array) -> Array:
                return scatter_fn_single(acc_in, tgt_ids, pair_acc, tgt_mask)

            has_valid = jnp.any(tgt_mask)
            return lax.cond(has_valid, _apply_scatter, lambda acc_in: acc_in, acc), None

        accelerations, _ = lax.scan(
            _chunk_body,
            accelerations,
            starts,
            unroll=chunk_unroll,
        )
        return accelerations

    # Batch chunk probes so we reduce scan overhead and maximize vectorized work.
    chunk_group = superchunk if superchunk > 1 else scan_batch
    super_starts = jnp.arange(0, starts.shape[0], chunk_group, dtype=INDEX_DTYPE)
    super_offsets = jnp.arange(chunk_group, dtype=INDEX_DTYPE)

    if sorted_scatter_hint:
        if grouped_sorted_scatter:
            scatter_fn = _scatter_contributions_grouped_sorted
        else:
            scatter_fn = _scatter_contributions_sorted_hint
    else:
        scatter_fn = _scatter_contributions

    def _superchunk_body(acc, super_start_idx):
        def _chunk_probe(offset_idx):
            chunk_idx = super_start_idx + offset_idx
            in_super_range = chunk_idx < starts.shape[0]
            safe_chunk_idx = jnp.where(in_super_range, chunk_idx, 0)
            start = starts[safe_chunk_idx]
            safe_start = jnp.where(in_super_range, start, 0)
            return _chunk_probe_from_start(safe_start, in_super_range)

        super_leaf, super_ids, super_values, super_mask = jax.vmap(_chunk_probe)(
            super_offsets
        )
        if superchunk_target_reduce and superchunk > 1:
            flat_valid = jnp.any(super_mask, axis=-1).reshape(-1)
            flat_tgt_leaf = super_leaf.reshape(-1)
            reduced_leaf, reduced_values, reduced_valid = (
                _reduce_pair_bucket_by_target_leaf(
                    flat_tgt_leaf,
                    flat_valid,
                    super_values.reshape(
                        -1, super_values.shape[-2], super_values.shape[-1]
                    ),
                )
            )
            reduced_ids = leaf_particle_idx[reduced_leaf]
            reduced_mask = leaf_mask[reduced_leaf] & reduced_valid[:, None]
            return (
                _scatter_contributions(
                    acc,
                    reduced_ids,
                    reduced_values,
                    reduced_mask,
                ),
                None,
            )

        flat_ids = super_ids.reshape(-1, super_ids.shape[-1])
        flat_values = super_values.reshape(
            -1,
            super_values.shape[-2],
            super_values.shape[-1],
        )
        flat_mask = super_mask.reshape(-1, super_mask.shape[-1])
        if disable_chunk_cond:
            return scatter_fn(acc, flat_ids, flat_values, flat_mask), None

        def _apply_scatter(acc_in: Array) -> Array:
            return scatter_fn(acc_in, flat_ids, flat_values, flat_mask)

        has_valid = jnp.any(flat_mask)
        return lax.cond(has_valid, _apply_scatter, lambda acc_in: acc_in, acc), None

    accelerations, _ = lax.scan(
        _superchunk_body,
        accelerations,
        super_starts,
        unroll=super_unroll,
    )
    return accelerations


def _accumulate_target_block_tile_sequence(
    target_pos: Array,
    target_mask: Array,
    tile_source_ids_seq: Array,
    tile_source_valid_seq: Array,
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    *,
    g_const: Array,
    softening_sq: Array,
    tile_unroll: int,
    skip_empty_tiles: bool = False,
    componentwise_pairs: bool = False,
) -> Array:
    """Accumulate target-leaf accelerations from fixed-shape tile sequences."""
    dtype = target_pos.dtype
    leaf_batch = int(target_pos.shape[0])
    block_tile = int(tile_source_ids_seq.shape[2])
    block_size = int(tile_source_ids_seq.shape[3])
    leaf_size = int(target_pos.shape[1])

    flat_target_pos_base = jnp.reshape(
        jnp.broadcast_to(
            target_pos[:, None, None, :, :],
            (leaf_batch, block_tile, block_size, leaf_size, 3),
        ),
        (leaf_batch * block_tile * block_size, leaf_size, 3),
    )
    flat_target_mask_base = jnp.reshape(
        jnp.broadcast_to(
            target_mask[:, None, None, :],
            (leaf_batch, block_tile, block_size, leaf_size),
        ),
        (leaf_batch * block_tile * block_size, leaf_size),
    )

    def _tile_body(local_acc, tile_data):
        tile_source_ids, tile_source_valid = tile_data

        def _apply_tile(acc_in):
            safe_src_leaf_ids = jnp.where(tile_source_valid, tile_source_ids, 0)
            src_pos = leaf_positions[safe_src_leaf_ids]
            src_mass = leaf_masses[safe_src_leaf_ids]
            src_mask = leaf_mask[safe_src_leaf_ids] & tile_source_valid[:, :, :, None]

            flat_src_pos = src_pos.reshape(
                (leaf_batch * block_tile * block_size, leaf_size, 3)
            )
            flat_src_mass = src_mass.reshape(
                (leaf_batch * block_tile * block_size, leaf_size)
            )
            flat_src_mask = src_mask.reshape(
                (leaf_batch * block_tile * block_size, leaf_size)
            )
            flat_pair_valid = tile_source_valid.reshape(
                (leaf_batch * block_tile * block_size)
            )
            flat_target_mask = flat_target_mask_base & flat_pair_valid[:, None]

            pair_reducer = (
                _pair_contributions_batched_componentwise
                if bool(componentwise_pairs)
                else _pair_contributions_batched
            )
            pair_acc, _ = pair_reducer(
                flat_target_pos_base,
                flat_target_mask,
                flat_src_pos,
                flat_src_mass,
                flat_src_mask,
                softening_sq=softening_sq,
                G=g_const,
                compute_potential=False,
            )
            tile_acc = jnp.sum(
                pair_acc.reshape((leaf_batch, block_tile, block_size, leaf_size, 3)),
                axis=(1, 2),
            )
            return acc_in + tile_acc

        if bool(skip_empty_tiles):
            local_acc = lax.cond(
                jnp.any(tile_source_valid),
                _apply_tile,
                lambda acc_in: acc_in,
                local_acc,
            )
        else:
            local_acc = _apply_tile(local_acc)
        return local_acc, None

    target_leaf_acc, _ = lax.scan(
        _tile_body,
        jnp.zeros((leaf_batch, leaf_size, 3), dtype=dtype),
        (tile_source_ids_seq, tile_source_valid_seq),
        unroll=int(tile_unroll),
    )
    return target_leaf_acc


def _collect_target_leaf_batch_acc(
    num_leaves: int,
    leaf_size: int,
    target_leaf_batch_size: int,
    batch_scan_unroll: int,
    batch_body,
) -> Array:
    """Collect fixed-shape target-leaf batch accumulations into leaf-major form."""
    leaf_batch = int(target_leaf_batch_size)
    if leaf_batch <= 0:
        raise ValueError("target_leaf_batch_size must be positive")
    scan_unroll = int(batch_scan_unroll)
    if scan_unroll <= 0:
        raise ValueError("batch_scan_unroll must be positive")

    leaf_batch_starts = jnp.arange(0, num_leaves, leaf_batch, dtype=INDEX_DTYPE)

    def _collect_batch(_, batch_start):
        return None, batch_body(batch_start)

    _, target_leaf_batch_acc = lax.scan(
        _collect_batch,
        None,
        leaf_batch_starts,
        unroll=scan_unroll,
    )
    return target_leaf_batch_acc.reshape((-1, leaf_size, 3))[:num_leaves]


def _compute_target_block_pairs_from_source_tiles(
    positions: Array,
    source_leaf_ids_tiles: Array,
    source_valid_tiles: Array,
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    *,
    g_const: Array,
    softening_sq: Array,
    target_leaf_batch_size: int,
    target_block_tile_scan_unroll: int,
    target_block_batch_scan_unroll: int,
    skip_empty_tiles: bool = False,
    componentwise_pairs: bool = False,
) -> Array:
    """Evaluate TONB pair contributions from canonical [tile, leaf, lane_block, lane] tensors."""
    num_leaves = int(leaf_positions.shape[0])
    leaf_size = int(leaf_positions.shape[1])

    if num_leaves == 0:
        return jnp.zeros_like(positions)

    leaf_batch = int(target_leaf_batch_size)
    if leaf_batch <= 0:
        raise ValueError("target_leaf_batch_size must be positive")
    tile_unroll = int(target_block_tile_scan_unroll)
    if tile_unroll <= 0:
        raise ValueError("target_block_tile_scan_unroll must be positive")
    batch_unroll = int(target_block_batch_scan_unroll)
    if batch_unroll <= 0:
        raise ValueError("target_block_batch_scan_unroll must be positive")

    leaf_batch_offsets = jnp.arange(leaf_batch, dtype=INDEX_DTYPE)

    def _batch_body(batch_start):
        target_leaf_ids = batch_start + leaf_batch_offsets
        target_active = target_leaf_ids < num_leaves
        safe_target_leaf_ids = jnp.where(target_active, target_leaf_ids, 0)

        target_pos = leaf_positions[safe_target_leaf_ids]
        target_mask = leaf_mask[safe_target_leaf_ids] & target_active[:, None]

        tile_source_ids_seq = source_leaf_ids_tiles[:, safe_target_leaf_ids, :, :]
        tile_source_valid_seq = (
            source_valid_tiles[:, safe_target_leaf_ids, :, :]
            & target_active[None, :, None, None]
        )

        target_leaf_acc = _accumulate_target_block_tile_sequence(
            target_pos,
            target_mask,
            tile_source_ids_seq,
            tile_source_valid_seq,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            g_const=g_const,
            softening_sq=softening_sq,
            tile_unroll=tile_unroll,
            skip_empty_tiles=bool(skip_empty_tiles),
            componentwise_pairs=bool(componentwise_pairs),
        )
        return jnp.where(target_active[:, None, None], target_leaf_acc, 0.0)

    acc_leaf_major = _collect_target_leaf_batch_acc(
        num_leaves,
        leaf_size,
        target_leaf_batch_size=leaf_batch,
        batch_scan_unroll=batch_unroll,
        batch_body=_batch_body,
    )

    accelerations = jnp.zeros_like(positions)
    return _scatter_contributions(
        accelerations,
        leaf_particle_idx,
        acc_leaf_major,
        leaf_mask,
    )


@partial(
    jax.jit,
    static_argnames=(
        "target_leaf_batch_size",
        "target_block_tile_size",
        "target_block_tile_scan_unroll",
        "target_block_batch_scan_unroll",
    ),
)
def _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl(
    positions: Array,
    block_offsets: Array,
    block_target_leaf_ids: Array,
    block_source_leaf_ids: Array,
    block_valid_mask: Array,
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    *,
    G: Union[float, Array],
    softening_sq: Array,
    target_leaf_batch_size: int,
    target_block_tile_size: int,
    target_block_tile_scan_unroll: int,
    target_block_batch_scan_unroll: int,
) -> Array:
    """Target-owned pair path over prepacked fixed-width source-leaf blocks."""
    del block_target_leaf_ids  # kept for API compatibility with prepared state

    dtype = positions.dtype
    g_const = jnp.asarray(G, dtype=dtype)
    num_leaves = int(leaf_positions.shape[0])
    leaf_size = int(leaf_positions.shape[1])
    num_blocks = int(block_source_leaf_ids.shape[0])
    block_size = int(block_source_leaf_ids.shape[1])

    if num_leaves == 0 or num_blocks == 0 or block_size == 0:
        return jnp.zeros_like(positions)

    leaf_batch = int(target_leaf_batch_size)
    if leaf_batch <= 0:
        raise ValueError("target_leaf_batch_size must be positive")
    block_tile = int(target_block_tile_size)
    if block_tile <= 0:
        raise ValueError("target_block_tile_size must be positive")
    tile_unroll = int(target_block_tile_scan_unroll)
    if tile_unroll <= 0:
        raise ValueError("target_block_tile_scan_unroll must be positive")
    batch_unroll = int(target_block_batch_scan_unroll)
    if batch_unroll <= 0:
        raise ValueError("target_block_batch_scan_unroll must be positive")

    leaf_batch_offsets = jnp.arange(leaf_batch, dtype=INDEX_DTYPE)
    block_tile_offsets = jnp.arange(block_tile, dtype=INDEX_DTYPE)
    max_tiles_global = (num_blocks + block_tile - 1) // block_tile
    tile_starts = jnp.arange(
        0,
        max_tiles_global * block_tile,
        block_tile,
        dtype=INDEX_DTYPE,
    )

    def _batch_body(batch_start):
        target_leaf_ids = batch_start + leaf_batch_offsets
        target_active = target_leaf_ids < num_leaves
        safe_target_leaf_ids = jnp.where(target_active, target_leaf_ids, 0)

        target_pos = leaf_positions[safe_target_leaf_ids]
        target_mask = leaf_mask[safe_target_leaf_ids] & target_active[:, None]

        block_start = block_offsets[safe_target_leaf_ids]
        block_stop = block_offsets[safe_target_leaf_ids + as_index(1)]
        block_count = jnp.where(target_active, block_stop - block_start, 0)

        local_block_idx = tile_starts[:, None, None] + block_tile_offsets[None, None, :]
        in_tile = target_active[None, :, None] & (
            local_block_idx < block_count[None, :, None]
        )
        block_idx = block_start[None, :, None] + local_block_idx
        safe_block_idx = jnp.where(in_tile, block_idx, 0)

        tile_source_ids_seq = block_source_leaf_ids[safe_block_idx]
        tile_source_valid_seq = (
            block_valid_mask[safe_block_idx] & in_tile[:, :, :, None]
        )

        target_leaf_acc = _accumulate_target_block_tile_sequence(
            target_pos,
            target_mask,
            tile_source_ids_seq,
            tile_source_valid_seq,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            g_const=g_const,
            softening_sq=softening_sq,
            tile_unroll=tile_unroll,
        )
        return jnp.where(target_active[:, None, None], target_leaf_acc, 0.0)

    acc_leaf_major = _collect_target_leaf_batch_acc(
        num_leaves,
        leaf_size,
        target_leaf_batch_size=leaf_batch,
        batch_scan_unroll=batch_unroll,
        batch_body=_batch_body,
    )

    accelerations = jnp.zeros_like(positions)
    return _scatter_contributions(
        accelerations,
        leaf_particle_idx,
        acc_leaf_major,
        leaf_mask,
    )


@partial(
    jax.jit,
    static_argnames=(
        "target_leaf_batch_size",
        "target_block_tile_size",
        "target_block_tile_scan_unroll",
        "target_block_batch_scan_unroll",
        "occupancy_sort",
        "skip_empty_tiles",
        "componentwise_pairs",
    ),
)
def _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl(
    positions: Array,
    block_source_leaf_ids_padded: Array,
    block_valid_mask_padded: Array,
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    *,
    G: Union[float, Array],
    softening_sq: Array,
    target_leaf_batch_size: int,
    target_block_tile_size: int,
    target_block_tile_scan_unroll: int,
    target_block_batch_scan_unroll: int,
    occupancy_sort: bool = False,
    skip_empty_tiles: bool = False,
    componentwise_pairs: bool = False,
) -> Array:
    """Target-major prepacked TONB path over [leaf, block, lane] prepared layout."""
    dtype = positions.dtype
    g_const = jnp.asarray(G, dtype=dtype)
    num_leaves = int(leaf_positions.shape[0])
    leaf_size = int(leaf_positions.shape[1])
    max_blocks = int(block_source_leaf_ids_padded.shape[1])
    block_size = int(block_source_leaf_ids_padded.shape[2])

    if num_leaves == 0 or max_blocks == 0 or block_size == 0:
        return jnp.zeros_like(positions)

    block_tile = int(target_block_tile_size)
    if block_tile <= 0:
        raise ValueError("target_block_tile_size must be positive")

    n_tiles = (max_blocks + block_tile - 1) // block_tile
    padded_blocks = n_tiles * block_tile

    source_leaf_ids_all = block_source_leaf_ids_padded
    source_valid_all = block_valid_mask_padded
    if bool(occupancy_sort):
        block_counts = jnp.sum(jnp.any(source_valid_all, axis=-1), axis=1)
        leaf_order = jnp.argsort(block_counts, stable=True)
        old_to_new = (
            jnp.zeros((num_leaves,), dtype=INDEX_DTYPE)
            .at[leaf_order]
            .set(jnp.arange(num_leaves, dtype=INDEX_DTYPE))
        )
        source_leaf_ids_all = source_leaf_ids_all[leaf_order]
        source_valid_all = source_valid_all[leaf_order]
        source_leaf_ids_all = jnp.where(
            source_valid_all,
            old_to_new[source_leaf_ids_all],
            0,
        )
        leaf_positions = leaf_positions[leaf_order]
        leaf_masses = leaf_masses[leaf_order]
        leaf_mask = leaf_mask[leaf_order]
        leaf_particle_idx = leaf_particle_idx[leaf_order]
    if padded_blocks != max_blocks:
        pad_blocks = padded_blocks - max_blocks
        source_leaf_ids_all = jnp.pad(
            source_leaf_ids_all,
            ((0, 0), (0, pad_blocks), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        source_valid_all = jnp.pad(
            source_valid_all,
            ((0, 0), (0, pad_blocks), (0, 0)),
            mode="constant",
            constant_values=False,
        )

    source_leaf_ids_tiles = jnp.swapaxes(
        source_leaf_ids_all.reshape((num_leaves, n_tiles, block_tile, block_size)),
        0,
        1,
    )
    source_valid_tiles = jnp.swapaxes(
        source_valid_all.reshape((num_leaves, n_tiles, block_tile, block_size)),
        0,
        1,
    )

    return _compute_target_block_pairs_from_source_tiles(
        positions,
        source_leaf_ids_tiles,
        source_valid_tiles,
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
        g_const=g_const,
        softening_sq=softening_sq,
        target_leaf_batch_size=target_leaf_batch_size,
        target_block_tile_scan_unroll=target_block_tile_scan_unroll,
        target_block_batch_scan_unroll=target_block_batch_scan_unroll,
        skip_empty_tiles=bool(skip_empty_tiles),
        componentwise_pairs=bool(componentwise_pairs),
    )


@partial(
    jax.jit,
    static_argnames=(
        "target_leaf_batch_size",
        "target_block_tile_size",
        "target_block_tile_scan_unroll",
        "target_block_batch_scan_unroll",
    ),
)
def _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_tiled_impl(
    positions: Array,
    block_offsets: Array,
    block_target_leaf_ids: Array,
    block_source_leaf_ids: Array,
    block_valid_mask: Array,
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    *,
    G: Union[float, Array],
    softening_sq: Array,
    target_leaf_batch_size: int,
    target_block_tile_size: int,
    target_block_tile_scan_unroll: int,
    target_block_batch_scan_unroll: int,
) -> Array:
    """Bounded overflow TONB pair kernel using canonical tiled source tensors."""
    del block_target_leaf_ids  # kept for API compatibility with prepared state

    dtype = positions.dtype
    g_const = jnp.asarray(G, dtype=dtype)
    num_leaves = int(leaf_positions.shape[0])
    num_blocks = int(block_source_leaf_ids.shape[0])
    block_size = int(block_source_leaf_ids.shape[1])

    if num_leaves == 0 or num_blocks == 0 or block_size == 0:
        return jnp.zeros_like(positions)

    block_tile = int(target_block_tile_size)
    if block_tile <= 0:
        raise ValueError("target_block_tile_size must be positive")

    leaf_ids = jnp.arange(num_leaves, dtype=INDEX_DTYPE)
    block_start = block_offsets[leaf_ids]
    block_stop = block_offsets[leaf_ids + as_index(1)]
    block_count = block_stop - block_start

    n_tiles = (num_blocks + block_tile - 1) // block_tile
    tile_starts = jnp.arange(0, n_tiles * block_tile, block_tile, dtype=INDEX_DTYPE)
    block_tile_offsets = jnp.arange(block_tile, dtype=INDEX_DTYPE)

    local_block_idx = tile_starts[:, None, None] + block_tile_offsets[None, None, :]
    in_tile = local_block_idx < block_count[None, :, None]
    block_idx = block_start[None, :, None] + local_block_idx
    safe_block_idx = jnp.where(in_tile, block_idx, 0)

    source_leaf_ids_tiles = block_source_leaf_ids[safe_block_idx]
    source_valid_tiles = block_valid_mask[safe_block_idx] & in_tile[:, :, :, None]

    return _compute_target_block_pairs_from_source_tiles(
        positions,
        source_leaf_ids_tiles,
        source_valid_tiles,
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
        g_const=g_const,
        softening_sq=softening_sq,
        target_leaf_batch_size=target_leaf_batch_size,
        target_block_tile_scan_unroll=target_block_tile_scan_unroll,
        target_block_batch_scan_unroll=target_block_batch_scan_unroll,
    )


def compute_leaf_p2p_accelerations_target_block_pairs_only(
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_particle_indices: Array,
    leaf_particle_mask: Array,
    block_offsets: Array,
    block_target_leaf_ids: Array,
    block_source_leaf_ids: Array,
    block_valid_mask: Array,
    *,
    G: Union[float, Array] = 1.0,
    softening: float = 0.0,
    target_leaf_batch_size: int = 32,
    target_block_tile_size: int = 8,
    target_block_tile_scan_unroll: int = 1,
    target_block_batch_scan_unroll: int = 1,
    target_block_overflow_fast_max_blocks: int = 65536,
) -> Array:
    """Evaluate target-block pair contributions without intra-leaf self work."""
    positions = jnp.asarray(positions_sorted)
    masses = jnp.asarray(masses_sorted)
    block_source_leaf_ids = jnp.asarray(block_source_leaf_ids, dtype=INDEX_DTYPE)
    block_valid_mask = jnp.asarray(block_valid_mask, dtype=bool)
    if int(block_source_leaf_ids.size) == 0:
        return jnp.zeros_like(positions)

    leaf_positions, leaf_masses, leaf_mask, leaf_particle_idx = (
        _prepare_leaf_data_from_groups(
            leaf_particle_indices,
            leaf_particle_mask,
            positions,
            masses,
        )
    )
    softening_sq = jnp.asarray(float(softening) ** 2, dtype=positions.dtype)
    use_tiled_overflow = int(block_source_leaf_ids.shape[0]) <= int(
        target_block_overflow_fast_max_blocks
    )
    overflow_pair_kernel = (
        _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_tiled_impl
        if use_tiled_overflow
        else _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl
    )
    return overflow_pair_kernel(
        positions,
        jnp.asarray(block_offsets, dtype=INDEX_DTYPE),
        jnp.asarray(block_target_leaf_ids, dtype=INDEX_DTYPE),
        block_source_leaf_ids,
        block_valid_mask,
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
        G=G,
        softening_sq=softening_sq,
        target_leaf_batch_size=int(target_leaf_batch_size),
        target_block_tile_size=int(target_block_tile_size),
        target_block_tile_scan_unroll=int(target_block_tile_scan_unroll),
        target_block_batch_scan_unroll=int(target_block_batch_scan_unroll),
    )


@partial(
    jax.jit,
    static_argnames=(
        "target_leaf_batch_size",
        "target_block_tile_size",
        "target_block_tile_scan_unroll",
        "target_block_batch_scan_unroll",
    ),
)
def _compute_leaf_p2p_prepared_large_n_accel_only_target_blocks_impl(
    positions: Array,
    block_offsets: Array,
    block_target_leaf_ids: Array,
    block_source_leaf_ids: Array,
    block_valid_mask: Array,
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    *,
    G: Union[float, Array],
    softening_sq: Array,
    target_leaf_batch_size: int,
    target_block_tile_size: int,
    target_block_tile_scan_unroll: int,
    target_block_batch_scan_unroll: int,
) -> Array:
    """Specialized accel-only kernel using prepacked target-owned source blocks."""
    self_acc = _compute_leaf_p2p_prepared_large_n_self_only_impl(
        positions,
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
        G=G,
        softening_sq=softening_sq,
    )
    pair_acc = _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl(
        positions,
        block_offsets,
        block_target_leaf_ids,
        block_source_leaf_ids,
        block_valid_mask,
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
        G=G,
        softening_sq=softening_sq,
        target_leaf_batch_size=target_leaf_batch_size,
        target_block_tile_size=target_block_tile_size,
        target_block_tile_scan_unroll=target_block_tile_scan_unroll,
        target_block_batch_scan_unroll=target_block_batch_scan_unroll,
    )
    return self_acc + pair_acc


@partial(
    jax.jit,
    static_argnames=(
        "edge_chunk_size",
        "chunks_per_superchunk",
        "chunk_scan_batch_size",
        "chunk_scan_unroll",
        "superchunk_scan_unroll",
        "sorted_scatter_hint",
        "grouped_sorted_scatter",
        "superchunk_target_reduce",
        "disable_chunk_cond",
    ),
)
def _compute_leaf_p2p_prepared_large_n_accel_only_impl(
    positions: Array,
    target_leaf_ids: Array,
    source_leaf_ids: Array,
    valid_pairs: Array,
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    *,
    G: Union[float, Array],
    softening_sq: Array,
    edge_chunk_size: int,
    chunks_per_superchunk: int,
    chunk_scan_batch_size: int = 1,
    chunk_scan_unroll: int = 1,
    superchunk_scan_unroll: int = 1,
    sorted_scatter_hint: bool,
    grouped_sorted_scatter: bool,
    superchunk_target_reduce: bool,
    disable_chunk_cond: bool,
) -> Array:
    """Specialized accel-only kernel for large-N bucketed prepared leaf data."""
    self_acc = _compute_leaf_p2p_prepared_large_n_self_only_impl(
        positions,
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
        G=G,
        softening_sq=softening_sq,
    )
    pair_acc = _compute_leaf_p2p_prepared_large_n_pairs_only_impl(
        positions,
        target_leaf_ids,
        source_leaf_ids,
        valid_pairs,
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
        G=G,
        softening_sq=softening_sq,
        edge_chunk_size=edge_chunk_size,
        chunks_per_superchunk=chunks_per_superchunk,
        chunk_scan_batch_size=chunk_scan_batch_size,
        chunk_scan_unroll=chunk_scan_unroll,
        superchunk_scan_unroll=superchunk_scan_unroll,
        sorted_scatter_hint=sorted_scatter_hint,
        grouped_sorted_scatter=grouped_sorted_scatter,
        superchunk_target_reduce=superchunk_target_reduce,
        disable_chunk_cond=disable_chunk_cond,
    )
    return self_acc + pair_acc


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
    G: Union[float, Array],
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
    """JIT near-field kernel over leaf-neighbor particle interactions.

    Pads the per-leaf particle data with :func:`_prepare_leaf_data` and then
    defers to :func:`_compute_leaf_p2p_from_prepared_leaf_data_impl`, which is
    the same kernel taking that padded data as arguments. The two used to be
    separate 460-line copies that were 94.4% line-identical and had already
    drifted apart; this follows the delegation precedent
    :func:`_compute_leaf_p2p_prepared_large_n_accel_only_impl` already sets in
    this module.

    ``max_leaf_size`` is recovered inside the callee as
    ``leaf_particle_idx.shape[1]``, which :func:`_prepare_leaf_data` sets to
    exactly this argument -- so it stays a compile-time constant on both sides
    of the call and neither side gains a traced shape.

    Parameters
    ----------
    node_ranges : Array
        Per-node ``[start, end]`` particle spans, ``[num_nodes, 2]``.
    leaf_nodes : Array
        Node indices of the leaves, ``[num_leaves]``.
    offsets : Array
        CSR row offsets into ``neighbors``, ``[num_leaves + 1]``.
    neighbors : Array
        Flat leaf-neighbour edge list, ``[num_edges]``.
    positions : Array
        Particle positions ``[N, 3]`` in Morton order.
    masses : Array
        Particle masses ``[N]`` in the same order.
    target_leaf_ids : Array
        Per-edge target leaf index, ``[num_edges]``.
    source_leaf_ids : Array
        Per-edge source leaf index, ``[num_edges]``.
    valid_pairs : Array
        Per-edge validity mask, ``[num_edges]``; padded edges are ``False``.
    precomputed_chunk_sort_indices : Array
        Scatter-schedule sort permutation, or an empty array when
        ``use_precomputed_scatter`` is ``False``.
    precomputed_chunk_group_ids : Array
        Scatter-schedule group ids, same convention.
    precomputed_chunk_unique_indices : Array
        Scatter-schedule unique-target indices, same convention.
    max_leaf_size : int
        Padded leaf width. Static under ``jit`` (``static_argnums=(12,)``).
    G : Union[float, Array]
        Gravitational constant.
    softening_sq : Array
        Squared Plummer softening length.
    return_potential : bool
        Also return per-particle potentials. Static under ``jit``.
    collect_neighbor_pairs : bool
        Also return the realised ``(target, source)`` leaf pairs and their
        count. Static under ``jit``.
    nearfield_mode : str
        ``"baseline"`` or ``"bucketed"``. Static under ``jit``.
    edge_chunk_size : int
        Edge-chunk width for ``"bucketed"``. Static under ``jit``.
    use_precomputed_scatter : bool
        Use the precomputed scatter schedules rather than deriving them.
        Static under ``jit``.

    Returns
    -------
    Union[Array, Tuple[Array, Array], Tuple[Array, Array, Array], Tuple[Array, Array, Array, Array]]
        Accelerations ``[N, 3]``, followed by potentials ``[N]`` when
        ``return_potential``, followed by the neighbour-pair array and count
        when ``collect_neighbor_pairs``. A one-element result is unwrapped.
    """
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
    return _compute_leaf_p2p_from_prepared_leaf_data_impl(
        offsets,
        neighbors,
        positions,
        target_leaf_ids,
        source_leaf_ids,
        valid_pairs,
        precomputed_chunk_sort_indices,
        precomputed_chunk_group_ids,
        precomputed_chunk_unique_indices,
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
        G=G,
        softening_sq=softening_sq,
        return_potential=return_potential,
        collect_neighbor_pairs=collect_neighbor_pairs,
        nearfield_mode=nearfield_mode,
        edge_chunk_size=edge_chunk_size,
        use_precomputed_scatter=use_precomputed_scatter,
    )


@partial(
    jax.jit,
    static_argnames=(
        "return_potential",
        "collect_neighbor_pairs",
        "nearfield_mode",
        "edge_chunk_size",
        "use_precomputed_scatter",
    ),
)
def _compute_leaf_p2p_from_prepared_leaf_data_impl(
    offsets: Array,
    neighbors: Array,
    positions: Array,
    target_leaf_ids: Array,
    source_leaf_ids: Array,
    valid_pairs: Array,
    precomputed_chunk_sort_indices: Array,
    precomputed_chunk_group_ids: Array,
    precomputed_chunk_unique_indices: Array,
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    *,
    G: Union[float, Array],
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
    """JIT near-field kernel over explicit per-leaf particle groups."""
    dtype = positions.dtype
    g_const = jnp.asarray(G, dtype=dtype)
    soft_sq = softening_sq
    max_leaf_size = int(leaf_particle_idx.shape[1])

    accelerations = jnp.zeros_like(positions)
    if return_potential:
        potentials = jnp.zeros((positions.shape[0],), dtype=dtype)
    else:
        potentials = None

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

                        def _compute(args):
                            acc_in, pot_in = args
                            tgt_leaf = target_leaf_ids[safe_edge_idx]
                            src_leaf = source_leaf_ids[safe_edge_idx]
                            tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                            src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                            tgt_pos = leaf_positions[tgt_leaf_local]
                            tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                            src_pos = leaf_positions[src_leaf_local]
                            src_mass = leaf_masses[src_leaf_local]
                            src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

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
                            acc_out = _scatter_vectors_with_schedule(
                                acc_in,
                                pair_acc,
                                tgt_mask,
                                sort_idx,
                                group_ids,
                                unique_indices,
                            )
                            pot_out = _scatter_scalars_with_schedule(
                                pot_in,
                                pair_pot,
                                tgt_mask,
                                sort_idx,
                                group_ids,
                                unique_indices,
                            )
                            return acc_out, pot_out

                        return (
                            lax.cond(
                                jnp.any(valid_edge),
                                _compute,
                                lambda args: args,
                                (acc, pot),
                            ),
                            None,
                        )

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

                        def _compute(args):
                            acc_in, pot_in = args
                            tgt_leaf = target_leaf_ids[safe_edge_idx]
                            src_leaf = source_leaf_ids[safe_edge_idx]
                            tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                            src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                            tgt_pos = leaf_positions[tgt_leaf_local]
                            tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                            tgt_ids = leaf_particle_idx[tgt_leaf_local]
                            src_pos = leaf_positions[src_leaf_local]
                            src_mass = leaf_masses[src_leaf_local]
                            src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

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
                            acc_out = _scatter_contributions(
                                acc_in,
                                tgt_ids,
                                pair_acc,
                                tgt_mask,
                            )
                            pot_out = _scatter_scalar_contributions(
                                pot_in,
                                tgt_ids,
                                pair_pot,
                                tgt_mask,
                            )
                            return acc_out, pot_out

                        return (
                            lax.cond(
                                jnp.any(valid_edge),
                                _compute,
                                lambda args: args,
                                (acc, pot),
                            ),
                            None,
                        )

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

                        def _compute(acc_in):
                            tgt_leaf = target_leaf_ids[safe_edge_idx]
                            src_leaf = source_leaf_ids[safe_edge_idx]
                            tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                            src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                            # Gather + pair evaluation, rematerialized: the
                            # composite is the dominant reverse-pass residual at
                            # galaxy N (see ``_bucketed_chunk_pair_accels``).
                            pair_acc, tgt_mask = _bucketed_chunk_pair_accels_remat(
                                leaf_positions,
                                leaf_masses,
                                leaf_mask,
                                tgt_leaf_local,
                                src_leaf_local,
                                valid_edge,
                                soft_sq,
                                g_const,
                            )
                            return _scatter_vectors_with_schedule(
                                acc_in,
                                pair_acc,
                                tgt_mask,
                                sort_idx,
                                group_ids,
                                unique_indices,
                            )

                        return (
                            lax.cond(
                                jnp.any(valid_edge),
                                _compute,
                                lambda acc_in: acc_in,
                                acc,
                            ),
                            None,
                        )

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

                        def _compute(acc_in):
                            tgt_leaf = target_leaf_ids[safe_edge_idx]
                            src_leaf = source_leaf_ids[safe_edge_idx]
                            tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                            src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                            tgt_ids = leaf_particle_idx[tgt_leaf_local]
                            # Gather + pair evaluation, rematerialized: the
                            # composite is the dominant reverse-pass residual at
                            # galaxy N (see ``_bucketed_chunk_pair_accels``).
                            pair_acc, tgt_mask = _bucketed_chunk_pair_accels_remat(
                                leaf_positions,
                                leaf_masses,
                                leaf_mask,
                                tgt_leaf_local,
                                src_leaf_local,
                                valid_edge,
                                soft_sq,
                                g_const,
                            )
                            return _scatter_contributions(
                                acc_in,
                                tgt_ids,
                                pair_acc,
                                tgt_mask,
                            )

                        return (
                            lax.cond(
                                jnp.any(valid_edge),
                                _compute,
                                lambda acc_in: acc_in,
                                acc,
                            ),
                            None,
                        )

                    accelerations, _ = lax.scan(
                        _chunk_body,
                        accelerations,
                        starts,
                    )
        elif return_potential and potentials is not None:

            def _edge_body(carry, data):
                acc, pot = carry
                tgt_idx, src_idx, is_valid = data

                def true_branch(
                    args: tuple[Array, Array, Array, Array],
                ) -> tuple[Array, Array]:
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

                    masked_acc = jnp.where(target_mask[:, None], pair_accel, 0.0)
                    masked_pot = jnp.where(target_mask, pair_pot, 0.0)

                    acc_state = acc_state.at[target_ids].add(masked_acc)
                    pot_state = pot_state.at[target_ids].add(masked_pot)
                    return acc_state, pot_state

                def false_branch(
                    args: tuple[Array, Array, Array, Array],
                ) -> tuple[Array, Array]:
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

                def true_branch(args: tuple[Array, Array, Array]) -> Array:
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

                    masked_acc = jnp.where(target_mask[:, None], pair_accel, 0.0)
                    acc_state = acc_state.at[target_ids].add(masked_acc)
                    return acc_state

                def false_branch(args: tuple[Array, Array, Array]) -> Array:
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
                pair = jnp.stack([target_leaf_ids[idx], source_leaf_ids[idx]], axis=0)
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
        outputs += (neighbor_pairs, pair_count)

    if len(outputs) == 1:
        return outputs[0]
    return outputs


@jaxtyped(typechecker=beartype)
def compute_leaf_p2p_accelerations(
    tree: Tree,
    neighbor_list: NodeNeighborList,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    G: Union[float, Array] = 1.0,
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
    node_ranges_override: Optional[Array] = None,
    leaf_nodes_override: Optional[Array] = None,
    neighbor_offsets_override: Optional[Array] = None,
    neighbor_indices_override: Optional[Array] = None,
    neighbor_counts_override: Optional[Array] = None,
    leaf_particle_indices_override: Optional[Array] = None,
    leaf_particle_mask_override: Optional[Array] = None,
) -> Union[
    Array,
    Tuple[Array, Array],
    Tuple[Array, Array, Array],
    Tuple[Array, Array, Array, Array],
]:
    """Compute near-field contributions for all leaf particle pairs.

    Parameters
    ----------
    tree : Tree
        Radix tree whose ``node_ranges`` map leaves to particle spans.
    neighbor_list : NodeNeighborList
        Precomputed leaf-neighbour CSR metadata (``leaf_indices``, ``offsets``,
        ``neighbors``, ``counts``). Each ``*_override`` below replaces exactly
        one of these fields.
    positions_sorted : Array
        Particle positions ``[N, 3]`` in Morton order.
    masses_sorted : Array
        Particle masses ``[N]`` in Morton order.
    G : Union[float, Array]
        Gravitational constant, applied as a plain multiplier.
    softening : float
        Plummer softening length. Squared host-side via ``float(softening)``, so
        it must be a concrete Python float, not a tracer.
    max_leaf_size : Optional[int]
        Static bound on per-leaf particle count. **Required under ``jit``**: when
        it is ``None`` the code reads the true maximum with ``.item()``, which
        only works on concrete values (see ``Raises``).
    return_potential : bool
        Also accumulate potentials. Static under ``jit`` -- it changes the number
        of returned arrays.
    collect_neighbor_pairs : bool
        Also return the processed ``(target, source)`` leaf-pair buffer. Static
        under ``jit``, and it additionally flips the internal edge ordering
        (``sort_by_source``), so enabling it is not purely additive.
    nearfield_mode : str
        Either ``"baseline"`` or ``"bucketed"``; static under ``jit``.
        ``"bucketed"`` is the minimum-memory large-N GPU path and preserves
        target-local edge order for scatter locality. The two modes are meant to
        produce the same accelerations.
    edge_chunk_size : int
        Chunk width for the bucketed edge scan. Static under ``jit``; a
        performance knob, not a numerical one.
    precomputed_target_leaf_ids : Optional[Array]
        Leaf-pair schedule buffer: target leaf id per edge. Supplied together
        with the other ``precomputed_*`` arrays by prepared state to skip
        re-deriving the schedule. Must match the neighbour-list edge order the
        rest of the schedule was built against.
    precomputed_source_leaf_ids : Optional[Array]
        Source leaf id per edge, same ordering contract.
    precomputed_valid_pairs : Optional[Array]
        Boolean mask marking which padded edge slots are real.
    precomputed_chunk_sort_indices : Optional[Array]
        Scatter-schedule permutation for the chunked accumulation.
    precomputed_chunk_group_ids : Optional[Array]
        Chunk group id per sorted edge.
    precomputed_chunk_unique_indices : Optional[Array]
        Unique-chunk boundary indices. This and the previous two are consumed as
        a set: the precomputed scatter path only engages when all three are given.
    node_ranges_override : Optional[Array]
        Replaces ``tree.node_ranges``.
    leaf_nodes_override : Optional[Array]
        Replaces ``neighbor_list.leaf_indices``.
    neighbor_offsets_override : Optional[Array]
        Replaces ``neighbor_list.offsets``.
    neighbor_indices_override : Optional[Array]
        Replaces ``neighbor_list.neighbors``.
    neighbor_counts_override : Optional[Array]
        Replaces ``neighbor_list.counts``.
    leaf_particle_indices_override : Optional[Array]
        Explicit per-leaf particle index table ``[num_leaves, max_leaf_size]``.
        Supplying it *sets* ``max_leaf_size`` from its second axis, overriding
        the argument.
    leaf_particle_mask_override : Optional[Array]
        Validity mask matching ``leaf_particle_indices_override``.

    Returns
    -------
    Union[Array, Tuple[Array, Array], Tuple[Array, Array, Array], Tuple[Array, Array, Array, Array]]
        Accelerations ``[N, 3]`` in Morton order, returned bare when neither flag
        is set. ``return_potential`` appends potentials ``[N]``;
        ``collect_neighbor_pairs`` appends the full ``(target, source)`` pair
        buffer and then the scalar count of valid entries (slice with
        ``neighbor_pairs[:neighbor_pair_count]``). With both flags the order is
        ``(accelerations, potentials, neighbor_pairs, pair_count)``.

    Raises
    ------
    ValueError
        If ``max_leaf_size`` is ``None`` while tracing. The bound is otherwise
        read from the data with ``.item()``; under a tracer that raises
        ``TypeError``, which is re-raised as this ``ValueError``. It is the one
        place in this function where a host sync is attempted deliberately, and
        it is the reason ``max_leaf_size`` must be passed by every jitted caller.

    Notes
    -----
    Differentiable in ``positions_sorted``, ``masses_sorted``, and ``G``. Not
    differentiable in ``softening`` (pulled to the host) or in any of the
    integer index/topology arrays.

    Every extent here is padded to a static bound -- leaf occupancy via
    ``max_leaf_size``, the edge list via the valid-pair mask. Padded slots must
    contribute exactly zero and must never form ``0 * inf``; see
    ``bench/audit_nearfield_padding.py``.

    An empty leaf set short-circuits to zeros of the right shape. Self-pairs and
    coincident particles are handled by the ``softening`` term; with
    ``softening == 0`` a coincident pair is a genuine singularity that this
    function does not guard.

    ``nearfield_mode="baseline"`` and ``"bucketed"`` agree **to a tolerance, not
    bit-exactly**, and that is the correct contract rather than a gap: the two
    deliberately differ in edge order (``sort_by_source``), which changes the
    order of the floating-point accumulation, so bit-equality is not expected.
    Asserted by
    ``tests/integration/test_fmm.py::test_nearfield_bucketed_matches_baseline``,
    parametrised over N in {96, 256} x {float32, float64} x
    ``edge_chunk_size`` in {64, 128}, at ``1e-6`` (fp32) and ``1e-13`` (fp64).
    Those bounds are derived from measured round-off (~1 eps in each dtype), not
    assumed, and the fp32/chunk-128 cases run in the CI smoke leg.

    The **float64 cases are the sharp instrument**: perturbing the bucketed path's
    softening by 3e-6 relative induces a ~1e-10 divergence, which fp64 catches
    against its 1e-13 bound and fp32 structurally cannot, since fp32 round-off is
    already ~1e-7.
    **The ``precomputed_*`` contract is shape-encoded, and partial sets are
    supported by design.** An earlier note here called this unvalidated; that was
    wrong. The mechanism: a ``None`` is converted to a zero-size sentinel by the
    caller in :mod:`jaccpot.runtime.fmm_evaluate`, and
    :mod:`jaccpot.runtime.kernels.core` then admits an array only if its leading
    dimension equals ``neighbor_list.neighbors.shape[0]`` (the scatter schedules
    must match ``(chunk_count, chunk_flat_size)``). Anything else -- including the
    sentinel -- is ignored and recomputed.

    Three groups fall back **independently**: the target/valid pair vectors, the
    source leaf ids, and the scatter schedules. That is what makes
    ``_prepare_bucketed_scatter_schedules_safe`` in
    :mod:`jaccpot.runtime.fmm_prepare` sound when it returns ``(None, None, None)``
    on int32 overflow, on exceeding the schedule cap, or on any exception, while
    the pair vectors stay populated. A 3-of-6 set is a normal production state,
    not an error.

    **The invariant that is *not* checked is edge order.** Precomputed vectors
    must be positionally aligned with ``neighbor_list.neighbors`` -- i.e. built
    with ``sort_by_source=False``. When ``precomputed_source_leaf_ids`` is absent
    the source ids are re-derived positionally as ``leaf_lookup[neighbors]``, so a
    source-sorted ``target_leaf_ids`` would be silently paired against unsorted
    sources. Both orderings have the same length, so no shape check can catch it,
    and the result is wrong forces with no error and no NaN. All in-repo producers
    pass ``sort_by_source=False``; see :func:`prepare_leaf_neighbor_pairs`, whose
    default is the *unsafe* value for this contract.
    """

    positions = jnp.asarray(positions_sorted)
    masses = jnp.asarray(masses_sorted)
    node_ranges = (
        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
        if node_ranges_override is None
        else jnp.asarray(node_ranges_override, dtype=INDEX_DTYPE)
    )

    leaf_nodes = (
        jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
        if leaf_nodes_override is None
        else jnp.asarray(leaf_nodes_override, dtype=INDEX_DTYPE)
    )
    offsets = (
        jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE)
        if neighbor_offsets_override is None
        else jnp.asarray(neighbor_offsets_override, dtype=INDEX_DTYPE)
    )
    neighbors = (
        jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE)
        if neighbor_indices_override is None
        else jnp.asarray(neighbor_indices_override, dtype=INDEX_DTYPE)
    )
    neighbor_counts = (
        jnp.asarray(neighbor_list.counts, dtype=INDEX_DTYPE)
        if neighbor_counts_override is None
        else jnp.asarray(neighbor_counts_override, dtype=INDEX_DTYPE)
    )

    if leaf_nodes.size == 0:
        zeros = jnp.zeros_like(positions)
        if return_potential:
            pot_zeros = jnp.zeros((positions.shape[0],), dtype=zeros.dtype)
            return zeros, pot_zeros
        return zeros

    if leaf_particle_indices_override is not None:
        explicit_leaf_particle_indices = jnp.asarray(
            leaf_particle_indices_override,
            dtype=INDEX_DTYPE,
        )
        explicit_leaf_particle_mask = (
            jnp.asarray(leaf_particle_mask_override, dtype=bool)
            if leaf_particle_mask_override is not None
            else jnp.ones_like(explicit_leaf_particle_indices, dtype=bool)
        )
        max_leaf_size = int(explicit_leaf_particle_indices.shape[1])
    else:
        explicit_leaf_particle_indices = None
        explicit_leaf_particle_mask = None
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

    use_precomputed_scatter = (
        precomputed_chunk_sort_indices is not None
        and precomputed_chunk_group_ids is not None
        and precomputed_chunk_unique_indices is not None
    )

    if precomputed_target_leaf_ids is None or precomputed_valid_pairs is None:
        # Precomputed scatter schedules are built against the neighbor-list edge
        # order used by prepared state. Re-derive leaf-pair vectors in that same
        # order so bucketed scans stay aligned with the schedule buffers.
        sort_by_source = not bool(collect_neighbor_pairs)
        if str(nearfield_mode).strip().lower() == "bucketed" and not bool(
            use_precomputed_scatter
        ):
            # The minimum-memory large-N GPU path uses direct scatter in the
            # bucketed loop, so preserving target-local edge order improves
            # output-update locality more than source-sorted gather locality.
            sort_by_source = False
        if use_precomputed_scatter:
            sort_by_source = False
        target_leaf_ids, source_leaf_ids, valid_pairs = prepare_leaf_neighbor_pairs(
            node_ranges,
            leaf_nodes,
            offsets,
            neighbors,
            sort_by_source=sort_by_source,
        )
    else:
        target_leaf_ids = jnp.asarray(precomputed_target_leaf_ids, dtype=INDEX_DTYPE)
        valid_pairs = jnp.asarray(precomputed_valid_pairs, dtype=bool)
        if precomputed_source_leaf_ids is None:
            # Compact prepared-state mode: derive source leaf ids directly from
            # neighbor edges while reusing precomputed target/valid buffers.
            total_nodes = node_ranges.shape[0]
            leaf_lookup = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
            leaf_lookup = leaf_lookup.at[leaf_nodes].set(
                jnp.arange(leaf_nodes.shape[0], dtype=INDEX_DTYPE)
            )
            source_leaf_ids = leaf_lookup[neighbors]
            valid_pairs = valid_pairs & (source_leaf_ids >= 0)
        else:
            source_leaf_ids = jnp.asarray(
                precomputed_source_leaf_ids,
                dtype=INDEX_DTYPE,
            )

    if use_precomputed_scatter:
        chunk_sort_indices = jnp.asarray(
            precomputed_chunk_sort_indices, dtype=INDEX_DTYPE
        )
        chunk_group_ids = jnp.asarray(precomputed_chunk_group_ids, dtype=INDEX_DTYPE)
        chunk_unique_indices = jnp.asarray(
            precomputed_chunk_unique_indices,
            dtype=INDEX_DTYPE,
        )
    else:
        chunk_sort_indices = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        chunk_group_ids = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        chunk_unique_indices = jnp.zeros((0, 0), dtype=INDEX_DTYPE)

    if explicit_leaf_particle_indices is None:
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

    (
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
    ) = _prepare_leaf_data_from_groups(
        explicit_leaf_particle_indices,
        explicit_leaf_particle_mask,
        positions,
        masses,
    )
    return _compute_leaf_p2p_from_prepared_leaf_data_impl(
        offsets,
        neighbors,
        positions,
        target_leaf_ids,
        source_leaf_ids,
        valid_pairs,
        chunk_sort_indices,
        chunk_group_ids,
        chunk_unique_indices,
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
        G=G,
        softening_sq=softening_sq,
        return_potential=return_potential,
        collect_neighbor_pairs=collect_neighbor_pairs,
        nearfield_mode=nearfield_mode,
        edge_chunk_size=int(edge_chunk_size),
        use_precomputed_scatter=use_precomputed_scatter,
    )


def compute_leaf_p2p_accelerations_large_n_accel_only(
    tree: Tree,
    neighbor_list: NodeNeighborList,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    G: Union[float, Array] = 1.0,
    softening: float = 0.0,
    edge_chunk_size: int = 256,
    precomputed_target_leaf_ids: Optional[Array] = None,
    precomputed_source_leaf_ids: Optional[Array] = None,
    precomputed_valid_pairs: Optional[Array] = None,
    leaf_particle_indices: Array,
    leaf_particle_mask: Optional[Array] = None,
    precomputed_target_block_leaf_ids: Optional[Array] = None,
    precomputed_target_block_source_leaf_ids: Optional[Array] = None,
    precomputed_target_block_valid_mask: Optional[Array] = None,
    precomputed_target_block_offsets: Optional[Array] = None,
    precomputed_target_block_source_leaf_ids_padded: Optional[Array] = None,
    precomputed_target_block_valid_mask_padded: Optional[Array] = None,
    delayed_scatter_chunks_per_superchunk: Optional[int] = None,
    chunk_scan_batch_size: Optional[int] = None,
    chunk_scan_unroll: Optional[int] = None,
    superchunk_scan_unroll: Optional[int] = None,
    sorted_scatter_hint: Optional[bool] = None,
    grouped_sorted_scatter: Optional[bool] = None,
    superchunk_target_reduce: Optional[bool] = None,
    disable_chunk_cond: Optional[bool] = None,
    target_leaf_batch_size: Optional[int] = None,
    target_block_tile_size: Optional[int] = None,
    target_block_tile_scan_unroll: Optional[int] = None,
    target_block_batch_scan_unroll: Optional[int] = None,
    target_block_overflow_fast_max_blocks: Optional[int] = None,
) -> Array:
    """Specialized accel-only bucketed near-field path for large-N prepared data."""
    positions = jnp.asarray(positions_sorted)
    masses = jnp.asarray(masses_sorted)
    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    offsets = jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE)
    neighbors = jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE)

    if leaf_nodes.size == 0:
        return jnp.zeros_like(positions)

    explicit_leaf_particle_indices = jnp.asarray(
        leaf_particle_indices,
        dtype=INDEX_DTYPE,
    )
    explicit_leaf_particle_mask = (
        jnp.asarray(leaf_particle_mask, dtype=bool)
        if leaf_particle_mask is not None
        else jnp.ones_like(explicit_leaf_particle_indices, dtype=bool)
    )

    if precomputed_target_leaf_ids is None or precomputed_valid_pairs is None:
        target_leaf_ids, source_leaf_ids, valid_pairs = prepare_leaf_neighbor_pairs(
            node_ranges,
            leaf_nodes,
            offsets,
            neighbors,
            sort_by_source=False,
        )
    else:
        target_leaf_ids = jnp.asarray(precomputed_target_leaf_ids, dtype=INDEX_DTYPE)
        valid_pairs = jnp.asarray(precomputed_valid_pairs, dtype=bool)
        if precomputed_source_leaf_ids is None:
            total_nodes = node_ranges.shape[0]
            leaf_lookup = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
            leaf_lookup = leaf_lookup.at[leaf_nodes].set(
                jnp.arange(leaf_nodes.shape[0], dtype=INDEX_DTYPE)
            )
            source_leaf_ids = leaf_lookup[neighbors]
            valid_pairs = valid_pairs & (source_leaf_ids >= 0)
        else:
            source_leaf_ids = jnp.asarray(
                precomputed_source_leaf_ids,
                dtype=INDEX_DTYPE,
            )

    (
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
    ) = _prepare_leaf_data_from_groups(
        explicit_leaf_particle_indices,
        explicit_leaf_particle_mask,
        positions,
        masses,
    )

    softening_sq = jnp.asarray(float(softening) ** 2, dtype=positions.dtype)
    delayed_scatter_chunks_per_superchunk = int(
        1
        if delayed_scatter_chunks_per_superchunk is None
        else delayed_scatter_chunks_per_superchunk
    )
    chunk_scan_batch_size = int(
        1 if chunk_scan_batch_size is None else chunk_scan_batch_size
    )
    chunk_scan_unroll = int(1 if chunk_scan_unroll is None else chunk_scan_unroll)
    superchunk_scan_unroll = int(
        1 if superchunk_scan_unroll is None else superchunk_scan_unroll
    )
    sorted_scatter_hint = bool(
        False if sorted_scatter_hint is None else sorted_scatter_hint
    )
    grouped_sorted_scatter = bool(
        False if grouped_sorted_scatter is None else grouped_sorted_scatter
    )
    superchunk_target_reduce = bool(
        False if superchunk_target_reduce is None else superchunk_target_reduce
    )
    disable_chunk_cond = bool(
        True if disable_chunk_cond is None else disable_chunk_cond
    )
    target_leaf_batch_size = int(
        32 if target_leaf_batch_size is None else target_leaf_batch_size
    )
    target_block_tile_size = int(
        8 if target_block_tile_size is None else target_block_tile_size
    )
    target_block_tile_scan_unroll = int(
        1 if target_block_tile_scan_unroll is None else target_block_tile_scan_unroll
    )
    target_block_batch_scan_unroll = int(
        1 if target_block_batch_scan_unroll is None else target_block_batch_scan_unroll
    )
    target_block_overflow_fast_max_blocks = int(
        65536
        if target_block_overflow_fast_max_blocks is None
        else target_block_overflow_fast_max_blocks
    )
    use_target_blocks = (
        precomputed_target_block_offsets is not None
        and precomputed_target_block_leaf_ids is not None
        and precomputed_target_block_source_leaf_ids is not None
        and precomputed_target_block_valid_mask is not None
    )
    use_target_blocks_prepacked = (
        precomputed_target_block_source_leaf_ids_padded is not None
        and precomputed_target_block_valid_mask_padded is not None
    )

    if use_target_blocks_prepacked:
        self_acc = _compute_leaf_p2p_prepared_large_n_self_only_impl(
            positions,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=G,
            softening_sq=softening_sq,
        )
        pair_acc = (
            _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl(
                positions,
                jnp.asarray(
                    precomputed_target_block_source_leaf_ids_padded,
                    dtype=INDEX_DTYPE,
                ),
                jnp.asarray(precomputed_target_block_valid_mask_padded, dtype=bool),
                leaf_positions,
                leaf_masses,
                leaf_mask,
                leaf_particle_idx,
                G=G,
                softening_sq=softening_sq,
                target_leaf_batch_size=target_leaf_batch_size,
                target_block_tile_size=target_block_tile_size,
                target_block_tile_scan_unroll=target_block_tile_scan_unroll,
                target_block_batch_scan_unroll=target_block_batch_scan_unroll,
            )
        )
        if use_target_blocks:
            overflow_block_count = int(
                precomputed_target_block_source_leaf_ids.shape[0]
            )
            overflow_pair_kernel = (
                _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_tiled_impl
                if overflow_block_count <= target_block_overflow_fast_max_blocks
                else _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl
            )
            overflow_pair_acc = overflow_pair_kernel(
                positions,
                jnp.asarray(precomputed_target_block_offsets, dtype=INDEX_DTYPE),
                jnp.asarray(precomputed_target_block_leaf_ids, dtype=INDEX_DTYPE),
                jnp.asarray(
                    precomputed_target_block_source_leaf_ids,
                    dtype=INDEX_DTYPE,
                ),
                jnp.asarray(precomputed_target_block_valid_mask, dtype=bool),
                leaf_positions,
                leaf_masses,
                leaf_mask,
                leaf_particle_idx,
                G=G,
                softening_sq=softening_sq,
                target_leaf_batch_size=target_leaf_batch_size,
                target_block_tile_size=target_block_tile_size,
                target_block_tile_scan_unroll=target_block_tile_scan_unroll,
                target_block_batch_scan_unroll=target_block_batch_scan_unroll,
            )
            pair_acc = pair_acc + overflow_pair_acc
        return self_acc + pair_acc

    if use_target_blocks:
        return _compute_leaf_p2p_prepared_large_n_accel_only_target_blocks_impl(
            positions,
            jnp.asarray(precomputed_target_block_offsets, dtype=INDEX_DTYPE),
            jnp.asarray(precomputed_target_block_leaf_ids, dtype=INDEX_DTYPE),
            jnp.asarray(precomputed_target_block_source_leaf_ids, dtype=INDEX_DTYPE),
            jnp.asarray(precomputed_target_block_valid_mask, dtype=bool),
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=G,
            softening_sq=softening_sq,
            target_leaf_batch_size=target_leaf_batch_size,
            target_block_tile_size=target_block_tile_size,
            target_block_tile_scan_unroll=target_block_tile_scan_unroll,
            target_block_batch_scan_unroll=target_block_batch_scan_unroll,
        )

    return _compute_leaf_p2p_prepared_large_n_accel_only_impl(
        positions,
        target_leaf_ids,
        source_leaf_ids,
        valid_pairs,
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
        G=G,
        softening_sq=softening_sq,
        edge_chunk_size=int(edge_chunk_size),
        chunks_per_superchunk=delayed_scatter_chunks_per_superchunk,
        chunk_scan_batch_size=chunk_scan_batch_size,
        chunk_scan_unroll=chunk_scan_unroll,
        superchunk_scan_unroll=superchunk_scan_unroll,
        sorted_scatter_hint=sorted_scatter_hint,
        grouped_sorted_scatter=grouped_sorted_scatter,
        superchunk_target_reduce=superchunk_target_reduce,
        disable_chunk_cond=disable_chunk_cond,
    )
