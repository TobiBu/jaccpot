"""Near-field evaluation helpers for the Fast Multipole Method."""

from __future__ import annotations

import os
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

_LARGE_N_NEARFIELD_DIAG_MODES = frozenset(("full", "self_only", "pairs_only", "zero"))


def _large_n_nearfield_diag_mode() -> str:
    mode = (
        str(os.environ.get("JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE", "full"))
        .strip()
        .lower()
    )
    return mode if mode in _LARGE_N_NEARFIELD_DIAG_MODES else "full"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)


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
    """Precompute neighbor-edge leaf mappings and reorder for source locality."""
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


# Analytic near-field P2P reverse is on by default; JACCPOT_ANALYTIC_P2P_VJP=0
# restores plain autodiff for A/B measurement.
_ANALYTIC_P2P_VJP = os.environ.get(
    "JACCPOT_ANALYTIC_P2P_VJP", "1"
).strip().lower() not in (
    "0",
    "false",
    "no",
)


def _pair_accel_masked_accels(
    target_positions: Array,
    source_positions: Array,
    source_masses: Array,
    target_mask: Array,
    source_mask: Array,
    softening_sq: Union[float, Array],
    G: Array,
) -> Array:
    """Accel-only batched pair contributions (matches the accel output of
    :func:`_pair_contributions_batched`)."""
    diff = target_positions[:, :, None, :] - source_positions[:, None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
    pair_mask = target_mask[:, :, None] & source_mask[:, None, :]
    safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
    inv_r = jnp.where(pair_mask, lax.rsqrt(safe_dist_sq), 0.0)
    inv_dist3 = jnp.where(pair_mask, inv_r * inv_r * inv_r, 0.0)
    weighted = inv_dist3 * source_masses[:, None, :]
    accels = -G * jnp.sum(weighted[..., None] * diff, axis=2)
    return jnp.where(target_mask[..., None], accels, 0.0)


def _pair_accel_pair_terms(
    target_positions: Array,
    source_positions: Array,
    target_mask: Array,
    source_mask: Array,
    softening_sq: Union[float, Array],
) -> Tuple[Array, Array, Array]:
    """``(diff, inv_dist3, inv_dist5)`` exactly as :func:`_pair_accel_masked_accels`
    forms them.

    Factored out so the analytic reverse rule can REMATERIALIZE these
    ``(B, Wt, Ws)``-shaped pair intermediates instead of carrying them in the
    ``custom_vjp`` residual, without the forward and reverse expressions drifting
    apart. See :func:`_pair_accel_cvjp_fwd` for why that matters.
    """
    diff = target_positions[:, :, None, :] - source_positions[:, None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
    pair_mask = target_mask[:, :, None] & source_mask[:, None, :]
    safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
    inv_r = jnp.where(pair_mask, lax.rsqrt(safe_dist_sq), 0.0)
    inv_dist3 = jnp.where(pair_mask, inv_r * inv_r * inv_r, 0.0)
    inv_dist5 = jnp.where(pair_mask, inv_dist3 * inv_r * inv_r, 0.0)
    return diff, inv_dist3, inv_dist5


@jax.custom_vjp
def _pair_accel_cvjp(
    target_positions: Array,
    source_positions: Array,
    source_masses: Array,
    target_mask_f: Array,
    source_mask_f: Array,
    softening_sq: Array,
    G: Array,
) -> Array:
    """Accel-only batched near-field pair kernel with an analytic reverse rule.

    All arguments are float arrays (masks as 0/1 floats), so the reverse returns
    ordinary zero cotangents for the non-differentiated inputs (masks, softening,
    G) -- no closure over tracers (which ``custom_vjp`` forbids). The forward is
    byte-identical to the accel output of :func:`_pair_contributions_batched`; the
    reverse is the analytic symmetric near-field tidal tensor
    ``J = -G Σ_s m_s (I/r³ − 3 r rᵀ/r⁵)`` contracted with the output cotangent
    (``Jᵀc = Jc``) in one extra pair pass -- exactly the reverse rule a future
    fused-Pallas near-field ``custom_vjp`` reuses. Verified bit-for-bit against
    autodiff in ``tests/unit/test_custom_vjp_parity.py``.
    """
    target_mask = target_mask_f > 0.5
    source_mask = source_mask_f > 0.5
    return _pair_accel_masked_accels(
        target_positions,
        source_positions,
        source_masses,
        target_mask,
        source_mask,
        softening_sq,
        G,
    )


def _pair_accel_cvjp_fwd(
    target_positions,
    source_positions,
    source_masses,
    target_mask_f,
    source_mask_f,
    softening_sq,
    G,
):
    # The residual carries only the O(B*W) INPUTS; the O(B*Wt*Ws) pair
    # intermediates are rematerialized in the reverse pass. Storing
    # (diff, inv_dist3, inv_dist5) instead cost 5 doubles per particle PAIR, and
    # because the bucketed near-field drives this kernel from a ``lax.scan`` over
    # edge chunks, reverse mode retained EVERY chunk's residual -- i.e.
    # ``40 * near_field_edges * max_leaf_size**2`` bytes in total, independent of
    # ``edge_chunk_size``. Measured: a single 52.14 GiB ``diff`` allocation at
    # N=65536 / leaf 64 on a 40 GB A100, versus 1.13 GB for the whole forward.
    # Recomputing costs one extra pair pass inside a backward that already
    # materializes a (B, Wt, Ws, 3) array, and keeps the reverse residual
    # O(edges * max_leaf_size) so galaxy-scale N fits.
    accels = _pair_accel_masked_accels(
        target_positions,
        source_positions,
        source_masses,
        target_mask_f > 0.5,
        source_mask_f > 0.5,
        softening_sq,
        G,
    )
    residual = (
        target_positions,
        source_positions,
        source_masses,
        target_mask_f,
        source_mask_f,
        jnp.asarray(softening_sq),
        jnp.asarray(G),
    )
    return accels, residual


def _pair_accel_cvjp_bwd(residual, cotangent):
    (
        target_positions,
        source_positions,
        source_masses,
        target_mask_f,
        source_mask_f,
        softening_sq,
        G,
    ) = residual
    # Rematerialize the pair intermediates the forward deliberately did not save
    # (see _pair_accel_cvjp_fwd). Same expressions, so the analytic reverse below
    # is unchanged bit-for-bit.
    diff, inv_dist3, inv_dist5 = _pair_accel_pair_terms(
        target_positions,
        source_positions,
        target_mask_f > 0.5,
        source_mask_f > 0.5,
        softening_sq,
    )
    # Forward masks accels by target_mask, so mask the incoming cotangent alike.
    cot = jnp.where(target_mask_f[..., None] > 0.5, cotangent, 0.0)  # (B, Wt, 3)
    cd = jnp.sum(cot[:, :, None, :] * diff, axis=-1)  # (B, Wt, Ws) = c_t · r_ts
    m = source_masses[:, None, :, None]  # (B, 1, Ws, 1)
    # per-pair P_{t,s,l} = m_s [ inv_dist3 c_{t,l} - 3 inv_dist5 (c_t·r) r_l ]
    pair = m * (
        inv_dist3[..., None] * cot[:, :, None, :]
        - 3.0 * inv_dist5[..., None] * cd[..., None] * diff
    )  # (B, Wt, Ws, 3)
    target_positions_bar = -G * jnp.sum(pair, axis=2)  # sum over sources
    source_positions_bar = G * jnp.sum(pair, axis=1)  # sum over targets (3rd law)
    source_masses_bar = -G * jnp.sum(inv_dist3 * cd, axis=1)  # sum over targets
    # Zero cotangents for the non-differentiated inputs (masks, softening, G).
    return (
        target_positions_bar,
        source_positions_bar,
        source_masses_bar,
        jnp.zeros_like(target_mask_f),
        jnp.zeros_like(source_mask_f),
        jnp.zeros_like(softening_sq),
        jnp.zeros_like(G),
    )


_pair_accel_cvjp.defvjp(_pair_accel_cvjp_fwd, _pair_accel_cvjp_bwd)


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
    if _ANALYTIC_P2P_VJP and not compute_potential:
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
    """JIT near-field kernel over leaf-neighbor particle interactions."""
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
                            pot_out = _scatter_scalars_with_schedule(  # type: ignore[arg-type]
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
                            pot_out = _scatter_scalar_contributions(  # type: ignore[arg-type]
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

                    masked_acc = jnp.where(
                        target_mask[:, None],
                        pair_accel,
                        0.0,
                    )
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

                    masked_acc = jnp.where(
                        target_mask[:, None],
                        pair_accel,
                        0.0,
                    )
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


@partial(
    jax.jit,
    static_argnames=(
        "target_leaf_batch_size",
        "source_slot_tile_size",
        "source_slot_scan_unroll",
        "target_batch_scan_unroll",
    ),
)
def _compute_radix_fast_lane_payload_pairs_impl(
    positions: Array,
    masses: Array,
    target_particle_ids: Array,
    target_particle_mask: Array,
    source_particle_ids: Array,
    source_particle_mask: Array,
    source_slot_valid_mask: Array,
    *,
    G: Union[float, Array],
    softening_sq: Array,
    target_leaf_batch_size: int,
    source_slot_tile_size: int,
    source_slot_scan_unroll: int,
    target_batch_scan_unroll: int,
) -> Array:
    """Dense payload-driven pair kernel for radix fast-lane nearfield."""
    dtype = positions.dtype
    g_const = jnp.asarray(G, dtype=dtype)

    num_target_leaves = int(target_particle_ids.shape[0])
    target_leaf_size = int(target_particle_ids.shape[1])
    num_source_slots = int(source_particle_ids.shape[1])
    source_leaf_size = int(source_particle_ids.shape[2])

    if (
        num_target_leaves == 0
        or target_leaf_size == 0
        or num_source_slots == 0
        or source_leaf_size == 0
    ):
        return jnp.zeros_like(positions)

    leaf_batch = int(target_leaf_batch_size)
    if leaf_batch <= 0:
        raise ValueError("target_leaf_batch_size must be positive")
    slot_tile = int(source_slot_tile_size)
    if slot_tile <= 0:
        raise ValueError("source_slot_tile_size must be positive")
    slot_unroll = int(source_slot_scan_unroll)
    if slot_unroll <= 0:
        raise ValueError("source_slot_scan_unroll must be positive")
    batch_unroll = int(target_batch_scan_unroll)
    if batch_unroll <= 0:
        raise ValueError("target_batch_scan_unroll must be positive")

    leaf_batch_offsets = jnp.arange(leaf_batch, dtype=INDEX_DTYPE)
    source_slot_offsets = jnp.arange(slot_tile, dtype=INDEX_DTYPE)
    n_source_tiles = (num_source_slots + slot_tile - 1) // slot_tile
    source_tile_starts = jnp.arange(
        0,
        n_source_tiles * slot_tile,
        slot_tile,
        dtype=INDEX_DTYPE,
    )

    def _batch_body(batch_start):
        target_leaf_ids = batch_start + leaf_batch_offsets
        target_active = target_leaf_ids < num_target_leaves
        safe_target_leaf_ids = jnp.where(target_active, target_leaf_ids, 0)

        batch_target_ids = target_particle_ids[safe_target_leaf_ids]
        batch_target_mask = (
            target_particle_mask[safe_target_leaf_ids] & target_active[:, None]
        )
        batch_target_pos = positions[batch_target_ids]

        batch_source_ids_all = source_particle_ids[safe_target_leaf_ids]
        batch_source_mask_all = source_particle_mask[safe_target_leaf_ids]
        batch_source_slot_valid_all = (
            source_slot_valid_mask[safe_target_leaf_ids] & target_active[:, None]
        )

        flat_target_pos_base = jnp.reshape(
            jnp.broadcast_to(
                batch_target_pos[:, None, :, :],
                (leaf_batch, slot_tile, target_leaf_size, 3),
            ),
            (leaf_batch * slot_tile, target_leaf_size, 3),
        )
        flat_target_mask_base = jnp.reshape(
            jnp.broadcast_to(
                batch_target_mask[:, None, :],
                (leaf_batch, slot_tile, target_leaf_size),
            ),
            (leaf_batch * slot_tile, target_leaf_size),
        )

        def _source_tile_body(local_acc, slot_start):
            slot_ids = slot_start + source_slot_offsets
            in_slot = slot_ids < num_source_slots
            safe_slot_ids = jnp.where(in_slot, slot_ids, 0)

            tile_source_ids = batch_source_ids_all[:, safe_slot_ids, :]
            tile_slot_valid = (
                batch_source_slot_valid_all[:, safe_slot_ids] & in_slot[None, :]
            )
            tile_source_mask = (
                batch_source_mask_all[:, safe_slot_ids, :] & tile_slot_valid[:, :, None]
            )

            src_pos = positions[tile_source_ids]
            src_mass = masses[tile_source_ids]

            flat_src_pos = src_pos.reshape(
                (leaf_batch * slot_tile, source_leaf_size, 3)
            )
            flat_src_mass = src_mass.reshape((leaf_batch * slot_tile, source_leaf_size))
            flat_src_mask = tile_source_mask.reshape(
                (leaf_batch * slot_tile, source_leaf_size)
            )
            flat_slot_valid = tile_slot_valid.reshape((leaf_batch * slot_tile,))
            flat_target_mask = flat_target_mask_base & flat_slot_valid[:, None]

            pair_acc, _ = _pair_contributions_batched(
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
                pair_acc.reshape((leaf_batch, slot_tile, target_leaf_size, 3)),
                axis=1,
            )
            return local_acc + tile_acc, None

        target_leaf_acc, _ = lax.scan(
            _source_tile_body,
            jnp.zeros((leaf_batch, target_leaf_size, 3), dtype=dtype),
            source_tile_starts,
            unroll=slot_unroll,
        )
        return jnp.where(target_active[:, None, None], target_leaf_acc, 0.0)

    acc_leaf_major = _collect_target_leaf_batch_acc(
        num_target_leaves,
        target_leaf_size,
        target_leaf_batch_size=leaf_batch,
        batch_scan_unroll=batch_unroll,
        batch_body=_batch_body,
    )

    accelerations = jnp.zeros_like(positions)
    return _scatter_contributions(
        accelerations,
        target_particle_ids,
        acc_leaf_major,
        target_particle_mask,
    )


def _compute_leaf_p2p_prepared_large_n_self_only_with_potential_impl(
    positions: Array,
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    *,
    G: Union[float, Array],
    softening_sq: Array,
) -> Tuple[Array, Array]:
    """Self-leaf accel + potential portion of the large-N kernel."""
    dtype = positions.dtype
    g_const = jnp.asarray(G, dtype=dtype)
    accelerations = jnp.zeros_like(positions)
    potentials = jnp.zeros(positions.shape[:1], dtype=dtype)
    self_accel, self_pot = _self_contributions(
        leaf_positions,
        leaf_masses,
        leaf_mask,
        softening_sq=softening_sq,
        G=g_const,
        compute_potential=True,
    )
    acc = _scatter_contributions(
        accelerations,
        leaf_particle_idx,
        self_accel,
        leaf_mask,
    )
    pot = _scatter_scalar_contributions(
        potentials,
        leaf_particle_idx,
        self_pot,
        leaf_mask,
    )
    return acc, pot


def _radix_fast_lane_pairs_pallas(
    positions: Array,
    masses: Array,
    target_particle_ids: Array,
    target_particle_mask: Array,
    source_particle_ids: Array,
    source_particle_mask: Array,
    *,
    G: Union[float, Array],
    softening_sq: Array,
    compute_potential: bool,
    num_warps: Optional[int] = None,
    num_stages: int = 1,
    target_subtile: Optional[int] = None,
    interpret: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """Fused Pallas cross-leaf pair path for the radix fast lane.

    Gathers leaf-major target/source tensors, evaluates the fused leaf kernel
    (no HBM ``W x W`` distance matrix), then scatters the leaf-major result back
    to particle order via the existing scatter helpers.  The intra-leaf self
    term is handled separately by the caller, matching the pure-JAX path.
    """
    from jaccpot.pallas.nearfield_fused_leaf import nearfield_fused_leaf_pallas

    dtype = positions.dtype
    g_const = jnp.asarray(G, dtype=dtype)

    num_target_leaves = int(target_particle_ids.shape[0])
    target_leaf_size = int(target_particle_ids.shape[1])
    num_source_slots = int(source_particle_ids.shape[1])
    source_leaf_size = int(source_particle_ids.shape[2])
    num_sources = num_source_slots * source_leaf_size

    accelerations = jnp.zeros_like(positions)
    if num_target_leaves == 0 or target_leaf_size == 0 or num_sources == 0:
        if compute_potential:
            return accelerations, jnp.zeros(positions.shape[:1], dtype=dtype)
        return accelerations

    safe_target_ids = jnp.where(target_particle_mask, target_particle_ids, 0)
    tgt_pos = positions[safe_target_ids]

    safe_source_ids = jnp.reshape(
        jnp.where(source_particle_mask, source_particle_ids, 0),
        (num_target_leaves, num_sources),
    )
    src_pos = positions[safe_source_ids]
    src_mass = masses[safe_source_ids]
    src_mask_flat = jnp.reshape(source_particle_mask, (num_target_leaves, num_sources))

    out = nearfield_fused_leaf_pallas(
        tgt_pos,
        target_particle_mask,
        src_pos,
        src_mass,
        src_mask_flat,
        softening_sq=softening_sq,
        G=g_const,
        num_warps=num_warps,
        num_stages=num_stages,
        target_subtile=target_subtile,
        interpret=interpret,
    )

    pair_acc = _scatter_contributions(
        accelerations, target_particle_ids, out[..., :3], target_particle_mask
    )
    if compute_potential:
        potentials = jnp.zeros(positions.shape[:1], dtype=dtype)
        pair_pot = _scatter_scalar_contributions(
            potentials, target_particle_ids, out[..., 3], target_particle_mask
        )
        return pair_acc, pair_pot
    return pair_acc


def _radix_fast_lane_prepacked_pallas(
    source_leaf_ids_padded: Array,
    source_valid_mask_padded: Array,
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    positions: Array,
    *,
    G: Union[float, Array],
    softening_sq: Array,
    compute_potential: bool,
    num_warps: Optional[int] = None,
    num_stages: int = 1,
    target_subtile: Optional[int] = None,
    interpret: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """Fused Pallas leaf-pair path over the compact prepacked source-leaf layout.

    Consumes the ``(num_leaves, max_blocks, block_size)`` source-leaf-id tensors
    used by the production fused near-field lane. Source leaves are gathered by
    id inside the kernel (no dense per-particle source materialization), then the
    leaf-major result is scattered to particle order.  The intra-leaf self term
    is handled separately by the caller, matching the pure-JAX path.
    """
    from jaccpot.pallas.nearfield_fused_leaf import nearfield_leafpair_pallas

    dtype = positions.dtype
    g_const = jnp.asarray(G, dtype=dtype)

    num_leaves = int(source_leaf_ids_padded.shape[0])
    num_source_slots = int(source_leaf_ids_padded.shape[1]) * int(
        source_leaf_ids_padded.shape[2]
    )

    accelerations = jnp.zeros_like(positions)
    if num_leaves == 0 or num_source_slots == 0 or int(leaf_positions.shape[1]) == 0:
        if compute_potential:
            return accelerations, jnp.zeros(positions.shape[:1], dtype=dtype)
        return accelerations

    source_leaf_ids_flat = source_leaf_ids_padded.reshape(
        (num_leaves, num_source_slots)
    )
    source_valid_flat = source_valid_mask_padded.reshape((num_leaves, num_source_slots))

    out = nearfield_leafpair_pallas(
        leaf_positions,
        leaf_masses,
        leaf_mask,
        source_leaf_ids_flat,
        source_valid_flat,
        softening_sq=softening_sq,
        G=g_const,
        num_warps=num_warps,
        num_stages=num_stages,
        target_subtile=target_subtile,
        interpret=interpret,
    )

    pair_acc = _scatter_contributions(
        accelerations, leaf_particle_idx, out[..., :3], leaf_mask
    )
    if compute_potential:
        potentials = jnp.zeros(positions.shape[:1], dtype=dtype)
        pair_pot = _scatter_scalar_contributions(
            potentials, leaf_particle_idx, out[..., 3], leaf_mask
        )
        return pair_acc, pair_pot
    return pair_acc


def _radix_fast_lane_prepacked_pallas_decoupled(
    source_leaf_ids_padded: Array,
    source_valid_mask_padded: Array,
    target_positions: Array,
    target_mask: Array,
    target_particle_idx: Array,
    source_positions: Array,
    source_masses: Array,
    source_mask: Array,
    positions: Array,
    *,
    G: Union[float, Array],
    softening_sq: Array,
    compute_potential: bool = False,
    num_warps: Optional[int] = None,
    num_stages: int = 1,
    target_subtile: Optional[int] = None,
    interpret: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """Decoupled twin of :func:`_radix_fast_lane_prepacked_pallas`.

    The TARGET leaves (``target_positions``/``target_mask``/``target_particle_idx``, a block of
    ``[num_targets, W, *]``) are separate from the full SOURCE gather pool
    (``source_positions``/``source_masses``/``source_mask``, ``[num_sources, W, *]``). Source-leaf
    ids in ``source_leaf_ids_padded`` reference global source rows in ``[0, num_sources)``. This
    lets a caller run one block of target leaves against the full source pool (near-field leaf-block
    chunking). Scatters the block's per-particle accel into a ``zeros_like(positions)`` buffer via
    the global ``target_particle_idx`` (scatter-add) so per-block partials compose by ``+``. Target
    masses are unused by the pair term. The intra-leaf self term is handled separately by the caller.
    """
    from jaccpot.pallas.nearfield_fused_leaf import (
        nearfield_leafpair_pallas_decoupled,
    )

    dtype = positions.dtype
    g_const = jnp.asarray(G, dtype=dtype)

    num_targets = int(source_leaf_ids_padded.shape[0])
    num_source_slots = int(source_leaf_ids_padded.shape[1]) * int(
        source_leaf_ids_padded.shape[2]
    )

    accelerations = jnp.zeros_like(positions)
    empty = (
        num_targets == 0
        or num_source_slots == 0
        or int(target_positions.shape[1]) == 0
        or int(source_positions.shape[0]) == 0
    )
    if empty:
        if compute_potential:
            return accelerations, jnp.zeros(positions.shape[:1], dtype=dtype)
        return accelerations

    source_leaf_ids_flat = source_leaf_ids_padded.reshape(
        (num_targets, num_source_slots)
    )
    source_valid_flat = source_valid_mask_padded.reshape(
        (num_targets, num_source_slots)
    )

    out = nearfield_leafpair_pallas_decoupled(
        target_positions,
        target_mask,
        source_positions,
        source_masses,
        source_mask,
        source_leaf_ids_flat,
        source_valid_flat,
        softening_sq=softening_sq,
        G=g_const,
        num_warps=num_warps,
        num_stages=num_stages,
        target_subtile=target_subtile,
        interpret=interpret,
    )

    pair_acc = _scatter_contributions(
        accelerations, target_particle_idx, out[..., :3], target_mask
    )
    if compute_potential:
        potentials = jnp.zeros(positions.shape[:1], dtype=dtype)
        pair_pot = _scatter_scalar_contributions(
            potentials, target_particle_idx, out[..., 3], target_mask
        )
        return pair_acc, pair_pot
    return pair_acc


_FLOAT32_EXACT_INT_LIMIT = 2**24


def _check_float_id_range(num_particles: int, dtype: Any, *, what: str) -> None:
    """Refuse float id encoding when the ids would not be exactly representable.

    ``custom_vjp`` rules must not close over tracers, so index/mask arrays have to
    be passed as ordinary arguments -- and integer args would yield ``float0``
    cotangents rather than plain zeros. The differentiable lanes therefore encode
    ids as floats (reconstructed with ``round().astype``), which is exact only
    below the mantissa bound.
    """
    if (
        jnp.dtype(dtype) == jnp.float32
        and int(num_particles) > _FLOAT32_EXACT_INT_LIMIT
    ):
        raise ValueError(
            f"{what}: float32 represents integers exactly only up to "
            f"{_FLOAT32_EXACT_INT_LIMIT} but N={int(num_particles)}. The "
            "differentiable near-field lane encodes particle ids as floats (a "
            "custom_vjp rule may not close over tracers, and integer args would "
            "yield float0 cotangents), so ids beyond that bound would be silently "
            "rounded. Use float64, or split the evaluation."
        )


_GRAD_REV_TIER_MAX = _env_int("JACCPOT_GRAD_REV_TIERS", 4)
# Minimum predicted slot-visit reduction before tiering is worth its
# indirection. See build_leafpair_reverse_tiers for the measurements behind 3.0.
_GRAD_REV_TIER_MIN_GAIN = float(
    os.environ.get("JACCPOT_GRAD_REV_TIER_MIN_GAIN", "3.0") or 3.0
)


def build_leafpair_reverse_tiers(
    source_valid: Any,
    *,
    slot_tile: int,
    max_tiers: Optional[int] = None,
) -> Optional[Tuple[Tuple[Tuple[int, ...], int], ...]]:
    """Partition leaves into static occupancy tiers for the analytic reverse.

    The prepacked payload is a rectangle whose width is the GLOBAL maximum
    neighbour count. That maximum comes from a single pathological leaf (measured:
    exactly ``leaves - 1``, i.e. it neighbours every other leaf, at every N and
    every geometry tried), while the median leaf uses a fraction of it -- fill is
    45% at N=200000 and 14.5% at N=1000000. Every leaf therefore pays the worst
    leaf's width.

    This splits the leaves into a few groups by how many slots they actually use,
    so each group's reverse pass reads only its own width. Two properties matter:

    * widths are **static** (computed here, on the host, from the frozen topology),
      so each pass compiles to a right-sized ``lax.scan``; and
    * leaves keep their **Morton order within a tier** -- only the visiting order
      across tiers changes, and source ids are never renumbered. A global
      occupancy sort was tried instead and ran ~7x slower precisely because it
      broke that locality.

    Returns ``None`` when tiering cannot pay (one tier, or no usable split), which
    keeps the single-pass path byte-identical. Leaf ids come back as plain int
    tuples so the result is hashable -- it rides through the ``custom_vjp`` in
    ``nondiff_argnums``, which JAX requires to be hashable and comparable.
    """
    valid = np.asarray(jax.device_get(source_valid))
    num_leaves = int(valid.shape[0])
    if num_leaves == 0:
        return None
    slots = int(np.prod(valid.shape[1:]))
    counts = valid.reshape(num_leaves, slots).sum(axis=1)
    tile = max(1, int(slot_tile))
    # Round each leaf's occupancy up to a whole tile: that is the width its pass
    # must read.
    widths = (np.maximum(counts, 1) + tile - 1) // tile * tile
    limit = int(_GRAD_REV_TIER_MAX if max_tiers is None else max_tiers)
    if limit <= 1:
        return None

    unique = np.unique(widths)
    if unique.size <= 1:
        return None  # uniform occupancy: the rectangle is already right-sized
    # Geometric tier edges: cheap, and matches how occupancy is distributed (a long
    # tail of near-empty leaves under one very wide outlier).
    edges = np.unique(
        np.round(
            np.geomspace(max(unique[0], tile), unique[-1], num=min(limit, unique.size))
        ).astype(np.int64)
    )
    edges = (edges + tile - 1) // tile * tile

    tiers: list[tuple[tuple[int, ...], int]] = []
    lower = 0
    for edge in edges:
        members = np.nonzero((widths > lower) & (widths <= edge))[0]
        if members.size:
            # np.nonzero returns ascending indices => Morton order preserved.
            tiers.append((tuple(int(i) for i in members), int(min(edge, slots))))
        lower = int(edge)
    if not tiers or len(tiers) == 1:
        return None
    covered = sum(len(t[0]) for t in tiers)
    if covered != num_leaves:  # defensive: never silently drop a leaf
        raise ValueError(
            f"reverse tiering covered {covered} of {num_leaves} leaves; "
            "every leaf must appear in exactly one tier"
        )
    # Accept only when the slot-visit saving is big enough to pay for the extra
    # indirection. Tiering makes the target-side gather go through an index array
    # instead of a consecutive range, and splits one scan into several smaller
    # ones; both cost throughput, and at small leaf counts the per-pass fixed cost
    # is amortised over very little work. MEASURED, near-field reverse only,
    # contention-corrected against the untouched far-field row:
    #
    #   N=200000  (782 leaves): predicted 1.64x fewer slot visits -> 4.3x SLOWER
    #   N=1000000 (3907 leaves): predicted 4.33x fewer slot visits -> 1.91x FASTER
    #
    # So the break-even predicted reduction lies between those; 3.0 is the
    # conservative pick. Calibrated on exactly two points on one A100 -- re-measure
    # before trusting it on other hardware or a very different occupancy profile.
    tiered_slots = sum(len(t[0]) * int(t[1]) for t in tiers)
    if tiered_slots <= 0:
        return None
    reduction = float(num_leaves * slots) / float(tiered_slots)
    if reduction < _GRAD_REV_TIER_MIN_GAIN:
        return None
    return tuple(tiers)


def _leafpair_accel_analytic_vjp(
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    source_leaf_ids: Array,
    source_valid: Array,
    cotangent: Array,
    *,
    softening_sq: Array,
    G: Array,
    leaf_batch: int,
    slot_tile: int,
    skip_empty_tiles: bool = True,
    occupancy_sort: bool = False,
    tiers: Optional[Tuple[Tuple[Any, int], ...]] = None,
) -> Tuple[Array, Array]:
    """Analytic reverse of the leaf-pair near field in **O(N) memory**.

    Returns ``(leaf_positions_bar, leaf_masses_bar)`` for a particle-order output
    ``cotangent``.

    Why hand-written rather than ``jax.vjp`` of a pure-JAX twin: a ``bwd`` rule is
    never itself differentiated, so everything it computes is a *transient* bounded
    by the tile size instead of a *residual* retained per scan iteration. Taking
    the reverse from ``jax.vjp(twin)`` is correct but linearizes the twin, which
    reinstates the twin's own O(edges * W) scan residuals as a peak during the
    backward -- ~67 GB at N=1048576 on the canonical leaf-256 config. This form
    keeps only O(N) accumulators, which is what makes 1M gradients reachable.

    The contraction is the same symmetric near-field tidal tensor already used by
    :func:`_pair_accel_cvjp_bwd`,
    ``J = -G sum_s m_s (I/r^3 - 3 r r^T / r^5)`` with ``J^T c = J c``, just applied
    per leaf pair and accumulated leaf-major so no particle-order scatter is
    needed (cotangents are w.r.t. the leaf-major tensors; the caller's gather
    transposes them back to particle order).

    Newton's third law supplies the source-side term: the contribution a target
    leaf receives from a source leaf gives equal and opposite position sensitivity
    to that source leaf, scattered by source leaf id.
    """
    dtype = leaf_positions.dtype
    num_leaves = int(leaf_positions.shape[0])
    width = int(leaf_positions.shape[1])
    num_slots = int(source_leaf_ids.shape[1]) * int(source_leaf_ids.shape[2])
    if num_leaves == 0 or width == 0 or num_slots == 0:
        return jnp.zeros_like(leaf_positions), jnp.zeros_like(leaf_masses)

    slot_ids = jnp.reshape(source_leaf_ids, (num_leaves, num_slots))
    slot_valid = jnp.reshape(source_valid, (num_leaves, num_slots))

    # Cotangent, gathered leaf-major and masked exactly as the forward masks its
    # output. O(N).
    cot_leaf = jnp.where(
        leaf_mask[..., None], cotangent[leaf_particle_idx], jnp.zeros((), dtype)
    )

    # DO NOT occupancy-SORT here (permuting the leaf arrays themselves). Tried and
    # measured **~6.8x SLOWER** at N=200000 (near-only reverse 1632 -> 20987 ms,
    # after dividing out a 1.9x GPU-contention factor read off the untouched
    # far-field row) and a wash at 1M. Leaves arrive in Morton order, which keeps
    # the per-tile source gather ``leaf_positions[safe_src]`` and the
    # ``.at[safe_src].add`` scatter spatially coherent; a global permutation
    # destroys that and the extra memory traffic dwarfs the arithmetic saved.
    #
    # ``tiers`` is the salvaged version of that idea. It changes only WHICH TARGET
    # leaves each pass visits and how many slots that pass reads -- source ids are
    # never renumbered and ``leaf_positions`` is never permuted, so source-side
    # locality is untouched. Each tier is a (leaf-index array, slot width) pair with
    # STATIC width, built on the host from the occupancy histogram, and leaves stay
    # in Morton order within a tier. A tier of near-empty leaves then reads only its
    # own narrow slot window instead of the global maximum, which is what actually
    # removes the padding: measured fill is 45% at N=200000 and 14.5% at N=1000000,
    # i.e. most of the rectangle is padding set by a single pathological leaf.
    del occupancy_sort

    batch = max(1, min(int(leaf_batch), num_leaves))
    tile = max(1, min(int(slot_tile), num_slots))
    leaf_offsets = jnp.arange(batch, dtype=INDEX_DTYPE)
    slot_offsets = jnp.arange(tile, dtype=INDEX_DTYPE)

    def _pass(carry, tier_leaves, tier_slots):
        """One occupancy tier: ``tier_leaves`` targets against ``tier_slots`` slots."""
        tier_count = int(tier_leaves.shape[0])
        leaf_starts = jnp.arange(0, tier_count, batch, dtype=INDEX_DTYPE)
        slot_starts = jnp.arange(0, tier_slots, tile, dtype=INDEX_DTYPE)

        def leaf_body(
            carry: Tuple[Array, Array], leaf_start: Array
        ) -> Tuple[Tuple[Array, Array], None]:
            pos_bar, mass_bar = carry
            pos_in_tier = leaf_start + leaf_offsets
            tgt_in_range = pos_in_tier < tier_count
            # Target leaf ids stay GLOBAL: only the visiting order changes.
            safe_tgt = tier_leaves[jnp.where(tgt_in_range, pos_in_tier, 0)]

            tgt_pos = leaf_positions[safe_tgt]  # (B, W, 3)
            tgt_mask = leaf_mask[safe_tgt] & tgt_in_range[:, None]
            cot_t = jnp.where(
                tgt_mask[..., None], cot_leaf[safe_tgt], jnp.zeros((), dtype)
            )

            def slot_body(
                inner: Tuple[Array, Array, Array], slot_start: Array
            ) -> Tuple[Tuple[Array, Array, Array], None]:
                pos_acc, mass_acc, tgt_acc = inner
                sl = slot_start + slot_offsets
                sl_in_range = sl < tier_slots
                safe_sl = jnp.where(sl_in_range, sl, 0)

                src_leaf = slot_ids[safe_tgt][:, safe_sl]  # (B, T)
                valid_slot = slot_valid[safe_tgt][:, safe_sl] & sl_in_range[None, :]
                valid_slot = valid_slot & tgt_in_range[:, None]
                safe_src = jnp.where(valid_slot, src_leaf, 0)

                def _apply(acc_in):
                    pos_in, mass_in, tgt_in = acc_in
                    src_pos = leaf_positions[safe_src]  # (B, T, W, 3)
                    src_mass = leaf_masses[safe_src]  # (B, T, W)
                    src_mask = leaf_mask[safe_src] & valid_slot[..., None]

                    # (B, T, Wt, Ws, 3)
                    diff = tgt_pos[:, None, :, None, :] - src_pos[:, :, None, :, :]
                    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
                    pair_mask = tgt_mask[:, None, :, None] & src_mask[:, :, None, :]
                    safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
                    inv_r = jnp.where(pair_mask, lax.rsqrt(safe_dist_sq), 0.0)
                    inv_dist3 = jnp.where(pair_mask, inv_r * inv_r * inv_r, 0.0)
                    inv_dist5 = jnp.where(pair_mask, inv_dist3 * inv_r * inv_r, 0.0)

                    cot_b = cot_t[:, None, :, None, :]  # (B, 1, Wt, 1, 3)
                    cd = jnp.sum(cot_b * diff, axis=-1)  # (B, T, Wt, Ws)
                    m = src_mass[:, :, None, :, None]  # (B, T, 1, Ws, 1)
                    pair = m * (
                        inv_dist3[..., None] * cot_b
                        - 3.0 * inv_dist5[..., None] * cd[..., None] * diff
                    )

                    # Target side: sum over sources (slots and their particles).
                    tgt_contrib = -G * jnp.sum(pair, axis=(1, 3))  # (B, Wt, 3)
                    # Source side (third law): sum over targets, scattered by source leaf.
                    src_contrib = G * jnp.sum(pair, axis=2)  # (B, T, Ws, 3)
                    src_mass_contrib = -G * jnp.sum(
                        inv_dist3 * cd, axis=2
                    )  # (B, T, Ws)

                    src_contrib = jnp.where(
                        valid_slot[..., None, None], src_contrib, 0.0
                    )
                    src_mass_contrib = jnp.where(
                        valid_slot[..., None], src_mass_contrib, 0.0
                    )

                    return (
                        pos_in.at[safe_src].add(src_contrib),
                        mass_in.at[safe_src].add(src_mass_contrib),
                        tgt_in + tgt_contrib,
                    )

                # Skip whole tiles that carry no valid source slot. Every term above is
                # masked by ``valid_slot``, so an all-invalid tile contributes exactly
                # zero -- this is a pure work saving, not an approximation. It matters
                # because the prepacked payload is padded to the GLOBAL maximum neighbour
                # count (one leaf typically neighbours every other), so the fill is
                # 45% at N=200000 and 14.5% at N=1000000 on the canonical galaxy config:
                # most tiles are pure padding. The forward lane has had this skip all
                # along (``_accumulate_target_block_tile_sequence``); its absence here is
                # why the near-field reverse ran ~18x its own forward.
                if skip_empty_tiles:
                    return (
                        lax.cond(jnp.any(valid_slot), _apply, lambda acc: acc, inner),
                        None,
                    )
                return _apply(inner), None

            (pos_bar, mass_bar, tgt_total), _ = lax.scan(
                slot_body,
                (pos_bar, mass_bar, jnp.zeros_like(tgt_pos)),
                slot_starts,
            )
            tgt_total = jnp.where(tgt_in_range[:, None, None], tgt_total, 0.0)
            pos_bar = pos_bar.at[safe_tgt].add(tgt_total)
            return (pos_bar, mass_bar), None

        return lax.scan(leaf_body, carry, leaf_starts)[0]

    carry = (jnp.zeros_like(leaf_positions), jnp.zeros_like(leaf_masses))
    if tiers is None:
        carry = _pass(carry, jnp.arange(num_leaves, dtype=INDEX_DTYPE), num_slots)
    else:
        # Python loop: tier count and each tier's slot width are STATIC, so every
        # pass compiles to its own right-sized scan.
        for tier_leaves, tier_slots in tiers:
            if len(tier_leaves) == 0 or int(tier_slots) == 0:
                continue
            carry = _pass(
                carry,
                jnp.asarray(tier_leaves, dtype=INDEX_DTYPE),
                int(tier_slots),
            )
    positions_bar, masses_bar = carry
    return positions_bar, masses_bar


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13, 14, 15, 16, 17))
def _radix_fast_lane_prepacked_accel_cvjp(
    leaf_positions: Array,
    leaf_masses: Array,
    positions: Array,
    source_leaf_ids_f: Array,
    source_valid_f: Array,
    leaf_mask_f: Array,
    leaf_particle_idx_f: Array,
    softening_sq: Array,
    G: Array,
    num_warps: Optional[int],
    num_stages: int,
    target_subtile: Optional[int],
    interpret: bool,
    rev_leaf_batch: int,
    rev_block_tile: int,
    rev_tile_unroll: int,
    rev_batch_unroll: int,
    rev_tiers: Optional[Tuple[Tuple[Tuple[int, ...], int], ...]],
) -> Array:
    """Differentiable prepacked-lane near field: Pallas forward, tiled-twin reverse.

    The forward IS the production fused Pallas call, so it is byte-identical and
    the grad path runs exactly the kernel the forward ships. The reverse is
    ``jax.vjp`` of the lane's own tiled pure-JAX fallback
    (:func:`_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl`),
    which is already routed through the rematerialized analytic P2P rule.

    Using the tiled fallback -- not the dense reference twin in
    ``jaccpot/pallas/nearfield_fused_leaf.py`` -- is essential: that reference
    materialises a ``(leaves, W_t, K, 3)`` difference tensor, which is ~50 TB at
    the fiducial configuration and is documented "test-scale only". It stays in the
    tree as the unit-level VJP oracle, not as a production reverse.

    Measured Pallas-vs-fallback force agreement at the canonical config: rel-L2
    4.2e-06 at N=65536 and 8.4e-06 at N=200000, i.e. fp32 summation reordering.
    That check is the gate on this whole approach -- the reverse is the derivative
    of the fallback, so if the two lanes disagreed materially we would be
    differentiating a different function (cf. the grouped-M2L lesson, where an
    assumed equivalence was really 7.1e-02 vs 1.1e-02 against the exact sum).

    Reverse tiling is deliberately independent of the forward's: the backward
    materialises per-tile pair tensors, so small ``rev_*`` tiles keep it bounded.
    """
    return _radix_fast_lane_prepacked_pallas(
        jnp.round(source_leaf_ids_f).astype(INDEX_DTYPE),
        source_valid_f > 0.5,
        leaf_positions,
        leaf_masses,
        leaf_mask_f > 0.5,
        jnp.round(leaf_particle_idx_f).astype(INDEX_DTYPE),
        positions,
        G=G,
        softening_sq=softening_sq,
        compute_potential=False,
        num_warps=num_warps,
        num_stages=num_stages,
        target_subtile=target_subtile,
        interpret=interpret,
    )


def _radix_fast_lane_prepacked_accel_fwd(
    leaf_positions,
    leaf_masses,
    positions,
    source_leaf_ids_f,
    source_valid_f,
    leaf_mask_f,
    leaf_particle_idx_f,
    softening_sq,
    G,
    num_warps,
    num_stages,
    target_subtile,
    interpret,
    rev_leaf_batch,
    rev_block_tile,
    rev_tile_unroll,
    rev_batch_unroll,
    rev_tiers,
):
    out = _radix_fast_lane_prepacked_accel_cvjp(
        leaf_positions,
        leaf_masses,
        positions,
        source_leaf_ids_f,
        source_valid_f,
        leaf_mask_f,
        leaf_particle_idx_f,
        softening_sq,
        G,
        num_warps,
        num_stages,
        target_subtile,
        interpret,
        rev_leaf_batch,
        rev_block_tile,
        rev_tile_unroll,
        rev_batch_unroll,
        rev_tiers,
    )
    residual = (
        leaf_positions,
        leaf_masses,
        positions,
        source_leaf_ids_f,
        source_valid_f,
        leaf_mask_f,
        leaf_particle_idx_f,
        jnp.asarray(softening_sq),
        jnp.asarray(G),
    )
    return out, residual


def _radix_fast_lane_prepacked_accel_bwd(
    num_warps,
    num_stages,
    target_subtile,
    interpret,
    rev_leaf_batch,
    rev_block_tile,
    rev_tile_unroll,
    rev_batch_unroll,
    rev_tiers,
    residual,
    cotangent,
):
    (
        leaf_positions,
        leaf_masses,
        positions,
        source_leaf_ids_f,
        source_valid_f,
        leaf_mask_f,
        leaf_particle_idx_f,
        softening_sq,
        G,
    ) = residual

    # Analytic reverse, NOT ``jax.vjp`` of the tiled twin. Both are correct and
    # agree to round-off, but ``jax.vjp`` linearizes the twin and so reinstates its
    # O(edges * W) scan residuals as a transient peak during the backward (~67 GB at
    # N=1048576). A hand-written bwd is never differentiated, so its intermediates
    # are tile-bounded transients and the memory is O(N) -- which is what makes
    # galaxy-scale gradients reachable. ``jax.vjp`` of the twin remains the
    # correctness oracle in tests/unit/test_custom_vjp_parity.py.
    leaf_positions_bar, leaf_masses_bar = _leafpair_accel_analytic_vjp(
        leaf_positions,
        leaf_masses,
        leaf_mask_f > 0.5,
        jnp.round(leaf_particle_idx_f).astype(INDEX_DTYPE),
        jnp.round(source_leaf_ids_f).astype(INDEX_DTYPE),
        source_valid_f > 0.5,
        cotangent,
        softening_sq=softening_sq,
        G=G,
        leaf_batch=int(rev_leaf_batch),
        slot_tile=int(rev_block_tile),
        skip_empty_tiles=_env_flag("JACCPOT_GRAD_REV_SKIP_EMPTY_TILES", True),
        tiers=rev_tiers,
    )
    return (
        leaf_positions_bar,
        leaf_masses_bar,
        jnp.zeros_like(positions),
        jnp.zeros_like(source_leaf_ids_f),
        jnp.zeros_like(source_valid_f),
        jnp.zeros_like(leaf_mask_f),
        jnp.zeros_like(leaf_particle_idx_f),
        jnp.zeros_like(softening_sq),
        jnp.zeros_like(G),
    )


_radix_fast_lane_prepacked_accel_cvjp.defvjp(
    _radix_fast_lane_prepacked_accel_fwd, _radix_fast_lane_prepacked_accel_bwd
)


def compute_leaf_p2p_accelerations_radix_fast_lane(
    *,
    positions_sorted: Array,
    masses_sorted: Array,
    payload: Any,
    G: Union[float, Array] = 1.0,
    softening: float = 0.0,
    return_potential: bool = False,
    use_pallas: bool = False,
    differentiable: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """Payload-driven nearfield entry for the radix fast lane.

    ``differentiable`` routes the fused Pallas lanes through their ``custom_vjp``
    wrappers so ``jax.grad`` works. ``pallas_call`` has no autodiff rule, so the
    raw lanes hard-fail in reverse mode; the wrapper keeps the Pallas forward
    (byte-identical) and takes the reverse from this lane's tiled pure-JAX
    fallback. Defaults False so the production forward path is untouched.
    """
    positions = jnp.asarray(positions_sorted)
    masses = jnp.asarray(masses_sorted)
    want_potential = bool(return_potential)
    dtype = positions.dtype

    target_particle_ids = jnp.asarray(payload.target_particle_ids, dtype=INDEX_DTYPE)
    target_particle_mask = jnp.asarray(payload.target_particle_mask, dtype=bool)
    source_particle_ids = jnp.asarray(payload.source_particle_ids, dtype=INDEX_DTYPE)
    source_particle_mask = jnp.asarray(payload.source_particle_mask, dtype=bool)

    # Decide whether a fused Pallas cross-leaf pair path is usable. Requires a
    # supported GPU (or forced interpret mode for CPU testing). Two layouts:
    #   - materialized per-particle source payload -> pairs kernel;
    #   - compact prepacked source-leaf-id layout (the production fused lane) ->
    #     leaf-pair kernel that gathers source leaves by id.
    pallas_interpret = _env_flag("JACCPOT_NEARFIELD_PALLAS_INTERPRET", False)
    pallas_available = False
    if bool(use_pallas):
        from jaccpot.pallas.nearfield_fused_leaf import (
            pallas_nearfield_fused_supported,
        )

        pallas_available = pallas_interpret or pallas_nearfield_fused_supported()
    has_materialized_sources = int(source_particle_ids.size) > 0
    has_prepacked_sources = (
        int(source_particle_ids.size) == 0
        and int(jnp.asarray(payload.source_leaf_ids).size) > 0
    )
    pallas_pairs = pallas_available and has_materialized_sources
    pallas_prepacked = pallas_available and has_prepacked_sources

    # Potential is only implemented on the fused Pallas paths; otherwise the
    # caller falls back to the generic W x W path (preserving prior behavior).
    if want_potential and not (pallas_pairs or pallas_prepacked):
        raise NotImplementedError(
            "compute_leaf_p2p_accelerations_radix_fast_lane supports "
            "return_potential=True only on the fused Pallas paths "
            "(use_pallas=True on a supported GPU)"
        )

    def _zeros_result():
        acc = jnp.zeros_like(positions)
        if want_potential:
            return acc, jnp.zeros(positions.shape[:1], dtype=dtype)
        return acc

    if int(target_particle_ids.size) == 0:
        return _zeros_result()

    safe_target_particle_ids = jnp.where(target_particle_mask, target_particle_ids, 0)
    leaf_positions = positions[safe_target_particle_ids]
    leaf_masses = masses[safe_target_particle_ids]
    leaf_mask = target_particle_mask
    leaf_particle_idx = safe_target_particle_ids

    diag_mode = _large_n_nearfield_diag_mode()
    if diag_mode == "zero":
        return _zeros_result()

    softening_sq = jnp.asarray(float(softening) ** 2, dtype=positions.dtype)
    self_acc = jnp.zeros_like(positions)
    self_pot = jnp.zeros(positions.shape[:1], dtype=dtype)
    if diag_mode != "pairs_only":
        if want_potential:
            (
                self_acc,
                self_pot,
            ) = _compute_leaf_p2p_prepared_large_n_self_only_with_potential_impl(
                positions,
                leaf_positions,
                leaf_masses,
                leaf_mask,
                leaf_particle_idx,
                G=G,
                softening_sq=softening_sq,
            )
        else:
            self_acc = _compute_leaf_p2p_prepared_large_n_self_only_impl(
                positions,
                leaf_positions,
                leaf_masses,
                leaf_mask,
                leaf_particle_idx,
                G=G,
                softening_sq=softening_sq,
            )
    if diag_mode == "self_only":
        if want_potential:
            return self_acc, self_pot
        return self_acc

    if pallas_pairs:
        pallas_num_warps = _env_int("JACCPOT_NEARFIELD_PALLAS_NUM_WARPS", 0)
        pallas_num_stages = max(1, _env_int("JACCPOT_NEARFIELD_PALLAS_NUM_STAGES", 1))
        pallas_subtile = _env_int("JACCPOT_NEARFIELD_PALLAS_TARGET_SUBTILE", 0)
        pairs_result = _radix_fast_lane_pairs_pallas(
            positions,
            masses,
            target_particle_ids,
            target_particle_mask,
            source_particle_ids,
            source_particle_mask,
            G=G,
            softening_sq=softening_sq,
            compute_potential=want_potential,
            num_warps=(pallas_num_warps if pallas_num_warps > 0 else None),
            num_stages=pallas_num_stages,
            target_subtile=(pallas_subtile if pallas_subtile > 0 else None),
            interpret=pallas_interpret,
        )
        if want_potential:
            pair_acc, pair_pot = pairs_result
            return self_acc + pair_acc, self_pot + pair_pot
        return self_acc + pairs_result

    if int(source_particle_ids.size) == 0:
        # Prepacked source-leaf-id layout (the production fused near-field lane).
        source_leaf_ids_padded = jnp.asarray(payload.source_leaf_ids, dtype=INDEX_DTYPE)
        source_valid_mask_padded = jnp.asarray(
            payload.source_leaf_valid_mask, dtype=bool
        )

        if pallas_prepacked:
            pallas_num_warps = _env_int("JACCPOT_NEARFIELD_PALLAS_NUM_WARPS", 0)
            pallas_num_stages = max(
                1, _env_int("JACCPOT_NEARFIELD_PALLAS_NUM_STAGES", 1)
            )
            pallas_subtile = _env_int("JACCPOT_NEARFIELD_PALLAS_TARGET_SUBTILE", 0)
            if differentiable and not want_potential:
                # Differentiable prepacked lane: the SAME Pallas forward wrapped in
                # a custom_vjp whose reverse is autodiff of this lane's own tiled
                # pure-JAX fallback. Forward is byte-identical, so the grad path
                # runs exactly the kernel production ships.
                _check_float_id_range(
                    int(positions.shape[0]),
                    dtype,
                    what="differentiable prepacked near-field lane",
                )
                rev_block_tile = _env_int("JACCPOT_GRAD_REV_BLOCK_TILE", 8)
                # Occupancy tiers for the reverse, built HERE rather than inside the
                # bwd rule: the payload's validity mask is frozen topology and is
                # concrete at this point, whereas inside the rule it is a residual
                # (a tracer under jit) from which no static slot width could be
                # read. Rides through as a nondiff arg, so each tier's width is a
                # compile-time constant.
                rev_tiers = None
                if _env_flag("JACCPOT_GRAD_REV_TIERED", True):
                    try:
                        rev_tiers = build_leafpair_reverse_tiers(
                            source_valid_mask_padded, slot_tile=rev_block_tile
                        )
                    except Exception:
                        # Tracer, or anything else non-concrete: fall back to the
                        # single full-width pass, which is always correct.
                        rev_tiers = None
                return self_acc + _radix_fast_lane_prepacked_accel_cvjp(
                    leaf_positions,
                    leaf_masses,
                    positions,
                    source_leaf_ids_padded.astype(dtype),
                    source_valid_mask_padded.astype(dtype),
                    leaf_mask.astype(dtype),
                    leaf_particle_idx.astype(dtype),
                    softening_sq,
                    jnp.asarray(G, dtype=dtype),
                    (pallas_num_warps if pallas_num_warps > 0 else None),
                    pallas_num_stages,
                    (pallas_subtile if pallas_subtile > 0 else None),
                    pallas_interpret,
                    _env_int("JACCPOT_GRAD_REV_LEAF_BATCH", 8),
                    rev_block_tile,
                    1,
                    1,
                    rev_tiers,
                )
            prepacked_result = _radix_fast_lane_prepacked_pallas(
                source_leaf_ids_padded,
                source_valid_mask_padded,
                leaf_positions,
                leaf_masses,
                leaf_mask,
                leaf_particle_idx,
                positions,
                G=G,
                softening_sq=softening_sq,
                compute_potential=want_potential,
                num_warps=(pallas_num_warps if pallas_num_warps > 0 else None),
                num_stages=pallas_num_stages,
                target_subtile=(pallas_subtile if pallas_subtile > 0 else None),
                interpret=pallas_interpret,
            )
            if want_potential:
                pair_acc, pair_pot = prepacked_result
                return self_acc + pair_acc, self_pot + pair_pot
            return self_acc + prepacked_result

        # Migration fallback: pure-JAX prepacked source-leaf path.
        tile_scan_unroll = max(1, int(getattr(payload, "fallback_tile_scan_unroll", 1)))
        batch_scan_unroll = max(
            1, int(getattr(payload, "fallback_batch_scan_unroll", 1))
        )
        fallback_block_tile_size = max(
            1,
            int(getattr(payload, "fallback_block_tile_size", 8)),
        )
        occupancy_sort = _env_flag(
            "JACCPOT_LARGE_N_RADIX_FAST_OCCUPANCY_SORT",
            True,
        )
        skip_empty_tiles = _env_flag(
            "JACCPOT_LARGE_N_RADIX_FAST_SKIP_EMPTY_TILES",
            True,
        )
        componentwise_pairs = _env_flag(
            "JACCPOT_LARGE_N_RADIX_FAST_COMPONENTWISE_PAIRS",
            True,
        )
        pair_acc = (
            _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl(
                positions,
                source_leaf_ids_padded,
                source_valid_mask_padded,
                leaf_positions,
                leaf_masses,
                leaf_mask,
                leaf_particle_idx,
                G=G,
                softening_sq=softening_sq,
                target_leaf_batch_size=int(payload.batch_tile_t),
                target_block_tile_size=int(fallback_block_tile_size),
                target_block_tile_scan_unroll=int(tile_scan_unroll),
                target_block_batch_scan_unroll=int(batch_scan_unroll),
                occupancy_sort=bool(occupancy_sort),
                skip_empty_tiles=bool(skip_empty_tiles),
                componentwise_pairs=bool(componentwise_pairs),
            )
        )
        return self_acc + pair_acc

    source_slot_valid_mask = jnp.any(source_particle_mask, axis=-1)
    source_slot_tile_size = max(1, int(payload.batch_tile_s))
    source_slot_scan_unroll = max(
        1,
        int(getattr(payload, "source_slot_scan_unroll", 1)),
    )
    target_batch_scan_unroll = max(
        1,
        int(getattr(payload, "target_batch_scan_unroll", 1)),
    )

    pair_acc = _compute_radix_fast_lane_payload_pairs_impl(
        positions,
        masses,
        target_particle_ids,
        target_particle_mask,
        source_particle_ids,
        source_particle_mask,
        source_slot_valid_mask,
        G=G,
        softening_sq=softening_sq,
        target_leaf_batch_size=int(payload.batch_tile_t),
        source_slot_tile_size=int(source_slot_tile_size),
        source_slot_scan_unroll=int(source_slot_scan_unroll),
        target_batch_scan_unroll=int(target_batch_scan_unroll),
    )
    return self_acc + pair_acc


def compute_leaf_p2p_accelerations_radix_payload_pairs_only(
    *,
    positions_sorted: Array,
    masses_sorted: Array,
    payload: Any,
    G: Union[float, Array] = 1.0,
    softening: float = 0.0,
    use_pallas: bool = False,
) -> Array:
    """Evaluate payload pair contributions without intra-leaf self work."""
    positions = jnp.asarray(positions_sorted)
    masses = jnp.asarray(masses_sorted)

    target_particle_ids = jnp.asarray(payload.target_particle_ids, dtype=INDEX_DTYPE)
    target_particle_mask = jnp.asarray(payload.target_particle_mask, dtype=bool)
    source_particle_ids = jnp.asarray(payload.source_particle_ids, dtype=INDEX_DTYPE)
    source_particle_mask = jnp.asarray(payload.source_particle_mask, dtype=bool)

    if int(target_particle_ids.size) == 0 or int(source_particle_ids.size) == 0:
        return jnp.zeros_like(positions)

    softening_sq = jnp.asarray(float(softening) ** 2, dtype=positions.dtype)
    if bool(use_pallas):
        from jaccpot.pallas.nearfield_fused_leaf import (
            pallas_nearfield_fused_supported,
        )

        pallas_interpret = _env_flag("JACCPOT_NEARFIELD_PALLAS_INTERPRET", False)
        if pallas_interpret or pallas_nearfield_fused_supported():
            return _radix_fast_lane_pairs_pallas(
                positions,
                masses,
                target_particle_ids,
                target_particle_mask,
                source_particle_ids,
                source_particle_mask,
                G=G,
                softening_sq=softening_sq,
                compute_potential=False,
                num_warps=(_env_int("JACCPOT_NEARFIELD_PALLAS_NUM_WARPS", 0) or None),
                num_stages=max(1, _env_int("JACCPOT_NEARFIELD_PALLAS_NUM_STAGES", 1)),
                target_subtile=(
                    _env_int("JACCPOT_NEARFIELD_PALLAS_TARGET_SUBTILE", 0) or None
                ),
                interpret=pallas_interpret,
            )

    source_slot_valid_mask = jnp.any(source_particle_mask, axis=-1)
    source_slot_tile_size = max(1, int(payload.batch_tile_s))
    source_slot_scan_unroll = max(
        1,
        int(getattr(payload, "source_slot_scan_unroll", 1)),
    )
    target_batch_scan_unroll = max(
        1,
        int(getattr(payload, "target_batch_scan_unroll", 1)),
    )
    return _compute_radix_fast_lane_payload_pairs_impl(
        positions,
        masses,
        target_particle_ids,
        target_particle_mask,
        source_particle_ids,
        source_particle_mask,
        source_slot_valid_mask,
        G=G,
        softening_sq=softening_sq,
        target_leaf_batch_size=int(payload.batch_tile_t),
        source_slot_tile_size=int(source_slot_tile_size),
        source_slot_scan_unroll=int(source_slot_scan_unroll),
        target_batch_scan_unroll=int(target_batch_scan_unroll),
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
