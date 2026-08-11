"""The radix fast lane: the production Pallas near-field path and its ``custom_vjp``.

WHAT THIS LANE IS. The leaf-major "fast lane" evaluates the near field from a
prepacked, fixed-shape radix payload instead of an edge list, which is what makes
the *reverse* pass fit: measured at N=200000, the bucketed reverse OOMs at 30 GB
peak while this lane completes in 6.8 GB (ARCHITECTURE §6), and that measurement
is why ``nearfield_lane="auto"`` crosses over to it at N >= 100000.

WHY IT IS ITS OWN MODULE. Two reasons, both about reviewability:

* ``_radix_fast_lane_prepacked_accel_cvjp`` is the production near-field
  ``custom_vjp``. NUMERICS_AND_JAX §1 calls ``custom_vjp`` on the Pallas kernels
  *load-bearing*: it must not be "simplified" into autodiff-through-the-kernel,
  and its saved residuals must not change without re-running
  ``bench/audit_reverse_residuals.py``. Isolating it means that rule applies to a
  file you can read in one sitting.
* All five function-local ``pallas/nearfield_fused_leaf`` imports live here. They
  are function-local **deliberately**, to defer the heavy Pallas import
  (STYLE_GUIDE §8) -- do not hoist them.

DIRECTION OF DEPENDENCE. This module imports the shared near-field primitives
(the self/pair arithmetic, the scatter family, the large-N target-block kernels)
from :mod:`jaccpot.nearfield.near_field`; nothing flows back. Consumers of the
fast lane import it from here.

Split out of ``near_field.py`` (Tier 1.4); every function body is unchanged.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jax import lax
from jaxtyping import Array
from yggdrax.dtypes import INDEX_DTYPE

from jaccpot.runtime.grad_options import LeafPairReverseOptions

from .grad import (
    _check_float_id_range,
    _leafpair_accel_analytic_vjp,
    _leafpair_reverse_tiers_cached,
)
from .near_field import (
    _collect_target_leaf_batch_acc,
    _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl,
    _compute_leaf_p2p_prepared_large_n_self_only_impl,
    _env_flag,
    _env_int,
    _large_n_nearfield_diag_mode,
    _pair_contributions_batched,
    _scatter_contributions,
    _scatter_scalar_contributions,
    _self_contributions,
)

__all__ = [
    "compute_leaf_p2p_accelerations_radix_fast_lane",
    "compute_leaf_p2p_accelerations_radix_payload_pairs_only",
]


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


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13, 14, 15, 16))
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
    rev_skip_empty: bool,
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
    rev_skip_empty,
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
        rev_skip_empty,
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
    rev_skip_empty,
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
        skip_empty_tiles=bool(rev_skip_empty),
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
    reverse_options: Optional["LeafPairReverseOptions"] = None,
) -> Union[Array, Tuple[Array, Array]]:
    """Payload-driven nearfield entry for the radix fast lane.

    ``differentiable`` routes the fused Pallas lanes through their ``custom_vjp``
    wrappers so ``jax.grad`` works. ``pallas_call`` has no autodiff rule, so the
    raw lanes hard-fail in reverse mode; the wrapper keeps the Pallas forward
    (byte-identical) and takes the reverse from this lane's tiled pure-JAX
    fallback. Defaults False so the production forward path is untouched.

    ``reverse_options`` carries the resolved reverse-pass tuning (tiering, tile
    sizes, the empty-tile skip). ``None`` means "resolve from the environment",
    which is what the forward-only production callers get; the differentiable
    entry point resolves a :class:`~jaccpot.config.GradConfig` once, outside the
    trace, and passes the result down so nothing reads the environment per call.
    """
    if reverse_options is None:
        from jaccpot.runtime.grad_options import resolve_grad_options

        reverse_options = resolve_grad_options(
            None, num_particles=0, supports_fast_lane=False
        ).reverse
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
                rev_block_tile = int(reverse_options.block_tile)
                # Occupancy tiers for the reverse, built HERE rather than inside the
                # bwd rule: the payload's validity mask is frozen topology and is
                # concrete at this point, whereas inside the rule it is a residual
                # (a tracer under jit) from which no static slot width could be
                # read. Rides through as a nondiff arg, so each tier's width is a
                # compile-time constant.
                rev_tiers = None
                if reverse_options.tiered:
                    rev_tiers = _leafpair_reverse_tiers_cached(
                        payload.source_leaf_valid_mask,
                        source_valid_mask_padded,
                        slot_tile=rev_block_tile,
                        max_tiers=reverse_options.max_tiers,
                        min_gain=reverse_options.tier_min_gain,
                    )
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
                    int(reverse_options.leaf_batch),
                    rev_block_tile,
                    bool(reverse_options.skip_empty_tiles),
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
