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
from typing import Any, Literal, Optional, Union, overload

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jax import lax
from jaxtyping import Array, Bool, Float, Int, jaxtyped
from yggdrax.dtypes import INDEX_DTYPE

from jaccpot.runtime.grad_options import LeafPairReverseOptions

from ._kernels import _pair_contributions_batched, _self_contributions
from ._large_n_blocks import (
    _collect_target_leaf_batch_acc,
    _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl,
    _compute_leaf_p2p_prepared_large_n_self_only_impl,
)
from ._scatter import _scatter_contributions, _scatter_scalar_contributions
from .grad import (
    _check_float_id_range,
    _leafpair_accel_analytic_vjp,
    _leafpair_reverse_tiers_cached,
)

# The diagnostics gate and the two env-reader aliases still live in `near_field`.
# They are the only thing this module needs from there, and after Tier 1.4 this
# module is their only consumer -- so they arguably belong here now. Left where
# they are deliberately: relocating them is a judgement about where the near-field
# diagnostics surface lives, not part of a mechanical seam split.
from .near_field import _env_flag, _env_int, _large_n_nearfield_diag_mode

__all__ = [
    "compute_leaf_p2p_accelerations_radix_fast_lane",
    "compute_leaf_p2p_accelerations_radix_payload_pairs_only",
]


# WHAT THE SHAPES ARE, AND WHAT THE CAPTURE COULD NOT SEE
#
# Derived by execution with `bench/annotation_capture` (STYLE_GUIDE section 4.2), never
# from the docstrings -- which are wrong here too: `RadixFastNearfieldPayload` documents
# `source_leaf_ids` as "per (target leaf, slot tile)", two axes, and every consumer in
# this file reads `shape[2]`.
#
#   n 3 / n                      positions and masses, in particle order
#   leaves w / leaves w 3        the padded near-field leaf table -- positions, masses,
#                                mask, particle indices
#   farleaves blocks blocksize   the prepacked source-leaf-id rectangle and its mask
#
# Extents behind those names, pooling three captures (`tests/unit/core/test_near_field.py`,
# `tests/unit/test_custom_vjp_parity.py`, and a forced octree run -- see below):
# `leaves` at 3, 6, 512, 528, 4096 and 4120; `w` at 2, 8, 32 and 64; `blocks` at 1 and 2;
# `blocksize` at 2, 3, 26, 27 and 54; `n` at 6, 48, 512 and 1500. Section 4.3 asks for two
# distinct extents per equality, because an equality seen at one problem size is one
# observation however many calls produced it.
#
# `farleaves`, NOT `leaves`, ON THE PREPACKED RECTANGLE, and the reason is inherited
# rather than rediscovered. `_large_n_blocks.py` takes the SAME two arrays as
# `block_source_leaf_ids_padded`/`block_valid_mask_padded` and names their leading axis
# `farleaves`, because that rectangle tracks the FAR-field leaf view, which the octree
# backend separates from the near-field leaf table (the section 4.3 incident, 5 against
# 3). Here the two were observed equal at six distinct extents -- including four from the
# octree lane -- and that is exactly the evidence section 4.3 says not to promote: the
# `farleaves` mistake was 64 honest captures agreeing. `farleaves` binds freely, since
# nothing else in these signatures uses it, so what is asserted is that the ids and their
# mask agree with each other in all three axes. That is measured, and it is all that is.
#
# THE OCTREE LANE IS UNREACHABLE FROM ANY TEST, which is worth stating rather than
# leaving as an absence. `_radix_fast_lane_prepacked_pallas` is called from
# `experimental/octree_fmm_uvwx.py`, but `_octree_near_field`'s `pallas_interpret` knob is
# never plumbed out to `octree_fmm_accelerations`, so on CPU that call site always falls
# through to the pure-JAX branch and on GPU there is no CI leg. The extents above come
# from calling `_octree_near_field(..., use_pallas=True, pallas_interpret=True)` directly;
# all three of its `near_mode` branches build `src_ids_3d` and `unit_pidx` with the same
# leading axis, which is why the octree lane cannot falsify these annotations.


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
    """Dense payload-driven pair kernel for radix fast-lane nearfield.

    The pure-JAX cross-leaf pair kernel of the fast lane, driven straight off the
    prepacked payload's dense particle-id tables rather than an edge list. This is
    the reference the Pallas twins must match.

    Differentiable in ``positions`` and ``masses``; the id tables are integer.

    Parameters
    ----------
    positions : Array
        Particle positions ``[N, 3]``; also fixes the output shape.
    masses : Array
        Particle masses ``[N]``.
    target_particle_ids : Array
        Target particle index per (leaf, slot), from the payload.
    target_particle_mask : Array
        Validity for ``target_particle_ids``, same shape.
    source_particle_ids : Array
        Source particle index per (leaf, slot).
    source_particle_mask : Array
        Validity for ``source_particle_ids``, same shape.
    source_slot_valid_mask : Array
        Per-slot-tile validity, used to skip wholly-empty source tiles.
    G : Union[float, Array]
        Gravitational constant.
    softening_sq : Array
        Squared Plummer softening length.
    target_leaf_batch_size : int
        Target leaves per scan step. Static under ``jit``; batching only.
    source_slot_tile_size : int
        Source slots per tile. Static under ``jit``; batching only.
    source_slot_scan_unroll : int
        Unroll factor for the source-slot scan. Static under ``jit``.
    target_batch_scan_unroll : int
        Unroll factor for the target-batch scan. Static under ``jit``.

    Returns
    -------
    Array
        Per-particle accelerations ``[N, 3]``.

    Raises
    ------
    ValueError
        If any of the four batching knobs is not positive; each sizes a traced
        shape, so a zero is a shape error rather than a slow configuration.
    """
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


@jaxtyped(typechecker=beartype)
def _compute_leaf_p2p_prepared_large_n_self_only_with_potential_impl(
    positions: Float[Array, "n 3"],
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    leaf_particle_idx: Int[Array, "leaves w"],
    *,
    G: Union[float, Array],
    softening_sq: Array,
) -> Tuple[Array, Array]:
    """Self-leaf accel + potential portion of the large-N kernel.

    The potential-carrying twin of
    :func:`~jaccpot.nearfield._large_n_blocks._compute_leaf_p2p_prepared_large_n_self_only_impl`,
    and its accelerations are the same numbers -- the extra return is an addition,
    not a different scheme. Intra-leaf interactions only; the cross-leaf pair
    blocks are added by the caller.

    Unlike its accel-only twin this is **not** wrapped in :func:`jax.jit` at the
    definition site; it is jitted by whatever encloses it.

    Parameters
    ----------
    positions : Float[Array, 'n 3']
        ``[N, 3]`` particle positions in tree order. Read for shape and dtype
        only -- the values used come in leaf-major through ``leaf_positions``.
    leaf_positions : Float[Array, 'leaves w 3']
        ``[num_leaves, W, 3]`` padded leaf-major positions.
    leaf_masses : Float[Array, 'leaves w']
        ``[num_leaves, W]`` padded leaf-major masses.
    leaf_mask : Bool[Array, 'leaves w']
        ``[num_leaves, W]`` occupancy; masked slots contribute exactly zero to
        both outputs and are skipped by both scatters.
    leaf_particle_idx : Int[Array, 'leaves w']
        ``[num_leaves, W]`` particle index behind each slot, clipped so a masked
        slot cannot address out of bounds.
    G : Union[float, Array]
        Gravitational constant, cast to ``positions.dtype`` internally.
    softening_sq : Array
        Plummer softening **squared**, not the softening length.

    Returns
    -------
    Array
        ``[N, 3]`` accelerations in particle order.
    Array
        ``[N]`` potentials in particle order. Self-leaf contributions only, so
        this is a partial sum and is not the near-field potential by itself.
    """
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


# Overloads: the return shape is keyed entirely on `compute_potential`, so a caller that does
# not set it should not be handed a union to narrow. Generated from this
# function's own signature rather than retyped -- including whether `compute_potential`
# has a default, which is what makes the stubs consistent with it. The
# implementation keeps the docstring: pydoclint ignores stubs and beartype
# only ever sees the implementation. Audit E.4.
@overload
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
    compute_potential: Literal[False],
    num_warps: Optional[int] = ...,
    num_stages: int = ...,
    target_subtile: Optional[int] = ...,
    interpret: bool = ...,
) -> Array: ...


@overload
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
    compute_potential: Literal[True],
    num_warps: Optional[int] = ...,
    num_stages: int = ...,
    target_subtile: Optional[int] = ...,
    interpret: bool = ...,
) -> Tuple[Array, Array]: ...


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

    **Numerically equivalent to** :func:`_compute_radix_fast_lane_payload_pairs_impl`
    -- the kernel is an execution accelerator, not different mathematics
    (NUMERICS_AND_JAX §1). Needs Ampere+ (sm_80) unless ``interpret``.

    Parameters
    ----------
    positions : Array
        Particle positions ``[N, 3]``; also fixes the output shape.
    masses : Array
        Particle masses ``[N]``.
    target_particle_ids : Array
        Target particle index per (leaf, slot).
    target_particle_mask : Array
        Validity for ``target_particle_ids``, same shape.
    source_particle_ids : Array
        Source particle index per (leaf, slot-tile, slot).
    source_particle_mask : Array
        Validity for ``source_particle_ids``, same shape.
    G : Union[float, Array]
        Gravitational constant.
    softening_sq : Array
        Squared Plummer softening length.
    compute_potential : bool
        Also return per-particle potentials. Static under ``jit``.
    num_warps : Optional[int]
        Triton warp count; ``None`` lets the kernel pick. **Unclamped on purpose**
        -- 0 means "unset" for these knobs, so a minimum of 1 would turn "auto"
        into a real, wrong value.
    num_stages : int
        Triton pipeline stages. Default ``1``.
    target_subtile : Optional[int]
        Target sub-tile width; ``None`` lets the kernel pick.
    interpret : bool
        Run through Pallas' reference interpreter instead of Triton. This is how
        CPU CI exercises the kernel at all, and how the parity tests assert the
        equivalence -- but the *shipped* callers hardcode ``interpret=False``, so
        an ``interpret=True`` pass says nothing about the Triton lowering.

    Returns
    -------
    Union[Array, Tuple[Array, Array]]
        Per-particle accelerations ``[N, 3]``, or ``(accelerations, potentials)``
        when ``compute_potential``.
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


# Overloads: the return shape is keyed entirely on `compute_potential`, so a caller that does
# not set it should not be handed a union to narrow. Generated from this
# function's own signature rather than retyped -- including whether `compute_potential`
# has a default, which is what makes the stubs consistent with it. The
# implementation keeps the docstring: pydoclint ignores stubs and beartype
# only ever sees the implementation. Audit E.4.
@overload
def _radix_fast_lane_prepacked_pallas(
    source_leaf_ids_padded: Int[Array, "farleaves blocks blocksize"],
    source_valid_mask_padded: Bool[Array, "farleaves blocks blocksize"],
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    leaf_particle_idx: Int[Array, "leaves w"],
    positions: Float[Array, "n 3"],
    *,
    G: Union[float, Array],
    softening_sq: Array,
    compute_potential: Literal[False],
    num_warps: Optional[int] = ...,
    num_stages: int = ...,
    target_subtile: Optional[int] = ...,
    interpret: bool = ...,
) -> Array: ...


@overload
def _radix_fast_lane_prepacked_pallas(
    source_leaf_ids_padded: Int[Array, "farleaves blocks blocksize"],
    source_valid_mask_padded: Bool[Array, "farleaves blocks blocksize"],
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    leaf_particle_idx: Int[Array, "leaves w"],
    positions: Float[Array, "n 3"],
    *,
    G: Union[float, Array],
    softening_sq: Array,
    compute_potential: Literal[True],
    num_warps: Optional[int] = ...,
    num_stages: int = ...,
    target_subtile: Optional[int] = ...,
    interpret: bool = ...,
) -> Tuple[Array, Array]: ...


@jaxtyped(typechecker=beartype)
def _radix_fast_lane_prepacked_pallas(
    source_leaf_ids_padded: Int[Array, "farleaves blocks blocksize"],
    source_valid_mask_padded: Bool[Array, "farleaves blocks blocksize"],
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    leaf_particle_idx: Int[Array, "leaves w"],
    positions: Float[Array, "n 3"],
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

    This is the forward half of the production differentiable near field: it is
    what :func:`_radix_fast_lane_prepacked_accel_cvjp` calls as its primal. Needs
    Ampere+ (sm_80) unless ``interpret``.

    Parameters
    ----------
    source_leaf_ids_padded : Int[Array, 'farleaves blocks blocksize']
        Source leaf ids ``[num_leaves, max_blocks, block_size]``, padded to a
        rectangle.
    source_valid_mask_padded : Bool[Array, 'farleaves blocks blocksize']
        Per-lane validity with the same shape; padded lanes contribute exactly zero.
    leaf_positions : Float[Array, 'leaves w 3']
        Padded per-leaf positions ``[num_leaves, W, 3]``.
    leaf_masses : Float[Array, 'leaves w']
        Padded per-leaf masses ``[num_leaves, W]``.
    leaf_mask : Bool[Array, 'leaves w']
        Padded per-leaf validity ``[num_leaves, W]``.
    leaf_particle_idx : Int[Array, 'leaves w']
        Particle index behind each padded slot ``[num_leaves, W]``.
    positions : Float[Array, 'n 3']
        Particle positions ``[N, 3]``; fixes the output shape.
    G : Union[float, Array]
        Gravitational constant.
    softening_sq : Array
        Squared Plummer softening length.
    compute_potential : bool
        Also return per-particle potentials. Static under ``jit``.
    num_warps : Optional[int]
        Triton warp count; ``None`` lets the kernel pick. **Unclamped on purpose**
        -- 0 means "unset" for these knobs.
    num_stages : int
        Triton pipeline stages. Default ``1``.
    target_subtile : Optional[int]
        Target sub-tile width; ``None`` lets the kernel pick.
    interpret : bool
        Run through Pallas' reference interpreter rather than Triton; the shipped
        callers hardcode ``False``.

    Returns
    -------
    Union[Array, Tuple[Array, Array]]
        Per-particle accelerations ``[N, 3]``, or ``(accelerations, potentials)``
        when ``compute_potential``.
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


# Overloads: the return shape is keyed entirely on `compute_potential`, so a caller that does
# not set it should not be handed a union to narrow. Generated from this
# function's own signature rather than retyped -- including whether `compute_potential`
# has a default, which is what makes the stubs consistent with it. The
# implementation keeps the docstring: pydoclint ignores stubs and beartype
# only ever sees the implementation. Audit E.4.
@overload
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
    compute_potential: Literal[False] = ...,
    num_warps: Optional[int] = ...,
    num_stages: int = ...,
    target_subtile: Optional[int] = ...,
    interpret: bool = ...,
    accum: str = ...,
) -> Array: ...


@overload
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
    compute_potential: Literal[True],
    num_warps: Optional[int] = ...,
    num_stages: int = ...,
    target_subtile: Optional[int] = ...,
    interpret: bool = ...,
    accum: str = ...,
) -> Tuple[Array, Array]: ...


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
    accum: str = "input",
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

    STATED EQUIVALENCE (F25, now asserted). Passing the **same** array as both
    target and source reproduces :func:`_radix_fast_lane_prepacked_pallas`
    bit-for-bit -- ``tests/unit/operators/test_pallas_nearfield_fused.py`` pins it.
    That matters because this is the kernel ``distributed/fmm.py`` runs, so the
    distributed near field's equivalence to the single-device one rests on it.

    Parameters
    ----------
    source_leaf_ids_padded : Array
        Source leaf ids ``[num_targets, max_blocks, block_size]``, referencing
        **global source rows** in ``[0, num_sources)``.
    source_valid_mask_padded : Array
        Per-lane validity with the same shape.
    target_positions : Array
        This block's target positions ``[num_targets, W, 3]``.
    target_mask : Array
        This block's target validity ``[num_targets, W]``.
    target_particle_idx : Array
        **Global** particle index per target slot ``[num_targets, W]``, so
        per-block partials scatter-add and compose by ``+``.
    source_positions : Array
        The full source gather pool ``[num_sources, W, 3]``, resident.
    source_masses : Array
        Source masses ``[num_sources, W]``.
    source_mask : Array
        Source validity ``[num_sources, W]``.
    positions : Array
        Particle positions ``[N, 3]``; supplies the output shape only.
    G : Union[float, Array]
        Gravitational constant.
    softening_sq : Array
        Squared Plummer softening length.
    compute_potential : bool
        Also return per-particle potentials. Static under ``jit``; default ``False``.
    num_warps : Optional[int]
        Triton warp count; ``None`` lets the kernel pick. Unclamped on purpose.
    num_stages : int
        Triton pipeline stages. Default ``1``.
    target_subtile : Optional[int]
        Target sub-tile width; ``None`` lets the kernel pick.
    interpret : bool
        Run through Pallas' reference interpreter rather than Triton.
    accum : str
        Per-target accumulator width. ``"input"`` (default) is byte-identical to
        the historical path -- the kernel takes the original code path verbatim,
        not a specialisation of the widened one. ``"wide"`` accumulates in float64
        with a float32 partial per source leaf, while every multiply and the
        ``rsqrt`` stay in the input dtype. See
        :func:`_nearfield_leafpair_kernel` for why widening only the accumulator is
        the whole fix: measured 439x in force accuracy for 1.8 % in time on the
        distributed lane at 10^7 particles.

    Returns
    -------
    Union[Array, Tuple[Array, Array]]
        Per-particle accelerations ``[N, 3]`` for this target block (zero
        elsewhere), or ``(accelerations, potentials)`` when ``compute_potential``.
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
        accum=accum,
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
@jaxtyped(typechecker=beartype)
def _radix_fast_lane_prepacked_accel_cvjp(
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    positions: Float[Array, "n 3"],
    source_leaf_ids_f: Float[Array, "farleaves blocks blocksize"],
    source_valid_f: Float[Array, "farleaves blocks blocksize"],
    leaf_mask_f: Float[Array, "leaves w"],
    leaf_particle_idx_f: Float[Array, "leaves w"],
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

    All arguments are positional because this is a ``jax.custom_vjp`` primal;
    ``nondiff_argnums=(9, 10, 11, 12, 13, 14, 15, 16)`` marks everything from
    ``num_warps`` on as non-differentiable, so those must not be passed by keyword.
    The ``*_f`` suffixes mark the integer/boolean tables that were cast to float to
    cross the ``custom_vjp`` boundary without acquiring a tangent.

    Parameters
    ----------
    leaf_positions : Float[Array, 'leaves w 3']
        Padded per-leaf positions ``[num_leaves, W, 3]``. **Differentiable.**
    leaf_masses : Float[Array, 'leaves w']
        Padded per-leaf masses ``[num_leaves, W]``. **Differentiable.**
    positions : Float[Array, 'n 3']
        Particle positions ``[N, 3]``; supplies the output shape.
    source_leaf_ids_f : Float[Array, 'farleaves blocks blocksize']
        Source leaf ids, float-cast, ``[num_leaves, blocks, lanes]``.
    source_valid_f : Float[Array, 'farleaves blocks blocksize']
        Per-lane validity, float-cast, same shape.
    leaf_mask_f : Float[Array, 'leaves w']
        Padded per-leaf validity, float-cast, ``[num_leaves, W]``.
    leaf_particle_idx_f : Float[Array, 'leaves w']
        Particle index per padded slot, float-cast, ``[num_leaves, W]``. See
        ``_check_float_id_range`` -- an id beyond float exact-integer range would
        be silently rounded, so it is validated rather than assumed.
    softening_sq : Array
        Squared Plummer softening length.
    G : Array
        Gravitational constant.
    num_warps : Optional[int]
        Triton warp count. Non-differentiable.
    num_stages : int
        Triton pipeline stages. Non-differentiable.
    target_subtile : Optional[int]
        Target sub-tile width. Non-differentiable.
    interpret : bool
        Use Pallas' reference interpreter. Non-differentiable.
    rev_leaf_batch : int
        Target leaves per reverse scan step. Non-differentiable; bounds the
        backward's peak, independent of the forward's tiling.
    rev_block_tile : int
        Source lanes per reverse tile. Non-differentiable; same role.
    rev_skip_empty : bool
        Skip all-invalid tiles in the reverse. Non-differentiable.
    rev_tiers : Optional[Tuple[Tuple[Tuple[int, ...], int], ...]]
        Precomputed tier plan from ``build_leafpair_reverse_tiers`` -- groups of
        target leaves sharing a slot width, so a low-occupancy leaf does not pay
        the global maximum. ``None`` runs untiered. Non-differentiable.

    Returns
    -------
    Array
        Per-particle accelerations ``[N, 3]``.
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


# THE SAVED RESIDUAL, SPELLED OUT, AND WHAT THAT IS AND IS NOT WORTH.
#
# NUMERICS_AND_JAX section 1 says the residuals a `custom_vjp` saves must not change
# without re-running `bench/audit_reverse_residuals.py`. A nine-tuple annotation is NOT a
# tripwire on that, and the first draft of this comment claimed it was. Measured, by
# re-registering `defvjp` with a `_fwd` that mutates its residual three ways:
#
#   swap `leaf_mask_f` with `leaf_particle_idx_f`   NOT NOTICED, before or after
#   swap the leaf table for the rectangle          IndexError  -> TypeCheckError
#   drop `leaf_masses`                             ValueError  -> TypeCheckError
#
# The two same-shaped `(leaves, w)` float entries are indistinguishable to any shape
# annotation, and that is the mutation a refactor is most likely to make. What the
# annotation actually buys is locality on the other two: a `TypeCheckError` naming
# `residual` instead of an `IndexError` raised inside the analytic VJP. Worth having, not
# worth mistaking for the audit script.
#
# The order is `_fwd`'s, not the primal's -- they agree here, but only because `_fwd`
# happens to save its first nine arguments in order, which is a fact about that body and
# not a rule. Derived by re-registering `defvjp` with recording wrappers over four
# `(leaves, w, blocks, blocksize, n)` configurations. `bench/annotation_capture` cannot
# see either rule and reports no observations for both: `defvjp` captured them at import,
# so rebinding the module attribute leaves the `custom_vjp` object calling the originals
# -- the "reference captured into a closure before the patch went on" case its own
# docstring warns about, which must not be read as "never called".
_LeafPairReverseResidual = Tuple[
    Float[Array, "leaves w 3"],  # leaf_positions
    Float[Array, "leaves w"],  # leaf_masses
    Float[Array, "n 3"],  # positions
    Float[Array, "farleaves blocks blocksize"],  # source_leaf_ids_f
    Float[Array, "farleaves blocks blocksize"],  # source_valid_f
    Float[Array, "leaves w"],  # leaf_mask_f
    Float[Array, "leaves w"],  # leaf_particle_idx_f
    Array,  # softening_sq -- scalar, observed `()` in every call
    Array,  # G -- scalar, likewise
]


# ANNOTATED BUT NOT DECORATED, AND THE REASON IS MEASURED RATHER THAN INHERITED.
#
# The obvious argument for a decorator here is that `@jaxtyped` on the primal cannot run
# in reverse mode, since `jax.grad` calls this rule and never the primal. That argument is
# wrong FOR THIS LANE, and the distinction is worth stating because it is a property of
# the body below rather than of `custom_vjp`: the first thing this rule does is call the
# `custom_vjp` object again, which is an ordinary forward call, so the primal -- and its
# decorator -- does run. Measured: stripping this decorator changes nothing, all six
# corruptions in `tests/unit/test_nearfield_fastlane_grad_path.py` still fail identically
# on the grad path.
#
# So a decorator here would be section 4.1's "annotate for consistency", at the cost of a
# beartype pass per trace on the production grad path (section 4.4: UNCONDITIONAL). The
# annotations stay, because they are not free: they document the 17-argument contract, and
# the `JACCPOT_RUNTIME_TYPECHECK=1` import hook enforces them on undecorated functions,
# which is the leg that exists to catch exactly this.
#
# WHAT WOULD CHANGE THE ANSWER: a `_fwd` that stops re-entering the primal -- calling
# `_radix_fast_lane_prepacked_pallas` directly, say, to avoid the double dispatch. The
# check would vanish silently. Decorate it then.
def _radix_fast_lane_prepacked_accel_fwd(
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    positions: Float[Array, "n 3"],
    source_leaf_ids_f: Float[Array, "farleaves blocks blocksize"],
    source_valid_f: Float[Array, "farleaves blocks blocksize"],
    leaf_mask_f: Float[Array, "leaves w"],
    leaf_particle_idx_f: Float[Array, "leaves w"],
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
) -> Tuple[Array, _LeafPairReverseResidual]:
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


# Undecorated for the same reason as `_fwd`, plus one of its own: what a decorator here
# would newly reject is the residual, and the table above measures that as locality rather
# than detection. Enforced under `JACCPOT_RUNTIME_TYPECHECK=1` by the import hook.
def _radix_fast_lane_prepacked_accel_bwd(
    num_warps: Optional[int],
    num_stages: int,
    target_subtile: Optional[int],
    interpret: bool,
    rev_leaf_batch: int,
    rev_block_tile: int,
    rev_skip_empty: bool,
    rev_tiers: Optional[Tuple[Tuple[Tuple[int, ...], int], ...]],
    residual: _LeafPairReverseResidual,
    cotangent: Float[Array, "n 3"],
) -> Tuple[Array, ...]:
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


# Overloads: the return shape is keyed entirely on `return_potential`, so a caller that does
# not set it should not be handed a union to narrow. Generated from this
# function's own signature rather than retyped -- including whether `return_potential`
# has a default, which is what makes the stubs consistent with it. The
# implementation keeps the docstring: pydoclint ignores stubs and beartype
# only ever sees the implementation. Audit E.4.
@overload
def compute_leaf_p2p_accelerations_radix_fast_lane(
    *,
    positions_sorted: Float[Array, "n 3"],
    masses_sorted: Float[Array, "n"],
    payload: Any,
    G: Union[float, Array] = ...,
    softening: float = ...,
    return_potential: Literal[False] = ...,
    use_pallas: bool = ...,
    differentiable: bool = ...,
    reverse_options: Optional["LeafPairReverseOptions"] = ...,
) -> Array: ...


@overload
def compute_leaf_p2p_accelerations_radix_fast_lane(
    *,
    positions_sorted: Float[Array, "n 3"],
    masses_sorted: Float[Array, "n"],
    payload: Any,
    G: Union[float, Array] = ...,
    softening: float = ...,
    return_potential: Literal[True],
    use_pallas: bool = ...,
    differentiable: bool = ...,
    reverse_options: Optional["LeafPairReverseOptions"] = ...,
) -> Tuple[Array, Array]: ...


@jaxtyped(typechecker=beartype)
def compute_leaf_p2p_accelerations_radix_fast_lane(
    *,
    positions_sorted: Float[Array, "n 3"],
    masses_sorted: Float[Array, "n"],
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

    Parameters
    ----------
    positions_sorted : Float[Array, 'n 3']
        Particle positions ``[N, 3]`` in Morton order.
    masses_sorted : Float[Array, 'n']
        Particle masses ``[N]`` in the same order.
    payload : Any
        The prepacked radix fast-lane payload
        (:class:`~jaccpot.runtime._large_n_types.RadixFastNearfieldPayload`), which
        carries the leaf tables **and** the batch tiling the kernels use.
    G : Union[float, Array]
        Gravitational constant. Default ``1.0``.
    softening : float
        Plummer softening **length** (squared internally). Must be a concrete
        Python float, not a tracer. Default ``0.0``.
    return_potential : bool
        Also return per-particle potentials. Static under ``jit``.
    use_pallas : bool
        Use the fused Pallas kernels rather than the pure-JAX payload kernel.
        Requires Ampere+ (sm_80).
    differentiable : bool
        Route through the ``custom_vjp`` wrappers so ``jax.grad`` works. Defaults
        ``False`` so the production forward path is untouched.
    reverse_options : Optional[LeafPairReverseOptions]
        Resolved reverse-pass tuning. ``None`` resolves from the environment,
        which is what forward-only callers get.

    Returns
    -------
    Union[Array, Tuple[Array, Array]]
        Per-particle accelerations ``[N, 3]``, or ``(accelerations, potentials)``
        when ``return_potential``.

    Raises
    ------
    NotImplementedError
        For a requested combination this lane does not implement, rather than
        silently substituting one it does -- ``"auto"`` may choose, an explicit
        request may not be quietly overridden (STYLE_GUIDE §9).
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
        # Branch on the value, not on `want_potential`. Both say the same thing --
        # the callee returns a pair exactly when the flag is set -- but the flag is
        # a runtime bool, so it selects the fallback overload and leaves the result
        # a union that cannot be added to an Array. `isinstance` narrows it, and a
        # JAX array is never a tuple, so the discriminator is exact.
        if isinstance(pairs_result, tuple):
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
            # Branch on the value, not on `want_potential`. Both say the same thing --
            # the callee returns a pair exactly when the flag is set -- but the flag is
            # a runtime bool, so it selects the fallback overload and leaves the result
            # a union that cannot be added to an Array. `isinstance` narrows it, and a
            # JAX array is never a tuple, so the discriminator is exact.
            if isinstance(prepacked_result, tuple):
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


@jaxtyped(typechecker=beartype)
def compute_leaf_p2p_accelerations_radix_payload_pairs_only(
    *,
    positions_sorted: Float[Array, "n 3"],
    masses_sorted: Float[Array, "n"],
    payload: Any,
    G: Union[float, Array] = 1.0,
    softening: float = 0.0,
    use_pallas: bool = False,
) -> Array:
    """Evaluate payload pair contributions without intra-leaf self work.

    Cross-leaf pair blocks only. The intra-leaf self term is deliberately absent
    -- callers that want the whole near field add
    :func:`_compute_leaf_p2p_prepared_large_n_self_only_impl`'s result to this
    one. Using this alone is the way to get a silently incomplete force.

    Keyword-only throughout, so the argument order here is not a contract.

    Parameters
    ----------
    positions_sorted : Float[Array, 'n 3']
        ``[N, 3]`` positions in tree order. Also fixes the output shape.
    masses_sorted : Float[Array, 'n']
        ``[N]`` masses under the same permutation.
    payload : Any
        A ``RadixFastNearfieldPayload``. Deliberately untyped, to keep this
        module free of a runtime import of the runtime layer. The ``batch_tile_s``
        / ``batch_tile_t`` tiles and the four particle-id/mask arrays are
        required; ``source_slot_scan_unroll`` and ``target_batch_scan_unroll``
        are read through ``getattr`` with a default of 1, so an older payload
        without them still works.
    G : Union[float, Array]
        Gravitational constant; defaults to 1.
    softening : float
        Plummer softening **length**; defaults to 0. Squared here, which is the
        opposite convention from the ``softening_sq`` arguments elsewhere in this
        module -- passing an already-squared value softens by its square.
    use_pallas : bool
        Request the fused Pallas lane; defaults to ``False``. A request, not a
        guarantee: it falls through to the scan lane unless
        ``pallas_nearfield_fused_supported()`` or
        ``JACCPOT_NEARFIELD_PALLAS_INTERPRET`` says otherwise. The two lanes are
        meant to produce the same numbers.

    Returns
    -------
    Array
        ``[N, 3]`` accelerations in particle order, all zero when the payload
        carries no target or source particles.
    """
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
