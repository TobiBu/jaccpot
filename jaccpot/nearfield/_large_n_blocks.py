"""The large-N target-block near-field kernels.

The leaf-major family the large-N GPU lane runs instead of the edge-list
traversal: each kernel owns a block of *target* leaves and scans source tiles into
it, so the reverse pass retains residuals proportional to the block rather than to
the edge list. That is the difference that makes gradients fit at galaxy N -- the
bucketed reverse OOMs at 30 GB peak at N=200000 while this family completes in
6.8 GB (ARCHITECTURE §6).

The variants differ only in how the source side is fed: ``_pairs_only`` from an
edge list, ``_target_blocks`` from owned blocks, ``_prepacked`` from a fixed-shape
payload, ``_tiled`` with an extra tile loop, and ``_accel_only`` skipping the
potential. Every one of them is padded to static shapes with masked slots
contributing exactly zero.

Split out of ``near_field.py`` (Tier 1.5, A.9 seam 3); every function body is
unchanged.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Union

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, Bool, Float, Int, jaxtyped
from yggdrax.dtypes import INDEX_DTYPE, as_index

from ._kernels import (
    _pair_contributions_batched,
    _pair_contributions_batched_componentwise,
    _self_contributions,
)
from ._scatter import (
    _reduce_pair_bucket_by_target_leaf,
    _scatter_contributions,
    _scatter_contributions_grouped_sorted,
    _scatter_contributions_sorted_hint,
)
from ._schedules import _prepare_leaf_data_from_groups

__all__ = ["compute_leaf_p2p_accelerations_target_block_pairs_only"]


# WHAT THE SHAPES ARE, AND WHICH EQUALITIES ARE PROVEN
#
# Derived by execution with `bench/annotation_pilot`, three calls per function across
# the 33 test files that reach this lane -- never from the docstrings, which
# STYLE_GUIDE section 4.2 does not trust and which are wrong elsewhere in the package.
# Every axis name already existed: this module adds no row to the section 4.3 table
# and no entry to the flake8 `--builtins` list.
#
#   n 3                          positions
#   leaves w / leaves w 3        the padded near-field leaf table -- positions,
#                                masses, mask, particle indices
#   pairs                        the edge-list variants' target/source/valid triple
#   blocks blocksize             per-block source ids and their mask
#   farleaves blocks blocksize   the prepacked rectangle, one block run per leaf
#   _                            block_offsets, block_target_leaf_ids
#
# `leaves w` IS PROVEN, at two distinct extents: `_self_only_impl` was captured at
# `(8, 128)` and `(16, 128)`, `_prepacked_impl` at `(5, 256)` and `(3, 2)`. Section 4.3
# asks for two, because an equality seen at one size is one observation however many
# calls produced it.
#
# THE TWO `_` ARE NOT LAZINESS, and they are where the first draft of this note was
# wrong. `runtime/kernels/_evaluate.py` -- which supplies these arrays -- annotates the
# same two as `Int[Array, "farleaves+1"]` and `Int[Array, "_"]`, and it has the
# measurements: the offsets track the FAR-field leaf view, radix 3 leaves -> 4 offsets
# against octree 5 -> 6. That view is not this signature's `leaves`, which is the
# near-field padded table, so writing `leaves+1` here would assert an equality the
# octree backend falsifies -- the `farleaves` mistake of section 4.3, verbatim, which
# cost 7 tests when it was made in `_evaluate.py`. Nor can the honest spelling be used:
# jaxtyping evaluates a symbolic axis in PARAMETER ORDER, `block_offsets` precedes the
# leaf table, and a symbolic dim cannot be the thing that introduces its own name --
# `AnnotationError: Cannot process symbolic axis 'leaves+1' as some axis names have not
# been resolved`, which is how this was found rather than guessed.
#
# Same reasoning puts `farleaves` on the prepacked rectangle instead of `leaves`. It
# binds freely there -- nothing else in that signature uses it -- so what is asserted
# is that the ids and their mask agree with each other in all three axes, which is
# measured, and nothing about the extent. That is the distinction `_evaluate.py` draws
# for `blocks blocksize`, applied here.
#
# ONE THING WORTH REVISITING, recorded rather than acted on: `_evaluate.py` keeps
# `block_target_leaf_ids` rank-only because "its length was 0 in every observation".
# This capture saw it at 15, beside `blocks` 15 -- the first nonzero evidence for the
# coupling its docstring claims. One observation is not two, so it stays `_`.
#
# THE SENTINEL HAZARD WAS CHECKED, because section 4.3 records it costing 87 tests.
# `_evaluate.py`'s `nearfield_leaf_particle_indices` is `(leaves, w)` live and `(0, 0)`
# absent, which is why it must stay `Int[Array, "_ _"]` there -- and it is the array
# that becomes `leaf_particle_idx` here. It cannot arrive as the sentinel: the
# `use_specialized_large_n` gate gives `shape[0] > 0` as a static Python check, so this
# family is only entered once the lane is live. An all-zero `leaves` would satisfy
# these annotations anyway, since the four leaf arrays only have to agree with each
# other.


@jax.jit
@jaxtyped(typechecker=beartype)
def _compute_leaf_p2p_prepared_large_n_self_only_impl(
    positions: Float[Array, "n 3"],
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    leaf_particle_idx: Int[Array, "leaves w"],
    *,
    G: Union[float, Array],
    softening_sq: Array,
) -> Array:
    """Self-leaf portion of the specialized large-N accel-only kernel.

    Intra-leaf interactions only, already scattered back to particle order. It is
    one of two additive halves -- the cross-leaf pair blocks are the other -- so
    the array returned here is not the near field on its own.

    Decorated with a bare :func:`jax.jit`, so every argument is traced and none
    may be a Python-level switch.

    Parameters
    ----------
    positions : Float[Array, 'n 3']
        ``[N, 3]`` particle positions in tree order. Read only for its shape and
        dtype: the values that matter come in through ``leaf_positions``, and
        this fixes the output shape and the working dtype.
    leaf_positions : Float[Array, 'leaves w 3']
        ``[num_leaves, W, 3]`` padded leaf-major positions.
    leaf_masses : Float[Array, 'leaves w']
        ``[num_leaves, W]`` padded leaf-major masses.
    leaf_mask : Bool[Array, 'leaves w']
        ``[num_leaves, W]`` occupancy; masked slots contribute exactly zero and
        are also excluded from the scatter.
    leaf_particle_idx : Int[Array, 'leaves w']
        ``[num_leaves, W]`` particle index behind each slot, clipped so a masked
        slot cannot gather or scatter out of bounds.
    G : Union[float, Array]
        Gravitational constant, cast to ``positions.dtype`` internally.
    softening_sq : Array
        Plummer softening **squared**. Squared, not linear -- passing the
        softening length itself gives a silently over-softened force.

    Returns
    -------
    Array
        ``[N, 3]`` accelerations in particle order, zero for every particle whose
        leaf slot was masked.
    """
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
@jaxtyped(typechecker=beartype)
def _compute_leaf_p2p_prepared_large_n_pairs_only_impl(
    positions: Float[Array, "n 3"],
    target_leaf_ids: Int[Array, "pairs"],
    source_leaf_ids: Int[Array, "pairs"],
    valid_pairs: Bool[Array, "pairs"],
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    leaf_particle_idx: Int[Array, "leaves w"],
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
    """Cross-leaf pair-bucket portion of the specialized large-N kernel.

    The edge-list half of the large-N bucketed lane: cross-leaf pairs only, no
    intra-leaf self block (its caller adds that). Edges are consumed in fixed-size
    chunks grouped into superchunks so every shape is static.

    Differentiable in ``positions``, ``leaf_positions`` and ``leaf_masses``.

    Parameters
    ----------
    positions : Float[Array, 'n 3']
        Particle positions ``[N, 3]``; also fixes the output shape.
    target_leaf_ids : Int[Array, 'pairs']
        Target leaf id per edge, ``[num_edges]``.
    source_leaf_ids : Int[Array, 'pairs']
        Source leaf id per edge, ``[num_edges]``.
    valid_pairs : Bool[Array, 'pairs']
        Per-edge validity ``[num_edges]``; padded edges contribute exactly zero.
    leaf_positions : Float[Array, 'leaves w 3']
        Padded per-leaf positions ``[num_leaves, W, 3]``.
    leaf_masses : Float[Array, 'leaves w']
        Padded per-leaf masses ``[num_leaves, W]``.
    leaf_mask : Bool[Array, 'leaves w']
        Padded per-leaf validity ``[num_leaves, W]``; masked slots contribute
        exactly zero.
    leaf_particle_idx : Int[Array, 'leaves w']
        Particle index behind each padded slot ``[num_leaves, W]``, clipped so a
        masked slot cannot gather out of bounds. Also fixes ``W``.
    G : Union[float, Array]
        Gravitational constant.
    softening_sq : Array
        Squared Plummer softening length.
    edge_chunk_size : int
        Edges per chunk. Static under ``jit``; batching only.
    chunks_per_superchunk : int
        Chunks grouped into one superchunk before the target reduction. Static
        under ``jit``.
    chunk_scan_batch_size : int
        Chunks consumed per scan step. Static under ``jit``.
    chunk_scan_unroll : int
        Unroll factor for the chunk scan. Static under ``jit``.
    superchunk_scan_unroll : int
        Unroll factor for the superchunk scan. Static under ``jit``.
    sorted_scatter_hint : bool
        Tell the scatter its indices are sorted. Static under ``jit``. A *promise*,
        not a request: it must match the data or the scatter is wrong.
    grouped_sorted_scatter : bool
        Use the segment-grouped scatter. Static under ``jit``; same caveat.
    superchunk_target_reduce : bool
        Reduce per target leaf within a superchunk before scattering. Static under
        ``jit``; changes the summation *grouping*, so it can move the last digits.
    disable_chunk_cond : bool
        Skip the per-chunk ``lax.cond`` early-out. Static under ``jit``.

    Returns
    -------
    Array
        Per-particle accelerations ``[N, 3]``, zero outside this kernel's share.

    Raises
    ------
    ValueError
        If any of ``edge_chunk_size``, ``chunks_per_superchunk``,
        ``chunk_scan_batch_size``, ``chunk_scan_unroll`` or
        ``superchunk_scan_unroll`` is not positive. These size a traced shape, so a
        zero would be a shape error rather than a slow configuration.
    """
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


# `tiles`, `tbatch` AND WHY THEY ARE NOT `blocks` OR `leaves`.
#
# Both names are new to the STYLE_GUIDE section 4.3 table and both were added with
# this change rather than ahead of it, because a vocabulary row nothing uses is
# speculative documentation. Neither goes in the flake8 `--builtins` list: they only
# ever appear inside multi-token specs, so pyflakes never sees them as bare names.
#
# `tbatch` IS NOT `leaves`, and the same signature proves it. `target_pos` arrives
# `(tbatch, w, 3)` beside `leaf_positions` at `(leaves, w, 3)` -- measured 16 against
# 5. `tbatch` is one scan step's worth of target leaves (`target_leaf_batch_size`),
# so it is a tuning knob, not a count of anything in the tree. `w` IS shared, and
# structurally rather than by luck: the caller builds `target_pos` as
# `leaf_positions[safe_target_leaf_ids]`, a gather that cannot change the slot width.
#
# `tiles` IS NOT `blocks`. It is the sequence axis OUTSIDE the block/lane pair
# (observed 1 and 4), so the full layout is `tiles tbatch blocks blocksize` and
# `blocks blocksize` keeps the meaning it has in `_evaluate.py` and in the kernels
# above.
#
# THE TILED LAYOUT'S LEAF AXIS IS `farleaves`, NOT `leaves`, and this is the one
# judgement call here. `source_leaf_ids_tiles` is a reshape of the prepacked
# rectangle, which `_evaluate.py` measures as the FAR-field leaf view (radix 3
# leaves against octree 5). Its axis 1 did match the near-field leaf table at two
# distinct extents -- 5 beside `(5, 256)`, and 3 beside `(3, 2)` -- and that is
# exactly the evidence section 4.3 says not to trust: both captures ran on the radix
# backend, which is the lane where the two views coincide. The `farleaves` incident
# was 64 honest captures making the same mistake.


@jaxtyped(typechecker=beartype)
def _accumulate_target_block_tile_sequence(
    target_pos: Float[Array, "tbatch w 3"],
    target_mask: Bool[Array, "tbatch w"],
    tile_source_ids_seq: Int[Array, "tiles tbatch blocks blocksize"],
    tile_source_valid_seq: Bool[Array, "tiles tbatch blocks blocksize"],
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    *,
    g_const: Array,
    softening_sq: Array,
    tile_unroll: int,
    skip_empty_tiles: bool = False,
    componentwise_pairs: bool = False,
) -> Array:
    """Accumulate target-leaf accelerations from fixed-shape tile sequences.

    The innermost target-owned loop: one batch of target leaves against a sequence
    of source tiles, accumulated in place. Every tile has the same shape, which is
    what lets the reverse pass retain a residual proportional to the *tile* rather
    than to the edge list -- the property that makes gradients fit at galaxy N.

    Parameters
    ----------
    target_pos : Float[Array, 'tbatch w 3']
        Target-leaf positions for this batch ``[batch, W, 3]``.
    target_mask : Bool[Array, 'tbatch w']
        Target-leaf validity ``[batch, W]``.
    tile_source_ids_seq : Int[Array, 'tiles tbatch blocks blocksize']
        Source leaf ids per tile, ``[num_tiles, batch, lane_block, lane]``. The
        docstring said ``[num_tiles, batch, lanes]`` until the shape was derived by
        execution: it is rank FOUR, and the sibling
        :func:`_compute_target_block_pairs_from_source_tiles` had the layout right
        all along.
    tile_source_valid_seq : Bool[Array, 'tiles tbatch blocks blocksize']
        Per-lane validity with the same shape as ``tile_source_ids_seq``.
    leaf_positions : Float[Array, 'leaves w 3']
        Padded per-leaf positions ``[num_leaves, W, 3]``.
    leaf_masses : Float[Array, 'leaves w']
        Padded per-leaf masses ``[num_leaves, W]``.
    leaf_mask : Bool[Array, 'leaves w']
        Padded per-leaf validity ``[num_leaves, W]``; masked slots contribute
        exactly zero.
    g_const : Array
        Gravitational constant, pre-cast to the working dtype.
    softening_sq : Array
        Squared Plummer softening length.
    tile_unroll : int
        Unroll factor for the tile scan. Static under ``jit``; batching only.
    skip_empty_tiles : bool
        Skip tiles whose lanes are all invalid. Static under ``jit``. Safe because
        an all-invalid tile contributes exactly zero.
    componentwise_pairs : bool
        Use the ``dx``/``dy``/``dz`` pair kernel instead of the vector one. Static
        under ``jit``; avoids materialising a ``[batch, W, W, 3]`` difference.

    Returns
    -------
    Array
        Accumulated accelerations for this target batch, ``[batch, W, 3]``.
    """
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
    batch_body: Callable[[Array], Array],
) -> Array:
    """Collect fixed-shape target-leaf batch accumulations into leaf-major form.

    The batching loop shared by the three target-block lanes. It owns only the
    scan over leaf batches and the reshape back to leaf-major; all the physics
    lives in ``batch_body``.

    The trailing ``[:num_leaves]`` is what makes a ragged leaf count work: the
    scan runs ``ceil(num_leaves / target_leaf_batch_size)`` fixed-width batches,
    so the last one overhangs, and the overhang rows are dropped here rather than
    guarded inside the body. ``batch_body`` must therefore still return a
    full-width, well-defined block for a partial batch -- its own
    ``target_active`` masking is what keeps those rows finite.

    Parameters
    ----------
    num_leaves : int
        Number of target leaves. Static; sets the batch count and the final trim.
    leaf_size : int
        Slot capacity ``W`` per leaf. Static; must match the trailing width of
        what ``batch_body`` returns, or the reshape fails.
    target_leaf_batch_size : int
        Leaves per batch. Static. A tuning knob only -- it changes the scan
        shape, not the result.
    batch_scan_unroll : int
        ``lax.scan`` unroll factor. Static, tuning only.
    batch_body : Callable[[Array], Array]
        Called once per batch with the traced first leaf id of that batch, and
        must return ``[target_leaf_batch_size, leaf_size, 3]``. It is invoked
        under ``scan``, so it must be traceable and shape-invariant across
        batches.

    Returns
    -------
    Array
        ``[num_leaves, leaf_size, 3]`` accelerations in leaf-major order --
        **not** particle order; the caller scatters.

    Raises
    ------
    ValueError
        If ``target_leaf_batch_size`` or ``batch_scan_unroll`` is not positive.
        Both are host-side checks on static values.
    """
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


@jaxtyped(typechecker=beartype)
def _compute_target_block_pairs_from_source_tiles(
    positions: Float[Array, "n 3"],
    source_leaf_ids_tiles: Int[Array, "tiles farleaves blocks blocksize"],
    source_valid_tiles: Bool[Array, "tiles farleaves blocks blocksize"],
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    leaf_particle_idx: Int[Array, "leaves w"],
    *,
    g_const: Array,
    softening_sq: Array,
    target_leaf_batch_size: int,
    target_block_tile_scan_unroll: int,
    target_block_batch_scan_unroll: int,
    skip_empty_tiles: bool = False,
    componentwise_pairs: bool = False,
) -> Array:
    """Evaluate TONB pair contributions from canonical tiled source tensors.

    Target-owned-neighbour-block (TONB) evaluation from the canonical
    ``[tile, leaf, lane_block, lane]`` layout, i.e. source tiles already arranged
    per target leaf. Wraps :func:`_accumulate_target_block_tile_sequence` in the
    target-batch scan and scatters the result back to particle order.

    Parameters
    ----------
    positions : Float[Array, 'n 3']
        Particle positions ``[N, 3]``; also fixes the output shape.
    source_leaf_ids_tiles : Int[Array, 'tiles farleaves blocks blocksize']
        Source leaf ids in the canonical tiled layout.
    source_valid_tiles : Bool[Array, 'tiles farleaves blocks blocksize']
        Per-lane validity with the same shape as ``source_leaf_ids_tiles``.
    leaf_positions : Float[Array, 'leaves w 3']
        Padded per-leaf positions ``[num_leaves, W, 3]``.
    leaf_masses : Float[Array, 'leaves w']
        Padded per-leaf masses ``[num_leaves, W]``.
    leaf_mask : Bool[Array, 'leaves w']
        Padded per-leaf validity ``[num_leaves, W]``; masked slots contribute
        exactly zero.
    leaf_particle_idx : Int[Array, 'leaves w']
        Particle index behind each padded slot ``[num_leaves, W]``, clipped so a
        masked slot cannot gather out of bounds. Also fixes ``W``.
    g_const : Array
        Gravitational constant, pre-cast to the working dtype.
    softening_sq : Array
        Squared Plummer softening length.
    target_leaf_batch_size : int
        Target leaves per scan step. Static under ``jit``; batching only.
    target_block_tile_scan_unroll : int
        Unroll factor for the tile scan. Static under ``jit``.
    target_block_batch_scan_unroll : int
        Unroll factor for the target-batch scan. Static under ``jit``.
    skip_empty_tiles : bool
        Skip all-invalid tiles. Static under ``jit``.
    componentwise_pairs : bool
        Use the componentwise pair kernel. Static under ``jit``.

    Returns
    -------
    Array
        Per-particle accelerations ``[N, 3]``, zero outside this kernel's share.

    Raises
    ------
    ValueError
        If ``target_leaf_batch_size``, ``target_block_tile_scan_unroll`` or
        ``target_block_batch_scan_unroll`` is not positive.
    """
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


# `block_target_leaf_ids` is `blocks`; its sibling `block_offsets` stays rank-only. The
# pair looks interchangeable -- both are per-target-leaf bookkeeping beside the block
# table -- and they are not the same axis. Measured with `bench/annotation_pilot.py` on a
# re-record of this module, over the four recorded calls into the two target-block kernels:
#
#     block_offsets  block_target_leaf_ids  block_source_leaf_ids  leaf_positions
#            (5,)                  (5,)                 (5, 1)      (4, 2, 3)
#            (5,)                  (5,)                 (5, 1)      (0, 0, 3)
#            (4,)                  (5,)                 (5, 1)      (4, 2, 3)
#            (6,)                 (15,)                (15, 1)      (5, 256, 3)
#
# `block_target_leaf_ids` equals `block_source_leaf_ids.shape[0]` in all four, at two
# distinct extents, and the constructor says why: `_large_n_nearfield.py` builds the ids
# from `jnp.arange(total_blocks)` and the source table as `(total_blocks, k)`, so they are
# one axis by construction. Tying it closes the two silent acceptances measured here.
#
# `num_leaves + 1` for `block_offsets` is FALSE, and these docstrings claimed it until now:
# rows 2 and 3 break it -- 5 against an empty table, 4 against a 4-leaf one. It is the
# far-leaf view's axis, not this signature's `leaves`, which is what
# `test_short_block_offsets_is_still_accepted_and_that_is_deliberate` pins; and jaxtyping
# could not evaluate `farleaves+1` here anyway, because the parameter precedes the table
# that would bind `farleaves`.
#
# What the pilot still accepts in this module is not closable on the same parameter.
# `positions` is `n 3` and `n` occurs ONCE per signature, so shortening it binds freely and
# asserts nothing (4.4). The relation that would catch it is `leaf_particle_idx` VALUES
# against `n`, and `_prepare_leaf_data_from_groups` already clamps those deliberately.
# 11 accepted of 212 before this change, 9 after: seven `positions` shortenings across
# the family, and the two deliberate `block_offsets` the test above pins.
@partial(
    jax.jit,
    static_argnames=(
        "target_leaf_batch_size",
        "target_block_tile_size",
        "target_block_tile_scan_unroll",
        "target_block_batch_scan_unroll",
    ),
)
@jaxtyped(typechecker=beartype)
def _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl(
    positions: Float[Array, "n 3"],
    block_offsets: Int[Array, "_"],
    block_target_leaf_ids: Int[Array, "blocks"],
    block_source_leaf_ids: Int[Array, "blocks blocksize"],
    block_valid_mask: Bool[Array, "blocks blocksize"],
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    leaf_particle_idx: Int[Array, "leaves w"],
    *,
    G: Union[float, Array],
    softening_sq: Array,
    target_leaf_batch_size: int,
    target_block_tile_size: int,
    target_block_tile_scan_unroll: int,
    target_block_batch_scan_unroll: int,
) -> Array:
    """Target-owned pair path over prepacked fixed-width source-leaf blocks.

    TONB from a ``block_offsets`` run-length layout: each target leaf owns a
    contiguous run of fixed-width source blocks. Cross-leaf pairs only.

    Parameters
    ----------
    positions : Float[Array, 'n 3']
        Particle positions ``[N, 3]``; also fixes the output shape.
    block_offsets : Int[Array, '_']
        Start offset of each target leaf's block run. **Not**
        ``[num_leaves + 1]``: the axis is the FAR-leaf view's, and it was measured
        at 5 against a 4-leaf table, 5 against an empty one and 4 against a 4-leaf
        one in the same lane. Rank-only on purpose -- see the note above
        :func:`_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl`.
    block_target_leaf_ids : Int[Array, 'blocks']
        Target leaf id per block, ``[num_blocks]``.
    block_source_leaf_ids : Int[Array, 'blocks blocksize']
        Source leaf ids per block, ``[num_blocks, lanes]``.
    block_valid_mask : Bool[Array, 'blocks blocksize']
        Per-lane validity with the same shape as ``block_source_leaf_ids``.
    leaf_positions : Float[Array, 'leaves w 3']
        Padded per-leaf positions ``[num_leaves, W, 3]``.
    leaf_masses : Float[Array, 'leaves w']
        Padded per-leaf masses ``[num_leaves, W]``.
    leaf_mask : Bool[Array, 'leaves w']
        Padded per-leaf validity ``[num_leaves, W]``; masked slots contribute
        exactly zero.
    leaf_particle_idx : Int[Array, 'leaves w']
        Particle index behind each padded slot ``[num_leaves, W]``, clipped so a
        masked slot cannot gather out of bounds. Also fixes ``W``.
    G : Union[float, Array]
        Gravitational constant.
    softening_sq : Array
        Squared Plummer softening length.
    target_leaf_batch_size : int
        Target leaves per scan step. Static under ``jit``; batching only.
    target_block_tile_size : int
        Source-leaf lanes per tile. Static under ``jit``; batching only.
    target_block_tile_scan_unroll : int
        Unroll factor for the tile scan. Static under ``jit``; batching only.
    target_block_batch_scan_unroll : int
        Unroll factor for the target-batch scan. Static under ``jit``; batching only.

    Returns
    -------
    Array
        Per-particle accelerations ``[N, 3]``, zero outside this kernel's share.

    Raises
    ------
    ValueError
        If any of the four ``target_block*`` / ``target_leaf_batch_size`` knobs is
        not positive.
    """
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
@jaxtyped(typechecker=beartype)
def _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl(
    positions: Float[Array, "n 3"],
    block_source_leaf_ids_padded: Int[Array, "farleaves blocks blocksize"],
    block_valid_mask_padded: Bool[Array, "farleaves blocks blocksize"],
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    leaf_particle_idx: Int[Array, "leaves w"],
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
    """Target-major prepacked TONB path over the ``[leaf, block, lane]`` layout.

    The production large-N gradient path. Sources arrive already padded to a
    rectangle per target leaf, so no offset arithmetic happens at trace time --
    which is what keeps the reverse pass's residual proportional to the block.

    That rectangle is also the cost: it is padded to the global maximum neighbour
    count, and ``bench/audit_nearfield_padding.py`` measures the fill (45% at
    N=200000, 14.5% at N=1000000). ``occupancy_sort`` and ``skip_empty_tiles``
    are the cheap mitigations; tiering via ``build_leafpair_reverse_tiers`` is the
    other.

    Parameters
    ----------
    positions : Float[Array, 'n 3']
        Particle positions ``[N, 3]``; also fixes the output shape.
    block_source_leaf_ids_padded : Int[Array, 'farleaves blocks blocksize']
        Source leaf ids ``[num_leaves, blocks, lanes]``, padded to a rectangle.
    block_valid_mask_padded : Bool[Array, 'farleaves blocks blocksize']
        Per-lane validity with the same shape; padded lanes contribute exactly zero.
    leaf_positions : Float[Array, 'leaves w 3']
        Padded per-leaf positions ``[num_leaves, W, 3]``.
    leaf_masses : Float[Array, 'leaves w']
        Padded per-leaf masses ``[num_leaves, W]``.
    leaf_mask : Bool[Array, 'leaves w']
        Padded per-leaf validity ``[num_leaves, W]``; masked slots contribute
        exactly zero.
    leaf_particle_idx : Int[Array, 'leaves w']
        Particle index behind each padded slot ``[num_leaves, W]``, clipped so a
        masked slot cannot gather out of bounds. Also fixes ``W``.
    G : Union[float, Array]
        Gravitational constant.
    softening_sq : Array
        Squared Plummer softening length.
    target_leaf_batch_size : int
        Target leaves per scan step. Static under ``jit``; batching only.
    target_block_tile_size : int
        Source-leaf lanes per tile. Static under ``jit``; batching only.
    target_block_tile_scan_unroll : int
        Unroll factor for the tile scan. Static under ``jit``; batching only.
    target_block_batch_scan_unroll : int
        Unroll factor for the target-batch scan. Static under ``jit``; batching only.
    occupancy_sort : bool
        Process target leaves in occupancy order so tiles fill more evenly. Static
        under ``jit``; a permutation of an associative sum, so it can move the last
        digits.
    skip_empty_tiles : bool
        Skip all-invalid tiles. Static under ``jit``.
    componentwise_pairs : bool
        Use the componentwise pair kernel. Static under ``jit``.

    Returns
    -------
    Array
        Per-particle accelerations ``[N, 3]``, zero outside this kernel's share.

    Raises
    ------
    ValueError
        If ``target_block_tile_size`` is not positive.
    """
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
@jaxtyped(typechecker=beartype)
def _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_tiled_impl(
    positions: Float[Array, "n 3"],
    block_offsets: Int[Array, "_"],
    block_target_leaf_ids: Int[Array, "blocks"],
    block_source_leaf_ids: Int[Array, "blocks blocksize"],
    block_valid_mask: Bool[Array, "blocks blocksize"],
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    leaf_particle_idx: Int[Array, "leaves w"],
    *,
    G: Union[float, Array],
    softening_sq: Array,
    target_leaf_batch_size: int,
    target_block_tile_size: int,
    target_block_tile_scan_unroll: int,
    target_block_batch_scan_unroll: int,
) -> Array:
    """Bounded overflow TONB pair kernel using canonical tiled source tensors.

    The overflow path: used when the prepacked rectangle would exceed its block
    cap, so the blocks are re-expressed as canonical tiles with a bounded extent
    instead. Same numbers as
    :func:`_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl`; it differs
    only in how the source side is laid out.

    Parameters
    ----------
    positions : Float[Array, 'n 3']
        Particle positions ``[N, 3]``; also fixes the output shape.
    block_offsets : Int[Array, '_']
        Start offset of each target leaf's block run. **Not**
        ``[num_leaves + 1]``: the axis is the FAR-leaf view's, and it was measured
        at 5 against a 4-leaf table, 5 against an empty one and 4 against a 4-leaf
        one in the same lane. Rank-only on purpose -- see the note above
        :func:`_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl`.
    block_target_leaf_ids : Int[Array, 'blocks']
        Target leaf id per block, ``[num_blocks]``.
    block_source_leaf_ids : Int[Array, 'blocks blocksize']
        Source leaf ids per block, ``[num_blocks, lanes]``.
    block_valid_mask : Bool[Array, 'blocks blocksize']
        Per-lane validity with the same shape as ``block_source_leaf_ids``.
    leaf_positions : Float[Array, 'leaves w 3']
        Padded per-leaf positions ``[num_leaves, W, 3]``.
    leaf_masses : Float[Array, 'leaves w']
        Padded per-leaf masses ``[num_leaves, W]``.
    leaf_mask : Bool[Array, 'leaves w']
        Padded per-leaf validity ``[num_leaves, W]``; masked slots contribute
        exactly zero.
    leaf_particle_idx : Int[Array, 'leaves w']
        Particle index behind each padded slot ``[num_leaves, W]``, clipped so a
        masked slot cannot gather out of bounds. Also fixes ``W``.
    G : Union[float, Array]
        Gravitational constant.
    softening_sq : Array
        Squared Plummer softening length.
    target_leaf_batch_size : int
        Target leaves per scan step. Static under ``jit``; batching only.
    target_block_tile_size : int
        Source-leaf lanes per tile. Static under ``jit``; batching only.
    target_block_tile_scan_unroll : int
        Unroll factor for the tile scan. Static under ``jit``; batching only.
    target_block_batch_scan_unroll : int
        Unroll factor for the target-batch scan. Static under ``jit``; batching only.

    Returns
    -------
    Array
        Per-particle accelerations ``[N, 3]``, zero outside this kernel's share.

    Raises
    ------
    ValueError
        If ``target_block_tile_size`` is not positive.
    """
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
    """Evaluate target-block pair contributions without intra-leaf self work.

    Public entry point for the TONB pair path. Cross-leaf only -- the intra-leaf
    self block is the caller's job, which is what lets the distributed driver
    evaluate the two on different device shares.

    Parameters
    ----------
    positions_sorted : Array
        Particle positions ``[N, 3]`` in Morton order.
    masses_sorted : Array
        Particle masses ``[N]`` in the same order.
    leaf_particle_indices : Array
        Explicit per-leaf particle membership ``[num_leaves, W]``.
    leaf_particle_mask : Array
        Validity for that membership table, same shape.
    block_offsets : Array
        Start offset of each target leaf's block run. **Not**
        ``[num_leaves + 1]``: the axis is the FAR-leaf view's, and it was measured
        at 5 against a 4-leaf table, 5 against an empty one and 4 against a 4-leaf
        one in the same lane. Rank-only on purpose -- see the note above
        :func:`_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl`.
    block_target_leaf_ids : Array
        Target leaf id per block, ``[num_blocks]``.
    block_source_leaf_ids : Array
        Source leaf ids per block, ``[num_blocks, lanes]``.
    block_valid_mask : Array
        Per-lane validity with the same shape as ``block_source_leaf_ids``.
    G : Union[float, Array]
        Gravitational constant. Default ``1.0``.
    softening : float
        Plummer softening **length** (squared internally). Must be a concrete
        Python float, not a tracer. Default ``0.0``.
    target_leaf_batch_size : int
        Target leaves per scan step. Static under ``jit``; batching only.
    target_block_tile_size : int
        Source-leaf lanes per tile. Static under ``jit``; batching only.
    target_block_tile_scan_unroll : int
        Unroll factor for the tile scan. Static under ``jit``; batching only.
    target_block_batch_scan_unroll : int
        Unroll factor for the target-batch scan. Static under ``jit``; batching only.
    target_block_overflow_fast_max_blocks : int
        Cap on the blocks the prepacked fast path will materialise; above it the
        bounded tiled overflow kernel runs instead. Static under ``jit``; see
        :data:`~jaccpot.runtime.fmm_constants._NEARFIELD_TARGET_BLOCK_OVERFLOW_FAST_MAX_BLOCKS`.

    Returns
    -------
    Array
        Per-particle accelerations ``[N, 3]``, zero outside this kernel's share.
    """
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


# NO `@jaxtyped` HERE, AND IT IS NOT AN OVERSIGHT -- its five siblings in this module have
# one, so the asymmetry is the kind that gets "fixed" on sight. The body is two delegated
# calls and nothing else: `_compute_leaf_p2p_prepared_large_n_self_only_impl` and
# `_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl`, both decorated, both
# passed every array UNMODIFIED. Between them all nine array parameters are already
# validated one frame deeper against the identical annotations -- `n 3` and the four
# `leaves w` from the first, `blocks blocksize` and the two `_` from the second. So this is
# STYLE_GUIDE section 4.1's own rule ("skip it when the value flows straight into
# something that checks it"), with the checker being a sibling in the same file rather than
# a library, and a decorator here would be the "annotate for consistency" that section
# warns against: an extra beartype pass per call on the production forward path, which
# section 4.4 records as UNCONDITIONAL, for a check that already runs.
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
    """Specialized accel-only kernel using prepacked target-owned source blocks.

    Self block **plus** cross-leaf pairs, accelerations only -- no potential. The
    accel-only specialisation exists because the potential doubles the reverse
    pass's retained state for a quantity the force path never reads.

    Parameters
    ----------
    positions : Array
        Particle positions ``[N, 3]``; also fixes the output shape.
    block_offsets : Array
        Start offset of each target leaf's block run. **Not**
        ``[num_leaves + 1]``: the axis is the FAR-leaf view's, and it was measured
        at 5 against a 4-leaf table, 5 against an empty one and 4 against a 4-leaf
        one in the same lane. Rank-only on purpose -- see the note above
        :func:`_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl`.
    block_target_leaf_ids : Array
        Target leaf id per block, ``[num_blocks]``.
    block_source_leaf_ids : Array
        Source leaf ids per block, ``[num_blocks, lanes]``.
    block_valid_mask : Array
        Per-lane validity with the same shape as ``block_source_leaf_ids``.
    leaf_positions : Array
        Padded per-leaf positions ``[num_leaves, W, 3]``.
    leaf_masses : Array
        Padded per-leaf masses ``[num_leaves, W]``.
    leaf_mask : Array
        Padded per-leaf validity ``[num_leaves, W]``; masked slots contribute
        exactly zero.
    leaf_particle_idx : Array
        Particle index behind each padded slot ``[num_leaves, W]``, clipped so a
        masked slot cannot gather out of bounds. Also fixes ``W``.
    G : Union[float, Array]
        Gravitational constant.
    softening_sq : Array
        Squared Plummer softening length.
    target_leaf_batch_size : int
        Target leaves per scan step. Static under ``jit``; batching only.
    target_block_tile_size : int
        Source-leaf lanes per tile. Static under ``jit``; batching only.
    target_block_tile_scan_unroll : int
        Unroll factor for the tile scan. Static under ``jit``; batching only.
    target_block_batch_scan_unroll : int
        Unroll factor for the target-batch scan. Static under ``jit``; batching only.

    Returns
    -------
    Array
        Per-particle accelerations ``[N, 3]``, zero outside this kernel's share.
    """
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
@jaxtyped(typechecker=beartype)
def _compute_leaf_p2p_prepared_large_n_accel_only_impl(
    positions: Float[Array, "n 3"],
    target_leaf_ids: Int[Array, "pairs"],
    source_leaf_ids: Int[Array, "pairs"],
    valid_pairs: Bool[Array, "pairs"],
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    leaf_particle_idx: Int[Array, "leaves w"],
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
    """Specialized accel-only kernel for large-N bucketed prepared leaf data.

    Self block plus cross-leaf pairs on the edge-list (bucketed) layout,
    accelerations only. A thin sum of
    :func:`_compute_leaf_p2p_prepared_large_n_self_only_impl` and
    :func:`_compute_leaf_p2p_prepared_large_n_pairs_only_impl`, and the delegation
    precedent the Tier 1.2 dedupe followed.

    Parameters
    ----------
    positions : Float[Array, 'n 3']
        Particle positions ``[N, 3]``; also fixes the output shape.
    target_leaf_ids : Int[Array, 'pairs']
        Target leaf id per edge, ``[num_edges]``.
    source_leaf_ids : Int[Array, 'pairs']
        Source leaf id per edge, ``[num_edges]``.
    valid_pairs : Bool[Array, 'pairs']
        Per-edge validity ``[num_edges]``; padded edges contribute exactly zero.
    leaf_positions : Float[Array, 'leaves w 3']
        Padded per-leaf positions ``[num_leaves, W, 3]``.
    leaf_masses : Float[Array, 'leaves w']
        Padded per-leaf masses ``[num_leaves, W]``.
    leaf_mask : Bool[Array, 'leaves w']
        Padded per-leaf validity ``[num_leaves, W]``; masked slots contribute
        exactly zero.
    leaf_particle_idx : Int[Array, 'leaves w']
        Particle index behind each padded slot ``[num_leaves, W]``, clipped so a
        masked slot cannot gather out of bounds. Also fixes ``W``.
    G : Union[float, Array]
        Gravitational constant.
    softening_sq : Array
        Squared Plummer softening length.
    edge_chunk_size : int
        Edges per chunk. Static under ``jit``; batching only.
    chunks_per_superchunk : int
        Chunks grouped into one superchunk before the target reduction. Static
        under ``jit``.
    chunk_scan_batch_size : int
        Chunks consumed per scan step. Static under ``jit``.
    chunk_scan_unroll : int
        Unroll factor for the chunk scan. Static under ``jit``.
    superchunk_scan_unroll : int
        Unroll factor for the superchunk scan. Static under ``jit``.
    sorted_scatter_hint : bool
        Tell the scatter its indices are sorted. Static under ``jit``. A *promise*,
        not a request: it must match the data or the scatter is wrong.
    grouped_sorted_scatter : bool
        Use the segment-grouped scatter. Static under ``jit``; same caveat.
    superchunk_target_reduce : bool
        Reduce per target leaf within a superchunk before scattering. Static under
        ``jit``; changes the summation *grouping*, so it can move the last digits.
    disable_chunk_cond : bool
        Skip the per-chunk ``lax.cond`` early-out. Static under ``jit``.

    Returns
    -------
    Array
        Per-particle accelerations ``[N, 3]``, zero outside this kernel's share.
    """
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
