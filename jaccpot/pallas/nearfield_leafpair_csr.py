"""Leaf-pair near field driven by the neighbour CSR: one Pallas program per row chunk.

The production leaf-pair kernel (:func:`jaccpot.pallas.nearfield_fused_leaf.
nearfield_leafpair_pallas`) runs over a per-leaf RECTANGLE of source slots,
``(num_leaves, S)`` with ``S`` the neighbour cap, on a grid of
``(num_leaves, subtiles, ceil(S / chunk))`` programs. Measured 2026-09-11 on
the leaf-64 fused step at N = 2x10^5 (plan ``sub-10ms-fmm-200k``, Phase 4.1):
``S = 4096`` slots per leaf hold 1.84M real entries out of 12.8M, so 86 % of the
400k programs loop over 64 invalid slots and do nothing, and the rectangle
itself (ids + mask, 64 MB) is rebuilt from the CSR every step.

This module keeps the kernel's inner loop -- one source leaf gathered by id,
op for op the same lane body, so forces agree to float32 summation order --
and changes only WHAT a program owns: a chunk of at most ``chunk`` consecutive
entries of ONE target leaf's CSR row, from a chunk table built once per step
in a few small XLA ops (:func:`build_leafpair_chunk_table`). Every program
does real work, the grid is ``(num_chunks, subtiles)`` with ``num_chunks``
the static capacity ``ceil(edge_capacity / chunk) + num_leaves`` (never
overflows: each leaf rounds up at most once), and the per-chunk partials are
reduced by a sorted segment sum onto the leaves. The intra-leaf self term
rides on each leaf's first chunk (``include_self``), as the rectangle kernel
does on chunk 0.

The two-level accumulator of the rectangle kernel is kept (``accum="wide"``:
a float32 partial per source leaf, float64 across leaves, one downcast at
the end -- memory ``nearfield-accumulator-fix``); the per-chunk partials are
emitted in the wide dtype so nothing is lost before the reduce.

Triton lowering notes (the same traps as ``treecode_walk_pallas.py``): every
dynamically indexed buffer is a whole-array ref; the target tile is a
``pl.ds`` vector load at a data-dependent leaf; the source loop is a
``fori_loop`` with a TRACED trip count (as ``m2l_real_csr.py`` does), which is
what removes the per-slot ``lax.cond`` of the rectangle kernel.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from jaccpot.pallas._compat import KernelRef
from jaccpot.pallas.nearfield_fused_leaf import (
    _OUT_WIDTH,
    _POS_WIDTH,
    _require_shape,
    _resolve_accum_dtype,
    _resolve_subtile,
)

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None
    plgpu = None

__all__ = [
    "LeafPairChunkTable",
    "build_leafpair_chunk_table",
    "leafpair_chunk_capacity",
    "nearfield_leafpair_csr_pallas",
    "nearfield_leafpair_csr_pallas_cvjp",
    "nearfield_leafpair_csr_reverse_pallas",
    "nearfield_leafpair_csr_jax",
]


class LeafPairChunkTable(NamedTuple):
    """Row chunks of a leaf neighbour CSR, one program each.

    Every field has the static length :func:`leafpair_chunk_capacity`; entries
    past the live prefix have ``leaf == -1`` and ``count == 0``.

    Attributes
    ----------
    leaf : Array
        Target leaf of the chunk (``-1`` = padding).
    start : Array
        Index into the flat neighbour array of the chunk's first entry.
    count : Array
        Number of live entries in the chunk, ``0 <= count <= chunk``.
    is_first : Array
        ``1`` on the first chunk of each leaf (the one that carries the
        intra-leaf self term when ``include_self``), else ``0``.
    """

    leaf: Array
    start: Array
    count: Array
    is_first: Array


def leafpair_chunk_capacity(edge_capacity: int, num_leaves: int, chunk: int) -> int:
    """Static number of chunk programs that can never overflow.

    Parameters
    ----------
    edge_capacity : int
        Length of the flat neighbour array (the directed near-pair cap).
    num_leaves : int
        Number of target leaves (rows).
    chunk : int
        Entries per chunk.

    Returns
    -------
    int
        ``ceil(edge_capacity / chunk) + num_leaves``: each row rounds up to a
        whole chunk at most once and every row has at least one chunk (for the
        self term), so the sum of per-row chunk counts is bounded by this.
    """
    chunk = max(1, int(chunk))
    return (int(edge_capacity) + chunk - 1) // chunk + int(num_leaves)


@jaxtyped(typechecker=beartype)
def build_leafpair_chunk_table(
    # `_` and not `leaves+1`, though that is what it is: jaxtyping evaluates a
    # symbolic axis only against names already bound, and `offsets` PRECEDES
    # `counts` here, so `leaves` is unbound at that point ("Cannot process
    # symbolic axis 'leaves+1' as some axis names have not been processed").
    # The same constraint is documented in runtime/_adaptive_policy.py and
    # nearfield/_large_n_blocks.py. A bare `leavesp1` reads as a forward
    # reference to flake8 and trips F821.
    offsets: Int[Array, "_"],
    counts: Int[Array, "leaves"],
    *,
    chunk: int,
    capacity: int,
) -> LeafPairChunkTable:
    """Split every leaf's CSR row into chunks of at most ``chunk`` entries.

    A leaf with an empty row still gets ONE chunk (``count == 0``,
    ``is_first == 1``) so its self term has a program to run on.

    Parameters
    ----------
    offsets : Int[Array, '_']
        CSR row starts, ``num_leaves + 1`` entries.
    counts : Int[Array, 'leaves']
        Row lengths (``offsets[1:] - offsets[:-1]``; passed separately because
        the callers already hold them).
    chunk : int
        Entries per chunk. Static.
    capacity : int
        Static table length; use :func:`leafpair_chunk_capacity` so it cannot
        overflow. Rows beyond the live prefix are padding.

    Returns
    -------
    LeafPairChunkTable
        The chunk table, every field of length ``capacity``.
    """
    idx = counts.dtype
    chunk_i = jnp.asarray(int(chunk), idx)
    num_leaves = int(counts.shape[0])
    capacity = int(capacity)
    per_leaf = jnp.maximum(jnp.asarray(1, idx), (counts + chunk_i - 1) // chunk_i)
    ends = jnp.cumsum(per_leaf, dtype=idx)  # inclusive
    first = ends - per_leaf  # exclusive
    c = jnp.arange(capacity, dtype=idx)
    # leaf of chunk c: the first row whose inclusive end exceeds c
    leaf = jnp.searchsorted(ends, c, side="right").astype(idx)
    valid = leaf < jnp.asarray(num_leaves, idx)
    leaf_safe = jnp.minimum(leaf, jnp.asarray(max(num_leaves - 1, 0), idx))
    k = c - first[leaf_safe]
    start = offsets[leaf_safe] + k * chunk_i
    count = jnp.clip(counts[leaf_safe] - k * chunk_i, 0, chunk_i)
    is_first = (k == 0) & valid
    return LeafPairChunkTable(
        leaf=jnp.where(valid, leaf, jnp.asarray(-1, idx)).astype(idx),
        start=jnp.where(valid, start, jnp.asarray(0, idx)).astype(idx),
        count=jnp.where(valid, count, jnp.asarray(0, idx)).astype(idx),
        is_first=is_first.astype(idx),
    )


def _nearfield_leafpair_csr_kernel(
    table_pos_ref: KernelRef,
    table_mass_ref: KernelRef,
    table_mask_ref: KernelRef,
    leaf_count_ref: KernelRef,
    neighbors_ref: KernelRef,
    chunk_leaf_ref: KernelRef,
    chunk_start_ref: KernelRef,
    chunk_count_ref: KernelRef,
    chunk_first_ref: KernelRef,
    softening_sq_ref: KernelRef,
    g_ref: KernelRef,
    out_ref: KernelRef,
    *,
    leaf_width: int,
    subtile: int,
    accum_dtype: Any,
    out_dtype: Any,
    include_self: bool,
) -> None:
    """One CSR row chunk against one target subtile of its leaf.

    Parameters
    ----------
    table_pos_ref : KernelRef
        FULL particle position table ``(L, W, _POS_WIDTH)``: the target tile is
        a dynamic ``pl.ds`` load at the chunk's leaf and every source leaf is
        gathered by id.
    table_mass_ref : KernelRef
        Full mass table ``(L, W)``.
    table_mask_ref : KernelRef
        Full validity table ``(L, W)``.
    leaf_count_ref : KernelRef
        Occupancy per leaf ``(L,)``; the source and target loops run to it
        instead of the padded width.
    neighbors_ref : KernelRef
        Flat neighbour array of LEAF indices ``(E,)``.
    chunk_leaf_ref : KernelRef
        Chunk table: target leaf per chunk ``(C,)``, ``-1`` = padding.
    chunk_start_ref : KernelRef
        Chunk table: first entry in ``neighbors`` ``(C,)``.
    chunk_count_ref : KernelRef
        Chunk table: live entries ``(C,)``.
    chunk_first_ref : KernelRef
        Chunk table: ``1`` on each leaf's first chunk ``(C,)``.
    softening_sq_ref : KernelRef
        Squared softening length ``(1,)``.
    g_ref : KernelRef
        Gravitational constant ``(1,)``.
    out_ref : KernelRef
        **Output** ``(1, Bt, _OUT_WIDTH)``: the chunk's partial acceleration
        (lanes 0:3) and potential (lane 3) for this subtile.
    leaf_width : int
        ``W``. Static.
    subtile : int
        ``Bt``. Static.
    accum_dtype : Any
        ``None`` = accumulate in the input dtype; else the wide dtype of the
        cross-leaf accumulator (float32 partial per source leaf).
    out_dtype : Any
        Dtype of ``out_ref`` (the wide dtype when ``accum_dtype`` is set, so
        the caller's reduce sees the full precision).
    include_self : bool
        Add the leaf's own particles, diagonal removed, on its first chunk.

    Returns
    -------
    None
        The result is the write to ``out_ref``.
    """
    c = pl.program_id(0)
    sub = pl.program_id(1)
    bt = int(subtile)
    tl = chunk_leaf_ref[c]
    live = tl >= 0
    tl_safe = jnp.maximum(tl, 0)
    off = sub * bt
    lanes = pl.ds(off, bt)
    tcount = leaf_count_ref[tl_safe]

    # The block is written whatever happens; the work below only runs for a
    # live chunk whose target subtile holds at least one particle. Cell leaves
    # average 18 of 64 slots at N=2e5, so the second subtile of most leaves is
    # empty and used to loop over every source for nothing.
    zeros_bt = jnp.zeros((bt,), out_dtype)
    for comp in range(4):
        out_ref[0, :, comp] = zeros_bt

    @pl.when(live & (off < tcount))
    def _work():
        tx = table_pos_ref[tl_safe, lanes, 0]
        ty = table_pos_ref[tl_safe, lanes, 1]
        tz = table_pos_ref[tl_safe, lanes, 2]
        tvalid = table_mask_ref[tl_safe, lanes]
        soft = softening_sq_ref[0]
        g_value = g_ref[0]

        zero = jnp.zeros_like(tx)
        wide = accum_dtype is not None
        acc0 = (
            tuple(jnp.zeros(tx.shape, accum_dtype) for _ in range(4))
            if wide
            else (zero, zero, zero, zero)
        )

        def _leaf_pass(sid, acc, exclude_lane=None):
            # The rectangle kernel's lane body, op for op (nearfield_fused_leaf.py),
            # so the two lanes agree to float32 summation order. The loop runs to
            # the source leaf's occupancy (its live slots are a prefix): the
            # skipped slots were masked to exact zeros, so the sums are unchanged.
            def _lane_body(j, acc):
                acc_x, acc_y, acc_z, acc_p = acc
                sx = table_pos_ref[sid, j, 0]
                sy = table_pos_ref[sid, j, 1]
                sz = table_pos_ref[sid, j, 2]
                sm = table_mass_ref[sid, j]
                lane_valid = table_mask_ref[sid, j]
                dx = tx - sx
                dy = ty - sy
                dz = tz - sz
                dist_sq = dx * dx + dy * dy + dz * dz + soft
                active = tvalid & lane_valid
                if exclude_lane is not None:
                    active = active & (exclude_lane != j)
                safe_dist_sq = jnp.where(active, dist_sq, 1.0)
                inv_r = lax.rsqrt(safe_dist_sq)
                inv_r = jnp.where(active, inv_r, 0.0)
                inv_dist3 = inv_r * inv_r * inv_r
                scale = -g_value * inv_dist3 * sm
                acc_x = acc_x + scale * dx
                acc_y = acc_y + scale * dy
                acc_z = acc_z + scale * dz
                acc_p = acc_p - g_value * inv_r * sm
                return (acc_x, acc_y, acc_z, acc_p)

            scount = leaf_count_ref[sid]
            if not wide:
                return lax.fori_loop(0, scount, _lane_body, acc)
            part = lax.fori_loop(0, scount, _lane_body, (zero, zero, zero, zero))
            return tuple(a + q.astype(accum_dtype) for a, q in zip(acc, part))

        start = chunk_start_ref[c]
        cnt = chunk_count_ref[c]

        def _slot_body(s, acc):
            sid = neighbors_ref[start + s]
            return _leaf_pass(sid, acc)

        # Traced trip count: padding chunks and short rows cost one comparison,
        # not ``chunk`` predicated iterations.
        acc = lax.fori_loop(0, cnt, _slot_body, acc0)

        if include_self:
            lane_idx = off + lax.broadcasted_iota(jnp.int32, (bt,), 0)

            def _self_pass(acc):
                return _leaf_pass(tl_safe, acc, exclude_lane=lane_idx)

            acc = lax.cond(chunk_first_ref[c] != 0, _self_pass, lambda acc: acc, acc)

        acc_x, acc_y, acc_z, acc_p = acc
        if wide and out_dtype != accum_dtype:
            acc_x = acc_x.astype(out_dtype)
            acc_y = acc_y.astype(out_dtype)
            acc_z = acc_z.astype(out_dtype)
            acc_p = acc_p.astype(out_dtype)
        zero_out = jnp.zeros_like(acc_x)
        out_ref[0, :, 0] = jnp.where(tvalid, acc_x, zero_out)
        out_ref[0, :, 1] = jnp.where(tvalid, acc_y, zero_out)
        out_ref[0, :, 2] = jnp.where(tvalid, acc_z, zero_out)
        out_ref[0, :, 3] = jnp.where(tvalid, acc_p, zero_out)


@jaxtyped(typechecker=beartype)
def nearfield_leafpair_csr_pallas(
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    neighbors: Int[Array, "edges"],
    chunks: LeafPairChunkTable,
    *,
    softening_sq: Array,
    G: Array,
    chunk: int,
    num_warps: int | None = None,
    num_stages: int = 1,
    target_subtile: int | None = None,
    interpret: bool = False,
    accum: str = "input",
    include_self: bool = True,
) -> Array:
    """Leaf-pair near field from a neighbour CSR, one Pallas program per row chunk.

    Parameters
    ----------
    leaf_positions : Float[Array, 'leaves w 3']
        Particle coordinates per leaf ``(L, W, 3)`` -- targets and the source
        gather table alike.
    leaf_masses : Float[Array, 'leaves w']
        Particle masses ``(L, W)``.
    leaf_mask : Bool[Array, 'leaves w']
        Which slots hold a real particle ``(L, W)``.
    neighbors : Int[Array, 'edges']
        Flat neighbour array of LEAF indices (``0 <= id < L``); entries past the
        rows' live ranges are never read.
    chunks : LeafPairChunkTable
        From :func:`build_leafpair_chunk_table` over this CSR.
    softening_sq : Array
        Scalar squared softening length.
    G : Array
        Scalar gravitational constant.
    chunk : int
        Entries per chunk, the value the table was built with. Static.
    num_warps : int | None
        Triton warps per program; ``None`` = ``max(1, Bt // 32)``.
    num_stages : int
        Triton pipelining depth.
    target_subtile : int | None
        Targets per program (``Bt``); ``None`` = 32, clamped to ``W``.
    interpret : bool
        Pallas interpret mode (CPU semantics).
    accum : str
        ``"input"`` (one float32 accumulator) or ``"wide"`` (float64 across
        source leaves, float32 within one) -- see the rectangle kernel.
    include_self : bool
        Add the intra-leaf term on each leaf's first chunk. Default True: the
        CSR never lists a leaf in its own row (the walk emits near pairs for
        ``different_nodes`` only).

    Returns
    -------
    Array
        ``(L, W, _OUT_WIDTH)``: acceleration lanes 0:3, potential lane 3, in
        the input dtype.

    Raises
    ------
    RuntimeError
        If Pallas or its Triton backend could not be imported.
    """
    if pl is None or plgpu is None:
        raise RuntimeError("jax.experimental.pallas is not available")

    leaf_positions = jnp.asarray(leaf_positions)
    dtype = leaf_positions.dtype
    leaf_masses = jnp.asarray(leaf_masses, dtype=dtype)
    leaf_mask = jnp.asarray(leaf_mask, dtype=bool)
    num_leaves = int(leaf_positions.shape[0])
    leaf_width = int(leaf_positions.shape[1])
    _require_shape("leaf_masses", leaf_masses, (num_leaves, leaf_width))
    _require_shape("leaf_mask", leaf_mask, (num_leaves, leaf_width))
    capacity = int(chunks.leaf.shape[0])
    for name in ("start", "count", "is_first"):
        _require_shape(f"chunks.{name}", getattr(chunks, name), (capacity,))
    if num_leaves == 0 or leaf_width == 0:
        return jnp.zeros((num_leaves, leaf_width, _OUT_WIDTH), dtype=dtype)

    idx = chunks.leaf.dtype
    neighbors = jnp.asarray(neighbors, dtype=idx)
    softening_sq_arr = jnp.asarray([softening_sq], dtype=dtype)
    g_arr = jnp.asarray([G], dtype=dtype)

    pos_padded = jnp.pad(leaf_positions, ((0, 0), (0, 0), (0, _POS_WIDTH - 3)))
    bt = _resolve_subtile(target_subtile, leaf_width)
    width_pad = ((leaf_width + bt - 1) // bt) * bt
    n_sub = width_pad // bt
    pad_t = width_pad - leaf_width
    if pad_t:
        # The target tile is loaded with ``pl.ds`` from the same table the
        # sources come from, so pad the table itself; padded lanes are masked.
        pos_padded = jnp.pad(pos_padded, ((0, 0), (0, pad_t), (0, 0)))
        leaf_masses = jnp.pad(leaf_masses, ((0, 0), (0, pad_t)))
        leaf_mask = jnp.pad(leaf_mask, ((0, 0), (0, pad_t)))
    if num_warps is None:
        num_warps = max(1, bt // 32)
    accum_dtype = _resolve_accum_dtype(accum, dtype)
    partial_dtype = accum_dtype if accum_dtype is not None else dtype
    include_self = bool(include_self)

    def _kernel(*refs):
        return _nearfield_leafpair_csr_kernel(
            *refs,
            leaf_width=width_pad,
            subtile=bt,
            accum_dtype=accum_dtype,
            out_dtype=partial_dtype,
            include_self=include_self,
        )

    def _full(arr: Array) -> pl.BlockSpec:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    # one past the last live slot of each leaf: the source loops run to it
    slot_no = jnp.arange(width_pad, dtype=jnp.int32)[None, :] + 1
    leaf_count = jnp.max(
        jnp.where(leaf_mask, slot_no, jnp.zeros_like(slot_no)), axis=1
    ).astype(jnp.int32)
    kernel = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(
            (capacity, width_pad, _OUT_WIDTH), partial_dtype
        ),
        in_specs=[
            _full(pos_padded),
            _full(leaf_masses),
            _full(leaf_mask),
            _full(leaf_count),
            _full(neighbors),
            _full(chunks.leaf),
            _full(chunks.start),
            _full(chunks.count),
            _full(chunks.is_first),
            pl.BlockSpec((1,), lambda c, sub: (0,)),
            pl.BlockSpec((1,), lambda c, sub: (0,)),
        ],
        out_specs=pl.BlockSpec((1, bt, _OUT_WIDTH), lambda c, sub: (c, sub, 0)),
        grid=(capacity, n_sub),
        compiler_params=plgpu.CompilerParams(
            num_warps=int(num_warps), num_stages=int(num_stages)
        ),
        interpret=bool(interpret),
        name=(
            f"nearfield_leafpair_csr_t{bt}_c{int(chunk)}_w{leaf_width}_a{accum}"
            f"{'_self' if include_self else ''}"
        ),
    )
    partials = kernel(
        pos_padded,
        leaf_masses,
        leaf_mask,
        leaf_count,
        neighbors,
        chunks.leaf,
        chunks.start,
        chunks.count,
        chunks.is_first,
        softening_sq_arr,
        g_arr,
    )
    # Sorted segment sum onto the leaves: chunks of one leaf are consecutive and
    # padding chunks (exact zeros) go to an extra segment that is sliced off.
    seg = jnp.where(chunks.leaf < 0, jnp.asarray(num_leaves, idx), chunks.leaf)
    out = jax.ops.segment_sum(
        partials, seg, num_segments=num_leaves + 1, indices_are_sorted=True
    )[:num_leaves]
    out = out.astype(dtype)
    if pad_t:
        out = out[:, :leaf_width, :]
    return out


@jaxtyped(typechecker=beartype)
def nearfield_leafpair_csr_jax(
    leaf_positions: Float[Array, "leaves w 3"],
    leaf_masses: Float[Array, "leaves w"],
    leaf_mask: Bool[Array, "leaves w"],
    neighbors: Int[Array, "edges"],
    # `_` and not `leaves+1`, though that is what it is: jaxtyping evaluates a
    # symbolic axis only against names already bound, and `offsets` PRECEDES
    # `counts` here, so `leaves` is unbound at that point ("Cannot process
    # symbolic axis 'leaves+1' as some axis names have not been processed").
    # The same constraint is documented in runtime/_adaptive_policy.py and
    # nearfield/_large_n_blocks.py. A bare `leavesp1` reads as a forward
    # reference to flake8 and trips F821.
    offsets: Int[Array, "_"],
    counts: Int[Array, "leaves"],
    *,
    softening_sq: Array,
    G: Array,
    include_self: bool = True,
) -> Array:
    """Dense pure-JAX twin of :func:`nearfield_leafpair_csr_pallas` (the reference).

    Expands the CSR into the rectangle ``(L, max_count)`` and sums every valid
    source leaf per target with a float64-free but order-independent
    formulation (a full ``(L, W, S*W)`` reduction), so it is exact up to the
    input dtype's round-off and small enough only for tests.

    Parameters
    ----------
    leaf_positions : Float[Array, 'leaves w 3']
        Particle coordinates per leaf.
    leaf_masses : Float[Array, 'leaves w']
        Particle masses.
    leaf_mask : Bool[Array, 'leaves w']
        Slot validity.
    neighbors : Int[Array, 'edges']
        Flat neighbour array of leaf indices.
    offsets : Int[Array, '_']
        CSR row starts.
    counts : Int[Array, 'leaves']
        CSR row lengths.
    softening_sq : Array
        Scalar squared softening length.
    G : Array
        Scalar gravitational constant.
    include_self : bool
        Add the intra-leaf term, diagonal removed.

    Returns
    -------
    Array
        ``(L, W, _OUT_WIDTH)`` acceleration and potential.
    """
    from jaccpot.pallas.nearfield_fused_leaf import nearfield_leafpair_jax

    num_leaves = int(counts.shape[0])
    max_count = int(jnp.max(counts)) if num_leaves else 0
    slots = jnp.arange(max(max_count, 1), dtype=counts.dtype)
    valid = slots[None, :] < counts[:, None]
    edge_idx = jnp.where(valid, offsets[:-1, None] + slots[None, :], 0)
    ids = jnp.where(valid, neighbors[edge_idx], 0)
    return nearfield_leafpair_jax(
        leaf_positions,
        leaf_masses,
        leaf_mask,
        ids,
        valid,
        softening_sq=softening_sq,
        G=G,
        include_self=bool(include_self),
    )


# ------------------------------------------------------------------ reverse
# Analytic adjoint of the CSR near field (plan fast-gradients, Phase 2), the
# same tidal-tensor contraction as ``nearfield/grad.py`` (``k = -G (I/r^3 -
# 3 d d^T / r^5)``, symmetric in the pair), driven by the SAME CSR rows and
# chunk table as the forward, with the output cotangent gathered into the leaf
# layout like the masses. The near list is SYMMETRIC (a leaf's row lists every
# leaf that lists it -- the one-sided forward is only a correct force because
# it is), so a target-centric pass sees every pair from both ends: for target
# particle p and source q, with ``d = x_p - x_q``, ``c`` the acceleration
# cotangents,
#
#     x_p_bar += -G m_q (c_p / r^3 - 3 d (c_p . d) / r^5)     own term
#     x_p_bar +=  G m_p (c_q / r^3 - 3 d (c_q . d) / r^5)     p as a source of q
#     m_p_bar +=  G (c_q . d) / r^3
#     G_bar   += -m_q (c_p . d) / r^3
#     soft_bar += 3/2 G m_q (c_p . d) / r^5                    (softening_sq)
#
# so nothing is scattered by source leaf, no rectangle is materialised, and the
# per-chunk partials reduce onto the leaves exactly as the forward's do.

_REV_OUT_WIDTH = 8  # x, y, z, mass, G, softening_sq, pad, pad


def _nearfield_leafpair_csr_rev_kernel(
    table_pos_ref: KernelRef,
    table_mass_ref: KernelRef,
    table_mask_ref: KernelRef,
    table_cot_ref: KernelRef,
    leaf_count_ref: KernelRef,
    neighbors_ref: KernelRef,
    chunk_leaf_ref: KernelRef,
    chunk_start_ref: KernelRef,
    chunk_count_ref: KernelRef,
    chunk_first_ref: KernelRef,
    softening_sq_ref: KernelRef,
    g_ref: KernelRef,
    out_ref: KernelRef,
    *,
    leaf_width: int,
    subtile: int,
    accum_dtype: Any,
    out_dtype: Any,
    include_self: bool,
) -> None:
    """One CSR row chunk against one target subtile: the six cotangent lanes.

    Parameters
    ----------
    table_pos_ref : KernelRef
        Full padded position table ``(L, W, _POS_WIDTH)``.
    table_mass_ref : KernelRef
        Full mass table ``(L, W)``.
    table_mask_ref : KernelRef
        Full validity table ``(L, W)``.
    table_cot_ref : KernelRef
        Full acceleration-cotangent table ``(L, W, _POS_WIDTH)`` (lanes 0:3).
    leaf_count_ref : KernelRef
        One past the last live slot per leaf ``(L,)``.
    neighbors_ref : KernelRef
        Flat neighbour array of leaf indices ``(E,)``.
    chunk_leaf_ref : KernelRef
        Chunk table: target leaf ``(C,)``.
    chunk_start_ref : KernelRef
        Chunk table: first entry ``(C,)``.
    chunk_count_ref : KernelRef
        Chunk table: live entries ``(C,)``.
    chunk_first_ref : KernelRef
        Chunk table: first chunk of its leaf ``(C,)``.
    softening_sq_ref : KernelRef
        Squared softening ``(1,)``.
    g_ref : KernelRef
        Gravitational constant ``(1,)``.
    out_ref : KernelRef
        **Output** ``(1, Bt, _REV_OUT_WIDTH)``: position cotangent (0:3), mass
        cotangent (3), and this tile's partial ``G`` (4) and ``softening_sq``
        (5) cotangents, per target lane.
    leaf_width : int
        ``W``. Static.
    subtile : int
        ``Bt``. Static.
    accum_dtype : Any
        Wide accumulator dtype or ``None`` (as the forward).
    out_dtype : Any
        Output dtype.
    include_self : bool
        Add the intra-leaf term on the leaf's first chunk.

    Returns
    -------
    None
        Writes ``out_ref``.
    """
    c = pl.program_id(0)
    sub = pl.program_id(1)
    bt = int(subtile)
    tl = chunk_leaf_ref[c]
    live = tl >= 0
    tl_safe = jnp.maximum(tl, 0)
    off = sub * bt
    lanes = pl.ds(off, bt)
    tcount = leaf_count_ref[tl_safe]

    zeros_bt = jnp.zeros((bt,), out_dtype)
    for comp in range(_REV_OUT_WIDTH):
        out_ref[0, :, comp] = zeros_bt

    @pl.when(live & (off < tcount))
    def _work():
        tx = table_pos_ref[tl_safe, lanes, 0]
        ty = table_pos_ref[tl_safe, lanes, 1]
        tz = table_pos_ref[tl_safe, lanes, 2]
        tm = table_mass_ref[tl_safe, lanes]
        tvalid = table_mask_ref[tl_safe, lanes]
        cx = table_cot_ref[tl_safe, lanes, 0]
        cy = table_cot_ref[tl_safe, lanes, 1]
        cz = table_cot_ref[tl_safe, lanes, 2]
        soft = softening_sq_ref[0]
        g_value = g_ref[0]

        zero = jnp.zeros_like(tx)
        wide = accum_dtype is not None
        n_acc = 6
        acc0 = (
            tuple(jnp.zeros(tx.shape, accum_dtype) for _ in range(n_acc))
            if wide
            else tuple(zero for _ in range(n_acc))
        )

        def _leaf_pass(sid, acc, exclude_lane=None):
            def _lane_body(j, acc):
                xb, yb, zb, mb, gb, sb = acc
                sx = table_pos_ref[sid, j, 0]
                sy = table_pos_ref[sid, j, 1]
                sz = table_pos_ref[sid, j, 2]
                sm = table_mass_ref[sid, j]
                lane_valid = table_mask_ref[sid, j]
                qx = table_cot_ref[sid, j, 0]
                qy = table_cot_ref[sid, j, 1]
                qz = table_cot_ref[sid, j, 2]
                dx = tx - sx
                dy = ty - sy
                dz = tz - sz
                dist_sq = dx * dx + dy * dy + dz * dz + soft
                active = tvalid & lane_valid
                if exclude_lane is not None:
                    active = active & (exclude_lane != j)
                safe_dist_sq = jnp.where(active, dist_sq, 1.0)
                inv_r = lax.rsqrt(safe_dist_sq)
                inv_r = jnp.where(active, inv_r, 0.0)
                inv_r2 = inv_r * inv_r
                inv3 = inv_r2 * inv_r
                inv5 = inv3 * inv_r2
                cd = cx * dx + cy * dy + cz * dz
                qd = qx * dx + qy * dy + qz * dz
                a_own = -g_value * sm * inv3
                a_src = g_value * tm * inv3
                b_d = 3.0 * g_value * inv5 * (sm * cd - tm * qd)
                xb = xb + a_own * cx + a_src * qx + b_d * dx
                yb = yb + a_own * cy + a_src * qy + b_d * dy
                zb = zb + a_own * cz + a_src * qz + b_d * dz
                mb = mb + g_value * qd * inv3
                gb = gb - sm * cd * inv3
                sb = sb + 1.5 * g_value * sm * cd * inv5
                return (xb, yb, zb, mb, gb, sb)

            scount = leaf_count_ref[sid]
            if not wide:
                return lax.fori_loop(0, scount, _lane_body, acc)
            part = lax.fori_loop(
                0, scount, _lane_body, tuple(zero for _ in range(n_acc))
            )
            return tuple(a + q.astype(accum_dtype) for a, q in zip(acc, part))

        start = chunk_start_ref[c]
        cnt = chunk_count_ref[c]

        def _slot_body(s, acc):
            sid = neighbors_ref[start + s]
            return _leaf_pass(sid, acc)

        acc = lax.fori_loop(0, cnt, _slot_body, acc0)

        if include_self:
            lane_idx = off + lax.broadcasted_iota(jnp.int32, (bt,), 0)

            def _self_pass(acc):
                return _leaf_pass(tl_safe, acc, exclude_lane=lane_idx)

            acc = lax.cond(chunk_first_ref[c] != 0, _self_pass, lambda acc: acc, acc)

        zero_out = jnp.zeros((bt,), out_dtype)
        for comp, a in enumerate(acc):
            if wide and out_dtype != accum_dtype:
                a = a.astype(out_dtype)
            out_ref[0, :, comp] = jnp.where(tvalid, a, zero_out)


def nearfield_leafpair_csr_reverse_pallas(
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    neighbors: Array,
    chunks: LeafPairChunkTable,
    cotangent: Array,
    *,
    softening_sq: Array,
    G: Array,
    chunk: int,
    num_warps: int | None = None,
    num_stages: int = 1,
    target_subtile: int | None = None,
    interpret: bool = False,
    accum: str = "input",
    include_self: bool = True,
) -> tuple[Array, Array, Array, Array]:
    """Adjoint of :func:`nearfield_leafpair_csr_pallas` for the acceleration lanes.

    Parameters
    ----------
    leaf_positions : Array
        ``(L, W, 3)`` per-leaf positions.
    leaf_masses : Array
        ``(L, W)`` per-leaf masses.
    leaf_mask : Array
        ``(L, W)`` slot validity.
    neighbors : Array
        Flat neighbour array of leaf indices. Must be SYMMETRIC (see the module
        comment); the one-sided forward is only a correct force when it is.
    chunks : LeafPairChunkTable
        The forward's chunk table.
    cotangent : Array
        ``(L, W, 3)`` cotangent of the acceleration lanes, leaf layout. The
        potential lane has no reverse here (the differentiable lane is
        acceleration-only).
    softening_sq : Array
        Scalar squared softening.
    G : Array
        Scalar gravitational constant.
    chunk : int
        Entries per chunk. Static.
    num_warps : int | None
        Warps per program.
    num_stages : int
        Pipelining depth.
    target_subtile : int | None
        Targets per program.
    interpret : bool
        Pallas interpret mode.
    accum : str
        ``"input"`` or ``"wide"``, as the forward.
    include_self : bool
        Intra-leaf term on each leaf's first chunk.

    Returns
    -------
    tuple[Array, Array, Array, Array]
        ``(positions_bar (L, W, 3), masses_bar (L, W), softening_sq_bar (),
        G_bar ())`` in the input dtype.

    Raises
    ------
    RuntimeError
        If Pallas or its Triton backend could not be imported.
    """
    if pl is None or plgpu is None:
        raise RuntimeError("jax.experimental.pallas is not available")
    leaf_positions = jnp.asarray(leaf_positions)
    dtype = leaf_positions.dtype
    leaf_masses = jnp.asarray(leaf_masses, dtype=dtype)
    leaf_mask = jnp.asarray(leaf_mask, dtype=bool)
    cot = jnp.asarray(cotangent, dtype=dtype)
    num_leaves = int(leaf_positions.shape[0])
    leaf_width = int(leaf_positions.shape[1])
    _require_shape("leaf_masses", leaf_masses, (num_leaves, leaf_width))
    _require_shape("leaf_mask", leaf_mask, (num_leaves, leaf_width))
    _require_shape("cotangent", cot, (num_leaves, leaf_width, 3))
    capacity = int(chunks.leaf.shape[0])
    if num_leaves == 0 or leaf_width == 0:
        return (
            jnp.zeros_like(leaf_positions),
            jnp.zeros_like(leaf_masses),
            jnp.zeros((), dtype),
            jnp.zeros((), dtype),
        )
    idx = chunks.leaf.dtype
    neighbors = jnp.asarray(neighbors, dtype=idx)
    softening_sq_arr = jnp.asarray([softening_sq], dtype=dtype).reshape(1)
    g_arr = jnp.asarray([G], dtype=dtype).reshape(1)
    pos_padded = jnp.pad(leaf_positions, ((0, 0), (0, 0), (0, _POS_WIDTH - 3)))
    cot_padded = jnp.pad(cot, ((0, 0), (0, 0), (0, _POS_WIDTH - 3)))
    bt = _resolve_subtile(target_subtile, leaf_width)
    width_pad = ((leaf_width + bt - 1) // bt) * bt
    n_sub = width_pad // bt
    pad_t = width_pad - leaf_width
    if pad_t:
        pos_padded = jnp.pad(pos_padded, ((0, 0), (0, pad_t), (0, 0)))
        cot_padded = jnp.pad(cot_padded, ((0, 0), (0, pad_t), (0, 0)))
        leaf_masses = jnp.pad(leaf_masses, ((0, 0), (0, pad_t)))
        leaf_mask = jnp.pad(leaf_mask, ((0, 0), (0, pad_t)))
    if num_warps is None:
        num_warps = max(1, bt // 32)
    accum_dtype = _resolve_accum_dtype(accum, dtype)
    partial_dtype = accum_dtype if accum_dtype is not None else dtype
    include_self = bool(include_self)

    def _kernel(*refs):
        return _nearfield_leafpair_csr_rev_kernel(
            *refs,
            leaf_width=width_pad,
            subtile=bt,
            accum_dtype=accum_dtype,
            out_dtype=partial_dtype,
            include_self=include_self,
        )

    def _full(arr: Array) -> pl.BlockSpec:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    slot_no = jnp.arange(width_pad, dtype=jnp.int32)[None, :] + 1
    leaf_count = jnp.max(
        jnp.where(leaf_mask, slot_no, jnp.zeros_like(slot_no)), axis=1
    ).astype(jnp.int32)
    kernel = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(
            (capacity, width_pad, _REV_OUT_WIDTH), partial_dtype
        ),
        in_specs=[
            _full(pos_padded),
            _full(leaf_masses),
            _full(leaf_mask),
            _full(cot_padded),
            _full(leaf_count),
            _full(neighbors),
            _full(chunks.leaf),
            _full(chunks.start),
            _full(chunks.count),
            _full(chunks.is_first),
            pl.BlockSpec((1,), lambda c, sub: (0,)),
            pl.BlockSpec((1,), lambda c, sub: (0,)),
        ],
        out_specs=pl.BlockSpec((1, bt, _REV_OUT_WIDTH), lambda c, sub: (c, sub, 0)),
        grid=(capacity, n_sub),
        compiler_params=plgpu.CompilerParams(
            num_warps=int(num_warps), num_stages=int(num_stages)
        ),
        interpret=bool(interpret),
        name=(
            f"near_rev_leafpair_csr_t{bt}_c{int(chunk)}_w{leaf_width}_a{accum}"
            f"{'_self' if include_self else ''}"
        ),
    )
    partials = kernel(
        pos_padded,
        leaf_masses,
        leaf_mask,
        cot_padded,
        leaf_count,
        neighbors,
        chunks.leaf,
        chunks.start,
        chunks.count,
        chunks.is_first,
        softening_sq_arr,
        g_arr,
    )
    seg = jnp.where(chunks.leaf < 0, jnp.asarray(num_leaves, idx), chunks.leaf)
    out = jax.ops.segment_sum(
        partials, seg, num_segments=num_leaves + 1, indices_are_sorted=True
    )[:num_leaves]
    g_bar = jnp.sum(partials[:, :, 4]).astype(dtype)
    soft_bar = jnp.sum(partials[:, :, 5]).astype(dtype)
    out = out.astype(dtype)
    if pad_t:
        out = out[:, :leaf_width, :]
    return out[..., :3], out[..., 3], soft_bar, g_bar


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13))
def nearfield_leafpair_csr_pallas_cvjp(
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    neighbors: Array,
    chunks: LeafPairChunkTable,
    softening_sq: Array,
    G: Array,
    chunk: int,
    num_warps: int | None,
    num_stages: int,
    target_subtile: int | None,
    interpret: bool,
    accum: str,
    include_self: bool,
) -> Array:
    """Differentiable :func:`nearfield_leafpair_csr_pallas` (forward byte-identical).

    ACCELERATION ONLY: the reverse contracts the cotangent of lanes ``0:3`` and
    ignores the potential lane, so a caller must not differentiate through
    ``out[..., 3]`` (the differentiable near-field lane never returns it).

    Parameters
    ----------
    leaf_positions : Array
        ``(L, W, 3)``. Differentiable.
    leaf_masses : Array
        ``(L, W)``. Differentiable.
    leaf_mask : Array
        ``(L, W)`` bool (topology).
    neighbors : Array
        Flat SYMMETRIC neighbour array (topology).
    chunks : LeafPairChunkTable
        Chunk table (topology).
    softening_sq : Array
        Scalar. Differentiable.
    G : Array
        Scalar. Differentiable.
    chunk : int
        ``nondiff_argnums``.
    num_warps : int | None
        ``nondiff_argnums``.
    num_stages : int
        ``nondiff_argnums``.
    target_subtile : int | None
        ``nondiff_argnums``.
    interpret : bool
        ``nondiff_argnums``.
    accum : str
        ``nondiff_argnums``.
    include_self : bool
        ``nondiff_argnums``.

    Returns
    -------
    Array
        ``(L, W, 4)`` acceleration lanes 0:3 and potential lane 3.
    """
    return nearfield_leafpair_csr_pallas(
        leaf_positions,
        leaf_masses,
        leaf_mask,
        neighbors,
        chunks,
        softening_sq=softening_sq,
        G=G,
        chunk=chunk,
        num_warps=num_warps,
        num_stages=num_stages,
        target_subtile=target_subtile,
        interpret=interpret,
        accum=accum,
        include_self=include_self,
    )


def _near_csr_cvjp_fwd(
    leaf_positions,
    leaf_masses,
    leaf_mask,
    neighbors,
    chunks,
    softening_sq,
    G,
    chunk,
    num_warps,
    num_stages,
    target_subtile,
    interpret,
    accum,
    include_self,
):
    out = nearfield_leafpair_csr_pallas(
        leaf_positions,
        leaf_masses,
        leaf_mask,
        neighbors,
        chunks,
        softening_sq=softening_sq,
        G=G,
        chunk=chunk,
        num_warps=num_warps,
        num_stages=num_stages,
        target_subtile=target_subtile,
        interpret=interpret,
        accum=accum,
        include_self=include_self,
    )
    return out, (
        leaf_positions,
        leaf_masses,
        leaf_mask,
        neighbors,
        chunks,
        jnp.asarray(softening_sq),
        jnp.asarray(G),
    )


def _near_csr_cvjp_bwd(
    chunk,
    num_warps,
    num_stages,
    target_subtile,
    interpret,
    accum,
    include_self,
    residual,
    cotangent,
):
    leaf_positions, leaf_masses, leaf_mask, neighbors, chunks, softening_sq, G = (
        residual
    )
    pos_bar, mass_bar, soft_bar, g_bar = nearfield_leafpair_csr_reverse_pallas(
        leaf_positions,
        leaf_masses,
        leaf_mask,
        neighbors,
        chunks,
        cotangent[..., :3],
        softening_sq=softening_sq,
        G=G,
        chunk=chunk,
        num_warps=num_warps,
        num_stages=num_stages,
        target_subtile=target_subtile,
        interpret=interpret,
        accum=accum,
        include_self=include_self,
    )
    return (
        pos_bar.astype(leaf_positions.dtype),
        mass_bar.astype(leaf_masses.dtype),
        None,
        None,
        None,
        jnp.reshape(soft_bar.astype(softening_sq.dtype), jnp.shape(softening_sq)),
        jnp.reshape(g_bar.astype(G.dtype), jnp.shape(G)),
    )


nearfield_leafpair_csr_pallas_cvjp.defvjp(_near_csr_cvjp_fwd, _near_csr_cvjp_bwd)
