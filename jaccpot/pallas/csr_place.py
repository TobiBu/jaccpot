"""The directed CSR of canonical pairs without a sort over the pair list.

``_directed_csr_from_canonical`` turns the walk's canonical pairs ``(a, b)``,
``a < b``, into the directed list sorted by (target, source): row ``t`` holds the
sources of every pair touching ``t``, ascending. With XLA that is two sorts over
the ``W`` pairs (a composite int64 sort, then a stable key-value re-sort), and at
their placement the step held both sorted copies plus the output: 24 B per
canonical slot, the fused step's peak.

Here, in three passes and no sort:

1. **Counts** per row (an integer scatter-add, exact) and their prefix sum, the
   row offsets.
2. **Placement** (:func:`_place_kernel`): each live pair claims one slot in each
   of its two rows from a per-row atomic cursor (float32: see the kernel) and
   writes the other end there.
   The rows are complete; their order is the atomics' (not deterministic).
3. **Rank** (:func:`_rank_kernel`): one program per row ranks every entry among
   the row's entries (tile against tile, ``K x K`` compares) and writes it to
   ``offset + rank``. The entries of a row are distinct node ids, so the rank is a
   permutation and the result is the row sorted ascending -- the same array, bit
   for bit, as the sorted build, whatever order the atomics left.

The live set is the pairs plus one unsorted and one sorted ``2W`` list: 16 B per
canonical slot. At the 8e6 far list's size (2.9e7 slots, 1.96e7 live, rows of up to
~50) on an A100 the build takes 10.3-10.7 ms against the two sorts' 10.6-11.3 ms
(placement ~70 % of it: acquire-release float atomics), at half their temporaries.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from jaccpot.pallas._compat import KernelRef, pallas_backend_kwargs
from jaccpot.pallas.m2l_real_csr import pallas_m2l_real_csr_supported

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None
    plgpu = None

__all__ = ["directed_csr_pallas", "pallas_directed_csr_supported"]


def pallas_directed_csr_supported() -> bool:
    """True where the Triton lowering runs (sm_80+, as the other real kernels).

    Returns
    -------
    bool
        Whether the kernels can run natively here.
    """
    return pallas_m2l_real_csr_supported()


def _place_kernel(
    a_ref: KernelRef,
    b_ref: KernelRef,
    off_ref: KernelRef,
    count_ref: KernelRef,
    cursor_in: KernelRef,
    src_in: KernelRef,
    cursor_ref: KernelRef,
    src_ref: KernelRef,
    *,
    block: int,
    row_offset: int,
    width: int,
    vector: bool,
) -> None:
    """``block`` canonical pairs: one slot in each of their two rows.

    The cursors are FLOAT32: on the installed jax (0.10.2, probed on an A100) the
    vector form of ``plgpu.atomic_add`` on int32 neither updates memory nor
    returns the old values, while float32 does both, and a cursor counts a row's
    entries exactly far below 2**24. Interpret mode can only emulate scalar
    atomics (``vector=False``: the pairs one after another).

    Parameters
    ----------
    a_ref : KernelRef
        Whole ``[W]`` lower node of each canonical pair (read as is: the tail
        program's lanes past ``count`` are masked, never loaded out of range).
    b_ref : KernelRef
        Whole ``[W]`` upper node.
    off_ref : KernelRef
        Whole ``[R + 1]`` row offsets.
    count_ref : KernelRef
        ``[1]`` live canonical pairs (a prefix).
    cursor_in : KernelRef
        Zeroed float32 ``[R]`` cursors, aliased to ``cursor_ref`` (not read).
    src_in : KernelRef
        ``[2W]`` list, aliased to ``src_ref`` (not read).
    cursor_ref : KernelRef
        **Output** per-row cursors; the atomics target it.
    src_ref : KernelRef
        **Output** the unsorted directed list.
    block : int
        Pairs per program. Static.
    row_offset : int
        Node id of row 0. Static.
    width : int
        ``W``, so ``2W`` is the list length. Static.
    vector : bool
        Masked vector atomics (native) or a scalar loop (interpret). Static.

    Returns
    -------
    None
        Fills the slots of the block's pairs.
    """
    del cursor_in, src_in
    i0 = pl.program_id(0) * block
    count = count_ref[0]
    if not vector:
        one = jnp.ones((), cursor_ref.dtype)
        for k in range(block):

            @pl.when(i0 + k < count)
            def _(k: int = k) -> None:
                na = a_ref[i0 + k]
                nb = b_ref[i0 + k]
                ra = na - row_offset
                rb = nb - row_offset
                old_hi = plgpu.atomic_add(cursor_ref, (ra,), one)
                old_lo = plgpu.atomic_add(cursor_ref, (rb,), one)
                hi = off_ref[ra] + old_hi.astype(off_ref.dtype)
                lo = off_ref[rb] + old_lo.astype(off_ref.dtype)
                src_ref[hi] = nb.astype(src_ref.dtype)
                src_ref[lo] = na.astype(src_ref.dtype)

        return
    lane = lax.broadcasted_iota(jnp.int32, (block,), 0)
    live = (i0 + lane) < count  # count <= W: the live lanes are in range
    i = jnp.where(live, i0 + lane, 0)
    na = a_ref[i]
    nb = b_ref[i]
    ra = jnp.where(live, na - row_offset, 0)
    rb = jnp.where(live, nb - row_offset, 0)
    one = jnp.ones((block,), cursor_ref.dtype)
    old_hi = plgpu.atomic_add(cursor_ref, (ra,), one, mask=live)
    old_lo = plgpu.atomic_add(cursor_ref, (rb,), one, mask=live)
    past = 2 * width + lane  # dead lanes point past the list (never written)
    hi = jnp.where(live, off_ref[ra] + old_hi.astype(off_ref.dtype), past)
    lo = jnp.where(live, off_ref[rb] + old_lo.astype(off_ref.dtype), past)
    plgpu.store(src_ref.at[hi], nb.astype(src_ref.dtype), mask=live)
    plgpu.store(src_ref.at[lo], na.astype(src_ref.dtype), mask=live)


def _rank_kernel(
    src_ref: KernelRef,
    off_ref: KernelRef,
    out_in: KernelRef,
    out_ref: KernelRef,
    *,
    lanes: int,
    width: int,
    rows: int,
    num_rows: int,
) -> None:
    """``rows`` rows: every entry to ``offset + its rank among the row's entries``.

    Parameters
    ----------
    src_ref : KernelRef
        Whole ``[2W]`` unsorted directed list.
    off_ref : KernelRef
        Whole ``[R + 1]`` row offsets.
    out_in : KernelRef
        The padded ``[2W]`` output, aliased to ``out_ref`` (not read).
    out_ref : KernelRef
        **Output** the sorted list.
    lanes : int
        Tile width ``K`` (a power of two; one warp's worth). Static.
    width : int
        ``W``. Static.
    rows : int
        Rows per program (most rows are a tile or two: one per program left the
        programs idle). Static.
    num_rows : int
        ``R``. Static.

    Returns
    -------
    None
        Writes the rows' entries in ascending order.
    """
    del out_in
    r0 = pl.program_id(0) * rows
    lane = lax.broadcasted_iota(jnp.int32, (lanes,), 0)

    def one_row(j: Array, carry: Array) -> Array:
        r = jnp.minimum(r0 + j, num_rows - 1)
        start = off_ref[r]
        n = jnp.where(r0 + j < num_rows, off_ref[r + 1] - start, 0)
        n_tiles = (n + (lanes - 1)) // lanes

        def tile(t: Array, c: Array) -> Array:
            pos = t * lanes + lane
            ok = pos < n
            v = src_ref[jnp.where(ok, start + pos, start)]

            def against(u: Array, rank: Array) -> Array:
                pos_u = u * lanes + lane
                ok_u = pos_u < n
                w = src_ref[jnp.where(ok_u, start + pos_u, start)]
                # ties (a repeated pair) by position: the ranks stay a permutation
                before = (w[None, :] < v[:, None]) | (
                    (w[None, :] == v[:, None]) & (pos_u[None, :] < pos[:, None])
                )
                return rank + jnp.sum(before & ok_u[None, :], axis=1, dtype=jnp.int32)

            rank = lax.fori_loop(0, n_tiles, against, jnp.zeros((lanes,), jnp.int32))
            dest = jnp.where(ok, start + rank, 2 * width + lane)
            plgpu.store(out_ref.at[dest], v, mask=ok)
            return c

        return lax.fori_loop(0, n_tiles, tile, carry)

    lax.fori_loop(0, rows, one_row, jnp.int32(0))


def directed_csr_pallas(
    nodes_a: Array,
    nodes_b: Array,
    count: Array,
    *,
    num_rows: int,
    row_offset: int,
    pad_source: int,
    idx: Any,
    block: int = 256,
    lanes: int = 32,
    rows_per_program: int = 64,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 4,
) -> tuple[Array, Array, Array]:
    """``(sources [2W], offsets [R + 1], counts [R])`` of the canonical pairs.

    Parameters
    ----------
    nodes_a : Array
        ``[W]`` lower node of each canonical pair (row ``a - row_offset``); only
        the live prefix is read.
    nodes_b : Array
        ``[W]`` upper node.
    count : Array
        Live canonical pairs (clipped to ``W``).
    num_rows : int
        Rows ``R``. Static.
    row_offset : int
        Node id of row 0 (added back to the sources). Static.
    pad_source : int
        Source value past the live entries. Static.
    idx : Any
        Index dtype. Static.
    block : int
        Pairs per placement program (8 in interpret mode, a scalar loop). Static.
    lanes : int
        Rank tile width. Static.
    rows_per_program : int
        Rows per rank program. Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per placement program (the rank programs run one warp).

    Returns
    -------
    tuple[Array, Array, Array]
        The directed list sorted by (target, source), its row offsets and
        counts, as :func:`jaccpot.runtime._interaction_cache._directed_csr_from_canonical`.

    Raises
    ------
    RuntimeError
        If Pallas is not available.
    """
    if pl is None:
        raise RuntimeError("jax.experimental.pallas is not available")
    W = int(nodes_a.shape[0])
    R = int(num_rows)
    B = 8 if interpret else int(block)
    a = jnp.asarray(nodes_a, idx)
    b = jnp.asarray(nodes_b, idx)
    n_live = jnp.minimum(jnp.asarray(count, idx), jnp.asarray(W, idx))
    live = jnp.arange(W, dtype=idx) < n_live
    ro = jnp.asarray(int(row_offset), idx)
    # counts: exact integer scatter-adds, the row computed inside the scatter
    # (no (W,) row arrays beside the pairs); the dead slots (row R) are dropped
    counts = (
        jnp.zeros((R,), idx)
        .at[jnp.where(live, a - ro, R)]
        .add(jnp.ones((W,), idx), mode="drop")
        .at[jnp.where(live, b - ro, R)]
        .add(jnp.ones((W,), idx), mode="drop")
    )
    offsets = jnp.concatenate([jnp.zeros((1,), idx), jnp.cumsum(counts).astype(idx)])
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )

    def _full(arr: Array) -> Any:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    count_arr = jnp.reshape(n_live, (1,))
    cursor0 = jnp.zeros((R,), jnp.float32)
    unsorted0 = jnp.zeros((2 * W,), idx)
    place = functools.partial(
        _place_kernel,
        block=B,
        row_offset=int(row_offset),
        width=W,
        vector=not bool(interpret),
    )
    _, unsorted = pl.pallas_call(
        place,
        grid=(-(-W // B),),
        in_specs=[_full(a), _full(b), _full(offsets), _full(count_arr)]
        + [_full(cursor0), _full(unsorted0)],
        out_specs=[_full(cursor0), _full(unsorted0)],
        out_shape=[
            jax.ShapeDtypeStruct(cursor0.shape, cursor0.dtype),
            jax.ShapeDtypeStruct(unsorted0.shape, unsorted0.dtype),
        ],
        input_output_aliases={4: 0, 5: 1},
        interpret=bool(interpret),
        name="directed_csr_place",
        **backend_kwargs,
    )(a, b, offsets, count_arr, cursor0, unsorted0)
    out0 = jnp.full((2 * W,), int(pad_source), idx)
    P = max(1, int(rows_per_program))
    rank = functools.partial(
        _rank_kernel, lanes=int(lanes), width=W, rows=P, num_rows=R
    )
    rank_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in rank_kwargs:
        rank_kwargs["compiler_params"] = type(rank_kwargs["compiler_params"])(
            num_warps=1
        )
    sources = pl.pallas_call(
        rank,
        grid=(-(-R // P),),
        in_specs=[_full(unsorted), _full(offsets), _full(out0)],
        out_specs=_full(out0),
        out_shape=jax.ShapeDtypeStruct(out0.shape, out0.dtype),
        input_output_aliases={2: 0},
        interpret=bool(interpret),
        name=f"directed_csr_rank_k{int(lanes)}",
        **rank_kwargs,
    )(unsorted, offsets, out0)
    return sources, offsets, counts
