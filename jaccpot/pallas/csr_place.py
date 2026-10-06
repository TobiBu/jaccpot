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


#: Rows longer than this are ranked with their tiles spread over programs
#: (:func:`_rank_long_kernel`). The standard configurations' rows are far below it
#: (near lists: 85-634 at cell_min_level 8; far rows a few hundred), so they keep the
#: one-program-per-row kernel.
RANK_LONG_ROW = 2048


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
    row_lo: int = 0,
    row_hi: int = 2**31 - 1,
) -> None:
    """``block`` canonical pairs: one slot in each of their two rows in ``[row_lo, row_hi)``.

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
    row_lo : int
        First row this pass places (rows outside ``[row_lo, row_hi)`` are left to
        another pass over the pairs). Static.
    row_hi : int
        One past the last row this pass places. Static.

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

                @pl.when((ra >= row_lo) & (ra < row_hi))
                def _() -> None:
                    old_hi = plgpu.atomic_add(cursor_ref, (ra,), one)
                    hi = off_ref[ra] + old_hi.astype(off_ref.dtype)
                    src_ref[hi] = nb.astype(src_ref.dtype)

                @pl.when((rb >= row_lo) & (rb < row_hi))
                def _() -> None:
                    old_lo = plgpu.atomic_add(cursor_ref, (rb,), one)
                    lo = off_ref[rb] + old_lo.astype(off_ref.dtype)
                    src_ref[lo] = na.astype(src_ref.dtype)

        return
    lane = lax.broadcasted_iota(jnp.int32, (block,), 0)
    live = (i0 + lane) < count  # count <= W: the live lanes are in range
    i = jnp.where(live, i0 + lane, 0)
    na = a_ref[i]
    nb = b_ref[i]
    ra = jnp.where(live, na - row_offset, 0)
    rb = jnp.where(live, nb - row_offset, 0)
    in_a = live & (ra >= row_lo) & (ra < row_hi)
    in_b = live & (rb >= row_lo) & (rb < row_hi)
    ra = jnp.where(in_a, ra, 0)
    rb = jnp.where(in_b, rb, 0)
    one = jnp.ones((block,), cursor_ref.dtype)
    old_hi = plgpu.atomic_add(cursor_ref, (ra,), one, mask=in_a)
    old_lo = plgpu.atomic_add(cursor_ref, (rb,), one, mask=in_b)
    past = 2 * width + lane  # dead lanes point past the list (never written)
    hi = jnp.where(in_a, off_ref[ra] + old_hi.astype(off_ref.dtype), past)
    lo = jnp.where(in_b, off_ref[rb] + old_lo.astype(off_ref.dtype), past)
    plgpu.store(src_ref.at[hi], nb.astype(src_ref.dtype), mask=in_a)
    plgpu.store(src_ref.at[lo], na.astype(src_ref.dtype), mask=in_b)


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
    long_row: int = 2**31 - 1,
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
    long_row : int
        Rows longer than this are left to :func:`_rank_long_kernel`. Static.

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
        # rows past ``long_row`` are the long-row kernel's (:func:`_rank_long_kernel`)
        n = jnp.where(n <= long_row, n, 0)
        n_tiles = (n + (lanes - 1)) // lanes

        def tile(t: Array, c: Array) -> Array:
            _rank_tile(src_ref, out_ref, start, n, n_tiles, t, lane, width)
            return c

        return lax.fori_loop(0, n_tiles, tile, carry)

    lax.fori_loop(0, rows, one_row, jnp.int32(0))


def _rank_tile(
    src_ref: KernelRef,
    out_ref: KernelRef,
    start: Array,
    n: Array,
    n_tiles: Array,
    t: Array,
    lane: Array,
    width: int,
) -> None:
    """Rank tile ``t`` of the row at ``start`` (``n`` entries) and store it.

    Parameters
    ----------
    src_ref : KernelRef
        Whole ``[2W]`` unsorted directed list.
    out_ref : KernelRef
        **Output** the sorted list.
    start : Array
        The row's first entry.
    n : Array
        The row's entry count.
    n_tiles : Array
        ``ceil(n / lanes)``.
    t : Array
        The tile to rank.
    lane : Array
        ``arange(lanes)``.
    width : int
        ``W``. Static.
    """
    lanes = lane.shape[0]
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


def _rank_long_kernel(
    src_ref: KernelRef,
    off_ref: KernelRef,
    long_ref: KernelRef,
    out_in: KernelRef,
    out_ref: KernelRef,
    *,
    lanes: int,
    width: int,
    split: int,
) -> None:
    """Program ``(i, s)``: tiles ``s, s + split, ...`` of long row ``long_ref[i]``.

    A row's rank costs ``(n / lanes)^2`` tile pairs, and :func:`_rank_kernel` runs a
    whole row in one program: a near row of 101,870 entries (an outskirt cell at
    cell_min_level 6) took 19.9 s per list build there. Here the row's tiles are
    spread over ``split`` programs; each still ranks against the whole row, so the
    result is the same permutation.

    Parameters
    ----------
    src_ref : KernelRef
        Whole ``[2W]`` unsorted directed list.
    off_ref : KernelRef
        Whole ``[R + 1]`` row offsets.
    long_ref : KernelRef
        ``[B]`` this batch's long rows (``-1``: none).
    out_in : KernelRef
        The output so far, aliased to ``out_ref`` (not read).
    out_ref : KernelRef
        **Output** the sorted list.
    lanes : int
        Tile width. Static.
    width : int
        ``W``. Static.
    split : int
        Programs per long row. Static.
    """
    del out_in
    r = long_ref[pl.program_id(0)]
    s = pl.program_id(1)
    live = r >= 0
    rs = jnp.maximum(r, 0)
    start = off_ref[rs]
    n = jnp.where(live, off_ref[rs + 1] - start, 0)
    n_tiles = (n + (lanes - 1)) // lanes
    lane = lax.broadcasted_iota(jnp.int32, (lanes,), 0)
    trips = jnp.maximum(n_tiles - s + (split - 1), 0) // split

    def tile(k: Array, c: Array) -> Array:
        _rank_tile(src_ref, out_ref, start, n, n_tiles, s + k * split, lane, width)
        return c

    lax.fori_loop(0, trips, tile, jnp.int32(0))


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
    slices: int = 1,
    long_row: int | None = None,
    long_batch: int = 64,
    long_split: int = 256,
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
    slices : int
        Placement passes over the pairs, each placing the entries of one
        contiguous range of rows, so that its cursors and slots stay in cache.
        The rank pass orders every row afterwards, so the result does not
        depend on it. Static.
    long_row : int | None
        Rows longer than this are ranked by :func:`_rank_long_kernel`, their
        tiles spread over ``long_split`` programs, ``long_batch`` rows a launch,
        under a ``lax.cond`` that skips it when no row is that long. ``None``:
        :data:`RANK_LONG_ROW`. Static.
    long_batch : int
        Long rows per launch. Static.
    long_split : int
        Programs per long row. Static.

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
    # never read (the rank reads the placed slots only); its fill differs from the
    # output's so XLA does not merge the two fills into one buffer it then copies
    # into both aliased operands (near lists pad with 0: 2 x 680 MB at 1e8)
    unsorted0 = jnp.full((2 * W,), int(pad_source) + 1, idx)
    S = max(1, min(int(slices), R))
    bounds = [(R * k) // S for k in range(S + 1)]
    cursor, unsorted = cursor0, unsorted0
    for k in range(S):
        place = functools.partial(
            _place_kernel,
            block=B,
            row_offset=int(row_offset),
            width=W,
            vector=not bool(interpret),
            row_lo=int(bounds[k]),
            row_hi=int(bounds[k + 1]) if k + 1 < S else 2**31 - 1,
        )
        cursor, unsorted = pl.pallas_call(
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
            name="directed_csr_place" if S == 1 else f"directed_csr_place_s{k}",
            **backend_kwargs,
        )(a, b, offsets, count_arr, cursor, unsorted)
    out0 = jnp.full((2 * W,), int(pad_source), idx)
    P = max(1, int(rows_per_program))
    L = RANK_LONG_ROW if long_row is None else max(int(lanes), int(long_row))
    rank = functools.partial(
        _rank_kernel, lanes=int(lanes), width=W, rows=P, num_rows=R, long_row=L
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
    # rows past L: their tiles spread over programs (a whole row in one program costs
    # (n / lanes)^2 tile pairs serially -- 19.9 s for one row of 101,870 entries)
    max_long = (2 * W) // (L + 1)  # rows longer than L, at most
    if max_long > 0:
        Bb, S = max(1, int(long_batch)), max(1, int(long_split))
        is_long = counts > jnp.asarray(L, idx)
        n_long = jnp.sum(is_long, dtype=idx)
        n_ids = -(-max_long // Bb) * Bb
        long_call = pl.pallas_call(
            functools.partial(_rank_long_kernel, lanes=int(lanes), width=W, split=S),
            grid=(Bb, S),
            in_specs=[_full(unsorted), _full(offsets), _full(jnp.zeros((Bb,), idx))]
            + [_full(out0)],
            out_specs=_full(out0),
            out_shape=jax.ShapeDtypeStruct(out0.shape, out0.dtype),
            input_output_aliases={3: 0},
            interpret=bool(interpret),
            name=f"directed_csr_rank_long_k{int(lanes)}",
            **rank_kwargs,
        )

        def _long_rows(out: Array) -> Array:
            # inside the branch: the row scan runs only when a long row exists
            long_ids = jnp.nonzero(is_long, size=n_ids, fill_value=-1)[0].astype(idx)

            def _batch(b: Array, out_b: Array) -> Array:
                ids = lax.dynamic_slice(long_ids, (b * Bb,), (Bb,))
                return long_call(unsorted, offsets, ids, out_b)

            return lax.fori_loop(0, (n_long + (Bb - 1)) // Bb, _batch, out)

        sources = lax.cond(n_long > 0, _long_rows, lambda out: out, sources)
    return sources, offsets, counts
