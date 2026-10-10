"""COM MAC radii: each leaf's particles read once for a chunk of its ancestor chain.

:func:`jaccpot.runtime._mac_geometry.com_mac_geometry` needs, for every node, the
largest distance from its expansion centre to any of its particles. A node's
particles are the union of its descendant leaves', so the exact radius is the max
over (leaf, ancestor) pairs of the leaf's farthest particle from the ancestor's
centre. The XLA form ran one ``(L, w)`` reduction per ancestor level -- every
level re-gathering every leaf's padded positions: ~19 of the stage's ~26 ms per
step at 8e6 (40 levels x 56M padded slots).

Here one program owns ``B`` leaves and a chunk of ``levels`` of their ancestors,
gathered by the caller into a ``(L, levels)`` table (one pointer step per level
over all leaves): it reads the leaves' particles a few lanes at a time and writes
per (leaf, level) the max SQUARED distance. The caller reduces that table by runs
of equal ancestor (a node's leaves are consecutive) and scatters the run ends,
then takes one square root per node (sqrt is monotone, so this is the max of the
distances).

The first form of the kernel walked each leaf's ancestor chain itself, over the
leaf capacity's lanes (``JACCPOT_COM_RADII_VARIANT=chain``); the table form gives
the same radii to the bit, 11.4 -> 4.8 ms at 8e6 and 78 -> 36.5 ms at 1e8 (A100,
2026-10-05), and the chain form was removed in the 2026-10 cleanup (X5).
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from jaccpot.pallas._compat import KernelRef, pallas_backend_kwargs

try:
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None

__all__ = ["com_radii_table_pallas"]


def _next_pow2(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def _com_radii_table_kernel(
    pos_ref: KernelRef,
    cent_ref: KernelRef,
    start_ref: KernelRef,
    count_ref: KernelRef,
    anc_ref: KernelRef,
    d2_out: KernelRef,
    *,
    lanes: int,
    levels: int,
) -> None:
    """``B`` leaves against a given ``(B, levels)`` table of their ancestors.

    The removed chain kernel (cleanup 2026-10, X5) walked the chain itself: each
    level's centre waited on the previous level's parent load, so a program was
    ``2 x levels`` dependent global loads deep. Here the caller hands the chain in
    (one parallel gather per level over all leaves), every level's centre is
    loaded up front, and the leaf's particles are read ``lanes`` at a time up to
    the block's largest leaf, not over the leaf capacity. Same differences, same
    squares and sums, and max is exact in any order: the output was the chain
    kernel's, bit for bit.

    Parameters
    ----------
    pos_ref : KernelRef
        Whole sorted positions ``[N, 3]``.
    cent_ref : KernelRef
        Whole expansion centres ``[nodes, 3]``.
    start_ref : KernelRef
        This block's leaf starts ``[B]``.
    count_ref : KernelRef
        This block's leaf counts ``[B]`` (0 for padding leaves).
    anc_ref : KernelRef
        This block's ancestors ``[B, levels]`` (``-1``: none).
    d2_out : KernelRef
        ``[B, levels]`` max squared distance per (leaf, level); 0 where none.
    lanes : int
        Particles per leaf per iteration (a power of two). Static.
    levels : int
        Ancestors per call. Static.

    Returns
    -------
    None
        Writes the output block.
    """
    start = start_ref[...]
    count = count_ref[...]
    lane = lax.broadcasted_iota(start.dtype, (start.shape[0], lanes), 1)
    cents = []
    lives = []
    for j in range(levels):
        anc = anc_ref[:, j]
        live = anc >= 0
        a = jnp.where(live, anc, 0)
        cents.append(
            (cent_ref[a, 0][:, None], cent_ref[a, 1][:, None], cent_ref[a, 2][:, None])
        )
        lives.append(live)

    def body(t: Array, acc: tuple) -> tuple:
        off = t * lanes + lane
        valid = off < count[:, None]
        idx = jnp.where(valid, start[:, None] + off, 0)
        px = pos_ref[idx, 0]
        py = pos_ref[idx, 1]
        pz = pos_ref[idx, 2]
        zero = jnp.zeros_like(px)
        new = []
        for j in range(levels):
            cx, cy, cz = cents[j]
            dx = px - cx
            dy = py - cy
            dz = pz - cz
            d2 = dx * dx + dy * dy + dz * dz
            new.append(jnp.maximum(acc[j], jnp.max(jnp.where(valid, d2, zero), axis=1)))
        return tuple(new)

    # a block whose ancestors are all past the root (the deep chunks of shallow
    # leaves) reads no particles: its row is zero either way
    any_live = functools.reduce(
        jnp.maximum, [jnp.max(lv.astype(jnp.int32)) for lv in lives]
    )
    trips = jnp.where(any_live > 0, (jnp.max(count) + (lanes - 1)) // lanes, 0)
    zero_row = jnp.zeros(start.shape, pos_ref.dtype)
    acc = lax.fori_loop(0, trips, body, tuple(zero_row for _ in range(levels)))
    for j in range(levels):
        d2_out[:, j] = jnp.where(lives[j], acc[j], jnp.zeros_like(acc[j]))


def com_radii_table_pallas(
    positions_sorted: Array,
    centers: Array,
    leaf_start: Array,
    leaf_count: Array,
    anc_table: Array,
    *,
    lanes: int = 16,
    block: int = 16,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 4,
) -> Array:
    """Max squared distance per (leaf, ancestor) for a given ancestor table ``[L, levels]``.

    Parameters
    ----------
    positions_sorted : Array
        ``[N, 3]`` positions in tree order.
    centers : Array
        ``[nodes, 3]`` expansion centres (the positions' dtype).
    leaf_start : Array
        ``[L]`` first particle per leaf.
    leaf_count : Array
        ``[L]`` particles per leaf (0 for padding leaves).
    anc_table : Array
        ``[L, levels]`` the nodes to measure each leaf against (``-1``: none).
    lanes : int
        Particles per leaf per iteration (a power of two). Static.
    block : int
        Leaves per program (a power of two). Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.

    Returns
    -------
    Array
        ``[L, levels]`` max squared distances (0 where the ancestor is ``-1``).

    Raises
    ------
    RuntimeError
        If Pallas is not available.
    """
    if pl is None:
        raise RuntimeError("jax.experimental.pallas is not available")
    L, levels = (int(d) for d in anc_table.shape)
    B = _next_pow2(block)
    Lp = -(-L // B) * B
    pad = Lp - L
    idx = jnp.asarray(anc_table).dtype
    start = jnp.pad(jnp.asarray(leaf_start, idx), (0, pad))
    count = jnp.pad(jnp.asarray(leaf_count, idx), (0, pad))
    anc = jnp.pad(jnp.asarray(anc_table, idx), ((0, pad), (0, 0)), constant_values=-1)
    pos = jnp.asarray(positions_sorted)
    cent = jnp.asarray(centers, pos.dtype)
    kernel = functools.partial(
        _com_radii_table_kernel, lanes=_next_pow2(lanes), levels=levels
    )
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )

    def _full(arr: Array) -> Any:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    d2 = pl.pallas_call(
        kernel,
        grid=(Lp // B,),
        in_specs=[
            _full(pos),
            _full(cent),
            pl.BlockSpec((B,), lambda i: (i,)),
            pl.BlockSpec((B,), lambda i: (i,)),
            pl.BlockSpec((B, levels), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((B, levels), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((Lp, levels), pos.dtype),
        interpret=bool(interpret),
        name=f"com_radii_table_l{levels}_b{B}_k{_next_pow2(lanes)}",
        **backend_kwargs,
    )(pos, cent, start, count, anc)
    return d2[:L]
