"""COM MAC radii: each leaf's particles read once for a chunk of its ancestor chain.

:func:`jaccpot.runtime._mac_geometry.com_mac_geometry` needs, for every node, the
largest distance from its expansion centre to any of its particles. A node's
particles are the union of its descendant leaves', so the exact radius is the max
over (leaf, ancestor) pairs of the leaf's farthest particle from the ancestor's
centre. The XLA form ran one ``(L, w)`` reduction per ancestor level -- every
level re-gathering every leaf's padded positions: ~19 of the stage's ~26 ms per
step at 8e6 (40 levels x 56M padded slots).

Here one program owns ``B`` leaves: it loads their particles once (``(B, Wp)``
lanes, masked by the leaf count), then walks ``levels`` ancestors by ``parent``,
writing per (leaf, level) the max SQUARED distance and the ancestor id. The
caller reduces the ``(L, levels)`` table by runs of equal ancestor (a node's
leaves are consecutive) and scatters the run ends, as before, then takes one
square root per node (sqrt is monotone, so this is the max of the distances).
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

__all__ = ["com_radii_chunk_pallas"]


def _next_pow2(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def _com_radii_chunk_kernel(
    pos_ref: KernelRef,
    cent_ref: KernelRef,
    parent_ref: KernelRef,
    start_ref: KernelRef,
    count_ref: KernelRef,
    anc_ref: KernelRef,
    d2_out: KernelRef,
    anc_out: KernelRef,
    next_out: KernelRef,
    *,
    width: int,
    levels: int,
) -> None:
    """``B`` leaves: their particles once, ``levels`` ancestors walked by ``parent``.

    Parameters
    ----------
    pos_ref : KernelRef
        Whole sorted positions ``[N, 3]``.
    cent_ref : KernelRef
        Whole expansion centres ``[nodes, 3]``.
    parent_ref : KernelRef
        Whole parent table ``[nodes]`` (``-1`` at the root).
    start_ref : KernelRef
        This block's leaf starts ``[B]``.
    count_ref : KernelRef
        This block's leaf counts ``[B]`` (0 for padding leaves).
    anc_ref : KernelRef
        This block's first ancestor of the chunk ``[B]`` (``-1``: none).
    d2_out : KernelRef
        ``[B, levels]`` max squared distance per (leaf, level); 0 where none.
    anc_out : KernelRef
        ``[B, levels]`` the ancestor of each column (``-1``: none).
    next_out : KernelRef
        ``[B]`` the ancestor the next chunk starts from.
    width : int
        ``Wp``, lanes per leaf (a power of two >= the leaf capacity). Static.
    levels : int
        Ancestors per chunk. Static.

    Returns
    -------
    None
        Writes the three output blocks.
    """
    start = start_ref[...]
    count = count_ref[...]
    anc = anc_ref[...]
    lane = lax.broadcasted_iota(start.dtype, (start.shape[0], width), 1)
    valid = lane < count[:, None]
    idx = jnp.where(valid, start[:, None] + lane, 0)
    px = pos_ref[idx, 0]
    py = pos_ref[idx, 1]
    pz = pos_ref[idx, 2]
    zero = jnp.zeros_like(px)
    for j in range(levels):
        live = anc >= 0
        a = jnp.where(live, anc, 0)
        dx = px - cent_ref[a, 0][:, None]
        dy = py - cent_ref[a, 1][:, None]
        dz = pz - cent_ref[a, 2][:, None]
        d2 = dx * dx + dy * dy + dz * dz
        m = jnp.max(jnp.where(valid, d2, zero), axis=1)
        d2_out[:, j] = jnp.where(live, m, jnp.zeros_like(m))
        anc_out[:, j] = anc
        up = parent_ref[a]
        anc = jnp.where(live & (up >= 0), up, -1).astype(anc.dtype)
    next_out[...] = anc


def com_radii_chunk_pallas(
    positions_sorted: Array,
    centers: Array,
    parent: Array,
    leaf_start: Array,
    leaf_count: Array,
    anc: Array,
    *,
    leaf_cap: int,
    levels: int = 8,
    block: int = 4,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 4,
) -> tuple[Array, Array, Array]:
    """One chunk of ancestors: ``(d2 [L, levels], anc [L, levels], next [L])``.

    Parameters
    ----------
    positions_sorted : Array
        ``[N, 3]`` positions in tree order.
    centers : Array
        ``[nodes, 3]`` expansion centres (the positions' dtype).
    parent : Array
        ``[nodes]`` parents, ``-1`` at the root.
    leaf_start : Array
        ``[L]`` first particle per leaf.
    leaf_count : Array
        ``[L]`` particles per leaf (``<= leaf_cap``; 0 for padding leaves).
    anc : Array
        ``[L]`` each leaf's first node of the chunk (the leaf itself for the first
        chunk; ``-1``: past the root).
    leaf_cap : int
        Most particles a leaf holds. Static.
    levels : int
        Ancestors per chunk. Static.
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
    tuple[Array, Array, Array]
        Max squared distance and ancestor id per (leaf, level), and the next
        chunk's starting ancestor per leaf.

    Raises
    ------
    RuntimeError
        If Pallas is not available.
    """
    if pl is None:
        raise RuntimeError("jax.experimental.pallas is not available")
    L = int(anc.shape[0])
    B = _next_pow2(block)
    Lp = -(-L // B) * B
    pad = Lp - L
    idx = jnp.asarray(anc).dtype
    start = jnp.pad(jnp.asarray(leaf_start, idx), (0, pad))
    count = jnp.pad(jnp.asarray(leaf_count, idx), (0, pad))
    anc_p = jnp.pad(jnp.asarray(anc, idx), (0, pad), constant_values=-1)
    pos = jnp.asarray(positions_sorted)
    cent = jnp.asarray(centers, pos.dtype)
    par = jnp.asarray(parent, idx)
    kernel = functools.partial(
        _com_radii_chunk_kernel, width=_next_pow2(leaf_cap), levels=int(levels)
    )
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )

    def _full(arr: Array) -> Any:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    def _rows(shape: tuple) -> Any:
        if len(shape) == 1:
            return pl.BlockSpec((B,), lambda i: (i,))
        return pl.BlockSpec((B, shape[1]), lambda i: (i, 0))

    d2, ancs, nxt = pl.pallas_call(
        kernel,
        grid=(Lp // B,),
        in_specs=[
            _full(pos),
            _full(cent),
            _full(par),
            _rows((Lp,)),
            _rows((Lp,)),
            _rows((Lp,)),
        ],
        out_specs=[_rows((Lp, levels)), _rows((Lp, levels)), _rows((Lp,))],
        out_shape=[
            jax.ShapeDtypeStruct((Lp, int(levels)), pos.dtype),
            jax.ShapeDtypeStruct((Lp, int(levels)), idx),
            jax.ShapeDtypeStruct((Lp,), idx),
        ],
        interpret=bool(interpret),
        name=f"com_radii_chunk_l{int(levels)}_b{B}",
        **backend_kwargs,
    )(pos, cent, par, start, count, anc_p)
    return d2[:L], ancs[:L], nxt[:L]
