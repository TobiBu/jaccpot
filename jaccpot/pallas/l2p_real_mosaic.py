"""The particle-major L2P forward on Mosaic GPU (spike S1b; opt-in, not a default).

The same function as :func:`jaccpot.pallas.l2p_real.l2p_real_particles_pallas` (the
Triton kernel): the far-field gradient at every particle in its own leaf's local
expansion, by the Cartesian recurrence of
:func:`~jaccpot.pallas.l2p_real.l2p_real_grad_cartesian` -- the identical Python
arithmetic, traced into a different backend. It exists to answer whether jaccpot's
kernels can leave Triton-Pallas (deprecated in jax 0.11.2) for Mosaic GPU on an
A100 (sm_80); the findings are in ``docs/mosaic_gpu_spike.md``.

What Mosaic GPU on jax 0.11.2 can and cannot express here, measured:

* ``pl.pallas_call`` has no Mosaic-GPU lowering rule (only Triton and TPU register
  one): the entry point is :func:`jax.experimental.pallas.mosaic_gpu.kernel`, a
  grid of warpgroups (128 threads) whose refs are whole GMEM arrays.
* Arrays are warpgroup-distributed ``FragmentedArray`` s; here ``(128,)`` in the
  ``WG_STRIDED`` layout, one particle per thread, the Triton kernel's shape.
* A load indexed by an ARRAY (``ref[idx]``, the Triton kernel's gather) does not
  lower (``NotImplementedError: int_indexer_shape non-empty``). Scalar dynamic
  loads (``ref[j, k]``, one address for the whole warpgroup) and contiguous slices
  (``ref[pl.ds(s, 128), c]``, also on a column of an ``[N, 3]`` array) do.

So the gather is done one of two ways (``gather=``):

* ``"select"`` -- pure Pallas, portable: the block's particles lie in a window of
  consecutive leaves; a loop over that window loads each leaf's centre and
  coefficients as scalars and selects them into the lanes of its particles. Every
  value a lane ends up with is the stored value, bit for bit, so the arithmetic is
  the Triton kernel's. Cost: ``(leaves in the window) x ((p+1)^2 + 3)`` selects per
  lane on top of the recurrence. Runs in Mosaic GPU's CPU interpret mode.
* ``"inline"`` -- the loop selects only the leaf's node id; the centre and the
  ``(p+1)^2`` coefficients are then gathered per thread by hand-written Mosaic GPU
  IR (:func:`jax.experimental.pallas.mosaic_gpu.inline_mgpu`: one ``memref.load``
  per register element). Not supported by the interpret mode
  (``ValueError: GPU interpret mode does not support inline_mgpu``) and built on
  ``jax._src`` MLIR bindings.

The block covers rows ``[s, s + 128)`` with ``s = min(128 b, N - 128)``: the last
block overlaps its neighbour instead of running past ``N`` (no masked stores
exist on the Pallas side), recomputing identical values for the overlap.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from jaccpot.pallas._compat import KernelRef
from jaccpot.pallas.l2p_real import (
    l2p_real_grad_cartesian,
    l2p_real_particles_reverse_pallas,
)

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as mgpu_pl
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None
    mgpu_pl = None

#: Particles per program: one warpgroup, one particle per thread.
_LANES = 128

#: Gather strategies (see the module docstring).
GATHER_MODES = ("select", "inline")


def _layout() -> Any:
    return mgpu_pl.Layout.WG_STRIDED((_LANES,), vec_size=1)


def _gather_rows(ref: KernelRef, idx: Array, cols: tuple[int, ...]) -> list:
    """``[ref[idx, c] for c in cols]`` per thread, in hand-written Mosaic GPU IR.

    Parameters
    ----------
    ref : KernelRef
        A 2-D GMEM ref.
    idx : Array
        ``(128,)`` int32 row of each lane, ``WG_STRIDED`` layout.
    cols : tuple[int, ...]
        Static columns to read.

    Returns
    -------
    list
        One ``(128,)`` array per column, in ``ref``'s dtype.
    """
    from jax._src.lib.mlir import ir
    from jax._src.lib.mlir.dialects import arith, memref

    lay = _layout()
    ret = mgpu_pl.ShapeDtypeStruct((_LANES,), ref.dtype, layout=lay)

    @mgpu_pl.inline_mgpu(
        arg_types=(mgpu_pl.RefType(), lay), return_type=tuple(ret for _ in cols)
    )
    def _g(ctx: Any, r: Any, ix: Any) -> tuple:
        del ctx
        index = ir.IndexType.get()
        out = []
        for c in cols:
            col = arith.constant(index, int(c))

            def one(v: Any, _: Any, col: Any = col) -> Any:
                return memref.load(r, [arith.index_cast(index, v), col])

            out.append(ix.foreach(one, create_array=True, is_signed=None))
        return tuple(out)

    return list(_g(ref, idx))


def _l2p_mosaic_kernel(
    pos_ref: KernelRef,
    cent_ref: KernelRef,
    coef_ref: KernelRef,
    leaf_node_ref: KernelRef,
    leaf_start_ref: KernelRef,
    leaf_end_ref: KernelRef,
    first_ref: KernelRef,
    nwin_ref: KernelRef,
    out_ref: KernelRef,
    *,
    order: int,
    rows: int,
    gather: str,
) -> None:
    """128 consecutive particles, each in its own leaf's expansion.

    Parameters
    ----------
    pos_ref : KernelRef
        Whole sorted positions ``[N, 3]``.
    cent_ref : KernelRef
        Whole expansion centres ``[nodes, 3]``.
    coef_ref : KernelRef
        Whole local coefficients ``[nodes, (p+1)^2]``.
    leaf_node_ref : KernelRef
        Whole ``[L]`` node id of each leaf.
    leaf_start_ref : KernelRef
        Whole ``[L]`` first particle of each leaf.
    leaf_end_ref : KernelRef
        Whole ``[L]`` one past each leaf's last particle.
    first_ref : KernelRef
        Whole ``[blocks]`` first leaf of each block's window.
    nwin_ref : KernelRef
        Whole ``[blocks]`` leaves in each block's window (0: no live particle).
    out_ref : KernelRef
        **Output** ``[N, 3]`` gradient (every row written).
    order : int
        Expansion order. Static.
    rows : int
        ``N`` (at least 128). Static.
    gather : str
        ``"select"`` or ``"inline"``. Static.

    Returns
    -------
    None
        Writes the block's rows.
    """
    W = _LANES
    b = lax.axis_index("blocks")
    s = jnp.minimum(b * W, rows - W)
    lane = mgpu_pl.broadcasted_iota(jnp.int32, (W,), 0, layout=_layout())
    i = s + lane
    first = first_ref[b]
    nwin = nwin_ref[b]
    C = (int(order) + 1) ** 2
    dtype = coef_ref.dtype
    # carries derived from `lane` keep its layout through the loop. Under Lane
    # semantics an iota-derived int array is tagged unsigned while `where` with an
    # int scalar returns a signed one, and the loop rejects the carry
    # (PyTreeDef ... False vs True); a `where` makes the initial value signed too.
    izero = jnp.where(lane >= 0, 0, lane)
    fzero = izero.astype(dtype)

    def _hit(t: Array) -> tuple:
        j = first + t
        return j, (i >= leaf_start_ref[j]) & (i < leaf_end_ref[j])

    if gather == "select":

        def body(t: Array, acc: tuple) -> tuple:
            j, hit = _hit(t)
            nd = leaf_node_ref[j]
            vals = [cent_ref[nd, a] for a in range(3)]
            vals += [coef_ref[nd, k] for k in range(C)]
            return (
                jnp.where(hit, 1, acc[0]),
                *(jnp.where(hit, v, a) for v, a in zip(vals, acc[1:])),
            )

        found, cx, cy, cz, *coeff = lax.fori_loop(
            0, nwin, body, (izero, *(fzero for _ in range(C + 3)))
        )
    else:

        def body_nd(t: Array, acc: tuple) -> tuple:
            j, hit = _hit(t)
            return jnp.where(hit, 1, acc[0]), jnp.where(hit, leaf_node_ref[j], acc[1])

        found, nd = lax.fori_loop(0, nwin, body_nd, (izero, izero))
        cx, cy, cz = _gather_rows(cent_ref, nd, (0, 1, 2))
        coeff = _gather_rows(coef_ref, nd, tuple(range(C)))
    live = found > 0
    x = cx - pos_ref[pl.ds(s, W), 0].astype(dtype)
    y = cy - pos_ref[pl.ds(s, W), 1].astype(dtype)
    z = cz - pos_ref[pl.ds(s, W), 2].astype(dtype)
    gx, gy, gz = l2p_real_grad_cartesian(coeff, x, y, z, order=order)
    for a, g in enumerate((gx, gy, gz)):
        out_ref[pl.ds(s, W), a] = jnp.where(live, g, 0).astype(out_ref.dtype)


def l2p_real_particles_mosaic(
    coefficients: Array,
    centers: Array,
    positions_sorted: Array,
    leaf_nodes: Array,
    leaf_start: Array,
    leaf_count: Array,
    *,
    order: int,
    gather: str = "inline",
    interpret: bool = False,
    lowering_semantics: str = "Lane",
) -> Array:
    """``[N, 3]`` far-field gradient (with respect to ``delta``) on Mosaic GPU.

    Same inputs and outputs as
    :func:`~jaccpot.pallas.l2p_real.l2p_real_particles_pallas`.

    Parameters
    ----------
    coefficients : Array
        ``[nodes, (p+1)^2]`` real local coefficients.
    centers : Array
        ``[nodes, 3]`` expansion centres.
    positions_sorted : Array
        ``[N, 3]`` positions in tree order.
    leaf_nodes : Array
        ``[L]`` node id of each leaf, in particle order (empty leaves at the end).
    leaf_start : Array
        ``[L]`` first particle of each leaf.
    leaf_count : Array
        ``[L]`` particles of each leaf (0: empty).
    order : int
        Expansion order. Static.
    gather : str
        ``"inline"`` (hand-written per-thread loads; GPU only) or ``"select"``
        (pure Pallas; also runs in interpret mode). Static.
    interpret : bool
        Mosaic GPU's CPU interpret mode (``gather="select"`` only).
    lowering_semantics : str
        ``"Lane"`` or ``"Warpgroup"`` (``gather="select"`` only). Static.

    Returns
    -------
    Array
        ``[N, 3]`` gradient in the positions' dtype; zero for rows past the leaves'
        particles.

    Raises
    ------
    RuntimeError
        If Mosaic GPU Pallas is not available.
    ValueError
        If ``gather`` is unknown, or ``"inline"`` is asked of the interpret mode or
        of Warpgroup semantics.
    """
    if mgpu_pl is None:
        raise RuntimeError("jax.experimental.pallas.mosaic_gpu is not available")
    if gather not in GATHER_MODES:
        raise ValueError(f"gather={gather!r}: expected one of {GATHER_MODES}")
    if gather == "inline" and (interpret or lowering_semantics != "Lane"):
        raise ValueError(
            "gather='inline' needs the Lane semantics on a GPU: inline_mgpu has no "
            "interpret mode and takes FragmentedArrays only under Lane"
        )
    pos = jnp.asarray(positions_sorted)
    n = int(pos.shape[0])
    L = int(leaf_nodes.shape[0])
    if n == 0 or L == 0:
        return jnp.zeros((n, 3), pos.dtype)
    W = _LANES
    rows = max(n, W)
    if rows > n:
        # fewer particles than one warpgroup: pad (the extra rows are dead)
        pos = jnp.concatenate([pos, jnp.zeros((rows - n, 3), pos.dtype)])
    blocks = -(-rows // W)
    idx = jnp.asarray(leaf_nodes).dtype
    count = jnp.asarray(leaf_count, idx)
    start = jnp.asarray(leaf_start, idx)
    end = start + count
    live = jnp.sum(count).astype(idx)
    starts = jnp.minimum(jnp.arange(blocks, dtype=idx) * W, rows - W)
    # the window: from the leaf of the block's first row to the leaf of its last
    # LIVE row (never the empty padding leaves past them)
    first = jnp.searchsorted(end, starts, side="right", method="scan_unrolled")
    last = jnp.searchsorted(
        end, jnp.minimum(starts + W, live) - 1, side="right", method="scan_unrolled"
    )
    nwin = jnp.where(starts < live, last - first + 1, 0).astype(idx)
    first = jnp.minimum(first, L - 1).astype(idx)
    coef = jnp.asarray(coefficients)
    cent = jnp.asarray(centers, coef.dtype)
    kw: dict[str, Any] = {}
    if interpret:
        kw["interpret"] = mgpu_pl.InterpretGPUParams()
    out = mgpu_pl.kernel(
        functools.partial(
            _l2p_mosaic_kernel, order=int(order), rows=rows, gather=str(gather)
        ),
        out_type=jax.ShapeDtypeStruct((rows, 3), pos.dtype),
        grid=(blocks,),
        grid_names=("blocks",),
        compiler_params=mgpu_pl.CompilerParams(
            lowering_semantics=getattr(mgpu_pl.LoweringSemantics, lowering_semantics)
        ),
        **kw,
    )(pos, cent, coef, jnp.asarray(leaf_nodes, idx), start, end, first, nwin)
    return out[:n] if rows > n else out


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def l2p_real_particles_mosaic_cvjp(
    coefficients: Array,
    centers: Array,
    positions_sorted: Array,
    leaf_nodes: Array,
    leaf_start: Array,
    leaf_count: Array,
    order: int,
    gather: str,
    interpret: bool,
) -> Array:
    """Differentiable :func:`l2p_real_particles_mosaic`: the reverse is Triton's.

    The spike ports the forward only; the cotangents come from
    :func:`~jaccpot.pallas.l2p_real.l2p_real_particles_reverse_pallas` (Triton, or
    Triton's interpret mode under ``interpret``), the reverse the Triton forward has.

    Parameters
    ----------
    coefficients : Array
        ``[nodes, (p+1)^2]`` real local coefficients. Differentiable.
    centers : Array
        ``[nodes, 3]`` expansion centres. Differentiable.
    positions_sorted : Array
        ``[N, 3]`` positions in tree order. Differentiable.
    leaf_nodes : Array
        ``[L]`` node id of each leaf (topology).
    leaf_start : Array
        ``[L]`` first particle of each leaf (topology).
    leaf_count : Array
        ``[L]`` particles of each leaf (topology).
    order : int
        Expansion order. ``nondiff_argnums``.
    gather : str
        The forward's gather. ``nondiff_argnums``.
    interpret : bool
        Interpret mode (both directions). ``nondiff_argnums``.

    Returns
    -------
    Array
        ``[N, 3]`` gradient with respect to ``delta``.
    """
    return l2p_real_particles_mosaic(
        coefficients,
        centers,
        positions_sorted,
        leaf_nodes,
        leaf_start,
        leaf_count,
        order=order,
        gather=gather,
        interpret=interpret,
    )


def _mosaic_cvjp_fwd(
    coefficients, centers, positions_sorted, leaf_nodes, leaf_start, leaf_count,
    order, gather, interpret,
):  # fmt: skip
    out = l2p_real_particles_mosaic(
        coefficients,
        centers,
        positions_sorted,
        leaf_nodes,
        leaf_start,
        leaf_count,
        order=order,
        gather=gather,
        interpret=interpret,
    )
    residual = (
        coefficients,
        centers,
        positions_sorted,
        leaf_nodes,
        leaf_start,
        leaf_count,
    )
    return out, residual


def _mosaic_cvjp_bwd(order, gather, interpret, residual, grad_bar):
    del gather
    coef_bar, cent_bar, pos_bar = l2p_real_particles_reverse_pallas(
        *residual, grad_bar, order=order, interpret=interpret
    )
    return (coef_bar, cent_bar, pos_bar, None, None, None)


l2p_real_particles_mosaic_cvjp.defvjp(_mosaic_cvjp_fwd, _mosaic_cvjp_bwd)

__all__ = [
    "GATHER_MODES",
    "l2p_real_particles_mosaic",
    "l2p_real_particles_mosaic_cvjp",
]
