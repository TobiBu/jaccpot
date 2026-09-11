"""Tiled real M2L over the target CSR: K source pairs per iteration (plan sub-10ms, Phase 5).

:mod:`jaccpot.pallas.m2l_real_csr` runs one program per target and, inside it,
one pair per ``fori_loop`` iteration: nine broadcast-reductions over
``(Bp, Wp, Wp)`` temporaries (four rotations ``B``, four ``Dz``, the z-core)
for ~20 kflop of useful work. Measured on synthetic lists (order 5, fp32,
A100): 21-25 ns per pair whatever the row shape (rows of 5 .. 1348 pairs, one
or 32 rows of 762 among rows of 21) and whatever the warp count -- about 14 us
per pair-iteration per program, i.e. bound by the latency of that chain, not
by rows and not by flops.

This kernel keeps the math and the CSR-by-target layout and changes the
iteration unit: a TILE of ``K`` consecutive sources of one target row per
iteration, every coefficient held as ``p + 1`` per-degree ``(K, Wp)`` tiles,
so each rotation stage is one ``(K, Wp) @ (Wp, Wp)`` dot per degree
(``precision=HIGHEST``: the Pallas Triton lowering maps DEFAULT fp32 dots to
TF32) and each ``Dz`` is an elementwise product plus one dot with the constant
antisymmetric pattern. The z-core mixes degrees with constant factorials and
per-lane radius powers (tile FMAs). Lanes past the row's end load a valid row
with a benign displacement and are masked out of the accumulation. Degrees
``p+1 .. Bp-1`` of the centred layout are padding and are written as zeros.

``Wp >= 16`` is required by ``tl.dot`` (orders 4 .. 7); lower orders keep the
per-pair kernel.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jaxtyping import Array

from jaccpot.pallas._compat import KernelRef, pallas_backend_kwargs
from jaccpot.pallas.m2l_real_csr import (
    csr_by_target,
    m2l_real_csr_tables,
    pack_centred,
    unpack_centred,
)

__all__ = ["m2l_real_csr_tiled_pallas", "m2l_real_csr_tiled_supported"]

_HIGHEST = lax.Precision.HIGHEST


def m2l_real_csr_tiled_supported(order: int) -> bool:
    """Whether the tiled kernel handles ``order`` (needs ``Wp >= 16``).

    Parameters
    ----------
    order : int
        Expansion order.

    Returns
    -------
    bool
        ``True`` for orders whose centred width pads to at least 16 columns.
    """
    return int(m2l_real_csr_tables(int(order))["Wp"]) >= 16


def _m2l_tiled_kernel(
    mult_ref: KernelRef,
    cent_ref: KernelRef,
    src_ref: KernelRef,
    off_ref: KernelRef,
    cnt_ref: KernelRef,
    bt_ref: KernelRef,
    b_ref: KernelRef,
    apat_t_ref: KernelRef,
    mabs_ref: KernelRef,
    signm_ref: KernelRef,
    out_ref: KernelRef,
    *,
    p: int,
    bp: int,
    wp: int,
    k_tile: int,
    zf: tuple,
) -> None:
    """One program per target: tiles of ``k_tile`` sources of its CSR row.

    Parameters
    ----------
    mult_ref : KernelRef
        Centred multipole table ``[n, Bp*Wp]``.
    cent_ref : KernelRef
        Padded centres ``[n, 4]``.
    src_ref : KernelRef
        Target-sorted sources ``[P]``.
    off_ref, cnt_ref : KernelRef
        Row start and length per target ``[n]``.
    bt_ref : KernelRef
        Per-degree ``B_l^T`` stacked ``[Bp*Wp, Wp]`` (rows ``l*Wp .. (l+1)*Wp``).
    b_ref : KernelRef
        Per-degree ``B_l`` stacked the same way.
    apat_t_ref : KernelRef
        ``Apat^T`` ``[Wp, Wp]``.
    mabs_ref, signm_ref : KernelRef
        ``|m|`` and the z-core sign per column ``[Wp]``.
    out_ref : KernelRef
        This target's centred local row ``[1, Bp*Wp]``.
    p : int
        Order. Static.
    bp, wp : int
        Centred layout. Static.
    k_tile : int
        Sources per iteration. Static.
    zf : tuple
        ``Zf[n][k]`` factorial table as Python floats. Static.

    Returns
    -------
    None
        Writes the row.
    """
    dtype = out_ref.dtype
    tgt = pl.program_id(0)
    start = off_ref[tgt]
    cnt = cnt_ref[tgt]
    ctx = cent_ref[tgt, 0]
    cty = cent_ref[tgt, 1]
    ctz = cent_ref[tgt, 2]
    lane = lax.broadcasted_iota(jnp.int32, (k_tile,), 0)
    cols = lax.broadcasted_iota(jnp.int32, (wp,), 0)
    mabs = mabs_ref[...]
    signm = signm_ref[...]
    apat_t = apat_t_ref[...]
    bt_mats = [bt_ref[pl.ds(l * wp, wp), :] for l in range(p + 1)]
    b_mats = [b_ref[pl.ds(l * wp, wp), :] for l in range(p + 1)]
    n_tiles = (cnt + (k_tile - 1)) // k_tile
    acc0 = tuple(jnp.zeros((wp,), dtype) for _ in range(p + 1))

    def dz(tiles, cosv, sinv):
        # out[k, i] = cos_i v[k, i] + sum_j Apat[i, j] sin_j v[k, j]
        return [
            v * cosv + jnp.dot(v * sinv, apat_t, precision=_HIGHEST) for v in tiles
        ]

    def bapply(tiles, mats):
        # out[k, i] = sum_j M_l[i, j] v[k, j]  ->  v @ M_l^T; ``mats`` already transposed
        return [jnp.dot(v, m, precision=_HIGHEST) for v, m in zip(tiles, mats)]

    def body(t, accs):
        pos = t * k_tile + lane
        valid = pos < cnt
        i_safe = jnp.where(valid, start + pos, start)
        sid = src_ref[i_safe]
        dx = ctx - cent_ref[sid, 0]
        dy = cty - cent_ref[sid, 1]
        dz_ = ctz - cent_ref[sid, 2]
        zero = jnp.zeros_like(dx)
        dx = jnp.where(valid, dx, zero)
        dy = jnp.where(valid, dy, zero)
        dz_ = jnp.where(valid, dz_, jnp.ones_like(dz_))
        rho2 = dx * dx + dy * dy
        rho = jnp.sqrt(rho2)
        az = jnp.arctan2(dx, dy)
        ax = jnp.arctan2(rho, dz_)
        r = jnp.sqrt(rho2 + dz_ * dz_)
        r = jnp.maximum(r, jnp.asarray(1.0e-30, dtype=dtype))
        log_rinv = -jnp.log(r)
        m_az = mabs[None, :] * az[:, None]
        m_ax = mabs[None, :] * ax[:, None]
        cos_az = jnp.cos(m_az)
        sin_az = jnp.sin(m_az)
        cos_ax = jnp.cos(m_ax)
        sin_ax = jnp.sin(m_ax)
        tiles = [
            mult_ref[sid[:, None], (l * wp + cols)[None, :]] for l in range(p + 1)
        ]
        # world -> z (multipole): B Dz(-ax) B Dz(az)
        v = dz(tiles, cos_az, sin_az)
        v = bapply(v, bt_mats)
        v = dz(v, cos_ax, -sin_ax)
        v = bapply(v, bt_mats)
        # z-core: same m, degree x degree; r^-(n+k+1) = rinv^(n+1) rinv^k
        rinv_k = [jnp.exp(float(k) * log_rinv)[:, None] for k in range(p + 1)]
        vk = [v[k] * rinv_k[k] for k in range(p + 1)]
        w = []
        for n in range(p + 1):
            acc = None
            for k in range(p - n + 1):
                term = vk[k] * float(zf[n][k])
                acc = term if acc is None else acc + term
            rinv_n = jnp.exp(float(n + 1) * log_rinv)[:, None]
            w.append(acc * rinv_n * signm[None, :])
        # z -> world (local): Dz(-az) B^T Dz(ax) B^T
        w = bapply(w, b_mats)
        w = dz(w, cos_ax, sin_ax)
        w = bapply(w, b_mats)
        w = dz(w, cos_az, -sin_az)
        m = valid.astype(dtype)[:, None]
        return tuple(a + jnp.sum(wl * m, axis=0) for a, wl in zip(accs, w))

    accs = lax.fori_loop(0, n_tiles, body, acc0)
    for l in range(bp):
        row = accs[l] if l <= p else jnp.zeros((wp,), dtype)
        out_ref[0, pl.ds(l * wp, wp)] = row


def _tiled_tables(order: int, dtype: Any) -> dict:
    """Constant tables of the tiled kernel at ``dtype``.

    Parameters
    ----------
    order : int
        Expansion order.
    dtype : Any
        Working dtype.

    Returns
    -------
    dict
        ``bt`` (``[Bp*Wp, Wp]`` stacked ``B_l^T``), ``b`` (stacked ``B_l``),
        ``apat_t``, ``mabs``, ``signm`` as arrays; ``zf`` as nested Python
        floats; the layout scalars.
    """
    tb = m2l_real_csr_tables(int(order))
    Bp, Wp = int(tb["Bp"]), int(tb["Wp"])
    bstack = np.asarray(tb["Bstack"])  # (Bp, Wp, Wp): out_i = sum_j B[l,i,j] v_j
    bt = np.swapaxes(bstack, -1, -2).reshape(Bp * Wp, Wp)  # v @ B_l^T
    b = bstack.reshape(Bp * Wp, Wp)  # v @ B_l  (== v @ (B_l^T)^T)
    return dict(
        Bp=Bp,
        Wp=Wp,
        C=int(tb["C"]),
        bt=jnp.asarray(bt, dtype),
        b=jnp.asarray(b, dtype),
        apat_t=jnp.asarray(np.asarray(tb["Apat"]).T, dtype),
        mabs=jnp.asarray(tb["mabs"], dtype),
        signm=jnp.asarray(tb["signm"], dtype),
        zf=tuple(tuple(float(x) for x in row) for row in np.asarray(tb["Zf"])),
    )


def m2l_real_csr_tiled_pallas(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    *,
    order: int,
    active_pair_count: Optional[Array] = None,
    k_tile: int = 16,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 4,
) -> Array:
    """Local increments from a flat far-pair list, tiled per target row.

    Same contract as :func:`jaccpot.pallas.m2l_real_csr.m2l_real_csr_pallas`.

    Parameters
    ----------
    multipoles : Array
        ``[n, C]`` real multipoles.
    centers : Array
        ``[n, 3]`` expansion centres.
    sources, targets : Array
        ``[P]`` directed far pairs (negative = padding).
    order : int
        Expansion order (4 .. 7 for this kernel).
    active_pair_count : Optional[Array]
        Live prefix length of the pair list.
    k_tile : int
        Sources per iteration (multiple of 16). Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.

    Returns
    -------
    Array
        ``[n, C]`` local increments, dtype of ``multipoles``.

    Raises
    ------
    ValueError
        On a shape mismatch, an unsupported order, or a ``k_tile`` that is not
        a positive multiple of 16.
    """
    p = int(order)
    if not m2l_real_csr_tiled_supported(p):
        raise ValueError(f"tiled M2L needs Wp >= 16 (order >= 4); got order {p}")
    if int(k_tile) < 16 or int(k_tile) % 16:
        raise ValueError("k_tile must be a positive multiple of 16")
    mult = jnp.asarray(multipoles)
    dtype = mult.dtype
    tb = _tiled_tables(p, dtype)
    C, Bp, Wp = tb["C"], tb["Bp"], tb["Wp"]
    n = int(mult.shape[0])
    if int(mult.shape[1]) != C:
        raise ValueError(f"multipoles must have {C} coefficients for order {p}")
    cent = jnp.asarray(centers, dtype=dtype)
    if cent.ndim != 2 or int(cent.shape[1]) != 3 or int(cent.shape[0]) != n:
        raise ValueError("centers must have shape (n, 3) aligned with multipoles")
    mult_c = pack_centred(mult, order=p)
    cent_p = jnp.pad(cent, ((0, 0), (0, 1)))
    src_sorted, offsets, counts = csr_by_target(
        sources, targets, total_nodes=n, active_pair_count=active_pair_count
    )
    if n == 0 or int(src_sorted.shape[0]) == 0:
        return jnp.zeros((n, C), dtype=dtype)
    kernel = functools.partial(
        _m2l_tiled_kernel, p=p, bp=Bp, wp=Wp, k_tile=int(k_tile), zf=tb["zf"]
    )
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )

    def bs_full(arr: Array) -> pl.BlockSpec:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    operands = [
        mult_c, cent_p, src_sorted, offsets, counts,
        tb["bt"], tb["b"], tb["apat_t"], tb["mabs"], tb["signm"],
    ]
    out = pl.pallas_call(
        kernel,
        grid=(n,),
        in_specs=[bs_full(o) for o in operands],
        out_specs=pl.BlockSpec((1, Bp * Wp), lambda t: (t, 0)),
        out_shape=jax.ShapeDtypeStruct((n, Bp * Wp), dtype),
        interpret=bool(interpret),
        name=f"m2l_real_csr_tiled_p{p}_k{int(k_tile)}",
        **backend_kwargs,
    )(*operands)
    return unpack_centred(out, order=p)
