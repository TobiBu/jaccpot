"""Real M2L over the target CSR, one PAIR PER LANE (plan sub-10ms, Phase 5).

The per-pair kernel (:mod:`jaccpot.pallas.m2l_real_csr`) and the K-source
tiled kernel (:mod:`jaccpot.pallas.m2l_real_csr_tiled`) both spread ONE pair's
``(Bp, Wp)`` coefficient tile over the program's threads, so every rotation
stage is a cross-thread reduction or a ``tl.dot`` with its layout
conversions through shared memory. Measured on an A100 (order 5, fp32): the
tiled kernel ran at the same speed with IEEE dots and with single-pass TF32
tensor-core dots -- the arithmetic is not what costs; the data movement
between the tiny tiles is.

Here a lane (thread) owns one pair end to end: its 36 (order 5) source
coefficients live in registers as separate ``(K,)`` vectors -- one per
``(degree, m)`` -- and the whole M2L (``Dz(az)``, ``B``, ``Dz(-ax)``, ``B``,
z-core, ``B^T``, ``Dz(ax)``, ``B^T``, ``Dz(-az)``) is straight-line
elementwise code generated at trace time with the rotation matrices baked in
as immediates (``B_l`` has ``(2l+1)^2`` entries per degree, zeros skipped).
``cos(m az)``, ``sin(m az)`` come from ``cos(az) = dy / rho``, ``sin(az) = dx /
rho`` by the angle-addition recurrence (no transcendentals but two square
roots and a reciprocal per pair), ``r^-k`` by repeated multiplication. Lanes
accumulate their pairs across the row's tiles; the ``K`` lanes are reduced
once per program at the end. Any order.

Grid: one program per target node, ``K`` lanes per tile of its CSR row
(``K = 32`` = one warp by default: rows average ~32 pairs at N=2e5 leaf 64).
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
from jaccpot.pallas.m2l_real_csr import csr_by_target, m2l_real_csr_tables

__all__ = ["m2l_real_csr_lanes_pallas"]


def _lane_tables(order: int) -> dict:
    """Constants of the lane kernel as Python floats (baked into the trace).

    Parameters
    ----------
    order : int
        Expansion order.

    Returns
    -------
    dict
        ``B[l][i][j]`` (``i, j`` in ``0 .. 2l``, ``= m + l``), ``Zf[n][k]``,
        ``signm[m]`` for ``m = -p .. p`` (index ``m + p``), ``packed[(l, m)]``
        the packed coefficient index, ``C``.
    """
    tb = m2l_real_csr_tables(int(order))
    p = int(order)
    bstack = np.asarray(tb["Bstack"])
    B = []
    for ell in range(p + 1):
        blk = bstack[ell, p - ell : p + ell + 1, p - ell : p + ell + 1]
        B.append([[float(x) for x in row] for row in blk])
    zf = [[float(x) for x in row] for row in np.asarray(tb["Zf"])]
    signm = [float(x) for x in np.asarray(tb["signm"])[: 2 * p + 1]]
    idx = np.asarray(tb["idx"])
    packed = {(ell, m): int(idx[ell, p + m]) for ell in range(p + 1) for m in range(-ell, ell + 1)}
    return dict(p=p, C=int(tb["C"]), B=B, Zf=zf, signm=signm, packed=packed)


def _bapply_lanes(v: dict, B: list, *, transpose: bool, p: int) -> dict:
    """``out[l, i] = sum_j B_l[i, j] v[l, j]`` (or ``B_l^T``) on lane vectors."""
    out = {}
    for ell in range(p + 1):
        for i in range(-ell, ell + 1):
            acc = None
            for j in range(-ell, ell + 1):
                b = B[ell][ell + j][ell + i] if transpose else B[ell][ell + i][ell + j]
                if b == 0.0:
                    continue
                term = v[(ell, j)] * b if b != 1.0 else v[(ell, j)]
                acc = term if acc is None else acc + term
            out[(ell, i)] = acc if acc is not None else jnp.zeros_like(v[(0, 0)])
    return out


def _dz_lanes(v: dict, cosm: list, sinm: list, *, sign: float, p: int) -> dict:
    """``Dz(t)``: ``out[+m] = cos v[+m] - sin v[-m]``, ``out[-m] = cos v[-m] + sin v[+m]``.

    ``cosm[m]``, ``sinm[m]`` are ``cos(m t)``, ``sin(m t)`` for ``t`` the base
    angle; ``sign = -1`` gives ``Dz(-t)``.
    """
    out = {}
    for ell in range(p + 1):
        out[(ell, 0)] = v[(ell, 0)]
        for m in range(1, ell + 1):
            c = cosm[m]
            s = sinm[m] * sign if sign != 1.0 else sinm[m]
            out[(ell, m)] = v[(ell, m)] * c - v[(ell, -m)] * s
            out[(ell, -m)] = v[(ell, -m)] * c + v[(ell, m)] * s
    return out


def _angle_powers(c1: Array, s1: Array, p: int) -> tuple[list, list]:
    """``cos(m t)``, ``sin(m t)`` for ``m = 0 .. p`` from ``cos t``, ``sin t``."""
    cosm = [jnp.ones_like(c1), c1]
    sinm = [jnp.zeros_like(s1), s1]
    for _ in range(2, p + 1):
        c_prev, s_prev = cosm[-1], sinm[-1]
        cosm.append(c1 * c_prev - s1 * s_prev)
        sinm.append(s1 * c_prev + c1 * s_prev)
    return cosm[: p + 1], sinm[: p + 1]


def _m2l_lanes_kernel(
    mult_ref: KernelRef,
    cent_ref: KernelRef,
    src_ref: KernelRef,
    off_ref: KernelRef,
    cnt_ref: KernelRef,
    out_ref: KernelRef,
    *,
    p: int,
    k_lanes: int,
    tables: dict,
) -> None:
    """One program per target; ``k_lanes`` pairs per iteration, one per lane.

    Parameters
    ----------
    mult_ref : KernelRef
        Packed multipoles ``[n, C]``.
    cent_ref : KernelRef
        Padded centres ``[n, 4]``.
    src_ref : KernelRef
        Target-sorted sources ``[P]``.
    off_ref, cnt_ref : KernelRef
        Row start and length per target ``[n]``.
    out_ref : KernelRef
        This target's packed local row ``[1, C]``.
    p : int
        Order. Static.
    k_lanes : int
        Lanes per tile. Static.
    tables : dict
        :func:`_lane_tables`. Static.

    Returns
    -------
    None
        Writes the row.
    """
    dtype = out_ref.dtype
    B = tables["B"]
    Zf = tables["Zf"]
    signm = tables["signm"]
    packed = tables["packed"]
    tgt = pl.program_id(0)
    start = off_ref[tgt]
    cnt = cnt_ref[tgt]
    ctx = cent_ref[tgt, 0]
    cty = cent_ref[tgt, 1]
    ctz = cent_ref[tgt, 2]
    lane = lax.broadcasted_iota(jnp.int32, (k_lanes,), 0)
    n_tiles = (cnt + (k_lanes - 1)) // k_lanes
    keys = [(ell, m) for ell in range(p + 1) for m in range(-ell, ell + 1)]
    acc0 = tuple(jnp.zeros((k_lanes,), dtype) for _ in keys)

    def body(t, accs):
        pos = t * k_lanes + lane
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
        r = jnp.sqrt(rho2 + dz_ * dz_)
        r = jnp.maximum(r, jnp.asarray(1.0e-30, dtype=dtype))
        rinv = 1.0 / r
        # az = atan2(dx, dy): cos az = dy / rho, sin az = dx / rho (rho = 0: az = 0)
        rho_ok = rho > 0.0
        rho_inv = jnp.where(rho_ok, 1.0 / jnp.where(rho_ok, rho, 1.0), 0.0)
        cos_az = jnp.where(rho_ok, dy * rho_inv, jnp.ones_like(dy))
        sin_az = dx * rho_inv
        # ax = atan2(rho, dz): cos ax = dz / r, sin ax = rho / r
        cos_ax = dz_ * rinv
        sin_ax = rho * rinv
        cos_maz, sin_maz = _angle_powers(cos_az, sin_az, p)
        cos_max, sin_max = _angle_powers(cos_ax, sin_ax, p)
        rinv_pow = [jnp.ones_like(rinv)]
        for _ in range(2 * p + 1):
            rinv_pow.append(rinv_pow[-1] * rinv)
        v = {key: mult_ref[sid, packed[key]] for key in keys}
        # world -> z (multipole): B Dz(-ax) B Dz(az)
        v = _dz_lanes(v, cos_maz, sin_maz, sign=1.0, p=p)
        v = _bapply_lanes(v, B, transpose=False, p=p)
        v = _dz_lanes(v, cos_max, sin_max, sign=-1.0, p=p)
        v = _bapply_lanes(v, B, transpose=False, p=p)
        # z-core: same m; out[n, m] = signm[m] r^-(n+1) sum_k (n+k)! r^-k v[k, m]
        w = {}
        for n in range(p + 1):
            for m in range(-n, n + 1):
                acc = None
                for k in range(abs(m), p - n + 1):
                    term = v[(k, m)] * (rinv_pow[k] * Zf[n][k])
                    acc = term if acc is None else acc + term
                if acc is None:
                    acc = jnp.zeros_like(rinv)
                w[(n, m)] = acc * (rinv_pow[n + 1] * signm[m + p])
        # z -> world (local): Dz(-az) B^T Dz(ax) B^T
        w = _bapply_lanes(w, B, transpose=True, p=p)
        w = _dz_lanes(w, cos_max, sin_max, sign=1.0, p=p)
        w = _bapply_lanes(w, B, transpose=True, p=p)
        w = _dz_lanes(w, cos_maz, sin_maz, sign=-1.0, p=p)
        mask = valid.astype(dtype)
        return tuple(a + w[key] * mask for a, key in zip(accs, keys))

    accs = lax.fori_loop(0, n_tiles, body, acc0)
    for key, a in zip(keys, accs):
        out_ref[0, packed[key]] = jnp.sum(a)


def m2l_real_csr_lanes_pallas(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    *,
    order: int,
    active_pair_count: Optional[Array] = None,
    k_lanes: int = 32,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 1,
) -> Array:
    """Local increments from a flat far-pair list, one pair per lane.

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
        Expansion order.
    active_pair_count : Optional[Array]
        Live prefix length of the pair list.
    k_lanes : int
        Pairs per iteration (lanes). Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program (``k_lanes / 32`` is natural).

    Returns
    -------
    Array
        ``[n, C]`` local increments, dtype of ``multipoles``.

    Raises
    ------
    ValueError
        On a shape mismatch or a non-positive ``k_lanes``.
    """
    p = int(order)
    if int(k_lanes) < 1:
        raise ValueError("k_lanes must be positive")
    tables = _lane_tables(p)
    C = tables["C"]
    mult = jnp.asarray(multipoles)
    dtype = mult.dtype
    n = int(mult.shape[0])
    if int(mult.shape[1]) != C:
        raise ValueError(f"multipoles must have {C} coefficients for order {p}")
    cent = jnp.asarray(centers, dtype=dtype)
    if cent.ndim != 2 or int(cent.shape[1]) != 3 or int(cent.shape[0]) != n:
        raise ValueError("centers must have shape (n, 3) aligned with multipoles")
    cent_p = jnp.pad(cent, ((0, 0), (0, 1)))
    src_sorted, offsets, counts = csr_by_target(
        sources, targets, total_nodes=n, active_pair_count=active_pair_count
    )
    if n == 0 or int(src_sorted.shape[0]) == 0:
        return jnp.zeros((n, C), dtype=dtype)
    kernel = functools.partial(_m2l_lanes_kernel, p=p, k_lanes=int(k_lanes), tables=tables)
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )

    def bs_full(arr: Array) -> pl.BlockSpec:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    operands = [mult, cent_p, src_sorted, offsets, counts]
    return pl.pallas_call(
        kernel,
        grid=(n,),
        in_specs=[bs_full(o) for o in operands],
        out_specs=pl.BlockSpec((1, C), lambda t: (t, 0)),
        out_shape=jax.ShapeDtypeStruct((n, C), dtype),
        interpret=bool(interpret),
        name=f"m2l_real_csr_lanes_p{p}_k{int(k_lanes)}",
        **backend_kwargs,
    )(*operands)
