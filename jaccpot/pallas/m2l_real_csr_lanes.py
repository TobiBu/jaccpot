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

__all__ = [
    "m2l_real_csr_lanes_pallas",
    "m2l_real_csr_lanes_pallas_cvjp",
    "m2l_real_csr_lanes_reverse_pallas",
]


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
    packed = {
        (ell, m): int(idx[ell, p + m])
        for ell in range(p + 1)
        for m in range(-ell, ell + 1)
    }
    return dict(p=p, C=int(tb["C"]), B=B, Zf=zf, signm=signm, packed=packed)


def _bapply_lanes(v: dict, B: list, *, transpose: bool, p: int) -> dict:
    """``out[l, i] = sum_j B_l[i, j] v[l, j]`` (or ``B_l^T``) on lane vectors.

    Parameters
    ----------
    v : dict
        Lane vectors keyed by ``(ell, m)``.
    B : list
        Per-degree dense blocks, ``B[ell][i][j]``.
    transpose : bool
        Apply ``B_l^T`` instead of ``B_l``.
    p : int
        Expansion order.

    Returns
    -------
    dict
        The transformed lane vectors, same keys as ``v``.
    """
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

    Parameters
    ----------
    v : dict
        Lane vectors keyed by ``(ell, m)``.
    cosm : list
        ``cos(m t)`` for ``m = 0 .. p``, ``t`` the base angle.
    sinm : list
        ``sin(m t)`` for the same ``m``.
    sign : float
        ``-1`` gives ``Dz(-t)``.
    p : int
        Expansion order.

    Returns
    -------
    dict
        The rotated lane vectors, same keys as ``v``.
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
    """``cos(m t)``, ``sin(m t)`` for ``m = 0 .. p`` from ``cos t``, ``sin t``.

    Parameters
    ----------
    c1 : Array
        ``cos t``.
    s1 : Array
        ``sin t``.
    p : int
        Highest multiple required.

    Returns
    -------
    tuple[list, list]
        ``(cosm, sinm)``, each of length ``p + 1``.
    """
    cosm = [jnp.ones_like(c1), c1]
    sinm = [jnp.zeros_like(s1), s1]
    for _ in range(2, p + 1):
        c_prev, s_prev = cosm[-1], sinm[-1]
        cosm.append(c1 * c_prev - s1 * s_prev)
        sinm.append(s1 * c_prev + c1 * s_prev)
    return cosm[: p + 1], sinm[: p + 1]


def _m2l_lane_pair(
    v: dict,
    dx: Array,
    dy: Array,
    dz_: Array,
    *,
    p: int,
    tables: dict,
    safe: bool = False,
) -> dict:
    """The whole real M2L of one pair per lane: ``(source coeffs, delta) -> local coeffs``.

    Parameters
    ----------
    v : dict
        Source multipole per ``(degree, m)`` key, lane vectors ``(K,)``.
    dx : Array
        Target minus source centre, ``x``; lanes.
    dy : Array
        Same, ``y``.
    dz_ : Array
        Same, ``z`` (padding lanes carry ``1``).
    p : int
        Order. Static.
    tables : dict
        :func:`_lane_tables`. Static.
    safe : bool
        Guard ``sqrt(rho2)`` at ``rho == 0`` with a double-``where`` so the
        transpose stays finite (the reverse pass; every padding lane sits at
        ``rho == 0``). The primal is bit-identical.

    Returns
    -------
    dict
        Local coefficients per key, lane vectors.
    """
    B = tables["B"]
    Zf = tables["Zf"]
    signm = tables["signm"]
    dtype = dx.dtype
    rho2 = dx * dx + dy * dy
    if safe:
        rho_pos = rho2 > 0.0
        rho = jnp.where(rho_pos, jnp.sqrt(jnp.where(rho_pos, rho2, 1.0)), 0.0)
    else:
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
    return _dz_lanes(w, cos_maz, sin_maz, sign=-1.0, p=p)


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
    off_ref : KernelRef
        Row start per target ``[n]``.
    cnt_ref : KernelRef
        Row length per target ``[n]``.
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

    def body(t: Array, accs: tuple[Array, ...]) -> tuple[Array, ...]:
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
        v = {key: mult_ref[sid, packed[key]] for key in keys}
        w = _m2l_lane_pair(v, dx, dy, dz_, p=p, tables=tables)
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
    sources : Array
        ``[P]`` far-pair sources (negative = padding).
    targets : Array
        ``[P]`` far-pair targets, aligned with ``sources``.
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
    kernel = functools.partial(
        _m2l_lanes_kernel, p=p, k_lanes=int(k_lanes), tables=tables
    )
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


# ------------------------------------------------------------------ reverse
# Adjoint over the TRANSPOSED list (plan fast-gradients, Phase 2): the forward
# sums sources per target; the multipole cotangent sums targets per source, so
# the reverse runs one program per SOURCE over the CSR by source (the same
# ``csr_by_target`` with the roles swapped). Per pair, ``jax.vjp`` of the same
# lane body gives both halves at once: ``mult_bar[s] += J^T loc_bar[t]`` and the
# geometry cotangent ``dbar = <loc_bar[t], dM2L/ddelta mult[s]>`` with
# ``delta = c_t - c_s``, so ``c_s`` takes ``-dbar`` (program-local) and ``c_t``
# takes ``+dbar`` -- collected by a SECOND pass of the same body over the
# forward's CSR by target (one program per target, sources per lane), which
# recomputes each pair's vjp instead of writing anything per pair: the per-pair
# table + segment sum it replaced was a 2.9 ms XLA scatter over the 2M-row
# padded list at N = 2x10^5, the second pass ~0.5 ms.


def _m2l_rev_lanes_kernel(
    mult_ref: KernelRef,
    cent_ref: KernelRef,
    idx_ref: KernelRef,
    off_ref: KernelRef,
    cnt_ref: KernelRef,
    lbar_ref: KernelRef,
    mbar_ref: KernelRef,
    cbar_ref: KernelRef,
    *,
    p: int,
    k_lanes: int,
    tables: dict,
    by_target: bool,
) -> None:
    """One program per row of a CSR; ``k_lanes`` of its pairs per iteration.

    Two instantiations of one body (plan fast-gradients, follow-up: the per-pair
    scatter this replaced was 2.9 ms at N = 2x10^5):

    * ``by_target=False`` -- one program per SOURCE over the CSR by source: the
      source multipole is program-constant, the local cotangent is gathered per
      lane, and the program owns ``mult_bar[s]`` and the source half of the
      centre cotangent (``-sum dbar``);
    * ``by_target=True`` -- one program per TARGET over the forward's CSR by
      target: the local cotangent is program-constant, the multipoles are
      gathered per lane, and the program owns the target half (``+sum dbar``).

    Both pull ``jax.vjp`` of the same pair body; the second pass recomputes the
    pair to avoid writing anything per pair.

    Parameters
    ----------
    mult_ref : KernelRef
        Packed multipoles ``[n, C]``.
    cent_ref : KernelRef
        Padded centres ``[n, 4]``.
    idx_ref : KernelRef
        Row-sorted partner ids ``[P]`` (targets by source, or sources by target).
    off_ref : KernelRef
        Row start per node ``[n]``.
    cnt_ref : KernelRef
        Row length per node ``[n]``.
    lbar_ref : KernelRef
        Local cotangents ``[n, C]``.
    mbar_ref : KernelRef
        **Output** multipole cotangent row ``[1, C]`` (zeros in the by-target pass).
    cbar_ref : KernelRef
        **Output** this node's half of its centre cotangent ``[1, 4]``.
    p : int
        Order. Static.
    k_lanes : int
        Lanes per tile. Static.
    tables : dict
        :func:`_lane_tables`. Static.
    by_target : bool
        Which pass. Static.

    Returns
    -------
    None
        Writes the rows.
    """
    dtype = mbar_ref.dtype
    packed = tables["packed"]
    me = pl.program_id(0)
    start = off_ref[me]
    cnt = cnt_ref[me]
    cx = cent_ref[me, 0]
    cy = cent_ref[me, 1]
    cz = cent_ref[me, 2]
    lane = lax.broadcasted_iota(jnp.int32, (k_lanes,), 0)
    n_tiles = (cnt + (k_lanes - 1)) // k_lanes
    keys = [(ell, m) for ell in range(p + 1) for m in range(-ell, ell + 1)]
    ones = jnp.ones((k_lanes,), dtype)
    zeros = jnp.zeros((k_lanes,), dtype)
    if by_target:
        gb_const = tuple(lbar_ref[me, packed[key]] * ones for key in keys)
    else:
        v_const = tuple(mult_ref[me, packed[key]] * ones for key in keys)
    acc0 = (tuple(zeros for _ in keys), zeros, zeros, zeros)

    def pair(vs: tuple, dx: Array, dy: Array, dz_: Array) -> tuple:
        v = {key: vs[i] for i, key in enumerate(keys)}
        w = _m2l_lane_pair(v, dx, dy, dz_, p=p, tables=tables, safe=True)
        return tuple(w[key] for key in keys)

    def body(t: Array, carry: tuple) -> tuple:
        accs, ax_, ay_, az_ = carry
        pos = t * k_lanes + lane
        valid = pos < cnt
        i_safe = jnp.where(valid, start + pos, start)
        other = idx_ref[i_safe]
        # delta = target - source, whichever end this program is
        if by_target:
            dx = cx - cent_ref[other, 0]
            dy = cy - cent_ref[other, 1]
            dz_ = cz - cent_ref[other, 2]
        else:
            dx = cent_ref[other, 0] - cx
            dy = cent_ref[other, 1] - cy
            dz_ = cent_ref[other, 2] - cz
        zero = jnp.zeros_like(dx)
        dx = jnp.where(valid, dx, zero)
        dy = jnp.where(valid, dy, zero)
        dz_ = jnp.where(valid, dz_, jnp.ones_like(dz_))
        mask = valid.astype(dtype)
        if by_target:
            vs = tuple(mult_ref[other, packed[key]] for key in keys)
            gb = tuple(g * mask for g in gb_const)
        else:
            vs = v_const
            gb = tuple(lbar_ref[other, packed[key]] * mask for key in keys)
        _, vjp = jax.vjp(pair, vs, dx, dy, dz_)
        vb, dxb, dyb, dzb = vjp(gb)
        if by_target:
            return (accs, ax_ + dxb, ay_ + dyb, az_ + dzb)
        return (tuple(a + b for a, b in zip(accs, vb)), ax_ + dxb, ay_ + dyb, az_ + dzb)

    accs, ax_, ay_, az_ = lax.fori_loop(0, n_tiles, body, acc0)
    sign = 1.0 if by_target else -1.0
    for key, a in zip(keys, accs):
        mbar_ref[0, packed[key]] = jnp.sum(a)
    cbar_ref[0, 0] = sign * jnp.sum(ax_)
    cbar_ref[0, 1] = sign * jnp.sum(ay_)
    cbar_ref[0, 2] = sign * jnp.sum(az_)
    cbar_ref[0, 3] = jnp.zeros((), dtype)


def m2l_real_csr_lanes_reverse_pallas(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    loc_bar: Array,
    *,
    order: int,
    active_pair_count: Optional[Array] = None,
    k_lanes: int = 32,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 1,
) -> tuple[Array, Array]:
    """Adjoint of :func:`m2l_real_csr_lanes_pallas`: a by-source and a by-target pass.

    Parameters
    ----------
    multipoles : Array
        ``[n, C]`` real multipoles (the forward's).
    centers : Array
        ``[n, 3]`` expansion centres.
    sources : Array
        ``[P]`` source ids of the directed far pairs (negative = padding).
    targets : Array
        ``[P]`` target ids, aligned.
    loc_bar : Array
        ``[n, C]`` cotangent of the local increments.
    order : int
        Expansion order.
    active_pair_count : Optional[Array]
        Live prefix length of the pair list.
    k_lanes : int
        Pairs per iteration. Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.

    Returns
    -------
    tuple[Array, Array]
        ``(multipoles_bar [n, C], centers_bar [n, 3])``.
    """
    p = int(order)
    tables = _lane_tables(p)
    C = tables["C"]
    mult = jnp.asarray(multipoles)
    dtype = mult.dtype
    n = int(mult.shape[0])
    cent = jnp.asarray(centers, dtype=dtype)
    cent_p = jnp.pad(cent, ((0, 0), (0, 1)))
    lbar = jnp.asarray(loc_bar, dtype)
    K = int(k_lanes)
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )

    def bs_full(arr: Array) -> pl.BlockSpec:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    def one_pass(idx_sorted: Array, offsets: Array, counts: Array, *, by_target: bool):
        kernel = functools.partial(
            _m2l_rev_lanes_kernel, p=p, k_lanes=K, tables=tables, by_target=by_target
        )
        operands = [mult, cent_p, idx_sorted, offsets, counts, lbar]
        return pl.pallas_call(
            kernel,
            grid=(n,),
            in_specs=[bs_full(o) for o in operands],
            out_specs=[
                pl.BlockSpec((1, C), lambda s: (s, 0)),
                pl.BlockSpec((1, 4), lambda s: (s, 0)),
            ],
            out_shape=[
                jax.ShapeDtypeStruct((n, C), dtype),
                jax.ShapeDtypeStruct((n, 4), dtype),
            ],
            interpret=bool(interpret),
            name=f"m2l_rev_real_csr_lanes_{'tgt' if by_target else 'src'}_p{p}_k{K}",
            **backend_kwargs,
        )(*operands)

    # by SOURCE (the same sort with the roles swapped): multipole cotangents + the
    # source half of the centre cotangents
    tgt_sorted, off_s, cnt_s = csr_by_target(
        targets, sources, total_nodes=n, active_pair_count=active_pair_count
    )
    if n == 0 or int(tgt_sorted.shape[0]) == 0:
        return jnp.zeros((n, C), dtype), jnp.zeros_like(centers)
    mbar, cbar_src = one_pass(tgt_sorted, off_s, cnt_s, by_target=False)
    # by TARGET (the forward's CSR): the target half of the centre cotangents
    src_sorted, off_t, cnt_t = csr_by_target(
        sources, targets, total_nodes=n, active_pair_count=active_pair_count
    )
    _, cbar_tgt = one_pass(src_sorted, off_t, cnt_t, by_target=True)
    centers_bar = (cbar_src[:, :3] + cbar_tgt[:, :3]).astype(centers.dtype)
    return mbar, centers_bar


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def m2l_real_csr_lanes_pallas_cvjp(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    active_pair_count: Optional[Array],
    order: int,
    k_lanes: int,
    interpret: bool,
    backend: str,
    num_warps: int,
) -> Array:
    """Differentiable :func:`m2l_real_csr_lanes_pallas` (forward byte-identical).

    Parameters
    ----------
    multipoles : Array
        ``[n, C]`` real multipoles. Differentiable.
    centers : Array
        ``[n, 3]`` expansion centres. Differentiable.
    sources : Array
        ``[P]`` source ids (topology).
    targets : Array
        ``[P]`` target ids (topology).
    active_pair_count : Optional[Array]
        Live prefix length, or ``None``.
    order : int
        Expansion order. ``nondiff_argnums``.
    k_lanes : int
        Pairs per iteration. ``nondiff_argnums``.
    interpret : bool
        Pallas interpret mode. ``nondiff_argnums``.
    backend : str
        Pallas GPU lowering. ``nondiff_argnums``.
    num_warps : int
        Warps per program. ``nondiff_argnums``.

    Returns
    -------
    Array
        ``[n, C]`` local increments.
    """
    return m2l_real_csr_lanes_pallas(
        multipoles,
        centers,
        sources,
        targets,
        order=order,
        active_pair_count=active_pair_count,
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )


def _m2l_lanes_cvjp_fwd(
    multipoles,
    centers,
    sources,
    targets,
    active_pair_count,
    order,
    k_lanes,
    interpret,
    backend,
    num_warps,
):
    out = m2l_real_csr_lanes_pallas(
        multipoles,
        centers,
        sources,
        targets,
        order=order,
        active_pair_count=active_pair_count,
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return out, (multipoles, centers, sources, targets, active_pair_count)


def _m2l_lanes_cvjp_bwd(
    order, k_lanes, interpret, backend, num_warps, residual, loc_bar
):
    multipoles, centers, sources, targets, active_pair_count = residual
    mult_bar, centers_bar = m2l_real_csr_lanes_reverse_pallas(
        multipoles,
        centers,
        sources,
        targets,
        loc_bar,
        order=order,
        active_pair_count=active_pair_count,
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return (mult_bar, centers_bar, None, None, None)


m2l_real_csr_lanes_pallas_cvjp.defvjp(_m2l_lanes_cvjp_fwd, _m2l_lanes_cvjp_bwd)
