"""Real M2L over the target CSR, one PAIR PER LANE (plan sub-10ms, Phase 5).

The per-pair kernel and the K-source tiled kernel this replaced (both removed in
the 2026-10 cleanup, X5; :mod:`jaccpot.pallas.m2l_real_csr` keeps their shared
algebra and twin) spread ONE pair's ``(Bp, Wp)`` coefficient tile over the
program's threads, so every rotation stage is a cross-thread reduction or a
``tl.dot`` with its layout conversions through shared memory. Measured on an A100 (order 5, fp32): the
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

**Force-scale lane (S3b).** With ``force_scale``
(:class:`~jaccpot.runtime._force_scale_levels.FarForceScaleInputs`) the same
launch also sums ``mac_type='dehnen_error'``'s far term of eq (16b) per target
row, ``G M_s / ((|c_s - c_t| + rho_t)^2 + eps^2)`` over the live pairs, in one
more lane accumulator reduced with the locals; the Pallas L2L cascade then
carries it to the leaves. This is where the fused lane's per-step far ``f_b`` is
computed on the GPU: the separate segment sum it replaced
(:func:`~jaccpot.runtime._force_scale_levels.far_force_scale_sorted`) was a
scatter over the whole far list every step. Without the argument the kernel is
the same program as before.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jaxtyping import Array

from jaccpot.operators.real_rotations import _real_rotation_generator_block
from jaccpot.pallas._compat import KernelRef, pallas_backend_kwargs
from jaccpot.pallas._tangent import linearize_lanes, make_tangent_op
from jaccpot.pallas.m2l_real_csr import (
    csr_by_target,
    m2l_real_csr_tables,
    targets_from_csr_offsets,
)
from jaccpot.runtime._force_scale_levels import FarForceScaleInputs

__all__ = [
    "m2l_real_csr_lanes_fb_pallas_cvjp",
    "m2l_real_csr_lanes_pallas",
    "m2l_real_csr_lanes_pallas_cvjp",
    "m2l_real_csr_lanes_pallas_tangent",
    "m2l_real_csr_lanes_pallas_tangent_jnp",
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
        the packed coefficient index, ``C``, and the rotation generators in
        ``B``'s layout that the reverse needs on the axis
        (:func:`_on_axis_pair_cotangent`): ``GmpX`` / ``GmpY`` on the multipole
        (input) side, ``GlocX`` / ``GlocY`` on the local (output) side.
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
    generators = {
        key: [
            [
                [float(x) for x in row]
                for row in _real_rotation_generator_block(ell, axis, representation)
            ]
            for ell in range(p + 1)
        ]
        for key, axis, representation in (
            ("GmpX", "x", "multipole"),
            ("GmpY", "y", "multipole"),
            ("GlocX", "x", "local"),
            ("GlocY", "y", "local"),
        )
    }
    return dict(
        p=p, C=int(tb["C"]), B=B, Zf=zf, signm=signm, packed=packed, **generators
    )


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
        ``rho == 0``). The primal is bit-identical. The transverse cotangent it
        gives on the axis is zero, and the reverse replaces it there
        (:func:`_on_axis_pair_cotangent`).

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
    *refs: KernelRef,
    p: int,
    k_lanes: int,
    tables: dict,
    force_scale: Optional[tuple[float, bool]] = None,
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
    *refs : KernelRef
        ``out_ref``, this target's packed local row ``[1, C]``. With
        ``force_scale``: ``mass_ref [n]``, ``rad_ref [n]`` (walk radii),
        ``eps_ref [1]`` (``eps^2``), ``wcent_ref [n, 4]`` (padded walk centres,
        only when they are not ``cent_ref``), then ``out_ref`` and ``fb_ref``,
        this target's far force-scale term ``[1]``.
    p : int
        Order. Static.
    k_lanes : int
        Lanes per tile. Static.
    tables : dict
        :func:`_lane_tables`. Static.
    force_scale : Optional[tuple[float, bool]]
        ``(G, walk_centres)``: also sum ``G M_s / ((|c_s - c_t| + rho_t)^2 +
        eps^2)`` over the row's live pairs (:class:`FarForceScaleInputs`), lanes
        reduced once at the end as the locals are. ``None`` (default) is the plain
        M2L, the same program as without the argument. Static.

    Returns
    -------
    None
        Writes the row (and the force-scale term).
    """
    if force_scale is None:
        (out_ref,) = refs
    else:
        g_fb, walk_centres = force_scale
        if walk_centres:
            mass_ref, rad_ref, eps_ref, wcent_ref, out_ref, fb_ref = refs
        else:
            mass_ref, rad_ref, eps_ref, out_ref, fb_ref = refs
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
    if force_scale is not None:
        rho_t = rad_ref[tgt]
        eps2 = eps_ref[0]
        if walk_centres:
            wtx, wty, wtz = wcent_ref[tgt, 0], wcent_ref[tgt, 1], wcent_ref[tgt, 2]
        acc0 = acc0 + (jnp.zeros((k_lanes,), dtype),)

    def body(t: Array, accs: tuple[Array, ...]) -> tuple[Array, ...]:
        pos = t * k_lanes + lane
        valid = pos < cnt
        i_safe = jnp.where(valid, start + pos, start)
        sid = src_ref[i_safe]
        dx = ctx - cent_ref[sid, 0]
        dy = cty - cent_ref[sid, 1]
        dz_ = ctz - cent_ref[sid, 2]
        zero = jnp.zeros_like(dx)
        if force_scale is not None:
            # eq (16b)'s far term, in far_force_scale_own's operation order
            # (|c_s - c_t|^2 as ((x^2 + y^2) + z^2), then (G M) / (reach^2 + eps^2))
            if walk_centres:
                fx = wtx - wcent_ref[sid, 0]
                fy = wty - wcent_ref[sid, 1]
                fz = wtz - wcent_ref[sid, 2]
            else:
                fx, fy, fz = dx, dy, dz_
            reach = jnp.sqrt(fx * fx + fy * fy + fz * fz) + rho_t
            term = g_fb * mass_ref[sid] / (reach * reach + eps2)
            fb_lanes = accs[-1] + jnp.where(valid & (reach > 0.0), term, zero)
            accs = accs[:-1]
        dx = jnp.where(valid, dx, zero)
        dy = jnp.where(valid, dy, zero)
        dz_ = jnp.where(valid, dz_, jnp.ones_like(dz_))
        v = {key: mult_ref[sid, packed[key]] for key in keys}
        w = _m2l_lane_pair(v, dx, dy, dz_, p=p, tables=tables)
        mask = valid.astype(dtype)
        out = tuple(a + w[key] * mask for a, key in zip(accs, keys))
        if force_scale is not None:
            out = out + (fb_lanes,)
        return out

    accs = lax.fori_loop(0, n_tiles, body, acc0)
    if force_scale is not None:
        fb_ref[0] = jnp.sum(accs[-1])
        accs = accs[:-1]
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
    n_targets: Optional[int] = None,
    k_lanes: int = 32,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 1,
    presorted: bool = False,
    force_scale: Optional[FarForceScaleInputs] = None,
) -> Union[Array, tuple[Array, Array]]:
    """Local increments from a flat far-pair list, one pair per lane.

    The reference is :func:`jaccpot.pallas.m2l_real_csr.m2l_real_csr_jax` (same
    arguments, the per-pair twin reduced by target).

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
    n_targets : Optional[int]
        Number of TARGET rows, when the source array is longer than the target
        range. ``None`` (default) means ``n``, the single-tree case. Static.

        This exists for the one-sided distributed lane, where the multipole array
        is a concatenation ``[local ; imported]`` and only the local prefix
        receives. Without it the grid, the ``out_shape`` and ``csr_by_target``'s
        ``total_nodes`` all come from ``multipoles.shape[0]``, so the kernel emits
        ``n_local + n_remote`` rows -- twice the launches and twice the output
        buffer for rows that are discarded, and then a shape mismatch when the
        result is added to the local-only accumulator. Rows beyond ``n_targets``
        cost exactly nothing; sources may still index the whole array.
    k_lanes : int
        Pairs per iteration (lanes). Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program (``k_lanes / 32`` is natural).
    presorted : bool
        ``sources`` is already in CSR order (``TargetSortedFarPairs``) and
        ``targets`` holds its row offsets (``[>= n_targets + 1]``), not one target
        per entry: no CSR sort, no copy (``csr_by_target(presorted=True)``).
        Static.
    force_scale : Optional[FarForceScaleInputs]
        Also return eq (16b)'s far term per target row (``mac_type=
        'dehnen_error'``'s per-step force scale; :class:`FarForceScaleInputs`):
        the same launch, one more output. ``None`` (default) is the plain M2L,
        the same program as without the argument. Forward only: differentiate
        through :func:`m2l_real_csr_lanes_fb_pallas_cvjp`.

    Returns
    -------
    Union[Array, tuple[Array, Array]]
        ``[n_targets, C]`` local increments, dtype of ``multipoles``; with
        ``force_scale``, ``(increments, far_term [n_targets])``.

    Raises
    ------
    ValueError
        On a shape mismatch, a non-positive ``k_lanes``, an ``n_targets``
        outside ``[0, n]``, or ``force_scale`` arrays that do not match the
        nodes or the multipoles' dtype.
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
    nt = n if n_targets is None else int(n_targets)
    if not (0 <= nt <= n):
        raise ValueError(f"n_targets must lie in [0, {n}], got {nt}")
    src_sorted, offsets, counts = csr_by_target(
        sources,
        targets,
        total_nodes=nt,
        active_pair_count=active_pair_count,
        presorted=bool(presorted),
    )
    fb_operands = []
    fb_static = None
    if force_scale is not None:
        fb_operands, fb_static = _force_scale_operands(force_scale, n=n, dtype=dtype)
    if nt == 0 or int(src_sorted.shape[0]) == 0:
        if force_scale is not None:
            return jnp.zeros((nt, C), dtype=dtype), jnp.zeros((nt,), dtype=dtype)
        return jnp.zeros((nt, C), dtype=dtype)
    if force_scale is None:
        kernel = functools.partial(
            _m2l_lanes_kernel, p=p, k_lanes=int(k_lanes), tables=tables
        )
    else:
        kernel = functools.partial(
            _m2l_lanes_kernel,
            p=p,
            k_lanes=int(k_lanes),
            tables=tables,
            force_scale=fb_static,
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
    if force_scale is not None:
        operands = operands + fb_operands
        return pl.pallas_call(
            kernel,
            grid=(nt,),
            in_specs=[bs_full(o) for o in operands],
            out_specs=[
                pl.BlockSpec((1, C), lambda t: (t, 0)),
                pl.BlockSpec((1,), lambda t: (t,)),
            ],
            out_shape=[
                jax.ShapeDtypeStruct((nt, C), dtype),
                jax.ShapeDtypeStruct((nt,), dtype),
            ],
            interpret=bool(interpret),
            name=f"m2l_real_csr_lanes_fb_p{p}_k{int(k_lanes)}",
            **backend_kwargs,
        )(*operands)
    return pl.pallas_call(
        kernel,
        grid=(nt,),
        in_specs=[bs_full(o) for o in operands],
        out_specs=pl.BlockSpec((1, C), lambda t: (t, 0)),
        out_shape=jax.ShapeDtypeStruct((nt, C), dtype),
        interpret=bool(interpret),
        name=f"m2l_real_csr_lanes_p{p}_k{int(k_lanes)}",
        **backend_kwargs,
    )(*operands)


def _force_scale_operands(
    force_scale: FarForceScaleInputs, *, n: int, dtype: Any
) -> tuple[list, tuple[float, bool]]:
    """The force-scale lane's kernel operands and its static ``(G, walk_centres)``.

    Parameters
    ----------
    force_scale : FarForceScaleInputs
        The lane's inputs.
    n : int
        Node count (the multipoles' rows).
    dtype : Any
        The multipoles' dtype, which the lane computes in.

    Returns
    -------
    tuple[list, tuple[float, bool]]
        ``[mass [n], radii [n], eps^2 [1](, padded walk centres [n, 4])]`` and
        the kernel's ``force_scale`` argument.

    Raises
    ------
    ValueError
        If an array does not have one entry per node or is not in ``dtype``: the
        lane would compute in another precision than the function it replaces
        (:func:`jaccpot.runtime._force_scale_levels.far_force_scale_own`).
    """
    mass = jnp.asarray(force_scale.node_mass)
    radii = jnp.asarray(force_scale.node_radii)
    eps2 = jnp.asarray(force_scale.softening_sq).reshape((1,))
    arrays = [mass, radii, eps2]
    walk_centres = force_scale.node_centers is not None
    if walk_centres:
        wcent = jnp.asarray(force_scale.node_centers)
        if wcent.shape != (n, 3):
            raise ValueError(f"force_scale.node_centers must be ({n}, 3)")
        arrays.append(jnp.pad(wcent, ((0, 0), (0, 1))))
    if mass.shape != (n,) or radii.shape != (n,):
        raise ValueError(f"force_scale node_mass / node_radii must be ({n},)")
    if any(a.dtype != dtype for a in arrays):
        raise ValueError(
            f"force_scale arrays must be {jnp.dtype(dtype)}, the multipoles' dtype"
        )
    return arrays, (float(force_scale.gravitational_constant), bool(walk_centres))


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


def _on_axis_pair_cotangent(
    keys: list,
    vs: tuple,
    w: tuple,
    vb: tuple,
    gb: tuple,
    delta3: tuple,
    valid: Array,
    dxb: Array,
    dyb: Array,
    *,
    p: int,
    tables: dict,
) -> tuple[Array, Array]:
    """The pairs' transverse cotangents, with on-axis lanes given the analytic limit.

    :func:`_m2l_lane_pair` reaches ``dx`` and ``dy`` only through ``rho`` and the
    azimuth, so on the axis its guards give a zero transverse cotangent and just
    off it (``rho/r`` below ``sqrt(eps)``) a cancellation-dominated one, while the
    M2L itself is differentiable there. For such a lane this returns the limit the
    pure-JAX operators' rule supplies (:mod:`jaccpot.operators._transverse_degeneracy_jvp`,
    formula (2)/(3), transposed); with ``in_bar = F^T out_bar``,

        dL/dx = +(1/z) (<out_bar, G_loc^y out> - <in_bar, G_mp^y in>)
        dL/dy = -(1/z) (<out_bar, G_loc^x out> - <in_bar, G_mp^x in>)

    and the polar values everywhere else, bit for bit. Same band as that rule
    (``rho^2 <= eps r^2``, ``z != 0``). The generator images are computed only for
    a tile that has an on-axis lane.

    Parameters
    ----------
    keys : list
        The ``(degree, m)`` keys, in the order of the tuples below.
    vs : tuple
        Source multipoles per key, lane vectors.
    w : tuple
        The pairs' local outputs per key.
    vb : tuple
        ``F^T gb`` per key.
    gb : tuple
        The cotangent of ``w`` per key.
    delta3 : tuple
        ``(dx, dy, dz)`` lane vectors, target minus source.
    valid : Array
        Lane mask of real pairs.
    dxb : Array
        The polar route's ``x`` cotangent per lane.
    dyb : Array
        Same, ``y``.
    p : int
        Order. Static.
    tables : dict
        :func:`_lane_tables`. Static.

    Returns
    -------
    tuple[Array, Array]
        ``(dxb, dyb)`` per lane.
    """
    dx, dy, dz_ = delta3
    rho2 = dx * dx + dy * dy
    epsilon = float(jnp.finfo(dx.dtype).eps)
    on_axis = valid & (rho2 <= epsilon * (rho2 + dz_ * dz_)) & (dz_ != 0)

    def _pair(g_in: list, g_out: list) -> Array:
        image_out = _bapply_lanes(dict(zip(keys, w)), g_out, transpose=False, p=p)
        image_in = _bapply_lanes(dict(zip(keys, vs)), g_in, transpose=False, p=p)
        acc = jnp.zeros_like(dx)
        for i, key in enumerate(keys):
            acc = acc + gb[i] * image_out[key] - vb[i] * image_in[key]
        return acc

    def _analytic() -> tuple[Array, Array]:
        inverse_z = jnp.where(on_axis, 1.0 / jnp.where(on_axis, dz_, 1.0), 0.0)
        axis_x = inverse_z * _pair(tables["GmpY"], tables["GlocY"])
        axis_y = -inverse_z * _pair(tables["GmpX"], tables["GlocX"])
        return jnp.where(on_axis, axis_x, dxb), jnp.where(on_axis, axis_y, dyb)

    # a count, not `jnp.any`: Pallas Triton has no lowering for `reduce_or`
    any_on_axis = jnp.sum(on_axis.astype(jnp.int32)) > 0
    return lax.cond(any_on_axis, _analytic, lambda: (dxb, dyb))


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
        w, vjp = jax.vjp(pair, vs, dx, dy, dz_)
        vb, dxb, dyb, dzb = vjp(gb)
        dxb, dyb = _on_axis_pair_cotangent(
            keys, vs, w, vb, gb, (dx, dy, dz_), valid, dxb, dyb, p=p, tables=tables
        )
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
    target_row_offsets: Optional[Array] = None,
    symmetric: bool = False,
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
    target_row_offsets : Optional[Array]
        The list is already in CSR order by target (a ``TargetSortedFarPairs``
        list, the forward's ``presorted``) with these row offsets
        (``[>= n + 1]``): the by-target pass reads it as is instead of sorting it
        again -- one ``P``-pair sort less. ``targets`` still holds one target per
        entry, for the by-source sort.
    symmetric : bool
        The list holds every directed pair in both directions (the frozen list of a
        mutual walk; the caller checks it): row ``s`` of the forward's CSR -- the
        sources of target ``s`` -- are then exactly the targets of source ``s``, so
        the by-source pass reads the forward's CSR too, and the reverse sorts
        nothing (at 1e8 the by-source sort ran over a 542M-entry capacity for 361M
        live pairs). Needs ``target_row_offsets``; ``targets`` is then unused.
        Static.

    Returns
    -------
    tuple[Array, Array]
        ``(multipoles_bar [n, C], centers_bar [n, 3])``.

    Raises
    ------
    ValueError
        If ``symmetric`` is set without ``target_row_offsets``.
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

    def one_pass(
        idx_sorted: Array, offsets: Array, counts: Array, *, by_target: bool
    ) -> tuple[Array, Array]:
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

    if symmetric:
        if target_row_offsets is None:
            raise ValueError("symmetric needs the forward's CSR (target_row_offsets)")
        src_sorted, off_t, cnt_t = csr_by_target(
            sources, target_row_offsets, total_nodes=n, presorted=True
        )
        if n == 0 or int(src_sorted.shape[0]) == 0:
            return jnp.zeros((n, C), dtype), jnp.zeros_like(centers)
        # row s lists the partners of s at both ends of a pair: by source the
        # multipole cotangents and the source half, by target the target half
        mbar, cbar_src = one_pass(src_sorted, off_t, cnt_t, by_target=False)
        _, cbar_tgt = one_pass(src_sorted, off_t, cnt_t, by_target=True)
        centers_bar = (cbar_src[:, :3] + cbar_tgt[:, :3]).astype(centers.dtype)
        return mbar, centers_bar
    # by SOURCE (the same sort with the roles swapped): multipole cotangents + the
    # source half of the centre cotangents
    tgt_sorted, off_s, cnt_s = csr_by_target(
        targets, sources, total_nodes=n, active_pair_count=active_pair_count
    )
    if n == 0 or int(tgt_sorted.shape[0]) == 0:
        return jnp.zeros((n, C), dtype), jnp.zeros_like(centers)
    mbar, cbar_src = one_pass(tgt_sorted, off_s, cnt_s, by_target=False)
    # by TARGET (the forward's CSR): the target half of the centre cotangents
    if target_row_offsets is None:
        src_sorted, off_t, cnt_t = csr_by_target(
            sources, targets, total_nodes=n, active_pair_count=active_pair_count
        )
    else:
        src_sorted, off_t, cnt_t = csr_by_target(
            sources, target_row_offsets, total_nodes=n, presorted=True
        )
    _, cbar_tgt = one_pass(src_sorted, off_t, cnt_t, by_target=True)
    centers_bar = (cbar_src[:, :3] + cbar_tgt[:, :3]).astype(centers.dtype)
    return mbar, centers_bar


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11, 12))
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
    n_targets: Optional[int] = None,
    presorted: bool = False,
    symmetric: bool = False,
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
    n_targets : Optional[int]
        Forwarded to :func:`m2l_real_csr_lanes_pallas`. Static.

        The FORWARD honours it; the reverse does not, and says so rather than
        returning a quietly wrong gradient. The reverse runs a second pass over the
        by-target CSR whose grid and `loc_bar` rows would both have to be rebased,
        while `mult_bar` stays full length -- real work, and gradients through the
        cross-domain import are a later phase by decision, so it raises.
    presorted : bool
        ``targets`` are the row offsets of a CSR-ordered list: the forward reads
        it as is, the reverse expands one target per entry before its by-source
        sort. ``nondiff_argnums``.
    symmetric : bool
        With ``presorted``: the list holds every directed pair both ways, so the
        reverse reads the forward's CSR in both passes and sorts nothing
        (:func:`m2l_real_csr_lanes_reverse_pallas`). The forward ignores it.
        ``nondiff_argnums``.

    Returns
    -------
    Array
        ``[n_targets, C]`` local increments (``[n, C]`` when ``n_targets`` is None).
    """
    return m2l_real_csr_lanes_pallas(
        multipoles,
        centers,
        sources,
        targets,
        order=order,
        active_pair_count=active_pair_count,
        n_targets=n_targets,
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
        presorted=presorted,
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
    n_targets,
    presorted,
    symmetric,
):
    out = m2l_real_csr_lanes_pallas(
        multipoles,
        centers,
        sources,
        targets,
        order=order,
        active_pair_count=active_pair_count,
        n_targets=n_targets,
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
        presorted=presorted,
    )
    return out, (multipoles, centers, sources, targets, active_pair_count)


def _csr_entry_rows(row_offsets: Array, num_entries: int) -> Array:
    """:func:`~jaccpot.pallas.m2l_real_csr.targets_from_csr_offsets` in two passes.

    The same integers -- entry ``j``'s row is the number of interior offsets
    ``<= j``, ``-1`` past the live prefix -- from one scatter of the row
    boundaries and one prefix sum over the entries, instead of a binary search
    per entry (one compare-select pass over the entries per bisection step: 9.8
    of the far gradient's 95 ms at 8e6, 44M pairs).

    Parameters
    ----------
    row_offsets : Array
        ``[R + 1]`` non-decreasing row offsets, ``row_offsets[0] == 0``.
    num_entries : int
        Length of the list (live prefix plus padding). Static.

    Returns
    -------
    Array
        ``[num_entries]`` row ids in ``row_offsets``' dtype, ``-1`` past the live
        prefix.
    """
    off = jnp.asarray(row_offsets)
    n = int(num_entries)
    starts = (
        jnp.zeros((n + 1,), off.dtype)
        .at[off[1:-1]]
        .add(jnp.ones_like(off[1:-1]), mode="drop")
    )
    row = jnp.cumsum(starts[:n], dtype=off.dtype)
    pos = jnp.arange(n, dtype=off.dtype)
    return jnp.where(pos < off[-1], row, jnp.asarray(-1, off.dtype))


def _m2l_lanes_cvjp_bwd(
    order,
    k_lanes,
    interpret,
    backend,
    num_warps,
    n_targets,
    presorted,
    symmetric,
    residual,
    loc_bar,
):
    multipoles, centers, sources, targets, active_pair_count = residual
    row_offsets = None
    if presorted:
        # the forward got the row offsets; the reverse's by-target pass reads the
        # forward's CSR as is, and so does its by-source pass on a symmetric list;
        # otherwise it sorts by source and needs one target per entry (-1 past the
        # live prefix, as a COO list pads)
        row_offsets = targets
        if not symmetric:
            targets = _csr_entry_rows(targets, int(sources.shape[0]))
    if n_targets is not None and int(n_targets) != int(multipoles.shape[0]):
        raise NotImplementedError(
            "m2l_real_csr_lanes_pallas_cvjp: the reverse does not support "
            f"n_targets={n_targets} against {int(multipoles.shape[0])} multipole "
            "rows. The one-sided cross-domain lane is forward-only for now; "
            "differentiate the single-tree call instead."
        )
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
        target_row_offsets=row_offsets,
        symmetric=bool(symmetric) and bool(presorted),
    )
    return (mult_bar, centers_bar, None, None, None)


m2l_real_csr_lanes_pallas_cvjp.defvjp(_m2l_lanes_cvjp_fwd, _m2l_lanes_cvjp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13, 14, 15, 16))
def m2l_real_csr_lanes_fb_pallas_cvjp(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    active_pair_count: Optional[Array],
    node_mass: Array,
    node_radii: Array,
    softening_sq: Array,
    node_centers: Optional[Array],
    order: int,
    k_lanes: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    n_targets: Optional[int],
    presorted: bool,
    gravitational_constant: float,
) -> tuple[Array, Array]:
    """:func:`m2l_real_csr_lanes_pallas_cvjp` that also returns eq (16b)'s far term.

    The forward is :func:`m2l_real_csr_lanes_pallas` with ``force_scale``: one
    launch, the local increments and the per-target far term. The reverse is
    :func:`m2l_real_csr_lanes_pallas_cvjp`'s for the increments; the far term is
    a topology quantity (it only sets the next walk's eq (16a) thresholds) and is
    not differentiated: its inputs get no cotangent and its own cotangent is
    dropped.

    Parameters
    ----------
    multipoles : Array
        ``[n, C]`` real multipoles. Differentiable.
    centers : Array
        ``[n, 3]`` expansion centres. Differentiable.
    sources : Array
        ``[P]`` source ids (topology).
    targets : Array
        ``[P]`` target ids, or the row offsets with ``presorted`` (topology).
    active_pair_count : Optional[Array]
        Live prefix length, or ``None``.
    node_mass : Array
        :attr:`FarForceScaleInputs.node_mass`. Not differentiated.
    node_radii : Array
        :attr:`FarForceScaleInputs.node_radii`. Not differentiated.
    softening_sq : Array
        :attr:`FarForceScaleInputs.softening_sq`. Not differentiated.
    node_centers : Optional[Array]
        :attr:`FarForceScaleInputs.node_centers`. Not differentiated.
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
    n_targets : Optional[int]
        As :func:`m2l_real_csr_lanes_pallas_cvjp`. ``nondiff_argnums``.
    presorted : bool
        As :func:`m2l_real_csr_lanes_pallas_cvjp`. ``nondiff_argnums``.
    gravitational_constant : float
        :attr:`FarForceScaleInputs.gravitational_constant`. ``nondiff_argnums``.

    Returns
    -------
    tuple[Array, Array]
        ``([n_targets, C]`` local increments, ``[n_targets]`` far term ``)``.
    """
    return _m2l_lanes_fb_forward(
        multipoles,
        centers,
        sources,
        targets,
        active_pair_count,
        node_mass,
        node_radii,
        softening_sq,
        node_centers,
        order,
        k_lanes,
        interpret,
        backend,
        num_warps,
        n_targets,
        presorted,
        gravitational_constant,
    )


def _m2l_lanes_fb_forward(
    multipoles,
    centers,
    sources,
    targets,
    active_pair_count,
    node_mass,
    node_radii,
    softening_sq,
    node_centers,
    order,
    k_lanes,
    interpret,
    backend,
    num_warps,
    n_targets,
    presorted,
    gravitational_constant,
):
    out = m2l_real_csr_lanes_pallas(
        multipoles,
        centers,
        sources,
        targets,
        order=order,
        active_pair_count=active_pair_count,
        n_targets=n_targets,
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
        presorted=presorted,
        force_scale=FarForceScaleInputs(
            node_mass=node_mass,
            node_radii=node_radii,
            softening_sq=softening_sq,
            gravitational_constant=gravitational_constant,
            node_centers=node_centers,
        ),
    )
    return out[0], out[1]


def _m2l_lanes_fb_cvjp_fwd(
    multipoles,
    centers,
    sources,
    targets,
    active_pair_count,
    node_mass,
    node_radii,
    softening_sq,
    node_centers,
    order,
    k_lanes,
    interpret,
    backend,
    num_warps,
    n_targets,
    presorted,
    gravitational_constant,
):
    out = _m2l_lanes_fb_forward(
        multipoles,
        centers,
        sources,
        targets,
        active_pair_count,
        node_mass,
        node_radii,
        softening_sq,
        node_centers,
        order,
        k_lanes,
        interpret,
        backend,
        num_warps,
        n_targets,
        presorted,
        gravitational_constant,
    )
    return out, (multipoles, centers, sources, targets, active_pair_count)


def _m2l_lanes_fb_cvjp_bwd(
    order,
    k_lanes,
    interpret,
    backend,
    num_warps,
    n_targets,
    presorted,
    gravitational_constant,
    residual,
    cotangents,
):
    loc_bar, _ = cotangents
    # symmetric=False: the sorting reverse is right for any list, and nothing
    # guarantees that a dehnen_error far list holds every pair both ways
    grads = _m2l_lanes_cvjp_bwd(
        order,
        k_lanes,
        interpret,
        backend,
        num_warps,
        n_targets,
        presorted,
        False,
        residual,
        loc_bar,
    )
    return grads + (None, None, None, None)


m2l_real_csr_lanes_fb_pallas_cvjp.defvjp(_m2l_lanes_fb_cvjp_fwd, _m2l_lanes_fb_cvjp_bwd)


# ------------------------------------------------------------------ tangent
# Forward mode (``ad_mode="forward"``, plan forward-mode Phase 1, track C). The
# tangent of the local increments flows the same way as the primal -- sources into
# each target row -- so the tangent kernel is the forward's grid and loop (one
# program per target, ``k_lanes`` pairs per tile, the same CSR by target, the same
# accumulation order) with the pair body linearised in-kernel
# (:func:`~jaccpot.pallas._tangent.linearize_lanes`): the pair's primal
# intermediates once, then ``K`` applications of its linear map. No transposed
# list, no by-source sort, no symmetric requirement (those are the reverse's).


def _m2l_pair_tangent_lanes(
    keys: list,
    vs: tuple,
    delta3: tuple,
    tvs: Optional[list],
    tdeltas: Optional[list],
    valid: Array,
    *,
    p: int,
    tables: dict,
) -> list:
    """The ``K`` tangents of one tile of pairs, on-axis lanes given the analytic term.

    The tangent of :func:`_m2l_lane_pair` with respect to the source multipole
    (linear) and the displacement ``delta = c_t - c_s``. :func:`_m2l_lane_pair`
    reaches ``dx`` and ``dy`` only through ``rho`` and the azimuth, so its
    linearisation gives a zero transverse derivative on the axis and a
    cancellation-dominated one just off it. In the band ``rho^2 <= eps r^2``,
    ``z != 0`` (the band of :func:`jaccpot.operators._transverse_degeneracy_jvp.split_transverse_tangent`)
    the transverse tangent is withdrawn from the polar route and replaced by the
    commutator of formula (2)/(3) there, applied (not transposed: the reverse's
    :func:`_on_axis_pair_cotangent` is its transpose)::

        dF . (tx, ty) = (tx / z) [G_loc^y F0 - F0 G_mp^y] v
                      - (ty / z) [G_loc^x F0 - F0 G_mp^x] v

    ``F0 v`` is the pair's primal output; ``F0 G v`` is the pair evaluated on the
    generator image of ``v`` (the pair is linear in the multipole). The two
    commutator images are shared by the ``K`` tangents and computed only for a tile
    that has an on-axis lane.

    Parameters
    ----------
    keys : list
        The ``(degree, m)`` keys, in the order of ``vs``.
    vs : tuple
        Source multipoles per key, lane vectors.
    delta3 : tuple
        ``(dx, dy, dz)`` lane vectors, target minus source (padding lanes at
        ``(0, 0, 1)``).
    tvs : Optional[list]
        Per tangent, the multipole tangents per key (a tuple of lane vectors);
        ``None`` when the multipole tangent is known zero.
    tdeltas : Optional[list]
        Per tangent, ``(tdx, tdy, tdz)``; ``None`` when the centre tangent is known
        zero.
    valid : Array
        Lane mask of real pairs.
    p : int
        Order. Static.
    tables : dict
        :func:`_lane_tables`. Static.

    Returns
    -------
    list
        Per tangent, the local tangents per key (a tuple of lane vectors), NOT
        masked by ``valid``.
    """
    dx, dy, dz_ = delta3

    def pair(v: tuple, ax: Array, ay: Array, az: Array) -> tuple:
        w = _m2l_lane_pair(
            dict(zip(keys, v)), ax, ay, az, p=p, tables=tables, safe=True
        )
        return tuple(w[key] for key in keys)

    if tdeltas is None:
        assert tvs is not None
        _, lin_v = linearize_lanes(lambda v: pair(v, dx, dy, dz_), (vs,))
        return [lin_v(tv) for tv in tvs]
    if tvs is None:
        w, lin_d = linearize_lanes(lambda a, b, c: pair(vs, a, b, c), (dx, dy, dz_))

        def tangent(k: int, ax: Array, ay: Array, az: Array) -> tuple:
            return lin_d(ax, ay, az)

    else:
        tvs_live = tvs
        w, lin_vd = linearize_lanes(pair, (vs, dx, dy, dz_))

        def tangent(k: int, ax: Array, ay: Array, az: Array) -> tuple:
            return lin_vd(tvs_live[k], ax, ay, az)

    rho2 = dx * dx + dy * dy
    epsilon = float(jnp.finfo(dx.dtype).eps)
    on_axis = valid & (rho2 <= epsilon * (rho2 + dz_ * dz_)) & (dz_ != 0)

    def _commutators() -> tuple[tuple, tuple]:
        def one(g_in: list, g_out: list) -> tuple:
            image_out = _bapply_lanes(dict(zip(keys, w)), g_out, transpose=False, p=p)
            image_in = _bapply_lanes(dict(zip(keys, vs)), g_in, transpose=False, p=p)
            f_image = pair(tuple(image_in[key] for key in keys), dx, dy, dz_)
            return tuple(image_out[key] - f for key, f in zip(keys, f_image))

        return (
            one(tables["GmpY"], tables["GlocY"]),
            one(tables["GmpX"], tables["GlocX"]),
        )

    zeros = tuple(jnp.zeros_like(dx) for _ in keys)
    # a count, not `jnp.any`: Pallas Triton has no lowering for `reduce_or`
    any_on_axis = jnp.sum(on_axis.astype(jnp.int32)) > 0
    comm_y, comm_x = lax.cond(any_on_axis, _commutators, lambda: (zeros, zeros))
    inverse_z = jnp.where(on_axis, 1.0 / jnp.where(on_axis, dz_, 1.0), 0.0)
    zero = jnp.zeros_like(dx)
    out = []
    for k, (tdx, tdy, tdz) in enumerate(tdeltas):
        scale_x = inverse_z * tdx
        scale_y = inverse_z * tdy
        routed_x = jnp.where(on_axis, zero, tdx)
        routed_y = jnp.where(on_axis, zero, tdy)
        tw = tangent(k, routed_x, routed_y, tdz)
        out.append(
            tuple(
                t + scale_x * cy - scale_y * cx for t, cy, cx in zip(tw, comm_y, comm_x)
            )
        )
    return out


def _m2l_tangent_lanes_kernel(
    mult_ref: KernelRef,
    cent_ref: KernelRef,
    src_ref: KernelRef,
    off_ref: KernelRef,
    cnt_ref: KernelRef,
    tmult_ref: KernelRef,
    tcent_ref: KernelRef,
    out_ref: KernelRef,
    *,
    p: int,
    k_lanes: int,
    tables: dict,
    k_tangents: int,
    tpp: int,
    live_mult: bool,
    live_cent: bool,
) -> None:
    """One program per (target, tangent group); ``k_lanes`` pairs per iteration.

    :func:`_m2l_lanes_kernel`'s grid, loop and accumulation order, the pair body
    replaced by :func:`_m2l_pair_tangent_lanes` over the ``tpp`` tangents of this
    program's group (grid axis 1, :mod:`jaccpot.pallas._tangent_grid`).

    Parameters
    ----------
    mult_ref : KernelRef
        Packed multipoles ``[n, C]``.
    cent_ref : KernelRef
        Padded centres ``[n, 4]``.
    src_ref : KernelRef
        Target-sorted sources ``[P]``.
    off_ref : KernelRef
        Row start per target ``[n_targets]``.
    cnt_ref : KernelRef
        Row length per target ``[n_targets]``.
    tmult_ref : KernelRef
        Multipole tangents ``[K, n, C]`` (unread when not ``live_mult``).
    tcent_ref : KernelRef
        Padded centre tangents ``[K, n, 4]`` (unread when not ``live_cent``).
    out_ref : KernelRef
        This target's local tangents ``[K, 1, C]`` (this group writes its rows).
    p : int
        Order. Static.
    k_lanes : int
        Lanes per tile. Static.
    tables : dict
        :func:`_lane_tables`. Static.
    k_tangents : int
        ``K``, the tangents of the launch. Static.
    tpp : int
        Tangents per program: tangents ``g tpp .. g tpp + tpp - 1`` of group ``g``,
        those past ``K`` clamped on load and not stored. Static.
    live_mult : bool
        The multipole tangent is not known zero. Static.
    live_cent : bool
        The centre tangent is not known zero. Static.

    Returns
    -------
    None
        Writes the row's tangents of this group.
    """
    dtype = out_ref.dtype
    packed = tables["packed"]
    tgt = pl.program_id(0)
    group = pl.program_id(1)
    tails = int(k_tangents) % int(tpp) != 0
    js = [group * tpp + jj for jj in range(tpp)]
    # a partial last group reads tangent K - 1 in its spare slots, stores none
    jl = [jnp.minimum(j, k_tangents - 1) for j in js] if tails else js
    start = off_ref[tgt]
    cnt = cnt_ref[tgt]
    ctx = cent_ref[tgt, 0]
    cty = cent_ref[tgt, 1]
    ctz = cent_ref[tgt, 2]
    lane = lax.broadcasted_iota(jnp.int32, (k_lanes,), 0)
    n_tiles = (cnt + (k_lanes - 1)) // k_lanes
    keys = [(ell, m) for ell in range(p + 1) for m in range(-ell, ell + 1)]
    n_keys = len(keys)
    acc0 = tuple(jnp.zeros((k_lanes,), dtype) for _ in range(tpp * n_keys))
    # the target's centre tangents, once per program
    ttc: list = []
    if live_cent:
        ttc = [tuple(tcent_ref[j, tgt, a] for a in range(3)) for j in jl]

    def body(t: Array, accs: tuple[Array, ...]) -> tuple[Array, ...]:
        pos = t * k_lanes + lane
        valid = pos < cnt
        i_safe = jnp.where(valid, start + pos, start)
        sid = src_ref[i_safe]
        zero = jnp.zeros((k_lanes,), dtype)
        dx = jnp.where(valid, ctx - cent_ref[sid, 0], zero)
        dy = jnp.where(valid, cty - cent_ref[sid, 1], zero)
        dz_ = jnp.where(valid, ctz - cent_ref[sid, 2], jnp.ones_like(zero))
        vs = tuple(mult_ref[sid, packed[key]] for key in keys)
        tvs = None
        if live_mult:
            tvs = [
                tuple(
                    jnp.where(valid, tmult_ref[j, sid, packed[key]], zero)
                    for key in keys
                )
                for j in jl
            ]
        tdeltas = None
        if live_cent:
            tdeltas = [
                tuple(
                    jnp.where(valid, ttc[jj][a] - tcent_ref[j, sid, a], zero)
                    for a in range(3)
                )
                for jj, j in enumerate(jl)
            ]
        tws = _m2l_pair_tangent_lanes(
            keys, vs, (dx, dy, dz_), tvs, tdeltas, valid, p=p, tables=tables
        )
        mask = valid.astype(dtype)
        out = []
        for jj in range(tpp):
            row = accs[jj * n_keys : (jj + 1) * n_keys]
            out.extend(a + tw * mask for a, tw in zip(row, tws[jj]))
        return tuple(out)

    accs = lax.fori_loop(0, n_tiles, body, acc0)

    def _store(jj: int) -> None:
        for i, key in enumerate(keys):
            out_ref[jl[jj], 0, packed[key]] = jnp.sum(accs[jj * n_keys + i])

    for jj in range(tpp):
        if tails:
            pl.when(js[jj] < k_tangents)(functools.partial(_store, jj))
        else:
            _store(jj)


def _m2l_tangent_csr(
    sources: Array,
    targets: Array,
    active_pair_count: Array,
    *,
    nt: int,
    presorted: bool,
    has_active_count: bool,
) -> tuple[Array, Array, Array]:
    """The forward's CSR by target: :func:`csr_by_target` with the forward's arguments.

    Parameters
    ----------
    sources : Array
        ``[P]`` sources.
    targets : Array
        ``[P]`` targets, or the row offsets with ``presorted``.
    active_pair_count : Array
        Live prefix length (a dummy when not ``has_active_count``).
    nt : int
        Target rows. Static.
    presorted : bool
        As the forward. Static.
    has_active_count : bool
        ``active_pair_count`` is real (the seam's ``None`` otherwise). Static.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(sources_sorted, offsets, counts)``.
    """
    return csr_by_target(
        sources,
        targets,
        total_nodes=nt,
        active_pair_count=active_pair_count if has_active_count else None,
        presorted=bool(presorted),
    )


def _m2l_tangent_impl(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    active_pair_count: Array,
    t_multipoles: Array,
    t_centers: Array,
    *,
    order: int,
    k_lanes: int = 32,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 1,
    n_targets: Optional[int] = None,
    presorted: bool = False,
    symmetric: bool = False,
    has_active_count: bool = True,
    live: tuple[bool, ...] = (True, True),
    tangents_per_program: Optional[int] = None,
) -> list:
    """The Pallas tangent kernel for any ``K`` (see :func:`m2l_real_csr_lanes_pallas_tangent`).

    Parameters
    ----------
    multipoles : Array
        ``[n, C]``.
    centers : Array
        ``[n, 3]``.
    sources : Array
        ``[P]``.
    targets : Array
        ``[P]`` (or row offsets with ``presorted``).
    active_pair_count : Array
        Live prefix length, or a dummy.
    t_multipoles : Array
        ``[K, n, C]``.
    t_centers : Array
        ``[K, n, 3]``.
    order : int
        Expansion order.
    k_lanes : int
        Pairs per iteration.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.
    n_targets : Optional[int]
        Target rows (``None``: ``n``).
    presorted : bool
        CSR-ordered list with row offsets.
    symmetric : bool
        Accepted for the seam's signature; the tangent does not need it.
    has_active_count : bool
        ``active_pair_count`` is real.
    live : tuple[bool, ...]
        ``(multipoles, centers)`` tangent liveness.
    tangents_per_program : Optional[int]
        Tangents per program (``None``: ``JACCPOT_TANGENT_TPP_M2L``; see
        :func:`~jaccpot.pallas._tangent_grid.tangents_per_program`). Static.

    Returns
    -------
    list
        ``[t_locals [K, n_targets, C]]``.

    Raises
    ------
    ValueError
        On a shape mismatch or an ``n_targets`` outside ``[0, n]``.
    """
    from jaccpot.pallas._tangent_grid import tangent_groups
    from jaccpot.pallas._tangent_grid import tangents_per_program as _tpp

    del symmetric
    p = int(order)
    tables = _lane_tables(p)
    C = tables["C"]
    mult = jnp.asarray(multipoles)
    dtype = mult.dtype
    n = int(mult.shape[0])
    if int(mult.shape[1]) != C:
        raise ValueError(f"multipoles must have {C} coefficients for order {p}")
    k_tan = int(t_multipoles.shape[0])
    if tuple(t_multipoles.shape) != (k_tan, n, C) or tuple(t_centers.shape) != (
        k_tan,
        n,
        3,
    ):
        raise ValueError(
            "tangents must be [K, n, C] and [K, n, 3], got "
            f"{tuple(t_multipoles.shape)} and {tuple(t_centers.shape)}"
        )
    nt = n if n_targets is None else int(n_targets)
    if not (0 <= nt <= n):
        raise ValueError(f"n_targets must lie in [0, {n}], got {nt}")
    live_mult, live_cent = (bool(x) for x in live)
    src_sorted, offsets, counts = _m2l_tangent_csr(
        sources,
        targets,
        active_pair_count,
        nt=nt,
        presorted=presorted,
        has_active_count=has_active_count,
    )
    if nt == 0 or int(src_sorted.shape[0]) == 0 or not (live_mult or live_cent):
        return [jnp.zeros((k_tan, nt, C), dtype)]
    cent_p = jnp.pad(jnp.asarray(centers, dtype), ((0, 0), (0, 1)))
    tmult = jnp.asarray(t_multipoles, dtype)
    tcent_p = jnp.pad(jnp.asarray(t_centers, dtype), ((0, 0), (0, 0), (0, 1)))
    tpp = _tpp("M2L", k_tan, tangents_per_program)
    kernel = functools.partial(
        _m2l_tangent_lanes_kernel,
        p=p,
        k_lanes=int(k_lanes),
        tables=tables,
        k_tangents=k_tan,
        tpp=tpp,
        live_mult=live_mult,
        live_cent=live_cent,
    )
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )

    def bs_full(arr: Array) -> pl.BlockSpec:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    operands = [mult, cent_p, src_sorted, offsets, counts, tmult, tcent_p]
    live_tag = ("m" if live_mult else "") + ("c" if live_cent else "")
    out = pl.pallas_call(
        kernel,
        grid=(nt, tangent_groups(k_tan, tpp)),
        in_specs=[bs_full(o) for o in operands],
        out_specs=pl.BlockSpec((k_tan, 1, C), lambda t, g: (0, t, 0)),
        out_shape=jax.ShapeDtypeStruct((k_tan, nt, C), dtype),
        interpret=bool(interpret),
        name=(
            f"m2l_tan_real_csr_lanes_p{p}_k{int(k_lanes)}_K{k_tan}_tpp{tpp}_{live_tag}"
        ),
        **backend_kwargs,
    )(*operands)
    return [out]


def _m2l_tangent_abstract(
    multipoles: Any,
    centers: Any,
    sources: Any,
    targets: Any,
    active_pair_count: Any,
    t_multipoles: Any,
    t_centers: Any,
    **params: Any,
) -> list:
    """Closed-form output shape of the M2L tangent op.

    Parameters
    ----------
    multipoles : Any
        ``[n, C]`` shape/dtype.
    centers : Any
        Unused.
    sources : Any
        Unused.
    targets : Any
        Unused.
    active_pair_count : Any
        Unused.
    t_multipoles : Any
        ``[K, n, C]`` shape/dtype.
    t_centers : Any
        Unused.
    **params : Any
        The op's parameters (``n_targets`` is read).

    Returns
    -------
    list
        ``[ShapeDtypeStruct((K, n_targets, C))]``.
    """
    del centers, sources, targets, active_pair_count, t_centers
    n, C = (int(s) for s in multipoles.shape)
    nt = n if params.get("n_targets") is None else int(params["n_targets"])
    return [jax.ShapeDtypeStruct((int(t_multipoles.shape[0]), nt, C), multipoles.dtype)]


#: Forward-mode tangent of :func:`m2l_real_csr_lanes_pallas_cvjp` (the seam table of
#: the forward-mode plan): ``op(multipoles, centers, sources, targets,
#: active_pair_count, t_multipoles [K, n, C], t_centers [K, n, 3], *, order,
#: k_lanes, interpret, backend, num_warps, n_targets, presorted, symmetric,
#: has_active_count, live[, tangents_per_program]) -> (t_locals [K, n_targets, C],)``
#: for any ``K`` (the kernel's tangent-group grid axis masks a partial last group).
#: ``active_pair_count``
#: is a dummy ``int32 [1]`` when ``has_active_count`` is False (the seam's ``None``).
#: The reference is :func:`m2l_real_csr_lanes_pallas_tangent_jnp`.
m2l_real_csr_lanes_pallas_tangent = make_tangent_op(
    "m2l_real_csr_lanes",
    _m2l_tangent_impl,
    num_primals=5,
    abstract_eval=_m2l_tangent_abstract,
)


@functools.lru_cache(maxsize=None)
def _packed_keys(order: int) -> tuple:
    """The ``(degree, m)`` key of each packed coefficient index.

    Parameters
    ----------
    order : int
        Expansion order.

    Returns
    -------
    tuple
        ``keys[i]`` is the key stored at packed index ``i``.
    """
    packed = _lane_tables(int(order))["packed"]
    by_index = {i: key for key, i in packed.items()}
    return tuple(by_index[i] for i in range(len(by_index)))


def _m2l_lane_pair_packed(coeffs: Array, delta: Array, *, order: int) -> Array:
    """:func:`_m2l_lane_pair` (guarded) on packed rows: ``[P, C], [P, 3] -> [P, C]``.

    Parameters
    ----------
    coeffs : Array
        ``[P, C]`` source multipoles.
    delta : Array
        ``[P, 3]`` target minus source centres.
    order : int
        Expansion order.

    Returns
    -------
    Array
        ``[P, C]`` local contributions, the lane kernel's arithmetic per pair.
    """
    p = int(order)
    tables = _lane_tables(p)
    keys = _packed_keys(p)
    v = {key: coeffs[..., i] for i, key in enumerate(keys)}
    w = _m2l_lane_pair(
        v, delta[..., 0], delta[..., 1], delta[..., 2], p=p, tables=tables, safe=True
    )
    return jnp.stack([w[key] for key in keys], axis=-1)


def _m2l_lane_pair_packed_with_axis(
    coeffs: Array, delta: Array, *, order: int
) -> Array:
    """:func:`_m2l_lane_pair_packed` with the analytic transverse derivative on the axis.

    Parameters
    ----------
    coeffs : Array
        ``[P, C]`` source multipoles.
    delta : Array
        ``[P, 3]`` target minus source centres.
    order : int
        Expansion order.

    Returns
    -------
    Array
        ``[P, C]``; its ``jax.jvp`` carries formula (2)/(3) of
        :mod:`jaccpot.operators._transverse_degeneracy_jvp` in the band.
    """
    from jaccpot.operators._transverse_degeneracy_jvp import (
        with_transverse_degeneracy_jvp,
    )
    from jaccpot.operators.real_rotations import _M2L_TRANSVERSE_GENERATORS

    rule = with_transverse_degeneracy_jvp(
        _m2l_lane_pair_packed, generators=_M2L_TRANSVERSE_GENERATORS
    )
    return rule(coeffs, delta, order=int(order))


def _m2l_tangent_twin(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    active_pair_count: Array,
    *,
    order: int,
    n_targets: Optional[int],
    presorted: bool,
    has_active_count: bool,
) -> Callable[[Array, Array], Array]:
    """The pure-JAX twin of the lane kernel, ``(multipoles, centers) -> locals``.

    The forward's CSR by target, :func:`_m2l_lane_pair_packed_with_axis` on every
    entry (padding entries at ``delta = (0, 0, 1)``, masked), a ``segment_sum`` by
    target row.

    Parameters
    ----------
    multipoles : Array
        ``[n, C]``: fixes ``n`` and the dtype.
    centers : Array
        ``[n, 3]``.
    sources : Array
        ``[P]``.
    targets : Array
        ``[P]`` (or row offsets with ``presorted``).
    active_pair_count : Array
        Live prefix length, or a dummy.
    order : int
        Expansion order.
    n_targets : Optional[int]
        Target rows (``None``: ``n``).
    presorted : bool
        CSR-ordered list with row offsets.
    has_active_count : bool
        ``active_pair_count`` is real.

    Returns
    -------
    Callable[[Array, Array], Array]
        ``f(multipoles, centers) -> [n_targets, C]``.
    """
    del centers
    p = int(order)
    dtype = jnp.asarray(multipoles).dtype
    n = int(multipoles.shape[0])
    C = int(multipoles.shape[1])
    nt = n if n_targets is None else int(n_targets)
    src_sorted, offsets, counts = _m2l_tangent_csr(
        sources,
        targets,
        active_pair_count,
        nt=nt,
        presorted=presorted,
        has_active_count=has_active_count,
    )
    num = int(src_sorted.shape[0])
    if nt == 0 or num == 0:
        return lambda m, c: jnp.zeros((nt, C), dtype)

    entry = jnp.arange(num, dtype=offsets.dtype)
    # rows are consecutive and ordered: an entry's row is the last one starting at
    # or before it, if the entry lies inside that row
    row = jnp.searchsorted(offsets, entry, side="right").astype(offsets.dtype) - 1
    row_safe = jnp.clip(row, 0, nt - 1)
    valid = (row >= 0) & (entry < offsets[row_safe] + counts[row_safe])
    src = jnp.where(valid, src_sorted, 0)
    tgt = jnp.where(valid, row_safe, 0)
    unit_z = jnp.asarray([0.0, 0.0, 1.0], dtype)

    def primal(m: Array, c: Array) -> Array:
        delta = jnp.where(valid[:, None], c[tgt] - c[src], unit_z)
        contrib = _m2l_lane_pair_packed_with_axis(m[src], delta, order=p)
        contrib = jnp.where(valid[:, None], contrib, 0.0)
        return jax.ops.segment_sum(contrib, tgt, num_segments=nt)

    return primal


def m2l_real_csr_lanes_pallas_tangent_jnp(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    active_pair_count: Array,
    t_multipoles: Array,
    t_centers: Array,
    *,
    order: int,
    k_lanes: int = 32,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 1,
    n_targets: Optional[int] = None,
    presorted: bool = False,
    symmetric: bool = False,
    has_active_count: bool = True,
    live: tuple[bool, ...] = (True, True),
) -> tuple[Array]:
    """Pure-JAX reference of :data:`m2l_real_csr_lanes_pallas_tangent` (same signature).

    ``jax.vmap`` over ``K`` of ``jax.jvp`` of a twin of the lane kernel: the same
    CSR by target (:func:`csr_by_target` with the forward's arguments), the lane
    kernel's pair arithmetic (:func:`_m2l_lane_pair`, guarded) on every entry, the
    analytic transverse derivative on the axis
    (:func:`~jaccpot.operators._transverse_degeneracy_jvp.with_transverse_degeneracy_jvp`
    with the real M2L generators), and a ``segment_sum`` by target. Its primal is
    the Pallas forward to round-off (another summation order). The parity
    reference of the tangent kernel and track A's fallback; it materialises
    ``[P, C]`` per tangent, so it is for tests and modest sizes.

    Parameters
    ----------
    multipoles : Array
        ``[n, C]``.
    centers : Array
        ``[n, 3]``.
    sources : Array
        ``[P]``.
    targets : Array
        ``[P]`` (or row offsets with ``presorted``).
    active_pair_count : Array
        Live prefix length, or a dummy.
    t_multipoles : Array
        ``[K, n, C]``.
    t_centers : Array
        ``[K, n, 3]``.
    order : int
        Expansion order.
    k_lanes : int
        Unused (the kernel's tile).
    interpret : bool
        Unused.
    backend : str
        Unused.
    num_warps : int
        Unused.
    n_targets : Optional[int]
        Target rows (``None``: ``n``).
    presorted : bool
        CSR-ordered list with row offsets.
    symmetric : bool
        Unused.
    has_active_count : bool
        ``active_pair_count`` is real.
    live : tuple[bool, ...]
        Unused (zero tangents give zero).

    Returns
    -------
    tuple[Array]
        ``(t_locals [K, n_targets, C],)``.
    """
    del k_lanes, interpret, backend, num_warps, symmetric, live
    mult = jnp.asarray(multipoles)
    dtype = mult.dtype
    primal = _m2l_tangent_twin(
        mult,
        centers,
        sources,
        targets,
        active_pair_count,
        order=order,
        n_targets=n_targets,
        presorted=presorted,
        has_active_count=has_active_count,
    )
    cent = jnp.asarray(centers, dtype)

    def one(tm: Array, tc: Array) -> Array:
        return jax.jvp(primal, (mult, cent), (tm.astype(dtype), tc.astype(dtype)))[1]

    return (jax.vmap(one)(t_multipoles, t_centers),)
