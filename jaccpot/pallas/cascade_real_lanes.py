"""Real M2M / L2L cascades with ONE NODE PER LANE (fused-memory round 2, Phase 5).

:mod:`jaccpot.pallas.cascade_real_level` runs one program per node of a level,
the node's ``(Bp, Wp)`` centred tile spread over the program's threads, every
program loading every constant table (``B``, ``B^-1`` and their transposes, the
one-hot shift selectors: ~40 KB at order 5) and the grid as wide as the widest
level for every level. Measured on an A100 at N = 8x10^6 (Plummer clipped, leaf
64, order 5): M2M 54 ms and L2L 44 ms per step, 40 % of the step, for ~10^7
translations whose arithmetic is ~1 ms -- the same disease the M2L had before
:mod:`jaccpot.pallas.m2l_real_csr_lanes` (data movement between tiny tiles).

The cure is that module's: a lane owns one node end to end. Its coefficients
are ``(K,)`` vectors per ``(degree, m)``, gathered straight from the PACKED
``[nodes, C]`` table (no centred ``[nodes, Bp*Wp]`` copy, 3.6x the packed bytes
at order 5); the translation -- ``Dz(az)``, ``B``, ``Dz(-ax)``, ``B``, the
z-shift core, ``B^-1``, ``Dz(ax)``, ``B^-1``, ``Dz(-az)`` for multipoles,
``B^-T`` in and ``B^T`` out for locals -- is straight-line code with the
constants baked in (:func:`~jaccpot.pallas.m2l_real_csr_lanes._bapply_lanes`,
``_dz_lanes``, ``_angle_powers``); the shift core's ``r^k / k!`` by repeated
multiplication. Each program covers ``K`` nodes of the level, so the grid is the
widest level over ``K``.

* M2M: one lane per PARENT; both children translated and summed, the parent's
  row stored once.
* L2L: one lane per node below the root; the parent's local translated and
  added to the node's own row.

Every lane stores only its own node's row (masked, out-of-range lanes), reading
rows of the level above or below, so the in-place update of the aliased table
is race-free, as for the level kernels. Forward only: the custom VJPs keep the
level kernels' reverse passes, which read the forward's output as a residual
(the two forwards agree to round-off -- different operation order).
"""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from jaccpot.pallas._compat import KernelRef, pallas_backend_kwargs
from jaccpot.pallas.m2l_real_csr_lanes import (
    _angle_powers,
    _bapply_lanes,
    _dz_lanes,
    _lane_tables,
)

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None
    plgpu = None

__all__ = [
    "l2l_real_levels_lanes_pallas",
    "m2m_real_levels_lanes_pallas",
]


@functools.lru_cache(maxsize=None)
def _cascade_lane_tables(order: int) -> dict:
    """:func:`_lane_tables` plus the per-degree inverse blocks and ``1/k!``.

    Parameters
    ----------
    order : int
        Expansion order.

    Returns
    -------
    dict
        ``B``, ``packed``, ``C``, ``p`` as :func:`_lane_tables`; ``Binv[l][i][j]``
        the inverse of degree ``l``'s block (the alignment ``T`` is not
        orthogonal); ``invfact[k] = 1/k!``.
    """
    from jaccpot.pallas.cascade_real_level import cascade_level_tables

    p = int(order)
    t = dict(_lane_tables(p))
    binv = np.asarray(cascade_level_tables(p)["Binv"])
    t["Binv"] = [
        [
            [float(x) for x in row]
            for row in binv[ell, p - ell : p + ell + 1][:, p - ell : p + ell + 1]
        ]
        for ell in range(p + 1)
    ]
    t["invfact"] = [1.0 / float(math.factorial(k)) for k in range(p + 1)]
    return t


def _translate_lane(
    v: dict, dx: Array, dy: Array, dz_: Array, *, which: str, p: int, tables: dict
) -> dict:
    """M2M (``which="m2m"``) or L2L of one node per lane: rotate, shift along z, rotate back.

    Parameters
    ----------
    v : dict
        Source coefficients per ``(degree, m)``, lane vectors ``(K,)``.
    dx : Array
        Source centre minus destination centre, ``x``; lanes.
    dy : Array
        Same, ``y``.
    dz_ : Array
        Same, ``z``.
    which : str
        ``"m2m"`` (child multipole -> parent) or ``"l2l"`` (parent local ->
        child). Static.
    p : int
        Order. Static.
    tables : dict
        :func:`_cascade_lane_tables`. Static.

    Returns
    -------
    dict
        Translated coefficients per key, lane vectors.
    """
    B, Binv, invfact = tables["B"], tables["Binv"], tables["invfact"]
    dtype = dx.dtype
    one = jnp.ones_like(dx)
    rho2 = dx * dx + dy * dy
    rho = jnp.sqrt(rho2)
    r = jnp.sqrt(rho2 + dz_ * dz_)
    # az = atan2(dx, dy), ax = atan2(rho, dz); both 0 where undefined (rho == 0,
    # r == 0), as atan2(0, 0) is -- a parent whose only child shares its centre
    # translates by the identity
    rho_ok = rho > 0.0
    rho_inv = jnp.where(rho_ok, 1.0 / jnp.where(rho_ok, rho, one), 0.0)
    r_ok = r > 0.0
    r_inv = jnp.where(r_ok, 1.0 / jnp.where(r_ok, r, one), 0.0)
    cos_az = jnp.where(rho_ok, dy * rho_inv, one)
    sin_az = dx * rho_inv
    cos_ax = jnp.where(r_ok, dz_ * r_inv, one)
    sin_ax = rho * r_inv
    cos_maz, sin_maz = _angle_powers(cos_az, sin_az, p)
    cos_max, sin_max = _angle_powers(cos_ax, sin_ax, p)
    # shift-core weights c_k = r^k / k! (c_0 = 1; r == 0 -> identity)
    r_pow = [one]
    for _ in range(p):
        r_pow.append(r_pow[-1] * r)
    c = [r_pow[k] * jnp.asarray(invfact[k], dtype) for k in range(p + 1)]
    if which == "m2m":
        b_in, t_in, b_out, t_out = B, False, Binv, False
    else:
        b_in, t_in, b_out, t_out = Binv, True, B, True
    # world -> z: B_in Dz(-ax) B_in Dz(az)
    v = _dz_lanes(v, cos_maz, sin_maz, sign=1.0, p=p)
    v = _bapply_lanes(v, b_in, transpose=t_in, p=p)
    v = _dz_lanes(v, cos_max, sin_max, sign=-1.0, p=p)
    v = _bapply_lanes(v, b_in, transpose=t_in, p=p)
    # z-shift core, per column m: M2M out[n] = sum_{j <= n} c_{n-j} v[j];
    # L2L out[n] = sum_{j >= n} c_{j-n} v[j] (v[j, m] exists for j >= |m|)
    w = {}
    for n in range(p + 1):
        for m in range(-n, n + 1):
            js = range(abs(m), n + 1) if which == "m2m" else range(n, p + 1)
            acc = None
            for j in js:
                k = n - j if which == "m2m" else j - n
                term = v[(j, m)] if k == 0 else v[(j, m)] * c[k]
                acc = term if acc is None else acc + term
            w[(n, m)] = acc
    # z -> world: Dz(-az) B_out Dz(ax) B_out
    w = _bapply_lanes(w, b_out, transpose=t_out, p=p)
    w = _dz_lanes(w, cos_max, sin_max, sign=1.0, p=p)
    w = _bapply_lanes(w, b_out, transpose=t_out, p=p)
    return _dz_lanes(w, cos_maz, sin_maz, sign=-1.0, p=p)


def _flat(node: Array, C: int, size: int) -> Array:
    """``node * C``, in int64 when the flat table outgrows int32 offsets (static).

    Parameters
    ----------
    node : Array
        Node ids.
    C : int
        Coefficients per node.
    size : int
        Length of the flat table. Static.

    Returns
    -------
    Array
        The row offsets.
    """
    if int(size) + int(C) * 64 >= 2**31 - 1:
        node = node.astype(jnp.int64)
    return node * C


def _keys(p: int) -> list:
    return [(ell, m) for ell in range(p + 1) for m in range(-ell, ell + 1)]


def _store_rows(
    out_ref: KernelRef,
    node: Array,
    valid: Array,
    vals: dict,
    *,
    p: int,
    C: int,
    packed: dict,
    lane: Array,
) -> None:
    # masked lanes point PAST the table (unique, out of range): Triton never
    # touches them and the interpreter drops them
    n_rows = out_ref.shape[0] // C
    base = _flat(jnp.where(valid, node, n_rows + lane), C, out_ref.shape[0])
    for key in _keys(p):
        plgpu.store(out_ref.at[base + packed[key]], vals[key], mask=valid)


def _m2m_lanes_kernel(
    coef_ref: KernelRef,
    cent_ref: KernelRef,
    left_ref: KernelRef,
    right_ref: KernelRef,
    nbl_ref: KernelRef,
    start_ref: KernelRef,
    count_ref: KernelRef,
    out_ref: KernelRef,
    *,
    p: int,
    k_lanes: int,
    tables: dict,
    num_internal: int,
) -> None:
    """``K`` parents of one level per program: each lane sums its children's M2M.

    Parameters
    ----------
    coef_ref : KernelRef
        Flat packed multipoles ``[nodes * C]`` (aliased to ``out_ref``).
    cent_ref : KernelRef
        Padded centres ``[nodes, 4]``.
    left_ref : KernelRef
        Left child per internal node ``[internal]``.
    right_ref : KernelRef
        Right child per internal node ``[internal]``.
    nbl_ref : KernelRef
        ``nodes_by_level`` padded with ``-1``.
    start_ref : KernelRef
        This level's start in ``nodes_by_level`` ``[1]``.
    count_ref : KernelRef
        This level's node count ``[1]``.
    out_ref : KernelRef
        The flat table, written at this level's parents.
    p : int
        Order. Static.
    k_lanes : int
        Lanes (nodes) per program. Static.
    tables : dict
        :func:`_cascade_lane_tables`. Static.
    num_internal : int
        Internal node count. Static.

    Returns
    -------
    None
        Writes the parents' rows.
    """
    C, packed = tables["C"], tables["packed"]
    base = pl.program_id(0) * k_lanes
    count = count_ref[0]

    @pl.when(base < count)
    def _live():
        lane = lax.broadcasted_iota(jnp.int32, (k_lanes,), 0)
        slot = base + lane
        valid = slot < count
        node = nbl_ref[start_ref[0] + jnp.where(valid, slot, 0)]
        valid = valid & (node >= 0) & (node < num_internal)
        node_safe = jnp.where(valid, node, 0)
        px = cent_ref[node_safe, 0]
        py = cent_ref[node_safe, 1]
        pz = cent_ref[node_safe, 2]
        acc = None
        for child_ref in (left_ref, right_ref):
            ch = child_ref[node_safe]
            ok = valid & (ch >= 0)
            ch = jnp.where(ok, ch, 0)
            dx = jnp.where(ok, cent_ref[ch, 0] - px, 0.0)
            dy = jnp.where(ok, cent_ref[ch, 1] - py, 0.0)
            dz_ = jnp.where(ok, cent_ref[ch, 2] - pz, 1.0)
            row = _flat(ch, C, coef_ref.shape[0])
            v = {key: coef_ref[row + packed[key]] for key in _keys(p)}
            w = _translate_lane(v, dx, dy, dz_, which="m2m", p=p, tables=tables)
            w = {key: jnp.where(ok, val, 0.0) for key, val in w.items()}
            acc = w if acc is None else {key: acc[key] + w[key] for key in w}
        _store_rows(out_ref, node_safe, valid, acc, p=p, C=C, packed=packed, lane=lane)


def _l2l_lanes_kernel(
    coef_ref: KernelRef,
    cent_ref: KernelRef,
    parent_ref: KernelRef,
    nbl_ref: KernelRef,
    start_ref: KernelRef,
    count_ref: KernelRef,
    out_ref: KernelRef,
    *,
    p: int,
    k_lanes: int,
    tables: dict,
) -> None:
    """``K`` nodes of one level per program: each lane adds its parent's L2L.

    Parameters
    ----------
    coef_ref : KernelRef
        Flat packed locals ``[nodes * C]`` (aliased to ``out_ref``).
    cent_ref : KernelRef
        Padded centres ``[nodes, 4]``.
    parent_ref : KernelRef
        Parent per node ``[nodes]`` (``-1`` at the root).
    nbl_ref : KernelRef
        ``nodes_by_level`` padded with ``-1``.
    start_ref : KernelRef
        This level's start ``[1]``.
    count_ref : KernelRef
        This level's node count ``[1]``.
    out_ref : KernelRef
        The flat table, written at this level's nodes.
    p : int
        Order. Static.
    k_lanes : int
        Lanes (nodes) per program. Static.
    tables : dict
        :func:`_cascade_lane_tables`. Static.

    Returns
    -------
    None
        Writes the nodes' rows.
    """
    C, packed = tables["C"], tables["packed"]
    base = pl.program_id(0) * k_lanes
    count = count_ref[0]

    @pl.when(base < count)
    def _live():
        lane = lax.broadcasted_iota(jnp.int32, (k_lanes,), 0)
        slot = base + lane
        valid = slot < count
        node = nbl_ref[start_ref[0] + jnp.where(valid, slot, 0)]
        valid = valid & (node >= 0)
        node_safe = jnp.where(valid, node, 0)
        par = parent_ref[node_safe]
        valid = valid & (par >= 0)
        par = jnp.where(valid, par, 0)
        dx = jnp.where(valid, cent_ref[par, 0] - cent_ref[node_safe, 0], 0.0)
        dy = jnp.where(valid, cent_ref[par, 1] - cent_ref[node_safe, 1], 0.0)
        dz_ = jnp.where(valid, cent_ref[par, 2] - cent_ref[node_safe, 2], 1.0)
        prow = _flat(par, C, coef_ref.shape[0])
        v = {key: coef_ref[prow + packed[key]] for key in _keys(p)}
        w = _translate_lane(v, dx, dy, dz_, which="l2l", p=p, tables=tables)
        nrow = _flat(node_safe, C, coef_ref.shape[0])
        new = {key: coef_ref[nrow + packed[key]] + w[key] for key in _keys(p)}
        _store_rows(out_ref, node_safe, valid, new, p=p, C=C, packed=packed, lane=lane)


def _full(arr: Array) -> Any:
    shp = tuple(arr.shape)
    return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))


def _lanes_level_call(
    kernel: Any,
    operands: list,
    *,
    num_programs: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    name: str,
) -> Array:
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )
    flat = operands[0]
    return pl.pallas_call(
        kernel,
        grid=(int(num_programs),),
        in_specs=[_full(a) for a in operands],
        out_specs=_full(flat),
        out_shape=jax.ShapeDtypeStruct(flat.shape, flat.dtype),
        input_output_aliases={0: 0},
        interpret=bool(interpret),
        name=name,
        **backend_kwargs,
    )(*operands)


def _check(order: int, coeffs: Array) -> int:
    if pl is None or plgpu is None:
        raise RuntimeError("jax.experimental.pallas is not available")
    C = (int(order) + 1) ** 2
    if coeffs.ndim != 2 or int(coeffs.shape[1]) != C:
        raise ValueError(f"coefficients must be [nodes, {C}] for order {order}")
    return C


def m2m_real_levels_lanes_pallas(
    packed: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    *,
    order: int,
    num_internal: int,
    num_levels: int,
    level_batch_width: int,
    k_lanes: int = 32,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 1,
) -> Array:
    """Upward M2M cascade, one node per lane: drop-in for ``m2m_real_levels_pallas``.

    Parameters
    ----------
    packed : Array
        ``[nodes, C]`` packed multipoles with the leaves filled (P2M done).
    centers : Array
        ``[nodes, 3]`` expansion centres.
    left_child : Array
        ``[internal]`` left children.
    right_child : Array
        ``[internal]`` right children.
    nodes_by_level : Array
        Level-major node ids.
    level_offsets : Array
        ``[levels + 1]`` starts into ``nodes_by_level``.
    order : int
        Expansion order. Static.
    num_internal : int
        Internal node count. Static.
    num_levels : int
        Levels the loop covers (the deepest ``num_levels - 1`` internal levels). Static.
    level_batch_width : int
        Nodes per level at most (>= the widest level). Static.
    k_lanes : int
        Nodes per program. Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program (``k_lanes / 32``).

    Returns
    -------
    Array
        ``[nodes, C]`` packed multipoles with the internal nodes filled.
    """
    p = int(order)
    C = _check(p, packed)
    if int(num_internal) <= 0:
        return packed
    tables = _cascade_lane_tables(p)
    dtype = packed.dtype
    nodes = int(packed.shape[0])
    k = max(1, int(k_lanes))
    width = int(max(level_batch_width, 1))
    idx = level_offsets.dtype
    nbl = jnp.concatenate(
        [jnp.asarray(nodes_by_level, idx), jnp.full((width + k,), -1, idx)]
    )
    offs = jnp.asarray(level_offsets, idx)
    cent = jnp.pad(jnp.asarray(centers, dtype), ((0, 0), (0, 1)))
    kernel = functools.partial(
        _m2m_lanes_kernel, p=p, k_lanes=k, tables=tables, num_internal=int(num_internal)
    )
    left = jnp.asarray(left_child, idx)
    right = jnp.asarray(right_child, idx)

    def body(rev: Array, flat: Array) -> Array:
        level = (int(num_levels) - 2) - rev
        return _lanes_level_call(
            kernel,
            [
                flat,
                cent,
                left,
                right,
                nbl,
                offs[level][None],
                (offs[level + 1] - offs[level])[None],
            ],
            num_programs=-(-width // k),
            interpret=interpret,
            backend=backend,
            num_warps=num_warps,
            name=f"m2m_real_lanes_p{p}_k{k}",
        )

    flat = lax.fori_loop(
        0, max(int(num_levels) - 1, 0), body, packed.reshape(-1), unroll=True
    )
    return flat.reshape(nodes, C)


def l2l_real_levels_lanes_pallas(
    coeffs_local: Array,
    centers: Array,
    parent: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    *,
    order: int,
    num_levels: int,
    level_batch_width: int,
    k_lanes: int = 32,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 1,
) -> Array:
    """Downward L2L cascade, one node per lane: drop-in for ``l2l_real_levels_pallas``.

    Parameters
    ----------
    coeffs_local : Array
        ``[nodes, C]`` packed locals after M2L.
    centers : Array
        ``[nodes, 3]`` expansion centres.
    parent : Array
        ``[nodes]`` parent per node (``-1`` at the root).
    nodes_by_level : Array
        Level-major node ids.
    level_offsets : Array
        ``[levels + 1]`` starts into ``nodes_by_level``.
    order : int
        Expansion order. Static.
    num_levels : int
        Levels present (``max level + 1``). Static.
    level_batch_width : int
        Nodes per level at most. Static.
    k_lanes : int
        Nodes per program. Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.

    Returns
    -------
    Array
        ``[nodes, C]`` packed locals with every ancestor's field cascaded down.
    """
    p = int(order)
    C = _check(p, coeffs_local)
    tables = _cascade_lane_tables(p)
    dtype = coeffs_local.dtype
    nodes = int(coeffs_local.shape[0])
    k = max(1, int(k_lanes))
    width = int(max(level_batch_width, 1))
    idx = level_offsets.dtype
    nbl = jnp.concatenate(
        [jnp.asarray(nodes_by_level, idx), jnp.full((width + k,), -1, idx)]
    )
    offs = jnp.asarray(level_offsets, idx)
    cent = jnp.pad(jnp.asarray(centers, dtype), ((0, 0), (0, 1)))
    par = jnp.asarray(parent, idx)
    kernel = functools.partial(_l2l_lanes_kernel, p=p, k_lanes=k, tables=tables)

    def body(level: Array, flat: Array) -> Array:
        return _lanes_level_call(
            kernel,
            [
                flat,
                cent,
                par,
                nbl,
                offs[level][None],
                (offs[level + 1] - offs[level])[None],
            ],
            num_programs=-(-width // k),
            interpret=interpret,
            backend=backend,
            num_warps=num_warps,
            name=f"l2l_real_lanes_p{p}_k{k}",
        )

    # level 0 is the root (nothing above it); levels 1 .. num_levels-1 receive
    flat = lax.fori_loop(
        1, max(int(num_levels), 1), body, coeffs_local.reshape(-1), unroll=True
    )
    return flat.reshape(nodes, C)
