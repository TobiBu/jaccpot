"""Real-basis M2M and L2L cascades: the custom VJPs and the pure-jnp twins (plan sub-10ms, Phase 3).

The level loops of the upward (``aggregate_m2m_real_by_level``) and downward
(``_propagate_solidfmm_locals_by_level``) sweeps translate every node of a
level with the vectorised rotation operators, which XLA breaks into dozens of
small fusions per level: ~0.8 ms per level, 66 ms for the 40-level radix tree
over Morton-cell leaves at N = 2x10^5 (11 ms on the 13-level bucket tree).

The forward of both cascades is the one-node-per-lane kernel of
:mod:`jaccpot.pallas.cascade_real_lanes`, reached through the custom VJPs at the
end of this module (:func:`m2m_real_levels_pallas_cvjp`,
:func:`l2l_real_levels_pallas_cvjp`). The one-program-per-node level forward
kernels this module started with were removed in the 2026-10 cleanup (X5); the
lane kernel replaced them as the default in fused-memory round 2.

What stays here: the custom VJPs, the pure-jnp pair twins and the shared
tables. The reverses are the one-node-per-lane kernels of
:mod:`jaccpot.pallas.cascade_real_lanes` as well (plan S8); the pair twins
(:func:`_translate_rows`) rotate a node's centred rows onto the pair axis,
apply the shift core along z and rotate back. The rotation halves are exactly
the M2L CSR twin's (:mod:`jaccpot.pallas.m2l_real_csr`) with one block swapped:
the alignment ``T`` is not orthogonal, so multipoles come back with ``T^-1``
(per-degree inverse blocks, a constant table) and locals go in with ``T^-T``
and come back with ``T^T`` (``real_rotations.py``); the cores differ too:

* M2M (child multipole -> parent, ``delta = child - parent``):
  ``out[n, m] = sum_{j <= n} dz^(n-j)/(n-j)! * in[j, m]`` -- lower triangular in
  the degree, per column ``m`` (the centred layout's zero padding removes the
  ``|m| <= j`` condition);
* L2L (parent local -> child, ``delta = parent - child``):
  ``out[n, m] = sum_{j >= n} dz^(j-n)/(j-n)! * in[j, m]`` -- upper triangular;

with ``dz = |delta|`` (the source sits at ``+z`` after the alignment), the
same convention as :func:`jaccpot.operators.real_translations.m2m_real` and
``l2l_real``. Results agree with those operators to round-off, not bit for
bit (different operation order); the interpret tests pin that.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaccpot.operators.real_rotations import _real_rotation_generator_block
from jaccpot.pallas.m2l_real_csr import (
    _bapply,
    _dz,
    _tables_to_jnp,
    m2l_real_csr_tables,
    pack_centred,
    pallas_m2l_real_csr_supported,
    unpack_centred,
)

__all__ = [
    "cascade_level_tables",
    "l2l_real_centred_pair_jax",
    "l2l_real_levels_fb_pallas_cvjp",
    "l2l_real_levels_pallas_cvjp",
    "l2l_real_levels_reverse_pallas",
    "m2m_real_centred_pair_jax",
    "m2m_real_levels_pallas_cvjp",
    "m2m_real_levels_reverse_pallas",
    "pallas_cascade_level_supported",
]


def pallas_cascade_level_supported() -> bool:
    """True where the Triton lowering of these kernels runs (sm_80+, as the CSR kernel).

    Returns
    -------
    bool
        Whether the kernels can run natively here.
    """
    return pallas_m2l_real_csr_supported()


@functools.lru_cache(maxsize=None)
def cascade_level_tables(order: int) -> dict:
    """Shift-core constants on top of :func:`m2l_real_csr_tables`.

    Parameters
    ----------
    order : int
        Expansion order ``p``.

    Returns
    -------
    dict
        The CSR tables plus ``Km2m`` / ``Kl2l`` ``[Bp, Bp]`` (exponent of ``dz``
        for output degree ``n`` from source degree ``j``, ``-1`` where the term
        does not exist) and ``invfact [Bp]`` (``1/k!``).
    """
    t = dict(m2l_real_csr_tables(int(order)))
    p = int(order)
    Bp = t["Bp"]
    Km2m = np.full((Bp, Bp), -1, dtype=np.int32)
    Kl2l = np.full((Bp, Bp), -1, dtype=np.int32)
    for n in range(p + 1):
        for j in range(p + 1):
            if j <= n:
                Km2m[n, j] = n - j
            if j >= n:
                Kl2l[n, j] = j - n
    invfact = np.zeros((Bp,), dtype=np.float64)
    for k in range(p + 1):
        invfact[k] = 1.0 / float(math.factorial(k))
    # The alignment rotation T = B Dz(-ax) B Dz(az) is NOT orthogonal (its inverse
    # is not its transpose), so the multipole z->world rotation F = T^-1 needs the
    # per-degree inverse blocks: F = Dz(-az) B^-1 Dz(ax) B^-1, and the local
    # world->z rotation is F^T = B^-T Dz(-ax) B^-T Dz(az).
    Bstack = np.asarray(t["Bstack"], dtype=np.float64)
    Binv = np.zeros_like(Bstack)
    for ell in range(p + 1):
        lo, hi = p - ell, p + ell + 1
        Binv[ell, lo:hi, lo:hi] = np.linalg.inv(Bstack[ell, lo:hi, lo:hi])
    # one-hot selectors: Sm2m[k, n, j] = 1 where the (n, j) term uses dz^k / k!
    # (Triton lowers no gather, so the core is a sum over k of table x scalar)
    Sm2m = np.zeros((Bp, Bp, Bp), dtype=np.float64)
    Sl2l = np.zeros((Bp, Bp, Bp), dtype=np.float64)
    for n in range(Bp):
        for j in range(Bp):
            if Km2m[n, j] >= 0:
                Sm2m[Km2m[n, j], n, j] = 1.0
            if Kl2l[n, j] >= 0:
                Sl2l[Kl2l[n, j], n, j] = 1.0
    t["Km2m"] = Km2m
    t["Kl2l"] = Kl2l
    t["Sm2m"] = Sm2m
    t["Sl2l"] = Sl2l
    t["invfact"] = invfact
    t["kpow"] = np.arange(Bp, dtype=np.float64)
    t["Binv"] = Binv
    t["BinvT"] = np.swapaxes(Binv, -1, -2).copy()
    # The rotation generators about x and y per degree, centred like Bstack: the
    # reverse kernels' on-axis edge cotangent (`_write_on_axis_edge_cotangent`).
    # Taken from the block builder the pure-JAX operators' rule uses, so the two
    # lanes share one calibration of every sign.
    for key, axis, representation in _AXIS_GENERATORS:
        stack = np.zeros((Bp, t["Wp"], t["Wp"]), dtype=np.float64)
        for ell in range(p + 1):
            stack[ell, p - ell : p + ell + 1, p - ell : p + ell + 1] = (
                _real_rotation_generator_block(ell, axis, representation)
            )
        t[key] = stack
    return t


_CORE_KEYS = ("Sm2m", "Sl2l", "invfact", "kpow", "Binv", "BinvT")

# (table key, rotation axis, representation): the rotation generators of formula
# (2)/(3) in `operators/_transverse_degeneracy_jvp`, part of `cascade_level_tables`
# (the level reverse kernels' on-axis rule read them until plan S8).
_AXIS_GENERATORS = (
    ("GmpX", "x", "multipole"),
    ("GmpY", "y", "multipole"),
    ("GlocX", "x", "local"),
    ("GlocY", "y", "local"),
)


def _core_tables_to_jnp(order: int, dtype: Any) -> dict[str, Array]:
    t = cascade_level_tables(int(order))
    out = _tables_to_jnp(int(order), dtype)
    for k in (*_CORE_KEYS, *(key for key, _, _ in _AXIS_GENERATORS)):
        out[k] = jnp.asarray(t[k], dtype=dtype)
    return out


def _shift_core(v: Array, r: Array, sel: Array, invfact: Array, kpow: Array) -> Array:
    """Apply the z-shift core ``out[n, :] = sum_j S[n, j] v[j, :]``, ``S = sum_k c_k sel[k]``.

    Parameters
    ----------
    v : Array
        Centred rows in the z-aligned frame, ``(Bp, Wp)``.
    r : Array
        Shift distance (scalar).
    sel : Array
        One-hot selector ``(Bp, Bp, Bp)``: ``sel[k, n, j] = 1`` where the
        ``(n, j)`` term carries ``c_k = r^k / k!``.
    invfact : Array
        ``1/k!`` per exponent ``(Bp,)``.
    kpow : Array
        ``k`` as a float per slot ``(Bp,)``.

    Returns
    -------
    Array
        ``(Bp, Wp)``.
    """
    dtype = v.dtype
    # c_k = r^k / k!; r == 0 -> identity (c_0 = 1, others 0)
    safe_r = jnp.maximum(r, jnp.asarray(1.0e-30, dtype=dtype))
    c = jnp.exp(kpow * jnp.log(safe_r)) * invfact
    c = jnp.where(kpow == 0, jnp.asarray(1.0, dtype=dtype), jnp.where(r > 0, c, 0.0))
    S = jnp.sum(sel * c[:, None, None], axis=0)  # (Bp, Bp)
    return jnp.sum(S[:, :, None] * v[None, :, :], axis=1)


def _translate_rows(
    rows: Array, delta3: tuple, t: dict[str, Array], which: str, *, safe: bool = False
) -> Array:
    """Rotate onto the pair axis, shift along z, rotate back (centred layout).

    Multipoles (``which="m2m"``) rotate with ``T`` in and ``T^-1`` out; locals
    (``which="l2l"``) with ``T^-T`` in and ``T^T`` out, so each half is the
    CSR kernel's half with ``B`` replaced by the matching block (``B``,
    ``B^-1``, ``B^-T`` or ``B^T``).

    Parameters
    ----------
    rows : Array
        Source coefficients in centred rows, ``(Bp, Wp)``.
    delta3 : tuple
        ``(x, y, z)`` scalars, source centre minus destination centre.
    t : dict[str, Array]
        Tables from :func:`_core_tables_to_jnp`.
    which : str
        ``"m2m"`` or ``"l2l"``.
    safe : bool
        Compute the alignment angles and radii through NaN-safe double-``where``
        guards (:mod:`jaccpot.operators.real_rotations` does the same). The
        primal is bit-identical; only the derivative changes, and only where the
        unguarded form is singular: ``sqrt`` and ``arctan2`` at ``rho == 0`` or
        ``r == 0`` give ``0 * inf`` cotangents, which is a NaN. The reverse
        kernels set this; the pair twins keep the unguarded trace.

    Returns
    -------
    Array
        Translated coefficients at the destination, ``(Bp, Wp)``.
    """
    if which == "m2m":
        b_in, b_out, sel = t["Bstack"], t["Binv"], t["Sm2m"]
    else:
        b_in, b_out, sel = t["BinvT"], t["BstackT"], t["Sl2l"]
    x, y, z = delta3
    dtype = rows.dtype
    rho2 = x * x + y * y
    if safe:
        # Every branch selects the same VALUE the unguarded form gives
        # (arctan2(0, 0) = 0, sqrt(0) = 0); the guards only keep the transpose
        # finite. On the axis the transverse cotangent comes out zero -- the
        # polar route cannot resolve it -- and the reverse kernels overwrite it
        # with the analytic limit the pure-JAX operators use
        # (`_write_on_axis_edge_cotangent`); at delta == 0 the two centres are
        # the same function of the positions and the cotangent cancels whatever
        # finite value it takes.
        one = jnp.asarray(1.0, dtype=dtype)
        zero = jnp.asarray(0.0, dtype=dtype)
        rho_pos = rho2 > 0
        rho = jnp.where(rho_pos, jnp.sqrt(jnp.where(rho_pos, rho2, one)), zero)
        az = jnp.where(rho_pos, jnp.arctan2(jnp.where(rho_pos, x, one), y), zero)
        r2 = rho2 + z * z
        r_pos = r2 > 0
        r = jnp.where(r_pos, jnp.sqrt(jnp.where(r_pos, r2, one)), zero)
        ax = jnp.where(r_pos, jnp.arctan2(rho, jnp.where(r_pos, z, one)), zero)
    else:
        rho = jnp.sqrt(rho2)
        az = jnp.arctan2(x, y)
        ax = jnp.arctan2(rho, z)
        r = jnp.sqrt(rho2 + z * z)
    mabs = t["mabs"]
    cos_az = jnp.cos(mabs * az)
    sin_az = jnp.sin(mabs * az)
    cos_ax = jnp.cos(mabs * ax)
    sin_ax = jnp.sin(mabs * ax)
    apat = t["Apat"]
    # world -> z
    v = _dz(rows, cos_az, sin_az, apat)
    v = _bapply(b_in, v)
    v = _dz(v, cos_ax, -sin_ax, apat)
    v = _bapply(b_in, v)
    v = _shift_core(v, r, sel, t["invfact"], t["kpow"])
    # z -> world
    v = _bapply(b_out, v)
    v = _dz(v, cos_ax, sin_ax, apat)
    v = _bapply(b_out, v)
    return _dz(v, cos_az, -sin_az, apat)


# ----------------------------------------------------------------- pure-jnp twins


def m2m_real_centred_pair_jax(
    child_packed: Array, delta: Array, *, order: int
) -> Array:
    """M2M of one packed child expansion to a parent at ``-delta`` (twin of :func:`m2m_real`).

    Parameters
    ----------
    child_packed : Array
        ``(C,)`` packed real multipole of the child.
    delta : Array
        ``(3,)`` child centre minus parent centre.
    order : int
        Expansion order.

    Returns
    -------
    Array
        ``(C,)`` packed multipole at the parent.
    """
    t = _core_tables_to_jnp(int(order), child_packed.dtype)
    rows = pack_centred(child_packed[None, :], order=int(order))[0].reshape(
        t["invfact"].shape[0], -1
    )
    out = _translate_rows(rows, (delta[0], delta[1], delta[2]), t, "m2m")
    return unpack_centred(out.reshape(1, -1), order=int(order))[0]


def l2l_real_centred_pair_jax(
    parent_packed: Array, delta: Array, *, order: int
) -> Array:
    """L2L of one packed parent local to a child at ``-delta`` (twin of :func:`l2l_real`).

    Parameters
    ----------
    parent_packed : Array
        ``(C,)`` packed real local of the parent.
    delta : Array
        ``(3,)`` parent centre minus child centre.
    order : int
        Expansion order.

    Returns
    -------
    Array
        ``(C,)`` packed local at the child.
    """
    t = _core_tables_to_jnp(int(order), parent_packed.dtype)
    rows = pack_centred(parent_packed[None, :], order=int(order))[0].reshape(
        t["invfact"].shape[0], -1
    )
    out = _translate_rows(rows, (delta[0], delta[1], delta[2]), t, "l2l")
    return unpack_centred(out.reshape(1, -1), order=int(order))[0]


# ------------------------------------------------------------- reverse
# The adjoints of both cascades are the one-node-per-lane reverse kernels of
# :mod:`jaccpot.pallas.cascade_real_lanes` (plan S8): the packed ``[nodes, C]``
# table, ``K`` nodes per program, the geometry cotangent from the translation
# generator (exact on the axis and at a zero edge). The one-program-per-node level
# reverse they replaced -- centred ``(Bp, Wp)`` tiles (``[nodes, Bp*Wp]`` rows,
# 3.6x the packed bytes at order 5, 33.5 floats per particle at 8e6), the grid
# as wide as the widest level for every level, an analytic on-axis correction --
# was 82 % of the far field's gradient at 8e6 (487 of 595 ms of kernel time).


def _fold_edge_cotangents(
    dbar: Array, parent: Array, *, sign: float, dtype: Any
) -> Array:
    """Centre cotangents from per-node edge cotangents.

    ``dbar[c]`` is the cotangent of the edge vector of node ``c``; ``sign = +1``
    for ``delta = child - parent`` (M2M), ``-1`` for ``parent - child`` (L2L).

    Parameters
    ----------
    dbar : Array
        ``[nodes, 3]`` edge cotangents (zero at the root).
    parent : Array
        ``[nodes]`` parent per node.
    sign : float
        Orientation of the edge vector.
    dtype : Any
        Output dtype.

    Returns
    -------
    Array
        ``[nodes, 3]`` centre cotangents.
    """
    par_ok = parent >= 0
    par_safe = jnp.where(par_ok, parent, 0)
    onto_parent = (
        jnp.zeros_like(dbar).at[par_safe].add(jnp.where(par_ok[:, None], dbar, 0.0))
    )
    return (sign * (dbar - onto_parent)).astype(dtype)


def m2m_real_levels_reverse_pallas(
    out_packed: Array,
    centers: Array,
    parent: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    out_bar: Array,
    *,
    order: int,
    num_internal: int,
    num_levels: int,
    level_batch_width: int,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 4,
) -> tuple[Array, Array]:
    """Adjoint of the upward M2M cascade: one launch per level, top down, a node per lane.

    Parameters
    ----------
    out_packed : Array
        ``[nodes, C]`` the forward's OUTPUT (every node's final multipole).
    centers : Array
        ``[nodes, 3]`` expansion centres.
    parent : Array
        ``[nodes]`` parent per node (``-1`` at the root).
    nodes_by_level : Array
        Level-major node ids (all nodes, leaves included).
    level_offsets : Array
        ``[levels + 1]`` starts into ``nodes_by_level``.
    out_bar : Array
        ``[nodes, C]`` cotangent of the output table.
    order : int
        Expansion order. Static.
    num_internal : int
        Internal node count. Static.
    num_levels : int
        Levels the forward covered. Static.
    level_batch_width : int
        Nodes per level at most. Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Not read: the lane kernels run ``k / 32`` warps, ``k`` the forward's
        ``JACCPOT_CASCADE_LANES_K``. Kept for the custom VJP's signature.

    Returns
    -------
    tuple[Array, Array]
        ``(packed_bar, centers_bar)``: the cotangent of the INPUT table (leaf
        rows; the internal rows of the input are overwritten by the forward and
        get zero) and of the centres.
    """
    from jaccpot.pallas.cascade_real_lanes import m2m_real_levels_lanes_reverse_pallas

    del num_warps
    if int(num_internal) <= 0:
        return jnp.asarray(out_bar, out_packed.dtype), jnp.zeros_like(centers)
    k = _cascade_lanes_k()
    g, dbar = m2m_real_levels_lanes_reverse_pallas(
        out_packed,
        centers,
        parent,
        nodes_by_level,
        level_offsets,
        out_bar,
        order=order,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        k_lanes=k,
        interpret=interpret,
        backend=backend,
        num_warps=max(1, k // 32),
    )
    packed_bar = g.at[: int(num_internal)].set(0.0)
    par = jnp.asarray(parent, jnp.asarray(level_offsets).dtype)
    centers_bar = _fold_edge_cotangents(dbar, par, sign=1.0, dtype=centers.dtype)
    return packed_bar, centers_bar


def l2l_real_levels_reverse_pallas(
    out_locals: Array,
    centers: Array,
    parent: Array,
    left_child: Array,
    right_child: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    out_bar: Array,
    *,
    order: int,
    num_levels: int,
    level_batch_width: int,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 4,
) -> tuple[Array, Array]:
    """Adjoint of the downward L2L cascade: one launch per level, bottom up, a parent per lane.

    Parameters
    ----------
    out_locals : Array
        ``[nodes, C]`` the forward's OUTPUT (fully cascaded locals).
    centers : Array
        ``[nodes, 3]`` expansion centres.
    parent : Array
        ``[nodes]`` parent per node.
    left_child : Array
        ``[internal]`` left children.
    right_child : Array
        ``[internal]`` right children.
    nodes_by_level : Array
        Level-major node ids.
    level_offsets : Array
        ``[levels + 1]`` starts.
    out_bar : Array
        ``[nodes, C]`` cotangent of the output table.
    order : int
        Expansion order. Static.
    num_levels : int
        Levels present. Static.
    level_batch_width : int
        Nodes per level at most. Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Not read (see :func:`m2m_real_levels_reverse_pallas`).

    Returns
    -------
    tuple[Array, Array]
        ``(locals_bar, centers_bar)``: the cotangent of the INPUT table (every
        row, since ``out[n] = in[n] + L2L(out[parent])``) and of the centres.
    """
    from jaccpot.pallas.cascade_real_lanes import l2l_real_levels_lanes_reverse_pallas

    del num_warps
    k = _cascade_lanes_k()
    locals_bar, dbar = l2l_real_levels_lanes_reverse_pallas(
        out_locals,
        centers,
        left_child,
        right_child,
        nodes_by_level,
        level_offsets,
        out_bar,
        order=order,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        k_lanes=k,
        interpret=interpret,
        backend=backend,
        num_warps=max(1, k // 32),
    )
    par = jnp.asarray(parent, jnp.asarray(level_offsets).dtype)
    centers_bar = _fold_edge_cotangents(dbar, par, sign=-1.0, dtype=centers.dtype)
    return locals_bar, centers_bar


# ------------------------------------------------------------- custom_vjp seams
# ``pallas_call`` has no autodiff rule (its generic JVP rule dies on
# ``program_id``), so the cascades cross the gradient path through these. The
# forward IS the production launch sequence; the reverse is the kernels above.
# Statics ride in ``nondiff_argnums``; the integer topology arrays are ordinary
# arguments whose cotangents are ``None`` (symbolic zero).


def _cascade_lanes_k() -> int:
    """``JACCPOT_CASCADE_LANES_K``: nodes per program of the lane forward kernels.

    The lane kernels are the only forward since the 2026-10 cleanup (X5) removed
    the one-program-per-node level kernels (``JACCPOT_CASCADE_KERNEL=level``, no
    longer read).

    Returns
    -------
    int
        Nodes per program (``k_lanes``) of
        :func:`~jaccpot.pallas.cascade_real_lanes.m2m_real_levels_lanes_pallas`
        and :func:`~jaccpot.pallas.cascade_real_lanes.l2l_real_levels_lanes_pallas`.
    """
    from jaccpot._env import env_int

    return env_int("JACCPOT_CASCADE_LANES_K", 32, minimum=1)


def _m2m_forward(
    packed: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    **kw: Any,
) -> Array:
    """The M2M custom VJP's forward: the one-node-per-lane kernel.

    Parameters
    ----------
    packed : Array
        As :func:`m2m_real_levels_pallas_cvjp`.
    centers : Array
        As :func:`m2m_real_levels_pallas_cvjp`.
    left_child : Array
        As :func:`m2m_real_levels_pallas_cvjp`.
    right_child : Array
        As :func:`m2m_real_levels_pallas_cvjp`.
    nodes_by_level : Array
        As :func:`m2m_real_levels_pallas_cvjp`.
    level_offsets : Array
        As :func:`m2m_real_levels_pallas_cvjp`.
    **kw : Any
        Its keyword arguments. ``num_warps`` is the reverse kernels'; the lane
        kernel runs ``k / 32`` warps.

    Returns
    -------
    Array
        ``[nodes, C]`` packed multipoles with the internal nodes filled.
    """
    from jaccpot.pallas.cascade_real_lanes import m2m_real_levels_lanes_pallas

    k = _cascade_lanes_k()
    kw = dict(kw, k_lanes=k, num_warps=max(1, k // 32))
    return m2m_real_levels_lanes_pallas(
        packed,
        centers,
        left_child,
        right_child,
        nodes_by_level,
        level_offsets,
        **kw,
    )


def _l2l_forward(
    coeffs_local: Array,
    centers: Array,
    parent: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    **kw: Any,
) -> Union[Array, tuple[Array, Array]]:
    """The L2L custom VJP's forward: the one-node-per-lane kernel.

    Parameters
    ----------
    coeffs_local : Array
        As :func:`l2l_real_levels_pallas_cvjp`.
    centers : Array
        As :func:`l2l_real_levels_pallas_cvjp`.
    parent : Array
        As :func:`l2l_real_levels_pallas_cvjp`.
    nodes_by_level : Array
        As :func:`l2l_real_levels_pallas_cvjp`.
    level_offsets : Array
        As :func:`l2l_real_levels_pallas_cvjp`.
    **kw : Any
        Its keyword arguments (``num_warps`` is the reverse kernels'; the force
        scale's twin, :func:`l2l_real_levels_fb_pallas_cvjp`, adds
        ``force_scale``).

    Returns
    -------
    Union[Array, tuple[Array, Array]]
        ``[nodes, C]`` packed locals, every ancestor's field cascaded down; with
        ``force_scale``, ``(locals, sums)``.
    """
    from jaccpot.pallas.cascade_real_lanes import l2l_real_levels_lanes_pallas

    k = _cascade_lanes_k()
    kw = dict(kw, k_lanes=k, num_warps=max(1, k // 32))
    return l2l_real_levels_lanes_pallas(
        coeffs_local, centers, parent, nodes_by_level, level_offsets, **kw
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13))
def m2m_real_levels_pallas_cvjp(
    packed: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    parent: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    order: int,
    num_internal: int,
    num_levels: int,
    level_batch_width: int,
    interpret: bool,
    backend: str,
    num_warps: int,
) -> Array:
    """Upward M2M cascade through a custom VJP: the lane forward, the level reverse.

    The forward is
    :func:`~jaccpot.pallas.cascade_real_lanes.m2m_real_levels_lanes_pallas`
    (``JACCPOT_CASCADE_LANES_K`` nodes per program), byte-identical to calling it;
    the reverse is :func:`m2m_real_levels_reverse_pallas`.

    Parameters
    ----------
    packed : Array
        ``[nodes, C]`` packed multipoles with the leaves filled. Differentiable.
    centers : Array
        ``[nodes, 3]`` expansion centres. Differentiable.
    left_child : Array
        ``[internal]`` left children.
    right_child : Array
        ``[internal]`` right children.
    parent : Array
        ``[nodes]`` parent per node (needed by the reverse).
    nodes_by_level : Array
        Level-major node ids.
    level_offsets : Array
        ``[levels + 1]`` starts.
    order : int
        Expansion order. ``nondiff_argnums``.
    num_internal : int
        Internal node count. ``nondiff_argnums``.
    num_levels : int
        Levels covered. ``nondiff_argnums``.
    level_batch_width : int
        Programs per level. ``nondiff_argnums``.
    interpret : bool
        Pallas interpret mode. ``nondiff_argnums``.
    backend : str
        Pallas GPU lowering. ``nondiff_argnums``.
    num_warps : int
        Warps per program of the reverse kernels (the lane forward runs
        ``k / 32``). ``nondiff_argnums``.

    Returns
    -------
    Array
        ``[nodes, C]`` packed multipoles with the internal nodes filled.
    """
    return _m2m_forward(
        packed,
        centers,
        left_child,
        right_child,
        nodes_by_level,
        level_offsets,
        order=order,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )


def _m2m_cvjp_fwd(
    packed,
    centers,
    left_child,
    right_child,
    parent,
    nodes_by_level,
    level_offsets,
    order,
    num_internal,
    num_levels,
    level_batch_width,
    interpret,
    backend,
    num_warps,
):
    out = _m2m_forward(
        packed,
        centers,
        left_child,
        right_child,
        nodes_by_level,
        level_offsets,
        order=order,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return out, (out, centers, parent, nodes_by_level, level_offsets)


def _m2m_cvjp_bwd(
    order,
    num_internal,
    num_levels,
    level_batch_width,
    interpret,
    backend,
    num_warps,
    residual,
    out_bar,
):
    out, centers, parent, nodes_by_level, level_offsets = residual
    packed_bar, centers_bar = m2m_real_levels_reverse_pallas(
        out,
        centers,
        parent,
        nodes_by_level,
        level_offsets,
        out_bar,
        order=order,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return (packed_bar, centers_bar, None, None, None, None, None)


m2m_real_levels_pallas_cvjp.defvjp(_m2m_cvjp_fwd, _m2m_cvjp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12))
def l2l_real_levels_pallas_cvjp(
    coeffs_local: Array,
    centers: Array,
    parent: Array,
    left_child: Array,
    right_child: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    order: int,
    num_levels: int,
    level_batch_width: int,
    interpret: bool,
    backend: str,
    num_warps: int,
) -> Array:
    """Downward L2L cascade through a custom VJP: the lane forward, the level reverse.

    The forward is
    :func:`~jaccpot.pallas.cascade_real_lanes.l2l_real_levels_lanes_pallas`,
    byte-identical to calling it; the reverse is
    :func:`l2l_real_levels_reverse_pallas`.

    Parameters
    ----------
    coeffs_local : Array
        ``[nodes, C]`` packed locals after M2L. Differentiable.
    centers : Array
        ``[nodes, 3]`` expansion centres. Differentiable.
    parent : Array
        ``[nodes]`` parent per node.
    left_child : Array
        ``[internal]`` left children (needed by the reverse).
    right_child : Array
        ``[internal]`` right children (needed by the reverse).
    nodes_by_level : Array
        Level-major node ids.
    level_offsets : Array
        ``[levels + 1]`` starts.
    order : int
        Expansion order. ``nondiff_argnums``.
    num_levels : int
        Levels present. ``nondiff_argnums``.
    level_batch_width : int
        Programs per level. ``nondiff_argnums``.
    interpret : bool
        Pallas interpret mode. ``nondiff_argnums``.
    backend : str
        Pallas GPU lowering. ``nondiff_argnums``.
    num_warps : int
        Warps per program of the reverse kernels (the lane forward runs
        ``k / 32``). ``nondiff_argnums``.

    Returns
    -------
    Array
        ``[nodes, C]`` fully cascaded locals.
    """
    return _l2l_forward(
        coeffs_local,
        centers,
        parent,
        nodes_by_level,
        level_offsets,
        order=order,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )


def _l2l_cvjp_fwd(
    coeffs_local,
    centers,
    parent,
    left_child,
    right_child,
    nodes_by_level,
    level_offsets,
    order,
    num_levels,
    level_batch_width,
    interpret,
    backend,
    num_warps,
):
    out = _l2l_forward(
        coeffs_local,
        centers,
        parent,
        nodes_by_level,
        level_offsets,
        order=order,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return out, (
        out,
        centers,
        parent,
        left_child,
        right_child,
        nodes_by_level,
        level_offsets,
    )


def _l2l_cvjp_bwd(
    order,
    num_levels,
    level_batch_width,
    interpret,
    backend,
    num_warps,
    residual,
    out_bar,
):
    out, centers, parent, left_child, right_child, nodes_by_level, level_offsets = (
        residual
    )
    locals_bar, centers_bar = l2l_real_levels_reverse_pallas(
        out,
        centers,
        parent,
        left_child,
        right_child,
        nodes_by_level,
        level_offsets,
        out_bar,
        order=order,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return (locals_bar, centers_bar, None, None, None, None, None)


l2l_real_levels_pallas_cvjp.defvjp(_l2l_cvjp_fwd, _l2l_cvjp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10, 11, 12, 13))
def l2l_real_levels_fb_pallas_cvjp(
    coeffs_local: Array,
    force_scale: Array,
    centers: Array,
    parent: Array,
    left_child: Array,
    right_child: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    order: int,
    num_levels: int,
    level_batch_width: int,
    interpret: bool,
    backend: str,
    num_warps: int,
) -> tuple[Array, Array]:
    """:func:`l2l_real_levels_pallas_cvjp` that also pushes a per-node scalar down.

    The forward is the lane kernel with ``force_scale``
    (:func:`~jaccpot.pallas.cascade_real_lanes.l2l_real_levels_lanes_pallas`): the
    same launches carry ``mac_type='dehnen_error'``'s far force-scale term from
    every node to its descendants. The reverse is
    :func:`l2l_real_levels_pallas_cvjp`'s for the locals; the scalar is a topology
    quantity (the next walk's eq (16a) thresholds) and is not differentiated.

    Parameters
    ----------
    coeffs_local : Array
        ``[nodes, C]`` packed locals after M2L. Differentiable.
    force_scale : Array
        ``[nodes]`` each node's own value. Not differentiated.
    centers : Array
        ``[nodes, 3]`` expansion centres. Differentiable.
    parent : Array
        ``[nodes]`` parent per node.
    left_child : Array
        ``[internal]`` left children (needed by the reverse).
    right_child : Array
        ``[internal]`` right children (needed by the reverse).
    nodes_by_level : Array
        Level-major node ids.
    level_offsets : Array
        ``[levels + 1]`` starts.
    order : int
        Expansion order. ``nondiff_argnums``.
    num_levels : int
        Levels present. ``nondiff_argnums``.
    level_batch_width : int
        Programs per level. ``nondiff_argnums``.
    interpret : bool
        Pallas interpret mode. ``nondiff_argnums``.
    backend : str
        Pallas GPU lowering. ``nondiff_argnums``.
    num_warps : int
        Warps per program of the reverse kernels. ``nondiff_argnums``.

    Returns
    -------
    tuple[Array, Array]
        ``([nodes, C]`` cascaded locals, ``[nodes]`` each node's value plus all
        its ancestors' ``)``.
    """
    return _l2l_fb_forward(
        coeffs_local,
        force_scale,
        centers,
        parent,
        nodes_by_level,
        level_offsets,
        order=order,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )


def _l2l_fb_forward(
    coeffs_local: Array,
    force_scale: Array,
    centers: Array,
    parent: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    **kw: Any,
) -> tuple[Array, Array]:
    out = _l2l_forward(
        coeffs_local,
        centers,
        parent,
        nodes_by_level,
        level_offsets,
        force_scale=force_scale,
        **kw,
    )
    return out[0], out[1]


def _l2l_fb_cvjp_fwd(
    coeffs_local,
    force_scale,
    centers,
    parent,
    left_child,
    right_child,
    nodes_by_level,
    level_offsets,
    order,
    num_levels,
    level_batch_width,
    interpret,
    backend,
    num_warps,
):
    out, fb = _l2l_fb_forward(
        coeffs_local,
        force_scale,
        centers,
        parent,
        nodes_by_level,
        level_offsets,
        order=order,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return (out, fb), (
        out,
        centers,
        parent,
        left_child,
        right_child,
        nodes_by_level,
        level_offsets,
    )


def _l2l_fb_cvjp_bwd(
    order,
    num_levels,
    level_batch_width,
    interpret,
    backend,
    num_warps,
    residual,
    cotangents,
):
    out_bar, _ = cotangents
    locals_bar, centers_bar, *rest = _l2l_cvjp_bwd(
        order,
        num_levels,
        level_batch_width,
        interpret,
        backend,
        num_warps,
        residual,
        out_bar,
    )
    return (locals_bar, None, centers_bar, *rest)


l2l_real_levels_fb_pallas_cvjp.defvjp(_l2l_fb_cvjp_fwd, _l2l_fb_cvjp_bwd)
