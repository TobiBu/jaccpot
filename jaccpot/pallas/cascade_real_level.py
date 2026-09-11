"""Real-basis M2M and L2L cascades as ONE Pallas launch per tree level (plan sub-10ms, Phase 3).

The level loops of the upward (``aggregate_m2m_real_by_level``) and downward
(``_propagate_solidfmm_locals_by_level``) sweeps translate every node of a
level with the vectorised rotation operators, which XLA breaks into dozens of
small fusions per level: ~0.8 ms per level, 66 ms for the 40-level radix tree
over Morton-cell leaves at N = 2x10^5 (11 ms on the 13-level bucket tree).

Here a level is one ``pallas_call``: one program per node of the level (a
static batch of programs, the level's node count masked), each gathering its
partner rows, rotating them onto the pair axis, applying the shift core along
z and rotating back, then writing its own row with a dynamic store into the
coefficient array it aliases. The rotation halves are exactly the M2L CSR
kernel's (:mod:`jaccpot.pallas.m2l_real_csr`) with one block swapped: the
alignment ``T`` is not orthogonal, so multipoles come back with ``T^-1``
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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from jaccpot.pallas._compat import KernelRef, pallas_backend_kwargs
from jaccpot.pallas.m2l_real_csr import (
    _TABLE_KEYS,
    _bapply,
    _dz,
    _tables_to_jnp,
    m2l_real_csr_tables,
    pack_centred,
    pallas_m2l_real_csr_supported,
    unpack_centred,
)

try:
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None

__all__ = [
    "cascade_level_tables",
    "l2l_real_centred_pair_jax",
    "l2l_real_levels_pallas",
    "m2m_real_centred_pair_jax",
    "m2m_real_levels_pallas",
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
    return t


_CORE_KEYS = ("Sm2m", "Sl2l", "invfact", "kpow", "Binv", "BinvT")


def _core_tables_to_jnp(order: int, dtype: Any) -> dict[str, Array]:
    t = cascade_level_tables(int(order))
    out = _tables_to_jnp(int(order), dtype)
    for k in _CORE_KEYS:
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


def _translate_rows(rows: Array, delta3: tuple, t: dict[str, Array], which: str) -> Array:
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


def m2m_real_centred_pair_jax(child_packed: Array, delta: Array, *, order: int) -> Array:
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
    rows = pack_centred(child_packed[None, :], order=int(order))[0].reshape(t["invfact"].shape[0], -1)
    out = _translate_rows(rows, (delta[0], delta[1], delta[2]), t, "m2m")
    return unpack_centred(out.reshape(1, -1), order=int(order))[0]


def l2l_real_centred_pair_jax(parent_packed: Array, delta: Array, *, order: int) -> Array:
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
    rows = pack_centred(parent_packed[None, :], order=int(order))[0].reshape(t["invfact"].shape[0], -1)
    out = _translate_rows(rows, (delta[0], delta[1], delta[2]), t, "l2l")
    return unpack_centred(out.reshape(1, -1), order=int(order))[0]


# ------------------------------------------------------------------ level kernels


def _m2m_level_kernel(
    rows_ref: KernelRef,
    cent_ref: KernelRef,
    left_ref: KernelRef,
    right_ref: KernelRef,
    nbl_ref: KernelRef,
    start_ref: KernelRef,
    count_ref: KernelRef,
    *table_and_out_refs: KernelRef,
    bp: int,
    wp: int,
    num_internal: int,
) -> None:
    """One parent of the level: sum the M2M of its two children into its row.

    Parameters
    ----------
    rows_ref : KernelRef
        Whole centred coefficient table ``[nodes + 1, Bp*Wp]`` (row ``nodes`` is
        the dead row invalid programs write).
    cent_ref : KernelRef
        Whole padded centre table ``[nodes + 1, 4]``.
    left_ref : KernelRef
        Left child per internal node ``[internal]``.
    right_ref : KernelRef
        Right child per internal node ``[internal]``.
    nbl_ref : KernelRef
        ``nodes_by_level`` padded with ``-1`` ``[nodes + batch]``.
    start_ref : KernelRef
        This level's start in ``nodes_by_level`` ``[1]``.
    count_ref : KernelRef
        This level's node count ``[1]``.
    *table_and_out_refs : KernelRef
        The constant tables (``_TABLE_KEYS`` then ``_CORE_KEYS``) and the
        output ref ``[nodes + 1, Bp*Wp]`` aliased to ``rows_ref``.
    bp : int
        ``Bp``. Static.
    wp : int
        ``Wp``. Static.
    num_internal : int
        Internal node count. Static.

    Returns
    -------
    None
        Writes the parent's row.
    """
    n_tables = len(_TABLE_KEYS) + len(_CORE_KEYS)
    table_refs = table_and_out_refs[:n_tables]
    (out_ref,) = table_and_out_refs[n_tables:]
    t = {k: ref[...] for k, ref in zip((*_TABLE_KEYS, *_CORE_KEYS), table_refs)}
    slot = pl.program_id(0)
    start = start_ref[0]
    count = count_ref[0]
    node = nbl_ref[start + slot]
    valid = (slot < count) & (node >= 0) & (node < num_internal)
    node_safe = jnp.where(valid, node, 0)
    px = cent_ref[node_safe, 0]
    py = cent_ref[node_safe, 1]
    pz = cent_ref[node_safe, 2]
    acc = jnp.zeros((bp, wp), dtype=out_ref.dtype)
    for child_ref in (left_ref, right_ref):
        c = child_ref[node_safe]
        c_valid = valid & (c >= 0)
        c_safe = jnp.where(c_valid, c, 0)
        rows = rows_ref[c_safe, :].reshape(bp, wp)
        dx = cent_ref[c_safe, 0] - px
        dy = cent_ref[c_safe, 1] - py
        dz_ = cent_ref[c_safe, 2] - pz
        contrib = _translate_rows(rows, (dx, dy, dz_), t, "m2m")
        acc = acc + jnp.where(c_valid, contrib, 0.0)
    dead = jnp.asarray(out_ref.shape[0] - 1, dtype=node.dtype)
    target = jnp.where(valid, node_safe, dead)
    out_ref[target, :] = acc.reshape(bp * wp).astype(out_ref.dtype)


def _l2l_level_kernel(
    rows_ref: KernelRef,
    cent_ref: KernelRef,
    parent_ref: KernelRef,
    nbl_ref: KernelRef,
    start_ref: KernelRef,
    count_ref: KernelRef,
    *table_and_out_refs: KernelRef,
    bp: int,
    wp: int,
) -> None:
    """One node of the level: add its parent's translated local to its own row.

    Parameters
    ----------
    rows_ref : KernelRef
        Whole centred local table ``[nodes + 1, Bp*Wp]``.
    cent_ref : KernelRef
        Whole padded centre table ``[nodes + 1, 4]``.
    parent_ref : KernelRef
        Parent per node ``[nodes]`` (``-1`` at the root).
    nbl_ref : KernelRef
        ``nodes_by_level`` padded with ``-1``.
    start_ref : KernelRef
        This level's start ``[1]``.
    count_ref : KernelRef
        This level's node count ``[1]``.
    *table_and_out_refs : KernelRef
        Constant tables (``_TABLE_KEYS`` then ``_CORE_KEYS``) and the
        aliased output ref.
    bp : int
        ``Bp``. Static.
    wp : int
        ``Wp``. Static.

    Returns
    -------
    None
        Writes the node's row.
    """
    n_tables = len(_TABLE_KEYS) + len(_CORE_KEYS)
    table_refs = table_and_out_refs[:n_tables]
    (out_ref,) = table_and_out_refs[n_tables:]
    t = {k: ref[...] for k, ref in zip((*_TABLE_KEYS, *_CORE_KEYS), table_refs)}
    slot = pl.program_id(0)
    start = start_ref[0]
    count = count_ref[0]
    node = nbl_ref[start + slot]
    valid = (slot < count) & (node >= 0)
    node_safe = jnp.where(valid, node, 0)
    par = parent_ref[node_safe]
    valid = valid & (par >= 0)
    par_safe = jnp.where(valid, par, 0)
    rows = rows_ref[par_safe, :].reshape(bp, wp)
    dx = cent_ref[par_safe, 0] - cent_ref[node_safe, 0]
    dy = cent_ref[par_safe, 1] - cent_ref[node_safe, 1]
    dz_ = cent_ref[par_safe, 2] - cent_ref[node_safe, 2]
    contrib = _translate_rows(rows, (dx, dy, dz_), t, "l2l")
    own = rows_ref[node_safe, :].reshape(bp, wp)
    new = own + jnp.where(valid, contrib, 0.0)
    dead = jnp.asarray(out_ref.shape[0] - 1, dtype=node.dtype)
    target = jnp.where(valid, node_safe, dead)
    out_ref[target, :] = new.reshape(bp * wp).astype(out_ref.dtype)


def _full(arr: Array) -> "pl.BlockSpec":
    shp = tuple(arr.shape)
    return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))


def _level_call(kernel, operands: list, *, num_programs: int, interpret: bool,
                backend: str, num_warps: int, name: str):
    """One level launch: whole-array refs, ``num_programs`` programs.

    ``operands[0]`` is the coefficient table; it is aliased to the output, so
    every row a program does not write keeps its value (programs read rows of
    OTHER levels and write only their own, so the in-place update is safe).
    """
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(num_warps=int(num_warps))
    rows = operands[0]
    return pl.pallas_call(
        kernel,
        grid=(int(num_programs),),
        in_specs=[_full(a) for a in operands],
        out_specs=_full(rows),
        out_shape=jax.ShapeDtypeStruct(rows.shape, rows.dtype),
        input_output_aliases={0: 0},
        interpret=bool(interpret),
        name=name,
        **backend_kwargs,
    )(*operands)


def m2m_real_levels_pallas(
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
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 4,
) -> Array:
    """Upward M2M cascade: one Pallas launch per internal level, deepest first.

    Drop-in for :func:`jaccpot.upward.real_tree_expansions.aggregate_m2m_real_by_level`
    (same arguments), agreeing with it to round-off.

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
        Programs per level (>= the widest level). Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.

    Returns
    -------
    Array
        ``[nodes, C]`` packed multipoles with the internal nodes filled.
    """
    p = int(order)
    t = cascade_level_tables(p)
    Bp, Wp = t["Bp"], t["Wp"]
    dtype = packed.dtype
    total = int(packed.shape[0])
    if int(num_internal) <= 0:
        return packed
    tables = _core_tables_to_jnp(p, dtype)
    table_arrays = [tables[k] for k in (*_TABLE_KEYS, *_CORE_KEYS)]
    rows = pack_centred(packed, order=p)
    rows = jnp.concatenate([rows, jnp.zeros((1, Bp * Wp), dtype)], axis=0)  # dead row
    cent = jnp.pad(jnp.asarray(centers, dtype), ((0, 1), (0, 1)))
    width = int(max(level_batch_width, 1))
    idx = level_offsets.dtype
    nbl = jnp.concatenate([jnp.asarray(nodes_by_level, idx), jnp.full((width,), -1, idx)])
    offs = jnp.asarray(level_offsets, idx)
    kernel = functools.partial(_m2m_level_kernel, bp=Bp, wp=Wp, num_internal=int(num_internal))

    def body(rev, rows_state):
        level = (int(num_levels) - 2) - rev
        start = offs[level][None]
        count = (offs[level + 1] - offs[level])[None]
        return _level_call(
            kernel,
            [rows_state, cent, jnp.asarray(left_child, idx), jnp.asarray(right_child, idx), nbl, start, count, *table_arrays],
            num_programs=width, interpret=interpret, backend=backend, num_warps=num_warps,
            name=f"m2m_real_level_p{p}",
        )

    rows = lax.fori_loop(0, max(int(num_levels) - 1, 0), body, rows)
    return unpack_centred(rows[:total], order=p)


def l2l_real_levels_pallas(
    coeffs_local: Array,
    centers: Array,
    parent: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    *,
    order: int,
    num_levels: int,
    level_batch_width: int,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 4,
) -> Array:
    """Downward L2L cascade: one Pallas launch per level below the root, top first.

    Parameters
    ----------
    coeffs_local : Array
        ``[nodes, C]`` packed locals after M2L (every node carries its own level's
        far field).
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
        Programs per level. Static.
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
    t = cascade_level_tables(p)
    Bp, Wp = t["Bp"], t["Wp"]
    dtype = coeffs_local.dtype
    total = int(coeffs_local.shape[0])
    tables = _core_tables_to_jnp(p, dtype)
    table_arrays = [tables[k] for k in (*_TABLE_KEYS, *_CORE_KEYS)]
    rows = pack_centred(coeffs_local, order=p)
    rows = jnp.concatenate([rows, jnp.zeros((1, Bp * Wp), dtype)], axis=0)
    cent = jnp.pad(jnp.asarray(centers, dtype), ((0, 1), (0, 1)))
    width = int(max(level_batch_width, 1))
    idx = level_offsets.dtype
    nbl = jnp.concatenate([jnp.asarray(nodes_by_level, idx), jnp.full((width,), -1, idx)])
    offs = jnp.asarray(level_offsets, idx)
    par = jnp.asarray(parent, idx)
    kernel = functools.partial(_l2l_level_kernel, bp=Bp, wp=Wp)

    def body(level, rows_state):
        start = offs[level][None]
        count = (offs[level + 1] - offs[level])[None]
        return _level_call(
            kernel,
            [rows_state, cent, par, nbl, start, count, *table_arrays],
            num_programs=width, interpret=interpret, backend=backend, num_warps=num_warps,
            name=f"l2l_real_level_p{p}",
        )

    # level 0 is the root (nothing above it); levels 1 .. num_levels-1 receive
    rows = lax.fori_loop(1, max(int(num_levels), 1), body, rows)
    return unpack_centred(rows[:total], order=p)
