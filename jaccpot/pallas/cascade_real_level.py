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
    "l2l_real_levels_pallas_cvjp",
    "l2l_real_levels_reverse_pallas",
    "m2m_real_centred_pair_jax",
    "m2m_real_levels_pallas",
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
        kernels set this; the forward kernels keep the unguarded trace.

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
        # polar route cannot resolve it (see operators/_transverse_degeneracy_jvp
        # for the analytic limit the pure-JAX operators add); at delta == 0 the
        # two centres are the same function of the positions and the cotangent
        # cancels whatever finite value it takes.
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
    slot = pl.program_id(0)
    start = start_ref[0]
    count = count_ref[0]

    # The grid is the widest level's width for EVERY level; programs past this
    # level's count exit before touching the tables (47 launches x the widest
    # level's work = 8.4 ms at N=2e5 without this, ~1 ms with it).
    @pl.when(slot < count)
    def _live():
        t = {k: ref[...] for k, ref in zip((*_TABLE_KEYS, *_CORE_KEYS), table_refs)}
        node = nbl_ref[start + slot]
        valid = (node >= 0) & (node < num_internal)
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
    slot = pl.program_id(0)
    start = start_ref[0]
    count = count_ref[0]

    @pl.when(slot < count)  # see _m2m_level_kernel
    def _live():
        t = {k: ref[...] for k, ref in zip((*_TABLE_KEYS, *_CORE_KEYS), table_refs)}
        node = nbl_ref[start + slot]
        valid = node >= 0
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


def _level_call(
    kernel: Any,
    operands: list,
    *,
    num_programs: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    name: str,
) -> Array:
    """One level launch: whole-array refs, ``num_programs`` programs.

    ``operands[0]`` is the coefficient table; it is aliased to the output, so
    every row a program does not write keeps its value (programs read rows of
    OTHER levels and write only their own, so the in-place update is safe).

    Parameters
    ----------
    kernel : Any
        The Pallas kernel to launch, already bound to its static arguments.
    operands : list
        Kernel operands; ``operands[0]`` is the aliased coefficient table.
    num_programs : int
        Programs in the 1-D grid.
    interpret : bool
        Run Pallas' reference interpreter instead of lowering.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.
    name : str
        Kernel name, as it appears in a profile.

    Returns
    -------
    Array
        The updated coefficient table.
    """
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )
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
    nbl = jnp.concatenate(
        [jnp.asarray(nodes_by_level, idx), jnp.full((width,), -1, idx)]
    )
    offs = jnp.asarray(level_offsets, idx)
    kernel = functools.partial(
        _m2m_level_kernel, bp=Bp, wp=Wp, num_internal=int(num_internal)
    )

    def body(rev: Array, rows_state: Array) -> Array:
        level = (int(num_levels) - 2) - rev
        start = offs[level][None]
        count = (offs[level + 1] - offs[level])[None]
        return _level_call(
            kernel,
            [
                rows_state,
                cent,
                jnp.asarray(left_child, idx),
                jnp.asarray(right_child, idx),
                nbl,
                start,
                count,
                *table_arrays,
            ],
            num_programs=width,
            interpret=interpret,
            backend=backend,
            num_warps=num_warps,
            name=f"m2m_real_level_p{p}",
        )

    # static trip count: unrolled so the per-level launches sit in the parent
    # computation (graph-capturable) instead of a while body
    rows = lax.fori_loop(0, max(int(num_levels) - 1, 0), body, rows, unroll=True)
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
    nbl = jnp.concatenate(
        [jnp.asarray(nodes_by_level, idx), jnp.full((width,), -1, idx)]
    )
    offs = jnp.asarray(level_offsets, idx)
    par = jnp.asarray(parent, idx)
    kernel = functools.partial(_l2l_level_kernel, bp=Bp, wp=Wp)

    def body(level: Array, rows_state: Array) -> Array:
        start = offs[level][None]
        count = (offs[level + 1] - offs[level])[None]
        return _level_call(
            kernel,
            [rows_state, cent, par, nbl, start, count, *table_arrays],
            num_programs=width,
            interpret=interpret,
            backend=backend,
            num_warps=num_warps,
            name=f"l2l_real_level_p{p}",
        )

    # level 0 is the root (nothing above it); levels 1 .. num_levels-1 receive
    rows = lax.fori_loop(1, max(int(num_levels), 1), body, rows, unroll=True)
    return unpack_centred(rows[:total], order=p)


# ------------------------------------------------------------- reverse kernels
# Adjoints of the two cascades (plan fast-gradients, Phase 2). Each level is
# linear in the coefficients at fixed geometry, so the coefficient half of the
# adjoint is the transposed translate and the geometry half a contraction of
# the cotangent with d(translate)/d(delta). Both come out of ONE ``jax.vjp`` of
# the same ``_translate_rows`` body the forward runs (guarded, ``safe=True``),
# traced inside the kernel: the transpose of a rotate/shift chain is the chain
# of transposes, and JAX writes it. The pass structures swap:
#
# * M2M reverse: top-down, one program per NODE, ``g[c] += J_c^T g[parent]``
#   (the L2L kernel's shape) -- each node has one parent, so no two programs
#   write the same row;
# * L2L reverse: bottom-up, one program per PARENT, ``g[p] += sum_c J_c^T g[c]``
#   (the M2M kernel's shape) -- gathering the two children avoids two programs
#   adding into one parent row.
#
# The per-edge geometry cotangent is written to the CHILD's slot of a
# ``[nodes + 1, 4]`` table (one writer per node) and folded onto the two centres
# afterwards in XLA; the kernels never scatter-add.


def _dbar_row(dxb: Array, dyb: Array, dzb: Array, valid: Array, dtype: Any) -> Array:
    """``[dxb, dyb, dzb, 0]`` as a ``(4,)`` row without ``concatenate`` (Triton-safe).

    Parameters
    ----------
    dxb : Array
        Scalar cotangent of the edge vector's ``x``.
    dyb : Array
        Same, ``y``.
    dzb : Array
        Same, ``z``.
    valid : Array
        Scalar bool; an invalid program writes zeros.
    dtype : Any
        Row dtype.

    Returns
    -------
    Array
        ``(4,)`` row.
    """
    lane = lax.broadcasted_iota(jnp.int32, (4,), 0)
    zero = jnp.asarray(0.0, dtype=dtype)
    row = jnp.where(
        lane == 0, dxb, jnp.where(lane == 1, dyb, jnp.where(lane == 2, dzb, zero))
    )
    return jnp.where(valid, row, zero).astype(dtype)


def _m2m_rev_level_kernel(
    g_ref: KernelRef,
    _dbar_in_ref: KernelRef,
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
    """One node of the level: pull its parent's multipole cotangent down onto its own.

    Parameters
    ----------
    g_ref : KernelRef
        Cotangent table ``[nodes + 1, Bp*Wp]`` (aliased to the first output);
        the parent's row is final, this node's row is read and rewritten.
    _dbar_in_ref : KernelRef
        Geometry-cotangent table ``[nodes + 1, 4]`` (aliased to the second
        output; write-only here).
    rows_ref : KernelRef
        The forward's OUTPUT multipoles in centred rows (the linearisation point).
    cent_ref : KernelRef
        Padded centres ``[nodes + 1, 4]``.
    parent_ref : KernelRef
        Parent per node ``[nodes]`` (``-1`` at the root).
    nbl_ref : KernelRef
        ``nodes_by_level`` padded with ``-1``.
    start_ref : KernelRef
        This level's start ``[1]``.
    count_ref : KernelRef
        This level's node count ``[1]``.
    *table_and_out_refs : KernelRef
        Constant tables, then the two aliased outputs (``g``, ``dbar``).
    bp : int
        ``Bp``. Static.
    wp : int
        ``Wp``. Static.

    Returns
    -------
    None
        Writes the node's cotangent row and its edge geometry cotangent.
    """
    n_tables = len(_TABLE_KEYS) + len(_CORE_KEYS)
    table_refs = table_and_out_refs[:n_tables]
    g_out_ref, dbar_ref = table_and_out_refs[n_tables:]
    slot = pl.program_id(0)
    start = start_ref[0]
    count = count_ref[0]

    @pl.when(slot < count)  # see _m2m_level_kernel
    def _live():
        t = {k: ref[...] for k, ref in zip((*_TABLE_KEYS, *_CORE_KEYS), table_refs)}
        node = nbl_ref[start + slot]
        valid = node >= 0
        node_safe = jnp.where(valid, node, 0)
        par = parent_ref[node_safe]
        valid = valid & (par >= 0)
        par_safe = jnp.where(valid, par, 0)
        rows_c = rows_ref[node_safe, :].reshape(bp, wp)
        g_p = g_ref[par_safe, :].reshape(bp, wp)
        # M2M delta = child - parent
        dx = cent_ref[node_safe, 0] - cent_ref[par_safe, 0]
        dy = cent_ref[node_safe, 1] - cent_ref[par_safe, 1]
        dz_ = cent_ref[node_safe, 2] - cent_ref[par_safe, 2]

        def translate(r: Array, d: tuple) -> Array:
            return _translate_rows(r, d, t, "m2m", safe=True)

        _, vjp = jax.vjp(translate, rows_c, (dx, dy, dz_))
        rb, (dxb, dyb, dzb) = vjp(g_p)
        own = g_ref[node_safe, :].reshape(bp, wp)
        new = own + jnp.where(valid, rb, 0.0)
        dead = jnp.asarray(g_out_ref.shape[0] - 1, dtype=node.dtype)
        target = jnp.where(valid, node_safe, dead)
        g_out_ref[target, :] = new.reshape(bp * wp).astype(g_out_ref.dtype)
        dbar_ref[target, :] = _dbar_row(dxb, dyb, dzb, valid, dbar_ref.dtype)


def _l2l_rev_level_kernel(
    g_ref: KernelRef,
    _dbar_in_ref: KernelRef,
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
    """One parent of the level: gather its two children's local cotangents onto its own.

    Parameters
    ----------
    g_ref : KernelRef
        Cotangent table ``[nodes + 1, Bp*Wp]`` (aliased to the first output);
        the children's rows are final, the parent's is read and rewritten.
    _dbar_in_ref : KernelRef
        Geometry-cotangent table ``[nodes + 1, 4]`` (aliased; write-only).
    rows_ref : KernelRef
        The forward's OUTPUT locals in centred rows (the parent row is the
        linearisation point of both edges).
    cent_ref : KernelRef
        Padded centres ``[nodes + 1, 4]``.
    left_ref : KernelRef
        Left child per internal node.
    right_ref : KernelRef
        Right child per internal node.
    nbl_ref : KernelRef
        ``nodes_by_level`` padded with ``-1``.
    start_ref : KernelRef
        This level's start ``[1]``.
    count_ref : KernelRef
        This level's node count ``[1]``.
    *table_and_out_refs : KernelRef
        Constant tables, then the two aliased outputs (``g``, ``dbar``).
    bp : int
        ``Bp``. Static.
    wp : int
        ``Wp``. Static.
    num_internal : int
        Internal node count. Static.

    Returns
    -------
    None
        Writes the parent's cotangent row and both children's edge cotangents.
    """
    n_tables = len(_TABLE_KEYS) + len(_CORE_KEYS)
    table_refs = table_and_out_refs[:n_tables]
    g_out_ref, dbar_ref = table_and_out_refs[n_tables:]
    slot = pl.program_id(0)
    start = start_ref[0]
    count = count_ref[0]

    @pl.when(slot < count)
    def _live():
        t = {k: ref[...] for k, ref in zip((*_TABLE_KEYS, *_CORE_KEYS), table_refs)}
        node = nbl_ref[start + slot]
        valid = (node >= 0) & (node < num_internal)
        node_safe = jnp.where(valid, node, 0)
        rows_p = rows_ref[node_safe, :].reshape(bp, wp)
        px = cent_ref[node_safe, 0]
        py = cent_ref[node_safe, 1]
        pz = cent_ref[node_safe, 2]
        dead = jnp.asarray(g_out_ref.shape[0] - 1, dtype=node.dtype)
        acc = jnp.zeros((bp, wp), dtype=g_out_ref.dtype)

        def translate(r: Array, d: tuple) -> Array:
            return _translate_rows(r, d, t, "l2l", safe=True)

        for child_ref in (left_ref, right_ref):
            c = child_ref[node_safe]
            c_valid = valid & (c >= 0)
            c_safe = jnp.where(c_valid, c, 0)
            g_c = g_ref[c_safe, :].reshape(bp, wp)
            # L2L delta = parent - child
            dx = px - cent_ref[c_safe, 0]
            dy = py - cent_ref[c_safe, 1]
            dz_ = pz - cent_ref[c_safe, 2]
            _, vjp = jax.vjp(translate, rows_p, (dx, dy, dz_))
            rb, (dxb, dyb, dzb) = vjp(g_c)
            acc = acc + jnp.where(c_valid, rb, 0.0)
            ctarget = jnp.where(c_valid, c_safe, dead)
            dbar_ref[ctarget, :] = _dbar_row(dxb, dyb, dzb, c_valid, dbar_ref.dtype)
        own = g_ref[node_safe, :].reshape(bp, wp)
        target = jnp.where(valid, node_safe, dead)
        g_out_ref[target, :] = (own + acc).reshape(bp * wp).astype(g_out_ref.dtype)


def _level_call2(
    kernel: Any,
    operands: list,
    *,
    num_programs: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    name: str,
) -> tuple[Array, Array]:
    """One reverse level launch with two in-place outputs.

    Parameters
    ----------
    kernel : Any
        The level kernel (a ``functools.partial`` of a ``_*_rev_level_kernel``).
    operands : list
        Whole-array operands; ``operands[0]`` (``g``) and ``operands[1]``
        (``dbar``) are aliased to the two outputs, so rows a program does not
        write keep their values.
    num_programs : int
        Programs per launch (the level batch width).
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.
    name : str
        Kernel name.

    Returns
    -------
    tuple[Array, Array]
        The updated ``(g, dbar)``.
    """
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )
    g, dbar = operands[0], operands[1]
    return pl.pallas_call(
        kernel,
        grid=(int(num_programs),),
        in_specs=[_full(a) for a in operands],
        out_specs=[_full(g), _full(dbar)],
        out_shape=[
            jax.ShapeDtypeStruct(g.shape, g.dtype),
            jax.ShapeDtypeStruct(dbar.shape, dbar.dtype),
        ],
        input_output_aliases={0: 0, 1: 1},
        interpret=bool(interpret),
        name=name,
        **backend_kwargs,
    )(*operands)


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
    """Adjoint of :func:`m2m_real_levels_pallas`: one Pallas launch per level, top down.

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
        Programs per level. Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.

    Returns
    -------
    tuple[Array, Array]
        ``(packed_bar, centers_bar)``: the cotangent of the INPUT table (leaf
        rows; the internal rows of the input are overwritten by the forward and
        get zero) and of the centres.
    """
    p = int(order)
    t = cascade_level_tables(p)
    Bp, Wp = t["Bp"], t["Wp"]
    dtype = out_packed.dtype
    total = int(out_packed.shape[0])
    if int(num_internal) <= 0:
        return jnp.asarray(out_bar, dtype), jnp.zeros_like(centers)
    tables = _core_tables_to_jnp(p, dtype)
    table_arrays = [tables[k] for k in (*_TABLE_KEYS, *_CORE_KEYS)]
    dead = jnp.zeros((1, Bp * Wp), dtype)
    rows = jnp.concatenate([pack_centred(out_packed, order=p), dead], axis=0)
    g = jnp.concatenate(
        [pack_centred(jnp.asarray(out_bar, dtype), order=p), dead], axis=0
    )
    dbar = jnp.zeros((total + 1, 4), dtype)
    cent = jnp.pad(jnp.asarray(centers, dtype), ((0, 1), (0, 1)))
    width = int(max(level_batch_width, 1))
    idx = level_offsets.dtype
    nbl = jnp.concatenate(
        [jnp.asarray(nodes_by_level, idx), jnp.full((width,), -1, idx)]
    )
    offs = jnp.asarray(level_offsets, idx)
    par = jnp.asarray(parent, idx)
    kernel = functools.partial(_m2m_rev_level_kernel, bp=Bp, wp=Wp)

    def body(level: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        g_state, dbar_state = carry
        start = offs[level][None]
        count = (offs[level + 1] - offs[level])[None]
        return _level_call2(
            kernel,
            [g_state, dbar_state, rows, cent, par, nbl, start, count, *table_arrays],
            num_programs=width,
            interpret=interpret,
            backend=backend,
            num_warps=num_warps,
            name=f"m2m_rev_real_level_p{p}",
        )

    # the forward filled child levels 1 .. num_levels-1 from their parents; the
    # cotangent flows the other way, parent -> child, shallowest first
    g, dbar = lax.fori_loop(1, max(int(num_levels), 1), body, (g, dbar), unroll=True)
    packed_bar = unpack_centred(g[:total], order=p).at[: int(num_internal)].set(0.0)
    centers_bar = _fold_edge_cotangents(
        dbar[:total, :3], par, sign=1.0, dtype=centers.dtype
    )
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
    """Adjoint of :func:`l2l_real_levels_pallas`: one Pallas launch per level, bottom up.

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
        Programs per level. Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.

    Returns
    -------
    tuple[Array, Array]
        ``(locals_bar, centers_bar)``: the cotangent of the INPUT table (every
        row, since ``out[n] = in[n] + L2L(out[parent])``) and of the centres.
    """
    p = int(order)
    t = cascade_level_tables(p)
    Bp, Wp = t["Bp"], t["Wp"]
    dtype = out_locals.dtype
    total = int(out_locals.shape[0])
    num_internal = int(left_child.shape[0])
    if num_internal <= 0 or int(num_levels) <= 1:
        return jnp.asarray(out_bar, dtype), jnp.zeros_like(centers)
    tables = _core_tables_to_jnp(p, dtype)
    table_arrays = [tables[k] for k in (*_TABLE_KEYS, *_CORE_KEYS)]
    dead = jnp.zeros((1, Bp * Wp), dtype)
    rows = jnp.concatenate([pack_centred(out_locals, order=p), dead], axis=0)
    g = jnp.concatenate(
        [pack_centred(jnp.asarray(out_bar, dtype), order=p), dead], axis=0
    )
    dbar = jnp.zeros((total + 1, 4), dtype)
    cent = jnp.pad(jnp.asarray(centers, dtype), ((0, 1), (0, 1)))
    width = int(max(level_batch_width, 1))
    idx = level_offsets.dtype
    nbl = jnp.concatenate(
        [jnp.asarray(nodes_by_level, idx), jnp.full((width,), -1, idx)]
    )
    offs = jnp.asarray(level_offsets, idx)
    kernel = functools.partial(
        _l2l_rev_level_kernel, bp=Bp, wp=Wp, num_internal=num_internal
    )
    lc = jnp.asarray(left_child, idx)
    rc = jnp.asarray(right_child, idx)

    def body(rev: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        g_state, dbar_state = carry
        level = (int(num_levels) - 2) - rev  # parent level, deepest first
        start = offs[level][None]
        count = (offs[level + 1] - offs[level])[None]
        return _level_call2(
            kernel,
            [g_state, dbar_state, rows, cent, lc, rc, nbl, start, count, *table_arrays],
            num_programs=width,
            interpret=interpret,
            backend=backend,
            num_warps=num_warps,
            name=f"l2l_rev_real_level_p{p}",
        )

    g, dbar = lax.fori_loop(
        0, max(int(num_levels) - 1, 0), body, (g, dbar), unroll=True
    )
    locals_bar = unpack_centred(g[:total], order=p)
    centers_bar = _fold_edge_cotangents(
        dbar[:total, :3], jnp.asarray(parent, idx), sign=-1.0, dtype=centers.dtype
    )
    return locals_bar, centers_bar


# ------------------------------------------------------------- custom_vjp seams
# ``pallas_call`` has no autodiff rule (its generic JVP rule dies on
# ``program_id``), so the cascades cross the gradient path through these. The
# forward IS the production launch sequence; the reverse is the kernels above.
# Statics ride in ``nondiff_argnums``; the integer topology arrays are ordinary
# arguments whose cotangents are ``None`` (symbolic zero).


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
    """Differentiable :func:`m2m_real_levels_pallas` (forward byte-identical).

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
        Warps per program. ``nondiff_argnums``.

    Returns
    -------
    Array
        ``[nodes, C]`` packed multipoles with the internal nodes filled.
    """
    return m2m_real_levels_pallas(
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
    out = m2m_real_levels_pallas(
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
    """Differentiable :func:`l2l_real_levels_pallas` (forward byte-identical).

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
        Warps per program. ``nondiff_argnums``.

    Returns
    -------
    Array
        ``[nodes, C]`` fully cascaded locals.
    """
    return l2l_real_levels_pallas(
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
    out = l2l_real_levels_pallas(
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
