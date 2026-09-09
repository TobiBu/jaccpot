"""Target-tiled real-basis M2L Pallas kernel with on-chip rotations (CSR by target).

One program per TARGET node. The program loops over that target's far-pair
segment of a source list sorted by target (CSR), and for every source builds the
whole rotate -> z-translate -> rotate-back M2L on chip from two angles, summing
into one register-resident local row. It exists because the chunked pure-JAX
lane (``runtime/kernels/_m2l.py``) is launch-bound: at N=200k, leaf 64 the far
field is a 21k-launch/step storm (~130 ns per directed pair with nothing above
27 ms in the kernel table), and the two shipped fused Pallas M2L shapes lose
because they take the world<->z rotation blocks as ``(pairs, p+1, 2p+1, 2p+1)``
HBM operands (32 KB per pair). Here nothing per pair is materialised: the
program owns its output row, so there is no ``segment_sum`` scatter and no
argsort per chunk, and the coefficient traffic is one ``(Cp,)`` row load per pair.

Arithmetic, per pair, in the CENTRED padded layout of
:func:`jaccpot.operators.m2l_real_rot_scale._centred_degree_maps` (degree ``l``
occupies columns ``p-l .. p+l`` of a width ``2p+1`` row, so ``m`` sits at column
``p+m`` for every degree at once):

* world -> z multipole block, degree ``l``: ``B_l Dz(-ax) B_l Dz(az)`` with
  ``az = atan2(x, y)``, ``ax = atan2(rho, z)`` (the conventions of
  :func:`jaccpot.operators.real_rotations._multipole_align_to_z_block`).
  ``B_l`` is a compile-time constant stack; ``Dz(t) v = cos(|m| t) * v +
  A (sin(|m| t) * v)`` with ``A`` the constant antisymmetric pattern
  ``A[p+m, p-m] = -1, A[p-m, p+m] = +1`` -- read off
  :func:`jaccpot.operators.real_rotations.real_Dz_diagonal`.
* z-core: ``F_n^m = sum_k sign(m) (n+k)! r^-(n+k+1) M_k^m`` from
  :func:`jaccpot.operators.real_harmonics.z_m2l_translation_tables`, the single
  source of truth, as the SEPARABLE dense form
  ``Z = Zsf * outer(rinv^(n+1), rinv^k)`` so the radius enters through two
  ``(Cp,)`` power vectors (``exp(deg * log rinv)``, the form the fused kernel
  lowers), not ``Cp^2`` transcendental calls.
* z -> world local block = transpose of the multipole block
  (:func:`jaccpot.operators.real_rotations.real_rotation_from_z_axis_local`):
  ``Dz(-az) B_l^T Dz(ax) B_l^T``.

The kernel works entirely in the centred ``(Bp, Wp)`` layout: the wrapper packs
the multipole table into it once (an XLA gather over ``n`` rows) and unpacks the
output once, so no per-pair pack/unpack matvec and no ``Cp x Cp`` table exist in
the kernel -- the z-core preserves ``m`` and is a ``Bp x Bp`` degree operator
per column. Every contraction is a broadcast-multiply + ``jnp.sum`` (no ``dot``,
no TF32), as in :mod:`jaccpot.pallas.m2l_real_fused`. Padded lanes of every
constant are exactly zero, so they are inert in every reduction.

Forward only (the fused strict lane is forward-only); the pure-JAX lane stays the
differentiable path, and the transverse-degeneracy JVP treatment of
``m2l_rot_scale_real_batch`` is not needed here.

HARDWARE: real Pallas GPU execution needs Ampere (sm_80+); ``interpret=True``
runs the same arithmetic on CPU.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jaxtyping import Array

from jaccpot.operators.real_dehnen_q import compute_real_B_matrix_multipole
from jaccpot.operators.real_harmonics import (
    sh_offset,
    sh_size,
    z_m2l_translation_tables,
)
from jaccpot.pallas._compat import KernelRef, pallas_backend_kwargs
from jaccpot.pallas.m2l_real_fused import pallas_m2l_real_fused_supported

__all__ = [
    "pallas_m2l_real_csr_supported",
    "m2l_real_csr_tables",
    "m2l_real_csr_pair_jax",
    "m2l_real_csr_jax",
    "m2l_real_csr_pallas",
    "csr_by_target",
    "pack_centred",
    "unpack_centred",
]

_TABLE_KEYS = (
    "Bstack",
    "BstackT",
    "Apat",
    "mabs",
    "signm",
    "Zf",
    "degn",
    "degk",
)


def pallas_m2l_real_csr_supported() -> bool:
    """True on a GPU with compute capability >= 8.0 (same predicate as the fused kernel).

    Returns
    -------
    bool
        Whether the Triton lowering of this kernel can run here.
    """
    return pallas_m2l_real_fused_supported()


def _next_pow2(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


@functools.lru_cache(maxsize=None)
def m2l_real_csr_tables(order: int) -> dict:
    """Compile-time constants of the kernel for one expansion order.

    Everything lives in the CENTRED ``(Bp, Wp)`` layout: row = degree ``l``,
    column ``p + m``; degrees ``> p`` and columns with ``|m| > p`` are padding
    and every constant is exactly zero there.

    Parameters
    ----------
    order : int
        Expansion order ``p``.

    Returns
    -------
    dict
        NumPy float64 arrays (cast to the working dtype by the caller) plus the
        shape scalars ``C = (p+1)^2``, ``W = 2p+1``, ``Wp`` (pow2 >= W), ``Bp``
        (pow2 >= p+1), and the pack/unpack maps ``idx [Bp, Wp]`` (packed
        coefficient index of each centred slot) and ``mask [Bp, Wp]``.

        ``Bstack [Bp, Wp, Wp]``: ``B_U(l)`` centred per degree; ``BstackT`` its
        per-degree transpose. ``Apat [Wp, Wp]``: the Dz sine pattern.
        ``mabs [Wp]``: ``|m|`` per column. ``signm [Wp]``: the z-core's
        ``sign(m) = (-1)^m (2 if m != 0 else 1)`` per column.
        ``Zf [Bp, Bp]``: ``(n+k)!`` where ``k <= p - n``, else 0 -- the z-core
        preserves ``m``, so it is a degree x degree operator per column.
        ``degn [Bp]``: ``n + 1`` (radius exponent of the output degree);
        ``degk [Bp]``: ``k`` (radius exponent of the source degree).
    """
    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    C = sh_size(p)
    W = 2 * p + 1
    Wp = _next_pow2(W)
    Bp = _next_pow2(p + 1)

    idx = np.zeros((Bp, Wp), dtype=np.int32)
    mask = np.zeros((Bp, Wp), dtype=bool)
    for ell in range(p + 1):
        for m in range(-ell, ell + 1):
            idx[ell, p + m] = sh_offset(ell) + ell + m
            mask[ell, p + m] = True

    Bstack = np.zeros((Bp, Wp, Wp), dtype=np.float64)
    # `compute_real_B_matrix_multipole` is jitted: evaluated eagerly here so the
    # table is a literal even when this cache is first filled inside a trace.
    with jax.ensure_compile_time_eval():
        for ell in range(p + 1):
            b = np.asarray(
                compute_real_B_matrix_multipole(ell, dtype=jnp.float64),
                dtype=np.float64,
            )
            Bstack[ell, p - ell : p + ell + 1, p - ell : p + ell + 1] = b
    BstackT = np.swapaxes(Bstack, -1, -2).copy()

    Apat = np.zeros((Wp, Wp), dtype=np.float64)
    mabs = np.zeros((Wp,), dtype=np.float64)
    signm = np.zeros((Wp,), dtype=np.float64)
    signm[p] = 1.0
    for m in range(1, p + 1):
        Apat[p + m, p - m] = -1.0
        Apat[p - m, p + m] = 1.0
        mabs[p + m] = float(m)
        mabs[p - m] = float(m)
        signm[p + m] = signm[p - m] = (-1.0 if (m % 2) else 1.0) * 2.0

    # z-core in the centred layout, cross-checked against the single source of truth
    src_index, valid, fact_index, r_exponent, sign = z_m2l_translation_tables(p)
    fact = np.asarray([math.factorial(i) for i in range(2 * p + 1)], dtype=np.float64)
    Zf = np.zeros((Bp, Bp), dtype=np.float64)
    for n in range(p + 1):
        for k in range(p - n + 1):
            Zf[n, k] = fact[n + k]
    for n in range(p + 1):
        for m in range(-n, n + 1):
            out = sh_offset(n) + n + m
            assert abs(sign[out] - signm[p + m]) < 1e-12
            for k in range(p + 1):
                if bool(valid[out, k]):
                    assert int(src_index[out, k]) == sh_offset(k) + k + m  # same m
                    assert int(r_exponent[out, k]) == n + k + 1
                    assert fact[int(fact_index[out, k])] == Zf[n, k]
                else:
                    assert k < abs(m) or k > p - n
    degn = np.zeros((Bp,), dtype=np.float64)
    degk = np.zeros((Bp,), dtype=np.float64)
    degn[: p + 1] = np.arange(p + 1) + 1
    degk[: p + 1] = np.arange(p + 1)
    return dict(
        p=p,
        C=C,
        W=W,
        Wp=Wp,
        Bp=Bp,
        idx=idx,
        mask=mask,
        Bstack=Bstack,
        BstackT=BstackT,
        Apat=Apat,
        mabs=mabs,
        signm=signm,
        Zf=Zf,
        degn=degn,
        degk=degk,
    )


def _tables_to_jnp(order: int, dtype: Any) -> dict[str, Array]:
    t = m2l_real_csr_tables(order)
    return {k: jnp.asarray(t[k], dtype=dtype) for k in _TABLE_KEYS}


def pack_centred(coeffs: Array, *, order: int) -> Array:
    """``[N, C]`` packed coefficients -> ``[N, Bp*Wp]`` centred rows (zeros on padding).

    Parameters
    ----------
    coeffs : Array
        Packed coefficients, ``[N, (p+1)^2]``.
    order : int
        Expansion order. Static.

    Returns
    -------
    Array
        ``[N, Bp*Wp]``.
    """
    t = m2l_real_csr_tables(int(order))
    idx = jnp.asarray(t["idx"])
    mask = jnp.asarray(t["mask"])
    rows = jnp.where(mask[None], coeffs[:, idx], jnp.zeros((), coeffs.dtype))
    return rows.reshape(coeffs.shape[0], t["Bp"] * t["Wp"])


def unpack_centred(rows: Array, *, order: int) -> Array:
    """Inverse of :func:`pack_centred`: ``[N, Bp*Wp]`` -> ``[N, C]``.

    Parameters
    ----------
    rows : Array
        Centred rows, ``[N, Bp*Wp]``.
    order : int
        Expansion order. Static.

    Returns
    -------
    Array
        ``[N, (p+1)^2]`` packed coefficients.
    """
    t = m2l_real_csr_tables(int(order))
    mask = np.asarray(t["mask"])
    flat_slots = np.nonzero(mask.reshape(-1))[0]
    packed_idx = np.asarray(t["idx"]).reshape(-1)[flat_slots]
    out = jnp.zeros((rows.shape[0], t["C"]), dtype=rows.dtype)
    return out.at[:, packed_idx].set(rows[:, flat_slots])


# --------------------------------------------------------------------------- math
# Every helper below is written for BOTH the Pallas kernel (values loaded from
# refs) and the pure-jnp twin: plain broadcast-multiply + sum, no dot, no gather.


def _bapply(bstack: Array, rows: Array) -> Array:
    """``out[l, i] = sum_j bstack[l, i, j] rows[l, j]`` (block-diagonal by degree)."""
    return jnp.sum(bstack * rows[:, None, :], axis=-1)


def _dz(rows: Array, cosv: Array, sinv: Array, apat: Array) -> Array:
    """``Dz(t)`` on every degree row at once: ``cos(|m|t) v + A (sin(|m|t) v)``."""
    sv = rows * sinv[None, :]
    return rows * cosv[None, :] + jnp.sum(apat[None, :, :] * sv[:, None, :], axis=-1)


def _m2l_pair_rows(rows: Array, delta3: tuple, t: dict[str, Array]) -> Array:
    """Full real M2L for one pair in the centred layout.

    Parameters
    ----------
    rows : Array
        Source multipole in centred rows, ``(Bp, Wp)``.
    delta3 : tuple
        ``(x, y, z)`` scalars, target centre minus source centre.
    t : dict[str, Array]
        Tables from :func:`_tables_to_jnp` at the working dtype.

    Returns
    -------
    Array
        Local contribution in centred rows, ``(Bp, Wp)``.
    """
    x, y, z = delta3
    dtype = rows.dtype
    rho2 = x * x + y * y
    rho = jnp.sqrt(rho2)
    az = jnp.arctan2(x, y)
    ax = jnp.arctan2(rho, z)
    r = jnp.sqrt(rho2 + z * z)
    r = jnp.maximum(r, jnp.asarray(1.0e-30, dtype=dtype))
    log_rinv = -jnp.log(r)
    mabs = t["mabs"]
    cos_az = jnp.cos(mabs * az)
    sin_az = jnp.sin(mabs * az)
    cos_ax = jnp.cos(mabs * ax)
    sin_ax = jnp.sin(mabs * ax)
    apat = t["Apat"]

    # world -> z (multipole): B Dz(-ax) B Dz(az), applied right to left
    v = _dz(rows, cos_az, sin_az, apat)
    v = _bapply(t["Bstack"], v)
    v = _dz(v, cos_ax, -sin_ax, apat)
    v = _bapply(t["Bstack"], v)

    # z-core: same m, degree x degree; r^-(n+k+1) = rinv^(n+1) rinv^k
    rinv_n = jnp.exp(t["degn"] * log_rinv)  # (Bp,)
    rinv_k = jnp.exp(t["degk"] * log_rinv)  # (Bp,)
    v = v * rinv_k[:, None]
    v = jnp.sum(t["Zf"][:, :, None] * v[None, :, :], axis=1)  # (Bp, Wp)
    v = v * rinv_n[:, None] * t["signm"][None, :]

    # z -> world (local) = transpose of the multipole block: Dz(-az) B^T Dz(ax) B^T
    v = _bapply(t["BstackT"], v)
    v = _dz(v, cos_ax, sin_ax, apat)
    v = _bapply(t["BstackT"], v)
    return _dz(v, cos_az, -sin_az, apat)


# ----------------------------------------------------------------- pure-jnp twin


def m2l_real_csr_pair_jax(multipoles: Array, deltas: Array, *, order: int) -> Array:
    """Per-pair local contributions with this kernel's arithmetic (the twin).

    Parameters
    ----------
    multipoles : Array
        ``[N, C]`` source multipoles.
    deltas : Array
        ``[N, 3]`` target minus source centres.
    order : int
        Expansion order ``p``. Static.

    Returns
    -------
    Array
        ``[N, C]`` local contributions, one per pair (NOT reduced by target).
    """
    tb = m2l_real_csr_tables(int(order))
    Bp, Wp = tb["Bp"], tb["Wp"]
    mult = jnp.asarray(multipoles)
    dtype = mult.dtype
    t = _tables_to_jnp(int(order), dtype)
    rows = pack_centred(mult, order=int(order)).reshape(-1, Bp, Wp)
    d = jnp.asarray(deltas, dtype=dtype)

    def one(rw, dd):
        return _m2l_pair_rows(rw, (dd[0], dd[1], dd[2]), t)

    out_rows = jax.vmap(one)(rows, d).reshape(-1, Bp * Wp)
    return unpack_centred(out_rows, order=int(order))


def csr_by_target(
    sources: Array,
    targets: Array,
    *,
    total_nodes: int,
    active_pair_count: Optional[Array] = None,
) -> tuple[Array, Array, Array]:
    """Sort a (padded) flat far-pair list by target into CSR form.

    Parameters
    ----------
    sources : Array
        ``[P]`` source node ids; ``-1`` (or anything negative) marks padding.
    targets : Array
        ``[P]`` target node ids, aligned; negative marks padding.
    total_nodes : int
        Number of target rows. Static.
    active_pair_count : Optional[Array]
        Number of leading live entries; ``None`` means every non-negative entry
        is live.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(sources_sorted [P], offsets [total_nodes], counts [total_nodes])``:
        target ``t``'s sources are ``sources_sorted[offsets[t] : offsets[t] +
        counts[t]]``. Padding sorts to the end and is never addressed.
    """
    src = jnp.asarray(sources, dtype=jnp.int32)
    tgt = jnp.asarray(targets, dtype=jnp.int32)
    P = int(src.shape[0])
    valid = (src >= 0) & (tgt >= 0)
    if active_pair_count is not None:
        valid = valid & (
            jnp.arange(P, dtype=jnp.int32) < jnp.asarray(active_pair_count, jnp.int32)
        )
    key = jnp.where(valid, tgt, jnp.asarray(total_nodes, jnp.int32))
    perm = jnp.argsort(key, stable=True)
    src_sorted = jnp.where(valid[perm], src[perm], 0)
    counts = jax.ops.segment_sum(
        valid.astype(jnp.int32), jnp.where(valid, tgt, 0), num_segments=int(total_nodes)
    )
    offsets = jnp.cumsum(counts) - counts
    return src_sorted, offsets.astype(jnp.int32), counts.astype(jnp.int32)


def m2l_real_csr_jax(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    *,
    order: int,
    active_pair_count: Optional[Array] = None,
) -> Array:
    """Reference: per-pair twin reduced by target with ``segment_sum``.

    Parameters
    ----------
    multipoles : Array
        ``[n, C]`` node multipoles.
    centers : Array
        ``[n, 3]`` node centres.
    sources : Array
        ``[P]`` source ids (negative = padding).
    targets : Array
        ``[P]`` target ids (negative = padding).
    order : int
        Expansion order. Static.
    active_pair_count : Optional[Array]
        Live prefix length, as in :func:`csr_by_target`.

    Returns
    -------
    Array
        ``[n, C]`` local coefficient increments.
    """
    n = int(multipoles.shape[0])
    src = jnp.asarray(sources, jnp.int32)
    tgt = jnp.asarray(targets, jnp.int32)
    valid = (src >= 0) & (tgt >= 0)
    if active_pair_count is not None:
        valid = valid & (
            jnp.arange(src.shape[0], dtype=jnp.int32)
            < jnp.asarray(active_pair_count, jnp.int32)
        )
    s = jnp.where(valid, src, 0)
    tt = jnp.where(valid, tgt, 0)
    deltas = centers[tt] - centers[s]
    contrib = m2l_real_csr_pair_jax(multipoles[s], deltas, order=order)
    contrib = jnp.where(valid[:, None], contrib, 0)
    return jax.ops.segment_sum(contrib, tt, num_segments=n)


# -------------------------------------------------------------------- the kernel


def _m2l_real_csr_kernel(
    mult_ref: KernelRef,
    cent_ref: KernelRef,
    src_ref: KernelRef,
    off_ref: KernelRef,
    cnt_ref: KernelRef,
    *table_and_out_refs: KernelRef,
    bp: int,
    wp: int,
) -> None:
    """One program per target: loop over its CSR segment, accumulate one local row.

    Parameters
    ----------
    mult_ref : KernelRef
        Whole centred multipole table ``[n, Bp*Wp]`` (gathered by source id).
    cent_ref : KernelRef
        Whole padded centre table ``[n, 4]``.
    src_ref : KernelRef
        Whole target-sorted source list ``[P]``.
    off_ref : KernelRef
        Segment start per target ``[n]``.
    cnt_ref : KernelRef
        Segment length per target ``[n]``.
    *table_and_out_refs : KernelRef
        The ``_TABLE_KEYS`` constants (whole arrays) followed by the output ref
        ``[1, Bp*Wp]``.
    bp : int
        ``Bp``. Static.
    wp : int
        ``Wp``. Static.

    Returns
    -------
    None
        Writes the target's centred local row.
    """
    table_refs = table_and_out_refs[: len(_TABLE_KEYS)]
    (out_ref,) = table_and_out_refs[len(_TABLE_KEYS) :]
    t = {k: ref[...] for k, ref in zip(_TABLE_KEYS, table_refs)}
    tgt = pl.program_id(0)
    start = off_ref[tgt]
    cnt = cnt_ref[tgt]
    ctx = cent_ref[tgt, 0]
    cty = cent_ref[tgt, 1]
    ctz = cent_ref[tgt, 2]
    acc0 = jnp.zeros((bp, wp), dtype=out_ref.dtype)

    def body(k, acc):
        sid = src_ref[start + k]
        rows = mult_ref[sid, :].reshape(bp, wp)
        dx = ctx - cent_ref[sid, 0]
        dy = cty - cent_ref[sid, 1]
        dz_ = ctz - cent_ref[sid, 2]
        return acc + _m2l_pair_rows(rows, (dx, dy, dz_), t)

    acc = lax.fori_loop(0, cnt, body, acc0)
    out_ref[0, :] = acc.reshape(bp * wp)


def m2l_real_csr_pallas(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    *,
    order: int,
    active_pair_count: Optional[Array] = None,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 4,
) -> Array:
    """Local coefficient increments from a flat far-pair list, one Pallas program per target.

    Parameters
    ----------
    multipoles : Array
        ``[n, C]`` node multipoles (real basis).
    centers : Array
        ``[n, 3]`` node centres.
    sources : Array
        ``[P]`` source node ids; negative = padding.
    targets : Array
        ``[P]`` target node ids; negative = padding. Need NOT be sorted -- the
        list is sorted by target here (one argsort of ``P`` keys).
    order : int
        Expansion order ``p``. Static.
    active_pair_count : Optional[Array]
        Live prefix length of the padded list; ``None`` = all non-negative.
    interpret : bool
        Pallas interpret mode (CPU semantics).
    backend : str
        Pallas GPU lowering, ``"triton"`` by default.
    num_warps : int
        Warps per program. The working tile is ``(Bp, Wp)`` = 128 elements at
        p <= 7 and the two rotation stacks are 2 x ``Bp*Wp*Wp`` constants held in
        registers, so 4 warps (128 threads) keeps the per-thread register count
        low; 1 warp spilled.

    Returns
    -------
    Array
        ``[n, C]`` local increments, same dtype as ``multipoles``.
    """
    tb = m2l_real_csr_tables(int(order))
    C, Bp, Wp = tb["C"], tb["Bp"], tb["Wp"]
    mult = jnp.asarray(multipoles)
    dtype = mult.dtype
    n = int(mult.shape[0])
    if int(mult.shape[1]) != C:
        raise ValueError(f"multipoles must have {C} coefficients for order {order}")
    cent = jnp.asarray(centers, dtype=dtype)
    if cent.ndim != 2 or int(cent.shape[1]) != 3 or int(cent.shape[0]) != n:
        raise ValueError("centers must have shape (n, 3) aligned with multipoles")
    mult_c = pack_centred(mult, order=int(order))  # [n, Bp*Wp]
    cent_p = jnp.pad(cent, ((0, 0), (0, 1)))
    src_sorted, offsets, counts = csr_by_target(
        sources, targets, total_nodes=n, active_pair_count=active_pair_count
    )
    P = int(src_sorted.shape[0])
    if n == 0 or P == 0:
        return jnp.zeros((n, C), dtype=dtype)
    tables = _tables_to_jnp(int(order), dtype)
    table_arrays = [tables[k] for k in _TABLE_KEYS]

    def bs_full(arr: Array) -> pl.BlockSpec:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    kernel = functools.partial(_m2l_real_csr_kernel, bp=Bp, wp=Wp)
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )
    out_rows = pl.pallas_call(
        kernel,
        grid=(n,),
        in_specs=[
            bs_full(mult_c),
            bs_full(cent_p),
            bs_full(src_sorted),
            bs_full(offsets),
            bs_full(counts),
            *[bs_full(a) for a in table_arrays],
        ],
        out_specs=pl.BlockSpec((1, Bp * Wp), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((n, Bp * Wp), dtype),
        interpret=bool(interpret),
        **backend_kwargs,
        name=f"m2l_real_csr_p{int(order)}",
    )(mult_c, cent_p, src_sorted, offsets, counts, *table_arrays)
    return unpack_centred(out_rows, order=int(order))
