"""Real-basis M2L over a far-pair list sorted by target (CSR): the shared algebra and helpers.

The CSR Pallas M2L kernel is :mod:`jaccpot.pallas.m2l_real_csr_lanes` (one pair
per lane, the default since plan sub-10ms Phase 6). This module holds what it,
the cascade kernels and the tests share:

* the hardware gate :func:`pallas_m2l_real_csr_supported` (the real Pallas
  kernels' common sm_80 predicate);
* the CSR helpers :func:`csr_by_target` and :func:`targets_from_csr_offsets`;
* the CENTRED padded layout (:func:`pack_centred` / :func:`unpack_centred`) and
  the per-order constant tables (:func:`m2l_real_csr_tables`);
* the per-pair rotate -> z-translate -> rotate-back M2L in that layout
  (``_m2l_pair_rows``), its pure-jnp twin :func:`m2l_real_csr_pair_jax` and the
  per-target reference :func:`m2l_real_csr_jax`, which the lane kernel's tests
  and its custom VJP compare against.

The first kernel built on this algebra ran one program per TARGET node, looping
over the target's CSR segment with the pair's ``(Bp, Wp)`` tile spread over the
program's threads (24 ns per pair on an A100 at order 5); a K-source tiled
variant followed (8.5-11.7 ns). The lane kernel runs 1.3 ns per pair
(``docs/sub10ms_2026-09.md``, Phase 5), and both were removed in the 2026-10
cleanup (X5).

Arithmetic, per pair, in the CENTRED padded layout (degree ``l`` occupies
columns ``p-l .. p+l`` of a width ``2p+1`` row, so ``m`` sits at column ``p+m``
for every degree at once; :func:`pack_centred`):

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
  ``(Cp,)`` power vectors (``exp(deg * log rinv)``), not ``Cp^2`` transcendental
  calls.
* z -> world local block = transpose of the multipole block
  (:func:`jaccpot.operators.real_rotations.real_rotation_from_z_axis_local`):
  ``Dz(-az) B_l^T Dz(ax) B_l^T``.

The z-core preserves ``m`` and is a ``Bp x Bp`` degree operator per column.
Every contraction is a broadcast-multiply + ``jnp.sum`` (no ``dot``, no TF32), as
in :mod:`jaccpot.pallas.m2l_real_fused`. Padded lanes of every constant are
exactly zero, so they are inert in every reduction.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaccpot._searchsorted import searchsorted_method
from jaccpot.operators.real_dehnen_q import compute_real_B_matrix_multipole
from jaccpot.operators.real_harmonics import (
    sh_offset,
    sh_size,
    z_m2l_translation_tables,
)
from jaccpot.pallas.m2l_real_fused import pallas_m2l_real_fused_supported

__all__ = [
    "pallas_m2l_real_csr_supported",
    "m2l_real_csr_tables",
    "m2l_real_csr_pair_jax",
    "m2l_real_csr_jax",
    "csr_by_target",
    "targets_from_csr_offsets",
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
        Whether the Triton lowering of the real Pallas kernels can run here.
    """
    return pallas_m2l_real_fused_supported()


def _next_pow2(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


@functools.lru_cache(maxsize=None)
def m2l_real_csr_tables(order: int) -> dict:
    """Compile-time constants of the centred-layout M2L for one expansion order.

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

    Raises
    ------
    ValueError
        If ``order`` is negative.
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
    # packed_idx is a permutation of range(C): invert it and gather, rather
    # than scatter into zeros (an XLA scatter kernel per call; four of them sat
    # in the gradient's kernel table at ~0.2 ms each)
    slot_of_coeff = np.empty(int(t["C"]), dtype=np.int64)
    slot_of_coeff[packed_idx] = flat_slots
    return rows[:, slot_of_coeff]


# --------------------------------------------------------------------------- math
# Every helper below is written for BOTH Pallas kernels (values loaded from refs:
# the cascade reverse kernels use ``_bapply`` and ``_dz``) and the pure-jnp twin:
# plain broadcast-multiply + sum, no dot, no gather.


def _bapply(bstack: Array, rows: Array) -> Array:
    """Apply the per-degree constant blocks: ``out[l, i] = sum_j bstack[l, i, j] rows[l, j]``.

    Parameters
    ----------
    bstack : Array
        Block-diagonal-by-degree operator stack, ``(Bp, Wp, Wp)``.
    rows : Array
        Centred coefficient rows, ``(Bp, Wp)``.

    Returns
    -------
    Array
        ``(Bp, Wp)``.
    """
    return jnp.sum(bstack * rows[:, None, :], axis=-1)


def _dz(rows: Array, cosv: Array, sinv: Array, apat: Array) -> Array:
    """``Dz(t)`` on every degree row at once: ``cos(|m|t) v + A (sin(|m|t) v)``.

    Parameters
    ----------
    rows : Array
        Centred coefficient rows, ``(Bp, Wp)``.
    cosv : Array
        ``cos(|m| t)`` per column, ``(Wp,)``.
    sinv : Array
        ``sin(|m| t)`` per column, ``(Wp,)``; pass its negative for ``Dz(-t)``.
    apat : Array
        The constant antisymmetric sine pattern, ``(Wp, Wp)``.

    Returns
    -------
    Array
        ``(Bp, Wp)``.
    """
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
    """Per-pair local contributions in the centred-layout arithmetic (the twin).

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

    def one(rw: Array, dd: Array) -> Array:
        return _m2l_pair_rows(rw, (dd[0], dd[1], dd[2]), t)

    out_rows = jax.vmap(one)(rows, d).reshape(-1, Bp * Wp)
    return unpack_centred(out_rows, order=int(order))


def csr_by_target(
    sources: Array,
    targets: Array,
    *,
    total_nodes: int,
    active_pair_count: Optional[Array] = None,
    presorted: bool = False,
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
    presorted : bool
        The list is already in CSR order (the deterministic flat walk's
        ``TargetSortedFarPairs``) and ``targets`` holds its ROW OFFSETS
        (``[>= total_nodes + 1]``), not one target per entry: no sort, no copy of
        the sources (the kernels read only ``[offsets[t], offsets[t + 1])``, so the
        padding behind the live prefix is never addressed) and
        ``active_pair_count`` is unused. Static.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(sources_sorted [P], offsets [total_nodes], counts [total_nodes])``:
        target ``t``'s sources are ``sources_sorted[offsets[t] : offsets[t] +
        counts[t]]``. Padding sorts to the end and is never addressed.

    Raises
    ------
    ValueError
        If ``presorted`` and ``targets`` holds fewer than ``total_nodes + 1``
        row offsets.
    """
    src = jnp.asarray(sources, dtype=jnp.int32)
    tgt = jnp.asarray(targets, dtype=jnp.int32)
    if presorted:
        # targets are the row offsets: the CSR is the list itself. (A copy of the
        # sources with the padding zeroed was a 2P array live through the M2L and
        # the L2L, the step's second-highest window.)
        nt = int(total_nodes)
        if int(tgt.shape[0]) < nt + 1:
            raise ValueError(
                f"presorted targets must be the {nt + 1} row offsets, got "
                f"{int(tgt.shape[0])} entries"
            )
        row_offsets = tgt[: nt + 1]
        return src, row_offsets[:-1], (row_offsets[1:] - row_offsets[:-1])
    P = int(src.shape[0])
    valid = (src >= 0) & (tgt >= 0)
    if active_pair_count is not None:
        valid = valid & (
            jnp.arange(P, dtype=jnp.int32) < jnp.asarray(active_pair_count, jnp.int32)
        )
    key = jnp.where(valid, tgt, jnp.asarray(total_nodes, jnp.int32))
    # One stable key-value sort carries the sources along: no permutation, no
    # gathers. (An argsort here also carried an int64 iota through this
    # every-step sort -- yggdrax enables x64 -- twice the bytes per pass.)
    sorted_key, src_sorted = jax.lax.sort(
        (key, jnp.where(valid, src, 0)), num_keys=1, is_stable=True
    )
    # offsets from the SORTED key (a searchsorted), not a scatter-add over the
    # P entries: that segment_sum was a 1.3 ms int32 scatter at P = 2^21
    # unrolled: the default is a while loop with one small kernel per bisection
    # step (~23 here), launch-bound inside the fused step; unrolled, XLA fuses it
    offsets = jnp.searchsorted(
        sorted_key,
        jnp.arange(int(total_nodes) + 1, dtype=jnp.int32),
        side="left",
        method=searchsorted_method(),
    ).astype(jnp.int32)
    counts = offsets[1:] - offsets[:-1]
    return src_sorted, offsets[:-1], counts.astype(jnp.int32)


def targets_from_csr_offsets(row_offsets: Array, num_entries: int) -> Array:
    """The target of every entry of a CSR list, from its row offsets.

    The inverse of storing a target-sorted list as ``(sources, row offsets)``:
    entry ``j`` belongs to the last row whose offset is ``<= j``. Entries past
    the live prefix (``j >= row_offsets[-1]``) get ``-1``, the padding value of a
    COO far list.

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
    pos = jnp.arange(int(num_entries), dtype=off.dtype)
    row = jnp.searchsorted(off, pos, side="right", method=searchsorted_method()).astype(
        off.dtype
    ) - jnp.asarray(1, off.dtype)
    return jnp.where(pos < off[-1], row, jnp.asarray(-1, off.dtype))


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
