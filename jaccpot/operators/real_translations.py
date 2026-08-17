"""Real-basis z-axis translations and the assembled M2L / M2M / L2L operators.

Two layers, kept together because the second is only the first wrapped in the
rotations of :mod:`jaccpot.operators.real_rotations`:

* the **z-axis shifts**, where a translation along ``+z`` is a banded
  ``(dz)^k / k!`` sum per ``m`` channel (M2M / L2L) or the ``(-1)^k`` M2L
  variant, with the per-term structure precomputed host-side and cached;
* the **assembled operators** ``m2l_a6_real_only``, ``m2m_real`` and
  ``l2l_real``, each of which rotates the displacement onto ``+z``, applies the
  z-shift, and rotates back.

DO NOT RESTRUCTURE THE ALGEBRA. NUMERICS_AND_JAX §1: the
rotate -> z-translate -> rotate-back decomposition and the involutory
B-matrices are ordered for cancellation reasons, not aesthetics.

The three assembled operators carry a ``custom_jvp``
(:mod:`jaccpot.operators._transverse_degeneracy_jvp`) that supplies the
transverse derivative at ``rho == 0`` analytically; it sits inside the ``jax.jit``
so it adds no dispatch boundary, and outside ``highest_matmul_precision`` so the
pinned precision still covers the body.

Split out of ``real_harmonics.py`` (Tier 1.3); the mathematics is unchanged and
``real_harmonics`` re-exports the public names.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from ._precision import highest_matmul_precision
from ._sh_indexing import _factorial_table_jax, sh_index, sh_offset, sh_size
from ._transverse_degeneracy_jvp import with_transverse_degeneracy_jvp
from .dtypes import floor_squared_radius
from .real_rotations import (
    _L2L_TRANSVERSE_GENERATORS,
    _M2L_TRANSVERSE_GENERATORS,
    _M2M_TRANSVERSE_GENERATORS,
    real_rotation_from_z_axis_local,
    real_rotation_from_z_axis_multipole,
    real_rotation_to_z_axis_local,
    real_rotation_to_z_axis_multipole,
)

__all__ = [
    "z_shift_translation_tables",
    "translate_along_z_m2m_real",
    "z_m2l_translation_tables",
    "translate_along_z_m2l_real",
    "translate_along_z_l2l_real",
    "m2l_a6_real_only",
    "m2l_real",
    "m2l_optimized_real",
    "m2m_real",
    "l2l_real",
]


# ===========================================================================
# Z-axis translations in real basis
# ===========================================================================


@lru_cache(maxsize=None)
def z_shift_translation_tables(order: int, which: str) -> Tuple[np.ndarray, np.ndarray]:
    """Static per-term structure for the real z-axis M2M / L2L shift (same-type translate).

    Both the M2M (``M'_n^m = sum_k (dz)^k/k! * M_{n-k}^m``) and L2L
    (``F'_n^m = sum_k (dz)^k/k! * F_{n+k}^m``) z-shifts have the identical coefficient
    ``(dz)^k/k!`` per slot ``k`` (shared across all outputs); they differ ONLY in the source
    degree and the valid range:

    * ``which='m2m'``: source degree ``n - k``, valid when ``|m| <= n - k`` (k in ``0..n-|m|``);
    * ``which='l2l'``: source degree ``n + k``, valid when ``n + k <= p`` (k in ``0..p-n``).

    Returns ``(src_index, valid)`` indexed by ``[out, k]`` where ``out = sh_index(n, m)`` and
    ``k in [0, p]``: ``src_index`` is the packed index of the source coeff (0 on invalid slots,
    read harmlessly then zeroed), ``valid`` the contribution mask. Mirrors
    :func:`z_m2l_translation_tables` so the recurrence is defined once and the vectorised kernel
    cannot drift from the reference.

    Parameters
    ----------
    order : int
        Maximum SH degree ``p``.
    which : str
        ``"m2m"`` or ``"l2l"``; selects which same-type shift the tables encode.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Per-term source-index and coefficient tables. NumPy rather than JAX
        because these are compile-time constants built once per order.

    Raises
    ------
    ValueError
        If ``which`` is neither ``"m2m"`` nor ``"l2l"``.
    """
    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    if which not in ("m2m", "l2l"):
        raise ValueError("which must be 'm2m' or 'l2l'")
    coeff_count = sh_size(p)
    n_slots = p + 1
    src_index = np.zeros((coeff_count, n_slots), dtype=np.int32)
    valid = np.zeros((coeff_count, n_slots), dtype=np.bool_)
    for n in range(p + 1):
        for m in range(-n, n + 1):
            out_idx = sh_index(n, m)
            m_abs = abs(m)
            for k in range(n_slots):
                src_n = n - k if which == "m2m" else n + k
                ok = (src_n >= m_abs) if which == "m2m" else (src_n <= p)
                if ok:
                    src_index[out_idx, k] = sh_index(src_n, m)
                    valid[out_idx, k] = True
    return src_index, valid


def _translate_along_z_shift_real(
    coeffs: Array, dz: Array, *, order: int, which: str
) -> Array:
    """Vectorised real z-axis same-type shift (M2M/L2L), shared kernel.

    ``out[n,m] = sum_k (dz)^k/k! * coeffs[src(n,m,k)]`` over the static tables from
    :func:`z_shift_translation_tables`.

    Summation runs over slot ``k`` in **ascending order**, matching the term order of
    the ``(dz)^k/k!`` series, so the accumulation order is fixed by construction and
    reproducible across orders and batch shapes. That ordering is part of the numerics
    and must not be rewritten -- reassociating this sum, or letting a reduction library
    choose the order, changes results (see ``NUMERICS_AND_JAX.md``).

    This docstring used to claim bit-identity to a "per-(n,m) unrolled reference". That
    reference no longer exists -- it was replaced when the M2M/L2L z-translates were
    table-vectorised -- so the claim was untestable as written and has been replaced by
    the property above, which is the one this implementation actually guarantees.

    Parameters
    ----------
    coeffs : Array
        Packed real coefficients to shift.
    dz : Array
        Signed shift along ``+z``.
    order : int
        Maximum SH degree ``p``. Static: it selects the term tables.
    which : str
        ``"m2m"`` or ``"l2l"``, forwarded to
        :func:`z_shift_translation_tables`.

    Returns
    -------
    Array
        Packed real coefficients at the shifted centre.
    """
    p = int(order)
    coeffs = jnp.asarray(coeffs)
    dz = jnp.asarray(dz).reshape(())
    dtype = coeffs.dtype
    fact = _factorial_table_jax(p, dtype)
    src_index_np, valid_np = z_shift_translation_tables(p, which)
    src_index = jnp.asarray(src_index_np)
    valid = jnp.asarray(valid_np)
    # coeff_k = (dz)^k / k!, shared across outputs (integer exponents match the unrolled form).
    coeff_k = (dz ** jnp.arange(p + 1)) / fact[: p + 1]  # (p+1,)
    src = coeffs[src_index]  # (ncoeff, p+1)
    terms = jnp.where(valid, src * coeff_k[None, :], jnp.asarray(0.0, dtype=dtype))
    return jnp.sum(terms, axis=1).astype(dtype)


@partial(jax.jit, static_argnames=("order",))
def translate_along_z_m2m_real(
    multipole: Array,
    dz: Array,
    *,
    order: int,
) -> Array:
    """Translate multipole along +z in real harmonic basis.

    M'_n^m = sum_{k=0}^{n-|m|} (dz)^k / k! * M_{n-k}^m

    For real harmonics, this is the SAME formula as for complex harmonics
    because the z-axis translation is diagonal in m (doesn't mix different m).
    Vectorised over the shared per-term tables (:func:`z_shift_translation_tables`).

    Parameters
    ----------
    multipole : Array
        Packed real multipole coefficients.
    dz : Array
        Signed shift along ``+z``.
    order : int
        Maximum SH degree ``p``.

    Returns
    -------
    Array
        Packed real multipole coefficients at the shifted centre.
    """
    return _translate_along_z_shift_real(multipole, dz, order=order, which="m2m")


@lru_cache(maxsize=None)
def z_m2l_translation_tables(order: int) -> Tuple[np.ndarray, ...]:
    """Single source of truth for the real z-axis M2L recurrence structure.

    The real (Dehnen no-sqrt2) z-axis M2L is

        F_n^m = sum_{k=|m|}^{p-n} sign(m) * (n+k)! / r^{n+k+1} * M_k^m,
        sign(m) = (-1)^m * (2 if m != 0 else 1),

    where the factor of 2 on the m != 0 channels comes from the no-sqrt2 pairing
    of the complex +m/-m channels (dropping it halves every m != 0 coefficient
    and caps off-axis M2L accuracy regardless of expansion order).

    Returns static per-term metadata, indexed by ``[out, k]`` where
    ``out = sh_index(n, m)`` and ``k`` is the slot in ``[0, p]``:

        src_index[out, k]   packed index ``sh_index(k, m)`` of the source coeff
        valid[out, k]       True when slot k contributes (``|m| <= k <= p - n``)
        fact_index[out, k]  numerator factorial index (``n + k``)
        r_exponent[out, k]  radius exponent (``n + k + 1``)
        sign[out]           ``sign(m)`` (shared across all slots of an output)

    BOTH the pure-JAX kernel :func:`translate_along_z_m2l_real` and the Pallas
    kernel (``jaccpot.pallas.m2l_core_z_real``) build from these tables, so the
    recurrence is defined exactly once and the two encodings cannot drift. The
    parity test in ``tests/unit/operators/test_pallas_m2l_core_z_real.py``
    guards this invariant on CPU (Pallas interpret mode).

    Parameters
    ----------
    order : int
        Maximum SH degree ``p``.

    Returns
    -------
    Tuple[np.ndarray, ...]
        Host-side tables describing the M2L recurrence. NumPy because they are
        compile-time constants, and shared with the Pallas kernel -- which is
        why they live here once rather than being re-derived per lane.

    Raises
    ------
    ValueError
        If ``order`` is outside the supported range.
    """
    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    coeff_count = sh_size(p)
    n_slots = p + 1
    src_index = np.zeros((coeff_count, n_slots), dtype=np.int32)
    valid = np.zeros((coeff_count, n_slots), dtype=np.bool_)
    fact_index = np.zeros((coeff_count, n_slots), dtype=np.int32)
    r_exponent = np.zeros((coeff_count, n_slots), dtype=np.int32)
    sign = np.ones((coeff_count,), dtype=np.float64)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            out_idx = sh_index(n, m)
            sign[out_idx] = (-1.0 if (m % 2) else 1.0) * (2.0 if m != 0 else 1.0)
            m_abs = abs(m)
            for k in range(m_abs, p - n + 1):
                src_index[out_idx, k] = sh_index(k, m)
                valid[out_idx, k] = True
                fact_index[out_idx, k] = n + k
                r_exponent[out_idx, k] = n + k + 1
    return src_index, valid, fact_index, r_exponent, sign


@partial(jax.jit, static_argnames=("order",))
def translate_along_z_m2l_real(
    multipole: Array,
    r: Array,
    *,
    order: int,
) -> Array:
    """Translate multipole to local along +z in the real harmonic basis.

    F_n^m = sum_{k=|m|}^{p-n} sign(m) * (n+k)! / r^{n+k+1} * M_k^m

    Implemented as a vectorized contraction over the shared per-term tables from
    :func:`z_m2l_translation_tables` (the single source of the recurrence, also
    used by the Pallas kernel).

    Parameters
    ----------
    multipole : Array
        Packed real multipole coefficients.
    r : Array
        Separation along ``+z``. The recurrence forms ``r**-(n+k+1)``, so the
        caller is responsible for flooring it away from zero.
    order : int
        Maximum SH degree ``p``.

    Returns
    -------
    Array
        Packed real local coefficients at the target centre.
    """
    p = int(order)
    multipole = jnp.asarray(multipole)
    r = jnp.asarray(r).reshape(())
    dtype = multipole.dtype

    fact = _factorial_table_jax(2 * p, dtype)
    src_index_np, valid_np, fact_index_np, r_exponent_np, sign_np = (
        z_m2l_translation_tables(p)
    )
    src_index = jnp.asarray(src_index_np)
    valid = jnp.asarray(valid_np)
    fact_index = jnp.asarray(fact_index_np)
    r_exponent = jnp.asarray(r_exponent_np, dtype=dtype)
    sign = jnp.asarray(sign_np, dtype=dtype)

    # Per-(output, slot) term: sign(m) * (n+k)! / r^(n+k+1) * M_k^m, masked to the
    # valid slots. Invalid slots read index 0 / exponent 0 (harmless) and are
    # zeroed before the reduction.
    src_mult = multipole[src_index]
    fact_num = fact[fact_index]
    r_pow = jnp.power(r, r_exponent)
    terms = sign[:, None] * fact_num / r_pow * src_mult
    terms = jnp.where(valid, terms, jnp.asarray(0.0, dtype=dtype))
    return jnp.sum(terms, axis=1).astype(dtype)


@partial(jax.jit, static_argnames=("order",))
def translate_along_z_l2l_real(
    local: Array,
    dz: Array,
    *,
    order: int,
) -> Array:
    """Translate local expansion along +z in real harmonic basis.

    From Dehnen (2014) eq 3e with translation along z-axis (only l=0 survives):

        F'_n^m = sum_{k=0}^{p-n} Υ_k^0*(dz * z_hat) * F_{n+k}^m

    where Υ_k^0(dz) = (dz)^k / k! from eq 2b with m=0.

    Parameters
    ----------
    local : Array
        Packed real local coefficients of length (order+1)^2.
    dz : Array
        Translation distance along z-axis. Positive dz means child is at
        higher z than parent (child_z = parent_z + dz).
    order : int
        Maximum SH degree.

    Returns
    -------
    Array
        Packed real local coefficients at new (translated) center.
    """
    return _translate_along_z_shift_real(local, dz, order=order, which="l2l")


# ===========================================================================
# Full M2L with rotation-accelerated kernel
# ===========================================================================


# The `with_transverse_degeneracy_jvp` layer on the three cascade operators below
# sits *inside* the `jax.jit` (so it adds no dispatch boundary) and *outside*
# `highest_matmul_precision` (so the pinned precision still covers the body, and the
# JVP rule pins its own). It leaves the primal bit-identical and supplies only the
# transverse derivative on the `rho == 0` axis; see
# :mod:`jaccpot.operators._transverse_degeneracy_jvp`.
@partial(jax.jit, static_argnames=("order",))
@partial(with_transverse_degeneracy_jvp, generators=_M2L_TRANSVERSE_GENERATORS)
@highest_matmul_precision
def m2l_a6_real_only(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """M2L using Dehnen A6 with real-only rotations and z-translation.

    This implementation rotates multipoles using real B_U/Dz blocks, applies
    the real-only z-axis M2L recurrence, and rotates locals back with B_T/Dz.

    Differentiable in both arguments, forward and reverse. Near the ``rho == 0`` axis
    the ``d/dx`` and ``d/dy`` cotangents come from a ``custom_jvp`` rather than from
    differentiating the alignment azimuth, which is undefined there and ill-conditioned
    nearby; the analytic branch applies inside exactly zero outside a narrow band around that axis (``rho <= sqrt(eps) * |delta|``, the measured crossover between the two routes' errors) and the polar route is left
    untouched outside it.

    Parameters
    ----------
    multipole : Array
        Packed real multipole coefficients of the source.
    delta : Array
        3-vector from the multipole (source) centre to the local (target)
        centre, i.e. ``target - source``. Note this is the OPPOSITE sign
        convention from :func:`m2m_real` and :func:`l2l_real`, which take
        ``source - destination``. See ``docs/operator_conventions.md`` section 1,
        which measures what each wrong sign costs.
    order : int
        Maximum SH degree ``p``.

    Returns
    -------
    Array
        Packed real local coefficients at the target centre.
    """
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)
    dtype = multipole.dtype
    p = int(order)

    # Convention: delta points from multipole (source) center to local (target)
    # center, i.e. delta = target - source.

    # Extract delta components
    x, y, z = delta[0], delta[1], delta[2]
    r2 = jnp.dot(delta, delta, precision=lax.Precision.HIGHEST)
    r = jnp.sqrt(floor_squared_radius(r2))

    # Step 1: Rotate MULTIPOLE into the z-aligned frame.
    M_rotated = jnp.zeros_like(multipole)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_to = real_rotation_to_z_axis_multipole(x, y, z, ell, dtype=dtype)
        M_rotated = M_rotated.at[sl].set(D_to @ multipole[sl])

    # Step 2: Z-axis M2L translation in real basis
    L_z = translate_along_z_m2l_real(M_rotated, r, order=p)

    # Step 3: Rotate the LOCAL expansion back from the z-aligned frame.
    out = jnp.zeros_like(L_z)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_from = real_rotation_from_z_axis_local(x, y, z, ell, dtype=dtype)
        out = out.at[sl].set(D_from @ L_z[sl])

    return out


@partial(jax.jit, static_argnames=("order",))
def m2l_real(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """M2L in real harmonic basis using Dehnen A6 rotation/translation.

    Uses a real-only Dehnen A6 rotation/translation path (no complex basis).

    A thin alias for :func:`m2l_a6_real_only`; see that function for the
    differentiability treatment near the ``rho == 0`` axis.

    Parameters
    ----------
    multipole : Array
        Packed real multipole coefficients of the source.
    delta : Array
        3-vector from the multipole (source) centre to the local (target)
        centre, i.e. ``target - source``. Note this is the OPPOSITE sign
        convention from :func:`m2m_real` and :func:`l2l_real`, which take
        ``source - destination``. See ``docs/operator_conventions.md`` section 1,
        which measures what each wrong sign costs.
    order : int
        Maximum SH degree ``p``.

    Returns
    -------
    Array
        Packed real local coefficients at the target centre.
    """
    return m2l_a6_real_only(multipole, delta, order=order)


@partial(jax.jit, static_argnames=("order",))
def m2l_optimized_real(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Optimized M2L in real harmonic basis.

    Delegates to the real-only Dehnen A6 implementation. This keeps behavior
    aligned with :func:`m2l_real`.

    Despite the name, there is no separate optimisation here -- it is the same
    call. The name is retained for API compatibility.

    Parameters
    ----------
    multipole : Array
        Packed real multipole coefficients of the source.
    delta : Array
        3-vector from the multipole (source) centre to the local (target)
        centre, i.e. ``target - source``. Note this is the OPPOSITE sign
        convention from :func:`m2m_real` and :func:`l2l_real`, which take
        ``source - destination``. See ``docs/operator_conventions.md`` section 1,
        which measures what each wrong sign costs.
    order : int
        Maximum SH degree ``p``.

    Returns
    -------
    Array
        Packed real local coefficients at the target centre.
    """
    return m2l_a6_real_only(multipole, delta, order=order)


# ===========================================================================
# Full M2M and L2L with rotation-accelerated kernels
# ===========================================================================


@partial(jax.jit, static_argnames=("order",))
@partial(with_transverse_degeneracy_jvp, generators=_M2M_TRANSVERSE_GENERATORS)
@highest_matmul_precision
def m2m_real(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """M2M in real harmonic basis: rotate → z-translate → rotate back.

    Translates a multipole expansion from one center to another using the
    Dehnen rotation-accelerated approach (pure real arithmetic).

    Parameters
    ----------
    multipole : Array
        Packed real multipole coefficients of length (order+1)^2.
    delta : Array
        3-vector from destination center to source center.
        (So the source is at destination + delta.)
    order : int
        Maximum SH degree.

    Returns
    -------
    Array
        Packed real multipole coefficients at the destination center.

    Notes
    -----
    Differentiable in both arguments, forward and reverse. Near the ``rho == 0`` axis the
    ``d/dx`` and ``d/dy`` cotangents come from a ``custom_jvp`` rather than from
    differentiating the alignment azimuth, which is undefined there and ill-conditioned
    nearby; the analytic branch applies inside exactly zero outside a narrow band around that axis (``rho <= sqrt(eps) * |delta|``, the measured crossover between the two routes' errors). At ``delta == 0`` -- the identity
    translation -- the cotangent stays zero, because ``|delta|`` has no derivative at the
    origin.
    """
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)
    dtype = multipole.dtype
    p = int(order)

    # Extract delta components
    x, y, z = delta[0], delta[1], delta[2]
    r2 = jnp.dot(delta, delta, precision=lax.Precision.HIGHEST)
    r = jnp.sqrt(floor_squared_radius(r2))

    # Handle zero displacement case
    # (return original multipole if |delta| ~ 0)
    is_zero = r < 1e-30
    dz = jnp.where(is_zero, 0.0, r)

    # Step 1: Rotate MULTIPOLE into the z-aligned frame.
    M_rotated = jnp.zeros_like(multipole)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_to = real_rotation_to_z_axis_multipole(x, y, z, ell, dtype=dtype)
        M_rotated = M_rotated.at[sl].set(D_to @ multipole[sl])

    # Step 2: Z-axis M2M translation (distance dz = |delta| along +z).
    M_z = translate_along_z_m2m_real(M_rotated, dz, order=p)

    # Step 3: Rotate MULTIPOLE back from the z-aligned frame.
    out = jnp.zeros_like(M_z)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_from = real_rotation_from_z_axis_multipole(x, y, z, ell, dtype=dtype)
        out = out.at[sl].set(D_from @ M_z[sl])

    return out


@partial(jax.jit, static_argnames=("order",))
@partial(with_transverse_degeneracy_jvp, generators=_L2L_TRANSVERSE_GENERATORS)
@highest_matmul_precision
def l2l_real(
    local: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """L2L in real harmonic basis: rotate → z-translate → rotate back.

    Translates a local expansion from a parent center to a child center using
    the Dehnen rotation-accelerated approach (pure real arithmetic).

    Parameters
    ----------
    local : Array
        Packed real local coefficients of length (order+1)^2.
    delta : Array
        3-vector ``old_center - new_center`` (parent center minus child
        center). This matches the ``dest -> source`` sign convention used by
        :func:`m2m_real` and is what makes the rotated translation converge.
    order : int
        Maximum SH degree.

    Returns
    -------
    Array
        Packed real local coefficients at the child center.

    Notes
    -----
    Differentiable in both arguments, forward and reverse, with the same ``rho == 0``
    treatment as :func:`m2m_real`.
    """
    local = jnp.asarray(local)
    delta = jnp.asarray(delta)
    dtype = local.dtype
    p = int(order)

    # Extract delta components
    x, y, z = delta[0], delta[1], delta[2]
    r2 = jnp.dot(delta, delta, precision=lax.Precision.HIGHEST)
    r = jnp.sqrt(floor_squared_radius(r2))

    # Handle zero displacement case
    is_zero = r < 1e-30
    dz = jnp.where(is_zero, 0.0, r)

    # Step 1: Rotate the LOCAL expansion into the z-aligned frame.
    L_rotated = jnp.zeros_like(local)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_to = real_rotation_to_z_axis_local(x, y, z, ell, dtype=dtype)
        L_rotated = L_rotated.at[sl].set(D_to @ local[sl])

    # Step 2: Z-axis L2L translation (distance dz = |delta| along +z).
    L_z = translate_along_z_l2l_real(L_rotated, dz, order=p)

    # Step 3: Rotate the LOCAL expansion back from the z-aligned frame.
    out = jnp.zeros_like(L_z)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_from = real_rotation_from_z_axis_local(x, y, z, ell, dtype=dtype)
        out = out.at[sl].set(D_from @ L_z[sl])

    return out
