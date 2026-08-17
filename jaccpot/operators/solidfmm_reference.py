"""Reference complex-basis operators inspired by solidfmm (non-optimized).

This module provides a minimal, readable implementation of the complex-basis
rotation/translation pipeline following Dehnen's A6 scheme, using the same
normalization as jaccpot.operators.real_harmonics (no √2 real basis).

It is intended as a correctness reference to compare against the fast real
operators and to help debug rotation/translation conventions.

**It did not earn that description off axis until the two conventions in
``_angles_from_delta`` were corrected** -- an azimuth and a polar tilt, which made
:func:`m2l_solidfmm_reference` wrong by 1.2e-2 relative for any non-axis-aligned
``delta``, flat in expansion order. ``translate_along_z_m2l_complex`` (the z-only
piece, cross-checked against ``complex_ops`` in ``test_complex_ops.py``) was never
affected. See ``docs/operator_conventions.md`` section 4 for the diagnosis and the
note on ``_angles_from_delta`` for the expressions. Nothing outside ``tests/``
imports this module, so the defect only ever weakened an oracle.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from ._precision import highest_matmul_precision
from .real_harmonics import (
    _compute_dehnen_B_matrix_complex,
    build_Q_dehnen_no_sqrt2,
    sh_offset,
    sh_size,
)


@lru_cache(maxsize=None)
def _complex_Dz(ell: int, angle: float) -> np.ndarray:
    """Complex z-rotation for degree ell: diag(exp(i m angle)).

    Memoized on ``(ell, angle)``, so it is only useful for the handful of angles
    a reference run visits -- an unbounded cache keyed on a float would be a leak
    in production, which this module is not.

    Parameters
    ----------
    ell : int
        Degree ``l``.
    angle : float
        Rotation angle about ``+z``, in radians. Must be a Python float: it is
        part of the cache key.

    Returns
    -------
    np.ndarray
        ``[2l+1, 2l+1]`` complex diagonal matrix.
    """
    m_vals = np.arange(-ell, ell + 1)
    diag = np.exp(1j * m_vals * angle)
    return np.diag(diag)


@lru_cache(maxsize=None)
def _complex_swap_matrices(ell: int) -> Tuple[np.ndarray, np.ndarray]:
    """Complex swap matrices for local (B_T) and multipole (B_U) bases.

    Parameters
    ----------
    ell : int
        Degree ``l``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(B_T, B_U)`` as ``(B, B.T)`` -- the local and multipole swaps are
        transposes of one another, which is why one matrix builds both.
    """
    B = _compute_dehnen_B_matrix_complex(ell, "float64")
    return B, B.T


def _angles_from_delta(delta: np.ndarray) -> Tuple[float, float]:
    """The two alignment angles this module's rotations consume.

    **Both conventions here were wrong until this commit, and they were the whole
    reason** :func:`m2l_solidfmm_reference` **was wrong off axis** -- by 1.2e-2
    relative, *flat in expansion order*, which is the signature of a convention
    error rather than truncation. Diagnosis and measurements:
    ``docs/operator_conventions.md`` section 4.

    Two expressions, matching
    :func:`~jaccpot.operators.complex_ops._angles_from_delta_solidfmm`, whose
    rotation composition is otherwise identical to the one here:

    * ``alpha`` is ``atan2(x, y)``, the azimuth that removes the *x* component,
      because the coded ``B`` is the ``x <-> z`` swap and so ``B Dz(theta) B``
      turns about ``x``. The old ``atan2(y, x)`` suited a ``y <-> z`` swap, which
      is not the matrix this module uses -- the same error already fixed once in
      the real rotations.
    * ``beta`` is ``atan2(-rho, z)``, the negated polar tilt.

    Do not "simplify" either back. ``atan2(y, x)`` and ``atan2(rho, z)`` are what
    a reader expects from spherical-coordinate habit, which is exactly why this
    went unnoticed;
    ``test_solidfmm_reference_matches_m2l_real_off_axis`` is what now catches it.

    Parameters
    ----------
    delta : np.ndarray
        3-vector, ``target centre - source centre``.

    Returns
    -------
    Tuple[float, float]
        ``(alpha, beta)`` in radians, as Python floats -- they become
        :func:`_complex_Dz` cache keys.
    """
    x, y, z = float(delta[0]), float(delta[1]), float(delta[2])
    rho = math.hypot(x, y)
    alpha = math.atan2(x, y)
    beta = math.atan2(-rho, z)
    return alpha, beta


@highest_matmul_precision
def _real_block_to_complex(block_real: np.ndarray, ell: int) -> np.ndarray:
    Q = build_Q_dehnen_no_sqrt2(ell)
    Q_inv = np.linalg.inv(Q)
    return Q_inv @ block_real


@highest_matmul_precision
def _complex_block_to_real(block_complex: np.ndarray, ell: int) -> np.ndarray:
    Q = build_Q_dehnen_no_sqrt2(ell)
    block_real = Q @ block_complex
    return np.real(block_real)


@highest_matmul_precision
def _rotate_multipole_to_z(
    block_complex: np.ndarray, delta: np.ndarray, ell: int
) -> np.ndarray:
    alpha, beta = _angles_from_delta(delta)
    B_T, B_U = _complex_swap_matrices(ell)
    Dz_alpha = _complex_Dz(ell, alpha)
    Dz_beta = _complex_Dz(ell, beta)
    # scale-rot(alpha)-swap-rot(beta)-swap (complex basis)
    D = B_U @ Dz_beta @ B_U @ Dz_alpha
    return D @ block_complex


@highest_matmul_precision
def _rotate_local_from_z(
    block_complex: np.ndarray, delta: np.ndarray, ell: int
) -> np.ndarray:
    alpha, beta = _angles_from_delta(delta)
    B_T, _ = _complex_swap_matrices(ell)
    Dz_alpha = _complex_Dz(ell, -alpha)
    Dz_beta = _complex_Dz(ell, -beta)
    # swap-rot(-beta)-swap-rot(-alpha)
    D = Dz_alpha @ B_T @ Dz_beta @ B_T
    return D @ block_complex


def translate_along_z_m2l_complex(
    multipole: np.ndarray,
    r: float,
    *,
    order: int,
) -> np.ndarray:
    """Complex-basis M2L along +z using Dehnen's series (reference).

    A direct triple loop in NumPy, not a vectorised kernel: readability over
    speed, which is this module's whole purpose.

    Parameters
    ----------
    multipole : np.ndarray
        Packed complex multipole coefficients in the z-aligned frame.
    r : float
        Separation along ``+z``. The series forms ``r**-(n+k+1)``, so it must be
        strictly positive; nothing here floors it.
    order : int
        Expansion order ``p``.

    Returns
    -------
    np.ndarray
        Packed complex local coefficients, ``complex128``.
    """
    p = int(order)
    ncoeff = sh_size(p)
    out = np.zeros((ncoeff,), dtype=np.complex128)

    # precompute factorials
    fact = np.array([math.factorial(i) for i in range(2 * p + 1)], dtype=np.float64)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            m_abs = abs(m)
            acc = 0.0 + 0.0j
            for k in range(m_abs, p - n + 1):
                src_idx = sh_offset(k) + (m + k)
                coeff = ((-1.0) ** m) * fact[n + k] / (r ** (n + k + 1))
                acc += coeff * multipole[src_idx]
            out[sh_offset(n) + (m + n)] = acc

    return out


def m2l_solidfmm_reference(
    multipole_real: jnp.ndarray,
    delta: jnp.ndarray,
    *,
    order: int,
) -> jnp.ndarray:
    """Reference M2L using complex basis and solidfmm-style rotations.

    Takes and returns REAL packed coefficients, but works internally in the
    genuine-complex basis: real -> complex per degree, rotate to z, z-translate,
    rotate back, complex -> real.

    **Not a drop-in replacement for**
    :func:`~jaccpot.operators.real_translations.m2l_real`, despite the identical
    signature. It carries the genuine-complex normalisation, which is a factor of
    two smaller on every ``m != 0`` channel than the real no-sqrt2 basis. On an
    axis-aligned ``delta``, rescaling those channels by two reproduces
    ``m2l_real`` to round-off (measured 3.5e-17 at order 4); the relationship is
    pinned on-axis against the pure z-translate by
    ``test_solidfmm_reference_matches_z_axis_m2l``.

    That factor-of-two rescale *is* the correct relationship at every ``delta`` --
    it relates the two bases, not the geometry, and
    ``test_dehnen_local_channel_factor_holds_at_any_delta`` shows it holding to
    3.1e-16 off axis for the production complex M2L. **This function failed it
    until the two conventions in** :func:`_angles_from_delta` **were corrected**,
    by 3.9e-2 (order 2) to 1.7 (order 6), with an evaluated potential that
    plateaued at 2.1e-2 to 8.5e-2 relative error, flat in ``order``. It now agrees
    with ``m2l_real`` to 3e-17 on the rescaled coefficients, and its potential
    converges identically (1.1e-6, 3.8e-10, 1.2e-13 at orders 2, 4, 6).

    Two earlier versions of this note were wrong in opposite directions: the first
    blamed the comparison rather than the code; the second, written from a field
    check that omitted the channel rescale, claimed a second undiagnosed defect
    remained. There was one defect, in two expressions.

    The ``m != 0`` factor of two still applies. If you compare this function's
    output against a real-basis quantity, rescale first -- omitting it is what
    produced the second wrong note.

    Parameters
    ----------
    multipole_real : jnp.ndarray
        Packed REAL multipole coefficients.
    delta : jnp.ndarray
        3-vector, ``target centre - source centre``; see
        ``docs/operator_conventions.md`` section 1.
    order : int
        Expansion order ``p``.

    Returns
    -------
    jnp.ndarray
        Packed REAL local coefficients, cast back to ``multipole_real``'s dtype,
        in the genuine-complex normalisation described above.
    """
    p = int(order)
    delta_np = np.asarray(delta, dtype=np.float64)
    multipole_np = np.asarray(multipole_real, dtype=np.float64)

    # Rotate multipoles to z-axis in complex basis
    multipole_rot = np.zeros_like(multipole_np, dtype=np.complex128)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        block_real = multipole_np[sl]
        block_c = _real_block_to_complex(block_real, ell)
        block_c_rot = _rotate_multipole_to_z(block_c, delta_np, ell)
        multipole_rot[sl] = block_c_rot

    r = float(np.linalg.norm(delta_np))
    local_z = translate_along_z_m2l_complex(multipole_rot, r, order=p)

    # Rotate locals back to global frame
    local_real = np.zeros_like(multipole_np, dtype=np.float64)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        block_c = local_z[sl]
        block_c_rot = _rotate_local_from_z(block_c, delta_np, ell)
        local_real[sl] = _complex_block_to_real(block_c_rot, ell)

    return jnp.asarray(local_real, dtype=multipole_real.dtype)
