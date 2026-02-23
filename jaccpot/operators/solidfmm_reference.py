"""Reference complex-basis operators inspired by solidfmm (non-optimized).

This module provides a minimal, readable implementation of the complex-basis
rotation/translation pipeline following Dehnen's A6 scheme, using the same
normalization as jaccpot.operators.real_harmonics (no √2 real basis).

It is intended as a correctness reference to compare against the fast real
operators and to help debug rotation/translation conventions.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from .real_harmonics import (
    _compute_dehnen_B_matrix_complex,
    build_Q_dehnen_no_sqrt2,
    sh_offset,
    sh_size,
)


@lru_cache(maxsize=None)
def _complex_Dz(ell: int, angle: float) -> np.ndarray:
    """Complex z-rotation for degree ell: diag(exp(i m angle))."""
    m_vals = np.arange(-ell, ell + 1)
    diag = np.exp(1j * m_vals * angle)
    return np.diag(diag)


@lru_cache(maxsize=None)
def _complex_swap_matrices(ell: int) -> Tuple[np.ndarray, np.ndarray]:
    """Complex swap matrices for local (B_T) and multipole (B_U) bases."""
    B = _compute_dehnen_B_matrix_complex(ell, "float64")
    return B, B.T


def _angles_from_delta(delta: np.ndarray) -> Tuple[float, float]:
    x, y, z = float(delta[0]), float(delta[1]), float(delta[2])
    rho = math.hypot(x, y)
    alpha = math.atan2(y, x)
    beta = math.atan2(rho, z)
    return alpha, beta


def _real_block_to_complex(block_real: np.ndarray, ell: int) -> np.ndarray:
    Q = build_Q_dehnen_no_sqrt2(ell)
    Q_inv = np.linalg.inv(Q)
    return Q_inv @ block_real


def _complex_block_to_real(block_complex: np.ndarray, ell: int) -> np.ndarray:
    Q = build_Q_dehnen_no_sqrt2(ell)
    block_real = Q @ block_complex
    return np.real(block_real)


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
    """Complex-basis M2L along +z using Dehnen's series (reference)."""
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
    """Reference M2L using complex basis and solidfmm-style rotations."""
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
