"""Complex solid harmonics (Dehnen / solidfmm-style) in JAX.

These routines implement the same recurrences used in solidfmm's harmonics.cpp
for the complex-valued regular (R) and singular (S) solid harmonics, but
return packed complex coefficients compatible with the (p+1)^2 layout.

Conventions:
- Dehnen normalization (no √2 real basis)
- No Condon–Shortley phase in the associated Legendre parts
- Complex conjugate symmetry: H_n^{-m} = (-1)^m * conj(H_n^{m})
"""

from __future__ import annotations

import math
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from .real_harmonics import sh_index, sh_offset, sh_size


def _pack_complex(full_nm: jnp.ndarray) -> jnp.ndarray:
    """Pack a (p+1, p+1) complex array (m>=0) into (p+1)^2 with m in [-n,n]."""
    p = full_nm.shape[0] - 1
    out = jnp.zeros((sh_size(p),), dtype=full_nm.dtype)
    real_dtype = jnp.real(jnp.zeros((), dtype=full_nm.dtype)).dtype

    for n in range(p + 1):
        for m in range(-n, n + 1):
            if m >= 0:
                val = full_nm[n, m]
            else:
                m_abs = -m
                sign = jnp.asarray(-1.0 if (m_abs % 2) else 1.0, dtype=real_dtype)
                val = sign.astype(full_nm.dtype) * jnp.conjugate(full_nm[n, m_abs])
            out = out.at[sh_index(n, m)].set(val)

    return out


def complex_R_solidfmm(delta: jnp.ndarray, *, order: int) -> jnp.ndarray:
    """Regular complex solid harmonics R (solidfmm recursion).

    Returns packed complex coefficients for degrees 0..order, m in [-n,n].
    """
    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    d = jnp.asarray(delta, dtype=jnp.float64)
    x, y, z = d[0], d[1], d[2]
    r2 = x * x + y * y + z * z

    val = jnp.zeros((p + 1, p + 1), dtype=jnp.complex128)
    val = val.at[0, 0].set(1.0 + 0.0j)

    if p == 0:
        return _pack_complex(val)

    val = val.at[1, 0].set(z)
    val = val.at[1, 1].set((x + 1j * y) * 0.5)

    for m in range(2, p + 1):
        fac = 1.0 / (2.0 * m)
        val = val.at[m, m].set(val[m - 1, m - 1] * (x + 1j * y) * fac)

    for m in range(1, p):
        val = val.at[m + 1, m].set(z * val[m, m])

    for m in range(0, p - 1):
        for n in range(m + 2, p + 1):
            fac = 1.0 / ((n + m) * (n - m))
            val = val.at[n, m].set(
                ((2 * n - 1) * z * val[n - 1, m] - r2 * val[n - 2, m]) * fac
            )

    return _pack_complex(val)


def complex_R_solidfmm_preserve_dtype(delta: jnp.ndarray, *, order: int) -> jnp.ndarray:
    """Regular solid harmonics preserving the real dtype of ``delta``.

    The reference ``complex_R_solidfmm`` intentionally computes in float64/complex128.
    This variant keeps float32 inputs in complex64 for the large-N GPU local-eval
    path while retaining complex128 when the incoming positions are float64.
    """
    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    delta_arr = jnp.asarray(delta)
    if jnp.issubdtype(delta_arr.dtype, jnp.floating):
        real_dtype = delta_arr.dtype
    else:
        real_dtype = jnp.float32
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64

    d = jnp.asarray(delta, dtype=real_dtype)
    x, y, z = d[0], d[1], d[2]
    r2 = x * x + y * y + z * z
    xy = x.astype(complex_dtype) + jnp.asarray(1j, dtype=complex_dtype) * y.astype(
        complex_dtype
    )

    val = jnp.zeros((p + 1, p + 1), dtype=complex_dtype)
    one = jnp.asarray(1.0, dtype=real_dtype).astype(complex_dtype)
    val = val.at[0, 0].set(one)

    if p == 0:
        return _pack_complex(val)

    val = val.at[1, 0].set(z.astype(complex_dtype))
    val = val.at[1, 1].set(
        xy * jnp.asarray(0.5, dtype=real_dtype).astype(complex_dtype)
    )

    for m in range(2, p + 1):
        fac = jnp.asarray(1.0 / (2.0 * m), dtype=real_dtype).astype(complex_dtype)
        val = val.at[m, m].set(val[m - 1, m - 1] * xy * fac)

    zc = z.astype(complex_dtype)
    r2c = r2.astype(complex_dtype)
    for m in range(1, p):
        val = val.at[m + 1, m].set(zc * val[m, m])

    for m in range(0, p - 1):
        for n in range(m + 2, p + 1):
            fac = jnp.asarray(1.0 / ((n + m) * (n - m)), dtype=real_dtype).astype(
                complex_dtype
            )
            val = val.at[n, m].set(
                ((2 * n - 1) * zc * val[n - 1, m] - r2c * val[n - 2, m]) * fac
            )

    return _pack_complex(val)


def complex_S_solidfmm(delta: jnp.ndarray, *, order: int) -> jnp.ndarray:
    """Singular complex solid harmonics S (solidfmm recursion).

    Returns packed complex coefficients for degrees 0..order, m in [-n,n].
    """
    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    d = jnp.asarray(delta, dtype=jnp.float64)
    x, y, z = d[0], d[1], d[2]
    r2 = x * x + y * y + z * z
    r2 = jnp.maximum(r2, 1e-60)
    r2inv = 1.0 / r2
    r = jnp.sqrt(r2)

    val = jnp.zeros((p + 1, p + 1), dtype=jnp.complex128)
    val = val.at[0, 0].set(1.0 / r)

    if p == 0:
        return _pack_complex(val)

    val = val.at[1, 0].set(z * r2inv * val[0, 0])
    val = val.at[1, 1].set((x + 1j * y) * r2inv * val[0, 0])

    for m in range(2, p + 1):
        fac = (2.0 * m - 1.0) * r2inv
        val = val.at[m, m].set(val[m - 1, m - 1] * (x + 1j * y) * fac)

    for m in range(1, p):
        fac = (2.0 * m + 1.0) * z * r2inv
        val = val.at[m + 1, m].set(fac * val[m, m])

    for m in range(0, p - 1):
        for n in range(m + 2, p + 1):
            fac = r2inv
            coeff = (n + m - 1) * (n - m - 1)
            val = val.at[n, m].set(
                ((2 * n - 1) * z * val[n - 1, m] - coeff * val[n - 2, m]) * fac
            )

    return _pack_complex(val)


def p2m_complex(delta: jnp.ndarray, mass: jnp.ndarray, *, order: int) -> jnp.ndarray:
    """P2M for a single point mass in the solidfmm complex basis.

    Computes M_n^m = mass * R_n^m(delta).
    """
    coeffs = complex_R_solidfmm(delta, order=order)
    return jnp.asarray(mass) * coeffs


@partial(jax.jit, static_argnames=("order",))
def p2m_complex_batch(
    deltas: jnp.ndarray,
    masses: jnp.ndarray,
    *,
    order: int,
) -> jnp.ndarray:
    """Batch P2M for point masses in the solidfmm complex basis."""
    return jax.vmap(
        lambda d, m: p2m_complex(d, m, order=order),
        in_axes=(0, 0),
        out_axes=0,
    )(deltas, masses)
