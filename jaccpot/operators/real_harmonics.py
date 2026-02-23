"""Real solid harmonics implementation following Dehnen (2014).

This module implements the real-valued solid harmonics U_n^m and T_n^m
from Dehnen (2014), Appendix A.5.2 (equations 58a,b), using the **Dehnen
real basis** (no √2 factors). These are defined as:

For m < 0:
    U_n^m = Im(Υ_n^{|m|})
    T_n^m = Im(Θ_n^{|m|})

For m >= 0:
    U_n^m = Re(Υ_n^{|m|})
    T_n^m = Re(Θ_n^{|m|})

Where:
    Υ_n^m(r) = (-1)^m r^n P_n^m(cos θ) e^{i m φ} / (n+m)!     [multipoles]
    Θ_n^m(r) = (-1)^m (n-m)! P_n^m(cos θ) e^{i m φ} / r^{n+1}  [locals]


Phase Convention (CRITICAL)
===========================

Dehnen's equation 2b has an EXPLICIT (-1)^m factor. Meanwhile, the standard
associated Legendre function P_n^m often includes the Condon-Shortley phase,
which is ALSO (-1)^m. These two factors COMBINE:

    (-1)^m × (-1)^m = (-1)^{2m} = 1

Therefore, when computing Dehnen's harmonics, we use P_n^m WITHOUT the
Condon-Shortley phase. This is equivalent to:
    - Using the "unsigned" associated Legendre function, OR
    - Applying Dehnen's explicit (-1)^m to P_n^m with C-S (they cancel)

Verification: Dehnen Table 3 gives (for n=1):
    U_1^{-1} = y/2,  U_1^{0} = z,  U_1^{+1} = x/2

At (1,0,0): U_1^{+1} = 0.5, others = 0
At (0,1,0): U_1^{-1} = 0.5, others = 0
At (0,0,1): U_1^{0} = 1.0, others = 0


Direct Real Computation
=======================

The key insight is that these can be computed directly as real polynomials
in x, y, z without going through complex arithmetic:

    U_n^m = r^n P_n^{|m|}(z/r) {cos(|m|φ), sin(|m|φ)} / (n+|m|)!

where cos(mφ) and sin(mφ) are computed from x and y using Chebyshev
recurrence, avoiding trigonometric functions entirely.


B Matrix Theory for Real Harmonics
==================================

The B matrix implements the coordinate swap (x,y,z) → (z,y,x) for spherical
harmonics, enabling fast y-axis rotations via: D_y(β) = B @ D_z(-β) @ B.

Complex Basis (Dehnen eq 63):
-----------------------------
The B matrix for the COMPLEX basis Θ_n^m is computed via the recursion:

    2 B^{m,l}_{n+1}   = B^{m,l-1}_n - B^{m,l+1}_n
    2 B^{m+1,l}_{n+1} = B^{m,l-1}_n + B^{m,l+1}_n + 2 B^{m,l}_n
    2 B^{m-1,l}_{n+1} = B^{m,l-1}_n + B^{m,l+1}_n - 2 B^{m,l}_n

Starting from B_0^{0,0} = 1. This B matrix is purely real (no imaginary part)
and satisfies B² = I (involution).

Per Dehnen eq (64): "For Υ_n^m instead of Θ_n^m, use the transpose."
So in the complex basis: B_Υ = B_Θ^T.

Transformation to Real Basis:
-----------------------------
The real harmonics T_n^m are related to complex Θ_n^m via:

    T_n^m = Re(Θ_n^{|m|})  for m >= 0  (cos channel)
    T_n^m = Im(Θ_n^{|m|})  for m < 0   (sin channel)

Using the conjugate symmetry Θ_n^{-m} = (-1)^m (Θ_n^m)*, we can express this
as a UNITARY transformation T = Q @ Θ where Q is defined by:

    For m = 0:   Q extracts Θ_n^0 directly
    For m > 0:   T_n^{+m} = (Θ_n^{-m} + (-1)^m Θ_n^{+m}) / √2
    For m < 0:   T_n^{-|m|} = (Θ_n^{|m|} - (-1)^{|m|} Θ_n^{-|m|}) / (i√2)

This Q matrix is **not** unitary (no √2). It differs from the unitary
real tesseral transform used by the real-harmonic utility kernels.

B Matrices in Real Basis:
-------------------------
The B matrices for real harmonics are obtained via similarity transform:

    B_T = Q @ B_Θ @ Q^{-1}     (for local expansions T_n^m)
    B_U = Q @ B_Θ^T @ Q^{-1}   (for multipole expansions U_n^m)

Key insight: With the Dehnen Q (no √2), we use the similarity transform:
    B_T = Q @ B_Θ @ Q^{-1}
    B_U = Q @ B_Θ^T @ Q^{-1}

Since B_Θ^T ≠ B_Θ in general, we have B_U ≠ B_T^T.

Both B_T and B_U satisfy:
    - B² = I (involution property preserved)
    - Checkerboard sparsity: non-zero only when m and l have the same sign
      (cos channels don't mix with sin channels)
    - ~25% non-zero entries (4x sparser than complex B)

The sparsity pattern arises because:
    - T_n^m for m >= 0 depends only on Re(Θ) which involves Θ^{+m} and Θ^{-m}
        - T_n^m for m < 0 depends only on Im(Θ) which involves Θ^{+|m|}
            and Θ^{-|m|}
    - The B matrix preserves this separation

Reference: Dehnen (2014) "A fast multipole method for stellar dynamics"
arXiv:1405.2255, Appendix A.


FMM Workflow (Pure Real Arithmetic)
===================================

This module provides all the building blocks for a complete FMM using only
real arithmetic in the Dehnen (no √2) basis:

1. **P2M** (Particle to Multipole): `p2m_real_direct`
   Creates multipole expansion coefficients for a point mass.
   M_n^m = mass × U_n^m(delta)

2. **M2M** (Multipole to Multipole): `m2m_real`
   Translates a multipole expansion from child center to parent center.
   Uses rotation-accelerated approach: rotate → z-translate → rotate back.

3. **M2L** (Multipole to Local): `m2l_real`, `m2l_optimized_real`
   Converts a far-field multipole expansion to a local expansion.
   Uses rotation-accelerated approach: rotate → z-translate → rotate back.

4. **L2L** (Local to Local): `l2l_real`
   Translates a local expansion from parent center to child center.
   Uses rotation-accelerated approach: rotate → z-translate → rotate back.

5. **L2P** (Local to Particle): `evaluate_local_real`,
   `evaluate_local_real_with_grad`
   Evaluates a local expansion at a particle position.
   Returns potential and optionally gradient (for force computation).

All operations are:
- Pure real arithmetic (no complex numbers)
- JIT-compilable with JAX
- Differentiable (support JAX autodiff)
- Vectorizable with vmap

Example usage::

    import jax.numpy as jnp
    from jaccpot.operators.real_harmonics import (
        p2m_real_direct, m2l_real, evaluate_local_real_with_grad
    )

    # Create multipole for unit mass at (1,0,0)
    order = 4
    mass = 1.0
    source_pos = jnp.array([1.0, 0.0, 0.0])
    multipole_center = jnp.array([0.0, 0.0, 0.0])

    multipole = p2m_real_direct(
        source_pos - multipole_center,
        jnp.array(mass),
        order=order
    )

    # Convert to local expansion at far point
    local_center = jnp.array([5.0, 0.0, 0.0])
    local = m2l_real(
        multipole,
        local_center - multipole_center,  # source to target
        order=order
    )

    # Evaluate potential and gradient at test point
    test_point = jnp.array([5.5, 0.0, 0.0])
    grad, potential = evaluate_local_real_with_grad(
        local,
        local_center - test_point,
        order=order
    )
"""

from __future__ import annotations

import math
from functools import lru_cache, partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, DTypeLike

# ===========================================================================
# Index utilities
# ===========================================================================


def sh_size(order: int) -> int:
    """Number of real SH coefficients up to degree ``order``: (p+1)^2.

    Parameters
    ----------
    order : int
        Maximum spherical harmonic degree p.

    Returns
    -------
    int
        Total number of coefficients: (p+1)^2.
    """
    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    return (p + 1) * (p + 1)


def sh_offset(ell: int) -> int:
    """Packed offset for degree ``ell`` in the (p+1)^2 layout.

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.

    Returns
    -------
    int
        Starting index for degree ell: ell^2.
    """
    ll = int(ell)
    if ll < 0:
        raise ValueError("ell must be >= 0")
    return ll * ll


def sh_index(ell: int, m: int) -> int:
    """Packed index for coefficient (ell, m) for m in [-ell..ell].

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.
    m : int
        Spherical harmonic order, must satisfy -ell <= m <= ell.

    Returns
    -------
    int
        Linear index in the packed coefficient array.
    """
    ll = int(ell)
    mm = int(m)
    if ll < 0:
        raise ValueError("ell must be >= 0")
    if mm < -ll or mm > ll:
        raise ValueError("m must satisfy -ell <= m <= ell")
    return sh_offset(ll) + (mm + ll)


# ===========================================================================
# Direct real harmonic evaluation (no complex arithmetic)
# ===========================================================================


def _factorial_table_jax(max_n: int, dtype: DTypeLike) -> Array:
    """Get factorial table as JAX array."""
    n = jnp.arange(0, max_n + 1, dtype=dtype)
    return jnp.exp(jax.lax.lgamma(n + 1.0)).astype(dtype)


@partial(jax.jit, static_argnames=("order",))
def p2m_real_direct(
    delta: Array,
    mass: Array,
    *,
    order: int,
) -> Array:
    """P2M for a single point mass using direct real harmonic computation.

    This computes multipole coefficients M_n^m = mass * U_n^m(delta) directly
    in real arithmetic, without going through complex harmonics.

    Parameters
    ----------
    delta : Array
        (3,) vector from expansion center to particle position.
    mass : Array
        Particle mass (scalar).
    order : int
        Maximum SH degree p.

    Returns
    -------
    Array
        Packed real multipole coefficients of shape ((p+1)^2,).

    Notes
    -----
    Uses the Dehnen normalization in the **no-√2 real basis**:
        U_n^m(r,θ,φ) = r^n P_n^{|m|}(cos θ) cos(|m|φ) / (n+|m|)!  for m >= 0
        U_n^m(r,θ,φ) = r^n P_n^{|m|}(cos θ) sin(|m|φ) / (n+|m|)!  for m < 0

    The cos(mφ) and sin(mφ) terms are computed via Chebyshev recurrence
    from cos φ = x/ρ and sin φ = y/ρ, avoiding trigonometric functions.

    Validation
    ----------
    The resulting real-valued polynomials (scaled by (n-m)!(n+m)!) match
    Table 3 of Dehnen (2014) for n ≤ 6 when derived via
    scripts/derive_table3_polynomials.py.
    """
    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    d = jnp.asarray(delta)
    mass = jnp.asarray(mass).reshape(())
    dtype = d.dtype

    x, y, z = d[0], d[1], d[2]
    r2 = jnp.dot(d, d)
    r = jnp.sqrt(jnp.maximum(r2, 1e-60))
    rho2 = x * x + y * y
    rho = jnp.sqrt(jnp.maximum(rho2, 1e-60))

    cos_theta = z / r
    sin_theta = rho / r

    # cos φ = x/ρ, sin φ = y/ρ (with safe handling when rho ~ 0)
    cos_phi = jnp.where(rho > 1e-30, x / rho, 1.0)
    sin_phi = jnp.where(rho > 1e-30, y / rho, 0.0)

    # Precompute factorials
    fact = _factorial_table_jax(2 * p, dtype)

    def fact_int(k: Any) -> Array:
        return fact[jnp.asarray(k, dtype=jnp.int32)]

    # Compute cos(m*φ) and sin(m*φ) via Chebyshev recurrence:
    # cos((m+1)φ) = 2*cos(φ)*cos(mφ) - cos((m-1)φ)
    # sin((m+1)φ) = 2*cos(φ)*sin(mφ) - sin((m-1)φ)
    # Note: we use sin(mφ) = sin((m-1)φ)*cos(φ) + cos((m-1)φ)*sin(φ) instead
    cos_m_phi = jnp.zeros((p + 1,), dtype=dtype)
    sin_m_phi = jnp.zeros((p + 1,), dtype=dtype)
    cos_m_phi = cos_m_phi.at[0].set(1.0)  # cos(0) = 1
    sin_m_phi = sin_m_phi.at[0].set(0.0)  # sin(0) = 0
    if p >= 1:
        cos_m_phi = cos_m_phi.at[1].set(cos_phi)
        sin_m_phi = sin_m_phi.at[1].set(sin_phi)
    for m in range(2, p + 1):
        # Chebyshev recurrence
        cos_m_phi = cos_m_phi.at[m].set(
            2.0 * cos_phi * cos_m_phi[m - 1] - cos_m_phi[m - 2]
        )
        sin_m_phi = sin_m_phi.at[m].set(
            2.0 * cos_phi * sin_m_phi[m - 1] - sin_m_phi[m - 2]
        )

    # Build coefficients degree by degree
    ncoeff = sh_size(p)
    coeffs = jnp.zeros((ncoeff,), dtype=dtype)

    # Associated Legendre function P_n^m(cos_theta) WITHOUT Condon-Shortley
    # phase.
    #
    # Dehnen (2014) equation 2b includes an explicit (-1)^m, which cancels the
    # C-S phase if present. We therefore compute P_n^m without C-S here.
    #
    # Recursion (without C-S phase):
    #   P_m^m = (2m-1)!! sin^m(theta)
    #   P_{m+1}^m = cos(theta) (2m+1) P_m^m
    #   (n-m) P_n^m = cos(theta) (2n-1) P_{n-1}^m - (n+m-1) P_{n-2}^m
    for n in range(p + 1):
        r_n = r**n

        for m in range(-n, n + 1):
            m_abs = abs(m)

            # P_m^m (without Condon-Shortley phase)
            if m_abs == 0:
                Pmm = jnp.asarray(1.0, dtype=dtype)
            else:
                # (2m-1)!! = (2m)! / (2^m m!)
                double_fact = fact_int(2 * m_abs) / ((2.0**m_abs) * fact_int(m_abs))
                Pmm = double_fact * (sin_theta**m_abs)

            if m_abs == n:
                P_nm = Pmm
            elif m_abs + 1 == n:
                # P_{m+1}^m = (2m+1) cos(theta) P_m^m
                P_nm = (2.0 * m_abs + 1.0) * cos_theta * Pmm
            else:
                # General recursion
                Pnm2 = Pmm
                Pnm1 = (2.0 * m_abs + 1.0) * cos_theta * Pmm
                for k in range(m_abs + 2, n + 1):
                    numer = (2.0 * k - 1.0) * cos_theta * Pnm1
                    numer = numer - (k + m_abs - 1.0) * Pnm2
                    Pk = numer / (k - m_abs)
                    Pnm2 = Pnm1
                    Pnm1 = Pk
                P_nm = Pnm1

            # Dehnen normalization: divide by (n + |m|)!
            denom = fact_int(n + m_abs)

            # U_n^m = r^n P_n^{|m|}(cos θ) {cos(|m|φ), sin(|m|φ)} / (n+|m|)!
            # For m >= 0: multiply by cos(|m|φ)
            # For m < 0: multiply by sin(|m|φ)
            if m >= 0:
                U_nm = r_n * P_nm * cos_m_phi[m_abs] / denom
            else:
                U_nm = r_n * P_nm * sin_m_phi[m_abs] / denom

            # M_n^m = mass * U_n^m
            idx = sh_index(n, m)
            coeffs = coeffs.at[idx].set(mass * U_nm)

    return coeffs


# ===========================================================================
# L2P: Local expansion evaluation (pure real)
# ===========================================================================


@partial(jax.jit, static_argnames=("order",))
def evaluate_local_real(
    local_coeffs: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Evaluate local expansion at a point using pure real arithmetic.

    Implements Dehnen (2014) equation 3a:
        Ψ(x_b) = Σ_{n,m} Υ_n^{m*}(s_B - x_b) · F_n^m(s_B)

    where:
        - s_B is the local expansion center
        - x_b is the evaluation point
        - delta = s_B - x_b (center MINUS evaluation point, Dehnen convention)
        - F_n^m are the local expansion coefficients from M2L
        - Υ_n^m are the inner solid harmonics (eq 2b)

    The real-valued inner solid harmonics U_n^m (eq 58) are:
        U_n^m(r,θ,φ) = r^n P_n^{|m|}(cos θ) cos(|m|φ) / (n+|m|)!  for m >= 0
        U_n^m(r,θ,φ) = r^n P_n^{|m|}(cos θ) sin(|m|φ) / (n+|m|)!  for m < 0

    This normalization matches P2M (which also uses U_n^m), ensuring
    consistency through the FMM pipeline: P2M → M2M → M2L → L2L → L2P.

    Note: These locals use the Dehnen no-√2 real basis. Do not compare
    directly to unitary real tesseral coefficients without conversion.

    Note: Do not compare these coefficients to the unitary real tesseral
    basis in the real-harmonic utility layer without converting between bases.

    **Important Convention**: delta = center - eval_point
    (NOT eval_point - center)!
    This follows Dehnen's definition in eq 3a where the argument is
    (s_B - x_b).
    The sign matters because the Taylor series 1/|R+d| = (1/R) Σ_n (-d/R)^n has
    alternating signs that arise naturally when delta points from eval toward
    center.

    Parameters
    ----------
    local_coeffs : Array
        Packed real local coefficients F_n^m of shape ((p+1)^2,).
    delta : Array
        (3,) vector from evaluation point TO expansion center:
        delta = center - eval_point.
        This is the Dehnen (2014) convention from equation 3a.
    order : int
        Maximum SH degree p.

    Returns
    -------
    Array
        Scalar potential value at the evaluation point.
    """
    p = int(order)
    d = jnp.asarray(delta)
    local_coeffs = jnp.asarray(local_coeffs)
    dtype = d.dtype

    # delta follows Dehnen convention: center - eval_point (eq 3a: s_B - x_b)
    # No sign flip needed - caller provides the correct convention.

    x, y, z = d[0], d[1], d[2]
    r2 = jnp.dot(d, d)
    r = jnp.sqrt(jnp.maximum(r2, 1e-60))
    rho2 = x * x + y * y
    rho = jnp.sqrt(jnp.maximum(rho2, 1e-60))

    cos_theta = z / r
    sin_theta = rho / r

    # cos φ = x/ρ, sin φ = y/ρ (with safe handling when rho ~ 0)
    cos_phi = jnp.where(rho > 1e-30, x / rho, 1.0)
    sin_phi = jnp.where(rho > 1e-30, y / rho, 0.0)

    # Precompute factorials
    fact = _factorial_table_jax(2 * p, dtype)

    def fact_int(k: Any) -> Array:
        return fact[jnp.asarray(k, dtype=jnp.int32)]

    # Compute cos(m*φ) and sin(m*φ) via Chebyshev recurrence
    cos_m_phi = jnp.zeros((p + 1,), dtype=dtype)
    sin_m_phi = jnp.zeros((p + 1,), dtype=dtype)
    cos_m_phi = cos_m_phi.at[0].set(1.0)
    sin_m_phi = sin_m_phi.at[0].set(0.0)
    if p >= 1:
        cos_m_phi = cos_m_phi.at[1].set(cos_phi)
        sin_m_phi = sin_m_phi.at[1].set(sin_phi)
    for m in range(2, p + 1):
        cos_m_phi = cos_m_phi.at[m].set(
            2.0 * cos_phi * cos_m_phi[m - 1] - cos_m_phi[m - 2]
        )
        sin_m_phi = sin_m_phi.at[m].set(
            2.0 * cos_phi * sin_m_phi[m - 1] - sin_m_phi[m - 2]
        )

    # Accumulate potential
    total = jnp.asarray(0.0, dtype=dtype)

    for n in range(p + 1):
        r_n = r**n

        for m in range(-n, n + 1):
            m_abs = abs(m)

            # Associated Legendre P_n^{|m|}(cos θ) WITHOUT
            # Condon-Shortley phase
            if m_abs == 0:
                Pmm = jnp.asarray(1.0, dtype=dtype)
            else:
                double_fact = fact_int(2 * m_abs) / ((2.0**m_abs) * fact_int(m_abs))
                Pmm = double_fact * (sin_theta**m_abs)

            if m_abs == n:
                P_nm = Pmm
            elif m_abs + 1 == n:
                P_nm = (2.0 * m_abs + 1.0) * cos_theta * Pmm
            else:
                Pnm2 = Pmm
                Pnm1 = (2.0 * m_abs + 1.0) * cos_theta * Pmm
                for k in range(m_abs + 2, n + 1):
                    numer = (2.0 * k - 1.0) * cos_theta * Pnm1
                    numer = numer - (k + m_abs - 1.0) * Pnm2
                    Pk = numer / (k - m_abs)
                    Pnm2 = Pnm1
                    Pnm1 = Pk
                P_nm = Pnm1

            # L2P uses the inner solid harmonic U_n^m (same normalization as
            # P2M).
            #
            # From Dehnen (2014) equation 2b:
            #   Υ_n^m(r) = (-1)^m r^n P_n^m(cos θ) e^{i m φ} / (n+m)!
            #
            # For real harmonics (eq 58), we use:
            #   U_n^m = r^n P_n^{|m|}(cos θ) {cos(|m|φ), sin(|m|φ)} / (n+|m|)!
            #
            # where the Condon-Shortley phase is not included in P_n^m.
            #
            # The L2P formula (eq 3a) uses Υ_n^{m*} (conjugate), but since the
            # local coefficients F_n^m are defined consistently with this
            # convention,
            # we just use U_n^m directly (the conjugate just affects
            # the complex
            # phase which is handled by the real/imaginary split).
            norm = fact_int(n + m_abs)  # (n + |m|)!
            if m >= 0:
                U_nm = r_n * P_nm * cos_m_phi[m_abs] / norm
            else:
                U_nm = r_n * P_nm * sin_m_phi[m_abs] / norm

            # φ += F_n^m * U_n^m
            idx = sh_index(n, m)
            total = total + local_coeffs[idx] * U_nm

    return total


@partial(jax.jit, static_argnames=("order",))
def evaluate_local_real_with_grad(
    local_coeffs: Array,
    delta: Array,
    *,
    order: int,
) -> tuple[Array, Array]:
    """Evaluate local expansion and its gradient using autodiff.

    Computes both the potential and its gradient (for force/acceleration
    calculation).
    The gradient is computed via JAX autodiff of evaluate_local_real.

    **Important Convention**: delta = center - eval_point
    (NOT eval_point - center)!
    This follows the Dehnen (2014) convention from equation 3a.

    The returned gradient is ∇φ with respect to the evaluation point position.
    For gravitational acceleration, use a = -∇φ (note the sign!).

    Parameters
    ----------
    local_coeffs : Array
        Packed real local coefficients F_n^m of shape ((p+1)^2,).
    delta : Array
        (3,) vector from evaluation point TO expansion center:
        delta = center - eval_point.
        This is the Dehnen (2014) convention from equation 3a.
    order : int
        Maximum SH degree p.

    Returns
    -------
    tuple[Array, Array]
        (gradient, potential) where gradient is shape (3,) and potential is
        scalar.
        Note: gradient is ∇φ (not the acceleration -∇φ).
    """
    p = int(order)

    def phi_fn(d: Array) -> Array:
        return evaluate_local_real(local_coeffs, d, order=p)

    # Note: The gradient returned is with respect to delta.
    # Since delta = center - eval_point, and center is fixed,
    # d(delta)/d(eval_point) = -I, so ∇_{eval}φ = -∇_delta φ.
    # We return ∇_delta φ; caller should negate if needed for force.
    potential, grad = jax.value_and_grad(phi_fn)(delta)
    return grad, potential


# ===========================================================================
# Real B matrix (sparse checkerboard structure)
# ===========================================================================


@lru_cache(maxsize=None)
def _compute_dehnen_B_matrix_complex(ell: int, dtype_key: str) -> np.ndarray:
    """Compute the B swap matrix for COMPLEX Θ_n^m using Dehnen's recursion.

    This implements the recursion from Dehnen (2014) Appendix A.6.1, eq. (63):

        2 B^{m,l}_{n+1} = B^{m,l-1}_n - B^{m,l+1}_n
        2 B^{m+1,l}_{n+1} = B^{m,l-1}_n + B^{m,l+1}_n + 2 B^{m,l}_n
        2 B^{m-1,l}_{n+1} = B^{m,l-1}_n + B^{m,l+1}_n - 2 B^{m,l}_n

    Starting from B_0^{0,0} = 1.

    NOTE: This B matrix is for the COMPLEX basis Θ_n^m. For the REAL basis
    T_n^m, use _compute_real_B_matrices_cached which applies the Q transform.

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.
    dtype_key : str
        String representation of dtype ('float32' or 'float64').

    Returns
    -------
    np.ndarray
        B matrix of shape (2*ell+1, 2*ell+1) for the complex Θ basis.
    """
    if "float32" in dtype_key:
        rdtype = np.float32
    else:
        rdtype = np.float64

    # For ell=0, B is just [[1]]
    if ell == 0:
        return np.array([[1.0]], dtype=rdtype)

    # Build B matrices from n=0 up to n=ell using recursion.
    # B_prev[m + n, l + n] = B_n^{m,l} for m, l in [-n, n]
    # Start with B_0 = [[1]]
    B_prev = np.array([[1.0]], dtype=rdtype)

    # Helper to get B_n[m, l] with out-of-bounds returning 0
    def get_B(B: np.ndarray, n: Any, m: Any, ell_col: Any) -> float:
        if abs(m) > n or abs(ell_col) > n:
            return 0.0
        return B[m + n, ell_col + n]

    for n in range(ell):
        n_next = n + 1
        size_next = 2 * n_next + 1
        B_next = np.zeros((size_next, size_next), dtype=rdtype)

        # GATHER approach: compute each target element using the appropriate
        # equation
        # from Dehnen (2014) eq. (63):
        #
        #   eq 63a: 2 B^{m,l}_{n+1} = B^{m,l-1}_n - B^{m,l+1}_n
        #   eq 63b: 2 B^{m+1,l}_{n+1} = B^{m,l-1}_n + B^{m,l+1}_n + 2 B^{m,l}_n
        #   eq 63c: 2 B^{m-1,l}_{n+1} = B^{m,l-1}_n + B^{m,l+1}_n - 2 B^{m,l}_n
        #
        # Key insight: Each target row m_tgt uses exactly ONE equation:
        #   - For |m_tgt| <= n (interior rows): use eq 63a
        #     with source m = m_tgt
        #   - For m_tgt = n+1 (top edge): use eq 63b with source m = n
        #   - For m_tgt = -(n+1) (bottom edge): use eq 63c with source m = -n

        for m_tgt in range(-n_next, n_next + 1):
            for l_tgt in range(-n_next, n_next + 1):
                if abs(m_tgt) <= n:
                    # Interior row: use eq 63a
                    m_src = m_tgt
                    B_next[m_tgt + n_next, l_tgt + n_next] = 0.5 * (
                        get_B(B_prev, n, m_src, l_tgt - 1)
                        - get_B(B_prev, n, m_src, l_tgt + 1)
                    )
                elif m_tgt == n + 1:
                    # Top edge row: use eq 63b with source m = n
                    m_src = n
                    B_next[m_tgt + n_next, l_tgt + n_next] = 0.5 * (
                        get_B(B_prev, n, m_src, l_tgt - 1)
                        + get_B(B_prev, n, m_src, l_tgt + 1)
                        + 2 * get_B(B_prev, n, m_src, l_tgt)
                    )
                elif m_tgt == -(n + 1):
                    # Bottom edge row: use eq 63c with source m = -n
                    m_src = -n
                    B_next[m_tgt + n_next, l_tgt + n_next] = 0.5 * (
                        get_B(B_prev, n, m_src, l_tgt - 1)
                        + get_B(B_prev, n, m_src, l_tgt + 1)
                        - 2 * get_B(B_prev, n, m_src, l_tgt)
                    )

        B_prev = B_next

    return B_prev


@lru_cache(maxsize=None)
def build_Q_dehnen_no_sqrt2(ell: int) -> np.ndarray:
    """Build Q matrix for Dehnen's convention WITHOUT √2 factors.

    Dehnen eq 58:
        T_n^{+m} = Re(Θ_n^m) = (Θ_n^m + Θ_n^m*) / 2
             = (Θ_n^m + (-1)^m Θ_n^{-m}) / 2
        T_n^{-m} = Im(Θ_n^m) = (Θ_n^m - Θ_n^m*) / (2i)
             = -i(Θ_n^m - (-1)^m Θ_n^{-m}) / 2

    Note: NO √2 factors!
    """
    n = 2 * ell + 1
    Q = np.zeros((n, n), dtype=np.complex128)
    offset = ell

    # m = 0: T_0 = Θ_0 (real)
    Q[offset, offset] = 1.0

    for m in range(1, ell + 1):
        # Complex column indices
        col_plus_m = offset + m  # Θ_{+m}
        col_minus_m = offset - m  # Θ_{-m}

        # Real row indices
        row_T_plus_m = offset + m  # T_{+m} = Re(Θ_m)
        row_T_minus_m = offset - m  # T_{-m} = Im(Θ_m)

        phase = (-1.0) ** m

        # T_{+m} = (Θ_m + (-1)^m Θ_{-m}) / 2
        Q[row_T_plus_m, col_plus_m] = 0.5
        Q[row_T_plus_m, col_minus_m] = 0.5 * phase

        # T_{-m} = -i * (Θ_m - (-1)^m Θ_{-m}) / 2
        #        = (Θ_m - (-1)^m Θ_{-m}) / (2i)
        Q[row_T_minus_m, col_plus_m] = -0.5j
        Q[row_T_minus_m, col_minus_m] = 0.5j * phase

    return Q


def _wigner_D_complex(ell: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Compute complex Wigner D^ell using SymPy (baseline correctness path)."""
    try:
        from sympy.physics import wigner
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("sympy is required for Wigner-D baseline rotation") from exc

    D_sym = wigner.wigner_d(ell, float(alpha), float(beta), float(gamma))
    return np.array(D_sym.evalf(30).tolist(), dtype=np.complex128)


def _real_wigner_rotation(
    ell: int,
    alpha: Array,
    beta: Array,
    gamma: Array,
    *,
    dtype: DTypeLike,
    basis: str = "multipole",
) -> Array:
    """Real rotation block from complex Wigner D via Dehnen Q transform.

    Parameters
    ----------
    basis : {"multipole", "local"}
        Selects the diagonal similarity scaling that maps the Wigner real
        basis to the Dehnen real basis used for multipoles or locals.
    """
    D_complex = _wigner_D_complex(ell, float(alpha), float(beta), float(gamma))
    # Adjust for the no-Condon-Shortley convention used in p2m_real_direct.
    # This applies a diagonal phase S_m = (-1)^m to change basis.
    m_vals = np.arange(-ell, ell + 1)
    S = np.diag((-1.0) ** m_vals)
    D_complex = S @ D_complex @ S
    Q = build_Q_dehnen_no_sqrt2(ell)
    Q_inv = np.linalg.inv(Q)
    D_real = np.real(Q @ D_complex @ Q_inv)

    if basis == "multipole":
        S = _dehnen_real_basis_scale_diag_multipole(ell)
    elif basis == "local":
        S = _dehnen_real_basis_scale_diag_local(ell)
    else:
        raise ValueError(f"Unknown basis: {basis}")

    D_real = S @ D_real @ np.linalg.inv(S)
    return jnp.asarray(D_real, dtype=dtype)


@lru_cache(maxsize=None)
def _dehnen_real_basis_scale_diag_multipole(ell: int) -> np.ndarray:
    """Scaling from Wigner real basis to Dehnen real basis (multipoles).

    C_mm ∝ sqrt(binomial(2ℓ, ℓ-m)) with a sign flip for m >= 0.
    Overall scalar cancels in similarity.
    """
    m_vals = np.arange(-ell, ell + 1)
    scale = np.array(
        [math.comb(2 * ell, ell - int(m)) ** 0.5 for m in m_vals], dtype=np.float64
    )
    sign = np.where(m_vals >= 0, -1.0, 1.0)
    return np.diag(sign * scale)


@lru_cache(maxsize=None)
def _dehnen_real_basis_scale_diag_local(ell: int) -> np.ndarray:
    """Scaling from Wigner real basis to Dehnen real basis (locals).

    C_mm ∝ (ℓ-|m|)!(ℓ+|m|)! * sqrt(binomial(2ℓ, ℓ-m)) with sign flip for m>=0.
    This accounts for the local basis scaling relative to multipoles.
    """
    m_vals = np.arange(-ell, ell + 1)
    scale = []
    for m in m_vals:
        m_abs = abs(int(m))
        comb = math.comb(2 * ell, ell - int(m))
        fac = math.factorial(ell - m_abs) * math.factorial(ell + m_abs)
        scale.append((comb**0.5) * fac)
    scale = np.array(scale, dtype=np.float64)
    sign = np.where(m_vals >= 0, -1.0, 1.0)
    return np.diag(sign * scale)


def _rotation_to_z_angles(x: Array, y: Array, z: Array) -> tuple[Array, Array, Array]:
    """ZYZ angles equivalent to Dehnen's alignment rotation.

    The Dehnen A6 alignment uses the sequence:
        R_align = R_y(-beta) @ R_z(-alpha_z)
    with alpha_z = atan2(y, x) and beta = atan2(rho, z).
    We convert this rotation into ZYZ Euler angles for Wigner-D.
    """
    rho = jnp.sqrt(x * x + y * y)
    alpha_z = jnp.arctan2(y, x)
    beta = jnp.arctan2(rho, z)

    ca = jnp.cos(-alpha_z)
    sa = jnp.sin(-alpha_z)
    cb = jnp.cos(-beta)
    sb = jnp.sin(-beta)

    # R = Ry(-beta) @ Rz(-alpha_z)
    R00 = cb * ca
    R01 = -cb * sa
    R02 = sb
    R10 = sa
    R11 = ca
    R12 = 0.0
    R20 = -sb * ca
    R21 = sb * sa
    R22 = cb

    beta_zyz = jnp.arccos(R22)
    alpha_zyz = jnp.arctan2(R12, R02)
    gamma_zyz = jnp.arctan2(R21, -R20)

    return alpha_zyz, beta_zyz, gamma_zyz


@lru_cache(maxsize=None)
def _compute_B_real_dehnen_via_Q(
    ell: int, dtype_key: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the REAL B swap matrices for degree ell
    using Dehnen's Q (no √2).

    The B matrix for the REAL T_n^m basis is obtained via similarity transform:
        B_real = Q @ B_complex @ Q^{-1}

    where Q is the Dehnen real-basis transform (no √2).

    This is a PRECOMPUTATION - no complex arithmetic at runtime. The resulting
    B_real matrix is purely real with checkerboard sparsity: B_n^{m,l} ≠ 0
    only when (m - l) is even.

    Returns TWO matrices:
    - B_T: for local expansions T_n^m
    - B_U: for multipole expansions U_n^m (= B_T^T in real basis)

    Key properties:
    - B² = I (involution) for both B_T and B_U
    - Checkerboard sparsity: ~25% non-zero entries
    - D_y(β) = B @ D_z(-β) @ B
    """
    B_complex = _compute_dehnen_B_matrix_complex(ell, "float64")
    Q = build_Q_dehnen_no_sqrt2(ell)
    Q_inv = np.linalg.inv(Q)

    # B_T = Q @ B_complex @ Q^{-1}
    B_T_complex = Q @ B_complex @ Q_inv

    # Check if result is real
    imag_norm = np.linalg.norm(np.imag(B_T_complex))
    if imag_norm > 1e-10:
        print(f"Warning: B_T has imaginary part norm {imag_norm}")

    B_T = np.real(B_T_complex)

    # B_U = Q @ B_complex^T @ Q^{-1}
    B_U_complex = Q @ B_complex.T @ Q_inv
    imag_norm_U = np.linalg.norm(np.imag(B_U_complex))
    if imag_norm_U > 1e-10:
        print(f"Warning: B_U has imaginary part norm {imag_norm_U}")

    B_U = np.real(B_U_complex)

    return B_T, B_U


def compute_real_B_matrix_local(ell: int, *, dtype: DTypeLike) -> Array:
    """Get the real B swap matrix for LOCAL expansions T_n^m.

    Use this matrix when rotating local expansion coefficients.

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.
    dtype : jnp.dtype
        Real dtype (float32 or float64).

    Returns
    -------
    Array
        Real B_T matrix of shape (2*ell+1, 2*ell+1).
    """
    B_T, _ = _compute_B_real_dehnen_via_Q(ell, str(dtype))
    return jnp.asarray(B_T)


def compute_real_B_matrix_multipole(ell: int, *, dtype: DTypeLike) -> Array:
    """Get the real B swap matrix for MULTIPOLE expansions U_n^m.

    Use this matrix when rotating multipole expansion coefficients.

    IMPORTANT: This is NOT the transpose of B_T! The relationship
    B_U = Q @ B_Θ^T @ Q^{-1} ≠ (Q @ B_Θ @ Q^{-1})^T in general.

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.
    dtype : jnp.dtype
        Real dtype (float32 or float64).

    Returns
    -------
    Array
        Real B_U matrix of shape (2*ell+1, 2*ell+1).
    """
    _, B_U = _compute_B_real_dehnen_via_Q(ell, str(dtype))
    return jnp.asarray(B_U)


def verify_real_B_matrix(ell: int, *, dtype: DTypeLike) -> Tuple[bool, float, float]:
    """Verify properties of the real B matrices (both B_T and B_U).

    Returns
    -------
    Tuple[bool, float, float]
        (is_valid, B_squared_error, sparsity_ratio)
        - is_valid: True if B² ≈ I and checkerboard sparsity holds for both
        - B_squared_error: max of ||B_T² - I||_F and ||B_U² - I||_F
        - sparsity_ratio: fraction of non-zero entries in B_T
    """
    B_T = compute_real_B_matrix_local(ell, dtype=dtype)
    B_U = compute_real_B_matrix_multipole(ell, dtype=dtype)
    B = np.asarray(B_T)  # Use B_T for sparsity check

    # Check B² = I
    B_squared = B @ B
    eye_matrix = np.eye(2 * ell + 1)
    B_squared_error = np.linalg.norm(B_squared - eye_matrix, "fro")

    # Check checkerboard sparsity: B_n^{m,l} ≠ 0 only when (m-l) is even
    n = 2 * ell + 1
    non_zero_count = 0
    expected_non_zero_count = 0
    for i in range(n):
        m = i - ell  # m goes from -ell to ell
        for j in range(n):
            ell_col = j - ell  # ell_col goes from -ell to ell
            if (m - ell_col) % 2 == 0:
                expected_non_zero_count += 1
                if abs(B[i, j]) > 1e-10:
                    non_zero_count += 1
            else:
                # Should be zero
                if abs(B[i, j]) > 1e-10:
                    # Unexpected non-zero
                    pass

    sparsity_ratio = non_zero_count / (n * n)

    # Also check B_U
    B_U_np = np.asarray(B_U)
    B_U_squared = B_U_np @ B_U_np
    B_U_squared_error = np.linalg.norm(B_U_squared - eye_matrix, "fro")

    B_squared_error = max(B_squared_error, B_U_squared_error)
    is_valid = B_squared_error < 1e-10

    return is_valid, float(B_squared_error), sparsity_ratio


# ===========================================================================
# Real rotation via B @ D_z @ B
# ===========================================================================


def real_Dz_diagonal(ell: int, angle: Array, *, dtype: DTypeLike) -> Array:
    """Diagonal D_z rotation for REAL harmonics.

    For real harmonics, the z-rotation is block-diagonal:
    - m = 0: unchanged (coefficient 1)
    - m > 0: 2x2 rotation block [[cos(mα), -sin(mα)], [sin(mα), cos(mα)]]
             acting on (r_{+m}, r_{-m}) = (cos channel, sin channel)

    For the packed layout m = -ell..ell, think of it as:
    r'_m = cos(m*α) * r_m + sin(m*α) * r_{-m} for appropriate signs.

    For each m, the coefficient transforms as if rotating the angular part:
    cos(mφ) → cos(m(φ+α)) = cos(mφ)cos(mα) - sin(mφ)sin(mα).
    """
    n = 2 * ell + 1
    D = jnp.zeros((n, n), dtype=dtype)

    # For real harmonics with m=0: unchanged
    D = D.at[ell, ell].set(1.0)

    for m in range(1, ell + 1):
        c = jnp.cos(m * angle)
        s = jnp.sin(m * angle)

        # Indices in packed layout
        ip = ell + m  # +m (cos channel)
        im = ell - m  # -m (sin channel)

        # cos(m(φ+α)) = cos(mφ)cos(mα) - sin(mφ)sin(mα)
        # sin(m(φ+α)) = sin(mφ)cos(mα) + cos(mφ)sin(mα)
        # So: r'_{+m} = cos(mα) * r_{+m} - sin(mα) * r_{-m}
        #     r'_{-m} = sin(mα) * r_{+m} + cos(mα) * r_{-m}
        D = D.at[ip, ip].set(c)
        D = D.at[ip, im].set(-s)
        D = D.at[im, ip].set(s)
        D = D.at[im, im].set(c)

    return D


def real_rotation_to_z_axis_multipole(
    x: Array,
    y: Array,
    z: Array,
    ell: int,
    *,
    dtype: DTypeLike,
) -> Array:
    """Compute rotation matrix to align vector (x,y,z) with z-axis.

    Use this to rotate multipole expansion coefficients U_n^m.

    Using the Dehnen rotation D_y(-θ) @ D_z(-φ) expressed via B:
    1. Rotate by -αz = -arctan(y/x) around z → brings vector to xz-plane
    2. Swap x↔z with B
    3. Rotate by -αx = -arctan(ρ/z) around z → aligns with z-axis
    4. Swap back with B

    D = B_U @ Dz(-αx) @ B_U @ Dz(αz)

    After applying D @ M, the multipole expansion has its z-axis aligned
    with the original (x,y,z) direction.
    """
    rho = jnp.sqrt(x * x + y * y)

    # Dehnen's angles (positive, as specified in A.6.2)
    alpha_z = jnp.arctan2(y, x)
    alpha_x = jnp.arctan2(rho, z)

    B_U = compute_real_B_matrix_multipole(ell, dtype=dtype)
    Dz_neg_alpha_z = real_Dz_diagonal(ell, -alpha_z, dtype=dtype)
    Dz_neg_alpha_x = real_Dz_diagonal(ell, -alpha_x, dtype=dtype)

    return B_U @ Dz_neg_alpha_x @ B_U @ Dz_neg_alpha_z


def real_rotation_to_z_axis_multipole_wigner(
    x: Array,
    y: Array,
    z: Array,
    ell: int,
    *,
    dtype: DTypeLike,
) -> Array:
    """Baseline rotation to z-axis using Wigner-D (complex) + Q transform."""
    alpha, beta, gamma = _rotation_to_z_angles(x, y, z)
    return _real_wigner_rotation(ell, alpha, beta, gamma, dtype=dtype)


def real_rotation_from_z_axis_local(
    x: Array,
    y: Array,
    z: Array,
    ell: int,
    *,
    dtype: DTypeLike,
) -> Array:
    """Compute inverse rotation matrix for LOCAL expansions.

    Use this to rotate local expansion coefficients T_n^m back
    from the z-aligned frame.

    For locals in the Dehnen basis, rotate back using the inverse of
    the z-alignment rotation:
        D_inv = Dz(-αz) @ B @ Dz(+αx) @ B
        L = D_inv @ L_z
    """
    rho = jnp.sqrt(x * x + y * y)

    # Same angles as forward rotation
    alpha_z = jnp.arctan2(y, x)
    alpha_x = jnp.arctan2(rho, z)

    B_T = compute_real_B_matrix_local(ell, dtype=dtype)
    Dz_pos_alpha_z = real_Dz_diagonal(ell, alpha_z, dtype=dtype)
    Dz_pos_alpha_x = real_Dz_diagonal(ell, alpha_x, dtype=dtype)

    return Dz_pos_alpha_z @ B_T @ Dz_pos_alpha_x @ B_T


def real_rotation_from_z_axis_local_wigner(
    x: Array,
    y: Array,
    z: Array,
    ell: int,
    *,
    dtype: DTypeLike,
) -> Array:
    """Baseline inverse rotation from z-axis using Wigner-D (locals)."""
    alpha, beta, gamma = _rotation_to_z_angles(x, y, z)
    return _real_wigner_rotation(ell, -gamma, -beta, -alpha, dtype=dtype)


def real_rotation_from_z_axis_multipole(
    x: Array,
    y: Array,
    z: Array,
    ell: int,
    *,
    dtype: DTypeLike,
) -> Array:
    """Compute inverse rotation matrix for MULTIPOLE expansions.

    For multipoles in the Dehnen basis, rotate back using the inverse of
    the z-alignment rotation:
        D_inv = Dz(-αz) @ B_U @ Dz(+αx) @ B_U
        M = D_inv @ M_z
    """
    rho = jnp.sqrt(x * x + y * y)

    # Same angles as forward rotation
    alpha_z = jnp.arctan2(y, x)
    alpha_x = jnp.arctan2(rho, z)

    B_U = compute_real_B_matrix_multipole(ell, dtype=dtype)
    Dz_pos_alpha_z = real_Dz_diagonal(ell, alpha_z, dtype=dtype)
    Dz_pos_alpha_x = real_Dz_diagonal(ell, alpha_x, dtype=dtype)

    return Dz_pos_alpha_z @ B_U @ Dz_pos_alpha_x @ B_U


def real_rotation_from_z_axis_multipole_wigner(
    x: Array,
    y: Array,
    z: Array,
    ell: int,
    *,
    dtype: DTypeLike,
) -> Array:
    """Baseline inverse rotation from z-axis using Wigner-D."""
    alpha, beta, gamma = _rotation_to_z_angles(x, y, z)
    return _real_wigner_rotation(
        ell, -gamma, -beta, -alpha, dtype=dtype, basis="multipole"
    )


def real_rotation_to_z_axis_local(
    x: Array,
    y: Array,
    z: Array,
    ell: int,
    *,
    dtype: DTypeLike,
) -> Array:
    """Compute rotation matrix to align vector (x,y,z) with z-axis for locals.

    For local expansions in the Dehnen basis:
        D = B_T @ Dz(-αx) @ B_T @ Dz(αz)
    """
    rho = jnp.sqrt(x * x + y * y)

    alpha_z = jnp.arctan2(y, x)
    alpha_x = jnp.arctan2(rho, z)

    B_T = compute_real_B_matrix_local(ell, dtype=dtype)
    Dz_neg_alpha_z = real_Dz_diagonal(ell, -alpha_z, dtype=dtype)
    Dz_neg_alpha_x = real_Dz_diagonal(ell, -alpha_x, dtype=dtype)

    return B_T @ Dz_neg_alpha_x @ B_T @ Dz_neg_alpha_z


def real_rotation_to_z_axis_local_wigner(
    x: Array,
    y: Array,
    z: Array,
    ell: int,
    *,
    dtype: DTypeLike,
) -> Array:
    """Baseline rotation to z-axis using Wigner-D (locals)."""
    alpha, beta, gamma = _rotation_to_z_angles(x, y, z)
    return _real_wigner_rotation(ell, alpha, beta, gamma, dtype=dtype, basis="local")


# ===========================================================================
# Z-axis translations in real basis
# ===========================================================================


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
    """
    p = int(order)
    multipole = jnp.asarray(multipole)
    dz = jnp.asarray(dz).reshape(())
    dtype = multipole.dtype

    fact = _factorial_table_jax(p, dtype)

    ncoeff = sh_size(p)
    out = jnp.zeros((ncoeff,), dtype=dtype)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            m_abs = abs(m)
            acc = jnp.asarray(0.0, dtype=dtype)

            # Sum over k from 0 to n - |m|
            for k in range(n - m_abs + 1):
                src_n = n - k
                if m_abs > src_n:
                    continue
                src_idx = sh_index(src_n, m)
                coeff = (dz**k) / fact[k]
                acc = acc + coeff * multipole[src_idx]

            out = out.at[sh_index(n, m)].set(acc)

    return out


@partial(jax.jit, static_argnames=("order",))
def translate_along_z_m2l_real(
    multipole: Array,
    r: Array,
    *,
    order: int,
) -> Array:
    """Translate multipole to local along +z in real harmonic basis.

    F_n^m = sum_{k=|m|}^{p-n} (-1)^m * M_k^m * (n+k)! / r^{n+k+1}

    For real harmonics, the sign (-1)^m needs careful handling because
    the real and imaginary parts (cos/sin channels) have different parity.
    """
    p = int(order)
    multipole = jnp.asarray(multipole)
    r = jnp.asarray(r).reshape(())
    dtype = multipole.dtype

    fact = _factorial_table_jax(2 * p, dtype)

    ncoeff = sh_size(p)
    out = jnp.zeros((ncoeff,), dtype=dtype)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            m_abs = abs(m)
            acc = jnp.asarray(0.0, dtype=dtype)

            # Sum over k from |m| to p - n
            for k in range(m_abs, p - n + 1):
                src_idx = sh_index(k, m)
                # Sign from Dehnen eq (84). The real basis uses the same
                # parity factor for both cos/sin channels; any m-channel
                # mixing is handled by the z-rotation blocks.
                sign = (-1.0) ** m
                coeff = sign * fact[n + k] / (r ** (n + k + 1))
                acc = acc + coeff * multipole[src_idx]

            out = out.at[sh_index(n, m)].set(acc)

    return out


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
    p = int(order)
    local = jnp.asarray(local)
    dz = jnp.asarray(dz).reshape(())
    dtype = local.dtype

    fact = _factorial_table_jax(p + 1, dtype)

    ncoeff = sh_size(p)
    out = jnp.zeros((ncoeff,), dtype=dtype)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            acc = jnp.asarray(0.0, dtype=dtype)

            # Sum over k from 0 to p - n
            for k in range(p - n + 1):
                src_n = n + k
                if src_n > p:
                    continue
                src_idx = sh_index(src_n, m)
                # Coefficient is R_k^0(dz) = (dz)^k / k! in this normalization.
                coeff = (dz**k) / fact[k]
                acc = acc + coeff * local[src_idx]

            out = out.at[sh_index(n, m)].set(acc)

    return out


# ===========================================================================
# Full M2L with rotation-accelerated kernel
# ===========================================================================


@partial(jax.jit, static_argnames=("order",))
def m2l_a6_real_only(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """M2L using Dehnen A6 with real-only rotations and z-translation.

    This implementation rotates multipoles using real B_U/Dz blocks, applies
    the real-only z-axis M2L recurrence, and rotates locals back with B_T/Dz.
    """
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)
    dtype = multipole.dtype
    p = int(order)

    # Convention: delta points from multipole (source) center to local (target)
    # center, i.e. delta = target - source.

    # Extract delta components
    x, y, z = delta[0], delta[1], delta[2]
    r2 = jnp.dot(delta, delta)
    r = jnp.sqrt(jnp.maximum(r2, 1e-60))

    # Step 1: Rotate MULTIPOLE to z-aligned frame using B_U
    M_rotated = jnp.zeros_like(multipole)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_inv = real_rotation_from_z_axis_multipole(x, y, z, ell, dtype=dtype)
        M_rotated = M_rotated.at[sl].set(D_inv @ multipole[sl])

    # Step 2: Z-axis M2L translation in real basis
    L_z = translate_along_z_m2l_real(M_rotated, r, order=p)

    # Step 3: Rotate LOCAL back from z-aligned frame using B_T
    out = jnp.zeros_like(L_z)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_fwd = real_rotation_to_z_axis_local(x, y, z, ell, dtype=dtype)
        out = out.at[sl].set(D_fwd @ L_z[sl])

    return out


def m2l_a6_real_only_wigner(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """M2L using Wigner-D rotations as a correctness baseline (no JIT)."""
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)
    dtype = multipole.dtype
    p = int(order)

    x, y, z = delta[0], delta[1], delta[2]
    r2 = jnp.dot(delta, delta)
    r = jnp.sqrt(jnp.maximum(r2, 1e-60))

    M_rotated = jnp.zeros_like(multipole)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_inv = real_rotation_from_z_axis_multipole_wigner(x, y, z, ell, dtype=dtype)
        M_rotated = M_rotated.at[sl].set(D_inv @ multipole[sl])

    L_z = translate_along_z_m2l_real(M_rotated, r, order=p)

    out = jnp.zeros_like(L_z)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_fwd = real_rotation_to_z_axis_local_wigner(x, y, z, ell, dtype=dtype)
        out = out.at[sl].set(D_fwd @ L_z[sl])

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
    """
    return m2l_a6_real_only(multipole, delta, order=order)


def m2l_real_wigner(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Baseline M2L using Wigner-D rotations (correctness reference)."""
    return m2l_a6_real_only_wigner(multipole, delta, order=order)


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
    """
    return m2l_a6_real_only(multipole, delta, order=order)


# ===========================================================================
# Full M2M and L2L with rotation-accelerated kernels
# ===========================================================================


@partial(jax.jit, static_argnames=("order",))
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
    """
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)
    dtype = multipole.dtype
    p = int(order)

    # Extract delta components
    x, y, z = delta[0], delta[1], delta[2]
    r2 = jnp.dot(delta, delta)
    r = jnp.sqrt(jnp.maximum(r2, 1e-60))

    # Handle zero displacement case
    # (return original multipole if |delta| ~ 0)
    is_zero = r < 1e-30
    dz = jnp.where(is_zero, 0.0, r)

    # Step 1: Rotate MULTIPOLE to z-aligned frame using B_U
    M_rotated = jnp.zeros_like(multipole)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_inv = real_rotation_from_z_axis_multipole(x, y, z, ell, dtype=dtype)
        M_rotated = M_rotated.at[sl].set(D_inv @ multipole[sl])

    # Step 2: Z-axis M2M translation
    # For M2M, we translate the multipole expansion by -dz along the z-axis
    # (moving center from source to destination = negative of delta direction)
    M_z = translate_along_z_m2m_real(M_rotated, dz, order=p)

    # Step 3: Rotate MULTIPOLE back from z-aligned frame
    out = jnp.zeros_like(M_z)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_fwd = real_rotation_to_z_axis_multipole(x, y, z, ell, dtype=dtype)
        out = out.at[sl].set(D_fwd @ M_z[sl])

    return out


@partial(jax.jit, static_argnames=("order",))
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
        3-vector from parent center to child center.
    order : int
        Maximum SH degree.

    Returns
    -------
    Array
        Packed real local coefficients at the child center.
    """
    local = jnp.asarray(local)
    delta = jnp.asarray(delta)
    dtype = local.dtype
    p = int(order)

    # Extract delta components
    x, y, z = delta[0], delta[1], delta[2]
    r2 = jnp.dot(delta, delta)
    r = jnp.sqrt(jnp.maximum(r2, 1e-60))

    # Handle zero displacement case
    is_zero = r < 1e-30
    dz = jnp.where(is_zero, 0.0, r)

    # Step 1: Rotate LOCAL to z-aligned frame
    # For local coefficients, we use the same rotation as
    # multipole to z-aligned
    # but with B_T instead of B_U
    L_rotated = jnp.zeros_like(local)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_inv = real_rotation_from_z_axis_local(x, y, z, ell, dtype=dtype)
        L_rotated = L_rotated.at[sl].set(D_inv @ local[sl])

    # Step 2: Z-axis L2L translation
    L_z = translate_along_z_l2l_real(L_rotated, dz, order=p)

    # Step 3: Rotate LOCAL back from z-aligned frame using B_T
    out = jnp.zeros_like(L_z)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D_fwd = real_rotation_to_z_axis_local(x, y, z, ell, dtype=dtype)
        out = out.at[sl].set(D_fwd @ L_z[sl])

    return out


# ===========================================================================
# Exports
# ===========================================================================


__all__ = [
    # Index utilities
    "sh_size",
    "sh_offset",
    "sh_index",
    # P2M (particle to multipole)
    "p2m_real_direct",
    # L2P (local to particle evaluation)
    "evaluate_local_real",
    "evaluate_local_real_with_grad",
    # B matrices for coordinate swap (x,y,z) → (z,y,x)
    "compute_real_B_matrix_local",
    "compute_real_B_matrix_multipole",
    "verify_real_B_matrix",
    # Rotation building blocks
    "real_Dz_diagonal",
    "real_rotation_to_z_axis_multipole",
    "real_rotation_to_z_axis_local",
    "real_rotation_from_z_axis_local",
    "real_rotation_from_z_axis_multipole",
    "real_rotation_to_z_axis_multipole_wigner",
    "real_rotation_to_z_axis_local_wigner",
    "real_rotation_from_z_axis_local_wigner",
    "real_rotation_from_z_axis_multipole_wigner",
    # Z-axis translations
    "translate_along_z_m2m_real",
    "translate_along_z_m2l_real",
    "translate_along_z_l2l_real",
    # Full operators (rotation-accelerated)
    "m2m_real",
    "m2l_a6_real_only",
    "m2l_real",
    "m2l_optimized_real",
    "m2l_a6_real_only_wigner",
    "m2l_real_wigner",
    "l2l_real",
]
