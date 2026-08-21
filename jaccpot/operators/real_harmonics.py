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

The B matrix implements the coordinate swap (x,y,z) → (z,y,x) for real
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

from ._sh_indexing import sh_index, sh_offset, sh_size
from .real_dehnen_q import (
    _compute_dehnen_B_matrix_complex,
    _dehnen_real_Q_full,
    build_Q_dehnen_no_sqrt2,
    complex_to_dehnen_real_coeffs,
    compute_real_B_matrix_local,
    compute_real_B_matrix_multipole,
    verify_real_B_matrix,
)
from .real_p2m_l2p import (
    evaluate_local_real,
    evaluate_local_real_derivative_tower,
    evaluate_local_real_derivative_tower_batch,
    evaluate_local_real_with_grad,
    p2m_real_direct,
)
from .real_rotations import (
    real_Dz_diagonal,
    real_rotation_from_z_axis_local,
    real_rotation_from_z_axis_multipole,
    real_rotation_to_z_axis_local,
    real_rotation_to_z_axis_multipole,
    real_transverse_generators,
)
from .real_translations import (
    l2l_real,
    m2l_a6_real_only,
    m2l_optimized_real,
    m2l_real,
    m2m_real,
    translate_along_z_l2l_real,
    translate_along_z_m2l_real,
    translate_along_z_m2m_real,
    z_m2l_translation_tables,
)

# RE-EXPORTS. Both names are imported for other modules to reach through this
# aggregator and are unused *here*. Holding references makes that a fact the
# interpreter can see, so `pyflakes` reports only genuinely dead re-exports for
# this module. `__all__` cannot carry them: it declares the PUBLIC surface, and
# these are private by name -- which is why an `__all__` alone left this module
# reporting 10 unused imports (audit A.11).
_REEXPORTS = (
    _compute_dehnen_B_matrix_complex,
    _dehnen_real_Q_full,
)

# ===========================================================================
# Exports
# ===========================================================================


__all__ = [
    # Index utilities
    "sh_size",
    "sh_offset",
    "sh_index",
    # Complex -> Dehnen real basis conversion
    "build_Q_dehnen_no_sqrt2",
    "complex_to_dehnen_real_coeffs",
    # P2M (particle to multipole)
    "p2m_real_direct",
    # L2P (local to particle evaluation)
    "evaluate_local_real",
    "evaluate_local_real_with_grad",
    "evaluate_local_real_derivative_tower",
    "evaluate_local_real_derivative_tower_batch",
    # B matrices for coordinate swap (x,y,z) → (z,y,x)
    "compute_real_B_matrix_local",
    "compute_real_B_matrix_multipole",
    "verify_real_B_matrix",
    # Rotation building blocks
    "real_Dz_diagonal",
    "real_transverse_generators",
    "real_rotation_to_z_axis_multipole",
    "real_rotation_to_z_axis_local",
    "real_rotation_from_z_axis_local",
    "real_rotation_from_z_axis_multipole",
    # Z-axis translations
    "translate_along_z_m2m_real",
    "translate_along_z_m2l_real",
    "z_m2l_translation_tables",
    "translate_along_z_l2l_real",
    # Full operators (rotation-accelerated)
    "m2m_real",
    "m2l_a6_real_only",
    "m2l_real",
    "m2l_optimized_real",
    "l2l_real",
]
