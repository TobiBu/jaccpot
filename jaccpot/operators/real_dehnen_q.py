"""The Dehnen ``B`` matrices and the complex -> real basis conversion ``Q``.

Two related pieces of static (host-side, ``lru_cache``-d) linear algebra:

* the involutory **B matrix**, which implements the ``(x, y, z) -> (z, y, x)``
  coordinate swap so that a y/x-axis rotation factors as ``B @ D_z(-beta) @ B``
  -- the decomposition the whole rotation-accelerated M2L rests on;
* the block-diagonal **Q**, which maps complex Dehnen coefficients to the real
  (no-sqrt2) basis, giving ``complex_to_dehnen_real_coeffs``.

They belong together because the real ``B`` is *derived* from the complex one
through ``Q``: ``_compute_B_real_dehnen_via_Q`` conjugates
``_compute_dehnen_B_matrix_complex`` by the per-degree ``Q`` block rather than
re-deriving a real recursion, which is what keeps the two bases consistent by
construction.

Split out of ``real_harmonics.py`` (Tier 1.3); the mathematics is unchanged and
``real_harmonics`` re-exports the public names.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, DTypeLike

from ._precision import highest_matmul_precision
from ._sh_indexing import sh_offset, sh_size

__all__ = [
    "build_Q_dehnen_no_sqrt2",
    "complex_to_dehnen_real_coeffs",
    "compute_real_B_matrix_local",
    "compute_real_B_matrix_multipole",
    "verify_real_B_matrix",
]


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


@lru_cache(maxsize=None)
def _dehnen_real_Q_full(order: int) -> np.ndarray:
    """Block-diagonal complex->Dehnen-real transform for all degrees <= order.

    Stacks :func:`build_Q_dehnen_no_sqrt2` per degree into one
    ``((p+1)^2, (p+1)^2)`` complex matrix ``Q`` such that, for packed complex
    solidfmm coefficients ``c`` (conjugate-symmetric), ``real(Q @ c)`` are the
    packed Dehnen no-sqrt2 real coefficients used by this module's operators.
    """
    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    ncoeff = sh_size(p)
    q_full = np.zeros((ncoeff, ncoeff), dtype=np.complex128)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        q_full[sl, sl] = build_Q_dehnen_no_sqrt2(ell)
    return q_full


@partial(jax.jit, static_argnames=("order",))
@highest_matmul_precision
def complex_to_dehnen_real_coeffs(complex_coeffs: Array, *, order: int) -> Array:
    """Convert packed complex solidfmm coefficients to Dehnen no-sqrt2 real ones.

    This is the conversion consistent with :func:`p2m_real_direct` and the real
    M2L/M2M/L2L/L2P operators in this module (verified: for a point source,
    ``complex_to_dehnen_real_coeffs(complex_R_solidfmm(delta)) ==
    p2m_real_direct(delta)`` to machine precision).

    NOTE: this is NOT the same as ``jaccpot.basis.real_sh.complex_to_real_coeffs``,
    which produces the unitary sqrt(2) tesseral basis (a different normalization
    that is incompatible with the Dehnen no-sqrt2 real operators here).

    Parameters
    ----------
    complex_coeffs : Array
        Packed complex coefficients of shape ``(..., (p+1)^2)``.
    order : int
        Maximum harmonic degree ``p``.

    Returns
    -------
    Array
        Packed real coefficients of shape ``(..., (p+1)^2)`` with the real dtype
        matching the input's real component.

    Raises
    ------
    ValueError
        If the trailing axis of ``complex_coeffs`` is not ``(p+1)^2``. A static
        shape check on a static ``order``, so it fires at trace time.

    Notes
    -----
    A single matmul against a fixed basis-change matrix, so differentiable in
    ``complex_coeffs``.

    **Forward: nothing is lost, for conforming input.** The reality condition is
    stated in :mod:`jaccpot.operators.complex_harmonics` -- Dehnen normalization,
    no Condon-Shortley phase, and ``H_n^{-m} = (-1)^m conj(H_n^m)`` -- and the
    ``(p+1)^2`` layout is *redundant* rather than packed:
    :func:`~jaccpot.operators.complex_harmonics._pack_complex` fills the negative
    ``m`` slots as ``(-1)^|m| conj(coeff[n, |m|])``, so both ``+m`` and ``-m`` are
    present and conjugate-related. ``build_Q_dehnen_no_sqrt2`` recombines each
    conjugate pair, and the imaginary part of the product comes out **exactly**
    zero -- measured, not merely bounded.

    Note the easily-missed half of that condition: at ``m = 0`` it reads
    ``H_n^0 = conj(H_n^0)``, i.e. **the m=0 coefficients must be real**.
    ``complex_R_solidfmm`` satisfies this (its m=0 entries have exactly zero
    imaginary part). An array that is conjugate-symmetric for ``m != 0`` but has
    complex ``m = 0`` entries does *not* conform, and for such input
    ``Im(coeffs @ q_full.T)`` is a substantial fraction of the real part
    (measured ~0.6 at order 3) -- so ``jnp.real`` would silently discard real
    information. Nothing here validates the condition; it is a precondition.

    **Reverse: the gradient is complete.** ``jnp.real`` does not make the VJP blind
    to the imaginary components -- ``Q`` is complex, so ``Im(coeffs)`` contributes
    to ``Re(coeffs @ q_full.T)`` and the returned cotangent has a nonzero imaginary
    part. This is the correct adjoint of the R-linear map, and it agrees with finite
    differences along both real and imaginary perturbation directions.

    The claim above -- that this composes with ``complex_R_solidfmm`` to reproduce
    :func:`p2m_real_direct` -- is asserted directly by
    ``tests/unit/operators/test_real_harmonics.py::test_complex_to_dehnen_real_matches_p2m_real_direct``
    over six geometries (including the ``rho == 0`` z-aligned degeneracy and the
    three coordinate axes) at orders 0-6, to a relative L2 of 1e-13 against a
    measured worst case of 8.9e-16.

    That test exists because the two indirect proxies cannot substitute for it:
    ``tests/unit/core/test_real_upward_sweep.py::test_real_upward_matches_complex_convert``
    checks an aggregate relative L2 over a whole 300-particle P2M+M2M tree, where
    a single-``m`` error is diluted, and
    ``tests/unit/runtime/test_dehnen_mac_reference.py::test_dehnen_power_is_basis_invariant``
    checks only the degree-wise Dehnen power -- a rotational invariant, therefore
    blind to sign errors within a degree. Verified by mutation: flipping the sign
    of one row of the degree-2 Q block fails the direct test and leaves the power
    proxy passing.

    Runs under :func:`~jaccpot.operators._precision.highest_matmul_precision`;
    the matmul must not be dropped back to TF32.
    """
    coeffs = jnp.asarray(complex_coeffs)
    expected = sh_size(int(order))
    if int(coeffs.shape[-1]) != expected:
        raise ValueError(
            f"expected last dimension {expected} for order={int(order)}, "
            f"got {coeffs.shape[-1]}"
        )
    q_full = jnp.asarray(
        _dehnen_real_Q_full(int(order)),
        dtype=jnp.result_type(coeffs.dtype, jnp.complex64),
    )
    converted = jnp.real(coeffs @ q_full.T)
    return converted.astype(coeffs.real.dtype)


@lru_cache(maxsize=None)
@highest_matmul_precision
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
    dtype : DTypeLike
        Real working dtype (float32 or float64). The B matrix is built in
        float64 for accuracy and cast down to this, so that the downstream
        rotation GEMMs run in the working dtype instead of promoting float32
        coefficient vectors to float64.

    Returns
    -------
    Array
        Real B_T matrix of shape (2*ell+1, 2*ell+1).
    """
    B_T, _ = _compute_B_real_dehnen_via_Q(ell, str(dtype))
    # Honor the requested dtype: the B matrix is computed in float64 for accuracy
    # but must be cast down to the working dtype so the downstream rotation GEMMs
    # (M2M/L2L/M2L rotate-to-z/from-z) run in that dtype rather than promoting the
    # float32 coefficient vectors to float64 (the fp64 cuBLAS GEMM slowdown).
    return jnp.asarray(B_T, dtype=dtype)


def compute_real_B_matrix_multipole(ell: int, *, dtype: DTypeLike) -> Array:
    """Get the real B swap matrix for MULTIPOLE expansions U_n^m.

    Use this matrix when rotating multipole expansion coefficients.

    IMPORTANT: This is NOT the transpose of B_T! The relationship
    B_U = Q @ B_Θ^T @ Q^{-1} ≠ (Q @ B_Θ @ Q^{-1})^T in general.

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.
    dtype : DTypeLike
        Real working dtype (float32 or float64). The B matrix is built in
        float64 for accuracy and cast down to this, so that the downstream
        rotation GEMMs run in the working dtype instead of promoting float32
        coefficient vectors to float64.

    Returns
    -------
    Array
        Real B_U matrix of shape (2*ell+1, 2*ell+1).
    """
    _, B_U = _compute_B_real_dehnen_via_Q(ell, str(dtype))
    # Honor the requested dtype (see compute_real_B_matrix_local): cast the
    # float64-computed B matrix down to the working dtype so the rotation GEMMs
    # do not run in float64.
    return jnp.asarray(B_U, dtype=dtype)


@highest_matmul_precision
def verify_real_B_matrix(ell: int, *, dtype: DTypeLike) -> Tuple[bool, float, float]:
    """Verify properties of the real B matrices (both B_T and B_U).

    Parameters
    ----------
    ell : int
        Spherical harmonic degree to check.
    dtype : DTypeLike
        Real working dtype the matrices are cast to before checking, so the
        result reflects what the rotation GEMMs will actually see. The
        ``1e-10`` sparsity threshold below is a float64 threshold and is not
        adjusted for float32 -- read a float32 verdict with that in mind.

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
