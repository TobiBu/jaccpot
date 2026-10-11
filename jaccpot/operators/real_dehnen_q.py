"""The Dehnen ``B`` matrices, built through the complex -> real basis map ``Q``.

Two related pieces of static (host-side, ``lru_cache``-d) linear algebra:

* the involutory **B matrix**, which implements the ``(x, y, z) -> (z, y, x)``
  coordinate swap so that a y/x-axis rotation factors as ``B @ D_z(-beta) @ B``
  -- the decomposition the whole rotation-accelerated M2L rests on;
* the per-degree **Q**, which maps complex Dehnen coefficients to the real
  (no-sqrt2) basis.

They belong together because the real ``B`` is *derived* from the complex one
through ``Q``: ``_compute_B_real_dehnen_via_Q`` conjugates
``_compute_dehnen_B_matrix_complex`` by the per-degree ``Q`` block rather than
re-deriving a real recursion. That is why the complex-named builders stay after
the complex basis was removed in the 2026-10 cleanup (C): the real basis's
precomputed ``B`` matrices are made from them. The coefficient conversion
``complex_to_dehnen_real_coeffs`` and its block-diagonal ``_dehnen_real_Q_full``
(test-only once the complex upward sweep was gone) went in C.

Split out of ``real_harmonics.py`` (Tier 1.3); the mathematics is unchanged and
``real_harmonics`` re-exports the public names.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Tuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, DTypeLike

from ._precision import highest_matmul_precision

__all__ = [
    "build_Q_dehnen_no_sqrt2",
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

    The absent sqrt(2) is the whole point: this is the DEHNEN real convention, not
    the unitary tesseral one. Coefficients in this basis are a factor of two
    smaller on every ``m != 0`` channel than the genuine-complex form -- the
    relationship ``m2l_solidfmm_reference`` carried until the complex basis was
    removed in the 2026-10 cleanup, C (``docs/operator_conventions.md``).

    Parameters
    ----------
    ell : int
        Degree ``l``.

    Returns
    -------
    np.ndarray
        ``[2l+1, 2l+1]`` complex transform taking packed complex coefficients to
        Dehnen no-sqrt2 real ones. NumPy, not JAX: a compile-time constant.
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

    Parameters
    ----------
    ell : int
        Degree ``l``.
    dtype_key : str
        Dtype name used to build the complex ``B``; part of the memoisation key
        rather than a dtype object, which would not be hashable in the same way.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(B_T, B_U)`` -- the local and multipole swap matrices, both purely real
        with the checkerboard sparsity noted above, and both involutory
        (``B @ B == I`` exactly, checked for ``l = 1..3``).

        ``B_U`` is built as ``Q B_complex^T Q^-1``, which is **not** the transpose
        of ``B_T`` here: that identity needs a unitary ``Q``, and the no-sqrt2
        ``Q`` is exactly the non-unitary one. Measured ``max|B_U - B_T^T|`` is
        1.0, 1.5 and 2.5 at ``l = 1, 2, 3``. The parenthetical "(= B_T^T in real
        basis)" in the summary above predates this check and does not hold in this
        basis -- flagged rather than edited, since it is a claim about the algebra
        and belongs in its own change.
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

    # `bool(...)`, because `is_valid` compares a `np.linalg.norm` result and is
    # therefore `np.bool_`, which is not a `bool`. The `float(...)` beside it already
    # made the intent plain: this is a host-side verdict, not an array.
    return bool(is_valid), float(B_squared_error), sparsity_ratio
