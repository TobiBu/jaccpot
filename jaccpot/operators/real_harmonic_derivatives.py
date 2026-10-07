"""Exact Cartesian derivatives of the real regular solid harmonics.

The real harmonics ``U_n^m`` that ``p2m_real_direct`` and ``evaluate_local_real``
use (Dehnen's no-sqrt(2) real basis) are homogeneous polynomials of degree ``n``,
and a Cartesian derivative maps degree ``n`` onto degree ``n - 1``:

* ``d/dz U_n^m = U_{n-1}^m``
* ``m = 0``:  ``d/dx U_n^0 = -U_{n-1}^1``, ``d/dy U_n^0 = -U_{n-1}^{-1}``
* ``m >= 1``: ``d/dx U_n^m = (U_{n-1}^{m-1} - U_{n-1}^{m+1}) / 2``,
  ``d/dy U_n^m = -(U_{n-1}^{-(m-1)} + U_{n-1}^{-(m+1)}) / 2``
* ``m = -mu``: ``d/dx U_n^{-mu} = (U_{n-1}^{-(mu-1)} - U_{n-1}^{-(mu+1)}) / 2``,
  ``d/dy U_n^{-mu} = (U_{n-1}^{mu-1} + U_{n-1}^{mu+1}) / 2``

where ``U^{-0}`` and every ``U_{n-1}^{m'}`` with ``|m'| > n - 1`` are zero. These
follow from ``U_n^m = Re R_n^{|m|}`` / ``Im R_n^{|m|}`` and the complex lowering
identities; they were checked against a least-squares fit of ``d U / d delta`` from
the code's own harmonics (residual 2e-15 at order 5) and are pinned against the
Jacobian in ``tests/unit/operators/test_real_harmonic_derivatives.py``.

As a matrix, ``d/d axis U(delta) = A_axis U(delta)``; derivatives commute, so a
multi-index ``alpha`` gives ``A_alpha = A_x^a A_y^b A_z^c``, and for a local
expansion ``phi = F . U`` the derivative is ``d^alpha phi = (A_alpha^T F) . U``:
lower the coefficients, then evaluate. That is exact at every offset, including
``delta = 0`` and the z-axis where differentiating the polar form loses the
curvature (see :func:`jaccpot.operators.real_p2m_l2p.evaluate_local_real_derivative_tower`).
"""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._precision import highest_matmul_precision
from ._sh_indexing import sh_index, sh_size

__all__ = ["lower_real_coefficients", "real_harmonic_lowering_matrix"]


@lru_cache(maxsize=None)
def _lowering_matrix(order: int, axis: int) -> np.ndarray:
    nc = sh_size(order)
    a = np.zeros((nc, nc), dtype=np.float64)

    def put(row: int, n: int, m: int, value: float) -> None:
        # U_{n}^{m} with |m| > n does not exist; neither does "U^{-0}"
        if abs(m) <= n:
            a[row, sh_index(n, m)] += value

    for n in range(1, order + 1):
        k = n - 1
        for m in range(-n, n + 1):
            row = sh_index(n, m)
            if axis == 2:
                put(row, k, m, 1.0)
            elif m == 0:
                put(row, k, 1 if axis == 0 else -1, -1.0)
            elif m > 0:
                if axis == 0:
                    put(row, k, m - 1, 0.5)
                    put(row, k, m + 1, -0.5)
                else:
                    if m - 1 > 0:
                        put(row, k, -(m - 1), -0.5)
                    put(row, k, -(m + 1), -0.5)
            else:
                mu = -m
                if axis == 0:
                    if mu - 1 > 0:
                        put(row, k, -(mu - 1), 0.5)
                    put(row, k, -(mu + 1), -0.5)
                else:
                    put(row, k, mu - 1, 0.5)
                    put(row, k, mu + 1, 0.5)
    a.setflags(write=False)
    return a


@lru_cache(maxsize=None)
def _multi_index_matrix(order: int, alpha: tuple[int, int, int]) -> np.ndarray:
    out = np.eye(sh_size(order))
    for axis, power in enumerate(alpha):
        for _ in range(int(power)):
            out = np.matmul(out, _lowering_matrix(order, axis))
    out.setflags(write=False)
    return out


def real_harmonic_lowering_matrix(order: int, axis: int) -> np.ndarray:
    """The matrix ``A`` with ``d/d axis U(delta) = A U(delta)``.

    Parameters
    ----------
    order : int
        Expansion order ``p``; the matrix is ``(p+1)^2`` square in the packed
        ``(n, m)`` layout of :func:`~jaccpot.operators._sh_indexing.sh_index`.
    axis : int
        ``0``, ``1`` or ``2`` for x, y, z.

    Returns
    -------
    np.ndarray
        ``A[dst, src]``: row ``(n, m)`` reads only degree ``n - 1``, with entries
        in ``{0, +-1/2, +-1}``. Read-only (cached).

    Raises
    ------
    ValueError
        If ``axis`` is not 0, 1 or 2, or ``order`` is negative.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis!r}")
    if int(order) < 0:
        raise ValueError(f"order must be non-negative, got {order!r}")
    return _lowering_matrix(int(order), int(axis))


@highest_matmul_precision
def lower_real_coefficients(
    coeffs: Array, alpha: tuple[int, int, int], *, order: int
) -> Array:
    """Coefficients whose expansion is the ``alpha`` derivative of ``coeffs``'.

    For ``phi(delta) = sum_nm F_n^m U_n^m(delta)`` this returns ``G`` with
    ``sum_nm G_n^m U_n^m(delta) = d^alpha phi / d delta^alpha`` at every
    ``delta``: ``G = A_alpha^T F``.

    Parameters
    ----------
    coeffs : Array
        Packed real coefficients, shape ``(..., (p+1)^2)``.
    alpha : tuple[int, int, int]
        Derivative multi-index ``(a, b, c)`` for ``d^a/dx^a d^b/dy^b d^c/dz^c``.
    order : int
        Expansion order ``p``.

    Returns
    -------
    Array
        Lowered coefficients, same shape and dtype as ``coeffs``.
    """
    a, b, c = (int(v) for v in alpha)
    matrix = _multi_index_matrix(int(order), (a, b, c))
    coeffs = jnp.asarray(coeffs)
    return coeffs @ jnp.asarray(matrix, dtype=coeffs.dtype)
