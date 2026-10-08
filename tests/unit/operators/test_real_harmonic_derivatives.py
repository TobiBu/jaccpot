"""The real basis's exact lowering operator and the derivative tower built on it.

The real regular harmonics ``U_n^m`` used by ``p2m_real_direct`` /
``evaluate_local_real`` are polynomials, and a Cartesian derivative maps degree n
onto degree n - 1 with coefficients 0, +-1/2 and +-1. The tower used to take its
derivatives with nested ``jax.jacfwd`` through the polar form, whose floored
``r`` and ``rho`` lose the curvature at ``delta = 0`` and on the z-axis: measured,
the U_2^0 Hessian at 0 came out (0, 0, 1.5) on the diagonal instead of
(-1/2, -1/2, 1), and D2 / D3 jumped by up to 0.5 / 1.5 across those points for
unit coefficients. A target alone in its leaf sits exactly on the leaf's centre
of mass, so that is not a measure-zero case.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators._sh_indexing import sh_index, sh_size
from jaccpot.operators.real_harmonic_derivatives import (
    lower_real_coefficients,
    real_harmonic_lowering_matrix,
)
from jaccpot.operators.real_p2m_l2p import (
    _evaluate_local_real_derivative_tower_autodiff,
    evaluate_local_real,
    evaluate_local_real_derivative_tower,
)
from jaccpot.operators.symmetric_tensors import symmetric_multi_indices_3d

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="exactness checks need float64"
)


def _harmonics(delta, order):
    """The harmonic vector L2P contracts the coefficients with: d phi / d F."""
    zeros = jnp.zeros((sh_size(order),))
    return jax.jacfwd(lambda f: evaluate_local_real(f, delta, order=order))(zeros)


@pytest.mark.parametrize("order", [1, 2, 4, 7])
def test_lowering_matrices_are_the_cartesian_derivatives(order):
    rng = np.random.default_rng(order)
    for delta in rng.normal(scale=0.7, size=(6, 3)):
        d = jnp.asarray(delta)
        jac = np.asarray(jax.jacfwd(lambda x: _harmonics(x, order))(d))  # (nc, 3)
        u = np.asarray(_harmonics(d, order))
        for axis in range(3):
            lowered = real_harmonic_lowering_matrix(order, axis) @ u
            np.testing.assert_allclose(lowered, jac[:, axis], rtol=0, atol=1e-13)


def test_lowering_coefficients_are_halves_and_ones():
    for axis in range(3):
        a = real_harmonic_lowering_matrix(6, axis)
        assert set(np.unique(a)) <= {-1.0, -0.5, 0.0, 0.5, 1.0}
        # strictly degree-lowering: row (n, m) only reads degree n - 1
        for n in range(7):
            for m in range(-n, n + 1):
                cols = np.nonzero(a[sh_index(n, m)])[0]
                degrees = {int(np.floor(np.sqrt(c))) for c in cols}
                assert degrees <= {n - 1}


@pytest.mark.parametrize("order, k_max", [(4, 3), (6, 2)])
def test_exact_tower_matches_autodiff_at_generic_offsets(order, k_max):
    rng = np.random.default_rng(10 + order)
    coeffs = jnp.asarray(rng.normal(size=sh_size(order)))
    for delta in rng.normal(scale=0.6, size=(4, 3)):
        exact = evaluate_local_real_derivative_tower(
            coeffs, jnp.asarray(delta), order=order, max_derivative_order=k_max
        )
        auto = _evaluate_local_real_derivative_tower_autodiff(
            coeffs, jnp.asarray(delta), order=order, max_derivative_order=k_max
        )
        for got, want in zip(exact, auto):
            scale = max(float(np.max(np.abs(np.asarray(want)))), 1.0)
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize(
    "point", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, -0.3)], ids=str
)
def test_exact_tower_is_continuous_through_the_degenerate_points(point):
    """At delta = 0 and on the z-axis the tower equals its limit from nearby."""
    order, k_max = 4, 3
    rng = np.random.default_rng(3)
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    d0 = np.asarray(point)
    for column in range(sh_size(order)):
        coeffs = jnp.zeros((sh_size(order),)).at[column].set(1.0)
        at = evaluate_local_real_derivative_tower(
            coeffs, jnp.asarray(d0), order=order, max_derivative_order=k_max
        )
        near = evaluate_local_real_derivative_tower(
            coeffs, jnp.asarray(d0 + 1e-7 * u), order=order, max_derivative_order=k_max
        )
        for got, want in zip(at, near):
            # polynomials of degree <= 4: a 1e-7 step moves them by <~1e-6
            np.testing.assert_allclose(got, want, rtol=0, atol=2e-6)


def test_u20_hessian_at_the_expansion_centre():
    """U_2^0 = (3 z^2 - r^2) / 4: Hessian diag (-1/2, -1/2, 1), off-diagonal 0."""
    coeffs = jnp.zeros((sh_size(2),)).at[sh_index(2, 0)].set(1.0)
    hessian = evaluate_local_real_derivative_tower(
        coeffs, jnp.zeros(3), order=2, max_derivative_order=2
    )[2]
    expected = {(2, 0, 0): -0.5, (0, 2, 0): -0.5, (0, 0, 2): 1.0}
    for value, index in zip(np.asarray(hessian), symmetric_multi_indices_3d(2)):
        assert value == pytest.approx(expected.get(index, 0.0), abs=1e-14)

    # The defect this replaces, kept visible: autodiff through the floored polar
    # form drops the r^2 curvature at the centre.
    old = _evaluate_local_real_derivative_tower_autodiff(
        coeffs, jnp.zeros(3), order=2, max_derivative_order=2
    )[2]
    assert not np.allclose(np.asarray(old), np.asarray(hessian))


def test_lower_real_coefficients_composes_the_axes():
    """Lowering by a multi-index equals lowering axis by axis, in any order."""
    order = 5
    rng = np.random.default_rng(7)
    coeffs = jnp.asarray(rng.normal(size=sh_size(order)))
    once = lower_real_coefficients(coeffs, (1, 2, 1), order=order)
    stepwise = coeffs
    for alpha in ((0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 0)):
        stepwise = lower_real_coefficients(stepwise, alpha, order=order)
    np.testing.assert_allclose(once, stepwise, rtol=0, atol=1e-14)
