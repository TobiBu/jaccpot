"""Tests for pure-real spherical harmonics implementation.

This module tests the real_harmonics module which implements the Dehnen (2014)
solid harmonics U_n^m (multipole) and T_n^m (local) using pure real arithmetic.

The key functions tested are:
- p2m_real_direct: Particle-to-multipole (creates multipole coefficients)
- evaluate_local_real: Local-to-particle evaluation (evaluates local expansion)
- m2m_real: Multipole-to-multipole translation
- m2l_real: Multipole-to-local translation
- l2l_real: Local-to-local translation

All operations use only real arithmetic (no complex numbers).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.complex_harmonics import _pack_complex, complex_R_solidfmm
from jaccpot.operators.complex_ops import m2l_complex_reference
from jaccpot.operators.real_harmonics import (  # Index utilities; P2M; L2P; B matrices; Rotation; Z-axis translations;; Full operators
    _dehnen_real_Q_full,
    complex_to_dehnen_real_coeffs,
    compute_real_B_matrix_local,
    compute_real_B_matrix_multipole,
    evaluate_local_real,
    evaluate_local_real_with_grad,
    l2l_real,
    m2l_optimized_real,
    m2l_real,
    m2m_real,
    p2m_real_direct,
    real_Dz_diagonal,
    real_rotation_from_z_axis_local,
    real_rotation_from_z_axis_multipole,
    real_rotation_to_z_axis_local,
    real_rotation_to_z_axis_multipole,
    sh_index,
    sh_offset,
    sh_size,
    translate_along_z_l2l_real,
    translate_along_z_m2l_real,
    translate_along_z_m2m_real,
    verify_real_B_matrix,
)
from jaccpot.operators.solidfmm_reference import m2l_solidfmm_reference

# ===========================================================================
# Index utility tests
# ===========================================================================


def test_sh_size_formula():
    """sh_size(p) = (p+1)^2."""
    for p in range(10):
        assert sh_size(p) == (p + 1) ** 2


def test_sh_offset_cumulative():
    """sh_offset(ell) = ell^2."""
    for ell in range(10):
        assert sh_offset(ell) == ell * ell


def test_sh_index_within_bounds():
    """sh_index returns valid indices within [0, sh_size(ell))."""
    for ell in range(8):
        for m in range(-ell, ell + 1):
            idx = sh_index(ell, m)
            assert 0 <= idx < sh_size(ell), f"ell={ell}, m={m}, idx={idx}"


def test_sh_index_unique():
    """Each (ell, m) pair maps to a unique index."""
    p = 6
    indices = []
    for ell in range(p + 1):
        for m in range(-ell, ell + 1):
            indices.append(sh_index(ell, m))
    assert len(indices) == len(set(indices)), "Indices should be unique"
    assert len(indices) == sh_size(p)


# ===========================================================================
# P2M tests (particle to multipole)
# ===========================================================================


def test_p2m_real_direct_dehnen_table3():
    """Verify P2M matches Dehnen Table 3 for ell=1.

    Dehnen Table 3 gives:
        U_1^{-1} = y/2
        U_1^{0}  = z
        U_1^{+1} = x/2
    """
    order = 1
    mass = 1.0

    # At (1, 0, 0): expect U_1^{+1} = 0.5
    M = p2m_real_direct(jnp.array([1.0, 0.0, 0.0]), jnp.array(mass), order=order)
    assert jnp.isclose(M[sh_index(1, 1)], 0.5, atol=1e-10)
    assert jnp.isclose(M[sh_index(1, 0)], 0.0, atol=1e-10)
    assert jnp.isclose(M[sh_index(1, -1)], 0.0, atol=1e-10)

    # At (0, 1, 0): expect U_1^{-1} = 0.5
    M = p2m_real_direct(jnp.array([0.0, 1.0, 0.0]), jnp.array(mass), order=order)
    assert jnp.isclose(M[sh_index(1, -1)], 0.5, atol=1e-10)
    assert jnp.isclose(M[sh_index(1, 0)], 0.0, atol=1e-10)
    assert jnp.isclose(M[sh_index(1, 1)], 0.0, atol=1e-10)

    # At (0, 0, 1): expect U_1^{0} = 1.0
    M = p2m_real_direct(jnp.array([0.0, 0.0, 1.0]), jnp.array(mass), order=order)
    assert jnp.isclose(M[sh_index(1, 0)], 1.0, atol=1e-10)
    assert jnp.isclose(M[sh_index(1, -1)], 0.0, atol=1e-10)
    assert jnp.isclose(M[sh_index(1, 1)], 0.0, atol=1e-10)


# ===========================================================================
# Absolute anchor for the Dehnen normalisation: the derivative recurrences
# ===========================================================================
#
# `test_p2m_real_direct_dehnen_table3` pins degree 1 against Dehnen (2014) Table 3,
# and `test_complex_to_dehnen_real_matches_p2m_real_direct` cross-checks degrees 0-6
# against the complex basis. The second is a *relative* check: a convention error
# shared between `Q` and `p2m_real_direct` cancels in it. Nothing pinned degrees 2-6
# absolutely, which the docstring on `p2m_real_direct` used to call the most serious
# gap in the file.
#
# These two tests close it without a table. The regular solid harmonics normalised
# by 1/(n+|m|)! satisfy derivative recurrences that map degree n onto degree n-1, and
# those recurrences plus U_0^0 = 1 determine every U_n^m UNIQUELY -- so asserting them
# is an absolute anchor, reached from theory rather than from a transcribed table this
# repo cannot check.
DERIVATIVE_RECURRENCE_ORDER = 6


def _u_and_jacobian(delta, order):
    """``U_n^m(delta)`` for all ``(n, m)`` and its Jacobian w.r.t. ``delta``."""

    def evaluate(d):
        return p2m_real_direct(d, jnp.asarray(1.0, dtype=jnp.float64), order=order)

    point = jnp.asarray(delta, dtype=jnp.float64)
    return np.asarray(evaluate(point)), np.asarray(jax.jacobian(evaluate)(point))


def test_solid_harmonic_z_derivative_lowers_the_degree():
    """``dU_n^m/dz == U_{n-1}^m`` exactly, for every ``(n, m)`` up to order 6.

    The cleanest of the two recurrences and the one with no case analysis. It pins the
    normalisation *across* degrees at fixed ``m``: given ``U_{|m|}^m``, it determines
    every higher ``U_n^m``. Combined with ``U_0^0 = 1`` (asserted here too, since it is
    the scale the whole chain hangs from) and the transverse recurrence in the next
    test, the entire basis is fixed.

    Exact equality is the right assertion, not a tolerance: both sides come from the
    same evaluation and the identity is algebraic, so only float64 round-off is
    admissible. Measured worst deviation is 0 to ~1e-16, and the bound is 1e-13.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("exact-identity tolerance requires float64 (JAX_ENABLE_X64=1)")

    order = DERIVATIVE_RECURRENCE_ORDER
    rng = np.random.default_rng(20260806)
    for _ in range(4):
        delta = rng.normal(size=3)
        values, jacobian = _u_and_jacobian(delta, order)

        # The scale the recurrences propagate from.
        assert abs(values[sh_index(0, 0)] - 1.0) < 1e-13

        for n in range(1, order + 1):
            for m in range(-n, n + 1):
                if abs(m) > n - 1:
                    # U_{n-1}^m does not exist; the derivative must vanish instead.
                    assert abs(jacobian[sh_index(n, m), 2]) < 1e-13, (
                        f"dU_{n}^{m}/dz should vanish (no U_{n - 1}^{m}), got "
                        f"{jacobian[sh_index(n, m), 2]:.3e}"
                    )
                    continue
                got = jacobian[sh_index(n, m), 2]
                want = values[sh_index(n - 1, m)]
                assert abs(got - want) < 1e-13 * max(
                    abs(want), 1.0
                ), f"dU_{n}^{m}/dz = {got:.9e} but U_{n - 1}^{m} = {want:.9e}"


def test_solid_harmonic_transverse_derivative_coefficients_are_half_integers():
    """``dU_n^m/dx`` and ``/dy`` are half-integer combinations of degree ``n-1``.

    This is the half that pins normalisation *between* different ``m``, which the
    z-recurrence cannot and which the complex->real cross-check can cancel. Two
    properties are asserted, and it is the second that does the work:

    1. **Structural.** Coefficients solved from one set of points reproduce the
       derivative at a disjoint set to round-off (measured worst absolute residual
       8.3e-15). So this is an identity, not a per-point fit.
    2. **Half-integrality.** Every coefficient is an exact multiple of 1/2 -- measured,
       they are all 0, +/-1/2 or +/-1. This is what catches a per-``m`` normalisation
       error: scaling one ``U_n^m`` keeps the relation linear and point-independent, so
       property 1 still holds, but the coefficients pick up the erroneous ratio and stop
       being half-integers.

    Mutation check, which is the reason to trust it. Scaling a single ``U_n^m`` by
    ``1 + 1e-3`` -- exactly the error class this is meant to catch -- leaves the
    held-out residual unchanged at ~8e-15 and produces non-half-integer coefficients in
    6 entries for ``U_3^2``, 6 for ``U_4^{-3}`` and 2 for ``U_6^5``. Property 1 alone
    would not have noticed any of them.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("exact-identity tolerance requires float64 (JAX_ENABLE_X64=1)")

    order = DERIVATIVE_RECURRENCE_ORDER
    fit_rng = np.random.default_rng(11)
    check_rng = np.random.default_rng(9901)
    fit = [_u_and_jacobian(fit_rng.normal(size=3), order) for _ in range(14)]
    check = [_u_and_jacobian(check_rng.normal(size=3), order) for _ in range(6)]

    for n in range(1, order + 1):
        lower = list(range(-(n - 1), n))
        fit_basis = np.array([[v[sh_index(n - 1, m2)] for m2 in lower] for v, _ in fit])
        check_basis = np.array(
            [[v[sh_index(n - 1, m2)] for m2 in lower] for v, _ in check]
        )
        for m in range(-n, n + 1):
            for axis, axis_name in ((0, "x"), (1, "y")):
                target = np.array([j[sh_index(n, m), axis] for _, j in fit])
                solved, *_ = np.linalg.lstsq(fit_basis, target, rcond=None)

                held_out = np.array([j[sh_index(n, m), axis] for _, j in check])
                residual = float(np.max(np.abs(check_basis @ solved - held_out)))
                assert residual < 1e-12, (
                    f"dU_{n}^{m}/d{axis_name} is not a fixed combination of degree "
                    f"{n - 1}: held-out residual {residual:.3e}"
                )

                for m2, coefficient in zip(lower, solved):
                    doubled = 2.0 * float(coefficient)
                    assert abs(doubled - round(doubled)) < 1e-9, (
                        f"dU_{n}^{m}/d{axis_name} has coefficient {coefficient:.9f} on "
                        f"U_{n - 1}^{m2}, which is not a multiple of 1/2 -- the "
                        "signature of a per-m normalisation error"
                    )


def test_p2m_real_direct_monopole():
    """Monopole (ell=0) should equal mass."""
    for order in [0, 1, 2, 4]:
        mass = 2.5
        delta = jnp.array([1.0, 2.0, 3.0])
        M = p2m_real_direct(delta, jnp.array(mass), order=order)
        assert jnp.isclose(M[0], mass, atol=1e-10)


def test_p2m_real_direct_at_origin():
    """P2M at origin should be zero except monopole."""
    order = 4
    mass = 1.0
    delta = jnp.array([0.0, 0.0, 0.0])
    M = p2m_real_direct(delta, jnp.array(mass), order=order)
    assert jnp.isclose(M[0], mass, atol=1e-10)
    # All higher moments should be zero (or very small due to r^n factor)
    for ell in range(1, order + 1):
        for m in range(-ell, ell + 1):
            assert jnp.abs(M[sh_index(ell, m)]) < 1e-10


def test_p2m_real_direct_linearity():
    """P2M should be linear in mass."""
    order = 3
    delta = jnp.array([1.0, 2.0, 0.5])
    M1 = p2m_real_direct(delta, jnp.array(1.0), order=order)
    M2 = p2m_real_direct(delta, jnp.array(2.0), order=order)
    assert jnp.allclose(M2, 2.0 * M1, atol=1e-10)


# ===========================================================================
# L2P tests (local to particle evaluation)
# ===========================================================================


def test_evaluate_local_real_monopole():
    """Monopole local gives constant potential."""
    order = 4
    # Local expansion with only monopole term
    local_coeffs = jnp.zeros(sh_size(order))
    local_coeffs = local_coeffs.at[0].set(1.0)

    # Evaluate at various points - should all give the same value
    delta1 = jnp.array([1.0, 0.0, 0.0])
    delta2 = jnp.array([0.5, 0.5, 0.5])
    delta3 = jnp.array([0.0, 0.0, 1.0])

    # For monopole, T_0^0 = 1, so potential = F_0^0 * 1 = 1.0
    # (but local basis has r^0 = 1 dependence for ell=0)
    val1 = evaluate_local_real(local_coeffs, delta1, order=order)
    val2 = evaluate_local_real(local_coeffs, delta2, order=order)
    val3 = evaluate_local_real(local_coeffs, delta3, order=order)

    assert jnp.isclose(val1, val2, atol=1e-10)
    assert jnp.isclose(val2, val3, atol=1e-10)


def test_evaluate_local_real_with_grad_consistency():
    """Gradient matches numerical gradient."""
    order = 3
    rng = np.random.default_rng(0)
    local_coeffs = jnp.array(rng.standard_normal(sh_size(order)))
    delta = jnp.array([0.5, 0.3, 0.2])

    grad, val = evaluate_local_real_with_grad(local_coeffs, delta, order=order)

    # Numerical gradient
    eps = 1e-5
    grad_num = jnp.zeros(3)
    for i in range(3):
        d_plus = delta.at[i].set(delta[i] + eps)
        d_minus = delta.at[i].set(delta[i] - eps)
        val_plus = evaluate_local_real(local_coeffs, d_plus, order=order)
        val_minus = evaluate_local_real(local_coeffs, d_minus, order=order)
        grad_num = grad_num.at[i].set((val_plus - val_minus) / (2 * eps))

    assert jnp.allclose(grad, grad_num, atol=1e-4)


# The displacements where the cylindrical radius rho = sqrt(x^2+y^2) collapses onto
# the squared-radius floor: the expansion centre itself (a particle at its leaf's
# centre of mass -- guaranteed for a one-particle leaf) and the centre's z axis.
DEGENERATE_L2P_DELTAS = [
    ("origin", [0.0, 0.0, 0.0]),
    ("plus_z", [0.0, 0.0, 0.37]),
    ("minus_z", [0.0, 0.0, -0.37]),
]


@pytest.mark.parametrize("label,delta", DEGENERATE_L2P_DELTAS, ids=lambda v: str(v))
@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_evaluate_local_real_grad_at_rho_zero_matches_limit(label, delta, order):
    """L2P gradient at rho == 0 equals the limit from a tiny transverse nudge.

    Regression: the azimuth used to be selected as a *constant* (cos phi, sin phi)
    = (1, 0) wherever rho hit the floor. That is harmless in the forward pass (the
    sin^|m| theta factor annihilates the arbitrary azimuth) but under ``jax.grad``
    a constant has no x/y derivative, so the transverse gradient of every m != 0
    term was dropped -- the far-field force came back with x and y exactly zero.
    """
    rng = np.random.default_rng(0)
    local_coeffs = jnp.array(rng.standard_normal(sh_size(order)))
    delta = jnp.asarray(delta)

    grad, pot = evaluate_local_real_with_grad(local_coeffs, delta, order=order)

    # The limit is direction independent, so any tiny transverse displacement off
    # the degeneracy gives the answer the degenerate point must reproduce.
    for nudge in ([1e-14, 0.0, 0.0], [0.0, 1e-14, 0.0], [3e-15, -4e-15, 0.0]):
        grad_off, pot_off = evaluate_local_real_with_grad(
            local_coeffs, delta + jnp.asarray(nudge), order=order
        )
        assert jnp.allclose(
            grad, grad_off, rtol=1e-9, atol=1e-12
        ), f"{label}: gradient at rho == 0 disagrees with the limit from {nudge}"
        assert jnp.allclose(pot, pot_off, rtol=1e-9, atol=1e-12)

    # Guard against a vacuous pass: the transverse components are genuinely
    # nonzero here, which is exactly what the old code zeroed out. The bound only
    # has to separate them from *exactly* 0.0 -- their size varies with the random
    # coefficients and the order.
    assert jnp.abs(grad[0]) > 1e-6
    assert jnp.abs(grad[1]) > 1e-6


def test_evaluate_local_real_grad_at_origin_is_the_degree_one_term():
    """At delta == 0 only the degree-1 coefficients survive in the gradient.

    U_1^1 = x/2!, U_1^-1 = y/2!, U_1^0 = z, so the gradient is closed form.
    """
    order = 4
    rng = np.random.default_rng(0)
    local_coeffs = jnp.array(rng.standard_normal(sh_size(order)))

    grad, _ = evaluate_local_real_with_grad(local_coeffs, jnp.zeros(3), order=order)
    expected = jnp.array(
        [
            local_coeffs[sh_index(1, 1)] / 2.0,
            local_coeffs[sh_index(1, -1)] / 2.0,
            local_coeffs[sh_index(1, 0)],
        ]
    )
    assert jnp.allclose(grad, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("label,delta", DEGENERATE_L2P_DELTAS, ids=lambda v: str(v))
@pytest.mark.parametrize("order", [1, 2, 4])
def test_p2m_real_direct_jacobian_at_rho_zero_matches_limit(label, delta, order):
    """P2M has the same rho == 0 degeneracy as L2P, and the same fix.

    The multipole Jacobian w.r.t. the particle position is what the
    differentiable FMM backpropagates through, and a particle sitting exactly at
    its leaf's expansion centre is the degenerate case.
    """
    delta = jnp.asarray(delta)
    mass = jnp.asarray(1.7)

    def jac(d):
        return jax.jacrev(lambda dd: p2m_real_direct(dd, mass, order=order))(d)

    J = jac(delta)
    J_off = jac(delta + jnp.asarray([3e-15, -4e-15, 0.0]))
    assert jnp.allclose(
        J, J_off, rtol=1e-9, atol=1e-12
    ), f"{label}: P2M Jacobian at rho == 0 disagrees with the nearby limit"

    # d M_1^{+-1} / d(x, y) = mass / 2! -- the entries the old code zeroed.
    assert jnp.isclose(J[sh_index(1, 1), 0], mass / 2.0, rtol=1e-12)
    assert jnp.isclose(J[sh_index(1, -1), 1], mass / 2.0, rtol=1e-12)


# ===========================================================================
# B matrix tests
# ===========================================================================


def test_B_matrix_involution():
    """B @ B = I (B is an involution)."""
    dtype = jnp.float64
    for ell in range(6):
        B_T = compute_real_B_matrix_local(ell, dtype=dtype)
        B_U = compute_real_B_matrix_multipole(ell, dtype=dtype)

        eye = jnp.eye(2 * ell + 1)
        assert jnp.allclose(B_T @ B_T, eye, atol=1e-10), f"B_T² != I for ell={ell}"
        assert jnp.allclose(B_U @ B_U, eye, atol=1e-10), f"B_U² != I for ell={ell}"


def test_B_matrix_verify_passes():
    """verify_real_B_matrix should pass for correct matrices."""
    dtype = jnp.float64
    for ell in range(5):
        # This should not raise an exception
        verify_real_B_matrix(ell, dtype=dtype)


# ===========================================================================
# Rotation tests
# ===========================================================================


def test_Dz_diagonal_identity_at_zero():
    """Dz(0) = I."""
    dtype = jnp.float64
    for ell in range(5):
        # `jnp.asarray`, not a bare `0.0`: these builders declare `angle: Array`
        # and passing a float made the test violate its own contract (F40).
        Dz = real_Dz_diagonal(ell, jnp.asarray(0.0, dtype=dtype), dtype=dtype)
        eye = jnp.eye(2 * ell + 1)
        assert jnp.allclose(Dz, eye, atol=1e-10)


def test_rotation_z_axis_is_identity():
    """Rotation to z-axis when already aligned should be near identity.

    When the direction is already along z-axis, the rotation should be trivial
    (just a diagonal phase matrix at most).
    """
    dtype = jnp.float64

    # Direction along z-axis. As arrays, because the builders declare
    # `x, y, z: Array` -- floats worked but violated the contract (F40).
    x, y, z = (jnp.asarray(v, dtype=dtype) for v in (0.0, 0.0, 1.0))

    for ell in range(4):
        D = real_rotation_to_z_axis_multipole(x, y, z, ell, dtype=dtype)
        # For z-aligned direction, alpha_z = 0 and alpha_x = 0, so D ≈ I
        eye = jnp.eye(2 * ell + 1)
        assert jnp.allclose(
            D, eye, atol=1e-10
        ), f"D != I for ell={ell} when aligned with z"


def test_rotation_preserves_monopole():
    """Rotation should not affect the monopole (ell=0) coefficient."""
    dtype = jnp.float64

    # Arbitrary direction
    # Arrays, not floats: the builders declare `x, y, z: Array` (F40).
    x, y, z = (jnp.asarray(v, dtype=dtype) for v in (1.0, 2.0, 3.0))

    D = real_rotation_to_z_axis_multipole(x, y, z, 0, dtype=dtype)
    # For ell=0, D should be 1x1 identity
    assert D.shape == (1, 1)
    assert jnp.isclose(D[0, 0], 1.0, atol=1e-10)


# ===========================================================================
# Z-axis translation tests
# ===========================================================================


@pytest.mark.parametrize(
    "translate_fn, seed",
    [
        (translate_along_z_m2m_real, 0),
        (translate_along_z_l2l_real, 1),
    ],
)
def test_z_translate_identity_at_zero(translate_fn, seed):
    """M2M/L2L with dz=0 should be identity."""
    order = 4
    key = jax.random.PRNGKey(seed)
    coeffs = jax.random.normal(key, (sh_size(order),))

    result = translate_fn(coeffs, jnp.array(0.0), order=order)
    assert jnp.allclose(result, coeffs, atol=1e-10)


@pytest.mark.parametrize(
    "m2l_call",
    [
        # Z-axis M2L: translation distance r=2.0 along z.
        lambda mp, order: translate_along_z_m2l_real(mp, jnp.array(2.0), order=order),
        # Full (rotated) M2L: delta of length 2.0 along x.
        lambda mp, order: m2l_real(mp, jnp.array([2.0, 0.0, 0.0]), order=order),
    ],
)
def test_m2l_monopole_gives_inverse_distance(m2l_call):
    """M2L of a unit monopole at distance r gives a 1/r local monopole term."""
    order = 4
    r = 2.0
    multipole = jnp.zeros(sh_size(order))
    multipole = multipole.at[0].set(1.0)  # Unit monopole

    local = m2l_call(multipole, order)

    # The ell=0 local coefficient should be 1/r
    assert jnp.isclose(local[0], 1.0 / r, atol=1e-10)


def test_z_m2l_error_improves_with_order():
    """Z-axis M2L error should decrease geometrically with expansion order.

    For a well-separated configuration (eval_offset << R), the truncation error
    should scale as (eval_offset / R)^{p+1} where p is the expansion order.

    This test verifies that our z-axis M2L formula is correct by checking that
    higher orders give better accuracy.
    """
    # Source monopole at origin, target center on z-axis
    R = 10.0  # Distance to target center
    eval_offset = 0.5  # Small offset from target center

    # Evaluation point
    eval_point = jnp.array(
        [eval_offset * 0.6, eval_offset * 0.8, R - eval_offset * 0.2]
    )

    # Direct potential: 1/|eval_point - origin|
    direct = 1.0 / jnp.linalg.norm(eval_point)

    # Representative orders (smallest / middle / largest) are enough to
    # establish the "error improves with order" trend and the final-accuracy
    # margin, while keeping the JAX compile cost low.
    orders = (1, 5, 9)
    errors = []
    for order in orders:
        # Unit monopole
        multipole = jnp.zeros(sh_size(order))
        multipole = multipole.at[0].set(1.0)

        # Z-axis M2L
        local = translate_along_z_m2l_real(multipole, jnp.array(R), order=order)

        # L2P: delta = center - eval_point
        target_center = jnp.array([0.0, 0.0, R])
        delta_l2p = target_center - eval_point
        fmm_potential = evaluate_local_real(local, delta_l2p, order=order)

        rel_error = abs(fmm_potential - direct) / abs(direct)
        errors.append(rel_error)

    # Verify error decreases with order (at least for the first several orders)
    for i in range(len(errors) - 1):
        # Each order should reduce error (allowing some numerical noise)
        # At very high orders, error may plateau due to machine precision
        if errors[i] > 1e-12:
            # Only check while error is above machine precision
            assert errors[i + 1] <= errors[i] * 1.1, (
                "Error did not decrease: order "
                f"{orders[i]} error {errors[i]:.2e} -> "
                f"order {orders[i+1]} error {errors[i+1]:.2e}"
            )

    # The highest order should achieve very good accuracy
    assert errors[-1] < 1e-9, f"Final error {errors[-1]:.2e} not small enough"


def test_m2l_convergence_radius_respected_in_rotated_geometry():
    """Off-axis geometry should converge when eval_offset << R (coordinate rotation)."""
    dtype = jnp.float64

    # Build an off-axis geometry and rotate coordinates to z-axis.
    source_pos = jnp.array([0.0, 0.0, 0.0], dtype=dtype)
    local_center = jnp.array([3.0, 2.0, 4.0], dtype=dtype)
    eval_offset = jnp.array([0.2, -0.1, 0.15], dtype=dtype)
    eval_point = local_center + eval_offset

    R = jnp.linalg.norm(local_center - source_pos)
    r_eval = jnp.linalg.norm(eval_point - source_pos)
    assert float(jnp.linalg.norm(eval_offset)) < float(R)

    # Rotate coordinates so the local center lies on +z.
    x, y, z = local_center - source_pos
    rho = jnp.sqrt(x * x + y * y)
    alpha_z = jnp.arctan2(y, x)
    alpha_x = jnp.arctan2(rho, z)

    def rot_z(v, ang):
        c = jnp.cos(ang)
        s = jnp.sin(ang)
        return jnp.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], dtype=dtype)

    def rot_y(v, ang):
        c = jnp.cos(ang)
        s = jnp.sin(ang)
        return jnp.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]], dtype=dtype)

    # Active rotation: Rz(-alpha_z) then Ry(-alpha_x).
    local_center_rot = rot_y(rot_z(local_center - source_pos, -alpha_z), -alpha_x)
    eval_point_rot = rot_y(rot_z(eval_point - source_pos, -alpha_z), -alpha_x)

    # Ensure rotation preserves distances.
    assert jnp.isclose(jnp.linalg.norm(local_center_rot), R, rtol=1e-12)
    assert jnp.isclose(jnp.linalg.norm(eval_point_rot), r_eval, rtol=1e-12)

    direct = 1.0 / r_eval
    errors = []
    # Representative orders (smallest / middle / largest) suffice to check the
    # monotone-improvement trend while keeping the JAX compile cost low.
    for order in (2, 4, 7):
        multipole = jnp.zeros(sh_size(order), dtype=dtype).at[0].set(1.0)
        local = translate_along_z_m2l_real(multipole, jnp.asarray(R), order=order)
        delta_l2p = local_center_rot - eval_point_rot
        fmm_potential = evaluate_local_real(local, delta_l2p, order=order)
        rel_error = float(jnp.abs(fmm_potential - direct) / jnp.abs(direct))
        errors.append(rel_error)

    # Should decrease as order increases for a convergent setup.
    for i in range(len(errors) - 1):
        if errors[i] > 1e-12:
            assert errors[i + 1] <= errors[i] * 1.1


def test_solidfmm_reference_matches_z_axis_m2l():
    """Genuine-complex solidfmm reference relates to the real-basis z-M2L.

    The complex reference (``m2l_solidfmm_reference``) runs a standard complex
    M2L, so it reproduces the real-basis result on the m = 0 channel and is a
    factor of 2 smaller on every m != 0 channel -- exactly the no-sqrt2
    real-basis factor carried by ``translate_along_z_m2l_real``.
    """
    dtype = jnp.float64
    order = 6

    source_pos = jnp.array([0.2, -0.1, 0.3], dtype=dtype)
    multipole_center = jnp.array([0.0, 0.0, 0.0], dtype=dtype)
    local_center = jnp.array([0.0, 0.0, 4.0], dtype=dtype)

    multipole = p2m_real_direct(
        source_pos - multipole_center, jnp.array(1.0, dtype=dtype), order=order
    )
    delta_m2l = local_center - multipole_center

    local_ref = translate_along_z_m2l_real(
        multipole, jnp.linalg.norm(delta_m2l), order=order
    )
    local_solidfmm = np.asarray(
        m2l_solidfmm_reference(multipole, delta_m2l, order=order)
    )

    # Rescale the genuine-complex reference by the real-basis channel factor.
    channel_scale = np.ones(sh_size(order))
    for n in range(order + 1):
        for m in range(-n, n + 1):
            if m != 0:
                channel_scale[sh_index(n, m)] = 2.0
    local_solidfmm_real = local_solidfmm * channel_scale

    assert jnp.allclose(local_solidfmm_real, local_ref, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# The genuine-complex <-> Dehnen-no-sqrt2 relationship for LOCAL coefficients,
# and the off-axis defect in solidfmm_reference that it exposes.
# See docs/operator_conventions.md section 4.
# ---------------------------------------------------------------------------

# Directions used by both tests below. Chosen to include the cases where a
# convention error can hide: one with two near-equal components (a wrong
# azimuth argument order is invisible when x == y), one with z == 0, one with
# every component negative, and one at rho ~ 0 where the m != 0 channels carry
# (rho/r)^|m| and suppress any azimuth error whatsoever.
_M2L_PROBE_DELTAS = (
    (4.0, 1.5, -2.5),
    (3.0, 3.0000001, 1.0),
    (5.0, 0.0, 0.0),
    (-2.0, -3.0, -1.5),
    (-1.0, 2.5, 3.5),
    (1e-6, 0.0, 5.0),
)


def _dehnen_local_channel_factor(order: int) -> np.ndarray:
    """Diagonal complex -> Dehnen-no-sqrt2 factor for LOCAL coefficients.

    Multipole coefficients are harmonic *values* (``M_n^m = mass * U_n^m``), so
    they convert with ``Q`` alone -- that is what
    :func:`~jaccpot.operators.real_harmonics.complex_to_dehnen_real_coeffs`
    does. Local coefficients are the *dual* objects: ``evaluate_local_real``
    forms the plain sum ``Psi = sum_{n,m} F_n^m U_n^m``, and collapsing the
    complex sum over ``m`` in ``[-n, n]`` onto the ``m >= 0`` real channels
    folds each conjugate pair together, which contributes one extra factor of
    two on every ``m != 0`` channel. So locals convert with ``D @ Q``, this
    ``D``, and multipoles with ``Q``. The asymmetry is the whole reason the
    factor of two exists.

    Parameters
    ----------
    order : int
        Maximum SH degree ``p``.

    Returns
    -------
    np.ndarray
        Packed ``(sh_size(order),)`` factor: 1.0 on every ``m == 0`` channel and
        2.0 everywhere else.
    """
    factor = np.ones(sh_size(order))
    for n in range(order + 1):
        for m in range(-n, n + 1):
            if m != 0:
                factor[sh_index(n, m)] = 2.0
    return factor


@pytest.mark.parametrize("order", [2, 4, 6])
def test_dehnen_local_channel_factor_holds_at_any_delta(order: int) -> None:
    """``D @ Q`` converts complex locals to Dehnen-real ones at ANY ``delta``.

    The factor-of-two-on-``m != 0`` relationship is a statement about the two
    *bases*, not about the geometry, so it must hold for an arbitrary off-axis
    displacement and not only for an axis-aligned one. This pins that against
    the production complex M2L (``complex_ops.m2l_complex_reference``), which is
    independently validated end-to-end, so the comparison here needs no
    reference of its own.

    Measured worst case over these six directions and orders 2/4/6: 3.1e-16
    relative to ``max|m2l_real|``, i.e. round-off. The tolerance below is 1e-12,
    which leaves ~3 orders of magnitude of headroom for the order-6 matmul
    chains while staying far under the 1.6e-3 floor that the inertness gate
    measures -- so the test cannot pass by being loose.

    Parameters
    ----------
    order : int
        Expansion order under test.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("the 1e-12 basis-identity tolerance requires float64")

    dtype = jnp.float64
    source = jnp.array([0.2, -0.1, 0.3], dtype=dtype)
    factor = _dehnen_local_channel_factor(order)
    q_full = np.asarray(_dehnen_real_Q_full(order))

    multipole_real = p2m_real_direct(source, jnp.array(1.0, dtype=dtype), order=order)
    multipole_complex = complex_R_solidfmm(source, order=order)

    for delta_tuple in _M2L_PROBE_DELTAS:
        delta = jnp.asarray(delta_tuple, dtype=dtype)

        want = np.asarray(m2l_real(multipole_real, delta, order=order))
        local_complex = np.asarray(
            m2l_complex_reference(multipole_complex, delta, order=order)
        )
        product = q_full @ local_complex

        # Conjugate symmetry is a documented PRECONDITION of the Q conversion,
        # not something it enforces; if the complex locals ever stopped
        # conforming, taking the real part would silently discard signal.
        assert np.max(np.abs(product.imag)) < 1e-12 * max(
            np.max(np.abs(product.real)), 1.0
        )

        got = product.real * factor
        scale = max(np.max(np.abs(want)), 1.0)
        assert np.max(np.abs(got - want)) < 1e-12 * scale, (
            f"D @ Q must reproduce m2l_real at delta={delta_tuple}, order={order}; "
            f"got max deviation {np.max(np.abs(got - want)) / scale:.3e}"
        )

        # Inertness gate. Without the factor of two the two bases disagree by at
        # least 1.6e-3 (measured minimum, at the rho ~ 0 direction and order 2;
        # the genuinely off-axis directions reach 4.3e-1). A test that only
        # pinned the right answer would pass just as happily if the factor
        # stopped mattering, which is exactly when it has stopped being pinned.
        undone = np.max(np.abs(product.real - want)) / scale
        assert undone > 1e-4, (
            "dropping the m != 0 factor of two must break the identity, else "
            f"this test pins nothing; got {undone:.3e} at delta={delta_tuple}"
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "m2l_solidfmm_reference rotates with the wrong two conventions off "
        "axis: docs/operator_conventions.md section 4. If this XPASSes the "
        "reference has been repaired -- drop the marker, do not relax the test."
    ),
)
def test_solidfmm_reference_matches_m2l_real_off_axis():
    """``m2l_solidfmm_reference`` must agree with ``m2l_real`` off axis.

    Same relationship as
    :func:`test_dehnen_local_channel_factor_holds_at_any_delta`, which shows it
    holding to round-off at every one of these directions for the production
    complex M2L. The reference fails it: measured deviation 3.9e-2 (order 2) to
    1.7 (order 6) relative to ``max|m2l_real|``, and the resulting *potential*
    -- which is basis-independent, so no convention argument can excuse it --
    plateaus at 2.1e-2 to 8.5e-2 relative error, flat in ``order`` where
    ``m2l_real`` converges 4.5e-4 -> 6.1e-7 -> 2.7e-8. An error that does not
    fall with ``order`` is not truncation.

    ``(1e-6, 0, 5)`` is deliberately excluded: at rho ~ 0 the defect is
    suppressed to 1.9e-8 by the ``(rho/r)^|m|`` factors, so a near-axis
    direction would make this test vacuous.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("the 1e-12 basis-identity tolerance requires float64")

    dtype = jnp.float64
    order = 4
    source = jnp.array([0.2, -0.1, 0.3], dtype=dtype)
    eval_offset = jnp.array([0.15, 0.2, -0.1], dtype=dtype)
    factor = _dehnen_local_channel_factor(order)

    multipole = p2m_real_direct(source, jnp.array(1.0, dtype=dtype), order=order)

    for delta_tuple in _M2L_PROBE_DELTAS[:-1]:
        delta = jnp.asarray(delta_tuple, dtype=dtype)

        want = np.asarray(m2l_real(multipole, delta, order=order))
        got = np.asarray(m2l_solidfmm_reference(multipole, delta, order=order)) * factor
        scale = max(np.max(np.abs(want)), 1.0)
        assert np.max(np.abs(got - want)) < 1e-12 * scale, (
            f"reference M2L must match m2l_real at delta={delta_tuple}; got max "
            f"deviation {np.max(np.abs(got - want)) / scale:.3e}"
        )

        # And the physics anchor: the evaluated potential against the direct sum.
        # `evaluate_local_real` takes centre - eval_point (section 1 of
        # docs/operator_conventions.md), hence the minus sign.
        exact = 1.0 / float(jnp.linalg.norm(delta + eval_offset - source))
        potential = float(
            evaluate_local_real(jnp.asarray(got), -eval_offset, order=order)
        )
        # 1e-4 sits between the two regimes with room on both sides: at order 4
        # `m2l_real` lands at 6.1e-7 to 6.5e-6 over these five directions (15x
        # under the bound), while the reference's plateau is 2.1e-2 to 8.5e-2
        # (210x over it). So neither a slightly loose tolerance nor a slightly
        # unlucky geometry decides the outcome.
        assert abs(potential - exact) / exact < 1e-4, (
            f"reference potential must converge at delta={delta_tuple}; got "
            f"relative error {abs(potential - exact) / exact:.3e}"
        )


def test_z_m2l_higher_multipoles():
    """Z-axis M2L should work correctly for dipole and higher multipoles.

    Test with a source offset ALONG Z-AXIS to generate non-trivial multipoles
    while keeping the M2L translation along z.
    """
    order = 8

    # Source offset along z-axis (generates higher multipoles on z)
    source_offset = jnp.array([0.0, 0.0, 0.3])
    mass = 1.0

    # P2M at the offset position
    multipole = p2m_real_direct(source_offset, jnp.array(mass), order=order)

    # Target center far along z-axis
    R = 10.0
    target_center = jnp.array([0.0, 0.0, R])

    # Z-axis M2L (translation distance is R, not R - source_offset[2])
    # The M2L translates from multipole center (origin) to local center
    local = translate_along_z_m2l_real(multipole, jnp.array(R), order=order)

    # Evaluation point near target center (small z-offset is OK)
    eval_point = jnp.array([0.3, 0.2, R - 0.1])

    # L2P
    delta_l2p = target_center - eval_point
    fmm_potential = evaluate_local_real(local, delta_l2p, order=order)

    # Direct potential
    direct = mass / jnp.linalg.norm(eval_point - source_offset)

    rel_error = abs(fmm_potential - direct) / abs(direct)

    # Should achieve good accuracy for well-separated expansion
    assert (
        rel_error < 1e-6
    ), f"Z-axis M2L with higher multipoles failed: error {rel_error:.2e}"


# ===========================================================================
# Full M2M/M2L/L2L tests
# ===========================================================================


def test_m2m_real_preserves_monopole():
    """M2M translation preserves the monopole (total mass)."""
    order = 4
    key = jax.random.PRNGKey(10)
    multipole = jax.random.normal(key, (sh_size(order),))
    delta = jnp.array([1.0, 0.5, -0.3])

    result = m2m_real(multipole, delta, order=order)

    # Monopole should be unchanged
    assert jnp.isclose(result[0], multipole[0], atol=1e-10)


def test_m2l_real_matches_optimized():
    """m2l_real and m2l_optimized_real should give identical results."""
    order = 4
    key = jax.random.PRNGKey(20)
    multipole = jax.random.normal(key, (sh_size(order),))
    delta = jnp.array([3.0, 1.0, 2.0])

    local1 = m2l_real(multipole, delta, order=order)
    local2 = m2l_optimized_real(multipole, delta, order=order)

    assert jnp.allclose(local1, local2, atol=1e-10)


def test_l2l_real_preserves_evaluation():
    """L2L translation should preserve the evaluated potential at a point.

    If we have local expansion L at center C, and translate to L' at center C',
    then evaluating L' at point P should give the same result when using
    the correct delta convention (center - eval_point) for each expansion.

    Uses z-axis alignment where the FMM operations are exact.

    Dehnen (2014) eq 3e: F'_n^m = Σ Υ_k^0(s - s') F_{n+k}^m
    where s = old center, s' = new center.
    So dz_l2l = old_z - new_z = parent_z - child_z.
    """
    order = 3
    key = jax.random.PRNGKey(30)
    local = jax.random.normal(key, (sh_size(order),))

    # Translation along z-axis from parent to child
    parent_z = 0.0
    child_z = 0.5
    # Dehnen convention: dz = old - new = parent - child
    dz = parent_z - child_z  # = -0.5

    # Translate local expansion along z
    local_child = translate_along_z_l2l_real(local, dz, order=order)

    # Pick a test point along z-axis
    parent_center = jnp.array([0.0, 0.0, parent_z])
    child_center = jnp.array([0.0, 0.0, child_z])
    test_point = jnp.array([0.0, 0.0, 0.6])  # point near child, along z

    # Dehnen convention: delta_l2p = center - eval_point
    delta_parent = parent_center - test_point  # for parent evaluation
    delta_child = child_center - test_point  # for child evaluation

    # Evaluate at both
    val_parent = evaluate_local_real(local, delta_parent, order=order)
    val_child = evaluate_local_real(local_child, delta_child, order=order)

    assert jnp.isclose(val_parent, val_child, rtol=1e-6)


# ===========================================================================
# Integration tests: full P2M → M2L → L2P pipeline
# ===========================================================================


def test_full_pipeline_single_particle():
    """Test complete P2M → M2L → L2P pipeline for single point mass.

    A unit mass at origin should produce potential 1/r at distance r.
    Uses z-axis alignment where the FMM is exact via rotation-accelerated M2L.
    Dehnen (2014) convention: L2P delta = center - eval_point.
    """
    order = 6

    # Source: unit mass at origin
    source_pos = jnp.array([0.0, 0.0, 0.0])
    mass = 1.0

    # Expansion centers along z-axis for exact evaluation
    multipole_center = jnp.array([0.0, 0.0, 0.0])
    local_center = jnp.array([0.0, 0.0, 5.0])  # Well separated along z

    # P2M: create multipole expansion
    delta_p2m = source_pos - multipole_center
    multipole = p2m_real_direct(delta_p2m, jnp.array(mass), order=order)

    # M2L: translate to local expansion using z-axis optimized function
    R = 5.0
    local = translate_along_z_m2l_real(multipole, R, order=order)

    # L2P: evaluate at test point
    # Dehnen convention: delta = center - eval_point
    test_point = jnp.array([0.0, 0.0, 5.5])  # Near local center, along z
    # center MINUS eval_point
    delta_l2p = local_center - test_point
    potential_fmm = evaluate_local_real(local, delta_l2p, order=order)

    # Direct calculation
    r = jnp.linalg.norm(test_point - source_pos)
    potential_direct = mass / r

    # Should match to good precision for well-separated expansion
    assert jnp.isclose(potential_fmm, potential_direct, rtol=1e-6)


def test_full_pipeline_with_m2m():
    """Test P2M → M2M → M2L → L2L → L2P pipeline along z-axis."""
    order = 5

    # Source: mass at small offset from origin (still near z-axis)
    source_pos = jnp.array([0.0, 0.0, 0.1])
    mass = 2.0

    # Leaf center (where P2M is computed)
    leaf_center = jnp.array([0.0, 0.0, 0.0])

    # Parent center along z-axis (M2M translates to here)
    parent_center = jnp.array([0.0, 0.0, 0.25])

    # Far target center along z-axis (M2L target)
    target_center = jnp.array([0.0, 0.0, 4.0])

    # Child of target (L2L target) - closer to source
    child_center = jnp.array([0.0, 0.0, 3.8])

    # P2M
    delta_p2m = source_pos - leaf_center
    multipole_leaf = p2m_real_direct(delta_p2m, jnp.array(mass), order=order)

    # M2M: leaf → parent (along z-axis)
    # M2M direction: dz = old_z - new_z = leaf_z - parent_z (Dehnen convention)
    dz_m2m = leaf_center[2] - parent_center[2]
    multipole_parent = translate_along_z_m2m_real(multipole_leaf, dz_m2m, order=order)

    # M2L: parent → target (along z-axis)
    R_m2l = target_center[2] - parent_center[2]
    local_target = translate_along_z_m2l_real(multipole_parent, R_m2l, order=order)

    # L2L: target → child (along z-axis)
    # Dehnen L2L direction: dz = old_z - new_z = target_z - child_z
    dz_l2l = target_center[2] - child_center[2]
    local_child = translate_along_z_l2l_real(local_target, dz_l2l, order=order)

    # L2P: evaluate using Dehnen convention (center - eval_point)
    test_point = jnp.array([0.0, 0.0, 3.7])  # Along z-axis
    # center MINUS eval_point
    delta_l2p = child_center - test_point
    potential_fmm = evaluate_local_real(local_child, delta_l2p, order=order)

    # Direct calculation
    r = jnp.linalg.norm(test_point - source_pos)
    potential_direct = mass / r

    # Should match reasonably well (not perfect due to truncation)
    assert jnp.isclose(potential_fmm, potential_direct, rtol=1e-4)


def test_gradient_accuracy():
    """Test that gradient from L2P matches expected gravitational acceleration.

    Uses Dehnen convention: delta = center - eval_point.
    The gradient returned is ∇_delta φ. Since delta = center - eval_point,
    ∇_{eval_point} φ = -∇_delta φ (chain rule).
    """
    order = 6

    # Source: unit mass at origin
    source_pos = jnp.array([0.0, 0.0, 0.0])
    mass = 1.0

    # Create multipole and local expansion along z-axis
    multipole_center = jnp.array([0.0, 0.0, 0.0])
    local_center = jnp.array([0.0, 0.0, 4.0])

    delta_p2m = source_pos - multipole_center
    multipole = p2m_real_direct(delta_p2m, jnp.array(mass), order=order)

    # Use z-axis optimized M2L
    R = 4.0
    local = translate_along_z_m2l_real(multipole, R, order=order)

    # Evaluate gradient at test point using Dehnen convention
    test_point = jnp.array([0.0, 0.0, 4.5])
    # center MINUS eval_point
    delta_l2p = local_center - test_point
    grad_delta, potential = evaluate_local_real_with_grad(local, delta_l2p, order=order)

    # grad_delta is ∇_delta φ. To get ∇_{eval_point} φ, we negate (chain rule).
    grad = -grad_delta

    # Direct gravitational field: for potential φ = mass/r,
    # gradient is ∇φ = -mass * r_vec / r³
    r_vec = test_point - source_pos
    r = jnp.linalg.norm(r_vec)
    grad_direct = -mass * r_vec / (r**3)

    # Use rtol=1e-4 to account for truncation error at finite expansion order
    assert jnp.allclose(grad, grad_direct, rtol=1e-4)


# ===========================================================================
# Rotated M2L tests (off-axis translation)
# ===========================================================================


def test_m2l_rotated_creates_valid_local_expansion():
    """M2L with rotation should create a valid local expansion."""
    order = 6

    # Source at origin with unit mass
    source_pos = jnp.array([0.0, 0.0, 0.0])
    mass = 1.0

    # Multipole center at origin, local center off-axis (generic direction)
    multipole_center = jnp.array([0.0, 0.0, 0.0])
    local_center = jnp.array([3.0, 2.0, 4.0])  # Generic off-axis direction

    # P2M: create multipole
    delta_p2m = source_pos - multipole_center
    multipole = p2m_real_direct(delta_p2m, jnp.array(mass), order=order)

    # M2L with rotation (off-axis)
    delta_m2l = local_center - multipole_center
    local = m2l_real(multipole, delta_m2l, order=order)

    # L2P: evaluate at test point near local center
    # Test point offset from local center (within convergence radius)
    test_point = local_center + jnp.array([0.3, -0.2, 0.1])
    # center MINUS eval_point (Dehnen convention)
    delta_l2p = local_center - test_point
    potential_fmm = evaluate_local_real(local, delta_l2p, order=order)

    # Direct calculation: φ = mass / r
    r = jnp.linalg.norm(test_point - source_pos)
    potential_direct = mass / r

    # At order=6, with well-separated source and eval point, should be accurate
    # Using rtol=5e-2 as a basic sanity check; scaling test is stricter
    assert jnp.isclose(potential_fmm, potential_direct, rtol=5e-2), (
        f"FMM potential {potential_fmm} vs direct {potential_direct}, "
        "error = "
        f"{abs(potential_fmm - potential_direct) / abs(potential_direct):.2%}"
    )


def test_m2l_rotated_matches_alignment_pipeline():
    """Rotated M2L should match an explicit alignment pipeline.

    This validates the fast-rotation approach: rotate multipoles to the
    z-axis, translate along z, then rotate locals back.
    """
    order = 6
    dtype = jnp.float64

    # Source near origin to exercise higher-order terms.
    source_pos = jnp.array([0.2, -0.1, 0.3], dtype=dtype)
    mass = jnp.asarray(1.0, dtype=dtype)
    multipole_center = jnp.array([0.0, 0.0, 0.0], dtype=dtype)

    # Local center at generic off-axis position
    local_center = jnp.array([3.0, 2.0, 4.0], dtype=dtype)

    delta_p2m = source_pos - multipole_center
    multipole = p2m_real_direct(delta_p2m, mass, order=order)

    # Fast rotated M2L
    delta_m2l = local_center - multipole_center
    local_fast = m2l_real(multipole, delta_m2l, order=order)

    # Manual alignment pipeline
    x, y, z = delta_m2l
    R = jnp.linalg.norm(delta_m2l)

    multipole_rot = jnp.zeros_like(multipole)
    for ell in range(order + 1):
        sl = slice(ell * ell, (ell + 1) * (ell + 1))
        D_to = real_rotation_to_z_axis_multipole(x, y, z, ell, dtype=dtype)
        multipole_rot = multipole_rot.at[sl].set(D_to @ multipole[sl])

    local_z = translate_along_z_m2l_real(multipole_rot, jnp.asarray(R), order=order)

    local_manual = jnp.zeros_like(local_z)
    for ell in range(order + 1):
        sl = slice(ell * ell, (ell + 1) * (ell + 1))
        D_from = real_rotation_from_z_axis_local(x, y, z, ell, dtype=dtype)
        local_manual = local_manual.at[sl].set(D_from @ local_z[sl])

    assert jnp.allclose(local_fast, local_manual, rtol=1e-10, atol=1e-10)


def test_m2l_rotated_error_improves_with_order():
    """Off-axis M2L (rotation-accelerated real path) converges with order.

    Regression test for the two bugs that previously capped off-axis M2L at a
    few percent regardless of order: (1) the missing factor of 2 on the m != 0
    channels of the z-axis M2L, and (2) the wrong alignment azimuth in the
    B-matrix rotations (plus the local rotation being a separate ``B_T`` matrix
    rather than the transpose of the multipole rotation).
    """
    dtype = jnp.float64
    # Multipole is expanded about the ORIGIN (p2m_real_direct(source_pos)), so
    # the M2L displacement is origin -> local_center = local_center.
    source_pos = jnp.array([0.2, -0.1, 0.3], dtype=dtype)
    local_center = jnp.array([3.0, 2.0, 4.0], dtype=dtype)
    eval_offset = jnp.array([0.2, -0.1, 0.15], dtype=dtype)

    eval_point = local_center + eval_offset
    r = jnp.linalg.norm(eval_point - source_pos)
    potential_direct = 1.0 / r

    # 3 representative points span the geometric-convergence claim (each step
    # cuts >5x, order 8 reaches ~machine precision) with one fewer compile.
    orders = [2, 5, 8]
    errors = []
    for order in orders:
        multipole = p2m_real_direct(
            source_pos, jnp.asarray(1.0, dtype=dtype), order=order
        )
        delta_m2l = local_center
        local = m2l_real(multipole, delta_m2l, order=order)
        delta_l2p = local_center - eval_point
        potential_fmm = evaluate_local_real(local, delta_l2p, order=order)
        err = float(
            jnp.abs(potential_fmm - potential_direct) / jnp.abs(potential_direct)
        )
        errors.append(err)

    # Geometric convergence: each step should cut the error substantially, and
    # the highest order should reach near machine precision.
    for lo, hi in zip(errors, errors[1:]):
        if lo > 1e-12:
            assert hi < lo * 0.2
    assert errors[-1] < 1e-9


def test_full_rotated_pipeline_m2m_m2l_l2l_converges():
    """Fully off-axis P2M -> M2M -> M2L -> L2L -> L2P converges with order.

    Exercises the rotation-accelerated M2M and L2L operators (which had no
    off-axis correctness coverage before) together with the fixed M2L.
    """
    dtype = jnp.float64
    source = jnp.array([0.15, 0.1, 0.05], dtype=dtype)
    leaf = jnp.array([0.0, 0.0, 0.0], dtype=dtype)
    parent = jnp.array([0.3, 0.2, 0.1], dtype=dtype)
    target = jnp.array([3.0, 2.0, 4.0], dtype=dtype)
    child = jnp.array([3.1, 2.1, 3.8], dtype=dtype)
    eval_point = child + jnp.array([0.05, -0.06, 0.04], dtype=dtype)
    direct = 1.0 / jnp.linalg.norm(eval_point - source)

    errors = []
    # 3 representative points keep the monotone-convergence + final-accuracy
    # claim (order 9 unchanged) with one fewer compile.
    for order in (3, 6, 9):
        multipole_leaf = p2m_real_direct(
            source - leaf, jnp.asarray(1.0, dtype=dtype), order=order
        )
        # M2M/L2L use the old_center - new_center (dest -> source) convention.
        multipole_parent = m2m_real(multipole_leaf, leaf - parent, order=order)
        local_target = m2l_real(multipole_parent, target - parent, order=order)
        local_child = l2l_real(local_target, target - child, order=order)
        potential = evaluate_local_real(local_child, child - eval_point, order=order)
        errors.append(float(jnp.abs(potential - direct) / jnp.abs(direct)))

    for lo, hi in zip(errors, errors[1:]):
        if lo > 1e-12:
            assert hi < lo
    assert errors[-1] < 1e-9


# ===========================================================================
# Rotation blocks against the physical rotation they claim to represent
# ===========================================================================


def _rot_z(angle: float) -> np.ndarray:
    """Right-handed coordinate-space rotation about +z."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle: float) -> np.ndarray:
    """Right-handed coordinate-space rotation about +x."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _physical_alignment_rotation(direction) -> np.ndarray:
    """The coordinate-space ``g`` with ``g @ direction == (0, 0, |direction|)``.

    This is the rotation named in :func:`_multipole_align_to_z_block`'s docstring:
    ``g = Rx(ax) @ Rz(az)`` with ``az = atan2(x, y)`` and ``ax = atan2(rho, z)``.
    Built here from first principles in NumPy so the test does not borrow the
    production azimuth convention it is trying to check -- the assertion below
    would fail if either the convention or the block were wrong, and
    ``g @ direction == (0, 0, r)`` is verified inline so a wrong ``g`` cannot
    silently make the test vacuous.
    """
    x, y, z = (float(t) for t in direction)
    return _rot_x(np.arctan2(np.hypot(x, y), z)) @ _rot_z(np.arctan2(x, y))


def _block_diagonal_rotation(builder, direction, order: int, dtype) -> np.ndarray:
    """Assemble a builder's per-degree blocks into one ``[(p+1)^2, (p+1)^2]``."""
    size = sh_size(order)
    full = np.zeros((size, size))
    args = [jnp.asarray(float(t), dtype=dtype) for t in direction]
    for ell in range(order + 1):
        block = np.asarray(builder(*args, ell, dtype=dtype))
        lo, hi = ell * ell, (ell + 1) * (ell + 1)
        full[lo:hi, lo:hi] = block
    return full


# Generic off-axis directions. Every component is nonzero and the octants differ,
# so a per-``m`` sign error or a swapped azimuth cannot hide behind a symmetry.
_ROTATION_DIRECTIONS = [
    pytest.param([1.2, -0.7, 2.5], id="x+y-z+"),
    pytest.param([0.7, -0.3, 0.45], id="near-diagonal"),
    pytest.param([-1.1, 2.0, -0.4], id="x-y+z-"),
    pytest.param([0.3, 0.9, -1.7], id="steep-negative-z"),
]

# Measured worst case across these 4 directions x 3 source/target draws x
# ell = 0..6, for all four identities below: 2.5e-15 relative (~10 eps_f64). These
# are exact algebraic identities, not truncations, so round-off is the only
# admissible error. 1e-12 keeps ~400x headroom and is still a sharp instrument:
# perturbing the alignment azimuth by a relative 1e-12 moves the multipole
# identity to 5.1e-12, i.e. this bound catches an azimuth error of ~2e-13
# relative. The historical defect this guards -- azimuth atan2(x, y) written as
# atan2(y, x), the convention flagged CRITICAL at real_harmonics.py:1524 -- gives
# 1.8e+00, so it fails by twelve orders of magnitude.
_ROTATION_IDENTITY_TOL = 1.0e-12


@pytest.mark.parametrize("direction", _ROTATION_DIRECTIONS)
def test_multipole_rotation_blocks_match_p2m_of_the_rotated_source(direction):
    """``D_to @ p2m(s) == p2m(g @ s)``, and ``D_from`` undoes it.

    This is the identity :func:`_multipole_align_to_z_block` asserts in its own
    docstring (*"(this block) @ p2m(s)[block] == p2m(g @ s)[block]"*) and that
    nothing checked until now. It is the only assertion in this file that pins the
    rotation blocks against something **independent** of themselves.

    The four pre-existing rotation tests cannot: ``test_rotation_z_axis_is_identity``
    uses a z-aligned direction, where a wrong azimuth is unobservable;
    ``test_rotation_preserves_monopole`` tests ``ell=0``, true under any
    normalisation; ``test_rotation_to_from_z_axis_are_inverses`` asserts
    ``D_from @ D_to == I``, an involution that any consistently-wrong pair
    satisfies; and ``test_alignment_pipeline_steps_match_p2m`` checks the ``B`` and
    ``Dz`` *building blocks* using its own ``arctan2(y, x)``, so it never
    constructs the assembled block and never exercises the production
    ``arctan2(x, y)`` convention.

    That matters because a wrong azimuth here does not stay local: it surfaces
    four layers downstream as "the real basis does not converge", which is how it
    was found the last time.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("exact-identity tolerance requires float64 (JAX_ENABLE_X64=1)")

    dtype = jnp.float64
    order = 6
    g = _physical_alignment_rotation(direction)

    # Guard against a vacuous test: if `g` does not actually align `direction`
    # with +z, the identities below would be comparing two wrong things.
    aligned = g @ np.asarray(direction, dtype=np.float64)
    radius = float(np.linalg.norm(direction))
    np.testing.assert_allclose(aligned, [0.0, 0.0, radius], rtol=0, atol=1e-14)

    d_to = _block_diagonal_rotation(
        real_rotation_to_z_axis_multipole, direction, order, dtype
    )
    d_from = _block_diagonal_rotation(
        real_rotation_from_z_axis_multipole, direction, order, dtype
    )

    rng = np.random.default_rng(4242)
    for _ in range(3):
        source = rng.normal(size=3)
        unit_mass = jnp.asarray(1.0, dtype=dtype)
        world = np.asarray(
            p2m_real_direct(jnp.asarray(source, dtype=dtype), unit_mass, order=order)
        )
        rotated = np.asarray(
            p2m_real_direct(
                jnp.asarray(g @ source, dtype=dtype), unit_mass, order=order
            )
        )

        for label, got, want in (
            ("world->z", d_to @ world, rotated),
            ("z->world", d_from @ rotated, world),
        ):
            rel_l2 = float(
                np.linalg.norm(got - want) / max(float(np.linalg.norm(want)), 1e-300)
            )
            assert rel_l2 < _ROTATION_IDENTITY_TOL, (
                f"multipole {label} rotation disagrees with the physical rotation "
                f"of P2M at direction={direction}, source={source}: "
                f"rel-L2 {rel_l2:.3e}"
            )


@pytest.mark.parametrize("direction", _ROTATION_DIRECTIONS)
def test_local_rotation_blocks_leave_the_evaluated_potential_invariant(direction):
    """Rotating a local expansion and its evaluation point cancels exactly.

    The local blocks have no P2M analogue to compare against -- local coefficients
    contract against the *same* ``U_n^m`` as P2M (see
    :func:`real_rotation_from_z_axis_local`), so what pins them is the physical
    invariant behind that choice: a potential does not care which frame it is
    evaluated in. ``evaluate_local_real(D_to @ L, g @ t) == evaluate_local_real(L, t)``.

    This is the assertion that distinguishes the transpose convention from its
    inverse. ``real_rotation_to_z_axis_local`` is
    ``_multipole_align_from_z_block(...).T`` -- transpose, not inverse -- and the
    two coincide only because these blocks are orthogonal up to the Dehnen basis
    scaling. If that ever stops holding, the potential stops being invariant and
    this test says so; ``test_rotation_to_from_z_axis_are_inverses`` would not.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("exact-identity tolerance requires float64 (JAX_ENABLE_X64=1)")

    dtype = jnp.float64
    order = 6
    g = _physical_alignment_rotation(direction)

    d_to = _block_diagonal_rotation(
        real_rotation_to_z_axis_local, direction, order, dtype
    )
    d_from = _block_diagonal_rotation(
        real_rotation_from_z_axis_local, direction, order, dtype
    )

    rng = np.random.default_rng(4243)
    for _ in range(3):
        coeffs = rng.normal(size=sh_size(order))
        # Kept well inside the convergence radius of the expansion; the identity
        # is algebraic, but a wild evaluation point makes the numbers meaningless.
        target = 0.5 * rng.normal(size=3)

        def potential(local_coeffs, delta):
            return float(
                evaluate_local_real(
                    jnp.asarray(local_coeffs, dtype=dtype),
                    jnp.asarray(delta, dtype=dtype),
                    order=order,
                )
            )

        for label, got, want in (
            (
                "world->z",
                potential(d_to @ coeffs, g @ target),
                potential(coeffs, target),
            ),
            (
                "z->world",
                potential(d_from @ coeffs, target),
                potential(coeffs, g @ target),
            ),
        ):
            rel = abs(got - want) / max(abs(want), 1e-300)
            assert rel < _ROTATION_IDENTITY_TOL, (
                f"local {label} rotation changes the evaluated potential at "
                f"direction={direction}, target={target}: {got!r} vs {want!r} "
                f"(rel {rel:.3e})"
            )


# The rotate -> z-translate -> rotate-back cascade, at the degenerate separation
# rho == 0 where the alignment azimuth is undefined. Kept together because one half
# of the behaviour is correct and must not regress, and the other half is a tracked
# defect (docs/refactor_audit_2026-08.md G.10).
_ROTATION_CASCADE_OPERATORS = [
    pytest.param(m2l_real, id="m2l_real"),
    pytest.param(m2m_real, id="m2m_real"),
    pytest.param(l2l_real, id="l2l_real"),
]

_CASCADE_ORDER = 4
_CASCADE_Z = 2.5


def _cascade_multipole():
    """A fixed multipole with nonzero m != 0 content, so the azimuth matters."""
    coeffs = np.zeros(sh_size(_CASCADE_ORDER))
    coeffs[0] = 1.0
    coeffs[1] = 0.3
    coeffs[4] = -0.2
    coeffs[7] = 0.15
    return jnp.asarray(coeffs, dtype=jnp.float64)


def _cascade_gradient(operator, delta):
    """``grad`` of a fixed-cotangent scalar loss on ``operator`` w.r.t. ``delta``."""
    multipole = _cascade_multipole()
    weights = jax.random.normal(
        jax.random.PRNGKey(5), (sh_size(_CASCADE_ORDER),), dtype=jnp.float64
    )

    def loss(d):
        return jnp.sum(weights * operator(multipole, d, order=_CASCADE_ORDER))

    return np.asarray(
        jax.grad(loss)(jnp.asarray(delta, dtype=jnp.float64)), dtype=np.float64
    )


def _cascade_offaxis_gradient_limit(operator, num_directions=8, rho=1.0e-9):
    """The rho -> 0 limit of the gradient, averaged over approach directions.

    Averaging is safe *because* the limit is direction-independent, which
    :func:`test_rotation_cascade_gradient_limit_is_direction_independent` asserts
    separately -- so this helper is never hiding a spread.
    """
    grads = [
        _cascade_gradient(
            operator,
            [
                rho * np.cos(2.0 * np.pi * k / num_directions),
                rho * np.sin(2.0 * np.pi * k / num_directions),
                _CASCADE_Z,
            ],
        )
        for k in range(num_directions)
    ]
    return np.mean(np.array(grads), axis=0)


@pytest.mark.parametrize("operator", _ROTATION_CASCADE_OPERATORS)
def test_rotation_cascade_gradient_limit_is_direction_independent(operator):
    """The cascade is genuinely differentiable at ``rho == 0``.

    This is the premise the next two tests rest on, so it is asserted rather than
    assumed. If the gradient limit depended on the approach direction there would be
    no derivative at ``rho == 0``, a zero cotangent would be a defensible subgradient
    choice, and G.10 would not be a defect.

    Measured spread across eight approach directions at ``rho = 1e-9`` is ~1.4e-07,
    which is finite-difference noise at that step, not structure. The bound is 1e-5:
    loose enough not to be measuring round-off, tight enough that a genuinely
    direction-dependent limit (which would be O(1) here -- the components themselves
    are order unity) fails it.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("gradient limits require float64 (JAX_ENABLE_X64=1)")

    grads = np.array(
        [
            _cascade_gradient(
                operator,
                [
                    1.0e-9 * np.cos(2.0 * np.pi * k / 8),
                    1.0e-9 * np.sin(2.0 * np.pi * k / 8),
                    _CASCADE_Z,
                ],
            )
            for k in range(8)
        ]
    )
    spread = float(np.max(grads.max(axis=0) - grads.min(axis=0)))
    assert spread < 1.0e-5, (
        "the rho -> 0 gradient limit must be direction-independent for the "
        f"derivative to exist; spread across 8 directions is {spread:.3e}"
    )


@pytest.mark.parametrize("operator", _ROTATION_CASCADE_OPERATORS)
def test_rotation_cascade_radial_gradient_at_rho_zero_is_correct(operator):
    """The ``d/dz`` component at ``rho == 0`` is right, and finite -- do not regress it.

    The degeneracy guards in ``_multipole_align_{to,from}_z_block`` lose the two
    transverse components (the next test), but they do *not* damage the radial one,
    and they do keep the whole gradient finite. Both halves are worth pinning: the
    finiteness is the guards' actual purpose, and a future fix for G.10 must not
    trade a wrong transverse component for a wrong radial one.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("gradient limits require float64 (JAX_ENABLE_X64=1)")

    at_zero = _cascade_gradient(operator, [0.0, 0.0, _CASCADE_Z])
    limit = _cascade_offaxis_gradient_limit(operator)

    assert np.all(np.isfinite(at_zero)), f"gradient is not finite: {at_zero}"
    # Relative bound: these components span ~0.07 to ~9.7 across the three operators.
    assert abs(at_zero[2] - limit[2]) <= 1.0e-6 * max(
        abs(limit[2]), 1.0
    ), f"d/dz at rho == 0 is {at_zero[2]:.9f}, off-axis limit is {limit[2]:.9f}"


@pytest.mark.parametrize("operator", _ROTATION_CASCADE_OPERATORS)
def test_rotation_cascade_transverse_gradient_at_rho_zero(operator):
    """``d/dx`` and ``d/dy`` at ``rho == 0`` must equal the off-axis limit.

    This was G.10, a strict xfail: the degeneracy guards in
    ``_multipole_align_{to,from}_z_block`` returned a zero cotangent for both
    transverse components (true values m2l_real -1.502050 / -0.523434, m2m_real
    -6.416905 / +1.769043, l2l_real +0.305315 / +0.003498), and no guard choice could
    recover them -- the code reaches ``(x, y)`` only through ``rho`` and
    ``atan2(x, y)``, so the polar parametrisation has already divided out the
    ``O(rho)`` coefficient the derivative needs. The three operators now carry a
    ``custom_jvp`` that supplies it analytically instead; see
    :mod:`jaccpot.operators._transverse_degeneracy_jvp` and
    ``docs/rotation_degeneracy_derivative.md``.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("gradient limits require float64 (JAX_ENABLE_X64=1)")

    at_zero = _cascade_gradient(operator, [0.0, 0.0, _CASCADE_Z])
    limit = _cascade_offaxis_gradient_limit(operator)

    for component, axis in ((0, "x"), (1, "y")):
        assert abs(at_zero[component] - limit[component]) <= 1.0e-5 * max(
            abs(limit[component]), 1.0
        ), (
            f"d/d{axis} at rho == 0 is {at_zero[component]:.9f}, but the "
            f"off-axis limit is {limit[component]:.9f}"
        )


def test_rotation_to_from_z_axis_are_inverses():
    """Rotation to/from z-axis should compose to identity per degree."""
    dtype = jnp.float64
    x, y, z = (
        jnp.asarray(1.3, dtype=dtype),
        jnp.asarray(-0.7, dtype=dtype),
        jnp.asarray(2.1, dtype=dtype),
    )

    for ell in range(1, 7):
        n = 2 * ell + 1
        eye = jnp.eye(n, dtype=dtype)

        D_to_m = real_rotation_to_z_axis_multipole(x, y, z, ell, dtype=dtype)
        D_from_m = real_rotation_from_z_axis_multipole(x, y, z, ell, dtype=dtype)
        assert jnp.allclose(D_from_m @ D_to_m, eye, rtol=1e-12, atol=1e-12)
        assert jnp.allclose(D_to_m @ D_from_m, eye, rtol=1e-12, atol=1e-12)

        D_to_l = real_rotation_to_z_axis_local(x, y, z, ell, dtype=dtype)
        D_from_l = real_rotation_from_z_axis_local(x, y, z, ell, dtype=dtype)
        assert jnp.allclose(D_from_l @ D_to_l, eye, rtol=1e-12, atol=1e-12)
        assert jnp.allclose(D_to_l @ D_from_l, eye, rtol=1e-12, atol=1e-12)


def test_alignment_pipeline_steps_match_p2m():
    """Stepwise rotation/swap pipeline should match P2M at each step."""
    order = 6
    dtype = jnp.float64

    x, y, z = (
        jnp.asarray(1.2, dtype=dtype),
        jnp.asarray(-0.7, dtype=dtype),
        jnp.asarray(2.5, dtype=dtype),
    )
    delta0 = jnp.array([x, y, z], dtype=dtype)
    coeffs0 = p2m_real_direct(delta0, jnp.asarray(1.0, dtype=dtype), order=order)

    def rot_z(vec, angle):
        c = jnp.cos(angle)
        s = jnp.sin(angle)
        return jnp.array(
            [c * vec[0] - s * vec[1], s * vec[0] + c * vec[1], vec[2]], dtype=dtype
        )

    rho = jnp.sqrt(x * x + y * y)
    alpha_z = jnp.arctan2(y, x)
    alpha_x = jnp.arctan2(rho, z)

    # Step 1: z-rotation by -alpha_z
    delta1 = rot_z(delta0, -alpha_z)
    coeffs1_ref = p2m_real_direct(delta1, jnp.asarray(1.0, dtype=dtype), order=order)

    # Step 2: swap (x,y,z) -> (z,y,x)
    delta2 = jnp.array([delta1[2], delta1[1], delta1[0]], dtype=dtype)
    coeffs2_ref = p2m_real_direct(delta2, jnp.asarray(1.0, dtype=dtype), order=order)

    # Step 3: z-rotation in swapped frame by -alpha_x
    delta3 = rot_z(delta2, -alpha_x)
    coeffs3_ref = p2m_real_direct(delta3, jnp.asarray(1.0, dtype=dtype), order=order)

    # Step 4: swap back
    delta4 = jnp.array([delta3[2], delta3[1], delta3[0]], dtype=dtype)
    coeffs4_ref = p2m_real_direct(delta4, jnp.asarray(1.0, dtype=dtype), order=order)

    for ell in range(order + 1):
        sl = slice(ell * ell, (ell + 1) * (ell + 1))
        B = compute_real_B_matrix_multipole(ell, dtype=dtype)
        Dz = lambda a: real_Dz_diagonal(ell, a, dtype=dtype)

        # Step 1 coefficients
        coeffs1 = Dz(-alpha_z) @ coeffs0[sl]
        assert jnp.allclose(coeffs1, coeffs1_ref[sl], rtol=1e-10, atol=1e-10)

        # Step 2 coefficients
        coeffs2 = B @ coeffs1
        assert jnp.allclose(coeffs2, coeffs2_ref[sl], rtol=1e-10, atol=1e-10)

        # Step 3 coefficients
        coeffs3 = Dz(-alpha_x) @ coeffs2
        assert jnp.allclose(coeffs3, coeffs3_ref[sl], rtol=1e-10, atol=1e-10)

        # Step 4 coefficients
        coeffs4 = B @ coeffs3
        assert jnp.allclose(coeffs4, coeffs4_ref[sl], rtol=1e-10, atol=1e-10)


def test_z_m2l_respects_z_rotation_symmetry():
    """Z-axis M2L should commute with z-rotations (checks sin-channel signs)."""
    order = 4
    dtype = jnp.float64

    # Two sources related by +90° rotation about z: x->y.
    source_x = jnp.array([0.5, 0.0, 0.2], dtype=dtype)
    source_y = jnp.array([0.0, 0.5, 0.2], dtype=dtype)
    mass = jnp.asarray(1.0, dtype=dtype)

    multipole_x = p2m_real_direct(source_x, mass, order=order)
    multipole_y = p2m_real_direct(source_y, mass, order=order)

    R = jnp.asarray(4.0, dtype=dtype)
    local_x = translate_along_z_m2l_real(multipole_x, R, order=order)
    local_y = translate_along_z_m2l_real(multipole_y, R, order=order)

    # Rotating local_x by +90° about z should match local_y.
    phi = jnp.asarray(jnp.pi / 2, dtype=dtype)
    for ell in range(order + 1):
        sl = slice(ell * ell, (ell + 1) * (ell + 1))
        Dz = real_Dz_diagonal(ell, phi, dtype=dtype)
        rotated = Dz @ local_x[sl]
        assert jnp.allclose(rotated, local_y[sl], rtol=1e-10, atol=1e-10)


def test_m2l_delta_convention_matches_direct_and_wrong_sign_is_worse():
    """Correct delta sign should match direct potential; wrong sign should be worse."""
    order = 6
    dtype = jnp.float64

    source_pos = jnp.array([0.0, 0.0, 0.0], dtype=dtype)
    local_center = jnp.array([3.0, 2.0, 4.0], dtype=dtype)
    eval_offset = jnp.array([0.2, -0.1, 0.15], dtype=dtype)
    eval_point = local_center + eval_offset

    multipole = p2m_real_direct(source_pos, jnp.asarray(1.0, dtype=dtype), order=order)

    delta_correct = local_center - source_pos
    delta_wrong = -delta_correct

    local_correct = m2l_real(multipole, delta_correct, order=order)
    local_wrong = m2l_real(multipole, delta_wrong, order=order)

    delta_l2p = local_center - eval_point
    potential_correct = evaluate_local_real(local_correct, delta_l2p, order=order)
    potential_wrong = evaluate_local_real(local_wrong, delta_l2p, order=order)

    r = jnp.linalg.norm(eval_point - source_pos)
    potential_direct = 1.0 / r

    err_correct = jnp.abs(potential_correct - potential_direct) / jnp.abs(
        potential_direct
    )
    err_wrong = jnp.abs(potential_wrong - potential_direct) / jnp.abs(potential_direct)

    assert err_correct < 5e-2
    assert err_wrong > err_correct * 2.0


# ===========================================================================
# JIT compilation tests
# ===========================================================================


def test_all_functions_jittable():
    """All main functions should be JIT-compilable."""
    order = 3
    key = jax.random.PRNGKey(100)

    delta = jnp.array([1.0, 2.0, 0.5])
    mass = jnp.array(1.0)
    multipole = jax.random.normal(key, (sh_size(order),))
    local = jax.random.normal(jax.random.PRNGKey(101), (sh_size(order),))

    # JIT compile and run each function
    jax.jit(lambda d, m: p2m_real_direct(d, m, order=order))(delta, mass)
    jax.jit(lambda loc, d: evaluate_local_real(loc, d, order=order))(local, delta)
    jax.jit(lambda loc, d: evaluate_local_real_with_grad(loc, d, order=order))(
        local, delta
    )
    jax.jit(lambda m, d: m2m_real(m, d, order=order))(multipole, delta)
    jax.jit(lambda m, d: m2l_real(m, d, order=order))(multipole, delta)
    jax.jit(lambda m, d: m2l_optimized_real(m, d, order=order))(multipole, delta)
    jax.jit(lambda loc, d: l2l_real(loc, d, order=order))(local, delta)


def test_functions_vmappable():
    """Main functions should work with vmap."""
    order = 3
    n_particles = 10
    key = jax.random.PRNGKey(200)

    deltas = jax.random.normal(key, (n_particles, 3))
    masses = jax.random.uniform(jax.random.PRNGKey(201), (n_particles,))

    # vmap P2M
    multipoles = jax.vmap(lambda d, m: p2m_real_direct(d, m, order=order))(
        deltas, masses
    )
    assert multipoles.shape == (n_particles, sh_size(order))


# ===========================================================================
# Complex -> Dehnen-real conversion: the identity the docstring promises
# ===========================================================================


# Geometries chosen to cover the degenerate azimuths as well as a generic point:
# `rho == 0` (z-aligned) is where the azimuth is undefined, and the axis cases
# are where a per-m sign or normalisation error shows up most cleanly.
_CONVERSION_DELTAS = [
    pytest.param([0.7, -0.3, 0.45], id="generic-off-axis"),
    pytest.param([0.0, 0.0, 0.8], id="z-aligned-rho-zero"),
    pytest.param([0.9, 0.0, 0.0], id="x-aligned"),
    pytest.param([0.0, -0.6, 0.0], id="y-aligned"),
    pytest.param([0.5, 0.5, 0.0], id="xy-plane"),
    pytest.param([1e-8, -2e-8, 3e-8], id="near-origin"),
]


@pytest.mark.parametrize("delta", _CONVERSION_DELTAS)
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 6])
def test_complex_to_dehnen_real_matches_p2m_real_direct(delta, order):
    """``complex_to_dehnen_real_coeffs(complex_R_solidfmm(d)) == p2m_real_direct(d)``.

    This is the equivalence ``complex_to_dehnen_real_coeffs`` claims in its own
    docstring and that nothing asserted until now. It is the seam between the
    complex solidfmm basis and the Dehnen no-sqrt2 real operators, so a per-``m``
    sign or normalisation error here silently corrupts every real-basis M2L.

    The two proxies that existed cannot catch that:
    ``tests/unit/core/test_real_upward_sweep.py::test_real_upward_matches_complex_convert``
    checks an aggregate relative L2 over a whole tree, where a single-``m`` error
    is diluted, and ``test_dehnen_power_is_basis_invariant`` checks only the
    degree-wise Dehnen power, which is a rotational invariant and therefore blind
    to sign errors within a degree.

    Tolerance: this is an exact algebraic identity, not a truncation, so the only
    admissible error is float64 round-off in the two independent recurrences.
    Measured worst case across these 36 combinations is 8.9e-16 relative L2
    (~4 eps); 1e-13 leaves ~2 orders of headroom without admitting a real defect.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("exact-identity tolerance requires float64 (JAX_ENABLE_X64=1)")

    d = jnp.asarray(delta, dtype=jnp.float64)
    unit_mass = jnp.asarray(1.0, dtype=jnp.float64)

    converted = complex_to_dehnen_real_coeffs(
        complex_R_solidfmm(d, order=order), order=order
    )
    direct = p2m_real_direct(d, unit_mass, order=order)

    assert converted.shape == direct.shape == (sh_size(order),)

    converted_np = np.asarray(converted)
    direct_np = np.asarray(direct)
    rel_l2 = float(
        np.linalg.norm(converted_np - direct_np)
        / max(float(np.linalg.norm(direct_np)), 1e-300)
    )
    assert rel_l2 < 1e-13, (
        f"complex->real conversion disagrees with p2m_real_direct at "
        f"delta={delta}, order={order}: rel-L2 {rel_l2:.3e}"
    )


def _conjugate_symmetric_packed(order: int, *, m0_real: bool, seed: int):
    """Build a packed complex array via ``_pack_complex``, m=0 real or not.

    ``_pack_complex`` enforces ``H_n^{-m} = (-1)^m conj(H_n^m)`` for the negative
    ``m`` slots, but it copies ``m = 0`` through verbatim -- so whether the array
    satisfies the *full* reality condition depends on whether the caller made the
    m=0 entries real. That is the distinction this helper exists to expose.
    """
    rng = np.random.default_rng(seed)
    half = np.zeros((order + 1, order + 1), dtype=np.complex128)
    for n in range(order + 1):
        for m in range(n + 1):
            imag = 0.0 if (m == 0 and m0_real) else rng.normal()
            half[n, m] = rng.normal() + 1j * imag
    return _pack_complex(jnp.asarray(half))


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_complex_to_dehnen_real_discards_only_exact_zero_for_conforming_input(order):
    """``Im(coeffs @ Q^T)`` is EXACTLY zero when the reality condition holds.

    That is what makes ``jnp.real`` lossless here rather than a projection. The
    condition is ``H_n^{-m} = (-1)^m conj(H_n^m)``, whose ``m = 0`` case reads
    ``H_n^0 = conj(H_n^0)`` -- i.e. **the m=0 coefficients must be real**. That half
    is easy to miss, and it is the half this test pins: with m=0 real the imaginary
    part of the product is identically 0.0, and with m=0 complex it is a substantial
    fraction of the real part, so ``jnp.real`` would be discarding real information
    rather than round-off.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("exact-zero check requires float64")

    q_full = np.asarray(_dehnen_real_Q_full(order))

    conforming = np.asarray(_conjugate_symmetric_packed(order, m0_real=True, seed=9))
    product = conforming @ q_full.T
    assert np.max(np.abs(product.imag)) == 0.0, (
        "the imaginary part must vanish exactly for conforming coefficients, "
        f"got max {np.max(np.abs(product.imag)):.3e}"
    )
    assert np.linalg.norm(product.real) > 0.0

    # And the near-miss: conjugate-symmetric for m != 0 but complex at m = 0.
    non_conforming = np.asarray(
        _conjugate_symmetric_packed(order, m0_real=False, seed=9)
    )
    bad = non_conforming @ q_full.T
    discarded = np.linalg.norm(bad.imag) / max(np.linalg.norm(bad.real), 1e-300)
    assert discarded > 1e-3, (
        "a complex m=0 entry should make the discarded imaginary part significant, "
        f"so that this precondition is worth documenting; got ratio {discarded:.3e}"
    )


def test_complex_to_dehnen_real_vjp_carries_the_imaginary_part():
    """The VJP is the complete adjoint -- it is NOT blind to ``Im(coeffs)``.

    ``jnp.real`` on the *output* does not decouple the input's imaginary part,
    because ``Q`` is complex and ``Im(coeffs)`` therefore contributes to
    ``Re(coeffs @ Q^T)``. This asserts both halves of that: the returned cotangent
    has a nonzero imaginary component, and the directional derivative it predicts
    matches finite differences along an **imaginary** perturbation as well as a real
    one.

    Written because an earlier revision of this function's docstring claimed the
    opposite -- that the VJP discarded the imaginary cotangent -- inferred from the
    presence of ``jnp.real`` rather than measured. A test is what stops that
    reappearing.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("finite-difference comparison requires float64")

    order = 3
    size = sh_size(order)
    rng = np.random.default_rng(4)
    coeffs = jnp.asarray(
        rng.normal(size=size) + 1j * rng.normal(size=size), dtype=jnp.complex128
    )
    cotangent = jnp.asarray(rng.normal(size=size), dtype=jnp.float64)

    def convert(z):
        return complex_to_dehnen_real_coeffs(z, order=order)

    _, vjp_fn = jax.vjp(convert, coeffs)
    (grad,) = vjp_fn(cotangent)

    assert jnp.linalg.norm(jnp.imag(grad)) > 0.0, (
        "the cotangent must carry an imaginary component; a purely real one would "
        "mean the gradient is blind to Im(coeffs)"
    )

    step = 1e-7
    for name, direction in (
        ("real", jnp.asarray(rng.normal(size=size), dtype=jnp.complex128)),
        ("imaginary", jnp.asarray(1j * rng.normal(size=size), dtype=jnp.complex128)),
    ):
        finite_difference = float(
            np.dot(
                np.asarray(cotangent),
                (
                    np.asarray(convert(coeffs + step * direction))
                    - np.asarray(convert(coeffs - step * direction))
                )
                / (2 * step),
            )
        )
        predicted = float(jnp.real(jnp.sum(grad * direction)))
        rel = abs(predicted - finite_difference) / max(abs(finite_difference), 1e-300)
        assert rel < 1e-6, (
            f"VJP disagrees with the finite difference along the {name} direction: "
            f"predicted {predicted:.10e} vs fd {finite_difference:.10e} "
            f"(rel {rel:.3e})"
        )


def test_complex_to_dehnen_real_scales_linearly_in_mass():
    """The conversion is mass-independent; P2M is linear in it.

    Guards the other half of the identity above: ``complex_R_solidfmm`` carries no
    mass, so the equivalence only holds because ``p2m_real_direct`` factorises as
    ``mass * U_n^m(delta)``. If that factorisation broke, the unit-mass test above
    would still pass.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("exact-identity tolerance requires float64 (JAX_ENABLE_X64=1)")

    order = 4
    d = jnp.asarray([0.35, 0.2, -0.5], dtype=jnp.float64)
    mass = 3.75

    unit = p2m_real_direct(d, jnp.asarray(1.0, dtype=jnp.float64), order=order)
    scaled = p2m_real_direct(d, jnp.asarray(mass, dtype=jnp.float64), order=order)

    np.testing.assert_allclose(
        np.asarray(scaled), mass * np.asarray(unit), rtol=1e-14, atol=1e-300
    )
