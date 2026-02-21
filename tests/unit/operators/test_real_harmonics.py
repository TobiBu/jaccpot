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

from jaccpot.operators.real_harmonics import (  # Index utilities; P2M; L2P; B matrices; Rotation; Z-axis translations;; Full operators
    compute_real_B_matrix_local,
    compute_real_B_matrix_multipole,
    evaluate_local_real,
    evaluate_local_real_with_grad,
    l2l_real,
    m2l_optimized_real,
    m2l_real,
    m2l_real_wigner,
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
    local_coeffs = jnp.array(np.random.randn(sh_size(order)))
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
        Dz = real_Dz_diagonal(ell, 0.0, dtype=dtype)
        eye = jnp.eye(2 * ell + 1)
        assert jnp.allclose(Dz, eye, atol=1e-10)


def test_rotation_z_axis_is_identity():
    """Rotation to z-axis when already aligned should be near identity.

    When the direction is already along z-axis, the rotation should be trivial
    (just a diagonal phase matrix at most).
    """
    dtype = jnp.float64

    # Direction along z-axis
    x, y, z = 0.0, 0.0, 1.0

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
    x, y, z = 1.0, 2.0, 3.0

    D = real_rotation_to_z_axis_multipole(x, y, z, 0, dtype=dtype)
    # For ell=0, D should be 1x1 identity
    assert D.shape == (1, 1)
    assert jnp.isclose(D[0, 0], 1.0, atol=1e-10)


# ===========================================================================
# Z-axis translation tests
# ===========================================================================


def test_z_m2m_identity_at_zero():
    """M2M with dz=0 should be identity."""
    order = 4
    key = jax.random.PRNGKey(0)
    multipole = jax.random.normal(key, (sh_size(order),))

    result = translate_along_z_m2m_real(multipole, jnp.array(0.0), order=order)
    assert jnp.allclose(result, multipole, atol=1e-10)


def test_z_l2l_identity_at_zero():
    """L2L with dz=0 should be identity."""
    order = 4
    key = jax.random.PRNGKey(1)
    local = jax.random.normal(key, (sh_size(order),))

    result = translate_along_z_l2l_real(local, jnp.array(0.0), order=order)
    assert jnp.allclose(result, local, atol=1e-10)


def test_z_m2l_monopole_gives_inverse_distance():
    """M2L of unit monopole at distance r gives 1/r."""
    order = 4
    multipole = jnp.zeros(sh_size(order))
    multipole = multipole.at[0].set(1.0)  # Unit monopole

    r = 2.0
    local = translate_along_z_m2l_real(multipole, jnp.array(r), order=order)

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

    errors = []
    for order in range(1, 12):
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
                f"{i+1} error {errors[i]:.2e} -> "
                f"order {i+2} error {errors[i+1]:.2e}"
            )

    # The highest order should achieve very good accuracy
    assert errors[-1] < 1e-10, f"Final error {errors[-1]:.2e} not small enough"


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
    for order in range(2, 8):
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
    """Solidfmm reference should match z-axis M2L in the real basis."""
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
    local_solidfmm = m2l_solidfmm_reference(multipole, delta_m2l, order=order)

    assert jnp.allclose(local_solidfmm, local_ref, rtol=1e-10, atol=1e-10)


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


def test_m2l_real_monopole_gives_inverse_distance():
    """M2L of unit monopole gives a 1/r local monopole term."""
    order = 4
    multipole = jnp.zeros(sh_size(order))
    multipole = multipole.at[0].set(1.0)  # Unit monopole

    delta = jnp.array([2.0, 0.0, 0.0])  # Distance 2 along x
    local = m2l_real(multipole, delta, order=order)

    # The ell=0 local coefficient should be 1/|delta| = 0.5
    r = jnp.linalg.norm(delta)
    assert jnp.isclose(local[0], 1.0 / r, atol=1e-10)


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
        D_inv = real_rotation_from_z_axis_multipole(x, y, z, ell, dtype=dtype)
        multipole_rot = multipole_rot.at[sl].set(D_inv @ multipole[sl])

    local_z = translate_along_z_m2l_real(multipole_rot, jnp.asarray(R), order=order)

    local_manual = jnp.zeros_like(local_z)
    for ell in range(order + 1):
        sl = slice(ell * ell, (ell + 1) * (ell + 1))
        D_fwd = real_rotation_to_z_axis_local(x, y, z, ell, dtype=dtype)
        local_manual = local_manual.at[sl].set(D_fwd @ local_z[sl])

    assert jnp.allclose(local_fast, local_manual, rtol=1e-10, atol=1e-10)


@pytest.mark.xfail(
    reason="Off-axis M2L rotation basis mismatch; Wigner baseline not yet aligned with real basis.",
)
def test_m2l_rotated_error_improves_with_order():
    """Off-axis M2L should improve with expansion order (Wigner baseline)."""
    pytest.importorskip("sympy")
    dtype = jnp.float64
    source_pos = jnp.array([0.0, 0.0, 0.0], dtype=dtype)
    local_center = jnp.array([3.0, 2.0, 4.0], dtype=dtype)
    eval_offset = jnp.array([0.2, -0.1, 0.15], dtype=dtype)

    eval_point = local_center + eval_offset
    r = jnp.linalg.norm(eval_point - source_pos)
    potential_direct = 1.0 / r

    orders = [2, 3, 4, 5, 6]
    errors = []
    for order in orders:
        multipole = p2m_real_direct(
            source_pos, jnp.asarray(1.0, dtype=dtype), order=order
        )
        delta_m2l = local_center - source_pos
        local = m2l_real_wigner(multipole, delta_m2l, order=order)
        delta_l2p = local_center - eval_point
        potential_fmm = evaluate_local_real(local, delta_l2p, order=order)
        err = float(
            jnp.abs(potential_fmm - potential_direct) / jnp.abs(potential_direct)
        )
        errors.append(err)

    # Expect meaningful improvement with order on off-axis path.
    assert errors[-1] < errors[0] * 0.4


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
