"""Tests for Fast Multipole Method."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaccpot.runtime.fmm as fmm_module
import yggdrasil.interactions as tree_interactions_module
from jaccpot import FMMPreset
from yggdrasil.dtypes import INDEX_DTYPE
from jaccpot.runtime.fmm import (
    FastMultipoleMethod,
    compute_gravitational_acceleration,
    compute_gravitational_potential,
)
from jaccpot.downward.local_expansions import TreeDownwardData
from jaccpot.downward.local_expansions import (
    prepare_downward_sweep as prepare_local_downward_sweep,
)
from jaccpot.downward.local_expansions import run_downward_sweep as run_local_downward_sweep
from jaccpot.nearfield.near_field import (
    compute_leaf_p2p_accelerations,
    prepare_bucketed_scatter_schedules,
    prepare_leaf_neighbor_pairs,
)
from yggdrasil.tree import build_tree
from yggdrasil.geometry import compute_tree_geometry
from yggdrasil.interactions import DualTreeTraversalConfig, build_leaf_neighbor_lists

DEFAULT_TEST_LEAF_SIZE = 1


def _direct_sum(positions, masses, *, G: float, softening: float):
    positions_np = np.asarray(positions)
    masses_np = np.asarray(masses)
    n = positions_np.shape[0]
    accelerations = np.zeros_like(positions_np)
    potentials = np.zeros((n,), dtype=positions_np.dtype)
    eps = np.finfo(positions_np.dtype).eps
    soft_sq = softening**2

    for i in range(n):
        diff = positions_np[i] - positions_np
        dist_sq = np.sum(diff * diff, axis=1) + soft_sq
        dist = np.sqrt(dist_sq)
        denom = dist_sq * dist + eps
        inv_dist3 = 1.0 / denom
        inv_dist3[i] = 0.0
        weighted = masses_np[:, None] * inv_dist3[:, None] * diff
        accelerations[i] = -G * np.sum(weighted, axis=0)

        inv_r = 1.0 / (dist + eps)
        inv_r[i] = 0.0
        potentials[i] = -G * np.sum(masses_np * inv_r)

    return accelerations, potentials


def test_compute_expansion_orders():
    """Multipole coefficients reflect requested order."""
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([2.0, 1.0])

    # Order 0: only monopole, dipole/quadrupole zero
    exp0 = FastMultipoleMethod.compute_expansion(positions, masses, order=0)
    assert jnp.isclose(exp0.monopole, 3.0)
    assert jnp.allclose(exp0.dipole, jnp.zeros(3))
    assert jnp.allclose(exp0.quadrupole, jnp.zeros((3, 3)))

    # Order 1: dipole enabled, quad still zero
    exp1 = FastMultipoleMethod.compute_expansion(positions, masses, order=1)
    assert jnp.isclose(exp1.monopole, exp0.monopole)
    assert jnp.all(jnp.isfinite(exp1.dipole))
    assert jnp.allclose(exp1.quadrupole, jnp.zeros((3, 3)))

    # Order 2: quadrupole enabled
    exp2 = FastMultipoleMethod.compute_expansion(positions, masses, order=2)
    assert jnp.isclose(exp2.monopole, exp0.monopole)
    assert jnp.all(jnp.isfinite(exp2.dipole))
    assert jnp.all(jnp.isfinite(exp2.quadrupole))


def test_evaluate_expansion_consistency():
    """evaluate_expansion gives consistent results across orders."""
    fmm = FastMultipoleMethod(G=1.0, softening=0.0)
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 2.0])
    point = jnp.array([2.0, 0.5, -1.0])

    exp0 = FastMultipoleMethod.compute_expansion(positions, masses, order=0)
    exp2 = FastMultipoleMethod.compute_expansion(positions, masses, order=2)

    # Monopole evaluation should match regardless of expansion order
    a0 = fmm.evaluate_expansion(exp0, order=0, eval_point=point)
    a0_ref = fmm.evaluate_expansion(exp2, order=0, eval_point=point)
    assert jnp.allclose(a0, a0_ref)

    # Order-2 evaluation should be finite
    a2 = fmm.evaluate_expansion(exp2, order=2, eval_point=point)
    assert jnp.all(jnp.isfinite(a2))


def test_multipole_accuracy_improves_with_order():
    """Higher order multipoles should not worsen far-field accuracy."""
    key = jax.random.PRNGKey(0)
    n = 20
    # Cluster within small radius around origin
    pos = 0.1 * jax.random.normal(key, (n, 3))
    mass = jnp.ones((n,))

    fmm = FastMultipoleMethod(G=1.0, softening=0.0)
    eval_point = jnp.array([3.0, 0.5, -1.0])

    # Reference direct sum at eval point
    a_ref = fmm.direct_sum(pos, mass, eval_point)

    # Expansions around CoM
    exp0 = FastMultipoleMethod.compute_expansion(pos, mass, order=0)
    exp1 = FastMultipoleMethod.compute_expansion(pos, mass, order=1)
    exp2 = FastMultipoleMethod.compute_expansion(pos, mass, order=2)

    a0 = fmm.evaluate_expansion(exp0, order=0, eval_point=eval_point)
    a1 = fmm.evaluate_expansion(exp1, order=1, eval_point=eval_point)
    a2 = fmm.evaluate_expansion(exp2, order=2, eval_point=eval_point)

    def err(a):
        return jnp.linalg.norm(a - a_ref)

    e0 = err(a0)
    e1 = err(a1)
    e2 = err(a2)

    # Non-increasing error with order
    assert e0 >= e1 - 1e-7
    assert e1 >= e2 - 1e-7


def test_zero_total_mass_expansion():
    """Zero total mass yields zero multipole moments (finite center)."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    masses = jnp.array([0.0, 0.0])

    exp = FastMultipoleMethod.compute_expansion(positions, masses, order=2)
    assert jnp.isclose(exp.monopole, 0.0)
    assert jnp.allclose(exp.dipole, 0.0)
    assert jnp.allclose(exp.quadrupole, 0.0)
    # Center should be finite numbers
    assert jnp.all(jnp.isfinite(exp.center))


def test_monopole_expansion():
    """Test monopole expansion calculation."""
    # Two particles with known center of mass
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 1.0])

    expansion = FastMultipoleMethod.compute_expansion(
        positions,
        masses,
        order=0,
    )

    # Total mass should be 2.0
    assert jnp.isclose(expansion.monopole, 2.0)

    # Center of mass should be at (0.5, 0.0, 0.0)
    assert jnp.allclose(expansion.center, jnp.array([0.5, 0.0, 0.0]))


def test_direct_acceleration():
    """Test direct summation of gravitational acceleration."""
    # Two particles on x-axis
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 1.0])

    G = 1.0
    softening = 0.0

    # Compute acceleration
    accelerations = compute_gravitational_acceleration(
        positions, masses, G=G, softening=softening
    )

    # First particle should be accelerated in +x direction
    # Second particle should be accelerated in -x direction
    assert accelerations[0, 0] > 0  # +x
    assert accelerations[1, 0] < 0  # -x


def test_prepare_state_fixed_depth_tree():
    n = 64
    positions = jnp.stack(
        [jnp.linspace(-1.0, 1.0, n), jnp.zeros((n,)), jnp.zeros((n,))],
        axis=1,
    )
    masses = jnp.ones((n,))
    fmm = FastMultipoleMethod(
        theta=0.6,
        tree_build_mode="fixed_depth",
        target_leaf_particles=8,
    )

    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=1,
        jit_tree=False,
    )

    assert state.tree.num_particles == n

    leaf_ranges = state.tree.node_ranges[state.tree.num_internal_nodes :]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    assert state.max_leaf_size == int(jnp.max(counts))


def _fixed_depth_sample():
    positions = jnp.array(
        [
            [-0.6, -0.2, 0.1],
            [-0.1, 0.4, -0.3],
            [0.3, -0.5, 0.2],
            [0.7, 0.1, -0.4],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 0.8, 1.2, 0.9], dtype=jnp.float64)
    return positions, masses


def _line_cluster_sample():
    xs = jnp.linspace(-0.9, 0.9, 16)
    positions = jnp.stack(
        [xs, jnp.zeros_like(xs), jnp.zeros_like(xs)],
        axis=1,
        dtype=jnp.float64,
    )
    masses = jnp.ones((xs.shape[0],), dtype=jnp.float64)
    return positions, masses


def test_compute_accelerations_fixed_depth_matches_direct():
    positions, masses = _fixed_depth_sample()
    theta = 0.7
    G = 1.1
    softening = 0.02

    fmm = FastMultipoleMethod(
        theta=theta,
        G=G,
        softening=softening,
        tree_build_mode="fixed_depth",
        target_leaf_particles=2,
    )
    acc, pot = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=2,
        return_potential=True,
        jit_tree=False,
        jit_traversal=False,
    )

    direct_acc, direct_pot = _direct_sum(
        np.asarray(positions),
        np.asarray(masses),
        G=G,
        softening=softening,
    )

    assert np.allclose(np.asarray(acc), direct_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(pot), direct_pot, rtol=1e-6, atol=1e-6)


def test_compute_accelerations_refined_tree_matches_non_refined():
    """Ensure both refine_local modes stay accurate when leaves retain
    multiple particles."""
    positions, masses = _line_cluster_sample()
    theta = 0.7
    G = 1.1
    softening = 0.02
    leaf_size = 8
    target_leaf_particles = 8
    max_refine_levels = 1
    aspect_threshold = 4.0

    def run(refine_local_flag: bool):
        fmm = FastMultipoleMethod(
            theta=theta,
            G=G,
            softening=softening,
            tree_build_mode="fixed_depth",
            target_leaf_particles=target_leaf_particles,
        )
        acc, pot = fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=leaf_size,
            return_potential=True,
            refine_local=refine_local_flag,
            jit_tree=False,
            jit_traversal=False,
            max_refine_levels=max_refine_levels,
            aspect_threshold=aspect_threshold,
        )
        return np.asarray(acc), np.asarray(pot)

    acc_no, pot_no = run(False)
    acc_ref, pot_ref = run(True)
    direct_acc, direct_pot = _direct_sum(
        np.asarray(positions),
        np.asarray(masses),
        G=G,
        softening=softening,
    )

    assert np.allclose(acc_ref, direct_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(pot_ref, direct_pot, rtol=1e-6, atol=1e-6)
    assert np.allclose(acc_no, direct_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(pot_no, direct_pot, rtol=1e-6, atol=1e-6)


def test_compute_accelerations_fixed_depth_jitted_matches_eager():
    positions, masses = _fixed_depth_sample()

    def run(jit_tree: bool, jit_traversal: bool):
        fmm = FastMultipoleMethod(
            theta=0.7,
            G=1.1,
            softening=0.02,
            tree_build_mode="fixed_depth",
            target_leaf_particles=2,
        )
        acc, pot = fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=2,
            return_potential=True,
            jit_tree=jit_tree,
            jit_traversal=jit_traversal,
        )
        return np.asarray(acc), np.asarray(pot)

    eager_acc, eager_pot = run(False, False)
    jit_acc, jit_pot = run(True, True)

    assert np.allclose(eager_acc, jit_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(eager_pot, jit_pot, rtol=1e-6, atol=1e-6)


def test_acceleration_magnitude():
    """Test that acceleration magnitude follows inverse square law."""
    # Single particle at origin, evaluate at different distances
    positions = jnp.array([[0.0, 0.0, 0.0]])
    masses = jnp.array([1.0])

    fmm = FastMultipoleMethod(G=1.0, softening=0.0)

    # Points at distance 1 and 2
    point1 = jnp.array([1.0, 0.0, 0.0])
    point2 = jnp.array([2.0, 0.0, 0.0])

    expansion = FastMultipoleMethod.compute_expansion(
        positions,
        masses,
        order=0,
    )
    accel1 = fmm.evaluate_expansion(expansion, order=0, eval_point=point1)
    accel2 = fmm.evaluate_expansion(expansion, order=0, eval_point=point2)

    mag1 = jnp.sqrt(jnp.sum(accel1**2))
    mag2 = jnp.sqrt(jnp.sum(accel2**2))

    # Acceleration at distance 2 should be 1/4 of acceleration at distance 1
    assert jnp.isclose(mag2, mag1 / 4.0, rtol=1e-5)


def test_gravitational_potential():
    """Test gravitational potential calculation."""
    # Single particle at origin
    positions = jnp.array([[0.0, 0.0, 0.0]])
    masses = jnp.array([1.0])

    # Evaluate potential at distance 1
    eval_points = jnp.array([[1.0, 0.0, 0.0]])

    G = 1.0
    softening = 0.0

    potential = compute_gravitational_potential(
        positions, masses, eval_points, G=G, softening=softening
    )

    # Potential should be -G*M/r = -1.0
    assert jnp.isclose(potential[0], -1.0, rtol=1e-5)


def test_softening():
    """Test that softening prevents singularities."""
    # Particle very close to itself (should not diverge with softening)
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1e-10, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 1.0])

    # With softening
    accelerations = compute_gravitational_acceleration(
        positions, masses, G=1.0, softening=0.1
    )

    # Should not have NaN or Inf
    assert jnp.all(jnp.isfinite(accelerations))

    # Acceleration magnitudes should be reasonable
    mags = jnp.sqrt(jnp.sum(accelerations**2, axis=1))
    assert jnp.all(mags < 1000.0)


def test_zero_mass():
    """Test handling of zero mass particles."""
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 0.0])

    accelerations = compute_gravitational_acceleration(
        positions, masses, G=1.0, softening=0.0
    )

    # Should not have NaN or Inf
    assert jnp.all(jnp.isfinite(accelerations))

    # Zero mass particle should experience acceleration
    # but should not contribute to acceleration of other particle
    assert jnp.abs(accelerations[1, 0]) > 0  # Experiences acceleration


def test_stf_symmetry_and_trace_free():
    """Quadrupole, octupole, hexadecapole are symmetric and trace-free."""
    key = jax.random.PRNGKey(1)
    n = 10
    pos = 0.2 * jax.random.normal(key, (n, 3))
    mass = jax.random.uniform(key, (n,), minval=0.1, maxval=2.0)

    exp = FastMultipoleMethod.compute_expansion(pos, mass, order=4)

    # Quadrupole symmetry and trace-free
    Q = exp.quadrupole
    assert jnp.allclose(Q, jnp.swapaxes(Q, 0, 1), atol=1e-6)
    tr_Q = jnp.trace(Q)
    assert jnp.allclose(tr_Q, 0.0, atol=1e-5)

    # Octupole: symmetric under pairwise swaps, single traces vanish
    oct_t = exp.octupole
    assert jnp.allclose(oct_t, jnp.transpose(oct_t, (1, 0, 2)), atol=1e-6)
    assert jnp.allclose(oct_t, jnp.transpose(oct_t, (2, 1, 0)), atol=1e-6)
    # single traces
    tr1 = jnp.trace(oct_t, axis1=0, axis2=1)  # shape (3,)
    tr2 = jnp.trace(oct_t, axis1=0, axis2=2)  # shape (3,)
    tr3 = jnp.trace(oct_t, axis1=1, axis2=2)  # shape (3,)
    assert jnp.allclose(tr1, 0.0, atol=1e-5)
    assert jnp.allclose(tr2, 0.0, atol=1e-5)
    assert jnp.allclose(tr3, 0.0, atol=1e-5)

    # Hexadecapole: symmetric under swaps, single traces vanish
    hex_t = exp.hexadecapole
    assert jnp.allclose(hex_t, jnp.transpose(hex_t, (1, 0, 2, 3)), atol=1e-6)
    assert jnp.allclose(hex_t, jnp.transpose(hex_t, (0, 2, 1, 3)), atol=1e-6)
    assert jnp.allclose(hex_t, jnp.transpose(hex_t, (0, 1, 3, 2)), atol=1e-6)
    # single traces along different pairs
    tr_01 = jnp.trace(hex_t, axis1=0, axis2=1)  # (3,3)
    tr_02 = jnp.trace(hex_t, axis1=0, axis2=2)  # (3,3)
    tr_03 = jnp.trace(hex_t, axis1=0, axis2=3)  # (3,3)
    assert jnp.allclose(tr_01, 0.0, atol=1e-4)
    assert jnp.allclose(tr_02, 0.0, atol=1e-4)
    assert jnp.allclose(tr_03, 0.0, atol=1e-4)
    # double trace ~ 0
    dt = jnp.trace(tr_01)
    assert jnp.allclose(dt, 0.0, atol=1e-4)


def test_prepare_downward_sweep_matches_module_helper():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.9, -0.1, 0.05],
            [0.7, 0.0, -0.05],
            [0.9, -0.1, 0.1],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 0.8, 1.2, 0.9], dtype=jnp.float64)
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    fmm = FastMultipoleMethod(theta=0.4)
    upward = fmm.prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=2,
    )

    via_method = fmm.prepare_downward_sweep(tree, upward)
    via_module = prepare_local_downward_sweep(
        tree,
        upward,
        theta=0.4,
    )

    assert isinstance(via_method, TreeDownwardData)
    assert jnp.array_equal(
        via_method.interactions.offsets,
        via_module.interactions.offsets,
    )
    assert jnp.array_equal(
        via_method.interactions.sources,
        via_module.interactions.sources,
    )
    assert jnp.allclose(
        via_method.locals.coefficients,
        via_module.locals.coefficients,
    )
    assert jnp.allclose(via_method.locals.centers, via_module.locals.centers)

    theta_override = 0.3
    alt_method = fmm.prepare_downward_sweep(
        tree,
        upward,
        theta=theta_override,
    )
    alt_module = prepare_local_downward_sweep(
        tree,
        upward,
        theta=theta_override,
    )
    assert jnp.array_equal(
        alt_method.interactions.offsets,
        alt_module.interactions.offsets,
    )
    assert jnp.array_equal(
        alt_method.interactions.sources,
        alt_module.interactions.sources,
    )

    run_result = fmm.run_downward_sweep(
        tree,
        upward.multipoles,
        via_module.interactions,
    )
    module_result = run_local_downward_sweep(
        tree,
        upward.multipoles,
        via_module.interactions,
    )
    assert jnp.allclose(run_result.coefficients, module_result.coefficients)
    assert jnp.allclose(run_result.centers, module_result.centers)


def test_fmm_dense_downward_matches_sparse_path():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.4, -0.2, 0.25],
            [0.2, 0.3, -0.1],
            [0.6, -0.15, 0.15],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 0.7, 1.3, 0.9], dtype=jnp.float64)
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )

    kwargs = dict(bounds=bounds, leaf_size=2, max_order=2, theta=0.6)
    fmm_sparse = FastMultipoleMethod(theta=0.6, use_dense_interactions=False)
    fmm_dense = FastMultipoleMethod(theta=0.6, use_dense_interactions=True)

    state_sparse = fmm_sparse.prepare_state(positions, masses, **kwargs)
    state_dense = fmm_dense.prepare_state(positions, masses, **kwargs)

    acc_sparse = fmm_sparse.evaluate_prepared_state(state_sparse)
    acc_dense = fmm_dense.evaluate_prepared_state(state_dense)

    assert jnp.allclose(acc_sparse, acc_dense, rtol=1e-10, atol=1e-10)


def test_far_field_accuracy_order3_vs_order4():
    """Order 4 should be at least as accurate as order 3 in far field."""
    key = jax.random.PRNGKey(2)
    n = 30
    pos = 0.1 * jax.random.normal(key, (n, 3))
    mass = jax.random.uniform(key, (n,), minval=0.5, maxval=1.5)

    fmm = FastMultipoleMethod(G=1.0, softening=0.0)
    eval_point = jnp.array([6.0, -3.0, 2.0])

    a_ref = fmm.direct_sum(pos, mass, eval_point)

    exp3 = FastMultipoleMethod.compute_expansion(pos, mass, order=3)
    exp4 = FastMultipoleMethod.compute_expansion(pos, mass, order=4)

    a3 = fmm.evaluate_expansion(exp3, order=3, eval_point=eval_point)
    a4 = fmm.evaluate_expansion(exp4, order=4, eval_point=eval_point)

    e3 = jnp.linalg.norm(a3 - a_ref)
    e4 = jnp.linalg.norm(a4 - a_ref)

    assert e4 <= e3 + 1e-6


def test_evaluate_tree_matches_direct_sum_all_near_field():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.6, -0.2, 0.1],
            [0.2, 0.3, -0.2],
            [0.4, -0.1, 0.2],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 1.2, 0.9, 1.1], dtype=jnp.float64)
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )

    tree, pos_sorted, mass_sorted, inv = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=2,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=5.0)

    fmm = FastMultipoleMethod(theta=5.0, G=1.3, softening=0.05)
    upward = fmm.prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=2,
    )
    downward = fmm.prepare_downward_sweep(tree, upward, theta=5.0)

    accelerations, potentials = fmm.evaluate_tree(
        tree,
        pos_sorted,
        mass_sorted,
        downward,
        neighbor_list,
        return_potential=True,
    )

    inv_idx = np.asarray(inv)
    accel_orig = np.asarray(accelerations)[inv_idx]
    pot_orig = np.asarray(potentials)[inv_idx]

    direct_acc, direct_pot = _direct_sum(
        positions,
        masses,
        G=1.3,
        softening=0.05,
    )

    assert np.allclose(accel_orig, direct_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(pot_orig, direct_pot, rtol=1e-6, atol=1e-6)


def test_evaluate_tree_far_field_accuracy():
    positions = jnp.array(
        [
            [-0.9, -0.2, 0.1],
            [-0.7, 0.1, -0.1],
            [-0.6, 0.3, 0.05],
            [0.6, -0.1, -0.2],
            [0.75, 0.2, 0.0],
            [0.9, -0.05, 0.2],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 0.8, 1.1, 0.9, 1.2, 0.7], dtype=jnp.float64)
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )

    tree, pos_sorted, mass_sorted, inv = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=2,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.4)

    fmm = FastMultipoleMethod(theta=0.4, G=1.0, softening=0.01)
    upward = fmm.prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=2,
    )
    downward = fmm.prepare_downward_sweep(tree, upward, theta=0.4)

    max_leaf_size = int(
        np.max(
            np.asarray(
                tree.node_ranges[neighbor_list.leaf_indices, 1]
                - tree.node_ranges[neighbor_list.leaf_indices, 0]
                + 1,
                dtype=np.int64,
            )
        )
    )

    accelerations, potentials = fmm.evaluate_tree(
        tree,
        pos_sorted,
        mass_sorted,
        downward,
        neighbor_list,
        max_leaf_size=max_leaf_size,
        return_potential=True,
    )

    inv_idx = np.asarray(inv)
    accel_orig = np.asarray(accelerations)[inv_idx]
    pot_orig = np.asarray(potentials)[inv_idx]

    direct_acc, direct_pot = _direct_sum(
        positions,
        masses,
        G=1.0,
        softening=0.01,
    )

    assert np.allclose(accel_orig, direct_acc, rtol=5e-3, atol=5e-3)
    assert np.allclose(pot_orig, direct_pot, rtol=5e-3, atol=5e-3)


def test_fmm_pipeline_matches_direct_sum():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.6, -0.2, 0.1],
            [0.2, 0.3, -0.2],
            [0.4, -0.1, 0.2],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 1.2, 0.9, 1.1], dtype=jnp.float64)

    fmm = FastMultipoleMethod(theta=5.0, G=1.3, softening=0.05)
    acc_class, pot_class = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=2,
        return_potential=True,
    )

    acc_func, pot_func = compute_gravitational_acceleration(
        positions,
        masses,
        theta=5.0,
        G=1.3,
        softening=0.05,
        leaf_size=2,
        return_potential=True,
    )

    direct_acc, direct_pot = _direct_sum(
        np.asarray(positions),
        np.asarray(masses),
        G=1.3,
        softening=0.05,
    )

    acc_class_np = np.asarray(acc_class)
    pot_class_np = np.asarray(pot_class)
    acc_func_np = np.asarray(acc_func)
    pot_func_np = np.asarray(pot_func)

    assert np.allclose(acc_class_np, direct_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(pot_class_np, direct_pot, rtol=1e-6, atol=1e-6)
    assert np.allclose(acc_func_np, direct_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(pot_func_np, direct_pot, rtol=1e-6, atol=1e-6)


def test_evaluate_tree_compiled_matches_eager():
    positions = jnp.array(
        [
            [0.3, -0.2, 0.1],
            [-0.4, 0.5, -0.6],
            [0.7, 0.1, -0.3],
            [-0.2, -0.4, 0.8],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.1, 0.9, 1.3, 0.8], dtype=jnp.float64)

    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=2,
        return_reordered=True,
    )

    fmm = FastMultipoleMethod(theta=0.7, G=1.0, softening=0.02)
    upward = fmm.prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=2,
    )
    downward = fmm.prepare_downward_sweep(
        tree,
        upward,
        theta=fmm.theta,
    )
    neighbor_list = build_leaf_neighbor_lists(
        tree,
        upward.geometry,
        theta=fmm.theta,
    )

    kwargs = dict(max_leaf_size=2, return_potential=True)

    eager_acc, eager_pot = fmm.evaluate_tree(
        tree,
        pos_sorted,
        mass_sorted,
        downward,
        neighbor_list,
        **kwargs,
    )
    jit_acc, jit_pot = fmm.evaluate_tree_compiled(
        tree,
        pos_sorted,
        mass_sorted,
        downward,
        neighbor_list,
        **kwargs,
    )

    assert np.allclose(np.asarray(eager_acc), np.asarray(jit_acc))
    assert np.allclose(np.asarray(eager_pot), np.asarray(jit_pot))


def test_nearfield_bucketed_matches_baseline():
    key = jax.random.PRNGKey(515)
    num_particles = 128
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    base_kwargs = dict(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
        fixed_order=4,
        fixed_max_leaf_size=16,
        grouped_interactions=True,
        farfield_mode="class_major",
    )

    fmm_baseline = FastMultipoleMethod(
        nearfield_mode="baseline",
        **base_kwargs,
    )
    fmm_bucketed = FastMultipoleMethod(
        nearfield_mode="bucketed",
        nearfield_edge_chunk_size=128,
        **base_kwargs,
    )

    acc_baseline = fmm_baseline.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
        jit_tree=False,
    )
    acc_bucketed = fmm_bucketed.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
        jit_tree=False,
    )

    assert np.allclose(np.asarray(acc_bucketed), np.asarray(acc_baseline), rtol=1e-5, atol=1e-5)


def test_nearfield_precomputed_leaf_pairs_matches_inline_mapping():
    key = jax.random.PRNGKey(516)
    num_particles = 96
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbors = build_leaf_neighbor_lists(tree, geometry, theta=0.6)
    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    leaf_nodes = jnp.asarray(neighbors.leaf_indices, dtype=INDEX_DTYPE)
    offsets = jnp.asarray(neighbors.offsets, dtype=INDEX_DTYPE)
    neighbor_ids = jnp.asarray(neighbors.neighbors, dtype=INDEX_DTYPE)
    tgt_leaf, src_leaf, valid = prepare_leaf_neighbor_pairs(
        node_ranges,
        leaf_nodes,
        offsets,
        neighbor_ids,
    )

    acc_inline = compute_leaf_p2p_accelerations(
        tree,
        neighbors,
        pos_sorted,
        mass_sorted,
        softening=1e-3,
        max_leaf_size=16,
        nearfield_mode="bucketed",
        edge_chunk_size=64,
    )
    acc_precomputed = compute_leaf_p2p_accelerations(
        tree,
        neighbors,
        pos_sorted,
        mass_sorted,
        softening=1e-3,
        max_leaf_size=16,
        nearfield_mode="bucketed",
        edge_chunk_size=64,
        precomputed_target_leaf_ids=tgt_leaf,
        precomputed_source_leaf_ids=src_leaf,
        precomputed_valid_pairs=valid,
    )

    assert np.allclose(np.asarray(acc_precomputed), np.asarray(acc_inline), rtol=1e-6, atol=1e-6)


def test_nearfield_precomputed_bucketed_scatter_matches_inline():
    key = jax.random.PRNGKey(615)
    num_particles = 96
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbors = build_leaf_neighbor_lists(tree, geometry, theta=0.6)
    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    leaf_nodes = jnp.asarray(neighbors.leaf_indices, dtype=INDEX_DTYPE)
    offsets = jnp.asarray(neighbors.offsets, dtype=INDEX_DTYPE)
    neighbor_ids = jnp.asarray(neighbors.neighbors, dtype=INDEX_DTYPE)
    tgt_leaf, src_leaf, valid = prepare_leaf_neighbor_pairs(
        node_ranges,
        leaf_nodes,
        offsets,
        neighbor_ids,
    )
    sort_idx, group_ids, unique_indices = prepare_bucketed_scatter_schedules(
        node_ranges,
        leaf_nodes,
        tgt_leaf,
        valid,
        max_leaf_size=16,
        edge_chunk_size=64,
    )

    acc_inline = compute_leaf_p2p_accelerations(
        tree,
        neighbors,
        pos_sorted,
        mass_sorted,
        softening=1e-3,
        max_leaf_size=16,
        nearfield_mode="bucketed",
        edge_chunk_size=64,
        precomputed_target_leaf_ids=tgt_leaf,
        precomputed_source_leaf_ids=src_leaf,
        precomputed_valid_pairs=valid,
    )
    acc_precomputed = compute_leaf_p2p_accelerations(
        tree,
        neighbors,
        pos_sorted,
        mass_sorted,
        softening=1e-3,
        max_leaf_size=16,
        nearfield_mode="bucketed",
        edge_chunk_size=64,
        precomputed_target_leaf_ids=tgt_leaf,
        precomputed_source_leaf_ids=src_leaf,
        precomputed_valid_pairs=valid,
        precomputed_chunk_sort_indices=sort_idx,
        precomputed_chunk_group_ids=group_ids,
        precomputed_chunk_unique_indices=unique_indices,
    )

    assert np.allclose(np.asarray(acc_precomputed), np.asarray(acc_inline), rtol=1e-6, atol=1e-6)


def test_prepare_state_reuses_cached_interactions_when_inputs_match():
    key = jax.random.PRNGKey(123)
    num_particles = 32
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
    )
    masses = jnp.linspace(0.5, 1.5, num_particles, dtype=jnp.float32)

    fmm = FastMultipoleMethod(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
    )
    state_first = fmm.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=2,
        jit_tree=False,
    )

    with mock.patch.object(
        fmm_module,
        "build_interactions_and_neighbors",
        side_effect=AssertionError("should not rebuild interactions"),
    ):
        state_second = fmm.prepare_state(
            positions,
            masses,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
        )

    assert state_first.interactions.sources is state_second.interactions.sources
    assert jnp.array_equal(
        state_first.neighbor_list.neighbors,
        state_second.neighbor_list.neighbors,
    )


def test_compute_accelerations_reuses_prepared_state_when_enabled():
    key = jax.random.PRNGKey(211)
    num_particles = 48
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.linspace(0.75, 1.25, num_particles, dtype=jnp.float32)

    fmm = FastMultipoleMethod(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
    )
    with mock.patch.object(fmm, "prepare_state", wraps=fmm.prepare_state) as spy_prepare:
        acc_first = fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
            reuse_prepared_state=True,
        )
        acc_second = fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
            reuse_prepared_state=True,
        )

    assert spy_prepare.call_count == 1
    assert np.allclose(np.asarray(acc_second), np.asarray(acc_first), rtol=1e-6, atol=1e-6)


def test_compute_accelerations_reuse_cache_invalidates_on_parameter_change():
    key = jax.random.PRNGKey(311)
    num_particles = 40
    positions = jax.random.normal(key, (num_particles, 3), dtype=jnp.float32)
    masses = jnp.ones((num_particles,), dtype=jnp.float32)

    fmm = FastMultipoleMethod(
        theta=0.55,
        softening=1e-3,
        working_dtype=jnp.float32,
    )
    with mock.patch.object(fmm, "prepare_state", wraps=fmm.prepare_state) as spy_prepare:
        fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
            reuse_prepared_state=True,
        )
        fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=8,
            max_order=3,
            jit_tree=False,
            reuse_prepared_state=True,
        )

    assert spy_prepare.call_count == 2


def test_compute_accelerations_reuses_prepared_state_for_value_equal_copies():
    key = jax.random.PRNGKey(312)
    num_particles = 40
    positions = jax.random.normal(key, (num_particles, 3), dtype=jnp.float32)
    masses = jnp.ones((num_particles,), dtype=jnp.float32)
    positions_copy = jnp.array(np.asarray(positions))
    masses_copy = jnp.array(np.asarray(masses))

    fmm = FastMultipoleMethod(
        theta=0.55,
        softening=1e-3,
        working_dtype=jnp.float32,
    )
    with mock.patch.object(fmm, "prepare_state", wraps=fmm.prepare_state) as spy_prepare:
        acc_first = fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
            reuse_prepared_state=True,
        )
        acc_second = fmm.compute_accelerations(
            positions_copy,
            masses_copy,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
            reuse_prepared_state=True,
        )

    assert spy_prepare.call_count == 1
    assert np.allclose(np.asarray(acc_second), np.asarray(acc_first), rtol=1e-6, atol=1e-6)


def test_compute_accelerations_reuse_cache_invalidates_on_value_change():
    key = jax.random.PRNGKey(313)
    num_particles = 40
    positions = jax.random.normal(key, (num_particles, 3), dtype=jnp.float32)
    masses = jnp.ones((num_particles,), dtype=jnp.float32)
    masses_changed = masses.at[0].set(jnp.float32(1.5))

    fmm = FastMultipoleMethod(
        theta=0.55,
        softening=1e-3,
        working_dtype=jnp.float32,
    )
    with mock.patch.object(fmm, "prepare_state", wraps=fmm.prepare_state) as spy_prepare:
        fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
            reuse_prepared_state=True,
        )
        fmm.compute_accelerations(
            positions,
            masses_changed,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
            reuse_prepared_state=True,
        )

    assert spy_prepare.call_count == 2


def test_prepare_state_precomputes_bucketed_scatter_schedule():
    key = jax.random.PRNGKey(911)
    num_particles = 128
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.ones((num_particles,), dtype=jnp.float32)

    fmm = FastMultipoleMethod(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        nearfield_mode="bucketed",
        nearfield_edge_chunk_size=64,
    )
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=2,
        jit_tree=False,
    )
    assert state.nearfield_chunk_sort_indices is not None
    assert state.nearfield_chunk_group_ids is not None
    assert state.nearfield_chunk_unique_indices is not None

    acc_state = fmm.evaluate_prepared_state(state, jit_traversal=True)
    acc_full = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=2,
        jit_tree=False,
    )
    assert np.allclose(np.asarray(acc_state), np.asarray(acc_full), rtol=1e-5, atol=1e-5)


def test_prepare_state_cache_respects_theta_changes():
    key = jax.random.PRNGKey(321)
    num_particles = 24
    positions = jax.random.normal(key, (num_particles, 3), dtype=jnp.float32)
    masses = jnp.ones((num_particles,), dtype=jnp.float32)

    fmm = FastMultipoleMethod(
        theta=0.5,
        softening=1e-3,
        working_dtype=jnp.float32,
    )
    fmm.prepare_state(
        positions,
        masses,
        leaf_size=6,
        max_order=2,
        jit_tree=False,
    )

    theta_override = 0.65
    with mock.patch.object(
        fmm_module,
        "build_interactions_and_neighbors",
        wraps=fmm_module.build_interactions_and_neighbors,
    ) as spy_build:
        fmm.prepare_state(
            positions,
            masses,
            leaf_size=6,
            max_order=2,
            theta=theta_override,
            jit_tree=False,
        )

    assert spy_build.call_count == 1


def test_prepare_state_cache_respects_dehnen_radius_scale_changes():
    key = jax.random.PRNGKey(411)
    num_particles = 32
    positions = jax.random.normal(key, (num_particles, 3), dtype=jnp.float32)
    masses = jnp.ones((num_particles,), dtype=jnp.float32)

    fmm = FastMultipoleMethod(
        theta=0.55,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
        dehnen_radius_scale=1.0,
    )
    fmm.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
        jit_tree=False,
    )

    fmm.dehnen_radius_scale = 0.9
    with mock.patch.object(
        fmm_module,
        "build_interactions_and_neighbors",
        wraps=fmm_module.build_interactions_and_neighbors,
    ) as spy_build:
        fmm.prepare_state(
            positions,
            masses,
            leaf_size=8,
            max_order=3,
            jit_tree=False,
        )

    assert spy_build.call_count == 1
    assert float(spy_build.call_args.kwargs["dehnen_radius_scale"]) == pytest.approx(0.9)


def test_prepare_state_cache_respects_traversal_config_changes():
    key = jax.random.PRNGKey(654)
    num_particles = 28
    positions = jax.random.normal(key, (num_particles, 3), dtype=jnp.float32)
    masses = jnp.linspace(0.25, 1.0, num_particles, dtype=jnp.float32)

    config_a = DualTreeTraversalConfig(
        max_pair_queue=4096,
        process_block=256,
        max_interactions_per_node=4096,
    )
    config_b = DualTreeTraversalConfig(
        max_pair_queue=8192,
        process_block=512,
        max_interactions_per_node=8192,
        max_neighbors_per_leaf=4096,
    )

    fmm = FastMultipoleMethod(
        theta=0.55,
        softening=5e-4,
        working_dtype=jnp.float32,
        traversal_config=config_a,
    )
    fmm.prepare_state(
        positions,
        masses,
        leaf_size=6,
        max_order=2,
        jit_tree=False,
    )

    fmm.traversal_config = config_b
    with mock.patch.object(
        fmm_module,
        "build_interactions_and_neighbors",
        wraps=fmm_module.build_interactions_and_neighbors,
    ) as spy_build:
        fmm.prepare_state(
            positions,
            masses,
            leaf_size=6,
            max_order=2,
            jit_tree=False,
        )

    assert spy_build.call_count == 1
    assert spy_build.call_args.kwargs["traversal_config"] is config_b


def test_prepare_state_reuses_grouped_buffers_from_cache():
    key = jax.random.PRNGKey(808)
    num_particles = 64
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    fmm = FastMultipoleMethod(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
        grouped_interactions=True,
    )
    fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
        jit_tree=False,
    )

    with mock.patch.object(
        tree_interactions_module,
        "build_grouped_interactions_from_pairs",
        side_effect=AssertionError("should reuse grouped buffers from cache"),
    ):
        fmm.prepare_state(
            positions,
            masses,
            leaf_size=16,
            max_order=4,
            jit_tree=False,
        )


def test_prepare_state_reuses_grouped_class_segments_from_cache():
    key = jax.random.PRNGKey(810)
    num_particles = 96
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    fmm = FastMultipoleMethod(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
        grouped_interactions=True,
        farfield_mode="class_major",
        m2l_chunk_size=128,
    )
    fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
        jit_tree=False,
    )

    with mock.patch.object(
        fmm_module,
        "_build_grouped_class_segments",
        side_effect=AssertionError("should reuse grouped class segments from cache"),
    ):
        fmm.prepare_state(
            positions,
            masses,
            leaf_size=16,
            max_order=4,
            jit_tree=False,
        )


def test_prepare_state_cache_key_respects_center_mode():
    key = jax.random.PRNGKey(809)
    num_particles = 64
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    fmm = FastMultipoleMethod(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
        grouped_interactions=False,
    )
    fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
        jit_tree=False,
    )

    fmm.grouped_interactions = True
    fmm._explicit_grouped_interactions = True
    with mock.patch.object(
        fmm_module,
        "build_interactions_and_neighbors",
        wraps=fmm_module.build_interactions_and_neighbors,
    ) as spy_build:
        fmm.prepare_state(
            positions,
            masses,
            leaf_size=16,
            max_order=4,
            jit_tree=False,
        )

    assert spy_build.call_count == 1


def test_fast_preset_sets_fixed_depth_defaults():
    key = jax.random.PRNGKey(111)
    num_particles = 48
    positions = jax.random.normal(key, (num_particles, 3), dtype=jnp.float32)
    masses = jnp.ones((num_particles,), dtype=jnp.float32)

    fmm = FastMultipoleMethod(preset="fast", theta=0.6, softening=1e-3)

    assert fmm.tree_build_mode == "fixed_depth"
    assert fmm.target_leaf_particles == 64
    assert fmm.refine_local is False
    assert isinstance(fmm.traversal_config, DualTreeTraversalConfig)
    assert fmm.m2l_chunk_size == 512

    accelerations = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=8,
        max_order=2,
    )

    assert accelerations.shape == (num_particles, 3)


def test_fast_preset_allows_explicit_overrides():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        tree_build_mode="lbvh",
        target_leaf_particles=12,
        refine_local=True,
    )

    assert fmm.tree_build_mode == "lbvh"
    assert fmm.target_leaf_particles == 12
    assert fmm.refine_local is True
    assert isinstance(fmm.traversal_config, DualTreeTraversalConfig)


def test_fast_preset_defaults_to_auto_jit_tree_policy():
    fmm = FastMultipoleMethod(preset=FMMPreset.FAST)
    assert fmm._jit_tree_default == "auto"


def test_solidfmm_float32_uses_complex64_locals():
    key = jax.random.PRNGKey(7)
    num_particles = 64
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    fmm = FastMultipoleMethod(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
    )
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
        jit_tree=False,
    )
    assert state.downward.locals.coefficients.dtype == jnp.complex64


def test_solidfmm_float64_uses_complex128_locals():
    with jax.experimental.enable_x64():
        key = jax.random.PRNGKey(9)
        num_particles = 64
        positions = jax.random.uniform(
            key,
            (num_particles, 3),
            minval=-1.0,
            maxval=1.0,
            dtype=jnp.float64,
        )
        masses = (
            jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float64)) + 1.0
        )

        fmm = FastMultipoleMethod(
            theta=0.6,
            softening=1e-3,
            working_dtype=jnp.float64,
            expansion_basis="solidfmm",
            complex_rotation="solidfmm",
            mac_type="dehnen",
        )
        state = fmm.prepare_state(
            positions,
            masses,
            leaf_size=16,
            max_order=4,
            jit_tree=False,
        )
        assert state.downward.locals.coefficients.dtype == jnp.complex128


def test_solidfmm_chunked_m2l_matches_fullbatch():
    key = jax.random.PRNGKey(13)
    num_particles = 128
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    base_kwargs = dict(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
        tree_build_mode="lbvh",
        fixed_order=4,
        fixed_max_leaf_size=16,
    )

    full = FastMultipoleMethod(m2l_chunk_size=4096, **base_kwargs)
    chunked = FastMultipoleMethod(m2l_chunk_size=32, **base_kwargs)

    acc_full = full.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
        jit_tree=False,
    )
    acc_chunked = chunked.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
        jit_tree=False,
    )

    assert np.allclose(np.asarray(acc_chunked), np.asarray(acc_full), rtol=1e-6, atol=1e-6)


def test_fast_preset_adaptive_large_cpu_policy_applies():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
    )

    overrides = fmm._resolve_runtime_execution_overrides(
        num_particles=131072,
        backend="cpu",
    )

    assert overrides.adaptive_applied is True
    assert overrides.m2l_chunk_size == 32768
    assert overrides.traversal_config is not None
    assert overrides.traversal_config.process_block == 4096
    assert overrides.traversal_config.max_interactions_per_node == 65536
    assert overrides.grouped_interactions is True
    assert overrides.farfield_mode == "pair_grouped"
    assert overrides.center_mode == "aabb"
    assert overrides.refine_local_override is False


def test_fast_preset_adaptive_class_major_threshold():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
    )

    overrides = fmm._resolve_runtime_execution_overrides(
        num_particles=262144,
        backend="cpu",
    )

    assert overrides.grouped_interactions is True
    assert overrides.farfield_mode == "class_major"


def test_adaptive_nearfield_edge_chunk_size_auto_policy():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
        nearfield_mode="auto",
        nearfield_edge_chunk_size=256,
    )

    assert fmm._resolve_nearfield_edge_chunk_size(
        num_particles=131072,
        nearfield_mode="baseline",
    ) == 256
    assert fmm._resolve_nearfield_edge_chunk_size(
        num_particles=262144,
        nearfield_mode="bucketed",
    ) == 1024
    assert fmm._resolve_nearfield_edge_chunk_size(
        num_particles=1000000,
        nearfield_mode="bucketed",
    ) == 2048
    assert fmm._resolve_nearfield_edge_chunk_size(
        num_particles=2000000,
        nearfield_mode="bucketed",
    ) == 4096

def test_fast_preset_adaptive_policy_respects_explicit_overrides():
    cfg = DualTreeTraversalConfig(
        max_pair_queue=4096,
        process_block=256,
        max_interactions_per_node=4096,
        max_neighbors_per_leaf=4096,
    )
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
        traversal_config=cfg,
        m2l_chunk_size=2048,
    )

    overrides = fmm._resolve_runtime_execution_overrides(
        num_particles=131072,
        backend="cpu",
    )

    assert overrides.adaptive_applied is False
    assert overrides.traversal_config is cfg
    assert overrides.m2l_chunk_size == 2048


def test_solidfmm_grouped_interactions_matches_sparse_path():
    key = jax.random.PRNGKey(23)
    num_particles = 192
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    fmm = FastMultipoleMethod(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
        fixed_order=4,
    )

    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=16,
        return_reordered=True,
    )
    upward = fmm.prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=4,
        center_mode="aabb",
    )
    downward_sparse = fmm.prepare_downward_sweep(
        tree,
        upward,
        theta=0.6,
        grouped_interactions=False,
    )
    downward_grouped = fmm.prepare_downward_sweep(
        tree,
        upward,
        theta=0.6,
        interactions=downward_sparse.interactions,
        grouped_interactions=True,
    )

    assert np.allclose(
        np.asarray(downward_grouped.locals.coefficients),
        np.asarray(downward_sparse.locals.coefficients),
        rtol=1e-5,
        atol=1e-5,
    )


def test_solidfmm_grouped_class_major_matches_pair_grouped():
    key = jax.random.PRNGKey(31)
    num_particles = 160
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    fmm = FastMultipoleMethod(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
        grouped_interactions=True,
        farfield_mode="pair_grouped",
        fixed_order=4,
    )

    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=16,
        return_reordered=True,
    )
    upward = fmm.prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=4,
        center_mode="aabb",
    )
    downward_pair = fmm.prepare_downward_sweep(
        tree,
        upward,
        theta=0.6,
        grouped_interactions=True,
        farfield_mode="pair_grouped",
    )
    downward_class = fmm.prepare_downward_sweep(
        tree,
        upward,
        theta=0.6,
        interactions=downward_pair.interactions,
        grouped_interactions=True,
        farfield_mode="class_major",
    )

    assert np.allclose(
        np.asarray(downward_class.locals.coefficients),
        np.asarray(downward_pair.locals.coefficients),
        rtol=1e-5,
        atol=1e-5,
    )


def _benchmark_like_distribution(
    num_particles: int,
    *,
    key: jax.Array,
    dtype: jnp.dtype,
):
    """Match the benchmark notebook's synthetic distribution."""
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (num_particles, 3),
        dtype=dtype,
        minval=-1.0,
        maxval=1.0,
    )
    masses = jnp.abs(jax.random.normal(key_mass, (num_particles,), dtype=dtype)) + 1.0
    return positions, masses


def _direct_accelerations_vectorized(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    softening: float,
):
    diff = positions[:, None, :] - positions[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1) + softening**2
    mask = ~np.eye(positions.shape[0], dtype=bool)
    inv_r3 = np.where(mask, 1.0 / (dist_sq * np.sqrt(dist_sq)), 0.0)
    weighted = diff * masses[None, :, None] * inv_r3[..., None]
    return -np.sum(weighted, axis=1)


def test_solidfmm_basis_rejects_non_solidfmm_rotation():
    with pytest.raises(
        ValueError,
        match="expansion_basis='solidfmm' requires complex_rotation='solidfmm'",
    ):
        FastMultipoleMethod(expansion_basis="solidfmm", complex_rotation="cached")


def test_dehnen_radius_scale_must_be_positive():
    with pytest.raises(ValueError, match="dehnen_radius_scale must be > 0"):
        FastMultipoleMethod(dehnen_radius_scale=0.0)


def test_nearfield_mode_validation():
    with pytest.raises(ValueError, match="nearfield_mode must be 'auto', 'baseline', or 'bucketed'"):
        FastMultipoleMethod(nearfield_mode="unknown")
    with pytest.raises(ValueError, match="nearfield_edge_chunk_size must be positive"):
        FastMultipoleMethod(nearfield_edge_chunk_size=0)


def test_solidfmm_dehnen_accuracy_improves_with_order():
    """Regression: solidfmm+dehnen should improve strongly with expansion order."""
    with jax.experimental.enable_x64():
        num_particles = 320
        softening = 1e-3
        positions, masses = _benchmark_like_distribution(
            num_particles,
            key=jax.random.PRNGKey(2),
            dtype=jnp.float64,
        )

        reference = _direct_accelerations_vectorized(
            np.asarray(positions),
            np.asarray(masses),
            softening=softening,
        )
        ref_norm = np.linalg.norm(reference)

        traversal = DualTreeTraversalConfig(
            max_pair_queue=65536,
            process_block=512,
            max_interactions_per_node=16384,
            max_neighbors_per_leaf=8192,
        )

        errors = []
        for order in (1, 2, 4, 6):
            fmm = FastMultipoleMethod(
                theta=0.6,
                softening=softening,
                working_dtype=jnp.float64,
                traversal_config=traversal,
                expansion_basis="solidfmm",
                complex_rotation="solidfmm",
                mac_type="dehnen",
                fixed_order=order,
                fixed_max_leaf_size=16,
            )
            accelerations = np.asarray(
                fmm.compute_accelerations(
                    positions,
                    masses,
                    leaf_size=16,
                    max_order=order,
                )
            )
            rel_l2 = np.linalg.norm(accelerations - reference) / ref_norm
            errors.append(rel_l2)

    assert errors[0] > errors[1] > errors[2] > errors[3]
    # Keep a strong-margin guard against accidental convention/sign regressions.
    assert errors[0] / errors[3] > 20.0
