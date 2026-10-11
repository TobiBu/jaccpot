"""Numerical correctness checks against direct summation.

Since the 2026-10 cleanup (Z-a) this also holds, from ``test_fmm.py``, the
engine's reference expansions (``compute_expansion`` / ``evaluate_expansion``),
the physics checks (inverse square, potential, softening, zero mass) and the tree
FMM's accuracy against the direct sum.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.geometry import compute_tree_geometry
from yggdrax.interactions import DualTreeTraversalConfig, build_leaf_neighbor_lists
from yggdrax.tree import build_tree

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, RuntimePolicyConfig
from jaccpot.runtime.fmm import (
    FMMEngine,
    compute_gravitational_acceleration,
    compute_gravitational_potential,
)


def _direct_sum_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    G: float,
    softening: float,
) -> np.ndarray:
    """Reference O(N^2) accelerations with self-interaction removed."""
    n = int(positions.shape[0])
    out = np.zeros_like(positions)
    eps = np.finfo(positions.dtype).eps
    soft_sq = softening * softening

    for i in range(n):
        delta = positions[i] - positions
        dist_sq = np.sum(delta * delta, axis=1) + soft_sq
        dist = np.sqrt(dist_sq)
        inv_dist3 = 1.0 / (dist_sq * dist + eps)
        inv_dist3[i] = 0.0
        out[i] = -G * np.sum((masses[:, None] * inv_dist3[:, None]) * delta, axis=0)

    return out


@pytest.mark.parametrize(
    ("dtype", "rel_tol", "abs_tol"),
    [
        pytest.param(jnp.float32, 9.0e-2, 3.0e-1, id="float32"),
        pytest.param(jnp.float64, 2.0e-2, 6.0e-2, id="float64"),
    ],
)
def test_fmm_acceleration_matches_direct_sum(dtype, rel_tol: float, abs_tol: float):
    """FMM accelerations should agree with direct sum for small systems."""
    if dtype == jnp.float64 and not jax.config.jax_enable_x64:
        pytest.skip("float64 disabled in this JAX runtime")

    n = 128
    key = jax.random.PRNGKey(7)
    key_pos, key_mass = jax.random.split(key)

    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        dtype=dtype,
        minval=jnp.asarray(-1.0, dtype=dtype),
        maxval=jnp.asarray(1.0, dtype=dtype),
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )

    g_const = 1.0
    softening = 1.0e-2

    fmm = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=0.5,
        G=g_const,
        softening=softening,
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(
                traversal_config=DualTreeTraversalConfig(
                    max_pair_queue=131072,
                    process_block=512,
                    max_interactions_per_node=65536,
                    max_neighbors_per_leaf=65536,
                )
            )
        ),
        softening_kernel="plummer",
    )

    acc_fmm = np.asarray(
        fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=8,
            max_order=4,
        )
    )
    acc_ref = _direct_sum_accelerations(
        np.asarray(positions),
        np.asarray(masses),
        G=g_const,
        softening=softening,
    )

    diff = acc_fmm - acc_ref
    rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(acc_ref) + 1.0e-12)
    max_abs = float(np.max(np.abs(diff)))

    assert rel_l2 < rel_tol
    assert max_abs < abs_tol


# ---------------------------------------------------------------------------
# From tests/integration/test_fmm.py (2026-10 cleanup, Z-a): the engine's reference expansions and the physics
# ---------------------------------------------------------------------------


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
    exp0 = FMMEngine.compute_expansion(positions, masses, order=0)
    assert jnp.isclose(exp0.monopole, 3.0)
    assert jnp.allclose(exp0.dipole, jnp.zeros(3))
    assert jnp.allclose(exp0.quadrupole, jnp.zeros((3, 3)))

    # Order 1: dipole enabled, quad still zero
    exp1 = FMMEngine.compute_expansion(positions, masses, order=1)
    assert jnp.isclose(exp1.monopole, exp0.monopole)
    assert jnp.all(jnp.isfinite(exp1.dipole))
    assert jnp.allclose(exp1.quadrupole, jnp.zeros((3, 3)))

    # Order 2: quadrupole enabled
    exp2 = FMMEngine.compute_expansion(positions, masses, order=2)
    assert jnp.isclose(exp2.monopole, exp0.monopole)
    assert jnp.all(jnp.isfinite(exp2.dipole))
    assert jnp.all(jnp.isfinite(exp2.quadrupole))


def test_evaluate_expansion_consistency():
    """evaluate_expansion gives consistent results across orders."""
    fmm = FMMEngine(G=1.0, softening=0.0, softening_kernel="plummer")
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 2.0])
    point = jnp.array([2.0, 0.5, -1.0])

    exp0 = FMMEngine.compute_expansion(positions, masses, order=0)
    exp2 = FMMEngine.compute_expansion(positions, masses, order=2)

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

    fmm = FMMEngine(G=1.0, softening=0.0, softening_kernel="plummer")
    eval_point = jnp.array([3.0, 0.5, -1.0])

    # Reference direct sum at eval point
    a_ref = fmm.direct_sum(pos, mass, eval_point)

    # Expansions around CoM
    exp0 = FMMEngine.compute_expansion(pos, mass, order=0)
    exp1 = FMMEngine.compute_expansion(pos, mass, order=1)
    exp2 = FMMEngine.compute_expansion(pos, mass, order=2)

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

    exp = FMMEngine.compute_expansion(positions, masses, order=2)
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

    expansion = FMMEngine.compute_expansion(
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
        positions, masses, G=G, softening=softening, softening_kernel="plummer"
    )

    # First particle should be accelerated in +x direction
    # Second particle should be accelerated in -x direction
    assert accelerations[0, 0] > 0  # +x
    assert accelerations[1, 0] < 0  # -x


def test_acceleration_magnitude():
    """Test that acceleration magnitude follows inverse square law."""
    # Single particle at origin, evaluate at different distances
    positions = jnp.array([[0.0, 0.0, 0.0]])
    masses = jnp.array([1.0])

    fmm = FMMEngine(G=1.0, softening=0.0, softening_kernel="plummer")

    # Points at distance 1 and 2
    point1 = jnp.array([1.0, 0.0, 0.0])
    point2 = jnp.array([2.0, 0.0, 0.0])

    expansion = FMMEngine.compute_expansion(
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
        positions,
        masses,
        eval_points,
        G=G,
        softening=softening,
        softening_kernel="plummer",
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
        positions, masses, G=1.0, softening=0.1, softening_kernel="plummer"
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
        positions, masses, G=1.0, softening=0.0, softening_kernel="plummer"
    )

    # Should not have NaN or Inf
    assert jnp.all(jnp.isfinite(accelerations))

    # Zero mass particle should experience acceleration
    # but should not contribute to acceleration of other particle
    assert jnp.abs(accelerations[1, 0]) > 0  # Experiences acceleration


def test_far_field_accuracy_order3_vs_order4():
    """Order 4 should be at least as accurate as order 3 in far field."""
    key = jax.random.PRNGKey(2)
    n = 30
    pos = 0.1 * jax.random.normal(key, (n, 3))
    mass = jax.random.uniform(key, (n,), minval=0.5, maxval=1.5)

    fmm = FMMEngine(G=1.0, softening=0.0, softening_kernel="plummer")
    eval_point = jnp.array([6.0, -3.0, 2.0])

    a_ref = fmm.direct_sum(pos, mass, eval_point)

    exp3 = FMMEngine.compute_expansion(pos, mass, order=3)
    exp4 = FMMEngine.compute_expansion(pos, mass, order=4)

    a3 = fmm.evaluate_expansion(exp3, order=3, eval_point=eval_point)
    a4 = fmm.evaluate_expansion(exp4, order=4, eval_point=eval_point)

    e3 = jnp.linalg.norm(a3 - a_ref)
    e4 = jnp.linalg.norm(a4 - a_ref)

    assert e4 <= e3 + 1e-6


# ---------------------------------------------------------------------------
# From tests/integration/test_fmm.py (2026-10 cleanup, Z-a): the tree FMM against the direct sum
# ---------------------------------------------------------------------------


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

    fmm = FMMEngine(theta=5.0, G=1.3, softening=0.05, softening_kernel="plummer")
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

    fmm = FMMEngine(theta=0.4, G=1.0, softening=0.01, softening_kernel="plummer")
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

    fmm = FMMEngine(theta=5.0, G=1.3, softening=0.05, softening_kernel="plummer")
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
        softening_kernel="plummer",
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

    fmm = FMMEngine(theta=0.7, G=1.0, softening=0.02, softening_kernel="plummer")
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


def test_solidfmm_dehnen_accuracy_improves_with_order():
    """Regression: solidfmm+dehnen should improve strongly with expansion order."""
    with jax.enable_x64(True):
        num_particles = 224
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
        for order in (1, 2, 4):
            fmm = FMMEngine(
                theta=0.9,
                softening=softening,
                working_dtype=jnp.float64,
                runtime_policy=RuntimePolicyConfig(traversal_config=traversal),
                expansion_basis="solidfmm",
                mac_type="dehnen",
                fixed_order=order,
                fixed_max_leaf_size=16,
                softening_kernel="plummer",
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

    assert errors[0] > errors[1] > errors[2]
    # Keep a strong-margin guard against accidental convention/sign regressions.
    assert errors[0] / errors[2] > 12.0
