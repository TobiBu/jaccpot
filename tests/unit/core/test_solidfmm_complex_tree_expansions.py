"""Tests for solidfmm complex upward sweep helpers."""

import jax.numpy as jnp
import numpy as np
from yggdrax.tree import build_tree

from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_source_motion_multipoles,
    prepare_solidfmm_complex_upward_sweep,
)


def _build_sample_tree(leaf_size=2):
    """The six-particle fixture.

    ``leaf_size`` is a parameter so the same particles can be built into a tree with
    internal nodes (the default, ``2``) or into a **single leaf** (any value >= 6),
    which is the ``num_internal_nodes == 0`` branch of
    ``_prepare_solidfmm_downward_sweep``. Keeping one fixture means the two regimes
    differ only in the tree, not in the physics.
    """
    positions = jnp.array(
        [
            [-0.7, -0.4, -0.1],
            [-0.2, 0.1, 0.5],
            [0.4, -0.3, 0.2],
            [0.8, 0.6, -0.4],
            [0.1, 0.9, 0.7],
            [-0.6, 0.4, -0.8],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 0.7, 1.3, 0.8, 1.1, 0.9], dtype=jnp.float64)
    velocities = jnp.array(
        [
            [0.02, -0.01, 0.03],
            [-0.03, 0.04, -0.02],
            [0.01, 0.02, -0.01],
            [0.05, -0.02, 0.01],
            [-0.04, 0.03, 0.02],
            [0.03, 0.01, -0.04],
        ],
        dtype=jnp.float64,
    )
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=leaf_size,
    )
    vel_sorted = velocities[jnp.asarray(tree.particle_indices)]
    return tree, pos_sorted, mass_sorted, vel_sorted


def _build_multilevel_tree(n=600, leaf_size=8, seed=0):
    rng = np.random.default_rng(seed)
    pos = jnp.asarray(np.clip(rng.normal(0.0, 0.3, (n, 3)), -0.99, 0.99))
    mass = jnp.asarray(rng.uniform(0.5, 1.5, n))
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=pos.dtype),
        jnp.array([1.0, 1.0, 1.0], dtype=pos.dtype),
    )
    tree, ps, msorted, _ = build_tree(
        pos, mass, bounds, return_reordered=True, leaf_size=leaf_size
    )
    return tree, ps, msorted


def test_static_num_levels_bit_identical_to_padded():
    """Passing the concrete (unpadded) depth must be bit-identical to deriving
    the M2M level count from the padded level_offsets shape."""
    from yggdrax.tree import get_num_levels

    tree, ps, ms = _build_multilevel_tree()
    actual_num_levels = int(get_num_levels(tree))

    padded = prepare_solidfmm_complex_upward_sweep(
        tree, ps, ms, max_order=4, center_mode="com"
    )
    optimized = prepare_solidfmm_complex_upward_sweep(
        tree,
        ps,
        ms,
        max_order=4,
        center_mode="com",
        static_num_levels=actual_num_levels,
    )
    # Exact equality: same arithmetic, only the empty padded levels are skipped.
    assert jnp.array_equal(padded.multipoles.packed, optimized.multipoles.packed)


def test_prepare_solidfmm_upward_source_motion_optional_none():
    tree, pos_sorted, mass_sorted, _ = _build_sample_tree()
    upward = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=3,
        center_mode="com",
    )
    assert upward.multipoles.source_motion_packed is None


def test_prepare_solidfmm_upward_source_motion_matches_finite_difference():
    tree, pos_sorted, mass_sorted, vel_sorted = _build_sample_tree()
    order = 4
    dt = jnp.asarray(1e-6, dtype=pos_sorted.dtype)

    base = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=order,
        center_mode="com",
    )
    centers = base.multipoles.centers
    analytic = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        velocities_sorted=vel_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    plus = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted + dt * vel_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    minus = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted - dt * vel_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )

    assert analytic.multipoles.source_motion_packed is not None
    ref = (plus.multipoles.packed - minus.multipoles.packed) / (2.0 * dt)
    got = analytic.multipoles.source_motion_packed
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=2e-5, atol=1e-7)


def test_prepare_solidfmm_source_motion_multipoles_matches_upward_bundle():
    tree, pos_sorted, mass_sorted, vel_sorted = _build_sample_tree()
    order = 4
    base = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=order,
        center_mode="com",
    )
    centers = base.multipoles.centers
    bundle = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        velocities_sorted=vel_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    direct = prepare_solidfmm_complex_source_motion_multipoles(
        tree,
        pos_sorted,
        mass_sorted,
        vel_sorted,
        max_order=order,
        centers=centers,
    )
    assert bundle.multipoles.source_motion_packed is not None
    assert np.allclose(
        np.asarray(direct),
        np.asarray(bundle.multipoles.source_motion_packed),
        rtol=1e-12,
        atol=1e-12,
    )


def test_prepare_solidfmm_second_time_derivative_multipoles_matches_fd():
    tree, pos_sorted, mass_sorted, vel_sorted = _build_sample_tree()
    order = 4
    dt = jnp.asarray(1e-5, dtype=pos_sorted.dtype)
    base = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=order,
        center_mode="com",
    )
    centers = base.multipoles.centers
    got = prepare_solidfmm_complex_source_motion_multipoles(
        tree,
        pos_sorted,
        mass_sorted,
        vel_sorted,
        max_order=order,
        centers=centers,
        time_derivative_order=2,
    )
    plus = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted + dt * vel_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    zero = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    minus = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted - dt * vel_sorted,
        mass_sorted,
        max_order=order,
        center_mode="explicit",
        explicit_centers=centers,
    )
    ref = (
        plus.multipoles.packed - 2.0 * zero.multipoles.packed + minus.multipoles.packed
    ) / (dt * dt)
    rel = np.linalg.norm(np.asarray(got - ref)) / (
        np.linalg.norm(np.asarray(ref)) + 1e-12
    )
    assert rel < 2e-3
