"""Tests for solidfmm complex upward sweep helpers."""

import jax.numpy as jnp
import numpy as np
from yggdrax.tree import build_tree

from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_upward_sweep,
)


def _build_sample_tree():
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
        leaf_size=2,
    )
    vel_sorted = velocities[jnp.asarray(tree.particle_indices)]
    return tree, pos_sorted, mass_sorted, vel_sorted


def test_prepare_solidfmm_upward_source_motion_optional_none():
    tree, pos_sorted, mass_sorted, _ = _build_sample_tree()
    upward = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=3,
        center_mode="aabb",
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
        center_mode="aabb",
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
