"""Tests for solidfmm complex upward sweep helpers."""

import jax.numpy as jnp
import numpy as np
from yggdrax.tree import build_tree

from jaccpot.runtime._fmm_impl import FastMultipoleMethod
from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_source_motion_multipoles,
    prepare_solidfmm_complex_upward_sweep,
)
from jaccpot.upward.tree_expansions import NodeMultipoleData, TreeUpwardData


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


def test_prepare_solidfmm_source_motion_multipoles_matches_upward_bundle():
    tree, pos_sorted, mass_sorted, vel_sorted = _build_sample_tree()
    order = 4
    base = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=order,
        center_mode="aabb",
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


def _as_tree_upward_data(complex_upward) -> TreeUpwardData:
    multipoles = NodeMultipoleData(
        order=int(complex_upward.multipoles.order),
        centers=complex_upward.multipoles.centers,
        moments=None,  # type: ignore[arg-type]
        packed=complex_upward.multipoles.packed,
        component_matrix=complex_upward.multipoles.packed,
        source_motion_packed=complex_upward.multipoles.source_motion_packed,
    )
    return TreeUpwardData(
        geometry=complex_upward.geometry,
        mass_moments=complex_upward.mass_moments,
        multipoles=multipoles,
    )


def test_solidfmm_downward_source_motion_locals_match_finite_difference():
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

    fmm = FastMultipoleMethod(expansion_basis="solidfmm")
    base_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(base),
        theta=0.6,
    )
    analytic_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(analytic),
        theta=0.6,
        interactions=base_down.interactions,
    )
    plus_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(plus),
        theta=0.6,
        interactions=base_down.interactions,
    )
    minus_down = fmm.prepare_downward_sweep(
        tree,
        _as_tree_upward_data(minus),
        theta=0.6,
        interactions=base_down.interactions,
    )

    assert analytic_down.source_motion_locals is not None
    ref = (plus_down.locals.coefficients - minus_down.locals.coefficients) / (2.0 * dt)
    got = analytic_down.source_motion_locals.coefficients
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=3e-5, atol=1e-7)
