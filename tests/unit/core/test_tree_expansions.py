"""Tests for node multipole expansion helpers."""

import jax.numpy as jnp
import pytest
from yggdrax.tree import build_tree
from yggdrax.tree_moments import (
    compute_tree_mass_moments,
    compute_tree_multipole_moments,
    pack_multipole_expansions,
)

from jaccpot.upward.tree_expansions import compute_node_multipoles, prepare_upward_sweep

DEFAULT_TEST_LEAF_SIZE = 1


def _build_sample_tree():
    positions = jnp.array(
        [
            [-0.5, -0.5, -0.5],
            [0.1, 0.0, 0.3],
            [0.6, 0.4, -0.2],
        ]
    )
    masses = jnp.array([1.0, 2.0, 3.0])
    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    return tree, pos_sorted, mass_sorted


def test_compute_node_multipoles_com_matches_mass_moments():
    tree, pos_sorted, mass_sorted = _build_sample_tree()

    result = compute_node_multipoles(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=2,
        center_mode="com",
    )

    mass_moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)
    assert jnp.allclose(result.centers, mass_moments.center_of_mass)

    direct = compute_tree_multipole_moments(
        tree,
        pos_sorted,
        mass_sorted,
    )
    expected = pack_multipole_expansions(direct, max_order=2)
    assert jnp.allclose(result.packed, expected)


def test_compute_node_multipoles_high_order_matches_direct():
    tree, pos_sorted, mass_sorted = _build_sample_tree()

    result = compute_node_multipoles(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=4,
        center_mode="com",
    )

    direct = compute_tree_multipole_moments(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=4,
    )

    expected = pack_multipole_expansions(direct, max_order=4)

    assert result.order == 4
    assert result.moments.max_order == 4
    assert jnp.allclose(result.centers, direct.center)
    assert jnp.allclose(result.moments.mass, direct.mass)
    assert jnp.allclose(result.moments.raw_packed, direct.raw_packed)
    assert jnp.allclose(result.packed, expected)


def test_compute_node_multipoles_aabb_uses_geometry_center():
    from yggdrax.geometry import compute_tree_geometry

    tree, pos_sorted, mass_sorted = _build_sample_tree()
    geom = compute_tree_geometry(tree, pos_sorted)

    result = compute_node_multipoles(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=1,
        center_mode="aabb",
    )

    assert jnp.allclose(result.centers, geom.center)


def test_compute_node_multipoles_explicit_requires_centers():
    tree, pos_sorted, mass_sorted = _build_sample_tree()

    with pytest.raises(ValueError):
        compute_node_multipoles(
            tree,
            pos_sorted,
            mass_sorted,
            center_mode="explicit",
        )

    centers = jnp.zeros((tree.parent.shape[0], 3), dtype=pos_sorted.dtype)
    result = compute_node_multipoles(
        tree,
        pos_sorted,
        mass_sorted,
        center_mode="explicit",
        explicit_centers=centers,
    )
    assert jnp.allclose(result.centers, centers)


def test_compute_node_multipoles_rejects_unknown_mode():
    tree, pos_sorted, mass_sorted = _build_sample_tree()

    with pytest.raises(ValueError):
        compute_node_multipoles(
            tree,
            pos_sorted,
            mass_sorted,
            center_mode="nope",
        )


def test_prepare_upward_sweep_returns_consistent_data():
    from yggdrax.geometry import compute_tree_geometry
    from yggdrax.tree_moments import compute_tree_mass_moments

    tree, pos_sorted, mass_sorted = _build_sample_tree()

    prepared = prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=2,
        center_mode="aabb",
    )

    geom = compute_tree_geometry(tree, pos_sorted)
    mass_moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)

    assert jnp.allclose(prepared.geometry.center, geom.center)
    assert jnp.allclose(
        prepared.mass_moments.center_of_mass,
        mass_moments.center_of_mass,
    )
    assert jnp.allclose(prepared.multipoles.centers, geom.center)
    assert prepared.multipoles.order == 2

    direct = compute_tree_multipole_moments(
        tree,
        pos_sorted,
        mass_sorted,
        expansion_centers=geom.center,
    )
    direct_packed = pack_multipole_expansions(direct, max_order=2)
    assert jnp.allclose(prepared.multipoles.packed, direct_packed)
