"""Tests for node multipole expansion helpers."""

import jax.numpy as jnp
import pytest
from yggdrax.tree import build_tree
from yggdrax.tree_moments import (
    compute_tree_mass_moments,
    compute_tree_multipole_moments,
    pack_multipole_expansions,
)

from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_upward_sweep,
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


@pytest.mark.parametrize("mode", ["aabb", "AABB", "geometric"])
def test_removed_geometric_centres_raise_a_removal_error(mode):
    """AABB expansion centres went in the 2026-10 cleanup (X3); naming them must say so.

    Every upward sweep that takes a ``center_mode`` shares the check, so all three
    are asserted: an old caller gets the removal message, not an "unknown mode".
    """
    tree, pos_sorted, mass_sorted = _build_sample_tree()

    with pytest.raises(ValueError, match="expansion centres were removed"):
        compute_node_multipoles(tree, pos_sorted, mass_sorted, center_mode=mode)
    with pytest.raises(ValueError, match="expansion centres were removed"):
        prepare_upward_sweep(tree, pos_sorted, mass_sorted, center_mode=mode)
    with pytest.raises(ValueError, match="expansion centres were removed"):
        prepare_solidfmm_complex_upward_sweep(
            tree, pos_sorted, mass_sorted, center_mode=mode
        )


def test_prepare_upward_sweep_returns_consistent_data():
    """Geometry, mass moments and multipoles agree with the direct builders.

    Expanded about the centres of mass, the production centres. This used AABB
    centres until they went in the 2026-10 cleanup (X3), so the geometry and the
    expansion centres are now two different checks.
    """
    from yggdrax.geometry import compute_tree_geometry
    from yggdrax.tree_moments import compute_tree_mass_moments

    tree, pos_sorted, mass_sorted = _build_sample_tree()

    prepared = prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=2,
        center_mode="com",
    )

    geom = compute_tree_geometry(tree, pos_sorted)
    mass_moments = compute_tree_mass_moments(tree, pos_sorted, mass_sorted)

    assert jnp.allclose(prepared.geometry.center, geom.center)
    assert jnp.allclose(
        prepared.mass_moments.center_of_mass,
        mass_moments.center_of_mass,
    )
    assert jnp.allclose(prepared.multipoles.centers, mass_moments.center_of_mass)
    assert prepared.multipoles.order == 2

    direct = compute_tree_multipole_moments(
        tree,
        pos_sorted,
        mass_sorted,
        expansion_centers=mass_moments.center_of_mass,
    )
    direct_packed = pack_multipole_expansions(direct, max_order=2)
    assert jnp.allclose(prepared.multipoles.packed, direct_packed)
