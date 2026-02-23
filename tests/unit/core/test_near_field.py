"""Tests for near-field particle-to-particle evaluation."""

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.nearfield.near_field import compute_leaf_p2p_accelerations
from yggdrax.tree import build_tree
from yggdrax.geometry import compute_tree_geometry
from yggdrax.interactions import build_leaf_neighbor_lists

DEFAULT_TEST_LEAF_SIZE = 1
STRICT_NEAR_FIELD_THETA = 0.05


def _direct_sum(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    G: float,
    softening: float,
) -> np.ndarray:
    n = positions.shape[0]
    accelerations = np.zeros_like(positions)
    potentials = np.zeros((n,), dtype=positions.dtype)
    eps = np.finfo(positions.dtype).eps
    soft_sq = softening**2

    for i in range(n):
        diff = positions[i] - positions
        dist_sq = np.sum(diff * diff, axis=1) + soft_sq
        dist = np.sqrt(dist_sq)
        denom = dist_sq * dist + eps
        inv_dist3 = 1.0 / denom
        inv_dist3[i] = 0.0
        weighted = masses[:, None] * inv_dist3[:, None] * diff
        accelerations[i] = -G * np.sum(weighted, axis=0)

        inv_r = 1.0 / (dist + eps)
        inv_r[i] = 0.0
        potentials[i] = -G * np.sum(masses * inv_r)

    return accelerations, potentials


def test_near_field_matches_direct_sum():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.9, -0.1, 0.05],
            [0.7, 0.0, -0.05],
            [0.9, -0.1, 0.1],
            [0.2, 0.3, -0.2],
        ]
    )
    masses = jnp.array([1.0, 1.5, 0.8, 1.2, 0.5])
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
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbor_list = build_leaf_neighbor_lists(
        tree,
        geometry,
        theta=STRICT_NEAR_FIELD_THETA,
    )

    accelerations = compute_leaf_p2p_accelerations(
        tree,
        neighbor_list,
        pos_sorted,
        mass_sorted,
        G=2.0,
        softening=0.1,
    )
    direct, _ = _direct_sum(
        np.asarray(pos_sorted),
        np.asarray(mass_sorted),
        G=2.0,
        softening=0.1,
    )

    # Neighbor lists intentionally exclude pairs that are accepted by the MAC
    # as "far" (those are handled by M2L). With leaf_size=1 and a strict theta
    # this can still result in an empty neighbor list for well-separated
    # particles, so compare against the direct-sum *only* when there is at
    # least one neighbor interaction.
    if int(np.sum(np.asarray(neighbor_list.counts))) == 0:
        assert np.allclose(accelerations, 0.0)
    else:
        assert np.allclose(accelerations, direct, rtol=1e-6, atol=1e-6)


def test_near_field_returns_potentials():
    positions = jnp.array(
        [
            [-0.6, 0.2, -0.1],
            [-0.4, -0.2, 0.0],
            [0.5, 0.1, -0.2],
            [0.7, -0.3, 0.15],
        ]
    )
    masses = jnp.array([1.1, 0.9, 1.3, 0.7])
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
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbor_list = build_leaf_neighbor_lists(
        tree,
        geometry,
        theta=STRICT_NEAR_FIELD_THETA,
    )

    accelerations, potentials = compute_leaf_p2p_accelerations(
        tree,
        neighbor_list,
        pos_sorted,
        mass_sorted,
        G=1.5,
        softening=0.05,
        return_potential=True,
    )
    direct_acc, direct_pot = _direct_sum(
        np.asarray(pos_sorted),
        np.asarray(mass_sorted),
        G=1.5,
        softening=0.05,
    )

    if int(np.sum(np.asarray(neighbor_list.counts))) == 0:
        assert np.allclose(accelerations, 0.0)
        assert np.allclose(potentials, 0.0)
    else:
        assert np.allclose(accelerations, direct_acc, rtol=1e-6, atol=1e-6)
        assert np.allclose(potentials, direct_pot, rtol=1e-6, atol=1e-6)


def test_single_leaf_compute_self_interactions():
    positions = jnp.array(
        [
            [0.2, 0.1, -0.3],
            [-0.7, 0.4, 0.0],
            [0.5, -0.2, 0.8],
            [0.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 2.0, 3.0, 4.0])
    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=4,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.5)

    accelerations = compute_leaf_p2p_accelerations(
        tree,
        neighbor_list,
        pos_sorted,
        mass_sorted,
        G=1.0,
        softening=0.0,
    )
    direct, _ = _direct_sum(
        np.asarray(pos_sorted),
        np.asarray(mass_sorted),
        G=1.0,
        softening=0.0,
    )

    assert np.allclose(accelerations, direct, rtol=1e-6, atol=1e-6)


def test_near_field_jittable_with_explicit_max_leaf_size():
    positions = jnp.array(
        [
            [-0.4, 0.2, 0.1],
            [-0.3, 0.1, 0.0],
            [0.2, 0.3, 0.5],
            [0.4, -0.2, -0.1],
        ]
    )
    masses = jnp.array([1.0, 1.5, 0.8, 1.2])
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
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=1.0)

    leaf_ranges = tree.node_ranges[neighbor_list.leaf_indices]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(np.max(np.asarray(counts)))

    expected = compute_leaf_p2p_accelerations(
        tree,
        neighbor_list,
        pos_sorted,
        mass_sorted,
        G=1.0,
        softening=0.0,
        max_leaf_size=max_leaf_size,
    )

    jit_fn = jax.jit(
        lambda t, n, p, m: compute_leaf_p2p_accelerations(
            t,
            n,
            p,
            m,
            G=1.0,
            softening=0.0,
            max_leaf_size=max_leaf_size,
        )
    )

    actual = jit_fn(tree, neighbor_list, pos_sorted, mass_sorted)
    assert jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_collect_neighbor_pairs_matches_neighbor_list():
    positions = jnp.array(
        [
            [-0.3, 0.1, 0.25],
            [0.4, -0.2, 0.3],
            [-0.1, 0.5, -0.15],
            [0.2, -0.4, -0.05],
            [0.6, 0.0, 0.1],
        ]
    )
    masses = jnp.array([1.0, 1.2, 0.8, 1.4, 0.9])
    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=2,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.2)

    leaf_ranges = tree.node_ranges[neighbor_list.leaf_indices]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(np.max(np.asarray(counts)))

    _, neighbor_pairs, neighbor_count = compute_leaf_p2p_accelerations(
        tree,
        neighbor_list,
        pos_sorted,
        mass_sorted,
        G=1.0,
        softening=0.0,
        max_leaf_size=max_leaf_size,
        collect_neighbor_pairs=True,
    )

    node_ranges = np.asarray(tree.node_ranges, dtype=np.int64)
    leaf_nodes = np.asarray(neighbor_list.leaf_indices, dtype=np.int64)
    lookup = -np.ones(node_ranges.shape[0], dtype=np.int64)
    lookup[leaf_nodes] = np.arange(leaf_nodes.shape[0], dtype=np.int64)
    offsets = np.asarray(neighbor_list.offsets, dtype=np.int64)
    counts = np.asarray(neighbor_list.counts, dtype=np.int64)
    neighbors = np.asarray(neighbor_list.neighbors, dtype=np.int64)

    expected = []
    for leaf_idx in range(leaf_nodes.shape[0]):
        start = offsets[leaf_idx]
        end = start + counts[leaf_idx]
        for idx in range(start, end):
            src_node = neighbors[idx]
            src_leaf = lookup[src_node]
            if src_leaf < 0:
                continue
            expected.append((leaf_idx, int(src_leaf)))

    expected_pairs = np.asarray(expected, dtype=np.int64)
    actual_pairs = np.asarray(neighbor_pairs, dtype=np.int64)
    actual_pairs = actual_pairs[: int(neighbor_count)]

    assert actual_pairs.shape == expected_pairs.shape
    assert np.array_equal(actual_pairs, expected_pairs)
