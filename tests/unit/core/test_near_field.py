"""Tests for near-field particle-to-particle evaluation."""

import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.geometry import compute_tree_geometry
from yggdrax.interactions import build_leaf_neighbor_lists
from yggdrax.tree import build_tree

from jaccpot.nearfield.near_field import (
    _compact_reduced_pair_bucket_rows,
    collect_radix_fast_lane_counters,
    compute_leaf_p2p_accelerations,
    compute_leaf_p2p_accelerations_large_n_accel_only,
    compute_leaf_p2p_accelerations_radix_fast_lane,
    prepare_leaf_neighbor_pairs,
)
from jaccpot.runtime._large_n_types import RadixFastNearfieldPayload

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


def test_large_n_accel_only_prepared_bucketed_matches_generic():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.7, -0.1, 0.05],
            [-0.2, 0.3, -0.2],
            [0.15, 0.25, 0.1],
            [0.6, -0.2, -0.05],
            [0.75, 0.05, 0.2],
        ]
    )
    masses = jnp.array([1.0, 1.5, 0.7, 0.9, 1.2, 0.8])
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
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.3)

    node_ranges = jnp.asarray(tree.node_ranges)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices)
    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(np.max(np.asarray(counts)))
    offsets = jnp.arange(max_leaf_size, dtype=leaf_ranges.dtype)
    leaf_particle_indices = leaf_ranges[:, 0][:, None] + offsets[None, :]
    leaf_particle_mask = offsets[None, :] < counts[:, None]

    generic = compute_leaf_p2p_accelerations(
        tree,
        neighbor_list,
        pos_sorted,
        mass_sorted,
        G=1.25,
        softening=0.05,
        max_leaf_size=max_leaf_size,
        nearfield_mode="bucketed",
        edge_chunk_size=2,
        leaf_particle_indices_override=leaf_particle_indices,
        leaf_particle_mask_override=leaf_particle_mask,
    )

    specialized = compute_leaf_p2p_accelerations_large_n_accel_only(
        tree,
        neighbor_list,
        pos_sorted,
        mass_sorted,
        G=1.25,
        softening=0.05,
        edge_chunk_size=2,
        leaf_particle_indices=leaf_particle_indices,
        leaf_particle_mask=leaf_particle_mask,
    )

    assert np.allclose(
        np.asarray(specialized),
        np.asarray(generic),
        rtol=1e-6,
        atol=1e-6,
    )


def test_compact_reduced_pair_bucket_rows_packs_valid_prefix():
    reduced_target_leaf_ids = jnp.array([4, 0, 7, 0], dtype=jnp.int32)
    reduced_pair_acc = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    reduced_valid = jnp.array([True, False, True, False])

    compact_leaf_ids, compact_pair_acc, compact_valid = (
        _compact_reduced_pair_bucket_rows(
            reduced_target_leaf_ids,
            reduced_pair_acc,
            reduced_valid,
        )
    )

    assert np.array_equal(
        np.asarray(compact_leaf_ids),
        np.asarray([4, 7, 0, 0], dtype=np.int32),
    )
    assert np.array_equal(
        np.asarray(compact_valid),
        np.asarray([True, True, False, False]),
    )
    assert np.allclose(
        np.asarray(compact_pair_acc),
        np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
    )


def test_large_n_accel_only_delayed_scatter_chunking_matches_baseline():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.7, -0.1, 0.05],
            [-0.2, 0.3, -0.2],
            [0.15, 0.25, 0.1],
            [0.6, -0.2, -0.05],
            [0.75, 0.05, 0.2],
        ]
    )
    masses = jnp.array([1.0, 1.5, 0.7, 0.9, 1.2, 0.8])
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
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.3)

    node_ranges = jnp.asarray(tree.node_ranges)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices)
    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(np.max(np.asarray(counts)))
    offsets = jnp.arange(max_leaf_size, dtype=leaf_ranges.dtype)
    leaf_particle_indices = leaf_ranges[:, 0][:, None] + offsets[None, :]
    leaf_particle_mask = offsets[None, :] < counts[:, None]

    old_env = os.environ.get("JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS")
    try:
        os.environ["JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS"] = "1"
        baseline = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )

        os.environ["JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS"] = "2"
        delayed = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )
    finally:
        if old_env is None:
            os.environ.pop("JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS", None)
        else:
            os.environ["JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS"] = old_env

    assert np.allclose(
        np.asarray(delayed),
        np.asarray(baseline),
        rtol=1e-6,
        atol=1e-6,
    )


def test_large_n_accel_only_target_owned_accum_matches_baseline():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.7, -0.1, 0.05],
            [-0.2, 0.3, -0.2],
            [0.15, 0.25, 0.1],
            [0.6, -0.2, -0.05],
            [0.75, 0.05, 0.2],
        ]
    )
    masses = jnp.array([1.0, 1.5, 0.7, 0.9, 1.2, 0.8])
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
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.3)

    node_ranges = jnp.asarray(tree.node_ranges)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices)
    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(np.max(np.asarray(counts)))
    offsets = jnp.arange(max_leaf_size, dtype=leaf_ranges.dtype)
    leaf_particle_indices = leaf_ranges[:, 0][:, None] + offsets[None, :]
    leaf_particle_mask = offsets[None, :] < counts[:, None]

    old_target_owned = os.environ.get("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM")
    old_batch = os.environ.get("JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE")
    old_neighbor_block = os.environ.get(
        "JACCPOT_LARGE_N_TARGET_LEAF_NEIGHBOR_BLOCK_SIZE"
    )
    try:
        os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = "0"
        baseline = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )

        os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = "1"
        os.environ["JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE"] = "2"
        os.environ["JACCPOT_LARGE_N_TARGET_LEAF_NEIGHBOR_BLOCK_SIZE"] = "2"
        target_owned = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )
    finally:
        if old_target_owned is None:
            os.environ.pop("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM", None)
        else:
            os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = old_target_owned
        if old_batch is None:
            os.environ.pop("JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE", None)
        else:
            os.environ["JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE"] = old_batch
        if old_neighbor_block is None:
            os.environ.pop("JACCPOT_LARGE_N_TARGET_LEAF_NEIGHBOR_BLOCK_SIZE", None)
        else:
            os.environ["JACCPOT_LARGE_N_TARGET_LEAF_NEIGHBOR_BLOCK_SIZE"] = (
                old_neighbor_block
            )

    assert np.allclose(
        np.asarray(target_owned),
        np.asarray(baseline),
        rtol=1e-6,
        atol=1e-6,
    )


def test_large_n_accel_only_sorted_scatter_hint_matches_baseline():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.7, -0.1, 0.05],
            [-0.2, 0.3, -0.2],
            [0.15, 0.25, 0.1],
            [0.6, -0.2, -0.05],
            [0.75, 0.05, 0.2],
        ]
    )
    masses = jnp.array([1.0, 1.5, 0.7, 0.9, 1.2, 0.8])
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
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.3)

    node_ranges = jnp.asarray(tree.node_ranges)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices)
    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(np.max(np.asarray(counts)))
    offsets = jnp.arange(max_leaf_size, dtype=leaf_ranges.dtype)
    leaf_particle_indices = leaf_ranges[:, 0][:, None] + offsets[None, :]
    leaf_particle_mask = offsets[None, :] < counts[:, None]

    old_sorted_hint = os.environ.get("JACCPOT_LARGE_N_SORTED_SCATTER_HINT")
    old_target_owned = os.environ.get("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM")
    try:
        os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = "0"
        os.environ["JACCPOT_LARGE_N_SORTED_SCATTER_HINT"] = "0"
        baseline = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )

        os.environ["JACCPOT_LARGE_N_SORTED_SCATTER_HINT"] = "1"
        sorted_hint = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )
    finally:
        if old_target_owned is None:
            os.environ.pop("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM", None)
        else:
            os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = old_target_owned
        if old_sorted_hint is None:
            os.environ.pop("JACCPOT_LARGE_N_SORTED_SCATTER_HINT", None)
        else:
            os.environ["JACCPOT_LARGE_N_SORTED_SCATTER_HINT"] = old_sorted_hint

    assert np.allclose(
        np.asarray(sorted_hint),
        np.asarray(baseline),
        rtol=1e-6,
        atol=1e-6,
    )


def test_large_n_accel_only_grouped_sorted_scatter_matches_baseline():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.7, -0.1, 0.05],
            [-0.2, 0.3, -0.2],
            [0.15, 0.25, 0.1],
            [0.6, -0.2, -0.05],
            [0.75, 0.05, 0.2],
        ]
    )
    masses = jnp.array([1.0, 1.5, 0.7, 0.9, 1.2, 0.8])
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
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.3)

    node_ranges = jnp.asarray(tree.node_ranges)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices)
    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(np.max(np.asarray(counts)))
    offsets = jnp.arange(max_leaf_size, dtype=leaf_ranges.dtype)
    leaf_particle_indices = leaf_ranges[:, 0][:, None] + offsets[None, :]
    leaf_particle_mask = offsets[None, :] < counts[:, None]

    old_sorted_hint = os.environ.get("JACCPOT_LARGE_N_SORTED_SCATTER_HINT")
    old_grouped_sorted = os.environ.get("JACCPOT_LARGE_N_GROUPED_SORTED_SCATTER")
    old_target_owned = os.environ.get("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM")
    try:
        os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = "0"
        os.environ["JACCPOT_LARGE_N_SORTED_SCATTER_HINT"] = "0"
        os.environ["JACCPOT_LARGE_N_GROUPED_SORTED_SCATTER"] = "0"
        baseline = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )

        os.environ["JACCPOT_LARGE_N_SORTED_SCATTER_HINT"] = "1"
        os.environ["JACCPOT_LARGE_N_GROUPED_SORTED_SCATTER"] = "1"
        grouped_sorted = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )
    finally:
        if old_target_owned is None:
            os.environ.pop("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM", None)
        else:
            os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = old_target_owned
        if old_sorted_hint is None:
            os.environ.pop("JACCPOT_LARGE_N_SORTED_SCATTER_HINT", None)
        else:
            os.environ["JACCPOT_LARGE_N_SORTED_SCATTER_HINT"] = old_sorted_hint
        if old_grouped_sorted is None:
            os.environ.pop("JACCPOT_LARGE_N_GROUPED_SORTED_SCATTER", None)
        else:
            os.environ["JACCPOT_LARGE_N_GROUPED_SORTED_SCATTER"] = old_grouped_sorted

    assert np.allclose(
        np.asarray(grouped_sorted),
        np.asarray(baseline),
        rtol=1e-6,
        atol=1e-6,
    )


def test_large_n_accel_only_target_owned_accum_v2_matches_baseline():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.7, -0.1, 0.05],
            [-0.2, 0.3, -0.2],
            [0.15, 0.25, 0.1],
            [0.6, -0.2, -0.05],
            [0.75, 0.05, 0.2],
        ]
    )
    masses = jnp.array([1.0, 1.5, 0.7, 0.9, 1.2, 0.8])
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
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.3)

    node_ranges = jnp.asarray(tree.node_ranges)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices)
    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(np.max(np.asarray(counts)))
    offsets = jnp.arange(max_leaf_size, dtype=leaf_ranges.dtype)
    leaf_particle_indices = leaf_ranges[:, 0][:, None] + offsets[None, :]
    leaf_particle_mask = offsets[None, :] < counts[:, None]

    old_target_owned = os.environ.get("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM")
    old_target_owned_v2 = os.environ.get("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM_V2")
    old_batch = os.environ.get("JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE")
    old_neighbor_block = os.environ.get(
        "JACCPOT_LARGE_N_TARGET_LEAF_NEIGHBOR_BLOCK_SIZE"
    )
    try:
        os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = "0"
        os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM_V2"] = "0"
        baseline = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )

        os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = "1"
        os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM_V2"] = "1"
        os.environ["JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE"] = "2"
        os.environ["JACCPOT_LARGE_N_TARGET_LEAF_NEIGHBOR_BLOCK_SIZE"] = "2"
        target_owned_v2 = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )
    finally:
        if old_target_owned is None:
            os.environ.pop("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM", None)
        else:
            os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = old_target_owned
        if old_target_owned_v2 is None:
            os.environ.pop("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM_V2", None)
        else:
            os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM_V2"] = old_target_owned_v2
        if old_batch is None:
            os.environ.pop("JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE", None)
        else:
            os.environ["JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE"] = old_batch
        if old_neighbor_block is None:
            os.environ.pop("JACCPOT_LARGE_N_TARGET_LEAF_NEIGHBOR_BLOCK_SIZE", None)
        else:
            os.environ["JACCPOT_LARGE_N_TARGET_LEAF_NEIGHBOR_BLOCK_SIZE"] = (
                old_neighbor_block
            )

    assert np.allclose(
        np.asarray(target_owned_v2),
        np.asarray(baseline),
        rtol=1e-6,
        atol=1e-6,
    )


def test_large_n_accel_only_superchunk_target_reduce_matches_baseline():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.7, -0.1, 0.05],
            [-0.2, 0.3, -0.2],
            [0.15, 0.25, 0.1],
            [0.6, -0.2, -0.05],
            [0.75, 0.05, 0.2],
        ]
    )
    masses = jnp.array([1.0, 1.5, 0.7, 0.9, 1.2, 0.8])
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
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.3)

    node_ranges = jnp.asarray(tree.node_ranges)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices)
    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(np.max(np.asarray(counts)))
    offsets = jnp.arange(max_leaf_size, dtype=leaf_ranges.dtype)
    leaf_particle_indices = leaf_ranges[:, 0][:, None] + offsets[None, :]
    leaf_particle_mask = offsets[None, :] < counts[:, None]

    old_target_owned = os.environ.get("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM")
    old_chunks = os.environ.get("JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS")
    old_superchunk_reduce = os.environ.get("JACCPOT_LARGE_N_SUPERCHUNK_TARGET_REDUCE")
    try:
        os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = "0"
        os.environ["JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS"] = "2"
        os.environ["JACCPOT_LARGE_N_SUPERCHUNK_TARGET_REDUCE"] = "0"
        baseline = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )

        os.environ["JACCPOT_LARGE_N_SUPERCHUNK_TARGET_REDUCE"] = "1"
        reduced = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )
    finally:
        if old_target_owned is None:
            os.environ.pop("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM", None)
        else:
            os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = old_target_owned
        if old_chunks is None:
            os.environ.pop("JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS", None)
        else:
            os.environ["JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS"] = old_chunks
        if old_superchunk_reduce is None:
            os.environ.pop("JACCPOT_LARGE_N_SUPERCHUNK_TARGET_REDUCE", None)
        else:
            os.environ["JACCPOT_LARGE_N_SUPERCHUNK_TARGET_REDUCE"] = (
                old_superchunk_reduce
            )

    assert np.allclose(
        np.asarray(reduced),
        np.asarray(baseline),
        rtol=1e-6,
        atol=1e-6,
    )


def test_large_n_accel_only_disable_chunk_cond_matches_baseline():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.7, -0.1, 0.05],
            [-0.2, 0.3, -0.2],
            [0.15, 0.25, 0.1],
            [0.6, -0.2, -0.05],
            [0.75, 0.05, 0.2],
        ]
    )
    masses = jnp.array([1.0, 1.5, 0.7, 0.9, 1.2, 0.8])
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
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.3)

    node_ranges = jnp.asarray(tree.node_ranges)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices)
    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(np.max(np.asarray(counts)))
    offsets = jnp.arange(max_leaf_size, dtype=leaf_ranges.dtype)
    leaf_particle_indices = leaf_ranges[:, 0][:, None] + offsets[None, :]
    leaf_particle_mask = offsets[None, :] < counts[:, None]

    old_target_owned = os.environ.get("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM")
    old_disable_chunk_cond = os.environ.get("JACCPOT_LARGE_N_DISABLE_CHUNK_COND")
    old_chunks = os.environ.get("JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS")
    try:
        os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = "0"
        os.environ["JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS"] = "1"
        os.environ["JACCPOT_LARGE_N_DISABLE_CHUNK_COND"] = "0"
        baseline = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )

        os.environ["JACCPOT_LARGE_N_DISABLE_CHUNK_COND"] = "1"
        no_cond = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            pos_sorted,
            mass_sorted,
            G=1.25,
            softening=0.05,
            edge_chunk_size=2,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
        )
    finally:
        if old_target_owned is None:
            os.environ.pop("JACCPOT_LARGE_N_TARGET_OWNED_ACCUM", None)
        else:
            os.environ["JACCPOT_LARGE_N_TARGET_OWNED_ACCUM"] = old_target_owned
        if old_disable_chunk_cond is None:
            os.environ.pop("JACCPOT_LARGE_N_DISABLE_CHUNK_COND", None)
        else:
            os.environ["JACCPOT_LARGE_N_DISABLE_CHUNK_COND"] = old_disable_chunk_cond
        if old_chunks is None:
            os.environ.pop("JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS", None)
        else:
            os.environ["JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS"] = old_chunks

    assert np.allclose(
        np.asarray(no_cond),
        np.asarray(baseline),
        rtol=1e-6,
        atol=1e-6,
    )


def _build_test_radix_fast_payload(
    *,
    tree,
    neighbor_list,
    leaf_particle_indices,
    leaf_particle_mask,
    batch_tile_t: int = 2,
    batch_tile_s: int = 2,
):
    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    offsets = jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE)
    neighbors = jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE)

    target_leaf_ids, source_leaf_ids, valid_pairs = prepare_leaf_neighbor_pairs(
        node_ranges,
        leaf_nodes,
        offsets,
        neighbors,
        sort_by_source=False,
    )

    num_leaves = int(leaf_particle_indices.shape[0])
    max_neighbors = (
        int(np.max(np.asarray(neighbor_list.counts))) if num_leaves > 0 else 0
    )
    max_leaf_size = int(leaf_particle_indices.shape[1]) if num_leaves > 0 else 0

    source_leaf_ids_padded = jnp.zeros((num_leaves, max_neighbors), dtype=INDEX_DTYPE)
    source_leaf_valid_mask_padded = jnp.zeros((num_leaves, max_neighbors), dtype=bool)

    if max_neighbors > 0 and int(target_leaf_ids.shape[0]) > 0:
        edge_indices = jnp.arange(target_leaf_ids.shape[0], dtype=INDEX_DTYPE)
        local_edge_idx = edge_indices - offsets[target_leaf_ids]
        in_bounds = local_edge_idx < max_neighbors
        keep = valid_pairs & in_bounds
        source_leaf_ids_padded = source_leaf_ids_padded.at[
            target_leaf_ids[keep], local_edge_idx[keep]
        ].set(source_leaf_ids[keep])
        source_leaf_valid_mask_padded = source_leaf_valid_mask_padded.at[
            target_leaf_ids[keep], local_edge_idx[keep]
        ].set(True)

    if max_neighbors > 0 and max_leaf_size > 0:
        source_particle_ids = leaf_particle_indices[source_leaf_ids_padded]
        source_particle_mask = (
            leaf_particle_mask[source_leaf_ids_padded]
            & source_leaf_valid_mask_padded[..., None]
        )
    else:
        source_particle_ids = jnp.zeros(
            (num_leaves, 0, max_leaf_size), dtype=INDEX_DTYPE
        )
        source_particle_mask = jnp.zeros((num_leaves, 0, max_leaf_size), dtype=bool)

    return RadixFastNearfieldPayload(
        target_leaf_ids=jnp.arange(num_leaves, dtype=INDEX_DTYPE),
        target_particle_ids=jnp.asarray(leaf_particle_indices, dtype=INDEX_DTYPE),
        target_particle_mask=jnp.asarray(leaf_particle_mask, dtype=bool),
        source_leaf_ids=jnp.asarray(source_leaf_ids_padded, dtype=INDEX_DTYPE),
        source_leaf_valid_mask=jnp.asarray(source_leaf_valid_mask_padded, dtype=bool),
        source_particle_ids=jnp.asarray(source_particle_ids, dtype=INDEX_DTYPE),
        source_particle_mask=jnp.asarray(source_particle_mask, dtype=bool),
        batch_tile_t=int(batch_tile_t),
        batch_tile_s=int(batch_tile_s),
    )


def test_radix_fast_lane_accel_matches_large_n_specialized_small():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.7, -0.1, 0.05],
            [-0.2, 0.3, -0.2],
            [0.15, 0.25, 0.1],
            [0.6, -0.2, -0.05],
            [0.75, 0.05, 0.2],
        ]
    )
    masses = jnp.array([1.0, 1.5, 0.7, 0.9, 1.2, 0.8])
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
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.3)

    node_ranges = jnp.asarray(tree.node_ranges)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices)
    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(np.max(np.asarray(counts)))
    offsets = jnp.arange(max_leaf_size, dtype=leaf_ranges.dtype)
    leaf_particle_indices = leaf_ranges[:, 0][:, None] + offsets[None, :]
    leaf_particle_mask = offsets[None, :] < counts[:, None]

    baseline = compute_leaf_p2p_accelerations_large_n_accel_only(
        tree,
        neighbor_list,
        pos_sorted,
        mass_sorted,
        G=1.25,
        softening=0.05,
        edge_chunk_size=2,
        leaf_particle_indices=leaf_particle_indices,
        leaf_particle_mask=leaf_particle_mask,
    )

    payload = _build_test_radix_fast_payload(
        tree=tree,
        neighbor_list=neighbor_list,
        leaf_particle_indices=leaf_particle_indices,
        leaf_particle_mask=leaf_particle_mask,
    )
    fast_lane = compute_leaf_p2p_accelerations_radix_fast_lane(
        positions_sorted=pos_sorted,
        masses_sorted=mass_sorted,
        payload=payload,
        G=1.25,
        softening=0.05,
    )

    assert np.allclose(
        np.asarray(fast_lane),
        np.asarray(baseline),
        rtol=1e-6,
        atol=1e-6,
    )


def test_radix_fast_lane_occupancy_sort_and_empty_tile_skip_match_fallback():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.7, -0.1, 0.05],
            [-0.2, 0.3, -0.2],
            [0.15, 0.25, 0.1],
            [0.6, -0.2, -0.05],
            [0.75, 0.05, 0.2],
        ]
    )
    masses = jnp.array([1.0, 1.5, 0.7, 0.9, 1.2, 0.8])
    bounds = (jnp.array([-1.0, -1.0, -1.0]), jnp.array([1.0, 1.0, 1.0]))
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=2,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.3)
    node_ranges = jnp.asarray(tree.node_ranges)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices)
    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(np.max(np.asarray(counts)))
    offsets = jnp.arange(max_leaf_size, dtype=leaf_ranges.dtype)
    leaf_particle_indices = leaf_ranges[:, 0][:, None] + offsets[None, :]
    leaf_particle_mask = offsets[None, :] < counts[:, None]

    payload = _build_test_radix_fast_payload(
        tree=tree,
        neighbor_list=neighbor_list,
        leaf_particle_indices=leaf_particle_indices,
        leaf_particle_mask=leaf_particle_mask,
    )
    num_leaves, source_slots = payload.source_leaf_ids.shape
    block_size = 2
    padded_slots = ((source_slots + block_size - 1) // block_size) * block_size
    pad_slots = padded_slots - source_slots
    source_leaf_ids = jnp.pad(payload.source_leaf_ids, ((0, 0), (0, pad_slots)))
    source_leaf_valid = jnp.pad(
        payload.source_leaf_valid_mask,
        ((0, 0), (0, pad_slots)),
    )
    payload = replace(
        payload,
        source_leaf_ids=source_leaf_ids.reshape((num_leaves, -1, block_size)),
        source_leaf_valid_mask=source_leaf_valid.reshape(
            (num_leaves, -1, block_size)
        ),
        source_particle_ids=jnp.zeros((0, 0, 0), dtype=INDEX_DTYPE),
        source_particle_mask=jnp.zeros((0, 0, 0), dtype=bool),
        fallback_block_tile_size=1,
    )

    flag_names = (
        "JACCPOT_LARGE_N_RADIX_FAST_OCCUPANCY_SORT",
        "JACCPOT_LARGE_N_RADIX_FAST_SKIP_EMPTY_TILES",
        "JACCPOT_LARGE_N_RADIX_FAST_COMPONENTWISE_PAIRS",
    )
    old_flags = {name: os.environ.get(name) for name in flag_names}
    try:
        for name in flag_names:
            os.environ[name] = "0"
        baseline = compute_leaf_p2p_accelerations_radix_fast_lane(
            positions_sorted=pos_sorted,
            masses_sorted=mass_sorted,
            payload=payload,
            G=1.25,
            softening=0.05,
        )
        for name in flag_names:
            os.environ[name] = "1"
        optimized = compute_leaf_p2p_accelerations_radix_fast_lane(
            positions_sorted=pos_sorted,
            masses_sorted=mass_sorted,
            payload=payload,
            G=1.25,
            softening=0.05,
        )
    finally:
        for name, value in old_flags.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    assert np.allclose(
        np.asarray(optimized),
        np.asarray(baseline),
        rtol=1e-6,
        atol=1e-6,
    )


def test_collect_radix_fast_lane_counters_matches_payload_formula():
    positions = jnp.array(
        [
            [-0.4, 0.2, -0.1],
            [-0.2, -0.1, 0.0],
            [0.2, 0.3, 0.1],
            [0.5, -0.3, 0.2],
        ],
        dtype=jnp.float32,
    )
    masses = jnp.array([1.0, 1.2, 0.8, 1.1], dtype=jnp.float32)
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
    )

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=2,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.4)

    node_ranges = jnp.asarray(tree.node_ranges)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices)
    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    max_leaf_size = int(np.max(np.asarray(counts)))
    offsets = jnp.arange(max_leaf_size, dtype=leaf_ranges.dtype)
    leaf_particle_indices = leaf_ranges[:, 0][:, None] + offsets[None, :]
    leaf_particle_mask = offsets[None, :] < counts[:, None]

    payload = _build_test_radix_fast_payload(
        tree=tree,
        neighbor_list=neighbor_list,
        leaf_particle_indices=leaf_particle_indices,
        leaf_particle_mask=leaf_particle_mask,
        batch_tile_t=2,
        batch_tile_s=2,
    )

    counters = collect_radix_fast_lane_counters(
        payload=payload,
        positions_dtype=pos_sorted.dtype,
        masses_dtype=mass_sorted.dtype,
        accelerations_dtype=pos_sorted.dtype,
    )

    target_slots = int(payload.target_particle_ids.size)
    source_slots = int(payload.source_particle_ids.size)
    itemsize = int(jnp.dtype(pos_sorted.dtype).itemsize)
    mass_itemsize = int(jnp.dtype(mass_sorted.dtype).itemsize)
    expected_gather_bytes = target_slots * (
        3 * itemsize + mass_itemsize
    ) + source_slots * (3 * itemsize + mass_itemsize)
    expected_scatter_bytes = target_slots * 3 * itemsize
    expected_scatter_ops = target_slots

    assert int(counters.gather_bytes) == int(expected_gather_bytes)
    assert int(counters.scatter_bytes) == int(expected_scatter_bytes)
    assert int(counters.scatter_ops) == int(expected_scatter_ops)
    assert int(counters.target_batches) >= 1
    assert int(counters.source_slot_tiles) >= 0
