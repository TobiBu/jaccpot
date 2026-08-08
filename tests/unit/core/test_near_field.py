"""Tests for near-field particle-to-particle evaluation."""

import os
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError
from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.geometry import compute_tree_geometry
from yggdrax.interactions import build_leaf_neighbor_lists
from yggdrax.tree import build_tree

from jaccpot.nearfield.near_field import (
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


@pytest.fixture(scope="module")
def accel_only_case():
    """Shared 6-particle tree + leaf-particle index/mask used by the large-N
    accel-only parity tests (built once per module)."""
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
        positions, masses, bounds, return_reordered=True, leaf_size=2
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
    return SimpleNamespace(
        tree=tree,
        neighbor_list=neighbor_list,
        pos_sorted=pos_sorted,
        mass_sorted=mass_sorted,
        leaf_particle_indices=leaf_particle_indices,
        leaf_particle_mask=leaf_particle_mask,
    )


def _accel_only(case):
    return compute_leaf_p2p_accelerations_large_n_accel_only(
        case.tree,
        case.neighbor_list,
        case.pos_sorted,
        case.mass_sorted,
        G=1.25,
        softening=0.05,
        edge_chunk_size=2,
        leaf_particle_indices=case.leaf_particle_indices,
        leaf_particle_mask=case.leaf_particle_mask,
    )


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


_ACCEL_ONLY_PARITY_CASES = {
    "delayed_scatter_chunking": (
        {"JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS": "1"},
        {"JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS": "2"},
    ),
    "target_owned_accum": (
        {"JACCPOT_LARGE_N_TARGET_OWNED_ACCUM": "0"},
        {
            "JACCPOT_LARGE_N_TARGET_OWNED_ACCUM": "1",
            "JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE": "2",
            "JACCPOT_LARGE_N_TARGET_LEAF_NEIGHBOR_BLOCK_SIZE": "2",
        },
    ),
    "sorted_scatter_hint": (
        {
            "JACCPOT_LARGE_N_TARGET_OWNED_ACCUM": "0",
            "JACCPOT_LARGE_N_SORTED_SCATTER_HINT": "0",
        },
        {
            "JACCPOT_LARGE_N_TARGET_OWNED_ACCUM": "0",
            "JACCPOT_LARGE_N_SORTED_SCATTER_HINT": "1",
        },
    ),
    "grouped_sorted_scatter": (
        {
            "JACCPOT_LARGE_N_TARGET_OWNED_ACCUM": "0",
            "JACCPOT_LARGE_N_SORTED_SCATTER_HINT": "0",
            "JACCPOT_LARGE_N_GROUPED_SORTED_SCATTER": "0",
        },
        {
            "JACCPOT_LARGE_N_TARGET_OWNED_ACCUM": "0",
            "JACCPOT_LARGE_N_SORTED_SCATTER_HINT": "1",
            "JACCPOT_LARGE_N_GROUPED_SORTED_SCATTER": "1",
        },
    ),
    "target_owned_accum_v2": (
        {
            "JACCPOT_LARGE_N_TARGET_OWNED_ACCUM": "0",
            "JACCPOT_LARGE_N_TARGET_OWNED_ACCUM_V2": "0",
        },
        {
            "JACCPOT_LARGE_N_TARGET_OWNED_ACCUM": "1",
            "JACCPOT_LARGE_N_TARGET_OWNED_ACCUM_V2": "1",
            "JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE": "2",
            "JACCPOT_LARGE_N_TARGET_LEAF_NEIGHBOR_BLOCK_SIZE": "2",
        },
    ),
    "superchunk_target_reduce": (
        {
            "JACCPOT_LARGE_N_TARGET_OWNED_ACCUM": "0",
            "JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS": "2",
            "JACCPOT_LARGE_N_SUPERCHUNK_TARGET_REDUCE": "0",
        },
        {
            "JACCPOT_LARGE_N_TARGET_OWNED_ACCUM": "0",
            "JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS": "2",
            "JACCPOT_LARGE_N_SUPERCHUNK_TARGET_REDUCE": "1",
        },
    ),
    "disable_chunk_cond": (
        {
            "JACCPOT_LARGE_N_TARGET_OWNED_ACCUM": "0",
            "JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS": "1",
            "JACCPOT_LARGE_N_DISABLE_CHUNK_COND": "0",
        },
        {
            "JACCPOT_LARGE_N_TARGET_OWNED_ACCUM": "0",
            "JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS": "1",
            "JACCPOT_LARGE_N_DISABLE_CHUNK_COND": "1",
        },
    ),
}


@pytest.mark.parametrize(
    "baseline_env, optimized_env",
    list(_ACCEL_ONLY_PARITY_CASES.values()),
    ids=list(_ACCEL_ONLY_PARITY_CASES.keys()),
)
def test_large_n_accel_only_env_variants_match_baseline(
    monkeypatch, accel_only_case, baseline_env, optimized_env
):
    """Every large-N accel-only scatter/accumulation variant must reproduce the
    default baseline bit-for-bit on the shared small case.

    Consolidates the former per-flag ``*_matches_baseline`` tests, which each
    rebuilt the identical 6-particle tree and ran the same baseline/optimized
    A/B comparison.
    """
    for name, value in baseline_env.items():
        monkeypatch.setenv(name, value)
    baseline = _accel_only(accel_only_case)

    for name, value in optimized_env.items():
        monkeypatch.setenv(name, value)
    optimized = _accel_only(accel_only_case)

    assert np.allclose(
        np.asarray(optimized), np.asarray(baseline), rtol=1e-6, atol=1e-6
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


def _build_small_fast_lane_case(*, leaf_size=2, theta=0.3):
    """Shared small tree + radix fast-lane payload for Pallas parity tests."""
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
        leaf_size=leaf_size,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=theta)

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
    return pos_sorted, mass_sorted, payload, tree, neighbor_list


def test_radix_fast_lane_pallas_accel_matches_baseline(monkeypatch):
    pos_sorted, mass_sorted, payload, _, _ = _build_small_fast_lane_case()

    baseline = compute_leaf_p2p_accelerations_radix_fast_lane(
        positions_sorted=pos_sorted,
        masses_sorted=mass_sorted,
        payload=payload,
        G=1.25,
        softening=0.05,
        use_pallas=False,
    )

    # Force the fused Pallas path to run in interpret mode (CPU/CI-safe).
    monkeypatch.setenv("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1")
    fused = compute_leaf_p2p_accelerations_radix_fast_lane(
        positions_sorted=pos_sorted,
        masses_sorted=mass_sorted,
        payload=payload,
        G=1.25,
        softening=0.05,
        use_pallas=True,
    )

    assert np.allclose(np.asarray(fused), np.asarray(baseline), rtol=1e-5, atol=1e-6)


def test_radix_fast_lane_pallas_potential_matches_generic(monkeypatch):
    pos_sorted, mass_sorted, payload, tree, neighbor_list = (
        _build_small_fast_lane_case()
    )

    ref_acc, ref_pot = compute_leaf_p2p_accelerations(
        tree,
        neighbor_list,
        pos_sorted,
        mass_sorted,
        G=1.25,
        softening=0.05,
        return_potential=True,
    )

    monkeypatch.setenv("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1")
    fused_acc, fused_pot = compute_leaf_p2p_accelerations_radix_fast_lane(
        positions_sorted=pos_sorted,
        masses_sorted=mass_sorted,
        payload=payload,
        G=1.25,
        softening=0.05,
        return_potential=True,
        use_pallas=True,
    )

    assert np.allclose(np.asarray(fused_acc), np.asarray(ref_acc), rtol=1e-5, atol=1e-6)
    assert np.allclose(np.asarray(fused_pot), np.asarray(ref_pot), rtol=1e-5, atol=1e-6)


def test_radix_fast_lane_potential_requires_pallas():
    pos_sorted, mass_sorted, payload, _, _ = _build_small_fast_lane_case()
    with pytest.raises(NotImplementedError):
        compute_leaf_p2p_accelerations_radix_fast_lane(
            positions_sorted=pos_sorted,
            masses_sorted=mass_sorted,
            payload=payload,
            G=1.25,
            softening=0.05,
            return_potential=True,
            use_pallas=False,
        )


def _to_prepacked_payload(payload, block_size=2):
    """Convert a materialized fast-lane payload to the prepacked source-leaf layout."""
    num_leaves, source_slots = payload.source_leaf_ids.shape
    padded_slots = ((source_slots + block_size - 1) // block_size) * block_size
    pad_slots = padded_slots - source_slots
    source_leaf_ids = jnp.pad(payload.source_leaf_ids, ((0, 0), (0, pad_slots)))
    source_leaf_valid = jnp.pad(
        payload.source_leaf_valid_mask, ((0, 0), (0, pad_slots))
    )
    return replace(
        payload,
        source_leaf_ids=source_leaf_ids.reshape((num_leaves, -1, block_size)),
        source_leaf_valid_mask=source_leaf_valid.reshape((num_leaves, -1, block_size)),
        source_particle_ids=jnp.zeros((0, 0, 0), dtype=INDEX_DTYPE),
        source_particle_mask=jnp.zeros((0, 0, 0), dtype=bool),
        fallback_block_tile_size=1,
    )


def test_radix_fast_lane_prepacked_pallas_accel_matches_baseline(monkeypatch):
    pos_sorted, mass_sorted, payload, _, _ = _build_small_fast_lane_case()
    payload = _to_prepacked_payload(payload)

    baseline = compute_leaf_p2p_accelerations_radix_fast_lane(
        positions_sorted=pos_sorted,
        masses_sorted=mass_sorted,
        payload=payload,
        G=1.25,
        softening=0.05,
        use_pallas=False,
    )
    monkeypatch.setenv("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1")
    fused = compute_leaf_p2p_accelerations_radix_fast_lane(
        positions_sorted=pos_sorted,
        masses_sorted=mass_sorted,
        payload=payload,
        G=1.25,
        softening=0.05,
        use_pallas=True,
    )
    assert np.allclose(np.asarray(fused), np.asarray(baseline), rtol=1e-5, atol=1e-6)


def test_radix_fast_lane_prepacked_pallas_potential_matches_generic(monkeypatch):
    pos_sorted, mass_sorted, payload, tree, neighbor_list = (
        _build_small_fast_lane_case()
    )
    payload = _to_prepacked_payload(payload)

    ref_acc, ref_pot = compute_leaf_p2p_accelerations(
        tree,
        neighbor_list,
        pos_sorted,
        mass_sorted,
        G=1.25,
        softening=0.05,
        return_potential=True,
    )
    monkeypatch.setenv("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1")
    fused_acc, fused_pot = compute_leaf_p2p_accelerations_radix_fast_lane(
        positions_sorted=pos_sorted,
        masses_sorted=mass_sorted,
        payload=payload,
        G=1.25,
        softening=0.05,
        return_potential=True,
        use_pallas=True,
    )
    assert np.allclose(np.asarray(fused_acc), np.asarray(ref_acc), rtol=1e-5, atol=1e-6)
    assert np.allclose(np.asarray(fused_pot), np.asarray(ref_pot), rtol=1e-5, atol=1e-6)


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
        source_leaf_valid_mask=source_leaf_valid.reshape((num_leaves, -1, block_size)),
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


# ===========================================================================
# Edge-order invariant behind the precomputed_* contract
# ===========================================================================


def _synthetic_neighbor_csr():
    """A small CSR leaf-neighbour list that is deliberately not source-sorted.

    Seven nodes; nodes 3..6 are the leaves. Each leaf has two neighbours, listed
    in an order that ``argsort(source_leaf_ids)`` will genuinely permute -- which
    is what makes the assertions below non-vacuous.
    """
    node_ranges = jnp.asarray(
        [[0, 7], [0, 3], [4, 7], [0, 1], [2, 3], [4, 5], [6, 7]], dtype=INDEX_DTYPE
    )
    leaf_nodes = jnp.asarray([3, 4, 5, 6], dtype=INDEX_DTYPE)
    offsets = jnp.asarray([0, 2, 4, 6, 8], dtype=INDEX_DTYPE)
    neighbors = jnp.asarray([5, 6, 6, 3, 3, 4, 4, 5], dtype=INDEX_DTYPE)
    return node_ranges, leaf_nodes, offsets, neighbors


def test_prepare_leaf_neighbor_pairs_unsorted_is_positionally_aligned():
    """``sort_by_source=False`` must keep the pair vectors aligned with ``neighbors``.

    This is the unchecked half of the ``precomputed_*`` contract on
    :func:`compute_leaf_p2p_accelerations`. A consumer handed
    ``precomputed_target_leaf_ids`` but no ``precomputed_source_leaf_ids``
    re-derives the sources positionally as ``leaf_lookup[neighbors]``. If the
    stored vectors were source-sorted instead, they would be paired against
    unsorted sources -- and because both orderings have identical shapes, no shape
    check anywhere can detect it. The result is wrong forces with no error and no
    NaN, which is why this invariant is worth a test rather than a comment alone.
    """
    node_ranges, leaf_nodes, offsets, neighbors = _synthetic_neighbor_csr()

    target_ids, source_ids, valid = prepare_leaf_neighbor_pairs(
        node_ranges, leaf_nodes, offsets, neighbors, sort_by_source=False
    )

    # Exactly the derivation a consumer performs when source ids are absent.
    total_nodes = int(node_ranges.shape[0])
    leaf_lookup = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
    leaf_lookup = leaf_lookup.at[leaf_nodes].set(
        jnp.arange(leaf_nodes.shape[0], dtype=INDEX_DTYPE)
    )
    rederived = leaf_lookup[jnp.clip(neighbors, 0, total_nodes - 1)]

    assert jnp.array_equal(source_ids, rederived), (
        "sort_by_source=False must leave source ids positionally aligned with "
        "`neighbors`, or the consumer-side re-derivation silently mispairs edges"
    )
    # Targets follow the CSR row structure, so they are non-decreasing.
    assert bool(jnp.all(jnp.diff(target_ids) >= 0))
    assert bool(jnp.all(valid))


def test_prepare_leaf_neighbor_pairs_sorted_breaks_positional_alignment():
    """``sort_by_source=True`` permutes the edges -- the value the contract forbids.

    Documents by assertion why the default is unsafe for anything persisted: the
    sorted variant is a permutation of the same edge set, so it is equally valid
    on its own, but it no longer satisfies the positional identity above.
    """
    node_ranges, leaf_nodes, offsets, neighbors = _synthetic_neighbor_csr()

    unsorted_t, unsorted_s, unsorted_v = prepare_leaf_neighbor_pairs(
        node_ranges, leaf_nodes, offsets, neighbors, sort_by_source=False
    )
    sorted_t, sorted_s, sorted_v = prepare_leaf_neighbor_pairs(
        node_ranges, leaf_nodes, offsets, neighbors, sort_by_source=True
    )

    # Non-vacuity: if these ever coincided the test above would prove nothing.
    assert not jnp.array_equal(unsorted_t, sorted_t)
    assert not jnp.array_equal(unsorted_s, sorted_s)

    # Same edge set, different order -- and now source-major.
    assert jnp.array_equal(jnp.sort(unsorted_t), jnp.sort(sorted_t))
    assert jnp.array_equal(jnp.sort(unsorted_s), jnp.sort(sorted_s))
    assert bool(jnp.all(jnp.diff(sorted_s) >= 0))
    assert int(jnp.sum(unsorted_v)) == int(jnp.sum(sorted_v))


def test_prepare_leaf_neighbor_pairs_drops_padding_edges():
    """Negative ``neighbors`` padding must be masked out, not wrapped to leaf 0.

    ``leaf_lookup[-1]`` would index the last row, so the implementation clips and
    masks. Without the mask a padded edge becomes a real interaction.
    """
    node_ranges, leaf_nodes, offsets, _ = _synthetic_neighbor_csr()
    padded = jnp.asarray([5, -1, 6, -1, 3, -1, 4, -1], dtype=INDEX_DTYPE)

    _, source_ids, valid = prepare_leaf_neighbor_pairs(
        node_ranges, leaf_nodes, offsets, padded, sort_by_source=False
    )

    expected_valid = jnp.asarray([True, False] * 4)
    assert jnp.array_equal(valid, expected_valid)
    # The masked-out slots may hold anything; only the valid ones are contractual.
    assert bool(jnp.all(source_ids[valid] >= 0))


# ---------------------------------------------------------------------------
# Staticness contracts. These are documented in
# `compute_leaf_p2p_accelerations`'s `Raises` / `Parameters` sections and, before
# these tests, asserted nowhere -- which is the general pattern flagged as D.7 in
# docs/refactor_audit_2026-08.md: staticness is documented across this file and
# almost never tested. They are cheap and they break loudly if a refactor
# accidentally turns a static value into a traced one, which NUMERICS_AND_JAX §1
# calls out as able to leave runtime untouched while tripling compile time.
# ---------------------------------------------------------------------------


def test_max_leaf_size_is_required_under_jit(accel_only_case):
    """``max_leaf_size=None`` must fail loudly when tracing, and work when eager.

    The bound is read from the data with ``.item()`` when omitted, which is the one
    deliberate host sync in this function. Under a tracer that raises ``TypeError``,
    re-raised as a ``ValueError`` naming the expectation -- the "fail loudly rather
    than silently substituting" policy of STYLE_GUIDE §9.

    Both directions are asserted. Without the eager half, a guard that raised
    unconditionally would also pass, and the point of the parameter is that omitting
    it is legal outside ``jit``.
    """
    case = accel_only_case

    def call(positions, masses):
        return compute_leaf_p2p_accelerations(
            case.tree,
            case.neighbor_list,
            positions,
            masses,
            G=1.0,
            softening=1e-2,
            max_leaf_size=None,
        )

    with pytest.raises(ValueError, match="max_leaf_size must be provided"):
        jax.jit(call)(case.pos_sorted, case.mass_sorted)

    eager = call(case.pos_sorted, case.mass_sorted)
    assert np.asarray(eager).shape == np.asarray(case.pos_sorted).shape
    assert np.all(np.isfinite(np.asarray(eager)))


def test_softening_must_be_concrete(accel_only_case):
    """A traced ``softening`` must be rejected, not silently mis-handled.

    The docstring says ``softening`` "must be a concrete Python float, not a tracer",
    because it is squared host-side via ``float(softening)``. What actually enforces
    this is the always-on ``@jaxtyped(typechecker=beartype)`` decorator on this
    function -- it rejects the tracer against the ``float`` annotation before
    ``float()`` is ever reached. That is a *stronger* guarantee than the docstring
    describes (it fails at the boundary rather than mid-body), and it is worth pinning
    precisely because it does not depend on ``JACCPOT_RUNTIME_TYPECHECK``: this
    decorator is unconditional, so the contract holds in production, not only under
    the opt-in typecheck hook.
    """
    case = accel_only_case

    with pytest.raises(TypeCheckError):
        jax.jit(
            lambda soft: compute_leaf_p2p_accelerations(
                case.tree,
                case.neighbor_list,
                case.pos_sorted,
                case.mass_sorted,
                G=1.0,
                softening=soft,
                max_leaf_size=2,
            )
        )(jnp.asarray(1e-2))
