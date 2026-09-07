"""Unit tests for the solver-side adaptive traversal policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError
from yggdrax.tree import Tree

from jaccpot.runtime._adaptive_policy import (
    AdaptivePolicyState,
    adaptive_pair_policy,
    bucket_far_pairs_by_tag,
    compute_leaf_enclosing_sphere_geometry,
    compute_leaf_ritter_sphere_geometry,
    compute_node_force_scale_from_sorted_acc,
    compute_smallest_enclosing_sphere_geometry,
    compute_tree_merged_sphere_geometry,
    dehnen_like_pair_error_by_order_from_degree_power,
    dehnen_multipole_power_by_degree,
    dehnen_paper_pair_error_by_order,
    merge_bounding_spheres,
    source_error_proxy_by_order_from_degree_power,
    source_error_proxy_by_order_from_multipoles,
    source_power_by_degree_from_multipoles,
)


def _policy_state() -> AdaptivePolicyState:
    return AdaptivePolicyState(
        source_error_proxy_by_order=jnp.asarray(
            [
                [0.8, 0.2, 0.05],
                [0.5, 0.1, 0.01],
            ],
            dtype=jnp.float32,
        ),
        source_degree_power=jnp.asarray(
            [
                [0.64, 0.04, 0.0025],
                [0.25, 0.01, 0.0001],
            ],
            dtype=jnp.float32,
        ),
        source_dehnen_power=jnp.asarray(
            [
                [0.8, 0.4, 0.1],
                [0.5, 0.2, 0.05],
            ],
            dtype=jnp.float32,
        ),
        source_mass=jnp.asarray([1.0, 0.75], dtype=jnp.float32),
        source_mac_center=jnp.asarray(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32
        ),
        target_mac_center=jnp.asarray(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32
        ),
        source_radius_bound=jnp.asarray([0.5, 0.4], dtype=jnp.float32),
        target_radius_bound=jnp.asarray([0.5, 0.4], dtype=jnp.float32),
        target_accept_threshold=jnp.asarray([0.25, 0.5], dtype=jnp.float32),
        order_tags=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        order_values=jnp.asarray([2, 3, 4], dtype=jnp.int32),
        order_values_float=jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float32),
        dehnen_binomial_masked_by_order=jnp.asarray(
            [
                [1.0, 2.0, 1.0],
                [1.0, 3.0, 3.0],
                [1.0, 4.0, 6.0],
            ],
            dtype=jnp.float32,
        ),
        dehnen_exponent_by_order=jnp.asarray(
            [
                [2, 1, 0],
                [3, 2, 1],
                [4, 3, 2],
            ],
            dtype=jnp.int32,
        ),
        relaxed_theta_sq=jnp.asarray(0.8**2, dtype=jnp.float32),
        error_model_code=jnp.asarray(0, dtype=jnp.int32),
    )


def test_adaptive_pair_policy_supports_jit():
    state = _policy_state()

    @jax.jit
    def run(policy_state: AdaptivePolicyState):
        return adaptive_pair_policy(
            policy_state,
            valid_pairs=jnp.asarray([True, True, True]),
            mac_ok=jnp.asarray([True, False, True]),
            different_nodes=jnp.asarray([True, True, True]),
            target_leaf=jnp.asarray([False, True, False]),
            source_leaf=jnp.asarray([False, True, False]),
            same_node=jnp.asarray([False, False, False]),
            target_nodes=jnp.asarray([0, 1, 0], dtype=jnp.int32),
            source_nodes=jnp.asarray([0, 1, 1], dtype=jnp.int32),
            center_target=jnp.zeros((3, 3), dtype=jnp.float32),
            center_source=jnp.zeros((3, 3), dtype=jnp.float32),
            dist_sq=jnp.asarray([16.0, 4.0, 1.0], dtype=jnp.float32),
            extent_target=jnp.asarray([1.0, 0.5, 0.5], dtype=jnp.float32),
            extent_source=jnp.asarray([1.0, 0.5, 0.5], dtype=jnp.float32),
        )

    actions, tags = run(state)
    assert actions.shape == (3,)
    assert tags.shape == (3,)
    assert int(actions[0]) == 0
    assert int(tags[0]) == 1
    assert int(actions[1]) == 1
    assert int(tags[1]) == -1


def test_adaptive_pair_policy_rejects_all_false_pass_rows():
    state = _policy_state()
    actions, tags = adaptive_pair_policy(
        state,
        valid_pairs=jnp.asarray([True], dtype=jnp.bool_),
        mac_ok=jnp.asarray([True], dtype=jnp.bool_),
        different_nodes=jnp.asarray([True], dtype=jnp.bool_),
        target_leaf=jnp.asarray([False], dtype=jnp.bool_),
        source_leaf=jnp.asarray([False], dtype=jnp.bool_),
        same_node=jnp.asarray([False], dtype=jnp.bool_),
        target_nodes=jnp.asarray([0], dtype=jnp.int32),
        source_nodes=jnp.asarray([0], dtype=jnp.int32),
        center_target=jnp.zeros((1, 3), dtype=jnp.float32),
        center_source=jnp.zeros((1, 3), dtype=jnp.float32),
        dist_sq=jnp.asarray([0.01], dtype=jnp.float32),
        extent_target=jnp.asarray([1.0], dtype=jnp.float32),
        extent_source=jnp.asarray([1.0], dtype=jnp.float32),
    )

    assert int(actions[0]) == 2
    assert int(tags[0]) == -1


def test_bucket_far_pairs_by_tag_counts_match():
    buckets = bucket_far_pairs_by_tag(
        jnp.asarray([3, 4, 5, 6], dtype=jnp.int32),
        jnp.asarray([7, 8, 9, 10], dtype=jnp.int32),
        jnp.asarray([0, 2, 2, 1], dtype=jnp.int32),
        num_tags=3,
    )

    counts = [int(src.shape[0]) for src, _ in buckets]
    assert counts == [1, 1, 2]
    assert np.array_equal(np.asarray(buckets[2][0]), np.asarray([4, 5], dtype=np.int32))


def test_source_power_by_degree_matches_flat_tail_proxy():
    packed = jnp.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 0.25, 0.75, 1.25],
        ],
        dtype=jnp.float32,
    )
    degree_power = source_power_by_degree_from_multipoles(multipole_packed=packed)
    proxy_from_power = source_error_proxy_by_order_from_degree_power(
        degree_power=degree_power,
        p_gears=(0, 1),
    )
    proxy_from_packed = source_error_proxy_by_order_from_multipoles(
        multipole_packed=packed,
        p_gears=(0, 1),
    )

    assert degree_power.shape == (2, 2)
    assert np.allclose(np.asarray(proxy_from_power), np.asarray(proxy_from_packed))


def test_source_power_by_degree_accepts_the_complex_basis():
    """The complex basis is a live lane and hands these functions `c64`/`c128`.

    `dehnen_multipole_power_by_degree` already has its basis-invariance test; this
    covers the two entry points that did not, `source_power_by_degree_from_multipoles`
    and the `..._from_multipoles` proxy above it, whose annotations were derived from a
    pilot recording taken entirely on the real basis.
    """

    packed = jnp.asarray(
        [
            [1.0 + 0.0j, 2.0 - 1.0j, 3.0, 4.0 + 2.0j],
            [0.5, 0.25 + 0.25j, 0.75, 1.25 - 0.5j],
        ],
        dtype=jnp.complex64,
    )
    degree_power = source_power_by_degree_from_multipoles(multipole_packed=packed)

    # Per-degree power is sum |M|^2 over the degree's slice -- real for either basis.
    magnitudes_sq = np.abs(np.asarray(packed)) ** 2
    expected = np.stack(
        [magnitudes_sq[:, 0:1].sum(axis=1), magnitudes_sq[:, 1:4].sum(axis=1)],
        axis=1,
    )
    assert degree_power.shape == (2, 2)
    assert not np.iscomplexobj(np.asarray(degree_power))
    assert np.allclose(np.asarray(degree_power), expected)

    proxy_from_packed = source_error_proxy_by_order_from_multipoles(
        multipole_packed=packed,
        p_gears=(0, 1),
    )
    proxy_from_power = source_error_proxy_by_order_from_degree_power(
        degree_power=degree_power,
        p_gears=(0, 1),
    )
    assert np.allclose(np.asarray(proxy_from_power), np.asarray(proxy_from_packed))


def test_dehnen_like_pair_error_is_monotone_in_opening():
    degree_power = jnp.asarray([[1.0, 4.0, 9.0]], dtype=jnp.float32)
    small = dehnen_like_pair_error_by_order_from_degree_power(
        degree_power=degree_power,
        opening=jnp.asarray([0.2], dtype=jnp.float32),
        order_values=jnp.asarray([0, 1], dtype=jnp.int32),
    )
    large = dehnen_like_pair_error_by_order_from_degree_power(
        degree_power=degree_power,
        opening=jnp.asarray([0.6], dtype=jnp.float32),
        order_values=jnp.asarray([0, 1], dtype=jnp.int32),
    )

    assert np.all(np.asarray(small) <= np.asarray(large))
    assert float(large[0, 0]) >= float(large[0, 1])


def test_dehnen_degree_error_model_supports_jit():
    state = _policy_state()._replace(error_model_code=jnp.asarray(1, dtype=jnp.int32))

    @jax.jit
    def run(policy_state: AdaptivePolicyState):
        return adaptive_pair_policy(
            policy_state,
            valid_pairs=jnp.asarray([True, True], dtype=jnp.bool_),
            mac_ok=jnp.asarray([False, False], dtype=jnp.bool_),
            different_nodes=jnp.asarray([True, True], dtype=jnp.bool_),
            target_leaf=jnp.asarray([False, False], dtype=jnp.bool_),
            source_leaf=jnp.asarray([False, False], dtype=jnp.bool_),
            same_node=jnp.asarray([False, False], dtype=jnp.bool_),
            target_nodes=jnp.asarray([0, 1], dtype=jnp.int32),
            source_nodes=jnp.asarray([0, 1], dtype=jnp.int32),
            center_target=jnp.zeros((2, 3), dtype=jnp.float32),
            center_source=jnp.zeros((2, 3), dtype=jnp.float32),
            dist_sq=jnp.asarray([16.0, 16.0], dtype=jnp.float32),
            extent_target=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
            extent_source=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        )

    actions, tags = run(state)
    assert actions.shape == (2,)
    assert tags.shape == (2,)
    assert np.all(np.isfinite(np.asarray(tags)))


def test_dehnen_multipole_power_matches_degree0_mass():
    packed = jnp.asarray([[2.0, 3.0, 4.0, 5.0]], dtype=jnp.float32)
    power = dehnen_multipole_power_by_degree(multipole_packed=packed)

    assert power.shape == (1, 2)
    assert np.isclose(float(power[0, 0]), 2.0)


def test_dehnen_paper_error_is_monotone_in_distance():
    power = jnp.asarray([[1.0, 0.5, 0.25]], dtype=jnp.float32)
    order_values = jnp.asarray([1, 2], dtype=jnp.int32)
    order_values_float = order_values.astype(jnp.float32)
    degree_idx = jnp.arange(power.shape[1], dtype=jnp.int32)
    binom = jnp.asarray([[1.0, 1.0, 0.0], [1.0, 2.0, 1.0]], dtype=jnp.float32)
    include = degree_idx[None, :] <= order_values[:, None]
    masked_binom = binom * include.astype(jnp.float32)
    exponent = jnp.maximum(order_values[:, None] - degree_idx[None, :], 0)
    near = dehnen_paper_pair_error_by_order(
        source_power=power,
        source_mass=jnp.asarray([1.0], dtype=jnp.float32),
        source_radius=jnp.asarray([0.4], dtype=jnp.float32),
        target_radius=jnp.asarray([0.3], dtype=jnp.float32),
        distance=jnp.asarray([1.0], dtype=jnp.float32),
        order_values_float=order_values_float,
        masked_binomial_by_order=masked_binom,
        exponent_by_order=exponent,
    )
    far = dehnen_paper_pair_error_by_order(
        source_power=power,
        source_mass=jnp.asarray([1.0], dtype=jnp.float32),
        source_radius=jnp.asarray([0.4], dtype=jnp.float32),
        target_radius=jnp.asarray([0.3], dtype=jnp.float32),
        distance=jnp.asarray([2.0], dtype=jnp.float32),
        order_values_float=order_values_float,
        masked_binomial_by_order=masked_binom,
        exponent_by_order=exponent,
    )

    assert np.all(np.asarray(far) <= np.asarray(near))


def test_dehnen_paper_error_model_supports_jit():
    state = _policy_state()._replace(error_model_code=jnp.asarray(2, dtype=jnp.int32))

    @jax.jit
    def run(policy_state: AdaptivePolicyState):
        return adaptive_pair_policy(
            policy_state,
            valid_pairs=jnp.asarray([True, True], dtype=jnp.bool_),
            mac_ok=jnp.asarray([False, False], dtype=jnp.bool_),
            different_nodes=jnp.asarray([True, True], dtype=jnp.bool_),
            target_leaf=jnp.asarray([False, False], dtype=jnp.bool_),
            source_leaf=jnp.asarray([False, False], dtype=jnp.bool_),
            same_node=jnp.asarray([False, False], dtype=jnp.bool_),
            target_nodes=jnp.asarray([0, 1], dtype=jnp.int32),
            source_nodes=jnp.asarray([0, 1], dtype=jnp.int32),
            center_target=jnp.zeros((2, 3), dtype=jnp.float32),
            center_source=jnp.zeros((2, 3), dtype=jnp.float32),
            dist_sq=jnp.asarray([16.0, 16.0], dtype=jnp.float32),
            extent_target=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
            extent_source=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        )

    actions, tags = run(state)
    assert actions.shape == (2,)
    assert tags.shape == (2,)


def test_compute_smallest_enclosing_sphere_geometry_matches_simple_tetrahedron():
    centers, radii = compute_smallest_enclosing_sphere_geometry(
        node_ranges=jnp.asarray([[0, 3], [0, 1]], dtype=jnp.int32),
        positions_sorted=jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=jnp.float32,
        ),
    )

    assert np.allclose(
        np.asarray(centers[0]), np.asarray([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]), atol=1e-5
    )
    assert np.isclose(float(radii[0]), np.sqrt(8.0 / 3.0), atol=1e-5)
    assert np.allclose(np.asarray(centers[1]), np.asarray([1.0, 0.0, 0.0]), atol=1e-5)
    assert np.isclose(float(radii[1]), 1.0, atol=1e-5)


def test_merge_bounding_spheres_contains_inputs():
    center, radius = merge_bounding_spheres(
        jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([4.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.asarray(1.5, dtype=jnp.float32),
    )

    assert np.allclose(np.asarray(center), np.asarray([2.25, 0.0, 0.0]), atol=1e-5)
    assert np.isclose(float(radius), 3.25, atol=1e-5)


def test_tree_merged_sphere_geometry_contains_exact_leaf_spheres():
    positions = jnp.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    masses = jnp.ones((4,), dtype=jnp.float32)
    tree = Tree.from_particles(
        positions,
        masses,
        leaf_size=2,
        tree_type="radix",
        target_leaf_particles=2,
        refine_local=False,
    )
    positions_sorted = positions[tree.particle_indices]

    leaf_centers, leaf_radii = compute_leaf_enclosing_sphere_geometry(
        tree=tree, positions_sorted=positions_sorted
    )
    centers, radii = compute_tree_merged_sphere_geometry(
        tree=tree, positions_sorted=positions_sorted
    )

    num_internal = int(tree.num_internal_nodes)
    leaf_nodes = np.arange(num_internal, int(tree.parent.shape[0]))
    assert np.allclose(
        np.asarray(centers[leaf_nodes]), np.asarray(leaf_centers[leaf_nodes]), atol=1e-5
    )
    assert np.allclose(
        np.asarray(radii[leaf_nodes]), np.asarray(leaf_radii[leaf_nodes]), atol=1e-5
    )
    for idx in leaf_nodes:
        child_dist = np.linalg.norm(np.asarray(centers[idx]) - np.asarray(centers[0]))
        assert child_dist + float(radii[idx]) <= float(radii[0]) + 1e-5


def test_compute_node_force_scale_matches_reference_reductions():
    positions = jnp.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    masses = jnp.ones((4,), dtype=jnp.float32)
    tree = Tree.from_particles(
        positions,
        masses,
        leaf_size=2,
        tree_type="radix",
        target_leaf_particles=2,
        refine_local=False,
    )
    accelerations_sorted = jnp.asarray(
        [[3.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )[tree.particle_indices]

    def reference(reduction: str) -> np.ndarray:
        mags = np.linalg.norm(np.asarray(accelerations_sorted), axis=1)
        values = np.zeros((int(tree.node_ranges.shape[0]),), dtype=mags.dtype)
        for idx, (start, end) in enumerate(np.asarray(tree.node_ranges)):
            s = int(start)
            e = int(end)
            vals = mags[s : e + 1]
            values[idx] = (
                float(np.min(vals)) if reduction == "min" else float(np.max(vals))
            )
        return values

    computed_max = compute_node_force_scale_from_sorted_acc(
        tree=tree,
        accelerations_sorted=accelerations_sorted,
        reduction="max",
    )
    computed_min = compute_node_force_scale_from_sorted_acc(
        tree=tree,
        accelerations_sorted=accelerations_sorted,
        reduction="min",
    )

    assert np.allclose(np.asarray(computed_max), reference("max"))
    assert np.allclose(np.asarray(computed_min), reference("min"))


def test_compute_node_force_scale_root_matches_global_particle_extrema():
    key = jax.random.PRNGKey(77)
    positions = jax.random.uniform(
        key,
        (128, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.ones((128,), dtype=jnp.float32)
    tree = Tree.from_particles(
        positions,
        masses,
        leaf_size=8,
        tree_type="radix",
        target_leaf_particles=8,
        refine_local=False,
    )
    accel_key = jax.random.PRNGKey(78)
    accelerations_sorted = jax.random.normal(
        accel_key,
        (128, 3),
        dtype=jnp.float32,
    )[tree.particle_indices]
    magnitudes = np.linalg.norm(np.asarray(accelerations_sorted), axis=1)

    computed_max = compute_node_force_scale_from_sorted_acc(
        tree=tree,
        accelerations_sorted=accelerations_sorted,
        reduction="max",
    )
    computed_min = compute_node_force_scale_from_sorted_acc(
        tree=tree,
        accelerations_sorted=accelerations_sorted,
        reduction="min",
    )

    assert np.isclose(float(computed_max[0]), float(np.max(magnitudes)))
    assert np.isclose(float(computed_min[0]), float(np.min(magnitudes)))
    assert float(np.min(np.asarray(computed_min))) > 0.0


def test_leaf_ritter_sphere_contains_leaf_points():
    positions = jnp.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    masses = jnp.ones((4,), dtype=jnp.float32)
    tree = Tree.from_particles(
        positions,
        masses,
        leaf_size=2,
        tree_type="radix",
        target_leaf_particles=2,
        refine_local=False,
    )
    positions_sorted = positions[tree.particle_indices]
    centers, radii = compute_leaf_ritter_sphere_geometry(
        tree=tree, positions_sorted=positions_sorted
    )
    num_internal = int(tree.num_internal_nodes)
    node_ranges = np.asarray(tree.node_ranges)
    for node_idx in range(num_internal, int(tree.parent.shape[0])):
        start, end = node_ranges[node_idx]
        pts = np.asarray(positions_sorted[start : end + 1])
        d = np.linalg.norm(pts - np.asarray(centers[node_idx])[None, :], axis=1)
        assert np.all(d <= float(radii[node_idx]) + 1e-5)


# The force-scale pair, whose `nodes` axis the re-recorded pilot found unconstrained.
# `node_centers` and `node_radii` are one axis: 5 acceptances on `_far_field_force_scale_by_node`
# and 5 on `estimate_particle_force_scale`, and a `node_radii` that disagrees with
# `node_centers` silently rescales every node's far-field contribution.


def _real_tree(n: int = 64, leaf_size: int = 8):
    """Build a real yggdrax tree, because zeros are not a tree.

    Parameters
    ----------
    n : int
        Particle count.
    leaf_size : int
        Maximum leaf occupancy.

    Returns
    -------
    tuple
        `(tree, positions_sorted, masses_sorted)`.
    """
    from yggdrax.tree import build_tree

    key = jax.random.PRNGKey(0)
    positions = jax.random.uniform(key, (n, 3), dtype=jnp.float64) * 2.0 - 1.0
    masses = jnp.ones((n,), dtype=jnp.float64)
    bounds = (jnp.full((3,), -1.0), jnp.full((3,), 1.0))
    tree, positions_sorted, masses_sorted, _ = build_tree(
        positions, masses, bounds, leaf_size=leaf_size, return_reordered=True
    )
    return tree, positions_sorted, masses_sorted


def _far_field_args():
    """Build one valid `_far_field_force_scale_by_node` call.

    Returns
    -------
    dict
        Keyword arguments, with `node_centers` and `node_radii` agreeing on `nodes`.
    """
    from jaccpot.runtime._adaptive_policy import _far_field_force_scale_by_node

    tree, _, masses_sorted = _real_tree()
    nodes = int(tree.num_nodes)
    return _far_field_force_scale_by_node, {
        "tree": tree,
        "masses": masses_sorted,
        "node_centers": jnp.zeros((nodes, 3), dtype=jnp.float64),
        "node_radii": jnp.ones((nodes,), dtype=jnp.float64),
        "interaction_sources": jnp.zeros((2,), dtype=jnp.int32),
        "interaction_targets": jnp.ones((2,), dtype=jnp.int32),
        "g": jnp.asarray(1.0),
        "eps_sq": jnp.asarray(0.0),
        "inflation": jnp.asarray(1.0),
    }


def test_the_far_field_force_scale_still_runs_on_a_real_tree():
    """The control. Every rejection below is worthless without it."""
    fn, args = _far_field_args()
    out = fn(**args)
    assert out.shape == (int(args["node_centers"].shape[0]),)


def test_node_radii_that_disagree_with_node_centers_are_rejected():
    """One axis, not two: a short `node_radii` rescales every node's far field."""
    fn, args = _far_field_args()
    args["node_radii"] = args["node_radii"][:-1]
    with pytest.raises(TypeCheckError):
        fn(**args)


def test_a_two_component_node_centre_is_rejected():
    """The spatial literal the centre-to-centre displacement depends on."""
    fn, args = _far_field_args()
    args["node_centers"] = args["node_centers"][:, :-1]
    with pytest.raises(TypeCheckError):
        fn(**args)


def test_a_batched_mass_vector_is_rejected():
    """`masses` is `n`; its LENGTH stays free because `n` occurs once here, but rank does not.

    That distinction is the point of the note above `resolve_dehnen_geometry`: a
    single-occurrence axis asserts nothing about length, so the leading-axis acceptance the
    pilot reports on `masses` is not closable on this parameter -- only its rank is.
    """
    fn, args = _far_field_args()
    args["masses"] = args["masses"][None]
    with pytest.raises(TypeCheckError):
        fn(**args)


# Slices 2 and 3. Every function below was bare and indicted by the re-record, and these
# three carry a real cross-parameter equality rather than rank protection alone.

NODES, LEAVES, W, DEGREES = 5, 4, 8, 3


def test_a_leaf_mask_that_disagrees_with_its_points_is_rejected():
    """`leaf_points` and `leaf_valid` are one `leaves w`, observed at (64, 8) and (32, 16).

    A mask that disagrees with its points silently includes padding slots in a bounding
    sphere, which inflates the node radius the MAC then trusts.
    """
    from jaccpot.runtime._adaptive_policy import _batched_ritter_leaf_spheres

    points = jnp.zeros((LEAVES, W, 3), dtype=jnp.float64)
    valid = jnp.ones((LEAVES, W), dtype=bool)
    centers, radii = _batched_ritter_leaf_spheres(points, valid)
    assert centers.shape == (LEAVES, 3) and radii.shape == (LEAVES,)

    with pytest.raises(TypeCheckError):
        _batched_ritter_leaf_spheres(points, valid[:, :-1])
    with pytest.raises(TypeCheckError):
        _batched_ritter_leaf_spheres(points, valid[:-1])


def test_theta_nodes_must_agree_with_the_radius_it_rescales():
    """The equality is structural, not a coincidence of one observed extent.

    `fmm_policy.py` calls `per_node_effective_theta(radius_bound=...)` and then
    `per_node_mac_radius(radius_bound=<the same array>, theta_nodes=<that return>)`, so
    the two share a length by construction. Only one extent (511) was ever recorded, so
    the call site is the evidence rather than the sample.
    """
    from jaccpot.runtime._adaptive_policy import per_node_mac_radius

    radius = jnp.ones((NODES,), dtype=jnp.float64)
    theta = jnp.full((NODES,), 0.5, dtype=jnp.float64)
    assert per_node_mac_radius(
        radius_bound=radius, theta_nodes=theta, theta_global=0.5
    ).shape == (NODES,)

    with pytest.raises(TypeCheckError):
        per_node_mac_radius(
            radius_bound=radius, theta_nodes=theta[:-1], theta_global=0.5
        )


def test_the_per_order_tables_must_agree_with_the_power_on_degrees():
    """`masked_binomial` and `exponent` are a row of an `(orders, degrees)` table.

    `fmm_policy.py` passes `dehnen_binomial_masked_by_order[0]` and
    `dehnen_exponent_by_order[0]`, whose length is therefore `source_power`'s trailing
    axis. Getting that wrong contracts the error model against the wrong degree count.
    """
    from jaccpot.runtime._adaptive_policy import per_node_effective_theta

    kwargs = {
        "source_power": jnp.ones((NODES, DEGREES), dtype=jnp.float64),
        "radius_bound": jnp.ones((NODES,), dtype=jnp.float64),
        "force_scale": jnp.ones((NODES,), dtype=jnp.float64),
        "masked_binomial": jnp.ones((DEGREES,), dtype=jnp.float64),
        "exponent": jnp.arange(DEGREES, dtype=jnp.int32),
        "order": 2,
        "eps": 1e-3,
    }
    assert per_node_effective_theta(**kwargs).shape == (NODES,)

    short = dict(kwargs, masked_binomial=kwargs["masked_binomial"][:-1])
    with pytest.raises(TypeCheckError):
        per_node_effective_theta(**short)

    with pytest.raises(TypeCheckError):
        per_node_effective_theta(**dict(kwargs, exponent=kwargs["exponent"][:-1]))
