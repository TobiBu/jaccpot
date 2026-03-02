"""Unit tests for the solver-side adaptive traversal policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.runtime._adaptive_policy import (
    AdaptivePolicyState,
    adaptive_pair_policy,
    bucket_far_pairs_by_tag,
    compute_smallest_enclosing_sphere_geometry,
    dehnen_like_pair_error_by_order_from_degree_power,
    dehnen_multipole_power_by_degree,
    dehnen_paper_pair_error_by_order,
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
        dehnen_binomial_by_order=jnp.asarray(
            [
                [1.0, 2.0, 1.0],
                [1.0, 3.0, 3.0],
                [1.0, 4.0, 6.0],
            ],
            dtype=jnp.float32,
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
    binom = jnp.asarray([[1.0, 1.0, 0.0], [1.0, 2.0, 1.0]], dtype=jnp.float32)
    near = dehnen_paper_pair_error_by_order(
        source_power=power,
        source_mass=jnp.asarray([1.0], dtype=jnp.float32),
        source_radius=jnp.asarray([0.4], dtype=jnp.float32),
        target_radius=jnp.asarray([0.3], dtype=jnp.float32),
        distance=jnp.asarray([1.0], dtype=jnp.float32),
        order_values=order_values,
        binomial_by_order=binom,
    )
    far = dehnen_paper_pair_error_by_order(
        source_power=power,
        source_mass=jnp.asarray([1.0], dtype=jnp.float32),
        source_radius=jnp.asarray([0.4], dtype=jnp.float32),
        target_radius=jnp.asarray([0.3], dtype=jnp.float32),
        distance=jnp.asarray([2.0], dtype=jnp.float32),
        order_values=order_values,
        binomial_by_order=binom,
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
