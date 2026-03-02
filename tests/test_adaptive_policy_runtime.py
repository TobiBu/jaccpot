"""Unit tests for the solver-side adaptive traversal policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.runtime._adaptive_policy import (
    AdaptivePolicyState,
    adaptive_pair_policy,
    bucket_far_pairs_by_tag,
    dehnen_like_pair_error_by_order_from_degree_power,
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
        target_accept_threshold=jnp.asarray([0.25, 0.5], dtype=jnp.float32),
        order_tags=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        relaxed_theta_sq=jnp.asarray(0.8**2, dtype=jnp.float32),
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
        p_gears=(0, 1),
    )
    large = dehnen_like_pair_error_by_order_from_degree_power(
        degree_power=degree_power,
        opening=jnp.asarray([0.6], dtype=jnp.float32),
        p_gears=(0, 1),
    )

    assert np.all(np.asarray(small) <= np.asarray(large))
    assert float(large[0, 0]) >= float(large[0, 1])
