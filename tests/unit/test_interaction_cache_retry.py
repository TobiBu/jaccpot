"""Tests for dual-tree traversal capacity retry behavior."""

from types import SimpleNamespace

import jax.numpy as jnp

from jaccpot.runtime import _interaction_cache as interaction_cache


def test_build_dual_tree_artifacts_retries_on_capacity_overflow(monkeypatch):
    calls = []
    interactions = SimpleNamespace(
        sources=jnp.asarray([0], dtype=jnp.int32),
        targets=jnp.asarray([0], dtype=jnp.int32),
        level_offsets=None,
    )
    neighbor_list = SimpleNamespace()
    traversal_result = SimpleNamespace()

    def fake_build_interactions_and_neighbors(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "Pair queue capacity exceeded; increase max_pair_queue and rebuild."
            )
        return interactions, neighbor_list, traversal_result

    monkeypatch.setattr(
        "jaccpot.runtime.fmm.build_interactions_and_neighbors",
        fake_build_interactions_and_neighbors,
    )

    initial_cfg = interaction_cache.DualTreeTraversalConfig(
        max_pair_queue=1024,
        process_block=32,
        max_interactions_per_node=256,
        max_neighbors_per_leaf=128,
    )
    artifacts, _cache = interaction_cache._build_dual_tree_artifacts(
        tree=SimpleNamespace(),
        geometry=SimpleNamespace(),
        theta=0.6,
        mac_type="dehnen",
        dehnen_radius_scale=1.0,
        cache_key=None,
        cache_entry=None,
        max_pair_queue=None,
        pair_process_block=None,
        traversal_config=initial_cfg,
        retry_logger=None,
        fail_fast=False,
        use_dense_interactions=False,
        grouped_interactions=False,
        grouped_chunk_size=None,
        need_traversal_result=True,
        need_compact_far_pairs=False,
        need_node_interactions=True,
        precompute_grouped_class_segments=False,
        grouped_schedule_budget_bytes=None,
        pair_policy=None,
        policy_state=None,
    )

    assert len(calls) == 2
    retry_cfg = calls[1]["traversal_config"]
    assert int(retry_cfg.max_pair_queue) == 262_144
    assert int(retry_cfg.process_block) == 256
    assert int(retry_cfg.max_interactions_per_node) == 8192
    assert int(retry_cfg.max_neighbors_per_leaf) == 4096
    assert artifacts.interactions is interactions
    assert artifacts.neighbor_list is neighbor_list


def test_next_retry_traversal_settings_jump_to_retry_floor():
    next_cfg, next_queue, next_block = interaction_cache._next_retry_traversal_settings(
        traversal_config=interaction_cache.DualTreeTraversalConfig(
            max_pair_queue=32_768,
            process_block=64,
            max_interactions_per_node=1024,
            max_neighbors_per_leaf=256,
        ),
        max_pair_queue=None,
        pair_process_block=None,
    )

    assert next_queue == 262_144
    assert next_block == 256
    assert int(next_cfg.max_interactions_per_node) == 8192
    assert int(next_cfg.max_neighbors_per_leaf) == 4096


def test_build_dual_tree_artifacts_fail_fast_raises_hinted_capacity_error(
    monkeypatch,
):
    calls = []

    def fake_build_interactions_and_neighbors(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError(
            "Pair queue capacity exceeded; increase max_pair_queue and rebuild."
        )

    monkeypatch.setattr(
        "jaccpot.runtime.fmm.build_interactions_and_neighbors",
        fake_build_interactions_and_neighbors,
    )

    initial_cfg = interaction_cache.DualTreeTraversalConfig(
        max_pair_queue=32_768,
        process_block=64,
        max_interactions_per_node=1024,
        max_neighbors_per_leaf=256,
    )

    try:
        interaction_cache._build_dual_tree_artifacts(
            tree=SimpleNamespace(),
            geometry=SimpleNamespace(),
            theta=0.6,
            mac_type="dehnen",
            dehnen_radius_scale=1.0,
            cache_key=None,
            cache_entry=None,
            max_pair_queue=None,
            pair_process_block=None,
            traversal_config=initial_cfg,
            retry_logger=None,
            fail_fast=True,
            use_dense_interactions=False,
            grouped_interactions=False,
            grouped_chunk_size=None,
            need_traversal_result=False,
            need_compact_far_pairs=False,
            need_node_interactions=True,
            precompute_grouped_class_segments=False,
            grouped_schedule_budget_bytes=None,
            pair_policy=None,
            policy_state=None,
        )
    except RuntimeError as exc:
        msg = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError")

    assert len(calls) == 1
    assert "fail_fast enabled" in msg
    assert "max_pair_queue=32768" in msg
    assert "process_block=64" in msg
