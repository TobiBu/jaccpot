"""Tests for dual-tree traversal capacity retry behavior."""

from types import SimpleNamespace

import jax.numpy as jnp
from yggdrax.geometry import compute_tree_geometry
from yggdrax.tree import build_tree

from jaccpot.runtime import _interaction_cache as interaction_cache


def _real_tree_and_geometry():
    """A real ``Tree`` and ``TreeGeometry``, not empty namespaces.

    These tests monkeypatch ``build_interactions_and_neighbors``, so the tree is
    never read -- which is exactly why ``SimpleNamespace()`` used to be passed. It
    made the tests silently violate ``_build_dual_tree_artifacts``'s declared
    ``tree: Tree`` / ``geometry: TreeGeometry`` contract, and it was part of what
    made ``JACCPOT_RUNTIME_TYPECHECK=1 pytest tests/unit`` red (F40). Eight
    particles is enough and costs milliseconds.
    """
    positions = jnp.asarray(
        [
            [-0.8, 0.1, 0.0],
            [-0.9, -0.1, 0.05],
            [0.7, 0.0, -0.05],
            [0.9, -0.1, 0.1],
            [0.2, 0.3, -0.2],
            [-0.2, -0.3, 0.2],
            [0.4, -0.4, 0.4],
            [-0.4, 0.4, -0.4],
        ]
    )
    masses = jnp.ones((positions.shape[0],))
    bounds = (jnp.array([-1.0, -1.0, -1.0]), jnp.array([1.0, 1.0, 1.0]))
    tree, pos_sorted, _mass_sorted, _perm = build_tree(
        positions, masses, bounds, return_reordered=True, leaf_size=2
    )
    return tree, compute_tree_geometry(tree, pos_sorted)


def _real_build_outputs(tree, geometry):
    """Genuine builder outputs, from one real build on the same eight particles.

    The retry test needs three objects to hand back from its fake builder, and it
    asserts *identity* on two of them -- so as far as those assertions go, any object
    would do. That is why ``SimpleNamespace`` was enough, and it is also what kept
    ``_dual_tree_unpack_build_output``'s declared return type violated under
    ``JACCPOT_RUNTIME_TYPECHECK=1`` (F40): the namespaces travel through it on their
    way into ``_DualTreeArtifacts``.

    Building the real types by hand is not worth it -- ``NodeInteractionList``,
    ``NodeNeighborList`` and ``DualTreeWalkResult`` are NamedTuples with 6, 12 and 17
    fields and **no** defaults, so 35 hand-written fields would dwarf the test and
    assert nothing. One real build costs milliseconds and is correct by construction.
    """
    artifacts, _cache = interaction_cache._build_dual_tree_artifacts(
        tree=tree,
        geometry=geometry,
        theta=0.6,
        mac_type="dehnen",
        dehnen_radius_scale=1.0,
        cache_key=None,
        cache_entry=None,
        max_pair_queue=None,
        pair_process_block=None,
        traversal_config=None,
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
    return artifacts.interactions, artifacts.neighbor_list, artifacts.traversal_result


def test_build_dual_tree_artifacts_retries_on_capacity_overflow(monkeypatch):
    tree, geometry = _real_tree_and_geometry()
    calls = []
    # Real objects, not namespaces -- see `_real_build_outputs`. Captured before the
    # monkeypatch so the real builder is still installed when they are made.
    interactions, neighbor_list, traversal_result = _real_build_outputs(tree, geometry)

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
        tree=tree,
        geometry=geometry,
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
    tree, geometry = _real_tree_and_geometry()
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
            tree=tree,
            geometry=geometry,
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
