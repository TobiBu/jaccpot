"""The flat-walk seam of the strict fused lane: same lists as the dual walk, as sets.

``_build_flat_walk_artifacts_strict_streamed`` (plan "tree walk", 2026-09-10)
replaces yggdrax's traced dual-tree walk with its flat-emission
``dual_tree_walk_mutual`` fed the dual walk's own ``mac_extents`` and
``mac_type``. The contract pinned here on a real tree, on CPU, every commit:

* far pairs are DIRECTED and PREFIX-LIVE (every consumer masks
  ``idx < far_pair_count``), each canonical pair present in both directions, the
  tail ``-1``;
* the far set equals the dual walk's far set, the per-leaf neighbour sets equal
  the dual walk's, no self neighbour, no duplicate ``(leaf, neighbour)``;
* the neighbour CSR is valid (offsets monotone, counts consistent, width equal to
  the edge cap);
* the capacity report is emitted with ``peak_wavefront`` and marks the lane;
* a too-small far or near capacity raises eagerly, naming the knob.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax import DualTreeTraversalConfig
from yggdrax._geometry_impl import compute_tree_geometry
from yggdrax._interactions_impl import build_interactions_and_neighbors
from yggdrax.tree import Tree

from jaccpot.runtime._interaction_cache import (
    _build_flat_walk_artifacts_strict_streamed,
)

_LEAF = 8


@pytest.fixture(scope="module")
def tree_and_geometry():
    points = jax.random.uniform(jax.random.PRNGKey(3), (512, 3), dtype=jnp.float64)
    masses = jnp.ones((512,), dtype=jnp.float64)
    tree = Tree.from_particles(points, masses, leaf_size=_LEAF, tree_type="radix")
    geometry = compute_tree_geometry(
        tree.topology, tree.positions_sorted, max_leaf_size=_LEAF
    )
    num_internal = int(tree.topology.left_child.shape[0])
    total_nodes = int(tree.topology.parent.shape[0])
    return tree, geometry, num_internal, total_nodes


def _flat(
    tree,
    geometry,
    *,
    theta=0.5,
    mac_type="dehnen",
    scale=1.0,
    far_cap=1 << 15,
    near_cap=1 << 15,
    queue=1 << 14,
    report=None,
):
    return _build_flat_walk_artifacts_strict_streamed(
        tree=tree,
        geometry=geometry,
        theta=theta,
        mac_type=mac_type,
        dehnen_radius_scale=scale,
        compact_far_pair_capacity=far_cap,
        near_edge_capacity=near_cap,
        max_pair_queue=queue,
        capacity_report=report,
    )


def _dual_sets(tree, geometry, *, theta, mac_type, scale=1.0):
    config = DualTreeTraversalConfig(
        max_pair_queue=1 << 15,
        process_block=64,
        max_interactions_per_node=2048,
        max_neighbors_per_leaf=2048,
    )
    _i, neighbors, result = build_interactions_and_neighbors(
        tree.topology,
        geometry,
        theta=theta,
        traversal_config=config,
        mac_type=mac_type,
        dehnen_radius_scale=scale,
        return_result=True,
    )
    assert not (
        bool(result.queue_overflow)
        or bool(result.far_overflow)
        or bool(result.near_overflow)
    )
    src = np.asarray(result.interaction_sources)
    tgt = np.asarray(result.interaction_targets)
    live = (src >= 0) & (tgt >= 0)
    far = set(zip(tgt[live].tolist(), src[live].tolist()))
    offsets, counts = np.asarray(neighbors.offsets), np.asarray(neighbors.counts)
    nbrs, leaves = np.asarray(neighbors.neighbors), np.asarray(neighbors.leaf_indices)
    near = {}
    for row, leaf in enumerate(leaves.tolist()):
        near[leaf] = {int(nbrs[int(offsets[row]) + k]) for k in range(int(counts[row]))}
    return far, near


def _flat_far_set(cfp):
    n = int(cfp.far_pair_count)
    src, tgt = np.asarray(cfp.sources), np.asarray(cfp.targets)
    assert np.all(src[:n] >= 0) and np.all(tgt[:n] >= 0), "live prefix has no padding"
    assert np.all(src[n:] == -1) and np.all(tgt[n:] == -1), "-1 tail after the prefix"
    pairs = list(zip(tgt[:n].tolist(), src[:n].tolist()))
    assert len(set(pairs)) == len(pairs), "duplicate directed far pair"
    return set(pairs)


def _flat_near_sets(nl, num_internal, total_nodes):
    offsets, counts = np.asarray(nl.offsets), np.asarray(nl.counts)
    nbrs, leaves = np.asarray(nl.neighbors), np.asarray(nl.leaf_indices)
    assert np.array_equal(leaves, np.arange(num_internal, total_nodes))
    assert np.all(np.diff(offsets) >= 0) and np.array_equal(
        offsets[1:] - offsets[:-1], counts
    )
    near = {}
    for row, leaf in enumerate(leaves.tolist()):
        block = nbrs[int(offsets[row]) : int(offsets[row]) + int(counts[row])].tolist()
        assert leaf not in block, "self neighbour"
        assert len(set(block)) == len(block), "duplicate neighbour"
        assert all(
            num_internal <= b < total_nodes for b in block
        ), "neighbour is not a leaf"
        near[leaf] = set(block)
    return near


@pytest.mark.parametrize("mac_type", ["bh", "dehnen"])
@pytest.mark.parametrize("theta", [0.3, 0.5, 0.9])
def test_flat_walk_lists_equal_the_dual_walk_as_sets(
    tree_and_geometry, mac_type, theta
):
    tree, geometry, num_internal, total_nodes = tree_and_geometry
    far_ref, near_ref = _dual_sets(tree, geometry, theta=theta, mac_type=mac_type)
    reports = []
    art = _flat(tree, geometry, theta=theta, mac_type=mac_type, report=reports.append)
    assert art.interactions is None and art.traversal_result is None
    far = _flat_far_set(art.compact_far_pairs)
    assert far == far_ref
    assert all((b, a) in far for a, b in far), "every far pair in both directions"
    near = _flat_near_sets(art.neighbor_list, num_internal, total_nodes)
    assert near == near_ref
    assert int(art.neighbor_list.neighbors.shape[0]) == 1 << 15  # edge-cap width
    (report,) = reports
    assert report["flat_walk"] is True and report["traced"] is False
    assert report["peak_wavefront"] > 0 and report["rounds"] > 0
    assert report["far_pair_count"] == len(far)
    assert report["total_neighbors"] == sum(len(v) for v in near.values())
    assert report["max_neighbors_per_leaf_used"] is None


def test_dehnen_radius_scale_is_honoured(tree_and_geometry):
    tree, geometry, num_internal, total_nodes = tree_and_geometry
    far_ref, near_ref = _dual_sets(
        tree, geometry, theta=0.5, mac_type="dehnen", scale=1.4
    )
    art = _flat(tree, geometry, theta=0.5, mac_type="dehnen", scale=1.4)
    assert _flat_far_set(art.compact_far_pairs) == far_ref
    assert _flat_near_sets(art.neighbor_list, num_internal, total_nodes) == near_ref
    far_unscaled, _ = _dual_sets(
        tree, geometry, theta=0.5, mac_type="dehnen", scale=1.0
    )
    assert far_unscaled != far_ref


def test_traced_call_keeps_capacity_width_and_reports_traced(tree_and_geometry):
    tree, geometry, num_internal, total_nodes = tree_and_geometry
    reports = []
    eager = _flat(tree, geometry, report=reports.append)

    def run(positions_sorted):
        geom = compute_tree_geometry(
            tree.topology, positions_sorted, max_leaf_size=_LEAF
        )
        art = _flat(tree, geom, report=reports.append)
        return (
            art.compact_far_pairs.sources,
            art.compact_far_pairs.targets,
            art.compact_far_pairs.far_pair_count,
            art.neighbor_list.neighbors,
            art.neighbor_list.counts,
        )

    src, tgt, n, nbrs, counts = jax.jit(run)(tree.positions_sorted)
    assert int(n) == int(eager.compact_far_pairs.far_pair_count)
    assert src.shape == eager.compact_far_pairs.sources.shape
    assert nbrs.shape == eager.neighbor_list.neighbors.shape
    assert np.array_equal(np.asarray(counts), np.asarray(eager.neighbor_list.counts))
    assert set(
        zip(np.asarray(tgt)[: int(n)].tolist(), np.asarray(src)[: int(n)].tolist())
    ) == _flat_far_set(eager.compact_far_pairs)
    assert [r["traced"] for r in reports] == [False, True]
    assert "peak_wavefront" not in reports[1]


def test_far_and_near_overflow_raise_eagerly_naming_the_knob(tree_and_geometry):
    tree, geometry, _, _ = tree_and_geometry
    with pytest.raises(RuntimeError, match="COMPACT_FAR_PAIR_CAP"):
        _flat(tree, geometry, far_cap=8)
    with pytest.raises(RuntimeError, match="NEIGHBOR_EDGE_PROFILE_FIXED_CAP"):
        _flat(tree, geometry, near_cap=8)


def test_odd_capacities_are_rejected(tree_and_geometry):
    tree, geometry, _, _ = tree_and_geometry
    with pytest.raises(ValueError, match="even"):
        _flat(tree, geometry, far_cap=(1 << 15) + 1)
    with pytest.raises(ValueError, match="even"):
        _flat(tree, geometry, near_cap=(1 << 15) + 1)


def test_queue_ladder_grows_from_a_small_queue(tree_and_geometry):
    tree, geometry, _, _ = tree_and_geometry
    reports = []
    art = _flat(tree, geometry, queue=8, report=reports.append)
    (report,) = reports
    assert report["queue_capacity"] > 8 and report["grew"], report
    assert int(art.compact_far_pairs.far_pair_count) > 0
