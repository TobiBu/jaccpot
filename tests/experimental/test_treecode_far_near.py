"""Structural correctness of the treecode far-COO + near-CSR producer.

Locks the compaction against the raw per-leaf walk (interpret mode, CPU): the far
COO must be the walk's far list grouped by target leaf (targets all leaves), and
the near CSR must be the walk's near list with each leaf's own id removed. Physics
(these lists -> M2L/L2P/P2P == direct N-body) is covered by
``test_treecode_vs_direct.py`` and, for the solidfmm fast-lane, by the on-device
graft test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax._geometry_impl import compute_tree_geometry
from yggdrax.tree import Tree

from jaccpot.experimental.treecode_far_near import (
    build_treecode_far_pairs_and_neighbors,
)
from jaccpot.experimental.treecode_walk import treecode_leaf_walk


def _bundle(n, seed, leaf_size, theta):
    key = jax.random.PRNGKey(seed)
    pts = jax.random.uniform(key, (n, 3), dtype=jnp.float64)
    mass = jnp.ones((n,), dtype=jnp.float64)
    tree = Tree.from_particles(pts, mass, leaf_size=leaf_size)
    topo = tree.topology
    pos_sorted = getattr(tree, "positions_sorted", pts)
    geom = compute_tree_geometry(topo, pos_sorted, max_leaf_size=leaf_size)
    num_internal = int(topo.num_internal_nodes)
    total_nodes = int(topo.parent.shape[0])
    num_leaves = total_nodes - num_internal
    idx = topo.parent.dtype
    left_full = jnp.concatenate(
        [topo.left_child.astype(idx), jnp.full((num_leaves,), -1, dtype=idx)]
    )
    right_full = jnp.concatenate(
        [topo.right_child.astype(idx), jnp.full((num_leaves,), -1, dtype=idx)]
    )
    leaf_nodes = jnp.arange(num_internal, total_nodes, dtype=idx)
    root_idx = jnp.argmin(topo.parent).astype(idx)
    return dict(
        leaf_nodes=leaf_nodes,
        centers=geom.center,
        mac_extents=geom.max_extent,
        left_full=left_full,
        right_full=right_full,
        theta_sq=jnp.asarray(theta * theta, dtype=jnp.float64),
        root_idx=root_idx,
        num_internal=num_internal,
        total_nodes=total_nodes,
        num_leaves=num_leaves,
    )


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs float64 (JAX_ENABLE_X64=1)"
)
@pytest.mark.parametrize(
    ("n", "seed", "leaf_size", "theta"),
    [(256, 3, 8, 0.5), (256, 3, 8, 0.9), (500, 7, 16, 0.6)],
)
def test_producer_matches_raw_walk(n, seed, leaf_size, theta):
    b = _bundle(n, seed, leaf_size, theta)
    cap = b["total_nodes"]
    common = dict(
        num_internal=b["num_internal"],
        max_far=cap,
        max_near=cap,
        max_stack=2 * cap + 4,
        max_iters=b["total_nodes"] + 1,
    )
    walk = treecode_leaf_walk(
        b["leaf_nodes"],
        b["centers"],
        b["mac_extents"],
        b["left_full"],
        b["right_full"],
        b["theta_sq"],
        b["root_idx"],
        **common,
    )
    prod = build_treecode_far_pairs_and_neighbors(
        b["leaf_nodes"],
        b["centers"],
        b["mac_extents"],
        b["left_full"],
        b["right_full"],
        b["theta_sq"],
        b["root_idx"],
        far_pair_capacity=b["num_leaves"] * cap,
        near_capacity=b["num_leaves"] * cap,
        interpret=True,
        **common,
    )
    assert not bool(prod.overflow)

    far_nodes = np.asarray(walk.far_nodes)
    far_count = np.asarray(walk.far_count)
    near_leaves = np.asarray(walk.near_leaves)
    near_count = np.asarray(walk.near_count)
    leaf_ids = np.asarray(b["leaf_nodes"])
    num_leaves = b["num_leaves"]

    # ---- FAR: flat COO grouped by target leaf, targets all leaves ----
    fpc = int(np.asarray(prod.far_pair_count))
    assert fpc == int(far_count.sum())
    exp_src, exp_tgt = [], []
    for i in range(num_leaves):
        row = far_nodes[i, : int(far_count[i])]
        exp_src.append(row)
        exp_tgt.append(np.full(row.shape[0], leaf_ids[i]))
    exp_src = np.concatenate(exp_src) if exp_src else np.array([], dtype=np.int64)
    exp_tgt = np.concatenate(exp_tgt) if exp_tgt else np.array([], dtype=np.int64)
    got_src = np.asarray(prod.far_sources)[:fpc]
    got_tgt = np.asarray(prod.far_targets)[:fpc]
    assert np.array_equal(got_src, exp_src)
    assert np.array_equal(got_tgt, exp_tgt)
    assert (got_tgt >= b["num_internal"]).all(), "far targets must all be leaves"
    assert np.all(np.asarray(prod.far_tags)[:fpc] == -1)

    # ---- NEAR: CSR with self excluded ----
    offsets = np.asarray(prod.near_offsets)
    neighbors = np.asarray(prod.near_neighbors)
    counts = np.asarray(prod.near_counts)
    assert np.array_equal(np.asarray(prod.near_leaf_indices), leaf_ids)
    assert offsets[0] == 0 and np.array_equal(np.diff(offsets), counts)
    for i in range(num_leaves):
        raw = near_leaves[i, : int(near_count[i])]
        exp = raw[raw != leaf_ids[i]]  # drop self
        got = neighbors[int(offsets[i]) : int(offsets[i + 1])]
        assert np.array_equal(got, exp), f"leaf row {i} near mismatch"
        assert leaf_ids[i] not in got.tolist(), "self must be excluded"
        assert int(counts[i]) == exp.shape[0]
