"""Coverage-invariant test for the per-leaf treecode walk.

Structural correctness independent of the multipole math: for every target leaf, the
accepted far source nodes' particle ranges UNION the near source leaves' particle
ranges must PARTITION all source particles exactly once. This holds for any valid
treecode descent (accept | leaf->near | internal->refine are mutually exclusive and
exhaustive along each root->leaf path), so it validates the walk logic (push/pop/
accept/near/refine) without depending on the MAC geometry.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax._geometry_impl import compute_tree_geometry
from yggdrax.tree import Tree

from jaccpot.experimental.treecode_walk import treecode_leaf_walk


def _build_bundle(n, seed, leaf_size, theta):
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
        node_ranges=np.asarray(topo.node_ranges),
        n=n,
    )


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs float64 (JAX_ENABLE_X64=1)"
)
@pytest.mark.parametrize(
    ("n", "seed", "leaf_size", "theta"),
    [(64, 1, 8, 0.5), (256, 3, 8, 0.5), (256, 3, 8, 0.9), (500, 7, 16, 0.6)],
)
def test_treecode_coverage_partition(n, seed, leaf_size, theta):
    b = _build_bundle(n, seed, leaf_size, theta)
    cap = b["total_nodes"]  # generous capacities; must not overflow
    out = treecode_leaf_walk(
        b["leaf_nodes"],
        b["centers"],
        b["mac_extents"],
        b["left_full"],
        b["right_full"],
        b["theta_sq"],
        b["root_idx"],
        num_internal=b["num_internal"],
        max_far=cap,
        max_near=cap,
        max_stack=2 * cap + 4,
        max_iters=b["total_nodes"] + 1,
    )
    assert not bool(out.overflow), "walk overflowed a capacity"

    far = np.asarray(out.far_nodes)
    near = np.asarray(out.near_leaves)
    fc = np.asarray(out.far_count)
    nc = np.asarray(out.near_count)
    nr = b["node_ranges"]  # [start, end] inclusive per spec

    def particles_of(node):
        s, e = int(nr[node, 0]), int(nr[node, 1])
        return np.arange(s, e + 1)  # inclusive

    n_leaves = far.shape[0]
    for i in range(n_leaves):
        covered = []
        for j in range(int(fc[i])):
            covered.append(particles_of(far[i, j]))
        for j in range(int(nc[i])):
            covered.append(particles_of(near[i, j]))
        allp = np.concatenate(covered) if covered else np.array([], dtype=int)
        allp_sorted = np.sort(allp)
        # partition of exactly {0..n-1}, each once (no gaps, no overlap)
        assert (
            allp_sorted.shape[0] == b["n"]
        ), f"leaf row {i}: covered {allp_sorted.shape[0]} != N={b['n']}"
        assert np.array_equal(
            allp_sorted, np.arange(b["n"])
        ), f"leaf row {i}: coverage is not the exact partition {{0..N-1}}"
