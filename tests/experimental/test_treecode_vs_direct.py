"""Physical-accuracy gate for the per-leaf treecode: forces vs direct N-body.

Assembles the full treecode force path — per-leaf treecode walk -> M2L into each
leaf's local expansion -> L2P (SKIP L2L) -> P2P near-field — and asserts it matches
direct softened N-body to FMM truncation error. This is the make-or-break correctness
gate for the treecode approach (parity defined vs direct, not vs the current dual-tree).

Conventions (validated empirically; see benchmark_a100/WALK_SPEC.md):
- far side is unsoftened, G applied as far_acc = +G * L2P_grad (the hand-assembled
  M2L->L2P delta convention flips the sign vs the solidfmm runtime path).
- near P2P applies G + softening internally and adds each leaf's self-block once, so
  the near list must EXCLUDE self.
CPU / float64. Host-side CSR assembly (this is the correctness reference; the jittable
on-device fallback is a later step).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax._geometry_impl import compute_tree_geometry
from yggdrax._interactions_impl import NodeInteractionList
from yggdrax.bounds import infer_bounds
from yggdrax.interactions import build_interactions_and_neighbors
from yggdrax.tree import Tree

try:
    from yggdrax.dtypes import INDEX_DTYPE
except Exception:  # pragma: no cover
    INDEX_DTYPE = jnp.int64

from jaccpot.downward.local_expansions import (
    accumulate_m2l_contributions,
    initialize_local_expansions,
)
from jaccpot.experimental.treecode_walk import treecode_leaf_walk
from jaccpot.nearfield.near_field import compute_leaf_p2p_accelerations
from jaccpot.runtime._fmm_impl import _evaluate_local_expansions_for_particles
from jaccpot.upward.tree_expansions import compute_node_multipoles


def _treecode_total_and_direct(pos, mass, *, P, LEAF, THETA, G, SOFT):
    bounds = infer_bounds(pos)
    tree = Tree.from_particles(
        pos,
        mass,
        tree_type="radix",
        bounds=bounds,
        return_reordered=True,
        leaf_size=LEAF,
    )
    lp, lm = tree.positions_sorted, tree.masses_sorted
    topo = tree.topology
    geom = compute_tree_geometry(topo, lp, max_leaf_size=LEAF)
    num_internal = int(topo.num_internal_nodes)
    total_nodes = int(topo.parent.shape[0])
    num_leaves = total_nodes - num_internal
    idx = topo.parent.dtype
    left_full = jnp.concatenate(
        [topo.left_child.astype(idx), jnp.full((num_leaves,), -1, idx)]
    )
    right_full = jnp.concatenate(
        [topo.right_child.astype(idx), jnp.full((num_leaves,), -1, idx)]
    )
    leaf_nodes = jnp.arange(num_internal, total_nodes, dtype=idx)
    root_idx = jnp.argmin(topo.parent).astype(idx)

    mp = compute_node_multipoles(tree, lp, lm, max_order=P)._replace(order=P)

    cap = total_nodes
    out = treecode_leaf_walk(
        leaf_nodes,
        geom.center,
        geom.max_extent,
        left_full,
        right_full,
        jnp.asarray(THETA * THETA, jnp.float64),
        root_idx,
        num_internal=num_internal,
        max_far=cap,
        max_near=cap,
        max_stack=2 * cap + 4,
        max_iters=total_nodes + 1,
    )
    assert not bool(out.overflow)
    far_nodes = np.asarray(out.far_nodes)
    far_cnt = np.asarray(out.far_count)
    near_lv = np.asarray(out.near_leaves)
    near_cnt = np.asarray(out.near_count)
    assert far_cnt.sum() > 0, "config produced no far interactions (M2L path untested)"

    # --- M2L interaction list keyed by NODE id (nonzero counts only at leaves) ---
    counts_node = np.zeros(total_nodes, dtype=np.int64)
    src_flat = []
    for i in range(num_leaves):
        fs = far_nodes[i, : int(far_cnt[i])]
        counts_node[num_internal + i] = fs.shape[0]
        src_flat.append(fs)
    src_flat = np.concatenate(src_flat).astype(np.int64)
    offsets_node = np.zeros(total_nodes, dtype=np.int64)
    offsets_node[1:] = np.cumsum(counts_node)[:-1]
    src_j = jnp.asarray(src_flat, INDEX_DTYPE)
    ilist = NodeInteractionList(
        offsets=jnp.asarray(offsets_node, INDEX_DTYPE),
        sources=src_j,
        targets=jnp.zeros_like(src_j),
        counts=jnp.asarray(counts_node, INDEX_DTYPE),
        level_offsets=jnp.zeros((1,), INDEX_DTYPE),
        target_levels=jnp.zeros_like(src_j),
    )
    local = initialize_local_expansions(tree, mp.centers, max_order=P)._replace(order=P)
    local = accumulate_m2l_contributions(ilist, mp, local)._replace(order=P)
    far_grad = _evaluate_local_expansions_for_particles(
        local,
        lp,
        leaf_nodes=jnp.asarray(leaf_nodes, INDEX_DTYPE),
        node_ranges=jnp.asarray(topo.node_ranges, INDEX_DTYPE),
        max_leaf_size=LEAF,
        order=P,
        expansion_basis="cartesian",
        return_potential=False,
    )[0]
    far_acc = G * np.asarray(far_grad)  # +G: hand-assembled M2L->L2P sign

    # --- near P2P via overrides (rows = leaves; EXCLUDE self) ---
    nr = np.asarray(topo.node_ranges)
    leaf_pidx = np.zeros((num_leaves, LEAF), dtype=np.int64)
    leaf_mask = np.zeros((num_leaves, LEAF), dtype=bool)
    for i in range(num_leaves):
        s, e = int(nr[num_internal + i, 0]), int(nr[num_internal + i, 1])
        c = e - s + 1
        leaf_pidx[i, :c] = np.arange(s, e + 1)
        leaf_mask[i, :c] = True
    nb_off = np.zeros(num_leaves + 1, dtype=np.int64)
    nb_idx = []
    for i in range(num_leaves):
        self_node = num_internal + i
        rows = [
            int(x) - num_internal
            for x in near_lv[i, : int(near_cnt[i])]
            if int(x) != self_node
        ]
        nb_idx.append(np.asarray(rows, np.int64))
        nb_off[i + 1] = nb_off[i] + len(rows)
    nb_idx = np.concatenate(nb_idx).astype(np.int64)

    _, nbr = build_interactions_and_neighbors(tree, geom, theta=THETA)  # placeholder
    near = compute_leaf_p2p_accelerations(
        tree,
        nbr,
        lp,
        lm,
        G=G,
        softening=SOFT,
        nearfield_mode="baseline",
        node_ranges_override=jnp.zeros((num_leaves + 1, 2), INDEX_DTYPE),
        leaf_nodes_override=jnp.arange(num_leaves, dtype=INDEX_DTYPE),
        neighbor_offsets_override=jnp.asarray(nb_off, INDEX_DTYPE),
        neighbor_indices_override=jnp.asarray(nb_idx, INDEX_DTYPE),
        neighbor_counts_override=jnp.asarray(np.diff(nb_off), INDEX_DTYPE),
        leaf_particle_indices_override=jnp.asarray(leaf_pidx, INDEX_DTYPE),
        leaf_particle_mask_override=jnp.asarray(leaf_mask),
    )
    total = far_acc + np.asarray(near)[: pos.shape[0]]

    lp_np, lm_np = np.asarray(lp), np.asarray(lm)
    d = lp_np[:, None, :] - lp_np[None, :, :]
    d2 = (d**2).sum(-1) + SOFT**2
    np.fill_diagonal(d2, np.inf)
    direct = -G * (lm_np[None, :, None] * d * (d2**-1.5)[..., None]).sum(axis=1)
    return total, direct


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs float64 (JAX_ENABLE_X64=1)"
)
@pytest.mark.parametrize(
    ("n", "seed", "leaf", "theta", "p"),
    [(2000, 1, 16, 0.5, 4), (4000, 7, 16, 0.6, 4), (3000, 3, 32, 0.5, 4)],
)
def test_treecode_matches_direct(n, seed, leaf, theta, p):
    kp, km = jax.random.split(jax.random.PRNGKey(seed))
    pos = jax.random.uniform(kp, (n, 3), dtype=jnp.float64, minval=-1.0, maxval=1.0)
    mass = jnp.abs(jax.random.normal(km, (n,), dtype=jnp.float64)) + 0.5
    total, direct = _treecode_total_and_direct(
        pos, mass, P=p, LEAF=leaf, THETA=theta, G=1.0, SOFT=1e-2
    )
    rel = np.linalg.norm(total - direct, axis=1) / (
        np.linalg.norm(direct, axis=1) + 1e-12
    )
    med, p90 = float(np.median(rel)), float(np.percentile(rel, 90))
    assert med < 0.02, f"median rel {med:.3e} too large"
    assert p90 < 0.05, f"p90 rel {p90:.3e} too large"
