"""End-to-end distributed-FMM force assembly parity (Phase 4c, single device).

Emulates one GPU that owns domain A and treats domain B as remote, assembling
A's accelerations exactly as the distributed pipeline will:

  far  = self M2L (A vs A)  +  remote M2L (A targets <- B multipoles, via the
         cross-tree walk far list)  -> L2L -> L2P            (far_acc = -G * grad)
  near = ONE combined P2P over [A ; B] with a unified neighbour list
         (A-self-near leaves + A<-B-near leaves; the kernel's per-leaf
         self-block is counted exactly once)

and checks A's total (far+near) against a direct N-body sum over all particles.
A correct assembly matches direct to FMM truncation error (~%); any wiring bug
(wrong source array, neighbour mapping, sign) blows the error up.

This is the single-device validation of the force path; the shard_map wrapping
is mechanical on top.
"""

import jax.numpy as jnp
import numpy as np
from yggdrax import (
    build_interactions_and_neighbors,
    compute_tree_geometry,
    get_leaf_nodes,
    infer_bounds,
)
from yggdrax.distributed import dual_tree_walk_cross
from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.interactions import NodeInteractionList
from yggdrax.tree import Tree

from jaccpot.downward.local_expansions import (
    accumulate_m2l_contributions,
    initialize_local_expansions,
    propagate_local_expansions,
)
from jaccpot.nearfield.near_field import compute_leaf_p2p_accelerations
from jaccpot.runtime._fmm_impl import _evaluate_local_expansions_for_particles
from jaccpot.upward.tree_expansions import compute_node_multipoles

_P = 2
_THETA = 0.4
_MAC = "bh"
_LEAF = 8
_G = 1.0
_SOFT = 0.01


def _build(points, bounds):
    pts = jnp.asarray(points)
    mass = jnp.asarray(points_mass(points))
    tree = Tree.from_particles(
        pts,
        mass,
        tree_type="radix",
        bounds=bounds,
        return_reordered=True,
        leaf_size=_LEAF,
    )
    geom = compute_tree_geometry(tree, tree.positions_sorted, max_leaf_size=_LEAF)
    return tree, geom


def points_mass(points):
    rng = np.random.default_rng(hash(points.tobytes()) % (2**32))
    return rng.uniform(0.5, 2.0, size=(points.shape[0],)).astype(np.float32)


def _ilist_from_cross(res):
    return NodeInteractionList(
        offsets=jnp.asarray(res.interaction_offsets, INDEX_DTYPE),
        sources=jnp.asarray(res.interaction_sources, INDEX_DTYPE),
        targets=jnp.asarray(res.interaction_targets, INDEX_DTYPE),
        counts=jnp.asarray(res.interaction_counts, INDEX_DTYPE),
        level_offsets=jnp.zeros((1,), INDEX_DTYPE),
        target_levels=jnp.zeros_like(jnp.asarray(res.interaction_sources), INDEX_DTYPE),
    )


def _direct(all_pos, all_mass, tgt_pos, G, soft):
    diff = tgt_pos[:, None, :] - all_pos[None, :, :]
    d2 = (diff**2).sum(-1) + soft**2
    inv = d2 ** (-1.5)
    return -G * (all_mass[None, :, None] * diff * inv[..., None]).sum(axis=1)


def test_distributed_fmm_assembly_matches_direct():
    rng = np.random.default_rng(7)
    n = 128
    pts = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    bounds = infer_bounds(jnp.asarray(pts))
    # Split into two spatially-contiguous domains by x (A = remote's neighbour).
    order = np.argsort(pts[:, 0])
    A_pts = pts[order[: n // 2]]
    B_pts = pts[order[n // 2 :]]

    tree_A, geom_A = _build(A_pts, bounds)
    tree_B, geom_B = _build(B_pts, bounds)
    A_pos = np.asarray(tree_A.positions_sorted)
    A_mass = np.asarray(tree_A.masses_sorted)
    B_pos = np.asarray(tree_B.positions_sorted)
    B_mass = np.asarray(tree_B.masses_sorted)
    nA, nB = A_pos.shape[0], B_pos.shape[0]

    mp_A = compute_node_multipoles(
        tree_A, tree_A.positions_sorted, tree_A.masses_sorted, max_order=_P
    )
    mp_B = compute_node_multipoles(
        tree_B, tree_B.positions_sorted, tree_B.masses_sorted, max_order=_P
    )

    # ---- FAR: self M2L + remote M2L -> L2L -> L2P ----
    inter_A, nbr_A = build_interactions_and_neighbors(
        tree_A, geom_A, theta=_THETA, mac_type=_MAC
    )
    cross = dual_tree_walk_cross(
        tree_A,
        geom_A,
        tree_B,
        geom_B,
        _THETA,
        mac_type=_MAC,
        max_interactions_per_node=512,
        max_neighbors_per_leaf=512,
        max_pair_queue=16384,
    )
    assert not bool(cross.far_overflow) and not bool(cross.near_overflow)
    assert not bool(cross.queue_overflow)

    local = initialize_local_expansions(tree_A, mp_A.centers, max_order=_P)
    local = accumulate_m2l_contributions(inter_A, mp_A, local)  # A <- A
    local = accumulate_m2l_contributions(
        _ilist_from_cross(cross), mp_B, local
    )  # A <- B
    local = propagate_local_expansions(tree_A, local)  # L2L

    A_leaf_nodes = np.asarray(nbr_A.leaf_indices)
    far_grad = _evaluate_local_expansions_for_particles(
        local,
        tree_A.positions_sorted,
        leaf_nodes=jnp.asarray(A_leaf_nodes, INDEX_DTYPE),
        node_ranges=jnp.asarray(tree_A.node_ranges, INDEX_DTYPE),
        max_leaf_size=_LEAF,
        order=_P,
        expansion_basis="cartesian",
        return_potential=False,
    )[0]
    far_acc = -_G * np.asarray(far_grad)[:nA]

    # ---- NEAR: one combined P2P over [A ; B] with a unified neighbour list ----
    B_leaf_nodes = np.asarray(get_leaf_nodes(tree_B))
    La, Lb = len(A_leaf_nodes), len(B_leaf_nodes)
    A_ranges = np.asarray(tree_A.node_ranges)
    B_ranges = np.asarray(tree_B.node_ranges)
    A_node_to_row = {int(nd): r for r, nd in enumerate(A_leaf_nodes)}
    B_node_to_row = {int(nd): r for r, nd in enumerate(B_leaf_nodes)}

    # leaf -> particle indices into the concatenated [A ; B] buffer
    idx = np.zeros((La + Lb, _LEAF), dtype=np.int64)
    mask = np.zeros((La + Lb, _LEAF), dtype=bool)
    for r, nd in enumerate(A_leaf_nodes):
        s, e = int(A_ranges[nd, 0]), int(A_ranges[nd, 1])
        c = e - s + 1
        idx[r, :c] = np.arange(s, e + 1)
        mask[r, :c] = True
    for r, nd in enumerate(B_leaf_nodes):
        s, e = int(B_ranges[nd, 0]), int(B_ranges[nd, 1])
        c = e - s + 1
        idx[La + r, :c] = nA + np.arange(s, e + 1)
        mask[La + r, :c] = True

    # per-A-node neighbour source nodes (self from nbr_A, remote from cross)
    def _csr_map(off, nbr, cnt, rowleaf):
        off, nbr, cnt, rowleaf = map(np.asarray, (off, nbr, cnt, rowleaf))
        return {
            int(rowleaf[r]): nbr[int(off[r]) : int(off[r]) + int(cnt[r])].tolist()
            for r in range(len(rowleaf))
        }

    self_nbr = _csr_map(
        nbr_A.offsets, nbr_A.neighbors, nbr_A.counts, nbr_A.leaf_indices
    )
    cross_nbr = _csr_map(
        cross.neighbor_offsets,
        cross.neighbor_indices,
        cross.neighbor_counts,
        cross.leaf_indices,
    )

    uni_neighbors, uni_offsets = [], [0]
    for nd in A_leaf_nodes:
        for sn in self_nbr.get(int(nd), []):
            uni_neighbors.append(A_node_to_row[int(sn)])  # A self -> A row
        for sn in cross_nbr.get(int(nd), []):
            uni_neighbors.append(La + B_node_to_row[int(sn)])  # A<-B  -> B row
        uni_offsets.append(len(uni_neighbors))
    for _ in range(Lb):  # B rows target nothing (their outputs are discarded)
        uni_offsets.append(len(uni_neighbors))
    uni_counts = np.diff(uni_offsets)

    concat_pos = jnp.asarray(np.concatenate([A_pos, B_pos], axis=0))
    concat_mass = jnp.asarray(np.concatenate([A_mass, B_mass], axis=0))
    # tree_A / nbr_A are passed only to satisfy the typed signature; every field
    # they expose is bypassed by the overrides below.
    near = compute_leaf_p2p_accelerations(
        tree_A,
        nbr_A,
        concat_pos,
        concat_mass,
        G=_G,
        softening=_SOFT,
        nearfield_mode="baseline",
        node_ranges_override=jnp.zeros((La + Lb, 2), INDEX_DTYPE),
        leaf_nodes_override=jnp.arange(La + Lb, dtype=INDEX_DTYPE),
        neighbor_offsets_override=jnp.asarray(uni_offsets, INDEX_DTYPE),
        neighbor_indices_override=jnp.asarray(uni_neighbors or [0], INDEX_DTYPE),
        neighbor_counts_override=jnp.asarray(uni_counts, INDEX_DTYPE),
        leaf_particle_indices_override=jnp.asarray(idx, INDEX_DTYPE),
        leaf_particle_mask_override=jnp.asarray(mask),
    )
    near_acc = np.asarray(near)[:nA]

    total_A = far_acc + near_acc

    # ---- reference: direct N-body over ALL particles, forces on A ----
    all_pos = jnp.asarray(np.concatenate([A_pos, B_pos], axis=0))
    all_mass = jnp.asarray(np.concatenate([A_mass, B_mass], axis=0))
    direct_A = np.asarray(_direct(all_pos, all_mass, jnp.asarray(A_pos), _G, _SOFT))

    rel = np.linalg.norm(total_A - direct_A, axis=1) / (
        np.linalg.norm(direct_A, axis=1) + 1e-12
    )
    # correct assembly -> FMM truncation error only (order 2, theta 0.4)
    assert np.median(rel) < 0.05, f"median rel err {np.median(rel):.3f}"
    assert np.percentile(rel, 90) < 0.15, f"p90 rel err {np.percentile(rel, 90):.3f}"
