"""Live multi-GPU distributed far-field parity under jax.shard_map (solidfmm).

Companion to test_distributed_fmm_shardmap.py (which keeps theta_cross->0 so all
remote goes through the bit-exact near-field LET). This test ENGAGES the coarse
far-field with a real theta_cross, using the solidfmm (spherical-harmonic) path:

  per device, inside one shard_map:
    local tree + solidfmm upward (P2M/M2M) multipoles
    build remote coarse tree; all_gather local solidfmm multipoles; seed each
      coarse leaf with the real order-p multipole of the remote leaf it stands
      for (expansion centres coincide -- both the leaf COM), M2M up the coarse
      tree with the solidfmm rotation
    cross-walk local tree vs coarse tree at theta_cross -> far (M2L) + near
    far = self solidfmm M2L (local interaction list)
        + remote solidfmm M2L (cross list, separate source/target centres)
        -> level-by-level L2L cascade -> solidfmm L2P            (far_acc=-G*grad)
    near = combined P2P over [local ; halo]

This depends on two solidfmm bug fixes:
  * M2M now populates the ROOT multipole (was clobbered to 0 by padding-slot
    scatter collisions) -- required for coarse root<-root far interactions.
  * L2L now cascades level-by-level (a single parent->child pass left coarse
    far-field never reaching the leaves) -- required whenever a far interaction
    is accepted above the leaf level.

The far field is theta_cross-controlled: single-device this recipe matches direct
to ~8e-5 at theta_cross=0.1. Here we assert the 4-GPU full force is within 1% of
a global-id direct N-body sum.

    CUDA_VISIBLE_DEVICES=$(autocvd -n 4 -l -o) \
        pytest tests/test_distributed_solidfmm_far_shardmap.py -q
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

from yggdrax import build_interactions_and_neighbors, compute_tree_geometry
from yggdrax.distributed import device_count, make_mesh
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl
from yggdrax.distributed.let import (
    build_coarse_frontier,
    build_remote_coarse_tree,
    import_near_halo,
)
from yggdrax.distributed.local_tree import sanitize_padding
from yggdrax.distributed.partition import global_bounds
from yggdrax.dtypes import INDEX_DTYPE, complex_dtype_for_real
from yggdrax.interactions import DualTreeTraversalConfig
from yggdrax.morton import morton_encode_impl
from yggdrax.tree import (
    Tree,
    get_level_offsets,
    get_node_levels,
    get_nodes_by_level,
)
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.downward.local_expansions import LocalExpansionData
from jaccpot.nearfield.near_field import compute_leaf_p2p_accelerations
from jaccpot.operators.complex_ops import (
    enforce_conjugate_symmetry_batch,
    m2l_complex_reference_batch,
)
from jaccpot.operators.real_harmonics import sh_size
from jaccpot.runtime.kernels.core import (
    _accumulate_m2l_fullbatch,
    _evaluate_local_expansions_for_particles,
    _propagate_solidfmm_locals_by_level,
)
from jaccpot.upward.solidfmm_complex_tree_expansions import (
    _aggregate_m2m_complex_by_level,
    prepare_solidfmm_complex_upward_sweep,
)

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="distributed FMM needs >= 2 devices"
)

_P = 3
_THETA = 0.4  # local self-interaction MAC
_THETA_CROSS = 0.1  # engaged cross-domain far-field MAC
_MAC = "bh"
_LEAF = 8
_G = 1.0
_SOFT = 0.02
_ROT = "solidfmm"


def _direct(all_pos, all_mass, G, soft):
    diff = all_pos[:, None, :] - all_pos[None, :, :]
    d2 = (diff**2).sum(-1) + soft**2
    inv = d2 ** (-1.5)
    return -G * (all_mass[None, :, None] * diff * inv[..., None]).sum(axis=1)


def test_distributed_solidfmm_far_matches_direct():
    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    per = 64
    n = per * ndev
    cap = per
    rng = np.random.default_rng(4)
    # ndev spatially SEPARATED clusters (one per Morton domain) so cross-domain
    # interactions are genuinely far-field -- otherwise adjacent domains resolve
    # everything as near and the far path is never exercised.
    cluster_centers = np.array(
        [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
        dtype=np.float32,
    )[:ndev]
    pts = np.concatenate(
        [cluster_centers[d] + rng.uniform(-0.5, 0.5, (per, 3)) for d in range(ndev)]
    ).astype(np.float32)
    mass = rng.uniform(0.5, 2.0, size=(n,)).astype(np.float32)

    lo, hi = pts.min(0), pts.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    b = (jnp.asarray(lo - span * 1e-6), jnp.asarray(hi + span * 1e-6))
    codes = np.asarray(morton_encode_impl(jnp.asarray(pts), b))
    order = np.argsort(codes)
    pos_g = np.zeros((ndev, cap, 3), np.float32)
    mass_g = np.zeros((ndev, cap), np.float32)
    gid_g = np.full((ndev, cap), -1, np.int64)
    for d in range(ndev):
        chunk = order[d * per : (d + 1) * per]
        pos_g[d, :per] = pts[chunk]
        mass_g[d, :per] = mass[chunk]
        gid_g[d, :per] = chunk
        pos_g[d, per:] = pts[chunk[0]]

    Lloc = cap // _LEAF
    max_req = (ndev - 1) * Lloc
    max_recv = (ndev - 1) * Lloc
    cfg = DualTreeTraversalConfig(
        max_interactions_per_node=512,
        max_neighbors_per_leaf=128,
        max_pair_queue=1 << 15,
        process_block=64,
    )
    KC = 512
    KN = 128
    C = sh_size(_P)

    def _combined_neighbors(tree, nbr, cross, rct, halo, n_halo_rows, local_only=False):
        """Vectorised unified P2P neighbour CSR over [local leaves ; halo blocks]."""
        leaf_nodes = jnp.asarray(nbr.leaf_indices, INDEX_DTYPE)
        n_lloc = leaf_nodes.shape[0]
        total_nodes = jnp.asarray(tree.node_ranges).shape[0]
        node_to_row = (
            jnp.full((total_nodes + 1,), -1, INDEX_DTYPE)
            .at[leaf_nodes]
            .set(jnp.arange(n_lloc, dtype=INDEX_DTYPE))
        )
        u_leaves = n_lloc + n_halo_rows

        lc = jnp.asarray(nbr.counts, INDEX_DTYPE)
        l_src_node = jnp.asarray(nbr.neighbors, INDEX_DTYPE)
        Ne_loc = l_src_node.shape[0]
        l_cum = jnp.cumsum(lc)
        e_loc = jnp.arange(Ne_loc, dtype=INDEX_DTYPE)
        l_tgt = jnp.searchsorted(l_cum, e_loc, side="right")
        l_valid = e_loc < l_cum[-1]
        l_src_row = node_to_row[jnp.clip(l_src_node, 0, total_nodes)]
        l_valid = l_valid & (l_src_row >= 0)

        cc = jnp.asarray(cross.neighbor_counts, INDEX_DTYPE)
        r_src_cnode = jnp.asarray(cross.neighbor_indices, INDEX_DTYPE)
        Ne_rem = r_src_cnode.shape[0]
        c_cum = jnp.cumsum(cc)
        e_rem = jnp.arange(Ne_rem, dtype=INDEX_DTYPE)
        c_row = jnp.searchsorted(c_cum, e_rem, side="right")
        r_valid = e_rem < c_cum[-1]
        cross_leaf = jnp.asarray(cross.leaf_indices, INDEX_DTYPE)
        c_tgt_node = cross_leaf[jnp.clip(c_row, 0, cross_leaf.shape[0] - 1)]
        r_tgt = node_to_row[jnp.clip(c_tgt_node, 0, total_nodes)]
        coarse_ranges = jnp.asarray(rct.tree.node_ranges, INDEX_DTYPE)
        coarse_pos = coarse_ranges[
            jnp.clip(r_src_cnode, 0, coarse_ranges.shape[0] - 1), 0
        ]
        block = halo.coarse_to_halo[
            jnp.clip(coarse_pos, 0, halo.coarse_to_halo.shape[0] - 1)
        ]
        r_src_row = n_lloc + block
        r_valid = r_valid & (r_tgt >= 0) & (block >= 0)

        if local_only:
            tgt, src, valid = l_tgt, l_src_row, l_valid
        else:
            tgt = jnp.concatenate([l_tgt, r_tgt])
            src = jnp.concatenate([l_src_row, r_src_row])
            valid = jnp.concatenate([l_valid, r_valid])
        sentinel = jnp.asarray(u_leaves, INDEX_DTYPE)
        tgt = jnp.where(valid, tgt, sentinel)
        src = jnp.where(valid, src, sentinel)

        srt = jnp.argsort(tgt)
        src_s = src[srt]
        counts = jnp.bincount(tgt, length=u_leaves).astype(INDEX_DTYPE)
        offsets = jnp.concatenate([jnp.zeros((1,), INDEX_DTYPE), jnp.cumsum(counts)])
        return offsets, src_s, counts, u_leaves

    def fn(pos, mass, gid, count):
        bounds = global_bounds(pos)
        pos_s, mass_s = sanitize_padding(pos, mass, count)
        tree = Tree.from_particles(
            pos_s,
            mass_s,
            tree_type="radix",
            bounds=bounds,
            return_reordered=True,
            leaf_size=_LEAF,
        )
        lp = tree.positions_sorted
        lm = tree.masses_sorted
        gid_sorted = gid[jnp.asarray(tree.particle_indices, INDEX_DTYPE)]
        geom = compute_tree_geometry(tree, lp, max_leaf_size=_LEAF)
        total_nodes = jnp.asarray(tree.node_ranges).shape[0]
        cdtype = complex_dtype_for_real(lp.dtype)

        # local solidfmm upward
        up = prepare_solidfmm_complex_upward_sweep(
            tree, lp, lm, max_order=_P, max_leaf_size=_LEAF, rotation=_ROT
        )
        centers = up.multipoles.centers
        packed = up.multipoles.packed

        inter, nbr = build_interactions_and_neighbors(
            tree,
            geom,
            theta=_THETA,
            traversal_config=cfg,
            mac_type=_MAC,
        )

        # remote coarse tree over frontier (leaf COM + mass)
        mm = compute_tree_mass_moments(tree, lp, lm)
        fr = build_coarse_frontier(tree, mm.mass, mm.center_of_mass)
        rct = build_remote_coarse_tree(fr, ndev, bounds=bounds)

        # coarse solidfmm centres (COM) + level structure
        upc = prepare_solidfmm_complex_upward_sweep(
            rct.tree,
            rct.positions_sorted,
            rct.masses_sorted,
            max_order=_P,
            max_leaf_size=1,
            rotation=_ROT,
        )
        c_centers = upc.multipoles.centers
        c_lc = jnp.asarray(rct.tree.left_child, INDEX_DTYPE)
        c_rc = jnp.asarray(rct.tree.right_child, INDEX_DTYPE)
        c_total = c_centers.shape[0]
        c_nint = int(c_lc.shape[0])
        c_nr = jnp.asarray(rct.tree.node_ranges, INDEX_DTYPE)
        c_leaves = jnp.arange(c_nint, c_total, dtype=INDEX_DTYPE)

        # seed coarse leaves with remote leaves' REAL solidfmm multipoles
        gpacked = jax.lax.all_gather(packed, "gpus", tiled=False)  # [ndev,N,C]
        spos = c_nr[c_leaves, 0]
        dom = rct.tag_domain[spos]
        nod = rct.tag_node_id[spos]
        okm = nod >= 0
        leafp = gpacked[jnp.where(okm, dom, 0), jnp.where(okm, nod, 0)]
        leafp = jnp.where(okm[:, None], leafp, 0.0)
        seed = (
            jnp.zeros((c_total, C), dtype=cdtype).at[c_leaves].set(leafp.astype(cdtype))
        )
        c_nbl = get_nodes_by_level(rct.tree)
        c_loff = get_level_offsets(rct.tree)
        c_numlev = int(c_loff.shape[0] - 1)
        coarse_packed = _aggregate_m2m_complex_by_level(
            seed,
            c_centers,
            c_lc,
            c_rc,
            jnp.asarray(c_nbl, INDEX_DTYPE),
            jnp.asarray(c_loff, INDEX_DTYPE),
            order=_P,
            num_internal=c_nint,
            num_levels=c_numlev,
            level_batch_width=max(c_nint, 1),
            rotation=_ROT,
        )

        cross = dual_tree_walk_cross_impl(
            tree,
            geom,
            rct.tree,
            rct.geometry,
            _THETA_CROSS,
            mac_type=_MAC,
            max_interactions_per_node=KC,
            max_neighbors_per_leaf=KN,
            max_pair_queue=1 << 15,
        )
        halo = import_near_halo(
            rct,
            cross,
            lp,
            lm,
            ndev,
            leaf_size=_LEAF,
            max_req_leaves=max_req,
            max_recv_leaves=max_recv,
        )

        leaf_nodes = jnp.asarray(nbr.leaf_indices, INDEX_DTYPE)
        nr = jnp.asarray(tree.node_ranges, INDEX_DTYPE)
        node_levels = get_node_levels(tree)
        lc_full = jnp.asarray(tree.left_child, INDEX_DTYPE)
        rc_full = jnp.asarray(tree.right_child, INDEX_DTYPE)

        def _l2p(loc_coeffs):
            loc_coeffs = enforce_conjugate_symmetry_batch(loc_coeffs, order=_P)
            loc_coeffs = _propagate_solidfmm_locals_by_level(
                loc_coeffs,
                centers,
                lc_full,
                rc_full,
                node_levels,
                order=_P,
                rotation=_ROT,
                total_nodes=int(total_nodes),
            )
            ld = LocalExpansionData(order=_P, centers=centers, coefficients=loc_coeffs)
            g = _evaluate_local_expansions_for_particles(
                ld,
                lp,
                leaf_nodes=leaf_nodes,
                node_ranges=nr,
                max_leaf_size=_LEAF,
                order=_P,
                expansion_basis="solidfmm",
                return_potential=False,
            )[0]
            return -_G * g

        # far: self solidfmm M2L (local list) + remote solidfmm M2L (cross list)
        zeros = jnp.zeros((int(total_nodes), C), dtype=cdtype)
        s_src = jnp.asarray(inter.sources, INDEX_DTYPE)
        s_tgt = jnp.asarray(inter.targets, INDEX_DTYPE)
        s_active = jnp.sum((s_tgt >= 0).astype(INDEX_DTYPE))
        loc_self = _accumulate_m2l_fullbatch(
            zeros,
            packed,
            centers,
            s_src,
            s_tgt,
            s_active,
            order=_P,
            basis_mode="complex",
            rotation=_ROT,
            total_nodes=int(total_nodes),
        )
        far_self = _l2p(loc_self)

        # cross M2L: separate source (coarse) / target (local) centres
        x_src = jnp.asarray(cross.interaction_sources, INDEX_DTYPE)
        x_tgt = jnp.asarray(cross.interaction_targets, INDEX_DTYPE)
        x_valid = x_tgt >= 0
        xs = jnp.where(x_valid, x_src, 0)
        xt = jnp.where(x_valid, x_tgt, 0)
        x_contribs = m2l_complex_reference_batch(
            coarse_packed[xs], centers[xt] - c_centers[xs], order=_P, rotation=_ROT
        ).astype(cdtype)
        x_contribs = jnp.where(x_valid[:, None], x_contribs, 0)
        loc_full = loc_self + jax.ops.segment_sum(x_contribs, xt, int(total_nodes))
        far_full = _l2p(loc_full)

        # near: leaf particle-index groups into [local ; halo] buffer
        lranges = nr[leaf_nodes]
        kk = jnp.arange(_LEAF, dtype=INDEX_DTYPE)
        loc_idx = jnp.clip(lranges[:, 0][:, None] + kk[None, :], 0, cap - 1)
        loc_mask = kk[None, :] < (lranges[:, 1] - lranges[:, 0] + 1)[:, None]
        halo_idx = (cap + jnp.arange(max_req * _LEAF, dtype=INDEX_DTYPE)).reshape(
            max_req, _LEAF
        )
        halo_mask = halo.valid.reshape(max_req, _LEAF)
        concat_pos = jnp.concatenate([lp, halo.positions], axis=0)
        concat_mass = jnp.concatenate([lm, halo.masses], axis=0)

        offs_s, srcs_s, cnts_s, ul_s = _combined_neighbors(
            tree, nbr, cross, rct, halo, 0, local_only=True
        )
        near_self = compute_leaf_p2p_accelerations(
            tree,
            nbr,
            lp,
            lm,
            G=_G,
            softening=_SOFT,
            nearfield_mode="baseline",
            node_ranges_override=jnp.zeros((ul_s + 1, 2), INDEX_DTYPE),
            leaf_nodes_override=jnp.arange(ul_s, dtype=INDEX_DTYPE),
            neighbor_offsets_override=offs_s,
            neighbor_indices_override=srcs_s,
            neighbor_counts_override=cnts_s,
            leaf_particle_indices_override=loc_idx,
            leaf_particle_mask_override=loc_mask,
        )

        offsets, src_s, counts, u_leaves = _combined_neighbors(
            tree, nbr, cross, rct, halo, max_req
        )
        lp_idx = jnp.concatenate([loc_idx, halo_idx], axis=0)
        lp_mask = jnp.concatenate([loc_mask, halo_mask], axis=0)
        near_full = compute_leaf_p2p_accelerations(
            tree,
            nbr,
            concat_pos,
            concat_mass,
            G=_G,
            softening=_SOFT,
            nearfield_mode="baseline",
            node_ranges_override=jnp.zeros((u_leaves + 1, 2), INDEX_DTYPE),
            leaf_nodes_override=jnp.arange(u_leaves, dtype=INDEX_DTYPE),
            neighbor_offsets_override=offsets,
            neighbor_indices_override=src_s,
            neighbor_counts_override=counts,
            leaf_particle_indices_override=lp_idx,
            leaf_particle_mask_override=lp_mask,
        )

        diag = jnp.array(
            [
                jnp.sum(x_valid.astype(jnp.float64)),  # cross far pairs
                jnp.sum((s_tgt >= 0).astype(jnp.float64)),  # local self far pairs
                jnp.sum(jnp.asarray(cross.neighbor_counts)).astype(jnp.float64),
                jnp.linalg.norm(far_full - far_self).astype(jnp.float64),  # remote far
            ]
        )
        return (
            far_self[:cap],
            near_self[:cap],
            far_full[:cap],
            near_full[:cap],
            gid_sorted[:, None].astype(jnp.float64),
            diag[None, :],
        )

    counts_dev = jnp.full((ndev,), per, dtype=INDEX_DTYPE)
    far_o, near_o, far_full_o, near_full_o, gid_out, diag_o = shard_map(
        fn,
        mesh=mesh,
        in_specs=(P("gpus"), P("gpus"), P("gpus"), P("gpus")),
        out_specs=(P("gpus"),) * 6,
        check_vma=False,
    )(
        jnp.asarray(pos_g.reshape(ndev * cap, 3)),
        jnp.asarray(mass_g.reshape(ndev * cap)),
        jnp.asarray(gid_g.reshape(ndev * cap)),
        counts_dev,
    )
    far_o = np.asarray(far_o)
    near_o = np.asarray(near_o)
    far_full_o = np.asarray(far_full_o)
    near_full_o = np.asarray(near_full_o)
    gid_out = np.asarray(gid_out).reshape(-1).astype(np.int64)
    diag_o = np.asarray(diag_o)
    print(
        "per-domain [cross_far_pairs, self_far_pairs, cross_near_edges, "
        "||remote_far||]:\n" + str(np.round(diag_o, 4))
    )

    dist = np.zeros((n, 3), np.float64)
    dist_self = np.zeros((n, 3), np.float64)
    domain_of = np.full(n, -1, np.int64)
    seen = np.zeros(n, bool)
    rows_per = far_o.shape[0] // ndev
    for row in range(far_o.shape[0]):
        g = gid_out[row]
        if g >= 0:
            dist_self[g] = far_o[row] + near_o[row]
            dist[g] = far_full_o[row] + near_full_o[row]
            domain_of[g] = row // rows_per
            seen[g] = True
    assert seen.all(), "some global particles missing from the distributed result"

    direct = np.asarray(_direct(jnp.asarray(pts), jnp.asarray(mass), _G, _SOFT))
    direct_local = np.zeros((n, 3), np.float64)
    for d in range(ndev):
        idx = np.where(domain_of == d)[0]
        direct_local[idx] = np.asarray(
            _direct(jnp.asarray(pts[idx]), jnp.asarray(mass[idx]), _G, _SOFT)
        )

    def _agg(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))

    err_self = _agg(dist_self, direct_local)
    err_full = _agg(dist, direct)
    print(f"LOCAL-ONLY (self far+near) aggL2={err_self:.6f}")
    print(f"FULL  (solidfmm far engaged) aggL2={err_full:.6f}")
    assert err_self < 1e-2, f"LOCAL aggL2 err {err_self:.6f}"
    assert err_full < 1e-2, f"FULL aggL2 err {err_full:.6f}"
