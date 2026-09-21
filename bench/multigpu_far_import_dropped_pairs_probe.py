"""Pairs the FAR import cannot serve: the receiver walk's NEAR list against multipole-only nodes.

The hook returns ``rl.far_*`` for the far import and records ``rl.near_count`` as a
diagnostic (``near_pairs``). A far payload row is a multipole with children ``-1``,
so the receiver walk treats it as a leaf: when the MAC fails against a local LEAF
the pair is classified NEAR (``~mac_ok & target_leaf & source_leaf``) -- and a
near pair against a node that shipped no particles is served by nobody. It is
not in the near import either: the sender's far and near sets are disjoint
subtrees. Every such pair is a missing force, independent of expansion order,
which is exactly the shape of the residual the p-sweep left (the ratio to the
single-GPU lane widened 1.26x -> 2.36x over p = 4 -> 6).

This probe reproduces the hook's exchange on CPU with yggdrax alone (no fused
lane, no GPU) and counts those pairs, then checks the explanation: the sender
accepted (cell, node) and the receiver refines the TARGET side only, so a
descendant can fail only if its bounding sphere is not inside the cell's. It
also prices the obvious repair -- serve them by M2L on the sender's authority --
by reporting how far past theta they sit.

    JAX_PLATFORMS=cpu PROBE_N=200000 python bench/multigpu_far_import_dropped_pairs_probe.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from multigpu_import_volume_probe import build_domain_tree, morton_domains  # noqa: E402
from multigpu_oversized_cell_probe import load_ic  # noqa: E402
from yggdrax._interactions_impl import _build_mac_extents, _compute_mac_ok  # noqa: E402
from yggdrax.bounds import infer_bounds  # noqa: E402
from yggdrax.distributed.export import build_send_buffers, export_walk  # noqa: E402
from yggdrax.distributed.import_cells import receiver_interaction_lists  # noqa: E402
from yggdrax.distributed.summary import occupancy_cut  # noqa: E402

N = int(os.environ.get("PROBE_N", "20000"))
NDEV = int(os.environ.get("PROBE_NDEV", "2"))
LEAF = int(os.environ.get("PROBE_LEAF", "64"))
THETA = float(os.environ.get("PROBE_THETA", "0.8"))
IC = os.environ.get("PROBE_IC", "plummer")
MAX_LEAVES = int(os.environ.get("PROBE_MAX_LEAVES", "4"))  # CrossCapacities default
MAC = "dehnen"


def children_full(tree):
    total = int(tree.parent.shape[0])
    nint = int(tree.left_child.shape[0])
    fill = jnp.full((total - nint,), -1, tree.parent.dtype)
    return (
        jnp.concatenate([jnp.asarray(tree.left_child, tree.parent.dtype), fill]),
        jnp.concatenate([jnp.asarray(tree.right_child, tree.parent.dtype), fill]),
    )


def mac_ok(theta, cen_a, rad_a, cen_b, rad_b):
    d = cen_a - cen_b
    d2 = jnp.sum(d * d, axis=-1)
    ok = _compute_mac_ok(
        mac_type=MAC,
        theta_sq=jnp.asarray(theta**2, d2.dtype),
        dist_sq=d2,
        extent_target=rad_a,
        extent_source=rad_b,
        valid_pairs=jnp.ones(d2.shape, bool),
        different_nodes=jnp.ones(d2.shape, bool),
    )
    return np.asarray(ok), np.asarray(jnp.sqrt(d2))


def main():
    pos, mass = load_ic(IC, N)
    bounds = infer_bounds(jnp.asarray(pos))
    dom = morton_domains(pos, bounds, NDEV)
    doms = []
    for d in range(NDEV):
        sel = dom == d
        doms.append(build_domain_tree(pos[sel], mass[sel], bounds, LEAF) + (int(sel.sum()),))
    print(f"N={N} ndev={NDEV} leaf={LEAF} theta={THETA} max_leaves={MAX_LEAVES} ic={IC}")

    # summaries, padded to a common capacity (the hook's all_gather)
    cuts = []
    for tree, geom, _ in doms:
        nint = int(tree.left_child.shape[0])
        c = occupancy_cut(
            tree.parent,
            tree.node_ranges,
            nint,
            max_leaves=MAX_LEAVES,
            capacity=int(tree.parent.shape[0]),
        )
        assert not bool(c.overflow)
        cuts.append(c)
    cap = max(int(c.num_cells) for c in cuts)
    all_cen = np.zeros((NDEV, cap, 3), np.float32)
    all_rad = np.zeros((NDEV, cap), np.float32)
    all_act = np.zeros((NDEV, cap), bool)
    for d, ((tree, geom, _), c) in enumerate(zip(doms, cuts)):
        n = int(c.num_cells)
        cells = np.asarray(c.cells)[:n]
        all_cen[d, :n] = np.asarray(geom.center)[cells]
        all_rad[d, :n] = np.asarray(geom.radius)[cells]
        all_act[d, :n] = True
    all_cen, all_rad, all_act = map(jnp.asarray, (all_cen, all_rad, all_act))

    for s in range(NDEV):
        for r in range(NDEV):
            if s == r:
                continue
            s_tree, s_geom, n_s_part = doms[s]
            r_tree, r_geom, n_r_part = doms[r]
            s_left, s_right = children_full(s_tree)
            idx = s_tree.parent.dtype
            queue = 1 << 18
            while True:
                ex = export_walk(
                    s_left,
                    s_right,
                    jnp.asarray(s_geom.center),
                    jnp.asarray(s_geom.radius),
                    jnp.argmin(s_tree.parent).astype(idx),
                    all_cen,
                    all_rad,
                    all_act,
                    THETA,
                    jnp.asarray(s),
                    max_pair_queue=queue,
                    far_cap=1 << 21,
                    near_cap=1 << 21,
                    mac_type=MAC,
                )
                if not bool(ex.queue_overflow):
                    break
                queue *= 4
            assert not (bool(ex.far_overflow) or bool(ex.near_overflow))
            n_s = int(s_tree.parent.shape[0])
            sb = build_send_buffers(
                ex.far_cell,
                ex.far_node,
                ex.far_count,
                ndev=NDEV,
                max_cells=cap,
                num_nodes=n_s,
                node_capacity=1 << 16,
                csr_capacity=1 << 21,
            )
            assert not (bool(sb.node_overflow) or bool(sb.csr_overflow))
            ns = np.asarray(sb.node_sizes)
            cs = np.asarray(sb.csr_sizes)
            n_off = int(np.concatenate([[0], np.cumsum(ns)])[r])
            c_off = int(np.concatenate([[0], np.cumsum(cs)])[r])
            imp_nodes = np.asarray(sb.node_rows)[n_off : n_off + ns[r]]
            csr_cell = np.asarray(sb.csr_cell)[c_off : c_off + cs[r]]
            csr_row = np.asarray(sb.csr_row)[c_off : c_off + cs[r]]
            k = int(imp_nodes.size)

            # the receiver's combined space: [local ; imported], imported childless
            n_local = int(r_tree.parent.shape[0])
            ridx = r_tree.parent.dtype
            left, right = children_full(r_tree)
            left = jnp.concatenate([left, jnp.full((k,), -1, ridx)])
            right = jnp.concatenate([right, jnp.full((k,), -1, ridx)])
            centers = jnp.concatenate(
                [jnp.asarray(r_geom.center), jnp.asarray(s_geom.center)[imp_nodes]]
            )
            extents = jnp.concatenate(
                [jnp.asarray(r_geom.radius), jnp.asarray(s_geom.radius)[imp_nodes]]
            )
            cut = cuts[r]
            queue = 1 << 20
            while True:
                rl = receiver_interaction_lists(
                    left,
                    right,
                    centers,
                    extents,
                    n_local,
                    cut.cells,
                    jnp.asarray(csr_cell),
                    jnp.asarray(csr_row),
                    jnp.asarray(csr_cell.size),
                    THETA,
                    max_pair_queue=queue,
                    far_cap=1 << 22,
                    near_cap=1 << 22,
                    mac_type=MAC,
                )
                if not bool(rl.queue_overflow):
                    break
                queue *= 4
            assert not (bool(rl.far_overflow) or bool(rl.near_overflow))
            nf, nn = int(rl.far_count), int(rl.near_count)
            print(f"\n== sender {s} -> receiver {r}: imported far nodes {k}, csr {csr_cell.size}")
            print(f"   receiver far pairs {nf}   receiver NEAR pairs against far import (DROPPED) {nn}"
                  f"   = {100.0 * nn / max(nf + nn, 1):.2f} % of the far-import pairs")

            # control: the seeds themselves pass at the receiver (the centre fix)
            cell_roots = np.asarray(cut.cells)[csr_cell]
            ok_seed, _ = mac_ok(
                THETA,
                jnp.asarray(r_geom.center)[cell_roots],
                jnp.asarray(r_geom.radius)[cell_roots],
                centers[n_local + csr_row],
                extents[n_local + csr_row],
            )
            print(f"   seeds failing the MAC at the receiver: {int((~ok_seed).sum())} / {ok_seed.size}")

            # ---- the MAC the LANE applies to its own pairs is not the one the cross
            # exchange applies. The lane's walk uses _build_mac_extents: radii
            # PROPAGATED up the tree plus a depth pad on zero-radius leaves. The
            # export walk and the receiver walk use the RAW geom.radius on both
            # sides. If the raw radius is smaller, cross pairs are accepted at a
            # larger true opening ratio than local pairs, and an M2L at a ratio
            # near theta converges slowly in p -- the shape of a widening ratio.
            ext_s = np.asarray(
                _build_mac_extents(s_tree.parent, s_geom, int(s_tree.left_child.shape[0]), MAC, 1.0)[0]
            )
            ext_r = np.asarray(
                _build_mac_extents(r_tree.parent, r_geom, int(r_tree.left_child.shape[0]), MAC, 1.0)[0]
            )
            ft = np.asarray(rl.far_target)[:nf]
            fs = imp_nodes[np.asarray(rl.far_source)[:nf]]
            rc = np.asarray(r_geom.center)
            rr = np.asarray(r_geom.radius)
            sc = np.asarray(s_geom.center)
            sr = np.asarray(s_geom.radius)
            d = np.linalg.norm(rc[ft] - sc[fs], axis=1)
            ratio_raw = (rr[ft] + sr[fs]) / d
            ratio_eff = (ext_r[ft] + ext_s[fs]) / d
            print(f"   lane extents vs raw radius on exported nodes: eff/raw median "
                  f"{np.median(ext_s[fs] / np.maximum(sr[fs], 1e-30)):.3f}  max "
                  f"{np.max(ext_s[fs] / np.maximum(sr[fs], 1e-30)):.3f}; on target nodes median "
                  f"{np.median(ext_r[ft] / np.maximum(rr[ft], 1e-30)):.3f}")
            qr = np.quantile(ratio_raw, [0.5, 0.9, 1.0])
            qe = np.quantile(ratio_eff, [0.5, 0.9, 1.0])
            print(f"   accepted cross pairs, (r_t+r_s)/d RAW: median {qr[0]:.3f} p90 {qr[1]:.3f} max {qr[2]:.3f}")
            print(f"   same pairs under the LANE's extents:   median {qe[0]:.3f} p90 {qe[1]:.3f} max {qe[2]:.3f}"
                  f"   -> {int((ratio_eff > THETA).sum())} of {nf} pairs would FAIL the lane's MAC "
                  f"({100.0 * (ratio_eff > THETA).mean():.2f} %)")
            zero_leaf_src = int((sr[fs] <= 0).sum())
            print(f"   exported sources with zero raw radius (single-particle nodes): {zero_leaf_src}")

            if nn == 0:
                continue
            nt = np.asarray(rl.near_target)[:nn]
            nsrc = np.asarray(rl.near_source)[:nn]
            ok_n, dist = mac_ok(
                THETA,
                jnp.asarray(r_geom.center)[nt],
                jnp.asarray(r_geom.radius)[nt],
                centers[n_local + nsrc],
                extents[n_local + nsrc],
            )
            rt = np.asarray(r_geom.radius)[nt]
            rs = np.asarray(extents)[n_local + nsrc]
            theta_eff = (rt + rs) / np.maximum(dist, 1e-30)
            print(f"   control: dropped pairs failing the MAC when recomputed: {int((~ok_n).sum())} / {nn}")
            q = np.quantile(theta_eff, [0.5, 0.9, 0.99, 1.0])
            print(f"   (r_t + r_s)/d of dropped pairs: median {q[0]:.3f}  p90 {q[1]:.3f}  "
                  f"p99 {q[2]:.3f}  max {q[3]:.3f}   (theta = {THETA})")

            # explanation: the target leaf's sphere pokes out of its cell's sphere
            # find each dropped target's cell (ancestor in the cut)
            parent = np.asarray(r_tree.parent)
            cutset = set(np.asarray(cut.cells)[: int(cut.num_cells)].tolist())
            root = int(np.argmin(parent))
            cell_of = np.empty(nn, np.int64)
            for i, t in enumerate(nt.tolist()):
                n = t
                while n not in cutset and n != root:
                    n = int(parent[n])
                cell_of[i] = n
            rc = np.asarray(r_geom.center)
            rr = np.asarray(r_geom.radius)
            overhang = np.linalg.norm(rc[nt] - rc[cell_of], axis=1) + rt - rr[cell_of]
            print(f"   target sphere outside its cell's sphere: {int((overhang > 0).sum())} / {nn}"
                  f"   (max overhang {overhang.max():.4f}, max cell radius {rr[cell_of].max():.3f})")

            # size of the hole: sender particles behind the dropped sources, per target leaf
            s_nr = np.asarray(s_tree.node_ranges)
            src_nodes = imp_nodes[nsrc]
            cnt = s_nr[src_nodes, 1] - s_nr[src_nodes, 0] + 1
            per_leaf = {}
            for t, c in zip(nt.tolist(), cnt.tolist()):
                per_leaf[t] = per_leaf.get(t, 0) + c
            vals = np.asarray(list(per_leaf.values()), np.float64) / n_s_part
            print(f"   target leaves with a hole: {len(per_leaf)} / {int(r_tree.parent.shape[0]) - int(r_tree.left_child.shape[0])}"
                  f"   sender fraction missing per such leaf: median {np.median(vals):.4f} max {vals.max():.4f}")


if __name__ == "__main__":
    main()
