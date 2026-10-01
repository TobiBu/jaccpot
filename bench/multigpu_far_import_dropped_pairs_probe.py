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
        doms.append(
            build_domain_tree(pos[sel], mass[sel], bounds, LEAF) + (int(sel.sum()),)
        )
    dom_pm = [(pos[dom == d], mass[dom == d]) for d in range(NDEV)]
    print(
        f"N={N} ndev={NDEV} leaf={LEAF} theta={THETA} max_leaves={MAX_LEAVES} ic={IC}"
    )

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
            print(
                f"\n== sender {s} -> receiver {r}: imported far nodes {k}, csr {csr_cell.size}"
            )
            print(
                f"   receiver far pairs {nf}   receiver NEAR pairs against far import (DROPPED) {nn}"
                f"   = {100.0 * nn / max(nf + nn, 1):.2f} % of the far-import pairs"
            )

            # control: the seeds themselves pass at the receiver (the centre fix)
            cell_roots = np.asarray(cut.cells)[csr_cell]
            ok_seed, _ = mac_ok(
                THETA,
                jnp.asarray(r_geom.center)[cell_roots],
                jnp.asarray(r_geom.radius)[cell_roots],
                centers[n_local + csr_row],
                extents[n_local + csr_row],
            )
            print(
                f"   seeds failing the MAC at the receiver: {int((~ok_seed).sum())} / {ok_seed.size}"
            )

            # ---- the MAC the LANE applies to its own pairs is not the one the cross
            # exchange applies. The lane's walk uses _build_mac_extents: radii
            # PROPAGATED up the tree plus a depth pad on zero-radius leaves. The
            # export walk and the receiver walk use the RAW geom.radius on both
            # sides. If the raw radius is smaller, cross pairs are accepted at a
            # larger true opening ratio than local pairs, and an M2L at a ratio
            # near theta converges slowly in p -- the shape of a widening ratio.
            ext_s = np.asarray(
                _build_mac_extents(
                    s_tree.parent, s_geom, int(s_tree.left_child.shape[0]), MAC, 1.0
                )[0]
            )
            ext_r = np.asarray(
                _build_mac_extents(
                    r_tree.parent, r_geom, int(r_tree.left_child.shape[0]), MAC, 1.0
                )[0]
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
            print(
                f"   lane extents vs raw radius on exported nodes: eff/raw median "
                f"{np.median(ext_s[fs] / np.maximum(sr[fs], 1e-30)):.3f}  max "
                f"{np.max(ext_s[fs] / np.maximum(sr[fs], 1e-30)):.3f}; on target nodes median "
                f"{np.median(ext_r[ft] / np.maximum(rr[ft], 1e-30)):.3f}"
            )
            qr = np.quantile(ratio_raw, [0.5, 0.9, 1.0])
            qe = np.quantile(ratio_eff, [0.5, 0.9, 1.0])
            print(
                f"   accepted cross pairs, (r_t+r_s)/d RAW: median {qr[0]:.3f} p90 {qr[1]:.3f} max {qr[2]:.3f}"
            )
            print(
                f"   same pairs under the LANE's extents:   median {qe[0]:.3f} p90 {qe[1]:.3f} max {qe[2]:.3f}"
                f"   -> {int((ratio_eff > THETA).sum())} of {nf} pairs would FAIL the lane's MAC "
                f"({100.0 * (ratio_eff > THETA).mean():.2f} %)"
            )
            asymmetry_report(
                rr[ft],
                sr[fs],
                d,
                "cross pairs (target = receiver node, source = imported)",
            )
            s_com, s_m, s_rcom = node_com(s_tree, *dom_pm[s])
            r_com_, _r_m, r_rcom = node_com(r_tree, *dom_pm[r])
            com_report(rc, rr, sc, sr, s_com, s_m, ft, fs, "cross pairs")
            com_report_exact(r_com_, r_rcom, s_com, s_rcom, s_m, ft, fs, "cross pairs")
            zero_leaf_src = int((sr[fs] <= 0).sum())
            print(
                f"   exported sources with zero raw radius (single-particle nodes): {zero_leaf_src}"
            )

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
            print(
                f"   control: dropped pairs failing the MAC when recomputed: {int((~ok_n).sum())} / {nn}"
            )
            q = np.quantile(theta_eff, [0.5, 0.9, 0.99, 1.0])
            print(
                f"   (r_t + r_s)/d of dropped pairs: median {q[0]:.3f}  p90 {q[1]:.3f}  "
                f"p99 {q[2]:.3f}  max {q[3]:.3f}   (theta = {THETA})"
            )

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
            print(
                f"   target sphere outside its cell's sphere: {int((overhang > 0).sum())} / {nn}"
                f"   (max overhang {overhang.max():.4f}, max cell radius {rr[cell_of].max():.3f})"
            )

            # size of the hole: sender particles behind the dropped sources, per target leaf
            s_nr = np.asarray(s_tree.node_ranges)
            src_nodes = imp_nodes[nsrc]
            cnt = s_nr[src_nodes, 1] - s_nr[src_nodes, 0] + 1
            per_leaf = {}
            for t, c in zip(nt.tolist(), cnt.tolist()):
                per_leaf[t] = per_leaf.get(t, 0) + c
            vals = np.asarray(list(per_leaf.values()), np.float64) / n_s_part
            print(
                f"   target leaves with a hole: {len(per_leaf)} / {int(r_tree.parent.shape[0]) - int(r_tree.left_child.shape[0])}"
                f"   sender fraction missing per such leaf: median {np.median(vals):.4f} max {vals.max():.4f}"
            )


def asymmetry_report(rt, rs, d, tag):
    """Per-pair truncation factors, not the MAC sum.

    The MAC bounds (r_t + r_s)/d. The M2L's two truncations are bounded
    separately: the multipole side by r_s/(d - r_t), the local side by
    r_t/(d - r_s). For a symmetric pair at the MAC limit each is ~0.667^(p+1);
    for a lopsided one (a small cell against a huge node) the bigger side is
    ~0.8^(p+1) -- and the one-sided export refines ONLY the source against a
    fixed small cell, so it is the construction that makes lopsided pairs.
    """
    ms = rs / np.maximum(d - rt, 1e-30)
    ml = rt / np.maximum(d - rs, 1e-30)
    eff = np.maximum(ms, ml)
    lop = np.maximum(rs, rt) / np.maximum(np.minimum(rs, rt), 1e-30)
    q = lambda a: np.quantile(a, [0.5, 0.9, 0.99])
    qs, ql, qe, qq = q(ms), q(ml), q(eff), q(lop)
    print(
        f"   {tag}: multipole-side r_s/(d-r_t) median {qs[0]:.3f} p90 {qs[1]:.3f} p99 {qs[2]:.3f} | "
        f"local-side r_t/(d-r_s) median {ql[0]:.3f} p90 {ql[1]:.3f} | "
        f"worse side median {qe[0]:.3f} p90 {qe[1]:.3f} p99 {qe[2]:.3f} | "
        f"r_big/r_small median {qq[0]:.1f} p90 {qq[1]:.1f}"
    )
    # the p-convergence this population predicts, worse side, p4 -> p6, error-weighted crudely by eff^(p+1)
    e4 = np.sum(eff**5)
    e6 = np.sum(eff**7)
    print(
        f"   {tag}: sum eff^(p+1) improvement p4 -> p6 = {e4 / max(e6, 1e-300):.2f}x  "
        f"(symmetric-at-theta pairs would give {(1/0.6667)**2:.2f}x, lopsided-at-theta {(1/0.8)**2:.2f}x)"
    )


def node_com(tree, pos, mass):
    """Mass centre and its offset from the geometric centre, per node.

    The fast lane's real-basis upward sweep expands about the COM
    (`center_mode='com'` only), while every MAC here is tested on the GEOMETRIC
    sphere. A multipole about the COM converges only outside the sphere that
    bounds the node's particles ABOUT THE COM, whose radius is up to
    r_geo + |COM - gcen| -- larger than the MAC's r_geo by the offset.
    """
    order = np.asarray(tree.particle_indices).astype(np.int64)
    ps = np.asarray(pos, np.float64)[order]
    ms = np.asarray(mass, np.float64)[order]
    cm = np.concatenate([[0.0], np.cumsum(ms)])
    cx = np.concatenate([np.zeros((1, 3)), np.cumsum(ms[:, None] * ps, axis=0)])
    nr = np.asarray(tree.node_ranges).astype(np.int64)
    st, en = nr[:, 0], nr[:, 1]
    live = en >= st
    m = np.where(
        live, cm[np.minimum(en + 1, len(cm) - 1)] - cm[np.minimum(st, len(cm) - 1)], 0.0
    )
    x = np.where(
        live[:, None],
        cx[np.minimum(en + 1, len(cm) - 1)] - cx[np.minimum(st, len(cm) - 1)],
        0.0,
    )
    com = np.where((m > 0)[:, None], x / np.maximum(m, 1e-300)[:, None], 0.0)
    # the EXACT convergence radius about the COM: the farthest particle from it.
    # r_geo + |COM - gcen| is only a bound, and a loose one (it reported 11-23 %
    # "divergent" pairs in populations whose error demonstrably converges).
    r_com = np.zeros(len(m))
    for i in np.flatnonzero(live & (m > 0)):
        seg = ps[st[i] : en[i] + 1]
        r_com[i] = np.sqrt(np.max(np.sum((seg - com[i]) ** 2, axis=1)))
    return com, m, r_com


def com_report_exact(t_com, t_rcom, s_com, s_rcom, s_mass, ft, fs, tag):
    """Both truncation factors about the lane's ACTUAL expansion centres (COM) with
    the EXACT particle radii about them; and the mass/distance-weighted error proxy."""
    d = np.linalg.norm(t_com[ft] - s_com[fs], axis=1)
    rho_m = s_rcom[fs] / np.maximum(d - t_rcom[ft], 1e-30)  # multipole side
    rho_l = t_rcom[ft] / np.maximum(d - s_rcom[fs], 1e-30)  # local side
    rho = np.maximum(rho_m, rho_l)
    q = lambda a: np.quantile(a, [0.5, 0.9, 0.99, 1.0])
    qm, qw = q(rho_m), q(rho)
    print(
        f"   {tag}: EXACT COM factors -- multipole side median {qm[0]:.3f} p90 {qm[1]:.3f} p99 {qm[2]:.3f} "
        f"max {qm[3]:.3f} | worse side median {qw[0]:.3f} p90 {qw[1]:.3f} p99 {qw[2]:.3f} max {qw[3]:.3f} | "
        f">=0.9: {int((rho>=0.9).sum())} ({100*(rho>=0.9).mean():.2f} %), >=1: {int((rho>=1).sum())} of {len(rho)}"
    )
    w = s_mass[fs] / np.maximum(d, 1e-30) ** 2
    # A pair whose series does not converge contributes an error of the order of
    # its whole field whatever p is -- a FLOOR -- so cap rho at 1 in the proxy
    # instead of letting rho^(p+1) blow up. The uncapped version reported 0.00x.
    rc = np.minimum(rho, 1.0)
    e = lambda p: float(np.sum(w * rc ** (p + 1)))
    share1 = float(np.sum(w[rho >= 1.0]) / max(np.sum(w), 1e-300))
    share09 = float(np.sum(w[rho >= 0.9]) / max(np.sum(w), 1e-300))
    print(
        f"   {tag}: EXACT error proxy sum m_s min(rho,1)^(p+1)/d^2: p4 -> p5 {e(4)/e(5):.2f}x, "
        f"p5 -> p6 {e(5)/e(6):.2f}x, p4 -> p6 {e(4)/e(6):.2f}x   (measured: reference 3.58x, distributed "
        f"excess 1.41x); weight share of non-converging pairs (rho >= 1) {100*share1:.2f} %, rho >= 0.9 {100*share09:.2f} %"
    )


def com_report(t_cen, t_rad, s_cen, s_rad, s_com, s_mass, ft, fs, tag):
    """Multipole-side truncation factor ABOUT THE COM, and a mass/distance-weighted
    error proxy sum m_s rho^(p+1) / d^2 whose p4 -> p6 ratio predicts how fast this
    population's M2L error converges."""
    delta = np.linalg.norm(s_com[fs] - s_cen[fs], axis=1)
    d_com = np.linalg.norm(t_cen[ft] - s_com[fs], axis=1)
    rho = (s_rad[fs] + delta) / np.maximum(d_com - t_rad[ft], 1e-30)
    q = np.quantile(rho, [0.5, 0.9, 0.99, 1.0])
    n_div = int((rho >= 1.0).sum())
    n_09 = int((rho >= 0.9).sum())
    w = s_mass[fs] / np.maximum(d_com, 1e-30) ** 2
    e = lambda p: float(np.sum(w * rho ** (p + 1)))
    print(
        f"   {tag}: COM-based multipole factor (r_geo+|COM-gcen|)/(d_com-r_t) median {q[0]:.3f} "
        f"p90 {q[1]:.3f} p99 {q[2]:.3f} max {q[3]:.3f}; >=0.9: {n_09} ({100*n_09/max(len(rho),1):.2f} %), "
        f">=1 (DIVERGENT): {n_div}; |COM-gcen|/r_geo median {np.median(delta/np.maximum(s_rad[fs],1e-30)):.3f} "
        f"p99 {np.quantile(delta/np.maximum(s_rad[fs],1e-30),0.99):.3f}"
    )
    print(
        f"   {tag}: error proxy sum m_s rho^(p+1)/d^2: p4 -> p5 {e(4)/e(5):.2f}x, p5 -> p6 {e(5)/e(6):.2f}x, "
        f"p4 -> p6 {e(4)/e(6):.2f}x   (measured: reference 3.58x, distributed excess 1.41x)"
    )


def local_pair_ratios(tree, geom, tag, pos=None, mass=None):
    """(r_t + r_s)/d over the pairs the LANE's own mutual walk accepts on ``tree``.

    The cross pairs cluster just below theta (median 0.68, p90 0.78): the export
    walk refines the SOURCE only, against a fixed receiver cell, and stops at the
    first admissible level. If the mutual walk's local pairs sit lower, cross
    pairs converge more slowly in p than local ones and the ratio to the
    single-GPU lane widens with order without any coverage hole.
    """
    from yggdrax.interactions import dual_tree_walk_mutual

    left, right = children_full(tree)
    idx = tree.parent.dtype
    cen = jnp.asarray(geom.center)
    rad = jnp.asarray(geom.radius)
    root = jnp.argmin(tree.parent).astype(idx)
    queue, fcap = 1 << 20, 1 << 22
    while True:
        res = dual_tree_walk_mutual(
            left,
            right,
            cen,
            rad,
            THETA,
            root,
            max_pair_queue=queue,
            far_cap=fcap,
            near_cap=1 << 22,
            mac_type=MAC,
        )
        if bool(res.queue_overflow):
            queue *= 4
            continue
        if bool(res.far_overflow):
            fcap *= 4
            continue
        break
    n = int(res.far_count)
    a = np.asarray(res.far_a)[:n]
    b = np.asarray(res.far_b)[:n]
    c = np.asarray(cen)
    r = np.asarray(rad)
    d = np.linalg.norm(c[a] - c[b], axis=1)
    ratio = (r[a] + r[b]) / d
    q = np.quantile(ratio, [0.1, 0.5, 0.9, 0.99, 1.0])
    print(
        f"   {tag}: {n} mutual far pairs, (r_t+r_s)/d p10 {q[0]:.3f} median {q[1]:.3f} "
        f"p90 {q[2]:.3f} p99 {q[3]:.3f} max {q[4]:.3f}; near pairs {int(res.near_count)}"
    )
    # mutual pairs serve BOTH directions, so either node is a target: report both orientations
    asymmetry_report(
        np.concatenate([r[a], r[b]]),
        np.concatenate([r[b], r[a]]),
        np.concatenate([d, d]),
        tag,
    )
    if pos is not None:
        com, m, rcom = node_com(tree, pos, mass)
        com_report(
            c, r, c, r, com, m, np.concatenate([a, b]), np.concatenate([b, a]), tag
        )
        com_report_exact(
            com, rcom, com, rcom, m, np.concatenate([a, b]), np.concatenate([b, a]), tag
        )
    return ratio


if __name__ == "__main__":
    main()

if __name__ == "__main__" and os.environ.get("PROBE_LOCAL_PAIRS") == "1":
    pos, mass = load_ic(IC, N)
    bounds = infer_bounds(jnp.asarray(pos))
    print("\n== the LANE's own pairs, for comparison with the cross pairs above")
    t_full, g_full = build_domain_tree(pos, mass, bounds, LEAF)
    local_pair_ratios(t_full, g_full, "single-GPU reference tree (all N)", pos, mass)
    dom = morton_domains(pos, bounds, NDEV)
    for d in range(NDEV):
        sel = dom == d
        t, g = build_domain_tree(pos[sel], mass[sel], bounds, LEAF)
        local_pair_ratios(t, g, f"device {d} local tree", pos[sel], mass[sel])
