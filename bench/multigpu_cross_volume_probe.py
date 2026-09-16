"""Plan Phase 3.1: what does a correct cross-domain export actually cost?

Measures, on the GLOBAL cell tree under the real COM MAC, for Morton-aligned
domains:
  (a) cross-domain NEAR volume as a fraction of the local near volume -- the one
      that can invalidate the whole approach;
  (b) the exported source NODE set per ordered domain pair (what a LET ships);
  (c) the opened leaves' PARTICLES per ordered domain pair.

Why (a) first: with bucket leaves the cross half is already the majority of near
work (54 % of near leaf pairs at 4 devices, 80 % at 6, from the 21M rollout).
Cell leaves cut the LOCAL near volume 24x on one card; their effect on the CROSS
share has never been measured, and near cost is leaf_pairs x leaf^2 -- shrinking
the leaf multiplies the pair count while dividing the per-pair cost, and the two
need not cancel the same way across a domain boundary as inside one.

Geometry caveat: this uses the BOX geometry (compute_tree_geometry) rather than
the COM geometry the fused lane's MAC actually tests, because the ratio being
measured -- what fraction of the near volume crosses a domain boundary -- is a
partition property and far less sensitive to that choice than the accepted set
itself is. If a ratio comes out borderline, re-measure with com_mac_geometry
before deciding anything on it.

CPU-friendly at small N; use a card for N >= 1e5.
"""
import os, sys
sys.path.insert(0, "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from yggdrax._tree_impl import build_static_cells_tree, build_static_radix_tree
from yggdrax.bounds import infer_bounds
from yggdrax.morton import morton_encode
from yggdrax._cell_partition import adaptive_cell_leaf_partition_numpy, MORTON_LEVELS
from yggdrax.interactions import dual_tree_walk_mutual
from yggdrax.geometry import compute_tree_geometry
from common.ic import IC_GENERATORS

N     = int(os.environ.get("PROBE_N", "50000"))
IC    = os.environ.get("PROBE_IC", "plummer")
THETA = float(os.environ.get("PROBE_THETA", "0.8"))

def rcb_domains(points, ndev):
    """Recursive coordinate bisection: split the longest axis at the median.

    Morton ranges are compact in curve order but their SPATIAL boundary is
    fractal, so a thin spread of cross pairs can still reach every leaf of a
    neighbour. RCB gives compact boxes and is the control that tells the two
    apart.
    """
    if ndev & (ndev - 1):
        raise ValueError(f"rcb_domains wants a power of two, got {ndev}")
    dom = np.zeros(len(points), np.int32)
    groups = [np.arange(len(points))]
    while len(groups) < ndev:
        nxt = []
        for g in groups:
            ext = points[g].max(0) - points[g].min(0)
            ax = int(np.argmax(ext))
            o = g[np.argsort(points[g][:, ax], kind="stable")]
            mid = len(o) // 2
            nxt.append(o[:mid])
            nxt.append(o[mid:])
        groups = nxt
    for d, g in enumerate(groups):
        dom[g] = d
    return dom, [int((dom == d).sum()) for d in range(ndev)]


def domains_of(codes, ndev, align_level):
    """Morton-range domains snapped to level-`align_level` cell edges."""
    shift = np.uint64(max(0, 63 - 3 * align_level))
    cell = codes >> shift
    order = np.argsort(codes, kind="stable")
    # split by equal counts, then snap the cut to a cell boundary
    cuts = [0]
    for d in range(1, ndev):
        i = d * len(order) // ndev
        c = cell[order[i]]
        while i < len(order) and cell[order[i]] == c:
            i += 1
        cuts.append(i)
    cuts.append(len(order))
    dom = np.empty(len(order), np.int32)
    for d in range(ndev):
        dom[order[cuts[d]:cuts[d+1]]] = d
    return dom, [cuts[d+1]-cuts[d] for d in range(ndev)]

_ic = IC_GENERATORS[IC](N, seed=0)
pos = np.asarray(_ic[0], np.float32); mass = np.asarray(_ic[1], np.float32)
P = jnp.asarray(pos); B = infer_bounds(P)
codes = np.asarray(morton_encode(P, B))

print(f"IC={IC} N={N} theta={THETA}")
print(f"{'leaf':>6} {'ndev':>5} {'N/dev':>9} {'srcs/tgt':>9} {'xpair':>9} "
      f"{'cross frac':>10} {'far cross':>9} {'impNodes':>10} {'impParts':>11} {'halo/local':>10} {'impLeaf':>9} {'perSrc':>8}")

for LEAF in (int(x) for x in os.environ.get("PROBE_LEAVES", "64,256").split(",")):
    k = int(adaptive_cell_leaf_partition_numpy(np.sort(codes), leaf_size=LEAF)[0].size)
    cap = 1 << int(np.ceil(np.log2(1.25 * k)))
    tree, psorted, msorted, inv = build_static_cells_tree(
        P, jnp.asarray(mass), B, leaf_size=LEAF, leaf_capacity=cap, return_reordered=True)
    geom = compute_tree_geometry(tree, jnp.asarray(psorted), max_leaf_size=LEAF)
    nr = np.asarray(tree.node_ranges); nint = int(tree.left_child.shape[0])
    occ = np.where(nr[:,1] >= nr[:,0], nr[:,1]-nr[:,0]+1, 0)

    total = nr.shape[0]
    def _pad(child):
        c = jnp.asarray(child)
        return jnp.concatenate([c, jnp.full((total - c.shape[0],), -1, c.dtype)])
    root = int(np.argmin(np.asarray(tree.parent)))
    # LIVE LEAVES, not live nodes: occ is indexed over the whole node array, so
    # counting occ>0 across it includes every internal node and doubles the
    # figure -- which makes a per-domain leaf count look half its true size and
    # inverts the reading of the import volumes.
    live = int((occ[nint:] > 0).sum())
    # Caps sized from the live leaf count: the walk allocates these outright, and
    # an unsized call asks for >10 GiB at this N.
    far_cap  = 1 << int(np.ceil(np.log2(max(1 << 16, 80 * live))))
    near_cap = 1 << int(np.ceil(np.log2(max(1 << 16, 40 * live))))
    queue    = 1 << int(np.ceil(np.log2(max(1 << 15, 6 * live))))
    res = dual_tree_walk_mutual(
        _pad(tree.left_child), _pad(tree.right_child),
        jnp.asarray(geom.center), jnp.asarray(geom.radius),
        float(THETA), jnp.asarray(root),
        max_pair_queue=queue, far_cap=far_cap, near_cap=near_cap,
        mac_type="dehnen", node_active=jnp.asarray(nr[:, 1] >= nr[:, 0]))
    if bool(res.far_overflow) or bool(res.near_overflow) or bool(res.queue_overflow):
        print(f"  leaf={LEAF}: OVERFLOW (far={bool(res.far_overflow)} "
              f"near={bool(res.near_overflow)} queue={bool(res.queue_overflow)}) -- caps too small")
        continue
    nf = int(res.far_count); nn = int(res.near_count)
    fa = np.asarray(res.far_a)[:nf]; fb = np.asarray(res.far_b)[:nf]
    na = np.asarray(res.near_a)[:nn]; nb = np.asarray(res.near_b)[:nn]
    keep_f = (fa >= 0) & (fb >= 0); fa, fb = fa[keep_f], fb[keep_f]
    keep_n = (na >= 0) & (nb >= 0); na, nb = na[keep_n], nb[keep_n]
    leaf_occ_all = occ[nint:]
    live_occ = leaf_occ_all[leaf_occ_all > 0]
    print(f"  leaf={LEAF}: live leaves={live} far={fa.size} near={na.size} "
          f"| occupancy mean={live_occ.mean():.1f} p50={np.median(live_occ):.0f} "
          f"p99={np.percentile(live_occ,99):.0f} max={live_occ.max()}", flush=True)

    # particle -> domain, then sorted-particle -> domain, then leaf -> domain
    perm = np.asarray(tree.particle_indices)
    for ndev in (int(x) for x in os.environ.get("PROBE_NDEVS", "2,4,8").split(",")):
        align = max(1, int(np.floor(np.log(512 * ndev) / np.log(8))))
        part_mode = os.environ.get("PROBE_PART", "morton")
        if part_mode == "rcb":
            dom, sizes = rcb_domains(pos, ndev)
        else:
            dom, sizes = domains_of(codes, ndev, align)
        dom_sorted = dom[perm]
        leaf_dom = np.full(nr.shape[0], -1, np.int32)
        straddle = 0
        for nd in range(nint, nr.shape[0]):
            a, b = nr[nd]
            if b >= a:
                seg = dom_sorted[a:b+1]
                leaf_dom[nd] = seg[0]
                if np.any(seg != seg[0]): straddle += 1

        near_vol = (occ[na].astype(np.int64) * occ[nb]).sum() * 2   # both directions
        cross_mask = leaf_dom[na] != leaf_dom[nb]
        cross_pair_frac = float(cross_mask.mean()) if cross_mask.size else 0.0
        cross_vol = (occ[na][cross_mask].astype(np.int64) * occ[nb][cross_mask]).sum() * 2
        far_cross = float(np.mean(leaf_dom_far := (leaf_dom[np.clip(fa,0,None)] != leaf_dom[np.clip(fb,0,None)]))) if fa.size else 0.0

        # Per-DEVICE import: the union over every source domain of the leaves this
        # device needs. Reported per device, because a mean over ordered pairs is
        # dominated by the many distant pairs that export nothing.
        import_parts, import_leaves, import_nodes = [], [], []
        for dst in range(ndev):
            need_leaves, need_nodes = [], []
            for src in range(ndev):
                if src == dst:
                    continue
                q = (leaf_dom[nb] == src) & (leaf_dom[na] == dst)
                q2 = (leaf_dom[na] == src) & (leaf_dom[nb] == dst)
                if q.any() or q2.any():
                    need_leaves.append(np.concatenate([nb[q], na[q2]]))
                m = (leaf_dom[np.clip(fb, 0, None)] == src) & (leaf_dom[np.clip(fa, 0, None)] == dst)
                m2 = (leaf_dom[np.clip(fa, 0, None)] == src) & (leaf_dom[np.clip(fb, 0, None)] == dst)
                if m.any() or m2.any():
                    need_nodes.append(np.concatenate([fb[m], fa[m2]]))
            u_leaves = np.unique(np.concatenate(need_leaves)) if need_leaves else np.empty(0, int)
            u_nodes = np.unique(np.concatenate(need_nodes)) if need_nodes else np.empty(0, int)
            import_parts.append(int(occ[u_leaves].sum()))
            import_leaves.append(int(u_leaves.size))
            import_nodes.append(int(u_nodes.size))
        exp_nodes, exp_parts, exp_leaf_frac, exp_occ = [], [], [], []
        for src in range(0):
            for dst in range(0):
                pass
                m = (leaf_dom[np.clip(fb,0,None)] == src) & (leaf_dom[np.clip(fa,0,None)] == dst)
                m2 = (leaf_dom[np.clip(fa,0,None)] == src) & (leaf_dom[np.clip(fb,0,None)] == dst)
                nodes = np.unique(np.concatenate([fb[m], fa[m2]])) if (m.any() or m2.any()) else np.empty(0, int)
                exp_nodes.append(nodes.size)
                q = (leaf_dom[nb] == src) & (leaf_dom[na] == dst)
                q2 = (leaf_dom[na] == src) & (leaf_dom[nb] == dst)
                leaves = np.unique(np.concatenate([nb[q], na[q2]])) if (q.any() or q2.any()) else np.empty(0, int)
                exp_parts.append(int(occ[leaves].sum()))
                src_leaves = int((leaf_dom == src).sum())
                exp_leaf_frac.append(leaves.size / max(src_leaves, 1))
                if leaves.size:
                    exp_occ.append(float(occ[leaves].mean()))
        per_dev = N / ndev
        halo_ratio = float(np.mean(import_parts)) / per_dev
        leaves_per_dom = live / ndev
        # share of a SINGLE foreign domain's leaves that this device imports
        per_src_share = float(np.mean(import_leaves)) / max(ndev - 1, 1) / max(leaves_per_dom, 1)
        imp_leaves = float(np.mean(import_leaves))
        imp_nodes = float(np.mean(import_nodes))
        imp_occ = float(np.mean(import_parts)) / max(imp_leaves, 1.0)
        print(f"{LEAF:>6} {ndev:>5} {per_dev:>9.0f} {near_vol/N:>9.1f} {cross_pair_frac:>9.3f} "
              f"{cross_vol/max(near_vol,1):>10.3f} {far_cross:>9.3f} "
              f"{imp_nodes:>10.0f} {np.mean(import_parts):>11.0f} {halo_ratio:>10.2f} "
              f"{imp_leaves:>9.0f} {per_src_share:>8.3f}"
              + (f"  straddle={straddle}" if straddle else ""))
