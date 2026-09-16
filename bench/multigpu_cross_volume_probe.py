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
print(f"{'leaf':>6} {'ndev':>5} {'align':>6} {'leaves':>8} {'near vol/N':>11} "
      f"{'cross frac':>11} {'far cross':>10} {'exp nodes/pair':>15} {'exp parts/pair':>15}")

for LEAF in (int(x) for x in os.environ.get("PROBE_LEAVES", "64,256").split(",")):
    k = int(adaptive_cell_leaf_partition_numpy(np.sort(codes), leaf_size=LEAF)[0].size)
    cap = 1 << int(np.ceil(np.log2(1.25 * k)))
    tree, psorted, msorted, inv = build_static_cells_tree(
        P, jnp.asarray(mass), B, leaf_size=LEAF, leaf_capacity=cap, return_reordered=True)
    geom = compute_tree_geometry(tree, jnp.asarray(psorted), max_leaf_size=LEAF)
    nr = np.asarray(tree.node_ranges); nint = int(tree.left_child.shape[0])
    occ = np.where(nr[:,1] >= nr[:,0], nr[:,1]-nr[:,0]+1, 0)

    res = dual_tree_walk_mutual(tree, geom, THETA, mac_type="dehnen",
                                node_active=jnp.asarray(nr[:,1] >= nr[:,0]))
    fa = np.asarray(res.far_a); fb = np.asarray(res.far_b)
    na = np.asarray(res.near_a); nb = np.asarray(res.near_b)
    fa, fb = fa[fa >= 0], fb[fb >= 0]
    na, nb = na[na >= 0], nb[nb >= 0]

    # particle -> domain, then sorted-particle -> domain, then leaf -> domain
    perm = np.asarray(tree.particle_indices)
    for ndev in (int(x) for x in os.environ.get("PROBE_NDEVS", "2,4,8").split(",")):
        align = max(1, int(np.floor(np.log(512 * ndev) / np.log(8))))
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
        cross_vol = (occ[na][cross_mask].astype(np.int64) * occ[nb][cross_mask]).sum() * 2
        far_cross = float(np.mean(leaf_dom_far := (leaf_dom[np.clip(fa,0,None)] != leaf_dom[np.clip(fb,0,None)]))) if fa.size else 0.0

        # exported node set / particles per ordered domain pair (a LET's payload)
        exp_nodes, exp_parts = [], []
        for src in range(ndev):
            for dst in range(ndev):
                if src == dst: continue
                m = (leaf_dom[np.clip(fb,0,None)] == src) & (leaf_dom[np.clip(fa,0,None)] == dst)
                m2 = (leaf_dom[np.clip(fa,0,None)] == src) & (leaf_dom[np.clip(fb,0,None)] == dst)
                nodes = np.unique(np.concatenate([fb[m], fa[m2]])) if (m.any() or m2.any()) else np.empty(0, int)
                exp_nodes.append(nodes.size)
                q = (leaf_dom[nb] == src) & (leaf_dom[na] == dst)
                q2 = (leaf_dom[na] == src) & (leaf_dom[nb] == dst)
                leaves = np.unique(np.concatenate([nb[q], na[q2]])) if (q.any() or q2.any()) else np.empty(0, int)
                exp_parts.append(int(occ[leaves].sum()))
        print(f"{LEAF:>6} {ndev:>5} {align:>6} {int((occ[nint:]>0).sum()):>8} "
              f"{near_vol/N:>11.4f} {cross_vol/max(near_vol,1):>11.3f} {far_cross:>10.3f} "
              f"{np.mean(exp_nodes):>15.0f} {np.mean(exp_parts):>15.0f}"
              + (f"   straddling leaves={straddle}" if straddle else ""))
