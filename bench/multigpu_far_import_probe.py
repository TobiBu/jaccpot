"""Phase 3.4: how many distinct source NODES does a device need as FAR sources?

3.1 and 3.3 sized the NEAR half of the cross-domain import -- which of a neighbour's
PARTICLES a device must receive. The far half is a different object and a different
currency: the receiver needs a MULTIPOLE per accepted source node, not particles, and
the same ordered domain pair that produced 17,385 near pairs produced **56,821 far
pairs**. Pairs are not the payload, though: the exchange ships each distinct source
node ONCE however many target nodes name it, so the question Phase 3.5 has to be sized
on is the count of distinct nodes and their depth, not the pair count.

What this reports, per (IC, ndev) and per ordered (receiver, sender) pair:

* far pairs, and the distinct source NODES behind them -- the compression the union
  buys over the pair list;
* those nodes as a fraction of the sender's whole tree, and their DEPTH distribution.
  Shallow means cheap: a far list dominated by nodes near the root is a handful of
  multipoles whatever the pair count says;
* the payload in bytes at several expansion orders, against the near import measured
  the same way on the same trees, so the two halves are finally comparable. A real
  multipole of order p is (p+1)^2 coefficients; a particle is 4 floats (position and
  mass). At p = 4 one node costs 6.25 particles, so the far half only matters if the
  node count is within about a factor of six of the particle count.

The statistic is the WORST ordered pair, not a mean over devices -- at a plane split of
a Plummer sphere the two devices' near imports differ threefold, and there is no reason
the far half is better behaved.

Capacities are GROWN until the walk reports no overflow, never accepted as a
truncation. Both `far_overflow` and `near_overflow` sit in the walk's `cond_fun`, so an
overflowing row does not truncate itself -- it HALTS the walk and every other row loses
its remaining rounds. That misreads the import SMALLER than it is, i.e. in the
direction that flatters the design; it cost a 30x wrong number in Phase 3.3 before it
was caught, and `multigpu_import_locality_probe.py` still asserts on `near_overflow`
but not on `far_overflow`.

Run:

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \\
    PROBE_IC=plummer PROBE_N=200000 PROBE_NDEVS=2 python bench/multigpu_far_import_probe.py
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

from multigpu_import_volume_probe import (  # noqa: E402
    build_domain_tree,
    morton_domains,
    plane_domains,
    rcb_domains,
)
from multigpu_oversized_cell_probe import load_ic  # noqa: E402
from yggdrax.bounds import infer_bounds  # noqa: E402
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl  # noqa: E402

N = int(os.environ.get("PROBE_N", "200000"))
LEAF = int(os.environ.get("PROBE_LEAF", "64"))
THETA = float(os.environ.get("PROBE_THETA", "0.8"))
IC = os.environ.get("PROBE_IC", "plummer")
NDEVS = [int(x) for x in os.environ.get("PROBE_NDEVS", "2,4").split(",")]
PARTS = os.environ.get("PROBE_PARTS", "morton").split(",")
ORDERS = [int(x) for x in os.environ.get("PROBE_ORDERS", "4,5,6").split(",")]

BYTES_PER_PARTICLE = 4 * 4  # position (3) + mass, fp32
BYTES_PER_COEFF = 4  # fp32 real multipole coefficient


def cross_walk(t_dst, g_dst, t_src, g_src, n_leaves_src):
    """One ordered pair's far AND near lists, exact.

    Both output capacities and the pair queue are grown until the walk reports no
    overflow; see the module docstring for why a cap may never be accepted here.
    """
    full_leaf = max(64, int(n_leaves_src))
    full_node = max(64, int(t_src.parent.shape[0]))
    kf, kn, queue = 256, 1024, 1 << 16
    while True:
        res = dual_tree_walk_cross_impl(
            t_dst,
            g_dst,
            t_src,
            g_src,
            float(THETA),
            mac_type="dehnen",
            max_interactions_per_node=kf,
            max_neighbors_per_leaf=kn,
            max_pair_queue=queue,
            collect_far=True,
            collect_near=True,
        )
        if bool(res.queue_overflow):
            queue *= 4
            if queue > (1 << 24):
                raise RuntimeError("pair queue did not fit at 2^24")
            continue
        if bool(res.far_overflow):
            if kf >= full_node:
                raise RuntimeError(f"far overflow at the full node count {full_node}")
            kf = min(kf * 4, full_node)
            continue
        if bool(res.near_overflow):
            if kn >= full_leaf:
                raise RuntimeError(f"near overflow at the full leaf count {full_leaf}")
            kn = min(kn * 4, full_leaf)
            continue
        return res


def main():
    pos, mass = load_ic(IC, N)
    bounds = infer_bounds(jnp.asarray(pos))
    print(f"IC={IC} N={N} leaf={LEAF} theta={THETA}")

    for part in PARTS:
        for ndev in NDEVS:
            if part == "plane":
                dom = plane_domains(pos, ndev)
            elif part == "rcb":
                dom = rcb_domains(pos, ndev)
            else:
                dom = morton_domains(pos, bounds, ndev)

            trees, geoms, nloc = [], [], []
            for d in range(ndev):
                m = dom == d
                nloc.append(int(m.sum()))
                t, g = build_domain_tree(pos[m], mass[m], bounds, LEAF)
                trees.append(t)
                geoms.append(g)

            # per source domain: node depth, and which nodes are live
            depth, live_node, occ = [], [], []
            for d in range(ndev):
                nr = np.asarray(trees[d].node_ranges)
                lv = (nr[:, 1] >= nr[:, 0]) & (nr[:, 0] < nloc[d])
                live_node.append(lv)
                occ.append(np.where(lv, nr[:, 1] - nr[:, 0] + 1, 0))
                depth.append(np.asarray(trees[d].node_level, np.int64))

            print(f"\n== {part} ndev={ndev} N/dev={N // ndev}")
            print(
                f"  source trees: "
                + ", ".join(
                    f"dev{d} {int(live_node[d].sum())} nodes "
                    f"(depth {int(depth[d][live_node[d]].max())})"
                    for d in range(ndev)
                )
            )
            hdr = (
                f"{'dst<-src':>9} {'farPairs':>9} {'farNodes':>9} {'/srcTree':>9} "
                f"{'dMed':>5} {'dP90':>5} {'dMax':>5} {'nearPart':>9} "
                + " ".join(f"{'p' + str(p) + 'far/near':>12}" for p in ORDERS)
            )
            print(hdr)

            worst = {}
            for dst in range(ndev):
                for src in range(ndev):
                    if src == dst:
                        continue
                    res = cross_walk(
                        trees[dst],
                        geoms[dst],
                        trees[src],
                        geoms[src],
                        int(
                            (
                                live_node[src]
                                & (
                                    np.arange(live_node[src].size)
                                    >= int(trees[src].left_child.shape[0])
                                )
                            ).sum()
                        ),
                    )
                    tt = np.asarray(res.interaction_targets)
                    ss = np.asarray(res.interaction_sources)
                    far_src = ss[tt >= 0]
                    far_pairs = int(far_src.size)
                    nodes = np.unique(far_src[far_src >= 0])
                    nodes = nodes[live_node[src][nodes]]
                    n_nodes = int(nodes.size)

                    nbr = np.asarray(res.neighbor_indices)
                    used = np.unique(nbr[nbr >= 0])
                    used = used[live_node[src][used]]
                    near_part = int(occ[src][used].sum()) if used.size else 0

                    # is the union one saturated row again, or broadly spread?
                    order = np.argsort(tt[tt >= 0], kind="stable")
                    keys = tt[tt >= 0][order]
                    _, starts = np.unique(keys, return_index=True)
                    rows = np.diff(np.append(starts, keys.size))

                    dd = depth[src][nodes] if n_nodes else np.zeros(1, np.int64)
                    n_src_nodes = int(live_node[src].sum())
                    # A pair with no near import at all has no ratio -- printing
                    # far/max(near,1) there reads as a colossal number when it only
                    # means the denominator is zero.
                    ratios = []
                    for p in ORDERS:
                        far_b = n_nodes * (p + 1) ** 2 * BYTES_PER_COEFF
                        near_b = near_part * BYTES_PER_PARTICLE
                        ratios.append(far_b / near_b if near_b else None)
                    print(
                        f"{dst:>4}<-{src:<3} {far_pairs:>9} {n_nodes:>9} "
                        f"{n_nodes / max(n_src_nodes, 1):>9.3f} "
                        f"{int(np.median(dd)):>5} {int(np.quantile(dd, 0.9)):>5} "
                        f"{int(dd.max()):>5} {near_part:>9} "
                        + " ".join(
                            f"{r:>12.3f}" if r is not None else f"{'--':>12}"
                            for r in ratios
                        )
                        + f"   rows med/p99/max {int(np.median(rows))}/"
                        f"{int(np.quantile(rows, 0.99))}/{int(rows.max())}"
                    )
                    key = (n_nodes, near_part, far_pairs)
                    if key[0] > worst.get("nodes", (0,))[0]:
                        worst["nodes"] = key
            if "nodes" in worst:
                n_nodes, near_part, far_pairs = worst["nodes"]
                print(
                    f"  WORST ordered pair: {n_nodes} far nodes ({far_pairs} pairs, "
                    f"{far_pairs / max(n_nodes, 1):.1f} pairs per node shipped), "
                    f"{near_part} near particles"
                )
                for p in ORDERS:
                    print(
                        f"    p={p}: far {n_nodes * (p + 1) ** 2 * BYTES_PER_COEFF / 1e6:.3f} MB "
                        f"vs near {near_part * BYTES_PER_PARTICLE / 1e6:.3f} MB"
                    )


if __name__ == "__main__":
    main()
