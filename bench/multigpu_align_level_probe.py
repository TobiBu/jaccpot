"""Phase 2: what does Morton ``align_level`` actually buy, now that Phase 3 is built?

The plan gave two reasons to snap domain boundaries to level-k cell edges.

**One of them is void.** "It makes each device's node set disjoint" was true of the LET
design, where every device built a coarse tree over other devices' leaves in a shared
numbering. The lane built in Phase 3 gives each device its own tree over its own
particles, in its own index space, so the node sets are disjoint whatever the partition
does, and the exchange addresses cells by an occupancy cut of real tree nodes rather
than by Morton cells. Alignment is not needed for either.

**The other is a real, testable claim**: a pivot cutting INSIDE a cell splits its
particles across two devices, and the two half-cells occupy the SAME region on either
side of the boundary. Each is then spatially interleaved with the other's domain, so
each must import for the other -- inflating exactly the cross-domain work Phase 3
measured. This probe tests that claim and prices it.

Reported, with and without alignment:

* **straddle**: level-k cells holding particles from more than one device, which is the
  mechanism, and how many particles sit in them;
* **import**: the distinct source nodes and particles a device must actually take, from
  the same per-domain trees and the same real cross walk as Phase 3.1/3.4 -- the
  outcome, which is what decides whether the mechanism matters;
* **imbalance**: what alignment costs, since snapping pivots gives up exact equal
  counts. The plan predicted about 1.5 % of N/device, bounded by one cell.

The control is ``align_level=None`` against itself: the partition is deterministic, so
two identical runs must agree exactly, or the comparison is measuring noise.

Run:

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \\
    PROBE_N=200000 PROBE_NDEVS=2,4 python bench/multigpu_align_level_probe.py
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

from multigpu_import_volume_probe import build_domain_tree  # noqa: E402
from multigpu_oversized_cell_probe import load_ic  # noqa: E402
from yggdrax._cell_partition import MORTON_LEVELS  # noqa: E402
from yggdrax.bounds import infer_bounds  # noqa: E402
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl  # noqa: E402
from yggdrax.morton import morton_encode  # noqa: E402

N = int(os.environ.get("PROBE_N", "200000"))
LEAF = int(os.environ.get("PROBE_LEAF", "64"))
IC = os.environ.get("PROBE_IC", "plummer")
NDEVS = [int(x) for x in os.environ.get("PROBE_NDEVS", "2,4").split(",")]
LEVELS = [int(x) for x in os.environ.get("PROBE_ALIGN", "2,3,4,5").split(",")]
STRADDLE_AT = [int(x) for x in os.environ.get("PROBE_STRADDLE_AT", "6,8,10").split(",")]


def cross_import(t_dst, g_dst, t_src, g_src, theta=0.8):
    """Imported source NODES, and the NEAR particles behind them.

    Two numbers because only one of them can move. The node count is the payload the
    exchange pays for. The PARTICLE count is reported for the near list ONLY: the far
    list reaches nodes near the root, whose ranges cover the sender outright, so a
    coverage figure over far+near is pinned at 1.000 by construction and can never
    show a reduction. That pinning cost two runs before it was spotted.
    """
    kf, kn, queue = 256, 1024, 1 << 16
    full = int(t_src.parent.shape[0])
    while True:
        res = dual_tree_walk_cross_impl(
            t_dst,
            g_dst,
            t_src,
            g_src,
            float(theta),
            mac_type="dehnen",
            max_interactions_per_node=kf,
            max_neighbors_per_leaf=kn,
            max_pair_queue=queue,
            collect_far=True,
            collect_near=True,
        )
        if bool(res.queue_overflow):
            queue *= 4
            continue
        if bool(res.far_overflow):
            kf = min(kf * 4, full)
            continue
        if bool(res.near_overflow):
            kn = min(kn * 4, full)
            continue
        break
    tt = np.asarray(res.interaction_targets)
    far = np.asarray(res.interaction_sources)[tt >= 0]
    nbr = np.asarray(res.neighbor_indices)
    near = nbr[nbr >= 0]
    nodes = np.unique(np.concatenate([far[far >= 0], near]))
    nr = np.asarray(t_src.node_ranges)
    n_s = int(t_src.num_particles)
    cov = np.zeros(max(n_s, 1), bool)
    for node in np.unique(near).tolist():
        lo, hi = int(nr[node, 0]), int(nr[node, 1])
        if hi >= lo and lo < n_s:
            cov[lo : min(hi + 1, n_s)] = True
    return int(nodes.size), int(cov.sum())


def partition(codes_sorted, ndev, align_level):
    """Equal-count Morton pivots, optionally snapped to level-k cell edges.

    Mirrors ``yggdrax.distributed.partition._align_pivots``: masking the low
    ``3 * (MORTON_LEVELS - k)`` bits moves every pivot down to a level-k cell edge,
    so no cell at depth >= k can straddle two domains.
    """
    n = codes_sorted.size
    cuts = np.array([(i + 1) * n // ndev for i in range(ndev - 1)], np.int64)
    pivots = codes_sorted[cuts].astype(np.uint64)
    if align_level is not None:
        shift = np.uint64(3 * (MORTON_LEVELS - int(align_level)))
        pivots = (pivots >> shift) << shift
    return np.searchsorted(pivots, codes_sorted, side="right")


def straddle(codes_sorted, dom, level):
    """Level-``level`` cells holding particles from more than one device."""
    shift = np.uint64(3 * (MORTON_LEVELS - int(level)))
    cell = codes_sorted >> shift
    uniq, inv = np.unique(cell, return_inverse=True)
    lo = np.full(uniq.size, 1 << 30, np.int64)
    hi = np.full(uniq.size, -1, np.int64)
    np.minimum.at(lo, inv, dom)
    np.maximum.at(hi, inv, dom)
    split = hi > lo
    return int(split.sum()), int(np.isin(inv, np.flatnonzero(split)).sum())


def main():
    pos, mass = load_ic(IC, N)
    bounds = infer_bounds(jnp.asarray(pos))
    codes = np.asarray(morton_encode(jnp.asarray(pos), bounds))
    order = np.argsort(codes, kind="stable")
    cs, ps, ms = codes[order], pos[order], mass[order]
    print(f"IC={IC} N={N} leaf={LEAF}")

    for ndev in NDEVS:
        print(f"\n== ndev={ndev} N/dev={N // ndev}")
        hdr = (
            f"{'align':>6} {'cells':>6} {'imbal%':>8} "
            + " ".join(f"{'strdl@' + str(L):>9}" for L in STRADDLE_AT)
            + f" {'impNodes':>9} {'nearPart':>9} {'near/own':>8}"
        )
        print(hdr)
        base = None
        for align in [None, None] + LEVELS:  # the repeated None is the control
            dom = partition(cs, ndev, align)
            # how many level-`align` cells the system occupies AT ALL: alignment
            # cannot balance across fewer cells than there are devices
            if align is None:
                occ = 0
            else:
                sh = np.uint64(3 * (MORTON_LEVELS - int(align)))
                occ = int(np.unique(cs >> sh).size)
            counts = np.bincount(dom, minlength=ndev)
            imbal = 100.0 * (counts.max() - counts.min()) / counts.mean()
            st = [straddle(cs, dom, L) for L in STRADDLE_AT]

            if counts.min() < LEAF:
                # an aligned pivot can empty a device outright; that is the result,
                # not an error, so report it instead of crashing on an empty tree
                print(
                    f"{str(align):>6} {occ:>6} {imbal:>8.3f} "
                    + " ".join(f"{x[0]:>9}" for x in st)
                    + f" {'--':>9} {'--':>9} {'--':>8}   "
                    f"UNUSABLE: a device holds {counts.min()} particles"
                )
                continue
            trees, geoms = [], []
            for d in range(ndev):
                m = dom == d
                t, g = build_domain_tree(ps[m], ms[m], bounds, LEAF)
                trees.append(t)
                geoms.append(g)
            worst_nodes = worst_part = 0
            for r in range(ndev):
                for s in range(ndev):
                    if r == s:
                        continue
                    nodes, parts = cross_import(trees[r], geoms[r], trees[s], geoms[s])
                    worst_nodes = max(worst_nodes, nodes)
                    worst_part = max(worst_part, parts)
            own = counts.mean()
            row = (
                f"{str(align):>6} {occ:>6} {imbal:>8.3f} "
                + " ".join(f"{s[0]:>9}" for s in st)
                + f" {worst_nodes:>9} {worst_part:>9} {worst_part / own:>8.3f}"
            )
            print(row)
            if align is None:
                if base is None:
                    base = row
                elif row != base:
                    raise RuntimeError(
                        "CONTROL FAILED: two identical unaligned runs disagree, so "
                        "this comparison is measuring noise\n"
                        f"  {base}\n  {row}"
                    )


if __name__ == "__main__":
    main()
