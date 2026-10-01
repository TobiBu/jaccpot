"""Phase 3.3: how few cells must be reclassified to make the cross import affordable?

``multigpu_import_volume_probe.py`` established the pathology: with per-domain cell
trees at cells64 the cross import is **one leaf**. Half of all target leaves import
nothing (median 0 cross-neighbours, p90 8, p99 23) and one leaf imports 5,523 of the
5,534 source leaves there are. That leaf's radius is 119.88 against a median of 0.128.

The cause is structural, not statistical: **a Morton cell bounds OCCUPANCY, not
EXTENT**. In a sparse halo the coarsest cell that holds ``leaf_size`` particles is
geometrically vast, so nothing is far from it and it fails the MAC against every node
in the other domain. Cell leaves fixed exactly this for the LOCAL near field
(0.1225 N -> 0.0052 N); across a domain boundary it returns.

This probe measures the threshold before a rule is written, on a Plummer sphere AND
on the disc+bulge IC, because a disc's sparse regions are shaped differently from a
sphere's and the whole import hinges on a handful of cells.

What it reports, per (IC, N, ndev, partitioner):

* the per-domain leaf RADIUS distribution (median / p90 / p99 / p99.9 / max), which
  is what any rule has to be written against;
* the import as a function of a reclassification quantile ``q``: leaves whose radius
  exceeds their own domain's ``q``-quantile are served by a MULTIPOLE instead of by
  imported particles -- on the target side (an oversized local leaf takes expansions
  rather than importing) and on the source side (an oversized remote leaf is shipped
  as one multipole rather than as its particles).

``worst1src`` is the gate statistic: the largest single ordered (receiver, sender)
import, as a fraction of a domain's particle count. The MEAN over devices is not a
summary of it -- at a plane split of a Plummer sphere at N = 2x10^5 the two devices
differ threefold, one importing its whole neighbour and the other a third of it, and
`multigpu_import_locality_probe.py`'s "100 %" is the FIRST of those two ordered pairs
while a mean over both reads 66 %. Report the worst pair, or the design is flattered
by whichever device happened to be easy.

Two rule FORMS are swept, because they are not equally implementable. A radius
quantile needs a per-device order statistic over every leaf, and it is the wrong shape
anyway: at ndev = 4 on a Plummer sphere one device has max/median leaf radius 1838 and
another 4.4, so a per-device quantile forces the healthy device to cut good leaves
while under-cutting the sick one. A **Morton leaf DEPTH** threshold needs nothing: the
frame is global, so depth ``d`` is exactly cell size ``box / 2^d``, the same absolute
geometric test on every device with no reduction and no communication. Depth is also
the more honest statement of the cause -- a cell-partition leaf's bounding radius is at
most ``sqrt(3)/2`` of its own cell, so a leaf is vast only when its CELL is shallow.
Measured on a plane-split Plummer domain, Spearman(-depth, radius) = +0.73 while
Spearman(radius, import size) is +0.05.

The quantile, not a radius multiple, is the sweep variable on purpose: it makes the
gate's two axes -- "import <= 40 % of a neighbour domain" and "no more than ~0.1 % of
cells reclassified" -- directly comparable on one row. The absolute cut radius and
its ratio to the median are reported alongside so a rule can be written in whichever
form transfers between the two ICs.

THREE THINGS THIS PROBE DOES NOT SAY, stated because each would flatter the design:

1. Reclassification is modelled POST HOC on the walk's own neighbour lists. That is
   exact for "this pair is served by an expansion instead of by particles" -- the pair
   still exists, only who serves it changes -- and it is NOT a model of SPLITTING the
   oversized cell, which would change the walk. Splitting has to be measured by
   rebuilding the trees, which this probe does not do.
2. It measures VOLUME only. Serving a vast target leaf by a local expansion is an
   accuracy change (the expansion is not valid over the leaf's extent), and this
   probe cannot see it. Splitting the cell has no accuracy cost at all, only leaves.
   That asymmetry is the reason to measure both.
3. A truncated neighbour list understates the import, so no capacity here is ever
   accepted as a truncation -- both the neighbour cap and the pair queue are GROWN
   until the walk reports no overflow. See ``KN_START`` for why a fixed cap is not
   merely imprecise but wrong: near overflow ends the walk for every row, not just
   the row that overflowed.

The control is the identity row: at ``q = 1.0`` nothing is reclassified and the
import must reproduce ``multigpu_import_volume_probe.py`` exactly. If it does not,
the reclassification bookkeeping is broken and no row below it means anything.

Two diagnostics sit above the sweep, because the sweep alone cannot tell a rule that
is wrong from a premise that is wrong:

* the **offenders table** -- the target leaves with the largest import, with their
  radius and where that radius sits in their own domain's distribution. The
  oversized-cell story predicts the biggest importer is also among the biggest
  radii. If it is not, radius is the wrong discriminant however the sweep reads.
* the **oracle** -- cut the top-k importers directly, by import size, which is the
  best ANY target-side rule can do. If the oracle at k = a few does not collapse the
  import either, then no handful of cells explains it and the import is the union of
  many small rows, which is a different problem with a different fix.

Run (CPU is fine and is what this was measured on):

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \\
    PROBE_IC=plummer PROBE_N=200000 PROBE_NDEVS=2,4 python bench/multigpu_oversized_cell_probe.py
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
from yggdrax.bounds import infer_bounds  # noqa: E402
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl  # noqa: E402

N = int(os.environ.get("PROBE_N", "200000"))
LEAF = int(os.environ.get("PROBE_LEAF", "64"))
THETA = float(os.environ.get("PROBE_THETA", "0.8"))
IC = os.environ.get("PROBE_IC", "plummer")
NDEVS = [int(x) for x in os.environ.get("PROBE_NDEVS", "2,4").split(",")]
PARTS = os.environ.get("PROBE_PARTS", "morton").split(",")
QUANTILES = [
    float(x)
    for x in os.environ.get(
        "PROBE_QUANTILES", "1.0,0.99999,0.9999,0.999,0.998,0.995,0.99,0.98,0.95"
    ).split(",")
]
# Where the per-target-leaf neighbour cap STARTS. It is grown until the walk reports
# no overflow, never accepted as a truncation: ``near_overflow`` sits in the walk's
# own ``cond_fun``, so an overflowing row does not truncate ITSELF -- it HALTS the
# whole walk and every other row loses its remaining rounds. Measured, because a
# fixed cap was nearly shipped here as a way to reach ndev = 8: capping at 256 on the
# plane split at N = 2x10^5 read one device's import as 0.010 of its neighbour
# against a true 0.304, i.e. 30x too SMALL, while the OTHER device's number and the
# whole radius sweep were unaffected and looked perfectly sane. A starting size is a
# cost knob only: it saves the many benign pairs from paying the pathological one's
# buffer.
KN_START = int(os.environ.get("PROBE_KN_START", "1024"))

DISC_BULGE_NPZ = "/export/scratch/tbuck/odisseo_ic/disk_bulge_21m_v2.npz"


def load_ic(name, n, seed=0):
    """Positions and masses for ``name`` at ``n`` particles.

    ``disc_bulge`` subsamples the cached 21M disc+bulge IC (memory
    ``disc-bulge-rollout-and-cap-cliff``). Subsampling is uniform, so it keeps the
    SHAPE of the density field and changes its sampling density -- which is the right
    thing here, because the object under study is the cell that holds ``leaf_size``
    particles in a sparse region, and how vast that cell is depends on N. Each N is
    therefore its own data point, not a proxy for another.
    """
    if name == "disc_bulge":
        # the rollout IC stores one (N, 2, 3) phase-space array; [:, 0] is position
        with np.load(DISC_BULGE_NPZ) as f:
            pos_all = np.asarray(f["state0"][:, 0, :], np.float64)
            mass_all = np.asarray(f["mass"], np.float64)
        if n < pos_all.shape[0]:
            rng = np.random.default_rng(seed)
            idx = rng.choice(pos_all.shape[0], size=n, replace=False)
            frac = pos_all.shape[0] / n
            pos_all, mass_all = pos_all[idx], mass_all[idx] * frac
        return pos_all.astype(np.float32), mass_all.astype(np.float32)
    from common.ic import IC_GENERATORS

    p, m = IC_GENERATORS[name](n, seed=seed)
    return np.asarray(p, np.float32), np.asarray(m, np.float32)


def leaf_view(tree, geom, n_local):
    """Per-LEAF radius, occupancy and Morton depth for a domain tree, live leaves only.

    A padding leaf carries ``start == end == n_particles``, which the naive
    ``end - start + 1`` reads as ONE particle rather than as empty -- hence the
    explicit ``start < n_local`` test as well.
    """
    nint = int(tree.left_child.shape[0])
    nr = np.asarray(tree.node_ranges)
    live_all = (nr[:, 1] >= nr[:, 0]) & (nr[:, 0] < n_local)
    occ_all = np.where(live_all, nr[:, 1] - nr[:, 0] + 1, 0)
    radius_all = np.asarray(geom.radius, np.float64)
    # leaf_depths is indexed by leaf SLOT, i.e. node id - num_internal
    depth_leaf = np.asarray(tree.leaf_depths, np.int64)
    return nint, live_all, occ_all, radius_all, depth_leaf


def cross_lists(t_dst, g_dst, t_src, g_src, n_leaves_src):
    """Per-target-leaf cross-neighbour rows against one source domain, EXACT.

    Both capacities are grown until the walk reports no overflow. Neither may be
    accepted as a truncation: a truncated neighbour list makes the import look
    SMALLER than it is, which is the direction that would flatter the design, and it
    is not even a per-row truncation -- see the note on ``KN_START``.

    Returns
    -------
    tuple
        ``(leaf_node_ids, rows)`` -- the source leaf node ids per target leaf row.
    """
    full = max(64, int(n_leaves_src))
    kn = min(max(64, KN_START), full)
    queue = 1 << 16
    while True:
        res = dual_tree_walk_cross_impl(
            t_dst,
            g_dst,
            t_src,
            g_src,
            float(THETA),
            mac_type="dehnen",
            # the far buffer is allocated even when it is not collected, so it is
            # sized to nothing here rather than to the source leaf count
            max_interactions_per_node=1,
            max_neighbors_per_leaf=kn,
            max_pair_queue=queue,
            collect_far=False,
            collect_near=True,
        )
        if bool(res.queue_overflow):
            queue *= 4
            if queue > (1 << 24):
                raise RuntimeError("pair queue did not fit at 2^24")
            continue
        if bool(res.near_overflow):
            if kn >= full:
                raise RuntimeError(
                    f"near overflow at the FULL source leaf count {full}"
                )
            kn = min(kn * 4, full)
            continue
        break
    off = np.asarray(res.neighbor_offsets)
    idx = np.asarray(res.neighbor_indices)
    cnt = np.asarray(res.neighbor_counts)
    leaf_nodes = np.asarray(res.leaf_indices)
    rows = []
    for r in range(leaf_nodes.shape[0]):
        o, c = int(off[r]), int(cnt[r])
        rows.append(idx[o : o + c] if c else np.empty(0, np.int64))
    return leaf_nodes, rows


def q_stats(x, qs=(0.5, 0.9, 0.99, 0.999)):
    if x.size == 0:
        return [0.0] * (len(qs) + 1)
    return [float(np.quantile(x, q)) for q in qs] + [float(x.max())]


def main():
    pos, mass = load_ic(IC, N)
    bounds = infer_bounds(jnp.asarray(pos))
    print(f"IC={IC} N={N} leaf={LEAF} theta={THETA}  (quantiles are PER DOMAIN)")

    for part in PARTS:
        for ndev in NDEVS:
            if part == "plane":
                dom = plane_domains(pos, ndev)
            elif part == "rcb":
                dom = rcb_domains(pos, ndev)
            else:
                dom = morton_domains(pos, bounds, ndev)

            trees, geoms, nloc, views = [], [], [], []
            for d in range(ndev):
                m = dom == d
                nl = int(m.sum())
                t, g = build_domain_tree(pos[m], mass[m], bounds, LEAF)
                trees.append(t)
                geoms.append(g)
                nloc.append(nl)
                views.append(leaf_view(t, g, nl))

            # ---- the radius distribution any rule has to be written against ----
            print(f"\n== {part} ndev={ndev} N/dev={N // ndev}")
            print(
                f"{'dev':>4} {'leaves':>8} {'r_med':>10} {'r_p90':>10} {'r_p99':>10} "
                f"{'r_p99.9':>10} {'r_max':>12} {'max/med':>9}"
            )
            leaf_r, leaf_occ, leaf_pos, leaf_d = [], [], [], []
            for d in range(ndev):
                nint, live_all, occ_all, radius_all, depth_leaf = views[d]
                live_leaf = live_all[nint:]
                lr = radius_all[nint:][live_leaf]
                lo = occ_all[nint:][live_leaf]
                leaf_r.append(lr)
                leaf_occ.append(lo)
                leaf_d.append(depth_leaf[: live_leaf.size][live_leaf])
                # map leaf NODE id -> row in the live-leaf arrays
                node_ids = np.flatnonzero(live_all)
                node_ids = node_ids[node_ids >= nint]
                leaf_pos.append({int(v): i for i, v in enumerate(node_ids)})
                med, p90, p99, p999, mx = q_stats(lr)
                print(
                    f"{d:>4} {lr.size:>8} {med:>10.4g} {p90:>10.4g} {p99:>10.4g} "
                    f"{p999:>10.4g} {mx:>12.4g} {mx / max(med, 1e-30):>9.1f}"
                )

            # ---- the cross relation, walked once; every policy applied post hoc ----
            # R[(dst, src)][ti] = source live-leaf rows for target live-leaf ``ti``
            R = {}
            for dst in range(ndev):
                for src in range(ndev):
                    if src == dst:
                        continue
                    leaf_nodes, rws = cross_lists(
                        trees[dst],
                        geoms[dst],
                        trees[src],
                        geoms[src],
                        leaf_r[src].size,
                    )
                    rr = [np.empty(0, np.int64)] * leaf_r[dst].size
                    for node, r in zip(leaf_nodes.tolist(), rws):
                        ti = leaf_pos[dst].get(int(node))
                        if ti is None:  # a padding row: nothing real behind it
                            continue
                        si = np.array(
                            [leaf_pos[src].get(int(v), -1) for v in r], np.int64
                        )
                        rr[ti] = si[si >= 0]
                    R[(dst, src)] = rr

            def one_import(dst, src, drop_t, drop_s):
                """Particles ``dst`` imports from ``src`` under a policy.

                ``drop_t`` / ``drop_s`` are boolean masks over live leaves: a dropped
                target leaf is served by expansions and imports nothing, a dropped
                source leaf is shipped as one multipole.

                Returns
                -------
                int
                    Imported particles.
                """
                used = [
                    si[~drop_s[si]]
                    for ti, si in enumerate(R[(dst, src)])
                    if si.size and not drop_t[ti]
                ]
                if not used:
                    return 0
                u = np.unique(np.concatenate(used))
                return int(leaf_occ[src][u].sum()) if u.size else 0

            def policy(drops):
                """Per-device totals and the worst single ordered pair, under ``drops``.

                Returns
                -------
                tuple
                    ``(mean_total, max_total, worst_pair)`` in particles.
                """
                totals, worst = [], 0
                for dst in range(ndev):
                    tot = 0
                    for src in range(ndev):
                        if src == dst:
                            continue
                        got = one_import(dst, src, drops[dst], drops[src])
                        tot += got
                        worst = max(worst, got)
                    totals.append(tot)
                return float(np.mean(totals)), float(max(totals)), worst

            own = float(np.mean(nloc))
            none_drop = [np.zeros(leaf_r[d].size, bool) for d in range(ndev)]

            # ---- is radius even the discriminant? ----
            # row size per target leaf, summed over source domains
            tot_row = []
            for d in range(ndev):
                t = np.zeros(leaf_r[d].size, np.float64)
                for src in range(ndev):
                    if src == d:
                        continue
                    t += np.array([si.size for si in R[(d, src)]], np.float64)
                tot_row.append(t)

            d0 = int(np.argmax([t.max(initial=0) for t in tot_row]))
            rr = leaf_r[d0]
            med_r = max(float(np.median(rr)), 1e-30)
            print(f"  offenders on dev {d0} (its worst importer is the worst overall):")
            print(
                f"  {'row':>9} {'radius':>12} {'r/med':>9} {'r pctile':>9} {'occ':>5}"
            )
            for i in np.argsort(-tot_row[d0])[:5].tolist():
                pct = 100.0 * float((rr <= rr[i]).mean())
                print(
                    f"  {tot_row[d0][i]:>9.0f} {rr[i]:>12.5g} {rr[i] / med_r:>9.1f} "
                    f"{pct:>9.4f} {leaf_occ[d0][i]:>5}"
                )
            if rr.size > 2:
                rank_r = np.argsort(np.argsort(rr))
                rank_t = np.argsort(np.argsort(tot_row[d0]))
                print(
                    f"  spearman(radius, row size) = "
                    f"{float(np.corrcoef(rank_r, rank_t)[0, 1]):+.3f}"
                )

            # ---- the oracle: cut the top-k IMPORTERS, the best any target rule can do
            n_leaves_avg = float(np.mean([leaf_r[d].size for d in range(ndev)]))
            print("  oracle (cut by import size -- an upper bound on any target rule)")
            print(
                f"  {'k':>6} {'%cells':>8} {'impMean':>8} {'impMax':>7} {'worst1src':>10}"
            )
            for k in (0, 1, 2, 5, 10, 50, 200):
                if k > n_leaves_avg:
                    break
                drops = []
                for d in range(ndev):
                    m = np.zeros(leaf_r[d].size, bool)
                    m[np.argsort(-tot_row[d])[:k]] = True
                    drops.append(m)
                mn, mx, w = policy(drops)
                print(
                    f"  {k:>6} {100.0 * k / n_leaves_avg:>8.4f} {mn / own:>8.3f} "
                    f"{mx / own:>7.3f} {w / own:>10.3f}"
                )
            base_mean = policy(none_drop)[0]

            # ---- the rule under test: cut by leaf RADIUS quantile ----
            print(
                f"\n{'q':>9} {'r_cut/med':>10} {'%cellsCut':>10} {'impMean':>8} "
                f"{'impMax':>7} {'worst1src':>10} {'rowMed':>7} {'rowP99':>8} {'rowMax':>8}"
            )
            for q in QUANTILES:
                if q >= 1.0:
                    cuts = [np.inf] * ndev
                else:
                    cuts = [float(np.quantile(leaf_r[d], q)) for d in range(ndev)]
                drops = [leaf_r[d] > cuts[d] for d in range(ndev)]
                pct_cut = (
                    100.0 * sum(m.sum() for m in drops) / sum(m.size for m in drops)
                )
                mn, mx, w = policy(drops)
                if q >= 1.0 and abs(mn - base_mean) > 0.5:
                    raise RuntimeError(
                        f"CONTROL FAILED: the q=1.0 row imports {mn:.0f} but the k=0 "
                        f"oracle imports {base_mean:.0f}; the two reclassification "
                        "paths disagree, so no row below means anything"
                    )
                kept = []
                for dst in range(ndev):
                    for s_ in range(ndev):
                        if s_ == dst:
                            continue
                        for ti, si in enumerate(R[(dst, s_)]):
                            kept.append(
                                0 if drops[dst][ti] else int((~drops[s_][si]).sum())
                            )
                ar = np.asarray(kept) if kept else np.zeros(1, np.int64)
                r_ratio = float(
                    np.mean(
                        [
                            (
                                cuts[d] / max(float(np.median(leaf_r[d])), 1e-30)
                                if np.isfinite(cuts[d])
                                else np.inf
                            )
                            for d in range(ndev)
                        ]
                    )
                )
                print(
                    f"{q:>9.5f} {r_ratio:>10.4g} {pct_cut:>10.4f} {mn / own:>8.3f} "
                    f"{mx / own:>7.3f} {w / own:>10.3f} "
                    f"{np.median(ar):>7.0f} {np.quantile(ar, 0.99):>8.0f} {ar.max():>8.0f}"
                )

            # ---- the same policy cut by Morton leaf DEPTH: global and local ----
            box = float(np.max(np.asarray(bounds[1]) - np.asarray(bounds[0])))
            dlo = int(min(d.min() for d in leaf_d))
            dhi = int(max(d.max() for d in leaf_d))
            print(
                f"\n{'depth<=':>9} {'cellsize':>10} {'%cellsCut':>10} {'impMean':>8} "
                f"{'impMax':>7} {'worst1src':>10}"
            )
            for dcut in range(dlo - 1, min(dhi, dlo + 9) + 1):
                drops = [leaf_d[d] <= dcut for d in range(ndev)]
                pct_cut = (
                    100.0 * sum(m.sum() for m in drops) / sum(m.size for m in drops)
                )
                mn, mx, w = policy(drops)
                print(
                    f"{dcut:>9} {box / 2.0 ** max(dcut, 0):>10.4g} {pct_cut:>10.4f} "
                    f"{mn / own:>8.3f} {mx / own:>7.3f} {w / own:>10.3f}"
                )


if __name__ == "__main__":
    main()
