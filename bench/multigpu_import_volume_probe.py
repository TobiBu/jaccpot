"""How much of a neighbour domain must a device actually import?

This supersedes the import half of ``multigpu_cross_volume_probe.py``, which
measured the wrong object: it built ONE global tree and merely LABELLED its
leaves by domain. The real lane gives each device its own tree over its own
particles, and the two differ in a way that matters --

* a global tree's leaves are cut against every particle, so a leaf spanning the
  boundary region relates to foreign leaves that a per-domain tree would never
  have formed the same way;
* leaves can straddle a domain cut, which is why RCB could not be measured at
  all there (3133-9382 straddling leaves, miscounted as cross-domain, inflating
  RCB's cross fraction 0.014 -> 0.168).

Per-domain trees remove both problems: leaves cannot straddle by construction,
so RCB needs no special handling, and the cross relation is the real
``dual_tree_walk_cross_impl`` the distributed lane itself uses.

**The control is a PLANE split.** It is the most compact boundary there is, so
its halo must be a thin slab -- a few per cent. If the measurement reports
"import nearly everything" for a plane, the measurement is broken and no
conclusion about Morton or RCB may be drawn from it.

Reports, per receiving device: the distinct foreign leaves and particles it must
import, against its own local count.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from common.ic import IC_GENERATORS  # noqa: E402
from yggdrax._cell_partition import adaptive_cell_leaf_partition_numpy  # noqa: E402
from yggdrax._tree_impl import build_static_cells_tree  # noqa: E402
from yggdrax.bounds import infer_bounds  # noqa: E402
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl  # noqa: E402
from yggdrax.geometry import compute_tree_geometry  # noqa: E402
from yggdrax.morton import morton_encode  # noqa: E402

N = int(os.environ.get("PROBE_N", "200000"))
LEAF = int(os.environ.get("PROBE_LEAF", "64"))
THETA = float(os.environ.get("PROBE_THETA", "0.8"))
IC = os.environ.get("PROBE_IC", "plummer")


def plane_domains(points, ndev):
    """Split along the longest axis into ndev equal-count slabs: a compact control."""
    ax = int(np.argmax(points.max(0) - points.min(0)))
    order = np.argsort(points[:, ax], kind="stable")
    dom = np.empty(len(points), np.int32)
    for d, chunk in enumerate(np.array_split(order, ndev)):
        dom[chunk] = d
    return dom


def rcb_domains(points, ndev):
    """Recursive coordinate bisection; needs a power-of-two device count."""
    if ndev & (ndev - 1):
        raise ValueError(f"rcb wants a power of two, got {ndev}")
    dom = np.zeros(len(points), np.int32)
    groups = [np.arange(len(points))]
    while len(groups) < ndev:
        nxt = []
        for g in groups:
            ax = int(np.argmax(points[g].max(0) - points[g].min(0)))
            o = g[np.argsort(points[g][:, ax], kind="stable")]
            mid = len(o) // 2
            nxt.append(o[:mid])
            nxt.append(o[mid:])
        groups = nxt
    for d, g in enumerate(groups):
        dom[g] = d
    return dom


def morton_domains(points, bounds, ndev):
    """Contiguous Morton ranges, the partition the aligned-SFC plan assumes."""
    codes = np.asarray(morton_encode(jnp.asarray(points), bounds))
    order = np.argsort(codes, kind="stable")
    dom = np.empty(len(points), np.int32)
    for d, chunk in enumerate(np.array_split(order, ndev)):
        dom[chunk] = d
    return dom


def build_domain_tree(points, masses, bounds, leaf_size):
    """One device's own tree over its own particles, in the GLOBAL Morton frame.

    The leaf capacity is the EXACT cell count, not a padded power of two. The
    production lane pads for static shapes and masks the padding out of the walk
    with ``node_active``; ``dual_tree_walk_cross_impl`` takes no such mask, so a
    padded tree here would put radius-0 padding leaves into the walk, where they
    fail the MAC against everything and become neighbours of every leaf -- the
    documented 30M-spurious-edge failure. A probe does not need static shapes, so
    the honest fix is to build with no padding at all.
    """
    codes = np.sort(np.asarray(morton_encode(jnp.asarray(points), bounds)))
    k = int(adaptive_cell_leaf_partition_numpy(codes, leaf_size=leaf_size)[0].size)
    cap = int(k)
    tree, psorted, _msorted, _inv = build_static_cells_tree(
        jnp.asarray(points),
        jnp.asarray(masses),
        bounds,
        leaf_size=leaf_size,
        leaf_capacity=cap,
        return_reordered=True,
    )
    geom = compute_tree_geometry(tree, jnp.asarray(psorted), max_leaf_size=leaf_size)
    return tree, geom


def main():
    ic = IC_GENERATORS[IC](N, seed=0)
    pos = np.asarray(ic[0], np.float32)
    mass = np.asarray(ic[1], np.float32)
    bounds = infer_bounds(jnp.asarray(pos))

    print(f"IC={IC} N={N} leaf={LEAF} theta={THETA}")
    print(
        f"{'part':>8} {'ndev':>5} {'N/dev':>9} {'locLeaf':>8} {'impLeaf':>8} "
        f"{'impPart':>9} {'halo/local':>10} {'perSrc':>7}"
    )

    for part in os.environ.get("PROBE_PARTS", "plane,rcb,morton").split(","):
        for ndev in (int(x) for x in os.environ.get("PROBE_NDEVS", "2,4").split(",")):
            if part == "plane":
                dom = plane_domains(pos, ndev)
            elif part == "rcb":
                dom = rcb_domains(pos, ndev)
            else:
                dom = morton_domains(pos, bounds, ndev)

            trees, geoms, sel = [], [], []
            for d in range(ndev):
                m = dom == d
                sel.append(m)
                t, g = build_domain_tree(pos[m], mass[m], bounds, LEAF)
                trees.append(t)
                geoms.append(g)

            def occ_of(tree, n_local):
                # A padding leaf carries start == end == n_particles, which the
                # naive end-start+1 reads as ONE particle rather than as empty.
                nr = np.asarray(tree.node_ranges)
                live = (nr[:, 1] >= nr[:, 0]) & (nr[:, 0] < n_local)
                return np.where(live, nr[:, 1] - nr[:, 0] + 1, 0)

            imp_leaves, imp_parts, loc_leaves = [], [], []
            for dst in range(ndev):
                nint_d = int(trees[dst].left_child.shape[0])
                occ_dst = occ_of(trees[dst], int(sel[dst].sum()))
                loc_leaves.append(int((occ_dst[nint_d:] > 0).sum()))
                need_l, need_p = 0, 0
                for src in range(ndev):
                    if src == dst:
                        continue
                    nint_s = int(trees[src].left_child.shape[0])
                    occ_src = occ_of(trees[src], int(sel[src].sum()))
                    n_leaves_s = int((occ_src[nint_s:] > 0).sum())
                    # The worst leaf's near list is COMPLETE at every size measured
                    # -- a sparse outer leaf gets a sphere wide enough to fail the
                    # MAC against everything -- so no constant sizes this; the only
                    # safe cap is the full source leaf count. A truncated list makes
                    # the import look SMALLER than it is, which is exactly the
                    # direction that would flatter the design.
                    full = max(64, n_leaves_s)
                    res = dual_tree_walk_cross_impl(
                        trees[dst], geoms[dst], trees[src], geoms[src], float(THETA),
                        mac_type="dehnen",
                        max_interactions_per_node=full,
                        max_neighbors_per_leaf=full,
                        max_pair_queue=1 << 22,
                        collect_far=True, collect_near=True,
                    )
                    if bool(res.near_overflow) or bool(res.queue_overflow) or bool(res.far_overflow):
                        raise RuntimeError(
                            f"{part} ndev={ndev} dst={dst} src={src}: overflow "
                            f"(near={bool(res.near_overflow)} far={bool(res.far_overflow)} "
                            f"queue={bool(res.queue_overflow)}) with cap={full}; "
                            "a truncated list would understate the import"
                        )
                    nbr = np.asarray(res.neighbor_indices)
                    cnt = np.asarray(res.neighbor_counts)
                    # dense [leaves, K] rows: take each row's live prefix
                    if nbr.ndim == 2:
                        rows = [nbr[i, : cnt[i]] for i in range(nbr.shape[0])]
                        used = np.unique(np.concatenate(rows)) if rows else np.empty(0, int)
                    else:
                        used = np.unique(nbr[nbr >= 0])
                    used = used[used >= 0]
                    used = used[occ_src[used] > 0]   # never count an empty leaf
                    need_l += int(used.size)
                    need_p += int(occ_src[used].sum()) if used.size else 0
                imp_leaves.append(need_l)
                imp_parts.append(need_p)

            per_dev = N / ndev
            halo = float(np.mean(imp_parts)) / per_dev
            per_src = float(np.mean(imp_leaves)) / max(ndev - 1, 1) / max(np.mean(loc_leaves), 1)
            print(
                f"{part:>8} {ndev:>5} {per_dev:>9.0f} {np.mean(loc_leaves):>8.0f} "
                f"{np.mean(imp_leaves):>8.0f} {np.mean(imp_parts):>9.0f} "
                f"{halo:>10.3f} {per_src:>7.3f}"
            )


if __name__ == "__main__":
    main()
