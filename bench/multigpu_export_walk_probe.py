"""Phase 3.5: can a coarse SUMMARY replace the every-leaf frontier, and at what cost?

The receiver does not have the sender's tree -- that is the whole difficulty of the
exchange, and it is why `build_coarse_frontier` all-gathers EVERY local leaf today
(382k remote leaves per device at 8 cards, which is the number that forces this phase
to change something).

The alternative this probe measures is a **sender-side export walk**. Each device
publishes a small level-k Morton summary of itself; each SENDER then walks its own
full tree against every receiver's summary cells and decides unilaterally what to
send. One ragged round, no request round. It is correct by construction because a
summary cell BOUNDS every real target inside it, so a MAC decision taken against the
cell is conservative: a node far from the cell is far from each of its targets. It can
never miss; it can only over-send.

So the whole question is the over-send factor, and that is what this reports:

    exported nodes (walk against the receiver's k-level summary)
    ------------------------------------------------------------
    needed nodes   (walk against the receiver's real tree)

against summary level k, i.e. against how much the summary costs to publish. Coarse
summaries are cheap to gather and over-send badly; fine ones are the frontier again.

**The control is k = the real leaves.** Feeding the receiver's own leaf set in as its
"summary" must reproduce the exact import to the node, because the two walks then
differ in nothing. If that row is not 1.000, the comparison is measuring the harness
and no other row means anything.

Both walks use the SAME per-node MAC extents (`_build_mac_extents` on the sender's
geometry), so the only difference between them is the target side.

The export walk is `dual_tree_walk_mutual` seeded with `(cell_i, sender_root)` over a
combined `[cells ; sender_tree]` index space -- the cells carry no children, so the
walk refines only on the sender's side and terminates there. That is the multi-pair
seed of Phase 4.1 used in the opposite direction from the evaluation.

Run:

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \\
    PROBE_IC=plummer PROBE_N=200000 PROBE_NDEVS=2 python bench/multigpu_export_walk_probe.py
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
)
from multigpu_oversized_cell_probe import load_ic  # noqa: E402
from yggdrax._interactions_impl import _build_mac_extents  # noqa: E402
from yggdrax.bounds import infer_bounds  # noqa: E402
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl  # noqa: E402
from yggdrax.interactions import dual_tree_walk_mutual  # noqa: E402
from yggdrax._cell_partition import MORTON_LEVELS  # noqa: E402
from yggdrax.morton import morton_encode  # noqa: E402

N = int(os.environ.get("PROBE_N", "200000"))
LEAF = int(os.environ.get("PROBE_LEAF", "64"))
THETA = float(os.environ.get("PROBE_THETA", "0.8"))
IC = os.environ.get("PROBE_IC", "plummer")
NDEVS = [int(x) for x in os.environ.get("PROBE_NDEVS", "2,4").split(",")]
PART = os.environ.get("PROBE_PART", "morton")
LEVELS = [int(x) for x in os.environ.get("PROBE_LEVELS", "2,3,4,5,6,8").split(",")]
MAC = "dehnen"


def summary_cells(pos_sorted, codes_sorted, level):
    """Bounding spheres of the level-``level`` Morton cells of one domain.

    A cell's sphere covers every particle in it, so it covers every tree node the
    receiver could build inside it -- which is what makes the export conservative.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(centres (K, 3), radii (K,))``.
    """
    shift = 3 * (MORTON_LEVELS - int(level))
    cell = (
        np.asarray(codes_sorted) >> np.uint64(shift)
        if shift > 0
        else np.asarray(codes_sorted)
    )
    _uniq, inv = np.unique(cell, return_inverse=True)
    k = int(inv.max()) + 1
    p = np.asarray(pos_sorted, np.float64)
    lo = np.full((k, 3), np.inf)
    hi = np.full((k, 3), -np.inf)
    np.minimum.at(lo, inv, p)
    np.maximum.at(hi, inv, p)
    cen = 0.5 * (lo + hi)
    rad = np.zeros(k)
    d = np.linalg.norm(p - cen[inv], axis=1)
    np.maximum.at(rad, inv, d)
    return cen, rad


def sender_extents(tree, geom):
    """The same per-node MAC extents the cross walk builds, so both walks agree."""
    ext, _ = _build_mac_extents(
        tree.parent, geom, int(tree.left_child.shape[0]), MAC, 1.0
    )
    return np.asarray(ext)


def export_walk(cen, rad, s_tree, s_geom, dtype):
    """Distinct sender nodes an export walk against ``(cen, rad)`` would ship.

    Combined index space ``[cells ; sender_tree]``; the cells carry no children, so
    ``split_a`` never fires and the walk refines only on the sender's side.
    """
    k = int(cen.shape[0])
    n_s = int(s_tree.parent.shape[0])
    n_si = int(s_tree.left_child.shape[0])
    idx = s_tree.parent.dtype
    left = jnp.concatenate(
        [
            jnp.full((k,), -1, idx),
            jnp.asarray(s_tree.left_child, idx),
            jnp.full((n_s - n_si,), -1, idx),
        ]
    )
    right = jnp.concatenate(
        [
            jnp.full((k,), -1, idx),
            jnp.asarray(s_tree.right_child, idx),
            jnp.full((n_s - n_si,), -1, idx),
        ]
    )
    # child ids in the sender's own space must be shifted into the combined one
    left = jnp.where(left >= 0, left + k, left).at[:k].set(-1)
    right = jnp.where(right >= 0, right + k, right).at[:k].set(-1)
    centers = jnp.concatenate(
        [jnp.asarray(cen, dtype), jnp.asarray(s_geom.center, dtype)]
    )
    radii = jnp.concatenate(
        [jnp.asarray(rad, dtype), jnp.asarray(sender_extents(s_tree, s_geom), dtype)]
    )
    s_root = int(np.argmin(np.asarray(s_tree.parent))) + k
    queue = 1 << 16
    while True:
        res = dual_tree_walk_mutual(
            left,
            right,
            centers,
            radii,
            float(THETA),
            jnp.asarray(s_root, idx),
            max_pair_queue=queue,
            far_cap=1 << 22,
            near_cap=1 << 22,
            mac_type=MAC,
            seed_a=jnp.asarray(np.arange(k), idx),
            seed_b=jnp.full((k,), s_root, idx),
        )
        if not bool(res.queue_overflow):
            break
        queue *= 4
        if queue > (1 << 24):
            raise RuntimeError("export walk queue did not fit at 2^24")
    if bool(res.far_overflow) or bool(res.near_overflow):
        raise RuntimeError("export walk list overflow")
    fb = np.asarray(res.far_b)[: int(res.far_count)]
    nb = np.asarray(res.near_b)[: int(res.near_count)]
    got = (
        np.unique(np.concatenate([fb, nb]))
        if (fb.size or nb.size)
        else np.empty(0, int)
    )
    return np.unique(got[got >= k] - k)


def exact_import(t_tree, t_geom, s_tree, s_geom):
    """Distinct sender nodes the receiver's REAL tree names -- what must be shipped."""
    kf, kn, queue = 256, 1024, 1 << 16
    full_node = int(s_tree.parent.shape[0])
    while True:
        res = dual_tree_walk_cross_impl(
            t_tree,
            t_geom,
            s_tree,
            s_geom,
            float(THETA),
            mac_type=MAC,
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
            kf = min(kf * 4, full_node)
            continue
        if bool(res.near_overflow):
            kn = min(kn * 4, full_node)
            continue
        break
    tt = np.asarray(res.interaction_targets)
    far = np.asarray(res.interaction_sources)[tt >= 0]
    nbr = np.asarray(res.neighbor_indices)
    near = nbr[nbr >= 0]
    both = np.concatenate([far[far >= 0], near])
    return np.unique(both) if both.size else np.empty(0, int)


def main():
    pos, mass = load_ic(IC, N)
    bounds = infer_bounds(jnp.asarray(pos))
    print(f"IC={IC} N={N} leaf={LEAF} theta={THETA} part={PART}")

    for ndev in NDEVS:
        dom = (
            plane_domains(pos, ndev)
            if PART == "plane"
            else morton_domains(pos, bounds, ndev)
        )
        trees, geoms, codes, psort = [], [], [], []
        for d in range(ndev):
            m = dom == d
            t, g = build_domain_tree(pos[m], mass[m], bounds, LEAF)
            trees.append(t)
            geoms.append(g)
            order = np.asarray(t.particle_indices)
            ps = np.asarray(pos[m])[order]
            psort.append(ps)
            codes.append(np.asarray(morton_encode(jnp.asarray(ps), bounds)))

        exact = {}
        for r in range(ndev):
            for s_ in range(ndev):
                if r == s_:
                    continue
                exact[(r, s_)] = exact_import(
                    trees[r], geoms[r], trees[s_], geoms[s_]
                ).size

        print(f"\n== ndev={ndev} N/dev={N // ndev}")
        print(
            f"{'level':>6} {'cells/dev':>10} {'summaryKB':>10} {'export/need':>12} "
            f"{'worstPair':>10}"
        )
        dtype = jnp.asarray(geoms[0].center).dtype
        # control row first: the receiver's own leaves as its "summary"
        for level in ["leaves"] + LEVELS:
            ratios, cells = [], []
            for r in range(ndev):
                if level == "leaves":
                    nint = int(trees[r].left_child.shape[0])
                    nr = np.asarray(trees[r].node_ranges)
                    live = (nr[:, 1] >= nr[:, 0]) & (nr[:, 0] < psort[r].shape[0])
                    rows = np.flatnonzero(live)
                    rows = rows[rows >= nint]
                    cen = np.asarray(geoms[r].center)[rows]
                    rad = sender_extents(trees[r], geoms[r])[rows]
                else:
                    cen, rad = summary_cells(psort[r], codes[r], level)
                cells.append(cen.shape[0])
                for s_ in range(ndev):
                    if r == s_:
                        continue
                    got = export_walk(cen, rad, trees[s_], geoms[s_], dtype).size
                    ratios.append(got / max(exact[(r, s_)], 1))
            kb = float(np.mean(cells)) * 4 * 4 / 1024.0
            tag = "leaves" if level == "leaves" else str(level)
            print(
                f"{tag:>6} {np.mean(cells):>10.0f} {kb:>10.1f} "
                f"{np.mean(ratios):>12.3f} {max(ratios):>10.3f}"
            )


if __name__ == "__main__":
    main()
