"""Is the import a boundary shell, or the whole domain?

Plane split at ndev=2. For every imported source leaf, how far is it from the
cut plane, and how big is it? A compact boundary means imports hug the plane. If
they are spread through the domain, the near relation is reaching far -- and the
reason will be leaf RADIUS, because a Morton cell bounds OCCUPANCY, not extent:
in a sparse halo the coarsest cell holding <=64 particles is geometrically huge.
"""

import os
import sys

sys.path.insert(0, "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu")
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from common.ic import IC_GENERATORS
from yggdrax._cell_partition import adaptive_cell_leaf_partition_numpy
from yggdrax._tree_impl import build_static_cells_tree
from yggdrax.bounds import infer_bounds
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl
from yggdrax.geometry import compute_tree_geometry
from yggdrax.morton import morton_encode

N, LEAF, THETA = int(os.environ.get("PROBE_N", "200000")), 64, 0.8
ic = IC_GENERATORS["plummer"](N, seed=0)
pos = np.asarray(ic[0], np.float32)
mass = np.asarray(ic[1], np.float32)
B = infer_bounds(jnp.asarray(pos))
ax = int(np.argmax(pos.max(0) - pos.min(0)))
order = np.argsort(pos[:, ax], kind="stable")
halves = np.array_split(order, 2)
cut = 0.5 * (pos[halves[0][-1], ax] + pos[halves[1][0], ax])
print(f"plane split on axis {ax} at {cut:.4f}")


def build(sel):
    codes = np.sort(np.asarray(morton_encode(jnp.asarray(pos[sel]), B)))
    k = int(adaptive_cell_leaf_partition_numpy(codes, leaf_size=LEAF)[0].size)
    t, ps, _, _ = build_static_cells_tree(
        jnp.asarray(pos[sel]),
        jnp.asarray(mass[sel]),
        B,
        leaf_size=LEAF,
        leaf_capacity=k,
        return_reordered=True,
    )
    g = compute_tree_geometry(t, jnp.asarray(ps), max_leaf_size=LEAF)
    return t, g


m0 = np.zeros(N, bool)
m0[halves[0]] = True
m1 = ~m0
t0, g0 = build(m0)
t1, g1 = build(m1)
nint1 = int(t1.left_child.shape[0])
n1 = int(m1.sum())
nr1 = np.asarray(t1.node_ranges)
occ1 = np.where(
    (nr1[:, 1] >= nr1[:, 0]) & (nr1[:, 0] < n1), nr1[:, 1] - nr1[:, 0] + 1, 0
)
live1 = np.flatnonzero(occ1 > 0)
live1 = live1[live1 >= nint1]
cen1 = np.asarray(g1.center)
rad1 = np.asarray(g1.radius)

nl1 = int((occ1[nint1:] > 0).sum())
res = dual_tree_walk_cross_impl(
    t0,
    g0,
    t1,
    g1,
    float(THETA),
    mac_type="dehnen",
    max_interactions_per_node=max(64, nl1),
    max_neighbors_per_leaf=max(64, nl1),
    max_pair_queue=1 << 22,
    collect_far=True,
    collect_near=True,
)
assert not (bool(res.near_overflow) or bool(res.queue_overflow)), "overflow"
nbr = np.asarray(res.neighbor_indices)
cnt = np.asarray(res.neighbor_counts)
rows = (
    [nbr[i, : cnt[i]] for i in range(nbr.shape[0])]
    if nbr.ndim == 2
    else [nbr[nbr >= 0]]
)
used = np.unique(np.concatenate(rows)) if rows else np.empty(0, int)
used = used[(used >= 0)]
used = used[occ1[used] > 0]

d_all = np.abs(cen1[live1][:, ax] - cut)
d_used = np.abs(cen1[used][:, ax] - cut)
print(
    f"source live leaves={live1.size} imported={used.size} ({used.size/live1.size:.1%})"
)
print(
    f"  distance from the cut plane   all: p50={np.median(d_all):8.3f} p90={np.percentile(d_all,90):8.3f} max={d_all.max():8.3f}"
)
print(
    f"                            imported: p50={np.median(d_used):8.3f} p90={np.percentile(d_used,90):8.3f} max={d_used.max():8.3f}"
)
print(
    f"  leaf RADIUS                   all: p50={np.median(rad1[live1]):8.3f} p90={np.percentile(rad1[live1],90):8.3f} max={rad1[live1].max():8.3f}"
)
print(
    f"                            imported: p50={np.median(rad1[used]):8.3f} p90={np.percentile(rad1[used],90):8.3f} max={rad1[used].max():8.3f}"
)
# how many imported leaves are further from the plane than their own radius?
far = d_used > rad1[used]
print(
    f"  imported leaves further from the plane than their own radius: {far.sum()} / {used.size} ({far.mean():.1%})"
)
print(
    f"  imported particles: {int(occ1[used].sum())} of {int(occ1[live1].sum())} ({occ1[used].sum()/occ1[live1].sum():.1%})"
)

print("\n--- is the MAC ever accepting in the cross walk? ---")
print(f"  far_pair_count  = {int(res.far_pair_count)}")
print(f"  near_pair_count = {int(res.near_pair_count)}")
print(f"  target leaves={int(res.leaf_indices.shape[0])}  source live leaves={nl1}")
print(f"  near rows: mean={cnt.mean():.1f} max={cnt.max()} (a full row would be {nl1})")
# control: the MUTUAL walk on ONE tree over the same particles, same theta
from yggdrax.interactions import dual_tree_walk_mutual

nr_all = np.asarray(t1.node_ranges)
tot = nr_all.shape[0]


def pad(c):
    c = jnp.asarray(c)
    return jnp.concatenate([c, jnp.full((tot - c.shape[0],), -1, c.dtype)])


mres = dual_tree_walk_mutual(
    pad(t1.left_child),
    pad(t1.right_child),
    jnp.asarray(g1.center),
    jnp.asarray(g1.radius),
    float(THETA),
    jnp.asarray(int(np.argmin(np.asarray(t1.parent)))),
    max_pair_queue=1 << 21,
    far_cap=1 << 22,
    near_cap=1 << 21,
    mac_type="dehnen",
    node_active=jnp.asarray(nr_all[:, 1] >= nr_all[:, 0]),
)
print(
    f"  MUTUAL walk on the SAME tree: far={int(mres.far_count)} near={int(mres.near_count)}"
    f"  -> near/leaf = {int(mres.near_count)/max(nl1,1):.1f}"
)

print("\n--- how concentrated is the import in a few pathological leaves? ---")
o = np.argsort(cnt)[::-1]
print(
    f"  target-leaf cross-neighbour counts: p50={np.median(cnt):.0f} p90={np.percentile(cnt,90):.0f} "
    f"p99={np.percentile(cnt,99):.0f} max={cnt.max()}"
)
print(f"  top 10 rows: {cnt[o[:10]]}")
tgt_rad = np.asarray(g0.radius)
tl = np.asarray(res.leaf_indices)
print(f"  radius of the top-10 target leaves: {np.round(tgt_rad[tl[o[:10]]],2)}")
print(f"  median target-leaf radius: {np.median(tgt_rad[tl]):.3f}")
off = np.asarray(res.neighbor_offsets)  # flat CSR, not a dense [leaves, K] table


def row(i):
    return nbr[off[i] : off[i] + cnt[i]]


for drop in (0, 1, 2, 3, 5, 10, 50):
    keep = o[drop:]
    rows2 = [row(i) for i in keep if cnt[i] > 0]
    u = np.unique(np.concatenate(rows2)) if rows2 else np.empty(0, int)
    u = u[(u >= 0)]
    u = u[occ1[u] > 0]
    print(
        f"  drop top {drop:>3} target leaves -> union {u.size:>5} leaves "
        f"({u.size/live1.size:>6.1%})  particles {int(occ1[u].sum()):>7} "
        f"({occ1[u].sum()/occ1[live1].sum():>6.1%})"
    )
