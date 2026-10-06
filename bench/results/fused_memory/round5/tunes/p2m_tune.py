"""Time the leaf P2M Pallas launch alone on a real cell tree, per (block, chunk, warps).

usage: python com_radii_tune.py --n 8000000 --cml 6 --variants 4:8:4 8:8:4 16:8:4 ...

Builds the static cells tree the fused lane builds (clipped Plummer, leaf 64), the
node centres of mass, the tree's level count, then jit-compiles ``_com_radii`` per
variant and times it. Every variant's radii are compared bitwise with the first.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from yggdrax._tree_impl import build_static_cells_tree

from jaccpot.pallas.p2m_real_leaf import p2m_real_leaves_pallas

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8_000_000)
ap.add_argument("--leaf", type=int, default=64)
ap.add_argument("--cml", type=int, default=6)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--variants", nargs="+", default=["0:0:0"])
ap.add_argument("--order", type=int, default=5)
args = ap.parse_args()

rng = np.random.default_rng(0)
rmax = 20.0
x_max = rmax**3 / (rmax**2 + 1.0) ** 1.5
x = rng.uniform(0.0, x_max, size=args.n)
r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
mu = rng.uniform(-1.0, 1.0, size=args.n)
phi = rng.uniform(0.0, 2.0 * np.pi, size=args.n)
st = np.sqrt(1.0 - mu * mu)
pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1).astype(
    np.float32
)
mass = np.full(args.n, 1.0 / args.n, np.float32)
lo, hi = pos.min(0) - 1e-3, pos.max(0) + 1e-3
bounds = (jnp.asarray(lo), jnp.asarray(hi))


def build(cap):
    return build_static_cells_tree(
        jnp.asarray(pos),
        jnp.asarray(mass),
        bounds,
        leaf_size=args.leaf,
        leaf_capacity=cap,
        return_reordered=True,
        return_overflow=True,
        min_level=args.cml,
    )


topo, ps, ms, _, ovf = build(args.n // 4 + 1024)
nr = np.asarray(topo.node_ranges)
ni = int(np.asarray(topo.left_child).shape[0])
live = int(np.sum(nr[ni:, 1] >= nr[ni:, 0]))
cap = int(np.ceil(1.15 * live / 1024) * 1024)
topo, ps, ms, _, ovf = build(cap)
assert not bool(ovf)
nr = np.asarray(topo.node_ranges)
ni = int(np.asarray(topo.left_child).shape[0])
par = np.asarray(topo.parent)
ps_np = np.asarray(ps, np.float64)
ms_np = np.asarray(ms, np.float64)
cw = np.concatenate([np.zeros((1, 3)), np.cumsum(ps_np * ms_np[:, None], 0)])
cm = np.concatenate([[0.0], np.cumsum(ms_np)])
s, e = nr[:, 0], nr[:, 1]
ok = e >= s
ss, ee = np.where(ok, s, 0), np.where(ok, e + 1, 0)
msum = cm[ee] - cm[ss]
cent = np.where(ok[:, None], (cw[ee] - cw[ss]) / np.maximum(msum, 1e-300)[:, None], 0.0)
depth = np.zeros(par.shape[0], np.int64)
cur = par.copy()
while np.any(cur >= 0):
    depth += cur >= 0
    cur = np.where(cur >= 0, par[np.maximum(cur, 0)], -1)
levels = int(depth.max()) + 1
print(
    f"N={args.n} cml={args.cml}: live leaves {live}, capacity {cap}, nodes {nr.shape[0]}, "
    f"tree levels {levels}",
    flush=True,
)

lr = jnp.asarray(nr[ni:], jnp.int32)
cl = jnp.asarray(cent[ni:], jnp.float32)
tot = nr.shape[0]
ref = None
for v in args.variants:
    b, c, w = (int(t) for t in v.split(":"))
    kw = dict(
        order=args.order,
        num_internal=ni,
        total_nodes=tot,
        leaf_width=args.leaf,
        num_warps=None if w == 0 else w,
        block=b,
        chunk=max(c, 1),
    )
    f = jax.jit(lambda ps, ms, cl, lr: p2m_real_leaves_pallas(ps, ms, cl, lr, **kw))
    out = jax.block_until_ready(f(ps, ms, cl, lr))
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(ps, ms, cl, lr))
        ts.append(time.perf_counter() - t0)
    o = np.asarray(out)
    if ref is None:
        ref = o
    scale = np.abs(ref).max(axis=1, keepdims=True) + 1e-30
    err = float((np.abs(o - ref) / scale).max())
    print(
        f"  p{args.order} block {b:3d} chunk {c:3d} warps {w}: min {1e3 * min(ts):7.2f} ms "
        f"median {1e3 * np.median(ts):7.2f}  max row-scaled diff vs first {err:.1e}",
        flush=True,
    )
