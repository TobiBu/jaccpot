"""Time near-field kernel variants alone on near-field inputs captured from a real step.

usage: python bench/nearfield_kernel_tune.py CAPTURE.npz --variants 32:0:1 16:16:1 ...

``CAPTURE.npz`` (+ ``CAPTURE.json``) comes from ``bench/nearfield_capture.py``. A
variant is ``target_subtile:source_tile:num_warps[:flags]`` (``source_tile`` 0 = the
scalar source loop; ``flags`` = the kernel's ``source_flags``, letters a, p, r). Each variant runs ``nearfield_leafpair_csr_sorted_direct_pallas``
jitted with the captured static options; the first variant is the reference for
the max relative difference, and every variant is scored against an fp64 sum of
the SAME interaction list (row + self) on ``--check`` random particles.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.pallas.nearfield_leafpair_csr import (
    nearfield_leafpair_csr_sorted_direct_pallas,
)

ap = argparse.ArgumentParser()
ap.add_argument("capture")
ap.add_argument("--variants", nargs="+", default=["32:0:1"])
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--check", type=int, default=2048)
ap.add_argument("--stats", action="store_true", help="print leaf and row statistics")
args = ap.parse_args()

data = dict(np.load(args.capture))
meta = json.load(open(os.path.splitext(args.capture)[0] + ".json"))
n = data["positions"].shape[0]
starts, lcount = data["leaf_start"], data["leaf_count"]
offsets, counts, nbr = data["offsets"], data["counts"], data["neighbors"]
live = lcount > 0
print(
    f"N={n} leaves {starts.size} (live {int(live.sum())}) leaf_width {meta['leaf_width']} "
    f"near entries {int(counts.sum())} (capacity {nbr.size}) "
    f"with_potential {meta.get('with_potential')} chunk {meta.get('chunk')}",
    flush=True,
)
if args.stats:
    lc = lcount[live]
    rows = counts[live]
    q = [50, 90, 99, 100]
    print(f"  particles per live leaf: mean {lc.mean():.1f} pct{q} {np.percentile(lc, q)}")
    print(f"  row entries per live leaf: mean {rows.mean():.1f} pct{q} {np.percentile(rows, q)}")
    src = lcount[nbr[: int(counts.sum())]] if counts.sum() else np.zeros(0)
    # pairs: targets x sources over rows, plus self
    tgt = np.repeat(lcount, counts)
    pairs = float(np.sum(tgt.astype(np.float64) * src)) + float(np.sum(lc.astype(np.float64) ** 2))
    print(f"  particle pairs {pairs:.3e}", flush=True)
    for bt in (16, 32):
        sub = np.ceil(lc / bt)
        print(f"  Bt={bt}: target-lane fill {lc.sum() / (sub.sum() * bt):.2f}")
    for bs in (8, 16, 32):
        blocks = np.ceil(src / bs)
        print(f"  Bs={bs}: source-lane fill {src.sum() / max(blocks.sum() * bs, 1):.2f}")

dev = {k: jnp.asarray(v) for k, v in data.items()}
static = {
    k: meta[k]
    for k in ("leaf_width", "chunk", "num_stages", "with_potential")
    if meta.get(k) is not None
}
soft = jnp.asarray(meta["softening_sq"], dev["positions"].dtype)
G = jnp.asarray(meta["G"], dev["positions"].dtype)


def make(bt: int, bs: int, warps: int, pf: str):
    @jax.jit
    def f(pos, mass, ls, lc, nb, off, cnt):
        return nearfield_leafpair_csr_sorted_direct_pallas(
            pos,
            mass,
            ls,
            lc,
            nb,
            off,
            cnt,
            softening_sq=soft,
            G=G,
            target_subtile=bt,
            num_warps=warps,
            source_tile=bs,
            source_flags=pf,
            **static,
        )

    return f


ins = tuple(
    dev[k]
    for k in ("positions", "masses", "leaf_start", "leaf_count", "neighbors", "offsets", "counts")
)

# fp64 truth of the same list on random particles
rng = np.random.default_rng(1)
leaf_of = np.searchsorted(starts, np.arange(n), side="right") - 1
pick = rng.choice(np.flatnonzero(lcount[np.clip(leaf_of, 0, None)] > 0), args.check, replace=False)
P = data["positions"].astype(np.float64)
M = data["masses"].astype(np.float64)
eps2 = float(meta["softening_sq"])
g = float(meta["G"])
truth = np.zeros((args.check, 3))
for i, p in enumerate(pick):
    lf = leaf_of[p]
    src_leaves = np.concatenate([nbr[offsets[lf] : offsets[lf] + counts[lf]], [lf]])
    idx = np.concatenate([np.arange(starts[s], starts[s] + lcount[s]) for s in src_leaves])
    idx = idx[idx != p]
    d = P[p] - P[idx]
    r2 = np.sum(d * d, 1) + eps2
    truth[i] = -g * np.sum((M[idx] * r2**-1.5)[:, None] * d, 0)
tnorm = np.linalg.norm(truth)

ref = None
for v in args.variants:
    parts = v.split(":")
    bt, bs, w = (int(t) for t in parts[:3])
    pf = parts[3] if len(parts) > 3 else ""
    f = make(bt, bs, w, pf)
    try:
        acc, _ = jax.block_until_ready(f(*ins))
    except Exception as exc:  # noqa: BLE001
        print(f"  Bt {bt:2d} Bs {bs:2d} warps {w} {pf:3s}: FAILED {str(exc)[:300]}", flush=True)
        continue
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*ins))
        ts.append(time.perf_counter() - t0)
    a = np.asarray(acc)
    if ref is None:
        ref = a
    scale = np.abs(ref).max()
    err = np.linalg.norm(a[pick].astype(np.float64) - truth) / tnorm
    print(
        f"  Bt {bt:2d} Bs {bs:2d} warps {w} {pf:3s}: min {1e3 * min(ts):8.2f} ms median "
        f"{1e3 * np.median(ts):8.2f}  max|d|/max|ref| {np.abs(a - ref).max() / scale:.2e} "
        f"bitwise {np.array_equal(a, ref)}  rel-L2 vs fp64 list {err:.3e}",
        flush=True,
    )
