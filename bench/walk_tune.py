"""Time the Pallas mutual walk alone on walk inputs captured from a real step.

usage: python bench/walk_tune.py CAPTURE_walk.npz --row ROW.json --variants soa:64:2:2048 ...

``CAPTURE_walk.npz`` (+ ``.json``) comes from ``bench/nearfield_capture.py``; the list
and queue capacities are the ones the bench row ``ROW.json`` settled on (its
``counts``). A variant is ``node_layout:block:num_warps:max_programs`` (the fifth
``fused_emit`` field went with the unfused emission in the 2026-10 cleanup, X5).
Every variant
must find the same lists: the far and near counts, the rounds, the peak wavefront and
two order-free checksums of each list (sum and sum of squares of ``a * 2^32 + b``
modulo 2^64) are compared with the first variant.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaccpot.pallas.mutual_walk_pallas import mutual_walk_pallas  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("capture")
ap.add_argument("--row", required=True)
ap.add_argument("--variants", nargs="+", default=["soa:64:2:2048"])
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()

data = dict(np.load(args.capture))
meta = json.load(open(os.path.splitext(args.capture)[0] + ".json"))
counts = json.load(open(args.row))["counts"]
queue = int(counts["queue_capacity"])
far_cap = int(counts["compact_far_pair_capacity"])
near_cap = int(counts["near_edge_capacity"])
print(
    f"nodes {data['left'].shape[0]} theta {meta['theta']} queue {queue} far_cap {far_cap} "
    f"near_cap {near_cap} (row: far {counts['far_pair_count']} near "
    f"{counts['total_neighbors']} rounds {counts['rounds']} peak {counts['peak_wavefront']})",
    flush=True,
)
dev = {k: jnp.asarray(v) for k, v in data.items()}


def _sums(a, b, n):
    live = jnp.arange(a.shape[0]) < n
    key = a.astype(jnp.uint64) * jnp.uint64(1 << 32) + b.astype(jnp.uint64)
    key = jnp.where(live, key, jnp.uint64(0))
    return jnp.sum(key), jnp.sum(key * key)


def make(layout: str, block: int, warps: int, programs: int):
    os.environ["JACCPOT_WALK_MAX_PROGRAMS"] = str(programs)
    # the env var is read at trace time and is not a static argument of the walk's
    # inner jit, whose cache would otherwise hand back the previous grid
    jax.clear_caches()

    @jax.jit
    def f(left, right, cent, rad, root, act):
        r = mutual_walk_pallas(
            left,
            right,
            cent,
            rad,
            float(meta["theta"]),
            root,
            max_pair_queue=queue,
            far_cap=far_cap,
            near_cap=near_cap,
            node_active=act,
            block=block,
            num_warps=warps,
            node_layout=layout,
        )
        return (
            r.far_count,
            r.near_count,
            r.rounds,
            r.peak_wavefront,
            r.far_overflow | r.near_overflow | r.queue_overflow,
            *_sums(r.far_a, r.far_b, r.far_count),
            *_sums(r.near_a, r.near_b, r.near_count),
        )

    return f


ins = tuple(
    dev[k] for k in ("left", "right", "centers", "radii", "root", "node_active")
)
ref = None
for v in args.variants:
    parts = v.split(":")
    layout, block, warps, programs = parts[:4]
    f = make(layout, int(block), int(warps), int(programs))
    try:
        out = [np.asarray(x) for x in jax.block_until_ready(f(*ins))]
    except Exception as exc:  # noqa: BLE001
        print(f"  {v}: FAILED {str(exc)[:300]}", flush=True)
        continue
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*ins))
        ts.append(time.perf_counter() - t0)
    if ref is None:
        ref = out
    same = all(np.array_equal(a, b) for a, b in zip(out, ref))
    print(
        f"  {v:20s}: min {1e3 * min(ts):8.2f} ms median {1e3 * np.median(ts):8.2f}  "
        f"far {int(out[0])} near {int(out[1])} rounds {int(out[2])} peak {int(out[3])} "
        f"overflow {bool(out[4])}  same-lists-as-first {same}",
        flush=True,
    )
