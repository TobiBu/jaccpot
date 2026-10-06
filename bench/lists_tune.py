"""Time the directed CSR build (counts, placement, rank) alone on pairs captured from a real step.

usage: python bench/lists_tune.py CAPTURE_lists0.npz --variants base slices=2 slices=4 ...

``CAPTURE_lists{0,1}.npz`` (+ ``.json``) comes from ``bench/nearfield_capture.py``
(``0``: the far list's canonical pairs, ``1``: the near list's). A variant is
``base`` (the captured options) or comma-separated ``key=value`` overrides of
:func:`jaccpot.pallas.csr_place.directed_csr_pallas`'s keywords (``block``, ``lanes``,
``rows_per_program``, ``num_warps``, ``slices``, ...). Every variant must
build the same ``(sources, offsets, counts)`` to the bit as the first.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.pallas.csr_place import directed_csr_pallas  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("capture")
ap.add_argument("--variants", nargs="+", default=["base"])
ap.add_argument("--reps", type=int, default=7)
ap.add_argument(
    "--rounds", type=int, default=1, help="interleaved rounds over the variants"
)
args = ap.parse_args()

data = dict(np.load(args.capture))
meta = json.load(open(os.path.splitext(args.capture)[0] + ".json"))
idx = jnp.int64 if "64" in str(meta.get("idx", "int32")) else jnp.int32
base_kw = {
    k: meta[k] for k in ("num_rows", "row_offset", "pad_source", "slices") if k in meta
}
a = jnp.asarray(data["a"], idx)
b = jnp.asarray(data["b"], idx)
count = jnp.asarray(data["count"], idx)
print(
    f"{args.capture}: W {a.shape[0]} live {int(count)} rows {base_kw['num_rows']} "
    f"options {base_kw}",
    flush=True,
)


def parse(spec: str) -> dict:
    kw = dict(base_kw)
    if spec == "base":
        return kw
    for item in spec.split(","):
        k, v = item.split("=")
        kw[k] = int(v)
    return kw


fns, best, med = {}, {}, {}
ref = None
for spec in args.variants:
    kw = parse(spec)
    fns[spec] = jax.jit(
        lambda a, b, c, kw=kw: directed_csr_pallas(a, b, c, idx=idx, **kw)
    )
    out = jax.block_until_ready(fns[spec](a, b, count))
    host = [np.asarray(x) for x in out]
    if ref is None:
        ref = host
        same = "reference"
    else:
        same = (
            "bitwise"
            if all(np.array_equal(x, y) for x, y in zip(host, ref))
            else "DIFFERENT"
        )
    print(f"  {spec:<40} {same}", flush=True)
    best[spec], med[spec] = [], []
for r in range(args.rounds):
    for spec in args.variants:
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fns[spec](a, b, count))
            ts.append(1e3 * (time.perf_counter() - t0))
        best[spec].append(min(ts))
        med[spec].append(sorted(ts)[len(ts) // 2])
for spec in args.variants:
    print(
        f"  {spec:<40} min per round "
        + " ".join(f"{x:7.3f}" for x in best[spec])
        + f"  | median {min(med[spec]):7.3f} ms",
        flush=True,
    )
