"""Time the COM-radii pass alone on a tree captured from a real step.

usage: python bench/com_radii_tune.py CAPTURE_comr.npz --variants table table:block=32 chain ...

``CAPTURE_comr.npz`` (+ ``.json``) comes from ``bench/nearfield_capture.py``. A variant is
``table`` / ``chain`` with optional ``:key=value,...`` overrides of
:func:`jaccpot.runtime._mac_geometry._com_radii`'s ``block``, ``chunk``, ``lanes``,
``num_warps``. Every variant must give the first one's radii to the bit.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.runtime._mac_geometry import _com_radii  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("capture")
ap.add_argument("--variants", nargs="+", default=["table", "table:block=32"])
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--rounds", type=int, default=3)
args = ap.parse_args()

data = {k: jnp.asarray(v) for k, v in np.load(args.capture).items()}
meta = json.load(open(os.path.splitext(args.capture)[0] + ".json"))
base = {
    k: meta[k] for k in ("leaf_cap", "internal", "num_levels", "kernel") if k in meta
}
order = (
    "node_ranges",
    "left_child",
    "right_child",
    "parent",
    "positions_sorted",
    "centers",
)
arrs = [data[k] for k in order]
print(
    f"nodes {arrs[0].shape[0]} particles {arrs[4].shape[0]} options {base}", flush=True
)

fns, ref, best = {}, None, {}
for spec in args.variants:
    name, _, rest = spec.partition(":")
    kw = dict(base, variant=name)
    for item in filter(None, rest.split(",")):
        k, v = item.split("=")
        kw[k] = int(v)
    fns[spec] = jax.jit(lambda *a, kw=kw: _com_radii(*a, **kw))
    out = np.asarray(jax.block_until_ready(fns[spec](*arrs)))
    if ref is None:
        ref, same = out, "reference"
    else:
        same = (
            "bitwise"
            if np.array_equal(out, ref)
            else f"DIFFERENT (max {np.max(np.abs(out - ref)):.3e})"
        )
    print(f"  {spec:<32} {same}", flush=True)
    best[spec] = []
for _ in range(args.rounds):
    for spec in args.variants:
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fns[spec](*arrs))
            ts.append(1e3 * (time.perf_counter() - t0))
        best[spec].append(min(ts))
for spec in args.variants:
    print(
        f"  {spec:<32} min per round " + " ".join(f"{x:7.3f}" for x in best[spec]),
        flush=True,
    )
