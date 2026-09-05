"""Is the distributed ceiling the method, or the auto-sized caps? -- ANSWERED.

Result, on two A100s at leaf 256, M=2048, order 4, theta 0.5:

    N=131072  auto-sized                        OK, no overflow
    N=131072  max_pair_queue=4e6                OK, no overflow
    N=131072  max_pair_queue=1e6                OK, no overflow
    N=131072  pair_queue=1e6 + interactions=512 OK, no overflow
    N=262144  auto-sized                        out of memory (after 617 s)
    N=262144  max_pair_queue=4e6                out of memory (52 s)
    N=262144  max_pair_queue=1e6                out of memory (51 s)
    N=262144  pair_queue=1e6 + interactions=512 out of memory (51 s)

**The ceiling is not a caps artifact.** Tightening the pair queue by orders of
magnitude does not get N=262144 through at two devices, and the failing
allocation is 3.35 GiB in every tightened case -- the caps change how long it
takes to fail, not whether. So the distributed path's ceiling of P=393216 at two
devices, four times below the single-device radix path's P=3145728, is a
property of that path rather than of a default someone chose.

That mattered enough to measure because the same session had already mistaken a
leaf-size default for a real ceiling once, and for a set of impossible sub-unity
timing ratios.

The original question, kept because it is why the script exists:

At ndev=2 the distributed path runs N=131072 and fails at N=262144, four times
below the single-device radix path. Every buffer cap defaults to None -- auto
sized from N -- so this asks whether tightening them lets N=262144 through, and
whether the force is still right when they are tightened.

A too-small cap does NOT fail loudly: it drops interactions and returns a wrong
force. So every attempt reports the evaluator's overflow flag, and a run that
overflowed is a failure however well it fits in memory.
"""

import argparse
import json
import os
import sys
import time
import traceback

_p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_p.add_argument("--n", default="131072,262144", help="Source counts to try")
_p.add_argument("--tracers", type=int, default=2048, help="Tracer count M")
_p.add_argument("--leaf-size", type=int, default=256, help="Leaf occupancy")
_p.add_argument("--device-count", type=int, default=2, help="Devices in the mesh")
_p.add_argument(
    "--json-out",
    default="",
    help="Output path; defaults to the committed artifact location",
)
_p.add_argument(
    "--host-devices",
    type=int,
    default=0,
    help=(
        "Force this many CPU devices instead of using the real ones. Only for "
        "the CI smoke test -- it makes the probe runnable without a GPU, and "
        "must be set before JAX initialises, which is why the CLI is parsed "
        "above the jax import rather than in a main()"
    ),
)
_args = _p.parse_args()
if _args.host_devices:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + " --xla_force_host_platform_device_count=%d" % _args.host_devices
    )

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import pathlib as _pathlib

REPO_ROOT = _pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.applications.density_reconstruction.distributed import (
    make_distributed_forward_operator,
)
from jaccpot.applications.density_reconstruction.truth import (
    TruthConfig,
    sample_composite,
    sample_tracers,
)

print("devices:", [str(d) for d in jax.devices()], flush=True)

NDEV, M, LEAF = _args.device_count, _args.tracers, _args.leaf_size
results = []


def attempt(n, caps, label):
    cfg = TruthConfig(num_particles=n, num_tracers=M, seed=0, softening=1e-2)
    pos = sample_composite(cfg)
    trac = sample_tracers(cfg, pos)
    t0 = time.perf_counter()
    try:
        op = make_distributed_forward_operator(
            tracer_positions=trac,
            source_mass=1.0 / n,
            num_sources=n,
            num_devices=NDEV,
            softening=1e-2,
            order=4,
            theta=0.5,
            leaf_size=LEAF,
            caps=caps or None,
        )
        part = op.prepare(pos)
        acc = op.evaluate_at_topology(part, jnp.asarray(pos))
        g = jax.grad(lambda p: jnp.sum(op.evaluate_at_topology(part, p) ** 2))(
            jnp.asarray(pos)
        )
        jax.block_until_ready(g)
        ov = part.diagnostics["overflowed"]
        finite = bool(jnp.all(jnp.isfinite(g)))
        verdict = (
            "OVERFLOWED (force is wrong)" if ov else ("OK" if finite else "NONFINITE")
        )
        print(
            f"  N={n:>8} {label:34s} {verdict:28s} {time.perf_counter()-t0:6.0f}s",
            flush=True,
        )
        results.append(
            dict(
                N=n,
                caps=caps,
                label=label,
                ran=True,
                overflowed=bool(ov),
                gradient_finite=finite,
                seconds=time.perf_counter() - t0,
            )
        )
    except Exception as exc:
        print(
            f"  N={n:>8} {label:34s} FAILED {type(exc).__name__:20s} "
            f"{time.perf_counter()-t0:6.0f}s",
            flush=True,
        )
        results.append(
            dict(
                N=n,
                caps=caps,
                label=label,
                ran=False,
                error_type=type(exc).__name__,
                error=str(exc)[:300],
                seconds=time.perf_counter() - t0,
            )
        )


# The baseline the ceiling was measured at, then progressively tighter caps.
LADDER = [
    ({}, "auto-sized (the default)"),
    ({"max_pair_queue": 4_000_000}, "max_pair_queue=4e6"),
    ({"max_pair_queue": 1_000_000}, "max_pair_queue=1e6"),
    (
        {"max_pair_queue": 1_000_000, "max_interactions_per_node": 512},
        "pair_queue=1e6 + interactions=512",
    ),
]
for n in [int(v) for v in str(_args.n).split(",") if v.strip()]:
    for caps, label in LADDER:
        attempt(n, dict(caps), label)

out = str(
    _pathlib.Path(_args.json_out).expanduser()
    if _args.json_out
    else REPO_ROOT
    / "bench"
    / "results"
    / "density_reconstruction"
    / "distributed_caps_probe.json"
)
# Written as the same {config, meta, data} envelope every other artifact in
# this directory uses. The COMMITTED distributed_caps_probe.json predates this
# and is a bare list with no provenance block: it is the record of a two-A100
# run whose configuration is in this module's docstring, and it is not
# regenerated here because nothing reads it as a figure source and re-running it
# costs two devices for half an hour. Runs from now on carry their provenance.
sys.path.insert(0, str(REPO_ROOT / "examples" / "jaccpot_paper"))
from common import runmeta  # noqa: E402

payload = {
    "config": {
        "n": [int(v) for v in str(_args.n).split(",") if v.strip()],
        "theta": 0.5,
        "order": 4,
        "basis": "solidfmm",
        "seed": 0,
        "device": runmeta.device_label(),
        "precision": "float64",
        "M": M,
        "leaf_size": LEAF,
        "device_count": NDEV,
        "ladder": [label for _, label in LADDER],
    },
    "meta": runmeta.run_meta(),
    "data": {"attempts": results},
}
json.dump(payload, open(out, "w"), indent=2)
print("wrote", out, flush=True)
