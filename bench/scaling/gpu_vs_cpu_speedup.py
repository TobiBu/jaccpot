"""Figure 07 -- single-GPU vs CPU speedup vs N.

JAX fixes its backend at first use, process-wide, so one process cannot time both
devices. This script therefore re-executes *itself* once per device with
``JAX_PLATFORMS`` pinned, collects each worker's JSON from stdout, and writes one
combined artifact. That keeps the figure reproducible from a single command --
``python -m bench.scaling.gpu_vs_cpu_speedup`` -- rather than requiring the runner
to remember to launch two processes and merge them by hand.

Both arms time the identical quantity: ``evaluate_prepared_state`` on a prebuilt
tree, at the same N, seed, order, theta, leaf size and precision, using the shared
protocol from ``_timing``. Only the backend differs, which is the whole point --
if anything else differed the ratio would not be a speedup.

Two honesty notes the figure has to carry:

* The CPU arm is cut off at ``--cpu-max-n`` (the O(N log N) constant is large
  enough on CPU that the top of the GPU ladder would take hours). N values above
  the cutoff have a GPU time and no ratio, and the notebook must not extrapolate
  the curve past the last measured pair.
* This host is a **shared** 72-core box with 8 GPUs. A CPU timing here is not a
  dedicated-machine CPU timing; the JSON records the core count and the fact that
  the run was shared, so the ratio is read as indicative rather than as a
  hardware review.

Usage
-----
    python -m bench.scaling.gpu_vs_cpu_speedup --n 4096,16384,65536,262144
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "scaling/gpu_vs_cpu.json"
WORKER_SENTINEL = "--worker-device"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n",
        default="4096,16384,65536,262144,1048576",
        help="Comma-separated particle counts",
    )
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--theta", type=float, default=0.6)
    p.add_argument("--basis", default="real")
    # `large_n_gpu`, not `accurate`. Measured on an A100 at N=16384, leaf=64,
    # p=4, evaluating a prebuilt tree: accurate 27.3 s vs large_n_gpu 197 ms, a
    # factor of 139 -- and steady state, not compilation (calls 1-5 all within 2%
    # of each other). Timing the GPU on a preset that does not use the radix fast
    # lane would report an artefact of preset choice as a property of the device.
    # The accuracy figures (01-03) stay on `accurate`, where accuracy is the point
    # and N is modest.
    p.add_argument("--preset", default="large_n_gpu")
    p.add_argument("--leaf-size", type=int, default=64)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--softening", type=float, default=1e-3)
    p.add_argument(
        "--cpu-max-n",
        type=int,
        default=1 << 17,
        help="Skip the CPU arm above this N (default: 131072)",
    )
    p.add_argument(
        WORKER_SENTINEL,
        default=None,
        choices=("cpu", "gpu"),
        help="Internal: run as a single-device worker and emit JSON on stdout",
    )
    runmeta.add_common_args(p)
    p.set_defaults(dtype="float32")
    return p.parse_args()


def _worker(args: argparse.Namespace) -> int:
    """Time one device and print a JSON blob on stdout."""

    device = getattr(args, "worker_device")
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        runmeta.select_gpu(args.gpu_select)
    runmeta.enable_x64(args.dtype)

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402

    from bench.scaling import _timing as T  # noqa: E402
    from jaccpot import FastMultipoleMethod  # noqa: E402

    backend = jax.default_backend()
    if device == "gpu" and backend == "cpu":
        print(
            json.dumps({"device": device, "error": "no GPU backend available"}),
            flush=True,
        )
        return 0

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    ns = [int(v) for v in str(args.n).split(",") if v.strip()]
    rows: list[dict[str, Any]] = []

    for n in ns:
        if device == "cpu" and n > int(args.cpu_max_n):
            rows.append({"n": int(n), "skipped": "above --cpu-max-n"})
            continue
        key = jax.random.PRNGKey(runmeta.seed_sequence(args.seed, "uniform_cube", n))
        points, charges = T.distribution(key, n, "uniform_cube", dtype)
        solver = FastMultipoleMethod(
            preset=args.preset,
            basis=args.basis,
            theta=float(args.theta),
            softening=float(args.softening),
        )
        try:
            prepared = solver.prepare_state(
                points,
                charges,
                leaf_size=int(args.leaf_size),
                max_order=int(args.order),
                theta=float(args.theta),
            )
            tmin, tmean, tstd = T.time_min_repeat(
                lambda: solver.evaluate_prepared_state(prepared),
                warmup=int(args.warmup),
                repeats=int(args.repeats),
            )
            rows.append({"n": int(n), "min_s": tmin, "mean_s": tmean, "std_s": tstd})
        except Exception as exc:
            rows.append({"n": int(n), "error": str(exc)[:300]})
        print(f"[{device}] N={n} done", file=sys.stderr, flush=True)

    payload = {
        "device": device,
        "backend": backend,
        "device_kind": runmeta.device_label(),
        "rows": rows,
        "meta": runmeta.run_meta(),
    }
    print(json.dumps(payload), flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    if getattr(args, "worker_device", None):
        return _worker(args)

    forwarded = [
        "--n",
        str(args.n),
        "--order",
        str(args.order),
        "--theta",
        str(args.theta),
        "--basis",
        str(args.basis),
        "--preset",
        str(args.preset),
        "--leaf-size",
        str(args.leaf_size),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--softening",
        str(args.softening),
        "--cpu-max-n",
        str(args.cpu_max_n),
        "--seed",
        str(args.seed),
        "--dtype",
        str(args.dtype),
        "--gpu-select",
        str(args.gpu_select),
    ]

    results: dict[str, Any] = {}
    for device in ("gpu", "cpu"):
        print(f"[fig07] launching {device} worker ...", flush=True)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "bench.scaling.gpu_vs_cpu_speedup",
                WORKER_SENTINEL,
                device,
                *forwarded,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"[fig07] {device} worker exited {proc.returncode}")
            print(proc.stderr[-2000:])
            results[device] = {"device": device, "error": "worker failed"}
            continue
        # The worker's JSON is the last non-empty stdout line, so incidental
        # prints upstream cannot corrupt the parse.
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        try:
            results[device] = json.loads(lines[-1])
        except Exception as exc:
            print(f"[fig07] could not parse {device} worker output: {exc}")
            results[device] = {"device": device, "error": "unparseable output"}

    def _by_n(payload: dict[str, Any]) -> dict[int, Any]:
        return {int(r["n"]): r for r in payload.get("rows", []) if "n" in r}

    gpu_rows = _by_n(results.get("gpu", {}))
    cpu_rows = _by_n(results.get("cpu", {}))
    ns = sorted(set(gpu_rows) | set(cpu_rows))

    records: list[dict[str, Any]] = []
    for n in ns:
        g = gpu_rows.get(n, {})
        c = cpu_rows.get(n, {})
        g_t = g.get("min_s")
        c_t = c.get("min_s")
        records.append(
            {
                "n": n,
                "gpu_min_s": g_t,
                "cpu_min_s": c_t,
                "gpu_std_s": g.get("std_s"),
                "cpu_std_s": c.get("std_s"),
                "speedup": (c_t / g_t) if (g_t and c_t) else None,
                "cpu_skipped": c.get("skipped"),
                "gpu_error": g.get("error"),
                "cpu_error": c.get("error"),
            }
        )
        if records[-1]["speedup"]:
            print(
                f"N={n:<9d} cpu={c_t*1e3:9.2f} ms  gpu={g_t*1e3:8.2f} ms  "
                f"speedup={records[-1]['speedup']:6.1f}x"
            )
        else:
            print(f"N={n:<9d} cpu={c.get('skipped') or c.get('error') or 'n/a'}")

    config = {
        "n": ns,
        "theta": float(args.theta),
        "order": int(args.order),
        "basis": args.basis,
        "seed": int(args.seed),
        "device": (
            f"{results.get('gpu', {}).get('device_kind', '?')} vs "
            f"{results.get('cpu', {}).get('device_kind', '?')}"
        ),
        "precision": args.dtype,
        "leaf_size": int(args.leaf_size),
        "preset": args.preset,
        "distribution": "uniform_cube",
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "softening": float(args.softening),
        "cpu_max_n": int(args.cpu_max_n),
        "timed_region": "evaluate_prepared_state on a prebuilt tree, both arms",
        "host_cpu_count": os.cpu_count(),
        "shared_host": (
            "measured on a shared 72-core / 8-GPU host; the CPU arm is not a "
            "dedicated-machine measurement and the ratio is indicative"
        ),
    }
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config=config,
        meta=runmeta.run_meta(
            {
                "argv": sys.argv[1:],
                "gpu_worker_meta": results.get("gpu", {}).get("meta"),
                "cpu_worker_meta": results.get("cpu", {}).get("meta"),
            }
        ),
        data={"records": records},
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
