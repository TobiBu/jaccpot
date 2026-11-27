"""Table row (was figure 07) -- per-step time of the fused lane on one device, GPU vs CPU.

REBASE_PLAN demotes the GPU-vs-CPU figure to a sentence or a table row: the fused
lane now runs on CPU too (#509, the CPU route: ``RuntimePolicyConfig(cpu_fused_lane
=True)``, its Pallas kernels interpreted), so "the same code on two devices" is a
pair of numbers, not a curve.

One run times ONE device -- whatever backend JAX starts on -- at a few N:
``FusedForceProvider.step`` jitted (rebuild + evaluate, the per-step cost, as
figure 04's headline) and evaluation only, minimum over ``--repeats``
(``_timing.time_min_repeat``), on ``bench/scaling/_galaxy.py``'s solver
(``preset="galaxy"``, D3). ``bench/scaling/_galaxy.galaxy_solver`` switches the CPU
route on by itself when the backend is a CPU, so both runs execute the lane's
programs; only the backend differs. A row whose capacity verdict fails is recorded
``valid: false`` and gets no ratio.

``--pair GPU_JSON CPU_JSON`` then merges a GPU run and a CPU run made with the same
``--n``, seed, distribution and softening into the table (``scaling/gpu_vs_cpu.json``,
the schema figure 07's notebook reads: ``gpu_min_s``, ``cpu_min_s``, ``speedup``
per N). It refuses runs whose configurations differ.

Honesty notes the table has to carry:

* The CPU arm runs the Pallas kernels in **interpret mode** (the CPU route), not
  a CPU-tuned FMM: the ratio is "this code on this CPU vs this code on this GPU",
  not a statement about CPUs. ``--cpu-max-n`` caps the CPU arm (interpreted, the
  walk alone takes seconds at N = 4096).
* A CPU timing on a shared host is not a dedicated-machine timing; the record
  carries the host's core count, and ``card_load`` on the GPU run.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu python -m bench.scaling.gpu_vs_cpu_speedup --n 2048 \\
        --repeats 2 --warmup 1 --gpu-select none --json-out /tmp/cpu.json

Paper rows (one run per device, then the pair)::

    python -m bench.scaling.gpu_vs_cpu_speedup --json-out scaling/per_step_gpu.json
    JAX_PLATFORMS=cpu python -m bench.scaling.gpu_vs_cpu_speedup --gpu-select none \\
        --json-out scaling/per_step_cpu.json
    python -m bench.scaling.gpu_vs_cpu_speedup \\
        --pair scaling/per_step_gpu.json scaling/per_step_cpu.json
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "scaling/per_step_{backend}.json"
PAIR_OUT = "scaling/gpu_vs_cpu.json"
#: config fields two runs must share to be paired
PAIR_KEYS = ("n", "seed", "distribution", "softening", "G", "galaxy_settings")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n",
        default="1e4,1e5,1e6",
        help="Comma-separated particle counts, 1e6 notation allowed",
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--distribution", default="plummer_clipped")
    p.add_argument("--rmax", type=float, default=20.0)
    p.add_argument("--softening", type=float, default=1e-7)
    p.add_argument("--G", dest="g", type=float, default=1.0)
    p.add_argument(
        "--cpu-max-n",
        type=int,
        default=1 << 20,
        help="On a CPU backend, skip N above this (default: 1048576)",
    )
    p.add_argument(
        "--pair",
        nargs=2,
        metavar=("GPU_JSON", "CPU_JSON"),
        default=None,
        help="Merge a GPU run and a CPU run into the table instead of timing",
    )
    runmeta.add_common_args(p)
    p.set_defaults(dtype="float32")
    return p.parse_args()


def _time_device(args: argparse.Namespace) -> int:
    runmeta.select_gpu(args.gpu_select)
    from bench.scaling import _timing as T  # noqa: E402  (imports no jax)

    env = T.launch_env()

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402

    from bench.scaling import _galaxy as GX  # noqa: E402

    backend = jax.default_backend()
    ns = [int(float(v)) for v in str(args.n).split(",") if v.strip()]
    rows: list[dict[str, Any]] = []
    galaxy: dict[str, Any] = {}
    for n in ns:
        if backend == "cpu" and n > int(args.cpu_max_n):
            rows.append({"n": int(n), "skipped": "above --cpu-max-n"})
            continue
        key = jax.random.PRNGKey(runmeta.seed_sequence(args.seed, args.distribution, n))
        x, m = T.distribution(key, n, args.distribution, jnp.float32, rmax=args.rmax)
        x, m = jax.block_until_ready((x, m))
        load = GX.card_load()
        try:
            solver = GX.galaxy_solver(softening=float(args.softening), G=float(args.g))
            galaxy = galaxy or GX.describe(solver)
            fs, setup = T.warm_fused_step(solver, x, m)
            reps = dict(warmup=int(args.warmup), repeats=int(args.repeats))
            s_min, s_mean, s_std = T.time_min_repeat(lambda: fs.step(x, m), **reps)
            e_min, e_mean, e_std = T.time_min_repeat(lambda: fs.evaluate(), **reps)
            acc, aux = fs.step(x, m)
            jax.block_until_ready(acc)
            ok = bool(fs.ok(aux)) and bool(setup["first_step_ok"])
            load_after = GX.card_load()
            ok = ok and GX.card_free(load, load_after)
            rows.append(
                {
                    "n": int(n),
                    "valid": ok,
                    "min_s": s_min,
                    "mean_s": s_mean,
                    "std_s": s_std,
                    "evaluate_min_s": e_min,
                    "evaluate_mean_s": e_mean,
                    "evaluate_std_s": e_std,
                    "setup": setup,
                    "card_load_before": load,
                    "card_load_after": load_after,
                }
            )
            fs = None  # free the provider before the next N
            print(
                f"[{backend}] N={n:<10d} valid={ok} step={s_min * 1e3:10.2f} ms "
                f"evaluate={e_min * 1e3:9.2f} ms",
                flush=True,
            )
        except Exception as exc:
            print(f"[{backend}] N={n:<10d} FAILED: {str(exc)[:200]}")
            rows.append({"n": int(n), "valid": False, "error": str(exc)[:400]})

    config = {
        "n": ns,
        "theta": float(GX.GALAXY.theta),
        "order": int(GX.GALAXY.max_order),
        "basis": "real",
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "backend": backend,
        "precision": "float32",
        "leaf_size": int(GX.GALAXY.leaf_size),
        "preset": "galaxy",
        "galaxy": galaxy,
        "galaxy_settings": {k: v for k, v in galaxy.items() if k != "cpu_route"},
        "distribution": args.distribution,
        "softening": float(args.softening),
        "G": float(args.g),
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "cpu_max_n": int(args.cpu_max_n),
        "host_cpu_count": os.cpu_count(),
        "launch_env": env,
        "timed_region": "FusedForceProvider.step jitted (rebuild + evaluate); "
        "evaluate_* = evaluation only on the prepare's topology",
    }
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT.format(backend=backend),
        config=config,
        meta=runmeta.run_meta({"argv": sys.argv[1:]}),
        data={"records": rows},
    )
    print(f"\nwrote {out}")
    return 0


def _pair(args: argparse.Namespace) -> int:
    gpu = jsonio.read_result(args.pair[0])
    cpu = jsonio.read_result(args.pair[1])
    if gpu["config"].get("backend") != "gpu" or cpu["config"].get("backend") != "cpu":
        print(
            "[fig07] --pair wants a GPU run first and a CPU run second; got "
            f"{gpu['config'].get('backend')!r} and {cpu['config'].get('backend')!r}"
        )
        return 2
    differ = [k for k in PAIR_KEYS if gpu["config"].get(k) != cpu["config"].get(k)]
    if differ:
        print(f"[fig07] the two runs differ in {differ}; they are not a pair")
        return 2

    def _by_n(art: dict[str, Any]) -> dict[int, Any]:
        return {int(r["n"]): r for r in art["data"]["records"]}

    g_rows, c_rows = _by_n(gpu), _by_n(cpu)
    records: list[dict[str, Any]] = []
    for n in sorted(set(g_rows) | set(c_rows)):
        g, c = g_rows.get(n, {}), c_rows.get(n, {})
        g_t = g.get("min_s") if g.get("valid") else None
        c_t = c.get("min_s") if c.get("valid") else None
        rec = {
            "n": n,
            "gpu_min_s": g_t,
            "cpu_min_s": c_t,
            "gpu_std_s": g.get("std_s"),
            "cpu_std_s": c.get("std_s"),
            "gpu_evaluate_min_s": g.get("evaluate_min_s"),
            "cpu_evaluate_min_s": c.get("evaluate_min_s"),
            "speedup": (c_t / g_t) if (g_t and c_t) else None,
            "cpu_skipped": c.get("skipped"),
            "gpu_error": g.get("error")
            or (None if g.get("valid", True) else "capacity verdict failed"),
            "cpu_error": c.get("error")
            or (None if c.get("valid", True) else "capacity verdict failed"),
        }
        records.append(rec)
        print(
            f"N={n:<10d} gpu={g_t} cpu={c_t} speedup="
            + (f"{rec['speedup']:.1f}x" if rec["speedup"] else "n/a")
        )
    config = dict(gpu["config"])
    config.update(
        {
            "device": f"{gpu['config']['device']} vs {cpu['config']['device']}",
            "backend": "gpu vs cpu",
            "gpu_config": gpu["config"],
            "cpu_config": cpu["config"],
            "paired_from": list(args.pair),
            "timed_region": gpu["config"].get("timed_region"),
            "cpu_route": "the CPU arm runs the lane's Pallas kernels in interpret "
            "mode (cpu_fused_lane); the ratio is not a CPU-FMM comparison",
        }
    )
    out = jsonio.write_result(
        args.json_out or PAIR_OUT,
        config=config,
        meta=runmeta.merged_meta([gpu["meta"], cpu["meta"]]),
        data={"records": records},
    )
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    args = _parse_args()
    if args.dtype != "float32":
        print("[fig07] preset='galaxy' is fp32 (D3); --dtype float64 is not this row")
        return 2
    if args.pair:
        return _pair(args)
    return _time_device(args)


if __name__ == "__main__":
    raise SystemExit(main())
