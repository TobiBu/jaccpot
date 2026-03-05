"""Run bench_parallel_paths.py and fail on latency regressions vs baseline."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=pathlib.Path,
        default=pathlib.Path("bench/benchmark_baseline.json"),
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.25,
        help="Maximum allowed slowdown ratio (e.g. 0.25 = +25%%).",
    )
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--p", type=int, default=4)
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument("--theta", type=float, default=0.6)
    parser.add_argument("--target-frac", type=float, default=0.10)
    parser.add_argument("--p-gears", type=str, default="2,2,3,3,4")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    return parser.parse_args()


def _load_baseline(path: pathlib.Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"baseline file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("target_eval_mean_s", "adaptive_prepare_mean_s")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"baseline missing keys: {', '.join(missing)}")
    return {k: float(payload[k]) for k in required}


def _run_benchmark(args: argparse.Namespace) -> Dict[str, float]:
    cmd = [
        sys.executable,
        "bench/bench_parallel_paths.py",
        "--n",
        str(int(args.n)),
        "--p",
        str(int(args.p)),
        "--leaf-size",
        str(int(args.leaf_size)),
        "--theta",
        str(float(args.theta)),
        "--target-frac",
        str(float(args.target_frac)),
        "--p-gears",
        str(args.p_gears),
        "--warmup",
        str(int(args.warmup)),
        "--runs",
        str(int(args.runs)),
        "--dtype",
        str(args.dtype),
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "bench_parallel_paths.py failed.\n"
            f"stdout:\n{exc.stdout}\n\nstderr:\n{exc.stderr}"
        ) from exc
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    timing_lines = [ln for ln in lines if ln.startswith("timings_s ")]
    if not timing_lines:
        raise RuntimeError(
            "bench_parallel_paths.py did not produce a timings_s line.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    fields = timing_lines[-1].split()[1:]
    parsed: Dict[str, float] = {}
    for field in fields:
        key, value = field.split("=", 1)
        parsed[key] = float(value)
    required = ("target_eval_mean", "adaptive_prepare_mean")
    missing = [k for k in required if k not in parsed]
    if missing:
        raise RuntimeError(f"timings_s missing keys: {', '.join(missing)}")
    return {
        "target_eval_mean_s": float(parsed["target_eval_mean"]),
        "adaptive_prepare_mean_s": float(parsed["adaptive_prepare_mean"]),
    }


def main() -> None:
    args = _parse_args()
    baseline = _load_baseline(args.baseline)
    observed = _run_benchmark(args)
    allowed = 1.0 + float(args.max_regression)

    failures = []
    for metric_name in ("target_eval_mean_s", "adaptive_prepare_mean_s"):
        base = baseline[metric_name]
        value = observed[metric_name]
        ratio = value / base if base > 0 else float("inf")
        status = "PASS" if ratio <= allowed else "FAIL"
        print(
            f"{metric_name}: observed={value:.6f}s baseline={base:.6f}s "
            f"ratio={ratio:.3f} allowed<={allowed:.3f} [{status}]"
        )
        if ratio > allowed:
            failures.append(metric_name)

    if failures:
        raise SystemExit(
            "Benchmark regression guard failed for: " + ", ".join(failures)
        )


if __name__ == "__main__":
    main()
