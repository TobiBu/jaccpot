"""CI guard for acceleration/jerk runtime path sanity.

This guard intentionally uses broad ratio bounds to avoid hardware-specific
flakiness while still catching major runtime regressions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from bench.bench_parallel_paths import collect_metrics


def _validate_metrics(metrics: dict[str, float]) -> None:
    acc = float(metrics["acc_mean_seconds"])
    jerk_fast = float(metrics["jerk_fast_mean_seconds"])
    jerk_acc = float(metrics["jerk_accurate_mean_seconds"])
    ratio_fast = float(metrics["jerk_fast_over_acc"])
    ratio_acc = float(metrics["jerk_accurate_over_fast"])

    if not (acc > 0.0 and jerk_fast > 0.0 and jerk_acc > 0.0):
        raise RuntimeError("non-positive benchmark timings observed")
    if not (0.4 <= ratio_fast <= 8.0):
        raise RuntimeError(
            f"jerk_fast_over_acc out of guard range: {ratio_fast:.3f} "
            "(expected 0.4..8.0)"
        )
    if not (1.2 <= ratio_acc <= 15.0):
        raise RuntimeError(
            f"jerk_accurate_over_fast out of guard range: {ratio_acc:.3f} "
            "(expected 1.2..15.0)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--preset", type=str, default="fast")
    parser.add_argument("--basis", type=str, default="solidfmm")
    parser.add_argument("--theta", type=float, default=0.6)
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--jerk-fd-dt", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    metrics = collect_metrics(
        n=int(args.n),
        runs=int(args.runs),
        warmup=int(args.warmup),
        preset=str(args.preset),
        basis=str(args.basis),
        theta=float(args.theta),
        leaf_size=int(args.leaf_size),
        max_order=int(args.max_order),
        jerk_fd_dt=float(args.jerk_fd_dt),
        seed=int(args.seed),
        dtype=dtype,
    )

    _validate_metrics(metrics)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
