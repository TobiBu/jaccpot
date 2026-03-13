"""Benchmark acceleration and jerk execution paths.

This script focuses on relative runtime behavior between:
- acceleration-only evaluation,
- jerk in `fast_approx` mode,
- jerk in `accurate` mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from examples.benchmark_utils import time_callable
from jaccpot import FastMultipoleMethod


def _sample_problem(
    n: int,
    *,
    key: jax.Array,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    key_pos, key_mass, key_vel = jax.random.split(key, 3)
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=dtype,
    )
    masses = jax.random.uniform(
        key_mass,
        (n,),
        minval=0.5,
        maxval=1.5,
        dtype=dtype,
    )
    velocities = jax.random.uniform(
        key_vel,
        (n, 3),
        minval=-0.2,
        maxval=0.2,
        dtype=dtype,
    )
    return positions, masses, velocities


def collect_metrics(
    *,
    n: int,
    runs: int,
    warmup: int,
    preset: str,
    basis: str,
    theta: float,
    leaf_size: int,
    max_order: int,
    jerk_fd_dt: float,
    seed: int,
    dtype: jnp.dtype,
) -> dict[str, Any]:
    key = jax.random.PRNGKey(seed)
    positions, masses, velocities = _sample_problem(n, key=key, dtype=dtype)
    solver = FastMultipoleMethod(preset=preset, basis=basis, theta=float(theta))

    acc_timing = time_callable(
        solver.compute_accelerations,
        positions,
        masses,
        leaf_size=leaf_size,
        max_order=max_order,
        warmup=warmup,
        runs=runs,
    )
    jerk_fast_timing = time_callable(
        solver.compute_accelerations_and_jerk,
        positions,
        masses,
        velocities,
        leaf_size=leaf_size,
        max_order=max_order,
        jerk_mode="fast_approx",
        jerk_fd_dt=jerk_fd_dt,
        warmup=warmup,
        runs=runs,
    )
    jerk_acc_timing = time_callable(
        solver.compute_accelerations_and_jerk,
        positions,
        masses,
        velocities,
        leaf_size=leaf_size,
        max_order=max_order,
        jerk_mode="accurate",
        jerk_fd_dt=jerk_fd_dt,
        warmup=warmup,
        runs=runs,
    )

    return {
        "n": int(n),
        "dtype": str(jnp.dtype(dtype)),
        "preset": str(preset),
        "basis": str(basis),
        "theta": float(theta),
        "leaf_size": int(leaf_size),
        "max_order": int(max_order),
        "warmup": int(warmup),
        "runs": int(runs),
        "acc_mean_seconds": float(acc_timing.mean),
        "jerk_fast_mean_seconds": float(jerk_fast_timing.mean),
        "jerk_accurate_mean_seconds": float(jerk_acc_timing.mean),
        "jerk_fast_over_acc": float(jerk_fast_timing.mean / acc_timing.mean),
        "jerk_accurate_over_fast": float(jerk_acc_timing.mean / jerk_fast_timing.mean),
    }


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

    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
