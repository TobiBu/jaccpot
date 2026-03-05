"""Benchmark target-subset eval and adaptive-order prepare_state latency."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=12000, help="Number of particles")
    parser.add_argument("--p", type=int, default=4, help="Multipole order")
    parser.add_argument("--leaf-size", type=int, default=16, help="Leaf size")
    parser.add_argument("--theta", type=float, default=0.6, help="MAC opening angle")
    parser.add_argument(
        "--target-frac",
        type=float,
        default=0.10,
        help="Fraction of particles used in target-subset evaluation",
    )
    parser.add_argument(
        "--p-gears",
        type=str,
        default="2,3,4",
        help="Comma-separated adaptive orders for prepare_state benchmark",
    )
    parser.add_argument("--device", choices=("cpu", "gpu", "tpu"), default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=4)
    return parser.parse_args()


ARGS = _parse_args()
if ARGS.device:
    os.environ["JAX_PLATFORM_NAME"] = ARGS.device

try:
    import jax
    import jax.numpy as jnp

    from examples.benchmark_utils import time_callable
    from jaccpot import FastMultipoleMethod
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing runtime dependency. Install jaccpot deps (notably yggdrax) "
        f"before running this benchmark. Original error: {exc}"
    ) from exc


def _parse_gears(gears: str) -> tuple[int, ...]:
    parsed = tuple(int(v.strip()) for v in gears.split(",") if v.strip())
    if len(parsed) == 0:
        raise ValueError("p_gears cannot be empty")
    return parsed


def main() -> None:
    if ARGS.dtype == "float64" and not jax.config.jax_enable_x64:
        raise SystemExit("float64 requested, but JAX x64 is disabled")

    dtype = jnp.float64 if ARGS.dtype == "float64" else jnp.float32
    key = jax.random.PRNGKey(0)
    key_pos, key_mass, key_target = jax.random.split(key, 3)
    n = int(ARGS.n)
    num_targets = max(1, int(float(ARGS.target_frac) * n))
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        dtype=dtype,
        minval=jnp.asarray(-1.0, dtype=dtype),
        maxval=jnp.asarray(1.0, dtype=dtype),
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )
    target_indices = jax.random.choice(
        key_target,
        n,
        shape=(num_targets,),
        replace=False,
    )
    p_gears = _parse_gears(str(ARGS.p_gears))

    base = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=float(ARGS.theta),
        softening=1.0e-3,
    )
    state = base.prepare_state(
        positions,
        masses,
        leaf_size=int(ARGS.leaf_size),
        max_order=int(ARGS.p),
    )
    target_eval = time_callable(
        base.evaluate_prepared_state,
        state,
        target_indices=target_indices,
        return_potential=False,
        warmup=int(ARGS.warmup),
        runs=int(ARGS.runs),
    )

    adaptive = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=float(ARGS.theta),
        softening=1.0e-3,
        adaptive_order=True,
        p_gears=p_gears,
        mac_force_scale_mode="prev",
        adaptive_error_model="tail_proxy",
    )
    adaptive_prepare = time_callable(
        adaptive.prepare_state,
        positions,
        masses,
        leaf_size=int(ARGS.leaf_size),
        max_order=int(ARGS.p),
        warmup=int(ARGS.warmup),
        runs=int(ARGS.runs),
    )

    print(
        f"device={jax.devices()[0]} dtype={ARGS.dtype} n={n} p={ARGS.p} "
        f"leaf_size={ARGS.leaf_size} theta={ARGS.theta:.3f} num_targets={num_targets}"
    )
    print(
        "timings_s "
        f"target_eval_mean={target_eval.mean:.6f} "
        f"target_eval_std={target_eval.std:.6f} "
        f"adaptive_prepare_mean={adaptive_prepare.mean:.6f} "
        f"adaptive_prepare_std={adaptive_prepare.std:.6f}"
    )
    print("adaptive_p_gears " + ",".join(str(int(v)) for v in p_gears))


if __name__ == "__main__":
    main()
