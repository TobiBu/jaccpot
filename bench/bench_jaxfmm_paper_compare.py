"""Reproduce jaxFMM paper timing settings and compare against jaccpot."""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runner",
        choices=("jaxfmm", "jaccpot", "both"),
        default="both",
        help="Which implementation(s) to benchmark.",
    )
    parser.add_argument(
        "--distribution",
        choices=("uniform_cube", "sphere_surface", "normal"),
        default="uniform_cube",
        help="Particle distribution.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu", "tpu"),
        default=None,
        help="Optional JAX platform override.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Array dtype.",
    )
    parser.add_argument(
        "--n-min-exp",
        type=int,
        default=11,
        help="Minimum exponent for particle count (N=2**exp).",
    )
    parser.add_argument(
        "--n-max-exp",
        type=int,
        default=25,
        help="Maximum exponent for particle count (N=2**exp).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=14,
        help="Number of log-even particle counts between min/max exponents.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Number of timed repetitions (paper uses 100).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup repetitions before timing.",
    )
    parser.add_argument(
        "--leaf-size",
        type=int,
        default=128,
        help="Max particles per leaf for jaccpot (paper uses N_max=128 in jaxFMM).",
    )
    parser.add_argument(
        "--param-sets",
        choices=("all", "1", "2", "3", "4"),
        default="all",
        help="Run all paper parameter sets or just one.",
    )
    parser.add_argument(
        "--custom-p",
        type=int,
        default=None,
        help="Custom expansion order p (overrides --param-sets when set together with --custom-theta/--custom-s).",
    )
    parser.add_argument(
        "--custom-theta",
        type=float,
        default=None,
        help="Custom theta (overrides --param-sets when set together with --custom-p/--custom-s).",
    )
    parser.add_argument(
        "--custom-s",
        type=int,
        default=None,
        help="Custom split parameter s (overrides --param-sets when set together with --custom-p/--custom-theta).",
    )
    parser.add_argument(
        "--basis",
        choices=("real", "solidfmm", "complex"),
        default="real",
        help="jaccpot basis to benchmark.",
    )
    parser.add_argument(
        "--preset",
        choices=("fast", "balanced", "accurate", "large_n_gpu"),
        default="accurate",
        help="jaccpot preset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


ARGS = _parse_args()
if ARGS.device:
    os.environ["JAX_PLATFORM_NAME"] = ARGS.device

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

JAXFMM_ROOT = REPO_ROOT / "external" / "jaxfmm"
if JAXFMM_ROOT.exists() and str(JAXFMM_ROOT) not in sys.path:
    sys.path.insert(0, str(JAXFMM_ROOT))

import jax
import jax.numpy as jnp

from jaccpot import FastMultipoleMethod

try:
    from jaxfmm.fmm import eval_potential
    from jaxfmm.hierarchy import gen_hierarchy

    HAVE_JAXFMM = True
except Exception:
    HAVE_JAXFMM = False


@dataclass(frozen=True)
class ParamSet:
    name: str
    p: int
    theta: float
    s: int


PAPER_PARAM_SETS = (
    ParamSet("1_default", p=3, theta=0.77, s=3),
    ParamSet("2_high_p", p=6, theta=0.77, s=3),
    ParamSet("3_low_theta", p=3, theta=0.50, s=3),
    ParamSet("4_low_s", p=3, theta=0.77, s=2),
)


def _block_until_ready(value: Any) -> Any:
    def _maybe_block(x: Any) -> Any:
        if hasattr(x, "block_until_ready"):
            return x.block_until_ready()
        return x

    return jax.tree_util.tree_map(_maybe_block, value)


def _time_min_repeat(
    fn: Callable[[], Any], *, warmup: int, repeats: int
) -> tuple[float, float, float]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    for _ in range(warmup):
        _block_until_ready(fn())

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        _block_until_ready(out)
        end = time.perf_counter()
        samples.append(end - start)

    return (
        float(min(samples)),
        float(statistics.mean(samples)),
        float(statistics.pstdev(samples)),
    )


def _n_values(min_exp: int, max_exp: int, steps: int) -> list[int]:
    if steps <= 1:
        return [int(round(2 ** float(min_exp)))]
    xs = jnp.linspace(float(min_exp), float(max_exp), steps)
    vals = [int(round(2 ** float(x))) for x in xs]
    out: list[int] = []
    seen = set()
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _distribution(
    key: jax.Array, n: int, name: str, dtype: jnp.dtype
) -> tuple[jax.Array, jax.Array]:
    k1, k2 = jax.random.split(key)
    if name == "uniform_cube":
        pts = jax.random.uniform(k1, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype)
    elif name == "sphere_surface":
        raw = jax.random.normal(k1, (n, 3), dtype=dtype)
        norm = jnp.linalg.norm(raw, axis=1, keepdims=True)
        pts = raw / jnp.maximum(norm, jnp.asarray(1e-12, dtype=dtype))
    elif name == "normal":
        pts = jax.random.normal(k1, (n, 3), dtype=dtype)
    else:
        raise ValueError(f"unknown distribution: {name}")
    charges = jax.random.uniform(k2, (n,), minval=0.0, maxval=1.0, dtype=dtype)
    return pts, charges


def _run_jaxfmm_case(
    *,
    points: jax.Array,
    charges: jax.Array,
    p: int,
    theta: float,
    s: int,
    repeats: int,
    warmup: int,
) -> tuple[float, float, float]:
    hier = gen_hierarchy(points, p=p, theta=theta, s=s, N_max=128)
    fn = lambda: eval_potential(charges, **hier)
    return _time_min_repeat(fn, warmup=warmup, repeats=repeats)


def _run_jaccpot_case(
    *,
    points: jax.Array,
    masses: jax.Array,
    p: int,
    theta: float,
    repeats: int,
    warmup: int,
    leaf_size: int,
    basis: str,
    preset: str,
) -> tuple[float, float, float]:
    solver = FastMultipoleMethod(preset=preset, basis=basis)
    prepared = solver.prepare_state(
        points,
        masses,
        leaf_size=leaf_size,
        max_order=p,
        theta=float(theta),
    )
    fn = lambda: solver.evaluate_prepared_state(prepared, return_potential=True)[1]
    return _time_min_repeat(fn, warmup=warmup, repeats=repeats)


def main() -> None:
    dtype = jnp.float32 if ARGS.dtype == "float32" else jnp.float64
    ns = _n_values(ARGS.n_min_exp, ARGS.n_max_exp, ARGS.n_steps)
    key = jax.random.PRNGKey(int(ARGS.seed))

    custom_args = (ARGS.custom_p, ARGS.custom_theta, ARGS.custom_s)
    using_custom = any(v is not None for v in custom_args)
    if using_custom and not all(v is not None for v in custom_args):
        raise SystemExit(
            "Custom parameter mode requires all of --custom-p, --custom-theta, and --custom-s."
        )

    selected = PAPER_PARAM_SETS
    if using_custom:
        selected = (
            ParamSet(
                "custom",
                p=int(ARGS.custom_p),
                theta=float(ARGS.custom_theta),
                s=int(ARGS.custom_s),
            ),
        )
    elif ARGS.param_sets != "all":
        idx = int(ARGS.param_sets) - 1
        selected = (PAPER_PARAM_SETS[idx],)

    rows: list[dict[str, Any]] = []

    for n in ns:
        key, nkey = jax.random.split(key)
        points, charges = _distribution(nkey, n, ARGS.distribution, dtype)
        for ps in selected:
            if ARGS.runner in ("jaxfmm", "both"):
                row = {
                    "runner": "jaxfmm",
                    "distribution": ARGS.distribution,
                    "n": n,
                    "param_set": ps.name,
                    "p": ps.p,
                    "theta": ps.theta,
                    "s": ps.s,
                    "repeats": ARGS.repeats,
                    "warmup": ARGS.warmup,
                    "status": "ok",
                    "note": "",
                }
                try:
                    if not HAVE_JAXFMM:
                        raise RuntimeError(
                            "jaxfmm import failed. Install it or place package in external/jaxfmm."
                        )
                    tmin, tmean, tstd = _run_jaxfmm_case(
                        points=points,
                        charges=charges,
                        p=ps.p,
                        theta=ps.theta,
                        s=ps.s,
                        repeats=ARGS.repeats,
                        warmup=ARGS.warmup,
                    )
                    row.update(
                        {
                            "min_seconds": tmin,
                            "mean_seconds": tmean,
                            "std_seconds": tstd,
                        }
                    )
                except Exception as exc:
                    row.update(
                        {
                            "status": "error",
                            "note": str(exc).replace("\n", " "),
                            "min_seconds": math.nan,
                            "mean_seconds": math.nan,
                            "std_seconds": math.nan,
                        }
                    )
                rows.append(row)
                print(
                    f"[jaxfmm] n={n} set={ps.name} p={ps.p} theta={ps.theta} s={ps.s} "
                    f"status={row['status']} min={row['min_seconds']}"
                )

            if ARGS.runner in ("jaccpot", "both"):
                row = {
                    "runner": "jaccpot",
                    "distribution": ARGS.distribution,
                    "n": n,
                    "param_set": ps.name,
                    "p": ps.p,
                    "theta": ps.theta,
                    "s": ps.s,
                    "leaf_size": ARGS.leaf_size,
                    "basis": ARGS.basis,
                    "preset": ARGS.preset,
                    "repeats": ARGS.repeats,
                    "warmup": ARGS.warmup,
                    "status": "ok",
                    "note": "",
                }
                try:
                    if ps.s != 3:
                        raise RuntimeError(
                            "paper set 4 changes split parameter s, which has no direct jaccpot control; skipped."
                        )
                    tmin, tmean, tstd = _run_jaccpot_case(
                        points=points,
                        masses=charges,
                        p=ps.p,
                        theta=ps.theta,
                        repeats=ARGS.repeats,
                        warmup=ARGS.warmup,
                        leaf_size=ARGS.leaf_size,
                        basis=ARGS.basis,
                        preset=ARGS.preset,
                    )
                    row.update(
                        {
                            "min_seconds": tmin,
                            "mean_seconds": tmean,
                            "std_seconds": tstd,
                        }
                    )
                except Exception as exc:
                    row.update(
                        {
                            "status": "error",
                            "note": str(exc).replace("\n", " "),
                            "min_seconds": math.nan,
                            "mean_seconds": math.nan,
                            "std_seconds": math.nan,
                        }
                    )
                rows.append(row)
                print(
                    f"[jaccpot] n={n} set={ps.name} p={ps.p} theta={ps.theta} "
                    f"status={row['status']} min={row['min_seconds']}"
                )

    if ARGS.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = REPO_ROOT / "benchmarks" / f"jaxfmm_paper_compare_{stamp}.csv"
    else:
        out = ARGS.output
        if not out.is_absolute():
            out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to: {out}")


if __name__ == "__main__":
    main()
