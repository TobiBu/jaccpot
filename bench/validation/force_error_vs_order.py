"""Force error vs. direct summation, across expansion orders p.

Adapt bench/bench_real_vs_complex.py and bench/bench_fmm.py (repo root) --
both already compute forces at a given order against a direct summation
oracle. Writes results/validation/force_error_vs_order.json:
{"orders": [...], "mean_rel_error": [...], "rms_rel_error": [...], "max_rel_error": [...]}
No plotting logic belongs in this file.
"""

from __future__ import annotations

import argparse
import json
import pathlib

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results" / "validation"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # TODO: for order in args.orders: run jaccpot FMM at fixed theta, compare
    # against direct summation (see bench/bench_fmm.py's oracle), collect
    # relative force error (mean/rms/max) with a fixed seed.
    raise NotImplementedError(
        "Wire this up against bench/bench_fmm.py's direct-summation oracle."
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "force_error_vs_order.json", "w") as f:
        json.dump({}, f, indent=2)


if __name__ == "__main__":
    main()
