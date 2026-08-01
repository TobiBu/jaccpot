"""Finite-difference vs. autodiff gradient agreement vs. theta.

Uses grad_bench_lib.py (this directory). Sweep theta with a small N (finite
differences are expensive -- see grad_bench_lib.py's docstring caveat).
Writes results/differentiability/grad_correctness.json:
{"thetas": [...], "mean_rel_grad_error": [...], "max_rel_grad_error": [...]}
-- quantifies the MAC-boundary discontinuity noted in
docs/multigpu_differentiability_model.md.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from grad_bench_lib import finite_difference_check

RESULTS_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "results" / "differentiability"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n", type=int, default=2_000, help="keep small -- FD is O(N) forward evals"
    )
    p.add_argument(
        "--thetas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    )
    p.add_argument("--n-subsample", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # TODO: for each theta, run finite_difference_check on a random
    # subsample of particle positions vs. jax.grad of the same loss.
    raise NotImplementedError("Wire up jaccpot/runtime/fmm_evaluate.py's forward call.")


if __name__ == "__main__":
    main()
