"""Loss/parameter-error convergence curve for gradient-based potential
recovery -- the paper's payoff experiment.

DEFERRED. This targets parametric potential recovery from kinematics, which is
out of scope for Paper I; the jaccpot/applications/potential_recovery/ package it
was written against has been deleted rather than retargeted. Nothing here runs
until that case is picked up in a later paper. Writes bench/results/payoff/recovery.json:
{"iteration": [...], "loss": [...], "param_rel_error": [...]}
"""

from __future__ import annotations

import argparse
import json
import pathlib

RESULTS_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "bench" / "results" / "payoff"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--n-iterations", type=int, default=500)
    p.add_argument(
        "--method", choices=("hmc", "vi", "grad_descent"), default="grad_descent"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    raise NotImplementedError(
        "Deferred: parametric potential recovery is out of scope for Paper I, and "
        "jaccpot/applications/potential_recovery/ has been deleted."
    )


if __name__ == "__main__":
    main()
