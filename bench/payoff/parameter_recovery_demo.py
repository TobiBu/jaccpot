"""Loss/parameter-error convergence curve for gradient-based potential
recovery -- the paper's payoff experiment.

Depends on jaccpot/applications/potential_recovery/ (this branch, Phase 4).
Sequence this last, once N and theta operating points are known from
Phases 1-3. Writes results/payoff/recovery.json:
{"iteration": [...], "loss": [...], "param_rel_error": [...]}
"""

from __future__ import annotations

import argparse
import json
import pathlib

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results" / "payoff"


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
    # TODO: depends on jaccpot/applications/potential_recovery/recover.py.
    raise NotImplementedError(
        "Depends on jaccpot/applications/potential_recovery/ (see PROJECT_PLAN.md Phase 4)."
    )


if __name__ == "__main__":
    main()
