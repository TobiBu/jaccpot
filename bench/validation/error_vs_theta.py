"""Force error vs. opening angle theta (Dehnen-2014-style curve).

Reuse theta sweep parameters from docs/treecode_mac_stability.md (repo root)
so this is directly comparable to the engineering validation already done
there. Writes results/validation/error_vs_theta.json:
{"thetas": [...], "mean_rel_error": [...], "rms_rel_error": [...], "max_rel_error": [...]}
"""

from __future__ import annotations

import argparse
import json
import pathlib

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results" / "validation"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument(
        "--thetas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # TODO: sweep theta, collect relative-error distribution stats.
    raise NotImplementedError("Pull theta grid from docs/treecode_mac_stability.md.")


if __name__ == "__main__":
    main()
