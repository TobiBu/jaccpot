"""Interaction count at matched accuracy: fixed-theta vs. mass-dependent MAC.

Adapt jaccpot/runtime/fmm_policy.py and _adaptive_policy.py (repo root),
which already implement both MAC variants. Writes
results/validation/mac_comparison.json:
{"n": [...], "mac_type": [...], "interaction_count": [...], "achieved_rel_error": [...]}
"""

from __future__ import annotations

import argparse
import json
import pathlib

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results" / "validation"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    p.add_argument("--target-rel-error", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # TODO: for each N, binary-search theta (fixed MAC) and the mass-dependent
    # MAC's own parameter to hit target_rel_error; record interaction counts.
    raise NotImplementedError(
        "Needs the accuracy-matching search; see jaccpot/runtime/fmm_policy.py."
    )


if __name__ == "__main__":
    main()
