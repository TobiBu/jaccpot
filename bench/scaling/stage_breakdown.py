"""Stacked time breakdown (build / M2M / M2L / L2L / P2P / autodiff) vs. N.

Aggregates bench/profile_refresh_stage_breakdown.py and
bench/profile_downward_breakdown.py (repo root) across an N sweep. Writes
results/scaling/stage_breakdown.json: {"n": [...], "stage": [...], "time_s": [...]}
(long format -- the notebook pivots into a stacked bar).
"""

from __future__ import annotations

import argparse
import json
import pathlib

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results" / "scaling"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # TODO: for each N, run profile_refresh_stage_breakdown.py /
    # profile_downward_breakdown.py, collect per-stage wall time.
    raise NotImplementedError(
        "Aggregate bench/profile_refresh_stage_breakdown.py and "
        "bench/profile_downward_breakdown.py output."
    )


if __name__ == "__main__":
    main()
