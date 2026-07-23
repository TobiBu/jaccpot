"""Wall-clock vs. N, log-log -- jaccpot vs. direct sum vs. literature.

Calls bench/bench_jaxfmm_paper_compare.py (repo root), which already
reproduces jaxFMM-paper timing settings and compares against jaccpot across
N=2**11..2**25. Writes results/scaling/wallclock.json:
{"n": [...], "runner": [...], "wall_clock_s": [...]}
(runner in {"jaxfmm", "jaccpot", "direct_sum", "literature_dehnen2014"})
"""

from __future__ import annotations

import argparse
import json
import pathlib

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results" / "scaling"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-min-exp", type=int, default=11)
    p.add_argument("--n-max-exp", type=int, default=24)
    p.add_argument("--n-steps", type=int, default=14)
    p.add_argument("--runner", choices=("jaxfmm", "jaccpot", "both"), default="both")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # TODO: invoke bench/bench_jaxfmm_paper_compare.py programmatically or
    # shell out and parse its CSV, reshape into the JSON schema above. Add a
    # direct-summation O(N^2) reference series and, if available, digitized
    # Dehnen (2014) timing points.
    raise NotImplementedError("Wire up bench/bench_jaxfmm_paper_compare.py's output.")


if __name__ == "__main__":
    main()
