"""Single-GPU vs. CPU speedup vs. N.

Calls bench/bench_jaxfmm_paper_compare.py --device cpu and --device gpu
(repo root) across the same N grid. Writes
results/scaling/gpu_vs_cpu_speedup.json: {"n": [...], "device": [...], "wall_clock_s": [...]}
-- ratio computed in the notebook, not here.
"""

from __future__ import annotations

import argparse
import json
import pathlib

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results" / "scaling"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-min-exp", type=int, default=11)
    p.add_argument("--n-max-exp", type=int, default=22)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # TODO: run bench_jaxfmm_paper_compare.py --device cpu and --device gpu.
    raise NotImplementedError(
        "Run bench_jaxfmm_paper_compare.py at --device cpu and gpu."
    )


if __name__ == "__main__":
    main()
