"""Forward vs. forward+backward wall-clock ratio vs. N (and vs. #GPUs).

Uses grad_bench_lib.py (this directory). Wrap jaccpot/runtime/fmm_evaluate.py's
single-device entry point for the N-sweep panel, and bench/multigpu/harness.py's
distributed driver for a secondary panel vs #GPUs at fixed N. Writes
results/differentiability/autodiff_overhead.json:
{"n": [...], "n_gpus": [...], "fwd_s": [...], "fwd_bwd_s": [...]}
(n_gpus=1 rows cover the single-device N sweep.)
"""

from __future__ import annotations

import argparse
import json
import pathlib

from grad_bench_lib import time_forward_and_backward

RESULTS_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "results" / "differentiability"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-min-exp", type=int, default=11)
    p.add_argument("--n-max-exp", type=int, default=20)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--gpu-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # TODO: import jaccpot.runtime.fmm_evaluate's forward call, sweep N, call
    # time_forward_and_backward; repeat vs n_gpus at fixed N using
    # bench/multigpu/harness.py's distributed driver.
    raise NotImplementedError("Wire up jaccpot/runtime/fmm_evaluate.py's forward call.")


if __name__ == "__main__":
    main()
