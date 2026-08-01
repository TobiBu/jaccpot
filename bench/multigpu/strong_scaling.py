"""Multi-GPU strong scaling: fixed N, wall-clock vs. #GPUs.

Uses harness.py (this directory). Writes results/multigpu/strong_scaling.json:
{"n": [...], "n_gpus": [...], "wall_clock_total_s": [...]}
"""

from __future__ import annotations

import argparse
import json
import pathlib

from harness import strong_scaling

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results" / "multigpu"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10_000_000)
    p.add_argument("--gpu-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results = strong_scaling(args.n, args.gpu_counts)
    payload = {
        "n": [r.n_particles for r in results],
        "n_gpus": [r.n_gpus for r in results],
        "wall_clock_total_s": [r.wall_clock_total for r in results],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "strong_scaling.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {RESULTS_DIR / 'strong_scaling.json'}")


if __name__ == "__main__":
    main()
