"""Multi-GPU weak scaling: N scaled with #GPUs.

Uses harness.py (this directory). Writes results/multigpu/weak_scaling.json:
{"n": [...], "n_gpus": [...], "wall_clock_total_s": [...]}
Throughput-per-GPU (n / wall_clock / n_gpus) is computed in the notebook.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from harness import weak_scaling

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results" / "multigpu"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-per-gpu", type=int, default=1_000_000)
    p.add_argument("--gpu-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results = weak_scaling(args.n_per_gpu, args.gpu_counts)
    payload = {
        "n": [r.n_particles for r in results],
        "n_gpus": [r.n_gpus for r in results],
        "wall_clock_total_s": [r.wall_clock_total for r in results],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "weak_scaling.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {RESULTS_DIR / 'weak_scaling.json'}")


if __name__ == "__main__":
    main()
