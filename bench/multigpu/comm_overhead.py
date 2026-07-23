"""Communication vs. compute fraction vs. #GPUs.

Reuses harness.py's per-stage timers (this directory). Writes
results/multigpu/comm_overhead.json:
{"n": [...], "n_gpus": [...], "stage": [...], "time_s": [...]}
(long format -- the notebook sums by comm/compute category using
harness.COMM_STAGES, so the category split stays a plotting choice, not
baked into the data file.)
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
    payload = {"n": [], "n_gpus": [], "stage": [], "time_s": []}
    for r in results:
        for stage, t in r.stage_times.items():
            payload["n"].append(r.n_particles)
            payload["n_gpus"].append(r.n_gpus)
            payload["stage"].append(stage)
            payload["time_s"].append(t)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "comm_overhead.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {RESULTS_DIR / 'comm_overhead.json'}")


if __name__ == "__main__":
    main()
