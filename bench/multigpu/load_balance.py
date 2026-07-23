"""Per-GPU work for a clustered (non-uniform) particle distribution.

Uses harness.py's per_gpu_interaction_counts (this directory). Use a
Plummer-sphere-like or NFW-like clustered distribution rather than
uniform_cube -- this is specifically the case where naive
space-filling-curve partitioning can go unbalanced. Writes
results/multigpu/load_balance.json:
{"gpu_index": [...], "interaction_count": [...]}
"""

from __future__ import annotations

import argparse
import json
import pathlib

from harness import run_once

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results" / "multigpu"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10_000_000)
    p.add_argument("--n-gpus", type=int, default=8)
    p.add_argument(
        "--distribution", choices=("uniform_cube", "plummer", "nfw"), default="plummer"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_once(args.n, args.n_gpus, distribution=args.distribution)
    payload = {
        "gpu_index": list(range(len(result.per_gpu_interaction_counts))),
        "interaction_count": [int(c) for c in result.per_gpu_interaction_counts],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "load_balance.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {RESULTS_DIR / 'load_balance.json'}")


if __name__ == "__main__":
    main()
