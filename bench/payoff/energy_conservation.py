"""Energy/angular-momentum conservation over a long integration.

Check nornax's integrator examples first -- this is a standard N-body
sanity test that may already exist there (Hermite integrator suite built on
jaccpot forces). Only write new code if nornax has nothing suitable. Writes
results/payoff/energy_conservation.json:
{"step": [...], "time": [...], "energy": [...], "lz": [...]}
"""

from __future__ import annotations

import argparse
import json
import pathlib

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "results" / "payoff"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--ic", choices=("plummer", "cold_collapse"), default="plummer")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # TODO: check nornax examples/tests for an existing long-integration
    # energy-conservation test before writing this from scratch.
    raise NotImplementedError(
        "Check nornax repo for existing energy-conservation test first."
    )


if __name__ == "__main__":
    main()
