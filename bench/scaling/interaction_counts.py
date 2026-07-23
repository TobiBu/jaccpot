"""M2L and P2P interaction counts vs. N.

Adapt jaccpot/runtime/fmm_diagnostics.py (repo root); bench/bench_fmm.py is
the natural driver to sweep N. Writes results/scaling/interaction_counts.json:
{"n": [...], "m2l_count": [...], "p2p_count": [...]}
Power-law exponent fitting belongs in the notebook, not here.
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
    p.add_argument("--n-steps", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # TODO: sweep N, pull M2L/P2P counts from fmm_diagnostics.py.
    raise NotImplementedError("Wire up jaccpot/runtime/fmm_diagnostics.py.")


if __name__ == "__main__":
    main()
