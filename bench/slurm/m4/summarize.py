"""SUMMARY.md of an M4 job: per arm the step wall, the LET rows vs all_gather's, the work imbalance.

python bench/slurm/m4/summarize.py OUT_DIR > OUT_DIR/SUMMARY.md
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def _f(x, fmt="{:.3g}"):
    return "-" if x is None else fmt.format(x)


def main(out: str) -> None:
    root = Path(out)
    print(f"# M4 job `{root.name}`\n")
    print(
        "| arm | devices | N | exchange | cuts | step wall s (median) | LET rows recv (max) | all_gather rows | "
        "imbalance before -> after (last rebalance) | retries | momentum total end |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for res in sorted(root.glob("*/result.json")):
        r = json.loads(res.read_text())
        edges = [e for e in r.get("edges", []) if e.get("wall_s") is not None]
        steps = max(int(r.get("steps", 1)), 1)
        walls = [e["wall_s"] / steps for e in edges[1:]] or [
            e["wall_s"] / steps for e in edges
        ]
        segs = r.get("segments_report", [])
        let_rows = max((s.get("let_recv_rows_max") or 0 for s in segs), default=None)
        ag = next(
            (s.get("all_gather_rows") for s in segs if s.get("all_gather_rows")), None
        )
        reb = [s for s in segs if s.get("imbalance_before") is not None]
        imb = (
            f"{reb[-1]['imbalance_before']:.3f} -> {reb[-1]['imbalance_after']:.3f}"
            if reb
            else "-"
        )
        print(
            f"| {res.parent.name} | {r.get('devices')} | {r.get('num_particles'):.3g} | {r.get('exchange')} | "
            f"{r.get('cuts')} | {_f(statistics.median(walls) if walls else None)} | "
            f"{let_rows if r.get('exchange') == 'let' else '-'} | {ag} | {imb} | "
            f"{sum(int(s.get('retries') or 0) for s in segs)} | "
            f"{_f((r.get('split_end') or {}).get('total'), '{:.1e}')} |"
        )


if __name__ == "__main__":
    main(sys.argv[1])
