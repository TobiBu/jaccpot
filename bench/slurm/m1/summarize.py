"""SUMMARY.md for an M1 job directory: per arm the momentum split and drift, walls, peaks.

python bench/slurm/m1/summarize.py $H0_ROOT/m1/<run id>/node1 > SUMMARY.md
"""

from __future__ import annotations

import glob
import json
import os
import sys


def _fmt(x: float) -> str:
    return f"{x:.2e}"


def main(out: str) -> None:
    print(f"# M1 {os.path.basename(out)}: {out}\n")
    print(
        "| arm | processes | N | MAC | start: total / local / cross (share) "
        "| end: total / local / cross (share) | max momentum_drift | walls per edge (s) "
        "| retries | peak GiB (process 0) |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for path in sorted(glob.glob(os.path.join(out, "*", "result.json"))):
        arm = os.path.basename(os.path.dirname(path))
        with open(path) as fh:
            r = json.load(fh)
        cells = []
        for key in ("split_start", "split_end"):
            s = r[key]
            cells.append(
                f"{_fmt(s['total'])} / {_fmt(s['local_part'])} / {_fmt(s['cross_part'])}"
                f" ({s['cross_share']:.2f})"
            )
        drift = [
            e["momentum_drift"]
            for e in r["edges"]
            if e.get("momentum_drift") is not None
        ]
        walls = [
            round(e["wall_s"], 2) for e in r["edges"] if e.get("wall_s") is not None
        ]
        retries = sum(int(e.get("retries") or 0) for e in r["edges"])
        peaks = [round(b / 2**30, 2) for b in r.get("peak_bytes", [])]
        print(
            f"| {arm} | {r['processes']} | {r['num_particles']:.2e} | {r['mac']} | {cells[0]}"
            f" | {cells[1]} | {_fmt(max(drift)) if drift else '-'} | {walls} | {retries}"
            f" | {peaks} |"
        )


if __name__ == "__main__":
    main(sys.argv[1])
