"""SUMMARY.md for a MESC job directory: per row the step and edge times, escapers, ledger, peaks per card.

python bench/slurm/mesc/summarize.py $H0_ROOT/mesc/<run id>/node1 > SUMMARY.md
"""

from __future__ import annotations

import glob
import json
import os
import sys


def main(out: str) -> None:
    print(f"# MESC {os.path.basename(out)}: {out}\n")
    print(
        "| row | GPUs | N | step s | edge s | escapers first -> last | max abs ledgered dE | "
        "max momentum drift | retries | peak GiB per card (max) | peak B/p per card (max) |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row_dir in sorted(glob.glob(os.path.join(out, "*", ""))):
        files = sorted(glob.glob(os.path.join(row_dir, "result_p*.json")))
        if not files:
            continue
        rows = [json.load(open(f)) for f in files]
        r = rows[0]
        diags = [e["diag"] or {} for e in r["edges"]]
        counts = [d.get("n_escapers") for d in diags]
        ledgered = [
            d.get("energy_ledgered_drift", d.get("energy_drift")) for d in diags
        ]
        ledgered = [abs(v) for v in ledgered if v is not None]
        mom = [
            d.get("momentum_drift")
            for d in diags
            if d.get("momentum_drift") is not None
        ]
        peaks = [v for x in rows for v in x["peak_bytes"].values()]
        bpp = [v for x in rows for v in x["peak_bytes_per_particle"].values()]
        print(
            f"| {os.path.basename(os.path.dirname(row_dir))} | {r['ndev']} | {r['n']} | "
            f"{r['step_s']} | {r['edge_s']} | {counts[0]} -> {counts[-1]} | "
            f"{max(ledgered) if ledgered else None} | {max(mom) if mom else None} | "
            f"{[e['retries'] for e in r['edges']]} | {max(peaks) / 2**30:.2f} | {max(bpp):.1f} |"
        )


if __name__ == "__main__":
    main(sys.argv[1])
