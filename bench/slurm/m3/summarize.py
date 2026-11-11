"""M3's HoreKa steps as one Markdown table: python bench/slurm/m3/summarize.py <job dir>."""

from __future__ import annotations

import glob
import json
import os
import sys


def main(root: str) -> None:
    print(f"# M3 checkpoint I/O: {root}\n")
    print("| step | rank/procs | devices | N | write s (reps) | restore s (reps) | shard MB | disk MB | RSS MB | dev peak GB | layout/status |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for path in sorted(glob.glob(os.path.join(root, "*", "result_*_p*.json"))):
        r = json.load(open(path))
        step = os.path.basename(os.path.dirname(path))
        writes = [f"{w['wall_s']:.2f}" for w in r.get("writes", [])]
        restores = [f"{w['wall_s']:.2f}" for w in r.get("restores", [])]
        shard = r["writes"][-1]["bytes_local"] / 2**20 if r.get("writes") else None
        disk = r.get("disk_bytes_this_process")
        state = (r.get("restored") or {}).get("layout") or r.get("status") or ""
        peak = max(r.get("device_peak_bytes") or [0]) / 2**30
        print(
            f"| {step} | {r['process']}/{r['processes']} | {r['devices']} | {r.get('n')} | "
            f"{' '.join(writes)} | {' '.join(restores)} | "
            f"{'' if shard is None else f'{shard:.0f}'} | {'' if disk is None else f'{disk / 2**20:.0f}'} | "
            f"{r['peak_rss_mb']:.0f} | {peak:.1f} | {state} |"
        )


if __name__ == "__main__":
    main(sys.argv[1])
