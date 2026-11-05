"""SUMMARY.md for an M2G job directory: per arm the walls, retries and peaks; every compare verdict.

python bench/slurm/m2g/summarize.py $H0_ROOT/m2g/<job id>/node1 > SUMMARY.md
"""

from __future__ import annotations

import glob
import json
import os
import sys


def main(out: str) -> None:
    print(f"# M2G {os.path.basename(out)}: {out}\n")
    print(
        "| arm | processes | N | repeat walls (s) | warm per-edge walls (s) | retries | peak GiB (process 0) |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for path in sorted(glob.glob(os.path.join(out, "*", "result_p0.json"))):
        arm = os.path.basename(os.path.dirname(path))
        with open(path) as fh:
            r = json.load(fh)
        walls = [round(x["wall_s"], 2) for x in r["runs"]]
        warm = r["runs"][-1]
        edges = [
            round(e["wall_s"], 2) for e in warm["edges"] if e.get("wall_s") is not None
        ]
        peaks = [round(b / 2**30, 2) for b in r.get("peak_bytes", [])]
        print(
            f"| {arm} | {r['processes']} | {r['num_particles']:.3g} | {walls} | {edges} | "
            f"{warm['retries']} | {peaks} |"
        )
    print(
        "\n## Compares (bitwise by id: state, acceleration, owner; edges' diagnostics)\n"
    )
    for path in sorted(glob.glob(os.path.join(out, "compare_*.json"))):
        with open(path) as fh:
            c = json.load(fh)
        arrays = {
            k: (v["differ"] if isinstance(v, dict) else v)
            for k, v in c["arrays"].items()
        }
        print(
            f"- {os.path.basename(path)[8:-5]}: bitwise **{c['bitwise']}**, differing {arrays}, "
            f"edges equal {c['edges_equal']}"
        )
    print("\n## Collectives\n")
    for path in sorted(glob.glob(os.path.join(out, "*", "collectives.json"))):
        with open(path) as fh:
            c = json.load(fh)
        oks = {k: v for k, v in c.items() if k.endswith("_ok")}
        order = {k: v for k, v in c.items() if "differs" in k or "rows_wrong" in k}
        print(
            f"- {os.path.basename(os.path.dirname(path))} ({c.get('process_count')} processes): "
            f"{oks}; {order}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
