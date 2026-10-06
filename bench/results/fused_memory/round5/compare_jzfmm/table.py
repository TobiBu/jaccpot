"""Markdown table of the jz-fmm vs jaccpot rows in rows/ (one block per N)."""

import glob
import json
import os
import sys

O = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(os.path.dirname(__file__), "rows")
)
G = 2**30
CFG = {
    "jzA": "jz-fmm leaf32 p4 θ0.6",
    "jzB": "jz-fmm leaf32 p5 θ0.8",
    "jzC": "jz-fmm leaf32 p4 θ0.7",
    "jzD": "jz-fmm leaf32 p5 θ0.9",
    "jzE": "jz-fmm leaf32 p6 θ0.9",
    "ouA": "jaccpot p5 θ0.8 cml8 (default)",
    "ouB": "jaccpot p5 θ0.8 cml6",
    "ouC": "jaccpot p5 θ0.7 cml8",
    "ouD": "jaccpot p4 θ0.8 cml8",
    "ouE": "jaccpot p6 θ0.8 cml6",
    "ouF": "jaccpot p5 θ0.7 cml6",
    "ouG": "jaccpot p5 θ0.8 cml5",
    "ouH": "jaccpot p5 θ0.8 cml4",
}
for n in (8000000, 32000000, 100000000):
    rows = []
    for k in ("jzA", "jzB", "jzC", "jzD", "jzE"):
        p = f"{O}/{k}_{n}.json"
        if not os.path.exists(p):
            continue
        r = json.load(open(p))["rows"][0]
        if "timing" not in r:
            rows.append((CFG[k], "failed", "", "", "", ""))
            continue
        rows.append(
            (
                CFG[k],
                f"{1e3*r['timing']['min']:.1f}",
                "(force incl. tree)",
                f"{r['error']['aggL2']:.2e}",
                f"{r['error']['p90']:.2e}",
                f"{r['memory']['peak_gib']:.2f} ({r['memory']['peak_bytes_per_particle']:.0f})",
            )
        )
    for k in ("ouA", "ouB", "ouC", "ouD", "ouE", "ouF", "ouG", "ouH"):
        f, s = f"{O}/{k}_{n}_f.json", f"{O}/{k}_{n}_s.json"
        if not (os.path.exists(f) or os.path.exists(s)):
            continue
        df = json.load(open(f)) if os.path.exists(f) else {}
        ds = json.load(open(s)) if os.path.exists(s) else {}
        acc = df.get("accuracy", {})
        step = f"{ds['step_ms']['min']:.1f}" if "step_ms" in ds else "failed"
        ev = f"{df['eval_ms']['min']:.1f}" if "eval_ms" in df else "-"
        pk = ds.get("peak_after_scan_gib")
        rows.append(
            (
                CFG[k],
                step,
                f"(force alone {ev})",
                f"{acc['rel_l2']:.2e}" if acc else "-",
                f"{acc['p90']:.2e}" if acc else "-",
                f"{pk:.2f} ({pk*G/n:.0f})" if pk else "-",
            )
        )
    if not rows:
        continue
    print(
        f"\n**N = {n:.1e}**\n\n| code / config | ms (step or force) | | rel-L2 | p90 | peak GiB (B/p) |\n| --- | --- | --- | --- | --- | --- |"
    )
    for r in rows:
        print("| " + " | ".join(r) + " |")
