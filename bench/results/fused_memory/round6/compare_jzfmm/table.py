"""Step 0 table: jz-fmm vs jaccpot main at matched accuracy (round-6 rows + round-5 jz rows)."""

import json
import os
import sys

O = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "round5", "compare_jzfmm"
)
G = 2**30
JZ = {"jzF": "p6 θ0.8", "jzG": "p5 θ0.7", "jzH": "p6 θ0.7", "jzB": "p5 θ0.8"}
JZ5 = {
    "jzB": "p5 θ0.8",
    "jzD": "p5 θ0.9",
    "jzE": "p6 θ0.9",
    "jzC": "p4 θ0.7",
    "jzA": "p4 θ0.6",
}
OU = {"ouP": "p6 θ0.8 cml8 (default)", "ouQ": "p5 θ0.8 cml8"}
for n in (8000000, 100000000):
    rows = []

    def jz(path, label):
        if not os.path.exists(path):
            return
        r = json.load(open(path))["rows"][0]
        if "timing" not in r:
            rows.append((label, "failed", "", "", "", ""))
            return
        rows.append(
            (
                label,
                f"{1e3*r['timing']['min']:.1f}",
                "force incl. tree",
                f"{r['error']['aggL2']:.2e}",
                f"{r['error']['p90']:.2e}",
                f"{r['memory']['peak_gib']:.2f} ({r['memory']['peak_bytes_per_particle']:.0f})",
            )
        )

    for k, v in JZ.items():
        jz(f"{O}/{k}_{n}.json", f"jz-fmm {v} (r6)")
    for k, v in JZ5.items():
        jz(f"{R5}/{k}_{n}.json", f"jz-fmm {v} (r5, 10-05)")
    for k, v in OU.items():
        f, s = f"{O}/{k}_{n}_f.json", f"{O}/{k}_{n}_s.json"
        if not (os.path.exists(f) or os.path.exists(s)):
            continue
        df = json.load(open(f)) if os.path.exists(f) else {}
        ds = json.load(open(s)) if os.path.exists(s) else {}
        acc = df.get("accuracy", {})
        pk = ds.get("peak_after_scan_gib")
        rows.append(
            (
                f"jaccpot {v}",
                f"{ds['step_ms']['min']:.1f}" if "step_ms" in ds else "failed",
                (
                    f"step (force alone {df['eval_ms']['min']:.1f})"
                    if "eval_ms" in df
                    else "step"
                ),
                f"{acc['rel_l2']:.2e}" if acc else "-",
                f"{acc['p90']:.2e}" if acc else "-",
                f"{pk:.2f} ({pk*G/n:.0f})" if pk else "-",
            )
        )
    if not rows:
        continue
    print(
        f"\n**N = {n:.1e}**\n\n| code / config | ms | what | rel-L2 | p90 | peak GiB (B/p) |\n| --- | --- | --- | --- | --- | --- |"
    )
    for r in rows:
        print("| " + " | ".join(r) + " |")
