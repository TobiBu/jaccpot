"""Copy finished figure PDFs into the manuscript repo and update its provenance.

The manuscript repo never computes: it receives PDFs. What makes that safe is the
provenance table, so this writes the table from the artifacts themselves rather
than from anyone's memory -- each row's commit, date and config are read out of
the JSON the figure was built from, so a row cannot describe a run that did not
happen.

    python examples/jaccpot_paper/export_to_paper_repo.py [--paper-repo PATH]

Figures whose PDF or artifact is missing are listed as absent rather than skipped
silently, so a partial export is visible in the table instead of looking complete.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# (pdf stem, figure label, notebook, artifact)
FIGURES = [
    (
        "fig01_force_error_vs_order",
        "01",
        "fig_01_force_error_vs_order.ipynb",
        "validation/force_error_vs_order.json",
    ),
    (
        "fig02_error_vs_theta",
        "02",
        "fig_02_error_vs_theta.ipynb",
        "validation/error_vs_theta.json",
    ),
    (
        "fig03_mac_comparison",
        "03",
        "fig_03_mac_comparison.ipynb",
        "validation/mac_comparison.json",
    ),
    (
        "fig04_wallclock_vs_n",
        "04",
        "fig_04_wallclock_vs_n.ipynb",
        "scaling/wallclock_vs_n.json",
    ),
    (
        "fig05_interaction_counts",
        "05",
        "fig_05_interaction_counts.ipynb",
        "scaling/interaction_counts.json",
    ),
    (
        "fig06_stage_breakdown",
        "06",
        "fig_06_stage_breakdown.ipynb",
        "scaling/stage_breakdown.json",
    ),
    (
        "fig07_gpu_vs_cpu",
        "07",
        "fig_07_gpu_vs_cpu.ipynb",
        "scaling/gpu_vs_cpu.json",
    ),
    (
        "fig12_autodiff_overhead",
        "12",
        "fig_12_autodiff_overhead.ipynb",
        "differentiability/autodiff_overhead.json",
    ),
    (
        "fig13_grad_correctness",
        "13",
        "fig_13_grad_correctness.ipynb",
        "differentiability/grad_correctness.json",
    ),
]

HEADER = """# Figure provenance

Figures are never computed here. They are generated in the `jaccpot` repo
(`examples/jaccpot_paper/fig_*.ipynb`, reading from `results/**/*.json` produced
by `bench/**/*.py` on branch `paper/jaccpot-i`), exported, and copied into this
directory.

**This table is generated, not hand-maintained.** Regenerate it with

```bash
python examples/jaccpot_paper/export_to_paper_repo.py
```

from the `jaccpot` checkout, which reads each figure's commit, date and
configuration out of the artifact it was built from. Editing a row by hand
defeats the point: the table exists so that no number in the manuscript is
traceable only to someone's recollection.

Every row's `Measured at` is the commit the *benchmark* ran on, and `Dirty` flags
a working tree that had uncommitted changes when it ran -- a dirty run is not
fully described by its commit and should be regenerated before submission.

"""

TABLE_HEADER = (
    "| Fig | Figure file | Source notebook | Source data | Measured at | Dirty | Device | Config | Date |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


def _config_summary(config: dict) -> str:
    keys = ("n", "order", "theta", "basis", "leaf_size", "preset", "precision", "seed")
    parts = []
    for key in keys:
        if key not in config or config[key] is None:
            continue
        value = config[key]
        if isinstance(value, list):
            if len(value) > 3:
                value = f"{value[0]}..{value[-1]}"
            else:
                value = ",".join(str(v) for v in value)
        parts.append(f"{key}={value}")
    return "<br>".join(parts) if parts else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--paper-repo",
        default=str(REPO_ROOT.parent / "jaccpot-paper-i"),
        help="Path to the jaccpot-paper-i checkout",
    )
    args = ap.parse_args()

    paper = pathlib.Path(args.paper_repo)
    figures_dir = paper / "figures"
    if not figures_dir.exists():
        print(f"error: {figures_dir} does not exist", file=sys.stderr)
        return 1

    pdf_src = REPO_ROOT / "bench" / "results" / "figures"
    rows: list[str] = []
    copied = missing = 0

    for stem, label, notebook, artifact in FIGURES:
        pdf = pdf_src / f"{stem}.pdf"
        art_path = REPO_ROOT / "bench" / "results" / artifact

        if not pdf.exists() or not art_path.exists():
            what = []
            if not pdf.exists():
                what.append("PDF")
            if not art_path.exists():
                what.append("artifact")
            rows.append(
                f"| {label} | `{stem}.pdf` | `examples/jaccpot_paper/{notebook}` "
                f"| `results/{artifact}` | _not yet generated ({' and '.join(what)} "
                f"missing)_ | | | | |"
            )
            missing += 1
            continue

        payload = json.loads(art_path.read_text())
        meta, config = payload.get("meta", {}), payload.get("config", {})
        shutil.copy2(pdf, figures_dir / pdf.name)
        copied += 1

        sha = str(meta.get("git_sha", ""))[:12] or "?"
        dirty = "**yes**" if meta.get("git_dirty") else "no"
        device = str(config.get("device", "?"))
        date = datetime.datetime.fromtimestamp(art_path.stat().st_mtime).strftime(
            "%Y-%m-%d"
        )
        rows.append(
            f"| {label} | `{stem}.pdf` | `examples/jaccpot_paper/{notebook}` "
            f"| `results/{artifact}` | `{sha}` | {dirty} | {device} "
            f"| {_config_summary(config)} | {date} |"
        )

    readme = figures_dir / "README.md"
    readme.write_text(HEADER + TABLE_HEADER + "\n".join(rows) + "\n")

    print(f"copied {copied} PDF(s) into {figures_dir}")
    if missing:
        print(f"{missing} figure(s) not yet generated; recorded as such in the table")
    print(f"wrote {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
