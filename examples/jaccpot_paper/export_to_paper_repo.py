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
import subprocess
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
        "fig08_strong_scaling",
        "08",
        "fig_08_strong_scaling.ipynb",
        "multigpu/strong_scaling.json",
    ),
    (
        "fig09_weak_scaling",
        "09",
        "fig_09_weak_scaling.ipynb",
        "multigpu/weak_scaling.json",
    ),
    (
        "fig11_load_balance",
        "11",
        "fig_11_load_balance.ipynb",
        "multigpu/load_balance.json",
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
    (
        "fig16_gradient_cost_vs_nparams",
        "16",
        "fig_16_gradient_cost_vs_nparams.ipynb",
        "density_reconstruction/gradient_cost_vs_nparams.json",
    ),
    (
        "fig17_reconstruction_quality",
        "17",
        "fig_17_reconstruction_quality.ipynb",
        "density_reconstruction/reconstruction_runs.json",
    ),
    (
        "fig18_convergence_and_degeneracy",
        "18",
        "fig_18_convergence_and_degeneracy.ipynb",
        "density_reconstruction/reconstruction_runs.json",
    ),
    (
        "fig19_topology_switching",
        "19",
        "fig_19_topology_switching.ipynb",
        "density_reconstruction/topology_switching.json",
    ),
    (
        "fig20_multigpu_reconstruction_scaling",
        "20",
        "fig_20_multigpu_reconstruction_scaling.ipynb",
        "density_reconstruction/multigpu_scaling.json",
    ),
]

HEADER = """# Figure provenance

Figures are never computed here. They are generated in the `jaccpot` repo
(`examples/jaccpot_paper/fig_*.ipynb`, reading from `bench/results/**/*.json` produced
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

`Dirty` counts changes to **source**, not to `bench/results/`. Several benches
rewrite their own tracked artifact after every measured point, so that a sweep
interrupted after hours still leaves something usable; the side effect is that
two sweeps running at once each see the other's output as a dirty tree. Marking
a figure unfit because a *different* figure's data was being written would say
nothing about whether this row's commit describes the code that produced it.
Artifacts written before `git_dirty_sources` existed fall back to the
whole-tree flag.

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


def _existing_dates(readme: pathlib.Path) -> dict[str, tuple[str, str]]:
    """Read ``{figure label: (sha, date)}`` out of a table already written.

    Parameters
    ----------
    readme : pathlib.Path
        An existing ``figures/README.md``, or a path that does not exist.

    Returns
    -------
    dict[str, tuple[str, str]]
        Label to the commit and date its row records. Empty when there is no
        table yet or it cannot be parsed -- a malformed row is skipped rather
        than guessed at.
    """
    if not readme.exists():
        return {}
    out: dict[str, tuple[str, str]] = {}
    for line in readme.read_text().splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 9 or not cells[0].isdigit():
            continue
        out[cells[0]] = (cells[4].strip("`"), cells[8])
    return out


def _commit_date(sha: str) -> str:
    """Return the commit date of ``sha``, or an empty string.

    Parameters
    ----------
    sha : str
        A commit hash recorded in an artifact.

    Returns
    -------
    str
        ``YYYY-MM-DD``, or ``""`` when the commit is unknown to this checkout.

    Notes
    -----
    A weaker source than a timestamp the run wrote down -- a measurement can be
    taken long after the commit it ran at -- but far better than the artifact
    file's mtime, which in a fresh worktree is the checkout time and therefore
    identical for every figure. This is stable across checkouts, which is the
    property the table needs.
    """
    if not sha or sha == "?":
        return ""
    try:
        out = subprocess.run(
            ["git", "show", "-s", "--format=%cs", sha],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover
        return ""
    line = out.stdout.strip().splitlines()
    return line[0] if line and out.returncode == 0 else ""


def _measurement_date(
    meta: dict,
    art_path: pathlib.Path,
    previous: tuple[str, str] | None,
    sha: str,
) -> str:
    """Return the date a measurement was taken, most trustworthy source first.

    Parameters
    ----------
    meta : dict
        The artifact's ``meta`` block.
    art_path : pathlib.Path
        The artifact, used only for the last-resort mtime.
    previous : tuple[str, str] | None
        The ``(sha, date)`` the existing table records for this figure, if any.
    sha : str
        The commit this artifact records.

    Returns
    -------
    str
        ``YYYY-MM-DD``.

    Notes
    -----
    The mtime is the WEAKEST source and used to be the only one. It is the
    checkout time in a fresh worktree, so exporting from a different checkout
    rewrote every figure's date to the day that tree was created -- observed
    here as thirteen rows moving from 2026-08-16/22 to 2026-09-03 with nothing
    else about them changing. Preferred instead: the timestamp the run wrote
    into the artifact, and failing that the date already in the table for the
    SAME commit, which is a record of when that artifact was actually measured.
    """
    recorded = str(meta.get("measured_at", ""))
    if recorded:
        return recorded[:10]
    if previous and previous[0] == sha and previous[1]:
        return previous[1]
    committed = _commit_date(sha)
    if committed:
        return committed
    return datetime.datetime.fromtimestamp(art_path.stat().st_mtime).strftime(
        "%Y-%m-%d"
    )


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
    previous_dates = _existing_dates(figures_dir / "README.md")
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
                f"| `bench/results/{artifact}` | _not yet generated ({' and '.join(what)} "
                f"missing)_ | | | | |"
            )
            missing += 1
            continue

        payload = json.loads(art_path.read_text())
        meta, config = payload.get("meta", {}), payload.get("config", {})
        shutil.copy2(pdf, figures_dir / pdf.name)
        copied += 1

        sha = str(meta.get("git_sha", ""))[:12] or "?"
        # Prefer git_dirty_sources, which is what this column has always MEANT:
        # the header says a dirty run "is not fully described by its commit",
        # and a rewritten results JSON does not affect that. Several benches now
        # rewrite their tracked artifact after every point, so two sweeps
        # running side by side each see the other's output and both set
        # git_dirty -- which would flag every section-7 figure as unfit from a
        # clean checkout. Artifacts written before that field existed fall back
        # to the old flag.
        if "git_dirty_sources" in meta:
            dirty = "**yes**" if meta.get("git_dirty_sources") else "no"
        else:
            dirty = "**yes**" if meta.get("git_dirty") else "no"
        device = str(config.get("device", "?"))
        date = _measurement_date(meta, art_path, previous_dates.get(label), sha)
        rows.append(
            f"| {label} | `{stem}.pdf` | `examples/jaccpot_paper/{notebook}` "
            f"| `bench/results/{artifact}` | `{sha}` | {dirty} | {device} "
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
