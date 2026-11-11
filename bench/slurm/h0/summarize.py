"""HoreKa bundle: one ``SUMMARY.md`` and one ``rows.jsonl`` for a run directory.

    python bench/slurm/h0/summarize.py RUN_DIR        # (collect.sh calls it)

Reads every job's output under ``RUN_DIR/<job>/`` (whatever exists: a job that did not
run or did not finish is reported as such) and writes:

* ``RUN_DIR/SUMMARY.md`` -- a table per job, the derived numbers (A-vs-A spreads, the
  #104 cost at 1e8, strong / weak scaling, the 4-card ceiling, auto vs level 8) and the
  sanity row against the local single-card pins (``local_reference.json``); for the jobs
  folded in by track H1 (scaling_mp, forward, mgrad, escaper, a1, a3, m1_*, m2g_*,
  mesc_*, a4) each job's own summary (the matrix's or track's ``SUMMARY.md``, the bench's
  ``summary.txt``) or a table of its rows;
* ``RUN_DIR/rows.jsonl`` -- one JSON object per row, each with its job's provenance
  header (commits, jax, node, GPU, driver) under ``"provenance"``.

Numbers are read, never re-derived from logs; a missing field prints ``-``. It needs only
numpy and the standard library, except two optional helpers that import from the bench
scripts they summarise (the scaling matrix's needs-vector index, mimirax's row reader):
when an import fails the section says so and lists the raw rows instead.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JACCPOT = HERE.parents[2]
LOCAL_REFERENCE = HERE / "local_reference.json"
#: the local tags the sanity row compares with (local_reference.json: the bundle's pins, track H1)
SANITY_TAGS = ("h1-L-main", "h1-L-main-b")
#: every job of submit_all.sh, in its order
JOBS = (
    "preflight",
    "pins",
    "auto",
    "s1b",
    "grad",
    "scaling_a",
    "scaling_mp",
    "scaling_b",
    "forward",
    "mgrad",
    "escaper",
    "a1",
    "a3",
    "m1_node1",
    "m1_node2",
    "m2g_node1",
    "m2g_node2",
    "mesc_node1",
    "mesc_node2",
    "a4",
)


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _f(x, spec="{:.2f}"):
    if x is None:
        return "-"
    try:
        return spec.format(x)
    except (TypeError, ValueError):
        return str(x)


def _pct(a, b):
    """(b - a) / a in percent."""
    if a in (None, 0) or b is None:
        return None
    return 100.0 * (b - a) / a


class Out:
    def __init__(self):
        self.lines: list[str] = []
        self.rows: list[dict] = []

    def __call__(self, line: str = "") -> None:
        self.lines.append(line)

    def table(self, header: list[str], rows: list[list]) -> None:
        self("| " + " | ".join(header) + " |")
        self("| " + " | ".join("---" for _ in header) + " |")
        for r in rows:
            self("| " + " | ".join(str(c) for c in r) + " |")
        self()

    def row(self, job: str, kind: str, data: dict, prov: dict | None) -> None:
        self.rows.append({"job": job, "kind": kind, "provenance": prov, **data})


def _provenance_brief(prov: dict | None) -> dict:
    if not prov:
        return {}
    git = prov.get("git", {})
    gpu_rows = (prov.get("gpus", {}).get("rows") or "").splitlines()
    first = [x.strip() for x in gpu_rows[0].split(",")] if gpu_rows else []
    return {
        "host": prov.get("host"),
        "time": prov.get("time"),
        "gpu": first[1] if len(first) > 1 else None,
        "driver": first[4] if len(first) > 4 else None,
        "gpus": len(gpu_rows),
        "jax": (prov.get("jax") or {}).get("jax"),
        "jaccpot": (git.get("jaccpot") or {}).get("head"),
        "jaccpot_dirty": bool((git.get("jaccpot") or {}).get("status")),
        "yggdrax": (git.get("yggdrax") or {}).get("head"),
        "nornax": (git.get("nornax") or {}).get("head"),
        "mimirax": (git.get("mimirax") or {}).get("head"),
        "slurm_job": (prov.get("env") or {}).get("SLURM_JOB_ID"),
    }


# ----------------------------------------------------------------------- sections


def section_provenance(o: Out, run: Path, provs: dict) -> None:
    o("## Provenance and job status")
    o()
    rows = []
    for job in JOBS:
        d = run / job
        b = _provenance_brief(provs.get(job))
        status = (
            "skipped: " + (d / "SKIPPED").read_text().strip()[:80]
            if (d / "SKIPPED").exists()
            else (
                "done"
                if (d / "DONE").exists()
                else ("not run" if not d.exists() else "INCOMPLETE")
            )
        )
        short = lambda s: (s or "-")[:10]  # noqa: E731
        rows.append(
            [
                job,
                status,
                b.get("slurm_job") or "-",
                b.get("host") or "-",
                f"{b.get('gpus', '-')} x {b.get('gpu') or '-'}",
                b.get("driver") or "-",
                b.get("jax") or "-",
                short(b.get("jaccpot"))
                + (" (dirty)" if b.get("jaccpot_dirty") else ""),
                short(b.get("yggdrax")),
                short(b.get("nornax")),
                short(b.get("mimirax")),
            ]
        )
    o.table(
        [
            "job",
            "status",
            "slurm",
            "node",
            "GPUs",
            "driver",
            "jax",
            "jaccpot",
            "yggdrax",
            "nornax",
            "mimirax",
        ],
        rows,
    )
    st = run / "preflight" / "status"
    if st.is_dir():
        o(
            "Preflight entry points (a broken one stops its job at once): "
            + ", ".join(
                f"{p.name} {p.read_text().strip()}" for p in sorted(st.iterdir())
            )
        )
        o()


def _pin(root: Path, tag: str, pin: str):
    return _load(root / tag / f"{pin}.json")


def section_pins(o: Out, run: Path, prov) -> None:
    o("## 1. Single-card speed pins L1-L3 (one card of an exclusive node)")
    o()
    root = run / "pins" / "pins"
    if not root.is_dir():
        o("_not run_")
        o()
        return
    o(
        "`bench/dce_pins.py record`: L1 clipped Plummer 8e6, L2 1e8 (one step each, the fused "
        "production sequence, particle carry), L3 the 25M disc + bulge step and force (ferrers3, eps "
        "1.5e-3, rel-L2 on 4096 targets against fp64). min / median over 3 reps of 4 steps. B arm = "
        "yggdrax main with `_cell_partition.py` from before #104."
    )
    o()
    rows = []
    for tagdir in sorted(p for p in root.iterdir() if p.is_dir()):
        tag = tagdir.name
        for pin in ("L1", "L2", "L3"):
            d = _pin(root, tag, pin)
            if not d:
                continue
            s, f = d.get("step") or {}, d.get("force") or {}
            rows.append(
                [
                    tag,
                    pin,
                    f"{d.get('n', 0):,}",
                    _f(s.get("ms_min")),
                    _f(s.get("ms_median")),
                    _f(s.get("bytes_per_particle"), "{:.1f}"),
                    _f(s.get("far_pair_count"), "{:,}"),
                    _f(f.get("ms_min")),
                    _f(f.get("rel_l2"), "{:.4e}"),
                    "yes" if s.get("contaminated") else "no",
                ]
            )
            o.row(
                "pins", pin, {"tag": tag, "n": d.get("n"), "step": s, "force": f}, prov
            )
    o.table(
        [
            "tag",
            "pin",
            "N",
            "step ms min",
            "step ms median",
            "peak B/p",
            "far pairs",
            "force ms min",
            "rel-L2",
            "contaminated",
        ],
        rows,
    )
    # A-vs-A spread
    o("**A-vs-A** (`h0-L-main` vs `h0-L-main-b`, same code, same node):")
    o()
    spread = []
    for pin in ("L1", "L2", "L3"):
        a, b = _pin(root, "h0-L-main", pin), _pin(root, "h0-L-main-b", pin)
        if a and b:
            ma, mb = a["step"].get("ms_min"), b["step"].get("ms_min")
            spread.append([pin, _f(ma), _f(mb), _f(_pct(ma, mb), "{:+.2f} %")])
    o.table(["pin", "A ms min", "A-b ms min", "change"], spread)
    # #104
    o(
        "**5. The cost of yggdrax #104's extra cummax at 1e8** (L2, A = main, B = before #104, run A B A B A):"
    )
    o()
    arms = {
        "A": ["h0-L-main", "h0-L-main-b", "h0-L2-main-c"],
        "B": ["h0-L2-pre104", "h0-L2-pre104-b"],
    }
    vals = {}
    for arm, tags in arms.items():
        vals[arm] = [
            (_pin(root, t, "L2") or {}).get("step", {}).get("ms_min")
            for t in tags
            if _pin(root, t, "L2")
        ]
    if vals["A"] and vals["B"]:
        ma, mb = statistics.mean(vals["A"]), statistics.mean(vals["B"])
        o.table(
            ["arm", "L2 step ms min per record", "mean", "range"],
            [
                [
                    arm,
                    ", ".join(_f(v) for v in vals[arm]),
                    _f(statistics.mean(vals[arm])),
                    _f(max(vals[arm]) - min(vals[arm])),
                ]
                for arm in ("A", "B")
            ],
        )
        o(
            f"main (with #104) vs before: **{_f(_pct(mb, ma), '{:+.2f} %')}** "
            f"({_f(ma)} vs {_f(mb)} ms); the dce_pins envelope verdicts are in `pins/compare_*.txt`."
        )
        o.row(
            "pins",
            "cummax104",
            {"A_ms_min": vals["A"], "B_ms_min": vals["B"], "change_pct": _pct(mb, ma)},
            prov,
        )
    else:
        o("_the A / B L2 records are incomplete_")
    o()
    for txt in sorted((run / "pins").glob("compare_*.txt")):
        o(f"<details><summary>{txt.name}</summary>")
        o()
        o("```")
        o(txt.read_text().rstrip())
        o("```")
        o("</details>")
        o()
    # sanity against local
    o("## 7. Sanity: HoreKa vs the local single-card pins (`local_reference.json`)")
    o()
    ref = _load(LOCAL_REFERENCE) or {}
    rows = []
    for pin in ("L1", "L2", "L3"):
        h = [
            (_pin(root, t, pin) or {}).get("step", {}).get("ms_min")
            for t in ("h0-L-main", "h0-L-main-b")
            if _pin(root, t, pin)
        ]
        hf = [
            (_pin(root, t, pin) or {}).get("force", {}).get("rel_l2")
            for t in ("h0-L-main",)
            if _pin(root, t, pin)
        ]
        hb = [
            (_pin(root, t, pin) or {}).get("step", {}).get("bytes_per_particle")
            for t in ("h0-L-main",)
            if _pin(root, t, pin)
        ]
        for lt in SANITY_TAGS:
            loc = ((ref.get("tags") or {}).get(lt) or {}).get(pin) or {}
            hm = statistics.mean(h) if h else None
            rows.append(
                [
                    pin,
                    lt,
                    _f(loc.get("step_ms_min")),
                    _f(hm),
                    _f(
                        (
                            hm / loc["step_ms_min"]
                            if hm and loc.get("step_ms_min")
                            else None
                        ),
                        "{:.3f}",
                    ),
                    _f(loc.get("bytes_per_particle"), "{:.1f}")
                    + " / "
                    + _f(hb[0] if hb else None, "{:.1f}"),
                    (
                        (
                            _f(loc.get("rel_l2"), "{:.4e}")
                            + " / "
                            + _f(hf[0] if hf else None, "{:.4e}")
                        )
                        if pin == "L3"
                        else "-"
                    ),
                ]
            )
    o.table(
        [
            "pin",
            "local tag",
            "local ms min",
            "HoreKa ms min (mean A, A-b)",
            "HoreKa / local",
            "B/p local / HoreKa",
            "L3 rel-L2 local / HoreKa",
        ],
        rows,
    )
    o(
        "Local: A100-PCIE-40GB; HoreKa: the node's A100 (see provenance). B/p and rel-L2 must agree to the "
        "digits shown (same program, same IC); the time ratio is the hardware's."
    )
    o()


def _needs_index():
    sys.path.insert(0, str(JACCPOT / "bench"))
    try:
        import multigpu_scaling_matrix as m  # noqa: F401

        return m._needs_index(), m
    except Exception as exc:  # pragma: no cover - reported in the summary
        return exc, None


def section_scaling(o: Out, run: Path, provs: dict) -> None:
    o("## 2. Multi-GPU scaling on one NVLink node (#480 matrix)")
    o()
    index, mod = _needs_index()
    all_rows = {}
    for job in ("scaling_a", "scaling_b"):
        d = run / job / "matrix"
        if not d.is_dir():
            o(f"_{job}: not run_")
            o()
            continue
        if mod is not None:
            for path in sorted(d.glob("*.json")):
                if path.name in ("rows.json",):
                    continue
                try:
                    r = mod._row_summary(path.stem, d, index)
                except Exception as exc:  # noqa: BLE001
                    r = {"tag": path.stem, "error": repr(exc)}
                all_rows[r["tag"]] = r
                o.row(job, "scaling", r, provs.get(job))
        else:
            o(
                f"_cannot import bench/multigpu_scaling_matrix.py ({index!r}); raw SUMMARY.md below_"
            )
        md = d / "SUMMARY.md"
        if md.exists():
            o(f"<details><summary>{job}: the matrix's own SUMMARY.md</summary>")
            o()
            o(md.read_text().rstrip())
            o()
            o("</details>")
            o()
    if not all_rows:
        return

    def ms(tag):
        r = all_rows.get(tag) or {}
        return r.get("step_ms")

    o(
        "**Strong scaling** (step ms; speed-up against the one-device mesh and the single-card runner):"
    )
    o()
    rows = []
    for name in ("D2.5e7_de", "D2.5e7_geo", "P8e6_de", "P8e6_geo"):
        one, solo = ms(f"strong_{name}_1c"), ms(f"strong_{name}_solo")
        two, four = ms(f"strong_{name}_2c"), ms(f"strong_{name}_4c")
        rows.append(
            [
                name,
                _f(solo),
                _f(one),
                _f(two),
                _f(four),
                _f(one / two if one and two else None, "{:.2f}x"),
                _f(one / four if one and four else None, "{:.2f}x"),
                _f(solo / four if solo and four else None, "{:.2f}x"),
            ]
        )
    o.table(
        [
            "config",
            "1 card (solo)",
            "1-device mesh",
            "2 cards",
            "4 cards",
            "2c speed-up",
            "4c speed-up",
            "4c vs solo",
        ],
        rows,
    )
    o("**Weak scaling** (4e6 Plummer per card, `dehnen_error`, migrating):")
    o()
    w1 = ms("weak_P4e6pc_1c")
    o.table(
        ["cards", "step ms", "efficiency (t1 / tk)"],
        [
            [
                k,
                _f(ms(f"weak_P4e6pc_{k}c")),
                _f(
                    (
                        w1 / ms(f"weak_P4e6pc_{k}c")
                        if w1 and ms(f"weak_P4e6pc_{k}c")
                        else None
                    ),
                    "{:.0%}",
                ),
            ]
            for k in (1, 2, 4)
        ],
    )
    o(
        "**Memory per card and the 4-card ceiling** (Plummer, migrating, 3 x 4 steps; the first failed row stops the tier):"
    )
    o()
    mem = sorted(
        (r for t, r in all_rows.items() if t.startswith("memory_")),
        key=lambda r: r.get("n") or 0,
    )
    rows = []
    for r in mem:
        smi = _load(run / "scaling_a" / "matrix" / f"{r['tag']}.smi.txt") or {}
        rows.append(
            [
                r["tag"],
                _f(r.get("n"), "{:,}"),
                smi.get("rc", "-"),
                _f(r.get("step_ms")),
                _f(r.get("bpp"), "{:.0f}"),
                _f(
                    (r.get("bpp") or 0)
                    * (r.get("n") or 0)
                    / max(r.get("ndev") or 1, 1)
                    / 2**30,
                    "{:.1f}",
                ),
            ]
        )
    failed = [
        t
        for t in (run / "scaling_a" / "matrix").glob("memory_*.smi.txt")
        if (_load(t) or {}).get("rc") not in (0, None)
    ]
    o.table(
        [
            "row",
            "N (4 cards)",
            "rc",
            "step ms",
            "peak B/p (max device)",
            "GiB / card (approx.)",
        ],
        rows,
    )
    if failed:
        o(
            "First failed memory row(s): "
            + ", ".join(p.name.replace(".smi.txt", "") for p in failed)
            + " (the ceiling lies below it; see its log)."
        )
    else:
        o("No memory row failed: the ceiling is above the largest row.")
    o()
    ab = sorted(t for t in all_rows if t.startswith("ab_"))
    if ab:
        o("**Migration on / off** (`ab_` rows; step ms per repetition):")
        o()
        groups: dict = {}
        for t in ab:
            base, rep = t.rsplit("_", 1)
            cfg, arm = base.rsplit("_", 1)
            groups.setdefault(cfg, {}).setdefault(arm, []).append(ms(t))
        rows = []
        for cfg, arms in sorted(groups.items()):
            off = [v for v in arms.get("off", []) if v]
            on = [v for v in arms.get("on", []) if v]
            rows.append(
                [
                    cfg,
                    ", ".join(_f(v) for v in off),
                    ", ".join(_f(v) for v in on),
                    _f(
                        (
                            statistics.mean(on) / statistics.mean(off)
                            if on and off
                            else None
                        ),
                        "{:.2f}",
                    ),
                ]
            )
        o.table(["config", "off", "on", "on / off"], rows)


def section_text(o: Out, title: str, path: Path) -> None:
    o(title)
    o()
    if path.exists():
        o("```")
        o(path.read_text().rstrip() or "(empty)")
        o("```")
    else:
        o(f"_no {path.name}_")
    o()


def section_grad(o: Out, run: Path, prov) -> None:
    d = run / "grad" / "timing"
    section_text(
        o,
        "## 3. S9 gradient timing (`grad_rollout_timing.py --summarize`)",
        d / "summary.txt",
    )
    for path in sorted((d / "rows").glob("*.json")) if (d / "rows").is_dir() else []:
        o.row("grad", "grad", {"tag": path.stem, "row": _load(path)}, prov)


def section_s1b(o: Out, run: Path, prov) -> None:
    o("## 4. S1b: Mosaic GPU vs Triton-Pallas L2P (`s1b_mosaic_l2p.py --mode time`)")
    o()
    path = run / "s1b" / "l2p" / "rows.jsonl"
    if not path.exists():
        o("_not run_")
        o()
        return
    rows = []
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        o.row("s1b", "l2p", r, prov)
        for key in sorted(k for k in r if k.endswith("_ms") and isinstance(r[k], dict)):
            for kern, t in r[key].items():
                if isinstance(t, dict):
                    rows.append(
                        [
                            _f(r.get("n"), "{:,}"),
                            r.get("ic", "-"),
                            key,
                            kern,
                            _f(t.get("min")),
                            _f(t.get("median")),
                        ]
                    )
    o.table(["N", "IC", "program", "kernel", "ms min", "ms median"], rows)
    o(
        "Bitwise / max diff against Triton and the temp bytes per kernel are in the rows (`rows.jsonl`)."
    )
    o()


def section_auto(o: Out, run: Path, prov) -> None:
    o(
        '## 6. `cell_min_level="auto"` vs the preset\'s level 8 (mimirax S10 bench, one card)'
    )
    o()
    d = run / "auto" / "rows"
    if not d.is_dir():
        o("_not run_")
        o()
        return
    rows = []
    for path in sorted(d.glob("*.json")):
        rec = _load(path) or {}
        cfg = rec.get("config", {})
        for key, t in (rec.get("timing") or {}).items():
            if isinstance(t, dict) and "median_s" in t:
                rows.append(
                    dict(
                        tag=path.stem,
                        program=f"{cfg.get('only')}:{key}",
                        n=cfg.get("n"),
                        median_s=t.get("median_s"),
                        min_s=t.get("min_s"),
                        per_step_ms=t.get("per_step_ms"),
                        level=cfg.get("cell_min_level") or "8 (preset)",
                        contended=rec.get("contended"),
                    )
                )
        o.row("auto", "auto", {"tag": path.stem, "row": rec}, prov)
    o.table(
        [
            "row",
            "program",
            "N",
            "cell_min_level",
            "median s",
            "min s",
            "ms / step",
            "contended",
        ],
        [
            [
                r["tag"],
                r["program"],
                _f(r["n"], "{:,}"),
                r["level"],
                _f(r["median_s"], "{:.4f}"),
                _f(r["min_s"], "{:.4f}"),
                _f(r["per_step_ms"]),
                r["contended"],
            ]
            for r in rows
        ],
    )
    # auto / base per (kind, N, program), over the r1 / r2 repetitions
    pairs: dict = {}
    for r in rows:
        stem = r["tag"].rsplit("_r", 1)[0]
        arm = "auto" if "_auto" in stem else "base"
        key = (stem.replace("_auto", ""), r["program"])
        pairs.setdefault(key, {}).setdefault(arm, []).append(
            r["per_step_ms"] or r["median_s"]
        )
    out = []
    for (stem, prog), arms in sorted(pairs.items()):
        b = [v for v in arms.get("base", []) if v]
        a = [v for v in arms.get("auto", []) if v]
        out.append(
            [
                stem,
                prog,
                ", ".join(_f(v, "{:.4g}") for v in b),
                ", ".join(_f(v, "{:.4g}") for v in a),
                _f(
                    statistics.mean(a) / statistics.mean(b) if a and b else None,
                    "{:.3f}",
                ),
            ]
        )
    o(
        "**auto / level 8** (ms per step for the segment gradients, seconds for the pre-pass; < 1 = auto faster):"
    )
    o()
    o.table(["row", "program", "level 8 (r1, r2)", "auto (r1, r2)", "auto / 8"], out)
    o(
        "The level `auto` picked is in each row's JSON (the bench's `config` / the plan's tree record); the S10 "
        "follow-ups found 5 at 1e5, 6 at 1e6, 7 at 1e7 on this disc."
    )
    o()


def _json_rows(o: Out, job: str, d: Path, prov) -> list:
    """Every ``*.json`` row under ``d`` into rows.jsonl; ``[(tag, row)]``."""
    out = []
    for path in sorted(d.glob("*.json")) if d.is_dir() else []:
        row = _load(path)
        if row is None:
            continue
        out.append((path.stem, row))
        o.row(job, job, {"tag": path.stem, "row": row}, prov)
    return out


def section_folded(o: Out, run: Path, provs: dict) -> None:
    """The jobs track H1 folded into the bundle: each job's own summary, or its rows."""
    o("## 8. The jobs folded in by H1")
    o()
    d = run / "scaling_mp" / "matrix"
    section_text(
        o,
        "### scaling_mp: N processes x 1 GPU vs 1 process x N GPUs (`multigpu_scaling_matrix.py --tiers mp`)",
        d / "SUMMARY.md",
    )
    for path in [d / "rows_mp.jsonl"] if (d / "rows_mp.jsonl").exists() else []:
        for line in path.read_text().splitlines():
            if line.strip():
                o.row("scaling_mp", "mp", json.loads(line), provs.get("scaling_mp"))
    d = run / "forward" / "timing"
    section_text(
        o,
        "### forward: forward mode (`forward_mode_timing.py --summarize`)",
        d / "summary.txt",
    )
    _json_rows(o, "forward", d / "rows", provs.get("forward"))
    d = run / "mgrad" / "matrix"
    section_text(
        o,
        "### mgrad: WP-G multi-GPU gradients (`multigpu_grad_matrix.py --summary`)",
        d / "SUMMARY.md",
    )
    _json_rows(o, "mgrad", d, provs.get("mgrad"))
    # escaper: one card, 1e8 Sgr
    rows = _json_rows(
        o, "escaper", run / "escaper" / "timing" / "rows", provs.get("escaper")
    )
    o("### escaper: escapers on / off at 1e8, one card (`escaper_timing.py`)")
    o()
    if rows:
        o.table(
            ["row", "N", "step s", "edge s", "peak GiB", "peak B/p", "escapers"],
            [
                [
                    tag,
                    r.get("n", "-"),
                    _f(r.get("step_s"), "{:.3f}"),
                    _f(r.get("edge_s"), "{:.3f}"),
                    _f((r.get("peak_bytes") or 0) / 2**30),
                    _f(r.get("peak_bytes_per_particle"), "{:.1f}"),
                    _f(r.get("escapers"), "{}"),
                ]
                for tag, r in rows
            ],
        )
    else:
        o("_no rows_")
        o()
    # a1: the restore rows and the err rows
    rows = _json_rows(o, "a1", run / "a1" / "stall" / "rows", provs.get("a1"))
    o(
        "### a1: the dehnen_error stall from the 1e8 restarts (`dehnen_error_stall.py`, `fused_memory_budget.py`)"
    )
    o()
    if rows:
        table = []
        for tag, r in rows:
            segs = [
                x.get("wall_s")
                for x in r.get("segments", [])
                if x.get("wall_s") is not None
            ]
            step = r.get("step_ms") or {}
            table.append(
                [
                    tag,
                    _f(r.get("total_s")),
                    _f(statistics.median(segs[1:]) if len(segs) > 1 else None),
                    _f(max(segs) if segs else None),
                    _f(step.get("min")),
                    _f(
                        max(r.get("peak_bytes") or [0]) / 2**30
                        if isinstance(r.get("peak_bytes"), list)
                        else r.get("final_peak_gib")
                    ),
                ]
            )
        o.table(
            [
                "row",
                "total s",
                "median step s (after the first)",
                "max step s",
                "err rows: step ms min",
                "peak GiB",
            ],
            table,
        )
    else:
        o("_no rows_")
        o()
    rows = _json_rows(o, "a3", run / "a3" / "timing" / "rows", provs.get("a3"))
    o(
        "### a3: summed vs legacy drift (`fused_memory_budget.py --runner --drift-order`)"
    )
    o()
    if rows:
        table = []
        for tag, r in rows:
            n = r.get("n") or (r.get("config") or {}).get("n") or 0
            peak = r.get("final_peak_gib") or r.get("peak_after_scan_gib") or 0.0
            st = r.get("step_ms") or {}
            table.append(
                [
                    tag,
                    n,
                    _f(st.get("min"), "{:.1f}"),
                    _f(st.get("median"), "{:.1f}"),
                    _f(peak),
                    _f(peak * 2**30 / n if n else None, "{:.1f}"),
                ]
            )
        o.table(["row", "N", "step ms min", "step ms median", "peak GiB", "B/p"], table)
    else:
        o("_no rows_")
        o()
    for job in (
        "m1_node1",
        "m1_node2",
        "m2g_node1",
        "m2g_node2",
        "mesc_node1",
        "mesc_node2",
    ):
        section_text(
            o,
            f"### {job} (`bench/slurm/{job.split('_')[0]}/summarize.py`)",
            run / job / "SUMMARY.md",
        )
    for tier in ("horeka-1card", "horeka-node"):
        path = run / "a4" / f"SUMMARY-{tier}.md"
        if path.exists():
            section_text(o, f"### a4: accuracy suite, tier {tier}", path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    run = args.run.resolve()
    provs = {job: _load(run / job / "provenance.json") for job in JOBS}
    o = Out()
    o(f"# HoreKa bundle run: {run.name}")
    o()
    o(
        f"Run directory `{run}`. Written by `bench/slurm/h0/summarize.py`. Timings are from exclusive nodes; "
        "every row carries its job's provenance in `rows.jsonl`."
    )
    o()
    section_provenance(o, run, provs)
    section_pins(o, run, provs.get("pins"))
    section_scaling(o, run, provs)
    section_grad(o, run, provs.get("grad"))
    section_s1b(o, run, provs.get("s1b"))
    section_auto(o, run, provs.get("auto"))
    section_folded(o, run, provs)
    (run / "SUMMARY.md").write_text("\n".join(o.lines) + "\n")
    with open(run / "rows.jsonl", "w") as fh:
        for r in o.rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"wrote {run / 'SUMMARY.md'} and {run / 'rows.jsonl'} ({len(o.rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
