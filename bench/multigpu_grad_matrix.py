"""WP-G G5: the multi-GPU force's reverse-mode gradient matrix on ONE dedicated node.

Every row is ``bench/multigpu_grad_gate.py`` in a FRESH process (XLA reads its flags
once; the device peak is a per-process high-water mark) on the first ``k`` devices
of the allocation (``CUDA_VISIBLE_DEVICES`` as the scheduler set it, or every device
``nvidia-smi`` lists), with the fusion autotuner off and, on two or more cards, the
ragged-exchange flags (``bench/_ragged_exchange_flags.py``). Written for a HoreKa
accelerated node (four A100-40GB on NVLink, ``bench/slurm/multigpu_grad.sbatch``).
The operating point is ``--preset`` (``galaxy``: ``preset="galaxy"``, dehnen_error
1e-4 at p6, the production one; ``geometric``: the G1 pin's ``mac_type="dehnen"`` at
p5, with ``G`` and ``eps2`` differentiated too). The tiers (``--tiers``, all by
default):

=========  ==========================================================================
weak       Plummer at 2e5, 1e6, 4e6, 1.6e7 particles per card on 1 / 2 / 4 cards
           (the one-device mesh at 1), plus the single card's provider path
           (``--mode solo``) at each N per card: the ratio grad / force and the
           peak B/p against N per card and cards
strong     fixed N on solo / 1 / 2 / 4 cards: Plummer 1.6e7 and 5e7, the 25M disc +
           bulge's first 1.6e7 rows; rows identical to a weak row run once
memory     Plummer at 2e7, 3e7, 4e7, 5e7 (``--memory-per-card``) per card on every
           card, ``jax.grad`` only, no MG1 (BFC preallocated at 0.95, as S9's 1e8
           rows): stops at the first row that fails and records it (``oom``)
check      accuracy at N = 262,144 with ``--check`` (the gradient by global id vs
           the single card's S8 gradient and an fp64 direct sum on 2048 weighted
           targets): Plummer and the disc, seeds 0-2, on 1 / 2 / 4 cards (flat in
           cards?); and the geometric MAC at p = 3, 4, 5, 6 on 2 cards with ``G`` and
           ``eps2`` (MG9) -- the error against p
=========  ==========================================================================

Rows: weak 16, strong 12 (9 not already in weak), memory 4, check 22: 51 runs on a
four-card node. Expected wall (an ESTIMATE, nothing of the multi-card gradient is
timed on dedicated cards yet): per row the IC and the decomposition, the eager
caps-only prepare, the cross sizing's 1-3 start-force compiles, and five program
compiles (force, differentiable forward, grad, value_and_grad; the check rows also
the single card's gradient), 1-4 min each on an A100 from the S9 / S11 rows, plus
the timed calls (``--reps`` 5 of four programs, seconds at 5e7): ~6 min a check row,
~8-12 min a weak / strong row, ~15 min a memory row -> ~7.5 h (25 x 10 + 4 x 15
+ 22 x 6 min); ``--time=12:00:00``
leaves room, and a resubmission continues where the last one stopped.

Outputs under ``--out-dir``: ``TAG.json`` (the gate's record, written after every
phase: a row that died has its last phase), ``TAG.log``, ``TAG.smi.txt`` (the
driver's record: return code, wall, ``nvidia-smi`` before and after -- a row with
another process on its cards is ``contended`` in the summary -- and ``oom`` when the
log shows an allocation failure), ``matrix.json`` (the rows and the aliases of
shared rows), ``rows.jsonl`` and ``SUMMARY.md``. ``--dry-run`` prints the commands;
``--summary`` rebuilds the summary from what is there. A row that completed or
already ran (its ``.smi.txt`` exists) is skipped (``--retry-failed`` reruns the
failed ones, ``--redo`` all), so a job that ran out of time continues.

    python bench/multigpu_grad_matrix.py --out-dir OUT [--tiers weak,check] [--reps 5]

Off this machine the gate needs Odisseo's ``benchmark_multigpu/`` (the Plummer IC;
``BENCH_DIR``) and the 25M disc + bulge IC (``--disc-ic``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _ragged_exchange_flags import RAGGED_EXCHANGE_XLA_FLAGS  # noqa: E402
from multigpu_scaling_matrix import _devices, _smi  # noqa: E402

GATE = HERE / "multigpu_grad_gate.py"
AUTOTUNER_OFF = "--xla_gpu_experimental_enable_fusion_autotuner=false"
TIERS = ("weak", "strong", "memory", "check")
WEAK_PER_CARD = (2e5, 1e6, 4e6, 1.6e7)
STRONG = (("plummer", 1.6e7), ("disc", 1.6e7), ("plummer", 5e7))
CHECK_N = 262_144
CHECK_ORDERS = (3, 4, 5, 6)
#: the memory rows' allocator (S9's 1e8 rows: BFC could only place the gradient
#: preallocated); ``peak_bytes_in_use`` is the allocator's own count either way
MEMORY_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
}
_OOM = re.compile(r"RESOURCE_EXHAUSTED|out of memory|OOM when allocating", re.I)
#: the summary's "cards free" column, from the row's nvidia-smi record
_FREE = {True: "no (contended)", False: "yes"}


def _args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tiers", default=",".join(TIERS))
    p.add_argument("--preset", choices=("galaxy", "geometric"), default="galaxy")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--max-cards", type=int, default=4)
    p.add_argument(
        "--memory-per-card",
        default="2e7,3e7,4e7,5e7",
        help="memory tier: Plummer particles per card, ascending (stops at the first "
        "row that fails)",
    )
    p.add_argument("--disc-ic", default=None, help="the gate's --disc-ic")
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds per row")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--redo", action="store_true", help="rerun every row")
    p.add_argument(
        "--retry-failed", action="store_true", help="rerun rows that ran and failed"
    )
    p.add_argument("--summary", action="store_true", help="only rebuild SUMMARY.md")
    return p.parse_args(argv)


def _on_cpu() -> bool:
    """A CPU dry run (``JAX_PLATFORMS=cpu``): the cards are forced host devices."""
    return os.environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu"


def _short(n: float) -> str:
    return f"{n:.2g}".replace("+0", "").replace("+", "")


def matrix(args: argparse.Namespace, ncards: int) -> tuple[list[dict], dict]:
    """The rows in run order (``dict(tag, tier, cards, gate_args, env)``) and the
    aliases ``{dropped tag: the identical row's tag}``."""
    tiers = set(args.tiers.split(","))
    unknown = tiers - set(TIERS)
    if unknown:
        raise SystemExit(f"unknown tiers {sorted(unknown)}; known {TIERS}")
    preset = ["--galaxy"] if args.preset == "galaxy" else ["--params"]
    rows: list[dict] = []
    seen: dict = {}
    aliases: dict = {}

    def add(tag, tier, k, ic, n, *extra, env=None, base=None, reps=None) -> None:
        if k > ncards:
            return
        solo = "solo" in extra
        gate = ["--mode", "solo" if solo else "mesh", "--ndev", "1" if solo else str(k)]
        gate += ["--ic", ic, "--n", str(int(n))]
        gate += list(preset if base is None else base)
        gate += ["--reps", str(reps or args.reps)]
        if ic == "disc" and args.disc_ic:
            gate += ["--disc-ic", args.disc_ic]
        gate += [e for e in extra if e != "solo"]
        key = (tuple(gate), tuple(sorted((env or {}).items())))
        if key in seen:
            aliases[tag] = seen[key]
            return
        seen[key] = tag
        rows.append(dict(tag=tag, tier=tier, cards=k, gate_args=gate, env=env or {}))

    if "weak" in tiers:
        for per in WEAK_PER_CARD:
            add(f"weak_P{_short(per)}pc_solo", "weak", 1, "plummer", per, "solo")
            for k in (1, 2, 4):
                add(f"weak_P{_short(per)}pc_{k}c", "weak", k, "plummer", per * k)
    if "strong" in tiers:
        for ic, n in STRONG:
            name = f"{ic[0].upper()}{_short(n)}"
            add(f"strong_{name}_solo", "strong", 1, ic, n, "solo")
            for k in (1, 2, 4):
                add(f"strong_{name}_{k}c", "strong", k, ic, n)
    if "memory" in tiers:
        k = min(ncards, args.max_cards)
        for per in (float(x) for x in args.memory_per_card.split(",") if x):
            add(
                f"memory_P{_short(per)}pc_{k}c",
                "memory",
                k,
                "plummer",
                per * k,
                "--grad-only",
                "--no-forward",
                env=MEMORY_ENV,
                reps=3,
            )
    if "check" in tiers:
        for ic in ("plummer", "disc"):
            for seed in (0, 1, 2):
                for k in (1, 2, 4):
                    add(
                        f"check_{ic[0].upper()}_s{seed}_{k}c",
                        "check",
                        k,
                        ic,
                        CHECK_N,
                        "--seed",
                        str(seed),
                        "--check",
                        reps=3,
                    )
        for order in CHECK_ORDERS:
            add(
                f"check_P_geo_p{order}_2c",
                "check",
                2,
                "plummer",
                CHECK_N,
                "--order",
                str(order),
                "--check",
                base=["--params"],
                reps=3,
            )
    return rows, aliases


def _row_env(row: dict, cards: list) -> dict:
    flags = AUTOTUNER_OFF + (
        " " + RAGGED_EXCHANGE_XLA_FLAGS if row["cards"] >= 2 else ""
    )
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(cards),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_FLAGS=(env.get("XLA_FLAGS", "") + " " + flags).strip(),
        PYTHONUNBUFFERED="1",
    )
    env.update(row["env"])
    if _on_cpu():
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _command(row: dict, out: Path) -> list:
    cmd = [sys.executable, str(GATE), *row["gate_args"], "--out", str(out)]
    if _on_cpu():
        cmd.append("--cpu-smoke")
    return cmd


def _complete(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text()).get("complete"))
    except (OSError, ValueError):
        return False


def _run_row(row: dict, devices: list, args: argparse.Namespace) -> dict:
    out = args.out_dir / f"{row['tag']}.json"
    cards = devices[: row["cards"]]
    env = _row_env(row, cards)
    cmd = _command(row, out)
    if args.dry_run:
        extra = " ".join(f"{k}={v}" for k, v in row["env"].items())
        print(
            f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {extra} {' '.join(cmd)}".replace(
                "  ", " "
            )
        )
        return dict(tag=row["tag"], dry_run=True)
    before = _smi(cards) if not _on_cpu() else dict(cards=[], processes=[])
    begun = time.time()
    log_path = args.out_dir / f"{row['tag']}.log"
    with open(log_path, "w") as log:
        try:
            rc = subprocess.run(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
            ).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    after = _smi(cards) if not _on_cpu() else dict(cards=[], processes=[])
    others = {p["pid"] for p in before["processes"] + after["processes"]}
    try:
        oom = bool(_OOM.search(log_path.read_text(errors="replace")))
    except OSError:
        oom = False
    record = dict(
        tag=row["tag"],
        tier=row["tier"],
        cards=cards,
        rc=rc,
        oom=oom and rc != 0,
        wall_s=time.time() - begun,
        gate_args=row["gate_args"],
        env=row["env"],
        smi_before=before,
        smi_after=after,
        # the gate's process is gone by the after-snapshot: any pid belonged to
        # someone else
        contended=bool(others),
    )
    (args.out_dir / f"{row['tag']}.smi.txt").write_text(json.dumps(record, indent=1))
    return record


# --------------------------------------------------------------------------
# The summary.
# --------------------------------------------------------------------------


def _min2(t: dict, a: str, b: str):
    vals = [t[k][0] for k in (a, b) if k in t]
    return min(vals) if vals else None


def row_summary(tag: str, out_dir: Path) -> dict:
    """One row of ``rows.jsonl`` from ``TAG.json`` and ``TAG.smi.txt`` (either may
    be missing)."""
    jpath, spath = out_dir / f"{tag}.json", out_dir / f"{tag}.smi.txt"
    d = json.loads(jpath.read_text()) if jpath.exists() else {}
    s = json.loads(spath.read_text()) if spath.exists() else {}
    r = dict(
        tag=tag,
        tier=s.get("tier", tag.split("_")[0]),
        mode=d.get("mode"),
        ic=d.get("ic"),
        n=d.get("n"),
        ndev=d.get("ndev"),
        n_per_card=d.get("n_per_card"),
        mac=d.get("mac"),
        order=d.get("order"),
        complete=bool(d.get("complete")),
        phase=d.get("phase"),
        rc=s.get("rc"),
        oom=s.get("oom"),
        error=d.get("error"),
        # the driver's nvidia-smi record, else the gate's own
        contended=s.get("contended", d.get("contended")),
    )
    if not d:
        return r
    t = d.get("time_ms", {})
    force = _min2(t, "force", "force_b")
    main = _min2(t, "main", "main_b")
    ma = d.get("memory_analysis") or {}
    grad_ma = ma.get("grad") or {}
    vg_ma = ma.get("value_and_grad") or {}
    cross = d.get("cross") or {}
    check = d.get("check") or {}
    r.update(
        mg1_differ=(d.get("MG1") or {}).get("differing"),
        bits_ok=(d.get("force") or {}).get("bits_ok"),
        grad_ok=(d.get("grad_bits") or {}).get("grad_ok"),
        finite=d.get("grad_finite"),
        cross_far=max(cross["far_pairs"]) if cross.get("far_pairs") else None,
        cross_near=max(cross["near_pairs"]) if cross.get("near_pairs") else None,
        cross_nonempty=cross.get("nonempty"),
        force_ms=force,
        main_ms=main,
        main_is=t.get("main_is"),
        grad_ms=(
            t["grad"][0]
            if "grad" in t
            else (main if t.get("main_is") == "grad" else None)
        ),
        ratio_vg=d.get("ratio_value_and_grad_over_force"),
        ratio_grad=d.get("ratio_grad_over_force"),
        peak_bpp=max(d["peak_B_per_p"]) if d.get("peak_B_per_p") else None,
        grad_program_bpp=grad_ma.get("total_B_per_p"),
        vg_program_bpp=vg_ma.get("total_B_per_p"),
        avsa=d.get("grad_AvsA_differing"),
        compile_s=sum((d.get("compile_s") or {}).values()) or None,
        x_vs_direct=check.get("x_mesh_vs_direct"),
        x_single_vs_direct=check.get("x_single_vs_direct"),
        x_ratio=check.get("x_ratio"),
        m_ratio=check.get("m_ratio"),
        mg9_rel=(check.get("MG9") or {}).get("rel"),
    )
    return r


def _fmt(x, spec: str = "{:.2f}") -> str:
    return "-" if x is None else spec.format(x)


def _status(r: dict) -> str:
    if r["complete"]:
        return "ok"
    if r.get("oom"):
        return f"OOM ({r.get('phase') or 'before prepare'})"
    if r.get("error"):
        return str(r["error"])[:60]
    if r.get("rc") is not None:
        return f"rc {r['rc']} at {r.get('phase') or 'start'}"
    return "running?"


def summary(out_dir: Path) -> list:
    tags = sorted(
        {p.name[: -len(".smi.txt")] for p in out_dir.glob("*.smi.txt")}
        | {p.stem for p in out_dir.glob("*.json") if p.name != "matrix.json"}
    )
    rows = []
    for tag in tags:
        try:
            rows.append(row_summary(tag, out_dir))
        except (KeyError, ValueError, TypeError) as exc:
            rows.append(dict(tag=tag, unreadable=repr(exc)))
    with open(out_dir / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    lines = [
        "# Multi-GPU gradient matrix (WP-G G5)",
        "",
        "Timings and ratios are claims only on rows whose cards were free "
        "(last column). Ratios: min of the program over min of the production force.",
        "",
        "| row | cards | N/card | status | MG1 diff | bits / grad ok | cross far / near "
        "| force ms | v&g ms | grad ms | v&g / force | grad / force | peak B/p "
        "| grad prog B/p | A-vs-A | x err vs direct (ratio to 1 card) | cards free |",
        "| --- " * 17 + "|",
    ]
    for r in rows:
        if "unreadable" in r:
            lines.append(f"| {r['tag']} | unreadable: {r['unreadable']} |")
            continue
        cards = (
            "-"
            if r.get("ndev") is None
            else f"{r['ndev']}{' (solo)' if r.get('mode') == 'solo' else ''}"
        )
        cross = (
            "-"
            if r.get("cross_far") is None
            else f"{r['cross_far']:,} / {r['cross_near']:,}"
        )
        err = (
            "-"
            if r.get("x_vs_direct") is None
            else f"{r['x_vs_direct']:.3e} ({_fmt(r.get('x_ratio'))})"
        )
        lines.append(
            f"| {r['tag']} | {cards} | {_fmt(r.get('n_per_card'), '{:.3g}')} | {_status(r)} "
            f"| {_fmt(r.get('mg1_differ'), '{:d}')} | {r.get('bits_ok', '-')} / {r.get('grad_ok', '-')} "
            f"| {cross} | {_fmt(r.get('force_ms'))} "
            f"| {_fmt(r.get('main_ms') if r.get('main_is') == 'value_and_grad' else None)} "
            f"| {_fmt(r.get('grad_ms'))} | {_fmt(r.get('ratio_vg'))} | {_fmt(r.get('ratio_grad'))} "
            f"| {_fmt(r.get('peak_bpp'), '{:.0f}')} | {_fmt(r.get('grad_program_bpp'), '{:.0f}')} "
            f"| {r.get('avsa', '-')} | {err} | {_FREE.get(r.get('contended'), '?')} |"
        )
    manifest = out_dir / "matrix.json"
    if manifest.exists():
        aliases = json.loads(manifest.read_text()).get("aliases") or {}
        if aliases:
            lines += [
                "",
                "Rows shared between tiers (run once, under the second name):",
                "",
            ]
            lines += [f"- {a} = {b}" for a, b in sorted(aliases.items())]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rows


def main(argv=None) -> int:
    args = _args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.summary:
        summary(args.out_dir)
        return 0
    devices = [str(i) for i in range(args.max_cards)] if _on_cpu() else _devices()
    ncards = min(len(devices), args.max_cards)
    rows, aliases = matrix(args, ncards)
    print(
        f"{len(rows)} rows on cards {devices[:ncards]} ({len(aliases)} shared)",
        flush=True,
    )
    if not args.dry_run:
        (args.out_dir / "matrix.json").write_text(
            json.dumps(dict(rows=rows, aliases=aliases, argv=sys.argv[1:]), indent=1)
        )
    memory_failed = False
    for row in rows:
        tag = row["tag"]
        done = _complete(args.out_dir / f"{tag}.json")
        tried = (args.out_dir / f"{tag}.smi.txt").exists()
        if (
            not args.redo
            and not args.dry_run
            and (done or (tried and not args.retry_failed))
        ):
            if row["tier"] == "memory" and not done:
                memory_failed = True
            continue
        if memory_failed and row["tier"] == "memory":
            print(f"skip {tag}: a smaller memory row failed", flush=True)
            continue
        record = _run_row(row, devices, args)
        if not args.dry_run:
            print(
                f"{tag}: rc {record.get('rc')} oom {record.get('oom')} {record.get('wall_s', 0):.0f} s",
                flush=True,
            )
        if row["tier"] == "memory" and record.get("rc") not in (0, None):
            memory_failed = True
    if not args.dry_run:
        summary(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
