"""The fused multi-GPU lane's scaling matrix on ONE dedicated node (S11 session 2).

Track T measured this matrix on four free A100-PCIE-40GB cards of a shared machine
(``docs/production_2026-10.md``, "Free-card measurements (T)"): there the mesh bought
capacity, not speed beyond two cards, because every card-to-card byte crossed PCIe.
This driver repeats it where the cards are the job's own and talk over NVLink (a
HoreKa accelerated node: four A100-40GB, ``bench/slurm/multigpu_scaling.sbatch``).

Every row is ``bench/multigpu_segment_gate.py`` in a FRESH process (XLA reads its
flags once; the peak is a per-process high-water mark) on the first ``k`` devices of
the allocation (``CUDA_VISIBLE_DEVICES`` as the scheduler set it, or every device
``nvidia-smi`` lists), with the fusion autotuner off and, on two or more cards, the
ragged-exchange flags (``jaccpot.distributed.fused.RAGGED_EXCHANGE_XLA_FLAGS``).
The tiers (``--tiers``, all by default):

=========  ==========================================================================
ab         migration on / off, interleaved repetitions (``--reps``), two and four
           cards: Plummer 8e6 (four cards: and 1.6e7), the disc + bulge 2e6 (four
           cards: and 8e6), the 25M disc + bulge, each MAC of ``--macs``
strong     fixed N on 1 / 2 / 4 cards, migrating: the 25M disc + bulge and Plummer
           8e6; one card = the gate's single-card lane (``--mode solo``) and its
           one-device mesh (``--ndev 1``)
weak       4e6 Plummer particles per card on 1 / 2 / 4 cards, migrating
memory     Plummer at 1e7, 2e7, 3e7, 4e7 particles per card on every card,
           migrating, 3 segments of 4 steps; stops at the first row that fails
           (out of memory) and records it
=========  ==========================================================================

Outputs under ``--out-dir``: ``TAG.json`` (the gate's record), ``TAG.log``,
``TAG.smi.txt`` (``nvidia-smi`` before and after: a row with another process on its
cards is labelled ``contended`` in the summary), ``rows.jsonl`` and ``SUMMARY.md``.
``--dry-run`` prints the commands; ``--summary`` rebuilds the summary from what is
there. A row whose JSON exists is skipped (``--redo`` runs it again), so a job that
ran out of time continues where it stopped.

    python bench/multigpu_scaling_matrix.py --out-dir OUT [--tiers ab,strong] [--reps 2]

Off this machine the gate needs two inputs it finds here by default: Odisseo's
``benchmark_multigpu/`` (the Plummer IC and the fp64 reference; ``BENCH_DIR``) and the
25M disc + bulge IC (``--disc-ic``, 573 MB, ``/export/scratch/tbuck/odisseo_ic/
disk_bulge_25m.npz`` here). The full matrix at ``--reps 2`` is ~90 rows, several hours
on a node; ``--tiers`` and ``--macs`` cut it down.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GATE = HERE / "multigpu_segment_gate.py"
AUTOTUNER_OFF = "--xla_gpu_experimental_enable_fusion_autotuner=false"
#: ``jaccpot.distributed.fused.RAGGED_EXCHANGE_XLA_FLAGS``, spelled out so the driver
#: needs no jax (the flags must reach the gate's process before jax starts)
RAGGED_FLAGS = (
    "--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false "
    "--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true "
    "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false"
)
TIERS = ("ab", "strong", "weak", "memory")
#: the summary's "cards free" column, from the row's nvidia-smi record (none: unknown)
_FREE = {True: "no (contended)", False: "yes"}


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tiers", default=",".join(TIERS))
    p.add_argument("--macs", default="dehnen_error,dehnen")
    p.add_argument("--reps", type=int, default=2, help="ab tier: repetitions per arm")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--segments", type=int, default=8)
    p.add_argument("--max-cards", type=int, default=4)
    p.add_argument(
        "--disc-ic",
        default=None,
        help="the gate's --disc-ic (the 25M disc + bulge .npz; default the gate's path)",
    )
    p.add_argument("--timeout", type=float, default=7200.0, help="seconds per row")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--summary", action="store_true", help="only rebuild SUMMARY.md")
    return p.parse_args()


def _devices() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [d for d in visible.split(",") if d]
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def _short(n: float) -> str:
    return f"{n:.2g}".replace("+0", "").replace("+", "")


def _matrix(args: argparse.Namespace, ncards: int) -> list[dict]:
    """The rows, in run order: each ``dict(tag, cards, gate_args)``."""
    macs = [m for m in args.macs.split(",") if m]
    tiers = set(args.tiers.split(","))
    run = ["--steps", str(args.steps), "--segments", str(args.segments)]
    run += ["--repartition-every", "1"]
    rows: list[dict] = []

    def add(tag: str, k: int, ic: str, n: float, mac: str, *extra: str) -> None:
        if k > ncards:
            return
        solo = "solo" in extra
        gate = ["--mode", "solo" if solo else "segment", "--ic", ic, "--n", str(int(n))]
        gate += ["--mac", mac]
        if ic == "disc" and n >= 2.5e7:
            # the L3 pin's 25M disc + bulge (the gate's disc formula: 7.6e-3)
            gate += ["--softening", "1.5e-3"]
        if ic == "disc" and args.disc_ic:
            gate += ["--disc-ic", args.disc_ic]
        if not solo:
            gate += run + ["--ndev", str(k)]
        gate += [e for e in extra if e != "solo"]
        if "--dump" not in gate:
            # the gate writes the run's positions by id there (required)
            gate += ["--dump", str(args.out_dir / f"dump_{tag}")]
        rows.append(dict(tag=tag, cards=k, gate_args=gate))

    abbrev = {"dehnen_error": "de", "dehnen": "geo"}
    if "ab" in tiers:
        for k, configs in (
            (2, (("plummer", 8e6), ("disc", 2e6), ("disc", 2.5e7))),
            (
                4,
                (
                    ("plummer", 8e6),
                    ("plummer", 1.6e7),
                    ("disc", 2e6),
                    ("disc", 8e6),
                    ("disc", 2.5e7),
                ),
            ),
        ):
            for ic, n in configs:
                for mac in macs:
                    for rep in "abcdefgh"[: args.reps]:
                        for arm in ("off", "on"):
                            tag = f"ab_{ic[0].upper()}{_short(n)}_{abbrev[mac]}_{k}c_{arm}_{rep}"
                            extra = ("--migrate",) if arm == "on" else ()
                            add(tag, k, ic, n, mac, *extra)
    if "strong" in tiers:
        for ic, n in (("disc", 2.5e7), ("plummer", 8e6)):
            for mac in macs:
                name = f"{ic[0].upper()}{_short(n)}_{abbrev[mac]}"
                # the single-card lane on the one-device mesh's particles (its dump)
                dump = str(args.out_dir / f"dump_strong_{name}_1c")
                add(f"strong_{name}_1c", 1, ic, n, mac, "--dump", dump)
                add(f"strong_{name}_solo", 1, ic, n, mac, "solo", "--dump", dump)
                for k in (2, 4):
                    add(f"strong_{name}_{k}c", k, ic, n, mac, "--migrate")
    if "weak" in tiers:
        for k in (1, 2, 4):
            extra = ("--migrate",) if k > 1 else ()
            add(f"weak_P4e6pc_{k}c", k, "plummer", 4e6 * k, macs[0], *extra)
    if "memory" in tiers:
        k = min(ncards, args.max_cards)
        for per_card in (1e7, 2e7, 3e7, 4e7):
            n = per_card * k
            add(
                f"memory_P{_short(per_card)}pc_{k}c",
                k,
                "plummer",
                n,
                macs[0],
                "--migrate",
                "--steps",
                "4",
                "--segments",
                "3",
            )
    return rows


def _smi(devices: list[str]) -> dict:
    """Memory and utilisation per card, and the compute processes on these cards."""
    q = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    ).stdout
    cards = {}
    for line in q.splitlines():
        idx, uuid, mem, util = (x.strip() for x in line.split(","))
        if idx in devices:
            cards[uuid] = dict(index=idx, memory_mib=int(mem), util=int(util))
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    ).stdout
    procs = []
    for line in apps.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[1] in cards:
            procs.append(dict(pid=int(parts[0]), card=cards[parts[1]]["index"]))
    return dict(cards=list(cards.values()), processes=procs)


def _run_row(row: dict, devices: list[str], args: argparse.Namespace) -> dict:
    out = args.out_dir / f"{row['tag']}.json"
    cards = devices[: row["cards"]]
    flags = AUTOTUNER_OFF + (" " + RAGGED_FLAGS if row["cards"] >= 2 else "")
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(cards),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_FLAGS=(env.get("XLA_FLAGS", "") + " " + flags).strip(),
    )
    cmd = [sys.executable, str(GATE), *row["gate_args"], "--out", str(out)]
    if args.dry_run:
        print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {' '.join(cmd)}")
        return dict(tag=row["tag"], dry_run=True)
    before = _smi(cards)
    begun = time.time()
    with open(args.out_dir / f"{row['tag']}.log", "w") as log:
        try:
            rc = subprocess.run(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
            ).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    after = _smi(cards)
    others = {p["pid"] for p in before["processes"] + after["processes"]}
    record = dict(
        tag=row["tag"],
        cards=cards,
        rc=rc,
        wall_s=time.time() - begun,
        gate_args=row["gate_args"],
        smi_before=before,
        smi_after=after,
        # the gate's own process is gone by the after-snapshot; any pid in either
        # snapshot belonged to someone else
        contended=bool(others),
    )
    (args.out_dir / f"{row['tag']}.smi.txt").write_text(json.dumps(record, indent=1))
    return record


def _needs_index() -> dict:
    """Positions in the gate's needs vector (NEEDS_FIELDS then CROSS_NEEDS_FIELDS)."""
    from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS
    from jaccpot.runtime.fused_topology import NEEDS_FIELDS

    base = len(NEEDS_FIELDS)
    return dict(
        cross_near=base + CROSS_NEEDS_FIELDS.index("export_near_cap"),
        shipped=base + CROSS_NEEDS_FIELDS.index("send_particle_cap"),
    )


def _row_summary(tag: str, out_dir: Path, index: dict) -> dict:
    d = json.loads((out_dir / f"{tag}.json").read_text())
    smi_path = out_dir / f"{tag}.smi.txt"
    smi = json.loads(smi_path.read_text()) if smi_path.exists() else {}
    r = dict(
        tag=tag,
        mode=d.get("mode"),
        ic=d.get("ic"),
        n=d.get("n"),
        ndev=d.get("ndev", 1),
        mac=d.get("mac"),
        migrate=bool(d.get("migrate")),
        contended=smi.get("contended"),
    )
    if d.get("mode") == "solo":
        start = d.get("solo_start") or {}
        r.update(
            step_ms=min(start["force_ms"]) if start.get("force_ms") else None,
            bpp=max(d.get("peak_bytes", [0])) / d["n"],
            rel_l2=start.get("rel_l2"),
        )
        return r
    seg = d.get("segment_s", [])
    steps = d.get("steps", 1)
    reports = d.get("reports", [])
    r.update(
        # segments after the third (an uncached process compiles in the first three)
        step_ms=1e3 * float(np.median(seg[3:])) / steps if len(seg) > 3 else None,
        force_ms=min(d["force_only_ms"]) if d.get("force_only_ms") else None,
        bpp=max(d.get("peak_bytes_per_particle") or [0]),
        rel_l2=d.get("rel_l2_end"),
        retries=sum(rep.get("retries", 0) for rep in reports),
        bits=sorted({rep.get("bits", 0) for rep in reports}),
        cross_near=max(
            (rep["needs"][index["cross_near"]] for rep in reports), default=None
        ),
        shipped=max((rep["needs"][index["shipped"]] for rep in reports), default=None),
        migrated_per_step=max(
            (rep.get("migrated", 0) / max(rep.get("steps", 1), 1) for rep in reports),
            default=0,
        ),
    )
    return r


def _fmt(x, spec: str = "{:.2f}") -> str:
    return "-" if x is None else spec.format(x)


def _summary(out_dir: Path) -> None:
    index = _needs_index()
    rows = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            rows.append(_row_summary(path.stem, out_dir, index))
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            rows.append(dict(tag=path.stem, error=repr(exc)))
    with open(out_dir / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    lines = [
        "| row | cards | MAC | migrate | step ms | force ms | peak B/p (max device) | rel-L2 end "
        "| retries | bits | cross near | shipped | migrants/step | cards free |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['tag']} | unreadable: {r['error']} |")
            continue
        lines.append(
            f"| {r['tag']} | {r['ndev']}{' (solo)' if r['mode'] == 'solo' else ''} | {r['mac']} "
            f"| {'on' if r['migrate'] else 'off'} | {_fmt(r.get('step_ms'))} "
            f"| {_fmt(r.get('force_ms'))} | {_fmt(r.get('bpp'), '{:.0f}')} "
            f"| {_fmt(r.get('rel_l2'), '{:.4e}')} | {r.get('retries', '-')} | {r.get('bits', '-')} "
            f"| {_fmt(r.get('cross_near'), '{:,.0f}')} | {_fmt(r.get('shipped'), '{:,.0f}')} "
            f"| {_fmt(r.get('migrated_per_step'), '{:,.0f}')} "
            f"| {_FREE.get(r.get('contended'), '?')} |"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    args = _args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.summary:
        _summary(args.out_dir)
        return 0
    devices = _devices()
    ncards = min(len(devices), args.max_cards)
    rows = _matrix(args, ncards)
    print(f"{len(rows)} rows on cards {devices[:ncards]}", flush=True)
    memory_failed = False
    for row in rows:
        if (args.out_dir / f"{row['tag']}.json").exists() and not args.redo:
            continue
        if memory_failed and row["tag"].startswith("memory_"):
            print(f"skip {row['tag']}: a smaller memory row failed", flush=True)
            continue
        record = _run_row(row, devices, args)
        print(f"{row['tag']}: rc {record.get('rc')}", flush=True)
        if row["tag"].startswith("memory_") and record.get("rc") not in (0, None):
            memory_failed = True
    if not args.dry_run:
        _summary(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
