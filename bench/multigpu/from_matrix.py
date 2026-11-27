"""Figures 08 and 09 data: convert a scaling-matrix run into the paper's artifacts.

The paper does not keep a multi-GPU harness of its own. Strong and weak scaling are
measured by the library's scaling matrix (``bench/multigpu_scaling_matrix.py``: every
row is ``bench/multigpu_segment_gate.py`` in a fresh process on the fused mesh lane,
``MeshSegmentRunner`` + ``FusedMeshForce``, ``preset="galaxy"``), which is also what
the HoreKa bundle runs (``bench/slurm/h0``, job ``scaling_a``). This script reads a
matrix ``--out-dir`` and writes

* ``bench/results/multigpu/strong_scaling.json`` (rows ``strong_*``: fixed N on
  1 / 2 / 4 cards, plus the single-card lane's reference row ``*_solo``), and
* ``bench/results/multigpu/weak_scaling.json`` (rows ``weak_*``: fixed N per card),

through ``examples/jaccpot_paper/common/jsonio``. It imports no device and measures
nothing; every number in its output is read from the matrix's files.

What it reads, per row ``TAG``
------------------------------
``TAG.json``
    The gate's record (segment mode: ``segment_s``, ``steps``, ``force_only_ms``,
    ``reports`` / ``start_report`` with the overflow ``bits`` and ``needs``,
    ``rel_l2_start`` / ``rel_l2_end``, ``peak_bytes_per_particle``, ``counts0``,
    ``devices``; solo mode: ``solo_start`` / ``solo_end`` with ``ok``, ``rel_l2`` and
    ``force_ms``). The gate writes it only when the row completed.
``TAG.smi.txt``
    The matrix's own record of the row: the return code, the cards, and
    ``contended`` (another process on the row's cards before or after it ran).
    A row whose process failed has this file and no ``TAG.json``; it is recorded as
    an invalid point with its return code and the tail of ``TAG.log``.
``rows.jsonl``
    The matrix's summary (``--summary``); copied into each record as
    ``matrix_summary`` when present, so a figure value can be checked against
    ``SUMMARY.md``.
``provenance.json``
    ``bench/slurm/h0/provenance.py``'s header (the bundle writes it in the job
    directory, the matrix's parent; ``bench/multigpu/run_matrix.py`` writes it in the
    out-dir). The device type is its ``nvidia-smi`` name column and the interconnect
    is read from its ``nvidia-smi topo -m`` matrix (``NV#``: NVLink; ``PIX`` /
    ``PXB`` / ``PHB`` / ``NODE`` / ``SYS``: PCIe). Without it both stay ``unknown``
    unless ``--device-kind`` / ``--interconnect`` say otherwise, in which case the
    artifact records that the label was asserted by the operator, not measured.

Validity
--------
A point is plotted only when ``valid`` is true, and ``valid`` requires all of

* the row completed (``TAG.json`` exists; return code 0 when recorded);
* the capacity verdict held: every kept run's overflow ``bits`` (the start force
  and every segment) clear of ``fused_topology.OVERFLOW_FAIL_MASK``; for a solo
  row ``solo_start.ok`` and ``solo_end.ok``;
* a finite end-of-run rel-L2 against the gate's fp64 direct sum;
* a steady-state time exists (more than three segments, as the matrix summary);
* the cards were free: ``contended`` is false in ``TAG.smi.txt``. A contended row
  and a row with no card record are kept, with the reason, but not plotted;
* the row ran on a GPU (``devices`` in the gate record). A CPU smoke's rows are
  structurally complete and always invalid: emulated devices share one CPU.

``capacity_ok``, ``contended`` and ``invalid_reasons`` are carried separately, so
an overflowed point is visible in the artifact as the finding it is.

Times
-----
``step_s``: median of the gate's ``segment_s`` after the third segment (an uncached
process compiles in the first three) divided by ``steps``: one velocity-Verlet
step including the force, the drift, the migration and the repartition between
segments (the matrix summary's ``step ms``). ``step_s_gate``: the gate's own
``step_ms_steady`` (segments after the second). ``force_s``: the minimum of
``force_only_ms`` (the start program alone at the final positions: force without
integrator). ``median_s`` is ``step_s``, kept so the September notebooks' key
still resolves; it is a STEP time now, not a force-evaluation time.

Usage
-----
::

    python -m bench.multigpu.from_matrix --matrix-dir H0_RUN/scaling_a/matrix
    python -m bench.multigpu.from_matrix --matrix-dir OUT --tiers strong \\
        --strong-out multigpu/strong_scaling_local.json

CPU smoke (a tiny real matrix on two emulated devices, then the conversion;
every point comes out invalid by design)::

    JAX_PLATFORMS=cpu H0_SMOKE_CPU=1 JAX_ENABLE_X64=1 \\
    PYTHONPATH=$PWD/bench/slurm/h0/smoke_site:$PWD \\
    BENCH_DIR=$PWD/bench/slurm/h0/harness/benchmark_multigpu \\
    python bench/multigpu_scaling_matrix.py --out-dir /tmp/m --tiers strong,weak \\
        --macs dehnen --smoke-n 2048 --steps 2 --segments 5 --max-cards 2 \\
        --rows '^(strong_P8e6_geo_(1c|solo|2c)|weak_P4e6pc_(1c|2c))$'
    python -m bench.multigpu.from_matrix --matrix-dir /tmp/m \\
        --strong-out /tmp/strong.json --weak-out /tmp/weak.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import statistics
import sys
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

STRONG_OUT = "multigpu/strong_scaling.json"
WEAK_OUT = "multigpu/weak_scaling.json"

#: What ``bench/multigpu_segment_gate.py`` passes to ``MeshSegmentRunner.fused`` /
#: ``FusedForceProvider.prepare`` as literals in its source; its record does not
#: carry them. Recorded under ``gate_constants`` and flagged as read from source.
GATE_CONSTANTS = {
    "preset": "galaxy",
    "leaf_size": 64,
    "max_order": 6,
    "theta": 0.8,
    "cross": "library (cross_caps='auto') unless the record says otherwise",
}

_TOPO_LINK = re.compile(r"^(NV\d+|PIX|PXB|PHB|NODE|SYS|X)$")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="See the module docstring for the validity rules and the CPU smoke.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--matrix-dir",
        type=pathlib.Path,
        required=True,
        help="a bench/multigpu_scaling_matrix.py --out-dir (TAG.json, TAG.smi.txt)",
    )
    p.add_argument("--tiers", default="strong,weak", help="which artifacts to write")
    p.add_argument("--strong-out", default=STRONG_OUT)
    p.add_argument("--weak-out", default=WEAK_OUT)
    p.add_argument(
        "--provenance",
        type=pathlib.Path,
        default=None,
        help="provenance.json (default: the matrix dir's, else its parent's)",
    )
    p.add_argument(
        "--device-kind",
        default=None,
        help="only without a provenance.json: the card type, recorded as asserted",
    )
    p.add_argument(
        "--interconnect",
        default=None,
        help="only without a provenance.json: e.g. 'NVLink' or 'PCIe', recorded as "
        "asserted",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# provenance: device type and interconnect, read, never guessed
# --------------------------------------------------------------------------- #


def _interconnect_from_topo(topo: str) -> Optional[dict[str, Any]]:
    """Classify ``nvidia-smi topo -m`` GPU-to-GPU links.

    Parameters
    ----------
    topo : str
        The command's output.

    Returns
    -------
    dict or None
        ``{"kind": "NVLink" | "PCIe" | "mixed", "links": [...]}`` from the GPU rows'
        GPU columns, or ``None`` when no GPU-to-GPU cell could be read.
    """
    lines = [ln.rstrip() for ln in (topo or "").splitlines() if ln.strip()]
    if not lines:
        return None
    header = lines[0].split()
    gpu_cols = [i for i, h in enumerate(header) if re.fullmatch(r"GPU\d+", h)]
    links: list[str] = []
    for ln in lines[1:]:
        cells = ln.split()
        if not cells or not re.fullmatch(r"GPU\d+", cells[0]):
            continue
        # row label first, then one cell per header column
        for col in gpu_cols:
            if col + 1 < len(cells):
                cell = cells[col + 1]
                if _TOPO_LINK.match(cell) and cell != "X":
                    links.append(cell)
    if not links:
        return None
    nv = [c for c in links if c.startswith("NV")]
    kind = "NVLink" if len(nv) == len(links) else ("PCIe" if not nv else "mixed")
    return {"kind": kind, "links": sorted(set(links))}


def _provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Device type and interconnect from ``provenance.json``, or labelled unknown."""
    candidates = (
        [args.provenance]
        if args.provenance
        else [
            args.matrix_dir / "provenance.json",
            args.matrix_dir.parent / "provenance.json",
        ]
    )
    for path in candidates:
        if path and path.exists():
            prov = json.loads(path.read_text())
            gpus = prov.get("gpus") or {}
            rows = str(gpus.get("rows") or "")
            names = sorted(
                {
                    parts[1].strip()
                    for parts in (r.split(",") for r in rows.splitlines())
                    if len(parts) > 1 and not rows.startswith("unavailable")
                }
            )
            link = _interconnect_from_topo(str(gpus.get("topology") or ""))
            return {
                "source": str(path),
                "device_kind": ", ".join(names) if names else "unknown",
                "interconnect": link["kind"] if link else "unknown",
                "interconnect_links": link["links"] if link else None,
                "host": prov.get("host"),
                "jaccpot_git": (prov.get("git") or {}).get("jaccpot"),
                "yggdrax_git": (prov.get("git") or {}).get("yggdrax"),
                "packages": prov.get("packages"),
                "time": prov.get("time"),
                "asserted_by_operator": False,
            }
    asserted = bool(args.device_kind or args.interconnect)
    return {
        "source": None,
        "device_kind": args.device_kind or "unknown",
        "interconnect": args.interconnect or "unknown",
        "interconnect_links": None,
        "asserted_by_operator": asserted,
    }


# --------------------------------------------------------------------------- #
# one row
# --------------------------------------------------------------------------- #


def _library_layout() -> tuple[int, tuple[str, ...]]:
    """``OVERFLOW_FAIL_MASK`` and the needs vector's field names, from the library.

    Imported, not copied, so a changed layout cannot be misread. No device is used
    (``JAX_PLATFORMS`` defaults to ``cpu`` here, so a converter on a GPU node never
    initialises a card).
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS
    from jaccpot.runtime.fused_topology import NEEDS_FIELDS, OVERFLOW_FAIL_MASK

    return int(OVERFLOW_FAIL_MASK), (*NEEDS_FIELDS, *CROSS_NEEDS_FIELDS)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _platform(devices: list[str]) -> str:
    """``gpu`` / ``cpu`` / ``unknown`` from the gate record's device strings."""
    joined = " ".join(devices or []).lower()
    if "cuda" in joined or "gpu" in joined or "rocm" in joined:
        return "gpu"
    if "cpu" in joined:
        return "cpu"
    return "unknown"


def _tail(path: pathlib.Path, n: int = 2000) -> Optional[str]:
    return path.read_text(errors="replace")[-n:] if path.exists() else None


def _row(
    tag: str,
    mdir: pathlib.Path,
    fail_mask: int,
    needs_fields: tuple[str, ...],
    summary: dict[str, dict],
) -> dict[str, Any]:
    """One matrix row as a paper record (see the module docstring)."""
    smi_path = mdir / f"{tag}.smi.txt"
    smi = json.loads(smi_path.read_text()) if smi_path.exists() else None
    rec: dict[str, Any] = {
        "tag": tag,
        "tier": tag.split("_", 1)[0],
        "rc": None if smi is None else smi.get("rc"),
        "cards": None if smi is None else smi.get("cards"),
        "contended": None if smi is None else smi.get("contended"),
        "row_wall_s": None if smi is None else smi.get("wall_s"),
        "gate_args": None if smi is None else smi.get("gate_args"),
        "matrix_summary": summary.get(tag),
    }
    reasons: list[str] = []
    jpath = mdir / f"{tag}.json"
    if not jpath.exists():
        rec.update(
            valid=False,
            capacity_ok=None,
            invalid_reasons=[f"row did not complete (rc {rec['rc']}): no {tag}.json"],
            log_tail=_tail(mdir / f"{tag}.log"),
        )
        return rec
    d = json.loads(jpath.read_text())
    ndev = int(d.get("ndev", 1))
    n = int(d["n"])
    platform = _platform(d.get("devices") or [])
    rec.update(
        mode=d.get("mode"),
        lane="solo" if d.get("mode") == "solo" else "mesh",
        ic=d.get("ic"),
        n=n,
        ndev=ndev,
        per_device_n=-(-n // ndev),
        mac=d.get("mac"),
        migrate=bool(d.get("migrate")),
        softening=d.get("softening"),
        platform=platform,
        devices=d.get("devices"),
        cuda_visible=d.get("cuda_visible"),
        xla_flags=d.get("xla_flags"),
        gate_smi_before=d.get("smi_before"),
        gate_smi_after=d.get("smi_after"),
    )
    if rec["rc"] not in (0, None):
        reasons.append(f"rc {rec['rc']}")

    if d.get("mode") == "solo":
        start, end = d.get("solo_start") or {}, d.get("solo_end") or {}
        cap_ok = bool(start.get("ok")) and bool(end.get("ok"))
        force_ms = start.get("force_ms") or []
        rec.update(
            capacity_ok=cap_ok,
            leaf_capacity=start.get("leaf_capacity"),
            rel_l2_start=start.get("rel_l2"),
            rel_l2_end=end.get("rel_l2"),
            # the solo mode times the force alone; it has no step
            force_ms_all=force_ms,
            force_s=(min(force_ms) / 1e3) if force_ms else None,
            step_s=None,
            peak_bytes=d.get("peak_bytes"),
        )
        rel_end = end.get("rel_l2")
        if not cap_ok:
            reasons.append("capacity verdict failed (solo ok false)")
        if not force_ms:
            reasons.append("no force time")
    else:
        reports = list(d.get("reports") or [])
        kept = ([d["start_report"]] if d.get("start_report") else []) + reports
        bits = [int(r.get("bits", 0)) for r in kept]
        cap_ok = bool(kept) and all((b & fail_mask) == 0 for b in bits)
        seg = list(d.get("segment_s") or [])
        steps = int(d.get("steps") or 1)
        step_s = statistics.median(seg[3:]) / steps if len(seg) > 3 else None
        force_ms = d.get("force_only_ms") or []
        needs_max = None
        if reports and all(len(r.get("needs", [])) == len(needs_fields) for r in kept):
            needs_max = {
                name: max(int(r["needs"][i]) for r in kept)
                for i, name in enumerate(needs_fields)
            }
        rec.update(
            capacity_ok=cap_ok,
            bits=sorted(set(bits)),
            retries=sum(int(r.get("retries", 0)) for r in kept),
            declined_repartitions=sum(1 for r in reports if r.get("declined")),
            counts0=d.get("counts0"),
            counts_per_segment=[r.get("counts") for r in reports],
            migrated=sum(int(r.get("migrated", 0)) for r in reports),
            needs_max=needs_max,
            leaf_capacity=d.get("leaf_capacity"),
            plan=d.get("plan"),
            cross_caps_final=d.get("cross_caps_final"),
            rel_l2_start=d.get("rel_l2_start"),
            rel_l2_end=d.get("rel_l2_end"),
            steps=steps,
            segments=d.get("segments"),
            segment_s=seg,
            setup_s=d.get("setup_s"),
            start_s=d.get("start_s"),
            step_s=step_s,
            step_s_gate=(
                d["step_ms_steady"] / 1e3 if _finite(d.get("step_ms_steady")) else None
            ),
            force_ms_all=force_ms,
            force_s=(min(force_ms) / 1e3) if force_ms else None,
            peak_bytes=d.get("peak_bytes"),
            peak_bytes_per_particle=d.get("peak_bytes_per_particle"),
        )
        rel_end = d.get("rel_l2_end")
        if not cap_ok:
            reasons.append(f"capacity verdict failed (bits {sorted(set(bits))})")
        if step_s is None:
            reasons.append(f"no steady step time ({len(seg)} segments, need > 3)")

    if not _finite(rel_end):
        reasons.append(f"rel_l2_end not finite ({rel_end!r})")
    if platform != "gpu":
        reasons.append(f"platform {platform}: not a timing (emulated devices)")
    elif rec["contended"] is None:
        reasons.append("card load unknown (no TAG.smi.txt)")
    elif rec["contended"]:
        reasons.append("contended: another process on the row's cards")
    # the September notebooks' keys, kept where they still mean something
    t = rec.get("step_s") if rec.get("lane") == "mesh" else rec.get("force_s")
    rec["median_s"] = t
    rec["median_s_is"] = "step" if rec.get("lane") == "mesh" else "force"
    rec["throughput_particles_per_s"] = (n / t) if t else None
    rec["overflowed"] = [] if rec.get("capacity_ok") else ["capacity verdict"]
    rec["invalid_reasons"] = reasons
    rec["valid"] = not reasons
    return rec


def _series_key(rec: dict[str, Any]) -> str:
    """The matrix's own name of the row's series: its tag without tier and cards.

    ``strong_D2.5e7_de_4c`` -> ``D2.5e7_de`` (IC initial and N, MAC: ``de`` =
    ``dehnen_error``, ``geo`` = ``dehnen``); ``weak_P4e6pc_2c`` -> ``P4e6pc``. Taken
    from the tag so a row that never wrote its record still lands in its series.
    """
    return re.sub(r"^(strong|weak)_", "", re.sub(r"_(\d+c|solo)$", "", rec["tag"]))


def _seed(records: list[dict[str, Any]]) -> Any:
    """The gate's ``--seed`` from the rows' recorded arguments (its default is 0)."""
    seeds = set()
    for r in records:
        argv = r.get("gate_args") or []
        seeds.add(int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 0)
    return seeds.pop() if len(seeds) == 1 else sorted(seeds)


def _write(
    tier: str,
    out: str,
    records: list[dict[str, Any]],
    prov: dict[str, Any],
    mdir: pathlib.Path,
) -> pathlib.Path:
    for r in records:
        r["series"] = _series_key(r)
    valid = [r for r in records if r["valid"]]
    invalid = [r for r in records if not r["valid"]]
    if invalid:
        print(f"[{tier}] {len(invalid)} of {len(records)} row(s) NOT plottable:")
        for r in invalid:
            print(f"  {r['tag']}: {'; '.join(r['invalid_reasons'])}")
    config = {
        # jsonio.REQUIRED_CONFIG_KEYS
        "n": sorted({r["n"] for r in records if r.get("n") is not None}),
        "theta": GATE_CONSTANTS["theta"],
        "order": GATE_CONSTANTS["max_order"],
        "basis": "real",
        "seed": _seed(records),
        "device": prov["device_kind"],
        # the gate does not record its working dtype: an explicit "not recorded"
        "precision": None,
        # the rest
        "source": "bench/multigpu_scaling_matrix.py -> bench/multigpu_segment_gate.py",
        "matrix_dir": str(mdir),
        "tier": tier,
        "series": sorted({r["series"] for r in records}),
        "ndevs": sorted({r["ndev"] for r in records if r.get("ndev")}),
        "macs": sorted({r["mac"] for r in records if r.get("mac")}),
        "leaf_size": GATE_CONSTANTS["leaf_size"],
        "gate_constants": GATE_CONSTANTS,
        "gate_constants_from_source": True,
        "interconnect": prov["interconnect"],
        "interconnect_links": prov.get("interconnect_links"),
        "device_label_asserted_by_operator": prov["asserted_by_operator"],
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "time_definitions": {
            "step_s": "median(segment_s[3:]) / steps: one step incl. force, drift, "
            "migration and the repartition between segments",
            "step_s_gate": "the gate's step_ms_steady (segments after the second)",
            "force_s": "min(force_only_ms): the start program alone",
            "median_s": "step_s on the mesh lane, force_s on a solo row (legacy key)",
        },
        "notes": [
            "seed: the matrix passes no --seed, so the gate's default 0 (its "
            "Plummer draw); the 25M disc + bulge is a prefix of a shuffled IC",
            "precision: the gate does not record its working dtype (preset "
            "'galaxy'; positions are passed as float32)",
            "theta/order/leaf_size: literals of the gate's source, not its record",
        ],
    }
    if tier == "weak":
        per = sorted({r["per_device_n"] for r in records if r.get("per_device_n")})
        config["per_device_n_fixed"] = per[0] if len(per) == 1 else per
    meta: dict[str, Any] = dict(runmeta.git_meta())
    meta.update(
        {
            "argv": sys.argv[1:],
            "converter": "bench/multigpu/from_matrix.py",
            "converter_measured_nothing": True,
            "measurement_provenance": prov,
        }
    )
    path = jsonio.write_result(out, config=config, meta=meta, data={"records": records})
    print(f"[{tier}] wrote {path} ({len(valid)} plottable, {len(invalid)} not)")
    return path


def main() -> int:
    """Convert the matrix's strong / weak rows; 0 when at least one row was found."""
    args = _parse_args()
    mdir = args.matrix_dir.resolve()
    if not mdir.is_dir():
        raise SystemExit(f"--matrix-dir {mdir} is not a directory")
    fail_mask, needs_fields = _library_layout()
    summary: dict[str, dict] = {}
    rows_jsonl = mdir / "rows.jsonl"
    if rows_jsonl.exists():
        for line in rows_jsonl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                summary[row.get("tag", "")] = row
    tags = sorted(
        {p.name[: -len(".json")] for p in mdir.glob("*.json")}
        | {p.name[: -len(".smi.txt")] for p in mdir.glob("*.smi.txt")}
    )
    prov = _provenance(args)
    wrote = 0
    for tier, out in (("strong", args.strong_out), ("weak", args.weak_out)):
        if tier not in args.tiers.split(","):
            continue
        records = [
            _row(t, mdir, fail_mask, needs_fields, summary)
            for t in tags
            if t.startswith(f"{tier}_")
        ]
        if not records:
            print(f"[{tier}] no {tier}_* rows in {mdir}; nothing written")
            continue
        _write(tier, out, records, prov, mdir)
        wrote += 1
    return 0 if wrote else 1


if __name__ == "__main__":
    raise SystemExit(main())
