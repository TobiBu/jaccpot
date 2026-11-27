"""Figure 10 data: collective vs compute device time per step on the mesh lane.

Section 5 claims the multi-GPU cost is set by the hierarchy (the per-device tree,
walk and kernels), not by communication. This bench tests that on the production
mesh lane (``MeshSegmentRunner.fused``, the configuration figures 08 and 09 time)
from a ``jax.profiler`` trace of warm mesh steps, with no instrumentation inside
the program: what runs is the program that is measured.

How the time is attributed
--------------------------
Every device event of the trace names the HLO instruction that launched it
(``args.hlo_op``, ``args.hlo_module``). The worker dumps the optimized HLO of every
program it compiles (``--xla_dump_to``, text) and reads each launched instruction's
OPCODE there, so the attribution survives fusion and renaming:

* collective: ``all-to-all``, ``ragged-all-to-all``, ``all-gather``, ``all-reduce``,
  ``reduce-scatter``, ``collective-permute``, ``collective-broadcast``, ``send`` /
  ``recv`` (and their ``-start`` / ``-done`` halves, and an ``async-start`` whose
  wrapped computation holds one of them); on a GPU also any kernel whose name
  contains ``nccl``;
* compute: every other launched instruction;
* ``copy_or_unmapped``: device events with no HLO instruction (memcpy / memset).

Without a dump mapping (``--no-hlo-dump``, or an instruction the dump lacks) the
instruction name is matched against the same opcodes and JAX's primitive names
(``psum``, ``ppermute``, ``all_gather``, ...); the record counts how many events
each route classified, so the attribution's reliability is on the record.

Per device and per step (``--trace-segments`` segments of ``--steps`` steps:
force, drift, migration and the repartition programs between segments, exactly as
the segment runner runs them), from the UNION of each class's event intervals:

* ``busy_us``: any event running; ``window_us``: first start to last end;
* ``collective_us`` and ``compute_us``: each class's own union;
* ``exposed_collective_us``: collective time with no compute event running (what a
  faster interconnect could remove); ``overlapped_us``: collective and compute at
  once (NCCL runs on its own streams);
* ``collective_by_kind_us``: the summed durations per collective opcode;
* ``top_ops``: device time and launch count per instruction name.

A collective kernel's duration is transfer PLUS waiting for the slowest peer to
arrive: load imbalance shows up as collective time on the devices that wait. Read
``exposed_collective_us`` next to figure 11's per-device work before calling it
communication. The force alone is traced too (``--trace-force`` calls of the
start program), to separate the force's exchanges from the migration's.

On GPUs the worker clears ``--xla_gpu_enable_command_buffer`` (inside a CUDA graph
every kernel reports the graph's name, as ``bench/analyse_trace_by_stage.py``
notes); the untraced step time is measured in the same process, with that flag,
and recorded next to the traced one. ``--keep-command-buffers`` keeps the
production setting and records that kernels inside graphs are unattributed.

On the CPU (``JAX_PLATFORMS=cpu``, forced host devices) the split is MEANINGLESS as
a number: the emulated devices share one CPU and its thread pool, and an XLA:CPU
collective is a rendezvous between threads. The CPU run only proves the plumbing
(trace, dump, per-device attribution); its records say so and are never
``valid``.

``--dtype`` (the paper benches' common flag) is accepted and IGNORED: the
galaxy preset sets the working precision, as in the scaling matrix.

Usage
-----
::

    python -m bench.multigpu.comm_overhead --ndevs 2,4 --per-device-n 4000000

CPU smoke (seconds per point)::

    JAX_PLATFORMS=cpu python -m bench.multigpu.comm_overhead --ndevs 2 \\
        --per-device-n 1024 --mac dehnen --steps 2 --warm-segments 1 \\
        --time-segments 1 --trace-segments 1 --trace-force 1 \\
        --json-out /tmp/comm_overhead_smoke.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.multigpu import _mesh_common as mc  # noqa: E402
from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "multigpu/comm_overhead.json"
MODULE = "bench.multigpu.comm_overhead"

#: HLO opcodes that are communication
COLLECTIVE_OPCODES = frozenset(
    f"{base}{half}"
    for base in (
        "all-to-all",
        "ragged-all-to-all",
        "all-gather",
        "all-reduce",
        "reduce-scatter",
        "collective-permute",
        "collective-broadcast",
        "send",
        "recv",
    )
    for half in ("", "-start", "-done")
)
#: the fallback when the dump has no entry: opcodes and JAX's primitive names
_COLLECTIVE_NAME = re.compile(
    r"^(ragged[-_]all[-_]to[-_]all|all[-_]to[-_]all|all[-_]gather|all[-_]reduce|"
    r"reduce[-_]scatter|collective[-_]permute|collective[-_]broadcast|send|recv|"
    r"psum|pmax|pmin|pmean|ppermute|psum_scatter|psum_invariant)"
    r"([-_](start|done))?([.#]\d+)?$"
)
_DUMP_FILE = re.compile(
    r"^module_\d+\.(?P<name>.+?)\.(?:cpu_|sm_\d+\.\d+_gpu_|gpu_|rocm_)?"
    r"after_optimizations\.txt$"
)
_COMP_HEAD = re.compile(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s+\(.*\)\s+->\s+.*\{\s*$")
_INST = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-#]+)\s*=\s*(.*)$")
_CALLS = re.compile(r"calls=%?([\w.\-]+)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="See the module docstring for the attribution and its limits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    runmeta.add_common_args(p)
    p.add_argument("--ndevs", default="2,4", help="device counts, one process each")
    p.add_argument(
        "--per-device-n",
        type=int,
        default=4_000_000,
        help="particles per device (default: the weak tier's 4e6 per card)",
    )
    p.add_argument("--distribution", default="plummer", help="plummer, uniform, disc")
    p.add_argument("--disc-ic", default=None)
    p.add_argument("--mac", choices=("dehnen_error", "dehnen"), default="dehnen_error")
    p.add_argument("--steps", type=int, default=8, help="steps per segment (matrix: 8)")
    p.add_argument(
        "--warm-segments",
        type=int,
        default=3,
        help="untimed segments first (the segment, repartition and migration "
        "programs compile in the first ones)",
    )
    p.add_argument("--time-segments", type=int, default=3, help="untraced, timed")
    p.add_argument("--trace-segments", type=int, default=2, help="traced segments")
    p.add_argument(
        "--trace-force", type=int, default=3, help="traced start-force calls (0: off)"
    )
    p.add_argument(
        "--no-migrate",
        action="store_true",
        help="MeshSegmentRunner(migrate=False) instead of the production migration",
    )
    p.add_argument(
        "--no-hlo-dump",
        action="store_true",
        help="classify by instruction NAME only (no --xla_dump_to)",
    )
    p.add_argument(
        "--keep-command-buffers",
        action="store_true",
        help="GPU: keep XLA's CUDA graphs (kernels inside them go unattributed)",
    )
    p.add_argument(
        "--gpu-thread-re",
        default=r"^Stream",
        help="GPU: the trace threads holding kernel events (default '^Stream'; the "
        "derived 'XLA Ops' / 'XLA Modules' lines would count kernels twice)",
    )
    p.add_argument(
        "--keep-trace-dir",
        default=None,
        help="keep the traces and HLO dumps under this directory (default: a "
        "temporary directory, deleted)",
    )
    # worker-only
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--ndev", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--work-dir", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# HLO dump: instruction -> opcode, per module
# --------------------------------------------------------------------------- #


def _opcode(rhs: str) -> Optional[str]:
    """The opcode of one HLO instruction's right-hand side (after ``=``)."""
    rhs = rhs.lstrip()
    if rhs.startswith("("):
        depth = 0
        for i, ch in enumerate(rhs):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0:
                rest = rhs[i + 1 :].lstrip()
                break
        else:
            return None
    else:
        parts = rhs.split(" ", 1)
        if len(parts) < 2:
            return None
        rest = parts[1].lstrip()
    m = re.match(r"([\w\-]+)\(", rest)
    return m.group(1) if m else None


def read_hlo_opcodes(dump_dir: Optional[str]) -> dict[str, dict[str, str]]:
    """``{module name: {instruction: opcode}}`` from ``*after_optimizations.txt``.

    An ``async-start`` / ``-update`` / ``-done`` whose called computation holds a
    collective is mapped to ``async:<that opcode>``. ``""`` holds every module's
    entries merged (the first module wins), the fallback for events whose module
    was not dumped under the trace's name.

    Parameters
    ----------
    dump_dir : str or None
        The ``--xla_dump_to`` directory.

    Returns
    -------
    dict
        Empty when there is no dump.
    """
    per: dict[str, dict[str, str]] = {}
    merged: dict[str, str] = {}
    if not dump_dir or not os.path.isdir(dump_dir):
        return {"": merged}
    for path in sorted(glob.glob(os.path.join(dump_dir, "*after_optimizations.txt"))):
        m = _DUMP_FILE.match(os.path.basename(path))
        if not m:
            continue
        ops: dict[str, str] = {}
        calls: dict[str, str] = {}
        holds: dict[str, str] = {}
        comp = None
        with open(path, errors="replace") as fh:
            for line in fh:
                head = _COMP_HEAD.match(line)
                if head:
                    comp = head.group(1)
                    continue
                if line.startswith("}"):
                    comp = None
                    continue
                inst = _INST.match(line)
                if not inst:
                    continue
                op = _opcode(inst.group(2))
                if not op:
                    continue
                ops[inst.group(1)] = op
                if comp and op in COLLECTIVE_OPCODES:
                    holds.setdefault(comp, op)
                c = _CALLS.search(inst.group(2))
                if c and op.startswith("async-"):
                    calls[inst.group(1)] = c.group(1)
        for inst_name, callee in calls.items():
            if callee in holds:
                ops[inst_name] = f"async:{holds[callee]}"
        per[m.group("name")] = ops  # sorted: the last compile of a module wins
        for k, v in ops.items():
            merged.setdefault(k, v)
    per[""] = merged
    return per


# --------------------------------------------------------------------------- #
# trace analysis
# --------------------------------------------------------------------------- #


def _union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _measure(spans: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in spans)


def _leaves(events: list[dict]) -> list[dict]:
    """The events with no other event of the same thread nested inside them.

    XLA:CPU's thunk executor emits a ``while`` thunk around its body's thunks; a
    GPU stream's kernels never nest, so this is a no-op there.
    """
    by_thread: dict[Any, list[dict]] = collections.defaultdict(list)
    for e in events:
        by_thread[e.get("tid")].append(e)
    keep = []
    for evs in by_thread.values():
        evs.sort(key=lambda e: (e["ts"], -e["dur"]))
        for i, e in enumerate(evs):
            end = e["ts"] + e["dur"]
            nxt = evs[i + 1] if i + 1 < len(evs) else None
            if nxt is not None and nxt["ts"] < end and nxt["ts"] + nxt["dur"] <= end:
                continue  # a container: its children carry the time
            keep.append(e)
    return keep


def _classify(e: dict, opcodes: dict[str, dict[str, str]]) -> tuple[str, str, str]:
    """``(class, kind, route)``: class collective / compute / copy_or_unmapped."""
    args = e.get("args") or {}
    hop = str(args.get("hlo_op") or "")
    name = str(e.get("name") or "")
    if "nccl" in name.lower():
        kind = opcodes.get(str(args.get("hlo_module") or ""), {}).get(hop) or "nccl"
        return "collective", kind, "kernel_name"
    if not hop:
        return "copy_or_unmapped", name[:40], "no_hlo_op"
    module_ops = opcodes.get(str(args.get("hlo_module") or ""))
    op = (module_ops or {}).get(hop) or opcodes.get("", {}).get(hop)
    if op is not None:
        base = op.split(":", 1)[-1]
        if base in COLLECTIVE_OPCODES:
            return "collective", base.replace("-start", "").replace("-done", ""), "dump"
        return "compute", op, "dump"
    m = _COLLECTIVE_NAME.match(hop)
    if m:
        return "collective", m.group(1), "name"
    return "compute", "unmapped", "name"


def analyse_trace(
    trace_dir: str,
    opcodes: dict[str, dict[str, str]],
    units: int,
    gpu_thread_re: str,
) -> dict[str, Any]:
    """Per-device collective / compute split of one ``jax.profiler`` trace.

    Parameters
    ----------
    trace_dir : str
        The ``jax.profiler.trace`` directory (its ``*trace.json.gz``).
    opcodes : dict
        :func:`read_hlo_opcodes`.
    units : int
        What the totals are divided by (steps, or force calls).
    gpu_thread_re : str
        GPU: the threads whose events are kernels.

    Returns
    -------
    dict
        ``devices`` (per device, per unit: see the module docstring), ``route_counts``
        (events classified by the dump, by name, by kernel name, without an
        instruction), the trace file and the thread names used and skipped.
    """
    files = sorted(
        glob.glob(os.path.join(trace_dir, "**", "*trace.json.gz"), recursive=True)
    )
    if not files:
        raise RuntimeError(f"no *trace.json.gz under {trace_dir}")
    # the perfetto file and the TensorBoard one hold the same events; take one
    path = next((f for f in files if "perfetto" in os.path.basename(f)), files[-1])
    with gzip.open(path, "rt") as fh:
        raw = json.load(fh)
    events = raw["traceEvents"] if isinstance(raw, dict) else raw
    pname = {
        e["pid"]: str((e.get("args") or {}).get("name", ""))
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name"
    }
    tname = {
        (e["pid"], e.get("tid")): str((e.get("args") or {}).get("name", ""))
        for e in events
        if e.get("ph") == "M" and e.get("name") == "thread_name"
    }
    gpu_pids = {p: n for p, n in pname.items() if re.search(r"/device:GPU:\d+", n)}
    thread_re = re.compile(gpu_thread_re)
    used_threads: set[str] = set()
    skipped_threads: set[str] = set()
    by_device: dict[str, list[dict]] = collections.defaultdict(list)
    for pid, label in gpu_pids.items():
        tids = {t for (p, t) in tname if p == pid}
        keep = {t for t in tids if thread_re.search(tname[(pid, t)])} or tids
        for t in tids:
            (used_threads if t in keep else skipped_threads).add(tname[(pid, t)])
        dev = re.search(r"/device:GPU:\d+", label).group(0)
        for e in events:
            if (
                e.get("ph") == "X"
                and e.get("pid") == pid
                and e.get("tid") in keep
                and float(e.get("dur") or 0) > 0
            ):
                by_device[dev].append(e)
    if not gpu_pids:
        # XLA:CPU: thunk events on the host carry the device ordinal they ran for
        for e in events:
            args = e.get("args") or {}
            if (
                e.get("ph") == "X"
                and "device_ordinal" in args
                and args.get("hlo_op")
                and float(e.get("dur") or 0) > 0
            ):
                by_device[f"cpu:{args['device_ordinal']}"].append(e)
                used_threads.add(tname.get((e["pid"], e.get("tid")), "?"))
    u = max(1, int(units))
    routes: collections.Counter = collections.Counter()
    devices: dict[str, Any] = {}
    for dev in sorted(by_device):
        evs = _leaves(by_device[dev])
        spans: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
        by_kind: collections.Counter = collections.Counter()
        op_time: collections.Counter = collections.Counter()
        op_calls: collections.Counter = collections.Counter()
        for e in evs:
            cls, kind, route = _classify(e, opcodes)
            routes[route] += 1
            span = (float(e["ts"]), float(e["ts"]) + float(e["dur"]))
            spans[cls].append(span)
            if cls == "collective":
                by_kind[kind] += float(e["dur"])
            label = str((e.get("args") or {}).get("hlo_op") or e.get("name"))[:60]
            op_time[(cls, label)] += float(e["dur"])
            op_calls[(cls, label)] += 1
        coll = _union(spans["collective"])
        comp = _union(spans["compute"])
        copy = _union(spans["copy_or_unmapped"])
        every = _union(
            spans["collective"] + spans["compute"] + spans["copy_or_unmapped"]
        )
        c_us, k_us = _measure(coll), _measure(comp)
        both = c_us + k_us - _measure(_union(spans["collective"] + spans["compute"]))
        busy = _measure(every)
        window = (every[-1][1] - every[0][0]) if every else 0.0
        devices[dev] = {
            "busy_us": busy / u,
            "window_us": window / u,
            "collective_us": c_us / u,
            "compute_us": k_us / u,
            "copy_or_unmapped_us": _measure(copy) / u,
            "exposed_collective_us": (c_us - both) / u,
            "overlapped_us": both / u,
            "exposed_collective_fraction_of_busy": (
                ((c_us - both) / busy) if busy else None
            ),
            "collective_by_kind_us": {k: v / u for k, v in by_kind.most_common()},
            "launches": len(evs) / u,
            "top_ops": [
                {
                    "class": cls,
                    "op": label,
                    "us": t / u,
                    "calls": op_calls[(cls, label)] / u,
                }
                for (cls, label), t in op_time.most_common(15)
            ],
        }
    return {
        "trace_file": path,
        "units": u,
        "route_counts": dict(routes),
        "threads_used": sorted(used_threads),
        "threads_skipped": sorted(skipped_threads),
        "devices": devices,
    }


def _mesh_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    """Max and mean over devices of the per-unit fields the figure plots."""
    devs = list(analysis["devices"].values())
    if not devs:
        return {}
    out = {}
    for key in (
        "busy_us",
        "collective_us",
        "compute_us",
        "exposed_collective_us",
        "overlapped_us",
        "copy_or_unmapped_us",
    ):
        vals = [d[key] for d in devs]
        out[f"{key}_max"] = max(vals)
        out[f"{key}_mean"] = sum(vals) / len(vals)
    busy = out["busy_us_max"]
    out["exposed_collective_fraction_max"] = max(
        (d["exposed_collective_fraction_of_busy"] or 0.0) for d in devs
    )
    out["exposed_collective_over_busy_max"] = (
        out["exposed_collective_us_max"] / busy if busy else None
    )
    return out


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #


def _worker(args: argparse.Namespace) -> int:
    ndev = int(args.ndev)
    work = pathlib.Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    dump_dir = None if args.no_hlo_dump else str(work / "hlo")
    extra = []
    if dump_dir:
        extra += [f"--xla_dump_to={dump_dir}", "--xla_dump_hlo_as_text"]
    if not mc.on_cpu() and not args.keep_command_buffers:
        extra.append("--xla_gpu_enable_command_buffer=")
    os.environ["XLA_FLAGS"] = " ".join(
        [os.environ.get("XLA_FLAGS", "")] + extra
    ).strip()
    chosen = mc.worker_preamble(ndev, args.gpu_select)

    import jax

    jax.config.update("jax_enable_x64", True)

    n = int(args.per_device_n) * ndev
    pos, vel, mass, g, soft, dt = mc.make_ic(
        args.distribution, n, int(args.seed), args.disc_ic
    )
    solver = mc.make_solver(args.mac, g, soft)
    runner, decomposition = mc.build_runner(
        solver, pos, vel, mass, ndev, migrate=not args.no_migrate
    )
    reports = []

    def segment() -> float:
        t0 = time.perf_counter()
        rep = runner.run(int(args.steps), dt=dt)
        jax.block_until_ready(runner.state.state)
        reports.append(
            {k: getattr(rep, k) for k in ("segment", "retries", "bits", "counts")}
        )
        return time.perf_counter() - t0

    from bench.scaling import _galaxy as GX

    load_before = GX.card_load()
    runner.start()
    jax.block_until_ready(runner.state.acceleration)
    for _ in range(int(args.warm_segments)):
        segment()
    timed = [segment() for _ in range(int(args.time_segments))]
    steps = int(args.steps)

    seg_dir = str(work / "trace_segments")
    with jax.profiler.trace(seg_dir, create_perfetto_trace=True):
        for _ in range(int(args.trace_segments)):
            segment()
    force_dir = None
    if int(args.trace_force) > 0:
        force_dir = str(work / "trace_force")
        s = runner.state
        inputs = (s.state, s.masses, s.count, s.ids, *runner._scale_args(s))
        jax.block_until_ready(runner._call("start", 0, None, *inputs))  # warm
        with jax.profiler.trace(force_dir, create_perfetto_trace=True):
            for _ in range(int(args.trace_force)):
                jax.block_until_ready(runner._call("start", 0, None, *inputs))

    from jaccpot.runtime.fused_topology import OVERFLOW_FAIL_MASK

    opcodes = read_hlo_opcodes(dump_dir)
    seg = analyse_trace(
        seg_dir, opcodes, int(args.trace_segments) * steps, args.gpu_thread_re
    )
    force = (
        analyse_trace(force_dir, opcodes, int(args.trace_force), args.gpu_thread_re)
        if force_dir
        else None
    )
    load_after = GX.card_load()
    platform = mc.platform_of_devices()
    cap_ok = all((int(r["bits"]) & int(OVERFLOW_FAIL_MASK)) == 0 for r in reports)
    # a retry re-plans and recompiles inside the segment: the timed and traced
    # segments must have none, or their per-step split includes a compile
    measured = reports[int(args.warm_segments) :]
    retried = [r["segment"] for r in measured if int(r["retries"]) != 0]
    reasons = []
    if retried:
        reasons.append(f"segment retries in timed/traced segments {retried}")
    if not GX.card_free(load_before, load_after):
        reasons.append("contended card (another process on a visible card)")
    if not cap_ok:
        reasons.append("capacity verdict failed in a segment")
    if platform["platform"] != "gpu":
        reasons.append(
            f"platform {platform['platform']}: emulated devices share one CPU; the "
            "split is plumbing only, not a measurement"
        )
    if not seg["devices"]:
        reasons.append("no device events found in the trace")
    if args.keep_command_buffers and platform["platform"] == "gpu":
        reasons.append("command buffers kept: kernels inside CUDA graphs unattributed")
    record = {
        "ndev": ndev,
        "n": n,
        "per_device_n": int(args.per_device_n),
        "distribution": args.distribution,
        "mac": args.mac,
        "migrate": not args.no_migrate,
        "steps": steps,
        "decomposition": decomposition,
        "step_s_untraced": [t / steps for t in timed],
        # the whole run clears --xla_gpu_enable_command_buffer (attribution needs
        # the kernels outside CUDA graphs): NOT production's launch environment
        "step_s_untraced_env": (
            "command buffers kept"
            if args.keep_command_buffers
            else "command buffers off (not the production launch environment)"
        ),
        "card_load_before": load_before,
        "card_load_after": load_after,
        "step_s_untraced_median": (
            (sorted(timed)[len(timed) // 2] / steps) if timed else None
        ),
        "segment_reports": reports,
        "capacity_ok": cap_ok,
        "per_step": seg,
        "per_step_mesh": _mesh_summary(seg),
        "per_force_call": force,
        "per_force_call_mesh": _mesh_summary(force) if force else None,
        "hlo_dump": bool(dump_dir),
        "hlo_modules_read": len([k for k in opcodes if k]),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "meaningless_on_cpu": platform["platform"] != "gpu",
        "valid": not reasons,
        "invalid_reasons": reasons,
        "cuda_visible_devices": chosen,
        **platform,
        "meta": runmeta.run_meta({"argv": sys.argv[1:]}),
    }
    print(json.dumps(record, default=str))
    return 0 if record["valid"] else 1


# --------------------------------------------------------------------------- #
# parent
# --------------------------------------------------------------------------- #


def main() -> int:
    """One process per device count; write the artifact.

    Returns
    -------
    int
        0 when every point's worker completed its trace analysis (on the CPU its
        records are still never ``valid``: that is in the artifact, not the exit
        code), 1 when a worker failed or found no device events.
    """
    args = _parse_args()
    if args.worker:
        return _worker(args)
    ndevs = [int(v) for v in str(args.ndevs).split(",") if v.strip()]
    if any(d < 2 for d in ndevs):
        raise SystemExit("--ndevs must all be >= 2: one device has no collectives")
    root = args.keep_trace_dir or tempfile.mkdtemp(prefix="comm_overhead_")
    common = [
        "--per-device-n",
        str(args.per_device_n),
        "--distribution",
        args.distribution,
        "--mac",
        args.mac,
        "--steps",
        str(args.steps),
        "--warm-segments",
        str(args.warm_segments),
        "--time-segments",
        str(args.time_segments),
        "--trace-segments",
        str(args.trace_segments),
        "--trace-force",
        str(args.trace_force),
        "--gpu-thread-re",
        args.gpu_thread_re,
        "--seed",
        str(args.seed),
        "--gpu-select",
        args.gpu_select,
        "--dtype",
        args.dtype,
    ]
    for flag in ("no_migrate", "no_hlo_dump", "keep_command_buffers"):
        if getattr(args, flag):
            common.append("--" + flag.replace("_", "-"))
    if args.disc_ic:
        common += ["--disc-ic", args.disc_ic]
    points = [
        {"ndev": d, "argv": ["--work-dir", os.path.join(root, f"ndev{d}")]}
        for d in ndevs
    ]
    try:
        records = mc.run_points(MODULE, points, common)
    finally:
        if not args.keep_trace_dir:
            shutil.rmtree(root, ignore_errors=True)
    for r in records:
        m = r.get("per_step_mesh") or {}
        flag = "" if r.get("valid") else f"  [NOT VALID: {r.get('invalid_reasons')}]"
        print(
            f"ndev={r.get('ndev')} busy/step={m.get('busy_us_max')} us "
            f"collective={m.get('collective_us_max')} "
            f"exposed={m.get('exposed_collective_us_max')}{flag}"
        )
    worker_meta = next((r.get("meta") for r in records if r.get("meta")), {}) or {}
    config = {
        "n": [r.get("n") for r in records],
        "theta": mc.THETA,
        "order": mc.MAX_ORDER,
        "basis": "real",
        "seed": args.seed,
        "device": worker_meta.get("device_kind", "unknown"),
        "precision": None,
        "ndevs": ndevs,
        "per_device_n_fixed": int(args.per_device_n),
        "distribution": args.distribution,
        "leaf_size": mc.LEAF_SIZE,
        "preset": "galaxy",
        "mac": args.mac,
        "migrate": not args.no_migrate,
        "steps": args.steps,
        "trace_segments": args.trace_segments,
        "trace_force": args.trace_force,
        "hlo_dump": not args.no_hlo_dump,
        "command_buffers_kept": bool(args.keep_command_buffers),
        "collective_opcodes": sorted(COLLECTIVE_OPCODES),
        "trace_dir": args.keep_trace_dir,
        "n_valid": sum(1 for r in records if r.get("valid")),
        "n_invalid": sum(1 for r in records if not r.get("valid")),
    }
    meta = dict(runmeta.git_meta())
    meta.update(
        {
            "argv": sys.argv[1:],
            "sweep_parent": True,
            "parent_imported_jax": False,
            "worker_jax": worker_meta.get("jax_version"),
            "worker_device_kind": worker_meta.get("device_kind"),
            "measured_at": worker_meta.get("measured_at"),
        }
    )
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config=config,
        meta=meta,
        data={"records": records},
    )
    print(f"wrote {out}")
    ran = all((r.get("per_step") or {}).get("devices") for r in records)
    return 0 if ran else 1


if __name__ == "__main__":
    raise SystemExit(main())
