"""Figure 06 -- where a fused-lane step's time goes, per stage, vs N.

The question this figure answers for section 4 is whether the step is still
"launch-bound". The September version read the strict non-fused refresh path's
host-side stage timers on ``preset="large_n_gpu"``; the production lane is now the
fused one (``preset="galaxy"``, D3), where the whole step is ONE jitted program and
no host timer can see inside it. So this measures it from the device's side.

Two measurements per N, both on ``bench/scaling/_galaxy.py``'s solver:

1. **Rebuild vs evaluate** (always). ``jaccpot_step`` (``FusedForceProvider.step``:
   rebuild + evaluate), ``evaluate`` (evaluation only, on the prepare's topology)
   and ``rebuild`` (``provider.rebuild`` alone, jitted, returning the topology),
   each the minimum over ``--repeats`` (``_timing.time_min_repeat``). The split
   ``step - evaluate`` is reported next to the directly timed rebuild; they need
   not agree exactly (one program vs two; the rebuild's outputs are materialised).
2. **Per stage, from a profiler trace** (``--attribution trace``, the default).
   ``jax.profiler.trace`` around ``--trace-calls`` warm steps; every device event
   (a GPU kernel; on CPU, an XLA:CPU thunk) carries the HLO instruction that ran
   it, the step's optimized HLO (``lower().compile().as_text()``) maps each
   instruction to its ``op_name`` metadata, and the event is credited to the
   INNERMOST ``fmm_*`` ``jax.named_scope`` on that path -- the method of
   ``bench/analyse_trace_by_stage.py``. Nothing is rescaled.

What the bands are (``SCOPE_BANDS``): ``tree_build`` (``fmm_tree``), ``upward``
(P2M + M2M, ``fmm_upward``), ``mac_geometry`` (COM radii, MAC extents, the
criterion table), ``walk`` (``fmm_walk``, ``fmm_lists``), ``m2l`` (what remains in
``fmm_downward`` once the finer scopes are taken out: the M2L and the downward
glue -- and, on the CPU route, the flat walk, which has no scope of its own there;
``fmm_walk`` wraps the GPU's Pallas walk), ``l2l``, ``nearfield_p2p``
(``fmm_near``), ``l2p``, ``evaluate_other`` (the evaluation's own glue,
``fmm_large_n_state``), ``unscoped`` (device time under no ``fmm_*`` scope: the
step's scatter, verdict and anything not yet named; plotted, not hidden),
``unmapped`` (events whose HLO op carries no ``op_name`` metadata: copies and
other instructions XLA inserted; 7-12 % of the summed thunk time on the CPU smoke),
``memcpy_memset`` (GPU events with no HLO op) and
``unattributed`` -- the per-step wall clock (the timed step, no profiler) minus all
device time: on a GPU, the time the device sat idle (launch gaps, host syncs).
``busy_fraction`` (union of device intervals / trace window) and
``launches_per_step`` are recorded next to it: they, not this script, decide
whether "launch-bound" survives.

Honesty notes the figure has to carry:

* **CUDA graphs are off** for the whole run under ``--attribution trace``
  (``_timing.launch_env(command_buffers=False)``): inside a graph a trace names the
  graph, not the kernels. That changes the launch cost, so this run's step time is
  NOT figure 04's; ``config.launch_env`` says which it was. ``--attribution none``
  keeps production's launch environment and records the split only.
* **On CPU** XLA:CPU runs thunks concurrently on a thread pool; the bands are
  summed EXCLUSIVE thunk time (nested while/call thunks have their children's time
  removed), which can exceed the wall clock, and a CPU breakdown with interpreted
  Pallas kernels says nothing about a GPU. CPU rows are smoke only.
* The scopes are the library's own; if a stage is not under one, it lands in
  ``unscoped``. The mapping is data (``SCOPE_BANDS``), and any ``fmm_*`` scope it
  does not know is reported under its own name rather than folded in.

A row whose capacity verdict fails is recorded with ``error`` (the notebook drops
it): a breakdown of a wrong force is not a breakdown.

Usage
-----
CPU smoke (one traced call; the CPU trace is large)::

    JAX_PLATFORMS=cpu python -m bench.scaling.stage_breakdown --n 2048 \\
        --repeats 2 --warmup 1 --trace-calls 1 --gpu-select none \\
        --json-out /tmp/smoke.json

Paper run (one free A100)::

    python -m bench.scaling.stage_breakdown --n 1e6,1e7,1e8
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import pathlib
import re
import sys
import tempfile
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "scaling/stage_breakdown.json"

#: innermost ``fmm_*`` named scope -> the figure's band. Scopes are the library's
#: (grep ``named_scope("fmm_`` in jaccpot/); an unknown one keeps its own name.
SCOPE_BANDS: dict[str, str] = {
    "fmm_tree": "tree_build",
    "fmm_upward": "upward",
    "fmm_com_radii": "mac_geometry",
    "fmm_mac_extents": "mac_geometry",
    "fmm_walk_criterion_table": "mac_geometry",
    "fmm_walk": "walk",
    "fmm_lists": "walk",
    "fmm_downward": "m2l",
    "fmm_l2l": "l2l",
    "fmm_near": "nearfield_p2p",
    "fmm_l2p": "l2p",
    "fmm_large_n_state": "evaluate_other",
}
BAND_ORDER = (
    "tree_build",
    "upward",
    "mac_geometry",
    "walk",
    "m2l",
    "l2l",
    "nearfield_p2p",
    "l2p",
    "evaluate_other",
    "unscoped",
    "unmapped",
    "memcpy_memset",
)
#: a path component that IS a named scope (not a function name containing one,
#: e.g. ``_propagate_solidfmm_locals_by_level``)
_SCOPE_RE = re.compile(r"(?:^|/)(fmm_[a-z0-9_]+)(?=/|$)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n",
        default="1e6,1e7,1e8",
        help="Comma-separated particle counts, 1e6 notation allowed",
    )
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--distribution", default="plummer_clipped")
    p.add_argument("--rmax", type=float, default=20.0)
    p.add_argument("--softening", type=float, default=1e-7)
    p.add_argument("--G", dest="g", type=float, default=1.0)
    p.add_argument(
        "--attribution",
        choices=("trace", "none"),
        default="trace",
        help="trace: per-stage device time from a profiler trace (CUDA graphs "
        "off); none: the rebuild/evaluate split only, production launch flags",
    )
    p.add_argument("--trace-calls", type=int, default=5)
    p.add_argument(
        "--trace-dir",
        default=None,
        help="Keep the traces here (default: a temporary directory)",
    )
    p.add_argument("--kernels", type=int, default=3, help="heaviest ops per band")
    runmeta.add_common_args(p)
    p.set_defaults(dtype="float32")
    return p.parse_args()


# --------------------------------------------------------------------------- HLO


def _op_names(hlo_text: str) -> tuple[str, dict[str, str]]:
    """``(module name, {instruction: op_name})`` from an optimized HLO module's text.

    A fusion without metadata of its own takes the op_name most of its body
    carries (``bench/analyse_trace_by_stage.py``'s fallback).
    """
    module = "unknown"
    head = re.match(r"\s*HloModule\s+([\w.\-]+)", hlo_text)
    if head:
        module = head.group(1)
    comp_re = re.compile(r"^\s*(?:ENTRY\s+)?%?([\w.\-]+)\s+\(.*\)\s*->\s*.*\{\s*$")
    inst_re = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(.*)$")
    name_re = re.compile(r'op_name="([^"]*)"')
    calls_re = re.compile(r"calls=%?([\w.\-]+)")
    names: dict[str, str] = {}
    body: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    calls: dict[str, str] = {}
    comp: Optional[str] = None
    for line in hlo_text.splitlines():
        c = comp_re.match(line)
        if c:
            comp = c.group(1)
            continue
        if line.startswith("}"):
            comp = None
            continue
        m = inst_re.match(line)
        if not m:
            continue
        n = name_re.search(m.group(2))
        if n:
            names.setdefault(m.group(1), n.group(1))
            if comp is not None:
                body[comp][n.group(1)] += 1
        cm = calls_re.search(m.group(2))
        if cm:
            calls.setdefault(m.group(1), cm.group(1))
    for inst, callee in calls.items():
        if inst not in names and body.get(callee):
            names[inst] = body[callee].most_common(1)[0][0]
    return module, names


def _band(op_name: str) -> tuple[str, str]:
    """``(band, innermost scope)`` of an op_name path."""
    hits = _SCOPE_RE.findall(op_name or "")
    if not hits:
        return "unscoped", ""
    scope = hits[-1]
    return SCOPE_BANDS.get(scope, scope), scope


# ------------------------------------------------------------------------- trace


def _attribute(
    trace_dir: str, module: str, names: dict[str, str], calls: int, top: int
) -> dict[str, Any]:
    """Credit every device event of ``module`` to a band; per-call seconds."""
    files = sorted(glob.glob(f"{trace_dir}/**/*.trace.json.gz", recursive=True))
    if not files:
        raise RuntimeError(f"no trace written under {trace_dir}")
    ev = json.load(gzip.open(files[-1], "rt"))["traceEvents"]
    pid_name = {
        e["pid"]: e["args"]["name"]
        for e in ev
        if e.get("ph") == "M" and e.get("name") == "process_name"
    }
    gpus = {p for p, n in pid_name.items() if re.search(r"/device:GPU:\d+$", n)}
    xs = [e for e in ev if e.get("ph") == "X" and "dur" in e]
    if gpus:
        platform = "gpu"
        dev = [e for e in xs if e["pid"] in gpus]
        # GPU kernels do not nest: every event's duration is its own
        events = [(e, float(e["dur"])) for e in dev]
    else:
        platform = "cpu"
        dev = [
            e
            for e in xs
            if e.get("args", {}).get("hlo_op")
            and str(e["args"].get("hlo_module", "")).startswith(module)
            and not str(e.get("name", "")).startswith("end: ")
        ]
        # thunks nest (a while thunk holds its body's): exclusive time per thread
        excl: dict[int, float] = {}
        by_tid: dict[Any, list] = collections.defaultdict(list)
        for i, e in enumerate(dev):
            by_tid[(e["pid"], e["tid"])].append(i)
        for idxs in by_tid.values():
            idxs.sort(key=lambda i: (dev[i]["ts"], -dev[i]["dur"]))
            stack: list[int] = []
            for i in idxs:
                start = dev[i]["ts"]
                # pop the enclosing events that ended before this one started
                while stack and dev[stack[-1]]["ts"] + dev[stack[-1]]["dur"] <= start:
                    stack.pop()
                excl[i] = float(dev[i]["dur"])
                if stack:  # the innermost enclosing event loses this child's time
                    excl[stack[-1]] -= float(dev[i]["dur"])
                stack.append(i)
        events = [(dev[i], max(excl[i], 0.0)) for i in range(len(dev))]
    tot: collections.Counter = collections.Counter()
    cnt: collections.Counter = collections.Counter()
    heavy: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    scopes_seen: collections.Counter = collections.Counter()
    for e, dur in events:
        hop = e.get("args", {}).get("hlo_op", "")
        if hop.startswith("command_buffer"):
            band = "IN_CUDA_GRAPH"
        elif not hop:
            band = "memcpy_memset"
        elif hop not in names:
            band = "unmapped"  # no op_name metadata (XLA-inserted copies and the like)
        else:
            band, scope = _band(names[hop])
            if scope:
                scopes_seen[scope] += 1
        tot[band] += dur
        cnt[band] += 1
        heavy[band][str(e.get("name", ""))[:60]] += dur
    spans = sorted((e["ts"], e["ts"] + e["dur"]) for e, _ in events)
    busy, cur_s, cur_e = 0.0, None, None
    for s, t in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, t
        else:
            cur_e = max(cur_e, t)
    if cur_e is not None:
        busy += cur_e - cur_s
    window = (spans[-1][1] - spans[0][0]) if spans else 0.0
    us = 1e-6 / max(1, calls)
    return {
        "platform": platform,
        "trace_file": files[-1],
        "module": module,
        "bands_s": {k: v * us for k, v in tot.items()},
        "events_per_step": {k: cnt[k] // max(1, calls) for k in cnt},
        "launches_per_step": sum(cnt.values()) // max(1, calls),
        "device_time_per_step_s": sum(tot.values()) * us,
        "busy_per_step_s": busy * us,
        "window_per_step_s": window * us,
        "busy_fraction": (busy / window) if window else None,
        "heaviest": {
            k: [(n, t * us) for n, t in heavy[k].most_common(top)] for k in heavy
        },
        "scopes_seen": dict(scopes_seen),
        "unmapped_scopes": sorted(s for s in scopes_seen if s not in SCOPE_BANDS),
    }


def main() -> int:
    args = _parse_args()
    if args.dtype != "float32":
        print(
            "[fig06] preset='galaxy' is fp32 (D3); --dtype float64 is not this figure"
        )
        return 2
    runmeta.select_gpu(args.gpu_select)
    from bench.scaling import _timing as T  # noqa: E402  (imports no jax)

    env = T.launch_env(command_buffers=(args.attribution == "none"))

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402

    from bench.scaling import _galaxy as GX  # noqa: E402

    ns = [int(float(v)) for v in str(args.n).split(",") if v.strip()]
    records: list[dict[str, Any]] = []
    galaxy: dict[str, Any] = {}
    keep = args.trace_dir is not None
    trace_root = args.trace_dir or tempfile.mkdtemp(prefix="fig06_trace_")

    for n in ns:
        key = jax.random.PRNGKey(runmeta.seed_sequence(args.seed, args.distribution, n))
        x, m = T.distribution(key, n, args.distribution, jnp.float32, rmax=args.rmax)
        x, m = jax.block_until_ready((x, m))
        load_before = GX.card_load()
        try:
            solver = GX.galaxy_solver(softening=float(args.softening), G=float(args.g))
            galaxy = galaxy or GX.describe(solver)
            fs, setup = T.warm_fused_step(solver, x, m)
            provider = fs.provider
            rebuild = jax.jit(
                lambda xx, mm, fb: provider.rebuild(xx, mm, force_scale=fb)
            )

            def _t(fn):
                return T.time_min_repeat(
                    fn, warmup=int(args.warmup), repeats=int(args.repeats)
                )

            step_t = _t(lambda: fs.step(x, m))
            eval_t = _t(lambda: fs.evaluate())
            rebuild_t = _t(lambda: rebuild(x, m, fs.aux[0]))
            acc, aux = fs.step(x, m)
            jax.block_until_ready(acc)
            ok = bool(fs.ok(aux)) and bool(setup["first_step_ok"])
        except Exception as exc:
            print(f"N={n:<10d} FAILED: {str(exc)[:200]}")
            records.append({"n": int(n), "valid": False, "error": str(exc)[:400]})
            continue

        wall = step_t[0]
        record: dict[str, Any] = {
            "n": int(n),
            "valid": ok,
            "per_step_wall_s": wall,
            "split_s": {
                "step": {"min_s": step_t[0], "mean_s": step_t[1], "std_s": step_t[2]},
                "evaluate": {
                    "min_s": eval_t[0],
                    "mean_s": eval_t[1],
                    "std_s": eval_t[2],
                },
                "rebuild": {
                    "min_s": rebuild_t[0],
                    "mean_s": rebuild_t[1],
                    "std_s": rebuild_t[2],
                },
                "rebuild_by_difference": step_t[0] - eval_t[0],
            },
            "setup": setup,
            "card_load_before": load_before,
        }
        if not ok:
            record["error"] = (
                f"capacity verdict failed (bits {int(aux[1]):#x}): the force is wrong"
            )

        if args.attribution == "trace" and ok:
            try:
                compiled = fs._step.lower(x, m, None, fs.aux).compile()
                module, names = _op_names(compiled.as_text())
                tdir = str(pathlib.Path(trace_root) / f"n{n}")
                with jax.profiler.trace(tdir):
                    for _ in range(int(args.trace_calls)):
                        jax.block_until_ready(fs.step(x, m))
                att = _attribute(
                    tdir, module, names, int(args.trace_calls), int(args.kernels)
                )
                if not keep:
                    att.pop("trace_file", None)
                bands = {b: att["bands_s"].get(b, 0.0) for b in BAND_ORDER}
                for extra, v in att["bands_s"].items():
                    if extra not in bands:
                        bands[extra] = v  # an unknown scope, under its own name
                device = sum(bands.values())
                bands["unattributed"] = max(wall - device, 0.0)
                record["stages_s"] = bands
                record["stage_sum_s"] = device
                record["attributed_fraction"] = device / wall if wall else None
                record["trace"] = att
                if att["unmapped_scopes"]:
                    print(
                        f"  [warn] fmm_* scopes not in SCOPE_BANDS: "
                        f"{att['unmapped_scopes']}"
                    )
            except Exception as exc:
                record["trace_error"] = str(exc)[:400]
                print(f"  [trace] N={n} failed: {str(exc)[:200]}")
        if "stages_s" not in record:
            # the minimum always reported: the rebuild / evaluate split
            record["stages_s"] = {
                "rebuild": max(step_t[0] - eval_t[0], 0.0),
                "evaluate": eval_t[0],
            }
            record["stage_sum_s"] = step_t[0]
            record["attributed_fraction"] = 1.0
            record["stages_from"] = "split (step - evaluate, evaluate)"
        else:
            record["stages_from"] = "profiler trace, innermost fmm_* scope"
        record["card_load_after"] = GX.card_load()
        if record.get("valid") and not GX.card_free(
            record.get("card_load_before"), record["card_load_after"]
        ):
            record["valid"] = False
            record["invalid_reason"] = "contended card"
        records.append(record)
        fs = rebuild = None  # free the provider before the next N
        shares = "  ".join(
            f"{k}={v / wall * 100:4.1f}%"
            for k, v in record["stages_s"].items()
            if wall and v > 0
        )
        extra = ""
        if "trace" in record:
            tr = record["trace"]
            bf = tr["busy_fraction"]
            extra = (
                f"\n            launches/step={tr['launches_per_step']} "
                f"busy={tr['busy_per_step_s'] * 1e3:.2f} ms of a "
                f"{tr['window_per_step_s'] * 1e3:.2f} ms window"
                + (f" ({bf:.1%})" if bf is not None else "")
            )
        print(
            f"N={n:<10d} valid={ok} step={wall * 1e3:9.2f} ms "
            f"evaluate={eval_t[0] * 1e3:8.2f} ms rebuild={rebuild_t[0] * 1e3:8.2f} ms"
            f"\n            {shares}{extra}",
            flush=True,
        )

    config = {
        "n": ns,
        "theta": float(GX.GALAXY.theta),
        "order": int(GX.GALAXY.max_order),
        "basis": "real",
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": "float32",
        "leaf_size": int(GX.GALAXY.leaf_size),
        "preset": "galaxy",
        "galaxy": galaxy,
        "distribution": args.distribution,
        "softening": float(args.softening),
        "G": float(args.g),
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "attribution": args.attribution,
        "trace_calls": int(args.trace_calls),
        "launch_env": env,
        "scope_bands": dict(SCOPE_BANDS),
        "band_order": list(BAND_ORDER) + ["unattributed"],
        "unattributed": (
            "per-step wall clock (timed step, no profiler) minus all device time "
            "in the trace: on a GPU, device idle (launch gaps, host syncs)"
        ),
        "note": (
            "The fused step is one jitted program: stages come from a profiler "
            "trace attributed by the innermost fmm_* named scope of each event's "
            "HLO op (CUDA graphs off under attribution=trace, so this run's step "
            "time is not figure 04's). On CPU the bands are summed exclusive "
            "thunk time of a thread pool, smoke only."
        ),
    }
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config=config,
        meta=runmeta.run_meta({"argv": sys.argv[1:]}),
        data={"records": records},
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
