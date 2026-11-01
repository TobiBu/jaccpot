"""Time a rollout's gradient against its forward on one card (S9, the HoreKa job).

usage: python bench/grad_rollout_timing.py --n N --mode {step,nested,offload}
       --steps T [--inner K] [--only {all,forward,diffforward,grad}] [--reps R]
       [--trace DIR] [--reference JSON] --out OUT.json
       python bench/grad_rollout_timing.py --summarize DIR

One configuration per process, on the S8/S9 operating point of
``bench/grad_rollout_memory.py`` (``preset="galaxy"``, clipped Plummer, float32,
``dehnen_error``, the galaxy leaf size and order, ``dt = 1e-3``), measured with
track D's clean-allocator recipe (``exec_step.py``): prepare, drop the solver's
``_last_*`` side channels, re-place the provider's template so it lands first
(``--no-compact`` keeps it where the prepare left it), compile each program from
shapes, then place the inputs. Per program, ``--reps`` warm calls (after one
untimed call) under ``block_until_ready``:

a. ``forward``: the production forward rollout, ``differentiable_rollout`` over
   ``FusedForceProvider.step`` with ``checkpoint=None`` -- ``strict_run_v2(carry=
   "particles")``'s trajectory bit for bit (G1) and the reference of the ratios;
b. ``diffforward``: the same forward-only rollout over ``differentiable_step``.
   ``b - a`` is what the gradient's forward pays beyond production: the grad
   lists and the duplicate far chain. Its positions and velocities must equal
   (a)'s bitwise (G1): with both in one process the elements that differ are
   counted; with ``--only diffforward`` the sha256 of each is compared with the
   ``--reference`` JSON of the ``--only forward`` row;
c. ``grad``: ``jax.value_and_grad`` of ``sum(w . x_T) + sum(w . v_T)`` in the start
   positions, velocities and masses under ``--mode`` (and ``--inner``), the verdict
   (the folded aux's bits) returned beside it. A-vs-A: the first and last call's
   gradients' sha256.

Each program: min / median / IQR of the calls in ms per call and per step (a call
is the start force and ``T`` steps: ``T + 1`` forces), its ``memory_analysis()``
(temporaries, host temporaries -- ``"offload"``'s pinned host stacks -- arguments,
outputs), the allocator's ``bytes_in_use`` before and ``peak_bytes_in_use`` after
(the peak is the process's: it belongs to this program only when it rose during
it, ``peak_set_here``; ``--only`` gives one program per process, the clean
measurement), the verdict bits and finiteness. Under ``"offload"`` also the
gradient program's host transfer sites (``grad_rollout_memory._host_transfers``).
The ratios ``c / a`` and ``b / a`` are written when the programs share a process;
``--summarize DIR`` joins the rows of a directory (the sbatch's ``--only`` runs)
into one table with the ratios.

CONTENTION. ``nvidia-smi``'s compute processes on this card (other than this one)
and its utilisation, before the prepare and after the last call; a row with
another process on the card is flagged ``contended`` and its timings are not to be
quoted. Local cards are shared: their rows are functional checks only.

``--trace DIR``: a ``jax.profiler`` trace of two gradient calls after the timing,
and a table of the device kernels' time per call by stage
(``grad_step_profile._stage_of``: ``near_rev_pallas``, ``m2l_rev_pallas``, ...).

The allocator is the caller's: ``XLA_PYTHON_CLIENT_PREALLOCATE``,
``XLA_PYTHON_CLIENT_MEM_FRACTION`` and ``XLA_PYTHON_CLIENT_ALLOCATOR`` are read from
the environment and recorded (``bench/slurm/grad_rollout_timing.sbatch`` sets them
per row).
"""

from __future__ import annotations

import argparse
import collections
import gc
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROGRAMS = ("forward", "diffforward", "grad")
_ENV_KEYS = (
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "TF_GPU_ALLOCATOR",
    "CUDA_VISIBLE_DEVICES",
    "XLA_FLAGS",
)


def _args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument(
        "--mode", default="step", choices=("step", "nested", "offload"), help="(c)"
    )
    ap.add_argument("--steps", type=int, default=1, help="T, the rollout's steps")
    ap.add_argument("--inner", type=int, default=None, help="nested / offload")
    ap.add_argument("--only", default="all", choices=("all",) + PROGRAMS)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--softening", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-compact",
        action="store_true",
        help="keep the provider's template where the prepare left it",
    )
    ap.add_argument("--trace", default=None, help="profile 2 gradient calls here")
    ap.add_argument(
        "--reference",
        default=None,
        help="the --only forward row's JSON: (b)'s digests are checked against it",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--summarize", default=None, help="join the JSON rows of this directory"
    )
    return ap.parse_args(argv)


# --------------------------------------------------------------------------
# The card.
# --------------------------------------------------------------------------


def _card_state() -> dict:
    """Other compute processes and the utilisation of this process's card."""
    from grad_production_gate import _card_processes

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    util = None
    try:
        util = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                cvd,
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        util = f"nvidia-smi failed: {exc}"
    return {"other_processes": _card_processes(), "utilization": util}


def _alloc(device) -> dict:
    stats = device.memory_stats() or {}
    return {
        k: stats.get(k)
        for k in (
            "bytes_in_use",
            "peak_bytes_in_use",
            "largest_free_block_bytes",
            "bytes_limit",
            "largest_alloc_size",
        )
    }


def _gib(v: Any) -> Any:
    return None if v is None else round(v / 2**30, 3)


def _digest(*arrays: Any) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def _stats_ms(ts: list[float], steps: int) -> dict:
    ms = 1e3 * np.asarray(ts)
    q25, q75 = np.percentile(ms, [25, 75])
    out = {
        "calls": len(ts),
        "min_ms": float(ms.min()),
        "median_ms": float(np.median(ms)),
        "iqr_ms": float(q75 - q25),
        "all_ms": [float(t) for t in ms],
    }
    out.update({f"{k}_per_step": v / steps for k, v in list(out.items())[1:4]})
    return out


# --------------------------------------------------------------------------
# The trace's kernel table.
# --------------------------------------------------------------------------


def _kernel_table(trace_dir: str, calls: int) -> dict:
    """Device kernel time per call from a perfetto trace, by kernel and by stage.

    Parameters
    ----------
    trace_dir : str
        The ``jax.profiler.trace`` directory (``create_perfetto_trace=True``).
    calls : int
        Gradient calls traced.

    Returns
    -------
    dict
        ``{"stages": {stage: {"ms_per_call", "launches_per_call"}}, "top": [...],
        "kernel_ms_per_call"}``, or ``{"error": ...}``.
    """
    from grad_step_profile import _stage_of

    files = sorted(glob.glob(f"{trace_dir}/**/*.trace.json.gz", recursive=True))
    if not files:
        return {"error": f"no perfetto trace under {trace_dir}"}
    with gzip.open(files[-1], "rt") as fh:
        data = json.load(fh)
    events = data["traceEvents"] if isinstance(data, dict) else data
    pnames, tnames = {}, {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pnames[e["pid"]] = e["args"]["name"]
        elif e.get("ph") == "M" and e.get("name") == "thread_name":
            tnames[(e["pid"], e["tid"])] = e["args"]["name"]
    dev = {
        p for p, n in pnames.items() if "GPU" in n.upper() and "HOST" not in n.upper()
    }
    by_name: dict = collections.defaultdict(lambda: [0.0, 0])
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in dev:
            continue
        thread = tnames.get((e["pid"], e["tid"]), "").lower()
        if "module" in thread or "step" in thread:
            continue
        by_name[e["name"]][0] += float(e.get("dur", 0.0))
        by_name[e["name"]][1] += 1
    stages: dict = {}
    for name, (us, count) in by_name.items():
        s = stages.setdefault(
            _stage_of(name), {"ms_per_call": 0.0, "launches_per_call": 0.0}
        )
        s["ms_per_call"] += us / 1e3 / calls
        s["launches_per_call"] += count / calls
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1][0])[:40]
    return {
        "trace_file": files[-1],
        "kernel_ms_per_call": sum(v[0] for v in by_name.values()) / 1e3 / calls,
        "stages": dict(sorted(stages.items(), key=lambda kv: -kv[1]["ms_per_call"])),
        "top": [
            {
                "name": n,
                "stage": _stage_of(n),
                "ms_per_call": v[0] / 1e3 / calls,
                "launches_per_call": v[1] / calls,
            }
            for n, v in ranked
        ],
    }


# --------------------------------------------------------------------------
# One configuration.
# --------------------------------------------------------------------------


def _prepare(args: argparse.Namespace) -> tuple:
    """The provider on a clean allocator, and the host copies of the inputs."""
    import grad_rollout_memory as grm
    import jax
    import jax.numpy as jnp

    bargs = grm._args(
        ["--force", "provider", "--n", str(args.n), "--softening", str(args.softening)]
    )
    bargs.check = True  # the provider prepared for differentiable_step too
    pos, vel, mass = grm._plummer(args.n, args.seed)
    x0, m = jnp.asarray(pos, jnp.float32), jnp.asarray(mass, jnp.float32)
    prov, aux0 = grm._provider_force(bargs, x0, m)
    fb = np.asarray(aux0[0])
    del x0, m, aux0
    # the solver's per-call side channels of the eager prepare
    impl = getattr(prov.engine, "_impl", None)
    for holder in (prov.engine, impl):
        if holder is None:
            continue
        for name in list(vars(holder)):
            if name.startswith("_last_"):
                setattr(holder, name, None)
    gc.collect()
    if not args.no_compact:
        # the template back through the host: with everything else freed its
        # arrays land first, and the rest of the region stays one free block
        from jaccpot import FusedForceProvider

        engine, statics = prov.engine, prov.grad_statics
        kw = dict(
            leaf_size=prov.leaf_size,
            max_order=prov.max_order,
            theta=prov.theta,
            runtime_overrides=prov.runtime_overrides,
        )
        host = jax.tree_util.tree_map(
            lambda a: np.asarray(a) if isinstance(a, jax.Array) else a, prov.template
        )
        del prov
        gc.collect()
        template = jax.tree_util.tree_map(
            lambda a: jnp.asarray(a) if isinstance(a, np.ndarray) else a, host
        )
        del host
        prov = FusedForceProvider(engine, template=template, grad_statics=statics, **kw)
    w = np.random.default_rng(args.seed + 1).normal(size=pos.shape)
    host_inputs = (
        pos.astype(np.float32),
        vel.astype(np.float32),
        mass.astype(np.float32),
        fb.astype(np.float32),
        w.astype(np.float32),
    )
    return prov, host_inputs


def _program(name: str, prov: Any, args: argparse.Namespace):
    """The jitted function of one measured program, and whether it is the gradient."""
    import jax
    import jax.numpy as jnp

    from jaccpot.runtime.grad_rollout import differentiable_rollout

    if name in ("forward", "diffforward"):
        step_fn = prov.step if name == "forward" else prov.differentiable_step

        def rollout(x, v, mm, a, w_in):
            del w_in
            return differentiable_rollout(
                step_fn, x, v, mm, args.dt, args.steps, aux=a, checkpoint=None
            )

        return jax.jit(rollout)

    def loss(x, v, mm, a, w_in):
        out = differentiable_rollout(
            prov.differentiable_step,
            x,
            v,
            mm,
            args.dt,
            args.steps,
            aux=a,
            checkpoint=args.mode,
            inner_steps=args.inner if args.mode != "step" else None,
        )
        value = jnp.sum(w_in * out.positions) + jnp.sum(w_in * out.velocities)
        # the verdict's bits only: the whole aux would add f_b's [N] output
        # (0.37 GiB at 1e8, which the T = 4 offload row does not have beside
        # its 25.7 GiB of temporaries)
        return value, out.aux[1]

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))


def _measure(name: str, compiled: Any, call: tuple, args, device, rec: dict) -> None:
    """Run one compiled program: timings, verdict, digests, the allocator."""
    import jax
    import jax.numpy as jnp

    row: dict = {"before": _alloc(device)}
    t0 = time.perf_counter()
    out = compiled(*call)
    jax.block_until_ready(out)
    row["first_call_s"] = time.perf_counter() - t0
    if name == "grad":
        (value, bits), grads = out
        aux = (None, bits, None)
        row["finite"] = bool(
            all(bool(jnp.all(jnp.isfinite(g))) for g in grads)
            and bool(jnp.isfinite(value))
        )
        first = _digest(*grads)
        # nothing of a call outlives it: at 1e8 a second set of outputs (28
        # B/particle) does not fit beside the next call's temporaries
        del grads, value, bits
    else:
        aux = out.aux
        row["finite"] = bool(
            jnp.all(jnp.isfinite(out.positions)) & jnp.all(jnp.isfinite(out.velocities))
        )
        row["digest_positions"] = _digest(out.positions)
        row["digest_velocities"] = _digest(out.velocities)
        if name == "forward" and args.only == "all":
            rec["_forward_state"] = (
                np.asarray(out.positions),
                np.asarray(out.velocities),
            )
        if name == "diffforward" and "_forward_state" in rec:
            ref_x, ref_v = rec["_forward_state"]
            row["elements_differ_from_forward"] = int(
                np.sum(np.asarray(out.positions) != ref_x)
                + np.sum(np.asarray(out.velocities) != ref_v)
            )
    row["bits"] = int(np.asarray(aux[1]))
    from jaccpot.runtime.grad_rollout import rollout_ok

    row["verdict_ok"] = bool(rollout_ok(aux, gradient=name != "forward"))
    del out, aux
    ts = []
    for i in range(args.reps):
        t0 = time.perf_counter()
        res = compiled(*call)
        jax.block_until_ready(res)
        ts.append(time.perf_counter() - t0)
        if name == "grad" and i == args.reps - 1:
            # A-vs-A: the last call's gradients against the first call's
            row["a_vs_a_bitwise"] = _digest(*res[1]) == first
            row["digest_grads"] = first
        del res
    row["timing"] = _stats_ms(ts, args.steps)
    row["after"] = _alloc(device)
    peak_before = row["before"].get("peak_bytes_in_use") or 0
    peak_after = row["after"].get("peak_bytes_in_use") or 0
    row["peak_bytes"] = peak_after
    row["peak_set_here"] = peak_after > peak_before
    # what this program needs at most: the bytes in use before it, its
    # temporaries and its outputs (the measured peak equals it to the MiB at 1e8
    # when the program sets it)
    prog = rec["programs"][name]
    row["model_peak_bytes"] = (
        (row["before"].get("bytes_in_use") or 0)
        + prog["temp_bytes"]
        + prog["output_bytes"]
    )
    rec["programs"][name].update(row)
    t = row["timing"]
    print(
        f"{name}: {t['min_ms']:.1f} / {t['median_ms']:.1f} ms per call (min / median, "
        f"IQR {t['iqr_ms']:.1f}), {t['median_ms_per_step']:.1f} ms per step; bits "
        f"{row['bits']:#x}, finite {row['finite']}; peak {_gib(peak_after)} GiB"
        + ("" if row["peak_set_here"] else " (set earlier)"),
        flush=True,
    )


def run(args: argparse.Namespace) -> dict:
    """Measure one configuration; the JSON record."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import grad_rollout_memory as grm

    device = jax.devices()[0]
    rec: dict = {
        "config": {k: v for k, v in vars(args).items() if k != "summarize"},
        "env": {k: os.environ.get(k) for k in _ENV_KEYS},
        "device": str(device),
        "jax": jax.__version__,
        "card_before": _card_state(),
        "programs": {},
    }
    t0 = time.perf_counter()
    prov, host = _prepare(args)
    rec["prepare_s"] = time.perf_counter() - t0
    rec["after_prepare"] = _alloc(device)
    print(
        f"prepared in {rec['prepare_s']:.0f} s; {_gib(rec['after_prepare']['bytes_in_use'])} GiB in use",
        flush=True,
    )
    pos, vel, mass, fb, w = host
    shapes = (
        jax.ShapeDtypeStruct(pos.shape, jnp.float32),
        jax.ShapeDtypeStruct(vel.shape, jnp.float32),
        jax.ShapeDtypeStruct(mass.shape, jnp.float32),
        (jax.ShapeDtypeStruct(fb.shape, jnp.float32), None, None),
        jax.ShapeDtypeStruct(w.shape, jnp.float32),
    )
    names = PROGRAMS if args.only == "all" else (args.only,)
    call = None
    compiled = traced = None
    for name in names:
        t0 = time.perf_counter()
        compiled = _program(name, prov, args).lower(*shapes).compile()
        ma = compiled.memory_analysis()
        prog = {
            "compile_s": time.perf_counter() - t0,
            "temp_bytes": int(ma.temp_size_in_bytes),
            "host_temp_bytes": int(ma.host_temp_size_in_bytes),
            "argument_bytes": int(ma.argument_size_in_bytes),
            "output_bytes": int(ma.output_size_in_bytes),
        }
        if name == "grad" and args.mode == "offload":
            prog["host_transfers"] = grm._host_transfers(compiled.as_text(), args.n)
        rec["programs"][name] = prog
        print(
            f"{name}: compiled in {prog['compile_s']:.0f} s, temp "
            f"{_gib(prog['temp_bytes'])} GiB, host temp "
            f"{_gib(prog['host_temp_bytes'])} GiB",
            flush=True,
        )
        if call is None:
            # the inputs after the first compile, on the template's heels
            call = (
                jnp.asarray(pos),
                jnp.asarray(vel),
                jnp.asarray(mass),
                (jnp.asarray(fb), None, None),
                jnp.asarray(w),
            )
        _measure(name, compiled, call, args, device, rec)
        if name == "grad" and args.trace:
            traced = compiled
        del compiled
        gc.collect()
    rec.pop("_forward_state", None)
    if args.reference and "diffforward" in rec["programs"]:
        with open(args.reference) as fh:
            ref = json.load(fh)["programs"]["forward"]
        d = rec["programs"]["diffforward"]
        d["bitwise_to_reference"] = (
            d["digest_positions"] == ref["digest_positions"]
            and d["digest_velocities"] == ref["digest_velocities"]
        )
    progs = rec["programs"]
    if "forward" in progs:
        a = progs["forward"]["timing"]["median_ms"]
        for name in ("diffforward", "grad"):
            if name in progs:
                rec[f"ratio_{name}_over_forward"] = (
                    progs[name]["timing"]["median_ms"] / a
                )
    if traced is not None:
        os.makedirs(args.trace, exist_ok=True)
        with jax.profiler.trace(args.trace, create_perfetto_trace=True):
            for _ in range(2):
                jax.block_until_ready(traced(*call))
        rec["trace"] = _kernel_table(args.trace, 2)
        stages = rec["trace"].get("stages", {})
        print(
            "trace stages (ms per call): "
            + ", ".join(f"{k} {v['ms_per_call']:.1f}" for k, v in stages.items()),
            flush=True,
        )
    rec["card_after"] = _card_state()
    rec["contended"] = (
        rec["card_before"]["other_processes"] != "none"
        or rec["card_after"]["other_processes"] != "none"
    )
    return rec


# --------------------------------------------------------------------------
# The summary of a directory of rows.
# --------------------------------------------------------------------------


def summarize(directory: str) -> list[dict]:
    """Join the rows of a directory: one line per (N, T, mode), with the ratios.

    Parameters
    ----------
    directory : str
        Where the ``--out`` JSONs are.

    Returns
    -------
    list[dict]
        The table rows (also printed).
    """
    forward: dict = {}
    rows = []
    recs = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        if "programs" not in rec:
            continue
        recs.append((path, rec))
        cfg = rec["config"]
        for name in ("forward", "diffforward"):
            if name in rec["programs"]:
                forward[(cfg["n"], cfg["steps"], name)] = rec["programs"][name]
    for path, rec in recs:
        cfg = rec["config"]
        for name, prog in rec["programs"].items():
            if "timing" not in prog:
                continue
            a = forward.get((cfg["n"], cfg["steps"], "forward"))
            row = {
                "file": os.path.basename(path),
                "n": cfg["n"],
                "steps": cfg["steps"],
                "program": name,
                "mode": cfg["mode"] if name == "grad" else "-",
                "inner": cfg["inner"] if name == "grad" else None,
                "ms_per_step": prog["timing"]["median_ms_per_step"],
                "iqr_ms": prog["timing"]["iqr_ms"],
                "over_forward": (
                    prog["timing"]["median_ms"] / a["timing"]["median_ms"]
                    if a and "timing" in a
                    else None
                ),
                "temp_GiB": _gib(prog["temp_bytes"]),
                "host_temp_GiB": _gib(prog["host_temp_bytes"]),
                "peak_GiB": _gib(prog.get("peak_bytes")),
                "bits": prog.get("bits"),
                "finite": prog.get("finite"),
                "contended": rec.get("contended"),
            }
            rows.append(row)
    rows.sort(
        key=lambda r: (r["n"], r["steps"], PROGRAMS.index(r["program"]), r["mode"])
    )
    for r in rows:
        ratio = "-" if r["over_forward"] is None else f"{r['over_forward']:.2f}x"
        print(
            f"N {r['n']:>11,} T {r['steps']:>3} {r['program']:>11} {r['mode']:>7} "
            f"{r['ms_per_step']:10.1f} ms/step (IQR {r['iqr_ms']:.1f} ms/call) "
            f"{ratio:>7}  temp {r['temp_GiB']} host {r['host_temp_GiB']} peak "
            f"{r['peak_GiB']} GiB  bits {r['bits']} finite {r['finite']}"
            + ("  CONTENDED" if r["contended"] else "")
        )
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    args = _args(argv)
    if args.summarize:
        summarize(args.summarize)
        return 0
    rec = run(args)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(rec, fh, indent=1, default=str)
        print(f"wrote {args.out}", flush=True)
    if rec["contended"]:
        print("CONTENDED: another process was on the card; do not quote the timings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
