"""Figure 12 data -- the gradient's cost as a multiple of the forward, vs N.

The claim this figure tests: reverse mode through the FMM force costs a bounded
multiple of the forward, independent of N and of the number of parameters (one
reverse pass gives d/dx and d/dm of all N particles at once). The plotted quantity
is a RATIO of two times measured on the same card, problem and process.

Operating point (REBASE_PLAN D3): ``preset="galaxy"`` on the fused lane (cell
leaves of 64, p6, ``dehnen_error`` at 1e-4, ferrers3), fp32 by default
(``--dtype``), the far pairs kept for the gradient plan, built by
``bench/scaling/_galaxy`` (``grad_bench_lib.solver``). ``--mac dehnen`` moves to
the geometric criterion at ``--theta``; the JSON records what moved off the preset
(``config.solver.off_preset``).

What is timed, per N (``--mode single``, one card)
--------------------------------------------------
Two arms, each a pair (forward, gradient) of jitted programs with every array a
jit ARGUMENT (closed over, ``jax.jit`` would embed them as constants), compiled
before timing (``lower().compile()``, seconds recorded):

``step`` (the figure's)
    forward = the production per-step force, ``FusedForceProvider.step``
    (topology rebuild + evaluation: what a simulation pays per step); gradient =
    ``jax.value_and_grad`` and ``jax.grad`` of ``L = sum(w . acc)`` through
    ``differentiable_step`` w.r.t. positions AND masses (``wrt="both"``: one
    reverse pass). Both sides rebuild the topology.
``fixed_topology`` (``bench/grad_production_gate.py``'s methodology, the source
of the 4.1-4.4x quoted in REBASE_PLAN)
    forward = evaluation only on the prepare's topology
    (``provider.accelerations``); gradient = ``differentiable_accelerations`` on
    that topology with its gradient plan, w.r.t. positions and masses.

Masses are differentiated under ``dehnen_error`` too: the criterion's f_b only
sets the next walk's thresholds and its gradient is stopped
(``docs/fused_provider.md``); the zero-mass conflict does not arise (no tracers).

Per arm: min and median of ``--reps`` warm calls, interleaved forward /
value_and_grad / forward / value_and_grad / grad as the gate does (the ratios use
the minima); the differentiable program's forward against the production forward
(elements whose bits differ: expected 0); ``memory_analysis()`` of every program
in bytes per particle; on a GPU the device's ``peak_bytes_in_use`` after one
value_and_grad call, per particle (a per-process high-water mark: run one N per
process for a per-N peak). Validity: the forward's capacity verdict
(``overflow_ok``), the gradient's (``grad_ok``, ``OVERFLOW_GRAD_LISTS``
included) and a finite gradient. A point that fails is written with ``error``
and ``invalid: true`` and must not be plotted.

``--jvp`` adds one forward-mode row per N (``jax.jvp`` of the step's force along
one random position direction, under ``ad_mode_override("forward")``): at most a
sentence in the paper.

``--mode mesh`` (the multi-GPU point, WP-G): one row of
``bench/multigpu_grad_gate.py --mode mesh`` run as a subprocess on ``--ndev``
cards (a process's device count is fixed): the same loss and timing protocol on
the fused mesh lane (``FusedMeshForce`` + ``make_mesh_grad_fn``), its whole row
kept under ``gate_row``. Its Plummer IC is Odisseo's generator when importable
(``BENCH_DIR``), else the gate's own sampler (``ic_source`` says which), not this
script's harness Plummer. Write it to its own ``--json-out``.

CPU. The reverse runs on the CPU only through the gates' route (the S9 interpret
set, ``jax.default_backend`` reporting ``"gpu"``: ``grad_bench_lib``). Every
kernel is interpreted, so CPU times and ratios are smoke, not data, and the p6
programs compile for minutes.

Usage
-----
CPU smoke (single card; p2, because the interpret-mode p6 reverse alone compiles
for ~40 min on a laptop: measured 2026-11-27, N=512, fp64)::

    JAX_PLATFORMS=cpu python -m bench.differentiability.autodiff_overhead \\
        --n 512 --order 2 --reps 2 --dtype float64 --jvp --gpu-select none \\
        --json-out /tmp/fig12_smoke.json

CPU smoke (mesh, 2 emulated devices; geometric p4 keeps the compile bearable)::

    JAX_PLATFORMS=cpu python -m bench.differentiability.autodiff_overhead \\
        --mode mesh --ndev 2 --n 2048 --distribution cube --softening 1e-2 \\
        --mac dehnen --theta 0.8 --order 4 --reps 2 --dtype float64 \\
        --gpu-select none --json-out /tmp/fig12_mesh.json

Paper run (A100; one N per process keeps each peak its own)::

    for n in 1e5 1e6 8e6 3e7 1e8; do
      python -m bench.differentiability.autodiff_overhead --n $n \\
          --json-out differentiability/autodiff_overhead_n$n.json
    done
    python -m bench.differentiability.autodiff_overhead --mode mesh --ndev 4 \\
        --n 1e8 --json-out differentiability/autodiff_overhead_mesh.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.differentiability import grad_bench_lib as L  # noqa: E402
from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "differentiability/autodiff_overhead.json"
DEFAULT_MESH_OUT = "differentiability/autodiff_overhead_mesh.json"
#: OVERFLOW_GRAD_LISTS (jaccpot.runtime.fused_topology): the lists' symmetry verdict
_GRAD_LISTS_BIT = 1 << 11


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=("single", "mesh"), default="single")
    p.add_argument(
        "--n",
        default="1e5,1e6,8e6",
        help="comma-separated N (1e8 style allowed); --mode mesh takes one",
    )
    p.add_argument(
        "--distribution",
        default="plummer",
        help="single: bench.validation._harness; mesh: the gate's --ic (plummer, cube)",
    )
    p.add_argument("--softening", type=float, default=1e-3)
    p.add_argument(
        "--mac",
        choices=("preset", "dehnen"),
        default="preset",
        help="'preset': the galaxy preset's dehnen_error; 'dehnen': geometric",
    )
    p.add_argument("--theta", type=float, default=None, help="with --mac dehnen")
    p.add_argument(
        "--order",
        type=int,
        default=None,
        help="expansion order (default: the preset's 6). Off the figure's "
        "operating point: for CPU smokes, whose interpret-mode p6 reverse "
        "compiles for tens of minutes; recorded in off_preset. Mesh mode: only "
        "with --mac dehnen (the gate's --galaxy is p6)",
    )
    p.add_argument("--arms", default="step,fixed_topology")
    p.add_argument(
        "--wrt",
        default="both",
        help="'both' (positions and masses, one pass) and/or 'positions'",
    )
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--jvp", action="store_true", help="one forward-mode row per N")
    p.add_argument(
        "--no-forward-check",
        action="store_true",
        help="skip the bitwise forward check (one program less to compile)",
    )
    p.add_argument("--ndev", type=int, default=2, help="--mode mesh: devices")
    p.add_argument(
        "--gate-args",
        default="",
        help="--mode mesh: extra arguments for bench/multigpu_grad_gate.py",
    )
    runmeta.add_common_args(p)
    p.set_defaults(dtype="float32")
    args = p.parse_args()
    args.ns = [int(float(v)) for v in str(args.n).split(",") if v.strip()]
    args.arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
    args.wrt_list = [w.strip() for w in args.wrt.split(",") if w.strip()]
    if args.mac == "preset" and args.theta is not None:
        p.error("--theta moves the geometric criterion: use it with --mac dehnen")
    if args.order is not None and args.mode == "mesh" and args.mac == "preset":
        p.error("--order in mesh mode: only with --mac dehnen (--galaxy is p6)")
    if args.mode == "mesh" and len(args.ns) != 1:
        p.error("--mode mesh is one row: give one --n")
    bad = set(args.arm_list) - {"step", "fixed_topology"}
    bad |= set(args.wrt_list) - {"both", "positions"}
    if bad:
        p.error(f"unknown --arms / --wrt values {sorted(bad)}")
    return args


# --------------------------------------------------------------------------
# single card
# --------------------------------------------------------------------------


def _arm_programs(arm: str, prov: Any, topo: Any, aux0: Any, x, m, w) -> tuple:
    """``(forward, diff forward, loss, forward args, loss args, argnums by wrt)``.

    Every function returns ``(value, bits)``; bits are the verdict to check.
    """
    import jax.numpy as jnp

    if arm == "step":

        def fwd(x, m, aux):
            acc, out = prov.step(x, m, None, aux)
            return acc, out[1]

        def dfwd(x, m, aux):
            acc, out = prov.differentiable_step(x, m, None, aux)
            return acc, out[1]

        def loss(x, m, w, aux):
            acc, out = prov.differentiable_step(x, m, None, aux)
            return jnp.sum(w * acc), out[1]

        return (
            fwd,
            dfwd,
            loss,
            (x, m, aux0),
            (x, m, w, aux0),
            {"both": (0, 1), "positions": 0},
        )

    def sym_bits(sym):
        return jnp.where(sym, jnp.uint32(0), jnp.uint32(_GRAD_LISTS_BIT))

    def fwd(t, x, m):
        return prov.accelerations(t), jnp.uint32(0)

    def dfwd(t, x, m):
        plan, sym = prov.grad_plan(t)
        return prov.differentiable_accelerations(t, x, m, plan=plan), sym_bits(sym)

    def loss(t, x, m, w):
        plan, sym = prov.grad_plan(t)
        acc = prov.differentiable_accelerations(t, x, m, plan=plan)
        return jnp.sum(w * acc), sym_bits(sym)

    return (
        fwd,
        dfwd,
        loss,
        (topo, x, m),
        (topo, x, m, w),
        {"both": (1, 2), "positions": 1},
    )


def _single_point(args: argparse.Namespace, n: int) -> tuple[list, dict]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from bench.validation import _harness as H
    from jaccpot.runtime.fused_topology import grad_ok, overflow_ok

    seed = runmeta.seed_sequence(args.seed, args.distribution, str(n))
    pos, mass = H.make_distribution(args.distribution, n, seed)
    dt = L.jnp_dtype(args.dtype)
    x, m = jnp.asarray(pos, dt), jnp.asarray(mass, dt)
    w = jnp.asarray(L.weights(n, seed), dt)
    del pos, mass
    fmm, scfg = L.solver(
        softening=args.softening,
        dtype=args.dtype,
        mac_type=None if args.mac == "preset" else "dehnen",
        theta=args.theta,
        order=args.order,
    )
    t0 = time.perf_counter()
    prov, topo, aux0 = L.provider(fmm, x, m, order=args.order)
    jax.block_until_ready(topo)
    prepare_s = time.perf_counter() - t0
    print(f"N={n}: prepared in {prepare_s:.1f} s", flush=True)
    base = {
        "n": int(n),
        "basis": "real",
        "lane": "fused",
        "seed": int(seed),
        "prepare_s": prepare_s,
        "memory_after_prepare": L.device_memory(),
    }

    rows = []
    for arm in args.arm_list:
        fwd, dfwd, loss, fargs, gargs, argmap = _arm_programs(
            arm, prov, topo, aux0, x, m, w
        )
        rec: dict[str, Any] = dict(base, mode=arm, grads={}, compile_s={})
        try:
            fwd_c, rec["compile_s"]["forward"] = L.compile_timed(jax.jit(fwd), *fargs)
            rec["memory_forward"] = L.memory(fwd_c, n)
            acc_p, bits_p = fwd_c(*fargs)
            # the step's own verdict; the fixed arm's is the prepare topology's
            rec["force_ok"] = bool(
                overflow_ok(bits_p) if arm == "step" else prov.capacity_ok(topo)
            )
            if not args.no_forward_check:
                dfwd_c, rec["compile_s"]["diff_forward"] = L.compile_timed(
                    jax.jit(dfwd), *fargs
                )
                acc_d = dfwd_c(*fargs)[0]
                rec["forward_differing_elements"] = L.differing(acc_d, acc_p)
                rec["forward_max_abs"] = float(
                    np.max(np.abs(np.asarray(acc_d, np.float64) - np.asarray(acc_p)))
                )
                del dfwd_c, acc_d
            del acc_p
            valid = rec["force_ok"]
            for wrt in args.wrt_list:
                argnums = argmap[wrt]
                vg_c, cs = L.compile_timed(
                    jax.jit(jax.value_and_grad(loss, argnums=argnums, has_aux=True)),
                    *gargs,
                )
                go_c, cs_go = L.compile_timed(
                    jax.jit(jax.grad(loss, argnums=argnums, has_aux=True)), *gargs
                )
                rec["compile_s"][f"value_and_grad_{wrt}"] = cs
                rec["compile_s"][f"grad_{wrt}"] = cs_go
                (_, gbits), grads = vg_c(*gargs)
                jax.block_until_ready(grads)
                mem = L.device_memory()
                g_ok = bool(grad_ok(gbits))
                finite = bool(
                    all(
                        np.all(np.isfinite(np.asarray(g)))
                        for g in jax.tree_util.tree_leaves(grads)
                    )
                )
                del grads
                valid = valid and g_ok and finite
                f_a = L.timed(fwd_c, *fargs, reps=args.reps)
                g_a = L.timed(vg_c, *gargs, reps=args.reps)
                f_b = L.timed(fwd_c, *fargs, reps=args.reps)
                g_b = L.timed(vg_c, *gargs, reps=args.reps)
                g_only = L.timed(go_c, *gargs, reps=args.reps)
                f_min = min(f_a["min_s"], f_b["min_s"])
                g_min = min(g_a["min_s"], g_b["min_s"])
                rec["forward_min_s"] = f_min
                rec["forward_median_s"] = min(f_a["median_s"], f_b["median_s"])
                rec["grads"][wrt] = {
                    "forward_backward_min_s": g_min,
                    "forward_backward_median_s": min(g_a["median_s"], g_b["median_s"]),
                    "ratio": g_min / f_min,
                    "grad_only_min_s": g_only["min_s"],
                    "grad_only_median_s": g_only["median_s"],
                    "grad_only_ratio": g_only["min_s"] / f_min,
                    "grad_ok": g_ok,
                    "grad_finite": finite,
                    "memory_value_and_grad": L.memory(vg_c, n),
                    "memory_grad": L.memory(go_c, n),
                    "device_peak_B_per_p": (
                        None if mem["peak_bytes"] is None else mem["peak_bytes"] / n
                    ),
                    "timings": {"forward": [f_a, f_b], "value_and_grad": [g_a, g_b]},
                }
                print(
                    f"N={n:<10d} {arm:<15s} d/d{wrt:<9s} fwd {f_min*1e3:9.2f} ms  "
                    f"v&g {g_min*1e3:9.2f} ms  ratio {g_min/f_min:5.2f}  "
                    f"grad-only {g_only['min_s']/f_min:5.2f}  ok={g_ok and finite}",
                    flush=True,
                )
                del vg_c, go_c
            if args.jvp and arm == "step":
                rec["jvp"] = _jvp_row(prov, x, m, aux0, fwd_c, fargs, n, args, seed)
            rec["valid"] = bool(valid)
            if not valid:
                verdicts = {k: v["grad_ok"] for k, v in rec["grads"].items()}
                rec["invalid"] = True
                rec["error"] = (
                    f"verdict failed (force_ok={rec['force_ok']}, grad_ok={verdicts}): "
                    "not a measurement; do not plot"
                )
        except Exception as exc:  # noqa: BLE001 - recorded, not plotted
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
            print(f"N={n} {arm}: FAILED {rec['error']}", flush=True)
        rows.append(rec)
    return rows, scfg


def _jvp_row(prov, x, m, aux0, fwd_c, fargs, n, args, seed) -> dict:
    """One forward-mode (``jax.jvp``) call of the step's force, against the step."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaccpot.runtime.grad_options import ad_mode_override

    v = jnp.asarray(np.random.default_rng(seed + 7).normal(size=(n, 3)), x.dtype)

    def tangent(x, m, aux, v):
        return jax.jvp(
            lambda xx: prov.differentiable_step(xx, m, None, aux)[0], (x,), (v,)
        )

    try:
        with ad_mode_override("forward"):
            jvp_c, cs = L.compile_timed(jax.jit(tangent), x, m, aux0, v)
        out = jvp_c(x, m, aux0, v)
        finite = bool(np.all(np.isfinite(np.asarray(out[1]))))
        t = L.timed(jvp_c, x, m, aux0, v, reps=args.reps)
        f = L.timed(fwd_c, *fargs, reps=args.reps)
    except Exception as exc:  # noqa: BLE001 - optional row
        return {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    print(f"N={n:<10d} jvp ratio {t['min_s']/f['min_s']:5.2f}", flush=True)
    return {
        "wrt": "positions, one direction",
        "compile_s": cs,
        "min_s": t["min_s"],
        "median_s": t["median_s"],
        "ratio": t["min_s"] / f["min_s"],
        "tangent_finite": finite,
        "memory": L.memory(jvp_c, n),
    }


def _run_single(args: argparse.Namespace) -> int:
    env_set = L.prepare_env()
    import jax

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    L.patch_cpu_lane()
    from bench.scaling import _galaxy

    records: list = []
    scfg: dict = {}
    load_before = _galaxy.card_load()
    for n in args.ns:
        rows, scfg = _single_point(args, n)
        records.extend(rows)
    load_after = _galaxy.card_load()
    # A timing is a measurement only on a free card, and never on the CPU route
    # (patch_cpu_lane makes jax report "gpu" there, so check the request itself).
    if L.cpu_requested():
        invalid_reason = "cpu smoke (interpreted kernels)"
    elif not _galaxy.card_free(load_before, load_after):
        invalid_reason = "contended card"
    else:
        invalid_reason = None
    if invalid_reason is not None:
        for rec in records:
            if rec.get("valid"):
                rec["valid"] = False
                rec["invalid"] = True
                rec["invalid_reason"] = invalid_reason
    config = {
        "n": args.ns,
        "theta": scfg["theta"],
        "order": scfg["max_order"],
        "basis": ["real"],
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": args.dtype,
        "mode": "single",
        "solver": scfg,
        "mac_type": scfg["mac_type"],
        "leaf_size": scfg["leaf_size"],
        "distribution": args.distribution,
        "softening": float(args.softening),
        "wrt": args.wrt_list,
        "arms": args.arm_list,
        "lanes": ["fused"],
        "reps": int(args.reps),
        "loss": "sum(w . acc), w fixed-seed standard normal (grad_bench_lib.weights)",
        "timed_region": (
            "step: FusedForceProvider.step vs value_and_grad / grad through "
            "differentiable_step (topology rebuilt on both sides); fixed_topology: "
            "provider.accelerations on the prepare's topology vs "
            "differentiable_accelerations with its gradient plan. Every array a jit "
            "argument, compiled before timing; ratio = min(value_and_grad) / "
            "min(forward) over two interleaved series"
        ),
        "cpu_route_env": env_set or None,
        "card_load_before": load_before,
        "card_load_after": load_after,
        "note": (
            "rows with `error` (incl. `invalid: true`: a capacity or gradient "
            "verdict failed) are not measurements and are not plotted. On a CPU "
            "every kernel is interpreted: times are smoke only."
        ),
    }
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config=config,
        meta=runmeta.run_meta({"argv": sys.argv[1:]}),
        data={"records": records},
    )
    print(f"\nwrote {out}")
    return 0 if all("error" not in r for r in records) else 1


# --------------------------------------------------------------------------
# mesh (WP-G): one row of bench/multigpu_grad_gate.py
# --------------------------------------------------------------------------


def _gate_command(args: argparse.Namespace, n: int, out: pathlib.Path) -> list:
    gate = REPO_ROOT / "bench" / "multigpu_grad_gate.py"
    cmd = [sys.executable, str(gate), "--mode", "mesh", "--n", str(n)]
    cmd += ["--ndev", str(args.ndev), "--ic", args.distribution]
    cmd += ["--seed", str(args.seed), "--softening", str(args.softening)]
    cmd += ["--dtype", args.dtype, "--reps", str(args.reps), "--out", str(out)]
    if args.mac == "preset":
        cmd.append("--galaxy")
    else:
        if args.theta is not None:
            cmd += ["--theta", str(args.theta)]
        if args.order is not None:
            cmd += ["--order", str(args.order)]
    if L.cpu_requested():
        # the CPU gates' sizes: a fixed leaf capacity and CrossCapacities' defaults
        cmd += ["--cpu-smoke", "--leaf-capacity", "128", "--cross", "default"]
    return cmd + args.gate_args.split()


def _mesh_record(args: argparse.Namespace, n: int, row: dict, rc: int) -> dict:
    rec: dict[str, Any] = {
        "n": int(n),
        "basis": "real",
        "lane": "mesh",
        "mode": "step",
        "ndev": int(args.ndev),
        "seed": int(args.seed),
        "gate_returncode": rc,
        "grads": {},
    }
    t = row.get("time_ms")
    if rc != 0 or not row.get("complete") or not t:
        rec["error"] = row.get("error") or f"bench/multigpu_grad_gate.py exited {rc}"
        return rec
    f_min = min(t["force"][0], t["force_b"][0]) / 1e3
    g_min = min(t["main"][0], t["main_b"][0]) / 1e3
    force_ok = bool(row["force"]["bits_ok"])
    g_ok = bool(row["grad_bits"]["grad_ok"]) and bool(row["grad_finite"])
    rec.update(
        forward_min_s=f_min,
        forward_median_s=min(t["force"][1], t["force_b"][1]) / 1e3,
        force_ok=force_ok,
        forward_differing_elements=(row.get("MG1") or {}).get("differing"),
        valid=force_ok and g_ok,
        contended=row.get("contended"),
    )
    rec["grads"]["both"] = {
        "forward_backward_min_s": g_min,
        "forward_backward_median_s": min(t["main"][1], t["main_b"][1]) / 1e3,
        "ratio": g_min / f_min,
        "grad_only_min_s": t["grad"][0] / 1e3 if "grad" in t else None,
        "grad_only_ratio": row.get("ratio_grad_over_force"),
        "grad_ok": g_ok,
        "device_peak_B_per_p": row.get("peak_B_per_p"),
        "memory_analysis": row.get("memory_analysis"),
    }
    if not rec["valid"]:
        rec["invalid"] = True
        rec["error"] = "capacity or gradient verdict failed: not a measurement"
    elif L.cpu_requested() or row.get("contended") is not False:
        # a correct row, but not a timing: interpreted kernels, or a card that
        # was shared (or whose processes could not be read). Not an error.
        rec["valid"] = False
        rec["invalid"] = True
        rec["invalid_reason"] = (
            "cpu smoke (interpreted kernels)"
            if L.cpu_requested()
            else "contended card (or nvidia-smi unreadable)"
        )
    print(
        f"mesh N={n} ndev={args.ndev}: fwd {f_min*1e3:.2f} ms, v&g {g_min*1e3:.2f} "
        f"ms, ratio {g_min/f_min:.2f}, valid={rec['valid']}",
        flush=True,
    )
    return rec


def _run_mesh(args: argparse.Namespace) -> int:
    n = args.ns[0]
    with tempfile.TemporaryDirectory() as tmp:
        row_path = pathlib.Path(tmp) / "row.json"
        cmd = _gate_command(args, n, row_path)
        print("running", " ".join(cmd), flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=dict(os.environ))
        row = json.loads(row_path.read_text()) if row_path.exists() else {}
    rec = _mesh_record(args, n, row, proc.returncode)
    rec["gate_wall_s"] = time.time() - t0
    rec["gate_row"] = row
    config = {
        "n": [int(n)],
        "theta": row.get("theta"),
        "order": row.get("order"),
        "basis": ["real"],
        "seed": int(args.seed),
        "device": (
            "cpu (emulated host devices)"
            if L.cpu_requested()
            else (row.get("devices") or [None])[0]
        ),
        "precision": args.dtype,
        "mode": "mesh",
        "ndev": int(args.ndev),
        "mac_type": row.get("mac"),
        "preset": "galaxy" if row.get("galaxy") else "large_n_gpu (geometric)",
        "leaf_size": row.get("leaf"),
        "softening": float(args.softening),
        "distribution": args.distribution,
        "ic_source": row.get("ic_source"),
        "wrt": ["both"],
        "lanes": ["mesh"],
        "reps": int(args.reps),
        "loss": "sum(w . acc) built outside the shard_map (make_mesh_grad_fn)",
        "timed_region": (
            "FusedMeshForce.shard (rebuild + evaluate) vs value_and_grad / grad "
            "through make_mesh_grad_fn; bench/multigpu_grad_gate.py's protocol"
        ),
        "gate_command": cmd[1:],
    }
    out = jsonio.write_result(
        args.json_out or DEFAULT_MESH_OUT,
        config=config,
        meta=runmeta.run_meta({"argv": sys.argv[1:]}),
        data={"records": [rec]},
    )
    print(f"\nwrote {out}")
    return 0 if "error" not in rec else 1


def main() -> int:
    """Run one mode and write its artifact; nonzero if any row failed."""
    args = _parse_args()
    runmeta.select_gpu(
        args.gpu_select, num_gpus=args.ndev if args.mode == "mesh" else 1
    )
    if args.mode == "mesh":
        return _run_mesh(args)
    return _run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
