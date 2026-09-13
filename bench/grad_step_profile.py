#!/usr/bin/env python
"""Phase 1 of ``fast-gradients-for-the-sub10ms-fmm``: time ``jax.grad`` of the fused step, attributed.

Builds the sub-10 ms operating point (N = 2x10^5 Plummer, Morton-cell leaves of 64,
theta 0.8, order 5, fp32, ``preset="large_n_gpu"``) with the far pairs retained for
the gradient, then measures on one idle A100:

* the production eval (``strict_fused_prepared_eval_fn``) -- the forward the record quotes;
* the differentiable forward (``differentiable_accelerations``) and ``jax.grad`` of a
  quadratic loss through it, w.r.t. positions and masses;
* the same split by stage: the upward chain alone (P2M + M2M, loss on the packed
  multipoles), the far field alone and the near field alone
  (``evaluate_large_n_state_at_positions_and_masses_sorted(include_*)``);
* a Perfetto kernel table of the full reverse (kernel names -> stages where named;
  the loop cascades' XLA fusions are anonymous, which is why the stage split above
  exists).

Every timing is min / IQR of ``--reps`` warm calls under ``GpuMonitor``; a contended row
is flagged in the JSON and must not be quoted. Uses the bench harness of
``Odisseo-bench-multigpu/benchmark_multigpu`` (``--bench-root``).

    B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
    PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=$PWD YGGDRAX_WORKTREE=... \
      $B/codes/run_when_idle.sh 36000 /export/home/tbuck/jaccpot/.venv/bin/python \
      bench/grad_step_profile.py --tag phase0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

DEFAULT_BENCH_ROOT = "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu"


def _stage_of(name: str) -> str:
    low = name.lower()
    for needle, label in (
        ("nearfield_leafpair_csr", "near_csr_pallas"),
        ("nearfield_leafpair", "near_rect_pallas"),
        ("nearfield", "near_other"),
        ("m2l_real_csr", "m2l_csr_pallas"),
        ("m2l_real", "m2l_pallas_other"),
        ("m2l", "m2l_other"),
        ("m2m_real_level", "m2m_pallas"),
        ("l2l_real_level", "l2l_pallas"),
        ("p2m_real_leaf", "p2m_pallas"),
        ("p2m_rev", "p2m_rev_pallas"),
        ("m2m_rev", "m2m_rev_pallas"),
        ("l2l_rev", "l2l_rev_pallas"),
        ("m2l_rev", "m2l_rev_pallas"),
        ("near_rev", "near_rev_pallas"),
        ("mutual_walk", "walk"),
        ("treecode_walk", "walk"),
        ("sort", "sort"),
        ("scatter", "scatter"),
        ("gather", "gather"),
        ("reduce", "reduce"),
        ("memcpy", "memcpy"),
        ("memset", "memset"),
    ):
        if needle in low:
            return label
    return "xla_fusion_other" if "fusion" in low else "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--theta", type=float, default=0.8)
    ap.add_argument("--softening", type=float, default=1e-7)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--leaf-partition", default="cells", choices=["buckets", "cells"])
    ap.add_argument("--bench-root", default=DEFAULT_BENCH_ROOT)
    ap.add_argument("--env", nargs="+", default=[], metavar="KEY=VAL")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--no-jit", action="store_true", help="time the eager grad only")
    ap.add_argument(
        "--skip-eager-grad",
        action="store_true",
        help="eager grad retraces per call (~s); jit only",
    )
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bench_root = Path(args.bench_root)
    sys.path.insert(0, str(bench_root))
    sys.path.insert(0, str(bench_root / "codes"))
    from common.gpu_guard import (  # noqa: E402
        idle_gpus,
        pick_idle_gpus,
        set_cuda_visible,
        timed_calls,
    )
    from common.ic import IC_GENERATORS  # noqa: E402
    from compare_force import (  # noqa: E402
        FAST_LANE_ENV_BY_LEAF,
        apply_fast_lane_env,
        fast_lane_overrides_for_leaf,
    )

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ["JACCPOT_RETAIN_FAR_PAIRS_FOR_GRAD"] = "1"
    if "jax" in sys.modules:
        raise SystemExit("JAX was imported before this ran; restart the process")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        devices = [int(x) for x in cvd.split(",") if x.strip()][:1]
        idle, reasons = idle_gpus(settle_s=2.0, samples=4)
        if devices[0] not in idle and not args.allow_busy:
            raise SystemExit(
                f"CUDA_VISIBLE_DEVICES names a busy card: {reasons.get(devices[0])}"
            )
    else:
        devices = pick_idle_gpus(1)
        set_cuda_visible(devices)

    extra = dict(kv.split("=", 1) for kv in args.env)
    overrides = fast_lane_overrides_for_leaf(args.leaf, args.n, extra)
    preset_trav = dict(
        (FAST_LANE_ENV_BY_LEAF.get(args.leaf) or {}).get("_traversal_overrides", {})
    )
    overrides.update(extra)
    env = apply_fast_lane_env(args.n, overrides=overrides)

    import jax
    import jax.numpy as jnp

    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        RuntimePolicyConfig,
        TraversalOverrides,
        TreeConfig,
    )
    from jaccpot.runtime._large_n_grad import (
        _engine,
        evaluate_large_n_state_at_positions_and_masses_sorted,
        prepare_large_n_grad_plan,
    )
    from jaccpot.runtime._level_shapes import pallas_cascades_enabled
    from jaccpot.runtime.grad_options import resolve_grad_options

    tag = (
        args.tag
        or f"grad_{args.leaf_partition}{args.leaf}_th{args.theta:g}_p{args.order}"
    )
    print(
        f"[{tag}] devices={devices} load={os.getloadavg()} overrides={json.dumps(overrides)}",
        flush=True,
    )

    pos, mass = IC_GENERATORS["plummer"](args.n, seed=0)
    n = int(pos.shape[0])
    P = jnp.asarray(pos, jnp.float32)
    M = jnp.asarray(mass, jnp.float32)

    runtime_cfg = RuntimePolicyConfig()
    if preset_trav:
        runtime_cfg = RuntimePolicyConfig(
            traversal_config=TraversalOverrides(
                **{k: int(v) for k, v in preset_trav.items()}
            )
        )
    leaf_capacity = None
    if args.leaf_partition == "cells":
        from yggdrax._cell_partition import adaptive_cell_leaf_partition_numpy
        from yggdrax.bounds import infer_bounds
        from yggdrax.morton import morton_encode

        _b = infer_bounds(P)
        _codes = np.sort(np.asarray(morton_encode(P, _b)).astype(np.uint64))
        _k = adaptive_cell_leaf_partition_numpy(_codes, leaf_size=args.leaf)[0].size
        leaf_capacity = 1 << int(np.ceil(np.log2(1.25 * _k)))
        print(
            f"[{tag}] cell leaves: {_k} live -> leaf_capacity {leaf_capacity}",
            flush=True,
        )

    def build_solver():
        return FastMultipoleMethod(
            preset="large_n_gpu",
            runtime_path="large_n",
            basis="real",
            theta=args.theta,
            G=1.0,
            softening=args.softening,
            working_dtype=jnp.float32,
            advanced=FMMAdvancedConfig(
                tree=TreeConfig(
                    mode="static_radix",
                    leaf_target=args.leaf,
                    leaf_partition=args.leaf_partition,
                    leaf_capacity=leaf_capacity,
                ),
                farfield=FarFieldConfig(mode="auto", retain_far_pairs_for_grad=True),
                nearfield=NearFieldConfig(mode="auto"),
                runtime=runtime_cfg,
                mac_type="dehnen",
            ),
            fixed_order=args.order,
        )

    result: dict = dict(
        tag=tag,
        n=n,
        leaf=args.leaf,
        order=args.order,
        theta=args.theta,
        reps=args.reps,
        devices=devices,
        env_overrides=overrides,
        traversal_overrides=preset_trav,
        fast_lane_env=env,
        worktree=os.environ.get("JACCPOT_WORKTREE"),
        leaf_partition=args.leaf_partition,
        leaf_capacity=leaf_capacity,
        load_at_start=os.getloadavg(),
        pallas_cascades_enabled_outside_grad=bool(pallas_cascades_enabled()),
        git_head=os.popen(
            f"git -C {os.environ.get('JACCPOT_WORKTREE', '.')} rev-parse --short HEAD"
        )
        .read()
        .strip(),
    )
    out_path = Path(args.out or (bench_root / "artifacts" / "grad" / f"{tag}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write():
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)

    def block(x):
        return jax.block_until_ready(x)

    def timed(label, fn):
        try:
            out, timing, cont = timed_calls(
                fn, repeats=args.reps, warmup=args.warmup, devices=devices, block=block
            )
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            first = (str(e).splitlines() or [""])[0][:300]
            print(f"[{tag}] {label}: FAILED {type(e).__name__}: {first}", flush=True)
            # the innermost jaccpot frame names the kernel that has no AD rule
            frames = [
                ln.strip()
                for ln in tb.splitlines()
                if "jaccpot/" in ln and "File" in ln
            ]
            result[label] = dict(
                error=f"{type(e).__name__}: {str(e)[:2000]}",
                traceback=tb[-4000:],
                innermost_jaccpot_frames=frames[-3:],
            )
            if frames:
                print(f"[{tag}]   innermost: {frames[-1]}", flush=True)
            write()
            return None
        row = dict(timing=timing, contention=cont.as_dict(), load=os.getloadavg())
        result[label] = row
        print(
            f"[{tag}] {label}: min {timing['min']*1e3:.2f} ms (IQR {timing['iqr']*1e3:.2f}, med {timing['median']*1e3:.2f}) "
            f"flags={cont.flags or '-'} load={os.getloadavg()[0]:.0f}",
            flush=True,
        )
        write()
        return out

    # ------------------------------------------------------------- production eval
    solver = build_solver()
    t0 = time.perf_counter()
    prepared, eval_fn = solver.strict_fused_prepared_eval_fn(
        positions=P,
        masses=M,
        leaf_size=args.leaf,
        max_order=args.order,
        theta=args.theta,
    )
    a_eval = np.asarray(block(eval_fn(prepared)), np.float64)
    result["prepare_eval_s_incl_compile"] = time.perf_counter() - t0
    from common.error import rel_errors
    from common.reference import direct_accelerations

    ref_idx = np.sort(np.random.default_rng(12345).choice(n, 4096, replace=False))
    a_ref = direct_accelerations(
        pos,
        mass,
        G=1.0,
        softening=args.softening,
        block_size=1024,
        target_indices=ref_idx,
    )
    result["eval_production_aggL2"] = float(rel_errors(a_eval[ref_idx], a_ref)["aggL2"])
    print(
        f"[{tag}] production eval aggL2 vs direct sum {result['eval_production_aggL2']:.3e}",
        flush=True,
    )
    timed("eval_production", lambda: eval_fn(prepared))
    prepared = eval_fn = None  # release the fused state before the differentiable one

    # ------------------------------------------------------------ differentiable state
    solver = build_solver()
    t0 = time.perf_counter()
    state = solver.prepare_state(
        P, M, leaf_size=args.leaf, max_order=args.order, theta=args.theta
    )
    result["prepare_state_s"] = time.perf_counter() - t0
    result["state_type"] = type(state).__name__
    print(
        f"[{tag}] state {type(state).__name__} in {result['prepare_state_s']:.1f} s",
        flush=True,
    )
    try:
        plan = prepare_large_n_grad_plan(solver, state)
        result["far_pairs"] = int(plan.num_far_pairs)
        if plan.far_pair_active_count is not None:
            result["far_pairs_active"] = int(plan.far_pair_active_count)
    except Exception as e:  # noqa: BLE001
        print(
            f"[{tag}] prepare_large_n_grad_plan FAILED: {type(e).__name__}: {str(e)[:400]}",
            flush=True,
        )
        result["grad_plan_error"] = f"{type(e).__name__}: {str(e)[:2000]}"
        write()
        return 2
    tree = state.tree
    nbr = state.neighbor_list
    counts = np.asarray(nbr.counts)
    offs = np.asarray(nbr.offsets)
    nbrs = np.asarray(nbr.neighbors)
    leaf_nodes = np.asarray(nbr.leaf_indices)
    # is the near CSR symmetric? (the CSR near-field reverse assumes it)
    rows_t = np.repeat(np.arange(counts.size), counts)
    cols = (
        nbrs[
            np.concatenate(
                [np.arange(offs[i], offs[i] + counts[i]) for i in range(counts.size)]
            )
        ]
        - leaf_nodes[0]
        if counts.sum()
        else np.zeros(0, int)
    )
    fwd = set(zip(rows_t.tolist(), cols.tolist()))
    rev = set(zip(cols.tolist(), rows_t.tolist()))
    result["near_csr"] = dict(
        edges=int(counts.sum()),
        rows=int(counts.size),
        row_max=int(counts.max()),
        symmetric=bool(fwd == rev),
        asymmetric_edges=int(len(fwd ^ rev)),
    )
    lo = np.asarray(tree.level_offsets)
    result["tree"] = dict(
        total_nodes=int(np.asarray(tree.parent).shape[0]),
        num_internal=int(np.asarray(tree.left_child).shape[0]),
        levels_live=int(np.count_nonzero(np.diff(lo))),
        widest_level=int(np.max(np.diff(lo))),
    )
    # single-child parents (delta == 0 in M2M): how many?
    lc, rc = np.asarray(tree.left_child), np.asarray(tree.right_child)
    result["tree"]["single_child_parents"] = int(np.sum((lc < 0) | (rc < 0)))
    print(
        f"[{tag}] far pairs cap {result.get('far_pairs')} active {result.get('far_pairs_active')} near edges {result['near_csr']['edges']} symmetric={result['near_csr']['symmetric']} "
        f"tree {result['tree']}",
        flush=True,
    )
    write()

    options = resolve_grad_options(None, num_particles=n, supports_fast_lane=False)
    result["grad_options"] = dict(
        nearfield_lane=options.nearfield_lane,
        cascade_pallas=options.cascade_pallas,
        fused_m2l_pallas=options.fused_m2l_pallas,
    )

    def loss(p, m):
        return jnp.sum(
            solver.differentiable_accelerations(state, p, m, grad_plan=plan) ** 2
        )

    fwd_fn = lambda p, m: solver.differentiable_accelerations(
        state, p, m, grad_plan=plan
    )  # noqa: E731
    a_diff = timed("forward_differentiable_eager", lambda: fwd_fn(P, M))
    if a_diff is not None:
        a_diff = np.asarray(a_diff, np.float64)
        result["forward_vs_eval_relL2"] = float(
            np.linalg.norm(a_diff - a_eval) / np.linalg.norm(a_eval)
        )
        result["forward_differentiable_aggL2"] = float(
            rel_errors(a_diff[ref_idx], a_ref)["aggL2"]
        )
        print(
            f"[{tag}] differentiable forward vs production eval rel-L2 {result['forward_vs_eval_relL2']:.3e}; "
            f"vs direct sum aggL2 {result['forward_differentiable_aggL2']:.3e}",
            flush=True,
        )

    grad_fn = jax.grad(loss, argnums=(0, 1))
    g = timed("grad_eager", lambda: grad_fn(P, M)) if not args.skip_eager_grad else None
    if g is not None:
        gp, gm = (np.asarray(x, np.float64) for x in g)
        result["grad_eager"]["finite"] = bool(
            np.all(np.isfinite(gp)) and np.all(np.isfinite(gm))
        )
        result["grad_eager"]["norm_pos"] = float(np.linalg.norm(gp))
        result["grad_eager"]["norm_mass"] = float(np.linalg.norm(gm))
        print(
            f"[{tag}] grad finite={result['grad_eager']['finite']} |gp|={np.linalg.norm(gp):.4e} |gm|={np.linalg.norm(gm):.4e}",
            flush=True,
        )
        write()
    traced_grad = None if g is None else grad_fn
    if not args.no_jit:
        try:
            jf = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
            t0 = time.perf_counter()
            block(jf(P, M))
            result["grad_jit_compile_s"] = time.perf_counter() - t0
            if timed("grad_jit", lambda: jf(P, M)) is not None:
                traced_grad = (
                    jf  # the trace prefers the compiled grad (no host retrace noise)
                )
        except Exception as e:  # noqa: BLE001
            print(
                f"[{tag}] jit(grad) FAILED: {type(e).__name__}: {str(e).splitlines()[0][:300]}",
                flush=True,
            )
            result["grad_jit"] = dict(error=f"{type(e).__name__}: {str(e)[:2000]}")
            write()

    # -------------------------------------------------------------- stage split
    engine = _engine(solver)
    Ps = jnp.asarray(state.positions_sorted)
    Ms = jnp.asarray(state.masses_sorted)

    def upward_loss(p, m):
        up = engine.prepare_upward_sweep(
            tree,
            p,
            m,
            max_order=plan.order,
            center_mode=plan.center_mode,
            max_leaf_size=plan.max_leaf_size,
        )
        return jnp.sum(up.multipoles.packed**2)

    def far_loss(p, m):
        return jnp.sum(
            evaluate_large_n_state_at_positions_and_masses_sorted(
                solver, state, p, m, plan=plan, include_near=False
            )
            ** 2
        )

    def near_loss(p, m):
        return jnp.sum(
            evaluate_large_n_state_at_positions_and_masses_sorted(
                solver, state, p, m, plan=plan, include_far=False
            )
            ** 2
        )

    for label, fn in (("upward", upward_loss), ("far", far_loss), ("near", near_loss)):
        timed(f"{label}_forward_eager", lambda fn=fn: fn(Ps, Ms))
        timed(f"{label}_grad_eager", lambda fn=fn: jax.grad(fn, argnums=(0, 1))(Ps, Ms))
        if not args.no_jit:
            try:
                jf = jax.jit(jax.grad(fn, argnums=(0, 1)))
                block(jf(Ps, Ms))
                timed(f"{label}_grad_jit", lambda jf=jf: jf(Ps, Ms))
            except Exception as e:  # noqa: BLE001
                print(
                    f"[{tag}] jit({label} grad) FAILED: {type(e).__name__}: {str(e).splitlines()[0][:300]}",
                    flush=True,
                )
                result[f"{label}_grad_jit"] = dict(
                    error=f"{type(e).__name__}: {str(e)[:2000]}"
                )
                write()

    # ------------------------------------------------------------------- trace
    if not args.no_trace and traced_grad is not None:
        from profile_eval import analyse, load_perfetto

        tdir = bench_root / "artifacts" / "traces" / f"grad_{tag}"
        tdir.mkdir(parents=True, exist_ok=True)
        calls = 3
        with jax.profiler.trace(str(tdir), create_perfetto_trace=True):
            for _ in range(calls):
                block(traced_grad(P, M))
        res = analyse(load_perfetto(tdir), calls)
        stages: dict[str, dict] = {}
        for k in res["top"]:
            s = stages.setdefault(
                _stage_of(k["name"]),
                dict(ms_per_call=0.0, launches_per_call=0.0, kernels=0),
            )
            s["ms_per_call"] += float(k["total_ms_per_call"])
            s["launches_per_call"] += float(k["count_per_call"])
            s["kernels"] += 1
        result["grad_trace"] = dict(
            per_call=res["per_call"], stages=stages, top=res["top"][:40]
        )
        print(
            f"[{tag}] grad trace: busy {res['per_call']['busy_ms']:.2f}/{res['per_call']['window_ms']:.2f} ms, "
            f"launches {res['per_call']['launches']:.0f}; stages "
            + ", ".join(
                f"{k}={v['ms_per_call']:.2f}"
                for k, v in sorted(stages.items(), key=lambda kv: -kv[1]["ms_per_call"])
            ),
            flush=True,
        )
        write()
    result["load_at_end"] = os.getloadavg()
    write()
    print(f"[{tag}] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
