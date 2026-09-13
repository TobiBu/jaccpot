#!/usr/bin/env python
"""Attribute the XLA scatter fusions of the jitted large-N gradient (plan fast-gradients, follow-up).

The Perfetto trace of ``jax.grad`` at N = 2x10^5 shows seven anonymous ``input_scatter_fusion_N``
kernels costing ~13 ms -- more than the five reverse Pallas kernels together. This compiles the SAME
jitted gradient on the GPU, dumps its optimized HLO (whose fusion names are the trace's names), and
prints, per scatter fusion: the scatter's operand/update/result shapes, its combiner, and the
parameters that feed it (traced back through the fusion's operands to named ops), plus the kernel's
per-call time when a trace JSON from ``grad_step_profile.py`` is given.

    PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=$PWD YGGDRAX_WORKTREE=... \
      $B/codes/run_when_idle.sh 36000 python bench/grad_scatter_attribution.py \
      --trace-json $B/artifacts/grad/phase2_fast_kernels.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_BENCH_ROOT = "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--theta", type=float, default=0.8)
    ap.add_argument("--softening", type=float, default=1e-7)
    ap.add_argument("--bench-root", default=DEFAULT_BENCH_ROOT)
    ap.add_argument("--trace-json", default=None)
    ap.add_argument("--env", nargs="+", default=[], metavar="KEY=VAL")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bench_root = Path(args.bench_root)
    sys.path.insert(0, str(bench_root))
    sys.path.insert(0, str(bench_root / "codes"))
    from common.gpu_guard import idle_gpus, pick_idle_gpus, set_cuda_visible
    from common.ic import IC_GENERATORS
    from compare_force import (
        FAST_LANE_ENV_BY_LEAF,
        apply_fast_lane_env,
        fast_lane_overrides_for_leaf,
    )

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ["JACCPOT_RETAIN_FAR_PAIRS_FOR_GRAD"] = "1"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        set_cuda_visible(pick_idle_gpus(1))
    extra = dict(kv.split("=", 1) for kv in args.env)
    overrides = fast_lane_overrides_for_leaf(args.leaf, args.n, extra)
    preset_trav = dict(
        (FAST_LANE_ENV_BY_LEAF.get(args.leaf) or {}).get("_traversal_overrides", {})
    )
    overrides.update(extra)
    apply_fast_lane_env(args.n, overrides=overrides)

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
    from jaccpot.runtime._large_n_grad import prepare_large_n_grad_plan

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
    from yggdrax._cell_partition import adaptive_cell_leaf_partition_numpy
    from yggdrax.bounds import infer_bounds
    from yggdrax.morton import morton_encode

    codes = np.sort(np.asarray(morton_encode(P, infer_bounds(P))).astype(np.uint64))
    k = adaptive_cell_leaf_partition_numpy(codes, leaf_size=args.leaf)[0].size
    leaf_capacity = 1 << int(np.ceil(np.log2(1.25 * k)))
    solver = FastMultipoleMethod(
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
                leaf_partition="cells",
                leaf_capacity=leaf_capacity,
            ),
            farfield=FarFieldConfig(mode="auto", retain_far_pairs_for_grad=True),
            nearfield=NearFieldConfig(mode="auto"),
            runtime=runtime_cfg,
            mac_type="dehnen",
        ),
        fixed_order=args.order,
    )
    state = solver.prepare_state(
        P, M, leaf_size=args.leaf, max_order=args.order, theta=args.theta
    )
    plan = prepare_large_n_grad_plan(solver, state)
    L = int(np.asarray(state.neighbor_list.counts).shape[0])
    W = int(state.max_leaf_size)
    nodes = int(np.asarray(state.tree.parent).shape[0])
    edges = int(np.asarray(state.neighbor_list.counts).sum())
    print(
        f"N {n} leaves {L} W {W} nodes {nodes} far cap {plan.num_far_pairs} near edges {edges}",
        flush=True,
    )

    def loss(p, m):
        return jnp.sum(
            solver.differentiable_accelerations(state, p, m, grad_plan=plan) ** 2
        )

    compiled = jax.jit(jax.value_and_grad(loss, argnums=(0, 1))).lower(P, M).compile()
    hlo = compiled.as_text()
    out_dir = Path(args.out or (bench_root / "artifacts" / "grad"))
    out_dir.mkdir(parents=True, exist_ok=True)
    hlo_path = out_dir / "grad_jit_hlo_gpu.txt"
    hlo_path.write_text(hlo)
    print(f"wrote {hlo_path} ({len(hlo)//1024} KB)", flush=True)

    trace_ms: dict[str, float] = {}
    if args.trace_json:
        d = json.load(open(args.trace_json))
        for kk in d.get("grad_trace", {}).get("top", []):
            trace_ms[kk["name"]] = float(kk["total_ms_per_call"])

    # --- parse: every fusion computation and the scatter instructions inside it
    comps: dict[str, list[str]] = {}
    cur = None
    for line in hlo.splitlines():
        m = re.match(r"^(?:ENTRY )?%?([\w.\-]+) \(", line)
        if m and line.rstrip().endswith("{"):
            cur = m.group(1)
            comps[cur] = []
        elif cur is not None:
            if line.strip() == "}":
                cur = None
            else:
                comps[cur].append(line.rstrip())
    # fusion instruction lines in the entry / callers: "%input_scatter_fusion_10 = f32[...] fusion(%a, %b, ...), kind=kInput, calls=%..."
    fusions = {}
    for cname, body in comps.items():
        for line in body:
            m = re.match(
                r"\s*%?([\w.\-]+) = (\S+) fusion\((.*?)\), kind=(\w+), calls=%?([\w.\-]+)",
                line,
            )
            if m and "scatter" in m.group(1):
                fusions[m.group(1)] = dict(
                    result=m.group(2),
                    operands=m.group(3),
                    kind=m.group(4),
                    calls=m.group(5),
                    caller=cname,
                )
    rows = []
    for fname, f in sorted(fusions.items(), key=lambda kv: -trace_ms.get(kv[0], 0.0)):
        body = comps.get(f["calls"], [])
        scat = [ln.strip() for ln in body if " scatter(" in ln]
        shapes = []
        for s in scat:
            m = re.match(r"%?[\w.\-]+ = (\S+) scatter\((.*?)\), update_window_dims", s)
            if m:
                ops = re.findall(
                    r"([a-z0-9]+\[[0-9,]*\](?:\{[^}]*\})?)\s*%", m.group(2) + " %"
                )
                shapes.append((m.group(1), ops))
            comb = re.search(r"to_apply=%?([\w.\-]+)", s)
            if comb:
                shapes.append(("combiner", comb.group(1)))
        # what feeds the fusion: operand names -> their defining instructions in the caller
        caller_body = comps.get(f["caller"], [])
        feeders = []
        for op in re.findall(r"%([\w.\-]+)", f["operands"]):
            for ln in caller_body:
                if re.match(rf"\s*%{re.escape(op)} = ", ln):
                    kind = re.match(r"\s*%[\w.\-]+ = (\S+) ([\w\-]+)\(", ln)
                    feeders.append(
                        f"{op}: {kind.group(2) if kind else '?'} {kind.group(1) if kind else ''}"
                    )
                    break
        ms = trace_ms.get(fname)
        rows.append(
            dict(
                fusion=fname,
                ms_per_call=ms,
                result=f["result"],
                scatters=shapes,
                feeders=feeders,
            )
        )
        print(
            f"\n== {fname}  {'' if ms is None else f'{ms:.3f} ms'}  result {f['result']}"
        )
        for sh in shapes:
            print("   scatter:", sh)
        for fe in feeders[:8]:
            print("   feeder:", fe[:140])
    with open(out_dir / "grad_scatter_attribution.json", "w") as fh:
        json.dump(
            dict(
                n=n,
                leaves=L,
                W=W,
                nodes=nodes,
                far_cap=plan.num_far_pairs,
                near_edges=edges,
                rows=rows,
            ),
            fh,
            indent=2,
        )
    print("\nwrote", out_dir / "grad_scatter_attribution.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
