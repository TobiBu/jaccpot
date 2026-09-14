#!/usr/bin/env python
"""Why is the L2L reverse 5.5x its forward while the M2M reverse is 1.3x? (plan fast-gradients, follow-up).

After the scatter round the biggest single kernel of the jitted gradient at N = 2x10^5 is
``l2l_rev_real_level_p5``: 4.2 ms over 47 launches, 5.5x its own forward, where the M2M
reverse is 1.3x its forward. The two reverses differ in shape -- the L2L one is one program
per PARENT doing TWO ``jax.vjp`` traces of the translate body (both children) and two
geometry-cotangent stores, the M2M one is one program per NODE doing one -- so the suspect is
register pressure / spilling in the doubled body, not arithmetic.

This times the four level kernels standalone on the production tree shape, sweeping the warp
count, so the answer is a table rather than a guess. It builds the tree once with the same
cell partition as the step and drives the kernels directly (no FMM, no grad).

    PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=$PWD YGGDRAX_WORKTREE=... \
      $B/codes/run_when_idle.sh 36000 python bench/grad_cascade_reverse_microbench.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_BENCH_ROOT = "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--bench-root", default=DEFAULT_BENCH_ROOT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bench_root = Path(args.bench_root)
    sys.path.insert(0, str(bench_root))
    sys.path.insert(0, str(bench_root / "codes"))
    from common.gpu_guard import pick_idle_gpus, set_cuda_visible, timed_calls
    from common.ic import IC_GENERATORS

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        set_cuda_visible(pick_idle_gpus(1))
    devices = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]

    import jax
    import jax.numpy as jnp
    from yggdrax._tree_impl import build_static_cells_tree
    from yggdrax.bounds import infer_bounds
    from yggdrax.morton import morton_encode
    from yggdrax.tree_moments import compute_tree_mass_moments

    from jaccpot.pallas.cascade_real_level import (
        l2l_real_levels_pallas,
        l2l_real_levels_reverse_pallas,
        m2m_real_levels_pallas,
        m2m_real_levels_reverse_pallas,
    )

    pos, mass = IC_GENERATORS["plummer"](args.n, seed=0)
    P = jnp.asarray(pos, jnp.float32)
    M = jnp.asarray(mass, jnp.float32)
    codes = np.sort(np.asarray(morton_encode(P, infer_bounds(P))).astype(np.uint64))
    from yggdrax._cell_partition import adaptive_cell_leaf_partition_numpy

    k = adaptive_cell_leaf_partition_numpy(codes, leaf_size=args.leaf)[0].size
    capacity = 1 << int(np.ceil(np.log2(1.25 * k)))
    topo, ps, ms, _ = build_static_cells_tree(
        P,
        M,
        infer_bounds(P),
        leaf_size=args.leaf,
        leaf_capacity=capacity,
        return_reordered=True,
    )
    com = jnp.asarray(
        compute_tree_mass_moments(topo, ps, ms).center_of_mass, jnp.float32
    )
    total = int(np.asarray(topo.parent).shape[0])
    num_internal = int(np.asarray(topo.left_child).shape[0])
    offs = topo.level_offsets
    counts = np.asarray(offs)[1:] - np.asarray(offs)[:-1]
    num_levels = int(np.count_nonzero(counts)) + 1
    width = int(counts.max())
    C = (args.order + 1) ** 2
    rng = np.random.default_rng(3)
    coeffs = jnp.asarray(rng.standard_normal((total, C)), jnp.float32)
    cot = jnp.asarray(rng.standard_normal((total, C)), jnp.float32)
    print(
        f"N {args.n} nodes {total} internal {num_internal} levels {num_levels} widest {width} C {C}",
        flush=True,
    )

    common = dict(order=args.order, level_batch_width=width)
    variants = {
        "m2m_fwd": lambda w: jax.jit(
            lambda x: m2m_real_levels_pallas(
                x,
                com,
                topo.left_child,
                topo.right_child,
                topo.nodes_by_level,
                offs,
                num_internal=num_internal,
                num_levels=num_levels,
                num_warps=w,
                **common,
            )
        ),
        "m2m_rev": lambda w: jax.jit(
            lambda g: m2m_real_levels_reverse_pallas(
                coeffs,
                com,
                topo.parent,
                topo.nodes_by_level,
                offs,
                g,
                num_internal=num_internal,
                num_levels=num_levels,
                num_warps=w,
                **common,
            )
        ),
        "l2l_fwd": lambda w: jax.jit(
            lambda x: l2l_real_levels_pallas(
                x,
                com,
                topo.parent,
                topo.nodes_by_level,
                offs,
                num_levels=num_levels,
                num_warps=w,
                **common,
            )
        ),
        "l2l_rev": lambda w: jax.jit(
            lambda g: l2l_real_levels_reverse_pallas(
                coeffs,
                com,
                topo.parent,
                topo.left_child,
                topo.right_child,
                topo.nodes_by_level,
                offs,
                g,
                num_levels=num_levels,
                num_warps=w,
                **common,
            )
        ),
    }
    arg = {"m2m_fwd": coeffs, "l2l_fwd": coeffs, "m2m_rev": cot, "l2l_rev": cot}
    rows: dict[str, dict[int, float]] = {}
    for name, make in variants.items():
        rows[name] = {}
        for w in args.warps:
            try:
                f = make(w)
                _, timing, cont = timed_calls(
                    lambda f=f, name=name: f(arg[name]),
                    repeats=args.reps,
                    warmup=2,
                    devices=devices,
                    block=jax.block_until_ready,
                )
                rows[name][w] = timing["min"] * 1e3
                print(
                    f"  {name:8s} warps {w}: {timing['min']*1e3:7.3f} ms (IQR {timing['iqr']*1e3:.3f}) flags={cont.flags or '-'}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"  {name:8s} warps {w}: FAILED {type(e).__name__}: {str(e).splitlines()[0][:120]}",
                    flush=True,
                )
    print("\nbest per kernel and the reverse/forward ratio at each kernel's own best:")
    best = {k: min(v.items(), key=lambda kv: kv[1]) for k, v in rows.items() if v}
    for k, (w, t) in best.items():
        print(f"  {k:8s} {t:7.3f} ms at {w} warps")
    for fam in ("m2m", "l2l"):
        if f"{fam}_fwd" in best and f"{fam}_rev" in best:
            print(
                f"  {fam}: reverse / forward = {best[f'{fam}_rev'][1] / best[f'{fam}_fwd'][1]:.2f}x"
            )
    out = Path(
        args.out or (bench_root / "artifacts" / "grad" / "cascade_reverse_warps.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(
            dict(
                n=args.n,
                order=args.order,
                nodes=total,
                levels=num_levels,
                width=width,
                load=os.getloadavg(),
                ms=rows,
            ),
            fh,
            indent=2,
        )
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
