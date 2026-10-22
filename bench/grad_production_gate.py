"""The full force's gradient on the production layout: bitwise forward, memory, time (S8).

usage: python bench/grad_production_gate.py N [--galaxy] [--order P] [--leaf L]
       [--leaf-capacity pow2|auto] [--theta T] [--softening EPS] [--kernel K]
       [--params] [--reps R]
       [--no-time] [--no-forward] [--trace DIR] [--json OUT]

On one large-N state (``prepare_state`` with the far pairs retained; clipped Plummer,
``bench/dce_pins.py``'s IC; cells of ``L``, ``preset="large_n_gpu"``, fp32, the G1
pin's ``mac_type="dehnen"`` operating point by default, ``--galaxy`` for
``preset="galaxy"``, i.e. ``dehnen_error`` 1e-4 at p6) and its grad plan:

* the production force eval of the state (``evaluate_prepared_state``: the fused
  lane's ``evaluate_large_n_state`` -- near P2P, L2P, scatter) against the
  differentiable forward (``differentiable_accelerations``), jitted with the state and
  the plan as ARGUMENTS (closed over, ``jax.jit`` would embed the whole topology as
  constants): the number of elements whose bits differ (plan S8: zero);
* ``jax.value_and_grad`` of ``L = sum(w . acc)`` w.r.t. the positions and masses
  (input order; ``--params`` adds ``G`` and ``eps2`` as traced ``ForceParams``),
  and ``jax.grad`` alone, jitted the same way: each program's ``memory_analysis()``
  per particle and the largest arrays it defines in units of ``N`` floats (the S8
  gate: none of ``N x 16`` floats or more); the device's peak bytes in use after
  one call of the value-and-gradient (``--grad-only``: of the gradient), the state
  and the prepare included (the peak after the prepare is printed too);
* min and median of ``--reps`` calls of the force eval and of the gradient
  (``value_and_grad``, and ``grad`` alone, which drops the near field's forward:
  the loss is linear in it), the gradient A-vs-A (two calls, elements that
  differ), the ratios;
* ``--trace DIR``: a ``jax.profiler`` trace of three gradients (``DIR_grad``) and
  three force evals (``DIR_eval``).

Timings are hints unless the card had no other process before and after (printed).
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

import numpy as np

HERE = Path(__file__).resolve().parent


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("n", type=int)
    ap.add_argument("--galaxy", action="store_true", help='preset="galaxy"')
    ap.add_argument("--order", type=int, default=None, help="5; 6 with --galaxy")
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument(
        "--leaf-capacity",
        default="pow2",
        choices=("pow2", "auto"),
        help="pow2(1.25 x live) as G1 and grad_step_profile.py; auto: the "
        "production rule (TreeConfig(leaf_capacity='auto'))",
    )
    ap.add_argument("--theta", type=float, default=0.8)
    ap.add_argument("--softening", type=float, default=1e-7)
    ap.add_argument("--kernel", default=None, help="softening kernel (solver default)")
    ap.add_argument("--params", action="store_true", help="also d/dG and d/deps2")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--no-time", action="store_true", help="memory and parity only")
    ap.add_argument("--no-forward", action="store_true", help="skip the parity check")
    ap.add_argument(
        "--grad-only",
        action="store_true",
        help="run (and measure the peak of) jax.grad, not value_and_grad",
    )
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--trace", default=None, help="jax.profiler trace of 3 gradients, 3 evals"
    )
    return ap.parse_args()


def _card_processes() -> str:
    """The other compute processes nvidia-smi lists on this process's card."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                cvd,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi failed: {exc}"
    mine = str(os.getpid())
    others = [line for line in out.splitlines() if not line.startswith(mine + ",")]
    return "; ".join(others) if others else "none"


def _largest_arrays(text: str, n: int, top: int = 6) -> list:
    """The largest arrays the compiled program's instructions define, in N floats.

    Parameters, views and tuples are not allocations of the program; everything
    else that defines an array (fusions, custom calls, sorts, scatters) is.
    """
    views = ("parameter", "bitcast", "get-tuple-element", "tuple", "constant")
    seen: dict = {}
    for dt, dims, op in re.findall(
        r"= (f32|f64|s32|u32|s64|u64|pred|s8|u8)\[([0-9,]+)\][^ ]* ([a-z-]+)\(", text
    ):
        if op in views:
            continue
        size = int(np.prod([int(x) for x in dims.split(",")]))
        per = (
            8
            if dt in ("f64", "s64", "u64")
            else (1 if dt in ("pred", "s8", "u8") else 4)
        )
        key = f"{op} {dt}[{dims}]"
        seen[key] = size * per
    rows = sorted(seen.items(), key=lambda kv: -kv[1])[:top]
    return [{"array": k, "bytes": b, "N_floats": b / (4.0 * n)} for k, b in rows]


def _time(fn, *args, reps):
    import jax

    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return 1e3 * min(ts), 1e3 * float(np.median(ts))


def main() -> int:
    args = _args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    assert jax.devices()[0].platform == "gpu", jax.devices()
    sys.path.insert(0, str(HERE))
    from dce_pins import _plummer
    from yggdrax._cell_partition import adaptive_cell_leaf_partition_numpy
    from yggdrax.bounds import infer_bounds
    from yggdrax.morton import morton_encode

    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        ForceParams,
        NearFieldConfig,
        TreeConfig,
    )
    from jaccpot.runtime._large_n_grad import prepare_large_n_grad_plan

    order = args.order or (6 if args.galaxy else 5)
    row: dict = {
        "n": args.n,
        "galaxy": args.galaxy,
        "order": order,
        "leaf": args.leaf,
        "leaf_capacity": args.leaf_capacity,
        "theta": args.theta,
        "softening": args.softening,
        "kernel": args.kernel,
        "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "others_before": _card_processes(),
    }
    print(json.dumps(row), flush=True)

    pos, _, mass = _plummer(args.n)
    p, m = jnp.asarray(pos), jnp.asarray(mass)
    codes = np.sort(np.asarray(morton_encode(p, infer_bounds(p))).astype(np.uint64))
    leaves = adaptive_cell_leaf_partition_numpy(codes, leaf_size=args.leaf)[0].size
    del codes
    extra = {} if args.kernel is None else {"softening_kernel": args.kernel}
    fmm = FastMultipoleMethod(
        preset="galaxy" if args.galaxy else "large_n_gpu",
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
                leaf_capacity=(
                    "auto"
                    if args.leaf_capacity == "auto"
                    else 1 << int(np.ceil(np.log2(1.25 * leaves)))
                ),
            ),
            farfield=FarFieldConfig(retain_far_pairs_for_grad=True),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen_error" if args.galaxy else "dehnen",
        ),
        fixed_order=order,
        **extra,
    )
    t0 = time.perf_counter()
    state = fmm.prepare_state(
        p, m, leaf_size=args.leaf, max_order=order, theta=args.theta
    )
    plan = prepare_large_n_grad_plan(fmm, state)
    row["prepare_s"] = time.perf_counter() - t0
    stats = jax.devices()[0].memory_stats() or {}
    row["peak_after_prepare_GiB"] = stats.get("peak_bytes_in_use", 0) / 2**30
    row["in_use_after_prepare_GiB"] = stats.get("bytes_in_use", 0) / 2**30
    counts = np.asarray(state.neighbor_list.counts)
    row["lists"] = {
        "far_pairs": int(plan.far_pair_active_count),
        "far_capacity": int(plan.num_far_pairs),
        "far_target_sorted": bool(plan.far_pairs_target_sorted),
        "far_symmetric": bool(plan.far_pairs_symmetric),
        "near_edges": int(counts.sum()),
        "leaves": int(counts.shape[0]),
        "max_leaf": int(plan.max_leaf_size),
    }
    print(
        json.dumps(
            {
                k: row[k]
                for k in (
                    "prepare_s",
                    "lists",
                    "peak_after_prepare_GiB",
                    "in_use_after_prepare_GiB",
                )
            }
        ),
        flush=True,
    )
    w = jnp.asarray(np.random.default_rng(1).normal(size=(args.n, 3)), jnp.float32)
    params0 = ForceParams(
        G=jnp.asarray(1.0, jnp.float32),
        eps2=jnp.asarray(args.softening**2, jnp.float32),
    )

    def forward(st, pl, x, mm):
        return fmm.differentiable_accelerations(st, x, mm, grad_plan=pl)

    def loss(st, pl, ww, x, mm, fp=None):
        acc = fmm.differentiable_accelerations(st, x, mm, grad_plan=pl, force_params=fp)
        return jnp.sum(ww * acc)

    # every array an argument (closed over, jax.jit embeds it as a constant)
    argnums = (3, 4, 5) if args.params else (3, 4)
    gargs = (state, plan, w, p, m) + ((params0,) if args.params else ())

    if not args.no_forward:
        production = np.asarray(fmm.evaluate_prepared_state(state))
        diff = np.asarray(jax.jit(forward)(state, plan, p, m))
        row["forward_differing_elements"] = int(
            np.sum(production.view(np.uint32) != diff.view(np.uint32))
        )
        row["forward_max_abs"] = float(np.max(np.abs(production - diff)))
        print(
            f"differentiable forward vs production: "
            f"{row['forward_differing_elements']} of {production.size} elements "
            f"differ (max {row['forward_max_abs']:.3e})",
            flush=True,
        )
        del production, diff

    n = args.n

    def memory(compiled_fn) -> dict:
        ma = compiled_fn.memory_analysis()
        return {
            "argument_B_per_p": ma.argument_size_in_bytes / n,
            "output_B_per_p": ma.output_size_in_bytes / n,
            "temp_B_per_p": ma.temp_size_in_bytes / n,
            "temp_GiB": ma.temp_size_in_bytes / 2**30,
            "largest_arrays": _largest_arrays(compiled_fn.as_text(), n),
        }

    # the gradient alone: the near field's forward is dead code then (the loss is
    # linear in it), XLA drops it
    t0 = time.perf_counter()
    grad_only = jax.jit(jax.grad(loss, argnums=argnums)).lower(*gargs).compile()
    row["grad_only_compile_s"] = time.perf_counter() - t0
    row["grad_only_memory"] = memory(grad_only)
    print(json.dumps({"grad_only_memory": row["grad_only_memory"]}), flush=True)
    if args.grad_only:
        compiled = grad_only
    else:
        t0 = time.perf_counter()
        compiled = (
            jax.jit(jax.value_and_grad(loss, argnums=argnums)).lower(*gargs).compile()
        )
        row["grad_compile_s"] = time.perf_counter() - t0
    row["grad_memory"] = memory(compiled)
    stats = jax.devices()[0].memory_stats() or {}
    row["grad_memory"]["in_use_before_GiB"] = stats.get("bytes_in_use", 0) / 2**30
    print(json.dumps(row["grad_memory"]), flush=True)
    out1 = compiled(*gargs)
    jax.block_until_ready(out1)
    stats = jax.devices()[0].memory_stats() or {}
    row["grad_memory"]["device_peak_GiB"] = stats.get("peak_bytes_in_use", 0) / 2**30
    print(f"device peak {row['grad_memory']['device_peak_GiB']:.2f} GiB", flush=True)
    grads1 = [
        np.asarray(g)
        for g in jax.tree_util.tree_leaves(out1 if args.grad_only else out1[1])
    ]
    row["grad_finite"] = bool(all(np.all(np.isfinite(g)) for g in grads1))
    del out1
    out2 = compiled(*gargs)
    grads2 = [
        np.asarray(g)
        for g in jax.tree_util.tree_leaves(out2 if args.grad_only else out2[1])
    ]
    del out2
    row["grad_AvsA_differing"] = [
        int(
            np.sum(
                a.view(np.uint32 if a.dtype == np.float32 else np.uint64)
                != b.view(np.uint32 if b.dtype == np.float32 else np.uint64)
            )
        )
        for a, b in zip(grads1, grads2)
    ]
    print(
        f"gradient finite {row['grad_finite']}; A-vs-A differing "
        f"{row['grad_AvsA_differing']}",
        flush=True,
    )
    del grads1, grads2

    if not args.no_time:
        evaluate = fmm.evaluate_prepared_state
        row["force_eval_ms"] = _time(evaluate, state, reps=args.reps)
        row["grad_ms"] = _time(compiled, *gargs, reps=args.reps)
        row["force_eval_ms_b"] = _time(evaluate, state, reps=args.reps)
        row["grad_ms_b"] = _time(compiled, *gargs, reps=args.reps)
        row["grad_only_ms"] = _time(grad_only, *gargs, reps=args.reps)
        eval_min = min(row["force_eval_ms"][0], row["force_eval_ms_b"][0])
        row["ratio_min"] = min(row["grad_ms"][0], row["grad_ms_b"][0]) / eval_min
        row["ratio_grad_only_min"] = row["grad_only_ms"][0] / eval_min
        print(
            "force eval min/median {0[0]:.2f}/{0[1]:.2f} ms (again {1[0]:.2f}/{1[1]:.2f}); "
            "value_and_grad {2[0]:.2f}/{2[1]:.2f} ms (again {3[0]:.2f}/{3[1]:.2f}); "
            "grad {4[0]:.2f}/{4[1]:.2f} ms; ratios of the minima {5:.2f} / "
            "{6:.2f}".format(
                row["force_eval_ms"],
                row["force_eval_ms_b"],
                row["grad_ms"],
                row["grad_ms_b"],
                row["grad_only_ms"],
                row["ratio_min"],
                row["ratio_grad_only_min"],
            ),
            flush=True,
        )
    if args.trace:
        for label, fn, fargs in (
            ("grad", compiled, gargs),
            ("eval", fmm.evaluate_prepared_state, (state,)),
        ):
            jax.block_until_ready(fn(*fargs))
            with jax.profiler.trace(f"{args.trace}_{label}"):
                for _ in range(3):
                    jax.block_until_ready(fn(*fargs))
    row["others_after"] = _card_processes()
    print("RESULT " + json.dumps(row), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(row, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
