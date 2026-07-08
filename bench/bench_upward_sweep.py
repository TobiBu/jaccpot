"""Microbenchmark for the solidfmm complex upward pass (geometry/mass/P2M/M2M).

Isolates the per-step upward-sweep cost that dominates the strict fused lane
(``runtime_refresh_tree_upward_seconds``) without needing the full galaxy lane or
a fitted strict cap profile. Reports the per-stage breakdown so we can see the
share of each stage (especially M2M) and decide where to optimize.

Free-GPU selection uses ``autocvd`` and MUST run before ``import jax`` (mirrors
``bench/bench_fused_eval_vs_jaxfmm.py``). Example:

    JAX_ENABLE_X64=1 micromamba run -n odisseo \
        python bench/bench_upward_sweep.py --n-particles 200000 --leaf-size 256 \
        --max-order 4 --runs 5 --warmup 2 --gpu-select free
"""

from __future__ import annotations

import argparse
import json
import os
import time


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-particles", type=int, default=200_000)
    p.add_argument("--leaf-size", type=int, default=256)
    p.add_argument("--max-order", type=int, default=4)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--center-mode", choices=("com", "aabb"), default="com")
    p.add_argument(
        "--gpu-select",
        choices=("free", "least-used", "none"),
        default="free",
        help="Pick a free GPU via autocvd before importing jax.",
    )
    p.add_argument("--gpu-wait-timeout", type=float, default=120.0)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def _select_gpu(args: argparse.Namespace) -> None:
    if args.gpu_select == "none" or "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    try:
        from autocvd import autocvd
    except Exception as exc:  # pragma: no cover - env-dependent
        print(f"[bench_upward] autocvd unavailable ({exc}); using default device")
        return
    autocvd(num_gpus=1, least_used=(args.gpu_select == "least-used"))


def main() -> None:
    args = _parse_args()
    _select_gpu(args)

    # Detailed per-stage timing inside the sweep is gated by this env var.
    os.environ.setdefault("JACCPOT_PROFILE_UPWARD_STAGES", "1")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from yggdrax.tree import build_tree, get_level_offsets, get_num_levels

    from jaccpot.upward.solidfmm_complex_tree_expansions import (
        prepare_solidfmm_complex_upward_sweep,
    )

    backend = jax.default_backend()
    dev = jax.devices()[0]
    cc = getattr(dev, "compute_capability", "n/a")

    # Concentrated blob (roughly galaxy-like): normal cloud in a unit box.
    rng = np.random.default_rng(args.seed)
    n = int(args.n_particles)
    pos_np = rng.normal(0.0, 0.3, size=(n, 3)).astype(np.float64)
    pos_np = np.clip(pos_np, -0.99, 0.99)
    mass_np = rng.uniform(0.5, 1.5, size=(n,)).astype(np.float64)
    positions = jnp.asarray(pos_np)
    masses = jnp.asarray(mass_np)
    lo = jnp.asarray([-1.0, -1.0, -1.0], dtype=jnp.float64)
    hi = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64)

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        (lo, hi),
        return_reordered=True,
        leaf_size=int(args.leaf_size),
    )

    # Topology stats (concrete here; illustrate the M2M right-sizing headroom).
    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    total_nodes = int(jnp.asarray(tree.parent).shape[0])
    num_leaves = total_nodes - num_internal
    level_offsets = np.asarray(jax.device_get(get_level_offsets(tree)))
    widths = level_offsets[1:] - level_offsets[:-1]
    padded_num_levels = int(level_offsets.shape[0] - 1)
    # Actual (unpadded) depth: get_num_levels on a concrete tree returns
    # int(max(node_levels))+1, whereas level_offsets is padded to Morton depth.
    actual_num_levels = int(get_num_levels(tree))
    # Levels the M2M loop visits: internal levels 0 .. actual_num_levels-2.
    internal_widths = widths[: max(actual_num_levels - 1, 0)]
    max_internal_level_width = int(internal_widths.max()) if internal_widths.size else 1

    def _run_once(collect, overrides):
        cb = None
        if collect is not None:

            def cb(name: str, seconds: float) -> None:
                collect[name] = collect.get(name, 0.0) + float(seconds)

        out = prepare_solidfmm_complex_upward_sweep(
            tree,
            pos_sorted,
            mass_sorted,
            max_order=int(args.max_order),
            center_mode=args.center_mode,
            max_leaf_size=int(args.leaf_size),
            upward_timing_callback=cb,
            **overrides,
        )
        jax.block_until_ready(out.multipoles.packed)
        return out

    runs = int(args.runs)

    def _measure(overrides):
        for _ in range(max(1, int(args.warmup))):
            _run_once(None, overrides)
        totals = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _run_once(None, overrides)
            totals.append(time.perf_counter() - t0)
        stage_acc: dict[str, float] = {}
        for _ in range(runs):
            _run_once(stage_acc, overrides)
        return {
            "total_seconds_mean": float(np.mean(totals)),
            "total_seconds_min": float(np.min(totals)),
            "stage_seconds_mean": {k: v / runs for k, v in stage_acc.items()},
        }

    baseline = _measure({})
    optimized = _measure({"static_num_levels": actual_num_levels})

    # Correctness: optimized multipoles must match the padded-shape baseline.
    base_out = _run_once(None, {})
    opt_out = _run_once(
        None,
        {"static_num_levels": actual_num_levels},
    )
    max_abs_diff = float(
        jnp.max(jnp.abs(base_out.multipoles.packed - opt_out.multipoles.packed))
    )
    parity_ok = bool(max_abs_diff <= 1e-9)

    def _speedups(b, o):
        out = {
            "total": (
                b["total_seconds_mean"] / o["total_seconds_mean"]
                if o["total_seconds_mean"]
                else 1.0
            )
        }
        for k in b["stage_seconds_mean"]:
            bo, oo = b["stage_seconds_mean"][k], o["stage_seconds_mean"].get(k, 0.0)
            out[k] = (bo / oo) if oo else float("inf")
        return out

    result = {
        "backend": backend,
        "compute_capability": str(cc),
        "n_particles": n,
        "leaf_size": int(args.leaf_size),
        "max_order": int(args.max_order),
        "center_mode": args.center_mode,
        "runs": runs,
        "baseline": baseline,
        "optimized": optimized,
        "speedup": _speedups(baseline, optimized),
        "parity_max_abs_diff": max_abs_diff,
        "parity_ok": parity_ok,
        "topology": {
            "num_internal": num_internal,
            "num_leaves": num_leaves,
            "total_nodes": total_nodes,
            "padded_num_levels": padded_num_levels,
            "actual_num_levels": actual_num_levels,
            "max_internal_level_width": max_internal_level_width,
            "current_m2m_batch_width": num_internal,
            "level_widths": [int(w) for w in widths.tolist() if w],
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[bench_upward] wrote {args.output}")


if __name__ == "__main__":
    main()
