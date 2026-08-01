"""Figure 05 -- M2L and P2P interaction counts vs N, with fitted exponents.

Sweeps N and reads the counts straight off the solver's public
``get_runtime_diagnostics()`` -- ``recent_dual_far_pair_count`` (accepted
node-node pairs, i.e. M2L) and ``recent_dual_neighbor_count`` (leaf-leaf
neighbour pairs, i.e. P2P). This is the hardware-independent half of the
complexity story: figure 04 measures seconds, which mix in memory bandwidth,
kernel launch overhead and whatever else shares the GPU, whereas a pair count is
a property of the tree and the acceptance criterion alone.

Counts and work are both recorded, because they scale differently:

* an **M2L pair count** times ``(p+1)^2`` coefficients approximates far-field
  work, so at fixed p the count alone is the right x-axis;
* a **P2P pair count** hides the leaf occupancy -- two runs with the same number
  of neighbour pairs do different amounts of arithmetic if their leaves are
  differently full. ``near_work`` (the summed target x source particle products)
  is therefore recorded next to it.

The fitted exponent is what the figure quotes. It is fitted above ``--fit-min-n``
because the small-N end is dominated by the root levels of the tree, where the
asymptotic scaling has not started yet, and including it drags the exponent
around.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu python -m bench.scaling.interaction_counts \\
        --n-min-exp 10 --n-max-exp 12 --n-steps 3 --gpu-select none \\
        --json-out /tmp/smoke.json

Paper run::

    python -m bench.scaling.interaction_counts --n-min-exp 12 --n-max-exp 21 --n-steps 10
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "scaling/interaction_counts.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n-min-exp", type=int, default=12)
    p.add_argument("--n-max-exp", type=int, default=21)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--theta", type=float, default=0.6)
    p.add_argument("--basis", default="real")
    p.add_argument("--preset", default="accurate")
    p.add_argument("--leaf-size", type=int, default=32)
    p.add_argument(
        "--distribution",
        default="uniform_cube",
        help="Timing-ladder distribution from bench.scaling._timing.distribution",
    )
    p.add_argument("--fit-min-n", type=int, default=1 << 14)
    p.add_argument("--softening", type=float, default=1e-3)
    runmeta.add_common_args(p)
    p.set_defaults(dtype="float32")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    runmeta.select_gpu(args.gpu_select)
    runmeta.enable_x64(args.dtype)

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402

    from bench.scaling import _timing as T  # noqa: E402
    from jaccpot import FastMultipoleMethod  # noqa: E402

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    ns = T.n_values(args.n_min_exp, args.n_max_exp, args.n_steps)

    records: list[dict[str, Any]] = []
    for n in ns:
        key = jax.random.PRNGKey(runmeta.seed_sequence(args.seed, args.distribution, n))
        points, charges = T.distribution(key, n, args.distribution, dtype)

        solver = FastMultipoleMethod(
            preset=args.preset,
            basis=args.basis,
            theta=float(args.theta),
            softening=float(args.softening),
        )
        try:
            state = solver.prepare_state(
                points,
                charges,
                leaf_size=int(args.leaf_size),
                max_order=int(args.order),
                theta=float(args.theta),
            )
            accel = solver.evaluate_prepared_state(state)
            jax.block_until_ready(accel)
        except Exception as exc:
            print(f"N={n:<9d} FAILED: {str(exc)[:140]}")
            records.append({"n": int(n), "error": str(exc)[:300]})
            continue

        diag = solver.get_runtime_diagnostics()
        far_pairs = int(diag.get("recent_dual_far_pair_count", 0))
        near_pairs = int(diag.get("recent_dual_neighbor_count", 0))
        coeffs = (int(args.order) + 1) ** 2

        # Leaf occupancy, so `near_work` can be reported rather than inferred.
        near_work = None
        try:
            node_ranges = np.asarray(state.tree.node_ranges)
            nb = state.neighbor_list
            leaves = np.asarray(nb.leaf_indices)
            counts = np.asarray(nb.counts)
            offsets = np.asarray(nb.offsets)
            neighbors = np.asarray(nb.neighbors)
            sizes = (node_ranges[:, 1] - node_ranges[:, 0] + 1).clip(min=0)
            total = 0
            for slot, leaf in enumerate(leaves.tolist()):
                n_t = int(sizes[leaf])
                start = int(offsets[slot])
                for k in range(int(counts[slot])):
                    src = int(neighbors[start + k])
                    if src < 0:
                        continue
                    total += n_t * int(sizes[src])
                total += n_t * n_t  # the leaf's own block
            near_work = int(total)
        except Exception as exc:  # pragma: no cover - layout dependent
            print(f"  [warn] near_work unavailable at N={n}: {str(exc)[:100]}")

        record = {
            "n": int(n),
            "far_pairs": far_pairs,
            "near_pairs": near_pairs,
            "far_work": far_pairs * coeffs,
            "near_work": near_work,
            "coeffs_per_pair": coeffs,
            "nodes": int(diag.get("recent_dual_node_count", 0)),
            "leaves": int(diag.get("recent_dual_leaf_count", 0)),
            "far_pairs_per_particle": far_pairs / n if n else None,
            "near_pairs_per_particle": near_pairs / n if n else None,
        }
        records.append(record)
        print(
            f"N={n:<9d} far={far_pairs:<10d} near={near_pairs:<10d} "
            f"nodes={record['nodes']:<8d} leaves={record['leaves']:<7d} "
            f"far/N={record['far_pairs_per_particle']:.2f} "
            f"near/N={record['near_pairs_per_particle']:.2f}",
            flush=True,
        )

    fits: dict[str, dict[str, Any]] = {}
    for field in ("far_pairs", "near_pairs", "far_work", "near_work"):
        xs = [r["n"] for r in records if r.get(field)]
        ys = [float(r[field]) for r in records if r.get(field)]
        if len(xs) >= 2:
            fits[field] = T.fit_log_log_exponent(xs, ys, min_n=int(args.fit_min_n))

    print("\n=== fitted exponents (log count = alpha log N + c) ===")
    for name, fit in fits.items():
        print(
            f"  {name:<12s} alpha={fit['exponent']:.3f} R^2={fit['r_squared']:.4f} "
            f"over {fit['n_points']} pts N={fit.get('fit_min_n')}..{fit.get('fit_max_n')}"
        )

    config = {
        "n": ns,
        "theta": float(args.theta),
        "order": int(args.order),
        "basis": args.basis,
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": args.dtype,
        "leaf_size": int(args.leaf_size),
        "preset": args.preset,
        "distribution": args.distribution,
        "softening": float(args.softening),
        "fit_min_n": int(args.fit_min_n),
        "counts_from": (
            "get_runtime_diagnostics(): recent_dual_far_pair_count (M2L) and "
            "recent_dual_neighbor_count (P2P)"
        ),
    }
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config=config,
        meta=runmeta.run_meta({"argv": sys.argv[1:]}),
        data={"records": records, "fits": fits},
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
