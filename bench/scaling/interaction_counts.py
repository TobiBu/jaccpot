"""Figure 05 -- M2L and P2P interaction counts vs N on the fused lane, with fitted exponents.

Same configuration as figure 04 (``preset="galaxy"``, D3; ``bench/scaling/_galaxy.py``):
cell leaves of 64, p6, ``dehnen_error`` at ``adaptive_eps`` 1e-4, ferrers3, fp32.
The September version read ``get_runtime_diagnostics()`` after an eager
``prepare_state`` on ``preset="accurate"``; the fused lane builds its lists inside
the traced rebuild, so the counts are read there instead.

Where the counts come from
--------------------------
One jitted ``FusedForceProvider.rebuild(x, m, force_scale=f_b)`` -- the topology
half of the step figure 04 times, at the same positions and with the steady-state
f_b (the previous evaluation's; ``_timing.warm_fused_step``) -- and, from the
``FusedTopology`` it returns:

* ``far_pairs`` -- the flat walk's ``far_needed`` (``topology.needs()[0]``): the
  **directed** far pairs, i.e. M2L applications (2 x the walk's canonical count).
* ``near_pairs`` -- the CSR near list's length (``neighbor_list.offsets[-1]``):
  **directed** leaf-leaf neighbour pairs, self pairs excluded. It equals the walk's
  ``near_needed`` (checked per row: ``near_pairs_walk``).
* ``near_work`` -- the P2P particle products the near field evaluates:
  ``sum_leaf n_t (n_t + sum_{s in CSR row} n_s)`` with leaf occupancies from the
  tree's ``node_ranges``, computed on the device (no list leaves the card).
* ``far_work`` -- ``far_pairs x (p+1)^2``: a proxy for M2L arithmetic at fixed p,
  not a FLOP count.
* ``leaves`` -- live cell leaves (``tree_needs``' ``leaf_needed``); ``nodes`` --
  nodes holding at least one particle.

The counts are exact only when the walk did not overflow; a row whose capacity
verdict fails (``overflow_ok`` False: the force at that step is wrong and the
walk's counts are lower bounds) is recorded with ``valid: false`` and an ``error``
so the notebook does not plot it, and is not fitted.

Why counts as well as seconds: figure 04's seconds mix in memory bandwidth, kernel
launches and whatever else shares the GPU; a pair count is a property of the tree
and the acceptance criterion alone. Under ``dehnen_error`` it also depends on the
particles' force scale f_b, which is why the steady-state f_b is used.

The fitted exponent is fitted above ``--fit-min-n``: the small-N end is the
degenerate galaxy tree (``cell_min_level=8``: at N = 8192 nearly every leaf holds
one particle and nearly every pair is far; see figure 04's docstring).

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu python -m bench.scaling.interaction_counts --n 2048,4096 \\
        --fit-min-n 0 --gpu-select none --json-out /tmp/smoke.json

Paper run (one A100; the same ladder as figure 04)::

    python -m bench.scaling.interaction_counts
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
DEFAULT_N = "1e4,3e4,1e5,3e5,1e6,3e6,1e7,3e7,1e8,1.9e8"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n",
        default=None,
        help=f"Comma-separated N, 1e6 notation allowed (default: {DEFAULT_N})",
    )
    p.add_argument(
        "--n-min-exp",
        type=int,
        default=None,
        help="Instead of --n: a log-even ladder from 2**n-min-exp ...",
    )
    p.add_argument(
        "--n-max-exp", type=int, default=None, help="... to 2**n-max-exp ..."
    )
    p.add_argument("--n-steps", type=int, default=None, help="... in this many steps")
    p.add_argument(
        "--distribution",
        default="plummer_clipped",
        help="bench.scaling._timing.distribution (default: the production rows' IC)",
    )
    p.add_argument("--rmax", type=float, default=20.0)
    p.add_argument("--softening", type=float, default=1e-7)
    p.add_argument("--G", dest="g", type=float, default=1.0)
    p.add_argument("--fit-min-n", type=int, default=1_000_000)
    runmeta.add_common_args(p)
    p.set_defaults(dtype="float32")
    return p.parse_args()


def _ladder(args: argparse.Namespace) -> list[int]:
    from bench.scaling import _timing as T

    if args.n is not None:
        return [int(float(v)) for v in str(args.n).split(",") if v.strip()]
    if args.n_min_exp is not None or args.n_max_exp is not None:
        lo = args.n_min_exp if args.n_min_exp is not None else args.n_max_exp
        hi = args.n_max_exp if args.n_max_exp is not None else lo
        return T.n_values(lo, hi, args.n_steps or (hi - lo + 1))
    return [int(float(v)) for v in DEFAULT_N.split(",")]


def _counts_fn(provider: Any) -> Any:
    """A jitted ``counts(x, m, f_b)`` over one rebuild: scalars only leave the card."""
    import jax
    import jax.numpy as jnp

    from jaccpot.runtime.fused_topology import NEEDS_FIELDS

    idx = {name: i for i, name in enumerate(NEEDS_FIELDS)}

    @jax.jit
    def counts(x, m, f_b):
        topo = provider.rebuild(x, m, force_scale=f_b)
        needs = topo.needs()
        bits = topo.overflow_bits()
        prepared = topo.prepared
        ranges = jnp.asarray(prepared.tree.node_ranges)
        sizes = jnp.clip(ranges[:, 1] - ranges[:, 0] + 1, 0).astype(jnp.int64)
        nl = prepared.neighbor_list
        offsets = jnp.asarray(nl.offsets)
        neighbors = jnp.asarray(nl.neighbors)
        leaves = jnp.asarray(nl.leaf_indices)
        num_rows = leaves.shape[0]
        edge = jnp.arange(neighbors.shape[0])
        row = jnp.clip(
            jnp.searchsorted(offsets, edge, side="right") - 1, 0, num_rows - 1
        )
        live = edge < offsets[-1]
        n_t = sizes[leaves]
        n_s = sizes[jnp.clip(neighbors, 0, sizes.shape[0] - 1)]
        cross = jnp.sum(jnp.where(live, n_t[row] * n_s, 0))
        own = jnp.sum(n_t * n_t)
        return {
            "needs": needs,
            "bits": bits,
            "far_pairs": needs[idx["far_needed"]],
            "near_pairs_walk": needs[idx["near_needed"]],
            "near_pairs": offsets[-1],
            "near_work": cross + own,
            "leaves": needs[idx["leaf_needed"]],
            "leaves_occupied": jnp.sum(n_t > 0),
            "nodes": jnp.sum(sizes > 0),
            "peak_wavefront": needs[idx["peak_wavefront"]],
            "max_neighbors_per_leaf": jnp.max(offsets[1:] - offsets[:-1]),
        }

    return counts


def main() -> int:
    args = _parse_args()
    if args.dtype != "float32":
        print(
            "[fig05] preset='galaxy' is fp32 (D3); --dtype float64 is not this figure"
        )
        return 2
    runmeta.select_gpu(args.gpu_select)
    from bench.scaling import _timing as T  # noqa: E402  (imports no jax)

    env = T.launch_env()

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402

    from bench.scaling import _galaxy as GX  # noqa: E402
    from jaccpot.runtime.fused_topology import NEEDS_FIELDS, overflow_ok  # noqa: E402

    ns = _ladder(args)
    order = int(GX.GALAXY.max_order)
    coeffs = (order + 1) ** 2
    records: list[dict[str, Any]] = []
    galaxy: dict[str, Any] = {}

    for n in ns:
        key = jax.random.PRNGKey(runmeta.seed_sequence(args.seed, args.distribution, n))
        x, m = T.distribution(key, n, args.distribution, jnp.float32, rmax=args.rmax)
        try:
            solver = GX.galaxy_solver(softening=float(args.softening), G=float(args.g))
            galaxy = galaxy or GX.describe(solver)
            fs, setup = T.warm_fused_step(solver, x, m)
            raw = jax.device_get(_counts_fn(fs.provider)(x, m, fs.aux[0]))
        except Exception as exc:
            print(f"N={n:<10d} FAILED: {str(exc)[:200]}")
            records.append({"n": int(n), "valid": False, "error": str(exc)[:400]})
            continue

        ok = bool(overflow_ok(raw["bits"])) and bool(setup["first_step_ok"])
        needs = {k: int(v) for k, v in zip(NEEDS_FIELDS, np.asarray(raw["needs"]))}
        far_pairs = int(raw["far_pairs"])
        near_pairs = int(raw["near_pairs"])
        record = {
            "n": int(n),
            "valid": ok,
            "far_pairs": far_pairs,
            "near_pairs": near_pairs,
            "near_pairs_walk": int(raw["near_pairs_walk"]),
            "far_work": far_pairs * coeffs,
            "near_work": int(raw["near_work"]),
            "coeffs_per_pair": coeffs,
            "nodes": int(raw["nodes"]),
            "leaves": int(raw["leaves"]),
            "leaves_occupied": int(raw["leaves_occupied"]),
            "peak_wavefront": int(raw["peak_wavefront"]),
            "max_neighbors_per_leaf": int(raw["max_neighbors_per_leaf"]),
            "far_pairs_per_particle": far_pairs / n,
            "near_pairs_per_particle": near_pairs / n,
            "near_work_per_particle": int(raw["near_work"]) / n,
            "mean_leaf_occupancy": n / max(int(raw["leaves"]), 1),
            "overflow_bits": int(raw["bits"]),
            "needs": needs,
            "setup": setup,
        }
        if not ok:
            # the notebook drops rows carrying "error": a failed verdict is not data
            record["error"] = (
                f"capacity verdict failed (bits {int(raw['bits']):#x}): the walk's "
                "counts are lower bounds and the force is wrong"
            )
        records.append(record)
        del fs
        print(
            f"N={n:<10d} valid={ok} far={far_pairs:<11d} near={near_pairs:<10d} "
            f"near_work={record['near_work']:<13d} leaves={record['leaves']:<9d} "
            f"far/N={record['far_pairs_per_particle']:.2f} "
            f"near/N={record['near_pairs_per_particle']:.3f} "
            f"occupancy={record['mean_leaf_occupancy']:.2f}",
            flush=True,
        )

    fits: dict[str, dict[str, Any]] = {}
    valid = [r for r in records if r.get("valid")]
    for field in ("far_pairs", "near_pairs", "far_work", "near_work"):
        pts = [(r["n"], float(r[field])) for r in valid if r.get(field)]
        if len(pts) >= 2:
            fits[field] = T.fit_log_log_exponent(
                [a for a, _ in pts], [b for _, b in pts], min_n=int(args.fit_min_n)
            )

    print("\n=== fitted exponents (log count = alpha log N + c) ===")
    for name, fit in fits.items():
        print(
            f"  {name:<12s} alpha={fit['exponent']:.3f} R^2={fit['r_squared']:.4f} "
            f"over {fit['n_points']} pts N={fit.get('fit_min_n')}..{fit.get('fit_max_n')}"
        )

    config = {
        "n": ns,
        "theta": float(GX.GALAXY.theta),
        "order": order,
        "basis": "real",
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": "float32",
        "leaf_size": int(GX.GALAXY.leaf_size),
        "preset": "galaxy",
        "galaxy": galaxy,
        "distribution": args.distribution,
        "rmax": float(args.rmax) if args.distribution == "plummer_clipped" else None,
        "softening": float(args.softening),
        "G": float(args.g),
        "fit_min_n": int(args.fit_min_n),
        "launch_env": env,
        "counts_from": (
            "one jitted FusedForceProvider.rebuild at the step's positions with the "
            "steady-state f_b: far_pairs = topology.needs() far_needed (directed, "
            "= M2L applications); near_pairs = CSR neighbor_list.offsets[-1] "
            "(directed leaf pairs, self excluded); near_work = sum_leaf n_t (n_t + "
            "sum_CSR n_s); far_work = far_pairs (p+1)^2 (proxy)"
        ),
        "valid_rule": "rows with valid=false failed the capacity verdict; they carry "
        "'error' and are neither fitted nor plotted",
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
