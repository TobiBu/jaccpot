"""Figure 01 -- force error vs expansion order, against an exact direct sum.

Sweeps ``max_order`` at fixed N and theta and records the resulting force error
for each expansion basis. The reference is the O(N^2) direct sum in
``_harness.chunked_direct_accelerations``, evaluated at the same G and softening
as the solver, so the comparison isolates truncation error rather than a kernel
mismatch.

Two error measures are recorded, because either alone is misleading:

* **rel-L2** over the whole field -- the headline convergence number, but an L2
  norm averages away a tail.
* **worst single-component error**, normalised by the rms ``|a|`` -- what an
  integrator actually trips over. A figure that shows only rel-L2 cannot
  distinguish "uniformly slightly worse" from "one particle badly wrong".

The full per-particle percentile summary goes into the JSON as well, so the
validation section can quote a tail without a rerun.

Accuracy is reported *as measured* at the configuration recorded in the JSON's
``config`` block. The cartesian basis is deliberately not swept: it sits at
~1.8e-1 rel-L2 independent of order (a divergent-series signature, tracked
separately in docs/dehnen_mass_mac_status_and_plan.md), so ``real`` and
``solidfmm`` are the quantitative bases.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m bench.validation.force_error_vs_order \\
        --n 1024 --orders 2,4 --gpu-select none --json-out /tmp/smoke.json

Paper run (picks a free GPU itself)::

    python -m bench.validation.force_error_vs_order \\
        --n 16384 --orders 1,2,3,4,6,8 --theta 0.5
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

DEFAULT_OUT = "validation/force_error_vs_order.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n", type=int, default=16384, help="Particle count (default: 16384)"
    )
    p.add_argument(
        "--orders",
        default="1,2,3,4,6,8",
        help="Comma-separated expansion orders to sweep (default: 1,2,3,4,6,8)",
    )
    p.add_argument(
        "--basis",
        default="real,solidfmm",
        help="Comma-separated bases (default: real,solidfmm)",
    )
    p.add_argument("--theta", type=float, default=0.5, help="Opening angle, held fixed")
    # See error_vs_theta.py: at N=16384 a 32-particle leaf can leave the clustered
    # far field empty, which floors the error at round-off and makes an order
    # sweep meaningless. 16 keeps the far field populated.
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument(
        "--preset",
        default="accurate",
        choices=("fast", "balanced", "accurate", "large_n_gpu"),
    )
    p.add_argument(
        "--distribution",
        default="uniform,plummer",
        help="Comma-separated distributions from _harness.make_distribution",
    )
    p.add_argument("--softening", type=float, default=1e-3)
    p.add_argument("--G", dest="g", type=float, default=1.0)
    runmeta.add_common_args(p)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    # autocvd must run before the first `import jax`, or the device choice is
    # silently ignored.
    runmeta.select_gpu(args.gpu_select)
    runmeta.enable_x64(args.dtype)

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402

    from bench.validation import _harness as H  # noqa: E402
    from jaccpot import FastMultipoleMethod  # noqa: E402

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    orders = [int(v) for v in str(args.orders).split(",") if v.strip()]
    bases = [v.strip() for v in str(args.basis).split(",") if v.strip()]
    dists = [v.strip() for v in str(args.distribution).split(",") if v.strip()]

    records: list[dict[str, Any]] = []
    for dist in dists:
        # One seed per (distribution, N) so rerunning a single case reproduces
        # exactly the sample the full sweep used.
        seed = runmeta.seed_sequence(args.seed, dist, args.n)
        pos_np, mass_np = H.make_distribution(dist, int(args.n), seed)
        positions = jnp.asarray(pos_np, dtype=dtype)
        masses = jnp.asarray(mass_np, dtype=dtype)

        reference = H.chunked_direct_accelerations(
            positions, masses, softening=args.softening, G=args.g
        )
        jax.block_until_ready(reference)

        for basis in bases:
            for order in orders:
                fmm = FastMultipoleMethod(
                    preset=args.preset,
                    basis=basis,
                    theta=float(args.theta),
                    softening=float(args.softening),
                    G=float(args.g),
                )
                accel = fmm.compute_accelerations(
                    positions,
                    masses,
                    leaf_size=int(args.leaf_size),
                    max_order=int(order),
                )
                jax.block_until_ready(accel)

                rel = H.per_particle_relative_error(accel, reference)
                scaled = H.per_particle_scaled_error(accel, reference)
                diag = fmm.get_runtime_diagnostics()
                record: dict[str, Any] = {
                    "distribution": dist,
                    "basis": basis,
                    "order": int(order),
                    "seed": int(seed),
                    "rel_l2": H.rel_l2(accel, reference),
                    "worst_component": H.worst_component_error(accel, reference),
                    # Interaction counts at this order: figure 05 plots the same
                    # quantity, so recording it here lets the text relate accuracy
                    # to cost without a second run.
                    "far_pairs": int(diag.get("recent_dual_far_pair_count", 0)),
                    "near_pairs": int(diag.get("recent_dual_neighbor_count", 0)),
                }
                record.update(H.error_summary(rel))
                record.update(H.error_summary(scaled, prefix="scaled_"))
                records.append(record)
                print(
                    f"{dist:>10s} {basis:>9s} p={order:<2d} "
                    f"rel_l2={record['rel_l2']:.3e} "
                    f"worst_comp={record['worst_component']:.3e} "
                    f"p99={record['p99']:.3e} far={record['far_pairs']}",
                    flush=True,
                )

    config = {
        "n": int(args.n),
        "theta": float(args.theta),
        # `order` is the swept axis; each record carries the value it was measured at.
        "order": orders,
        "basis": bases,
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": args.dtype,
        "leaf_size": int(args.leaf_size),
        "preset": args.preset,
        "distribution": dists,
        "softening": float(args.softening),
        "G": float(args.g),
        "reference": "direct O(N^2) sum at matched G and softening",
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
