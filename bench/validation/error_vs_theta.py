"""Figure 02 -- force error vs opening angle theta (Dehnen-2014-style curve).

Sweeps theta at fixed N with two or three expansion orders overlaid, against the
same direct-sum reference and the same error measures as figure 01 (both come
from ``_harness``, so the two figures cannot disagree about what "force error"
means).

The order overlay is the point of the figure: theta and p trade off against each
other, and a single-order curve cannot show that a tighter opening angle buys
less than a higher order past a certain accuracy. Orders are drawn from the
sequential ramp in the figure notebook, since order is an *ordered* axis rather
than a set of unrelated categories.

One caution when reading the low-theta end: as theta shrinks the far field
empties out and the solver converges to an exact P2P direct sum, so the error
floors at round-off. That floor is a property of the configuration (N and leaf
size fix how many leaves there are to open), not of the expansion, so the JSON
records the far/near pair counts at every point and the notebook marks any point
whose far field is empty. Without that, a flat left-hand tail reads as
convergence when it is really "there was nothing to approximate".

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m bench.validation.error_vs_theta \\
        --n 1024 --thetas 0.4,0.7 --orders 4 --gpu-select none --json-out /tmp/smoke.json

Paper run::

    python -m bench.validation.error_vs_theta --n 16384 \\
        --thetas 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --orders 2,4,8
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

DEFAULT_OUT = "validation/error_vs_theta.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=16384)
    p.add_argument(
        "--thetas",
        default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated opening angles to sweep",
    )
    p.add_argument(
        "--orders", default="2,4,8", help="Comma-separated orders to overlay"
    )
    p.add_argument("--basis", default="real", help="Comma-separated bases")
    # leaf=16 rather than 32: measured at N=16384, a 32-particle leaf leaves the
    # Plummer far field completely empty below theta ~= 0.4 (far_pairs = 0), which
    # would put a round-off floor across the interesting end of the sweep. At
    # leaf=16 the far field stays populated from theta = 0.3 up.
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument(
        "--preset",
        default="accurate",
        choices=("fast", "balanced", "accurate", "large_n_gpu"),
    )
    p.add_argument("--distribution", default="uniform,plummer")
    p.add_argument("--softening", type=float, default=1e-3)
    p.add_argument("--G", dest="g", type=float, default=1.0)
    runmeta.add_common_args(p)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    runmeta.select_gpu(args.gpu_select)
    runmeta.enable_x64(args.dtype)

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402

    from bench.validation import _harness as H  # noqa: E402
    from jaccpot import FastMultipoleMethod  # noqa: E402

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    thetas = [float(v) for v in str(args.thetas).split(",") if v.strip()]
    orders = [int(v) for v in str(args.orders).split(",") if v.strip()]
    bases = [v.strip() for v in str(args.basis).split(",") if v.strip()]
    dists = [v.strip() for v in str(args.distribution).split(",") if v.strip()]

    records: list[dict[str, Any]] = []
    for dist in dists:
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
                for theta in thetas:
                    fmm = FastMultipoleMethod(
                        preset=args.preset,
                        basis=basis,
                        theta=float(theta),
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
                    far_pairs = int(diag.get("recent_dual_far_pair_count", 0))
                    record: dict[str, Any] = {
                        "distribution": dist,
                        "basis": basis,
                        "order": int(order),
                        "theta": float(theta),
                        "seed": int(seed),
                        "rel_l2": H.rel_l2(accel, reference),
                        "worst_component": H.worst_component_error(accel, reference),
                        "far_pairs": far_pairs,
                        "near_pairs": int(diag.get("recent_dual_neighbor_count", 0)),
                        # The notebook marks these points: with an empty far
                        # field the solver is an exact direct sum, so the error
                        # is round-off and says nothing about the expansion.
                        "far_field_empty": far_pairs == 0,
                    }
                    record.update(H.error_summary(rel))
                    record.update(H.error_summary(scaled, prefix="scaled_"))
                    records.append(record)
                    print(
                        f"{dist:>10s} {basis:>9s} p={order:<2d} theta={theta:<4.2f} "
                        f"rel_l2={record['rel_l2']:.3e} "
                        f"worst_comp={record['worst_component']:.3e} "
                        f"far={far_pairs}"
                        + ("  [far field empty]" if far_pairs == 0 else ""),
                        flush=True,
                    )

    config = {
        "n": int(args.n),
        # `theta` is the swept axis; each record carries the value it was measured at.
        "theta": thetas,
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
