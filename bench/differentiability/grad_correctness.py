"""Figure 13 data -- finite-difference vs autodiff gradient error vs theta.

``differentiable_accelerations`` gives exact gradients **at fixed topology**: the
tree carried by ``state`` is held constant while the numeric pipeline is
re-evaluated on the live positions and masses. So the finite-difference check has
to perturb *that same function* -- the frozen-topology one -- or it is not testing
autodiff, it is testing whether the tree happened to re-sort. Both arms here call
``differentiable_accelerations`` on one fixed ``state``.

That leaves a real question the frozen-topology check cannot answer: does the
frozen-topology gradient describe the *physical* force, the one you get when the
tree is rebuilt? A separate ``full_pipeline`` arm answers it by finite-differencing
``compute_accelerations`` end to end at a deliberately small step, where the tree
is rebuilt at every perturbed point. It is reported alongside, not merged: a
disagreement there is a statement about topology sensitivity, not about autodiff.

Both differentiation targets are swept, because they exercise different seams --
positions flow through the geometry (P2M centres, M2L translation vectors, the
near-field kernel), masses only through the monopole weights.

Reading the theta axis
----------------------
The MAC decision is piecewise constant in the particle positions, so the exact
force is only piecewise smooth: a perturbation that moves a pair across the
acceptance boundary changes which interactions exist. Autodiff never sees that --
it differentiates the frozen decision -- whereas a finite difference straddling a
boundary does. Points where the two disagree are therefore expected near boundary
crossings, and `far_pairs` is recorded at every theta so a disagreement can be
attributed rather than guessed at.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m bench.differentiability.grad_correctness \\
        --n 128 --thetas 0.5 --fd-samples 4 --gpu-select none --json-out /tmp/smoke.json

Paper run::

    python -m bench.differentiability.grad_correctness \\
        --n 512 --thetas 0.3,0.4,0.5,0.6,0.7,0.8 --fd-samples 24
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

DEFAULT_OUT = "differentiability/grad_correctness.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--thetas", default="0.3,0.4,0.5,0.6,0.7,0.8")
    p.add_argument("--orders", default="4")
    p.add_argument("--basis", default="real,solidfmm")
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--preset", default="accurate")
    p.add_argument("--distribution", default="plummer")
    p.add_argument(
        "--wrt",
        default="positions,masses",
        help="Which inputs to differentiate with respect to",
    )
    p.add_argument(
        "--fd-samples",
        type=int,
        default=24,
        help=(
            "Number of randomly chosen coordinates to finite-difference. Each "
            "costs two extra forward evaluations, so the full array is not "
            "affordable; a fixed-seed subsample is (default: 24)"
        ),
    )
    p.add_argument(
        "--fd-eps",
        type=float,
        default=1e-6,
        help="Central-difference step for the frozen-topology check",
    )
    p.add_argument(
        "--fd-eps-full",
        type=float,
        default=1e-8,
        help=(
            "Step for the full-pipeline check. Smaller, because a larger step is "
            "more likely to rebuild the tree differently and so measure topology "
            "sensitivity rather than gradient error"
        ),
    )
    p.add_argument("--softening", type=float, default=1e-3)
    p.add_argument("--G", dest="g", type=float, default=1.0)
    runmeta.add_common_args(p)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    runmeta.select_gpu(args.gpu_select)
    # Finite differences at eps=1e-6 need float64: in float32 the subtraction
    # would be dominated by round-off, and the check would "fail" on precision.
    args.dtype = "float64"
    runmeta.enable_x64(args.dtype)

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402

    from bench.validation import _harness as H  # noqa: E402
    from jaccpot import FastMultipoleMethod  # noqa: E402

    thetas = [float(v) for v in str(args.thetas).split(",") if v.strip()]
    orders = [int(v) for v in str(args.orders).split(",") if v.strip()]
    bases = [v.strip() for v in str(args.basis).split(",") if v.strip()]
    wrts = [v.strip() for v in str(args.wrt).split(",") if v.strip()]

    seed = runmeta.seed_sequence(args.seed, args.distribution, args.n)
    pos_np, mass_np = H.make_distribution(args.distribution, int(args.n), seed)
    positions = jnp.asarray(pos_np, dtype=jnp.float64)
    masses = jnp.asarray(mass_np, dtype=jnp.float64)

    # A fixed random probe turns the (N, 3) force field into a scalar loss. Using
    # sum(a**2) instead would make the loss insensitive to sign errors that cancel.
    probe = jnp.asarray(
        np.random.default_rng(seed ^ 0x5EED).normal(size=(int(args.n), 3)),
        dtype=jnp.float64,
    )

    records: list[dict[str, Any]] = []
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
                state = fmm.prepare_state(
                    positions,
                    masses,
                    leaf_size=int(args.leaf_size),
                    max_order=int(order),
                    theta=float(theta),
                )
                diag = fmm.get_runtime_diagnostics()
                far_pairs = int(diag.get("recent_dual_far_pair_count", 0))

                def frozen_loss(pos, mass):
                    return jnp.sum(
                        probe * fmm.differentiable_accelerations(state, pos, mass)
                    )

                def full_loss(pos, mass):
                    return jnp.sum(
                        probe
                        * fmm.compute_accelerations(
                            pos,
                            mass,
                            leaf_size=int(args.leaf_size),
                            max_order=int(order),
                        )
                    )

                for wrt in wrts:
                    argnum = 0 if wrt == "positions" else 1
                    target = positions if argnum == 0 else masses
                    target_np = np.asarray(target, dtype=np.float64)

                    try:
                        ad = np.asarray(
                            jax.grad(frozen_loss, argnums=argnum)(positions, masses),
                            dtype=np.float64,
                        )
                    except Exception as exc:
                        print(
                            f"{basis:>9s} p={order} theta={theta:<4.2f} {wrt:>9s} "
                            f"AD FAILED: {str(exc)[:90]}"
                        )
                        records.append(
                            {
                                "basis": basis,
                                "order": order,
                                "theta": theta,
                                "wrt": wrt,
                                "error": str(exc)[:300],
                            }
                        )
                        continue

                    rng = np.random.default_rng(seed ^ (argnum + 1))
                    idx = rng.choice(
                        target_np.size,
                        size=min(int(args.fd_samples), target_np.size),
                        replace=False,
                    )

                    def fd_at(indices, eps: float, loss) -> np.ndarray:
                        out = np.zeros(len(indices), dtype=np.float64)
                        for k, flat_i in enumerate(indices):
                            bumped = target_np.copy().ravel()
                            bumped[flat_i] += eps
                            plus = float(
                                loss(
                                    jnp.asarray(
                                        bumped.reshape(target_np.shape),
                                        dtype=jnp.float64,
                                    ),
                                    masses,
                                )
                                if argnum == 0
                                else loss(
                                    positions,
                                    jnp.asarray(
                                        bumped.reshape(target_np.shape),
                                        dtype=jnp.float64,
                                    ),
                                )
                            )
                            bumped[flat_i] -= 2.0 * eps
                            minus = float(
                                loss(
                                    jnp.asarray(
                                        bumped.reshape(target_np.shape),
                                        dtype=jnp.float64,
                                    ),
                                    masses,
                                )
                                if argnum == 0
                                else loss(
                                    positions,
                                    jnp.asarray(
                                        bumped.reshape(target_np.shape),
                                        dtype=jnp.float64,
                                    ),
                                )
                            )
                            out[k] = (plus - minus) / (2.0 * eps)
                        return out

                    ad_sub = ad.ravel()[idx]
                    fd_sub = fd_at(idx, float(args.fd_eps), frozen_loss)
                    denom = float(np.linalg.norm(fd_sub)) or 1.0
                    rel_l2 = float(np.linalg.norm(ad_sub - fd_sub) / denom)
                    worst = float(
                        np.max(
                            np.abs(ad_sub - fd_sub)
                            / np.maximum(np.abs(fd_sub), denom / len(fd_sub))
                        )
                    )

                    record = {
                        "basis": basis,
                        "order": int(order),
                        "theta": float(theta),
                        "wrt": wrt,
                        "seed": int(seed),
                        "far_pairs": far_pairs,
                        "fd_samples": int(len(idx)),
                        "fd_eps": float(args.fd_eps),
                        "frozen_rel_l2": rel_l2,
                        "frozen_worst_rel": worst,
                        "ad_norm": float(np.linalg.norm(ad_sub)),
                        "fd_norm": float(np.linalg.norm(fd_sub)),
                    }

                    # One small-step full-pipeline point per (basis, order, theta,
                    # wrt): the physical check, on a handful of coordinates.
                    try:
                        few = idx[: min(4, len(idx))]
                        fd_full = fd_at(few, float(args.fd_eps_full), full_loss)
                        ad_few = ad.ravel()[few]
                        den_full = float(np.linalg.norm(fd_full)) or 1.0
                        record["full_pipeline_rel_l2"] = float(
                            np.linalg.norm(ad_few - fd_full) / den_full
                        )
                        record["full_pipeline_samples"] = int(len(few))
                        record["fd_eps_full"] = float(args.fd_eps_full)
                    except Exception as exc:
                        record["full_pipeline_rel_l2"] = None
                        record["full_pipeline_error"] = str(exc)[:200]

                    records.append(record)
                    print(
                        f"{basis:>9s} p={order} theta={theta:<4.2f} {wrt:>9s} "
                        f"frozen_rel_l2={rel_l2:.3e} worst={worst:.3e} "
                        f"full={record.get('full_pipeline_rel_l2')} far={far_pairs}",
                        flush=True,
                    )

    config = {
        "n": int(args.n),
        "theta": thetas,
        "order": orders,
        "basis": bases,
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": args.dtype,
        "leaf_size": int(args.leaf_size),
        "preset": args.preset,
        "distribution": args.distribution,
        "wrt": wrts,
        "fd_samples": int(args.fd_samples),
        "fd_eps": float(args.fd_eps),
        "fd_eps_full": float(args.fd_eps_full),
        "softening": float(args.softening),
        "G": float(args.g),
        "loss": "sum(probe * a), fixed-seed standard-normal probe",
        "note": (
            "frozen_* compares autodiff against a finite difference of the SAME "
            "fixed-topology function (differentiable_accelerations on one state). "
            "full_pipeline_* finite-differences compute_accelerations end to end, "
            "rebuilding the tree, and so also measures topology sensitivity."
        ),
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
