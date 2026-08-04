"""CI guard for acceleration/time-derivative runtime path sanity.

This guard intentionally uses broad ratio bounds to avoid hardware-specific
flakiness while still catching major runtime regressions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from bench.bench_parallel_paths import _sample_problem, collect_metrics
from jaccpot import FastMultipoleMethod


def _validate_metrics(metrics: dict[str, float]) -> None:
    acc = float(metrics["acc_mean_seconds"])
    jerk_fast = float(metrics["jerk_fast_mean_seconds"])
    jerk_acc = float(metrics["jerk_accurate_mean_seconds"])
    ratio_fast = float(metrics["jerk_fast_over_acc"])
    ratio_acc = float(metrics["jerk_accurate_over_fast"])
    td2 = float(metrics["time_deriv2_accurate_mean_seconds"])
    td3 = float(metrics["time_deriv3_accurate_mean_seconds"])
    ratio_td2 = float(metrics["time_deriv2_accurate_over_jerk_accurate"])
    ratio_td3 = float(metrics["time_deriv3_accurate_over_time_deriv2_accurate"])

    if not (
        acc > 0.0 and jerk_fast > 0.0 and jerk_acc > 0.0 and td2 > 0.0 and td3 > 0.0
    ):
        raise RuntimeError("non-positive benchmark timings observed")
    if not (0.4 <= ratio_fast <= 8.0):
        raise RuntimeError(
            f"jerk_fast_over_acc out of guard range: {ratio_fast:.3f} "
            "(expected 0.4..8.0)"
        )
    # Lower bound relaxed 1.2 -> 0.9, and the protection it was standing in for
    # moved to _validate_accurate_jerk_differs_from_fast below.
    #
    # This bound was a proxy: "accurate must cost visibly more than fast_approx,
    # therefore it must actually be doing the extra source-motion work". The proxy
    # broke when compute_tree_geometry got a compiled dispatch. The accurate jerk
    # path makes two EXTRA geometry calls for the source-motion term, and those
    # were being dispatched op-by-op; compiling them cut the accurate mode from
    # 1.045 s to 0.611 s at N=512 on CPU (measured, runs=3), taking the ratio from
    # 1.87 to 1.01. The two modes now cost nearly the same, so no timing bound can
    # separate them -- lowering this one to 1.0 would have deleted the guard rather
    # than relaxed it.
    #
    # What is left here is a sanity floor: accurate being dramatically CHEAPER
    # than fast_approx would still indicate something mis-wired.
    if not (0.9 <= ratio_acc <= 15.0):
        raise RuntimeError(
            f"jerk_accurate_over_fast out of guard range: {ratio_acc:.3f} "
            "(expected 0.9..15.0)"
        )
    # Lower bounds 1.0 -> 0.8 on both derivative ratios. The 1.0 floor asserted
    # that each extra time derivative costs at least as much as the previous one,
    # which is true in expectation but sits *inside* the run-to-run spread of a
    # 3-sample timing on a shared runner -- so it was a knife-edge, not a guard.
    #
    # Measured over 5 consecutive local CPU runs at the CI configuration
    # (n=512, runs=3, warmup=1, float32): td3/td2 = 1.008, 1.038, 1.094, 1.119
    # and one run below 1.0 -- a ~20% false-failure rate. CI itself produced
    # 0.953. td2/jerk sits equally close, at 1.074..1.225.
    #
    # 0.8 keeps the check meaningful (a derivative becoming dramatically cheaper
    # than the one below it still trips it) while clearing the observed spread,
    # which is what this module's docstring already says it is trying to do.
    if not (0.8 <= ratio_td2 <= 12.0):
        raise RuntimeError(
            f"time_deriv2_accurate_over_jerk_accurate out of guard range: {ratio_td2:.3f} "
            "(expected 0.8..12.0)"
        )
    if not (0.8 <= ratio_td3 <= 12.0):
        raise RuntimeError(
            f"time_deriv3_accurate_over_time_deriv2_accurate out of guard range: {ratio_td3:.3f} "
            "(expected 0.8..12.0)"
        )


def _validate_accurate_jerk_differs_from_fast(
    *,
    n: int,
    preset: str,
    basis: str,
    theta: float,
    leaf_size: int,
    max_order: int,
    jerk_fd_dt: float,
    seed: int,
    dtype,
) -> float:
    """Assert the two jerk modes compute different jerks, and return how different.

    This is what the ``jerk_accurate_over_fast`` timing floor was a proxy for.
    ``jerk_mode="accurate"`` adds the analytic far-field source-motion term, so its
    jerk MUST differ from ``fast_approx`` by far more than round-off. Checking the
    output says so directly, is independent of how fast either path happens to be,
    and cannot be defeated by one mode getting cheaper.

    The accelerations are asserted to agree, because the mode changes only the
    jerk -- that catches the opposite failure, a mode that perturbs the force.
    """

    import jax

    key = jax.random.PRNGKey(int(seed))
    positions, masses, velocities = _sample_problem(int(n), key=key, dtype=dtype)
    solver = FastMultipoleMethod(
        preset=str(preset), basis=str(basis), theta=float(theta)
    )

    def run(mode: str):
        return solver.compute_accelerations_and_jerk(
            positions,
            masses,
            velocities,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            jerk_mode=mode,
            jerk_fd_dt=float(jerk_fd_dt),
        )

    acc_fast, jerk_fast = run("fast_approx")
    acc_accurate, jerk_accurate = run("accurate")

    acc_fast = np.asarray(acc_fast, dtype=np.float64)
    acc_accurate = np.asarray(acc_accurate, dtype=np.float64)
    jerk_fast = np.asarray(jerk_fast, dtype=np.float64)
    jerk_accurate = np.asarray(jerk_accurate, dtype=np.float64)

    acc_scale = float(np.linalg.norm(acc_fast)) or 1.0
    acc_drift = float(np.linalg.norm(acc_accurate - acc_fast)) / acc_scale
    # Generous: fp32 by default, and the two modes take different code paths
    # through the same force, so reassociation is expected.
    if acc_drift > 1.0e-4:
        raise RuntimeError(
            "jerk_mode changed the ACCELERATION, which it must not: rel-L2 "
            f"{acc_drift:.3e} between fast_approx and accurate"
        )

    jerk_scale = float(np.linalg.norm(jerk_fast)) or 1.0
    jerk_delta = float(np.linalg.norm(jerk_accurate - jerk_fast)) / jerk_scale
    # Threshold set FROM the measurement, not assumed. At this guard's
    # configuration (N=512, leaf 16, p=4, theta 0.6, fp32) the source-motion term
    # contributes 3.826e-05 rel-L2 -- measured identically on `main` and on the
    # branch that introduced this check, so it is a property of the physics here
    # and not of either tree. It is a small correction, not a leading-order one,
    # which is why a *timing* proxy for "accurate is doing more work" was always
    # weak: it was measuring dispatch overhead rather than the extra term.
    #
    # 1e-6 sits ~40x below the measured value and far above where the term going
    # missing would land: dropping source-motion makes accurate identical to
    # fast_approx, i.e. 0 up to reassociation.
    if jerk_delta < 1.0e-6:
        raise RuntimeError(
            "jerk_mode='accurate' produced essentially the same jerk as "
            f"'fast_approx' (rel-L2 {jerk_delta:.3e} < 1e-6, against a measured "
            "3.826e-05): the analytic far-field source-motion term looks to be "
            "missing"
        )
    return jerk_delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--preset", type=str, default="fast")
    parser.add_argument("--basis", type=str, default="solidfmm")
    parser.add_argument("--theta", type=float, default=0.6)
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--jerk-fd-dt", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    metrics = collect_metrics(
        n=int(args.n),
        runs=int(args.runs),
        warmup=int(args.warmup),
        preset=str(args.preset),
        basis=str(args.basis),
        theta=float(args.theta),
        leaf_size=int(args.leaf_size),
        max_order=int(args.max_order),
        jerk_fd_dt=float(args.jerk_fd_dt),
        seed=int(args.seed),
        dtype=dtype,
    )

    _validate_metrics(metrics)
    metrics["jerk_accurate_vs_fast_rel_l2"] = _validate_accurate_jerk_differs_from_fast(
        n=int(args.n),
        preset=str(args.preset),
        basis=str(args.basis),
        theta=float(args.theta),
        leaf_size=int(args.leaf_size),
        max_order=int(args.max_order),
        jerk_fd_dt=float(args.jerk_fd_dt),
        seed=int(args.seed),
        dtype=dtype,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
