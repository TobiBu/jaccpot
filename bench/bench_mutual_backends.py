"""Phase-5 backend A/B for the mutual (momentum-conserving) FMM.

Answers the three questions ``docs/momentum_conserving_fmm.md`` "Completing
Phase 5" asks, by measurement rather than assumption:

* **near field** -- is a fused mutual Pallas kernel (double-sided scatter, each
  pair evaluated once) actually faster than the pure-JAX bucketed lane? PR-2
  descoped the *gather*-shaped fused near field after measuring ~1.1x at N~1024
  and slower at moderate N; the mutual kernel has a different arithmetic
  intensity, so the verdict is re-taken here rather than inherited.
* **far field** -- how much does the fully fused real M2L (rotate + z-translate +
  rotate-back in one launch) beat the three-stage sandwich (two pure-JAX rotation
  ``vmap``s around the Pallas z-core)?
* **momentum** -- what does the residual actually do on the Pallas lanes, and in
  float32, where the reduction order changes but the pair antisymmetry does not?

Forward *and* reverse are timed: the reverse is the reason the hand-written
analytic VJP exists, and a kernel that only wins forward is not worth shipping.

Usage::

    python -m bench.bench_mutual_backends --sizes 10000 100000 --theta 0.7 --order 4
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


def _system(n: int, seed: int = 0, dtype: Any = jnp.float64):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(0.0, 1.0, (n, 3)), dtype=dtype)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, n), dtype=dtype)
    return positions, masses


def _time(fn: Callable, *args, repeats: int = 3) -> tuple[float, Any]:
    """Best wall-clock of ``repeats`` runs; the first call (compile) is discarded."""
    try:
        result = jax.block_until_ready(fn(*args))
    except Exception as exc:  # pragma: no cover - benchmark robustness
        return float("nan"), exc
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - start)
    return best, result


def _momentum_residual(forces: Any) -> float:
    """``|sum_i F_i|`` normalised by the scale of the terms being summed."""
    total = jnp.sum(forces, axis=0)
    scale = jnp.sum(jnp.abs(forces), axis=0)
    return float(jnp.linalg.norm(total) / (jnp.linalg.norm(scale) + 1e-300))


def _rel_l2(a: Any, b: Any) -> float:
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-300))


def _speedup(base: float, other: float) -> str:
    if not np.isfinite(base) or not np.isfinite(other) or other <= 0:
        return "   n/a"
    return f"{base / other:5.2f}x"


def _near_field_callables(state, pos_sorted, mass_sorted, rung_sorted, weights):
    """Return ``{lane: fn(positions) -> forces}`` for the near field alone."""
    from jaccpot.mutual.nearfield import mutual_near_field_forces

    def make(use_pallas: bool):
        def fn(p):
            return mutual_near_field_forces(
                p,
                mass_sorted,
                leaf_particles=state.leaf_particles,
                leaf_particle_valid=state.leaf_particle_valid,
                near_a=state.near_a,
                near_b=state.near_b,
                near_valid=state.near_valid,
                self_leaves=state.self_leaves,
                softening=state.softening,
                G=state.G,
                rung=rung_sorted,
                level_weights=weights,
                use_pallas=use_pallas,
            )

        return jax.jit(fn)

    return {"jax": make(False), "pallas": make(True)}


def _far_field_callables(state, mass_sorted, far_weights):
    """Return ``{lane: fn(positions) -> forces}`` for the far field alone.

    Three lanes, because the Phase-5 question is which of the two Pallas M2L
    shapes (if either) beats pure JAX:

    * ``jax``          -- pure-JAX rot-scale, the oracle;
    * ``pallas``       -- fully fused rotate + z-translate + rotate-back;
    * ``pallas-zcore`` -- the original three-stage sandwich, i.e. two pure-JAX
      rotation ``vmap``s around the Pallas z-core.

    The lane is selected by ``JACCPOT_MUTUAL_M2L``, which
    ``jaccpot.mutual.farfield`` reads at *trace* time, so setting it around the
    ``jax.jit`` call is what pins each lane. Each ``make()`` builds its own
    ``jax.jit`` object, so the lanes cannot collide in the jit cache.
    """
    import os

    from jaccpot.mutual.farfield import mutual_far_field_forces

    def make(use_pallas: bool, lane: str = "jax"):
        def fn(p):
            return mutual_far_field_forces(
                p,
                mass_sorted,
                state.tree,
                G=state.G,
                far_weights=far_weights,
                use_pallas=use_pallas,
            )

        jitted = jax.jit(fn)

        def traced(p):
            previous = os.environ.get("JACCPOT_MUTUAL_M2L")
            os.environ["JACCPOT_MUTUAL_M2L"] = lane
            try:
                return jitted(p)
            finally:
                if previous is None:
                    os.environ.pop("JACCPOT_MUTUAL_M2L", None)
                else:
                    os.environ["JACCPOT_MUTUAL_M2L"] = previous

        return traced

    return {
        "jax": make(False),
        "pallas-auto": make(True, lane="auto"),
        "pallas-fused": make(True, lane="fused"),
        "pallas-zcore": make(True, lane="zcore"),
    }


def _grad_of(fn: Callable) -> Callable:
    """Scalarise a force field and take its gradient w.r.t. positions."""
    return jax.jit(jax.grad(lambda p: jnp.sum(fn(p) ** 2)))


def _report_lane(
    label: str,
    lanes: dict[str, Callable],
    positions,
    *,
    repeats: int,
    do_grad: bool,
) -> None:
    base_fwd, base_out = _time(lanes["jax"], positions, repeats=repeats)
    print(f"  {label:<6} forward  {'jax':<13} {base_fwd * 1e3:8.2f} ms")
    for name, fn in lanes.items():
        if name == "jax":
            continue
        lane_fwd, lane_out = _time(fn, positions, repeats=repeats)
        if isinstance(lane_out, Exception):
            print(
                f"  {label:<6} forward  {name:<13}      failed   "
                f"{type(lane_out).__name__}: {str(lane_out)[:90]}"
            )
            continue
        parity = (
            float("nan")
            if isinstance(base_out, Exception)
            else _rel_l2(lane_out, base_out)
        )
        print(
            f"  {label:<6} forward  {name:<13} {lane_fwd * 1e3:8.2f} ms   "
            f"{_speedup(base_fwd, lane_fwd)}   rel-L2 {parity:.2e}"
        )
    if not do_grad:
        return
    g_base, g_base_out = _time(_grad_of(lanes["jax"]), positions, repeats=repeats)
    print(f"  {label:<6} reverse  {'jax':<13} {g_base * 1e3:8.2f} ms")
    for name, fn in lanes.items():
        if name == "jax":
            continue
        g_lane, g_lane_out = _time(_grad_of(fn), positions, repeats=repeats)
        if isinstance(g_lane_out, Exception):
            print(
                f"  {label:<6} reverse  {name:<13}      failed   "
                f"{type(g_lane_out).__name__}: {str(g_lane_out)[:90]}"
            )
            continue
        gpar = (
            float("nan")
            if isinstance(g_base_out, Exception)
            else _rel_l2(g_lane_out, g_base_out)
        )
        print(
            f"  {label:<6} reverse  {name:<13} {g_lane * 1e3:8.2f} ms   "
            f"{_speedup(g_base, g_lane)}   rel-L2 {gpar:.2e}"
        )


def run(
    sizes,
    *,
    theta: float,
    order: int,
    leaf_size: int,
    softening: float,
    k_max: int,
    repeats: int,
    dtype_name: str,
    do_grad: bool,
    weighted: bool,
    lanes: tuple[str, ...],
) -> None:
    from jaccpot.mutual import (
        boundary_level_weights,
        build_mutual_state,
        build_mutual_topology,
        mutual_accelerations,
    )
    from jaccpot.mutual.force import _far_pair_weights
    from jaccpot.pallas.m2l_real_fused import pallas_m2l_real_fused_supported
    from jaccpot.pallas.nearfield_mutual import pallas_nearfield_mutual_supported

    dtype = jnp.float32 if dtype_name == "float32" else jnp.float64
    print(f"device            : {jax.devices()[0]}")
    print(f"dtype             : {dtype_name}")
    print(f"fused M2L kernel  : {pallas_m2l_real_fused_supported()}")
    print(f"mutual P2P kernel : {pallas_nearfield_mutual_supported()}")
    print(
        f"config            : theta={theta} order={order} leaf={leaf_size} "
        f"softening={softening} weighted={weighted}"
    )
    print()

    for n in sizes:
        positions, masses = _system(n, dtype=dtype)
        topology, _ = build_mutual_topology(
            positions, masses, theta=theta, order=order, leaf_size=leaf_size
        )
        state = build_mutual_state(topology, softening=softening)
        fwd = state.forward_permutation
        pos_sorted, mass_sorted = positions[fwd], masses[fwd]

        rung_sorted: Optional[Any] = None
        weights: Optional[Any] = None
        far_weights: Optional[Any] = None
        if weighted:
            rung = jnp.asarray(
                np.random.default_rng(1).integers(0, k_max + 1, n), dtype=jnp.int32
            )
            rung_sorted = rung[fwd]
            weights = boundary_level_weights(1, k_max, 1.0e-3, dtype=dtype)
            far_weights = _far_pair_weights(state, rung_sorted, weights)

        print(
            f"N = {n}  leaves={topology.num_leaves}  "
            f"near_pairs={topology.num_near_pairs}  far_pairs={topology.num_far_pairs}"
        )

        if "near" in lanes:
            _report_lane(
                "near",
                _near_field_callables(
                    state, pos_sorted, mass_sorted, rung_sorted, weights
                ),
                pos_sorted,
                repeats=repeats,
                do_grad=do_grad,
            )
        if "far" in lanes:
            _report_lane(
                "far",
                _far_field_callables(state, mass_sorted, far_weights),
                pos_sorted,
                repeats=repeats,
                do_grad=do_grad,
            )
        if "total" not in lanes:
            print()
            continue

        # ---- whole force, as a user selects it -----------------------------
        total_lanes = {}
        for name, use_pallas in (("jax", False), ("pallas", True)):
            st = build_mutual_state(
                topology, softening=softening, use_pallas=use_pallas
            )
            total_lanes[name] = jax.jit(
                lambda p, st=st: mutual_accelerations(
                    st,
                    p,
                    masses,
                    rung=(
                        rung_sorted[state.inverse_permutation]
                        if rung_sorted is not None
                        else None
                    ),
                    level_weights=weights,
                )
            )
        _report_lane("total", total_lanes, positions, repeats=repeats, do_grad=do_grad)

        for name, fn in total_lanes.items():
            acc = fn(positions)
            residual = _momentum_residual(masses[:, None] * acc)
            print(f"  momentum residual  backend={name:<7} {residual:.3e}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10000, 100000])
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--leaf-size", type=int, default=32)
    parser.add_argument("--softening", type=float, default=1.0e-3)
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--no-grad", action="store_true", help="skip reverse timings")
    parser.add_argument(
        "--weighted", action="store_true", help="apply block-step level weights"
    )
    parser.add_argument(
        "--lanes",
        default="near,far,total",
        help="comma-separated subset of near,far,total to measure",
    )
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    run(
        args.sizes,
        theta=args.theta,
        order=args.order,
        leaf_size=args.leaf_size,
        softening=args.softening,
        k_max=args.k_max,
        repeats=args.repeats,
        dtype_name=args.dtype,
        do_grad=not args.no_grad,
        weighted=args.weighted,
        lanes=tuple(x.strip() for x in args.lanes.split(",") if x.strip()),
    )


if __name__ == "__main__":
    main()
