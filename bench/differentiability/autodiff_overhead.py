"""Figure 12 data -- forward vs forward+backward wall-clock ratio vs N.

The claim this figure supports is that the reverse pass costs a bounded multiple
of the forward pass, so gradient-based inference over an FMM force is affordable
rather than merely possible. The plotted quantity is therefore a **ratio**: an
absolute reverse time says nothing without the forward time it was measured
against on the same device and problem. The translation cascade (M2M/M2L/L2L plus
rotations) is linear, so its reverse pass is a transpose and a small constant
factor is the expected answer.

Adapted from ``examples/differentiable_fmm_overhead.py``, which is the canonical
version of this measurement; three things it gets right and a naive version does
not, all of which are kept here:

* **The fused M2L lane is selected with** ``GradConfig(fused_m2l_pallas=True)``,
  not with the constructor's ``use_pallas``, which steers the *near-field* fast
  lane instead. Requesting it on pre-Ampere hardware silently falls back to the
  pure-JAX M2L, so this script asks the runtime whether the lane actually
  *engaged* and records that, rather than reporting the same measurement twice
  under two names.
* **Whether the call was jitted is part of the measurement.** Wrapping the whole
  call in an outer ``jax.jit`` works at moderate N but hits host-side ops in the
  prepared-state sweeps at large N (see ``docs/differentiable_fmm_design.md``,
  "jit limitation"). Timing the eager path without saying so would report
  re-tracing cost as compute -- measured on CPU at N=512, eager dispatch is
  ~2.7 s against ~15 ms for a comparable jitted evaluation. Each row records its
  ``mode``, and the notebook must not compare a jitted row against an eager one.
* **Both arms differentiate the fixed-topology entry point** on one prepared
  ``state``. Rebuilding the tree inside the timed region would put tree
  construction in the forward number and not in the backward one.

Timing uses the shared ``bench/scaling/_timing`` protocol (warm up, then minimum
over repeats), so every timing figure in the paper reports the same statistic.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu python -m bench.differentiability.autodiff_overhead \\
        --n 256,512 --repeats 3 --gpu-select none --json-out /tmp/smoke.json

Paper run::

    python -m bench.differentiability.autodiff_overhead
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

DEFAULT_OUT = "differentiability/autodiff_overhead.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # N = 256/1024/4096, matching examples/differentiable_fmm_overhead.py, and for
    # the reason that example uses them: the outer jax.jit over the whole
    # differentiable call becomes impractical to COMPILE above ~10^4. Measured at
    # N=16384/float64 on an A100, XLA spent 3m21s on `jit__accumulate_m2l_fullbatch`
    # alone and had not finished the module after 30 minutes at 0% GPU utilisation.
    # That is not a runtime cost this figure could report -- it is a compile wall,
    # and it is why the ratio is measured where the jitted path is reachable.
    p.add_argument("--n", default="256,1024,4096")
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument(
        "--basis",
        default="real,complex",
        help="Comma-separated bases (the canonical example reports both)",
    )
    p.add_argument("--leaf-size", type=int, default=32)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--softening", type=float, default=1e-2)
    p.add_argument(
        "--wrt",
        default="positions,masses,both",
        help=(
            "Differentiation targets. 'both' is the realistic inference case and "
            "is not the sum of the other two -- one reverse pass yields both"
        ),
    )
    p.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "jit", "eager", "step"),
        help=(
            "Dispatch arm to time. 'auto' (default, historical behaviour) tries an "
            "outer jax.jit and falls back to eager. 'step' is the route out of the "
            "compile wall documented in docs/fig12_autodiff_overhead_blocked.md: "
            "build the state once, then take a compiled f(positions, masses) from "
            "differentiable_step_fn and differentiate *that*, so the compile is "
            "paid once at setup and the timed region is execution rather than "
            "tracing. Prefer 'step' for a publishable ratio"
        ),
    )
    p.add_argument(
        "--no-jit",
        action="store_true",
        help=(
            "Skip the jax.jit attempt and time eager dispatch directly. The jit "
            "attempt is not merely slow at large N -- measured on an A100 it had "
            "not finished compiling after 18 minutes even at N=256, so the cost is "
            "the unrolled differentiable pipeline rather than the problem size, and "
            "the try/except fallback never fires because slowness is not an "
            "exception. With this set, both arms run eagerly and the ratio is "
            "self-consistent but includes per-call re-tracing on both sides, which "
            "makes it an upper bound rather than a compute ratio"
        ),
    )
    p.add_argument(
        "--nearfield-lane",
        default="auto",
        choices=("auto", "bucketed", "fast_lane"),
        help="GradConfig(nearfield_lane=...); 'auto' picks per N as the library does",
    )
    runmeta.add_common_args(p)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    runmeta.select_gpu(args.gpu_select)
    # float64 throughout, matching the canonical example: a ratio measured in
    # fp32 on one arm and fp64 on the other would not be a ratio.
    args.dtype = "float64"
    runmeta.enable_x64(args.dtype)

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402

    from bench.scaling import _timing as T  # noqa: E402
    from jaccpot import FastMultipoleMethod, GradConfig  # noqa: E402
    from jaccpot.runtime.grad_options import (  # noqa: E402
        grad_option_overrides,
        resolve_grad_options,
    )
    from jaccpot.runtime.kernels.core import (  # noqa: E402
        _fused_complex_m2l_pallas_active,
        _real_m2l_pallas_active,
    )

    def lane_engaged(cfg: GradConfig) -> bool:
        """Ask the runtime whether the fused-Pallas M2L actually engages here.

        The gates are context-locals, so they have to be read inside the same
        override scope the measurement will run under.
        """

        options = resolve_grad_options(cfg, num_particles=0, supports_fast_lane=True)
        with grad_option_overrides(options):
            return bool(
                _fused_complex_m2l_pallas_active() and _real_m2l_pallas_active()
            )

    device = jax.devices()[0]
    cc = str(getattr(device, "compute_capability", "") or "")

    lanes: list[tuple[str, GradConfig]] = [
        (
            "pure_jax",
            GradConfig(nearfield_lane=args.nearfield_lane, fused_m2l_pallas=False),
        )
    ]
    fused_cfg = GradConfig(nearfield_lane=args.nearfield_lane, fused_m2l_pallas=True)
    fused_available = lane_engaged(fused_cfg)
    if fused_available:
        lanes.append(("fused_pallas", fused_cfg))
    else:
        print(
            f"[fig12] fused-Pallas M2L does not engage on this device "
            f"(compute capability {cc or 'n/a'}); it would fall back to the "
            "pure-JAX M2L, so it is recorded as unavailable rather than measured "
            "as a duplicate row."
        )

    ns = [int(v) for v in str(args.n).split(",") if v.strip()]
    bases = [v.strip() for v in str(args.basis).split(",") if v.strip()]
    wrts = [v.strip() for v in str(args.wrt).split(",") if v.strip()]

    records: list[dict[str, Any]] = []
    for basis in bases:
        for n in ns:
            rng = np.random.default_rng(
                runmeta.seed_sequence(args.seed, basis, n) % (2**32)
            )
            positions = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
            masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(n,)), dtype=jnp.float64)
            probe = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)

            for lane_name, grad_config in lanes:

                def measure(mode: str) -> dict[str, Any]:
                    """Time one (mode) arm on a freshly built solver and state.

                    The solver is rebuilt per attempt on purpose. A failed
                    ``jax.jit`` trace leaves a tracer behind in the solver's
                    stateful caches -- that is the "side effect" the leaked-tracer
                    error names -- so a subsequent eager call on the *same* solver
                    fails too, and the fallback silently becomes a second failure
                    rather than a measurement.
                    """

                    fmm = FastMultipoleMethod(
                        basis=basis,
                        theta=float(args.theta),
                        G=1.0,
                        softening=float(args.softening),
                    )
                    state = fmm.prepare_state(
                        positions,
                        masses,
                        leaf_size=int(args.leaf_size),
                        max_order=int(args.order),
                    )

                    def accel(p, m):
                        return fmm.differentiable_accelerations(
                            state, p, m, grad_config=grad_config
                        )

                    if mode == "step":
                        # Route 1 of docs/fig12_autodiff_overhead_blocked.md. The
                        # compile is paid here, at setup, via compile_now -- so it
                        # is outside the timed region on BOTH arms rather than
                        # being amortised unevenly between them.
                        step_fn = fmm.differentiable_step_fn(
                            state,
                            grad_config=grad_config,
                            compile_now=(positions, masses),
                        )

                        def accel(p, m):  # noqa: F811 -- the step-mode forward
                            return step_fn(p, m)

                    def scalar(p, m):
                        return jnp.sum(probe * accel(p, m))

                    # Both arms must cross the SAME jit boundary or the ratio is
                    # not a ratio. In step mode the reverse is jax.jit over a
                    # function that calls the compiled step, which JAX inlines into
                    # one module; timing the standalone step against that compares a
                    # bare compiled call to a fused one, and measured on the complex
                    # basis at N=4096 it produced ratios below 1.0 -- a
                    # forward+backward apparently cheaper than its own forward.
                    # Wrapping the forward the same way restores the symmetry. The
                    # compile is still outside the timed region on both arms: the
                    # inner one via compile_now, the outer via block_until_ready.
                    fwd: Any = jax.jit(accel) if mode in ("jit", "step") else accel
                    jax.block_until_ready(fwd(positions, masses))
                    t_fwd, _, s_fwd = T.time_min_repeat(
                        lambda: fwd(positions, masses),
                        warmup=int(args.warmup),
                        repeats=int(args.repeats),
                    )

                    out: dict[str, Any] = {
                        "n": int(n),
                        "basis": basis,
                        "lane": lane_name,
                        "mode": mode,
                        "forward_min_s": t_fwd,
                        "forward_std_s": s_fwd,
                        "grads": {},
                    }
                    for wrt in wrts:
                        argnums = (
                            0
                            if wrt == "positions"
                            else 1 if wrt == "masses" else (0, 1)
                        )
                        vg_eager = jax.value_and_grad(scalar, argnums=argnums)
                        vg: Any = (
                            jax.jit(vg_eager) if mode in ("jit", "step") else vg_eager
                        )
                        jax.block_until_ready(vg(positions, masses))
                        t_bwd, _, s_bwd = T.time_min_repeat(
                            lambda: vg(positions, masses),
                            warmup=int(args.warmup),
                            repeats=int(args.repeats),
                        )
                        out["grads"][wrt] = {
                            "forward_backward_min_s": t_bwd,
                            "forward_backward_std_s": s_bwd,
                            "ratio": (t_bwd / t_fwd) if t_fwd else None,
                        }
                    return out

                try:
                    if args.mode == "step":
                        row = measure("step")
                    elif args.mode == "jit":
                        row = measure("jit")
                    elif args.mode == "eager" or args.no_jit:
                        row = measure("eager")
                        row["jit_skipped"] = (
                            "--no-jit" if args.no_jit else "--mode eager"
                        )
                    else:
                        try:
                            row = measure("jit")
                        except Exception as jit_exc:
                            row = measure("eager")
                            row["jit_fallback_reason"] = str(jit_exc)[:200]
                    mode = row["mode"]
                    t_fwd = row["forward_min_s"]
                    records.append(row)
                    ratios = "  ".join(
                        f"{k}={v['ratio']:.2f}x" if v["ratio"] else f"{k}=n/a"
                        for k, v in row["grads"].items()
                    )
                    print(
                        f"{basis:>8s} N={n:<8d} {lane_name:<12s} {mode:>5s} "
                        f"fwd={t_fwd*1e3:8.2f} ms  {ratios}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"{basis:>8s} N={n:<8d} {lane_name:<12s} FAILED: "
                        f"{str(exc)[:120]}"
                    )
                    records.append(
                        {
                            "n": int(n),
                            "basis": basis,
                            "lane": lane_name,
                            "error": str(exc)[:400],
                        }
                    )

    config = {
        "n": ns,
        "theta": float(args.theta),
        "order": int(args.order),
        "basis": bases,
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": args.dtype,
        "leaf_size": int(args.leaf_size),
        "preset": "default (FMMPreset.FAST)",
        "distribution": "gaussian (numpy default_rng normal)",
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "softening": float(args.softening),
        "wrt": wrts,
        "lanes": [name for name, _ in lanes],
        "nearfield_lane": args.nearfield_lane,
        "mode_requested": args.mode,
        "compute_capability": cc or None,
        "fused_m2l_pallas_engages": bool(fused_available),
        "timed_region": (
            "differentiable_accelerations on a prebuilt state (fixed topology); "
            "tree construction is outside both arms"
        ),
        "loss": "sum(probe * a), fixed-seed standard-normal probe",
        "jit_compile_wall": (
            "the outer jax.jit over the whole differentiable call does not finish "
            "compiling in a usable time above N ~ 10^4 (measured: >30 min at "
            "N=16384 float64 on an A100), so this sweep stays where it is reachable"
        ),
        "note": (
            "Each row records `mode` (jit, eager or step). Eager rows include "
            "per-call re-tracing and must not be compared against jitted or step "
            "rows; the ratio is null where the forward and reverse arms did not "
            "share a mode. `step` rows are the publishable ones: the forward is a "
            "compiled f(positions, masses) from differentiable_step_fn with the "
            "compile paid at setup, and the reverse is jitted over that same "
            "function, so both arms time execution rather than tracing."
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
