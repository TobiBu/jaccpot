"""Figure 16 data -- the headline: gradient cost against free-parameter count.

The claim this figure carries:

    Reverse-mode differentiation through jaccpot's FMM delivers gradients with
    respect to O(1e6-1e7) free source positions at a cost of a small constant
    multiple of one forward force evaluation, making gradient-based inference
    over discrete mass distributions tractable at a parameter count where finite
    differences are impossible by many orders of magnitude.

So three curves, all against free-parameter count ``P``:

* **forward only** -- flat in ``P`` at fixed ``N``, because the forward map does
  not know how many of its inputs you intend to call parameters.
* **forward + backward** -- also flat, at a small multiple. That multiple *is*
  the claim, and this script's job is to measure it rather than assert it.
* **finite differences, extrapolated** as ``(P + 1) x forward``. Steep. The
  extrapolation uses **this script's own measured forward time** on the same
  device and problem, not a literature figure, which is what makes the
  annotated wall-clock at ``P = 1e7`` an argument rather than a rhetorical
  flourish.

Two honesty rules the numbers here have to obey.

**The FD curve is extrapolated, and is labelled as such.** Nobody ran 3e7
finite-difference evaluations. What is measured is one forward evaluation; the
rest is multiplication, and the record says so in ``method``. Central
differences would be ``2P`` and one-sided ``P + 1``; the cheaper of the two is
used, so the comparison cannot be accused of inflating the baseline.

**Both parameterisations are measured on the same operator.** The parametric
case (7 or 11 parameters) and the positions case (3N) differ *only* in the width
of the pytree handed to ``jax.grad`` -- same tracers, same softening, same
expansion order, same prepared state. A cost difference between them is
therefore a statement about parameter count and nothing else, which is exactly
what the figure claims. This is also what makes the low-``P`` end of the plot
real measurement rather than a single point with a line drawn through it.

**Jitted, and every row says so.** The claim is about the cost of a gradient in
a compiled inference loop, so both arms are wrapped in ``jax.jit`` and the
compile is paid in the untimed warmup. This is not a detail: measured on the CPU
backend at N=256, the *eager* ratio is 20.1x and the jitted ratio on an A100 at
N=4096 is 2.69x, because eager timing measures Python-level dispatch of many
small operations rather than compute. ``docs/differentiable_fmm_design.md``
records that an outer ``jax.jit`` works at moderate ``N`` but meets host-side
ops in the prepared-state sweeps at large ``N``, so a row that cannot be jitted
falls back to eager and records ``mode: "eager"`` -- and **the notebook must not
compare a jitted row against an eager one**, exactly as fig12 must not.

Timing follows the shared protocol in ``bench/scaling/_timing`` -- warm up, then
the minimum over repeats, blocked on device -- so this figure's statistic
matches every other timing figure in the paper. Peak device memory is read from
JAX's own allocator stats, and the first evaluation is excluded from every timing
because it carries a one-time XLA compile measured in tens of seconds.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu python -m bench.payoff_static.gradient_cost_vs_nparams \\
        --n 256,512 --repeats 2 --warmup 1 --gpu-select none \\
        --json-out /tmp/smoke.json

Paper run::

    python -m bench.payoff_static.gradient_cost_vs_nparams
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "density_reconstruction/gradient_cost_vs_nparams.json"

#: Parameter counts the figure annotates in human units. 1e7 is the headline.
ANNOTATE_AT = (1_000_000, 10_000_000, 30_000_000)


def _parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--n",
        default="1024,4096,16384,65536,262144,1048576",
        help="Comma-separated source counts N; free parameters are 3N",
    )
    p.add_argument("--tracers", type=int, default=4096, help="Tracer count M")
    p.add_argument("--order", type=int, default=4, help="Expansion order")
    p.add_argument("--theta", type=float, default=0.5, help="MAC parameter")
    p.add_argument("--leaf-size", type=int, default=64, help="Leaf size")
    p.add_argument("--softening", type=float, default=1.0e-2, help="Plummer softening")
    p.add_argument("--preset", default="accurate", help="jaccpot preset")
    p.add_argument("--basis", default="solidfmm", help="Expansion basis")
    p.add_argument("--repeats", type=int, default=5, help="Timed repeats")
    p.add_argument("--warmup", type=int, default=2, help="Untimed warmup calls")
    p.add_argument(
        "--mode",
        choices=("jit", "eager"),
        default="jit",
        help=(
            "Time the compiled path (default) or the eager one. Eager timing "
            "measures Python dispatch, not compute; see the module docstring"
        ),
    )
    p.add_argument(
        "--parametric",
        action="store_true",
        default=True,
        help="Also measure the low-parameter-count parametric arm (default on)",
    )
    p.add_argument(
        "--no-parametric",
        dest="parametric",
        action="store_false",
        help="Skip the parametric arm",
    )
    runmeta.add_common_args(p)
    return p.parse_args()


def _peak_memory_bytes() -> Optional[int]:
    """Return peak device memory in bytes, if the backend reports it.

    Returns
    -------
    Optional[int]
        ``peak_bytes_in_use`` from the device's allocator, or ``None`` on a
        backend that keeps no such statistic (the CPU backend does not).
    """
    import jax

    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:  # pragma: no cover - backend dependent
        return None
    if not stats:
        return None
    for key in ("peak_bytes_in_use", "peak_pool_bytes", "bytes_in_use"):
        if key in stats:
            return int(stats[key])
    return None


def _reset_peak_memory() -> None:
    """Ask the backend to forget the previous point's peak, if it can.

    A peak carried over from a larger ``N`` would be reported against a smaller
    one, which is the kind of error that makes a memory panel meaningless.
    """
    import jax

    try:
        jax.devices()[0].clear_memory_stats()
    except Exception:  # pragma: no cover - not all backends implement it
        pass


def _cost_analysis(fn: Any, argument: Any) -> Optional[Dict[str, float]]:
    """Return XLA's FLOP and byte counts for a compiled callable.

    Wall-clock alone cannot carry this figure's claim, because at small ``N``
    the FMM is **launch-latency bound**: measured on an idle A100 at N=4096,
    the compiled forward does 3.66e7 flops -- microseconds of arithmetic at
    fp64 peak -- and takes 60 ms, which is essentially all kernel-launch and
    dispatch overhead. In that regime a wall-clock ratio between two graphs
    reports which one XLA fused into fewer kernels, not what the reverse pass
    costs, and it can come out *below one*.

    FLOPs and bytes are hardware- and launch-independent, so they answer the
    "small constant multiple" question at every ``N``. Both are recorded.

    Parameters
    ----------
    fn : Any
        A ``jax.jit``-wrapped callable.
    argument : Any
        One example argument, for lowering.

    Returns
    -------
    Optional[Dict[str, float]]
        ``flops`` and ``bytes_accessed``, or ``None`` if the backend offers no
        cost analysis (or the callable is not jitted).
    """
    try:
        analysis = fn.lower(argument).compile().cost_analysis()
    except Exception:  # pragma: no cover - backend dependent
        return None
    if isinstance(analysis, list):
        analysis = analysis[0] if analysis else None
    if not analysis:
        return None
    return {
        "flops": float(analysis.get("flops", float("nan"))),
        "bytes_accessed": float(analysis.get("bytes accessed", float("nan"))),
    }


def _measure_one(
    *,
    n: int,
    args: argparse.Namespace,
    timing: Any,
) -> List[Dict[str, Any]]:
    """Measure forward and forward+backward cost at one ``N``.

    Parameters
    ----------
    n : int
        Source count.
    args : argparse.Namespace
        Parsed command line.
    timing : Any
        The ``bench.scaling._timing`` module, imported after device selection.

    Returns
    -------
    List[Dict[str, Any]]
        One record per parameterisation measured at this ``N``.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaccpot.applications.density_reconstruction.forward import (
        make_forward_operator,
    )
    from jaccpot.applications.density_reconstruction.parameterize import (
        make_parameterization,
    )
    from jaccpot.applications.density_reconstruction.truth import (
        TruthConfig,
        sample_composite,
        sample_tracers,
    )

    config = TruthConfig(
        num_particles=int(n),
        num_tracers=int(args.tracers),
        seed=int(args.seed),
        softening=float(args.softening),
    )
    positions = sample_composite(config)
    tracers = sample_tracers(config, positions)
    source_mass = float(config.total_mass) / float(n)

    operator = make_forward_operator(
        tracer_positions=tracers,
        source_mass=source_mass,
        num_sources=int(n),
        softening=float(args.softening),
        order=int(args.order),
        theta=float(args.theta),
        leaf_size=int(args.leaf_size),
        preset=str(args.preset),
        basis=str(args.basis),
    )
    # One prepared state for every arm: rebuilding inside a timed region would
    # put host-side tree construction into the forward number and not into the
    # backward one.
    state = operator.prepare(positions)
    observed = jnp.zeros((int(args.tracers), 3), dtype=jnp.float64)

    kinds = ["positions"] + (["parametric"] if args.parametric else [])
    records: List[Dict[str, Any]] = []
    for kind in kinds:
        parameterization = make_parameterization(kind, config=config)
        params = (
            parameterization.pack(positions)
            if kind == "positions"
            else parameterization.pack(parameterization.true_params(config))
        )

        def forward_fn(p: Any) -> Any:
            return operator.evaluate_at_topology(
                state, parameterization.to_positions(p)
            )

        def objective(p: Any) -> Any:
            predicted = operator.evaluate_at_topology(
                state, parameterization.to_positions(p)
            )
            return jnp.sum((predicted - observed) ** 2)

        want_jit = str(args.mode) == "jit"
        mode = "jit" if want_jit else "eager"
        forward_call = jax.jit(forward_fn) if want_jit else forward_fn
        gradient_call = (
            jax.jit(jax.grad(objective)) if want_jit else jax.grad(objective)
        )
        if want_jit:
            # Trip the jit limitation here, in the untimed region, rather than
            # inside a timed repeat. A fallback is a result about the pipeline,
            # not a failure of the measurement.
            try:
                jax.block_until_ready(forward_call(params))
                jax.block_until_ready(gradient_call(params))
            except Exception as exc:
                print(
                    f"    N={n} {kind}: jit unavailable "
                    f"({type(exc).__name__}); falling back to eager",
                    flush=True,
                )
                mode = "eager"
                forward_call = forward_fn
                gradient_call = jax.grad(objective)

        _reset_peak_memory()
        forward_min, forward_mean, forward_sd = timing.time_min_repeat(
            lambda: forward_call(params),
            warmup=int(args.warmup),
            repeats=int(args.repeats),
        )
        forward_peak = _peak_memory_bytes()

        _reset_peak_memory()
        grad_min, grad_mean, grad_sd = timing.time_min_repeat(
            lambda: gradient_call(params),
            warmup=int(args.warmup),
            repeats=int(args.repeats),
        )
        grad_peak = _peak_memory_bytes()

        forward_cost = _cost_analysis(forward_call, params) if mode == "jit" else None
        gradient_cost = _cost_analysis(gradient_call, params) if mode == "jit" else None
        flop_ratio = None
        if forward_cost and gradient_cost and forward_cost["flops"]:
            flop_ratio = gradient_cost["flops"] / forward_cost["flops"]

        num_free = int(parameterization.num_free)
        # One-sided finite differences need P + 1 forward evaluations; central
        # would need 2P. The cheaper baseline is the honest one to compare to.
        fd_seconds = (num_free + 1) * forward_min

        record = {
            "N": int(n),
            "M": int(args.tracers),
            "parameterization": kind,
            "num_free_parameters": num_free,
            "forward_seconds": forward_min,
            "forward_seconds_mean": forward_mean,
            "forward_seconds_stdev": forward_sd,
            "forward_backward_seconds": grad_min,
            "forward_backward_seconds_mean": grad_mean,
            "forward_backward_seconds_stdev": grad_sd,
            "backward_over_forward": grad_min / forward_min if forward_min else None,
            "forward_peak_bytes": forward_peak,
            "forward_backward_peak_bytes": grad_peak,
            "finite_difference_extrapolated_seconds": fd_seconds,
            "finite_difference_method": "one_sided_(P+1)_forward_evaluations",
            "finite_difference_is_extrapolated": True,
            "autodiff_speedup_over_fd": (fd_seconds / grad_min if grad_min else None),
            "mode": mode,
            "forward_cost_analysis": forward_cost,
            "forward_backward_cost_analysis": gradient_cost,
            "backward_over_forward_flops": flop_ratio,
            # A wall-clock ratio at or below 1 is arithmetically impossible for
            # a reverse pass over the same forward graph, so it is the marker
            # that this point is launch-latency bound rather than compute
            # bound. Such a point still carries the figure's OTHER claim --
            # cost flat in P -- but its ratio must not be quoted as the
            # reverse-pass overhead, and the notebook must not plot it as one.
            "latency_bound": bool(
                grad_min <= forward_min * 1.05 if forward_min else False
            ),
        }
        records.append(record)
        flop_note = f"  flops x{flop_ratio:.2f}" if flop_ratio else ""
        latency_note = "  [LATENCY-BOUND]" if record["latency_bound"] else ""
        print(
            f"  N={n:>9} {kind:>10}: P={num_free:>9}  "
            f"fwd {forward_min*1e3:9.3f} ms  fwd+bwd {grad_min*1e3:9.3f} ms  "
            f"ratio {record['backward_over_forward']:.2f}x{flop_note}{latency_note}  "
            f"FD(extrap) {fd_seconds:.3e} s",
            flush=True,
        )
        del parameterization, params
    del state
    return records


def main() -> int:
    """Run the sweep and write the results JSON.

    Returns
    -------
    int
        Process exit status.
    """
    args = _parse_args()
    runmeta.select_gpu(args.gpu_select)
    runmeta.enable_x64(args.dtype)

    import jax  # noqa: F401  -- imported for its side effect of backend init

    from bench.scaling import _timing as timing

    n_list = [int(v) for v in str(args.n).split(",") if v]
    print(
        f"gradient_cost_vs_nparams: N in {n_list}, M={args.tracers}, "
        f"order={args.order}, theta={args.theta}, leaf={args.leaf_size}",
        flush=True,
    )

    records: List[Dict[str, Any]] = []
    for n in n_list:
        try:
            records.extend(_measure_one(n=n, args=args, timing=timing))
        except Exception as exc:  # pragma: no cover - OOM at the top of the sweep
            # An out-of-memory at the largest N is a result, not a crash: it is
            # where the single-device ceiling is, and fig16's memory panel needs
            # the point recorded rather than the whole sweep discarded.
            print(f"  N={n}: FAILED ({type(exc).__name__}: {exc})", flush=True)
            records.append(
                {
                    "N": int(n),
                    "M": int(args.tracers),
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            break

    # The annotation the figure's argument rests on, computed here so the
    # notebook cannot derive it differently.
    usable = [r for r in records if not r.get("failed")]
    forward_reference = min(
        (r["forward_seconds"] for r in usable if r["parameterization"] == "positions"),
        default=None,
    )
    annotations = None
    if usable:
        largest = max(
            (r for r in usable if r["parameterization"] == "positions"),
            key=lambda r: r["N"],
        )
        annotations = {
            "reference_N": largest["N"],
            "reference_forward_seconds": largest["forward_seconds"],
            "reference_forward_backward_seconds": largest["forward_backward_seconds"],
            "finite_difference_at": {
                str(p): {
                    "seconds": (p + 1) * largest["forward_seconds"],
                    "years": (p + 1) * largest["forward_seconds"] / 31_557_600.0,
                }
                for p in ANNOTATE_AT
            },
        }

    payload = {
        "records": records,
        "annotations": annotations,
        "min_forward_seconds": forward_reference,
    }
    out = args.json_out or DEFAULT_OUT
    written = jsonio.write_result(
        out,
        config={
            "n": n_list,
            "theta": float(args.theta),
            "order": int(args.order),
            "basis": str(args.basis),
            "seed": int(args.seed),
            "device": runmeta.device_label(),
            "precision": str(args.dtype),
            "M": int(args.tracers),
            "leaf_size": int(args.leaf_size),
            "softening": float(args.softening),
            "preset": str(args.preset),
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "mode": str(args.mode),
        },
        meta=runmeta.run_meta(),
        data=payload,
    )
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
