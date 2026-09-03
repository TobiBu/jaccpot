"""Figure 20 data -- the same fit on 1/2/4/8 GPUs, and the ceiling it moves.

The secondary claim of section 7: *the same optimization runs sharded across
multiple GPUs, so parameter count is bounded by aggregate device memory rather
than by a single device.* Two measurements, because the claim has two halves.

**Strong scaling** -- a fixed parameter count on an increasing device count.
Answers "does it get faster", and links back to section 5, whose strong- and
weak-scaling figures this one should be read against.

**The ceiling** -- the largest parameter count that fits, per device count. This
is the half that carries the claim, and it is measured the only honest way: by
running until it fails and recording where. A ceiling asserted from a memory
model is not a ceiling.

What is and is not sharded
--------------------------
This script shards the **optimisation**: the parameter array is placed across
the mesh, so a parameter count exceeding one device's memory becomes runnable.
That is precisely the claim as stated. It is *not* the same thing as jaccpot's
distributed FMM force evaluation (``jaccpot.distributed``, section 5), which
partitions the *sources* across devices and exchanges halos. A run's record
says which of the two it used in ``sharding_mode``, because a wall-clock number
from one is not comparable with the other, and because "runs on 8 GPUs" would
otherwise be ambiguous between them.

An OOM is a data point, not a crash. Every failure is recorded with its device
count, parameter count and exception type, and the sweep continues -- the whole
figure is about where the ceiling is.

Usage
-----
CPU smoke (one "device")::

    JAX_PLATFORMS=cpu python -m bench.payoff_static.multigpu_scaling \\
        --n 128 --tracers 32 --iterations 3 --device-counts 1 \\
        --gpu-select none --json-out /tmp/smoke.json

Paper run (claims the whole mesh in one autocvd selection)::

    python -m bench.payoff_static.multigpu_scaling --device-counts 1,2,4,8
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

DEFAULT_OUT = "density_reconstruction/multigpu_scaling.json"


def _parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--device-counts", default="1,2,4,8", help="Device counts to measure"
    )
    p.add_argument(
        "--n",
        default="262144",
        help="Fixed source count for the strong-scaling arm",
    )
    p.add_argument("--tracers", type=int, default=8192, help="Tracer count M")
    p.add_argument("--iterations", type=int, default=20, help="Gradient steps")
    p.add_argument("--learning-rate", type=float, default=1.0e-3, help="Step size")
    p.add_argument("--order", type=int, default=4, help="Expansion order")
    p.add_argument("--theta", type=float, default=0.5, help="MAC parameter")
    p.add_argument("--leaf-size", type=int, default=64, help="Leaf size")
    p.add_argument("--softening", type=float, default=1.0e-2, help="Plummer softening")
    p.add_argument(
        "--mode",
        choices=("distributed", "parameter_sharding"),
        default="distributed",
        help=(
            "Which multi-device path to measure. 'distributed' partitions the "
            "SOURCES across devices via jaccpot.distributed (section 5's path) "
            "and is the one that supports the claim; 'parameter_sharding' "
            "shards only the parameter array and is kept because its failure "
            "to help is a measured result"
        ),
    )
    p.add_argument(
        "--caps",
        default="",
        help=(
            "Comma-separated DistributedFMMConfig cap overrides, e.g. "
            "'max_pair_queue=2000000,process_block=512'. They default to "
            "auto-sizing from N, and the auto-sized values are what set the "
            "distributed path's memory ceiling, so a ceiling reported without "
            "these is a statement about the defaults"
        ),
    )
    p.add_argument(
        "--ceiling-iterations",
        type=int,
        default=1,
        help=(
            "Gradient steps per ceiling rung. The ceiling is a MEMORY question "
            "-- does one forward-plus-backward fit -- so one step answers it, "
            "and the distributed path costs ~134 s per gradient at N=65536, "
            "which makes a full fit per rung unaffordable"
        ),
    )
    p.add_argument(
        "--ceiling-ladder",
        default="",
        help=(
            "Comma-separated N ladder for the ceiling arm, largest last. Empty "
            "skips the ceiling measurement (which is what the CI smoke wants)"
        ),
    )
    runmeta.add_common_args(p)
    return p.parse_args()


def _parse_caps(text: str) -> Dict[str, int]:
    """Parse ``name=value`` cap overrides from the command line.

    Parameters
    ----------
    text : str
        Comma-separated ``name=value`` pairs, or empty.

    Returns
    -------
    Dict[str, int]
        Field name to integer value; empty when nothing was given.

    Raises
    ------
    ValueError
        If an entry is not ``name=value`` with an integer value. Silently
        ignoring a malformed cap would report a ceiling measured at settings
        nobody chose.
    """
    caps: Dict[str, int] = {}
    for entry in str(text).split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"cap override {entry!r} is not name=value")
        name, _, value = entry.partition("=")
        try:
            caps[name.strip()] = int(value)
        except ValueError as exc:
            raise ValueError(f"cap override {entry!r} has a non-integer value") from exc
    return caps


def _fit_once(
    *,
    n: int,
    devices: List[Any],
    args: argparse.Namespace,
    iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one instrumented fit at one ``N`` on one device set.

    Parameters
    ----------
    n : int
        Source count.
    devices : List[Any]
        Devices to shard the parameters over.
    args : argparse.Namespace
        Parsed command line.
    iterations : Optional[int]
        Override the step count. The ceiling arm passes 1: whether a parameter
        count fits is a memory question, and one gradient answers it.

    Returns
    -------
    Dict[str, Any]
        The record for this point.
    """
    import numpy as np

    from jaccpot.applications.density_reconstruction.diagnostics import field_residual
    from jaccpot.applications.density_reconstruction.fit import FitConfig, run_fit
    from jaccpot.applications.density_reconstruction.forward import (
        make_forward_operator,
    )
    from jaccpot.applications.density_reconstruction.loss import Regularization
    from jaccpot.applications.density_reconstruction.parameterize import (
        initial_positions,
        make_parameterization,
    )
    from jaccpot.applications.density_reconstruction.truth import (
        TruthConfig,
        make_ground_truth,
    )

    config = TruthConfig(
        num_particles=int(n),
        num_tracers=int(args.tracers),
        seed=int(args.seed),
        softening=float(args.softening),
        generating_order=max(int(args.order) + 2, 6),
        generating_theta=min(float(args.theta), 0.4),
        generating_leaf_size=int(args.leaf_size),
    )
    truth = make_ground_truth(config)
    if str(args.mode) == "distributed":
        from jaccpot.applications.density_reconstruction.distributed import (
            make_distributed_forward_operator,
        )

        operator = make_distributed_forward_operator(
            tracer_positions=truth.tracer_positions,
            source_mass=truth.source_mass,
            num_sources=int(n),
            num_devices=len(devices),
            softening=float(args.softening),
            order=int(args.order),
            theta=float(args.theta),
            leaf_size=int(args.leaf_size),
            devices=list(devices),
            caps=_parse_caps(args.caps),
        )
    else:
        operator = make_forward_operator(
            tracer_positions=truth.tracer_positions,
            source_mass=truth.source_mass,
            num_sources=int(n),
            softening=float(args.softening),
            order=int(args.order),
            theta=float(args.theta),
            leaf_size=int(args.leaf_size),
        )
    parameterization = make_parameterization("positions", config=config)
    start = initial_positions(
        truth.source_positions,
        mode="perturbed_truth",
        seed=int(args.seed) + 5,
        perturbation=0.05,
    )

    distributed = str(args.mode) == "distributed"
    result = run_fit(
        operator=operator,
        observed=truth.observed_accelerations,
        parameterization=parameterization,
        initial_params=parameterization.pack(start),
        config=FitConfig(
            num_iterations=int(args.iterations if iterations is None else iterations),
            learning_rate=float(args.learning_rate),
            rebuild_cadence=1,
            regularization=Regularization.none(),
            history_every=1,
            # The intensive churn is O(P) host work per rebuild and this arm is
            # about wall-clock, so it is sampled rather than measured every step.
            intensive_every=max(int(args.iterations) // 4, 1),
            seed=int(args.seed),
            # The distributed operator exposes no radix prepared state, so the
            # switch instrumentation has nothing to fingerprint.
            track_switches=not distributed,
        ),
        # Parameter sharding is what `devices` does; the distributed operator
        # already owns its mesh, and handing it `devices` too would shard the
        # parameters on top of a partitioned force, mixing the two modes.
        devices=None if distributed else devices,
    )
    after = field_residual(
        operator.evaluate(result.positions),
        truth.observed_accelerations,
        clean=truth.clean_accelerations,
    )
    return {
        "N": int(n),
        "M": int(args.tracers),
        "num_free_parameters": int(parameterization.num_free),
        "num_devices": len(devices),
        "devices": [str(d) for d in devices],
        "sharding_mode": operator.record().get("sharding_mode", str(args.mode)),
        "mode": str(args.mode),
        "operator": operator.record(),
        "iterations": int(args.iterations if iterations is None else iterations),
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "field_residual_after": after,
        "timing": result.timing,
        "switch_summary": result.switch_summary,
    }


def main() -> int:
    """Run the sweep and write the results JSON.

    Returns
    -------
    int
        Process exit status.

    Raises
    ------
    SystemExit
        If distributed mode is asked for several device counts in one process,
        which deadlocks in the collective rendezvous.
    """
    args = _parse_args()
    device_counts = [int(v) for v in str(args.device_counts).split(",") if v]
    if str(args.mode) == "distributed" and len(device_counts) > 1:
        raise SystemExit(
            "distributed mode takes ONE device count per process. Asking one "
            "process for several DEADLOCKS in the collective rendezvous once a "
            "clique of a different size has been formed -- measured here as 59 "
            "minutes of idle GPUs on 'Acquire clique: devices=4' after a "
            "2-device fit had completed. Run once per count with its own "
            "--json-out and merge. Parameter sharding has no collectives and "
            "takes several counts in one process happily."
        )
    # One selection for the whole mesh: claiming devices one at a time races
    # another process onto the same card and then reports a scaling number
    # measured against a competitor.
    runmeta.select_gpu(args.gpu_select, num_gpus=max(device_counts))
    runmeta.enable_x64(args.dtype)

    import jax

    available = jax.devices()
    print(
        f"multigpu_scaling: JAX sees {len(available)} device(s): "
        f"{[str(d) for d in available]}",
        flush=True,
    )

    strong_n = int(str(args.n).split(",")[0])
    ladder = [int(v) for v in str(args.ceiling_ladder).split(",") if v]

    out = args.json_out or DEFAULT_OUT
    config_record = {
        "n": [strong_n],
        "theta": float(args.theta),
        "order": int(args.order),
        "basis": "solidfmm",
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": str(args.dtype),
        "M": int(args.tracers),
        "leaf_size": int(args.leaf_size),
        "softening": float(args.softening),
        "iterations": int(args.iterations),
        "device_counts": device_counts,
        "ceiling_ladder": ladder,
        "ceiling_iterations": int(args.ceiling_iterations),
        "caps": _parse_caps(args.caps),
        "sharding_mode": str(args.mode),
        "mode": str(args.mode),
    }
    records: List[Dict[str, Any]] = []
    ceiling: List[Dict[str, Any]] = []

    def flush() -> Any:
        """Write the results JSON as it currently stands.

        Returns
        -------
        Any
            The path written.

        Notes
        -----
        After every point. A distributed gradient costs ~150 s at N=65536, so a
        run stopped or deadlocked partway would otherwise leave nothing behind
        -- which is exactly what the clique deadlock did, twice.
        """
        return jsonio.write_result(
            out,
            config=config_record,
            meta=runmeta.run_meta(),
            data={"records": records, "ceiling": ceiling},
        )

    for count in device_counts:
        if count > len(available):
            print(
                f"  ndev={count}: SKIPPED, JAX sees only {len(available)} "
                "device(s). Refusing to fabricate a scaling point.",
                flush=True,
            )
            records.append(
                {
                    "num_devices": count,
                    "skipped": True,
                    "reason": f"only {len(available)} devices visible",
                }
            )
            flush()
            continue
        if str(args.mode) == "distributed" and count < 2:
            # Upstream limitation, not a configuration error: the distributed
            # evaluator's per-device tree build asserts "Need at least one
            # particle" inside yggdrax when there is a single domain. The path
            # exists for cross-device halos and is not built for ndev=1. The
            # single-device number belongs to the radix operator, a DIFFERENT
            # code path -- run --mode parameter_sharding for it, and do not put
            # the two on one strong-scaling curve without saying so.
            print(
                f"  ndev={count}: SKIPPED in distributed mode -- the "
                "distributed evaluator requires at least 2 devices "
                "(yggdrax build_tree asserts on a single domain)",
                flush=True,
            )
            records.append(
                {
                    "num_devices": count,
                    "skipped": True,
                    "mode": str(args.mode),
                    "reason": (
                        "distributed force evaluation requires ndev >= 2; "
                        "yggdrax's per-device build_tree asserts 'Need at "
                        "least one particle' with a single domain. The "
                        "single-device baseline is the radix operator, a "
                        "different code path."
                    ),
                }
            )
            flush()
            continue

        devices = list(available[:count])
        try:
            record = _fit_once(n=strong_n, devices=devices, args=args)
            records.append(record)
            print(
                f"  ndev={count}: P={record['num_free_parameters']:>9} "
                f"wall {record['timing']['wall_seconds']:8.2f}s  "
                f"mean grad "
                f"{record['timing']['mean_later_gradient_seconds']*1e3:8.2f} ms  "
                f"loss {record['initial_loss']:.3e}->{record['final_loss']:.3e}",
                flush=True,
            )
        except Exception as exc:
            print(f"  ndev={count}: FAILED ({type(exc).__name__}: {exc})", flush=True)
            records.append(
                {
                    "num_devices": count,
                    "N": strong_n,
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
        flush()

        # The ceiling arm: climb the ladder until it breaks, and record where.
        for n in ladder:
            try:
                point = _fit_once(
                    n=n,
                    devices=devices,
                    args=args,
                    iterations=int(args.ceiling_iterations),
                )
                point["arm"] = "ceiling"
                ceiling.append(point)
                print(
                    f"    ceiling ndev={count} N={n:>10} "
                    f"P={point['num_free_parameters']:>10}: OK",
                    flush=True,
                )
                flush()
            except Exception as exc:
                print(
                    f"    ceiling ndev={count} N={n:>10}: FAILED "
                    f"({type(exc).__name__})  <-- this is the ceiling",
                    flush=True,
                )
                ceiling.append(
                    {
                        "arm": "ceiling",
                        "num_devices": count,
                        "N": int(n),
                        "num_free_parameters": 3 * int(n),
                        "failed": True,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                )
                flush()
                break

    print(f"wrote {flush()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
