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
``--mode distributed`` (the default) partitions the **sources** across the
devices: the observation operator is
:class:`~jaccpot.applications.density_reconstruction.distributed.DistributedForwardOperator`,
the fused mesh lane (``FusedMeshForce`` with its WP-G gradient,
``make_mesh_grad_fn``), particles placed as Morton domains, the cross-domain
exchange sized by ``CrossCapacities``. ``--mode parameter_sharding`` keeps the
single-card operator (``make_forward_operator(lane=--lane)``, the fused lane by
default) and only places the parameter array across the devices; September
measured that this moves neither the wall clock nor the ceiling, and it is kept
because that negative result is part of the argument. A run's record says which
of the two it used in ``sharding_mode``, because a wall-clock number from one is
not comparable with the other.

ndev = 1 is a mesh point
------------------------
The September path's per-device tree build asserted on a single domain, so
distributed mode used to skip one device. The mesh lane runs a one-device mesh
(no cross exchange), so ndev=1 is now measured on the same code path as 2, 4
and 8, and the strong-scaling curve has its own baseline instead of borrowing
the single-card operator's.

Failures are data, and of two kinds
-----------------------------------
An OOM is a data point, not a crash. Every failure is recorded with its device
count, parameter count, exception type and ``failure_kind``, the JSON is
rewritten after every point, and the sweep continues. ``failure_kind`` separates
``"out_of_memory"`` (the ceiling) from ``"verdict"`` -- the mesh force's bits
said some device's lists overflowed or are not ones the reverses read
(``DistributedForwardOperator.prepare`` refuses those, since their gradient
would be NaN). A verdict failure is a capacity setting, **not** a ceiling, and
the ceiling arm says so.

Cross capacities
----------------
``--caps`` takes ``CrossCapacities`` keywords (``name=value,...``); the
September ``DistributedFMMConfig`` caps (``max_pair_queue`` and friends) do not
exist on this lane and are refused. Without ``--caps`` the operator passes
``cross_caps="auto"``, which it sizes at every epoch's ``prepare`` with the
force's own ``plan_cross`` (fixed 2026-11-27; as first ported in A3 it ran at the
measurement capacities and failed from N=16384 on two devices). That planning is
up to a few measurement compiles per epoch, and they land in the fit's
``wall_seconds`` and ``mean_later_prepare_seconds``. A fig20 TIMING run should
therefore pass the explicit ``--caps`` that ``distributed_caps_probe.py`` prints,
so prepare times measure the prepare, not the sizing.

One device count per process, in distributed mode
-------------------------------------------------
JAX fixes the device count when it initialises, and asking one process for
several mesh sizes **deadlocked** the September path: after a 2-device clique
had been formed, acquiring a 4-device one hung in the collective rendezvous
(``Acquire clique: devices=4``) with every GPU idle -- 59 minutes of nothing. So
more than one ``--device-counts`` entry is refused in distributed mode; run the
script once per count and merge::

    for n in 1 2 4; do
        python -m bench.payoff_static.multigpu_scaling --mode distributed \\
            --device-counts $n --json-out density_reconstruction/mg_$n.json
    done
    python -m bench.payoff_static.multigpu_scaling --merge-from \\
        density_reconstruction/mg_1.json,density_reconstruction/mg_2.json,\\
density_reconstruction/mg_4.json

Parameter sharding has no collectives to deadlock on and takes several counts in
one process happily.

Usage
-----
CPU smoke of the mesh lane, one process per device count (``--cpu-interpret``
forces that many host devices and applies the S9 interpret set, the mesh
gradient's only CPU route; it compiles for minutes and measures no timing)::

    for n in 1 2; do
        python -m bench.payoff_static.multigpu_scaling --cpu-interpret \\
            --jit-within-epoch --n 128 --tracers 32 --iterations 2 \\
            --rebuild-cadence 2 \\
            --order 3 --leaf-size 16 --device-counts $n \\
            --gpu-select none --json-out /tmp/mg_$n.json
    done

CPU smoke of parameter sharding (the single-card fused operator)::

    JAX_PLATFORMS=cpu python -m bench.payoff_static.multigpu_scaling \\
        --mode parameter_sharding --n 128 --tracers 32 --iterations 3 \\
        --order 3 --leaf-size 16 --device-counts 1 \\
        --gpu-select none --json-out /tmp/smoke.json

Paper run (claims the whole mesh in one autocvd selection, one count per
process). Size the cross capacities first; ``--caps`` is the probe's
``data.suggested_caps[0].cli`` for the same N and device count::

    python -m bench.payoff_static.distributed_caps_probe --device-count 4 \\
        --n 262144 --json-out density_reconstruction/caps_probe_4.json
    python -m bench.payoff_static.multigpu_scaling --device-counts 4 \\
        --caps CAPS_FROM_PROBE --json-out density_reconstruction/mg_4.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.payoff_static import _common  # noqa: E402
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
            "SOURCES across devices on the fused mesh lane "
            "(DistributedForwardOperator) and is the one that supports the "
            "claim; 'parameter_sharding' shards only the parameter array of "
            "the single-card operator (--lane) and is kept because its failure "
            "to help is a measured result"
        ),
    )
    p.add_argument(
        "--merge-from",
        action="append",
        default=[],
        help=(
            "Result JSONs to concatenate into --json-out and exit, measuring "
            "nothing. Repeatable, and each value may itself be a "
            "comma-separated list -- it used to be comma-only, and a repeated "
            "flag then silently kept just the last one and wrote a merge of "
            "one slice over a complete artifact. This is how the per-device-count runs "
            "that distributed mode forces become the one artifact a figure "
            "reads"
        ),
    )
    p.add_argument(
        "--caps",
        default="",
        help=(
            "Comma-separated CrossCapacities overrides for the mesh lane's "
            "cross-domain exchange, e.g. 'send_particle_cap=65536,"
            "recv_particle_cap=65536' (distributed_caps_probe.py prints a "
            "measured set). Empty: cross_caps='auto', sized per epoch by "
            "plan_cross (adds measurement compiles to prepare times; see the "
            "module docstring). The September "
            "DistributedFMMConfig caps are refused"
        ),
    )
    p.add_argument(
        "--rebuild-cadence",
        type=int,
        default=1,
        help="Re-place / rebuild the topology every k steps (fit default 1)",
    )
    p.add_argument(
        "--jit-within-epoch",
        action="store_true",
        help=(
            "Compile each epoch's gradient (run_fit's jit_within_epoch). Off by "
            "default, as in run_fit; the record names the path"
        ),
    )
    _common.add_lane_args(p)
    _common.add_cpu_interpret_arg(p)
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


def _failure_kind(exc: BaseException) -> str:
    """Classify a failed point: memory, the mesh force's verdict, or other.

    Parameters
    ----------
    exc : BaseException
        What the point raised.

    Returns
    -------
    str
        ``"out_of_memory"`` (an allocator refusal: the ceiling),
        ``"verdict"`` (the operator refused a placement whose lists overflowed
        or that the reverses cannot read: a capacity setting, not a ceiling),
        or ``"error"``.
    """
    text = f"{type(exc).__name__}: {exc}"
    if "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower():
        return "out_of_memory"
    if "mesh force bits" in text or "overflowed" in text:
        return "verdict"
    return "error"


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
            mac_type=args.mac,
            working_dtype=str(args.dtype),
            devices=list(devices),
            caps=_parse_caps(args.caps) or None,
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
            **_common.operator_kwargs(args),
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
            rebuild_cadence=int(args.rebuild_cadence),
            jit_within_epoch=bool(args.jit_within_epoch),
            regularization=Regularization.none(),
            history_every=1,
            # The intensive churn is O(P) host work per rebuild and this arm is
            # about wall-clock, so it is sampled rather than measured every step.
            intensive_every=max(int(args.iterations) // 4, 1),
            seed=int(args.seed),
            # The mesh operator's partition is per-device trees with no global
            # leaf partition, so the switch instrumentation has nothing to
            # fingerprint.
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
        "lane": operator.record().get("lane"),
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


def _merge_device_count_runs(args: argparse.Namespace) -> int:
    """Concatenate per-device-count result files into one artifact.

    Distributed mode forces one device count per process -- the collective
    rendezvous deadlocks otherwise -- so the single artifact a figure reads has
    to be assembled afterwards. This does that and measures nothing.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line; ``merge_from`` names the inputs and ``json_out``
        the destination.

    Returns
    -------
    int
        Process exit status.

    Raises
    ------
    SystemExit
        If an input is missing, or if the inputs disagree on a configuration
        axis that would make them incomparable. Merging a leaf-256 run with a
        leaf-64 one yields a scaling curve over two different problems, which
        is worse than no curve.
    """
    sources = _merge_sources(args.merge_from)
    records: List[Dict[str, Any]] = []
    ceiling: List[Dict[str, Any]] = []
    merged_config: Optional[Dict[str, Any]] = None
    metas: List[Dict[str, Any]] = []
    pinned = ("theta", "order", "basis", "leaf_size", "softening", "M", "mode", "lane")

    for source in sources:
        path = source if source.startswith("/") else str(jsonio.results_path(source))
        try:
            artifact = jsonio.read_result(path)
        except FileNotFoundError as exc:
            raise SystemExit(f"merge input not found: {path}") from exc
        config = dict(artifact["config"])
        if merged_config is None:
            merged_config = config
        else:
            clashes = {
                key: (merged_config.get(key), config.get(key))
                for key in pinned
                if merged_config.get(key) != config.get(key)
            }
            if clashes:
                raise SystemExit(
                    f"refusing to merge {path}: inputs disagree on {clashes}. "
                    "A scaling curve assembled from different configurations is "
                    "not a scaling curve."
                )
        records.extend(artifact["data"].get("records", []))
        ceiling.extend(artifact["data"].get("ceiling", []))
        metas.append(dict(artifact.get("meta", {})))

    if merged_config is None:
        raise SystemExit("--merge-from named no readable inputs")
    merged_config["device_counts"] = sorted(
        {r["num_devices"] for r in records if "num_devices" in r}
    )
    merged_config["merged_from"] = sources

    out = args.json_out or DEFAULT_OUT
    # Carry the MEASUREMENT provenance forward rather than stamping the merge
    # over it. run_meta() here describes the process doing the stitching, which
    # measured nothing; using it made the provenance table name a commit that
    # did no work -- observed as figures 17, 18 and 19 all reporting the merge
    # commit instead of the runs'. When every slice agrees on a commit that is
    # the artifact's commit; when they disagree the newest is used, because that
    # is when the artifact became complete, and the full set is recorded beside
    # it so the spread is visible rather than hidden.
    merged = runmeta.merged_meta(metas)

    written = jsonio.write_result(
        out,
        config=merged_config,
        meta=merged,
        data={"records": records, "ceiling": ceiling},
    )
    print(f"merged {len(sources)} run(s) -> {written}")
    return 0


def _merge_sources(raw: object) -> List[str]:
    """Return the merge inputs named on the command line, in order.

    Parameters
    ----------
    raw : object
        The ``--merge-from`` value: a list of occurrences, each possibly a
        comma-separated list, or a bare string from an older invocation.

    Returns
    -------
    list of str
        Non-empty paths, duplicates preserved so a repeat is visible.
    """
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    return [
        part.strip()
        for value in values
        for part in str(value).split(",")
        if part.strip()
    ]


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
    if _merge_sources(args.merge_from):
        return _merge_device_count_runs(args)
    device_counts = [int(v) for v in str(args.device_counts).split(",") if v]
    if args.cpu_interpret:
        # Before any jax import: the platform and the device count are read once.
        _common.apply_cpu_interpret(max(device_counts))
    if str(args.mode) == "distributed":
        # The mesh lane's NCCL ragged-exchange flags, which XLA reads only when
        # the backend starts (the library merely warns when they are missing).
        from bench._ragged_exchange_flags import add_ragged_exchange_flags

        add_ragged_exchange_flags()
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

    if args.cpu_interpret:
        _common.patch_cpu_lane()
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
        "basis": "real",
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
        "caps_kind": "CrossCapacities" if str(args.mode) == "distributed" else None,
        "sharding_mode": str(args.mode),
        "mode": str(args.mode),
        "rebuild_cadence": int(args.rebuild_cadence),
        "jit_within_epoch": bool(args.jit_within_epoch),
        "cpu_interpret": bool(args.cpu_interpret),
        # The single-card lane (parameter_sharding); distributed mode is always
        # the fused mesh lane, named in each record's operator block.
        **_common.lane_config(args),
    }
    if str(args.mode) == "distributed":
        config_record["lane"] = "fused_mesh"
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
            kind = _failure_kind(exc)
            print(
                f"  ndev={count}: FAILED [{kind}] ({type(exc).__name__}: {exc})",
                flush=True,
            )
            records.append(
                {
                    "num_devices": count,
                    "N": strong_n,
                    "failed": True,
                    "failure_kind": kind,
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
                kind = _failure_kind(exc)
                note = (
                    "<-- this is the ceiling"
                    if kind == "out_of_memory"
                    else "<-- NOT a memory ceiling: see failure_kind"
                )
                print(
                    f"    ceiling ndev={count} N={n:>10}: FAILED [{kind}] "
                    f"({type(exc).__name__})  {note}",
                    flush=True,
                )
                ceiling.append(
                    {
                        "arm": "ceiling",
                        "num_devices": count,
                        "N": int(n),
                        "num_free_parameters": 3 * int(n),
                        "failed": True,
                        "failure_kind": kind,
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
