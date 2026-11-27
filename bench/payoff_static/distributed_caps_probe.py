"""Which cross capacities keep the mesh fit's verdict ok at the fit's N? (fig20).

Rewritten in November 2026 for the fused mesh lane. The September probe asked
whether the old distributed path's ``DistributedFMMConfig`` caps
(``max_pair_queue`` and friends) set its memory ceiling. They did not, and that
answer is kept at tag ``paper-i-sep2026``. The mesh lane has none of those caps.
Its capacities are ``CrossCapacities``, the buffers of the cross-domain
exchange. They fail **loudly**: an overflow sets ``OVERFLOW_CROSS`` in the mesh
force's bits, and ``DistributedForwardOperator.prepare`` refuses a placement
whose bits are not clean, because its gradient would be NaN. So the September
worry, a too-small cap quietly returning a wrong force, is gone. What remains
is a question that fig20 has to answer before it can run.

**Does the operator's default hold at the fit's N, and if not, which caps do?**
As first ported in step A3, ``make_distributed_forward_operator(caps=None)``
passed ``cross_caps="auto"`` to a fresh ``FusedMeshForce`` at every epoch and
never ran its ``plan_cross``, so the exchange ran at the *measurement*
capacities and, on two forced CPU devices (S9 interpret set, N=16384, M=128,
leaf 64, p3), ``prepare`` refused with bits ``[2560, 2560]``
(``OVERFLOW_CROSS | OVERFLOW_GRAD_LISTS``; 8328 shipped particles per device
against the granule's 1024). Since 2026-11-27 the operator plans "auto" itself,
and this probe's ``auto`` rung passes at that N (CPU smoke). What the probe still
answers: the explicit caps a fig20 TIMING run should pass, so that its prepare
times do not include the per-epoch sizing compiles, and the memory each headroom
costs.

So, per ``N``, on one mesh of ``--device-count`` devices, the probe runs these
rungs:

``auto``
    The operator exactly as the library builds it (``caps=None``).
``planned``
    The library's own sizing: ``FusedMeshForce.plan_cross`` with a measurement
    of the cross needs (the forward ``shard`` under the mesh's ``shard_map``,
    needs max-reduced over the devices), at the fit's start positions. This is
    what the operator would do if it planned its capacities. It is also the
    ``--caps`` string printed for ``multigpu_scaling.py``.
``headroom=h``
    ``CrossCapacities.from_needs`` of the same measured needs at headroom
    ``h``, for each ``--headrooms`` value. ``h=1`` is the smallest set the
    measurement allows, the floor under the question "do the caps set the
    ceiling".

Every rung is checked through the operator itself, at two placements: the fit's
start (``perturbed_truth``, 5 %, as ``multigpu_scaling.py`` starts) and the truth
(where a converging fit ends up). For each, the record holds the verdict bits.
At the start placement it also holds the gradient program's compiled temporary
bytes (``memory_analysis``, one compile reused for the call) and whether the
gradient is finite. A rung whose verdict fails is recorded as such, with
``ran=False``. It is never retried, and it is never counted as a memory result.

Usage
-----
CPU smoke (two forced host devices, S9 interpret set; it compiles for minutes
and measures no timing)::

    python -m bench.payoff_static.distributed_caps_probe --cpu-interpret \\
        --device-count 2 --n 256 --tracers 32 --order 3 --leaf-size 16 \\
        --gpu-select none --json-out /tmp/caps.json

Paper run (one mesh size per process; feed ``data.suggested_caps[*].cli`` to
``multigpu_scaling.py --caps``)::

    python -m bench.payoff_static.distributed_caps_probe --device-count 4 \\
        --n 262144 --json-out density_reconstruction/caps_probe_4.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.payoff_static import _common  # noqa: E402
from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "density_reconstruction/distributed_caps_probe.json"


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
        default="262144",
        help="Comma-separated source counts N (fig20's strong-scaling N first)",
    )
    p.add_argument("--tracers", type=int, default=8192, help="Tracer count M")
    p.add_argument("--device-count", type=int, default=2, help="Devices in the mesh")
    p.add_argument("--order", type=int, default=4, help="Expansion order")
    p.add_argument("--theta", type=float, default=0.5, help="MAC parameter")
    p.add_argument("--leaf-size", type=int, default=64, help="Cell-leaf size")
    p.add_argument("--softening", type=float, default=1.0e-2, help="Softening")
    p.add_argument(
        "--mac",
        default=None,
        help="MAC; default the mesh lane's geometric 'dehnen'",
    )
    p.add_argument(
        "--headrooms",
        default="1.0",
        help=(
            "Comma-separated headrooms over the measured needs, one rung each "
            "(1.0: the smallest caps the measurement allows; the library's "
            "planned rung uses its own 1.5)"
        ),
    )
    _common.add_cpu_interpret_arg(p)
    runmeta.add_common_args(p)
    return p.parse_args()


def _measure_needs(force: Any, placement: Any) -> Any:
    """Return the cross needs of one forward at ``placement``, max over devices.

    The measurement ``plan_cross`` expects: the mesh force's forward ``shard``
    under its ``shard_map`` with the force's CURRENT hook (traced afresh, since
    the hook changes between measurements), its needs vector reduced over the
    mesh and cut to the cross fields.

    Parameters
    ----------
    force : Any
        A prepared ``FusedMeshForce``.
    placement : Any
        The ``rollout.decompose`` shard (positions, masses, count, ids).

    Returns
    -------
    numpy.ndarray
        ``CROSS_NEEDS_FIELDS`` of the forward, host side.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from jaccpot.runtime.fused_topology import NEEDS_FIELDS

    spec = P(force.axis_name)

    def body(x: Any, m: Any, n: Any, i: Any) -> Any:
        out = force.shard(x, m, jnp.minimum(n[0], x.shape[0]), i)
        return out[2].reshape(1, -1)

    program = jax.jit(
        jax.shard_map(
            body,
            mesh=force.mesh,
            in_specs=(spec, spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )
    )
    with force.context():
        force.before_trace()
        needs = program(
            placement.positions, placement.masses, placement.count, placement.ids
        )
    needs = np.asarray(jax.device_get(needs)).max(axis=0)
    return needs[len(NEEDS_FIELDS) :]


def _caps_dict(caps: Any) -> Dict[str, int]:
    """Return a ``CrossCapacities`` as the operator's ``caps`` keywords.

    Parameters
    ----------
    caps : Any
        A ``CrossCapacities``.

    Returns
    -------
    Dict[str, int]
        Every field that is set; ``None`` fields keep the class default.
    """
    return {k: int(v) for k, v in vars(caps).items() if v is not None}


def _plan(operator: Any, positions: Any, *, headrooms: List[float]) -> Dict[str, Any]:
    """Size the cross capacities the way the library would, at ``positions``.

    Places the particles as ``DistributedForwardOperator.prepare`` does (same
    ``decompose`` call, same row capacity), prepares a ``FusedMeshForce`` with
    ``cross_caps="auto"`` and runs its ``plan_cross`` on
    :func:`_measure_needs`. Then derives one capacity set per headroom from the
    measured needs.

    Parameters
    ----------
    operator : Any
        The ``DistributedForwardOperator`` (its solver, mesh and settings).
    positions : Any
        ``(N, 3)`` source positions to plan at.
    headrooms : List[float]
        Headrooms for the ``from_needs`` rungs.

    Returns
    -------
    Dict[str, Any]
        ``planned`` (the library's capacities), ``needs`` (by field), and
        ``by_headroom`` (capacity dicts).

    Raises
    ------
    RuntimeError
        If the force has nothing to size (a one-device mesh).
    """
    import math

    import numpy as np

    from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS, CrossCapacities
    from jaccpot.distributed.rollout import decompose
    from jaccpot.distributed.template_force import FusedMeshForce

    combined = np.asarray(operator.combined_positions(np.asarray(positions)))
    masses = operator.combined_masses()
    total = int(combined.shape[0])
    ndev = int(operator.num_devices)
    cap = int(math.ceil(total / ndev * float(operator.capacity_headroom)))
    shard, _ = decompose(
        operator.mesh,
        combined,
        np.zeros_like(combined),
        masses,
        cap=cap,
        num_samples=int(min(64, max(8, total // (4 * ndev)))),
    )
    force = FusedMeshForce(
        operator.solver,
        operator.mesh,
        leaf_size=int(operator.leaf_size),
        max_order=int(operator.order),
        theta=float(operator.theta),
        cross_caps="auto",
    )
    force.prepare(shard.positions, shard.masses, shard.count)
    if not force.needs_cross_sizing:
        raise RuntimeError("the mesh force has no cross capacities to size")
    force.plan_cross(lambda: _measure_needs(force, shard))
    planned = force.cross_caps
    needs = np.asarray(force.cross_needs_measured)
    leaf_capacity = int(force._setup_or_raise().leaf_capacity)
    by_headroom = {
        str(h): _caps_dict(
            CrossCapacities.from_needs(
                needs,
                like=planned,
                headroom=float(h),
                leaf_capacity=leaf_capacity,
                leaf_size=int(operator.leaf_size),
            )
        )
        for h in headrooms
    }
    return {
        "planned": _caps_dict(planned),
        "needs": {
            name: int(value) for name, value in zip(CROSS_NEEDS_FIELDS, needs.tolist())
        },
        "by_headroom": by_headroom,
    }


def _check(
    *,
    make_operator: Any,
    caps: Optional[Dict[str, int]],
    positions: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run one rung through the operator at each placement.

    Parameters
    ----------
    make_operator : Any
        ``make_operator(caps)`` returns a ``DistributedForwardOperator``.
    caps : Optional[Dict[str, int]]
        The rung's capacities; ``None`` for the library default.
    positions : Dict[str, Any]
        Placement label to ``(N, 3)`` source positions; the gradient is taken at
        the first one.

    Returns
    -------
    List[Dict[str, Any]]
        One outcome per placement.
    """
    import jax
    import jax.numpy as jnp

    outcomes: List[Dict[str, Any]] = []
    for index, (label, x) in enumerate(positions.items()):
        started = time.perf_counter()
        outcome: Dict[str, Any] = {"placement": label}
        try:
            operator = make_operator(caps)
            partition = operator.prepare(x)
            outcome.update(
                ran=True,
                verdict_ok=True,
                bits=partition.diagnostics.get("bits"),
                rows_per_device=partition.diagnostics.get("counts"),
                cap=partition.diagnostics.get("cap"),
            )
            if index == 0:
                sources = jnp.asarray(x)

                def loss(s: Any) -> Any:
                    return jnp.sum(operator.evaluate_at_topology(partition, s) ** 2)

                with operator.program_context(partition):
                    compiled = jax.jit(jax.grad(loss)).lower(sources).compile()
                    memory = compiled.memory_analysis()
                    gradient = jax.block_until_ready(compiled(sources))
                outcome["gradient_finite"] = bool(jnp.all(jnp.isfinite(gradient)))
                outcome["gradient_temp_bytes"] = (
                    None
                    if memory is None
                    else int(getattr(memory, "temp_size_in_bytes", 0))
                )
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            outcome.update(
                ran=False,
                verdict_ok=False if "mesh force bits" in text else None,
                error_type=type(exc).__name__,
                error=str(exc)[:600],
            )
        outcome["seconds"] = time.perf_counter() - started
        outcomes.append(outcome)
    return outcomes


def main() -> int:
    """Run the rungs and write the results JSON.

    Returns
    -------
    int
        Process exit status.
    """
    args = _parse_args()
    if int(args.device_count) < 2:
        raise SystemExit(
            "--device-count must be >= 2: a one-device mesh has no cross "
            "exchange, so there is nothing to size"
        )
    if args.cpu_interpret:
        _common.apply_cpu_interpret(int(args.device_count))
    # The mesh lane's NCCL ragged-exchange flags, which XLA reads only when the
    # backend starts (the library merely warns when they are missing).
    from bench._ragged_exchange_flags import add_ragged_exchange_flags

    add_ragged_exchange_flags()
    runmeta.select_gpu(args.gpu_select, num_gpus=int(args.device_count))
    runmeta.enable_x64(args.dtype)

    import jax

    if args.cpu_interpret:
        _common.patch_cpu_lane()

    from jaccpot.applications.density_reconstruction.distributed import (
        make_distributed_forward_operator,
    )
    from jaccpot.applications.density_reconstruction.parameterize import (
        initial_positions,
    )
    from jaccpot.applications.density_reconstruction.truth import (
        TruthConfig,
        sample_composite,
        sample_tracers,
    )

    devices = jax.devices()
    print(f"distributed_caps_probe: devices {[str(d) for d in devices]}", flush=True)
    n_list = [int(float(v)) for v in str(args.n).split(",") if v.strip()]
    headrooms = [float(v) for v in str(args.headrooms).split(",") if v.strip()]
    ladder = ["auto", "planned"] + [f"headroom={h:g}" for h in headrooms]

    out = args.json_out or DEFAULT_OUT
    config_record = {
        "n": n_list,
        "theta": float(args.theta),
        "order": int(args.order),
        "basis": "real",
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": str(args.dtype),
        "M": int(args.tracers),
        "leaf_size": int(args.leaf_size),
        "softening": float(args.softening),
        "mac_type": args.mac,
        "device_count": int(args.device_count),
        "lane": "fused_mesh",
        "caps_kind": "CrossCapacities",
        "ladder": ladder,
        "headrooms": headrooms,
        "placements": ["start_perturbed_truth_5pct", "truth"],
        "cpu_interpret": bool(args.cpu_interpret),
    }
    attempts: List[Dict[str, Any]] = []
    sizing: List[Dict[str, Any]] = []
    suggested: List[Dict[str, Any]] = []

    def flush() -> Any:
        """Write the results JSON as it currently stands, after every rung.

        Returns
        -------
        Any
            The path written.
        """
        return jsonio.write_result(
            out,
            config=config_record,
            meta=runmeta.run_meta(),
            data={
                "attempts": attempts,
                "sizing": sizing,
                "suggested_caps": suggested,
            },
        )

    for n in n_list:
        truth_config = TruthConfig(
            num_particles=n,
            num_tracers=int(args.tracers),
            seed=int(args.seed),
            softening=float(args.softening),
        )
        truth = sample_composite(truth_config)
        tracers = sample_tracers(truth_config, truth)
        start = initial_positions(
            truth, mode="perturbed_truth", seed=int(args.seed) + 5, perturbation=0.05
        )
        placements = {"start_perturbed_truth_5pct": start, "truth": truth}

        def make_operator(caps: Optional[Dict[str, int]], n: int = n) -> Any:
            return make_distributed_forward_operator(
                tracer_positions=tracers,
                source_mass=float(truth_config.total_mass) / float(n),
                num_sources=n,
                num_devices=int(args.device_count),
                softening=float(args.softening),
                order=int(args.order),
                theta=float(args.theta),
                leaf_size=int(args.leaf_size),
                mac_type=args.mac,
                working_dtype=str(args.dtype),
                devices=list(devices),
                caps=caps,
            )

        rungs: List[Any] = [("auto", None)]
        started = time.perf_counter()
        try:
            plan = _plan(make_operator(None), start, headrooms=headrooms)
            plan_record = {"N": n, **plan, "seconds": time.perf_counter() - started}
            rungs.append(("planned", plan["planned"]))
            rungs.extend(
                (f"headroom={h:g}", plan["by_headroom"][str(h)]) for h in headrooms
            )
            suggested.append(
                {
                    "N": n,
                    "num_devices": int(args.device_count),
                    "caps": plan["planned"],
                    "cli": ",".join(f"{k}={v}" for k, v in plan["planned"].items()),
                }
            )
        except Exception as exc:
            plan_record = {
                "N": n,
                "failed": True,
                "error_type": type(exc).__name__,
                "error": str(exc)[:600],
                "seconds": time.perf_counter() - started,
            }
            print(f"  N={n:>9} sizing FAILED: {type(exc).__name__}: {exc}", flush=True)
        sizing.append(plan_record)
        flush()

        for label, caps in rungs:
            for outcome in _check(
                make_operator=make_operator, caps=caps, positions=placements
            ):
                attempts.append({"N": n, "label": label, "caps": caps, **outcome})
                if outcome.get("ran"):
                    verdict = "OK"
                elif outcome.get("verdict_ok") is False:
                    verdict = "VERDICT FAILED"
                else:
                    verdict = f"FAILED {outcome.get('error_type')}"
                extra = ""
                if outcome.get("gradient_temp_bytes") is not None:
                    extra = (
                        f"  grad temp {outcome['gradient_temp_bytes'] / 2**20:.1f} MiB"
                        f" finite={outcome.get('gradient_finite')}"
                    )
                print(
                    f"  N={n:>9} {label:<14} {outcome['placement']:<28} "
                    f"{verdict}{extra}  {outcome['seconds']:.0f}s",
                    flush=True,
                )
                flush()

    for row in suggested:
        print(f"  suggested --caps for N={row['N']}: {row['cli']}", flush=True)
    print(f"wrote {flush()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
