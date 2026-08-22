"""One multi-GPU force evaluation, measured -- and the sweep driver over it.

Shared by ``strong_scaling.py``, ``weak_scaling.py`` and ``load_balance.py``.

Two structural facts decide this module's shape, and both differ from the
scaffold it replaces.

**A device-count sweep cannot happen inside one process.** JAX initialises its
backend once, and the device count is fixed for the lifetime of the process, so
``[run_once(n, g) for g in gpu_counts]`` -- which is what the scaffold did --
cannot work however it is implemented. Every point is therefore its own process.
:func:`sweep` spawns one, and this module is runnable as that worker
(``python -m bench.multigpu.harness --ndev 4 --n 32000``), which is also what
makes the sweep portable to a batch scheduler: the same command line is an array
job, one task per point.

**Running one evaluation per process is required anyway.** The ``jit=True``
illegal-address fault documented in ``docs/phase5_multigpu_pallas_foldin_plan.md``
is intermittent and nondeterministic. Isolating points means a fault kills one
measurement rather than the sweep, and cannot silently corrupt the ones after it.

What is measured
----------------
Steady-state device time for a *built* evaluator: ``make_force_evaluator`` once,
warm up, then time repeated calls and report the median. This is deliberately not
``distributed_fmm_accelerations``, which rebuilds and recompiles per call (50-80 s)
and would report compile time dressed up as force time.

An overflowing point is **invalid, not slow**. When a traversal buffer overflows
the forces are truncated, so the wall clock is padding overhead over a wrong
answer. Above roughly 8000 particles per device the caps overflow even after the
maximum number of retries. Such a point is returned with ``valid=False`` and the
offending counters attached; callers must drop it rather than plot it.

What is NOT measured, and why
-----------------------------
**Per-stage timings do not exist on this path.** The driver's per-device
diagnostic vector (``DIAG_FIELDS``) carries interaction counts and overflow flags
and no timers at all. The single-device breakdown of figure 06 comes from
``_refresh_timing_*`` counters that live on the strict refresh path with fusion
disabled -- "fusing the stages is exactly what makes them unmeasurable" -- and
the distributed ``shard_map`` pipeline has no equivalent instrumentation.

So the comm/compute split (figure 10) has no mechanism behind it yet, and this
module does not pretend otherwise: it exposes no ``STAGE_NAMES`` and no
``COMM_STAGES``. The scaffold defined both, naming eight stages that nothing in
the code emits; anything reading them would have been reporting invented
structure. Getting figure 10 needs real instrumentation inside the sharded
region -- a host callback per stage, or attributing a profiler trace's XLA ops to
stages -- which is library work, not analysis, and belongs in its own change.

What *is* available per device is the interaction census, which is what figure 11
(load balance) actually needs: how much pair work each device was handed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import runmeta  # noqa: E402

# Per-device pair counts. These are the load-balance signal: the P2P cost tracks
# near pairs and the far-field cost tracks far pairs, so an imbalance in these is
# an imbalance in work, which an equal-particle-count partition would not show.
WORK_FIELDS = (
    "self_near_pairs",
    "self_far_pairs",
    "cross_near_pairs",
    "cross_far_pairs",
)

# A nonzero value in any of these means the forces were truncated.
OVERFLOW_FIELDS = (
    "self_queue_overflow",
    "self_near_overflow",
    "self_far_overflow",
    "cross_queue_overflow",
    "cross_near_overflow",
    "cross_far_overflow",
)


def make_distribution(name: str, n: int, ndev: int, seed: int) -> tuple[Any, Any]:
    """Return ``(positions, masses)`` for the named distribution.

    Parameters
    ----------
    name : str
        ``uniform`` (a cube) or ``plummer`` (centrally concentrated).
    n : int
        Total particle count.
    ndev : int
        Device count; ``plummer`` is scaled so the domains are comparably filled.
    seed : int
        RNG seed.

    Returns
    -------
    tuple
        ``(positions[n, 3], masses[n])`` as float64 NumPy arrays.
    """

    import numpy as np

    rng = np.random.default_rng(seed)
    if name == "uniform":
        return rng.uniform(-1.0, 1.0, size=(n, 3)), rng.uniform(0.5, 1.5, size=n)
    if name == "plummer":
        # Clustered on purpose: a space-filling-curve split of a uniform cube is
        # balanced by construction, so a uniform distribution cannot show whether
        # the partition balances *work*. Plummer concentrates particles centrally,
        # which is where equal-particle-count partitioning and equal-work
        # partitioning come apart.
        u = rng.uniform(size=n)
        radius = np.minimum(
            u ** (1.0 / 3.0) / np.sqrt(np.maximum(1.0 - u ** (2.0 / 3.0), 1e-12)), 20.0
        )
        cos_t = rng.uniform(-1.0, 1.0, size=n)
        sin_t = np.sqrt(np.maximum(1.0 - cos_t**2, 0.0))
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
        pos = np.stack(
            [
                radius * sin_t * np.cos(phi),
                radius * sin_t * np.sin(phi),
                radius * cos_t,
            ],
            axis=1,
        )
        return pos, np.full(n, 1.0 / n)
    raise SystemExit(f"unknown --distribution {name!r} (use uniform or plummer)")


def measure_point(
    *,
    ndev: int,
    n: int,
    order: int = 3,
    theta: float = 0.4,
    leaf_size: int = 128,
    basis: str = "real",
    mac_type: str = "dehnen",
    nearfield_backend: str = "auto",
    distribution: str = "uniform",
    seed: int = 0,
    repeats: int = 5,
    warmup: int = 2,
    auto_scale_caps: bool = True,
    max_cap_retries: int = 4,
    max_pair_queue: Optional[int] = None,
    cross_max_pair_queue: Optional[int] = None,
    max_interactions_per_node: Optional[int] = None,
    max_neighbors_per_leaf: Optional[int] = None,
    process_block: Optional[int] = None,
    l2l_num_levels: Optional[int] = None,
    accuracy_targets: int = 0,
    cross_far_cap: Optional[int] = None,
    cross_max_interactions_per_node: Optional[int] = None,
) -> dict[str, Any]:
    """Measure one ``(ndev, n)`` point in this process.

    Requires ``ndev`` visible devices; the caller is responsible for having
    selected them before JAX was imported.

    Parameters
    ----------
    ndev : int
        Device count, which must match what JAX sees.
    n : int
        Total particle count across all devices.
    order : int
        Multipole order.
    theta : float
        Local self MAC opening angle.
    leaf_size : int
        Leaf occupancy.
    basis : str
        Far-field expansion basis.
    mac_type : str
        Acceptance criterion.
    nearfield_backend : str
        ``auto``, ``pallas`` or ``baseline``.
    distribution : str
        See :func:`make_distribution`.
    seed : int
        RNG seed.
    repeats : int
        Timed calls after warmup; the median is reported.
    warmup : int
        Untimed calls, to pay compilation and let the allocator settle.
    auto_scale_caps : bool
        Grow an overflowing traversal buffer and retry.
    max_cap_retries : int
        Retry ceiling for the above. Each retry rebuilds *and recompiles*, so a
        long ladder is expensive; prefer starting from a cap that fits.
    max_pair_queue : int, optional
        Starting self pair-queue capacity, overriding the config default. The
        default (32768) is a fixed constant with no particle-count dependence,
        unlike the single-device presets, which is why a large per-device load
        overflows it. Setting it directly is cheaper than growing into it by
        recompiling.
    cross_max_pair_queue : int, optional
        As above, for the cross-domain walk.
    max_interactions_per_node : int, optional
        Starting far-list capacity per node.
    max_neighbors_per_leaf : int, optional
        Starting near-list capacity per leaf.
    accuracy_targets : int
        When > 0, compare the force against a direct-sum reference on this many
        randomly chosen target particles. The reference sums over **all** sources,
        so it is exact; only the target set is subsampled, which is what makes it
        affordable. 0 disables it.

        This exists because a speed measurement is worthless for any knob that
        trades accuracy -- notably ``theta``. Without it a
        criterion sweep can only report that a looser criterion is faster, which
        is true by construction and says nothing.
    l2l_num_levels : int, optional
        Static level bound for the L2L cascade. The default derives
        ``num_internal - 1`` from a shape because the tree depth is not knowable
        inside ``shard_map``; a balanced tree is only ~log2(num_leaves) deep, so
        the default can be orders of magnitude loose and the extra levels cost
        time while contributing exactly zero.
    process_block : int, optional
        Dual-tree walk process block. Its default (64) coincides exactly with the
        measured ceiling of 64 leaves per device, which is why it is exposed.

    Returns
    -------
    dict
        The measurement record. ``valid`` is False when a buffer overflowed,
        in which case the timings describe a truncated force and must be dropped.
    """

    import dataclasses
    import time

    import jax
    import jax.numpy as jnp
    import numpy as np
    from yggdrax.distributed import device_count, make_mesh

    from jaccpot.distributed import DistributedFMMConfig
    from jaccpot.distributed.fmm import (
        DIAG_FIELDS,
        make_force_evaluator,
        partition_for_devices,
    )

    if device_count() < ndev:
        raise SystemExit(
            f"need {ndev} devices, JAX sees {device_count()}. Refusing to measure: "
            "a scaling point taken on a different device count is not the point "
            "that was asked for."
        )

    config = dataclasses.replace(
        DistributedFMMConfig(),
        order=order,
        theta=theta,
        leaf_size=leaf_size,
        basis=basis,
        mac_type=mac_type,
        nearfield_backend=nearfield_backend,
    )
    overrides = {
        k: int(v)
        for k, v in (
            ("max_pair_queue", max_pair_queue),
            ("cross_max_pair_queue", cross_max_pair_queue),
            ("max_interactions_per_node", max_interactions_per_node),
            ("max_neighbors_per_leaf", max_neighbors_per_leaf),
            ("process_block", process_block),
            ("cross_far_cap", cross_far_cap),
            ("cross_max_interactions_per_node", cross_max_interactions_per_node),
        )
        if v is not None
    }
    if overrides:
        config = dataclasses.replace(config, **overrides)
    positions, masses = make_distribution(distribution, n, ndev, seed)
    part = partition_for_devices(positions, masses, ndev, leaf_size=config.leaf_size)
    mesh = make_mesh(ndev)
    args = (
        jnp.asarray(part["pos_flat"]),
        jnp.asarray(part["mass_flat"]),
        jnp.asarray(part["gid_flat"]),
        jnp.asarray(part["counts"]),
    )

    # Build once, then grow caps only if a buffer actually overflowed. Rebuilding
    # per call would report compile time as force time.
    attempt = 0
    while True:
        evaluate = make_force_evaluator(
            config,
            ndev,
            part["cap"],
            mesh,
            jit=True,
            l2l_num_levels=l2l_num_levels,
        )
        accel, _gid, diag = evaluate(*args)
        jax.block_until_ready(accel)
        diag_np = np.asarray(diag)
        counters = {f: diag_np[:, i].tolist() for i, f in enumerate(DIAG_FIELDS)}
        overflowed = [f for f in OVERFLOW_FIELDS if any(counters.get(f, []))]
        if not overflowed or not auto_scale_caps or attempt >= max_cap_retries:
            break
        config = config.with_selective_scaled_caps(diag_np, 2.0)
        attempt += 1

    for _ in range(max(0, warmup)):
        jax.block_until_ready(evaluate(*args)[0])

    times: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        jax.block_until_ready(evaluate(*args)[0])
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]

    accuracy: dict[str, Any] = {}
    if int(accuracy_targets) > 0:
        accel_out, gid_out, _ = evaluate(*args)
        accel_np = np.asarray(accel_out)
        # The gid the evaluator RETURNS, not the ``gid_flat`` in ``args[2]`` that went
        # in. The accelerations come back in the per-device TREE order, and once a
        # device is padded that is not the partition order the input gid names:
        # ``partition_for_devices`` puts its padding rows at the device's
        # Morton-minimum particle, so the tree build sorts them to the front and
        # displaces every real particle after the first by the padding count. Scoring
        # output row r against input particle r then compares each particle's force to
        # a Morton neighbour's -- smooth, plausible, and wrong by tens of percent with
        # every overflow counter clean. That, and not the solver, is the whole of
        # docs/distributed_padding_force_defect.md. Unpadded runs are unaffected: the
        # two arrays are identical whenever cap == count, so previously reported
        # accuracy numbers on leaf-multiple per-device counts do not move.
        gid_np = np.asarray(gid_out).reshape(-1)
        valid_rows = np.flatnonzero(gid_np >= 0)
        k = min(int(accuracy_targets), valid_rows.size)
        pick = np.random.default_rng(seed).choice(valid_rows, size=k, replace=False)
        tgt_ids = gid_np[pick].astype(np.int64)

        src_pos = jnp.asarray(positions)
        src_mass = jnp.asarray(masses)
        soft_sq = jnp.asarray(config.softening**2, dtype=src_pos.dtype)
        g_const = jnp.asarray(config.G, dtype=src_pos.dtype)

        def direct_block(tid: Any) -> Any:
            """Exact acceleration on the targets in ``tid``, summing all sources."""
            tp = src_pos[tid]
            delta = src_pos[None, :, :] - tp[:, None, :]
            dsq = jnp.sum(delta * delta, axis=2) + soft_sq
            # self-interaction excluded by index, not by distance
            same = jnp.arange(src_pos.shape[0])[None, :] == tid[:, None]
            inv = jnp.where(same, 0.0, dsq ** (-1.5))
            return g_const * jnp.einsum("ij,j,ijk->ik", inv, src_mass, delta)

        # chunk the targets so the [k, N, 3] intermediate never materialises whole
        block = 256
        ref_parts = [
            np.asarray(direct_block(jnp.asarray(tgt_ids[i : i + block])))
            for i in range(0, k, block)
        ]
        ref = np.concatenate(ref_parts, axis=0)
        got = accel_np[pick]
        denom = float(np.linalg.norm(ref))
        accuracy = {
            "targets": int(k),
            "rel_l2_vs_direct": float(np.linalg.norm(got - ref) / (denom + 1e-300)),
            "max_abs_err": float(np.abs(got - ref).max()),
            "ref_rms": float(np.sqrt(np.mean(np.sum(ref * ref, axis=1)))),
        }

    per_device_work = {f: counters[f] for f in WORK_FIELDS if f in counters}
    total_work = [sum(vals) for vals in zip(*per_device_work.values())] or [0] * ndev

    return {
        "ndev": ndev,
        "n": int(part["n"]),
        "per_device_n": int(-(-int(part["n"]) // ndev)),
        "cap": int(part["cap"]),
        "cap_retries": attempt,
        "distribution": distribution,
        "seed": seed,
        "median_s": median,
        "min_s": times[0],
        "max_s": times[-1],
        "repeats": len(times),
        "throughput_particles_per_s": (int(part["n"]) / median) if median > 0 else None,
        # Per-device pair counts, and their per-device total: the load-balance signal.
        "per_device_work": per_device_work,
        "per_device_work_total": total_work,
        "work_imbalance": (
            (max(total_work) / (sum(total_work) / len(total_work)))
            if total_work and sum(total_work) > 0
            else None
        ),
        "accuracy": accuracy or None,
        "overflowed": overflowed,
        # A truncated force is not a slow force. Drop these points.
        "valid": not overflowed,
        "config": {
            "order": order,
            "theta": theta,
            "leaf_size": leaf_size,
            "basis": basis,
            "mac_type": mac_type,
            "nearfield_backend": nearfield_backend,
            # The caps the final (possibly retried) evaluation actually ran with.
            # A per-device load means nothing without these and leaf_size.
            "max_pair_queue": int(config.max_pair_queue),
            "cross_max_pair_queue": int(config.cross_max_pair_queue),
            "max_interactions_per_node": int(config.max_interactions_per_node),
            "max_neighbors_per_leaf": int(config.max_neighbors_per_leaf),
            "process_block": int(config.process_block),
            "cross_far_cap": config.cross_far_cap,
            "cross_max_interactions_per_node": int(
                config.cross_max_interactions_per_node
            ),
            "l2l_num_levels": l2l_num_levels,
        },
    }


def sweep(
    points: list[dict[str, Any]], *, extra_argv: Optional[list[str]] = None
) -> list[dict[str, Any]]:
    """Run each ``(ndev, n)`` point in a fresh process and collect the records.

    Parameters
    ----------
    points : list
        Dicts carrying at least ``ndev`` and ``n``.
    extra_argv : list, optional
        Further flags forwarded verbatim to every worker.

    Returns
    -------
    list
        One record per point, in input order. A point whose worker failed is
        recorded with ``valid=False`` and the worker's stderr, so a partial sweep
        is visible in the artifact instead of looking complete.
    """

    out: list[dict[str, Any]] = []
    for p in points:
        cmd = [
            sys.executable,
            "-m",
            "bench.multigpu.harness",
            "--ndev",
            str(p["ndev"]),
            "--n",
            str(p["n"]),
            "--emit-json",
        ] + list(extra_argv or [])
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        payload = None
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = None
                break
        if payload is None:
            print(
                f"[harness] point ndev={p['ndev']} n={p['n']} FAILED "
                f"(exit {proc.returncode})",
                file=sys.stderr,
            )
            payload = {
                "ndev": p["ndev"],
                "n": p["n"],
                "valid": False,
                "error": (proc.stderr or "").strip()[-2000:],
                "returncode": proc.returncode,
            }
        else:
            flag = "" if payload.get("valid") else "  [INVALID: overflow]"
            print(
                f"[harness] ndev={payload['ndev']} n={payload['n']} "
                f"median={payload['median_s']*1e3:.1f}ms{flag}"
            )
        out.append(payload)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure one multi-GPU point.")
    runmeta.add_common_args(p)
    p.add_argument("--ndev", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--theta", type=float, default=0.4)
    p.add_argument("--leaf-size", type=int, default=128)
    p.add_argument("--basis", default="real")
    p.add_argument("--mac-type", default="dehnen")
    p.add_argument("--nearfield-backend", default="auto")
    p.add_argument("--distribution", default="uniform")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max-cap-retries", type=int, default=4)
    p.add_argument("--max-pair-queue", type=int, default=None)
    p.add_argument("--cross-max-pair-queue", type=int, default=None)
    p.add_argument("--max-interactions-per-node", type=int, default=None)
    p.add_argument("--max-neighbors-per-leaf", type=int, default=None)
    p.add_argument("--process-block", type=int, default=None)
    p.add_argument("--cross-far-cap", type=int, default=None)
    p.add_argument("--cross-max-interactions-per-node", type=int, default=None)
    p.add_argument("--l2l-num-levels", type=int, default=None)
    p.add_argument(
        "--accuracy-targets",
        type=int,
        default=0,
        help=(
            "compare against a direct sum on this many sampled targets (0 = off). "
            "Required for any theta sweep: speed alone cannot judge a "
            "criterion that trades accuracy"
        ),
    )
    p.add_argument(
        "--emit-json",
        action="store_true",
        help="print the record as one JSON line on stdout (used by sweep())",
    )
    return p.parse_args()


def main() -> int:
    """Single-point worker entry point.

    Returns
    -------
    int
        0 on success, 1 when the point overflowed and is therefore invalid.
    """

    args = _parse_args()
    runmeta.select_gpu(args.gpu_select, num_gpus=int(args.ndev))
    runmeta.enable_x64(args.dtype)

    record = measure_point(
        ndev=int(args.ndev),
        n=int(args.n),
        order=int(args.order),
        theta=float(args.theta),
        leaf_size=int(args.leaf_size),
        basis=args.basis,
        mac_type=args.mac_type,
        nearfield_backend=args.nearfield_backend,
        distribution=args.distribution,
        seed=int(args.seed),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        max_cap_retries=int(args.max_cap_retries),
        max_pair_queue=args.max_pair_queue,
        cross_max_pair_queue=args.cross_max_pair_queue,
        max_interactions_per_node=args.max_interactions_per_node,
        max_neighbors_per_leaf=args.max_neighbors_per_leaf,
        process_block=args.process_block,
        l2l_num_levels=args.l2l_num_levels,
        accuracy_targets=int(args.accuracy_targets),
        cross_far_cap=args.cross_far_cap,
        cross_max_interactions_per_node=args.cross_max_interactions_per_node,
    )
    record["meta"] = runmeta.run_meta({"argv": sys.argv[1:]})
    if args.emit_json:
        print(json.dumps(record))
    else:
        print(json.dumps(record, indent=2))
    return 0 if record["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
