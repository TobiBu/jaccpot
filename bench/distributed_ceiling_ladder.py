"""A4 -- climb the per-device ladder on a disc, with checks that cannot lie.

``docs/plan_2026-08_A_ceiling.md``'s A4: per-device N from 8 192 upward, at ndev 2
and 4, on a **galaxy disc** rather than separated clusters -- clusters make the
cross-domain term negligible and so exercise nothing. Two independent validity
checks at every point, both **before** any wall clock is read, because on this
lane the failure mode reads as a speedup: a truncated walk does less work, so it
looks faster, and the only witness is a diagnostic counter nobody plots.

1. ``self_near_pairs`` must grow monotonically with load, and every overflow flag
   must be clear. A drop is a truncation and its timing is meaningless.
2. A subsampled exact fp64 direct sum must agree to a few parts in 10^3. Sampled
   because the full N^2 is unaffordable at 10^6, fp64 because the reference must
   never be the noisy side of the comparison.

Timing uses ``make_force_evaluator(..., jit=True)`` called repeatedly after a
warm-up. Do **not** use ``distributed_fmm_accelerations`` here: it rebuilds and
recompiles on every call (~200 s per call at moderate N), so its wall-clock column
measures compilation. That is why the audit discarded its timings.

``auto_scale_caps`` is deliberately unavailable: the point of the ladder is what
the *default* config reaches, and a retry ladder turns a wall into a 15-minute
timeout (each retry is a fresh compile) rather than an answer either way.

WHAT IT REACHED, 2xA100-PCIE-40GB, jax 0.10.2, x64, disc IC, order 3, theta=0.4,
dehnen MAC, derived caps (no manual tuning, no ``auto_scale_caps`` retries):

    leaf  N/device   self_near_pairs   rel_l2    s/force   note
      64      8192            29 942  6.6e-05     0.070
      64     32768           351 246  5.2e-04     0.112
      64    131072         3 186 626  1.0e-03     0.600
     256     32768            29 378  1.6e-04     0.110
     256    131072           357 946  4.1e-04     0.196
     256    524288         3 232 328  1.3e-03     1.112
     256   1048576         9 522 178  1.8e-03     4.174   m2l_chunk + nearfield_chunk

10^6 particles per device -- 2 097 152 on two cards -- with `self_near_pairs`
monotone throughout and no overflow flag set anywhere. Before this track a card
carried ~10^4 in a mesh.

THE WALLS, IN THE ORDER THEY APPEAR, all named rather than tuned around:

1. The traced wavefront came from `process_block`, not `max_pair_queue`
   (yggdrax `fix/traced-wavefront-capacity`). Fixing it is what makes the rest
   measurable, because until then growing a cap did nothing.
2. `max_neighbors_per_leaf=128` truncates above 128 leaves.
3. `cross_max_neighbors_per_leaf`, then `cross_max_pair_queue`, immediately
   behind it -- 49 % force error with every self flag clear.
4. The self wavefront: `num_leaves ** 1.5`, so a linear rule under-sizes further
   the bigger the problem gets.
5. **The full-batch far-field M2L.** This is the one that actually stops the
   ladder, and it is worth naming exactly because the plan lists three candidates
   and it is none of the two obvious ones. At 524288/device and leaf 64 the run
   dies autotuning `_m2l_real_batch_kernel` on `f32[469733376, 7]` -- 12.25 GiB in
   a single allocation -- *with* `nearfield_chunk` already set, so it is not the
   near-field densification. `cross_far_cap` scales as
   `cross_max_interactions_per_node * num_target_leaves`, so deriving that cap
   upward is what enlarges it: the fix for one wall built the next one.
   `m2l_chunk` is the lever, and it is why the 10^6 row sets it. Recorded in
   `results/distributed_ceiling/ladder_ndev2_leaf64_524k_m2l_wall.json`.
6. The near-field densification `[u_leaves, S_near]` is the plan's other named
   candidate and is a genuine `num_leaves ** 2` cost, but nothing here caught it
   binding: `nearfield_chunk` was set on the only run that got far enough to ask.
   Listed as unmeasured rather than as a wall.

Both levers are numerics-identical by construction, and both bound how much of a
device's work fits in one batch rather than how much a device can hold -- so the
remaining ceiling is throughput, not capacity.

USAGE
-----
    export CUDA_VISIBLE_DEVICES=$(autocvd -n 2 -l -q -o)
    python -u -m bench.distributed_ceiling_ladder --ndev 2 --leaf 64
    python -u -m bench.distributed_ceiling_ladder --ndev 4 --leaf 256 \
        --sizes 131072,524288,1048576

Results land in ``bench/results/distributed_ceiling/`` as JSON. One process per
run: the ``jit=True`` illegal-address crash on this lane is an intermittent
nondeterministic OOB, so a crash costs one point rather than the sweep.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import time
import traceback
from typing import Any, Optional

import jax
import numpy as np

_RESULTS = pathlib.Path(__file__).resolve().parent / "results" / "distributed_ceiling"

#: Per-device particle counts, doubling from the audit's healthy regime to the
#: track's target.
_SIZES = (8192, 32768, 131072, 524288, 1048576)

#: Source-chunk width for the fp64 oracle, so the N x probe difference array stays
#: bounded regardless of N.
_ORACLE_CHUNK = 8192


def _disc(
    n: int, radius: float = 10.0, thickness: float = 0.2, seed: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    """A thin exponential-ish disc, in float32 as the production lane runs it."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    pos = np.stack(
        [r * np.cos(th), r * np.sin(th), rng.normal(scale=thickness, size=n)], axis=1
    )
    return pos.astype(np.float32), rng.uniform(0.8, 1.2, n).astype(np.float32)


def _oracle_subset(
    positions: np.ndarray,
    masses: np.ndarray,
    targets: np.ndarray,
    *,
    G: float,
    softening: float,
) -> np.ndarray:
    """Exact softened fp64 acceleration on ``targets`` from every particle.

    Chunked over sources so the temporary is ``[chunk, len(targets), 3]`` rather
    than ``[N, len(targets), 3]``: at N = 2 x 10^6 the unchunked form is 100 GB.

    Parameters
    ----------
    positions : np.ndarray
        All particle positions ``[N, 3]``, input order.
    masses : np.ndarray
        All particle masses ``[N]``.
    targets : np.ndarray
        Indices of the particles to evaluate.
    G : float
        Gravitational constant.
    softening : float
        Plummer softening length.

    Returns
    -------
    np.ndarray
        Accelerations ``[len(targets), 3]`` in float64.
    """
    pos = np.asarray(positions, np.float64)
    mass = np.asarray(masses, np.float64)
    tgt = pos[targets]
    soft2 = np.float64(softening) ** 2
    out = np.zeros((tgt.shape[0], 3), np.float64)
    for start in range(0, pos.shape[0], _ORACLE_CHUNK):
        stop = min(start + _ORACLE_CHUNK, pos.shape[0])
        diff = tgt[:, None, :] - pos[None, start:stop, :]
        d2 = (diff**2).sum(-1) + soft2
        inv = d2 ** (-1.5)
        # Self-interaction: a target inside this source chunk sees itself at
        # distance 0, which softening makes finite rather than infinite -- so it
        # has to be masked rather than left to cancel.
        rows = np.nonzero((targets >= start) & (targets < stop))[0]
        inv[rows, targets[rows] - start] = 0.0
        out -= np.float64(G) * (
            mass[None, start:stop, None] * diff * inv[..., None]
        ).sum(axis=1)
    return out


def _one_point(
    *,
    per_device_n: int,
    ndev: int,
    args: argparse.Namespace,
) -> dict:
    """Evaluate, validate, then time one rung of the ladder."""
    import dataclasses

    from jax.sharding import Mesh

    from jaccpot.distributed.fmm import (
        DIAG_FIELDS,
        DistributedFMMConfig,
        make_force_evaluator,
        partition_for_devices,
        scatter_to_input_order,
    )

    overrides: dict[str, Any] = dict(
        leaf_size=args.leaf,
        theta=args.theta,
        order=args.order,
        mac_type=args.mac_type,
        nearfield_backend=args.nearfield_backend,
    )
    if args.m2l_chunk:
        overrides["m2l_chunk"] = args.m2l_chunk
    if args.nearfield_chunk:
        overrides["nearfield_chunk"] = args.nearfield_chunk
    # The cross-walk caps, overridable so the rung that walls on one of them can be
    # bisected without editing the config's defaults.
    for name, value in (
        ("cross_max_neighbors_per_leaf", args.cross_neighbors),
        ("cross_max_interactions_per_node", args.cross_interactions),
        ("cross_max_pair_queue", args.cross_queue),
    ):
        if value:
            overrides[name] = value
    config = dataclasses.replace(DistributedFMMConfig(), **overrides)

    positions, masses = _disc(per_device_n * ndev, seed=args.seed)
    part = partition_for_devices(
        positions, masses, ndev, leaf_size=args.leaf, partitioner=config.partitioner
    )
    cap = int(part["cap"])
    resolved = config.resolved_for(cap)
    mesh = Mesh(np.asarray(jax.devices()[:ndev]).reshape(ndev), ("gpus",))

    row: dict[str, Any] = {
        "per_device_n": per_device_n,
        "ndev": ndev,
        "n_total": int(part["n"]),
        "cap": cap,
        "leaf_size": args.leaf,
        "caps": {
            "process_block": resolved.process_block,
            "max_pair_queue": resolved.max_pair_queue,
            "max_interactions_per_node": resolved.max_interactions_per_node,
            "max_neighbors_per_leaf": resolved.max_neighbors_per_leaf,
            "cross_max_neighbors_per_leaf": resolved.cross_max_neighbors_per_leaf,
            "cross_max_interactions_per_node": resolved.cross_max_interactions_per_node,
            "cross_max_pair_queue": resolved.cross_max_pair_queue,
        },
    }

    t0 = time.perf_counter()
    evaluate = make_force_evaluator(config, ndev, cap, mesh, jit=True)
    accel_o, gid_o, diag_o = evaluate(
        part["pos_flat"], part["mass_flat"], part["gid_flat"], part["counts"]
    )
    jax.block_until_ready(accel_o)
    row["compile_and_first_call_s"] = round(time.perf_counter() - t0, 2)

    diag = np.asarray(diag_o)
    for i, name in enumerate(DIAG_FIELDS):
        row[name] = [float(v) for v in diag[:, i]]
    row["self_near_pairs_total"] = float(sum(row["self_near_pairs"]))
    row["overflow"] = {
        name: float(sum(row[name])) for name in DIAG_FIELDS if name.endswith("overflow")
    }
    row["any_overflow"] = any(v > 0 for v in row["overflow"].values())

    # Validity check 2, still before any timing.
    accel = scatter_to_input_order(accel_o, gid_o, int(part["n"]))
    rng = np.random.default_rng(args.seed + 1)
    probe = min(args.probe, int(part["n"]))
    targets = np.sort(rng.choice(int(part["n"]), size=probe, replace=False))
    t0 = time.perf_counter()
    exact = _oracle_subset(
        positions, masses, targets, G=config.G, softening=config.softening
    )
    got = np.asarray(accel)[targets].astype(np.float64)
    row["oracle_s"] = round(time.perf_counter() - t0, 2)
    row["probe"] = probe
    row["rel_l2"] = float(
        np.linalg.norm(got - exact) / (np.linalg.norm(exact) + 1e-300)
    )

    if row["any_overflow"] or row["rel_l2"] > args.max_rel_l2:
        row["timing_skipped"] = (
            "overflow: "
            + ",".join(n for n, c in sorted(row["overflow"].items()) if c > 0)
            if row["any_overflow"]
            else f"rel_l2 {row['rel_l2']:.4g}"
        )
        return row

    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        out = evaluate(
            part["pos_flat"], part["mass_flat"], part["gid_flat"], part["counts"]
        )
        jax.block_until_ready(out[0])
        times.append(time.perf_counter() - t0)
    row["seconds_median"] = round(float(np.median(times)), 4)
    row["seconds_min"] = round(float(np.min(times)), 4)
    row["seconds_max"] = round(float(np.max(times)), 4)
    return row


def _line(row: dict) -> str:
    caps = row["caps"]
    verdict = row.get("timing_skipped") or f"{row.get('seconds_median')}s"
    # Name the buffer, not just the fact. "overflow" sends the reader to tune every
    # cap; "self_queue" sends them to the one that ran out.
    fired = ",".join(
        name.replace("_overflow", "")
        for name, count in sorted(row["overflow"].items())
        if count > 0
    )
    return (
        f"N/dev={row['per_device_n']:8d} ndev={row['ndev']} leaf={row['leaf_size']:4d} | "
        f"queue={caps['max_pair_queue']:8d} nbr={caps['max_neighbors_per_leaf']:6d} "
        f"far/node={caps['max_interactions_per_node']:6d} "
        f"xnbr={caps['cross_max_neighbors_per_leaf']:6d} | "
        f"self_near={row['self_near_pairs_total']:13.0f} "
        f"ovf={fired or '-'} "
        f"rel_l2={row['rel_l2']:.3e} | {verdict} "
        f"(build {row['compile_and_first_call_s']}s)"
    )


def _monotonicity(rows: list[dict]) -> list[str]:
    """Report every rung where ``self_near_pairs`` failed to grow with the load."""
    problems = []
    previous: Optional[dict] = None
    for row in rows:
        if (
            previous is not None
            and row["self_near_pairs_total"] < previous["self_near_pairs_total"]
        ):
            problems.append(
                f"self_near_pairs fell from {previous['self_near_pairs_total']:.0f} at "
                f"{previous['per_device_n']}/device to {row['self_near_pairs_total']:.0f} "
                f"at {row['per_device_n']}/device -- the walk truncated"
            )
        previous = row
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    """Climb the ladder and write the result to ``bench/results/distributed_ceiling``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ndev", type=int, default=2)
    ap.add_argument(
        "--sizes",
        default=",".join(str(n) for n in _SIZES),
        help="comma-separated per-device particle counts",
    )
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--theta", type=float, default=0.4)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--mac-type", default="dehnen")
    ap.add_argument("--nearfield-backend", default="auto")
    ap.add_argument("--cross-neighbors", type=int, default=0)
    ap.add_argument("--cross-interactions", type=int, default=0)
    ap.add_argument("--cross-queue", type=int, default=0)
    ap.add_argument("--m2l-chunk", type=int, default=0)
    ap.add_argument("--nearfield-chunk", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--probe", type=int, default=512)
    ap.add_argument(
        "--max-rel-l2",
        type=float,
        default=5e-3,
        help="a few parts in 1e3, per the plan; above this the timing is not read",
    )
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    devices = jax.devices()
    if len(devices) < args.ndev:
        raise SystemExit(f"need {args.ndev} devices, saw {len(devices)}")
    print(
        f"# ndev={args.ndev} leaf={args.leaf} order={args.order} theta={args.theta} "
        f"mac={args.mac_type} nearfield={args.nearfield_backend} "
        f"devices={[d.device_kind for d in devices[: args.ndev]]}",
        flush=True,
    )

    rows: list[dict] = []
    for per_device_n in (int(x) for x in args.sizes.split(",") if x):
        try:
            row = _one_point(per_device_n=per_device_n, ndev=args.ndev, args=args)
        except Exception as exc:  # noqa: BLE001 -- a wall is a result, not a crash
            # Name the wall rather than tuning around it: the exception text is
            # what says which buffer ran out, and losing it to a traceback on
            # stderr is how a ceiling becomes folklore.
            rows.append(
                {
                    "per_device_n": per_device_n,
                    "ndev": args.ndev,
                    "leaf_size": args.leaf,
                    "failed": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=6),
                }
            )
            print(
                f"N/dev={per_device_n:8d} FAILED {type(exc).__name__}: {exc}",
                flush=True,
            )
            break
        rows.append(row)
        print(_line(row), flush=True)

    valid = [r for r in rows if "self_near_pairs_total" in r]
    problems = _monotonicity(valid)
    for problem in problems:
        print(f"# INVALID: {problem}", flush=True)

    record = {
        "args": vars(args),
        "jax": jax.__version__,
        "x64": bool(jax.config.read("jax_enable_x64")),
        "devices": [d.device_kind for d in devices[: args.ndev]],
        "host": platform.node(),
        "points": rows,
        "monotonicity_problems": problems,
    }
    _RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"ladder_ndev{args.ndev}_leaf{args.leaf}"
    path = _RESULTS / f"{stem}{('_' + args.tag) if args.tag else ''}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"# wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
