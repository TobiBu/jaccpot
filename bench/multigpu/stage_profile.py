"""Attribute distributed force-evaluation time to pipeline stages.

Stage 1 of ``docs/multigpu_efficiency_plan.md``. It exists because four attempts
to locate the dominant cost by varying configuration all failed: relaxing
``theta_cross``, toggling the fused-M2L flag, and sweeping the near-field cap and
the traversal pair queue each moved the total by less than the noise band, while a
problem generating sixty cell-pair interactions still took ~50 ms. Guessing does
not work here; the profile has to say.

How it works
------------
``jaccpot/distributed/fmm.py``'s ``shard_map`` body wraps its stages in
``jax.named_scope("jc_<stage>")``. Those names survive into the XLA op metadata, so
a profiler trace can be bucketed by stage with no host callbacks and no ablation.
``named_scope`` is metadata only -- it does not introduce a call boundary and does
not inhibit fusion -- so the profiled executable is the one that actually runs.
That is the whole reason for preferring it to ``jax.named_call``, which would
change the thing being measured.

What it cannot tell you, measured
---------------------------------
**Scope attribution does not survive fusion here.** In practice only ~5% of device
time lands in a named scope: the collectives keep their identity (they do not fuse)
and everything else is folded into fusion kernels whose names carry no scope. The
unattributed remainder is reported rather than spread over the stages, because
that remainder is the finding, not a gap in the bookkeeping.

So the useful output is ``top_ops``: total device time and **call count** per op.
That is what located the cost structure at 4 devices, 16384 particles/device
(~75 ms of device time per device per evaluation):

- ``nearfield_leafpair_*`` ~16 ms, and exactly one launch per device per
  evaluation -- the fused near-field kernel is efficient;
- collectives ~10 ms (AllGather, RaggedAllToAll, MultiGpuBarrier);
- and ~60% in a long tail of thousands of tiny launches -- over a thousand
  ``memcpy32_post`` and a few hundred each of several small fusions, *per device
  per evaluation*.

The call counts are the diagnostic. A stage that is slow shows up as time; a path
that is dispatch-bound shows up as a count, and no amount of configuration
tuning moves it -- which is why five separate hypotheses about this floor
(``theta_cross``, the fused-M2L flag, the near-field cap, the traversal queue, and
the L2L level bound) were each measured and each wrong.

Usage
-----
::

    CUDA_VISIBLE_DEVICES=2,4 python -m bench.multigpu.stage_profile \\
        --ndev 2 --n 32768 --leaf-size 256 --process-block 256

Compare a tiny and a production-sized problem -- the difference is what scales:

::

    ... --n 4096   --leaf-size 256   # ~60 pairs of work
    ... --n 65536  --leaf-size 256   # production
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import pathlib
import sys
import tempfile
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "multigpu/stage_profile.json"

# The scopes annotated in jaccpot/distributed/fmm.py, in pipeline order.
STAGES = (
    "jc_geometry",
    "jc_upward_local",
    "jc_self_walk",
    "jc_coarse_frontier",
    "jc_upward_coarse",
    "jc_all_gather_coarse",
    "jc_coarse_m2m",
    "jc_cross_walk",
    "jc_halo_import",
    "jc_l2p",
    "jc_near_p2p_self",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    runmeta.add_common_args(p)
    p.add_argument("--ndev", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--leaf-size", type=int, default=256)
    p.add_argument("--process-block", type=int, default=256)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--theta", type=float, default=0.4)
    p.add_argument("--theta-cross", type=float, default=None)
    p.add_argument("--max-pair-queue", type=int, default=262144)
    p.add_argument("--cross-max-pair-queue", type=int, default=262144)
    p.add_argument("--distribution", default="uniform")
    p.add_argument(
        "--reps",
        type=int,
        default=3,
        help="traced evaluations; per-stage totals are divided by this",
    )
    p.add_argument("--trace-dir", default=None, help="keep the raw trace here")
    return p.parse_args()


def _gpu_pids(events: list[dict[str, Any]]) -> set[int]:
    """Return pids whose process_name metadata looks like a device track."""

    pids: set[int] = set()
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            label = str((e.get("args") or {}).get("name", ""))
            if "GPU" in label or "device" in label.lower():
                pids.add(e.get("pid"))
    return pids


def _aggregate(trace_path: pathlib.Path, reps: int) -> dict[str, Any]:
    """Bucket device-op durations by named scope.

    Parameters
    ----------
    trace_path : pathlib.Path
        A ``*.trace.json.gz`` written by ``jax.profiler.trace``.
    reps : int
        Number of traced evaluations, to report per-evaluation figures.

    Returns
    -------
    dict
        Per-stage microseconds per evaluation, the unattributed remainder, and
        the total device time seen.
    """

    with gzip.open(trace_path, "rt") as fh:
        trace = json.load(fh)
    events = trace.get("traceEvents", [])
    gpu = _gpu_pids(events)

    per_stage: dict[str, float] = collections.defaultdict(float)
    per_op: dict[str, float] = collections.defaultdict(float)
    per_op_calls: dict[str, int] = collections.defaultdict(int)
    total = 0.0
    unattributed = 0.0
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in gpu:
            continue
        dur = float(e.get("dur") or 0.0)
        if dur <= 0:
            continue
        total += dur
        name = str(e.get("name", ""))
        per_op[name] += dur
        per_op_calls[name] += 1
        hit = next((s for s in STAGES if s in name), None)
        if hit is None:
            # scope names can also live in the metadata rather than the op name
            args = e.get("args") or {}
            blob = " ".join(str(v) for v in args.values())
            hit = next((s for s in STAGES if s in blob), None)
        if hit is None:
            unattributed += dur
        else:
            per_stage[hit] += dur

    r = max(1, reps)
    top = sorted(per_op.items(), key=lambda kv: -kv[1])[:20]
    return {
        "per_stage_us_per_eval": {s: per_stage.get(s, 0.0) / r for s in STAGES},
        "unattributed_us_per_eval": unattributed / r,
        "total_device_us_per_eval": total / r,
        "attributed_fraction": (total - unattributed) / total if total else 0.0,
        # The op listing is the instrument that actually works here; see the
        # module docstring on why scope attribution does not survive fusion.
        "top_ops": [
            {
                "name": n,
                "us_per_eval": us / r,
                "calls_per_eval": per_op_calls[n] / r,
            }
            for n, us in top
        ],
        "distinct_ops": len(per_op),
        "total_calls_per_eval": sum(per_op_calls.values()) / r,
    }


def main() -> int:
    """Profile one configuration and write the per-stage attribution.

    Returns
    -------
    int
        0 on success, 1 when no trace could be parsed.
    """

    args = _parse_args()
    runmeta.select_gpu(args.gpu_select, num_gpus=int(args.ndev))
    runmeta.enable_x64(args.dtype)

    import jax

    from bench.multigpu.harness import measure_point  # noqa: E402

    # measure_point builds, warms up and times; we reuse it so the profiled
    # configuration is exactly a harness point rather than a second code path.
    record = measure_point(
        ndev=int(args.ndev),
        n=int(args.n),
        order=int(args.order),
        theta=float(args.theta),
        theta_cross=args.theta_cross,
        leaf_size=int(args.leaf_size),
        distribution=args.distribution,
        seed=int(args.seed),
        repeats=1,
        warmup=1,
        max_pair_queue=int(args.max_pair_queue),
        cross_max_pair_queue=int(args.cross_max_pair_queue),
        process_block=int(args.process_block),
    )
    if not record["valid"]:
        raise SystemExit(
            f"configuration is invalid ({record['overflowed']}); the walk truncated, "
            "so profiling it would attribute time to work that was skipped"
        )

    # Rebuild the same evaluator and trace it. measure_point does not hand back
    # its closure, so this repeats the (cheap, cached-compile) build.
    tmp = pathlib.Path(args.trace_dir or tempfile.mkdtemp(prefix="jc_stage_prof_"))
    tmp.mkdir(parents=True, exist_ok=True)

    import dataclasses

    import jax.numpy as jnp
    from yggdrax.distributed import make_mesh

    from bench.multigpu.harness import make_distribution
    from jaccpot.distributed import DistributedFMMConfig
    from jaccpot.distributed.fmm import make_force_evaluator, partition_for_devices

    ndev = int(args.ndev)
    config = dataclasses.replace(
        DistributedFMMConfig(),
        order=int(args.order),
        theta=float(args.theta),
        leaf_size=int(args.leaf_size),
        process_block=int(args.process_block),
        max_pair_queue=int(args.max_pair_queue),
        cross_max_pair_queue=int(args.cross_max_pair_queue),
        **(
            {} if args.theta_cross is None else {"theta_cross": float(args.theta_cross)}
        ),
    )
    pos_np, mass_np = make_distribution(
        args.distribution, int(args.n), ndev, int(args.seed)
    )
    part = partition_for_devices(pos_np, mass_np, ndev, leaf_size=config.leaf_size)
    ev = make_force_evaluator(config, ndev, part["cap"], make_mesh(ndev), jit=True)
    a = (
        jnp.asarray(part["pos_flat"]),
        jnp.asarray(part["mass_flat"]),
        jnp.asarray(part["gid_flat"]),
        jnp.asarray(part["counts"]),
    )
    jax.block_until_ready(ev(*a)[0])  # warm up outside the trace

    with jax.profiler.trace(str(tmp)):
        for _ in range(int(args.reps)):
            jax.block_until_ready(ev(*a)[0])

    traces = sorted(glob.glob(str(tmp / "**" / "*.trace.json.gz"), recursive=True))
    if not traces:
        raise SystemExit(f"no trace written under {tmp}")
    agg = _aggregate(pathlib.Path(traces[-1]), int(args.reps))

    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config={
            "n": int(record["n"]),
            "theta": float(config.theta),
            "order": int(config.order),
            "basis": config.basis,
            "seed": int(args.seed),
            "device": runmeta.device_label(),
            "precision": args.dtype,
            "ndev": ndev,
            "per_device_n": int(record["per_device_n"]),
            "leaf_size": int(config.leaf_size),
            "process_block": int(config.process_block),
            "theta_cross": float(config.theta_cross),
            "max_pair_queue": int(config.max_pair_queue),
            "reps": int(args.reps),
        },
        meta=runmeta.run_meta({"argv": sys.argv[1:], "trace_dir": str(tmp)}),
        data={"profile": agg, "harness_record": record},
    )

    wall_us = record["median_s"] * 1e6
    print(f"\nwall clock (harness median): {wall_us/1000:.1f} ms/eval")
    print(
        f"device time seen in trace:   {agg['total_device_us_per_eval']/1000:.1f} ms/eval"
    )
    print(f"attributed to named stages:  {100*agg['attributed_fraction']:.1f}%\n")
    print(f"  {'stage':24s} {'ms/eval':>9} {'% of device':>12}")
    tot = agg["total_device_us_per_eval"] or 1.0
    for s, us in sorted(agg["per_stage_us_per_eval"].items(), key=lambda kv: -kv[1]):
        print(f"  {s:24s} {us/1000:>9.2f} {100*us/tot:>11.1f}%")
    print(
        f"  {'(unattributed)':24s} "
        f"{agg['unattributed_us_per_eval']/1000:>9.2f} "
        f"{100*agg['unattributed_us_per_eval']/tot:>11.1f}%"
    )
    # The op listing is the part that works; scopes do not survive fusion.
    nd = max(1, int(args.ndev))
    print(
        f"\ntop device ops ({agg['distinct_ops']} distinct, "
        f"{agg['total_calls_per_eval']/nd:.0f} launches per device per eval):"
    )
    print(f"  {'ms/dev/eval':>12} {'calls/dev/eval':>15}  op")
    for op in agg["top_ops"][:12]:
        print(
            f"  {op['us_per_eval']/1000/nd:>12.2f} "
            f"{op['calls_per_eval']/nd:>15.0f}  {op['name'][:70]}"
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
