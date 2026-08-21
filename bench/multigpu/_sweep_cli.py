"""Shared argument handling and artifact writing for the multi-GPU sweeps.

The parent process here deliberately **never imports JAX**. It spawns one worker
per point (see :mod:`bench.multigpu.harness`), and a parent that had initialised
a backend would be holding a device its own children need. That is also why the
device label in the artifact is read back out of a worker's provenance rather
than queried locally.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio  # noqa: E402

# The per-device ceiling is a LEAF-COUNT ceiling, not a particle-count one, and
# it is set by ``process_block``. Measured 2026-08-21 on 2xA100:
#
#   64 leaves/device  @ process_block 64  -> valid, 0 retries, at every leaf size
#   128 leaves/device @ process_block 64  -> truncated
#   128 leaves/device @ process_block 128 -> truncated
#   128 leaves/device @ process_block 256 -> valid
#
# Per-device capacity is therefore (leaves per device) x leaf_size, and both
# factors are configuration. 32768 particles/device runs clean at leaf 512, and
# 16384 at leaf 128 once process_block is 256 -- against the 8000 the phase-5 log
# calls "healthy", which was leaf 128 x 64 leaves and nothing more fundamental.
#
# The failure mode is a TRUNCATED WALK, and it reads as *faster*, because a
# truncated walk does less work. Never infer validity from a wall clock here:
# check self_near_pairs, which is ~constant for a given leaf count and collapses
# when the walk truncates.
DEFAULT_LEAF_SIZE = 256
DEFAULT_PROCESS_BLOCK = 256
DEFAULT_PAIR_QUEUE = 262144
# 64 leaves/device at the default leaf size -- the conservative point that needed
# no capacity retries at every leaf size tested.
HEALTHY_PER_DEVICE_N = DEFAULT_LEAF_SIZE * 64


def add_sweep_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the flags every multi-GPU sweep shares."""

    p.add_argument(
        "--ndevs",
        default="2,3,4",
        help="comma-separated device counts, one process each (default: 2,3,4)",
    )
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--theta", type=float, default=0.4)
    p.add_argument("--leaf-size", type=int, default=DEFAULT_LEAF_SIZE)
    p.add_argument(
        "--process-block",
        type=int,
        default=DEFAULT_PROCESS_BLOCK,
        help=(
            "dual-tree walk block. This is the knob that sets the per-device leaf "
            "ceiling; too small truncates the walk silently"
        ),
    )
    p.add_argument("--max-pair-queue", type=int, default=DEFAULT_PAIR_QUEUE)
    p.add_argument(
        "--cross-far-cap",
        type=int,
        default=1 << 20,
        help=(
            "cross-domain far-list capacity. Its default is floored at 16384, which "
            "the honest coarse extents now exceed at higher device counts -- the "
            "far field is engaged, so the far list is no longer nearly empty"
        ),
    )
    p.add_argument("--cross-max-pair-queue", type=int, default=DEFAULT_PAIR_QUEUE)
    p.add_argument("--basis", default="real")
    p.add_argument("--mac-type", default="dehnen")
    p.add_argument("--nearfield-backend", default="auto")
    p.add_argument("--distribution", default="uniform")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument(
        "--gpu-select",
        choices=("least-used", "first", "none"),
        default="least-used",
        help="passed to each worker; workers claim their whole mesh at once",
    )
    p.add_argument(
        "--accuracy-targets",
        type=int,
        default=0,
        help=(
            "per point, compare against a direct sum on this many sampled targets. "
            "Section 5 needs it: a scaling curve without an accuracy column cannot "
            "distinguish a working far field from a truncated one"
        ),
    )
    p.add_argument("--json-out", default=None)
    return p


def worker_argv(args: argparse.Namespace) -> list[str]:
    """Translate parsed sweep args into the flags every worker takes."""

    return [
        "--order",
        str(args.order),
        "--theta",
        str(args.theta),
        "--leaf-size",
        str(args.leaf_size),
        "--process-block",
        str(args.process_block),
        "--max-pair-queue",
        str(args.max_pair_queue),
        "--cross-max-pair-queue",
        str(args.cross_max_pair_queue),
        "--basis",
        args.basis,
        "--mac-type",
        args.mac_type,
        "--nearfield-backend",
        args.nearfield_backend,
        "--distribution",
        args.distribution,
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--dtype",
        args.dtype,
        "--gpu-select",
        args.gpu_select,
        "--accuracy-targets",
        str(args.accuracy_targets),
        "--cross-far-cap",
        str(args.cross_far_cap),
    ]


def _first_meta(records: list[dict[str, Any]], key: str) -> Any:
    """Return the first worker's value for ``key``, or None."""

    for r in records:
        val = (r.get("meta") or {}).get(key)
        if val:
            return val
    return None


def _device_from(records: list[dict[str, Any]]) -> str:
    return str(_first_meta(records, "device_kind") or "unknown")


def write_sweep(
    default_out: str,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    *,
    extra_config: Optional[dict[str, Any]] = None,
) -> pathlib.Path:
    """Write a sweep artifact, and say plainly what was dropped.

    Invalid points are kept in the artifact rather than filtered out: a point
    that overflowed is a *finding* about the operating regime, and silently
    dropping it would make a truncated-force regime look like a region nobody
    measured.

    Parameters
    ----------
    default_out : str
        Artifact path used when ``--json-out`` was not given.
    args : argparse.Namespace
        Parsed sweep arguments.
    records : list
        Worker records, in point order.
    extra_config : dict, optional
        Additional config axes to record.

    Returns
    -------
    pathlib.Path
        The path written.
    """

    valid = [r for r in records if r.get("valid")]
    invalid = [r for r in records if not r.get("valid")]
    if invalid:
        print(f"\n{len(invalid)} of {len(records)} point(s) are NOT valid:")
        for r in invalid:
            why = r.get("overflowed") or r.get("error", "unknown")
            print(f"  ndev={r.get('ndev')} n={r.get('n')}: {why}")
        print("Recorded with valid=false; do not plot them as scaling points.")

    config: dict[str, Any] = {
        "n": [r.get("n") for r in records],
        "theta": args.theta,
        "order": args.order,
        "basis": args.basis,
        "seed": args.seed,
        "device": _device_from(records),
        "precision": args.dtype,
        "ndevs": [r.get("ndev") for r in records],
        "leaf_size": args.leaf_size,
        "mac_type": args.mac_type,
        "nearfield_backend": args.nearfield_backend,
        "distribution": args.distribution,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "healthy_per_device_n": HEALTHY_PER_DEVICE_N,
        "accuracy_targets": args.accuracy_targets,
        "process_block": args.process_block,
        "max_pair_queue": args.max_pair_queue,
        "cross_max_pair_queue": args.cross_max_pair_queue,
        "n_valid": len(valid),
        "n_invalid": len(invalid),
    }
    if extra_config:
        config.update(extra_config)

    # git_meta(), NOT run_meta(): the latter imports jax to record its version
    # and backend, which would initialise a backend in the parent and grab a
    # device for nothing. The jax provenance is taken from a worker instead --
    # the workers are the processes that actually measured.
    from examples.jaccpot_paper.common import runmeta

    meta: dict[str, Any] = dict(runmeta.git_meta())
    meta.update(
        {
            "argv": sys.argv[1:],
            "sweep_parent": True,
            "parent_imported_jax": False,
            "worker_jax": _first_meta(records, "jax_version"),
            "worker_device_kind": _first_meta(records, "device_kind"),
            "worker_cuda_visible_devices": [
                (r.get("meta") or {}).get("cuda_visible_devices") for r in records
            ],
        }
    )

    out = jsonio.write_result(
        args.json_out or default_out,
        config=config,
        meta=meta,
        data={"records": records},
    )
    print(f"wrote {out}")
    return out
