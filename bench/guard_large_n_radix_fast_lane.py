"""Regression guard for the large-N radix fast lane.

WHAT THIS USED TO BE, AND WHY IT COULD NOT WORK
-----------------------------------------------
It ran the worker twice -- ``target_owned_block_size`` 0 against 8 -- labelled the
arms "baseline" and "fast_lane", and required the second to be 2.0x faster
(``steady_eval``) or 1.03x faster (``full``). On unmodified ``main``, measured on an
uncontended A100 at N=1048576, leaf 256, order 4, fp32::

    steady_eval   baseline 0.3957 s   fast 0.4053 s   0.976x  < 2.00x   FAIL
    full          baseline 0.7035 s   fast 0.7035 s   1.000x  < 1.03x   FAIL

Both arms were the same lane. ``runtime/_large_n_nearfield.resolve_large_n_execution_config``
sets ``radix_fast_lane = True`` unconditionally once five configuration requirements
hold, and block size 0 means "default to 32", so the comparison was 32 against 8.

**There is no A/B axis here at all**, which is why this file no longer tries to be a
speedup ratio. Four candidates were checked and all four are gone:

* ``target_owned_block_size`` is *measurably inert*. Swept 1, 2, 4, 8, 16, 32, 64,
  128 in two interleaved passes: every median between 0.3934 s and 0.3997 s, a total
  spread of **1.016x**, while two samples of a single setting differ by up to 2.6 %.
  Between-block variation is smaller than within-block noise
  (``bench/sweep_large_n_target_block_size.py``).
* ``nearfield_mode="bucketed"`` is not an alternative mode: the resolver *hardcodes*
  it, so bucketed **is** the fast lane's mode.
* ``JACCPOT_DISABLE_LARGE_N_SPECIALIZED_NEARFIELD`` is **structurally unreachable**
  on this path. ``evaluate_large_n_state`` takes its fast-lane branch (lines
  2176-2278) and returns from inside it; the flag is only consumed at line 2423,
  below that branch. Its 1.016x reading was a vacuous measurement of a dead switch,
  not evidence of a performance-neutral one.
* There is no non-fast-lane arm to fall back to: accel-only large-N evaluation
  *raises* when ``radix_fast_lane`` is False.

WHAT IT IS NOW
--------------
Three checks, in increasing order of how much they depend on the hardware:

1. **Structural.** The prepared state must actually be on the fast lane -- lane flag
   set, payload present, mode bucketed, block size resolved to a positive value.
   This is the check the old guard could not make (the worker did not report the lane
   until now) and it is the one that matters: it is free, hardware-independent, and
   it is what "radix fast lane guard" should mean. A vacuity guard beats a tolerance.
2. **Near-field share.** The fast lane honours ``JACCPOT_LARGE_N_EVAL_DISABLE_FAR``,
   so the far-only cost can be measured against the full cost in the same process on
   the same card. Track B rests on the far field being under 1 % of the evaluation at
   galaxy scale; this asserts it, which makes the guard fail if the far field or the
   surrounding overheads ever grow. Being a *ratio* of two back-to-back timings, it
   is immune to card speed and to contention.
3. **Fixed reference.** Only if ``--reference-seconds`` is given, because there is no
   relative timing axis left. Contention makes an absolute bar the weakest of the
   three -- the audit saw one configuration read 1.35 s and 2.23 s depending on who
   else was on the card -- so it is opt-in and its tolerance is generous by default.

USAGE
-----
    # structural + share, no absolute bar (the default; safe to run anywhere)
    python -m bench.guard_large_n_radix_fast_lane \
        --num-particles 1048576 --leaf-size 256 --max-order 4

    # add the absolute bar, on the card the reference was taken on
    python -m bench.guard_large_n_radix_fast_lane --reference-seconds 0.396 \
        --reference-tolerance 1.25 --cuda-visible-devices 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "examples" / "benchmark_gpu_radix_worker.py"
YGGDRAX_ROOT = REPO_ROOT.parent / "yggdrax"

FROZEN_CONFIG: dict[str, Any] = {
    "preset": "large_n_gpu",
    "basis": "solidfmm",
    "tree_type": "radix",
    "leaf_target": 256,
    "theta": 0.6,
    "softening": 0.001,
    "working_dtype": "float32",
    "memory_objective": "minimum_memory",
    "nearfield_mode": "bucketed",
    "nearfield_edge_chunk_size": 512,
    "streamed_far_pairs": True,
    "grouped_interactions": False,
    "enable_interaction_cache": False,
    "retain_traversal_result": False,
    "retain_interactions": False,
    "traversal_config": {
        "max_pair_queue": 524288,
        "process_block": 256,
        "max_interactions_per_node": 16384,
        "max_neighbors_per_leaf": 8192,
    },
    "worker_autotune_traversal": False,
    "worker_autotune_nearfield_chunk": False,
}

#: Ceiling on the far field's share of the evaluation. Track B's ablation measured
#: 0.05 % to 0.8 % across leaf 128-1024 and order 4-8 at N=1e6, so 5 % is two
#: decades of headroom over the worst of those and still fails loudly if the far
#: field stops being negligible.
_MAX_FAR_SHARE = 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-particles", type=int, default=1_048_576)
    parser.add_argument("--leaf-size", type=int, default=256)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--benchmark-scope", choices=("steady_eval", "full"), default="steady_eval"
    )
    parser.add_argument(
        "--reference-seconds",
        type=float,
        default=None,
        help="absolute reference for this host/card; omitted means no absolute bar",
    )
    parser.add_argument(
        "--reference-tolerance",
        type=float,
        default=1.25,
        help="fail if the measurement exceeds reference * this",
    )
    parser.add_argument(
        "--max-far-share",
        type=float,
        default=_MAX_FAR_SHARE,
        help="fail if the far field exceeds this share of the evaluation",
    )
    parser.add_argument(
        "--skip-far-share",
        action="store_true",
        help="skip the ablation arm (one fewer worker run)",
    )
    parser.add_argument("--output-prefix", type=pathlib.Path, default=None)
    parser.add_argument("--use-autocvd", action="store_true", default=True)
    parser.add_argument("--no-autocvd", dest="use_autocvd", action="store_false")
    parser.add_argument("--autocvd-num-gpus", type=int, default=1)
    parser.add_argument("--autocvd-exclude", nargs="*", default=[])
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--allow-missing-autocvd", action="store_true")
    return parser.parse_args()


def _extract_worker_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if text.startswith("{") and text.endswith("}"):
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "num_particles" in parsed:
                return parsed
    raise RuntimeError("Could not find worker JSON payload in command output.")


def _build_worker_env(args: argparse.Namespace) -> tuple[dict[str, str], str]:
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices).strip()
    elif args.use_autocvd:
        try:
            from autocvd import autocvd

            autocvd(
                num_gpus=int(args.autocvd_num_gpus),
                least_used=True,
                exclude=list(args.autocvd_exclude),
            )
            env["CUDA_VISIBLE_DEVICES"] = os.environ.get(
                "CUDA_VISIBLE_DEVICES",
                env.get("CUDA_VISIBLE_DEVICES", ""),
            )
        except ImportError:
            if not args.allow_missing_autocvd:
                raise
    env.setdefault("JAX_ENABLE_X64", "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    visible = str(env.get("CUDA_VISIBLE_DEVICES", "")).strip()
    first_visible = ""
    if visible:
        first_visible = visible.split(",")[0].strip()
        if first_visible:
            env["JACCPOT_NVIDIA_SMI_GPU_INDEX"] = first_visible
    pythonpath_parts = [str(REPO_ROOT)]
    if YGGDRAX_ROOT.exists():
        pythonpath_parts.append(str(YGGDRAX_ROOT))
    existing = str(env.get("PYTHONPATH", "")).strip()
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env, visible


def _run_case(
    *,
    block_size: int,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    worker_env: dict[str, str],
) -> dict[str, Any]:
    """Run the worker once and return its JSON row.

    Parameters
    ----------
    block_size
        Value for the two target-block-size env vars. 0 leaves the lane's own
        default (32) in place. Kept as a parameter because
        ``bench/sweep_large_n_target_block_size.py`` sweeps it -- the guard itself no
        longer varies it, having measured it inert.
    args
        Parsed arguments supplying the problem size and run counts.
    cfg
        Frozen worker configuration.
    worker_env
        Environment for the worker subprocess.

    Returns
    -------
    dict[str, Any]
        The worker's JSON payload.

    Raises
    ------
    RuntimeError
        If the worker exits nonzero or returns an error payload.
    """
    cfg_payload = dict(cfg)
    cfg_payload["benchmark_scope"] = str(args.benchmark_scope)
    cfg_json = json.dumps(cfg_payload, separators=(",", ":"))

    env = dict(worker_env)
    env["YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE"] = str(int(block_size))
    env["JACCPOT_LARGE_N_TARGET_BLOCK_SIZE"] = str(int(block_size))
    command = [
        sys.executable,
        str(WORKER),
        "--mode",
        "sweep",
        "--num-particles",
        str(int(args.num_particles)),
        "--leaf-size",
        str(int(args.leaf_size)),
        "--max-order",
        str(int(args.max_order)),
        "--runs",
        str(int(args.runs)),
        "--warmup",
        str(int(args.warmup)),
        "--dtype",
        "float32",
        "--seed",
        str(int(args.seed)),
        "--config-json",
        cfg_json,
    ]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker command failed (block_size={block_size}):\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    row = _extract_worker_json(proc.stdout)
    row_error = str(row.get("error", "")).strip()
    if row_error:
        raise RuntimeError(
            f"worker returned error payload (block_size={block_size}): {row_error}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return row


def _metric_key(benchmark_scope: str) -> str:
    return "mean_seconds" if benchmark_scope == "full" else "evaluate_mean_seconds"


def _check_lane(row: dict[str, Any]) -> list[str]:
    """Return the structural failures, if any, for one worker row.

    The fast-lane branch of ``evaluate_large_n_state`` is entered only when the lane
    flag is set AND the payload is present, so both are asserted rather than just the
    flag. A run that quietly prepared some other state would otherwise be timed and
    reported as healthy.

    Parameters
    ----------
    row
        A worker JSON payload.

    Returns
    -------
    list[str]
        Human-readable failure descriptions; empty when the lane is engaged.
    """
    problems: list[str] = []
    if not bool(row.get("large_n_radix_fast_lane", False)):
        problems.append(
            "prepared state is not on the radix fast lane "
            "(large_n_radix_fast_lane is false)"
        )
    if not bool(row.get("large_n_radix_payload_present", False)):
        problems.append(
            "radix fast-lane payload is absent, so evaluate_large_n_state would not "
            "take its fast-lane branch"
        )
    mode = str(row.get("large_n_nearfield_mode", "")).strip().lower()
    if mode != "bucketed":
        problems.append(f"nearfield_mode is {mode!r}, expected 'bucketed'")
    block = int(row.get("large_n_target_block_size", -1))
    if block <= 0:
        problems.append(
            f"target block size resolved to {block}, expected a positive value"
        )
    return problems


def main() -> None:
    """Run the guard and raise on the first check that fails."""
    args = _parse_args()
    cfg = dict(FROZEN_CONFIG)
    worker_env, selected_cuda_visible = _build_worker_env(args)
    metric = _metric_key(str(args.benchmark_scope))

    full_row = _run_case(block_size=0, args=args, cfg=cfg, worker_env=worker_env)
    full_seconds = float(full_row.get(metric, float("nan")))
    if not full_seconds > 0.0:
        raise RuntimeError(f"Non-positive benchmark metric: {full_seconds}")

    # 1. Structural, and first, because a timing taken off the wrong lane means
    #    nothing at all.
    problems = _check_lane(full_row)
    if problems:
        raise RuntimeError(
            "Radix fast-lane guard failed its structural checks:\n  - "
            + "\n  - ".join(problems)
        )

    # 2. Near-field share, from a second run with the far field ablated. A ratio of
    #    two timings on the same card, so contention cannot move it.
    far_seconds: Optional[float] = None
    far_share: Optional[float] = None
    if not args.skip_far_share:
        far_env = dict(worker_env)
        far_env["JACCPOT_LARGE_N_EVAL_DISABLE_NEAR"] = "1"
        far_row = _run_case(block_size=0, args=args, cfg=cfg, worker_env=far_env)
        far_seconds = float(far_row.get(metric, float("nan")))
        if far_seconds > 0.0:
            far_share = far_seconds / full_seconds
            if far_share > float(args.max_far_share):
                raise RuntimeError(
                    f"Far field is {far_share:.2%} of the evaluation, above the "
                    f"{float(args.max_far_share):.2%} ceiling "
                    f"(full={full_seconds:.6f}s, far_only={far_seconds:.6f}s). Track B "
                    "assumes the near field is essentially the whole cost; if that has "
                    "stopped being true the near-field conclusions need revisiting."
                )

    # 3. Absolute reference, last and opt-in, because it is the check contention can
    #    break on its own.
    reference = args.reference_seconds
    if reference is not None:
        ceiling = float(reference) * float(args.reference_tolerance)
        if full_seconds > ceiling:
            raise RuntimeError(
                f"Radix fast-lane timing regressed: {full_seconds:.6f}s > "
                f"{ceiling:.6f}s (reference {float(reference):.6f}s x tolerance "
                f"{float(args.reference_tolerance):.2f}, metric={metric}). Re-run on an "
                "uncontended card before believing this."
            )

    csv_path, json_path = _make_output_paths(args.output_prefix)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "arm": "full",
            "benchmark_scope": args.benchmark_scope,
            "metric_key": metric,
            "metric_seconds": full_seconds,
            "cuda_visible_devices": selected_cuda_visible,
            **full_row,
        }
    ]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "cwd": str(REPO_ROOT),
        "hostname": os.uname().nodename,
        "benchmark_scope": str(args.benchmark_scope),
        "cuda_visible_devices": selected_cuda_visible,
        "metric_key": metric,
        "metric_seconds": full_seconds,
        "far_only_seconds": far_seconds,
        "far_share": far_share,
        "max_far_share": float(args.max_far_share),
        "reference_seconds": reference,
        "reference_tolerance": float(args.reference_tolerance),
        "lane": {
            key: full_row.get(key)
            for key in (
                "large_n_radix_fast_lane",
                "large_n_radix_payload_present",
                "large_n_nearfield_mode",
                "large_n_target_block_size",
            )
        },
        "num_particles": int(args.num_particles),
        "leaf_size": int(args.leaf_size),
        "max_order": int(args.max_order),
        "runs": int(args.runs),
        "warmup": int(args.warmup),
        "seed": int(args.seed),
        "frozen_config": cfg,
        "rows": rows,
        "csv_path": str(csv_path),
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    share_text = "skipped" if far_share is None else f"{far_share:.3%}"
    print(
        f"PASS  lane=engaged  block={full_row.get('large_n_target_block_size')}  "
        f"{metric}={full_seconds:.4f}s  far_share={share_text}  "
        f"reference={'none' if reference is None else f'{reference:.4f}s'}"
    )
    print(json.dumps(payload, sort_keys=True))
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")


def _make_output_paths(
    prefix_arg: pathlib.Path | None,
) -> tuple[pathlib.Path, pathlib.Path]:
    if prefix_arg is not None:
        prefix = prefix_arg if prefix_arg.is_absolute() else (REPO_ROOT / prefix_arg)
        return prefix.with_suffix(".csv"), prefix.with_suffix(".json")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = REPO_ROOT / "benchmarks" / f"radix_fast_lane_guard_1m_{stamp}"
    return out_prefix.with_suffix(".csv"), out_prefix.with_suffix(".json")


if __name__ == "__main__":
    main()
