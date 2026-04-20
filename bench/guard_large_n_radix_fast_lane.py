"""Regression guard for the radix fast lane 1M large-N benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime
from typing import Any


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-particles", type=int, default=1_048_576)
    parser.add_argument("--leaf-size", type=int, default=256)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-scope", choices=("steady_eval", "full"), default="steady_eval")
    parser.add_argument("--fast-block-size", type=int, default=8)
    parser.add_argument("--min-speedup", type=float, default=None)
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


def _make_output_paths(prefix_arg: pathlib.Path | None) -> tuple[pathlib.Path, pathlib.Path]:
    if prefix_arg is not None:
        prefix = prefix_arg if prefix_arg.is_absolute() else (REPO_ROOT / prefix_arg)
        return prefix.with_suffix(".csv"), prefix.with_suffix(".json")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = REPO_ROOT / "benchmarks" / f"radix_fast_lane_guard_1m_{stamp}"
    return out_prefix.with_suffix(".csv"), out_prefix.with_suffix(".json")


def main() -> None:
    args = _parse_args()
    cfg = dict(FROZEN_CONFIG)
    worker_env, selected_cuda_visible = _build_worker_env(args)

    baseline_row = _run_case(block_size=0, args=args, cfg=cfg, worker_env=worker_env)
    fast_row = _run_case(
        block_size=int(args.fast_block_size),
        args=args,
        cfg=cfg,
        worker_env=worker_env,
    )

    metric = _metric_key(str(args.benchmark_scope))
    base = float(baseline_row.get(metric, float("nan")))
    fast = float(fast_row.get(metric, float("nan")))
    if not (base > 0.0 and fast > 0.0):
        raise RuntimeError(f"Non-positive benchmark metric values: baseline={base}, fast={fast}")

    speedup = base / fast
    min_speedup = float(args.min_speedup) if args.min_speedup is not None else (
        2.0 if str(args.benchmark_scope) == "steady_eval" else 1.03
    )
    if speedup < float(min_speedup):
        raise RuntimeError(
            f"Radix fast-lane speedup guard failed: {speedup:.3f}x < {min_speedup:.3f}x "
            f"(metric={metric}, baseline={base:.6f}, fast={fast:.6f})"
        )

    csv_path, json_path = _make_output_paths(args.output_prefix)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "lane_label": "baseline",
            "target_block_size": 0,
            "benchmark_scope": args.benchmark_scope,
            "metric_key": metric,
            "metric_seconds": base,
            "cuda_visible_devices": selected_cuda_visible,
            **baseline_row,
        },
        {
            "lane_label": "fast_lane",
            "target_block_size": int(args.fast_block_size),
            "benchmark_scope": args.benchmark_scope,
            "metric_key": metric,
            "metric_seconds": fast,
            "cuda_visible_devices": selected_cuda_visible,
            **fast_row,
        },
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
        "baseline_metric_seconds": base,
        "fast_lane_metric_seconds": fast,
        "speedup": speedup,
        "min_speedup": float(min_speedup),
        "fast_block_size": int(args.fast_block_size),
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
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(payload, sort_keys=True))
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")


if __name__ == "__main__":
    main()
