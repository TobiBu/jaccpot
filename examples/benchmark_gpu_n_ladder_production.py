"""Production-oriented GPU N-ladder sweep for stable large-N runtime knobs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import replace
from typing import Any


def _configure_environment(args: argparse.Namespace) -> int:
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        print("Set CUDA_VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])
    elif args.use_autocvd:
        try:
            from autocvd import autocvd

            autocvd(
                num_gpus=int(args.autocvd_num_gpus),
                least_used=bool(args.autocvd_least_used),
                exclude=list(args.autocvd_exclude),
            )
            print(
                "autocvd selected CUDA_VISIBLE_DEVICES =",
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
        except ImportError:
            print("autocvd is not installed. Using existing CUDA visibility.")
    else:
        print(
            "Using existing CUDA visibility:",
            os.environ.get("CUDA_VISIBLE_DEVICES", "<all visible>"),
        )

    os.environ.setdefault("JACCPOT_INDEX_PRECISION", str(args.index_precision))
    os.environ.setdefault("YGGDRAX_INDEX_PRECISION", str(args.index_precision))
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    if "--xla_gpu_enable_command_buffer=" not in os.environ.get("XLA_FLAGS", ""):
        existing_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
        command_buffer_off = "--xla_gpu_enable_command_buffer="
        os.environ["XLA_FLAGS"] = (
            f"{existing_xla_flags} {command_buffer_off}".strip()
            if existing_xla_flags
            else command_buffer_off
        )

    visible_physical_gpus = [
        part.strip()
        for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if part.strip()
    ]
    nvidia_smi_gpu_index = int(visible_physical_gpus[0]) if visible_physical_gpus else 0
    os.environ["JACCPOT_NVIDIA_SMI_GPU_INDEX"] = str(nvidia_smi_gpu_index)

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<all visible>"))
    print("JACCPOT_INDEX_PRECISION:", os.environ.get("JACCPOT_INDEX_PRECISION"))
    print("nvidia-smi physical GPU index:", nvidia_smi_gpu_index)
    return nvidia_smi_gpu_index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--particle-counts",
        type=int,
        nargs="+",
        default=[65536, 131072, 262144, 524288],
    )
    parser.add_argument("--leaf-size", type=int, default=128)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--softening", type=float, default=1e-3)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--theta", type=float, default=0.6)
    parser.add_argument("--peak-poll-interval-s", type=float, default=0.02)
    parser.add_argument(
        "--baseline-profile",
        default="engblom_n_ladder_production",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--emit-markdown", action="store_true")
    parser.add_argument("--use-autocvd", action="store_true")
    parser.add_argument("--autocvd-num-gpus", type=int, default=1)
    parser.add_argument("--autocvd-least-used", action="store_true", default=True)
    parser.add_argument("--autocvd-exclude", nargs="*", default=[])
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--index-precision", default="int32")
    parser.add_argument(
        "--runtime-path",
        choices=("auto", "legacy", "large_n"),
        default="auto",
    )
    return parser.parse_args()


ARGS = _parse_args()
NVIDIA_SMI_GPU_INDEX = _configure_environment(ARGS)

import jax
import jax.numpy as jnp
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jaccpot import (  # noqa: E402
    FMMAdvancedConfig,
    FMMPreset,
    FarFieldConfig,
    FastMultipoleMethod,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)
from yggdrax.interactions import DualTreeTraversalConfig  # noqa: E402

WORKING_DTYPE = getattr(jnp, ARGS.dtype)
OUTPUT_DIR = (
    pathlib.Path(ARGS.output_dir).resolve()
    if ARGS.output_dir is not None
    else REPO_ROOT / "benchmarks" / "n_ladder_production"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAVERSAL_CANDIDATES = [
    {
        "max_pair_queue": 65536,
        "process_block": 64,
        "max_interactions_per_node": 4096,
        "max_neighbors_per_leaf": 1024,
    },
    {
        "max_pair_queue": 131072,
        "process_block": 128,
        "max_interactions_per_node": 8192,
        "max_neighbors_per_leaf": 2048,
    },
    {
        "max_pair_queue": 262144,
        "process_block": 256,
        "max_interactions_per_node": 8192,
        "max_neighbors_per_leaf": 4096,
    },
    {
        "max_pair_queue": 262144,
        "process_block": 256,
        "max_interactions_per_node": 16384,
        "max_neighbors_per_leaf": 4096,
    },
    {
        "max_pair_queue": 524288,
        "process_block": 256,
        "max_interactions_per_node": 16384,
        "max_neighbors_per_leaf": 4096,
    },
]
M2L_CHUNK_CANDIDATES = [1024, 512]
NEARFIELD_CHUNK_CANDIDATES = [128, 64]

TRAVERSAL_GUIDANCE = [
    {
        "particle_limit": 65536,
        "recommended": {
            "max_pair_queue": 65536,
            "process_block": 64,
            "max_interactions_per_node": 4096,
            "max_neighbors_per_leaf": 1024,
        },
        "rationale": (
            "Prepare-state profiling on 2026-03-20 found the lean traversal seed "
            "faster and stable at 65,536 particles on GPU 3."
        ),
    },
    {
        "particle_limit": None,
        "recommended": {
            "max_pair_queue": 262144,
            "process_block": 256,
            "max_interactions_per_node": 8192,
            "max_neighbors_per_leaf": 4096,
        },
        "rationale": (
            "The larger-cap seed remained the best stable choice among tested "
            "configs at 131,072 and 262,144 particles."
        ),
    },
]


def _traversal_cfg_label(traversal_cfg: dict[str, int]) -> str:
    return (
        f"{int(traversal_cfg['max_pair_queue'])}/"
        f"{int(traversal_cfg['process_block'])}/"
        f"{int(traversal_cfg['max_interactions_per_node'])}/"
        f"{int(traversal_cfg['max_neighbors_per_leaf'])}"
    )


def traversal_guidance_for_num_particles(num_particles: int) -> dict[str, Any]:
    for entry in TRAVERSAL_GUIDANCE:
        particle_limit = entry["particle_limit"]
        if particle_limit is None or int(num_particles) <= int(particle_limit):
            return entry
    return TRAVERSAL_GUIDANCE[-1]


def traversal_candidates_for_num_particles(num_particles: int) -> list[dict[str, int]]:
    guidance = traversal_guidance_for_num_particles(int(num_particles))
    preferred_cfg = guidance["recommended"]
    preferred_key = tuple(
        int(preferred_cfg[key])
        for key in (
            "max_pair_queue",
            "process_block",
            "max_interactions_per_node",
            "max_neighbors_per_leaf",
        )
    )
    ordered_candidates: list[dict[str, int]] = [preferred_cfg]
    for candidate in TRAVERSAL_CANDIDATES:
        candidate_key = tuple(
            int(candidate[key])
            for key in (
                "max_pair_queue",
                "process_block",
                "max_interactions_per_node",
                "max_neighbors_per_leaf",
            )
        )
        if candidate_key != preferred_key:
            ordered_candidates.append(candidate)
    return ordered_candidates


def classify_worker_error(message: str) -> str:
    text = str(message).lower().strip()
    if text == "":
        return ""
    if "pair queue capacity exceeded" in text or "interaction capacity exceeded" in text:
        return "interaction_capacity"
    if "neighbor list capacity exceeded" in text:
        return "neighbor_capacity"
    if "resource_exhausted" in text or "out of memory" in text:
        return "oom"
    return "other_error"


def _query_gpu_memory_mb_by_pid(pid: int) -> float:
    cmd = [
        "nvidia-smi",
        f"--id={int(NVIDIA_SMI_GPU_INDEX)}",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    total_used_mb = 0.0
    for line in out.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            row_pid = int(parts[0])
            row_used = float(parts[1])
        except Exception:
            continue
        if row_pid == int(pid):
            total_used_mb += float(row_used)
    return float(total_used_mb)


def _base_fmm_kwargs() -> dict[str, Any]:
    advanced = FMMAdvancedConfig(
        tree=TreeConfig(
            tree_type="radix",
            mode="lbvh",
            leaf_target=64,
            refine_local=False,
            max_refine_levels=0,
            aspect_threshold=16.0,
        ),
        farfield=FarFieldConfig(
            grouped_interactions=False,
            mode="pair_grouped",
            rotation="solidfmm",
            m2l_chunk_size=1024,
            l2l_chunk_size=None,
            streamed_far_pairs=True,
            mixed_order=False,
            mixed_order_min_order=None,
        ),
        nearfield=NearFieldConfig(
            mode="bucketed",
            edge_chunk_size=128,
            precompute_scatter_schedules=False,
        ),
        runtime=RuntimePolicyConfig(
            host_refine_mode="off",
            fail_fast=True,
            jit_tree=True,
            jit_traversal=True,
            memory_objective="minimum_memory",
            traversal_config=DualTreeTraversalConfig(
                max_pair_queue=262144,
                process_block=256,
                max_interactions_per_node=8192,
                max_neighbors_per_leaf=4096,
            ),
            pair_process_block=None,
            enable_interaction_cache=False,
            retain_traversal_result=False,
            retain_interactions=False,
            autotune_m2l_chunk=False,
            upward_leaf_batch_size=2048,
        ),
        mac_type="engblom",
        dehnen_radius_scale=1.0,
    )
    return {
        "preset": FMMPreset.LARGE_N_GPU,
        "basis": "solidfmm",
        "runtime_path": str(ARGS.runtime_path).strip().lower(),
        "precision": "fp32",
        "theta": float(ARGS.theta),
        "softening": float(ARGS.softening),
        "working_dtype": WORKING_DTYPE,
        "adaptive_order": False,
        "advanced": advanced,
    }


def serialize_fmm_kwargs_for_worker(fmm_kwargs: dict[str, Any]) -> dict[str, Any]:
    probe_fmm = FastMultipoleMethod(**fmm_kwargs)
    try:
        advanced = probe_fmm.advanced
        traversal_cfg = advanced.runtime.traversal_config
        traversal_payload = None
        if traversal_cfg is not None:
            traversal_payload = {
                "process_block": int(traversal_cfg.process_block),
                "max_neighbors_per_leaf": int(traversal_cfg.max_neighbors_per_leaf),
                "max_interactions_per_node": int(traversal_cfg.max_interactions_per_node),
                "max_pair_queue": int(traversal_cfg.max_pair_queue),
            }
        return {
            "preset": "large_n_gpu",
            "basis": str(fmm_kwargs.get("basis", "solidfmm")),
            "runtime_path": str(fmm_kwargs.get("runtime_path", "auto")),
            "theta": float(fmm_kwargs.get("theta", 0.6)),
            "softening": float(fmm_kwargs.get("softening", 1e-3)),
            "working_dtype": str(
                jnp.dtype(getattr(probe_fmm._impl, "working_dtype", jnp.float32))
            ),
            "tree_type": str(advanced.tree.tree_type),
            "leaf_target": int(advanced.tree.leaf_target),
            "farfield_rotation": str(advanced.farfield.rotation),
            "farfield_mode": str(advanced.farfield.mode),
            "grouped_interactions": bool(advanced.farfield.grouped_interactions),
            "streamed_far_pairs": advanced.farfield.streamed_far_pairs,
            "mixed_order": bool(advanced.farfield.mixed_order),
            "mixed_order_min_order": advanced.farfield.mixed_order_min_order,
            "nearfield_mode": str(advanced.nearfield.mode),
            "nearfield_edge_chunk_size": int(advanced.nearfield.edge_chunk_size),
            "precompute_scatter_schedules": bool(
                advanced.nearfield.precompute_scatter_schedules
            ),
            "pair_process_block": (
                None
                if advanced.runtime.pair_process_block is None
                else int(advanced.runtime.pair_process_block)
            ),
            "memory_objective": str(advanced.runtime.memory_objective),
            "fail_fast": bool(advanced.runtime.fail_fast),
            "jit_traversal": bool(advanced.runtime.jit_traversal),
            "traversal_config": traversal_payload,
            "enable_interaction_cache": bool(advanced.runtime.enable_interaction_cache),
            "retain_traversal_result": bool(advanced.runtime.retain_traversal_result),
            "retain_interactions": bool(advanced.runtime.retain_interactions),
            "autotune_m2l_chunk": bool(advanced.runtime.autotune_m2l_chunk),
            "adaptive_order": bool(getattr(probe_fmm._impl, "adaptive_order", False)),
            "p_gears": [int(v) for v in getattr(probe_fmm._impl, "p_gears", tuple())],
            "adaptive_error_model": str(
                getattr(probe_fmm._impl, "adaptive_error_model", "tail_proxy")
            ),
            "adaptive_eps": (
                None
                if getattr(probe_fmm._impl, "adaptive_eps", None) is None
                else float(getattr(probe_fmm._impl, "adaptive_eps"))
            ),
            "mac_force_scale_mode": str(
                getattr(probe_fmm._impl, "mac_force_scale_mode", "prev")
            ),
            "mac_type": str(advanced.mac_type),
            "benchmark_scope": "steady_eval",
            "worker_autotune_traversal": False,
            "worker_autotune_nearfield_chunk": False,
            "traversal_candidates": [],
            "nearfield_chunk_candidates": [],
        }
    finally:
        clear_fn = getattr(probe_fmm, "clear_runtime_caches", None)
        if callable(clear_fn):
            clear_fn(clear_jax_compilation=True)
        jax.clear_caches()
        gc.collect()
        del probe_fmm


def with_runtime_knobs(
    base_fmm_kwargs: dict[str, Any],
    *,
    traversal_cfg: dict[str, int],
    m2l_chunk_size: int,
    nearfield_edge_chunk_size: int,
) -> dict[str, Any]:
    advanced = base_fmm_kwargs["advanced"]
    runtime_cfg = replace(
        advanced.runtime,
        traversal_config=DualTreeTraversalConfig(
            max_pair_queue=int(traversal_cfg["max_pair_queue"]),
            process_block=int(traversal_cfg["process_block"]),
            max_interactions_per_node=int(
                traversal_cfg["max_interactions_per_node"]
            ),
            max_neighbors_per_leaf=int(traversal_cfg["max_neighbors_per_leaf"]),
        ),
    )
    farfield_cfg = replace(advanced.farfield, m2l_chunk_size=int(m2l_chunk_size))
    nearfield_cfg = replace(
        advanced.nearfield, edge_chunk_size=int(nearfield_edge_chunk_size)
    )
    out = dict(base_fmm_kwargs)
    out["advanced"] = replace(
        advanced,
        runtime=runtime_cfg,
        farfield=farfield_cfg,
        nearfield=nearfield_cfg,
    )
    return out


def run_worker_case(
    num_particles: int,
    *,
    traversal_cfg: dict[str, int],
    m2l_chunk_size: int,
    nearfield_edge_chunk_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    worker_script = REPO_ROOT / "examples" / "benchmark_gpu_radix_worker.py"
    runtime_kwargs = with_runtime_knobs(
        _base_fmm_kwargs(),
        traversal_cfg=traversal_cfg,
        m2l_chunk_size=m2l_chunk_size,
        nearfield_edge_chunk_size=nearfield_edge_chunk_size,
    )
    payload = serialize_fmm_kwargs_for_worker(runtime_kwargs)
    ready_marker = "__JACCPOT_WORKER_READY__"
    cmd = [
        sys.executable,
        str(worker_script),
        "--mode",
        "sweep",
        "--num-particles",
        str(int(num_particles)),
        "--leaf-size",
        str(int(ARGS.leaf_size)),
        "--max-order",
        str(int(ARGS.max_order)),
        "--runs",
        str(int(ARGS.runs)),
        "--warmup",
        str(int(ARGS.warmup)),
        "--dtype",
        str(jnp.dtype(WORKING_DTYPE)),
        "--seed",
        str(int(ARGS.seed)),
        "--emit-ready-marker",
        "--config-json",
        json.dumps(payload),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    ready_seen = False
    assert proc.stdout is not None
    while proc.poll() is None:
        line = proc.stdout.readline()
        if not line:
            continue
        line = line.strip()
        if not line:
            continue
        if line == ready_marker:
            ready_seen = True
            break
        stdout_lines.append(line)
    samples: list[dict[str, float]] = []
    baseline_used_mb = float(_query_gpu_memory_mb_by_pid(proc.pid)) if ready_seen else 0.0
    while proc.poll() is None:
        timestamp_s = time.perf_counter()
        try:
            used_mb = _query_gpu_memory_mb_by_pid(proc.pid)
            samples.append({"t_s": timestamp_s, "gpu_used_mb": used_mb})
        except Exception:
            pass
        time.sleep(float(ARGS.peak_poll_interval_s))
    stdout, stderr = proc.communicate()
    if stdout:
        stdout_lines.extend(line.strip() for line in stdout.splitlines() if line.strip())
    trace_df = pd.DataFrame(samples)
    peak_used_mb = (
        float(trace_df["gpu_used_mb"].max())
        if not trace_df.empty
        else float(baseline_used_mb)
    )
    row: dict[str, Any] = {
        "num_particles": int(num_particles),
        "max_pair_queue": int(traversal_cfg["max_pair_queue"]),
        "process_block": int(traversal_cfg["process_block"]),
        "max_interactions_per_node": int(
            traversal_cfg["max_interactions_per_node"]
        ),
        "max_neighbors_per_leaf": int(traversal_cfg["max_neighbors_per_leaf"]),
        "m2l_chunk_size": int(m2l_chunk_size),
        "nearfield_edge_chunk_size": int(nearfield_edge_chunk_size),
        "gpu_peak_delta_mb": float(max(0.0, peak_used_mb - baseline_used_mb)),
        "fit_status": "ok",
        "error": "",
    }
    if proc.returncode != 0:
        msg = (stderr or "\n".join(stdout_lines) or "").strip()
        row.update({"fit_status": classify_worker_error(msg), "error": msg})
        return row, trace_df
    lines = [line for line in stdout_lines if line != ready_marker]
    if not lines:
        row.update({"fit_status": "other_error", "error": "worker produced no output"})
        return row, trace_df
    payload_out = json.loads(lines[-1])
    row.update(payload_out)
    row["fit_status"] = (
        "ok"
        if str(payload_out.get("error", "")) == ""
        else classify_worker_error(payload_out.get("error", ""))
    )
    return row, trace_df


def capacity_rank_key(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        [
            "max_pair_queue",
            "max_interactions_per_node",
            "max_neighbors_per_leaf",
            "process_block",
            "prepare_mean_seconds",
            "evaluate_mean_seconds",
        ],
        ascending=[True, True, True, True, True, True],
    )


def summarize_recommendations(ladder_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    stable_df = ladder_df[ladder_df["fit_status"] == "ok"].copy()
    fail_df = ladder_df[ladder_df["fit_status"] != "ok"].copy()

    if stable_df.empty:
        fastest_df = stable_df.copy()
        recommended_df = stable_df.copy()
    else:
        fastest_df = (
            stable_df.assign(
                total_phase_seconds=stable_df["prepare_mean_seconds"]
                + stable_df["evaluate_mean_seconds"]
            )
            .sort_values(
                ["num_particles", "total_phase_seconds", "prepare_mean_seconds"],
                ascending=[True, True, True],
            )
            .groupby("num_particles", as_index=False)
            .first()
        )
        recommended_df = (
            capacity_rank_key(stable_df)
            .groupby("num_particles", as_index=False)
            .first()
            .assign(
                total_phase_seconds=lambda df: df["prepare_mean_seconds"]
                + df["evaluate_mean_seconds"]
            )
        )
    return {
        "stable": stable_df,
        "failures": fail_df,
        "fastest": fastest_df,
        "recommended": recommended_df,
    }


def _write_markdown_summary(
    *,
    path: pathlib.Path,
    args: argparse.Namespace,
    summary: dict[str, pd.DataFrame],
) -> None:
    recommended_df = summary["recommended"]
    fastest_df = summary["fastest"]
    failures_df = summary["failures"]

    lines = [
        "# N-Ladder Production Recommendations",
        "",
        "This file is generated by `examples/benchmark_gpu_n_ladder_production.py`.",
        "",
        "## Sweep Configuration",
        "",
        f"- baseline_profile: `{args.baseline_profile}`",
        f"- particle_counts: `{', '.join(str(v) for v in args.particle_counts)}`",
        f"- leaf_size: `{args.leaf_size}`",
        f"- max_order: `{args.max_order}`",
        f"- theta: `{args.theta}`",
        f"- softening: `{args.softening}`",
        f"- dtype: `{args.dtype}`",
        f"- runs: `{args.runs}`",
        f"- warmup: `{args.warmup}`",
        "",
        "## Traversal Seed Guidance",
        "",
        (
            "The traversal-cap recommendation is size-dependent. Small-`N` "
            "prepare-state profiling favored a leaner seed, while larger `N` "
            "favored the larger-cap stable seed among the tested configs."
        ),
        "",
    ]

    for entry in TRAVERSAL_GUIDANCE:
        particle_limit = entry["particle_limit"]
        target_label = "N > 65536" if particle_limit is None else f"N <= {int(particle_limit)}"
        lines.append(
            (
                f"- {target_label}: "
                f"`{_traversal_cfg_label(entry['recommended'])}`. "
                f"{entry['rationale']}"
            )
        )
    lines.extend(
        [
            "",
            (
                "For the underlying measurements and crossover notes, see "
                "`docs/runtime_traversal_comparison_2026-03-20.md`."
            ),
            "",
        ]
    )

    if recommended_df.empty:
        lines.extend(
            [
                "## Recommended Stable Capacity",
                "",
                "No stable configurations were recorded in the latest sweep yet.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Recommended Stable Capacity",
                "",
                _df_to_markdown_table(
                    recommended_df[
                        [
                            "num_particles",
                            "total_phase_seconds",
                            "prepare_mean_seconds",
                            "evaluate_mean_seconds",
                            "gpu_peak_delta_mb",
                            "max_pair_queue",
                            "process_block",
                            "max_interactions_per_node",
                            "max_neighbors_per_leaf",
                            "m2l_chunk_size",
                            "nearfield_edge_chunk_size",
                        ]
                    ]
                ),
                "",
            ]
        )

    if not fastest_df.empty:
        lines.extend(
            [
                "## Fastest Stable Configuration",
                "",
                _df_to_markdown_table(
                    fastest_df[
                        [
                            "num_particles",
                            "total_phase_seconds",
                            "prepare_mean_seconds",
                            "evaluate_mean_seconds",
                            "max_pair_queue",
                            "process_block",
                            "max_interactions_per_node",
                            "max_neighbors_per_leaf",
                            "m2l_chunk_size",
                            "nearfield_edge_chunk_size",
                        ]
                    ]
                ),
                "",
            ]
        )

    if not failures_df.empty:
        lines.extend(
            [
                "## Failures",
                "",
                _df_to_markdown_table(
                    failures_df[
                        [
                            "num_particles",
                            "fit_status",
                            "max_pair_queue",
                            "process_block",
                            "max_interactions_per_node",
                            "max_neighbors_per_leaf",
                            "m2l_chunk_size",
                            "nearfield_edge_chunk_size",
                        ]
                    ]
                ),
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [len(column) for column in columns]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def _format_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(width) for value, width in zip(values, widths)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([_format_row(columns), separator] + [_format_row(row) for row in rows])


def main() -> None:
    print("JAX backend:", jax.default_backend())
    print("Visible devices:", jax.devices())
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("No GPU visible to JAX.")

    ladder_run_id = time.strftime("%Y%m%d_%H%M%S")
    print(f"Starting N-ladder production sweep run_id={ladder_run_id}")

    sweep_rows: list[dict[str, Any]] = []
    for num_particles in ARGS.particle_counts:
        guidance = traversal_guidance_for_num_particles(int(num_particles))
        preferred_label = _traversal_cfg_label(guidance["recommended"])
        print(
            f"Running N={int(num_particles)} "
            f"(preferred traversal seed {preferred_label})"
        )
        for traversal_cfg in traversal_candidates_for_num_particles(int(num_particles)):
            for m2l_chunk_size in M2L_CHUNK_CANDIDATES:
                for nearfield_edge_chunk_size in NEARFIELD_CHUNK_CANDIDATES:
                    row, _ = run_worker_case(
                        int(num_particles),
                        traversal_cfg=traversal_cfg,
                        m2l_chunk_size=int(m2l_chunk_size),
                        nearfield_edge_chunk_size=int(nearfield_edge_chunk_size),
                    )
                    row["run_id"] = ladder_run_id
                    row["baseline_profile"] = str(ARGS.baseline_profile)
                    sweep_rows.append(row)

    ladder_df = pd.DataFrame(sweep_rows)
    summary = summarize_recommendations(ladder_df)

    csv_run_path = OUTPUT_DIR / f"n_ladder_production_{ladder_run_id}.csv"
    csv_latest_path = OUTPUT_DIR / "n_ladder_production_latest.csv"
    json_latest_path = OUTPUT_DIR / "n_ladder_production_recommendations.json"
    md_latest_path = OUTPUT_DIR / "n_ladder_production_recommendations.md"

    ladder_df.to_csv(csv_run_path, index=False)
    ladder_df.to_csv(csv_latest_path, index=False)

    json_payload = {
        "run_id": ladder_run_id,
        "baseline_profile": str(ARGS.baseline_profile),
        "particle_counts": [int(v) for v in ARGS.particle_counts],
        "leaf_size": int(ARGS.leaf_size),
        "max_order": int(ARGS.max_order),
        "theta": float(ARGS.theta),
        "softening": float(ARGS.softening),
        "dtype": str(ARGS.dtype),
        "runs": int(ARGS.runs),
        "warmup": int(ARGS.warmup),
        "traversal_guidance": [
            {
                "particle_limit": (
                    None
                    if entry["particle_limit"] is None
                    else int(entry["particle_limit"])
                ),
                "recommended": {
                    key: int(value) for key, value in entry["recommended"].items()
                },
                "label": _traversal_cfg_label(entry["recommended"]),
                "rationale": str(entry["rationale"]),
            }
            for entry in TRAVERSAL_GUIDANCE
        ],
        "recommended": summary["recommended"].to_dict(orient="records"),
        "fastest": summary["fastest"].to_dict(orient="records"),
        "failures": summary["failures"].to_dict(orient="records"),
    }
    json_latest_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    _write_markdown_summary(path=md_latest_path, args=ARGS, summary=summary)

    print(f"Wrote sweep rows to {csv_run_path}")
    print(f"Updated {csv_latest_path}")
    print(f"Updated {json_latest_path}")
    print(f"Updated {md_latest_path}")

    if not summary["recommended"].empty:
        print("\nRecommended stable configurations by N:")
        print(
            summary["recommended"][
                [
                    "num_particles",
                    "total_phase_seconds",
                    "gpu_peak_delta_mb",
                    "max_pair_queue",
                    "process_block",
                    "max_interactions_per_node",
                    "max_neighbors_per_leaf",
                    "m2l_chunk_size",
                    "nearfield_edge_chunk_size",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
