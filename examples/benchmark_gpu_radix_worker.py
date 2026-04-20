"""Per-N GPU radix benchmark worker for process-isolated runtime measurements."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Optional


def _configure_worker_environment() -> None:
    """Reduce worker-side CUDA allocator pressure before JAX initializes."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    # The local yggdrax checkout still needs x64 enabled for reliable imports.
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    # Keep the production large-N nearfield path on the current faster default.
    os.environ.setdefault("JACCPOT_LARGE_N_DISABLE_CHUNK_COND", "1")

    gpu_index_raw = os.environ.get("JACCPOT_NVIDIA_SMI_GPU_INDEX")
    if gpu_index_raw is None or str(gpu_index_raw).strip() == "":
        visible_physical_gpus = [
            part.strip()
            for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if part.strip()
        ]
        if visible_physical_gpus:
            os.environ["JACCPOT_NVIDIA_SMI_GPU_INDEX"] = visible_physical_gpus[0]


_configure_worker_environment()

import jax
import jax.numpy as jnp

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import benchmark_utils as bench_utils


def _load_jaccpot_symbols() -> dict[str, Any]:
    from jaccpot import (  # noqa: E402
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        FMMPreset,
        NearFieldConfig,
        RuntimePolicyConfig,
        TreeConfig,
    )

    return {
        "FarFieldConfig": FarFieldConfig,
        "FastMultipoleMethod": FastMultipoleMethod,
        "FMMAdvancedConfig": FMMAdvancedConfig,
        "FMMPreset": FMMPreset,
        "NearFieldConfig": NearFieldConfig,
        "RuntimePolicyConfig": RuntimePolicyConfig,
        "TreeConfig": TreeConfig,
    }


def _load_jaccpot_internal_symbols() -> dict[str, Any]:
    from jaccpot.runtime import _fmm_impl as runtime_impl  # noqa: E402
    from jaccpot.runtime._adaptive_policy import (  # noqa: E402
        adaptive_pair_policy,
        adaptive_policy_tolerance,
    )
    from jaccpot.runtime._interaction_cache import (  # noqa: E402
        _build_dual_tree_artifacts,
    )

    return {
        "_M2L_FULLBATCH_MAX_PAIRS": runtime_impl._M2L_FULLBATCH_MAX_PAIRS,
        "INDEX_DTYPE": runtime_impl.INDEX_DTYPE,
        "_build_nearfield_interop_data": runtime_impl._build_nearfield_interop_data,
        "_accumulate_real_m2l_chunked_scan": runtime_impl._accumulate_real_m2l_chunked_scan,
        "_accumulate_real_m2l_fullbatch": runtime_impl._accumulate_real_m2l_fullbatch,
        "_accumulate_solidfmm_m2l_chunked_scan": runtime_impl._accumulate_solidfmm_m2l_chunked_scan,
        "_accumulate_solidfmm_m2l_fullbatch": runtime_impl._accumulate_solidfmm_m2l_fullbatch,
        "_accumulate_solidfmm_m2l_grouped": runtime_impl._accumulate_solidfmm_m2l_grouped,
        "_accumulate_solidfmm_m2l_grouped_class_major": runtime_impl._accumulate_solidfmm_m2l_grouped_class_major,
        "_build_dual_tree_artifacts": _build_dual_tree_artifacts,
        "_propagate_real_locals_to_children": runtime_impl._propagate_real_locals_to_children,
        "_propagate_solidfmm_locals_to_children": runtime_impl._propagate_solidfmm_locals_to_children,
        "adaptive_pair_policy": adaptive_pair_policy,
        "adaptive_policy_tolerance": adaptive_policy_tolerance,
        "complex_dtype_for_real": runtime_impl.complex_dtype_for_real,
        "complex_to_real_coeffs": runtime_impl.complex_to_real_coeffs,
        "enforce_conjugate_symmetry_batch": runtime_impl.enforce_conjugate_symmetry_batch,
        "sh_size": runtime_impl.sh_size,
    }


def _load_yggdrax_symbols() -> dict[str, Any]:
    from yggdrax import Tree, compute_tree_geometry  # noqa: E402
    from yggdrax.grouped_interactions import build_grouped_interactions  # noqa: E402
    from yggdrax.interactions import (  # noqa: E402
        DualTreeTraversalConfig,
        build_interactions_and_neighbors,
    )

    return {
        "Tree": Tree,
        "build_grouped_interactions": build_grouped_interactions,
        "compute_tree_geometry": compute_tree_geometry,
        "DualTreeTraversalConfig": DualTreeTraversalConfig,
        "build_interactions_and_neighbors": build_interactions_and_neighbors,
    }


def _load_large_n_runtime_symbols() -> dict[str, Any]:
    from jaccpot.runtime._large_n_farfield import evaluate_large_n_farfield  # noqa: E402
    from jaccpot.runtime._large_n_nearfield import evaluate_large_n_nearfield  # noqa: E402
    from jaccpot.runtime._large_n_types import LargeNPreparedState  # noqa: E402

    return {
        "LargeNPreparedState": LargeNPreparedState,
        "evaluate_large_n_farfield": evaluate_large_n_farfield,
        "evaluate_large_n_nearfield": evaluate_large_n_nearfield,
    }


def _load_nearfield_symbols() -> dict[str, Any]:
    from jaccpot.nearfield.near_field import (  # noqa: E402
        INDEX_DTYPE,
        _compact_reduced_pair_bucket_rows,
        _compute_leaf_p2p_prepared_large_n_pairs_only_impl,
        _compute_leaf_p2p_prepared_large_n_self_only_impl,
        _pair_contributions_batched,
        _prepare_leaf_data_from_groups,
        _reduce_pair_bucket_by_target_leaf,
        _scatter_contributions,
        collect_radix_fast_lane_counters,
        prepare_leaf_neighbor_pairs,
    )
    from jaccpot.pallas import (  # noqa: E402
        apply_packed_particle_vector_updates,
        nearfield_tile_pair_accel,
        nearfield_tile_pair_backend,
        nearfield_unique_updates_backend,
        pack_unique_particle_vector_updates,
        pallas_nearfield_tile_pair_supported,
        pallas_nearfield_unique_updates_supported,
    )

    return {
        "INDEX_DTYPE": INDEX_DTYPE,
        "apply_packed_particle_vector_updates": apply_packed_particle_vector_updates,
        "nearfield_tile_pair_accel": nearfield_tile_pair_accel,
        "nearfield_tile_pair_backend": nearfield_tile_pair_backend,
        "_compact_reduced_pair_bucket_rows": _compact_reduced_pair_bucket_rows,
        "_compute_leaf_p2p_prepared_large_n_pairs_only_impl": (
            _compute_leaf_p2p_prepared_large_n_pairs_only_impl
        ),
        "_compute_leaf_p2p_prepared_large_n_self_only_impl": (
            _compute_leaf_p2p_prepared_large_n_self_only_impl
        ),
        "_pair_contributions_batched": _pair_contributions_batched,
        "_prepare_leaf_data_from_groups": _prepare_leaf_data_from_groups,
        "_reduce_pair_bucket_by_target_leaf": _reduce_pair_bucket_by_target_leaf,
        "_scatter_contributions": _scatter_contributions,
        "collect_radix_fast_lane_counters": collect_radix_fast_lane_counters,
        "pallas_nearfield_tile_pair_supported": (
            pallas_nearfield_tile_pair_supported
        ),
        "nearfield_unique_updates_backend": nearfield_unique_updates_backend,
        "pack_unique_particle_vector_updates": pack_unique_particle_vector_updates,
        "pallas_nearfield_unique_updates_supported": (
            pallas_nearfield_unique_updates_supported
        ),
        "prepare_leaf_neighbor_pairs": prepare_leaf_neighbor_pairs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "sweep",
            "audit",
            "nearfield_components",
            "nearfield_components_production",
            "nearfield_fused_check",
            "prepare",
            "peak_prepare",
            "peak_evaluate",
            "tree",
            "interactions",
            "m2l",
            "l2l",
            "downward_trace",
        ),
        required=True,
    )
    parser.add_argument("--num-particles", type=int, required=True)
    parser.add_argument("--leaf-size", type=int, required=True)
    parser.add_argument("--max-order", type=int, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--autotune-cache", default=None)
    parser.add_argument("--emit-ready-marker", action="store_true")
    parser.add_argument("--config-json", required=True)
    return parser.parse_args()


_READY_MARKER = "__JACCPOT_WORKER_READY__"
_EMIT_READY_MARKER = False


@dataclass(frozen=True)
class _DownwardStageInputs:
    tree_artifacts: Any
    interactions: Any
    dense_buffers: Any
    grouped_buffers: Any
    grouped_segment_starts: Any
    grouped_segment_lengths: Any
    grouped_segment_class_ids: Any
    grouped_segment_sort_permutation: Any
    grouped_segment_group_ids: Any
    grouped_segment_unique_targets: Any
    locals_coeffs: Any
    multip_packed_kernel: Any
    centers: Any
    src: Any
    tgt: Any
    grouped_interactions: bool
    basis_mode_norm: str
    rotation_mode: str
    farfield_mode: str
    total_nodes: int
    chunk_size: int
    order: int


def _fresh_downward_stage_inputs(
    stage_inputs: _DownwardStageInputs,
) -> _DownwardStageInputs:
    """Clone donated stage buffers so repeated benchmark calls remain valid."""
    return _DownwardStageInputs(
        tree_artifacts=stage_inputs.tree_artifacts,
        interactions=stage_inputs.interactions,
        dense_buffers=stage_inputs.dense_buffers,
        grouped_buffers=stage_inputs.grouped_buffers,
        grouped_segment_starts=stage_inputs.grouped_segment_starts,
        grouped_segment_lengths=stage_inputs.grouped_segment_lengths,
        grouped_segment_class_ids=stage_inputs.grouped_segment_class_ids,
        grouped_segment_sort_permutation=stage_inputs.grouped_segment_sort_permutation,
        grouped_segment_group_ids=stage_inputs.grouped_segment_group_ids,
        grouped_segment_unique_targets=stage_inputs.grouped_segment_unique_targets,
        locals_coeffs=jnp.array(stage_inputs.locals_coeffs, copy=True),
        multip_packed_kernel=stage_inputs.multip_packed_kernel,
        centers=stage_inputs.centers,
        src=stage_inputs.src,
        tgt=stage_inputs.tgt,
        grouped_interactions=stage_inputs.grouped_interactions,
        basis_mode_norm=stage_inputs.basis_mode_norm,
        rotation_mode=stage_inputs.rotation_mode,
        farfield_mode=stage_inputs.farfield_mode,
        total_nodes=stage_inputs.total_nodes,
        chunk_size=stage_inputs.chunk_size,
        order=stage_inputs.order,
    )


def _dtype_from_name(name: str) -> jnp.dtype:
    if name == "float64":
        return jnp.float64
    return jnp.float32


def _block_ready(value: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        value,
    )


def _emit_ready_marker() -> None:
    if _EMIT_READY_MARKER:
        print(_READY_MARKER, flush=True)


def _array_nbytes_runtime(arr: Any) -> int:
    nbytes = getattr(arr, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    shape = getattr(arr, "shape", None)
    dtype = getattr(arr, "dtype", None)
    if shape is None or dtype is None:
        return 0
    try:
        itemsize = int(jnp.dtype(dtype).itemsize)
    except Exception:
        return 0
    total = int(itemsize)
    for dim in shape:
        total *= int(dim)
    return int(total)


def _prepared_state_total_mb(state: Any) -> float:
    total_bytes = 0
    for leaf in jax.tree_util.tree_leaves(state):
        total_bytes += _array_nbytes_runtime(leaf)
    return float(total_bytes) / (1024**2)


def _resolved_nearfield_runtime_report(fmm: Any, *, num_particles: int) -> dict[str, Any]:
    impl = getattr(fmm, "_impl", None)
    if impl is None:
        return {
            "resolved_nearfield_mode": None,
            "resolved_nearfield_edge_chunk_size": None,
        }
    nearfield_mode = str(
        impl._resolve_nearfield_mode(num_particles=int(num_particles))
    ).strip().lower()
    nearfield_edge_chunk_size = int(
        impl._resolve_nearfield_edge_chunk_size(
            num_particles=int(num_particles),
            nearfield_mode=nearfield_mode,
        )
    )
    return {
        "resolved_nearfield_mode": nearfield_mode,
        "resolved_nearfield_edge_chunk_size": nearfield_edge_chunk_size,
    }


def _query_gpu_memory_mb() -> tuple[float, float]:
    """Query current GPU used/total memory via nvidia-smi."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    gpu_index_raw = os.environ.get("JACCPOT_NVIDIA_SMI_GPU_INDEX")
    if gpu_index_raw is not None and str(gpu_index_raw).strip() != "":
        cmd.insert(1, f"--id={int(gpu_index_raw)}")
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    lines = out.strip().splitlines()
    if len(lines) == 0:
        raise RuntimeError("nvidia-smi returned no output")
    used_mb, total_mb = [float(x.strip()) for x in lines[0].split(",")[:2]]
    return used_mb, total_mb


def _query_gpu_memory_mb_by_pid(pid: int) -> float:
    """Query GPU memory attributed to one process ID via nvidia-smi."""
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    gpu_index_raw = os.environ.get("JACCPOT_NVIDIA_SMI_GPU_INDEX")
    if gpu_index_raw is not None and str(gpu_index_raw).strip() != "":
        cmd.insert(1, f"--id={int(gpu_index_raw)}")
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    total_used_mb = 0.0
    for line in out.strip().splitlines():
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


def _evaluate_prepared_kwargs(fmm: FastMultipoleMethod) -> dict[str, Any]:
    params = set(inspect.signature(fmm.evaluate_prepared_state).parameters)
    if "jit_traversal" in params:
        return {"jit_traversal": True}
    return {}


def _runtime_overrides(
    fmm_kwargs: dict[str, Any],
    *,
    traversal_cfg_dict: Optional[dict[str, int]] = None,
    nearfield_edge_chunk_size: Optional[int] = None,
) -> dict[str, Any]:
    yggdrax_symbols = _load_yggdrax_symbols()
    DualTreeTraversalConfig = yggdrax_symbols["DualTreeTraversalConfig"]
    advanced = fmm_kwargs["advanced"]
    runtime_cfg = advanced.runtime
    nearfield_cfg = advanced.nearfield
    if traversal_cfg_dict is not None:
        traversal_cfg = DualTreeTraversalConfig(
            process_block=int(traversal_cfg_dict["process_block"]),
            max_neighbors_per_leaf=int(traversal_cfg_dict["max_neighbors_per_leaf"]),
            max_interactions_per_node=int(
                traversal_cfg_dict["max_interactions_per_node"]
            ),
            max_pair_queue=int(traversal_cfg_dict["max_pair_queue"]),
        )
        runtime_cfg = replace(runtime_cfg, traversal_config=traversal_cfg)
    if nearfield_edge_chunk_size is not None:
        nearfield_cfg = replace(
            nearfield_cfg,
            edge_chunk_size=int(nearfield_edge_chunk_size),
        )
    out = dict(fmm_kwargs)
    out["advanced"] = replace(advanced, runtime=runtime_cfg, nearfield=nearfield_cfg)
    return out


def _resolved_prepare_context(
    fmm: FastMultipoleMethod,
    *,
    num_particles: int,
) -> dict[str, Any]:
    impl = fmm._impl
    runtime_overrides = impl._resolve_runtime_execution_overrides(
        num_particles=int(num_particles)
    )
    refine_local_val = bool(impl.refine_local)
    if runtime_overrides.refine_local_override is not None:
        refine_local_val = bool(runtime_overrides.refine_local_override)
    return {
        "runtime_overrides": runtime_overrides,
        "runtime_traversal_config": runtime_overrides.traversal_config,
        "runtime_m2l_chunk_size": runtime_overrides.m2l_chunk_size,
        "runtime_l2l_chunk_size": runtime_overrides.l2l_chunk_size,
        "grouped_interactions": runtime_overrides.grouped_interactions,
        "farfield_mode": runtime_overrides.farfield_mode,
        "upward_center_mode": runtime_overrides.center_mode,
        "refine_local_val": refine_local_val,
        "max_refine_levels_val": int(impl.max_refine_levels),
        "aspect_threshold_val": float(impl.aspect_threshold),
        "theta_val": float(impl.theta),
        "mac_type_val": "dehnen" if impl.mac_type == "dehnen_error" else impl.mac_type,
        "dehnen_radius_scale": float(impl.dehnen_radius_scale),
    }


def _worker_traversal_floor(
    *,
    fmm_kwargs: dict[str, Any],
    num_particles: int,
) -> Optional[dict[str, int]]:
    FastMultipoleMethod = _load_jaccpot_symbols()["FastMultipoleMethod"]
    fmm = FastMultipoleMethod(**fmm_kwargs)
    traversal_cfg = _resolved_prepare_context(
        fmm,
        num_particles=int(num_particles),
    )["runtime_traversal_config"]
    if traversal_cfg is None:
        return None
    return {
        "max_pair_queue": int(traversal_cfg.max_pair_queue),
        "process_block": int(traversal_cfg.process_block),
        "max_interactions_per_node": int(traversal_cfg.max_interactions_per_node),
        "max_neighbors_per_leaf": int(traversal_cfg.max_neighbors_per_leaf),
    }


def _normalize_worker_traversal_candidate(
    candidate: dict[str, Any],
    *,
    floor: Optional[dict[str, int]],
) -> dict[str, int]:
    normalized = {
        "max_pair_queue": int(candidate["max_pair_queue"]),
        "process_block": int(candidate["process_block"]),
        "max_interactions_per_node": int(candidate["max_interactions_per_node"]),
        "max_neighbors_per_leaf": int(candidate["max_neighbors_per_leaf"]),
    }
    if floor is None:
        return normalized
    for key, floor_value in floor.items():
        normalized[key] = max(int(normalized[key]), int(floor_value))
    return normalized


def _prepare_tree_upward_artifacts_once(
    *,
    fmm: FastMultipoleMethod,
    positions: Any,
    masses: Any,
    leaf_size: int,
    max_order: int,
) -> tuple[Any, dict[str, Any]]:
    ctx = _resolved_prepare_context(fmm, num_particles=int(positions.shape[0]))
    tree_artifacts = fmm._impl._prepare_state_tree_and_upward(
        positions_arr=positions,
        masses_arr=masses,
        bounds=None,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        refine_local_val=ctx["refine_local_val"],
        max_refine_levels_val=ctx["max_refine_levels_val"],
        aspect_threshold_val=ctx["aspect_threshold_val"],
        jit_tree_override=fmm.advanced.runtime.jit_tree,
        upward_center_mode=ctx["upward_center_mode"],
        allow_stateful_cache=False,
    )
    return tree_artifacts, ctx


def _build_dual_tree_artifacts_once(
    *,
    fmm: FastMultipoleMethod,
    tree_artifacts: Any,
    ctx: dict[str, Any],
) -> Any:
    impl = fmm._impl
    dual_downward_artifacts = impl._prepare_state_dual_and_downward(
        tree_artifacts=tree_artifacts,
        force_scale_nodes=None,
        upward_center_mode=ctx["upward_center_mode"],
        theta_val=ctx["theta_val"],
        mac_type_val=ctx["mac_type_val"],
        dehnen_radius_scale=ctx["dehnen_radius_scale"],
        runtime_traversal_config=ctx["runtime_traversal_config"],
        runtime_m2l_chunk_size=ctx["runtime_m2l_chunk_size"],
        runtime_l2l_chunk_size=ctx["runtime_l2l_chunk_size"],
        grouped_interactions=ctx["grouped_interactions"],
        farfield_mode=ctx["farfield_mode"],
        record_retry=lambda _event: None,
        refine_local_val=ctx["refine_local_val"],
        max_refine_levels_val=ctx["max_refine_levels_val"],
        aspect_threshold_val=ctx["aspect_threshold_val"],
        allow_stateful_cache=False,
    )
    return dual_downward_artifacts


def _build_raw_dual_tree_artifacts_once(
    *,
    fmm: FastMultipoleMethod,
    tree_artifacts: Any,
    ctx: dict[str, Any],
) -> Any:
    internal_symbols = _load_jaccpot_internal_symbols()
    _build_dual_tree_artifacts = internal_symbols["_build_dual_tree_artifacts"]
    adaptive_pair_policy = internal_symbols["adaptive_pair_policy"]
    adaptive_policy_tolerance = internal_symbols["adaptive_policy_tolerance"]
    impl = fmm._impl

    pair_policy = None
    policy_state = None
    use_paper_fixed_policy = (
        not impl.adaptive_order
    ) and impl._uses_dehnen_paper_error_model()
    if impl.adaptive_order or use_paper_fixed_policy:
        policy_orders = impl.p_gears
        if use_paper_fixed_policy:
            policy_orders = (int(tree_artifacts.upward.multipoles.order),)
        policy_state = impl._build_adaptive_policy_state(
            upward=tree_artifacts.upward,
            tree=tree_artifacts.tree,
            positions_sorted=tree_artifacts.positions_sorted,
            p_gears=policy_orders,
            force_scale_nodes=None,
            eps=jnp.asarray(
                (
                    impl.adaptive_eps
                    if impl.adaptive_eps is not None
                    else adaptive_policy_tolerance(
                        theta=ctx["theta_val"],
                        p_gears=policy_orders,
                        dtype=tree_artifacts.upward.multipoles.packed.real.dtype,
                    )
                ),
                dtype=tree_artifacts.upward.multipoles.packed.real.dtype,
            ),
            theta=jnp.asarray(
                ctx["theta_val"],
                dtype=tree_artifacts.upward.multipoles.packed.real.dtype,
            ),
            error_model_code=jnp.asarray(
                impl._adaptive_error_model_code(),
                dtype=jnp.int32,
            ),
            dehnen_geometry_mode=impl.dehnen_geometry_mode,
        )
        pair_policy = adaptive_pair_policy

    need_traversal_result = bool(impl.retain_traversal_result) or bool(
        use_paper_fixed_policy
    )
    need_compact_far_pairs = bool(impl.adaptive_order) and not bool(
        need_traversal_result
    )
    dual_artifacts, _ = _build_dual_tree_artifacts(
        tree_artifacts.tree,
        tree_artifacts.upward.geometry,
        theta=ctx["theta_val"],
        mac_type=ctx["mac_type_val"],
        dehnen_radius_scale=ctx["dehnen_radius_scale"],
        cache_key=None,
        cache_entry=None,
        max_pair_queue=impl.max_pair_queue,
        pair_process_block=impl.pair_process_block,
        traversal_config=ctx["runtime_traversal_config"],
        retry_logger=lambda _event: None,
        fail_fast=impl.fail_fast,
        use_dense_interactions=impl.use_dense_interactions,
        grouped_interactions=ctx["grouped_interactions"],
        grouped_chunk_size=ctx["runtime_m2l_chunk_size"],
        need_traversal_result=need_traversal_result,
        need_compact_far_pairs=need_compact_far_pairs,
        # The audit path needs explicit node-interaction buffers so we can
        # isolate M2L/L2L timing even when the production runtime would stream.
        need_node_interactions=True,
        precompute_grouped_class_segments=impl._should_precompute_grouped_class_segments(
            grouped_chunk_size=ctx["runtime_m2l_chunk_size"],
            farfield_mode=ctx["farfield_mode"],
        ),
        grouped_schedule_budget_bytes=impl._grouped_schedule_item_budget(),
        pair_policy=pair_policy,
        policy_state=policy_state,
    )
    return dual_artifacts


def _prepare_nearfield_artifacts_once(
    *,
    fmm: FastMultipoleMethod,
    tree_artifacts: Any,
    dual_downward_artifacts: Any,
    num_particles: int,
) -> Any:
    internal_symbols = _load_jaccpot_internal_symbols()
    _build_nearfield_interop_data = internal_symbols["_build_nearfield_interop_data"]
    nearfield_interop = _build_nearfield_interop_data(
        tree_artifacts.tree,
        dual_downward_artifacts.neighbor_list,
    )
    return fmm._impl._prepare_state_nearfield_artifacts(
        neighbor_list=dual_downward_artifacts.neighbor_list,
        nearfield_interop=nearfield_interop,
        leaf_cap=tree_artifacts.leaf_cap,
        num_particles=int(num_particles),
        cache_entry=dual_downward_artifacts.cache_entry,
        allow_stateful_cache=False,
    )


def _prepare_downward_stage_inputs_once(
    *,
    fmm: FastMultipoleMethod,
    positions: Any,
    masses: Any,
    leaf_size: int,
    max_order: int,
) -> _DownwardStageInputs:
    internal_symbols = _load_jaccpot_internal_symbols()
    yggdrax_symbols = _load_yggdrax_symbols()
    INDEX_DTYPE = internal_symbols["INDEX_DTYPE"]
    sh_size = internal_symbols["sh_size"]
    complex_dtype_for_real = internal_symbols["complex_dtype_for_real"]
    complex_to_real_coeffs = internal_symbols["complex_to_real_coeffs"]
    build_grouped_interactions = yggdrax_symbols["build_grouped_interactions"]

    impl = fmm._impl
    if impl.adaptive_order:
        raise NotImplementedError(
            "downward stage breakdown does not support adaptive_order yet"
        )

    tree_artifacts, ctx = _prepare_tree_upward_artifacts_once(
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )
    dual_artifacts = _build_raw_dual_tree_artifacts_once(
        fmm=fmm,
        tree_artifacts=tree_artifacts,
        ctx=ctx,
    )
    (
        interactions,
        _neighbor_list,
        _traversal_result,
        _compact_far_pairs,
        dense_buffers,
        grouped_buffers,
        grouped_segment_starts,
        grouped_segment_lengths,
        grouped_segment_class_ids,
        grouped_segment_sort_permutation,
        grouped_segment_group_ids,
        grouped_segment_unique_targets,
    ) = impl._unpack_dual_tree_artifacts(dual_artifacts)

    p = int(tree_artifacts.upward.multipoles.order)
    centers = jnp.asarray(tree_artifacts.upward.multipoles.centers)
    total_nodes = int(centers.shape[0])
    coeff_count = int(sh_size(p))
    basis_mode_norm = str(impl._solidfmm_basis_mode()).strip().lower()
    if basis_mode_norm not in ("complex", "real"):
        raise ValueError("solidfmm basis mode must be complex or real")
    coeff_dtype = (
        complex_dtype_for_real(centers.dtype)
        if basis_mode_norm == "complex"
        else centers.dtype
    )
    if tree_artifacts.locals_template is not None:
        locals_coeffs = jnp.asarray(tree_artifacts.locals_template.coefficients)
    else:
        locals_coeffs = jnp.zeros((total_nodes, coeff_count), dtype=coeff_dtype)

    src = jnp.asarray(interactions.sources, dtype=INDEX_DTYPE)
    tgt = jnp.asarray(interactions.targets, dtype=INDEX_DTYPE)
    multip_packed = jnp.asarray(tree_artifacts.upward.multipoles.packed)
    if basis_mode_norm == "complex":
        multip_packed_kernel = multip_packed.astype(coeff_dtype)
    else:
        multip_packed_kernel = complex_to_real_coeffs(multip_packed, order=p).astype(
            coeff_dtype
        )
    if grouped_buffers is None and bool(ctx["grouped_interactions"]):
        grouped_buffers = build_grouped_interactions(
            tree_artifacts.tree,
            tree_artifacts.upward.geometry,
            interactions,
        )

    return _DownwardStageInputs(
        tree_artifacts=tree_artifacts,
        interactions=interactions,
        dense_buffers=dense_buffers,
        grouped_buffers=grouped_buffers,
        grouped_segment_starts=grouped_segment_starts,
        grouped_segment_lengths=grouped_segment_lengths,
        grouped_segment_class_ids=grouped_segment_class_ids,
        grouped_segment_sort_permutation=grouped_segment_sort_permutation,
        grouped_segment_group_ids=grouped_segment_group_ids,
        grouped_segment_unique_targets=grouped_segment_unique_targets,
        locals_coeffs=locals_coeffs,
        multip_packed_kernel=multip_packed_kernel,
        centers=centers,
        src=src,
        tgt=tgt,
        grouped_interactions=bool(ctx["grouped_interactions"]),
        basis_mode_norm=basis_mode_norm,
        rotation_mode=str(impl.complex_rotation).strip().lower(),
        farfield_mode=str(ctx["farfield_mode"]).strip().lower(),
        total_nodes=total_nodes,
        chunk_size=(
            4096
            if ctx["runtime_m2l_chunk_size"] is None
            else int(ctx["runtime_m2l_chunk_size"])
        ),
        order=p,
    )


def _run_m2l_stage_once(
    *,
    fmm: FastMultipoleMethod,
    stage_inputs: _DownwardStageInputs,
) -> Any:
    internal_symbols = _load_jaccpot_internal_symbols()
    M2L_FULLBATCH_MAX_PAIRS = int(internal_symbols["_M2L_FULLBATCH_MAX_PAIRS"])
    acc_grouped = internal_symbols["_accumulate_solidfmm_m2l_grouped"]
    acc_grouped_class_major = internal_symbols[
        "_accumulate_solidfmm_m2l_grouped_class_major"
    ]
    acc_complex_fullbatch = internal_symbols["_accumulate_solidfmm_m2l_fullbatch"]
    acc_complex_chunked = internal_symbols["_accumulate_solidfmm_m2l_chunked_scan"]
    acc_real_fullbatch = internal_symbols["_accumulate_real_m2l_fullbatch"]
    acc_real_chunked = internal_symbols["_accumulate_real_m2l_chunked_scan"]

    impl = fmm._impl
    pair_count = int(stage_inputs.src.shape[0])
    chunk_size = int(stage_inputs.chunk_size)
    if chunk_size <= 0:
        raise ValueError("m2l_chunk_size must be positive")

    if stage_inputs.basis_mode_norm == "complex" and stage_inputs.grouped_interactions:
        if stage_inputs.farfield_mode == "class_major":
            return acc_grouped_class_major(
                stage_inputs.locals_coeffs,
                stage_inputs.multip_packed_kernel,
                stage_inputs.centers,
                stage_inputs.grouped_buffers,
                grouped_segment_starts=stage_inputs.grouped_segment_starts,
                grouped_segment_lengths=stage_inputs.grouped_segment_lengths,
                grouped_segment_class_ids=stage_inputs.grouped_segment_class_ids,
                grouped_segment_sort_permutation=stage_inputs.grouped_segment_sort_permutation,
                grouped_segment_group_ids=stage_inputs.grouped_segment_group_ids,
                grouped_segment_unique_targets=stage_inputs.grouped_segment_unique_targets,
                order=int(stage_inputs.order),
                rotation=stage_inputs.rotation_mode,
                total_nodes=int(stage_inputs.total_nodes),
                chunk_size=chunk_size,
            )
        return acc_grouped(
            stage_inputs.locals_coeffs,
            stage_inputs.multip_packed_kernel,
            stage_inputs.centers,
            stage_inputs.grouped_buffers,
            order=int(stage_inputs.order),
            rotation=stage_inputs.rotation_mode,
            total_nodes=int(stage_inputs.total_nodes),
            chunk_size=chunk_size,
        )
    if pair_count == 0:
        return stage_inputs.locals_coeffs
    if stage_inputs.basis_mode_norm == "complex":
        if pair_count <= min(chunk_size, M2L_FULLBATCH_MAX_PAIRS):
            return acc_complex_fullbatch(
                stage_inputs.locals_coeffs,
                stage_inputs.multip_packed_kernel,
                stage_inputs.centers,
                stage_inputs.src,
                stage_inputs.tgt,
                order=int(stage_inputs.order),
                rotation=stage_inputs.rotation_mode,
                total_nodes=int(stage_inputs.total_nodes),
            )
        return acc_complex_chunked(
            stage_inputs.locals_coeffs,
            stage_inputs.multip_packed_kernel,
            stage_inputs.centers,
            stage_inputs.src,
            stage_inputs.tgt,
            order=int(stage_inputs.order),
            rotation=stage_inputs.rotation_mode,
            total_nodes=int(stage_inputs.total_nodes),
            chunk_size=chunk_size,
        )
    if pair_count <= min(chunk_size, M2L_FULLBATCH_MAX_PAIRS):
        return acc_real_fullbatch(
            stage_inputs.locals_coeffs,
            stage_inputs.multip_packed_kernel,
            stage_inputs.centers,
            stage_inputs.src,
            stage_inputs.tgt,
            order=int(stage_inputs.order),
            m2l_impl=("rot_scale" if impl.m2l_impl is None else str(impl.m2l_impl)),
            total_nodes=int(stage_inputs.total_nodes),
        )
    return acc_real_chunked(
        stage_inputs.locals_coeffs,
        stage_inputs.multip_packed_kernel,
        stage_inputs.centers,
        stage_inputs.src,
        stage_inputs.tgt,
        order=int(stage_inputs.order),
        m2l_impl=("rot_scale" if impl.m2l_impl is None else str(impl.m2l_impl)),
        total_nodes=int(stage_inputs.total_nodes),
        chunk_size=chunk_size,
    )


def _run_l2l_stage_once(
    *,
    fmm: FastMultipoleMethod,
    stage_inputs: _DownwardStageInputs,
) -> Any:
    internal_symbols = _load_jaccpot_internal_symbols()
    enforce_conjugate_symmetry_batch = internal_symbols[
        "enforce_conjugate_symmetry_batch"
    ]
    propagate_complex = internal_symbols["_propagate_solidfmm_locals_to_children"]
    propagate_real = internal_symbols["_propagate_real_locals_to_children"]

    tree = stage_inputs.tree_artifacts.tree
    locals_after_m2l = _run_m2l_stage_once(
        fmm=fmm,
        stage_inputs=stage_inputs,
    )
    if stage_inputs.basis_mode_norm == "complex":
        locals_after_m2l = enforce_conjugate_symmetry_batch(
            locals_after_m2l,
            order=int(stage_inputs.order),
        )
    num_internal_nodes = int(tree.num_internal_nodes)
    if num_internal_nodes == 0:
        return locals_after_m2l
    left_child = jnp.asarray(tree.left_child[:num_internal_nodes], dtype=jnp.int32)
    right_child = jnp.asarray(tree.right_child[:num_internal_nodes], dtype=jnp.int32)
    if stage_inputs.basis_mode_norm == "complex":
        return propagate_complex(
            locals_after_m2l,
            stage_inputs.centers,
            left_child,
            right_child,
            order=int(stage_inputs.order),
            rotation=stage_inputs.rotation_mode,
            total_nodes=int(stage_inputs.total_nodes),
        )
    return propagate_real(
        locals_after_m2l,
        stage_inputs.centers,
        left_child,
        right_child,
        order=int(stage_inputs.order),
        total_nodes=int(stage_inputs.total_nodes),
    )


def _run_l2l_propagation_once(
    *,
    stage_inputs: _DownwardStageInputs,
    locals_after_m2l: Any,
) -> Any:
    internal_symbols = _load_jaccpot_internal_symbols()
    enforce_conjugate_symmetry_batch = internal_symbols[
        "enforce_conjugate_symmetry_batch"
    ]
    propagate_complex = internal_symbols["_propagate_solidfmm_locals_to_children"]
    propagate_real = internal_symbols["_propagate_real_locals_to_children"]

    tree = stage_inputs.tree_artifacts.tree
    locals_ready = jnp.array(locals_after_m2l, copy=True)
    if stage_inputs.basis_mode_norm == "complex":
        locals_ready = enforce_conjugate_symmetry_batch(
            locals_ready,
            order=int(stage_inputs.order),
        )
    num_internal_nodes = int(tree.num_internal_nodes)
    if num_internal_nodes == 0:
        return locals_ready
    left_child = jnp.asarray(tree.left_child[:num_internal_nodes], dtype=jnp.int32)
    right_child = jnp.asarray(tree.right_child[:num_internal_nodes], dtype=jnp.int32)
    if stage_inputs.basis_mode_norm == "complex":
        return propagate_complex(
            locals_ready,
            stage_inputs.centers,
            left_child,
            right_child,
            order=int(stage_inputs.order),
            rotation=stage_inputs.rotation_mode,
            total_nodes=int(stage_inputs.total_nodes),
        )
    return propagate_real(
        locals_ready,
        stage_inputs.centers,
        left_child,
        right_child,
        order=int(stage_inputs.order),
        total_nodes=int(stage_inputs.total_nodes),
    )


def _run_m2l_stage_fresh(
    *,
    fmm: FastMultipoleMethod,
    stage_inputs: _DownwardStageInputs,
) -> Any:
    return _run_m2l_stage_once(
        fmm=fmm,
        stage_inputs=_fresh_downward_stage_inputs(stage_inputs),
    )


def _run_l2l_stage_fresh(
    *,
    fmm: FastMultipoleMethod,
    stage_inputs: _DownwardStageInputs,
) -> Any:
    return _run_l2l_stage_once(
        fmm=fmm,
        stage_inputs=_fresh_downward_stage_inputs(stage_inputs),
    )


def _nearest_used_mb(samples: list[dict[str, float]], t_s: float) -> float:
    if len(samples) == 0:
        return float("nan")
    best = min(samples, key=lambda row: abs(float(row["t_s"]) - float(t_s)))
    return float(best["gpu_used_mb"])


def _segment_peak_used_mb(
    samples: list[dict[str, float]],
    *,
    t_start: float,
    t_end: float,
) -> float:
    in_window = [
        float(row["gpu_used_mb"])
        for row in samples
        if float(t_start) <= float(row["t_s"]) <= float(t_end)
    ]
    if len(in_window) == 0:
        return _nearest_used_mb(samples, t_start)
    return max(in_window)


def _run_downward_trace_case(
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    dtype: jnp.dtype,
    seed: int,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    FastMultipoleMethod = _load_jaccpot_symbols()["FastMultipoleMethod"]
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    tuned_kwargs, worker_tune_info = _worker_autotune_runtime_kwargs(
        cfg=cfg,
        fmm_kwargs=fmm_kwargs,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        autotune_cache_path=autotune_cache_path,
    )
    fmm = FastMultipoleMethod(**tuned_kwargs)
    stage_inputs = _prepare_downward_stage_inputs_once(
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )

    # Warm both kernels before measuring segmented runtime memory.
    warm_m2l = _run_m2l_stage_fresh(fmm=fmm, stage_inputs=stage_inputs)
    _block_ready(warm_m2l)
    _block_ready(
        _run_l2l_propagation_once(
            stage_inputs=stage_inputs,
            locals_after_m2l=warm_m2l,
        )
    )

    samples: list[dict[str, float]] = []
    stop_event = threading.Event()
    poll_interval_s = 0.02
    worker_pid = int(os.getpid())

    def _poll() -> None:
        while not stop_event.is_set():
            ts = time.perf_counter()
            try:
                used_mb = _query_gpu_memory_mb_by_pid(worker_pid)
                samples.append(
                    {
                        "t_s": float(ts),
                        "gpu_used_mb": float(used_mb),
                    }
                )
            except Exception:
                break
            stop_event.wait(poll_interval_s)

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()
    marks: dict[str, float] = {}
    try:
        marks["ready"] = time.perf_counter()
        m2l_out = _run_m2l_stage_fresh(fmm=fmm, stage_inputs=stage_inputs)
        marks["m2l_end"] = time.perf_counter()
        _block_ready(m2l_out)
        marks["m2l_ready"] = time.perf_counter()
        l2l_out = _run_l2l_propagation_once(
            stage_inputs=stage_inputs,
            locals_after_m2l=m2l_out,
        )
        marks["l2l_end"] = time.perf_counter()
        _block_ready(l2l_out)
        marks["l2l_ready"] = time.perf_counter()
    finally:
        stop_event.set()
        poller.join(timeout=1.0)

    m2l_before = _nearest_used_mb(samples, marks["ready"])
    m2l_peak = _segment_peak_used_mb(
        samples,
        t_start=marks["ready"],
        t_end=marks["m2l_ready"],
    )
    l2l_before = _nearest_used_mb(samples, marks["m2l_ready"])
    l2l_peak = _segment_peak_used_mb(
        samples,
        t_start=marks["m2l_ready"],
        t_end=marks["l2l_ready"],
    )

    rows = []
    for component, before, peak, t0, t1 in (
        ("m2l", m2l_before, m2l_peak, marks["ready"], marks["m2l_ready"]),
        ("l2l", l2l_before, l2l_peak, marks["m2l_ready"], marks["l2l_ready"]),
    ):
        row = {
            "component": component,
            "gpu_used_before_mb": float(before),
            "gpu_used_after_mb": float(_nearest_used_mb(samples, t1)),
            "gpu_peak_used_mb": float(peak),
            "gpu_peak_delta_mb": float(peak - before),
            "wall_seconds": float(t1 - t0),
            "error": "",
            "error_type": "",
            "num_particles": int(num_particles),
            "mean_seconds": float(t1 - t0),
            "std_seconds": 0.0,
        }
        row.update(worker_tune_info)
        rows.append(row)
    return {"component_rows": rows}


def _device_autotune_signature(
    *,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    num_particles: int,
    max_order: int,
    dtype: Any,
) -> str:
    """Build a stable key for worker-side runtime autotune reuse."""
    try:
        dev = jax.devices()[0]
        device_name = str(getattr(dev, "device_kind", getattr(dev, "platform", "cpu")))
    except Exception:
        device_name = "unknown"
    payload = {
        "device": device_name,
        "platform": jax.default_backend(),
        "index_precision": os.environ.get("JACCPOT_INDEX_PRECISION", "int64"),
        "dtype": str(jnp.dtype(dtype)),
        "preset": str(cfg.get("preset", "")),
        "basis": str(cfg.get("basis", "")),
        "theta": float(cfg.get("theta", 0.6)),
        "adaptive_order": bool(cfg.get("adaptive_order", False)),
        "max_order": int(max_order),
        "num_particles": int(num_particles),
        "tree_type": str(cfg.get("tree_type", "")),
        "farfield_mode": str(cfg.get("farfield_mode", "")),
        "benchmark_scope": str(cfg.get("benchmark_scope", "steady_eval")).strip().lower(),
        "worker_autotune_objective": str(
            cfg.get("worker_autotune_objective", "")
        ).strip().lower(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def _runtime_autotune_cache_path(
    *,
    cfg: dict[str, Any],
    autotune_cache_path: Optional[str],
) -> Optional[pathlib.Path]:
    """Resolve worker runtime autotune cache path."""
    raw = cfg.get("runtime_autotune_cache_path")
    if raw is None and autotune_cache_path:
        base = pathlib.Path(str(autotune_cache_path))
        raw = str(base.with_name("runtime_worker_autotune_cache.json"))
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    return pathlib.Path(text)


def _load_runtime_autotune_entry(
    *,
    path: Optional[pathlib.Path],
    signature: str,
) -> Optional[dict[str, Any]]:
    """Load one runtime-autotune cache entry."""
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, dict):
        return None
    entry = entries.get(signature)
    return entry if isinstance(entry, dict) else None


def _save_runtime_autotune_entry(
    *,
    path: Optional[pathlib.Path],
    signature: str,
    traversal_cfg: Optional[dict[str, int]],
    nearfield_edge_chunk_size: Optional[int],
) -> None:
    """Persist one runtime-autotune cache entry."""
    if path is None:
        return
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {}
    entries = payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        payload["entries"] = entries
    entries[signature] = {
        "worker_traversal_config": traversal_cfg,
        "worker_nearfield_edge_chunk_size": (
            None
            if nearfield_edge_chunk_size is None
            else int(nearfield_edge_chunk_size)
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _measure_prepare_once(
    *,
    fmm_kwargs: dict[str, Any],
    positions: Any,
    masses: Any,
    leaf_size: int,
    max_order: int,
    autotune_cache_path: Optional[str],
) -> float:
    jaccpot_symbols = _load_jaccpot_symbols()
    FastMultipoleMethod = jaccpot_symbols["FastMultipoleMethod"]
    fmm = FastMultipoleMethod(**fmm_kwargs)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        if cache_path.exists():
            fmm.load_m2l_autotune_cache(str(cache_path), merge=True)
    t0 = time.perf_counter()
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )
    _ = _block_ready(state)
    dt = float(time.perf_counter() - t0)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fmm.save_m2l_autotune_cache(str(cache_path))
    return dt


def _measure_runtime_once(
    *,
    fmm_kwargs: dict[str, Any],
    positions: Any,
    masses: Any,
    leaf_size: int,
    max_order: int,
    autotune_cache_path: Optional[str],
    benchmark_scope: str,
) -> float:
    jaccpot_symbols = _load_jaccpot_symbols()
    FastMultipoleMethod = jaccpot_symbols["FastMultipoleMethod"]
    fmm = FastMultipoleMethod(**fmm_kwargs)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        if cache_path.exists():
            fmm.load_m2l_autotune_cache(str(cache_path), merge=True)

    scope = str(benchmark_scope).strip().lower()
    if scope not in ("steady_eval", "full"):
        scope = "steady_eval"

    _warm_sweep_case(
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        benchmark_scope=scope,
    )

    if scope == "full":
        t0 = time.perf_counter()
        out = fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            reuse_prepared_state=False,
        )
        _ = _block_ready(out)
        dt = float(time.perf_counter() - t0)
    else:
        state = fmm.prepare_state(
            positions,
            masses,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
        )
        state = _block_ready(state)
        t0 = time.perf_counter()
        out = fmm.evaluate_prepared_state(
            state,
            **_evaluate_prepared_kwargs(fmm),
        )
        _ = _block_ready(out)
        dt = float(time.perf_counter() - t0)

    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fmm.save_m2l_autotune_cache(str(cache_path))
    return dt


def _warm_tree_case(
    *,
    positions: Any,
    masses: Any,
    leaf_size: int,
) -> None:
    Tree = _load_yggdrax_symbols()["Tree"]
    _block_ready(
        Tree.from_particles(
            positions,
            masses,
            tree_type="radix",
            build_mode="adaptive",
            return_reordered=True,
            leaf_size=int(leaf_size),
        )
    )


def _warm_interactions_case(
    *,
    fmm: FastMultipoleMethod,
    positions: Any,
    masses: Any,
    leaf_size: int,
    max_order: int,
    num_particles: int,
) -> None:
    tree_artifacts, ctx = _prepare_tree_upward_artifacts_once(
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )
    dual_downward_artifacts = _build_dual_tree_artifacts_once(
        fmm=fmm,
        tree_artifacts=tree_artifacts,
        ctx=ctx,
    )
    _block_ready(dual_downward_artifacts.downward.locals.coefficients)
    _block_ready(
        _prepare_nearfield_artifacts_once(
            fmm=fmm,
            tree_artifacts=tree_artifacts,
            dual_downward_artifacts=dual_downward_artifacts,
            num_particles=int(num_particles),
        )
    )


def _warm_prepare_case(
    *,
    fmm: FastMultipoleMethod,
    positions: Any,
    masses: Any,
    leaf_size: int,
    max_order: int,
) -> None:
    _block_ready(
        fmm.prepare_state(
            positions,
            masses,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
        )
    )


def _warm_sweep_case(
    *,
    fmm: FastMultipoleMethod,
    positions: Any,
    masses: Any,
    leaf_size: int,
    max_order: int,
    benchmark_scope: str,
) -> None:
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )
    state = _block_ready(state)
    _block_ready(
        fmm.evaluate_prepared_state(
            state,
            **_evaluate_prepared_kwargs(fmm),
        )
    )
    if benchmark_scope == "full":
        _block_ready(
            fmm.compute_accelerations(
                positions,
                masses,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                reuse_prepared_state=False,
            )
        )


def _worker_autotune_runtime_kwargs(
    *,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    positions: Any,
    masses: Any,
    leaf_size: int,
    max_order: int,
    autotune_cache_path: Optional[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    tuned_kwargs = dict(fmm_kwargs)
    autotune_default = str(cfg.get("preset", "")).strip().lower() == "large_n_gpu"
    autotune_traversal_enabled = bool(
        cfg.get("worker_autotune_traversal", autotune_default)
    )
    autotune_nearfield_enabled = bool(
        cfg.get("worker_autotune_nearfield_chunk", autotune_default)
    )
    benchmark_scope = str(cfg.get("benchmark_scope", "steady_eval")).strip().lower()
    if benchmark_scope not in ("steady_eval", "full"):
        benchmark_scope = "steady_eval"
    autotune_objective = str(
        cfg.get("worker_autotune_objective", "")
    ).strip().lower()
    if autotune_objective not in ("prepare", "steady_eval", "full"):
        if autotune_default:
            autotune_objective = benchmark_scope
        else:
            autotune_objective = "prepare"
    runtime_cache_path = _runtime_autotune_cache_path(
        cfg=cfg,
        autotune_cache_path=autotune_cache_path,
    )
    if not autotune_traversal_enabled and not autotune_nearfield_enabled:
        runtime_cache_path = None
    signature = _device_autotune_signature(
        cfg=cfg,
        fmm_kwargs=tuned_kwargs,
        num_particles=int(positions.shape[0]),
        max_order=int(max_order),
        dtype=positions.dtype,
    )
    info: dict[str, Any] = {
        "worker_traversal_config": None,
        "worker_nearfield_edge_chunk_size": None,
        "worker_autotune_objective": autotune_objective,
    }
    traversal_floor = _worker_traversal_floor(
        fmm_kwargs=tuned_kwargs,
        num_particles=int(positions.shape[0]),
    )
    cached_entry = _load_runtime_autotune_entry(
        path=runtime_cache_path,
        signature=signature,
    )
    if isinstance(cached_entry, dict):
        cached_traversal = cached_entry.get("worker_traversal_config")
        cached_nf = cached_entry.get("worker_nearfield_edge_chunk_size")
        if isinstance(cached_traversal, dict):
            try:
                normalized_cached = _normalize_worker_traversal_candidate(
                    cached_traversal,
                    floor=traversal_floor,
                )
                tuned_kwargs = _runtime_overrides(
                    tuned_kwargs,
                    traversal_cfg_dict=normalized_cached,
                )
                info["worker_traversal_config"] = normalized_cached
            except Exception:
                pass
        if cached_nf is not None:
            try:
                tuned_kwargs = _runtime_overrides(
                    tuned_kwargs,
                    nearfield_edge_chunk_size=int(cached_nf),
                )
                info["worker_nearfield_edge_chunk_size"] = int(cached_nf)
            except Exception:
                pass
        if (
            info["worker_traversal_config"] is not None
            or info["worker_nearfield_edge_chunk_size"] is not None
        ):
            return tuned_kwargs, info
    traversal_candidates_raw = cfg.get("traversal_candidates")
    if not isinstance(traversal_candidates_raw, list):
        traversal_candidates_raw = []
    if len(traversal_candidates_raw) == 0 and autotune_default:
        baseline_fmm = _load_jaccpot_symbols()["FastMultipoleMethod"](**tuned_kwargs)
        baseline_overrides = _resolved_prepare_context(
            baseline_fmm,
            num_particles=int(positions.shape[0]),
        )
        baseline_cfg = baseline_overrides["runtime_traversal_config"]
        if baseline_cfg is not None:
            traversal_candidates_raw = [
                {
                    "max_pair_queue": int(baseline_cfg.max_pair_queue),
                    "process_block": int(baseline_cfg.process_block),
                    "max_interactions_per_node": int(
                        baseline_cfg.max_interactions_per_node
                    ),
                    "max_neighbors_per_leaf": int(baseline_cfg.max_neighbors_per_leaf),
                }
            ]
        else:
            traversal_candidates_raw = []
    if autotune_traversal_enabled and isinstance(traversal_candidates_raw, list):
        best_time = float("inf")
        best_cfg: Optional[dict[str, int]] = None
        baseline_floor: Optional[dict[str, int]] = traversal_floor
        if len(traversal_candidates_raw) > 0 and isinstance(
            traversal_candidates_raw[0], dict
        ):
            try:
                candidate_floor = _normalize_worker_traversal_candidate(
                    traversal_candidates_raw[0],
                    floor=traversal_floor,
                )
                if baseline_floor is None:
                    baseline_floor = candidate_floor
                else:
                    baseline_floor = {
                        key: max(
                            int(baseline_floor[key]),
                            int(candidate_floor[key]),
                        )
                        for key in baseline_floor
                    }
            except Exception:
                pass
        for candidate in traversal_candidates_raw:
            if not isinstance(candidate, dict):
                continue
            try:
                normalized_candidate = _normalize_worker_traversal_candidate(
                    candidate,
                    floor=traversal_floor,
                )
                if baseline_floor is not None:
                    if (
                        int(normalized_candidate["max_pair_queue"])
                        < baseline_floor["max_pair_queue"]
                    ):
                        continue
                    if (
                        int(normalized_candidate["max_interactions_per_node"])
                        < baseline_floor["max_interactions_per_node"]
                    ):
                        continue
                    if (
                        int(normalized_candidate["max_neighbors_per_leaf"])
                        < baseline_floor["max_neighbors_per_leaf"]
                    ):
                        continue
                trial_kwargs = _runtime_overrides(
                    tuned_kwargs,
                    traversal_cfg_dict=normalized_candidate,
                )
                if autotune_objective == "prepare":
                    t = _measure_prepare_once(
                        fmm_kwargs=trial_kwargs,
                        positions=positions,
                        masses=masses,
                        leaf_size=int(leaf_size),
                        max_order=int(max_order),
                        autotune_cache_path=autotune_cache_path,
                    )
                else:
                    t = _measure_runtime_once(
                        fmm_kwargs=trial_kwargs,
                        positions=positions,
                        masses=masses,
                        leaf_size=int(leaf_size),
                        max_order=int(max_order),
                        autotune_cache_path=autotune_cache_path,
                        benchmark_scope=autotune_objective,
                    )
                if t < best_time:
                    best_time = t
                    best_cfg = normalized_candidate
            except Exception:
                continue
        if best_cfg is not None:
            tuned_kwargs = _runtime_overrides(tuned_kwargs, traversal_cfg_dict=best_cfg)
            info["worker_traversal_config"] = best_cfg

    nf_candidates_raw = cfg.get("nearfield_chunk_candidates")
    if not isinstance(nf_candidates_raw, list):
        nf_candidates_raw = []
    if len(nf_candidates_raw) == 0 and autotune_default:
        nf_candidates_raw = [64, 128, 256, 512]
    if autotune_nearfield_enabled and isinstance(nf_candidates_raw, list):
        best_time = float("inf")
        best_nf: Optional[int] = None
        for candidate in nf_candidates_raw:
            try:
                candidate_nf = int(candidate)
            except Exception:
                continue
            if candidate_nf <= 0:
                continue
            try:
                trial_kwargs = _runtime_overrides(
                    tuned_kwargs,
                    nearfield_edge_chunk_size=candidate_nf,
                )
                if autotune_objective == "prepare":
                    t = _measure_prepare_once(
                        fmm_kwargs=trial_kwargs,
                        positions=positions,
                        masses=masses,
                        leaf_size=int(leaf_size),
                        max_order=int(max_order),
                        autotune_cache_path=autotune_cache_path,
                    )
                else:
                    t = _measure_runtime_once(
                        fmm_kwargs=trial_kwargs,
                        positions=positions,
                        masses=masses,
                        leaf_size=int(leaf_size),
                        max_order=int(max_order),
                        autotune_cache_path=autotune_cache_path,
                        benchmark_scope=autotune_objective,
                    )
                if t < best_time:
                    best_time = t
                    best_nf = candidate_nf
            except Exception:
                continue
        if best_nf is not None:
            tuned_kwargs = _runtime_overrides(
                tuned_kwargs,
                nearfield_edge_chunk_size=int(best_nf),
            )
            info["worker_nearfield_edge_chunk_size"] = int(best_nf)

    _save_runtime_autotune_entry(
        path=runtime_cache_path,
        signature=signature,
        traversal_cfg=info["worker_traversal_config"],
        nearfield_edge_chunk_size=info["worker_nearfield_edge_chunk_size"],
    )
    return tuned_kwargs, info


def _build_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    jaccpot_symbols = _load_jaccpot_symbols()
    yggdrax_symbols = _load_yggdrax_symbols()
    FarFieldConfig = jaccpot_symbols["FarFieldConfig"]
    FMMPreset = jaccpot_symbols["FMMPreset"]
    NearFieldConfig = jaccpot_symbols["NearFieldConfig"]
    RuntimePolicyConfig = jaccpot_symbols["RuntimePolicyConfig"]
    TreeConfig = jaccpot_symbols["TreeConfig"]
    FMMAdvancedConfig = jaccpot_symbols["FMMAdvancedConfig"]
    DualTreeTraversalConfig = yggdrax_symbols["DualTreeTraversalConfig"]
    preset_norm = str(config.get("preset", "fast")).strip().lower()
    autotune_default = preset_norm == "large_n_gpu"
    memory_objective = str(config.get("memory_objective", "balanced")).strip().lower()
    traversal_raw = config.get("traversal_config")
    if traversal_raw is None:
        # Older benchmark notes and handoff commands used this name.
        traversal_raw = config.get("runtime_traversal_config")
    traversal_cfg: Optional[DualTreeTraversalConfig]
    if traversal_raw is None:
        traversal_cfg = None
    else:
        traversal_cfg = DualTreeTraversalConfig(
            process_block=int(traversal_raw["process_block"]),
            max_neighbors_per_leaf=int(traversal_raw["max_neighbors_per_leaf"]),
            max_interactions_per_node=int(traversal_raw["max_interactions_per_node"]),
            max_pair_queue=int(traversal_raw["max_pair_queue"]),
        )

    advanced = FMMAdvancedConfig(
        tree=TreeConfig(
            tree_type=str(config["tree_type"]),
            leaf_target=int(config["leaf_target"]),
        ),
        farfield=FarFieldConfig(
            rotation=str(config.get("farfield_rotation", "solidfmm")),
            mode=str(config.get("farfield_mode", "auto")),
            grouped_interactions=bool(config.get("grouped_interactions", False)),
            streamed_far_pairs=config.get("streamed_far_pairs"),
            mixed_order=bool(config.get("mixed_order", False)),
            mixed_order_min_order=(
                None
                if config.get("mixed_order_min_order") is None
                else int(config["mixed_order_min_order"])
            ),
        ),
        nearfield=NearFieldConfig(
            mode=str(config.get("nearfield_mode", "auto")),
            edge_chunk_size=int(config.get("nearfield_edge_chunk_size", 256)),
            precompute_scatter_schedules=bool(
                config.get("precompute_scatter_schedules", True)
            ),
        ),
        runtime=RuntimePolicyConfig(
            fail_fast=bool(config.get("fail_fast", False)),
            pair_process_block=(
                None
                if config.get("pair_process_block") is None
                else int(config["pair_process_block"])
            ),
            memory_objective=str(memory_objective),
            traversal_config=traversal_cfg,
            jit_traversal=bool(config.get("jit_traversal", True)),
            enable_interaction_cache=bool(config.get("enable_interaction_cache", True)),
            retain_traversal_result=bool(config.get("retain_traversal_result", True)),
            retain_interactions=bool(config.get("retain_interactions", True)),
            autotune_m2l_chunk=bool(config.get("autotune_m2l_chunk", autotune_default)),
        ),
        mac_type=str(config.get("mac_type", "dehnen")),
    )
    return dict(
        preset=FMMPreset(str(config["preset"])),
        basis=str(config["basis"]),
        theta=float(config["theta"]),
        softening=float(config["softening"]),
        working_dtype=_dtype_from_name(str(config["working_dtype"])),
        adaptive_order=bool(config.get("adaptive_order", False)),
        p_gears=tuple(int(v) for v in config.get("p_gears", [])),
        adaptive_error_model=str(config.get("adaptive_error_model", "tail_proxy")),
        adaptive_eps=(
            None
            if config.get("adaptive_eps") is None
            else float(config.get("adaptive_eps"))
        ),
        mac_force_scale_mode=str(config.get("mac_force_scale_mode", "prev")),
        advanced=advanced,
    )


def _make_row_error(*, mode: str, num_particles: int, message: str) -> dict[str, Any]:
    if mode == "sweep":
        return {
            "num_particles": int(num_particles),
            "mean_seconds": float("nan"),
            "std_seconds": float("nan"),
            "prepare_mean_seconds": float("nan"),
            "prepare_std_seconds": float("nan"),
            "evaluate_mean_seconds": float("nan"),
            "evaluate_std_seconds": float("nan"),
            "error": str(message),
        }
    if mode == "downward_trace":
        component_rows = []
        for component in ("m2l", "l2l"):
            component_rows.append(
                {
                    "component": component,
                    "num_particles": int(num_particles),
                    "gpu_used_before_mb": float("nan"),
                    "gpu_used_after_mb": float("nan"),
                    "gpu_peak_used_mb": float("nan"),
                    "gpu_peak_delta_mb": float("nan"),
                    "wall_seconds": float("nan"),
                    "mean_seconds": float("nan"),
                    "std_seconds": float("nan"),
                    "error": str(message),
                    "error_type": "other_error",
                }
            )
        return {
            "num_particles": int(num_particles),
            "component_rows": component_rows,
            "error": str(message),
        }
    return {
        "num_particles": int(num_particles),
        "tree_build_mean_seconds": float("nan"),
        "upward_mean_seconds": float("nan"),
        "interactions_mean_seconds": float("nan"),
        "downward_mean_seconds": float("nan"),
        "prepare_component_sum_seconds": float("nan"),
        "error": str(message),
    }


def _run_sweep_case(
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    seed: int,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    jaccpot_symbols = _load_jaccpot_symbols()
    FastMultipoleMethod = jaccpot_symbols["FastMultipoleMethod"]
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    tuned_kwargs, worker_tune_info = _worker_autotune_runtime_kwargs(
        cfg=cfg,
        fmm_kwargs=fmm_kwargs,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        autotune_cache_path=autotune_cache_path,
    )
    fmm = FastMultipoleMethod(**tuned_kwargs)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        if cache_path.exists():
            fmm.load_m2l_autotune_cache(str(cache_path), merge=True)
    resolved_path_info = bench_utils.resolved_large_n_memory_path_report(fmm)
    benchmark_scope = str(cfg.get("benchmark_scope", "steady_eval")).strip().lower()
    if benchmark_scope not in ("full", "steady_eval"):
        benchmark_scope = "steady_eval"
    _warm_sweep_case(
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        benchmark_scope=benchmark_scope,
    )
    _emit_ready_marker()

    prepare_once_timing = bench_utils.time_callable(
        fmm.prepare_state,
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        warmup=int(warmup),
        runs=int(runs),
    )
    prepared_state = prepare_once_timing.result
    eval_timing = bench_utils.time_callable(
        fmm.evaluate_prepared_state,
        prepared_state,
        warmup=int(warmup),
        runs=int(runs),
        **_evaluate_prepared_kwargs(fmm),
    )

    if benchmark_scope == "steady_eval":
        full_mean = float(eval_timing.mean)
        full_std = float(eval_timing.std)
    else:
        full_timing = bench_utils.time_callable(
            fmm.compute_accelerations,
            positions,
            masses,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            reuse_prepared_state=False,
            warmup=int(warmup),
            runs=int(runs),
        )
        full_mean = float(full_timing.mean)
        full_std = float(full_timing.std)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fmm.save_m2l_autotune_cache(str(cache_path))
    row = {
        "num_particles": int(num_particles),
        "mean_seconds": full_mean,
        "std_seconds": full_std,
        "prepare_mean_seconds": float(prepare_once_timing.mean),
        "prepare_std_seconds": float(prepare_once_timing.std),
        "evaluate_mean_seconds": float(eval_timing.mean),
        "evaluate_std_seconds": float(eval_timing.std),
        "benchmark_scope": benchmark_scope,
        "error": "",
    }
    row.update(resolved_path_info)
    row.update(
        _resolved_nearfield_runtime_report(
            fmm,
            num_particles=int(num_particles),
        )
    )
    row.update(worker_tune_info)
    return row


def _run_peak_prepare_case(
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    dtype: jnp.dtype,
    seed: int,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    jaccpot_symbols = _load_jaccpot_symbols()
    FastMultipoleMethod = jaccpot_symbols["FastMultipoleMethod"]
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    tuned_kwargs, worker_tune_info = _worker_autotune_runtime_kwargs(
        cfg=cfg,
        fmm_kwargs=fmm_kwargs,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        autotune_cache_path=autotune_cache_path,
    )
    fmm = FastMultipoleMethod(**tuned_kwargs)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        if cache_path.exists():
            fmm.load_m2l_autotune_cache(str(cache_path), merge=True)

    warm_state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )
    warm_state = _block_ready(warm_state)
    _block_ready(
        fmm.evaluate_prepared_state(warm_state, **_evaluate_prepared_kwargs(fmm))
    )
    del warm_state

    _emit_ready_marker()
    t0 = time.perf_counter()
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )
    state = _block_ready(state)
    t1 = time.perf_counter()

    row = {
        "num_particles": int(num_particles),
        "prepared_state_mb": float(_prepared_state_total_mb(state)),
        "prepare_seconds": float(t1 - t0),
        "error": "",
    }
    row.update(worker_tune_info)
    return row


def _run_peak_evaluate_case(
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    dtype: jnp.dtype,
    seed: int,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    jaccpot_symbols = _load_jaccpot_symbols()
    FastMultipoleMethod = jaccpot_symbols["FastMultipoleMethod"]
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    tuned_kwargs, worker_tune_info = _worker_autotune_runtime_kwargs(
        cfg=cfg,
        fmm_kwargs=fmm_kwargs,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        autotune_cache_path=autotune_cache_path,
    )
    fmm = FastMultipoleMethod(**tuned_kwargs)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        if cache_path.exists():
            fmm.load_m2l_autotune_cache(str(cache_path), merge=True)

    warm_state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )
    warm_state = _block_ready(warm_state)
    _block_ready(
        fmm.evaluate_prepared_state(warm_state, **_evaluate_prepared_kwargs(fmm))
    )
    del warm_state

    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )
    state = _block_ready(state)

    _emit_ready_marker()
    t0 = time.perf_counter()
    _block_ready(fmm.evaluate_prepared_state(state, **_evaluate_prepared_kwargs(fmm)))
    t1 = time.perf_counter()

    row = {
        "num_particles": int(num_particles),
        "prepared_state_mb": float(_prepared_state_total_mb(state)),
        "evaluate_seconds": float(t1 - t0),
        "error": "",
    }
    row.update(worker_tune_info)
    return row


def _run_prepare_case(
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    seed: int,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    jaccpot_symbols = _load_jaccpot_symbols()
    FastMultipoleMethod = jaccpot_symbols["FastMultipoleMethod"]
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    tuned_kwargs, worker_tune_info = _worker_autotune_runtime_kwargs(
        cfg=cfg,
        fmm_kwargs=fmm_kwargs,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        autotune_cache_path=autotune_cache_path,
    )
    fmm = FastMultipoleMethod(**tuned_kwargs)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        if cache_path.exists():
            fmm.load_m2l_autotune_cache(str(cache_path), merge=True)
    _warm_prepare_case(
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )
    _emit_ready_marker()

    tree_timing = bench_utils.time_callable(
        _prepare_tree_upward_artifacts_once,
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        warmup=int(warmup),
        runs=int(runs),
    )
    tree_artifacts, ctx = tree_timing.result
    interactions_timing = bench_utils.time_callable(
        _build_dual_tree_artifacts_once,
        fmm=fmm,
        tree_artifacts=tree_artifacts,
        ctx=ctx,
        warmup=int(warmup),
        runs=int(runs),
    )
    dual_downward_artifacts = interactions_timing.result
    nearfield_timing = bench_utils.time_callable(
        _prepare_nearfield_artifacts_once,
        fmm=fmm,
        tree_artifacts=tree_artifacts,
        dual_downward_artifacts=dual_downward_artifacts,
        num_particles=int(num_particles),
        warmup=int(warmup),
        runs=int(runs),
    )
    prepare_timing = bench_utils.time_callable(
        fmm.prepare_state,
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        warmup=int(warmup),
        runs=int(runs),
    )
    residual = max(
        float(prepare_timing.mean)
        - float(tree_timing.mean)
        - float(interactions_timing.mean),
        0.0,
    )
    residual = max(
        residual - float(nearfield_timing.mean),
        0.0,
    )
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fmm.save_m2l_autotune_cache(str(cache_path))
    row = {
        "num_particles": int(num_particles),
        "tree_build_mean_seconds": float(tree_timing.mean),
        "interactions_mean_seconds": float(interactions_timing.mean),
        "upward_mean_seconds": float(residual),
        "downward_mean_seconds": float(nearfield_timing.mean),
        "prepare_component_sum_seconds": float(prepare_timing.mean),
        "error": "",
    }
    row.update(worker_tune_info)
    return row


def _run_tree_case(
    *,
    num_particles: int,
    leaf_size: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    seed: int,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    del cfg, fmm_kwargs, autotune_cache_path
    yggdrax_symbols = _load_yggdrax_symbols()
    Tree = yggdrax_symbols["Tree"]
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    _warm_tree_case(
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
    )
    _emit_ready_marker()
    tree_timing = bench_utils.time_callable(
        Tree.from_particles,
        positions,
        masses,
        tree_type="radix",
        build_mode="adaptive",
        return_reordered=True,
        leaf_size=int(leaf_size),
        warmup=int(warmup),
        runs=int(runs),
    )
    return {
        "num_particles": int(num_particles),
        "component": "tree",
        "mean_seconds": float(tree_timing.mean),
        "std_seconds": float(tree_timing.std),
        "error": "",
    }


def _run_interactions_case(
    *,
    num_particles: int,
    leaf_size: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    seed: int,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    jaccpot_symbols = _load_jaccpot_symbols()
    FastMultipoleMethod = jaccpot_symbols["FastMultipoleMethod"]
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    tuned_kwargs, worker_tune_info = _worker_autotune_runtime_kwargs(
        cfg=cfg,
        fmm_kwargs=fmm_kwargs,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=1,
        autotune_cache_path=autotune_cache_path,
    )
    fmm = FastMultipoleMethod(**tuned_kwargs)
    _warm_interactions_case(
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=1,
        num_particles=int(num_particles),
    )
    _emit_ready_marker()
    tree_artifacts, ctx = _prepare_tree_upward_artifacts_once(
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=1,
    )
    timing = bench_utils.time_callable(
        _build_dual_tree_artifacts_once,
        fmm=fmm,
        tree_artifacts=tree_artifacts,
        ctx=ctx,
        warmup=int(warmup),
        runs=int(runs),
    )
    row = {
        "num_particles": int(num_particles),
        "component": "interactions",
        "mean_seconds": float(timing.mean),
        "std_seconds": float(timing.std),
        "error": "",
    }
    row.update(worker_tune_info)
    return row


def _run_m2l_case(
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    seed: int,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    FastMultipoleMethod = _load_jaccpot_symbols()["FastMultipoleMethod"]
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    tuned_kwargs, worker_tune_info = _worker_autotune_runtime_kwargs(
        cfg=cfg,
        fmm_kwargs=fmm_kwargs,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        autotune_cache_path=autotune_cache_path,
    )
    fmm = FastMultipoleMethod(**tuned_kwargs)
    stage_inputs = _prepare_downward_stage_inputs_once(
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )
    _block_ready(_run_m2l_stage_fresh(fmm=fmm, stage_inputs=stage_inputs))
    _emit_ready_marker()
    timing = bench_utils.time_callable(
        _run_m2l_stage_fresh,
        fmm=fmm,
        stage_inputs=stage_inputs,
        warmup=int(warmup),
        runs=int(runs),
    )
    row = {
        "num_particles": int(num_particles),
        "component": "m2l",
        "mean_seconds": float(timing.mean),
        "std_seconds": float(timing.std),
        "error": "",
    }
    row.update(worker_tune_info)
    return row


def _run_l2l_case(
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    seed: int,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    FastMultipoleMethod = _load_jaccpot_symbols()["FastMultipoleMethod"]
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    tuned_kwargs, worker_tune_info = _worker_autotune_runtime_kwargs(
        cfg=cfg,
        fmm_kwargs=fmm_kwargs,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        autotune_cache_path=autotune_cache_path,
    )
    fmm = FastMultipoleMethod(**tuned_kwargs)
    stage_inputs = _prepare_downward_stage_inputs_once(
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )
    _block_ready(_run_l2l_stage_fresh(fmm=fmm, stage_inputs=stage_inputs))
    _emit_ready_marker()
    timing = bench_utils.time_callable(
        _run_l2l_stage_fresh,
        fmm=fmm,
        stage_inputs=stage_inputs,
        warmup=int(warmup),
        runs=int(runs),
    )
    row = {
        "num_particles": int(num_particles),
        "component": "l2l",
        "mean_seconds": float(timing.mean),
        "std_seconds": float(timing.std),
        "error": "",
    }
    row.update(worker_tune_info)
    return row


def _run_audit_case(
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    seed: int,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    large_n_symbols = _load_large_n_runtime_symbols()
    FastMultipoleMethod = _load_jaccpot_symbols()["FastMultipoleMethod"]
    LargeNPreparedState = large_n_symbols["LargeNPreparedState"]
    evaluate_large_n_farfield = large_n_symbols["evaluate_large_n_farfield"]
    evaluate_large_n_nearfield = large_n_symbols["evaluate_large_n_nearfield"]

    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    tuned_kwargs, worker_tune_info = _worker_autotune_runtime_kwargs(
        cfg=cfg,
        fmm_kwargs=fmm_kwargs,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        autotune_cache_path=autotune_cache_path,
    )
    fmm = FastMultipoleMethod(**tuned_kwargs)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        if cache_path.exists():
            fmm.load_m2l_autotune_cache(str(cache_path), merge=True)

    _warm_sweep_case(
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        benchmark_scope="steady_eval",
    )
    _emit_ready_marker()

    tree_upward_timing = bench_utils.time_callable(
        _prepare_tree_upward_artifacts_once,
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        warmup=int(warmup),
        runs=int(runs),
    )
    tree_artifacts, ctx = tree_upward_timing.result

    dual_downward_timing = bench_utils.time_callable(
        _build_dual_tree_artifacts_once,
        fmm=fmm,
        tree_artifacts=tree_artifacts,
        ctx=ctx,
        warmup=int(warmup),
        runs=int(runs),
    )
    dual_downward_artifacts = dual_downward_timing.result

    nearfield_prepare_timing = bench_utils.time_callable(
        _prepare_nearfield_artifacts_once,
        fmm=fmm,
        tree_artifacts=tree_artifacts,
        dual_downward_artifacts=dual_downward_artifacts,
        num_particles=int(num_particles),
        warmup=int(warmup),
        runs=int(runs),
    )

    prepare_total_timing = bench_utils.time_callable(
        fmm.prepare_state,
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        warmup=int(warmup),
        runs=int(runs),
    )
    prepared_state = prepare_total_timing.result

    evaluate_total_timing = bench_utils.time_callable(
        fmm.evaluate_prepared_state,
        prepared_state,
        warmup=int(warmup),
        runs=int(runs),
        **_evaluate_prepared_kwargs(fmm),
    )

    audit_stage_breakdown_mode = "full"
    if isinstance(prepared_state, LargeNPreparedState):
        # Reconstructing explicit node-interaction buffers for standalone M2L/L2L
        # timing can OOM at 1M+ even when the real large-N runtime fits, so keep
        # the audit focused on the production nearfield/farfield split.
        m2l_mean = float("nan")
        m2l_std = float("nan")
        l2l_mean = float("nan")
        l2l_std = float("nan")
        audit_stage_breakdown_mode = "large_n_light"
    else:
        stage_inputs = _prepare_downward_stage_inputs_once(
            fmm=fmm,
            positions=positions,
            masses=masses,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
        )
        m2l_timing = bench_utils.time_callable(
            _run_m2l_stage_fresh,
            fmm=fmm,
            stage_inputs=stage_inputs,
            warmup=int(warmup),
            runs=int(runs),
        )
        l2l_timing = bench_utils.time_callable(
            _run_l2l_stage_fresh,
            fmm=fmm,
            stage_inputs=stage_inputs,
            warmup=int(warmup),
            runs=int(runs),
        )
        m2l_mean = float(m2l_timing.mean)
        m2l_std = float(m2l_timing.std)
        l2l_mean = float(l2l_timing.mean)
        l2l_std = float(l2l_timing.std)

    row = {
        "num_particles": int(num_particles),
        "prepared_state_mb": float(_prepared_state_total_mb(prepared_state)),
        "prepare_total_seconds": float(prepare_total_timing.mean),
        "prepare_total_std_seconds": float(prepare_total_timing.std),
        "prepare_tree_upward_seconds": float(tree_upward_timing.mean),
        "prepare_tree_upward_std_seconds": float(tree_upward_timing.std),
        "prepare_dual_downward_seconds": float(dual_downward_timing.mean),
        "prepare_dual_downward_std_seconds": float(dual_downward_timing.std),
        "prepare_nearfield_artifacts_seconds": float(nearfield_prepare_timing.mean),
        "prepare_nearfield_artifacts_std_seconds": float(nearfield_prepare_timing.std),
        "prepare_residual_seconds": max(
            float(prepare_total_timing.mean)
            - float(tree_upward_timing.mean)
            - float(dual_downward_timing.mean)
            - float(nearfield_prepare_timing.mean),
            0.0,
        ),
        "downward_m2l_seconds": m2l_mean,
        "downward_m2l_std_seconds": m2l_std,
        "downward_l2l_seconds": l2l_mean,
        "downward_l2l_std_seconds": l2l_std,
        "evaluate_total_seconds": float(evaluate_total_timing.mean),
        "evaluate_total_std_seconds": float(evaluate_total_timing.std),
        "audit_stage_breakdown_mode": audit_stage_breakdown_mode,
        "error": "",
    }
    row.update(
        _resolved_nearfield_runtime_report(
            fmm,
            num_particles=int(num_particles),
        )
    )

    if isinstance(prepared_state, LargeNPreparedState):
        nearfield_eval_timing = bench_utils.time_callable(
            evaluate_large_n_nearfield,
            fmm._impl,
            prepared_state,
            warmup=int(warmup),
            runs=int(runs),
            return_potential=False,
        )
        farfield_eval_timing = bench_utils.time_callable(
            evaluate_large_n_farfield,
            prepared_state,
            warmup=int(warmup),
            runs=int(runs),
            return_potential=False,
        )
        row.update(
            {
                "state_execution_backend": "large_n",
                "evaluate_large_n_nearfield_seconds": float(nearfield_eval_timing.mean),
                "evaluate_large_n_nearfield_std_seconds": float(
                    nearfield_eval_timing.std
                ),
                "evaluate_large_n_farfield_seconds": float(farfield_eval_timing.mean),
                "evaluate_large_n_farfield_std_seconds": float(
                    farfield_eval_timing.std
                ),
            }
        )
    else:
        row["state_execution_backend"] = str(
            getattr(prepared_state, "execution_backend", "unknown")
        )

    row.update(bench_utils.resolved_large_n_memory_path_report(fmm))
    row.update(worker_tune_info)
    return row


def _run_nearfield_components_case(
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    seed: int,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
    focused_only: bool = False,
    production_only: bool = False,
) -> dict[str, Any]:
    large_n_symbols = _load_large_n_runtime_symbols()
    nearfield_symbols = _load_nearfield_symbols()
    FastMultipoleMethod = _load_jaccpot_symbols()["FastMultipoleMethod"]
    LargeNPreparedState = large_n_symbols["LargeNPreparedState"]
    evaluate_large_n_nearfield = large_n_symbols["evaluate_large_n_nearfield"]
    INDEX_DTYPE = nearfield_symbols["INDEX_DTYPE"]
    apply_packed_particle_vector_updates = nearfield_symbols[
        "apply_packed_particle_vector_updates"
    ]
    nearfield_tile_pair_accel = nearfield_symbols["nearfield_tile_pair_accel"]
    nearfield_tile_pair_backend = nearfield_symbols["nearfield_tile_pair_backend"]
    compute_self_only = nearfield_symbols[
        "_compute_leaf_p2p_prepared_large_n_self_only_impl"
    ]
    compute_pairs_only = nearfield_symbols[
        "_compute_leaf_p2p_prepared_large_n_pairs_only_impl"
    ]
    pair_contributions_batched = nearfield_symbols["_pair_contributions_batched"]
    prepare_leaf_data_from_groups = nearfield_symbols["_prepare_leaf_data_from_groups"]
    reduce_pair_bucket_by_target_leaf = nearfield_symbols[
        "_reduce_pair_bucket_by_target_leaf"
    ]
    compact_reduced_pair_bucket_rows = nearfield_symbols[
        "_compact_reduced_pair_bucket_rows"
    ]
    nearfield_unique_updates_backend = nearfield_symbols[
        "nearfield_unique_updates_backend"
    ]
    collect_radix_fast_lane_counters = nearfield_symbols[
        "collect_radix_fast_lane_counters"
    ]
    pack_unique_particle_vector_updates = nearfield_symbols[
        "pack_unique_particle_vector_updates"
    ]
    pallas_nearfield_tile_pair_supported = nearfield_symbols[
        "pallas_nearfield_tile_pair_supported"
    ]
    pallas_nearfield_unique_updates_supported = nearfield_symbols[
        "pallas_nearfield_unique_updates_supported"
    ]
    scatter_contributions = nearfield_symbols["_scatter_contributions"]
    prepare_leaf_neighbor_pairs = nearfield_symbols["prepare_leaf_neighbor_pairs"]

    @partial(jax.jit, static_argnames=("edge_chunk_size",))
    def _compute_pair_arith_only_impl(
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        edge_count_local = target_leaf_ids.shape[0]
        if edge_count_local == 0:
            return jnp.asarray(0.0, dtype=dtype_local)

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)

        def _chunk_body(total: Any, start: Any) -> tuple[Any, None]:
            edge_idx = start + chunk_offsets_local
            in_range = edge_idx < edge_count_local
            safe_edge_idx = jnp.where(in_range, edge_idx, 0)
            valid_edge = in_range & valid_pairs[safe_edge_idx]

            def _compute(total_in: Any) -> Any:
                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos = leaf_positions[tgt_leaf_local]
                tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                src_pos = leaf_positions[src_leaf_local]
                src_mass = leaf_masses[src_leaf_local]
                src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos,
                    tgt_mask,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                return total_in + jnp.sum(jnp.abs(pair_acc), dtype=dtype_local)

            return (
                jax.lax.cond(
                    jnp.any(valid_edge),
                    _compute,
                    lambda total_in: total_in,
                    total,
                ),
                None,
            )

        total_probe, _ = jax.lax.scan(
            _chunk_body,
            jnp.asarray(0.0, dtype=dtype_local),
            starts_local,
        )
        return total_probe

    @partial(jax.jit, static_argnames=("edge_chunk_size",))
    def _compute_pair_reduction_only_impl(
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        edge_count_local = target_leaf_ids.shape[0]
        if edge_count_local == 0:
            return jnp.asarray(0.0, dtype=dtype_local)

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)

        def _chunk_body(total: Any, start: Any) -> tuple[Any, None]:
            edge_idx = start + chunk_offsets_local
            in_range = edge_idx < edge_count_local
            safe_edge_idx = jnp.where(in_range, edge_idx, 0)
            valid_edge = in_range & valid_pairs[safe_edge_idx]

            def _compute(total_in: Any) -> Any:
                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos = leaf_positions[tgt_leaf_local]
                tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                src_pos = leaf_positions[src_leaf_local]
                src_mass = leaf_masses[src_leaf_local]
                src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos,
                    tgt_mask,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                reduced_tgt_leaf_local, reduced_pair_acc, reduced_valid = (
                    reduce_pair_bucket_by_target_leaf(
                        tgt_leaf_local,
                        valid_edge,
                        pair_acc,
                    )
                )
                return (
                    total_in
                    + jnp.sum(reduced_pair_acc, dtype=dtype_local)
                    + jnp.sum(reduced_tgt_leaf_local.astype(dtype_local))
                    + jnp.sum(reduced_valid.astype(dtype_local))
                )

            return (
                jax.lax.cond(
                    jnp.any(valid_edge),
                    _compute,
                    lambda total_in: total_in,
                    total,
                ),
                None,
            )

        total_probe, _ = jax.lax.scan(
            _chunk_body,
            jnp.asarray(0.0, dtype=dtype_local),
            starts_local,
        )
        return total_probe

    @partial(jax.jit, static_argnames=("edge_chunk_size",))
    def _compute_pair_gather_only_impl(
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        edge_chunk_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        edge_count_local = target_leaf_ids.shape[0]
        if edge_count_local == 0:
            return jnp.asarray(0.0, dtype=dtype_local)

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)

        def _chunk_body(total: Any, start: Any) -> tuple[Any, None]:
            edge_idx = start + chunk_offsets_local
            in_range = edge_idx < edge_count_local
            safe_edge_idx = jnp.where(in_range, edge_idx, 0)
            valid_edge = in_range & valid_pairs[safe_edge_idx]

            def _compute(total_in: Any) -> Any:
                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos = leaf_positions[tgt_leaf_local]
                tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                tgt_ids = leaf_particle_idx[tgt_leaf_local]
                src_pos = leaf_positions[src_leaf_local]
                src_mass = leaf_masses[src_leaf_local]
                src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                return (
                    total_in
                    + jnp.sum(jnp.abs(tgt_pos), dtype=dtype_local)
                    + jnp.sum(tgt_mask.astype(dtype_local))
                    + jnp.sum(tgt_ids.astype(dtype_local))
                    + jnp.sum(jnp.abs(src_pos), dtype=dtype_local)
                    + jnp.sum(jnp.abs(src_mass), dtype=dtype_local)
                    + jnp.sum(src_mask.astype(dtype_local))
                )

            return (
                jax.lax.cond(
                    jnp.any(valid_edge),
                    _compute,
                    lambda total_in: total_in,
                    total,
                ),
                None,
            )

        total_probe, _ = jax.lax.scan(
            _chunk_body,
            jnp.asarray(0.0, dtype=dtype_local),
            starts_local,
        )
        return total_probe

    @partial(jax.jit, static_argnames=("edge_chunk_size",))
    def _compute_pair_scatter_only_impl(
        positions_sorted: Any,
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        edge_count_local = target_leaf_ids.shape[0]
        acc0 = jnp.zeros_like(positions_sorted)
        if edge_count_local == 0:
            return acc0

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)

        def _chunk_body(acc: Any, start: Any) -> tuple[Any, None]:
            edge_idx = start + chunk_offsets_local
            in_range = edge_idx < edge_count_local
            safe_edge_idx = jnp.where(in_range, edge_idx, 0)
            valid_edge = in_range & valid_pairs[safe_edge_idx]

            def _compute(acc_in: Any) -> Any:
                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos = leaf_positions[tgt_leaf_local]
                tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                src_pos = leaf_positions[src_leaf_local]
                src_mass = leaf_masses[src_leaf_local]
                src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos,
                    tgt_mask,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                reduced_tgt_leaf_local, reduced_pair_acc, reduced_valid = (
                    reduce_pair_bucket_by_target_leaf(
                        tgt_leaf_local,
                        valid_edge,
                        pair_acc,
                    )
                )
                reduced_tgt_ids = leaf_particle_idx[reduced_tgt_leaf_local]
                reduced_tgt_mask = leaf_mask[reduced_tgt_leaf_local] & reduced_valid[:, None]
                return scatter_contributions(
                    acc_in,
                    reduced_tgt_ids,
                    reduced_pair_acc,
                    reduced_tgt_mask,
                )

            return (
                jax.lax.cond(
                    jnp.any(valid_edge),
                    _compute,
                    lambda acc_in: acc_in,
                    acc,
                ),
                None,
            )

        acc_out, _ = jax.lax.scan(
            _chunk_body,
            acc0,
            starts_local,
        )
        return acc_out

    @partial(jax.jit, static_argnames=("edge_chunk_size",))
    def _compute_pair_compacted_scatter_only_impl(
        positions_sorted: Any,
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        edge_count_local = target_leaf_ids.shape[0]
        acc0 = jnp.zeros_like(positions_sorted)
        if edge_count_local == 0:
            return acc0

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)

        def _chunk_body(acc: Any, start: Any) -> tuple[Any, None]:
            edge_idx = start + chunk_offsets_local
            in_range = edge_idx < edge_count_local
            safe_edge_idx = jnp.where(in_range, edge_idx, 0)
            valid_edge = in_range & valid_pairs[safe_edge_idx]

            def _compute(acc_in: Any) -> Any:
                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos = leaf_positions[tgt_leaf_local]
                tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                src_pos = leaf_positions[src_leaf_local]
                src_mass = leaf_masses[src_leaf_local]
                src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos,
                    tgt_mask,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                reduced_tgt_leaf_local, reduced_pair_acc, reduced_valid = (
                    reduce_pair_bucket_by_target_leaf(
                        tgt_leaf_local,
                        valid_edge,
                        pair_acc,
                    )
                )
                compact_tgt_leaf_local, compact_pair_acc, compact_valid = (
                    compact_reduced_pair_bucket_rows(
                        reduced_tgt_leaf_local,
                        reduced_pair_acc,
                        reduced_valid,
                    )
                )
                compact_tgt_ids = leaf_particle_idx[compact_tgt_leaf_local]
                compact_tgt_mask = leaf_mask[compact_tgt_leaf_local] & compact_valid[:, None]
                return scatter_contributions(
                    acc_in,
                    compact_tgt_ids,
                    compact_pair_acc,
                    compact_tgt_mask,
                )

            return (
                jax.lax.cond(
                    jnp.any(valid_edge),
                    _compute,
                    lambda acc_in: acc_in,
                    acc,
                ),
                None,
            )

        acc_out, _ = jax.lax.scan(
            _chunk_body,
            acc0,
            starts_local,
        )
        return acc_out

    @partial(jax.jit, static_argnames=("edge_chunk_size",))
    def _compute_pair_lax_scatter_only_impl(
        positions_sorted: Any,
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        edge_count_local = target_leaf_ids.shape[0]
        acc0 = jnp.zeros_like(positions_sorted)
        if edge_count_local == 0:
            return acc0

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)
        scatter_dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )

        def _chunk_body(acc: Any, start: Any) -> tuple[Any, None]:
            edge_idx = start + chunk_offsets_local
            in_range = edge_idx < edge_count_local
            safe_edge_idx = jnp.where(in_range, edge_idx, 0)
            valid_edge = in_range & valid_pairs[safe_edge_idx]

            def _compute(acc_in: Any) -> Any:
                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos = leaf_positions[tgt_leaf_local]
                tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                src_pos = leaf_positions[src_leaf_local]
                src_mass = leaf_masses[src_leaf_local]
                src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos,
                    tgt_mask,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                reduced_tgt_leaf_local, reduced_pair_acc, reduced_valid = (
                    reduce_pair_bucket_by_target_leaf(
                        tgt_leaf_local,
                        valid_edge,
                        pair_acc,
                    )
                )
                reduced_tgt_ids = leaf_particle_idx[reduced_tgt_leaf_local]
                reduced_tgt_mask = leaf_mask[reduced_tgt_leaf_local] & reduced_valid[:, None]
                flat_indices = reduced_tgt_ids.reshape(-1, 1)
                flat_values = reduced_pair_acc.reshape(-1, reduced_pair_acc.shape[-1])
                flat_mask = reduced_tgt_mask.reshape(-1)
                masked_values = jnp.where(flat_mask[:, None], flat_values, 0.0)
                return jax.lax.scatter_add(
                    acc_in,
                    flat_indices,
                    masked_values,
                    scatter_dnums,
                    indices_are_sorted=False,
                    unique_indices=False,
                    mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                )

            return (
                jax.lax.cond(
                    jnp.any(valid_edge),
                    _compute,
                    lambda acc_in: acc_in,
                    acc,
                ),
                None,
            )

        acc_out, _ = jax.lax.scan(
            _chunk_body,
            acc0,
            starts_local,
        )
        return acc_out

    @partial(jax.jit, static_argnames=("edge_chunk_size",))
    def _compute_pair_compacted_only_impl(
        positions_sorted: Any,
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        edge_count_local = target_leaf_ids.shape[0]
        acc0 = jnp.zeros_like(positions_sorted)
        if edge_count_local == 0:
            return acc0

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)

        def _chunk_body(acc: Any, start: Any) -> tuple[Any, None]:
            edge_idx = start + chunk_offsets_local
            in_range = edge_idx < edge_count_local
            safe_edge_idx = jnp.where(in_range, edge_idx, 0)
            valid_edge = in_range & valid_pairs[safe_edge_idx]

            def _compute(acc_in: Any) -> Any:
                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos = leaf_positions[tgt_leaf_local]
                tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                src_pos = leaf_positions[src_leaf_local]
                src_mass = leaf_masses[src_leaf_local]
                src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos,
                    tgt_mask,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                reduced_tgt_leaf_local, reduced_pair_acc, reduced_valid = (
                    reduce_pair_bucket_by_target_leaf(
                        tgt_leaf_local,
                        valid_edge,
                        pair_acc,
                    )
                )
                compact_tgt_leaf_local, compact_pair_acc, compact_valid = (
                    compact_reduced_pair_bucket_rows(
                        reduced_tgt_leaf_local,
                        reduced_pair_acc,
                        reduced_valid,
                    )
                )
                compact_tgt_ids = leaf_particle_idx[compact_tgt_leaf_local]
                compact_tgt_mask = leaf_mask[compact_tgt_leaf_local] & compact_valid[:, None]
                return scatter_contributions(
                    acc_in,
                    compact_tgt_ids,
                    compact_pair_acc,
                    compact_tgt_mask,
                )

            return (
                jax.lax.cond(
                    jnp.any(valid_edge),
                    _compute,
                    lambda acc_in: acc_in,
                    acc,
                ),
                None,
            )

        acc_out, _ = jax.lax.scan(
            _chunk_body,
            acc0,
            starts_local,
        )
        return acc_out

    @partial(jax.jit, static_argnames=("edge_chunk_size",))
    def _compute_pair_leaf_accum_only_impl(
        positions_sorted: Any,
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        edge_count_local = target_leaf_ids.shape[0]
        leaf_acc0 = jnp.zeros_like(leaf_positions)
        if edge_count_local == 0:
            return scatter_contributions(
                jnp.zeros_like(positions_sorted),
                leaf_particle_idx,
                leaf_acc0,
                leaf_mask,
            )

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)
        leaf_scatter_dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(1, 2),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )

        def _chunk_body(leaf_acc: Any, start: Any) -> tuple[Any, None]:
            edge_idx = start + chunk_offsets_local
            in_range = edge_idx < edge_count_local
            safe_edge_idx = jnp.where(in_range, edge_idx, 0)
            valid_edge = in_range & valid_pairs[safe_edge_idx]

            def _compute(leaf_acc_in: Any) -> Any:
                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos = leaf_positions[tgt_leaf_local]
                tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                src_pos = leaf_positions[src_leaf_local]
                src_mass = leaf_masses[src_leaf_local]
                src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos,
                    tgt_mask,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                reduced_tgt_leaf_local, reduced_pair_acc, reduced_valid = (
                    reduce_pair_bucket_by_target_leaf(
                        tgt_leaf_local,
                        valid_edge,
                        pair_acc,
                    )
                )
                masked_reduced_pair_acc = jnp.where(
                    reduced_valid[:, None, None],
                    reduced_pair_acc,
                    0.0,
                )
                return jax.lax.scatter_add(
                    leaf_acc_in,
                    reduced_tgt_leaf_local[:, None],
                    masked_reduced_pair_acc,
                    leaf_scatter_dnums,
                    indices_are_sorted=False,
                    unique_indices=False,
                    mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                )

            return (
                jax.lax.cond(
                    jnp.any(valid_edge),
                    _compute,
                    lambda leaf_acc_in: leaf_acc_in,
                    leaf_acc,
                ),
                None,
            )

        leaf_acc_out, _ = jax.lax.scan(
            _chunk_body,
            leaf_acc0,
            starts_local,
        )
        return scatter_contributions(
            jnp.zeros_like(positions_sorted),
            leaf_particle_idx,
            leaf_acc_out,
            leaf_mask,
        )

    @partial(jax.jit, static_argnames=("edge_chunk_size",))
    def _compute_pair_target_sorted_leaf_accum_only_impl(
        positions_sorted: Any,
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        edge_count_local = target_leaf_ids.shape[0]
        leaf_acc0 = jnp.zeros_like(leaf_positions)
        if edge_count_local == 0:
            return scatter_contributions(
                jnp.zeros_like(positions_sorted),
                leaf_particle_idx,
                leaf_acc0,
                leaf_mask,
            )

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)
        leaf_scatter_dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(1, 2),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )

        def _chunk_body(leaf_acc: Any, start: Any) -> tuple[Any, None]:
            edge_idx = start + chunk_offsets_local
            in_range = edge_idx < edge_count_local
            safe_edge_idx = jnp.where(in_range, edge_idx, 0)
            valid_edge = in_range & valid_pairs[safe_edge_idx]

            def _compute(leaf_acc_in: Any) -> Any:
                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos = leaf_positions[tgt_leaf_local]
                tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                src_pos = leaf_positions[src_leaf_local]
                src_mass = leaf_masses[src_leaf_local]
                src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos,
                    tgt_mask,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                reduced_tgt_leaf_local, reduced_pair_acc, reduced_valid = (
                    reduce_pair_bucket_by_target_leaf(
                        tgt_leaf_local,
                        valid_edge,
                        pair_acc,
                    )
                )
                valid_count = jnp.sum(reduced_valid.astype(INDEX_DTYPE))
                masked_reduced_tgt_leaf_local = jnp.where(
                    reduced_valid,
                    reduced_tgt_leaf_local,
                    0,
                )
                masked_reduced_pair_acc = jnp.where(
                    reduced_valid[:, None, None],
                    reduced_pair_acc,
                    0.0,
                )
                return jax.lax.cond(
                    valid_count > 0,
                    lambda acc_in: jax.lax.scatter_add(
                        acc_in,
                        masked_reduced_tgt_leaf_local[:, None],
                        masked_reduced_pair_acc,
                        leaf_scatter_dnums,
                        indices_are_sorted=True,
                        unique_indices=True,
                        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                    ),
                    lambda acc_in: acc_in,
                    leaf_acc_in,
                )

            return (
                jax.lax.cond(
                    jnp.any(valid_edge),
                    _compute,
                    lambda leaf_acc_in: leaf_acc_in,
                    leaf_acc,
                ),
                None,
            )

        leaf_acc_out, _ = jax.lax.scan(
            _chunk_body,
            leaf_acc0,
            starts_local,
        )
        return scatter_contributions(
            jnp.zeros_like(positions_sorted),
            leaf_particle_idx,
            leaf_acc_out,
            leaf_mask,
        )

    @partial(jax.jit, static_argnames=("max_neighbors_per_target",))
    def _compute_pair_target_leaf_owned_impl(
        positions_sorted: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        target_offsets: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        max_neighbors_per_target: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        num_targets = int(leaf_positions.shape[0])
        leaf_size_local = int(leaf_positions.shape[1])
        slots = jnp.arange(int(max_neighbors_per_target), dtype=INDEX_DTYPE)
        leaf_acc0 = jnp.zeros(
            (num_targets, leaf_size_local, positions_sorted.shape[-1]),
            dtype=dtype_local,
        )
        if source_leaf_ids.shape[0] == 0 or num_targets == 0:
            return scatter_contributions(
                jnp.zeros_like(positions_sorted),
                leaf_particle_idx,
                leaf_acc0,
                leaf_mask,
            )

        def _target_body(leaf_acc: Any, target_idx: Any) -> tuple[Any, None]:
            start = target_offsets[target_idx]
            end = target_offsets[target_idx + 1]
            count = end - start

            tgt_pos = leaf_positions[target_idx]
            tgt_mask = leaf_mask[target_idx]

            def _source_body(acc: Any, slot: Any) -> tuple[Any, None]:
                edge_idx = start + slot
                in_range = slot < count
                safe_edge_idx = jnp.where(in_range, edge_idx, 0)
                valid_edge = in_range & valid_pairs[safe_edge_idx]
                src_leaf = jnp.where(valid_edge, source_leaf_ids[safe_edge_idx], 0)

                src_pos = leaf_positions[src_leaf][None, ...]
                src_mass = leaf_masses[src_leaf][None, ...]
                src_mask = (leaf_mask[src_leaf] & valid_edge)[None, ...]
                tgt_pos_batch = tgt_pos[None, ...]
                tgt_mask_batch = tgt_mask[None, ...]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos_batch,
                    tgt_mask_batch,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                pair_acc_single = jnp.where(valid_edge, pair_acc[0], 0.0)
                return acc + pair_acc_single, None

            target_acc, _ = jax.lax.scan(
                _source_body,
                jnp.zeros((leaf_size_local, positions_sorted.shape[-1]), dtype=dtype_local),
                slots,
            )
            return (leaf_acc.at[target_idx].set(target_acc), None)

        leaf_acc_out, _ = jax.lax.scan(
            _target_body,
            leaf_acc0,
            jnp.arange(num_targets, dtype=INDEX_DTYPE),
        )
        return scatter_contributions(
            jnp.zeros_like(positions_sorted),
            leaf_particle_idx,
            leaf_acc_out,
            leaf_mask,
        )

    @partial(
        jax.jit,
        static_argnames=("max_neighbors_per_target", "target_batch_size"),
    )
    def _compute_pair_target_leaf_batched_impl(
        positions_sorted: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        target_offsets: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        max_neighbors_per_target: int,
        target_batch_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        num_targets = int(leaf_positions.shape[0])
        leaf_size_local = int(leaf_positions.shape[1])
        target_batch_local = int(target_batch_size)
        if target_batch_local <= 0:
            raise ValueError("target_batch_size must be positive")
        neighbor_slots = jnp.arange(int(max_neighbors_per_target), dtype=INDEX_DTYPE)
        batch_slots = jnp.arange(target_batch_local, dtype=INDEX_DTYPE)
        leaf_acc0 = jnp.zeros(
            (num_targets, leaf_size_local, positions_sorted.shape[-1]),
            dtype=dtype_local,
        )
        if source_leaf_ids.shape[0] == 0 or num_targets == 0:
            return scatter_contributions(
                jnp.zeros_like(positions_sorted),
                leaf_particle_idx,
                leaf_acc0,
                leaf_mask,
            )

        batch_starts = jnp.arange(0, num_targets, target_batch_local, dtype=INDEX_DTYPE)

        def _batch_body(leaf_acc: Any, batch_start: Any) -> tuple[Any, None]:
            target_idx = batch_start + batch_slots
            in_range = target_idx < num_targets
            safe_target_idx = jnp.where(in_range, target_idx, 0)
            start = target_offsets[safe_target_idx]
            end = target_offsets[safe_target_idx + 1]
            count = end - start

            tgt_pos = leaf_positions[safe_target_idx]
            tgt_mask = leaf_mask[safe_target_idx] & in_range[:, None]

            def _source_body(acc: Any, slot: Any) -> tuple[Any, None]:
                edge_idx = start + slot
                in_range_slot = (slot < count) & in_range
                safe_edge_idx = jnp.where(in_range_slot, edge_idx, 0)
                valid_edge = in_range_slot & valid_pairs[safe_edge_idx]
                src_leaf = jnp.where(valid_edge, source_leaf_ids[safe_edge_idx], 0)

                src_pos = leaf_positions[src_leaf]
                src_mass = leaf_masses[src_leaf]
                src_mask = leaf_mask[src_leaf] & valid_edge[:, None]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos,
                    tgt_mask,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                return acc + jnp.where(valid_edge[:, None, None], pair_acc, 0.0), None

            batch_acc, _ = jax.lax.scan(
                _source_body,
                jnp.zeros(
                    (target_batch_local, leaf_size_local, positions_sorted.shape[-1]),
                    dtype=dtype_local,
                ),
                neighbor_slots,
            )
            batch_acc = jnp.where(in_range[:, None, None], batch_acc, 0.0)
            updated = leaf_acc.at[safe_target_idx].add(batch_acc)
            return updated, None

        leaf_acc_out, _ = jax.lax.scan(
            _batch_body,
            leaf_acc0,
            batch_starts,
        )
        return scatter_contributions(
            jnp.zeros_like(positions_sorted),
            leaf_particle_idx,
            leaf_acc_out,
            leaf_mask,
        )

    @partial(
        jax.jit,
        static_argnames=(
            "max_neighbors_per_target",
            "target_batch_size",
            "neighbor_block_size",
        ),
    )
    def _compute_pair_target_leaf_bucketed_batched_impl(
        positions_sorted: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        target_offsets: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        max_neighbors_per_target: int,
        target_batch_size: int,
        neighbor_block_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        num_targets = int(leaf_positions.shape[0])
        leaf_size_local = int(leaf_positions.shape[1])
        target_batch_local = int(target_batch_size)
        neighbor_block_local = int(neighbor_block_size)
        if target_batch_local <= 0:
            raise ValueError("target_batch_size must be positive")
        if neighbor_block_local <= 0:
            raise ValueError("neighbor_block_size must be positive")

        leaf_acc0 = jnp.zeros(
            (num_targets, leaf_size_local, positions_sorted.shape[-1]),
            dtype=dtype_local,
        )
        if source_leaf_ids.shape[0] == 0 or num_targets == 0:
            return scatter_contributions(
                jnp.zeros_like(positions_sorted),
                leaf_particle_idx,
                leaf_acc0,
                leaf_mask,
            )

        batch_slots = jnp.arange(target_batch_local, dtype=INDEX_DTYPE)
        neighbor_block_offsets = jnp.arange(
            neighbor_block_local,
            dtype=INDEX_DTYPE,
        )
        batch_starts = jnp.arange(0, num_targets, target_batch_local, dtype=INDEX_DTYPE)
        neighbor_block_starts = jnp.arange(
            0,
            int(max_neighbors_per_target),
            neighbor_block_local,
            dtype=INDEX_DTYPE,
        )

        def _batch_body(leaf_acc: Any, batch_start: Any) -> tuple[Any, None]:
            target_idx = batch_start + batch_slots
            target_in_range = target_idx < num_targets
            safe_target_idx = jnp.where(target_in_range, target_idx, 0)
            start = target_offsets[safe_target_idx]
            end = target_offsets[safe_target_idx + 1]
            count = end - start

            tgt_pos = leaf_positions[safe_target_idx]
            tgt_mask = leaf_mask[safe_target_idx] & target_in_range[:, None]

            def _neighbor_block_body(acc: Any, block_start: Any) -> tuple[Any, None]:
                slot_idx = block_start + neighbor_block_offsets
                slot_in_range = slot_idx[None, :] < count[:, None]
                edge_idx = start[:, None] + slot_idx[None, :]
                edge_valid = target_in_range[:, None] & slot_in_range
                safe_edge_idx = jnp.where(edge_valid, edge_idx, 0)
                pair_valid = edge_valid & valid_pairs[safe_edge_idx]
                src_leaf = jnp.where(pair_valid, source_leaf_ids[safe_edge_idx], 0)

                # Flatten the target-batch x neighbor-block work into one dense
                # batch so XLA sees a larger regular kernel than the scan-owned
                # target-leaf variants.
                tgt_pos_block = jnp.broadcast_to(
                    tgt_pos[:, None, :, :],
                    (target_batch_local, neighbor_block_local, leaf_size_local, tgt_pos.shape[-1]),
                ).reshape(-1, leaf_size_local, tgt_pos.shape[-1])
                tgt_mask_block = jnp.broadcast_to(
                    tgt_mask[:, None, :],
                    (target_batch_local, neighbor_block_local, leaf_size_local),
                ).reshape(-1, leaf_size_local)
                src_pos_block = leaf_positions[src_leaf].reshape(
                    -1,
                    leaf_size_local,
                    tgt_pos.shape[-1],
                )
                src_mass_block = leaf_masses[src_leaf].reshape(-1, leaf_size_local)
                src_mask_block = (leaf_mask[src_leaf] & pair_valid[:, :, None]).reshape(
                    -1,
                    leaf_size_local,
                )

                pair_acc_flat, _ = pair_contributions_batched(
                    tgt_pos_block,
                    tgt_mask_block,
                    src_pos_block,
                    src_mass_block,
                    src_mask_block,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                pair_acc = pair_acc_flat.reshape(
                    target_batch_local,
                    neighbor_block_local,
                    leaf_size_local,
                    positions_sorted.shape[-1],
                )
                masked_pair_acc = jnp.where(
                    pair_valid[:, :, None, None],
                    pair_acc,
                    0.0,
                )
                return acc + jnp.sum(masked_pair_acc, axis=1), None

            batch_acc, _ = jax.lax.scan(
                _neighbor_block_body,
                jnp.zeros(
                    (target_batch_local, leaf_size_local, positions_sorted.shape[-1]),
                    dtype=dtype_local,
                ),
                neighbor_block_starts,
            )
            batch_acc = jnp.where(target_in_range[:, None, None], batch_acc, 0.0)
            updated = leaf_acc.at[safe_target_idx].add(batch_acc)
            return updated, None

        leaf_acc_out, _ = jax.lax.scan(
            _batch_body,
            leaf_acc0,
            batch_starts,
        )
        return scatter_contributions(
            jnp.zeros_like(positions_sorted),
            leaf_particle_idx,
            leaf_acc_out,
            leaf_mask,
        )

    @partial(jax.jit, static_argnames=("edge_chunk_size", "tile_size"))
    def _compute_pair_target_sorted_particle_tile_accum_only_impl(
        positions_sorted: Any,
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
        tile_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        edge_count_local = target_leaf_ids.shape[0]
        tile_local = int(tile_size)
        if tile_local <= 0:
            raise ValueError("tile_size must be positive")

        num_particles = int(positions_sorted.shape[0])
        num_tiles = (num_particles + tile_local - 1) // tile_local
        tile_acc0 = jnp.zeros((num_tiles, tile_local, positions_sorted.shape[-1]), dtype=dtype_local)
        if edge_count_local == 0:
            return tile_acc0.reshape(-1, positions_sorted.shape[-1])[:num_particles]

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)
        tile_scatter_dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1),
        )

        def _chunk_body(tile_acc: Any, start: Any) -> tuple[Any, None]:
            edge_idx = start + chunk_offsets_local
            in_range = edge_idx < edge_count_local
            safe_edge_idx = jnp.where(in_range, edge_idx, 0)
            valid_edge = in_range & valid_pairs[safe_edge_idx]

            def _compute(tile_acc_in: Any) -> Any:
                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos = leaf_positions[tgt_leaf_local]
                tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                src_pos = leaf_positions[src_leaf_local]
                src_mass = leaf_masses[src_leaf_local]
                src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos,
                    tgt_mask,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                reduced_tgt_leaf_local, reduced_pair_acc, reduced_valid = (
                    reduce_pair_bucket_by_target_leaf(
                        tgt_leaf_local,
                        valid_edge,
                        pair_acc,
                    )
                )
                reduced_tgt_ids = leaf_particle_idx[reduced_tgt_leaf_local]
                reduced_tgt_mask = (
                    leaf_mask[reduced_tgt_leaf_local] & reduced_valid[:, None]
                )

                unique_particle_ids, unique_particle_values, unique_particle_valid = (
                    pack_unique_particle_vector_updates(
                        reduced_tgt_ids,
                        reduced_pair_acc,
                        reduced_tgt_mask,
                    )
                )
                safe_particle_ids = jnp.where(
                    unique_particle_valid,
                    unique_particle_ids,
                    0,
                )
                tile_ids = safe_particle_ids // tile_local
                tile_slots = safe_particle_ids % tile_local
                tile_indices = jnp.stack((tile_ids, tile_slots), axis=1)
                masked_values = jnp.where(
                    unique_particle_valid[:, None],
                    unique_particle_values,
                    0.0,
                )
                valid_count = jnp.sum(unique_particle_valid.astype(INDEX_DTYPE))
                return jax.lax.cond(
                    valid_count > 0,
                    lambda acc_in: jax.lax.scatter_add(
                        acc_in,
                        tile_indices,
                        masked_values,
                        tile_scatter_dnums,
                        indices_are_sorted=True,
                        unique_indices=True,
                        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                    ),
                    lambda acc_in: acc_in,
                    tile_acc_in,
                )

            return (
                jax.lax.cond(
                    jnp.any(valid_edge),
                    _compute,
                    lambda tile_acc_in: tile_acc_in,
                    tile_acc,
                ),
                None,
            )

        tile_acc_out, _ = jax.lax.scan(
            _chunk_body,
            tile_acc0,
            starts_local,
        )
        return tile_acc_out.reshape(-1, positions_sorted.shape[-1])[:num_particles]

    @partial(jax.jit, static_argnames=("edge_chunk_size", "tile_size"))
    def _compute_pair_target_sorted_leaf_tile_microkernel_impl(
        positions_sorted: Any,
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
        tile_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        edge_count_local = target_leaf_ids.shape[0]
        leaf_size_local = int(leaf_positions.shape[1])
        tile_local = int(tile_size)
        if tile_local <= 0:
            raise ValueError("tile_size must be positive")

        tile_count = (leaf_size_local + tile_local - 1) // tile_local
        padded_leaf_size = tile_count * tile_local
        leaf_pad = padded_leaf_size - leaf_size_local

        padded_leaf_positions = jnp.pad(
            leaf_positions,
            ((0, 0), (0, leaf_pad), (0, 0)),
        )
        padded_leaf_masses = jnp.pad(
            leaf_masses,
            ((0, 0), (0, leaf_pad)),
        )
        padded_leaf_mask = jnp.pad(
            leaf_mask,
            ((0, 0), (0, leaf_pad)),
            constant_values=False,
        )

        leaf_pos_tiles = padded_leaf_positions.reshape(
            leaf_positions.shape[0],
            tile_count,
            tile_local,
            leaf_positions.shape[-1],
        )
        leaf_mass_tiles = padded_leaf_masses.reshape(
            leaf_masses.shape[0],
            tile_count,
            tile_local,
        )
        leaf_mask_tiles = padded_leaf_mask.reshape(
            leaf_mask.shape[0],
            tile_count,
            tile_local,
        )

        leaf_tile_acc0 = jnp.zeros(
            (
                leaf_positions.shape[0],
                tile_count,
                tile_local,
                leaf_positions.shape[-1],
            ),
            dtype=dtype_local,
        )
        if edge_count_local == 0:
            flat_leaf_acc = leaf_tile_acc0.reshape(
                leaf_positions.shape[0],
                padded_leaf_size,
                leaf_positions.shape[-1],
            )[:, :leaf_size_local]
            return scatter_contributions(
                jnp.zeros_like(positions_sorted),
                leaf_particle_idx,
                flat_leaf_acc,
                leaf_mask,
            )

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)
        leaf_tile_scatter_dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(1, 2, 3),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )

        def _chunk_body(leaf_tile_acc: Any, start: Any) -> tuple[Any, None]:
            edge_idx = start + chunk_offsets_local
            in_range = edge_idx < edge_count_local
            safe_edge_idx = jnp.where(in_range, edge_idx, 0)
            valid_edge = in_range & valid_pairs[safe_edge_idx]

            def _compute(leaf_tile_acc_in: Any) -> Any:
                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos_tiles = leaf_pos_tiles[tgt_leaf_local]
                tgt_mask_tiles = leaf_mask_tiles[tgt_leaf_local] & valid_edge[:, None, None]
                src_pos_tiles = leaf_pos_tiles[src_leaf_local]
                src_mass_tiles = leaf_mass_tiles[src_leaf_local]
                src_mask_tiles = leaf_mask_tiles[src_leaf_local] & valid_edge[:, None, None]

                batch_tgt_pos = jnp.broadcast_to(
                    tgt_pos_tiles[:, :, None, :, :],
                    (
                        chunk_local,
                        tile_count,
                        tile_count,
                        tile_local,
                        leaf_positions.shape[-1],
                    ),
                ).reshape(-1, tile_local, leaf_positions.shape[-1])
                batch_tgt_mask = jnp.broadcast_to(
                    tgt_mask_tiles[:, :, None, :],
                    (chunk_local, tile_count, tile_count, tile_local),
                ).reshape(-1, tile_local)
                batch_src_pos = jnp.broadcast_to(
                    src_pos_tiles[:, None, :, :, :],
                    (
                        chunk_local,
                        tile_count,
                        tile_count,
                        tile_local,
                        leaf_positions.shape[-1],
                    ),
                ).reshape(-1, tile_local, leaf_positions.shape[-1])
                batch_src_mass = jnp.broadcast_to(
                    src_mass_tiles[:, None, :, :],
                    (chunk_local, tile_count, tile_count, tile_local),
                ).reshape(-1, tile_local)
                batch_src_mask = jnp.broadcast_to(
                    src_mask_tiles[:, None, :, :],
                    (chunk_local, tile_count, tile_count, tile_local),
                ).reshape(-1, tile_local)

                pair_tile_acc, _ = pair_contributions_batched(
                    batch_tgt_pos,
                    batch_tgt_mask,
                    batch_src_pos,
                    batch_src_mass,
                    batch_src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                edge_target_tile_acc = pair_tile_acc.reshape(
                    chunk_local,
                    tile_count,
                    tile_count,
                    tile_local,
                    leaf_positions.shape[-1],
                ).sum(axis=2)
                edge_leaf_acc = edge_target_tile_acc.reshape(
                    chunk_local,
                    padded_leaf_size,
                    leaf_positions.shape[-1],
                )
                reduced_tgt_leaf_local, reduced_leaf_acc, reduced_valid = (
                    reduce_pair_bucket_by_target_leaf(
                        tgt_leaf_local,
                        valid_edge,
                        edge_leaf_acc,
                    )
                )
                reduced_leaf_tile_acc = reduced_leaf_acc.reshape(
                    chunk_local,
                    tile_count,
                    tile_local,
                    leaf_positions.shape[-1],
                )
                masked_reduced_leaf_tile_acc = jnp.where(
                    reduced_valid[:, None, None, None],
                    reduced_leaf_tile_acc,
                    0.0,
                )
                valid_count = jnp.sum(reduced_valid.astype(INDEX_DTYPE))
                masked_reduced_tgt_leaf_local = jnp.where(
                    reduced_valid,
                    reduced_tgt_leaf_local,
                    0,
                )
                return jax.lax.cond(
                    valid_count > 0,
                    lambda acc_in: jax.lax.scatter_add(
                        acc_in,
                        masked_reduced_tgt_leaf_local[:, None],
                        masked_reduced_leaf_tile_acc,
                        leaf_tile_scatter_dnums,
                        indices_are_sorted=True,
                        unique_indices=True,
                        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                    ),
                    lambda acc_in: acc_in,
                    leaf_tile_acc_in,
                )

            return (
                jax.lax.cond(
                    jnp.any(valid_edge),
                    _compute,
                    lambda leaf_tile_acc_in: leaf_tile_acc_in,
                    leaf_tile_acc,
                ),
                None,
            )

        leaf_tile_acc_out, _ = jax.lax.scan(
            _chunk_body,
            leaf_tile_acc0,
            starts_local,
        )
        flat_leaf_acc = leaf_tile_acc_out.reshape(
            leaf_positions.shape[0],
            padded_leaf_size,
            leaf_positions.shape[-1],
        )[:, :leaf_size_local]
        return scatter_contributions(
            jnp.zeros_like(positions_sorted),
            leaf_particle_idx,
            flat_leaf_acc,
            leaf_mask,
        )

    @partial(jax.jit, static_argnames=("edge_chunk_size", "tile_size"))
    def _compute_pair_target_sorted_leaf_tile_fused_primitive_impl(
        positions_sorted: Any,
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
        tile_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        edge_count_local = target_leaf_ids.shape[0]
        leaf_size_local = int(leaf_positions.shape[1])
        tile_local = int(tile_size)
        if tile_local <= 0:
            raise ValueError("tile_size must be positive")

        tile_count = (leaf_size_local + tile_local - 1) // tile_local
        padded_leaf_size = tile_count * tile_local
        leaf_pad = padded_leaf_size - leaf_size_local

        padded_leaf_positions = jnp.pad(
            leaf_positions,
            ((0, 0), (0, leaf_pad), (0, 0)),
        )
        padded_leaf_masses = jnp.pad(
            leaf_masses,
            ((0, 0), (0, leaf_pad)),
        )
        padded_leaf_mask = jnp.pad(
            leaf_mask,
            ((0, 0), (0, leaf_pad)),
            constant_values=False,
        )

        leaf_pos_tiles = padded_leaf_positions.reshape(
            leaf_positions.shape[0],
            tile_count,
            tile_local,
            leaf_positions.shape[-1],
        )
        leaf_mass_tiles = padded_leaf_masses.reshape(
            leaf_masses.shape[0],
            tile_count,
            tile_local,
        )
        leaf_mask_tiles = padded_leaf_mask.reshape(
            leaf_mask.shape[0],
            tile_count,
            tile_local,
        )

        leaf_tile_acc0 = jnp.zeros(
            (
                leaf_positions.shape[0],
                tile_count,
                tile_local,
                leaf_positions.shape[-1],
            ),
            dtype=dtype_local,
        )
        if edge_count_local == 0:
            flat_leaf_acc = leaf_tile_acc0.reshape(
                leaf_positions.shape[0],
                padded_leaf_size,
                leaf_positions.shape[-1],
            )[:, :leaf_size_local]
            return scatter_contributions(
                jnp.zeros_like(positions_sorted),
                leaf_particle_idx,
                flat_leaf_acc,
                leaf_mask,
            )

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)
        leaf_tile_scatter_dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(1, 2, 3),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )

        def _chunk_body(leaf_tile_acc: Any, start: Any) -> tuple[Any, None]:
            edge_idx = start + chunk_offsets_local
            in_range = edge_idx < edge_count_local
            safe_edge_idx = jnp.where(in_range, edge_idx, 0)
            valid_edge = in_range & valid_pairs[safe_edge_idx]

            def _compute(leaf_tile_acc_in: Any) -> Any:
                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos_tiles = leaf_pos_tiles[tgt_leaf_local]
                tgt_mask_tiles = leaf_mask_tiles[tgt_leaf_local] & valid_edge[:, None, None]
                src_pos_tiles = leaf_pos_tiles[src_leaf_local]
                src_mass_tiles = leaf_mass_tiles[src_leaf_local]
                src_mask_tiles = leaf_mask_tiles[src_leaf_local] & valid_edge[:, None, None]

                batch_tgt_pos = tgt_pos_tiles.reshape(
                    -1,
                    tile_local,
                    leaf_positions.shape[-1],
                )
                batch_tgt_mask = tgt_mask_tiles.reshape(-1, tile_local)

                def _src_tile_body(tile_acc_flat: Any, src_tile_idx: Any) -> tuple[Any, None]:
                    src_pos_one = src_pos_tiles[:, src_tile_idx, :, :]
                    src_mass_one = src_mass_tiles[:, src_tile_idx, :]
                    src_mask_one = src_mask_tiles[:, src_tile_idx, :]

                    batch_src_pos = jnp.repeat(
                        src_pos_one[:, None, :, :],
                        tile_count,
                        axis=1,
                    ).reshape(-1, tile_local, leaf_positions.shape[-1])
                    batch_src_mass = jnp.repeat(
                        src_mass_one[:, None, :],
                        tile_count,
                        axis=1,
                    ).reshape(-1, tile_local)
                    batch_src_mask = jnp.repeat(
                        src_mask_one[:, None, :],
                        tile_count,
                        axis=1,
                    ).reshape(-1, tile_local)

                    contrib = jax.vmap(
                        lambda tp, tm, sp, sm, smask: nearfield_tile_pair_accel(
                            tp,
                            tm,
                            sp,
                            sm,
                            smask,
                            softening_sq=softening_sq,
                            G=G,
                        )
                    )(
                        batch_tgt_pos,
                        batch_tgt_mask,
                        batch_src_pos,
                        batch_src_mass,
                        batch_src_mask,
                    )
                    return (
                        tile_acc_flat + contrib.reshape(
                            chunk_local,
                            tile_count,
                            tile_local,
                            leaf_positions.shape[-1],
                        ),
                        None,
                    )

                edge_target_tile_acc0 = jnp.zeros(
                    (
                        chunk_local,
                        tile_count,
                        tile_local,
                        leaf_positions.shape[-1],
                    ),
                    dtype=dtype_local,
                )
                edge_target_tile_acc, _ = jax.lax.scan(
                    _src_tile_body,
                    edge_target_tile_acc0,
                    jnp.arange(tile_count, dtype=INDEX_DTYPE),
                )
                edge_leaf_acc = edge_target_tile_acc.reshape(
                    chunk_local,
                    padded_leaf_size,
                    leaf_positions.shape[-1],
                )
                reduced_tgt_leaf_local, reduced_leaf_acc, reduced_valid = (
                    reduce_pair_bucket_by_target_leaf(
                        tgt_leaf_local,
                        valid_edge,
                        edge_leaf_acc,
                    )
                )
                reduced_leaf_tile_acc = reduced_leaf_acc.reshape(
                    chunk_local,
                    tile_count,
                    tile_local,
                    leaf_positions.shape[-1],
                )
                masked_reduced_leaf_tile_acc = jnp.where(
                    reduced_valid[:, None, None, None],
                    reduced_leaf_tile_acc,
                    0.0,
                )
                valid_count = jnp.sum(reduced_valid.astype(INDEX_DTYPE))
                masked_reduced_tgt_leaf_local = jnp.where(
                    reduced_valid,
                    reduced_tgt_leaf_local,
                    0,
                )
                return jax.lax.cond(
                    valid_count > 0,
                    lambda acc_in: jax.lax.scatter_add(
                        acc_in,
                        masked_reduced_tgt_leaf_local[:, None],
                        masked_reduced_leaf_tile_acc,
                        leaf_tile_scatter_dnums,
                        indices_are_sorted=True,
                        unique_indices=True,
                        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                    ),
                    lambda acc_in: acc_in,
                    leaf_tile_acc_in,
                )

            return (
                jax.lax.cond(
                    jnp.any(valid_edge),
                    _compute,
                    lambda leaf_tile_acc_in: leaf_tile_acc_in,
                    leaf_tile_acc,
                ),
                None,
            )

        leaf_tile_acc_out, _ = jax.lax.scan(
            _chunk_body,
            leaf_tile_acc0,
            starts_local,
        )
        flat_leaf_acc = leaf_tile_acc_out.reshape(
            leaf_positions.shape[0],
            padded_leaf_size,
            leaf_positions.shape[-1],
        )[:, :leaf_size_local]
        return scatter_contributions(
            jnp.zeros_like(positions_sorted),
            leaf_particle_idx,
            flat_leaf_acc,
            leaf_mask,
        )

    @partial(
        jax.jit,
        static_argnames=("edge_chunk_size", "chunks_per_superchunk"),
    )
    def _compute_pair_delayed_scatter_only_impl(
        positions_sorted: Any,
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
        chunks_per_superchunk: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        edge_count_local = target_leaf_ids.shape[0]
        acc0 = jnp.zeros_like(positions_sorted)
        if edge_count_local == 0:
            return acc0

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")
        superchunk_local = int(chunks_per_superchunk)
        if superchunk_local <= 0:
            raise ValueError("chunks_per_superchunk must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)
        super_starts = jnp.arange(
            0,
            starts_local.shape[0],
            superchunk_local,
            dtype=INDEX_DTYPE,
        )
        super_offsets = jnp.arange(superchunk_local, dtype=INDEX_DTYPE)

        def _superchunk_body(acc: Any, super_start_idx: Any) -> tuple[Any, None]:
            def _chunk_probe(offset_idx: Any) -> tuple[Any, Any, Any]:
                chunk_idx = super_start_idx + offset_idx
                in_super_range = chunk_idx < starts_local.shape[0]
                safe_chunk_idx = jnp.where(in_super_range, chunk_idx, 0)
                start = starts_local[safe_chunk_idx]
                edge_idx = start + chunk_offsets_local
                in_range = in_super_range & (edge_idx < edge_count_local)
                safe_edge_idx = jnp.where(in_range, edge_idx, 0)
                valid_edge = in_range & valid_pairs[safe_edge_idx]

                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos = leaf_positions[tgt_leaf_local]
                tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                src_pos = leaf_positions[src_leaf_local]
                src_mass = leaf_masses[src_leaf_local]
                src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos,
                    tgt_mask,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                reduced_tgt_leaf_local, reduced_pair_acc, reduced_valid = (
                    reduce_pair_bucket_by_target_leaf(
                        tgt_leaf_local,
                        valid_edge,
                        pair_acc,
                    )
                )
                reduced_tgt_ids = leaf_particle_idx[reduced_tgt_leaf_local]
                reduced_tgt_mask = leaf_mask[reduced_tgt_leaf_local] & reduced_valid[:, None]
                return reduced_tgt_ids, reduced_pair_acc, reduced_tgt_mask

            super_ids, super_values, super_mask = jax.vmap(_chunk_probe)(super_offsets)
            return (
                scatter_contributions(
                    acc,
                    super_ids.reshape(-1, super_ids.shape[-1]),
                    super_values.reshape(-1, super_values.shape[-2], super_values.shape[-1]),
                    super_mask.reshape(-1, super_mask.shape[-1]),
                ),
                None,
            )

        acc_out, _ = jax.lax.scan(
            _superchunk_body,
            acc0,
            super_starts,
        )
        return acc_out

    @partial(
        jax.jit,
        static_argnames=("edge_chunk_size", "chunks_per_superchunk"),
    )
    def _compute_pair_packed_unique_scatter_only_impl(
        positions_sorted: Any,
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
        chunks_per_superchunk: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        edge_count_local = target_leaf_ids.shape[0]
        acc0 = jnp.zeros_like(positions_sorted)
        if edge_count_local == 0:
            return acc0

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")
        superchunk_local = int(chunks_per_superchunk)
        if superchunk_local <= 0:
            raise ValueError("chunks_per_superchunk must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)
        super_starts = jnp.arange(
            0,
            starts_local.shape[0],
            superchunk_local,
            dtype=INDEX_DTYPE,
        )
        super_offsets = jnp.arange(superchunk_local, dtype=INDEX_DTYPE)

        def _superchunk_body(acc: Any, super_start_idx: Any) -> tuple[Any, None]:
            def _chunk_probe(offset_idx: Any) -> tuple[Any, Any, Any]:
                chunk_idx = super_start_idx + offset_idx
                in_super_range = chunk_idx < starts_local.shape[0]
                safe_chunk_idx = jnp.where(in_super_range, chunk_idx, 0)
                start = starts_local[safe_chunk_idx]
                edge_idx = start + chunk_offsets_local
                in_range = in_super_range & (edge_idx < edge_count_local)
                safe_edge_idx = jnp.where(in_range, edge_idx, 0)
                valid_edge = in_range & valid_pairs[safe_edge_idx]

                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos = leaf_positions[tgt_leaf_local]
                tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                src_pos = leaf_positions[src_leaf_local]
                src_mass = leaf_masses[src_leaf_local]
                src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos,
                    tgt_mask,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                reduced_tgt_leaf_local, reduced_pair_acc, reduced_valid = (
                    reduce_pair_bucket_by_target_leaf(
                        tgt_leaf_local,
                        valid_edge,
                        pair_acc,
                    )
                )
                reduced_tgt_ids = leaf_particle_idx[reduced_tgt_leaf_local]
                reduced_tgt_mask = leaf_mask[reduced_tgt_leaf_local] & reduced_valid[:, None]
                return reduced_tgt_ids, reduced_pair_acc, reduced_tgt_mask

            super_ids, super_values, super_mask = jax.vmap(_chunk_probe)(super_offsets)
            unique_indices, unique_values, unique_valid = (
                pack_unique_particle_vector_updates(
                    super_ids.reshape(-1, super_ids.shape[-1]),
                    super_values.reshape(
                        -1,
                        super_values.shape[-2],
                        super_values.shape[-1],
                    ),
                    super_mask.reshape(-1, super_mask.shape[-1]),
                )
            )
            return (
                apply_packed_particle_vector_updates(
                    acc,
                    unique_indices,
                    unique_values,
                    unique_valid,
                ),
                None,
            )

        acc_out, _ = jax.lax.scan(
            _superchunk_body,
            acc0,
            super_starts,
        )
        return acc_out

    @partial(jax.jit, static_argnames=("edge_chunk_size",))
    def _compute_pair_particle_index_only_impl(
        target_leaf_ids: Any,
        source_leaf_ids: Any,
        valid_pairs: Any,
        leaf_positions: Any,
        leaf_masses: Any,
        leaf_mask: Any,
        leaf_particle_idx: Any,
        *,
        G: Any,
        softening_sq: Any,
        edge_chunk_size: int,
    ) -> Any:
        dtype_local = leaf_positions.dtype
        g_const_local = jnp.asarray(G, dtype=dtype_local)
        edge_count_local = target_leaf_ids.shape[0]
        if edge_count_local == 0:
            return jnp.asarray(0.0, dtype=dtype_local)

        chunk_local = int(edge_chunk_size)
        if chunk_local <= 0:
            raise ValueError("edge_chunk_size must be positive")

        chunk_offsets_local = jnp.arange(chunk_local, dtype=INDEX_DTYPE)
        starts_local = jnp.arange(0, edge_count_local, chunk_local, dtype=INDEX_DTYPE)

        def _chunk_body(total: Any, start: Any) -> tuple[Any, None]:
            edge_idx = start + chunk_offsets_local
            in_range = edge_idx < edge_count_local
            safe_edge_idx = jnp.where(in_range, edge_idx, 0)
            valid_edge = in_range & valid_pairs[safe_edge_idx]

            def _compute(total_in: Any) -> Any:
                tgt_leaf = target_leaf_ids[safe_edge_idx]
                src_leaf = source_leaf_ids[safe_edge_idx]
                tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                tgt_pos = leaf_positions[tgt_leaf_local]
                tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                src_pos = leaf_positions[src_leaf_local]
                src_mass = leaf_masses[src_leaf_local]
                src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                pair_acc, _ = pair_contributions_batched(
                    tgt_pos,
                    tgt_mask,
                    src_pos,
                    src_mass,
                    src_mask,
                    softening_sq=softening_sq,
                    G=g_const_local,
                    compute_potential=False,
                )
                reduced_tgt_leaf_local, reduced_pair_acc, reduced_valid = (
                    reduce_pair_bucket_by_target_leaf(
                        tgt_leaf_local,
                        valid_edge,
                        pair_acc,
                    )
                )
                reduced_tgt_ids = leaf_particle_idx[reduced_tgt_leaf_local]
                reduced_tgt_mask = leaf_mask[reduced_tgt_leaf_local] & reduced_valid[:, None]
                return (
                    total_in
                    + jnp.sum(reduced_pair_acc, dtype=dtype_local)
                    + jnp.sum(reduced_tgt_ids.astype(dtype_local))
                    + jnp.sum(reduced_tgt_mask.astype(dtype_local))
                )

            return (
                jax.lax.cond(
                    jnp.any(valid_edge),
                    _compute,
                    lambda total_in: total_in,
                    total,
                ),
                None,
            )

        total_probe, _ = jax.lax.scan(
            _chunk_body,
            jnp.asarray(0.0, dtype=dtype_local),
            starts_local,
        )
        return total_probe

    def _prepare_specialized_nearfield_inputs(
        *,
        sort_by_target: bool = False,
    ) -> tuple[Any, ...]:
        positions_sorted = jnp.asarray(prepared_state.positions_sorted)
        masses_sorted = jnp.asarray(prepared_state.masses_sorted)
        node_ranges = jnp.asarray(prepared_state.tree.node_ranges, dtype=INDEX_DTYPE)
        leaf_nodes = jnp.asarray(prepared_state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
        offsets = jnp.asarray(prepared_state.neighbor_list.offsets, dtype=INDEX_DTYPE)
        neighbors = jnp.asarray(prepared_state.neighbor_list.neighbors, dtype=INDEX_DTYPE)

        if prepared_state.nearfield_target_leaf_ids is None or prepared_state.nearfield_valid_pairs is None:
            target_leaf_ids, source_leaf_ids, valid_pairs = prepare_leaf_neighbor_pairs(
                node_ranges,
                leaf_nodes,
                offsets,
                neighbors,
                sort_by_source=False,
            )
        else:
            target_leaf_ids = jnp.asarray(
                prepared_state.nearfield_target_leaf_ids,
                dtype=INDEX_DTYPE,
            )
            valid_pairs = jnp.asarray(prepared_state.nearfield_valid_pairs, dtype=bool)
            if prepared_state.nearfield_source_leaf_ids is None:
                total_nodes = node_ranges.shape[0]
                leaf_lookup = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
                leaf_lookup = leaf_lookup.at[leaf_nodes].set(
                    jnp.arange(leaf_nodes.shape[0], dtype=INDEX_DTYPE)
                )
                source_leaf_ids = leaf_lookup[neighbors]
                valid_pairs = valid_pairs & (source_leaf_ids >= 0)
            else:
                source_leaf_ids = jnp.asarray(
                    prepared_state.nearfield_source_leaf_ids,
                    dtype=INDEX_DTYPE,
                )

        if sort_by_target:
            invalid_key = jnp.asarray(
                jnp.iinfo(INDEX_DTYPE).max,
                dtype=INDEX_DTYPE,
            )
            sort_key = jnp.where(valid_pairs, target_leaf_ids, invalid_key)
            sort_idx = jnp.argsort(sort_key, stable=True)
            target_leaf_ids = target_leaf_ids[sort_idx]
            source_leaf_ids = source_leaf_ids[sort_idx]
            valid_pairs = valid_pairs[sort_idx]

        (
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
        ) = prepare_leaf_data_from_groups(
            jnp.asarray(leaf_particle_indices, dtype=INDEX_DTYPE),
            (
                jnp.asarray(leaf_particle_mask, dtype=bool)
                if leaf_particle_mask is not None
                else jnp.ones_like(jnp.asarray(leaf_particle_indices, dtype=INDEX_DTYPE), dtype=bool)
            ),
            positions_sorted,
            masses_sorted,
        )
        softening_sq = jnp.asarray(
            float(getattr(fmm._impl, "softening")) ** 2,
            dtype=positions_sorted.dtype,
        )
        return (
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        )

    def _prepare_target_leaf_owned_nearfield_inputs() -> tuple[Any, ...]:
        positions_sorted = jnp.asarray(prepared_state.positions_sorted)
        masses_sorted = jnp.asarray(prepared_state.masses_sorted)
        node_ranges = jnp.asarray(prepared_state.tree.node_ranges, dtype=INDEX_DTYPE)
        leaf_nodes = jnp.asarray(prepared_state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
        offsets = jnp.asarray(prepared_state.neighbor_list.offsets, dtype=INDEX_DTYPE)
        neighbors = jnp.asarray(prepared_state.neighbor_list.neighbors, dtype=INDEX_DTYPE)

        if (
            prepared_state.nearfield_source_leaf_ids is None
            or prepared_state.nearfield_valid_pairs is None
        ):
            _target_leaf_ids, source_leaf_ids, valid_pairs = prepare_leaf_neighbor_pairs(
                node_ranges,
                leaf_nodes,
                offsets,
                neighbors,
                sort_by_source=False,
            )
        else:
            source_leaf_ids = jnp.asarray(
                prepared_state.nearfield_source_leaf_ids,
                dtype=INDEX_DTYPE,
            )
            valid_pairs = jnp.asarray(prepared_state.nearfield_valid_pairs, dtype=bool)

        (
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
        ) = prepare_leaf_data_from_groups(
            jnp.asarray(leaf_particle_indices, dtype=INDEX_DTYPE),
            (
                jnp.asarray(leaf_particle_mask, dtype=bool)
                if leaf_particle_mask is not None
                else jnp.ones_like(
                    jnp.asarray(leaf_particle_indices, dtype=INDEX_DTYPE),
                    dtype=bool,
                )
            ),
            positions_sorted,
            masses_sorted,
        )
        softening_sq = jnp.asarray(
            float(getattr(fmm._impl, "softening")) ** 2,
            dtype=positions_sorted.dtype,
        )
        max_neighbors_per_target = int(
            jnp.max(offsets[1:] - offsets[:-1]) if int(offsets.shape[0]) > 1 else 0
        )
        return (
            positions_sorted,
            source_leaf_ids,
            valid_pairs,
            offsets,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
            max_neighbors_per_target,
        )

    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    tuned_kwargs, worker_tune_info = _worker_autotune_runtime_kwargs(
        cfg=cfg,
        fmm_kwargs=fmm_kwargs,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        autotune_cache_path=autotune_cache_path,
    )
    fmm = FastMultipoleMethod(**tuned_kwargs)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        if cache_path.exists():
            fmm.load_m2l_autotune_cache(str(cache_path), merge=True)

    _warm_sweep_case(
        fmm=fmm,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        benchmark_scope="steady_eval",
    )

    prepared_state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )
    prepared_state = _block_ready(prepared_state)
    _emit_ready_marker()

    row = {
        "num_particles": int(num_particles),
        "prepared_state_mb": float(_prepared_state_total_mb(prepared_state)),
        "nearfield_component_mode": "unsupported",
        "error": "",
    }
    row.update(
        _resolved_nearfield_runtime_report(
            fmm,
            num_particles=int(num_particles),
        )
    )
    row.update(bench_utils.resolved_large_n_memory_path_report(fmm))
    row.update(worker_tune_info)
    delayed_scatter_chunks_per_superchunk = int(
        cfg.get("delayed_scatter_chunks_per_superchunk", 4)
    )
    row["nearfield_specialized_pair_delayed_scatter_chunks_per_superchunk"] = int(
        delayed_scatter_chunks_per_superchunk
    )
    target_tile_size = int(cfg.get("target_tile_size", 32))
    row["nearfield_specialized_pair_target_tile_size"] = int(target_tile_size)
    target_leaf_batch_size = int(cfg.get("target_leaf_batch_size", 32))
    row["nearfield_specialized_pair_target_leaf_batch_size"] = int(
        target_leaf_batch_size
    )
    target_leaf_neighbor_block_size = int(
        cfg.get("target_leaf_neighbor_block_size", 16)
    )
    row["nearfield_specialized_pair_target_leaf_neighbor_block_size"] = int(
        target_leaf_neighbor_block_size
    )
    row["nearfield_specialized_pair_tile_primitive_supported"] = bool(
        pallas_nearfield_tile_pair_supported()
    )
    row["nearfield_specialized_pair_tile_primitive_backend"] = (
        nearfield_tile_pair_backend()
    )
    row["nearfield_specialized_pair_packed_unique_updates_supported"] = bool(
        pallas_nearfield_unique_updates_supported()
    )
    row["nearfield_specialized_pair_packed_unique_updates_backend"] = (
        nearfield_unique_updates_backend()
    )

    if not isinstance(prepared_state, LargeNPreparedState):
        row["error"] = "nearfield component breakdown requires LargeNPreparedState"
        return row

    row["radix_fast_lane_active"] = bool(
        getattr(prepared_state, "radix_fast_lane", False)
    )
    if (
        bool(getattr(prepared_state, "radix_fast_lane", False))
        and getattr(prepared_state, "radix_fast_payload", None) is not None
    ):
        fast_payload = prepared_state.radix_fast_payload
        counters = collect_radix_fast_lane_counters(
            payload=fast_payload,
            positions_dtype=prepared_state.positions_sorted.dtype,
            masses_dtype=prepared_state.masses_sorted.dtype,
            accelerations_dtype=prepared_state.positions_sorted.dtype,
        )
        row.update(
            {
                "nearfield_radix_fast_lane_gather_bytes": int(counters.gather_bytes),
                "nearfield_radix_fast_lane_scatter_bytes": int(counters.scatter_bytes),
                "nearfield_radix_fast_lane_scatter_ops": int(counters.scatter_ops),
                "nearfield_radix_fast_lane_target_batches": int(
                    counters.target_batches
                ),
                "nearfield_radix_fast_lane_source_slot_tiles": int(
                    counters.source_slot_tiles
                ),
            }
        )

    leaf_particle_indices = prepared_state.nearfield_leaf_particle_indices
    leaf_particle_mask = prepared_state.nearfield_leaf_particle_mask
    specialized_path_active = (
        int(leaf_particle_indices.size) > 0
        and str(prepared_state.nearfield_mode).strip().lower() == "bucketed"
        and prepared_state.nearfield_chunk_sort_indices is None
        and prepared_state.nearfield_chunk_group_ids is None
        and prepared_state.nearfield_chunk_unique_indices is None
        and str(
            os.environ.get("JACCPOT_DISABLE_LARGE_N_SPECIALIZED_NEARFIELD", "0")
        ).strip().lower()
        not in {"1", "true", "yes", "on"}
    )
    row["specialized_path_active"] = bool(specialized_path_active)
    if not specialized_path_active:
        row["nearfield_component_mode"] = "generic_only"
        row["error"] = "specialized large-N nearfield path is not active for this run"
        return row

    nearfield_total_timing = bench_utils.time_callable(
        evaluate_large_n_nearfield,
        fmm._impl,
        prepared_state,
        warmup=int(warmup),
        runs=int(runs),
        return_potential=False,
    )

    def _self_component() -> Any:
        (
            positions_sorted,
            _target_leaf_ids,
            _source_leaf_ids,
            _valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs()
        return compute_self_only(
            positions_sorted,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
        )

    def _pair_component() -> Any:
        (
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs()
        return compute_pairs_only(
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
            chunks_per_superchunk=1,
            sorted_scatter_hint=False,
            grouped_sorted_scatter=False,
            superchunk_target_reduce=False,
            disable_chunk_cond=True,
        )

    def _pair_arith_component() -> Any:
        (
            _positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            _leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs()
        return _compute_pair_arith_only_impl(
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
        )

    def _pair_reduction_component() -> Any:
        (
            _positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            _leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs()
        return _compute_pair_reduction_only_impl(
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
        )

    def _pair_scatter_component() -> Any:
        (
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs()
        return _compute_pair_scatter_only_impl(
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
        )

    def _pair_lax_scatter_component() -> Any:
        (
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs()
        return _compute_pair_lax_scatter_only_impl(
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
        )

    def _pair_compacted_scatter_component() -> Any:
        (
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs()
        return _compute_pair_compacted_scatter_only_impl(
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
        )

    def _pair_compacted_component() -> Any:
        (
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs()
        return _compute_pair_compacted_only_impl(
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
        )

    def _pair_leaf_accum_component() -> Any:
        (
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs()
        return _compute_pair_leaf_accum_only_impl(
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
        )

    def _pair_target_sorted_leaf_accum_component() -> Any:
        (
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs(sort_by_target=True)
        return _compute_pair_target_sorted_leaf_accum_only_impl(
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
        )

    def _pair_target_leaf_owned_component() -> Any:
        (
            positions_sorted,
            source_leaf_ids,
            valid_pairs,
            target_offsets,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
            max_neighbors_per_target,
        ) = _prepare_target_leaf_owned_nearfield_inputs()
        return _compute_pair_target_leaf_owned_impl(
            positions_sorted,
            source_leaf_ids,
            valid_pairs,
            target_offsets,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            max_neighbors_per_target=int(max_neighbors_per_target),
        )

    def _pair_target_leaf_batched_component() -> Any:
        (
            positions_sorted,
            source_leaf_ids,
            valid_pairs,
            target_offsets,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
            max_neighbors_per_target,
        ) = _prepare_target_leaf_owned_nearfield_inputs()
        return _compute_pair_target_leaf_batched_impl(
            positions_sorted,
            source_leaf_ids,
            valid_pairs,
            target_offsets,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            max_neighbors_per_target=int(max_neighbors_per_target),
            target_batch_size=int(target_leaf_batch_size),
        )

    def _pair_target_leaf_bucketed_batched_component() -> Any:
        (
            positions_sorted,
            source_leaf_ids,
            valid_pairs,
            target_offsets,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
            max_neighbors_per_target,
        ) = _prepare_target_leaf_owned_nearfield_inputs()
        return _compute_pair_target_leaf_bucketed_batched_impl(
            positions_sorted,
            source_leaf_ids,
            valid_pairs,
            target_offsets,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            max_neighbors_per_target=int(max_neighbors_per_target),
            target_batch_size=int(target_leaf_batch_size),
            neighbor_block_size=int(target_leaf_neighbor_block_size),
        )

    def _pair_target_sorted_particle_tile_accum_component() -> Any:
        (
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs(sort_by_target=True)
        return _compute_pair_target_sorted_particle_tile_accum_only_impl(
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
            tile_size=int(target_tile_size),
        )

    def _pair_target_sorted_leaf_tile_microkernel_component() -> Any:
        (
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs(sort_by_target=True)
        return _compute_pair_target_sorted_leaf_tile_microkernel_impl(
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
            tile_size=int(target_tile_size),
        )

    def _pair_target_sorted_leaf_tile_fused_primitive_component() -> Any:
        (
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs(sort_by_target=True)
        return _compute_pair_target_sorted_leaf_tile_fused_primitive_impl(
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
            tile_size=int(target_tile_size),
        )

    def _pair_delayed_scatter_component() -> Any:
        (
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs()
        return _compute_pair_delayed_scatter_only_impl(
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
            chunks_per_superchunk=int(delayed_scatter_chunks_per_superchunk),
        )

    def _pair_packed_unique_scatter_component() -> Any:
        (
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs()
        return _compute_pair_packed_unique_scatter_only_impl(
            positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
            chunks_per_superchunk=int(delayed_scatter_chunks_per_superchunk),
        )

    def _pair_particle_index_component() -> Any:
        (
            _positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            softening_sq,
        ) = _prepare_specialized_nearfield_inputs()
        return _compute_pair_particle_index_only_impl(
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=getattr(fmm._impl, "G"),
            softening_sq=softening_sq,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
        )

    def _pair_gather_component() -> Any:
        (
            _positions_sorted,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            _softening_sq,
        ) = _prepare_specialized_nearfield_inputs()
        return _compute_pair_gather_only_impl(
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            edge_chunk_size=int(prepared_state.nearfield_edge_chunk_size),
        )

    self_timing = bench_utils.time_callable(
        _self_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_timing = bench_utils.time_callable(
        _pair_component,
        warmup=int(warmup),
        runs=int(runs),
    )

    row.update(
        {
            "nearfield_component_mode": "specialized_split",
            "evaluate_large_n_nearfield_seconds": float(nearfield_total_timing.mean),
            "evaluate_large_n_nearfield_std_seconds": float(nearfield_total_timing.std),
            "nearfield_specialized_self_seconds": float(self_timing.mean),
            "nearfield_specialized_self_std_seconds": float(self_timing.std),
            "nearfield_specialized_pairs_seconds": float(pair_timing.mean),
            "nearfield_specialized_pairs_std_seconds": float(pair_timing.std),
        }
    )
    if production_only:
        row["nearfield_component_mode"] = "specialized_split_production"
        return row

    pair_target_sorted_leaf_tile_microkernel_timing = bench_utils.time_callable(
        _pair_target_sorted_leaf_tile_microkernel_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_target_sorted_leaf_tile_fused_primitive_timing = bench_utils.time_callable(
        _pair_target_sorted_leaf_tile_fused_primitive_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_delayed_scatter_timing = bench_utils.time_callable(
        _pair_delayed_scatter_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_target_leaf_batched_timing = bench_utils.time_callable(
        _pair_target_leaf_batched_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_target_leaf_bucketed_batched_timing = bench_utils.time_callable(
        _pair_target_leaf_bucketed_batched_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    row.update(
        {
            "nearfield_specialized_pair_target_sorted_leaf_tile_microkernel_seconds": float(
                pair_target_sorted_leaf_tile_microkernel_timing.mean
            ),
            "nearfield_specialized_pair_target_sorted_leaf_tile_microkernel_std_seconds": float(
                pair_target_sorted_leaf_tile_microkernel_timing.std
            ),
            "nearfield_specialized_pair_target_sorted_leaf_tile_fused_primitive_seconds": float(
                pair_target_sorted_leaf_tile_fused_primitive_timing.mean
            ),
            "nearfield_specialized_pair_target_sorted_leaf_tile_fused_primitive_std_seconds": float(
                pair_target_sorted_leaf_tile_fused_primitive_timing.std
            ),
            "nearfield_specialized_pair_delayed_scatter_seconds": float(
                pair_delayed_scatter_timing.mean
            ),
            "nearfield_specialized_pair_delayed_scatter_std_seconds": float(
                pair_delayed_scatter_timing.std
            ),
            "nearfield_specialized_pair_target_leaf_batched_seconds": float(
                pair_target_leaf_batched_timing.mean
            ),
            "nearfield_specialized_pair_target_leaf_batched_std_seconds": float(
                pair_target_leaf_batched_timing.std
            ),
            "nearfield_specialized_pair_target_leaf_bucketed_batched_seconds": float(
                pair_target_leaf_bucketed_batched_timing.mean
            ),
            "nearfield_specialized_pair_target_leaf_bucketed_batched_std_seconds": float(
                pair_target_leaf_bucketed_batched_timing.std
            ),
        }
    )

    if not focused_only:
        pair_arith_timing = bench_utils.time_callable(
            _pair_arith_component,
            warmup=int(warmup),
            runs=int(runs),
        )
        pair_reduction_timing = bench_utils.time_callable(
            _pair_reduction_component,
            warmup=int(warmup),
            runs=int(runs),
        )
        pair_scatter_timing = bench_utils.time_callable(
            _pair_scatter_component,
            warmup=int(warmup),
            runs=int(runs),
        )
    if focused_only:
        return row

    row.update(
        {
            "nearfield_specialized_pair_arith_probe_seconds": float(
                pair_arith_timing.mean
            ),
            "nearfield_specialized_pair_arith_probe_std_seconds": float(
                pair_arith_timing.std
            ),
            "nearfield_specialized_pair_reduction_probe_seconds": float(
                pair_reduction_timing.mean
            ),
            "nearfield_specialized_pair_reduction_probe_std_seconds": float(
                pair_reduction_timing.std
            ),
            "nearfield_specialized_pair_scatter_probe_seconds": float(
                pair_scatter_timing.mean
            ),
            "nearfield_specialized_pair_scatter_probe_std_seconds": float(
                pair_scatter_timing.std
            ),
        }
    )

    pair_lax_scatter_timing = bench_utils.time_callable(
        _pair_lax_scatter_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_compacted_scatter_timing = bench_utils.time_callable(
        _pair_compacted_scatter_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_compacted_timing = bench_utils.time_callable(
        _pair_compacted_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_leaf_accum_timing = bench_utils.time_callable(
        _pair_leaf_accum_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_target_sorted_leaf_accum_timing = bench_utils.time_callable(
        _pair_target_sorted_leaf_accum_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_target_leaf_owned_timing = bench_utils.time_callable(
        _pair_target_leaf_owned_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_target_sorted_particle_tile_accum_timing = bench_utils.time_callable(
        _pair_target_sorted_particle_tile_accum_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_packed_unique_scatter_timing = bench_utils.time_callable(
        _pair_packed_unique_scatter_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_particle_index_timing = bench_utils.time_callable(
        _pair_particle_index_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    pair_gather_timing = bench_utils.time_callable(
        _pair_gather_component,
        warmup=int(warmup),
        runs=int(runs),
    )
    row.update(
        {
            "nearfield_specialized_pair_lax_scatter_probe_seconds": float(
                pair_lax_scatter_timing.mean
            ),
            "nearfield_specialized_pair_lax_scatter_probe_std_seconds": float(
                pair_lax_scatter_timing.std
            ),
            "nearfield_specialized_pair_compacted_scatter_probe_seconds": float(
                pair_compacted_scatter_timing.mean
            ),
            "nearfield_specialized_pair_compacted_scatter_probe_std_seconds": float(
                pair_compacted_scatter_timing.std
            ),
            "nearfield_specialized_pair_compacted_seconds": float(
                pair_compacted_timing.mean
            ),
            "nearfield_specialized_pair_compacted_std_seconds": float(
                pair_compacted_timing.std
            ),
            "nearfield_specialized_pair_leaf_accum_seconds": float(
                pair_leaf_accum_timing.mean
            ),
            "nearfield_specialized_pair_leaf_accum_std_seconds": float(
                pair_leaf_accum_timing.std
            ),
            "nearfield_specialized_pair_target_sorted_leaf_accum_seconds": float(
                pair_target_sorted_leaf_accum_timing.mean
            ),
            "nearfield_specialized_pair_target_sorted_leaf_accum_std_seconds": float(
                pair_target_sorted_leaf_accum_timing.std
            ),
            "nearfield_specialized_pair_target_leaf_owned_seconds": float(
                pair_target_leaf_owned_timing.mean
            ),
            "nearfield_specialized_pair_target_leaf_owned_std_seconds": float(
                pair_target_leaf_owned_timing.std
            ),
            "nearfield_specialized_pair_target_sorted_particle_tile_accum_seconds": float(
                pair_target_sorted_particle_tile_accum_timing.mean
            ),
            "nearfield_specialized_pair_target_sorted_particle_tile_accum_std_seconds": float(
                pair_target_sorted_particle_tile_accum_timing.std
            ),
            "nearfield_specialized_pair_packed_unique_scatter_seconds": float(
                pair_packed_unique_scatter_timing.mean
            ),
            "nearfield_specialized_pair_packed_unique_scatter_std_seconds": float(
                pair_packed_unique_scatter_timing.std
            ),
            "nearfield_specialized_pair_particle_index_probe_seconds": float(
                pair_particle_index_timing.mean
            ),
            "nearfield_specialized_pair_particle_index_probe_std_seconds": float(
                pair_particle_index_timing.std
            ),
            "nearfield_specialized_pair_gather_probe_seconds": float(
                pair_gather_timing.mean
            ),
            "nearfield_specialized_pair_gather_probe_std_seconds": float(
                pair_gather_timing.std
            ),
        }
    )
    return row


def main() -> None:
    global _EMIT_READY_MARKER
    args = _parse_args()
    _EMIT_READY_MARKER = bool(args.emit_ready_marker)
    cfg = json.loads(args.config_json)
    fmm_kwargs = _build_runtime_config(cfg)
    dtype = _dtype_from_name(args.dtype)
    autotune_cache_path: Optional[str] = args.autotune_cache
    if autotune_cache_path is None:
        env_cache = os.environ.get("JACCPOT_AUTOTUNE_CACHE_PATH")
        autotune_cache_path = None if env_cache is None else str(env_cache).strip()
    if autotune_cache_path == "":
        autotune_cache_path = None
    try:
        if args.mode == "sweep":
            row = _run_sweep_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                max_order=args.max_order,
                runs=args.runs,
                warmup=args.warmup,
                dtype=dtype,
                seed=args.seed,
                cfg=cfg,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
            )
        elif args.mode == "audit":
            row = _run_audit_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                max_order=args.max_order,
                runs=args.runs,
                warmup=args.warmup,
                dtype=dtype,
                seed=args.seed,
                cfg=cfg,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
            )
        elif args.mode == "nearfield_components":
            row = _run_nearfield_components_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                max_order=args.max_order,
                runs=args.runs,
                warmup=args.warmup,
                dtype=dtype,
                seed=args.seed,
                cfg=cfg,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
            )
        elif args.mode == "nearfield_components_production":
            row = _run_nearfield_components_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                max_order=args.max_order,
                runs=args.runs,
                warmup=args.warmup,
                dtype=dtype,
                seed=args.seed,
                cfg=cfg,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
                production_only=True,
            )
        elif args.mode == "nearfield_fused_check":
            row = _run_nearfield_components_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                max_order=args.max_order,
                runs=args.runs,
                warmup=args.warmup,
                dtype=dtype,
                seed=args.seed,
                cfg=cfg,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
                focused_only=True,
            )
        elif args.mode == "prepare":
            row = _run_prepare_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                max_order=args.max_order,
                runs=args.runs,
                warmup=args.warmup,
                dtype=dtype,
                seed=args.seed,
                cfg=cfg,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
            )
        elif args.mode == "peak_prepare":
            row = _run_peak_prepare_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                max_order=args.max_order,
                dtype=dtype,
                seed=args.seed,
                cfg=cfg,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
            )
        elif args.mode == "peak_evaluate":
            row = _run_peak_evaluate_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                max_order=args.max_order,
                dtype=dtype,
                seed=args.seed,
                cfg=cfg,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
            )
        elif args.mode == "tree":
            row = _run_tree_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                runs=args.runs,
                warmup=args.warmup,
                dtype=dtype,
                seed=args.seed,
                cfg=cfg,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
            )
        elif args.mode == "m2l":
            row = _run_m2l_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                max_order=args.max_order,
                runs=args.runs,
                warmup=args.warmup,
                dtype=dtype,
                seed=args.seed,
                cfg=cfg,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
            )
        elif args.mode == "l2l":
            row = _run_l2l_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                max_order=args.max_order,
                runs=args.runs,
                warmup=args.warmup,
                dtype=dtype,
                seed=args.seed,
                cfg=cfg,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
            )
        elif args.mode == "downward_trace":
            row = _run_downward_trace_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                max_order=args.max_order,
                dtype=dtype,
                seed=args.seed,
                cfg=cfg,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
            )
        else:
            row = _run_interactions_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                runs=args.runs,
                warmup=args.warmup,
                dtype=dtype,
                seed=args.seed,
                cfg=cfg,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
            )
    except Exception as exc:  # pragma: no cover - worker fallback path
        row = _make_row_error(
            mode=args.mode,
            num_particles=args.num_particles,
            message=f"{type(exc).__name__}: {exc}",
        )
    print(json.dumps(row))


if __name__ == "__main__":
    main()
