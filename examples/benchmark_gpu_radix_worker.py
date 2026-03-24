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
from typing import Any, Optional


def _configure_worker_environment() -> None:
    """Reduce worker-side CUDA allocator pressure before JAX initializes."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "sweep",
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
        use_dense_interactions=impl.use_dense_interactions,
        grouped_interactions=ctx["grouped_interactions"],
        grouped_chunk_size=ctx["runtime_m2l_chunk_size"],
        need_traversal_result=need_traversal_result,
        need_compact_far_pairs=need_compact_far_pairs,
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
    return fmm._impl._prepare_state_nearfield_artifacts(
        tree=tree_artifacts.tree,
        neighbor_list=dual_downward_artifacts.neighbor_list,
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
    runtime_cache_path = _runtime_autotune_cache_path(
        cfg=cfg,
        autotune_cache_path=autotune_cache_path,
    )
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
    if bool(cfg.get("worker_autotune_traversal", autotune_default)) and isinstance(
        traversal_candidates_raw, list
    ):
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
                t = _measure_prepare_once(
                    fmm_kwargs=trial_kwargs,
                    positions=positions,
                    masses=masses,
                    leaf_size=int(leaf_size),
                    max_order=int(max_order),
                    autotune_cache_path=autotune_cache_path,
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
    if bool(
        cfg.get("worker_autotune_nearfield_chunk", autotune_default)
    ) and isinstance(nf_candidates_raw, list):
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
                t = _measure_prepare_once(
                    fmm_kwargs=trial_kwargs,
                    positions=positions,
                    masses=masses,
                    leaf_size=int(leaf_size),
                    max_order=int(max_order),
                    autotune_cache_path=autotune_cache_path,
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
