"""Measure cold vs warm GPU memory peaks for prepare sub-stages.

This script is intended to answer a narrower question than the existing
single-N notebook: where does the first-call prepare spike happen?

It reports coarse wall-clock time and whole-device GPU memory peaks for:

- tree+upward
- dual+downward
- prepare_state
- evaluate_prepared_state

Each phase can be measured cold and warm so the first-call overhead can be
attributed more precisely than the notebook's `prepare_cold - prepare_warm`
summary.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jaccpot import FastMultipoleMethod, FMMPreset
from examples.benchmark_gpu_radix_worker import _build_runtime_config
from jaccpot.runtime._adaptive_policy import (
    adaptive_pair_policy,
    adaptive_policy_tolerance,
)
from jaccpot.runtime._fmm_impl import (
    _cap_minimum_memory_streamed_gpu_traversal_config_for_tree,
)
from jaccpot.runtime._fmm_impl import _build_nearfield_interop_data
from jaccpot.runtime._interaction_cache import _build_dual_tree_artifacts
from jaccpot.runtime._interaction_cache import _build_dual_tree_artifacts_split
from jaccpot.runtime._interaction_cache import _can_split_dual_tree_build
from jaccpot.runtime._interaction_cache import _dual_tree_build_raw
from jaccpot.runtime._interaction_cache import _dual_tree_unpack_build_output
from yggdrax.interactions import build_compact_far_pairs, build_leaf_neighbor_lists


@dataclass(frozen=True)
class MemorySnapshot:
    label: str
    gpu_used_mb: Optional[float]
    gpu_total_mb: Optional[float]
    wall_time_s: float


def _sample_problem(n: int, *, dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
    key = jax.random.PRNGKey(0)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=dtype,
    )
    masses = jax.random.uniform(
        key_mass,
        (n,),
        minval=0.5,
        maxval=1.5,
        dtype=dtype,
    )
    return positions, masses


def _block_until_ready(value: Any) -> Any:
    def _maybe_block(x: Any) -> Any:
        if hasattr(x, "block_until_ready"):
            return x.block_until_ready()
        return x

    return jax.tree_util.tree_map(_maybe_block, value)


def _tree_nbytes(value: Any) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "dtype") and hasattr(leaf, "shape"):
            arr = jnp.asarray(leaf)
            total += int(arr.size) * int(arr.dtype.itemsize)
    return total


def _gpu_memory_snapshot(label: str, *, gpu_index: int) -> MemorySnapshot:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            text=True,
        ).strip()
        used_mb_str, total_mb_str = [part.strip() for part in output.split(",", 1)]
        return MemorySnapshot(
            label=label,
            gpu_used_mb=float(used_mb_str),
            gpu_total_mb=float(total_mb_str),
            wall_time_s=time.perf_counter(),
        )
    except Exception:
        return MemorySnapshot(
            label=label,
            gpu_used_mb=None,
            gpu_total_mb=None,
            wall_time_s=time.perf_counter(),
        )


def _peak_gpu_memory_trace(
    fn,
    *args,
    label: str,
    gpu_index: int,
    poll_interval_s: float = 0.02,
    **kwargs,
):
    trace_rows: list[dict[str, Any]] = []
    stop_event = threading.Event()

    def _poll() -> None:
        while not stop_event.is_set():
            snap = _gpu_memory_snapshot(label, gpu_index=int(gpu_index))
            trace_rows.append(asdict(snap))
            time.sleep(float(poll_interval_s))

    before = _gpu_memory_snapshot(f"{label}_before", gpu_index=int(gpu_index))
    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    t0 = time.perf_counter()
    error = ""
    error_type = ""
    result = None
    try:
        result = fn(*args, **kwargs)
        result = _block_until_ready(result)
    except Exception as exc:  # pragma: no cover - diagnostic path
        error = str(exc)
        error_type = type(exc).__name__
    wall_seconds = time.perf_counter() - t0
    stop_event.set()
    thread.join(timeout=max(0.1, float(poll_interval_s) * 4.0))
    after = _gpu_memory_snapshot(f"{label}_after", gpu_index=int(gpu_index))

    used_values = [
        float(row["gpu_used_mb"])
        for row in trace_rows
        if row.get("gpu_used_mb") is not None
    ]
    before_used = before.gpu_used_mb
    after_used = after.gpu_used_mb
    peak_used = max(used_values) if used_values else None
    if peak_used is None:
        gpu_peak_delta_mb = None
    elif before_used is None:
        gpu_peak_delta_mb = None
    else:
        gpu_peak_delta_mb = float(peak_used - before_used)

    peak_df = {
        "component": str(label),
        "gpu_used_before_mb": before_used,
        "gpu_used_after_mb": after_used,
        "gpu_peak_used_mb": peak_used,
        "gpu_peak_delta_mb": gpu_peak_delta_mb,
        "wall_seconds": float(wall_seconds),
        "error": str(error),
        "error_type": str(error_type),
    }
    return result, peak_df, trace_rows, before, after


def _clear_runtime_memory(fmm: Optional[FastMultipoleMethod] = None) -> None:
    if fmm is not None:
        clear_fn = getattr(fmm, "clear_runtime_caches", None)
        if callable(clear_fn):
            clear_fn(clear_jax_compilation=False)
        elif hasattr(fmm, "clear_prepared_state_cache"):
            fmm.clear_prepared_state_cache()
    gc.collect()
    jax.clear_caches()


def _resolved_prepare_context(impl: Any, *, num_particles: int) -> dict[str, Any]:
    runtime_overrides = impl._resolve_runtime_execution_overrides(
        num_particles=int(num_particles)
    )
    refine_local_val = bool(impl.refine_local)
    if runtime_overrides.refine_local_override is not None:
        refine_local_val = bool(runtime_overrides.refine_local_override)
    return {
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


def _measure_prepare_stage_split(
    solver: FastMultipoleMethod,
    *,
    positions: Any,
    masses: Any,
    leaf_size: int,
    max_order: int,
    gpu_index: int,
    poll_interval_s: float,
    warmup: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    impl = solver._impl
    ctx = _resolved_prepare_context(impl, num_particles=int(positions.shape[0]))
    phase_rows: list[dict[str, Any]] = []

    def tree_upward_fn():
        return impl._prepare_state_tree_and_upward(
            positions_arr=positions,
            masses_arr=masses,
            bounds=None,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            refine_local_val=ctx["refine_local_val"],
            max_refine_levels_val=ctx["max_refine_levels_val"],
            aspect_threshold_val=ctx["aspect_threshold_val"],
            jit_tree_override=solver.advanced.runtime.jit_tree,
            upward_center_mode=ctx["upward_center_mode"],
            allow_stateful_cache=False,
        )

    tree_artifacts_cold, peak_row, _, _, _ = _peak_gpu_memory_trace(
        tree_upward_fn,
        label="tree_upward_cold",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(tree_artifacts_cold)) if tree_artifacts_cold is not None else None
    phase_rows.append(peak_row)

    for _ in range(max(0, int(warmup))):
        warm_tree = tree_upward_fn()
        _block_until_ready(warm_tree.upward.multipoles.packed)
        del warm_tree
        _clear_runtime_memory()

    tree_artifacts_warm, peak_row, _, _, _ = _peak_gpu_memory_trace(
        tree_upward_fn,
        label="tree_upward_warm",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(tree_artifacts_warm)) if tree_artifacts_warm is not None else None
    phase_rows.append(peak_row)

    def _raw_dual_tree_policy(tree_artifacts):
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
        return pair_policy, policy_state, use_paper_fixed_policy

    def _effective_runtime_traversal_config(tree_artifacts):
        traversal_config = ctx["runtime_traversal_config"]
        if traversal_config is None:
            return None
        total_nodes = int(tree_artifacts.tree.parent.shape[0])
        num_internal = int(jnp.asarray(tree_artifacts.tree.left_child).shape[0])
        num_leaves = max(1, total_nodes - num_internal)
        return _cap_minimum_memory_streamed_gpu_traversal_config_for_tree(
            traversal_config=traversal_config,
            total_nodes=total_nodes,
            num_leaves=num_leaves,
            num_particles=int(tree_artifacts.positions_sorted.shape[0]),
        )

    def dual_tree_build_raw_fn(tree_artifacts):
        pair_policy, policy_state, use_paper_fixed_policy = _raw_dual_tree_policy(
            tree_artifacts
        )
        need_traversal_result = bool(impl.retain_traversal_result) or bool(
            use_paper_fixed_policy
        )
        use_compact_streamed_pairs = (
            bool(impl.streamed_far_pairs)
            and not bool(impl.adaptive_order)
            and not bool(ctx["grouped_interactions"])
            and not bool(impl.mixed_order_farfield)
            and not bool(impl.retain_interactions)
            and not bool(need_traversal_result)
        )
        need_compact_far_pairs = (
            bool(impl.adaptive_order) and not bool(need_traversal_result)
        ) or bool(use_compact_streamed_pairs)
        need_node_interactions = not bool(use_compact_streamed_pairs)
        traversal_config = _effective_runtime_traversal_config(tree_artifacts)
        return _dual_tree_build_raw(
            tree=tree_artifacts.tree,
            geometry=tree_artifacts.upward.geometry,
            theta=ctx["theta_val"],
            mac_type=ctx["mac_type_val"],
            dehnen_radius_scale=ctx["dehnen_radius_scale"],
            max_pair_queue=impl.max_pair_queue,
            pair_process_block=impl.pair_process_block,
            traversal_config=traversal_config,
            retry_logger=lambda _event: None,
            fail_fast=impl.fail_fast,
            need_traversal_result=need_traversal_result,
            need_compact_far_pairs=need_compact_far_pairs,
            need_node_interactions=need_node_interactions,
            grouped_interactions=ctx["grouped_interactions"],
            pair_policy=pair_policy,
            policy_state=policy_state,
            jit_traversal=bool(impl._jit_traversal_default),
        )

    def dual_tree_split_build_fn(tree_artifacts):
        pair_policy, policy_state, use_paper_fixed_policy = _raw_dual_tree_policy(
            tree_artifacts
        )
        need_traversal_result = bool(impl.retain_traversal_result) or bool(
            use_paper_fixed_policy
        )
        use_compact_streamed_pairs = (
            bool(impl.streamed_far_pairs)
            and not bool(impl.adaptive_order)
            and not bool(ctx["grouped_interactions"])
            and not bool(impl.mixed_order_farfield)
            and not bool(impl.retain_interactions)
            and not bool(need_traversal_result)
        )
        need_compact_far_pairs = (
            bool(impl.adaptive_order) and not bool(need_traversal_result)
        ) or bool(use_compact_streamed_pairs)
        need_node_interactions = not bool(use_compact_streamed_pairs)
        use_dense_interactions_for_prepare = bool(impl.use_dense_interactions) and (
            impl.expansion_basis != "solidfmm"
        )
        if not _can_split_dual_tree_build(
            grouped_interactions=ctx["grouped_interactions"],
            need_traversal_result=need_traversal_result,
            pair_policy=pair_policy,
            policy_state=policy_state,
        ):
            raise RuntimeError("split dual-tree build is not eligible for this config")
        traversal_config = _effective_runtime_traversal_config(tree_artifacts)
        return _build_dual_tree_artifacts_split(
            tree=tree_artifacts.tree,
            geometry=tree_artifacts.upward.geometry,
            theta=ctx["theta_val"],
            mac_type=ctx["mac_type_val"],
            dehnen_radius_scale=ctx["dehnen_radius_scale"],
            max_pair_queue=impl.max_pair_queue,
            pair_process_block=impl.pair_process_block,
            traversal_config=traversal_config,
            retry_logger=lambda _event: None,
            need_node_interactions=need_node_interactions,
            need_compact_far_pairs=need_compact_far_pairs,
            use_dense_interactions=use_dense_interactions_for_prepare,
        )

    def dual_tree_split_far_only_fn(tree_artifacts):
        pair_policy, policy_state, use_paper_fixed_policy = _raw_dual_tree_policy(
            tree_artifacts
        )
        need_traversal_result = bool(impl.retain_traversal_result) or bool(
            use_paper_fixed_policy
        )
        use_compact_streamed_pairs = (
            bool(impl.streamed_far_pairs)
            and not bool(impl.adaptive_order)
            and not bool(ctx["grouped_interactions"])
            and not bool(impl.mixed_order_farfield)
            and not bool(impl.retain_interactions)
            and not bool(need_traversal_result)
        )
        need_compact_far_pairs = (
            bool(impl.adaptive_order) and not bool(need_traversal_result)
        ) or bool(use_compact_streamed_pairs)
        if not bool(need_compact_far_pairs):
            return None
        if not _can_split_dual_tree_build(
            grouped_interactions=ctx["grouped_interactions"],
            need_traversal_result=need_traversal_result,
            pair_policy=pair_policy,
            policy_state=policy_state,
        ):
            raise RuntimeError("split dual-tree far-only build is not eligible for this config")
        traversal_config = _effective_runtime_traversal_config(tree_artifacts)
        return build_compact_far_pairs(
            tree_artifacts.tree,
            tree_artifacts.upward.geometry,
            theta=ctx["theta_val"],
            mac_type=ctx["mac_type_val"],
            dehnen_radius_scale=ctx["dehnen_radius_scale"],
            max_pair_queue=impl.max_pair_queue,
            process_block=impl.pair_process_block,
            traversal_config=traversal_config,
            retry_logger=lambda _event: None,
        )

    def dual_tree_split_near_only_fn(tree_artifacts):
        pair_policy, policy_state, use_paper_fixed_policy = _raw_dual_tree_policy(
            tree_artifacts
        )
        need_traversal_result = bool(impl.retain_traversal_result) or bool(
            use_paper_fixed_policy
        )
        if not _can_split_dual_tree_build(
            grouped_interactions=ctx["grouped_interactions"],
            need_traversal_result=need_traversal_result,
            pair_policy=pair_policy,
            policy_state=policy_state,
        ):
            raise RuntimeError("split dual-tree near-only build is not eligible for this config")
        traversal_cfg = _effective_runtime_traversal_config(tree_artifacts)
        max_neighbors_per_leaf = (
            int(traversal_cfg.max_neighbors_per_leaf)
            if traversal_cfg is not None
            else 2048
        )
        return build_leaf_neighbor_lists(
            tree_artifacts.tree,
            tree_artifacts.upward.geometry,
            theta=ctx["theta_val"],
            max_neighbors_per_leaf=max_neighbors_per_leaf,
            mac_type=ctx["mac_type_val"],
            dehnen_radius_scale=ctx["dehnen_radius_scale"],
            max_pair_queue=impl.max_pair_queue,
            process_block=impl.pair_process_block,
            traversal_config=traversal_cfg,
            retry_logger=lambda _event: None,
        )

    def dual_tree_unpack_fn(build_raw_out):
        build_out, _current_traversal_config, _current_max_pair_queue, _current_pair_process_block = build_raw_out
        pair_policy, policy_state, use_paper_fixed_policy = (None, None, False)
        del pair_policy, policy_state
        need_traversal_result = bool(impl.retain_traversal_result) or bool(
            use_paper_fixed_policy
        )
        use_compact_streamed_pairs = (
            bool(impl.streamed_far_pairs)
            and not bool(impl.adaptive_order)
            and not bool(ctx["grouped_interactions"])
            and not bool(impl.mixed_order_farfield)
            and not bool(impl.retain_interactions)
            and not bool(need_traversal_result)
        )
        need_compact_far_pairs = (
            bool(impl.adaptive_order) and not bool(need_traversal_result)
        ) or bool(use_compact_streamed_pairs)
        return _dual_tree_unpack_build_output(
            build_out=build_out,
            grouped_interactions=ctx["grouped_interactions"],
            need_traversal_result=need_traversal_result,
            need_compact_far_pairs=need_compact_far_pairs,
        )

    build_raw_cold, peak_row, _, _, _ = _peak_gpu_memory_trace(
        dual_tree_build_raw_fn,
        tree_artifacts_cold,
        label="dual_tree_build_raw_cold",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(build_raw_cold)) if build_raw_cold is not None else None
    phase_rows.append(peak_row)

    for _ in range(max(0, int(warmup))):
        warm_build_raw = dual_tree_build_raw_fn(tree_artifacts_warm)
        _block_until_ready(warm_build_raw)
        del warm_build_raw
        _clear_runtime_memory()

    build_raw_warm, peak_row, _, _, _ = _peak_gpu_memory_trace(
        dual_tree_build_raw_fn,
        tree_artifacts_warm,
        label="dual_tree_build_raw_warm",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(build_raw_warm)) if build_raw_warm is not None else None
    phase_rows.append(peak_row)

    split_build_cold, peak_row, _, _, _ = _peak_gpu_memory_trace(
        dual_tree_split_build_fn,
        tree_artifacts_cold,
        label="dual_tree_split_build_cold",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(split_build_cold)) if split_build_cold is not None else None
    phase_rows.append(peak_row)

    for _ in range(max(0, int(warmup))):
        warm_split_build = dual_tree_split_build_fn(tree_artifacts_warm)
        _block_until_ready(warm_split_build)
        del warm_split_build
        _clear_runtime_memory()

    split_build_warm, peak_row, _, _, _ = _peak_gpu_memory_trace(
        dual_tree_split_build_fn,
        tree_artifacts_warm,
        label="dual_tree_split_build_warm",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(split_build_warm)) if split_build_warm is not None else None
    phase_rows.append(peak_row)

    split_far_only_cold, peak_row, _, _, _ = _peak_gpu_memory_trace(
        dual_tree_split_far_only_fn,
        tree_artifacts_cold,
        label="dual_tree_split_far_only_cold",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(split_far_only_cold)) if split_far_only_cold is not None else None
    phase_rows.append(peak_row)

    for _ in range(max(0, int(warmup))):
        warm_split_far_only = dual_tree_split_far_only_fn(tree_artifacts_warm)
        _block_until_ready(warm_split_far_only)
        del warm_split_far_only
        _clear_runtime_memory()

    split_far_only_warm, peak_row, _, _, _ = _peak_gpu_memory_trace(
        dual_tree_split_far_only_fn,
        tree_artifacts_warm,
        label="dual_tree_split_far_only_warm",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(split_far_only_warm)) if split_far_only_warm is not None else None
    phase_rows.append(peak_row)

    split_near_only_cold, peak_row, _, _, _ = _peak_gpu_memory_trace(
        dual_tree_split_near_only_fn,
        tree_artifacts_cold,
        label="dual_tree_split_near_only_cold",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(split_near_only_cold)) if split_near_only_cold is not None else None
    phase_rows.append(peak_row)

    for _ in range(max(0, int(warmup))):
        warm_split_near_only = dual_tree_split_near_only_fn(tree_artifacts_warm)
        _block_until_ready(warm_split_near_only)
        del warm_split_near_only
        _clear_runtime_memory()

    split_near_only_warm, peak_row, _, _, _ = _peak_gpu_memory_trace(
        dual_tree_split_near_only_fn,
        tree_artifacts_warm,
        label="dual_tree_split_near_only_warm",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(split_near_only_warm)) if split_near_only_warm is not None else None
    phase_rows.append(peak_row)

    unpack_cold, peak_row, _, _, _ = _peak_gpu_memory_trace(
        dual_tree_unpack_fn,
        build_raw_cold,
        label="dual_tree_unpack_cold",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(unpack_cold)) if unpack_cold is not None else None
    phase_rows.append(peak_row)

    for _ in range(max(0, int(warmup))):
        warm_unpack = dual_tree_unpack_fn(build_raw_warm)
        _block_until_ready(warm_unpack)
        del warm_unpack
        _clear_runtime_memory()

    unpack_warm, peak_row, _, _, _ = _peak_gpu_memory_trace(
        dual_tree_unpack_fn,
        build_raw_warm,
        label="dual_tree_unpack_warm",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(unpack_warm)) if unpack_warm is not None else None
    phase_rows.append(peak_row)

    def raw_dual_tree_fn(tree_artifacts):
        pair_policy, policy_state, use_paper_fixed_policy = _raw_dual_tree_policy(
            tree_artifacts
        )
        need_traversal_result = bool(impl.retain_traversal_result) or bool(
            use_paper_fixed_policy
        )
        use_compact_streamed_pairs = (
            bool(impl.streamed_far_pairs)
            and not bool(impl.adaptive_order)
            and not bool(ctx["grouped_interactions"])
            and not bool(impl.mixed_order_farfield)
            and not bool(impl.retain_interactions)
            and not bool(need_traversal_result)
        )
        need_compact_far_pairs = (
            bool(impl.adaptive_order) and not bool(need_traversal_result)
        ) or bool(use_compact_streamed_pairs)
        need_node_interactions = not bool(use_compact_streamed_pairs)
        use_dense_interactions_for_prepare = bool(impl.use_dense_interactions) and (
            impl.expansion_basis != "solidfmm"
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
            use_dense_interactions=use_dense_interactions_for_prepare,
            grouped_interactions=ctx["grouped_interactions"],
            grouped_chunk_size=ctx["runtime_m2l_chunk_size"],
            need_traversal_result=need_traversal_result,
            need_compact_far_pairs=need_compact_far_pairs,
            need_node_interactions=need_node_interactions,
            precompute_grouped_class_segments=impl._should_precompute_grouped_class_segments(
                grouped_chunk_size=ctx["runtime_m2l_chunk_size"],
                farfield_mode=ctx["farfield_mode"],
            ),
            grouped_schedule_budget_bytes=impl._grouped_schedule_item_budget(),
            pair_policy=pair_policy,
            policy_state=policy_state,
            jit_traversal=bool(impl._jit_traversal_default),
        )
        return dual_artifacts

    raw_dual_cold, peak_row, _, _, _ = _peak_gpu_memory_trace(
        raw_dual_tree_fn,
        tree_artifacts_cold,
        label="raw_dual_tree_cold",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(raw_dual_cold)) if raw_dual_cold is not None else None
    phase_rows.append(peak_row)

    for _ in range(max(0, int(warmup))):
        warm_raw_dual = raw_dual_tree_fn(tree_artifacts_warm)
        _block_until_ready(warm_raw_dual)
        del warm_raw_dual
        _clear_runtime_memory()

    raw_dual_warm, peak_row, _, _, _ = _peak_gpu_memory_trace(
        raw_dual_tree_fn,
        tree_artifacts_warm,
        label="raw_dual_tree_warm",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(raw_dual_warm)) if raw_dual_warm is not None else None
    phase_rows.append(peak_row)

    def downward_only_fn(tree_artifacts, raw_dual_artifacts):
        (
            interactions,
            _neighbor_list,
            traversal_result,
            compact_far_pairs,
            dense_buffers,
            grouped_buffers,
            grouped_segment_starts,
            grouped_segment_lengths,
            grouped_segment_class_ids,
            grouped_segment_sort_permutation,
            grouped_segment_group_ids,
            grouped_segment_unique_targets,
        ) = impl._unpack_dual_tree_artifacts(raw_dual_artifacts)

        far_pair_plan = impl._prepare_state_plan_far_pairs_for_downward(
            interactions=interactions,
            traversal_result=traversal_result,
            compact_far_pairs=compact_far_pairs,
            upward=tree_artifacts.upward,
        )
        runtime_m2l_chunk_size = impl._prepare_state_autotune_downward_chunk_size(
            upward=tree_artifacts.upward,
            far_pairs_by_gear=far_pair_plan.far_pairs_by_gear,
            p_gears_for_downward=far_pair_plan.p_gears_for_downward,
            runtime_m2l_chunk_size=ctx["runtime_m2l_chunk_size"],
        )
        interactions_for_downward = impl._prepare_state_select_interactions_for_downward(
            interactions=interactions,
            far_pairs_coo=far_pair_plan.far_pairs_coo,
        )
        return impl._prepare_downward_with_artifacts(
            tree=tree_artifacts.tree,
            upward=tree_artifacts.upward,
            theta_val=ctx["theta_val"],
            locals_template=tree_artifacts.locals_template,
            interactions=interactions_for_downward,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=ctx["runtime_l2l_chunk_size"],
            runtime_traversal_config=ctx["runtime_traversal_config"],
            record_retry=lambda _event: None,
            dense_buffers=dense_buffers,
            grouped_interactions=ctx["grouped_interactions"],
            grouped_buffers=grouped_buffers,
            grouped_segment_starts=grouped_segment_starts,
            grouped_segment_lengths=grouped_segment_lengths,
            grouped_segment_class_ids=grouped_segment_class_ids,
            grouped_segment_sort_permutation=grouped_segment_sort_permutation,
            grouped_segment_group_ids=grouped_segment_group_ids,
            grouped_segment_unique_targets=grouped_segment_unique_targets,
            farfield_mode=ctx["farfield_mode"],
            far_pairs_coo=far_pair_plan.far_pairs_coo,
            far_pairs_by_gear=far_pair_plan.far_pairs_by_gear,
            adaptive_order=far_pair_plan.adaptive_order_for_downward,
            p_gears=far_pair_plan.p_gears_for_downward,
        )

    downward_only_cold, peak_row, _, _, _ = _peak_gpu_memory_trace(
        downward_only_fn,
        tree_artifacts_cold,
        raw_dual_cold,
        label="downward_only_cold",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(downward_only_cold)) if downward_only_cold is not None else None
    phase_rows.append(peak_row)

    for _ in range(max(0, int(warmup))):
        warm_downward = downward_only_fn(tree_artifacts_warm, raw_dual_warm)
        _block_until_ready(warm_downward.locals.coefficients)
        del warm_downward
        _clear_runtime_memory()

    downward_only_warm, peak_row, _, _, _ = _peak_gpu_memory_trace(
        downward_only_fn,
        tree_artifacts_warm,
        raw_dual_warm,
        label="downward_only_warm",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(downward_only_warm)) if downward_only_warm is not None else None
    phase_rows.append(peak_row)

    def nearfield_prepare_fn(tree_artifacts, raw_dual_artifacts):
        (
            _interactions,
            neighbor_list,
            _traversal_result,
            _compact_far_pairs,
            _dense_buffers,
            _grouped_buffers,
            _grouped_segment_starts,
            _grouped_segment_lengths,
            _grouped_segment_class_ids,
            _grouped_segment_sort_permutation,
            _grouped_segment_group_ids,
            _grouped_segment_unique_targets,
        ) = impl._unpack_dual_tree_artifacts(raw_dual_artifacts)
        nearfield_interop = _build_nearfield_interop_data(
            tree_artifacts.tree,
            neighbor_list,
        )
        return impl._prepare_state_nearfield_artifacts(
            neighbor_list=neighbor_list,
            nearfield_interop=nearfield_interop,
            leaf_cap=int(tree_artifacts.leaf_cap),
            num_particles=int(positions.shape[0]),
            cache_entry=None,
            allow_stateful_cache=False,
        )

    nearfield_prepare_cold, peak_row, _, _, _ = _peak_gpu_memory_trace(
        nearfield_prepare_fn,
        tree_artifacts_cold,
        raw_dual_cold,
        label="nearfield_prepare_cold",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(nearfield_prepare_cold)) if nearfield_prepare_cold is not None else None
    phase_rows.append(peak_row)

    for _ in range(max(0, int(warmup))):
        warm_nearfield = nearfield_prepare_fn(tree_artifacts_warm, raw_dual_warm)
        _block_until_ready(warm_nearfield)
        del warm_nearfield
        _clear_runtime_memory()

    nearfield_prepare_warm, peak_row, _, _, _ = _peak_gpu_memory_trace(
        nearfield_prepare_fn,
        tree_artifacts_warm,
        raw_dual_warm,
        label="nearfield_prepare_warm",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(nearfield_prepare_warm)) if nearfield_prepare_warm is not None else None
    phase_rows.append(peak_row)

    def dual_downward_fn(tree_artifacts):
        return impl._prepare_state_dual_and_downward(
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

    dual_cold, peak_row, _, _, _ = _peak_gpu_memory_trace(
        dual_downward_fn,
        tree_artifacts_cold,
        label="dual_downward_cold",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(dual_cold)) if dual_cold is not None else None
    phase_rows.append(peak_row)

    for _ in range(max(0, int(warmup))):
        warm_dual = dual_downward_fn(tree_artifacts_warm)
        _block_until_ready(warm_dual.downward.locals.coefficients)
        del warm_dual
        _clear_runtime_memory()

    dual_warm, peak_row, _, _, _ = _peak_gpu_memory_trace(
        dual_downward_fn,
        tree_artifacts_warm,
        label="dual_downward_warm",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(dual_warm)) if dual_warm is not None else None
    phase_rows.append(peak_row)

    def prepare_fn():
        return solver.prepare_state(
            positions,
            masses,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
        )

    _clear_runtime_memory(solver)
    state_cold, peak_row, _, _, _ = _peak_gpu_memory_trace(
        prepare_fn,
        label="prepare_cold",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(state_cold)) if state_cold is not None else None
    phase_rows.append(peak_row)

    for _ in range(max(0, int(warmup))):
        warm_state = prepare_fn()
        _block_until_ready(warm_state)
        del warm_state
        _clear_runtime_memory(solver)

    state_warm, peak_row, _, _, _ = _peak_gpu_memory_trace(
        prepare_fn,
        label="prepare_warm",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    peak_row["retained_bytes"] = int(_tree_nbytes(state_warm)) if state_warm is not None else None
    phase_rows.append(peak_row)

    def evaluate_fn(state):
        return solver.evaluate_prepared_state(state)

    _block_until_ready(evaluate_fn(state_warm))
    _, peak_row, _, _, _ = _peak_gpu_memory_trace(
        evaluate_fn,
        state_warm,
        label="evaluate_warm",
        gpu_index=int(gpu_index),
        poll_interval_s=float(poll_interval_s),
    )
    phase_rows.append(peak_row)

    summary = {
        "dual_tree_build_raw_compile_overhead_mb": _peak_delta(phase_rows, "dual_tree_build_raw_cold", "dual_tree_build_raw_warm"),
        "dual_tree_split_build_compile_overhead_mb": _peak_delta(phase_rows, "dual_tree_split_build_cold", "dual_tree_split_build_warm"),
        "dual_tree_split_far_only_compile_overhead_mb": _peak_delta(phase_rows, "dual_tree_split_far_only_cold", "dual_tree_split_far_only_warm"),
        "dual_tree_split_near_only_compile_overhead_mb": _peak_delta(phase_rows, "dual_tree_split_near_only_cold", "dual_tree_split_near_only_warm"),
        "dual_tree_unpack_compile_overhead_mb": _peak_delta(phase_rows, "dual_tree_unpack_cold", "dual_tree_unpack_warm"),
        "raw_dual_compile_overhead_mb": _peak_delta(phase_rows, "raw_dual_tree_cold", "raw_dual_tree_warm"),
        "downward_compile_overhead_mb": _peak_delta(phase_rows, "downward_only_cold", "downward_only_warm"),
        "nearfield_prepare_compile_overhead_mb": _peak_delta(phase_rows, "nearfield_prepare_cold", "nearfield_prepare_warm"),
        "tree_compile_overhead_mb": _peak_delta(phase_rows, "tree_upward_cold", "tree_upward_warm"),
        "dual_compile_overhead_mb": _peak_delta(phase_rows, "dual_downward_cold", "dual_downward_warm"),
        "prepare_compile_overhead_mb": _peak_delta(phase_rows, "prepare_cold", "prepare_warm"),
    }

    del dual_cold
    del dual_warm
    del build_raw_cold
    del build_raw_warm
    del split_build_cold
    del split_build_warm
    del split_far_only_cold
    del split_far_only_warm
    del split_near_only_cold
    del split_near_only_warm
    del unpack_cold
    del unpack_warm
    del raw_dual_cold
    del raw_dual_warm
    del downward_only_cold
    del downward_only_warm
    del nearfield_prepare_cold
    del nearfield_prepare_warm
    del tree_artifacts_cold
    del tree_artifacts_warm
    del state_cold
    del state_warm
    _clear_runtime_memory(solver)

    return phase_rows, summary


def _peak_delta(rows: list[dict[str, Any]], cold_label: str, warm_label: str) -> Optional[float]:
    cold = None
    warm = None
    for row in rows:
        if row.get("component") == cold_label:
            cold = row.get("gpu_peak_delta_mb")
        if row.get("component") == warm_label:
            warm = row.get("gpu_peak_delta_mb")
    if cold is None or warm is None:
        return None
    return float(cold) - float(warm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-particles", type=int, default=131072)
    parser.add_argument("--leaf-size", type=int, default=128)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--preset", type=str, default="large_n_gpu")
    parser.add_argument("--basis", type=str, default="solidfmm")
    parser.add_argument(
        "--runtime-path",
        choices=("auto", "legacy", "large_n"),
        default="large_n",
    )
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--poll-interval-s", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--config-json", default=None)
    args = parser.parse_args()

    dtype = getattr(jnp, str(args.dtype))
    if args.config_json is not None:
        cfg = json.loads(str(args.config_json))
        solver_kwargs = _build_runtime_config(cfg)
        solver_kwargs["runtime_path"] = str(args.runtime_path).strip().lower()
        solver = FastMultipoleMethod(**solver_kwargs)
    else:
        solver = FastMultipoleMethod(
            preset=FMMPreset(str(args.preset).strip().lower()),
            basis=str(args.basis),
            runtime_path=str(args.runtime_path).strip().lower(),
        )
    positions, masses = _sample_problem(int(args.num_particles), dtype=dtype)
    positions, masses = _block_until_ready((positions, masses))

    phase_rows, summary = _measure_prepare_stage_split(
        solver,
        positions=positions,
        masses=masses,
        leaf_size=int(args.leaf_size),
        max_order=int(args.max_order),
        gpu_index=int(args.gpu_index),
        poll_interval_s=float(args.poll_interval_s),
        warmup=int(args.warmup),
    )

    output = {
        "backend": jax.default_backend(),
        "num_particles": int(args.num_particles),
        "leaf_size": int(args.leaf_size),
        "max_order": int(args.max_order),
        "preset": str(args.preset),
        "basis": str(args.basis),
        "runtime_path": str(args.runtime_path),
        "dtype": str(dtype),
        "phase_rows": phase_rows,
        "summary": summary,
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
