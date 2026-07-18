"""CLI profiler for jaccpot prepare_state residual timing.

This script mirrors the deep residual notebook profiling and adds explicit
timing buckets for octree-specific tail work that happens after the current
dual/downward helper returns.

Run with:
    JAX_ENABLE_X64=1 micromamba run -n odisseo python \
        examples/profile_prepare_residuals.py
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any, Iterator

import benchmark_utils
import jax
import jax.numpy as jnp
from yggdrax.interactions import DualTreeTraversalConfig

import jaccpot.runtime._fmm_impl as _rt_mod
import jaccpot.runtime._interaction_cache as _interaction_cache_mod
import jaccpot.upward.solidfmm_complex_tree_expansions as _up_solid_mod
from jaccpot import (
    FarFieldConfig,
    FastMultipoleMethod,
    FMMAdvancedConfig,
    FMMPreset,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)


def _block_tree(value: Any) -> Any:
    try:
        return jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            value,
        )
    except Exception:
        return value


@contextmanager
def _timed_method(
    obj: Any,
    attr_name: str,
    sink: dict[str, list[float]],
    *,
    label: str,
    sync_output: bool = True,
) -> Iterator[None]:
    """Monkeypatch one callable and record wall time in milliseconds."""

    original = getattr(obj, attr_name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        out = original(*args, **kwargs)
        if sync_output:
            _block_tree(out)
        sink[label].append((time.perf_counter() - start) * 1e3)
        return out

    setattr(obj, attr_name, wrapped)
    try:
        yield
    finally:
        setattr(obj, attr_name, original)


def _timed_if_present(
    obj: Any,
    attr_name: str,
    sink: dict[str, list[float]],
    *,
    label: str,
    sync_output: bool = True,
) -> Any:
    if not hasattr(obj, attr_name):
        return nullcontext()
    attr = getattr(obj, attr_name)
    if not callable(attr):
        return nullcontext()
    return _timed_method(
        obj,
        attr_name,
        sink,
        label=label,
        sync_output=sync_output,
    )


def _build_fmm_kwargs(
    *,
    tree_type: str,
    execution_backend: str,
    leaf_size: int,
    theta: float,
    working_dtype: jnp.dtype,
    enable_interaction_cache: bool,
    retain_interactions: bool,
    retain_traversal_result: bool,
    memory_objective: str,
    max_pair_queue: int,
    process_block: int,
    max_interactions_per_node: int,
    max_neighbors_per_leaf: int,
) -> dict[str, Any]:
    traversal_config = DualTreeTraversalConfig(
        max_pair_queue=int(max_pair_queue),
        process_block=int(process_block),
        max_interactions_per_node=int(max_interactions_per_node),
        max_neighbors_per_leaf=int(max_neighbors_per_leaf),
    )
    advanced = FMMAdvancedConfig(
        tree=TreeConfig(
            tree_type=tree_type,
            mode="lbvh",
            leaf_target=int(leaf_size),
            refine_local=False,
            max_refine_levels=0,
            aspect_threshold=16.0,
        ),
        farfield=FarFieldConfig(
            rotation="solidfmm",
            mode="pair_grouped",
            grouped_interactions=False,
            streamed_far_pairs=True,
            m2l_chunk_size=256,
        ),
        nearfield=NearFieldConfig(
            mode="bucketed",
            edge_chunk_size=128,
            precompute_scatter_schedules=False,
        ),
        runtime=RuntimePolicyConfig(
            execution_backend=execution_backend,
            host_refine_mode="off",
            fail_fast=True,
            jit_tree=True,
            jit_traversal=True,
            memory_objective=memory_objective,
            max_pair_queue=int(max_pair_queue),
            pair_process_block=None,
            traversal_config=traversal_config,
            enable_interaction_cache=enable_interaction_cache,
            retain_traversal_result=retain_traversal_result,
            retain_interactions=retain_interactions,
            autotune_m2l_chunk=True,
            upward_leaf_batch_size=2048,
        ),
        mac_type="dehnen",
    )
    return dict(
        preset=FMMPreset.LARGE_N_GPU,
        basis="solidfmm",
        theta=float(theta),
        softening=1e-3,
        working_dtype=working_dtype,
        advanced=advanced,
    )


def profile_prepare_residuals(
    num_particles: int,
    *,
    fmm_kwargs: dict[str, Any],
    leaf_size: int,
    max_order: int,
    dtype: jnp.dtype,
    key: jax.Array,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    fmm = FastMultipoleMethod(**fmm_kwargs)
    positions, masses, _ = benchmark_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )

    stats: dict[str, list[float]] = defaultdict(list)
    specs = [
        (_rt_mod, "_build_tree_with_config", "tree_build_lowlevel"),
        (_rt_mod, "prepare_solidfmm_complex_upward_sweep", "upward_total"),
        (_up_solid_mod, "_p2m_leaves_complex", "P2M"),
        (_up_solid_mod, "_aggregate_m2m_complex_by_level", "M2M"),
        (_rt_mod, "_build_dual_tree_artifacts", "dual_tree_artifacts"),
        (_interaction_cache_mod, "_dual_tree_cache_lookup", "dual_cache_lookup"),
        (_interaction_cache_mod, "_dual_tree_build_raw", "dual_raw_build"),
        (
            _interaction_cache_mod,
            "_dual_tree_unpack_build_output",
            "dual_unpack_build_output",
        ),
        (
            _interaction_cache_mod,
            "_dual_tree_build_grouped_buffers",
            "dual_grouped_buffers",
        ),
        (
            _interaction_cache_mod,
            "_dual_tree_build_grouped_class_segments",
            "dual_grouped_segments",
        ),
        (
            _interaction_cache_mod,
            "_dual_tree_build_dense_buffers",
            "dual_dense_buffers",
        ),
        (
            fmm._impl,
            "_prepare_state_extract_adaptive_far_pairs",
            "downward_adaptive_payload",
        ),
        (
            fmm._impl,
            "_prepare_state_build_streamed_far_pair_plan",
            "downward_streamed_pair_plan",
        ),
        (
            fmm._impl,
            "_prepare_state_plan_far_pairs_for_downward",
            "downward_far_pair_plan",
        ),
        (
            fmm._impl,
            "_prepare_state_autotune_downward_chunk_size",
            "downward_m2l_autotune",
        ),
        (
            fmm._impl,
            "_prepare_state_select_interactions_for_downward",
            "downward_interaction_handoff",
        ),
        (_rt_mod, "_prepare_solidfmm_downward_sweep", "downward_total"),
        (
            _rt_mod,
            "_prepare_solidfmm_downward_interaction_inputs",
            "downward_input_arrays",
        ),
        (
            _rt_mod,
            "_prepare_solidfmm_downward_init",
            "downward_local_init",
        ),
        (
            _rt_mod,
            "_prepare_solidfmm_downward_multipole_inputs",
            "downward_multipole_staging",
        ),
        (
            _rt_mod,
            "_prepare_solidfmm_downward_child_inputs",
            "downward_child_prep",
        ),
        (
            _rt_mod,
            "_solidfmm_downward_accumulate_from_multipoles",
            "downward_accumulate_total",
        ),
        (_rt_mod, "enforce_conjugate_symmetry_batch", "downward_symmetry"),
        (_rt_mod, "_accumulate_m2l_fullbatch", "M2L"),
        (_rt_mod, "_accumulate_m2l_chunked_scan", "M2L"),
        (_rt_mod, "_accumulate_solidfmm_m2l_grouped", "M2L"),
        (_rt_mod, "_accumulate_solidfmm_m2l_grouped_class_major", "M2L"),
        (_rt_mod, "_accumulate_real_m2l_chunked_scan_pallas", "M2L"),
        (_rt_mod, "_propagate_solidfmm_locals_to_children", "L2L"),
        (_rt_mod, "_propagate_real_locals_to_children", "L2L"),
        (_rt_mod, "_build_nearfield_interop_data", "nearfield_interop"),
        (_rt_mod, "build_octree_native_neighbor_lists", "octree_native_neighbors"),
        (_rt_mod, "build_octree_native_far_pairs", "octree_native_far_pairs"),
        (_rt_mod, "_build_octree_downward_artifacts", "octree_downward_plan"),
        (_rt_mod, "_finalize_octree_downward_artifacts", "octree_downward_finalize"),
        (fmm._impl, "_prepare_state_nearfield_artifacts", "nearfield_artifacts"),
        (fmm._impl, "_prepare_leaf_neighbor_pairs_safe", "nearfield_pairs"),
        (
            fmm._impl,
            "_prepare_bucketed_scatter_schedules_safe",
            "nearfield_scatter",
        ),
    ]

    total_ms: list[float] = []
    with ExitStack() as stack:
        for obj, attr_name, label in specs:
            stack.enter_context(
                _timed_if_present(
                    obj,
                    attr_name,
                    stats,
                    label=label,
                    sync_output=True,
                )
            )

        for _ in range(int(warmup) + int(runs)):
            start = time.perf_counter()
            state = fmm.prepare_state(
                positions,
                masses,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
            )
            _block_tree(state)
            total_ms.append((time.perf_counter() - start) * 1e3)

    keep_slice = slice(int(warmup), None)
    total_keep = total_ms[keep_slice] if len(total_ms) > int(warmup) else total_ms

    row: dict[str, float] = {
        "num_particles": float(num_particles),
        "prepare_total_ms": float(jnp.mean(jnp.asarray(total_keep))),
    }
    for key_name, values in stats.items():
        kept = values[keep_slice] if len(values) > int(warmup) else values
        row[f"{key_name}_ms"] = float(jnp.mean(jnp.asarray(kept))) if len(kept) else 0.0

    p2m_ms = row.get("P2M_ms", 0.0)
    m2m_ms = row.get("M2M_ms", 0.0)
    m2l_ms = row.get("M2L_ms", 0.0)
    l2l_ms = row.get("L2L_ms", 0.0)
    upward_total = row.get("upward_total_ms", 0.0)
    downward_total = row.get("downward_total_ms", 0.0)
    nearfield_total = (
        row.get("nearfield_interop_ms", 0.0)
        + row.get("nearfield_artifacts_ms", 0.0)
        + row.get("nearfield_pairs_ms", 0.0)
        + row.get("nearfield_scatter_ms", 0.0)
    )
    octree_tail_total = (
        row.get("octree_native_neighbors_ms", 0.0)
        + row.get("octree_native_far_pairs_ms", 0.0)
        + row.get("octree_downward_plan_ms", 0.0)
        + row.get("octree_downward_finalize_ms", 0.0)
    )

    row["operator_sum_ms"] = p2m_ms + m2m_ms + m2l_ms + l2l_ms
    row["residual_target_ms"] = max(
        row["prepare_total_ms"] - row["operator_sum_ms"], 0.0
    )
    row["upward_non_operator_est_ms"] = max(upward_total - p2m_ms - m2m_ms, 0.0)
    row["downward_non_operator_est_ms"] = max(downward_total - m2l_ms - l2l_ms, 0.0)
    row["dual_artifacts_profiled_ms"] = (
        row.get("dual_cache_lookup_ms", 0.0)
        + row.get("dual_raw_build_ms", 0.0)
        + row.get("dual_unpack_build_output_ms", 0.0)
        + row.get("dual_grouped_buffers_ms", 0.0)
        + row.get("dual_grouped_segments_ms", 0.0)
        + row.get("dual_dense_buffers_ms", 0.0)
    )
    row["dual_artifacts_residual_ms"] = max(
        row.get("dual_tree_artifacts_ms", 0.0) - row["dual_artifacts_profiled_ms"],
        0.0,
    )
    row["downward_accumulate_residual_ms"] = max(
        row.get("downward_accumulate_total_ms", 0.0)
        - row.get("M2L_ms", 0.0)
        - row.get("downward_symmetry_ms", 0.0),
        0.0,
    )
    row["downward_accumulate_non_operator_ms"] = row.get(
        "downward_symmetry_ms", 0.0
    ) + row.get("downward_accumulate_residual_ms", 0.0)
    row["downward_support_profiled_ms"] = (
        row.get("downward_adaptive_payload_ms", 0.0)
        + row.get("downward_streamed_pair_plan_ms", 0.0)
        + row.get("downward_far_pair_plan_ms", 0.0)
        + row.get("downward_m2l_autotune_ms", 0.0)
        + row.get("downward_interaction_handoff_ms", 0.0)
        + row.get("downward_input_arrays_ms", 0.0)
        + row.get("downward_local_init_ms", 0.0)
        + row.get("downward_multipole_staging_ms", 0.0)
        + row.get("downward_child_prep_ms", 0.0)
        + row.get("downward_accumulate_non_operator_ms", 0.0)
    )
    row["downward_support_residual_ms"] = max(
        row["downward_non_operator_est_ms"] - row["downward_support_profiled_ms"],
        0.0,
    )
    row["nearfield_total_ms"] = nearfield_total
    row["octree_tail_total_ms"] = octree_tail_total

    explained = (
        row.get("tree_build_lowlevel_ms", 0.0)
        + row["upward_non_operator_est_ms"]
        + row.get("dual_tree_artifacts_ms", 0.0)
        + row["downward_non_operator_est_ms"]
        + nearfield_total
        + octree_tail_total
    )
    row["residual_unexplained_ms"] = max(row["residual_target_ms"] - explained, 0.0)
    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-particles",
        type=int,
        nargs="+",
        default=[65_536, 131_072, 262_144],
    )
    parser.add_argument("--leaf-size", type=int, default=128)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--theta", type=float, default=0.6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--tree-type",
        type=str,
        default="radix",
        choices=["radix", "octree", "kdtree"],
    )
    parser.add_argument(
        "--execution-backend",
        type=str,
        default="radix",
        choices=["radix", "octree"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
    )
    parser.add_argument("--max-pair-queue", type=int, default=262_144)
    parser.add_argument("--process-block", type=int, default=256)
    parser.add_argument("--max-interactions-per-node", type=int, default=8192)
    parser.add_argument("--max-neighbors-per-leaf", type=int, default=4096)
    parser.add_argument(
        "--memory-objective",
        type=str,
        default="minimum_memory",
        choices=["minimum_memory", "balanced"],
    )
    parser.add_argument("--enable-interaction-cache", action="store_true")
    parser.add_argument("--retain-interactions", action="store_true")
    parser.add_argument("--retain-traversal-result", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    fmm_kwargs = _build_fmm_kwargs(
        tree_type=args.tree_type,
        execution_backend=args.execution_backend,
        leaf_size=int(args.leaf_size),
        theta=float(args.theta),
        working_dtype=dtype,
        enable_interaction_cache=bool(args.enable_interaction_cache),
        retain_interactions=bool(args.retain_interactions),
        retain_traversal_result=bool(args.retain_traversal_result),
        memory_objective=args.memory_objective,
        max_pair_queue=int(args.max_pair_queue),
        process_block=int(args.process_block),
        max_interactions_per_node=int(args.max_interactions_per_node),
        max_neighbors_per_leaf=int(args.max_neighbors_per_leaf),
    )

    rows = []
    for idx, num_particles in enumerate(args.num_particles):
        row = profile_prepare_residuals(
            int(num_particles),
            fmm_kwargs=fmm_kwargs,
            leaf_size=int(args.leaf_size),
            max_order=int(args.max_order),
            dtype=dtype,
            key=jax.random.PRNGKey(1100 + idx),
            warmup=int(args.warmup),
            runs=int(args.runs),
        )
        rows.append(row)

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    columns = [
        "num_particles",
        "prepare_total_ms",
        "operator_sum_ms",
        "residual_target_ms",
        "tree_build_lowlevel_ms",
        "upward_non_operator_est_ms",
        "dual_tree_artifacts_ms",
        "downward_non_operator_est_ms",
        "downward_support_profiled_ms",
        "downward_support_residual_ms",
        "nearfield_total_ms",
        "octree_tail_total_ms",
        "residual_unexplained_ms",
    ]
    print(" ".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, 0.0)
            if column == "num_particles":
                values.append(str(int(round(value))))
            else:
                values.append(f"{value:.3f}")
        print(" ".join(values))


if __name__ == "__main__":
    main()
