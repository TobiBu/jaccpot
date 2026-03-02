"""Benchmark FMM runtime split (tree, traversal, M2L/downward, total)."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20000, help="Number of particles")
    parser.add_argument("--p", type=int, default=6, help="Multipole order")
    parser.add_argument(
        "--preset",
        choices=("fast", "accurate"),
        default="fast",
        help="Solver preset used for the benchmark",
    )
    parser.add_argument(
        "--basis",
        choices=("real", "complex", "solidfmm"),
        default="solidfmm",
        help="Expansion basis used for the benchmark",
    )
    parser.add_argument("--leaf-size", type=int, default=32, help="Leaf size")
    parser.add_argument("--theta", type=float, default=0.6, help="MAC opening angle")
    parser.add_argument("--device", choices=("cpu", "gpu", "tpu"), default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--adaptive-order",
        action="store_true",
        help="Enable adaptive-order M2L gear buckets",
    )
    parser.add_argument(
        "--p-gears",
        type=str,
        default="4,6,8,10",
        help="Comma-separated adaptive orders (used with --adaptive-order)",
    )
    parser.add_argument(
        "--mac-force-scale-mode",
        choices=("prev", "prepass"),
        default="prev",
        help="Force-scale strategy used when adaptive order is enabled",
    )
    parser.add_argument(
        "--adaptive-error-model",
        choices=("tail_proxy", "dehnen_degree"),
        default="tail_proxy",
        help="Adaptive error estimator used when adaptive order is enabled",
    )
    parser.add_argument(
        "--adaptive-eps",
        type=float,
        default=None,
        help="Optional direct adaptive tolerance overriding the theta-derived heuristic",
    )
    return parser.parse_args()


ARGS = _parse_args()
if ARGS.device:
    os.environ["JAX_PLATFORM_NAME"] = ARGS.device

try:
    import jax
    import jax.numpy as jnp

    from examples.benchmark_utils import time_callable
    from jaccpot import FastMultipoleMethod
    from jaccpot.runtime._adaptive_policy import (
        adaptive_pair_policy,
        adaptive_policy_tolerance,
        bucket_far_pairs_by_tag,
    )
    from jaccpot.runtime._interaction_cache import _build_dual_tree_artifacts
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing runtime dependency. Install jaccpot deps (notably yggdrax) "
        f"before running this benchmark. Original error: {exc}"
    ) from exc


@dataclass(frozen=True)
class StageArtifacts:
    tree_artifacts: Any
    runtime_overrides: Any
    positions_arr: jax.Array
    masses_arr: jax.Array


def _build_tree_and_upward(
    fmm: FastMultipoleMethod, positions, masses
) -> StageArtifacts:
    impl = fmm._impl
    positions_arr, masses_arr, _ = impl._prepare_state_input_arrays(positions, masses)
    runtime_overrides = impl._resolve_runtime_execution_overrides(
        num_particles=int(positions_arr.shape[0])
    )
    tree_artifacts = impl._prepare_state_tree_and_upward(
        positions_arr=positions_arr,
        masses_arr=masses_arr,
        bounds=None,
        leaf_size=int(ARGS.leaf_size),
        max_order=int(ARGS.p),
        refine_local_val=impl.refine_local,
        max_refine_levels_val=impl.max_refine_levels,
        aspect_threshold_val=impl.aspect_threshold,
        jit_tree_override=None,
        upward_center_mode=runtime_overrides.center_mode,
        allow_stateful_cache=False,
    )
    return StageArtifacts(
        tree_artifacts=tree_artifacts,
        runtime_overrides=runtime_overrides,
        positions_arr=positions_arr,
        masses_arr=masses_arr,
    )


def _build_traversal(fmm: FastMultipoleMethod, staged: StageArtifacts):
    impl = fmm._impl
    tree_artifacts = staged.tree_artifacts
    runtime_overrides = staged.runtime_overrides
    pair_policy = None
    policy_state = None
    if impl.adaptive_order:
        pair_policy = adaptive_pair_policy
        policy_state = impl._build_adaptive_policy_state(
            upward=tree_artifacts.upward,
            p_gears=impl.p_gears,
            force_scale_nodes=jnp.ones(
                (tree_artifacts.tree.parent.shape[0],),
                dtype=tree_artifacts.positions_sorted.dtype,
            ),
            eps=jnp.asarray(
                (
                    ARGS.adaptive_eps
                    if ARGS.adaptive_eps is not None
                    else adaptive_policy_tolerance(
                        theta=float(ARGS.theta),
                        p_gears=impl.p_gears,
                        dtype=tree_artifacts.positions_sorted.dtype,
                    )
                ),
                dtype=tree_artifacts.positions_sorted.dtype,
            ),
            theta=jnp.asarray(
                float(ARGS.theta),
                dtype=tree_artifacts.positions_sorted.dtype,
            ),
            error_model_code=jnp.asarray(
                1 if ARGS.adaptive_error_model == "dehnen_degree" else 0,
                dtype=jnp.int32,
            ),
        )
    dual_artifacts, _ = _build_dual_tree_artifacts(
        tree_artifacts.tree,
        tree_artifacts.upward.geometry,
        theta=float(ARGS.theta),
        mac_type="dehnen" if impl.mac_type == "dehnen_error" else impl.mac_type,
        dehnen_radius_scale=impl.dehnen_radius_scale,
        cache_key=None,
        cache_entry=None,
        max_pair_queue=impl.max_pair_queue,
        pair_process_block=impl.pair_process_block,
        traversal_config=runtime_overrides.traversal_config,
        retry_logger=None,
        use_dense_interactions=impl.use_dense_interactions,
        grouped_interactions=runtime_overrides.grouped_interactions,
        grouped_chunk_size=runtime_overrides.m2l_chunk_size,
        pair_policy=pair_policy,
        policy_state=policy_state,
    )
    return dual_artifacts


def _fresh_locals_template(fmm: FastMultipoleMethod, staged: StageArtifacts):
    impl = fmm._impl
    tree_artifacts = staged.tree_artifacts
    return impl._build_locals_template_for_prepare_state(
        tree=tree_artifacts.tree,
        upward=tree_artifacts.upward,
        max_order=int(ARGS.p),
        pos_sorted=tree_artifacts.positions_sorted,
    )


def _adaptive_far_pairs_by_gear(fmm: FastMultipoleMethod, dual_artifacts):
    if not fmm._impl.adaptive_order:
        return None
    traversal_result = dual_artifacts.traversal_result
    far_total = int(traversal_result.far_pair_count)
    return bucket_far_pairs_by_tag(
        jnp.asarray(traversal_result.interaction_sources[:far_total], dtype=jnp.int32),
        jnp.asarray(traversal_result.interaction_targets[:far_total], dtype=jnp.int32),
        jnp.asarray(traversal_result.interaction_tags[:far_total], dtype=jnp.int32),
        num_tags=len(fmm._impl.p_gears),
    )


def _bucket_adaptive_pairs(fmm: FastMultipoleMethod, dual_artifacts):
    return _adaptive_far_pairs_by_gear(fmm, dual_artifacts)


def _run_m2l_downward(fmm: FastMultipoleMethod, staged: StageArtifacts, dual_artifacts):
    impl = fmm._impl
    tree_artifacts = staged.tree_artifacts
    runtime_overrides = staged.runtime_overrides

    (
        interactions,
        _neighbor_list,
        _traversal_result,
        dense_buffers,
        grouped_buffers,
        grouped_segment_starts,
        grouped_segment_lengths,
        grouped_segment_class_ids,
        grouped_segment_sort_permutation,
        grouped_segment_group_ids,
        grouped_segment_unique_targets,
    ) = impl._unpack_dual_tree_artifacts(dual_artifacts)

    locals_template = _fresh_locals_template(fmm, staged)
    far_pairs_by_gear = _adaptive_far_pairs_by_gear(fmm, dual_artifacts)

    return impl._prepare_downward_with_artifacts(
        tree=tree_artifacts.tree,
        upward=tree_artifacts.upward,
        theta_val=float(ARGS.theta),
        locals_template=locals_template,
        interactions=interactions,
        runtime_m2l_chunk_size=runtime_overrides.m2l_chunk_size,
        runtime_l2l_chunk_size=runtime_overrides.l2l_chunk_size,
        runtime_traversal_config=runtime_overrides.traversal_config,
        record_retry=lambda _: None,
        dense_buffers=dense_buffers,
        grouped_interactions=runtime_overrides.grouped_interactions,
        grouped_buffers=grouped_buffers,
        grouped_segment_starts=grouped_segment_starts,
        grouped_segment_lengths=grouped_segment_lengths,
        grouped_segment_class_ids=grouped_segment_class_ids,
        grouped_segment_sort_permutation=grouped_segment_sort_permutation,
        grouped_segment_group_ids=grouped_segment_group_ids,
        grouped_segment_unique_targets=grouped_segment_unique_targets,
        farfield_mode=runtime_overrides.farfield_mode,
        far_pairs_by_gear=far_pairs_by_gear,
        adaptive_order=impl.adaptive_order,
        p_gears=impl.p_gears,
    )


def _count_interactions(dual_artifacts) -> dict[str, int]:
    interactions = dual_artifacts.interactions
    far_pairs = int(interactions.sources.shape[0])
    level_offsets = getattr(interactions, "level_offsets", None)
    levels = int(level_offsets.shape[0] - 1) if level_offsets is not None else 0
    near_pairs = int(dual_artifacts.neighbor_list.neighbors.shape[0])
    return {
        "far_pairs": far_pairs,
        "near_pairs": near_pairs,
        "levels": levels,
    }


def main() -> None:
    if ARGS.dtype == "float64" and not jax.config.jax_enable_x64:
        raise SystemExit("float64 requested, but JAX x64 is disabled")

    dtype = jnp.float64 if ARGS.dtype == "float64" else jnp.float32
    key = jax.random.PRNGKey(0)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (int(ARGS.n), 3),
        dtype=dtype,
        minval=jnp.asarray(-1.0, dtype=dtype),
        maxval=jnp.asarray(1.0, dtype=dtype),
    )
    masses = jnp.abs(
        jax.random.normal(key_mass, (int(ARGS.n),), dtype=dtype)
    ) + jnp.asarray(0.5, dtype=dtype)

    p_gears: tuple[int, ...] = tuple()
    if ARGS.adaptive_order:
        p_gears = tuple(int(v.strip()) for v in ARGS.p_gears.split(",") if v.strip())
        if len(p_gears) == 0:
            raise SystemExit("--adaptive-order requires non-empty --p-gears")

    basis = str(ARGS.basis)
    fmm = FastMultipoleMethod(
        preset=str(ARGS.preset),
        basis=basis,
        theta=float(ARGS.theta),
        softening=1.0e-3,
        adaptive_order=bool(ARGS.adaptive_order),
        p_gears=p_gears,
        mac_force_scale_mode=str(ARGS.mac_force_scale_mode),
    )

    tree_timing = time_callable(
        _build_tree_and_upward,
        fmm,
        positions,
        masses,
        warmup=int(ARGS.warmup),
        runs=int(ARGS.runs),
    )
    staged = tree_timing.result

    traversal_timing = time_callable(
        _build_traversal,
        fmm,
        staged,
        warmup=int(ARGS.warmup),
        runs=int(ARGS.runs),
    )
    dual_artifacts = traversal_timing.result

    bucket_timing = None
    if ARGS.adaptive_order:
        bucket_timing = time_callable(
            _bucket_adaptive_pairs,
            fmm,
            dual_artifacts,
            warmup=int(ARGS.warmup),
            runs=int(ARGS.runs),
        )

    m2l_timing = time_callable(
        _run_m2l_downward,
        fmm,
        staged,
        dual_artifacts,
        warmup=int(ARGS.warmup),
        runs=int(ARGS.runs),
    )

    total_timing = time_callable(
        fmm.compute_accelerations,
        positions,
        masses,
        leaf_size=int(ARGS.leaf_size),
        max_order=int(ARGS.p),
        warmup=int(ARGS.warmup),
        runs=int(ARGS.runs),
    )

    counts = _count_interactions(dual_artifacts)
    device_str = str(jax.devices()[0])

    print(
        f"device={device_str} dtype={ARGS.dtype} n={ARGS.n} p={ARGS.p} "
        f"preset={ARGS.preset} basis={ARGS.basis}"
    )
    timing_parts = [
        "timings_s",
        f"tree_build={tree_timing.mean:.6f}",
        f"traversal={traversal_timing.mean:.6f}",
    ]
    if bucket_timing is not None:
        timing_parts.append(f"adaptive_bucket={bucket_timing.mean:.6f}")
    timing_parts.extend(
        [
            f"m2l={m2l_timing.mean:.6f}",
            f"total={total_timing.mean:.6f}",
        ]
    )
    print(" ".join(timing_parts))
    print(
        "interaction_counts "
        f"far_pairs={counts['far_pairs']} "
        f"near_pairs={counts['near_pairs']} "
        f"interaction_levels={counts['levels']}"
    )
    gear_counts = tuple(getattr(fmm._impl, "_recent_far_pairs_by_gear_counts", ()))
    if gear_counts:
        print(
            "interaction_counts_by_gear "
            + " ".join(
                f"gear{idx}_pairs={int(val)}" for idx, val in enumerate(gear_counts)
            )
        )


if __name__ == "__main__":
    main()
