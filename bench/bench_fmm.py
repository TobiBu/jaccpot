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
    parser.add_argument("--leaf-size", type=int, default=32, help="Leaf size")
    parser.add_argument("--theta", type=float, default=0.6, help="MAC opening angle")
    parser.add_argument("--device", choices=("cpu", "gpu", "tpu"), default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


ARGS = _parse_args()
if ARGS.device:
    os.environ["JAX_PLATFORM_NAME"] = ARGS.device

try:
    import jax
    import jax.numpy as jnp

    from examples.benchmark_utils import time_callable
    from jaccpot import FastMultipoleMethod
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
    dual_artifacts, _ = _build_dual_tree_artifacts(
        tree_artifacts.tree,
        tree_artifacts.upward.geometry,
        theta=float(ARGS.theta),
        mac_type=impl.mac_type,
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
    )
    return dual_artifacts


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

    return impl._prepare_downward_with_artifacts(
        tree=tree_artifacts.tree,
        upward=tree_artifacts.upward,
        theta_val=float(ARGS.theta),
        locals_template=tree_artifacts.locals_template,
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

    fmm = FastMultipoleMethod(
        preset="fast",
        basis="solidfmm",
        theta=float(ARGS.theta),
        softening=1.0e-3,
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

    print(f"device={device_str} dtype={ARGS.dtype} n={ARGS.n} p={ARGS.p}")
    print(
        "timings_s "
        f"tree_build={tree_timing.mean:.6f} "
        f"traversal={traversal_timing.mean:.6f} "
        f"m2l={m2l_timing.mean:.6f} "
        f"total={total_timing.mean:.6f}"
    )
    print(
        "interaction_counts "
        f"far_pairs={counts['far_pairs']} "
        f"near_pairs={counts['near_pairs']} "
        f"interaction_levels={counts['levels']}"
    )


if __name__ == "__main__":
    main()
