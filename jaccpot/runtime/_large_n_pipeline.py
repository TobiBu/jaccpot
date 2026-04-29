"""Dedicated prepare/evaluate pipeline for large-N GPU radix solidfmm runs."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Tuple
from jaxtyping import Array
from yggdrax.interactions import DualTreeRetryEvent, DualTreeTraversalConfig, MACType

from ._large_n_nearfield import (
    build_large_n_leaf_particle_groups,
    build_large_n_nearfield_precompute,
    build_large_n_target_owned_blocks,
    build_large_n_target_owned_blocks_static,
    evaluate_large_n_nearfield_fast_lane,
    resolve_large_n_execution_config,
)
from ._large_n_types import LargeNPreparedState, RadixFastNearfieldPayload
from .dtypes import INDEX_DTYPE


def prepare_large_n_state(
    fmm: object,
    *,
    positions_arr: Array,
    masses_arr: Array,
    input_dtype: jnp.dtype,
    bounds: Optional[Tuple[Array, Array]],
    leaf_size: int,
    max_order: int,
    theta_val: float,
    mac_type_val: MACType,
    refine_local_val: bool,
    max_refine_levels_val: int,
    aspect_threshold_val: float,
    jit_tree_override: Optional[bool],
    allow_stateful_cache: bool,
    runtime_traversal_config: Optional[DualTreeTraversalConfig],
    runtime_m2l_chunk_size: Optional[int],
    runtime_l2l_chunk_size: Optional[int],
    upward_center_mode: str,
    record_retry: Callable[[DualTreeRetryEvent], None],
    collected_retries: list[DualTreeRetryEvent],
) -> LargeNPreparedState:
    """Prepare the slim large-N state using the dedicated narrow path."""

    refresh_timing_active = bool(getattr(fmm, "_refresh_timing_active", False))

    stage_t0 = time.perf_counter()
    tree_artifacts = fmm._prepare_state_tree_and_upward(
        positions_arr=positions_arr,
        masses_arr=masses_arr,
        bounds=bounds,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        refine_local_val=refine_local_val,
        max_refine_levels_val=max_refine_levels_val,
        aspect_threshold_val=aspect_threshold_val,
        jit_tree_override=jit_tree_override,
        upward_center_mode=upward_center_mode,
        allow_stateful_cache=allow_stateful_cache,
    )
    if bool(getattr(fmm, "_refresh_timing_active", False)):
        setattr(
            fmm,
            "_refresh_timing_tree_upward_seconds",
            float(getattr(fmm, "_refresh_timing_tree_upward_seconds", 0.0))
            + float(time.perf_counter() - stage_t0),
        )

    stage_t0 = time.perf_counter()
    dual_downward_artifacts = fmm._prepare_state_dual_and_downward(
        tree_artifacts=tree_artifacts,
        force_scale_nodes=None,
        upward_center_mode=upward_center_mode,
        theta_val=theta_val,
        mac_type_val=mac_type_val,
        dehnen_radius_scale=fmm.dehnen_radius_scale,
        runtime_traversal_config=runtime_traversal_config,
        runtime_m2l_chunk_size=runtime_m2l_chunk_size,
        runtime_l2l_chunk_size=runtime_l2l_chunk_size,
        grouped_interactions=False,
        farfield_mode="pair_grouped",
        record_retry=record_retry,
        refine_local_val=refine_local_val,
        max_refine_levels_val=max_refine_levels_val,
        aspect_threshold_val=aspect_threshold_val,
        allow_stateful_cache=allow_stateful_cache,
    )
    if bool(getattr(fmm, "_refresh_timing_active", False)):
        setattr(
            fmm,
            "_refresh_timing_dual_downward_seconds",
            float(getattr(fmm, "_refresh_timing_dual_downward_seconds", 0.0))
            + float(time.perf_counter() - stage_t0),
        )

    stage_t0 = time.perf_counter()
    if allow_stateful_cache:
        fmm._update_locals_template_cache_after_prepare(
            locals_template=tree_artifacts.locals_template,
            upward=tree_artifacts.upward,
            max_order=int(max_order),
        )

    retry_events_tuple = tuple(collected_retries)
    if allow_stateful_cache:
        fmm._recent_retry_events = retry_events_tuple

    execution_config = resolve_large_n_execution_config(
        fmm,
        num_particles=int(positions_arr.shape[0]),
    )

    def _env_bool(name: str, default: bool) -> bool:
        raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _env_pos_int(name: str, default: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
        except Exception:
            value = int(default)
        return max(1, int(value))

    def _canonical_static_int(
        value_env: str,
        default_value: int,
        options_env: str,
        default_options: str,
    ) -> int:
        try:
            raw_value = int(os.environ.get(value_env, str(default_value)))
        except Exception:
            raw_value = int(default_value)
        options_raw = str(os.environ.get(options_env, default_options)).strip()
        options: list[int] = []
        for token in options_raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                val = int(token)
            except Exception:
                continue
            if val > 0 and val not in options:
                options.append(val)
        if not options:
            options = [int(default_value)]
        if raw_value in options:
            return int(raw_value)
        return int(min(options, key=lambda v: (abs(v - raw_value), v)))

    nearfield_delayed_scatter_chunks_per_superchunk = _env_pos_int(
        "JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS",
        1,
    )
    nearfield_chunk_scan_batch_size = _env_pos_int(
        "JACCPOT_LARGE_N_CHUNK_SCAN_BATCH_SIZE",
        1,
    )
    nearfield_chunk_scan_unroll = _env_pos_int(
        "JACCPOT_LARGE_N_CHUNK_SCAN_UNROLL",
        1,
    )
    nearfield_superchunk_scan_unroll = _env_pos_int(
        "JACCPOT_LARGE_N_SUPERCHUNK_SCAN_UNROLL",
        1,
    )
    nearfield_sorted_scatter_hint = _env_bool(
        "JACCPOT_LARGE_N_SORTED_SCATTER_HINT",
        False,
    )
    nearfield_grouped_sorted_scatter = _env_bool(
        "JACCPOT_LARGE_N_GROUPED_SORTED_SCATTER",
        False,
    )
    nearfield_superchunk_target_reduce = _env_bool(
        "JACCPOT_LARGE_N_SUPERCHUNK_TARGET_REDUCE",
        False,
    )
    nearfield_disable_chunk_cond = _env_bool(
        "JACCPOT_LARGE_N_DISABLE_CHUNK_COND",
        True,
    )
    nearfield_target_leaf_batch_size = _canonical_static_int(
        "JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE",
        32,
        "JACCPOT_LARGE_N_TARGET_LEAF_BATCH_OPTIONS",
        "16,32,64",
    )
    nearfield_target_block_tile_size = _canonical_static_int(
        "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SIZE",
        8,
        "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_OPTIONS",
        "4,8,16",
    )
    nearfield_target_block_tile_scan_unroll = _canonical_static_int(
        "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SCAN_UNROLL",
        1,
        "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SCAN_UNROLL_OPTIONS",
        "1,2,4",
    )
    nearfield_target_block_batch_scan_unroll = _canonical_static_int(
        "JACCPOT_LARGE_N_TARGET_BLOCK_BATCH_SCAN_UNROLL",
        1,
        "JACCPOT_LARGE_N_TARGET_BLOCK_BATCH_SCAN_UNROLL_OPTIONS",
        "1,2,4",
    )
    nearfield_target_block_overflow_fast_max_blocks = _canonical_static_int(
        "JACCPOT_LARGE_N_TARGET_BLOCK_OVERFLOW_FAST_MAX_BLOCKS",
        65536,
        "JACCPOT_LARGE_N_TARGET_BLOCK_OVERFLOW_FAST_MAX_BLOCKS_OPTIONS",
        "16384,32768,65536,131072",
    )
    static_target_blocks_enabled = _env_bool(
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
        True,
    )
    static_target_blocks_max_per_leaf = _canonical_static_int(
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
        32,
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF_OPTIONS",
        "8,16,32,64,128",
    )
    overflow_profile_headroom_raw = os.environ.get(
        "JACCPOT_LARGE_N_OVERFLOW_PROFILE_HEADROOM",
        "2.0",
    )
    try:
        overflow_profile_headroom = max(1.0, float(overflow_profile_headroom_raw))
    except Exception:
        overflow_profile_headroom = 2.0
    overflow_profile_caps_raw = os.environ.get(
        "JACCPOT_LARGE_N_OVERFLOW_PROFILE_CAP_OPTIONS",
        "64,128,256,512,1024,2048,4096,8192,16384,32768,65536",
    )
    overflow_profile_caps: list[int] = []
    for token in str(overflow_profile_caps_raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0 and value not in overflow_profile_caps:
            overflow_profile_caps.append(value)
    overflow_profile_caps = sorted(overflow_profile_caps)
    if not overflow_profile_caps:
        overflow_profile_caps = [64, 128, 256, 512, 1024]
    neighbor_profile_headroom_raw = os.environ.get(
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_HEADROOM",
        "1.0",
    )
    try:
        neighbor_profile_headroom = max(1.0, float(neighbor_profile_headroom_raw))
    except Exception:
        neighbor_profile_headroom = 1.0
    neighbor_profile_caps_raw = os.environ.get(
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_CAP_OPTIONS",
        "4096,8192,12288,16384,20480,24576,28672,32768,49152,65536,98304,131072",
    )
    neighbor_profile_caps: list[int] = []
    for token in str(neighbor_profile_caps_raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0 and value not in neighbor_profile_caps:
            neighbor_profile_caps.append(value)
    neighbor_profile_caps = sorted(neighbor_profile_caps)
    if not neighbor_profile_caps:
        neighbor_profile_caps = [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]
    neighbor_profile_bootstrap_cap_raw = os.environ.get(
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_BOOTSTRAP_CAP",
        "0",
    )
    try:
        neighbor_profile_bootstrap_cap = max(
            0,
            int(neighbor_profile_bootstrap_cap_raw),
        )
    except Exception:
        neighbor_profile_bootstrap_cap = 0
    overflow_profile_bootstrap_cap_raw = os.environ.get(
        "JACCPOT_LARGE_N_OVERFLOW_PROFILE_BOOTSTRAP_CAP",
        "0",
    )
    try:
        overflow_profile_bootstrap_cap = max(
            0,
            int(overflow_profile_bootstrap_cap_raw),
        )
    except Exception:
        overflow_profile_bootstrap_cap = 0

    def _pick_overflow_profile_capacity(required: int) -> int:
        required = max(0, int(required))
        for cap in overflow_profile_caps:
            if int(cap) >= required:
                return int(cap)
        return int(required)

    def _pick_neighbor_profile_capacity(required: int) -> int:
        required = max(0, int(required))
        for cap in neighbor_profile_caps:
            if int(cap) >= required:
                return int(cap)
        return int(required)
    nearfield_total_t0 = stage_t0
    nearfield_stage_sum = 0.0

    def _record_nf(attr: str, start: float) -> None:
        nonlocal nearfield_stage_sum
        elapsed = float(time.perf_counter() - start)
        nearfield_stage_sum += elapsed
        if refresh_timing_active:
            setattr(fmm, attr, float(getattr(fmm, attr, 0.0)) + elapsed)

    disable_specialized_large_n_nearfield = str(
        os.environ.get("JACCPOT_DISABLE_LARGE_N_SPECIALIZED_NEARFIELD", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    substage_t0 = time.perf_counter()
    if bool(execution_config.retain_leaf_groups):
        leaf_particle_indices, leaf_particle_mask = build_large_n_leaf_particle_groups(
            tree_artifacts.tree,
            dual_downward_artifacts.neighbor_list,
            max_leaf_size=int(tree_artifacts.leaf_cap),
        )
    else:
        leaf_particle_indices = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        leaf_particle_mask = jnp.zeros((0, 0), dtype=bool)
    _record_nf("_refresh_timing_nearfield_leaf_groups_seconds", substage_t0)

    substage_t0 = time.perf_counter()
    nearfield_artifacts = build_large_n_nearfield_precompute(
        tree=tree_artifacts.tree,
        neighbor_list=dual_downward_artifacts.neighbor_list,
        leaf_particle_indices=leaf_particle_indices,
        leaf_particle_mask=leaf_particle_mask,
        execution_config=execution_config,
    )
    _record_nf("_refresh_timing_nearfield_precompute_seconds", substage_t0)

    substage_t0 = time.perf_counter()
    neighbor_payload = dual_downward_artifacts.neighbor_list
    payload_block_leaf_ids = getattr(neighbor_payload, "target_block_leaf_ids", None)
    payload_block_source_leaf_ids = getattr(
        neighbor_payload, "target_block_source_leaf_ids", None
    )
    payload_block_valid_mask = getattr(
        neighbor_payload, "target_block_valid_mask", None
    )
    payload_block_offsets = getattr(neighbor_payload, "target_block_offsets", None)
    payload_block_size = int(getattr(neighbor_payload, "target_block_size", 0))
    num_leaves = int(jnp.asarray(neighbor_payload.leaf_indices).shape[0])
    target_blocks_leaf_major = False
    block_size = int(execution_config.target_owned_block_size)
    target_block_source_leaf_ids_padded = None
    target_block_valid_mask_padded = None
    static_target_blocks_used = False
    if (
        bool(static_target_blocks_enabled)
        and bool(execution_config.speed_prepared_layout)
        and block_size > 0
        and int(leaf_particle_indices.size) > 0
    ):
        (
            static_source_leaf_ids_padded,
            static_valid_mask_padded,
            static_capacity_ok,
        ) = build_large_n_target_owned_blocks_static(
            tree=tree_artifacts.tree,
            neighbor_list=neighbor_payload,
            block_size=block_size,
            max_blocks_per_leaf=int(static_target_blocks_max_per_leaf),
        )
        if bool(static_capacity_ok):
            target_block_source_leaf_ids_padded = static_source_leaf_ids_padded
            target_block_valid_mask_padded = static_valid_mask_padded
            target_block_source_leaf_ids = jnp.zeros(
                (0, block_size), dtype=INDEX_DTYPE
            )
            target_block_valid_mask = jnp.zeros((0, block_size), dtype=bool)
            target_block_leaf_ids = jnp.zeros((0,), dtype=INDEX_DTYPE)
            target_block_offsets = jnp.zeros((num_leaves + 1,), dtype=INDEX_DTYPE)
            target_blocks_leaf_major = True
            static_target_blocks_used = True
    if (
        not bool(static_target_blocks_used)
        and int(execution_config.target_owned_block_size) > 0
        and payload_block_leaf_ids is not None
        and payload_block_source_leaf_ids is not None
        and payload_block_valid_mask is not None
        and payload_block_size == int(execution_config.target_owned_block_size)
    ):
        target_block_leaf_ids = jnp.asarray(payload_block_leaf_ids, dtype=INDEX_DTYPE)
        target_block_source_leaf_ids = jnp.asarray(
            payload_block_source_leaf_ids, dtype=INDEX_DTYPE
        )
        target_block_valid_mask = jnp.asarray(payload_block_valid_mask, dtype=bool)
        if payload_block_offsets is not None:
            payload_offsets = jnp.asarray(payload_block_offsets, dtype=INDEX_DTYPE)
            if payload_offsets.shape == (num_leaves + 1,):
                target_block_offsets = payload_offsets
                target_blocks_leaf_major = True
            else:
                if int(target_block_leaf_ids.shape[0]) > 0:
                    block_counts = jnp.bincount(
                        target_block_leaf_ids, length=num_leaves
                    )
                    target_block_offsets = jnp.concatenate(
                        [
                            jnp.zeros((1,), dtype=INDEX_DTYPE),
                            jnp.cumsum(block_counts, dtype=INDEX_DTYPE),
                        ]
                    )
                else:
                    target_block_offsets = jnp.zeros(
                        (num_leaves + 1,), dtype=INDEX_DTYPE
                    )
        else:
            if int(target_block_leaf_ids.shape[0]) > 0:
                block_counts = jnp.bincount(target_block_leaf_ids, length=num_leaves)
                target_block_offsets = jnp.concatenate(
                    [
                        jnp.zeros((1,), dtype=INDEX_DTYPE),
                        jnp.cumsum(block_counts, dtype=INDEX_DTYPE),
                    ]
                )
            else:
                target_block_offsets = jnp.zeros((num_leaves + 1,), dtype=INDEX_DTYPE)
    elif not bool(static_target_blocks_used):
        (
            target_block_leaf_ids,
            target_block_source_leaf_ids,
            target_block_valid_mask,
            target_block_offsets,
        ) = build_large_n_target_owned_blocks(
            tree=tree_artifacts.tree,
            neighbor_list=neighbor_payload,
            block_size=int(execution_config.target_owned_block_size),
        )
        target_blocks_leaf_major = True
    _record_nf("_refresh_timing_nearfield_target_blocks_seconds", substage_t0)

    substage_t0 = time.perf_counter()
    if int(target_block_leaf_ids.shape[0]) > 0 and not bool(target_blocks_leaf_major):
        # Normalize to stable leaf-major ordering once at prepare time so the
        # runtime TONB kernel can reduce contiguous target runs without
        # per-batch sort overhead.
        block_order = jnp.argsort(target_block_leaf_ids, stable=True)
        target_block_leaf_ids = target_block_leaf_ids[block_order]
        target_block_source_leaf_ids = target_block_source_leaf_ids[block_order]
        target_block_valid_mask = target_block_valid_mask[block_order]
        block_counts = jnp.bincount(target_block_leaf_ids, length=num_leaves)
        target_block_offsets = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=INDEX_DTYPE),
                jnp.cumsum(block_counts, dtype=INDEX_DTYPE),
            ]
        )
    _record_nf("_refresh_timing_nearfield_block_sort_seconds", substage_t0)

    if bool(execution_config.speed_prepared_layout):
        target_leaf_block_counts = target_block_offsets[1:] - target_block_offsets[:-1]
    else:
        target_leaf_block_counts = None

    substage_t0 = time.perf_counter()
    if bool(execution_config.speed_prepared_layout):
        if (
            not bool(static_target_blocks_used)
            and
            block_size > 0
            and int(target_block_source_leaf_ids.shape[0]) > 0
            and target_leaf_block_counts is not None
        ):
            fast_blocks_raw = os.environ.get(
                "JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS",
                "8",
            )
            try:
                fast_blocks = max(1, int(fast_blocks_raw))
            except Exception:
                fast_blocks = 8
            max_leaf_blocks = int(jnp.max(target_leaf_block_counts))
            logical_fast_blocks = min(fast_blocks, max_leaf_blocks)
            target_block_tile_size = int(nearfield_target_block_tile_size)
            aligned_fast_blocks = (
                (max(1, logical_fast_blocks) + target_block_tile_size - 1)
                // target_block_tile_size
            ) * target_block_tile_size
            speed_layout_max_mb_raw = os.environ.get(
                "JACCPOT_LARGE_N_SPEED_PREPARED_MAX_MB",
                "256",
            )
            try:
                speed_layout_max_mb = max(0.0, float(speed_layout_max_mb_raw))
            except Exception:
                speed_layout_max_mb = 256.0
            est_layout_bytes = float(
                num_leaves
                * max(1, aligned_fast_blocks)
                * block_size
                * (jnp.dtype(INDEX_DTYPE).itemsize + jnp.dtype(bool).itemsize)
            )
            est_layout_mb = est_layout_bytes / (1024.0 * 1024.0)
            if logical_fast_blocks > 0 and est_layout_mb <= speed_layout_max_mb:
                block_idx_offsets = jnp.arange(aligned_fast_blocks, dtype=INDEX_DTYPE)
                block_idx = target_block_offsets[:-1, None] + block_idx_offsets[None, :]
                block_valid = (
                    block_idx_offsets[None, :] < int(logical_fast_blocks)
                ) & (block_idx_offsets[None, :] < target_leaf_block_counts[:, None])
                safe_block_idx = jnp.where(block_valid, block_idx, 0)
                target_block_source_leaf_ids_padded = jnp.where(
                    block_valid[:, :, None],
                    target_block_source_leaf_ids[safe_block_idx],
                    0,
                )
                target_block_valid_mask_padded = (
                    target_block_valid_mask[safe_block_idx] & block_valid[:, :, None]
                )
                # Compact overflow blocks so fallback target-block kernels only
                # process high-degree tail work instead of all blocks.
                offsets_np = np.asarray(target_block_offsets, dtype=np.int64)
                source_np = np.asarray(target_block_source_leaf_ids)
                valid_np = np.asarray(target_block_valid_mask)
                block_leaf_ids_np = np.asarray(target_block_leaf_ids, dtype=np.int64)
                counts_np = np.diff(offsets_np)
                fast_counts_np = np.minimum(counts_np, np.int64(logical_fast_blocks))
                overflow_counts_np = counts_np - fast_counts_np
                overflow_offsets_np = np.zeros((num_leaves + 1,), dtype=np.int64)
                overflow_offsets_np[1:] = np.cumsum(overflow_counts_np, dtype=np.int64)
                overflow_total = int(overflow_offsets_np[-1])
                if overflow_total > 0:
                    block_ids_np = np.arange(block_leaf_ids_np.shape[0], dtype=np.int64)
                    block_local_idx_np = block_ids_np - offsets_np[block_leaf_ids_np]
                    keep_np = block_local_idx_np >= fast_counts_np[block_leaf_ids_np]
                    overflow_source_np = source_np[keep_np]
                    overflow_valid_np = valid_np[keep_np]
                    overflow_leaf_ids_np = block_leaf_ids_np[keep_np]
                    if int(overflow_source_np.shape[0]) != int(overflow_total):
                        raise RuntimeError(
                            "overflow compaction mismatch: "
                            f"expected={overflow_total}, got={overflow_source_np.shape[0]}"
                        )
                else:
                    overflow_source_np = np.zeros(
                        (0, block_size),
                        dtype=source_np.dtype,
                    )
                    overflow_valid_np = np.zeros(
                        (0, block_size),
                        dtype=valid_np.dtype,
                    )
                    overflow_leaf_ids_np = np.zeros((0,), dtype=np.int64)
                if overflow_total > 0:
                    target_block_source_leaf_ids = jnp.asarray(
                        overflow_source_np,
                        dtype=INDEX_DTYPE,
                    )
                    target_block_valid_mask = jnp.asarray(overflow_valid_np, dtype=bool)
                    target_block_leaf_ids = jnp.asarray(
                        overflow_leaf_ids_np,
                        dtype=INDEX_DTYPE,
                    )
                    target_block_offsets = jnp.asarray(
                        overflow_offsets_np,
                        dtype=INDEX_DTYPE,
                    )
                else:
                    target_block_source_leaf_ids = jnp.zeros(
                        (0, block_size),
                        dtype=INDEX_DTYPE,
                    )
                    target_block_valid_mask = jnp.zeros((0, block_size), dtype=bool)
                    target_block_leaf_ids = jnp.zeros((0,), dtype=INDEX_DTYPE)
                    target_block_offsets = jnp.zeros(
                        (num_leaves + 1,), dtype=INDEX_DTYPE
                    )
                target_leaf_block_counts = (
                    target_block_offsets[1:] - target_block_offsets[:-1]
                )
    _record_nf("_refresh_timing_nearfield_speed_layout_seconds", substage_t0)

    substage_t0 = time.perf_counter()
    overflow_active_blocks = int(target_block_source_leaf_ids.shape[0])
    overflow_profile_capacity = int(getattr(fmm, "_large_n_overflow_profile_cap", 0))
    if overflow_profile_capacity <= 0 and overflow_profile_bootstrap_cap > 0:
        overflow_profile_capacity = _pick_overflow_profile_capacity(
            int(overflow_profile_bootstrap_cap)
        )
        setattr(fmm, "_large_n_overflow_profile_cap", int(overflow_profile_capacity))
    if overflow_active_blocks > overflow_profile_capacity:
        required_blocks = int(
            np.ceil(float(overflow_active_blocks) * float(overflow_profile_headroom))
        )
        next_capacity = _pick_overflow_profile_capacity(required_blocks)
        if overflow_profile_capacity > 0 and next_capacity > overflow_profile_capacity:
            setattr(
                fmm,
                "_large_n_overflow_profile_reprofiles",
                int(getattr(fmm, "_large_n_overflow_profile_reprofiles", 0)) + 1,
            )
        overflow_profile_capacity = int(next_capacity)
        setattr(fmm, "_large_n_overflow_profile_cap", int(overflow_profile_capacity))

    if overflow_profile_capacity > 0 and overflow_active_blocks < overflow_profile_capacity:
        pad_rows = int(overflow_profile_capacity - overflow_active_blocks)
        block_size = int(target_block_source_leaf_ids.shape[1])
        target_block_leaf_ids = jnp.concatenate(
            [
                target_block_leaf_ids,
                jnp.zeros((pad_rows,), dtype=INDEX_DTYPE),
            ],
            axis=0,
        )
        target_block_source_leaf_ids = jnp.concatenate(
            [
                target_block_source_leaf_ids,
                jnp.zeros((pad_rows, block_size), dtype=INDEX_DTYPE),
            ],
            axis=0,
        )
        target_block_valid_mask = jnp.concatenate(
            [
                target_block_valid_mask,
                jnp.zeros((pad_rows, block_size), dtype=bool),
            ],
            axis=0,
        )
    _record_nf("_refresh_timing_nearfield_overflow_profile_seconds", substage_t0)

    radix_fast_payload = None
    substage_t0 = time.perf_counter()
    if (
        bool(execution_config.radix_fast_lane)
        and target_block_source_leaf_ids_padded is not None
        and target_block_valid_mask_padded is not None
        and int(leaf_particle_indices.size) > 0
    ):
        source_slot_tile_raw = os.environ.get(
            "JACCPOT_LARGE_N_RADIX_FAST_SOURCE_SLOT_TILE",
            "64",
        )
        batch_tile_t = int(nearfield_target_leaf_batch_size)
        try:
            source_slot_tile = max(1, int(source_slot_tile_raw))
        except Exception:
            source_slot_tile = 64
        source_slot_scan_unroll = int(nearfield_target_block_tile_scan_unroll)
        target_batch_scan_unroll = int(nearfield_target_block_batch_scan_unroll)
        fallback_block_tile_size = int(nearfield_target_block_tile_size)

        target_particle_ids = jnp.asarray(leaf_particle_indices, dtype=INDEX_DTYPE)
        target_particle_mask = jnp.asarray(leaf_particle_mask, dtype=bool)
        source_leaf_ids_padded = jnp.asarray(
            target_block_source_leaf_ids_padded, dtype=INDEX_DTYPE
        )
        source_leaf_valid_padded = jnp.asarray(
            target_block_valid_mask_padded, dtype=bool
        )

        num_target_leaves = int(target_particle_ids.shape[0])
        target_leaf_ids = jnp.arange(num_target_leaves, dtype=INDEX_DTYPE)
        source_slots = int(source_leaf_ids_padded.shape[1]) * int(
            source_leaf_ids_padded.shape[2]
        )
        source_leaf_size = int(target_particle_ids.shape[1])

        source_leaf_ids_flat = source_leaf_ids_padded.reshape(
            (num_target_leaves, source_slots)
        )
        source_leaf_valid_flat = source_leaf_valid_padded.reshape(
            (num_target_leaves, source_slots)
        )
        safe_source_leaf_ids = jnp.where(
            source_leaf_valid_flat, source_leaf_ids_flat, 0
        )

        payload_max_mb_raw = os.environ.get(
            "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB",
            "1024",
        )
        try:
            payload_max_mb = max(0.0, float(payload_max_mb_raw))
        except Exception:
            payload_max_mb = 1024.0
        est_payload_bytes = float(
            num_target_leaves
            * max(1, source_slots)
            * max(1, source_leaf_size)
            * (jnp.dtype(INDEX_DTYPE).itemsize + jnp.dtype(bool).itemsize)
        )
        est_payload_mb = est_payload_bytes / (1024.0 * 1024.0)

        if source_slots > 0 and est_payload_mb <= payload_max_mb:
            source_particle_ids = target_particle_ids[safe_source_leaf_ids]
            source_particle_mask = (
                target_particle_mask[safe_source_leaf_ids]
                & source_leaf_valid_flat[:, :, None]
            )
        else:
            source_particle_ids = jnp.zeros((0, 0, 0), dtype=INDEX_DTYPE)
            source_particle_mask = jnp.zeros((0, 0, 0), dtype=bool)

        radix_fast_payload = RadixFastNearfieldPayload(
            target_leaf_ids=target_leaf_ids,
            target_particle_ids=target_particle_ids,
            target_particle_mask=target_particle_mask,
            source_leaf_ids=source_leaf_ids_padded,
            source_leaf_valid_mask=source_leaf_valid_padded,
            source_particle_ids=source_particle_ids,
            source_particle_mask=source_particle_mask,
            batch_tile_t=int(batch_tile_t),
            batch_tile_s=int(source_slot_tile),
            source_slot_scan_unroll=int(source_slot_scan_unroll),
            target_batch_scan_unroll=int(target_batch_scan_unroll),
            fallback_block_tile_size=int(fallback_block_tile_size),
            fallback_tile_scan_unroll=int(source_slot_scan_unroll),
            fallback_batch_scan_unroll=int(target_batch_scan_unroll),
        )
    _record_nf("_refresh_timing_nearfield_radix_payload_seconds", substage_t0)

    substage_t0 = time.perf_counter()
    state_neighbor_list = neighbor_payload
    if bool(execution_config.radix_fast_lane):
        # Memory trim for radix fast lane:
        # neighbor_leaf_positions duplicates information recoverable from
        # offsets+neighbors and is not needed by the fast-lane accel path.
        # Keep it out of prepared state to reduce resident memory.
        state_neighbor_list = neighbor_payload._replace(neighbor_leaf_positions=None)
        # The radix fast-lane evaluator does not require generic edge-list
        # precompute vectors. Keeping them optional/empty avoids carrying
        # topology-varying edge payloads that can trigger extra recompiles.
        state_target_leaf_ids = None
        state_source_leaf_ids = None
        state_valid_pairs = None
        neighbor_edges = jnp.asarray(state_neighbor_list.neighbors, dtype=INDEX_DTYPE)
        neighbor_active_edges = int(neighbor_edges.shape[0])
        neighbor_profile_capacity = int(
            getattr(fmm, "_large_n_neighbor_edges_profile_cap", 0)
        )
        if neighbor_profile_capacity <= 0 and neighbor_profile_bootstrap_cap > 0:
            neighbor_profile_capacity = _pick_neighbor_profile_capacity(
                int(neighbor_profile_bootstrap_cap)
            )
            setattr(
                fmm,
                "_large_n_neighbor_edges_profile_cap",
                int(neighbor_profile_capacity),
            )
        if neighbor_active_edges > neighbor_profile_capacity:
            required_edges = int(
                np.ceil(float(neighbor_active_edges) * float(neighbor_profile_headroom))
            )
            next_capacity = _pick_neighbor_profile_capacity(required_edges)
            if (
                neighbor_profile_capacity > 0
                and next_capacity > neighbor_profile_capacity
            ):
                setattr(
                    fmm,
                    "_large_n_neighbor_edges_profile_reprofiles",
                    int(
                        getattr(
                            fmm,
                            "_large_n_neighbor_edges_profile_reprofiles",
                            0,
                        )
                    )
                    + 1,
                )
            neighbor_profile_capacity = int(next_capacity)
            setattr(
                fmm,
                "_large_n_neighbor_edges_profile_cap",
                int(neighbor_profile_capacity),
            )
        if neighbor_profile_capacity > 0 and neighbor_active_edges < neighbor_profile_capacity:
            pad_edges = int(neighbor_profile_capacity - neighbor_active_edges)
            neighbor_edges = jnp.concatenate(
                [
                    neighbor_edges,
                    jnp.zeros((pad_edges,), dtype=INDEX_DTYPE),
                ],
                axis=0,
            )
            state_neighbor_list = state_neighbor_list._replace(neighbors=neighbor_edges)
    else:
        state_target_leaf_ids = nearfield_artifacts.target_leaf_ids
        state_source_leaf_ids = nearfield_artifacts.source_leaf_ids
        state_valid_pairs = nearfield_artifacts.valid_pairs
    _record_nf("_refresh_timing_nearfield_neighbor_padding_seconds", substage_t0)

    substage_t0 = time.perf_counter()
    out_state = LargeNPreparedState(
        tree=tree_artifacts.tree,
        local_data=dual_downward_artifacts.downward.locals,
        neighbor_list=state_neighbor_list,
        nearfield_leaf_particle_indices=leaf_particle_indices,
        nearfield_leaf_particle_mask=leaf_particle_mask,
        nearfield_target_leaf_ids=state_target_leaf_ids,
        nearfield_source_leaf_ids=state_source_leaf_ids,
        nearfield_valid_pairs=state_valid_pairs,
        nearfield_chunk_sort_indices=nearfield_artifacts.chunk_sort_indices,
        nearfield_chunk_group_ids=nearfield_artifacts.chunk_group_ids,
        nearfield_chunk_unique_indices=nearfield_artifacts.chunk_unique_indices,
        nearfield_target_block_leaf_ids=target_block_leaf_ids,
        nearfield_target_block_source_leaf_ids=target_block_source_leaf_ids,
        nearfield_target_block_valid_mask=target_block_valid_mask,
        nearfield_target_block_offsets=target_block_offsets,
        nearfield_target_block_source_leaf_ids_padded=(
            target_block_source_leaf_ids_padded
        ),
        nearfield_target_block_valid_mask_padded=target_block_valid_mask_padded,
        nearfield_target_block_size=int(execution_config.target_owned_block_size),
        max_leaf_size=int(tree_artifacts.leaf_cap),
        input_dtype=jnp.dtype(input_dtype),
        working_dtype=jnp.dtype(positions_arr.dtype),
        theta=float(theta_val),
        topology_key=tree_artifacts.topology_key,
        retry_events=retry_events_tuple,
        execution_backend="large_n",
        expansion_basis="solidfmm",
        nearfield_mode=str(execution_config.nearfield_mode),
        nearfield_edge_chunk_size=int(execution_config.nearfield_edge_chunk_size),
        nearfield_delayed_scatter_chunks_per_superchunk=int(
            nearfield_delayed_scatter_chunks_per_superchunk
        ),
        nearfield_chunk_scan_batch_size=int(nearfield_chunk_scan_batch_size),
        nearfield_chunk_scan_unroll=int(nearfield_chunk_scan_unroll),
        nearfield_superchunk_scan_unroll=int(nearfield_superchunk_scan_unroll),
        nearfield_sorted_scatter_hint=bool(nearfield_sorted_scatter_hint),
        nearfield_grouped_sorted_scatter=bool(nearfield_grouped_sorted_scatter),
        nearfield_superchunk_target_reduce=bool(nearfield_superchunk_target_reduce),
        nearfield_disable_chunk_cond=bool(nearfield_disable_chunk_cond),
        nearfield_target_leaf_batch_size=int(nearfield_target_leaf_batch_size),
        nearfield_target_block_tile_size=int(nearfield_target_block_tile_size),
        nearfield_target_block_tile_scan_unroll=int(
            nearfield_target_block_tile_scan_unroll
        ),
        nearfield_target_block_batch_scan_unroll=int(
            nearfield_target_block_batch_scan_unroll
        ),
        nearfield_target_block_overflow_fast_max_blocks=int(
            nearfield_target_block_overflow_fast_max_blocks
        ),
        nearfield_target_block_overflow_profile_capacity=int(overflow_profile_capacity),
        nearfield_target_block_overflow_active_blocks=int(overflow_active_blocks),
        speed_prepared_layout=bool(execution_config.speed_prepared_layout),
        radix_fast_lane=bool(execution_config.radix_fast_lane),
        disable_specialized_large_n_nearfield=bool(
            disable_specialized_large_n_nearfield
        ),
        radix_fast_payload=radix_fast_payload,
    )
    _record_nf("_refresh_timing_nearfield_state_pack_seconds", substage_t0)
    if bool(getattr(fmm, "_refresh_timing_active", False)):
        setattr(
            fmm,
            "_refresh_timing_nearfield_seconds",
            float(getattr(fmm, "_refresh_timing_nearfield_seconds", 0.0))
            + float(time.perf_counter() - stage_t0),
        )
        setattr(
            fmm,
            "_refresh_timing_nearfield_residual_seconds",
            float(getattr(fmm, "_refresh_timing_nearfield_residual_seconds", 0.0))
            + max(
                0.0,
                float(time.perf_counter() - nearfield_total_t0)
                - float(nearfield_stage_sum),
            ),
        )
    return out_state


def evaluate_large_n_state(
    fmm: object,
    state: LargeNPreparedState,
    *,
    target_indices: Optional[Array],
    return_potential: bool,
    max_acc_derivative_order: int,
) -> Any:
    """Evaluate large-N prepared state for the full particle set.

    Acceleration evaluation on the production large-N path is locked to the
    radix fast-lane payload route. Potential evaluation still falls back to the
    compiled generic evaluator until a dedicated fast-lane potential path is
    implemented.
    """

    from ._fmm_impl import (
        _evaluate_local_expansions_for_particles,
        _evaluate_tree_compiled_impl,
    )

    if target_indices is not None:
        raise NotImplementedError(
            "Large-N runtime path currently supports full-particle evaluation only."
        )
    if int(max_acc_derivative_order) != 0:
        raise NotImplementedError(
            "Large-N runtime path currently does not support acceleration derivatives."
        )

    leaf_nodes = jnp.asarray(state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    node_ranges = jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE)
    nearfield_mode = str(state.nearfield_mode).strip().lower()
    if nearfield_mode != "bucketed":
        raise RuntimeError(
            "large_n evaluation requires nearfield_mode='bucketed' prepared state"
        )
    if (not bool(getattr(state, "radix_fast_lane", False))) and (
        not bool(return_potential)
    ):
        raise RuntimeError(
            "large_n acceleration evaluation requires radix fast-lane state; "
            "prepare state with the large_n_gpu radix profile before accel-only evaluate"
        )
    if bool(getattr(state, "radix_fast_lane", False)) and not bool(return_potential):
        near_acc = evaluate_large_n_nearfield_fast_lane(
            fmm,
            state,
            return_potential=False,
        )
        far_grad, _, _ = _evaluate_local_expansions_for_particles(
            state.local_data,
            state.positions_sorted,
            leaf_nodes=leaf_nodes,
            node_ranges=node_ranges,
            max_leaf_size=int(state.max_leaf_size),
            order=int(state.local_data.order),
            expansion_basis="solidfmm",
            return_potential=False,
            max_acc_derivative_order=0,
        )
        far_acc = -float(getattr(fmm, "G")) * far_grad
        accelerations_sorted = near_acc + far_acc
        if jnp.issubdtype(state.input_dtype, jnp.floating):
            output_dtype = state.input_dtype
        else:
            output_dtype = state.working_dtype
        return jnp.asarray(accelerations_sorted)[state.inverse_permutation].astype(
            output_dtype
        )

    nearfield_edge_chunk_size = int(state.nearfield_edge_chunk_size)
    eval_out = _evaluate_tree_compiled_impl(
        state.tree,
        state.positions_sorted,
        state.masses_sorted,
        state.local_data,
        state.neighbor_list,
        leaf_nodes,
        node_ranges,
        jnp.asarray(state.neighbor_list.offsets, dtype=INDEX_DTYPE),
        jnp.asarray(state.neighbor_list.neighbors, dtype=INDEX_DTYPE),
        jnp.asarray(state.neighbor_list.counts, dtype=INDEX_DTYPE),
        (
            jnp.asarray(state.nearfield_leaf_particle_indices, dtype=INDEX_DTYPE)
            if int(state.nearfield_leaf_particle_indices.size) > 0
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_leaf_particle_mask, dtype=bool)
            if int(state.nearfield_leaf_particle_mask.size) > 0
            else jnp.zeros((0, 0), dtype=bool)
        ),
        leaf_nodes,
        node_ranges,
        (
            jnp.asarray(state.nearfield_target_leaf_ids, dtype=INDEX_DTYPE)
            if state.nearfield_target_leaf_ids is not None
            else jnp.zeros((0,), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_source_leaf_ids, dtype=INDEX_DTYPE)
            if state.nearfield_source_leaf_ids is not None
            else jnp.zeros((0,), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_valid_pairs, dtype=bool)
            if state.nearfield_valid_pairs is not None
            else jnp.zeros((0,), dtype=bool)
        ),
        (
            jnp.asarray(state.nearfield_chunk_sort_indices, dtype=INDEX_DTYPE)
            if state.nearfield_chunk_sort_indices is not None
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_chunk_group_ids, dtype=INDEX_DTYPE)
            if state.nearfield_chunk_group_ids is not None
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_chunk_unique_indices, dtype=INDEX_DTYPE)
            if state.nearfield_chunk_unique_indices is not None
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_target_block_offsets, dtype=INDEX_DTYPE)
            if state.nearfield_target_block_offsets is not None
            else jnp.zeros((leaf_nodes.shape[0] + 1,), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_target_block_leaf_ids, dtype=INDEX_DTYPE)
            if state.nearfield_target_block_leaf_ids is not None
            else jnp.zeros((0,), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_target_block_source_leaf_ids, dtype=INDEX_DTYPE)
            if state.nearfield_target_block_source_leaf_ids is not None
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state.nearfield_target_block_valid_mask, dtype=bool)
            if state.nearfield_target_block_valid_mask is not None
            else jnp.zeros((0, 0), dtype=bool)
        ),
        (
            jnp.asarray(
                state.nearfield_target_block_source_leaf_ids_padded,
                dtype=INDEX_DTYPE,
            )
            if state.nearfield_target_block_source_leaf_ids_padded is not None
            else jnp.zeros((leaf_nodes.shape[0], 0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(
                state.nearfield_target_block_valid_mask_padded,
                dtype=bool,
            )
            if state.nearfield_target_block_valid_mask_padded is not None
            else jnp.zeros((leaf_nodes.shape[0], 0, 0), dtype=bool)
        ),
        G=float(getattr(fmm, "G")),
        softening=float(getattr(fmm, "softening")),
        order=int(state.local_data.order),
        expansion_basis="solidfmm",
        max_leaf_size=int(state.max_leaf_size),
        return_potential=bool(return_potential),
        nearfield_mode=nearfield_mode,
        nearfield_edge_chunk_size=nearfield_edge_chunk_size,
        nearfield_delayed_scatter_chunks_per_superchunk=int(
            state.nearfield_delayed_scatter_chunks_per_superchunk
        ),
        nearfield_chunk_scan_batch_size=int(state.nearfield_chunk_scan_batch_size),
        nearfield_chunk_scan_unroll=int(state.nearfield_chunk_scan_unroll),
        nearfield_superchunk_scan_unroll=int(state.nearfield_superchunk_scan_unroll),
        nearfield_sorted_scatter_hint=bool(state.nearfield_sorted_scatter_hint),
        nearfield_grouped_sorted_scatter=bool(state.nearfield_grouped_sorted_scatter),
        nearfield_superchunk_target_reduce=bool(
            state.nearfield_superchunk_target_reduce
        ),
        nearfield_disable_chunk_cond=bool(state.nearfield_disable_chunk_cond),
        nearfield_target_leaf_batch_size=int(state.nearfield_target_leaf_batch_size),
        nearfield_target_block_tile_size=int(state.nearfield_target_block_tile_size),
        nearfield_target_block_tile_scan_unroll=int(
            state.nearfield_target_block_tile_scan_unroll
        ),
        nearfield_target_block_batch_scan_unroll=int(
            state.nearfield_target_block_batch_scan_unroll
        ),
        nearfield_target_block_overflow_fast_max_blocks=int(
            state.nearfield_target_block_overflow_fast_max_blocks
        ),
        disable_specialized_large_n_nearfield=bool(
            state.disable_specialized_large_n_nearfield
        ),
    )

    if jnp.issubdtype(state.input_dtype, jnp.floating):
        output_dtype = state.input_dtype
    else:
        output_dtype = state.working_dtype

    if return_potential:
        accelerations_sorted, potentials_sorted = eval_out
    else:
        accelerations_sorted = eval_out

    if not return_potential:
        return jnp.asarray(accelerations_sorted)[state.inverse_permutation].astype(
            output_dtype
        )

    accelerations = jnp.asarray(accelerations_sorted)[state.inverse_permutation].astype(
        output_dtype
    )
    potentials = jnp.asarray(potentials_sorted)[state.inverse_permutation].astype(
        output_dtype
    )
    return accelerations, potentials


def can_use_large_n_prepare_path(
    fmm: object,
    *,
    positions_arr: Array,
    masses_arr: Array,
    allow_stateful_cache: bool,
) -> bool:
    """Decide whether prepare_state should dispatch to the large-N path."""

    runtime_path = str(getattr(fmm, "runtime_path", "auto")).strip().lower()
    if runtime_path not in ("auto", "legacy", "large_n"):
        return False
    if (
        runtime_path == "auto"
        and str(getattr(fmm, "preset", "")).strip().lower() != "large_n_gpu"
    ):
        return False
    if not allow_stateful_cache:
        return False
    if jax.default_backend() != "gpu":
        return False
    if str(getattr(fmm, "tree_type", "")).strip().lower() != "radix":
        return False
    if str(getattr(fmm, "expansion_basis", "")).strip().lower() != "solidfmm":
        return False
    if str(getattr(fmm, "execution_backend", "auto")).strip().lower() == "octree":
        return False
    if bool(getattr(fmm, "adaptive_order", False)):
        return False
    if bool(getattr(fmm, "mixed_order_farfield", False)):
        return False
    if str(getattr(fmm, "complex_rotation", "")).strip().lower() != "solidfmm":
        return False
    if (
        bool(getattr(fmm, "_uses_paper_style_force_scale"))
        and fmm._uses_paper_style_force_scale()
    ):
        return False
    if int(positions_arr.shape[0]) != int(masses_arr.shape[0]):
        return False
    return True
