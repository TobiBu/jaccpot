"""Dedicated prepare/evaluate pipeline for large-N GPU radix solidfmm runs."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional, Union

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
from ._large_n_types import (
    LargeNCompiledState,
    LargeNPreparedState,
    RadixFastNearfieldPayload,
    large_n_as_prepared_state,
    large_n_to_compiled_state,
)
from .dtypes import INDEX_DTYPE


def _contains_jax_tracer(value: Any) -> bool:
    return any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(value)
    )


def _read_large_n_env_config() -> dict[str, Any]:
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

    static_runtime_fixed_sizing = _env_bool(
        "JACCPOT_STATIC_RUNTIME_FIXED_SIZING",
        True,
    )
    try:
        overflow_profile_fixed_cap = max(
            0,
            int(
                os.environ.get(
                    "JACCPOT_LARGE_N_OVERFLOW_PROFILE_FIXED_CAP",
                    str(int(overflow_profile_bootstrap_cap)),
                )
            ),
        )
    except Exception:
        overflow_profile_fixed_cap = int(max(0, int(overflow_profile_bootstrap_cap)))
    try:
        neighbor_profile_fixed_cap = max(
            0,
            int(
                os.environ.get(
                    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
                    str(int(neighbor_profile_bootstrap_cap)),
                )
            ),
        )
    except Exception:
        neighbor_profile_fixed_cap = int(max(0, int(neighbor_profile_bootstrap_cap)))

    # Static target-block cap. Supports "auto" (data-driven sizing; sentinel 0)
    # in addition to explicit ints, and — for any value — auto-grows to fit the
    # densest leaf at build time (see _large_n_pipeline static-block region),
    # mirroring the neighbor/overflow cap profiling (headroom + caps ladder).
    static_target_blocks_cap_raw = (
        str(os.environ.get("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32"))
        .strip()
        .lower()
    )
    static_target_blocks_auto = static_target_blocks_cap_raw in {"auto", "-1"}
    if static_target_blocks_auto:
        static_target_blocks_max_per_leaf = 0
    else:
        try:
            static_target_blocks_max_per_leaf = max(
                1, int(static_target_blocks_cap_raw)
            )
        except Exception:
            static_target_blocks_max_per_leaf = 0
            static_target_blocks_auto = True
    static_target_blocks_headroom_raw = os.environ.get(
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_HEADROOM", "1.25"
    )
    try:
        static_target_blocks_headroom = max(
            1.0, float(static_target_blocks_headroom_raw)
        )
    except Exception:
        static_target_blocks_headroom = 1.25
    static_target_blocks_cap_options: list[int] = []
    for token in str(
        os.environ.get(
            "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF_OPTIONS",
            "8,16,32,64,128,256,512,1024,2048,4096",
        )
    ).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0 and value not in static_target_blocks_cap_options:
            static_target_blocks_cap_options.append(value)
    static_target_blocks_cap_options = sorted(static_target_blocks_cap_options)
    if not static_target_blocks_cap_options:
        static_target_blocks_cap_options = [8, 16, 32, 64, 128, 256, 512, 1024]

    return {
        "nearfield_delayed_scatter_chunks_per_superchunk": _env_pos_int(
            "JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS", 1
        ),
        "nearfield_chunk_scan_batch_size": _env_pos_int(
            "JACCPOT_LARGE_N_CHUNK_SCAN_BATCH_SIZE", 1
        ),
        "nearfield_chunk_scan_unroll": _env_pos_int(
            "JACCPOT_LARGE_N_CHUNK_SCAN_UNROLL", 1
        ),
        "nearfield_superchunk_scan_unroll": _env_pos_int(
            "JACCPOT_LARGE_N_SUPERCHUNK_SCAN_UNROLL", 1
        ),
        "nearfield_sorted_scatter_hint": _env_bool(
            "JACCPOT_LARGE_N_SORTED_SCATTER_HINT", False
        ),
        "nearfield_grouped_sorted_scatter": _env_bool(
            "JACCPOT_LARGE_N_GROUPED_SORTED_SCATTER", False
        ),
        "nearfield_superchunk_target_reduce": _env_bool(
            "JACCPOT_LARGE_N_SUPERCHUNK_TARGET_REDUCE", False
        ),
        "nearfield_disable_chunk_cond": _env_bool(
            "JACCPOT_LARGE_N_DISABLE_CHUNK_COND", True
        ),
        "nearfield_target_leaf_batch_size": _canonical_static_int(
            "JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE",
            16,
            "JACCPOT_LARGE_N_TARGET_LEAF_BATCH_OPTIONS",
            "16,32,64",
        ),
        "nearfield_target_block_tile_size": _canonical_static_int(
            "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SIZE",
            4,
            "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_OPTIONS",
            "4,8,16",
        ),
        "nearfield_target_block_tile_scan_unroll": _canonical_static_int(
            "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SCAN_UNROLL",
            1,
            "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SCAN_UNROLL_OPTIONS",
            "1,2,4",
        ),
        "nearfield_target_block_batch_scan_unroll": _canonical_static_int(
            "JACCPOT_LARGE_N_TARGET_BLOCK_BATCH_SCAN_UNROLL",
            1,
            "JACCPOT_LARGE_N_TARGET_BLOCK_BATCH_SCAN_UNROLL_OPTIONS",
            "1,2,4",
        ),
        "nearfield_target_block_overflow_fast_max_blocks": _canonical_static_int(
            "JACCPOT_LARGE_N_TARGET_BLOCK_OVERFLOW_FAST_MAX_BLOCKS",
            65536,
            "JACCPOT_LARGE_N_TARGET_BLOCK_OVERFLOW_FAST_MAX_BLOCKS_OPTIONS",
            "16384,32768,65536,131072",
        ),
        "static_target_blocks_enabled": _env_bool(
            "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS", True
        ),
        "static_target_blocks_max_per_leaf": int(static_target_blocks_max_per_leaf),
        "static_target_blocks_auto": bool(static_target_blocks_auto),
        "static_target_blocks_headroom": float(static_target_blocks_headroom),
        "static_target_blocks_cap_options": tuple(
            int(v) for v in static_target_blocks_cap_options
        ),
        "overflow_profile_headroom": float(overflow_profile_headroom),
        "overflow_profile_caps": tuple(int(v) for v in overflow_profile_caps),
        "neighbor_profile_headroom": float(neighbor_profile_headroom),
        "neighbor_profile_caps": tuple(int(v) for v in neighbor_profile_caps),
        "neighbor_profile_bootstrap_cap": int(neighbor_profile_bootstrap_cap),
        "overflow_profile_bootstrap_cap": int(overflow_profile_bootstrap_cap),
        "static_runtime_fixed_sizing": bool(static_runtime_fixed_sizing),
        "overflow_profile_fixed_cap": int(overflow_profile_fixed_cap),
        "neighbor_profile_fixed_cap": int(neighbor_profile_fixed_cap),
        "disable_specialized_large_n_nearfield": _env_bool(
            "JACCPOT_DISABLE_LARGE_N_SPECIALIZED_NEARFIELD", False
        ),
    }


def _large_n_env_config_for_fmm(fmm: object) -> dict[str, Any]:
    cfg = getattr(fmm, "_large_n_env_config_cached", None)
    if cfg is None:
        cfg = _read_large_n_env_config()
        setattr(fmm, "_large_n_env_config_cached", cfg)
    return cfg


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
    tree_artifacts: Optional[Any] = None,
    dual_downward_artifacts: Optional[Any] = None,
    fused_device_mode: bool = False,
    execution_config_override: Optional[Any] = None,
    large_n_env_cfg_override: Optional[dict[str, Any]] = None,
    return_compiled_state: bool = False,
) -> Union[LargeNPreparedState, LargeNCompiledState]:
    """Prepare the slim large-N state using the dedicated narrow path."""

    refresh_timing_active = bool(
        getattr(fmm, "_refresh_timing_active", False)
    ) and not (
        bool(fused_device_mode)
        and bool(getattr(fmm, "_strict_fused_disable_hot_timing", False))
    )

    def _now() -> float:
        if not refresh_timing_active:
            return 0.0
        return float(time.perf_counter())

    def _elapsed(start: float) -> float:
        return float(_now() - start)

    disable_fused_tree_dual_prepare = str(
        os.environ.get("JACCPOT_LARGE_N_DISABLE_FUSED_TREE_DUAL_PREPARE", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}

    built_tree_artifacts = tree_artifacts is None
    built_dual_downward_artifacts = dual_downward_artifacts is None

    if (
        not bool(disable_fused_tree_dual_prepare)
        and tree_artifacts is None
        and dual_downward_artifacts is None
    ):
        stage_t0 = _now()
        tree_artifacts, dual_downward_artifacts = (
            fmm._prepare_state_tree_upward_and_dual_downward(
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
                theta_val=theta_val,
                mac_type_val=mac_type_val,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                record_retry=record_retry,
            )
        )
        if refresh_timing_active:
            elapsed = float(_now() - stage_t0)
            setattr(
                fmm,
                "_refresh_timing_tree_upward_seconds",
                float(getattr(fmm, "_refresh_timing_tree_upward_seconds", 0.0))
                + elapsed,
            )
            setattr(
                fmm,
                "_refresh_timing_dual_downward_seconds",
                float(getattr(fmm, "_refresh_timing_dual_downward_seconds", 0.0)) + 0.0,
            )
    else:
        stage_t0 = _now()
        if tree_artifacts is None:
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
        if refresh_timing_active and built_tree_artifacts:
            setattr(
                fmm,
                "_refresh_timing_tree_upward_seconds",
                float(getattr(fmm, "_refresh_timing_tree_upward_seconds", 0.0))
                + float(_now() - stage_t0),
            )

        stage_t0 = _now()
        if dual_downward_artifacts is None:
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
        if refresh_timing_active and built_dual_downward_artifacts:
            setattr(
                fmm,
                "_refresh_timing_dual_downward_seconds",
                float(getattr(fmm, "_refresh_timing_dual_downward_seconds", 0.0))
                + float(_now() - stage_t0),
            )

    stage_t0 = _now()
    if allow_stateful_cache:
        fmm._update_locals_template_cache_after_prepare(
            locals_template=tree_artifacts.locals_template,
            upward=tree_artifacts.upward,
            max_order=int(max_order),
        )

    retry_events_tuple = tuple(collected_retries)
    if allow_stateful_cache:
        fmm._recent_retry_events = retry_events_tuple

    if execution_config_override is None:
        execution_config = resolve_large_n_execution_config(
            fmm,
            num_particles=int(positions_arr.shape[0]),
        )
    else:
        execution_config = execution_config_override

    if large_n_env_cfg_override is None:
        large_n_env_cfg = _large_n_env_config_for_fmm(fmm)
    else:
        large_n_env_cfg = large_n_env_cfg_override
    nearfield_delayed_scatter_chunks_per_superchunk = int(
        large_n_env_cfg["nearfield_delayed_scatter_chunks_per_superchunk"]
    )
    nearfield_chunk_scan_batch_size = int(
        large_n_env_cfg["nearfield_chunk_scan_batch_size"]
    )
    nearfield_chunk_scan_unroll = int(large_n_env_cfg["nearfield_chunk_scan_unroll"])
    nearfield_superchunk_scan_unroll = int(
        large_n_env_cfg["nearfield_superchunk_scan_unroll"]
    )
    nearfield_sorted_scatter_hint = bool(
        large_n_env_cfg["nearfield_sorted_scatter_hint"]
    )
    nearfield_grouped_sorted_scatter = bool(
        large_n_env_cfg["nearfield_grouped_sorted_scatter"]
    )
    nearfield_superchunk_target_reduce = bool(
        large_n_env_cfg["nearfield_superchunk_target_reduce"]
    )
    nearfield_disable_chunk_cond = bool(large_n_env_cfg["nearfield_disable_chunk_cond"])
    nearfield_target_leaf_batch_size = int(
        large_n_env_cfg["nearfield_target_leaf_batch_size"]
    )
    nearfield_target_block_tile_size = int(
        large_n_env_cfg["nearfield_target_block_tile_size"]
    )
    nearfield_target_block_tile_scan_unroll = int(
        large_n_env_cfg["nearfield_target_block_tile_scan_unroll"]
    )
    nearfield_target_block_batch_scan_unroll = int(
        large_n_env_cfg["nearfield_target_block_batch_scan_unroll"]
    )
    nearfield_target_block_overflow_fast_max_blocks = int(
        large_n_env_cfg["nearfield_target_block_overflow_fast_max_blocks"]
    )
    static_target_blocks_enabled = bool(large_n_env_cfg["static_target_blocks_enabled"])
    static_target_blocks_max_per_leaf = int(
        large_n_env_cfg["static_target_blocks_max_per_leaf"]
    )
    static_target_blocks_auto = bool(
        large_n_env_cfg.get("static_target_blocks_auto", False)
    )
    static_target_blocks_headroom = float(
        large_n_env_cfg.get("static_target_blocks_headroom", 1.25)
    )
    static_target_blocks_cap_options = tuple(
        int(v) for v in large_n_env_cfg.get("static_target_blocks_cap_options", ())
    )
    overflow_profile_headroom = float(large_n_env_cfg["overflow_profile_headroom"])
    overflow_profile_caps = tuple(
        int(v) for v in large_n_env_cfg["overflow_profile_caps"]
    )
    neighbor_profile_headroom = float(large_n_env_cfg["neighbor_profile_headroom"])
    neighbor_profile_caps = tuple(
        int(v) for v in large_n_env_cfg["neighbor_profile_caps"]
    )
    neighbor_profile_bootstrap_cap = int(
        large_n_env_cfg["neighbor_profile_bootstrap_cap"]
    )
    overflow_profile_bootstrap_cap = int(
        large_n_env_cfg["overflow_profile_bootstrap_cap"]
    )
    static_runtime_fixed_sizing = bool(
        large_n_env_cfg.get("static_runtime_fixed_sizing", True)
    )
    overflow_profile_fixed_cap = int(
        large_n_env_cfg.get("overflow_profile_fixed_cap", 0)
    )
    neighbor_profile_fixed_cap = int(
        large_n_env_cfg.get("neighbor_profile_fixed_cap", 0)
    )

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

    def _pick_static_target_blocks_capacity(required: int) -> int:
        required = max(1, int(required))
        for cap in static_target_blocks_cap_options:
            if int(cap) >= required:
                return int(cap)
        return int(required)

    nearfield_total_t0 = stage_t0
    nearfield_stage_sum = 0.0

    def _record_nf(attr: str, start: float) -> None:
        nonlocal nearfield_stage_sum
        elapsed = float(_now() - start)
        nearfield_stage_sum += elapsed
        if refresh_timing_active:
            setattr(fmm, attr, float(getattr(fmm, attr, 0.0)) + elapsed)

    disable_specialized_large_n_nearfield = bool(
        large_n_env_cfg["disable_specialized_large_n_nearfield"]
    )
    substage_t0 = _now()
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

    substage_t0 = _now()
    skip_generic_nearfield_precompute = bool(fused_device_mode) and bool(
        execution_config.radix_fast_lane
    )
    if skip_generic_nearfield_precompute:
        precomputed_target_leaf_ids = None
        precomputed_source_leaf_ids = None
        precomputed_valid_pairs = None
        precomputed_chunk_sort_indices = None
        precomputed_chunk_group_ids = None
        precomputed_chunk_unique_indices = None
    else:
        nearfield_artifacts = build_large_n_nearfield_precompute(
            tree=tree_artifacts.tree,
            neighbor_list=dual_downward_artifacts.neighbor_list,
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
            execution_config=execution_config,
        )
        precomputed_target_leaf_ids = nearfield_artifacts.target_leaf_ids
        precomputed_source_leaf_ids = nearfield_artifacts.source_leaf_ids
        precomputed_valid_pairs = nearfield_artifacts.valid_pairs
        precomputed_chunk_sort_indices = nearfield_artifacts.chunk_sort_indices
        precomputed_chunk_group_ids = nearfield_artifacts.chunk_group_ids
        precomputed_chunk_unique_indices = nearfield_artifacts.chunk_unique_indices
    _record_nf("_refresh_timing_nearfield_precompute_seconds", substage_t0)

    substage_t0 = _now()
    neighbor_payload = dual_downward_artifacts.neighbor_list
    payload_block_leaf_ids = getattr(neighbor_payload, "target_block_leaf_ids", None)
    payload_block_source_leaf_ids = getattr(
        neighbor_payload, "target_block_source_leaf_ids", None
    )
    payload_block_valid_mask = getattr(
        neighbor_payload, "target_block_valid_mask", None
    )
    payload_block_offsets = getattr(neighbor_payload, "target_block_offsets", None)
    num_leaves = int(jnp.asarray(neighbor_payload.leaf_indices).shape[0])
    target_blocks_leaf_major = False
    block_size = int(execution_config.target_owned_block_size)
    payload_block_size = (
        block_size
        if bool(fused_device_mode) and bool(skip_generic_nearfield_precompute)
        else int(getattr(neighbor_payload, "target_block_size", 0))
    )
    target_block_source_leaf_ids_padded = None
    target_block_valid_mask_padded = None
    static_target_blocks_used = False
    fused_payload_enabled = str(
        os.environ.get(
            "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
            "1",
        )
    ).strip().lower() in {"1", "true", "yes", "on"}
    allow_static_target_blocks_in_fused = (not bool(fused_device_mode)) or bool(
        fused_payload_enabled
    )
    traced_target_block_payload = _contains_jax_tracer(
        (
            getattr(neighbor_payload, "offsets", None),
            getattr(neighbor_payload, "neighbors", None),
            leaf_particle_indices,
        )
    )
    # Resolve the effective static-target-block cap. On an eager prepare we can
    # inspect the concrete neighbour degree and auto-size the cap to fit the
    # densest leaf (mirrors the neighbor/overflow cap profiling); the resolved
    # cap is cached on the fmm so the traced strict refresh reuses the identical
    # fixed shape (zero-recompile). Grows monotonically across eager refreshes.
    resolved_cap_attr = "_large_n_fused_static_target_blocks_resolved_cap"
    cached_static_cap = int(getattr(fmm, resolved_cap_attr, 0) or 0)
    if bool(traced_target_block_payload):
        effective_static_cap = (
            cached_static_cap
            if cached_static_cap > 0
            else int(static_target_blocks_max_per_leaf)
        )
    else:
        _sb_offsets = jnp.asarray(
            getattr(neighbor_payload, "offsets", jnp.zeros((1,), dtype=INDEX_DTYPE)),
            dtype=INDEX_DTYPE,
        )
        if int(_sb_offsets.shape[0]) >= 2:
            _sb_max_count = int(jnp.max(_sb_offsets[1:] - _sb_offsets[:-1]))
        else:
            _sb_max_count = 0
        _sb_required = (
            (_sb_max_count + int(block_size) - 1) // int(block_size)
            if int(block_size) > 0
            else 0
        )
        _sb_required = max(1, int(_sb_required))
        if bool(static_target_blocks_auto) or (
            int(static_target_blocks_max_per_leaf) < _sb_required
        ):
            _sb_target = int(np.ceil(_sb_required * static_target_blocks_headroom))
            _sb_candidate = _pick_static_target_blocks_capacity(_sb_target)
        else:
            _sb_candidate = int(static_target_blocks_max_per_leaf)
        effective_static_cap = max(cached_static_cap, int(_sb_candidate), 1)
        setattr(fmm, resolved_cap_attr, int(effective_static_cap))

    preflight_key = (
        int(num_leaves),
        int(block_size),
        int(effective_static_cap),
    )
    preflight_attr = "_large_n_fused_payload_static_target_block_preflight"
    preflight_ok = getattr(fmm, preflight_attr, None) == preflight_key
    if (
        bool(fused_device_mode)
        and bool(fused_payload_enabled)
        and bool(static_target_blocks_enabled)
        and bool(execution_config.speed_prepared_layout)
        and block_size > 0
        and int(leaf_particle_indices.size) > 0
        and bool(traced_target_block_payload)
        and not bool(preflight_ok)
    ):
        raise RuntimeError(
            "fused payload static target-block cap was not preflighted before "
            "entering traced strict refresh: "
            f"num_leaves={int(num_leaves)} block_size={int(block_size)} "
            f"max_blocks_per_leaf={int(effective_static_cap)}. "
            "Run an eager prepare/refresh with the same cap first."
        )
    if (
        bool(allow_static_target_blocks_in_fused)
        and bool(static_target_blocks_enabled)
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
            max_blocks_per_leaf=int(effective_static_cap),
            check_capacity=not (
                bool(fused_device_mode)
                and bool(fused_payload_enabled)
                and bool(traced_target_block_payload)
                and bool(preflight_ok)
            ),
        )
        if (
            bool(fused_device_mode)
            and bool(fused_payload_enabled)
            and not bool(traced_target_block_payload)
            and not bool(static_capacity_ok)
        ):
            raise RuntimeError(
                "fused payload static target-block cap exceeded after auto-size: "
                f"num_leaves={int(num_leaves)} block_size={int(block_size)} "
                f"max_blocks_per_leaf={int(effective_static_cap)}. This should not "
                "happen with auto-sizing; set "
                "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=auto, raise the "
                "cap options ladder, or disable "
                "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED."
            )
        if (
            bool(fused_device_mode)
            and bool(fused_payload_enabled)
            and not bool(traced_target_block_payload)
            and bool(static_capacity_ok)
        ):
            setattr(fmm, preflight_attr, preflight_key)
        if bool(static_capacity_ok):
            target_block_source_leaf_ids_padded = static_source_leaf_ids_padded
            target_block_valid_mask_padded = static_valid_mask_padded
            target_block_source_leaf_ids = jnp.zeros((0, block_size), dtype=INDEX_DTYPE)
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
    elif (not bool(static_target_blocks_used)) and (not bool(fused_device_mode)):
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
    elif not bool(static_target_blocks_used):
        # Fused mode fallback: avoid tracer-unsafe dynamic target-block build.
        target_block_leaf_ids = jnp.zeros((0,), dtype=INDEX_DTYPE)
        target_block_source_leaf_ids = jnp.zeros((0, block_size), dtype=INDEX_DTYPE)
        target_block_valid_mask = jnp.zeros((0, block_size), dtype=bool)
        target_block_offsets = jnp.zeros((num_leaves + 1,), dtype=INDEX_DTYPE)
        target_blocks_leaf_major = True
    _record_nf("_refresh_timing_nearfield_target_blocks_seconds", substage_t0)

    substage_t0 = _now()
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

    substage_t0 = _now()
    if bool(execution_config.speed_prepared_layout) and (not bool(fused_device_mode)):
        if (
            not bool(static_target_blocks_used)
            and block_size > 0
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
            speed_layout_max_mb_raw = os.environ.get(
                "JACCPOT_LARGE_N_SPEED_PREPARED_MAX_MB",
                "256",
            )
            try:
                speed_layout_max_mb = max(0.0, float(speed_layout_max_mb_raw))
            except Exception:
                speed_layout_max_mb = 256.0

            def _aligned_block_count(block_count: int) -> int:
                return (
                    (max(1, int(block_count)) + target_block_tile_size - 1)
                    // target_block_tile_size
                ) * target_block_tile_size

            def _layout_mb(block_count: int) -> float:
                return float(
                    num_leaves
                    * max(1, int(block_count))
                    * block_size
                    * (jnp.dtype(INDEX_DTYPE).itemsize + jnp.dtype(bool).itemsize)
                ) / (1024.0 * 1024.0)

            auto_full_blocks_raw = (
                str(
                    os.environ.get(
                        "JACCPOT_LARGE_N_SPEED_PREPARED_AUTO_FULL_BLOCKS",
                        "1",
                    )
                )
                .strip()
                .lower()
            )
            auto_full_blocks = auto_full_blocks_raw in {"1", "true", "yes", "on"}
            if bool(auto_full_blocks) and max_leaf_blocks > logical_fast_blocks:
                candidate_aligned_blocks = _aligned_block_count(max_leaf_blocks)
                if _layout_mb(candidate_aligned_blocks) <= speed_layout_max_mb:
                    logical_fast_blocks = int(max_leaf_blocks)

            aligned_fast_blocks = _aligned_block_count(logical_fast_blocks)
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
                if not bool(fused_device_mode):
                    # Compact overflow blocks so fallback target-block kernels only
                    # process high-degree tail work instead of all blocks.
                    offsets_np = np.asarray(target_block_offsets, dtype=np.int64)
                    source_np = np.asarray(target_block_source_leaf_ids)
                    valid_np = np.asarray(target_block_valid_mask)
                    block_leaf_ids_np = np.asarray(
                        target_block_leaf_ids, dtype=np.int64
                    )
                    counts_np = np.diff(offsets_np)
                    fast_counts_np = np.minimum(
                        counts_np, np.int64(logical_fast_blocks)
                    )
                    overflow_counts_np = counts_np - fast_counts_np
                    overflow_offsets_np = np.zeros((num_leaves + 1,), dtype=np.int64)
                    overflow_offsets_np[1:] = np.cumsum(
                        overflow_counts_np, dtype=np.int64
                    )
                    overflow_total = int(overflow_offsets_np[-1])
                    if overflow_total > 0:
                        block_ids_np = np.arange(
                            block_leaf_ids_np.shape[0], dtype=np.int64
                        )
                        block_local_idx_np = (
                            block_ids_np - offsets_np[block_leaf_ids_np]
                        )
                        keep_np = (
                            block_local_idx_np >= fast_counts_np[block_leaf_ids_np]
                        )
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
                        target_block_valid_mask = jnp.asarray(
                            overflow_valid_np, dtype=bool
                        )
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

    substage_t0 = _now()
    overflow_active_blocks = int(target_block_source_leaf_ids.shape[0])
    if bool(fused_device_mode) and static_runtime_fixed_sizing:
        overflow_profile_capacity = int(overflow_profile_fixed_cap)
        if overflow_profile_capacity <= 0:
            overflow_profile_capacity = int(
                getattr(fmm, "_large_n_overflow_profile_cap", 0)
            )
            if overflow_profile_capacity <= 0:
                overflow_profile_capacity = int(overflow_active_blocks)
            setattr(
                fmm, "_large_n_overflow_profile_cap", int(overflow_profile_capacity)
            )
        if overflow_active_blocks > overflow_profile_capacity:
            raise RuntimeError(
                "static runtime sizing overflow cap exceeded: "
                f"active_blocks={overflow_active_blocks} cap={overflow_profile_capacity}. "
                "Increase JACCPOT_LARGE_N_OVERFLOW_PROFILE_FIXED_CAP."
            )
        if overflow_active_blocks < overflow_profile_capacity:
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
    elif bool(fused_device_mode):
        # Backward-compatible non-fixed fused mode: keep dynamic active size.
        overflow_profile_capacity = int(overflow_active_blocks)
    elif static_runtime_fixed_sizing:
        overflow_profile_capacity = int(overflow_profile_fixed_cap)
        if (
            overflow_profile_capacity > 0
            and overflow_active_blocks > overflow_profile_capacity
        ):
            raise RuntimeError(
                "static runtime sizing overflow cap exceeded: "
                f"active_blocks={overflow_active_blocks} cap={overflow_profile_capacity}. "
                "Increase JACCPOT_LARGE_N_OVERFLOW_PROFILE_FIXED_CAP."
            )
        if overflow_profile_capacity <= 0:
            overflow_profile_capacity = int(overflow_active_blocks)
        elif overflow_active_blocks < overflow_profile_capacity:
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
    else:
        overflow_profile_capacity = int(
            getattr(fmm, "_large_n_overflow_profile_cap", 0)
        )
        if overflow_profile_capacity <= 0 and overflow_profile_bootstrap_cap > 0:
            overflow_profile_capacity = _pick_overflow_profile_capacity(
                int(overflow_profile_bootstrap_cap)
            )
            setattr(
                fmm, "_large_n_overflow_profile_cap", int(overflow_profile_capacity)
            )
        if overflow_active_blocks > overflow_profile_capacity:
            required_blocks = int(
                np.ceil(
                    float(overflow_active_blocks) * float(overflow_profile_headroom)
                )
            )
            next_capacity = _pick_overflow_profile_capacity(required_blocks)
            if (
                overflow_profile_capacity > 0
                and next_capacity > overflow_profile_capacity
            ):
                setattr(
                    fmm,
                    "_large_n_overflow_profile_reprofiles",
                    int(getattr(fmm, "_large_n_overflow_profile_reprofiles", 0)) + 1,
                )
            overflow_profile_capacity = int(next_capacity)
            setattr(
                fmm, "_large_n_overflow_profile_cap", int(overflow_profile_capacity)
            )

        if (
            overflow_profile_capacity > 0
            and overflow_active_blocks < overflow_profile_capacity
        ):
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
    radix_overflow_payload = None
    substage_t0 = _now()
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

        materialize_source_particle_payload = (
            source_slots > 0
            and est_payload_mb <= payload_max_mb
            and ((not bool(fused_device_mode)) or bool(fused_payload_enabled))
        )
        if bool(materialize_source_particle_payload):
            source_particle_ids = target_particle_ids[safe_source_leaf_ids]
            source_particle_mask = (
                target_particle_mask[safe_source_leaf_ids]
                & source_leaf_valid_flat[:, :, None]
            )
        else:
            # Fused mode defaults to the smaller source-leaf fallback to keep
            # production memory stable; the source-particle payload can be
            # enabled explicitly for nearfield launch-count A/B tests.
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

        if (
            (not bool(fused_device_mode))
            and overflow_active_blocks > 0
            and target_block_offsets is not None
            and target_block_source_leaf_ids is not None
            and target_block_valid_mask is not None
        ):
            overflow_counts = target_block_offsets[1:] - target_block_offsets[:-1]
            max_overflow_blocks = (
                int(jnp.max(overflow_counts))
                if int(overflow_counts.shape[0]) > 0
                else 0
            )
            if max_overflow_blocks > 0:
                overflow_block_tile = max(1, int(nearfield_target_block_tile_size))
                aligned_overflow_blocks = (
                    (max_overflow_blocks + overflow_block_tile - 1)
                    // overflow_block_tile
                ) * overflow_block_tile
                overflow_source_slots = int(aligned_overflow_blocks) * int(block_size)
                overflow_payload_max_mb_raw = os.environ.get(
                    "JACCPOT_LARGE_N_RADIX_OVERFLOW_PAYLOAD_MAX_MB",
                    "1024",
                )
                try:
                    overflow_payload_max_mb = max(
                        0.0,
                        float(overflow_payload_max_mb_raw),
                    )
                except Exception:
                    overflow_payload_max_mb = 1024.0
                est_overflow_payload_bytes = float(
                    num_target_leaves
                    * max(1, overflow_source_slots)
                    * max(1, source_leaf_size)
                    * (jnp.dtype(INDEX_DTYPE).itemsize + jnp.dtype(bool).itemsize)
                )
                est_overflow_payload_mb = est_overflow_payload_bytes / (1024.0 * 1024.0)
                if overflow_source_slots > 0 and (
                    est_overflow_payload_mb <= overflow_payload_max_mb
                ):
                    overflow_block_offsets = jnp.arange(
                        aligned_overflow_blocks,
                        dtype=INDEX_DTYPE,
                    )
                    overflow_block_idx = (
                        target_block_offsets[:-1, None]
                        + overflow_block_offsets[None, :]
                    )
                    overflow_block_valid = (
                        overflow_block_offsets[None, :] < overflow_counts[:, None]
                    )
                    safe_overflow_block_idx = jnp.where(
                        overflow_block_valid,
                        overflow_block_idx,
                        0,
                    )
                    overflow_source_leaf_ids_padded = jnp.where(
                        overflow_block_valid[:, :, None],
                        target_block_source_leaf_ids[safe_overflow_block_idx],
                        0,
                    )
                    overflow_source_leaf_valid_padded = (
                        target_block_valid_mask[safe_overflow_block_idx]
                        & overflow_block_valid[:, :, None]
                    )
                    overflow_source_leaf_ids_flat = (
                        overflow_source_leaf_ids_padded.reshape(
                            (num_target_leaves, overflow_source_slots)
                        )
                    )
                    overflow_source_leaf_valid_flat = (
                        overflow_source_leaf_valid_padded.reshape(
                            (num_target_leaves, overflow_source_slots)
                        )
                    )
                    safe_overflow_source_leaf_ids = jnp.where(
                        overflow_source_leaf_valid_flat,
                        overflow_source_leaf_ids_flat,
                        0,
                    )
                    overflow_source_particle_ids = target_particle_ids[
                        safe_overflow_source_leaf_ids
                    ]
                    overflow_source_particle_mask = (
                        target_particle_mask[safe_overflow_source_leaf_ids]
                        & overflow_source_leaf_valid_flat[:, :, None]
                    )
                    radix_overflow_payload = RadixFastNearfieldPayload(
                        target_leaf_ids=target_leaf_ids,
                        target_particle_ids=target_particle_ids,
                        target_particle_mask=target_particle_mask,
                        source_leaf_ids=overflow_source_leaf_ids_padded,
                        source_leaf_valid_mask=overflow_source_leaf_valid_padded,
                        source_particle_ids=overflow_source_particle_ids,
                        source_particle_mask=overflow_source_particle_mask,
                        batch_tile_t=int(batch_tile_t),
                        batch_tile_s=int(source_slot_tile),
                        source_slot_scan_unroll=int(source_slot_scan_unroll),
                        target_batch_scan_unroll=int(target_batch_scan_unroll),
                        fallback_block_tile_size=int(fallback_block_tile_size),
                        fallback_tile_scan_unroll=int(source_slot_scan_unroll),
                        fallback_batch_scan_unroll=int(target_batch_scan_unroll),
                    )
    _record_nf("_refresh_timing_nearfield_radix_payload_seconds", substage_t0)

    substage_t0 = _now()
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
        neighbor_profile_capacity = 0
        if static_runtime_fixed_sizing:
            neighbor_profile_capacity = int(neighbor_profile_fixed_cap)
            if neighbor_profile_capacity <= 0:
                if bool(fused_device_mode):
                    neighbor_profile_capacity = int(
                        getattr(fmm, "_large_n_neighbor_edges_profile_cap", 0)
                    )
                    if neighbor_profile_capacity <= 0:
                        neighbor_profile_capacity = int(neighbor_active_edges)
                    setattr(
                        fmm,
                        "_large_n_neighbor_edges_profile_cap",
                        int(neighbor_profile_capacity),
                    )
                else:
                    neighbor_profile_capacity = int(neighbor_active_edges)
        elif not bool(fused_device_mode):
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
                    np.ceil(
                        float(neighbor_active_edges) * float(neighbor_profile_headroom)
                    )
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

        if (
            neighbor_profile_capacity > 0
            and neighbor_active_edges > neighbor_profile_capacity
        ):
            raise RuntimeError(
                "static runtime sizing neighbor-edge cap exceeded: "
                f"active_edges={neighbor_active_edges} cap={neighbor_profile_capacity}. "
                "Increase JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP."
            )

        if (
            neighbor_profile_capacity > 0
            and neighbor_active_edges < neighbor_profile_capacity
        ):
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
        state_target_leaf_ids = precomputed_target_leaf_ids
        state_source_leaf_ids = precomputed_source_leaf_ids
        state_valid_pairs = precomputed_valid_pairs
    _record_nf("_refresh_timing_nearfield_neighbor_padding_seconds", substage_t0)

    substage_t0 = _now()
    out_state = LargeNPreparedState(
        tree=tree_artifacts.tree,
        local_data=dual_downward_artifacts.downward.locals,
        neighbor_list=state_neighbor_list,
        nearfield_leaf_particle_indices=leaf_particle_indices,
        nearfield_leaf_particle_mask=leaf_particle_mask,
        nearfield_target_leaf_ids=state_target_leaf_ids,
        nearfield_source_leaf_ids=state_source_leaf_ids,
        nearfield_valid_pairs=state_valid_pairs,
        nearfield_chunk_sort_indices=precomputed_chunk_sort_indices,
        nearfield_chunk_group_ids=precomputed_chunk_group_ids,
        nearfield_chunk_unique_indices=precomputed_chunk_unique_indices,
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
        local_order=int(dual_downward_artifacts.downward.locals.order),
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
        radix_overflow_payload=radix_overflow_payload,
        compact_far_pairs=getattr(dual_downward_artifacts, "compact_far_pairs", None),
    )
    _record_nf("_refresh_timing_nearfield_state_pack_seconds", substage_t0)
    if refresh_timing_active:
        setattr(
            fmm,
            "_refresh_timing_nearfield_seconds",
            float(getattr(fmm, "_refresh_timing_nearfield_seconds", 0.0))
            + float(_now() - stage_t0),
        )
        setattr(
            fmm,
            "_refresh_timing_nearfield_residual_seconds",
            float(getattr(fmm, "_refresh_timing_nearfield_residual_seconds", 0.0))
            + max(
                0.0,
                float(_now() - nearfield_total_t0) - float(nearfield_stage_sum),
            ),
        )
    if bool(return_compiled_state):
        return large_n_to_compiled_state(out_state)
    return out_state


def evaluate_large_n_state(
    fmm: object,
    state: Union[LargeNPreparedState, LargeNCompiledState],
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

    from .kernels.core import (
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

    state_prepared = large_n_as_prepared_state(state)
    if isinstance(state, LargeNCompiledState):
        max_leaf_size = int(state.max_leaf_size)
        local_order = int(state.local_order)
    else:
        max_leaf_size = int(state_prepared.max_leaf_size)
        local_order = int(
            getattr(
                state_prepared,
                "local_order",
                getattr(state_prepared.local_data, "order", 0),
            )
        )

    leaf_nodes = jnp.asarray(
        state_prepared.neighbor_list.leaf_indices, dtype=INDEX_DTYPE
    )
    node_ranges = jnp.asarray(state_prepared.tree.node_ranges, dtype=INDEX_DTYPE)
    nearfield_mode = str(state_prepared.nearfield_mode).strip().lower()
    if nearfield_mode != "bucketed":
        raise RuntimeError(
            "large_n evaluation requires nearfield_mode='bucketed' prepared state"
        )
    if (not bool(getattr(state_prepared, "radix_fast_lane", False))) and (
        not bool(return_potential)
    ):
        raise RuntimeError(
            "large_n acceleration evaluation requires radix fast-lane state; "
            "prepare state with the large_n_gpu radix profile before accel-only evaluate"
        )
    if (
        bool(getattr(state_prepared, "radix_fast_lane", False))
        and (not bool(return_potential))
        and (getattr(state_prepared, "radix_fast_payload", None) is not None)
    ):
        eval_diag_mode = (
            str(os.environ.get("JACCPOT_LARGE_N_EVAL_DIAG_MODE", "full"))
            .strip()
            .lower()
        )
        if eval_diag_mode not in {
            "full",
            "near_only",
            "far_only",
            "local_only",
            "near_zero",
            "far_zero",
            "permutation_only",
            "zero",
        }:
            eval_diag_mode = "full"
        disable_near_eval = str(
            os.environ.get("JACCPOT_LARGE_N_EVAL_DISABLE_NEAR", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        disable_far_eval = str(
            os.environ.get("JACCPOT_LARGE_N_EVAL_DISABLE_FAR", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if eval_diag_mode in {
            "far_only",
            "local_only",
            "near_zero",
            "permutation_only",
            "zero",
        }:
            disable_near_eval = True
        if eval_diag_mode in {"near_only", "far_zero", "permutation_only", "zero"}:
            disable_far_eval = True
        if jnp.issubdtype(state_prepared.input_dtype, jnp.floating):
            output_dtype = state_prepared.input_dtype
        else:
            output_dtype = state_prepared.working_dtype
        if eval_diag_mode == "zero":
            return jnp.zeros_like(state_prepared.positions_sorted).astype(output_dtype)
        if bool(disable_near_eval):
            near_acc = jnp.zeros_like(state_prepared.positions_sorted)
        else:
            near_acc = evaluate_large_n_nearfield_fast_lane(
                fmm,
                state_prepared,
                return_potential=False,
            )
        if bool(disable_far_eval):
            far_acc = jnp.zeros_like(state_prepared.positions_sorted)
        else:
            far_grad, _, _ = _evaluate_local_expansions_for_particles(
                state_prepared.local_data,
                state_prepared.positions_sorted,
                leaf_nodes=leaf_nodes,
                node_ranges=node_ranges,
                max_leaf_size=max_leaf_size,
                order=local_order,
                expansion_basis="solidfmm",
                return_potential=False,
                max_acc_derivative_order=0,
            )
            far_acc = -float(getattr(fmm, "G")) * far_grad
        if eval_diag_mode == "permutation_only":
            accelerations_sorted = state_prepared.positions_sorted * jnp.asarray(
                0.0,
                dtype=state_prepared.positions_sorted.dtype,
            )
        else:
            accelerations_sorted = near_acc + far_acc
        return jnp.asarray(accelerations_sorted)[
            state_prepared.inverse_permutation
        ].astype(output_dtype)

    nearfield_edge_chunk_size = int(state_prepared.nearfield_edge_chunk_size)
    eval_out = _evaluate_tree_compiled_impl(
        state_prepared.tree,
        state_prepared.positions_sorted,
        state.masses_sorted,
        state_prepared.local_data,
        state_prepared.neighbor_list,
        leaf_nodes,
        node_ranges,
        jnp.asarray(state_prepared.neighbor_list.offsets, dtype=INDEX_DTYPE),
        jnp.asarray(state_prepared.neighbor_list.neighbors, dtype=INDEX_DTYPE),
        jnp.asarray(state_prepared.neighbor_list.counts, dtype=INDEX_DTYPE),
        (
            jnp.asarray(
                state_prepared.nearfield_leaf_particle_indices, dtype=INDEX_DTYPE
            )
            if int(state_prepared.nearfield_leaf_particle_indices.size) > 0
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state_prepared.nearfield_leaf_particle_mask, dtype=bool)
            if int(state_prepared.nearfield_leaf_particle_mask.size) > 0
            else jnp.zeros((0, 0), dtype=bool)
        ),
        leaf_nodes,
        node_ranges,
        (
            jnp.asarray(state_prepared.nearfield_target_leaf_ids, dtype=INDEX_DTYPE)
            if state_prepared.nearfield_target_leaf_ids is not None
            else jnp.zeros((0,), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state_prepared.nearfield_source_leaf_ids, dtype=INDEX_DTYPE)
            if state_prepared.nearfield_source_leaf_ids is not None
            else jnp.zeros((0,), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state_prepared.nearfield_valid_pairs, dtype=bool)
            if state_prepared.nearfield_valid_pairs is not None
            else jnp.zeros((0,), dtype=bool)
        ),
        (
            jnp.asarray(state_prepared.nearfield_chunk_sort_indices, dtype=INDEX_DTYPE)
            if state_prepared.nearfield_chunk_sort_indices is not None
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state_prepared.nearfield_chunk_group_ids, dtype=INDEX_DTYPE)
            if state_prepared.nearfield_chunk_group_ids is not None
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(
                state_prepared.nearfield_chunk_unique_indices, dtype=INDEX_DTYPE
            )
            if state_prepared.nearfield_chunk_unique_indices is not None
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(
                state_prepared.nearfield_target_block_offsets, dtype=INDEX_DTYPE
            )
            if state_prepared.nearfield_target_block_offsets is not None
            else jnp.zeros((leaf_nodes.shape[0] + 1,), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(
                state_prepared.nearfield_target_block_leaf_ids, dtype=INDEX_DTYPE
            )
            if state_prepared.nearfield_target_block_leaf_ids is not None
            else jnp.zeros((0,), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(
                state_prepared.nearfield_target_block_source_leaf_ids, dtype=INDEX_DTYPE
            )
            if state_prepared.nearfield_target_block_source_leaf_ids is not None
            else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(state_prepared.nearfield_target_block_valid_mask, dtype=bool)
            if state_prepared.nearfield_target_block_valid_mask is not None
            else jnp.zeros((0, 0), dtype=bool)
        ),
        (
            jnp.asarray(
                state_prepared.nearfield_target_block_source_leaf_ids_padded,
                dtype=INDEX_DTYPE,
            )
            if state_prepared.nearfield_target_block_source_leaf_ids_padded is not None
            else jnp.zeros((leaf_nodes.shape[0], 0, 0), dtype=INDEX_DTYPE)
        ),
        (
            jnp.asarray(
                state_prepared.nearfield_target_block_valid_mask_padded,
                dtype=bool,
            )
            if state_prepared.nearfield_target_block_valid_mask_padded is not None
            else jnp.zeros((leaf_nodes.shape[0], 0, 0), dtype=bool)
        ),
        G=float(getattr(fmm, "G")),
        softening=float(getattr(fmm, "softening")),
        order=local_order,
        expansion_basis="solidfmm",
        max_leaf_size=max_leaf_size,
        return_potential=bool(return_potential),
        nearfield_mode=nearfield_mode,
        nearfield_edge_chunk_size=nearfield_edge_chunk_size,
        nearfield_delayed_scatter_chunks_per_superchunk=int(
            state_prepared.nearfield_delayed_scatter_chunks_per_superchunk
        ),
        nearfield_chunk_scan_batch_size=int(
            state_prepared.nearfield_chunk_scan_batch_size
        ),
        nearfield_chunk_scan_unroll=int(state_prepared.nearfield_chunk_scan_unroll),
        nearfield_superchunk_scan_unroll=int(
            state_prepared.nearfield_superchunk_scan_unroll
        ),
        nearfield_sorted_scatter_hint=bool(
            state_prepared.nearfield_sorted_scatter_hint
        ),
        nearfield_grouped_sorted_scatter=bool(
            state_prepared.nearfield_grouped_sorted_scatter
        ),
        nearfield_superchunk_target_reduce=bool(
            state_prepared.nearfield_superchunk_target_reduce
        ),
        nearfield_disable_chunk_cond=bool(state_prepared.nearfield_disable_chunk_cond),
        nearfield_target_leaf_batch_size=int(
            state_prepared.nearfield_target_leaf_batch_size
        ),
        nearfield_target_block_tile_size=int(
            state_prepared.nearfield_target_block_tile_size
        ),
        nearfield_target_block_tile_scan_unroll=int(
            state_prepared.nearfield_target_block_tile_scan_unroll
        ),
        nearfield_target_block_batch_scan_unroll=int(
            state_prepared.nearfield_target_block_batch_scan_unroll
        ),
        nearfield_target_block_overflow_fast_max_blocks=int(
            state_prepared.nearfield_target_block_overflow_fast_max_blocks
        ),
        disable_specialized_large_n_nearfield=bool(
            state_prepared.disable_specialized_large_n_nearfield
        ),
    )

    if jnp.issubdtype(state_prepared.input_dtype, jnp.floating):
        output_dtype = state_prepared.input_dtype
    else:
        output_dtype = state_prepared.working_dtype

    if return_potential:
        accelerations_sorted, potentials_sorted = eval_out
    else:
        accelerations_sorted = eval_out

    if not return_potential:
        return jnp.asarray(accelerations_sorted)[
            state_prepared.inverse_permutation
        ].astype(output_dtype)

    accelerations = jnp.asarray(accelerations_sorted)[
        state_prepared.inverse_permutation
    ].astype(output_dtype)
    potentials = jnp.asarray(potentials_sorted)[
        state_prepared.inverse_permutation
    ].astype(output_dtype)
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
    if runtime_path not in ("auto", "large_n"):
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
