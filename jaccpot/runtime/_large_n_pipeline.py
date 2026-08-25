"""Dedicated prepare/evaluate pipeline for large-N GPU radix solidfmm runs."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Tuple
from jaxtyping import Array
from yggdrax.interactions import DualTreeRetryEvent, DualTreeTraversalConfig, MACType

from jaccpot._jax_compat import Tracer

# `_read_large_n_env_config` is re-imported although only its memoising wrapper
# is called here: it was a module attribute of this module before Tier 1.8, and
# the F16 lesson from `near_field.py` is that a module's attribute surface is
# wider than its import surface. Keeping it costs nothing and cannot break a
# caller that reaches it as `_large_n_pipeline._read_large_n_env_config`.
from ._large_n_env import _large_n_env_config_for_fmm
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
    LargeNExecutionConfig,
    LargeNPreparedState,
    RadixFastNearfieldPayload,
    large_n_as_prepared_state,
    large_n_to_compiled_state,
)
from .dtypes import INDEX_DTYPE

if TYPE_CHECKING:  # pragma: no cover - annotations only, no runtime import
    # The engine lives in `_fmm_impl`, which reaches this module through the
    # mixins it inherits, so this import must stay under TYPE_CHECKING or it
    # would form the cycle ARCHITECTURE section 8 forbids. Unlike `self` on a
    # mixin, `fmm` here is an ordinary parameter, so naming the engine is both
    # valid and what lets the `fmm.<attribute>` reads below be checked at all
    # (audit E.4 bucket E).
    from ._fmm_impl import FMMEngine

__all__ = [
    "can_use_large_n_prepare_path",
    "evaluate_large_n_state",
    "prepare_large_n_state",
]


def _contains_jax_tracer(value: Any) -> bool:
    """Return ``True`` when a pytree contains any JAX tracer leaf.

    Parameters
    ----------
    value : Any
        Pytree to inspect.

    Returns
    -------
    bool
        ``True`` if any leaf is a tracer, which is the gate on every host-side
        decision this module makes.
    """
    return any(isinstance(leaf, Tracer) for leaf in jax.tree_util.tree_leaves(value))


def _build_radix_fast_lane_payloads(
    *,
    execution_config: LargeNExecutionConfig,
    leaf_particle_indices: Array,
    leaf_particle_mask: Array,
    target_block_source_leaf_ids_padded: Optional[Array],
    target_block_valid_mask_padded: Optional[Array],
    target_block_source_leaf_ids: Optional[Array],
    target_block_valid_mask: Optional[Array],
    target_block_offsets: Optional[Array],
    block_size: int,
    overflow_active_blocks: int,
    fused_device_mode: bool,
    fused_payload_enabled: bool,
    nearfield_target_leaf_batch_size: int,
    nearfield_target_block_tile_size: int,
    nearfield_target_block_tile_scan_unroll: int,
    nearfield_target_block_batch_scan_unroll: int,
) -> tuple[Optional[RadixFastNearfieldPayload], Optional[RadixFastNearfieldPayload]]:
    """Build the radix fast-lane near-field payloads, or ``(None, None)``.

    Extracted verbatim from :func:`prepare_large_n_state` (audit item **F11**):
    the body, its guard and every value it computes are unchanged. The guard
    lives inside rather than at the call site so the driver reads as a single
    assignment, and ``(None, None)`` -- the values the driver initialised
    immediately before this block -- is what comes back when the lane does not
    apply.

    Parameters
    ----------
    execution_config : LargeNExecutionConfig
        Resolved large-N execution configuration; ``radix_fast_lane`` gates the
        whole build.
    leaf_particle_indices : Array
        Per-leaf particle indices. An empty array disables the lane.
    leaf_particle_mask : Array
        Validity mask aligned with ``leaf_particle_indices``.
    target_block_source_leaf_ids_padded : Optional[Array]
        Padded source-leaf ids per target block; ``None`` disables the lane.
    target_block_valid_mask_padded : Optional[Array]
        Padded validity mask for those ids; ``None`` disables the lane.
    target_block_source_leaf_ids : Optional[Array]
        Unpadded source-leaf ids, used for the overflow payload.
    target_block_valid_mask : Optional[Array]
        Unpadded validity mask, used for the overflow payload.
    target_block_offsets : Optional[Array]
        CSR offsets over target blocks.
    block_size : int
        Target-owned block size.
    overflow_active_blocks : int
        Active block count for the overflow payload.
    fused_device_mode : bool
        Whether the fused device lane is in force.
    fused_payload_enabled : bool
        Whether the fused payload is enabled for this build.
    nearfield_target_leaf_batch_size : int
        Target-leaf batch size for the fast lane.
    nearfield_target_block_tile_size : int
        Fallback block tile size.
    nearfield_target_block_tile_scan_unroll : int
        Source-slot scan unroll factor.
    nearfield_target_block_batch_scan_unroll : int
        Target-batch scan unroll factor.

    Returns
    -------
    tuple[Optional[RadixFastNearfieldPayload], Optional[RadixFastNearfieldPayload]]
        The fast-lane payload and the overflow payload, either of which may be
        ``None``.
    """
    radix_fast_payload = None
    radix_overflow_payload = None
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
    return radix_fast_payload, radix_overflow_payload


def _apply_speed_prepared_target_block_layout(
    *,
    execution_config: LargeNExecutionConfig,
    fused_device_mode: bool,
    static_target_blocks_used: bool,
    num_leaves: int,
    block_size: int,
    nearfield_target_block_tile_size: int,
    target_leaf_block_counts: Optional[Array],
    target_block_leaf_ids: Optional[Array],
    target_block_source_leaf_ids: Optional[Array],
    target_block_valid_mask: Optional[Array],
    target_block_offsets: Optional[Array],
    target_block_source_leaf_ids_padded: Optional[Array],
    target_block_valid_mask_padded: Optional[Array],
) -> tuple[
    Optional[Array],
    Optional[Array],
    Optional[Array],
    Optional[Array],
    Optional[Array],
    Optional[Array],
]:
    """Apply the speed-prepared target-block layout, or pass the inputs through.

    Extracted verbatim from :func:`prepare_large_n_state` (audit item **F11**);
    the body, its guards and every value it computes are unchanged.

    All six returned arrays are **also parameters**, and that is not redundancy:
    each already holds a value before this block in the driver, and the block
    only conditionally reassigns them. Taking them in and handing them back is
    what makes the guarded path return the caller's own values rather than
    ``None`` -- returning ``None`` on the untaken path would silently drop a
    prepared layout, which is a wrong near field rather than a crash.

    Parameters
    ----------
    execution_config : LargeNExecutionConfig
        Resolved large-N execution configuration; ``speed_prepared_layout``
        gates the whole block.
    fused_device_mode : bool
        Whether the fused device lane is in force; the layout applies only when
        it is not.
    static_target_blocks_used : bool
        Whether the static target-block path already supplied the blocks.
    num_leaves : int
        Leaf count for the CSR offsets.
    block_size : int
        Target-owned block size.
    nearfield_target_block_tile_size : int
        Fallback block tile size.
    target_leaf_block_counts : Optional[Array]
        Per-leaf block counts, or ``None`` when the layout is not prepared.
    target_block_leaf_ids : Optional[Array]
        Leaf id per target block, in and out.
    target_block_source_leaf_ids : Optional[Array]
        Source-leaf ids per target block, in and out.
    target_block_valid_mask : Optional[Array]
        Validity mask over those ids, in and out.
    target_block_offsets : Optional[Array]
        CSR offsets over target blocks, in and out.
    target_block_source_leaf_ids_padded : Optional[Array]
        Padded source-leaf ids, in and out.
    target_block_valid_mask_padded : Optional[Array]
        Padded validity mask, in and out.

    Returns
    -------
    Optional[Array]
        ``target_block_leaf_ids``, reassigned where the layout applied and
        passed through otherwise. The same holds for all five below.
    Optional[Array]
        ``target_block_source_leaf_ids``.
    Optional[Array]
        ``target_block_valid_mask``.
    Optional[Array]
        ``target_block_offsets``.
    Optional[Array]
        ``target_block_source_leaf_ids_padded``.
    Optional[Array]
        ``target_block_valid_mask_padded``.

    Raises
    ------
    RuntimeError
        If the fused payload's static target-block cap is exceeded after
        auto-sizing -- carried over verbatim with the block.
    """
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
    return (
        target_block_leaf_ids,
        target_block_source_leaf_ids,
        target_block_valid_mask,
        target_block_offsets,
        target_block_source_leaf_ids_padded,
        target_block_valid_mask_padded,
    )


def _size_fused_static_overflow_profile(
    *,
    fmm: "FMMEngine",
    fused_device_mode: bool,
    static_runtime_fixed_sizing: bool,
    overflow_active_blocks: int,
    overflow_profile_fixed_cap: int,
    overflow_profile_bootstrap_cap: int,
    overflow_profile_headroom: int,
    block_size: int,
    target_block_leaf_ids: Optional[Array],
    target_block_source_leaf_ids: Optional[Array],
    target_block_valid_mask: Optional[Array],
    pick_overflow_profile_capacity: Callable[[int], int],
) -> tuple[int, Optional[Array], Optional[Array], Optional[Array], int]:
    """Size the fused static overflow profile and pad the target blocks to it.

    Extracted verbatim from :func:`prepare_large_n_state` (audit item **F11**);
    the body, both branches of its guard and every value it computes are
    unchanged, including the cap-exceeded ``RuntimeError``.

    ``pick_overflow_profile_capacity`` is passed in because the original is a
    closure over the driver's ``overflow_profile_caps`` ladder. Handing the
    closure across keeps the ladder and its lookup exactly as they were rather
    than duplicating the search here.

    Parameters
    ----------
    fmm : FMMEngine
        Engine whose ``_large_n_overflow_profile_cap`` carries the bootstrapped
        capacity between builds.
    fused_device_mode : bool
        Whether the fused device lane is in force.
    static_runtime_fixed_sizing : bool
        Whether the static runtime is using fixed sizing; with
        ``fused_device_mode`` this selects the fixed-cap branch.
    overflow_active_blocks : int
        Active overflow blocks this build needs.
    overflow_profile_fixed_cap : int
        Configured fixed cap, or non-positive to derive one.
    overflow_profile_bootstrap_cap : int
        Cap used when bootstrapping a profile.
    overflow_profile_headroom : int
        Headroom added when growing the capacity.
    block_size : int
        Target-owned block size, in and out.
    target_block_leaf_ids : Optional[Array]
        Leaf id per target block, in and out -- padded to the chosen capacity.
    target_block_source_leaf_ids : Optional[Array]
        Source-leaf ids per target block, in and out.
    target_block_valid_mask : Optional[Array]
        Validity mask over those ids, in and out.
    pick_overflow_profile_capacity : Callable[[int], int]
        Maps a required block count onto the next capacity on the ladder.

    Returns
    -------
    int
        ``block_size``, passed through or reassigned.
    Optional[Array]
        ``target_block_leaf_ids``, padded where the profile required it.
    Optional[Array]
        ``target_block_source_leaf_ids``.
    Optional[Array]
        ``target_block_valid_mask``.
    int
        ``overflow_profile_capacity`` -- the sized capacity. Both branches of
        the guard assign it, which is why it needs no prior value.

    Raises
    ------
    RuntimeError
        If the active block count exceeds the fixed overflow profile cap --
        carried over verbatim with the block.
    """
    _pick_overflow_profile_capacity = pick_overflow_profile_capacity
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
    return (
        block_size,
        target_block_leaf_ids,
        target_block_source_leaf_ids,
        target_block_valid_mask,
        overflow_profile_capacity,
    )


def _trim_radix_fast_lane_neighbor_list(
    *,
    execution_config: LargeNExecutionConfig,
    fmm: "FMMEngine",
    fused_device_mode: bool,
    static_runtime_fixed_sizing: bool,
    neighbor_payload: Any,
    state_neighbor_list: Any,
    precomputed_target_leaf_ids: Optional[Array],
    precomputed_source_leaf_ids: Optional[Array],
    precomputed_valid_pairs: Optional[Array],
    neighbor_profile_fixed_cap: int,
    neighbor_profile_bootstrap_cap: int,
    neighbor_profile_headroom: float,
    pick_neighbor_profile_capacity: Callable[[int], int],
) -> tuple[Any, Optional[Array], Optional[Array], Optional[Array]]:
    """Trim the neighbour list for the radix fast lane, or take the other branch.

    Extracted verbatim from :func:`prepare_large_n_state` (audit item **F11**);
    the body, both branches of its guard and every value it computes are
    unchanged.

    The guard is an ``if``/``else`` and **both** branches assign the three
    ``state_*`` outputs, which is why they need no prior value and why the
    extraction covers the whole statement rather than the ``if`` body -- moving
    one branch and leaving the other would bind them on one path only.

    ``pick_neighbor_profile_capacity`` is passed in because the original is a
    closure over the driver's ``neighbor_profile_caps`` ladder; handing it
    across keeps that ladder and its lookup exactly as they were.

    Parameters
    ----------
    execution_config : LargeNExecutionConfig
        Resolved large-N execution configuration; ``radix_fast_lane`` selects
        the trimming branch.
    fmm : FMMEngine
        Engine carrying the bootstrapped neighbour profile capacity between
        builds.
    fused_device_mode : bool
        Whether the fused device lane is in force.
    static_runtime_fixed_sizing : bool
        Whether the static runtime is using fixed sizing.
    neighbor_payload : Any
        Neighbour list built by the dual/downward stage.
    state_neighbor_list : Any
        The list as it stands before trimming, in and out.
    precomputed_target_leaf_ids : Optional[Array]
        Precomputed target-leaf ids, when the near-field precompute supplied
        them.
    precomputed_source_leaf_ids : Optional[Array]
        Precomputed source-leaf ids.
    precomputed_valid_pairs : Optional[Array]
        Precomputed validity mask over those pairs.
    neighbor_profile_fixed_cap : int
        Configured fixed neighbour-edge cap, or non-positive to derive one.
    neighbor_profile_bootstrap_cap : int
        Cap used when bootstrapping a profile.
    neighbor_profile_headroom : float
        Headroom factor applied when growing the capacity.
    pick_neighbor_profile_capacity : Callable[[int], int]
        Maps a required edge count onto the next capacity on the ladder.

    Returns
    -------
    Any
        ``state_neighbor_list``, trimmed on the fast-lane branch and passed
        through otherwise.
    Optional[Array]
        ``state_target_leaf_ids``.
    Optional[Array]
        ``state_source_leaf_ids``.
    Optional[Array]
        ``state_valid_pairs``.

    Raises
    ------
    RuntimeError
        If the active neighbour-edge count exceeds the fixed profile cap --
        carried over verbatim with the block.
    """
    _pick_neighbor_profile_capacity = pick_neighbor_profile_capacity
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
    return (
        state_neighbor_list,
        state_target_leaf_ids,
        state_source_leaf_ids,
        state_valid_pairs,
    )


def prepare_large_n_state(
    fmm: "FMMEngine",
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
    supplied_force_scale: Optional[Array] = None,
    fused_device_mode: bool = False,
    execution_config_override: Optional[Any] = None,
    large_n_env_cfg_override: Optional[dict[str, Any]] = None,
    return_compiled_state: bool = False,
) -> Union[LargeNPreparedState, LargeNCompiledState]:
    """Prepare the slim large-N state using the dedicated narrow path.
    "Narrow" is the point: this path materialises only the artifacts the fast
    lane reads, which is what keeps peak memory tractable at galaxy scale. The
    twenty-seven parameters are the already-resolved runtime configuration --
    the caller (``PrepareMixin``) owns resolution, and nothing is re-resolved
    here.

    Parameters
    ----------
    fmm : FMMEngine
        The engine, typed loosely to avoid the ARCHITECTURE section 8 import
        cycle.
    positions_arr : Array
        Particle positions in the caller's order.
    masses_arr : Array
        Particle masses, same order.
    input_dtype : jnp.dtype
        Dtype the caller supplied, recorded so outputs can be cast back.
    bounds : Optional[Tuple[Array, Array]]
        Explicit domain bounds for the tree build.
    leaf_size : int
        Target particles per leaf.
    max_order : int
        Expansion order.
    theta_val : float
        Resolved MAC opening angle.
    mac_type_val : MACType
        Resolved traversal-facing acceptance criterion.
    refine_local_val : bool
        Resolved local-refinement toggle.
    max_refine_levels_val : int
        Resolved refinement depth cap.
    aspect_threshold_val : float
        Resolved leaf aspect-ratio split threshold.
    jit_tree_override : Optional[bool]
        Force or forbid the jitted tree build.
    allow_stateful_cache : bool
        Permit process-level caches; cleared when the caller needs purity.
    runtime_traversal_config : Optional[DualTreeTraversalConfig]
        Resolved traversal capacities.
    runtime_m2l_chunk_size : Optional[int]
        Resolved M2L chunk size.
    runtime_l2l_chunk_size : Optional[int]
        Resolved L2L chunk size.
    upward_center_mode : str
        Centre selection for the upward sweep.
    record_retry : Callable[[DualTreeRetryEvent], None]
        Sink for traversal retry events.
    collected_retries : list[DualTreeRetryEvent]
        Accumulator the sink appends to; read back by the caller.
    tree_artifacts : Optional[Any]
        Reuse an already-built tree instead of building one.
    dual_downward_artifacts : Optional[Any]
        Reuse an already-built downward plan.
    supplied_force_scale : Optional[Array]
        Per-node force scale for the Dehnen paper MAC. Required by that MAC --
        see ``Raises``.
    fused_device_mode : bool
        Run the fused device-resident refresh rather than the eager path.
    execution_config_override : Optional[Any]
        Pre-resolved large-N execution policy.
    large_n_env_cfg_override : Optional[dict[str, Any]]
        Pre-read environment configuration, for tests.
    return_compiled_state : bool
        Return a ``LargeNCompiledState`` rather than a ``LargeNPreparedState``.

    Returns
    -------
    Union[LargeNPreparedState, LargeNCompiledState]
        The prepared state, compiled variant when ``return_compiled_state``.

    Raises
    ------
    RuntimeError
        If the Dehnen paper MAC is requested with no resolvable per-node force
        scale -- the traversal would silently fall back to a unit scale, i.e. a
        different acceptance threshold -- or if a fused-payload static cap was
        not preflighted or is exceeded, or a compaction invariant fails.
    """

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

    # The Dehnen mass-dependent MAC needs its per-node force scale resolved
    # *between* the tree/upward build and the dual build: the criterion's
    # threshold is `eps * min_b |a_b|` (eq 16a) or `eps * min_b f_b` (eq 16b), and
    # `_prepare_state_dual_and_downward` builds the policy state from whatever
    # `force_scale_nodes` it is handed. Handing it None is not a no-op --
    # `build_adaptive_policy_state` substitutes `jnp.ones(...)`, so the traversal
    # would run a threshold of `eps * 1`: a different criterion, entirely
    # silently. The fused tree+dual helper leaves no seam for the prepass, so a
    # criterion run takes the unfused path.
    criterion_active = bool(
        getattr(fmm, "_uses_paper_style_force_scale", None)
    ) and bool(fmm._uses_paper_style_force_scale())

    if (
        not bool(disable_fused_tree_dual_prepare)
        and not criterion_active
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
        force_scale_nodes = None
        if criterion_active and dual_downward_artifacts is None:
            force_scale_nodes = fmm._resolve_force_scale_nodes_for_prepare(
                tree_artifacts=tree_artifacts,
                supplied_force_scale=supplied_force_scale,
                positions_arr=positions_arr,
                masses_arr=masses_arr,
                bounds=bounds,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                jit_tree=jit_tree_override,
                upward_center_mode=upward_center_mode,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                grouped_interactions=False,
                farfield_mode="pair_grouped",
                record_retry=record_retry,
                refine_local_val=refine_local_val,
                max_refine_levels_val=max_refine_levels_val,
                aspect_threshold_val=aspect_threshold_val,
            )
            if force_scale_nodes is None:
                # Unreachable via the public surface, but the consequence of it
                # ever becoming reachable is a silent unit force scale, so refuse.
                raise RuntimeError(
                    "the large-N lane was asked to run the Dehnen paper MAC but "
                    "no per-node force scale could be resolved; the traversal "
                    "would silently fall back to a unit scale, i.e. a threshold "
                    "of eps*1 rather than eps*min_b|a_b|"
                )
        if dual_downward_artifacts is None:
            dual_downward_artifacts = fmm._prepare_state_dual_and_downward(
                tree_artifacts=tree_artifacts,
                force_scale_nodes=force_scale_nodes,
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
    # Declared here and asserted non-None after the branch chain below, rather
    # than left to the chain to bind. Every path does bind all four -- the last
    # `elif` is unconditional once `static_target_blocks_used` is False, and when
    # it is True the static branch above has already bound them -- but that is a
    # correlation between a flag and a binding, which no type checker can follow
    # (41 "possibly unbound" reports, all false; audit E.4 bucket D).
    #
    # `None`, not zero-sized sentinels, on purpose: a sentinel would turn a
    # broken invariant into a silently wrong force, which is the one failure mode
    # this library must not have. `None` keeps it loud.
    target_block_leaf_ids: Optional[Array] = None
    target_block_source_leaf_ids: Optional[Array] = None
    target_block_valid_mask: Optional[Array] = None
    target_block_offsets: Optional[Array] = None
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
    if (
        target_block_leaf_ids is None
        or target_block_source_leaf_ids is None
        or target_block_valid_mask is None
        or target_block_offsets is None
    ):  # pragma: no cover - unreachable, see the declarations above
        raise RuntimeError(
            "large-N target-block arrays were not bound: the branch chain above "
            "must bind all four on every path. This is an internal invariant, "
            "not a configuration error."
        )
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
    (
        target_block_leaf_ids,
        target_block_source_leaf_ids,
        target_block_valid_mask,
        target_block_offsets,
        target_block_source_leaf_ids_padded,
        target_block_valid_mask_padded,
    ) = _apply_speed_prepared_target_block_layout(
        execution_config=execution_config,
        fused_device_mode=fused_device_mode,
        static_target_blocks_used=static_target_blocks_used,
        num_leaves=num_leaves,
        block_size=block_size,
        nearfield_target_block_tile_size=nearfield_target_block_tile_size,
        target_leaf_block_counts=target_leaf_block_counts,
        target_block_leaf_ids=target_block_leaf_ids,
        target_block_source_leaf_ids=target_block_source_leaf_ids,
        target_block_valid_mask=target_block_valid_mask,
        target_block_offsets=target_block_offsets,
        target_block_source_leaf_ids_padded=target_block_source_leaf_ids_padded,
        target_block_valid_mask_padded=target_block_valid_mask_padded,
    )
    _record_nf("_refresh_timing_nearfield_speed_layout_seconds", substage_t0)

    substage_t0 = _now()
    overflow_active_blocks = int(target_block_source_leaf_ids.shape[0])
    (
        block_size,
        target_block_leaf_ids,
        target_block_source_leaf_ids,
        target_block_valid_mask,
        overflow_profile_capacity,
    ) = _size_fused_static_overflow_profile(
        fmm=fmm,
        fused_device_mode=fused_device_mode,
        static_runtime_fixed_sizing=static_runtime_fixed_sizing,
        overflow_active_blocks=overflow_active_blocks,
        overflow_profile_fixed_cap=overflow_profile_fixed_cap,
        overflow_profile_bootstrap_cap=overflow_profile_bootstrap_cap,
        overflow_profile_headroom=overflow_profile_headroom,
        block_size=block_size,
        target_block_leaf_ids=target_block_leaf_ids,
        target_block_source_leaf_ids=target_block_source_leaf_ids,
        target_block_valid_mask=target_block_valid_mask,
        pick_overflow_profile_capacity=_pick_overflow_profile_capacity,
    )
    _record_nf("_refresh_timing_nearfield_overflow_profile_seconds", substage_t0)

    radix_fast_payload, radix_overflow_payload = _build_radix_fast_lane_payloads(
        execution_config=execution_config,
        leaf_particle_indices=leaf_particle_indices,
        leaf_particle_mask=leaf_particle_mask,
        target_block_source_leaf_ids_padded=target_block_source_leaf_ids_padded,
        target_block_valid_mask_padded=target_block_valid_mask_padded,
        target_block_source_leaf_ids=target_block_source_leaf_ids,
        target_block_valid_mask=target_block_valid_mask,
        target_block_offsets=target_block_offsets,
        block_size=block_size,
        overflow_active_blocks=overflow_active_blocks,
        fused_device_mode=fused_device_mode,
        fused_payload_enabled=fused_payload_enabled,
        nearfield_target_leaf_batch_size=nearfield_target_leaf_batch_size,
        nearfield_target_block_tile_size=nearfield_target_block_tile_size,
        nearfield_target_block_tile_scan_unroll=nearfield_target_block_tile_scan_unroll,
        nearfield_target_block_batch_scan_unroll=nearfield_target_block_batch_scan_unroll,
    )
    _record_nf("_refresh_timing_nearfield_radix_payload_seconds", substage_t0)

    substage_t0 = _now()
    state_neighbor_list = neighbor_payload
    (
        state_neighbor_list,
        state_target_leaf_ids,
        state_source_leaf_ids,
        state_valid_pairs,
    ) = _trim_radix_fast_lane_neighbor_list(
        execution_config=execution_config,
        fmm=fmm,
        fused_device_mode=fused_device_mode,
        static_runtime_fixed_sizing=static_runtime_fixed_sizing,
        neighbor_payload=neighbor_payload,
        state_neighbor_list=state_neighbor_list,
        precomputed_target_leaf_ids=precomputed_target_leaf_ids,
        precomputed_source_leaf_ids=precomputed_source_leaf_ids,
        precomputed_valid_pairs=precomputed_valid_pairs,
        neighbor_profile_fixed_cap=neighbor_profile_fixed_cap,
        neighbor_profile_bootstrap_cap=neighbor_profile_bootstrap_cap,
        neighbor_profile_headroom=neighbor_profile_headroom,
        pick_neighbor_profile_capacity=_pick_neighbor_profile_capacity,
    )
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


def _large_n_fastlane_eval_fn(
    fmm: Any,
    state: Any,
    *,
    body: Callable[[Any], Array],
    cache_key: tuple[Any, ...],
) -> Optional[Callable[[Any], Array]]:
    """Return a cached ``jax.jit`` of ``body``, or None to run it eagerly.

    The radix fast-lane acceleration evaluate was dispatched **op by op**: the
    index gathers, the fused Pallas near-field call, the L2P expansion gradient
    and the scatter-add each went to the device as their own executable, with
    nothing able to fuse or overlap across the boundaries. That is the fixed cost
    behind the paper's figure 04 -- measured on an idle A100, ``large_n_gpu``,
    leaf 64, p=4, real basis, evaluating a prebuilt tree:

    =======  =========  ========  =======
    N        op-by-op   compiled  factor
    =======  =========  ========  =======
    4096      187.2 ms   13.9 ms   13.5x
    16384     189.0 ms   16.7 ms   11.3x
    =======  =========  ========  =======

    ...and the output is **bit-identical** (max abs diff 0.0): compiling the same
    graph does not reassociate it. The near field alone accounted for 185 ms of
    the 187 (measured via JACCPOT_LARGE_N_EVAL_DIAG_MODE: zero 1.0 ms,
    permutation_only 2.5 ms, far_only 1.7 ms, near_only 185.2 ms), and it barely
    moved from N=2048 to N=65536 -- a constant, which is why figure 04's fit gave
    alpha=0.20 at R^2=0.89 over three decades.

    Returns None -- meaning "run eagerly" -- when compiling would be wrong or
    pointless:

    * the state already contains tracers, i.e. an outer ``jax.jit``/``grad`` is
      in charge and nesting a second one would only re-trace;
    * ``JACCPOT_LARGE_N_EVAL_JIT=0``, the escape hatch for A/B measurement and
      for bisecting a suspected compile problem.

    The compile is paid once per (state structure, shapes, tuning) and cached on
    the solver, so a refresh loop reusing topology recompiles nothing. ``jax.jit``
    keys on the pytree structure and leaf avals by itself; ``cache_key`` adds the
    Python-level constants the body closes over, which jit cannot see.

    Parameters
    ----------
    fmm : Any
        The engine, whose per-solver cache the compiled function is stored on.
    state : Any
        Prepared state; its pytree structure participates in the jit key.
    body : Callable[[Any], Array]
        The evaluation to compile.
    cache_key : tuple[Any, ...]
        The Python-level constants ``body`` closes over. These are invisible to
        ``jax.jit``, so omitting one would silently reuse a stale executable.

    Returns
    -------
    Optional[Callable[[Any], Array]]
        The cached compiled callable, or ``None`` to signal the caller should
        run ``body`` eagerly -- which is the escape hatch for measuring the
        compile's benefit and for bisecting a suspected compile problem.
    """

    if not str(os.environ.get("JACCPOT_LARGE_N_EVAL_JIT", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    if _contains_jax_tracer(state):
        return None

    cache = getattr(fmm, "_large_n_fastlane_eval_jit_cache", None)
    if cache is None:
        cache = {}
        try:
            setattr(fmm, "_large_n_fastlane_eval_jit_cache", cache)
        except Exception:  # pragma: no cover - defensive, fmm is a normal object
            return None
    cached = cache.get(cache_key)
    if cached is None:
        cached = jax.jit(body)
        cache[cache_key] = cached
    return cached


def evaluate_large_n_state(
    fmm: "FMMEngine",
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

    Parameters
    ----------
    fmm : FMMEngine
        The engine, typed loosely to avoid the import cycle.
    state : Union[LargeNPreparedState, LargeNCompiledState]
        Prepared or compiled large-N state.
    target_indices : Optional[Array]
        Must be ``None`` -- this lane evaluates the full particle set only.
    return_potential : bool
        Also return potentials, which routes to the compiled generic evaluator
        rather than the fast lane.
    max_acc_derivative_order : int
        Must be ``0``; acceleration derivatives are not wired on this lane.

    Returns
    -------
    Any
        Accelerations, or ``(accelerations, potentials)`` when
        ``return_potential`` is set.

    Raises
    ------
    NotImplementedError
        If ``target_indices`` is given, or ``max_acc_derivative_order`` is
        non-zero. Both are unimplemented rather than invalid, so they raise
        loudly instead of silently ignoring the request.
    RuntimeError
        If the state was not prepared with ``nearfield_mode='bucketed'``, or
        carries no radix fast-lane payload. A wiring fault, not user input.
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

        def _fastlane_body(state_in: Any) -> Array:
            if bool(disable_near_eval):
                near_acc = jnp.zeros_like(state_in.positions_sorted)
            else:
                near_acc = evaluate_large_n_nearfield_fast_lane(
                    fmm,
                    state_in,
                    return_potential=False,
                )
            if bool(disable_far_eval):
                far_acc = jnp.zeros_like(state_in.positions_sorted)
            else:
                far_grad, _, _ = _evaluate_local_expansions_for_particles(
                    state_in.local_data,
                    state_in.positions_sorted,
                    leaf_nodes=jnp.asarray(
                        state_in.neighbor_list.leaf_indices, dtype=INDEX_DTYPE
                    ),
                    node_ranges=jnp.asarray(
                        state_in.tree.node_ranges, dtype=INDEX_DTYPE
                    ),
                    max_leaf_size=max_leaf_size,
                    order=local_order,
                    expansion_basis="solidfmm",
                    return_potential=False,
                    max_acc_derivative_order=0,
                )
                far_acc = -float(getattr(fmm, "G")) * far_grad
            if eval_diag_mode == "permutation_only":
                accelerations_sorted = state_in.positions_sorted * jnp.asarray(
                    0.0,
                    dtype=state_in.positions_sorted.dtype,
                )
            else:
                accelerations_sorted = near_acc + far_acc
            return jnp.asarray(accelerations_sorted)[
                state_in.inverse_permutation
            ].astype(output_dtype)

        compiled = _large_n_fastlane_eval_fn(
            fmm,
            state_prepared,
            body=_fastlane_body,
            cache_key=(
                "fastlane_accel",
                str(eval_diag_mode),
                bool(disable_near_eval),
                bool(disable_far_eval),
                int(max_leaf_size),
                int(local_order),
                jnp.dtype(output_dtype).name,
                float(getattr(fmm, "G")),
                float(getattr(fmm, "softening")),
                bool(getattr(fmm, "use_pallas", False)),
            ),
        )
        if compiled is not None:
            return compiled(state_prepared)
        return _fastlane_body(state_prepared)

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

    # The early return comes before the unpack so the two-value case is bound and
    # used in one place; splitting them left `potentials_sorted` bound under one
    # `return_potential` test and read under another, which reads as possibly
    # unbound even though it never is.
    if not return_potential:
        accelerations_sorted = eval_out
        return jnp.asarray(accelerations_sorted)[
            state_prepared.inverse_permutation
        ].astype(output_dtype)

    accelerations_sorted, potentials_sorted = eval_out
    accelerations = jnp.asarray(accelerations_sorted)[
        state_prepared.inverse_permutation
    ].astype(output_dtype)
    potentials = jnp.asarray(potentials_sorted)[
        state_prepared.inverse_permutation
    ].astype(output_dtype)
    return accelerations, potentials


def _record_large_n_decline(fmm: Any, reason: str) -> None:
    """Record why the large-N prepare path declined, for runtime diagnostics.

    The lane is selected silently, so a caller that configures a feature the lane
    does not support sees only an unexplained slowdown. Surfacing the reason makes
    "my large-N run is slow" answerable without bisecting the selection predicate.

    Parameters
    ----------
    fmm : Any
        The engine the reason is recorded on.
    reason : str
        Why the lane declined, phrased for the user who will read it back.

    Returns
    -------
    None
        Records the reason on ``fmm`` in place.
    """

    try:
        fmm._large_n_path_declined_reason = str(reason)
    except AttributeError:  # pragma: no cover - defensive, fmm is always a solver
        pass


def can_use_large_n_prepare_path(
    fmm: "FMMEngine",
    *,
    positions_arr: Array,
    masses_arr: Array,
    allow_stateful_cache: bool,
) -> bool:
    """Decide whether prepare_state should dispatch to the large-N path.
    Declining is silent by design -- the caller just uses the general path --
    which is why every rejection also goes through
    :func:`_record_large_n_decline`.

    Parameters
    ----------
    fmm : FMMEngine
        The engine whose configuration is being tested.
    positions_arr : Array
        Particle positions; consulted for count and concreteness.
    masses_arr : Array
        Particle masses, likewise.
    allow_stateful_cache : bool
        Whether process-level caches may be used.

    Returns
    -------
    bool
        ``True`` when the large-N path should be dispatched.

    Raises
    ------
    RuntimeError
        If ``runtime_path='large_n'`` was requested **explicitly** but the lane
        cannot honour the configuration. An explicit request is the one case
        where declining silently would be wrong, so it raises instead.
    """

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
        bool(getattr(fmm, "_uses_per_node_effective_theta", None))
        and fmm._uses_per_node_effective_theta()
    ):
        # `dehnen_theta` folds the criterion into `geometry.radius` *before* the
        # dual build, via `_apply_per_node_effective_theta`. This lane has no such
        # step, so it would run the plain geometric MAC at the solver's theta --
        # and paper mode pins that at 1.0, so acceptance would be wildly loose
        # rather than merely different. The mode is refuted anyway (see
        # `_uses_per_node_effective_theta`), so decline rather than plumb it.
        _record_large_n_decline(fmm, "per_node_effective_theta")
        if runtime_path == "large_n":
            raise RuntimeError(
                "runtime_path='large_n' was requested explicitly, but the large-N "
                "prepare path cannot run mac_type='dehnen_theta': the criterion is "
                "folded into per-node opening angles before the dual build, which "
                "this lane does not do, so it would silently run the geometric MAC "
                "at theta=1.0. Use mac_type='dehnen_error' (which this lane does "
                "carry) or mac_type='dehnen'."
            )
        return False
    if int(positions_arr.shape[0]) != int(masses_arr.shape[0]):
        return False
    return True
