"""Scattering per-block near-field contributions back onto particles.

The adjoint of the gather that :mod:`jaccpot.nearfield._schedules` sets up: each
kernel produces contributions indexed by ``[leaf, slot]`` or ``[edge, slot]``, and
these helpers add them into the per-particle output.

WHY THERE ARE SEVERAL. They are the same reduction under different assumptions
about the index vector, and the assumption is what buys the speed:
``_scatter_contributions`` is the general unsorted ``.at[].add()``;
``_scatter_contributions_sorted_hint`` and ``_scatter_contributions_grouped_sorted``
tell XLA the indices are sorted or segment-grouped; and the
``*_with_schedule`` pair consumes a precomputed permutation from
``prepare_bucketed_scatter_schedules``. **They must agree numerically**, and the
one thing that could break that is reassociating the adds -- so do not "unify"
them without re-checking the bucketed-vs-baseline parity test.

Split out of ``near_field.py`` (Tier 1.5, A.9 seam 2); every function body is
unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jaxtyping import Array
from yggdrax.dtypes import INDEX_DTYPE, as_index

__all__: list[str] = []


def _scatter_contributions(
    base_acc: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Scatter-add vector contributions into particle-ordered output."""
    if values.size == 0:
        return base_acc
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)
    masked_values = jnp.where(flat_mask[:, None], flat_values, 0.0)
    return base_acc.at[flat_indices].add(masked_values)


def _scatter_contributions_sorted_hint(
    base_acc: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Scatter-add vector contributions assuming index order is nondecreasing."""
    if values.size == 0:
        return base_acc
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)
    masked_values = jnp.where(flat_mask[:, None], flat_values, 0.0)
    return base_acc.at[flat_indices].add(masked_values, indices_are_sorted=True)


def _scatter_contributions_grouped_sorted(
    base_acc: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Scatter-add for sorted indices with grouped reduction before add."""
    if values.size == 0:
        return base_acc
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)
    masked_values = jnp.where(flat_mask[:, None], flat_values, 0.0)

    item_count = flat_indices.shape[0]
    is_new = jnp.concatenate(
        [
            jnp.array([True], dtype=bool),
            flat_indices[1:] != flat_indices[:-1],
        ]
    )
    group_ids = jnp.cumsum(is_new.astype(INDEX_DTYPE)) - as_index(1)
    reduced = jax.ops.segment_sum(masked_values, group_ids, item_count)
    unique_indices = (
        jnp.zeros((item_count,), dtype=INDEX_DTYPE).at[group_ids].set(flat_indices)
    )
    return base_acc.at[unique_indices].add(reduced, indices_are_sorted=True)


def _build_scatter_schedule(
    indices: Array,
    mask: Array,
) -> Tuple[Array, Array, Array]:
    """Build a reusable two-stage scatter schedule for a fixed index/mask layout."""
    flat_indices = indices.reshape(-1)
    flat_mask = mask.reshape(-1)
    safe_indices = jnp.where(flat_mask, flat_indices, 0)

    sort_idx = jnp.argsort(safe_indices)
    idx_sorted = safe_indices[sort_idx]
    item_count = idx_sorted.shape[0]
    is_new = jnp.concatenate(
        [
            jnp.array([True]),
            idx_sorted[1:] != idx_sorted[:-1],
        ]
    )
    group_ids = jnp.cumsum(is_new.astype(INDEX_DTYPE)) - as_index(1)
    unique_indices = (
        jnp.zeros((item_count,), dtype=INDEX_DTYPE).at[group_ids].set(idx_sorted)
    )
    return sort_idx, group_ids, unique_indices


def _scatter_vectors_with_schedule(
    base_acc: Array,
    values: Array,
    mask: Array,
    sort_idx: Array,
    group_ids: Array,
    unique_indices: Array,
) -> Array:
    """Apply a precomputed scatter schedule to vector contributions."""
    if values.size == 0:
        return base_acc
    flat_values = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)
    safe_values = jnp.where(flat_mask[:, None], flat_values, 0.0)
    values_sorted = safe_values[sort_idx]
    item_count = values_sorted.shape[0]
    reduced = jax.ops.segment_sum(values_sorted, group_ids, item_count)
    return base_acc.at[unique_indices].add(reduced)


def _scatter_scalar_contributions(
    base: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Scatter-add scalar contributions into particle-ordered output."""
    if values.size == 0:
        return base
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1)
    flat_mask = mask.reshape(-1)
    masked = jnp.where(flat_mask, flat_values, 0.0)
    return base.at[flat_indices].add(masked)


def _scatter_scalars_with_schedule(
    base: Array,
    values: Array,
    mask: Array,
    sort_idx: Array,
    group_ids: Array,
    unique_indices: Array,
) -> Array:
    """Apply a precomputed scatter schedule to scalar contributions."""
    if values.size == 0:
        return base
    flat_values = values.reshape(-1)
    flat_mask = mask.reshape(-1)
    safe_values = jnp.where(flat_mask, flat_values, 0.0)
    values_sorted = safe_values[sort_idx]
    item_count = values_sorted.shape[0]
    reduced = jax.ops.segment_sum(values_sorted, group_ids, item_count)
    return base.at[unique_indices].add(reduced)


def _reduce_pair_bucket_by_target_leaf(
    target_leaf_ids: Array,
    valid_edge: Array,
    pair_acc: Array,
) -> Tuple[Array, Array, Array]:
    """Collapse target-local bucket rows before direct scatter.

    The minimum-memory large-N path preserves neighbor-list edge order, so
    edges for the same target leaf arrive in contiguous runs. We can exploit
    that to reduce repeated target-leaf updates inside a chunk without
    rebuilding a full per-particle scatter schedule.
    """

    chunk = int(target_leaf_ids.shape[0])
    if chunk == 0:
        empty_leaf = jnp.zeros((0,), dtype=INDEX_DTYPE)
        empty_pair = jnp.zeros_like(pair_acc)
        empty_valid = jnp.zeros((0,), dtype=bool)
        return empty_leaf, empty_pair, empty_valid

    invalid_leaf = jnp.asarray(-1, dtype=INDEX_DTYPE)
    grouped_leaf_ids = jnp.where(valid_edge, target_leaf_ids, invalid_leaf)
    is_new_group = jnp.concatenate(
        [
            jnp.array([True], dtype=bool),
            grouped_leaf_ids[1:] != grouped_leaf_ids[:-1],
        ]
    )
    group_ids = jnp.cumsum(is_new_group.astype(INDEX_DTYPE)) - as_index(1)
    masked_pair_acc = jnp.where(valid_edge[:, None, None], pair_acc, 0.0)
    reduced_pair_acc = jax.ops.segment_sum(
        masked_pair_acc,
        group_ids,
        chunk,
    )
    reduced_target_leaf_ids = (
        jnp.zeros((chunk,), dtype=INDEX_DTYPE)
        .at[group_ids]
        .set(jnp.where(valid_edge, target_leaf_ids, 0))
    )
    reduced_valid = jnp.zeros((chunk,), dtype=bool).at[group_ids].set(valid_edge)
    return reduced_target_leaf_ids, reduced_pair_acc, reduced_valid
