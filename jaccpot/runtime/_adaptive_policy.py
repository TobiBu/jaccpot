"""Solver-owned adaptive traversal policy helpers."""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaccpot.operators.real_harmonics import sh_size
from jaccpot.upward.tree_expansions import TreeUpwardData

_ACTION_ACCEPT = 0
_ACTION_NEAR = 1
_ACTION_REFINE = 2

_ERROR_MODEL_TAIL_PROXY = 0
_ERROR_MODEL_DEHNEN_DEGREE = 1


class AdaptivePolicyState(NamedTuple):
    """Solver-owned per-node summaries used by adaptive traversal policies."""

    source_error_proxy_by_order: Array
    source_degree_power: Array
    target_accept_threshold: Array
    order_tags: Array
    order_values: Array
    relaxed_theta_sq: Array
    error_model_code: Array


def adaptive_policy_tolerance(
    *, theta: float, p_gears: tuple[int, ...], dtype: object
) -> Array:
    """Return a conservative solver-side adaptive tolerance derived from ``theta``."""

    if len(p_gears) == 0:
        raise ValueError("adaptive policy tolerance requires non-empty p_gears")
    return jnp.asarray(float(theta) ** (max(int(v) for v in p_gears) + 2), dtype=dtype)


def source_power_by_degree_from_multipoles(*, multipole_packed: Array) -> Array:
    """Return per-node multipole power grouped by spherical-harmonic degree.

    The packed coefficient layout stores all orders up to ``p`` in cumulative
    ``sh_size(ell)`` blocks. This helper aggregates the squared coefficient
    magnitudes degree-by-degree, which is closer to Dehnen's use of per-order
    source power summaries than a single flat tail norm.
    """

    packed = jnp.asarray(multipole_packed)
    total_p = int(round(np.sqrt(int(packed.shape[1])) - 1))
    magnitudes_sq = jnp.square(jnp.abs(packed))
    powers: list[Array] = []
    start = 0
    for ell in range(total_p + 1):
        stop = sh_size(ell)
        powers.append(jnp.sum(magnitudes_sq[:, start:stop], axis=1))
        start = stop
    return jnp.stack(powers, axis=1)


def source_error_proxy_by_order_from_degree_power(
    *,
    degree_power: Array,
    p_gears: tuple[int, ...],
) -> Array:
    """Return the residual tail proxy for each candidate order from degree power."""

    power = jnp.asarray(degree_power)
    if len(p_gears) == 0:
        return jnp.zeros((power.shape[0], 0), dtype=power.dtype)
    total_p = int(power.shape[1] - 1)
    tails: list[Array] = []
    for p_gear in p_gears:
        p_clip = int(max(0, min(int(p_gear), total_p)))
        tail_power = jnp.sum(power[:, p_clip + 1 :], axis=1)
        tails.append(jnp.sqrt(tail_power))
    return jnp.stack(tails, axis=1)


def dehnen_like_pair_error_by_order_from_degree_power(
    *,
    degree_power: Array,
    opening: Array,
    order_values: Array,
) -> Array:
    """Return a Dehnen-style degree-weighted pair error estimate by order.

    The source multipole power is retained degree-by-degree and weighted by an
    interaction-specific opening factor. For each candidate order ``p``, the
    residual estimate sums only degrees ``ell > p``.
    """

    power = jnp.asarray(degree_power)
    opening_arr = jnp.asarray(opening, dtype=power.dtype)
    if opening_arr.ndim == 0:
        opening_arr = opening_arr[None]
    opening_arr = jnp.clip(opening_arr, 0.0, 1.0)
    order_arr = jnp.asarray(order_values, dtype=jnp.int32)
    if order_arr.ndim == 0:
        order_arr = order_arr[None]
    if int(power.shape[0]) != int(opening_arr.shape[0]):
        raise ValueError(
            "degree_power and opening must have matching leading dimensions"
        )
    if int(order_arr.shape[0]) == 0:
        return jnp.zeros((opening_arr.shape[0], 0), dtype=power.dtype)
    total_p = int(power.shape[1] - 1)
    degree_idx = jnp.arange(total_p + 1, dtype=jnp.int32)
    opening_weights = jnp.power(
        opening_arr[:, None],
        degree_idx[None, :].astype(power.dtype) + 2.0,
    )
    weighted_power = power * opening_weights
    include_mask = degree_idx[None, None, :] > order_arr[None, :, None]
    tail_power = jnp.sum(
        weighted_power[:, None, :] * include_mask.astype(power.dtype), axis=2
    )
    return jnp.sqrt(jnp.maximum(tail_power, jnp.asarray(0.0, dtype=power.dtype)))


def source_error_proxy_by_order_from_multipoles(
    *,
    multipole_packed: Array,
    p_gears: tuple[int, ...],
) -> Array:
    """Compute a conservative per-node residual proxy for each candidate order."""

    degree_power = source_power_by_degree_from_multipoles(
        multipole_packed=multipole_packed,
    )
    return source_error_proxy_by_order_from_degree_power(
        degree_power=degree_power,
        p_gears=p_gears,
    )


def compute_node_force_scale_from_sorted_acc(
    *,
    node_ranges: Array,
    accelerations_sorted: Array,
) -> Array:
    """Estimate per-node force scales from sorted per-particle accelerations."""

    node_ranges_np = np.asarray(node_ranges, dtype=np.int64)
    acc_sorted_np = np.asarray(accelerations_sorted)
    mag = np.linalg.norm(acc_sorted_np, axis=1)
    force_scale = np.zeros((node_ranges_np.shape[0],), dtype=mag.dtype)
    for idx, (start, end) in enumerate(node_ranges_np):
        s = int(start)
        e = int(end)
        if e < s:
            continue
        force_scale[idx] = float(np.max(mag[s : e + 1]))
    return jnp.asarray(force_scale, dtype=accelerations_sorted.dtype)


def build_adaptive_policy_state(
    *,
    upward: TreeUpwardData,
    p_gears: tuple[int, ...],
    force_scale_nodes: Optional[Array],
    eps: Array,
    theta: Array,
    error_model_code: Array,
) -> AdaptivePolicyState:
    """Build the solver-owned adaptive traversal state from upward data."""

    if len(p_gears) == 0:
        raise ValueError("adaptive policy state requires non-empty p_gears")
    degree_power = source_power_by_degree_from_multipoles(
        multipole_packed=upward.multipoles.packed,
    )
    error_proxy = source_error_proxy_by_order_from_degree_power(
        degree_power=degree_power,
        p_gears=p_gears,
    )
    if force_scale_nodes is None:
        target_force_scale = jnp.ones((error_proxy.shape[0],), dtype=error_proxy.dtype)
    else:
        target_force_scale = jnp.asarray(force_scale_nodes, dtype=error_proxy.dtype)
        if int(target_force_scale.shape[0]) != int(error_proxy.shape[0]):
            raise ValueError("force_scale_nodes length must match number of nodes")
        scale_norm = jnp.maximum(
            jnp.max(target_force_scale),
            jnp.asarray(1.0, dtype=error_proxy.dtype),
        )
        target_force_scale = target_force_scale / scale_norm
    order_tags = jnp.arange(len(p_gears), dtype=jnp.int32)
    order_values = jnp.asarray(tuple(int(v) for v in p_gears), dtype=jnp.int32)
    target_accept_threshold = (
        jnp.asarray(eps, dtype=error_proxy.dtype) * target_force_scale
    )
    theta_arr = jnp.asarray(theta, dtype=error_proxy.dtype)
    relaxed_theta = jnp.minimum(
        jnp.asarray(1.0, dtype=error_proxy.dtype),
        jnp.asarray(1.5, dtype=error_proxy.dtype) * theta_arr,
    )
    return AdaptivePolicyState(
        source_error_proxy_by_order=error_proxy,
        source_degree_power=degree_power,
        target_accept_threshold=target_accept_threshold,
        order_tags=order_tags,
        order_values=order_values,
        relaxed_theta_sq=jnp.square(relaxed_theta),
        error_model_code=jnp.asarray(error_model_code, dtype=jnp.int32),
    )


def _compute_passes_for_error_model(
    *,
    policy_state: AdaptivePolicyState,
    source_proxy: Array,
    source_degree_power: Array,
    target_threshold: Array,
    opening: Array,
    extent_sum_sq: Array,
    safe_dist_sq: Array,
) -> Array:
    """Return per-order pass decisions for the configured adaptive error model."""

    def _tail_proxy(_: None) -> Array:
        return (
            jnp.square(source_proxy) * extent_sum_sq[:, None]
            < jnp.square(target_threshold)[:, None] * safe_dist_sq[:, None]
        )

    def _dehnen_degree(_: None) -> Array:
        pair_error = dehnen_like_pair_error_by_order_from_degree_power(
            degree_power=source_degree_power,
            opening=opening,
            order_values=policy_state.order_values,
        )
        return pair_error < target_threshold[:, None]

    return jax.lax.cond(
        jnp.asarray(policy_state.error_model_code, dtype=jnp.int32)
        == _ERROR_MODEL_DEHNEN_DEGREE,
        _dehnen_degree,
        _tail_proxy,
        operand=None,
    )


def adaptive_pair_policy(
    policy_state: AdaptivePolicyState, **pair_data: Array
) -> tuple[Array, Array]:
    """Return traversal actions and order tags from solver-owned adaptive state."""

    valid_pairs = jnp.asarray(pair_data["valid_pairs"], dtype=jnp.bool_)
    mac_ok = jnp.asarray(pair_data["mac_ok"], dtype=jnp.bool_)
    different_nodes = jnp.asarray(pair_data["different_nodes"], dtype=jnp.bool_)
    target_leaf = jnp.asarray(pair_data["target_leaf"], dtype=jnp.bool_)
    source_leaf = jnp.asarray(pair_data["source_leaf"], dtype=jnp.bool_)
    target_nodes = jnp.asarray(pair_data["target_nodes"], dtype=jnp.int32)
    source_nodes = jnp.asarray(pair_data["source_nodes"], dtype=jnp.int32)
    dist_sq = jnp.asarray(pair_data["dist_sq"])
    extent_target = jnp.asarray(pair_data["extent_target"], dtype=dist_sq.dtype)
    extent_source = jnp.asarray(pair_data["extent_source"], dtype=dist_sq.dtype)

    safe_targets = jnp.where(valid_pairs, target_nodes, 0)
    safe_sources = jnp.where(valid_pairs, source_nodes, 0)
    safe_dist_sq = jnp.maximum(dist_sq, jnp.asarray(1e-24, dtype=dist_sq.dtype))
    extent_sum_sq = jnp.square(extent_target + extent_source)

    source_proxy = jnp.asarray(policy_state.source_error_proxy_by_order)[
        safe_sources, :
    ]
    source_degree_power = jnp.asarray(policy_state.source_degree_power)[safe_sources, :]
    target_threshold = jnp.asarray(policy_state.target_accept_threshold)[safe_targets]
    opening = jnp.sqrt(extent_sum_sq / safe_dist_sq)
    passes = _compute_passes_for_error_model(
        policy_state=policy_state,
        source_proxy=source_proxy,
        source_degree_power=source_degree_power,
        target_threshold=target_threshold,
        opening=opening,
        extent_sum_sq=extent_sum_sq,
        safe_dist_sq=safe_dist_sq,
    )
    pass_any = jnp.any(passes, axis=1)
    highest_order_pass = passes[:, -1]
    allow_solver_override = (~target_leaf) | (~source_leaf)
    relaxed_mac_ok = extent_sum_sq <= policy_state.relaxed_theta_sq * safe_dist_sq

    order_tags = jnp.asarray(policy_state.order_tags, dtype=jnp.int32)
    required_idx = jnp.argmax(passes.astype(jnp.int32), axis=1).astype(jnp.int32)
    raw_tags = order_tags[required_idx]
    del mac_ok
    accept_gate = highest_order_pass & allow_solver_override & relaxed_mac_ok
    accept_mask = valid_pairs & different_nodes & accept_gate & pass_any
    tags = jnp.where(accept_mask, raw_tags, -jnp.ones_like(raw_tags))

    actions = jnp.full(valid_pairs.shape, _ACTION_REFINE, dtype=jnp.int32)
    actions = jnp.where(accept_mask, _ACTION_ACCEPT, actions)
    near_mask = (
        valid_pairs & different_nodes & target_leaf & source_leaf & (~accept_mask)
    )
    actions = jnp.where(near_mask, _ACTION_NEAR, actions)
    return actions, tags


def bucket_far_pairs_by_tag(
    interaction_sources: Array,
    interaction_targets: Array,
    interaction_tags: Array,
    num_tags: int,
) -> tuple[tuple[Array, Array], ...]:
    """Group accepted far pairs by integer tag."""

    buckets: list[tuple[Array, Array]] = []
    src = jnp.asarray(interaction_sources)
    tgt = jnp.asarray(interaction_targets)
    tags = jnp.asarray(interaction_tags)
    for idx in range(int(num_tags)):
        mask = tags == idx
        buckets.append((src[mask], tgt[mask]))
    return tuple(buckets)


__all__ = [
    "AdaptivePolicyState",
    "adaptive_pair_policy",
    "bucket_far_pairs_by_tag",
    "build_adaptive_policy_state",
    "compute_node_force_scale_from_sorted_acc",
    "dehnen_like_pair_error_by_order_from_degree_power",
    "source_error_proxy_by_order_from_degree_power",
    "source_error_proxy_by_order_from_multipoles",
    "source_power_by_degree_from_multipoles",
]
