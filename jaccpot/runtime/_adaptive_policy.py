"""Solver-owned adaptive traversal policy helpers."""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaccpot.operators.real_harmonics import sh_size
from jaccpot.upward.tree_expansions import TreeUpwardData


class AdaptivePolicyState(NamedTuple):
    """Solver-owned per-node summaries used by adaptive traversal policies."""

    source_error_proxy_by_order: Array
    target_force_scale: Array
    order_tags: Array
    eps: Array


def source_error_proxy_by_order_from_multipoles(
    *,
    multipole_packed: Array,
    p_gears: tuple[int, ...],
) -> Array:
    """Compute a conservative per-node residual proxy for each candidate order."""

    packed = jnp.asarray(multipole_packed)
    if len(p_gears) == 0:
        return jnp.zeros((packed.shape[0], 0), dtype=packed.real.dtype)
    total_p = int(round(np.sqrt(int(packed.shape[1])) - 1))
    magnitudes = jnp.abs(packed)
    tails: list[Array] = []
    for p_gear in p_gears:
        p_clip = int(max(0, min(int(p_gear), total_p)))
        keep = sh_size(p_clip)
        tail = jnp.linalg.norm(magnitudes[:, keep:], axis=1)
        tails.append(tail)
    return jnp.stack(tails, axis=1)


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
) -> AdaptivePolicyState:
    """Build the solver-owned adaptive traversal state from upward data."""

    if len(p_gears) == 0:
        raise ValueError("adaptive policy state requires non-empty p_gears")
    error_proxy = source_error_proxy_by_order_from_multipoles(
        multipole_packed=upward.multipoles.packed,
        p_gears=p_gears,
    )
    if force_scale_nodes is None:
        target_force_scale = jnp.ones((error_proxy.shape[0],), dtype=error_proxy.dtype)
    else:
        target_force_scale = jnp.asarray(force_scale_nodes, dtype=error_proxy.dtype)
        if int(target_force_scale.shape[0]) != int(error_proxy.shape[0]):
            raise ValueError("force_scale_nodes length must match number of nodes")
    order_tags = jnp.arange(len(p_gears), dtype=jnp.int32)
    return AdaptivePolicyState(
        source_error_proxy_by_order=error_proxy,
        target_force_scale=target_force_scale,
        order_tags=order_tags,
        eps=jnp.asarray(eps, dtype=error_proxy.dtype),
    )


__all__ = [
    "AdaptivePolicyState",
    "build_adaptive_policy_state",
    "compute_node_force_scale_from_sorted_acc",
    "source_error_proxy_by_order_from_multipoles",
]
