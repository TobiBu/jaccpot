"""Per-leaf treecode walk (pure-JAX reference + sm_75/CPU fallback).

Part of the launch-reduction effort (see benchmark_a100/WALK_SPEC.md): the current
dual-tree walk is a host-iterated ``while_loop`` (the launch storm). A per-leaf
treecode — each target leaf independently descends the source tree with a PRIVATE
stack, accepting well-separated source nodes (M2L into that leaf's local expansion)
and marking near source leaves (P2P) — is the structure that ports to a single
Pallas kernel (one program per leaf, à la Bonsai's dev_approximate_gravity).

This module is the PURE-JAX version: correctness reference + the fallback the Pallas
dual-path needs on non-Ampere / CPU. It is intentionally launch-bound (the vmapped
``while_loop`` is host-iterated); perf comes from the Pallas port, not from here.

Correctness contract (validated separately vs direct N-body): for each target leaf,
the accepted far source nodes' particle ranges UNION the near source leaves' particle
ranges must PARTITION all source particles exactly once (treecode completeness).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

_SENTINEL = -1


class TreecodeLeafLists(NamedTuple):
    """Per-leaf treecode output (fixed-capacity, ``-1``-padded).

    far_nodes: (num_leaves, max_far) accepted well-separated source node ids.
    near_leaves: (num_leaves, max_near) near source leaf node ids.
    far_count / near_count: (num_leaves,) valid prefix length per leaf.
    overflow: () bool, any leaf exceeded a capacity.
    """

    far_nodes: jax.Array
    near_leaves: jax.Array
    far_count: jax.Array
    near_count: jax.Array
    overflow: jax.Array


def _mac_ok(center_t, ext_t, center_s, ext_s, theta_sq):
    """Barnes-Hut MAC: (r_t + r_s)^2 <= theta^2 * d^2, guarded for self/degenerate.

    ``ext_*`` are the per-node MAC extents supplied by the caller. Their MEANING is the
    caller's choice and it matters for multi-step stability: a box half-width
    (``max_extent``, "bh") UNDER-bounds the true source multipole radius and, in a
    frozen-topology integration, lets this test over-accept far pairs as leaves spread ->
    accumulating multipole error -> the run heats/blows up. Use bounding-SPHERE radii
    (dehnen) for stable dynamics. See ``_interaction_cache._treecode_mac_extents`` and
    docs/treecode_mac_stability.md.
    """
    delta = center_t - center_s
    dist_sq = jnp.sum(delta * delta)
    rsum = ext_t + ext_s
    ok = (rsum * rsum) <= (theta_sq * dist_sq)
    return jnp.logical_and(ok, dist_sq > 0.0)


def _single_leaf_walk(
    leaf_node,
    centers,
    mac_extents,
    left_child_full,
    right_child_full,
    theta_sq,
    root_idx,
    num_internal,
    max_far,
    max_near,
    max_stack,
    max_iters,
):
    """Treecode descent for ONE target leaf. Private fixed-capacity stack.

    Uses a FIXED-iteration ``fori_loop`` with masked pops (each node is visited at
    most once, so ``max_iters = total_nodes`` suffices). This vmaps cleanly (no
    scalar-cond issue) and mirrors the Pallas program-per-leaf body: no ``lax.cond``,
    all effects are masked scatters with ``mode="drop"`` for the inactive/overflow case.

    Returns (far_nodes[max_far], near_leaves[max_near], far_count, near_count, overflow).
    """
    idx_dtype = left_child_full.dtype
    center_t = centers[leaf_node]
    ext_t = mac_extents[leaf_node]

    stack = jnp.full((max_stack,), _SENTINEL, dtype=idx_dtype)
    stack = stack.at[0].set(root_idx.astype(idx_dtype))
    sp = jnp.int32(1)  # stack pointer (number of valid entries)
    far = jnp.full((max_far,), _SENTINEL, dtype=idx_dtype)
    near = jnp.full((max_near,), _SENTINEL, dtype=idx_dtype)
    fc = jnp.int32(0)
    nc = jnp.int32(0)
    overflow = jnp.bool_(False)

    def body(
        _: jax.Array,
        state: tuple[
            jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
        ],
    ) -> tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
    ]:
        stack, sp, far, near, fc, nc, overflow = state
        active = sp > 0
        sp_pop = jnp.where(active, sp - 1, 0)
        s = stack[sp_pop]  # safe read (index 0 when inactive)
        center_s = centers[s]
        ext_s = mac_extents[s]
        is_leaf = s >= num_internal
        well_sep = _mac_ok(center_t, ext_t, center_s, ext_s, theta_sq)

        do_far = jnp.logical_and(active, well_sep)
        do_near = jnp.logical_and(
            active, jnp.logical_and(is_leaf, jnp.logical_not(well_sep))
        )
        do_refine = jnp.logical_and(
            active, jnp.logical_and(jnp.logical_not(is_leaf), jnp.logical_not(well_sep))
        )

        # masked writes: out-of-range index -> dropped
        far = far.at[jnp.where(do_far, fc, max_far)].set(s, mode="drop")
        near = near.at[jnp.where(do_near, nc, max_near)].set(s, mode="drop")
        lc = left_child_full[s]
        rc = right_child_full[s]
        stack = stack.at[jnp.where(do_refine, sp_pop, max_stack)].set(lc, mode="drop")
        stack = stack.at[jnp.where(do_refine, sp_pop + 1, max_stack)].set(
            rc, mode="drop"
        )

        overflow = jnp.logical_or(
            overflow,
            jnp.logical_or(
                jnp.logical_and(do_far, fc >= max_far),
                jnp.logical_or(
                    jnp.logical_and(do_near, nc >= max_near),
                    jnp.logical_and(do_refine, sp_pop + 2 > max_stack),
                ),
            ),
        )
        fc = fc + do_far.astype(jnp.int32)
        nc = nc + do_near.astype(jnp.int32)
        # sp: refine -> sp_pop+2 ; accept/near -> sp_pop ; inactive -> unchanged
        sp = jnp.where(active, jnp.where(do_refine, sp_pop + 2, sp_pop), sp)
        return stack, sp, far, near, fc, nc, overflow

    state = (stack, sp, far, near, fc, nc, overflow)
    stack, sp, far, near, fc, nc, overflow = lax.fori_loop(0, max_iters, body, state)
    return far, near, fc, nc, overflow


@partial(
    jax.jit,
    static_argnames=("num_internal", "max_far", "max_near", "max_stack", "max_iters"),
)
def treecode_leaf_walk(
    leaf_nodes: jax.Array,
    centers: jax.Array,
    mac_extents: jax.Array,
    left_child_full: jax.Array,
    right_child_full: jax.Array,
    theta_sq: jax.Array,
    root_idx: jax.Array,
    *,
    num_internal: int,
    max_far: int,
    max_near: int,
    max_stack: int,
    max_iters: int,
) -> TreecodeLeafLists:
    """Vectorized per-leaf treecode walk (vmap over the single-leaf kernel body).

    ``max_iters`` bounds the descent (== total node count is always sufficient: each
    node is popped at most once). ``max_stack`` bounds the private stack depth.
    """
    walk = lambda ln: _single_leaf_walk(
        ln,
        centers,
        mac_extents,
        left_child_full,
        right_child_full,
        theta_sq,
        root_idx,
        num_internal,
        max_far,
        max_near,
        max_stack,
        max_iters,
    )
    far, near, fc, nc, overflow = jax.vmap(walk)(leaf_nodes)
    return TreecodeLeafLists(
        far_nodes=far,
        near_leaves=near,
        far_count=fc,
        near_count=nc,
        overflow=jnp.any(overflow),
    )
