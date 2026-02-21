"""Spherical-harmonics tree expansion helpers.

This module implements the *upward sweep* (P2M + M2M) for the spherical
real-tesseral SH basis used by the Dehnen-style backend.

The goal for the first iteration ("Option A") is:
- define a per-node multipole coefficient layout of size (p+1)^2
- compute leaf multipoles by accumulating particle contributions (P2M)
- aggregate multipoles upward by translating child expansions to the parent
  center and summing (M2M)

Downward sweep (M2L/L2L) and particle evaluation (L2P) are handled elsewhere.

Notes
-----
We intentionally keep this implementation correctness-first:
- it uses analytic/finite-difference helpers for low-order invariants
- we only guarantee order 0 and order 1 invariants initially

Once the full spherical pipeline is integrated, we will replace the P2M/M2M
kernels with Dehnen (2014) recurrences and add higher-order accuracy tests.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, jaxtyped

from yggdrasil.dtypes import INDEX_DTYPE
from jaccpot.operators.real_harmonics import m2m_real, p2m_real_direct, sh_size
from yggdrasil.tree import RadixTree
from yggdrasil.geometry import TreeGeometry, compute_tree_geometry
from yggdrasil.tree_moments import TreeMassMoments, compute_tree_mass_moments


class SphericalNodeMultipoleData(NamedTuple):
    """Packed spherical multipole coefficients and their metadata."""

    order: int
    centers: Array  # (num_nodes, 3)
    packed: Array  # (num_nodes, (p+1)^2)


class SphericalTreeUpwardData(NamedTuple):
    """Container bundling data needed for the spherical upward sweep."""

    geometry: TreeGeometry
    mass_moments: TreeMassMoments
    multipoles: SphericalNodeMultipoleData


_CENTER_MODES = ("com", "aabb", "explicit")


@partial(
    jax.jit,
    static_argnames=("order", "max_leaf_size", "num_internal", "total_nodes"),
)
def _p2m_leaves(
    tree: RadixTree,
    positions_sorted: Array,
    masses_sorted: Array,
    centers: Array,
    *,
    order: int,
    max_leaf_size: int,
    num_internal: int,
    total_nodes: int,
) -> Array:
    """Leaf P2M for Dehnen-style real SH coefficients.

    Computes per-leaf multipoles by summing point-mass contributions
    (with the same per-degree packing as :func:`p2m_real_direct`).
    """

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    num_internal = int(num_internal)
    total_nodes = int(total_nodes)
    coeffs = sh_size(p)

    packed = jnp.zeros((total_nodes, coeffs), dtype=positions_sorted.dtype)

    # Leaves live in [num_internal, total_nodes)
    leaf_nodes = jnp.arange(num_internal, total_nodes, dtype=INDEX_DTYPE)
    if leaf_nodes.size == 0:
        return packed

    ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)[leaf_nodes]
    starts = ranges[:, 0]
    ends_inclusive = ranges[:, 1]
    counts = ends_inclusive - starts + 1

    idx = jnp.arange(int(max_leaf_size), dtype=INDEX_DTYPE)
    particle_idx = starts[:, None] + idx[None, :]
    valid = idx[None, :] < counts[:, None]
    safe_idx = jnp.clip(particle_idx, 0, positions_sorted.shape[0] - 1)

    pos = positions_sorted[safe_idx]
    pos = jnp.where(valid[..., None], pos, 0.0)
    masses = masses_sorted[safe_idx]
    masses = jnp.where(valid, masses, 0.0)

    def leaf_accumulate(pos_i: Array, mass_i: Array, center_i: Array) -> Array:
        delta = pos_i - center_i
        return p2m_real_direct(delta, mass_i, order=p)

    # vmap over particles within each leaf
    particle_vm = jax.vmap(leaf_accumulate, in_axes=(0, 0, None))
    leaf_vm = jax.vmap(particle_vm, in_axes=(0, 0, 0))
    contribs = leaf_vm(pos, masses, centers[leaf_nodes])
    leaf_coeffs = jnp.sum(contribs, axis=1)
    packed = packed.at[leaf_nodes].set(leaf_coeffs)

    return packed


@partial(jax.jit, static_argnames=("order", "num_internal"))
def _aggregate_m2m(
    packed: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    *,
    order: int,
    num_internal: int,
) -> Array:
    """Upward aggregation by translating child multipoles to parent.

    Notes
    -----
    We intentionally keep the "multi-pass relaxation" strategy so we don't
    rely on any internal-node ordering.
    """

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    def add_child(
        node_coeff: Array,
        child_idx: Array,
        node_idx: Array,
        state: Array,
    ):
        def true_branch(idx):
            child_coeff = state[idx]
            delta = centers[idx] - centers[node_idx]
            translated = m2m_real(child_coeff, delta, order=p)
            return node_coeff + translated

        return lax.cond(
            child_idx >= 0,
            true_branch,
            lambda _: node_coeff,
            child_idx,
        )

    def body(node_idx, state):
        # Recompute the node's coefficients from scratch from its children.
        # (This is important because we may need multiple passes if internal
        # nodes are not in topological index order.)
        node_coeff = jnp.zeros_like(state[0])
        node_coeff = add_child(
            node_coeff,
            left_child[node_idx],
            node_idx,
            state,
        )
        node_coeff = add_child(
            node_coeff,
            right_child[node_idx],
            node_idx,
            state,
        )
        return state.at[node_idx].set(node_coeff)

    # NOTE:
    # The radix tree does *not* guarantee that internal nodes are arranged in
    # topological order by index (i.e. an internal node can reference another
    # internal node with a *smaller* index as its child). Because of that we
    # cannot reliably do a single bottom-up sweep by index.
    #
    # For the low-order scaffolding we instead perform a small, safe number of
    # relaxation passes. Each pass updates every internal node from its current
    # children, so after O(tree_depth) passes the values converge to the true
    # aggregated multipoles.
    #
    # IMPORTANT: We need O(tree_depth) passes, NOT O(num_internal) passes.
    # Using num_internal was causing O(N^2) behavior and 30+ second runtimes.
    # A safe upper bound for tree depth is ceil(log2(num_internal + 1)) + 1.
    import math

    max_depth = int(math.ceil(math.log2(max(num_internal + 1, 2)))) + 1

    def one_pass(state):
        return lax.fori_loop(0, num_internal, body, state)

    return lax.fori_loop(0, max_depth, lambda _, s: one_pass(s), packed)


@jaxtyped(typechecker=beartype)
def prepare_spherical_upward_sweep(
    tree: RadixTree,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    max_order: int = 2,
    center_mode: str = "com",
    explicit_centers: Optional[Array] = None,
    max_leaf_size: Optional[int] = None,
) -> SphericalTreeUpwardData:
    """Compute spherical multipoles for every node (Option A scaffolding).

    For now we support invariants for order<=1. For higher orders we allocate
    the correct (p+1)^2 buffers but only fill the low-order components.
    """

    p = int(max_order)
    if p < 0:
        raise ValueError("max_order must be >= 0")

    geometry = compute_tree_geometry(tree, positions_sorted)
    mass_moments = compute_tree_mass_moments(
        tree,
        positions_sorted,
        masses_sorted,
    )

    total_nodes = int(tree.parent.shape[0])
    mode = str(center_mode).strip().lower()
    if mode == "com":
        centers = mass_moments.center_of_mass
    elif mode == "aabb":
        centers = geometry.center
    elif mode == "explicit":
        if explicit_centers is None:
            raise ValueError(
                "explicit_centers must be provided for 'explicit'",
            )
        if explicit_centers.shape != (total_nodes, 3):
            raise ValueError("explicit_centers must have shape (num_nodes, 3)")
        centers = explicit_centers
    else:
        raise ValueError(f"Unknown center_mode '{center_mode}'")

    centers = jnp.asarray(centers, dtype=positions_sorted.dtype)

    if max_leaf_size is None:
        # Determine maximum leaf size from node_ranges using host values.
        num_internal = int(tree.num_internal_nodes)
        leaf_ranges = jax.device_get(tree.node_ranges)[num_internal:]
        if leaf_ranges.shape[0] == 0:
            max_leaf = 0
        else:
            counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
            max_leaf = int(counts.max())
    else:
        max_leaf = int(max_leaf_size)

    packed = _p2m_leaves(
        tree,
        jnp.asarray(positions_sorted),
        jnp.asarray(masses_sorted),
        centers,
        order=p,
        max_leaf_size=max_leaf,
        num_internal=int(tree.num_internal_nodes),
        total_nodes=int(tree.parent.shape[0]),
    )

    num_internal = int(tree.num_internal_nodes)
    if num_internal > 0:
        packed = _aggregate_m2m(
            packed,
            centers,
            jnp.asarray(tree.left_child, dtype=INDEX_DTYPE),
            jnp.asarray(tree.right_child, dtype=INDEX_DTYPE),
            order=p,
            num_internal=num_internal,
        )

    multipoles = SphericalNodeMultipoleData(
        order=p,
        centers=centers,
        packed=packed,
    )

    return SphericalTreeUpwardData(
        geometry=geometry,
        mass_moments=mass_moments,
        multipoles=multipoles,
    )


__all__ = [
    "SphericalNodeMultipoleData",
    "SphericalTreeUpwardData",
    "prepare_spherical_upward_sweep",
]
