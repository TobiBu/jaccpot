"""Helpers for constructing node multipole expansions."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, jaxtyped

from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.multipole_utils import total_coefficients
from yggdrax.tree import RadixTree
from yggdrax.geometry import TreeGeometry, compute_tree_geometry
from yggdrax.tree_moments import (
    TreeMassMoments,
    TreeMultipoleMoments,
    compute_tree_mass_moments,
    compute_tree_multipole_moments,
    pack_multipole_expansions,
    translate_packed_moments,
    tree_moments_from_raw,
)

_CENTER_MODES = ("com", "aabb", "explicit")


class NodeMultipoleData(NamedTuple):
    """Packed multipole expansions and their metadata."""

    order: int
    centers: Array
    moments: TreeMultipoleMoments
    packed: Array
    component_matrix: Array


@partial(jax.jit, static_argnames=("order", "num_internal"))
def _aggregate_m2m_impl(
    packed: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    *,
    order: int,
    num_internal: int,
) -> Array:
    prototype = packed[0]

    def add_child(
        node_coeff: Array,
        child_idx: Array,
        node_idx: Array,
        state: Array,
    ) -> Array:
        def true_branch(idx: Array) -> Array:
            delta = centers[idx] - centers[node_idx]
            translated = translate_packed_moments(
                state[idx],
                delta,
                order,
            )
            return node_coeff + translated

        return lax.cond(
            child_idx >= 0,
            true_branch,
            lambda _: node_coeff,
            child_idx,
        )

    def body(iter_idx: Array, state: Array) -> Array:
        node_idx = num_internal - 1 - iter_idx
        node_coeff = jnp.zeros_like(prototype)
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
        state = state.at[node_idx].set(node_coeff)
        return state

    return lax.fori_loop(0, num_internal, body, packed)


@jaxtyped(typechecker=beartype)
def compute_node_multipoles(
    tree: RadixTree,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    max_order: int = 2,
    center_mode: str = "com",
    explicit_centers: Optional[Array] = None,
) -> NodeMultipoleData:
    """Construct packed multipole expansions for every node in the tree.

    Parameters
    ----------
    tree:
        Radix tree built from Morton-sorted particles.
    positions_sorted, masses_sorted:
        Particle data reordered to match ``tree.particle_indices``.
    max_order:
        Highest multipole order to keep.
    center_mode:
        ``"com"`` uses each node's centre of mass, ``"aabb"`` uses the
        geometry centre, and ``"explicit"`` consumes ``explicit_centers``.
    explicit_centers:
        User-provided expansion centres when ``center_mode == "explicit"``.
    """

    mode = center_mode.lower()
    if mode not in _CENTER_MODES:
        raise ValueError(f"Unknown center_mode '{center_mode}'")

    centers: Optional[Array]
    if mode == "explicit":
        if explicit_centers is None:
            raise ValueError(
                "explicit_centers must be provided for 'explicit'",
            )
        if explicit_centers.shape != (tree.parent.shape[0], 3):
            raise ValueError("explicit_centers must have shape (num_nodes, 3)")
        centers = jnp.asarray(explicit_centers, dtype=positions_sorted.dtype)
    elif mode == "aabb":
        geom = compute_tree_geometry(tree, positions_sorted)
        centers = geom.center
    else:
        centers = None

    moments = compute_tree_multipole_moments(
        tree,
        positions_sorted,
        masses_sorted,
        expansion_centers=centers,
        max_order=max_order,
    )

    packed = pack_multipole_expansions(moments, max_order=max_order)
    component_matrix = jnp.asarray(packed)

    return NodeMultipoleData(
        order=int(moments.max_order),
        centers=moments.center,
        moments=moments,
        packed=packed,
        component_matrix=component_matrix,
    )


def _aggregate_multipoles_via_m2m(
    tree: RadixTree,
    centers: Array,
    base_moments: TreeMultipoleMoments,
) -> TreeMultipoleMoments:
    total_nodes = base_moments.mass.shape[0]
    num_internal = int(tree.num_internal_nodes)
    order = int(base_moments.max_order)
    coeffs = total_coefficients(order)
    packed = jnp.zeros(
        (total_nodes, coeffs),
        dtype=base_moments.raw_packed.dtype,
    )

    if num_internal < total_nodes:
        leaf_slice = slice(num_internal, total_nodes)
        packed = packed.at[leaf_slice].set(
            base_moments.raw_packed[leaf_slice, :coeffs],
        )

    if num_internal > 0:
        left_child = jnp.asarray(tree.left_child, dtype=INDEX_DTYPE)
        right_child = jnp.asarray(tree.right_child, dtype=INDEX_DTYPE)
        packed = _aggregate_m2m_impl(
            packed,
            centers,
            left_child,
            right_child,
            order=order,
            num_internal=num_internal,
        )

    return tree_moments_from_raw(packed, centers, order)


class TreeUpwardData(NamedTuple):
    """Container bundling data needed for the FMM upward sweep."""

    geometry: TreeGeometry
    mass_moments: TreeMassMoments
    multipoles: NodeMultipoleData


@jaxtyped(typechecker=beartype)
def prepare_upward_sweep(
    tree: RadixTree,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    max_order: int = 2,
    center_mode: str = "com",
    explicit_centers: Optional[Array] = None,
) -> TreeUpwardData:
    """Compute geometry, moments, and packed multipoles for a tree."""

    geometry = compute_tree_geometry(tree, positions_sorted)
    mass_moments = compute_tree_mass_moments(
        tree,
        positions_sorted,
        masses_sorted,
    )

    total_nodes = tree.parent.shape[0]
    mode = center_mode.lower()
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

    direct_moments = compute_tree_multipole_moments(
        tree,
        positions_sorted,
        masses_sorted,
        expansion_centers=centers,
        max_order=max_order,
    )

    aggregated = _aggregate_multipoles_via_m2m(
        tree,
        centers,
        direct_moments,
    )

    packed = pack_multipole_expansions(aggregated, max_order=max_order)
    component_matrix = jnp.asarray(packed)

    multipoles = NodeMultipoleData(
        order=int(aggregated.max_order),
        centers=centers,
        moments=aggregated,
        packed=packed,
        component_matrix=component_matrix,
    )

    return TreeUpwardData(
        geometry=geometry,
        mass_moments=mass_moments,
        multipoles=multipoles,
    )


__all__ = [
    "NodeMultipoleData",
    "TreeUpwardData",
    "compute_node_multipoles",
    "prepare_upward_sweep",
]
