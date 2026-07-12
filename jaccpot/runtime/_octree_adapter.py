"""Octree execution view adapters for yggdrax-backed prepared states."""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp
from jaxtyping import Array
from yggdrax.octree import build_explicit_octree_traversal_view

from .dtypes import INDEX_DTYPE


class OctreeExecutionData(NamedTuple):
    """JAX-friendly octree view retained alongside radix-indexed FMM state."""

    valid_mask: Array
    parent: Array
    children: Array
    child_counts: Array
    node_codes: Array
    node_depths: Array
    node_ranges: Array
    nodes_by_level: Array
    level_offsets: Array
    num_levels: Array
    leaf_mask: Array
    leaf_nodes: Array
    radix_node_to_oct: Array
    radix_leaf_to_oct: Array
    oct_to_radix_node: Array
    oct_to_radix_leaf: Array
    num_valid_nodes: Array
    num_leaf_nodes: Array
    box_centers: Array
    box_half_extents: Array
    box_radii: Array
    box_max_extents: Array


def _build_fallback_octree_execution_data(tree: object) -> OctreeExecutionData:
    """Build a consistent octree execution view from radix topology fields."""

    parent = jnp.asarray(getattr(tree, "parent"), dtype=INDEX_DTYPE)
    left_child = jnp.asarray(getattr(tree, "left_child"), dtype=INDEX_DTYPE)
    right_child = jnp.asarray(getattr(tree, "right_child"), dtype=INDEX_DTYPE)
    node_ranges = jnp.asarray(getattr(tree, "node_ranges"), dtype=INDEX_DTYPE)
    bounds_min = jnp.asarray(getattr(tree, "bounds_min"))
    bounds_max = jnp.asarray(getattr(tree, "bounds_max"))
    node_codes = jnp.asarray(getattr(tree, "morton_codes"), dtype=jnp.uint64)

    num_nodes = int(parent.shape[0])
    num_internal = int(left_child.shape[0])
    num_leaves = max(0, num_nodes - num_internal)

    children = jnp.full((num_nodes, 8), -1, dtype=INDEX_DTYPE)
    if num_internal > 0:
        children = children.at[:num_internal, 0].set(left_child[:num_internal])
        children = children.at[:num_internal, 1].set(right_child[:num_internal])
    child_counts = jnp.sum(children >= 0, axis=1, dtype=INDEX_DTYPE)

    depth = jnp.zeros((num_nodes,), dtype=INDEX_DTYPE)
    if num_nodes > 0:
        for idx in range(1, num_nodes):
            p = int(parent[idx])
            depth = depth.at[idx].set(
                depth[p] + 1 if p >= 0 else jnp.asarray(0, dtype=INDEX_DTYPE)
            )
    num_levels_int = (int(jnp.max(depth)) + 1) if num_nodes > 0 else 1
    num_levels = jnp.asarray(num_levels_int, dtype=INDEX_DTYPE)
    nodes_by_level = jnp.argsort(depth, stable=True).astype(INDEX_DTYPE)
    level_offsets = jnp.zeros((num_levels_int + 1,), dtype=INDEX_DTYPE)
    if num_nodes > 0:
        counts = jnp.bincount(depth, length=num_levels_int).astype(INDEX_DTYPE)
        level_offsets = level_offsets.at[1:].set(jnp.cumsum(counts, dtype=INDEX_DTYPE))

    leaf_mask = jnp.arange(num_nodes, dtype=INDEX_DTYPE) >= int(num_internal)
    leaf_nodes = jnp.arange(num_internal, num_nodes, dtype=INDEX_DTYPE)

    radix_node_to_oct = jnp.arange(num_nodes, dtype=INDEX_DTYPE)
    radix_leaf_to_oct = jnp.arange(num_internal, num_nodes, dtype=INDEX_DTYPE)
    oct_to_radix_node = jnp.arange(num_nodes, dtype=INDEX_DTYPE)
    oct_to_radix_leaf = jnp.full((num_nodes,), -1, dtype=INDEX_DTYPE)
    if num_leaves > 0:
        oct_to_radix_leaf = oct_to_radix_leaf.at[num_internal:num_nodes].set(
            jnp.arange(num_leaves, dtype=INDEX_DTYPE)
        )

    global_center = jnp.asarray((bounds_min + bounds_max) * 0.5)
    global_half_extent = jnp.asarray((bounds_max - bounds_min) * 0.5)
    box_centers = jnp.broadcast_to(global_center[None, :], (num_nodes, 3))
    box_half_extents = jnp.broadcast_to(global_half_extent[None, :], (num_nodes, 3))
    box_radii = jnp.broadcast_to(
        jnp.linalg.norm(global_half_extent)[None], (num_nodes,)
    )
    box_max_extents = jnp.broadcast_to(jnp.max(global_half_extent)[None], (num_nodes,))

    return OctreeExecutionData(
        valid_mask=jnp.ones((num_nodes,), dtype=bool),
        parent=parent,
        children=children,
        child_counts=child_counts,
        node_codes=node_codes,
        node_depths=depth,
        node_ranges=node_ranges,
        nodes_by_level=nodes_by_level,
        level_offsets=level_offsets,
        num_levels=num_levels,
        leaf_mask=leaf_mask,
        leaf_nodes=leaf_nodes,
        radix_node_to_oct=radix_node_to_oct,
        radix_leaf_to_oct=radix_leaf_to_oct,
        oct_to_radix_node=oct_to_radix_node,
        oct_to_radix_leaf=oct_to_radix_leaf,
        num_valid_nodes=jnp.asarray(num_nodes, dtype=INDEX_DTYPE),
        num_leaf_nodes=jnp.asarray(num_leaves, dtype=INDEX_DTYPE),
        box_centers=box_centers,
        box_half_extents=box_half_extents,
        box_radii=box_radii,
        box_max_extents=box_max_extents,
    )


def build_octree_execution_data_with_status(
    tree: object,
) -> tuple[Optional[OctreeExecutionData], bool]:
    """Build the octree execution view and report whether the NATIVE octree was used.

    Returns ``(data, used_native)``. ``used_native`` is ``False`` when the native
    explicit-octree view is DEGENERATE -- its root does not span all particles -- which
    happens when the tree's leaves are count-based rather than octree-cell-aligned (e.g.
    near-uniform data with unpopulated ``leaf_depths``), collapsing distinct leaves into a
    partial-domain octree. In that case a binary fallback view is returned with
    ``used_native=False``, signalling that the octree node space is INCONSISTENT with the
    native octree walk: callers must then build the far/near interaction lists from the
    same (compat/radix) tree, not from ``build_octree_native_far_pairs`` (which walks the
    degenerate native view). Mixing a native-octree far list with this fallback view is
    exactly what corrupted the octree backend on uniform data.
    """
    if not hasattr(tree, "oct_valid_mask"):
        return None, False
    view = build_explicit_octree_traversal_view(tree)
    root_oct = int(jnp.asarray(view.radix_node_to_oct, dtype=INDEX_DTYPE)[0])
    tree_root_range = jnp.asarray(getattr(tree, "node_ranges"))[0]
    oct_root_range = jnp.asarray(view.node_ranges)[root_oct]
    if not bool(jnp.all(tree_root_range == oct_root_range)):
        return _build_fallback_octree_execution_data(tree), False

    return (
        OctreeExecutionData(
            valid_mask=view.valid_mask,
            parent=view.parent,
            children=view.children,
            child_counts=view.child_counts,
            node_codes=view.node_codes,
            node_depths=view.node_depths,
            node_ranges=view.node_ranges,
            nodes_by_level=view.nodes_by_level,
            level_offsets=view.level_offsets,
            num_levels=view.num_levels,
            leaf_mask=view.leaf_mask,
            leaf_nodes=view.leaf_nodes,
            radix_node_to_oct=view.radix_node_to_oct,
            radix_leaf_to_oct=view.radix_leaf_to_oct,
            oct_to_radix_node=view.oct_to_radix_node,
            oct_to_radix_leaf=view.oct_to_radix_leaf,
            num_valid_nodes=view.num_valid_nodes,
            num_leaf_nodes=view.num_leaf_nodes,
            box_centers=view.box_centers,
            box_half_extents=view.box_half_extents,
            box_radii=view.box_radii,
            box_max_extents=view.box_max_extents,
        ),
        True,
    )


def build_octree_execution_data(tree: object) -> Optional[OctreeExecutionData]:
    """Return a padded octree execution view when explicit octree fields exist."""
    data, _ = build_octree_execution_data_with_status(tree)
    return data


__all__ = [
    "OctreeExecutionData",
    "build_octree_execution_data",
    "build_octree_execution_data_with_status",
]
