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


def build_octree_execution_data(tree: object) -> Optional[OctreeExecutionData]:
    """Return a padded octree execution view when explicit octree fields exist."""
    if not hasattr(tree, "oct_valid_mask"):
        return None
    view = build_explicit_octree_traversal_view(tree)

    return OctreeExecutionData(
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
    )


__all__ = ["OctreeExecutionData", "build_octree_execution_data"]
