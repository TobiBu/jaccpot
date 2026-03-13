"""Octree execution view adapters for yggdrax-backed prepared states."""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp
from jaxtyping import Array

from .dtypes import INDEX_DTYPE


class OctreeExecutionData(NamedTuple):
    """JAX-friendly octree view retained alongside radix-indexed FMM state."""

    valid_mask: Array
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


def build_octree_execution_data(tree: object) -> Optional[OctreeExecutionData]:
    """Return a padded octree execution view when explicit octree fields exist."""

    required = (
        "oct_valid_mask",
        "oct_children",
        "oct_child_counts",
        "oct_node_codes",
        "oct_node_depths",
        "oct_node_ranges",
        "oct_nodes_by_level",
        "oct_level_offsets",
        "oct_num_levels",
        "oct_leaf_mask",
        "oct_leaf_nodes",
        "radix_node_to_oct",
        "radix_leaf_to_oct",
    )
    if not all(hasattr(tree, name) for name in required):
        return None

    valid_mask = jnp.asarray(getattr(tree, "oct_valid_mask"), dtype=jnp.bool_)
    children = jnp.asarray(getattr(tree, "oct_children"), dtype=INDEX_DTYPE)
    child_counts = jnp.asarray(getattr(tree, "oct_child_counts"), dtype=INDEX_DTYPE)
    node_codes = jnp.asarray(getattr(tree, "oct_node_codes"), dtype=jnp.uint64)
    node_depths = jnp.asarray(getattr(tree, "oct_node_depths"), dtype=INDEX_DTYPE)
    node_ranges = jnp.asarray(getattr(tree, "oct_node_ranges"), dtype=INDEX_DTYPE)
    nodes_by_level = jnp.asarray(getattr(tree, "oct_nodes_by_level"), dtype=INDEX_DTYPE)
    level_offsets = jnp.asarray(getattr(tree, "oct_level_offsets"), dtype=INDEX_DTYPE)
    num_levels = jnp.asarray(getattr(tree, "oct_num_levels"), dtype=INDEX_DTYPE)
    leaf_mask = jnp.asarray(getattr(tree, "oct_leaf_mask"), dtype=jnp.bool_)
    leaf_nodes = jnp.asarray(getattr(tree, "oct_leaf_nodes"), dtype=INDEX_DTYPE)
    radix_node_to_oct = jnp.asarray(
        getattr(tree, "radix_node_to_oct"), dtype=INDEX_DTYPE
    )
    radix_leaf_to_oct = jnp.asarray(
        getattr(tree, "radix_leaf_to_oct"), dtype=INDEX_DTYPE
    )

    full_oct_nodes = valid_mask.shape[0]
    radix_nodes = radix_node_to_oct.shape[0]
    radix_node_ids = jnp.arange(radix_nodes, dtype=INDEX_DTYPE)
    node_fill = jnp.asarray(radix_nodes, dtype=INDEX_DTYPE)
    oct_to_radix_node = jnp.full((full_oct_nodes,), node_fill, dtype=INDEX_DTYPE)
    oct_to_radix_node = oct_to_radix_node.at[radix_node_to_oct].min(radix_node_ids)
    oct_to_radix_node = jnp.where(
        valid_mask & (oct_to_radix_node < node_fill),
        oct_to_radix_node,
        jnp.asarray(-1, dtype=INDEX_DTYPE),
    )

    num_internal = int(getattr(tree, "num_internal_nodes"))
    radix_leaf_nodes = jnp.arange(
        num_internal,
        num_internal + radix_leaf_to_oct.shape[0],
        dtype=INDEX_DTYPE,
    )
    leaf_fill = jnp.asarray(
        num_internal + radix_leaf_to_oct.shape[0], dtype=INDEX_DTYPE
    )
    oct_to_radix_leaf = jnp.full((full_oct_nodes,), leaf_fill, dtype=INDEX_DTYPE)
    oct_to_radix_leaf = oct_to_radix_leaf.at[radix_leaf_to_oct].min(radix_leaf_nodes)
    oct_to_radix_leaf = jnp.where(
        leaf_mask & (oct_to_radix_leaf < leaf_fill),
        oct_to_radix_leaf,
        jnp.asarray(-1, dtype=INDEX_DTYPE),
    )

    num_valid_nodes = jnp.sum(valid_mask.astype(INDEX_DTYPE))
    num_leaf_nodes = jnp.sum(leaf_mask.astype(INDEX_DTYPE))

    return OctreeExecutionData(
        valid_mask=valid_mask,
        children=children,
        child_counts=child_counts,
        node_codes=node_codes,
        node_depths=node_depths,
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
        num_valid_nodes=num_valid_nodes,
        num_leaf_nodes=num_leaf_nodes,
    )


__all__ = ["OctreeExecutionData", "build_octree_execution_data"]
