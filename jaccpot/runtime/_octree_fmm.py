"""Octree-native FMM scaffolding built from prepared-state octree metadata."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array

from ._octree_adapter import OctreeExecutionData
from .dtypes import INDEX_DTYPE


class OctreeUpwardPlan(NamedTuple):
    """Level-major octree inputs for future octree-native upward kernels."""

    valid_mask: Array
    leaf_mask: Array
    children: Array
    child_counts: Array
    node_depths: Array
    node_ranges: Array
    nodes_by_level: Array
    level_offsets: Array
    num_levels: Array
    leaf_nodes: Array
    num_valid_nodes: Array
    num_leaf_nodes: Array


def build_octree_upward_plan(octree: OctreeExecutionData) -> OctreeUpwardPlan:
    """Package explicit octree metadata into a future-proof upward-sweep plan."""

    return OctreeUpwardPlan(
        valid_mask=jnp.asarray(octree.valid_mask, dtype=jnp.bool_),
        leaf_mask=jnp.asarray(octree.leaf_mask, dtype=jnp.bool_),
        children=jnp.asarray(octree.children, dtype=INDEX_DTYPE),
        child_counts=jnp.asarray(octree.child_counts, dtype=INDEX_DTYPE),
        node_depths=jnp.asarray(octree.node_depths, dtype=INDEX_DTYPE),
        node_ranges=jnp.asarray(octree.node_ranges, dtype=INDEX_DTYPE),
        nodes_by_level=jnp.asarray(octree.nodes_by_level, dtype=INDEX_DTYPE),
        level_offsets=jnp.asarray(octree.level_offsets, dtype=INDEX_DTYPE),
        num_levels=jnp.asarray(octree.num_levels, dtype=INDEX_DTYPE),
        leaf_nodes=jnp.asarray(octree.leaf_nodes, dtype=INDEX_DTYPE),
        num_valid_nodes=jnp.asarray(octree.num_valid_nodes, dtype=INDEX_DTYPE),
        num_leaf_nodes=jnp.asarray(octree.num_leaf_nodes, dtype=INDEX_DTYPE),
    )


__all__ = ["OctreeUpwardPlan", "build_octree_upward_plan"]
