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
    box_centers: Array
    box_half_extents: Array
    box_radii: Array
    box_max_extents: Array


_MAX_MORTON_LEVEL = 21


def _compact3_u64(x: Array) -> Array:
    x = jnp.asarray(x, dtype=jnp.uint64) & jnp.uint64(0x1FFFFF)
    x = (x | (x << jnp.uint64(32))) & jnp.uint64(0x1F00000000FFFF)
    x = (x | (x << jnp.uint64(16))) & jnp.uint64(0x1F0000FF0000FF)
    x = (x | (x << jnp.uint64(8))) & jnp.uint64(0x100F00F00F00F00F)
    x = (x | (x << jnp.uint64(4))) & jnp.uint64(0x10C30C30C30C30C3)
    x = (x | (x << jnp.uint64(2))) & jnp.uint64(0x1249249249249249)
    return x


def _compute_octree_box_geometry(
    *,
    valid_mask: Array,
    node_codes: Array,
    node_depths: Array,
    bounds_min: Array,
    bounds_max: Array,
) -> tuple[Array, Array, Array, Array]:
    """Compute explicit octree box geometry directly from octree code/depth pairs."""

    bounds_min = jnp.asarray(bounds_min)
    bounds_max = jnp.asarray(bounds_max)
    domain = bounds_max - bounds_min
    depth_clamped = jnp.clip(
        jnp.asarray(node_depths, dtype=INDEX_DTYPE),
        jnp.asarray(0, dtype=INDEX_DTYPE),
        jnp.asarray(_MAX_MORTON_LEVEL, dtype=INDEX_DTYPE),
    )
    shift = jnp.asarray(_MAX_MORTON_LEVEL, dtype=jnp.uint64) - depth_clamped.astype(
        jnp.uint64
    )
    node_codes = jnp.asarray(node_codes, dtype=jnp.uint64)
    x_coords = _compact3_u64(node_codes)
    y_coords = _compact3_u64(node_codes >> jnp.uint64(1))
    z_coords = _compact3_u64(node_codes >> jnp.uint64(2))

    x_idx = x_coords >> shift
    y_idx = y_coords >> shift
    z_idx = z_coords >> shift
    indices = jnp.stack([x_idx, y_idx, z_idx], axis=1).astype(bounds_min.dtype)

    counts = jnp.left_shift(
        jnp.ones_like(depth_clamped, dtype=jnp.uint64),
        depth_clamped.astype(jnp.uint64),
    )
    counts = jnp.maximum(counts, jnp.uint64(1)).astype(bounds_min.dtype)
    cell_sizes = domain[None, :] / counts[:, None]
    mins = bounds_min[None, :] + cell_sizes * indices
    maxs = mins + cell_sizes
    centers = 0.5 * (mins + maxs)
    half_extents = 0.5 * (maxs - mins)
    radii = jnp.linalg.norm(half_extents, axis=1)
    max_extents = jnp.max(half_extents, axis=1)

    valid_mask = jnp.asarray(valid_mask, dtype=jnp.bool_)
    centers = jnp.where(valid_mask[:, None], centers, 0.0)
    half_extents = jnp.where(valid_mask[:, None], half_extents, 0.0)
    radii = jnp.where(valid_mask, radii, 0.0)
    max_extents = jnp.where(valid_mask, max_extents, 0.0)
    return centers, half_extents, radii, max_extents


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
    box_centers, box_half_extents, box_radii, box_max_extents = (
        _compute_octree_box_geometry(
            valid_mask=valid_mask,
            node_codes=node_codes,
            node_depths=node_depths,
            bounds_min=jnp.asarray(getattr(tree, "bounds_min")),
            bounds_max=jnp.asarray(getattr(tree, "bounds_max")),
        )
    )

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
        box_centers=box_centers,
        box_half_extents=box_half_extents,
        box_radii=box_radii,
        box_max_extents=box_max_extents,
    )


__all__ = ["OctreeExecutionData", "build_octree_execution_data"]
