"""Octree-native FMM scaffolding built from prepared-state octree metadata."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array
from yggdrax.interactions import CompactTaggedOctreeFarPairs, NodeInteractionList

from jaccpot.operators.complex_harmonics import p2m_complex_batch
from jaccpot.operators.complex_ops import (
    enforce_conjugate_symmetry_batch,
    l2l_complex_batch,
    m2l_complex_reference_batch,
    m2m_complex,
)
from jaccpot.operators.real_harmonics import sh_size

from ._octree_adapter import OctreeExecutionData
from .dtypes import INDEX_DTYPE, as_index, complex_dtype_for_real


class OctreeUpwardPlan(NamedTuple):
    """Level-major octree inputs for future octree-native upward kernels."""

    valid_mask: Array
    leaf_mask: Array
    parent: Array
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
    box_centers: Array
    box_half_extents: Array
    box_radii: Array
    box_max_extents: Array


class OctreeSolidFMMComplexMultipoles(NamedTuple):
    """Octree-native complex multipoles stored in octree node space."""

    order: int
    centers: Array
    packed: Array


class OctreeInteractionPlan(NamedTuple):
    """Level-major far-field pairs remapped into explicit octree node space."""

    valid_mask: Array
    target_nodes: Array
    source_nodes: Array
    target_levels: Array
    offsets: Array
    counts: Array
    level_offsets: Array
    num_pairs: Array


class OctreeSolidFMMDownwardPlan(NamedTuple):
    """Octree-native downward/M2L scaffold stored in explicit octree node space."""

    order: int
    centers: Array
    locals_packed: Array
    parent: Array
    nodes_by_level: Array
    level_offsets: Array
    target_nodes: Array
    source_nodes: Array
    target_levels: Array
    interaction_level_offsets: Array
    valid_interactions: Array
    num_pairs: Array


def build_octree_upward_plan(octree: OctreeExecutionData) -> OctreeUpwardPlan:
    """Package explicit octree metadata into a future-proof upward-sweep plan."""

    return OctreeUpwardPlan(
        valid_mask=jnp.asarray(octree.valid_mask, dtype=jnp.bool_),
        leaf_mask=jnp.asarray(octree.leaf_mask, dtype=jnp.bool_),
        parent=jnp.asarray(octree.parent, dtype=INDEX_DTYPE),
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
        box_centers=jnp.asarray(octree.box_centers),
        box_half_extents=jnp.asarray(octree.box_half_extents),
        box_radii=jnp.asarray(octree.box_radii),
        box_max_extents=jnp.asarray(octree.box_max_extents),
    )


def build_octree_interaction_plan(
    octree: OctreeExecutionData,
    interactions: NodeInteractionList,
) -> OctreeInteractionPlan:
    """Remap radix far-field pairs into explicit octree node space."""

    num_oct_nodes = int(octree.valid_mask.shape[0])
    sentinel_node = jnp.asarray(num_oct_nodes, dtype=INDEX_DTYPE)
    sentinel_level = jnp.asarray(int(octree.num_levels), dtype=INDEX_DTYPE)
    radix_sources = jnp.asarray(interactions.sources, dtype=INDEX_DTYPE)
    radix_targets = jnp.asarray(interactions.targets, dtype=INDEX_DTYPE)

    oct_sources = jnp.where(
        radix_sources >= 0,
        octree.radix_node_to_oct[jnp.clip(radix_sources, min=0)],
        jnp.asarray(-1, dtype=INDEX_DTYPE),
    )
    oct_targets = jnp.where(
        radix_targets >= 0,
        octree.radix_node_to_oct[jnp.clip(radix_targets, min=0)],
        jnp.asarray(-1, dtype=INDEX_DTYPE),
    )
    valid = (
        (radix_sources >= 0)
        & (radix_targets >= 0)
        & (oct_sources >= 0)
        & (oct_targets >= 0)
    )

    return build_octree_interaction_plan_from_octree_pairs(
        octree,
        oct_sources=oct_sources,
        oct_targets=oct_targets,
        valid=valid,
        sentinel_node=sentinel_node,
        sentinel_level=sentinel_level,
    )


def build_octree_interaction_plan_from_native_pairs(
    octree: OctreeExecutionData,
    far_pairs: CompactTaggedOctreeFarPairs,
) -> OctreeInteractionPlan:
    """Package octree-native far-field pairs for octree M2L scheduling."""

    oct_sources = jnp.asarray(far_pairs.sources, dtype=INDEX_DTYPE)
    oct_targets = jnp.asarray(far_pairs.targets, dtype=INDEX_DTYPE)
    num_oct_nodes = int(octree.valid_mask.shape[0])
    return build_octree_interaction_plan_from_octree_pairs(
        octree,
        oct_sources=oct_sources,
        oct_targets=oct_targets,
        valid=(
            (oct_sources >= 0)
            & (oct_targets >= 0)
            & (oct_sources < num_oct_nodes)
            & (oct_targets < num_oct_nodes)
        ),
        sentinel_node=jnp.asarray(num_oct_nodes, dtype=INDEX_DTYPE),
        sentinel_level=jnp.asarray(int(octree.num_levels), dtype=INDEX_DTYPE),
    )


def build_octree_interaction_plan_from_octree_pairs(
    octree: OctreeExecutionData,
    *,
    oct_sources: Array,
    oct_targets: Array,
    valid: Array,
    sentinel_node: Array,
    sentinel_level: Array,
) -> OctreeInteractionPlan:
    """Sort octree-space far-field pairs into level-major execution order."""

    num_oct_nodes = int(octree.valid_mask.shape[0])
    safe_targets = jnp.where(valid, oct_targets, sentinel_node)
    safe_sources = jnp.where(valid, oct_sources, sentinel_node)
    target_depths = octree.node_depths[jnp.clip(safe_targets, 0, num_oct_nodes - 1)]
    safe_levels = jnp.where(valid, target_depths, sentinel_level)
    order = jnp.lexsort((safe_sources, safe_targets, safe_levels))

    sorted_valid = valid[order]
    sorted_targets = safe_targets[order]
    sorted_sources = safe_sources[order]
    sorted_levels = safe_levels[order]

    counts = jnp.zeros((num_oct_nodes,), dtype=INDEX_DTYPE)
    counts = counts.at[jnp.where(sorted_valid, sorted_targets, 0)].add(
        sorted_valid.astype(INDEX_DTYPE)
    )
    offsets = jnp.cumsum(counts, dtype=INDEX_DTYPE) - counts

    num_levels = int(octree.num_levels)
    level_counts = jnp.zeros((num_levels,), dtype=INDEX_DTYPE)
    level_counts = level_counts.at[jnp.where(sorted_valid, sorted_levels, 0)].add(
        sorted_valid.astype(INDEX_DTYPE)
    )
    level_offsets = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=INDEX_DTYPE),
            jnp.cumsum(level_counts, dtype=INDEX_DTYPE),
        ],
        axis=0,
    )

    return OctreeInteractionPlan(
        valid_mask=sorted_valid,
        target_nodes=sorted_targets,
        source_nodes=sorted_sources,
        target_levels=sorted_levels,
        offsets=offsets,
        counts=counts,
        level_offsets=level_offsets,
        num_pairs=jnp.sum(sorted_valid.astype(INDEX_DTYPE)),
    )


def build_octree_downward_plan(
    octree: OctreeExecutionData,
    multipoles: OctreeSolidFMMComplexMultipoles,
    interactions: OctreeInteractionPlan,
) -> OctreeSolidFMMDownwardPlan:
    """Build octree-native downward scaffolding for future M2L/L2L execution."""

    locals_packed = jnp.zeros_like(multipoles.packed)
    return OctreeSolidFMMDownwardPlan(
        order=int(multipoles.order),
        centers=jnp.asarray(multipoles.centers),
        locals_packed=locals_packed,
        parent=jnp.asarray(octree.parent, dtype=INDEX_DTYPE),
        nodes_by_level=jnp.asarray(octree.nodes_by_level, dtype=INDEX_DTYPE),
        level_offsets=jnp.asarray(octree.level_offsets, dtype=INDEX_DTYPE),
        target_nodes=jnp.asarray(interactions.target_nodes, dtype=INDEX_DTYPE),
        source_nodes=jnp.asarray(interactions.source_nodes, dtype=INDEX_DTYPE),
        target_levels=jnp.asarray(interactions.target_levels, dtype=INDEX_DTYPE),
        interaction_level_offsets=jnp.asarray(
            interactions.level_offsets, dtype=INDEX_DTYPE
        ),
        valid_interactions=jnp.asarray(interactions.valid_mask, dtype=jnp.bool_),
        num_pairs=jnp.asarray(interactions.num_pairs, dtype=INDEX_DTYPE),
    )


@partial(jax.jit, static_argnames=("order", "chunk_size"))
def _accumulate_octree_m2l_complex_chunked(
    locals_coeffs: Array,
    multipoles: Array,
    centers: Array,
    target_nodes: Array,
    source_nodes: Array,
    valid_interactions: Array,
    *,
    order: int,
    chunk_size: int,
) -> Array:
    """Accumulate complex-basis octree M2L contributions in fixed-size chunks."""

    pair_count = int(target_nodes.shape[0])
    if pair_count == 0:
        return locals_coeffs

    chunk = int(max(chunk_size, 1))
    starts = jnp.arange(0, pair_count, chunk, dtype=INDEX_DTYPE)

    def body(local_accum: Array, start_idx: Array) -> tuple[Array, None]:
        offset = jnp.arange(chunk, dtype=INDEX_DTYPE)
        idx = start_idx + offset
        in_range = idx < pair_count
        safe_idx = jnp.where(in_range, idx, 0)

        tgt_chunk = target_nodes[safe_idx]
        src_chunk = source_nodes[safe_idx]
        valid = in_range & valid_interactions[safe_idx]
        safe_tgt = jnp.where(valid, tgt_chunk, 0)
        safe_src = jnp.where(valid, src_chunk, 0)

        deltas = centers[safe_tgt] - centers[safe_src]
        src_mult = multipoles[safe_src]
        contribs = m2l_complex_reference_batch(
            src_mult,
            deltas,
            order=order,
            rotation="solidfmm",
        ).astype(locals_coeffs.dtype)
        contribs = jnp.where(valid[:, None], contribs, 0.0)

        sort_idx = jnp.argsort(safe_tgt)
        tgt_sorted = safe_tgt[sort_idx]
        contribs_sorted = contribs[sort_idx]
        valid_sorted = valid[sort_idx]
        new_group = jnp.concatenate(
            (jnp.asarray([True], dtype=jnp.bool_), tgt_sorted[1:] != tgt_sorted[:-1]),
            axis=0,
        )
        group_ids = jnp.cumsum(new_group.astype(INDEX_DTYPE)) - jnp.asarray(
            1,
            dtype=INDEX_DTYPE,
        )
        reduced = jax.ops.segment_sum(contribs_sorted, group_ids, chunk)

        unique_targets = jnp.zeros((chunk,), dtype=INDEX_DTYPE)
        unique_targets = unique_targets.at[group_ids].set(tgt_sorted)
        unique_valid = jnp.zeros((chunk,), dtype=jnp.bool_)
        unique_valid = unique_valid.at[group_ids].set(valid_sorted)
        safe_targets = jnp.where(unique_valid, unique_targets, 0)
        reduced = jnp.where(unique_valid[:, None], reduced, 0.0)
        return local_accum.at[safe_targets].add(reduced), None

    accumulated, _ = jax.lax.scan(body, locals_coeffs, starts)
    return enforce_conjugate_symmetry_batch(accumulated, order=order)


def accumulate_octree_solidfmm_m2l(
    downward: OctreeSolidFMMDownwardPlan,
    multipoles: OctreeSolidFMMComplexMultipoles,
    *,
    chunk_size: int = 4096,
) -> OctreeSolidFMMDownwardPlan:
    """Accumulate octree-native complex M2L contributions into local buffers."""

    order = int(downward.order)
    if int(multipoles.order) != order:
        raise ValueError("octree downward and multipoles must use the same order")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")

    locals_packed = _accumulate_octree_m2l_complex_chunked(
        jnp.asarray(downward.locals_packed),
        jnp.asarray(multipoles.packed),
        jnp.asarray(downward.centers),
        jnp.asarray(downward.target_nodes, dtype=INDEX_DTYPE),
        jnp.asarray(downward.source_nodes, dtype=INDEX_DTYPE),
        jnp.asarray(downward.valid_interactions, dtype=jnp.bool_),
        order=order,
        chunk_size=int(chunk_size),
    )
    return OctreeSolidFMMDownwardPlan(
        order=downward.order,
        centers=downward.centers,
        locals_packed=locals_packed,
        parent=downward.parent,
        nodes_by_level=downward.nodes_by_level,
        level_offsets=downward.level_offsets,
        target_nodes=downward.target_nodes,
        source_nodes=downward.source_nodes,
        target_levels=downward.target_levels,
        interaction_level_offsets=downward.interaction_level_offsets,
        valid_interactions=downward.valid_interactions,
        num_pairs=downward.num_pairs,
    )


@partial(
    jax.jit,
    static_argnames=("order", "num_levels", "level_batch_width"),
)
def _propagate_octree_l2l_complex_by_level(
    locals_coeffs: Array,
    centers: Array,
    children: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    *,
    order: int,
    num_levels: int,
    level_batch_width: int,
) -> Array:
    """Propagate complex local expansions top-down over explicit octree levels."""

    if int(num_levels) <= 1:
        return locals_coeffs

    batch_width = int(max(level_batch_width, 1))
    level_offsets = jnp.asarray(level_offsets, dtype=INDEX_DTYPE)
    nodes_by_level = jnp.asarray(nodes_by_level, dtype=INDEX_DTYPE)
    level_slot = jnp.arange(batch_width, dtype=INDEX_DTYPE)
    child_slot = jnp.arange(int(children.shape[1]), dtype=INDEX_DTYPE)

    def level_body(level_idx: Array, state: Array) -> Array:
        start = level_offsets[level_idx]
        end = level_offsets[level_idx + 1]
        count = end - start
        batch_nodes = jax.lax.dynamic_slice_in_dim(
            nodes_by_level,
            start_index=start,
            slice_size=batch_width,
            axis=0,
        )
        valid_parent = level_slot < count
        safe_parents = jnp.where(valid_parent, batch_nodes, 0)
        child_idx = children[safe_parents]
        child_mask = valid_parent[:, None] & (child_idx >= 0)
        safe_child_idx = jnp.where(child_mask, child_idx, 0)

        parent_coeffs = state[safe_parents]
        parent_coeffs = jnp.broadcast_to(
            parent_coeffs[:, None, :],
            (batch_width, int(children.shape[1]), state.shape[1]),
        )
        # L2L delta convention is parent - child (opposite of M2M's child - parent);
        # l2l_complex is exact only with this sign.
        deltas = centers[safe_parents][:, None, :] - centers[safe_child_idx]
        translated = l2l_complex_batch(
            parent_coeffs.reshape(-1, state.shape[1]),
            deltas.reshape(-1, 3),
            order=order,
            rotation="solidfmm",
        ).reshape(batch_width, int(children.shape[1]), state.shape[1])
        translated = jnp.where(child_mask[..., None], translated, 0.0)

        flat_children = safe_child_idx.reshape(-1)
        flat_updates = translated.reshape(-1, state.shape[1])
        flat_valid = child_mask.reshape(-1)
        flat_children = jnp.where(flat_valid, flat_children, 0)
        flat_updates = jnp.where(flat_valid[:, None], flat_updates, 0.0)

        sort_idx = jnp.argsort(flat_children)
        tgt_sorted = flat_children[sort_idx]
        updates_sorted = flat_updates[sort_idx]
        valid_sorted = flat_valid[sort_idx]
        new_group = jnp.concatenate(
            (jnp.asarray([True], dtype=jnp.bool_), tgt_sorted[1:] != tgt_sorted[:-1]),
            axis=0,
        )
        group_ids = jnp.cumsum(new_group.astype(INDEX_DTYPE)) - jnp.asarray(
            1,
            dtype=INDEX_DTYPE,
        )
        reduced = jax.ops.segment_sum(
            updates_sorted,
            group_ids,
            batch_width * int(children.shape[1]),
        )
        unique_targets = jnp.zeros(
            (batch_width * int(children.shape[1]),), dtype=INDEX_DTYPE
        )
        unique_targets = unique_targets.at[group_ids].set(tgt_sorted)
        unique_valid = jnp.zeros(
            (batch_width * int(children.shape[1]),), dtype=jnp.bool_
        )
        unique_valid = unique_valid.at[group_ids].set(valid_sorted)
        safe_targets = jnp.where(unique_valid, unique_targets, 0)
        reduced = jnp.where(unique_valid[:, None], reduced, 0.0)
        return state.at[safe_targets].add(reduced)

    propagated = jax.lax.fori_loop(0, int(num_levels) - 1, level_body, locals_coeffs)
    return enforce_conjugate_symmetry_batch(propagated, order=order)


def propagate_octree_solidfmm_l2l(
    downward: OctreeSolidFMMDownwardPlan,
    octree: OctreeExecutionData,
) -> OctreeSolidFMMDownwardPlan:
    """Propagate octree local expansions top-down over the explicit child table."""

    order = int(downward.order)
    num_levels = int(octree.num_levels)
    active_level_offsets = jnp.asarray(
        octree.level_offsets[: num_levels + 1], dtype=INDEX_DTYPE
    )
    # Per-level dynamic_slice window: max per-level count from level_offsets, NOT
    # num_valid_nodes (total). See prepare_octree_solidfmm_complex_multipoles.
    level_batch_width = max(int(jnp.max(jnp.diff(active_level_offsets))), 1)
    locals_packed = _propagate_octree_l2l_complex_by_level(
        jnp.asarray(downward.locals_packed),
        jnp.asarray(downward.centers),
        jnp.asarray(octree.children, dtype=INDEX_DTYPE),
        jnp.asarray(octree.nodes_by_level, dtype=INDEX_DTYPE),
        active_level_offsets,
        order=order,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
    )
    return OctreeSolidFMMDownwardPlan(
        order=downward.order,
        centers=downward.centers,
        locals_packed=locals_packed,
        parent=downward.parent,
        nodes_by_level=downward.nodes_by_level,
        level_offsets=downward.level_offsets,
        target_nodes=downward.target_nodes,
        source_nodes=downward.source_nodes,
        target_levels=downward.target_levels,
        interaction_level_offsets=downward.interaction_level_offsets,
        valid_interactions=downward.valid_interactions,
        num_pairs=downward.num_pairs,
    )


def _prefix_mass_and_weighted_position(
    positions_sorted: Array,
    masses_sorted: Array,
) -> tuple[Array, Array]:
    masses = jnp.asarray(masses_sorted)
    positions = jnp.asarray(positions_sorted, dtype=masses.dtype)
    mass_prefix = jnp.concatenate(
        [jnp.zeros((1,), dtype=masses.dtype), jnp.cumsum(masses, axis=0)],
        axis=0,
    )
    weighted_prefix = jnp.concatenate(
        [
            jnp.zeros((1, 3), dtype=positions.dtype),
            jnp.cumsum(positions * masses[:, None], axis=0),
        ],
        axis=0,
    )
    return mass_prefix, weighted_prefix


def compute_octree_center_of_mass(
    plan: OctreeUpwardPlan,
    positions_sorted: Array,
    masses_sorted: Array,
) -> tuple[Array, Array]:
    """Compute per-octree-node mass and centre of mass from Morton ranges."""

    mass_prefix, weighted_prefix = _prefix_mass_and_weighted_position(
        positions_sorted,
        masses_sorted,
    )
    starts = jnp.clip(plan.node_ranges[:, 0], min=0)
    ends_inclusive = jnp.clip(plan.node_ranges[:, 1], min=-1)
    ends = jnp.clip(ends_inclusive + 1, min=0)
    valid = plan.valid_mask & (ends_inclusive >= 0)

    total_mass = mass_prefix[ends] - mass_prefix[starts]
    weighted_sum = weighted_prefix[ends] - weighted_prefix[starts]
    safe_mass = jnp.where(valid & (total_mass > 0), total_mass, 1.0)
    centers = weighted_sum / safe_mass[:, None]
    centers = jnp.where(valid[:, None], centers, 0.0)
    total_mass = jnp.where(valid, total_mass, 0.0)
    return total_mass, centers


@partial(
    jax.jit,
    static_argnames=("order", "max_leaf_size"),
)
def _p2m_octree_leaves_complex(
    leaf_nodes: Array,
    leaf_mask: Array,
    node_ranges: Array,
    positions_sorted: Array,
    masses_sorted: Array,
    centers: Array,
    *,
    order: int,
    max_leaf_size: int,
) -> Array:
    """Compute complex P2M coefficients for explicit octree leaves."""

    p = int(order)
    coeffs = sh_size(p)
    total_nodes = int(node_ranges.shape[0])
    dtype = complex_dtype_for_real(
        jnp.result_type(positions_sorted.dtype, masses_sorted.dtype)
    )
    packed = jnp.zeros((total_nodes, coeffs), dtype=dtype)

    leaf_nodes = jnp.asarray(leaf_nodes, dtype=INDEX_DTYPE)
    leaf_valid = jnp.asarray(leaf_mask, dtype=jnp.bool_)
    idx = jnp.arange(int(max_leaf_size), dtype=INDEX_DTYPE)

    def leaf_coeff(node_idx: Array, valid: Array) -> Array:
        ranges = node_ranges[node_idx]
        start = ranges[0]
        end = ranges[1]
        count = end - start + 1
        particle_idx = start + idx
        valid_particle = valid & (idx < count)
        safe_idx = jnp.clip(particle_idx, 0, positions_sorted.shape[0] - 1)
        pos = positions_sorted[safe_idx]
        pos = jnp.where(valid_particle[:, None], pos, 0.0)
        masses = masses_sorted[safe_idx]
        masses = jnp.where(valid_particle, masses, 0.0)
        contribs = p2m_complex_batch(pos - centers[node_idx], masses, order=p)
        coeff = jnp.sum(contribs, axis=0).astype(dtype)
        coeff = enforce_conjugate_symmetry_batch(coeff[None, :], order=p)[0]
        return jnp.where(valid, coeff, 0.0)

    coeffs_by_leaf = jax.vmap(leaf_coeff)(jnp.clip(leaf_nodes, 0), leaf_valid)
    # Scatter invalid leaves out of range (total_nodes) with mode="drop" so they never
    # collide with a real node's write at index 0 (no reserved dead node needed).
    safe_leaf_nodes = jnp.where(leaf_valid, leaf_nodes, total_nodes)
    return packed.at[safe_leaf_nodes].set(coeffs_by_leaf, mode="drop")


@partial(
    jax.jit,
    static_argnames=("order", "num_levels", "level_batch_width"),
)
def _aggregate_octree_m2m_complex_by_level(
    packed: Array,
    centers: Array,
    children: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    *,
    order: int,
    num_levels: int,
    level_batch_width: int,
) -> Array:
    """Aggregate octree multipoles level-by-level in explicit octree node space."""

    p = int(order)
    if int(num_levels) <= 1:
        return packed

    batch_width = int(max(level_batch_width, 1))
    total_nodes = int(packed.shape[0])
    level_offsets = jnp.asarray(level_offsets, dtype=INDEX_DTYPE)
    nodes_by_level = jnp.asarray(nodes_by_level, dtype=INDEX_DTYPE)
    level_slot = jnp.arange(batch_width, dtype=INDEX_DTYPE)

    def _translate_child(
        coeffs: Array, child_center: Array, parent_center: Array
    ) -> Array:
        delta = child_center - parent_center
        return m2m_complex(coeffs, delta, order=p, rotation="solidfmm").astype(
            packed.dtype
        )

    translate_children = jax.vmap(
        jax.vmap(_translate_child, in_axes=(0, 0, None)),
        in_axes=(0, 0, 0),
    )

    def level_body(level_rev_idx: Array, state: Array) -> Array:
        level_idx = as_index((num_levels - 2) - level_rev_idx)
        start = level_offsets[level_idx]
        end = level_offsets[level_idx + 1]
        count = end - start
        batch_nodes = jax.lax.dynamic_slice_in_dim(
            nodes_by_level,
            start_index=start,
            slice_size=batch_width,
            axis=0,
        )
        valid = level_slot < count
        # Gather with invalid batch slots clamped to index 0 (their values are masked
        # out below). The SCATTER instead sends invalid slots out of range
        # (total_nodes) with mode="drop", so a padding slot can never clobber a real
        # node's write -- in particular the root at index 0. This is what lets the
        # kernel run on the natural node layout (root at 0) with no reserved dead node.
        safe_nodes = jnp.where(valid, batch_nodes, 0)
        child_idx = children[safe_nodes]
        has_children = jnp.any(child_idx >= 0, axis=1)
        child_mask = (valid & has_children)[:, None] & (child_idx >= 0)
        safe_child_idx = jnp.where(child_mask, child_idx, 0)

        child_coeffs = state[safe_child_idx]
        child_centers = centers[safe_child_idx]
        parent_centers = centers[safe_nodes]
        translated = translate_children(child_coeffs, child_centers, parent_centers)
        translated = jnp.where(child_mask[..., None], translated, 0.0)
        node_coeffs = jnp.sum(translated, axis=1, dtype=translated.dtype)
        node_coeffs = enforce_conjugate_symmetry_batch(node_coeffs, order=p)
        current = state[safe_nodes]
        updates = jnp.where((valid & has_children)[:, None], node_coeffs, current)
        scatter_nodes = jnp.where(valid, batch_nodes, total_nodes)
        return state.at[scatter_nodes].set(updates, mode="drop")

    return jax.lax.fori_loop(0, int(num_levels) - 1, level_body, packed)


def prepare_octree_solidfmm_complex_multipoles(
    plan: OctreeUpwardPlan,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    max_order: int,
) -> OctreeSolidFMMComplexMultipoles:
    """Build octree-native solidfmm complex multipoles in explicit octree space."""

    total_mass, centers = compute_octree_center_of_mass(
        plan,
        positions_sorted,
        masses_sorted,
    )
    del total_mass

    leaf_nodes = jnp.asarray(plan.leaf_nodes, dtype=INDEX_DTYPE)
    leaf_valid = leaf_nodes >= 0
    safe_leaf_nodes = jnp.where(leaf_valid, leaf_nodes, 0)
    leaf_ranges = plan.node_ranges[safe_leaf_nodes]
    leaf_counts = jnp.where(leaf_valid, leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1, 0)
    max_leaf_size = max(int(jnp.max(leaf_counts)), 1)

    packed = _p2m_octree_leaves_complex(
        leaf_nodes,
        leaf_valid,
        plan.node_ranges,
        positions_sorted,
        masses_sorted,
        centers,
        order=int(max_order),
        max_leaf_size=max_leaf_size,
    )

    num_levels = int(plan.num_levels)
    active_level_offsets = jnp.asarray(
        plan.level_offsets[: num_levels + 1], dtype=INDEX_DTYPE
    )
    # level_batch_width is the per-level dynamic_slice window for the M2M level loop:
    # it MUST be the max per-level node count (from level_offsets), NOT num_valid_nodes
    # (the total). Using the total misaligns the front-anchored per-level reads for any
    # tree deeper than 2 levels (root+leaves), silently corrupting the M2M aggregation.
    level_batch_width = max(int(jnp.max(jnp.diff(active_level_offsets))), 1)
    packed = _aggregate_octree_m2m_complex_by_level(
        packed,
        centers,
        plan.children,
        plan.nodes_by_level,
        active_level_offsets,
        order=int(max_order),
        num_levels=num_levels,
        level_batch_width=level_batch_width,
    )

    return OctreeSolidFMMComplexMultipoles(
        order=int(max_order),
        centers=centers,
        packed=packed,
    )


__all__ = [
    "OctreeSolidFMMDownwardPlan",
    "OctreeInteractionPlan",
    "OctreeSolidFMMComplexMultipoles",
    "accumulate_octree_solidfmm_m2l",
    "build_octree_downward_plan",
    "OctreeUpwardPlan",
    "build_octree_interaction_plan",
    "build_octree_interaction_plan_from_native_pairs",
    "build_octree_upward_plan",
    "compute_octree_center_of_mass",
    "propagate_octree_solidfmm_l2l",
    "prepare_octree_solidfmm_complex_multipoles",
]
