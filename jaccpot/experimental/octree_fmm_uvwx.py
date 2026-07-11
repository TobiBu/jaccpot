"""Standalone O(N) octree FMM (uniform depth, U/V interaction lists).

yggdrax builds the uniform-depth Morton octree, the U/V interaction lists, and the
kernel-ready execution view (``build_uniform_octree_execution_view``); jaccpot runs the
FMM operators (P2M/M2M/M2L/L2L from ``runtime/_octree_fmm``), evaluates the resulting
local expansions on octree node space, and adds the U-list near-field P2P. The result
matches direct N-body to expansion-order tolerance.

This is the O(N) counterpart to the O(N log N) per-leaf treecode
(``experimental/treecode_walk`` / ``pallas/treecode_walk_pallas``): the octree shares
far-field work through an M2L-into-internal-nodes + L2L cascade instead of redoing every
far interaction per leaf. It runs on the NATURAL node layout (root at index 0, no
reserved sentinel node) enabled by the collision-free-scatter + derived-batch-width
kernel fixes in ``runtime/_octree_fmm``.

Near-field note: the U-list P2P here is a straightforward vectorized reference kernel
(pads to the max U-list width x max leaf size). It is O(N) for bounded occupancy but
materializes a dense per-leaf tensor; a production large-N path should route the U-list
through the optimized ``compute_leaf_p2p_accelerations`` near kernel instead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from yggdrax.octree_uvwx import (
    UniformOctreeExecutionView,
    build_uniform_octree_execution_view,
)

from ..downward.local_expansions import LocalExpansionData
from ..runtime._fmm_impl import _evaluate_local_expansions_for_particles
from ..runtime._octree_adapter import OctreeExecutionData
from ..runtime._octree_fmm import (
    accumulate_octree_solidfmm_m2l,
    build_octree_downward_plan,
    build_octree_interaction_plan_from_native_pairs,
    build_octree_upward_plan,
    prepare_octree_solidfmm_complex_multipoles,
    propagate_octree_solidfmm_l2l,
)

try:
    from ..runtime.dtypes import INDEX_DTYPE
except Exception:  # pragma: no cover - dtype module always present in practice
    INDEX_DTYPE = jnp.int64

try:
    from yggdrax.interactions import CompactTaggedOctreeFarPairs
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "octree_fmm_uvwx requires yggdrax.interactions.CompactTaggedOctreeFarPairs"
    ) from exc


def octree_execution_data_from_view(
    view: UniformOctreeExecutionView,
) -> OctreeExecutionData:
    """Wrap a yggdrax uniform-octree execution view in jaccpot's OctreeExecutionData.

    The view carries the tree topology + geometry + interaction lists (yggdrax owns tree
    construction); this adapter only fills the jaccpot-execution fields that the FMM
    operators do not use for the U/V path -- identity radix<->oct maps and zero box
    geometry (the operators use centre-of-mass centres computed during the upward pass,
    not box centres).
    """
    num_nodes = int(view.parent.shape[0])
    idx = jnp.arange(num_nodes, dtype=INDEX_DTYPE)
    zeros_n3 = jnp.zeros((num_nodes, 3))
    return OctreeExecutionData(
        valid_mask=jnp.asarray(view.valid_mask, dtype=bool),
        parent=jnp.asarray(view.parent, dtype=INDEX_DTYPE),
        children=jnp.asarray(view.children, dtype=INDEX_DTYPE),
        child_counts=jnp.asarray(view.child_counts, dtype=INDEX_DTYPE),
        node_codes=jnp.zeros(num_nodes, dtype=jnp.uint64),
        node_depths=jnp.asarray(view.node_depths, dtype=INDEX_DTYPE),
        node_ranges=jnp.asarray(view.node_ranges, dtype=INDEX_DTYPE),
        nodes_by_level=jnp.asarray(view.nodes_by_level, dtype=INDEX_DTYPE),
        level_offsets=jnp.asarray(view.level_offsets, dtype=INDEX_DTYPE),
        num_levels=jnp.asarray(int(view.num_levels), dtype=INDEX_DTYPE),
        leaf_mask=jnp.asarray(view.leaf_mask, dtype=bool),
        leaf_nodes=jnp.asarray(view.leaf_nodes, dtype=INDEX_DTYPE),
        radix_node_to_oct=idx,
        radix_leaf_to_oct=jnp.asarray(view.leaf_indices, dtype=INDEX_DTYPE),
        oct_to_radix_node=idx,
        oct_to_radix_leaf=idx,
        num_valid_nodes=jnp.asarray(int(view.num_valid_nodes), dtype=INDEX_DTYPE),
        num_leaf_nodes=jnp.asarray(int(view.num_leaf_nodes), dtype=INDEX_DTYPE),
        box_centers=jnp.asarray(view.centers),
        box_half_extents=zeros_n3,
        box_radii=jnp.zeros(num_nodes),
        box_max_extents=jnp.zeros(num_nodes),
    )


def _octree_far_field_grad(
    view: UniformOctreeExecutionView,
    octree: OctreeExecutionData,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    order: int,
    max_leaf_size: int,
) -> Array:
    """O(N) far field: upward (P2M/M2M) -> V-list M2L -> L2L -> evaluate locals."""
    plan = build_octree_upward_plan(octree)
    multipoles = prepare_octree_solidfmm_complex_multipoles(
        plan, positions_sorted, masses_sorted, max_order=int(order)
    )
    far_pairs = CompactTaggedOctreeFarPairs(
        sources=jnp.asarray(view.v_src, dtype=INDEX_DTYPE),
        targets=jnp.asarray(view.v_tgt, dtype=INDEX_DTYPE),
        tags=jnp.zeros(int(view.v_src.shape[0]), dtype=INDEX_DTYPE),
    )
    interactions = build_octree_interaction_plan_from_native_pairs(octree, far_pairs)
    downward = build_octree_downward_plan(octree, multipoles, interactions)
    downward = accumulate_octree_solidfmm_m2l(downward, multipoles, chunk_size=1024)
    downward = propagate_octree_solidfmm_l2l(downward, octree)

    local_data = LocalExpansionData(
        order=int(order),
        centers=downward.centers,
        coefficients=downward.locals_packed,
    )
    grad = _evaluate_local_expansions_for_particles(
        local_data,
        positions_sorted,
        leaf_nodes=jnp.asarray(view.leaf_indices, dtype=INDEX_DTYPE),
        node_ranges=jnp.asarray(view.node_ranges, dtype=INDEX_DTYPE),
        max_leaf_size=int(max_leaf_size),
        order=int(order),
        expansion_basis="solidfmm",
        return_potential=False,
    )[0]
    return grad


def _ulist_near_accelerations(
    positions_sorted: Array,
    masses_sorted: Array,
    view: UniformOctreeExecutionView,
    *,
    G: float,
    softening: float,
    max_leaf_size: int,
) -> Array:
    """Vectorized U-list near-field P2P (self-particle excluded), softened Plummer.

    Pads each leaf's U-list to the max width and each source/target leaf to
    ``max_leaf_size``; masks padding + the self particle. O(N) for bounded occupancy.
    """
    leaf_indices = np.asarray(view.leaf_indices, np.int64)
    node_ranges = np.asarray(view.node_ranges, np.int64)
    u_offsets = np.asarray(view.u_offsets, np.int64)
    u_neighbors = np.asarray(view.u_neighbors, np.int64)
    num_leaves = int(leaf_indices.shape[0])
    ml = int(max_leaf_size)

    widths = np.diff(u_offsets)
    max_width = int(widths.max()) if num_leaves else 1

    # Per-leaf padded source-node table (-1 = pad) and target ranges.
    src_nodes = np.full((num_leaves, max_width), -1, np.int64)
    for r in range(num_leaves):
        row = u_neighbors[u_offsets[r] : u_offsets[r + 1]]
        src_nodes[r, : row.shape[0]] = row
    tgt_starts = node_ranges[leaf_indices, 0]
    tgt_counts = node_ranges[leaf_indices, 1] - node_ranges[leaf_indices, 0] + 1
    # padded source ranges indexed by node id (append a pad entry for -1)
    src_starts_by_node = np.append(node_ranges[:, 0], 0)
    src_counts_by_node = np.append(node_ranges[:, 1] - node_ranges[:, 0] + 1, 0)

    n_part = int(positions_sorted.shape[0])
    pos = positions_sorted
    mass = masses_sorted
    soft2 = float(softening) ** 2

    src_nodes_j = jnp.asarray(src_nodes, dtype=INDEX_DTYPE)
    tgt_starts_j = jnp.asarray(tgt_starts, dtype=INDEX_DTYPE)
    tgt_counts_j = jnp.asarray(tgt_counts, dtype=INDEX_DTYPE)
    src_starts_j = jnp.asarray(src_starts_by_node, dtype=INDEX_DTYPE)
    src_counts_j = jnp.asarray(src_counts_by_node, dtype=INDEX_DTYPE)
    local = jnp.arange(ml, dtype=INDEX_DTYPE)

    def leaf_near(leaf_args):
        tgt_start, tgt_count, src_node_row = leaf_args
        # target particles [tgt_start, tgt_start+ml), masked to tgt_count
        tgt_pidx = tgt_start + local
        tgt_valid = local < tgt_count
        tgt_safe = jnp.clip(tgt_pidx, 0, n_part - 1)
        tgt_pos = pos[tgt_safe]  # (ml, 3)

        # pad node id -1 -> last table entry (count 0) so it contributes nothing
        src_node_safe = jnp.where(
            src_node_row >= 0, src_node_row, src_starts_j.shape[0] - 1
        )
        src_start = src_starts_j[src_node_safe]  # (W,)
        src_count = src_counts_j[src_node_safe]  # (W,)
        src_pidx = src_start[:, None] + local[None, :]  # (W, ml)
        src_valid = (local[None, :] < src_count[:, None]) & (src_node_row >= 0)[:, None]
        src_safe = jnp.clip(src_pidx, 0, n_part - 1)
        src_pos = pos[src_safe]  # (W, ml, 3)
        src_mass = mass[src_safe]  # (W, ml)

        # pairwise: target t vs source (w, s)
        d = tgt_pos[:, None, None, :] - src_pos[None, :, :, :]  # (ml, W, ml, 3)
        d2 = jnp.sum(d * d, axis=-1) + soft2  # (ml, W, ml)
        # exclude the self particle (same global index)
        same = tgt_pidx[:, None, None] == src_pidx[None, :, :]
        pair_valid = tgt_valid[:, None, None] & src_valid[None, :, :] & (~same)
        inv = jnp.where(pair_valid, d2 ** (-1.5), 0.0)  # (ml, W, ml)
        contrib = src_mass[None, :, :, None] * d * inv[..., None]  # (ml, W, ml, 3)
        acc_t = -float(G) * jnp.sum(contrib, axis=(1, 2))  # (ml, 3)
        return tgt_pidx, tgt_valid, acc_t

    # lax.map (sequential over leaves) instead of vmap: a uniform octree on clustered
    # data has one overcrowded centre leaf (large max_leaf_size), so materializing the
    # per-leaf (ml, W, ml, 3) tensor for ALL leaves at once OOMs. Processing one leaf at
    # a time bounds peak memory to a single leaf's tensor.
    tgt_pidx_all, tgt_valid_all, acc_all = jax.lax.map(
        leaf_near, (tgt_starts_j, tgt_counts_j, src_nodes_j)
    )
    # scatter per-leaf target accelerations back to a (n_part, 3) buffer
    flat_idx = jnp.clip(tgt_pidx_all.reshape(-1), 0, n_part - 1)
    flat_acc = acc_all.reshape(-1, 3)
    flat_valid = tgt_valid_all.reshape(-1)
    flat_acc = jnp.where(flat_valid[:, None], flat_acc, 0.0)
    near = jnp.zeros((n_part, 3), dtype=flat_acc.dtype)
    near = near.at[flat_idx].add(flat_acc)
    return near


def octree_fmm_accelerations(
    positions: Array,
    masses: Array,
    *,
    depth: int,
    order: int,
    G: float = 1.0,
    softening: float = 1e-2,
    return_view: bool = False,
):
    """O(N) octree-FMM gravitational accelerations (far V-list + near U-list P2P).

    ``positions`` (N, 3), ``masses`` (N,); returns accelerations in the SAME order as the
    inputs. ``depth`` = uniform octree levels, ``order`` = multipole expansion order.
    """
    positions = jnp.asarray(positions)
    masses = jnp.asarray(masses)
    pos_host = np.asarray(positions)

    view = build_uniform_octree_execution_view(pos_host, int(depth))
    octree = octree_execution_data_from_view(view)
    perm = jnp.asarray(view.perm, dtype=INDEX_DTYPE)
    positions_sorted = positions[perm]
    masses_sorted = masses[perm]

    counts = (
        view.node_ranges[view.leaf_indices, 1]
        - view.node_ranges[view.leaf_indices, 0]
        + 1
    )
    max_leaf_size = max(int(counts.max()), 1)

    far_grad = _octree_far_field_grad(
        view,
        octree,
        positions_sorted,
        masses_sorted,
        order=int(order),
        max_leaf_size=max_leaf_size,
    )
    far_acc = -float(G) * far_grad
    near_acc = _ulist_near_accelerations(
        positions_sorted,
        masses_sorted,
        view,
        G=G,
        softening=softening,
        max_leaf_size=max_leaf_size,
    )
    acc_sorted = far_acc + near_acc

    # invert the Morton permutation back to input order
    inv = (
        jnp.zeros_like(perm).at[perm].set(jnp.arange(perm.shape[0], dtype=INDEX_DTYPE))
    )
    acc = acc_sorted[inv]
    if return_view:
        return acc, view
    return acc


__all__ = [
    "octree_execution_data_from_view",
    "octree_fmm_accelerations",
]
