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

``device=True`` (default) builds the octree + U/V lists fully on-device with STATIC
shapes (yggdrax ``build_uniform_octree_execution_view_device``: a dense node space of
size ``(8^(L+1)-1)/7`` + fixed-capacity lists) and runs the near field on the optimized
device P2P kernel (``_compute_leaf_p2p_impl``). For a fixed ``(N, depth, order,
max_leaf_capacity)`` nothing recompiles when only positions/masses change -- the inner
FMM kernels' static args (``num_levels=L+1``, ``level_batch_width=8^L``, node count) are
functions of ``depth`` alone -- so it can be reused across ODISSEO time-integration steps
without retracing. ``device=False`` uses the host-numpy reference build + a sequential
``lax.map`` near field. Both give identical forces.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from yggdrax.octree_uvwx import (
    UniformOctreeExecutionView,
    build_sparse_uniform_octree_execution_view_device,
    build_uniform_octree_execution_view,
    build_uniform_octree_execution_view_device,
)

from ..downward.local_expansions import LocalExpansionData
from ..nearfield.near_field import (
    _compute_leaf_p2p_impl,
    _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl,
    _compute_leaf_p2p_prepared_large_n_self_only_impl,
    _radix_fast_lane_prepacked_pallas,
)
from ..runtime._fmm_impl import (
    _accumulate_solidfmm_m2l_chunked_scan,
    _evaluate_local_expansions_for_particles,
)
from ..runtime._octree_adapter import OctreeExecutionData
from ..runtime._octree_fmm import (
    _propagate_octree_l2l_complex_by_level,
    build_octree_upward_plan,
    prepare_octree_solidfmm_complex_multipoles,
)

try:
    from ..runtime.dtypes import INDEX_DTYPE
except Exception:  # pragma: no cover - dtype module always present in practice
    INDEX_DTYPE = jnp.int64


def octree_execution_data_from_view(
    view: UniformOctreeExecutionView,
    *,
    num_levels: Optional[int] = None,
) -> OctreeExecutionData:
    """Wrap a yggdrax uniform-octree execution view in jaccpot's OctreeExecutionData.

    The view carries the tree topology + geometry + interaction lists (yggdrax owns tree
    construction); this adapter only fills the jaccpot-execution fields that the FMM
    operators do not use for the U/V path -- identity radix<->oct maps and zero box
    geometry (the operators use centre-of-mass centres computed during the upward pass,
    not box centres).

    ``num_levels`` is the static level count. Leave ``None`` to read it from the view
    (works when the view is built INSIDE the same jit, so its num_levels is a python-int
    constant). Pass it explicitly (= max_depth+1) for STAGED compilation, where the view
    arrays are handed to a SEPARATE jit as traced args -- then ``view.num_levels`` is a
    traced scalar and ``int(...)`` on it would raise. Staging the build and the FMM into
    two jits (instead of one monolithic graph) avoids the pathological fused compile.
    """
    if num_levels is None:
        num_levels = int(view.num_levels)
    else:
        num_levels = int(num_levels)
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
        # STATIC python int (resolved above): num_levels indexes static slices
        # (level_offsets[:num_levels+1]) and jnp.zeros((num_levels,)) shapes, and is a
        # kernel static_argname -- must stay a python constant (never a traced value).
        num_levels=num_levels,
        leaf_mask=jnp.asarray(view.leaf_mask, dtype=bool),
        leaf_nodes=jnp.asarray(view.leaf_nodes, dtype=INDEX_DTYPE),
        radix_node_to_oct=idx,
        radix_leaf_to_oct=jnp.asarray(view.leaf_indices, dtype=INDEX_DTYPE),
        oct_to_radix_node=idx,
        oct_to_radix_leaf=idx,
        # NB: kept as (possibly traced) arrays, not int(...): for the SPARSE build these
        # are data-dependent jnp.sum results, so int() would break the fused jit. Nothing
        # downstream needs them as static ints (num_levels carries the static level count).
        num_valid_nodes=jnp.asarray(view.num_valid_nodes, dtype=INDEX_DTYPE),
        num_leaf_nodes=jnp.asarray(view.num_leaf_nodes, dtype=INDEX_DTYPE),
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
    num_levels: Optional[int] = None,
    level_batch_width: Optional[int] = None,
    m2l_chunk_size: int = 4096,
    v_active_capacity: Optional[int] = None,
) -> Array:
    """O(N) far field: upward (P2M/M2M) -> V-list M2L -> L2L -> evaluate locals.

    ``num_levels`` / ``level_batch_width`` are static (from tree depth) so the M2M/L2L
    kernels' ``static_argnames`` stay fixed -- the whole far field runs inside one jit.
    Pass them for the DENSE device build (num_levels=depth+1, level_batch_width=8**depth);
    leave ``None`` for the compact host build so they are derived from its actual per-level
    node counts (the dense 8**depth would over-run the compact node arrays).

    M2L runs as a fixed-chunk scatter-add scan (radix ``_accumulate_solidfmm_m2l_chunked_scan``)
    DIRECTLY over the yggdrax V-list pairs -- no global interaction plan / lexsort. The V-list
    is padded to ``node_capacity * v_capacity`` slots (~3.45M @200k, ~4x the ~800k real pairs),
    so ``v_active_capacity`` (static) STREAM-COMPACTS the real pairs into a fixed buffer of
    that size (sentinel -> -1, masked by the kernel) -- the M2L scan then iterates
    ``v_active_capacity / m2l_chunk_size`` chunks instead of the full padded count, which is
    the dominant far-field cost at 200k. Leave ``None`` to skip compaction (argsort real-first
    over the full padded array; correct but slow). ``v_active_capacity`` must be >= the true
    pair count or the excess pairs are silently dropped.
    """
    plan = build_octree_upward_plan(octree)
    multipoles = prepare_octree_solidfmm_complex_multipoles(
        plan,
        positions_sorted,
        masses_sorted,
        max_order=int(order),
        max_leaf_size=int(max_leaf_size),
        num_levels=num_levels,
        level_batch_width=level_batch_width,
    )
    # M2L: route the yggdrax V-list pairs through the RADIX fast-lane's streamed complex
    # M2L kernel (_accumulate_solidfmm_m2l_chunked_scan) -- same solidfmm math as the
    # octree-native kernel but with active_pair_count chunk-skipping + the efficient
    # _chunk_segment_scatter_add. The octree-native _accumulate_octree_m2l_complex_chunked
    # was ~40000x slower/pair (full per-chunk argsort, no chunk-skip). Compact the real
    # pairs to the front (sentinel node_capacity -> -1, which the kernel masks) so the
    # chunk-skip actually engages and the padded tail costs ~nothing.
    num_nodes = int(view.parent.shape[0])
    v_src = jnp.asarray(view.v_src, dtype=INDEX_DTYPE)
    v_tgt = jnp.asarray(view.v_tgt, dtype=INDEX_DTYPE)
    valid_pairs = (v_src < num_nodes) & (v_tgt < num_nodes)
    n_valid = jnp.sum(valid_pairs.astype(INDEX_DTYPE))
    if v_active_capacity is None:
        # No compaction: sort real pairs to the front over the full padded array.
        pair_perm = jnp.argsort(jnp.logical_not(valid_pairs))
        src_pairs = jnp.where(valid_pairs, v_src, -1)[pair_perm]
        tgt_pairs = jnp.where(valid_pairs, v_tgt, -1)[pair_perm]
        active_pairs = n_valid
    else:
        # Stream-compact the real pairs into a fixed v_active_capacity buffer (cumsum
        # scatter, cheaper than argsort); the M2L scan then only touches ~real/chunk chunks.
        cap = int(v_active_capacity)
        dest = jnp.where(
            valid_pairs, jnp.cumsum(valid_pairs.astype(INDEX_DTYPE)) - 1, cap
        )
        src_pairs = (
            jnp.full((cap,), -1, dtype=INDEX_DTYPE).at[dest].set(v_src, mode="drop")
        )
        tgt_pairs = (
            jnp.full((cap,), -1, dtype=INDEX_DTYPE).at[dest].set(v_tgt, mode="drop")
        )
        active_pairs = jnp.minimum(n_valid, cap)
    locals_packed = _accumulate_solidfmm_m2l_chunked_scan(
        jnp.zeros_like(multipoles.packed),
        jnp.asarray(multipoles.packed),
        jnp.asarray(multipoles.centers),
        src_pairs,
        tgt_pairs,
        active_pairs,
        order=int(order),
        rotation="solidfmm",
        total_nodes=num_nodes,
        chunk_size=int(m2l_chunk_size),
    )
    # L2L level cascade over the octree child table (parent - child delta; the complex
    # kernel is exact only with that sign). Derive the static level count / per-level
    # dynamic_slice window when not supplied (host build), matching
    # propagate_octree_solidfmm_l2l.
    l2l_levels = int(octree.num_levels) if num_levels is None else int(num_levels)
    active_level_offsets = jnp.asarray(
        octree.level_offsets[: l2l_levels + 1], dtype=INDEX_DTYPE
    )
    if level_batch_width is None:
        l2l_batch_width = max(int(jnp.max(jnp.diff(active_level_offsets))), 1)
    else:
        l2l_batch_width = int(level_batch_width)
    locals_packed = _propagate_octree_l2l_complex_by_level(
        locals_packed,
        jnp.asarray(multipoles.centers),
        jnp.asarray(octree.children, dtype=INDEX_DTYPE),
        jnp.asarray(octree.nodes_by_level, dtype=INDEX_DTYPE),
        active_level_offsets,
        order=int(order),
        num_levels=l2l_levels,
        level_batch_width=l2l_batch_width,
    )

    local_data = LocalExpansionData(
        order=int(order),
        centers=jnp.asarray(multipoles.centers),
        coefficients=locals_packed,
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


def _ulist_near_device(
    positions_sorted: Array,
    masses_sorted: Array,
    view: UniformOctreeExecutionView,
    *,
    G: float,
    softening: float,
    max_leaf_size: int,
) -> Array:
    """Device U-list near-field P2P via the optimized ``_compute_leaf_p2p_impl``.

    Consumes the device build's fixed-width U-list (row ids, self-excluded, sentinel =
    ``num_nodes``) as a precomputed (target, source, valid) pair list, so the shapes are
    static (``num_leaves * 26``) and the kernel does its own memory-safe chunking. The
    kernel adds each leaf's intra-leaf self block once, matching the self-excluded U-list.
    """
    num_nodes = int(view.parent.shape[0])
    num_leaves = int(view.leaf_indices.shape[0])
    # CSR width = fixed per-leaf near-list capacity. 26 for the uniform/sparse builds
    # (colleague-only U list); the adaptive build's extended-near (U + W + X) uses a wider
    # capacity, so derive it from the stored list rather than hardcoding 26.
    u_cap = int(view.u_neighbors.shape[0] // max(num_leaves, 1))
    src_rows_raw = jnp.asarray(view.u_neighbors, dtype=INDEX_DTYPE)
    valid_pairs = src_rows_raw < num_leaves  # sentinel (num_nodes) and pads excluded
    source_leaf_ids = jnp.where(valid_pairs, src_rows_raw, jnp.asarray(0, INDEX_DTYPE))
    target_leaf_ids = jnp.repeat(jnp.arange(num_leaves, dtype=INDEX_DTYPE), u_cap)
    empty = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
    return _compute_leaf_p2p_impl(
        jnp.asarray(view.node_ranges, dtype=INDEX_DTYPE),
        jnp.asarray(view.leaf_indices, dtype=INDEX_DTYPE),
        jnp.asarray(view.u_offsets, dtype=INDEX_DTYPE),
        source_leaf_ids,
        positions_sorted,
        masses_sorted,
        target_leaf_ids,
        source_leaf_ids,
        valid_pairs,
        empty,
        empty,
        empty,
        int(max_leaf_size),
        G=G,
        softening_sq=jnp.asarray(softening, dtype=positions_sorted.dtype) ** 2,
        return_potential=False,
        collect_neighbor_pairs=False,
        nearfield_mode="baseline",
        edge_chunk_size=256,
        use_precomputed_scatter=False,
    )


def _octree_near_field(
    positions_sorted: Array,
    masses_sorted: Array,
    view: UniformOctreeExecutionView,
    *,
    G: float,
    softening: float,
    max_leaf_size: int,
    use_pallas: bool = True,
    pallas_interpret: bool = False,
    near_block_size: Optional[int] = 128,
) -> Array:
    """U-list near-field P2P over fixed-width UNITS, reusing the radix fast-lane machinery.

    Packs particles into units, builds each unit's source-unit id list, then adds the
    intra-unit self block (``i != j``, via ``_compute_leaf_p2p_prepared_large_n_self_only_impl``)
    plus the cross-unit pairs via the Pallas ``nearfield_leafpair`` kernel
    (``_radix_fast_lane_prepacked_pallas``, gathers sources by id + cheaply cond-skips padded
    slots) when ``use_pallas`` + a supported accelerator, else the pure-JAX prepacked path
    (autodiff-able; the differentiable fallback). Result is in Morton (sorted) order.

    ``near_block_size`` selects the unit:

    * ``None`` -> one unit per leaf, width ``max_leaf_size``. An adaptive octree's leaves are
      badly under-full (8-way split: mean occ << ``max_leaf_size``), so every leaf pads to the
      global max occupancy -- the ``W^2`` particle term is dominated by padding.
    * ``B`` (default 128) -> DENSE-BLOCK packing: each leaf is chunked into ``ceil(occ/B)``
      fixed-``B`` blocks (respecting leaf boundaries), and blocks are the units (width ``B``,
      not max occ). A block's sources = its leaf's OTHER blocks (intra-leaf near) + its
      U-neighbour leaves' blocks. Decoupling the width from max occupancy cuts the near-field
      ~2x on the 200k disk (leaf occ mean ~34 vs max 256). Block count is bounded statically by
      ``ceil(N/B) + num_leaves``.
    """
    num_nodes = int(view.parent.shape[0])
    num_leaves = int(view.leaf_indices.shape[0])
    n_part = int(positions_sorted.shape[0])
    u_cap = int(view.u_neighbors.shape[0] // max(num_leaves, 1))
    dtype = positions_sorted.dtype
    softening_sq = jnp.asarray(softening, dtype=dtype) ** 2  # traced-safe (no float())

    # ---- leaf geometry from node ranges (shared by both unit modes) ----
    leaf_node = jnp.asarray(view.leaf_indices, dtype=INDEX_DTYPE)
    node_ranges = jnp.asarray(view.node_ranges, dtype=INDEX_DTYPE)
    valid_leaf = leaf_node < num_nodes
    leaf_node_safe = jnp.where(valid_leaf, leaf_node, 0)
    lo = node_ranges[leaf_node_safe, 0]
    occ = jnp.where(
        valid_leaf,
        node_ranges[leaf_node_safe, 1] - node_ranges[leaf_node_safe, 0] + 1,
        0,
    )
    src_rows = jnp.asarray(view.u_neighbors, dtype=INDEX_DTYPE).reshape(
        num_leaves, u_cap
    )

    if near_block_size is None:
        # one unit per leaf, width = max_leaf_size
        W = int(max_leaf_size)
        slot = jnp.arange(W, dtype=INDEX_DTYPE)
        unit_mask = (slot[None, :] < occ[:, None]) & valid_leaf[:, None]
        unit_pidx = jnp.where(
            unit_mask, jnp.clip(lo[:, None] + slot[None, :], 0, n_part - 1), 0
        )
        self_row = jnp.arange(num_leaves, dtype=INDEX_DTYPE)[:, None]
        s_valid = (src_rows < num_leaves) & (src_rows != self_row)
        src_ids_3d = jnp.where(s_valid, src_rows, 0)[:, None, :]  # (L, 1, u_cap)
        src_valid_3d = s_valid[:, None, :]
    else:
        # dense-block units: chunk each leaf into fixed-B blocks (width B, not max occ)
        B = int(near_block_size)
        block_cap = (n_part + B - 1) // B + num_leaves  # tight upper bound on #blocks
        mbpl = (int(max_leaf_size) + B - 1) // B  # static max blocks per leaf
        bpl = (occ + B - 1) // B  # blocks per leaf (0 if empty/invalid)
        cum = jnp.cumsum(bpl)
        block_start = cum - bpl
        total_blocks = jnp.sum(bpl)
        g = jnp.arange(block_cap, dtype=INDEX_DTYPE)
        block_leaf = jnp.clip(jnp.searchsorted(cum, g, side="right"), 0, num_leaves - 1)
        valid_block = g < total_blocks
        block_local = g - block_start[block_leaf]
        slot = jnp.arange(B, dtype=INDEX_DTYPE)
        within = block_local[:, None] * B + slot[None, :]
        unit_mask = valid_block[:, None] & (within < occ[block_leaf][:, None])
        unit_pidx = jnp.where(
            unit_mask, jnp.clip(lo[block_leaf][:, None] + within, 0, n_part - 1), 0
        )
        # sources: own leaf's other blocks (intra-leaf near) + U-neighbour leaves' blocks
        r = block_leaf
        nbrs = jnp.concatenate(
            [r[:, None], src_rows[r]], axis=1
        )  # (block_cap, 1+u_cap)
        nbr_valid = jnp.concatenate(
            [valid_block[:, None], src_rows[r] < num_leaves], axis=1
        )
        k = jnp.arange(mbpl, dtype=INDEX_DTYPE)
        sb = (
            block_start[nbrs][:, :, None] + k[None, None, :]
        )  # (block_cap, 1+u_cap, mbpl)
        sb_valid = (
            nbr_valid[:, :, None]
            & (k[None, None, :] < bpl[nbrs][:, :, None])
            & (sb != g[:, None, None])  # exclude the target block itself
        )
        n_src = (1 + u_cap) * mbpl
        src_ids_3d = jnp.where(sb_valid, jnp.clip(sb, 0, block_cap - 1), 0).reshape(
            block_cap, 1, n_src
        )
        src_valid_3d = sb_valid.reshape(block_cap, 1, n_src)

    unit_positions = positions_sorted[unit_pidx]
    unit_masses = masses_sorted[unit_pidx]
    num_units = int(unit_pidx.shape[0])

    self_acc = _compute_leaf_p2p_prepared_large_n_self_only_impl(
        positions_sorted,
        unit_positions,
        unit_masses,
        unit_mask,
        unit_pidx,
        G=G,
        softening_sq=softening_sq,
    )

    pallas_ok = False
    if bool(use_pallas):
        if bool(pallas_interpret):
            pallas_ok = True
        else:
            try:
                from ..pallas.nearfield_fused_leaf import (
                    pallas_nearfield_fused_supported,
                )

                pallas_ok = pallas_nearfield_fused_supported()
            except Exception:  # pragma: no cover - pallas import is env-dependent
                pallas_ok = False

    if pallas_ok:
        pair_acc = _radix_fast_lane_prepacked_pallas(
            src_ids_3d,
            src_valid_3d,
            unit_positions,
            unit_masses,
            unit_mask,
            unit_pidx,
            positions_sorted,
            G=G,
            softening_sq=softening_sq,
            compute_potential=False,
            interpret=bool(pallas_interpret),
        )
    else:
        pair_acc = (
            _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl(
                positions_sorted,
                src_ids_3d,
                src_valid_3d,
                unit_positions,
                unit_masses,
                unit_mask,
                unit_pidx,
                G=G,
                softening_sq=softening_sq,
                target_leaf_batch_size=min(num_units, 256),
                target_block_tile_size=8,
                target_block_tile_scan_unroll=1,
                target_block_batch_scan_unroll=1,
                occupancy_sort=True,
                skip_empty_tiles=True,
                componentwise_pairs=True,
            )
        )
    return self_acc + pair_acc


@partial(
    jax.jit,
    static_argnames=(
        "depth",
        "order",
        "max_leaf_size",
        "sparse",
        "node_capacity",
        "leaf_capacity",
        "near_use_pallas",
    ),
)
def _octree_uvwx_device_compute(
    positions: Array,
    masses: Array,
    G: Array,
    softening: Array,
    *,
    depth: int,
    order: int,
    max_leaf_size: int,
    sparse: bool = False,
    node_capacity: int = 0,
    leaf_capacity: int = 0,
    near_use_pallas: bool = True,
) -> Array:
    """Single jitted O(N) octree-FMM compute over the static-shape device build.

    Everything -- device octree + U/V build, upward/M2L/L2L, octree-node eval, U-list
    near P2P, perm inversion -- runs inside ONE jax.jit. The static args plus the
    static-shape build mean it compiles once and is reused for all subsequent calls at the
    same static config, regardless of the (traced) positions/masses/G/softening -- so no
    recompilation across integration steps.

    ``sparse=False`` uses the DENSE build (node count ``(8^(L+1)-1)/7``; only viable for
    small depth / near-uniform data); ``level_batch_width = 8**depth``. ``sparse=True`` uses
    the SPARSE-occupied build (only occupied cells, padded to ``node_capacity`` /
    ``leaf_capacity``; viable on concentrated data at the depth needed to bound leaf
    occupancy); ``level_batch_width = leaf_capacity``. ``num_levels = depth + 1`` either way.
    """
    num_levels = int(depth) + 1
    if sparse:
        view = build_sparse_uniform_octree_execution_view_device(
            positions,
            int(depth),
            node_capacity=int(node_capacity),
            leaf_capacity=int(leaf_capacity),
        )
        level_batch_width = int(leaf_capacity)
    else:
        view = build_uniform_octree_execution_view_device(positions, int(depth))
        level_batch_width = 8 ** int(depth)
    octree = octree_execution_data_from_view(view)
    perm = jnp.asarray(view.perm, dtype=INDEX_DTYPE)
    positions_sorted = positions[perm]
    masses_sorted = masses[perm]

    far_grad = _octree_far_field_grad(
        view,
        octree,
        positions_sorted,
        masses_sorted,
        order=int(order),
        max_leaf_size=int(max_leaf_size),
        num_levels=num_levels,
        level_batch_width=level_batch_width,
    )
    far_acc = -G * far_grad
    near_acc = _octree_near_field(
        positions_sorted,
        masses_sorted,
        view,
        G=G,
        softening=softening,
        max_leaf_size=int(max_leaf_size),
        use_pallas=bool(near_use_pallas),
    )
    acc_sorted = far_acc + near_acc
    inv = (
        jnp.zeros_like(perm).at[perm].set(jnp.arange(perm.shape[0], dtype=INDEX_DTYPE))
    )
    return acc_sorted[inv]


def octree_fmm_accelerations(
    positions: Array,
    masses: Array,
    *,
    depth: int,
    order: int,
    G: float = 1.0,
    softening: float = 1e-2,
    max_leaf_capacity: Optional[int] = None,
    device: bool = True,
    sparse: bool = False,
    node_capacity: Optional[int] = None,
    leaf_capacity: Optional[int] = None,
    bounds: Optional[tuple] = None,
    return_view: bool = False,
    near_use_pallas: bool = True,
):
    """O(N) octree-FMM gravitational accelerations (far V-list + near U-list P2P).

    ``positions`` (N, 3), ``masses`` (N,); returns accelerations in the SAME order as the
    inputs. ``depth`` = uniform octree levels, ``order`` = multipole expansion order.

    With ``device=True`` (default) the ENTIRE compute runs inside a single jax.jit over the
    static-shape device build (:func:`_octree_uvwx_device_compute`): for a fixed
    ``(N, depth, order, max_leaf_capacity)`` it compiles once and never recompiles when
    positions/masses/G/softening change -- ready to drop into an ODISSEO time-integration
    loop. Pass ``max_leaf_capacity`` (a static per-leaf particle cap) to fix that key;
    if ``None`` it is inferred once from the current occupancy (a host sync + an extra tree
    build) and changes to the max occupancy would recompile. ``device=False`` uses the
    host-numpy reference build (eager). ``bounds`` currently applies to ``device=False``
    only (the device path uses the per-step data extent).
    """
    positions = jnp.asarray(positions)
    masses = jnp.asarray(masses)

    if device:
        if sparse:
            if node_capacity is None or leaf_capacity is None:
                raise ValueError(
                    "sparse=True requires node_capacity and leaf_capacity (static caps)"
                )
            if max_leaf_capacity is None:
                raise ValueError(
                    "sparse=True requires max_leaf_capacity (>= max leaf occupancy)"
                )
            max_leaf_size = int(max_leaf_capacity)
            acc = _octree_uvwx_device_compute(
                positions,
                masses,
                jnp.asarray(G, dtype=positions.dtype),
                jnp.asarray(softening, dtype=positions.dtype),
                depth=int(depth),
                order=int(order),
                max_leaf_size=max_leaf_size,
                sparse=True,
                node_capacity=int(node_capacity),
                leaf_capacity=int(leaf_capacity),
                near_use_pallas=bool(near_use_pallas),
            )
            if return_view:
                return acc, build_sparse_uniform_octree_execution_view_device(
                    positions,
                    int(depth),
                    node_capacity=int(node_capacity),
                    leaf_capacity=int(leaf_capacity),
                )
            return acc

        if max_leaf_capacity is not None:
            max_leaf_size = int(max_leaf_capacity)
        else:
            probe = build_uniform_octree_execution_view_device(positions, int(depth))
            counts = (
                probe.node_ranges[probe.leaf_indices, 1]
                - probe.node_ranges[probe.leaf_indices, 0]
                + 1
            )
            max_leaf_size = max(int(counts.max()), 1)
        acc = _octree_uvwx_device_compute(
            positions,
            masses,
            jnp.asarray(G, dtype=positions.dtype),
            jnp.asarray(softening, dtype=positions.dtype),
            depth=int(depth),
            order=int(order),
            max_leaf_size=max_leaf_size,
            near_use_pallas=bool(near_use_pallas),
        )
        if return_view:
            return acc, build_uniform_octree_execution_view_device(
                positions, int(depth)
            )
        return acc

    # host-numpy reference path (eager)
    view = build_uniform_octree_execution_view(np.asarray(positions), int(depth))
    octree = octree_execution_data_from_view(view)
    perm = jnp.asarray(view.perm, dtype=INDEX_DTYPE)
    positions_sorted = positions[perm]
    masses_sorted = masses[perm]
    if max_leaf_capacity is not None:
        max_leaf_size = int(max_leaf_capacity)
    else:
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
    )  # host build is compact -> derive num_levels/level_batch_width from its node counts
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
