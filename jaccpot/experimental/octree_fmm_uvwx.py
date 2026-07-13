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

================================================================================
TRANSLATION-OPERATOR SIGN CONVENTION  (read this before touching M2M / M2L / L2L)
================================================================================
Every translation operator here (complex ``m2m_complex``/``l2l_complex`` and real
``m2m_real``/``l2l_real``) takes ``delta = source_center - destination_center`` -- the
vector FROM the destination TO the source. The trap is that "source" and "destination"
mean OPPOSITE things for the up-sweep vs the down-sweep, so the cell-relative sign FLIPS:

  * M2M (up-sweep):   translate a child's multipole INTO its parent.
                      source = CHILD, destination = PARENT
                      -> ``delta = child_center - parent_center``
  * M2L:              source = far node, destination = target node
                      -> ``delta = target_center - source_center``
  * L2L (down-sweep): translate a parent's local INTO its child.
                      source = PARENT, destination = CHILD
                      -> ``delta = parent_center - child_center``   (<-- OPPOSITE of M2M!)

So M2M uses ``child - parent`` but L2L uses ``parent - child``. This is NOT a bug and NOT
a per-operator quirk -- it is the single ``source - destination`` rule applied to the two
sweep directions. Using ``child - parent`` for L2L (the "obvious" symmetry) is WRONG and
silently corrupts the far field for a subset of particles (validated: 4e-16 with
parent-child vs 2.9e-2 with child-parent). This has bitten this code repeatedly; the L2L
cascades below carry an inline reminder. See ``l2l_real`` / ``m2m_real`` docstrings.
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
    _compute_leaf_p2p_prepared_large_n_self_only_impl,
    _radix_fast_lane_prepacked_pallas,
)
from ..operators.complex_ops import (
    complex_rotation_blocks_from_z_solidfmm_batch,
    complex_rotation_blocks_to_z_solidfmm_batch,
)
from ..operators.real_harmonics import (
    evaluate_local_real_with_grad,
    l2l_real,
    m2m_real,
    p2m_real_direct,
    sh_size,
)
from ..runtime._fmm_impl import (
    _accumulate_real_m2l_chunked_scan,
    _accumulate_solidfmm_m2l_chunked_scan,
    _evaluate_local_expansions_for_particles,
    _m2l_complex_batch_cached_kernel,
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
    geometric_centers: bool = False,
    m2l_grouped: bool = False,
    class_capacity: int = 8192,
    basis: str = "complex",
) -> Array:
    """O(N) far field: upward (P2M/M2M) -> V-list M2L -> L2L -> evaluate locals.

    ``basis="real"`` runs the whole far field in the real (Dehnen) harmonic basis
    (:func:`_octree_far_field_grad_real`) instead of the complex solidfmm basis -- half the
    coefficients ((p+1)^2 vs 2(p+1)^2) and an O(p^3) rotate-scale M2L. It always uses box
    centres and ignores ``geometric_centers``/``m2l_grouped``/``class_capacity``. Returns the
    same ``grad`` (caller applies ``far_acc = -G * grad``).

    ``geometric_centers=True`` expands about the octree box centres (``view.centers``)
    instead of centres of mass, making M2L displacements grid-quantised (the prerequisite
    for grouped/cached M2L). Slightly looser truncation error than COM at fixed order.

    ``m2l_grouped=True`` (requires ``geometric_centers=True`` + ``v_active_capacity``) runs the
    CACHED/GROUPED M2L: it groups the compacted V-pairs by their (now bit-identical)
    box-centre displacement via ``jnp.unique`` into ``<= class_capacity`` interaction classes,
    precomputes the solidfmm rotation blocks ONCE per class, and applies them with the cached
    kernel -- so per-pair work is a cheap cached z-translate instead of a full rotation
    recompute. This is the radix fast-lane's M2L trick; ~3.6x faster than the per-pair
    ``rotation="solidfmm"`` chunked scan @200k (190 -> 53 ms). ``class_capacity`` MUST exceed
    the true class count (else ``jnp.unique`` truncates and mis-assigns pairs).

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
    if str(basis).strip().lower() == "real":
        return _octree_far_field_grad_real(
            view,
            positions_sorted,
            masses_sorted,
            order=int(order),
            max_leaf_size=int(max_leaf_size),
            num_levels=num_levels,
            m2l_chunk_size=int(m2l_chunk_size),
            v_active_capacity=v_active_capacity,
        )
    plan = build_octree_upward_plan(octree)
    centers_override = jnp.asarray(view.centers) if bool(geometric_centers) else None
    multipoles = prepare_octree_solidfmm_complex_multipoles(
        plan,
        positions_sorted,
        masses_sorted,
        max_order=int(order),
        max_leaf_size=int(max_leaf_size),
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        centers_override=centers_override,
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
    packed = jnp.asarray(multipoles.packed)
    centers_arr = jnp.asarray(multipoles.centers)
    if m2l_grouped:
        # CACHED/GROUPED M2L (requires geometric centres so displacements quantise).
        if not bool(geometric_centers):
            raise ValueError("m2l_grouped=True requires geometric_centers=True")
        if v_active_capacity is None:
            raise ValueError("m2l_grouped=True requires v_active_capacity")
        cap = int(v_active_capacity)
        dest = jnp.where(
            valid_pairs, jnp.cumsum(valid_pairs.astype(INDEX_DTYPE)) - 1, cap
        )
        src_c = jnp.full((cap,), 0, dtype=INDEX_DTYPE).at[dest].set(v_src, mode="drop")
        tgt_c = jnp.full((cap,), 0, dtype=INDEX_DTYPE).at[dest].set(v_tgt, mode="drop")
        pad = jnp.arange(cap) >= jnp.minimum(n_valid, cap)
        cdt = packed.dtype
        deltas = centers_arr[tgt_c] - centers_arr[src_c]
        # group pairs by exact box-centre displacement -> interaction classes
        uniq, cls_id = jnp.unique(
            deltas,
            axis=0,
            size=int(class_capacity),
            return_inverse=True,
            fill_value=0.0,
        )
        cls_id = cls_id.reshape(-1)
        one_x = jnp.asarray([1.0, 0.0, 0.0], dtype=centers_arr.dtype)
        safe_disp = jnp.where(jnp.all(uniq == 0, axis=1, keepdims=True), one_x, uniq)
        blocks_to = complex_rotation_blocks_to_z_solidfmm_batch(
            safe_disp, order=int(order), basis="multipole", dtype=cdt
        )
        blocks_from = complex_rotation_blocks_from_z_solidfmm_batch(
            safe_disp, order=int(order), basis="local", dtype=cdt
        )
        deltas_safe = jnp.where(pad[:, None], one_x, deltas)  # avoid r=0 on padding
        contribs = _m2l_complex_batch_cached_kernel(
            packed[src_c],
            deltas_safe,
            blocks_to[cls_id],
            blocks_from[cls_id],
            order=int(order),
        ).astype(cdt)
        contribs = jnp.where(pad[:, None], 0, contribs)  # zero padding pairs
        locals_packed = jnp.zeros_like(packed).at[tgt_c].add(contribs)
    else:
        if v_active_capacity is None:
            # No compaction: sort real pairs to the front over the full padded array.
            pair_perm = jnp.argsort(jnp.logical_not(valid_pairs))
            src_pairs = jnp.where(valid_pairs, v_src, -1)[pair_perm]
            tgt_pairs = jnp.where(valid_pairs, v_tgt, -1)[pair_perm]
            active_pairs = n_valid
        else:
            # Stream-compact the real pairs into a fixed v_active_capacity buffer (cumsum
            # scatter, cheaper than argsort); the scan then touches ~real/chunk chunks.
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
            jnp.zeros_like(packed),
            packed,
            centers_arr,
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


def _octree_far_field_grad_real(
    view: UniformOctreeExecutionView,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    order: int,
    max_leaf_size: int,
    num_levels: Optional[int] = None,
    m2l_chunk_size: int = 4096,
    v_active_capacity: Optional[int] = None,
) -> Array:
    """Real-basis (Dehnen) octree far field: real P2M/M2M/M2L/L2L + real L2P (with grad).

    Expands about the octree BOX centres (``view.centers``) throughout -- real harmonics need
    consistent centres. Uses the ``(p+1)^2`` real coefficient layout (half the complex
    ``2(p+1)^2``) and the O(p^3) rotate-scale M2L (``_accumulate_real_m2l_chunked_scan``).
    Returns ``grad`` (caller applies ``far_acc = -G * grad``, matching the complex path).

    SIGN CONVENTION (see the module docstring -- ``delta = source_center - dest_center``):

    * P2M  : ``delta = particle - leaf_center``
    * M2M  : ``delta = child_center - parent_center``   (up-sweep: source=child)
    * M2L  : ``delta = target_center - source_center``
    * L2L  : ``delta = parent_center - child_center``   (down-sweep: source=parent;
             THIS IS THE OPPOSITE SIGN OF M2M -- using child-parent here silently corrupts
             the far field. Confirmed: parent-child 4e-16 vs child-parent 2.9e-2 vs direct.)
    * L2P  : ``delta = leaf_center - eval_point``; acceleration ``= -grad`` (caller's -G).

    Leaf particles are gathered by ``node_ranges`` (NOT a searchsorted-on-starts map, which
    mis-assigns particles of empty leaves and corrupts a subset of the L2P output).
    """
    num_nodes = int(view.parent.shape[0])
    num_leaves = int(view.leaf_indices.shape[0])
    n_part = int(positions_sorted.shape[0])
    p = int(order)
    ncoeff = sh_size(p)
    W = int(max_leaf_size)
    n_levels = int(view.num_levels) if num_levels is None else int(num_levels)

    centers = jnp.asarray(view.centers)
    depths = jnp.asarray(view.node_depths, dtype=INDEX_DTYPE)
    parent = jnp.asarray(view.parent, dtype=INDEX_DTYPE)
    parent_safe = jnp.clip(parent, 0, num_nodes - 1)

    # ---- leaf-major particle packing via node_ranges (no searchsorted) ----
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
    slot = jnp.arange(W, dtype=INDEX_DTYPE)
    leaf_mask = (slot[None, :] < occ[:, None]) & valid_leaf[:, None]  # (L, W)
    leaf_pidx = jnp.where(
        leaf_mask, jnp.clip(lo[:, None] + slot[None, :], 0, n_part - 1), 0
    )
    leaf_pos = positions_sorted[leaf_pidx]  # (L, W, 3)
    leaf_mass = masses_sorted[leaf_pidx]  # (L, W)
    leaf_center = centers[leaf_node_safe]  # (L, 3)

    # ---- P2M: sum_i p2m_real_direct(particle_i - leaf_center) per leaf ----
    d_p2m = leaf_pos - leaf_center[:, None, :]  # (L, W, 3)
    mp_slot = jax.vmap(jax.vmap(lambda d, m: p2m_real_direct(d, m, order=p)))(
        d_p2m, leaf_mass
    )  # (L, W, ncoeff)
    mp_slot = jnp.where(leaf_mask[:, :, None], mp_slot, 0.0)
    leaf_mp = jnp.sum(mp_slot, axis=1)  # (L, ncoeff)
    mp = (
        jnp.zeros((num_nodes, ncoeff), dtype=leaf_mp.dtype)
        .at[leaf_node_safe]
        .add(jnp.where(valid_leaf[:, None], leaf_mp, 0.0))
    )

    # ---- M2M up-sweep: delta = child - parent (source=child) ----
    delta_child_minus_parent = centers - centers[parent_safe]

    def m2m_body(step, mp):
        level = n_levels - 1 - step
        active = (depths == level) & (parent < num_nodes)
        contrib = jax.vmap(lambda m, d: m2m_real(m, d, order=p))(
            mp, delta_child_minus_parent
        )
        return mp.at[parent_safe].add(jnp.where(active[:, None], contrib, 0.0))

    mp = jax.lax.fori_loop(0, n_levels - 1, m2m_body, mp)

    # ---- M2L: delta = target - source; compact V-pairs -> real rot-scale chunked kernel ----
    v_src = jnp.asarray(view.v_src, dtype=INDEX_DTYPE)
    v_tgt = jnp.asarray(view.v_tgt, dtype=INDEX_DTYPE)
    valid_pairs = (v_src < num_nodes) & (v_tgt < num_nodes)
    n_valid = jnp.sum(valid_pairs.astype(INDEX_DTYPE))
    if v_active_capacity is None:
        pair_perm = jnp.argsort(jnp.logical_not(valid_pairs))
        src_pairs = jnp.where(valid_pairs, v_src, -1)[pair_perm]
        tgt_pairs = jnp.where(valid_pairs, v_tgt, -1)[pair_perm]
        active_pairs = n_valid
    else:
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
    locals_packed = _accumulate_real_m2l_chunked_scan(
        jnp.zeros_like(mp),
        mp,
        centers,
        src_pairs,
        tgt_pairs,
        active_pairs,
        order=p,
        m2l_impl="rot_scale",
        total_nodes=num_nodes,
        chunk_size=int(m2l_chunk_size),
    )

    # ---- L2L down-sweep: delta = PARENT - child (source=parent; OPPOSITE of M2M) ----
    delta_parent_minus_child = centers[parent_safe] - centers

    def l2l_body(step, loc):
        level = step + 1
        active = (depths == level) & (parent < num_nodes)
        contrib = jax.vmap(lambda lc, d: l2l_real(lc, d, order=p))(
            loc[parent_safe], delta_parent_minus_child
        )
        return loc + jnp.where(active[:, None], contrib, 0.0)

    locals_packed = jax.lax.fori_loop(0, n_levels - 1, l2l_body, locals_packed)

    # ---- L2P: real local -> field per leaf slot (delta = leaf_center - eval); scatter ----
    leaf_loc = locals_packed[leaf_node_safe]  # (L, ncoeff)
    d_l2p = leaf_center[:, None, :] - leaf_pos  # (L, W, 3) = center - eval
    grad_slot = jax.vmap(
        lambda lc, d_row: jax.vmap(
            lambda d: evaluate_local_real_with_grad(lc, d, order=p)[0]
        )(d_row)
    )(
        leaf_loc, d_l2p
    )  # (L, W, 3)
    grad_slot = jnp.where(leaf_mask[:, :, None], grad_slot, 0.0)
    grad = (
        jnp.zeros((n_part, 3), dtype=positions_sorted.dtype)
        .at[leaf_pidx.reshape(-1)]
        .add(jnp.where(leaf_mask.reshape(-1)[:, None], grad_slot.reshape(-1, 3), 0.0))
    )
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

    def leaf_near(
        leaf_args: tuple[Array, Array, Array],
    ) -> tuple[Array, Array, Array]:
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
    edge_capacity: Optional[int] = None,
    near_chunk_size: int = 512,
) -> Array:
    """U-list near-field P2P over fixed-width UNITS, reusing the radix fast-lane machinery.

    Packs particles into units, builds each unit's source-unit id list, then adds the
    intra-unit self block (``i != j``, via ``_compute_leaf_p2p_prepared_large_n_self_only_impl``)
    plus the cross-unit pairs. With ``use_pallas`` + a supported accelerator this uses the
    Pallas ``nearfield_leafpair`` kernel (``_radix_fast_lane_prepacked_pallas``: gathers
    sources by id + cheaply cond-skips padded slots; fastest, ~200 ms @200k). Otherwise it
    uses a memory-safe pure-JAX CHUNKED EDGE SCAN over the (target, source) unit pairs -- the
    AUTODIFF-ABLE / non-Ampere path (~780 ms @200k; the radix prepacked pure-JAX impl OOMs at
    36-132 GiB). ``edge_capacity`` (static; default ``num_units * source_slots``) bounds the
    compacted edge list and ``near_chunk_size`` is the per-chunk pair block. Morton order.

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
        # Memory-safe pure-JAX cross-pairs (autodiff-able / non-Ampere path): enumerate the
        # (target-unit, source-unit) edges from the source lists, stream-compact the real
        # edges to the front, and lax.scan over fixed-size chunks -- per-chunk memory is only
        # (near_chunk_size, W, W, 3), bounded, and whole padding chunks are cond-skipped. The
        # radix prepacked impl OOMs here (materializes dense tiles: 36-132 GiB @200k).
        n_src = int(src_ids_3d.shape[2])
        cap = int(edge_capacity) if edge_capacity is not None else num_units * n_src
        chunk = int(near_chunk_size)
        tgt_flat = jnp.repeat(jnp.arange(num_units, dtype=INDEX_DTYPE), n_src)
        src_flat = src_ids_3d.reshape(-1)
        edge_valid = src_valid_3d.reshape(-1)
        dest = jnp.where(
            edge_valid, jnp.cumsum(edge_valid.astype(INDEX_DTYPE)) - 1, cap
        )
        tgt_e = (
            jnp.full((cap,), 0, dtype=INDEX_DTYPE).at[dest].set(tgt_flat, mode="drop")
        )
        src_e = (
            jnp.full((cap,), 0, dtype=INDEX_DTYPE).at[dest].set(src_flat, mode="drop")
        )
        n_edges = jnp.minimum(jnp.sum(edge_valid.astype(INDEX_DTYPE)), cap)
        starts = jnp.arange(0, cap, chunk, dtype=INDEX_DTYPE)

        def _edge_body(near, start):
            def _active(near):
                rng = start + jnp.arange(chunk, dtype=INDEX_DTYPE)
                idx = jnp.clip(rng, 0, cap - 1)
                inr = rng < n_edges
                tb = tgt_e[idx]
                sbk = src_e[idx]
                tpos = unit_positions[tb]
                tmask = unit_mask[tb] & inr[:, None]
                tpidx = unit_pidx[tb]
                spos = unit_positions[sbk]
                smass = unit_masses[sbk]
                smask = unit_mask[sbk]
                d = tpos[:, :, None, :] - spos[:, None, :, :]
                d2 = jnp.sum(d * d, axis=-1) + softening_sq
                pv = tmask[:, :, None] & smask[:, None, :]
                inv = jnp.where(pv, d2 ** (-1.5), 0.0)
                ac = -G * jnp.sum(smass[:, None, :, None] * d * inv[..., None], axis=2)
                return near.at[tpidx.reshape(-1)].add(
                    jnp.where(tmask.reshape(-1)[:, None], ac.reshape(-1, 3), 0.0)
                )

            return jax.lax.cond(start < n_edges, _active, lambda nr: nr, near), None

        pair_acc, _ = jax.lax.scan(
            _edge_body, jnp.zeros((n_part, 3), dtype=positions_sorted.dtype), starts
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
        "geometric_centers",
        "m2l_grouped",
        "class_capacity",
        "v_active_capacity",
        "near_block_size",
        "edge_capacity",
        "near_chunk_size",
        "basis",
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
    geometric_centers: bool = False,
    m2l_grouped: bool = False,
    class_capacity: int = 8192,
    v_active_capacity: Optional[int] = None,
    near_block_size: Optional[int] = 128,
    edge_capacity: Optional[int] = None,
    near_chunk_size: int = 512,
    basis: str = "complex",
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
        v_active_capacity=v_active_capacity,
        geometric_centers=bool(geometric_centers),
        m2l_grouped=bool(m2l_grouped),
        class_capacity=int(class_capacity),
        basis=str(basis),
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
        near_block_size=near_block_size,
        edge_capacity=edge_capacity,
        near_chunk_size=int(near_chunk_size),
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
    geometric_centers: bool = False,
    m2l_grouped: bool = False,
    class_capacity: int = 8192,
    v_active_capacity: Optional[int] = None,
    near_block_size: Optional[int] = 128,
    edge_capacity: Optional[int] = None,
    near_chunk_size: int = 512,
    basis: str = "complex",
) -> Array | tuple[Array, UniformOctreeExecutionView]:
    """O(N) octree-FMM gravitational accelerations (far V-list + near U-list P2P).

    ``positions`` (N, 3), ``masses`` (N,); returns accelerations in the SAME order as the
    inputs. ``depth`` = uniform octree levels, ``order`` = multipole expansion order.

    Fast/opt-in knobs (all static, default to the safe COM + chunked-M2L + Pallas-near path):

    * ``near_use_pallas`` (default True) -- Pallas leaf-pair near (fastest); ``False`` uses the
      memory-safe chunked pure-JAX near (autodiff-able; runs on non-Ampere/CPU).
    * ``geometric_centers`` + ``m2l_grouped`` (+ ``v_active_capacity``, ``class_capacity``) --
      the grouped/cached M2L on box centres (~3.6x faster M2L). ``m2l_grouped`` requires
      ``geometric_centers=True`` and a ``v_active_capacity`` >= the true V-pair count.
    * ``near_block_size`` / ``edge_capacity`` / ``near_chunk_size`` -- near-field packing knobs.

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
                geometric_centers=bool(geometric_centers),
                m2l_grouped=bool(m2l_grouped),
                class_capacity=int(class_capacity),
                v_active_capacity=v_active_capacity,
                near_block_size=near_block_size,
                edge_capacity=edge_capacity,
                near_chunk_size=int(near_chunk_size),
                basis=str(basis),
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
            geometric_centers=bool(geometric_centers),
            m2l_grouped=bool(m2l_grouped),
            class_capacity=int(class_capacity),
            v_active_capacity=v_active_capacity,
            near_block_size=near_block_size,
            edge_capacity=edge_capacity,
            near_chunk_size=int(near_chunk_size),
            basis=str(basis),
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
