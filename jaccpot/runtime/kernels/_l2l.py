"""L2L propagation down the tree, and the downward-sweep driver.

Two layers: the per-level L2L translation (``_propagate_*_locals_*``), and
``_prepare_solidfmm_downward_sweep``, the driver that walks levels, calls the M2L
accumulate for each, and propagates the result to the children. ARCHITECTURE §5's
third family.

``_propagate_solidfmm_locals_by_level`` carries ``donate_argnums=(0,)``: the
locals buffer is donated so the level loop does not hold two copies of the whole
tree's coefficients. **Do not change the donation or the argument order** --
NUMERICS_AND_JAX §2 treats both as design decisions, and a silent permutation of
donated buffers is not something the goldens would localise.

Split out of ``core.py`` (Tier 1.6, A.9 seam 3); every function body is unchanged.
"""

from __future__ import annotations

import os
import time
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from beartype.typing import Callable
from jaxtyping import Array
from yggdrax.dense_interactions import DenseInteractionBuffers
from yggdrax.grouped_interactions import (
    GroupedInteractionBuffers,
)
from yggdrax.interactions import (
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    MACType,
    NodeInteractionList,
)
from yggdrax.tree import Tree, get_node_levels

from jaccpot._jax_compat import Tracer
from jaccpot.downward.local_expansions import (
    LocalExpansionData,
    TreeDownwardData,
)
from jaccpot.operators.complex_ops import (
    enforce_conjugate_symmetry_batch,
    l2l_complex_batch,
)
from jaccpot.operators.m2l_real_rot_scale import (
    l2l_rot_scale_real_batch_cached_blocks,
    real_rotation_blocks_from_z_local_batch,
    real_rotation_blocks_to_z_local_batch,
)
from jaccpot.operators.real_harmonics import (
    l2l_real,
)
from jaccpot.upward.tree_expansions import TreeUpwardData

from ..dtypes import INDEX_DTYPE
from ._downward_prep import (
    _FarPairCOO,
    _prepare_solidfmm_downward_child_inputs,
    _prepare_solidfmm_downward_init,
    _prepare_solidfmm_downward_interaction_inputs,
    _prepare_solidfmm_downward_multipole_inputs,
    _solidfmm_downward_accumulate_from_multipoles,
)
from ._shared import _normalize_strict_refresh_detail_diag_mode

__all__: list[str] = []


@partial(jax.jit, static_argnames=("order", "rotation"))
def _l2l_complex_batch_kernel(
    coeffs: Array,
    deltas: Array,
    *,
    order: int,
    rotation: str,
) -> Array:
    """Vectorized complex-basis L2L translation kernel.

    Parameters
    ----------
    coeffs : Array
        ``(pairs, C)`` parent local coefficients, one row per child.
    deltas : Array
        ``(pairs, 3)`` displacements, ``parent centre - child centre``.
    order : int
        Expansion order.
    rotation : str
        Rotation lane for the complex L2L.

    Returns
    -------
    Array
        ``(pairs, C)`` local coefficients re-centred on the children.
    """
    return l2l_complex_batch(coeffs, deltas, order=order, rotation=rotation)


@partial(jax.jit, static_argnames=("order",))
def _l2l_real_batch_kernel(
    coeffs: Array,
    deltas: Array,
    *,
    order: int,
) -> Array:
    """Vectorized real-basis L2L translation kernel.

    Parameters
    ----------
    coeffs : Array
        ``(pairs, C)`` parent local coefficients, one row per child.
    deltas : Array
        ``(pairs, 3)`` displacements, ``parent centre - child centre`` -- the
        same sign convention as the complex kernel.
    order : int
        Expansion order.

    Returns
    -------
    Array
        ``(pairs, C)`` local coefficients re-centred on the children.
    """
    return jax.vmap(lambda c, d: l2l_real(c, d, order=order))(coeffs, deltas)


@partial(
    jax.jit,
    static_argnames=("order", "rotation", "total_nodes"),
    donate_argnums=(0,),
)
def _propagate_solidfmm_locals_to_children(
    coeffs_local: Array,
    centers_local: Array,
    left_child: Array,
    right_child: Array,
    *,
    order: int,
    rotation: str,
    total_nodes: int,
) -> Array:
    """Apply solidfmm L2L translations from parents to their children.

    One level of descent only. Use
    :func:`_propagate_solidfmm_locals_by_level` when expansions may have been
    deposited above the leaves, which is the general case.

    Parameters
    ----------
    coeffs_local : Array
        ``(total_nodes, C)`` local coefficients; parents are read, children
        written.
    centers_local : Array
        ``(total_nodes, 3)`` expansion centres.
    left_child : Array
        Left child of each internal node.
    right_child : Array
        Right child of each internal node.
    order : int
        Expansion order.
    rotation : str
        Rotation lane for the complex L2L.
    total_nodes : int
        Node count, i.e. the scatter extent.

    Returns
    -------
    Array
        ``(total_nodes, C)`` locals with each child's inherited expansion added.
    """
    num_internal_nodes = left_child.shape[0]
    parent_idx = jnp.arange(num_internal_nodes, dtype=INDEX_DTYPE)
    child_idx = jnp.concatenate(
        [left_child[:num_internal_nodes], right_child[:num_internal_nodes]],
        axis=0,
    )
    parent_rep = jnp.concatenate([parent_idx, parent_idx], axis=0)
    valid = child_idx >= 0
    safe_child_idx = jnp.where(valid, child_idx, 0)

    parent_coeffs = coeffs_local[parent_rep]
    deltas = centers_local[safe_child_idx] - centers_local[parent_rep]
    translated = _l2l_complex_batch_kernel(
        parent_coeffs,
        deltas,
        order=order,
        rotation=rotation,
    )
    translated = translated.astype(coeffs_local.dtype)
    translated = jnp.where(valid[:, None], translated, 0)
    updates = jax.ops.segment_sum(translated, safe_child_idx, total_nodes)
    return coeffs_local + updates


@partial(
    jax.jit,
    static_argnames=(
        "order",
        "rotation",
        "total_nodes",
        "basis_mode",
        "l2l_grouped",
        "mm_class_capacity",
        "num_levels",
    ),
    donate_argnums=(0,),
)
def _propagate_solidfmm_locals_by_level(
    coeffs_local: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    node_levels: Array,
    *,
    order: int,
    rotation: str,
    total_nodes: int,
    basis_mode: str = "complex",
    l2l_grouped: bool = False,
    mm_class_capacity: int = 512,
    num_levels: Optional[int] = None,
) -> Array:
    """Top-down, level-by-level L2L cascade over a binary tree.

    A single parent->child pass (``_propagate_solidfmm_locals_to_children``)
    moves each node's local expansion down exactly one level. That is only
    sufficient when every node already carries the far-field appropriate to its
    own level. A local expansion deposited high in the tree (a well-separated
    interaction accepted at a coarse node) must instead cascade through every
    intermediate level to reach the leaves, or the leaves never see it and the
    evaluated field degrades with tree depth.

    Iterate levels root->leaf and translate only the parents that live at the
    current level, so each node's fully-accumulated expansion (its own plus
    everything inherited from shallower ancestors) is propagated to its children
    before those children are used as parents in turn.

    Parameters
    ----------
    coeffs_local : Array
        ``(total_nodes, C)`` local coefficients, updated in the cascade.
    centers : Array
        ``(total_nodes, 3)`` expansion centres.
    left_child : Array
        Left child of each internal node.
    right_child : Array
        Right child of each internal node.
    node_levels : Array
        Depth of each node; drives which parents are active per iteration.
    order : int
        Expansion order.
    rotation : str
        Rotation lane for the complex L2L.
    total_nodes : int
        Node count.
    basis_mode : str
        ``"complex"`` or ``"real"``; selects the batch kernel.
    l2l_grouped : bool
        Use the grouped L2L lane.
    mm_class_capacity : int
        Per-class capacity for the grouped lane.
    num_levels : Optional[int]
        Concrete tree depth. ``None`` falls back to the padded shape-derived
        depth, which is correct but iterates more levels than necessary.

    Returns
    -------
    Array
        ``(total_nodes, C)`` locals after the full root-to-leaf cascade.
    """
    num_internal = int(left_child.shape[0])
    if num_internal <= 0:
        return coeffs_local

    real_basis = str(basis_mode).strip().lower() == "real"
    left_internal = left_child[:num_internal]
    right_internal = right_child[:num_internal]
    parent_levels = node_levels[:num_internal].astype(INDEX_DTYPE)
    parent_idx = jnp.arange(num_internal, dtype=INDEX_DTYPE)
    parent_rep = jnp.concatenate([parent_idx, parent_idx], axis=0)
    # ``num_levels`` (the max node level) is pure tree topology. When a caller
    # that holds the tree concretely passes it as a static int, use it as the
    # loop bound so the level cascade runs with STATIC ``fori_loop`` bounds --
    # required for reverse-mode autodiff, which rejects ``fori_loop`` with
    # dynamic start/stop. Falls back to the dynamic device reduction when it is
    # unknown; numerics are identical either way (the loop performs exactly
    # ``max_level + 1`` iterations regardless of how the bound is obtained).
    max_level = int(num_levels) if num_levels is not None else jnp.max(parent_levels)
    minus_one = jnp.asarray(-1, dtype=left_internal.dtype)

    # GROUPED/CACHED real L2L: the parent->child displacement set is FIXED by the tree, so
    # precompute the real rotation blocks ONCE per displacement class (jnp.unique over all
    # internal->child pairs) and cached-apply per level, instead of rebuilding the rotation
    # matrices for every node at every level. Only helps when the class count is small (box/
    # geometric centres quantise the displacements); with COM centres it still works but each
    # pair is its own class. Bit-identical to the per-node kernel either way.
    use_grouped_l2l = bool(l2l_grouped) and real_basis
    if use_grouped_l2l:
        children_full = jnp.concatenate([left_internal, right_internal], axis=0)
        safe_full = jnp.where(children_full >= 0, children_full, 0)
        full_deltas = centers[parent_rep] - centers[safe_full]
        one_x = jnp.asarray([1.0, 0.0, 0.0], dtype=centers.dtype)
        l2l_uniq, l2l_cls = jnp.unique(
            full_deltas,
            axis=0,
            size=int(mm_class_capacity),
            return_inverse=True,
            fill_value=0.0,
        )
        l2l_cls = l2l_cls.reshape(-1)
        l2l_disp = jnp.where(
            jnp.all(l2l_uniq == 0, axis=1, keepdims=True), one_x, l2l_uniq
        )
        rdt = coeffs_local.dtype
        l2l_bt = real_rotation_blocks_to_z_local_batch(
            l2l_disp, order=order, dtype=rdt
        )[l2l_cls]
        l2l_bf = real_rotation_blocks_from_z_local_batch(
            l2l_disp, order=order, dtype=rdt
        )[l2l_cls]

    def _l2l_level_apply(
        state_in: Array,
        centers_all: Array,
        safe_child: Array,
        valid: Array,
    ) -> Array:
        """Gather one level's parents, L2L-translate them onto their children.

        Wrapped in ``jax.checkpoint`` below so reverse mode retains only these
        inputs. Un-rematerialized the level cascade keeps every level's rotation
        blocks and their bilinear construction intermediates -- measured at
        **14.2 kB per (level x node)** by ``bench/audit_reverse_residuals.py``,
        i.e. ``depth x nodes`` and 5.4 GB at N=1048576. ``centers_all`` is
        loop-invariant so it is hoisted and counted once.

        L2L uses the old_center - new_center (parent - child) displacement in BOTH
        bases. The complex path previously used child - parent here, which is the
        wrong sign: the far field was left uncorrected in proportion to the
        cascade depth, capping accuracy (~3e-3 at theta>=0.5) regardless of
        expansion order, while looking fine at small theta where the cascade is
        shallow.

        Parameters
        ----------
        state_in : Array
            ``(total_nodes, C)`` locals accumulated so far. Parents are read from
            here, so it must already carry everything inherited from shallower
            levels.
        centers_all : Array
            ``(total_nodes, 3)`` expansion centres. Loop-invariant, so it is
            hoisted out of the level scan and its residual counted once.
        safe_child : Array
            Child index per active parent slot, with invalid slots pointed at
            ``0`` so the gather stays in bounds.
        valid : Array
            Mask over ``safe_child``; invalid slots are zeroed after the
            translate rather than skipped.

        Returns
        -------
        Array
            ``(total_nodes, C)`` translated contributions, zero outside the
            active children, ready to be added to ``state_in``.
        """
        parent_coeffs = state_in[parent_rep]
        deltas = centers_all[parent_rep] - centers_all[safe_child]
        if use_grouped_l2l:
            translated = l2l_rot_scale_real_batch_cached_blocks(
                # Both are bound under the same `use_grouped_l2l` test that
                # gates this call -- E.4 bucket D.
                parent_coeffs,
                deltas,
                l2l_bt,  # pyright: ignore[reportPossiblyUnboundVariable]
                l2l_bf,  # pyright: ignore[reportPossiblyUnboundVariable]
                order=order,
            ).astype(state_in.dtype)
        elif real_basis:
            translated = _l2l_real_batch_kernel(
                parent_coeffs, deltas, order=order
            ).astype(state_in.dtype)
        else:
            translated = _l2l_complex_batch_kernel(
                parent_coeffs, deltas, order=order, rotation=rotation
            ).astype(state_in.dtype)
        return jnp.where(valid[:, None], translated, 0)

    _l2l_level = jax.checkpoint(_l2l_level_apply)

    def level_body(level: Array, state: Array) -> Array:
        active = parent_levels == level
        lc = jnp.where(active, left_internal, minus_one)
        rc = jnp.where(active, right_internal, minus_one)
        child_idx = jnp.concatenate([lc, rc], axis=0)
        valid = child_idx >= 0
        safe_child = jnp.where(valid, child_idx, 0)
        # Gather + translate + mask, rematerialized (see ``_l2l_level_apply``).
        translated = _l2l_level(state, centers, safe_child, valid)
        updates = jax.ops.segment_sum(translated, safe_child, total_nodes)
        return state + updates

    return jax.lax.fori_loop(0, max_level + 1, level_body, coeffs_local)


@partial(
    jax.jit,
    static_argnames=("order", "total_nodes"),
    donate_argnums=(0,),
)
def _propagate_real_locals_to_children(
    coeffs_local: Array,
    centers_local: Array,
    left_child: Array,
    right_child: Array,
    *,
    order: int,
    total_nodes: int,
) -> Array:
    """Apply real-basis L2L translations from parents to their children.

    Real-basis twin of :func:`_propagate_solidfmm_locals_to_children`; one level
    of descent only.

    Parameters
    ----------
    coeffs_local : Array
        ``(total_nodes, C)`` local coefficients; parents are read, children
        written.
    centers_local : Array
        ``(total_nodes, 3)`` expansion centres.
    left_child : Array
        Left child of each internal node.
    right_child : Array
        Right child of each internal node.
    order : int
        Expansion order.
    total_nodes : int
        Node count, i.e. the scatter extent.

    Returns
    -------
    Array
        ``(total_nodes, C)`` locals with each child's inherited expansion added.
    """
    num_internal_nodes = left_child.shape[0]
    parent_idx = jnp.arange(num_internal_nodes, dtype=INDEX_DTYPE)
    child_idx = jnp.concatenate(
        [left_child[:num_internal_nodes], right_child[:num_internal_nodes]],
        axis=0,
    )
    parent_rep = jnp.concatenate([parent_idx, parent_idx], axis=0)
    valid = child_idx >= 0
    safe_child_idx = jnp.where(valid, child_idx, 0)
    parent_coeffs = coeffs_local[parent_rep]
    deltas = centers_local[safe_child_idx] - centers_local[parent_rep]
    translated = _l2l_real_batch_kernel(parent_coeffs, deltas, order=order)
    translated = translated.astype(coeffs_local.dtype)
    translated = jnp.where(valid[:, None], translated, 0)
    updates = jax.ops.segment_sum(translated, safe_child_idx, total_nodes)
    return coeffs_local + updates


def _prepare_solidfmm_downward_sweep(
    tree: Tree,
    upward: TreeUpwardData,
    *,
    theta: float,
    mac_type: MACType,
    initial_locals: Optional[LocalExpansionData] = None,
    interactions: Optional[NodeInteractionList] = None,
    m2l_chunk_size: Optional[int] = None,
    l2l_chunk_size: Optional[int] = None,
    complex_rotation: str = "solidfmm",
    basis_mode: str = "complex",
    m2l_impl: Optional[str] = None,
    traversal_config: Optional[DualTreeTraversalConfig] = None,
    dense_buffers: Optional[DenseInteractionBuffers] = None,
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
    grouped_interactions: bool = False,
    grouped_buffers: Optional[GroupedInteractionBuffers] = None,
    grouped_segment_starts: Optional[Array] = None,
    grouped_segment_lengths: Optional[Array] = None,
    grouped_segment_class_ids: Optional[Array] = None,
    grouped_segment_sort_permutation: Optional[Array] = None,
    grouped_segment_group_ids: Optional[Array] = None,
    grouped_segment_unique_targets: Optional[Array] = None,
    farfield_mode: str = "pair_grouped",
    far_pairs_coo: Optional[_FarPairCOO] = None,
    far_pairs_by_gear: Optional[tuple[tuple[Array, Array], ...]] = None,
    adaptive_order: bool = False,
    p_gears: tuple[int, ...] = tuple(),
    dehnen_radius_scale: float = 1.0,
    use_pallas: bool = False,
    timing_recorder: Optional[Callable[[str, float], None]] = None,
) -> TreeDownwardData:
    """Prepare M2L accumulation for solidfmm-style complex or real locals.

    The returned value intentionally retains only the locals plus a minimal
    interaction handle. Grouped layouts, chunk schedules, and other M2L feed
    structures are execution inputs, not part of the long-lived downward state.

    Parameters
    ----------
    tree : Tree
        Tree being swept.
    upward : TreeUpwardData
        Upward-pass result supplying multipoles and geometry.
    theta : float
        MAC opening angle.
    mac_type : MACType
        Traversal-facing acceptance criterion. Already resolved -- the
        caller-facing vocabulary is mapped in
        :meth:`~jaccpot.runtime.fmm_sweeps.SweepsMixin.prepare_downward_sweep`.
    initial_locals : Optional[LocalExpansionData]
        Locals to accumulate into; ``None`` starts from zeros.
    interactions : Optional[NodeInteractionList]
        Pre-built far-pair list; ``None`` traverses.
    m2l_chunk_size : Optional[int]
        Pairs per M2L chunk.
    l2l_chunk_size : Optional[int]
        Nodes per L2L chunk.
    complex_rotation : str
        Rotation lane on the complex basis.
    basis_mode : str
        ``"complex"`` or ``"real"``.
    m2l_impl : Optional[str]
        M2L implementation selector for the flat lanes.
    traversal_config : Optional[DualTreeTraversalConfig]
        Traversal capacities and retry policy.
    dense_buffers : Optional[DenseInteractionBuffers]
        Pre-built dense interaction buffers.
    retry_logger : Optional[Callable[[DualTreeRetryEvent], None]]
        Called when the traversal retries with a larger capacity.
    grouped_interactions : bool
        Use the grouped M2L lanes.
    grouped_buffers : Optional[GroupedInteractionBuffers]
        Pre-built grouped buffers; built on demand when ``None``.
    grouped_segment_starts : Optional[Array]
        Class-major segment starts.
    grouped_segment_lengths : Optional[Array]
        Class-major segment lengths.
    grouped_segment_class_ids : Optional[Array]
        Translation class of each segment.
    grouped_segment_sort_permutation : Optional[Array]
        Permutation into class-major order.
    grouped_segment_group_ids : Optional[Array]
        Group id of each segment.
    grouped_segment_unique_targets : Optional[Array]
        Distinct target nodes per segment.
    farfield_mode : str
        ``"pair_grouped"`` or ``"class_major"`` on the grouped path.
    far_pairs_coo : Optional[_FarPairCOO]
        Pre-built COO far pairs, which take precedence over ``interactions``.
    far_pairs_by_gear : Optional[tuple[tuple[Array, Array], ...]]
        Per-gear far-pair lists for the adaptive-order path.
    adaptive_order : bool
        Choose the expansion order per interaction from ``p_gears``.
    p_gears : tuple[int, ...]
        Candidate orders for ``adaptive_order``.
    dehnen_radius_scale : float
        Node-radius scale for the Dehnen criteria.
    use_pallas : bool
        Allow the Pallas M2L lanes.
    timing_recorder : Optional[Callable[[str, float], None]]
        Per-stage timing sink. Passing one forces host synchronisation via
        ``block_until_ready``, so it changes timing behaviour and is for
        profiling only.

    Returns
    -------
    TreeDownwardData
        Local expansions plus a minimal interaction handle.

    Raises
    ------
    ValueError
        If a basis, rotation or far-field mode option is outside its documented
        domain.
    """

    interaction_inputs = _prepare_solidfmm_downward_interaction_inputs(
        tree=tree,
        upward=upward,
        theta=theta,
        mac_type=mac_type,
        interactions=interactions,
        far_pairs_coo=far_pairs_coo,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
        dehnen_radius_scale=dehnen_radius_scale,
    )
    interactions = interaction_inputs.interactions
    src = interaction_inputs.src
    tgt = interaction_inputs.tgt
    pair_count = interaction_inputs.pair_count
    active_pair_count = interaction_inputs.active_pair_count

    def _record_timed_array(attr: str, start: float, value: Array) -> Array:
        if timing_recorder is None:
            return value
        value = jax.block_until_ready(value)
        timing_recorder(attr, float(time.perf_counter() - start))
        return value

    p = int(upward.multipoles.order)
    downward_init = _prepare_solidfmm_downward_init(
        upward=upward,
        initial_locals=initial_locals,
        basis_mode=basis_mode,
    )
    centers = downward_init.centers
    locals_coeffs = downward_init.locals_coeffs
    total_nodes = downward_init.total_nodes
    coeff_count = downward_init.coeff_count
    dtype = downward_init.dtype

    if pair_count == 0:
        empty_locals = LocalExpansionData(
            order=p,
            centers=centers,
            coefficients=locals_coeffs,
        )
        empty_source_motion_locals: Optional[LocalExpansionData]
        if upward.multipoles.source_motion_packed is not None:
            empty_source_motion_locals = LocalExpansionData(
                order=p,
                centers=centers,
                coefficients=jnp.zeros_like(locals_coeffs),
            )
        else:
            empty_source_motion_locals = None
        return TreeDownwardData(
            interactions=interactions,
            locals=empty_locals,
            source_motion_locals=empty_source_motion_locals,
        )

    detail_diag_mode = _normalize_strict_refresh_detail_diag_mode(
        os.environ.get("JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE", "full")
    )

    def _detail_downward_data(
        coefficients: Array,
        source_motion_coefficients: Optional[Array] = None,
    ) -> TreeDownwardData:
        source_motion_locals: Optional[LocalExpansionData]
        if source_motion_coefficients is not None:
            source_motion_locals = LocalExpansionData(
                order=p,
                centers=centers,
                coefficients=source_motion_coefficients,
            )
        else:
            source_motion_locals = None
        return TreeDownwardData(
            interactions=interactions,
            locals=LocalExpansionData(
                order=p,
                centers=centers,
                coefficients=coefficients,
            ),
            source_motion_locals=source_motion_locals,
        )

    if detail_diag_mode == "downward_artifacts_only":
        source_motion_zeros = (
            jnp.zeros_like(locals_coeffs)
            if upward.multipoles.source_motion_packed is not None
            else None
        )
        return _detail_downward_data(locals_coeffs, source_motion_zeros)

    rotation_mode = str(complex_rotation).strip().lower()
    resolved_m2l_impl = (
        "rot_scale" if m2l_impl is None else str(m2l_impl).strip().lower()
    )
    source_motion_multip_packed = None
    if detail_diag_mode == "l2l_only":
        locals_updated = jnp.ones_like(locals_coeffs)
        chunk_size = 4096 if m2l_chunk_size is None else int(m2l_chunk_size)
        if chunk_size <= 0:
            raise ValueError("m2l_chunk_size must be positive")
    else:
        multipole_inputs = _prepare_solidfmm_downward_multipole_inputs(
            upward=upward,
            dtype=dtype,
            basis_mode=basis_mode,
            complex_rotation=complex_rotation,
        )
        multip_packed = multipole_inputs.multip_packed
        source_motion_multip_packed = multipole_inputs.source_motion_multip_packed
        multip_packed_kernel = multipole_inputs.multip_packed_kernel
        rotation_mode = multipole_inputs.rotation_mode

        chunk_size = 4096 if m2l_chunk_size is None else int(m2l_chunk_size)
        if chunk_size <= 0:
            raise ValueError("m2l_chunk_size must be positive")

        stage_t0 = time.perf_counter()
        locals_updated = _solidfmm_downward_accumulate_from_multipoles(
            locals_coeffs,
            multip_packed_kernel,
            tree=tree,
            upward=upward,
            interactions=interactions,
            centers=centers,
            src=src,
            tgt=tgt,
            pair_count=pair_count,
            active_pair_count=active_pair_count,
            order=p,
            rotation_mode=rotation_mode,
            total_nodes=total_nodes,
            chunk_size=chunk_size,
            grouped_interactions=grouped_interactions,
            grouped_buffers=grouped_buffers,
            grouped_segment_starts=grouped_segment_starts,
            grouped_segment_lengths=grouped_segment_lengths,
            grouped_segment_class_ids=grouped_segment_class_ids,
            grouped_segment_sort_permutation=grouped_segment_sort_permutation,
            grouped_segment_group_ids=grouped_segment_group_ids,
            grouped_segment_unique_targets=grouped_segment_unique_targets,
            farfield_mode=farfield_mode,
            basis_mode=basis_mode,
            m2l_impl=resolved_m2l_impl,
        )
        locals_updated = _record_timed_array(
            "_refresh_timing_dual_m2l_compute_seconds",
            stage_t0,
            locals_updated,
        )
        if detail_diag_mode == "m2l_only":
            return _detail_downward_data(locals_updated)

    if l2l_chunk_size is not None and int(l2l_chunk_size) <= 0:
        raise ValueError("l2l_chunk_size must be positive")

    child_inputs = _prepare_solidfmm_downward_child_inputs(tree)
    if child_inputs.num_internal_nodes > 0:
        left_child = child_inputs.left_child
        right_child = child_inputs.right_child
        node_levels = get_node_levels(tree)
        # Resolve the max node level to a STATIC int when the tree is held
        # concretely (e.g. the fixed-topology differentiable re-eval, where the
        # tree is a captured constant), so the L2L level cascade uses static
        # ``fori_loop`` bounds and is reverse-mode differentiable. When
        # ``node_levels`` is a tracer (jitted tree/traversal), fall back to the
        # kernel's internal dynamic reduction -- numerics are identical.
        if isinstance(node_levels, Tracer):
            l2l_num_levels: Optional[int] = None
        else:
            # ``node_levels`` is concrete here, but under an outer ``jax.jit`` any
            # jnp op on it (even a slice) is staged into the jaxpr and cannot be
            # read by ``int()``. Pull the whole array to the host FIRST, then
            # slice + reduce in numpy, so the level count is a plain Python int in
            # every trace context (mirrors ``_resolve_upward_num_levels``).
            node_levels_host = jax.device_get(node_levels)
            l2l_num_levels = int(
                node_levels_host[: child_inputs.num_internal_nodes].max()
            )
        stage_t0 = time.perf_counter()
        locals_updated = _propagate_solidfmm_locals_by_level(
            locals_updated,
            centers,
            left_child,
            right_child,
            node_levels,
            order=p,
            rotation=rotation_mode,
            total_nodes=total_nodes,
            basis_mode=basis_mode,
            num_levels=l2l_num_levels,
        )
        locals_updated = _record_timed_array(
            "_refresh_timing_dual_l2l_compute_seconds",
            stage_t0,
            locals_updated,
        )
        source_motion_locals_updated: Optional[Array]
        if source_motion_multip_packed is not None:
            stage_t0 = time.perf_counter()
            source_motion_locals_updated = (
                _solidfmm_downward_accumulate_from_multipoles(
                    jnp.zeros_like(locals_coeffs),
                    source_motion_multip_packed,
                    tree=tree,
                    upward=upward,
                    interactions=interactions,
                    centers=centers,
                    src=src,
                    tgt=tgt,
                    pair_count=pair_count,
                    active_pair_count=active_pair_count,
                    order=p,
                    rotation_mode=rotation_mode,
                    total_nodes=total_nodes,
                    chunk_size=chunk_size,
                    grouped_interactions=grouped_interactions,
                    grouped_buffers=grouped_buffers,
                    grouped_segment_starts=grouped_segment_starts,
                    grouped_segment_lengths=grouped_segment_lengths,
                    grouped_segment_class_ids=grouped_segment_class_ids,
                    grouped_segment_sort_permutation=grouped_segment_sort_permutation,
                    grouped_segment_group_ids=grouped_segment_group_ids,
                    grouped_segment_unique_targets=grouped_segment_unique_targets,
                    farfield_mode=farfield_mode,
                    basis_mode=basis_mode,
                    m2l_impl=resolved_m2l_impl,
                )
            )
            source_motion_locals_updated = _propagate_solidfmm_locals_by_level(
                source_motion_locals_updated,
                centers,
                left_child,
                right_child,
                node_levels,
                order=p,
                rotation=rotation_mode,
                total_nodes=total_nodes,
                basis_mode=basis_mode,
                num_levels=l2l_num_levels,
            )
            source_motion_locals_updated = _record_timed_array(
                "_refresh_timing_dual_source_motion_seconds",
                stage_t0,
                source_motion_locals_updated,
            )
        else:
            source_motion_locals_updated = None
    else:
        # No internal nodes: the tree is a single leaf. There is no M2L to do (a
        # lone node has no box pair to be far from) and no parent-to-child cascade
        # to run, so the source-motion far field is exactly ZERO -- the whole
        # interaction is near-field, which this function does not compute. That is
        # the same conclusion the `pair_count == 0` guard above already reaches, and
        # zero is what it returns there; these two agree on purpose.
        #
        # This used to call `_accumulate_from_multipoles`, which is defined nowhere
        # -- audit G.1a, a latent `NameError`. It never fired because the branch is
        # unreachable: getting here needs `pair_count > 0` AND
        # `num_internal_nodes == 0`, and the `pair_count == 0` return above already
        # took every single-leaf tree. Measured, not assumed -- forcing a one-node
        # tree with a foreign interaction list still does not reach this line. The
        # audit's reachability note missed that guard.
        #
        # Kept as zeros rather than a raise: it is the correct value if the guard
        # above is ever loosened, and it cannot introduce a new crash path.
        # `None` would be wrong -- it means "source motion was not requested" (the
        # else below), which the consumer cannot distinguish from "requested, and
        # the far field is empty".
        if source_motion_multip_packed is not None:
            source_motion_locals_updated = jnp.zeros_like(locals_coeffs)
        else:
            source_motion_locals_updated = None

    stage_t0 = time.perf_counter()
    locals_after = LocalExpansionData(
        order=p,
        centers=centers,
        coefficients=locals_updated,
    )

    # Conjugate symmetry is a property of the COMPLEX solidfmm coefficients
    # only. Real (Dehnen no-sqrt2) locals are not conjugate-symmetric, so
    # applying it there corrupts them (it silently caps far-field accuracy).
    real_basis = str(basis_mode).strip().lower() == "real"
    if not real_basis:
        coefficients_after = enforce_conjugate_symmetry_batch(
            jnp.asarray(locals_after.coefficients),
            order=p,
        )
        coefficients_after = _record_timed_array(
            "_refresh_timing_dual_final_symmetry_seconds",
            stage_t0,
            coefficients_after,
        )
        locals_after = locals_after._replace(coefficients=coefficients_after)
    source_motion_locals_after: Optional[LocalExpansionData]
    if source_motion_locals_updated is not None:
        source_motion_coefficients = (
            jnp.asarray(source_motion_locals_updated)
            if real_basis
            else enforce_conjugate_symmetry_batch(
                jnp.asarray(source_motion_locals_updated),
                order=p,
            )
        )
        source_motion_locals_after = LocalExpansionData(
            order=p,
            centers=centers,
            coefficients=source_motion_coefficients,
        )
    else:
        source_motion_locals_after = None

    return TreeDownwardData(
        interactions=interactions,
        locals=locals_after,
        source_motion_locals=source_motion_locals_after,
    )
