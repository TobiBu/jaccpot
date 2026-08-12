"""Building the inputs the solidfmm downward sweep consumes.

The downward sweep is driven from four bundles -- the initial locals, the
interaction lists, the multipole inputs and the per-level child views -- and this
module builds them, plus the accumulate-from-multipoles step that turns them into
local expansions. ARCHITECTURE §5 documents ``kernels/`` as four families; this is
the first.

The bundles are ``NamedTuple``s on purpose: they cross a ``jax.jit`` boundary, so
**field order is load-bearing** (NUMERICS_AND_JAX §2 -- reordering silently
permutes buffers). Do not re-order them, and do not convert them to dataclasses
without checking donation.

Split out of ``core.py`` (Tier 1.6, A.9 seam 1); every function body is unchanged.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax.numpy as jnp
from beartype.typing import Callable
from jaxtyping import Array
from yggdrax.grouped_interactions import (
    GroupedInteractionBuffers,
    build_grouped_interactions,
)
from yggdrax.interactions import (
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    MACType,
    NodeInteractionList,
    build_well_separated_interactions,
)
from yggdrax.tree import Tree

from jaccpot.downward.local_expansions import (
    LocalExpansionData,
)
from jaccpot.operators.complex_ops import (
    enforce_conjugate_symmetry_batch,
)
from jaccpot.operators.real_harmonics import (
    sh_size,
)
from jaccpot.upward.tree_expansions import TreeUpwardData

from ..dtypes import INDEX_DTYPE, complex_dtype_for_real
from ._m2l import (
    _accumulate_m2l_chunked_scan,
    _accumulate_m2l_fullbatch,
    _accumulate_solidfmm_m2l_grouped,
    _accumulate_solidfmm_m2l_grouped_class_major,
)


class _FarPairCOO(NamedTuple):
    """Compact COO-style far-pair representation for streamed M2L execution."""

    sources: Array
    targets: Array
    active_count: Optional[Array] = None


class _SolidFMMDownwardInit(NamedTuple):
    """Resolved local-buffer initialization for solidfmm downward prep."""

    centers: Array
    locals_coeffs: Array
    total_nodes: int
    coeff_count: int
    dtype: Any


class _SolidFMMDownwardInteractionInputs(NamedTuple):
    """Resolved far-pair arrays for solidfmm downward prep."""

    interactions: NodeInteractionList
    src: Array
    tgt: Array
    pair_count: int
    active_pair_count: Array


class _SolidFMMDownwardMultipoleInputs(NamedTuple):
    """Resolved multipole coefficient payloads for downward accumulation."""

    multip_packed: Array
    source_motion_multip_packed: Optional[Array]
    multip_packed_kernel: Array
    rotation_mode: str


class _SolidFMMDownwardChildInputs(NamedTuple):
    """Resolved child-index arrays for L2L propagation."""

    num_internal_nodes: int
    left_child: Optional[Array]
    right_child: Optional[Array]


def _empty_interaction_storage_for_tree(
    tree: Tree,
    *,
    index_dtype: Any = INDEX_DTYPE,
) -> NodeInteractionList:
    """Construct a minimal zero-pair interaction list for a given tree."""

    total_nodes = int(jnp.asarray(tree.parent).shape[0])
    return NodeInteractionList(
        offsets=jnp.zeros((total_nodes + 1,), dtype=index_dtype),
        sources=jnp.zeros((0,), dtype=index_dtype),
        targets=jnp.zeros((0,), dtype=index_dtype),
        counts=jnp.zeros((total_nodes,), dtype=index_dtype),
        level_offsets=jnp.zeros((1,), dtype=index_dtype),
        target_levels=jnp.zeros((0,), dtype=index_dtype),
    )


def _prepare_solidfmm_downward_interaction_inputs(
    *,
    tree: Tree,
    upward: TreeUpwardData,
    theta: float,
    mac_type: MACType,
    interactions: Optional[NodeInteractionList],
    far_pairs_coo: Optional[_FarPairCOO],
    traversal_config: Optional[DualTreeTraversalConfig],
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]],
    dehnen_radius_scale: float,
) -> _SolidFMMDownwardInteractionInputs:
    """Resolve interaction storage and far-pair arrays for downward prep."""

    resolved_interactions = interactions
    if resolved_interactions is None and far_pairs_coo is None:
        resolved_interactions = build_well_separated_interactions(
            tree,
            upward.geometry,
            theta=theta,
            mac_type=mac_type,
            dehnen_radius_scale=dehnen_radius_scale,
            traversal_config=traversal_config,
            retry_logger=retry_logger,
        )
    if resolved_interactions is None:
        resolved_interactions = _empty_interaction_storage_for_tree(tree)

    if far_pairs_coo is not None:
        src = jnp.asarray(far_pairs_coo.sources, dtype=INDEX_DTYPE)
        tgt = jnp.asarray(far_pairs_coo.targets, dtype=INDEX_DTYPE)
        active_pair_count = (
            jnp.asarray(far_pairs_coo.active_count, dtype=INDEX_DTYPE)
            if far_pairs_coo.active_count is not None
            else jnp.asarray(src.shape[0], dtype=INDEX_DTYPE)
        )
    else:
        src = jnp.asarray(resolved_interactions.sources, dtype=INDEX_DTYPE)
        tgt = jnp.asarray(resolved_interactions.targets, dtype=INDEX_DTYPE)
        active_pair_count = jnp.asarray(src.shape[0], dtype=INDEX_DTYPE)
    return _SolidFMMDownwardInteractionInputs(
        interactions=resolved_interactions,
        src=src,
        tgt=tgt,
        pair_count=int(src.shape[0]),
        active_pair_count=active_pair_count,
    )


def _prepare_solidfmm_downward_init(
    *,
    upward: TreeUpwardData,
    initial_locals: Optional[LocalExpansionData],
    basis_mode: str,
) -> _SolidFMMDownwardInit:
    """Resolve centers and local-expansion buffers for downward prep."""

    p = int(upward.multipoles.order)
    centers = jnp.asarray(upward.multipoles.centers)
    total_nodes = int(centers.shape[0])
    coeff_count = sh_size(p)
    basis_mode_norm = str(basis_mode).strip().lower()
    if basis_mode_norm not in ("complex", "real"):
        raise ValueError("basis_mode must be 'complex' or 'real'")
    dtype = (
        complex_dtype_for_real(centers.dtype)
        if basis_mode_norm == "complex"
        else centers.dtype
    )
    if initial_locals is not None:
        locals_coeffs = jnp.asarray(initial_locals.coefficients)
        if locals_coeffs.shape != (total_nodes, coeff_count):
            raise ValueError("initial_locals must match solidfmm layout")
    else:
        locals_coeffs = jnp.zeros((total_nodes, coeff_count), dtype=dtype)
    return _SolidFMMDownwardInit(
        centers=centers,
        locals_coeffs=locals_coeffs,
        total_nodes=total_nodes,
        coeff_count=coeff_count,
        dtype=dtype,
    )


def _prepare_solidfmm_downward_multipole_inputs(
    *,
    upward: TreeUpwardData,
    dtype: Any,
    basis_mode: str,
    complex_rotation: str,
) -> _SolidFMMDownwardMultipoleInputs:
    """Resolve multipole coefficient payloads for downward accumulation."""

    p = int(upward.multipoles.order)
    basis_mode_norm = str(basis_mode).strip().lower()
    rotation_mode = str(complex_rotation).strip().lower()
    # Complex basis: upward produces packed COMPLEX solidfmm multipoles.
    # Real basis: the native real upward sweep (prepare_real_upward_sweep) produces
    # packed REAL (Dehnen no-sqrt2) coefficients directly -- there is NO complex
    # intermediate and NO complex<->real conversion on the real path.
    packed_raw = jnp.asarray(upward.multipoles.packed)
    source_motion_raw = (
        jnp.asarray(upward.multipoles.source_motion_packed)
        if upward.multipoles.source_motion_packed is not None
        else None
    )
    if basis_mode_norm == "complex":
        if rotation_mode != "solidfmm":
            raise ValueError("complex_rotation must be 'solidfmm'")
        multip_packed = packed_raw.astype(dtype)
        source_motion_multip_packed = (
            source_motion_raw.astype(dtype) if source_motion_raw is not None else None
        )
        multip_packed_kernel = multip_packed
    else:
        # Real (Dehnen no-sqrt2) basis: pass the native real multipoles straight
        # through to the real M2L/L2L/L2P operators. The complex->real Dehnen Q
        # conversion (complex_to_dehnen_real_coeffs) has been REMOVED from the real
        # path per the "real everywhere, never convert bases" contract; the native
        # real upward sweep is the single source of real multipoles. Hard-error if
        # complex-packed multipoles ever reach here (a wiring regression) rather
        # than silently reintroducing a basis conversion.
        if jnp.iscomplexobj(packed_raw):
            raise TypeError(
                "real basis_mode expects REAL multipole coefficients from the "
                "native real upward sweep (prepare_real_upward_sweep), but received "
                "complex-packed multipoles. The complex->real conversion has been "
                "removed from the real path; check prepare_upward_sweep wiring."
            )
        multip_packed = packed_raw.astype(dtype)
        multip_packed_kernel = multip_packed
        source_motion_multip_packed = (
            source_motion_raw.astype(dtype) if source_motion_raw is not None else None
        )
    return _SolidFMMDownwardMultipoleInputs(
        multip_packed=multip_packed,
        source_motion_multip_packed=source_motion_multip_packed,
        multip_packed_kernel=multip_packed_kernel,
        rotation_mode=rotation_mode,
    )


def _prepare_solidfmm_downward_child_inputs(
    tree: Tree,
) -> _SolidFMMDownwardChildInputs:
    """Resolve child-index arrays for L2L propagation."""

    num_internal_nodes = int(jnp.asarray(tree.left_child).shape[0])
    if num_internal_nodes <= 0:
        return _SolidFMMDownwardChildInputs(
            num_internal_nodes=0,
            left_child=None,
            right_child=None,
        )
    return _SolidFMMDownwardChildInputs(
        num_internal_nodes=num_internal_nodes,
        left_child=jnp.asarray(tree.left_child[:num_internal_nodes], dtype=INDEX_DTYPE),
        right_child=jnp.asarray(
            tree.right_child[:num_internal_nodes], dtype=INDEX_DTYPE
        ),
    )


def _solidfmm_downward_accumulate_from_multipoles(
    initial_locals_coeffs: Array,
    multipoles_coeffs: Array,
    *,
    tree: Tree,
    upward: TreeUpwardData,
    interactions: NodeInteractionList,
    centers: Array,
    src: Array,
    tgt: Array,
    pair_count: int,
    active_pair_count: Array,
    order: int,
    rotation_mode: str,
    total_nodes: int,
    chunk_size: int,
    grouped_interactions: bool,
    grouped_buffers: Optional[GroupedInteractionBuffers],
    grouped_segment_starts: Optional[Array],
    grouped_segment_lengths: Optional[Array],
    grouped_segment_class_ids: Optional[Array],
    grouped_segment_sort_permutation: Optional[Array],
    grouped_segment_group_ids: Optional[Array],
    grouped_segment_unique_targets: Optional[Array],
    farfield_mode: str,
    basis_mode: str = "complex",
    m2l_impl: str = "rot_scale",
) -> Array:
    """Run one solidfmm M2L accumulation pass plus symmetry enforcement.

    Both the complex and real (Dehnen no-sqrt2) bases share the grouped /
    class-major / flat dispatch; ``basis_mode`` selects the cached rotation
    blocks and translation kernel (see :func:`_m2l_cached_kernel_dispatch`). The
    non-grouped path uses the dedicated real flat kernels for ``basis_mode ==
    "real"``. Real coefficients carry no conjugate symmetry, so the complex
    symmetry-enforcement step is skipped for the real basis.
    """

    real_basis = str(basis_mode).strip().lower() == "real"

    if grouped_interactions:
        grouped = (
            grouped_buffers
            if grouped_buffers is not None
            else build_grouped_interactions(tree, upward.geometry, interactions)
        )
        mode = str(farfield_mode).strip().lower()
        if mode not in ("pair_grouped", "class_major"):
            raise ValueError("farfield_mode must be 'pair_grouped' or 'class_major'")
        if mode == "class_major":
            locals_updated = _accumulate_solidfmm_m2l_grouped_class_major(
                initial_locals_coeffs,
                multipoles_coeffs,
                centers,
                grouped,
                grouped_segment_starts=grouped_segment_starts,
                grouped_segment_lengths=grouped_segment_lengths,
                grouped_segment_class_ids=grouped_segment_class_ids,
                grouped_segment_sort_permutation=grouped_segment_sort_permutation,
                grouped_segment_group_ids=grouped_segment_group_ids,
                grouped_segment_unique_targets=grouped_segment_unique_targets,
                order=order,
                rotation=rotation_mode,
                total_nodes=total_nodes,
                chunk_size=chunk_size,
                basis_mode=basis_mode,
            )
        else:
            locals_updated = _accumulate_solidfmm_m2l_grouped(
                initial_locals_coeffs,
                multipoles_coeffs,
                centers,
                grouped,
                order=order,
                rotation=rotation_mode,
                total_nodes=total_nodes,
                chunk_size=chunk_size,
                basis_mode=basis_mode,
            )
    else:
        if pair_count <= chunk_size:
            locals_updated = _accumulate_m2l_fullbatch(
                initial_locals_coeffs,
                multipoles_coeffs,
                centers,
                src,
                tgt,
                active_pair_count,
                order=order,
                basis_mode=basis_mode,
                rotation=rotation_mode,
                m2l_impl=m2l_impl,
                total_nodes=total_nodes,
            )
        else:
            locals_updated = _accumulate_m2l_chunked_scan(
                initial_locals_coeffs,
                multipoles_coeffs,
                centers,
                src,
                tgt,
                active_pair_count,
                order=order,
                basis_mode=basis_mode,
                rotation=rotation_mode,
                m2l_impl=m2l_impl,
                total_nodes=total_nodes,
                chunk_size=chunk_size,
            )

    if real_basis:
        return locals_updated
    return enforce_conjugate_symmetry_batch(locals_updated, order=order)
