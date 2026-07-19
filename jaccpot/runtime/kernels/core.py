"""Reusable FMM numerical kernel library (leaf core).

Extracted verbatim from _fmm_impl.py (Phase 2c). Contains the free-function
M2L/L2L batch kernels, solidfmm/real accumulate + propagate routines, Pallas
fast-lane gates, the downward-sweep driver + its input builders, and the
tree/prepared-state evaluation helpers. Leaf: imports operators/downward/
nearfield/upward/pallas + runtime.fmm_constants/fmm_caches, never the engine,
so distributed/ and experimental/ can import kernels without the orchestrator.
"""

from __future__ import annotations

import os
import time
from functools import partial
from typing import Any, Literal, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Callable, Tuple
from jaxtyping import Array
from yggdrax.dense_interactions import DenseInteractionBuffers
from yggdrax.grouped_interactions import (
    GroupedInteractionBuffers,
    build_grouped_interactions,
)
from yggdrax.interactions import (
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    MACType,
    NodeInteractionList,
    NodeNeighborList,
    OctreeNativeNeighborList,
    build_well_separated_interactions,
)
from yggdrax.tree import Tree, get_node_levels

from jaccpot.downward.local_expansions import (
    LocalExpansionData,
    TreeDownwardData,
    translate_local_expansion,
)
from jaccpot.nearfield.near_field import (
    compute_leaf_p2p_accelerations,
    compute_leaf_p2p_accelerations_large_n_accel_only,
)
from jaccpot.operators.complex_ops import (
    complex_rotation_blocks_from_z_solidfmm_batch,
    complex_rotation_blocks_to_z_solidfmm_batch,
    enforce_conjugate_symmetry_batch,
    evaluate_local_complex_derivative_tower_batch,
    evaluate_local_complex_grad_analytic,
    evaluate_local_complex_grad_analytic_preserve_dtype,
    evaluate_local_complex_grad_order4_unrolled,
    evaluate_local_complex_with_grad_analytic_batch,
    l2l_complex_batch,
    m2l_complex_reference_batch,
    m2l_complex_reference_batch_cached_blocks,
)
from jaccpot.operators.m2l_real_rot_scale import (
    m2l_rot_scale_real_batch,
    m2l_rot_scale_real_batch_cached_blocks,
    real_rotation_blocks_from_z_local_batch,
    real_rotation_blocks_to_z_multipole_batch,
)
from jaccpot.operators.multipole_utils import (
    MAX_MULTIPOLE_ORDER,
    level_offset,
    total_coefficients,
)
from jaccpot.operators.real_harmonics import (
    evaluate_local_real_derivative_tower_batch,
    evaluate_local_real_with_grad,
    l2l_real,
    sh_size,
)
from jaccpot.operators.symmetric_tensors import component_lift_index_map_3d
from jaccpot.upward.tree_expansions import TreeUpwardData

from .._octree_adapter import OctreeExecutionData
from ..dtypes import INDEX_DTYPE, as_index, complex_dtype_for_real
from ..fmm_caches import (
    _M2L_FULLBATCH_MAX_PAIRS,
    _grouped_operator_cache_get,
    _grouped_operator_cache_key,
    _grouped_operator_cache_put,
    _grouped_segment_cache_get,
    _grouped_segment_cache_key,
    _grouped_segment_cache_put,
)

ExpansionBasis = Literal["cartesian", "solidfmm", "complex"]
PackedAccelerationDerivatives = tuple[Array, ...]
_STRICT_REFRESH_DETAIL_DIAG_MODES = frozenset(
    {
        "full",
        "tree_sort_only",
        "tree_metadata_only",
        "p2m_only",
        "m2m_only",
        "m2l_only",
        "l2l_only",
        "downward_artifacts_only",
    }
)


def _normalize_strict_refresh_detail_diag_mode(raw: object) -> str:
    mode = str(raw if raw is not None else "full").strip().lower()
    if mode not in _STRICT_REFRESH_DETAIL_DIAG_MODES:
        return "full"
    return mode


class NearfieldInteropData(NamedTuple):
    """Explicit shared leaf/node view used to interoperate with nearfield code."""

    leaf_nodes: Array
    node_ranges: Array
    offsets: Array
    neighbors: Array
    counts: Array
    particle_order_node_ranges: Array
    particle_order_leaf_indices: Array
    particle_order_to_native_leaf: Array
    leaf_particle_indices: Optional[Array] = None
    leaf_particle_mask: Optional[Array] = None
    particle_to_leaf_position: Optional[Array] = None
    neighbor_leaf_positions: Optional[Array] = None


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


@partial(jax.jit, static_argnames=("order",))
def _evaluate_local_cartesian_with_grad_batch(
    coeffs: Array,
    offsets: Array,
    *,
    order: int,
) -> tuple[Array, Array]:
    """Evaluate cartesian local expansions and gradients at batch offsets."""
    leading_shape = coeffs.shape[:-1]
    coeffs_flat = jnp.reshape(coeffs, (-1, coeffs.shape[-1]))
    offsets_flat = jnp.reshape(offsets, (-1, offsets.shape[-1]))

    translated_flat = jax.vmap(
        lambda coeff_row, offset_row: translate_local_expansion(
            coeff_row,
            offset_row,
            order=order,
        )
    )(coeffs_flat, offsets_flat)

    translated = jnp.reshape(
        translated_flat,
        leading_shape + (translated_flat.shape[-1],),
    )

    potentials = translated[..., level_offset(0)]
    if order <= 0:
        gradients = jnp.zeros(leading_shape + (3,), dtype=translated.dtype)
    else:
        first = translated[..., level_offset(1) : level_offset(1) + 3]
        gradients = jnp.stack([first[..., 2], first[..., 1], first[..., 0]], axis=-1)
    return gradients, potentials


def _infer_bounds(positions: Array) -> tuple[Array, Array]:
    """Infer generous bounds for tree construction from particle positions."""

    minimum = jnp.min(positions, axis=0)
    maximum = jnp.max(positions, axis=0)
    span = maximum - minimum
    padding = jnp.maximum(span * 0.05, jnp.full_like(span, 1e-6))
    return minimum - padding, maximum + padding


def _max_leaf_size_from_tree(tree: Tree) -> int:
    """Compute maximum number of particles per leaf node."""
    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    leaf_ranges = tree.node_ranges[num_internal:]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + as_index(1)
    return int(jnp.max(counts))


class _TreeEvaluationSetup(NamedTuple):
    """Prevalidated inputs required by tree-evaluation entry points."""

    locals_data: LocalExpansionData
    positions: Array
    masses: Array
    leaf_nodes: Array
    node_ranges: Array
    max_leaf_size: int
    empty_output: Optional[Union[Array, Tuple[Array, Array]]]


class _EvaluationNodeViews(NamedTuple):
    """Resolved leaf/node metadata for shared nearfield and backend-specific farfield."""

    nearfield: NearfieldInteropData
    farfield_leaf_nodes: Array
    farfield_node_ranges: Array


def _infer_order_from_coeff_count(
    *,
    coeff_count: int,
    expansion_basis: ExpansionBasis,
) -> int:
    """Infer expansion order from static coefficient-array width."""
    if expansion_basis == "solidfmm":
        root = int(round(float(np.sqrt(coeff_count))))
        order = root - 1
        if (order + 1) ** 2 != int(coeff_count):
            raise ValueError(
                "Could not infer solidfmm order from coefficient shape; "
                f"got coeff_count={coeff_count}."
            )
        return order

    for order in range(MAX_MULTIPOLE_ORDER + 1):
        if int(total_coefficients(order)) == int(coeff_count):
            return int(order)
    raise ValueError(
        "Could not infer cartesian order from coefficient shape; "
        f"got coeff_count={coeff_count}."
    )


def _resolve_evaluation_node_views(
    tree: Tree,
    neighbor_list: NodeNeighborList,
    *,
    farfield_leaf_nodes: Optional[Array],
    farfield_node_ranges: Optional[Array],
) -> _EvaluationNodeViews:
    """Resolve shared nearfield views and optional backend-specific farfield views.

    Nearfield continues to use the shared radix-oriented neighbor/leaf layout.
    Farfield may override that view, which is how the octree backend evaluates
    octree-native locals without rewriting nearfield plumbing yet.
    """

    nearfield = _build_nearfield_interop_data(tree, neighbor_list)
    radix_leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    radix_node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    resolved_farfield_leaf_nodes = (
        radix_leaf_nodes
        if farfield_leaf_nodes is None
        else jnp.asarray(farfield_leaf_nodes, dtype=INDEX_DTYPE)
    )
    resolved_farfield_node_ranges = (
        radix_node_ranges
        if farfield_node_ranges is None
        else jnp.asarray(farfield_node_ranges, dtype=INDEX_DTYPE)
    )
    return _EvaluationNodeViews(
        nearfield=nearfield,
        farfield_leaf_nodes=resolved_farfield_leaf_nodes,
        farfield_node_ranges=resolved_farfield_node_ranges,
    )


def _build_nearfield_interop_data(
    tree: Tree,
    neighbor_list: NodeNeighborList,
    *,
    octree: Optional[OctreeExecutionData] = None,
    native_neighbors: Optional[OctreeNativeNeighborList] = None,
) -> NearfieldInteropData:
    """Build the explicit leaf/node view shared by current nearfield helpers.

    The source-of-truth leaf ordering comes from ``neighbor_list``. For octree
    trees, yggdrax now emits that neighbor list in octree-native order while
    still exposing the particle-order leaf mapping needed for target lookup.
    """
    if native_neighbors is not None:
        if octree is None:
            raise ValueError("native octree nearfield data requires octree metadata")
        leaf_nodes = jnp.asarray(native_neighbors.leaf_indices, dtype=INDEX_DTYPE)
        native_offsets = jnp.asarray(native_neighbors.offsets, dtype=INDEX_DTYPE)
        native_neighbors_flat = jnp.asarray(
            native_neighbors.neighbors, dtype=INDEX_DTYPE
        )
        native_counts = jnp.asarray(native_neighbors.counts, dtype=INDEX_DTYPE)
        leaf_count = int(leaf_nodes.shape[0])
        radix_leaf_nodes = jnp.asarray(
            getattr(
                neighbor_list,
                "particle_order_leaf_indices",
                neighbor_list.leaf_indices,
            ),
            dtype=INDEX_DTYPE,
        )
        radix_leaf_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)[
            radix_leaf_nodes
        ]
        radix_leaf_counts = radix_leaf_ranges[:, 1] - radix_leaf_ranges[:, 0] + 1
        carrier_lookup = jnp.full(
            (octree.parent.shape[0],),
            -1,
            dtype=INDEX_DTYPE,
        )
        carrier_lookup = carrier_lookup.at[leaf_nodes].set(
            jnp.arange(leaf_count, dtype=INDEX_DTYPE)
        )
        radix_carrier_pos = carrier_lookup[
            jnp.asarray(octree.radix_leaf_to_oct, dtype=INDEX_DTYPE)
        ]
        carrier_particle_counts = jax.ops.segment_sum(
            radix_leaf_counts.astype(INDEX_DTYPE),
            radix_carrier_pos,
            leaf_count,
        )
        max_particles = int(jnp.max(carrier_particle_counts)) if leaf_count > 0 else 0

        if max_particles > 0:
            max_radix_leaf_particles = int(jnp.max(radix_leaf_counts))
            local_offsets = jnp.arange(max_radix_leaf_particles, dtype=INDEX_DTYPE)
            radix_particle_idx = (
                radix_leaf_ranges[:, 0][:, None] + local_offsets[None, :]
            )
            radix_particle_valid = local_offsets[None, :] < radix_leaf_counts[:, None]
            flat_particle_idx = radix_particle_idx.reshape(-1)
            flat_valid = radix_particle_valid.reshape(-1)
            flat_carrier_pos = jnp.repeat(radix_carrier_pos, max_radix_leaf_particles)
            safe_carrier_pos = jnp.where(flat_valid, flat_carrier_pos, leaf_count)
            order = jnp.argsort(safe_carrier_pos, stable=True)
            sorted_valid = flat_valid[order]
            sorted_carrier = safe_carrier_pos[order]
            sorted_particle_idx = flat_particle_idx[order]
            valid_int = sorted_valid.astype(INDEX_DTYPE)
            running = jnp.cumsum(valid_int, dtype=INDEX_DTYPE) - valid_int
            changed = jnp.concatenate(
                [
                    jnp.ones((1,), dtype=bool),
                    sorted_carrier[1:] != sorted_carrier[:-1],
                ]
            )
            group_starts = jnp.where(
                sorted_valid & changed,
                running,
                jnp.zeros_like(running),
            )
            group_starts = jnp.maximum.accumulate(group_starts)
            sorted_slots = running - group_starts
            row = jnp.where(sorted_valid, sorted_carrier, leaf_count)
            col = jnp.where(sorted_valid, sorted_slots, 0)
            leaf_particle_indices = jnp.zeros(
                (leaf_count + 1, max_particles),
                dtype=INDEX_DTYPE,
            )
            leaf_particle_mask = jnp.zeros((leaf_count + 1, max_particles), dtype=bool)
            leaf_particle_indices = leaf_particle_indices.at[row, col].set(
                jnp.where(sorted_valid, sorted_particle_idx, 0),
                mode="drop",
            )
            leaf_particle_mask = leaf_particle_mask.at[row, col].set(
                sorted_valid,
                mode="drop",
            )
            leaf_particle_indices = leaf_particle_indices[:leaf_count]
            leaf_particle_mask = leaf_particle_mask[:leaf_count]
            particle_to_leaf_position = jnp.zeros(
                (tree.positions_sorted.shape[0],),
                dtype=INDEX_DTYPE,
            )
            particle_to_leaf_position = particle_to_leaf_position.at[
                flat_particle_idx[flat_valid]
            ].set(flat_carrier_pos[flat_valid])
        else:
            leaf_particle_indices = jnp.zeros((leaf_count, 0), dtype=INDEX_DTYPE)
            leaf_particle_mask = jnp.zeros((leaf_count, 0), dtype=bool)
            particle_to_leaf_position = jnp.zeros(
                (tree.positions_sorted.shape[0],),
                dtype=INDEX_DTYPE,
            )

        native_neighbor_leaf_positions = getattr(
            native_neighbors,
            "neighbor_leaf_positions",
            None,
        )
        if native_neighbor_leaf_positions is not None:
            neighbor_leaf_positions = jnp.asarray(
                native_neighbor_leaf_positions,
                dtype=INDEX_DTYPE,
            )
        else:
            if leaf_count > 0:
                max_nbr = int(jnp.max(native_counts))
            else:
                max_nbr = 0
            if max_nbr > 0:
                nbr_offsets = jnp.arange(max_nbr, dtype=INDEX_DTYPE)
                nbr_idx = native_offsets[:-1, None] + nbr_offsets[None, :]
                nbr_valid = nbr_offsets[None, :] < native_counts[:, None]
                nbr_safe_idx = jnp.where(nbr_valid, nbr_idx, 0)
                nbr_nodes = native_neighbors_flat[nbr_safe_idx]
                neighbor_leaf_positions = carrier_lookup[nbr_nodes]
                neighbor_leaf_positions = jnp.where(
                    nbr_valid,
                    neighbor_leaf_positions,
                    jnp.asarray(-1, dtype=INDEX_DTYPE),
                )
            else:
                neighbor_leaf_positions = jnp.zeros((leaf_count, 0), dtype=INDEX_DTYPE)

        oct_node_ranges = jnp.asarray(octree.node_ranges, dtype=INDEX_DTYPE)
        particle_order_leaf_indices = jnp.asarray(
            native_neighbors.particle_order_leaf_indices,
            dtype=INDEX_DTYPE,
        )
        return NearfieldInteropData(
            leaf_nodes=leaf_nodes,
            node_ranges=oct_node_ranges,
            offsets=native_offsets,
            neighbors=native_neighbors_flat,
            counts=native_counts,
            particle_order_node_ranges=oct_node_ranges,
            particle_order_leaf_indices=particle_order_leaf_indices,
            particle_order_to_native_leaf=jnp.asarray(
                native_neighbors.particle_order_to_native_leaf,
                dtype=INDEX_DTYPE,
            ),
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
            particle_to_leaf_position=particle_to_leaf_position,
            neighbor_leaf_positions=neighbor_leaf_positions,
        )

    del octree
    leaf_indices = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    particle_order_leaf_indices = jnp.asarray(
        getattr(
            neighbor_list, "particle_order_leaf_indices", neighbor_list.leaf_indices
        ),
        dtype=INDEX_DTYPE,
    )
    nbr_counts = jnp.asarray(neighbor_list.counts, dtype=INDEX_DTYPE)
    num_leaves = int(leaf_indices.shape[0])
    payload_neighbor_leaf_positions = getattr(
        neighbor_list,
        "neighbor_leaf_positions",
        None,
    )
    if payload_neighbor_leaf_positions is not None:
        neighbor_leaf_positions = jnp.asarray(
            payload_neighbor_leaf_positions,
            dtype=INDEX_DTYPE,
        )
    else:
        if num_leaves > 0:
            max_nbr = int(jnp.max(nbr_counts))
        else:
            max_nbr = 0
        if max_nbr > 0:
            total_nodes = int(tree.node_ranges.shape[0])
            leaf_lookup = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
            leaf_lookup = leaf_lookup.at[leaf_indices].set(
                jnp.arange(num_leaves, dtype=INDEX_DTYPE)
            )
            offsets = jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE)
            neighbors = jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE)
            nbr_offsets = jnp.arange(max_nbr, dtype=INDEX_DTYPE)
            nbr_idx = offsets[:-1, None] + nbr_offsets[None, :]
            nbr_valid = nbr_offsets[None, :] < nbr_counts[:, None]
            nbr_safe_idx = jnp.where(nbr_valid, nbr_idx, 0)
            nbr_nodes = neighbors[nbr_safe_idx]
            neighbor_leaf_positions = leaf_lookup[nbr_nodes]
            neighbor_leaf_positions = jnp.where(
                nbr_valid,
                neighbor_leaf_positions,
                jnp.asarray(-1, dtype=INDEX_DTYPE),
            )
        else:
            neighbor_leaf_positions = jnp.zeros((num_leaves, 0), dtype=INDEX_DTYPE)

    return NearfieldInteropData(
        leaf_nodes=leaf_indices,
        node_ranges=jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        offsets=jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE),
        neighbors=jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE),
        counts=jnp.asarray(neighbor_list.counts, dtype=INDEX_DTYPE),
        particle_order_node_ranges=jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        particle_order_leaf_indices=particle_order_leaf_indices,
        particle_order_to_native_leaf=jnp.asarray(
            getattr(
                neighbor_list,
                "particle_order_to_native_leaf",
                jnp.arange(leaf_indices.shape[0], dtype=INDEX_DTYPE),
            ),
            dtype=INDEX_DTYPE,
        ),
        leaf_particle_indices=None,
        leaf_particle_mask=None,
        particle_to_leaf_position=None,
        neighbor_leaf_positions=neighbor_leaf_positions,
    )


def _prepare_tree_evaluation_inputs(
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    locals_or_downward: Union[LocalExpansionData, TreeDownwardData],
    neighbor_list: NodeNeighborList,
    *,
    farfield_local_data: Optional[LocalExpansionData],
    farfield_leaf_nodes: Optional[Array],
    farfield_node_ranges: Optional[Array],
    max_leaf_size: Optional[int],
    return_potential: bool,
) -> _TreeEvaluationSetup:
    """Validate and normalize tree-evaluation inputs for eager/JIT paths."""
    locals_data = (
        locals_or_downward.locals
        if isinstance(locals_or_downward, TreeDownwardData)
        else locals_or_downward
    )
    farfield_locals = (
        locals_data if farfield_local_data is None else farfield_local_data
    )
    node_views = _resolve_evaluation_node_views(
        tree,
        neighbor_list,
        farfield_leaf_nodes=farfield_leaf_nodes,
        farfield_node_ranges=farfield_node_ranges,
    )

    if farfield_locals.centers.shape[0] != node_views.farfield_node_ranges.shape[0]:
        raise ValueError("local expansions must align with evaluation node ranges")
    if (
        farfield_locals.coefficients.shape[0]
        != node_views.farfield_node_ranges.shape[0]
    ):
        raise ValueError("local expansions must align with evaluation node ranges")

    positions = jnp.asarray(positions_sorted)
    masses = jnp.asarray(masses_sorted)
    leaf_nodes = node_views.farfield_leaf_nodes
    node_ranges = node_views.farfield_node_ranges

    if leaf_nodes.size == 0:
        zeros = jnp.zeros_like(positions)
        if return_potential:
            pot_zeros = jnp.zeros((positions.shape[0],), dtype=zeros.dtype)
            empty: Optional[Union[Array, Tuple[Array, Array]]] = (
                zeros,
                pot_zeros,
            )
        else:
            empty = zeros
        resolved_max_leaf = 0 if max_leaf_size is None else int(max_leaf_size)
        return _TreeEvaluationSetup(
            farfield_locals,
            positions,
            masses,
            leaf_nodes,
            node_ranges,
            resolved_max_leaf,
            empty,
        )
    if max_leaf_size is None:
        leaf_ranges = node_ranges[leaf_nodes]
        counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
        try:
            resolved_max_leaf = int(jnp.max(counts).item())
        except TypeError as exc:
            raise ValueError(
                "max_leaf_size must be provided when tracing or JIT-compiling"
            ) from exc
    else:
        resolved_max_leaf = int(max_leaf_size)

    return _TreeEvaluationSetup(
        farfield_locals,
        positions,
        masses,
        leaf_nodes,
        node_ranges,
        resolved_max_leaf,
        None,
    )


@partial(jax.jit, static_argnames=("order", "rotation"))
def _m2l_complex_batch_kernel(
    src_mult: Array,
    deltas: Array,
    *,
    order: int,
    rotation: str,
) -> Array:
    """Vectorized complex-basis M2L kernel for one interaction batch."""
    return m2l_complex_reference_batch(
        src_mult,
        deltas,
        order=order,
        rotation=rotation,
    )


@partial(jax.jit, static_argnames=("order",))
def _m2l_complex_batch_cached_kernel(
    src_mult: Array,
    deltas: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    *,
    order: int,
) -> Array:
    """Vectorized complex M2L kernel using precomputed rotation blocks."""
    return m2l_complex_reference_batch_cached_blocks(
        src_mult,
        deltas,
        blocks_to_z,
        blocks_from_z,
        order=order,
    )


def _m2l_cached_kernel_dispatch(
    src_mult: Array,
    deltas: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    *,
    order: int,
    basis_mode: str,
) -> Array:
    """Apply precomputed rotation blocks in the complex or real basis.

    ``basis_mode`` is a Python string (static under jit), so this branches at
    trace time. The real branch uses the Dehnen no-sqrt2 cached kernel; the
    complex branch is unchanged.
    """
    if str(basis_mode).strip().lower() == "real":
        return m2l_rot_scale_real_batch_cached_blocks(
            src_mult, deltas, blocks_to_z, blocks_from_z, order=order
        )
    return _m2l_complex_batch_cached_kernel(
        src_mult, deltas, blocks_to_z, blocks_from_z, order=order
    )


def _rotation_blocks_for_grouped_classes(
    *,
    order: int,
    rotation: str,
    class_keys: Array,
    class_deltas: Array,
    dtype: jnp.dtype,
    basis_mode: str = "complex",
) -> tuple[Array, Array]:
    """Resolve rotation blocks for all grouped classes with cache reuse.

    For ``basis_mode == "real"`` the Dehnen no-sqrt2 real rotation blocks are
    built (multipole world->z and local z->world) and the ``rotation`` argument
    is ignored (the real path has a single rotation construction).
    """
    num_classes = int(class_deltas.shape[0])
    max_m = 2 * int(order) + 1
    empty_shape = (0, int(order) + 1, max_m, max_m)
    if num_classes == 0:
        empty = jnp.zeros(empty_shape, dtype=dtype)
        return empty, empty

    real_basis = str(basis_mode).strip().lower() == "real"
    cache_key = _grouped_operator_cache_key(
        order=order,
        rotation=("real" if real_basis else rotation),
        dtype=dtype,
        class_keys=class_keys,
        class_deltas=class_deltas,
    )
    if cache_key is not None:
        cached = _grouped_operator_cache_get(cache_key)
        if cached is not None:
            return cached

    deltas = jnp.asarray(class_deltas)
    if real_basis:
        blocks_to = real_rotation_blocks_to_z_multipole_batch(
            deltas, order=order, dtype=dtype
        )
        blocks_from = real_rotation_blocks_from_z_local_batch(
            deltas, order=order, dtype=dtype
        )
        if cache_key is not None:
            _grouped_operator_cache_put(cache_key, (blocks_to, blocks_from))
        return blocks_to, blocks_from

    if rotation == "solidfmm":
        blocks_to = complex_rotation_blocks_to_z_solidfmm_batch(
            deltas,
            order=order,
            basis="multipole",
            dtype=dtype,
        )
        blocks_from = complex_rotation_blocks_from_z_solidfmm_batch(
            deltas,
            order=order,
            basis="local",
            dtype=dtype,
        )
    else:
        raise ValueError(
            "grouped operator cache currently supports rotation='solidfmm'"
        )
    if cache_key is not None:
        _grouped_operator_cache_put(cache_key, (blocks_to, blocks_from))
    return blocks_to, blocks_from


def _chunk_segment_scatter_add(
    local_accum: Array,
    contribs: Array,
    tgt_chunk: Array,
    valid: Array,
    *,
    chunk_size: int,
) -> Array:
    """Reduce one fixed-width chunk by target index and scatter-add into locals."""
    masked_targets = jnp.where(valid, tgt_chunk, jnp.iinfo(INDEX_DTYPE).max)
    sort_idx = jnp.argsort(masked_targets)
    sorted_keys = masked_targets[sort_idx]
    tgt_sorted = tgt_chunk[sort_idx]
    contribs_sorted = contribs[sort_idx]
    valid_sorted = valid[sort_idx]

    contribs_sorted = jnp.where(valid_sorted[:, None], contribs_sorted, 0)
    new_group = jnp.concatenate(
        (
            jnp.asarray([True], dtype=bool),
            sorted_keys[1:] != sorted_keys[:-1],
        ),
        axis=0,
    )
    group_ids = jnp.cumsum(new_group.astype(INDEX_DTYPE)) - jnp.asarray(
        1,
        dtype=INDEX_DTYPE,
    )
    reduced = jax.ops.segment_sum(contribs_sorted, group_ids, chunk_size)

    unique_targets = jnp.zeros((chunk_size,), dtype=INDEX_DTYPE)
    unique_targets = unique_targets.at[group_ids].set(tgt_sorted)
    unique_valid = jnp.zeros((chunk_size,), dtype=bool)
    unique_valid = unique_valid.at[group_ids].set(valid_sorted)
    safe_targets = jnp.where(unique_valid, unique_targets, 0)
    reduced = jnp.where(unique_valid[:, None], reduced, 0)
    return local_accum.at[safe_targets].add(reduced)


@partial(
    jax.jit,
    static_argnames=("order", "total_nodes", "chunk_size", "basis_mode"),
    donate_argnums=(0,),
)
def _accumulate_solidfmm_m2l_grouped_chunked_scan(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    src_sorted: Array,
    tgt_sorted: Array,
    class_ids_sorted: Array,
    blocks_to_classes: Array,
    blocks_from_classes: Array,
    *,
    order: int,
    total_nodes: int,
    chunk_size: int,
    basis_mode: str = "complex",
) -> Array:
    """Accumulate grouped solidfmm M2L contributions via chunked scan."""
    pair_count = src_sorted.shape[0]
    starts = jnp.arange(0, pair_count, chunk_size, dtype=INDEX_DTYPE)

    def body(local_accum: Array, start_idx: Array) -> tuple[Array, None]:
        offset = jnp.arange(chunk_size, dtype=INDEX_DTYPE)
        idx = start_idx + offset
        valid = idx < pair_count
        safe_idx = jnp.where(valid, idx, 0)

        src_chunk = src_sorted[safe_idx]
        tgt_chunk = tgt_sorted[safe_idx]
        cls_chunk = class_ids_sorted[safe_idx]
        src_mult = multip_packed[src_chunk]
        deltas = centers[tgt_chunk] - centers[src_chunk]
        blocks_to = blocks_to_classes[cls_chunk]
        blocks_from = blocks_from_classes[cls_chunk]

        contribs = _m2l_cached_kernel_dispatch(
            src_mult,
            deltas,
            blocks_to,
            blocks_from,
            order=order,
            basis_mode=basis_mode,
        ).astype(locals_coeffs.dtype)
        local_accum = _chunk_segment_scatter_add(
            local_accum,
            contribs,
            tgt_chunk,
            valid,
            chunk_size=chunk_size,
        )
        return local_accum, None

    local_accum, _ = jax.lax.scan(body, locals_coeffs, starts)
    return local_accum


@partial(
    jax.jit,
    static_argnames=("order", "total_nodes", "basis_mode"),
    donate_argnums=(0,),
)
def _accumulate_solidfmm_m2l_grouped_fullbatch(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    src_sorted: Array,
    tgt_sorted: Array,
    class_ids_sorted: Array,
    blocks_to_classes: Array,
    blocks_from_classes: Array,
    *,
    order: int,
    total_nodes: int,
    basis_mode: str = "complex",
) -> Array:
    """Accumulate grouped solidfmm M2L contributions in one full batch."""
    src_mult = multip_packed[src_sorted]
    deltas = centers[tgt_sorted] - centers[src_sorted]
    blocks_to = blocks_to_classes[class_ids_sorted]
    blocks_from = blocks_from_classes[class_ids_sorted]
    contribs = _m2l_cached_kernel_dispatch(
        src_mult,
        deltas,
        blocks_to,
        blocks_from,
        order=order,
        basis_mode=basis_mode,
    ).astype(locals_coeffs.dtype)
    return locals_coeffs + jax.ops.segment_sum(contribs, tgt_sorted, total_nodes)


def _build_grouped_class_segments(
    grouped: GroupedInteractionBuffers,
    *,
    chunk_size: int,
) -> tuple[Array, Array, Array]:
    """Build compact class-major segment metadata for chunked execution."""
    cache_key = _grouped_segment_cache_key(
        class_offsets=grouped.class_offsets,
        class_targets=grouped.class_targets,
        chunk_size=int(chunk_size),
    )
    if cache_key is not None:
        cached = _grouped_segment_cache_get(cache_key)
        if cached is not None:
            return cached

    class_offsets = np.asarray(jax.device_get(grouped.class_offsets), dtype=np.int64)
    if class_offsets.size <= 1:
        empty = jnp.zeros((0,), dtype=INDEX_DTYPE)
        result = (empty, empty, empty)
        if cache_key is not None:
            _grouped_segment_cache_put(cache_key, result)
        return result

    starts: list[int] = []
    lengths: list[int] = []
    class_ids: list[int] = []
    for class_idx in range(class_offsets.shape[0] - 1):
        start = int(class_offsets[class_idx])
        end = int(class_offsets[class_idx + 1])
        while start < end:
            seg_len = min(int(chunk_size), end - start)
            starts.append(start)
            lengths.append(seg_len)
            class_ids.append(class_idx)
            start += seg_len

    if len(starts) == 0:
        result = (
            jnp.asarray(starts, dtype=INDEX_DTYPE),
            jnp.asarray(lengths, dtype=INDEX_DTYPE),
            jnp.asarray(class_ids, dtype=INDEX_DTYPE),
        )
        if cache_key is not None:
            _grouped_segment_cache_put(cache_key, result)
        return result

    result = (
        jnp.asarray(starts, dtype=INDEX_DTYPE),
        jnp.asarray(lengths, dtype=INDEX_DTYPE),
        jnp.asarray(class_ids, dtype=INDEX_DTYPE),
    )
    if cache_key is not None:
        _grouped_segment_cache_put(cache_key, result)
    return result


@partial(
    jax.jit,
    static_argnames=("order", "total_nodes", "chunk_size", "basis_mode"),
    donate_argnums=(0,),
)
def _accumulate_solidfmm_m2l_class_major_chunked_scan(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    src_sorted: Array,
    tgt_sorted: Array,
    segment_starts: Array,
    segment_lengths: Array,
    segment_class_ids: Array,
    blocks_to_classes: Array,
    blocks_from_classes: Array,
    *,
    order: int,
    total_nodes: int,
    chunk_size: int,
    basis_mode: str = "complex",
) -> Array:
    """Accumulate class-major grouped M2L contributions via chunked scan."""
    num_segments = segment_starts.shape[0]
    if num_segments == 0:
        return locals_coeffs

    offsets = jnp.arange(chunk_size, dtype=INDEX_DTYPE)

    def body(local_accum: Array, seg_idx: Array) -> tuple[Array, None]:
        start = segment_starts[seg_idx]
        seg_len = segment_lengths[seg_idx]
        cls = segment_class_ids[seg_idx]
        idx = start + offsets
        valid = offsets < seg_len
        safe_idx = jnp.where(valid, idx, 0)

        src_chunk = src_sorted[safe_idx]
        tgt_chunk = tgt_sorted[safe_idx]
        src_mult = multip_packed[src_chunk]
        deltas = centers[tgt_chunk] - centers[src_chunk]

        block_to = blocks_to_classes[cls]
        block_from = blocks_from_classes[cls]
        blocks_to = jnp.broadcast_to(block_to, (chunk_size,) + block_to.shape)
        blocks_from = jnp.broadcast_to(block_from, (chunk_size,) + block_from.shape)

        contribs = _m2l_cached_kernel_dispatch(
            src_mult,
            deltas,
            blocks_to,
            blocks_from,
            order=order,
            basis_mode=basis_mode,
        ).astype(locals_coeffs.dtype)
        contribs = jnp.where(valid[:, None], contribs, 0)
        masked_targets = jnp.where(valid, tgt_chunk, jnp.iinfo(INDEX_DTYPE).max)
        sort_idx = jnp.argsort(masked_targets)
        tgt_sorted_chunk = tgt_chunk[sort_idx]
        contribs_sorted = contribs[sort_idx]
        valid_sorted = valid[sort_idx]
        prev_targets = jnp.concatenate(
            [
                jnp.asarray([-1], dtype=INDEX_DTYPE),
                tgt_sorted_chunk[:-1],
            ]
        )
        group_starts = valid_sorted & (
            jnp.logical_not(jnp.roll(valid_sorted, 1))
            | (tgt_sorted_chunk != prev_targets)
        )
        group_ids = jnp.cumsum(group_starts.astype(INDEX_DTYPE)) - 1
        safe_group_ids = jnp.where(valid_sorted, group_ids, 0)
        reduced = jax.ops.segment_sum(contribs_sorted, safe_group_ids, chunk_size)
        unique_targets = jnp.where(valid_sorted, tgt_sorted_chunk, 0)
        return local_accum.at[unique_targets].add(reduced), None

    local_accum, _ = jax.lax.scan(
        body,
        locals_coeffs,
        jnp.arange(num_segments, dtype=INDEX_DTYPE),
    )
    return local_accum


def _accumulate_solidfmm_m2l_grouped_class_major(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    grouped: GroupedInteractionBuffers,
    grouped_segment_starts: Optional[Array],
    grouped_segment_lengths: Optional[Array],
    grouped_segment_class_ids: Optional[Array],
    grouped_segment_sort_permutation: Optional[Array],
    grouped_segment_group_ids: Optional[Array],
    grouped_segment_unique_targets: Optional[Array],
    *,
    order: int,
    rotation: str,
    total_nodes: int,
    chunk_size: int,
    basis_mode: str = "complex",
) -> Array:
    """Class-major grouped accumulation without per-pair operator gathers."""
    del (
        grouped_segment_sort_permutation,
        grouped_segment_group_ids,
        grouped_segment_unique_targets,
    )

    if rotation not in ("solidfmm",):
        src = grouped.class_sources
        tgt = grouped.class_targets
        return _accumulate_m2l_fullbatch(
            locals_coeffs,
            multip_packed,
            centers,
            src,
            tgt,
            jnp.asarray(src.shape[0], dtype=INDEX_DTYPE),
            order=order,
            basis_mode=basis_mode,
            rotation=rotation,
            total_nodes=total_nodes,
        )

    blocks_to_classes, blocks_from_classes = _rotation_blocks_for_grouped_classes(
        order=order,
        rotation=rotation,
        class_keys=jnp.asarray(grouped.class_keys, dtype=jnp.int32),
        class_deltas=jnp.asarray(grouped.class_displacements),
        dtype=multip_packed.dtype,
        basis_mode=basis_mode,
    )
    if (
        grouped_segment_starts is None
        or grouped_segment_lengths is None
        or grouped_segment_class_ids is None
    ):
        (
            segment_starts,
            segment_lengths,
            segment_class_ids,
        ) = _build_grouped_class_segments(
            grouped,
            chunk_size=int(chunk_size),
        )
    else:
        segment_starts = jnp.asarray(grouped_segment_starts, dtype=INDEX_DTYPE)
        segment_lengths = jnp.asarray(grouped_segment_lengths, dtype=INDEX_DTYPE)
        segment_class_ids = jnp.asarray(grouped_segment_class_ids, dtype=INDEX_DTYPE)
    return _accumulate_solidfmm_m2l_class_major_chunked_scan(
        locals_coeffs,
        multip_packed,
        centers,
        jnp.asarray(grouped.class_sources, dtype=INDEX_DTYPE),
        jnp.asarray(grouped.class_targets, dtype=INDEX_DTYPE),
        segment_starts,
        segment_lengths,
        segment_class_ids,
        blocks_to_classes,
        blocks_from_classes,
        order=order,
        total_nodes=total_nodes,
        chunk_size=int(chunk_size),
        basis_mode=basis_mode,
    )


def _accumulate_solidfmm_m2l_grouped(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    grouped: GroupedInteractionBuffers,
    *,
    order: int,
    rotation: str,
    total_nodes: int,
    chunk_size: int,
    basis_mode: str = "complex",
) -> Array:
    """Grouped M2L accumulation using cached class blocks and pair chunking."""

    if rotation not in ("solidfmm",):
        # Keep existing sparse path semantics for other conventions.
        src = grouped.class_sources
        tgt = grouped.class_targets
        return _accumulate_m2l_fullbatch(
            locals_coeffs,
            multip_packed,
            centers,
            src,
            tgt,
            jnp.asarray(src.shape[0], dtype=INDEX_DTYPE),
            order=order,
            basis_mode=basis_mode,
            rotation=rotation,
            total_nodes=total_nodes,
        )

    src_sorted = grouped.class_sources
    tgt_sorted = grouped.class_targets
    class_ids = jnp.asarray(grouped.class_ids, dtype=INDEX_DTYPE)
    class_keys = jnp.asarray(grouped.class_keys, dtype=jnp.int32)
    class_deltas = jnp.asarray(grouped.class_displacements)

    blocks_to_classes, blocks_from_classes = _rotation_blocks_for_grouped_classes(
        order=order,
        rotation=rotation,
        class_keys=class_keys,
        class_deltas=class_deltas,
        dtype=multip_packed.dtype,
        basis_mode=basis_mode,
    )
    if int(src_sorted.shape[0]) <= min(int(chunk_size), _M2L_FULLBATCH_MAX_PAIRS):
        return _accumulate_solidfmm_m2l_grouped_fullbatch(
            locals_coeffs,
            multip_packed,
            centers,
            src_sorted,
            tgt_sorted,
            class_ids,
            blocks_to_classes,
            blocks_from_classes,
            order=order,
            total_nodes=total_nodes,
        )
    return _accumulate_solidfmm_m2l_grouped_chunked_scan(
        locals_coeffs,
        multip_packed,
        centers,
        src_sorted,
        tgt_sorted,
        class_ids,
        blocks_to_classes,
        blocks_from_classes,
        order=order,
        total_nodes=total_nodes,
        chunk_size=int(chunk_size),
        basis_mode=basis_mode,
    )


@partial(jax.jit, static_argnames=("order", "rotation"))
def _l2l_complex_batch_kernel(
    coeffs: Array,
    deltas: Array,
    *,
    order: int,
    rotation: str,
) -> Array:
    """Vectorized complex-basis L2L translation kernel."""
    return l2l_complex_batch(coeffs, deltas, order=order, rotation=rotation)


@partial(jax.jit, static_argnames=("order", "m2l_impl"))
def _m2l_real_batch_kernel(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    m2l_impl: str,
) -> Array:
    """Vectorized real-basis M2L translation kernel."""
    mode = str(m2l_impl).strip().lower()
    if mode != "rot_scale":
        raise ValueError("real-basis m2l_impl must be 'rot_scale'")
    return m2l_rot_scale_real_batch(multipoles, deltas, order=order, use_pallas=False)


def _real_m2l_pallas_active() -> bool:
    """Whether to route the real-basis M2L z-core through the Pallas kernel.

    Gated by ``JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS`` and the sm_80+ real-M2L
    support check (falls back to the pure-JAX rot-scale otherwise). Trace-time;
    the flag does not change within a compiled run.
    """
    flag = (
        str(os.environ.get("JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS", "0"))
        .strip()
        .lower()
    )
    if flag not in {"1", "true", "yes", "on"}:
        return False
    try:
        from jaccpot.pallas.m2l_core_z_real import pallas_m2l_real_supported

        return bool(pallas_m2l_real_supported())
    except Exception:
        return False


def _m2l_real_batch_kernel_fused_pallas(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    m2l_impl: str,
) -> Array:
    """Real-basis M2L via the FULLY-fused Pallas kernel (rotate+z-translate+rotate
    in one launch). Builds the real rotation blocks + radii from deltas."""
    mode = str(m2l_impl).strip().lower()
    if mode != "rot_scale":
        raise ValueError("real-basis m2l_impl must be 'rot_scale'")
    from jaccpot.operators.m2l_real_rot_scale import (
        real_rotation_blocks_from_z_local_batch,
        real_rotation_blocks_to_z_multipole_batch,
    )
    from jaccpot.pallas.m2l_real_fused import m2l_real_fused_pallas

    r = jnp.linalg.norm(deltas, axis=1)
    bto = real_rotation_blocks_to_z_multipole_batch(
        deltas, order=order, dtype=multipoles.dtype
    )
    bfr = real_rotation_blocks_from_z_local_batch(
        deltas, order=order, dtype=multipoles.dtype
    )
    return m2l_real_fused_pallas(multipoles, bto, bfr, r, order=order)


def _apply_real_m2l(src_mult, deltas, *, order, m2l_impl):
    """Real-basis batched M2L: fully-fused Pallas kernel when enabled, else pure-JAX.

    When the fused-M2L Pallas flag is active, route through the single-launch fused
    kernel (rotate -> z-translate -> rotate-back on-chip), collapsing the per-pair
    JAX rotation launches. Otherwise the pure-JAX rot-scale path.
    """
    if _real_m2l_pallas_active():
        return _m2l_real_batch_kernel_fused_pallas(
            src_mult, deltas, order=order, m2l_impl=m2l_impl
        )
    return _m2l_real_batch_kernel(src_mult, deltas, order=order, m2l_impl=m2l_impl)


@partial(jax.jit, static_argnames=("order",))
def _l2l_real_batch_kernel(
    coeffs: Array,
    deltas: Array,
    *,
    order: int,
) -> Array:
    """Vectorized real-basis L2L translation kernel."""
    return jax.vmap(lambda c, d: l2l_real(c, d, order=order))(coeffs, deltas)


def _fused_complex_m2l_pallas_active() -> bool:
    """Whether to route the complex-basis M2L through the fused Pallas kernel.

    Gated by ``JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS`` and the sm_80+ support
    check; falls back to the solidfmm reference batch on unsupported hardware.
    Evaluated at trace time; the flag does not change within a compiled run.

    Returns
    -------
    bool
        ``True`` when the flag is set and an Ampere+ GPU is available.
    """
    flag = (
        str(os.environ.get("JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS", "0"))
        .strip()
        .lower()
    )
    if flag not in {"1", "true", "yes", "on"}:
        return False
    try:
        from jaccpot.pallas.m2l_complex_fused import (
            pallas_m2l_complex_fused_supported,
        )

        return bool(pallas_m2l_complex_fused_supported())
    except Exception:
        return False


def _m2l_complex_batch_kernel_fused_pallas(
    src_mult: Array,
    deltas: Array,
    *,
    order: int,
) -> Array:
    """Complex-basis M2L via the fully-fused Pallas kernel.

    Adapter over solidfmm: solidfmm is the sole rotation strategy, and it already
    materialises the block-diagonal rotate-to-z / rotate-from-z matrices the fused
    kernel consumes (``complex_rotation_blocks_*_z_solidfmm_batch``, padded to
    ``[N, p+1, 2p+1, 2p+1]``). This builds those blocks plus the pair radii and
    hands them to the kernel, which keeps the rotate -> z-translate -> rotate-back
    intermediates on-chip. Numerically equivalent to ``_m2l_complex_batch_kernel``
    (the solidfmm reference); the kernel is purely an execution accelerator.

    Parameters
    ----------
    src_mult : Array
        Complex multipole coefficients ``[N, (p+1)^2]`` for each pair.
    deltas : Array
        Target-minus-source center displacements ``[N, 3]``.
    order : int
        Expansion order ``p``.

    Returns
    -------
    Array
        Complex local contributions ``[N, (p+1)^2]``.
    """
    from jaccpot.pallas.m2l_complex_fused import m2l_complex_fused_pallas

    r = jnp.sqrt(jnp.sum(deltas * deltas, axis=-1))
    blocks_to_z = complex_rotation_blocks_to_z_solidfmm_batch(
        deltas,
        order=order,
        basis="multipole",
        dtype=src_mult.dtype,
    )
    blocks_from_z = complex_rotation_blocks_from_z_solidfmm_batch(
        deltas,
        order=order,
        basis="local",
        dtype=src_mult.dtype,
    )
    return m2l_complex_fused_pallas(
        src_mult, blocks_to_z, blocks_from_z, r, order=order
    )


def _apply_complex_m2l(
    src_mult: Array,
    deltas: Array,
    *,
    order: int,
    rotation: str,
) -> Array:
    """Complex-basis batched M2L: fused Pallas kernel when enabled, else solidfmm.

    When the fused-M2L Pallas flag is active (and the GPU is Ampere+), route
    through the single-launch fused kernel fed by solidfmm rotation blocks.
    Otherwise use the default solidfmm rotate/z-translate/rotate-back reference
    batch. Both paths are numerically equivalent.

    Parameters
    ----------
    src_mult : Array
        Complex multipole coefficients ``[N, (p+1)^2]`` for each pair.
    deltas : Array
        Target-minus-source center displacements ``[N, 3]``.
    order : int
        Expansion order ``p``.
    rotation : str
        Rotation strategy; must be ``"solidfmm"``.

    Returns
    -------
    Array
        Complex local contributions ``[N, (p+1)^2]``.
    """
    if _fused_complex_m2l_pallas_active():
        return _m2l_complex_batch_kernel_fused_pallas(src_mult, deltas, order=order)
    return _m2l_complex_batch_kernel(src_mult, deltas, order=order, rotation=rotation)


def _apply_m2l(
    src_mult: Array,
    deltas: Array,
    *,
    order: int,
    basis_mode: str,
    rotation: Optional[str] = None,
    m2l_impl: Optional[str] = None,
) -> Array:
    """Basis-dispatched batched M2L apply seam.

    ``basis_mode`` is a static discriminator, so XLA specialises each branch to
    the exact HLO of the corresponding single-basis kernel. Real basis routes
    through :func:`_apply_real_m2l` (``m2l_impl``); solidfmm/complex through
    :func:`_apply_complex_m2l` (``rotation``).
    """
    if str(basis_mode).strip().lower() == "real":
        return _apply_real_m2l(src_mult, deltas, order=order, m2l_impl=m2l_impl)
    return _apply_complex_m2l(src_mult, deltas, order=order, rotation=rotation)


@partial(
    jax.jit,
    static_argnames=("order", "basis_mode", "rotation", "m2l_impl", "total_nodes"),
    donate_argnums=(0,),
)
def _accumulate_m2l_fullbatch(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    src: Array,
    tgt: Array,
    active_pair_count: Array,
    *,
    order: int,
    basis_mode: str,
    total_nodes: int,
    rotation: Optional[str] = None,
    m2l_impl: Optional[str] = None,
) -> Array:
    """Accumulate M2L contributions in one full interaction batch (both bases).

    Unifies the former ``_accumulate_{solidfmm,real}_m2l_fullbatch`` behind the
    static ``basis_mode`` seam. Numerics-preserving: every discriminator is a
    ``static_argname`` so XLA specialises the merged jit per basis to the exact
    HLO each single-basis kernel produced.
    """
    idx = jnp.arange(src.shape[0], dtype=INDEX_DTYPE)
    valid = (idx < active_pair_count) & (src >= 0) & (tgt >= 0)
    safe_src = jnp.where(valid, src, 0)
    safe_tgt = jnp.where(valid, tgt, 0)
    src_mult = multip_packed[safe_src]
    deltas = centers[safe_tgt] - centers[safe_src]
    contribs = _apply_m2l(
        src_mult,
        deltas,
        order=order,
        basis_mode=basis_mode,
        rotation=rotation,
        m2l_impl=m2l_impl,
    ).astype(locals_coeffs.dtype)
    contribs = jnp.where(valid[:, None], contribs, 0)
    return locals_coeffs + jax.ops.segment_sum(contribs, safe_tgt, total_nodes)


@partial(
    jax.jit,
    static_argnames=(
        "order",
        "basis_mode",
        "rotation",
        "m2l_impl",
        "total_nodes",
        "chunk_size",
    ),
    donate_argnums=(0,),
)
def _accumulate_m2l_chunked_scan(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    src: Array,
    tgt: Array,
    active_pair_count: Array,
    *,
    order: int,
    basis_mode: str,
    total_nodes: int,
    chunk_size: int,
    rotation: Optional[str] = None,
    m2l_impl: Optional[str] = None,
) -> Array:
    """Accumulate M2L contributions with chunked scan reduction (both bases).

    Unifies the former ``_accumulate_{solidfmm,real}_m2l_chunked_scan`` behind
    the static ``basis_mode`` seam; numerics-preserving (identical HLO per
    basis, single shared ``lax.scan`` body).
    """
    pair_count = src.shape[0]
    starts = jnp.arange(0, pair_count, chunk_size, dtype=INDEX_DTYPE)

    def body(local_accum: Array, start_idx: Array) -> tuple[Array, None]:
        def active_chunk(accum: Array) -> Array:
            offset = jnp.arange(chunk_size, dtype=INDEX_DTYPE)
            idx = start_idx + offset
            valid = idx < pair_count
            safe_idx = jnp.where(valid, idx, 0)
            src_chunk_raw = src[safe_idx]
            tgt_chunk_raw = tgt[safe_idx]
            valid = (
                valid
                & (idx < active_pair_count)
                & (src_chunk_raw >= 0)
                & (tgt_chunk_raw >= 0)
            )
            src_chunk = jnp.where(valid, src_chunk_raw, 0)
            tgt_chunk = jnp.where(valid, tgt_chunk_raw, 0)
            src_mult = multip_packed[src_chunk]
            deltas = centers[tgt_chunk] - centers[src_chunk]
            contribs = _apply_m2l(
                src_mult,
                deltas,
                order=order,
                basis_mode=basis_mode,
                rotation=rotation,
                m2l_impl=m2l_impl,
            ).astype(locals_coeffs.dtype)
            return _chunk_segment_scatter_add(
                accum,
                contribs,
                tgt_chunk,
                valid,
                chunk_size=chunk_size,
            )

        local_accum = jax.lax.cond(
            start_idx < active_pair_count,
            active_chunk,
            lambda accum: accum,
            local_accum,
        )
        return local_accum, None

    local_accum, _ = jax.lax.scan(body, locals_coeffs, starts)
    return local_accum


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
    """Apply solidfmm L2L translations from parents to their children."""
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
    static_argnames=("order", "rotation", "total_nodes", "basis_mode"),
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
    max_level = jnp.max(parent_levels)
    minus_one = jnp.asarray(-1, dtype=left_internal.dtype)

    def level_body(level: Array, state: Array) -> Array:
        active = parent_levels == level
        lc = jnp.where(active, left_internal, minus_one)
        rc = jnp.where(active, right_internal, minus_one)
        child_idx = jnp.concatenate([lc, rc], axis=0)
        valid = child_idx >= 0
        safe_child = jnp.where(valid, child_idx, 0)
        parent_coeffs = state[parent_rep]
        # L2L uses the old_center - new_center (parent - child) displacement in
        # BOTH bases. The complex path previously used child - parent here,
        # which is the wrong sign: the far field was left uncorrected in
        # proportion to the cascade depth, capping accuracy (~3e-3 at
        # theta>=0.5) regardless of expansion order, while looking fine at small
        # theta where the L2L cascade is shallow.
        deltas = centers[parent_rep] - centers[safe_child]
        if real_basis:
            translated = _l2l_real_batch_kernel(
                parent_coeffs, deltas, order=order
            ).astype(state.dtype)
        else:
            translated = _l2l_complex_batch_kernel(
                parent_coeffs, deltas, order=order, rotation=rotation
            ).astype(state.dtype)
        translated = jnp.where(valid[:, None], translated, 0)
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
    """Apply real-basis L2L translations from parents to their children."""
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
            )
            source_motion_locals_updated = _record_timed_array(
                "_refresh_timing_dual_source_motion_seconds",
                stage_t0,
                source_motion_locals_updated,
            )
        else:
            source_motion_locals_updated = None
    else:
        if source_motion_multip_packed is not None:
            stage_t0 = time.perf_counter()
            source_motion_locals_updated = _accumulate_from_multipoles(
                jnp.zeros_like(locals_coeffs), source_motion_multip_packed
            )
            source_motion_locals_updated = _record_timed_array(
                "_refresh_timing_dual_source_motion_seconds",
                stage_t0,
                source_motion_locals_updated,
            )
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


@partial(
    jax.jit,
    static_argnames=(
        "return_potential",
        "max_leaf_size",
        "order",
        "G",
        "softening",
        "expansion_basis",
        "nearfield_mode",
        "nearfield_edge_chunk_size",
        "nearfield_delayed_scatter_chunks_per_superchunk",
        "nearfield_chunk_scan_batch_size",
        "nearfield_chunk_scan_unroll",
        "nearfield_superchunk_scan_unroll",
        "nearfield_sorted_scatter_hint",
        "nearfield_grouped_sorted_scatter",
        "nearfield_superchunk_target_reduce",
        "nearfield_disable_chunk_cond",
        "nearfield_target_leaf_batch_size",
        "nearfield_target_block_tile_size",
        "nearfield_target_block_tile_scan_unroll",
        "nearfield_target_block_batch_scan_unroll",
        "nearfield_target_block_overflow_fast_max_blocks",
        "disable_specialized_large_n_nearfield",
    ),
)
def _evaluate_tree_compiled_impl(
    tree: Tree,
    positions: Array,
    masses: Array,
    locals_data: LocalExpansionData,
    neighbor_list: NodeNeighborList,
    nearfield_leaf_nodes: Array,
    nearfield_node_ranges: Array,
    nearfield_offsets: Array,
    nearfield_neighbors: Array,
    nearfield_counts: Array,
    nearfield_leaf_particle_indices: Array,
    nearfield_leaf_particle_mask: Array,
    leaf_nodes: Array,
    node_ranges: Array,
    precomputed_target_leaf_ids: Array,
    precomputed_source_leaf_ids: Array,
    precomputed_valid_pairs: Array,
    precomputed_chunk_sort_indices: Array,
    precomputed_chunk_group_ids: Array,
    precomputed_chunk_unique_indices: Array,
    precomputed_target_block_offsets: Array,
    precomputed_target_block_leaf_ids: Array,
    precomputed_target_block_source_leaf_ids: Array,
    precomputed_target_block_valid_mask: Array,
    precomputed_target_block_source_leaf_ids_padded: Array,
    precomputed_target_block_valid_mask_padded: Array,
    *,
    G: float,
    softening: float,
    order: int,
    expansion_basis: ExpansionBasis,
    max_leaf_size: int,
    return_potential: bool,
    nearfield_mode: str,
    nearfield_edge_chunk_size: int,
    nearfield_delayed_scatter_chunks_per_superchunk: int = 1,
    nearfield_chunk_scan_batch_size: int = 1,
    nearfield_chunk_scan_unroll: int = 1,
    nearfield_superchunk_scan_unroll: int = 1,
    nearfield_sorted_scatter_hint: bool = False,
    nearfield_grouped_sorted_scatter: bool = False,
    nearfield_superchunk_target_reduce: bool = False,
    nearfield_disable_chunk_cond: bool = True,
    nearfield_target_leaf_batch_size: int = 32,
    nearfield_target_block_tile_size: int = 8,
    nearfield_target_block_tile_scan_unroll: int = 1,
    nearfield_target_block_batch_scan_unroll: int = 1,
    nearfield_target_block_overflow_fast_max_blocks: int = 65536,
    disable_specialized_large_n_nearfield: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """JIT core for far/near field evaluation on a prepared tree state."""
    disable_specialized_large_n = bool(disable_specialized_large_n_nearfield)
    use_precomputed = (
        precomputed_target_leaf_ids.shape[0] == neighbor_list.neighbors.shape[0]
        and precomputed_valid_pairs.shape[0] == neighbor_list.neighbors.shape[0]
    )
    use_precomputed_source = (
        precomputed_source_leaf_ids.shape[0] == neighbor_list.neighbors.shape[0]
    )
    edge_count = int(neighbor_list.neighbors.shape[0])
    chunk_count = (
        (edge_count + int(nearfield_edge_chunk_size) - 1)
        // int(nearfield_edge_chunk_size)
        if edge_count > 0
        else 0
    )
    chunk_flat_size = int(nearfield_edge_chunk_size) * int(max_leaf_size)
    use_precomputed_scatter = (
        precomputed_chunk_sort_indices.shape == (chunk_count, chunk_flat_size)
        and precomputed_chunk_group_ids.shape == (chunk_count, chunk_flat_size)
        and precomputed_chunk_unique_indices.shape == (chunk_count, chunk_flat_size)
    )
    use_specialized_large_n = (
        not disable_specialized_large_n
        and not bool(return_potential)
        and str(nearfield_mode).strip().lower() == "bucketed"
        and nearfield_leaf_particle_indices.shape[0] > 0
        and not use_precomputed_scatter
    )
    use_target_blocks = (
        precomputed_target_block_offsets.shape[0]
        == (neighbor_list.leaf_indices.shape[0] + 1)
        and precomputed_target_block_leaf_ids.shape[0] > 0
        and precomputed_target_block_source_leaf_ids.shape[0]
        == precomputed_target_block_leaf_ids.shape[0]
        and precomputed_target_block_valid_mask.shape
        == precomputed_target_block_source_leaf_ids.shape
    )
    use_target_blocks_padded = (
        precomputed_target_block_source_leaf_ids_padded.shape[0]
        == neighbor_list.leaf_indices.shape[0]
        and precomputed_target_block_source_leaf_ids_padded.shape[1] > 0
        and precomputed_target_block_source_leaf_ids_padded.shape[2] > 0
        and precomputed_target_block_valid_mask_padded.shape
        == precomputed_target_block_source_leaf_ids_padded.shape
    )
    if use_specialized_large_n:
        near = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            positions,
            masses,
            G=G,
            softening=softening,
            edge_chunk_size=nearfield_edge_chunk_size,
            precomputed_target_leaf_ids=(
                precomputed_target_leaf_ids if use_precomputed else None
            ),
            precomputed_source_leaf_ids=(
                precomputed_source_leaf_ids
                if (use_precomputed and use_precomputed_source)
                else None
            ),
            precomputed_valid_pairs=(
                precomputed_valid_pairs if use_precomputed else None
            ),
            leaf_particle_indices=nearfield_leaf_particle_indices,
            leaf_particle_mask=nearfield_leaf_particle_mask,
            precomputed_target_block_leaf_ids=(
                precomputed_target_block_leaf_ids if use_target_blocks else None
            ),
            precomputed_target_block_source_leaf_ids=(
                precomputed_target_block_source_leaf_ids if use_target_blocks else None
            ),
            precomputed_target_block_valid_mask=(
                precomputed_target_block_valid_mask if use_target_blocks else None
            ),
            precomputed_target_block_offsets=(
                precomputed_target_block_offsets if use_target_blocks else None
            ),
            precomputed_target_block_source_leaf_ids_padded=(
                precomputed_target_block_source_leaf_ids_padded
                if use_target_blocks_padded
                else None
            ),
            precomputed_target_block_valid_mask_padded=(
                precomputed_target_block_valid_mask_padded
                if use_target_blocks_padded
                else None
            ),
            delayed_scatter_chunks_per_superchunk=(
                nearfield_delayed_scatter_chunks_per_superchunk
            ),
            chunk_scan_batch_size=nearfield_chunk_scan_batch_size,
            chunk_scan_unroll=nearfield_chunk_scan_unroll,
            superchunk_scan_unroll=nearfield_superchunk_scan_unroll,
            sorted_scatter_hint=nearfield_sorted_scatter_hint,
            grouped_sorted_scatter=nearfield_grouped_sorted_scatter,
            superchunk_target_reduce=nearfield_superchunk_target_reduce,
            disable_chunk_cond=nearfield_disable_chunk_cond,
            target_leaf_batch_size=nearfield_target_leaf_batch_size,
            target_block_tile_size=nearfield_target_block_tile_size,
            target_block_tile_scan_unroll=nearfield_target_block_tile_scan_unroll,
            target_block_batch_scan_unroll=nearfield_target_block_batch_scan_unroll,
            target_block_overflow_fast_max_blocks=(
                nearfield_target_block_overflow_fast_max_blocks
            ),
        )
    else:
        near = compute_leaf_p2p_accelerations(
            tree,
            neighbor_list,
            positions,
            masses,
            G=G,
            softening=softening,
            max_leaf_size=max_leaf_size,
            return_potential=return_potential,
            nearfield_mode=nearfield_mode,
            edge_chunk_size=nearfield_edge_chunk_size,
            precomputed_target_leaf_ids=(
                precomputed_target_leaf_ids if use_precomputed else None
            ),
            precomputed_source_leaf_ids=(
                precomputed_source_leaf_ids
                if (use_precomputed and use_precomputed_source)
                else None
            ),
            precomputed_valid_pairs=(
                precomputed_valid_pairs if use_precomputed else None
            ),
            precomputed_chunk_sort_indices=(
                precomputed_chunk_sort_indices if use_precomputed_scatter else None
            ),
            precomputed_chunk_group_ids=(
                precomputed_chunk_group_ids if use_precomputed_scatter else None
            ),
            precomputed_chunk_unique_indices=(
                precomputed_chunk_unique_indices if use_precomputed_scatter else None
            ),
            node_ranges_override=nearfield_node_ranges,
            leaf_nodes_override=nearfield_leaf_nodes,
            neighbor_offsets_override=nearfield_offsets,
            neighbor_indices_override=nearfield_neighbors,
            neighbor_counts_override=nearfield_counts,
            leaf_particle_indices_override=(
                nearfield_leaf_particle_indices
                if nearfield_leaf_particle_indices.shape[0] > 0
                else None
            ),
            leaf_particle_mask_override=(
                nearfield_leaf_particle_mask
                if nearfield_leaf_particle_mask.shape[0] > 0
                else None
            ),
        )

    far_grad, far_potential_pre, _ = _evaluate_local_expansions_for_particles(
        locals_data,
        positions,
        leaf_nodes=leaf_nodes,
        node_ranges=node_ranges,
        max_leaf_size=max_leaf_size,
        order=order,
        expansion_basis=expansion_basis,
        return_potential=return_potential,
        max_acc_derivative_order=0,
    )

    # far_grad is d/d(delta) of +1/r with delta = center - eval_point.
    # Physical acceleration is d/d(eval_point)(+1/r) * G = -d/d(delta)(+1/r) * G.
    far_acc = -G * far_grad

    if return_potential:
        near_acc, near_pot = near  # type: ignore[misc]
        far_pot = (
            -G * far_potential_pre
            if far_potential_pre is not None
            else jnp.zeros((positions.shape[0],), dtype=positions.dtype)
        )
        accelerations = near_acc + far_acc
        potentials = near_pot + far_pot
        return accelerations, potentials

    accelerations = near + far_acc  # type: ignore[operator]
    return accelerations


def _evaluate_prepared_tree(
    *,
    fmm: "FastMultipoleMethod",
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    downward: TreeDownwardData,
    neighbor_list: NodeNeighborList,
    nearfield_interop: Optional[NearfieldInteropData],
    farfield_local_data: Optional[LocalExpansionData],
    farfield_leaf_nodes: Optional[Array],
    farfield_node_ranges: Optional[Array],
    nearfield_target_leaf_ids: Optional[Array],
    nearfield_source_leaf_ids: Optional[Array],
    nearfield_valid_pairs: Optional[Array],
    nearfield_chunk_sort_indices: Optional[Array],
    nearfield_chunk_group_ids: Optional[Array],
    nearfield_chunk_unique_indices: Optional[Array],
    max_leaf_size: int,
    return_potential: bool,
    jit_traversal: bool,
    max_acc_derivative_order: int = 0,
) -> Union[
    Array,
    Tuple[Array, Array],
    Tuple[Array, PackedAccelerationDerivatives],
    Tuple[Array, Array, PackedAccelerationDerivatives],
]:
    """Run the prepared-tree evaluation returning Morton-sorted outputs."""

    if int(max_acc_derivative_order) > 0:
        nearfield_mode = fmm._resolve_nearfield_mode(
            num_particles=int(positions_sorted.shape[0])
        )
        nearfield_edge_chunk_size = fmm._resolve_nearfield_edge_chunk_size(
            num_particles=int(positions_sorted.shape[0]),
            nearfield_mode=nearfield_mode,
        )
        near = compute_leaf_p2p_accelerations(
            tree,
            neighbor_list,
            positions_sorted,
            masses_sorted,
            G=fmm.G,
            softening=fmm.softening,
            max_leaf_size=max_leaf_size,
            return_potential=return_potential,
            nearfield_mode=nearfield_mode,
            edge_chunk_size=nearfield_edge_chunk_size,
            precomputed_target_leaf_ids=nearfield_target_leaf_ids,
            precomputed_source_leaf_ids=nearfield_source_leaf_ids,
            precomputed_valid_pairs=nearfield_valid_pairs,
            precomputed_chunk_sort_indices=nearfield_chunk_sort_indices,
            precomputed_chunk_group_ids=nearfield_chunk_group_ids,
            precomputed_chunk_unique_indices=nearfield_chunk_unique_indices,
        )
        far_grad, far_potential_pre, far_derivatives = (
            _evaluate_local_expansions_for_particles(
                downward.locals,
                positions_sorted,
                leaf_nodes=jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE),
                node_ranges=jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
                max_leaf_size=max_leaf_size,
                order=int(downward.locals.order),
                expansion_basis=fmm.expansion_basis,
                return_potential=return_potential,
                max_acc_derivative_order=int(max_acc_derivative_order),
            )
        )
        far_acc = -fmm.G * far_grad
        if far_derivatives is None:
            raise RuntimeError("expected far-field acceleration derivatives")
        acc_derivatives = tuple(fmm.G * level for level in far_derivatives)

        if return_potential:
            near_acc, near_pot = near
            far_pot = (
                -fmm.G * far_potential_pre
                if far_potential_pre is not None
                else jnp.zeros(
                    (positions_sorted.shape[0],), dtype=positions_sorted.dtype
                )
            )
            return near_acc + far_acc, near_pot + far_pot, acc_derivatives
        return near + far_acc, acc_derivatives

    if jit_traversal:
        evaluate_fn = fmm.evaluate_tree_compiled
    else:
        evaluate_fn = fmm.evaluate_tree

    return evaluate_fn(
        tree,
        positions_sorted,
        masses_sorted,
        downward,
        neighbor_list,
        nearfield_interop=nearfield_interop,
        farfield_local_data=farfield_local_data,
        farfield_leaf_nodes=farfield_leaf_nodes,
        farfield_node_ranges=farfield_node_ranges,
        precomputed_target_leaf_ids=nearfield_target_leaf_ids,
        precomputed_source_leaf_ids=nearfield_source_leaf_ids,
        precomputed_valid_pairs=nearfield_valid_pairs,
        precomputed_chunk_sort_indices=nearfield_chunk_sort_indices,
        precomputed_chunk_group_ids=nearfield_chunk_group_ids,
        precomputed_chunk_unique_indices=nearfield_chunk_unique_indices,
        max_leaf_size=max_leaf_size,
        return_potential=return_potential,
    )


def _map_targets_to_leaf_positions(
    *,
    target_sorted_indices: Array,
    leaf_nodes: Array,
    node_ranges: Array,
) -> Array:
    """Map sorted particle indices to positions in the leaf-node array."""
    if int(target_sorted_indices.shape[0]) == 0:
        return jnp.zeros((0,), dtype=INDEX_DTYPE)
    leaf_ranges = node_ranges[leaf_nodes]
    starts = leaf_ranges[:, 0]
    ends = leaf_ranges[:, 1]
    leaf_pos = jnp.searchsorted(starts, target_sorted_indices, side="right") - 1
    leaf_pos = jnp.clip(leaf_pos, 0, leaf_nodes.shape[0] - 1)
    valid = (target_sorted_indices >= starts[leaf_pos]) & (
        target_sorted_indices <= ends[leaf_pos]
    )
    if not bool(jnp.all(valid)):
        raise ValueError("target_indices could not be mapped to prepared tree leaves")
    return leaf_pos.astype(INDEX_DTYPE)


def _build_target_nearfield_source_index_matrix(
    *,
    target_sorted_indices: Array,
    target_leaf_positions: Array,
    nearfield_interop: NearfieldInteropData,
) -> tuple[Array, Array]:
    """Build padded source-index lists for each target particle near-field eval."""
    targets = jnp.asarray(target_sorted_indices, dtype=INDEX_DTYPE)
    target_leaf_pos = jnp.asarray(target_leaf_positions, dtype=INDEX_DTYPE)
    node_ranges = jnp.asarray(nearfield_interop.node_ranges, dtype=INDEX_DTYPE)
    leaf_nodes = jnp.asarray(nearfield_interop.leaf_nodes, dtype=INDEX_DTYPE)
    offsets = jnp.asarray(nearfield_interop.offsets, dtype=INDEX_DTYPE)
    neighbors = jnp.asarray(nearfield_interop.neighbors, dtype=INDEX_DTYPE)

    num_targets = int(targets.shape[0])
    if num_targets == 0:
        empty_idx = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        empty_mask = jnp.zeros((0, 0), dtype=bool)
        return empty_idx, empty_mask

    num_leaves = int(leaf_nodes.shape[0])
    if num_leaves == 0:
        empty_idx = jnp.zeros((num_targets, 0), dtype=INDEX_DTYPE)
        empty_mask = jnp.zeros((num_targets, 0), dtype=bool)
        return empty_idx, empty_mask

    if (
        nearfield_interop.leaf_particle_indices is not None
        and nearfield_interop.leaf_particle_mask is not None
    ):
        leaf_particle_idx = jnp.asarray(
            nearfield_interop.leaf_particle_indices,
            dtype=INDEX_DTYPE,
        )
        leaf_particle_mask = jnp.asarray(
            nearfield_interop.leaf_particle_mask,
            dtype=bool,
        )
        max_leaf_particles = int(leaf_particle_idx.shape[1])
    else:
        leaf_ranges = node_ranges[leaf_nodes]
        leaf_counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
        max_leaf_particles = int(jnp.max(leaf_counts))
        if max_leaf_particles <= 0:
            empty_idx = jnp.zeros((num_targets, 0), dtype=INDEX_DTYPE)
            empty_mask = jnp.zeros((num_targets, 0), dtype=bool)
            return empty_idx, empty_mask

        particle_offsets = jnp.arange(max_leaf_particles, dtype=INDEX_DTYPE)
        leaf_particle_idx = leaf_ranges[:, 0][:, None] + particle_offsets[None, :]
        leaf_particle_mask = particle_offsets[None, :] < leaf_counts[:, None]

    if max_leaf_particles <= 0:
        empty_idx = jnp.zeros((num_targets, 0), dtype=INDEX_DTYPE)
        empty_mask = jnp.zeros((num_targets, 0), dtype=bool)
        return empty_idx, empty_mask

    if nearfield_interop.neighbor_leaf_positions is not None:
        nbr_leaf_pos = jnp.asarray(
            nearfield_interop.neighbor_leaf_positions,
            dtype=INDEX_DTYPE,
        )
    else:
        total_nodes = int(node_ranges.shape[0])
        leaf_lookup = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
        leaf_lookup = leaf_lookup.at[leaf_nodes].set(
            jnp.arange(num_leaves, dtype=INDEX_DTYPE)
        )
        nbr_counts = offsets[1:] - offsets[:-1]
        max_nbr = int(jnp.max(nbr_counts))
        if max_nbr > 0:
            nbr_offsets = jnp.arange(max_nbr, dtype=INDEX_DTYPE)
            nbr_idx = offsets[:-1, None] + nbr_offsets[None, :]
            nbr_valid = nbr_offsets[None, :] < nbr_counts[:, None]
            nbr_safe_idx = jnp.where(nbr_valid, nbr_idx, 0)
            nbr_nodes = neighbors[nbr_safe_idx]
            nbr_leaf_pos = leaf_lookup[nbr_nodes]
            nbr_leaf_pos = jnp.where(nbr_valid, nbr_leaf_pos, -1)
        else:
            nbr_leaf_pos = jnp.zeros((num_leaves, 0), dtype=INDEX_DTYPE)

    self_leaf = jnp.arange(num_leaves, dtype=INDEX_DTYPE)[:, None]
    source_leaf_positions = jnp.concatenate([self_leaf, nbr_leaf_pos], axis=1)
    source_leaf_valid = source_leaf_positions >= 0
    source_leaf_safe = jnp.where(source_leaf_valid, source_leaf_positions, 0)

    source_particle_idx_by_leaf = leaf_particle_idx[source_leaf_safe]
    source_particle_mask_by_leaf = (
        leaf_particle_mask[source_leaf_safe] & source_leaf_valid[..., None]
    )

    target_source_idx = source_particle_idx_by_leaf[target_leaf_pos]
    target_source_mask = source_particle_mask_by_leaf[target_leaf_pos]
    target_source_idx = target_source_idx.reshape((num_targets, -1))
    target_source_mask = target_source_mask.reshape((num_targets, -1))
    target_source_mask = target_source_mask & (target_source_idx != targets[:, None])

    sentinel = jnp.asarray(jnp.iinfo(INDEX_DTYPE).max, dtype=INDEX_DTYPE)
    sortable = jnp.where(target_source_mask, target_source_idx, sentinel)
    sorted_idx = jnp.sort(sortable, axis=1)
    non_sentinel = sorted_idx < sentinel
    first = jnp.ones((num_targets, 1), dtype=bool)
    changed = jnp.concatenate([first, sorted_idx[:, 1:] != sorted_idx[:, :-1]], axis=1)
    unique_mask = non_sentinel & changed
    padded = jnp.where(unique_mask, sorted_idx, 0)
    return padded, unique_mask


def _compute_targeted_nearfield(
    *,
    positions_sorted: Array,
    masses_sorted: Array,
    target_sorted_indices: Array,
    source_indices: Array,
    source_mask: Array,
    G: Union[float, Array],
    softening: float,
    return_potential: bool,
    velocities_sorted: Optional[Array] = None,
    return_jerk: bool = False,
    return_snap: bool = False,
    return_crackle: bool = False,
) -> tuple[Array, Optional[Array], Optional[Array], Optional[Array], Optional[Array]]:
    """Compute near-field contributions for target particles only."""
    if return_jerk and velocities_sorted is None:
        raise ValueError("velocities_sorted must be provided when return_jerk=True")
    if return_snap and velocities_sorted is None:
        raise ValueError("velocities_sorted must be provided when return_snap=True")
    if return_crackle and velocities_sorted is None:
        raise ValueError("velocities_sorted must be provided when return_crackle=True")
    target_positions = positions_sorted[target_sorted_indices]
    dtype = positions_sorted.dtype
    g_const = jnp.asarray(G, dtype=dtype)
    softening_sq = jnp.asarray(float(softening) ** 2, dtype=dtype)
    target_velocities = (
        velocities_sorted[target_sorted_indices]
        if velocities_sorted is not None
        else None
    )
    if int(source_indices.shape[1]) == 0:
        zeros = jnp.zeros((target_positions.shape[0], 3), dtype=positions_sorted.dtype)
        jerk_zeros = (
            jnp.zeros((target_positions.shape[0], 3), dtype=positions_sorted.dtype)
            if return_jerk
            else None
        )
        snap_zeros = (
            jnp.zeros((target_positions.shape[0], 3), dtype=positions_sorted.dtype)
            if return_snap
            else None
        )
        crackle_zeros = (
            jnp.zeros((target_positions.shape[0], 3), dtype=positions_sorted.dtype)
            if return_crackle
            else None
        )
        if return_potential:
            return (
                zeros,
                jnp.zeros((target_positions.shape[0],), dtype=zeros.dtype),
                jerk_zeros,
                snap_zeros,
                crackle_zeros,
            )
        return zeros, None, jerk_zeros, snap_zeros, crackle_zeros
    src_pos = positions_sorted[source_indices]
    src_mass = masses_sorted[source_indices]
    diff = target_positions[:, None, :] - src_pos
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
    eps = jnp.finfo(positions_sorted.dtype).eps
    one = jnp.asarray(1.0, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    three = jnp.asarray(3.0, dtype=dtype)
    six = jnp.asarray(6.0, dtype=dtype)
    nine = jnp.asarray(9.0, dtype=dtype)
    fifteen = jnp.asarray(15.0, dtype=dtype)
    forty_five = jnp.asarray(45.0, dtype=dtype)
    one_oh_five = jnp.asarray(105.0, dtype=dtype)

    inv_r = jnp.where(source_mask, one / (jnp.sqrt(dist_sq) + eps), zero)
    inv_dist3 = jnp.where(source_mask, inv_r / dist_sq, zero)
    weighted = inv_dist3 * src_mass
    near_acc = -g_const * jnp.sum(weighted[..., None] * diff, axis=1)
    near_jerk: Optional[Array]
    near_snap: Optional[Array]
    near_crackle: Optional[Array]
    if return_jerk:
        src_vel = velocities_sorted[source_indices]  # type: ignore[index]
        vel_diff = target_velocities[:, None, :] - src_vel  # type: ignore[index]
        inv_dist5 = jnp.where(source_mask, inv_dist3 / dist_sq, zero)
        rv = jnp.sum(diff * vel_diff, axis=-1)
        jerk_term = vel_diff * inv_dist3[..., None] - (
            three * rv[..., None] * diff * inv_dist5[..., None]
        )
        near_jerk = -g_const * jnp.sum(src_mass[..., None] * jerk_term, axis=1)
        if return_snap:
            inv_dist7 = jnp.where(source_mask, inv_dist5 / dist_sq, zero)
            vv = jnp.sum(vel_diff * vel_diff, axis=-1)
            snap_term = (
                six * rv[..., None] * vel_diff * inv_dist5[..., None]
                + three * vv[..., None] * diff * inv_dist5[..., None]
                - fifteen * (rv * rv)[..., None] * diff * inv_dist7[..., None]
            )
            near_snap = jnp.sum(src_mass[..., None] * snap_term, axis=1) * g_const
            if return_crackle:
                inv_dist9 = jnp.where(source_mask, inv_dist7 / dist_sq, zero)
                crackle_term = (
                    nine * vv[..., None] * vel_diff * inv_dist5[..., None]
                    - forty_five
                    * (rv * rv)[..., None]
                    * vel_diff
                    * inv_dist7[..., None]
                    - forty_five
                    * rv[..., None]
                    * vv[..., None]
                    * diff
                    * inv_dist7[..., None]
                    + one_oh_five
                    * (rv * rv * rv)[..., None]
                    * diff
                    * inv_dist9[..., None]
                )
                near_crackle = jnp.sum(src_mass[..., None] * crackle_term, axis=1) * (
                    g_const
                )
            else:
                near_crackle = None
        else:
            near_snap = None
            near_crackle = None
    else:
        near_jerk = None
        near_snap = None
        near_crackle = None
    if not return_potential:
        return near_acc, None, near_jerk, near_snap, near_crackle
    near_pot = -g_const * jnp.sum(inv_r * src_mass, axis=1)
    return near_acc, near_pot, near_jerk, near_snap, near_crackle


def _evaluate_local_expansions_for_target_particles(
    *,
    local_data: LocalExpansionData,
    positions_sorted: Array,
    target_sorted_indices: Array,
    target_leaf_positions: Array,
    leaf_nodes: Array,
    order: int,
    expansion_basis: ExpansionBasis,
    return_potential: bool,
    max_acc_derivative_order: int = 0,
) -> tuple[Array, Optional[Array], Optional[PackedAccelerationDerivatives]]:
    """Evaluate far-field local expansions for target particles only."""
    if order > MAX_MULTIPOLE_ORDER and expansion_basis != "solidfmm":
        raise NotImplementedError(
            "orders above 4 require expansion_basis='solidfmm'",
        )
    if int(target_sorted_indices.shape[0]) == 0:
        zeros = jnp.zeros((0, 3), dtype=positions_sorted.dtype)
        derivatives: Optional[PackedAccelerationDerivatives]
        if max_acc_derivative_order > 0:
            derivatives = tuple(
                jnp.zeros(
                    (
                        0,
                        3,
                        len(component_lift_index_map_3d(level)),
                    ),
                    dtype=positions_sorted.dtype,
                )
                for level in range(1, max_acc_derivative_order + 1)
            )
        else:
            derivatives = None
        if return_potential:
            return zeros, jnp.zeros((0,), dtype=positions_sorted.dtype), derivatives
        return zeros, None, derivatives

    target_leaf_nodes = leaf_nodes[target_leaf_positions]
    centers = local_data.centers[target_leaf_nodes]
    coeffs = local_data.coefficients[target_leaf_nodes]
    target_positions = positions_sorted[target_sorted_indices]

    if expansion_basis == "solidfmm":
        offsets_solid = centers - target_positions
        offsets_complex = offsets_solid

        # Real (Dehnen no-sqrt2) basis: real-typed locals, evaluated with the
        # real L2P operator (detected by coefficient dtype).
        if not jnp.iscomplexobj(coeffs):
            if int(max_acc_derivative_order) <= 0:
                grads, pots = jax.vmap(
                    lambda coeff_row, offset_row: evaluate_local_real_with_grad(
                        coeff_row, offset_row, order=int(order)
                    )
                )(coeffs, offsets_complex)
                if return_potential:
                    return grads, pots, None
                return grads, None, None

            tower = jax.vmap(
                lambda coeff_row, offset_row: (
                    evaluate_local_real_derivative_tower_batch(
                        coeff_row,
                        offset_row[jnp.newaxis, :],
                        order=int(order),
                        max_derivative_order=int(max_acc_derivative_order) + 1,
                    )
                ),
                in_axes=(0, 0),
            )(coeffs, offsets_complex)
            potentials = tower[0][:, 0, 0]
            gradients = tower[1][:, 0, :]
            derivatives_real: list[Array] = []
            for level in range(1, max_acc_derivative_order + 1):
                high = tower[level + 1][:, 0, :]
                gather = jnp.asarray(
                    component_lift_index_map_3d(level),
                    dtype=INDEX_DTYPE,
                )
                lifted = jnp.swapaxes(high[:, gather], 1, 2)
                sign = -1.0 if level % 2 == 0 else 1.0
                derivatives_real.append(sign * lifted)
            packed_real: PackedAccelerationDerivatives = tuple(derivatives_real)
            if return_potential:
                return gradients, potentials, packed_real
            return gradients, None, packed_real

        if max_acc_derivative_order <= 0:
            if return_potential:

                def eval_one(
                    coeff_row: Array, offset_row: Array
                ) -> tuple[Array, Array]:
                    grad, pot = evaluate_local_complex_with_grad_analytic(
                        coeff_row,
                        offset_row,
                        order=int(order),
                    )
                    return grad, pot

                gradients, potentials = jax.vmap(eval_one)(coeffs, offsets_complex)
                return gradients, potentials, None

            gradients = jax.vmap(
                lambda coeff_row, offset_row: evaluate_local_complex_grad_analytic(
                    coeff_row,
                    offset_row,
                    order=int(order),
                )
            )(coeffs, offsets_complex)
            return gradients, None, None

        tower = jax.vmap(
            lambda coeff_row, offset_row: evaluate_local_complex_derivative_tower_batch(
                coeff_row,
                offset_row[jnp.newaxis, :],
                order=int(order),
                max_derivative_order=int(max_acc_derivative_order) + 1,
            ),
            in_axes=(0, 0),
        )(coeffs, offsets_complex)

        potentials = tower[0][:, 0, 0]
        gradients = tower[1][:, 0, :]
        derivatives: list[Array] = []
        for level in range(1, max_acc_derivative_order + 1):
            high = tower[level + 1][:, 0, :]
            gather = jnp.asarray(
                component_lift_index_map_3d(level),
                dtype=INDEX_DTYPE,
            )
            # (targets, components(level), xyz) -> (targets, xyz, components(level))
            lifted = jnp.swapaxes(high[:, gather], 1, 2)
            sign = -1.0 if level % 2 == 0 else 1.0
            derivatives.append(sign * lifted)
        packed_derivatives: PackedAccelerationDerivatives = tuple(derivatives)
        if return_potential:
            return gradients, potentials, packed_derivatives
        return gradients, None, packed_derivatives

    offsets = target_positions - centers

    gradients, potentials = _evaluate_local_cartesian_with_grad_batch(
        coeffs,
        offsets,
        order=order,
    )
    if return_potential:
        return gradients, potentials, None
    return gradients, None, None


def _evaluate_prepared_tree_targets(
    *,
    fmm: "FastMultipoleMethod",
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    downward: TreeDownwardData,
    neighbor_list: NodeNeighborList,
    nearfield_interop: Optional[NearfieldInteropData],
    farfield_local_data: Optional[LocalExpansionData],
    farfield_leaf_nodes: Optional[Array],
    farfield_node_ranges: Optional[Array],
    target_sorted_indices: Array,
    return_potential: bool,
    max_acc_derivative_order: int = 0,
) -> Union[
    Array,
    Tuple[Array, Array],
    Tuple[Array, PackedAccelerationDerivatives],
    Tuple[Array, Array, PackedAccelerationDerivatives],
]:
    """Run prepared-tree evaluation for target particles only."""
    g_const = jnp.asarray(fmm.G, dtype=positions_sorted.dtype)
    nearfield_view = (
        _build_nearfield_interop_data(tree, neighbor_list)
        if nearfield_interop is None
        else nearfield_interop
    )
    node_views = _resolve_evaluation_node_views(
        tree,
        neighbor_list,
        farfield_leaf_nodes=farfield_leaf_nodes,
        farfield_node_ranges=farfield_node_ranges,
    )
    if nearfield_view.particle_to_leaf_position is not None:
        target_leaf_positions = jnp.asarray(
            nearfield_view.particle_to_leaf_position,
            dtype=INDEX_DTYPE,
        )[target_sorted_indices]
    else:
        target_leaf_positions = _map_targets_to_leaf_positions(
            target_sorted_indices=target_sorted_indices,
            leaf_nodes=nearfield_view.particle_order_leaf_indices,
            node_ranges=nearfield_view.particle_order_node_ranges,
        )
        target_leaf_positions = nearfield_view.particle_order_to_native_leaf[
            target_leaf_positions
        ]
    near_source_idx, near_source_mask = _build_target_nearfield_source_index_matrix(
        target_sorted_indices=target_sorted_indices,
        target_leaf_positions=target_leaf_positions,
        nearfield_interop=nearfield_view,
    )
    near_acc, near_pot, _, _, _ = _compute_targeted_nearfield(
        positions_sorted=positions_sorted,
        masses_sorted=masses_sorted,
        target_sorted_indices=target_sorted_indices,
        source_indices=near_source_idx,
        source_mask=near_source_mask,
        G=g_const,
        softening=float(fmm.softening),
        return_potential=return_potential,
    )
    far_grad, far_potential_pre, far_derivatives = (
        _evaluate_local_expansions_for_target_particles(
            local_data=downward.locals,
            positions_sorted=positions_sorted,
            target_sorted_indices=target_sorted_indices,
            target_leaf_positions=target_leaf_positions,
            leaf_nodes=node_views.farfield_leaf_nodes,
            order=int(downward.locals.order),
            expansion_basis=fmm.expansion_basis,
            return_potential=return_potential,
            max_acc_derivative_order=max_acc_derivative_order,
        )
    )
    far_acc = -g_const * far_grad
    acc_derivatives: Optional[PackedAccelerationDerivatives]
    if far_derivatives is not None:
        acc_derivatives = tuple(g_const * level for level in far_derivatives)
    else:
        acc_derivatives = None
    if return_potential:
        far_pot = (
            -g_const * far_potential_pre
            if far_potential_pre is not None
            else jnp.zeros(
                (target_sorted_indices.shape[0],), dtype=positions_sorted.dtype
            )
        )
        near_pot_resolved = (
            near_pot
            if near_pot is not None
            else jnp.zeros(
                (target_sorted_indices.shape[0],), dtype=positions_sorted.dtype
            )
        )
        if acc_derivatives is None:
            return near_acc + far_acc, near_pot_resolved + far_pot
        return near_acc + far_acc, near_pot_resolved + far_pot, acc_derivatives
    if acc_derivatives is None:
        return near_acc + far_acc
    return near_acc + far_acc, acc_derivatives


@partial(
    jax.jit,
    static_argnames=(
        "max_leaf_size",
        "return_potential",
        "order",
        "expansion_basis",
        "max_acc_derivative_order",
    ),
)
def _evaluate_local_expansions_for_particles(
    local_data: LocalExpansionData,
    positions: Array,
    *,
    leaf_nodes: Array,
    node_ranges: Array,
    max_leaf_size: int,
    order: int,
    expansion_basis: ExpansionBasis,
    return_potential: bool,
    max_acc_derivative_order: int = 0,
) -> tuple[Array, Optional[Array], Optional[PackedAccelerationDerivatives]]:
    """Evaluate node-local expansions at leaf particles and scatter results."""
    if order > MAX_MULTIPOLE_ORDER and expansion_basis != "solidfmm":
        raise NotImplementedError(
            "orders above 4 require expansion_basis='solidfmm'",
        )

    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1

    idx = jnp.arange(max_leaf_size, dtype=INDEX_DTYPE)
    starts = leaf_ranges[:, 0][:, None]
    particle_idx = starts + idx
    valid = idx[None, :] < counts[:, None]

    safe_idx = jnp.clip(
        particle_idx,
        min=0,
        max=positions.shape[0] - 1,
    )
    leaf_positions = positions[safe_idx]
    leaf_positions = jnp.where(valid[..., None], leaf_positions, 0.0)

    centers = local_data.centers[leaf_nodes]
    offsets = leaf_positions - centers[:, None, :]
    offsets = jnp.where(valid[..., None], offsets, 0.0)

    coeffs = local_data.coefficients[leaf_nodes]
    dtype = positions.dtype

    if expansion_basis == "solidfmm":
        p = int(order)

        # Complex solidfmm expects delta = center - eval_point (same as real)
        offsets_complex = centers[:, None, :] - leaf_positions
        offsets_complex = jnp.where(valid[..., None], offsets_complex, 0.0)

        # Real (Dehnen no-sqrt2) basis: locals are real-typed, evaluated with the
        # real L2P operator. Detected by coefficient dtype so no basis_mode needs
        # to be threaded through every caller.
        if not jnp.iscomplexobj(coeffs):
            # Real (Dehnen) branch: compute grad_field / potentials /
            # derivative_fields with the real L2P operators, then fall through
            # to the shared scatter below (identical to the complex path).
            if int(max_acc_derivative_order) <= 0:

                def evaluate_leaf_real(
                    coeffs_leaf: Array,
                    offsets_leaf: Array,
                    mask_leaf: Array,
                ) -> tuple[Array, Array]:
                    grads, values = jax.vmap(
                        lambda offset: evaluate_local_real_with_grad(
                            coeffs_leaf, offset, order=p
                        )
                    )(offsets_leaf)
                    # evaluate_local_real_with_grad returns d(phi)/d(delta) with
                    # delta = center - eval_point == the acceleration
                    # contribution consumed downstream.
                    grads = grads.astype(dtype)
                    values = values.astype(dtype)
                    grads = jnp.where(mask_leaf[..., None], grads, 0.0)
                    values = jnp.where(mask_leaf, values, 0.0)
                    return grads, values

                grad_field, potentials = jax.vmap(evaluate_leaf_real)(
                    coeffs,
                    offsets_complex,
                    valid,
                )
                derivative_fields = []
            else:

                def evaluate_leaf_real_with_derivatives(
                    coeffs_leaf: Array,
                    offsets_leaf: Array,
                    mask_leaf: Array,
                ) -> tuple[Array, Array, tuple[Array, ...]]:
                    tower = evaluate_local_real_derivative_tower_batch(
                        coeffs_leaf,
                        offsets_leaf,
                        order=p,
                        max_derivative_order=int(max_acc_derivative_order) + 1,
                    )
                    grads = tower[1].astype(dtype)
                    values = tower[0][:, 0].astype(dtype)
                    grads = jnp.where(mask_leaf[..., None], grads, 0.0)
                    values = jnp.where(mask_leaf, values, 0.0)
                    derivative_levels: list[Array] = []
                    for level in range(1, int(max_acc_derivative_order) + 1):
                        high = tower[level + 1]
                        gather = jnp.asarray(
                            component_lift_index_map_3d(level),
                            dtype=INDEX_DTYPE,
                        )
                        lifted = jnp.swapaxes(high[:, gather], 1, 2)
                        sign = -1.0 if level % 2 == 0 else 1.0
                        lifted = (sign * lifted).astype(dtype)
                        lifted = jnp.where(mask_leaf[:, None, None], lifted, 0.0)
                        derivative_levels.append(lifted)
                    return grads, values, tuple(derivative_levels)

                grad_field, potentials, derivative_fields_tuple = jax.vmap(
                    evaluate_leaf_real_with_derivatives
                )(
                    coeffs,
                    offsets_complex,
                    valid,
                )
                derivative_fields = list(derivative_fields_tuple)
        elif max_acc_derivative_order <= 0:
            if not bool(return_potential):
                flat_analytic = str(
                    os.environ.get(
                        "JACCPOT_LOCAL_EVAL_FLAT_ANALYTIC",
                        "0",
                    )
                ).strip().lower() in {"1", "true", "yes", "on"}

                dtype_preserve_analytic = str(
                    os.environ.get(
                        "JACCPOT_LOCAL_EVAL_DTYPE_PRESERVE",
                        "0",
                    )
                ).strip().lower() in {"1", "true", "yes", "on"}
                order4_unrolled_analytic = str(
                    os.environ.get(
                        "JACCPOT_LOCAL_EVAL_ORDER4_UNROLLED",
                        "0",
                    )
                ).strip().lower() in {"1", "true", "yes", "on"}
                eval_complex_grad = (
                    evaluate_local_complex_grad_order4_unrolled
                    if bool(order4_unrolled_analytic) and p == 4
                    else (
                        evaluate_local_complex_grad_analytic_preserve_dtype
                        if bool(dtype_preserve_analytic)
                        else evaluate_local_complex_grad_analytic
                    )
                )

                if bool(flat_analytic):
                    coeffs_flat = jnp.broadcast_to(
                        coeffs[:, None, :],
                        offsets_complex.shape[:-1] + (coeffs.shape[-1],),
                    ).reshape((-1, coeffs.shape[-1]))
                    offsets_flat = offsets_complex.reshape(
                        (-1, offsets_complex.shape[-1])
                    )
                    mask_flat = valid.reshape((-1,))
                    grad_flat = jax.vmap(
                        lambda coeff_row, offset_row: eval_complex_grad(
                            coeff_row,
                            offset_row,
                            order=p,
                        )
                    )(coeffs_flat, offsets_flat)
                    grad_flat = grad_flat.astype(dtype)
                    grad_flat = jnp.where(mask_flat[:, None], grad_flat, 0.0)
                    grad_field = grad_flat.reshape(valid.shape + (3,))
                else:

                    def evaluate_leaf_complex_grad_only(
                        coeffs_leaf: Array,
                        offsets_leaf: Array,
                        mask_leaf: Array,
                    ) -> Array:
                        grads = jax.vmap(
                            lambda offset: eval_complex_grad(
                                coeffs_leaf,
                                offset,
                                order=p,
                            )
                        )(offsets_leaf)
                        grads = grads.astype(dtype)
                        return jnp.where(mask_leaf[..., None], grads, 0.0)

                    grad_field = jax.vmap(evaluate_leaf_complex_grad_only)(
                        coeffs,
                        offsets_complex,
                        valid,
                    )
                potentials = None
            else:

                def evaluate_leaf_complex(
                    coeffs_leaf: Array,
                    offsets_leaf: Array,
                    mask_leaf: Array,
                ) -> tuple[Array, Array]:
                    grads, values = evaluate_local_complex_with_grad_analytic_batch(
                        coeffs_leaf,
                        offsets_leaf,
                        order=p,
                    )
                    grads = grads.astype(dtype)
                    values = values.astype(dtype)
                    grads = jnp.where(mask_leaf[..., None], grads, 0.0)
                    values = jnp.where(mask_leaf, values, 0.0)
                    return grads, values

                grad_field, potentials = jax.vmap(evaluate_leaf_complex)(
                    coeffs,
                    offsets_complex,
                    valid,
                )
            derivative_fields: list[Array] = []
        else:

            def evaluate_leaf_complex_with_derivatives(
                coeffs_leaf: Array,
                offsets_leaf: Array,
                mask_leaf: Array,
            ) -> tuple[Array, Array, tuple[Array, ...]]:
                tower = evaluate_local_complex_derivative_tower_batch(
                    coeffs_leaf,
                    offsets_leaf,
                    order=p,
                    max_derivative_order=int(max_acc_derivative_order) + 1,
                )
                grads = tower[1].astype(dtype)
                values = tower[0][:, 0].astype(dtype)
                grads = jnp.where(mask_leaf[..., None], grads, 0.0)
                values = jnp.where(mask_leaf, values, 0.0)
                derivative_levels: list[Array] = []
                for level in range(1, int(max_acc_derivative_order) + 1):
                    high = tower[level + 1]
                    gather = jnp.asarray(
                        component_lift_index_map_3d(level),
                        dtype=INDEX_DTYPE,
                    )
                    lifted = jnp.swapaxes(high[:, gather], 1, 2)
                    sign = -1.0 if level % 2 == 0 else 1.0
                    lifted = (sign * lifted).astype(dtype)
                    lifted = jnp.where(mask_leaf[:, None, None], lifted, 0.0)
                    derivative_levels.append(lifted)
                return grads, values, tuple(derivative_levels)

            grad_field, potentials, derivative_fields_tuple = jax.vmap(
                evaluate_leaf_complex_with_derivatives
            )(
                coeffs,
                offsets_complex,
                valid,
            )
            derivative_fields = list(derivative_fields_tuple)

        direct_leaf_flatten = str(
            os.environ.get(
                "JACCPOT_LOCAL_EVAL_DIRECT_LEAF_FLATTEN",
                "0",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        if bool(direct_leaf_flatten) and max_acc_derivative_order <= 0:
            gradients = grad_field.reshape((-1, grad_field.shape[-1]))[
                : positions.shape[0]
            ]
        else:
            gradients = _scatter_vectors(
                jnp.zeros_like(positions),
                safe_idx,
                grad_field,
                valid,
            )

        derivative_outputs: Optional[PackedAccelerationDerivatives]
        if max_acc_derivative_order > 0:
            derivative_outputs = []
            for level, deriv_field in enumerate(derivative_fields, start=1):
                scattered = _scatter_rank3(
                    jnp.zeros(
                        (
                            positions.shape[0],
                            3,
                            len(component_lift_index_map_3d(level)),
                        ),
                        dtype=positions.dtype,
                    ),
                    safe_idx,
                    deriv_field,
                    valid,
                )
                derivative_outputs.append(scattered)
            derivative_outputs = tuple(derivative_outputs)
        else:
            derivative_outputs = None

        if not return_potential:
            return gradients, None, derivative_outputs

        potentials_flat = _scatter_scalars(
            jnp.zeros((positions.shape[0],), dtype=dtype),
            safe_idx,
            potentials,
            valid,
        )
        return gradients, potentials_flat, derivative_outputs

    coeffs_broadcast = jnp.broadcast_to(
        coeffs[:, None, :],
        offsets.shape[:-1] + (coeffs.shape[-1],),
    )
    grad_field, potentials = _evaluate_local_cartesian_with_grad_batch(
        coeffs_broadcast,
        offsets,
        order=order,
    )
    grad_field = jnp.where(valid[..., None], grad_field, 0.0)
    potentials = jnp.where(valid, potentials, 0.0)

    gradients = _scatter_vectors(
        jnp.zeros_like(positions),
        safe_idx,
        grad_field,
        valid,
    )

    if not return_potential:
        return gradients, None, None

    potentials_flat = _scatter_scalars(
        jnp.zeros((positions.shape[0],), dtype=dtype),
        safe_idx,
        potentials,
        valid,
    )
    return gradients, potentials_flat, None


def _scatter_vectors(
    base: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Scatter-add vector values into a flat particle buffer with masking."""
    if values.size == 0:
        return base
    flat_idx = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)
    zero = jnp.zeros((), dtype=base.dtype)
    masked = jnp.where(flat_mask[:, None], flat_values, zero)
    return base.at[flat_idx].add(masked)


def _scatter_scalars(
    base: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Scatter-add scalar values into a flat particle buffer with masking."""
    if values is None or values.size == 0:
        return base
    flat_idx = indices.reshape(-1)
    flat_values = values.reshape(-1)
    flat_mask = mask.reshape(-1)
    zero = jnp.zeros((), dtype=base.dtype)
    masked = jnp.where(flat_mask, flat_values, zero)
    return base.at[flat_idx].add(masked)


def _scatter_rank3(
    base: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Scatter-add rank-3 values into a particle-major buffer."""
    if values.size == 0:
        return base
    flat_idx = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-2], values.shape[-1])
    flat_mask = mask.reshape(-1)
    zero = jnp.zeros((), dtype=base.dtype)
    masked = jnp.where(flat_mask[:, None, None], flat_values, zero)
    return base.at[flat_idx].add(masked)
