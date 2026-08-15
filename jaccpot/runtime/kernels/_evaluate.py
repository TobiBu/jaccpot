"""Evaluation: L2P, the targeted near field, and the tree-evaluation entry points.

The last family in ARCHITECTURE §5, and the one that produces the numbers a caller
sees. Three layers:

* **setup** -- inferring bounds, leaf width and expansion order from a prepared
  state, resolving the node views, and building the ``NearfieldInteropData``
  hand-off;
* **L2P** -- ``_evaluate_local_expansions_for_particles`` and its targeted twin,
  which contract local coefficients at particle positions in whichever basis the
  state carries, optionally with the derivative tower;
* **the targeted near field** -- ``_compute_targeted_nearfield`` and the source
  index matrix it needs when only a subset of particles is evaluated.

The scatter helpers at the end are the adjoint of the per-leaf gather; they are
here rather than in ``nearfield/`` because they scatter *evaluation* results.

Split out of ``core.py`` (Tier 1.6, A.9 seam 4); every function body is unchanged.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Tuple
from jaxtyping import Array
from yggdrax.interactions import (
    NodeNeighborList,
    OctreeNativeNeighborList,
)
from yggdrax.tree import Tree

from jaccpot.downward.local_expansions import (
    LocalExpansionData,
    TreeDownwardData,
    translate_local_expansion,
)
from jaccpot.nearfield.near_field import (
    compute_leaf_p2p_accelerations,
    compute_leaf_p2p_accelerations_large_n_accel_only,
)

# `evaluate_local_complex_with_grad_analytic` below is the single-row `with_grad`
# variant used by the complex `return_potential` branch of
# `_evaluate_local_expansions_for_target_particles`. It was CALLED there but never
# imported -- audit G.1b, a latent `NameError`. The name exists in `complex_ops` with
# exactly the signature the call site uses, so the import is the whole fix.
# (Comment kept OUTSIDE the parentheses: isort rewrites comments placed inside an
# import block onto the `from ... import (` line, semicolon-joined and unreadable.)
from jaccpot.operators.complex_ops import (
    evaluate_local_complex_derivative_tower_batch,
    evaluate_local_complex_grad_analytic,
    evaluate_local_complex_grad_analytic_preserve_dtype,
    evaluate_local_complex_grad_order4_unrolled,
    evaluate_local_complex_with_grad_analytic,
    evaluate_local_complex_with_grad_analytic_batch,
)
from jaccpot.operators.multipole_utils import (
    MAX_MULTIPOLE_ORDER,
    level_offset,
    total_coefficients,
)
from jaccpot.operators.real_harmonics import (
    evaluate_local_real_derivative_tower_batch,
    evaluate_local_real_with_grad,
)
from jaccpot.operators.symmetric_tensors import component_lift_index_map_3d

from .._octree_adapter import OctreeExecutionData
from ..dtypes import INDEX_DTYPE, as_index
from ._shared import (
    ExpansionBasis,
    NearfieldInteropData,
    PackedAccelerationDerivatives,
)


@partial(jax.jit, static_argnames=("order",))
def _evaluate_local_cartesian_with_grad_batch(
    coeffs: Array,
    offsets: Array,
    *,
    order: int,
) -> tuple[Array, Array]:
    """Evaluate cartesian local expansions and gradients at batch offsets.

    Parameters
    ----------
    coeffs : Array
        Local coefficients, ``[..., total_coefficients(order)]``.
    offsets : Array
        Evaluation points relative to each expansion centre, ``[..., 3]``, with
        leading shape matching ``coeffs``.
    order : int
        Expansion order ``p``. Static under ``jit``.

    Returns
    -------
    tuple[Array, Array]
        ``(gradients, potentials)`` -- gradients first, matching the caller's
        unpacking, with shapes ``[..., 3]`` and ``[...]``. The gradient is read
        off the degree-1 block with its components reversed, because the
        cartesian coefficient layout orders them ``z, y, x``. At ``order <= 0``
        there is no degree-1 block and the gradient is zero.
    """
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
    """Infer generous bounds for tree construction from particle positions.

    Parameters
    ----------
    positions : Array
        Particle positions ``[N, 3]``.

    Returns
    -------
    tuple[Array, Array]
        ``(lower, upper)``, the bounding box padded by 5% of its span per axis,
        floored at 1e-6 so a degenerate axis still yields a non-empty domain.
        "Generous" is deliberate: particles exactly on a boundary are a tree
        build hazard, and the padding costs nothing.
    """

    minimum = jnp.min(positions, axis=0)
    maximum = jnp.max(positions, axis=0)
    span = maximum - minimum
    padding = jnp.maximum(span * 0.05, jnp.full_like(span, 1e-6))
    return minimum - padding, maximum + padding


def _max_leaf_size_from_tree(tree: Tree) -> int:
    """Compute maximum number of particles per leaf node.

    Parameters
    ----------
    tree : Tree
        Built tree. Its leaves are the nodes after the internal ones, so the
        internal count is what separates the two.

    Returns
    -------
    int
        Widest leaf occupancy, as a host ``int``. Concrete, not traced: this
        sizes the padded leaf axis, so it cannot be a tracer. Callers inside a
        trace must pass ``max_leaf_size`` explicitly instead.
    """
    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    leaf_ranges = tree.node_ranges[num_internal:]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + as_index(1)
    return int(jnp.max(counts))


class _TreeEvaluationSetup(NamedTuple):
    """Prevalidated inputs required by tree-evaluation entry points.

    Attributes
    ----------
    locals_data : LocalExpansionData
        The local expansions to evaluate -- the far-field override when one was
        given, otherwise the state's own.
    positions : Array
        Morton-sorted particle positions ``[N, 3]``.
    masses : Array
        Morton-sorted particle masses ``[N]``.
    leaf_nodes : Array
        Node ids of the leaves, in the far-field view's order.
    node_ranges : Array
        Per-node ``[start, end]`` particle ranges, inclusive of ``end``.
    max_leaf_size : int
        Resolved padded leaf width.
    empty_output : Optional[Union[Array, Tuple[Array, Array]]]
        Pre-built zero result for the no-leaves case, shaped to match
        ``return_potential``. When this is not ``None`` the caller must return it
        directly: the rest of the tuple is filled in but not meaningful.
    """

    locals_data: LocalExpansionData
    positions: Array
    masses: Array
    leaf_nodes: Array
    node_ranges: Array
    max_leaf_size: int
    empty_output: Optional[Union[Array, Tuple[Array, Array]]]


class _EvaluationNodeViews(NamedTuple):
    """Resolved leaf/node metadata for shared nearfield and backend-specific farfield.

    The near field and the far field are allowed to disagree about the node view,
    which is the whole point of this split -- see
    :func:`_resolve_evaluation_node_views`.

    Attributes
    ----------
    nearfield : NearfieldInteropData
        The shared radix-oriented leaf/neighbour view. Never overridden.
    farfield_leaf_nodes : Array
        Leaf node ids for the far field; the radix view unless a backend
        supplied its own.
    farfield_node_ranges : Array
        Node ranges for the far field, overridable on the same terms.
    """

    nearfield: NearfieldInteropData
    farfield_leaf_nodes: Array
    farfield_node_ranges: Array


def _infer_order_from_coeff_count(
    *,
    coeff_count: int,
    expansion_basis: ExpansionBasis,
) -> int:
    """Infer expansion order from static coefficient-array width.

    The two bases pack coefficients differently, so the inversion differs: the
    solidfmm width is exactly ``(p + 1) ** 2`` and inverts in closed form, while
    the cartesian width is searched over the supported orders.

    Parameters
    ----------
    coeff_count : int
        Trailing width of the coefficient array. Static, not traced.
    expansion_basis : ExpansionBasis
        Which packing ``coeff_count`` refers to.

    Returns
    -------
    int
        The expansion order ``p``.

    Raises
    ------
    ValueError
        If no order in the basis's packing produces this width -- which means
        the coefficient array is malformed, not that the order is unusual.
    """
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

    Parameters
    ----------
    tree : Tree
        Built tree, supplying the radix node ranges.
    neighbor_list : NodeNeighborList
        Source of truth for the radix leaf ordering, and the near-field view.
    farfield_leaf_nodes : Optional[Array]
        Backend-specific leaf ids for the far field; ``None`` uses the radix
        view.
    farfield_node_ranges : Optional[Array]
        Backend-specific node ranges for the far field; ``None`` as above.

    Returns
    -------
    _EvaluationNodeViews
        The near-field view and the two resolved far-field views.
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

    Parameters
    ----------
    tree : Tree
        Built tree, supplying node ranges.
    neighbor_list : NodeNeighborList
        Leaf ordering and neighbour lists. Used even on the octree path, for the
        particle-order leaf mapping the native list does not carry.
    octree : Optional[OctreeExecutionData]
        Octree metadata. Required when ``native_neighbors`` is given, since the
        carrier lookup is built over the octree's node count.
    native_neighbors : Optional[OctreeNativeNeighborList]
        Octree-native neighbour list. ``None`` takes the radix path.

    Returns
    -------
    NearfieldInteropData
        The explicit leaf/node view the near-field helpers consume.

    Raises
    ------
    ValueError
        If ``native_neighbors`` was given without ``octree``.
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
    """Validate and normalize tree-evaluation inputs for eager/JIT paths.

    Parameters
    ----------
    tree : Tree
        Built tree.
    positions_sorted : Array
        Morton-sorted particle positions ``[N, 3]``.
    masses_sorted : Array
        Morton-sorted particle masses ``[N]``.
    locals_or_downward : Union[LocalExpansionData, TreeDownwardData]
        Either the locals themselves or the downward result carrying them;
        accepting both is what lets the eager and compiled entry points share
        this.
    neighbor_list : NodeNeighborList
        Leaf ordering and neighbour lists.
    farfield_local_data : Optional[LocalExpansionData]
        Far-field locals override; ``None`` uses the ones in
        ``locals_or_downward``.
    farfield_leaf_nodes : Optional[Array]
        Far-field leaf view override; ``None`` uses the radix view.
    farfield_node_ranges : Optional[Array]
        Far-field node-range override; ``None`` as above.
    max_leaf_size : Optional[int]
        Padded leaf width. ``None`` measures it from the tree, which needs
        concrete values -- see Raises.
    return_potential : bool
        Whether the caller wants potentials, which decides the shape of
        ``empty_output``.

    Returns
    -------
    _TreeEvaluationSetup
        Normalized inputs. Check ``empty_output`` first: when it is not ``None``
        there are no leaves and it is the whole answer.

    Raises
    ------
    ValueError
        If the local expansions do not align with the evaluation node ranges, or
        if ``max_leaf_size`` was left ``None`` under a trace, where measuring it
        would require concretizing a tracer.
    """
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
    """JIT core for far/near field evaluation on a prepared tree state.

    The compiled near+far seam. Its width is a consequence of being under
    ``jax.jit``: everything that selects a lane must arrive as an argument, and
    every schedule that was precomputed on the host must arrive as an array.

    **The ``precomputed_*`` arguments are opted into by shape, not by ``None``.**
    Each group is used only if its arrays match the shape the current edge list
    implies; anything else -- including a zero-sized dummy -- silently disables
    that group and falls back to computing the schedule inline. This is why they
    are not ``Optional``: a shape mismatch is the disable signal, so passing a
    stale schedule from a different tree does not raise, it just stops being
    used. Four groups gate independently: the pair lists, the chunk scatter
    schedule, the target blocks, and their padded form.

    Parameters
    ----------
    tree : Tree
        Built tree.
    positions : Array
        Morton-sorted particle positions ``[N, 3]``.
    masses : Array
        Morton-sorted particle masses ``[N]``.
    locals_data : LocalExpansionData
        Local expansions to contract at the particles.
    neighbor_list : NodeNeighborList
        Leaf ordering and neighbour lists. Its edge count is the reference every
        ``precomputed_*`` shape check compares against.
    nearfield_leaf_nodes : Array
        Near-field leaf node ids.
    nearfield_node_ranges : Array
        Near-field per-node particle ranges.
    nearfield_offsets : Array
        CSR offsets into ``nearfield_neighbors``.
    nearfield_neighbors : Array
        Flattened neighbour lists, one segment per leaf.
    nearfield_counts : Array
        Neighbour count per leaf.
    nearfield_leaf_particle_indices : Array
        Padded per-leaf particle indices. A non-empty leading axis is one of the
        conditions for the specialized large-N lane.
    nearfield_leaf_particle_mask : Array
        Validity mask for the padding above.
    leaf_nodes : Array
        Far-field leaf node ids.
    node_ranges : Array
        Far-field per-node particle ranges.
    precomputed_target_leaf_ids : Array
        Per-edge target leaf id. Used only if its length equals the edge count.
    precomputed_source_leaf_ids : Array
        Per-edge source leaf id, gated on its own length in addition to the
        pair-list gate -- so the target ids can be used without these.
    precomputed_valid_pairs : Array
        Per-edge validity mask, gated with the target ids.
    precomputed_chunk_sort_indices : Array
        Chunk-local sort permutation for the scatter, ``[chunks, chunk * leaf]``.
    precomputed_chunk_group_ids : Array
        Chunk-local scatter group ids, same shape.
    precomputed_chunk_unique_indices : Array
        Chunk-local unique-target indices, same shape. All three must match or
        the scatter schedule is recomputed.
    precomputed_target_block_offsets : Array
        CSR offsets over target blocks, length ``leaves + 1``.
    precomputed_target_block_leaf_ids : Array
        Target leaf id per block.
    precomputed_target_block_source_leaf_ids : Array
        Source leaf id per block entry.
    precomputed_target_block_valid_mask : Array
        Validity mask, shaped like the source ids above.
    precomputed_target_block_source_leaf_ids_padded : Array
        Rectangular form of the block source ids, ``[leaves, blocks, width]``.
    precomputed_target_block_valid_mask_padded : Array
        Validity mask for the padded form, gated together with it.
    G : float
        Gravitational constant. Static.
    softening : float
        Plummer softening length. Static.
    order : int
        Expansion order ``p``. Static.
    expansion_basis : ExpansionBasis
        Which expansion algebra the locals are in. Static.
    max_leaf_size : int
        Padded leaf width; sets the chunk flat size the scatter gate checks.
        Static.
    return_potential : bool
        Also return potentials. Static, and disables the specialized large-N
        lane, which is acceleration-only.
    nearfield_mode : str
        Near-field lane. Only ``"bucketed"`` admits the specialized large-N
        lane. Static.
    nearfield_edge_chunk_size : int
        Edges per chunk; with the edge count this fixes the chunk count. Static.
    nearfield_delayed_scatter_chunks_per_superchunk : int
        Chunks accumulated before a scatter is issued. Static.
    nearfield_chunk_scan_batch_size : int
        Chunks per scan step. Static.
    nearfield_chunk_scan_unroll : int
        Unroll factor for the chunk scan. Static.
    nearfield_superchunk_scan_unroll : int
        Unroll factor for the superchunk scan. Static.
    nearfield_sorted_scatter_hint : bool
        Assert the scatter indices are sorted. Static.
    nearfield_grouped_sorted_scatter : bool
        Use the grouped sorted-scatter path. Static.
    nearfield_superchunk_target_reduce : bool
        Reduce per target within a superchunk before scattering. Static.
    nearfield_disable_chunk_cond : bool
        Drop the per-chunk conditional. Static.
    nearfield_target_leaf_batch_size : int
        Target leaves per batch. Static.
    nearfield_target_block_tile_size : int
        Tile width for the target-block lane. Static.
    nearfield_target_block_tile_scan_unroll : int
        Unroll factor for the tile scan. Static.
    nearfield_target_block_batch_scan_unroll : int
        Unroll factor for the batch scan. Static.
    nearfield_target_block_overflow_fast_max_blocks : int
        Block-count ceiling above which the overflow path is taken. Static.
    disable_specialized_large_n_nearfield : bool
        Force the general lane even where the specialized one would apply.
        Static.

    Returns
    -------
    Union[Array, Tuple[Array, Array]]
        Accelerations ``[N, 3]``, or ``(accelerations, potentials)`` under
        ``return_potential``. Morton-sorted, matching the inputs.
    """
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
    # NOTE: deliberately an unresolvable forward reference. Every other module that
    # annotates the engine now imports it under `if TYPE_CHECKING:`, but this one must
    # not: `runtime/kernels/` is a true leaf and "never imports the engine" is the
    # hinge of the layering (ARCHITECTURE §1), which `distributed/` and
    # `experimental/` rely on by reaching past the orchestrator into here. A
    # TYPE_CHECKING import would not run, but it would still make this file name the
    # engine, so the annotation stays documentation-only.
    fmm: "FMMEngine",  # noqa: F821 -- deliberate; see the comment above
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
    nearfield_mode_override: Optional[str] = None,
    nearfield_reverse_options: Optional[Any] = None,
) -> Union[
    Array,
    Tuple[Array, Array],
    Tuple[Array, PackedAccelerationDerivatives],
    Tuple[Array, Array, PackedAccelerationDerivatives],
]:
    """Run the prepared-tree evaluation returning Morton-sorted outputs.

    ``nearfield_mode_override`` forces the near-field mode (the differentiable
    path passes ``"bucketed"`` or ``"fast_lane"``); ``None`` keeps the resolved
    policy. ``nearfield_reverse_options`` carries the grad path's resolved
    reverse-pass tuning down to the leaf-major lane; it is inert on every other
    mode and ``None`` everywhere except the differentiable seam.

    Two routes leave here. With ``max_acc_derivative_order > 0`` this assembles
    the answer itself, because the derivative tower has no compiled path: near
    and far are evaluated separately and summed here. Otherwise it delegates to
    the engine, and ``jit_traversal`` picks which entry point -- only the
    non-compiled one accepts the two override arguments, since the compiled
    traversal resolves its own policy.

    Parameters
    ----------
    fmm : FMMEngine
        The engine, for ``G``, ``softening``, the expansion basis, the near-field
        policy resolvers and the two evaluate entry points. Annotated as an
        unresolvable forward reference on purpose -- see the comment on the
        signature.
    tree : Tree
        Built tree.
    positions_sorted : Array
        Morton-sorted particle positions ``[N, 3]``.
    masses_sorted : Array
        Morton-sorted particle masses ``[N]``.
    downward : TreeDownwardData
        Downward-sweep result carrying the local expansions.
    neighbor_list : NodeNeighborList
        Leaf ordering and neighbour lists.
    nearfield_interop : Optional[NearfieldInteropData]
        Prebuilt near-field view; ``None`` lets the delegate build one.
    farfield_local_data : Optional[LocalExpansionData]
        Far-field locals override; ``None`` uses ``downward.locals``.
    farfield_leaf_nodes : Optional[Array]
        Far-field leaf view override; ``None`` uses the radix view.
    farfield_node_ranges : Optional[Array]
        Far-field node-range override; ``None`` as above.
    nearfield_target_leaf_ids : Optional[Array]
        Precomputed per-edge target leaf ids. ``None`` here, unlike in
        :func:`_evaluate_tree_compiled_impl`, genuinely means absent.
    nearfield_source_leaf_ids : Optional[Array]
        Precomputed per-edge source leaf ids.
    nearfield_valid_pairs : Optional[Array]
        Precomputed per-edge validity mask.
    nearfield_chunk_sort_indices : Optional[Array]
        Precomputed chunk scatter sort permutation.
    nearfield_chunk_group_ids : Optional[Array]
        Precomputed chunk scatter group ids.
    nearfield_chunk_unique_indices : Optional[Array]
        Precomputed chunk unique-target indices.
    max_leaf_size : int
        Padded leaf width.
    return_potential : bool
        Also return potentials.
    jit_traversal : bool
        Use the compiled traversal. ``False`` on the differentiable path, which
        is what lets the two overrides below reach the near field.
    max_acc_derivative_order : int
        Spatial derivative tower depth. Non-zero takes the assemble-here route.
    nearfield_mode_override : Optional[str]
        Forces the near-field mode; see above.
    nearfield_reverse_options : Optional[Any]
        Grad-path reverse-pass tuning; see above. ``Any`` to keep this leaf
        module from importing the grad types.

    Returns
    -------
    Union[Array, Tuple[Array, Array], Tuple[Array, PackedAccelerationDerivatives], Tuple[Array, Array, PackedAccelerationDerivatives]]
        Morton-sorted accelerations, with potentials inserted next under
        ``return_potential`` and the packed derivative tower appended last under
        ``max_acc_derivative_order > 0``.

    Raises
    ------
    RuntimeError
        If the derivative tower was requested but the far-field evaluation
        returned none -- an internal inconsistency, not a user error.
    """

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

    # Only the non-compiled evaluate_tree accepts a near-field mode override;
    # the compiled traversal resolves its own policy. The differentiable path
    # uses jit_traversal=False, so the override reaches the near-field there.
    extra_kwargs = {}
    if jit_traversal:
        evaluate_fn = fmm.evaluate_tree_compiled
    else:
        evaluate_fn = fmm.evaluate_tree
        extra_kwargs["nearfield_mode_override"] = nearfield_mode_override
        extra_kwargs["nearfield_reverse_options"] = nearfield_reverse_options

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
        **extra_kwargs,
    )


def _map_targets_to_leaf_positions(
    *,
    target_sorted_indices: Array,
    leaf_nodes: Array,
    node_ranges: Array,
) -> Array:
    """Map sorted particle indices to positions in the leaf-node array.

    A binary search over leaf start indices, which works because Morton sorting
    makes each leaf a contiguous run. The result indexes ``leaf_nodes``, not the
    tree's node array.

    Parameters
    ----------
    target_sorted_indices : Array
        Target particle indices, in Morton-sorted order.
    leaf_nodes : Array
        Leaf node ids.
    node_ranges : Array
        Per-node ``[start, end]`` particle ranges, inclusive of ``end``.

    Returns
    -------
    Array
        Position in ``leaf_nodes`` for each target, ``[T]``.

    Raises
    ------
    ValueError
        If any target falls in no leaf's range. The search always lands
        somewhere, so this is checked explicitly rather than inferred -- it means
        the targets came from a different tree than the one prepared.
    """
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
    """Build padded source-index lists for each target particle near-field eval.

    Gathers every near-field source each target sees into one rectangular
    matrix, so the targeted near field is a dense masked reduction rather than a
    ragged traversal. Each target's own leaf is concatenated with its neighbour
    leaves, so a leaf that also appears in its own neighbour list would
    contribute its particles twice; the sort-and-keep-first pass at the end
    removes that, and double-counting a source is a wrong force, not a slow one.

    Parameters
    ----------
    target_sorted_indices : Array
        Target particle indices, in Morton-sorted order.
    target_leaf_positions : Array
        Each target's position in the leaf array, from
        :func:`_map_targets_to_leaf_positions`.
    nearfield_interop : NearfieldInteropData
        Leaf/neighbour view supplying the sources.

    Returns
    -------
    tuple[Array, Array]
        ``(source_indices, source_mask)``, both ``[T, S]`` with ``S`` the padded
        source width. Entries where the mask is ``False`` hold index 0 and must
        not be read.
    """
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
    """Compute near-field contributions for target particles only.

    The exact P2P sum over the padded source matrix. The time-derivative terms
    are exact here too, not finite-differenced: they come from differentiating
    the softened kernel analytically, which is why each needs velocities.

    Parameters
    ----------
    positions_sorted : Array
        Morton-sorted particle positions ``[N, 3]``. All particles are sources.
    masses_sorted : Array
        Morton-sorted particle masses ``[N]``.
    target_sorted_indices : Array
        Target particle indices ``[T]``, in the same order.
    source_indices : Array
        Padded per-target source indices ``[T, S]``.
    source_mask : Array
        Validity mask for the padding, ``[T, S]``.
    G : Union[float, Array]
        Gravitational constant. Accepts an array so it can be traced.
    softening : float
        Plummer softening length; enters squared.
    return_potential : bool
        Also return the potential.
    velocities_sorted : Optional[Array]
        Morton-sorted velocities ``[N, 3]``. Required by each of the three
        derivative flags below.
    return_jerk : bool
        Also return the first time derivative.
    return_snap : bool
        Also return the second.
    return_crackle : bool
        Also return the third.

    Returns
    -------
    tuple[Array, Optional[Array], Optional[Array], Optional[Array], Optional[Array]]
        ``(acceleration, potential, jerk, snap, crackle)``. Every element after
        the first is ``None`` unless its flag was set, so the arity is fixed and
        the caller unpacks positionally.

    Raises
    ------
    ValueError
        If ``return_jerk``, ``return_snap`` or ``return_crackle`` was set without
        ``velocities_sorted``.
    """
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
    """Evaluate far-field local expansions for target particles only.

    The targeted twin of :func:`_evaluate_local_expansions_for_particles`: it
    gathers each target's own leaf expansion rather than sweeping padded leaves,
    so there is no scatter at the end.

    Parameters
    ----------
    local_data : LocalExpansionData
        Local expansions, indexed by node.
    positions_sorted : Array
        Morton-sorted particle positions ``[N, 3]``.
    target_sorted_indices : Array
        Target particle indices ``[T]``, in the same order.
    target_leaf_positions : Array
        Each target's position in ``leaf_nodes``.
    leaf_nodes : Array
        Leaf node ids.
    order : int
        Expansion order ``p``.
    expansion_basis : ExpansionBasis
        Which expansion algebra ``local_data`` is in.
    return_potential : bool
        Also return the potential.
    max_acc_derivative_order : int
        Spatial derivative tower depth; ``0`` returns no tower.

    Returns
    -------
    tuple[Array, Optional[Array], Optional[PackedAccelerationDerivatives]]
        ``(gradient, potential, derivatives)``. The first element is the
        *gradient* of the potential, not the acceleration -- the caller applies
        ``-G``. The other two are ``None`` unless requested.

    Raises
    ------
    NotImplementedError
        For orders above ``MAX_MULTIPOLE_ORDER`` in any basis but solidfmm,
        which is the only one whose recurrences are defined that far.
    """
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
    # NOTE: deliberately an unresolvable forward reference. Every other module that
    # annotates the engine now imports it under `if TYPE_CHECKING:`, but this one must
    # not: `runtime/kernels/` is a true leaf and "never imports the engine" is the
    # hinge of the layering (ARCHITECTURE §1), which `distributed/` and
    # `experimental/` rely on by reaching past the orchestrator into here. A
    # TYPE_CHECKING import would not run, but it would still make this file name the
    # engine, so the annotation stays documentation-only.
    fmm: "FMMEngine",  # noqa: F821 -- deliberate; see the comment above
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
    """Run prepared-tree evaluation for target particles only.

    The targeted counterpart of :func:`_evaluate_prepared_tree`. All particles
    remain sources; only the outputs are restricted. There is no compiled route
    here -- the near and far halves are always evaluated separately and summed.

    Parameters
    ----------
    fmm : FMMEngine
        The engine, for ``G``, ``softening`` and the expansion basis. Annotated
        as an unresolvable forward reference on purpose -- see the comment on the
        signature.
    tree : Tree
        Built tree.
    positions_sorted : Array
        Morton-sorted particle positions ``[N, 3]``.
    masses_sorted : Array
        Morton-sorted particle masses ``[N]``.
    downward : TreeDownwardData
        Downward-sweep result carrying the local expansions.
    neighbor_list : NodeNeighborList
        Leaf ordering and neighbour lists.
    nearfield_interop : Optional[NearfieldInteropData]
        Prebuilt near-field view; ``None`` builds one here.
    farfield_local_data : Optional[LocalExpansionData]
        Far-field locals override; ``None`` uses ``downward.locals``.
    farfield_leaf_nodes : Optional[Array]
        Far-field leaf view override; ``None`` uses the radix view.
    farfield_node_ranges : Optional[Array]
        Far-field node-range override; ``None`` as above.
    target_sorted_indices : Array
        Target particle indices ``[T]``, in Morton-sorted order.
    return_potential : bool
        Also return potentials.
    max_acc_derivative_order : int
        Spatial derivative tower depth; ``0`` returns no tower.

    Returns
    -------
    Union[Array, Tuple[Array, Array], Tuple[Array, PackedAccelerationDerivatives], Tuple[Array, Array, PackedAccelerationDerivatives]]
        Results for the targets only, ``[T, ...]``, in the order the targets were
        given. Potentials are inserted next under ``return_potential`` and the
        packed derivative tower appended last.
    """
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
    """Evaluate node-local expansions at leaf particles and scatter results.

    Sweeps every leaf as a padded ``[leaves, max_leaf_size]`` block, evaluates
    the expansion at each slot, and scatters the valid slots back to particle
    order. The padding is masked rather than skipped, which is what keeps the
    shapes static under ``jit``.

    Parameters
    ----------
    local_data : LocalExpansionData
        Local expansions, indexed by node.
    positions : Array
        Morton-sorted particle positions ``[N, 3]``.
    leaf_nodes : Array
        Leaf node ids.
    node_ranges : Array
        Per-node ``[start, end]`` particle ranges, inclusive of ``end``.
    max_leaf_size : int
        Padded leaf width. Static: it sets the block shape.
    order : int
        Expansion order ``p``.
    expansion_basis : ExpansionBasis
        Which expansion algebra ``local_data`` is in.
    return_potential : bool
        Also return the potential.
    max_acc_derivative_order : int
        Spatial derivative tower depth; ``0`` returns no tower.

    Returns
    -------
    tuple[Array, Optional[Array], Optional[PackedAccelerationDerivatives]]
        ``(gradient, potential, derivatives)``, all in particle order. As in the
        targeted twin, the first element is the gradient, not the acceleration.

    Raises
    ------
    NotImplementedError
        For orders above ``MAX_MULTIPOLE_ORDER`` in any basis but solidfmm.
    """
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
    """Scatter-add vector values into a flat particle buffer with masking.

    Masked entries are zeroed rather than dropped, so every padded slot still
    scatters -- adding zero at whatever index the padding holds. That keeps the
    shapes static, and is why the index for an invalid slot need not be valid.

    Parameters
    ----------
    base : Array
        Destination buffer ``[N, 3]``, added into rather than overwritten.
    indices : Array
        Destination particle index per slot; flattened before use.
    values : Array
        Values to add, ``[..., 3]``.
    mask : Array
        Validity mask over the slots.

    Returns
    -------
    Array
        ``base`` with the masked values scatter-added. Returned unchanged when
        ``values`` is empty.
    """
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
    """Scatter-add scalar values into a flat particle buffer with masking.

    The scalar form of :func:`_scatter_vectors`, used for potentials.

    Parameters
    ----------
    base : Array
        Destination buffer ``[N]``, added into rather than overwritten.
    indices : Array
        Destination particle index per slot.
    values : Array
        Values to add. Tolerates ``None`` as well as empty, since the potential
        is optional upstream.
    mask : Array
        Validity mask over the slots.

    Returns
    -------
    Array
        ``base`` with the masked values scatter-added.
    """
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
    """Scatter-add rank-3 values into a particle-major buffer.

    The rank-3 form of :func:`_scatter_vectors`, used for one level of the packed
    derivative tower, whose trailing two axes are the vector component and the
    packed-symmetric index.

    Parameters
    ----------
    base : Array
        Destination buffer ``[N, 3, C]``, added into rather than overwritten.
    indices : Array
        Destination particle index per slot.
    values : Array
        Values to add, ``[..., 3, C]``.
    mask : Array
        Validity mask over the slots.

    Returns
    -------
    Array
        ``base`` with the masked values scatter-added.
    """
    if values.size == 0:
        return base
    flat_idx = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-2], values.shape[-1])
    flat_mask = mask.reshape(-1)
    zero = jnp.zeros((), dtype=base.dtype)
    masked = jnp.where(flat_mask[:, None, None], flat_values, zero)
    return base.at[flat_idx].add(masked)
