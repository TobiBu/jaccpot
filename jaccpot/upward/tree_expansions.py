"""Helpers for constructing node multipole expansions."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, jaxtyped
from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.geometry import TreeGeometry
from yggdrax.multipole_utils import total_coefficients
from yggdrax.tree import Tree
from yggdrax.tree_moments import (
    TreeMassMoments,
    TreeMultipoleMoments,
    compute_tree_mass_moments,
    compute_tree_multipole_moments,
    pack_multipole_expansions,
    translate_packed_moments,
    tree_moments_from_raw,
)

from .tree_geometry import compute_tree_geometry_compiled

_CENTER_MODES = ("com", "aabb", "explicit")


class NodeMultipoleData(NamedTuple):
    """Packed multipole expansions and their metadata."""

    order: int
    centers: Array
    moments: TreeMultipoleMoments
    packed: Array
    component_matrix: Optional[Array]
    source_motion_packed: Optional[Array] = None


@partial(jax.jit, static_argnames=("order", "num_internal"))
def _aggregate_m2m_impl(
    packed: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    node_ranges: Array,
    *,
    order: int,
    num_internal: int,
) -> Array:
    """Translate child expansions into their parents, children first.

    The iteration order is the whole correctness content of this function: a
    parent must be visited only after both of its children hold their final
    expansions, or it aggregates whatever the array happened to contain -- zeros.

    This used to walk descending node index, which assumes children are stored
    after their parents. Radix internal nodes are *not* in postorder: measured on
    a 1024-particle radix tree, 26 of 63 internal nodes have an internal child
    with a lower index than the parent. Those parents were built from a zero
    child, and the hole propagated to the root, so 10-23% of the system mass was
    missing from the root monopole on default settings and every M2L sourced from
    an affected node silently dropped its particles' contribution.

    Reduce in ascending node span instead. Children partition their parent, so a
    parent's span is strictly wider than either child's, which makes ascending
    span a valid topological order for any tree shape. This matches the pattern
    ``compute_tree_merged_sphere_geometry`` and the force-scale node reduction
    already use for the same reason.
    """

    prototype = packed[0]
    internal_ranges = jnp.asarray(node_ranges)[:num_internal]
    internal_width = internal_ranges[:, 1] - internal_ranges[:, 0]
    internal_order = jnp.argsort(internal_width, stable=True)

    def add_child(
        node_coeff: Array,
        child_idx: Array,
        node_idx: Array,
        state: Array,
    ) -> Array:
        def true_branch(idx: Array) -> Array:
            delta = centers[idx] - centers[node_idx]
            translated = translate_packed_moments(
                state[idx],
                delta,
                order,
            )
            return node_coeff + translated

        return lax.cond(
            child_idx >= 0,
            true_branch,
            lambda _: node_coeff,
            child_idx,
        )

    def body(iter_idx: Array, state: Array) -> Array:
        node_idx = internal_order[iter_idx]
        node_coeff = jnp.zeros_like(prototype)
        node_coeff = add_child(
            node_coeff,
            left_child[node_idx],
            node_idx,
            state,
        )
        node_coeff = add_child(
            node_coeff,
            right_child[node_idx],
            node_idx,
            state,
        )
        state = state.at[node_idx].set(node_coeff)
        return state

    return lax.fori_loop(0, num_internal, body, packed)


@jaxtyped(typechecker=beartype)
def compute_node_multipoles(
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    max_order: int = 2,
    center_mode: str = "com",
    explicit_centers: Optional[Array] = None,
) -> NodeMultipoleData:
    """Construct packed multipole expansions for every node in the tree.

    Parameters
    ----------
    tree : Tree
        Radix tree built from Morton-sorted particles, as produced by
        ``yggdrax``. Its ``left_child`` / ``right_child`` / ``node_ranges``
        arrays define the aggregation order.
    positions_sorted : Array
        Particle positions ``[N, 3]``, reordered to match
        ``tree.particle_indices``. Passing unsorted positions is silently wrong,
        not an error.
    masses_sorted : Array
        Particle masses ``[N]``, in the same order as ``positions_sorted``. G=1
        is a caller convention; no gravitational constant is applied here.
    max_order : int
        Highest Cartesian multipole order to keep. Static under ``jit``: it fixes
        the packed coefficient count.
    center_mode : str
        ``"com"`` uses each node's centre of mass, ``"aabb"`` uses the geometry
        centre, and ``"explicit"`` consumes ``explicit_centers``. Static under
        ``jit`` -- it selects a Python branch, not a traced one.
    explicit_centers : Optional[Array]
        User-provided expansion centres ``[num_nodes, 3]``, required when
        ``center_mode == "explicit"`` and ignored otherwise.

    Returns
    -------
    NodeMultipoleData
        Packed expansions for every node plus the metadata needed downstream:
        the realised ``order``, the per-node ``centers`` ``[num_nodes, 3]``, the
        raw ``moments``, and the ``packed`` coefficients. ``source_motion_packed``
        is always ``None`` here -- only the derivative/jerk paths populate it.

    Raises
    ------
    ValueError
        If ``center_mode`` is not one of ``"com"``, ``"aabb"``, ``"explicit"``;
        or if ``center_mode == "explicit"`` and ``explicit_centers`` is missing
        or is not ``[num_nodes, 3]``. All three are host-side checks on static
        values, so they fire before tracing.

    Notes
    -----
    Differentiable in ``positions_sorted``, ``masses_sorted``, and
    ``explicit_centers``. Not differentiable in ``tree``, which carries integer
    topology only.

    A node holding a single particle, or several coincident particles, gives a
    zero-radius expansion: valid, and exactly the case where the far-field
    accuracy depends entirely on the caller's MAC. Empty nodes carry zero mass
    and contribute nothing.

    ``center_mode="com"`` makes the expansion centres a *differentiable function
    of the particle positions*, so a gradient through this function includes the
    centre's own motion; ``"aabb"`` and ``"explicit"`` do not have that coupling
    in the same way.
    TODO(docs): is the "com" centre-motion term actually carried through the
    downward sweep's gradient, or does something downstream stop_gradient the
    centres? `docs/differentiable_fmm_design.md` does not say either way, and
    the answer decides whether a "com" gradient is exact or approximate.
    """

    mode = center_mode.lower()
    if mode not in _CENTER_MODES:
        raise ValueError(f"Unknown center_mode '{center_mode}'")

    centers: Optional[Array]
    if mode == "explicit":
        if explicit_centers is None:
            raise ValueError(
                "explicit_centers must be provided for 'explicit'",
            )
        if explicit_centers.shape != (tree.parent.shape[0], 3):
            raise ValueError("explicit_centers must have shape (num_nodes, 3)")
        centers = jnp.asarray(explicit_centers, dtype=positions_sorted.dtype)
    elif mode == "aabb":
        geom = compute_tree_geometry_compiled(tree, positions_sorted)
        centers = geom.center
    else:
        centers = None

    moments = compute_tree_multipole_moments(
        tree,
        positions_sorted,
        masses_sorted,
        expansion_centers=centers,
        max_order=max_order,
    )

    packed = pack_multipole_expansions(moments, max_order=max_order)

    return NodeMultipoleData(
        order=int(moments.max_order),
        centers=moments.center,
        moments=moments,
        packed=packed,
        component_matrix=moments.raw_packed,
        source_motion_packed=None,
    )


def _aggregate_multipoles_via_m2m(
    tree: Tree,
    centers: Array,
    base_moments: TreeMultipoleMoments,
) -> TreeMultipoleMoments:
    total_nodes = base_moments.mass.shape[0]
    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    order = int(base_moments.max_order)
    coeffs = total_coefficients(order)
    packed = jnp.zeros(
        (total_nodes, coeffs),
        dtype=base_moments.raw_packed.dtype,
    )

    if num_internal < total_nodes:
        leaf_slice = slice(num_internal, total_nodes)
        packed = packed.at[leaf_slice].set(
            base_moments.raw_packed[leaf_slice, :coeffs],
        )

    if num_internal > 0:
        left_child = jnp.asarray(tree.left_child, dtype=INDEX_DTYPE)
        right_child = jnp.asarray(tree.right_child, dtype=INDEX_DTYPE)
        packed = _aggregate_m2m_impl(
            packed,
            centers,
            left_child,
            right_child,
            jnp.asarray(tree.node_ranges),
            order=order,
            num_internal=num_internal,
        )

    return tree_moments_from_raw(packed, centers, order)


class TreeUpwardData(NamedTuple):
    """Container bundling data needed for the FMM upward sweep."""

    geometry: TreeGeometry
    mass_moments: TreeMassMoments
    multipoles: NodeMultipoleData


@jaxtyped(typechecker=beartype)
def prepare_upward_sweep(
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    max_order: int = 2,
    center_mode: str = "com",
    explicit_centers: Optional[Array] = None,
    precomputed_geometry: Optional[TreeGeometry] = None,
) -> TreeUpwardData:
    """Compute geometry, moments, and packed multipoles for a tree."""

    geometry = (
        precomputed_geometry
        if precomputed_geometry is not None
        else compute_tree_geometry_compiled(tree, positions_sorted)
    )
    mass_moments = compute_tree_mass_moments(
        tree,
        positions_sorted,
        masses_sorted,
    )

    total_nodes = tree.parent.shape[0]
    mode = center_mode.lower()
    if mode == "com":
        centers = mass_moments.center_of_mass
    elif mode == "aabb":
        centers = geometry.center
    elif mode == "explicit":
        if explicit_centers is None:
            raise ValueError(
                "explicit_centers must be provided for 'explicit'",
            )
        if explicit_centers.shape != (total_nodes, 3):
            raise ValueError("explicit_centers must have shape (num_nodes, 3)")
        centers = explicit_centers
    else:
        raise ValueError(f"Unknown center_mode '{center_mode}'")

    centers = jnp.asarray(centers, dtype=positions_sorted.dtype)

    direct_moments = compute_tree_multipole_moments(
        tree,
        positions_sorted,
        masses_sorted,
        expansion_centers=centers,
        max_order=max_order,
    )

    aggregated = _aggregate_multipoles_via_m2m(
        tree,
        centers,
        direct_moments,
    )

    packed = pack_multipole_expansions(aggregated, max_order=max_order)

    multipoles = NodeMultipoleData(
        order=int(aggregated.max_order),
        centers=centers,
        moments=aggregated,
        packed=packed,
        component_matrix=aggregated.raw_packed,
        source_motion_packed=None,
    )

    return TreeUpwardData(
        geometry=geometry,
        mass_moments=mass_moments,
        multipoles=multipoles,
    )


__all__ = [
    "NodeMultipoleData",
    "TreeUpwardData",
    "compute_node_multipoles",
    "prepare_upward_sweep",
]
