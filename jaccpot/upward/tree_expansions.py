"""Helpers for constructing node multipole expansions.

Axis names used in the annotations below, per STYLE_GUIDE section 4:

    n         particles
    nodes     tree nodes (``tree.parent.shape[0]``)
    internal  internal nodes (``tree.left_child.shape[0]``)
    ct        packed coefficients per node

``ct`` is deliberately NOT the package-wide ``c``. Elsewhere ``C`` means
``sh_size(p) == (p+1)**2``, the spherical-harmonic packing. This module packs
CARTESIAN moments, so its coefficient count is
``total_coefficients(p) == (p+1)(p+2)(p+3)/6`` -- 10 at p=2 where ``sh_size``
is 9. The two agree only at p=1, which is why one symbol served both for so
long. Measured 2026-08-18; see the ``packed`` attribute below.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float, Int, jaxtyped
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
    """Packed multipole expansions and their metadata.

    Attributes
    ----------
    order : int
        Expansion order ``p``.
    centers : Float[Array, 'nodes 3']
        ``(nodes, 3)`` expansion centres.
    moments : TreeMultipoleMoments
        Raw (unpacked) moments the packing was built from.
    packed : Float[Array, 'nodes ct']
        ``(nodes, total_coefficients(order))`` packed coefficients -- what the
        sweeps actually consume.

        **Corrected 2026-08-18.** This said ``sh_size(order)``, i.e. ``(p+1)^2``,
        and that is wrong: the packing is Cartesian, so the count is
        ``total_coefficients(order) == (p+1)(p+2)(p+3)/6``. Measured for p=1..4:
        4, 10, 20, 35 columns against ``sh_size`` of 4, 9, 16, 25. Only p=1
        agrees, and p=2 is the default order -- so the documented shape was wrong
        for every order this package actually runs at.
    component_matrix : Optional[Float[Array, 'nodes ct']]
        Per-component view, when a lane needs one; ``None`` otherwise. Same
        shape as ``packed`` -- both come from ``moments.raw_packed``.
    source_motion_packed : Optional[Float[Array, 'nodes ct']]
        Time-differentiated multipoles, present only when a derivative path asked
        for them. Always ``None`` on the paths in this module.
    """

    order: int
    centers: Float[Array, "nodes 3"]
    moments: TreeMultipoleMoments
    packed: Float[Array, "nodes ct"]
    component_matrix: Optional[Float[Array, "nodes ct"]]
    source_motion_packed: Optional[Float[Array, "nodes ct"]] = None


@partial(jax.jit, static_argnames=("order", "num_internal"))
def _aggregate_m2m_impl(
    packed: Float[Array, "nodes ct"],
    centers: Float[Array, "nodes 3"],
    left_child: Int[Array, "internal"],
    right_child: Int[Array, "internal"],
    node_ranges: Int[Array, "nodes 2"],
    *,
    order: int,
    num_internal: int,
    # The return is deliberately bare `Array`, not `Float[Array, "nodes ct"]`.
    # pydoclint 0.9.1 CRASHES on a shaped return whose axis spec has more than
    # one token: `ReturnAnnotation.decompose()` re-parses the annotation as
    # Python and `Float[Array, nodes ct]` is a SyntaxError. Single-token axes
    # (`Float[Array, "n"]`) and shaped PARAMETERS are both fine -- it is
    # specifically multi-token axes in a return position. Measured 2026-08-18.
    # The shape is documented in the Returns section instead.
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

    Parameters
    ----------
    packed : Float[Array, 'nodes ct']
        ``(nodes, ct)`` packed coefficients, updated in place through the
        returned value; leaves must already hold their P2M result. ``ct`` is the
        Cartesian count, not ``(p+1)^2`` -- see the module docstring.
    centers : Float[Array, 'nodes 3']
        ``(nodes, 3)`` expansion centres.
    left_child : Int[Array, 'internal']
        Left child of each internal node. Length ``internal``, NOT ``nodes``:
        only internal nodes have children.
    right_child : Int[Array, 'internal']
        Right child of each internal node.
    node_ranges : Int[Array, 'nodes 2']
        Per-node particle span, ``(nodes, 2)``. The SPAN WIDTH is what orders the
        reduction -- see above; this is not merely bookkeeping here.
    order : int
        Expansion order ``p``.
    num_internal : int
        Internal-node count.

    Returns
    -------
    Array
        ``(nodes, ct)`` packed coefficients with every internal node
        aggregated from its children -- same shape and dtype as ``packed``.
        Not annotated as ``Float[Array, 'nodes ct']``; see the signature.
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
    positions_sorted: Float[Array, "n 3"],
    masses_sorted: Float[Array, "n"],
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
    positions_sorted : Float[Array, 'n 3']
        Particle positions ``[n, 3]``, reordered to match
        ``tree.particle_indices``. Passing unsorted positions is silently wrong,
        not an error.
    masses_sorted : Float[Array, 'n']
        Particle masses ``[n]``, in the same order as ``positions_sorted``. G=1
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
    of the particle positions*, and that coupling **is carried all the way
    through** -- a ``"com"`` gradient is exact for the fixed-topology force, not
    an approximation that drops the centre-motion term. ``"explicit"`` has no
    such coupling; ``"aabb"`` does have one, via the min/max subgradient of
    :func:`~jaccpot.upward.tree_geometry.compute_tree_geometry_compiled`, which
    is also a live function of the positions.

    The chain, for anyone tempted to insert a ``stop_gradient`` here:
    :meth:`~jaccpot.runtime.fmm_evaluate.EvaluateMixin.differentiable_accelerations`
    re-derives the centres from the live inputs on every call -- it calls
    ``prepare_upward_sweep`` on the live ``positions``/``masses`` and reads only
    ``int(state.downward.locals.order)``, a Python int, off the frozen state. The
    M2L/L2L pair displacements are then ``centers[tgt] - centers[src]`` on those
    live centres, and L2P uses ``leaf_positions - centers``. There is no
    ``stop_gradient`` anywhere on the single-GPU path; the only ones in the tree
    are in :mod:`jaccpot.distributed.fmm`, where they freeze the frontier used to
    *build* the coarse tree while ``_live_coarse_payload`` deliberately keeps the
    coarse COMs live -- its docstring records that freezing them "would silently
    drop a real gradient term".

    Guarded by ``tests/unit/test_gradient_correctness.py::test_fd_vs_ad_positions``,
    whose finite-difference reference perturbs the *same* frozen-topology
    function and therefore moves the COM centres too; it would fail if the
    centre-motion term were dropped from the reverse pass. Measured agreement
    ~1e-9. ``"com"`` is also the production default (resolved in
    :mod:`jaccpot.runtime.fmm_overrides`), and the real-basis upward sweep
    accepts nothing else.

    See ``docs/differentiable_fmm_design.md`` ("COM centers are differentiated
    through, not held fixed") and ``docs/differentiable_fmm_audit.md`` for the
    audit that established this.

    The one caveat: on degenerate displacements -- exactly-zero (structural under
    COM, since a single-child internal node shares its child's COM) and
    z-axis-aligned -- the rotation-angle builders return a zero cotangent by
    construction. See :func:`jaccpot.operators.complex_ops._angles_from_delta_solidfmm`.
    """

    # `explicit_centers` is deliberately left bare. Its shape IS known --
    # `(nodes, 3)`, and both functions check it explicitly below -- but this
    # function carries `@jaxtyped(typechecker=beartype)`, which is active in
    # every run, not only under JACCPOT_RUNTIME_TYPECHECK=1. Annotating the
    # shape would make a wrong-shaped array raise a beartype violation BEFORE
    # the body runs, replacing the `ValueError` this function documents in its
    # Raises section and making that branch unreachable for shape errors. That
    # is a behaviour change, not a documentation change, so it is not being
    # made in the same PR that adds annotations. See the pilot writeup.

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
    """Container bundling data needed for the FMM upward sweep.

    Attributes
    ----------
    geometry : TreeGeometry
        Node centres, radii and extents.
    mass_moments : TreeMassMoments
        Per-node mass moments.
    multipoles : NodeMultipoleData
        Packed expansions and their metadata.
    """

    geometry: TreeGeometry
    mass_moments: TreeMassMoments
    multipoles: NodeMultipoleData


@jaxtyped(typechecker=beartype)
def prepare_upward_sweep(
    tree: Tree,
    positions_sorted: Float[Array, "n 3"],
    masses_sorted: Float[Array, "n"],
    *,
    max_order: int = 2,
    center_mode: str = "com",
    explicit_centers: Optional[Array] = None,
    precomputed_geometry: Optional[TreeGeometry] = None,
) -> TreeUpwardData:
    """Compute geometry, moments, and packed multipoles for a tree.
    The P2M + M2M half of the pipeline, in the complex/solidfmm basis.

    Parameters
    ----------
    tree : Tree
        Tree to build expansions for.
    positions_sorted : Float[Array, 'n 3']
        ``(n, 3)`` positions in the tree's own order.
    masses_sorted : Float[Array, 'n']
        ``(n,)`` masses, same order.
    max_order : int
        Expansion order ``p``.
    center_mode : str
        How expansion centres are chosen.
    explicit_centers : Optional[Array]
        Caller-supplied centres, when ``center_mode`` asks for them.
    precomputed_geometry : Optional[TreeGeometry]
        Reuse an already-built geometry instead of recomputing it.

    Returns
    -------
    TreeUpwardData
        Geometry, mass moments and packed multipoles.

    Raises
    ------
    ValueError
        If ``center_mode`` is outside its documented domain, or
        ``explicit_centers`` is required and missing (or the wrong shape).
    """

    # `explicit_centers` is deliberately left bare. Its shape IS known --
    # `(nodes, 3)`, and both functions check it explicitly below -- but this
    # function carries `@jaxtyped(typechecker=beartype)`, which is active in
    # every run, not only under JACCPOT_RUNTIME_TYPECHECK=1. Annotating the
    # shape would make a wrong-shaped array raise a beartype violation BEFORE
    # the body runs, replacing the `ValueError` this function documents in its
    # Raises section and making that branch unreachable for shape errors. That
    # is a behaviour change, not a documentation change, so it is not being
    # made in the same PR that adds annotations. See the pilot writeup.

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
