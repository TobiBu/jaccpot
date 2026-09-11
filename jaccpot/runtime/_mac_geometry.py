"""MAC geometry about the expansion centres (plan sub-10ms, Phase 1.2).

The multipole-acceptance test the dual-tree walk evaluates -- ``(r_A + r_B)^2
<= theta^2 d^2`` in ``yggdrax._interactions_impl._compute_mac_ok`` -- is only a
convergence guarantee for the M2L expansion when ``r_A``, ``r_B`` bound the two
nodes' particles ABOUT THE CENTRES THE EXPANSIONS USE and ``d`` is the distance
between those same centres. The fused lane's real-basis upward sweep expands
about the centres of mass (``center_mode='com'`` is the only mode it accepts;
M2L, L2L and L2P all use those centres), but the walk was handed
``TreeGeometry`` -- bounding-BOX centres and the box half-diagonal as radius.

Measured 2026-09-11 (``probe_mac_consistency.py``, N=2x10^5 Plummer, leaf 64):
about the COMs the accepted far pairs reach convergence ratios of 1.56 at theta
0.6 (0.6 % of pairs at or above 1), 2.38 at theta 0.8 (8.6 %) and 4.84 at theta
1.0 (35 %) -- divergent expansions, which is why the force error at theta >= 0.8
stopped improving with the order and at theta 1.0 GREW with it (1.7e-2 -> 3.7e-2
from p = 4 to 6), and why jaccpot needed 16x more direct pairs than jz-fmm at
matched error.

:func:`com_mac_geometry` builds a ``TreeGeometry`` whose centres are the COMs
and whose radii bound every particle of the node about its COM: exact for the
leaves (one gather over the leaf particle table), and for internal nodes
either EXACT (default, ``internal="exact"``: every leaf walks its ancestor
chain by pointer jumping and each ancestor takes the segment-max of the leaf's
particle distances about the ancestor's centre -- one gather + segment-max per
tree level) or the conservative upward bound ``r_p = max_c (|c_c - c_p| + r_c)``
(``internal="bound"``, level by level like ``yggdrax._geometry_impl``). Measured
on the real 2e5 / leaf-64 tree at theta 0.8 (``probe_tree_volume.py``): the
bound is 1.38x (p50) / 1.94x (p90) over exact and costs 2.7x in far pairs
(565k -> 1541k directed) for 7 % fewer near edges, so exact is the default.
``half_extent`` and ``max_extent`` are set to the radius too, so the ``bh`` box
test degrades to the sphere test rather than to a stale box.

Selected in the fused lane by ``JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY``
(``"aabb"`` = the historical box geometry, ``"com"`` = this module); see
:func:`resolve_walk_geometry`.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array
from yggdrax.dtypes import INDEX_DTYPE, as_index
from yggdrax.geometry import TreeGeometry

__all__ = [
    "com_mac_geometry",
    "mac_geometry_mode",
    "mac_radius_mode",
    "resolve_walk_geometry",
]

_MAC_GEOMETRY_ENV = "JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY"
_MAC_GEOMETRY_MODES = ("aabb", "com")
_MAC_RADIUS_ENV = "JACCPOT_STATIC_STRICT_FUSED_MAC_RADIUS"
_MAC_RADIUS_MODES = ("exact", "bound")


def mac_geometry_mode() -> str:
    """Which geometry the fused lane's walk tests the MAC against.

    Read at call time (never captured at import), like every runtime knob.

    Returns
    -------
    str
        ``"aabb"`` (default: box centres and half-diagonals, the historical
        behaviour) or ``"com"`` (centres of mass with particle radii about them).

    Raises
    ------
    ValueError
        If the environment names a mode this module does not implement.
    """
    raw = os.environ.get(_MAC_GEOMETRY_ENV, "aabb").strip().lower()
    if raw not in _MAC_GEOMETRY_MODES:
        raise ValueError(
            f"{_MAC_GEOMETRY_ENV} must be one of {_MAC_GEOMETRY_MODES}, got {raw!r}"
        )
    return raw


def mac_radius_mode() -> str:
    """How ``com_mac_geometry`` sizes internal nodes: ``"exact"`` (default) or ``"bound"``.

    Returns
    -------
    str
        The mode named by ``JACCPOT_STATIC_STRICT_FUSED_MAC_RADIUS``.

    Raises
    ------
    ValueError
        If the environment names a mode this module does not implement.
    """
    raw = os.environ.get(_MAC_RADIUS_ENV, "exact").strip().lower()
    if raw not in _MAC_RADIUS_MODES:
        raise ValueError(
            f"{_MAC_RADIUS_ENV} must be one of {_MAC_RADIUS_MODES}, got {raw!r}"
        )
    return raw


def _node_depths(parent: Array) -> Array:
    """Depth of every node from the parent array by pointer doubling (root = 0)."""
    num_nodes = int(parent.shape[0])
    parent_safe = jnp.where(parent >= 0, parent, as_index(0))
    is_root = parent < 0
    init_dist = jnp.where(is_root, as_index(0), as_index(1))
    init_shortcut = jnp.where(is_root, jnp.arange(num_nodes, dtype=INDEX_DTYPE), parent_safe)

    def _cond(state):
        _sc, _d, changed = state
        return changed

    def _body(state):
        sc, d, _changed = state
        new_d = d + d[sc]
        new_sc = sc[sc]
        return new_sc, new_d, jnp.any(new_sc != sc)

    _, depth, _ = lax.while_loop(_cond, _body, (init_shortcut, init_dist, jnp.bool_(True)))
    return depth


def com_mac_geometry(
    tree: Any,
    positions_sorted: Array,
    centers: Array,
    *,
    leaf_cap: int,
    internal: str = "exact",
) -> TreeGeometry:
    """``TreeGeometry`` about the expansion centres: COM centres, particle radii about them.

    Parameters
    ----------
    tree : Any
        Static radix tree with ``node_ranges`` (inclusive particle ranges per
        node, ``(nodes, 2)``), ``left_child`` / ``right_child`` (``(internal,)``)
        and ``parent`` (``(nodes,)``); leaves are the last ``nodes - internal``
        nodes.
    positions_sorted : Array
        Particle positions in tree order, ``(n, 3)``.
    centers : Array
        The expansion centres per node, ``(nodes, 3)`` -- the upward sweep's
        ``multipoles.centers`` (centres of mass).
    leaf_cap : int
        Leaf capacity (particles per leaf at most). Static.
    internal : str
        ``"exact"`` (max particle distance about the node's own centre, via the
        leaves' ancestor chains) or ``"bound"`` (child-sphere bound). Static.

    Returns
    -------
    TreeGeometry
        ``center = centers``; ``radius`` bounds every particle of the node about
        its centre (exact for leaves and, with ``internal="exact"``, for every
        node); ``half_extent`` and ``max_extent`` equal the radius.

    Raises
    ------
    ValueError
        If ``internal`` is not ``"exact"`` or ``"bound"``.
    """
    if internal not in _MAC_RADIUS_MODES:
        raise ValueError(f"internal must be one of {_MAC_RADIUS_MODES}, got {internal!r}")
    positions_sorted = jnp.asarray(positions_sorted)
    dtype = positions_sorted.dtype
    centers = jnp.asarray(centers, dtype=dtype)
    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    left_child = jnp.asarray(tree.left_child, dtype=INDEX_DTYPE)
    right_child = jnp.asarray(tree.right_child, dtype=INDEX_DTYPE)
    parent = jnp.asarray(tree.parent, dtype=INDEX_DTYPE)
    num_nodes = int(node_ranges.shape[0])
    num_internal = int(left_child.shape[0])
    num_leaves = num_nodes - num_internal
    n = int(positions_sorted.shape[0])
    w = max(1, int(leaf_cap))

    # --- leaves: exact max distance about the centre over the leaf's particles
    leaf_ranges = node_ranges[num_internal:]
    lane = jnp.arange(w, dtype=INDEX_DTYPE)
    idx = leaf_ranges[:, 0][:, None] + lane[None, :]
    valid = idx <= leaf_ranges[:, 1][:, None]
    safe = jnp.clip(idx, 0, max(n - 1, 0))
    pts = positions_sorted[safe]  # (L, w, 3)
    d = jnp.linalg.norm(pts - centers[num_internal:][:, None, :], axis=-1)
    r_leaf = jnp.max(jnp.where(valid, d, jnp.asarray(0.0, dtype)), axis=1)

    radii = jnp.zeros((num_nodes,), dtype=dtype).at[num_internal:].set(r_leaf)

    if num_internal > 0 and internal == "exact":
        # Every internal node's particles are exactly the union of its
        # descendant leaves', so walking each leaf up its ancestor chain and
        # taking, per ancestor, the max of that leaf's particle distances about
        # the ancestor's centre gives the exact radius. Pointer jumping over the
        # parent array; the loop runs to the tree's depth (traced trip count).
        leaf_ids = jnp.arange(num_internal, num_nodes, dtype=INDEX_DTYPE)
        dist_masked = lambda c: jnp.max(  # noqa: E731 - (L,) max over the leaf's lanes
            jnp.where(valid, jnp.linalg.norm(pts - c[:, None, :], axis=-1), jnp.asarray(0.0, dtype)),
            axis=1,
        )

        def _cond(state):
            anc, _r = state
            return jnp.any(anc >= 0)

        def _body(state):
            anc, r = state
            live = anc >= 0
            anc_safe = jnp.where(live, anc, 0)
            d = jnp.where(live, dist_masked(centers[anc_safe]), jnp.asarray(0.0, dtype))
            r = r.at[anc_safe].max(d)
            return jnp.where(live, parent[anc_safe], jnp.asarray(-1, INDEX_DTYPE)), r

        _, radii = lax.while_loop(_cond, _body, (parent[leaf_ids], radii))

    elif num_internal > 0:
        depth = _node_depths(parent)
        max_depth = jnp.max(depth)
        internal_depth = depth[:num_internal]
        c_int = centers[:num_internal]
        off_l = jnp.linalg.norm(centers[left_child] - c_int, axis=-1)
        off_r = jnp.linalg.norm(centers[right_child] - c_int, axis=-1)

        def _body(rev_idx, r):
            level = max_depth - as_index(1) - as_index(rev_idx)
            at_level = internal_depth == level
            bound = jnp.maximum(off_l + r[left_child], off_r + r[right_child])
            return r.at[:num_internal].set(jnp.where(at_level, bound, r[:num_internal]))

        radii = lax.fori_loop(0, jnp.maximum(max_depth, as_index(0)), _body, radii)

    half_extent = jnp.broadcast_to(radii[:, None], (num_nodes, 3))
    return TreeGeometry(centers, half_extent, radii, radii)


def resolve_walk_geometry(
    tree: Any,
    positions_sorted: Array,
    box_geometry: Optional[TreeGeometry],
    expansion_centers: Optional[Array],
    *,
    leaf_cap: int,
    geometry_factory: Optional[Any] = None,
) -> tuple[Optional[TreeGeometry], Optional[Any]]:
    """The geometry the walk should test the MAC against, per ``mac_geometry_mode``.

    Parameters
    ----------
    tree : Any
        Built tree.
    positions_sorted : Array
        Positions in tree order.
    box_geometry : Optional[TreeGeometry]
        The upward stage's box geometry (may be ``None`` when deferred).
    expansion_centers : Optional[Array]
        The upward sweep's expansion centres; required for ``"com"``.
    leaf_cap : int
        Leaf capacity.
    geometry_factory : Optional[Any]
        The deferred box-geometry builder the caller would otherwise pass on.

    Returns
    -------
    tuple[Optional[TreeGeometry], Optional[Any]]
        ``(geometry, geometry_factory)`` to hand to the artifacts builder:
        unchanged in ``"aabb"`` mode; the COM geometry and ``None`` in ``"com"``.

    Raises
    ------
    RuntimeError
        ``"com"`` requested but the upward data carries no expansion centres.
    """
    if mac_geometry_mode() == "aabb":
        return box_geometry, geometry_factory
    if expansion_centers is None:
        raise RuntimeError(
            f"{_MAC_GEOMETRY_ENV}=com needs the upward sweep's expansion centres, "
            "and this lane's upward data has none."
        )
    geometry = com_mac_geometry(
        tree,
        positions_sorted,
        expansion_centers,
        leaf_cap=int(leaf_cap),
        internal=mac_radius_mode(),
    )
    return geometry, None
