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
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array
from yggdrax.dtypes import INDEX_DTYPE, as_index
from yggdrax.geometry import TreeGeometry

_MAX_TREE_LEVELS = (
    64  # yggdrax._tree_impl.MAX_TREE_LEVELS: level tables are padded to it
)

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


def mac_geometry_mode(default: str = "aabb") -> str:
    """Which geometry the fused lane's walk tests the MAC against.

    Read at call time (never captured at import), like every runtime knob.

    Parameters
    ----------
    default : str
        Mode to use when the environment does not name one. The strict fused lane
        passes ``"com"``; every other lane leaves it at ``"aabb"``.

        WHY NOT "com" EVERYWHERE. The COM MAC is *consistent* -- it tests the
        criterion about the centres the expansions actually use -- but at a fixed
        theta it is not equivalent to the box criterion: the box radius is the
        half-DIAGONAL, which is strictly larger than the exact COM radius, so the
        box test is the more conservative one and the same theta admits MORE far
        pairs under COM. Measured on tests/integration/test_fmm.py's order sweep
        (solidfmm+dehnen, theta 0.9, leaf 16, N=224), COM is worse at every order:
        rel-L2 5.8e-3 / 2.1e-3 / 3.4e-4 at orders 1/2/4 under the box geometry
        against 2.1e-2 / 7.7e-3 / 1.9e-3 under COM. Making it the global default
        therefore silently degraded accuracy at every caller's existing theta.
        It was measured and tuned on the strict fused lane at theta 0.6-1.0 with
        cell leaves, which is the only lane that gets it by default.

    Returns
    -------
    str
        ``"aabb"`` (box centres and half-diagonals, the historical
        behaviour) or ``"com"`` (centres of mass with particle radii about them).

    Raises
    ------
    ValueError
        If the environment names a mode this module does not implement.
    """
    raw = os.environ.get(_MAC_GEOMETRY_ENV, default).strip().lower()
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
    """Depth of every node from the parent array by pointer doubling (root = 0).

    Parameters
    ----------
    parent : Array
        Parent index per node ``[nodes]``; negative marks a root.

    Returns
    -------
    Array
        Depth per node, index dtype, roots at 0.
    """
    num_nodes = int(parent.shape[0])
    parent_safe = jnp.where(parent >= 0, parent, as_index(0))
    is_root = parent < 0
    init_dist = jnp.where(is_root, as_index(0), as_index(1))
    init_shortcut = jnp.where(
        is_root, jnp.arange(num_nodes, dtype=INDEX_DTYPE), parent_safe
    )

    def _cond(state):
        _sc, _d, changed = state
        return changed

    def _body(state):
        sc, d, _changed = state
        new_d = d + d[sc]
        new_sc = sc[sc]
        return new_sc, new_d, jnp.any(new_sc != sc)

    _, depth, _ = lax.while_loop(
        _cond, _body, (init_shortcut, init_dist, jnp.bool_(True))
    )
    return depth


@jax.named_scope("fmm_com_radii")
def com_mac_geometry(
    tree: Any,
    positions_sorted: Array,
    centers: Array,
    *,
    leaf_cap: int,
    internal: str = "exact",
    num_levels: Optional[int] = None,
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
    num_levels : Optional[int]
        Static bound on the tree's level count (the upward sweep's
        ``static_num_levels``; a traced rebuild deeper than it trips the
        capacity guard). ``None`` = the padded 64. Static.

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
        raise ValueError(
            f"internal must be one of {_MAC_RADIUS_MODES}, got {internal!r}"
        )
    positions_sorted = jnp.asarray(positions_sorted)
    # one compiled program even when the caller is eager (the prepare): op by op,
    # the leaf pass materialised its (L, w, 3) gather and (L, w) tables
    radii = _com_radii(
        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        jnp.asarray(tree.left_child, dtype=INDEX_DTYPE),
        jnp.asarray(tree.right_child, dtype=INDEX_DTYPE),
        jnp.asarray(tree.parent, dtype=INDEX_DTYPE),
        positions_sorted,
        jnp.asarray(centers, dtype=positions_sorted.dtype),
        leaf_cap=int(leaf_cap),
        internal=str(internal),
        num_levels=None if num_levels is None else int(num_levels),
    )
    centers = jnp.asarray(centers, dtype=positions_sorted.dtype)
    num_nodes = int(radii.shape[0])
    half_extent = jnp.broadcast_to(radii[:, None], (num_nodes, 3))
    return TreeGeometry(centers, half_extent, radii, radii)


@partial(jax.jit, static_argnames=("leaf_cap", "internal", "num_levels"))
def _com_radii(
    node_ranges: Array,
    left_child: Array,
    right_child: Array,
    parent: Array,
    positions_sorted: Array,
    centers: Array,
    *,
    leaf_cap: int,
    internal: str,
    num_levels: Optional[int],
) -> Array:
    """The radii of :func:`com_mac_geometry`, one jitted program.

    Parameters
    ----------
    node_ranges : Array
        ``(nodes, 2)`` inclusive particle ranges.
    left_child : Array
        ``(internal,)`` left children.
    right_child : Array
        ``(internal,)`` right children.
    parent : Array
        ``(nodes,)`` parents (``-1`` at the root).
    positions_sorted : Array
        ``(n, 3)`` positions in tree order.
    centers : Array
        ``(nodes, 3)`` expansion centres, in the positions' dtype.
    leaf_cap : int
        Leaf capacity. Static.
    internal : str
        ``"exact"`` or ``"bound"``. Static.
    num_levels : Optional[int]
        Level-count bound. Static.

    Returns
    -------
    Array
        ``(nodes,)`` radii.
    """
    dtype = positions_sorted.dtype
    num_nodes = int(node_ranges.shape[0])
    num_internal = int(left_child.shape[0])
    n = int(positions_sorted.shape[0])
    w = max(1, int(leaf_cap))

    lane = jnp.arange(w, dtype=INDEX_DTYPE)
    leaf_ranges = node_ranges[num_internal:]

    def _leaf_max_about(node: Array) -> Array:
        # (L,): max over each leaf's particles of the distance to centers[node],
        # one reduction over (L, w) lanes read straight from the sorted positions
        # (gather, difference, norm and max fuse; nothing (L, w)-sized is kept)
        live = node >= 0
        c = centers[jnp.where(live, node, 0)]
        idx = leaf_ranges[:, 0][:, None] + lane[None, :]
        valid = idx <= leaf_ranges[:, 1][:, None]
        pts = positions_sorted[jnp.clip(idx, 0, max(n - 1, 0))]
        diff = pts - c[:, None, :]
        # max of the SQUARED distances, one square root per leaf: sqrt is monotone
        # and correctly rounded, so sqrt(max d^2) == max sqrt(d^2) to the bit
        d2 = jnp.sum(diff * diff, axis=-1)
        d2 = jnp.max(jnp.where(valid, d2, jnp.asarray(0.0, dtype)), axis=1)
        return jnp.where(live, jnp.sqrt(d2), jnp.asarray(0.0, dtype))

    # --- leaves: exact max distance about the centre over the leaf's particles
    leaf_ids = jnp.arange(num_internal, num_nodes, dtype=INDEX_DTYPE)
    r_leaf = _leaf_max_about(leaf_ids)
    radii = jnp.zeros((num_nodes,), dtype=dtype).at[num_internal:].set(r_leaf)

    if num_internal > 0 and internal == "exact":
        # Every internal node's particles are exactly the union of its descendant
        # leaves', so the max over (leaf, ancestor) pairs of the leaf's particle
        # distances about the ancestor's centre is the exact radius. One pass per
        # ancestor level k (the leaves' k-th ancestors, by one pointer step per
        # pass): a node's leaves are consecutive, so it is ONE run of the column
        # ``anc``; a segmented max carries the run max to the run's last leaf,
        # which alone writes it (the other lanes point past the radii and are
        # dropped -- no sentinel row collecting every lane's atomic, and no
        # scatter-max on the few top nodes, 44 ms per step at 16k leaves once).
        # Max is exact in any order: the radii are the (leaves x 64 levels)
        # table's this replaced, to the few ulp by which the table's 4-level
        # broadcast rounded the 3-term norm differently, at (L,)-sized memory per
        # pass and ``num_levels`` passes instead of 64. (Four levels per pass was
        # tried and lost: XLA did not fuse the (L, w, 4) norm into the reduction,
        # 35 against 26 ms per step at 8e6 and 0.2 GiB more in the prepare.)
        parent_safe = jnp.where(parent >= 0, parent, jnp.asarray(0, INDEX_DTYPE))
        drop = jnp.asarray(num_nodes, INDEX_DTYPE)

        def _seg(a, b):
            ka, va = a
            kb, vb = b
            return kb, jnp.where(ka == kb, jnp.maximum(va, vb), vb)

        def _level(_, carry):
            radii, anc = carry
            live = anc >= 0
            _, run_max = lax.associative_scan(_seg, (anc, _leaf_max_about(anc)))
            last = jnp.concatenate([anc[1:] != anc[:-1], jnp.ones((1,), bool)]) & live
            radii = radii.at[jnp.where(last, anc, drop)].max(run_max, mode="drop")
            up = jnp.where(live, parent_safe[jnp.where(live, anc, 0)], -1)
            up = jnp.where(live & (parent[jnp.where(live, anc, 0)] >= 0), up, -1)
            return radii, up.astype(INDEX_DTYPE)

        levels = int(_MAX_TREE_LEVELS) if num_levels is None else int(num_levels)
        # a leaf at depth D has D ancestors, so depth-bound - 1 passes cover all
        passes = max(1, min(levels, int(_MAX_TREE_LEVELS)) - 1)
        radii, _ = lax.fori_loop(0, passes, _level, (radii, parent[leaf_ids]))

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

    return radii


def resolve_walk_geometry(
    tree: Any,
    positions_sorted: Array,
    box_geometry: Optional[TreeGeometry],
    expansion_centers: Optional[Array],
    *,
    leaf_cap: int,
    geometry_factory: Optional[Any] = None,
    radius_scale: Optional[Array] = None,
    default_mode: str = "aabb",
    num_levels: Optional[int] = None,
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
    radius_scale : Optional[Array]
        Per-node multiplier already folded into ``box_geometry.radius`` by the
        caller, ``(nodes,)``. ``mac_type='dehnen_theta'`` folds its criterion
        into the radii before the walk (``_apply_per_node_effective_theta``);
        recomputing radii about the COMs would otherwise DROP that criterion
        silently, which is what it did until 2026-09-12. The same multiplier is
        applied to the COM radii, so the walk tests
        ``r_t/theta_t + r_s/theta_s <= d`` about the expansion centres.

    default_mode : str
        Geometry to use when the environment names none; see
        :func:`mac_geometry_mode`. Only the strict fused lane passes ``"com"``.
    num_levels : Optional[int]
        Static level-count bound for :func:`com_mac_geometry`.

    Returns
    -------
    tuple[Optional[TreeGeometry], Optional[Any]]
        ``(geometry, geometry_factory)`` to hand to the artifacts builder:
        unchanged in ``"aabb"`` mode; the COM geometry and ``None`` in ``"com"``.

    Raises
    ------
    RuntimeError
        ``"com"`` requested EXPLICITLY but the upward data carries no expansion
        centres. Defaulted requests fall back to the box geometry instead.
    ValueError
        If ``radius_scale`` does not have one entry per node, or the environment
        names a mode this module does not implement.
    """
    if mac_geometry_mode(default_mode) == "aabb":
        return box_geometry, geometry_factory
    if expansion_centers is None:
        # Quiet fallback, NOT an error. While "com" was opt-in, a lane without
        # expansion centres asking for it was a caller mistake worth raising on.
        # Since Phase 6 made it the DEFAULT, the same raise turned every such lane
        # into a hard failure (tests/integration/test_adaptive_order_runtime.py).
        # A lane with no COM centres cannot be inconsistent with them, so the box
        # geometry is the correct answer for it, not an error. An EXPLICIT request
        # still raises, because then the caller asked for something it cannot have.
        if os.environ.get(_MAC_GEOMETRY_ENV):
            raise RuntimeError(
                f"{_MAC_GEOMETRY_ENV}=com needs the upward sweep's expansion "
                "centres, and this lane's upward data has none."
            )
        return box_geometry, geometry_factory
    geometry = com_mac_geometry(
        tree,
        positions_sorted,
        expansion_centers,
        leaf_cap=int(leaf_cap),
        internal=mac_radius_mode(),
        num_levels=num_levels,
    )
    if radius_scale is not None:
        scale = jnp.asarray(radius_scale, geometry.radius.dtype)
        if scale.shape != geometry.radius.shape:
            raise ValueError(
                f"radius_scale has shape {scale.shape}, expected "
                f"{geometry.radius.shape} (one factor per node)"
            )
        geometry = geometry._replace(radius=geometry.radius * scale)
    return geometry, None
