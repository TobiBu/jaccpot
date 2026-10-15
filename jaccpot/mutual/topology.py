"""Frozen symmetric topology for the mutual FMM.

Mutual accumulation is only well defined if the near/far partition is
**symmetric**: if ``(A, B)`` is well separated then so is ``(B, A)``, and the
pair must be visited exactly once. Jaccpot's production traversal emits both
directions (it is target-centric, so every pair appears twice), which is correct
for a gather but double-counts under ``+f``/``-f`` accumulation. This module
therefore runs its own **dual-tree** traversal that visits each unordered pair
once and emits it in a canonical ``a < b`` order.

The traversal is a level-synchronous BFS over *pairs* of nodes, vectorised with
NumPy on the host. That placement is deliberate, not a fallback: the discrete
topology (which pairs are near, which are far, which particles live in which
leaf) is exactly what
:meth:`jaccpot.FastMultipoleMethod.differentiable_accelerations` freezes and
severs from the gradient, and what nornax's ``stop_gradient``-ed rung schedule
freezes on its side. Building it on the host makes that constant-ness structural
-- there is no traced control flow to accidentally differentiate through -- and
keeps every device-side kernel a static-shape, statically-bounded computation.

Cost is ``O(N)``: the number of pairs visited is bounded by a constant per node
for a fixed ``theta`` (the standard FMM interaction-list bound), and the
per-level NumPy vectorisation means the Python interpreter runs once per tree
level, not once per pair.

Multipole acceptance criterion
------------------------------
A pair is accepted as well separated when

    ``theta * |c_B - c_A|  >  R_A + R_B``

with ``c`` the node centre of mass and ``R`` the node radius (the largest
distance from the centre of mass to a particle in the node). This is the
*mutual* MAC: it is symmetric in ``A`` and ``B`` by construction, which is what
lets the same acceptance decision serve both directions of the interaction. A
target-centric MAC (``R_source / d < theta``) is not symmetric and would accept a
pair in one direction only.

Drift margin
------------
A block-step base step evaluates the force at all ``2**k_max + 1`` boundaries
against lists built once, at its start, while the particles drift. The far
field recomputes its centres from the live positions, but the accept decision
is not revisited, so a pair accepted at the build can violate the MAC by a late
boundary -- and its expansion diverges once a particle of one cell drifts into
the other's convergence sphere. Measured on a 1e6 prefix of the 25M disc at its
production step (S0, ``docs/production_2026-10.md``): frozen-list rel-L2 1.3e-2
at ``k_max = 3`` against 7.6e-4 with fresh lists, 6.5 at ``k_max = 5``.

If no particle moves further than ``m`` from where the lists were built, a
node's centre of mass moves by at most ``m`` (it is a mass-weighted mean of the
displacements) and its radius grows by at most ``2 m``. So building with

    ``theta * |c_B - c_A| > (R_A + (2 + theta) m) + (R_B + (2 + theta) m)``

keeps the live configuration inside the build MAC,
``theta * |c_B' - c_A'| >= theta (|c_B - c_A| - 2 m) > R_A' + R_B'``. The pad
``(2 + theta) m`` is the supremum, not a loose bound: a light particle moving
outward while the rest of its cell moves toward the partner approaches it.
:func:`drift_radius_pad` returns it and both builders add it to every node
radius before the walk. A uniform shift leaves the walk's larger-node split
comparisons as they were (up to rounding of exact ties), so only acceptance
moves, and only towards stricter: the lists are still a symmetric canonical
partition, and momentum is still exact. A compact kernel's separation floor
tests the same inflated radii, which is conservative -- a particle-particle
distance shrinks by at most ``2 m``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from jax.typing import ArrayLike

__all__ = [
    "MutualTopology",
    "build_mutual_topology",
    "build_mutual_topology_from_tree",
    "drift_radius_pad",
]


def drift_radius_pad(theta: float, drift_margin: Optional[float]) -> float:
    """Return the radius inflation that keeps the mutual MAC valid under drift.

    See the module docstring's "Drift margin" for the derivation: with every
    particle within ``drift_margin`` of its build position, a pair accepted at
    radii ``R + (2 + theta) * drift_margin`` still satisfies the un-inflated MAC
    at the drifted positions.

    Parameters
    ----------
    theta : float
        Opening angle of the mutual MAC.
    drift_margin : Optional[float]
        Largest displacement any particle may make before the lists are rebuilt,
        in length units. ``None`` means no margin.

    Returns
    -------
    float
        ``(2 + theta) * drift_margin``, or ``0.0`` for ``None``.

    Raises
    ------
    ValueError
        If ``drift_margin`` is not a finite positive length.
    """
    if drift_margin is None:
        return 0.0
    margin = float(drift_margin)
    if not (math.isfinite(margin) and margin > 0.0):
        raise ValueError(
            "drift_margin must be a finite positive length or None; got "
            f"{drift_margin!r}"
        )
    return (2.0 + float(theta)) * margin


@dataclass(frozen=True)
class MutualTopology:
    """Frozen discrete structure driving one mutual FMM evaluation.

    Every array here is a host-side NumPy constant. They are converted to device
    arrays once, when a :class:`~jaccpot.mutual.force.MutualFMMState` is built,
    and are captured as compile-time constants by the numeric kernels.

    Attributes
    ----------
    num_particles : int
        Number of particles the topology was built for.
    num_nodes : int
        Total tree nodes. Ids ``[0, num_internal)`` are internal and
        ``[num_internal, num_nodes)`` are leaves -- the yggdrax radix convention.
    num_internal : int
        Number of internal nodes, i.e. the id at which the leaves start.
    max_leaf_size : int
        Widest leaf, and so the padded block width ``S``.
    order : int
        Multipole expansion order the far-field kernels will run at.
    theta : float
        Multipole acceptance parameter the traversal was built with.
    left_child : np.ndarray
        ``(num_internal,)`` global left-child node ids, ``-1`` where absent.
    right_child : np.ndarray
        ``(num_internal,)`` global right-child node ids, ``-1`` where absent.
    level_nodes : Tuple[np.ndarray, ...]
        Per-depth node groupings driving the M2M and L2L cascades as *static*
        Python loops -- one batched call per tree level, no traced loop bound.
        ``level_nodes[d]`` holds the nodes at depth ``d + 1``.
    parent_of_level_nodes : Tuple[np.ndarray, ...]
        The parents of ``level_nodes[d]``, so the pair is exactly that level's
        translation edge set.
    leaf_nodes : np.ndarray
        ``(num_leaves,)`` leaf node ids in ascending order.
    leaf_particles : np.ndarray
        ``(num_leaves, max_leaf_size)`` padded particle indices into the
        Morton-sorted order. Padding slots point at particle 0 and are masked.
    leaf_particle_valid : np.ndarray
        ``(num_leaves, max_leaf_size)`` mask marking the real slots.
    far_a : np.ndarray
        ``(num_far,)`` first node of each canonical well-separated pair.
    far_b : np.ndarray
        ``(num_far,)`` second node, with ``far_a < far_b``.
    near_a : np.ndarray
        ``(num_near,)`` first leaf of each canonical near leaf pair.
    near_b : np.ndarray
        ``(num_near,)`` second leaf, with ``near_a < near_b``. Leaf self-pairs are
        held separately in ``leaf_nodes``.
    node_particle_ranges : np.ndarray
        ``(num_nodes, 2)`` inclusive particle span per node, used to assign a rung
        to a cell.
    inverse_permutation : np.ndarray
        Morton-sorted -> original particle order.
    forward_permutation : np.ndarray
        Original -> Morton-sorted particle order.
    """

    num_particles: int
    num_nodes: int
    num_internal: int
    max_leaf_size: int
    order: int
    theta: float

    left_child: np.ndarray
    right_child: np.ndarray
    level_nodes: Tuple[np.ndarray, ...]
    parent_of_level_nodes: Tuple[np.ndarray, ...]

    leaf_nodes: np.ndarray
    leaf_particles: np.ndarray
    leaf_particle_valid: np.ndarray

    far_a: np.ndarray
    far_b: np.ndarray
    near_a: np.ndarray
    near_b: np.ndarray

    node_particle_ranges: np.ndarray
    inverse_permutation: np.ndarray
    forward_permutation: np.ndarray

    @property
    def num_leaves(self: "MutualTopology") -> int:
        """Number of leaf nodes."""
        return int(self.leaf_nodes.shape[0])

    @property
    def num_far_pairs(self: "MutualTopology") -> int:
        """Number of canonical well-separated node pairs."""
        return int(self.far_a.shape[0])

    @property
    def num_near_pairs(self: "MutualTopology") -> int:
        """Number of canonical near leaf pairs (excluding leaf self-pairs)."""
        return int(self.near_a.shape[0])

    def summary(self: "MutualTopology") -> dict[str, Any]:
        """Return a small dict of size counters, for tests and benchmarks.

        Returns
        -------
        dict[str, Any]
            Size counters only -- no arrays, so it is cheap to log and safe to
            compare across runs.
        """
        return {
            "num_particles": self.num_particles,
            "num_nodes": self.num_nodes,
            "num_leaves": self.num_leaves,
            "max_leaf_size": self.max_leaf_size,
            "num_far_pairs": self.num_far_pairs,
            "num_near_pairs": self.num_near_pairs,
            "num_levels": len(self.level_nodes),
            "theta": self.theta,
            "order": self.order,
        }


def _node_depths(
    parent: np.ndarray, left_child: np.ndarray, right_child: np.ndarray, root: int
) -> np.ndarray:
    """Return the depth of every node, computed by a breadth-first descent.

    Descending from the root rather than walking ``parent`` upward keeps this
    linear and, more importantly, leaves unreachable nodes marked ``-1`` so they
    can be dropped from the cascades. Radix-tree builders routinely leave dead
    node slots behind (a padded node array, a leaf that lost its particles to
    refinement); folding one of those into an M2M level would translate a stale
    expansion into a live parent.

    Parameters
    ----------
    parent : np.ndarray
        Parent index per node.
    left_child : np.ndarray
        Left child per internal node.
    right_child : np.ndarray
        Right child per internal node.
    root : int
        Index of the root node.

    Returns
    -------
    np.ndarray
        Depth per node, with ``-1`` marking every node unreachable from the
        root. Callers must treat ``-1`` as "drop", not as depth zero.
    """
    num_nodes = int(parent.shape[0])
    depth = np.full(num_nodes, -1, dtype=np.int64)
    depth[root] = 0
    frontier = np.array([root], dtype=np.int64)
    num_internal = int(left_child.shape[0])
    while frontier.size:
        internal = frontier[frontier < num_internal]
        if internal.size == 0:
            break
        children = np.concatenate([left_child[internal], right_child[internal]])
        parents = np.concatenate([internal, internal])
        keep = children >= 0
        children, parents = children[keep], parents[keep]
        # A node already stamped has been reached by a shorter path; in a tree
        # that cannot happen, but a malformed child array must not loop forever.
        fresh = depth[children] < 0
        children, parents = children[fresh], parents[fresh]
        if children.size == 0:
            break
        depth[children] = depth[parents] + 1
        frontier = children
    return depth


def _group_levels(
    depth: np.ndarray, parent: np.ndarray, root: int
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    """Group reachable non-root nodes by depth, together with their parents.

    ``level_nodes[d]`` are the nodes at depth ``d + 1``. Returned shallowest
    first, which is the L2L (push-down) order; the M2M (pull-up) cascade walks
    the same tuple in reverse.

    Parameters
    ----------
    depth : np.ndarray
        Per-node depth from :func:`_node_depths`; ``-1`` entries are skipped.
    parent : np.ndarray
        Parent index per node.
    root : int
        Index of the root node, which is excluded from the levels.

    Returns
    -------
    Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]
        ``(level_nodes, level_parents)``, aligned entry for entry, shallowest
        level first.
    """
    max_depth = int(depth.max())
    level_nodes: list[np.ndarray] = []
    level_parents: list[np.ndarray] = []
    for d in range(1, max_depth + 1):
        nodes = np.flatnonzero(depth == d).astype(np.int64)
        if nodes.size == 0:
            continue
        level_nodes.append(nodes)
        level_parents.append(parent[nodes].astype(np.int64))
    del root
    return tuple(level_nodes), tuple(level_parents)


def _dual_traverse(
    centers: np.ndarray,
    radii: np.ndarray,
    num_internal: int,
    left_child: np.ndarray,
    right_child: np.ndarray,
    theta: float,
    root: int,
    separation_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric dual-tree walk: canonical far node pairs and near leaf pairs.

    Maintains a frontier of unordered node pairs ``a <= b``, starting from
    ``(root, root)``. Each round classifies the whole frontier at once:

    * **accepted** (well separated, ``a != b``) -> the far list;
    * **both leaves** -> the near list (``a == b`` self-pairs are dropped here;
      they are recovered from ``leaf_nodes``, which already enumerates them);
    * otherwise **split** the node with the larger radius, or both when
      ``a == b``.

    Splitting the larger node is what keeps the two boxes of a pair comparable in
    size, which is the standard dual-traversal heuristic and bounds the number of
    pairs per node. Because a split always replaces an internal node with its
    children, every pair strictly descends and the walk terminates.

    Parameters
    ----------
    centers : np.ndarray
        Per-node centre used by the MAC.
    radii : np.ndarray
        Per-node radius used by the MAC.
    num_internal : int
        Number of internal nodes; a node index at or above this is a leaf.
    left_child : np.ndarray
        Left child per internal node.
    right_child : np.ndarray
        Right child per internal node.
    theta : float
        Opening angle of the mutual MAC, ``theta * |c_b - c_a| > r_a + r_b``.
    root : int
        Index of the root node; the walk starts from ``(root, root)``.
    separation_floor : float
        Refuse a far pair unless also ``|c_b - c_a| - r_a - r_b >=
        separation_floor``; ``0`` (default) adds no test. A compact softening
        kernel's support makes the unsoftened far field exact.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(far_a, far_b, near_a, near_b)``. Each pair appears **once**, in
        canonical ``a <= b`` order -- the mutual sweep supplies both directions
        from a single entry, so duplicating them here would double-count the
        force.
    """
    far_a: list[np.ndarray] = []
    far_b: list[np.ndarray] = []
    near_a: list[np.ndarray] = []
    near_b: list[np.ndarray] = []

    cur_a = np.array([root], dtype=np.int64)
    cur_b = np.array([root], dtype=np.int64)
    theta = float(theta)

    def _is_leaf(nodes: np.ndarray) -> np.ndarray:
        return nodes >= num_internal

    while cur_a.size:
        a, b = cur_a, cur_b
        delta = centers[b] - centers[a]
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        # Mutual MAC. `a == b` can never be accepted: a node is not well
        # separated from itself, and admitting it would drop the whole subtree's
        # internal interactions.
        accepted = (a != b) & (theta * dist > radii[a] + radii[b])
        if separation_floor > 0.0:
            accepted &= dist - radii[a] - radii[b] >= separation_floor
        if accepted.any():
            far_a.append(a[accepted])
            far_b.append(b[accepted])

        rest = ~accepted
        a, b = a[rest], b[rest]
        if a.size == 0:
            break

        leaf_pair = _is_leaf(a) & _is_leaf(b)
        if leaf_pair.any():
            cross = leaf_pair & (a != b)
            near_a.append(a[cross])
            near_b.append(b[cross])

        split = ~leaf_pair
        a, b = a[split], b[split]
        if a.size == 0:
            break

        diagonal = a == b
        next_a: list[np.ndarray] = []
        next_b: list[np.ndarray] = []

        # Self-pair of an internal node: its subtree's own interactions are
        # (l,l), (l,r) and (r,r). Dropping (l,r) here would lose every
        # cross-child interaction in the subtree.
        if diagonal.any():
            nodes = a[diagonal]
            left = left_child[nodes]
            right = right_child[nodes]
            both = (left >= 0) & (right >= 0)
            lo = np.minimum(left, right)
            hi = np.maximum(left, right)
            next_a.append(left[left >= 0])
            next_b.append(left[left >= 0])
            next_a.append(right[right >= 0])
            next_b.append(right[right >= 0])
            next_a.append(lo[both])
            next_b.append(hi[both])

        off = ~diagonal
        if off.any():
            oa, ob = a[off], b[off]
            # Split whichever box is larger; if the larger one is a leaf, split
            # the other (exactly one of them can be a leaf here, since
            # `leaf_pair` already removed the leaf-leaf case).
            a_leaf, b_leaf = _is_leaf(oa), _is_leaf(ob)
            split_a = (~a_leaf) & (b_leaf | (radii[oa] >= radii[ob]))
            for parent_side, other_side, mask in (
                (oa, ob, split_a),
                (ob, oa, ~split_a),
            ):
                if not mask.any():
                    continue
                p_nodes = parent_side[mask]
                o_nodes = other_side[mask]
                for child in (left_child[p_nodes], right_child[p_nodes]):
                    live = child >= 0
                    c, o = child[live], o_nodes[live]
                    next_a.append(np.minimum(c, o))
                    next_b.append(np.maximum(c, o))

        if not next_a:
            break
        cur_a = np.concatenate(next_a)
        cur_b = np.concatenate(next_b)

    def _cat(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(parts)

    return _cat(far_a), _cat(far_b), _cat(near_a), _cat(near_b)


def _leaf_blocks(
    leaf_nodes: np.ndarray, node_ranges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pad each leaf's particle span into a dense ``(num_leaves, S)`` block.
    ``S`` is the widest leaf's occupancy, so every leaf pays the widest one's
    width. That is what makes the near-field kernels a fixed shape.

    Parameters
    ----------
    leaf_nodes : np.ndarray
        Node index of each leaf.
    node_ranges : np.ndarray
        Per-node ``[start, stop)`` span into the Morton-sorted particle order.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        ``(particle_indices, valid_mask, S)``, the first two of shape
        ``(num_leaves, S)``.
    """
    starts = node_ranges[leaf_nodes, 0].astype(np.int64)
    ends = node_ranges[leaf_nodes, 1].astype(np.int64)
    counts = np.maximum(ends - starts + 1, 0)
    max_leaf_size = int(counts.max()) if counts.size else 1
    max_leaf_size = max(max_leaf_size, 1)
    offsets = np.arange(max_leaf_size, dtype=np.int64)[None, :]
    valid = offsets < counts[:, None]
    particles = np.where(valid, starts[:, None] + offsets, 0)
    return particles.astype(np.int64), valid, max_leaf_size


def build_mutual_topology_from_tree(
    tree: Any,
    positions_sorted: np.ndarray,
    masses_sorted: np.ndarray,
    *,
    theta: float,
    order: int,
    separation_floor: float = 0.0,
    drift_margin: Optional[float] = None,
) -> MutualTopology:
    """Build a :class:`MutualTopology` from a prebuilt yggdrax tree.

    ``positions_sorted``/``masses_sorted`` are used only to fix the *discrete*
    decisions (node centres and radii feeding the MAC). The numeric pipeline
    recomputes centres from live inputs, so these need not be the positions the
    force is later evaluated at -- they are the topology's reference
    configuration, exactly as in
    :meth:`jaccpot.FastMultipoleMethod.differentiable_accelerations`.

    Parameters
    ----------
    tree : Any
        Prebuilt yggdrax tree. Typed loosely so this module does not depend on
        a particular tree family.
    positions_sorted : np.ndarray
        Positions in the tree's Morton order, used only for the discrete
        decisions.
    masses_sorted : np.ndarray
        Masses in the same order, same purpose.
    theta : float
        Opening angle of the mutual MAC.
    order : int
        Expansion order recorded on the topology.

    separation_floor : float
        Minimum gap of an accepted far pair (length units); ``0`` for none.
    drift_margin : Optional[float]
        Build lists that stay valid while no particle moves further than this
        from ``positions_sorted``: every node radius is inflated by
        :func:`drift_radius_pad` before the walk. ``None`` (default) builds the
        plain MAC.

    Returns
    -------
    MutualTopology
        A frozen host-side topology.
    """
    positions = np.asarray(positions_sorted, dtype=np.float64)
    masses = np.asarray(masses_sorted, dtype=np.float64)
    parent = np.asarray(tree.parent, dtype=np.int64)
    left_child = np.asarray(tree.left_child, dtype=np.int64)
    right_child = np.asarray(tree.right_child, dtype=np.int64)
    node_ranges = np.asarray(tree.node_ranges, dtype=np.int64)

    num_nodes = int(parent.shape[0])
    num_internal = int(left_child.shape[0])
    num_particles = int(positions.shape[0])
    root = int(np.flatnonzero(parent < 0)[0]) if (parent < 0).any() else 0

    depth = _node_depths(parent, left_child, right_child, root)
    reachable = depth >= 0
    level_nodes, level_parents = _group_levels(depth, parent, root)

    # Node centre of mass and radius, from the reference configuration.
    starts = node_ranges[:, 0]
    ends = node_ranges[:, 1]
    centers = np.zeros((num_nodes, 3), dtype=np.float64)
    radii = np.zeros((num_nodes,), dtype=np.float64)
    weighted = np.concatenate(
        [np.zeros((1, 3)), np.cumsum(masses[:, None] * positions, axis=0)], axis=0
    )
    mass_cum = np.concatenate([np.zeros((1,)), np.cumsum(masses)], axis=0)
    for node in range(num_nodes):
        if not reachable[node]:
            continue
        s, e = int(starts[node]), int(ends[node])
        if e < s:
            continue
        total = mass_cum[e + 1] - mass_cum[s]
        if total <= 0.0:
            centers[node] = positions[s : e + 1].mean(axis=0)
        else:
            centers[node] = (weighted[e + 1] - weighted[s]) / total
        offsets = positions[s : e + 1] - centers[node]
        radii[node] = float(np.sqrt(np.einsum("ij,ij->i", offsets, offsets)).max())
    pad = drift_radius_pad(theta, drift_margin)
    if pad > 0.0:
        radii = radii + pad

    far_a, far_b, near_a, near_b = _dual_traverse(
        centers,
        radii,
        num_internal,
        left_child,
        right_child,
        float(theta),
        root,
        separation_floor=float(separation_floor),
    )

    leaf_nodes = np.flatnonzero(
        reachable & (np.arange(num_nodes) >= num_internal)
    ).astype(np.int64)
    leaf_particles, leaf_valid, max_leaf_size = _leaf_blocks(leaf_nodes, node_ranges)

    inverse_permutation = np.asarray(tree.inverse_permutation, dtype=np.int64)
    forward_permutation = np.argsort(inverse_permutation, kind="stable").astype(
        np.int64
    )

    return MutualTopology(
        num_particles=num_particles,
        num_nodes=num_nodes,
        num_internal=num_internal,
        max_leaf_size=int(max_leaf_size),
        order=int(order),
        theta=float(theta),
        left_child=left_child,
        right_child=right_child,
        level_nodes=level_nodes,
        parent_of_level_nodes=level_parents,
        leaf_nodes=leaf_nodes,
        leaf_particles=leaf_particles,
        leaf_particle_valid=leaf_valid,
        far_a=far_a,
        far_b=far_b,
        near_a=near_a,
        near_b=near_b,
        node_particle_ranges=node_ranges,
        inverse_permutation=inverse_permutation,
        forward_permutation=forward_permutation,
    )


def build_mutual_topology(
    positions: ArrayLike,
    masses: ArrayLike,
    *,
    theta: float = 0.6,
    order: int = 4,
    leaf_size: int = 16,
    solver: Optional[Any] = None,
    separation_floor: float = 0.0,
    drift_margin: Optional[float] = None,
) -> Tuple[MutualTopology, Any]:
    """Build a tree with jaccpot's production builder, then a mutual topology.

    Returns the topology together with the prepared state it came from, so the
    caller can keep the tree alive (the topology indexes into its Morton order).
    The tree build itself is reused rather than reimplemented: Morton ordering,
    local refinement and leaf sizing are exactly jaccpot's.

    Parameters
    ----------
    positions : ArrayLike
        ``(n, 3)`` particle positions. Forwarded unchanged to
        :meth:`jaccpot.FastMultipoleMethod.prepare_state`, whose annotation is
        a JAX array, so NumPy and JAX are both accepted here -- annotating this
        ``np.ndarray`` while forwarding to a JAX-annotated call left no input
        that satisfied both under ``JACCPOT_RUNTIME_TYPECHECK=1``.
    masses : ArrayLike
        ``(n,)`` particle masses. Same forwarding as ``positions``.
    theta : float
        Opening angle of the mutual MAC.
    order : int
        Expansion order recorded on the topology.
    leaf_size : int
        Target particles per leaf for the tree build.
    solver : Optional[Any]
        Reuse an existing solver instead of constructing one.

    separation_floor : float
        Minimum gap of an accepted far pair (length units); ``0`` for none.
    drift_margin : Optional[float]
        Drift the lists must stay valid under; see
        :func:`build_mutual_topology_from_tree`. ``None`` for none.

    Returns
    -------
    Tuple[MutualTopology, Any]
        ``(topology, prepared_state)``. The state must be kept alive: the
        topology indexes into its Morton order, so dropping it invalidates every
        index array.
    """
    from jaccpot import FastMultipoleMethod

    if solver is None:
        solver = FastMultipoleMethod(preset="balanced", basis="real")
    state = solver.prepare_state(
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(order),
        theta=float(theta),
    )
    topology = build_mutual_topology_from_tree(
        state.tree,
        np.asarray(state.positions_sorted),
        np.asarray(state.masses_sorted),
        theta=float(theta),
        separation_floor=float(separation_floor),
        order=int(order),
        drift_margin=drift_margin,
    )
    return topology, state
