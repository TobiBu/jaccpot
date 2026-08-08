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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

__all__ = [
    "MutualTopology",
    "build_mutual_topology",
    "build_mutual_topology_from_tree",
]


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
        """Return a small dict of size counters, for tests and benchmarks."""
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
    """Pad each leaf's particle span into a dense ``(num_leaves, S)`` block."""
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
) -> MutualTopology:
    """Build a :class:`MutualTopology` from a prebuilt yggdrax tree.

    ``positions_sorted``/``masses_sorted`` are used only to fix the *discrete*
    decisions (node centres and radii feeding the MAC). The numeric pipeline
    recomputes centres from live inputs, so these need not be the positions the
    force is later evaluated at -- they are the topology's reference
    configuration, exactly as in
    :meth:`jaccpot.FastMultipoleMethod.differentiable_accelerations`.
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

    far_a, far_b, near_a, near_b = _dual_traverse(
        centers, radii, num_internal, left_child, right_child, float(theta), root
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
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    theta: float = 0.6,
    order: int = 4,
    leaf_size: int = 16,
    solver: Optional[Any] = None,
) -> Tuple[MutualTopology, Any]:
    """Build a tree with jaccpot's production builder, then a mutual topology.

    Returns the topology together with the prepared state it came from, so the
    caller can keep the tree alive (the topology indexes into its Morton order).
    The tree build itself is reused rather than reimplemented: Morton ordering,
    local refinement and leaf sizing are exactly jaccpot's.
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
        order=int(order),
    )
    return topology, state
