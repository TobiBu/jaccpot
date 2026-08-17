"""Shared types for the kernel library: basis tag, output bundles, diag gate.

The five names the other kernel modules have in common, in a leaf module so the
seam split stays acyclic: ``core`` re-exports every submodule, and both
``_evaluate`` and ``_l2l`` need something from here, so these cannot live in
``core`` without making it import what imports it.

``NearfieldInteropData`` is the near-field hand-off contract -- the arrays the
evaluation path passes to ``nearfield/`` -- and is deliberately shape-encoded
rather than validated, so read the field comments before adding to it.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Optional

from jaxtyping import Array

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
    """Explicit shared leaf/node view used to interoperate with nearfield code.

    Built by ``_evaluate.py`` along one of two routes -- an octree route and the
    radix/default route -- which is what the four optional fields are about:
    the octree route populates all of them, the radix route leaves the first
    three ``None``. Consumers must branch on ``None`` rather than assume presence.

    The neighbour lists are CSR: leaf ``i``'s neighbour node ids are
    ``neighbors[offsets[i] : offsets[i] + counts[i]]``.

    Attributes
    ----------
    leaf_nodes : Array
        ``[num_leaves]`` node id of each leaf, i.e. leaf position -> node id.
    node_ranges : Array
        Per-node ``[start, end)`` particle ranges, indexed by node id.
    offsets : Array
        CSR row offsets into ``neighbors``, one longer than the leaf count.
    neighbors : Array
        Flat CSR neighbour **node ids** -- node ids, not leaf positions; use
        ``neighbor_leaf_positions`` when you need the latter.
    counts : Array
        ``[num_leaves]`` neighbours per leaf. Measured to equal
        ``diff(offsets)`` exactly -- checked on both a uniform and a strongly
        clustered configuration, no padded rows in either -- so it is redundant
        today. Use it anyway: it is what the derivation code bounds validity by,
        and the redundancy is an observation, not a stated contract.
    particle_order_node_ranges : Array
        Node ranges in particle order. In **both** current constructors this is
        the same array as ``node_ranges``; the field is kept separate so a future
        route can diverge, so do not rely on the identity.
    particle_order_leaf_indices : Array
        Leaf index of each particle, in particle order.
    particle_order_to_native_leaf : Array
        Map from particle-order leaf position to the tree's native leaf
        numbering. Identity on the radix route when the neighbour list does not
        supply one.
    leaf_particle_indices : Optional[Array]
        ``[num_leaves, W]`` explicit per-leaf membership, or ``None`` on the
        radix route where membership is implicit in the contiguous
        ``node_ranges``.
    leaf_particle_mask : Optional[Array]
        ``[num_leaves, W]`` occupancy matching ``leaf_particle_indices``; ``None``
        alongside it.
    particle_to_leaf_position : Optional[Array]
        Inverse of ``leaf_particle_indices``, or ``None`` alongside it.
    neighbor_leaf_positions : Optional[Array]
        ``[num_leaves, max_neighbors]`` neighbours as leaf positions, padded with
        ``-1``. Optional by type but populated on both routes -- derived from
        ``neighbors`` when the source list does not carry it. The ``-1`` padding
        must be masked, not clipped: it is a sentinel, and as a Python-negative
        index it would silently wrap to the last leaf.
    """

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
