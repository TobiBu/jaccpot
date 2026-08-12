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
