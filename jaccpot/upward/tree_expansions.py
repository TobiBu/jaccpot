"""The upward pass's containers.

What is left after the Cartesian upward sweep (``compute_node_multipoles``,
``prepare_upward_sweep`` and its M2M aggregation) was removed with the Cartesian
basis in the 2026-10 cleanup (docs/cleanup_2026-10.md, K), and the complex sweep
with its expansion-centre mode check (``_resolve_center_mode``) in C: the two
containers the real sweep fills. The sweep is
:mod:`jaccpot.upward.real_tree_expansions` (COM centres only), whose docstring
carries the note on why the COM expansion centres are differentiated through.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

from jaxtyping import Array, Float, Inexact
from yggdrax.geometry import TreeGeometry
from yggdrax.tree_moments import TreeMassMoments


class NodeMultipoleData(NamedTuple):
    """Packed multipole expansions and their metadata.

    Attributes
    ----------
    order : int
        Expansion order ``p``.
    centers : Float[Array, 'nodes 3']
        ``(nodes, 3)`` expansion centres.
    moments : Optional[Any]
        Always ``None``. It held the raw Cartesian moments, removed with the
        Cartesian basis in the 2026-10 cleanup (K); kept so the pytree structure
        is unchanged.
    packed : Inexact[Array, 'nodes sh']
        ``(nodes, sh_size(order))`` packed spherical-harmonic coefficients, real
        or complex by basis -- what the sweeps actually consume.
    component_matrix : Optional[Any]
        Always ``None``, for the same reason as ``moments``.
    source_motion_packed : Optional[Inexact[Array, 'nodes sh']]
        Time-differentiated multipoles, present only when a derivative path asked
        for them.
    """

    order: int
    centers: Float[Array, "nodes 3"]
    moments: Optional[Any]
    packed: Inexact[Array, "nodes sh"]
    component_matrix: Optional[Any]
    source_motion_packed: Optional[Inexact[Array, "nodes sh"]] = None


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


__all__ = [
    "NodeMultipoleData",
    "TreeUpwardData",
]
