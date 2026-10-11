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

from typing import NamedTuple

from jaxtyping import Array, Float
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
    packed : Float[Array, 'nodes sh']
        ``(nodes, sh_size(order))`` packed real spherical-harmonic coefficients --
        what the sweeps consume. (The always-``None`` ``moments``,
        ``component_matrix`` and ``source_motion_packed`` of the removed
        Cartesian and complex bases went in the 2026-10 cleanup, Z-b.)
    """

    order: int
    centers: Float[Array, "nodes 3"]
    packed: Float[Array, "nodes sh"]


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
