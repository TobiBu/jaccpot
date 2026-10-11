"""The downward pass's containers: local expansions and the sweep's bundle.

The Cartesian downward sweep that lived here (derivative stencils, the dense and
CSR M2L accumulators, L2L propagation and the module-level ``prepare`` / ``run``
entry points) was removed with the Cartesian basis in the 2026-10 cleanup
(docs/cleanup_2026-10.md, K). The spherical-harmonic sweeps build these two
containers from ``runtime/kernels/`` (``_downward_prep.py``, ``_l2l.py``) and the
distributed lane.
"""

from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Array
from yggdrax.interactions import NodeInteractionList


class LocalExpansionData(NamedTuple):
    """Local expansion coefficients and metadata.

    Attributes
    ----------
    order : int
        Expansion order ``p`` these coefficients were built at.
    centers : Array
        Expansion centre per node, shape ``(num_nodes, 3)``.
    coefficients : Array
        Packed local coefficients per node, shape ``(num_nodes, ncoeff(p))``.
    """

    order: int
    centers: Array
    coefficients: Array


class TreeDownwardData(NamedTuple):
    """Bundle for interaction lists and resulting local expansions.

    Attributes
    ----------
    interactions : NodeInteractionList
        The accepted far-field node pairs this sweep consumed.
    locals : LocalExpansionData
        The local expansions the sweep produced. (The always-``None``
        ``source_motion_locals`` went in the 2026-10 cleanup, Z-b: the time
        derivatives build their own source-motion locals per order.)
    """

    interactions: NodeInteractionList
    locals: LocalExpansionData


__all__ = [
    "LocalExpansionData",
    "TreeDownwardData",
]
