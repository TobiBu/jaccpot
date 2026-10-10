"""The upward pass's containers, and the expansion-centre mode check.

What is left after the Cartesian upward sweep (``compute_node_multipoles``,
``prepare_upward_sweep`` and its M2M aggregation) was removed with the Cartesian
basis in the 2026-10 cleanup (docs/cleanup_2026-10.md, K): the two containers the
spherical-harmonic sweeps fill, and :func:`_resolve_center_mode`, which the
complex upward sweep shares. The real sweep is
:mod:`jaccpot.upward.real_tree_expansions`, whose docstring now carries the note
on why the COM expansion centres are differentiated through.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

from jaxtyping import Array, Float, Inexact
from yggdrax.geometry import TreeGeometry
from yggdrax.tree_moments import TreeMassMoments

#: Expansion-centre modes the upward sweeps accept. ``"explicit"`` is the test seam
#: (caller-supplied centres). The AABB / geometric centres went with the grouped far
#: field that needed them in the 2026-10 cleanup (docs/cleanup_2026-10.md, X3).
_CENTER_MODES = ("com", "explicit")
_REMOVED_CENTER_MODES = ("aabb", "geometric")


def _resolve_center_mode(center_mode: str) -> str:
    """Normalise an expansion-centre mode and refuse the removed and unknown ones.

    Shared by every upward sweep that takes a ``center_mode``, so the removal
    message is the same wherever an old caller reaches it.

    Parameters
    ----------
    center_mode : str
        ``"com"`` or ``"explicit"``, in any case.

    Returns
    -------
    str
        The lower-cased mode.

    Raises
    ------
    ValueError
        If the mode is ``"aabb"`` / ``"geometric"`` (removed in the 2026-10
        cleanup, X3) or anything else outside :data:`_CENTER_MODES`.
    """
    mode = str(center_mode).strip().lower()
    if mode in _REMOVED_CENTER_MODES:
        raise ValueError(
            f"center_mode={center_mode!r} is no longer available: the AABB "
            "(geometric) expansion centres were removed in the 2026-10 cleanup "
            "(X3); see docs/cleanup_2026-10.md. Use 'com' (the default) or "
            "'explicit'."
        )
    if mode not in _CENTER_MODES:
        raise ValueError(f"Unknown center_mode '{center_mode}'")
    return mode


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
