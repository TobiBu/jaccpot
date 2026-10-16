"""Galaxy-scale production settings of the fused lane: what ``preset="galaxy"`` runs.

``FastMultipoleMethod(preset="galaxy")`` is the configuration every production row
of the fused lane was measured with (``bench/fused_memory_budget.py``'s defaults:
the L1-L3 rows, 8e6 and 1e8 particles on one A100 and the 25M disc + bulge), named
so a caller no longer copies it from the bench. It is built from ONE
:class:`GalaxySettings` instance, so the constructor half (the tree, the MAC, the
opening angle, the softening kernel, the precision) and the run-time half (leaf
size, expansion order, carry -- arguments of ``strict_run_v2``) cannot drift apart.

The module imports nothing heavy; the production runner that reads the run-time
half is not here yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

__all__ = ["GALAXY_SETTINGS", "GalaxySettings"]


@dataclass(frozen=True)
class GalaxySettings:
    """The galaxy production configuration of the fused lane.

    Frozen. The defaults are ``preset="galaxy"``; :data:`GALAXY_SETTINGS` is that
    instance.

    Attributes
    ----------
    leaf_size : int
        Particles per cell leaf at most (``strict_run_v2(leaf_size=...)``; also the
        tree's ``leaf_target``). 64: the leaf-size U-curve's minimum on one A100.
    max_order : int
        Expansion order (``strict_run_v2(max_order=...)``, and the solver's
        ``fixed_order``). p6 since round 5: at ``cell_min_level`` 8 it halves the
        fp64 rel-L2 of p5 for ~5 % of the step.
    theta : float
        Opening angle of the geometric Dehnen MAC.
    cell_min_level : int
        No leaf coarser than this Morton level (``TreeConfig.cell_min_level``).
    leaf_capacity : Union[int, Literal["auto"]]
        ``TreeConfig.leaf_capacity``; ``"auto"`` sizes it at the first eager
        prepare.
    leaf_headroom : float
        ``TreeConfig.leaf_headroom``: the factor over the live cell leaves.
    softening_kernel : str
        Pair softening kernel (:mod:`jaccpot.softening`).
    mac_type : str
        Multipole acceptance criterion. ``"dehnen"`` is geometric at ``theta``; the
        mass-dependent ``"dehnen_error"`` (with ``adaptive_eps``) is the accuracy
        alternative.
    adaptive_eps : Optional[float]
        ``"dehnen_error"``'s relative force-accuracy target; ``None`` with the
        geometric MAC.
    carry : Literal["particles", "state"]
        ``strict_run_v2(carry=...)``: the particle carry (input order, a handle
        between calls) is the production one.
    precision : Literal["fp32", "fp64"]
        Working precision; fp64 is the lane's validation mode.
    """

    leaf_size: int = 64
    max_order: int = 6
    theta: float = 0.8
    cell_min_level: int = 8
    leaf_capacity: Union[int, Literal["auto"]] = "auto"
    leaf_headroom: float = 1.15
    softening_kernel: str = "ferrers3"
    mac_type: str = "dehnen"
    adaptive_eps: Optional[float] = None
    carry: Literal["particles", "state"] = "particles"
    precision: Literal["fp32", "fp64"] = "fp32"


#: ``preset="galaxy"``.
GALAXY_SETTINGS = GalaxySettings()
