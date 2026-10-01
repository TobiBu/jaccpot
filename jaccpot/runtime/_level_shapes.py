"""Static per-level batch widths for the level cascades (plan sub-10ms, Phase 3 in pure JAX).

The M2M upward sweep (``aggregate_m2m_real_by_level``) and the L2L cascade
(``_propagate_solidfmm_locals_by_level``) run one iteration per tree level.
Their per-level work has to have a STATIC shape, and both used a batch of
``num_internal`` nodes (M2M) or every internal node masked by level (L2L), so
each level cost the whole tree: ``nodes x depth``. On the balanced bucket tree
(depth 12) that was tolerable; the radix tree over Morton-cell leaves is 39
levels deep at N = 2x10^5 and the L2L alone took 44 ms per step in its
full-array scatter.

The right static width is the widest LEVEL, not the node count. Under trace the
level table is a tracer, so the width comes from a process-level registry keyed
by the tree's static shape ``(total_nodes, num_internal)``: the eager prepare
that always precedes a traced refresh fills it (with headroom, never
shrinking), and the traced rebuild checks the live tree against it
(:func:`level_width_overflow`) so a wider level trips the capacity guard rather
than dropping nodes.

That "always precedes" holds on one device and **fails inside ``shard_map``**,
where the body is traced from the first call and the registry is cold: the width
falls back to ``num_internal`` and :func:`registered_num_levels` returns ``None``,
which deselects the per-level Pallas cascades outright, silently. An explicit
:class:`~jaccpot.runtime.capacity_plan.FusedCapacityPlan` therefore takes
precedence over the registry wherever one is installed -- including for CONCRETE
offsets, so that a planning pass and the traced refresh it plans for cannot
compile different widths. The registry still records what it observes, so a
planner can widen the next plan.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaccpot._jax_compat import Tracer

if TYPE_CHECKING:  # capacity_plan imports this module; annotation only
    from jaccpot.runtime.capacity_plan import FusedCapacityPlan

__all__ = [
    "level_batch_width",
    "level_width_overflow",
    "registered_level_batch_width",
    "registered_num_levels",
    "pallas_cascades_enabled",
]

_WIDTHS: dict[tuple[int, int], int] = {}
_LEVELS: dict[tuple[int, int], int] = {}
_HEADROOM = 1.25
_LEVEL_HEADROOM = 8


def _key(total_nodes: int, num_internal: int) -> tuple[int, int]:
    return (int(total_nodes), int(num_internal))


def _plan_for(*, total_nodes: int, num_internal: int) -> Optional[FusedCapacityPlan]:
    """The installed capacity plan, if it describes this tree shape.

    Parameters
    ----------
    total_nodes : int
        Node count being resolved.
    num_internal : int
        Internal node count being resolved.

    Returns
    -------
    Optional[FusedCapacityPlan]
        The plan, or ``None`` when none is installed or it is for another shape.
    """
    from jaccpot.runtime.capacity_plan import fused_capacity_plan

    plan = fused_capacity_plan()
    if plan is None:
        return None
    if not plan.matches(total_nodes=int(total_nodes), num_internal=int(num_internal)):
        return None
    return plan


def registered_level_batch_width(*, total_nodes: int, num_internal: int) -> int | None:
    """The registered width for this tree shape, or ``None`` before any eager visit.

    Parameters
    ----------
    total_nodes : int
        Node count of the tree.
    num_internal : int
        Internal node count.

    Returns
    -------
    int | None
        The planned width when a plan is installed for this shape, else the
        stashed one, else ``None`` before any eager visit.
    """
    plan = _plan_for(total_nodes=total_nodes, num_internal=num_internal)
    if plan is not None:
        return int(plan.level_batch_width)
    return _WIDTHS.get(_key(total_nodes, num_internal))


def level_batch_width(
    level_offsets: Array, *, total_nodes: int, num_internal: int
) -> int:
    """Static batch width for the level loops: the widest level, with headroom.

    Concrete ``level_offsets`` (eager) measure the tree and raise the registry
    entry to ``ceil(1.25 x widest level)``, never lowering it; a traced table
    returns the registry entry, or ``num_internal`` (the historical width) when
    nothing eager has been seen for this shape.

    Parameters
    ----------
    level_offsets : Array
        ``(levels + 1,)`` start offsets into ``nodes_by_level``.
    total_nodes : int
        Node count (registry key).
    num_internal : int
        Internal node count (registry key and fallback width).

    Returns
    -------
    int
        A Python int in ``[1, total_nodes]``.
    """
    key = _key(total_nodes, num_internal)
    plan = _plan_for(total_nodes=total_nodes, num_internal=num_internal)
    if isinstance(level_offsets, Tracer):
        if plan is not None:
            return int(plan.level_batch_width)
        return int(_WIDTHS.get(key) or max(int(num_internal), 1))
    offs = np.asarray(jax.device_get(level_offsets)).astype(np.int64)
    counts = np.diff(offs) if offs.size >= 2 else np.ones((1,), np.int64)
    widest = int(np.max(counts))
    live = int(np.count_nonzero(counts))
    levels = min(int(level_offsets.shape[0]) - 1, live + _LEVEL_HEADROOM)
    _LEVELS[key] = max(levels, _LEVELS.get(key, 0))
    width = max(1, int(math.ceil(widest * _HEADROOM)))
    width = min(width, max(int(total_nodes), 1))
    width = max(width, _WIDTHS.get(key, 0))
    _WIDTHS[key] = width
    if plan is not None:
        # Record what this tree actually needs (a planner reads it back through
        # ``plan_from_registry``), but hand back the PLANNED width: a planning
        # pass and the traced refresh it plans for must compile the same shape.
        # A tree too wide for the plan is caught by ``level_width_overflow``.
        return int(plan.level_batch_width)
    return width


def registered_num_levels(*, total_nodes: int, num_internal: int) -> int | None:
    """Static level-loop trip count for this tree shape: live levels + 8 headroom.

    Filled by :func:`level_batch_width` on an eager visit (never lowered);
    ``None`` before any. The traced rebuild guards ``depth > bound`` elsewhere.

    Parameters
    ----------
    total_nodes : int
        Node count (registry key).
    num_internal : int
        Internal node count (registry key).

    Returns
    -------
    int | None
        The planned level count when a plan is installed for this shape, else
        the stashed one, else ``None`` before any eager visit.
    """
    plan = _plan_for(total_nodes=total_nodes, num_internal=num_internal)
    if plan is not None:
        return int(plan.num_levels)
    return _LEVELS.get(_key(total_nodes, num_internal))


def pallas_cascades_enabled() -> bool:
    """Whether the M2M/L2L cascades run as one Pallas launch per level (plan sub-10ms 3).

    ``JACCPOT_CASCADE_PALLAS=1`` selects :mod:`jaccpot.pallas.cascade_real_level`
    on the real basis where its Triton lowering runs (or under
    ``JACCPOT_CASCADE_PALLAS_INTERPRET=1`` anywhere); default ON since Phase 6.

    On the gradient path the context-local override (the resolved
    ``GradConfig.cascade_pallas``) replaces the flag; the hardware check is the
    same. The kernels are differentiable through their ``custom_vjp`` seams
    (reverse Pallas kernels, plan fast-gradients), so the gradient path no
    longer needs the pure-JAX loops.

    Returns
    -------
    bool
        ``True`` when the Pallas cascades are selected.
    """
    from jaccpot._env import env_flag
    from jaccpot.runtime.grad_options import cascade_pallas_override

    override = cascade_pallas_override()
    requested = (
        bool(override)
        if override is not None
        else env_flag(
            "JACCPOT_CASCADE_PALLAS", True
        )  # default on since Phase 6 (2026-09-11)
    )
    if not requested:
        return False
    if env_flag("JACCPOT_CASCADE_PALLAS_INTERPRET", False):
        return True
    from jaccpot.pallas.cascade_real_level import pallas_cascade_level_supported

    return bool(pallas_cascade_level_supported())


def level_width_overflow(
    level_offsets: Array, *, total_nodes: int, num_internal: int
) -> Array:
    """Traced check that no level of a rebuilt tree exceeds the registered width.

    Parameters
    ----------
    level_offsets : Array
        ``(levels + 1,)`` start offsets of the rebuilt tree.
    total_nodes : int
        Node count (registry key).
    num_internal : int
        Internal node count (registry key).

    Returns
    -------
    Array
        Boolean scalar; ``True`` when a level is wider than the loops' batch.
    """
    width = registered_level_batch_width(
        total_nodes=total_nodes, num_internal=num_internal
    )
    if width is None:
        return jnp.asarray(False)
    offs = jnp.asarray(level_offsets)
    return jnp.max(offs[1:] - offs[:-1]) > jnp.asarray(int(width), offs.dtype)
