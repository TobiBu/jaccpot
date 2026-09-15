"""Static shapes for the fused lane, decided up front instead of by an eager visit.

The level cascades need a static per-level batch width and a static level count.
Today those come from a process-level registry (:mod:`jaccpot.runtime._level_shapes`)
that an eager ``prepare_state`` fills before the traced refresh reads it -- which
works because on one device an eager prepare **always** precedes a trace.

Inside ``shard_map`` it never does: the body is traced from the first call. The
registry is then cold, and the failure is silent rather than loud --
``level_batch_width`` falls back to ``num_internal`` (the ``nodes x depth`` loop
the registry exists to avoid) and ``registered_num_levels`` returns ``None``,
which drops the ``pallas_levels`` key and so **deselects the per-level Pallas L2L
cascade altogether**. Measured on a 1023-node cell tree: width 250 -> 511 and the
Pallas cascade skipped, with nothing raised. On the real cell tree the same loop
cost 44 ms per step.

There is a second failure that only appears with one process per GPU: the registry
is process-global and filled from each process's own shard, so the key
``(total_nodes, num_internal)`` matches everywhere while the *values* differ, and
two processes compile different static shapes for one SPMD program.

So the fused lane takes its shapes from an explicit :class:`FusedCapacityPlan`,
built once (from an eager pass over the worst shard), reduced across the mesh, and
installed for the trace. Delivery is context-local rather than through an argument
chain for the reason :mod:`jaccpot.runtime.grad_options` gives for the gradient
gates: the values are read far below the call that resolves them, and one of the
readers feeds a ``jax.jit`` with ``static_argnames``, so threading the plan would
put it in every static cache key.

**A ContextVar is read at trace time and does not retrace by itself.** Put
:meth:`FusedCapacityPlan.fingerprint` into the compile cache key of anything that
caches a traced program, or a re-plan will silently reuse the old shapes.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, replace
from typing import Iterable, Optional

__all__ = [
    "FusedCapacityPlan",
    "fused_capacity_plan",
    "fused_capacity_plan_overrides",
    "merge_plans",
    "plan_from_registry",
]


@dataclass(frozen=True)
class FusedCapacityPlan:
    """Static shapes one compiled fused-lane program must carry for every device.

    Attributes
    ----------
    total_nodes : int
        Node count of the per-device tree; with the registry key ``num_internal``
        this identifies the tree shape the plan describes.
    num_internal : int
        Internal node count of the per-device tree.
    level_batch_width : int
        Static batch width of the M2M/L2L level loops -- the widest level over
        every device, with headroom.
    num_levels : int
        Static trip count of the level loops, with headroom.
    upward_num_levels : int
        Static trip count of the upward M2M sweep: the tree DEPTH plus headroom,
        which is a different quantity from :attr:`num_levels` (the level table's
        live-level count plus headroom). They are kept apart deliberately --
        substituting one for the other truncates a sweep silently.
    """

    total_nodes: int
    num_internal: int
    level_batch_width: int
    num_levels: int
    upward_num_levels: int

    def __post_init__(self) -> None:
        for name in (
            "total_nodes",
            "num_internal",
            "level_batch_width",
            "num_levels",
            "upward_num_levels",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"FusedCapacityPlan.{name} must be >= 1")
        if int(self.level_batch_width) > int(self.total_nodes):
            raise ValueError(
                "level_batch_width cannot exceed total_nodes "
                f"({self.level_batch_width} > {self.total_nodes})"
            )

    def matches(self, *, total_nodes: int, num_internal: int) -> bool:
        """Whether this plan describes the given tree shape.

        A plan is consulted only for the shape it was built for; any other shape
        falls back to the registry, so installing a plan cannot silently reshape
        an unrelated tree.

        Parameters
        ----------
        total_nodes : int
            Node count being resolved.
        num_internal : int
            Internal node count being resolved.

        Returns
        -------
        bool
            ``True`` when the plan applies.
        """
        return int(total_nodes) == int(self.total_nodes) and int(num_internal) == int(
            self.num_internal
        )

    def fingerprint(self) -> tuple[int, ...]:
        """Compile-cache key contribution.

        A ContextVar does not retrace on its own, so a program cached across a
        re-plan would reuse the old static shapes. Include this in the key.

        Returns
        -------
        tuple[int, ...]
            The plan's fields, in declaration order.
        """
        return (
            int(self.total_nodes),
            int(self.num_internal),
            int(self.level_batch_width),
            int(self.num_levels),
            int(self.upward_num_levels),
        )

    def widened_to(
        self,
        *,
        level_batch_width: int = 0,
        num_levels: int = 0,
        upward_num_levels: int = 0,
    ):
        """A copy no narrower than the given shapes.

        Parameters
        ----------
        level_batch_width : int
            Observed width to accommodate.
        num_levels : int
            Observed level count to accommodate.
        upward_num_levels : int
            Observed upward depth to accommodate.

        Returns
        -------
        FusedCapacityPlan
            The widened plan (``self`` when nothing grew).
        """
        width = max(int(self.level_batch_width), int(level_batch_width))
        levels = max(int(self.num_levels), int(num_levels))
        upward = max(int(self.upward_num_levels), int(upward_num_levels))
        if (
            width == int(self.level_batch_width)
            and levels == int(self.num_levels)
            and upward == int(self.upward_num_levels)
        ):
            return self
        return replace(
            self,
            level_batch_width=width,
            num_levels=levels,
            upward_num_levels=upward,
        )


_plan: contextvars.ContextVar[Optional[FusedCapacityPlan]] = contextvars.ContextVar(
    "jaccpot_fused_capacity_plan", default=None
)


def fused_capacity_plan() -> Optional[FusedCapacityPlan]:
    """The plan installed for the current trace, if any.

    Returns
    -------
    Optional[FusedCapacityPlan]
        The context-local plan, or ``None`` when the registry is in charge.
    """
    return _plan.get()


@contextlib.contextmanager
def fused_capacity_plan_overrides(plan: Optional[FusedCapacityPlan]):
    """Install ``plan`` for the duration of the block.

    Wrap the ``jax.jit`` / ``shard_map`` **build** in this, not just the call:
    the shapes are read at trace time.

    Parameters
    ----------
    plan : Optional[FusedCapacityPlan]
        The plan to install; ``None`` restores registry behaviour.

    Yields
    ------
    Optional[FusedCapacityPlan]
        The installed plan.
    """
    token = _plan.set(plan)
    try:
        yield plan
    finally:
        _plan.reset(token)


def plan_from_registry(
    *, total_nodes: int, num_internal: int, upward_num_levels: Optional[int] = None
) -> Optional[FusedCapacityPlan]:
    """Read back what an eager pass recorded for this tree shape.

    This is how a plan is built: run the ordinary eager prepare over a device's
    shard, then read the shapes it measured.

    Parameters
    ----------
    total_nodes : int
        Node count of the tree the eager pass visited.
    num_internal : int
        Internal node count of that tree.
    upward_num_levels : Optional[int]
        The upward sweep's depth bound the same eager pass stashed
        (``_resolve_upward_num_levels``). ``None`` falls back to the level-table
        count, which is never smaller than the depth for these trees.

    Returns
    -------
    Optional[FusedCapacityPlan]
        The plan, or ``None`` when nothing eager has visited this shape.
    """
    from jaccpot.runtime._level_shapes import (
        registered_level_batch_width,
        registered_num_levels,
    )

    width = registered_level_batch_width(
        total_nodes=int(total_nodes), num_internal=int(num_internal)
    )
    levels = registered_num_levels(
        total_nodes=int(total_nodes), num_internal=int(num_internal)
    )
    if width is None or levels is None:
        return None
    return FusedCapacityPlan(
        total_nodes=int(total_nodes),
        num_internal=int(num_internal),
        level_batch_width=int(width),
        num_levels=int(levels),
        upward_num_levels=int(levels if upward_num_levels is None else upward_num_levels),
    )


def merge_plans(plans: Iterable[FusedCapacityPlan]) -> FusedCapacityPlan:
    """Field-wise maximum of per-device plans: the shapes every device can use.

    One compiled program serves every device, so each static shape has to cover
    the **worst** device, not the one that happened to be planned.

    Parameters
    ----------
    plans : Iterable[FusedCapacityPlan]
        Plans for the same tree shape, one per device or process.

    Returns
    -------
    FusedCapacityPlan
        The widest plan.

    Raises
    ------
    ValueError
        If ``plans`` is empty, or the plans describe different tree shapes --
        which would mean the devices disagree about ``leaf_capacity`` and no
        single program can serve them.
    """
    items = list(plans)
    if not items:
        raise ValueError("merge_plans needs at least one plan")
    shapes = {(p.total_nodes, p.num_internal) for p in items}
    if len(shapes) != 1:
        raise ValueError(
            "cannot merge plans for different tree shapes "
            f"{sorted(shapes)}: every device must share one leaf_capacity"
        )
    return FusedCapacityPlan(
        total_nodes=items[0].total_nodes,
        num_internal=items[0].num_internal,
        level_batch_width=max(int(p.level_batch_width) for p in items),
        num_levels=max(int(p.num_levels) for p in items),
        upward_num_levels=max(int(p.upward_num_levels) for p in items),
    )
