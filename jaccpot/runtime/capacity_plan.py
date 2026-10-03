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
from typing import Any, Iterable, Iterator, Optional

__all__ = [
    "FusedCapacityPlan",
    "fused_capacity_plan",
    "fused_capacity_plan_overrides",
    "install_walk_caps",
    "measure_shard_plan",
    "merge_plans",
    "merge_walk_caps",
    "plan_from_registry",
    "plan_level_overflow",
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
    ) -> "FusedCapacityPlan":
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
def fused_capacity_plan_overrides(
    plan: Optional[FusedCapacityPlan],
) -> Iterator[Optional[FusedCapacityPlan]]:
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
        upward_num_levels=int(
            levels if upward_num_levels is None else upward_num_levels
        ),
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


#: Walk-capacity report fields that size the TRACED walk; merged by maximum.
_WALK_CAP_MAX_FIELDS = (
    "queue_capacity",
    "max_pair_queue_requested",
    "compact_far_pair_capacity",
    "near_edge_capacity",
    "max_neighbors_observed",
    "peak_wavefront",
)


def merge_walk_caps(reports: "Iterable[Optional[dict]]") -> Optional[dict]:
    """Field-wise maximum of per-shard walk-capacity reports.

    Each eager prepare records what ITS shard's walk needed
    (``engine._strict_fused_validated_caps``), and the traced walk is sized from that
    record (queue = 1.5 x the peak wavefront, the flat-walk floors). Prepared one shard
    after another on the same engine, the record is simply the LAST shard's -- so every
    device's traced walk is sized for one device. Merge the records and install the
    result (:func:`install_walk_caps`) before building the evaluator.

    Parameters
    ----------
    reports : Iterable[Optional[dict]]
        One report per shard; ``None`` entries are skipped.

    Returns
    -------
    Optional[dict]
        The merged report (the first report's keys, maxima over the sizing fields),
        or ``None`` when no report was given.

    Raises
    ------
    ValueError
        If the reports disagree on ``flat_walk`` -- one compiled program cannot run
        both walks.
    """
    reports = [dict(r) for r in reports if r]
    if not reports:
        return None
    merged = dict(reports[0])
    flat = {bool(r.get("flat_walk")) for r in reports}
    if len(flat) > 1:
        raise ValueError("shards disagree on flat_walk; prepare them alike")
    for key in _WALK_CAP_MAX_FIELDS:
        values = [int(r[key]) for r in reports if r.get(key) is not None]
        if values:
            merged[key] = max(values)
    return merged


def install_walk_caps(engine: Any, caps: Optional[dict]) -> None:
    """Make ``caps`` the record the engine sizes its traced walk from.

    Parameters
    ----------
    engine : Any
        The runtime engine, or the ``FastMultipoleMethod`` facade.
    caps : Optional[dict]
        A (merged) walk-capacity report; ``None`` leaves the engine unchanged.
    """
    if caps is None:
        return
    impl = getattr(engine, "_impl", engine)
    impl._strict_fused_validated_caps = dict(caps)


def plan_level_overflow(tree: Any) -> Any:
    """Traced: whether ``tree`` outgrew the installed plan's static level loops.

    The M2M/L2L level loops run ``num_levels`` / ``upward_num_levels`` levels of at
    most ``level_batch_width`` nodes each. A rebuilt tree with a wider level or more
    levels loses the excess nodes from both cascades: every particle under them keeps
    its near field and gets NO far field. Measured on the mesh lane (two A100s,
    Plummer N = 4e5, seed 2): one particle at 0.024 against 0.37, rel-L2 4.2e-2
    against 8.8e-4, and with the plan's width halved on purpose 4.9e-1 -- with every
    capacity flag False both times. The refresh's own width/depth guard folds into the
    walk's far-pair saturation and never reached the mesh flag; this reads the tree
    directly.

    Parameters
    ----------
    tree : Any
        The rebuilt tree (``level_offsets``, ``node_level``).

    Returns
    -------
    Any
        Boolean scalar; ``False`` when no plan is installed.
    """
    import jax.numpy as jnp

    plan = fused_capacity_plan()
    if plan is None:
        return jnp.asarray(False)
    offs = jnp.asarray(tree.level_offsets)
    widest = jnp.max(offs[1:] - offs[:-1])
    depth = jnp.max(jnp.asarray(tree.node_level)) + 1
    levels = min(int(plan.num_levels), int(plan.upward_num_levels))
    return (widest > int(plan.level_batch_width)) | (depth > levels)


def _lift_registry_to_refresh_tree(
    impl: Any,
    template: Any,
    positions: Any,
    masses: Any,
    *,
    bounds: Any,
    num_valid: Optional[Any],
) -> None:
    """Raise the level registry and the depth stash to the tree the REFRESH builds.

    The eager prepare builds this shard's tree in the shard's own box over every row;
    the traced refresh rebuilds it from the same template in the GLOBAL mesh box over
    the live rows. Different boxes cut different cells, so the refresh tree can be
    wider or deeper than the one the plan was measured on (seed 2 at 4e5 on two
    cards: widest level 4564 against a planned 4525 = 1.25 x 3620). Rebuild it the
    way the refresh will and let the registry rise to it (it never lowers).

    Parameters
    ----------
    impl : Any
        The runtime engine (``FastMultipoleMethod._impl``).
    template : Any
        The prepared tree, the refresh's template.
    positions : Any
        This shard's padded positions, as the refresh receives them.
    masses : Any
        This shard's padded masses.
    bounds : Any
        The global mesh box (``fused.global_mesh_bounds``).
    num_valid : Optional[Any]
        Live rows; ``None`` treats every row as live.
    """
    import jax.numpy as jnp
    from yggdrax.tree import rebuild_static_radix_tree_from_template

    from jaccpot.runtime._level_shapes import level_batch_width

    cells = getattr(impl, "_tree_leaf_partition", "buckets") == "cells"
    if not cells:
        return  # only the cell partition is rebuilt per step against a box
    out = rebuild_static_radix_tree_from_template(
        jnp.asarray(positions),
        jnp.asarray(masses),
        template,
        bounds=bounds,
        return_reordered=True,
        leaf_partition="cells",
        return_overflow=True,
        cell_min_level=int(getattr(impl, "_tree_cell_min_level", 0)),
        **({"num_valid": jnp.asarray(num_valid)} if num_valid is not None else {}),
    )
    tree = out[0]
    level_batch_width(
        tree.level_offsets,
        total_nodes=int(tree.parent.shape[0]),
        num_internal=int(tree.left_child.shape[0]),
    )
    resolve = getattr(impl, "_resolve_upward_num_levels", None)
    if resolve is not None:
        resolve(tree)


def measure_shard_plan(
    solver: Any,
    positions: Any,
    masses: Any,
    *,
    leaf_size: int,
    max_order: int,
    theta: Optional[float] = None,
    bounds: Optional[Any] = None,
    num_valid: Optional[Any] = None,
) -> tuple:
    """Eagerly prepare one shard and read back what its static shapes must cover.

    The sequence every multi-device driver needs per shard: clear the process-level
    level registry (so this shard's widths are not mixed with another's), run the
    eager fused prepare, then read the level plan and the walk-capacity report it
    recorded. Merge the results over shards with :func:`merge_plans` and
    :func:`merge_walk_caps`.

    Parameters
    ----------
    solver : Any
        The ``FastMultipoleMethod`` (the SAME instance the evaluator will be built
        around: the traced body depends on caches the eager prepare fills).
    positions : Any
        This shard's padded positions.
    masses : Any
        This shard's padded masses (zero on padding rows).
    leaf_size : int
        Leaf target.
    max_order : int
        Expansion order.
    theta : Optional[float]
        Opening angle.
    bounds : Optional[Any]
        The GLOBAL mesh box the traced force will build in
        (:func:`jaccpot.distributed.fused.global_mesh_bounds`). Given, the eager
        prepare builds in it, and the plan also covers this shard's tree rebuilt in
        it over the live rows only -- the tree the force actually walks. ``None``
        measures the eager tree in the shard's own box, whose cells differ: it can
        be narrower than the refresh tree (a silently truncated level) or hold more
        leaves (a leaf-capacity raise at setup).
    num_valid : Optional[Any]
        This shard's live row count, with ``bounds``.

    Returns
    -------
    tuple
        ``(prepared_state, FusedCapacityPlan, walk_caps)``.
    """
    from jaccpot.runtime import _level_shapes as level_shapes

    level_shapes._WIDTHS.clear()
    level_shapes._LEVELS.clear()
    prepared = solver.strict_fused_prepared_eval_fn(
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        theta=theta,
        # the eager tree in the force's box: its leaf count, level widths and walk
        # caps are what the static shapes are sized from
        **({} if bounds is None else {"bounds": bounds}),
    )[0]
    impl = getattr(solver, "_impl", solver)
    if bounds is not None:
        _lift_registry_to_refresh_tree(
            impl, prepared.tree, positions, masses, bounds=bounds, num_valid=num_valid
        )
    total_nodes = int(prepared.tree.node_ranges.shape[0])
    num_internal = int(prepared.tree.left_child.shape[0])
    plan = plan_from_registry(
        total_nodes=total_nodes,
        num_internal=num_internal,
        upward_num_levels=getattr(impl, "_cells_upward_num_levels", None),
    )
    caps = getattr(impl, "_strict_fused_validated_caps", None)
    return prepared, plan, (dict(caps) if isinstance(caps, dict) else None)
