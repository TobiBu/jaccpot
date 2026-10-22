"""The fused multi-GPU force on ONE shape template: caps-only prepares, no stacked state.

:func:`jaccpot.distributed.rollout.setup_fused_force` builds the mesh force the way
the single card did before S2: a FULL eager prepare per device
(``capacity_plan.measure_shard_plan``), narrower shards prepared a second time, and
the whole prepared states -- tree, multipoles, interaction lists, near-field payload
of every device -- stacked into a mesh-sharded argument of the force program
(:func:`~jaccpot.distributed.fused.make_fused_force_evaluator`), held for the run's
lifetime. A traced refresh reads none of those values: it rebuilds every one of
them from the positions (the dead-code test in
``tests/integration/strict/test_strict_run_v2.py``), and only their SHAPES and
static fields matter.

This module is the single-card particle carry's answer (S2-S3) applied per device:

1. **Caps-only prepares in the global box.** Per device, in mesh order, on that
   device: ``prepare_state_caps_only`` (the tree, the upward pass and the eager flat
   walk's ladder, which records the walk caps; no list, no downward pass, no
   near-field payload) in the GLOBAL mesh box, then the level registry lifted to the
   tree the traced refresh builds there over the live rows
   (``capacity_plan._lift_registry_to_refresh_tree``, as ``measure_shard_plan``
   does).
2. **Merged once.** Level plans (``merge_plans``) and walk caps
   (``merge_walk_caps``) take the field-wise maximum over devices; the merged walk
   caps are installed on the engine, where the traced walk reads them.
   ``leaf_capacity="auto"`` is sized BEFORE the first prepare from the largest
   device: the cell leaves its padded shard cuts into in the mesh box (what the
   eager build sees; at least the live leaves the refresh cuts) times the
   configured headroom.
3. **One template, one provider.** The shape template is derived after the merge
   (the caps-only template: the shapes of a rebuild under ``jax.eval_shape``), so
   one :class:`~jaccpot.runtime.fused_provider.FusedForceProvider` serves every
   device. Inside the ``shard_map`` body each device materialises it
   (``provider.like()``: NaN broadcasts that XLA removes again) and rebuilds from
   it, in the global box, over its live rows, with the cross hook between the
   sweeps.
4. **Nothing eager survives.** The eager prepares are dropped (and the engine's
   prepared-state caches cleared) before anything is traced.

The cross NEAR term is added after the evaluation by the helper the stacked route
uses (:func:`jaccpot.distributed.fused._add_cross_near_term`), so both routes add the
same term with the same arithmetic.

:class:`FusedMeshForce` hands this route to the segment runner
(:mod:`jaccpot.distributed.segment`), with the segment retry's re-plan.

GATE A1 (``tests/distributed/test_mesh_segment_runner.py``): with the same caps,
this route's force (the runner's start force) equals the stacked route's BITWISE
on two forced CPU devices -- so the template's broadcasts reach no value of the
force.

The capacity verdict is the provider's bits (``FusedTopology.overflow_bits``) with
the two multi-GPU bits the topology module reserves for this lane:
``OVERFLOW_LEVEL_PLAN`` (a tree wider or deeper than the installed plan) and
``OVERFLOW_CROSS`` (any cross-domain capacity), and the walk's and tree's needs
(``FusedTopology.needs``) for a re-plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaccpot.distributed.fused import (
    AXIS_NAME,
    _add_cross_near_term,
    _fold_cross_flags,
    _refuse_error_criterion,
    _warn_if_slow_ragged_exchange,
    global_mesh_bounds,
    host_mesh_bounds,
)

__all__ = [
    "FusedMeshForce",
    "MeshForceSetup",
    "cross_needs",
    "make_mesh_cross_hook",
    "mesh_force_shard",
    "prepare_mesh_force",
]


@dataclass
class MeshForceSetup:
    """What the template route traces the mesh force from.

    Holds no device buffer of a prepared state: the provider carries the shape
    template, the plan and walk caps are host values.

    Attributes
    ----------
    provider : Any
        The :class:`~jaccpot.runtime.fused_provider.FusedForceProvider` every device
        rebuilds with (its template: one device's prepared-state shapes).
    plan : Any
        The merged :class:`~jaccpot.runtime.capacity_plan.FusedCapacityPlan`;
        install it around every trace (``fused_capacity_plan_overrides``).
    walk_caps : Optional[dict]
        The merged walk-capacity record, installed on the engine.
    leaf_capacity : int
        The cell-leaf tree's leaf capacity every device compiles.
    box : tuple
        The host mesh box the prepares were built in (diagnostics).
    """

    provider: Any
    plan: Any
    walk_caps: Optional[dict]
    leaf_capacity: int
    box: tuple

    def key(self) -> tuple:
        """What a compile cache of a program traced from this setup must hold.

        Returns
        -------
        tuple
            The provider's key (template shapes, walk caps, level bound, ...) and
            the plan's fingerprint: the ContextVar delivering the plan does not
            retrace by itself.
        """
        return (*self.provider.key(), *self.plan.fingerprint())


def _by_device(arr: Array, devices: list) -> list:
    """The per-device shards of a flat ``P(axis)`` array, in mesh order.

    Parameters
    ----------
    arr : Array
        A mesh-sharded array.
    devices : list
        The mesh's devices, in mesh order.

    Returns
    -------
    list
        One device-resident ``(cap, ...)`` array per device.
    """
    shards = {s.device: s.data for s in arr.addressable_shards}
    return [shards[d] for d in devices]


def prepare_mesh_force(
    solver: Any,
    mesh: Any,
    positions: Array,
    masses: Array,
    counts: Any,
    *,
    leaf_size: int,
    max_order: int,
    theta: Optional[float] = None,
    grow_levels: Optional[tuple[int, int]] = None,
) -> MeshForceSetup:
    """Caps-only prepares per device in the mesh box, merged into ONE template.

    See the module docstring for the steps. Run it on the SAME solver the force
    is traced with: the traced walk is sized from the walk caps it installs, and
    the near-field stage from what the prepares recorded. The level registry
    (process-global) is cleared per device, as ``measure_shard_plan`` does.

    Parameters
    ----------
    solver : Any
        The ``FastMultipoleMethod`` (large-N production profile, cell leaves,
        static-radix tree, the flat walk).
    mesh : Any
        The 1-D device mesh.
    positions : Array
        ``(ndev * cap, 3)`` flat, mesh-sharded positions; padding rows anywhere
        finite (they are folded onto the device's first row here).
    masses : Array
        ``(ndev * cap,)`` flat, mesh-sharded masses; zero on padding.
    counts : Any
        ``(ndev,)`` live rows per device.
    leaf_size : int
        Leaf target.
    max_order : int
        Expansion order.
    theta : Optional[float]
        Opening angle; ``None`` takes the solver's.
    grow_levels : Optional[tuple[int, int]]
        ``(widest, levels)``: the TREE re-plan of a failed segment, whose trees
        were wider or deeper than this prepare's start trees. The merged plan's
        level batch then covers a level of ``widest`` nodes (x1.25, as the
        registry sizes it) and its level loops ``levels`` levels (the engine's
        grown depth bound), as ``FusedForceProvider.replanned`` widens the
        single card's registry.

    Returns
    -------
    MeshForceSetup
        The provider, merged plan and walk caps.

    Raises
    ------
    NotImplementedError
        If the solver asks for Dehnen's error criterion
        (:func:`~jaccpot.distributed.fused._refuse_error_criterion`); raised before
        anything is prepared.
    RuntimeError
        If the profile is not the large-N production profile, the shard capacity is
        not in ``JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET``, or the walk is not the
        flat walk (no caps-only prepare: its lists are built as it walks).
    ValueError
        If the leaves are buckets (a template rebuild needs cell leaves).
    """
    from jaccpot.distributed.rollout import _mesh_devices, _pad_onto_live_row
    from jaccpot.runtime import _level_shapes as level_shapes
    from jaccpot.runtime.capacity_plan import (
        _lift_registry_to_refresh_tree,
        auto_leaf_capacity,
        fused_capacity_plan_overrides,
        install_walk_caps,
        live_cell_leaves,
        merge_plans,
        merge_walk_caps,
        plan_from_registry,
    )
    from jaccpot.runtime.fused_provider import FusedForceProvider, require_cell_leaves
    from jaccpot.runtime.strict_carry import PreparedCaps

    _refuse_error_criterion(solver, entry="prepare_mesh_force")
    engine = getattr(solver, "_impl", solver)
    require_cell_leaves(engine)
    devices = _mesh_devices(mesh)
    ndev = len(devices)
    counts_host = [int(c) for c in np.asarray(counts)]
    if len(counts_host) != ndev:
        raise ValueError(f"counts has {len(counts_host)} entries for {ndev} devices")
    pos_d, mass_d = _by_device(positions, devices), _by_device(masses, devices)
    cap = int(pos_d[0].shape[0])
    if not engine._is_large_n_gpu_production_profile():
        raise RuntimeError(
            "prepare_mesh_force requires the large_n_gpu production profile."
        )
    if not engine._strict_fused_profile_allows_n(cap):
        raise RuntimeError(
            f"prepare_mesh_force: the shard capacity {cap} is not in "
            "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET; add it, or leave the set empty "
            "to allow all N."
        )
    # The eager trees are built over every row of a shard: its padding sits on its
    # first (live) row, where it adds at most a few leaves. Folded here because a
    # shard's padding stays where the last repartition put it while its first row
    # moves on.
    pos_d = [
        _pad_onto_live_row(jnp.asarray(p), jnp.arange(cap) < counts_host[d])
        for d, p in enumerate(pos_d)
    ]
    box = host_mesh_bounds([np.asarray(p) for p in pos_d], counts_host)
    runtime_overrides = engine._resolve_runtime_execution_overrides(num_particles=cap)

    headroom = getattr(engine, "_tree_leaf_capacity_auto", None)
    if headroom is not None and getattr(engine, "_tree_leaf_capacity", None) is None:
        # leaf_capacity="auto", sized ONCE for every device: the eager prepare would
        # size it from the first shard alone, and every device compiles one shape.
        # Counted over the padded rows in the mesh box, as the eager build cuts
        # them -- never fewer than the live leaves the traced refresh cuts.
        live = max(
            live_cell_leaves(
                jnp.asarray(pos_d[d], engine.working_dtype),
                leaf_size=int(leaf_size),
                cell_min_level=int(getattr(engine, "_tree_cell_min_level", 0)),
                bounds=box,
            )
            for d in range(ndev)
        )
        engine._tree_leaf_capacity = auto_leaf_capacity(live, float(headroom))

    plans, walks = [], []
    stub_caps: Optional[PreparedCaps] = None
    for d in range(ndev):
        # this shard's widths only: the registry is process-global
        level_shapes._WIDTHS.clear()
        level_shapes._LEVELS.clear()
        with jax.default_device(devices[d]):
            caps = engine.prepare_state_caps_only(
                pos_d[d],
                mass_d[d],
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_tree=engine._jit_tree_default,
                runtime_overrides_override=runtime_overrides,
                bounds=box,
            )
            if not isinstance(caps, PreparedCaps):
                raise RuntimeError(
                    "prepare_mesh_force needs the flat walk's caps-only prepare; this "
                    f"configuration's prepare returned a {type(caps).__name__} (a walk "
                    "that builds its lists as it walks). Run the flat walk "
                    "(JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK=1, the default) or the "
                    "stacked route (rollout.setup_fused_force)."
                )
            # the plan must also cover the tree the traced refresh builds: this
            # shard's live rows only, in the same box
            _lift_registry_to_refresh_tree(
                engine,
                caps.stub.tree,
                pos_d[d],
                mass_d[d],
                bounds=box,
                num_valid=counts_host[d],
            )
        tree = caps.stub.tree
        plans.append(
            plan_from_registry(
                total_nodes=int(tree.node_ranges.shape[0]),
                num_internal=int(tree.left_child.shape[0]),
                upward_num_levels=getattr(engine, "_cells_upward_num_levels", None),
            )
        )
        validated = getattr(engine, "_strict_fused_validated_caps", None)
        walks.append(dict(validated) if isinstance(validated, dict) else None)
        if d == 0:
            stub_caps = caps
        del caps, tree
    plan = merge_plans(plans)
    if grow_levels is not None:
        widest, levels = (int(v) for v in grow_levels)
        assert stub_caps is not None
        table_levels = int(stub_caps.stub.tree.level_offsets.shape[0]) - 1
        plan = plan.widened_to(
            level_batch_width=min(
                int(plan.total_nodes), math.ceil(widest * level_shapes._HEADROOM)
            ),
            num_levels=min(table_levels, levels),
            upward_num_levels=levels,
        )
    walk_caps = merge_walk_caps(walks)
    install_walk_caps(engine, walk_caps)

    # ONE template for every device, derived after the merge: the rebuild's output
    # shapes (its lists among them) are sized from the merged caps
    assert stub_caps is not None
    with fused_capacity_plan_overrides(plan):
        provider, _ = FusedForceProvider(
            engine,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            theta=theta,
            runtime_overrides=runtime_overrides,
        )._with_caps_template(stub_caps, pos_d[0], mass_d[0])
    del stub_caps, pos_d, mass_d
    # caches that would keep an eager prepare's buffers alive through the run
    engine._topology_reuse_entry = None
    engine.clear_prepared_state_cache()
    return MeshForceSetup(
        provider=provider,
        plan=plan,
        walk_caps=walk_caps,
        leaf_capacity=int(engine._tree_leaf_capacity),
        box=tuple(np.asarray(b) for b in box),
    )


def mesh_force_shard(
    setup: MeshForceSetup,
    positions: Array,
    masses: Array,
    live: Array,
    *,
    cross_hook: Optional[Callable[[Any], Any]] = None,
    cross_near_sink: Optional[dict] = None,
    axis_name: str = AXIS_NAME,
) -> tuple[Array, Array, Array]:
    """One device's self-gravity on the template route (inside ``shard_map``).

    The global box (:func:`~jaccpot.distributed.fused.global_mesh_bounds`, one
    all-reduce), the provider's rebuild from the materialised template over the
    live rows with the cross hook between the sweeps, its evaluation, and the cross
    NEAR term. The same refresh and evaluation as
    :func:`~jaccpot.distributed.fused.fused_force_step`, from a template instead of
    a prepared state.

    Parameters
    ----------
    setup : MeshForceSetup
        From :func:`prepare_mesh_force`.
    positions : Array
        ``(cap, 3)`` this device's positions.
    masses : Array
        ``(cap,)`` masses, zero on padding.
    live : Array
        Scalar live row count (at most ``cap``).
    cross_hook : Optional[Callable[[Any], Any]]
        The cross-domain hook (``None`` on one device).
    cross_near_sink : Optional[dict]
        The hook's near sink.
    axis_name : str
        Mesh axis.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(acceleration, bits, needs)``: ``(cap, 3)`` in input row order (padding
        rows meaningless), the ``uint32`` overflow bits (with
        ``OVERFLOW_LEVEL_PLAN`` and ``OVERFLOW_CROSS``) and ``int32`` needs:
        ``fused_topology.NEEDS_FIELDS`` (whose ``tree_overflow`` also counts a
        level-plan overflow) followed by the cross hook's (:func:`cross_needs`).
    """
    from jaccpot.runtime.capacity_plan import plan_level_overflow
    from jaccpot.runtime.fused_topology import (
        NEEDS_FIELDS,
        OVERFLOW_CROSS,
        OVERFLOW_LEVEL_PLAN,
    )

    provider = setup.provider
    # a RE-TRACE must not read the previous trace's tracers (fused_force_step)
    if cross_near_sink is not None:
        cross_near_sink.clear()
    bounds = global_mesh_bounds(positions, num_valid=live, axis_name=axis_name)
    topology = provider.rebuild(
        positions,
        masses,
        bounds=bounds,
        num_valid=live,
        cross_hook=cross_hook,
    )
    acceleration = _add_cross_near_term(
        provider.engine,
        topology.prepared,
        jnp.asarray(provider.accelerations(topology)),
        cross_near_sink,
    )
    # a tree wider or deeper than the installed plan's level loops loses nodes from
    # both cascades (capacity_plan.plan_level_overflow); the refresh's own tree flag
    # checks the same plan, this one also the level-table bound
    level = jnp.asarray(plan_level_overflow(topology.prepared.tree), jnp.bool_)
    cross = _fold_cross_flags(jnp.asarray(False), cross_hook, cross_near_sink, None)
    extra = jnp.where(level, jnp.uint32(OVERFLOW_LEVEL_PLAN), jnp.uint32(0)) | (
        jnp.where(cross, jnp.uint32(OVERFLOW_CROSS), jnp.uint32(0))
    )
    bits = topology.overflow_bits(extra=extra)
    # a level-plan overflow IS a tree overflow: the re-plan widens the tree's bounds
    needs = topology.needs()
    tree_slot = NEEDS_FIELDS.index("tree_overflow")
    needs = needs.at[tree_slot].max(level.astype(jnp.int32))
    return acceleration, bits, jnp.concatenate([needs, cross_needs(cross_hook)])


def cross_needs(cross_hook: Any) -> Array:
    """The cross hook's needs vector of this trace: the seam for its re-plan.

    A hook that exposes ``needs_fields`` (names, a static tuple) and fills
    ``needs_sink[name]`` (an ``int32`` scalar per field: the exact demand of that
    cross capacity) inside the trace gets them carried by the segment runner --
    max-ed over the steps and the mesh -- and handed to its ``cross_replan``
    callback after a cross overflow. Today's hook exposes none: an empty vector.

    Parameters
    ----------
    cross_hook : Any
        The hook, or ``None``.

    Returns
    -------
    Array
        ``int32[len(needs_fields)]``.
    """
    fields = tuple(getattr(cross_hook, "needs_fields", ()) or ())
    if not fields:
        return jnp.zeros((0,), jnp.int32)
    sink = getattr(cross_hook, "needs_sink")
    return jnp.stack([jnp.asarray(sink[name], jnp.int32) for name in fields])


def make_mesh_cross_hook(
    solver: Any,
    mesh: Any,
    *,
    theta: float,
    cross_caps: Any = None,
    near_theta: Optional[float] = None,
    axis_name: str = AXIS_NAME,
) -> tuple[Optional[Callable[[Any], Any]], Optional[dict]]:
    """The cross hook and its near sink for this mesh, as ``setup_fused_force`` builds them.

    Parameters
    ----------
    solver : Any
        The solver (its walk separation floor).
    mesh : Any
        The 1-D mesh.
    theta : float
        Opening angle of the cross walks.
    cross_caps : Any
        :class:`~jaccpot.distributed.cross.CrossCapacities`; ``None`` takes defaults.
    near_theta : Optional[float]
        Passed to the hook.
    axis_name : str
        Mesh axis.

    Returns
    -------
    tuple[Optional[Callable[[Any], Any]], Optional[dict]]
        ``(hook, near_sink)``; ``(None, None)`` on one device.
    """
    from jaccpot.distributed.cross import make_cross_hook
    from jaccpot.distributed.rollout import _mesh_devices

    ndev = len(_mesh_devices(mesh))
    if ndev < 2:
        return None, None
    sink: dict = {}
    hook = make_cross_hook(
        ndev=ndev,
        theta=theta,
        caps=cross_caps,
        near_sink=sink,
        near_theta=near_theta,
        axis_name=axis_name,
        # the local walks' floor (a compact kernel's support): the cross walks must
        # keep their far pairs as far apart
        separation_floor=getattr(solver, "_impl", solver)._walk_separation_floor(),
    )
    return hook, sink


class FusedMeshForce:
    """The fused lane's force for :class:`~jaccpot.distributed.segment.MeshSegmentRunner`.

    Holds the template-route setup (:func:`prepare_mesh_force`) and the cross hook,
    and hands the runner what it traces and keys its programs on: :meth:`shard`
    inside its ``shard_map`` bodies, under :meth:`context` (the merged plan
    installed), cached under :meth:`key`. :meth:`prepare` runs at the runner's
    start and again at a re-planned segment's start.

    Parameters
    ----------
    solver : Any
        The ``FastMultipoleMethod`` (large-N production profile, cell leaves, the
        flat walk, a geometric MAC).
    mesh : Any
        The 1-D mesh.
    leaf_size : int
        Leaf target.
    max_order : int
        Expansion order.
    theta : Optional[float]
        Opening angle; ``None`` takes the solver's.
    cross_caps : Any
        :class:`~jaccpot.distributed.cross.CrossCapacities`; ``None`` takes the
        constructor defaults.
    near_theta : Optional[float]
        Passed to the cross hook.
    cross_replan : Optional[Callable[[np.ndarray, Any], Any]]
        THE CROSS RE-PLAN SEAM. After a segment whose cross capacities overflowed
        (``OVERFLOW_CROSS``): ``cross_replan(cross_needs, cross_caps) ->
        CrossCapacities | None``, with ``cross_needs`` the hook's needs vector
        (:func:`cross_needs`) max-ed over the segment's steps and the mesh. New
        capacities rebuild the hook and the segment is re-run with them; ``None``
        (or no callback, today's default) declines, and the segment raises.
    level_floor : Optional[tuple[int, int]]
        ``(widest, levels)``: every prepare's level plan covers at least a level
        of ``widest`` nodes and ``levels`` levels (:func:`prepare_mesh_force`'s
        ``grow_levels``). A tree re-plan raises it; give a re-planned run's
        :attr:`level_floor` to a fresh force to compile the same plan.
    axis_name : str
        Mesh axis.

    Raises
    ------
    NotImplementedError
        If the solver asks for Dehnen's error criterion; raised before anything is
        built.
    """

    def __init__(
        self,
        solver: Any,
        mesh: Any,
        *,
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
        cross_caps: Any = None,
        near_theta: Optional[float] = None,
        cross_replan: Optional[Callable[[np.ndarray, Any], Any]] = None,
        level_floor: Optional[tuple[int, int]] = None,
        axis_name: str = AXIS_NAME,
    ) -> None:
        _refuse_error_criterion(solver, entry="FusedMeshForce")
        self.solver = solver
        self.engine = getattr(solver, "_impl", solver)
        self.mesh = mesh
        self.leaf_size = int(leaf_size)
        self.max_order = int(max_order)
        self.theta = theta
        self.cross_caps = cross_caps
        self.near_theta = near_theta
        self.cross_replan = cross_replan
        self.level_floor = None if level_floor is None else tuple(level_floor)
        self.axis_name = axis_name
        self.setup: Optional[MeshForceSetup] = None
        #: the walk's peak wavefront a re-plan keeps for the next prepare
        self._peak_floor = 0
        self.hook, self.near_sink = self._make_hook()
        _warn_if_slow_ragged_exchange(mesh)

    def _make_hook(self) -> tuple[Optional[Callable[[Any], Any]], Optional[dict]]:
        """The cross hook for the current cross capacities.

        Returns
        -------
        tuple[Optional[Callable[[Any], Any]], Optional[dict]]
            ``(hook, near_sink)``.
        """
        return make_mesh_cross_hook(
            self.solver,
            self.mesh,
            theta=self.theta_value,
            cross_caps=self.cross_caps,
            near_theta=self.near_theta,
            axis_name=self.axis_name,
        )

    @property
    def theta_value(self) -> float:
        """The opening angle the local and the cross walks use.

        Returns
        -------
        float
            ``theta``, else the solver's.
        """
        return float(self.engine.theta if self.theta is None else self.theta)

    @property
    def cache_owner(self) -> Any:
        """What the runner's compiled programs are cached on: the engine.

        Returns
        -------
        Any
            The runtime engine; its programs are shared by runners on one solver.
        """
        return self.engine

    @property
    def needs_size(self) -> int:
        """Length of :meth:`shard`'s needs vector.

        Returns
        -------
        int
            ``len(NEEDS_FIELDS)`` plus the cross hook's fields (:func:`cross_needs`).
        """
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS

        cross = tuple(getattr(self.hook, "needs_fields", ()) or ())
        return len(NEEDS_FIELDS) + len(cross)

    def prepare(self, positions: Array, masses: Array, counts: Array) -> None:
        """The template-route setup at these particles (:func:`prepare_mesh_force`).

        Parameters
        ----------
        positions : Array
            ``(ndev * cap, 3)`` flat, mesh-sharded positions.
        masses : Array
            ``(ndev * cap,)`` masses.
        counts : Array
            ``(ndev,)`` live rows per device.
        """
        self.setup = None  # the old template, plan and caps go first
        self.setup = prepare_mesh_force(
            self.solver,
            self.mesh,
            positions,
            masses,
            counts,
            leaf_size=self.leaf_size,
            max_order=self.max_order,
            theta=self.theta,
            grow_levels=self.level_floor,
        )
        if self._peak_floor:
            # the eager walks saw the segment's START; keep the wavefront the
            # segment needed, so the retraced walk's queue covers it (as
            # FusedForceProvider.replanned does)
            validated = dict(self.engine._strict_fused_validated_caps or {})
            validated["peak_wavefront"] = max(
                int(validated.get("peak_wavefront") or 0), self._peak_floor
            )
            self.engine._strict_fused_validated_caps = validated
            self.setup.walk_caps = dict(validated)
            self._peak_floor = 0

    def _setup_or_raise(self) -> MeshForceSetup:
        """The prepared setup.

        Returns
        -------
        MeshForceSetup
            :attr:`setup`.

        Raises
        ------
        RuntimeError
            Before :meth:`prepare`.
        """
        if self.setup is None:
            raise RuntimeError("FusedMeshForce.prepare must run before the force")
        return self.setup

    def key(self) -> tuple:
        """What a program traced from this force must be cached under.

        Returns
        -------
        tuple
            The setup's key (template shapes, walk caps, level bound, plan), the
            opening angles and the cross capacities: the hook is rebuilt from them,
            and its buffers are shapes of the program.
        """
        caps = self.cross_caps
        caps_key = None if caps is None else tuple(sorted(vars(caps).items()))
        return (
            "fused",
            *self._setup_or_raise().key(),
            self.theta_value,
            self.near_theta,
            caps_key,
        )

    def context(self) -> Any:
        """The merged plan installed, for every trace AND call of a program.

        Returns
        -------
        Any
            A context manager (``fused_capacity_plan_overrides``).
        """
        from jaccpot.runtime.capacity_plan import fused_capacity_plan_overrides

        return fused_capacity_plan_overrides(self._setup_or_raise().plan)

    def before_trace(self) -> None:
        """Forget the last trace's walk caps record before a new program traces.

        The traced walk records its caps on the engine, and the verdict reads them
        in the same trace: a record another trace left must not be read as this
        one's (``strict_run_v2`` resets it the same way).
        """
        self.engine._strict_fused_traced_caps = None

    def shard(
        self, positions: Array, masses: Array, live: Array, ids: Array
    ) -> tuple[Array, Array, Array]:
        """One device's self-gravity (:func:`mesh_force_shard`; inside ``shard_map``).

        Parameters
        ----------
        positions : Array
            ``(cap, 3)``.
        masses : Array
            ``(cap,)``.
        live : Array
            Scalar live row count.
        ids : Array
            ``(cap,)`` global ids; unused (the rows come back in input order).

        Returns
        -------
        tuple[Array, Array, Array]
            ``(acceleration, bits, needs)``.
        """
        del ids
        return mesh_force_shard(
            self._setup_or_raise(),
            positions,
            masses,
            live,
            cross_hook=self.hook,
            cross_near_sink=self.near_sink,
            axis_name=self.axis_name,
        )

    # ------------------------------------------------------------------
    # The segment retry.
    # ------------------------------------------------------------------

    def retry_mode(self, needs: np.ndarray, retries: int, tree_first: bool) -> Any:
        """How a failed segment may be retried: the single card's policy.

        ``engine._segment_retry_mode`` on the local needs (at most two retries,
        the second for the walk only after a first that re-planned the tree;
        ``JACCPOT_STRICT_SEGMENT_RETRY=0`` turns them off).

        Parameters
        ----------
        needs : np.ndarray
            The failed run's needs, max-ed over its steps and the mesh.
        retries : int
            Retries the segment ran so far.
        tree_first : bool
            The segment's first failure was a tree overflow.

        Returns
        -------
        Any
            ``"full"``, ``"walk"`` or ``None`` (raise).
        """
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS

        local = np.asarray(needs, np.int64)[: len(NEEDS_FIELDS)]
        return self.engine._segment_retry_mode(local, int(retries), bool(tree_first))

    def replan(self, needs: np.ndarray, bits: int, *, walk_only: bool = False) -> bool:
        """Grow what a failed segment outgrew; the caller then re-prepares.

        The mesh's ``FusedForceProvider.replanned``, before its re-prepare:

        * the WALK: the unnamed flat-walk caps to cover the needs
          (``engine._replan_walk_caps_from_needs``; a named cap declines);
        * the TREE (not under ``walk_only``): the leaf capacity to
          ``auto_leaf_capacity(leaf_needed)`` (sticky on the engine), the depth
          bound to the deepest tree + 8, and :attr:`level_floor` to the widest
          level and that bound -- the installed PLAN's bounds are what the
          programs checked, so they are what is compared;
        * the CROSS capacities, through :attr:`cross_replan` (the seam): with no
          callback, or one that declines, a cross overflow declines the retry --
          the re-run would overflow again -- unless the tree overflowed in the
          same run, whose cross verdict was measured on a truncated leaf set.

        Parameters
        ----------
        needs : np.ndarray
            The failed run's needs, max-ed over its steps and the mesh:
            ``fused_topology.NEEDS_FIELDS``, then the cross hook's.
        bits : int
            Its overflow bits, OR-ed over the steps and the mesh.
        walk_only : bool
            Re-plan the walk caps only (the second retry).

        Returns
        -------
        bool
            Whether anything grew; ``False``: the caller raises.
        """
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS, OVERFLOW_CROSS

        engine = self.engine
        needs_arr = np.asarray(needs, np.int64)
        local = needs_arr[: len(NEEDS_FIELDS)]
        cross = needs_arr[len(NEEDS_FIELDS) :]
        if engine._walk_overflowed_a_named_cap(local):
            # a deliberate bound the re-run would overflow again, tree or not
            return False
        tree_overflowed = bool(local[NEEDS_FIELDS.index("tree_overflow")])
        walk = engine._replan_walk_caps_from_needs(local)
        tree = False if walk_only else self._replan_tree(local)
        crossed = False
        if int(bits) & OVERFLOW_CROSS:
            # after the tree's re-plan, so the cross capacities can follow the
            # grown leaf capacity (the summary publishes one row per live leaf)
            caps = (
                None
                if self.cross_replan is None
                else self.cross_replan(cross, self.cross_caps)
            )
            if caps is not None:
                self.cross_caps = caps
                self.hook, self.near_sink = self._make_hook()
                crossed = True
            elif not tree_overflowed:
                # a cross capacity nothing re-plans: the re-run would overflow too
                return False
            # On an overflowed tree the cross verdict was measured on a truncated
            # leaf set (it feeds the summary and the near export): measured, the
            # cross near flag fired on every such step of a drift into octant
            # clumps and on none once the tree fitted. The tree re-plan's re-run
            # decides, as a second retry re-measures the walk's needs.
        if not (walk or tree or crossed):
            return False
        self._peak_floor = int(
            (engine._strict_fused_validated_caps or {}).get("peak_wavefront") or 0
        )
        return True

    def _replan_tree(self, local: np.ndarray) -> bool:
        """The TREE half of :meth:`replan`, against the installed plan's bounds.

        ``engine._replan_tree_from_needs`` is the single card's: it compares the
        needs with the process registry and declines under an installed plan. On
        the mesh the programs checked the merged PLAN (its level batch, its level
        loops), so the needs are compared with those, and the plan of the next
        prepare is widened through :attr:`level_floor`. The leaf capacity and the
        depth bound grow as on the single card.

        Parameters
        ----------
        local : np.ndarray
            ``fused_topology.NEEDS_FIELDS`` of the failed run.

        Returns
        -------
        bool
            Whether a bound grew.
        """
        from jaccpot.runtime.capacity_plan import (
            DEFAULT_LEAF_HEADROOM,
            auto_leaf_capacity,
        )
        from jaccpot.runtime.fmm_sweeps import _CELLS_DEPTH_HEADROOM
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS

        n = dict(zip(NEEDS_FIELDS, (int(v) for v in local)))
        if not n["tree_overflow"]:
            return False
        engine = self.engine
        plan = self._setup_or_raise().plan
        capacity = int(engine._tree_leaf_capacity)
        leaf_over = n["leaf_needed"] > capacity
        depth_over = n["depth_needed"] > min(
            int(plan.num_levels), int(plan.upward_num_levels)
        )
        width_over = n["widest_needed"] > int(plan.level_batch_width)
        if not (leaf_over or depth_over or width_over):
            return False
        if leaf_over:
            headroom = (
                getattr(engine, "_tree_leaf_capacity_auto", None)
                or DEFAULT_LEAF_HEADROOM
            )
            engine._tree_leaf_capacity = max(
                capacity, auto_leaf_capacity(n["leaf_needed"], float(headroom))
            )
        levels = max(
            int(plan.upward_num_levels),
            int(getattr(engine, "_cells_upward_num_levels", None) or 0),
            min(64, n["depth_needed"] + _CELLS_DEPTH_HEADROOM),
        )
        engine._cells_upward_num_levels = levels
        widest_floor, levels_floor = self.level_floor or (0, 0)
        self.level_floor = (
            max(int(widest_floor), n["widest_needed"]),
            max(int(levels_floor), levels),
        )
        engine.clear_prepared_state_cache()
        return True

    def raise_overflow(
        self, needs: np.ndarray, bits: int, *, retries: int, where: str
    ) -> None:
        """Raise the failed segment's capacity error.

        Parameters
        ----------
        needs : np.ndarray
            The failed run's needs (mesh-reduced).
        bits : int
            Its overflow bits (mesh-reduced).
        retries : int
            Retries the segment ran.
        where : str
            Which segment and how far it got, for the message.

        Raises
        ------
        RuntimeError
            Always: the walk / tree message of the single card
            (``engine._raise_scan_capacity_saturated``) when they overflowed, else
            the cross capacities' or the shard count's.
        """
        from jaccpot.runtime.fused_topology import (
            NEEDS_FIELDS,
            OVERFLOW_CROSS,
            OVERFLOW_FAIL_MASK,
        )

        bits = int(bits)
        local = np.asarray(needs, np.int64)[: len(NEEDS_FIELDS)]
        cross = (
            " The cross-domain capacities (CrossCapacities) overflowed "
            "(OVERFLOW_CROSS) and no cross re-plan took them: give the hook larger "
            "capacities, or a cross_replan."
            if bits & OVERFLOW_CROSS
            else ""
        )
        if bits & OVERFLOW_FAIL_MASK & ~OVERFLOW_CROSS:
            try:
                self.engine._raise_scan_capacity_saturated(local, retries=retries)
            except RuntimeError as exc:
                raise RuntimeError(f"mesh {where}: {exc}{cross}") from exc
        if cross:
            raise RuntimeError(
                f"mesh {where}: {cross.strip()} Bits {bits:#x}, cross needs "
                f"{np.asarray(needs)[len(NEEDS_FIELDS):].tolist()}; segment "
                f"retries run: {int(retries)}."
            )
        raise RuntimeError(
            f"mesh {where}: a device's live count exceeds the shard capacity (bits "
            f"{bits:#x}): the rows past it were cut."
        )
