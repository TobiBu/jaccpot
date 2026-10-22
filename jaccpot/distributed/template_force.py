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

GATE A1 (``tests/distributed/test_mesh_segment_runner.py``): with the same caps,
this route's force equals the stacked route's BITWISE on two forced CPU devices --
so the template's broadcasts reach no value of the force.

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
    reduce_flag_across_mesh,
)

__all__ = [
    "MeshForceSetup",
    "cross_needs",
    "make_mesh_cross_hook",
    "make_template_force_evaluator",
    "mesh_force_shard",
    "prepare_mesh_force",
    "setup_template_force",
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


def make_template_force_evaluator(
    setup: MeshForceSetup,
    *,
    mesh: Any,
    cross_hook: Optional[Callable[[Any], Any]] = None,
    cross_near_sink: Optional[dict] = None,
    axis_name: str = AXIS_NAME,
) -> Callable[[Array, Array, Array], tuple]:
    """A jitted ``shard_map`` force on the template route.

    The counterpart of :func:`~jaccpot.distributed.fused.make_fused_force_evaluator`
    with the stacked prepared states replaced by the setup's template: the same
    call signature and outputs, so it drops into
    :class:`~jaccpot.distributed.rollout.FusedRollout`. The plan is installed
    around the build and every call.

    Parameters
    ----------
    setup : MeshForceSetup
        From :func:`prepare_mesh_force`.
    mesh : Any
        The 1-D mesh.
    cross_hook : Optional[Callable[[Any], Any]]
        The cross hook (``ndev > 1``).
    cross_near_sink : Optional[dict]
        Its near sink.
    axis_name : str
        Mesh axis.

    Returns
    -------
    Callable[[Array, Array, Array], tuple]
        ``(positions, masses, num_valid) -> (acceleration, overflow)``, flat as in
        ``make_fused_force_evaluator``; ``overflow`` is one replicated boolean (any
        verdict bit on any device, or a count above the capacity).
    """
    from jax.sharding import PartitionSpec as P

    from jaccpot.runtime.capacity_plan import fused_capacity_plan_overrides
    from jaccpot.runtime.fused_topology import overflow_ok

    _warn_if_slow_ragged_exchange(mesh)
    engine = setup.provider.engine

    def body(positions: Array, masses: Array, num_valid: Array) -> tuple:
        # a count above the shard's capacity cannot be real particles: clamp it so
        # it cannot index past the arrays, and report it
        cap_rows = int(positions.shape[0])
        count_over_cap = num_valid[0] > cap_rows
        live = jnp.minimum(num_valid[0], cap_rows)
        acceleration, bits, _ = mesh_force_shard(
            setup,
            positions,
            masses,
            live,
            cross_hook=cross_hook,
            cross_near_sink=cross_near_sink,
            axis_name=axis_name,
        )
        failed = ~overflow_ok(bits) | count_over_cap
        return acceleration, reduce_flag_across_mesh(failed, axis_name=axis_name)

    spec = P(axis_name)
    with fused_capacity_plan_overrides(setup.plan):
        compiled = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(spec, spec, spec),
                out_specs=(spec, P()),
                check_vma=False,
            )
        )
    traced: list[bool] = []

    def force(positions: Array, masses: Array, num_valid: Array) -> tuple:
        with fused_capacity_plan_overrides(setup.plan):
            if not traced:
                # the trace records its own walk caps; a previous trace's must not
                # be read as this one's (strict_run_v2 does the same)
                engine._strict_fused_traced_caps = None
                traced.append(True)
            return compiled(positions, masses, num_valid)

    return force


def setup_template_force(
    solver: Any,
    mesh: Any,
    state: Any,
    *,
    leaf_size: int,
    max_order: int,
    theta: float,
    cross_caps: Any = None,
    near_theta: Optional[float] = None,
    axis_name: str = AXIS_NAME,
) -> tuple[Callable[..., tuple], Any, Optional[dict]]:
    """:func:`~jaccpot.distributed.rollout.setup_fused_force` on the template route.

    The same contract -- ``(force, merged plan, merged walk caps)`` with
    ``force(positions, masses, count, ids) -> (acceleration, overflow)`` -- built
    from caps-only prepares and one shape template instead of stacked eager states
    (module docstring).

    Parameters
    ----------
    solver : Any
        The configured ``FastMultipoleMethod`` (large-N production profile).
    mesh : Any
        The 1-D mesh.
    state : Any
        A :class:`~jaccpot.distributed.rollout.ShardState`; only its positions,
        masses and counts are read.
    leaf_size : int
        Leaf target.
    max_order : int
        Expansion order.
    theta : float
        Opening angle.
    cross_caps : Any
        :class:`~jaccpot.distributed.cross.CrossCapacities`; ``None`` takes defaults.
    near_theta : Optional[float]
        Passed to the cross hook.
    axis_name : str
        Mesh axis.

    Returns
    -------
    tuple[Callable[..., tuple], Any, Optional[dict]]
        ``(force, merged plan, merged walk caps)``.

    Raises
    ------
    NotImplementedError
        If ``solver`` asks for Dehnen's error criterion; raised before any shard is
        prepared.
    """
    _refuse_error_criterion(solver, entry="setup_template_force")
    setup = prepare_mesh_force(
        solver,
        mesh,
        state.positions,
        state.masses,
        state.count,
        leaf_size=leaf_size,
        max_order=max_order,
        theta=theta,
    )
    hook, sink = make_mesh_cross_hook(
        solver, mesh, theta=theta, cross_caps=cross_caps, near_theta=near_theta
    )
    evaluator = make_template_force_evaluator(
        setup, mesh=mesh, cross_hook=hook, cross_near_sink=sink, axis_name=axis_name
    )

    def force(
        positions: Array, masses: Array, count: Array, ids: Optional[Array]
    ) -> tuple:
        del ids  # the rows come back in their input order
        return evaluator(positions, masses, count)

    return force, setup.plan, setup.walk_caps


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
