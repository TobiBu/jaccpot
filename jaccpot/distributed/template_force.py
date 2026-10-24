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

The self-gravity POTENTIAL, the energy diagnostic's (:func:`mesh_potential_shard`,
:meth:`FusedMeshForce.potential_shard`), is the single card's energy program on this
route: the force's rebuild, ``FusedForceProvider.potential`` on it (the local near and
far fields, and the cross far pairs through the local expansions their M2L wrote),
and the cross near pairs' potential from the cross near term's kernel launch, its
fourth lane in its default meaning
(:func:`jaccpot.distributed.fused._add_cross_near_potential`). A program of its own:
no force program changes.

:class:`FusedMeshForce` hands this route to the segment runner
(:mod:`jaccpot.distributed.segment`), with the segment retry's re-plan. Its CROSS
capacities are the library's by default (``cross_caps="auto"``): a first guess from
the merged leaf capacity and the worst shard's live leaves
(``CrossCapacities.for_shard``), measured by the runner's start force and sized from
what every capacity needed (``cross.plan_cross_capacities``); a retry grows them
with ``CrossCapacities.replanned`` from the needs vector
(``cross.CROSS_NEEDS_FIELDS``) the runner carries after ``NEEDS_FIELDS``, the summary
following a grown leaf capacity and clamped to what the tree can hold. No bench
formula and no harness environment.

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
    _add_cross_near_potential,
    _add_cross_near_term,
    _fold_cross_flags,
    _warn_if_slow_ragged_exchange,
    global_mesh_bounds,
    host_mesh_bounds,
)

__all__ = [
    "FusedMeshForce",
    "MeshForceSetup",
    "make_mesh_cross_hook",
    "mesh_force_shard",
    "mesh_potential_shard",
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
    live_leaves : int
        The most leaves holding a particle in any device's eager tree: the worst
        shard the cross capacities' first guess is sized for
        (``CrossCapacities.for_shard``).
    force_scale_seed : Optional[Array]
        ``mac_type='dehnen_error'``: eq (16b)'s ``f_b`` per row from each device's
        caps-only prepare (``_fused_force_scale_seed``: a geometric walk over the
        device's own particles, near pairs exact, far pairs as monopoles), flat
        ``(ndev * cap,)`` and mesh-sharded, zero on padding rows: the first
        force's thresholds. ``None`` without the criterion.
    """

    provider: Any
    plan: Any
    walk_caps: Optional[dict]
    leaf_capacity: int
    box: tuple
    live_leaves: int = 0
    force_scale_seed: Optional[Array] = None

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


def _refuse_unsupported_criterion(solver: Any, *, entry: str) -> None:
    """Refuse the error-criterion variants the template route does not carry.

    ``mac_type='dehnen_error'`` (eq 16a at a fixed order on the flat walk, the
    galaxy preset's MAC) runs on this route since S12: the force scale rides the
    segment carry and the cross walks evaluate the same eq (16a) from the
    published nodes' tables. What the single card's fused lane does not run on its
    flat walk either is refused here, before anything is prepared: the refuted
    ``dehnen_theta`` (a per-node opening angle folded into the radii, which the
    cross walks would not apply) and the paper criterion with an adaptive order.
    The stacked route (``setup_fused_force``) keeps refusing every criterion
    (:func:`~jaccpot.distributed.fused._refuse_error_criterion`).

    Parameters
    ----------
    solver : Any
        The ``FastMultipoleMethod`` (or its runtime impl).
    entry : str
        The entry point's name, for the message.

    Raises
    ------
    NotImplementedError
        For a paper-style criterion that is not the flat walk's eq (16a).
    """
    engine = getattr(solver, "_impl", solver)
    uses_criterion = getattr(engine, "_uses_paper_style_traversal_policy", None)
    if uses_criterion is None or not uses_criterion():
        return
    if engine._flat_walk_criterion_active():
        return
    raise NotImplementedError(
        f"{entry}: mac_type={str(getattr(engine, 'mac_type', ''))!r} with "
        f"adaptive_order={bool(getattr(engine, 'adaptive_order', False))} is not "
        "the flat walk's eq (16a) (mac_type='dehnen_error' at a fixed order), the "
        "only error criterion the fused multi-GPU lane carries. Use "
        "mac_type='dehnen_error' (the galaxy preset's) or a geometric MAC."
    )


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
        If the solver asks for an error criterion other than the flat walk's eq
        (16a) (:func:`_refuse_unsupported_criterion`); raised before anything is
        prepared.
    RuntimeError
        If the profile is not the large-N production profile, the shard capacity is
        not in ``JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET``, or the walk is not the
        flat walk (no caps-only prepare: its lists are built as it walks).
    ValueError
        If the leaves are buckets (a template rebuild needs cell leaves).
    """
    from jaccpot.distributed.rollout import (
        _leaf_counts,
        _mesh_devices,
        _pad_onto_live_row,
    )
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

    _refuse_unsupported_criterion(solver, entry="prepare_mesh_force")
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
    # the solver's configured box shape (preset="galaxy": a cube), as the single
    # card's box and every traced force below (mesh_force_shard) build it
    cubic = bool(getattr(engine, "_tree_cubic_bounds", False))
    box = host_mesh_bounds([np.asarray(p) for p in pos_d], counts_host, cubic=cubic)
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

    plans, walks, live_leaves, seeds = [], [], [], []
    stub_caps: Optional[PreparedCaps] = None
    scale_carry = bool(engine._flat_walk_criterion_active())
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
        live_leaves.append(_leaf_counts(caps.stub)[1])
        if scale_carry:
            with jax.default_device(devices[d]):
                seed_d = _force_scale_seed(
                    engine, tree, pos_d[d].dtype, counts_host[d], cap
                )
            seeds.append(jax.device_put(seed_d, devices[d]))
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
    seed = None
    if scale_carry:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        seed = jax.make_array_from_single_device_arrays(
            (ndev * cap,), NamedSharding(mesh, P(*mesh.axis_names)), seeds
        )
        # the last prepare's seed, which the single card's runner would start from
        engine._last_force_scale_particles = None
        engine._last_force_scale_nodes = None
    return MeshForceSetup(
        provider=provider,
        plan=plan,
        walk_caps=walk_caps,
        leaf_capacity=int(engine._tree_leaf_capacity),
        box=tuple(np.asarray(b) for b in box),
        live_leaves=int(max(live_leaves)),
        force_scale_seed=seed,
    )


def _force_scale_seed(
    engine: Any, tree: Any, dtype: Any, count: int, cap: int
) -> Array:
    """One device's eq (16b) seed in its input row order, zero on padding.

    The caps-only prepare leaves its ``_fused_force_scale_seed`` (the device's own
    particles: a geometric flat walk, near pairs exact, far pairs as monopoles) on
    the engine in its eager tree's order, as the single card's particle carry
    reads it (``strict_run_v2``'s ``_force_scale_seed``).

    On the mesh it is a lower bound of the single card's seed: eq (16b)'s ``f_b``
    is a sum of positive terms, and this one leaves out every other domain's, so
    the first force's thresholds are lower and its acceptance stricter, never
    looser -- the contract the single card's seed already has (it errs low).

    Parameters
    ----------
    engine : Any
        The runtime engine, right after this device's prepare.
    tree : Any
        That prepare's eager tree (its ``particle_indices``: sorted row -> input
        row, over every row of the shard).
    dtype : Any
        The positions' dtype (the carry's).
    count : int
        Live rows of the shard.
    cap : int
        Rows of the shard.

    Returns
    -------
    Array
        ``(cap,)`` on the prepare's device.

    Raises
    ------
    RuntimeError
        If the prepare left no seed of this shard's size.
    """
    fb_sorted = getattr(engine, "_last_force_scale_particles", None)
    perm = jnp.asarray(tree.particle_indices)
    if fb_sorted is None or int(fb_sorted.shape[0]) != cap or int(perm.shape[0]) != cap:
        raise RuntimeError(
            "mac_type='dehnen_error': the caps-only prepare left no eq (16b) seed "
            f"for a shard of {cap} rows"
        )
    seed = jnp.zeros((cap,), dtype).at[perm].set(jnp.asarray(fb_sorted, dtype))
    return jnp.where(jnp.arange(cap) < count, seed, jnp.zeros((), dtype))


def mesh_force_shard(
    setup: MeshForceSetup,
    positions: Array,
    masses: Array,
    live: Array,
    *,
    cross_hook: Optional[Callable[..., Any]] = None,
    cross_near_sink: Optional[dict] = None,
    axis_name: str = AXIS_NAME,
    force_scale: Optional[Array] = None,
) -> tuple[Array, ...]:
    """One device's self-gravity on the template route (inside ``shard_map``).

    The global box (:func:`~jaccpot.distributed.fused.global_mesh_bounds`, one
    all-reduce), the provider's rebuild from the materialised template over the
    live rows with the cross hook between the sweeps, its evaluation, and the cross
    NEAR term. The same refresh and evaluation as
    :func:`~jaccpot.distributed.fused.fused_force_step`, from a template instead of
    a prepared state.

    Under ``mac_type='dehnen_error'`` the rebuild takes the previous force's eq
    (16b) ``f_b`` (``force_scale``) for eq (16a)'s thresholds -- the local walk's
    and, through the hook, the cross walks' -- and the force returns the next
    one: the local near and far halves (the far half with the cross far pairs'
    monopole terms, in the downward sweep) plus the cross near term's share.

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
    cross_hook : Optional[Callable[..., Any]]
        The cross-domain hook (``None`` on one device).
    cross_near_sink : Optional[dict]
        The hook's near sink.
    axis_name : str
        Mesh axis.
    force_scale : Optional[Array]
        ``(cap,)`` the previous force's ``f_b`` in input row order (or the
        prepare's seed); required under the criterion, ``None`` without it.

    Returns
    -------
    tuple[Array, ...]
        ``(acceleration, bits, needs)``: ``(cap, 3)`` in input row order (padding
        rows meaningless), the ``uint32`` overflow bits (with
        ``OVERFLOW_LEVEL_PLAN`` and ``OVERFLOW_CROSS``) and ``int32`` needs:
        ``fused_topology.NEEDS_FIELDS`` (whose ``tree_overflow`` also counts a
        level-plan overflow) followed by ``cross.CROSS_NEEDS_FIELDS``
        (:func:`_hook_cross_needs`); with ``force_scale``, then the new ``(cap,)``
        ``f_b`` (zero on padding rows).
    """
    provider = setup.provider
    topology = _shard_rebuild(
        provider,
        positions,
        masses,
        live,
        cross_hook=cross_hook,
        cross_near_sink=cross_near_sink,
        axis_name=axis_name,
        force_scale=force_scale,
    )
    f_b = None
    if force_scale is None:
        acceleration = _add_cross_near_term(
            provider.engine,
            topology.prepared,
            jnp.asarray(provider.accelerations(topology)),
            cross_near_sink,
        )
    else:
        acc_local, fb_local = provider.accelerations(topology, force_scale=True)
        acceleration, f_b = _add_cross_near_term(
            provider.engine,
            topology.prepared,
            jnp.asarray(acc_local),
            cross_near_sink,
            force_scale=jnp.asarray(fb_local, force_scale.dtype),
        )
        live_rows = jnp.arange(int(positions.shape[0])) < live
        f_b = jnp.where(live_rows, f_b, jnp.zeros((), f_b.dtype))
    bits, needs = _shard_verdict(topology, cross_hook, cross_near_sink)
    if f_b is not None:
        return acceleration, bits, needs, f_b
    return acceleration, bits, needs


def _shard_rebuild(
    provider: Any,
    positions: Array,
    masses: Array,
    live: Array,
    *,
    cross_hook: Optional[Callable[..., Any]],
    cross_near_sink: Optional[dict],
    axis_name: str,
    force_scale: Optional[Array],
) -> Any:
    """One device's topology in the global box, the cross hook between the sweeps.

    The rebuild :func:`mesh_force_shard` and :func:`mesh_potential_shard` share:
    the near sink emptied (a RE-TRACE must not read the previous trace's tracers,
    as in ``fused_force_step``), the global box (one all-reduce), and the
    provider's rebuild over the live rows, with ``force_scale`` for eq (16a)'s
    thresholds under the criterion.

    Parameters
    ----------
    provider : Any
        The setup's :class:`~jaccpot.runtime.fused_provider.FusedForceProvider`.
    positions : Array
        ``(cap, 3)`` this device's positions.
    masses : Array
        ``(cap,)`` masses, zero on padding.
    live : Array
        Scalar live row count.
    cross_hook : Optional[Callable[..., Any]]
        The cross-domain hook (``None`` on one device).
    cross_near_sink : Optional[dict]
        The hook's near sink.
    axis_name : str
        Mesh axis.
    force_scale : Optional[Array]
        ``(cap,)`` eq (16b)'s ``f_b`` in input row order under the criterion, else
        ``None``.

    Returns
    -------
    Any
        The rebuilt :class:`~jaccpot.runtime.fused_topology.FusedTopology`.
    """
    if cross_near_sink is not None:
        cross_near_sink.clear()
    bounds = global_mesh_bounds(
        positions,
        num_valid=live,
        axis_name=axis_name,
        cubic=bool(getattr(provider.engine, "_tree_cubic_bounds", False)),
    )
    return provider.rebuild(
        positions,
        masses,
        bounds=bounds,
        num_valid=live,
        cross_hook=cross_hook,
        **({} if force_scale is None else {"force_scale": force_scale}),
    )


def _shard_verdict(
    topology: Any, cross_hook: Any, cross_near_sink: Optional[dict]
) -> tuple[Array, Array]:
    """One device's capacity bits and needs of a rebuilt topology (this trace's).

    The provider's bits with this lane's two (``OVERFLOW_LEVEL_PLAN``,
    ``OVERFLOW_CROSS``), and the walk's, tree's and cross needs.

    Parameters
    ----------
    topology : Any
        From :func:`_shard_rebuild`, in this trace.
    cross_hook : Any
        The hook that ran in the rebuild, or ``None``.
    cross_near_sink : Optional[dict]
        Its near sink.

    Returns
    -------
    tuple[Array, Array]
        ``(bits, needs)`` as :func:`mesh_force_shard` returns them.
    """
    from jaccpot.runtime.capacity_plan import plan_level_overflow
    from jaccpot.runtime.fused_topology import (
        NEEDS_FIELDS,
        OVERFLOW_CROSS,
        OVERFLOW_LEVEL_PLAN,
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
    needs = jnp.concatenate([needs, _hook_cross_needs(cross_hook)])
    return bits, needs


def mesh_potential_shard(
    setup: MeshForceSetup,
    positions: Array,
    masses: Array,
    live: Array,
    *,
    cross_hook: Optional[Callable[..., Any]] = None,
    cross_near_sink: Optional[dict] = None,
    axis_name: str = AXIS_NAME,
    force_scale: Optional[Array] = None,
) -> tuple[Array, Array, Array]:
    """One device's self-gravity POTENTIAL on the template route (inside ``shard_map``).

    The single card's energy program (``ProductionRunner.potential_energy``: a
    rebuild at the positions, then ``FusedForceProvider.potential`` on it) on the
    mesh: :func:`mesh_force_shard`'s rebuild -- the global box, the cross hook
    between the sweeps and, under ``mac_type='dehnen_error'``, eq (16a)'s
    thresholds from ``force_scale`` (the carried ``f_b``), so the lists are the
    force's at these positions -- and ``phi_i = -G sum_{j != i} m_j psi(r_ij)``
    with the force's pair kernel over every device's particles:

    * the local near and far fields: ``FusedForceProvider.potential`` (the near
      kernel's fourth lane, the self pair excluded; the far field's L2P value
      from the local expansions, ``evaluate_large_n_potential``);
    * the cross FAR pairs: their M2L writes the same local expansions (through the
      walk's CSR since S11-C; the flat lanes' concatenated list), which that L2P
      reads, as the force's L2P reads them;
    * the cross NEAR pairs: the cross near term's kernel launch with its fourth
      lane as the potential (:func:`~jaccpot.distributed.fused._add_cross_near_potential`),
      added in input row order after the local potential, as the force adds its
      acceleration.

    Its own program (no force program changes): nothing of the acceleration is
    evaluated but the cross near kernel's three force lanes, which share the launch.
    No capacity of its own: the rebuild's are the force's, and the cross near term
    reads the near sink the force reads.

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
    cross_hook : Optional[Callable[..., Any]]
        The cross-domain hook (``None`` on one device).
    cross_near_sink : Optional[dict]
        The hook's near sink.
    axis_name : str
        Mesh axis.
    force_scale : Optional[Array]
        ``(cap,)`` eq (16b)'s ``f_b`` in input row order (the force's carried one);
        required under the criterion, ``None`` without it.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(phi, bits, needs)``: ``(cap,)`` in input row order and the positions'
        dtype, zero on padding rows; the ``uint32`` overflow bits and the ``int32``
        needs of the rebuild, as :func:`mesh_force_shard` returns them (the
        potential is meaningless unless ``overflow_ok(bits)``).
    """
    provider = setup.provider
    topology = _shard_rebuild(
        provider,
        positions,
        masses,
        live,
        cross_hook=cross_hook,
        cross_near_sink=cross_near_sink,
        axis_name=axis_name,
        force_scale=force_scale,
    )
    phi = _add_cross_near_potential(
        provider.engine,
        topology.prepared,
        jnp.asarray(provider.potential(topology)),
        cross_near_sink,
    )
    live_rows = jnp.arange(int(positions.shape[0])) < live
    phi = jnp.where(live_rows, phi, jnp.zeros((), phi.dtype))
    bits, needs = _shard_verdict(topology, cross_hook, cross_near_sink)
    return phi, bits, needs


def _hook_cross_needs(cross_hook: Any) -> Array:
    """The cross hook's needs vector of this trace (:func:`~jaccpot.distributed.cross.cross_needs`).

    One protocol for both routes: :data:`~jaccpot.distributed.cross.CROSS_NEEDS_FIELDS`,
    read off the hook's ``needs_sink`` inside the trace that ran it. Zeros without a
    hook (one device: nothing crossed) and for a control exchange that measures none
    (``JACCPOT_CROSS_SYMMETRIC=0`` and the like): there a cross overflow finds no
    need to grow and the retry raises, naming the flag.

    Parameters
    ----------
    cross_hook : Any
        The hook, or ``None``.

    Returns
    -------
    Array
        ``int32[len(CROSS_NEEDS_FIELDS)]``, this device's.
    """
    from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS, cross_needs

    if cross_hook is None or "needs" not in (
        getattr(cross_hook, "needs_sink", None) or {}
    ):
        return jnp.zeros((len(CROSS_NEEDS_FIELDS),), jnp.int32)
    return jnp.asarray(cross_needs(cross_hook), jnp.int32)


def make_mesh_cross_hook(
    solver: Any,
    mesh: Any,
    *,
    theta: float,
    cross_caps: Any = None,
    near_theta: Optional[float] = None,
    axis_name: str = AXIS_NAME,
    record: Optional[dict] = None,
) -> tuple[Optional[Callable[..., Any]], Optional[dict]]:
    """The cross hook and its near sink for this mesh, as ``setup_fused_force`` builds them.

    Under ``mac_type='dehnen_error'`` (``_flat_walk_criterion_active``) the hook
    runs the criterion too (``make_cross_hook(error_criterion=True)``): the cross
    walks accept by the same eq (16a) as the local walk.

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
    record : Optional[dict]
        The hook's diagnostics record (tracers of the trace that ran it).

    Returns
    -------
    tuple[Optional[Callable[..., Any]], Optional[dict]]
        ``(hook, near_sink)``; ``(None, None)`` on one device.
    """
    from jaccpot.distributed.cross import make_cross_hook
    from jaccpot.distributed.rollout import _mesh_devices

    ndev = len(_mesh_devices(mesh))
    if ndev < 2:
        return None, None
    engine = getattr(solver, "_impl", solver)
    sink: dict = {}
    hook = make_cross_hook(
        ndev=ndev,
        theta=theta,
        caps=cross_caps,
        near_sink=sink,
        near_theta=near_theta,
        axis_name=axis_name,
        record=record,
        # the local walks' floor (a compact kernel's support): the cross walks must
        # keep their far pairs as far apart
        separation_floor=engine._walk_separation_floor(),
        error_criterion=bool(engine._flat_walk_criterion_active()),
    )
    return hook, sink


def _cross_by_field(needs: np.ndarray) -> dict:
    """The cross half of a runner needs vector, keyed by ``CROSS_NEEDS_FIELDS``.

    Parameters
    ----------
    needs : np.ndarray
        ``NEEDS_FIELDS`` then ``CROSS_NEEDS_FIELDS`` (host).

    Returns
    -------
    dict
        ``{field: value}`` of the cross half.
    """
    from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS
    from jaccpot.runtime.fused_topology import NEEDS_FIELDS

    cross = np.asarray(needs, np.int64).reshape(-1)[len(NEEDS_FIELDS) :]
    return dict(zip(CROSS_NEEDS_FIELDS, (int(v) for v in cross)))


class FusedMeshForce:
    """The fused lane's force for :class:`~jaccpot.distributed.segment.MeshSegmentRunner`.

    Holds the template-route setup (:func:`prepare_mesh_force`) and the cross hook,
    and hands the runner what it traces and keys its programs on: :meth:`shard`
    inside its ``shard_map`` bodies, under :meth:`context` (the merged plan
    installed), cached under :meth:`key`. :meth:`prepare` runs at the runner's
    start and again at a re-planned segment's start.

    Under ``mac_type='dehnen_error'`` (:attr:`scale_carry`) the force takes and
    returns eq (16b)'s per-particle ``f_b`` (:meth:`shard`), the runner carries it,
    and :attr:`force_scale_seed` is the prepare's seed for the first force.

    Parameters
    ----------
    solver : Any
        The ``FastMultipoleMethod`` (large-N production profile, cell leaves, the
        flat walk; a geometric MAC or ``mac_type='dehnen_error'``).
    mesh : Any
        The 1-D mesh.
    leaf_size : int
        Leaf target.
    max_order : int
        Expansion order.
    theta : Optional[float]
        Opening angle; ``None`` takes the solver's.
    cross_caps : Any
        ``"auto"`` (the default): sized in the library -- a first guess from the
        merged leaf capacity and the worst shard's live leaves
        (``CrossCapacities.for_shard``), measured by the runner's start force and
        sized from what it needed (``cross.plan_cross_capacities``, via
        :meth:`plan_cross`). A :class:`~jaccpot.distributed.cross.CrossCapacities`:
        used as given; ``None``: its constructor defaults. Either way a segment
        retry grows them from the measured needs.
    near_theta : Optional[float]
        Passed to the cross hook.
    cross_replan : Optional[Callable[[np.ndarray, Any], Any]]
        The cross re-plan of a segment retry: ``cross_replan(cross_needs, caps) ->
        CrossCapacities | None``, with ``cross_needs`` the
        :data:`~jaccpot.distributed.cross.CROSS_NEEDS_FIELDS` vector max-ed over
        the failed run's steps and the mesh. ``None`` (the default): the library's,
        ``caps.replanned(cross_needs, leaf_capacity=<the leaf capacity, grown when
        the tree was>)``.
    level_floor : Optional[tuple[int, int]]
        ``(widest, levels)``: every prepare's level plan covers at least a level
        of ``widest`` nodes and ``levels`` levels (:func:`prepare_mesh_force`'s
        ``grow_levels``). A tree re-plan raises it; give a re-planned run's
        :attr:`level_floor` to a fresh force to compile the same plan.
    axis_name : str
        Mesh axis.
    cross_record : Optional[dict]
        The cross hook's diagnostics record (``make_cross_hook(record=...)``):
        tracers of the trace that ran the hook, for a probe that reads them inside
        it. ``None`` (default): none.

    Raises
    ------
    NotImplementedError
        If the solver asks for an error criterion other than the flat walk's eq
        (16a) (:func:`_refuse_unsupported_criterion`); raised before anything is
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
        cross_caps: Any = "auto",
        near_theta: Optional[float] = None,
        cross_replan: Optional[Callable[[np.ndarray, Any], Any]] = None,
        level_floor: Optional[tuple[int, int]] = None,
        axis_name: str = AXIS_NAME,
        cross_record: Optional[dict] = None,
    ) -> None:
        from jaccpot.distributed.cross import CrossCapacities

        _refuse_unsupported_criterion(solver, entry="FusedMeshForce")
        if isinstance(cross_caps, str) and cross_caps != "auto":
            raise ValueError(
                f"cross_caps={cross_caps!r}: a CrossCapacities, None or 'auto'"
            )
        self.solver = solver
        self.engine = getattr(solver, "_impl", solver)
        self.mesh = mesh
        self.leaf_size = int(leaf_size)
        self.max_order = int(max_order)
        self.theta = theta
        #: sized in the library at the first prepare (plan_cross), else as given
        self.cross_auto = isinstance(cross_caps, str)
        self.cross_caps = (
            None
            if self.cross_auto
            else (CrossCapacities() if cross_caps is None else cross_caps)
        )
        #: the cross needs the library's sizing measured (host), once sized
        self.cross_needs_measured: Optional[np.ndarray] = None
        self.near_theta = near_theta
        self.cross_replan = cross_replan
        self.level_floor = None if level_floor is None else tuple(level_floor)
        self.axis_name = axis_name
        self.cross_record = cross_record
        self.setup: Optional[MeshForceSetup] = None
        #: the walk's peak wavefront a re-plan keeps for the next prepare
        self._peak_floor = 0
        #: the segment's first failure had lower-bound cross needs (a second retry
        #: re-measures them)
        self._cross_lower_first = False
        self.hook, self.near_sink = (None, None)
        if not self.cross_auto:
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
            record=self.cross_record,
        )

    @property
    def scale_carry(self) -> bool:
        """``mac_type='dehnen_error'``: the force takes and returns ``f_b``.

        Returns
        -------
        bool
            The engine's ``_flat_walk_criterion_active()``.
        """
        return bool(self.engine._flat_walk_criterion_active())

    @property
    def force_scale_seed(self) -> Optional[Array]:
        """The last prepare's eq (16b) seed (flat, mesh-sharded), under the criterion.

        Returns
        -------
        Optional[Array]
            ``(ndev * cap,)``, or ``None`` without the criterion.
        """
        return self._setup_or_raise().force_scale_seed

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
            ``len(NEEDS_FIELDS) + len(CROSS_NEEDS_FIELDS)``.
        """
        from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS

        return len(NEEDS_FIELDS) + len(CROSS_NEEDS_FIELDS)

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
        if self.cross_auto and self.cross_caps is None and len(self.devices) > 1:
            from jaccpot.distributed.cross import CrossCapacities

            # the library's first guess; plan_cross measures and right-sizes it
            self.cross_caps = CrossCapacities.for_shard(
                leaf_capacity=self.setup.leaf_capacity,
                ndev=len(self.devices),
                leaf_width=self.leaf_size,
                live_leaves=self.setup.live_leaves,
            )
            self.hook, self.near_sink = self._make_hook()

    @property
    def devices(self) -> list:
        """The mesh's devices, in mesh order.

        Returns
        -------
        list
            The devices.
        """
        from jaccpot.distributed.rollout import _mesh_devices

        return _mesh_devices(self.mesh)

    @property
    def needs_cross_sizing(self) -> bool:
        """Whether :meth:`plan_cross` still has to size the cross capacities.

        Returns
        -------
        bool
            ``cross_caps="auto"`` on a mesh of two or more devices, not yet sized.
        """
        return (
            self.cross_auto
            and self.hook is not None
            and self.cross_needs_measured is None
        )

    def plan_cross(self, measure: Callable[[], np.ndarray]) -> None:
        """Size ``cross_caps="auto"``: measure the first guess, size from the needs.

        ``cross.plan_cross_capacities`` with the runner's measurement: ``measure()``
        runs the start force at the current positions on the current hook and
        returns its cross needs, reduced over the mesh. While their overflow bit is
        set the capacities are re-planned and measured again; the clean
        measurement sizes every capacity (``from_needs``: 1.5x its need, the
        summary clamped to the tree). Each measurement is a compile of the start
        force. A no-op once sized, with explicit capacities, and on one device.

        Parameters
        ----------
        measure : Callable[[], np.ndarray]
            Runs the force with :attr:`hook` and returns the host cross needs.
        """
        from jaccpot.distributed.cross import plan_cross_capacities

        if not self.needs_cross_sizing:
            return

        def _measure(caps: Any) -> np.ndarray:
            self.cross_caps = caps
            self.hook, self.near_sink = self._make_hook()
            return np.asarray(measure())

        caps, needs = plan_cross_capacities(
            _measure,
            self.cross_caps,
            leaf_capacity=self._setup_or_raise().leaf_capacity,
            leaf_size=self.leaf_size,
            # a lower-bound need doubles per round: a first guess sized for a
            # Plummer sphere can be several doublings short (the disc + bulge)
            max_rounds=6,
        )
        self.cross_caps = caps
        self.cross_needs_measured = np.asarray(needs)
        self.hook, self.near_sink = self._make_hook()

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
        self,
        positions: Array,
        masses: Array,
        live: Array,
        ids: Array,
        force_scale: Optional[Array] = None,
    ) -> tuple[Array, ...]:
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
        force_scale : Optional[Array]
            ``(cap,)`` the previous force's ``f_b`` (:attr:`scale_carry` only).

        Returns
        -------
        tuple[Array, ...]
            ``(acceleration, bits, needs)``, then ``f_b`` with ``force_scale``.
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
            force_scale=force_scale,
        )

    def potential_shard(
        self,
        positions: Array,
        masses: Array,
        live: Array,
        ids: Array,
        force_scale: Optional[Array] = None,
    ) -> tuple[Array, Array]:
        """One device's self-gravity potential (:func:`mesh_potential_shard`; inside ``shard_map``).

        Called like :meth:`shard`: inside the caller's ``shard_map``, under
        :meth:`context`, with :meth:`before_trace` before a fresh trace. The
        energy's ``W = 1/2 sum_i m_i phi_i`` (summed over the mesh) counts every
        pair once.

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
        force_scale : Optional[Array]
            ``(cap,)`` the force's carried ``f_b`` (:attr:`scale_carry` only): the
            rebuild's thresholds, so the lists are the force's.

        Returns
        -------
        tuple[Array, Array]
            ``(phi, bits)``: ``(cap,)`` in the positions' dtype, zero on padding
            rows, and the ``uint32`` overflow bits of its rebuild (the potential is
            meaningless unless ``overflow_ok`` of their mesh-wide OR holds).
        """
        del ids
        phi, bits, _ = mesh_potential_shard(
            self._setup_or_raise(),
            positions,
            masses,
            live,
            cross_hook=self.hook,
            cross_near_sink=self.near_sink,
            axis_name=self.axis_name,
            force_scale=force_scale,
        )
        return phi, bits

    # ------------------------------------------------------------------
    # The segment retry.
    # ------------------------------------------------------------------

    def retry_mode(self, needs: np.ndarray, retries: int, tree_first: bool) -> Any:
        """How a failed segment may be retried: the single card's policy, plus one.

        ``engine._segment_retry_mode`` on the local needs: at most two retries, the
        second for the walk only after a first that re-planned the tree
        (``JACCPOT_STRICT_SEGMENT_RETRY=0`` turns them off). And a second retry
        after a first whose CROSS needs were lower bounds (``walk_lower_bound`` /
        ``pairs_lower_bound``: a cross walk or the summary lost pairs, so the
        re-plan from them can fall short; the re-run's needs are exact), when the
        re-run overflowed the cross field again: ``"walk"`` -- the local walk and
        the cross capacities, not the tree.

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

        needs_arr = np.asarray(needs, np.int64)
        local = needs_arr[: len(NEEDS_FIELDS)]
        cross = _cross_by_field(needs_arr)
        mode = self.engine._segment_retry_mode(local, int(retries), bool(tree_first))
        if int(retries) == 0:
            self._cross_lower_first = bool(
                cross["overflow"]
                and (cross["walk_lower_bound"] or cross["pairs_lower_bound"])
            )
        elif (
            mode is None
            and int(retries) == 1
            and self._cross_lower_first
            and cross["overflow"]
            and self.engine._segment_retry_mode(local, 0, False) is not None
        ):
            # the retry is on (the engine's policy allows a first), and this second
            # failure re-measured the cross needs a lower bound sized
            mode = "walk"
        return mode

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
        * the CROSS capacities, after the tree (so they follow its grown leaf
          capacity): :attr:`cross_replan`, by default ``caps.replanned(needs,
          leaf_capacity=...)`` -- monotone, the summary clamped to what the tree can
          hold (``max_cells`` <= the leaf capacity, ``max_summary_nodes`` <= 2x - 1:
          a larger summary does not trace). When the cross flag fired, and whenever
          the leaf capacity grew: then the summary's needs are raised to the tree's
          live leaves (``leaf_needed``, exact past the overflow), which a summary
          counted on a truncated tree cannot tell. A cross flag that grows nothing
          declines -- a mismatch between the
          two ends of an exchange, or a flag no cross capacity explains -- unless
          the tree overflowed in the same run, whose cross verdict was measured on
          a truncated leaf set (the re-run on a tree that fits decides).

        Parameters
        ----------
        needs : np.ndarray
            The failed run's needs, max-ed over its steps and the mesh:
            ``fused_topology.NEEDS_FIELDS``, then ``cross.CROSS_NEEDS_FIELDS``.
        bits : int
            Its overflow bits, OR-ed over the steps and the mesh.
        walk_only : bool
            Re-plan the walk caps (and the cross capacities) only: the second
            retry.

        Returns
        -------
        bool
            Whether anything grew; ``False``: the caller raises.
        """
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS, OVERFLOW_CROSS

        engine = self.engine
        needs_arr = np.asarray(needs, np.int64)
        local = needs_arr[: len(NEEDS_FIELDS)]
        if engine._walk_overflowed_a_named_cap(local):
            # a deliberate bound the re-run would overflow again, tree or not
            return False
        local_n = dict(zip(NEEDS_FIELDS, (int(v) for v in local)))
        tree_overflowed = bool(local_n["tree_overflow"])
        capacity_before = int(engine._tree_leaf_capacity)
        walk = engine._replan_walk_caps_from_needs(local)
        tree = False if walk_only else self._replan_tree(local)
        crossed = False
        cross_fired = bool(int(bits) & OVERFLOW_CROSS)
        leaves_grew = int(engine._tree_leaf_capacity) > capacity_before
        if self.cross_caps is not None and (cross_fired or leaves_grew):
            # a grown leaf capacity re-plans the cross field too, flag or not: the
            # summary publishes one row per live leaf, so it has to follow
            grown = self._replanned_cross(
                needs_arr,
                leaf_needed=local_n["leaf_needed"] if leaves_grew else 0,
                tree_overflowed=tree_overflowed,
            )
            if grown is not None and vars(grown) != vars(self.cross_caps):
                self.cross_caps = grown
                self.hook, self.near_sink = self._make_hook()
                crossed = True
            elif cross_fired and not tree_overflowed:
                return False
        if not (walk or tree or crossed):
            return False
        self._peak_floor = int(
            (engine._strict_fused_validated_caps or {}).get("peak_wavefront") or 0
        )
        return True

    def _replanned_cross(
        self, needs: np.ndarray, *, leaf_needed: int, tree_overflowed: bool = False
    ) -> Any:
        """The cross capacities a retry runs with (:meth:`replan`'s cross half).

        Parameters
        ----------
        needs : np.ndarray
            The failed run's full needs vector.
        leaf_needed : int
            The live leaves the trees needed when the leaf capacity grew, else 0.
        tree_overflowed : bool
            The failed run's tree overflowed: its walks ran on a truncated tree
            (the nodes the level loops dropped carry no multipole or geometry), so
            of its cross needs only the summary's are kept -- the walk's and the
            exchange's read that tree, not a demand, and a re-plan never narrows
            what it grew. The re-run on a tree that fits measures them.

        Returns
        -------
        Any
            :attr:`cross_replan`'s answer, or ``cross_caps.replanned(...)``.
        """
        from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS

        cross = np.array(needs[len(NEEDS_FIELDS) :], np.int64)
        if tree_overflowed:
            keep = [
                CROSS_NEEDS_FIELDS.index(k) for k in ("max_cells", "max_summary_nodes")
            ]
            summary = cross[keep]
            cross[:] = 0
            cross[keep] = summary
        if leaf_needed:
            # one summary cell per live leaf, and the summary tree over them
            cells = CROSS_NEEDS_FIELDS.index("max_cells")
            nodes = CROSS_NEEDS_FIELDS.index("max_summary_nodes")
            cross[cells] = max(int(cross[cells]), int(leaf_needed))
            cross[nodes] = max(int(cross[nodes]), 2 * int(leaf_needed) - 1)
        if self.cross_replan is not None:
            return self.cross_replan(cross, self.cross_caps)
        return self.cross_caps.replanned(
            cross,
            leaf_capacity=int(self.engine._tree_leaf_capacity),
            leaf_size=self.leaf_size,
        )

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
        cross = ""
        if bits & OVERFLOW_CROSS:
            need = _cross_by_field(np.asarray(needs, np.int64))
            why = (
                "the two ends of an exchange disagreed on a size (mismatch): no "
                "capacity covers that"
                if need["mismatch"]
                else "no cross capacity needed more than it holds, so the flag is "
                "not a cross capacity's"
            )
            cross = (
                " The cross-domain field overflowed (OVERFLOW_CROSS) and its "
                f"re-plan grew nothing: {why}. Cross needs {need}; capacities "
                f"{None if self.cross_caps is None else vars(self.cross_caps)}."
            )
        if bits & OVERFLOW_FAIL_MASK & ~OVERFLOW_CROSS:
            try:
                self.engine._raise_scan_capacity_saturated(local, retries=retries)
            except RuntimeError as exc:
                raise RuntimeError(f"mesh {where}: {exc}{cross}") from exc
        if cross:
            raise RuntimeError(
                f"mesh {where}: {cross.strip()} Bits {bits:#x}; segment retries "
                f"run: {int(retries)}."
            )
        raise RuntimeError(
            f"mesh {where}: a device's live count exceeds the shard capacity (bits "
            f"{bits:#x}): the rows past it were cut."
        )
