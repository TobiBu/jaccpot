"""The single-GPU fused lane, per device, under one ``shard_map``.

The distributed FMM (:mod:`jaccpot.distributed.fmm`) is a separately assembled
per-device pipeline: a bucket radix tree rebuilt in-trace, the traced dual-tree
wavefront walk, by-level JAX cascades, the pure-JAX rotate-scale M2L and the
rectangle near-field kernel. None of the sub-10 ms kernels reach it, and it runs
0.2-1.3 M particles/s against the fused lane's 17 M.

This module runs the *fused* lane's own pipeline on each device instead. The
per-device force is exactly what ``strict_run_v2`` computes per endpoint --
refresh the static template, then evaluate -- with three differences that the
mesh forces and that are the whole content of this module:

1. **The Morton box is global, not per shard.** The fused step infers its box
   from the positions it is handed; per device that gives every card its own
   frame and makes cell codes incomparable across domains. The box is taken from
   :func:`yggdrax.distributed.partition.global_bounds` and passed in explicitly.
   The refresh already accepts one, so nothing downstream changes.
2. **Dead padding rows are cut.** A shard is capacity-padded, and
   ``num_valid`` keeps those rows out of the leaves.
3. **Overflow leaves the map reduced.** A capacity that saturated on one device
   must be visible to all of them, or one card silently truncates its lists
   while its peers look clean.

**Array convention, and it is a trap.** ``shard_map`` does NOT remove the mapped
axis: with ``in_specs=P("gpus")`` a ``(ndev, cap, 3)`` input arrives inside the
body as ``(1, cap, 3)``, so a ``min(axis=0)`` reduces the device axis instead of
the particles and every subsequent collective compares whole shards elementwise
-- silently, with plausible shapes. This module therefore follows the convention
:mod:`jaccpot.distributed.fmm` already uses: particle arrays are passed FLAT as
``(ndev * cap, ...)`` so a device sees ``(cap, ...)``. Per-device pytrees that
genuinely need a leading device axis (the prepared states) are sliced with
``[0]`` on entry.

Static shapes come from a :class:`~jaccpot.runtime.capacity_plan.FusedCapacityPlan`
rather than from the process-level registry, because inside ``shard_map`` no
eager prepare ever fills that registry -- see the plan module for what fails
silently when it is cold.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

__all__ = [
    "AXIS_NAME",
    "assemble_prepared_states",
    "fused_force_step",
    "global_mesh_bounds",
    "make_fused_force_evaluator",
    "reduce_flag_across_mesh",
    "stack_prepared_states",
]

#: Mesh axis the distributed lanes agree on (matches ``yggdrax.distributed``).
AXIS_NAME = "gpus"


def global_mesh_bounds(
    positions: Array,
    *,
    num_valid: Optional[Array] = None,
    axis_name: str = AXIS_NAME,
    pad: float = 1e-6,
) -> tuple[Array, Array]:
    """Morton box covering every device's particles.

    Every device must encode into the SAME frame or its cell codes mean nothing
    to any other device: leaf codes stop being comparable, aligned domain
    boundaries stop lining up, and the imported forest indexes a different space
    from the local tree. One all-reduce per force buys all of that.

    Dead padding rows are excluded by folding them onto a live position rather
    than by masking with infinities, so the reduction stays finite even on a
    device whose shard is entirely padding.

    Parameters
    ----------
    positions : Array
        This device's padded positions ``(cap, 3)``.
    num_valid : Optional[Array]
        Number of leading live rows; ``None`` treats every row as live.
    axis_name : str
        Mesh axis to reduce over.
    pad : float
        Fractional slack added to each side, so a particle never sits exactly on
        the box edge.

    Returns
    -------
    tuple[Array, Array]
        ``(min_corner, max_corner)``, identical on every device.
    """
    pos = jnp.asarray(positions)
    if num_valid is not None:
        live = jnp.arange(pos.shape[0]) < jnp.asarray(num_valid)
        # Fold dead rows onto row 0: it is a real position, so it can only
        # tighten nothing and widen nothing. Masking with +-inf would make a
        # fully padded shard reduce to inf and poison the whole mesh.
        pos = jnp.where(live[:, None], pos, pos[0][None, :])
    lo = jax.lax.pmin(jnp.min(pos, axis=0), axis_name)
    hi = jax.lax.pmax(jnp.max(pos, axis=0), axis_name)
    span = jnp.maximum(hi - lo, jnp.asarray(pad, lo.dtype))
    slack = span * jnp.asarray(pad, lo.dtype)
    return lo - slack, hi + slack


def reduce_flag_across_mesh(flag: Array, *, axis_name: str = AXIS_NAME) -> Array:
    """OR a boolean across the mesh: one device's overflow is everyone's.

    A capacity guard read per device lets one card truncate its interaction
    lists while its peers report clean -- the same shape of defect as a halo
    that silently drops its payload, where every invariant looks healthy over a
    wrong force.

    Reduce ONCE per step over a single scalar. OR the local flags together
    first; nine separate collectives buy nothing.

    **Never feed the reduced flag back into a per-device count.** The fused
    walk saturates its own pair count when it overflows, and driving that from
    the mesh flag would corrupt a healthy device's lists for a step that is
    already known bad.

    Parameters
    ----------
    flag : Array
        This device's boolean scalar.
    axis_name : str
        Mesh axis to reduce over.

    Returns
    -------
    Array
        Boolean scalar, identical on every device.
    """
    as_int = jnp.asarray(flag, jnp.int32).astype(jnp.int32)
    return jax.lax.pmax(as_int, axis_name) > 0


def fused_force_step(
    solver: Any,
    prepared: Any,
    positions: Array,
    masses: Array,
    *,
    bounds: tuple[Array, Array],
    leaf_size: int,
    max_order: int,
    theta: Optional[float] = None,
    num_valid: Optional[Array] = None,
    cross_hook: Optional[Any] = None,
    cross_near_sink: Optional[dict] = None,
    cross_record: Optional[dict] = None,  # filled by the hook; read by the caller
) -> tuple[Any, Array]:
    """One fused-lane force: refresh the static template, then evaluate.

    This is the seam ``strict_run_v2`` calls per endpoint
    (``_refresh_and_evaluate_endpoint``), lifted out of its velocity-Verlet
    scan. The two entry points either side of it are both wrong for a force
    evaluator: ``strict_run_v2`` bundles the integrator, and
    ``strict_fused_prepared_eval_fn`` is a benchmarking seam that deliberately
    omits the refresh.

    Parameters
    ----------
    solver : Any
        A ``FastMultipoleMethod`` on the large-N production profile with the
        fused mode active.
    prepared : Any
        The device's ``LargeNPreparedState``; supplies the static template.
    positions : Array
        Padded positions ``(cap, 3)`` for this device.
    masses : Array
        Padded masses ``(cap,)``; dead rows carry zero.
    bounds : tuple[Array, Array]
        The GLOBAL Morton box, from :func:`global_mesh_bounds`. Passing ``None``
        here would re-infer a per-shard box and silently de-align the mesh, so
        it is required.
    leaf_size : int
        Leaf width the template was built with.
    max_order : int
        Expansion order.
    theta : Optional[float]
        Opening-angle override.
    num_valid : Optional[Array]
        Live row count of this shard.
    cross_hook : Optional[Any]
        Called between the upward and downward sweeps with the upward artifacts --
        the one point where the cross-domain exchange belongs, because the
        multipoles exist there and the downward sweep has not consumed them.
        ``None`` (the default) means the force is bit-identical to a run without it.

    Returns
    -------
    tuple[Any, Array]
        ``(prepared_refreshed, acceleration)``; the acceleration is in input row
        order, dead rows included and meaningless.

    Raises
    ------
    RuntimeError
        If the refresh rejects the state (topology or profile mismatch).
    """
    from jaccpot.runtime._large_n_pipeline import evaluate_large_n_state

    # ``FastMultipoleMethod`` is a facade; the refresh and the evaluator both
    # live on the runtime impl it delegates to. Accept either so callers can
    # pass whichever object they already hold.
    engine = getattr(solver, "_impl", solver)
    if bounds is None:
        raise ValueError(
            "fused_force_step needs an explicit global box: inferring it per "
            "device gives every card its own Morton frame and de-aligns the mesh."
        )
    # Clear before the hook refills it: on a RE-TRACE the dict would still hold
    # the previous trace's tracers, and reading those below would leak a value
    # across jaxprs.
    if cross_near_sink is not None:
        cross_near_sink.clear()
    if cross_record is not None:
        cross_record.clear()
    refreshed = engine._refresh_large_n_same_topology(
        prepared,
        jnp.asarray(positions),
        jnp.asarray(masses),
        bounds=bounds,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        theta=theta,
        runtime_overrides_override=None,
        fused_device_mode=True,
        num_valid=num_valid,
        cross_hook=cross_hook,
    )
    if refreshed is None:
        raise RuntimeError(
            "fused refresh failed: topology/profile mismatch on this device"
        )
    acceleration = evaluate_large_n_state(
        engine,
        refreshed,
        target_indices=None,
        return_potential=False,
        max_acc_derivative_order=0,
    )
    acceleration = jnp.asarray(acceleration)

    # The cross NEAR field is ADDED here rather than interleaved. A far contribution
    # arrives as a local expansion and would need a second L2L cascade, which is why
    # that half goes between the sweeps; a near contribution is a direct sum with no
    # cascade behind it, so one small extra kernel is cheaper than plumbing it
    # through the prepared state. Gravity is linear in the sources, so adding is
    # exact.
    if cross_near_sink:
        from jaccpot.distributed.cross import cross_near_acceleration

        tree = refreshed.tree
        idx = jnp.asarray(refreshed.nearfield_leaf_particle_indices)
        msk = jnp.asarray(refreshed.nearfield_leaf_particle_mask)
        psort = jnp.asarray(refreshed.positions_sorted)
        msort = jnp.asarray(refreshed.masses_sorted)
        # The near term assumes pool row r IS leaf node (num_internal + r). That
        # held in the unit test because the test BUILT the pool that way; here the
        # pool comes from the lane and the assumption is unverified. If it is
        # wrong the contribution lands on the wrong particles, which adds noise
        # rather than signal -- exactly what a rising error looks like.
        if cross_record is not None:
            _ni = int(jnp.asarray(tree.left_child).shape[0])
            _nr = jnp.asarray(tree.node_ranges)
            _L = int(idx.shape[0])
            _rows = jnp.arange(_L)
            _first = jnp.where(msk[:, 0], idx[:, 0], -1)
            _want = jnp.where(
                _ni + _rows < _nr.shape[0],
                _nr[jnp.clip(_ni + _rows, 0, _nr.shape[0] - 1), 0],
                -1,
            )
            _has = msk.any(axis=1)
            cross_record["leaf_pool_mismatch"] = jnp.sum(_has & (_first != _want))
            cross_record["leaf_pool_rows"] = jnp.sum(_has)
        safe = jnp.clip(idx, 0, psort.shape[0] - 1)
        leaf_pos = jnp.where(msk[..., None], psort[safe], 0.0)
        leaf_mass = jnp.where(msk, msort[safe], 0.0)
        # `leaf_particle_indices` indexes the MORTON-SORTED array -- that is what
        # the leaf_pool_mismatch check above establishes, since `node_ranges` is
        # defined in sorted order. The lane's acceleration is in the caller's
        # original order. Scattering the near term by the sorted index would add
        # it to the wrong particles: right magnitudes, wrong rows, which shows up
        # as a RISE in the error rather than as anything obviously broken.
        orig = jnp.asarray(tree.particle_indices)
        idx_orig = jnp.where(msk, orig[jnp.clip(idx, 0, orig.shape[0] - 1)], 0)
        if cross_record is not None:
            # if this permutation were the identity the remap would be a no-op
            cross_record["perm_nonidentity"] = jnp.sum(
                orig != jnp.arange(orig.shape[0], dtype=orig.dtype)
            )
        acceleration = acceleration + cross_near_acceleration(
            cross_near_sink,
            leaf_pos,
            leaf_mass,
            msk,
            idx_orig,
            int(psort.shape[0]),
            int(jnp.asarray(tree.left_child).shape[0]),
            softening_sq=jnp.asarray(engine.softening, acceleration.dtype) ** 2,
            G=jnp.asarray(engine.G, acceleration.dtype),
        )
    return refreshed, acceleration


def stack_prepared_states(states: "list[Any]") -> Any:
    """Stack per-device prepared states into one pytree with a leading device axis.

    Each device needs its OWN prepared state -- its own tree, multipoles and
    interaction lists -- so beyond a single device the state cannot ride as a
    closure constant. It becomes a ``shard_map`` input, which requires every leaf
    to share a shape across devices.

    Measured at N = 2x10^5 on two Morton domains: the states already stack, with
    an identical treedef, 55 leaves each and no shape disagreement, because the
    interaction-list capacities are fixed by configuration rather than measured
    per shard. What *does* differ per shard is the level-shape plan (widths 3205
    and 3460, depths 44 and 50 on those two domains), which is why
    :func:`~jaccpot.runtime.capacity_plan.merge_plans` exists and why it is not
    optional.

    Parameters
    ----------
    states : list[Any]
        One prepared state per device, in mesh order.

    Returns
    -------
    Any
        The same pytree with every leaf gaining a leading axis of ``len(states)``.

    Raises
    ------
    ValueError
        If ``states`` is empty, the structures differ, or any leaf's shape or
        dtype disagrees -- naming the offending leaf, because "cannot stack" with
        55 anonymous leaves is not an actionable message.
    """
    flats, reference = _check_stackable(states)
    stacked = [
        jnp.stack([jnp.asarray(f[i]) for f in flats]) for i in range(len(flats[0]))
    ]
    return jax.tree_util.tree_unflatten(reference, stacked)


def assemble_prepared_states(
    states: "list[Any]", mesh: Any, *, axis_name: str = AXIS_NAME
) -> Any:
    """Per-device prepared states as ONE pytree of mesh-sharded arrays.

    The same result as :func:`stack_prepared_states` followed by a ``device_put``
    with ``P(axis_name)``, but built from per-device pieces with
    ``jax.make_array_from_single_device_arrays``: no device ever holds the other
    devices' states, and nothing is re-sharded on each call (an unsharded stacked
    state lives on the default device and the jitted ``shard_map`` would split it
    every time). The same construction works when each process only holds its own
    devices' states, which is the one-process-per-GPU form.

    Parameters
    ----------
    states : list[Any]
        One prepared state per mesh device, in mesh order.
    mesh : Any
        The 1-D ``jax.sharding.Mesh`` the evaluator runs on.
    axis_name : str
        The mesh axis.

    Returns
    -------
    Any
        The pytree with every leaf a ``(ndev, ...)`` array sharded over the mesh.

    Raises
    ------
    ValueError
        If ``states`` does not hold one state per mesh device, or the states
        cannot be stacked (see :func:`stack_prepared_states`).
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = list(np.asarray(mesh.devices).reshape(-1))
    if len(states) != len(devices):
        raise ValueError(
            f"assemble_prepared_states got {len(states)} states for a mesh of "
            f"{len(devices)} devices"
        )
    flats, reference = _check_stackable(states)
    sharding = NamedSharding(mesh, P(axis_name))
    leaves = []
    for i in range(len(flats[0])):
        parts = [
            jax.device_put(jnp.asarray(flats[d][i])[None], devices[d])
            for d in range(len(devices))
        ]
        shape = (len(devices),) + tuple(parts[0].shape[1:])
        leaves.append(jax.make_array_from_single_device_arrays(shape, sharding, parts))
    return jax.tree_util.tree_unflatten(reference, leaves)


def _check_stackable(states: "list[Any]") -> tuple:
    """Flatten per-device states and check they stack; see :func:`stack_prepared_states`.

    Parameters
    ----------
    states : list[Any]
        One prepared state per device.

    Returns
    -------
    tuple
        ``(flats, treedef)``: the flattened leaves per device and the shared treedef.

    Raises
    ------
    ValueError
        If ``states`` is empty, the structures differ, or a leaf's shape disagrees.
    """
    if not states:
        raise ValueError("stack_prepared_states needs at least one state")
    flats, treedefs = zip(*(jax.tree_util.tree_flatten(s) for s in states))
    reference = treedefs[0]
    for device, treedef in enumerate(treedefs[1:], start=1):
        if treedef != reference:
            raise ValueError(
                f"prepared state {device} has a different pytree structure from "
                "device 0; every device must be prepared with the same "
                "configuration (leaf_capacity, order, caps)."
            )
    paths = [
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(states[0])[0]
    ]
    for device, flat in enumerate(flats[1:], start=1):
        for index, (a, b) in enumerate(zip(flats[0], flat)):
            shape_a = getattr(a, "shape", ())
            shape_b = getattr(b, "shape", ())
            if shape_a != shape_b:
                name = paths[index] if index < len(paths) else f"leaf #{index}"
                raise ValueError(
                    f"prepared states disagree on {name}: device 0 has "
                    f"{shape_a}, device {device} has {shape_b}. Every static "
                    "shape must cover the worst device -- size it from the "
                    "capacity plan rather than from each shard's own measurement."
                )
    return flats, reference


def make_fused_force_evaluator(
    solver: Any,
    prepared_stacked: Any,
    *,
    mesh: Any,
    plan: Any,
    leaf_size: int,
    max_order: int,
    theta: Optional[float] = None,
    axis_name: str = AXIS_NAME,
    cross_hook: Optional[Any] = None,
    cross_near_sink: Optional[dict] = None,
    cross_record: Optional[dict] = None,
    cross_record_keys: tuple = (),
) -> Callable[[Array, Array, Array], tuple]:
    """A jitted ``shard_map`` force: the fused lane per device, one program.

    The capacity plan is installed around the BUILD, not around the call: the
    static shapes are read at trace time. Its fingerprint belongs in whatever
    caches the returned callable, since a ContextVar does not retrace by itself.

    Parameters
    ----------
    solver : Any
        The fused-lane solver (or its runtime impl).
    prepared_stacked : Any
        Per-device prepared states from :func:`stack_prepared_states`.
    mesh : Any
        A 1-D device mesh whose axis is ``axis_name``.
    plan : Any
        The merged :class:`~jaccpot.runtime.capacity_plan.FusedCapacityPlan`.
    leaf_size : int
        Leaf width the templates were built with.
    max_order : int
        Expansion order.
    theta : Optional[float]
        Opening-angle override.
    axis_name : str
        Mesh axis name.

    Returns
    -------
    Callable
        ``(positions, masses, num_valid) -> (acceleration, overflow)``. Positions
        are ``(ndev * cap, 3)`` and masses ``(ndev * cap,)`` -- FLAT, see the
        module docstring on why not ``(ndev, cap, ...)``; ``num_valid`` is
        ``(ndev,)``. The acceleration comes back flat too, dead rows included and
        meaningless. ``overflow`` is one replicated boolean: a capacity that
        saturated on ANY device.
    """
    from jax.sharding import PartitionSpec as P

    from jaccpot.runtime.capacity_plan import fused_capacity_plan_overrides

    def body(prepared: Any, positions: Array, masses: Array, num_valid: Array) -> tuple:
        # shard_map keeps the mapped axis at size 1; strip it so the rest of the
        # body sees exactly what the single-device lane sees.
        prepared_local = jax.tree_util.tree_map(lambda leaf: leaf[0], prepared)
        # A count above the shard's capacity cannot be real particles; clamp so a
        # bad count cannot index past the arrays, and report it as an overflow.
        cap_rows = int(positions.shape[0])
        count_over_cap = num_valid[0] > cap_rows
        live = jnp.minimum(num_valid[0], cap_rows)
        bounds = global_mesh_bounds(positions, num_valid=live, axis_name=axis_name)
        refreshed, acceleration = fused_force_step(
            solver,
            prepared_local,
            positions,
            masses,
            cross_hook=cross_hook,
            cross_near_sink=cross_near_sink,
            cross_record=cross_record,
            bounds=bounds,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            theta=theta,
            num_valid=live,
        )
        local = _local_overflow(refreshed, getattr(solver, "_impl", solver))
        local = _fold_cross_flags(
            local | count_over_cap, cross_hook, cross_near_sink, cross_record
        )
        flag = reduce_flag_across_mesh(local, axis_name=axis_name)
        if not cross_record_keys:
            return acceleration, flag
        # The record holds TRACERS under jit, so a count can only be read by
        # leaving the mapped region as an OUTPUT. The key ORDER is fixed by the
        # caller so `out_specs` stays static.
        diag = tuple(
            # float64, not int: counts up to 2^53 are exact in it, and a
            # diagnostic that is a DISTANCE would be truncated to 0 by an int
            # cast -- which would make a real effect look like a no-op.
            jnp.asarray(cross_record[k]).reshape(1).astype(jnp.float64)
            for k in cross_record_keys
        )
        return acceleration, flag, diag

    out_specs = (P(axis_name), P())
    if cross_record_keys:
        out_specs = out_specs + (tuple(P(axis_name) for _ in cross_record_keys),)
    with fused_capacity_plan_overrides(plan):
        mapped = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(axis_name), P(axis_name), P(axis_name), P(axis_name)),
            out_specs=out_specs,
            check_vma=False,
        )
        compiled = jax.jit(mapped)

    def force(positions: Array, masses: Array, num_valid: Array) -> tuple:
        with fused_capacity_plan_overrides(plan):
            return compiled(prepared_stacked, positions, masses, num_valid)

    return force


def _fold_cross_flags(
    local: Array,
    cross_hook: Any,
    cross_near_sink: Optional[dict],
    cross_record: Optional[dict],
) -> Array:
    """OR every cross-domain overflow flag of this trace into the local flag.

    The cross buffers saturate independently of the local lane's, and their flags
    are tracers the caller cannot read host-side, so folding them in here is the
    only way they reach the caller. ``cross_hook.flag_sink`` is the hook's
    ALWAYS-ON channel and does not depend on a diagnostics record, which the
    production and timing paths do not pass; the near sink and the record are
    read as well, for hooks that predate the sink.

    Parameters
    ----------
    local : Array
        This device's flag so far.
    cross_hook : Any
        The hook, or ``None``.
    cross_near_sink : Optional[dict]
        The near-half sink, or ``None``.
    cross_record : Optional[dict]
        The diagnostics record, or ``None``.

    Returns
    -------
    Array
        Boolean scalar, ``True`` when anything saturated.
    """
    flag = jnp.asarray(local, jnp.bool_)
    sink = getattr(cross_hook, "flag_sink", None)
    if sink:
        for value in sink.values():
            flag = flag | jnp.asarray(value, jnp.bool_).any()
    if cross_near_sink and "overflow" in cross_near_sink:
        flag = flag | jnp.asarray(cross_near_sink["overflow"], jnp.bool_).any()
    if cross_record and "overflow" in cross_record:
        flag = flag | jnp.asarray(cross_record["overflow"], jnp.bool_).any()
    return flag


def _local_overflow(refreshed: Any, engine: Any = None) -> Array:
    """This device's saturation flag for the local (single-device) half of the force.

    Delegates to :func:`jaccpot.runtime.capacity_guard.fused_state_capacity_ok`, the
    same guard ``strict_run_v2`` uses. This function used to read
    ``leaf_capacity_overflow`` / ``walk_overflow`` / ``capacity_overflow``, none of
    which ``LargeNPreparedState`` has, so the flag was the constant ``False`` and a
    saturated walk or leaf partition on a mesh device went unreported.

    Parameters
    ----------
    refreshed : Any
        The refreshed prepared state.
    engine : Any
        The runtime engine whose recorded traced caps refine the check; ``None``
        uses the structural checks only.

    Returns
    -------
    Array
        Boolean scalar for this device, ``True`` when something saturated.
    """
    from jaccpot.nearfield._fast_lane import _nearfield_csr_lane_enabled
    from jaccpot.runtime.capacity_guard import (
        fused_state_capacity_ok,
        last_refresh_capacity_ok,
    )

    ok = fused_state_capacity_ok(
        refreshed,
        traced_caps=getattr(engine, "_strict_fused_traced_caps", None),
        rectangle_guard_active=not bool(_nearfield_csr_lane_enabled()),
    )
    # the RETURNED state carries the cached far-pair placeholder in the fresh-rebuild
    # mode, so the verdict on the lists this refresh built comes from the engine
    if engine is not None:
        ok = ok & last_refresh_capacity_ok(engine)
    return jnp.logical_not(ok)
