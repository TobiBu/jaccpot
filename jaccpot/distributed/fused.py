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

Static shapes come from a :class:`~jaccpot.runtime.capacity_plan.FusedCapacityPlan`
rather than from the process-level registry, because inside ``shard_map`` no
eager prepare ever fills that registry -- see the plan module for what fails
silently when it is cold.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

__all__ = [
    "AXIS_NAME",
    "fused_force_step",
    "global_mesh_bounds",
    "reduce_flag_across_mesh",
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
    return refreshed, jnp.asarray(acceleration)
