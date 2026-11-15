"""The reconstruction's observation operator on a multi-device mesh: the fused mesh lane.

Why this module exists, stated plainly, because the measurement that forced it
is a negative result about the alternative.

Section 7's secondary claim is that *the same optimisation runs sharded across
multiple GPUs, so parameter count is bounded by aggregate device memory rather
than a single device's*. The obvious way to try that is to shard the parameter
array and let XLA follow -- which is what
:func:`~jaccpot.applications.density_reconstruction.fit.run_fit`'s ``devices``
argument does. **It does not work, and the measurement says so** (September
2026, on the radix path of tag ``paper-i-sep2026``):

=========  =====================  ==============  ==================
devices    strong-scaling wall    mean gradient   largest N that ran
=========  =====================  ==============  ==================
1          950 s                  46.3 s          1 048 576
2          796 s                  49.0 s          1 048 576
4          825 s                  51.9 s          1 048 576
=========  =====================  ==============  ==================

The ceiling does not move and the per-gradient cost gets monotonically *worse*:
a force evaluation written over one full particle array all-gathers a sharded
input before the work starts. Parameter sharding distributes where the numbers
are *stored*, not where the force is *computed*, and it is the latter that sets
the ceiling.

What does distribute the computation is jaccpot's multi-GPU force, which
partitions the *sources* across devices. This module drives it differentiably.

The lane: ``FusedMeshForce`` and its gradient (WP-G)
----------------------------------------------------
Until 2026-11 this module drove ``jaccpot.distributed.fmm``'s
``make_force_evaluator``, the section-5 path of September. The production
multi-GPU force is now the fused mesh lane
(:class:`~jaccpot.distributed.template_force.FusedMeshForce`), and since WP-G
(#510) it is differentiable: :func:`~jaccpot.distributed.grad.make_mesh_grad_fn`
wraps ``FusedMeshForce.differentiable_shard`` in the mesh's ``shard_map``, so a
scalar loss built outside it differentiates with plain ``jax.grad``. On each
device the topology is rebuilt from the inputs' stopped gradients, the cross
layout of that trace is frozen, and the production kernels run at the live
positions with their reverses; the exchanges are never transposed. The WP-G gates
(``tests/distributed/test_mesh_grad_gates.py``) check it against central
differences, an exact direct sum and the single card on 2 and 4 devices.

What is frozen, and why it is the same contract as everywhere else
------------------------------------------------------------------
:meth:`DistributedForwardOperator.prepare` places the combined source + tracer
set on the mesh as Morton domains
(:func:`~jaccpot.distributed.rollout.decompose`), prepares the mesh force on that
placement and freezes two things for the epoch: **which device owns which
particle** (the row of each global id) and **which output rows are tracers**.
Within the epoch the force rebuilds each device's tree inside every call, as the
September path did; between epochs the placement is redone. Holding the readout
constant while the numeric pipeline is differentiated is hard decision 5, and
what the radix path's prepared state does. The mesh force returns rows in input
order (each device's rows as placed), so the readout is the placement's own
``ids``, with no output permutation to resolve.

The mesh force installs a capacity plan that must be active for every trace AND
call of its programs: :meth:`DistributedForwardOperator.program_context` returns
it, and ``run_fit`` enters it around the gradient step.

What this operator does not offer
---------------------------------
No leaf blocks, so no regularisers. The penalties in
:mod:`~jaccpot.applications.density_reconstruction.loss` are built on one global
leaf partition, which the per-device trees do not have. :meth:`leaf_blocks`
returns ``None`` and ``run_fit`` refuses a non-zero regularisation weight rather
than silently running an unregularised fit -- which matters, because fig17(d)
exists to show what dropping the regularisers does.

The mass-dependent criterion (``mac_type="dehnen_error"``, the fused lane's
production default) is refused for the reason
:func:`~jaccpot.applications.density_reconstruction.forward.assert_no_differentiated_mac`
gives; the default here is the geometric ``"dehnen"`` test.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, ContextManager, Dict, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = [
    "DistributedForwardOperator",
    "DistributedPartition",
    "make_distributed_forward_operator",
]


@dataclass(frozen=True)
class DistributedPartition:
    """One placement of the combined particle set on the mesh, readout frozen.

    The distributed analogue of a radix ``FMMPreparedState``: built host-side
    from concrete positions, held constant while the numeric pipeline is
    differentiated, and rebuilt between epochs.

    Attributes
    ----------
    force : Any
        The prepared ``FusedMeshForce``, with its gradient statics.
    grad_fn : Any
        :func:`~jaccpot.distributed.grad.make_mesh_grad_fn` of ``force``:
        ``f(positions, masses, counts, ids) -> (acceleration, bits)``.
    gid_index : jnp.ndarray
        ``(ndev * cap,)`` int32 global id of each mesh row, clamped at 0 so it
        can index; ``real_mask`` says which rows mean anything.
    real_mask : jnp.ndarray
        ``(ndev * cap, 1)`` bool, ``False`` on padding rows.
    pad_positions : jnp.ndarray
        ``(ndev * cap, 3)`` the placement's own positions, used verbatim for
        padding rows (each sits on a live particle of its device) so no
        cotangent reaches them.
    masses : jnp.ndarray
        ``(ndev * cap,)`` masses in row order, zero on padding. Constant: masses
        are frozen.
    counts : jnp.ndarray
        ``(ndev,)`` live rows per device.
    ids : jnp.ndarray
        ``(ndev * cap,)`` int32 global ids, ``-1`` on padding.
    tracer_rows : jnp.ndarray
        ``(M,)`` int32 mesh rows holding the tracers, ordered by global id.
    cap : int
        Rows per device.
    diagnostics : Dict[str, Any]
        The probe call's per-device bits (``0``: the lists are ones the reverses
        read) and live counts, so a run records whether any device failed.
    """

    force: Any
    grad_fn: Any
    gid_index: jnp.ndarray
    real_mask: jnp.ndarray
    pad_positions: jnp.ndarray
    masses: jnp.ndarray
    counts: jnp.ndarray
    ids: jnp.ndarray
    tracer_rows: jnp.ndarray
    cap: int
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class DistributedForwardOperator:
    """Positions -> accelerations at fixed tracers, on a device mesh.

    Mirrors :class:`~jaccpot.applications.density_reconstruction.forward.ForwardOperator`'s
    surface -- ``prepare``, ``evaluate_at_topology``, ``evaluate``,
    ``leaf_blocks``, ``program_context``, ``record`` -- so ``run_fit`` drives
    either without knowing which it has.

    Attributes
    ----------
    solver : Any
        The ``FastMultipoleMethod`` the mesh force evaluates with
        (:func:`~jaccpot.applications.density_reconstruction.forward.fused_solver`).
    mesh : Any
        The 1-D device mesh.
    num_devices : int
        Device count.
    tracer_positions : np.ndarray
        ``(M, 3)`` fixed observation points.
    source_mass : float
        The one mass every source carries.
    num_sources : int
        ``N``.
    softening : float
        Softening length, recorded because hard decision 6 says so.
    order : int
        Expansion order.
    leaf_size : int
        Cell-leaf size.
    theta : float
        Acceptance parameter.
    mac_type : str
        The MAC in use.
    working_dtype : str
        Working precision.
    capacity_headroom : float
        Rows per device over an even split, for the sample-sort's imbalance.
    cross_caps : Any
        ``CrossCapacities`` for the cross-domain exchange, or ``"auto"`` to have
        the mesh force size them at its first prepare.
    """

    solver: Any
    mesh: Any
    num_devices: int
    tracer_positions: np.ndarray
    source_mass: float
    num_sources: int
    softening: float
    order: int
    leaf_size: int
    theta: float
    mac_type: str
    working_dtype: str
    capacity_headroom: float
    cross_caps: Any

    @property
    def num_tracers(self: "DistributedForwardOperator") -> int:
        """Return ``M``.

        Returns
        -------
        int
            Tracer count.
        """
        return int(self.tracer_positions.shape[0])

    def combined_positions(
        self: "DistributedForwardOperator", source_positions: ArrayLike
    ) -> jnp.ndarray:
        """Append the fixed tracers to the (differentiated) sources.

        Parameters
        ----------
        source_positions : ArrayLike
            ``(N, 3)`` source positions.

        Returns
        -------
        jnp.ndarray
            ``(N + M, 3)``. The tracer block is a constant.
        """
        return jnp.concatenate(
            [
                jnp.asarray(source_positions, dtype=jnp.float64),
                jnp.asarray(self.tracer_positions, dtype=jnp.float64),
            ],
            axis=0,
        )

    def combined_masses(self: "DistributedForwardOperator") -> np.ndarray:
        """Return the combined mass vector: equal sources, zero tracers.

        Returns
        -------
        np.ndarray
            ``(N + M,)`` float64, host-side -- masses are frozen inputs.
        """
        return np.concatenate(
            [
                np.full((self.num_sources,), self.source_mass, dtype=np.float64),
                np.zeros((self.num_tracers,), dtype=np.float64),
            ]
        )

    def prepare(
        self: "DistributedForwardOperator", source_positions: Any
    ) -> DistributedPartition:
        """Place the particles on the mesh, prepare the force, freeze the readout.

        Parameters
        ----------
        source_positions : Any
            ``(N, 3)`` concrete positions.

        Returns
        -------
        DistributedPartition
            The placement, its prepared mesh force and gradient function, and
            the frozen tracer rows.

        Raises
        ------
        RuntimeError
            If the placement does not hold every particle exactly once, or the
            probe call reports a device whose lists the gradient's reverses
            cannot read (its gradient would be NaN, and every step of the epoch
            with it).
        """
        from jaccpot.distributed.grad import make_mesh_grad_fn
        from jaccpot.distributed.rollout import decompose
        from jaccpot.distributed.template_force import FusedMeshForce

        combined = np.asarray(
            self.combined_positions(np.asarray(source_positions)), dtype=np.float64
        )
        masses = self.combined_masses()
        total = int(combined.shape[0])
        ndev = int(self.num_devices)
        cap = int(math.ceil(total / ndev * float(self.capacity_headroom)))
        shard, _ = decompose(
            self.mesh,
            combined,
            np.zeros_like(combined),
            masses,
            cap=cap,
            num_samples=int(min(64, max(8, total // (4 * ndev)))),
        )
        ids = np.asarray(shard.ids).astype(np.int64)
        live_ids = ids[ids >= 0]
        if live_ids.size != total or np.unique(live_ids).size != total:
            raise RuntimeError(
                f"the mesh placement holds {live_ids.size} live rows "
                f"({np.unique(live_ids).size} distinct) for {total} particles"
            )

        force = FusedMeshForce(
            self.solver,
            self.mesh,
            leaf_size=int(self.leaf_size),
            max_order=int(self.order),
            theta=float(self.theta),
            cross_caps=self.cross_caps,
        )
        force.prepare(shard.positions, shard.masses, shard.count)
        force.with_grad_plan(shard.positions, shard.masses, shard.count)
        grad_fn = make_mesh_grad_fn(force)

        rows = np.where(ids >= int(self.num_sources))[0]
        rows = rows[np.argsort(ids[rows])]
        partition = DistributedPartition(
            force=force,
            grad_fn=grad_fn,
            gid_index=jnp.asarray(np.maximum(ids, 0).astype(np.int32)),
            real_mask=jnp.asarray((ids >= 0)[:, None]),
            pad_positions=jnp.asarray(shard.positions),
            masses=jnp.asarray(shard.masses),
            counts=jnp.asarray(shard.count),
            ids=jnp.asarray(shard.ids),
            tracer_rows=jnp.asarray(rows.astype(np.int32)),
            cap=cap,
            diagnostics={},
        )
        # One concrete call, to read every device's verdict on its lists.
        with force.context():
            force.before_trace()
            _acc, bits = jax.jit(grad_fn)(
                partition.pad_positions,
                partition.masses,
                partition.counts,
                partition.ids,
            )
        bits_host = np.asarray(bits).astype(np.int64).reshape(-1)
        partition.diagnostics.update(
            {
                "bits": bits_host.tolist(),
                "counts": np.asarray(shard.count).astype(np.int64).tolist(),
                "cap": cap,
                "failed": bool(np.any(bits_host != 0)),
            }
        )
        if partition.diagnostics["failed"]:
            raise RuntimeError(
                f"mesh force bits {bits_host.tolist()}: some device's lists are "
                "not ones the gradient's reverses read (overflow, a cross pair "
                "hash mismatch, or an asymmetric list); its gradient would be NaN"
            )
        return partition

    def program_context(
        self: "DistributedForwardOperator", partition: DistributedPartition
    ) -> ContextManager[Any]:
        """The mesh force's capacity plan, for every trace and call at ``partition``.

        Parameters
        ----------
        partition : DistributedPartition
            From :meth:`prepare`.

        Returns
        -------
        ContextManager[Any]
            ``partition.force.context()``.
        """
        return partition.force.context()

    def evaluate_at_topology(
        self: "DistributedForwardOperator",
        partition: DistributedPartition,
        source_positions: ArrayLike,
    ) -> jnp.ndarray:
        """Accelerations at the tracers, at this partition's frozen placement.

        Trace and call it under :meth:`program_context` (``run_fit`` does).

        Parameters
        ----------
        partition : DistributedPartition
            From :meth:`prepare`.
        source_positions : ArrayLike
            ``(N, 3)`` source positions, differentiated.

        Returns
        -------
        jnp.ndarray
            ``(M, 3)`` accelerations at the tracer positions, in tracer order.
        """
        combined = self.combined_positions(source_positions)
        # Live rows gather from the live parameters; padding rows keep the
        # placement's own coordinates, so no cotangent reaches them.
        rows = jnp.where(
            partition.real_mask,
            combined[partition.gid_index],
            partition.pad_positions,
        )
        # This runs at trace time: a new program must not read the walk caps
        # record another trace left on the engine.
        partition.force.before_trace()
        accelerations, _bits = partition.grad_fn(
            rows, partition.masses, partition.counts, partition.ids
        )
        return accelerations[partition.tracer_rows]

    def evaluate(
        self: "DistributedForwardOperator", source_positions: Any
    ) -> jnp.ndarray:
        """Re-place from ``source_positions`` and evaluate.

        Parameters
        ----------
        source_positions : Any
            ``(N, 3)`` concrete positions.

        Returns
        -------
        jnp.ndarray
            ``(M, 3)`` accelerations at the tracers.
        """
        partition = self.prepare(source_positions)
        sources = jnp.asarray(source_positions)
        with self.program_context(partition):
            return jax.jit(lambda s: self.evaluate_at_topology(partition, s))(sources)

    def leaf_blocks(
        self: "DistributedForwardOperator", partition: Optional[DistributedPartition]
    ) -> None:
        """Return ``None``: this path exposes no global leaf partition.

        Parameters
        ----------
        partition : Optional[DistributedPartition]
            Unused; present so the signature matches the single-card operators'.

        Returns
        -------
        None
            The regularisers are built on one global leaf partition, which the
            per-device trees do not have. ``run_fit`` refuses a non-zero
            regularisation weight against this operator rather than silently
            running an unregularised fit.
        """
        del partition
        return None

    def record(self: "DistributedForwardOperator") -> Dict[str, Any]:
        """Return the operator configuration for a results JSON.

        Returns
        -------
        Dict[str, Any]
            Counts, accuracy settings, the lane, and -- named explicitly -- the
            fact that this is the distributed force evaluation and not parameter
            sharding. A wall-clock from one is not comparable with the other.
        """
        engine = getattr(self.solver, "_impl", self.solver)
        caps = self.cross_caps
        return {
            "N": int(self.num_sources),
            "M": int(self.num_tracers),
            "softening": float(self.softening),
            "softening_kernel": getattr(engine, "softening_kernel", None),
            "order": int(self.order),
            "leaf_size": int(self.leaf_size),
            "theta": float(self.theta),
            "mac_type": self.mac_type,
            "working_dtype": self.working_dtype,
            "num_devices": int(self.num_devices),
            "capacity_headroom": float(self.capacity_headroom),
            "cross_caps": (caps if isinstance(caps, str) else dict(vars(caps))),
            "lane": "fused_mesh",
            "partitioner": "sfc",
            "sharding_mode": "distributed_force_evaluation",
        }


def make_distributed_forward_operator(
    *,
    tracer_positions: Any,
    source_mass: float,
    num_sources: int,
    num_devices: int,
    softening: float,
    order: int = 4,
    theta: float = 0.6,
    leaf_size: int = 64,
    basis: str = "real",
    mac_type: Optional[str] = None,
    working_dtype: str = "float64",
    softening_kernel: Optional[str] = None,
    leaf_capacity: Union[int, str] = "auto",
    capacity_headroom: float = 1.3,
    partitioner: str = "sfc",
    devices: Optional[Sequence[Any]] = None,
    caps: Optional[Dict[str, int]] = None,
) -> DistributedForwardOperator:
    """Configure the distributed observation operator and its mesh.

    Parameters
    ----------
    tracer_positions : Any
        ``(M, 3)`` fixed observation points.
    source_mass : float
        The one mass every source carries.
    num_sources : int
        ``N``.
    num_devices : int
        Devices to place the particles across.
    softening : float
        Softening length (hard decision 6).
    order : int
        Expansion order.
    theta : float
        Acceptance parameter.
    leaf_size : int
        Cell-leaf size.
    basis : str
        Expansion basis.
    mac_type : Optional[str]
        Multipole acceptance criterion, or ``None`` for the geometric
        ``"dehnen"``. Mass-dependent variants are refused, for the reason
        :func:`~jaccpot.applications.density_reconstruction.forward.assert_no_differentiated_mac`
        gives: zero-mass tracers have no force scale to drive them.
    working_dtype : str
        ``"float64"`` (default) or ``"float32"``.
    softening_kernel : Optional[str]
        Pair softening kernel, or ``None`` for the solver's (``ferrers3``).
    leaf_capacity : Union[int, str]
        The per-device cell tree's leaf capacity, ``"auto"`` by default.
    capacity_headroom : float
        Rows per device over an even split, for the sample-sort's imbalance.
    partitioner : str
        ``"sfc"``, the only placement the mesh lane has (Morton domains by a
        sample sort). The September path's ``"rcb"`` is gone with it, and asking
        for it raises rather than measuring a different decomposition.
    devices : Optional[Sequence[Any]]
        Explicit device list; defaults to the first ``num_devices`` JAX sees.
    caps : Optional[Dict[str, int]]
        Overrides for :class:`~jaccpot.distributed.cross.CrossCapacities`, the
        cross-domain exchange's buffers. ``None`` lets the mesh force size them
        at its first prepare. (The September path's ``DistributedFMMConfig``
        caps -- ``max_pair_queue`` and friends -- do not exist on this lane.)

    Returns
    -------
    DistributedForwardOperator
        The configured operator.

    Raises
    ------
    ValueError
        If ``tracer_positions`` is not ``(M, 3)``, if fewer devices are visible
        than requested -- a scaling point taken on a different device count is
        not the point that was asked for -- if ``partitioner`` is not ``"sfc"``,
        or if ``caps`` names a field ``CrossCapacities`` does not have.
    """
    from jaccpot.applications.density_reconstruction.forward import (
        FUSED_DEFAULT_MAC,
        assert_no_differentiated_mac,
        fused_solver,
    )
    from jaccpot.distributed.cross import CrossCapacities
    from jaccpot.distributed.fused import AXIS_NAME

    assert_no_differentiated_mac(mac_type=mac_type, has_tracers=True)
    tracers = np.asarray(tracer_positions, dtype=np.float64)
    if tracers.ndim != 2 or tracers.shape[1] != 3:
        raise ValueError(f"tracer_positions must be (M, 3), got {tracers.shape}")
    if partitioner != "sfc":
        raise ValueError(
            f"partitioner={partitioner!r}: the fused mesh lane places particles "
            "as Morton domains ('sfc') only"
        )

    available = list(devices) if devices is not None else list(jax.devices())
    if len(available) < int(num_devices):
        raise ValueError(
            f"requested {num_devices} devices but only {len(available)} are "
            "visible; refusing to measure on a different device count"
        )
    chosen = available[: int(num_devices)]

    cross_caps: Any = "auto"
    if caps:
        # A plain class: its fields are its constructor's keywords.
        known = set(inspect.signature(CrossCapacities.__init__).parameters) - {"self"}
        unknown = sorted(set(caps) - known)
        if unknown:
            raise ValueError(
                f"unknown CrossCapacities field(s) {unknown}; known: {sorted(known)}"
            )
        cross_caps = CrossCapacities(
            **{name: int(value) for name, value in caps.items()}
        )

    mac = FUSED_DEFAULT_MAC if mac_type is None else str(mac_type)
    solver = fused_solver(
        softening=softening,
        order=order,
        theta=theta,
        leaf_size=leaf_size,
        mac_type=mac,
        basis=basis,
        working_dtype=working_dtype,
        softening_kernel=softening_kernel,
        leaf_capacity=leaf_capacity,
    )
    return DistributedForwardOperator(
        solver=solver,
        mesh=jax.sharding.Mesh(np.asarray(chosen), (AXIS_NAME,)),
        num_devices=int(num_devices),
        tracer_positions=tracers,
        source_mass=float(source_mass),
        num_sources=int(num_sources),
        softening=float(softening),
        order=int(order),
        leaf_size=int(leaf_size),
        theta=float(theta),
        mac_type=mac,
        working_dtype=str(working_dtype),
        capacity_headroom=float(capacity_headroom),
        cross_caps=cross_caps,
    )
