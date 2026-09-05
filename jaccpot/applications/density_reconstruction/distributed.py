"""The reconstruction's observation operator on a multi-device mesh.

Why this module exists, stated plainly, because the measurement that forced it
is a negative result about the alternative.

Section 7's secondary claim is that *the same optimisation runs sharded across
multiple GPUs, so parameter count is bounded by aggregate device memory rather
than a single device's*. The obvious way to try that is to shard the parameter
array and let XLA follow -- which is what
:func:`~jaccpot.applications.density_reconstruction.fit.run_fit`'s ``devices``
argument does. **It does not work, and the measurement says so:**

=========  =====================  ==============  ==================
devices    strong-scaling wall    mean gradient   largest N that ran
=========  =====================  ==============  ==================
1          950 s                  46.3 s          1 048 576
2          796 s                  49.0 s          1 048 576
4          825 s                  51.9 s          1 048 576
=========  =====================  ==============  ==================

The ceiling does not move and the per-gradient cost gets monotonically *worse*.
The reason is structural: jaccpot's radix force evaluation is written over one
full particle array, so a sharded input is all-gathered before the work starts
and re-sharded afterwards. Parameter sharding therefore buys communication and
nothing else. It distributes where the numbers are *stored*, not where the force
is *computed*, and it is the latter that sets the ceiling.

What does distribute the computation is jaccpot's own distributed FMM -- the
section 5 path, ``jaccpot.distributed.fmm`` -- which partitions the *sources*
across devices and exchanges halos. This module drives that path
differentiably, so the claim can be tested as stated instead of asserted.

The frozen readout, and why it is the same contract as everywhere else
---------------------------------------------------------------------
The distributed evaluator has no separate prepare/state seam: it rebuilds its
per-device trees *inside* the call, and returns accelerations in each device's
own tree order together with the global ids naming those rows. That returned
``gid`` is the only trustworthy readout map -- ``partition_for_devices``'s
docstring records, with a pinned test, that reusing the *input* ``gid_flat``
compares each particle's force against a Morton neighbour's and is "plausible,
smooth, and wrong by tens of percent".

Since the readout permutation is an output, it is a tracer under ``jax.grad``
and cannot index anything. :meth:`DistributedForwardOperator.prepare` therefore
calls the evaluator once *concretely*, at the current positions, and freezes the
tracer rows it reports. That is not a workaround: holding the permutation
constant while differentiating the numeric pipeline is exactly hard decision 5,
and exactly what the radix path's prepared state does. The same rebuild-per-
iteration structure applies, with the same switching boundaries between epochs.

Verified before this module was written: on a two-device forced-CPU mesh at
N=256, M=32, the tracer field this produces agrees with
:func:`jaccpot.autodiff.direct_sum_gravitational_acceleration` over the same
combined set to **6.2e-16** relative L2, and the gradient is finite. (A first
attempt appeared to disagree by exactly 2.000, which is the signature of a sign
convention, and it was the hand-rolled oracle in the probe that had the sign
backwards, not the operator. Using jaccpot's own direct sum settled it.)

The ceiling is the path's, not the defaults'
-------------------------------------------
The ceiling this path reaches is well below the single-device radix operator's
-- P=393216 at two devices and P=786432 at four, against P=3145728 on one
device -- and it doubles with device count, which is the claim. Since every
buffer cap in ``DistributedFMMConfig`` auto-sizes from ``N``, the obvious
suspicion was that the number describes a default rather than the method.
Measured, and it does not (``bench/payoff_static/distributed_caps_probe.py``):

======== =================================== ==========================
N        caps                                two devices
======== =================================== ==========================
131072   auto-sized                          OK, no overflow
131072   max_pair_queue=4e6                  OK, no overflow
131072   max_pair_queue=1e6                  OK, no overflow
131072   pair_queue=1e6, interactions=512    OK, no overflow
262144   auto-sized                          out of memory (617 s)
262144   max_pair_queue=4e6                  out of memory (52 s)
262144   max_pair_queue=1e6                  out of memory (51 s)
262144   pair_queue=1e6, interactions=512    out of memory (51 s)
======== =================================== ==========================

Tightening the pair queue by orders of magnitude changes how long the failure
takes, not whether it happens, and the failing allocation is 3.35 GiB in every
tightened case. So the ceiling stands as a property of this path.

What this operator does not offer
---------------------------------
No leaf blocks, so no regularisers. The penalties in
:mod:`~jaccpot.applications.density_reconstruction.loss` are built on the radix
prepared state's leaf partition, which has no counterpart here.
:meth:`leaf_blocks` returns ``None`` and ``run_fit`` refuses a non-zero
regularisation weight rather than silently running an unregularised fit -- which
matters, because fig17(d) exists to show what dropping the regularisers does.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

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
    """One partition of the combined particle set, with its readout frozen.

    The distributed analogue of a radix ``FMMPreparedState``: built host-side
    from concrete positions, held constant while the numeric pipeline is
    differentiated, and rebuilt between iterations.

    Attributes
    ----------
    evaluator : Any
        The compiled differentiable force evaluator for this partition's caps.
    gid_index : jnp.ndarray
        ``(rows,)`` int32 global id of each layout row, clamped at 0 so it can
        index; ``real_mask`` says which entries mean anything.
    real_mask : jnp.ndarray
        ``(rows, 1)`` bool, ``False`` on padding rows.
    pad_positions : jnp.ndarray
        ``(rows, 3)`` the partitioner's own layout, used verbatim for padding
        rows so their geometry is exactly what it chose.
    mass_flat : jnp.ndarray
        ``(rows,)`` masses in layout order. Constant: masses are frozen.
    gid_flat : jnp.ndarray
        ``(rows,)`` the input global-id array the evaluator expects.
    counts : jnp.ndarray
        ``(ndev,)`` real particle count per device.
    tracer_rows : jnp.ndarray
        ``(M,)`` int32 output rows holding the tracers, ordered by global id.
        Frozen from a concrete forward call; see the module docstring.
    layout_shape : Tuple[int, ...]
        Shape the evaluator expects for ``pos_flat``.
    cap : int
        Per-device capacity.
    diagnostics : Dict[str, Any]
        The evaluator's overflow counters from the concrete probe call, so a
        run records whether any buffer overflowed.
    """

    evaluator: Any
    gid_index: jnp.ndarray
    real_mask: jnp.ndarray
    pad_positions: jnp.ndarray
    mass_flat: jnp.ndarray
    gid_flat: jnp.ndarray
    counts: jnp.ndarray
    tracer_rows: jnp.ndarray
    layout_shape: Tuple[int, ...]
    cap: int
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class DistributedForwardOperator:
    """Positions -> accelerations at fixed tracers, on a device mesh.

    Mirrors :class:`~jaccpot.applications.density_reconstruction.forward.ForwardOperator`'s
    surface -- ``prepare``, ``evaluate_at_topology``, ``evaluate``,
    ``leaf_blocks``, ``record`` -- so ``run_fit`` drives either without knowing
    which it has.

    Attributes
    ----------
    config : Any
        The ``DistributedFMMConfig`` in use.
    mesh : Any
        The device mesh.
    num_devices : int
        Device count.
    tracer_positions : np.ndarray
        ``(M, 3)`` fixed observation points.
    source_mass : float
        The one mass every source carries.
    num_sources : int
        ``N``.
    softening : float
        Plummer softening, recorded because hard decision 6 says so.
    order : int
        Expansion order.
    leaf_size : int
        Leaf size.
    theta : float
        Acceptance parameter.
    mac_type : Optional[str]
        The MAC in use.
    partitioner : str
        Domain decomposition used by ``partition_for_devices``.
    """

    config: Any
    mesh: Any
    num_devices: int
    tracer_positions: np.ndarray
    source_mass: float
    num_sources: int
    softening: float
    order: int
    leaf_size: int
    theta: float
    mac_type: Optional[str]
    partitioner: str

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
        """Partition the particles and freeze the readout permutation.

        Parameters
        ----------
        source_positions : Any
            ``(N, 3)`` concrete positions.

        Returns
        -------
        DistributedPartition
            The partition, its compiled evaluator, and the frozen tracer rows.

        Raises
        ------
        RuntimeError
            If the concrete probe call reports fewer tracer rows than there are
            tracers. That means the readout could not be resolved, and every
            number downstream would be quietly wrong -- the exact failure
            ``partition_for_devices``'s docstring warns about.
        """
        from jaccpot.distributed.fmm import make_force_evaluator, partition_for_devices

        combined = np.asarray(
            self.combined_positions(np.asarray(source_positions)), dtype=np.float64
        )
        masses = self.combined_masses()
        part = partition_for_devices(
            combined,
            masses,
            int(self.num_devices),
            leaf_size=int(self.config.leaf_size),
            partitioner=self.partitioner,
        )
        evaluator = make_force_evaluator(
            self.config,
            int(self.num_devices),
            part["cap"],
            self.mesh,
            jit=True,
            differentiable=True,
        )

        pos_flat = jnp.asarray(part["pos_flat"])
        mass_flat = jnp.asarray(part["mass_flat"])
        gid_flat = jnp.asarray(part["gid_flat"])
        counts = jnp.asarray(part["counts"])

        # One concrete call, to learn the readout permutation. It is an OUTPUT
        # of the evaluator, so it is a tracer under jax.grad and cannot index
        # anything; freezing it here is hard decision 5 applied to this path.
        _accel, out_gid, diag = evaluator(pos_flat, mass_flat, gid_flat, counts)
        out = np.asarray(out_gid).reshape(-1)
        rows = np.where(out >= int(self.num_sources))[0]
        rows = rows[np.argsort(out[rows])]
        if rows.size != self.num_tracers:
            raise RuntimeError(
                f"distributed readout resolved {rows.size} tracer rows but the "
                f"operator has {self.num_tracers}; the returned gid is the only "
                "trustworthy readout map and it did not name every tracer"
            )

        gid_host = np.asarray(gid_flat).reshape(-1)
        return DistributedPartition(
            evaluator=evaluator,
            gid_index=jnp.asarray(np.maximum(gid_host, 0).astype(np.int32)),
            real_mask=jnp.asarray((gid_host >= 0)[:, None]),
            pad_positions=pos_flat.reshape(-1, 3),
            mass_flat=mass_flat,
            gid_flat=gid_flat,
            counts=counts,
            tracer_rows=jnp.asarray(rows.astype(np.int32)),
            layout_shape=tuple(pos_flat.shape),
            cap=int(part["cap"]),
            diagnostics=_diagnostics_record(diag),
        )

    def evaluate_at_topology(
        self: "DistributedForwardOperator",
        partition: DistributedPartition,
        source_positions: ArrayLike,
    ) -> jnp.ndarray:
        """Accelerations at the tracers, at this partition's frozen readout.

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
        # Real rows gather from the live parameters; padding rows keep the
        # partitioner's own coordinates, so their geometry is exactly what it
        # chose and no cotangent reaches them.
        scattered = jnp.where(
            partition.real_mask,
            combined[partition.gid_index],
            partition.pad_positions,
        )
        accelerations, _gid, _diag = partition.evaluator(
            scattered.reshape(partition.layout_shape),
            partition.mass_flat,
            partition.gid_flat,
            partition.counts,
        )
        return accelerations.reshape(-1, 3)[partition.tracer_rows]

    def evaluate(
        self: "DistributedForwardOperator", source_positions: Any
    ) -> jnp.ndarray:
        """Repartition from ``source_positions`` and evaluate.

        Parameters
        ----------
        source_positions : Any
            ``(N, 3)`` concrete positions.

        Returns
        -------
        jnp.ndarray
            ``(M, 3)`` accelerations at the tracers.
        """
        return self.evaluate_at_topology(
            self.prepare(source_positions), jnp.asarray(source_positions)
        )

    def leaf_blocks(
        self: "DistributedForwardOperator", partition: DistributedPartition
    ) -> None:
        """Return ``None``: this path exposes no leaf partition.

        Parameters
        ----------
        partition : DistributedPartition
            Unused; present so the signature matches the radix operator's.

        Returns
        -------
        None
            The regularisers are built on the radix prepared state's leaf
            blocks and have no counterpart here. ``run_fit`` refuses a non-zero
            regularisation weight against this operator rather than silently
            running an unregularised fit.
        """
        return None

    def record(self: "DistributedForwardOperator") -> Dict[str, Any]:
        """Return the operator configuration for a results JSON.

        Returns
        -------
        Dict[str, Any]
            Counts, accuracy settings, and -- named explicitly -- the fact that
            this is the distributed force evaluation and not parameter
            sharding. A wall-clock from one is not comparable with the other.
        """
        return {
            "N": int(self.num_sources),
            "M": int(self.num_tracers),
            "softening": float(self.softening),
            "order": int(self.order),
            "leaf_size": int(self.leaf_size),
            "theta": float(self.theta),
            "mac_type": self.mac_type,
            "num_devices": int(self.num_devices),
            "partitioner": self.partitioner,
            "sharding_mode": "distributed_force_evaluation",
        }


def _diagnostics_record(diag: Any) -> Dict[str, Any]:
    """Summarise the evaluator's diagnostic counters.

    Parameters
    ----------
    diag : Any
        The third element of the evaluator's return, one row per device.

    Returns
    -------
    Dict[str, Any]
        Per-field lists plus an ``overflowed`` flag. An overflowed buffer means
        interactions were dropped, so a force from such a run is wrong and the
        artifact has to say so.
    """
    from jaccpot.distributed.fmm import DIAG_FIELDS

    values = np.asarray(diag)
    counters = {
        field: values[:, index].tolist() for index, field in enumerate(DIAG_FIELDS)
    }
    overflow_fields = [f for f in DIAG_FIELDS if "overflow" in f or "dropped" in f]
    return {
        "counters": counters,
        "overflow_fields": overflow_fields,
        "overflowed": bool(any(any(counters.get(f, [])) for f in overflow_fields)),
    }


def make_distributed_forward_operator(
    *,
    tracer_positions: Any,
    source_mass: float,
    num_sources: int,
    num_devices: int,
    softening: float,
    order: int = 4,
    theta: float = 0.5,
    leaf_size: int = 64,
    basis: str = "solidfmm",
    mac_type: Optional[str] = None,
    nearfield_backend: Optional[str] = None,
    partitioner: str = "rcb",
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
        Devices to partition the sources across.
    softening : float
        Plummer softening length (hard decision 6).
    order : int
        Expansion order.
    theta : float
        Acceptance parameter.
    leaf_size : int
        Leaf size.
    basis : str
        Expansion basis.
    mac_type : Optional[str]
        Multipole acceptance criterion, or ``None`` for the config default.
        Mass-dependent variants are refused, for the reason
        :func:`~jaccpot.applications.density_reconstruction.forward.assert_no_differentiated_mac`
        gives: zero-mass tracers have no force scale to drive them.
    nearfield_backend : Optional[str]
        Near-field lane, or ``None`` for the config default.
    partitioner : str
        ``partition_for_devices`` domain decomposition.
    devices : Optional[Sequence[Any]]
        Explicit device list; defaults to the first ``num_devices`` JAX sees.
    caps : Optional[Dict[str, int]]
        Overrides for ``DistributedFMMConfig``'s buffer caps --
        ``max_pair_queue``, ``max_interactions_per_node``,
        ``max_neighbors_per_leaf``, ``process_block`` and the ``cross_*``
        variants. They all default to ``None``, meaning auto-sized from ``N``,
        and the auto-sized values are what set the memory ceiling this path
        hits. Exposed so a ceiling can be reported at *tuned* caps rather than
        only at the defaults -- a number measured at whatever the defaults
        happened to choose is a statement about the defaults.

        A cap that is too small does not fail loudly: it silently drops
        interactions and returns a wrong force. Check
        :attr:`DistributedPartition.diagnostics`'s ``overflowed`` flag, which
        every run records, before believing a force from tightened caps.

    Returns
    -------
    DistributedForwardOperator
        The configured operator.

    Raises
    ------
    ValueError
        If ``tracer_positions`` is not ``(M, 3)``, if fewer devices are visible
        than requested -- a scaling point taken on a different device count is
        not the point that was asked for -- or if ``caps`` names a field
        ``DistributedFMMConfig`` does not have.
    """
    import jax
    from yggdrax.distributed import make_mesh

    from jaccpot.applications.density_reconstruction.forward import (
        assert_no_differentiated_mac,
    )
    from jaccpot.distributed import DistributedFMMConfig

    assert_no_differentiated_mac(mac_type=mac_type, has_tracers=True)
    tracers = np.asarray(tracer_positions, dtype=np.float64)
    if tracers.ndim != 2 or tracers.shape[1] != 3:
        raise ValueError(f"tracer_positions must be (M, 3), got {tracers.shape}")

    available = list(devices) if devices is not None else list(jax.devices())
    if len(available) < int(num_devices):
        raise ValueError(
            f"requested {num_devices} devices but only {len(available)} are "
            "visible; refusing to measure on a different device count"
        )
    chosen = available[: int(num_devices)]

    overrides: Dict[str, Any] = dict(
        order=int(order),
        theta=float(theta),
        leaf_size=int(leaf_size),
        basis=str(basis),
        softening=float(softening),
    )
    if mac_type is not None:
        overrides["mac_type"] = mac_type
    if nearfield_backend is not None:
        overrides["nearfield_backend"] = nearfield_backend
    if caps:
        known = {f.name for f in dataclasses.fields(DistributedFMMConfig)}
        unknown = sorted(set(caps) - known)
        if unknown:
            raise ValueError(
                f"unknown DistributedFMMConfig cap(s) {unknown}; known fields "
                f"include {sorted(n for n in known if 'cap' in n or 'max_' in n)}"
            )
        overrides.update({name: int(value) for name, value in caps.items()})
    config = dataclasses.replace(DistributedFMMConfig(), **overrides)

    return DistributedForwardOperator(
        config=config,
        mesh=make_mesh(int(num_devices), devices=chosen),
        num_devices=int(num_devices),
        tracer_positions=tracers,
        source_mass=float(source_mass),
        num_sources=int(num_sources),
        softening=float(softening),
        order=int(order),
        leaf_size=int(leaf_size),
        theta=float(theta),
        mac_type=mac_type,
        partitioner=str(partitioner),
    )
