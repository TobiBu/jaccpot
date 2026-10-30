"""The fused multi-GPU lane in segments: ONE compiled program per segment.

:class:`~jaccpot.distributed.rollout.FusedRollout` steps the mesh lane one step at a
time: three or four jitted programs and a host sync per STEP (kick-drift, an
optional repartition, the force, the kick), around a force that holds every
device's eager prepared state. The single card's production runner
(``ProductionRunner``: ``strict_run_v2(carry="particles")`` segments) runs a whole
segment as one program with the state donated, syncs the host once per segment
and re-plans a segment that outgrew its capacities. This module gives the mesh
lane the same, as a new path next to the old one (which stays the reference):

* **One program per segment.** ``jit(shard_map(lax.scan over K steps))`` with the
  packed state ``(ndev * cap, 2, 3)`` and the total acceleration ``(ndev * cap,
  3)`` donated. A step is the particle carry's velocity Verlet with the same
  association -- ``x + v dt + a dt^2 / 2``, the force at the drifted positions,
  ``v + (a + a_new) dt / 2`` (``fmm_state._velocity_verlet_kick_drifted``) -- and
  the force is the template route's (:mod:`jaccpot.distributed.template_force`:
  the provider's shape template materialised per step, no prepared state in the
  program). Per device the scan folds the verdict and the needs over its steps;
  they are reduced over the mesh ONCE, after the scan, and read on the host once
  per segment.
* **The start force in its own program**, as on the single card (inside the
  scan's program XLA overlaps it with the first step and doubles the temporary
  block): the first segment's, and a re-planned segment's.
* **The total acceleration is carried** from one segment into the next, so two
  segments of K steps are one of 2K, bit for bit (gate A2).
* **An external field per row**, ``external_acceleration_fn(state, params)``
  (``(state)`` without parameters) on each device's ``(cap, 2, 3)`` rows, added to
  the self-gravity as the single card adds it.
* **Segment retry.** A fired flag (any device, any step) re-plans what the
  segment outgrew from its needs -- the walk caps, the tree's leaf capacity,
  depth bound and level batch, the cross capacities (``CROSS_NEEDS_FIELDS``,
  carried after ``NEEDS_FIELDS``; sized in the library at the start) --
  re-prepares every device caps-only at the segment's START (a host copy of the
  donated start state is kept, as the single card keeps it), recompiles and
  re-runs from the start force: at most twice, the single card's policy, and
  again as long as each re-run makes progress -- fits more steps than every
  earlier attempt (the re-plan reads only the steps up to the first failure, so
  needs that grow step by step are learned a step at a time), or fails at the
  same step after a re-plan from lower bounds and needs more than they said
  (:func:`_retry_made_progress`, the single card's ``_segment_retry_next``).
* **Cross capacities sized at drifted positions too.** ``cross_caps="auto"`` is
  measured at the start positions when the runner is built and, at the first
  :meth:`MeshSegmentRunner.run` (its ``dt`` known), at that segment's drifted
  positions (:meth:`MeshSegmentRunner._measure_cross_drifted`): the drift after a
  decomposition grows the cross demand for a few steps, which a first segment
  sized at the start alone retried.
* **Repartition BETWEEN segments**, as its own program, every
  ``repartition_every`` segments (host cadence: no ``lax.cond`` around
  collectives, whose device-divergent predicate deadlocks, and none of its buffers
  in a segment's peak). With the velocity-Verlet form the carried acceleration
  must travel with its particle, so ``sfc_repartition`` routes the velocity, the
  total acceleration and the id (``rollout._repartition_shard``).
* **Migration to fixed cuts** (opt-in, ``migrate=True``): after every drift, before
  the force, each particle outside its device's splitter range moves to the owner
  (:mod:`jaccpot.distributed.migration`), in the scan, so every force sees exact
  Morton domains in its own box -- the start's zero-drift cross volume -- instead of
  the stragglers a drift leaves in its neighbour's cells, whose coarse leaves make
  most of the cross near pairs. The repartition between segments then only
  rebalances the counts: even cuts read off the ranks and a migration to them
  (``sfc_repartition`` once, at the start, puts the domains in the force's box).
  Counts, masses and ids change inside the segment and ride the carry; the
  exchange's buffer (``migration_cap``) is a capacity with an exact need and a
  re-plan.
* **The potential energy at the current state** (:meth:`MeshSegmentRunner.potential_energy`,
  S11 session 2): the force's ``potential_shard`` per device and the external
  potential per row, summed in fp64 over the mesh, in a program of its own that
  changes no number of the run.
* **Dehnen's error criterion** (``mac_type='dehnen_error'``, S12; the force's
  ``scale_carry``): eq (16b)'s per-particle ``f_b`` rides with the acceleration --
  in the start program (in: the carried one or, before any force, the prepare's
  seed; out: the force's own), in the scan carry (each force's thresholds are the
  previous force's ``f_b``, as the single card's particle carry), in
  :class:`MeshState` between segments and through the repartition. A retried
  segment restarts from the re-prepare's seed, as a fresh run there would.

**Padding rows** keep their place: their velocity and acceleration are zero, so the
drift leaves them where the last repartition put them (on a live row), and the
force cuts them with ``num_valid``.

**Array convention, and it is a trap** (``fused.py``'s module docstring):
``shard_map`` keeps the mapped axis, so every particle array is FLAT, ``(ndev *
cap, ...)``, and a device sees ``(cap, ...)``.

The force is injectable (:meth:`MeshSegmentRunner.__init__`'s ``force``): the
fused lane's (:class:`~jaccpot.distributed.template_force.FusedMeshForce`;
:meth:`MeshSegmentRunner.fused` builds it), or :class:`DirectMeshForce`, a
partition-independent direct sum for tests.

WHY A MODULE OF ITS OWN, not :mod:`jaccpot.distributed.rollout`: the rollout's
per-step path (``setup_fused_force``, ``FusedRollout``) stays the protected
reference until this one passes its gates on cards, and the two share only the
decomposition and the repartition's per-device body.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from jaccpot.distributed.fused import AXIS_NAME, reduce_flag_across_mesh
from jaccpot.runtime.dtypes import require_x64

__all__ = [
    "DirectMeshForce",
    "MeshSegmentRunner",
    "MeshState",
    "SegmentReport",
]

#: the attribute of a force's ``cache_owner`` (the engine, for the fused force) that
#: holds the runner's compiled programs: runners on one solver share them, every
#: static input is in the key, and they go with the engine (as the single card's
#: ``_strict_fused_jit_function_cache`` does)
_PROGRAMS_ATTR = "_mesh_segment_programs"
#: overflow bits reduced over the mesh (``fused_topology.OVERFLOW_*`` uses 0-10)
_BIT_COUNT = 16


@dataclass
class MeshState:
    """The mesh-resident particles in the segment runner's layout.

    Every array is flat and sharded ``P(axis)``: row ``r`` of device ``d`` is row
    ``d * cap + r``, with ``count[d]`` leading live rows.

    Attributes
    ----------
    state : Array
        ``(ndev * cap, 2, 3)`` positions and velocities; padding rows sit on a live
        row of their device with zero velocity.
    masses : Array
        ``(ndev * cap,)``; zero on padding.
    ids : Array
        ``(ndev * cap,)`` int32 global ids; ``-1`` on padding.
    count : Array
        ``(ndev,)`` int32 live rows per device.
    acceleration : Optional[Array]
        ``(ndev * cap, 3)`` TOTAL acceleration (self-gravity plus the external
        field) at ``state``'s positions, zero on padding; ``None`` until a start
        force computed it.
    force_scale : Optional[Array]
        ``mac_type='dehnen_error'``: ``(ndev * cap,)`` eq (16b)'s ``f_b`` of the
        force that gave ``acceleration`` (the next force's thresholds), zero on
        padding; ``None`` without the criterion or before a start force (the force's
        prepare seeds the first one).
    """

    state: Array
    masses: Array
    ids: Array
    count: Array
    acceleration: Optional[Array] = None
    force_scale: Optional[Array] = None

    @classmethod
    def from_shard_state(
        cls, shard: Any, mesh: Any, *, axis_name: str = AXIS_NAME
    ) -> "MeshState":
        """Pack a :class:`~jaccpot.distributed.rollout.ShardState` (``decompose``'s).

        Parameters
        ----------
        shard : Any
            The ``ShardState``; its acceleration is dropped (the runner computes
            the total acceleration in its start program).
        mesh : Any
            The mesh it lives on.
        axis_name : str
            Mesh axis.

        Returns
        -------
        MeshState
            Positions and velocities stacked into ``(ndev * cap, 2, 3)``.
        """
        return cls(
            state=_state_packer(NamedSharding(mesh, P(axis_name)))(
                shard.positions, shard.velocities
            ),
            masses=shard.masses,
            ids=shard.ids,
            count=shard.count,
        )


#: :func:`_state_packer`'s programs, by output sharding
_PACKERS: dict = {}


def _stack_state(x: Array, v: Array) -> Array:
    """Positions and velocities as one ``(rows, 2, 3)`` state.

    Parameters
    ----------
    x : Array
        ``(rows, 3)`` positions.
    v : Array
        ``(rows, 3)`` velocities.

    Returns
    -------
    Array
        ``(rows, 2, 3)``.
    """
    return jnp.stack([x, v], axis=1)


def _state_packer(sharding: Any) -> Callable:
    """:func:`_stack_state` jitted onto ``sharding``, built once per sharding.

    A ``jax.jit`` of a lambda built per call is a new function to jax: every fresh
    run of a runner compiled it again.

    Parameters
    ----------
    sharding : Any
        The output's ``NamedSharding``.

    Returns
    -------
    Callable
        ``(x, v) -> state``.
    """
    packer = _PACKERS.get(sharding)
    if packer is None:
        packer = _PACKERS[sharding] = jax.jit(_stack_state, out_shardings=sharding)
    return packer


@dataclass
class SegmentReport:
    """What one segment did, read back once from the replicated outputs.

    Attributes
    ----------
    segment : int
        Index of the segment just completed (1-based).
    steps : int
        Steps it advanced.
    start_force : bool
        Whether its start force was computed (the first segment, a changed
        external field) rather than carried.
    retries : int
        Re-plans (and re-runs from the segment's start) it needed.
    repartitioned : bool
        Whether a repartition ran before it.
    declined : bool
        That repartition was declined (a device would have overflowed): nothing
        moved, the old ownership stands.
    sent_off_device : int
        Live rows that repartition moved to another device.
    counts : list
        Live rows per device during the segment.
    bits : int
        The kept run's overflow bits, OR-ed over its steps and the mesh.
    needs : list
        The kept run's needs, max-ed over its steps and the mesh
        (``fused_topology.NEEDS_FIELDS``, then ``cross.CROSS_NEEDS_FIELDS``).
    failures : list
        One entry per failed attempt before the kept run: its ``bits``, ``needs``
        and the steps that fitted (``done``) -- what the retries re-planned from.
    migrated : int
        ``migrate=True``: rows the kept run's migrations moved, over its steps and
        the mesh (``0`` without migration).
    migration_need : int
        ``migrate=True``: the most rows a device sent or received in one of its
        migrations (what ``migration_cap`` has to hold).
    """

    segment: int
    steps: int
    start_force: bool
    retries: int = 0
    repartitioned: bool = False
    declined: bool = False
    sent_off_device: int = 0
    counts: list = field(default_factory=list)
    bits: int = 0
    needs: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    migrated: int = 0
    migration_need: int = 0


def _mesh_all(flag: Array, axis_name: str) -> Array:
    """A boolean that holds on every device (inside ``shard_map``).

    Parameters
    ----------
    flag : Array
        This device's boolean scalar.
    axis_name : str
        Mesh axis.

    Returns
    -------
    Array
        Boolean scalar, identical on every device.
    """
    return jnp.logical_not(
        reduce_flag_across_mesh(jnp.logical_not(flag), axis_name=axis_name)
    )


def _mesh_verdict(
    ok: Array, bits: Array, needs: Array, done: Array, *, axis_name: str
) -> tuple[Array, Array, Array, Array]:
    """One device's segment verdict reduced over the mesh (inside ``shard_map``).

    One device's overflow is everyone's (``fused.reduce_flag_across_mesh``): the
    segment failed if any device's did, the bits are OR-ed, the needs max-ed and the
    steps that fitted min-ed.

    Parameters
    ----------
    ok : Array
        Boolean scalar: every step of this device fitted.
    bits : Array
        ``uint32`` overflow bits OR-ed over the steps.
    needs : Array
        ``int32`` needs max-ed over the steps.
    done : Array
        ``int32`` steps that fitted, from the start.
    axis_name : str
        Mesh axis.

    Returns
    -------
    tuple[Array, Array, Array, Array]
        ``(ok, bits, needs, done)``, identical on every device.
    """
    ok_all = jnp.logical_not(
        reduce_flag_across_mesh(jnp.logical_not(ok), axis_name=axis_name)
    )
    shifts = jnp.arange(_BIT_COUNT, dtype=jnp.uint32)
    per_bit = ((jnp.asarray(bits, jnp.uint32) >> shifts) & jnp.uint32(1)).astype(
        jnp.int32
    )
    per_bit = jax.lax.pmax(per_bit, axis_name).astype(jnp.uint32)
    bits_all = jnp.sum(per_bit << shifts, dtype=jnp.uint32)
    return (
        ok_all,
        bits_all,
        jax.lax.pmax(jnp.asarray(needs, jnp.int32), axis_name),
        jax.lax.pmin(jnp.asarray(done, jnp.int32), axis_name),
    )


def _retry_made_progress(needs: np.ndarray, done: int, earlier: list) -> bool:
    """Whether a failed re-run made progress: the retry past the force's policy.

    The mesh's ``StrictRunMixin._segment_retry_next`` (the single card's rule, the
    same limits). A re-run made progress when

    * it fitted more steps than every earlier attempt (``done`` grows with each such
      retry, so there are at most the segment's steps of them), or
    * it failed at the SAME step as the attempt before, whose needs were LOWER
      BOUNDS, and it needed more than they said of what they bounded: their re-plan
      was a doubling, not a measurement (each such retry at least doubles a
      capacity). The local walk's lists and queue (``lower_bound``: the flat walk
      stops at its first overflow; ``queue_overflow``: a walk whose queue overflowed
      counted part of its pairs), the cross walk's lists and queue
      (``walk_lower_bound``: yggdrax's traced walk stops at its first overflow, the
      Pallas walk undercounts past a full queue) and the exchange's rows and
      particles (``pairs_lower_bound``: the walk or the summary lost pairs).

    A re-plan that grows nothing still raises (the force's ``replan``), and
    ``JACCPOT_STRICT_SEGMENT_RETRY=0`` turns every retry off (the caller).

    Parameters
    ----------
    needs : np.ndarray
        This attempt's needs, mesh-reduced: ``fused_topology.NEEDS_FIELDS``, then
        ``cross.CROSS_NEEDS_FIELDS``.
    done : int
        Steps this attempt fitted before its first failure.
    earlier : list
        The segment's earlier failed attempts, in order
        (:attr:`SegmentReport.failures` entries: ``done``, ``needs``).

    Returns
    -------
    bool
        Whether the segment is retried anyway.
    """
    from jaccpot.distributed.cross import _NEEDS_LOWER_BOUND_BIT, CROSS_NEEDS_FIELDS
    from jaccpot.runtime.fused_topology import NEEDS_FIELDS

    if not earlier:
        return False
    best = max(int(f["done"]) for f in earlier)
    if int(done) > best:
        return True
    last = earlier[-1]
    if not int(done) == best == int(last["done"]):
        return False
    now = np.asarray(needs, np.int64)
    before = np.asarray(last["needs"], np.int64)
    local = NEEDS_FIELDS.index
    # (the earlier attempt's lower-bound bit, the needs it qualifies)
    bounded = [
        (
            bool(before[local("lower_bound")] or before[local("queue_overflow")]),
            list(range(local("far_overflow"))),  # far, near, peak wavefront
        )
    ]
    if now.shape[0] >= len(NEEDS_FIELDS) + len(CROSS_NEEDS_FIELDS):
        cross = len(NEEDS_FIELDS)
        for bit in ("walk_lower_bound", "pairs_lower_bound"):
            fields = [
                cross + CROSS_NEEDS_FIELDS.index(name)
                for name, qualifier in _NEEDS_LOWER_BOUND_BIT.items()
                if qualifier == bit
            ]
            bounded.append(
                (bool(before[cross + CROSS_NEEDS_FIELDS.index(bit)]), fields)
            )
    return any(
        lower and bool(np.any(now[fields] > before[fields]))
        for lower, fields in bounded
    )


def _drift_probe_steps(steps: int) -> tuple[int, ...]:
    """The drifts, in steps, the cross sizing measures a first segment at.

    Every power of two below ``steps``, then ``steps``: how fast the demand after
    a decomposition grows depends on the leaf size at the domain cuts against the
    distance a step moves a particle (Plummer on two A100s: still growing at step
    5 at 1e5 particles per card, 2.8x the start's near pairs after ONE step at
    4e6), so the probes cover the segment on a log scale -- ``1 +
    ceil(log2(steps))`` forces.

    Parameters
    ----------
    steps : int
        The segment's steps (at least 1).

    Returns
    -------
    tuple[int, ...]
        Increasing drifts ``1, 2, 4, ..., steps``.
    """
    probes, s = [], 1
    while s < int(steps):
        probes.append(s)
        s *= 2
    return (*probes, int(steps))


@jax.jit
def _drifted(state: Array, t: Array) -> Array:
    """``state`` with its positions drifted by its velocities over ``t``.

    Parameters
    ----------
    state : Array
        ``(rows, 2, 3)`` positions and velocities (padding rows: zero velocity,
        so they stay).
    t : Array
        Scalar time in the state's dtype.

    Returns
    -------
    Array
        ``(rows, 2, 3)``, the velocities unchanged; sharded as ``state``.
    """
    return state.at[:, 0].add(t * state[:, 1])


def _programs_of(owner: Any) -> dict:
    """The compiled-program cache held by ``owner`` (created on first use).

    Parameters
    ----------
    owner : Any
        A force's ``cache_owner``.

    Returns
    -------
    dict
        Key -> compiled program.
    """
    cache = getattr(owner, _PROGRAMS_ATTR, None)
    if cache is None:
        cache = {}
        setattr(owner, _PROGRAMS_ATTR, cache)
    return cache


def _total_acceleration(
    field_fn: Optional[Callable[..., Array]],
    acc_self: Array,
    state_rows: Array,
    params: Any,
    live_rows: Array,
) -> Array:
    """Self-gravity plus the external field, zero on padding (one device).

    The single card's order (``strict_run_v2``'s ``_advance``): the self-gravity in
    the state's dtype, plus the field in the state's dtype.

    Parameters
    ----------
    field_fn : Optional[Callable[..., Array]]
        The external field, or ``None``.
    acc_self : Array
        ``(cap, 3)`` self-gravity.
    state_rows : Array
        ``(cap, 2, 3)`` the rows the field sees.
    params : Any
        The field's parameters, or ``None`` for a one-argument field.
    live_rows : Array
        ``(cap,)`` boolean.

    Returns
    -------
    Array
        ``(cap, 3)`` in the state's dtype.
    """
    acc = jnp.asarray(acc_self, state_rows.dtype)
    if field_fn is not None:
        value = field_fn(state_rows) if params is None else field_fn(state_rows, params)
        acc = acc + jnp.asarray(value, state_rows.dtype)
    return jnp.where(live_rows[:, None], acc, jnp.zeros((), acc.dtype))


def _direct_potential_shard(
    x: Array,
    m: Array,
    n_live: Array,
    ids: Array,
    *,
    G: float,
    softening: float,
    kernel: str,
    axis_name: str,
) -> Array:
    """One device's all-pairs potential (inside ``shard_map``), partition-independent.

    The potential half of ``rollout._reference_direct_shard``: the mesh's particles
    all-gathered and summed in GLOBAL-ID order, ``phi_i = -G sum_{j != i} m_j
    psi(r_ij)`` with the pair kernel's potential factor (:mod:`jaccpot.softening`;
    Newtonian beyond a compact kernel's support). The self pair is left out by id,
    not by ``r = 0``: two particles at one position still see each other's
    ``psi(0)``.

    Parameters
    ----------
    x : Array
        ``(cap, 3)`` positions.
    m : Array
        ``(cap,)`` masses.
    n_live : Array
        Scalar live row count.
    ids : Array
        ``(cap,)`` global ids.
    G : float
        Gravitational constant.
    softening : float
        Plummer-equivalent softening length.
    kernel : str
        The resolved pair kernel.
    axis_name : str
        Mesh axis.

    Returns
    -------
    Array
        ``(cap,)`` potentials in the positions' dtype, zero on padding rows.
    """
    from jaccpot.softening import masked_pair_factors, softening_params

    live = jnp.arange(x.shape[0]) < n_live
    xs = jax.lax.all_gather(x, axis_name, tiled=True)
    ms = jax.lax.all_gather(jnp.where(live, m, 0.0), axis_name, tiled=True)
    gs = jax.lax.all_gather(jnp.where(live, ids, -1), axis_name, tiled=True)
    big = jnp.iinfo(jnp.int32).max
    order = jnp.argsort(jnp.where(gs >= 0, gs, big), stable=True)
    xs, ms, gs = xs[order], ms[order], gs[order]
    d = xs[None, :, :] - x[:, None, :]
    r2 = jnp.sum(d * d, axis=-1)
    active = (gs[None, :] >= 0) & (gs[None, :] != ids[:, None])
    params = softening_params(kernel, softening, x.dtype)
    psi = masked_pair_factors(r2, active, params, kernel, potential=True)[1]
    # HIGHEST, as the reference force: no TF32 on Ampere
    phi = -jnp.asarray(G, x.dtype) * jnp.einsum(
        "ij,j->i", psi, ms, precision=jax.lax.Precision.HIGHEST
    )
    return jnp.where(live, phi, jnp.zeros((), phi.dtype))


def _force_box_cubic(force: Any) -> bool:
    """The box shape a force builds its trees in (``TreeConfig.cubic_bounds``).

    Parameters
    ----------
    force : Any
        The runner's force: the fused force's engine says; a force without one
        (:class:`DirectMeshForce`) builds no tree, and the per-axis box is taken.

    Returns
    -------
    bool
        ``True`` for a cube.
    """
    engine = getattr(force, "engine", None)
    return bool(getattr(engine, "_tree_cubic_bounds", False))


class DirectMeshForce:
    """An all-pairs force for the runner whose answer does not depend on the partition.

    :func:`~jaccpot.distributed.rollout.make_reference_direct_force`'s per-device sum
    (sources all-gathered and summed in global-id order, so a particle's force is
    bitwise the same whichever device and row hold it), with the runner's force
    interface: what lets a test require a repartitioned run and a never
    repartitioned one to agree bitwise. Never overflows. Small N only.

    Parameters
    ----------
    G : float
        Gravitational constant.
    softening : float
        Plummer-equivalent softening length.
    softening_kernel : Optional[str]
        The pair kernel (:mod:`jaccpot.softening`); ``None`` gives the default.
    axis_name : str
        Mesh axis.
    """

    def __init__(
        self,
        *,
        G: float = 1.0,
        softening: float = 0.0,
        softening_kernel: Optional[str] = None,
        axis_name: str = AXIS_NAME,
    ) -> None:
        from jaccpot.softening import resolve_softening_kernel

        self.G = float(G)
        self.softening = float(softening)
        self.kernel = resolve_softening_kernel(softening_kernel)
        self.axis_name = axis_name

    @property
    def cache_owner(self) -> "DirectMeshForce":
        """Programs are cached on the force itself.

        Returns
        -------
        DirectMeshForce
            ``self``.
        """
        return self

    @property
    def needs_size(self) -> int:
        """Length of the needs vector: the fused lane's, all zero.

        Returns
        -------
        int
            ``len(NEEDS_FIELDS) + len(CROSS_NEEDS_FIELDS)``, as the fused force's.
        """
        from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS

        return len(NEEDS_FIELDS) + len(CROSS_NEEDS_FIELDS)

    def prepare(self, positions: Array, masses: Array, counts: Array) -> None:
        """Nothing to prepare.

        Parameters
        ----------
        positions : Array
            Unused.
        masses : Array
            Unused.
        counts : Array
            Unused.
        """
        del positions, masses, counts

    def key(self) -> tuple:
        """The compile key's force part.

        Returns
        -------
        tuple
            ``("direct", G, softening, kernel)``.
        """
        return ("direct", self.G, self.softening, self.kernel)

    def context(self) -> Any:
        """No plan to install.

        Returns
        -------
        Any
            A null context manager.
        """
        import contextlib

        return contextlib.nullcontext()

    def before_trace(self) -> None:
        """Nothing recorded between traces."""

    def shard(
        self, positions: Array, masses: Array, live: Array, ids: Array
    ) -> tuple[Array, Array, Array]:
        """One device's direct sum (inside ``shard_map``).

        Parameters
        ----------
        positions : Array
            ``(cap, 3)``.
        masses : Array
            ``(cap,)``.
        live : Array
            Scalar live row count.
        ids : Array
            ``(cap,)`` global ids: the sources' summation order.

        Returns
        -------
        tuple[Array, Array, Array]
            ``(acceleration, bits = 0, needs = 0)``.
        """
        from jaccpot.distributed.rollout import _reference_direct_shard

        acc = _reference_direct_shard(
            positions,
            masses,
            live,
            ids,
            G=self.G,
            softening=self.softening,
            kernel=self.kernel,
            axis_name=self.axis_name,
        )
        return acc, jnp.uint32(0), jnp.zeros((self.needs_size,), jnp.int32)

    def potential_shard(
        self,
        positions: Array,
        masses: Array,
        live: Array,
        ids: Array,
        force_scale: Optional[Array] = None,
    ) -> tuple[Array, Array]:
        """One device's direct-sum potential (inside ``shard_map``).

        ``phi_i = -G sum_{j != i} m_j psi(r_ij)`` over every device's particles,
        with the force's pair kernel (:func:`_direct_potential_shard`): the
        sources in global-id order, so a particle's potential does not depend on
        which device and row hold it.

        Parameters
        ----------
        positions : Array
            ``(cap, 3)``.
        masses : Array
            ``(cap,)``.
        live : Array
            Scalar live row count.
        ids : Array
            ``(cap,)`` global ids: the sources' summation order and the self pair.
        force_scale : Optional[Array]
            Unused (no criterion in a direct sum).

        Returns
        -------
        tuple[Array, Array]
            ``(phi, bits = 0)``; ``phi`` ``(cap,)`` in the positions' dtype, zero on
            padding rows.
        """
        del force_scale
        phi = _direct_potential_shard(
            positions,
            masses,
            live,
            ids,
            G=self.G,
            softening=self.softening,
            kernel=self.kernel,
            axis_name=self.axis_name,
        )
        return phi, jnp.uint32(0)

    def retry_mode(self, needs: np.ndarray, retries: int, tree_first: bool) -> Any:
        """Never retried: the direct sum has no capacity.

        Parameters
        ----------
        needs : np.ndarray
            Unused.
        retries : int
            Unused.
        tree_first : bool
            Unused.

        Returns
        -------
        Any
            ``None``.
        """
        del needs, retries, tree_first
        return None

    def replan(self, needs: np.ndarray, bits: int, *, walk_only: bool = False) -> bool:
        """Nothing to re-plan.

        Parameters
        ----------
        needs : np.ndarray
            Unused.
        bits : int
            Unused.
        walk_only : bool
            Unused.

        Returns
        -------
        bool
            ``False``.
        """
        del needs, bits, walk_only
        return False

    def raise_overflow(
        self, needs: np.ndarray, bits: int, *, retries: int, where: str
    ) -> None:
        """Raise: only a count past the shard capacity can fail this force.

        Parameters
        ----------
        needs : np.ndarray
            Unused.
        bits : int
            The failed run's bits.
        retries : int
            Unused.
        where : str
            Which segment, for the message.

        Raises
        ------
        RuntimeError
            Always.
        """
        del needs, retries
        raise RuntimeError(
            f"mesh {where}: the direct force failed (bits {int(bits):#x}): a "
            "device's live count exceeds the shard capacity"
        )


class MeshSegmentRunner:
    """Velocity Verlet on the mesh, one compiled program per segment.

    See the module docstring. The runner owns its state: with ``donate`` (the
    default) the caller's :class:`MeshState` is consumed by the first segment, and
    every segment writes the next state into the last one's buffers.

    Parameters
    ----------
    mesh : Any
        The 1-D mesh.
    state : MeshState
        The decomposed initial state (``acceleration`` may be ``None``: the first
        segment computes it).
    force : Any
        The per-device force: ``shard(positions, masses, live, ids) ->
        (acceleration, bits, needs)`` inside ``shard_map``, with ``prepare``,
        ``key``, ``context``, ``before_trace``, ``needs_size``, ``cache_owner``
        and the retry's ``retry_mode``, ``replan`` and ``raise_overflow``
        (:class:`~jaccpot.distributed.template_force.FusedMeshForce`,
        :class:`DirectMeshForce`); for :meth:`potential_energy` also
        ``potential_shard(positions, masses, live, ids, force_scale=None) -> (phi,
        bits)``. ``prepare`` runs here, at ``state``, and again at a re-planned
        segment's start.
    external_acceleration_fn : Optional[Callable[..., Array]]
        ``field(state_rows, params)`` (or ``field(state_rows)`` when ``run`` gets no
        parameters) -> ``(cap, 3)``, on one device's ``(cap, 2, 3)`` rows. Keep ONE
        function object: its identity is part of the compile keys.
    repartition_every : int
        Repartition before the segment after every ``repartition_every`` completed
        segments (``0``: never after the initial decomposition). Its own program,
        between segments: the cadence is the host's, so no ``lax.cond`` around
        collectives (a device-divergent predicate deadlocks), and its buffers never
        add to a segment's peak.
    num_samples : int
        Pivot samples per device for the repartition.
    donate : bool
        Donate the state and the acceleration to each segment's program.
    axis_name : str
        Mesh axis.
    migrate : bool
        Migrate to fixed cuts after every drift (:mod:`jaccpot.distributed.migration`;
        off by default). The decomposition is redone here in the force's box (the
        cuts' frame; one repartition, before the force is prepared) unless
        ``splitters`` are given; every repartition after it is a rebalance
        (:meth:`_rebalance`: even cuts read off the ranks, a migration to them).
        No-op on one device.
    migration_cap : Any
        ``migrate=True``: the migration buffer's rows per device, each way: an int,
        or ``"auto"`` (the default: sized at the first segment from the rows one
        step's drift moves, :func:`~jaccpot.distributed.migration.migration_cap_for_need`).
        A migration that outgrows it fails its step (``OVERFLOW_MIGRATION``) and
        the segment is re-run with the buffer re-sized from the exact need.
    splitters : Any
        ``migrate=True``: the cuts of ``state``'s ownership, ``(ndev - 1,)``
        ``uint64`` (a restart's); ``None`` repartitions here to choose them.

    Raises
    ------
    ValueError
        If the state's rows do not split evenly over the mesh, or the migration's
        initial repartition was declined.
    """

    def __init__(
        self,
        mesh: Any,
        state: MeshState,
        force: Any,
        *,
        external_acceleration_fn: Optional[Callable[..., Array]] = None,
        repartition_every: int = 0,
        num_samples: int = 256,
        donate: bool = True,
        axis_name: str = AXIS_NAME,
        migrate: bool = False,
        migration_cap: Any = "auto",
        splitters: Any = None,
    ) -> None:
        from jaccpot.distributed.rollout import _mesh_devices

        require_x64("MeshSegmentRunner")
        self.mesh = mesh
        self.devices = _mesh_devices(mesh)
        self.ndev = len(self.devices)
        rows = int(state.state.shape[0])
        if rows % self.ndev:
            raise ValueError(f"{rows} rows do not split over {self.ndev} devices")
        self.cap = rows // self.ndev
        self.force = force
        #: mac_type='dehnen_error': f_b rides the carry (the force says so)
        self.scale_carry = bool(getattr(force, "scale_carry", False))
        self.external_acceleration_fn = external_acceleration_fn
        self.repartition_every = int(repartition_every)
        self.num_samples = int(num_samples)
        self.donate = bool(donate)
        self.axis_name = axis_name
        self.state = state
        self.segment_index = 0
        self.steps_done = 0
        self.reports: list[SegmentReport] = []
        #: what the carried acceleration was computed with (external_field_key)
        self._acceleration_key: Optional[tuple] = None
        #: the segment whose repartition already ran (a start() before it)
        self._repartitioned_for = 0
        #: migration to fixed cuts (one device: nothing to migrate)
        self.migrate = bool(migrate) and self.ndev > 1
        self.migration_cap: Optional[int] = None
        if migration_cap != "auto":
            self.migration_cap = int(migration_cap)
            if self.migration_cap < 1:
                raise ValueError(f"migration_cap must be >= 1, got {migration_cap}")
        #: the box shape the force builds its trees in, which the cuts live in
        self._migration_cubic = _force_box_cubic(force)
        self.splitters: Any = None
        if self.migrate:
            if splitters is None:
                # the cuts are chosen in the force's box: a decomposition in any
                # other frame is not a Morton range there
                self.state, diag = self._repartition(self.state)
                if bool(diag["declined"][0]):
                    raise ValueError(
                        "migrate=True: the repartition into the force's box was "
                        "declined: some device would receive "
                        f"{int(np.asarray(diag['recv_counts']).max())} rows > "
                        f"cap={self.cap}"
                    )
                state = self.state
            else:
                self.splitters = jax.device_put(
                    jnp.asarray(np.asarray(splitters, np.uint64)),
                    NamedSharding(mesh, P()),
                )
        self.force.prepare(state.state[:, 0], state.masses, state.count)
        if getattr(self.force, "needs_cross_sizing", False):
            # cross_caps="auto": the start force at ``state`` measures what every
            # cross capacity needs, and the library sizes them from it
            self.force.plan_cross(self._measure_cross)

    @classmethod
    def fused(
        cls,
        solver: Any,
        mesh: Any,
        state: MeshState,
        *,
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
        cross_caps: Any = "auto",
        near_theta: Optional[float] = None,
        cross_replan: Optional[Callable[[np.ndarray, Any], Any]] = None,
        level_floor: Optional[tuple[int, int]] = None,
        external_acceleration_fn: Optional[Callable[..., Array]] = None,
        repartition_every: int = 0,
        num_samples: int = 256,
        donate: bool = True,
        axis_name: str = AXIS_NAME,
        migrate: bool = False,
        migration_cap: Any = "auto",
    ) -> "MeshSegmentRunner":
        """A runner on the fused lane's force (the template route).

        Parameters
        ----------
        solver : Any
            The ``FastMultipoleMethod`` (large-N production profile, cell leaves,
            the flat walk; a geometric MAC or ``mac_type='dehnen_error'``).
        mesh : Any
            The 1-D mesh.
        state : MeshState
            The decomposed initial state.
        leaf_size : int
            Leaf target.
        max_order : int
            Expansion order.
        theta : Optional[float]
            Opening angle; ``None`` takes the solver's.
        cross_caps : Any
            ``"auto"`` (the default: sized in the library at ``state``), a
            :class:`~jaccpot.distributed.cross.CrossCapacities`, or ``None`` (its
            defaults); see :class:`~jaccpot.distributed.template_force.FusedMeshForce`.
        near_theta : Optional[float]
            Passed to the cross hook.
        cross_replan : Optional[Callable[[np.ndarray, Any], Any]]
            The cross capacities' re-plan; ``None``: the library's
            (:class:`~jaccpot.distributed.template_force.FusedMeshForce`).
        level_floor : Optional[tuple[int, int]]
            The level plan's floor (``FusedMeshForce``).
        external_acceleration_fn : Optional[Callable[..., Array]]
            As in the constructor.
        repartition_every : int
            As in the constructor.
        num_samples : int
            As in the constructor.
        donate : bool
            As in the constructor.
        axis_name : str
            Mesh axis.
        migrate : bool
            As in the constructor.
        migration_cap : Any
            As in the constructor.

        Returns
        -------
        MeshSegmentRunner
            Prepared at ``state``.

        Raises
        ------
        NotImplementedError
            If the solver asks for an error criterion other than the flat walk's
            eq (16a) (``mac_type='dehnen_error'`` runs); raised before anything is
            prepared.
        """
        from jaccpot.distributed.template_force import FusedMeshForce

        force = FusedMeshForce(
            solver,
            mesh,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
            cross_caps=cross_caps,
            near_theta=near_theta,
            cross_replan=cross_replan,
            level_floor=level_floor,
            axis_name=axis_name,
        )
        return cls(
            mesh,
            state,
            force,
            external_acceleration_fn=external_acceleration_fn,
            repartition_every=repartition_every,
            num_samples=num_samples,
            donate=donate,
            axis_name=axis_name,
            migrate=migrate,
            migration_cap=migration_cap,
        )

    # ------------------------------------------------------------------
    # The programs.
    # ------------------------------------------------------------------

    def _program(self, kind: str, steps: int, params: Any) -> tuple[Callable, bool]:
        """The compiled start (``kind="start"``) or segment program, cached.

        Parameters
        ----------
        kind : str
            ``"start"``, ``"measure"`` (the start force without the field) or
            ``"segment"``.
        steps : int
            The segment's steps (``0`` for the start programs).
        params : Any
            The external field's parameters (their structure is in the key).

        Returns
        -------
        tuple[Callable, bool]
            The program and whether it was just built (it traces at its first
            call).
        """
        leaves, treedef = jax.tree_util.tree_flatten(params)
        key = (
            "mesh_segment",
            kind,
            int(steps),
            tuple(int(d.id) for d in self.devices),
            self.axis_name,
            self.cap,
            str(self.state.state.dtype),
            str(self.state.masses.dtype),
            *self.force.key(),
            (
                None
                if self.external_acceleration_fn is None
                else id(self.external_acceleration_fn)
            ),
            str(treedef),
            tuple((tuple(np.shape(x)), str(jnp.result_type(x))) for x in leaves),
            self.donate,
            *self._migration_key(kind),
        )
        cache = _programs_of(self.force.cache_owner)
        program = cache.get(key)
        if program is not None:
            return program, False
        if kind == "segment" and self.migrate:
            program = self._build_migrating_segment(int(steps))
        elif kind == "segment":
            program = self._build_segment(int(steps))
        else:
            # "measure": the start force without the external field (the cross
            # needs do not depend on it, and its parameters are not known yet)
            program = self._build_start(int(steps), with_field=kind == "start")
        cache[key] = program
        return program, True

    def _build_start(self, steps: int, *, with_field: bool = True) -> Callable:
        """The start force: the total acceleration at the current positions.

        Parameters
        ----------
        steps : int
            Unused (the key's slot).
        with_field : bool
            Add the external field (``False``: the cross sizing's measurement).

        Returns
        -------
        Callable
            ``(state, masses, count, ids, params) -> (acceleration, ok, bits,
            needs)``; the acceleration flat, the rest replicated. Under the
            criterion (:attr:`scale_carry`) ``(state, masses, count, ids, f_b,
            params) -> (acceleration, f_b, ok, bits, needs)``: the force's
            thresholds from the given ``f_b``, and the force's own.
        """
        from jaccpot.runtime.fused_topology import overflow_ok

        del steps
        # locals, not ``self``: the program outlives this runner in the engine's
        # cache, and a closure over ``self`` would keep its state's buffers alive
        axis, force = self.axis_name, self.force
        field_fn = self.external_acceleration_fn if with_field else None

        def start_force(
            state: Array,
            masses: Array,
            count: Array,
            ids: Array,
            params: Any,
            f_b: Optional[Array],
        ) -> tuple:
            cap = int(state.shape[0])
            live = jnp.minimum(count[0], cap)
            live_rows = jnp.arange(cap) < live
            out = force.shard(
                state[:, 0, :],
                masses,
                live,
                ids,
                **({} if f_b is None else {"force_scale": f_b}),
            )
            acc_self, bits, needs = out[:3]
            acc = _total_acceleration(field_fn, acc_self, state, params, live_rows)
            # a count above the shard's capacity cannot be real particles
            ok = overflow_ok(bits) & (count[0] <= cap)
            ok, bits, needs, _ = _mesh_verdict(
                ok, bits, needs, jnp.int32(0), axis_name=axis
            )
            if f_b is None:
                return acc, ok, bits, needs
            return acc, out[3], ok, bits, needs

        spec = P(axis)
        if not self.scale_carry:

            def body(
                state: Array, masses: Array, count: Array, ids: Array, params: Any
            ) -> tuple:
                return start_force(state, masses, count, ids, params, None)

            in_specs: tuple = (spec, spec, spec, spec, P())
            out_specs: tuple = (spec, P(), P(), P())
        else:

            def body(  # type: ignore[misc]
                state: Array,
                masses: Array,
                count: Array,
                ids: Array,
                f_b: Array,
                params: Any,
            ) -> tuple:
                return start_force(state, masses, count, ids, params, f_b)

            in_specs = (spec, spec, spec, spec, spec, P())
            out_specs = (spec, spec, P(), P(), P())
        return jax.jit(
            jax.shard_map(
                body,
                mesh=self.mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )
        )

    def _build_segment(self, steps: int) -> Callable:
        """``steps`` velocity-Verlet steps as one program, state and acceleration donated.

        Parameters
        ----------
        steps : int
            Steps of the scan (static).

        Returns
        -------
        Callable
            ``(state, acceleration, masses, count, ids, dt, params) -> (state,
            acceleration, ok, bits, needs, done)``. Under the criterion
            (:attr:`scale_carry`) ``f_b`` rides the carry from each force to the
            next one's thresholds: ``(state, acceleration, f_b, masses, count, ids,
            dt, params) -> (state, acceleration, f_b, ok, bits, needs, done)``,
            ``f_b`` donated with the state.
        """
        from jaccpot.runtime.fmm_state import _velocity_verlet_kick_drifted
        from jaccpot.runtime.fused_topology import overflow_ok

        # locals, not ``self`` (see _build_start)
        axis, force, field_fn = (
            self.axis_name,
            self.force,
            self.external_acceleration_fn,
        )
        needs_size = int(force.needs_size)
        scale_carry = self.scale_carry

        def segment(
            state0: Array,
            acc0: Array,
            fb0: Optional[Array],
            masses: Array,
            count: Array,
            ids: Array,
            dt: Array,
            params: Any,
        ) -> tuple:
            cap = int(state0.shape[0])
            live = jnp.minimum(count[0], cap)
            live_rows = jnp.arange(cap) < live

            def step(carry: tuple, _: Any) -> tuple:
                state_now, acc_now, ok, bits, needs, done = carry[:6]
                # strict_run_v2's `_advance`: drift, force, kick of the drifted state
                position_new = (
                    state_now[:, 0] + state_now[:, 1] * dt + 0.5 * acc_now * dt**2
                )
                state_position = state_now.at[:, 0].set(position_new)
                out = force.shard(
                    state_position[:, 0, :],
                    masses,
                    live,
                    ids,
                    **({"force_scale": carry[6]} if scale_carry else {}),
                )
                acc_self, bits_new, needs_new = out[:3]
                acc_new = _total_acceleration(
                    field_fn, acc_self, state_position, params, live_rows
                )
                state_new = _velocity_verlet_kick_drifted(
                    state_position, acc_now, acc_new, dt
                )
                # Fold only what a step entered ON the trajectory needed. After the
                # first failure the forces come from truncated or misaligned lists,
                # the states leave the trajectory, and what their trees and walks
                # "need" sizes nothing real: a failed 8e6 segment on two A100s went
                # NaN by its fourth step, and the degenerate trees after it read a
                # whole shard per cross row (leaf width 4,002,489; the re-plan then
                # OOMed). `ok` is the MESH's: a device whose own flags held still
                # ran on a failed peer's exchange.
                ok_new = _mesh_all(ok & overflow_ok(bits_new), axis)
                return (
                    state_new,
                    acc_new,
                    ok_new,
                    bits | jnp.where(ok, bits_new, jnp.uint32(0)),
                    jnp.where(ok, jnp.maximum(needs, needs_new), needs),
                    done + ok_new.astype(jnp.int32),
                    *out[3:],
                ), None

            init = (
                state0,
                acc0,
                _mesh_all(count[0] <= cap, axis),
                jnp.uint32(0),
                jnp.zeros((needs_size,), jnp.int32),
                jnp.int32(0),
                *(() if fb0 is None else (fb0,)),
            )
            carry_f, _ = jax.lax.scan(step, init, None, length=int(steps))
            state_f, acc_f, ok, bits, needs, done = carry_f[:6]
            return (state_f, acc_f, *carry_f[6:]) + _mesh_verdict(
                ok, bits, needs, done, axis_name=axis
            )

        spec = P(axis)
        if not scale_carry:

            def body(
                state0: Array,
                acc0: Array,
                masses: Array,
                count: Array,
                ids: Array,
                dt: Array,
                params: Any,
            ) -> tuple:
                return segment(state0, acc0, None, masses, count, ids, dt, params)

            in_specs: tuple = (spec, spec, spec, spec, spec, P(), P())
            out_specs: tuple = (spec, spec, P(), P(), P(), P())
            donate: tuple = (0, 1)
        else:

            def body(  # type: ignore[misc]
                state0: Array,
                acc0: Array,
                fb0: Array,
                masses: Array,
                count: Array,
                ids: Array,
                dt: Array,
                params: Any,
            ) -> tuple:
                return segment(state0, acc0, fb0, masses, count, ids, dt, params)

            in_specs = (spec, spec, spec, spec, spec, spec, P(), P())
            out_specs = (spec, spec, spec, P(), P(), P(), P())
            donate = (0, 1, 2)
        return jax.jit(
            jax.shard_map(
                body,
                mesh=self.mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            ),
            donate_argnums=donate if self.donate else (),
        )

    def _migration_key(self, kind: str) -> tuple:
        """The compile key's migration part: empty unless a segment migrates.

        Parameters
        ----------
        kind : str
            The program's kind.

        Returns
        -------
        tuple
            ``()``, or ``("migrate", migration_cap, cubic)`` for a migrating
            segment.
        """
        if kind != "segment" or not self.migrate:
            return ()
        return ("migrate", int(self.migration_cap or 0), self._migration_cubic)

    def _build_migrating_segment(self, steps: int) -> Callable:
        """:meth:`_build_segment` with the migration between each drift and force.

        The step: the drift, the migration to the fixed cuts
        (:func:`~jaccpot.distributed.migration.migrate_shard`, in the force's box;
        the rows -- state, total acceleration, mass, id, ``f_b`` -- move with their
        particles), the force on the migrated rows, the kick. The counts, masses
        and ids are part of the carry. A declined migration fails its step (its
        bits on the trajectory); the force's bits and needs of that step are not
        folded (its ownership is not the trajectory's), so a failed attempt is
        either the migration's or the force's.

        Parameters
        ----------
        steps : int
            Steps of the scan (static).

        Returns
        -------
        Callable
            ``(state, acceleration, [f_b,] masses, ids, count, splitters, dt,
            params) -> (state, acceleration, [f_b,] masses, ids, count, ok, bits,
            needs, done, migration_need, migrated)``; everything up to ``count``
            flat and donated, the rest replicated.
        """
        from jaccpot.distributed.migration import migrate_shard
        from jaccpot.runtime.fmm_state import _velocity_verlet_kick_drifted
        from jaccpot.runtime.fused_topology import overflow_ok

        # locals, not ``self`` (see _build_start)
        axis, force, field_fn = (
            self.axis_name,
            self.force,
            self.external_acceleration_fn,
        )
        needs_size = int(force.needs_size)
        scale_carry = self.scale_carry
        mcap, cubic = int(self.migration_cap or 0), self._migration_cubic
        zero_bits = jnp.uint32(0)

        def segment(
            state0: Array,
            acc0: Array,
            fb0: Optional[Array],
            masses0: Array,
            ids0: Array,
            count0: Array,
            splitters: Array,
            dt: Array,
            params: Any,
        ) -> tuple:
            cap = int(state0.shape[0])

            def step(carry: tuple, _: Any) -> tuple:
                state_now, acc_now, masses, ids, count = carry[:5]
                ok, bits, needs, done, need_m, moved_m = carry[5:11]
                fb_now = carry[11] if scale_carry else None
                position_new = (
                    state_now[:, 0] + state_now[:, 1] * dt + 0.5 * acc_now * dt**2
                )
                state_position = state_now.at[:, 0].set(position_new)
                state_position, acc_now, masses, ids, count, fb_now, info = (
                    migrate_shard(
                        state_position,
                        acc_now,
                        masses,
                        ids,
                        count,
                        splitters,
                        migration_cap=mcap,
                        cubic=cubic,
                        force_scale=fb_now,
                        axis_name=axis,
                    )
                )
                # the migration's verdict is replicated (the all-gathered sizes)
                ok_mig = ok & (info["bits"] == zero_bits)
                live = jnp.minimum(count[0], cap)
                live_rows = jnp.arange(cap) < live
                out = force.shard(
                    state_position[:, 0, :],
                    masses,
                    live,
                    ids,
                    **({"force_scale": fb_now} if scale_carry else {}),
                )
                acc_self, bits_new, needs_new = out[:3]
                acc_new = _total_acceleration(
                    field_fn, acc_self, state_position, params, live_rows
                )
                state_new = _velocity_verlet_kick_drifted(
                    state_position, acc_now, acc_new, dt
                )
                # fold only what a step entered ON the trajectory needed (see
                # _build_segment); a declined migration ends the trajectory before
                # its force
                ok_new = _mesh_all(ok_mig & overflow_ok(bits_new), axis)
                return (
                    state_new,
                    acc_new,
                    masses,
                    ids,
                    count,
                    ok_new,
                    bits
                    | jnp.where(ok, info["bits"], zero_bits)
                    | jnp.where(ok_mig, bits_new, zero_bits),
                    jnp.where(ok_mig, jnp.maximum(needs, needs_new), needs),
                    done + ok_new.astype(jnp.int32),
                    jnp.where(ok, jnp.maximum(need_m, info["need"]), need_m),
                    moved_m + jnp.where(ok_mig, info["moved"], 0),
                    *out[3:],
                ), None

            init = (
                state0,
                acc0,
                masses0,
                ids0,
                count0,
                _mesh_all(count0[0] <= cap, axis),
                zero_bits,
                jnp.zeros((needs_size,), jnp.int32),
                jnp.int32(0),
                jnp.int32(0),
                jnp.int32(0),
                *(() if fb0 is None else (fb0,)),
            )
            carry_f, _ = jax.lax.scan(step, init, None, length=int(steps))
            state_f, acc_f, masses_f, ids_f, count_f = carry_f[:5]
            ok, bits, needs, done, need_m, moved_m = carry_f[5:11]
            return (
                (state_f, acc_f, *carry_f[11:], masses_f, ids_f, count_f)
                + _mesh_verdict(ok, bits, needs, done, axis_name=axis)
                + (need_m, moved_m)
            )

        spec = P(axis)
        rows = 5 if not scale_carry else 6
        if not scale_carry:

            def body(
                state0: Array,
                acc0: Array,
                masses0: Array,
                ids0: Array,
                count0: Array,
                splitters: Array,
                dt: Array,
                params: Any,
            ) -> tuple:
                return segment(
                    state0, acc0, None, masses0, ids0, count0, splitters, dt, params
                )

        else:

            def body(  # type: ignore[misc]
                state0: Array,
                acc0: Array,
                fb0: Array,
                masses0: Array,
                ids0: Array,
                count0: Array,
                splitters: Array,
                dt: Array,
                params: Any,
            ) -> tuple:
                return segment(
                    state0, acc0, fb0, masses0, ids0, count0, splitters, dt, params
                )

        return jax.jit(
            jax.shard_map(
                body,
                mesh=self.mesh,
                in_specs=(spec,) * rows + (P(), P(), P()),
                out_specs=(spec,) * rows + (P(),) * 6,
                check_vma=False,
            ),
            donate_argnums=tuple(range(rows)) if self.donate else (),
        )

    def _repartition_program(self) -> Callable:
        """The mesh repartition of the packed state, cached.

        ``rollout._repartition_shard`` (``sfc_repartition`` in the global box) with
        the velocity, the TOTAL acceleration and the id routed with each particle:
        the next segment's first drift and kick read the acceleration the last one
        ended with, row by row. Not donated: the ragged exchange's outputs must not
        alias its inputs.

        Under ``migrate=True`` the decomposition box is the FORCE's
        (:func:`_force_box_cubic`; the migration's cuts live there) and the program
        also returns the cuts of the new ownership
        (:func:`~jaccpot.distributed.migration.splitters_shard`) after ``diag``.

        Returns
        -------
        Callable
            ``(state, acceleration, masses, ids, count) -> (state, acceleration,
            masses, ids, count, diag)``; under the criterion (:attr:`scale_carry`)
            ``f_b`` is routed too: ``(state, acceleration, f_b, masses, ids, count)
            -> (state, acceleration, f_b, masses, ids, count, diag)`` (the next
            force's thresholds are its particles', wherever they now live).
        """
        from jaccpot.distributed.migration import splitters_shard
        from jaccpot.distributed.rollout import _repartition_shard

        scale_carry = self.scale_carry
        migrate, cubic = self.migrate, self._migration_cubic
        key = (
            "mesh_repartition",
            tuple(int(d.id) for d in self.devices),
            self.axis_name,
            self.cap,
            self.num_samples,
            str(self.state.state.dtype),
            *(("force_scale",) if scale_carry else ()),
            *(("migrate", cubic) if migrate else ()),
        )
        cache = _programs_of(self.force.cache_owner)
        program = cache.get(key)
        if program is not None:
            return program
        ndev, cap, samples, axis = self.ndev, self.cap, self.num_samples, self.axis_name

        def repartition(
            state: Array,
            acc: Array,
            f_b: Optional[Array],
            masses: Array,
            ids: Array,
            count: Array,
        ) -> tuple:
            x, v, m, i, c, a, diag, *routed = _repartition_shard(
                state[:, 0],
                state[:, 1],
                masses,
                ids,
                count,
                ndev=ndev,
                cap=cap,
                num_samples=samples,
                axis_name=axis,
                acceleration=acc,
                **({} if f_b is None else {"force_scale": f_b}),
                **({"cubic": cubic} if migrate else {}),
            )
            out = (jnp.stack([x, v], axis=1), a, *routed, m, i, c, diag)
            if not migrate:
                return out
            return out + (splitters_shard(x, c, cubic=cubic, axis_name=axis),)

        spec = P(axis)
        diag_spec = {
            "declined": spec,
            "recv_counts": spec,
            "sent_off_device": spec,
            "max_util": spec,
        }
        if not scale_carry:

            def body(
                state: Array, acc: Array, masses: Array, ids: Array, count: Array
            ) -> tuple:
                return repartition(state, acc, None, masses, ids, count)

            arrays = 5
        else:

            def body(  # type: ignore[misc]
                state: Array,
                acc: Array,
                f_b: Array,
                masses: Array,
                ids: Array,
                count: Array,
            ) -> tuple:
                return repartition(state, acc, f_b, masses, ids, count)

            arrays = 6
        program = jax.jit(
            jax.shard_map(
                body,
                mesh=self.mesh,
                in_specs=(spec,) * arrays,
                out_specs=(spec,) * arrays + (diag_spec,) + ((P(),) if migrate else ()),
                check_vma=False,
            )
        )
        cache[key] = program
        return program

    def _repartition(self, state: MeshState) -> tuple[MeshState, dict]:
        """Repartition ``state`` on the mesh (the acceleration routed along).

        A declined repartition (a device would overflow its capacity) moves
        nothing and leaves a valid partition: the run goes on in the old
        ownership, and the report says so.

        Parameters
        ----------
        state : MeshState
            The state at a segment edge (its acceleration computed).

        Returns
        -------
        tuple[MeshState, dict]
            The repartitioned state and the diagnostics on the host.
        """
        if self.migrate and self.splitters is not None:
            # exact ranges already: even cuts and a migration to them
            return self._rebalance(state)
        acc = state.acceleration
        if acc is None:
            acc = jnp.zeros(state.state.shape[:1] + (3,), state.state.dtype)
            acc = jax.device_put(acc, state.state.sharding)
        scale: tuple = ()
        if self.scale_carry:
            f_b = state.force_scale
            if f_b is None:
                f_b = jax.device_put(
                    jnp.zeros(state.state.shape[:1], state.state.dtype),
                    state.masses.sharding,
                )
            scale = (f_b,)
        out = self._repartition_program()(
            state.state, acc, *scale, state.masses, state.ids, state.count
        )
        if self.migrate:
            # the new ownership's cuts, which every migration until the next
            # repartition keeps
            self.splitters = out[-1]
            out = out[:-1]
        x, a = out[:2]
        f_b_new = out[2] if self.scale_carry else None
        m, i, c, diag = out[-4:]
        routed = MeshState(
            state=x,
            masses=m,
            ids=i,
            count=c,
            acceleration=None if state.acceleration is None else a,
            force_scale=None if state.force_scale is None else f_b_new,
        )
        return routed, {k: np.asarray(v) for k, v in diag.items()}

    def _measure_cross(self) -> np.ndarray:
        """The cross needs of the start force at the current state (host, mesh-reduced).

        Returns
        -------
        np.ndarray
            ``cross.CROSS_NEEDS_FIELDS`` of the force with the force's current hook.
        """
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS

        s = self.state
        needs = self._call(
            "measure", 0, None, s.state, s.masses, s.count, s.ids, *self._scale_args(s)
        )[-1]
        return np.asarray(jax.device_get(needs))[len(NEEDS_FIELDS) :]

    def _measure_cross_drifted(self, steps: int, dt: Any) -> Optional[np.ndarray]:
        """The start force's cross needs at the first segment's drifted positions.

        A decomposition (or a repartition) leaves the domains apart; a drift then
        carries each domain's boundary particles into Morton cells its neighbour
        fills, where they cut sparse leaves -- a few particles in a cell up to the
        tree's coarsest leaf level, their radius spanning it -- that fail the MAC
        against many of the neighbour's leaves. Every force of a segment comes at
        least one drift after the last cut; only the start force comes at none, so
        the start positions undersize the cross near field (Plummer on two A100s,
        near pairs / particles shipped: 2e5 11,603 / 36,231 at the start, 16,569 /
        70,250 after one step, up to 22,795 / 87,867 by step 5; 8e6 129,544 /
        430,677 at the start, 360,247 / 3,743,931 after one step, 337,626-621,796
        near pairs over steps 2-12). This measures the needs at ``x + s v dt`` for
        ``s`` in :func:`_drift_probe_steps` -- the drift alone, which is what moves
        the particles across the cuts (``a dt^2 / 2`` adds ~1e-3 of it at a
        stable step) -- with the ``"measure"`` program, compiled for the force's
        current hook, and max-es them over the probes. Under ``migrate=True`` each
        probe is migrated to the cuts first (:meth:`_migrated`), as every force of
        the segment is.

        Parameters
        ----------
        steps : int
            The first segment's steps.
        dt : Any
            Its time step.

        Returns
        -------
        Optional[np.ndarray]
            ``cross.CROSS_NEEDS_FIELDS`` max-ed over the probes (the bits OR-ed),
            mesh-reduced. The probes stop at the first whose own trees or walks
            overflowed (its cross needs would read a truncated tree): ``None`` if
            that was the first.
        """
        from jaccpot.runtime.fused_topology import (
            NEEDS_FIELDS,
            OVERFLOW_CROSS,
            OVERFLOW_FAIL_MASK,
        )

        s = self.state
        step = jnp.asarray(dt, s.state.dtype)
        worst: Optional[np.ndarray] = None
        for probe in _drift_probe_steps(int(steps)):
            moved = MeshState(
                state=_drifted(s.state, step * probe),
                masses=s.masses,
                ids=s.ids,
                count=s.count,
                force_scale=s.force_scale,
            )
            if self.migrate:
                moved = self._migrated(moved)
            out = self._call(
                "measure",
                0,
                None,
                moved.state,
                moved.masses,
                moved.count,
                moved.ids,
                *self._scale_args(moved),
            )
            bits, needs = jax.device_get(out[-2:])
            del out, moved
            if int(bits) & OVERFLOW_FAIL_MASK & ~OVERFLOW_CROSS:
                break
            cross = np.asarray(needs, np.int64)[len(NEEDS_FIELDS) :]
            worst = cross if worst is None else np.maximum(worst, cross)
        return worst

    def _migration_program(self, kind: str, migration_cap: int = 0) -> Callable:
        """The standalone migration programs (sizing, probes, rebalance), cached.

        Parameters
        ----------
        kind : str
            ``"need"``: ``(state, count, splitters) -> (need, moved)``, what a
            migration at the state's positions would move
            (:func:`~jaccpot.distributed.migration.migration_need_shard`);
            ``"cuts"``: ``(state, count) -> splitters``, the even cuts of an
            exact-range ownership
            (:func:`~jaccpot.distributed.migration.balanced_splitters_shard`);
            ``"move"``: ``(state, masses, ids, count, [f_b,] splitters) ->
            (state, masses, ids, count, [f_b,] need, moved, bits)``, the migration
            itself with a buffer of ``migration_cap`` rows and no acceleration
            (the cross sizing's probes measure a force); ``"rebalance"``: the same
            with the carried acceleration after the state, ``(state, acc, ...)
            -> (state, acc, ...)``.
        migration_cap : int
            ``"move"``'s and ``"rebalance"``'s buffer.

        Returns
        -------
        Callable
            The jitted ``shard_map`` program.
        """
        from jaccpot.distributed.migration import (
            balanced_splitters_shard,
            migrate_shard,
            migration_need_shard,
        )

        scale = self.scale_carry and kind in ("move", "rebalance")
        with_acc = kind == "rebalance"
        key = (
            "mesh_migration",
            kind,
            int(migration_cap),
            tuple(int(d.id) for d in self.devices),
            self.axis_name,
            self.cap,
            str(self.state.state.dtype),
            self._migration_cubic,
            scale,
        )
        cache = _programs_of(self.force.cache_owner)
        program = cache.get(key)
        if program is not None:
            return program
        axis, cubic, spec = self.axis_name, self._migration_cubic, P(self.axis_name)
        if kind == "need":

            def need(state: Array, count: Array, splitters: Array) -> tuple:
                return migration_need_shard(
                    state[:, 0], count, splitters, cubic=cubic, axis_name=axis
                )

            mapped = jax.shard_map(
                need,
                mesh=self.mesh,
                in_specs=(spec, spec, P()),
                out_specs=(P(), P()),
                check_vma=False,
            )
        elif kind == "cuts":

            def cuts(state: Array, count: Array) -> Array:
                return balanced_splitters_shard(
                    state[:, 0], count, cubic=cubic, axis_name=axis
                )

            mapped = jax.shard_map(
                cuts,
                mesh=self.mesh,
                in_specs=(spec, spec),
                out_specs=P(),
                check_vma=False,
            )
        else:

            def move(state: Array, *rest: Array) -> tuple:
                acc = rest[0] if with_acc else None
                masses, ids, count = rest[int(with_acc) : int(with_acc) + 3]
                f_b = rest[int(with_acc) + 3] if scale else None
                x, a, m, i, c, fb, info = migrate_shard(
                    state,
                    acc,
                    masses,
                    ids,
                    count,
                    rest[-1],
                    migration_cap=int(migration_cap),
                    cubic=cubic,
                    force_scale=f_b,
                    axis_name=axis,
                )
                head = (x,) + ((a,) if with_acc else ()) + (m, i, c)
                head = head + ((fb,) if scale else ())
                return head + (info["need"], info["moved"], info["bits"])

            rows = 4 + int(scale) + int(with_acc)
            mapped = jax.shard_map(
                move,
                mesh=self.mesh,
                in_specs=(spec,) * rows + (P(),),
                out_specs=(spec,) * rows + (P(),) * 3,
                check_vma=False,
            )
        # the rebalance consumes the edge's rows (a declined migration returns
        # them unchanged), so its outputs can take their buffers: one copy of the
        # rows less at the run's peak
        donate = tuple(range(rows)) if kind == "rebalance" and self.donate else ()
        program = jax.jit(mapped, donate_argnums=donate)
        cache[key] = program
        return program

    def _rebalance(self, state: MeshState) -> tuple[MeshState, dict]:
        """The repartition between segments under migration: even cuts, then a move.

        At a segment edge every device's rows are an exact Morton range in the
        force's box of the current positions (the last force's: a kick moves no
        position), so the even cuts are read off the ranks
        (:func:`~jaccpot.distributed.migration.balanced_splitters_shard`) and a
        migration to them moves only the rows that change owner, through a buffer
        sized to its exact need -- not ``sfc_repartition``, whose sample sort
        cannot rebalance exact ranges and whose exchange holds every row
        (cap-sized buffers: the run's peak once the migration shrank the cross
        field's; 8e6 Plummer on two A100s, 1.75 GB per device against 1.66-1.69
        GB in the segments).

        Parameters
        ----------
        state : MeshState
            The state at a segment edge, its acceleration computed.

        Returns
        -------
        tuple[MeshState, dict]
            The rebalanced state and ``_repartition``'s diagnostics
            (``declined``, ``recv_counts`` = the new counts, ``sent_off_device``
            on device 0, ``max_util``).
        """
        from jaccpot.distributed.migration import OVERFLOW_SHARD, migration_cap_for_need

        cuts = self._migration_program("cuts")(state.state, state.count)
        need, _ = self._migration_program("need")(state.state, state.count, cuts)
        # at least the segments' buffer: one program for every rebalance that
        # fits it (most of them, once the first has evened the counts)
        size = max(
            int(self.migration_cap or 0),
            migration_cap_for_need(int(jax.device_get(need)), self.cap),
        )
        acc = state.acceleration
        if acc is None:
            acc = jax.device_put(
                jnp.zeros(state.state.shape[:1] + (3,), state.state.dtype),
                state.state.sharding,
            )
        scale: tuple = ()
        if self.scale_carry:
            f_b = state.force_scale
            if f_b is None:
                f_b = jax.device_put(
                    jnp.zeros(state.state.shape[:1], state.state.dtype),
                    state.masses.sharding,
                )
            scale = (f_b,)
        out = self._migration_program("rebalance", size)(
            state.state, acc, state.masses, state.ids, state.count, *scale, cuts
        )
        moved, bits = (int(v) for v in jax.device_get(out[-2:]))
        declined = bool(bits & OVERFLOW_SHARD)
        if not declined:
            # (even cuts cannot overfill a shard; declined, the rows came back
            # unchanged and the old cuts stand)
            self.splitters = cuts
        counts = np.asarray(out[4])
        sent = np.zeros(self.ndev, np.int64)
        sent[0] = moved
        routed = MeshState(
            state=out[0],
            masses=out[2],
            ids=out[3],
            count=out[4],
            acceleration=None if state.acceleration is None else out[1],
            force_scale=(
                out[5] if self.scale_carry and state.force_scale is not None else None
            ),
        )
        return routed, {
            "declined": np.full(self.ndev, declined),
            "recv_counts": counts[None].repeat(self.ndev, 0),
            "sent_off_device": sent,
            "max_util": np.full(self.ndev, counts.max() / self.cap, np.float32),
        }

    def _migration_need(self, state: MeshState) -> int:
        """The most rows a device would send or receive migrating ``state``.

        Parameters
        ----------
        state : MeshState
            Positions (drifted) in the current ownership.

        Returns
        -------
        int
            The need (host).
        """
        need, _ = self._migration_program("need")(
            state.state, state.count, self.splitters
        )
        return int(jax.device_get(need))

    def _migrated(self, state: MeshState) -> MeshState:
        """``state`` migrated to the cuts, through a buffer its own need sizes.

        For the cross sizing's drifted probes (:meth:`_measure_cross_drifted`):
        the need first (:meth:`_migration_need`), then the migration with
        :func:`~jaccpot.distributed.migration.migration_cap_for_need`'s buffer for
        it (one compile per size), so it is never declined.

        Parameters
        ----------
        state : MeshState
            The drifted state (no acceleration moves; ``force_scale`` does).

        Returns
        -------
        MeshState
            The migrated rows, without acceleration.

        Raises
        ------
        RuntimeError
            If the migration would overfill a device's shard (``cap``).
        """
        from jaccpot.distributed.migration import OVERFLOW_SHARD, migration_cap_for_need

        size = migration_cap_for_need(self._migration_need(state), self.cap)
        scale = (state.force_scale,) if self.scale_carry else ()
        if self.scale_carry and state.force_scale is None:
            scale = (self.force.force_scale_seed,)
        out = self._migration_program("move", size)(
            state.state, state.masses, state.ids, state.count, *scale, self.splitters
        )
        if int(jax.device_get(out[-1])) & OVERFLOW_SHARD:
            raise RuntimeError(
                f"migrate=True: a drifted probe's migration would overfill a "
                f"device's shard (cap={self.cap}); raise the shard capacity"
            )
        return MeshState(
            state=out[0],
            masses=out[1],
            ids=out[2],
            count=out[3],
            force_scale=out[4] if self.scale_carry else None,
        )

    def _size_migration(self, dt: Any) -> None:
        """Size ``migration_cap="auto"`` from one step's drift at the current state.

        ``migration_cap_for_need`` of the rows the migration after one step's
        drift (``x + v dt``) would move: the boundary flux per step, which is
        steady; the factor 2 above it and the retry's exact re-plan cover the
        rest. Runs once, at the first segment.

        Parameters
        ----------
        dt : Any
            The time step.
        """
        from jaccpot.distributed.migration import migration_cap_for_need

        s = self.state
        moved = MeshState(
            state=_drifted(s.state, jnp.asarray(dt, s.state.dtype)),
            masses=s.masses,
            ids=s.ids,
            count=s.count,
        )
        self.migration_cap = migration_cap_for_need(
            self._migration_need(moved), self.cap
        )

    def _scale_args(self, state: MeshState) -> tuple:
        """The start program's ``f_b`` argument: the carried one, else the seed.

        Parameters
        ----------
        state : MeshState
            The state the start force runs at.

        Returns
        -------
        tuple
            ``()`` without the criterion; else ``(f_b,)``: ``state.force_scale``
            (the last force's), or before any force the prepare's seed
            (``force.force_scale_seed``).
        """
        if not self.scale_carry:
            return ()
        if state.force_scale is not None:
            return (state.force_scale,)
        return (self.force.force_scale_seed,)

    def _call(self, kind: str, steps: int, params: Any, *args: Any) -> tuple:
        """Run a program under the force's context (its plan installed).

        Parameters
        ----------
        kind : str
            ``"start"`` or ``"segment"``.
        steps : int
            The segment's steps.
        params : Any
            The external field's parameters (the program's last argument).
        *args : Any
            The program's other arguments.

        Returns
        -------
        tuple
            The program's outputs.
        """
        program, fresh = self._program(kind, steps, params)
        with self.force.context():
            if fresh:
                self.force.before_trace()
            return program(*args, params)

    # ------------------------------------------------------------------
    # The host loop.
    # ------------------------------------------------------------------

    def start(self, *, external_params: Any = None) -> SegmentReport:
        """The start force alone: the total acceleration at the current positions.

        :meth:`run` computes it when it needs one (the first segment, a changed
        external field); this computes it now, without stepping -- re-planned
        like a segment if it overflows -- and the next :meth:`run` starts from
        it.

        Parameters
        ----------
        external_params : Any
            The external field's parameters, as :meth:`run` will get them.

        Returns
        -------
        SegmentReport
            ``steps = 0``; ``segment`` is the segment the force starts.

        Raises
        ------
        RuntimeError
            As :meth:`run`.
        """
        return self._advance(0, None, external_params)

    def run(self, steps: int, *, dt: Any, external_params: Any = None) -> SegmentReport:
        """Advance one segment of ``steps`` steps.

        Parameters
        ----------
        steps : int
            Steps in this segment (at least 1).
        dt : Any
            Time step; an argument of the program (one compile for every value).
        external_params : Any
            The external field's parameters (any pytree; replicated), or ``None``
            for a one-argument field.

        Returns
        -------
        SegmentReport
            What the segment did.

        Raises
        ------
        ValueError
            If ``steps`` is below 1.
        RuntimeError
            If a capacity flag fired and the segment could not be re-planned (a
            named walk cap, a cross overflow without a re-plan, a third failure,
            or ``JACCPOT_STRICT_SEGMENT_RETRY=0``).
        """
        if int(steps) < 1:
            raise ValueError(f"a segment needs at least one step, got {steps}")
        return self._advance(int(steps), dt, external_params)

    def _advance(self, k: int, dt: Any, external_params: Any) -> SegmentReport:
        """:meth:`run` (``k >= 1``) or :meth:`start` (``k = 0``), with the retry.

        Parameters
        ----------
        k : int
            Steps; ``0`` computes the start force only.
        dt : Any
            Time step (unused when ``k = 0``).
        external_params : Any
            The external field's parameters.

        Returns
        -------
        SegmentReport
            What ran.
        """
        from jaccpot._env import env_flag
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS
        from jaccpot.runtime.strict_carry import external_field_key, host_copy

        n = self.segment_index + 1
        every = self.repartition_every
        # once per segment: a start() before the segment already ran it
        repartitioned = (
            every > 0
            and n > 1
            and (n - 1) % every == 0
            and self._repartitioned_for != n
        )
        diag: dict = {}
        if repartitioned:
            self.state, diag = self._repartition(self.state)
            self._repartitioned_for = n
        state = self.state
        key = external_field_key(self.external_acceleration_fn, external_params)
        start = state.acceleration is None or key != self._acceleration_key
        if k == 0 and not start:
            return SegmentReport(segment=n, steps=0, start_force=False)
        if k > 0 and self.migrate and self.migration_cap is None:
            # migration_cap="auto": one step's drift says how many rows move
            self._size_migration(dt)
        if k > 0 and getattr(self.force, "needs_drift_sizing", False):
            # cross_caps="auto" was sized at the start positions; the first
            # segment's dt sizes it at the drifted ones too
            # (`_measure_cross_drifted`). Capacities grown after a start() are a
            # different program: its force again, as a fresh run with them
            # computes it (from the seed, under the criterion)
            grown = self.force.plan_cross_drift(
                lambda: self._measure_cross_drifted(k, dt)
            )
            if grown and not start:
                state = MeshState(
                    state=state.state,
                    masses=state.masses,
                    ids=state.ids,
                    count=state.count,
                )
                start = True
        started = start
        acceleration = state.acceleration
        # the criterion's f_b: the start force's thresholds (carried, else the
        # prepare's seed) and, after it, the segment's first
        scale_now: tuple = self._scale_args(state)
        retry_enabled = env_flag("JACCPOT_STRICT_SEGMENT_RETRY", True)
        start_host: Any = None
        rows_host: tuple = ()
        if k > 0 and self.donate and retry_enabled:
            # the segment consumes its start state, and a retry re-runs the segment
            # from it: one copy per segment, off the device (on a GPU asynchronous,
            # into pinned memory), as strict_run_v2's particle carry keeps it
            start_host = host_copy(state.state)
            if self.migrate:
                # a migrating segment also consumes the rows' masses, ids, counts
                rows_host = tuple(
                    host_copy(a) for a in (state.masses, state.ids, state.count)
                )
        state_new = state.state
        rows_new = (state.masses, state.ids, state.count)
        acc_new: Optional[Array] = None
        scale_new: tuple = ()
        retries, tree_first = 0, False
        # the force's own policy counts its retries; a migration re-plan is exact
        force_retries = 0
        migrated_h, migration_need_h = 0, 0
        failures: list = []
        while True:
            consumed = False
            if start:
                outputs = self._call(
                    "start",
                    0,
                    external_params,
                    state.state,
                    state.masses,
                    state.count,
                    state.ids,
                    *scale_now,
                )
                acceleration = outputs[0]
                # the start force's own f_b: the segment's first thresholds
                scale_now = tuple(outputs[1:-3])
                ok_h, bits_h, needs_h = jax.device_get(outputs[-3:])
                del outputs
                done_h = 0
                if k == 0 and bool(ok_h):
                    acc_new, scale_new = acceleration, scale_now
                    break
            if k > 0 and (not start or bool(ok_h)) and self.migrate:
                outputs = self._call(
                    "segment",
                    k,
                    external_params,
                    state.state,
                    acceleration,
                    *scale_now,
                    state.masses,
                    state.ids,
                    state.count,
                    self.splitters,
                    jnp.asarray(dt, state.state.dtype),
                )
                state_new, acc_new = outputs[:2]
                scale_new = tuple(outputs[2:-9])
                rows_new = tuple(outputs[-9:-6])
                # the segment's one host sync
                ok_h, bits_h, needs_h, done_h, migration_need_h, migrated_h = (
                    jax.device_get(outputs[-6:])
                )
                del outputs
                consumed = self.donate
                if bool(ok_h):
                    break
            elif k > 0 and (not start or bool(ok_h)):
                outputs = self._call(
                    "segment",
                    k,
                    external_params,
                    state.state,
                    acceleration,
                    *scale_now,
                    state.masses,
                    state.count,
                    state.ids,
                    jnp.asarray(dt, state.state.dtype),
                )
                state_new, acc_new = outputs[:2]
                scale_new = tuple(outputs[2:-4])
                # the segment's one host sync
                ok_h, bits_h, needs_h, done_h = jax.device_get(outputs[-4:])
                del outputs
                consumed = self.donate
                if bool(ok_h):
                    break
            # A fired flag. Re-plan from what the segment needed (the walk caps,
            # the tree's bounds, the cross capacities through the force's seam),
            # re-prepare at the segment's START and run it again from its start
            # force: at most twice, the single card's policy, then while it makes
            # progress (below).
            needs_np = np.asarray(needs_h, np.int64)
            failures.append(
                dict(bits=int(bits_h), needs=needs_np.tolist(), done=int(done_h))
            )
            where = f"segment {n} ({int(done_h)} of {k} steps fitted)"
            migration_failed = self.migrate and self._migration_failed(
                int(bits_h), int(migration_need_h), where, retry_enabled
            )
            if migration_failed:
                failures[-1]["migration_need"] = int(migration_need_h)
                mode = "migration"
            else:
                mode = (
                    self.force.retry_mode(needs_np, force_retries, tree_first)
                    if retry_enabled
                    else None
                )
            force_failures = [f for f in failures[:-1] if "migration_need" not in f]
            if (
                mode is None
                and retry_enabled
                and _retry_made_progress(needs_np, int(done_h), force_failures)
            ):
                # PROGRESS (the single card's `_segment_retry_next`): this re-run
                # fitted more steps than every earlier attempt, or it failed at the
                # same step as the attempt before, whose needs were lower bounds,
                # and needed more than they said. The re-plan reads only the steps
                # up to the first failure (the rest left the trajectory), so a
                # segment whose needs grow step by step is learned one failing step
                # at a time: the outgrowing-segment gate under
                # mac_type='dehnen_error' ran out of the force's two retries at 1 of
                # 2 steps fitted. `done` grows with every retry of the first kind,
                # so there are at most the segment's steps of them; each of the
                # second kind at least doubles a capacity a stopped walk filled.
                tree_now = bool(needs_np[NEEDS_FIELDS.index("tree_overflow")])
                mode = "full" if tree_now else "walk"
            if mode is None:
                self.force.raise_overflow(
                    needs_np, int(bits_h), retries=force_retries, where=where
                )
            if mode != "migration":
                if force_retries == 0:
                    tree_first = bool(needs_np[NEEDS_FIELDS.index("tree_overflow")])
                force_retries += 1
            retries += 1
            acceleration = acc_new = None
            scale_new = ()
            state_new = state.state
            if consumed:
                assert start_host is not None
                sharding = state.masses.sharding
                if rows_host:
                    masses_r, ids_r, count_r = (
                        jax.device_put(a, sharding) for a in rows_host
                    )
                else:
                    masses_r, ids_r, count_r = state.masses, state.ids, state.count
                state = MeshState(
                    state=jax.device_put(start_host, sharding),
                    masses=masses_r,
                    ids=ids_r,
                    count=count_r,
                )
                state_new = state.state
            rows_new = (state.masses, state.ids, state.count)
            if mode != "migration" and not self.force.replan(
                needs_np, int(bits_h), walk_only=mode == "walk"
            ):
                self.force.raise_overflow(
                    needs_np, int(bits_h), retries=force_retries - 1, where=where
                )
            # a migration re-plan re-prepares too: under the criterion the start
            # force's seed is the prepare's, in the segment start's row order
            self.force.prepare(state.state[:, 0], state.masses, state.count)
            # the start force again, with the grown capacities, as a fresh run's:
            # under the criterion from the re-prepare's own seed at the segment's
            # start (a fresh run there has no carried f_b), as the single card's
            # retry reseeds
            scale_now = self._scale_args(
                MeshState(
                    state=state.state,
                    masses=state.masses,
                    ids=state.ids,
                    count=state.count,
                )
            )
            start = True
        self.state = MeshState(
            state=state_new,
            masses=rows_new[0],
            ids=rows_new[1],
            count=rows_new[2],
            acceleration=acc_new,
            force_scale=scale_new[0] if scale_new else None,
        )
        self._acceleration_key = key
        if k > 0:
            self.segment_index = n
            self.steps_done += k
        report = SegmentReport(
            segment=n,
            steps=k,
            start_force=started,
            retries=retries,
            repartitioned=repartitioned,
            declined=bool(diag["declined"][0]) if diag else False,
            sent_off_device=int(diag["sent_off_device"].sum()) if diag else 0,
            counts=[int(c) for c in np.asarray(self.state.count)],
            bits=int(bits_h),
            needs=np.asarray(needs_h).tolist(),
            failures=failures,
            migrated=int(migrated_h),
            migration_need=int(migration_need_h),
        )
        if k > 0:
            self.reports.append(report)
        return report

    def _migration_failed(
        self, bits: int, need: int, where: str, retry_enabled: bool
    ) -> bool:
        """Whether a failed attempt was its migration's, re-planned here.

        The first failing step of an attempt is either a declined migration or
        the force's (the migrating segment folds no force verdict after a decline).
        :data:`~jaccpot.distributed.migration.OVERFLOW_MIGRATION` re-sizes the
        buffer from the exact need
        (:func:`~jaccpot.distributed.migration.migration_cap_for_need`; it always
        grows, at most to the shard's ``cap``) and the caller re-runs the segment;
        :data:`~jaccpot.distributed.migration.OVERFLOW_SHARD` (a device's rows past
        its shard) has no re-plan and raises.

        Parameters
        ----------
        bits : int
            The attempt's bits (mesh-reduced).
        need : int
            Its migration need (rows).
        where : str
            Which segment, for the message.
        retry_enabled : bool
            ``JACCPOT_STRICT_SEGMENT_RETRY``.

        Returns
        -------
        bool
            ``True`` when the buffer was re-sized for a re-run; ``False`` when the
            failure was the force's.

        Raises
        ------
        RuntimeError
            On a shard overflow, or a buffer overflow with the retry turned off.
        """
        from jaccpot.distributed.migration import (
            OVERFLOW_MIGRATION,
            OVERFLOW_SHARD,
            migration_cap_for_need,
        )

        if bits & OVERFLOW_SHARD:
            raise RuntimeError(
                f"mesh {where}: a migration would leave a device with more rows "
                f"than its shard holds (cap={self.cap}; bits {bits:#x}): the "
                "domains' counts drifted past the headroom within the segment. "
                "Give the shards more headroom (cap) or repartition more often."
            )
        if not bits & OVERFLOW_MIGRATION:
            return False
        if not retry_enabled:
            raise RuntimeError(
                f"mesh {where}: the migration needed {need} rows per device, its "
                f"buffer holds {self.migration_cap} (OVERFLOW_MIGRATION), and "
                "JACCPOT_STRICT_SEGMENT_RETRY=0 turns the re-plan off"
            )
        self.migration_cap = max(
            int(self.migration_cap or 0), migration_cap_for_need(need, self.cap)
        )
        return True

    def gather(self) -> dict:
        """The state on the host, by global id, with identity checked.

        Returns
        -------
        dict
            ``positions``, ``velocities``, ``masses`` and (once computed)
            ``acceleration`` and, under the criterion, ``force_scale`` as ``(N,
            ...)`` arrays indexed by global id, plus ``owner`` (device per id).

        Raises
        ------
        ValueError
            If an id is missing or appears twice.
        """
        s = self.state
        cap = self.cap
        count = np.asarray(s.count)
        rows = np.concatenate(
            [np.arange(d * cap, d * cap + int(count[d])) for d in range(self.ndev)]
        )
        live_ids = np.asarray(s.ids)[rows]
        n = int(count.sum())
        if live_ids.min() < 0 or len(np.unique(live_ids)) != n or live_ids.max() >= n:
            raise ValueError(
                "particle identity broken: ids are not exactly 0..N-1 once"
            )
        packed = np.asarray(s.state)[rows]
        arrays = {
            "positions": packed[:, 0],
            "velocities": packed[:, 1],
            "masses": np.asarray(s.masses)[rows],
        }
        if s.acceleration is not None:
            arrays["acceleration"] = np.asarray(s.acceleration)[rows]
        if s.force_scale is not None:
            arrays["force_scale"] = np.asarray(s.force_scale)[rows]
        out = {}
        for name, arr in arrays.items():
            full = np.empty((n,) + arr.shape[1:], arr.dtype)
            full[live_ids] = arr
            out[name] = full
        owner = np.empty(n, np.int32)
        owner[live_ids] = rows // cap
        out["owner"] = owner
        return out

    def resume(
        self, *, segment_index: int, steps_done: int, external_params: Any = None
    ) -> None:
        """Continue a run restored from a restart file: its counters, its carried total.

        A restore builds the runner from the file's :class:`MeshState` and then
        sets what the run that wrote it had reached: the segments done (the
        repartition cadence counts them), the steps, and -- when the state carries
        its total acceleration -- the external field it was computed with, so the
        next segment starts from it as the uninterrupted run did (no start force).

        Parameters
        ----------
        segment_index : int
            Segments the run had completed.
        steps_done : int
            Steps the run had completed.
        external_params : Any
            The external field's parameters the carried acceleration was computed
            with (the next segments' own), or ``None``.
        """
        from jaccpot.runtime.strict_carry import external_field_key

        self.segment_index = int(segment_index)
        self.steps_done = int(steps_done)
        self._repartitioned_for = 0
        self._acceleration_key = (
            None
            if self.state.acceleration is None
            else external_field_key(self.external_acceleration_fn, external_params)
        )

    # ------------------------------------------------------------------
    # The energy.
    # ------------------------------------------------------------------

    def potential_energy(
        self,
        *,
        external_potential_fn: Optional[Callable[..., Array]] = None,
        external_params: Any = None,
    ) -> dict:
        """The potential energies at the current state: ``1/2 sum m phi``, ``sum m Phi_ext``.

        One compiled program per plan (cached with the segment programs, under the
        force's key, the external potential's identity and the parameters'
        structure): the force's ``potential_shard`` on every device, ``phi_i = -G
        sum_{j != i} m_j psi(r_ij)`` over all devices' particles, and the external
        potential per row, both summed in fp64 per device and over the mesh. Under
        ``mac_type='dehnen_error'`` the rebuild's thresholds are the carried f_b
        (before any force, the prepare's seed), the next force's own.

        It changes nothing: the state is read, never donated, and the trace leaves
        the force's and its cache owner's trace-time state as it found them
        (``production._trace_state_kept``), so a run with the energy is the run
        without, bitwise.

        Parameters
        ----------
        external_potential_fn : Optional[Callable[..., Array]]
            The external field's potential per unit mass, ``potential(state_rows,
            params)`` (``potential(state_rows)`` without parameters) -> ``(cap,)``
            on one device's ``(cap, 2, 3)`` rows: the field's own contract. ``None``:
            ``W_ext = 0``. Keep ONE function object (its identity is in the key).
        external_params : Any
            Its parameters (replicated), or ``None``.

        Returns
        -------
        dict
            Host values: ``self`` (``W_self``, fp64), ``external`` (``W_ext``) and
            ``ok`` (every device's potential fitted its capacities; the energies are
            meaningless otherwise).

        Raises
        ------
        NotImplementedError
            If the force has no ``potential_shard``.
        """
        from jaccpot.production import _trace_state_kept

        if getattr(self.force, "potential_shard", None) is None:
            raise NotImplementedError(
                f"{type(self.force).__name__} has no potential_shard: the mesh "
                "potential of this force is not implemented"
            )
        program, fresh = self._potential_program(external_potential_fn, external_params)
        s = self.state
        owners = [self.force.cache_owner]
        if self.force is not owners[0]:
            owners.append(self.force)
        with contextlib.ExitStack() as kept:
            # the program is in the cache already, so the cache's restore keeps it
            for owner in owners:
                kept.enter_context(_trace_state_kept(owner))
            with self.force.context():
                if fresh:
                    self.force.before_trace()
                out = program(
                    s.state,
                    s.masses,
                    s.count,
                    s.ids,
                    *self._scale_args(s),
                    external_params,
                )
        w_self, w_ext, ok, _ = jax.device_get(out)
        return {"self": float(w_self), "external": float(w_ext), "ok": bool(ok)}

    def _potential_program(
        self, field_fn: Optional[Callable[..., Array]], params: Any
    ) -> tuple[Callable, bool]:
        """The compiled potential-energy program, cached.

        Parameters
        ----------
        field_fn : Optional[Callable[..., Array]]
            The external potential, or ``None``.
        params : Any
            Its parameters (their structure is in the key).

        Returns
        -------
        tuple[Callable, bool]
            ``(state, masses, count, ids, [f_b,] params) -> (w_self, w_ext, ok,
            bits)``, replicated; and whether it was just built.
        """
        from jaccpot.runtime.fused_topology import overflow_ok

        leaves, treedef = jax.tree_util.tree_flatten(params)
        key = (
            "mesh_potential",
            tuple(int(d.id) for d in self.devices),
            self.axis_name,
            self.cap,
            str(self.state.state.dtype),
            str(self.state.masses.dtype),
            *self.force.key(),
            None if field_fn is None else id(field_fn),
            str(treedef),
            tuple((tuple(np.shape(x)), str(jnp.result_type(x))) for x in leaves),
        )
        cache = _programs_of(self.force.cache_owner)
        program = cache.get(key)
        if program is not None:
            return program, False
        # locals, not ``self`` (see _build_start)
        axis, force = self.axis_name, self.force

        def energy(
            state: Array,
            masses: Array,
            count: Array,
            ids: Array,
            f_b: Optional[Array],
            params_in: Any,
        ) -> tuple:
            cap = int(state.shape[0])
            live = jnp.minimum(count[0], cap)
            live_rows = jnp.arange(cap) < live
            phi, bits = force.potential_shard(
                state[:, 0, :],
                masses,
                live,
                ids,
                **({} if f_b is None else {"force_scale": f_b}),
            )
            # padding rows masked out of both sums (a field may be singular where
            # one sits), each sum in fp64, as the single card's
            zero = jnp.zeros((), jnp.float64)
            m = jnp.where(live_rows, jnp.asarray(masses).astype(jnp.float64), zero)
            phi64 = jnp.where(live_rows, jnp.asarray(phi).astype(jnp.float64), zero)
            w_self = 0.5 * jnp.sum(m * phi64)
            w_ext = zero
            if field_fn is not None:
                ext = (
                    field_fn(state) if params_in is None else field_fn(state, params_in)
                )
                ext64 = jnp.where(live_rows, jnp.asarray(ext).astype(jnp.float64), zero)
                w_ext = jnp.sum(m * ext64)
            ok = overflow_ok(bits) & (count[0] <= cap)
            ok, bits, _, _ = _mesh_verdict(
                ok,
                bits,
                jnp.zeros((1,), jnp.int32),
                jnp.int32(0),
                axis_name=axis,
            )
            return (
                jax.lax.psum(w_self, axis),
                jax.lax.psum(w_ext, axis),
                ok,
                bits,
            )

        spec = P(axis)
        if not self.scale_carry:

            def body(
                state: Array, masses: Array, count: Array, ids: Array, params_in: Any
            ) -> tuple:
                return energy(state, masses, count, ids, None, params_in)

            in_specs: tuple = (spec, spec, spec, spec, P())
        else:

            def body(  # type: ignore[misc]
                state: Array,
                masses: Array,
                count: Array,
                ids: Array,
                f_b: Array,
                params_in: Any,
            ) -> tuple:
                return energy(state, masses, count, ids, f_b, params_in)

            in_specs = (spec, spec, spec, spec, spec, P())
        program = jax.jit(
            jax.shard_map(
                body,
                mesh=self.mesh,
                in_specs=in_specs,
                out_specs=(P(), P(), P(), P()),
                check_vma=False,
            )
        )
        cache[key] = program
        return program, True
