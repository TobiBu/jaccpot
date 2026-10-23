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
  again as long as each re-run fits more steps than every earlier attempt (the
  re-plan reads only the steps up to the first failure, so needs that grow step
  by step are learned a step at a time).
* **Repartition BETWEEN segments**, as its own program, every
  ``repartition_every`` segments (host cadence: no ``lax.cond`` around
  collectives, whose device-divergent predicate deadlocks, and none of its buffers
  in a segment's peak). With the velocity-Verlet form the carried acceleration
  must travel with its particle, so ``sfc_repartition`` routes the velocity, the
  total acceleration and the id (``rollout._repartition_shard``).
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
        sharding = NamedSharding(mesh, P(axis_name))
        pack = jax.jit(lambda x, v: jnp.stack([x, v], axis=1), out_shardings=sharding)
        return cls(
            state=pack(shard.positions, shard.velocities),
            masses=shard.masses,
            ids=shard.ids,
            count=shard.count,
        )


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

    Raises
    ------
    ValueError
        If the state's rows do not split evenly over the mesh.
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
        )
        cache = _programs_of(self.force.cache_owner)
        program = cache.get(key)
        if program is not None:
            return program, False
        if kind == "segment":
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

    def _repartition_program(self) -> Callable:
        """The mesh repartition of the packed state, cached.

        ``rollout._repartition_shard`` (``sfc_repartition`` in the global box) with
        the velocity, the TOTAL acceleration and the id routed with each particle:
        the next segment's first drift and kick read the acceleration the last one
        ended with, row by row. Not donated: the ragged exchange's outputs must not
        alias its inputs.

        Returns
        -------
        Callable
            ``(state, acceleration, masses, ids, count) -> (state, acceleration,
            masses, ids, count, diag)``; under the criterion (:attr:`scale_carry`)
            ``f_b`` is routed too: ``(state, acceleration, f_b, masses, ids, count)
            -> (state, acceleration, f_b, masses, ids, count, diag)`` (the next
            force's thresholds are its particles', wherever they now live).
        """
        from jaccpot.distributed.rollout import _repartition_shard

        scale_carry = self.scale_carry
        key = (
            "mesh_repartition",
            tuple(int(d.id) for d in self.devices),
            self.axis_name,
            self.cap,
            self.num_samples,
            str(self.state.state.dtype),
            *(("force_scale",) if scale_carry else ()),
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
            )
            return (jnp.stack([x, v], axis=1), a, *routed, m, i, c, diag)

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
                out_specs=(spec,) * arrays + (diag_spec,),
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
        started = start
        acceleration = state.acceleration
        # the criterion's f_b: the start force's thresholds (carried, else the
        # prepare's seed) and, after it, the segment's first
        scale_now: tuple = self._scale_args(state)
        retry_enabled = env_flag("JACCPOT_STRICT_SEGMENT_RETRY", True)
        start_host: Any = None
        if k > 0 and self.donate and retry_enabled:
            # the segment consumes its start state, and a retry re-runs the segment
            # from it: one copy per segment, off the device (on a GPU asynchronous,
            # into pinned memory), as strict_run_v2's particle carry keeps it
            start_host = host_copy(state.state)
        state_new = state.state
        acc_new: Optional[Array] = None
        scale_new: tuple = ()
        retries, tree_first = 0, False
        failures: list = []
        # the most steps an earlier failed attempt fitted (-1: none failed yet)
        best_done = -1
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
            if k > 0 and (not start or bool(ok_h)):
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
            mode = (
                self.force.retry_mode(needs_np, retries, tree_first)
                if retry_enabled
                else None
            )
            if mode is None and retry_enabled and 0 <= best_done < int(done_h):
                # PROGRESS: this re-run fitted more steps than every earlier attempt.
                # The re-plan reads only the steps up to the first failure (the rest
                # left the trajectory), so a segment whose needs grow step by step is
                # learned one failing step at a time: the outgrowing-segment gate
                # under mac_type='dehnen_error' ran out of the force's two retries at
                # 1 of 2 steps fitted. `done` grows with every such retry, so they
                # are at most the segment's steps.
                tree_now = bool(needs_np[NEEDS_FIELDS.index("tree_overflow")])
                mode = "full" if tree_now else "walk"
            if mode is None:
                self.force.raise_overflow(
                    needs_np, int(bits_h), retries=retries, where=where
                )
            best_done = max(best_done, int(done_h))
            if retries == 0:
                tree_first = bool(needs_np[NEEDS_FIELDS.index("tree_overflow")])
            retries += 1
            acceleration = acc_new = None
            scale_new = ()
            state_new = state.state
            if consumed:
                assert start_host is not None
                state = MeshState(
                    state=jax.device_put(start_host, state.masses.sharding),
                    masses=state.masses,
                    ids=state.ids,
                    count=state.count,
                )
                state_new = state.state
            if not self.force.replan(needs_np, int(bits_h), walk_only=mode == "walk"):
                self.force.raise_overflow(
                    needs_np, int(bits_h), retries=retries - 1, where=where
                )
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
            masses=state.masses,
            ids=state.ids,
            count=state.count,
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
            counts=[int(c) for c in np.asarray(state.count)],
            bits=int(bits_h),
            needs=np.asarray(needs_h).tolist(),
            failures=failures,
        )
        if k > 0:
            self.reports.append(report)
        return report

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
