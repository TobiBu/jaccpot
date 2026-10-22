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

**Padding rows** keep their place: their velocity and acceleration are zero, so the
drift leaves them where the last repartition put them (on a live row), and the
force cuts them with ``num_valid``.

**Array convention, and it is a trap** (``fused.py``'s module docstring):
``shard_map`` keeps the mapped axis, so every particle array is FLAT, ``(ndev *
cap, ...)``, and a device sees ``(cap, ...)``.

The force is injectable (:meth:`MeshSegmentRunner.__init__`'s ``force``): the
fused lane's (:class:`~jaccpot.distributed.template_force.FusedMeshForce`;
:meth:`MeshSegmentRunner.fused` builds it), or a partition-independent direct sum
for tests.
"""

from __future__ import annotations

import weakref
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
    "MeshSegmentRunner",
    "MeshState",
    "SegmentReport",
]

#: compiled programs per force owner (the engine, for the fused force): runners on
#: one solver share them; every static input is in the key
_PROGRAMS: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()
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
    """

    state: Array
    masses: Array
    ids: Array
    count: Array
    acceleration: Optional[Array] = None

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
    sent_off_device : int
        Live rows that repartition moved to another device.
    counts : list
        Live rows per device during the segment.
    bits : int
        The kept run's overflow bits, OR-ed over its steps and the mesh.
    needs : list
        The kept run's needs, max-ed over its steps and the mesh
        (``fused_topology.NEEDS_FIELDS``, then the cross hook's).
    """

    segment: int
    steps: int
    start_force: bool
    retries: int = 0
    repartitioned: bool = False
    sent_off_device: int = 0
    counts: list = field(default_factory=list)
    bits: int = 0
    needs: list = field(default_factory=list)


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
        ``key``, ``context``, ``before_trace``, ``needs_size`` and ``cache_owner``
        (:class:`~jaccpot.distributed.template_force.FusedMeshForce`).
        ``prepare`` runs here, at ``state``.
    external_acceleration_fn : Optional[Callable[..., Array]]
        ``field(state_rows, params)`` (or ``field(state_rows)`` when ``run`` gets no
        parameters) -> ``(cap, 3)``, on one device's ``(cap, 2, 3)`` rows. Keep ONE
        function object: its identity is part of the compile keys.
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
        self.external_acceleration_fn = external_acceleration_fn
        self.donate = bool(donate)
        self.axis_name = axis_name
        self.state = state
        self.segment_index = 0
        self.steps_done = 0
        self.reports: list[SegmentReport] = []
        #: what the carried acceleration was computed with (external_field_key)
        self._acceleration_key: Optional[tuple] = None
        self.force.prepare(state.state[:, 0], state.masses, state.count)

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
        cross_caps: Any = None,
        near_theta: Optional[float] = None,
        external_acceleration_fn: Optional[Callable[..., Array]] = None,
        donate: bool = True,
        axis_name: str = AXIS_NAME,
    ) -> "MeshSegmentRunner":
        """A runner on the fused lane's force (the template route).

        Parameters
        ----------
        solver : Any
            The ``FastMultipoleMethod`` (large-N production profile, cell leaves,
            the flat walk, a geometric MAC).
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
            :class:`~jaccpot.distributed.cross.CrossCapacities`.
        near_theta : Optional[float]
            Passed to the cross hook.
        external_acceleration_fn : Optional[Callable[..., Array]]
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
            If the solver asks for Dehnen's error criterion (refused on this lane
            until it carries the force scale); raised before anything is prepared.
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
            axis_name=axis_name,
        )
        return cls(
            mesh,
            state,
            force,
            external_acceleration_fn=external_acceleration_fn,
            donate=donate,
            axis_name=axis_name,
        )

    # ------------------------------------------------------------------
    # The programs.
    # ------------------------------------------------------------------

    def _total_acceleration(
        self, acc_self: Array, state_rows: Array, params: Any, live_rows: Array
    ) -> Array:
        """Self-gravity plus the external field, zero on padding (one device).

        The single card's order (``strict_run_v2``'s ``_advance``): the self-gravity
        in the state's dtype, plus the field in the state's dtype.

        Parameters
        ----------
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
        field_fn = self.external_acceleration_fn
        if field_fn is not None:
            value = (
                field_fn(state_rows) if params is None else field_fn(state_rows, params)
            )
            acc = acc + jnp.asarray(value, state_rows.dtype)
        return jnp.where(live_rows[:, None], acc, jnp.zeros((), acc.dtype))

    def _program(self, kind: str, steps: int, params: Any) -> tuple[Callable, bool]:
        """The compiled start (``kind="start"``) or segment program, cached.

        Parameters
        ----------
        kind : str
            ``"start"`` or ``"segment"``.
        steps : int
            The segment's steps (``0`` for the start program).
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
        cache = _PROGRAMS.setdefault(self.force.cache_owner, {})
        program = cache.get(key)
        if program is not None:
            return program, False
        build = self._build_start if kind == "start" else self._build_segment
        program = build(int(steps))
        cache[key] = program
        return program, True

    def _build_start(self, steps: int) -> Callable:
        """The start force: the total acceleration at the current positions.

        Parameters
        ----------
        steps : int
            Unused (the key's slot).

        Returns
        -------
        Callable
            ``(state, masses, count, ids, params) -> (acceleration, ok, bits,
            needs)``; the acceleration flat, the rest replicated.
        """
        from jaccpot.runtime.fused_topology import overflow_ok

        del steps
        axis = self.axis_name

        def body(
            state: Array, masses: Array, count: Array, ids: Array, params: Any
        ) -> tuple:
            cap = int(state.shape[0])
            live = jnp.minimum(count[0], cap)
            live_rows = jnp.arange(cap) < live
            acc_self, bits, needs = self.force.shard(state[:, 0, :], masses, live, ids)
            acc = self._total_acceleration(acc_self, state, params, live_rows)
            # a count above the shard's capacity cannot be real particles
            ok = overflow_ok(bits) & (count[0] <= cap)
            ok, bits, needs, _ = _mesh_verdict(
                ok, bits, needs, jnp.int32(0), axis_name=axis
            )
            return acc, ok, bits, needs

        spec = P(axis)
        return jax.jit(
            jax.shard_map(
                body,
                mesh=self.mesh,
                in_specs=(spec, spec, spec, spec, P()),
                out_specs=(spec, P(), P(), P()),
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
            acceleration, ok, bits, needs, done)``.
        """
        from jaccpot.runtime.fmm_state import _velocity_verlet_kick_drifted
        from jaccpot.runtime.fused_topology import overflow_ok

        axis = self.axis_name
        needs_size = int(self.force.needs_size)

        def body(
            state0: Array,
            acc0: Array,
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
                state_now, acc_now, ok, bits, needs, done = carry
                # strict_run_v2's `_advance`: drift, force, kick of the drifted state
                position_new = (
                    state_now[:, 0] + state_now[:, 1] * dt + 0.5 * acc_now * dt**2
                )
                state_position = state_now.at[:, 0].set(position_new)
                acc_self, bits_new, needs_new = self.force.shard(
                    state_position[:, 0, :], masses, live, ids
                )
                acc_new = self._total_acceleration(
                    acc_self, state_position, params, live_rows
                )
                state_new = _velocity_verlet_kick_drifted(
                    state_position, acc_now, acc_new, dt
                )
                ok_new = ok & overflow_ok(bits_new)
                return (
                    state_new,
                    acc_new,
                    ok_new,
                    bits | bits_new,
                    jnp.maximum(needs, needs_new),
                    done + ok_new.astype(jnp.int32),
                ), None

            init = (
                state0,
                acc0,
                count[0] <= cap,
                jnp.uint32(0),
                jnp.zeros((needs_size,), jnp.int32),
                jnp.int32(0),
            )
            (state_f, acc_f, ok, bits, needs, done), _ = jax.lax.scan(
                step, init, None, length=int(steps)
            )
            return (state_f, acc_f) + _mesh_verdict(
                ok, bits, needs, done, axis_name=axis
            )

        spec = P(axis)
        return jax.jit(
            jax.shard_map(
                body,
                mesh=self.mesh,
                in_specs=(spec, spec, spec, spec, spec, P(), P()),
                out_specs=(spec, spec, P(), P(), P(), P()),
                check_vma=False,
            ),
            donate_argnums=(0, 1) if self.donate else (),
        )

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
            If a capacity flag fired.
        """
        from jaccpot.runtime.strict_carry import external_field_key

        k = int(steps)
        if k < 1:
            raise ValueError(f"a segment needs at least one step, got {steps}")
        n = self.segment_index + 1
        state = self.state
        dt_arr = jnp.asarray(dt, state.state.dtype)
        key = external_field_key(self.external_acceleration_fn, external_params)
        start = state.acceleration is None or key != self._acceleration_key
        acceleration = state.acceleration
        if start:
            acceleration, ok, bits, needs = self._call(
                "start",
                0,
                external_params,
                state.state,
                state.masses,
                state.count,
                state.ids,
            )
            ok_h, bits_h, needs_h = jax.device_get((ok, bits, needs))
            if not bool(ok_h):
                raise RuntimeError(
                    f"segment {n}: a capacity flag fired on the start force "
                    f"(bits {int(bits_h):#x}, needs {np.asarray(needs_h).tolist()})"
                )
        outputs = self._call(
            "segment",
            k,
            external_params,
            state.state,
            acceleration,
            state.masses,
            state.count,
            state.ids,
            dt_arr,
        )
        state_new, acc_new = outputs[:2]
        ok_h, bits_h, needs_h, done_h = jax.device_get(outputs[2:])
        if not bool(ok_h):
            raise RuntimeError(
                f"segment {n}: a capacity flag fired after {int(done_h)} of {k} steps "
                f"(bits {int(bits_h):#x}, needs {np.asarray(needs_h).tolist()})"
            )
        self.state = MeshState(
            state=state_new,
            masses=state.masses,
            ids=state.ids,
            count=state.count,
            acceleration=acc_new,
        )
        self._acceleration_key = key
        self.segment_index = n
        self.steps_done += k
        report = SegmentReport(
            segment=n,
            steps=k,
            start_force=start,
            counts=[int(c) for c in np.asarray(state.count)],
            bits=int(bits_h),
            needs=np.asarray(needs_h).tolist(),
        )
        self.reports.append(report)
        return report

    def gather(self) -> dict:
        """The state on the host, by global id, with identity checked.

        Returns
        -------
        dict
            ``positions``, ``velocities``, ``masses`` and (once computed)
            ``acceleration`` as ``(N, ...)`` arrays indexed by global id, plus
            ``owner`` (device per id).

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
        out = {}
        for name, arr in arrays.items():
            full = np.empty((n,) + arr.shape[1:], arr.dtype)
            full[live_ids] = arr
            out[name] = full
        owner = np.empty(n, np.int32)
        owner[live_ids] = rows // cap
        out["owner"] = owner
        return out
