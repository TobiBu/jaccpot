"""A kick-drift-kick rollout of the multi-GPU fused lane, repartitioned on device.

The particle set lives on the mesh as flat, capacity-padded shards (``ShardState``):
row ``r`` of device ``d`` is row ``d * cap + r`` of every array, with ``count[d]``
leading live rows. Ownership is refreshed every ``repartition_every`` steps by
:func:`yggdrax.distributed.partition.sfc_repartition`, which routes each particle's
half-step velocity and global id WITH it and declines (moves nothing) if any device
would overflow. Data never leaves the devices; only the step counter and the
replicated flags are read on the host.

Why three jits per step, not one:

* the drift (and repartition) and the kick are cheap element-wise maps, the force is
  the fused lane's own jitted ``shard_map``; keeping them apart keeps peak memory at
  the larger of the two rather than their sum -- fusing force and integrator at 21M
  particles on 5 cards deadlocked (ODISSEO's mesh lane, same lesson);
* the repartition cadence is decided on the HOST, so it is two compiled variants of
  the drift and no ``lax.cond`` around collectives (a device-divergent predicate
  deadlocks; a conditional's branches must also agree on manual-axis variance).

The repartition sits between the drift and the force: the old acceleration has
already been used by the first half-kick, so only positions, masses, the half-step
velocity and ids move, and the new force comes back in the new row order.

The force is injectable: :func:`setup_fused_force` builds the fused lane's,
:func:`make_reference_direct_force` an fp-exact direct sum whose result does not
depend on the partition (sources summed in global-id order), for tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from jaccpot.distributed.fused import AXIS_NAME, global_mesh_bounds

__all__ = [
    "FusedRollout",
    "RolloutConfig",
    "RolloutFlagError",
    "ShardState",
    "StepReport",
    "decompose",
    "make_reference_direct_force",
    "setup_fused_force",
]

#: force_fn(positions, masses, count, ids) -> (acceleration, overflow_flag)
ForceFn = Callable[[Array, Array, Array, Array], tuple]


@dataclass(frozen=True)
class RolloutConfig:
    """Static parameters of a rollout.

    Attributes
    ----------
    cap : int
        Rows per device (the shard capacity every shape depends on).
    dt : float
        Time step.
    repartition_every : int
        Repartition before the force of every step ``n`` with ``n % k == 0``;
        ``0`` never repartitions after the initial decomposition.
    num_samples : int
        Pivot samples per device (balance overshoot ~ ``2 * ndev / num_samples``).
    axis_name : str
        Mesh axis.
    """

    cap: int
    dt: float
    repartition_every: int = 16
    num_samples: int = 256
    axis_name: str = AXIS_NAME


@jax.tree_util.register_dataclass
@dataclass
class ShardState:
    """The mesh-resident particle state; every array flat and sharded ``P(axis)``.

    Attributes
    ----------
    positions : Array
        ``(ndev * cap, 3)``; padding rows sit on a live particle of their device.
    velocities : Array
        ``(ndev * cap, 3)`` synchronised velocities; zero on padding.
    masses : Array
        ``(ndev * cap,)``; zero on padding.
    ids : Array
        ``(ndev * cap,)`` int32 global ids; ``-1`` on padding.
    count : Array
        ``(ndev,)`` int32 live rows per device.
    accel : Array
        ``(ndev * cap, 3)`` acceleration at ``positions``, in the same row order.
    """

    positions: Array
    velocities: Array
    masses: Array
    ids: Array
    count: Array
    accel: Array


@dataclass
class StepReport:
    """What one step did, read back from the replicated diagnostics.

    Attributes
    ----------
    step : int
        Step index just completed (1-based).
    repartitioned : bool
        Whether this step ran a repartition.
    declined : bool
        The repartition was declined (some device would have overflowed).
    sent_off_device : int
        Live rows that changed device in this step's repartition.
    counts : list
        Live rows per device after the step.
    overflow : bool
        The force's capacity flag (any device). A rollout raises before reporting
        ``True``; it is recorded for completeness.
    """

    step: int
    repartitioned: bool
    declined: bool
    sent_off_device: int
    counts: list = field(default_factory=list)
    overflow: bool = False


class RolloutFlagError(RuntimeError):
    """A capacity flag fired; the step it names produced no trustworthy force."""


def _mesh_devices(mesh: Any) -> list:
    return list(np.asarray(mesh.devices).reshape(-1))


def _pad_onto_live_row(positions: Array, live: Array) -> Array:
    """Move padding rows onto the device's first row (live whenever count >= 1).

    ``sfc_repartition`` pads at the origin. The traced fused lane cuts dead rows with
    ``num_valid`` so the origin is harmless there, but an EAGER prepare treats padding
    as particles, and a pile of zero-mass rows at the centre of a cluster distorts the
    measured level plan and walk capacities.
    """
    return jnp.where(live[:, None], positions, positions[0][None, :])


def decompose(
    mesh: Any,
    positions: Any,
    velocities: Any,
    masses: Any,
    *,
    cap: int,
    ids: Optional[Any] = None,
    num_samples: int = 256,
    axis_name: str = AXIS_NAME,
) -> tuple[ShardState, dict]:
    """Place a host particle set on the mesh as Morton domains, on device.

    The host only splits the arrays evenly in their given order (no sort); one jitted
    :func:`~yggdrax.distributed.partition.sfc_repartition` then makes each device own
    a contiguous Morton range. Accelerations are zero until the first force.

    Parameters
    ----------
    mesh : Any
        A 1-D ``jax.sharding.Mesh``.
    positions : Any
        ``(N, 3)`` host positions.
    velocities : Any
        ``(N, 3)`` host velocities.
    masses : Any
        ``(N,)`` host masses.
    cap : int
        Rows per device. Must cover ``ceil(N / ndev)`` with headroom for the
        sample-sort imbalance.
    ids : Optional[Any]
        ``(N,)`` int32 global ids; ``None`` uses ``arange(N)``.
    num_samples : int
        Pivot samples per device.
    axis_name : str
        Mesh axis.

    Returns
    -------
    tuple[ShardState, dict]
        The state and the repartition diagnostics (``recv_counts``,
        ``sent_off_device``, ``max_util``).

    Raises
    ------
    ValueError
        If an even split does not fit ``cap``, or the decomposition was declined
        (there is no earlier ownership to fall back on).
    """
    devices = _mesh_devices(mesh)
    ndev = len(devices)
    pos = np.asarray(positions)
    vel = np.asarray(velocities, pos.dtype)
    mass = np.asarray(masses, pos.dtype)
    n = int(pos.shape[0])
    gid = np.arange(n, dtype=np.int32) if ids is None else np.asarray(ids, np.int32)
    if int(np.ceil(n / ndev)) > int(cap):
        raise ValueError(f"cap={cap} cannot hold an even split of N={n} over {ndev}")
    flat_pos = np.zeros((ndev * cap, 3), pos.dtype)
    flat_vel = np.zeros((ndev * cap, 3), pos.dtype)
    flat_mass = np.zeros((ndev * cap,), pos.dtype)
    flat_id = np.full((ndev * cap,), -1, np.int32)
    counts = np.zeros((ndev,), np.int32)
    for d, chunk in enumerate(np.array_split(np.arange(n), ndev)):
        k = len(chunk)
        lo = d * cap
        flat_pos[lo : lo + k] = pos[chunk]
        flat_vel[lo : lo + k] = vel[chunk]
        flat_mass[lo : lo + k] = mass[chunk]
        flat_id[lo : lo + k] = gid[chunk]
        if k:
            flat_pos[lo + k : lo + cap] = pos[chunk[0]]
        counts[d] = k
    sh = NamedSharding(mesh, P(axis_name))
    x, v, m, i, c = (
        jax.device_put(a, sh) for a in (flat_pos, flat_vel, flat_mass, flat_id, counts)
    )
    repart = _make_repartition(
        mesh, cap=cap, num_samples=num_samples, axis_name=axis_name
    )
    x, v, m, i, c, diag = repart(x, v, m, i, c)
    if bool(np.asarray(diag["declined"])[0]):
        raise ValueError(
            "the initial decomposition was declined: some device would receive "
            f"{int(np.asarray(diag['recv_counts']).max())} rows > cap={cap}"
        )
    state = ShardState(
        positions=x,
        velocities=v,
        masses=m,
        ids=i,
        count=c,
        accel=jax.device_put(np.zeros((ndev * cap, 3), pos.dtype), sh),
    )
    return state, {k: np.asarray(vv) for k, vv in diag.items()}


def _make_repartition(
    mesh: Any,
    *,
    cap: int,
    num_samples: int,
    axis_name: str,
    route_velocities: bool = True,
) -> Callable:
    """A jitted mesh repartition of ``(x, v, m, ids, count)``.

    ``route_velocities=False`` is a deliberate MUTATION for tests: velocities stay in
    the old row order while everything else moves, which must break the rollout's
    equivalence -- proof that the equivalence test can see a mis-route.
    """
    from yggdrax.distributed.partition import sfc_repartition

    ndev = len(_mesh_devices(mesh))

    def body(x: Array, v: Array, m: Array, ids: Array, count: Array) -> tuple:
        n_live = count[0]
        bounds = global_mesh_bounds(x, num_valid=n_live, axis_name=axis_name)
        payload = {"v": v, "id": ids} if route_velocities else {"id": ids}
        fill = {"v": 0.0, "id": -1} if route_velocities else {"id": -1}
        res = sfc_repartition(
            x,
            m,
            n_live,
            ndev,
            output_capacity=int(cap),
            bounds=bounds,
            num_samples=int(num_samples),
            axis_name=axis_name,
            payload=payload,
            payload_fill=fill,
        )
        live = jnp.arange(int(cap)) < res.live_count
        x_new = _pad_onto_live_row(res.positions, live)
        v_new = res.payload["v"] if route_velocities else v
        v_new = jnp.where(live[:, None], v_new, 0.0)
        diag = {
            "declined": res.declined.reshape(1),
            "recv_counts": res.recv_counts[None],
            "sent_off_device": res.sent_off_device.reshape(1),
            "max_util": res.max_util.reshape(1),
        }
        return (
            x_new,
            v_new,
            res.masses,
            res.payload["id"],
            res.live_count.reshape(1).astype(jnp.int32),
            diag,
        )

    spec = P(axis_name)
    mapped = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(spec,) * 5,
        out_specs=(spec,) * 5
        + (
            {
                "declined": spec,
                "recv_counts": spec,
                "sent_off_device": spec,
                "max_util": spec,
            },
        ),
        check_vma=False,
    )
    return jax.jit(mapped)


def make_reference_direct_force(
    mesh: Any,
    *,
    G: float = 1.0,
    softening: float = 0.0,
    axis_name: str = AXIS_NAME,
) -> ForceFn:
    """An all-pairs direct-sum force whose answer does not depend on the partition.

    Every device all-gathers the mesh's particles and sums the sources in GLOBAL-ID
    order, so the acceleration of a particle is bitwise the same whichever device and
    row hold it. That is what lets a test require a repartitioned rollout and an
    un-repartitioned one to agree bitwise. Small N only (``O(N^2)`` per device).

    Parameters
    ----------
    mesh : Any
        The 1-D mesh.
    G : float
        Gravitational constant.
    softening : float
        Plummer softening length.
    axis_name : str
        Mesh axis.

    Returns
    -------
    ForceFn
        ``force(positions, masses, count, ids) -> (acceleration, overflow=False)``.
    """

    def body(x: Array, m: Array, count: Array, ids: Array) -> tuple:
        live = jnp.arange(x.shape[0]) < count[0]
        xs = jax.lax.all_gather(x, axis_name, tiled=True)
        ms = jax.lax.all_gather(jnp.where(live, m, 0.0), axis_name, tiled=True)
        gs = jax.lax.all_gather(jnp.where(live, ids, -1), axis_name, tiled=True)
        big = jnp.iinfo(jnp.int32).max
        order = jnp.argsort(jnp.where(gs >= 0, gs, big), stable=True)
        xs, ms = xs[order], ms[order]
        d = xs[None, :, :] - x[:, None, :]
        r2 = jnp.sum(d * d, axis=-1) + jnp.asarray(softening, x.dtype) ** 2
        inv = jnp.where(r2 > 0, r2 ** jnp.asarray(-1.5, x.dtype), 0.0)
        acc = jnp.asarray(G, x.dtype) * jnp.einsum("ij,ijk->ik", ms[None, :] * inv, d)
        return jnp.where(live[:, None], acc, 0.0), jnp.zeros((1,), jnp.bool_)

    spec = P(axis_name)
    mapped = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * 4,
            out_specs=(spec, spec),
            check_vma=False,
        )
    )

    def force(positions: Array, masses: Array, count: Array, ids: Array) -> tuple:
        acc, flag = mapped(positions, masses, count, ids)
        return acc, jnp.any(flag)

    return force


def setup_fused_force(
    solver: Any,
    mesh: Any,
    state: ShardState,
    *,
    leaf_size: int,
    max_order: int,
    theta: float,
    cross_caps: Any = None,
    near_theta: Optional[float] = None,
    axis_name: str = AXIS_NAME,
) -> tuple[ForceFn, Any, Optional[dict]]:
    """Build the fused lane's force for this mesh state, once.

    Per device: the shard is read from that device and prepared eagerly ON it
    (:func:`~jaccpot.runtime.capacity_plan.measure_shard_plan`); the level plans and
    walk-capacity records are merged over devices and installed, the prepared states
    assembled sharded, and the evaluator built with the cross field (``ndev > 1``).
    The prepared state is a capacity-fixed TEMPLATE -- the tree, walk and lists are
    rebuilt from the positions on every call -- so a repartitioned state with the same
    ``cap`` needs no re-prepare.

    The process-wide fast-lane environment (``JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET``
    must admit ``cap``) is the caller's: it has to be set before jaccpot is imported.

    Parameters
    ----------
    solver : Any
        The configured ``FastMultipoleMethod`` (large-N production profile).
    mesh : Any
        The 1-D mesh.
    state : ShardState
        The decomposed state; only its shapes and current shards are read.
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
    tuple[ForceFn, Any, Optional[dict]]
        ``(force, merged plan, merged walk caps)``.
    """
    from jaccpot.distributed.cross import make_cross_hook
    from jaccpot.distributed.fused import (
        assemble_prepared_states,
        make_fused_force_evaluator,
    )
    from jaccpot.runtime.capacity_plan import (
        install_walk_caps,
        measure_shard_plan,
        merge_plans,
        merge_walk_caps,
    )

    devices = _mesh_devices(mesh)
    ndev = len(devices)

    def _by_device(arr):
        shards = {s.device: s.data for s in arr.addressable_shards}
        return [shards[d] for d in devices]

    pos_d, mass_d = _by_device(state.positions), _by_device(state.masses)
    preps, plans, walks = [], [], []
    for d, device in enumerate(devices):
        with jax.default_device(device):
            prepared, plan_d, caps_d = measure_shard_plan(
                solver,
                pos_d[d],
                mass_d[d],
                leaf_size=leaf_size,
                max_order=max_order,
                theta=theta,
            )
        preps.append(prepared)
        plans.append(plan_d)
        walks.append(caps_d)
    plan = merge_plans(plans)
    walk_caps = merge_walk_caps(walks)
    install_walk_caps(solver, walk_caps)
    stacked = assemble_prepared_states(preps, mesh, axis_name=axis_name)
    del preps

    hook, sink = None, None
    if ndev > 1:
        sink = {}
        hook = make_cross_hook(
            ndev=ndev,
            theta=theta,
            caps=cross_caps,
            near_sink=sink,
            near_theta=near_theta,
            axis_name=axis_name,
        )
    evaluator = make_fused_force_evaluator(
        solver,
        stacked,
        mesh=mesh,
        plan=plan,
        leaf_size=leaf_size,
        max_order=max_order,
        theta=theta,
        axis_name=axis_name,
        cross_hook=hook,
        cross_near_sink=sink,
    )

    def force(
        positions: Array, masses: Array, count: Array, ids: Optional[Array]
    ) -> tuple:
        del ids  # the fused lane's rows come back in their input order
        acc, flag = evaluator(positions, masses, count)
        return acc, flag

    return force, plan, walk_caps


class FusedRollout:
    """Kick-drift-kick on the mesh, repartitioning every ``repartition_every`` steps.

    Parameters
    ----------
    mesh : Any
        The 1-D mesh.
    state : ShardState
        The decomposed initial state (accelerations may be zero; :meth:`start`
        computes them).
    config : RolloutConfig
        Static parameters.
    force_fn : ForceFn
        ``force(positions, masses, count, ids) -> (acceleration, overflow)``.
    route_velocities : bool
        ``False`` is a test-only MUTATION (see :func:`_make_repartition`).
    """

    def __init__(
        self,
        mesh: Any,
        state: ShardState,
        config: RolloutConfig,
        force_fn: ForceFn,
        *,
        route_velocities: bool = True,
    ) -> None:
        self.mesh = mesh
        self.state = state
        self.config = config
        self.force_fn = force_fn
        self.step_index = 0
        self.reports: list[StepReport] = []
        self.ndev = len(_mesh_devices(mesh))
        cap, dt, ax = int(config.cap), float(config.dt), config.axis_name
        spec = P(ax)
        self._repartition = _make_repartition(
            mesh,
            cap=cap,
            num_samples=int(config.num_samples),
            axis_name=ax,
            route_velocities=route_velocities,
        )

        def _kick_drift(x, v, a, count):
            live = jnp.arange(cap) < count[0]
            vh = jnp.where(live[:, None], v + 0.5 * dt * a, 0.0)
            x_new = jnp.where(live[:, None], x + dt * vh, x)
            return _pad_onto_live_row(x_new, live), vh

        def _kick(vh, a, count):
            live = jnp.arange(cap) < count[0]
            return jnp.where(live[:, None], vh + 0.5 * dt * a, 0.0)

        self._kick_drift = jax.jit(
            jax.shard_map(
                _kick_drift,
                mesh=mesh,
                in_specs=(spec,) * 4,
                out_specs=(spec, spec),
                check_vma=False,
            )
        )
        self._kick = jax.jit(
            jax.shard_map(
                _kick, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec, check_vma=False
            )
        )

    def start(self) -> None:
        """Compute the acceleration at the initial positions.

        Raises
        ------
        RolloutFlagError
            If the force's capacity flag fires.
        """
        s = self.state
        acc, flag = self.force_fn(s.positions, s.masses, s.count, s.ids)
        if bool(np.asarray(flag)):
            raise RolloutFlagError("capacity flag on the initial force")
        self.state = ShardState(
            s.positions, s.velocities, s.masses, s.ids, s.count, acc
        )

    def step(self) -> StepReport:
        """Advance one step: kick, drift, (repartition), force, kick.

        Returns
        -------
        StepReport
            What the step did.

        Raises
        ------
        RolloutFlagError
            If the force's capacity flag fires, or a repartition was declined.
        """
        s = self.state
        n = self.step_index + 1
        k = int(self.config.repartition_every)
        part = k > 0 and n % k == 0
        x, vh = self._kick_drift(s.positions, s.velocities, s.accel, s.count)
        m, ids, count = s.masses, s.ids, s.count
        declined, sent = False, 0
        if part:
            x, vh, m, ids, count, diag = self._repartition(x, vh, m, ids, count)
            declined = bool(np.asarray(diag["declined"])[0])
            sent = int(np.asarray(diag["sent_off_device"]).sum())
        acc, flag = self.force_fn(x, m, count, ids)
        overflow = bool(np.asarray(flag))
        counts = [int(c) for c in np.asarray(count)]
        report = StepReport(n, part, declined, sent, counts, overflow)
        if overflow or declined:
            self.reports.append(report)
            raise RolloutFlagError(
                f"step {n}: "
                + ("capacity flag on the force" if overflow else "")
                + (" repartition declined" if declined else "")
                + f" (counts {counts}, cap {self.config.cap})"
            )
        v = self._kick(vh, acc, count)
        self.state = ShardState(x, v, m, ids, count, acc)
        self.step_index = n
        self.reports.append(report)
        return report

    def run(self, steps: int) -> list[StepReport]:
        """Advance ``steps`` steps.

        Parameters
        ----------
        steps : int
            Number of steps.

        Returns
        -------
        list[StepReport]
            One report per step.
        """
        return [self.step() for _ in range(int(steps))]

    def gather(self) -> dict:
        """The state on the host, by global id, with identity checked.

        Returns
        -------
        dict
            ``positions``, ``velocities``, ``masses``, ``accel`` as ``(N, ...)``
            arrays indexed by global id, plus ``owner`` (device per id).

        Raises
        ------
        ValueError
            If an id is missing or appears twice.
        """
        s = self.state
        cap = int(self.config.cap)
        count = np.asarray(s.count)
        ids = np.asarray(s.ids)
        rows = np.concatenate(
            [np.arange(d * cap, d * cap + int(count[d])) for d in range(self.ndev)]
        )
        live_ids = ids[rows]
        n = int(count.sum())
        if live_ids.min() < 0 or len(np.unique(live_ids)) != n or live_ids.max() >= n:
            raise ValueError(
                "particle identity broken: ids are not exactly 0..N-1 once"
            )
        out = {}
        for name in ("positions", "velocities", "masses", "accel"):
            arr = np.asarray(getattr(s, name))[rows]
            full = np.empty((n,) + arr.shape[1:], arr.dtype)
            full[live_ids] = arr
            out[name] = full
        owner = np.empty(n, np.int32)
        owner[live_ids] = rows // cap
        out["owner"] = owner
        return out
