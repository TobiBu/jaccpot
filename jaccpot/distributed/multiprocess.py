"""Multi-process runtime for the multi-GPU mode: one process per device (M2).

``ProductionRunner(mesh=MeshRunConfig(...))`` and :mod:`jaccpot.distributed.segment`
were built for ONE process that drives every device of the mesh. This module lets
the same programs run as N processes with one device each (``jax.distributed``):
on a multi-node cluster one process per GPU, on a CPU host N localhost processes
(the CI gate). Every entry point here is a no-op twin of the single-process code
when ``jax.process_count() == 1``: the single-process programs and host steps are
unchanged, bitwise.

**Initialisation.** :func:`initialize` before anything touches a backend: from an
explicit coordinator (``"host:port"``, the process count and this process's index)
or, with no arguments, from the environment ``jax.distributed`` detects (SLURM:
one task per GPU). On CPU it selects the gloo collectives (XLA:CPU has no other
cross-process implementation here: MPI needs an MPItrampoline build). Then
:func:`global_mesh` is the 1-D mesh over every process's devices, in process
order, which the runner takes as ``MeshRunConfig(mesh=...)``.

**What changes in multi-process mode** (the inventory is
``mn-scratch/m2/M2-DESIGN.md``):

* **Host reads of sharded arrays.** ``np.asarray`` of a ``P(axis)`` array that
  spans other processes' devices raises. :func:`host_value` reads it: one copy in
  a single process (as before), else the fully replicated value's local copy, else
  ``process_allgather`` (every process gets the global array: small arrays --
  counts, diagnostics -- and the test-scale gathers only).
* **Float reductions across devices.** XLA:CPU's in-process all-reduce folds the
  devices' partial sums in mesh order, ``((a0 + a1) + a2) + ...``; gloo's
  cross-process all-reduce does not (4 processes: 12,656 of 49,152 float64 bytes
  differ). :func:`mesh_sum` inside ``shard_map`` and :func:`mesh_moments` outside
  it reproduce the mesh-order fold exactly from an ``all_gather`` of the partials,
  so N processes give the single process's bits. Integer, min and max reductions
  keep their collectives (exact in any order).
* **Per-device eager work** (the template force's caps-only prepares) runs only on
  a process's own devices; what the devices' results merge into (plans, walk caps,
  the box, leaf counts) is all-gathered as small host values
  (:func:`allgather_by_device`, :func:`allgather_object`) and merged in mesh order
  on every process.
* **Host copies for a retry** (:func:`host_copy`, :func:`put_host_copy`): each
  process copies its own shards only, and puts them back as its shards.
* **Input**: :class:`ProcessLocalParticles` -- each process passes only the
  particles of its devices' initial chunks (:func:`local_id_range`: the contiguous
  id ranges the single process's decomposition hands each device), so no host
  holds the whole initial condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot._env import env_text

__all__ = [
    "ProcessInfo",
    "ProcessLocalParticles",
    "allgather_by_device",
    "allgather_host",
    "allgather_object",
    "env_coordinator",
    "global_mesh",
    "host_copy",
    "host_value",
    "initialize",
    "is_multiprocess",
    "local_device_indices",
    "local_id_range",
    "mesh_moments",
    "mesh_sum",
    "process_info",
    "put_host_copy",
    "refuse_in_multiprocess",
    "slurm_bound_device_ids",
]


@dataclass(frozen=True)
class ProcessInfo:
    """Where this process sits in the run.

    Attributes
    ----------
    process_index : int
        ``jax.process_index()``.
    process_count : int
        ``jax.process_count()``.
    local_devices : tuple
        This process's devices (``str``).
    global_device_count : int
        Devices over all processes.
    backend : str
        ``jax.default_backend()``.
    """

    process_index: int
    process_count: int
    local_devices: tuple
    global_device_count: int
    backend: str


def initialize(
    coordinator_address: Optional[str] = None,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
    *,
    local_device_ids: Optional[Sequence[int]] = None,
    cpu_collectives: str = "gloo",
    initialization_timeout: int = 300,
) -> ProcessInfo:
    """Join (or start) the multi-process run: ``jax.distributed.initialize``.

    Call it once per process, before anything initialises a backend (the first
    ``jax.devices()``, any array). With all three of ``coordinator_address``,
    ``num_processes`` and ``process_id`` ``None``, ``jax.distributed`` reads them
    from the cluster environment (SLURM: ``SLURM_*`` with one task per GPU). On CPU
    the cross-process collectives are gloo's.

    **One GPU per process under SLURM.** jax's SLURM detection sets
    ``local_device_ids=[SLURM_LOCALID]``: right when every task of a node sees all
    of the node's cards (``--ntasks-per-node=G --gres=gpu:G``, no per-task GPU
    binding: task ``l`` takes card ``l``), WRONG when SLURM binds one card to each
    task (``--gpus-per-task=1`` with binding, ``--gpu-bind=single:1``): the task
    then sees one card, as device 0, and ``[SLURM_LOCALID]`` names a device that
    does not exist from local rank 1 on. So when SLURM is detected, nothing names
    the devices (no ``local_device_ids``, no ``JAX_LOCAL_DEVICE_IDS``) and
    ``CUDA_VISIBLE_DEVICES`` lists exactly one card, this passes
    ``local_device_ids=[0]`` (:func:`slurm_bound_device_ids`). Both launches then
    give each process its own card.

    Parameters
    ----------
    coordinator_address : Optional[str]
        ``"host:port"`` of process 0's coordinator service.
    num_processes : Optional[int]
        Processes in the run.
    process_id : Optional[int]
        This process's index, ``0..num_processes-1``.
    local_device_ids : Optional[Sequence[int]]
        The devices this process owns (``None``: jax's default, or ``[0]`` for a
        SLURM task bound to one card; see above).
    cpu_collectives : str
        ``jax_cpu_collectives_implementation`` (``"gloo"``; ``"mpi"`` needs an
        MPItrampoline build of MPI).
    initialization_timeout : int
        Seconds to wait for every process to join.

    Returns
    -------
    ProcessInfo
        This process's place.
    """
    jax.config.update("jax_cpu_collectives_implementation", cpu_collectives)
    kwargs: dict = {"initialization_timeout": int(initialization_timeout)}
    if local_device_ids is None and None in (
        coordinator_address,
        num_processes,
        process_id,
    ):
        local_device_ids = slurm_bound_device_ids()
    if local_device_ids is not None:
        kwargs["local_device_ids"] = list(local_device_ids)
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        **kwargs,
    )
    return process_info()


#: what jax's SLURM detection reads (``jax._src.clusters.slurm_cluster``)
_SLURM_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_STEP_NODELIST",
    "SLURM_NTASKS",
    "SLURM_PROCID",
    "SLURM_LOCALID",
)


def slurm_bound_device_ids() -> Optional[list]:
    """``[0]`` for a SLURM task whose ``CUDA_VISIBLE_DEVICES`` is one card, else ``None``.

    ``None`` (jax's own choice, ``[SLURM_LOCALID]``) outside SLURM, when
    ``JAX_LOCAL_DEVICE_IDS`` is set, and when ``CUDA_VISIBLE_DEVICES`` is unset or
    lists several cards (a task that sees the node's cards). See
    :func:`initialize`.

    Returns
    -------
    Optional[list]
        ``[0]`` or ``None``.
    """
    if not all(env_text(name) for name in _SLURM_VARIABLES):
        return None
    if env_text("JAX_LOCAL_DEVICE_IDS"):
        return None
    cards = [c for c in env_text("CUDA_VISIBLE_DEVICES").split(",") if c.strip()]
    return [0] if len(cards) == 1 else None


def process_info() -> ProcessInfo:
    """This process's place in the run (one process: index 0 of 1).

    Returns
    -------
    ProcessInfo
        Index, count, local devices, global device count, backend.
    """
    return ProcessInfo(
        process_index=int(jax.process_index()),
        process_count=int(jax.process_count()),
        local_devices=tuple(str(d) for d in jax.local_devices()),
        global_device_count=int(jax.device_count()),
        backend=str(jax.default_backend()),
    )


def is_multiprocess() -> bool:
    """Whether the run spans more than one process.

    Returns
    -------
    bool
        ``jax.process_count() > 1``.
    """
    return jax.process_count() > 1


def global_mesh(axis_name: Optional[str] = None, devices: Any = None) -> Any:
    """The 1-D mesh over every process's devices, process by process.

    Process ``p``'s devices are contiguous in it (what :func:`local_id_range`
    and the runner's decomposition assume), in ``(process_index, id)`` order. In
    one process it is ``Mesh(jax.devices(), (axis_name,))``.

    Parameters
    ----------
    axis_name : Optional[str]
        The mesh axis (``None``: yggdrax's ``AXIS_NAME``, the runner's default).
    devices : Any
        Devices to use (``None``: ``jax.devices()``).

    Returns
    -------
    Any
        The ``jax.sharding.Mesh``.
    """
    from jax.sharding import Mesh

    if axis_name is None:
        from yggdrax.distributed.comm import AXIS_NAME

        axis_name = AXIS_NAME
    pool = list(jax.devices() if devices is None else devices)
    pool.sort(key=lambda d: (int(d.process_index), int(d.id)))
    return Mesh(np.asarray(pool), (str(axis_name),))


def refuse_in_multiprocess(what: str, why: str) -> None:
    """Raise when a single-process-only feature is asked for under ``jax.distributed``.

    Parameters
    ----------
    what : str
        The feature.
    why : str
        Why, and what to use instead.

    Raises
    ------
    NotImplementedError
        In multi-process mode.
    """
    if is_multiprocess():
        raise NotImplementedError(
            f"{what} is not available with {jax.process_count()} processes "
            f"(jax.distributed): {why}"
        )


# ----------------------------------------------------------------------
# Host reads and host copies.
# ----------------------------------------------------------------------


def host_value(arr: Any) -> np.ndarray:
    """A (sharded) device array's global value on this host.

    One process: ``np.asarray(jax.device_get(arr))`` (unchanged). Several: a
    fully addressable or fully replicated array is read locally; anything else is
    ``process_allgather``-ed, so every process holds the global value (use it for
    small arrays -- counts, verdicts, diagnostics -- and test-scale gathers).

    Parameters
    ----------
    arr : Any
        A ``jax.Array``, or a host value (returned as an array).

    Returns
    -------
    np.ndarray
        The global value.
    """
    if not is_multiprocess() or not isinstance(arr, jax.Array):
        return np.asarray(jax.device_get(arr))
    if arr.is_fully_addressable:
        return np.asarray(jax.device_get(arr))
    if arr.is_fully_replicated:
        return np.asarray(arr.addressable_data(0))
    from jax.experimental import multihost_utils

    return np.asarray(multihost_utils.process_allgather(arr, tiled=True))


def allgather_host(value: Any) -> np.ndarray:
    """Every process's ``value`` stacked in process order (``[process_count, ...]``).

    One process: ``value[None]``. The values must have the same shape and dtype
    on every process.

    Parameters
    ----------
    value : Any
        A host array or scalar.

    Returns
    -------
    np.ndarray
        ``[process_count, *shape]``.
    """
    arr = np.asarray(value)
    if not is_multiprocess():
        return arr[None]
    from jax.experimental import multihost_utils

    return np.asarray(multihost_utils.process_allgather(arr, tiled=False))


def allgather_object(obj: Any) -> list:
    """Every process's ``obj``, in process order.

    For the small, ragged host records the per-device setup merges (walk caps,
    level plans, leaf counts): pickled (types kept exactly: tuples stay tuples,
    NumPy scalars their dtype), padded to the longest, all-gathered. Only for
    values this run's own processes produced.

    Parameters
    ----------
    obj : Any
        A picklable value.

    Returns
    -------
    list
        One value per process (``[obj]`` in one process).
    """
    if not is_multiprocess():
        return [obj]
    import pickle

    raw = np.frombuffer(pickle.dumps(obj, protocol=4), np.uint8)
    lengths = allgather_host(np.int64(raw.shape[0]))
    width = int(lengths.max())
    padded = np.zeros((width,), np.uint8)
    padded[: raw.shape[0]] = raw
    rows = allgather_host(padded)
    return [
        pickle.loads(bytes(rows[p, : int(lengths[p])]))  # noqa: S301 (own processes)
        for p in range(rows.shape[0])
    ]


def allgather_by_device(mesh: Any, local: dict) -> list:
    """Per-device host values from every process, in mesh order.

    Parameters
    ----------
    mesh : Any
        The 1-D mesh.
    local : dict
        ``mesh position -> value`` for this process's devices.

    Returns
    -------
    list
        One value per mesh device.

    Raises
    ------
    ValueError
        If a device has no value or two.
    """
    ndev = int(np.asarray(mesh.devices).size)
    merged: dict = {}
    for part in allgather_object(dict(local)):
        for d, value in part.items():
            if int(d) in merged:
                raise ValueError(f"device {d} reported by two processes")
            merged[int(d)] = value
    missing = [d for d in range(ndev) if d not in merged]
    if missing:
        raise ValueError(f"no value for mesh devices {missing}")
    return [merged[d] for d in range(ndev)]


@dataclass
class _LocalCopy:
    """:func:`host_copy` in multi-process mode: this process's shards on the host.

    Attributes
    ----------
    shape : tuple
        The global shape.
    sharding : Any
        The array's sharding.
    shards : list
        ``(device, np.ndarray)`` for every addressable shard.
    """

    shape: tuple
    sharding: Any
    shards: list


def host_copy(state: Any) -> Any:
    """A copy of a (sharded) device array off the device, for a donated input.

    One process: ``runtime.strict_carry.host_copy`` (pinned host on a GPU, NumPy
    elsewhere), unchanged. Several: each process copies its own shards (on a GPU
    into pinned host memory, the same asynchronous copy; elsewhere NumPy).

    Parameters
    ----------
    state : Any
        The array about to be donated.

    Returns
    -------
    Any
        What :func:`put_host_copy` puts back.
    """
    from jaccpot.runtime.strict_carry import host_copy as single

    if not is_multiprocess() or state.is_fully_addressable:
        return single(state)
    devices = list(state.sharding.addressable_devices)
    pinned = all(
        d.platform == "gpu"
        and any(m.kind == "pinned_host" for m in d.addressable_memories())
        for d in devices
    )
    if pinned:
        return jax.device_put(state, state.sharding.with_memory_kind("pinned_host"))
    return _LocalCopy(
        shape=tuple(state.shape),
        sharding=state.sharding,
        shards=[
            (s.device, np.asarray(jax.device_get(s.data)))
            for s in state.addressable_shards
        ],
    )


def put_host_copy(copy: Any, sharding: Any) -> Any:
    """:func:`host_copy`'s copy back on the devices, bitwise.

    Parameters
    ----------
    copy : Any
        From :func:`host_copy`.
    sharding : Any
        The sharding to place it with.

    Returns
    -------
    Any
        The device array.
    """
    if not isinstance(copy, _LocalCopy):
        return jax.device_put(copy, sharding)
    arrays = [jax.device_put(data, device) for device, data in copy.shards]
    return jax.make_array_from_single_device_arrays(copy.shape, sharding, arrays)


# ----------------------------------------------------------------------
# Reductions in mesh order.
# ----------------------------------------------------------------------


def _fold(parts: Any) -> Any:
    """``((p0 + p1) + p2) + ...`` over the leading axis (XLA:CPU's all-reduce order).

    Parameters
    ----------
    parts : Any
        ``[ndev, ...]``.

    Returns
    -------
    Any
        The fold.
    """
    total = parts[0]
    for i in range(1, int(parts.shape[0])):
        total = total + parts[i]
    return total


def mesh_sum(value: Any, axis_name: str) -> Any:
    """``psum`` inside ``shard_map``, in mesh order under several processes.

    One process: ``jax.lax.psum`` (the program unchanged). Several: an
    ``all_gather`` of the per-device values folded in mesh order, which is what
    XLA:CPU's in-process all-reduce computes, so N processes give the single
    process's bits (gloo reduces in another order from 4 processes on). For the
    small reductions of the diagnostics; the trace-time choice follows
    ``jax.process_count()``.

    Parameters
    ----------
    value : Any
        This device's value.
    axis_name : str
        The mesh axis.

    Returns
    -------
    Any
        The sum, replicated.
    """
    if not is_multiprocess():
        return jax.lax.psum(value, axis_name)
    return _fold(jax.lax.all_gather(value, axis_name, tiled=False))


#: :func:`mesh_moments`' programs by (mesh devices, axis name)
_MOMENTS_PROGRAMS: dict = {}


def mesh_moments(state: Any, masses: Any, mesh: Any, axis_name: str) -> dict:
    """``production._moments`` of flat ``P(axis)`` arrays, in mesh order.

    One process: ``production._moments`` itself (one jitted reduction over the
    sharded arrays; XLA partitions it into per-device sums and an all-reduce).
    Several: the same per-device sums in a ``shard_map``, all-gathered and folded
    in mesh order -- the single process's all-reduce, bitwise -- then the same
    scalar factors.

    Parameters
    ----------
    state : Any
        ``(ndev * cap, 2, 3)``.
    masses : Any
        ``(ndev * cap,)``; zero on padding rows.
    mesh : Any
        The 1-D mesh.
    axis_name : str
        Its axis.

    Returns
    -------
    dict
        As ``production._moments``.
    """
    from jaccpot.production import _moments

    if not is_multiprocess():
        return _moments(state, masses)
    from jax.sharding import PartitionSpec as P

    key = (tuple(int(d.id) for d in np.asarray(mesh.devices).reshape(-1)), axis_name)
    program = _MOMENTS_PROGRAMS.get(key)
    if program is None:

        def sums(state_in: Any, masses_in: Any) -> dict:
            x = state_in[:, 0].astype(jnp.float64)
            v = state_in[:, 1].astype(jnp.float64)
            m = masses_in.astype(jnp.float64)
            local = {
                "mass": jnp.sum(m),
                "momentum": jnp.sum(m[:, None] * v, axis=0),
                "angular_momentum": jnp.sum(m[:, None] * jnp.cross(x, v), axis=0),
                "mass_moment": jnp.sum(m[:, None] * x, axis=0),
                "kinetic": jnp.sum(m * jnp.sum(v * v, axis=1)),
                "momentum_scale": jnp.sum(m * jnp.sqrt(jnp.sum(v * v, axis=1))),
            }
            out = {k: mesh_sum(val, axis_name) for k, val in local.items()}
            out["kinetic"] = 0.5 * out["kinetic"]
            return out

        spec = P(axis_name)
        program = jax.jit(
            jax.shard_map(
                sums, mesh=mesh, in_specs=(spec, spec), out_specs=P(), check_vma=False
            )
        )
        _MOMENTS_PROGRAMS[key] = program
    values = program(state, masses)
    return {k: np.asarray(host_value(v), np.float64) for k, v in values.items()}


# ----------------------------------------------------------------------
# Per-process input.
# ----------------------------------------------------------------------


def _device_chunks(num_particles: int, ndev: int) -> list:
    """Each device's initial id range (``np.array_split`` of ``0..N-1``).

    Parameters
    ----------
    num_particles : int
        N.
    ndev : int
        Devices.

    Returns
    -------
    list
        ``(lo, hi)`` per device, in mesh order.
    """
    base, extra = divmod(int(num_particles), int(ndev))
    out, lo = [], 0
    for d in range(int(ndev)):
        hi = lo + base + (1 if d < extra else 0)
        out.append((lo, hi))
        lo = hi
    return out


def local_device_indices(mesh: Any) -> list:
    """Mesh positions of this process's devices, ascending.

    Parameters
    ----------
    mesh : Any
        The 1-D mesh.

    Returns
    -------
    list
        Indices into the mesh's devices.
    """
    devices = list(np.asarray(mesh.devices).reshape(-1))
    me = jax.process_index()
    return [i for i, d in enumerate(devices) if int(d.process_index) == me]


def local_id_range(num_particles: int, mesh: Any) -> tuple[int, int]:
    """The ids this process provides as :class:`ProcessLocalParticles`.

    The single process's decomposition (``rollout.decompose``) first hands device
    ``d`` the ``d``-th of ``np.array_split(arange(N), ndev)`` (contiguous ids, in
    input order) and only then sorts by Morton key on the devices. A process
    provides its devices' chunks: ids ``lo..hi-1``.

    Parameters
    ----------
    num_particles : int
        N over the whole run.
    mesh : Any
        The 1-D mesh (its devices of one process contiguous: :func:`global_mesh`).

    Returns
    -------
    tuple[int, int]
        ``(lo, hi)``.

    Raises
    ------
    ValueError
        If this process's devices are not contiguous in the mesh.
    """
    local = local_device_indices(mesh)
    if not local:
        return 0, 0
    if local != list(range(local[0], local[-1] + 1)):
        raise ValueError(
            "this process's devices are not contiguous in the mesh: build it with "
            "multiprocess.global_mesh"
        )
    ndev = int(np.asarray(mesh.devices).size)
    chunks = _device_chunks(num_particles, ndev)
    return chunks[local[0]][0], chunks[local[-1]][1]


@dataclass
class ProcessLocalParticles:
    """A fresh run's input from one process: ITS particles only.

    Give it to :meth:`~jaccpot.production.ProductionRunner.run` as ``state`` (with
    ``masses=None``) in multi-process mode. The rows are the ids
    :func:`local_id_range` names, in id order: ``state[i]`` is particle ``lo + i``.
    Each process reads its own slice of the initial condition (by id range); the
    decomposition then runs on the devices, as from one host.

    Attributes
    ----------
    state : Any
        ``[hi - lo, 2, 3]``.
    masses : Any
        ``[hi - lo]``.
    num_particles : int
        N over the whole run.
    """

    state: Any
    masses: Any
    num_particles: int


def env_coordinator() -> Optional[dict]:
    """``initialize``'s arguments from ``JACCPOT_MP_*`` (the test harness's).

    ``JACCPOT_MP_COORDINATOR`` (``host:port``), ``JACCPOT_MP_NUM_PROCESSES`` and
    ``JACCPOT_MP_PROCESS_ID``.

    Returns
    -------
    Optional[dict]
        The keywords, or ``None`` when unset.
    """
    address = env_text("JACCPOT_MP_COORDINATOR")
    if not address:
        return None
    values = {}
    for key, name in (
        ("num_processes", "JACCPOT_MP_NUM_PROCESSES"),
        ("process_id", "JACCPOT_MP_PROCESS_ID"),
    ):
        text = env_text(name)
        try:
            values[key] = int(text)
        except ValueError:
            raise ValueError(
                f"JACCPOT_MP_COORDINATOR is set, so {name} must be an integer; "
                f"got {text!r}"
            ) from None
    return {"coordinator_address": address, **values}
