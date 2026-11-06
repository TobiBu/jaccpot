"""Escapers on the mesh: ``ProductionRunner(mesh=..., escapers=EscaperPolicy(...))`` (MESC).

The single card's escaper path (:mod:`jaccpot.escapers`, ESC) in the multi-GPU
mode, with the same Hamiltonian, ledger and policy. What differs is the layout.

THE LAYOUT: a separate escaper TAIL. The bound ("tree") rows are an ordinary
:class:`~jaccpot.distributed.segment.MeshState` -- the decomposition, the cuts,
the global box, the Morton box, the cross walk, the migration, the repartition
and every capacity see the bound rows ONLY, because the escapers are not in it.
The escapers live in :class:`EscaperTail`: flat ``(ndev * cap_e, ...)`` arrays
sharded ``P(axis)`` like the layout, ``count[d]`` leading live rows per device,
with their own capacity ``cap_e``. They need no tree and no neighbours, so the
tail is never sorted, cut or migrated; a particle stays on the device it escaped
from, and a returner re-enters the bound rows of the device whose tail held it
(the next migration or repartition moves it to its domain). Every edge move is
device-local: no exchange at an edge.

WHY NOT rows at the end of each device's layout ("beyond the local live cut"):
the layout's invariant -- ``count[d]`` live rows lead, the rest is padding --
is read by the force (``num_valid``), the migration, the repartition, the
gathers and the restart files; a second live block behind the padding would
have to be taught to all of them, and its capacity would be the bound rows'
headroom. A tail of its own touches none of them.

PER STEP (inside the segment's ``shard_map(scan)``): the tail drifts and kicks
with the bound rows (the same velocity Verlet association). After the tree force,
``M_b`` and ``X_b`` -- the bound rows' mass and centre of mass, per-device fp64
partial sums all-gathered and folded in mesh order -- give the escapers'
``a_e = -G M_b (x_e - X_b) / |x_e - X_b|^3``; the escapers' total pull, a second
all-gather of a 3-vector, gives every bound row the uniform ``a_u = G sum_e m_e
(x_e - X_b) / |x_e - X_b|^3``. The external field acts on both. All-gathers and a
fold, not ``psum``: the same bits in one process with N devices and in N
processes (gloo, NCCL), whatever order their all-reduce sums in.

AT AN EDGE (the runner's, every ``policy.every`` segments): :func:`membership`
counts the bound rows beyond ``r_escape`` and the escapers within ``r_return``
per device (the single card's test, in the state's dtype); a change runs
:func:`resplit` -- one device-local program, fixed shapes: one compile for every
split -- with the energy evaluated before and after for the ledger. A device
whose tail would overflow grows ``cap_e`` first (:func:`grown_tail`: the tail
programs compile again for the new capacity, recorded in the split).

WHILE NOBODY ESCAPES (an empty tail) the segment runner runs its DEFAULT
programs: escapers on but nobody escaping is the run without them, bitwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

__all__ = [
    "EscaperTail",
    "bound_frame_shard",
    "coupling_shard",
    "empty_tail",
    "grown_tail",
    "initial_membership",
    "membership",
    "place_split",
    "resplit",
    "tail_capacity",
    "tail_energy",
]

#: the smallest escaper tail per device (a run whose escapers appear later)
TAIL_FLOOR = 64
#: the tail's headroom over the most escapers one device holds
TAIL_HEADROOM = 1.5
#: the tail's rows per device are a multiple of this
TAIL_ALIGN = 64


@dataclass
class EscaperTail:
    """The escapers, one block per device (flat, sharded ``P(axis)``).

    Attributes
    ----------
    state : Any
        ``(ndev * cap_e, 2, 3)``; padding rows zero.
    masses : Any
        ``(ndev * cap_e,)``; zero on padding.
    ids : Any
        ``(ndev * cap_e,)`` int32 global ids; ``-1`` on padding.
    count : Any
        ``(ndev,)`` int32 escapers per device (leading rows).
    cap : int
        ``cap_e``: rows per device.
    total : int
        Escapers over the mesh (host).
    acceleration : Optional[Any]
        ``(ndev * cap_e, 3)`` total acceleration (coupling plus the external
        field) at ``state``'s positions; ``None`` until a start force computed it.
    """

    state: Any
    masses: Any
    ids: Any
    count: Any
    cap: int
    total: int
    acceleration: Optional[Any] = None


def tail_capacity(need: int) -> int:
    """``cap_e`` for a device that holds ``need`` escapers.

    Parameters
    ----------
    need : int
        The most escapers one device holds (or will).

    Returns
    -------
    int
        ``max(TAIL_FLOOR, ceil(TAIL_HEADROOM need))`` rounded up to a multiple of
        ``TAIL_ALIGN``.
    """
    rows = max(TAIL_FLOOR, int(math.ceil(TAIL_HEADROOM * int(need))))
    return int(-(-rows // TAIL_ALIGN) * TAIL_ALIGN)


def _wide(dtype: Any) -> Any:
    """The reduction dtype (fp64 under x64, as :mod:`jaccpot.escapers`)."""
    return jnp.float64 if jax.config.jax_enable_x64 else dtype


def _tiny(wide: Any) -> Any:
    """The mass floor of ``X_b``'s division."""
    return jnp.asarray(1e-300 if wide == jnp.float64 else 1e-30, wide)


def _fold_gather(value: Any, axis_name: str) -> Any:
    """``value`` summed over the mesh in mesh order (inside ``shard_map``).

    An all-gather and a fold, never ``psum``: the bits do not depend on the
    runtime's all-reduce order (one process or several, gloo or NCCL).

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
    from jaccpot.distributed.multiprocess import _fold

    return _fold(jax.lax.all_gather(value, axis_name, tiled=False))


def bound_frame_shard(
    positions: Any, masses: Any, live_rows: Any, axis_name: str
) -> tuple:
    """``(M_b, X_b)`` of the bound rows over the mesh (inside ``shard_map``).

    Per device the single card's sums (``jaccpot.escapers._bound_frame``: the
    mass-weighted positions in the state's dtype, summed in fp64), then one
    all-gather of the four partial sums, folded in mesh order.

    Parameters
    ----------
    positions : Any
        ``(cap, 3)`` this device's bound rows.
    masses : Any
        ``(cap,)``.
    live_rows : Any
        ``(cap,)`` boolean.
    axis_name : str
        The mesh axis.

    Returns
    -------
    tuple
        ``M_b`` and ``X_b`` (``(3,)``), in the wide dtype, replicated.
    """
    wide = _wide(positions.dtype)
    m_live = jnp.where(live_rows, masses, jnp.zeros((), masses.dtype))
    part = jnp.concatenate(
        [
            jnp.sum(m_live, dtype=wide)[None],
            jnp.sum(m_live[:, None] * positions, axis=0, dtype=wide),
        ]
    )
    total = _fold_gather(part, axis_name)
    m_b = total[0]
    return m_b, total[1:] / jnp.maximum(m_b, _tiny(wide))


def _escaper_geometry(positions: Any, x_b: Any, live: Any) -> tuple:
    """``d = x - X_b`` and the masked ``1/r^3`` of escaper rows (state dtype).

    Parameters
    ----------
    positions : Any
        ``(cap_e, 3)``.
    x_b : Any
        ``(3,)`` wide.
    live : Any
        ``(cap_e,)`` the escaper rows.

    Returns
    -------
    tuple
        ``(d, inv_r3)``; ``inv_r3`` zero on padding.
    """
    dtype = positions.dtype
    d = positions - x_b.astype(dtype)[None, :]
    r2 = jnp.sum(d * d, axis=1)
    inv_r = jnp.where(
        live, 1.0 / jnp.sqrt(jnp.where(live, r2, 1)), jnp.zeros((), dtype)
    )
    return d, inv_r * inv_r * inv_r


def coupling_shard(
    m_b: Any,
    x_b: Any,
    tail_positions: Any,
    tail_masses: Any,
    tail_live: Any,
    G: float,
    axis_name: str,
) -> tuple:
    """The monopole coupling on one device (inside ``shard_map``).

    Parameters
    ----------
    m_b : Any
        ``M_b`` (wide).
    x_b : Any
        ``X_b`` (wide).
    tail_positions : Any
        ``(cap_e, 3)`` this device's escapers.
    tail_masses : Any
        ``(cap_e,)``.
    tail_live : Any
        ``(cap_e,)`` boolean.
    G : float
        The gravitational constant.
    axis_name : str
        The mesh axis.

    Returns
    -------
    tuple
        ``(a_u, a_e, scale_e)``: the bound rows' uniform pull ``(3,)`` (wide,
        replicated: one all-gather of every device's escapers' pull), the
        escapers' ``(cap_e, 3)`` accelerations (zero on padding) and their
        magnitudes (their eq (16b) ``f_b``).
    """
    dtype = tail_positions.dtype
    wide = _wide(dtype)
    d, inv_r3 = _escaper_geometry(tail_positions, x_b, tail_live)
    a_e = (-jnp.asarray(G, dtype) * m_b.astype(dtype)) * d * inv_r3[:, None]
    m_e = jnp.where(tail_live, tail_masses, jnp.zeros((), tail_masses.dtype))
    pull = jnp.sum((m_e * inv_r3)[:, None] * d, axis=0, dtype=wide)
    a_u = jnp.asarray(G, wide) * _fold_gather(pull, axis_name)
    scale_e = jnp.sqrt(jnp.sum(a_e * a_e, axis=1))
    return a_u, a_e, scale_e


# ----------------------------------------------------------------------------
# The run's start: the membership and the placement on the host.
# ----------------------------------------------------------------------------


def _chunk_sums(pos: np.ndarray, masses: np.ndarray, bound: np.ndarray) -> np.ndarray:
    """One device chunk's ``[sum m, sum m x]`` over its bound rows (fp64).

    Parameters
    ----------
    pos : np.ndarray
        ``(k, 3)``.
    masses : np.ndarray
        ``(k,)``.
    bound : np.ndarray
        ``(k,)`` boolean.

    Returns
    -------
    np.ndarray
        ``(4,)`` float64.
    """
    m = np.where(bound, masses, np.zeros((), masses.dtype))
    out = np.zeros((4,), np.float64)
    out[0] = np.sum(m, dtype=np.float64)
    out[1:] = np.sum(m[:, None] * pos, axis=0, dtype=np.float64)
    return out


def initial_membership(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    lo: int,
    num_particles: int,
    mesh: Any,
    policy: Any,
) -> tuple:
    """The run's start: which of the given rows are escapers (host).

    The single card's :func:`jaccpot.escapers.split_state` (from a state with
    every row in the tree, the centre of mass taken again from the new tree rows
    until nothing changes, at most four passes), with the sums of each device's
    initial id chunk (``multiprocess._device_chunks``) folded in mesh order:
    the same bits whether one process holds every chunk or each process its own
    (:class:`~jaccpot.distributed.multiprocess.ProcessLocalParticles`).

    Parameters
    ----------
    positions : np.ndarray
        ``(hi - lo, 3)``: ids ``lo..hi-1`` (this process's chunks, or all).
    masses : np.ndarray
        ``(hi - lo,)``.
    lo : int
        The first id.
    num_particles : int
        N over the run.
    mesh : Any
        The 1-D mesh.
    policy : Any
        The :class:`~jaccpot.escapers.EscaperPolicy`.

    Returns
    -------
    tuple
        ``(escaper, transfers)``: ``(hi - lo,)`` boolean, and the rows that
        changed side over the passes, over every process (the single card's
        count).

    Raises
    ------
    RuntimeError
        If no particle would stay in the tree.
    """
    from jaccpot.distributed.multiprocess import (
        _device_chunks,
        allgather_by_device,
        allgather_host,
        is_multiprocess,
        local_device_indices,
    )

    ndev = int(np.asarray(mesh.devices).size)
    chunks = _device_chunks(int(num_particles), ndev)
    local = local_device_indices(mesh) if is_multiprocess() else list(range(ndev))
    dtype = positions.dtype
    r_out = np.asarray(float(policy.r_escape) ** 2, dtype)
    r_in = np.asarray(float(policy.return_radius) ** 2, dtype)
    escaper = np.zeros(positions.shape[0], bool)
    transfers = 0
    for _ in range(4):
        sums = {
            d: _chunk_sums(
                positions[chunks[d][0] - lo : chunks[d][1] - lo],
                masses[chunks[d][0] - lo : chunks[d][1] - lo],
                ~escaper[chunks[d][0] - lo : chunks[d][1] - lo],
            )
            for d in local
        }
        parts = allgather_by_device(mesh, sums) if is_multiprocess() else [
            sums[d] for d in range(ndev)
        ]
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        m_b = total[0]
        x_b = (total[1:] / max(m_b, 1e-300)).astype(dtype)
        d_vec = positions - x_b[None, :]
        r2 = np.sum(d_vec * d_vec, axis=1)
        new = np.where(escaper, ~(r2 < r_in), r2 > r_out)
        counts = allgather_host(
            np.asarray([int(np.sum(new != escaper)), int(np.sum(new))], np.int64)
        ).reshape(-1, 2)
        changed, count = int(counts[:, 0].sum()), int(counts[:, 1].sum())
        if changed == 0:
            break
        if count >= int(num_particles):
            raise RuntimeError(
                f"the escaper policy (r_escape {policy.r_escape}) leaves no "
                "particle in the tree"
            )
        escaper = new
        transfers += changed
    return escaper, transfers


def place_split(
    mesh: Any,
    state: np.ndarray,
    masses: np.ndarray,
    escaper: np.ndarray,
    *,
    lo: int,
    num_particles: int,
    cap: int,
    num_samples: int,
    axis_name: str,
    cubic: Optional[bool],
    tail_cap: Optional[int] = None,
) -> tuple:
    """The bound rows decomposed over the mesh and the escapers placed in their tail.

    Device ``d`` starts from the ``d``-th initial id chunk (``rollout.decompose``'s
    even split of ``0..N-1``) WITHOUT its escapers, laid out as ``decompose`` lays
    a chunk (live rows first, padding on the chunk's first bound particle), and the
    same Morton repartition runs over the bound rows only; the chunk's escapers
    (ascending ids) are its tail. From one host or from each process's chunks
    alike (bitwise).

    Parameters
    ----------
    mesh : Any
        The 1-D mesh.
    state : np.ndarray
        ``(hi - lo, 2, 3)`` ids ``lo..hi-1``.
    masses : np.ndarray
        ``(hi - lo,)``.
    escaper : np.ndarray
        ``(hi - lo,)`` boolean (:func:`initial_membership`).
    lo : int
        The first id.
    num_particles : int
        N over the run.
    cap : int
        Bound rows per device.
    num_samples : int
        Pivot samples per device.
    axis_name : str
        The mesh axis.
    cubic : Optional[bool]
        The decomposition's box shape (``rollout.decompose``).
    tail_cap : Optional[int]
        ``cap_e``; ``None``: :func:`tail_capacity` of the most escapers a chunk
        holds.

    Returns
    -------
    tuple
        ``(ShardState, EscaperTail)``.

    Raises
    ------
    ValueError
        If a chunk's bound rows do not fit ``cap``.
    """
    from jaccpot.distributed.multiprocess import (
        _device_chunks,
        allgather_host,
        is_multiprocess,
        local_device_indices,
    )
    from jaccpot.distributed.rollout import _decompose_placed, _mesh_devices

    devices = _mesh_devices(mesh)
    ndev = len(devices)
    chunks = _device_chunks(int(num_particles), ndev)
    local = local_device_indices(mesh) if is_multiprocess() else list(range(ndev))
    pos, vel = state[:, 0], state[:, 1]
    dtype = pos.dtype
    mass = np.asarray(masses, dtype)
    esc_local = np.asarray(
        [int(np.sum(escaper[chunks[d][0] - lo : chunks[d][1] - lo])) for d in local],
        np.int64,
    )
    most = int(allgather_host(np.asarray(esc_local.max() if len(local) else 0)).max())
    cap_e = tail_capacity(most) if tail_cap is None else int(tail_cap)
    if most > cap_e:
        raise ValueError(f"a device holds {most} escapers > tail capacity {cap_e}")
    bound_parts: list = [[] for _ in range(5)]
    tail_parts: list = [[] for _ in range(4)]
    for d in local:
        a, b = chunks[d]
        rows = np.arange(a - lo, b - lo)
        is_esc = escaper[rows]
        keep, out = rows[~is_esc], rows[is_esc]
        k = int(keep.shape[0])
        if k > int(cap):
            raise ValueError(f"cap={cap} cannot hold device {d}'s {k} bound rows")
        flat_pos = np.zeros((cap, 3), dtype)
        flat_vel = np.zeros((cap, 3), dtype)
        flat_mass = np.zeros((cap,), dtype)
        flat_id = np.full((cap,), -1, np.int32)
        flat_pos[:k] = pos[keep]
        flat_vel[:k] = vel[keep]
        flat_mass[:k] = mass[keep]
        flat_id[:k] = (keep + lo).astype(np.int32)
        if k:
            flat_pos[k:] = pos[keep[0]]
        for j, arr in enumerate(
            (flat_pos, flat_vel, flat_mass, flat_id, np.asarray([k], np.int32))
        ):
            bound_parts[j].append(jax.device_put(arr, devices[d]))
        e = int(out.shape[0])
        t_state = np.zeros((cap_e, 2, 3), dtype)
        t_mass = np.zeros((cap_e,), dtype)
        t_id = np.full((cap_e,), -1, np.int32)
        t_state[:e] = state[out]
        t_mass[:e] = mass[out]
        t_id[:e] = (out + lo).astype(np.int32)
        for j, arr in enumerate((t_state, t_mass, t_id, np.asarray([e], np.int32))):
            tail_parts[j].append(jax.device_put(arr, devices[d]))
    sh = NamedSharding(mesh, P(axis_name))

    def assembled(parts: list) -> Any:
        return jax.make_array_from_single_device_arrays(
            (ndev * int(parts[0].shape[0]),) + tuple(parts[0].shape[1:]), sh, parts
        )

    placed = tuple(assembled(parts) for parts in bound_parts)
    shard, _ = _decompose_placed(
        mesh,
        placed,
        cap=int(cap),
        num_samples=int(num_samples),
        axis_name=axis_name,
        cubic=cubic,
    )
    t_state, t_mass, t_id, t_count = (assembled(parts) for parts in tail_parts)
    total = int(allgather_host(np.asarray(int(np.sum(escaper)), np.int64)).sum())
    return shard, EscaperTail(
        state=t_state,
        masses=t_mass,
        ids=t_id,
        count=t_count,
        cap=cap_e,
        total=total,
    )


def empty_tail(
    mesh: Any, axis_name: str, dtype: Any, cap_e: Optional[int] = None
) -> EscaperTail:
    """A tail with no escapers (escapers on, nobody beyond ``r_escape``).

    Parameters
    ----------
    mesh : Any
        The 1-D mesh.
    axis_name : str
        The mesh axis.
    dtype : Any
        The state's dtype.
    cap_e : Optional[int]
        Rows per device; ``None``: :func:`tail_capacity` of none.

    Returns
    -------
    EscaperTail
        Zeros, ids ``-1``, counts zero.
    """
    if cap_e is None:
        cap_e = tail_capacity(0)
    ndev = int(np.asarray(mesh.devices).size)
    sh = NamedSharding(mesh, P(axis_name))
    rows = ndev * int(cap_e)
    return EscaperTail(
        state=jax.device_put(np.zeros((rows, 2, 3), dtype), sh),
        masses=jax.device_put(np.zeros((rows,), dtype), sh),
        ids=jax.device_put(np.full((rows,), -1, np.int32), sh),
        count=jax.device_put(np.zeros((ndev,), np.int32), sh),
        cap=int(cap_e),
        total=0,
    )


# ----------------------------------------------------------------------------
# The edge: membership, the re-split, the tail's growth, the energy.
# ----------------------------------------------------------------------------

#: the edge programs, by (kind, mesh devices, axis, static options)
_PROGRAMS: dict = {}


def _mesh_key(mesh: Any) -> tuple:
    """The mesh's devices as a key."""
    return tuple(int(d.id) for d in np.asarray(mesh.devices).reshape(-1))


def _masks(
    bstate: Any,
    bmasses: Any,
    bcount: Any,
    tstate: Any,
    tcount: Any,
    r_out: float,
    r_in: float,
    axis: str,
) -> tuple:
    """One device's membership (inside ``shard_map``): the single card's test.

    Returns
    -------
    tuple
        ``(live_b, live_t, out, back, m_b, x_b)``: the bound rows leaving, the
        escapers returning, and the bound frame.
    """
    cap, cap_e = int(bstate.shape[0]), int(tstate.shape[0])
    live_b = jnp.arange(cap) < jnp.minimum(bcount[0], cap)
    live_t = jnp.arange(cap_e) < jnp.minimum(tcount[0], cap_e)
    m_b, x_b = bound_frame_shard(bstate[:, 0], bmasses, live_b, axis)
    dtype = bstate.dtype

    def r2_of(pos: Any) -> Any:
        d = pos - x_b.astype(dtype)[None, :]
        return jnp.sum(d * d, axis=1)

    out = live_b & (r2_of(bstate[:, 0]) > r_out * r_out)
    back = live_t & (r2_of(tstate[:, 0]) < r_in * r_in)
    return live_b, live_t, out, back, m_b, x_b


def membership(
    mesh: Any, axis_name: str, layout: Any, tail: EscaperTail, policy: Any
) -> tuple:
    """Who changes side at this edge: per device, rows leaving and returning (host).

    Parameters
    ----------
    mesh : Any
        The 1-D mesh.
    axis_name : str
        Its axis.
    layout : Any
        The bound :class:`~jaccpot.distributed.segment.MeshState`.
    tail : EscaperTail
        The escapers.
    policy : Any
        The radii.

    Returns
    -------
    tuple
        ``(out, back)``: ``(ndev,)`` int arrays.
    """
    from jaccpot.distributed.multiprocess import host_value

    r_out, r_in = float(policy.r_escape), float(policy.return_radius)
    key = ("membership", _mesh_key(mesh), axis_name, r_out, r_in)
    program = _PROGRAMS.get(key)
    if program is None:

        def body(bstate: Any, bmasses: Any, bcount: Any, tstate: Any, tcount: Any):
            _, _, out, back, _, _ = _masks(
                bstate, bmasses, bcount, tstate, tcount, r_out, r_in, axis_name
            )
            return (
                jnp.sum(out, dtype=jnp.int32)[None],
                jnp.sum(back, dtype=jnp.int32)[None],
            )

        spec = P(axis_name)
        program = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(spec,) * 5,
                out_specs=(spec, spec),
                check_vma=False,
            )
        )
        _PROGRAMS[key] = program
    out, back = program(layout.state, layout.masses, layout.count, tail.state, tail.count)
    return host_value(out).astype(np.int64), host_value(back).astype(np.int64)


def _compact(mask: Any, rows: int) -> tuple:
    """Stable indices of ``mask``'s rows, ``rows`` of them, and the live mask."""
    n = jnp.sum(mask, dtype=jnp.int32)
    idx = jnp.nonzero(mask, size=int(rows), fill_value=0)[0]
    return idx, jnp.arange(int(rows)) < n, n


def resplit(
    mesh: Any,
    axis_name: str,
    layout: Any,
    tail: EscaperTail,
    policy: Any,
    G: float,
    *,
    donate: bool = True,
) -> tuple:
    """The rows that changed side moved, device-locally (one program, any split).

    Per device: the bound rows that stay, in their order, then the returners
    (the tail's order); the escapers that stay, then the new ones. Padding rows
    of the layout sit on its first live row with zero velocity, mass, id ``-1``
    (and zero ``f_b``); the tail's are zero. Under the criterion a returner's
    ``f_b`` is its coupling's magnitude at this edge (the single card's escaper
    row ``f_b``). The carried accelerations are dropped: the next segment's start
    force evaluates the new split. The caller checked that every device's rows
    fit (:func:`membership`).

    Parameters
    ----------
    mesh : Any
        The 1-D mesh.
    axis_name : str
        Its axis.
    layout : Any
        The bound :class:`~jaccpot.distributed.segment.MeshState` (donated).
    tail : EscaperTail
        The escapers (donated).
    policy : Any
        The radii.
    G : float
        The gravitational constant.
    donate : bool
        Donate the inputs.

    Returns
    -------
    tuple
        ``(layout, tail)``: the new :class:`MeshState` (no acceleration) and
        :class:`EscaperTail` (no acceleration; ``total`` left to the caller).
    """
    from jaccpot.distributed.segment import MeshState

    r_out, r_in = float(policy.r_escape), float(policy.return_radius)
    with_fb = layout.force_scale is not None
    key = ("resplit", _mesh_key(mesh), axis_name, r_out, r_in, float(G), with_fb, donate)
    program = _PROGRAMS.get(key)
    if program is None:

        def body(
            bstate: Any,
            bmasses: Any,
            bids: Any,
            bcount: Any,
            tstate: Any,
            tmasses: Any,
            tids: Any,
            tcount: Any,
            *fb: Any,
        ) -> tuple:
            cap, cap_e = int(bstate.shape[0]), int(tstate.shape[0])
            live_b, live_t, out, back, m_b, x_b = _masks(
                bstate, bmasses, bcount, tstate, tcount, r_out, r_in, axis_name
            )
            pool_state = jnp.concatenate([bstate, tstate])
            pool_m = jnp.concatenate([bmasses, tmasses])
            pool_id = jnp.concatenate([bids, tids])
            to_b = jnp.concatenate([live_b & ~out, back])
            to_t = jnp.concatenate([out, live_t & ~back])
            ib, rows_b, nb = _compact(to_b, cap)
            it, rows_t, nt = _compact(to_t, cap_e)
            new_state = pool_state[ib]
            pad = jnp.stack([new_state[0, 0], jnp.zeros_like(new_state[0, 0])])
            b_state = jnp.where(rows_b[:, None, None], new_state, pad[None])
            zero_m = jnp.zeros((), bmasses.dtype)
            b_m = jnp.where(rows_b, pool_m[ib], zero_m)
            b_id = jnp.where(rows_b, pool_id[ib], -1)
            t_state = jnp.where(
                rows_t[:, None, None], pool_state[it], jnp.zeros((), tstate.dtype)
            )
            t_m = jnp.where(rows_t, pool_m[it], zero_m)
            t_id = jnp.where(rows_t, pool_id[it], -1)
            outs = (b_state, b_m, b_id, nb[None], t_state, t_m, t_id, nt[None])
            if fb:
                _, _, scale_e = coupling_shard(
                    m_b, x_b, tstate[:, 0], tmasses, live_t, G, axis_name
                )
                pool_fb = jnp.concatenate([fb[0], scale_e.astype(fb[0].dtype)])
                outs = outs + (
                    jnp.where(rows_b, pool_fb[ib], jnp.zeros((), fb[0].dtype)),
                )
            return outs

        spec = P(axis_name)
        rows = 9 if with_fb else 8
        program = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(spec,) * rows,
                out_specs=(spec,) * rows,
                check_vma=False,
            ),
            donate_argnums=tuple(range(rows)) if donate else (),
        )
        _PROGRAMS[key] = program
    args = (
        layout.state,
        layout.masses,
        layout.ids,
        layout.count,
        tail.state,
        tail.masses,
        tail.ids,
        tail.count,
    ) + ((layout.force_scale,) if with_fb else ())
    stale = (layout.acceleration, tail.acceleration)
    outs = program(*args)
    for arr in stale:
        # the old split's accelerations: no reader any more (freed now, not when
        # the caller's references go)
        if isinstance(arr, jax.Array) and not arr.is_deleted():
            arr.delete()
    new_layout = MeshState(
        state=outs[0],
        masses=outs[1],
        ids=outs[2],
        count=outs[3],
        force_scale=outs[8] if with_fb else None,
    )
    new_tail = EscaperTail(
        state=outs[4],
        masses=outs[5],
        ids=outs[6],
        count=outs[7],
        cap=tail.cap,
        total=tail.total,
    )
    return new_layout, new_tail


def grown_tail(mesh: Any, axis_name: str, tail: EscaperTail, cap_e: int) -> EscaperTail:
    """``tail`` with ``cap_e`` rows per device (zero padding appended per device).

    Parameters
    ----------
    mesh : Any
        The 1-D mesh.
    axis_name : str
        Its axis.
    tail : EscaperTail
        The escapers.
    cap_e : int
        The new capacity (``> tail.cap``).

    Returns
    -------
    EscaperTail
        The same escapers, no acceleration (the next start force computes it).
    """
    extra = int(cap_e) - int(tail.cap)
    key = ("grow", _mesh_key(mesh), axis_name, int(tail.cap), int(cap_e))
    program = _PROGRAMS.get(key)
    if program is None:

        def body(state: Any, masses: Any, ids: Any) -> tuple:
            return (
                jnp.concatenate([state, jnp.zeros((extra, 2, 3), state.dtype)]),
                jnp.concatenate([masses, jnp.zeros((extra,), masses.dtype)]),
                jnp.concatenate([ids, jnp.full((extra,), -1, ids.dtype)]),
            )

        spec = P(axis_name)
        program = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(spec,) * 3,
                out_specs=(spec,) * 3,
                check_vma=False,
            )
        )
        _PROGRAMS[key] = program
    state, masses, ids = program(tail.state, tail.masses, tail.ids)
    return EscaperTail(
        state=state,
        masses=masses,
        ids=ids,
        count=tail.count,
        cap=int(cap_e),
        total=tail.total,
    )


def tail_energy(
    mesh: Any,
    axis_name: str,
    layout: Any,
    tail: EscaperTail,
    G: float,
    *,
    external_potential_fn: Optional[Callable[..., Any]] = None,
    external_params: Any = None,
) -> dict:
    """The split's energies over the mesh: ``W_c``, ``K`` and ``W_ext`` of the escapers.

    :func:`jaccpot.escapers.coupling_energy` with the per-device fp64 sums folded
    in mesh order; the external potential on each device's escaper rows (the
    field's own contract), padding masked.

    Parameters
    ----------
    mesh : Any
        The 1-D mesh.
    axis_name : str
        Its axis.
    layout : Any
        The bound :class:`~jaccpot.distributed.segment.MeshState`.
    tail : EscaperTail
        The escapers.
    G : float
        The gravitational constant.
    external_potential_fn : Optional[Callable[..., Any]]
        ``potential(rows, params)`` (``potential(rows)`` without parameters), or
        ``None``.
    external_params : Any
        Its parameters.

    Returns
    -------
    dict
        Host floats: ``couple``, ``kinetic_escapers``, ``external_escapers``,
        ``mass_bound``, ``mass_escapers``.
    """
    from jaccpot.distributed.multiprocess import host_value

    leaves, treedef = jax.tree_util.tree_flatten(external_params)
    field_fn = external_potential_fn
    key = (
        "energy",
        _mesh_key(mesh),
        axis_name,
        float(G),
        None if field_fn is None else id(field_fn),
        str(treedef),
        tuple((tuple(np.shape(x)), str(jnp.result_type(x))) for x in leaves),
    )
    program = _PROGRAMS.get(key)
    if program is None:

        def body(
            bstate: Any,
            bmasses: Any,
            bcount: Any,
            tstate: Any,
            tmasses: Any,
            tcount: Any,
            params: Any,
        ) -> tuple:
            cap, cap_e = int(bstate.shape[0]), int(tstate.shape[0])
            live_b = jnp.arange(cap) < jnp.minimum(bcount[0], cap)
            live_t = jnp.arange(cap_e) < jnp.minimum(tcount[0], cap_e)
            m_b, x_b = bound_frame_shard(bstate[:, 0], bmasses, live_b, axis_name)
            wide = _wide(tstate.dtype)
            zero = jnp.zeros((), wide)
            d = tstate[:, 0].astype(wide) - x_b[None, :]
            r = jnp.sqrt(jnp.where(live_t, jnp.sum(d * d, axis=1), 1.0))
            m_e = jnp.where(live_t, tmasses.astype(wide), zero)
            vel = tstate[:, 1].astype(wide)
            part = jnp.stack(
                [
                    jnp.sum(m_e / r),
                    0.5 * jnp.sum(m_e * jnp.sum(vel * vel, axis=1)),
                    zero,
                    jnp.sum(m_e),
                ]
            )
            if field_fn is not None:
                ext = field_fn(tstate) if params is None else field_fn(tstate, params)
                ext64 = jnp.where(live_t, jnp.asarray(ext).astype(wide), zero)
                part = part.at[2].set(jnp.sum(m_e * ext64))
            total = _fold_gather(part, axis_name)
            return (
                -jnp.asarray(G, wide) * m_b * total[0],
                total[1],
                total[2],
                m_b,
                total[3],
            )

        spec = P(axis_name)
        program = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(spec,) * 6 + (P(),),
                out_specs=(P(),) * 5,
                check_vma=False,
            )
        )
        _PROGRAMS[key] = program
    out = program(
        layout.state,
        layout.masses,
        layout.count,
        tail.state,
        tail.masses,
        tail.count,
        external_params,
    )
    couple, kin, ext, m_b, m_e = (float(host_value(v)) for v in out)
    return {
        "couple": couple,
        "kinetic_escapers": kin,
        "external_escapers": ext,
        "mass_bound": m_b,
        "mass_escapers": m_e,
    }
