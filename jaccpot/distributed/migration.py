"""Migration to fixed cuts: every force of a segment on exact Morton domains.

A decomposition or a repartition leaves each device an exact Morton range of the
particles; one drift later a domain's boundary particles sit in Morton cells its
neighbour fills densely. The owner sees only its few stragglers there, so its
cell-leaf tree does not subdivide that region, and one coarse sparse leaf -- a
handful of particles spread over a cell several units wide -- fails the MAC against
much of the neighbour's core (S11 session 2, track F: on 8e6 Plummer over two A100s
one 19-particle leaf of radius 3.07 made 194,255 of the 361,797 cross near pairs
after a single drift, and the particles a device ships went 430,677 -> 3,743,931).
It recurs after EVERY drift following a cut, so repartitioning more often does not
remove it.

This module keeps the cuts FIXED and moves the particles instead: after the drift
and before the force, each particle whose Morton key lies outside its device's
splitter range moves to the device that owns that range (:func:`migrate_shard`), so
every force of a segment sees exact Morton domains -- the start's zero-drift cross
volume -- and the repartition between segments only rebalances the counts (it
chooses the splitters, :func:`splitters_shard`).

The keys are computed in the FORCE's box: :func:`~jaccpot.distributed.fused.global_mesh_bounds`
of the drifted live positions with the force's box shape (``TreeConfig.cubic_bounds``;
the galaxy preset's cube), the call the force's rebuild makes. Migration moves rows
only, so the force reduces the same live positions to the same box bit for bit, and
its trees are built in exactly the box the ownership was decided in. A box frozen at
the last repartition is not enough: the host model (``bench/multigpu_migration_model.py``)
measured a frozen cube at 10.8-17.5k cross near pairs on 2e5 Plummer against 8.8k in
the force's own box -- a box that moves by a fraction of a finest cell still moves
the cut by that much, and the particles in that sliver are intruders again.

The exchange is unconditional (a ragged all-to-all of every device's leavers, no
device-divergent predicate around a collective) through a buffer of
``migration_cap`` rows per device each way. Its need -- the most rows a device sends
or receives -- is computed from the all-gathered send-size matrix BEFORE anything
moves, so it is exact and identical on every device; when it does not fit the
buffer, or a device would end with more rows than its shard holds (``cap``), the
migration is DECLINED on every device at once (nothing moves: the old ownership is a
valid partition) and the step reports :data:`OVERFLOW_MIGRATION` or
:data:`OVERFLOW_SHARD`, which fail it: a declined migration leaves the stragglers
where they are, and the cross field they cost is what migration is there to remove.

Every per-row array travels with its particle: the position and the velocity, the
carried TOTAL acceleration (velocity Verlet's next kick reads it), the mass, the id,
and under ``mac_type='dehnen_error'`` the carried eq (16b) ``f_b`` (the next force's
thresholds). Each array moves bitwise (no cast): the exchange groups them by dtype.
A device keeps its staying rows in their order and appends what it receives;
padding rows sit on its first row with zero velocity, acceleration, mass and
``f_b`` and id ``-1``, as after a repartition.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaccpot.distributed.fused import AXIS_NAME, global_mesh_bounds

__all__ = [
    "OVERFLOW_MIGRATION",
    "OVERFLOW_SHARD",
    "balanced_splitters_shard",
    "migrate_shard",
    "migration_cap_for_need",
    "migration_need_shard",
    "splitters_shard",
]

#: the migration's buffer could not hold a device's leavers or arrivals: declined
OVERFLOW_MIGRATION = 1 << 12
#: a device would hold more rows than its shard after the migration: declined
OVERFLOW_SHARD = 1 << 13
#: the Morton sentinel of an empty device's lowest code (sorts above every code)
_CODE_SENTINEL = np.uint64((1 << 64) - 1)
#: smallest buffer the library sizes (rows per device each way)
_MIN_CAP = 256


def migration_cap_for_need(need: int, cap: int, *, headroom: float = 2.0) -> int:
    """The buffer rows a measured per-step need asks for.

    The next power of two above ``headroom x need`` (a migration's size moves
    with the boundary flux, which is steady but not constant), at least 256 rows,
    at most the shard's ``cap`` (no device can send or receive more).

    Parameters
    ----------
    need : int
        The most rows a device sent or received in one migration.
    cap : int
        Rows per device.
    headroom : float
        Factor over the need.

    Returns
    -------
    int
        Buffer rows per device, each way.
    """
    want = max(_MIN_CAP, int(np.ceil(float(headroom) * max(int(need), 1))))
    return int(min(int(cap), 1 << int(np.ceil(np.log2(want)))))


def _codes(positions: Array, n_live: Array, *, cubic: bool, axis_name: str) -> Array:
    """Morton codes of this device's rows in the force's box (inside ``shard_map``)."""
    from yggdrax.morton import morton_encode_impl

    bounds = global_mesh_bounds(
        positions, num_valid=n_live, axis_name=axis_name, cubic=cubic
    )
    return morton_encode_impl(positions, bounds)


def splitters_shard(
    positions: Array, count: Array, *, cubic: bool, axis_name: str = AXIS_NAME
) -> Array:
    """The fixed cuts of the current ownership (inside ``shard_map``).

    For a state whose devices own exact Morton ranges in the force's box at these
    positions -- what ``sfc_repartition`` leaves, in that box -- the lowest live
    code of each device ``d >= 1`` is a splitter that reproduces the ownership
    under ``sfc_repartition``'s rule ``searchsorted(splitters, code, 'right')``:
    every code of device ``d - 1`` lies below every code of device ``d``. An empty
    device takes the next device's cut (a reverse running minimum), so it is
    assigned nothing.

    Parameters
    ----------
    positions : Array
        ``(cap, 3)`` this device's positions.
    count : Array
        ``(1,)`` its live rows.
    cubic : bool
        The force's box shape (``TreeConfig.cubic_bounds``).
    axis_name : str
        Mesh axis.

    Returns
    -------
    Array
        ``(ndev - 1,)`` ``uint64`` splitters, identical on every device.
    """
    n_live = jnp.minimum(count[0], positions.shape[0])
    codes = _codes(positions, n_live, cubic=cubic, axis_name=axis_name)
    live = jnp.arange(positions.shape[0]) < n_live
    lowest = jnp.min(jnp.where(live, codes, _CODE_SENTINEL))
    per_device = jax.lax.all_gather(lowest, axis_name, tiled=False)
    # a reverse running minimum: an empty device (sentinel) takes the next cut
    suffix = jax.lax.cummin(per_device, reverse=True)
    return suffix[1:]


def balanced_splitters_shard(
    positions: Array, count: Array, *, cubic: bool, axis_name: str = AXIS_NAME
) -> Array:
    """Cuts that split the particles evenly, for an ownership of exact ranges.

    The rebalance between segments under migration. ``sfc_repartition``'s sample
    sort draws the same number of samples from every device whatever its count,
    so on domains that are already exact Morton ranges its median sample is the
    current cut and it moves nothing however unequal the counts are (2e6 disc +
    bulge on two A100s: 999,037 / 1,000,963 drifted to 1,012,142 / 987,858 over
    12 steps of fixed cuts, through two repartitions that sent 0 rows). Here the
    devices' live codes, in device order, ARE the sorted global sequence, so the
    cut ``j`` is the code at global rank ``j * N // ndev`` exactly: the device
    holding that rank reads it from its sorted codes, the others offer the
    sentinel, and a ``pmin`` replicates it. A migration to these cuts
    (:func:`migrate_shard`) then moves only the rows that change owner.

    Parameters
    ----------
    positions : Array
        ``(cap, 3)`` this device's positions.
    count : Array
        ``(1,)`` its live rows.
    cubic : bool
        The force's box shape.
    axis_name : str
        Mesh axis.

    Returns
    -------
    Array
        ``(ndev - 1,)`` ``uint64`` cuts, identical on every device; valid when
        every device's live rows are an exact Morton range in the force's box at
        these positions, the ranges ascending with the device index (what
        :func:`splitters_shard` and every migration leave).
    """
    cap = positions.shape[0]
    n_live = jnp.minimum(count[0], cap).astype(jnp.int32)
    live = jnp.arange(cap) < n_live
    codes = _codes(positions, n_live, cubic=cubic, axis_name=axis_name)
    ordered = jnp.sort(jnp.where(live, codes, _CODE_SENTINEL))
    counts = jax.lax.all_gather(n_live, axis_name, tiled=False)
    ndev = counts.shape[0]
    me = jax.lax.axis_index(axis_name)
    offset = (jnp.cumsum(counts) - counts)[me]
    targets = (jnp.arange(1, ndev, dtype=jnp.int64) * jnp.sum(counts)) // ndev
    local = targets - offset
    mine = (local >= 0) & (local < n_live)
    offered = jnp.where(mine, ordered[jnp.clip(local, 0, cap - 1)], _CODE_SENTINEL)
    return jax.lax.pmin(offered, axis_name)


def _destinations(
    positions: Array,
    count: Array,
    splitters: Array,
    *,
    cubic: bool,
    axis_name: str,
) -> tuple[Array, Array, Array, Array]:
    """Each live row's owner under ``splitters`` and the send-size matrix.

    Returns
    -------
    tuple[Array, Array, Array, Array]
        ``(n_live, live, dest, full)``: the live count, the ``(cap,)`` live mask,
        ``(cap,)`` int32 owners (this device on padding) and the replicated
        ``[ndev, ndev]`` matrix ``full[s, r]`` = rows ``s`` sends to ``r``
        (zero diagonal).
    """
    cap = positions.shape[0]
    ndev = int(splitters.shape[0]) + 1
    me = jax.lax.axis_index(axis_name).astype(jnp.int32)
    n_live = jnp.minimum(count[0], cap).astype(jnp.int32)
    live = jnp.arange(cap) < n_live
    codes = _codes(positions, n_live, cubic=cubic, axis_name=axis_name)
    dest = jnp.searchsorted(splitters, codes, side="right").astype(jnp.int32)
    dest = jnp.where(live, dest, me)
    leaving = live & (dest != me)
    sizes = jnp.bincount(jnp.where(leaving, dest, ndev), length=ndev + 1)[:ndev]
    full = jax.lax.all_gather(sizes.astype(jnp.int32), axis_name, tiled=False)
    return n_live, live, dest, full


def _verdict(
    full: Array, n_live: Array, *, cap: int, migration_cap: int, axis_name: str
) -> tuple[Array, Array, Array, Array]:
    """``(need, moved, bits, declined)`` of a send-size matrix, replicated."""
    sent = jnp.sum(full, axis=1)
    received = jnp.sum(full, axis=0)
    counts = jax.lax.all_gather(n_live, axis_name, tiled=False)
    after = counts - sent + received
    need = jnp.maximum(jnp.max(sent), jnp.max(received)).astype(jnp.int32)
    over_buffer = need > int(migration_cap)
    over_shard = jnp.any(after > int(cap))
    bits = jnp.where(over_buffer, jnp.uint32(OVERFLOW_MIGRATION), jnp.uint32(0)) | (
        jnp.where(over_shard, jnp.uint32(OVERFLOW_SHARD), jnp.uint32(0))
    )
    return need, jnp.sum(sent).astype(jnp.int32), bits, over_buffer | over_shard


def migration_need_shard(
    positions: Array,
    count: Array,
    splitters: Array,
    *,
    cubic: bool,
    axis_name: str = AXIS_NAME,
) -> tuple[Array, Array]:
    """What a migration at these positions would move (inside ``shard_map``).

    Parameters
    ----------
    positions : Array
        ``(cap, 3)`` this device's positions.
    count : Array
        ``(1,)`` its live rows.
    splitters : Array
        ``(ndev - 1,)`` ``uint64`` cuts (:func:`splitters_shard`).
    cubic : bool
        The force's box shape.
    axis_name : str
        Mesh axis.

    Returns
    -------
    tuple[Array, Array]
        ``(need, moved)``: the most rows a device would send or receive, and the
        rows moved mesh-wide; both replicated ``int32``.
    """
    n_live, _, _, full = _destinations(
        positions, count, splitters, cubic=cubic, axis_name=axis_name
    )
    need, moved, _, _ = _verdict(
        full,
        n_live,
        cap=positions.shape[0],
        migration_cap=positions.shape[0],
        axis_name=axis_name,
    )
    return need, moved


def _exchange_rows(
    arrays: list, send_sizes: Array, *, migration_cap: int, axis_name: str
) -> list:
    """Ragged-exchange per-row arrays bitwise: one round per dtype.

    Every array is ``(migration_cap, ...)``; arrays sharing a dtype are
    concatenated into one ``(rows, k)`` block (no cast), so the state, the
    acceleration, the mass and ``f_b`` of a float32 run cost one round and the
    int32 ids a second.
    """
    from yggdrax.distributed.comm import ragged_all_to_all_exchange

    flat = [a.reshape(a.shape[0], -1) for a in arrays]
    groups: dict = {}
    for i, a in enumerate(flat):
        groups.setdefault(jnp.dtype(a.dtype).name, []).append(i)
    out: list = [None] * len(arrays)
    for members in groups.values():
        block = jnp.concatenate([flat[i] for i in members], axis=1)
        routed, _, _ = ragged_all_to_all_exchange(
            block,
            send_sizes,
            output_capacity=int(migration_cap),
            axis_name=axis_name,
            fill_value=0,
        )
        col = 0
        for i in members:
            width = flat[i].shape[1]
            out[i] = routed[:, col : col + width].reshape(arrays[i].shape)
            col += width
    return out


def migrate_shard(
    state: Array,
    acceleration: Optional[Array],
    masses: Array,
    ids: Array,
    count: Array,
    splitters: Array,
    *,
    migration_cap: int,
    cubic: bool,
    force_scale: Optional[Array] = None,
    axis_name: str = AXIS_NAME,
) -> tuple[Any, ...]:
    """Move every particle outside its device's splitter range to its owner.

    One device's part (inside ``shard_map``); see the module docstring. The keys
    are those of the force's box at ``state``'s positions; the send sizes are
    all-gathered first, and a migration that does not fit ``migration_cap`` (or
    a shard) is declined on every device -- the exchange still runs, empty.

    Parameters
    ----------
    state : Array
        ``(cap, 2, 3)`` positions and velocities.
    acceleration : Optional[Array]
        ``(cap, 3)`` the carried total acceleration, or ``None`` (none to move).
    masses : Array
        ``(cap,)``.
    ids : Array
        ``(cap,)`` int32 global ids.
    count : Array
        ``(1,)`` live rows.
    splitters : Array
        ``(ndev - 1,)`` ``uint64`` cuts, replicated.
    migration_cap : int
        Buffer rows per device, each way (static).
    cubic : bool
        The force's box shape (static).
    force_scale : Optional[Array]
        ``(cap,)`` the carried ``f_b`` under the criterion, else ``None``.
    axis_name : str
        Mesh axis.

    Returns
    -------
    tuple[Any, ...]
        ``(state, acceleration, masses, ids, count, force_scale, info)`` in the
        new ownership (``acceleration`` / ``force_scale`` ``None`` when none was
        given); ``info``
        holds the replicated ``need`` (most rows a device sent or received, also
        when declined), ``moved`` (rows moved mesh-wide; ``0`` when declined) and
        ``bits`` (:data:`OVERFLOW_MIGRATION`, :data:`OVERFLOW_SHARD`).
    """
    cap = int(state.shape[0])
    mcap = int(migration_cap)
    ndev = int(splitters.shape[0]) + 1
    me = jax.lax.axis_index(axis_name).astype(jnp.int32)
    n_live, live, dest, full = _destinations(
        state[:, 0], count, splitters, cubic=cubic, axis_name=axis_name
    )
    need, moved, bits, declined = _verdict(
        full, n_live, cap=cap, migration_cap=mcap, axis_name=axis_name
    )
    leaving = live & (dest != me) & jnp.logical_not(declined)
    send_sizes = jnp.where(declined, 0, full[me])
    # each leaver's slot in the send buffer: grouped by destination, in row order
    offsets = jnp.cumsum(send_sizes) - send_sizes
    slot = jnp.full((cap,), mcap, jnp.int32)
    for d in range(ndev):
        to_d = leaving & (dest == d)
        rank = jnp.cumsum(to_d.astype(jnp.int32)) - 1
        slot = jnp.where(to_d, offsets[d] + rank, slot)
    rows = jnp.arange(cap, dtype=jnp.int32)
    send_src = jnp.zeros((mcap,), jnp.int32).at[slot].set(rows, mode="drop")
    # the staying rows first, in their order
    staying = live & jnp.logical_not(leaving)
    n_stay = jnp.sum(staying.astype(jnp.int32))
    keep_slot = jnp.where(staying, jnp.cumsum(staying.astype(jnp.int32)) - 1, cap)
    keep_src = jnp.zeros((cap,), jnp.int32).at[keep_slot].set(rows, mode="drop")

    arrays = [state, masses, ids] + [
        a for a in (acceleration, force_scale) if a is not None
    ]
    received = _exchange_rows(
        [a[send_src] for a in arrays],
        send_sizes,
        migration_cap=mcap,
        axis_name=axis_name,
    )
    got = dict(zip(("state", "masses", "ids"), received[:3]))
    extra = iter(received[3:])
    if acceleration is not None:
        got["acceleration"] = next(extra)
    if force_scale is not None:
        got["force_scale"] = next(extra)
    n_received = jnp.sum(jnp.where(declined, 0, full[:, me])).astype(jnp.int32)
    new_count = n_stay + n_received
    j = rows - n_stay
    from_received = (j >= 0) & (j < n_received)
    j = jnp.clip(j, 0, mcap - 1)
    live_new = rows < new_count

    def place(old: Array, got: Array, fill: Any) -> Array:
        shape = (cap,) + (1,) * (old.ndim - 1)
        picked = jnp.where(from_received.reshape(shape), got[j], old[keep_src])
        return jnp.where(live_new.reshape(shape), picked, jnp.asarray(fill, old.dtype))

    state_new = place(state, got["state"], 0)
    # padding rows on the first row (live whenever the device holds a particle),
    # as after a repartition: an eager prepare sees them as particles
    state_new = state_new.at[:, 0].set(
        jnp.where(live_new[:, None], state_new[:, 0], state_new[0, 0][None, :])
    )
    acc_new = (
        None if acceleration is None else place(acceleration, got["acceleration"], 0)
    )
    masses_new = place(masses, got["masses"], 0)
    ids_new = place(ids, got["ids"], -1)
    fb_new = None if force_scale is None else place(force_scale, got["force_scale"], 0)
    info = {"need": need, "moved": jnp.where(declined, 0, moved), "bits": bits}
    return (
        state_new,
        acc_new,
        masses_new,
        ids_new,
        new_count.reshape(1).astype(count.dtype),
        fb_new,
        info,
    )
