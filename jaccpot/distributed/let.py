"""The LET (locally essential tree) exchange of the cross summaries (track M4).

The default exchange all_gathers every device's whole two-sided summary tree
(:func:`jaccpot.distributed.cross._summary_rows`), ``S`` rows per device, so every
device receives ``(ndev - 1) x S`` rows per force, a volume that grows with the
device count at fixed particles per device. The LET exchange sends a peer only the
part of the tree that the cross walk against that peer opens, found EXACTLY by the
walk itself, in three steps (:func:`let_exchange`):

1. **Coarse trees.** Every device cuts its summary tree where a subtree holds at
   most ``coarse_particles`` live particles: the TOP is the root and every child of
   a node above that size; a top node at or below it is a CUT node, whose subtree a
   peer may request. The tops travel in one all_gather (``coarse_capacity`` rows per
   device, a small fraction of the tree).
2. **Requests by the walk.** Every device runs the cross walk between its FULL tree
   and every peer's top, in the walk's own orientation (lower device first). An
   internal cut node gets two children: a PROBE (a leaf copy of itself) and a dead
   slot. The walk splits the cut node exactly when the full walk would split it --
   the pairs above the cut are the full walk's, decided on the same floats -- and
   every split leaves at least one output pair with the probe in it (accepted, near,
   or under the other side's split). A probe in any output pair is a request for
   that cut node's subtree; the requests travel back in one all_to_all.
3. **Blocks.** Each device sends every peer its top plus the subtrees that peer
   requested, with the child links rewritten into the block (an unrequested cut node
   arrives as a leaf), in one ragged all-to-all.

**Why the pair sets are the all_gather path's.** The full walk between two trees
splits only top nodes and nodes inside requested subtrees (step 2 is exact), and
every such node travels with its children; so the walk over the two blocks visits
the full walk's pairs, on bitwise-identical rows.

**Why momentum still cancels (M1).** Device ``s`` walks (its block for ``d``, the
block ``d`` sent it), device ``d`` (the block ``s`` sent it, its block for ``s``):
the SAME two blocks in the same orientation, so both ends derive the identical pair
set by construction.

Everything downstream -- the multipole and particle exchanges, the near and far
lists -- is the default exchange's (``cross._symmetric_exchange``), keyed in the
LET's own index space instead of the ``[ndev x S]`` gathered one.
"""

from __future__ import annotations

import math
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array
from yggdrax.distributed.comm import ragged_all_to_all_exchange

from jaccpot._searchsorted import searchsorted_method

__all__ = [
    "LetBlocks",
    "let_bytes",
    "let_exchange",
]

# two-sided summary row columns (``cross._summary_rows``)
_COL_RADIUS = 3
_COL_LEFT = 4
_COL_RIGHT = 5
_COL_ACTIVE = 6
_COL_COUNT = 7
_COL_TABLE = 8  # cross._SUMMARY_TABLE_COL: G M, threshold, s_1..s_p


class LetBlocks(NamedTuple):
    """The LET blocks this device sent and received, and the walk table over both.

    The walk's index space is ``[T + R)``: the send buffer (this device's pruned
    tree once per peer, peers in device order) then the receive buffer (each peer's
    pruned tree for this device, peers in device order). Within a block the rows
    keep their sender's ascending summary index.

    Attributes
    ----------
    left, right : Array
        ``(T + R,)`` child links in the walk's index space (``-1``: none).
    centers : Array
        ``(T + R, 3)``.
    radii : Array
        ``(T + R,)``.
    active : Array
        ``(T + R,)`` bool.
    table : Any
        ``(T + R, p + 2)`` eq (16a) columns, or ``None``.
    count : Array
        ``(T + R,)`` int32 live particles under each row (its sender's count).
    orig : Array
        ``(T + R,)`` int32 each row's summary index in its sender's tree (``-1``
        in the padding).
    peer : Array
        ``(T + R,)`` int32 the peer of each row's block (``ndev`` in the padding).
    seed_a, seed_b : Array
        ``(ndev,)`` per peer the walk's seed: the two block roots, the lower
        device's first; ``-1`` for this device and for a peer with an empty block.
    send_sizes, recv_sizes : Array
        ``(ndev,)`` rows sent to / received from each peer.
    send_needed, recv_needed, coarse_needed : Array
        Rows this device had to send, was sent, and has in its top (exact past the
        capacities).
    requested : Array
        ``(ndev,)`` cut nodes of this device each peer requested.
    request_walk : Any
        The request walk's result (its pair counts and overflow flags).
    overflow : Array
        A buffer overflowed or a request walk lost pairs (any device): the blocks
        may miss rows the walk opens.
    send_capacity : int
        ``T``. Static.
    """

    left: Array
    right: Array
    centers: Array
    radii: Array
    active: Array
    table: Any
    count: Array
    orig: Array
    peer: Array
    seed_a: Array
    seed_b: Array
    send_sizes: Array
    recv_sizes: Array
    send_needed: Array
    recv_needed: Array
    coarse_needed: Array
    requested: Array
    request_walk: Any
    overflow: Array
    send_capacity: int


def _block_slots(sizes: Array, capacity: int) -> tuple[Array, Array, Array]:
    """Per slot of a buffer laid out block by block: its block, liveness, starts.

    Parameters
    ----------
    sizes : Array
        ``(ndev,)`` block sizes.
    capacity : int
        Buffer rows. Static.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(block, live, start)``: per slot its block (``ndev`` past the live rows),
        whether it is live, and every block's first slot.
    """
    idx = jnp.int32
    sizes = sizes.astype(idx)
    start = jnp.cumsum(sizes, dtype=idx) - sizes
    total = jnp.sum(sizes, dtype=idx)
    slot = jnp.arange(capacity, dtype=idx)
    # block of slot q: the number of block ends at or below q
    ends = jnp.cumsum(sizes, dtype=idx)
    block = jnp.searchsorted(
        ends, slot, side="right", method=searchsorted_method()
    ).astype(idx)
    live = slot < total
    return jnp.where(live, block, sizes.shape[0]), live, start


def _parents(left: Array, right: Array, internal: Array) -> Array:
    """Each summary node's parent (``-1`` at the root and the padding).

    Parameters
    ----------
    left, right : Array
        ``(S,)`` child links.
    internal : Array
        ``(S,)`` internal live nodes.

    Returns
    -------
    Array
        ``(S,)`` int32 parents.
    """
    S = int(left.shape[0])
    idx = jnp.int32
    at = jnp.arange(S, dtype=idx)
    sentinel = jnp.asarray(S, idx)
    return (
        jnp.full((S,), -1, idx)
        .at[jnp.where(internal, left, sentinel)]
        .set(at, mode="drop")
        .at[jnp.where(internal, right, sentinel)]
        .set(at, mode="drop")
    )


def _probes_in(a: Array, b: Array, n: Array, *, S: int, C: int, dead: int) -> Array:
    """The probe slots (``peer x C + top slot``) of a pair list's ends, ``-1`` else.

    Parameters
    ----------
    a, b : Array
        ``(P,)`` the pairs' ends in the request walk's index space.
    n : Array
        Live prefix length.
    S, C, dead : int
        This device's rows, the top capacity and the dead row. Static.

    Returns
    -------
    Array
        ``(2 P,)`` int32.
    """
    idx = jnp.int32
    live = jnp.arange(jnp.asarray(a).shape[0]) < n
    block = 2 * C
    out = []
    for end in (a, b):
        e = jnp.asarray(end).astype(idx)
        off = e - S
        is_probe = live & (e >= S) & (e < dead) & ((off % block) >= C)
        out.append(jnp.where(is_probe, (off // block) * C + (off % block) - C, -1))
    return jnp.concatenate(out)


def let_exchange(
    rows: Array,
    valid: Array,
    me: Array,
    *,
    ndev: int,
    send_capacity: int,
    recv_capacity: int,
    coarse_capacity: int,
    coarse_particles: int,
    axis_name: str,
    walk: Callable[..., Any],
    error_order: int = 0,
) -> LetBlocks:
    """Send every peer the part of this device's summary tree its walk opens.

    Parameters
    ----------
    rows : Array
        ``(S, k)`` this device's two-sided summary rows (``cross._summary_rows``).
    valid : Array
        ``(S,)`` rows holding a summary node (``block_nodes >= 0``).
    me : Array
        This device's index along the mesh axis.
    ndev : int
        Mesh size. Static.
    send_capacity, recv_capacity : int
        Static row capacities of the send and receive buffers.
    coarse_capacity : int
        Static rows of a device's top (the coarse tree every peer gathers).
    coarse_particles : int
        A subtree holding at most this many live particles is requested whole or
        not at all. Static.
    axis_name : str
        Mesh axis.
    walk : Callable[..., Any]
        The cross walk: ``walk(left, right, centers, radii, active, table, seed_a,
        seed_b)`` with the caller's MAC and capacities (``cross._let_walk``'s).
    error_order : int
        ``p`` under eq (16a) (the rows carry its table), else ``0``. Static.

    Returns
    -------
    LetBlocks
        The blocks and the walk table over them.
    """
    idx = jnp.int32
    S, k = int(rows.shape[0]), int(rows.shape[1])
    T, R, C = int(send_capacity), int(recv_capacity), int(coarse_capacity)
    nd = int(ndev)
    dtype = rows.dtype
    p = int(error_order)
    width = p + 2
    me = jnp.asarray(me).astype(idx)
    at = jnp.arange(S, dtype=idx)
    dev = jnp.arange(nd, dtype=idx)
    lower_mine = me < dev

    left = jnp.round(rows[:, _COL_LEFT]).astype(idx)
    right = jnp.round(rows[:, _COL_RIGHT]).astype(idx)
    internal = valid & (left >= 0)
    parent = _parents(left, right, internal)
    count = jnp.round(rows[:, _COL_COUNT]).astype(idx)

    # ---- 1. the top: the root and every child of a node above the size --------
    with jax.named_scope("cross_let_top"):
        big = internal & (count > int(coarse_particles))
        top = valid & ((parent < 0) | big[jnp.maximum(parent, 0)])
        cut = top & internal & ~big
        tpos = jnp.cumsum(top, dtype=idx) - 1
        coarse_needed = jnp.sum(top, dtype=idx)
        tslot = jnp.where(top & (tpos < C), tpos, C)
        c_left = jnp.where(big, tpos[jnp.maximum(left, 0)], -1)
        c_right = jnp.where(big, tpos[jnp.maximum(right, 0)], -1)
        coarse_rows = jnp.concatenate(
            [
                rows[:, :_COL_LEFT],
                c_left.astype(dtype)[:, None],
                c_right.astype(dtype)[:, None],
                rows[:, _COL_ACTIVE:],
                cut.astype(dtype)[:, None],
            ],
            axis=1,
        )
        coarse = jnp.zeros((C, k + 1), dtype).at[tslot].set(coarse_rows, mode="drop")
        n_top = jnp.minimum(coarse_needed, C)
    tops = jax.lax.all_gather(coarse, axis_name, tiled=False)
    n_tops = jax.lax.all_gather(n_top, axis_name, tiled=False)

    # ---- 2. the request walk: my full tree against every peer's top -----------
    with jax.named_scope("cross_let_request"):
        # index space: [my S rows | per peer: C top rows, C probes | one dead row]
        block = 2 * C
        base = S + dev * block
        dead = S + nd * block
        top_live = jnp.arange(C, dtype=idx)[None, :] < n_tops[:, None]
        t_live = top_live & (dev != me)[:, None]
        t_cut = t_live & (tops[..., k] > 0.5)
        t_l = jnp.round(tops[..., _COL_LEFT]).astype(idx)
        t_r = jnp.round(tops[..., _COL_RIGHT]).astype(idx)
        probe = base[:, None] + C + jnp.arange(C, dtype=idx)[None, :]
        top_left = jnp.where(t_cut, probe, jnp.where(t_l >= 0, base[:, None] + t_l, -1))
        top_right = jnp.where(t_cut, dead, jnp.where(t_r >= 0, base[:, None] + t_r, -1))
        none = jnp.full((nd, C), -1, idx)
        tail = jnp.full((1,), -1, idx)
        q_left = jnp.concatenate(
            [
                jnp.where(internal, left, -1),
                jnp.stack([top_left, none], 1).reshape(-1),
                tail,
            ]
        )
        q_right = jnp.concatenate(
            [
                jnp.where(internal, right, -1),
                jnp.stack([top_right, none], 1).reshape(-1),
                tail,
            ]
        )
        # a probe is a leaf copy of its cut node; live only under a cut node
        top_rows = tops[..., :k]
        q_rows = jnp.concatenate(
            [
                rows,
                jnp.stack([top_rows, top_rows], 1).reshape(nd * block, k),
                jnp.zeros((1, k), dtype),
            ],
            axis=0,
        )
        t_act = t_live & (tops[..., _COL_ACTIVE] > 0.5)
        q_active = jnp.concatenate(
            [
                valid & (rows[:, _COL_ACTIVE] > 0.5),
                jnp.stack([t_act, t_act & t_cut], 1).reshape(-1),
                jnp.zeros((1,), bool),
            ]
        )
        alive = (dev != me) & (n_tops > 0) & valid[0]
        root_mine = jnp.zeros((nd,), idx)
        seed_a = jnp.where(alive, jnp.where(lower_mine, root_mine, base), -1)
        seed_b = jnp.where(alive, jnp.where(lower_mine, base, root_mine), -1)
        req_walk = walk(
            q_left,
            q_right,
            q_rows[:, 0:3],
            q_rows[:, _COL_RADIUS],
            q_active,
            q_rows[:, _COL_TABLE : _COL_TABLE + width] if p > 0 else None,
            seed_a.astype(idx),
            seed_b.astype(idx),
        )
        kw = dict(S=S, C=C, dead=dead)
        hits = jnp.concatenate(
            [
                _probes_in(req_walk.far_a, req_walk.far_b, req_walk.far_count, **kw),
                _probes_in(req_walk.near_a, req_walk.near_b, req_walk.near_count, **kw),
            ]
        )
        span = nd * C
        asked = (
            jnp.zeros((span,), idx)
            .at[jnp.where(hits >= 0, hits, span)]
            .set(1, mode="drop")
            .reshape(nd, C)
        )
        req_overflow = (
            req_walk.far_overflow | req_walk.near_overflow | req_walk.queue_overflow
        )
    # row d of `asked` goes to device d; row r of `got` came from device r
    got = jax.lax.all_to_all(asked, axis_name, 0, 0, tiled=True)

    # ---- 3. the blocks: the top plus every requested subtree ------------------
    with jax.named_scope("cross_let_pack"):
        # each node's top ancestor (itself in the top): pointer doubling
        rounds = max(1, int(math.ceil(math.log2(max(S, 2)))))
        jump = jnp.where(top | (parent < 0), at, parent)
        for _ in range(rounds):
            jump = jump[jump]
        anc_slot = jnp.where(valid, jnp.minimum(tpos[jump], C), C)
        orig_col = at.astype(dtype)[:, None]

        def per_peer(d: Array, carry: tuple) -> tuple:
            buf, offset, sizes = carry
            wanted = jnp.concatenate([got[d] > 0, jnp.zeros((1,), bool)])
            sent = valid & (top | wanted[anc_slot]) & (d != me)
            pos = jnp.cumsum(sent, dtype=idx) - 1
            size = jnp.sum(sent, dtype=idx)
            go = sent & internal & sent[jnp.maximum(left, 0)]
            new_left = jnp.where(go, pos[jnp.maximum(left, 0)], -1).astype(dtype)
            new_right = jnp.where(go, pos[jnp.maximum(right, 0)], -1).astype(dtype)
            out = jnp.concatenate(
                [
                    rows[:, :_COL_LEFT],
                    new_left[:, None],
                    new_right[:, None],
                    rows[:, _COL_ACTIVE:],
                    orig_col,
                ],
                axis=1,
            )
            slot = jnp.where(sent, offset + pos, T)
            buf = buf.at[slot].set(out, mode="drop")
            return buf, offset + size, sizes.at[d].set(size)

        buf, send_needed, send_sizes = jax.lax.fori_loop(
            0,
            nd,
            per_peer,
            (jnp.zeros((T, k + 1), dtype), jnp.asarray(0, idx), jnp.zeros((nd,), idx)),
        )
        requested = jnp.sum(got, axis=1, dtype=idx)
    recv, recv_sizes, _ = jax.named_call(
        ragged_all_to_all_exchange, name="cross_exchange_let"
    )(buf, send_sizes, output_capacity=R, axis_name=axis_name)
    recv_sizes = recv_sizes.astype(idx)
    recv_needed = jnp.sum(recv_sizes, dtype=idx)
    # a short top or a lost request hides rows from a PEER: everyone flags
    lost = jax.lax.pmax(
        ((coarse_needed > C) | req_overflow).astype(idx), axis_name
    ).astype(bool)
    overflow = (send_needed > T) | (recv_needed > R) | lost

    # ---- the final walk's table over [send | recv] -----------------------------
    s_peer, s_live, s_start = _block_slots(send_sizes, T)
    r_peer, r_live, r_start = _block_slots(recv_sizes, R)
    both = jnp.concatenate([buf, recv], axis=0)
    live = jnp.concatenate([s_live, r_live])
    peer = jnp.concatenate([s_peer, r_peer])
    first = jnp.concatenate(
        [
            s_start[jnp.minimum(s_peer, nd - 1)],
            T + r_start[jnp.minimum(r_peer, nd - 1)],
        ]
    )

    def link(col: int) -> Array:
        local = jnp.round(both[:, col]).astype(idx)
        out = jnp.where(live & (local >= 0), first + local, -1)
        return jnp.where(out < T + R, out, -1)

    alive2 = (dev != me) & (send_sizes > 0) & (recv_sizes > 0)
    alive2 = alive2 & (s_start < T) & (r_start < R)
    root_peer = T + r_start
    seed2_a = jnp.where(alive2, jnp.where(lower_mine, s_start, root_peer), -1)
    seed2_b = jnp.where(alive2, jnp.where(lower_mine, root_peer, s_start), -1)
    return LetBlocks(
        left=link(_COL_LEFT),
        right=link(_COL_RIGHT),
        centers=both[:, 0:3],
        radii=both[:, _COL_RADIUS],
        active=live & (both[:, _COL_ACTIVE] > 0.5),
        table=both[:, _COL_TABLE : _COL_TABLE + width] if p > 0 else None,
        count=jnp.where(live, jnp.round(both[:, _COL_COUNT]).astype(idx), 0),
        orig=jnp.where(live, jnp.round(both[:, k]).astype(idx), -1),
        peer=jnp.where(live, peer, nd),
        seed_a=seed2_a.astype(idx),
        seed_b=seed2_b.astype(idx),
        send_sizes=send_sizes,
        recv_sizes=recv_sizes,
        send_needed=send_needed,
        recv_needed=recv_needed,
        coarse_needed=coarse_needed,
        requested=requested,
        request_walk=req_walk,
        overflow=overflow,
        send_capacity=T,
    )


def let_bytes(blocks: LetBlocks, *, row_width: int, itemsize: int) -> dict[str, Any]:
    """Bytes this device put on the wire for the LET rows (the used rows).

    Parameters
    ----------
    blocks : LetBlocks
        The exchange.
    row_width : int
        Floats per row on the wire (the summary row plus the index column).
    itemsize : int
        Bytes per float.

    Returns
    -------
    dict
        ``send_rows``, ``recv_rows``, ``send_bytes``, ``recv_bytes``.
    """
    send = jnp.sum(blocks.send_sizes)
    recv = jnp.sum(blocks.recv_sizes)
    per = int(row_width) * int(itemsize)
    return {
        "send_rows": send,
        "recv_rows": recv,
        "send_bytes": send * per,
        "recv_bytes": recv * per,
    }
