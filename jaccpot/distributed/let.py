"""The LET (locally essential tree) exchange of the cross summaries (track M4).

The default exchange all_gathers every device's whole two-sided summary tree
(:func:`jaccpot.distributed.cross._summary_rows`), ``S`` rows per device, so every
device receives ``(ndev - 1) x S`` rows per force, a volume that grows with the
device count at fixed particles per device. The LET exchange sends a peer only the
part of the tree that the cross walk against that peer opens, found EXACTLY by the
walk itself, in request rounds (:func:`let_exchange`):

1. **Round r** (granularity ``coarse_particles[r]`` live particles, decreasing):
   each device sends every peer the nodes whose parent is EXPANDED for that peer --
   above the round's size, or requested by that peer in an earlier round. Expanded
   is closed upward, so every ancestor of a sent node is expanded too. A sent
   internal node that is not expanded is a CUT node.
2. **Request walk.** Each device runs the cross walk between its FULL tree and every
   block it received, in the walk's own orientation (lower device first), with
   every cut node given two children: a PROBE (a leaf copy of itself) and a dead
   slot. Above the cuts the walk is the full walk on the same floats, so it splits
   a cut node exactly when the full walk would, and every split leaves an output
   pair holding the probe (accepted, near, or under the other side's split). A
   probe in an output pair is a request; the requests go back to the senders in a
   ragged all-to-all aligned with their send buffers.
3. **Final blocks.** The last round's blocks with every last-round request's
   subtree whole, in one ragged all-to-all.

**Why the pair sets are the all_gather path's.** A node the full walk splits has
every ancestor split; the highest one that is not expanded was a cut node in some
round's block and was requested there. So every node the full walk splits travels
with its children, and the walk over the final blocks visits the full walk's pairs
on bitwise-identical rows. No bound and no MAC geometry: the requests follow the
walk's own decisions (an earlier, bound-based version sent every row of a Plummer
domain: no compact summary of a peer's radix tree bounds which nodes it splits).

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
    send_needed, recv_needed : Array
        Rows this device had to send / was sent (exact past the capacities).
    coarse_needed : Array
        The largest request round's rows sent or received.
    requested : Array
        ``(ndev,)`` cut nodes of this device each peer requested, all rounds.
    request_walk : Any
        The last request walk's result (its pair counts and overflow flags).
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


def _round_blocks(
    rows: Array,
    valid: Array,
    internal: Array,
    left: Array,
    right: Array,
    parent: Array,
    expanded: Callable[[Array], Array],
    full_below: Callable[[Array], Array],
    me: Array,
    *,
    ndev: int,
    capacity: int,
    tail: Callable[[Array, Array], Array],
) -> tuple[Array, Array, Array]:
    """Lay out one block per peer: the nodes whose parent is expanded for it.

    Parameters
    ----------
    rows : Array
        ``(S, k)`` summary rows.
    valid, internal : Array
        ``(S,)`` summary nodes, and those with children.
    left, right, parent : Array
        ``(S,)`` links.
    expanded : Callable[[Array], Array]
        ``expanded(d) -> (S,)`` nodes whose children peer ``d`` gets. Closed
        upward (an expanded node's parent is expanded), so a node travels when its
        parent is expanded: every ancestor then is.
    full_below : Callable[[Array], Array]
        ``full_below(d) -> (S,)`` nodes that travel whatever their parent (inside
        a subtree ``d`` gets whole).
    me : Array
        This device's index.
    ndev : int
        Mesh size. Static.
    capacity : int
        Send buffer rows. Static.
    tail : Callable[[Array, Array], Array]
        ``tail(d, sent) -> (S, m)`` extra columns per row.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(buffer, needed, sizes)``: rows with the links rewritten into the block
        (``-1`` where the children stay home), then the summary index, then
        ``tail``; the rows the blocks needed; per peer sizes.
    """
    idx = jnp.int32
    S = int(rows.shape[0])
    dtype = rows.dtype
    orig_col = jnp.arange(S, dtype=idx).astype(dtype)[:, None]
    width = int(rows.shape[1]) + 1 + int(tail(jnp.asarray(0, idx), valid).shape[1])

    def per_peer(d: Array, carry: tuple) -> tuple:
        buf, offset, sizes = carry
        ex = expanded(d)
        sent = valid & ((parent < 0) | ex[jnp.maximum(parent, 0)] | full_below(d))
        sent = sent & (d != me)
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
                tail(d, sent).astype(dtype),
            ],
            axis=1,
        )
        slot = jnp.where(sent, offset + pos, capacity)
        buf = buf.at[slot].set(out, mode="drop")
        return buf, offset + size, sizes.at[d].set(size)

    return jax.lax.fori_loop(
        0,
        int(ndev),
        per_peer,
        (
            jnp.zeros((int(capacity), width), dtype),
            jnp.asarray(0, idx),
            jnp.zeros((int(ndev),), idx),
        ),
    )


def let_exchange(
    rows: Array,
    valid: Array,
    me: Array,
    *,
    ndev: int,
    send_capacity: int,
    recv_capacity: int,
    coarse_particles: tuple[int, ...],
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
        Static row capacities of the send and receive buffers (every round's
        blocks are part of the final ones).
    coarse_particles : tuple[int, ...]
        The request rounds' granularities, decreasing: in round ``r`` a node above
        ``coarse_particles[r]`` live particles, or one a peer requested earlier,
        has its children sent; the others are CUT nodes a peer may request. The
        last round's requests bring whole subtrees. Static.
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
        The final blocks and the walk table over them.
    """
    idx = jnp.int32
    S, k = int(rows.shape[0]), int(rows.shape[1])
    T, R = int(send_capacity), int(recv_capacity)
    nd = int(ndev)
    dtype = rows.dtype
    p = int(error_order)
    width = p + 2
    me = jnp.asarray(me).astype(idx)
    at = jnp.arange(S, dtype=idx)
    dev = jnp.arange(nd, dtype=idx)
    lower_mine = me < dev
    levels = tuple(int(m) for m in coarse_particles)
    if not levels or any(b >= a for a, b in zip(levels, levels[1:])):
        raise ValueError(f"coarse_particles={levels}: one or more, decreasing")

    left = jnp.round(rows[:, _COL_LEFT]).astype(idx)
    right = jnp.round(rows[:, _COL_RIGHT]).astype(idx)
    internal = valid & (left >= 0)
    parent = _parents(left, right, internal)
    count = jnp.round(rows[:, _COL_COUNT]).astype(idx)
    # opened[d, x]: peer d's walk splits my node x (a cut node it requested)
    opened = jnp.zeros((nd, S), bool)
    lost = jnp.asarray(False)
    rounds_needed = []
    requested = jnp.zeros((nd,), idx)
    req_walk = None

    for r, m in enumerate(levels):
        with jax.named_scope(f"cross_let_round{r}"):
            big = internal & (count > m)

            def expanded(d: Array, big: Array = big, opened: Array = opened) -> Array:
                return big | opened[d]

            def no_subtree(d: Array) -> Array:
                return jnp.zeros((S,), bool)

            def cut_col(
                d: Array, sent: Array, big: Array = big, opened: Array = opened
            ) -> Array:
                return (sent & internal & ~(big | opened[d]))[:, None]

            buf, needed, sizes = _round_blocks(
                rows,
                valid,
                internal,
                left,
                right,
                parent,
                expanded,
                no_subtree,
                me,
                ndev=nd,
                capacity=T,
                tail=cut_col,
            )
            recv, recv_sizes, _ = ragged_all_to_all_exchange(
                buf, sizes, output_capacity=R, axis_name=axis_name
            )
            recv_sizes = recv_sizes.astype(idx)
            rounds_needed.append(jnp.maximum(needed, jnp.sum(recv_sizes)))
            lost = lost | (needed > T) | (jnp.sum(recv_sizes) > R)
            # the request walk: [my S rows | the R received rows | R probes | dead]
            r_peer, r_live, r_start = _block_slots(recv_sizes, R)
            first = S + r_start[jnp.minimum(r_peer, nd - 1)]
            dead = S + 2 * R
            is_cut = r_live & (recv[:, k + 1] > 0.5)
            probe = S + R + jnp.arange(R, dtype=idx)

            def link(col: int, cut_to: Array) -> Array:
                local = jnp.round(recv[:, col]).astype(idx)
                out = jnp.where(r_live & (local >= 0), first + local, -1)
                return jnp.where(is_cut, cut_to, out)

            none = jnp.full((R,), -1, idx)
            q_left = jnp.concatenate(
                [jnp.where(internal, left, -1), link(_COL_LEFT, probe), none, none[:1]]
            )
            q_right = jnp.concatenate(
                [
                    jnp.where(internal, right, -1),
                    link(_COL_RIGHT, jnp.full((R,), dead, idx)),
                    none,
                    none[:1],
                ]
            )
            got_rows = recv[:, :k]
            q_rows = jnp.concatenate(
                [rows, got_rows, got_rows, jnp.zeros((1, k), dtype)], axis=0
            )
            r_act = r_live & (recv[:, _COL_ACTIVE] > 0.5)
            q_active = jnp.concatenate(
                [
                    valid & (rows[:, _COL_ACTIVE] > 0.5),
                    r_act,
                    r_act & is_cut,
                    jnp.zeros((1,), bool),
                ]
            )
            alive = (dev != me) & (recv_sizes > 0) & valid[0] & (r_start < R)
            root_peer = S + r_start
            root_mine = jnp.zeros((nd,), idx)
            seed_a = jnp.where(alive, jnp.where(lower_mine, root_mine, root_peer), -1)
            seed_b = jnp.where(alive, jnp.where(lower_mine, root_peer, root_mine), -1)
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
            hit = jnp.zeros((R + 1,), idx)
            for a, b, n in (
                (req_walk.far_a, req_walk.far_b, req_walk.far_count),
                (req_walk.near_a, req_walk.near_b, req_walk.near_count),
            ):
                live = jnp.arange(jnp.asarray(a).shape[0]) < n
                for end in (a, b):
                    e = jnp.asarray(end).astype(idx)
                    probe_hit = live & (e >= S + R) & (e < dead)
                    hit = hit.at[jnp.where(probe_hit, e - S - R, R)].set(1)
            lost = lost | req_walk.far_overflow | req_walk.near_overflow
            lost = lost | req_walk.queue_overflow
            # the requests back to their senders, aligned with their send buffers
            back, _, _ = ragged_all_to_all_exchange(
                hit[:R, None].astype(dtype),
                recv_sizes,
                output_capacity=T,
                axis_name=axis_name,
            )
            s_peer, s_live, _ = _block_slots(sizes, T)
            asked = s_live & (back[:, 0] > 0.5) & (buf[:, k + 1] > 0.5)
            node = jnp.round(buf[:, k]).astype(idx)
            opened = opened.at[
                jnp.where(asked, s_peer, nd), jnp.where(asked, node, 0)
            ].set(True, mode="drop")
            requested = requested + jax.ops.segment_sum(
                asked.astype(idx), jnp.where(asked, s_peer, nd), nd + 1
            )[:nd].astype(idx)

    # ---- the final blocks: the last round's, the last requests' subtrees whole --
    with jax.named_scope("cross_let_pack"):
        last = levels[-1]
        big = internal & (count > last)
        top = valid & ((parent < 0) | big[jnp.maximum(parent, 0)])
        # each node's top ancestor at the last granularity: pointer doubling
        jump = jnp.where(top | (parent < 0), at, parent)
        for _ in range(max(1, int(math.ceil(math.log2(max(S, 2)))))):
            jump = jump[jump]

        def expanded_final(d: Array) -> Array:
            return big | opened[d]

        def whole(d: Array) -> Array:
            return opened[d][jump] & ~top

        def no_tail(d: Array, sent: Array) -> Array:
            return jnp.zeros((S, 0), dtype)

        buf, send_needed, send_sizes = _round_blocks(
            rows,
            valid,
            internal,
            left,
            right,
            parent,
            expanded_final,
            whole,
            me,
            ndev=nd,
            capacity=T,
            tail=no_tail,
        )
    recv, recv_sizes, _ = jax.named_call(
        ragged_all_to_all_exchange, name="cross_exchange_let"
    )(buf, send_sizes, output_capacity=R, axis_name=axis_name)
    recv_sizes = recv_sizes.astype(idx)
    recv_needed = jnp.sum(recv_sizes, dtype=idx)
    # a lost request hides rows from a PEER: everyone flags
    lost = jax.lax.pmax(lost.astype(idx), axis_name).astype(bool)
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

    def final_link(col: int) -> Array:
        local = jnp.round(both[:, col]).astype(idx)
        out = jnp.where(live & (local >= 0), first + local, -1)
        return jnp.where(out < T + R, out, -1)

    alive2 = (dev != me) & (send_sizes > 0) & (recv_sizes > 0)
    alive2 = alive2 & (s_start < T) & (r_start < R)
    root_peer = T + r_start
    seed2_a = jnp.where(alive2, jnp.where(lower_mine, s_start, root_peer), -1)
    seed2_b = jnp.where(alive2, jnp.where(lower_mine, root_peer, s_start), -1)
    return LetBlocks(
        left=final_link(_COL_LEFT),
        right=final_link(_COL_RIGHT),
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
        coarse_needed=jnp.max(jnp.stack(rounds_needed)),
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
