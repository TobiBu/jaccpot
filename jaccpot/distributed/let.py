"""The LET (locally essential tree) exchange of the cross summaries (track M4).

The default exchange all_gathers every device's whole two-sided summary tree
(:func:`jaccpot.distributed.cross._summary_rows`), ``S`` rows per device, so every
device receives ``(ndev - 1) x S`` rows per force, a volume that grows with the
device count at fixed particles per device. The LET exchange sends a peer only the
part of the tree that the cross walk against that peer opens, found EXACTLY by the
walk itself, in request rounds (:func:`let_exchange`):

1. **Round r** (sizes ``coarse_particles``, live particles, decreasing): each
   device sends every peer the nodes whose parent is EXPANDED for that peer --
   above the FIRST size, requested by that peer, or inside a subtree it requested
   in round ``j < r`` and above round ``j + 1``'s size (whole when that subtree is
   already within it). Finer sizes apply only inside requested subtrees, so a far
   peer gets the first round's top alone. Expanded is closed upward, so every
   ancestor of a sent node is expanded too. A sent internal node that is not
   expanded is a CUT node.
2. **Request walk.** Each device runs the cross walk between its FULL tree and every
   block it received, in the walk's own orientation (lower device first), with
   every cut node given two children: a PROBE (a leaf copy of itself) and a dead
   slot. Above the cuts the walk is the full walk on the same floats, so it splits
   a cut node exactly when the full walk would, and every split leaves an output
   pair holding the probe (accepted, near, or under the other side's split). A
   probe in an output pair is a request; the requests go back to the senders in a
   ragged all-to-all aligned with their send buffers.
3. **Final blocks.** The same rule after the last round, whose requests go whole,
   in one ragged all-to-all.

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
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array
from yggdrax.distributed.comm import ragged_all_to_all_exchange

from jaccpot._searchsorted import searchsorted_method

__all__ = [
    "LetBlocks",
    "device_order_walk",
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
    walk_split : Array
        Received rows from devices below this one (:func:`device_order_walk`).
    far_peers : Any
        ``(ndev,)`` bool: the peers that got the far first round.
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
    walk_split: Array
    far_peers: Any = None


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


def device_order_walk(
    walk: Callable[..., Any],
    left: Array,
    right: Array,
    centers: Array,
    radii: Array,
    active: Array,
    table: Any,
    seed_a: Array,
    seed_b: Array,
    *,
    n_mine: int,
    n_recv: int,
    split: Array,
) -> Any:
    """Run ``walk`` with this device's rows between the peers' in device order.

    The tables here are laid out ``[mine (n_mine) | received (n_recv) | tail]``,
    the received rows in device order. The walk breaks a radius tie by INDEX order
    (yggdrax's walks queue every child pair as ``(lo, hi)``), and the all_gather
    path's index order is the device order, so both ends of a cross pair must see
    the lower device's rows at the lower indices: here the received rows of devices
    below this one (the first ``split``), then this device's, then the rest. The
    tail (probes, a dead row) keeps its indices. The pairs come back in the
    original layout.

    Parameters
    ----------
    walk : Callable[..., Any]
        ``walk(left, right, centers, radii, active, table, seed_a, seed_b)``.
    left, right, centers, radii, active, table, seed_a, seed_b
        The table in the original layout (links and seeds as original indices).
    n_mine, n_recv : int
        The two segments' lengths. Static.
    split : Array
        Received rows from devices below this one.

    Returns
    -------
    Any
        The walk result with its pair ends in the original layout.
    """
    idx = jnp.int32
    head = int(n_mine) + int(n_recv)
    total = int(jnp.asarray(left).shape[0])
    split = jnp.asarray(split).astype(idx)
    q_mine = jnp.arange(int(n_mine), dtype=idx)
    q_recv = jnp.arange(int(n_recv), dtype=idx)
    new_of_old = jnp.concatenate(
        [
            split + q_mine,
            jnp.where(q_recv < split, q_recv, q_recv + int(n_mine)),
            jnp.arange(head, total, dtype=idx),
        ]
    )
    old_of_new = (
        jnp.zeros((total,), idx).at[new_of_old].set(jnp.arange(total, dtype=idx))
    )

    def to_new(i: Array) -> Array:
        i = jnp.asarray(i).astype(idx)
        return jnp.where(i >= 0, new_of_old[jnp.clip(i, 0, total - 1)], i)

    def permute(x: Any) -> Any:
        return None if x is None else jnp.asarray(x)[old_of_new]

    res = walk(
        to_new(permute(left)),
        to_new(permute(right)),
        permute(centers),
        permute(radii),
        permute(active),
        permute(table),
        to_new(seed_a),
        to_new(seed_b),
    )

    def to_old(j: Array) -> Array:
        j = jnp.asarray(j)
        ok = (j >= 0) & (j < total)
        return jnp.where(ok, old_of_new[jnp.clip(j, 0, total - 1)], j).astype(j.dtype)

    return res._replace(
        far_a=to_old(res.far_a),
        far_b=to_old(res.far_b),
        near_a=to_old(res.near_a),
        near_b=to_old(res.near_b),
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
    far_particles: Optional[int] = None,
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
    far_particles : Optional[int]
        The first round's size for a FAR peer (its root sphere and this device's
        do not overlap: centres farther apart than the two root radii), coarser
        than ``coarse_particles[0]``: a far peer's walk opens little, and every
        peer gets the first round's top, so this is what keeps the rows per
        device from growing with the device count. The later rounds are the
        same. ``None``: every peer gets ``coarse_particles[0]``. Static.

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
    # tag[d, x] = r + 1: peer d's walk splits my node x, a cut node of round r it
    # requested (0: not requested)
    tag = jnp.zeros((nd, S), jnp.int8)
    lost = jnp.asarray(False)
    rounds_needed = []
    requested = jnp.zeros((nd,), idx)
    req_walk = None
    # each node's cut ancestor at every round's size: the node of that round's
    # top (the root and the children of nodes above the size) over it, itself in
    # the top; one pointer doubling per size, the same for every peer
    doubling = max(1, int(math.ceil(math.log2(max(S, 2)))))

    def top_ancestor(m: int) -> Array:
        big_m = internal & (count > m)
        in_top = valid & ((parent < 0) | big_m[jnp.maximum(parent, 0)])
        jump = jnp.where(in_top | (parent < 0), at, parent)
        for _ in range(doubling):
            jump = jump[jump]
        return jump

    cut_anc = [top_ancestor(m) for m in levels]
    first_near = internal & (count > levels[0])
    if far_particles is None:
        is_far = jnp.zeros((nd,), bool)
        first_far, anc_far = first_near, cut_anc[0]
    else:
        if int(far_particles) <= levels[0]:
            raise ValueError(
                f"far_particles={far_particles} must exceed the first size {levels[0]}"
            )
        # far peers: the root spheres (row 0: centre, radius) do not overlap
        root = jnp.where(valid[0], rows[0, 0:4], 0.0)
        roots = jax.lax.all_gather(root, axis_name, tiled=False)
        gap = jnp.sqrt(jnp.sum((roots[:, 0:3] - root[None, 0:3]) ** 2, axis=1))
        is_far = gap > roots[:, 3] + root[3]
        first_far = internal & (count > int(far_particles))
        anc_far = top_ancestor(int(far_particles))

    def expanded_after(d: Array, done: int, tag: Array) -> Array:
        """Nodes whose children peer ``d`` gets after ``done`` request rounds: above
        the first size, requested, or inside a subtree requested in round ``j``
        and above round ``j + 1``'s size (the whole subtree after the last round,
        or when it already holds no more than that size). Closed upward."""
        t = tag[d]
        far = is_far[d]
        out = jnp.where(far, first_far, first_near) | (t > 0)
        for j in range(done):
            # a far peer's first round cut at its own size (its cut nodes are
            # that top's nodes)
            anc = jnp.where(far, anc_far, cut_anc[0]) if j == 0 else cut_anc[j]
            inside = t[anc] == j + 1
            if j + 1 < len(levels):
                # down to the next size; whole when the requested subtree is
                # already within it. Every cut node of a round is then a node of
                # that round's top, so cut_anc finds the requester of every node
                # inside a requested subtree
                small = count[anc] <= levels[j + 1]
                inside = inside & ((count > levels[j + 1]) | small)
            out = out | (inside & internal)
        return out

    for r, m in enumerate(levels):
        with jax.named_scope(f"cross_let_round{r}"):

            def expanded(d: Array, r: int = r, tag: Array = tag) -> Array:
                return expanded_after(d, r, tag)

            def no_subtree(d: Array) -> Array:
                return jnp.zeros((S,), bool)

            def cut_col(d: Array, sent: Array, r: int = r, tag: Array = tag) -> Array:
                return (sent & internal & ~expanded_after(d, r, tag))[:, None]

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
            req_walk = device_order_walk(
                walk,
                q_left,
                q_right,
                q_rows[:, 0:3],
                q_rows[:, _COL_RADIUS],
                q_active,
                q_rows[:, _COL_TABLE : _COL_TABLE + width] if p > 0 else None,
                seed_a.astype(idx),
                seed_b.astype(idx),
                n_mine=S,
                n_recv=R,
                split=r_start[me],
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
            tag = tag.at[jnp.where(asked, s_peer, nd), jnp.where(asked, node, 0)].set(
                jnp.int8(r + 1), mode="drop"
            )
            requested = requested + jax.ops.segment_sum(
                asked.astype(idx), jnp.where(asked, s_peer, nd), nd + 1
            )[:nd].astype(idx)

    # ---- the final blocks: the last round's, the last requests' subtrees whole --
    with jax.named_scope("cross_let_pack"):

        def expanded_final(d: Array) -> Array:
            return expanded_after(d, len(levels), tag)

        def whole(d: Array) -> Array:
            return jnp.zeros((S,), bool)

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
        walk_split=r_start[me],
        far_peers=is_far,
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
