"""The LET (locally essential tree) exchange of the cross summaries (track M4).

The default exchange all_gathers every device's whole two-sided summary tree
(:func:`jaccpot.distributed.cross._summary_rows`), ``S`` rows per device, so every
device receives ``(ndev - 1) x S`` rows per force, a volume that grows with the
device count at fixed particles per device. The LET exchange sends a peer only the
part of the summary tree that the cross walk against that peer can open.

**What a peer receives.** Sender ``s`` decides, for every receiver ``d``, which of
its summary nodes the walk between the two trees could SPLIT (:func:`let_open`),
from a few numbers ``d`` publishes about its own tree (:func:`let_descriptor`: the
box of its node centres, its largest leaf radius and, under eq (16a), its masses,
thresholds and multipole powers binned by radius). A node goes to ``d`` when every
ancestor was opened for ``d`` (the root always), with the child links rewritten into
the block it travels in, so an unopened node arrives as a leaf. The blocks travel
in one ragged all-to-all (:func:`let_exchange`).

**Why momentum still cancels (M1).** Both ends of a cross pair must derive the
identical pair set from bitwise-identical data. Here both ends walk the SAME two
blocks: device ``s`` walks (its block for ``d``, the block ``d`` sent it), device
``d`` (the block ``s`` sent it, its block for ``s``) -- the same rows, bitwise, in
the same orientation (lower device first). So the pair sets agree by
construction, whatever :func:`let_open` decides.

**Why the pairs equal the all_gather path's.** The walk over the pruned trees
emits the full trees' pairs if it never splits a node that arrived as a leaf.
:func:`let_open` is a conservative bound of exactly that: a node is opened for
``d`` when SOME node of ``d`` could fail the MAC against it and be the side the walk
splits (yggdrax splits the larger radius, or the internal side against a leaf).
It is an over-estimate, and pinned against the all_gather path's pair sets in
``tests/distributed/test_mesh_let_exchange.py``.

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
    "LET_BINS",
    "LetBlocks",
    "let_descriptor",
    "let_bytes",
    "let_descriptor_width",
    "let_exchange",
    "let_open",
]

# two-sided summary row columns (``cross._summary_rows``)
_COL_RADIUS = 3
_COL_LEFT = 4
_COL_RIGHT = 5
_COL_ACTIVE = 6
_COL_COUNT = 7
_COL_TABLE = 8  # cross._SUMMARY_TABLE_COL: G M, threshold, s_1..s_p

#: radius bins of the descriptor: ``floor(log2 r) + _BIN_OFFSET``, clipped to
#: ``[0, LET_BINS)`` (single octaves from 2**-16 to 2**8; the extremes collect the
#: rest, which stays conservative: every bin carries its own largest radius and box)
LET_BINS = 24
_BIN_OFFSET = 16
#: per bin and kind: centre box lo (3), hi (3), largest radius, largest G M, least
#: threshold
_BIN_COLS = 9
#: relative slack of every bound (the walk's own distance and radius sums round)
_MARGIN = 1e-3
#: eq (15)'s improvement factor ``8 max(rho) / (rho_A + rho_B)`` is at most 8
_IMPROVEMENT_MAX = 8.0
_TINY = 1e-30


def let_descriptor_width(error_order: int) -> int:
    """Columns of :func:`let_descriptor`.

    Parameters
    ----------
    error_order : int
        ``0``: the geometric MAC; ``p``: eq (16a) at order ``p``.

    Returns
    -------
    int
        ``2 x LET_BINS x 9`` (per radius bin, for all nodes and for the leaves: the
        centre box, the largest radius, ``G M`` and the least threshold) plus the
        ``p`` largest normalised powers.
    """
    return 2 * LET_BINS * _BIN_COLS + int(error_order)


def _radius_bin(radius: Array) -> Array:
    r = jnp.maximum(radius, jnp.asarray(_TINY, radius.dtype))
    b = jnp.floor(jnp.log2(r)).astype(jnp.int32) + _BIN_OFFSET
    return jnp.clip(b, 0, LET_BINS - 1)


def _bin_table(rows: Array, member: Array, bins: Array, *, error_order: int) -> Array:
    """``(LET_BINS, 9)`` per radius bin of the member rows: box, radius, G M, threshold."""
    dtype = rows.dtype
    inf = jnp.asarray(jnp.inf, dtype)
    b = jnp.where(member, bins, LET_BINS)
    cen = rows[:, 0:3]
    lo = jnp.full((LET_BINS, 3), inf, dtype).at[b].min(cen, mode="drop")
    hi = jnp.full((LET_BINS, 3), -inf, dtype).at[b].max(cen, mode="drop")
    rmax = jnp.zeros((LET_BINS,), dtype).at[b].max(rows[:, _COL_RADIUS], mode="drop")
    if int(error_order) > 0:
        gm = jnp.zeros((LET_BINS,), dtype).at[b].max(rows[:, _COL_TABLE], mode="drop")
        thr = (
            jnp.full((LET_BINS,), inf, dtype)
            .at[b]
            .min(rows[:, _COL_TABLE + 1], mode="drop")
        )
    else:
        gm = jnp.zeros((LET_BINS,), dtype)
        thr = jnp.full((LET_BINS,), inf, dtype)
    return jnp.concatenate([lo, hi, rmax[:, None], gm[:, None], thr[:, None]], axis=1)


def let_descriptor(rows: Array, valid: Array, *, error_order: int = 0) -> Array:
    """What a device publishes about its summary tree for its peers' open tests.

    Its live nodes binned by radius (one octave per bin), twice -- all of them and
    the leaves alone -- and per bin the box of their centres, their largest radius
    and, under eq (16a), their largest ``G M`` and least threshold; then the largest
    normalised powers ``s_1..s_p`` of any live node. A peer bounds the distance to
    every node of a bin by the distance to the bin's box, so a few large leaves far
    out in a halo no longer pair their radius with the distance to the core.

    Parameters
    ----------
    rows : Array
        ``(S, k)`` this device's two-sided summary rows.
    valid : Array
        ``(S,)`` rows that hold a summary node.
    error_order : int
        ``0``: the geometric MAC; ``p``: eq (16a) at order ``p``. Static.

    Returns
    -------
    Array
        ``(let_descriptor_width(p),)``; an empty bin has the box ``+inf`` /
        ``-inf``.
    """
    dtype = rows.dtype
    live = valid & (rows[:, _COL_ACTIVE] > 0.5)
    leaf = rows[:, _COL_LEFT] < 0
    bins = _radius_bin(rows[:, _COL_RADIUS])
    p = int(error_order)
    tables = [
        _bin_table(rows, live, bins, error_order=p),
        _bin_table(rows, live & leaf, bins, error_order=p),
    ]
    cols = [t.reshape(-1) for t in tables]
    if p > 0:
        powers = rows[:, _COL_TABLE + 2 : _COL_TABLE + 2 + p]
        cols.append(jnp.max(jnp.where(live[:, None], powers, 0.0), axis=0))
    return jnp.concatenate(cols).astype(dtype)


def _eq16a_may_fail(
    ra: Array,
    gm_a: Array,
    thr_a: Array,
    s_a: list,
    dmin: Array,
    rb: Array,
    gm_b: Array,
    thr_b: Array,
    smax: Array,
    *,
    order: int,
    theta_max: float,
) -> Array:
    """Whether eq (16a) may refuse (A, B) for some B with radius at most ``rb``,
    ``G M`` at most ``gm_b``, threshold at least ``thr_b`` and powers at most
    ``smax``, at a centre distance of at least ``dmin`` (all broadcast).

    Every term of eq (15)/(16a) (:func:`yggdrax._dehnen_accept.dehnen_pair_accept`)
    grows with ``rho_B``, ``G M`` and the powers (all non-negative) and falls with
    ``r`` (a degree-``p`` polynomial in ``rho / r`` over ``r^2``); the improvement
    factor is at most 8. So the bound below is a supremum over such B.
    """
    p = int(order)
    r = jnp.maximum(dmin, jnp.asarray(_TINY, dmin.dtype))
    up = 1.0 + _MARGIN
    a = ra * up / r
    b = rb * up / r
    conv = (ra + rb) * up >= theta_max * dmin
    pa = [jnp.ones_like(a)]
    pb = [jnp.ones_like(b)]
    for _ in range(p):
        pa.append(pa[-1] * a)
        pb.append(pb[-1] * b)
    e_ab = pb[p]
    e_ba = pa[p]
    for n in range(1, p + 1):
        c = float(math.comb(p, n))
        e_ab = e_ab + c * s_a[n - 1] * pa[n] * pb[p - n]
        e_ba = e_ba + c * smax[n - 1] * pb[n] * pa[p - n]
    scale = _IMPROVEMENT_MAX / (r * r)
    down = 1.0 - _MARGIN
    ab = gm_a * e_ab * scale >= thr_b * down
    ba = gm_b * e_ba * scale >= thr_a * down
    return conv | ab | ba | (dmin <= 0.0)


def let_open(
    rows: Array,
    desc: Array,
    *,
    theta: float,
    separation_floor: float = 0.0,
    error_order: int = 0,
    theta_max: float = 1.0,
) -> Array:
    """The nodes the cross walk against the described device could split.

    yggdrax's mutual walk refines a refused pair by splitting the internal side,
    the larger radius when both are internal (``extent_target >= extent_source``
    splits the target). So node A of this device is split against B of the peer
    only if the MAC refuses (A, B) and B is a leaf, or ``rho_B <= rho_A``. The
    peer's nodes come in radius bins, each with the box of its centres, so for every
    bin ``r >= dmin`` (A's centre to the bin's box), and the MAC's refusal is tested
    at its supremum over the bin:

    * B a leaf: the leaves' bins, ``rho_B`` up to each bin's largest leaf radius
      (under eq (16a) with the bin's largest ``G M`` and least threshold);
    * ``rho_B <= rho_A``: all nodes' bins up to A's own, ``rho_B`` up to
      ``min(rho_A, the bin's largest radius)``.

    Parameters
    ----------
    rows : Array
        ``(S, k)`` this device's two-sided summary rows.
    desc : Array
        The peer's :func:`let_descriptor`.
    theta : float
        The geometric MAC's opening angle (``(rho_A + rho_B) <= theta r``).
    separation_floor : float
        The walks' separation floor (a far pair also needs
        ``rho_A + rho_B + floor <= r``).
    error_order : int
        ``0``: the geometric MAC; ``p``: eq (16a). Static.
    theta_max : float
        Eq (16a)'s convergence clause.

    Returns
    -------
    Array
        ``(S,)`` bool: open (send the children). Internal live nodes only.
    """
    p = int(error_order)
    dtype = rows.dtype
    cen = rows[:, 0:3]
    ra = rows[:, _COL_RADIUS][:, None]
    nb = LET_BINS * _BIN_COLS
    up = 1.0 + _MARGIN
    floor = float(separation_floor)
    if p > 0:
        gm_a = rows[:, _COL_TABLE][:, None]
        thr_a = rows[:, _COL_TABLE + 1][:, None]
        s_a = [rows[:, _COL_TABLE + 2 + n][:, None] for n in range(p)]
        smax = [desc[2 * nb + n] for n in range(p)]

    def may_fail(table: Array, rb: Array) -> Array:
        lo, hi = table[:, 0:3], table[:, 3:6]
        gap = jnp.maximum(
            jnp.maximum(lo[None, :, :] - cen[:, None, :], cen[:, None, :] - hi[None]),
            0.0,
        )
        dmin = jnp.sqrt(jnp.sum(gap * gap, axis=-1)) * (1.0 - _MARGIN)
        if p == 0:
            fail = (ra + rb) * up >= float(theta) * dmin
        else:
            fail = _eq16a_may_fail(
                ra,
                gm_a,
                thr_a,
                s_a,
                dmin,
                rb,
                table[None, :, 7],
                table[None, :, 8],
                smax,
                order=p,
                theta_max=float(theta_max),
            )
        if floor > 0.0:
            fail = fail | ((ra + rb + jnp.asarray(floor, dtype)) * up >= dmin)
        return fail

    every = desc[0:nb].reshape(LET_BINS, _BIN_COLS)
    leaves = desc[nb : 2 * nb].reshape(LET_BINS, _BIN_COLS)
    smaller = (
        jnp.arange(LET_BINS)[None, :] <= _radius_bin(rows[:, _COL_RADIUS])[:, None]
    )
    rb_small = jnp.minimum(ra, every[None, :, 6])
    fail = jnp.any(smaller & may_fail(every, rb_small), axis=1)
    fail = fail | jnp.any(may_fail(leaves, leaves[None, :, 6]), axis=1)
    internal = rows[:, _COL_LEFT] >= 0
    live = rows[:, _COL_ACTIVE] > 0.5
    return fail & internal & live


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
    table : Optional[Array]
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
        Total rows this device had to send / was sent (exact past the capacities).
    overflow : Array
        Either buffer overflowed (the walk then reads a truncated tree).
    send_capacity : int
        ``T``. Static.
    """

    left: Array
    right: Array
    centers: Array
    radii: Array
    active: Array
    table: Optional[Array]
    count: Array
    orig: Array
    peer: Array
    seed_a: Array
    seed_b: Array
    send_sizes: Array
    recv_sizes: Array
    send_needed: Array
    recv_needed: Array
    overflow: Array
    send_capacity: int


def _block_slots(sizes: Array, capacity: int) -> tuple[Array, Array, Array]:
    """Per slot of a buffer laid out block by block: its block, whether it is live,
    and every block's start."""
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


def let_exchange(
    rows: Array,
    valid: Array,
    me: Array,
    *,
    ndev: int,
    send_capacity: int,
    recv_capacity: int,
    axis_name: str,
    theta: float,
    separation_floor: float = 0.0,
    error_order: int = 0,
    theta_max: float = 1.0,
    open_fn: Optional[Callable[..., Array]] = None,
) -> LetBlocks:
    """Send every peer the part of this device's summary tree its walk can open.

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
    axis_name : str
        Mesh axis.
    theta, separation_floor, error_order, theta_max
        The cross walk's MAC (:func:`let_open`).
    open_fn : Optional[Callable[..., Array]]
        Replaces :func:`let_open` (``(rows, desc) -> (S,) bool``): a test seam
        (opening everything gives the all_gather path's trees).

    Returns
    -------
    LetBlocks
        The blocks and the walk table over them.
    """
    idx = jnp.int32
    S, k = int(rows.shape[0]), int(rows.shape[1])
    T, R = int(send_capacity), int(recv_capacity)
    dtype = rows.dtype
    p = int(error_order)
    me = jnp.asarray(me).astype(idx)

    desc = jax.named_call(let_descriptor, name="cross_let_descriptor")(
        rows, valid, error_order=p
    )
    all_desc = jax.lax.all_gather(desc, axis_name, tiled=False)

    left = jnp.round(rows[:, _COL_LEFT]).astype(idx)
    right = jnp.round(rows[:, _COL_RIGHT]).astype(idx)
    internal = valid & (left >= 0)
    sentinel = jnp.asarray(S, idx)
    parent = (
        jnp.full((S,), -1, idx)
        .at[jnp.where(internal, left, sentinel)]
        .set(jnp.arange(S, dtype=idx), mode="drop")
        .at[jnp.where(internal, right, sentinel)]
        .set(jnp.arange(S, dtype=idx), mode="drop")
    )
    has_parent = parent >= 0
    anc0 = jnp.where(has_parent, parent, jnp.arange(S, dtype=idx))
    rounds = max(1, int(math.ceil(math.log2(max(S, 2)))))
    orig_col = jnp.arange(S, dtype=idx).astype(dtype)[:, None]

    def opened(d_desc: Array) -> Array:
        if open_fn is not None:
            return jnp.asarray(open_fn(rows, d_desc), bool) & internal
        return let_open(
            rows,
            d_desc,
            theta=theta,
            separation_floor=separation_floor,
            error_order=p,
            theta_max=theta_max,
        )

    def per_peer(d: Array, carry: tuple) -> tuple:
        buf, offset, sizes = carry
        op = opened(all_desc[d]) & internal
        # a node travels when every ancestor was opened: pointer doubling over
        # "my parent was not opened" up the path to the root
        blocked = has_parent & ~op[jnp.maximum(parent, 0)]
        anc = anc0
        for _ in range(rounds):
            blocked = blocked | blocked[anc]
            anc = anc[anc]
        sent = valid & ~blocked & (d != me)
        pos = jnp.cumsum(sent, dtype=idx) - 1
        size = jnp.sum(sent, dtype=idx)
        go = op & sent
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

    buf0 = jnp.zeros((T, k + 1), dtype)
    with jax.named_scope("cross_let_pack"):
        buf, send_needed, send_sizes = jax.lax.fori_loop(
            0,
            int(ndev),
            per_peer,
            (buf0, jnp.asarray(0, idx), jnp.zeros((int(ndev),), idx)),
        )
    recv, recv_sizes, _ = jax.named_call(
        ragged_all_to_all_exchange, name="cross_exchange_let"
    )(buf, send_sizes, output_capacity=R, axis_name=axis_name)
    recv_sizes = recv_sizes.astype(idx)
    recv_needed = jnp.sum(recv_sizes, dtype=idx)
    overflow = (send_needed > T) | (recv_needed > R)

    # ---- the walk table over [send | recv] ------------------------------------
    s_peer, s_live, s_start = _block_slots(send_sizes, T)
    r_peer, r_live, r_start = _block_slots(recv_sizes, R)
    both = jnp.concatenate([buf, recv], axis=0)
    live = jnp.concatenate([s_live, r_live])
    peer = jnp.concatenate([s_peer, r_peer])
    base = jnp.concatenate(
        [
            s_start[jnp.minimum(s_peer, ndev - 1)],
            T + r_start[jnp.minimum(r_peer, ndev - 1)],
        ]
    )

    def link(col: int) -> Array:
        local = jnp.round(both[:, col]).astype(idx)
        out = jnp.where(live & (local >= 0), base + local, -1)
        return jnp.where(out < T + R, out, -1)

    orig = jnp.where(live, jnp.round(both[:, k]).astype(idx), -1)
    count = jnp.where(live, jnp.round(both[:, _COL_COUNT]).astype(idx), 0)
    width = p + 2
    table = both[:, _COL_TABLE : _COL_TABLE + width] if p > 0 else None

    dev = jnp.arange(int(ndev), dtype=idx)
    root_mine = s_start
    root_peer = T + r_start
    alive = (dev != me) & (send_sizes > 0) & (recv_sizes > 0)
    alive = alive & (s_start < T) & (r_start < R)
    lower_mine = me < dev
    seed_a = jnp.where(alive, jnp.where(lower_mine, root_mine, root_peer), -1)
    seed_b = jnp.where(alive, jnp.where(lower_mine, root_peer, root_mine), -1)
    return LetBlocks(
        left=link(_COL_LEFT),
        right=link(_COL_RIGHT),
        centers=both[:, 0:3],
        radii=both[:, _COL_RADIUS],
        active=live & (both[:, _COL_ACTIVE] > 0.5),
        table=table,
        count=count,
        orig=orig,
        peer=jnp.where(live, peer, int(ndev)),
        seed_a=seed_a.astype(idx),
        seed_b=seed_b.astype(idx),
        send_sizes=send_sizes,
        recv_sizes=recv_sizes,
        send_needed=send_needed,
        recv_needed=recv_needed,
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
