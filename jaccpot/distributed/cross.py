"""The cross-domain hook: what a device imports, assembled between the sweeps.

Plan ``~/.claude/plans/phase-c-interleaved-cross-field.md``. This is the callable
handed to :func:`~jaccpot.distributed.fused.fused_force_step` as ``cross_hook``. It
runs at the one point where the exchange belongs -- the multipoles exist and the
downward sweep has not consumed them -- and returns the imported far sources for the
M2L to concatenate, so the L2L cascade still runs once.

Every piece it calls is built and tested in yggdrax: ``occupancy_cut`` (the summary),
``export_walk`` (what this device owes everyone, in one walk), ``build_send_buffers``
(grouped and deduplicated), ``exchange_export_list`` (two ragged rounds), and
``receiver_interaction_lists`` (the imported per-cell lists expanded onto local
targets).

**The index convention is load-bearing.** Imported sources are returned to sit at
``[n_local, n_local + n_import)`` in the concatenated multipole array. Every local
index is then strictly below every imported one, which is what makes the walk's
``(min, max)`` canonicalisation an exact (local target, imported source) ordering.
Reversed, the M2L expands the wrong way round with a plausible-looking result.

**At ndev = 1 this returns nothing**, because a device never exports to itself -- which
is what makes it testable: the whole pipeline runs and the force must not move.

**The FAR half is interleaved; the NEAR half is added afterwards, and that asymmetry is
deliberate.** A far contribution arrives as a local EXPANSION, so adding it after the
force would need a second L2L cascade down to the leaves -- which is the whole reason
the far exchange sits between the sweeps. A near contribution is a direct sum evaluated
at the particles, with no cascade behind it, so computing it as its own small kernel and
adding the result costs one launch over a pool that Phase 3.1 measured at 1.4-4.3 % of
the local near field. Gravity is linear in the sources, so the sum is exact either way;
only the cost differs, and for the near half it differs the other way.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from yggdrax.distributed.comm import AXIS_NAME, ragged_all_to_all_exchange
from yggdrax.distributed.export import (
    ExportLists,
    build_send_buffers,
    export_walk,
    export_walk_two_sided,
)
from yggdrax.distributed.import_cells import (
    exchange_export_list,
    receiver_interaction_lists,
)
from yggdrax.distributed.summary import (
    occupancy_cut,
    subtree_leaf_counts,
    summary_tree,
)

from jaccpot._searchsorted import searchsorted_method

__all__ = [
    "CROSS_NEEDS_FIELDS",
    "CrossCapacities",
    "cross_near_acceleration",
    "cross_needs",
    "cross_walk_backend",
    "cross_walk_fn",
    "make_cross_hook",
    "merge_imported_blocks",
    "plan_cross_capacities",
]

#: Entries of the cross NEEDS vector (:func:`cross_needs`), in order. The first ten
#: are what the symmetric exchange (the default path) needed of the
#: :class:`CrossCapacities` attribute of the SAME NAME, on this device:
#:
#: * ``max_cells``, ``max_summary_nodes``: the summary cut's cells and the published
#:   summary tree's nodes. EXACT, past their own overflow too: recounted from the tree
#:   (:func:`_summary_needs`), not read off the cut, which saturates at its capacity.
#: * ``export_far_cap``, ``export_near_cap``, ``export_walk_queue``: the symmetric
#:   walk's far and near pairs and its peak wavefront. The Pallas walk keeps counting
#:   past a full list, so these are exact unless its queue overflowed; yggdrax's traced
#:   walk stops at its first overflow. Either way a lower bound when
#:   ``walk_lower_bound`` is set.
#: * ``send_node_cap``, ``recv_node_cap``, ``send_particle_cap``, ``recv_particle_cap``,
#:   ``leaf_width``: the rows (far and near; the larger) and particles this device
#:   ships and receives, and the most particles in one shipped leaf. Counted on the
#:   (peer, node) presence map, so EXACT past their own buffers' overflow; a lower
#:   bound when ``pairs_lower_bound`` is set (the walk or the summary lost pairs).
#:
#: Then four bits: ``overflow`` (the hook's own flag fired on this device),
#: ``walk_lower_bound``, ``pairs_lower_bound``, and ``mismatch`` (the two ends of an
#: exchange disagreed on a size while none of this device's exchange buffers
#: overflowed: no capacity covers that, and no re-plan fixes it).
#: Reduced over the mesh with ``pmax`` the vector is what every capacity, shared by
#: all devices, has to hold, and the bits are OR-ed.
CROSS_NEEDS_FIELDS = (
    "max_cells",
    "max_summary_nodes",
    "export_far_cap",
    "export_near_cap",
    "export_walk_queue",
    "send_node_cap",
    "recv_node_cap",
    "send_particle_cap",
    "recv_particle_cap",
    "leaf_width",
    "overflow",
    "walk_lower_bound",
    "pairs_lower_bound",
    "mismatch",
)
#: the capacity entries of :data:`CROSS_NEEDS_FIELDS`, and the lower-bound bit (if
#: any) that qualifies each
_NEEDS_LOWER_BOUND_BIT = {
    "max_cells": None,
    "max_summary_nodes": None,
    "export_far_cap": "walk_lower_bound",
    "export_near_cap": "walk_lower_bound",
    "export_walk_queue": "walk_lower_bound",
    "send_node_cap": "pairs_lower_bound",
    "recv_node_cap": "pairs_lower_bound",
    "send_particle_cap": "pairs_lower_bound",
    "recv_particle_cap": "pairs_lower_bound",
    "leaf_width": "pairs_lower_bound",
}


_CROSS_MAC_GEOMETRY_ENV = "JACCPOT_CROSS_MAC_GEOMETRY"


def _cross_mac_geometry_mode() -> str:
    """``"com"`` (default: the geometry the lane's own walk uses) or ``"aabb"`` (the
    box geometry this exchange used before Task 1; kept as a control)."""
    from jaccpot._env import env_choice

    # a malformed value warns and keeps the correct default, as every jaccpot switch does
    return env_choice(_CROSS_MAC_GEOMETRY_ENV, "com", ("com", "aabb"))


def _node_cell_edge(tree: Any) -> Array:
    """Per node, the edge (box units, ``2**-depth``) of the smallest Morton cell
    holding all of its particles.

    From the common prefix of the codes at the two ends of the node's particle range:
    63-bit codes, 3 bits per level. Monotone down the tree (a child's range lies inside
    its parent's), which is what :func:`occupancy_cut`'s size bound needs.

    Parameters
    ----------
    tree : Any
        The refreshed tree; its ``morton_codes`` are in sorted order, like
        ``node_ranges``.

    Returns
    -------
    Array
        ``(total_nodes,)`` float edges; empty nodes get an arbitrary value (they hold
        no leaves and never enter the cut).
    """
    codes = jnp.asarray(tree.morton_codes).astype(jnp.uint64)
    nr = jnp.asarray(tree.node_ranges)
    n = int(codes.shape[0])
    lo = codes[jnp.clip(nr[:, 0], 0, n - 1)]
    hi = codes[jnp.clip(nr[:, 1], 0, n - 1)]
    clz = jax.lax.clz(jnp.bitwise_xor(lo, hi)).astype(jnp.int32)  # 64 when equal
    depth = jnp.minimum((clz - 1) // 3, 21)
    return jnp.exp2(-depth.astype(jnp.float32))


def _near_tiles_on_the_wire() -> bool:
    """``JACCPOT_CROSS_NEAR_TILES=1``: ship W-wide particle tiles (the old format).

    Returns
    -------
    bool
        Whether to use the tile format; the default is the compact one.
    """
    from jaccpot._env import env_flag

    return env_flag("JACCPOT_CROSS_NEAR_TILES", False)


def _cross_two_sided() -> bool:
    """Export against every receiver's summary TREE (default on).

    ``JACCPOT_CROSS_TWO_SIDED=0`` keeps the one-sided export as the control.

    The one-sided export refines only the sender, so each receiver cell collects a
    treecode-style list of sender nodes: 5.1M far pairs per device at 1e6 particles
    per A100, against 0.46M two-sided (the same 8.7 per leaf as at 1e5, i.e. O(N)).
    The two-sided walk splits whichever side is larger, so far pairs can land on the
    receiver's internal nodes and its L2L cascade carries them down. Measured on two
    A100s, Plummer, p6: 2e6 71.2 -> 64.1 ms, 8e6 229.5 -> 202.3 ms, forces in class.

    Returns
    -------
    bool
        Whether to publish the summary tree and walk both sides.
    """
    from jaccpot._env import env_flag

    return env_flag("JACCPOT_CROSS_TWO_SIDED", True)


def _max_leaves_per_cell(cap: "CrossCapacities", two_sided: bool) -> int:
    """The summary cut's occupancy bound: the caller's, or the measured best per mode.

    Two-sided, 1: the summary tree is the receiver's whole tree above its leaves, so the
    sender decides "needs particles" per receiver LEAF. With cells of up to 4 leaves a
    small outskirt cell next to the remote core was near 48k sender leaves at 1e6 per
    device; its leaves then received them all as M2L sources in one CSR row (the M2L
    grid is one program per target) and the near import shipped ~2x the particles. At
    2e6 on two A100s: 61.3 -> 57.3 ms, the receiver near walk a pass-through. One-sided,
    4: per-leaf cells there would give every leaf its own treecode list.

    Parameters
    ----------
    cap : CrossCapacities
        Capacities; an explicit ``max_leaves_per_cell`` wins.
    two_sided : bool
        The export mode.

    Returns
    -------
    int
        Leaves per summary cell at most.
    """
    if cap.max_leaves_per_cell is not None:
        return int(cap.max_leaves_per_cell)
    return 1 if two_sided else 4


class _Published(NamedTuple):
    """What one device publishes for the export walks, and how to read it back.

    Attributes
    ----------
    cut : Any
        The occupancy cut (``TreeSummary``).
    block_nodes : Array
        ``(B,)`` this device's own node behind every published index, ``-1`` in the
        padding: the cut cells (one-sided) or the summary-tree nodes (two-sided). A
        received CSR entry names a published index; this maps it to a target.
    rows : Array
        ``(B, k)`` the packed rows every device all_gathers: centre 3, radius, then
        (two-sided) left and right child in the summary index space, active, and
        the live particle count under the entry. Indices and counts travel as
        floats -- exact below 2**24 in fp32.
    overflow : Array
        The cut's overflow, or the summary tree's (which includes the cut's).
    num_entries : Array
        Live published entries.
    """

    cut: Any
    block_nodes: Array
    rows: Array
    overflow: Array
    num_entries: Array


#: column of the first eq (16a) table entry in a two-sided summary row
#: (:func:`_summary_rows`): centre 3, radius, left, right, active, count, then the
#: table's ``G M``, threshold, ``s_1..s_p``
_SUMMARY_TABLE_COL = 8


def _summary_table_width(order: int) -> int:
    """Table columns a summary row carries under the criterion: ``G M``, the
    threshold and ``s_1..s_p`` (:func:`~jaccpot.runtime._walk_criterion.dehnen_walk_table`
    without its power-of-two padding).

    Parameters
    ----------
    order : int
        Expansion order ``p``.

    Returns
    -------
    int
        ``p + 2``.
    """
    from jaccpot.runtime._walk_criterion import WALK_TABLE_POWER0

    return WALK_TABLE_POWER0 + int(order)


def _live_node_ends(tree: Any) -> Array:
    """Each node's last LIVE particle (inclusive): its range end, clamped.

    A capacity-padded cells tree (a mesh shard, dead rows sorted to the end) gives
    its padding leaves the empty range ``(n, n - 1)`` and an internal node the
    (min start, max end) of its children, so the parent of the last live leaf and
    the padding subtree spans ``[that leaf's start, n - 1]``: the leaf's particles
    AND every dead row. Under one-leaf summary cells without ``summary_cell_level``
    that parent is a cut cell (it holds one live leaf), and laying its near
    particles out over its range shipped the dead rows -- zero mass, so the force
    was right, but the near half's ``full > W`` check fired whatever the width
    (two A100s, the rollout gate's capacities, 2e5 per card). Live particles are
    exactly the leaves' ranges, so the last of them bounds every node.

    Parameters
    ----------
    tree : Any
        The refreshed tree (``node_ranges``, ``left_child``).

    Returns
    -------
    Array
        ``(total_nodes,)`` inclusive ends; an empty node keeps ``end < start``.
    """
    nr = jnp.asarray(tree.node_ranges)
    leaves = nr[int(jnp.asarray(tree.left_child).shape[0]) :]
    last_live = jnp.max(jnp.where(leaves[:, 1] >= leaves[:, 0], leaves[:, 1], -1))
    return jnp.minimum(nr[:, 1], last_live)


def _summary_rows(
    tree: Any,
    geom: Any,
    cap: "CrossCapacities",
    *,
    two_sided: bool,
    criterion: Any = None,
) -> _Published:
    """This device's summary, packed for ONE all_gather.

    Parameters
    ----------
    tree : Any
        The refreshed tree (``parent``, ``left_child``, ``right_child``,
        ``node_ranges``, and ``morton_codes`` when ``cap.summary_cell_level`` is set).
    geom : Any
        The walk geometry (``center``, ``radius`` per node) the sender's MAC uses.
    cap : CrossCapacities
        Capacities; ``max_cells`` (one-sided) or ``max_summary_nodes`` (two-sided)
        is the block width.
    two_sided : bool
        Publish the summary tree (with child links) instead of the bare cells.
    criterion : Any
        ``mac_type='dehnen_error'`` (two-sided only): this step's
        :class:`~jaccpot.runtime._walk_criterion.FlatWalkCriterion` over the local
        nodes. Each published node's eq (16a) table row -- ``G M``, the threshold
        ``eps * min_b f_b`` of its subtree, ``s_1..s_p`` on the PUBLISHED radius,
        the one the cross walks read -- rides behind the geometry
        (:data:`_SUMMARY_TABLE_COL`), so every device evaluates every cross pair
        from the same floats. ``None``: the geometric rows.

    Returns
    -------
    _Published
        The packed rows and what the receiver needs to read its CSR back.
    """
    parent = jnp.asarray(tree.parent)
    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    size_bound = (
        {}
        if cap.summary_cell_level is None
        else dict(
            node_extent=_node_cell_edge(tree),
            max_extent=2.0 ** -int(cap.summary_cell_level),
        )
    )
    cut = jax.named_call(occupancy_cut, name="cross_summary")(
        parent,
        jnp.asarray(tree.node_ranges),
        num_internal,
        max_leaves=_max_leaves_per_cell(cap, two_sided),
        capacity=cap.max_cells,
        **size_bound,
    )
    center = jnp.asarray(geom.center)
    radius = jnp.asarray(geom.radius)
    dtype = center.dtype
    if not two_sided:
        live = jnp.arange(cap.max_cells) < cut.num_cells
        safe = jnp.where(live, cut.cells, 0)
        rows = jnp.concatenate(
            [
                jnp.where(live[:, None], center[safe], 0.0),
                jnp.where(live, radius[safe], 0.0)[:, None],
                live.astype(dtype)[:, None],
            ],
            axis=1,
        )
        return _Published(cut, cut.cells, rows, cut.overflow, cut.num_cells)

    S = int(cap.max_summary_nodes)
    if jnp.finfo(dtype).nmant < 52 and S >= (1 << (jnp.finfo(dtype).nmant + 1)):
        raise ValueError(
            f"max_summary_nodes={S} does not travel exactly as {dtype} indices"
        )
    st = jax.named_call(summary_tree, name="cross_summary_tree")(
        parent,
        jnp.asarray(tree.left_child),
        jnp.asarray(tree.right_child),
        jnp.asarray(tree.node_ranges),
        num_internal,
        cut,
        capacity=S,
    )
    ok = st.nodes >= 0
    safe = jnp.where(ok, st.nodes, 0)
    nr = jnp.asarray(tree.node_ranges)
    # live particles under each entry: what a peer receiving this node's particles
    # needs to lay them out (`_symmetric_exchange`), so no row counts travel. Not the
    # range length: a padded shard's ranges can run into its dead rows
    ends = _live_node_ends(tree)
    count = jnp.where(st.active, jnp.maximum(ends[safe] - nr[safe, 0] + 1, 0), 0)
    columns = [
        jnp.where(ok[:, None], center[safe], 0.0),
        jnp.where(ok, radius[safe], 0.0)[:, None],
        st.left.astype(dtype)[:, None],
        st.right.astype(dtype)[:, None],
        st.active.astype(dtype)[:, None],
        count.astype(dtype)[:, None],
    ]
    if criterion is not None:
        # the published nodes' table rows, built on the radius published above
        published = criterion._replace(
            multipole_packed=jnp.asarray(criterion.multipole_packed)[safe],
            mass=jnp.asarray(criterion.mass)[safe],
            threshold=jnp.asarray(criterion.threshold)[safe],
        )
        width = _summary_table_width(int(criterion.order))
        table = jnp.asarray(published.table(radius[safe])[:, :width], dtype)
        columns.append(jnp.where(ok[:, None], table, 0.0))
    rows = jnp.concatenate(columns, axis=1)
    return _Published(cut, st.nodes, rows, st.overflow, st.num_nodes)


def _summary_needs(
    tree: Any, cap: "CrossCapacities", *, two_sided: bool
) -> tuple[Array, Array]:
    """What :func:`_summary_rows` needs of ``max_cells`` and ``max_summary_nodes``.

    ``occupancy_cut`` and ``summary_tree`` report their sizes saturated at their
    capacities, so a re-plan cannot read the demand off them. This recounts both from
    the cut's own rule: a node FITS when it holds 1..``max_leaves`` live leaves and
    (internal nodes only, with ``summary_cell_level``) lies inside one Morton cell of
    that level; the cut is every fitting node whose parent does not fit. Fitting is
    monotone up the tree for live nodes, so the summary tree's ANCESTORS -- live,
    internal, not in or below the cut -- are exactly the live internal nodes that do
    not fit, and the summary tree is the root plus both children of every ancestor:
    ``2 A + 1`` nodes (``0`` for a shard without a live leaf). Exact whatever the
    capacities; equal to ``num_cells`` / ``num_nodes`` when nothing overflowed (pinned
    in ``tests/unit/distributed/test_cross_capacities.py``).

    Parameters
    ----------
    tree : Any
        The refreshed tree, as :func:`_summary_rows` reads it.
    cap : CrossCapacities
        Capacities; ``max_leaves_per_cell`` and ``summary_cell_level`` are read.
    two_sided : bool
        The export mode (decides the default ``max_leaves``).

    Returns
    -------
    tuple[Array, Array]
        ``(cells, summary_nodes)``, int32 scalars. One-sided, the second is the cells
        again (the published block is the cut itself).
    """
    parent = jnp.asarray(tree.parent)
    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    sub = subtree_leaf_counts(jnp.asarray(tree.node_ranges), num_internal)
    internal = jnp.arange(parent.shape[0]) < num_internal
    fits = (sub > 0) & (sub <= _max_leaves_per_cell(cap, two_sided))
    if cap.summary_cell_level is not None:
        edge = _node_cell_edge(tree)
        fits = fits & (~internal | (edge <= 2.0 ** -int(cap.summary_cell_level)))
    is_root = parent < 0
    in_cut = fits & jnp.where(is_root, True, ~fits[jnp.where(is_root, 0, parent)])
    cells = jnp.sum(in_cut, dtype=jnp.int32)
    if not two_sided:
        return cells, cells
    ancestors = jnp.sum(internal & (sub > 0) & ~fits, dtype=jnp.int32)
    live_root = sub[jnp.argmin(parent)] > 0
    return cells, jnp.where(live_root, 2 * ancestors + 1, 0).astype(jnp.int32)


def _export_from_rows(
    gathered: Array,
    left: Array,
    right: Array,
    geom: Any,
    root: Array,
    theta: float,
    me: Array,
    cap: "CrossCapacities",
    *,
    two_sided: bool,
    mac_type: str,
    walk_fn: Optional[Callable[..., Any]],
    node_active: Optional[Array] = None,
) -> ExportLists:
    """Unpack every device's published rows and run the export walk against them.

    Parameters
    ----------
    gathered : Array
        ``(ndev, B, k)`` every device's :attr:`_Published.rows`.
    left, right : Array
        This device's full child arrays (``-1`` at leaves).
    geom : Any
        This device's walk geometry.
    root : Array
        This device's root node.
    theta : float
        The export MAC parameter.
    me : Array
        This device's index along the mesh axis.
    cap : CrossCapacities
        Capacities (export walk queue and list caps).
    two_sided : bool
        The rows carry child links; walk both trees.
    mac_type : str
        MAC variant. Static.
    walk_fn : Optional[Callable[..., Any]]
        The walk implementation (``None``: yggdrax's traced walk).
    node_active : Optional[Array]
        This device's live nodes (non-empty particle range). A cells tree pads its
        leaves to a static capacity with empty leaves in one subtree, and an
        empty node must never be exported: ``None`` treats every node as live,
        which emits pairs for the padding and can exhaust the list caps.

    Returns
    -------
    ExportLists
        Pairs ``(device * B + published index, local node)``.
    """
    cen = gathered[..., 0:3]
    rad = gathered[..., 3]
    kw = dict(
        max_pair_queue=cap.export_walk_queue,
        far_cap=cap.export_far_cap,
        near_cap=cap.export_near_cap,
        mac_type=mac_type,
        walk_fn=walk_fn,
        node_active=node_active,
    )
    args = (left, right, jnp.asarray(geom.center), jnp.asarray(geom.radius), root)
    if not two_sided:
        act = gathered[..., 4] > 0.5
        return jax.named_call(export_walk, name="cross_export_walk")(
            *args, cen, rad, act, float(theta), me, **kw
        )
    idx = jnp.asarray(left).dtype
    kids_l = jnp.round(gathered[..., 4]).astype(idx)
    kids_r = jnp.round(gathered[..., 5]).astype(idx)
    act = gathered[..., 6] > 0.5
    return jax.named_call(export_walk_two_sided, name="cross_export_walk")(
        *args, cen, rad, kids_l, kids_r, act, float(theta), me, **kw
    )


def _row_of_slot(counts: Array, capacity: int) -> tuple[Array, Array, Array, Array]:
    """Map every slot of a flat buffer laid out row by row back to its row.

    Row ``r`` owns ``counts[r]`` consecutive slots starting at the exclusive cumsum.
    Each non-empty row's first slot is marked with the row id and a running maximum
    carries it forward -- O(capacity), no sort and no search.

    Parameters
    ----------
    counts : Array
        ``(rows,)`` non-negative slot counts.
    capacity : int
        Length of the flat buffer. Static.

    Returns
    -------
    tuple[Array, Array, Array, Array]
        ``(row, within, live, total)``: per slot its row and offset in that row, whether
        it holds data, and the total number of live slots.
    """
    idx = jnp.int32
    counts = jnp.asarray(counts, idx)
    start = jnp.cumsum(counts, dtype=idx) - counts
    total = jnp.sum(counts, dtype=idx)
    rows = counts.shape[0]
    mark = (
        jnp.full((capacity,), -1, idx)
        .at[jnp.where(counts > 0, start, capacity)]
        .set(jnp.arange(rows, dtype=idx), mode="drop")
    )
    row = jnp.maximum(jax.lax.cummax(mark, axis=0), 0)
    slot = jnp.arange(capacity, dtype=idx)
    live = slot < total
    row = jnp.where(live, row, 0)
    within = jnp.where(live, slot - start[row], 0)
    return row, within, live, total


def _near_exchange_tiles(
    sb_n: Any,
    leaf_rows: list,
    starts: Array,
    ends: Array,
    pos_sorted: Array,
    mass_sorted: Array,
    *,
    W: int,
    payload_capacity: int,
    csr_capacity: int,
    ndev: int,
    axis_name: str,
) -> tuple[Any, Array, Array, Array]:
    """The near import with every leaf's particles in a W-wide tile (the control).

    Parameters
    ----------
    sb_n : Any
        The near send buffers.
    leaf_rows : list
        Per-row geometry and multipole columns, in payload order.
    starts, ends : Array
        Each row's leaf particle range ``[start, end]`` in sorted order.
    pos_sorted, mass_sorted : Array
        This device's Morton-sorted particles.
    W : int
        Tile width. Static.
    payload_capacity, csr_capacity : int
        Receive capacities. Static.
    ndev : int
        Mesh size.
    axis_name : str
        Mesh axis.

    Returns
    -------
    tuple[Any, Array, Array, Array]
        ``(imported, imported_positions (rows, W, 3), imported_masses (rows, W),
        overflow)``; the tile format has no particle buffer to overflow.
    """
    okrow = (sb_n.node_rows >= 0)[:, None]
    slot = jnp.arange(W, dtype=starts.dtype)[None, :]
    idx_p = jnp.clip(starts[:, None] + slot, 0, pos_sorted.shape[0] - 1)
    valid = okrow & (starts[:, None] + slot <= ends[:, None])
    tile_pos = jnp.where(valid[..., None], pos_sorted[idx_p], 0.0)
    tile_mass = jnp.where(valid, mass_sorted[idx_p], 0.0)
    head = sum(int(c.shape[1]) for c in leaf_rows)
    payload = jnp.concatenate(
        leaf_rows + [tile_pos.reshape(tile_pos.shape[0], -1), tile_mass], axis=1
    )
    got = exchange_export_list(
        payload,
        sb_n.node_sizes,
        sb_n.csr_cell,
        sb_n.csr_row,
        sb_n.csr_sizes,
        payload_capacity=payload_capacity,
        csr_capacity=csr_capacity,
        ndev=ndev,
        axis_name=axis_name,
    )
    imp_pos = got.payload[:, head : head + 3 * W].reshape(-1, W, 3)
    imp_mass = got.payload[:, head + 3 * W : head + 4 * W]
    return got, imp_pos, imp_mass, jnp.asarray(False)


def _near_exchange_compact(
    sb_n: Any,
    leaf_rows: list,
    starts: Array,
    ends: Array,
    pos_sorted: Array,
    mass_sorted: Array,
    *,
    W: int,
    payload_capacity: int,
    csr_capacity: int,
    send_particle_cap: int,
    recv_particle_cap: int,
    ndev: int,
    axis_name: str,
) -> tuple[Any, Array, Array, Array]:
    """The near import with only the LIVE particles on the wire.

    A W-wide tile per leaf is ~70 % zeros at ~18 particles per 64-slot leaf, and the
    exchange sends whole rows. Here a row carries its geometry, multipole and particle
    COUNT, and the particles travel as one flat ``(P, 4)`` buffer in the same row order
    (rows and particles are both grouped by destination, so they arrive in the same
    sender order and one exclusive cumsum of the counts locates every row's particles).
    The receiver rebuilds the same tiles, so nothing downstream changes.

    Parameters
    ----------
    sb_n : Any
        The near send buffers.
    leaf_rows : list
        Per-row geometry and multipole columns, in payload order.
    starts, ends : Array
        Each row's leaf particle range ``[start, end]`` in sorted order.
    pos_sorted, mass_sorted : Array
        This device's Morton-sorted particles.
    W : int
        Tile width. Static.
    payload_capacity, csr_capacity : int
        Receive capacities of the row and CSR rounds. Static.
    send_particle_cap, recv_particle_cap : int
        Flat particle buffer capacities. Static.
    ndev : int
        Mesh size.
    axis_name : str
        Mesh axis.

    Returns
    -------
    tuple[Any, Array, Array, Array]
        ``(imported, imported_positions (rows, W, 3), imported_masses (rows, W),
        overflow)`` -- ``overflow`` when either particle buffer is too small.
    """
    idx = jnp.int32
    live_row = sb_n.node_rows >= 0
    counts = jnp.where(live_row, jnp.clip(ends - starts + 1, 0, W), 0).astype(idx)
    row, within, live_q, total = _row_of_slot(counts, send_particle_cap)
    src = jnp.clip(starts.astype(idx)[row] + within, 0, pos_sorted.shape[0] - 1)
    parts = jnp.where(
        live_q[:, None],
        jnp.concatenate([pos_sorted[src], mass_sorted[src][:, None]], axis=1),
        0.0,
    )
    # rows are grouped by destination: [row_bound[d], row_bound[d + 1])
    row_bound = jnp.concatenate(
        [jnp.zeros((1,), idx), jnp.cumsum(jnp.asarray(sb_n.node_sizes, idx), dtype=idx)]
    )
    count_bound = jnp.concatenate([jnp.zeros((1,), idx), jnp.cumsum(counts, dtype=idx)])
    part_sizes = count_bound[row_bound[1:]] - count_bound[row_bound[:-1]]
    payload = jnp.concatenate(
        leaf_rows + [counts.astype(pos_sorted.dtype)[:, None]], axis=1
    )
    got = exchange_export_list(
        payload,
        sb_n.node_sizes,
        sb_n.csr_cell,
        sb_n.csr_row,
        sb_n.csr_sizes,
        payload_capacity=payload_capacity,
        csr_capacity=csr_capacity,
        ndev=ndev,
        axis_name=axis_name,
    )
    recv, recv_sizes, _ = ragged_all_to_all_exchange(
        parts,
        part_sizes,
        output_capacity=recv_particle_cap,
        axis_name=axis_name,
    )
    # rows beyond what arrived are filled with zeros, i.e. count 0
    r_counts = jnp.clip(got.payload[:, -1], 0, W).astype(idx)
    r_start = jnp.cumsum(r_counts, dtype=idx) - r_counts
    slot = jnp.arange(W, dtype=idx)[None, :]
    take = jnp.clip(r_start[:, None] + slot, 0, recv_particle_cap - 1)
    valid = slot < r_counts[:, None]
    imp_pos = jnp.where(valid[..., None], recv[take, :3], 0.0)
    imp_mass = jnp.where(valid, recv[take, 3], 0.0)
    overflow = (total > send_particle_cap) | (jnp.sum(recv_sizes) > recv_particle_cap)
    return got, imp_pos, imp_mass, overflow


def _far_receiver_walk_needed(export_theta: Optional[float], theta: float) -> bool:
    """Whether the far receiver walk can refine anything (else it is a pass-through).

    Parameters
    ----------
    export_theta : Optional[float]
        The sender's export MAC parameter (``None``: ``theta``).
    theta : float
        The receiver's MAC parameter.

    Returns
    -------
    bool
        ``True`` when the walk must run: a sender exporting under a different theta,
        or ``JACCPOT_CROSS_FAR_RECEIVER_WALK=1``.
    """
    from jaccpot._env import env_flag

    if export_theta is not None and float(export_theta) != float(theta):
        return True
    return env_flag("JACCPOT_CROSS_FAR_RECEIVER_WALK", False)


def _direct_far_lists(cells: Array, got: Any) -> Any:
    """The far receiver lists read straight off the received CSR.

    Parameters
    ----------
    cells : Array
        ``(B,)`` this device's node behind every published summary index: its cut
        cells, or (two-sided export) its summary-tree nodes, internal ones included.
    got : Any
        The received far import (``ImportedCells``): its CSR is prefix-live, one
        contiguous block per sender, ``num_csr`` entries in all.

    Returns
    -------
    Any
        A :class:`yggdrax.distributed.import_cells.ReceiverLists` with every CSR
        entry as a far pair (local cell root, imported payload row) and no near pairs.
    """
    from yggdrax.distributed.import_cells import ReceiverLists

    csr_cell = jnp.asarray(got.csr_cell)
    csr_row = jnp.asarray(got.csr_row)
    idx = csr_row.dtype
    live = (jnp.arange(csr_cell.shape[0]) < got.num_csr) & (csr_cell >= 0)
    neg = jnp.asarray(-1, idx)
    target = jnp.where(live, jnp.asarray(cells, idx)[jnp.where(live, csr_cell, 0)], neg)
    source = jnp.where(live, csr_row, neg)
    empty = jnp.full((1,), -1, idx)
    false = jnp.asarray(False)
    return ReceiverLists(
        far_target=target,
        far_source=source,
        far_count=jnp.asarray(got.num_csr),
        near_target=empty,
        near_source=empty,
        near_count=jnp.asarray(0, idx),
        far_overflow=false,
        near_overflow=false,
        queue_overflow=false,
    )


def _near_receiver_walk_needed(
    *,
    two_sided: bool,
    max_leaves: int,
    theta: float,
    near_theta: Optional[float],
    export_theta: Optional[float],
) -> bool:
    """Whether the NEAR receiver walk can refine anything (else it is a pass-through).

    Two-sided over receiver LEAVES, a near pair the sender emits is (my cell, its
    leaf) where my cell holds exactly one live leaf, tested on the very centre and
    radius the sender used. Re-walking it can only bottom out at (my leaf, its leaf)
    and fail the MAC again: no far pairs, near pairs == the received CSR (tested on
    CPU). Any other mode, a different receiver or export theta, or
    ``JACCPOT_CROSS_NEAR_RECEIVER_WALK=1`` runs the walk.

    Parameters
    ----------
    two_sided : bool
        The export mode.
    max_leaves : int
        Leaves per summary cell at most.
    theta : float
        The hook's MAC parameter.
    near_theta, export_theta : Optional[float]
        The receiver's near-walk and the sender's export MAC parameters.

    Returns
    -------
    bool
        ``True`` when the walk must run.
    """
    from jaccpot._env import env_flag

    if env_flag("JACCPOT_CROSS_NEAR_RECEIVER_WALK", False):
        return True
    if not two_sided or int(max_leaves) != 1:
        return True
    return any(
        t is not None and float(t) != float(theta) for t in (near_theta, export_theta)
    )


def _direct_near_lists(
    cells: Array, node_ranges: Array, num_internal: int, got: Any
) -> Any:
    """The near receiver lists read straight off the received CSR (leaf summary).

    Parameters
    ----------
    cells : Array
        ``(B,)`` this device's node behind every published summary index.
    node_ranges : Array
        ``(total_nodes, 2)`` this device's inclusive particle ranges.
    num_internal : int
        Nodes at or above this index are leaves.
    got : Any
        The received near import (``ImportedCells``).

    Returns
    -------
    Any
        A :class:`yggdrax.distributed.import_cells.ReceiverLists` with every CSR
        entry as a near pair (my LEAF under the cell, imported row) and no far pairs.
        A cell with one live leaf can be an internal node whose other child is empty;
        its leaf is the one starting where the cell starts.
    """
    from yggdrax.distributed.import_cells import ReceiverLists

    csr_cell = jnp.asarray(got.csr_cell)
    csr_row = jnp.asarray(got.csr_row)
    idx = csr_row.dtype
    live = (jnp.arange(csr_cell.shape[0]) < got.num_csr) & (csr_cell >= 0)
    neg = jnp.asarray(-1, idx)
    nr = jnp.asarray(node_ranges)
    cell = jnp.asarray(cells, idx)[jnp.where(live, csr_cell, 0)]
    # leaves hold ascending, contiguous ranges (padding leaves start past every live
    # one), so the leaf under a one-leaf cell is found by the cell's start
    leaf = num_internal + jnp.searchsorted(
        nr[num_internal:, 0],
        nr[jnp.maximum(cell, 0), 0],
        side="left",
        method=searchsorted_method(),
    )
    target = jnp.where(live, leaf.astype(idx), neg)
    source = jnp.where(live, csr_row, neg)
    empty = jnp.full((1,), -1, idx)
    false = jnp.asarray(False)
    return ReceiverLists(
        far_target=empty,
        far_source=empty,
        far_count=jnp.asarray(0, idx),
        near_target=target,
        near_source=source,
        near_count=jnp.asarray(got.num_csr),
        far_overflow=false,
        near_overflow=false,
        queue_overflow=false,
    )


def _symmetric_exchange_enabled(
    *,
    two_sided: bool,
    max_leaves: int,
    theta: float,
    near_theta: Optional[float],
    export_theta: Optional[float],
) -> bool:
    """Whether the hook runs the SYMMETRIC exchange (no CSR on the wire).

    Under a two-sided leaf summary every device publishes its whole live tree, so the
    gathered summaries hold every device's tree and any two devices can run the same
    walk between their trees and agree on every pair without telling each other.
    ``JACCPOT_CROSS_SYMMETRIC=0`` keeps the CSR exchange (the control), as do the tile
    wire format, a receiver walk forced on, and any theta that differs between sender
    and receiver.

    Parameters
    ----------
    two_sided : bool
        The export mode.
    max_leaves : int
        Leaves per summary cell at most.
    theta : float
        The hook's MAC parameter.
    near_theta, export_theta : Optional[float]
        The receiver's near-walk and the sender's export MAC parameters.

    Returns
    -------
    bool
        ``True`` for the symmetric exchange.
    """
    from jaccpot._env import env_flag

    if not env_flag("JACCPOT_CROSS_SYMMETRIC", True) or _near_tiles_on_the_wire():
        return False
    if env_flag("JACCPOT_CROSS_FAR_RECEIVER_WALK", False):
        return False
    return not _near_receiver_walk_needed(
        two_sided=two_sided,
        max_leaves=max_leaves,
        theta=theta,
        near_theta=near_theta,
        export_theta=export_theta,
    )


def _symmetric_walk(
    gathered: Array,
    me: Array,
    theta: float,
    cap: "CrossCapacities",
    *,
    mac_type: str,
    walk_fn: Optional[Callable[..., Any]],
    error_order: int = 0,
) -> Any:
    """One walk between this device's tree and every peer's, over the gathered trees.

    The index space is ``[ndev x S]``, each device's published summary tree in its
    own block, and the seed for peer ``d`` is ``(min(me, d), max(me, d))`` in block
    terms -- the SAME seed, in the same orientation, that device ``d`` uses for this
    pair. The walk reads only the gathered arrays, which are identical on every
    device, so both devices of a pair get the identical pair set. The orientation is
    load-bearing: the walk splits side ``a`` on an exact radius tie, so a device
    walking "peer vs mine" and its peer walking "mine vs peer" could decompose
    differently.

    Parameters
    ----------
    gathered : Array
        ``(ndev, S, k)`` every device's summary rows (``_summary_rows``, two-sided).
    me : Array
        This device's index along the mesh axis.
    theta : float
        MAC parameter.
    cap : CrossCapacities
        Queue and list capacities (the export walk's).
    mac_type : str
        MAC variant. Static.
    walk_fn : Optional[Callable[..., Any]]
        The walk implementation (``None``: yggdrax's traced walk).
    error_order : int
        ``mac_type='dehnen_error'``: accept by eq (16a) at this order from the
        gathered table columns (:func:`_summary_rows`' ``criterion``); ``walk_fn``
        is then :func:`cross_walk_fn`'s criterion walk, which takes the table as
        ``error_table``. ``0``: the opening angle. Static.

    Returns
    -------
    Any
        The walk result: pairs ``(a, b)`` with ``a`` in the lower device's block.

    Raises
    ------
    ValueError
        If ``error_order > 0`` comes without a criterion walk.
    """
    from yggdrax.interactions import dual_tree_walk_mutual

    ndev, S = int(gathered.shape[0]), int(gathered.shape[1])
    criterion: dict = {}
    if int(error_order) > 0:
        if walk_fn is None:
            raise ValueError("the criterion's cross walk needs cross_walk_fn's walk")
        width = _summary_table_width(int(error_order))
        # every device's published rows, in the walk's [ndev x S] index space: the
        # same floats on every device, so both ends of a pair decide it alike
        criterion["error_table"] = gathered[
            ..., _SUMMARY_TABLE_COL : _SUMMARY_TABLE_COL + width
        ].reshape(ndev * S, width)
    idx = jnp.int32
    base = (jnp.arange(ndev, dtype=idx) * S)[:, None]

    def kids(col: int) -> Array:
        k = jnp.round(gathered[..., col]).astype(idx)
        return jnp.where(k >= 0, k + base, jnp.asarray(-1, idx)).reshape(ndev * S)

    dev = jnp.arange(ndev, dtype=idx)
    me_i = jnp.asarray(me).astype(idx)
    dead = dev == me_i
    return (walk_fn or dual_tree_walk_mutual)(
        kids(4),
        kids(5),
        gathered[..., 0:3].reshape(ndev * S, 3),
        gathered[..., 3].reshape(ndev * S),
        float(theta),
        jnp.asarray(0, idx),
        max_pair_queue=cap.export_walk_queue,
        far_cap=cap.export_far_cap,
        near_cap=cap.export_near_cap,
        mac_type=mac_type,  # pyright: ignore[reportArgumentType]
        node_active=(gathered[..., 6] > 0.5).reshape(ndev * S),
        seed_a=jnp.where(dead, jnp.asarray(-1, idx), jnp.minimum(dev, me_i) * S),
        seed_b=jnp.where(dead, jnp.asarray(-1, idx), jnp.maximum(dev, me_i) * S),
        **criterion,
    )


class _PeerRows(NamedTuple):
    """Unique (peer, node) rows of a pair list, grouped by peer, and each pair's row.

    Attributes
    ----------
    rows : Array
        ``(capacity,)`` the node (a summary index) of every unique (peer, node), in
        (peer, node) order, ``-1``-padded.
    sizes : Array
        ``(ndev,)`` unique rows per peer.
    pair_row : Array
        ``(P,)`` each pair's index into ``rows``; ``-1`` for dead pairs.
    overflow : Array
        More unique rows than ``capacity``.
    needed : Array
        Unique rows, laid out or not: the capacity the list needed.
    present : Array
        ``(ndev x S,)`` int32, 1 at every (peer, node) that has a row.
    """

    rows: Array
    sizes: Array
    pair_row: Array
    overflow: Array
    needed: Array
    present: Array


def _rows_by_peer(
    peer: Array, node: Array, live: Array, *, ndev: int, S: int, capacity: int
) -> _PeerRows:
    """Deduplicate a pair list's (peer, node) and lay the rows out peer by peer.

    Rows for peer ``d`` form a contiguous block in ascending summary index. A sender
    laying out what it ships to ``d`` and ``d`` laying out what it receives from this
    sender both order by the SENDER's summary index, so both ends agree on every row
    without exchanging an index.

    No sort: ``(peer, node)`` lives in the fixed ``[ndev x S]`` summary space, so a
    presence map over it and one prefix sum give every row and every pair's row in
    O(ndev x S) -- instead of an argsort over the whole pair-list capacity (2^21 at
    2e5 per card on four cards, four of them per force).

    Parameters
    ----------
    peer, node : Array
        ``(P,)`` per pair: the peer device and the node (summary index in the block
        that matters for this list).
    live : Array
        ``(P,)`` live pairs.
    ndev, S : int
        Mesh size and summary block width.
    capacity : int
        Static row capacity. Read the overflow flag.

    Returns
    -------
    _PeerRows
        The rows, per-peer sizes and each pair's row.
    """
    idx = jnp.int32
    span = ndev * S
    key = jnp.where(live, peer.astype(idx) * S + node.astype(idx), span)
    present = jnp.zeros((span,), idx).at[key].set(1, mode="drop")
    before = jnp.cumsum(present, dtype=idx) - present  # rows ahead of each key
    n_rows = jnp.sum(present, dtype=idx)
    slot = jnp.where((present > 0) & (before < capacity), before, capacity)
    rows = (
        jnp.full((capacity,), -1, idx)
        .at[slot]
        .set(jnp.arange(span, dtype=idx) % S, mode="drop")
    )
    sizes = jnp.sum(present.reshape(ndev, S), axis=1, dtype=idx)
    pair_row = jnp.where(live, before[jnp.minimum(key, span - 1)], -1)
    return _PeerRows(rows, sizes, pair_row, n_rows > capacity, n_rows, present)


def _symmetric_pair_sides(
    res_a: Array, res_b: Array, count: Array, me: Array, S: int
) -> tuple[Array, Array, Array, Array]:
    """Split walk pairs into (live, my summary index, peer, peer's summary index).

    Parameters
    ----------
    res_a, res_b : Array
        ``(P,)`` the walk's pair ends (global summary indices).
    count : Array
        Live prefix length.
    me : Array
        This device's index.
    S : int
        Summary block width.

    Returns
    -------
    tuple
        ``(live, mine, peer, theirs)``, all ``(P,)``.
    """
    idx = jnp.int32
    a = jnp.asarray(res_a).astype(idx)
    b = jnp.asarray(res_b).astype(idx)
    live = (jnp.arange(a.shape[0]) < count) & (a >= 0) & (b >= 0)
    a_mine = (a // S) == jnp.asarray(me).astype(idx)
    mine = jnp.where(a_mine, a, b)
    theirs = jnp.where(a_mine, b, a)
    return live, mine % S, theirs // S, theirs % S


def _leaf_under(cells: Array, node_ranges: Array, num_internal: int) -> Array:
    """The leaf under each one-leaf cell (itself when it is a leaf).

    Parameters
    ----------
    cells : Array
        Node indices, each holding exactly one live leaf.
    node_ranges : Array
        ``(total_nodes, 2)`` inclusive particle ranges.
    num_internal : int
        Nodes at or above this index are leaves.

    Returns
    -------
    Array
        The leaf node indices.
    """
    nr = jnp.asarray(node_ranges)
    return num_internal + jnp.searchsorted(
        nr[num_internal:, 0],
        nr[jnp.maximum(cells, 0), 0],
        side="left",
        method=searchsorted_method(),
    )


def _symmetric_exchange(
    gathered: Array,
    me: Array,
    nodes: Array,
    tree: Any,
    mp: Any,
    *,
    theta: float,
    cap: "CrossCapacities",
    ndev: int,
    mac_type: str,
    walk_fn: Optional[Callable[..., Any]],
    axis_name: str,
    with_near: bool,
    error_order: int = 0,
    node_mass: Optional[Array] = None,
) -> dict:
    """The cross field with only multipoles and particles on the wire.

    Every device runs :func:`_symmetric_walk` over the gathered trees and reads BOTH
    of its lists off the same pairs: a far pair (my node, peer's node) means "ship my
    multipole to the peer" and "apply the peer's multipole to my node"; a near pair
    (my leaf, peer's leaf) means the same for particles. The peer derives the
    identical pairs, so neither side sends the other a CSR, a row count or a size
    beyond the one per-exchange size gather -- two ragged exchanges in place of five.
    What a sender ships to ``d`` and what ``d`` expects from it are both laid out in
    the sender's ascending summary index (:func:`_rows_by_peer`); the receive sizes
    the size gather reports are checked against the ones predicted locally, so a
    divergence raises the overflow flag instead of misaligning rows.

    Parameters
    ----------
    gathered : Array
        ``(ndev, S, 8)`` every device's two-sided summary rows (with counts).
    me : Array
        This device's index along the mesh axis.
    nodes : Array
        ``(S,)`` this device's tree node behind each of its summary indices.
    tree : Any
        This device's refreshed tree.
    mp : Any
        This device's multipoles (``packed``, ``centers``).
    theta : float
        MAC parameter.
    cap : CrossCapacities
        Capacities.
    ndev : int
        Mesh size.
    mac_type : str
        MAC variant. Static.
    walk_fn : Optional[Callable[..., Any]]
        The walk implementation.
    axis_name : str
        Mesh axis.
    with_near : bool
        Also exchange particles and return the near list.
    error_order : int
        ``mac_type='dehnen_error'``: the walk accepts by eq (16a) at this order
        from the gathered table columns (:func:`_symmetric_walk`). Static.
    node_mass : Optional[Array]
        ``[nodes]`` this device's node masses (its upward pass's mass moments);
        with ``error_order`` they travel with the multipoles, one more column of
        the far payload, for eq (16b)'s far half on the receiver.

    Returns
    -------
    dict
        ``cross_far`` (the M2L tuple), ``far_overflow``, the walk result ``walk``, the
        far send/receive rows, ``needs`` (what each capacity it read needed, keyed by
        the :data:`CROSS_NEEDS_FIELDS` entry), with ``error_order`` the imported
        rows' masses ``cross_masses``, and with ``with_near`` the near-sink fields
        (``positions``, ``masses``, ``mask``, ``target_node``, ``source_row``,
        ``count``, ``overflow``), ``near_particles`` and ``send_particles``.

    Raises
    ------
    ValueError
        If ``error_order > 0`` comes without ``node_mass``.
    """
    if int(error_order) > 0 and node_mass is None:
        raise ValueError("the criterion's far payload needs the node masses")
    S = int(gathered.shape[1])
    idx = jnp.int32
    nodes = jnp.asarray(nodes).astype(idx)
    packed = jnp.asarray(mp.packed)
    n_local = int(packed.shape[0])
    n_coeff = int(packed.shape[1])
    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    nr = jnp.asarray(tree.node_ranges)

    res = jax.named_call(_symmetric_walk, name="cross_export_walk")(
        gathered,
        me,
        theta,
        cap,
        mac_type=mac_type,
        walk_fn=walk_fn,
        error_order=int(error_order),
    )
    out: dict = {"walk": res}

    # ---- far: my multipoles out, the peers' in, one ragged exchange ----------
    with jax.named_scope("cross_send_far"):
        fl, f_mine, f_peer, f_theirs = _symmetric_pair_sides(
            res.far_a, res.far_b, res.far_count, me, S
        )
        exp_f = _rows_by_peer(
            f_peer, f_mine, fl, ndev=ndev, S=S, capacity=cap.send_node_cap
        )
        imp_f = _rows_by_peer(
            f_peer, f_theirs, fl, ndev=ndev, S=S, capacity=cap.recv_node_cap
        )
        alive = (exp_f.rows >= 0)[:, None]
        send_node = nodes[jnp.maximum(exp_f.rows, 0)]
        columns = [
            jnp.where(alive, packed[send_node], 0.0),
            jnp.where(alive, jnp.asarray(mp.centers)[send_node], 0.0),
        ]
        if int(error_order) > 0:
            # the mass moment itself, not the packed monopole (another summation
            # order): the receiver's f_b far terms read the float its owner has
            assert node_mass is not None
            mass = jnp.asarray(node_mass, packed.dtype)[send_node][:, None]
            columns.append(jnp.where(alive, mass, 0.0))
        payload = jnp.concatenate(columns, axis=1)
    recv, recv_sizes, _ = jax.named_call(
        ragged_all_to_all_exchange, name="cross_exchange_far"
    )(payload, exp_f.sizes, output_capacity=cap.recv_node_cap, axis_name=axis_name)
    far_mismatch = jnp.any(recv_sizes.astype(idx) != imp_f.sizes)
    ok_f = fl & (imp_f.pair_row >= 0)
    src_f = jnp.where(ok_f, n_local + imp_f.pair_row, -1)
    tgt_f = jnp.where(ok_f, nodes[f_mine], -1)
    out["cross_far"] = (
        recv[:, :n_coeff],
        recv[:, n_coeff : n_coeff + 3],
        src_f,
        tgt_f,
    )
    if int(error_order) > 0:
        # only the rows a live cross pair names are read, as the multipoles
        out["cross_masses"] = recv[:, n_coeff + 3]
    out["exp_far"] = exp_f
    out["imp_far"] = imp_f
    out["far_overflow"] = (
        res.far_overflow
        | res.near_overflow
        | res.queue_overflow
        | exp_f.overflow
        | imp_f.overflow
        | far_mismatch
    )
    # What each capacity above NEEDED (`CROSS_NEEDS_FIELDS`), for a re-plan. The rows
    # are counted on the presence map, which no row capacity truncates; the walk's
    # counts survive a full list only in the Pallas walk (`far_needed`), yggdrax's
    # stops at its first overflow. Pairs a walk overflow dropped are missing from
    # every count after it.
    walk_overflow = res.far_overflow | res.near_overflow | res.queue_overflow
    out["needs"] = {
        "export_far_cap": getattr(res, "far_needed", res.far_count),
        "export_near_cap": getattr(res, "near_needed", res.near_count),
        "export_walk_queue": res.peak_wavefront,
        "send_node_cap": exp_f.needed,
        "recv_node_cap": imp_f.needed,
        "walk_lower_bound": (
            res.queue_overflow if hasattr(res, "far_needed") else walk_overflow
        ),
        "pairs_lower_bound": walk_overflow,
        # a truncated row buffer reads as a disagreement too; that one a re-plan fixes
        "mismatch": far_mismatch & ~(exp_f.overflow | imp_f.overflow),
    }
    if not with_near:
        return out

    # ---- near: my leaves' particles out, the peers' in, one ragged exchange ---
    W = int(cap.leaf_width)
    pos_sorted = jnp.asarray(tree.positions_sorted)
    mass_sorted = jnp.asarray(tree.masses_sorted)
    with jax.named_scope("cross_send_near"):
        nl, n_mine, n_peer, n_theirs = _symmetric_pair_sides(
            res.near_a, res.near_b, res.near_count, me, S
        )
        exp_n = _rows_by_peer(
            n_peer, n_mine, nl, ndev=ndev, S=S, capacity=cap.send_node_cap
        )
        imp_n = _rows_by_peer(
            n_peer, n_theirs, nl, ndev=ndev, S=S, capacity=cap.recv_node_cap
        )
        s_ok = exp_n.rows >= 0
        s_node = nodes[jnp.maximum(exp_n.rows, 0)]
        starts = nr[s_node, 0].astype(idx)
        # the LIVE particles only, as the receiver's count column says
        # (`_live_node_ends`: a padded shard's ranges can run into its dead rows)
        ends = _live_node_ends(tree)
        full = jnp.where(s_ok, ends[s_node].astype(idx) - starts + 1, 0)
        counts = jnp.clip(full, 0, W)
        row, within, live_q, total = _row_of_slot(counts, cap.send_particle_cap)
        src = jnp.clip(starts[row] + within, 0, pos_sorted.shape[0] - 1)
        parts = jnp.where(
            live_q[:, None],
            jnp.concatenate([pos_sorted[src], mass_sorted[src][:, None]], axis=1),
            0.0,
        )
        row_bound = jnp.concatenate(
            [jnp.zeros((1,), idx), jnp.cumsum(exp_n.sizes, dtype=idx)]
        )
        count_bound = jnp.concatenate(
            [jnp.zeros((1,), idx), jnp.cumsum(counts, dtype=idx)]
        )
        part_sizes = count_bound[row_bound[1:]] - count_bound[row_bound[:-1]]
    recv_p, recv_psizes, _ = jax.named_call(
        ragged_all_to_all_exchange, name="cross_exchange_near"
    )(parts, part_sizes, output_capacity=cap.recv_particle_cap, axis_name=axis_name)
    with jax.named_scope("cross_recv_near"):
        # each received row's particle count, read off the SENDER's published tree
        r_ok = imp_n.rows >= 0
        r_peer = jnp.searchsorted(
            jnp.cumsum(imp_n.sizes, dtype=idx),
            jnp.arange(imp_n.rows.shape[0], dtype=idx),
            side="right",
            method=searchsorted_method(),
        )
        r_peer = jnp.minimum(r_peer, ndev - 1)
        r_count = jnp.where(
            r_ok,
            jnp.round(gathered[r_peer, jnp.maximum(imp_n.rows, 0), 7]).astype(idx),
            0,
        )
        r_count = jnp.clip(r_count, 0, W)
        expect = jax.ops.segment_sum(
            r_count, jnp.where(r_ok, r_peer, ndev), num_segments=ndev + 1
        )[:ndev]
        near_mismatch = jnp.any(expect != recv_psizes.astype(idx))
        r_start = jnp.cumsum(r_count, dtype=idx) - r_count
        slot = jnp.arange(W, dtype=idx)[None, :]
        take = jnp.clip(r_start[:, None] + slot, 0, cap.recv_particle_cap - 1)
        valid = slot < r_count[:, None]
        imp_pos = jnp.where(valid[..., None], recv_p[take, :3], 0.0)
        imp_mass = jnp.where(valid, recv_p[take, 3], 0.0)
        ok_n = nl & (imp_n.pair_row >= 0)
        leaf = _leaf_under(nodes[n_mine], nr, num_internal).astype(idx)
    with jax.named_scope("cross_near_needs"):
        # particles per (peer, node) row, laid out or not: mine from my own tree (the
        # `full` above, for every row), the peers' from their published counts
        live_node = nodes >= 0
        safe_node = jnp.where(live_node, nodes, 0)
        mine = jnp.where(
            live_node, jnp.maximum(ends[safe_node] - nr[safe_node, 0] + 1, 0), 0
        ).astype(idx)
        shipped = exp_n.present.reshape(ndev, S) * mine[None, :]
        theirs = jnp.round(gathered[..., 7]).astype(idx)
        needs = out["needs"]
        needs["send_node_cap"] = jnp.maximum(needs["send_node_cap"], exp_n.needed)
        needs["recv_node_cap"] = jnp.maximum(needs["recv_node_cap"], imp_n.needed)
        needs["send_particle_cap"] = jnp.sum(shipped, dtype=idx)
        needs["recv_particle_cap"] = jnp.sum(
            imp_n.present.reshape(ndev, S) * theirs, dtype=idx
        )
        needs["leaf_width"] = jnp.max(shipped)
        buffers_overflowed = (
            exp_f.overflow
            | imp_f.overflow
            | exp_n.overflow
            | imp_n.overflow
            | (total > cap.send_particle_cap)
            | (jnp.sum(recv_psizes) > cap.recv_particle_cap)
            | jnp.any(full > W)
        )
        needs["mismatch"] = needs["mismatch"] | (near_mismatch & ~buffers_overflowed)
    out.update(
        positions=imp_pos,
        masses=imp_mass,
        mask=imp_mass != 0.0,
        target_node=jnp.where(ok_n, leaf, -1),
        source_row=jnp.where(ok_n, imp_n.pair_row, -1),
        count=res.near_count,
        near_particles=jnp.sum(recv_psizes),
        send_particles=total,
        exp_near=exp_n,
        imp_near=imp_n,
        overflow=(
            exp_n.overflow
            | imp_n.overflow
            | (total > cap.send_particle_cap)
            | (jnp.sum(recv_psizes) > cap.recv_particle_cap)
            | jnp.any(full > W)
            | near_mismatch
        ),
    )
    return out


def cross_walk_backend() -> str:
    """Walk implementation of the export and receiver walks.

    ``JACCPOT_CROSS_WALK``: ``"pallas"`` (one Pallas launch per round,
    :func:`jaccpot.pallas.mutual_walk_pallas.mutual_walk_pallas` with a seeded
    start) or ``"flat"`` (yggdrax ``dual_tree_walk_mutual``, traced JAX). Unset:
    whatever the local lane's own walk uses (``JACCPOT_STATIC_STRICT_FUSED_WALK``,
    Pallas on an Ampere+ GPU).

    Returns
    -------
    str
        ``"flat"`` or ``"pallas"``.
    """
    from jaccpot._env import env_choice
    from jaccpot.runtime._interaction_cache import strict_walk_backend

    return env_choice("JACCPOT_CROSS_WALK", strict_walk_backend(), ("flat", "pallas"))


def cross_walk_fn(
    mac_type: str, *, error_order: int = 0, theta_max: float = 1.0
) -> Optional[Callable[..., Any]]:
    """The ``walk_fn`` the export and receiver walks run, or ``None`` for yggdrax's.

    The traced walk runs one ``while_loop`` per wavefront width with ~40 ops per
    round and copies its full-capacity queues on every round: 64 MB twice per
    round at 1e6 particles per A100, ~8 ms of copies plus ~15 ms of walk kernels
    per force across the three cross walks. The Pallas walk classifies a round in
    one launch and checks the loop predicate every 16 rounds.

    Under ``mac_type='dehnen_error'`` (``error_order > 0``) the walk takes one more
    keyword, ``error_table`` (the ``[nodes, p + 2]`` eq (16a) rows, traced), and
    accepts by eq (16a) instead of the opening angle -- on either backend, with
    the single card's two implementations of one elementwise test: the Pallas
    walk's ``error_table`` / ``error_order`` / ``theta_max``, or yggdrax's walk
    with ``pair_accept=DehnenWalkAccept(p)``. The separation floor still applies.

    Parameters
    ----------
    mac_type : str
        The hook's MAC; the Pallas kernel implements ``dehnen`` / ``bh`` only, so
        any other keeps the traced walk.
    error_order : int
        ``0``: the opening angle. ``p > 0``: eq (16a) at expansion order ``p``.
        Static.
    theta_max : float
        Eq (16a)'s convergence clause (``rho_A + rho_B < theta_max r``). Static.

    Returns
    -------
    Optional[Callable[..., Any]]
        A function with ``dual_tree_walk_mutual``'s signature and result fields
        (plus ``error_table`` with ``error_order``); ``None`` for yggdrax's walk
        as it is.
    """
    if int(error_order) > 0:
        return _criterion_walk_fn(int(error_order), float(theta_max))
    if cross_walk_backend() != "pallas" or str(mac_type) not in ("dehnen", "bh"):
        return None

    def pallas_walk(
        left_child_full: Array,
        right_child_full: Array,
        centers: Array,
        radii: Array,
        theta: float,
        root: Array,
        *,
        max_pair_queue: int,
        far_cap: int,
        near_cap: int,
        mac_type: Optional[str] = None,
        node_active: Optional[Array] = None,
        seed_a: Optional[Array] = None,
        seed_b: Optional[Array] = None,
        seed_count: Optional[Array] = None,
        separation_floor: float = 0.0,
    ) -> Any:
        from jaccpot._env import env_flag
        from jaccpot.pallas.mutual_walk_pallas import mutual_walk_pallas

        del mac_type  # dehnen / bh: the kernel's (r_a + r_b)^2 <= theta^2 d^2
        return mutual_walk_pallas(
            left_child_full,
            right_child_full,
            centers,
            radii,
            float(theta),
            root,
            max_pair_queue=int(max_pair_queue),
            far_cap=int(far_cap),
            near_cap=int(near_cap),
            node_active=node_active,
            seed_a=seed_a,
            seed_b=seed_b,
            seed_count=seed_count,
            interpret=env_flag("JACCPOT_WALK_PALLAS_INTERPRET", False),
            separation_floor=float(separation_floor),
        )

    return pallas_walk


def _criterion_walk_fn(order: int, theta_max: float) -> Callable[..., Any]:
    """:func:`cross_walk_fn` under eq (16a): the walk with an ``error_table`` keyword.

    Parameters
    ----------
    order : int
        Expansion order ``p``.
    theta_max : float
        The convergence clause's bound.

    Returns
    -------
    Callable[..., Any]
        ``dual_tree_walk_mutual``'s signature plus ``error_table``, on the backend
        :func:`cross_walk_backend` names.
    """
    pallas = cross_walk_backend() == "pallas"

    def criterion_walk(
        left_child_full: Array,
        right_child_full: Array,
        centers: Array,
        radii: Array,
        theta: float,
        root: Array,
        *,
        error_table: Array,
        max_pair_queue: int,
        far_cap: int,
        near_cap: int,
        mac_type: Optional[str] = None,
        node_active: Optional[Array] = None,
        seed_a: Optional[Array] = None,
        seed_b: Optional[Array] = None,
        seed_count: Optional[Array] = None,
        separation_floor: float = 0.0,
    ) -> Any:
        # theta and mac_type are the geometric test's: eq (16a) replaces it
        del mac_type
        common = dict(
            max_pair_queue=int(max_pair_queue),
            far_cap=int(far_cap),
            near_cap=int(near_cap),
            node_active=node_active,
            seed_a=seed_a,
            seed_b=seed_b,
            seed_count=seed_count,
            separation_floor=float(separation_floor),
        )
        if pallas:
            from jaccpot._env import env_flag
            from jaccpot.pallas.mutual_walk_pallas import mutual_walk_pallas

            return mutual_walk_pallas(
                left_child_full,
                right_child_full,
                centers,
                radii,
                float(theta),
                root,
                interpret=env_flag("JACCPOT_WALK_PALLAS_INTERPRET", False),
                error_table=error_table,
                error_order=int(order),
                theta_max=float(theta_max),
                **common,
            )
        from yggdrax.interactions import dual_tree_walk_mutual

        from jaccpot.runtime._walk_criterion import DehnenWalkAccept

        return dual_tree_walk_mutual(
            left_child_full,
            right_child_full,
            centers,
            radii,
            float(theta),
            root,
            pair_accept=DehnenWalkAccept(int(order)),
            pair_accept_data={
                "table": error_table,
                "theta_max": jnp.asarray(float(theta_max), jnp.asarray(radii).dtype),
            },
            **common,
        )

    return criterion_walk


#: Capacities sized from a count are rounded up to a multiple of this.
_CROSS_CAP_GRANULE = 1024

#: A priori demand of the symmetric exchange's buffers per live leaf of the worst
#: shard and per peer, and the headroom over it: the factors the two-card probe sized
#: its buffers with (``bench/multigpu_c4_cross_force_probe.py``; Plummer, leaf 64,
#: theta 0.8, 2e5-1e6 per card). Only a first measurement runs on them
#: (:func:`plan_cross_capacities`); every later size comes from what it measured.
_A_PRIORI_PER_LEAF = {
    "export_far_cap": (16.0, 2.5),
    "export_near_cap": (10.0, 2.5),
    "export_walk_queue": (3.0, 1.5),
    "send_node_cap": (1.25, 2.5),
    "recv_node_cap": (1.25, 2.5),
    "send_particle_cap": (20.0, 2.0),
    "recv_particle_cap": (20.0, 2.0),
}


def _sized(need: Union[int, float], headroom: float) -> int:
    """``headroom x need`` (at least one entry), rounded up to the granule.

    Parameters
    ----------
    need : Union[int, float]
        Entries needed: a measured count, or an a priori estimate.
    headroom : float
        Factor over ``need``.

    Returns
    -------
    int
        A positive multiple of :data:`_CROSS_CAP_GRANULE`.
    """
    want = math.ceil(float(headroom) * max(float(need), 1.0))
    return -(-want // _CROSS_CAP_GRANULE) * _CROSS_CAP_GRANULE


def _needs_by_field(needs: Any) -> dict[str, int]:
    """A host needs vector as ``{field: value}``.

    Parameters
    ----------
    needs : Any
        ``[len(CROSS_NEEDS_FIELDS)]`` integers (:func:`cross_needs`, reduced).

    Returns
    -------
    dict[str, int]
        The vector keyed by :data:`CROSS_NEEDS_FIELDS`.

    Raises
    ------
    ValueError
        If the vector has the wrong length.
    """
    values = np.asarray(needs).astype(np.int64).reshape(-1)
    if values.shape[0] != len(CROSS_NEEDS_FIELDS):
        raise ValueError(
            f"a cross needs vector has {len(CROSS_NEEDS_FIELDS)} entries "
            f"(CROSS_NEEDS_FIELDS), got {values.shape[0]}"
        )
    return dict(zip(CROSS_NEEDS_FIELDS, (int(v) for v in values)))


def _default_headroom(headroom: Optional[float]) -> float:
    """The caller's headroom, or the local walk's (``JACCPOT_FLAT_WALK_CAP_HEADROOM``).

    Parameters
    ----------
    headroom : Optional[float]
        An explicit factor, or ``None``.

    Returns
    -------
    float
        The factor, at least 1.
    """
    if headroom is not None:
        return max(float(headroom), 1.0)
    from jaccpot.runtime._interaction_cache import flat_walk_cap_headroom

    return flat_walk_cap_headroom()


class CrossCapacities:
    """Static capacities for one cross exchange. Every one is an overflow risk.

    Attributes are read at trace time and fix every buffer shape, so they cannot be
    derived from what actually arrives. Three ways to size them, all for the default
    (symmetric) exchange:

    * :meth:`for_shard` -- a priori, from what the driver knows before any force: the
      shard's leaf capacity and live leaves, the mesh size and the leaf width;
    * :meth:`from_needs` -- from a MEASURED needs vector (:func:`cross_needs`), each
      capacity ``headroom x`` what it needed; :func:`plan_cross_capacities` measures
      one force at the start positions and does this;
    * :meth:`replanned` -- the segment retry: grown, never narrowed, to cover what a
      run whose flag fired needed.

    ``CrossCapacities()`` keeps the fixed defaults below.
    """

    def __init__(
        self,
        *,
        max_cells: int = 1024,
        max_leaves_per_cell: Optional[int] = None,
        export_far_cap: int = 1 << 16,
        export_near_cap: int = 1 << 16,
        send_node_cap: int = 1 << 13,
        send_csr_cap: int = 1 << 16,
        recv_node_cap: int = 1 << 13,
        recv_csr_cap: int = 1 << 16,
        walk_queue: int = 1 << 16,
        leaf_width: int = 64,
        recv_far_cap: int = 1 << 17,
        recv_near_cap: int = 1 << 17,
        recv_near_csr_cap: Optional[int] = None,
        export_walk_queue: Optional[int] = None,
        send_particle_cap: Optional[int] = None,
        recv_particle_cap: Optional[int] = None,
        summary_cell_level: Optional[int] = None,
        max_summary_nodes: Optional[int] = None,
    ) -> None:
        self.max_cells = int(max_cells)
        # None: 1 under the two-sided export, 4 under the one-sided one (see
        # `_max_leaves_per_cell`). At 1 the summary holds every live leaf, so
        # max_cells must cover the shard's live leaf count.
        self.max_leaves_per_cell = (
            None if max_leaves_per_cell is None else int(max_leaves_per_cell)
        )
        self.export_far_cap = int(export_far_cap)
        self.export_near_cap = int(export_near_cap)
        self.send_node_cap = int(send_node_cap)
        self.send_csr_cap = int(send_csr_cap)
        self.recv_node_cap = int(recv_node_cap)
        self.recv_csr_cap = int(recv_csr_cap)
        self.walk_queue = int(walk_queue)
        self.leaf_width = int(leaf_width)
        self.recv_far_cap = int(recv_far_cap)
        self.recv_near_cap = int(recv_near_cap)
        # The NEAR import's CSR: one entry per (cell, near leaf) the sender exported,
        # so at most export_near_cap x (ndev - 1) -- much shorter than the far CSR,
        # and it is the near receiver walk's seed width, i.e. a floor on walk_queue.
        # None keeps the old shared width.
        self.recv_near_csr_cap = (
            self.recv_csr_cap if recv_near_csr_cap is None else int(recv_near_csr_cap)
        )
        # The export walk's own queue: its peak is ~1/3 of the near receiver walk's
        # (0.72-0.90M vs 2.66M at 1e6 per device), and the Pallas walk launches one
        # program per 64 queue slots on EVERY round, so a shared queue sized for the
        # larger walk costs the smaller one in proportion. None: walk_queue.
        self.export_walk_queue = (
            self.walk_queue if export_walk_queue is None else int(export_walk_queue)
        )
        # The near import's LIVE particles, flat (4 floats each), instead of a
        # leaf_width tile per leaf: at ~18 particles per 64-slot leaf the tiles were
        # ~70 % zeros on the wire. None: the tile-equivalent bound node_cap x W,
        # which can never overflow.
        self.send_particle_cap = (
            self.send_node_cap * self.leaf_width
            if send_particle_cap is None
            else int(send_particle_cap)
        )
        self.recv_particle_cap = (
            self.recv_node_cap * self.leaf_width
            if recv_particle_cap is None
            else int(recv_particle_cap)
        )
        # A summary cell must fit inside one Morton cell of this level (leaves are
        # exempt). Without it a sparse node holding a few tiny far-apart leaves is one
        # huge cell; a sender's MAC against its bounding sphere fails for the whole
        # remote domain, so every remote particle ships every force (measured: one
        # cell per device held all 51k remote leaves in its near CSR at 2e6). None:
        # occupancy bound only.
        self.summary_cell_level = (
            None if summary_cell_level is None else int(summary_cell_level)
        )
        # The two-sided export's published summary TREE: the cut plus its ancestors
        # (and their empty children), a full binary tree over the cells -- 2 x cells
        # - 1 nodes, plus two per empty child. None: 2 x max_cells.
        self.max_summary_nodes = (
            2 * self.max_cells if max_summary_nodes is None else int(max_summary_nodes)
        )

    def __repr__(self) -> str:
        """The capacities, by attribute.

        Returns
        -------
        str
            ``CrossCapacities(name=value, ...)``.
        """
        fields = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"CrossCapacities({fields})"

    def _with(self, **changes: int) -> "CrossCapacities":
        """A copy with some capacities replaced.

        Parameters
        ----------
        **changes : int
            Attribute name to its new value.

        Returns
        -------
        CrossCapacities
            The copy; ``self`` is unchanged.
        """
        new = copy.copy(self)
        for name, value in changes.items():
            setattr(new, name, int(value))
        return new

    @classmethod
    def for_shard(
        cls,
        *,
        leaf_capacity: int,
        ndev: int,
        leaf_width: int,
        live_leaves: Optional[int] = None,
        summary_cell_level: Optional[int] = None,
    ) -> "CrossCapacities":
        """A priori capacities for a mesh of ``ndev`` shards of ``leaf_capacity`` leaves.

        The summary is bounded EXACTLY by the tree: under the default one-leaf cells
        there is one cut cell per live leaf, at most ``leaf_capacity``, and the summary
        tree is part of a tree of ``2 x leaf_capacity - 1`` nodes. The exchange buffers
        scale with the live leaves of the worst shard times the peers it can hear from,
        by the per-leaf demand the two-card probe measured, with its headroom
        (:data:`_A_PRIORI_PER_LEAF`). Those factors come from Plummer spheres; for
        anything else this is a first guess for :func:`plan_cross_capacities` to
        measure and right-size, not a setting.

        Parameters
        ----------
        leaf_capacity : int
            Leaf slots of every shard's tree (the template's leaf capacity).
        ndev : int
            Mesh size.
        leaf_width : int
            Most particles in one leaf (the lane's leaf size).
        live_leaves : Optional[int]
            Live leaves of the worst shard; ``None`` uses ``leaf_capacity``.
        summary_cell_level : Optional[int]
            Passed through (:class:`CrossCapacities`).

        Returns
        -------
        CrossCapacities
            The capacities; the one-sided and CSR exchange's keep their defaults.
        """
        leaves = int(leaf_capacity if live_leaves is None else live_leaves)
        peers = max(int(ndev) - 1, 1)
        sized = {
            name: _sized(per_leaf * leaves * peers, headroom)
            for name, (per_leaf, headroom) in _A_PRIORI_PER_LEAF.items()
        }
        return cls(
            max_cells=int(leaf_capacity),
            max_summary_nodes=2 * int(leaf_capacity) - 1,
            leaf_width=int(leaf_width),
            summary_cell_level=summary_cell_level,
            **sized,
        )

    @classmethod
    def from_needs(
        cls,
        needs: Any,
        *,
        like: Optional["CrossCapacities"] = None,
        headroom: Optional[float] = None,
        leaf_capacity: Optional[int] = None,
        leaf_size: Optional[int] = None,
    ) -> "CrossCapacities":
        """Capacities sized from a measured needs vector, wider or narrower than before.

        Every capacity :data:`CROSS_NEEDS_FIELDS` names becomes ``headroom x`` what it
        needed, rounded up to :data:`_CROSS_CAP_GRANULE` entries; the leaf width
        becomes the widest shipped leaf, never below ``like``'s. Read a vector from a
        run whose ``overflow`` bit is CLEAR: a lower-bound need sizes a buffer that
        overflows again (:func:`plan_cross_capacities` re-plans first). Whatever a
        need says, the structural bounds hold (:func:`_structural_bounds`): the leaf
        width at most ``leaf_size``, the particle buffers at most their rows x the
        leaf width.

        Parameters
        ----------
        needs : Any
            The needs vector, reduced over the mesh (host).
        like : Optional[CrossCapacities]
            Where every capacity the symmetric exchange does not read comes from
            (the one-sided and CSR controls'), with ``summary_cell_level`` and
            ``max_leaves_per_cell``; ``None``: the defaults.
        headroom : Optional[float]
            Factor over each need; ``None``: ``JACCPOT_FLAT_WALK_CAP_HEADROOM`` (1.5),
            the local walk's.
        leaf_capacity : Optional[int]
            The tree's leaf slots: the summary is clamped to what a tree of them can
            hold (``max_cells`` to it, ``max_summary_nodes`` to ``2 x it - 1``).
        leaf_size : Optional[int]
            The most particles a leaf of the template holds (its leaf size); the
            leaf width never exceeds it. ``None``: no bound on the width.

        Returns
        -------
        CrossCapacities
            The capacities.
        """
        base = like if like is not None else cls()
        need = _needs_by_field(needs)
        h = _default_headroom(headroom)
        changes = {
            name: _clamp_to_tree(name, _sized(need[name], h), leaf_capacity)
            for name in _NEEDS_LOWER_BOUND_BIT
            if name != "leaf_width"
        }
        changes["leaf_width"] = max(int(base.leaf_width), need["leaf_width"])
        return base._with(**_structural_bounds(changes, leaf_size))

    def replanned(
        self,
        needs: Any,
        *,
        headroom: Optional[float] = None,
        leaf_capacity: Optional[int] = None,
        leaf_size: Optional[int] = None,
    ) -> "CrossCapacities":
        """Capacities that cover what a run needed: the segment retry's re-plan.

        Monotone: every capacity stays at least what it was, as the local walk's
        re-plan (``engine._replan_walk_caps_from_needs``). A capacity grows to
        ``headroom x`` its need; one whose need is only a LOWER bound (the walk or the
        summary lost pairs: ``walk_lower_bound`` / ``pairs_lower_bound``) and that
        overflowed at least doubles, as the local walk's queue does. A lower bound can
        still fall short, so a run that overflowed its walk may need a second re-plan:
        its re-run's needs are then exact. No need grows a capacity past its
        structural bound (:func:`_structural_bounds`), and none narrows one.

        Parameters
        ----------
        needs : Any
            The failed run's needs vector, reduced over the mesh: a running maximum
            over the steps that STARTED on the trajectory (a step after the first
            failure runs on wrong forces, and what it reads is no demand).
        headroom : Optional[float]
            Factor over each need; ``None``: ``JACCPOT_FLAT_WALK_CAP_HEADROOM``.
        leaf_capacity : Optional[int]
            As :meth:`from_needs`; pass the GROWN one when the tree was re-planned.
        leaf_size : Optional[int]
            As :meth:`from_needs`: the leaf width never grows past it.

        Returns
        -------
        CrossCapacities
            The grown capacities (equal to these when nothing needed more).
        """
        need = _needs_by_field(needs)
        h = _default_headroom(headroom)
        changes = {}
        for name, bit in _NEEDS_LOWER_BOUND_BIT.items():
            cap = int(getattr(self, name))
            if name == "leaf_width":
                changes[name] = max(cap, need[name])
                continue
            grown = max(cap, _clamp_to_tree(name, _sized(need[name], h), leaf_capacity))
            if bit is not None and need[bit] and need[name] > cap:
                grown = max(grown, 2 * cap)
            if name == "export_walk_queue" and need["walk_lower_bound"]:
                # The walk stopped with pairs still queued, and the peak it reports
                # is the widest wavefront it ADMITTED, which can read below the
                # queue it overflowed (measured: the disc + bulge at 2e5 on two
                # A100s, a peak of 252,788 read off a queue of 389,120 that
                # overflowed; at 4x every capacity the walk ran clean, peak
                # 375,310): the queue doubles whatever the peak says, as the local
                # walk's queue does on a queue overflow.
                grown = max(grown, 2 * cap)
            changes[name] = grown
        bounded = _structural_bounds(changes, leaf_size)
        # monotone: a bound below the present width keeps the width
        return self._with(
            **{name: max(int(getattr(self, name)), v) for name, v in bounded.items()}
        )


def _structural_bounds(caps: dict, leaf_size: Optional[int]) -> dict:
    """Clamp proposed capacities to what the exchange can structurally need.

    * ``leaf_width`` <= ``leaf_size``: a near row is one summary cell holding one live
      cell leaf, and a cell leaf holds at most the leaf size the template was built
      with -- the local lane's own leaf tiles are that wide. A wider "need" is not a
      leaf (a degenerate tree: measured, the root of a NaN state as one summary cell
      of 4,002,489 particles) and no width serves it.
    * ``send_particle_cap`` <= ``send_node_cap x leaf_width``, ``recv_particle_cap``
      <= ``recv_node_cap x leaf_width``: every near row ships at most ``leaf_width``
      particles (the sender clips to it and flags the rest).

    Parameters
    ----------
    caps : dict
        Proposed values of the :data:`CROSS_NEEDS_FIELDS` capacities.
    leaf_size : Optional[int]
        The template's leaf size; ``None`` leaves the width alone.

    Returns
    -------
    dict
        The clamped values.
    """
    out = dict(caps)
    if leaf_size is not None:
        out["leaf_width"] = min(int(out["leaf_width"]), int(leaf_size))
    width = int(out["leaf_width"])
    for particles, rows in (
        ("send_particle_cap", "send_node_cap"),
        ("recv_particle_cap", "recv_node_cap"),
    ):
        out[particles] = min(int(out[particles]), int(out[rows]) * width)
    return out


def _clamp_to_tree(name: str, size: int, leaf_capacity: Optional[int]) -> int:
    """A summary capacity clamped to what a tree of ``leaf_capacity`` leaves holds.

    Parameters
    ----------
    name : str
        The capacity (:data:`CROSS_NEEDS_FIELDS`).
    size : int
        Its proposed size.
    leaf_capacity : Optional[int]
        The tree's leaf slots; ``None`` clamps nothing.

    Returns
    -------
    int
        ``size``, or the tree's bound when that is smaller.
    """
    if leaf_capacity is None:
        return int(size)
    bound = {
        "max_cells": int(leaf_capacity),
        "max_summary_nodes": 2 * int(leaf_capacity) - 1,
    }
    return min(int(size), bound.get(name, int(size)))


def plan_cross_capacities(
    measure: Callable[[CrossCapacities], Any],
    initial: CrossCapacities,
    *,
    headroom: Optional[float] = None,
    leaf_capacity: Optional[int] = None,
    leaf_size: Optional[int] = None,
    max_rounds: int = 3,
) -> tuple[CrossCapacities, np.ndarray]:
    """Size the cross capacities from a force measured at the start positions.

    ``measure(caps)`` runs one force with ``caps`` and returns its needs vector
    reduced over the mesh (``make_fused_force_evaluator(cross_needs=True)``). While
    its ``overflow`` bit is set the capacities are re-planned from it
    (:meth:`CrossCapacities.replanned`) and measured again; once a measurement is
    clean every capacity is sized from it (:meth:`CrossCapacities.from_needs`), which
    is usually NARROWER than the first guess. Each measurement with new capacities is
    a compile.

    Parameters
    ----------
    measure : Callable[[CrossCapacities], Any]
        Runs a force and returns the host needs vector.
    initial : CrossCapacities
        The first guess (:meth:`CrossCapacities.for_shard`).
    headroom : Optional[float]
        Factor over each need; ``None``: ``JACCPOT_FLAT_WALK_CAP_HEADROOM``.
    leaf_capacity : Optional[int]
        The tree's leaf slots, which bound the summary exactly.
    leaf_size : Optional[int]
        The template's leaf size, which bounds the leaf width.
    max_rounds : int
        Measurements at most.

    Returns
    -------
    tuple[CrossCapacities, np.ndarray]
        ``(capacities, the clean measurement's needs)``.

    Raises
    ------
    RuntimeError
        When a measurement overflowed and no capacity needed more (the two ends of
        an exchange disagreed: ``mismatch``), or still overflowed after
        ``max_rounds``.
    """
    caps = initial
    for _ in range(max(int(max_rounds), 1)):
        needs = np.asarray(measure(caps)).astype(np.int64)
        need = _needs_by_field(needs)
        if not need["overflow"]:
            caps = CrossCapacities.from_needs(
                needs,
                like=caps,
                headroom=headroom,
                leaf_capacity=leaf_capacity,
                leaf_size=leaf_size,
            )
            return caps, needs
        grown = caps.replanned(
            needs, headroom=headroom, leaf_capacity=leaf_capacity, leaf_size=leaf_size
        )
        if vars(grown) == vars(caps):
            raise RuntimeError(
                "plan_cross_capacities: the cross flag fired and no capacity needed "
                f"more ({need}); a mismatch between the two ends of an exchange is "
                "not a capacity."
            )
        caps = grown
    raise RuntimeError(
        f"plan_cross_capacities: still overflowing after {max_rounds} measurements "
        f"({need}, capacities {caps})"
    )


def _with_separation_floor(
    walk_fn: Optional[Callable[..., Any]], floor: float
) -> Optional[Callable[..., Any]]:
    """``walk_fn`` with the separation floor applied to every call.

    Every cross walk -- export, receiver, near import -- runs through ``walk_fn``
    (or yggdrax's mutual walk when it is ``None``), so the floor is applied once,
    here, to all of them.

    Parameters
    ----------
    walk_fn : Optional[Callable[..., Any]]
        The walk (``None``: yggdrax's).
    floor : float
        Minimum gap of an accepted far pair; ``0`` leaves ``walk_fn`` as it is.

    Returns
    -------
    Optional[Callable[..., Any]]
        The wrapped walk.
    """
    if floor <= 0.0:
        return walk_fn
    from yggdrax.interactions import dual_tree_walk_mutual

    inner_walk = walk_fn if walk_fn is not None else dual_tree_walk_mutual

    def floored_walk(*args: Any, **kwargs: Any) -> Any:
        return inner_walk(*args, separation_floor=floor, **kwargs)

    return floored_walk


def make_cross_hook(
    *,
    ndev: int,
    theta: float,
    caps: Optional[CrossCapacities] = None,
    mac_type: str = "dehnen",
    axis_name: str = AXIS_NAME,
    record: Optional[dict] = None,
    near_sink: Optional[dict] = None,
    near_theta: Optional[float] = None,
    export_theta: Optional[float] = None,
    two_sided: Optional[bool] = None,
    separation_floor: float = 0.0,
    error_criterion: bool = False,
) -> Callable[..., Optional[tuple]]:
    """Build the ``cross_hook`` for a mesh of ``ndev`` devices.

    Parameters
    ----------
    ndev:
        Mesh size. At 1 the hook still runs everything and returns an empty import.
    theta:
        MAC parameter -- must match what the local lane evaluates with, since the
        sender's export decisions and the receiver's expansion both use it.
    caps:
        Static capacities; see :class:`CrossCapacities`.
    mac_type:
        MAC variant for the export and receiver walks: a geometric one (``"bh"``,
        ``"dehnen"``, ``"engblom"``). Static.
    axis_name:
        Mesh axis; must match the enclosing ``shard_map``.
    record:
        Optional dict the hook writes diagnostics into (call count, import sizes,
        overflow flags). For probes -- it is host state, so it holds tracers under
        jit and concrete values only when the hook runs eagerly.
    near_sink:
        Optional dict the hook writes the imported NEAR payload into -- the leaf
        pool and the (local target leaf, imported leaf) pairs. The caller evaluates
        that term after the force and adds it; see the module docstring for why the
        near half is added rather than interleaved. Holds tracers under jit, so it
        must be consumed inside the same trace.
    near_theta:
        MAC parameter of the receiver's walk over the NEAR import; ``None`` uses
        ``theta``. The near-imported leaves travel WITH their multipoles, so the
        pairs this walk calls far go through the M2L (as a second imported block
        behind the far import) and the pairs it calls near are summed directly.
        ``0.0`` forces every pair into the direct sum: the same answer to expansion
        accuracy, at the price of ~90k direct pairs per device at N = 2e5 -- the
        control for the multipole route, not a setting.
    export_theta:
        MAC parameter of the SENDER's export walk alone; ``None`` uses ``theta``.
        A BISECTION knob, not a setting: ``0.0`` makes every exported pair bottom
        out at a sender leaf, so the far list is empty and the whole cross field
        travels as particles and is summed directly (with ``near_theta=0``). If
        the error then matches the single-GPU lane, the residual lives in the
        multipole path (import, M2L, cascade); if it does not, it lives elsewhere.
    two_sided:
        Publish each device's summary TREE (the cut plus its ancestors, with child
        links) and walk both trees in the export, so far pairs can land on the
        receiver's internal nodes instead of every cell collecting its own list.
        ``None`` reads ``JACCPOT_CROSS_TWO_SIDED`` (default on). Static.

    separation_floor:
        Minimum gap of an accepted far pair in every export and receiver walk
        (length units); a compact softening kernel's support, so the unsoftened
        cross far field is exact. ``0`` (default): none. Static.
    error_criterion:
        Dehnen's eq (16a) (``mac_type='dehnen_error'`` on the solver, S12): the
        cross walks accept by the criterion the local walk runs, from each
        published node's table row (``G M``, ``eps * min_b f_b`` of its subtree,
        ``s_1..s_p``; extra columns of the summary all_gather), so both devices of
        a pair evaluate the identical test on the identical floats; the far
        payload carries each node's mass for eq (16b)'s far half. The hook is then
        called ``hook(tree_artifacts, criterion=FlatWalkCriterion)`` and returns
        ``(multipoles, centers, src, tgt, masses)``
        (``FastMultipoleMethod``'s refresh does both, ``_call_cross_hook``).
        Symmetric exchange only (the default path). ``mac_type`` stays the
        geometric name (``theta`` is then unused by the walks). Static.

    Returns
    -------
    Callable
        ``hook(tree_artifacts) -> (multipoles, centers, src, tgt) | None``, with the
        imported sources indexed from ``n_local`` (with ``error_criterion``, see
        there).

    Raises
    ------
    NotImplementedError
        For ``mac_type`` ``"dehnen_error"`` / ``"dehnen_theta"``: the hook's MAC
        name is the walks' geometric one; the criterion is ``error_criterion``.
        And for ``error_criterion`` on any exchange but the symmetric one (a
        control: one-sided, CSR, tiles, a receiver walk, a separate near or export
        theta) or the ``aabb`` MAC geometry control: their walks re-test imported
        rows that carry no eq (16a) table, and the far half's f_b needs the
        expansion centres to be the walk centres.
    """
    if str(mac_type) in ("dehnen_error", "dehnen_theta"):
        raise NotImplementedError(
            f"make_cross_hook: mac_type={mac_type!r} is not a cross-walk MAC: the "
            "cross walks take the geometric name (mac_type='dehnen'), and Dehnen's "
            "error criterion (eq 16a, dehnen_error) through error_criterion=True, "
            "with each step's per-node force scale (FusedMeshForce and "
            "MeshSegmentRunner.fused build it so). dehnen_theta is not supported "
            "on the fused multi-GPU lane; the mesh lane (jaccpot.distributed.fmm) "
            "runs the criteria too."
        )
    cap = caps if caps is not None else CrossCapacities()
    floor = float(separation_floor)
    walk_fn = _with_separation_floor(cross_walk_fn(mac_type), floor)

    two_sided = _cross_two_sided() if two_sided is None else bool(two_sided)
    symmetric = two_sided and _symmetric_exchange_enabled(
        two_sided=two_sided,
        max_leaves=_max_leaves_per_cell(cap, two_sided),
        theta=theta,
        near_theta=near_theta,
        export_theta=export_theta,
    )
    if error_criterion and not symmetric:
        raise NotImplementedError(
            "make_cross_hook(error_criterion=True): Dehnen's eq (16a) runs on the "
            "symmetric exchange only (two-sided over leaves, no receiver walk, one "
            "theta; the default). This configuration selects a control exchange "
            "(JACCPOT_CROSS_SYMMETRIC=0, JACCPOT_CROSS_TWO_SIDED=0, "
            "JACCPOT_CROSS_NEAR_TILES=1, a receiver walk forced on, "
            "max_leaves_per_cell != 1, or a near/export theta of its own), whose "
            "walks re-test imported rows that carry no eq (16a) table."
        )
    if error_criterion and _cross_mac_geometry_mode() != "com":
        raise NotImplementedError(
            "make_cross_hook(error_criterion=True) needs the COM MAC geometry "
            "(JACCPOT_CROSS_MAC_GEOMETRY=com, the default): eq (16a)'s table and "
            "eq (16b)'s far terms are about the expansion centres."
        )
    # ALWAYS-ON overflow channel, independent of the diagnostics `record`: the
    # production and timing paths pass no record, and before this sink existed the
    # far half's flags then reached nobody. Holds tracers; the evaluator reads it
    # inside the same trace (`hook.flag_sink`) and ORs it into the mesh flag.
    flag_sink: dict = {}
    # ALWAYS-ON needs channel beside it (`cross_needs`): what every capacity of the
    # symmetric exchange needed in this trace, so a caller whose flag fired can
    # re-plan the capacities from exact demand (`CrossCapacities.replanned`)
    needs_sink: dict = {}

    def hook(tree_artifacts: Any, criterion: Any = None) -> Optional[tuple]:
        # stage labels for per-device traces (`jax.named_scope` costs nothing at
        # run time; it only names the ops). Every stage call below adds its own.
        with jax.named_scope("cross_hook"):
            return _hook_body(tree_artifacts, criterion)

    def _hook_body(tree_artifacts: Any, criterion: Any) -> Optional[tuple]:
        # a RE-TRACE must not see the previous trace's tracers
        flag_sink.clear()
        needs_sink.clear()
        if bool(error_criterion) != (criterion is not None):
            raise RuntimeError(
                "the cross hook was built with error_criterion="
                f"{bool(error_criterion)} and called "
                f"{'with' if criterion is not None else 'without'} a criterion"
            )
        tree = tree_artifacts.tree
        upward = tree_artifacts.upward
        mp = upward.multipoles
        # THE GEOMETRY THE MAC IS TESTED ON MUST BE THE ONE THE EXPANSIONS USE.
        # The lane's real-basis sweep expands about the COM (`center_mode='com'`
        # only) and its own walk tests the MAC about those centres with the exact
        # particle radii about them (`resolve_walk_geometry`, default "com" in the
        # strict fused lane -- plan sub-10ms Phase 1.2). This exchange used
        # `upward.geometry` -- AABB centres, box radii -- so the sender accepted
        # pairs that are admissible about the box centre and DIVERGENT about the
        # COM the receiver's M2L actually expands from. Divergent pairs do not
        # improve with order: measured as a floor near 1.4e-03 the reference lane
        # passes through (record, Task 1), while routing the whole cross field
        # through direct sums removed it (6.0e-04 at p6). Same defect class as
        # memory `mac-geometry-inconsistent-with-com-centres`, one level up.
        # `JACCPOT_CROSS_MAC_GEOMETRY=aabb` keeps the old behaviour as a control.
        box_geom = upward.geometry
        shared = getattr(tree_artifacts, "walk_geometry", None)
        if _cross_mac_geometry_mode() == "com" and shared is not None:
            # the refresh resolved the local walk's geometry once for both of us
            geom = shared[0] if shared[0] is not None else box_geom
        elif _cross_mac_geometry_mode() == "com":
            from jaccpot.runtime._mac_geometry import resolve_walk_geometry

            geom, _ = jax.named_call(resolve_walk_geometry, name="cross_geometry")(
                tree,
                tree_artifacts.positions_sorted,
                box_geom,
                getattr(mp, "centers", None),
                leaf_cap=int(tree_artifacts.leaf_cap),
                default_mode="com",
            )
            if geom is None:
                geom = box_geom
        else:
            geom = box_geom

        parent = jnp.asarray(tree.parent)
        n_local = int(jnp.asarray(mp.packed).shape[0])
        num_internal = int(jnp.asarray(tree.left_child).shape[0])

        # what every device publishes -- the cut cells, or (two-sided) the summary
        # tree over them, with eq (16a)'s table rows under the criterion -- packed
        # so that ONE all_gather carries it
        pub = _summary_rows(tree, geom, cap, two_sided=two_sided, criterion=criterion)
        summary = pub.cut
        block_nodes = pub.block_nodes
        block = int(block_nodes.shape[0])
        gathered = jax.lax.all_gather(pub.rows, axis_name, tiled=False)
        all_cen = gathered[..., 0:3]
        all_rad = gathered[..., 3]
        me = jax.lax.axis_index(axis_name)

        if symmetric:
            order = 0 if criterion is None else int(criterion.order)
            sym = _symmetric_exchange(
                gathered,
                me,
                block_nodes,
                tree,
                mp,
                theta=float(theta),
                cap=cap,
                ndev=ndev,
                mac_type=mac_type,
                walk_fn=(
                    walk_fn
                    if criterion is None
                    else _with_separation_floor(
                        cross_walk_fn(
                            mac_type,
                            error_order=order,
                            theta_max=float(criterion.theta_max),
                        ),
                        floor,
                    )
                ),
                axis_name=axis_name,
                with_near=near_sink is not None,
                error_order=order,
                node_mass=None if criterion is None else upward.mass_moments.mass,
            )
            far_overflow = pub.overflow | sym["far_overflow"]
            flag_sink["far"] = far_overflow
            if near_sink is not None:
                for k in ("positions", "masses", "mask", "target_node", "source_row"):
                    near_sink[k] = sym[k]
                near_sink["count"] = sym["count"]
                near_sink["overflow"] = pub.overflow | sym["overflow"]
                flag_sink["near"] = near_sink["overflow"]
            cells_needed, nodes_needed = _summary_needs(tree, cap, two_sided=True)
            needs = dict(
                sym["needs"],
                max_cells=cells_needed,
                max_summary_nodes=nodes_needed,
                overflow=far_overflow | flag_sink.get("near", False),
            )
            # a truncated summary hides part of a tree from every walk over it
            for bit in ("walk_lower_bound", "pairs_lower_bound"):
                needs[bit] = needs[bit] | pub.overflow
            needs_sink["needs"] = jnp.stack(
                [
                    jnp.asarray(needs.get(k, 0)).astype(jnp.int32)
                    for k in CROSS_NEEDS_FIELDS
                ]
            )
            if record is not None:
                w = sym["walk"]
                for k, v in zip(CROSS_NEEDS_FIELDS, needs_sink["needs"]):
                    record[f"need_{k}"] = v
                record["summary_cells"] = summary.num_cells
                record["summary_nodes"] = pub.num_entries
                record["summary_leaves"] = jnp.sum(summary.leaves_per_cell)
                record["export_far"] = w.far_count
                record["far_pairs"] = w.far_count
                record["send_nodes"] = jnp.sum(sym["exp_far"].sizes)
                record["recv_nodes"] = jnp.sum(sym["imp_far"].sizes)
                record["imported_rows"] = jnp.sum(sym["imp_far"].sizes)
                if getattr(w, "peak_wavefront", None) is not None:
                    record["export_walk_peak"] = w.peak_wavefront
                record["overflow"] = far_overflow
                if near_sink is not None:
                    record["export_near"] = w.near_count
                    record["near_list_pairs"] = w.near_count
                    record["near_csr"] = w.near_count
                    record["near_recv_nodes"] = jnp.sum(sym["imp_near"].sizes)
                    record["near_send_nodes"] = jnp.sum(sym["exp_near"].sizes)
                    record["near_particles"] = sym["near_particles"]
                    record["send_particles"] = sym["send_particles"]
                    record["near_walk_peak"] = jnp.asarray(0, jnp.int32)
                    record["near_walk_far_pairs"] = jnp.asarray(0, jnp.int32)
                # the walk's pairs and what it read, for a check of each device's
                # view of every cross pair (tests/distributed/
                # test_mesh_segment_dehnen_error.py)
                record["gathered"] = gathered
                for side in ("far_a", "far_b", "far_count", "near_a", "near_b"):
                    record[f"walk_{side}"] = getattr(w, side)
                record["walk_near_count"] = w.near_count
                # each imported far row's sender node (in the sender's summary
                # index, rows grouped by peer) and, under the criterion, its mass
                record["imp_far_rows"] = sym["imp_far"].rows
                record["imp_far_sizes"] = sym["imp_far"].sizes
                if criterion is not None:
                    record["cross_masses"] = sym["cross_masses"]
            if criterion is not None:
                return sym["cross_far"] + (sym["cross_masses"],)
            return sym["cross_far"]

        idx = parent.dtype
        ranges = jnp.asarray(tree.node_ranges)
        node_active = ranges[:, 1] >= ranges[:, 0]
        leaf_fill = jnp.full((n_local - num_internal,), -1, idx)
        left = jnp.concatenate([jnp.asarray(tree.left_child, idx), leaf_fill])
        right = jnp.concatenate([jnp.asarray(tree.right_child, idx), leaf_fill])

        ex = _export_from_rows(
            gathered,
            left,
            right,
            geom,
            jnp.argmin(parent).astype(idx),
            float(theta if export_theta is None else export_theta),
            me,
            cap,
            two_sided=two_sided,
            mac_type=mac_type,
            walk_fn=walk_fn,
            # a cells tree pads its leaves to a static capacity with EMPTY leaves in
            # one subtree; the walk must leave those nodes alone
            node_active=node_active,
        )

        if record is not None:
            # Recompute the REAL MAC on the pairs the sender emitted, using the
            # sender's own arrays. This is the control that separates "export_walk
            # emits inadmissible pairs" from "the geometry changes in transit":
            # the receiver runs the identical check on the identical pairs below.
            from yggdrax._interactions_impl import _compute_mac_ok

            mc = block
            fc = jnp.asarray(ex.far_cell)
            fn = jnp.asarray(ex.far_node)
            f_live = (jnp.arange(fc.shape[0]) < ex.far_count) & (fc >= 0)
            f_dev = jnp.where(f_live, fc // mc, 0)
            f_slot = jnp.where(f_live, fc % mc, 0)
            f_safe = jnp.where(f_live, fn, 0)
            d_s = all_cen[f_dev, f_slot] - jnp.asarray(geom.center)[f_safe]
            d2_s = jnp.sum(d_s * d_s, axis=-1)
            ok_s = _compute_mac_ok(
                mac_type=mac_type,
                theta_sq=jnp.asarray(float(theta) ** 2, d2_s.dtype),
                dist_sq=d2_s,
                extent_target=all_rad[f_dev, f_slot],
                extent_source=jnp.asarray(geom.radius)[f_safe],
                valid_pairs=f_live,
                different_nodes=jnp.ones_like(f_live),
                separation_floor=floor,
            )
            # Walk centres vs expansion centres. Under the COM geometry these are
            # the SAME array and both numbers must read 0 -- that is the check the
            # geometry fix is in place. Under `aabb` they differed on 11116/16383
            # nodes by up to 180 units, which is why both centres travel.
            _gc = jnp.asarray(geom.center)
            _mc_ = jnp.asarray(mp.centers)
            record["center_mismatch"] = jnp.sum(jnp.any(_gc != _mc_, axis=-1))
            record["center_max_delta"] = jnp.max(jnp.abs(_gc - _mc_))
            record["export_far_live"] = jnp.sum(f_live)
            record["export_mac_fail"] = jnp.sum(f_live & ~ok_s)

            # CONTROL. The near list is what BOTTOMED OUT -- those pairs failed the
            # MAC, which is why they are near. If this recomputation is meaningful
            # it must reject most of them while accepting the far list. If both
            # come out the same the recomputation is wrong, not the walk.
            nc = jnp.asarray(ex.near_cell)
            nn = jnp.asarray(ex.near_node)
            n_live = (jnp.arange(nc.shape[0]) < ex.near_count) & (nc >= 0)
            n_dev = jnp.where(n_live, nc // mc, 0)
            n_slot = jnp.where(n_live, nc % mc, 0)
            n_safe = jnp.where(n_live, nn, 0)
            d_n = all_cen[n_dev, n_slot] - jnp.asarray(geom.center)[n_safe]
            d2_n = jnp.sum(d_n * d_n, axis=-1)
            ok_n = _compute_mac_ok(
                mac_type=mac_type,
                theta_sq=jnp.asarray(float(theta) ** 2, d2_n.dtype),
                dist_sq=d2_n,
                extent_target=all_rad[n_dev, n_slot],
                extent_source=jnp.asarray(geom.radius)[n_safe],
                valid_pairs=n_live,
                different_nodes=jnp.ones_like(n_live),
                separation_floor=floor,
            )
            record["export_near_live"] = jnp.sum(n_live)
            record["export_near_mac_fail"] = jnp.sum(n_live & ~ok_n)

        sb = jax.named_call(build_send_buffers, name="cross_send_far")(
            ex.far_cell,
            ex.far_node,
            ex.far_count,
            ndev=ndev,
            max_cells=block,
            num_nodes=n_local,
            node_capacity=cap.send_node_cap,
            csr_capacity=cap.send_csr_cap,
        )
        rows = jnp.clip(sb.node_rows, 0, n_local - 1)
        alive = (sb.node_rows >= 0)[:, None]
        # The RADIUS travels with the multipole. Without it the receiver has to
        # invent one, and the only available default -- zero -- makes every imported
        # node pass the MAC at any distance: multipoles get used at close range and
        # the near list is starved of exactly the pairs that carry the largest
        # forces. The symptom is a cross field that reaches the force and still
        # leaves it badly wrong.
        payload = jnp.concatenate(
            [
                jnp.where(alive, jnp.asarray(mp.packed)[rows], 0.0),
                jnp.where(alive, jnp.asarray(mp.centers)[rows], 0.0),
                # BOTH centres travel, because they are different points and the
                # two consumers need different ones. `mp.centers` is the expansion
                # centre and belongs to the M2L; `geom.center` is the geometric
                # centre the MAC was computed with. Shipping only the first made
                # the receiver re-test the MAC at a point the sender never used,
                # so pairs the sender had accepted came back inadmissible -- and
                # `combined_cen` was then geometric for local nodes and expansion
                # for imported ones, inside the one array the walk reads.
                jnp.where(alive, jnp.asarray(geom.center)[rows], 0.0),
                jnp.where(alive, jnp.asarray(geom.radius)[rows][:, None], 0.0),
            ],
            axis=1,
        )

        got = jax.named_call(exchange_export_list, name="cross_exchange_far")(
            payload,
            sb.node_sizes,
            sb.csr_cell,
            sb.csr_row,
            sb.csr_sizes,
            payload_capacity=cap.recv_node_cap,
            csr_capacity=cap.recv_csr_cap,
            ndev=ndev,
            axis_name=axis_name,
        )

        n_coeff = int(jnp.asarray(mp.packed).shape[1])
        imp_mp = got.payload[:, :n_coeff]
        imp_cen = got.payload[:, n_coeff : n_coeff + 3]  # expansion -> M2L
        imp_gcen = got.payload[:, n_coeff + 3 : n_coeff + 6]  # geometric -> MAC
        imp_rad = got.payload[:, n_coeff + 6]

        combined_left = jnp.concatenate([left, jnp.full((cap.recv_node_cap,), -1, idx)])
        combined_right = jnp.concatenate(
            [right, jnp.full((cap.recv_node_cap,), -1, idx)]
        )
        # the WALK's geometry: geometric centres on both halves
        combined_cen = jnp.concatenate([jnp.asarray(geom.center), imp_gcen])
        combined_rad = jnp.concatenate([jnp.asarray(geom.radius), imp_rad])

        if record is not None:
            from yggdrax._interactions_impl import _compute_mac_ok as _mac_ok

            sc = jnp.asarray(got.csr_cell)
            sr = jnp.asarray(got.csr_row)
            s_live = (jnp.arange(sc.shape[0]) < got.num_csr) & (sc >= 0)
            sa = jnp.where(s_live, block_nodes[jnp.where(s_live, sc, 0)], 0)
            sb_ = jnp.where(s_live, n_local + sr, 0)
            d_r = combined_cen[sa] - combined_cen[sb_]
            d2_r = jnp.sum(d_r * d_r, axis=-1)
            ok_r = _mac_ok(
                mac_type=mac_type,
                theta_sq=jnp.asarray(float(theta) ** 2, d2_r.dtype),
                dist_sq=d2_r,
                extent_target=combined_rad[sa],
                extent_source=combined_rad[sb_],
                valid_pairs=s_live,
                different_nodes=jnp.ones_like(s_live),
                separation_floor=floor,
            )
            record["seed_live"] = jnp.sum(s_live)
            record["seed_mac_fail"] = jnp.sum(s_live & ~ok_r)

        if _far_receiver_walk_needed(export_theta, theta):
            rl = jax.named_call(receiver_interaction_lists, name="cross_recv_walk_far")(
                combined_left,
                combined_right,
                combined_cen,
                combined_rad,
                n_local,
                block_nodes,
                got.csr_cell,
                got.csr_row,
                got.num_csr,
                float(theta),
                max_pair_queue=cap.walk_queue,
                far_cap=cap.recv_far_cap,
                near_cap=cap.recv_near_cap,
                mac_type=mac_type,
                walk_fn=walk_fn,
            )
        else:
            # The far receiver walk is a PASS-THROUGH by construction: each seed is
            # (my cell's root, a node the sender ACCEPTED against that very cell),
            # tested on the same centre and radius the sender used -- the cell's came
            # from my summary, the node's travelled in the payload. Measured at 2e6 on
            # two cards: far pairs == received CSR entries (5,083,648 / 4,640,119),
            # zero near pairs. So map the CSR directly and skip the walk, whose seed
            # alone forced a queue of 2 x recv_csr_cap. JACCPOT_CROSS_FAR_RECEIVER_WALK=1
            # runs the walk (the control; `export_theta` != theta always does).
            rl = _direct_far_lists(block_nodes, got)

        # ---- the NEAR half: ship the exported leaves' PARTICLES -------------
        if near_sink is not None:
            W = int(cap.leaf_width)
            nr = jnp.asarray(tree.node_ranges)
            pos_sorted = jnp.asarray(tree.positions_sorted)
            mass_sorted = jnp.asarray(tree.masses_sorted)

            sb_n = jax.named_call(build_send_buffers, name="cross_send_near")(
                ex.near_cell,
                ex.near_node,
                ex.near_count,
                ndev=ndev,
                max_cells=block,
                num_nodes=n_local,
                node_capacity=cap.send_node_cap,
                csr_capacity=cap.send_csr_cap,
            )
            # a leaf's particles are a contiguous run [start, end]
            lrow = jnp.clip(sb_n.node_rows, 0, n_local - 1)
            starts = nr[lrow, 0]
            ends = nr[lrow, 1]
            okrow = (sb_n.node_rows >= 0)[:, None]
            # A leaf holding more than W particles would lose the excess here with
            # no other trace of it -- a MISSING force that every invariant passes.
            # W comes from a capacity, so this is a real possibility, not a
            # theoretical one, and it has to surface as an overflow.
            tile_truncated = ((sb_n.node_rows >= 0) & (ends - starts + 1 > W)).any()
            # The near import ships its OWN geometry. The far and near halves export
            # DIFFERENT node sets -- far sends internal nodes, near sends leaves --
            # so payload row k means a different node in each. Scoring the near walk
            # with the far import's centres, as this first did, pairs every imported
            # leaf's particles with an unrelated node's geometry: a wrong force that
            # looks entirely plausible.
            # The leaf's MULTIPOLE travels with its particles. The sender decided
            # (cell, leaf) is near at the granularity of the cell; the receiver
            # refines the cell down to its own leaves and, for most of those, the
            # leaf-leaf pair passes the MAC after all (72 % of the near walk at
            # N = 2e5: `near_walk_far_pairs`). Without a multipole behind the
            # imported leaf those pairs were unservable -- neither list had them --
            # and the only correct answer was near_theta = 0, every one a direct
            # sum. With the coefficients here they go back through the M2L, as a
            # second imported block behind the far one (Task 2 of the record).
            near_walk = _near_receiver_walk_needed(
                two_sided=two_sided,
                max_leaves=_max_leaves_per_cell(cap, two_sided),
                theta=theta,
                near_theta=near_theta,
                export_theta=export_theta,
            )
            # Without the receiver walk nothing reads the near rows' geometry or
            # multipoles (no near-walk far pairs reach the M2L), so only the particles
            # and their per-row counts travel: 52 floats per row fewer at p6.
            leaf_rows = (
                [
                    jnp.where(okrow, jnp.asarray(geom.center)[lrow], 0.0),
                    jnp.where(okrow, jnp.asarray(geom.radius)[lrow][:, None], 0.0),
                    jnp.where(okrow, jnp.asarray(mp.packed)[lrow], 0.0),
                    jnp.where(okrow, jnp.asarray(mp.centers)[lrow], 0.0),
                ]
                if near_walk or _near_tiles_on_the_wire()
                else []
            )
            if _near_tiles_on_the_wire():
                got_n, imp_pos, imp_mass, particle_overflow = jax.named_call(
                    _near_exchange_tiles, name="cross_exchange_near"
                )(
                    sb_n,
                    leaf_rows,
                    starts,
                    ends,
                    pos_sorted,
                    mass_sorted,
                    W=W,
                    payload_capacity=cap.recv_node_cap,
                    csr_capacity=cap.recv_near_csr_cap,
                    ndev=ndev,
                    axis_name=axis_name,
                )
            else:
                got_n, imp_pos, imp_mass, particle_overflow = jax.named_call(
                    _near_exchange_compact, name="cross_exchange_near"
                )(
                    sb_n,
                    leaf_rows,
                    starts,
                    ends,
                    pos_sorted,
                    mass_sorted,
                    W=W,
                    payload_capacity=cap.recv_node_cap,
                    csr_capacity=cap.recv_near_csr_cap,
                    send_particle_cap=cap.send_particle_cap,
                    recv_particle_cap=cap.recv_particle_cap,
                    ndev=ndev,
                    axis_name=axis_name,
                )
            if not near_walk:
                # a pass-through (`_near_receiver_walk_needed`): read the CSR directly
                rl_n = _direct_near_lists(
                    block_nodes, jnp.asarray(tree.node_ranges), num_internal, got_n
                )
                imp_mp_n = jnp.zeros((1, n_coeff), jnp.asarray(mp.packed).dtype)
                imp_ecen_n = jnp.zeros((1, 3), jnp.asarray(mp.centers).dtype)
            else:
                # geometry and multipole rows, the same layout in both wire formats
                imp_cen_n = got_n.payload[:, 0:3]  # geometric -> MAC
                imp_rad_n = got_n.payload[:, 3]
                imp_mp_n = got_n.payload[:, 4 : 4 + n_coeff]  # multipole -> M2L
                imp_ecen_n = got_n.payload[:, 4 + n_coeff : 7 + n_coeff]  # -> M2L
                rl_n = jax.named_call(
                    receiver_interaction_lists, name="cross_recv_walk_near"
                )(
                    combined_left,
                    combined_right,
                    jnp.concatenate([jnp.asarray(geom.center), imp_cen_n]),
                    jnp.concatenate([jnp.asarray(geom.radius), imp_rad_n]),
                    n_local,
                    block_nodes,
                    got_n.csr_cell,
                    got_n.csr_row,
                    got_n.num_csr,
                    # Pairs this walk calls far (89568/93005 at N = 2e5, 4-leaf cells)
                    # are served by the M2L from the multipole each near leaf
                    # carries; pairs it calls near are summed directly. Before the
                    # multipole travelled the far ones were unservable and dropped,
                    # and near_theta = 0 was the only correct setting. It remains as
                    # the control.
                    float(theta if near_theta is None else near_theta),
                    max_pair_queue=cap.walk_queue,
                    far_cap=cap.recv_far_cap,
                    near_cap=cap.recv_near_cap,
                    mac_type=mac_type,
                    walk_fn=walk_fn,
                )
            near_sink["positions"] = imp_pos
            near_sink["masses"] = imp_mass
            near_sink["mask"] = imp_mass != 0.0
            near_sink["target_node"] = rl_n.near_target
            near_sink["source_row"] = rl_n.near_source
            near_sink["count"] = rl_n.near_count
            if record is not None:
                record["near_recv_nodes"] = got_n.num_payload
                record["near_list_pairs"] = rl_n.near_count
                # pairs the NEAR walk classified as far: served by M2L from the
                # multipole that now travels with each near-exported leaf (before
                # Task 2 nothing could consume them and they were dropped)
                record["near_walk_far_pairs"] = rl_n.far_count
                # what the walk_queue / recv_near_csr_cap have to cover
                record["near_csr"] = got_n.num_csr
                # How much of the near import the receiver USES: imported leaves its
                # final near list touches (their particles are needed) vs those only
                # its far list touches (the multipole would have done), and how the
                # near CSR is spread over my cells (a few huge outskirt cells can pull
                # in the whole remote domain).
                _nr = int(got_n.payload.shape[0])
                _ns = jnp.where(
                    jnp.arange(rl_n.near_source.shape[0]) < rl_n.near_count,
                    rl_n.near_source,
                    _nr,
                )
                _fs = jnp.where(
                    jnp.arange(rl_n.far_source.shape[0]) < rl_n.far_count,
                    rl_n.far_source,
                    _nr,
                )
                _used_n = jnp.zeros((_nr,), jnp.int32).at[_ns].set(1, mode="drop")
                _used_f = jnp.zeros((_nr,), jnp.int32).at[_fs].set(1, mode="drop")
                record["near_rows_needing_particles"] = jnp.sum(_used_n)
                record["near_rows_far_only"] = jnp.sum(_used_f * (1 - _used_n))
                _cc = jnp.where(
                    jnp.arange(got_n.csr_cell.shape[0]) < got_n.num_csr,
                    got_n.csr_cell,
                    block,
                )
                _per_cell = jnp.zeros((block,), jnp.int32).at[_cc].add(1, mode="drop")
                _srt = jnp.sort(_per_cell)[::-1]
                record["near_csr_max_per_cell"] = _srt[0]
                record["near_csr_top100_cells"] = jnp.sum(_srt[:100])
                record["near_csr_cells_over_1000"] = jnp.sum(_per_cell > 1000)
                # who is the worst cell: node id, leaf?, particles, MAC radius, Morton
                # cell edge (box units) and distance of its centre from the box centre
                _w = jnp.argmax(_per_cell)
                _wn = jnp.maximum(block_nodes[_w], 0)
                _nr = jnp.asarray(tree.node_ranges)
                record["worst_cell_is_leaf"] = (_wn >= num_internal).astype(jnp.int32)
                record["worst_cell_particles"] = _nr[_wn, 1] - _nr[_wn, 0] + 1
                record["worst_cell_leaves"] = subtree_leaf_counts(_nr, num_internal)[
                    _wn
                ]
                record["worst_cell_radius"] = jnp.asarray(geom.radius)[_wn]
                record["worst_cell_edge"] = _node_cell_edge(tree)[_wn]
                _c = jnp.asarray(geom.center)[_wn]
                record["worst_cell_r_center"] = jnp.sqrt(jnp.sum(_c * _c))
                record["root_radius"] = jnp.asarray(geom.radius)[jnp.argmin(parent)]
                # live particles the near import carries (compact format only)
                record["near_particles"] = jnp.sum(
                    jnp.where(got_n.num_payload > 0, imp_mass != 0.0, False)
                )
                record["export_near"] = ex.near_count
                # 0 when the walk was skipped (a pass-through under a leaf summary)
                record["near_walk_peak"] = (
                    rl_n.peak_wavefront
                    if rl_n.peak_wavefront is not None
                    else jnp.asarray(0, jnp.int32)
                )
            near_sink["overflow"] = (
                pub.overflow
                | sb_n.node_overflow
                | sb_n.csr_overflow
                | rl_n.near_overflow
                | rl_n.queue_overflow
                # the near walk's FAR list now feeds the M2L (leaf multipoles are
                # shipped), so its capacity is a correctness condition too
                | rl_n.far_overflow
                | tile_truncated
                | particle_overflow
                # the exchange itself has no overflow flag: compare what arrived
                # with what the receive buffers can hold
                | (got_n.num_payload > cap.recv_node_cap)
                | (got_n.num_csr > cap.recv_near_csr_cap)
            )
            flag_sink["near"] = near_sink["overflow"]

        far_overflow = (
            # `summary.overflow` is the flag whose absence is silent: a truncated cut
            # drops part of the RECEIVER from the exchange, so those particles get
            # no cross field at all. It loses force, not accuracy. Two-sided, it is
            # the summary TREE's flag (max_summary_nodes), which includes the cut's.
            pub.overflow
            | ex.far_overflow
            | ex.near_overflow
            | ex.queue_overflow
            | sb.node_overflow
            | sb.csr_overflow
            | rl.far_overflow
            | rl.queue_overflow
            | (got.num_payload > cap.recv_node_cap)
            | (got.num_csr > cap.recv_csr_cap)
        )
        flag_sink["far"] = far_overflow

        if record is not None:
            record["export_far"] = ex.far_count
            if ex.peak_wavefront is not None:
                record["export_walk_peak"] = ex.peak_wavefront
            record["send_nodes"] = jnp.sum(sb.node_sizes)
            record["recv_nodes"] = got.num_payload
            record["recv_csr"] = got.num_csr
            record["far_pairs"] = rl.far_count
            record["near_pairs"] = rl.near_count
            # A zero imported radius is the failure this ships the radius to avoid,
            # and it is INVISIBLE in the force: it does not crash, it does not
            # overflow, it just moves close pairs into the far list. Count them.
            live_rows = jnp.arange(imp_rad.shape[0]) < got.num_payload
            record["imported_zero_radius"] = jnp.sum(
                jnp.where(live_rows, imp_rad <= 0.0, False)
            )
            record["imported_rows"] = got.num_payload
            record["summary_cells"] = summary.num_cells
            record["summary_nodes"] = pub.num_entries
            record["summary_leaves"] = jnp.sum(summary.leaves_per_cell)
            record["overflow"] = far_overflow

        # -1 padding on the pair list is dropped by the CSR build; the imported
        # sources are rebased to sit ABOVE every local index, which is what the
        # (min, max) canonicalisation downstream depends on
        if near_sink is not None:
            return jax.named_call(merge_imported_blocks, name="cross_merge")(
                imp_mp,
                imp_cen,
                rl.far_source,
                rl.far_target,
                rl.far_count,
                imp_mp_n,
                imp_ecen_n,
                rl_n.far_source,
                rl_n.far_target,
                rl_n.far_count,
                n_local=n_local,
            )
        live_pair = jnp.arange(rl.far_target.shape[0]) < rl.far_count
        src = jnp.where(live_pair, jnp.asarray(n_local) + rl.far_source, -1)
        tgt = jnp.where(live_pair, rl.far_target, -1)
        return imp_mp, imp_cen, src, tgt

    hook.flag_sink = flag_sink  # pyright: ignore[reportFunctionMemberAccess]
    hook.needs_sink = needs_sink  # pyright: ignore[reportFunctionMemberAccess]
    # what the refresh calls it with (`_call_cross_hook`)
    hook.error_criterion = bool(  # pyright: ignore[reportFunctionMemberAccess]
        error_criterion
    )
    return hook


def cross_needs(cross_hook: Any) -> Array:
    """What every cross capacity needed in the hook's most recent trace.

    Read inside the SAME trace, right after the force that ran the hook, like the
    hook's ``flag_sink``: it holds tracers. Reduce it over the mesh with ``pmax``
    (:func:`~jaccpot.distributed.fused.make_fused_force_evaluator` does with
    ``cross_needs=True``) before sizing anything from it; the capacities are shared
    by every device.

    Parameters
    ----------
    cross_hook : Any
        A hook from :func:`make_cross_hook`.

    Returns
    -------
    Array
        ``int32[len(CROSS_NEEDS_FIELDS)]``.

    Raises
    ------
    ValueError
        If the hook recorded no needs: it has not been traced, or it ran a control
        exchange (one-sided, CSR, tiles, a receiver walk) -- the needs are measured on
        the symmetric exchange, the default, only.
    """
    sink = getattr(cross_hook, "needs_sink", None) or {}
    if "needs" not in sink:
        raise ValueError(
            "cross_needs: the hook recorded no needs. They are measured by the "
            "symmetric exchange (the default) inside a traced force; a control "
            "exchange (JACCPOT_CROSS_SYMMETRIC=0, JACCPOT_CROSS_TWO_SIDED=0, tiles, "
            "a receiver walk) does not measure them."
        )
    return sink["needs"]


def merge_imported_blocks(
    far_mp: Array,
    far_cen: Array,
    far_src: Array,
    far_tgt: Array,
    far_count: Array,
    near_mp: Array,
    near_cen: Array,
    near_src: Array,
    near_tgt: Array,
    near_count: Array,
    *,
    n_local: int,
) -> tuple[Array, Array, Array, Array]:
    """Stack the far and near imports into ONE imported block for the M2L.

    The far import (internal nodes) and the near import (leaves) arrive in two
    capacity-padded payloads, each with its own receiver walk whose far pairs
    name a payload ROW. The M2L sees a single ``[local ; imported]`` space, so the
    near block is placed behind the far block and its rows are shifted by the far
    block's FULL capacity -- the padded length, not the live count, because the
    far payload's dead rows are still rows in the concatenated array. Getting
    that shift wrong aliases a near-leaf pair onto an unrelated far node's
    multipole: right shapes, plausible force, wrong answer.

    Dead pair slots become ``-1`` on both sides; the CSR build drops them.

    Returns
    -------
    tuple[Array, Array, Array, Array]
        ``(multipoles, centers, src, tgt)`` -- the ``cross_far`` tuple the sweep
        concatenates behind the local nodes with ``n_targets = n_local``.
    """
    far_mp = jnp.asarray(far_mp)
    near_mp = jnp.asarray(near_mp)
    far_rows = int(far_mp.shape[0])
    mp = jnp.concatenate([far_mp, near_mp.astype(far_mp.dtype)])
    cen = jnp.concatenate(
        [jnp.asarray(far_cen), jnp.asarray(near_cen, jnp.asarray(far_cen).dtype)]
    )
    far_src = jnp.asarray(far_src)
    near_src = jnp.asarray(near_src)
    base = jnp.asarray(n_local, far_src.dtype)
    live_f = jnp.arange(far_src.shape[0]) < far_count
    live_n = jnp.arange(near_src.shape[0]) < near_count
    src = jnp.concatenate(
        [
            jnp.where(live_f, base + far_src, -1),
            jnp.where(
                live_n, base + jnp.asarray(far_rows, far_src.dtype) + near_src, -1
            ),
        ]
    )
    tgt = jnp.concatenate(
        [
            jnp.where(live_f, jnp.asarray(far_tgt), -1),
            jnp.where(live_n, jnp.asarray(near_tgt, jnp.asarray(far_tgt).dtype), -1),
        ]
    )
    return mp, cen, src, tgt


def cross_near_acceleration(
    near: dict,
    local_leaf_positions: Array,
    local_leaf_masses: Array,
    local_leaf_mask: Array,
    leaf_particle_indices: Array,
    n_particles: int,
    num_internal: int,
    *,
    softening_sq: Array,
    G: Array,
    chunk: int = 64,
    interpret: bool = False,
    accum: Optional[str] = None,
    softening_kernel: Optional[str] = None,
    with_force_scale: bool = False,
    with_potential: bool = False,
) -> Union[Array, tuple[Array, Array]]:
    """Acceleration on local particles from the IMPORTED leaves, as its own term.

    Added to the force rather than interleaved, because a near contribution is a
    direct sum with no cascade behind it -- see the module docstring. Correct by
    linearity of gravity in the sources.

    The pool is ``[local leaves ; imported leaves]`` and ``num_target_leaves`` is the
    local count, so the kernel emits local rows only (Phase 4.3). The CSR carries the
    cross pairs alone: the local near field is already in the force this is added to,
    and including it here would double it.

    Parameters
    ----------
    near:
        The ``near_sink`` a :func:`make_cross_hook` filled.
    local_leaf_positions, local_leaf_masses, local_leaf_mask:
        This device's leaf-major pool, ``(L, W, 3)`` and ``(L, W)``.
    leaf_particle_indices:
        ``(L, W)`` particle index per padded leaf slot, for scattering the result
        back into particle order.
    n_particles:
        Rows in the output.
    num_internal:
        Internal-node count of the local tree. A target arrives as a NODE id and the
        pool is indexed by LEAF ROW, and the two differ by exactly this -- deriving
        it from the pool's own length instead only works for a balanced structure,
        which is not something to assume here.
    softening_sq, G:
        Scalars, matching the local lane's.
    chunk:
        Entries per chunk for the leafpair table. Static.
    interpret:
        Pallas interpret mode.
    accum:
        Accumulator width of the leafpair kernel, ``"input"`` or ``"wide"`` (a
        float32 partial per source leaf, float64 across leaves -- the two-level
        accumulator of the local lane's near field). ``None`` reads the SAME
        env choice the local lane reads, ``JACCPOT_NEARFIELD_ACCUM``, so the two
        near terms widen together. Before this knob existed the cross term was
        pinned to ``"input"`` whatever the local lane did, which made it the one
        fp32-only reduction in an otherwise widened force.

    softening_kernel:
        The pair kernel (:mod:`jaccpot.softening`); ``None`` gives the default.
    with_force_scale:
        Also return the imported leaves' share of eq (16b)'s force scale, ``sum G
        m / (r^2 + eps^2)`` over the same pairs (the kernel's fourth lane, in place
        of the potential), for ``mac_type='dehnen_error'``. Static.
    with_potential:
        Also return the imported leaves' share of the potential, ``-G sum m
        psi(r)`` with the pair kernel's ``psi`` over the same pairs: the kernel's
        fourth lane in its default meaning (the sorted kernel's
        ``with_potential``), which the force drops. For the mesh potential's
        program (``template_force.mesh_potential_shard``); exclusive with
        ``with_force_scale``. Static.

    Returns
    -------
    Union[Array, tuple[Array, Array]]
        ``(n_particles, 3)`` acceleration to ADD to the local force; with
        ``with_force_scale``, ``(acceleration, force_scale [n_particles])``; with
        ``with_potential``, ``(acceleration, potential [n_particles])``.

    Raises
    ------
    ValueError
        If both ``with_force_scale`` and ``with_potential`` are set: they share the
        kernel's fourth lane.
    """
    from jaccpot.nearfield.near_field import _env_choice
    from jaccpot.pallas.nearfield_leafpair_csr import (
        build_leafpair_chunk_table,
        leafpair_chunk_capacity,
        nearfield_leafpair_csr_pallas,
    )

    if with_force_scale and with_potential:
        raise ValueError("with_force_scale and with_potential share the fourth lane")
    if accum is None:
        accum = _env_choice("JACCPOT_NEARFIELD_ACCUM", "input", ("input", "wide"))

    loc_pos = jnp.asarray(local_leaf_positions)
    loc_mass = jnp.asarray(local_leaf_masses)
    loc_mask = jnp.asarray(local_leaf_mask)
    idx = jnp.asarray(leaf_particle_indices)
    imp_pos = jnp.asarray(near["positions"])
    imp_mass = jnp.asarray(near["masses"])
    imp_mask = jnp.asarray(near["mask"])

    L = int(loc_pos.shape[0])

    # The imported tiles are W wide because a CAPACITY said so; the local pool is
    # as wide as the tree made it. Concatenating requires they agree, and the two
    # are set in different places, so pad the narrower up rather than trusting a
    # caller to keep them in step. Padding is mask-false, so it changes nothing.
    W = max(int(loc_pos.shape[1]), int(imp_pos.shape[1]))

    def _widen(pos, mass, mask, extra_idx=None):
        w = int(pos.shape[1])
        if w == W:
            return pos, mass, mask, extra_idx
        pad = W - w
        pos = jnp.pad(pos, ((0, 0), (0, pad), (0, 0)))
        mass = jnp.pad(mass, ((0, 0), (0, pad)))
        mask = jnp.pad(mask, ((0, 0), (0, pad)))
        if extra_idx is not None:
            extra_idx = jnp.pad(extra_idx, ((0, 0), (0, pad)))
        return pos, mass, mask, extra_idx

    loc_pos, loc_mass, loc_mask, idx = _widen(loc_pos, loc_mass, loc_mask, idx)
    imp_pos, imp_mass, imp_mask, _ = _widen(imp_pos, imp_mass, imp_mask)

    pool_pos = jnp.concatenate([loc_pos, imp_pos])
    pool_mass = jnp.concatenate([loc_mass, imp_mass])
    pool_mask = jnp.concatenate([loc_mask, imp_mask])

    # the cross pairs, as a CSR over LOCAL target leaf rows
    tgt_node = jnp.asarray(near["target_node"])
    src_row = jnp.asarray(near["source_row"])
    n_pair = jnp.asarray(near["count"])
    P = int(tgt_node.shape[0])
    # A target arrives as a leaf NODE id; the pool is indexed by leaf ROW. The
    # upper bound is part of `live` rather than a clip: clipping an out-of-range
    # target would silently ADD its sources to row 0 or L-1, which is a wrong force,
    # whereas dropping it is a missing one. Neither is acceptable, but only the
    # second is detectable downstream, and the test asserts nothing is dropped.
    lrow = tgt_node - int(num_internal)
    live = (jnp.arange(P) < n_pair) & (lrow >= 0) & (lrow < L)
    row = jnp.where(live, lrow, L)
    order = jnp.argsort(jnp.where(live, row, L), stable=True)
    row_s = row[order]
    src_s = jnp.where(live[order], L + src_row[order], -1)
    counts = jnp.bincount(jnp.where(live[order], row_s, L), length=L + 1)[:L].astype(
        jnp.int32
    )
    offsets = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)]
    )
    cap_chunks = leafpair_chunk_capacity(P, L, int(chunk))
    table = build_leafpair_chunk_table(
        offsets, counts, chunk=int(chunk), capacity=cap_chunks
    )

    out = nearfield_leafpair_csr_pallas(
        pool_pos,
        pool_mass,
        pool_mask,
        src_s.astype(jnp.int32),
        table,
        softening_sq=softening_sq,
        G=G,
        chunk=int(chunk),
        interpret=bool(interpret),
        include_self=False,  # the local self term is already in the force
        num_target_leaves=L,
        accum=str(accum),
        softening_kernel=softening_kernel,
        **({"with_force_scale": True} if with_force_scale else {}),
    )
    acc = out[..., :3]
    # The scatter below is a PERMUTATION, not a reduction: every live particle sits
    # in exactly one leaf slot, so each output row receives one addend. Nothing is
    # summed in the input dtype here; the only reductions are inside the kernel,
    # under `accum`. Dead slots get the out-of-range row n_particles and are
    # DROPPED -- routing them onto one discard row made every dead slot (~70 % of
    # L x W at ~18 particles per 64-slot leaf) an atomic add on the same address.
    flat_idx = jnp.where(loc_mask, idx, n_particles).reshape(-1)
    acc_out = (
        jnp.zeros((n_particles, 3), acc.dtype)
        .at[flat_idx]
        .add(acc.reshape(-1, 3), mode="drop")
    )
    if not (with_force_scale or with_potential):
        return acc_out
    # lane 3: the force scale with `with_force_scale`, else the potential
    fourth = out[..., 3]
    return acc_out, (
        jnp.zeros((n_particles,), fourth.dtype)
        .at[flat_idx]
        .add(fourth.reshape(-1), mode="drop")
    )
