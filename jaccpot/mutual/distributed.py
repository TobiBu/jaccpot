"""A distributed mutual FMM: intra-domain pairs locally, cross-domain pairs exactly once.

Assembles the pieces TobiBu/jaccpot#173 decomposed the problem into -- the canonical
cross-domain emitter and straddling-node rule
(``yggdrax.distributed.cross_walk``), the locally essential tree
(``yggdrax.distributed.let``), the reverse halo that returns the ``-f`` half
(``yggdrax.distributed.reverse_halo``), and the capacity table
(:mod:`jaccpot.mutual.cap_presets`) -- into a force that conserves momentum across
devices.

**What makes this correct rather than merely parallel.** A mutual FMM evaluates each
unordered pair once and applies ``+f``/``-f``, so ``sum_i m_i a_i`` cancels
structurally rather than numerically. Within a domain that is free: both endpoints are
local and one kernel writes both halves. Across a boundary the evaluating device holds
only one endpoint, so the other half is *sent back*, and that return path is the
conservation mechanism rather than an optimisation. Drop it or double-count it and the
force is wrong at the percent level while every device's *local* momentum sum still
looks perfect -- which is why the tests assert a global sum.

**The one simplification, stated plainly: cross-domain interactions are EXACT.** The
cross walk is driven with ``theta = 0``, so its MAC never accepts and every
cross-domain pair refines down to leaf-leaf. Those leaf pairs are then summed
particle-by-particle, which is precisely what a near-field kernel does. So this lane
approximates *within* a domain exactly as the single-device lane does, and does not
approximate *between* domains at all.

That is a deliberate ordering of the work, not an oversight. It isolates the question
this lane has to answer first -- does the pair partition and the return path conserve
momentum globally, and does the force match a direct sum -- from the separate question
of cross-domain far-field accuracy. Routing accepted cross far pairs through M2L is the
optimisation that follows, and it will trade exactness for pair count; nothing about the
ownership rule or the exchange changes when it lands.

**Where remote structure comes from, and what theta = 0 does and does not buy.** Remote
nodes arrive as a *locally essential tree*: each device publishes only its leaves' mass
and centre of mass (``build_coarse_frontier``), every device merges every *other*
domain's frontier into one coarse tree (``build_remote_coarse_tree``), the local tree is
cross-walked against that, and only the remote leaves the walk actually names are
fetched, by ragged all-to-all (``import_near_halo``).

Be precise about the win, because at ``theta = 0`` it is not yet a volume win. What
changes:

* **One walk instead of ``ndev`` walks.** The stand-in looped over remote devices in
  PYTHON, so the traced program carried one ``while_loop`` and one near list per
  device. The coarse tree is a single merged structure, so the operation count stops
  tracking the mesh -- buffer shapes still scale with ``ndev``, the program structure
  no longer does.
* **Remote structure is a flat frontier**, not five full node arrays per device --
  ``(mass, com, node_range, node_id)`` per leaf, with the topology rebuilt locally
  rather than shipped.
* **Particles arrive on demand**, by ragged all-to-all over the leaves the walk
  actually named, rather than by a dense ``all_gather`` of everything unconditionally.
* **The ``-f`` exchange is sized by the halo**, one row per imported particle, not by
  the pair list -- so it does not grow with the pair count that ``theta = 0`` inflates.

What does NOT change yet: per-device traffic and residency are still ``O(N_total)``.
A MAC that never accepts refines every pair to leaf-leaf, so every local leaf pairs
with every remote leaf and the demand-driven import ends up fetching the whole remote
system anyway. Collapsing that to the surface halo is what lifting ``theta`` buys, and
it needs the coarse M2L; that is the next step, not this one. What lands here is the
machinery, in one program whose size no longer tracks the device count, with a force
that is still exact.

Two capacities inherit the same caveat and will shrink with ``theta``:
``max_pair_queue`` must hold a wavefront that at ``theta = 0`` reaches
``n_leaves * (ndev - 1) * n_leaves`` pairs, and ``near_cap`` roughly half of that.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array
from yggdrax.distributed.cross_walk import (
    dual_tree_walk_cross_mutual,
    single_owner_domain,
)
from yggdrax.distributed.let import (
    build_coarse_frontier,
    build_remote_coarse_tree,
    import_near_halo,
)
from yggdrax.distributed.partition import global_bounds
from yggdrax.distributed.reverse_halo import (
    apply_reverse_halo,
    export_reverse_halo,
    halo_return_addresses,
)
from yggdrax.distributed.sharding import AXIS_NAME
from yggdrax.dtypes import INDEX_DTYPE

from jaccpot.mutual.device_topology import leaf_blocks

__all__ = ["DistributedMutualResult", "distributed_mutual_accelerations"]


class DistributedMutualResult(NamedTuple):
    """Per-device accelerations plus the flags a caller must check.

    ``overflow`` means "this force is not trustworthy", and it covers more than a
    literal capacity: a walk or halo capacity exceeded, or a coarse tree whose leaves
    hold more than one frontier node so their particles cannot be addressed. It is
    reduced ACROSS devices, not per device, because a pair dropped anywhere is a wrong
    force everywhere while momentum stays exact -- a per-device check would miss
    exactly the case worth catching.
    """

    acceleration: Array
    cross_pairs: Array
    overflow: Array

    # `cross_pairs` and `overflow` carry a leading singleton axis so `shard_map` can
    # concatenate them across devices; after the map both are `(ndev,)`.


class _FrontierTree(NamedTuple):
    """The tree fields :func:`build_coarse_frontier` reads, for an array-holding caller.

    This module takes a tree as loose arrays rather than a ``yggdrax.Tree``, because
    its caller is a force routine that already has them. ``build_coarse_frontier``
    wants an object, and wants very little of it, so the adapter is four fields:

    ``node_ranges``
        inclusive ``[start, end]`` particle range per node, the convention yggdrax and
        jaccpot's own mutual lane both use.
    ``left_child``
        internal nodes only. Its LENGTH is what names the leaves -- yggdrax lays a
        binary tree out as ``num_leaves - 1`` internal nodes followed by the leaves --
        so this is the full child array truncated, not a separate table.
    ``parent``
        read only through ``argmin``, to locate the root. Synthesised from the root
        index the caller already passed rather than threaded through.
    ``positions_sorted``, ``leaf_size``
        for the per-leaf radius. ``build_coarse_frontier`` measures each frontier
        leaf's own extent -- the fix for coarse MAC extents that bounded centres of
        mass rather than particles (yggdrax#47) -- and RAISES rather than defaulting
        that radius to zero, since a zero would silently understate every coarse
        extent. It looks for exactly these two attributes: ``positions_sorted`` to
        measure from, and ``leaf_size`` to bound the reduction's gather width. Without
        the second the radius pass stages a ``(leaves, n)`` buffer instead of a
        ``(leaves, leaf_width)`` one -- harmless at test sizes, not at 1e9.
    """

    node_ranges: Array
    left_child: Array
    parent: Array
    positions_sorted: Array
    leaf_size: int


class _NearAsNeighbors(NamedTuple):
    """The one field :func:`import_near_halo` reads off a walk result.

    It was written against the target-centric cross walk, whose near list is per-target
    CSR with a ``neighbor_indices`` array of remote coarse node indices padded with
    ``-1``. The mutual walk's ``near_remote`` is a flat list of exactly that -- same
    meaning, same padding -- so naming it is the whole adapter.
    """

    neighbor_indices: Array


def _tile_forces(pos_a, mass_a, ok_a, pos_b, mass_b, ok_b, softening, g):
    """Antisymmetric leaf-pair block: force on every a-particle and every b-particle.

    One evaluation, two applications -- the ``b`` side is the negation of the same
    tensor, never an independent recomputation. That is what keeps the cancellation
    structural, so it holds whatever the softening or the masses.

    Parameters
    ----------
    pos_a:
        ``(k, w, 3)`` a-side positions.
    mass_a:
        ``(k, w)`` a-side masses.
    ok_a:
        ``(k, w)`` a-side validity.
    pos_b:
        ``(k, w, 3)`` b-side positions.
    mass_b:
        ``(k, w)`` b-side masses.
    ok_b:
        ``(k, w)`` b-side validity.
    softening:
        Plummer softening length.
    g:
        Gravitational constant.

    Returns
    -------
    tuple[Array, Array]
        ``(f_a, f_b)``, each ``(k, w, 3)``.
    """
    d = pos_b[:, None, :, :] - pos_a[:, :, None, :]
    r2 = jnp.sum(d * d, axis=-1) + jnp.asarray(softening, d.dtype) ** 2
    pair_ok = ok_a[:, :, None] & ok_b[:, None, :]
    inv3 = jnp.where(pair_ok, r2 ** (-1.5), 0.0)
    # shared[k, i, j] is the geometric factor for the (i, j) pair; both sides read it.
    shared = (g * inv3)[..., None] * d
    f_a = jnp.sum(shared * mass_b[:, None, :, None], axis=2)
    f_b = -jnp.sum(shared * mass_a[:, :, None, None], axis=1)
    return f_a, f_b


def distributed_mutual_accelerations(
    positions: Array,
    masses: Array,
    left_child_full: Array,
    right_child_full: Array,
    node_ranges: Array,
    centers: Array,
    radii: Array,
    root: Array,
    local_acceleration: Array,
    *,
    softening: float,
    g: float = 1.0,
    ndev: int,
    leaf_width: int,
    near_cap: int,
    max_pair_queue: int,
    recv_capacity: int,
    max_req_leaves: Optional[int] = None,
    max_recv_leaves: Optional[int] = None,
    coarse_depth_cap: int = 64,
    axis_name: str = AXIS_NAME,
) -> DistributedMutualResult:
    """Add every cross-domain contribution to an already-computed local force.

    Call inside ``shard_map``. ``local_acceleration`` is whatever the single-device
    mutual lane produced for this domain's own pairs; this adds the cross-domain half,
    applying ``+f`` here and returning ``-f`` to the domain that owns the other
    endpoint.

    Parameters
    ----------
    positions:
        ``(n, 3)`` this domain's positions, in tree order.
    masses:
        ``(n,)`` this domain's masses, in tree order.
    left_child_full:
        ``(nodes,)`` left children, -1 for leaves. ``nodes`` must be ``2 * L - 1`` for
        ``L`` leaves, which every binary tree yggdrax builds satisfies.
    right_child_full:
        ``(nodes,)`` right children, -1 for leaves.
    node_ranges:
        ``(nodes, 2)`` **inclusive** particle range per node.
    centers:
        ``(nodes, 3)`` centres of mass.
    radii:
        ``(nodes,)`` max centre-of-mass-to-particle radii.
    root:
        Root node index.
    local_acceleration:
        ``(n, 3)`` acceleration from this domain's own pairs.
    softening:
        Plummer softening length.
    g:
        Gravitational constant.
    ndev:
        Device count.
    leaf_width:
        Maximum particles per leaf, and so also the width of an imported halo block.
        Static.
    near_cap:
        Cross leaf-pair capacity.
    max_pair_queue:
        Cross wavefront capacity.
    recv_capacity:
        Reverse-halo receive capacity, in remote-particle rows. Bounded by
        ``(ndev - 1) * n``: every other device can address at most this domain's whole
        particle set.
    max_req_leaves:
        Remote leaves this device may import. ``None`` takes the worst case,
        ``(ndev - 1) * L``, which is what ``theta = 0`` actually reaches.
    max_recv_leaves:
        Import requests this device may receive. ``None`` takes the same worst case.
    coarse_depth_cap:
        Rounds of owner propagation up the coarse tree. Under-provisioning is safe:
        an unresolved node is treated as straddling, which the walk refines rather
        than accepts, and a near pair's remote endpoint is always a leaf whose owner
        is set directly.
    axis_name:
        Mesh axis.

    Returns
    -------
    DistributedMutualResult
        Total acceleration for this domain, owned cross leaf-pair count, and the
        across-device overflow flag.

    Raises
    ------
    ValueError
        If ``ndev`` is less than 2.
    """
    if int(ndev) < 2:
        # Not a shape accident to discover later: with one domain the remote coarse
        # tree has no particles to be built over, and there is nothing for this
        # function to add -- the local lane is already the whole force.
        raise ValueError(
            f"ndev must be >= 2, got {ndev}: a single domain has no cross-domain "
            "pairs, so the single-device mutual lane is the whole force"
        )

    me = jax.lax.axis_index(axis_name)
    n_local = int(positions.shape[0])
    total_nodes = int(left_child_full.shape[0])
    # A binary tree over L leaves has L - 1 internal nodes, and yggdrax lays the
    # internal ones first, so the shape alone names the split.
    n_leaves = (total_nodes + 1) // 2
    num_internal = total_nodes - n_leaves
    max_req = (ndev - 1) * n_leaves if max_req_leaves is None else int(max_req_leaves)
    max_recv = (
        (ndev - 1) * n_leaves if max_recv_leaves is None else int(max_recv_leaves)
    )

    node_ranges = jnp.asarray(node_ranges).astype(INDEX_DTYPE)
    root = jnp.asarray(root).astype(INDEX_DTYPE)

    # --- 1. this domain's frontier: one (mass, COM) record per leaf ------------
    # Node mass by prefix sum over the inclusive range, rather than asking the caller
    # for it: `centers` is already documented as the centre of mass, so mass is the
    # only missing moment and it is one gather.
    pad = jnp.zeros((1,), dtype=masses.dtype)
    mass_prefix = jnp.concatenate([pad, jnp.cumsum(masses)])
    node_mass = mass_prefix[node_ranges[:, 1] + 1] - mass_prefix[node_ranges[:, 0]]
    frontier = build_coarse_frontier(
        _FrontierTree(
            node_ranges=node_ranges,
            left_child=left_child_full[:num_internal],
            parent=jnp.zeros((total_nodes,), dtype=INDEX_DTYPE).at[root].set(-1),
            positions_sorted=positions,
            leaf_size=int(leaf_width),
        ),
        node_mass,
        centers,
    )

    # --- 2. every OTHER domain's frontier, merged into one coarse tree ---------
    bounds = global_bounds(positions, axis_name=axis_name)
    rct = build_remote_coarse_tree(
        frontier, ndev, bounds=bounds, axis_name=axis_name, coarse_leaf_size=1
    )
    c_ranges = jnp.asarray(rct.tree.node_ranges).astype(INDEX_DTYPE)
    c_total = int(c_ranges.shape[0])
    c_leaves = (c_total + 1) // 2
    c_internal = c_total - c_leaves
    c_left = jnp.concatenate(
        [
            jnp.asarray(rct.tree.left_child).astype(INDEX_DTYPE),
            jnp.full((c_leaves,), -1, dtype=INDEX_DTYPE),
        ]
    )
    c_right = jnp.concatenate(
        [
            jnp.asarray(rct.tree.right_child).astype(INDEX_DTYPE),
            jnp.full((c_leaves,), -1, dtype=INDEX_DTYPE),
        ]
    )
    c_root = jnp.argmin(jnp.asarray(rct.tree.parent)).astype(INDEX_DTYPE)
    # The coarse tree is built at leaf_size 1, so a leaf's range start IS the sorted
    # position of its single coarse particle -- the index the origin tags are in.
    c_first = c_ranges[:, 0]
    # ...which only holds if every leaf really does hold one. That is a guarantee of
    # `build_remote_coarse_tree`'s radix builder and not of the coarse tree in general
    # -- its own docstring notes that KD and octree bucket builds cannot honour
    # `leaf_size=1` exactly. Checked rather than assumed because the failure is silent:
    # the import would fetch the leaf's first frontier node and evaluate against it as
    # if it were the whole leaf. No input has been found that trips this, including
    # several frontier leaves sharing one COM, so treat it as a contract check on the
    # builder rather than as a guard against an observed case.
    degenerate = jnp.any(c_ranges[c_internal:, 1] != c_ranges[c_internal:, 0])

    c_com = rct.moments.center_of_mass
    # A radius about the centre of MASS, which is where the mutual MAC measures from,
    # derived from the bounding sphere the coarse build already computed:
    # |x - com| <= |x - c_box| + |c_box - com|. An upper bound, so it can only make
    # acceptance stricter -- the safe direction.
    #
    # WHAT the coarse sphere bounds is the part that had to be fixed upstream, and was
    # (yggdrax#47): built over the frontier's centres of mass as bare POINTS it covered
    # those points and not the remote particles they stand for, so a coarse leaf
    # presented an extent of ~0 however large the remote leaf was. Now each frontier
    # leaf publishes its own extent and the coarse geometry is inflated by it, so this
    # bounds the particles. Which is what a MAC needs, and is therefore what makes
    # lifting theta a question about the cross M2L rather than about these radii. At
    # theta = 0 none of it is consulted for acceptance -- radii only steer which side
    # of a pair refines first.
    c_radii = rct.geometry.radius + jnp.linalg.norm(
        rct.geometry.center - c_com, axis=-1
    )

    c_domain = rct.tag_domain[c_first]
    c_node_id = rct.tag_node_id[c_first]
    c_owner = single_owner_domain(
        c_left, c_right, c_domain, max_depth=int(coarse_depth_cap)
    )

    # --- 3. the cross walk, at theta = 0 --------------------------------------
    # theta = 0 makes the MAC `0 > (r_a + r_b)^2`, which is false for every pair
    # including two coincident points, so nothing is ever accepted: every cross pair
    # refines to leaf-leaf and lands in the near list. Hence far_cap = 1 -- the far
    # list cannot receive an entry -- and its overflow flag is still checked rather
    # than assumed.
    #
    # `remote_index_in_owner` is what makes the ownership filter a partition here. The
    # rule keys a pair on (device, node index) per endpoint, and the coarse tree's
    # numbering is this device's, not its owners' -- two devices give the same remote
    # node different indices. Left at the default the two sides disagree about who
    # owns a pair, which both drops and duplicates pairs, and neither shows up in a
    # per-device momentum check.
    res = dual_tree_walk_cross_mutual(
        left_child_full,
        right_child_full,
        centers,
        radii,
        root,
        c_left,
        c_right,
        c_com,
        c_radii,
        c_root,
        0.0,
        this_device=me,
        remote_owner=c_owner,
        remote_index_in_owner=c_node_id,
        max_pair_queue=max_pair_queue,
        far_cap=1,
        near_cap=near_cap,
    )
    overflow = res.far_overflow | res.near_overflow | res.queue_overflow | degenerate

    rows = jnp.arange(near_cap, dtype=INDEX_DTYPE)
    on_list = rows < res.near_count
    c_node = jnp.where(on_list, res.near_remote, c_root)
    cpos = c_first[c_node]
    # A massless frontier leaf (padding, or an empty leaf) is dropped: it is published
    # with a placeholder COM and `node_id = -1`, so it has no address to return `-f`
    # to. Nothing is lost -- it exerts no force -- and it cannot be double-counted
    # either, because the device on the other side of such a pair drops it
    # unconditionally too, whichever of the two the ownership rule picked. What it
    # does forgo is cross-domain force *on* those particles, which is what padding is
    # for.
    wanted = on_list & (rct.tag_node_id[cpos] >= 0)

    # --- 4. fetch only the remote leaves this device's own pairs named ---------
    halo = import_near_halo(
        rct,
        _NearAsNeighbors(jnp.where(wanted, c_node, jnp.asarray(-1, INDEX_DTYPE))),
        positions,
        masses,
        ndev,
        leaf_size=int(leaf_width),
        max_req_leaves=max_req,
        max_recv_leaves=max_recv,
        axis_name=axis_name,
    )
    overflow = overflow | halo.request_overflow

    # --- 5. evaluate local leaf x halo block, exactly --------------------------
    block = halo.coarse_to_halo[cpos]
    # `block >= 0` catches a leaf that was needed but did not fit in the request
    # buffer: the pair is dropped rather than evaluated against the wrong particles,
    # and `request_overflow` above is what says so.
    live = wanted & (block >= 0)

    local_leaf = jnp.where(live, res.near_local, root)
    ia, oka = leaf_blocks(node_ranges, local_leaf, leaf_size=int(leaf_width))
    oka = oka & live[:, None]

    n_halo = max_req * int(leaf_width)
    off = jnp.arange(int(leaf_width), dtype=INDEX_DTYPE)
    jb = jnp.where(live, block, 0)[:, None] * int(leaf_width) + off[None, :]
    okb = live[:, None] & halo.valid[jb]

    f_a, f_b = _tile_forces(
        positions[ia],
        jnp.where(oka, masses[ia], 0.0),
        oka,
        halo.positions[jb],
        jnp.where(okb, halo.masses[jb], 0.0),
        okb,
        softening,
        g,
    )
    acc = local_acceleration.at[jnp.where(oka, ia, n_local)].add(
        jnp.where(oka[..., None], f_a, 0.0), mode="drop"
    )
    # The `-f` half accumulates per imported PARTICLE, not per pair: several local
    # leaves hit the same halo block and their contributions must sum before they are
    # sent. That is also what makes the exchange's size the halo rather than the pair
    # list, so it does not grow when theta = 0 inflates the pair count.
    halo_force = (
        jnp.zeros((n_halo, 3), dtype=positions.dtype)
        .at[jnp.where(okb, jb, n_halo)]
        .add(jnp.where(okb[..., None], f_b, 0.0), mode="drop")
    )

    # --- 6. return every -f to the domain that owns its particle --------------
    # `coarse_to_halo` maps coarse position -> halo block; the addresses need the
    # inverse. It is injective, so one scatter inverts it.
    n_remote = int(rct.tag_domain.shape[0])
    c2h = halo.coarse_to_halo
    halo_to_coarse = (
        jnp.full((max_req,), -1, dtype=INDEX_DTYPE)
        .at[jnp.where(c2h >= 0, c2h, max_req)]
        .set(jnp.arange(n_remote, dtype=INDEX_DTYPE), mode="drop")
    )
    # A block that was imported but never evaluated against carries exactly zero, so
    # skipping it is an exact filter on the wire, not a truncation.
    block_used = (
        jnp.zeros((max_req,), dtype=jnp.bool_)
        .at[jnp.where(live, block, max_req)]
        .set(True, mode="drop")
    )
    hrow = jnp.arange(n_halo, dtype=INDEX_DTYPE)
    hblock = hrow // int(leaf_width)
    row_cpos = halo_to_coarse[hblock]
    row_live = halo.valid & (row_cpos >= 0) & block_used[hblock]
    owner, index = halo_return_addresses(
        row_cpos,
        hrow % int(leaf_width),
        rct.tag_domain,
        rct.tag_range,
        row_live,
    )
    rev = export_reverse_halo(
        owner,
        index,
        halo_force,
        ndev,
        recv_capacity=recv_capacity,
        axis_name=axis_name,
    )
    overflow = overflow | rev.overflow
    acc = apply_reverse_halo(acc, rev)

    # Shape (1,) rather than scalar for the two diagnostics: `shard_map` cannot
    # concatenate rank-0 per-device outputs, so a function documented as "call inside
    # shard_map" should hand back something its `out_specs` can actually describe.
    # The caller sees `(ndev,)` after the map.
    return DistributedMutualResult(
        acceleration=acc,
        cross_pairs=jnp.sum(live.astype(INDEX_DTYPE))[None],
        overflow=(jax.lax.pmax(overflow.astype(INDEX_DTYPE), axis_name) > 0)[None],
    )
