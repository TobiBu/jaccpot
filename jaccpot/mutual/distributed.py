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

**The default, stated plainly: cross-domain interactions are EXACT.** ``cross_theta``
defaults to ``0.0``, so the cross walk's MAC never accepts and every cross-domain pair
refines down to leaf-leaf. Those leaf pairs are then summed particle-by-particle, which
is precisely what a near-field kernel does. At the default this lane therefore
approximates *within* a domain exactly as the single-device lane does, and does not
approximate *between* domains at all.

That is a deliberate ordering of the work. It isolates the question this lane had to
answer first -- does the pair partition and the return path conserve momentum globally,
and does the force match a direct sum -- from the separate question of cross-domain
far-field accuracy. It remains the default for that reason, and because it is the
configuration every accuracy baseline here was taken at.

**Lifting it is implemented.** ``cross_theta > 0`` routes accepted cross pairs through
M2L against the remote leaf's multipole (section 7 below), which is what collapses the
residency described further down. It needs ``node_multipoles``, ``expansion_centers``
and ``tree_arrays``, and it is a SEPARATE additive pass rather than an injection into
the intra-domain far field -- exact rather than merely convenient, since
``_push_locals_down`` and ``_l2p_forces`` are both linear in the local coefficients.
Nothing about the ownership rule or the exchange changed when it landed, as predicted.

One asymmetry is worth carrying: ``far_recv_capacity`` is the only capacity here whose
starvation breaks MOMENTUM rather than merely accuracy. Everywhere else an overflow
drops a pair before it is evaluated, so both halves vanish together; a far pair's
``+f`` is applied locally the moment it is computed and only the ``-f`` travels, so a
dropped receive row leaves the ``+f`` with nothing to cancel it.

**Where remote structure comes from, and what ``cross_theta = 0`` does and does not
buy.** Remote
nodes arrive as a *locally essential tree*: each device publishes only its leaves' mass
and centre of mass (``build_coarse_frontier``), every device merges every *other*
domain's frontier into one coarse tree (``build_remote_coarse_tree``), the local tree is
cross-walked against that, and only the remote leaves the walk actually names are
fetched, by ragged all-to-all (``import_near_halo``).

Be precise about the win, because at ``cross_theta = 0`` it is not yet a volume win.
What changes:

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
  the pair list -- so it does not grow with the pair count that ``cross_theta = 0``
  inflates.

What does NOT change **at the default**: per-device traffic and residency are
``O(N_total)``. A MAC that never accepts refines every pair to leaf-leaf, so every
local leaf pairs with every remote leaf and the demand-driven import ends up fetching
the whole remote system anyway. That is the price of exactness, and it is the reason
``cross_theta`` exists -- collapsing the import to the surface halo is what lifting it
buys. Two capacities scale the same way and shrink with ``cross_theta``:
``max_pair_queue`` must hold a wavefront that at ``cross_theta = 0`` reaches
``n_leaves * (ndev - 1) * n_leaves`` pairs, and ``near_cap`` roughly half of that.

So the choice is: the default is exact and residency-bound, and ``cross_theta > 0``
trades exactness for a halo that scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

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

__all__ = [
    "DistributedMutualConfig",
    "DistributedMutualEvaluator",
    "DistributedMutualForce",
    "DistributedMutualResult",
    "distributed_mutual_accelerations",
    "distributed_mutual_fmm",
    "make_distributed_mutual_evaluator",
]


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
    cross_far_pairs: Array
    overflow: Array

    # `cross_pairs`, `cross_far_pairs` and `overflow` carry a leading singleton axis so
    # `shard_map` can concatenate them across devices; after the map all are `(ndev,)`.


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


def _tile_forces(pos_a, mass_a, ok_a, pos_b, mass_b, ok_b, softening, g, weights=None):
    """Antisymmetric leaf-pair block: force on every a-particle and every b-particle.

    One evaluation, two applications -- the ``b`` side is the negation of the same
    tensor, never an independent recomputation. That is what keeps the cancellation
    structural, so it holds whatever the softening or the masses.

    The level weight rides ``shared``, which is the one tensor both sides read, so a
    weighted block is antisymmetric for exactly the same reason an unweighted one is:
    a single symmetric scalar per pair multiplies ``+f`` and ``-f`` alike. Momentum is
    therefore exact for *any* weighting, which is what lets the block-step boundary be
    fused into one traversal.

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
    weights:
        ``(k, w, w)`` per-pair level weight, or ``None`` for unit weights.

    Returns
    -------
    tuple[Array, Array]
        ``(f_a, f_b)``, each ``(k, w, 3)``.
    """
    d = pos_b[:, None, :, :] - pos_a[:, :, None, :]
    r2 = jnp.sum(d * d, axis=-1) + jnp.asarray(softening, d.dtype) ** 2
    pair_ok = ok_a[:, :, None] & ok_b[:, None, :]
    inv3 = jnp.where(pair_ok, r2 ** (-1.5), 0.0)
    if weights is not None:
        inv3 = inv3 * weights
    # shared[k, i, j] is the geometric factor for the (i, j) pair; both sides read it.
    shared = (g * inv3)[..., None] * d
    f_a = jnp.sum(shared * mass_b[:, None, :, None], axis=2)
    f_b = -jnp.sum(shared * mass_a[:, :, None, None], axis=1)
    return f_a, f_b


def _pair_level_weights(level_weights, rung_a, rung_b):
    """``level_weights[max(rung_i, rung_j)]`` for a pair block, clamped in range.

    The exact per-particle predicate, matching
    :func:`jaccpot.mutual.nearfield._pair_weights` term for term: a pair belongs to
    the level of its FINER endpoint, so the reduction is a max.

    Parameters
    ----------
    level_weights:
        ``(k_max + 1,)`` weight per interaction level. May be a tracer.
    rung_a:
        ``(k, w)`` a-side per-slot rung.
    rung_b:
        ``(k, w)`` b-side per-slot rung.

    Returns
    -------
    Array
        ``(k, w, w)`` per-pair weight.
    """
    pair_level = jnp.maximum(rung_a[:, :, None], rung_b[:, None, :])
    # `jnp.take` defaults to mode="fill", which hands back NaN for an out-of-range
    # index on a float array -- one rung above k_max would poison the whole force.
    # Clamped for the same reason `mutual/nearfield.py` clamps: a caller that can see
    # concrete rungs rejects the case up front, and under trace nothing can.
    pair_level = jnp.clip(pair_level, 0, int(level_weights.shape[0]) - 1)
    return jnp.take(level_weights, pair_level, axis=0, mode="clip")


def _node_rungs(node_ranges, rung, masses, k_max):
    """Assign each node the rung of its most active (finest) massive particle.

    The cell-level split the far field uses -- falcON activity-gating, strategy B2 --
    computed from the inclusive particle ranges alone, because that is all this
    module has. :func:`jaccpot.mutual.force._cell_rungs` gets the same answer by
    propagating a leaf max up the tree, which needs the level schedule; here the
    equivalent is a range maximum, and a range maximum over a SMALL alphabet is a
    prefix count per level:

    ``node_rung = (#levels k with at least one particle at rung >= k in [s, e]) - 1``

    which is exact because ``rung >= k`` is monotone in ``k``. ``k_max`` is small
    (one row per block-step level), so this is ``O(n * k_max)`` and needs no parent
    array, no depth cap and no propagation scan.

    Empty and all-massless nodes come back ``-1``, the same sentinel ``_cell_rungs``
    uses; every consumer clamps into ``[0, k_max]`` afterwards, so ``-1`` and ``0``
    are indistinguishable in the weight -- and such a node carries no mass, hence no
    force. Masking by mass is what makes the two lanes agree on a padded domain:
    padding rows are massless, so they cannot raise a cell's rung on either side.

    Parameters
    ----------
    node_ranges:
        ``(nodes, 2)`` inclusive ``[start, end]`` particle range per node.
    rung:
        ``(n,)`` per-particle rung, in the same order the ranges index.
    masses:
        ``(n,)`` per-particle mass, same order.
    k_max:
        Highest level; levels run ``0 .. k_max``.

    Returns
    -------
    Array
        ``(nodes,)`` node rung, ``-1`` where a node holds no massive particle.
    """
    r = jnp.asarray(rung).astype(INDEX_DTYPE)
    # A massless particle exerts no force, so it must not decide a cell's level.
    eff = jnp.where(jnp.asarray(masses) > 0, r, jnp.asarray(-1, INDEX_DTYPE))
    ks = jnp.arange(int(k_max) + 1, dtype=INDEX_DTYPE)
    ge = (eff[None, :] >= ks[:, None]).astype(INDEX_DTYPE)
    zero = jnp.zeros((int(k_max) + 1, 1), dtype=INDEX_DTYPE)
    prefix = jnp.concatenate([zero, jnp.cumsum(ge, axis=1)], axis=1)
    lo = node_ranges[:, 0]
    hi = node_ranges[:, 1]
    has = (prefix[:, hi + 1] - prefix[:, lo]) > 0
    return jnp.sum(has.astype(INDEX_DTYPE), axis=0) - 1


def _far_payload(node_multipoles, node_rung):
    """The frontier payload for the cross far field: multipoles, then the cell rung.

    Both are per-LEAF quantities, so the frontier is the right channel: it rides one
    ``all_gather``, one row per remote leaf. The rung is appended rather than sent
    separately so it cannot be attached to the wrong leaf -- the coarse tree's Morton
    reorder moves whole payload rows, so a column stays with its multipole by
    construction.

    Parameters
    ----------
    node_multipoles:
        ``(nodes, sh_size(order))`` per-node multipoles.
    node_rung:
        ``(nodes,)`` per-node cell rung, or ``None`` when unweighted.

    Returns
    -------
    Array
        ``(nodes, sh_size(order))`` unweighted, or one column wider when weighted.
    """
    if node_rung is None:
        return node_multipoles
    return jnp.concatenate(
        [node_multipoles, node_rung.astype(node_multipoles.dtype)[:, None]], axis=1
    )


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
    cross_theta: float = 0.0,
    far_cap: int = 1,
    far_recv_capacity: Optional[int] = None,
    node_multipoles: Optional[Array] = None,
    expansion_centers: Optional[Array] = None,
    tree_arrays: Optional[Any] = None,
    rung: Optional[Array] = None,
    level_weights: Optional[Array] = None,
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
    cross_theta:
        Opening angle for the CROSS-domain MAC. ``0.0`` accepts nothing, so every
        cross pair refines to leaf-leaf and is summed exactly -- the behaviour this
        lane shipped with. Above zero, accepted pairs go through M2L against the
        remote leaf's multipole, which requires ``node_multipoles``,
        ``expansion_centers`` and ``tree_arrays``.
    far_cap:
        Output capacity for the cross FAR list. Irrelevant at ``cross_theta = 0``,
        where the list cannot receive an entry.
    far_recv_capacity:
        Receive capacity for the expansion exchange, in rows of
        ``sh_size(order)``. ``None`` takes ``(ndev - 1) * (2L - 1) * L`` for ``L``
        local leaves, which is the worst case rather than a guess.

        **Starving this one breaks MOMENTUM, not just accuracy**, which is not true
        of any other capacity here. Everywhere else an overflow drops a pair before
        it is evaluated, so both halves vanish together and the global sum stays
        exact while the force is merely incomplete. A far pair's ``+f`` is applied
        locally the moment it is computed and only the ``-f`` travels, so dropping
        a received row leaves the ``+f`` behind with nothing to cancel it. Measured:
        at ``cross_theta = 2`` on 4 devices the momentum residual went from 1.7e-2
        with a starved receive to 2.3e-17 with this bound.
    node_multipoles:
        ``(nodes, sh_size(order))`` this domain's multipoles, about
        ``expansion_centers``. Required when ``cross_theta > 0``.
    expansion_centers:
        ``(nodes, 3)`` the centres ``node_multipoles`` are expanded about. Separate
        from ``centers`` on purpose: ``centers``/``radii`` are the MAC's extents,
        while these have to be the *same* points the upward sweep used, or an M2L
        delta and the expansion it translates disagree. This is also what the
        frontier publishes as each leaf's COM, so a received expansion is applied
        at exactly the point it was expanded about -- bit-for-bit, because it is
        the same array value that crossed the wire, not a recomputation of it.
    tree_arrays:
        This domain's ``MutualTreeArrays``, for the L2L level schedule and the L2P.
        Required when ``cross_theta > 0``.
    rung:
        ``(n,)`` per-particle block-step rung for this domain, in tree order.
        Required whenever ``level_weights`` is given, ignored without it.
    level_weights:
        ``(k_max + 1,)`` weight per interaction level; every cross pair is scaled by
        ``level_weights[max(rung_i, rung_j)]``. ``None`` weights every level by one,
        which is the full cross-domain force and the behaviour this lane shipped with.

        **May be a tracer**, and is treated as one: nothing here branches on its
        value, so a caller can drive the whole sub-step boundary from a ``lax.scan``
        indexing :func:`jaccpot.mutual.force.boundary_weight_table` with a traced
        boundary index. Only its *length* is read, at trace time.

        Momentum survives the weighting on both halves, for two different reasons
        worth keeping apart. On the near half the weight is a single symmetric scalar
        per pair multiplying the one tensor both sides read (:func:`_tile_forces`).
        On the far half the ``-f`` is not a force at all but a local EXPANSION
        returned to the owner's node, so the weight is applied ONCE, on the
        evaluating device, to both directions of the batched M2L before either is
        exported -- never again on import, which would square it, and never on
        neither, which would leave the far half unweighted while the near half is
        weighted. That last one breaks the level partition WITHOUT breaking momentum,
        so no momentum residual can catch it; the level-partition test is what does.
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
        Total acceleration for this domain, owned cross near and far pair counts, and
        the across-device overflow flag.

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

    # A structural decision, not a value one: `level_weights` may be a tracer, so
    # only "was one given at all" may be branched on.
    weighted = level_weights is not None
    k_max = 0
    if weighted:
        if rung is None:
            raise ValueError(
                "level_weights needs rung: the weight of a pair is "
                "level_weights[max(rung_i, rung_j)], which cannot be formed without "
                "a per-particle rung"
            )
        level_weights = jnp.asarray(level_weights, dtype=positions.dtype)
        if level_weights.ndim != 1:
            raise ValueError(
                "level_weights must be a (k_max + 1,) vector, one weight per "
                f"interaction level; got shape {tuple(level_weights.shape)}. A caller "
                "scanning boundaries passes one ROW of boundary_weight_table, not the "
                "whole table"
            )
        rung = jnp.asarray(rung).astype(INDEX_DTYPE)
        k_max = int(level_weights.shape[0]) - 1

    do_far = float(cross_theta) > 0.0
    if do_far and (
        node_multipoles is None or expansion_centers is None or tree_arrays is None
    ):
        raise ValueError(
            "cross_theta > 0 routes accepted cross pairs through M2L, which needs "
            "node_multipoles, expansion_centers and tree_arrays; got "
            f"{'' if node_multipoles is not None else 'node_multipoles '}"
            f"{'' if expansion_centers is not None else 'expansion_centers '}"
            f"{'' if tree_arrays is not None else 'tree_arrays '}"
            "missing"
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

    # Cell rungs, for the FAR half only -- the near half has the exact per-particle
    # predicate and does not need them. Computed for every node because the far
    # pair's LOCAL endpoint may be an internal node: `accept_only_leaf_pairs`
    # constrains the remote side, which has to be addressable, not this one.
    node_rung = (
        _node_rungs(node_ranges, rung, masses, k_max) if (weighted and do_far) else None
    )

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
        # The COM the frontier publishes is the point a remote device will expand
        # about AND the point this device will apply a returned expansion at, so
        # with far work on it must be the upward sweep's own centres -- the same
        # array value shipped, not a second computation of the same quantity that
        # agrees only to 1e-13.
        expansion_centers if do_far else centers,
        node_payload=_far_payload(node_multipoles, node_rung) if do_far else None,
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
        float(cross_theta),
        this_device=me,
        remote_owner=c_owner,
        remote_index_in_owner=c_node_id,
        # Unconditional, not just when far work is on: the remote tree here is
        # ALWAYS a merged coarse tree, so an internal node never has an address in
        # its owner's own tree. At cross_theta = 0 nothing is accepted anyway, so
        # this states the invariant rather than changing behaviour.
        accept_only_leaf_pairs=True,
        max_pair_queue=max_pair_queue,
        far_cap=int(far_cap) if do_far else 1,
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
    # The remote endpoint's rung has to travel with the remote endpoint. The near
    # predicate is exact and per-particle, so a per-LEAF channel will not do -- and
    # the frontier is the wrong channel anyway: it is all_gather-ed, so publishing
    # `leaf_width` rung columns there would ship every remote particle's rung to
    # every device, O(N_total), which is what the demand-driven import exists to
    # avoid. It rides round B of the import instead, sized by the halo.
    # Passed CONDITIONALLY rather than as `payload_sorted=None`, so the UNWEIGHTED
    # lane still runs against a yggdrax that predates the parameter
    # (TobiBu/yggdrax#53) -- only the rung-weighted path needs it. A weighted call
    # against an older yggdrax raises a TypeError, which is the right failure: loud,
    # immediate, and naming the argument.
    halo_kwargs: dict[str, Any] = {}
    if weighted:
        halo_kwargs["payload_sorted"] = rung.astype(positions.dtype)[:, None]
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
        **halo_kwargs,
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

    near_w = None
    if weighted:
        # Integers survive the float round trip exactly -- a rung is a small
        # non-negative int and every dtype here holds those without loss -- so this
        # recovers the sender's own value rather than approximating it. A padding
        # slot arrives as 0.0 and is masked out by `okb` regardless.
        halo_rung = jnp.round(halo.payload[:, 0]).astype(INDEX_DTYPE)
        near_w = _pair_level_weights(level_weights, rung[ia], halo_rung[jb])

    f_a, f_b = _tile_forces(
        positions[ia],
        jnp.where(oka, masses[ia], 0.0),
        oka,
        halo.positions[jb],
        jnp.where(okb, halo.masses[jb], 0.0),
        okb,
        softening,
        g,
        weights=near_w,
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

    # --- 7. the cross FAR field: M2L against remote leaf multipoles -----------
    # Only reachable at cross_theta > 0. Structured as a SEPARATE additive pass rather
    # than an injection into the intra-domain far field, which is exact rather than
    # merely convenient: `_push_locals_down` and `_l2p_forces` are both linear in the
    # local coefficients, so pushing a cross-only `locals_` down and evaluating it is
    # identical to adding those coefficients before the domain's own push-down. That
    # keeps `mutual_far_field_forces` untouched.
    far_pairs = jnp.zeros((), dtype=INDEX_DTYPE)
    if do_far:
        from jaccpot.mutual.farfield import _l2p_forces, _m2l_batch, _push_locals_down
        from jaccpot.operators._sh_indexing import sh_size

        cap_f = int(far_cap)
        p_order = int(tree_arrays.order)
        n_nodes = int(expansion_centers.shape[0])
        # A device receives, from each of the other ndev-1, at most one expansion
        # per (their local node, one of MY leaves) pair -- so (2L-1)*L each. Sized
        # at that bound rather than optimistically, because a starved receive here
        # is worse than anywhere else in this lane: see the note below.
        far_recv = (
            (ndev - 1) * (2 * n_leaves - 1) * n_leaves
            if far_recv_capacity is None
            else int(far_recv_capacity)
        )

        frows = jnp.arange(cap_f, dtype=INDEX_DTYPE)
        f_on = frows < res.far_count
        f_local = jnp.where(f_on, res.far_local, root)
        f_cpos = c_first[jnp.where(f_on, res.far_remote, c_root)]
        # Same massless-leaf rule as the near path: no origin node id, no address.
        f_live = f_on & (rct.tag_node_id[f_cpos] >= 0)
        far_pairs = jnp.sum(f_live.astype(INDEX_DTYPE))

        # The coarse "particle" position IS the remote leaf's COM, which is the point
        # its multipole is expanded about -- so it is read from `positions_sorted`
        # rather than from the coarse tree's own moments, which would be a second
        # computation of the same point.
        c_leaf_com = rct.positions_sorted[f_cpos]
        tgt_c = expansion_centers[f_local]

        # BOTH directions in ONE batch, as `_dual_m2l` does for the same reason it
        # gives: same kernel, same order, same rounding for both halves of a pair is
        # what makes F_A + F_B cancel algebraically rather than to the M2L's accuracy.
        # Sliced from the END, so the split cannot drift out of step with
        # `_far_payload`: the rung is the last column BY CONSTRUCTION there. Indexing
        # it at a computed `sh_size(order)` offset instead would state the multipole
        # width twice, and a mismatch would read a coefficient as a rung -- which
        # survives the clamp as a perfectly valid pair level, so it is a wrong force
        # that momentum, the level partition and the linearity check are all blind to.
        pay = rct.payload[f_cpos]
        remote_mp = pay[:, :-1] if weighted else pay
        zero_mp = jnp.zeros_like(remote_mp)
        mp = jnp.concatenate(
            [
                jnp.where(f_live[:, None], remote_mp, zero_mp),
                jnp.where(f_live[:, None], node_multipoles[f_local], zero_mp),
            ]
        )
        dl = jnp.concatenate([tgt_c - c_leaf_com, c_leaf_com - tgt_c])
        loc = _m2l_batch(mp, dl, order=p_order, use_pallas=False, interpret=False)
        loc_here, loc_there = loc[:cap_f], loc[cap_f:]

        if weighted:
            # ONE weight per pair, applied ONCE, here -- on the device that evaluated
            # the pair, to both directions, before either leaves. The importing side
            # only adds what arrives.
            #
            # The cell rungs the two endpoints bring are each computed exactly once,
            # by the device that owns the particles: the local node's from this
            # domain's `rung`, the remote leaf's by its owner, published on the
            # frontier. Neither is recomputed on the far side of the wire, so the two
            # devices cannot disagree about which level a pair belongs to -- which is
            # the failure a momentum residual is structurally blind to.
            far_rung = jnp.round(pay[:, -1]).astype(INDEX_DTYPE)
            f_level = jnp.maximum(node_rung[f_local], far_rung)
            f_level = jnp.clip(f_level, 0, int(level_weights.shape[0]) - 1)
            f_w = jnp.take(level_weights, f_level, axis=0, mode="clip")[:, None]
            # Legal because everything downstream of an expansion is LINEAR in its
            # coefficients: L2L re-centres and L2P evaluates, so scaling the
            # coefficients scales the resulting force by the same factor. Weighting
            # the expansion is weighting the pair.
            loc_here = loc_here * f_w
            loc_there = loc_there * f_w

        # `+f` lands on a local node; `-f` is addressed to a NODE in the owner's own
        # tree (`tag_node_id`), not to a particle -- which is the whole reason far
        # pairs may only be accepted against coarse leaves.
        locals_ext = (
            jnp.zeros((n_nodes, sh_size(p_order)), dtype=positions.dtype)
            .at[jnp.where(f_live, f_local, n_nodes)]
            .add(jnp.where(f_live[:, None], loc_here, 0.0), mode="drop")
        )
        rev_far = export_reverse_halo(
            jnp.where(f_live, rct.tag_domain[f_cpos], -1),
            jnp.where(f_live, rct.tag_node_id[f_cpos], 0),
            jnp.where(f_live[:, None], loc_there, 0.0),
            ndev,
            recv_capacity=far_recv,
            axis_name=axis_name,
        )
        overflow = overflow | rev_far.overflow
        locals_ext = apply_reverse_halo(locals_ext, rev_far)

        locals_ext = _push_locals_down(locals_ext, expansion_centers, tree_arrays)
        # `_l2p_forces` returns FORCES -- the mass factor is already in -- while this
        # function's near path returns accelerations, so the division is here. Guarded
        # because padding rows have zero mass and 0/0 would poison the whole column.
        f_far = _l2p_forces(
            positions, masses, expansion_centers, locals_ext, tree_arrays
        )
        massive = masses > 0
        safe_m = jnp.where(massive, masses, jnp.ones_like(masses))
        acc = acc + jnp.where(
            massive[:, None],
            f_far / safe_m[:, None] * jnp.asarray(g, positions.dtype),
            0.0,
        )

    # Shape (1,) rather than scalar for the two diagnostics: `shard_map` cannot
    # concatenate rank-0 per-device outputs, so a function documented as "call inside
    # shard_map" should hand back something its `out_specs` can actually describe.
    # The caller sees `(ndev,)` after the map.
    return DistributedMutualResult(
        acceleration=acc,
        cross_pairs=jnp.sum(live.astype(INDEX_DTYPE))[None],
        cross_far_pairs=far_pairs[None],
        overflow=(jax.lax.pmax(overflow.astype(INDEX_DTYPE), axis_name) > 0)[None],
    )


# ---------------------------------------------------------------------------
# The driver: everything above is one device's share, called inside shard_map.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributedMutualConfig:
    """Everything the distributed mutual force needs beyond the particles.

    ``theta`` is the **intra-domain** opening angle and nothing else. Cross-domain
    interactions are exact regardless of it: the cross walk is driven at ``theta = 0``
    (see the module docstring), so this knob trades accuracy against work *within* a
    domain only. Turning it up does not make the boundary cheaper.

    The capacity defaults are heuristics, not measurements, and they exist so a first
    call works rather than because they are right. Under-provisioning is the safe
    direction here -- it raises ``local_overflow``/``overflow`` loudly rather than
    quietly returning a wrong force -- but it costs a rerun, so a repeated run at a
    known size should carry measured ones. :mod:`jaccpot.mutual.cap_presets` is the
    table for that, keyed on ``(per-device N, ndev, leaf_size, theta, order)``, and
    ``caps`` is where its answer goes.

    Attributes
    ----------
    leaf_size : int
        Particles per leaf, for the per-device local tree.
    theta : float
        INTRA-domain opening angle.
    cross_theta : float
        CROSS-domain opening angle. ``0.0`` (the default) makes every cross pair
        exact -- nothing is accepted, so all of them refine to leaf-leaf and are
        summed particle by particle. Above zero, accepted cross pairs go through
        M2L against the remote leaf's multipole, which is what collapses the
        import from the whole remote system to a surface halo.
    far_cap, far_recv_capacity : Optional[int]
        Cross far-list and expansion-receive capacities. ``None`` derives each.
    order : int
        Multipole order for the intra-domain far field.
    k_max : Optional[int]
        Highest block-step rung, when this lane is driven with rungs. Used only to
        CHECK the caller: ``level_weights`` must have ``k_max + 1`` entries and every
        rung must lie in ``[0, k_max]``. ``None`` skips both checks and takes the
        level count from ``level_weights`` itself. Setting it is worth it -- a rung
        above the table's length has no weight, so the traversal would have to invent
        one or drop the interaction, and either quietly integrates the wrong
        equations. Clamping makes it silent; this makes it a configuration error.
    softening : float
        Plummer softening length, shared by both halves.
    g : float
        Gravitational constant.
    caps : Optional[MutualCapacities]
        Capacities for the intra-domain lane. ``None`` derives them.
    near_cap, max_pair_queue, recv_capacity, max_req_leaves, max_recv_leaves : Optional[int]
        Cross-domain capacities. ``None`` derives each from ``(ndev, leaf_size)`` and
        the per-device capacity.
    coarse_depth_cap : int
        Owner-propagation rounds up the coarse tree.
    backend : str
        ``"jax"`` (the default) runs the pure-JAX kernels everywhere. ``"pallas"``
        routes the INTRA-domain near field through jaccpot's mutual P2P kernel where
        the hardware supports it -- worth **2.2-3.6x forward and 3.1-4.1x reverse** on
        one device, and at N = 1e5 per device the difference between the reverse pass
        running and requesting a 30 GiB allocation.

        The routing mirrors the single-device lane's exactly, which means two things
        that look like omissions and are not:

        * **The far field stays pure JAX.** Both Pallas M2L shapes measured *slower*
          there (0.84x / 0.61x forward, worse at larger N) -- see
          ``docs/momentum_conserving_fmm.md``, Phase 5 -- so ``backend="pallas"``
          selects pure JAX for it on hardware, on this lane as on the other.
        * **The gate is the hardware, not the flag.** Pallas needs sm_80+; below it
          the near field falls back to pure JAX rather than failing. Every Pallas
          lane is reached through its ``custom_vjp`` wrapper, never the bare
          ``pallas_call``, which has no JVP or transpose rule and would be silently
          non-differentiable.

        The CROSS-domain near field is pure JAX either way. It is a different kernel
        (:func:`_tile_forces`, a local-leaf x halo-block tile) and giving it a Pallas
        lane is a new kernel, not a wiring change.
    pallas_interpret : bool
        Run the Pallas kernels in interpret mode, which works without a GPU. That is
        what lets the ``backend="pallas"`` path be exercised on CPU at all -- but
        interpret mode validates the kernel's LOGIC, not its lowerability: it runs the
        jaxpr under CPU semantics and accepts primitives the Triton backend does not
        implement. Two mutual kernels passed every CPU interpret test and then failed
        on their first GPU run. Only a real GPU run stands between a lowering
        regression and a broken ``backend="pallas"``.
    partitioner : str
        How particles are assigned to devices -- ``"rcb"`` (the default here) or
        ``"morton"``. See :func:`jaccpot.distributed.fmm.partition_for_devices`.

        ``"rcb"`` keeps a flattened system's near neighbours on one device where a
        Morton split scatters them -- a thickness-0.4 disk on 4 devices goes from 0.509
        of each particle's nearest neighbours living off-device to 0.046. It is now the
        shared default too, so this field only exists to override it.
    """

    leaf_size: int = 32
    theta: float = 0.5
    order: int = 4
    k_max: Optional[int] = None
    softening: float = 1e-3
    g: float = 1.0
    caps: Optional[Any] = None
    cross_theta: float = 0.0
    far_cap: Optional[int] = None
    far_recv_capacity: Optional[int] = None
    near_cap: Optional[int] = None
    max_pair_queue: Optional[int] = None
    recv_capacity: Optional[int] = None
    max_req_leaves: Optional[int] = None
    max_recv_leaves: Optional[int] = None
    coarse_depth_cap: int = 64
    backend: str = "jax"
    pallas_interpret: bool = False
    partitioner: str = "rcb"


class DistributedMutualForce(NamedTuple):
    """What the driver returns.

    ``accelerations`` is in the caller's INPUT order, not any internal one: the driver
    owns two permutations (the SFC split across devices, and each device's local tree
    sort) and undoes both, so a caller never sees either.

    ``overflow`` is one bool for the whole system, already reduced across devices,
    because a capacity exceeded anywhere is a wrong force everywhere. The per-device
    arrays are carried alongside so a caller that has to *raise* a capacity can see
    which one and where.

    Attributes
    ----------
    accelerations : Array
        ``(n, 3)`` in input order. A jax array: the readout is a scatter, not a NumPy
        assignment, so a whole evaluation is traceable and an integrator can drive it
        from a ``lax.scan``.
    cross_pairs : Array
        ``(ndev,)`` owned cross-domain NEAR leaf pairs per device.
    cross_far_pairs : Array
        ``(ndev,)`` owned cross-domain FAR pairs per device -- zero unless
        ``cross_theta > 0``. Carried out because a test of the cross far field is
        vacuous without it: several tests in this repo once passed on a
        configuration that produced no far pairs at all.
    overflow : bool
        True if any capacity, on any device, in either half, was exceeded. A Python
        ``bool`` when the inputs are concrete; under trace it stays a traced scalar,
        which a caller cannot branch on -- so a traced driver must check it outside
        the trace, or use :class:`~jaccpot.nornax_adapter.DistributedBlockStepFMM`,
        which raises on it eagerly.
    cross_overflow : Array
        ``(ndev,)`` the cross-domain half's flag.
    local_overflow : Array
        ``(ndev,)`` the intra-domain half's flag.
    local_overflow_causes : Array
        ``(ndev,)`` packed cause bits from the local topology; decode against
        ``jaccpot.mutual.force.OVERFLOW_CAUSES``.
    """

    accelerations: Array
    cross_pairs: Array
    cross_far_pairs: Array
    overflow: bool
    cross_overflow: Array
    local_overflow: Array
    local_overflow_causes: Array


def _default_caps(per_device_n: int, leaf_size: int) -> Any:
    """Heuristic intra-domain capacities for a first call at this size.

    Deliberately generous on the pair lists and tight on ``depth``: a pair list is
    linear in the leaf count and cheap to over-provision, while ``depth`` sets the
    iteration count of the M2M/L2L cascade scan, so padding it costs work on every
    evaluation (the reasoning :mod:`jaccpot.mutual.cap_presets` records for its
    per-field scaling).

    Parameters
    ----------
    per_device_n : int
        Padded particle capacity of one device.
    leaf_size : int
        Particles per leaf.

    Returns
    -------
    Any
        A ``MutualCapacities``.
    """
    from jaccpot.mutual.force import MutualCapacities

    n_leaves = max(1, -(-int(per_device_n) // int(leaf_size)))
    return MutualCapacities(
        # ~64 near neighbours per leaf is the 3D close-packing order of magnitude;
        # the far list is the same order at moderate theta.
        near=max(1024, 64 * n_leaves),
        far=max(1024, 64 * n_leaves),
        # A balanced tree over n_leaves leaves is log2(n_leaves) deep; +8 absorbs the
        # imbalance a real distribution puts on top of that.
        depth=max(16, int(n_leaves).bit_length() + 8),
        width=max(256, 4 * n_leaves),
        queue=max(1 << 16, 8 * n_leaves),
    )


class DistributedMutualEvaluator:
    """A distributed mutual force compiled ONCE, at a frozen partition and capacities.

    :func:`distributed_mutual_fmm` does everything on every call: it re-partitions the
    particles, rebuilds the mapped program and hands it to ``jax.jit``. Since
    ``shard_map`` wraps a fresh closure each time, that is a fresh cache key and so a
    fresh compile -- measured in the tens of seconds even at test sizes, and the shared
    plan's own measurement note says ~200 s at moderate N on the sibling lane. Fine for
    one force; ruinous for a block-step base step, which asks for ``n_sub + 1``
    evaluations of the SAME program on moved particles.

    So the two halves are separated. This object owns the host side -- the domain
    assignment, the padding layout, the global bounds and the capacities -- and holds
    the compiled program; calling it evaluates live positions against them. It is the
    distributed counterpart of what ``topology_backend="device"`` plus
    :meth:`BlockStepFMM.freeze_template` do on one device: freeze the shapes once, then
    every later evaluation is the same program.

    **What is frozen, stated exactly, because it is not the topology.** The partition
    (which particle belongs to which device), the padding layout, the tree bounds and
    every capacity are fixed at construction. The TREE is not: each call rebuilds it
    inside the mapped program from the positions handed in, which is traceable and
    static-shape at fixed capacities. So successive calls within a base step see
    slightly different accepted pair sets -- more often than the single-device lane,
    which holds one topology for the whole base step. That is a *finer* rebuild
    cadence, not a coarser one, and it changes nothing about legality: every call is
    internally self-consistent, so its levels partition its own pairs and each level
    conserves momentum exactly. What it does mean is that the per-level decomposition
    of two different boundaries is taken on two slightly different trees.

    Rebuild this object when the partition should change -- the system's extent has
    moved materially, or the domains have gone badly out of balance. The bounds are
    padded by 1e-6 of the span at construction and a particle that drifts outside them
    is clamped by the Morton encoding rather than lost, so the failure is a gradual
    loss of partition quality, not a wrong force.

    Two programs are compiled at most, lazily: the unweighted force takes three
    operands and the weighted one five, so they are structurally different graphs. The
    weight VALUES are not: one compile serves every boundary of a base step, and every
    rung assignment, because only shapes are static.
    """

    def __init__(
        self,
        *,
        config: DistributedMutualConfig,
        mesh: Any,
        ndev: int,
        part: dict,
        build: Any,
    ) -> None:
        """Hold the frozen partition and the program builder.

        Parameters
        ----------
        config : DistributedMutualConfig
            Physics and capacities, already validated.
        mesh : Any
            The device mesh the program is mapped over.
        ndev : int
            Device count.
        part : dict
            The partition from :func:`~jaccpot.distributed.fmm.partition_for_devices`.
        build : Any
            ``build(weighted) -> callable``, the mapped (and possibly jitted) program.
        """
        import numpy as np

        self.config = config
        self.mesh = mesh
        self.ndev = int(ndev)
        self.n = int(part["n"])
        self.cap = int(part["cap"])
        self.bounds = part["bounds"]
        self._build = build
        self._programs: dict = {}

        gid = np.asarray(part["gid_flat"]).reshape(-1).astype(np.int64)
        self.gid_flat = gid
        # Where each row of the layout takes its position from. Real rows take their
        # own particle; a padding row takes its domain's FIRST particle, which is
        # where `partition_for_devices` puts it -- at a real point, so the tree build
        # sees no phantom geometry. Derived from `gid_flat` rather than reimplementing
        # the split, then checked against the partition's own arrays below, so this
        # cannot drift away from it silently.
        g2 = gid.reshape(self.ndev, self.cap)
        first = g2[:, 0]
        self.source_index = np.where(g2 >= 0, g2, first[:, None]).reshape(-1)

        pos_ref = np.asarray(part["pos_flat"])
        mass_ref = np.asarray(part["mass_flat"])
        pos_np = np.asarray(part["_positions"])
        mass_np = np.asarray(part["_masses"])
        if not np.array_equal(pos_np[self.source_index], pos_ref):
            raise AssertionError(
                "the reconstructed layout does not reproduce partition_for_devices' "
                "own pos_flat -- the padding rule changed and this object would then "
                "evaluate a different system than the one it was built for"
            )
        if not np.array_equal(
            np.where(gid >= 0, mass_np[self.source_index], 0.0), mass_ref
        ):
            raise AssertionError(
                "the reconstructed layout does not reproduce partition_for_devices' "
                "own mass_flat"
            )

        # Every real particle must appear in the layout exactly once. This is a
        # property of the frozen partition alone, so it is checked here rather than
        # on every call -- which is also what makes the readout traceable: there is
        # no per-call host predicate left to branch on.
        live = gid[gid >= 0]
        if live.shape[0] != self.n or np.unique(live).shape[0] != self.n:
            raise AssertionError(
                f"the partition names {np.unique(live).shape[0]} distinct particles "
                f"in {live.shape[0]} live rows, for {self.n} particles -- a padding "
                "or partition bug, not a physics one"
            )
        # Row -> output particle, with padding rows aimed past the end so a scatter
        # in `mode="drop"` discards them.
        self._scatter_index = jnp.asarray(np.where(gid >= 0, gid, self.n))

    def layout(self, positions: Any, masses: Any) -> tuple[Array, Array]:
        """Lay live positions and masses out in the frozen partition's row order.

        Parameters
        ----------
        positions : Any
            ``(n, 3)`` positions in the caller's order.
        masses : Any
            ``(n,)`` masses, same order.

        Returns
        -------
        tuple[Array, Array]
            ``(pos_flat, mass_flat)``, each ``(ndev * cap, ...)``.

        Raises
        ------
        ValueError
            If either input has the wrong length for the frozen partition.
        """
        positions = jnp.asarray(positions)
        masses = jnp.asarray(masses)
        if int(positions.shape[0]) != self.n or int(masses.shape[0]) != self.n:
            raise ValueError(
                f"this evaluator was built for {self.n} particles; got "
                f"{int(positions.shape[0])} positions and {int(masses.shape[0])} masses"
            )
        src = jnp.asarray(self.source_index)
        live = jnp.asarray(self.gid_flat) >= 0
        # Padding rows keep a real POSITION and zero MASS, exactly as the partition
        # built them: a massless row exerts and feels nothing, while a phantom
        # position would stretch every node's extent.
        return positions[src], jnp.where(live, masses[src], 0.0)

    def rung_layout(self, rung: Any) -> Array:
        """Lay a per-particle rung out in the frozen partition's row order.

        Parameters
        ----------
        rung : Any
            ``(n,)`` per-particle rung in the caller's order.

        Returns
        -------
        Array
            ``(ndev * cap,)`` rung, with padding rows at 0.

        Raises
        ------
        ValueError
            If ``rung`` has the wrong length, or a value outside ``[0, k_max]`` when
            the config names one.
        """
        r = jnp.asarray(rung).reshape(-1)
        if int(r.shape[0]) != self.n:
            raise ValueError(
                f"rung has {int(r.shape[0])} entries for {self.n} particles"
            )
        k_max = self.config.k_max
        if k_max is not None:
            # Attempted, not gated on `isinstance`, for the reason `__call__` gives:
            # a concrete array closed over by a scan body is not a Tracer but cannot
            # be read either. Under trace the range check is skipped and the kernels'
            # clamp is what stands -- so passing this is not proof the rungs are in
            # range, exactly as `BlockStepFMM._validate_rung` documents.
            try:
                lo, hi = int(jnp.min(r)), int(jnp.max(r))
            except Exception:  # pragma: no cover - only reachable under trace
                lo, hi = 0, int(k_max)
            if lo < 0 or hi > int(k_max):
                raise ValueError(
                    f"rung values must lie in [0, k_max={int(k_max)}]; got [{lo}, {hi}]"
                )
        # Padding takes rung 0, which is inert: a padding row is massless, so it
        # cannot raise any cell's rung and its own pairs carry no force.
        live = jnp.asarray(self.gid_flat) >= 0
        return jnp.where(live, r[jnp.asarray(self.source_index)], 0)

    def _program(self, weighted: bool) -> Any:
        """Return the compiled program for this weighting mode, building it once."""
        key = bool(weighted)
        if key not in self._programs:
            self._programs[key] = self._build(key)
        return self._programs[key]

    def __call__(
        self,
        positions: Any,
        masses: Any,
        *,
        rung: Any = None,
        level_weights: Any = None,
    ) -> DistributedMutualForce:
        """Evaluate the force on live positions at the frozen partition.

        Parameters
        ----------
        positions : Any
            ``(n, 3)`` positions in the caller's order.
        masses : Any
            ``(n,)`` masses, same order.
        rung : Any
            ``(n,)`` per-particle block-step rung. Required with ``level_weights``.
        level_weights : Any
            ``(k_max + 1,)`` weight per interaction level; ``None`` gives the full
            force. May be a traced array as far as this object is concerned -- only
            its length is read -- so a caller can drive a base step's boundaries from
            one weight table.

        Returns
        -------
        DistributedMutualForce
            Accelerations in the caller's input order, plus the per-device
            diagnostics. **Check ``.overflow``.**

        Raises
        ------
        ValueError
            If ``level_weights`` is given without ``rung``, has the wrong rank, or
            disagrees with ``config.k_max``.
        RuntimeError
            If a real particle is missing from the assembled result, which is a
            padding or capacity bug rather than a physics one.
        """
        import numpy as np

        weighted = level_weights is not None
        if weighted and rung is None:
            raise ValueError(
                "level_weights needs rung: a pair's weight is "
                "level_weights[max(rung_i, rung_j)]"
            )
        operands = list(self.layout(positions, masses))
        operands.append(jnp.asarray(self.gid_flat))
        if weighted:
            lw = jnp.asarray(level_weights)
            if lw.ndim != 1:
                raise ValueError(
                    "level_weights must be a (k_max + 1,) vector; got shape "
                    f"{tuple(lw.shape)}"
                )
            k_max = self.config.k_max
            if k_max is not None and int(lw.shape[0]) != int(k_max) + 1:
                raise ValueError(
                    f"level_weights has {int(lw.shape[0])} entries but "
                    f"k_max={int(k_max)} asks for {int(k_max) + 1}"
                )
            operands += [self.rung_layout(rung), lw]

        acc_o, _gid_o, cross_o, farx_o, xof_o, lof_o, causes_o = self._program(
            weighted
        )(*operands)

        # Back to input order with a scatter rather than a NumPy assignment, so this
        # whole call is traceable and an integrator can drive it from a `lax.scan`.
        # Padding rows are aimed past the end and dropped; that a real particle is
        # never among them was established when the partition was frozen, so there is
        # no per-call check to make here.
        out = (
            jnp.zeros((self.n, 3), acc_o.dtype)
            .at[self._scatter_index]
            .set(acc_o, mode="drop")
        )

        cross_of = xof_o.reshape(-1).astype(jnp.bool_)
        local_of = lof_o.reshape(-1).astype(jnp.bool_)
        overflow = jnp.any(cross_of) | jnp.any(local_of)
        # Attempted, not gated on `isinstance(..., Tracer)`: a CONCRETE array closed
        # over by a `lax.scan` body is not a Tracer, yet reducing it inside the trace
        # yields one, so `bool(...)` is the only test that actually asks "can this be
        # read here". Same discipline as `BlockStepFMM._validate_rung`. Under trace it
        # stays a traced scalar and the caller cannot branch on it -- which is why
        # `DistributedBlockStepFMM` reads it eagerly and raises.
        try:
            overflow = bool(overflow)
        except Exception:  # pragma: no cover - only reachable under trace
            pass
        return DistributedMutualForce(
            accelerations=out,
            cross_pairs=cross_o.reshape(-1),
            cross_far_pairs=farx_o.reshape(-1),
            overflow=overflow,
            cross_overflow=cross_of,
            local_overflow=local_of,
            local_overflow_causes=causes_o.reshape(-1),
        )


def make_distributed_mutual_evaluator(
    positions: Any,
    masses: Any,
    *,
    config: Optional[DistributedMutualConfig] = None,
    mesh: Any = None,
    ndev: Optional[int] = None,
    jit: bool = True,
) -> DistributedMutualEvaluator:
    """Partition once, compile once, and return something callable on live inputs.

    Use this instead of :func:`distributed_mutual_fmm` for anything that evaluates the
    force more than once at a fixed particle count -- a rollout, a boundary sweep, a
    timing loop. ``distributed_mutual_fmm`` is this function plus one call, and pays a
    fresh partition and a fresh compile every time it is invoked.

    ``positions``/``masses`` here are used for the PARTITION and the bounds. The
    returned object may be called on moved particles; see
    :class:`DistributedMutualEvaluator` for exactly what that freezes and what it
    does not.

    Parameters
    ----------
    positions : Any
        ``(n, 3)`` positions in any order.
    masses : Any
        ``(n,)`` masses, same order.
    config : Optional[DistributedMutualConfig]
        Physics and capacities. ``None`` uses the defaults, whose capacities are
        heuristics -- see :class:`DistributedMutualConfig`.
    mesh : Any
        Device mesh. ``None`` builds one over ``ndev`` devices.
    ndev : Optional[int]
        Device count when ``mesh`` is ``None``. ``None`` uses every visible device.
    jit : bool
        Compile the mapped program. Default True.

    Returns
    -------
    DistributedMutualEvaluator
        Callable on live positions and masses at the frozen partition.

    Raises
    ------
    ValueError
        If fewer than two devices are available, ``n < ndev``, or ``config.backend``
        is not a known name.
    """
    import numpy as np

    # `jax.shard_map` directly, with no `jax.experimental` fallback. The fallback was
    # not merely dead but wrong: `pyproject.toml` pins `jax>=0.10.2,<0.11` and the
    # floor itself exports `jax.shard_map`, so the `except ImportError` branch is
    # unreachable across the whole supported range -- and had it ever been taken it
    # would have raised, because `jax.experimental.shard_map.shard_map` takes
    # `check_rep`, not the `check_vma` passed below. A fallback that cannot run and
    # would break if it did is worse than none: `tests/distributed/` needs two cards
    # (audit F34), so nothing local would have caught it.
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    from yggdrax.distributed import device_count, make_mesh
    from yggdrax.tree import Tree

    from jaccpot.distributed.fmm import partition_for_devices
    from jaccpot.mutual.device_topology import (
        build_mutual_state_device,
        node_centers_and_radii,
    )
    from jaccpot.mutual.farfield import mutual_upward_sweep
    from jaccpot.mutual.force import (
        mutual_accelerations,
        mutual_weighted_accelerations,
    )

    cfg = config if config is not None else DistributedMutualConfig()
    if mesh is None:
        ndev = device_count() if ndev is None else int(ndev)
        mesh = make_mesh(ndev)
    else:
        ndev = int(np.prod(list(mesh.shape.values())))
    if ndev < 2:
        raise ValueError(
            f"the distributed mutual force needs >= 2 devices, got {ndev}: with one "
            "domain there are no cross-domain pairs, so jaccpot.mutual's single-device "
            "lane is the whole force"
        )

    backend = str(cfg.backend).lower()
    if backend not in ("jax", "pallas"):
        raise ValueError(
            f"backend must be 'jax' or 'pallas'; got {cfg.backend!r}. The same two "
            "the single-device lane takes -- see BlockStepFMM"
        )
    use_pallas = backend == "pallas"

    leaf = int(cfg.leaf_size)
    part = partition_for_devices(
        positions, masses, ndev, leaf_size=leaf, partitioner=cfg.partitioner
    )
    part["_positions"] = np.asarray(positions)
    part["_masses"] = np.asarray(masses)
    cap = int(part["cap"])
    n_leaves = max(1, -(-cap // leaf))
    bounds = part["bounds"]
    caps = cfg.caps if cfg.caps is not None else _default_caps(cap, leaf)

    # At theta = 0 every local leaf pairs with every remote leaf, so the cross pair
    # count is the full product; ownership halves what this device emits, and the
    # margin is left in rather than tuned out because overflow costs a whole rerun.
    cross_pairs_bound = n_leaves * n_leaves * (ndev - 1)
    near_cap = cfg.near_cap or cross_pairs_bound
    max_pair_queue = cfg.max_pair_queue or max(1 << 12, cross_pairs_bound)
    # Every other device can address at most this domain's whole particle set.
    recv_capacity = cfg.recv_capacity or (ndev - 1) * cap
    do_far = float(cfg.cross_theta) > 0.0
    # A far pair is (local node, remote coarse LEAF), so the bound is the local
    # node count times the remote leaf count -- and unlike the near list it does
    # not shrink with theta, it is what theta moves work INTO.
    far_cap = cfg.far_cap or max(1024, (2 * n_leaves - 1) * n_leaves * (ndev - 1))

    def fn(pos: Array, mass: Array, gid: Array, *weights: Array) -> tuple[Array, ...]:
        # Variadic rather than two defaulted parameters, so the unweighted call passes
        # THREE operands and `in_specs` has three entries. A `None` operand would make
        # the argument an empty pytree that its PartitionSpec no longer describes.
        rng_, lw = weights if weights else (None, None)

        # One local tree per domain, over the global bounds so every domain's Morton
        # order is the same order -- the coarse tree merges these frontiers.
        tree = Tree.from_particles(
            pos,
            mass,
            tree_type="radix",
            bounds=bounds,
            return_reordered=True,
            leaf_size=leaf,
        )
        ps = tree.positions_sorted
        ms = tree.masses_sorted
        perm = jnp.asarray(tree.particle_indices, dtype=INDEX_DTYPE)
        parent = jnp.asarray(tree.parent, dtype=INDEX_DTYPE)
        node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
        left = jnp.asarray(tree.left_child, dtype=INDEX_DTYPE)
        right = jnp.asarray(tree.right_child, dtype=INDEX_DTYPE)
        root = jnp.argmin(parent).astype(INDEX_DTYPE)
        n_local = ps.shape[0]
        total_nodes = int(parent.shape[0])
        num_internal = int(left.shape[0])
        n_leaf_nodes = total_nodes - num_internal
        leaf_nodes = jnp.arange(num_internal, total_nodes, dtype=INDEX_DTYPE)

        rung_sorted = None if rng_ is None else jnp.asarray(rng_)[perm]

        # `inverse_permutation = arange`: everything on device stays in TREE order, so
        # "the caller's order" and the tree's order are the same here. The two real
        # permutations (SFC across devices, tree sort within one) are undone on the
        # host, once, at the end.
        state = build_mutual_state_device(
            ps,
            ms,
            parent=parent,
            left_child=left,
            right_child=right,
            node_ranges=node_ranges,
            inverse_permutation=jnp.arange(n_local, dtype=INDEX_DTYPE),
            root=root,
            theta=float(cfg.theta),
            order=int(cfg.order),
            leaf_size=leaf,
            caps=caps,
            softening=float(cfg.softening),
            G=float(cfg.g),
            use_pallas=use_pallas,
            pallas_interpret=bool(cfg.pallas_interpret),
            max_pair_queue=int(caps.queue),
        )
        # The rung takes the same tree sort as the particles (`ps = pos[perm]`), and
        # this state's own permutations are the identity -- `inverse_permutation` is
        # `arange` above, everything on device stays in tree order -- so the weighted
        # call takes it in exactly this order.
        #
        # The intra-domain half applies the same two predicates the single-device lane
        # does, exact per-particle near and cell-level far, so both halves of the
        # system split their pairs the same way and the levels of the WHOLE system
        # still partition it.
        local_acc = (
            mutual_accelerations(state, ps, ms)
            if rung_sorted is None
            else mutual_weighted_accelerations(
                state, ps, ms, rung=rung_sorted, level_weights=lw
            )
        )

        # One upward sweep serves both jobs the cross far field needs: the
        # multipoles this domain publishes on its frontier, and the source
        # expansions for the `-f` direction. Its centres are what both sides then
        # expand about.
        node_mp = None
        exp_centers = None
        if do_far:
            _, exp_centers, node_mp = mutual_upward_sweep(ps, ms, state.tree)

        # The cross walk's MAC is defined on centres of MASS and exact
        # COM-to-particle radii, which is a different extent set from the geometry the
        # target-centric walk uses -- so they are computed here rather than taken from
        # the tree's bounding boxes.
        lp, lv = leaf_blocks(node_ranges, leaf_nodes, leaf_size=leaf)
        leaf_of_particle = (
            jnp.zeros((n_local,), dtype=INDEX_DTYPE)
            .at[lp]
            .max(jnp.where(lv, leaf_nodes[:, None], jnp.zeros((), INDEX_DTYPE)))
        )
        centers, radii = node_centers_and_radii(
            ps,
            ms,
            node_ranges,
            parent,
            leaf_of_particle,
            depth_cap=int(caps.depth) + 1,
        )
        pad = jnp.full((n_leaf_nodes,), -1, dtype=INDEX_DTYPE)
        res = distributed_mutual_accelerations(
            ps,
            ms,
            jnp.concatenate([left, pad]),
            jnp.concatenate([right, pad]),
            node_ranges,
            centers,
            radii,
            root,
            local_acc,
            softening=float(cfg.softening),
            g=float(cfg.g),
            ndev=ndev,
            leaf_width=leaf,
            near_cap=int(near_cap),
            max_pair_queue=int(max_pair_queue),
            recv_capacity=int(recv_capacity),
            max_req_leaves=cfg.max_req_leaves,
            max_recv_leaves=cfg.max_recv_leaves,
            coarse_depth_cap=int(cfg.coarse_depth_cap),
            cross_theta=float(cfg.cross_theta),
            far_cap=int(far_cap),
            far_recv_capacity=cfg.far_recv_capacity,
            node_multipoles=node_mp,
            expansion_centers=exp_centers,
            tree_arrays=state.tree if do_far else None,
            rung=rung_sorted,
            level_weights=lw,
        )
        # Undo the local tree sort, so a row of the output lines up with the row of
        # `pos`/`gid` this device was handed.
        acc = jnp.zeros_like(res.acceleration).at[perm].set(res.acceleration)
        return (
            acc,
            gid,
            res.cross_pairs,
            res.cross_far_pairs,
            res.overflow,
            state.topology_overflow[None],
            state.overflow_causes[None],
        )

    def build(weighted: bool) -> Any:
        """Map and compile `fn` for one weighting mode."""
        # `level_weights` is REPLICATED (`P()`), not sharded: it is one row of a
        # boundary's kick weights and every device must apply the same one, or the two
        # sides of a cross pair would weight it differently and the return path would
        # stop cancelling. `rung` is sharded like the particles it belongs to.
        in_specs = [P(AXIS_NAME)] * 3
        if weighted:
            in_specs += [P(AXIS_NAME), P()]
        mapped = shard_map(
            fn,
            mesh=mesh,
            in_specs=tuple(in_specs),
            out_specs=(P(AXIS_NAME),) * 7,
            check_vma=False,
        )
        return jax.jit(mapped) if jit else mapped

    return DistributedMutualEvaluator(
        config=cfg, mesh=mesh, ndev=ndev, part=part, build=build
    )


def distributed_mutual_fmm(
    positions: Any,
    masses: Any,
    *,
    rung: Any = None,
    level_weights: Any = None,
    config: Optional[DistributedMutualConfig] = None,
    mesh: Any = None,
    ndev: Optional[int] = None,
    jit: bool = True,
) -> DistributedMutualForce:
    """Momentum-conserving accelerations for a system split across devices.

    The one-shot entry point for the lane :func:`distributed_mutual_accelerations`
    implements one device's share of. It owns the host side: split the particles into
    SFC domains, build each domain's own tree, evaluate that domain's own pairs with
    the single-device mutual lane, add the cross-domain half, and put the result back
    in the caller's order.

    Momentum is conserved to round-off over the WHOLE system, not per device, and that
    is the property worth stating: the intra-domain half cancels structurally because
    one kernel writes both halves of every pair, and the cross-domain half cancels
    because the ``-f`` is returned to the domain owning the other endpoint. Neither
    mechanism is numerical.

    This is :func:`make_distributed_mutual_evaluator` plus one call, and it pays a
    fresh partition and a fresh COMPILE on every invocation -- ``shard_map`` wraps a
    new closure each time, so ``jax.jit`` sees a new cache key. **Build an evaluator
    instead for anything that evaluates the force more than once**, which includes
    every timing loop: timed this way you are measuring compilation.

    Parameters
    ----------
    positions : Any
        ``(n, 3)`` positions in any order.
    masses : Any
        ``(n,)`` masses, same order.
    rung : Any
        ``(n,)`` per-particle block-step rung, same order. Required whenever
        ``level_weights`` is given, ignored without it.
    level_weights : Any
        ``(k_max + 1,)`` weight per interaction level; every pair -- intra-domain and
        cross-domain, near and far -- is scaled by
        ``level_weights[max(rung_i, rung_j)]``. ``None`` gives the full force.

        A one-hot row isolates one level, so ``sum_k`` over one-hot rows reproduces
        the unweighted total; a boundary's kick weights evaluate a whole block-step
        sub-step boundary in ONE traversal, which is the point. Momentum is exact for
        any weighting: see :func:`distributed_mutual_accelerations`.
    config : Optional[DistributedMutualConfig]
        Physics and capacities. ``None`` uses the defaults, whose capacities are
        heuristics -- see :class:`DistributedMutualConfig`.
    mesh : Any
        Device mesh. ``None`` builds one over ``ndev`` devices.
    ndev : Optional[int]
        Device count when ``mesh`` is ``None``. ``None`` uses every visible device.
    jit : bool
        Compile the mapped program. Default True.

    Returns
    -------
    DistributedMutualForce
        Accelerations in input order plus the per-device diagnostics. **Check
        ``.overflow``**: a starved capacity is reported, never raised, because the
        force it returns is wrong in a way no norm reveals.

    Raises
    ------
    ValueError
        If fewer than two devices are available, ``n < ndev``, ``level_weights`` is
        given without ``rung``, or the two disagree with ``config.k_max``.
    """
    evaluator = make_distributed_mutual_evaluator(
        positions, masses, config=config, mesh=mesh, ndev=ndev, jit=jit
    )
    return evaluator(positions, masses, rung=rung, level_weights=level_weights)
