"""Device-side construction of the mutual FMM topology.

:mod:`jaccpot.mutual.topology` builds the same thing on the host: six
device-to-host transfers, a scalar Python loop over every node for the
centre/radius pass, a host BFS for the depths, and a NumPy wavefront dual-tree
walk. That is correct and untraceable, so a block-step run pays a host round-trip
per base step -- measured 22 s at N = 20 000 against 0.5 s for the whole rest of
the base step once the force is jitted, i.e. it *is* the remaining wall.

This module builds it on device instead, with static shapes throughout, so the
whole thing can live inside a ``jax.jit``.

Two pieces do the work:

* :func:`node_centers_and_radii` -- the MAC geometry. Centres are a pure gather
  from prefix sums. Radii are the exact ``max_i |x_i - c_n|`` the host computes,
  *not* the bounding-sphere merge ``r_parent = max_c(|c_c - c_p| + r_c)`` that a
  bottom-up cascade would give: that is an upper bound, and a looser radius
  changes MAC outcomes, which changes the accepted pair set and therefore the
  force. Exactness is obtained by walking *up* from every particle, one
  scatter-max per tree level -- ``O(N * depth)``, vectorised, fixed shapes.
* :func:`yggdrax.interactions.dual_tree_walk_mutual` -- the traversal, which is
  verified to reproduce the host walk pair-for-pair.

The gradient seam is the caller's responsibility and is documented on
:func:`node_centers_and_radii`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime
    # `force` imports this module, so these are annotation-only here.
    from jaccpot.mutual.force import MutualCapacities, MutualFMMState

__all__ = [
    "dense_level_schedule",
    "leaf_blocks",
    "node_centers_and_radii",
    "node_depths",
    "build_mutual_state_device",
]


def node_centers_and_radii(
    positions_sorted: Array,
    masses_sorted: Array,
    node_ranges: Array,
    parent: Array,
    leaf_of_particle: Array,
    *,
    depth_cap: int,
) -> Tuple[Array, Array]:
    """Return ``(centers, radii)`` for every node, on device.

    ``centers[n]`` is the centre of mass of the particles in
    ``node_ranges[n] = [start, end]`` (inclusive), falling back to the plain mean
    for a massless node exactly as the host does. ``radii[n]`` is
    ``max_i |x_i - centers[n]|`` over the same particles.

    Parameters
    ----------
    positions_sorted : Array
        ``(N, 3)`` particle positions in **tree order**.
    masses_sorted : Array
        ``(N,)`` particle masses in the same order.
    node_ranges : Array
        ``(num_nodes, 2)`` inclusive particle ranges.
    parent : Array
        ``(num_nodes,)`` parent index, negative at the root.
    leaf_of_particle : Array
        ``(N,)`` the leaf *node* index each particle sits in. Invert
        ``leaf_particles`` to get it rather than assuming ``i // leaf_size``:
        that identity holds for the bucket-slicing builders and is not part of
        the tree contract.
    depth_cap : int
        Static bound on tree depth. The upward walk saturates at the root, and a
        max-scatter is idempotent, so **over**-provisioning is free and only
        under-provisioning is wrong -- which is why this is a cap and not a
        measured depth.

    Returns
    -------
    Tuple[Array, Array]
        ``(centers, radii)``: the centre of mass of each node's particles, and
        the exact ``max_i |x_i - c_n|`` over the same particles.

    Notes
    -----
    **Gradients.** Centres are differentiable in ``positions``/``masses`` and
    *must stay so*: the mutual far field re-derives its expansion centres from
    the live positions on every call, and freezing them drops a real gradient
    term (see ``jaccpot/upward/tree_expansions.py``). The radii, by contrast,
    feed only the MAC -- a discrete accept/reject decision -- so a caller
    building a topology inside a differentiated window should
    ``stop_gradient`` the copy of ``positions`` it passes *here* while leaving
    the copy that feeds the upward sweep live. That is the same split
    ``jaccpot/distributed/fmm.py`` makes.
    """
    x = jnp.asarray(positions_sorted)
    m = jnp.asarray(masses_sorted)
    dtype = x.dtype
    num_nodes = int(node_ranges.shape[0])

    start = node_ranges[:, 0]
    end = node_ranges[:, 1]
    empty = end < start
    # Clamp so an empty range gathers a valid slot; `empty` masks it out after.
    lo = jnp.clip(start, 0, x.shape[0] - 1)
    hi = jnp.clip(end, 0, x.shape[0] - 1)

    zero3 = jnp.zeros((1, 3), dtype=dtype)
    zero1 = jnp.zeros((1,), dtype=dtype)
    cum_wx = jnp.concatenate([zero3, jnp.cumsum(m[:, None] * x, axis=0)], axis=0)
    cum_x = jnp.concatenate([zero3, jnp.cumsum(x, axis=0)], axis=0)
    cum_m = jnp.concatenate([zero1, jnp.cumsum(m)], axis=0)
    cum_n = jnp.concatenate([zero1, jnp.cumsum(jnp.ones_like(m))], axis=0)

    total_mass = cum_m[hi + 1] - cum_m[lo]
    total_count = cum_n[hi + 1] - cum_n[lo]
    massive = total_mass > 0

    # Double `where` around each reciprocal: the masked branch must not be
    # allowed to form an inf/NaN even though it is discarded, or the cotangent
    # comes back NaN.
    safe_mass = jnp.where(massive, total_mass, jnp.ones_like(total_mass))
    safe_count = jnp.where(total_count > 0, total_count, jnp.ones_like(total_count))
    com = (cum_wx[hi + 1] - cum_wx[lo]) / safe_mass[:, None]
    mean = (cum_x[hi + 1] - cum_x[lo]) / safe_count[:, None]
    centers = jnp.where(massive[:, None], com, mean)
    centers = jnp.where(empty[:, None], jnp.zeros_like(centers), centers)

    # Exact max-COM-distance, by walking each particle up to the root and
    # scatter-maxing into every ancestor it passes.
    #
    # Both the carry and the parent table are pinned to one index dtype. A
    # `lax.scan` carry must keep its dtype across the loop, and `parent` gathered
    # at an int32 carry silently promotes it to int64 the moment the caller's
    # arrays disagree -- which is a hard trace error, not a wrong answer, but only
    # for callers whose dtypes happen to differ.
    index_dtype = jnp.int32
    parent = jnp.asarray(parent).astype(index_dtype)
    leaf_of_particle = jnp.asarray(leaf_of_particle).astype(index_dtype)

    def step(node: Array, _: Any) -> Tuple[Array, Array]:
        delta = x - centers[node]
        dist = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
        contribution = jnp.zeros((num_nodes,), dtype=dtype).at[node].max(dist)
        # Saturate at the root so extra iterations are idempotent re-maxes.
        up = parent[node]
        return jnp.where(up < 0, node, up).astype(index_dtype), contribution

    _, per_level = lax.scan(step, leaf_of_particle, xs=None, length=int(depth_cap))
    radii = jnp.max(per_level, axis=0)
    return centers, radii


def node_depths(parent: Array, root: Array, *, depth_cap: int) -> Array:
    """Depth of every node, root at 0, unreachable nodes at -1.

    A fixed-point iteration rather than the host's BFS: ``depth`` is relaxed
    ``depth_cap`` times from the parent pointers, which converges once the
    iteration count reaches the true depth. Over-provisioning ``depth_cap`` is
    free (the relaxation is stable at its fixed point); under-provisioning
    silently leaves deep nodes short, so the caller must bound it correctly --
    the same contract as everywhere else in this module.

    Parameters
    ----------
    parent : Array
        ``(num_nodes,)`` parent index, negative at the root.
    root : Array
        Index of the root node.
    depth_cap : int
        Number of relaxation sweeps; must be at least the true depth.

    Returns
    -------
    Array
        ``(num_nodes,)`` int32 depth, root at 0 and unreachable nodes at -1.
    """
    parent = jnp.asarray(parent).astype(jnp.int32)
    num_nodes = int(parent.shape[0])
    # Unreachable nodes never acquire a depth: they are not the root and their
    # parent chain never reaches it.
    depth = jnp.where(jnp.arange(num_nodes) == root, 0, -1).astype(jnp.int32)

    def relax(d: Array, _: Any) -> Tuple[Array, None]:
        safe_parent = jnp.where(parent >= 0, parent, 0)
        from_parent = jnp.where(parent >= 0, d[safe_parent] + 1, -1)
        # Only adopt a depth once the parent has one.
        updated = jnp.where(
            (d < 0) & (d[safe_parent] >= 0) & (parent >= 0), from_parent, d
        )
        return updated, None

    depth, _ = lax.scan(relax, depth, xs=None, length=int(depth_cap))
    return depth


def dense_level_schedule(
    depth: Array,
    parent: Array,
    *,
    depth_cap: int,
    width_cap: int,
) -> Tuple[Array, Array, Array, Array]:
    """Pack nodes by depth into ``(depth_cap, width_cap)``, shallowest first.

    Row ``d`` holds the nodes at depth ``d + 1`` and their parents, matching the
    host ``_group_levels`` convention, with ``valid`` false in the padding. Every
    consumer masks a padded slot to its own identity, so a row beyond the real
    tree depth is an all-invalid no-op -- which is what lets one compiled program
    survive a rebuild that deepens the tree.

    Parameters
    ----------
    depth : Array
        ``(num_nodes,)`` node depth, as returned by :func:`node_depths`.
    parent : Array
        ``(num_nodes,)`` parent index, negative at the root.
    depth_cap : int
        Rows to emit, i.e. the maximum tree depth covered.
    width_cap : int
        Slots per row, i.e. the maximum nodes at any one level.

    Returns
    -------
    Tuple[Array, Array, Array, Array]
        ``(level_nodes, level_parents, level_valid, overflow)``. The first three are
        ``(depth_cap, width_cap)``; ``overflow`` is a scalar bool set when a level
        held more nodes than ``width_cap``.
    """
    depth = jnp.asarray(depth)
    index_dtype = jnp.int32
    parent = jnp.asarray(parent).astype(index_dtype)
    num_nodes = int(depth.shape[0])
    node_ids = jnp.arange(num_nodes, dtype=index_dtype)

    def pack(_: Any, d: Array) -> Tuple[None, Tuple[Array, Array, Array, Array]]:
        mask = depth == (d + 1)
        live = mask.astype(jnp.int32)
        prefix = jnp.cumsum(live) - live
        ok = mask & (prefix < width_cap)
        slot = jnp.where(ok, prefix, width_cap)
        zero = jnp.zeros((), dtype=index_dtype)
        nodes = (
            jnp.zeros((width_cap,), dtype=index_dtype)
            .at[slot]
            .set(jnp.where(ok, node_ids, zero), mode="drop")
        )
        parents = (
            jnp.zeros((width_cap,), dtype=index_dtype)
            .at[slot]
            .set(jnp.where(ok, jnp.where(parent >= 0, parent, zero), zero), mode="drop")
        )
        valid = jnp.zeros((width_cap,), dtype=bool).at[slot].set(ok, mode="drop")
        overflow = jnp.any(mask & (prefix >= width_cap))
        return None, (nodes, parents, valid, overflow)

    _, (nodes, parents, valid, overflow) = lax.scan(
        pack, None, jnp.arange(depth_cap, dtype=jnp.int32)
    )
    return nodes, parents, valid, jnp.any(overflow)


def leaf_blocks(
    node_ranges: Array, leaf_nodes: Array, *, leaf_size: int
) -> Tuple[Array, Array]:
    """Per-leaf particle blocks, ``(num_leaves, leaf_size)`` plus a validity mask.

    The host builds these with ``arange(start, end + 1)`` per leaf; here it is one
    broadcast, because a leaf's particles are always a contiguous run in tree
    order. ``leaf_size`` is the block width, not an assumption about occupancy:
    slots past a leaf's ``end`` are simply invalid.

    Parameters
    ----------
    node_ranges : Array
        ``(num_nodes, 2)`` inclusive particle ranges per node.
    leaf_nodes : Array
        ``(num_leaves,)`` node index of each leaf.
    leaf_size : int
        Block width.

    Returns
    -------
    Tuple[Array, Array]
        ``(particles, valid)``, both ``(num_leaves, leaf_size)``. Padded slots repeat
        the leaf's first particle so a gather stays in bounds; ``valid`` is what
        removes their contribution.
    """
    node_ranges = jnp.asarray(node_ranges)
    starts = node_ranges[leaf_nodes, 0]
    ends = node_ranges[leaf_nodes, 1]
    offsets = jnp.arange(leaf_size, dtype=starts.dtype)
    particles = starts[:, None] + offsets[None, :]
    valid = particles <= ends[:, None]
    # Clamp the padded indices so a gather by them stays in bounds; `valid` is
    # what removes their contribution.
    return jnp.where(valid, particles, starts[:, None]), valid


def build_mutual_state_device(
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    parent: Array,
    left_child: Array,
    right_child: Array,
    node_ranges: Array,
    inverse_permutation: Array,
    root: Array,
    theta: float,
    order: int,
    leaf_size: int,
    caps: "MutualCapacities",
    softening: float,
    G: float = 1.0,
    use_pallas: bool = False,
    near_chunk_size: Optional[int] = None,
    pallas_interpret: bool = False,
    max_pair_queue: int = 1 << 16,
    freeze_topology_gradient: bool = True,
) -> "MutualFMMState":
    """Build a :class:`~jaccpot.mutual.force.MutualFMMState` entirely on device.

    Drop-in replacement for
    :func:`~jaccpot.mutual.force.build_mutual_state` composed with
    :func:`~jaccpot.mutual.topology.build_mutual_topology_from_tree`, with no host
    round-trip: fully traceable, static shapes, and therefore usable inside a
    ``jax.jit`` and inside a ``lax.scan`` over base steps.

    Verified to reproduce the host construction **pair-for-pair** -- the accepted
    far and near sets are identical, and the centres and radii agree to ~3e-13
    (prefix-sum against direct summation).

    ``caps`` is a :class:`~jaccpot.mutual.force.MutualCapacities`; every output
    shape comes from it, so one compiled program serves every rebuild.

    Overflow is reported in the returned state's counters rather than raised: this
    runs under trace, where an exception is not available. Compare
    ``num_far_pairs``/``num_near_pairs`` against the capacities after the fact --
    a truncated mutual list still conserves momentum exactly (dropping a
    canonical pair drops both halves), so nothing else will tell you.

    Notes
    -----
    **Gradients.** With ``freeze_topology_gradient`` (the default) the positions and masses that
    feed the **MAC geometry** are ``stop_gradient``-ed, while the copies returned
    for the upward sweep stay live. That split is the whole gradient contract:
    the accept/reject decisions are discrete and must be severed, but the
    expansion centres are recomputed from live positions on every evaluation and
    carry a real gradient term -- freezing those would silently drop it. It is
    the same seam ``jaccpot/distributed/fmm.py`` uses.

    Parameters
    ----------
    positions_sorted : Array
        ``(N, 3)`` particle positions in tree order.
    masses_sorted : Array
        ``(N,)`` particle masses in tree order.
    parent : Array
        ``(num_nodes,)`` parent index, negative at the root.
    left_child : Array
        ``(num_internal,)`` left-child index.
    right_child : Array
        ``(num_internal,)`` right-child index.
    node_ranges : Array
        ``(num_nodes, 2)`` inclusive particle ranges.
    inverse_permutation : Array
        ``(N,)`` map from tree order back to the caller's original order.
    root : Array
        Index of the root node.
    theta : float
        Mutual MAC parameter.
    order : int
        Multipole expansion order.
    leaf_size : int
        Particles per leaf block.
    caps : MutualCapacities
        Fixed capacities; every output shape comes from these.
    softening : float
        Plummer softening length.
    G : float
        Gravitational constant. Default ``1.0``.
    use_pallas : bool
        Dispatch the fused Pallas near-field kernel.
    near_chunk_size : Optional[int]
        Pair-chunk size for the near kernel; ``None`` derives it.
    pallas_interpret : bool
        Run the Pallas kernels in interpret mode.
    max_pair_queue : int
        Wavefront capacity for the device traversal.
    freeze_topology_gradient : bool
        ``stop_gradient`` the copy of positions feeding the MAC geometry.

    Returns
    -------
    MutualFMMState
        Device-resident state, padded to ``caps``, with its occupancy counters and
        overflow flags set.
    """
    from yggdrax.interactions import dual_tree_walk_mutual

    from jaccpot.mutual.farfield import MutualTreeArrays
    from jaccpot.mutual.force import MutualFMMState

    index_dtype = jnp.int32
    x = jnp.asarray(positions_sorted)
    m = jnp.asarray(masses_sorted)
    x_topo = lax.stop_gradient(x) if freeze_topology_gradient else x
    m_topo = lax.stop_gradient(m) if freeze_topology_gradient else m

    parent = jnp.asarray(parent).astype(index_dtype)
    num_nodes = int(parent.shape[0])
    num_internal = int(jnp.asarray(left_child).shape[0])
    num_leaves = num_nodes - num_internal
    leaf_nodes = jnp.arange(num_internal, num_nodes, dtype=index_dtype)

    lc_full = jnp.concatenate(
        [
            jnp.asarray(left_child).astype(index_dtype),
            jnp.full((num_leaves,), -1, dtype=index_dtype),
        ]
    )
    rc_full = jnp.concatenate(
        [
            jnp.asarray(right_child).astype(index_dtype),
            jnp.full((num_leaves,), -1, dtype=index_dtype),
        ]
    )

    leaf_particles, leaf_valid = leaf_blocks(
        node_ranges, leaf_nodes, leaf_size=int(leaf_size)
    )
    # Invert the leaf blocks rather than assuming `i // leaf_size`: that identity
    # holds for the bucket-slicing builders and is not part of the tree contract.
    leaf_of_particle = (
        jnp.zeros((x.shape[0],), dtype=index_dtype)
        .at[leaf_particles]
        .max(jnp.where(leaf_valid, leaf_nodes[:, None], index_dtype(0)))
    )

    centers, radii = node_centers_and_radii(
        x_topo,
        m_topo,
        node_ranges,
        parent,
        leaf_of_particle,
        depth_cap=int(caps.depth) + 1,
    )
    walk = dual_tree_walk_mutual(
        lc_full,
        rc_full,
        centers,
        radii,
        float(theta),
        root,
        max_pair_queue=int(max_pair_queue),
        far_cap=int(caps.far),
        near_cap=int(caps.near),
    )

    depth = node_depths(parent, root, depth_cap=int(caps.depth) + 1)
    level_nodes, level_parents, level_valid, level_overflow = dense_level_schedule(
        depth, parent, depth_cap=int(caps.depth), width_cap=int(caps.width)
    )
    # A truncated level schedule drops nodes from the M2M/L2L cascade, which is a
    # ~1e-2 force error and nothing else -- no NaN, no shape mismatch, and
    # momentum still exact. It cost a wrong-looking accuracy number to find, so
    # the flag is carried out on the state rather than dropped here.
    deepest = jnp.max(depth)
    depth_overflow = deepest > int(caps.depth)
    overflow = (
        level_overflow
        | walk.far_overflow
        | walk.near_overflow
        | walk.queue_overflow
        | depth_overflow
    )
    # Which cap blew is not derivable from the counts after the fact -- a
    # truncated walk stops early, so its own counters undercount -- so the four
    # causes are carried out separately. Packed into one int so the state gains a
    # single leaf rather than four.
    overflow_causes = (
        walk.far_overflow.astype(jnp.int32)
        | (walk.near_overflow.astype(jnp.int32) << 1)
        | (walk.queue_overflow.astype(jnp.int32) << 2)
        | (level_overflow.astype(jnp.int32) << 3)
        | (depth_overflow.astype(jnp.int32) << 4)
    )

    # Near pairs are addressed by LEAF index in the state, not node index.
    node_to_leaf = (
        jnp.zeros((num_nodes,), dtype=index_dtype)
        .at[leaf_nodes]
        .set(jnp.arange(num_leaves, dtype=index_dtype))
    )
    near_live = jnp.arange(int(caps.near)) < walk.near_count
    near_a = node_to_leaf[jnp.where(near_live, walk.near_a, index_dtype(0))]
    near_b = node_to_leaf[jnp.where(near_live, walk.near_b, index_dtype(0))]

    far_live = jnp.arange(int(caps.far)) < walk.far_count
    far_a = jnp.where(far_live, walk.far_a, index_dtype(0))
    far_b = jnp.where(far_live, walk.far_b, index_dtype(0))
    far_source = jnp.concatenate([far_b, far_a])
    far_target = jnp.concatenate([far_a, far_b])
    far_valid = jnp.concatenate([far_live, far_live])

    tree = MutualTreeArrays(
        num_nodes=num_nodes,
        order=int(order),
        leaf_nodes=leaf_nodes,
        leaf_particles=leaf_particles.astype(index_dtype),
        leaf_particle_valid=leaf_valid,
        level_nodes=level_nodes,
        level_parents=level_parents,
        level_valid=level_valid,
        far_source=far_source,
        far_target=far_target,
        far_valid=far_valid,
    )
    inverse_permutation = jnp.asarray(inverse_permutation).astype(index_dtype)
    return MutualFMMState(
        tree=tree,
        leaf_particles=tree.leaf_particles,
        leaf_particle_valid=tree.leaf_particle_valid,
        near_a=near_a,
        near_b=near_b,
        near_valid=near_live,
        self_leaves=jnp.arange(num_leaves, dtype=index_dtype),
        far_a=far_a,
        far_b=far_b,
        forward_permutation=jnp.argsort(inverse_permutation).astype(index_dtype),
        inverse_permutation=inverse_permutation,
        softening=float(softening),
        G=float(G),
        order=int(order),
        use_pallas=bool(use_pallas),
        near_chunk_size=near_chunk_size,
        pallas_interpret=bool(pallas_interpret),
        num_particles_=int(x.shape[0]),
        num_near_pairs=walk.near_count,
        num_far_pairs=walk.far_count,
        topology_overflow=overflow,
        overflow_causes=overflow_causes,
    )
