"""Per-step pieces of Dehnen's eq (16b) force scale on the fused lane, by tree level.

The criterion's threshold for a node is ``eps * min_{b in node} f_b`` with the
cancellation-free ``f_b = sum_a G m_a / (|x_a - x_b|^2 + eps^2)``. On the fused
lane ``f_b`` is a by-product of the previous step's force: its near half from the
near-field kernel's force-scale lane, its far half from the far pairs as monopoles
pushed down the tree. Two reductions over the tree turn those into what the walk
needs, and both must stay cheap at 1e8 particles:

* :func:`ancestor_sum_by_level` -- each node's own far contribution plus all its
  ancestors', so a leaf holds the complete far term of its particles. The serial
  ``accumulate_own_down_parent_chain`` (one scatter per internal node, ~4e5 at
  25M) is replaced by a top-down pass over the tree's own level tables.
* :func:`subtree_min_by_level` -- the minimum of a per-leaf value over each node's
  subtree, deepest level first.

Both walk ``nodes_by_level`` / ``level_offsets`` exactly as the M2M pass does
(``aggregate_m2m_real_by_level``): static ``level_batch_width`` slots with the
same clamp guard, ~40 small vectorised steps and no extra memory beyond the
``[total_nodes]`` value.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array
from yggdrax.dtypes import INDEX_DTYPE, as_index

__all__ = [
    "ancestor_sum_by_level",
    "far_force_scale_own",
    "far_force_scale_sorted",
    "node_force_scale_min",
    "node_force_scale_min_sorted",
    "subtree_min_by_level",
]


def _level_windows(
    nodes_by_level: Array, level_offsets: Array, batch_width: int
) -> tuple[Array, Array, Array]:
    level_offsets = jnp.asarray(level_offsets, dtype=INDEX_DTYPE)
    # padded so the widest window is always in range: `dynamic_slice_in_dim`
    # CLAMPS an out-of-range start (see aggregate_m2m_real_by_level)
    nodes = jnp.concatenate(
        [
            jnp.asarray(nodes_by_level, dtype=INDEX_DTYPE),
            jnp.full((batch_width,), -1, dtype=INDEX_DTYPE),
        ]
    )
    slot = jnp.arange(batch_width, dtype=INDEX_DTYPE)
    return level_offsets, nodes, slot


def _level_nodes(
    level_idx: Array,
    level_offsets: Array,
    nodes: Array,
    slot: Array,
    batch_width: int,
    num_internal: int,
) -> tuple[Array, Array]:
    start = level_offsets[level_idx]
    count = level_offsets[level_idx + 1] - start
    batch = lax.dynamic_slice_in_dim(nodes, start, batch_width, axis=0)
    ok = (slot < count) & (batch >= as_index(0)) & (batch < as_index(num_internal))
    return batch, ok


def subtree_min_by_level(
    values: Array,
    left_child: Array,
    right_child: Array,
    parent: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    *,
    num_internal: int,
    num_levels: int,
    level_batch_width: int,
) -> Array:
    """Each internal node's value becomes the minimum over its children, deepest first.

    Parameters
    ----------
    values : Array
        ``[total_nodes]``; the leaves' entries are the inputs (``+inf`` for an
        empty leaf), the internal entries are overwritten.
    left_child : Array
        Left child per internal node, ``[num_internal]`` (``-1`` none).
    right_child : Array
        Right child per internal node.
    parent : Array
        ``[total_nodes]`` parent per node; a child counts only along its true
        parent edge (see :func:`ancestor_sum_by_level`).
    nodes_by_level : Array
        Internal nodes grouped by level (``yggdrax.tree.get_nodes_by_level``).
    level_offsets : Array
        Level starts into ``nodes_by_level`` (``get_level_offsets``).
    num_internal : int
        Internal node count. Static.
    num_levels : int
        Levels in the tables. Static.
    level_batch_width : int
        Slot width per level (``jaccpot.runtime._level_shapes.level_batch_width``).
        Static.

    Returns
    -------
    Array
        ``[total_nodes]``: the leaves unchanged, every internal node the minimum
        over the leaves below it.
    """
    if int(num_internal) <= 0:
        return values
    bw = int(max(level_batch_width, 1))
    offsets, nodes, slot = _level_windows(nodes_by_level, level_offsets, bw)
    left = jnp.asarray(left_child, dtype=INDEX_DTYPE)
    right = jnp.asarray(right_child, dtype=INDEX_DTYPE)
    par = jnp.asarray(parent, dtype=INDEX_DTYPE)
    inf = jnp.asarray(jnp.inf, values.dtype)
    dead = as_index(values.shape[0])
    state = jnp.concatenate([values, jnp.full((1,), inf, values.dtype)])

    def body(rev: Array, st: Array) -> Array:
        level_idx = as_index((num_levels - 2) - rev)
        batch, ok = _level_nodes(level_idx, offsets, nodes, slot, bw, num_internal)
        safe = jnp.where(ok, batch, as_index(0))
        lc, rc = left[safe], right[safe]
        lc_s, rc_s = jnp.maximum(lc, 0), jnp.maximum(rc, 0)
        lv = jnp.where((lc >= 0) & (par[lc_s] == safe), st[lc_s], inf)
        rv = jnp.where((rc >= 0) & (par[rc_s] == safe), st[rc_s], inf)
        return st.at[jnp.where(ok, batch, dead)].set(jnp.minimum(lv, rv))

    state = lax.fori_loop(0, max(int(num_levels) - 1, 0), body, state)
    return state[: values.shape[0]]


def ancestor_sum_by_level(
    own: Array,
    left_child: Array,
    right_child: Array,
    parent: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    *,
    num_internal: int,
    num_levels: int,
    level_batch_width: int,
) -> Array:
    """Each node's own value plus all of its ancestors', top-down by level.

    The level-order replacement for ``accumulate_own_down_parent_chain``: a node is
    final once its parent's level is done, so each level adds its (final) values
    onto its children. A value moves only along a TRUE parent edge
    (``parent[child] == node``): the dead internal nodes of a capacity-padded cell
    partition list children they do not own (150 of 2047 nodes at N=3000), and
    pushing through those would count a contribution twice.

    Parameters
    ----------
    own : Array
        ``[total_nodes]`` per-node contributions.
    left_child : Array
        Left child per internal node, ``[num_internal]`` (``-1`` none).
    right_child : Array
        Right child per internal node.
    parent : Array
        ``[total_nodes]`` parent per node (``-1`` for the root).
    nodes_by_level : Array
        Internal nodes grouped by level.
    level_offsets : Array
        Level starts into ``nodes_by_level``.
    num_internal : int
        Internal node count. Static.
    num_levels : int
        Levels in the tables. Static.
    level_batch_width : int
        Slot width per level. Static.

    Returns
    -------
    Array
        ``[total_nodes]`` sums over each node's ancestor chain, the node included.
    """
    if int(num_internal) <= 0:
        return own
    bw = int(max(level_batch_width, 1))
    offsets, nodes, slot = _level_windows(nodes_by_level, level_offsets, bw)
    left = jnp.asarray(left_child, dtype=INDEX_DTYPE)
    right = jnp.asarray(right_child, dtype=INDEX_DTYPE)
    par = jnp.asarray(parent, dtype=INDEX_DTYPE)
    dead = as_index(own.shape[0])
    state = jnp.concatenate([own, jnp.zeros((1,), own.dtype)])

    def body(level: Array, st: Array) -> Array:
        level_idx = as_index(level)
        batch, ok = _level_nodes(level_idx, offsets, nodes, slot, bw, num_internal)
        safe = jnp.where(ok, batch, as_index(0))
        val = jnp.where(ok, st[safe], jnp.zeros((), own.dtype))
        lc, rc = left[safe], right[safe]
        l_ok = ok & (lc >= 0) & (par[jnp.maximum(lc, 0)] == safe)
        r_ok = ok & (rc >= 0) & (par[jnp.maximum(rc, 0)] == safe)
        st = st.at[jnp.where(l_ok, lc, dead)].add(val)
        return st.at[jnp.where(r_ok, rc, dead)].add(val)

    state = lax.fori_loop(0, max(int(num_levels) - 1, 0), body, state)
    return state[: own.shape[0]]


def far_force_scale_own(
    *,
    sources: Array,
    targets: Array,
    live: Array,
    node_mass: Array,
    node_centers: Array,
    node_radii: Array,
    gravitational_constant: float,
    softening_sq: Array,
    num_nodes: int,
) -> Array:
    """Each node's own far term of eq (16b), from the directed far pairs.

    The eager estimator's form (``_far_field_force_scale_by_node``): every far pair
    ``(A -> B)`` adds ``G M_A / ((|c_A - c_B| + rho_B)^2 + eps^2)`` to node B -- a
    lower bound on ``G M_A / |x_a - x_b|^2`` for every particle of B, so the scale
    errs low (stricter, never looser). One segment-sum over the list.

    Parameters
    ----------
    sources : Array
        Source node per directed far pair.
    targets : Array
        Target node per directed far pair.
    live : Array
        Live pairs (the list's prefix).
    node_mass : Array
        ``[nodes]`` node masses.
    node_centers : Array
        ``[nodes, 3]`` the walk centres.
    node_radii : Array
        ``[nodes]`` the walk radii.
    gravitational_constant : float
        ``G``.
    softening_sq : Array
        ``eps^2``, the Plummer-equivalent softening squared.
    num_nodes : int
        Node count. Static.

    Returns
    -------
    Array
        ``[num_nodes]`` own far contributions (before the push-down).
    """
    src = jnp.maximum(jnp.asarray(sources, INDEX_DTYPE), 0)
    tgt = jnp.maximum(jnp.asarray(targets, INDEX_DTYPE), 0)
    dtype = node_mass.dtype
    delta = node_centers[src] - node_centers[tgt]
    reach = jnp.sqrt(jnp.sum(delta * delta, axis=1)) + node_radii[tgt]
    contrib = (
        jnp.asarray(gravitational_constant, dtype)
        * node_mass[src]
        / (reach * reach + jnp.asarray(softening_sq, dtype))
    )
    contrib = jnp.where(live & (reach > 0), contrib, jnp.zeros((), dtype))
    return jax.ops.segment_sum(contrib, tgt, num_segments=int(num_nodes))


#: far pairs per chunk of the per-step far force scale (bounds its temporaries)
_FAR_PAIR_CHUNK = 1 << 22


def far_force_scale_sorted(
    *,
    tree: Any,
    leaf_nodes: Array,
    far_pairs: Any,
    node_mass: Array,
    node_centers: Array,
    node_radii: Array,
    gravitational_constant: float,
    softening_sq: Array,
    num_levels: int,
    num_particles: int,
) -> Array:
    """The far half of eq (16b)'s ``f_b`` per SORTED particle, from one walk's lists.

    :func:`far_force_scale_own` on the directed far pairs, pushed down the tree
    with :func:`ancestor_sum_by_level`, and read at each particle's leaf. Leaves
    hold contiguous particle ranges in ``leaf_nodes`` order (the neighbour list's
    ``leaf_indices``), so the particle -> leaf map is the fast lane's
    ``repeat`` over the leaf counts.

    Parameters
    ----------
    tree : Any
        The step's tree (``node_ranges``, ``parent``, children, level tables).
    leaf_nodes : Array
        ``[L]`` leaf node ids in particle order (``neighbor_list.leaf_indices``).
    far_pairs : Any
        The walk's directed far list (``CompactTaggedFarPairs``, or its
        target-sorted subclass whose ``targets`` are CSR row offsets); its live
        prefix is ``far_pair_count``. Expanded inside the compiled program.
    node_mass : Array
        ``[nodes]`` node masses.
    node_centers : Array
        ``[nodes, 3]`` the walk centres.
    node_radii : Array
        ``[nodes]`` the walk radii.
    gravitational_constant : float
        ``G``.
    softening_sq : Array
        ``eps^2`` (Plummer-equivalent).
    num_levels : int
        Level-loop bound, the one the upward pass uses. Static.
    num_particles : int
        ``N``. Static.

    Returns
    -------
    Array
        ``[N]`` the far force scale in tree (sorted) order; zero past the leaves.
    """
    from yggdrax.tree import get_level_offsets, get_nodes_by_level

    from jaccpot.runtime._level_shapes import level_batch_width

    total = int(jnp.asarray(tree.parent).shape[0])
    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    offsets = get_level_offsets(tree)
    # one fused program: run eagerly, op by op, every per-pair gather over the far
    # list was its own temporary (~0.25 GiB at 2e6 on top of the prepare's peak)
    return _far_force_scale_sorted_jit(
        jnp.asarray(tree.parent, dtype=INDEX_DTYPE),
        jnp.asarray(tree.left_child, dtype=INDEX_DTYPE),
        jnp.asarray(tree.right_child, dtype=INDEX_DTYPE),
        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        jnp.asarray(get_nodes_by_level(tree), dtype=INDEX_DTYPE),
        jnp.asarray(offsets, dtype=INDEX_DTYPE),
        jnp.asarray(leaf_nodes, dtype=INDEX_DTYPE),
        jnp.asarray(far_pairs.sources, dtype=INDEX_DTYPE),
        jnp.asarray(far_pairs.targets, dtype=INDEX_DTYPE),
        jnp.asarray(far_pairs.far_pair_count, dtype=INDEX_DTYPE),
        jnp.asarray(node_mass),
        jnp.asarray(node_centers),
        jnp.asarray(node_radii),
        jnp.asarray(softening_sq),
        gravitational_constant=float(gravitational_constant),
        num_internal=num_internal,
        total_nodes=total,
        num_levels=int(num_levels),
        level_batch_width=level_batch_width(
            offsets, total_nodes=total, num_internal=num_internal
        ),
        num_particles=int(num_particles),
        target_sorted=_is_target_sorted(far_pairs),
    )


def _is_target_sorted(far_pairs: Any) -> bool:
    from jaccpot.runtime._interaction_cache import TargetSortedFarPairs

    return isinstance(far_pairs, TargetSortedFarPairs)


@partial(
    jax.jit,
    static_argnames=(
        "target_sorted",
        "gravitational_constant",
        "num_internal",
        "total_nodes",
        "num_levels",
        "level_batch_width",
        "num_particles",
    ),
)
def _far_force_scale_sorted_jit(
    parent: Array,
    left_child: Array,
    right_child: Array,
    node_ranges: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    leaf_nodes: Array,
    sources: Array,
    targets_raw: Array,
    far_pair_count: Array,
    node_mass: Array,
    node_centers: Array,
    node_radii: Array,
    softening_sq: Array,
    *,
    target_sorted: bool,
    gravitational_constant: float,
    num_internal: int,
    total_nodes: int,
    num_levels: int,
    level_batch_width: int,
    num_particles: int,
) -> Array:
    # The pairs in fixed chunks, so no per-pair temporary is ever list-sized: at
    # eps 1e-5 on the 2e6 disc the whole-list form still added 0.30 GiB to the
    # prepare's peak (targets, mask, contributions over 24M pairs), and the list
    # grows with N.
    num_pairs = int(sources.shape[0])
    chunk = max(1, min(num_pairs, _FAR_PAIR_CHUNK))
    num_chunks = -(-num_pairs // chunk) if num_pairs else 0
    dtype = node_mass.dtype
    g = jnp.asarray(gravitational_constant, dtype)
    soft = jnp.asarray(softening_sq, dtype)
    lane = jnp.arange(chunk, dtype=INDEX_DTYPE)
    live_end = targets_raw[-1].astype(INDEX_DTYPE) if target_sorted else jnp.asarray(0)

    def _chunk(i: Array, own: Array) -> Array:
        start = i * chunk
        # a clamped window (the last chunk) re-reads entries of the previous one:
        # they are masked by their absolute position, never counted twice
        first = jnp.minimum(start, max(num_pairs - chunk, 0))
        pos = first + lane
        src = lax.dynamic_slice_in_dim(sources, first, chunk)
        if target_sorted:
            row = jnp.searchsorted(targets_raw, pos, side="right").astype(
                INDEX_DTYPE
            ) - jnp.asarray(1, INDEX_DTYPE)
            tgt = jnp.where(pos < live_end, row, jnp.asarray(-1, INDEX_DTYPE))
        else:
            tgt = lax.dynamic_slice_in_dim(targets_raw, first, chunk)
        live = (
            (pos >= start)
            & (pos < num_pairs)
            & (pos < far_pair_count)
            & (src >= 0)
            & (tgt >= 0)
        )
        src_s = jnp.maximum(src, 0)
        tgt_s = jnp.maximum(tgt, 0)
        delta = node_centers[src_s] - node_centers[tgt_s]
        reach = jnp.sqrt(jnp.sum(delta * delta, axis=1)) + node_radii[tgt_s]
        contrib = g * node_mass[src_s] / (reach * reach + soft)
        contrib = jnp.where(live & (reach > 0), contrib, jnp.zeros((), dtype))
        return own + jax.ops.segment_sum(contrib, tgt_s, num_segments=int(total_nodes))

    own = lax.fori_loop(0, num_chunks, _chunk, jnp.zeros((int(total_nodes),), dtype))
    total_far = ancestor_sum_by_level(
        own,
        left_child,
        right_child,
        parent,
        nodes_by_level,
        level_offsets,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
    )
    ranges = node_ranges[leaf_nodes]
    counts = jnp.maximum(ranges[:, 1] - ranges[:, 0] + 1, 0)
    n = int(num_particles)
    leaf_of = jnp.repeat(
        jnp.arange(leaf_nodes.shape[0], dtype=INDEX_DTYPE),
        counts,
        total_repeat_length=n,
    )
    live_particle = jnp.arange(n, dtype=INDEX_DTYPE) < jnp.sum(counts)
    return jnp.where(live_particle, total_far[leaf_nodes[leaf_of]], 0.0).astype(
        node_mass.dtype
    )


def node_force_scale_min(
    *, tree: Any, force_scale_particles: Array, num_levels: int
) -> Array:
    """Per-node ``min_b f_b`` from a per-particle force scale in INPUT order.

    What eq (16a)'s threshold needs for every node: the force scale carried from
    the previous step's evaluation, sorted by this step's tree, then
    :func:`node_force_scale_min_sorted`.

    Parameters
    ----------
    tree : Any
        This step's tree.
    force_scale_particles : Array
        ``[N]`` ``f_b`` per particle in input order.
    num_levels : int
        Level-loop bound (the upward pass's). Static.

    Returns
    -------
    Array
        ``[total_nodes]`` the minimum ``f_b`` over each node's particles.
    """
    perm = jnp.asarray(tree.particle_indices, dtype=INDEX_DTYPE)
    return node_force_scale_min_sorted(
        tree=tree,
        force_scale_sorted=jnp.asarray(force_scale_particles)[perm],
        num_levels=num_levels,
    )


def node_force_scale_min_sorted(
    *, tree: Any, force_scale_sorted: Array, num_levels: int
) -> Array:
    """Per-node ``min_b f_b`` from a per-particle force scale in the tree's order.

    Each leaf's minimum over its particle range, then up the tree
    (:func:`subtree_min_by_level`). Empty leaves get ``+inf`` (their nodes are
    dead in the walk anyway). The level-order replacement, on the fused lane, of
    ``compute_node_force_scale_from_sorted_magnitudes``'s serial loop over the
    internal nodes.

    Parameters
    ----------
    tree : Any
        The tree whose order ``force_scale_sorted`` is in.
    force_scale_sorted : Array
        ``[N]`` ``f_b`` per sorted particle.
    num_levels : int
        Level-loop bound (the upward pass's). Static.

    Returns
    -------
    Array
        ``[total_nodes]`` the minimum ``f_b`` over each node's particles.
    """
    from yggdrax.tree import get_level_offsets, get_nodes_by_level

    from jaccpot.runtime._level_shapes import level_batch_width

    total = int(jnp.asarray(tree.parent).shape[0])
    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    offsets = get_level_offsets(tree)
    return _node_force_scale_min_sorted_jit(
        jnp.asarray(force_scale_sorted),
        jnp.asarray(tree.parent, dtype=INDEX_DTYPE),
        jnp.asarray(tree.left_child, dtype=INDEX_DTYPE),
        jnp.asarray(tree.right_child, dtype=INDEX_DTYPE),
        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        jnp.asarray(get_nodes_by_level(tree), dtype=INDEX_DTYPE),
        jnp.asarray(offsets, dtype=INDEX_DTYPE),
        num_internal=num_internal,
        num_levels=int(num_levels),
        level_batch_width=level_batch_width(
            offsets, total_nodes=total, num_internal=num_internal
        ),
    )


@partial(jax.jit, static_argnames=("num_internal", "num_levels", "level_batch_width"))
def _node_force_scale_min_sorted_jit(
    scale: Array,
    parent: Array,
    left_child: Array,
    right_child: Array,
    node_ranges: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    *,
    num_internal: int,
    num_levels: int,
    level_batch_width: int,
) -> Array:
    n = int(scale.shape[0])
    ranges = node_ranges[num_internal:]
    counts = jnp.maximum(ranges[:, 1] - ranges[:, 0] + 1, 0)
    # leaves in particle order (an empty leaf sorts last, its count is zero)
    order = jnp.argsort(jnp.where(counts > 0, ranges[:, 0], n), stable=True)
    leaf_of = order[
        jnp.repeat(
            jnp.arange(order.shape[0], dtype=INDEX_DTYPE),
            counts[order],
            total_repeat_length=n,
        )
    ]
    live = jnp.arange(n, dtype=INDEX_DTYPE) < jnp.sum(counts)
    inf = jnp.asarray(jnp.inf, scale.dtype)
    leaf_min = jax.ops.segment_min(
        jnp.where(live, scale, inf), leaf_of, num_segments=int(ranges.shape[0])
    )
    leaf_min = jnp.where(counts > 0, leaf_min, inf)
    values = jnp.concatenate([jnp.full((num_internal,), inf, scale.dtype), leaf_min])
    return subtree_min_by_level(
        values,
        left_child,
        right_child,
        parent,
        nodes_by_level,
        level_offsets,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
    )
