"""Dehnen eq (16b) force scale under domain decomposition.

The Dehnen (2014) §5 mass-dependent MAC accepts a pair when the estimated error
falls below ``eps * s``, where ``s = min_b f_b`` over the target cell and
``f_b = sum_{a != b} G m_a / |x_a - x_b|^2`` is the cancellation-free force scale.
On one device :func:`jaccpot.runtime._adaptive_policy.estimate_particle_force_scale`
builds ``f_b`` in O(N) from the interaction lists the traversal already produced,
splitting it the way the FMM splits the force itself: exact over near pairs,
monopole over far pairs.

**On a mesh that sum is incomplete, and incomplete in the dangerous direction is
not the issue -- incomplete is fine, wrong is not.** Each device holds only its own
particles, so a purely local estimate omits every remote source. Omission makes
``f_b`` *smaller*, which tightens ``eps * f_b``, which accepts *fewer* pairs: slower
and more accurate. That is the safe direction (see trap 6 in
``docs/dehnen_mass_mac_status_and_plan.md`` -- an over-large scale makes the solver
faster *and* wronger, which no cost measurement can detect). But omitting the remote
mass entirely is a large error, not a small one: at five devices a particle's own
domain holds a fifth of the system.

This module completes the sum from artifacts the distributed lane **already builds**,
with no extra communication:

* **local near, exact** -- the self walk's neighbour CSR plus each leaf's self block.
* **local far, monopole** -- the self walk's far pairs, accumulated down the tree.
* **remote far, monopole** -- the cross walk's far pairs against the all-gathered
  coarse (LET) tree, whose nodes carry the remote leaf frontier's mass and COM.
* **remote near, monopole** -- the cross walk's near CSR, same coarse sources.

The remote terms are monopole because the coarse tree is all the remote information
a device has; the halo carries particles, but only for the leaves the near walk
selected, and reading it here would put the estimate behind the halo exchange for no
accuracy that ``f_b`` can use.

**Why the remote near term inflates by BOTH radii.** The single-GPU far term divides
by ``(|c_A - c_U| + rho_U)^2``: the target radius bounds ``|c_A - x_b|`` from above
for every particle ``b`` under ``U``, so each term is an under-estimate and the total
is a lower bound. That argument uses the source's compactness, which a far pair has
by construction -- it passed the opening criterion. A *near* pair has not, so its
source spread is unbounded and ``1/r^2`` is **not** convex in the vector argument
(the Hessian of ``r^-2`` has a negative tangential eigenvalue), which means
monopole-at-COM is not a bound in either direction. Adding the source radius as well
restores it: ``|x_a - x_b| <= |c_A - c_U| + rho_U + rho_A`` for every pair in the two
nodes, so every term is again an under-estimate and the whole estimate stays a lower
bound on the exact ``f_b``.

Only eq (16b) is served here. eq (16a) puts ``min_b |a_b|`` on the right-hand side,
which is an *acceleration* and therefore a full extra distributed evaluation --
halo exchange included -- rather than a read of lists that already exist. See
:func:`distributed_force_scale_nodes` for what it raises instead.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from jaccpot.runtime._adaptive_policy import (
    _near_field_force_scale,
    _particle_leaf_ids,
    accumulate_own_down_parent_chain,
    compute_node_force_scale_from_sorted_magnitudes,
    node_span_mass,
)
from jaccpot.upward.tree_expansions import NodeMultipoleData, TreeUpwardData

__all__ = [
    "DISTRIBUTED_FORCE_SCALE_MODES",
    "cross_force_scale_own",
    "distributed_force_scale_nodes",
    "flatten_neighbor_csr",
    "policy_upward_view",
]

#: ``mac_force_scale_mode`` values this lane can serve. eq (16b) only -- see the
#: module docstring for why eq (16a) is a different piece of work.
DISTRIBUTED_FORCE_SCALE_MODES = ("paper_fb",)


def policy_upward_view(*, upward: Any, geometry: Any, mass_moments: Any) -> Any:
    """Bundle a per-device upward sweep into what the policy builder expects.

    ``build_adaptive_policy_state`` reads ``upward.geometry``,
    ``upward.mass_moments`` and ``upward.multipoles``. The real-basis sweep returns
    a :class:`RealTreeUpwardData`, which carries **only** ``multipoles`` -- it says
    so in its own docstring -- so handing it straight over raises
    ``AttributeError`` on the other two. The runtime solves this in
    ``fmm_sweeps`` by rebuilding a :class:`TreeUpwardData` around the real
    coefficients; this is the same construction, for the same reason.

    Both bases are wrapped, not only the real one, so the criterion measures its
    geometry from the SAME ``TreeGeometry`` the two walks used. The complex sweep
    computes a geometry of its own, and letting the criterion silently take that
    one instead would put the acceptance test and the traversal on different node
    radii.

    **Everything here is frozen with ``stop_gradient``, and that is the point.**
    The criterion decides the far/near split, which is part of the DISCRETE
    topology, and the distributed lane's differentiable mode is built on a
    fixed-topology seam: the tree, its geometry and every walk take
    ``stop_gradient``-ed inputs so no cotangent can reach the accept mask, the
    Morton order or node membership. The upward sweep handed in here is the LIVE
    one -- it has to be, the M2L consumes it -- so wrapping it without freezing
    would route cotangents into the traversal's ``lax.while_loop``, which is not
    reverse-mode differentiable. Freezing happens inside this function rather than
    at its call sites so a second caller cannot forget. The forward value is
    unchanged, in both modes.

    Parameters
    ----------
    upward : Any
        The per-device upward-sweep result, real or complex basis. May be live;
        it is frozen here.
    geometry : Any
        The ``TreeGeometry`` the walks were run against.
    mass_moments : Any
        Per-node mass moments for the same tree. May be live; frozen here.

    Returns
    -------
    Any
        A ``TreeUpwardData`` the policy builder can consume, carrying no cotangent
        path back to positions or masses.
    """

    return TreeUpwardData(
        geometry=jax.lax.stop_gradient(geometry),
        mass_moments=jax.lax.stop_gradient(mass_moments),
        multipoles=NodeMultipoleData(
            order=int(upward.multipoles.order),
            centers=jax.lax.stop_gradient(upward.multipoles.centers),
            moments=None,  # type: ignore[arg-type]
            packed=jax.lax.stop_gradient(upward.multipoles.packed),
            component_matrix=None,
            source_motion_packed=None,
        ),
    )


def flatten_neighbor_csr(
    *,
    counts: Array,
    indices: Array,
    leaf_indices: Array,
) -> tuple[Array, Array, Array]:
    """Expand a per-leaf neighbour CSR into flat ``(source, target)`` pair arrays.

    The near lists are keyed by target leaf; the monopole accumulation below wants
    the same shape as a far list, which is one entry per pair. ``searchsorted`` over
    the running count recovers each flat entry's owning leaf, exactly as
    ``_combined_neighbors`` does for the P2P build -- same trick, so the two cannot
    disagree about which target a neighbour belongs to.

    Parameters
    ----------
    counts : Array
        ``(num_leaves,)`` neighbours per target leaf.
    indices : Array
        Flat neighbour entries, source node ids, fixed capacity with a padded tail.
    leaf_indices : Array
        ``(num_leaves,)`` node index of each target leaf.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(sources, targets, valid)``: source node id, target node id and whether
        the slot holds a real pair, each of the flat buffer's length.
    """

    counts = jnp.asarray(counts, dtype=jnp.int32)
    sources = jnp.asarray(indices, dtype=jnp.int32)
    leaf_indices = jnp.asarray(leaf_indices, dtype=jnp.int32)

    num_edges = int(sources.shape[0])
    if num_edges == 0 or int(counts.shape[0]) == 0:
        empty = jnp.zeros((0,), dtype=jnp.int32)
        return empty, empty, jnp.zeros((0,), dtype=bool)

    running = jnp.cumsum(counts)
    edge = jnp.arange(num_edges, dtype=jnp.int32)
    slot = jnp.searchsorted(running, edge, side="right")
    valid = edge < running[-1]
    targets = leaf_indices[jnp.clip(slot, 0, int(leaf_indices.shape[0]) - 1)]
    return sources, targets, valid & (sources >= 0) & (targets >= 0)


def cross_force_scale_own(
    *,
    source_masses: Array,
    source_centers: Array,
    source_radii: Optional[Array],
    target_centers: Array,
    target_radii: Array,
    pair_sources: Array,
    pair_targets: Array,
    pair_valid: Optional[Array],
    num_target_nodes: int,
    g: Array,
    eps_sq: Array,
    inflation: Array,
) -> Array:
    """Accumulate a monopole force scale from remote nodes onto local target nodes.

    The cross-tree twin of
    :func:`jaccpot.runtime._adaptive_policy._far_field_force_scale_by_node`: sources
    live in the coarse (LET) tree and targets in the local tree, so node masses,
    centres and radii come from two different trees and cannot share an index space.
    Returns the *own* contribution per target node; the caller pushes it down the
    parent chain once, together with every other own-term.

    Parameters
    ----------
    source_masses : Array
        ``(num_source_nodes,)`` mass spanned by each coarse node.
    source_centers : Array
        ``(num_source_nodes, 3)`` coarse node centres.
    source_radii : Optional[Array]
        ``(num_source_nodes,)`` coarse node radii, added to the reach so every term
        stays an under-estimate. ``None`` reproduces the single-GPU far-field
        convention (target radius only), which is what a *far* pair wants because it
        has already passed the opening criterion.
    target_centers : Array
        ``(num_target_nodes, 3)`` local node centres.
    target_radii : Array
        ``(num_target_nodes,)`` local node radii, scaled by ``inflation``.
    pair_sources : Array
        Source (coarse) node of each pair.
    pair_targets : Array
        Target (local) node of each pair.
    pair_valid : Optional[Array]
        Which slots hold a real pair. ``None`` derives it from non-negative ids.
    num_target_nodes : int
        Segment count for the scatter; the local tree's node count.
    g : Array
        ``G`` as an array.
    eps_sq : Array
        Squared softening length.
    inflation : Array
        Multiplier on the target radius; ``1.0`` keeps every term an under-estimate.

    Returns
    -------
    Array
        ``(num_target_nodes,)`` own contribution, NOT yet accumulated down the
        parent chain.
    """

    dtype = jnp.asarray(target_centers).dtype
    src = jnp.asarray(pair_sources, dtype=jnp.int32)
    tgt = jnp.asarray(pair_targets, dtype=jnp.int32)
    if num_target_nodes <= 0 or int(src.shape[0]) == 0:
        return jnp.zeros((max(int(num_target_nodes), 0),), dtype=dtype)

    live = (src >= 0) & (tgt >= 0)
    if pair_valid is not None:
        live = live & jnp.asarray(pair_valid, dtype=bool)
    src_safe = jnp.maximum(src, 0)
    tgt_safe = jnp.maximum(tgt, 0)

    delta = (
        jnp.asarray(source_centers, dtype=dtype)[src_safe]
        - jnp.asarray(target_centers, dtype=dtype)[tgt_safe]
    )
    distance = jnp.sqrt(jnp.sum(delta * delta, axis=1))
    reach = distance + inflation * jnp.asarray(target_radii, dtype=dtype)[tgt_safe]
    if source_radii is not None:
        reach = reach + jnp.asarray(source_radii, dtype=dtype)[src_safe]

    contrib = (
        g * jnp.asarray(source_masses, dtype=dtype)[src_safe] / (reach * reach + eps_sq)
    )
    contrib = jnp.where(live & (reach > 0), contrib, jnp.asarray(0.0, dtype=dtype))
    return jax.ops.segment_sum(
        contrib,
        tgt_safe,
        num_segments=int(num_target_nodes),
        indices_are_sorted=False,
    )


def distributed_force_scale_nodes(
    *,
    tree: Any,
    positions_sorted: Array,
    masses_sorted: Array,
    node_centers: Array,
    node_radii: Array,
    self_far_sources: Array,
    self_far_targets: Array,
    self_near_offsets: Array,
    self_near_counts: Array,
    self_near_indices: Array,
    self_near_leaf_indices: Array,
    coarse_tree: Any,
    coarse_masses_sorted: Array,
    coarse_centers: Array,
    coarse_radii: Array,
    cross_far_sources: Array,
    cross_far_targets: Array,
    cross_near_counts: Array,
    cross_near_indices: Array,
    cross_near_leaf_indices: Array,
    max_leaf_size: int,
    softening: float,
    gravitational_constant: float,
    force_scale_mode: str = "paper_fb",
    far_center_inflation: float = 1.0,
    near_pair_chunk: int = 32,
) -> Array:
    """Return the per-node Dehnen eq (16b) force scale for one device's subdomain.

    Assembles the four terms named in the module docstring and reduces the
    per-particle result onto nodes with ``min``, which is what eq (16a)'s
    ``min_b`` over the target cell means. The result is a **lower bound** on the
    exact ``f_b``: every term is an under-estimate by construction and the remote
    contribution beyond the coarse tree's reach is simply absent.

    Runs entirely on local artifacts and the all-gathered coarse tree, so it is
    safe to call inside ``shard_map`` and adds no collective of its own.

    Parameters
    ----------
    tree : Any
        The device's local tree.
    positions_sorted : Array
        ``(n, 3)`` local positions in tree order.
    masses_sorted : Array
        ``(n,)`` local masses in tree order.
    node_centers : Array
        ``(num_nodes, 3)`` local node centres.
    node_radii : Array
        ``(num_nodes,)`` local node radii.
    self_far_sources : Array
        Source node of each local far pair.
    self_far_targets : Array
        Target node of each local far pair.
    self_near_offsets : Array
        CSR offsets into ``self_near_indices``, taken from the walk rather than
        rebuilt, so this cannot disagree with the list it indexes.
    self_near_counts : Array
        Neighbours per local target leaf.
    self_near_indices : Array
        Flat local near neighbour entries.
    self_near_leaf_indices : Array
        Node index of each local target leaf.
    coarse_tree : Any
        The all-gathered coarse (LET) tree over every device's leaf frontier.
    coarse_masses_sorted : Array
        Coarse-tree masses in its own sorted order.
    coarse_centers : Array
        ``(num_coarse_nodes, 3)`` coarse node centres.
    coarse_radii : Array
        ``(num_coarse_nodes,)`` coarse node radii.
    cross_far_sources : Array
        Coarse source node of each cross far pair.
    cross_far_targets : Array
        Local target node of each cross far pair.
    cross_near_counts : Array
        Coarse near neighbours per local target leaf.
    cross_near_indices : Array
        Flat cross near neighbour entries, coarse node ids.
    cross_near_leaf_indices : Array
        Node index of each local target leaf in the cross near list.
    max_leaf_size : int
        Leaf capacity.
    softening : float
        Plummer softening length, squared internally.
    gravitational_constant : float
        ``G``.
    force_scale_mode : str
        Must be in :data:`DISTRIBUTED_FORCE_SCALE_MODES`.
    far_center_inflation : float
        Scales the target radius into the reach; ``1.0`` keeps the lower bound.
    near_pair_chunk : int
        Pairs per chunk in the exact local near sum; a memory knob only.

    Returns
    -------
    Array
        ``(num_nodes,)`` per-node force scale, min-reduced over each node's
        particles, ready for ``build_adaptive_policy_state``.

    Raises
    ------
    ValueError
        If ``force_scale_mode`` is not one this lane can serve.
    """

    mode = str(force_scale_mode).strip().lower()
    if mode not in DISTRIBUTED_FORCE_SCALE_MODES:
        raise ValueError(
            "the distributed lane serves eq (16b) only: mac_force_scale_mode must "
            f"be one of {DISTRIBUTED_FORCE_SCALE_MODES}, got {force_scale_mode!r}. "
            "eq (16a) needs min_b |a_b|, which is an acceleration and therefore a "
            "second distributed evaluation including the halo exchange, not a read "
            "of lists that already exist."
        )

    positions = jnp.asarray(positions_sorted)
    dtype = positions.dtype
    masses = jnp.asarray(masses_sorted, dtype=dtype)
    num_particles = int(positions.shape[0])
    if num_particles == 0:
        return jnp.zeros((int(jnp.asarray(node_radii).shape[0]),), dtype=dtype)

    leaf_cap = max(int(max_leaf_size), 1)
    g = jnp.asarray(float(gravitational_constant), dtype=dtype)
    eps_sq = jnp.asarray(float(softening) ** 2, dtype=dtype)
    inflation = jnp.asarray(float(far_center_inflation), dtype=dtype)
    node_ranges = jnp.asarray(tree.node_ranges, dtype=jnp.int32)
    num_nodes = int(node_ranges.shape[0])

    # 1. local near, exact -- the same pair set P2P evaluates, self block included.
    near = _near_field_force_scale(
        positions=positions,
        masses=masses,
        node_ranges=node_ranges,
        neighbor_offsets=jnp.asarray(self_near_offsets, dtype=jnp.int32),
        neighbor_counts=jnp.asarray(self_near_counts, dtype=jnp.int32),
        neighbor_leaf_indices=jnp.asarray(self_near_leaf_indices, dtype=jnp.int32),
        neighbor_indices=jnp.asarray(self_near_indices, dtype=jnp.int32),
        leaf_cap=leaf_cap,
        g=g,
        eps_sq=eps_sq,
        chunk=max(int(near_pair_chunk), 1),
    )

    local_node_mass = node_span_mass(tree=tree, masses_sorted=masses)
    centers = jnp.asarray(node_centers, dtype=dtype)
    radii = jnp.asarray(node_radii, dtype=dtype)

    # 2. local far, monopole. Same convention as the single-GPU estimator: the
    #    target radius alone, because a far pair's source is compact by construction.
    own = cross_force_scale_own(
        source_masses=local_node_mass,
        source_centers=centers,
        source_radii=None,
        target_centers=centers,
        target_radii=radii,
        pair_sources=self_far_sources,
        pair_targets=self_far_targets,
        pair_valid=None,
        num_target_nodes=num_nodes,
        g=g,
        eps_sq=eps_sq,
        inflation=inflation,
    )

    coarse_node_mass = node_span_mass(
        tree=coarse_tree, masses_sorted=jnp.asarray(coarse_masses_sorted, dtype=dtype)
    )
    c_centers = jnp.asarray(coarse_centers, dtype=dtype)
    c_radii = jnp.asarray(coarse_radii, dtype=dtype)

    # 3. remote far, monopole -- coarse sources, same convention as the local far term.
    own = own + cross_force_scale_own(
        source_masses=coarse_node_mass,
        source_centers=c_centers,
        source_radii=None,
        target_centers=centers,
        target_radii=radii,
        pair_sources=cross_far_sources,
        pair_targets=cross_far_targets,
        pair_valid=None,
        num_target_nodes=num_nodes,
        g=g,
        eps_sq=eps_sq,
        inflation=inflation,
    )

    # 4. remote near, monopole -- BOTH radii, because a near pair's source is not
    #    compact and monopole-at-COM is then a bound in neither direction.
    x_src, x_tgt, x_valid = flatten_neighbor_csr(
        counts=cross_near_counts,
        indices=cross_near_indices,
        leaf_indices=cross_near_leaf_indices,
    )
    own = own + cross_force_scale_own(
        source_masses=coarse_node_mass,
        source_centers=c_centers,
        source_radii=c_radii,
        target_centers=centers,
        target_radii=radii,
        pair_sources=x_src,
        pair_targets=x_tgt,
        pair_valid=x_valid,
        num_target_nodes=num_nodes,
        g=g,
        eps_sq=eps_sq,
        inflation=inflation,
    )

    # One descent for all three monopole terms. A node's own list belongs to every
    # particle beneath it, so the push-down is the same for local and remote.
    by_node = accumulate_own_down_parent_chain(tree=tree, own=own)

    leaf_of_particle = _particle_leaf_ids(
        tree=tree,
        num_particles=num_particles,
        max_leaf_size=leaf_cap,
    )
    monopole = jnp.where(
        leaf_of_particle >= 0,
        by_node[jnp.maximum(leaf_of_particle, 0)],
        jnp.asarray(0.0, dtype=dtype),
    )
    return compute_node_force_scale_from_sorted_magnitudes(
        tree=tree,
        magnitudes_sorted=near + monopole,
        reduction="min",
        # Static, because the exact per-leaf maximum is a host read of a traced
        # value and this runs inside ``shard_map``. The leaf capacity is a true
        # bound, so the reduction is identical to the untraced path.
        max_leaf_size=leaf_cap,
    )
