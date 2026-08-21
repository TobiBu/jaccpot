"""A distributed mutual FMM: intra-domain pairs locally, cross-domain pairs exactly once.

Assembles the three pieces TobiBu/jaccpot#173 decomposed the problem into -- the
canonical cross-domain emitter and straddling-node rule
(``yggdrax.distributed.cross_walk``), the reverse halo that returns the ``-f`` half
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

Remote tree structure is obtained by ``all_gather`` of the per-device node arrays rather
than through the LET's coarse frontier: equivalent at the sizes this is tested at, O(N)
per device, and to be replaced by ``build_remote_coarse_tree`` before use at scale.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array
from yggdrax.distributed.cross_walk import (
    dual_tree_walk_cross_mutual,
    single_owner_domain,
)
from yggdrax.distributed.reverse_halo import apply_reverse_halo, export_reverse_halo
from yggdrax.distributed.sharding import AXIS_NAME
from yggdrax.dtypes import INDEX_DTYPE

__all__ = ["DistributedMutualResult", "distributed_mutual_accelerations"]


class DistributedMutualResult(NamedTuple):
    """Per-device accelerations plus the flags a caller must check.

    ``overflow`` is reduced ACROSS devices, not per device: a cap exceeded anywhere is
    a wrong force everywhere, and a dropped cross pair loses both halves, so momentum
    stays exact while the force is wrong. A per-device check would miss exactly the
    case worth catching.
    """

    acceleration: Array
    cross_pairs: Array
    overflow: Array

    # `cross_pairs` and `overflow` carry a leading singleton axis so `shard_map` can
    # concatenate them across devices; after the map both are `(ndev,)`.


def _leaf_tile(node_ranges: Array, leaf: Array, width: int):
    """Fixed-width particle indices and validity for each leaf in ``leaf``.

    Leaves hold at most ``width`` particles, so a fixed tile plus a mask keeps the
    shapes static without padding the physics: an invalid slot contributes zero.

    Parameters
    ----------
    node_ranges:
        ``(nodes, 2)`` particle range per node.
    leaf:
        ``(k,)`` leaf indices.
    width:
        Maximum particles per leaf. Static.

    Returns
    -------
    tuple[Array, Array]
        ``(index, valid)``, both ``(k, width)``.
    """
    start = node_ranges[leaf, 0]
    end = node_ranges[leaf, 1]
    off = jnp.arange(width, dtype=start.dtype)[None, :]
    idx = start[:, None] + off
    return idx, idx < end[:, None]


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
        ``(nodes,)`` left children, -1 for leaves.
    right_child_full:
        ``(nodes,)`` right children, -1 for leaves.
    node_ranges:
        ``(nodes, 2)`` particle range per node.
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
        Maximum particles per leaf. Static.
    near_cap:
        Cross leaf-pair capacity, per remote domain.
    max_pair_queue:
        Cross wavefront capacity.
    recv_capacity:
        Reverse-halo receive capacity.
    axis_name:
        Mesh axis.

    Returns
    -------
    DistributedMutualResult
        Total acceleration for this domain, owned cross leaf-pair count, and the
        across-device overflow flag.
    """
    me = jax.lax.axis_index(axis_name)

    all_left = jax.lax.all_gather(left_child_full, axis_name)
    all_right = jax.lax.all_gather(right_child_full, axis_name)
    all_centers = jax.lax.all_gather(centers, axis_name)
    all_radii = jax.lax.all_gather(radii, axis_name)
    all_ranges = jax.lax.all_gather(node_ranges, axis_name)
    all_pos = jax.lax.all_gather(positions, axis_name)
    all_mass = jax.lax.all_gather(masses, axis_name)

    acc = local_acceleration
    cap_total = ndev * near_cap
    # `INDEX_DTYPE`, not a hardcoded int32: yggdrax's index dtype is int64 when x64 is
    # enabled, and a scatter whose value is wider than its buffer is a FutureWarning
    # now and an error in later JAX. Matching the library's dtype is the fix; casting
    # the values down would silently narrow indices instead.
    owner_out = jnp.full((cap_total,), -1, dtype=INDEX_DTYPE)
    index_out = jnp.zeros((cap_total,), dtype=INDEX_DTYPE)
    value_out = jnp.zeros((cap_total, 3), dtype=positions.dtype)
    n_pairs = jnp.zeros((), dtype=INDEX_DTYPE)
    overflow = jnp.asarray(False)

    for other in range(ndev):
        # Every node of `other`'s tree belongs to `other`, so nothing straddles in this
        # stand-in; `single_owner_domain` is still called so the shape of the call is
        # identical once a merged LET replaces the all_gather.
        tag = jnp.full((left_child_full.shape[0],), other, dtype=INDEX_DTYPE)
        owner_nodes = single_owner_domain(
            all_left[other], all_right[other], tag, max_depth=64
        )
        # theta = 0: the MAC never accepts, so every cross pair refines to leaf-leaf
        # and lands in the near list, where it is summed exactly.
        res = dual_tree_walk_cross_mutual(
            left_child_full,
            right_child_full,
            centers,
            radii,
            root,
            all_left[other],
            all_right[other],
            all_centers[other],
            all_radii[other],
            root,
            0.0,
            this_device=me,
            remote_owner=owner_nodes,
            max_pair_queue=max_pair_queue,
            far_cap=1,
            near_cap=near_cap,
        )
        overflow = overflow | res.near_overflow | res.queue_overflow

        # A device never pairs with itself here: intra-domain pairs are the local
        # self-walk's job and counting them again would double them.
        rows = jnp.arange(near_cap, dtype=INDEX_DTYPE)
        live = (rows < res.near_count) & (me != other)
        la = jnp.where(live, res.near_local, 0)
        rb = jnp.where(live, res.near_remote, 0)

        ia, oka = _leaf_tile(node_ranges, la, leaf_width)
        ib, okb = _leaf_tile(all_ranges[other], rb, leaf_width)
        oka = oka & live[:, None]
        okb = okb & live[:, None]
        f_a, f_b = _tile_forces(
            positions[ia],
            jnp.where(oka, masses[ia], 0.0),
            oka,
            all_pos[other][ib],
            jnp.where(okb, all_mass[other][ib], 0.0),
            okb,
            softening,
            g,
        )
        acc = acc.at[jnp.where(oka, ia, positions.shape[0]).astype(INDEX_DTYPE)].add(
            jnp.where(oka[..., None], f_a, 0.0), mode="drop"
        )

        # The -f half, one row per remote PARTICLE, tagged with its owning domain.
        flat_owner = jnp.where(okb, res.near_owner[:, None], -1).reshape(-1)
        flat_index = jnp.where(okb, ib, 0).reshape(-1)
        flat_value = jnp.where(okb[..., None], f_b, 0.0).reshape(-1, 3)
        keep = flat_owner >= 0
        counted = keep.astype(INDEX_DTYPE)
        slot = n_pairs + jnp.cumsum(counted) - counted
        fits = keep & (slot < cap_total)
        overflow = overflow | jnp.any(keep & (slot >= cap_total))
        # int32 throughout: a scatter whose indices are wider than the buffer's dtype
        # is a FutureWarning now and an error in later JAX.
        safe = jnp.where(fits, slot, cap_total).astype(INDEX_DTYPE)
        owner_out = owner_out.at[safe].set(jnp.where(fits, flat_owner, -1), mode="drop")
        index_out = index_out.at[safe].set(jnp.where(fits, flat_index, 0), mode="drop")
        value_out = value_out.at[safe].set(
            jnp.where(fits[:, None], flat_value, 0.0), mode="drop"
        )
        n_pairs = n_pairs + jnp.sum(counted)

    halo = export_reverse_halo(
        owner_out,
        index_out,
        value_out,
        ndev,
        recv_capacity=recv_capacity,
        axis_name=axis_name,
    )
    overflow = overflow | halo.overflow
    acc = apply_reverse_halo(acc, halo)
    # Shape (1,) rather than scalar for the two diagnostics: `shard_map` cannot
    # concatenate rank-0 per-device outputs, so a function documented as "call inside
    # shard_map" should hand back something its `out_specs` can actually describe.
    # The caller sees `(ndev,)` after the map.
    return DistributedMutualResult(
        acceleration=acc,
        cross_pairs=n_pairs[None],
        overflow=(jax.lax.pmax(overflow.astype(INDEX_DTYPE), axis_name) > 0)[None],
    )
