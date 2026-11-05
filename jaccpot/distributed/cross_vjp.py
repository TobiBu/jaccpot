"""Reverse mode of the cross-domain NEAR term (multi-GPU gradients, seam B).

The fused multi-GPU lane adds, after each device's local force, the near field of the
particles it imported from its peers
(:func:`~jaccpot.distributed.cross.cross_near_acceleration_direct`): ONE sorted table,
this device's sorted particles then the flat receive buffer of the near exchange, the
local leaves the only targets (``num_target_leaves=L``), the cross pairs a CSR over
those leaves, no self term. :func:`cross_near_direct_cvjp` is that call under a
``custom_vjp``: its forward IS the production table and the production
:func:`~jaccpot.pallas.nearfield_leafpair_csr.nearfield_leafpair_csr_sorted_direct_pallas`
call (the same bits, the same program for traced operands), and its reverse needs no
transposed exchange.

THE REVERSE. With ``d = x_p - x_q`` and ``c`` the acceleration cotangents, the cross
term of a sink ``p`` on this device is ``a_p = -G sum_q m_q g(|d|^2) d`` over the
particles ``q`` of the imported leaves its leaf is paired with. Its adjoint has a
sink side (``p``'s own ``c_p``) and a source side (``p`` as a source of the remote
sinks ``q``, which needs THEIR cotangents ``c_q``):
``x_p_bar = G sum_q (m_p g c_q - m_q g c_p - g' (m_q (c_p . d) - m_p (c_q . d)) d)``,
``m_p_bar = G sum_q g (c_q . d)`` (the local direct reverse's algebra,
:mod:`jaccpot.pallas.nearfield_leafpair_csr_direct_vjp`). The cross pair set is
GLOBALLY SYMMETRIC (:func:`~jaccpot.distributed.cross._symmetric_walk` gives both
devices of a pair the identical pairs): when my leaf ``A`` lists the peer's leaf
``B``, the peer's leaf ``B`` lists ``A``, so every ``q`` I imported is a sink whose
cross term has ``p`` as a source, on the device that owns it -- and I hold ``q``'s
position and mass already, in the row the forward imported it to. Only ``c_q`` is
missing, and it travels the way ``q`` did: the backward sends
``where(live_q, c_sorted[src], 0)`` with the forward's ``part_sizes`` through the same
ragged exchange (same direction, same layout), so the received cotangent row ``r`` is
the cotangent of the particle in imported row ``r``. Then the direct reverse
(:func:`~jaccpot.pallas.nearfield_leafpair_csr_direct_vjp.
nearfield_leafpair_csr_sorted_direct_reverse_pallas` with ``num_target_leaves=L``,
``include_self=False``) runs on the table ``[pos; imported pos]``, ``[m; imported
m]`` with the cotangent table ``[c_sorted; c_imported]``: the L local leaves are its
only targets, each program writes its own particles' cotangents (both sides), nothing
is scattered and nothing is sent back. The exchange is never transposed (and JAX never
sees one to transpose: the imports are constants of the reverse).

WHAT IS DIFFERENTIATED: the ``(n, 3)`` cross acceleration of the local sorted rows
with respect to the local sorted positions and masses, ``softening_sq`` and ``G``.
The imported particles are the peers' particles: their cotangents are written by
their owners (the source side above), so they get none here (pass them
``stop_gradient``'d; :func:`cross_near_acceleration_direct_cvjp` does). The table,
the CSR, ``src`` / ``live_q`` / ``part_sizes`` and the receive capacity are topology
(frozen). The force-scale lane (``with_force_scale``, eq (16b)'s near half) is
computed as in production and is a constant: its cotangent is dropped.

``softening_sq`` AND ``G`` COTANGENTS ARE PER-DEVICE PARTIALS. ``softening_sq_bar`` is
this device's sinks' sum; ``G_bar`` is Euler over the SOURCE-side sums (``G G_bar =
sum_p m_p m_p_bar``), which relabels pairs across devices, so it is not this device's
sinks' share either. Only their sum over the mesh axis is the scalar's cotangent.
Inside a ``shard_map`` whose ``in_specs`` give the scalars ``P()`` (replicated), JAX's
transpose of the ``shard_map`` sums the per-device cotangents of a replicated input
(``check_vma=False``; the MG3b gate measures it); a caller that differentiates
inside the ``shard_map`` must ``psum`` them itself.

PRECONDITIONS (the caller's: G1 records them, G4 gates them). (1) The pair set is
globally symmetric and the near exchange did not overflow (an overflowed buffer cuts
rows: the forward is flagged, the gradient must be gated). (2) Every live row of a
target leaf was sent to every peer that pairs with that leaf -- true of the default
exchange, which ships each paired leaf's live particles (``_live_node_ends``, at most
``W``; more is the overflow flag) -- so a target's source-side terms have a remote
sink behind them. A padded shard's dead rows lie in no leaf range: no target, zero
cotangents. (3) The local and remote ends use the same softening kernel, ``G`` and
``softening_sq`` (replicated), as the mesh force does.

COST. Backward = one ragged exchange of ``(send_particle_cap, 3)`` floats (the
forward's near exchange carries 4 columns) and one direct reverse launch over the
L local leaves on the ``(n + recv_particle_cap)``-row table: the single-card
reverse's memory per row (two ``(rows + pad, 4)`` tables, five ``(rows,)`` lanes).
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array
from yggdrax.distributed.comm import AXIS_NAME, ragged_all_to_all_exchange

__all__ = [
    "cross_near_acceleration_direct_cvjp",
    "cross_near_direct_cvjp",
    "cross_near_direct_layout",
]


def cross_near_direct_layout(
    near: dict,
    leaf_ranges: Array,
    num_internal: int,
    n: int,
) -> tuple[Array, Array, Array, Array, Array]:
    """The cross near term's leaf table and CSR, exactly as the production call builds it.

    The lines of :func:`~jaccpot.distributed.cross.cross_near_acceleration_direct`
    that turn the near sink into the kernel's topology, op for op (the forward of
    :func:`cross_near_direct_cvjp` must be that call's bits): the local leaves'
    ranges, then each received row's range of the flat import shifted past the
    ``n`` local rows; the cross pairs as a CSR over the ``L`` local leaf rows in the
    pair list's order, a target outside the local leaves or a source past the import
    dropped.

    Parameters
    ----------
    near : dict
        The near sink of the default exchange (``particles``, ``row_start``,
        ``row_count``, ``target_node``, ``source_row``, ``count``).
    leaf_ranges : Array
        ``(L, 2)`` inclusive particle range of every local leaf row (leaf node
        ``num_internal + r``).
    num_internal : int
        Internal-node count of the local tree. Static.
    n : int
        Local (sorted) particle rows. Static.

    Returns
    -------
    tuple[Array, Array, Array, Array, Array]
        ``(leaf_start (L + R,), leaf_count (L + R,), neighbors (P,), offsets (L + 1,),
        counts (L,))``, int32; ``L`` is ``leaf_ranges``' length, ``R`` the received
        rows.
    """
    idx = jnp.int32
    n = int(n)
    ranges = jnp.asarray(leaf_ranges).astype(idx)
    L = int(ranges.shape[0])
    parts = jnp.asarray(near["particles"])
    capacity = int(parts.shape[0])
    r_start = jnp.asarray(near["row_start"]).astype(idx)
    r_count = jnp.asarray(near["row_count"]).astype(idx)
    R = int(r_start.shape[0])
    loc_start = ranges[:, 0]
    loc_count = jnp.maximum(ranges[:, 1] - ranges[:, 0] + 1, 0)
    imp_start = jnp.clip(r_start, 0, capacity)
    imp_count = jnp.clip(jnp.minimum(r_count, capacity - imp_start), 0)
    leaf_start = jnp.concatenate([loc_start, n + imp_start])
    leaf_count = jnp.concatenate([loc_count, imp_count])
    tgt_node = jnp.asarray(near["target_node"]).astype(idx)
    src_row = jnp.asarray(near["source_row"]).astype(idx)
    P = int(tgt_node.shape[0])
    lrow = tgt_node - int(num_internal)
    live = (
        (jnp.arange(P) < jnp.asarray(near["count"]))
        & (lrow >= 0)
        & (lrow < L)
        & (src_row >= 0)
        & (src_row < R)
    )
    row = jnp.where(live, lrow, L)
    order = jnp.argsort(row, stable=True)
    neighbors = jnp.where(live[order], L + src_row[order], 0)
    counts = jnp.bincount(row, length=L + 1)[:L].astype(idx)
    offsets = jnp.concatenate([jnp.zeros((1,), idx), jnp.cumsum(counts, dtype=idx)])
    return leaf_start, leaf_count, neighbors, offsets, counts


class _CrossNearStatic(NamedTuple):
    """The seam's static options, one hashable ``nondiff_argnums`` operand.

    Attributes
    ----------
    num_target_leaves : int
        ``L``, the local leaf rows (the kernels' grid).
    leaf_width : int
        The target tiles' width (the production call's ``max`` of the local and the
        imported width).
    chunk : int
        Entries per chunk (the direct kernels' piece floor).
    recv_particle_cap : int
        The near exchange's receive capacity (the imported rows).
    axis_name : str
        The mesh axis of the exchange.
    softening_kernel : Optional[str]
        The pair kernel.
    interpret : bool
        Pallas interpret mode, forward and reverse.
    with_force_scale : bool
        Also return the force-scale lane (not differentiated).
    """

    num_target_leaves: int
    leaf_width: int
    chunk: int
    recv_particle_cap: int
    axis_name: str
    softening_kernel: Optional[str]
    interpret: bool
    with_force_scale: bool


def _table(positions_sorted, masses_sorted, imported):
    """``[local; imported]`` positions and masses: the production table, op for op."""
    pos = jnp.asarray(positions_sorted)
    dtype = pos.dtype
    parts = jnp.asarray(imported)
    table_pos = jnp.concatenate([pos, parts[:, :3].astype(dtype)])
    table_mass = jnp.concatenate(
        [jnp.asarray(masses_sorted, dtype), parts[:, 3].astype(dtype)]
    )
    return table_pos, table_mass


def _cross_near_forward(
    positions_sorted,
    masses_sorted,
    imported,
    leaf_start,
    leaf_count,
    neighbors,
    offsets,
    counts,
    softening_sq,
    G,
    static,
):
    from jaccpot.pallas.nearfield_leafpair_csr import (
        nearfield_leafpair_csr_sorted_direct_pallas,
    )

    n = int(jnp.shape(positions_sorted)[0])
    table_pos, table_mass = _table(positions_sorted, masses_sorted, imported)
    acc, fourth = nearfield_leafpair_csr_sorted_direct_pallas(
        table_pos,
        table_mass,
        leaf_start,
        leaf_count,
        neighbors,
        offsets,
        counts,
        leaf_width=static.leaf_width,
        softening_sq=softening_sq,
        G=G,
        chunk=static.chunk,
        interpret=static.interpret,
        with_potential=False,
        with_force_scale=static.with_force_scale,
        softening_kernel=static.softening_kernel,
        include_self=False,  # the local self term is already in the force
        num_target_leaves=static.num_target_leaves,
    )
    if static.with_force_scale:
        assert fourth is not None
        return acc[:n], fourth[:n]
    return acc[:n]


@functools.partial(jax.custom_vjp, nondiff_argnums=(13,))
def _cross_near_cvjp(
    positions_sorted,
    masses_sorted,
    imported,
    leaf_start,
    leaf_count,
    neighbors,
    offsets,
    counts,
    src,
    live_q,
    part_sizes,
    softening_sq,
    G,
    static,
):
    return _cross_near_forward(
        positions_sorted,
        masses_sorted,
        imported,
        leaf_start,
        leaf_count,
        neighbors,
        offsets,
        counts,
        softening_sq,
        G,
        static,
    )


def _cross_near_cvjp_fwd(
    positions_sorted,
    masses_sorted,
    imported,
    leaf_start,
    leaf_count,
    neighbors,
    offsets,
    counts,
    src,
    live_q,
    part_sizes,
    softening_sq,
    G,
    static,
):
    out = _cross_near_forward(
        positions_sorted,
        masses_sorted,
        imported,
        leaf_start,
        leaf_count,
        neighbors,
        offsets,
        counts,
        softening_sq,
        G,
        static,
    )
    # the operands only: the table is rebuilt in the reverse
    return out, (
        positions_sorted,
        masses_sorted,
        imported,
        leaf_start,
        leaf_count,
        neighbors,
        offsets,
        counts,
        src,
        live_q,
        part_sizes,
        jnp.asarray(softening_sq),
        jnp.asarray(G),
    )


def _cross_near_cvjp_bwd(static, residual, cotangent):
    from jaccpot.pallas.nearfield_leafpair_csr_direct_vjp import (
        nearfield_leafpair_csr_sorted_direct_reverse_pallas,
    )

    (
        positions_sorted,
        masses_sorted,
        imported,
        leaf_start,
        leaf_count,
        neighbors,
        offsets,
        counts,
        src,
        live_q,
        part_sizes,
        softening_sq,
        G,
    ) = residual
    softening_sq, G = jnp.asarray(softening_sq), jnp.asarray(G)
    pos = jnp.asarray(positions_sorted)
    dtype = pos.dtype
    n = int(pos.shape[0])
    # the force-scale lane only sets the next walk's thresholds: dropped
    c_sorted = jnp.asarray(
        cotangent[0] if static.with_force_scale else cotangent, dtype
    )
    # my sinks' cotangents to the peers that imported them, along the forward's
    # layout: the receiver's row r is then the cotangent of its imported row r
    c_send = jnp.where(jnp.asarray(live_q)[:, None], c_sorted[jnp.asarray(src)], 0.0)
    c_imp, _, _ = ragged_all_to_all_exchange(
        c_send,
        jnp.asarray(part_sizes),
        output_capacity=static.recv_particle_cap,
        axis_name=static.axis_name,
    )
    table_pos, table_mass = _table(pos, masses_sorted, imported)
    cot = jnp.concatenate([c_sorted, c_imp.astype(dtype)])
    pos_bar, mass_bar, soft_bar, g_bar = (
        nearfield_leafpair_csr_sorted_direct_reverse_pallas(
            table_pos,
            table_mass,
            leaf_start,
            leaf_count,
            neighbors,
            offsets,
            counts,
            cot,
            leaf_width=static.leaf_width,
            softening_sq=softening_sq,
            G=G,
            chunk=static.chunk,
            softening_kernel=static.softening_kernel,
            interpret=static.interpret,
            include_self=False,
            num_target_leaves=static.num_target_leaves,
        )
    )
    return (
        pos_bar[:n].astype(pos.dtype),
        mass_bar[:n].astype(jnp.asarray(masses_sorted).dtype),
        None,  # the imports: their owners write their cotangents
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        jnp.reshape(soft_bar.astype(softening_sq.dtype), jnp.shape(softening_sq)),
        jnp.reshape(g_bar.astype(G.dtype), jnp.shape(G)),
    )


_cross_near_cvjp.defvjp(_cross_near_cvjp_fwd, _cross_near_cvjp_bwd)


def cross_near_direct_cvjp(
    positions_sorted: Array,
    masses_sorted: Array,
    imported: Array,
    leaf_start: Array,
    leaf_count: Array,
    neighbors: Array,
    offsets: Array,
    counts: Array,
    src: Array,
    live_q: Array,
    part_sizes: Array,
    softening_sq: Array,
    G: Array,
    *,
    num_target_leaves: int,
    leaf_width: int,
    recv_particle_cap: int,
    axis_name: str = AXIS_NAME,
    chunk: int = 64,
    softening_kernel: Optional[str] = None,
    interpret: bool = False,
    with_force_scale: bool = False,
) -> Union[Array, tuple[Array, Array]]:
    """The cross near term on the sorted rows, under a ``custom_vjp`` (seam B).

    Forward: the production table and kernel call of
    :func:`~jaccpot.distributed.cross.cross_near_acceleration_direct` (its bits),
    returned on the ``n`` local SORTED rows (the production call then permutes them
    to input order; :func:`cross_near_acceleration_direct_cvjp` does both). Reverse:
    the sinks' cotangents sent along the forward's near layout, then the direct
    reverse over the local leaves (see the module docstring). Call it inside the
    ``shard_map`` of the mesh force, with ``axis_name`` that mesh's axis.

    Parameters
    ----------
    positions_sorted : Array
        ``(n, 3)`` this device's particles in tree order. Differentiable.
    masses_sorted : Array
        ``(n,)`` their masses. Differentiable.
    imported : Array
        ``(recv_particle_cap, 4)`` the near exchange's receive buffer (``x, y, z,
        m`` per imported particle; the sink's ``particles``). No cotangent: pass it
        ``stop_gradient``'d (its owners differentiate those particles).
    leaf_start, leaf_count : Array
        ``(L + R,)`` the table's leaf ranges (:func:`cross_near_direct_layout`).
    neighbors, offsets, counts : Array
        The cross CSR over the ``L`` local leaves (:func:`cross_near_direct_layout`);
        ``counts`` has ``L`` entries.
    src : Array
        ``(send_particle_cap,)`` the forward exchange's send rows: slot ``s`` sent
        sorted row ``src[s]`` (``_symmetric_exchange``'s near ``src``).
    live_q : Array
        ``(send_particle_cap,)`` bool, the slots that carried a particle.
    part_sizes : Array
        ``(ndev,)`` the forward exchange's send sizes per peer.
    softening_sq : Array
        Scalar squared softening (replicated). Differentiable (a per-device partial,
        see the module docstring).
    G : Array
        Scalar gravitational constant (replicated). Differentiable (a per-device
        partial).
    num_target_leaves : int
        ``L``. Static.
    leaf_width : int
        The target tiles' width: the production call's ``max(local width, the
        sink's leaf_width)``. Static.
    recv_particle_cap : int
        ``imported``'s rows: the forward exchange's ``output_capacity``. Static.
    axis_name : str
        The mesh axis of the exchange. Static.
    chunk : int
        The direct kernels' piece floor (production: 64). Static.
    softening_kernel : Optional[str]
        The pair kernel; ``None`` gives the default. Static.
    interpret : bool
        Pallas interpret mode, forward and reverse. Static.
    with_force_scale : bool
        Also return the force-scale lane (not differentiated). Static.

    Returns
    -------
    Union[Array, tuple[Array, Array]]
        ``(n, 3)`` cross acceleration of the sorted local rows; with
        ``with_force_scale``, ``(acceleration, force_scale (n,))``.
    """
    static = _CrossNearStatic(
        num_target_leaves=int(num_target_leaves),
        leaf_width=int(leaf_width),
        chunk=int(chunk),
        recv_particle_cap=int(recv_particle_cap),
        axis_name=str(axis_name),
        softening_kernel=softening_kernel,
        interpret=bool(interpret),
        with_force_scale=bool(with_force_scale),
    )
    if int(jnp.shape(imported)[0]) != static.recv_particle_cap:
        raise ValueError(
            f"imported has {int(jnp.shape(imported)[0])} rows, the receive capacity "
            f"is {static.recv_particle_cap}: the cotangent exchange would misalign"
        )
    return _cross_near_cvjp(
        positions_sorted,
        masses_sorted,
        imported,
        leaf_start,
        leaf_count,
        neighbors,
        offsets,
        counts,
        src,
        live_q,
        part_sizes,
        softening_sq,
        G,
        static,
    )


def cross_near_acceleration_direct_cvjp(
    near: dict,
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_ranges: Array,
    particle_indices: Array,
    num_internal: int,
    *,
    src: Array,
    live_q: Array,
    part_sizes: Array,
    axis_name: str = AXIS_NAME,
    leaf_width: int,
    softening_sq: Array,
    G: Array,
    chunk: int = 64,
    interpret: bool = False,
    softening_kernel: Optional[str] = None,
    with_force_scale: bool = False,
) -> Union[Array, tuple[Array, Array]]:
    """Differentiable twin of :func:`~jaccpot.distributed.cross.cross_near_acceleration_direct`.

    The production call's arguments plus the forward exchange's send layout: the
    layout (:func:`cross_near_direct_layout`), :func:`cross_near_direct_cvjp` on the
    sorted rows, and the production permutation to input order. Its value is the
    production call's, bit for bit; under ``jax.grad`` the reverse is seam B's. The
    imported particles are ``stop_gradient``'d here (they are constants of the
    reverse; the import must also come from a ``stop_gradient``'d exchange so that
    JAX never differentiates through it).

    Parameters
    ----------
    near : dict
        The near sink of the default exchange (the flat import and the pairs).
    positions_sorted : Array
        ``(n, 3)`` this device's particles in tree order. Differentiable.
    masses_sorted : Array
        ``(n,)`` their masses. Differentiable.
    leaf_ranges : Array
        ``(L, 2)`` inclusive particle range of every local leaf row.
    particle_indices : Array
        ``(n,)`` the input row of each sorted particle.
    num_internal : int
        Internal-node count of the local tree. Static.
    src : Array
        ``(send_particle_cap,)`` the near exchange's send rows.
    live_q : Array
        ``(send_particle_cap,)`` its live slots.
    part_sizes : Array
        ``(ndev,)`` its send sizes.
    axis_name : str
        The mesh axis. Static.
    leaf_width : int
        The local leaves' slot width. Static.
    softening_sq : Array
        Scalar, matching the local lane's. Differentiable.
    G : Array
        Scalar gravitational constant. Differentiable.
    chunk : int
        Entries per chunk. Static.
    interpret : bool
        Pallas interpret mode.
    softening_kernel : Optional[str]
        The pair kernel; ``None`` gives the default.
    with_force_scale : bool
        Also return the force-scale lane (not differentiated). Static.

    Returns
    -------
    Union[Array, tuple[Array, Array]]
        ``(n, 3)`` acceleration in INPUT row order to add to the local force; with
        ``with_force_scale``, ``(acceleration, force_scale (n,))``.
    """
    pos = jnp.asarray(positions_sorted)
    n = int(pos.shape[0])
    parts = lax.stop_gradient(jnp.asarray(near["particles"]))
    layout = cross_near_direct_layout(near, leaf_ranges, num_internal, n)
    out: Any = cross_near_direct_cvjp(
        pos,
        masses_sorted,
        parts,
        *layout,
        src,
        live_q,
        part_sizes,
        softening_sq,
        G,
        num_target_leaves=int(jnp.shape(leaf_ranges)[0]),
        leaf_width=max(int(leaf_width), int(near.get("leaf_width", 0))),
        recv_particle_cap=int(parts.shape[0]),
        axis_name=axis_name,
        chunk=chunk,
        softening_kernel=softening_kernel,
        interpret=interpret,
        with_force_scale=with_force_scale,
    )
    acc = out[0] if with_force_scale else out
    perm = jnp.asarray(particle_indices)
    acc_out = jnp.zeros((n, 3), acc.dtype).at[perm].set(acc)
    if not with_force_scale:
        return acc_out
    return acc_out, jnp.zeros((n,), out[1].dtype).at[perm].set(out[1])
