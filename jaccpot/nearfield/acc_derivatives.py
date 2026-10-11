"""Near-field spatial derivatives of the acceleration.

``max_acc_derivative_order > 0`` returns the tower ``d^k a / dx^k`` of the full
acceleration at each target. The far half comes from the local expansions
(:func:`jaccpot.operators.real_p2m_l2p.evaluate_local_real_derivative_tower`);
this module is the near half: the direct sum over the same near pairs the
acceleration uses (:func:`jaccpot.nearfield.near_field.compute_leaf_p2p_accelerations`
on the full set, the padded source matrix of the target route), differentiated
exactly. Until 2026-10 the tower was far-field only.

A pair contributes ``a = -G m g(r^2) d``, with ``d = x_target - x_source`` and
``g`` the softening kernel's force factor (:mod:`jaccpot.softening`). Its
derivatives with respect to the target are derivatives with respect to ``d``:

* order 1 in closed form, ``-G m (g I + h d d^T)`` with ``h = (1/r) dg/dr``;
* order ``k >= 2`` by nested forward mode along the axes of each multi-index,
  through the same :func:`~jaccpot.softening.masked_pair_factors` the force uses.

The self pair is excluded. Its force is zero, but its Jacobian is
``-G m g(0) I``, not zero.

The layout matches the far tower. Level ``k`` is ``(N, 3, C_k)`` with
``[n, i, c] = d^{alpha_c} a_i``, where ``alpha_c`` is the ``c``-th entry of
:func:`~jaccpot.operators.symmetric_tensors.symmetric_multi_indices_3d`.

This module is forward-only and kept apart from the acceleration kernels, for two
reasons: their custom VJPs block forward mode, and their op sequence is pinned.
It costs one more pass over the near pairs, and only when the tower is requested.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, Bool, Float, Int, jaxtyped
from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.interactions import NodeNeighborList
from yggdrax.tree import Tree

from jaccpot.operators.symmetric_tensors import symmetric_multi_indices_3d
from jaccpot.softening import (
    masked_pair_factors,
    resolve_softening_kernel,
    softening_params_from_sq,
)

from ._scatter import _scatter_contributions
from ._schedules import _prepare_leaf_data, prepare_leaf_neighbor_pairs

__all__ = [
    "compute_leaf_p2p_acc_derivatives",
    "compute_targeted_p2p_acc_derivatives",
    "pair_acc_derivative_tower",
]


def _along(fun: Callable[[Array], Array], tangent: Array) -> Callable[[Array], Array]:
    """The directional derivative of ``fun`` along ``tangent``, as a function.

    Parameters
    ----------
    fun : Callable[[Array], Array]
        The function to differentiate, of the separations.
    tangent : Array
        The direction, shaped like ``fun``'s argument.

    Returns
    -------
    Callable[[Array], Array]
        ``d -> jvp(fun, d, tangent)``.
    """

    def derivative(d: Array) -> Array:
        return jax.jvp(fun, (d,), (tangent,))[1]

    return derivative


@jaxtyped(typechecker=beartype)
def pair_acc_derivative_tower(
    diff: Float[Array, "*batch s 3"],
    src_mass: Float[Array, "*batch s"],
    active: Bool[Array, "*batch s"],
    *,
    G: Union[float, Array],
    params: Any,
    kernel: Optional[str],
    max_order: int,
) -> tuple[Array, ...]:
    """Spatial derivatives of ``a = -G sum_s m_s g(|d_s|^2) d_s`` up to ``max_order``.

    ``d`` is target minus source, so a derivative with respect to the target is
    one with respect to ``d``. Order 1 is the closed form
    ``-G sum m (g I + h d d^T)``. Higher orders use nested ``jax.jvp`` of the
    per-pair force along the axes of each multi-index. That is exact, and works
    for every kernel and order.

    The C2 Wendland kernel is the one exception. Its factor goes through
    ``sqrt(r2)``, which has no forward-mode derivative at ``r2 = 0``. So at order
    2 and above, a coincident pair of distinct particles is left out. Its
    second derivative is zero there; the third and higher do not exist for a C2
    kernel. The other kernels are smooth in ``r2`` and need no exception.

    Parameters
    ----------
    diff : Float[Array, '*batch s 3']
        Target-minus-source separations.
    src_mass : Float[Array, '*batch s']
        Source masses, broadcast to ``diff``'s leading shape.
    active : Bool[Array, '*batch s']
        Live pairs. It must exclude the target itself and padding; inactive pairs
        contribute exactly zero.
    G : Union[float, Array]
        Gravitational constant.
    params : Any
        From :func:`~jaccpot.softening.softening_params_from_sq`.
    kernel : Optional[str]
        The pair kernel (static); ``None`` gives the default.
    max_order : int
        Highest derivative order, at least 1.

    Returns
    -------
    tuple[Array, ...]
        ``max_order`` levels; level ``k`` has shape ``(*batch, 3, C_k)``.

    Raises
    ------
    ValueError
        If ``max_order`` is below 1.
    """
    order = int(max_order)
    if order < 1:
        raise ValueError("max_order must be at least 1")
    name = resolve_softening_kernel(kernel)
    dtype = diff.dtype
    g_const = jnp.asarray(G, dtype=dtype)
    r2 = jnp.sum(diff * diff, axis=-1)
    g, _, h = masked_pair_factors(r2, active, params, name, derivative=True)
    jac = jnp.einsum(
        "...s,...si,...sj->...ij",
        src_mass * h,
        diff,
        diff,
        precision=lax.Precision.HIGHEST,
    )
    jac = jac + jnp.sum(src_mass * g, axis=-1)[..., None, None] * jnp.eye(
        3, dtype=dtype
    )
    levels = [-g_const * jac]
    if order == 1:
        return tuple(levels)

    smooth = active & (r2 > 0.0) if name == "wendland_c2" else active
    weight = jnp.where(smooth, src_mass, 0.0)[..., None]

    def pair_force(d: Array) -> Array:
        factor = masked_pair_factors(jnp.sum(d * d, axis=-1), smooth, params, name)[0]
        return factor[..., None] * d

    axes = [jnp.broadcast_to(row, diff.shape) for row in jnp.eye(3, dtype=dtype)]
    for k in range(2, order + 1):
        columns = []
        for alpha in symmetric_multi_indices_3d(k):
            derivative = pair_force
            for axis, count in enumerate(alpha):
                for _ in range(count):
                    derivative = _along(derivative, axes[axis])
            columns.append(jnp.sum(weight * derivative(diff), axis=-2))
        levels.append(-g_const * jnp.stack(columns, axis=-1))
    return tuple(levels)


@partial(
    jax.jit,
    static_argnames=("softening_kernel", "max_leaf_size", "max_order", "chunk_size"),
)
def _leaf_p2p_acc_derivatives_impl(
    node_ranges: Array,
    leaf_nodes: Array,
    offsets: Array,
    neighbors: Array,
    positions: Array,
    masses: Array,
    g_const: Array,
    softening_sq: Array,
    *,
    softening_kernel: Optional[str],
    max_leaf_size: int,
    max_order: int,
    chunk_size: int,
) -> tuple[Array, ...]:
    """Jitted body of :func:`compute_leaf_p2p_acc_derivatives`.

    Parameters
    ----------
    node_ranges : Array
        Per-node particle spans.
    leaf_nodes : Array
        Node index of each leaf.
    offsets : Array
        CSR offsets into ``neighbors``.
    neighbors : Array
        Flat neighbour-node list.
    positions : Array
        Sorted particle positions.
    masses : Array
        Sorted particle masses.
    g_const : Array
        Gravitational constant.
    softening_sq : Array
        Squared Plummer-equivalent softening.
    softening_kernel : Optional[str]
        The pair kernel (static).
    max_leaf_size : int
        Leaf width (static).
    max_order : int
        Highest derivative order (static).
    chunk_size : int
        Leaf pairs per scanned block (static).

    Returns
    -------
    tuple[Array, ...]
        The levels, ``(N, 3, C_k)`` each.
    """
    dtype = positions.dtype
    width = int(max_leaf_size)
    target_leaf_ids, source_leaf_ids, valid_pairs = prepare_leaf_neighbor_pairs(
        node_ranges, leaf_nodes, offsets, neighbors, sort_by_source=False
    )
    leaf_positions, leaf_masses, leaf_mask, leaf_particle_idx = _prepare_leaf_data(
        node_ranges, leaf_nodes, positions, masses, max_leaf_size=width
    )

    # Every leaf against itself, then every neighbour edge. The neighbour list
    # never holds the leaf itself; the acceleration would double-count if it did.
    own = jnp.arange(leaf_nodes.shape[0], dtype=INDEX_DTYPE)
    edge_count = int(target_leaf_ids.shape[0])
    tgt = jnp.concatenate([own, jnp.asarray(target_leaf_ids, dtype=INDEX_DTYPE)])
    src = jnp.concatenate([own, jnp.asarray(source_leaf_ids, dtype=INDEX_DTYPE)])
    valid = jnp.concatenate([jnp.ones_like(own, dtype=bool), valid_pairs])
    self_edge = jnp.concatenate(
        [jnp.ones_like(own, dtype=bool), jnp.zeros((edge_count,), dtype=bool)]
    )
    total = int(tgt.shape[0])
    chunk = max(1, min(int(chunk_size), total))
    pad = (-total) % chunk

    def chunked(values: Array, fill: Any) -> Array:
        padded = jnp.pad(values, (0, pad), constant_values=fill)
        return padded.reshape(-1, chunk)

    params = softening_params_from_sq(softening_kernel, softening_sq, dtype)
    not_same_slot = ~jnp.eye(width, dtype=bool)
    level_shapes = [
        (3, len(symmetric_multi_indices_3d(k))) for k in range(1, max_order + 1)
    ]
    flat_width = sum(rows * cols for rows, cols in level_shapes)

    def body(
        acc: Array, block: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, None]:
        t, s, ok, own_leaf = block
        target_mask = leaf_mask[t] & ok[:, None]
        diff = leaf_positions[t][:, :, None, :] - leaf_positions[s][:, None, :, :]
        active = (
            target_mask[:, :, None]
            & leaf_mask[s][:, None, :]
            & (not_same_slot[None] | ~own_leaf[:, None, None])
        )
        src_mass = jnp.broadcast_to(leaf_masses[s][:, None, :], active.shape)
        levels = pair_acc_derivative_tower(
            diff,
            src_mass,
            active,
            G=g_const,
            params=params,
            kernel=softening_kernel,
            max_order=max_order,
        )
        flat = jnp.concatenate(
            [level.reshape(level.shape[:2] + (-1,)) for level in levels], axis=-1
        )
        acc = _scatter_contributions(acc, leaf_particle_idx[t], flat, target_mask)
        return acc, None

    acc, _ = lax.scan(
        body,
        jnp.zeros((positions.shape[0], flat_width), dtype=dtype),
        (
            chunked(tgt, 0),
            chunked(src, 0),
            chunked(valid, False),
            chunked(self_edge, False),
        ),
    )
    out = []
    start = 0
    for rows, cols in level_shapes:
        out.append(acc[:, start : start + rows * cols].reshape(-1, rows, cols))
        start += rows * cols
    return tuple(out)


@jaxtyped(typechecker=beartype)
def compute_leaf_p2p_acc_derivatives(
    tree: Tree,
    neighbor_list: NodeNeighborList,
    positions_sorted: Float[Array, "n 3"],
    masses_sorted: Float[Array, "n"],
    *,
    G: Union[float, Array] = 1.0,
    softening: float = 0.0,
    softening_kernel: Optional[str] = None,
    max_leaf_size: int,
    max_derivative_order: int,
    edge_chunk_size: int = 256,
) -> tuple[Array, ...]:
    """Near-field acceleration derivatives for every particle, in sorted order.

    These are the same pairs as :func:`~jaccpot.nearfield.near_field.compute_leaf_p2p_accelerations`:
    each leaf against itself, without the self pair, and against every leaf in
    its neighbour list. The pairs are re-derived from ``neighbor_list`` in
    neighbour order; the near field's precomputed buffers hold the same edges.
    Static shapes throughout, so this works under ``jit``. Blocks of
    ``edge_chunk_size`` leaf pairs are scanned, and each scatters its
    ``(block, W, 3 * sum C_k)`` result into the particle accumulator.

    Parameters
    ----------
    tree : Tree
        Radix tree whose ``node_ranges`` map leaves to particle spans.
    neighbor_list : NodeNeighborList
        The leaf-neighbour CSR the near field uses.
    positions_sorted : Float[Array, 'n 3']
        Particle positions in Morton order.
    masses_sorted : Float[Array, 'n']
        Particle masses, same order.
    G : Union[float, Array]
        Gravitational constant.
    softening : float
        Plummer-equivalent softening length; a concrete float.
    softening_kernel : Optional[str]
        The pair kernel; ``None`` gives the default.
    max_leaf_size : int
        Static leaf width, as for the acceleration.
    max_derivative_order : int
        Highest derivative order, at least 1.
    edge_chunk_size : int
        Leaf pairs per scanned block.

    Returns
    -------
    tuple[Array, ...]
        ``max_derivative_order`` levels; level ``k`` has shape ``(N, 3, C_k)``,
        in Morton order.

    Raises
    ------
    ValueError
        If ``max_derivative_order`` is below 1.
    """
    order = int(max_derivative_order)
    if order < 1:
        raise ValueError("max_derivative_order must be at least 1")
    positions = jnp.asarray(positions_sorted)
    dtype = positions.dtype
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    if leaf_nodes.size == 0:
        return tuple(
            jnp.zeros(
                (positions.shape[0], 3, len(symmetric_multi_indices_3d(k))),
                dtype=dtype,
            )
            for k in range(1, order + 1)
        )
    return _leaf_p2p_acc_derivatives_impl(
        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        leaf_nodes,
        jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE),
        jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE),
        positions,
        jnp.asarray(masses_sorted, dtype=dtype),
        jnp.asarray(G, dtype=dtype),
        jnp.asarray(float(softening) ** 2, dtype=dtype),
        softening_kernel=softening_kernel,
        max_leaf_size=int(max_leaf_size),
        max_order=order,
        chunk_size=int(edge_chunk_size),
    )


@jaxtyped(typechecker=beartype)
def compute_targeted_p2p_acc_derivatives(
    positions_sorted: Float[Array, "n 3"],
    masses_sorted: Float[Array, "n"],
    target_sorted_indices: Int[Array, "t"],
    source_indices: Int[Array, "t srcslots"],
    source_mask: Bool[Array, "t srcslots"],
    *,
    G: Union[float, Array],
    softening: float,
    softening_kernel: Optional[str] = None,
    max_derivative_order: int,
) -> tuple[Array, ...]:
    """Near-field acceleration derivatives for a target subset.

    This is the target route's twin of :func:`compute_leaf_p2p_acc_derivatives`.
    It uses the padded per-target source matrix the targeted acceleration sums
    over, which already excludes the target itself and duplicate sources.

    Parameters
    ----------
    positions_sorted : Float[Array, 'n 3']
        Morton-sorted particle positions; all particles are sources.
    masses_sorted : Float[Array, 'n']
        Morton-sorted masses.
    target_sorted_indices : Int[Array, 't']
        Target particle indices, in sorted order.
    source_indices : Int[Array, 't srcslots']
        Padded per-target source indices.
    source_mask : Bool[Array, 't srcslots']
        Validity of each source slot.
    G : Union[float, Array]
        Gravitational constant.
    softening : float
        Plummer-equivalent softening length; a concrete float.
    softening_kernel : Optional[str]
        The pair kernel; ``None`` gives the default.
    max_derivative_order : int
        Highest derivative order, at least 1.

    Returns
    -------
    tuple[Array, ...]
        ``max_derivative_order`` levels; level ``k`` has shape ``(T, 3, C_k)``.
    """
    dtype = positions_sorted.dtype
    if int(source_indices.shape[1]) == 0:
        return tuple(
            jnp.zeros(
                (target_sorted_indices.shape[0], 3, len(symmetric_multi_indices_3d(k))),
                dtype=dtype,
            )
            for k in range(1, int(max_derivative_order) + 1)
        )
    diff = (
        positions_sorted[target_sorted_indices][:, None, :]
        - positions_sorted[source_indices]
    )
    params = softening_params_from_sq(
        softening_kernel, jnp.asarray(float(softening) ** 2, dtype=dtype), dtype
    )
    return pair_acc_derivative_tower(
        diff,
        masses_sorted[source_indices],
        source_mask,
        G=G,
        params=params,
        kernel=softening_kernel,
        max_order=int(max_derivative_order),
    )
