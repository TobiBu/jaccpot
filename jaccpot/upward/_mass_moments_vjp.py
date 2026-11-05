"""The tree's mass moments with a gather-only reverse, for the differentiable seam.

yggdrax's :func:`~yggdrax.tree_moments.compute_tree_mass_moments` takes every node's
mass and centre of mass as differences of two float64 prefix sums over the sorted
particles (``prefix[end] - prefix[start]``). Its autodiff reverse is the transpose
of that: two scatter-adds of the node cotangents into ``[N + 1]`` float64 rows --
many nodes share a start, so the adds collide (atomics, an order that varies from
run to run) -- then a reverse cumulative sum. On the large-N gradient at 1e8 those
``f64[N + 1, 3]`` buffers (2.4 GB each) were the largest temporaries, and at 8e6 the
scatters took 16 of the gradient's 175 ms and made the mass gradient differ between
two identical calls (about 1000 of 8e6 elements on a shared A100).

:func:`tree_mass_moments` is the same forward under a ``custom_vjp`` whose reverse
uses the tree instead: a particle is in a node exactly when the node is its leaf or
an ancestor of it (radix node ranges nest), so each particle's cotangent is the sum
of the node cotangents along its leaf's path to the root -- pointer doubling over
the parent array (seven gathers of the ``[nodes, 4]`` sums cover a depth of 128),
then one gather per particle. No scatter, no ``N``-sized float64 buffer, the same
sums in every run.

In forward mode (``ad_mode="forward"``) :func:`tree_mass_moments` is a
``custom_jvp`` instead (:func:`_mass_moments_forward`): the primal is yggdrax's call,
bitwise, and the tangent a tangent op that takes the ``K`` directions one at a time
(``lax.map``). JAX's own jvp of the prefix sums materialised ``f64[K, N, 3]``
weighted-position tangents and their prefix sums -- 24 to 48 B per particle per
direction (2e6 jacfwd K = 8: a 366 MiB ``f64[8, 2e6, 3]`` in ``fmm_upward``); one
direction at a time they cost the same as at ``K = 1``.

The reverse is OFF unless :func:`gather_reverse_mass_moments` is active: the forward is
yggdrax's call either way (the same bits), but the ``custom_vjp`` boundary would
move constants in the production programs' text. The large-N seam
(:func:`~jaccpot.runtime._large_n_grad.large_n_farfield_locals_at`) turns it on: its
static-radix trees order their leaves (the last nodes) as the particles.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jax import lax
from yggdrax.tree_moments import (
    TreeMassMoments,
    _accumulation_dtype,
    _prefix_range_sums,
    compute_tree_mass_moments,
)

from jaccpot.pallas._tangent import (
    drop_tangent_axis,
    make_tangent_op,
    with_tangent_axis,
)

__all__ = ["gather_reverse_mass_moments", "tree_mass_moments"]

#: doubling rounds of the ancestor sums: 2^7 = 128 levels, beyond a radix tree's
#: depth over 63-bit Morton codes plus its cell levels
_DOUBLING_ROUNDS = 7

_GATHER_REVERSE: ContextVar[bool] = ContextVar(
    "jaccpot_gather_reverse_mass_moments", default=False
)


@contextmanager
def gather_reverse_mass_moments() -> Iterator[None]:
    """Within the block, :func:`tree_mass_moments` traces the gather-only reverse.

    Read at trace time, like the gradient path's other options
    (:func:`~jaccpot.runtime.grad_options.grad_option_overrides`).

    Yields
    ------
    None
        The block runs with the reverse on.
    """
    token = _GATHER_REVERSE.set(True)
    try:
        yield
    finally:
        _GATHER_REVERSE.reset(token)


def tree_mass_moments(
    tree: Any, positions_sorted: Array, masses_sorted: Array
) -> TreeMassMoments:
    """:func:`~yggdrax.tree_moments.compute_tree_mass_moments`, reverse by ancestors.

    Parameters
    ----------
    tree : Any
        The radix tree (a pytree): ``parent``, ``left_child`` and ``node_ranges``;
        its leaves, the last nodes, in the particles' order.
    positions_sorted : Array
        ``[N, 3]`` positions in tree order.
    masses_sorted : Array
        ``[N]`` masses, same order.

    Returns
    -------
    TreeMassMoments
        yggdrax's, bitwise; under :func:`gather_reverse_mass_moments` its reverse
        is the gather-only one.
    """
    from jaccpot.runtime.grad_options import forward_mode_enabled

    if forward_mode_enabled():
        return _mass_moments_forward(tree, positions_sorted, masses_sorted)
    if not _GATHER_REVERSE.get():
        return compute_tree_mass_moments(tree, positions_sorted, masses_sorted)
    return _mass_moments_cvjp(tree, positions_sorted, masses_sorted)


@jax.custom_vjp
def _mass_moments_cvjp(
    tree: Any, positions_sorted: Array, masses_sorted: Array
) -> TreeMassMoments:
    return compute_tree_mass_moments(tree, positions_sorted, masses_sorted)


def _mass_moments_fwd(tree, positions_sorted, masses_sorted):
    out = compute_tree_mass_moments(tree, positions_sorted, masses_sorted)
    return out, (tree, positions_sorted, masses_sorted, out)


def _mass_moments_bwd(residual, cotangent):
    tree, positions, masses, out = residual
    acc = jnp.float64
    mass = out.mass.astype(acc)
    com = out.center_of_mass.astype(acc)
    g_mass = jnp.asarray(cotangent.mass).astype(acc)
    g_com = jnp.asarray(cotangent.center_of_mass).astype(acc)
    # com = moment / mass where mass > 0 (else 0): the node cotangents of the
    # moment sum and of the mass sum
    live = mass > 0
    inv = jnp.where(live, 1.0 / jnp.where(live, mass, 1.0), 0.0)
    moment_bar = g_com * inv[:, None]
    mass_bar = g_mass - jnp.sum(moment_bar * com, axis=1)
    # sums along each node's path to the root, by pointer doubling: after round k
    # a node holds the sum over itself and its 2^k - 1 nearest ancestors
    sums = jnp.concatenate([mass_bar[:, None], moment_bar], axis=1)
    hop = jnp.asarray(tree.parent).astype(jnp.int32)
    for _ in range(_DOUBLING_ROUNDS):
        up = hop >= 0
        safe = jnp.where(up, hop, 0)
        sums = sums + jnp.where(up[:, None], sums[safe], 0.0)
        hop = jnp.where(up, hop[safe], -1)
    # every particle reads its leaf's path sum (leaves: the last nodes, in order),
    # in the working dtype: N-sized float64 rows were this reverse's largest
    # buffers (3.2 GB for the gathered sums at 1e8)
    dtype = positions.dtype
    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    ranges = jnp.asarray(tree.node_ranges)[num_internal:]
    counts = jnp.maximum(ranges[:, 1] - ranges[:, 0] + 1, 0)
    n = int(positions.shape[0])
    leaf_of = jnp.repeat(
        jnp.arange(counts.shape[0], dtype=jnp.int32), counts, total_repeat_length=n
    )
    inside = jnp.arange(n) < jnp.sum(counts)
    path = jnp.where(
        inside[:, None],
        sums.astype(dtype)[num_internal + leaf_of],
        jnp.zeros((), dtype),
    )
    masses_bar = path[:, 0] + jnp.sum(path[:, 1:] * positions, axis=1)
    positions_bar = masses.astype(dtype)[:, None] * path[:, 1:]
    return (
        None,
        positions_bar.astype(positions.dtype),
        masses_bar.astype(masses.dtype),
    )


_mass_moments_cvjp.defvjp(_mass_moments_fwd, _mass_moments_bwd)


# --------------------------------------------------------------------------
# Forward mode: the tangent one direction at a time.
# --------------------------------------------------------------------------


def _mass_moments_tangent_impl(
    positions: Array,
    masses: Array,
    node_ranges: Array,
    t_positions: Array,
    t_masses: Array,
    *,
    live: tuple[bool, ...],
) -> tuple[Array, Array]:
    """``[K, nodes]`` mass and ``[K, nodes, 3]`` centre tangents, one ``k`` at a time.

    The same wide accumulation as the primal (:func:`_prefix_range_sums` in
    yggdrax's accumulation dtype); ``com' = (moment' - com * mass') / mass`` per
    node, zero for an empty node.

    Parameters
    ----------
    positions : Array
        ``[N, 3]`` sorted positions.
    masses : Array
        ``[N]`` sorted masses.
    node_ranges : Array
        ``[nodes, 2]`` inclusive particle ranges.
    t_positions : Array
        ``[K, N, 3]`` position tangents.
    t_masses : Array
        ``[K, N]`` mass tangents.
    live : tuple[bool, ...]
        Static: which of the two tangents are nonzero.

    Returns
    -------
    tuple[Array, Array]
        ``([K, nodes] mass tangents, [K, nodes, 3] centre tangents)``, in the
        primals' dtypes.
    """
    acc = _accumulation_dtype(positions.dtype)
    ranges = jnp.asarray(node_ranges).astype(jnp.int32)
    starts = ranges[:, 0]
    ends = ranges[:, 1] + 1
    m_acc = masses.astype(acc)
    total = _prefix_range_sums(m_acc, starts, ends)
    empty = total == 0
    safe = jnp.where(empty, jnp.ones_like(total), total)
    com = _prefix_range_sums(m_acc[:, None] * positions.astype(acc), starts, ends)
    com = com / safe[:, None]
    live_x, live_m = bool(live[0]), bool(live[1])
    nodes = int(ranges.shape[0])

    def one(tangent: tuple[Array, Array]) -> tuple[Array, Array]:
        tx, tm = tangent
        d_moment = jnp.zeros((nodes, 3), acc)
        d_mass = jnp.zeros((nodes,), acc)
        if live_x:
            d_moment = d_moment + _prefix_range_sums(
                m_acc[:, None] * tx.astype(acc), starts, ends
            )
        if live_m:
            tm_acc = tm.astype(acc)
            d_mass = _prefix_range_sums(tm_acc, starts, ends)
            d_moment = d_moment + _prefix_range_sums(
                tm_acc[:, None] * positions.astype(acc), starts, ends
            )
        d_com = (d_moment - com * d_mass[:, None]) / safe[:, None]
        d_com = jnp.where(empty[:, None], jnp.zeros((), acc), d_com)
        return d_mass.astype(masses.dtype), d_com.astype(positions.dtype)

    return lax.map(one, (t_positions, t_masses))


#: The tangent op of :func:`_mass_moments_forward` (a ``jnp`` impl: the prefix sums
#: are memory-bound scans, and ``lax.map`` keeps one direction's ``N``-sized float64
#: rows alive at a time).
mass_moments_tangent = make_tangent_op(
    "mass_moments", _mass_moments_tangent_impl, num_primals=3
)


@jax.custom_jvp
def _mass_moments_cjvp(
    tree: Any, positions_sorted: Array, masses_sorted: Array
) -> tuple[Array, Array]:
    out = compute_tree_mass_moments(tree, positions_sorted, masses_sorted)
    return out.mass, out.center_of_mass


def _mass_moments_jvp(primals: tuple, tangents: tuple) -> tuple[Any, Any]:
    tree, positions, masses = primals
    _, t_positions, t_masses = tangents
    out = _mass_moments_cjvp(tree, positions, masses)
    tk, live = with_tangent_axis([t_positions, t_masses], [positions, masses])
    t_mass, t_com = drop_tangent_axis(
        mass_moments_tangent(
            positions, masses, jnp.asarray(tree.node_ranges), *tk, live=live
        )
    )
    return out, (t_mass, t_com)


_mass_moments_cjvp.defjvp(_mass_moments_jvp)


def _mass_moments_forward(
    tree: Any, positions_sorted: Array, masses_sorted: Array
) -> TreeMassMoments:
    """Forward-mode :func:`tree_mass_moments`: yggdrax's primal, a one-at-a-time tangent.

    Parameters
    ----------
    tree : Any
        The radix tree (``node_ranges`` in the particles' order).
    positions_sorted : Array
        ``[N, 3]`` positions in tree order.
    masses_sorted : Array
        ``[N]`` masses, same order.

    Returns
    -------
    TreeMassMoments
        yggdrax's, bitwise.
    """
    mass, com = _mass_moments_cjvp(tree, positions_sorted, masses_sorted)
    return TreeMassMoments(mass=mass, center_of_mass=com)
