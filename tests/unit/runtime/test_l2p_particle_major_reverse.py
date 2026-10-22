"""The reverse of the production (particle-major, Pallas) L2P.

``l2p_real_particles_reverse_pallas`` is the adjoint of
``l2p_real_particles_pallas``, the fused lane's far-field evaluation: it gives each
particle's position cotangent (the expansion's Hessian applied to the gradient's
cotangent) and reduces the coefficient and centre cotangents per leaf inside the
kernel, in the program that owns the leaf -- so no ``N x (p+1)^2`` per-particle
buffer exists and no atomics are needed. These tests check it against ``jax.vjp``
of a pure-JAX L2P (the same Cartesian recurrence, gathered per particle) in
interpret mode, on trees with leaves of every occupancy, leaves walked over many
chunks, empty padding leaves and dead rows past the leaves' particles.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.downward.local_expansions import LocalExpansionData
from jaccpot.pallas.l2p_real import (
    l2p_real_grad_cartesian,
    l2p_real_particles_reverse_pallas,
)
from jaccpot.runtime.dtypes import INDEX_DTYPE
from jaccpot.runtime.kernels._evaluate import (
    _evaluate_local_expansions_particle_major,
)
from tests.unit._jaxpr_buffers import largest_xla_array


def _tree(seed: int, n_live: int, n_rows: int, num_pad: int, width: int, order: int):
    """Leaves of 1..``width`` particles in order, ``num_pad`` empty ones at the end.

    Parameters
    ----------
    seed : int
        RNG seed.
    n_live : int
        Particles in the leaves (a prefix of the rows).
    n_rows : int
        Rows of the position array (``>= n_live``; the rest are dead).
    num_pad : int
        Empty padding leaves after the live ones.
    width : int
        Largest leaf occupancy.
    order : int
        Expansion order.

    Returns
    -------
    tuple
        ``(coefficients, centers, positions, leaf_nodes, leaf_start, leaf_count,
        node_ranges)`` in float64.
    """
    rng = np.random.default_rng(seed)
    counts = []
    left = n_live
    while left > 0:
        c = int(min(left, rng.integers(1, width + 1)))
        counts.append(c)
        left -= c
    num_leaves = len(counts) + num_pad
    num_internal = num_leaves - 1
    ranges = np.zeros((num_internal + num_leaves, 2), np.int64)
    start = 0
    for i, c in enumerate(counts):
        ranges[num_internal + i] = (start, start + c - 1)
        start += c
    for j in range(num_pad):
        ranges[num_internal + len(counts) + j] = (n_live, n_live - 1)
    num_nodes = num_internal + num_leaves
    coeff_count = (order + 1) ** 2
    centers = rng.normal(size=(num_nodes, 3)) * 0.3
    coefficients = rng.normal(size=(num_nodes, coeff_count))
    positions = rng.normal(size=(n_rows, 3)) * 0.3
    leaf_ranges = ranges[num_internal:]
    return (
        jnp.asarray(coefficients),
        jnp.asarray(centers),
        jnp.asarray(positions),
        jnp.arange(num_internal, num_nodes, dtype=INDEX_DTYPE),
        jnp.asarray(leaf_ranges[:, 0], INDEX_DTYPE),
        jnp.asarray(
            np.maximum(leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1, 0), INDEX_DTYPE
        ),
        jnp.asarray(ranges, INDEX_DTYPE),
    )


def _l2p_jax(coefficients, centers, positions, leaf_nodes, leaf_count, *, order):
    """Pure-JAX L2P: each particle's leaf expansion by the same Cartesian recurrence.

    Parameters
    ----------
    coefficients : Array
        ``[nodes, (p+1)^2]``.
    centers : Array
        ``[nodes, 3]``.
    positions : Array
        ``[N, 3]``.
    leaf_nodes : Array
        ``[L]`` node of each leaf, in particle order.
    leaf_count : Array
        ``[L]`` particles per leaf.
    order : int
        Expansion order.

    Returns
    -------
    Array
        ``[N, 3]`` gradient with respect to ``delta``, zero on dead rows.
    """
    n = positions.shape[0]
    leaf_of = jnp.repeat(
        jnp.arange(leaf_nodes.shape[0]), leaf_count, total_repeat_length=n
    )
    live = jnp.arange(n) < jnp.sum(leaf_count)
    node = leaf_nodes[leaf_of]
    delta = centers[node] - positions
    coeff = [coefficients[node, k] for k in range((order + 1) ** 2)]
    g = l2p_real_grad_cartesian(
        coeff, delta[:, 0], delta[:, 1], delta[:, 2], order=order
    )
    return jnp.where(live[:, None], jnp.stack(g, axis=1), 0.0)


def _rel_max(got, want) -> float:
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-300))


# (n_live, n_rows, padding leaves, widest leaf, leaves per program, lanes per leaf):
# cell-like leaves in programs of 4; leaves of up to 70 particles walked 8 lanes at a
# time (nine chunks) in programs of 2; single-particle-heavy leaves in one program;
# dead rows throughout
_CASES = [(187, 200, 2, 8, 4, 2), (300, 300, 1, 70, 2, 8), (150, 160, 0, 3, 64, 4)]


@pytest.mark.parametrize("order", [3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("dtype, tol", [(jnp.float64, 1e-12), (jnp.float32, 2e-5)])
def test_reverse_equals_the_vjp_of_the_jax_l2p(order, dtype, tol):
    """Coefficient, centre and position cotangents equal ``jax.vjp`` of the pure-JAX
    L2P to float rounding, on every case's tree, at orders 3..8 (production p6)."""
    for seed, (n_live, n_rows, pad, width, block, chunk) in enumerate(_CASES):
        coef, cent, pos, leaf_nodes, start, count, _ = _tree(
            seed, n_live, n_rows, pad, width, order
        )
        coef, cent, pos = (a.astype(dtype) for a in (coef, cent, pos))
        out, vjp = jax.vjp(
            lambda c, x, q: _l2p_jax(c, x, q, leaf_nodes, count, order=order),
            coef,
            cent,
            pos,
        )
        gbar = jnp.asarray(
            np.random.default_rng(seed + 7).normal(size=out.shape), dtype
        )
        want = vjp(gbar)
        got = l2p_real_particles_reverse_pallas(
            coef,
            cent,
            pos,
            leaf_nodes,
            start,
            count,
            gbar,
            order=order,
            block=block,
            chunk=chunk,
            interpret=True,
        )
        for name, g, w, primal in zip(
            ("coefficients", "centers", "positions"), got, want, (coef, cent, pos)
        ):
            assert g.shape == primal.shape and g.dtype == primal.dtype, name
            err = _rel_max(g, w)
            assert err < tol, f"{name} p{order} case {seed}: {err:.3e}"
        # the cotangents are zero where the forward reads nothing: internal nodes,
        # empty leaves, dead rows
        num_internal = int(leaf_nodes[0])
        assert not np.any(np.asarray(got[0])[:num_internal])
        assert not np.any(np.asarray(got[1])[:num_internal])
        assert not np.any(np.asarray(got[2])[n_live:])
        if pad:
            dead_leaves = np.asarray(leaf_nodes)[-pad:]
            assert not np.any(np.asarray(got[0])[dead_leaves])


def test_reverse_equals_the_vjp_of_the_autodiff_l2p():
    """Against the spherical-angle autodiff form the production ``xla`` kernel runs
    (another function for the same polynomial): float64 rounding, p5."""
    order = 5
    coef, cent, pos, leaf_nodes, start, count, ranges = _tree(3, 187, 200, 2, 8, order)
    local = LocalExpansionData(order=order, centers=cent, coefficients=coef)
    kw = dict(leaf_nodes=leaf_nodes, node_ranges=ranges, order=order, chunk=64)

    def xla(c, x, q):
        return _evaluate_local_expansions_particle_major(
            LocalExpansionData(order=order, centers=x, coefficients=c), q, **kw
        )

    out, vjp = jax.vjp(xla, local.coefficients, local.centers, pos)
    gbar = jnp.asarray(np.random.default_rng(5).normal(size=out.shape))
    want = vjp(gbar)
    got = l2p_real_particles_reverse_pallas(
        coef, cent, pos, leaf_nodes, start, count, gbar, order=order, interpret=True
    )
    for name, g, w in zip(("coefficients", "centers", "positions"), got, want):
        assert _rel_max(g, w) < 1e-10, name


def test_the_reverse_holds_no_per_particle_coefficient_buffer():
    """No ``N x (p+1)^2`` array exists outside the kernel: the coefficient
    cotangents are reduced per leaf inside it (at 1e8 in float32 that buffer would
    be 19.6 GB at p6). The largest arrays are the outputs, ``[nodes, (p+1)^2]`` and
    ``[N, 3]``."""
    order = 6
    coef, cent, pos, leaf_nodes, start, count, _ = _tree(0, 300, 300, 1, 70, order)
    num_nodes, C = coef.shape
    n = pos.shape[0]
    assert num_nodes < n  # an N x C buffer would be the largest array
    largest = largest_xla_array(
        lambda c, x, q, g: l2p_real_particles_reverse_pallas(
            c, x, q, leaf_nodes, start, count, g, order=order, interpret=True
        ),
        coef,
        cent,
        pos,
        pos,
    )
    assert largest <= max(num_nodes * C, 3 * n), largest
