"""The direct CSR near field's reverse (the production layout) vs autodiff and differences.

:func:`nearfield_leafpair_csr_sorted_direct_reverse_pallas` is the production near-field
kernel's reverse on the same sorted layout. Pinned here, for every softening kernel,
in float64 on the Pallas interpreter (and natively on an Ampere card):

* its cotangents (positions, masses, ``softening_sq``, ``G``) equal ``jax.vjp`` of a
  pure-JAX pair sum over the same pairs (rows + self, diagonal excluded) to float64
  round-off -- on leaves of every occupancy, empty padding leaves, rows past every
  leaf's particles, an empty row, rows of consecutive leaves (the merged source runs)
  and rows longer than the row limit (the reverse's in-place pieces);
* with no neighbour entries at all (the self terms alone), an empty ``neighbors``
  array included, the same.

The neighbour CSR is built SYMMETRIC, as the walk emits it: the reverse takes each
source-side cotangent from the other end of the pair.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.m2l_real_csr import pallas_m2l_real_csr_supported
from jaccpot.pallas.nearfield_leafpair_csr import (
    nearfield_leafpair_csr_sorted_direct_pallas,
)
from jaccpot.pallas.nearfield_leafpair_csr_direct_vjp import (
    nearfield_leafpair_csr_sorted_direct_reverse_pallas,
)
from jaccpot.softening import pair_factors, softening_params_from_sq, support_factor

KERNELS = ("plummer", "ferrers3", "wendland_c2")
SOFTENING = 0.5  # a compact support of 1.23 (ferrers3) / 1.5 (wendland_c2)


def _native_or_skip(interpret: bool) -> None:
    """Native Triton lowering needs an Ampere+ GPU; interpret mode runs anywhere."""
    if not interpret and not pallas_m2l_real_csr_supported():
        pytest.skip("native Pallas GPU lowering not available here")


def _symmetric_case(
    seed,
    *,
    num_live=10,
    num_pad=2,
    W=8,
    n_dead=3,
    p_edge=0.45,
    consecutive=False,
    dtype=np.float64,
):
    """Sorted particles in leaf ranges and a SYMMETRIC leaf CSR over the live leaves.

    Leaf occupancies 1..W (leaf 0 full, leaf 1 a single particle), ``num_pad`` empty
    padding leaves starting past the live rows, ``n_dead`` rows past every leaf's
    particles, and leaf 3's row empty. Rows are sorted, so neighbouring entries are
    often adjacent particle ranges (the kernel's merged runs); ``consecutive`` lists
    each leaf's three nearest leaves on either side, so they always are.
    """
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, W + 1, size=num_live)
    counts[0], counts[1] = W, 1
    counts = np.concatenate([counts, np.zeros(num_pad, np.int64)])
    L = counts.size
    n_live = int(counts.sum())
    n = n_live + n_dead
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    starts[num_live:] = n_live
    if consecutive:
        gap = np.abs(np.arange(num_live)[:, None] - np.arange(num_live)[None, :])
        adj = gap <= 3
    else:
        adj = rng.random((num_live, num_live)) < p_edge
    adj = np.triu(adj, 1)
    adj = adj | adj.T
    adj[3, :] = adj[:, 3] = False
    rows = [np.flatnonzero(adj[i]) for i in range(num_live)]
    rows += [np.zeros(0, np.int64) for _ in range(num_pad)]
    row_counts = np.array([r.size for r in rows])
    nbr = np.concatenate(rows + [np.zeros(4, np.int64)]).astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum(row_counts)]).astype(np.int32)
    pos = rng.standard_normal((n, 3))
    mass = np.abs(rng.standard_normal(n)) + 0.1
    # the (target, source) pairs the kernel sums: rows + self, diagonal excluded
    pi, pj = [], []
    for t in range(L):
        for s in list(rows[t]) + [t]:
            for a in range(starts[t], starts[t] + counts[t]):
                for b in range(starts[s], starts[s] + counts[s]):
                    if a != b:
                        pi.append(a)
                        pj.append(b)
    return dict(
        pos=jnp.asarray(pos.astype(dtype)),
        mass=jnp.asarray(mass.astype(dtype)),
        starts=jnp.asarray(starts, jnp.int32),
        counts=jnp.asarray(counts, jnp.int32),
        nbr=jnp.asarray(nbr),
        offsets=jnp.asarray(offsets),
        row_counts=jnp.asarray(row_counts, jnp.int32),
        pairs=(np.asarray(pi), np.asarray(pj)),
        n=n,
        n_live=n_live,
        W=W,
        L=L,
    )


def _topology(c):
    return (c["starts"], c["counts"], c["nbr"], c["offsets"], c["row_counts"])


def _pair_sum(c, kernel):
    """Pure-JAX pair sum over the case's pairs: ``a_i = sum_j -G m_j g(r_ij^2) d_ij``."""
    pi, pj = (jnp.asarray(p) for p in c["pairs"])

    def acc(pos, mass, soft, g):
        params = softening_params_from_sq(kernel, soft, pos.dtype)
        d = pos[pi] - pos[pj]
        gk, _, _ = pair_factors(jnp.sum(d * d, axis=1), params, kernel)
        contrib = (-g * mass[pj] * gk)[:, None] * d
        return jax.ops.segment_sum(contrib, pi, num_segments=c["n"])

    return acc


def _rel(a, b):
    """Max-norm difference relative to the reference's max-norm."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


@pytest.mark.parametrize("interpret", [True, False])
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "case, row_limit",
    [
        (dict(seed=3), None),
        (dict(seed=5, consecutive=True), None),  # every row's sources merge into runs
        (dict(seed=7, W=6), 2),  # W padded to 8 target lanes; rows in pieces of 2
    ],
)
def test_reverse_matches_vjp_of_the_pure_jax_pair_sum(
    case, row_limit, kernel, interpret
):
    _native_or_skip(interpret)
    c = _symmetric_case(**case)
    soft = jnp.asarray(SOFTENING**2)
    G = jnp.asarray(1.7)
    if row_limit is not None:
        assert int(np.max(np.asarray(c["row_counts"]))) > 2 * row_limit  # 3+ pieces
    if kernel != "plummer":
        # both sides of the support are exercised: polynomial and Newtonian pairs
        pi, pj = c["pairs"]
        pos = np.asarray(c["pos"])
        r = np.linalg.norm(pos[pi] - pos[pj], axis=1)
        h = support_factor(kernel) * SOFTENING
        assert 0.05 < np.mean(r < h) < 0.95

    common = dict(
        leaf_width=c["W"],
        softening_sq=soft,
        G=G,
        chunk=1,
        row_limit=row_limit,
        softening_kernel=kernel,
        interpret=interpret,
    )
    out_r, vjp_r = jax.vjp(_pair_sum(c, kernel), c["pos"], c["mass"], soft, G)
    acc, _ = nearfield_leafpair_csr_sorted_direct_pallas(
        c["pos"], c["mass"], *_topology(c), **common
    )
    # float64 round-off: the kernel's tiled sums against a segment sum
    assert _rel(acc, out_r) < 1e-12
    w = jnp.asarray(np.random.default_rng(11).standard_normal((c["n"], 3)))
    got = nearfield_leafpair_csr_sorted_direct_reverse_pallas(
        c["pos"], c["mass"], *_topology(c), w, **common
    )
    names = ("positions", "masses", "softening_sq", "G")
    for name, a, b in zip(names, got, vjp_r(w)):
        assert np.all(np.isfinite(np.asarray(a))), name
        assert np.max(np.abs(np.asarray(b))) > 0, name
        # float64 round-off of sums of O(100) pair terms (measured <= 6.8e-15)
        assert _rel(a, b) < 1e-12, (name, _rel(a, b))
    pos_bar, mass_bar = got[:2]
    # rows no leaf holds get nothing
    assert not np.any(np.asarray(pos_bar)[c["n_live"] :])
    assert not np.any(np.asarray(mass_bar)[c["n_live"] :])


def test_self_terms_only_and_an_empty_neighbour_array():
    """No neighbour entries at all (the self terms alone): the reverse is the pure-JAX
    sum's, and an empty ``neighbors`` array is accepted."""
    c = _symmetric_case(23, p_edge=0.0)
    assert int(np.asarray(c["row_counts"]).sum()) == 0
    soft = jnp.asarray(SOFTENING**2)
    G = jnp.asarray(1.0)
    w = jnp.asarray(np.random.default_rng(8).standard_normal((c["n"], 3)))
    _, vjp_r = jax.vjp(_pair_sum(c, "plummer"), c["pos"], c["mass"], soft, G)
    want = vjp_r(w)
    for nbr in (c["nbr"], jnp.zeros((0,), jnp.int32)):
        got = nearfield_leafpair_csr_sorted_direct_reverse_pallas(
            c["pos"],
            c["mass"],
            c["starts"],
            c["counts"],
            nbr,
            c["offsets"],
            c["row_counts"],
            w,
            leaf_width=c["W"],
            softening_sq=soft,
            G=G,
            chunk=1,
            softening_kernel="plummer",
            interpret=True,
        )
        for a, b in zip(got, want):
            assert _rel(a, b) < 1e-12
