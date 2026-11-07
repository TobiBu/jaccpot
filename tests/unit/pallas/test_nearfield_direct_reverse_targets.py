"""The direct near reverse's target-leaves keywords (``num_target_leaves``, ``include_self``).

The cross-domain near term (``cross.cross_near_acceleration_direct``) runs the direct
near kernel with only the leading ``T`` leaves as targets, the rest of the table its
imported particles, and no self term. Its reverse
(``jaccpot.distributed.cross_vjp``) runs
:func:`nearfield_leafpair_csr_sorted_direct_reverse_pallas` with the same two
keywords. Pinned here on one device, float64, Pallas interpreter (natively on an
Ampere card):

* two "devices" A and B emulated in one process: a SYMMETRIC set of A-B leaf pairs,
  A's forward over the table ``[A; B]`` (A's leaves the targets) and B's over ``[B;
  A]``. Each reverse, given the cotangents of both ends (its own rows, then the other
  end's), returns its own particles' cotangents, and with the two partial scalars
  summed they equal ``jax.vjp`` of a pure-JAX sum over the A-B particle pairs, to
  float64 round-off -- also when the rows run in pieces (``row_limit``) and for the
  solver's default ``ferrers3`` kernel;
* rows past the target leaves get zero cotangents;
* ``num_target_leaves=L`` with the self term is the default call, the same program;
* ``num_target_leaves`` outside ``[0, L]`` is refused, and ``0`` returns zeros.
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

SOFTENING = 0.5
G_VALUE = 1.7


def _native_or_skip(interpret: bool) -> None:
    if not interpret and not pallas_m2l_real_csr_supported():
        pytest.skip("native Pallas GPU lowering not available here")


def _side(rng, counts, W):
    """Sorted particles of one side: leaf ranges of ``counts`` (an empty padding
    leaf last, starting past the live rows), two dead rows past them."""
    counts = np.asarray(counts)
    n_live = int(counts.sum())
    n = n_live + 2
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    starts[counts == 0] = n_live
    pos = rng.standard_normal((n, 3))
    mass = np.concatenate([np.abs(rng.standard_normal(n_live)) + 0.1, np.zeros(2)])
    return dict(pos=pos, mass=mass, starts=starts, counts=counts, n=n)


def _case(seed, *, W=8):
    """A (5 leaves + a padding leaf) and B (4 leaves), a symmetric A-B pair set."""
    rng = np.random.default_rng(seed)
    a = _side(rng, list(rng.integers(1, W + 1, 5)) + [0], W)
    b = _side(rng, list(rng.integers(1, W + 1, 4)), W)
    adj = rng.random((5, 4)) < 0.6
    adj[1, :] = False  # an A row without cross pairs
    adj[0, :] = True  # a long row (pieces at row_limit 2)
    return a, b, adj


def _call_layout(tgt, src, adj):
    """The table ``[tgt; src]`` with ``tgt``'s leaves the targets: leaf ranges over
    both sides, a CSR over the target leaves listing the source leaves (shifted)."""
    Lt = len(tgt["counts"])
    starts = np.concatenate([tgt["starts"], tgt["n"] + src["starts"]])
    counts = np.concatenate([tgt["counts"], src["counts"]])
    rows = [Lt + np.flatnonzero(adj[i]) if i < adj.shape[0] else [] for i in range(Lt)]
    row_counts = np.array([len(r) for r in rows], np.int32)
    nbr = np.concatenate([np.asarray(r, np.int64) for r in rows] + [np.zeros(2)])
    offsets = np.concatenate([[0], np.cumsum(row_counts)])
    return (
        jnp.asarray(np.concatenate([tgt["pos"], src["pos"]])),
        jnp.asarray(np.concatenate([tgt["mass"], src["mass"]])),
        jnp.asarray(starts, jnp.int32),
        jnp.asarray(counts, jnp.int32),
        jnp.asarray(nbr, jnp.int32),
        jnp.asarray(offsets, jnp.int32),
        jnp.asarray(row_counts, jnp.int32),
        Lt,
    )


def _pairs(a, b, adj):
    """(target, source) particle pairs of the A-B leaf pairs, both directions, in the
    global order ``[A; B]``."""
    pi, pj = [], []
    for i, j in zip(*np.nonzero(adj)):
        for p in range(a["starts"][i], a["starts"][i] + a["counts"][i]):
            for q in range(b["starts"][j], b["starts"][j] + b["counts"][j]):
                pi += [p, a["n"] + q]
                pj += [a["n"] + q, p]
    return np.asarray(pi), np.asarray(pj)


def _pair_sum(pi, pj, n, kernel):
    pi, pj = jnp.asarray(pi), jnp.asarray(pj)

    def acc(pos, mass, soft, g):
        params = softening_params_from_sq(kernel, soft, pos.dtype)
        d = pos[pi] - pos[pj]
        gk, _, _ = pair_factors(jnp.sum(d * d, axis=1), params, kernel)
        return jax.ops.segment_sum((-g * mass[pj] * gk)[:, None] * d, pi, n)

    return acc


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


@pytest.mark.parametrize("interpret", [True, False])
@pytest.mark.parametrize(
    "kernel, row_limit",
    [("plummer", None), ("plummer", 2), ("ferrers3", None), ("ferrers3", 2)],
)
def test_two_sides_together_are_the_vjp_of_the_cross_pair_sum(
    kernel, row_limit, interpret
):
    _native_or_skip(interpret)
    a, b, adj = _case(3)
    n_a, n_b = a["n"], b["n"]
    soft = jnp.asarray(SOFTENING**2)
    G = jnp.asarray(G_VALUE)
    pi, pj = _pairs(a, b, adj)
    assert pi.size > 0
    if kernel != "plummer":
        pos = np.concatenate([a["pos"], b["pos"]])
        r = np.linalg.norm(pos[pi] - pos[pj], axis=1)
        assert 0.05 < np.mean(r < support_factor(kernel) * SOFTENING) < 0.95
    pos_all = jnp.asarray(np.concatenate([a["pos"], b["pos"]]))
    mass_all = jnp.asarray(np.concatenate([a["mass"], b["mass"]]))
    want_acc, vjp = jax.vjp(
        _pair_sum(pi, pj, n_a + n_b, kernel), pos_all, mass_all, soft, G
    )
    w = jnp.asarray(np.random.default_rng(5).standard_normal((n_a + n_b, 3)))
    w_a, w_b = w[:n_a], w[n_a:]
    want = vjp(w)

    common = dict(
        softening_sq=soft,
        G=G,
        chunk=1,
        row_limit=row_limit,
        softening_kernel=kernel,
        interpret=interpret,
        include_self=False,
    )
    got = []
    for tgt, src, adj_t, w_t, w_s in ((a, b, adj, w_a, w_b), (b, a, adj.T, w_b, w_a)):
        p, m, ls, lc, nb, off, cnt, Lt = _call_layout(tgt, src, adj_t)
        if row_limit is not None:
            assert int(np.max(np.asarray(cnt))) > row_limit  # a row in pieces
        acc, _ = nearfield_leafpair_csr_sorted_direct_pallas(
            p, m, ls, lc, nb, off, cnt, leaf_width=8, num_target_leaves=Lt, **common
        )
        got.append(
            (
                acc[: tgt["n"]],
                nearfield_leafpair_csr_sorted_direct_reverse_pallas(
                    p,
                    m,
                    ls,
                    lc,
                    nb,
                    off,
                    cnt,
                    jnp.concatenate([w_t, w_s]),
                    leaf_width=8,
                    num_target_leaves=Lt,
                    **common,
                ),
                tgt["n"],
            )
        )
    (acc_a, rev_a, _), (acc_b, rev_b, _) = got
    # the forward with the same keywords is the cross pair sum on its targets
    assert _rel(jnp.concatenate([acc_a, acc_b]), want_acc) < 1e-12
    # the source rows of each call get nothing
    for rev, n_t in ((rev_a, n_a), (rev_b, n_b)):
        assert not np.any(np.asarray(rev[0])[n_t:])
        assert not np.any(np.asarray(rev[1])[n_t:])
    pos_bar = jnp.concatenate([rev_a[0][:n_a], rev_b[0][:n_b]])
    mass_bar = jnp.concatenate([rev_a[1][:n_a], rev_b[1][:n_b]])
    soft_bar = rev_a[2] + rev_b[2]  # per-call partials: only the sum is the cotangent
    g_bar = rev_a[3] + rev_b[3]
    for name, x, y in zip(
        ("positions", "masses", "softening_sq", "G"),
        (pos_bar, mass_bar, soft_bar, g_bar),
        want,
    ):
        assert np.max(np.abs(np.asarray(y))) > 0, name
        # measured <= 8.5e-16 (float64 round-off of O(100)-pair sums)
        assert _rel(x, y) < 1e-12, (name, _rel(x, y))
    # the dead rows (no leaf) get nothing
    assert not np.any(np.asarray(pos_bar)[n_a - 2 : n_a])
    assert not np.any(np.asarray(mass_bar)[n_a + n_b - 2 :])


def test_all_leaves_with_the_self_term_is_the_default_program():
    a, b, adj = _case(7)
    p, m, ls, lc, nb, off, cnt, Lt = _call_layout(a, b, adj)
    # every leaf a target: the CSR needs a row per leaf
    L = int(ls.shape[0])
    cnt = jnp.concatenate([cnt, jnp.zeros((L - Lt,), cnt.dtype)])
    off = jnp.concatenate([off, jnp.full((L - Lt,), off[-1], off.dtype)])
    w = jnp.asarray(np.random.default_rng(1).standard_normal((int(p.shape[0]), 3)))
    args = (p, m, ls, lc, nb, off, cnt, w, jnp.asarray(0.25), jnp.asarray(1.3))

    def call(**kw):
        def f(p, m, ls, lc, nb, off, cnt, w, s, g):
            return nearfield_leafpair_csr_sorted_direct_reverse_pallas(
                p,
                m,
                ls,
                lc,
                nb,
                off,
                cnt,
                w,
                leaf_width=8,
                softening_sq=s,
                G=g,
                chunk=1,
                interpret=True,
                **kw,
            )

        return jax.jit(f).lower(*args).as_text(debug_info=False)

    assert call() == call(num_target_leaves=L, include_self=True)
    assert call() != call(num_target_leaves=L, include_self=False)


def test_num_target_leaves_is_checked():
    a, b, adj = _case(11)
    p, m, ls, lc, nb, off, cnt, Lt = _call_layout(a, b, adj)
    w = jnp.zeros((int(p.shape[0]), 3))
    kw = dict(leaf_width=8, softening_sq=0.25, G=1.0, chunk=1, interpret=True)
    with pytest.raises(ValueError, match="num_target_leaves"):
        nearfield_leafpair_csr_sorted_direct_reverse_pallas(
            p, m, ls, lc, nb, off, cnt, w, num_target_leaves=int(ls.shape[0]) + 1, **kw
        )
    out = nearfield_leafpair_csr_sorted_direct_reverse_pallas(
        p, m, ls, lc, nb, off, cnt[:0], w, num_target_leaves=0, **kw
    )
    assert all(not np.any(np.asarray(o)) for o in out)
