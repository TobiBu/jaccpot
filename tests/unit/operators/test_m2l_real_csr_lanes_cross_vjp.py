"""Single-device pieces of the cross-domain M2L seam (WP-G seam A; fp64, interpret).

:func:`cross_far_m2l_reverse_pallas` is the symmetric sort-free reverse with the
program grid over the first ``n_local`` rows. On one device that is checkable
without a mesh: over a symmetric list it equals
:func:`m2l_real_csr_lanes_reverse_pallas` (``symmetric=True``) bit for bit -- on
every row when ``n_local`` covers them all, and on the first ``n_local`` rows (zero
after) when the remaining rows stand in for imported nodes whose local cotangents
are handed in. The seam's forward is the production launch bit for bit. The
multi-device gate (the exchange, the oracle over the global pair set) is
``tests/distributed/test_cross_far_m2l_vjp.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.m2l_real_csr_lanes import (
    m2l_real_csr_lanes_pallas_cvjp,
    m2l_real_csr_lanes_reverse_pallas,
)
from jaccpot.pallas.m2l_real_csr_lanes_cross_vjp import (
    cross_far_m2l_cvjp,
    cross_far_m2l_reverse_pallas,
)

jax.config.update("jax_enable_x64", True)

ORDER = 3
K = 8


def _symmetric_case(seed: int, n: int, pad: int = 5):
    """A symmetric CSR over ``n`` nodes (rows sorted by source), live prefix + pad."""
    rng = np.random.default_rng(seed)
    C = (ORDER + 1) ** 2
    grid = np.stack(np.meshgrid(*(np.arange(3),) * 3, indexing="ij"), -1).reshape(-1, 3)
    cent = grid[rng.permutation(len(grid))[:n]] * 3.0 + rng.uniform(-0.4, 0.4, (n, 3))
    mult = rng.standard_normal((n, C))
    adj = np.triu(rng.random((n, n)) < 0.6, 1)
    adj = adj | adj.T
    adj[0, :] = adj[:, 0] = False  # an empty row
    adj[1, 2:] = adj[2:, 1] = True  # a row longer than one tile
    adj[np.arange(n), np.arange(n)] = False
    rows = [np.flatnonzero(adj[t]) for t in range(n)]
    off = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
    src = np.concatenate([np.concatenate(rows), -np.ones(pad, np.int64)])
    return (
        jnp.asarray(mult),
        jnp.asarray(cent),
        jnp.asarray(src, jnp.int32),
        jnp.asarray(off, jnp.int32),
        jnp.asarray(rng.standard_normal((n, C))),
    )


@pytest.fixture(scope="module")
def case():
    return _symmetric_case(3, 14)


@pytest.fixture(scope="module")
def full_reverse(case):
    mult, cent, src, off, lbar = case
    return m2l_real_csr_lanes_reverse_pallas(
        mult,
        cent,
        src,
        src,  # targets: unused under symmetric=True (an Array for the type checker)
        lbar,
        order=ORDER,
        k_lanes=K,
        interpret=True,
        target_row_offsets=off,
        symmetric=True,
    )


def test_all_rows_local_is_the_symmetric_reverse_bitwise(case, full_reverse):
    mult, cent, src, off, lbar = case
    got = cross_far_m2l_reverse_pallas(
        mult,
        cent,
        src,
        off,
        lbar,
        order=ORDER,
        n_local=int(mult.shape[0]),
        k_lanes=K,
        interpret=True,
    )
    assert np.linalg.norm(np.asarray(full_reverse[1])) > 0
    for x, y in zip(got, full_reverse):
        assert np.array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("n_local", [5, 9])
def test_local_rows_equal_the_full_reverse_rest_zero(case, full_reverse, n_local):
    """Rows past ``n_local`` play the imported nodes: no program, zero cotangent;
    the local rows read their partners' cotangents from ``loc_bar_full``."""
    mult, cent, src, off, lbar = case
    mb, cb = cross_far_m2l_reverse_pallas(
        mult,
        cent,
        src,
        off,
        lbar,
        order=ORDER,
        n_local=n_local,
        k_lanes=K,
        interpret=True,
    )
    mb_full, cb_full = (np.asarray(a) for a in full_reverse)
    # the local rows have partners past n_local: the imported half is exercised
    off_np, src_np = np.asarray(off), np.asarray(src)
    assert np.any(src_np[: off_np[n_local]] >= n_local)
    assert np.array_equal(np.asarray(mb)[:n_local], mb_full[:n_local])
    assert np.array_equal(np.asarray(cb)[:n_local], cb_full[:n_local])
    assert not np.any(np.asarray(mb)[n_local:]) and not np.any(np.asarray(cb)[n_local:])


@pytest.mark.parametrize("n_local", [5, 14])
def test_forward_is_the_production_launch_bitwise(case, n_local):
    mult, cent, src, off, _ = case
    n = int(mult.shape[0])
    live = off[n_local]
    send = jnp.zeros((3,), jnp.int32)
    got = cross_far_m2l_cvjp(
        mult,
        cent,
        src,
        off[: n_local + 1],
        live,
        send,
        send >= 0,
        jnp.zeros((1,), jnp.int32),
        ORDER,
        K,
        True,
        "triton",
        1,
        n_local,
        n - n_local,
        "gpus",
    )
    want = m2l_real_csr_lanes_pallas_cvjp(
        mult,
        cent,
        src,
        off[: n_local + 1],
        live,
        ORDER,
        K,
        True,
        "triton",
        1,
        n_local,
        True,
        False,
    )
    assert got.shape == (n_local, (ORDER + 1) ** 2)
    assert np.array_equal(np.asarray(got), np.asarray(want))


def test_refuses_rows_that_are_not_local_plus_imported(case):
    mult, cent, src, off, _ = case
    with pytest.raises(ValueError, match="n_local"):
        cross_far_m2l_cvjp(
            mult,
            cent,
            src,
            off,
            None,
            jnp.zeros((1,), jnp.int32),
            jnp.ones((1,), bool),
            jnp.zeros((1,), jnp.int32),
            ORDER,
            K,
            True,
            "triton",
            1,
            5,
            3,
            "gpus",
        )
