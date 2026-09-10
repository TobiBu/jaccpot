"""``_chunk_segment_scatter_add``: the per-chunk M2L reduction into the locals.

Pinned to a numpy ``np.add.at`` reference because the 2026-09-07 rewrite
(segmented ``associative_scan`` + out-of-bounds sink instead of ``segment_sum``
+ index-0 dummies) changed every line of the body: the function must still add
exactly the valid rows to exactly their targets, count a target that appears in
many rows once per row, leave every other node untouched, and be deterministic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime.kernels._m2l import _chunk_segment_scatter_add


def _case(seed, *, chunk_size, total_nodes, ncoef, n_valid, max_target):
    rng = np.random.default_rng(seed)
    local = rng.standard_normal((total_nodes, ncoef)).astype(np.float32)
    contribs = rng.standard_normal((chunk_size, ncoef)).astype(np.float32)
    tgt = rng.integers(0, max_target, size=chunk_size).astype(np.int32)
    valid = np.zeros(chunk_size, bool)
    valid[:n_valid] = True
    rng.shuffle(valid)
    # invalid slots may hold anything, including out-of-range garbage
    tgt = np.where(
        valid, tgt, rng.integers(-5, total_nodes + 7, size=chunk_size)
    ).astype(np.int32)
    expected = local.astype(np.float64).copy()
    np.add.at(expected, tgt[valid], contribs[valid].astype(np.float64))
    return local, contribs, tgt, valid, expected


@pytest.mark.parametrize(
    "chunk_size,total_nodes,n_valid,max_target",
    [
        (64, 40, 64, 40),  # full chunk, every target hit many times
        (64, 40, 37, 40),  # padded tail
        (64, 400, 64, 400),  # mostly unique targets
        (64, 40, 64, 1),  # every row the SAME target (the serialised case)
        (64, 40, 0, 40),  # nothing valid: the accumulator is untouched
        (1, 3, 1, 3),  # degenerate width
    ],
)
def test_chunk_segment_scatter_add_matches_np_add_at(
    chunk_size, total_nodes, n_valid, max_target
):
    local, contribs, tgt, valid, expected = _case(
        7,
        chunk_size=chunk_size,
        total_nodes=total_nodes,
        ncoef=9,
        n_valid=n_valid,
        max_target=max_target,
    )
    got = _chunk_segment_scatter_add(
        jnp.asarray(local),
        jnp.asarray(contribs),
        jnp.asarray(tgt),
        jnp.asarray(valid),
        chunk_size=chunk_size,
    )
    got = np.asarray(got, np.float64)
    assert got.shape == expected.shape
    scale = np.max(np.abs(expected)) + 1.0
    assert np.allclose(got, expected, rtol=0, atol=2e-5 * scale)
    if n_valid == 0:
        np.testing.assert_array_equal(got, local.astype(np.float64))


def test_chunk_segment_scatter_add_hits_node_zero_only_when_targeted():
    """Node 0 used to receive every non-head slot as a zero add; now it must be
    touched only by rows that target it -- checked with a chunk that never does."""
    local, contribs, tgt, valid, expected = _case(
        3, chunk_size=32, total_nodes=16, ncoef=4, n_valid=32, max_target=16
    )
    tgt = np.where(tgt == 0, 5, tgt).astype(np.int32)
    expected = local.astype(np.float64).copy()
    np.add.at(expected, tgt[valid], contribs[valid].astype(np.float64))
    got = np.asarray(
        _chunk_segment_scatter_add(
            jnp.asarray(local),
            jnp.asarray(contribs),
            jnp.asarray(tgt),
            jnp.asarray(valid),
            chunk_size=32,
        ),
        np.float64,
    )
    np.testing.assert_array_equal(got[0], local[0].astype(np.float64))
    assert np.allclose(got, expected, rtol=0, atol=1e-4)


def test_chunk_segment_scatter_add_is_deterministic_under_jit():
    local, contribs, tgt, valid, _ = _case(
        11, chunk_size=256, total_nodes=50, ncoef=25, n_valid=250, max_target=50
    )
    fn = jax.jit(
        lambda a, c, t, v: _chunk_segment_scatter_add(a, c, t, v, chunk_size=256)
    )
    args = (
        jnp.asarray(local),
        jnp.asarray(contribs),
        jnp.asarray(tgt),
        jnp.asarray(valid),
    )
    a = np.asarray(fn(*args))
    b = np.asarray(fn(*args))
    np.testing.assert_array_equal(a, b)
