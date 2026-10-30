"""Tests for the CSR-driven leaf-pair near-field kernel (plan sub-10ms, Phase 4.1).

The Pallas kernel runs under ``interpret=True`` so these are CPU tests; on an
Ampere+ GPU the same kernel is A/B'd against the rectangle kernel by
``bench/nearfield_csr_microbench.py`` and by the fused-lane GPU tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.nearfield_fused_leaf import nearfield_leafpair_pallas
from jaccpot.pallas.nearfield_leafpair_csr import (
    build_leafpair_chunk_table,
    leafpair_chunk_capacity,
    nearfield_leafpair_csr_jax,
    nearfield_leafpair_csr_pallas,
)
from tests.unit._typecheck_budget import trim


def _random_csr(seed, L, W, *, max_row, edge_capacity=None, empty_rows=()):
    """Random leaf tables plus a CSR whose rows never contain the row's own leaf."""
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((L, W, 3)).astype(np.float32)
    mass = (np.abs(rng.standard_normal((L, W))) + 0.1).astype(np.float32)
    mask = rng.random((L, W)) > 0.2
    mask[:, 0] = True  # every leaf has at least one particle
    counts = rng.integers(0, max_row + 1, size=L)
    for r in empty_rows:
        counts[r] = 0
    rows = []
    for leaf in range(L):
        others = np.setdiff1d(np.arange(L), [leaf])
        rows.append(rng.choice(others, size=counts[leaf], replace=True))
    neighbors = np.concatenate(rows) if L else np.zeros(0, np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    cap = int(edge_capacity or max(1, neighbors.size))
    assert cap >= neighbors.size
    nbr_padded = np.zeros(cap, np.int32)
    nbr_padded[: neighbors.size] = neighbors
    i32 = jnp.int32
    return (
        jnp.asarray(pos),
        jnp.asarray(mass),
        jnp.asarray(mask),
        jnp.asarray(nbr_padded, i32),
        jnp.asarray(offsets, i32),
        jnp.asarray(counts, i32),
    )


@pytest.mark.parametrize("chunk", trim([3, 1, 2, 8]))
def test_chunk_table_covers_every_row_exactly_once(chunk):
    _, _, _, nbr, offsets, counts = _random_csr(
        3, L=7, W=4, max_row=9, empty_rows=(2, 5)
    )
    cap = leafpair_chunk_capacity(int(nbr.shape[0]), 7, chunk)
    tab = build_leafpair_chunk_table(offsets, counts, chunk=chunk, capacity=cap)
    leaf = np.asarray(tab.leaf)
    start = np.asarray(tab.start)
    count = np.asarray(tab.count)
    first = np.asarray(tab.is_first)
    live = leaf >= 0
    # live prefix, then padding
    assert np.all(live[: live.sum()]) and not np.any(live[live.sum() :])
    assert np.all(count[~live] == 0) and np.all(first[~live] == 0)
    off = np.asarray(offsets)
    cnt = np.asarray(counts)
    for l in range(7):
        mine = np.where(leaf == l)[0]
        assert len(mine) == max(1, -(-int(cnt[l]) // chunk))
        assert first[mine].sum() == 1 and first[mine[0]] == 1
        covered = np.concatenate(
            [np.arange(start[c], start[c] + count[c]) for c in mine]
        )
        assert np.array_equal(covered, np.arange(off[l], off[l + 1]))
        assert np.all(count[mine] <= chunk)
    # chunks of one leaf are consecutive (the sorted segment sum relies on it)
    assert np.all(np.diff(leaf[live]) >= 0)


@pytest.mark.parametrize("chunk", trim([2, 1, 5]))
@pytest.mark.parametrize("include_self", [False, True])
def test_csr_interpret_matches_dense_twin(chunk, include_self):
    L, W = 6, 8
    pos, mass, mask, nbr, offsets, counts = _random_csr(
        11, L=L, W=W, max_row=7, edge_capacity=64, empty_rows=(4,)
    )
    soft = jnp.float32(0.05**2)
    G = jnp.float32(1.3)
    cap = leafpair_chunk_capacity(int(nbr.shape[0]), L, chunk)
    tab = build_leafpair_chunk_table(offsets, counts, chunk=chunk, capacity=cap)
    got = nearfield_leafpair_csr_pallas(
        pos,
        mass,
        mask,
        nbr,
        tab,
        softening_sq=soft,
        G=G,
        chunk=chunk,
        interpret=True,
        include_self=include_self,
    )
    ref = nearfield_leafpair_csr_jax(
        pos,
        mass,
        mask,
        nbr,
        offsets,
        counts,
        softening_sq=soft,
        G=G,
        include_self=include_self,
    )
    assert got.shape == (L, W, 4)
    assert np.all(np.isfinite(np.asarray(got)))
    # non-vacuity: the empty row 4 has only its self term (or nothing)
    if not include_self:
        assert np.allclose(np.asarray(got)[4], 0.0)
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-6)


def test_csr_matches_rectangle_kernel_and_pads_subtile():
    """W = 6 is not a power of two: Bt = 4, the table is padded to 8 lanes."""
    L, W = 5, 6
    pos, mass, mask, nbr, offsets, counts = _random_csr(
        5, L=L, W=W, max_row=4, edge_capacity=32
    )
    soft = jnp.float32(0.02**2)
    G = jnp.float32(0.7)
    chunk = 3
    cap = leafpair_chunk_capacity(int(nbr.shape[0]), L, chunk)
    tab = build_leafpair_chunk_table(offsets, counts, chunk=chunk, capacity=cap)
    got = nearfield_leafpair_csr_pallas(
        pos,
        mass,
        mask,
        nbr,
        tab,
        softening_sq=soft,
        G=G,
        chunk=chunk,
        interpret=True,
        include_self=True,
    )
    # the rectangle kernel on the same lists
    max_count = int(jnp.max(counts))
    slots = jnp.arange(max_count, dtype=jnp.int32)
    valid = slots[None, :] < counts[:, None]
    ids = jnp.where(
        valid, nbr[jnp.where(valid, offsets[:-1, None] + slots[None, :], 0)], 0
    )
    rect = nearfield_leafpair_pallas(
        pos,
        mass,
        mask,
        ids,
        valid,
        softening_sq=soft,
        G=G,
        interpret=True,
        include_self=True,
        source_chunk=2,
    )
    assert got.shape == rect.shape == (L, W, 4)
    assert np.allclose(np.asarray(got), np.asarray(rect), rtol=1e-5, atol=1e-6)


def test_csr_wide_accumulator_interpret_matches_input_accumulator():
    L, W = 4, 8
    pos, mass, mask, nbr, offsets, counts = _random_csr(
        9, L=L, W=W, max_row=3, edge_capacity=16
    )
    soft = jnp.float32(0.1**2)
    G = jnp.float32(1.0)
    chunk = 2
    cap = leafpair_chunk_capacity(int(nbr.shape[0]), L, chunk)
    tab = build_leafpair_chunk_table(offsets, counts, chunk=chunk, capacity=cap)
    common = dict(
        softening_sq=soft, G=G, chunk=chunk, interpret=True, include_self=True
    )
    narrow = nearfield_leafpair_csr_pallas(
        pos, mass, mask, nbr, tab, accum="input", **common
    )
    wide = nearfield_leafpair_csr_pallas(
        pos, mass, mask, nbr, tab, accum="wide", **common
    )
    assert wide.dtype == narrow.dtype == jnp.float32
    assert np.allclose(np.asarray(wide), np.asarray(narrow), rtol=1e-5, atol=1e-6)


def test_csr_is_jittable_with_traced_table():
    L, W = 4, 4
    pos, mass, mask, nbr, offsets, counts = _random_csr(
        2, L=L, W=W, max_row=3, edge_capacity=16
    )
    chunk = 2
    cap = leafpair_chunk_capacity(int(nbr.shape[0]), L, chunk)

    @jax.jit
    def f(pos, mass, mask, nbr, offsets, counts):
        tab = build_leafpair_chunk_table(offsets, counts, chunk=chunk, capacity=cap)
        return nearfield_leafpair_csr_pallas(
            pos,
            mass,
            mask,
            nbr,
            tab,
            softening_sq=jnp.float32(1e-4),
            G=jnp.float32(1.0),
            chunk=chunk,
            interpret=True,
        )

    out = f(pos, mass, mask, nbr, offsets, counts)
    ref = nearfield_leafpair_csr_jax(
        pos,
        mass,
        mask,
        nbr,
        offsets,
        counts,
        softening_sq=jnp.float32(1e-4),
        G=jnp.float32(1.0),
    )
    assert np.allclose(np.asarray(out), np.asarray(ref), rtol=1e-5, atol=1e-6)


# ---- ordered_rows: a row's chunk partials summed in one fixed order (S11 s2, G) ----


@pytest.mark.parametrize("chunk", trim([2, 1, 3]))
def test_ordered_rows_sum_each_rows_chunks_alone_in_a_fixed_order(chunk):
    """``_sum_chunks_in_fixed_order``: each row's sum, from that row's chunks alone.

    The ``segment_sum`` it replaces under ``ordered_rows`` is one scatter-add, whose
    order within a row is the atomics' on a GPU (the multi-GPU cross near term
    differed run to run in rows of three or more chunks). Here: rows of one or two
    chunks are ``p_0`` and ``p_0 + p_1`` bitwise (written out in float32 numpy),
    longer ones (up to five chunks; empty rows too) the float64 sum to float32
    accuracy, and no row's bits move when every OTHER row's partials change (to
    other values, and to inf and NaN): the sum's order depends on the table alone."""
    from jaccpot.pallas.nearfield_leafpair_csr import _sum_chunks_in_fixed_order

    L, W = 7, 4
    _, _, _, nbr, offsets, counts = _random_csr(
        17, L=L, W=W, max_row=9, empty_rows=(1, 4)
    )
    cap = leafpair_chunk_capacity(int(nbr.shape[0]), L, chunk)
    tab = build_leafpair_chunk_table(offsets, counts, chunk=chunk, capacity=cap)
    leaf = np.asarray(tab.leaf)
    per_row = np.bincount(leaf[leaf >= 0], minlength=L)
    # non-vacuity: rows the scatter could add in another order, and short ones
    assert per_row.max() >= 3 and per_row.min() <= 2, per_row
    rng = np.random.default_rng(5)
    partials = rng.standard_normal((cap, W, 4)).astype(np.float32)
    rows = jnp.where(tab.leaf < 0, L, tab.leaf)
    reduce = jax.jit(_sum_chunks_in_fixed_order, static_argnums=3)

    got = np.asarray(reduce(jnp.asarray(partials), rows, tab.is_first, L))
    assert got.dtype == np.float32
    for row in range(L):
        mine = np.flatnonzero(leaf == row)
        if mine.size <= 2:
            want = (
                partials[mine[0]]
                if mine.size == 1
                else partials[mine[0]] + partials[mine[1]]
            )
            assert np.array_equal(got[row], want), row
        else:
            want = partials[mine].astype(np.float64).sum(axis=0)
            assert np.allclose(got[row], want, rtol=1e-5, atol=1e-6), row
        for poison in (None, np.inf, np.nan):
            other = rng.standard_normal(partials.shape).astype(np.float32)
            if poison is not None:
                other[:] = poison
            moved = np.where((leaf == row)[:, None, None], partials, other)
            again = np.asarray(reduce(jnp.asarray(moved), rows, tab.is_first, L))
            assert np.array_equal(again[row], got[row]), (row, poison)


@pytest.mark.parametrize("num_target_leaves", [None, 5])
def test_ordered_rows_keep_the_kernels_answer(num_target_leaves):
    """The kernel with ``ordered_rows`` gives the ``segment_sum`` route's answer.

    With fewer target rows than pool leaves too (the cross near term's
    ``[local ; imported]`` pool, its table over the local rows alone)."""
    L, W, chunk = 7, 8, 2
    pos, mass, mask, nbr, offsets, counts = _random_csr(
        23, L=L, W=W, max_row=9, edge_capacity=96
    )
    rows = L if num_target_leaves is None else num_target_leaves
    offsets, counts = offsets[: rows + 1], counts[:rows]
    cap = leafpair_chunk_capacity(int(nbr.shape[0]), rows, chunk)
    tab = build_leafpair_chunk_table(offsets, counts, chunk=chunk, capacity=cap)
    assert int(jnp.max(counts)) > 2 * chunk  # rows of three or more chunks
    common = dict(
        softening_sq=jnp.float32(0.05**2),
        G=jnp.float32(1.1),
        chunk=chunk,
        interpret=True,
        include_self=False,
        num_target_leaves=num_target_leaves,
    )
    ordered = nearfield_leafpair_csr_pallas(
        pos, mass, mask, nbr, tab, ordered_rows=True, **common
    )
    summed = nearfield_leafpair_csr_pallas(pos, mass, mask, nbr, tab, **common)
    assert ordered.shape == summed.shape == (rows, W, 4)
    assert np.all(np.isfinite(np.asarray(ordered)))
    assert np.abs(np.asarray(summed)).max() > 0.0
    assert np.allclose(np.asarray(ordered), np.asarray(summed), rtol=1e-6, atol=1e-7)
