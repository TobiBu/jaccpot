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


@pytest.mark.parametrize("chunk", [1, 2, 3, 8])
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


@pytest.mark.parametrize("chunk", [1, 2, 5])
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
