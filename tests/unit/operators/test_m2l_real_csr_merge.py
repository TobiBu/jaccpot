"""``merge_into_csr``: the cross-domain far pairs slotted into the local CSR far list.

The multi-GPU fused lane's local walk hands the M2L a far list in CSR order (its
row offsets in place of one target per entry), and the cross hook adds a COO list of
(local target, imported source) pairs. The concatenate-and-sort route expanded the
local list to one target per entry, put the cross pairs behind it and sorted the
whole far capacity again by target (``csr_by_target``, a stable sort). The merge
keeps the local list's order and sorts only the cross pairs; these tests pin that it
gives the SAME CSR entry for entry -- so the lanes kernel, which sums a row in list
order, gives bitwise the same locals.

    JAX_PLATFORMS=cpu pytest tests/unit/operators/test_m2l_real_csr_merge.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.m2l_real_csr import (
    csr_by_target,
    merge_into_csr,
    targets_from_csr_offsets,
)
from jaccpot.pallas.m2l_real_csr_lanes import m2l_real_csr_lanes_pallas

_ORDER = 3
_K = 4


def _sort_route(src, offsets, extra_src, extra_tgt, total_nodes):
    """The concatenate-and-sort route the merge replaces (``_l2l`` before S11)."""
    tgt = targets_from_csr_offsets(offsets, int(src.shape[0]))
    return csr_by_target(
        jnp.concatenate([src, extra_src]),
        jnp.concatenate([tgt, extra_tgt]),
        total_nodes=total_nodes,
    )


def _lists(seed, *, rows, own_pad, extra, empty_frac=0.4, max_row=7, garbage_pad=False):
    """A CSR list (empty rows, a padded tail) and a COO list with holes.

    The COO targets cover every row, so some rows get ONLY extra pairs; a few are
    ``-1`` on either side and a few point past the last row.
    """
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, max_row + 1, rows) * (rng.random(rows) >= empty_frac)
    live = int(counts.sum())
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    pad = (
        rng.integers(0, 3 * rows, own_pad) if garbage_pad else -np.ones(own_pad)
    ).astype(np.int32)
    src = np.concatenate([rng.integers(0, 3 * rows, live), pad]).astype(np.int32)
    extra_tgt = rng.integers(0, rows + 2, extra).astype(np.int32)
    extra_src = rng.integers(rows, 3 * rows, extra).astype(np.int32)
    # holes on either side independently: a pair is padding if either index is
    extra_src[rng.random(extra) < 0.12] = -1
    extra_tgt[rng.random(extra) < 0.12] = -1
    return (
        jnp.asarray(src),
        jnp.asarray(offsets),
        jnp.asarray(extra_src),
        jnp.asarray(extra_tgt),
    )


def _assert_same_csr(merged, sorted_route, width):
    src_m, off_m, live = (np.asarray(a) for a in merged)
    src_s, off_s, cnt_s = (np.asarray(a) for a in sorted_route)
    assert src_m.shape == (width,) and src_m.dtype == np.int32
    assert np.array_equal(off_m[:-1], off_s) and np.array_equal(np.diff(off_m), cnt_s)
    assert int(live) == int(off_m[-1])
    # every row's entries in the same order: the live prefixes are equal
    assert np.array_equal(src_m[:live], src_s[:live])
    assert np.all(src_m[live:] == -1)


@pytest.mark.parametrize(
    "case",
    [
        dict(seed=0, rows=12, own_pad=9, extra=20),
        dict(seed=1, rows=30, own_pad=0, extra=45),  # a full own capacity
        dict(seed=2, rows=7, own_pad=5, extra=60, empty_frac=0.8),  # mostly cross
        dict(seed=3, rows=25, own_pad=17, extra=3, garbage_pad=True),
        dict(seed=4, rows=1, own_pad=2, extra=6),
        dict(seed=5, rows=40, own_pad=4, extra=0),  # nothing to merge
        dict(seed=6, rows=9, own_pad=0, extra=11, empty_frac=1.0),  # no own entries
    ],
)
def test_the_merge_is_the_sorted_concatenation_entry_for_entry(case):
    src, offsets, extra_src, extra_tgt = _lists(**case)
    rows = case["rows"]
    _assert_same_csr(
        merge_into_csr(src, offsets, extra_src, extra_tgt, total_nodes=rows),
        _sort_route(src, offsets, extra_src, extra_tgt, rows),
        int(src.shape[0] + extra_src.shape[0]),
    )


def test_an_empty_own_capacity_and_no_valid_extra_pairs():
    empty = jnp.zeros((0,), jnp.int32)
    offsets = jnp.zeros((6,), jnp.int32)
    extra_src = jnp.asarray([3, -1, 4], jnp.int32)
    extra_tgt = jnp.asarray([-1, 2, 9], jnp.int32)  # a hole each way, one past the rows
    merged = merge_into_csr(empty, offsets, extra_src, extra_tgt, total_nodes=5)
    _assert_same_csr(merged, _sort_route(empty, offsets, extra_src, extra_tgt, 5), 3)
    assert int(merged[2]) == 0


def test_rows_past_total_nodes_are_dropped_as_the_sort_route_drops_them():
    """Offsets longer than ``total_nodes + 1``: the trailing rows' own entries go."""
    src, offsets, extra_src, extra_tgt = _lists(7, rows=15, own_pad=3, extra=12)
    _assert_same_csr(
        merge_into_csr(src, offsets, extra_src, extra_tgt, total_nodes=11),
        _sort_route(src, offsets, extra_src, extra_tgt, 11),
        int(src.shape[0] + extra_src.shape[0]),
    )


def test_the_merge_sorts_only_the_extra_pairs():
    """One sort, over the cross pairs: none over the own list's capacity."""
    src, offsets, extra_src, extra_tgt = _lists(8, rows=20, own_pad=64, extra=5)
    jaxpr = jax.make_jaxpr(lambda *a: merge_into_csr(*a, total_nodes=20))(
        src, offsets, extra_src, extra_tgt
    )
    sorts = [e for e in jaxpr.jaxpr.eqns if e.primitive.name == "sort"]
    assert [tuple(v.aval.shape for v in e.invars) for e in sorts] == [((5,), (5,))]


def test_the_lanes_kernel_is_bitwise_on_the_merged_list():
    """The kernel's locals from the merged CSR equal those from the sort route.

    Rows longer than the lane count, so a row's tiles and its lane reduction both
    depend on where each entry sits in the row; the sources reach past the target
    rows into the imported range, as on the cross lane (``n_targets``).
    """
    n_local, n_remote = 24, 16
    n = n_local + n_remote
    src, offsets, extra_src, extra_tgt = _lists(
        9, rows=n_local, own_pad=6, extra=50, max_row=11
    )
    # own sources local and never the row itself (a zero separation is singular),
    # extra sources imported (the cross pairs' convention)
    rows = targets_from_csr_offsets(offsets, int(src.shape[0]))
    src = jnp.where(src >= 0, (rows + 1 + src % (n_local - 1)) % n_local, src)
    extra_src = jnp.where(extra_src >= 0, n_local + extra_src % n_remote, extra_src)
    rng = np.random.default_rng(10)
    C = (_ORDER + 1) ** 2
    mult = jnp.asarray(rng.standard_normal((n, C)), jnp.float32)
    cent = jnp.asarray(rng.uniform(-1.0, 1.0, (n, 3)) * 8.0, jnp.float32)
    kw = dict(order=_ORDER, k_lanes=_K, interpret=True, n_targets=n_local)

    m_src, m_off, _live = merge_into_csr(
        src, offsets, extra_src, extra_tgt, total_nodes=n_local
    )
    merged = m2l_real_csr_lanes_pallas(mult, cent, m_src, m_off, presorted=True, **kw)
    tgt = targets_from_csr_offsets(offsets, int(src.shape[0]))
    sorted_route = m2l_real_csr_lanes_pallas(
        mult,
        cent,
        jnp.concatenate([src, extra_src]),
        jnp.concatenate([tgt, extra_tgt]),
        **kw,
    )
    assert int(np.max(np.diff(np.asarray(m_off)))) > 2 * _K
    assert np.all(np.isfinite(np.asarray(merged)))
    # exact, not close: the same pairs in the same lanes in the same order
    assert np.array_equal(np.asarray(merged), np.asarray(sorted_route))
    # the imported sources are load-bearing: dropping them moves the locals
    assert not np.array_equal(
        np.asarray(merged),
        np.asarray(
            m2l_real_csr_lanes_pallas(mult, cent, src, offsets, presorted=True, **kw)
        ),
    )
