"""The deterministic near CSR from the canonical pairs equals the directed sort.

``_flat_walk_lists(deterministic=True)`` used to build the leaf-neighbour CSR by
one composite sort of the ``2W`` directed pairs by ``(target, source)``; it now
sorts the ``W`` canonical pairs once by ``(a, b)``, re-sorts them stably by
``b`` and places both halves of every row. Pinned against a NumPy directed sort
on random canonical pair sets: live prefixes shorter than the width, a count
past a full width (an overflowed walk), leaves with no neighbour, rows with only
lower or only higher sources.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime._interaction_cache import (
    _directed_csr_from_canonical,
    _flat_walk_lists,
)


@pytest.fixture(params=["xla", "interpret"])
def route(request, monkeypatch):
    """Both builders: the two sorts, and the Pallas placement + ranks (interpreted)."""
    monkeypatch.setenv("JACCPOT_LIST_CSR_KERNEL", request.param)
    return request.param


def _reference(na, nb, count, *, width, num_internal, num_leaves):
    live = np.arange(width) < count
    a = np.asarray(na)[:width][live]
    b = np.asarray(nb)[:width][live]
    tgt = np.concatenate([a, b]) - num_internal
    src = np.concatenate([b, a])
    order = np.lexsort((src, tgt))
    tgt, src = tgt[order], src[order]
    neighbors = np.zeros(2 * width, np.int64)
    neighbors[: src.size] = src
    offsets = np.searchsorted(tgt, np.arange(num_leaves + 1), side="left")
    return neighbors, offsets, np.diff(offsets)


def _pairs(rng, num_leaves, n_pairs, num_internal):
    seen = set()
    while len(seen) < n_pairs:
        a, b = sorted(rng.integers(0, num_leaves, size=2).tolist())
        if a != b:
            seen.add((a, b))
    pairs = np.asarray(sorted(seen, key=lambda _: rng.random()), np.int64)
    return pairs[:, 0] + num_internal, pairs[:, 1] + num_internal


@pytest.mark.parametrize(
    "num_leaves, n_pairs, width, count",
    [(40, 150, 200, 150), (40, 150, 150, 150), (17, 30, 40, 25), (60, 16, 16, 99)],
)
def test_canonical_build_equals_the_directed_sort(
    num_leaves, n_pairs, width, count, route
):
    rng = np.random.default_rng(num_leaves + n_pairs)
    num_internal = num_leaves - 1
    na, nb = _pairs(rng, num_leaves, n_pairs, num_internal)
    pad = max(width - n_pairs, 0)
    na_p = jnp.asarray(np.concatenate([na, np.full(pad, 7)]), jnp.int32)
    nb_p = jnp.asarray(np.concatenate([nb, np.full(pad, 3)]), jnp.int32)
    far = jnp.zeros((4,), jnp.int32)
    out = _flat_walk_lists(
        far,
        far,
        jnp.asarray(0, jnp.int32),
        na_p,
        nb_p,
        jnp.asarray(count, jnp.int32),
        jnp.asarray(False),
        far_width=4,
        near_width=width,
        num_internal=num_internal,
        total_nodes=num_internal + num_leaves,
        deterministic=True,
        idx=jnp.int32,
    )
    neighbors, offsets, counts = (np.asarray(x) for x in out[4:])
    ref_n, ref_o, ref_c = _reference(
        np.asarray(na_p),
        np.asarray(nb_p),
        min(count, width),
        width=width,
        num_internal=num_internal,
        num_leaves=num_leaves,
    )
    assert neighbors.shape == (2 * width,)
    assert np.array_equal(offsets, ref_o)
    assert np.array_equal(counts, ref_c)
    assert np.array_equal(neighbors, ref_n)
    assert np.any(counts > 0)
    if 2 * n_pairs < num_leaves:
        assert np.any(counts == 0)  # the sparse case has leaves with no neighbour


@pytest.mark.parametrize(
    "width, n_pairs, count", [(64, 40, 40), (40, 40, 99), (30, 12, 9)]
)
def test_far_list_is_the_directed_target_sort(width, n_pairs, count, route):
    """The deterministic far list: (target, source)-sorted sources plus ROW OFFSETS.

    ``targets`` holds the ``total + 1`` row offsets; ``far_pair_targets`` expands
    them to the directed sort's targets (-1 padded). The M2L's presorted read of
    (sources, offsets) equals its sorting read of the expanded COO list.
    """
    from jaccpot.pallas.m2l_real_csr import csr_by_target
    from jaccpot.runtime._interaction_cache import (
        TargetSortedFarPairs,
        far_pair_targets,
    )

    num_leaves = 30
    num_internal = num_leaves - 1
    total = num_internal + num_leaves
    rng = np.random.default_rng(width + n_pairs)
    seen = set()
    while len(seen) < n_pairs:
        a, b = sorted(rng.integers(0, total, size=2).tolist())
        if a != b:
            seen.add((a, b))
    pairs = np.asarray(sorted(seen, key=lambda _: rng.random()), np.int64)
    pad = max(width - n_pairs, 0)
    fa = jnp.asarray(np.concatenate([pairs[:, 0], np.full(pad, -1)]), jnp.int32)
    fb = jnp.asarray(np.concatenate([pairs[:, 1], np.full(pad, -1)]), jnp.int32)
    near = jnp.zeros((4,), jnp.int32)
    out = _flat_walk_lists(
        fa,
        fb,
        jnp.asarray(count, jnp.int32),
        near,
        near,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(False),
        far_width=width,
        near_width=4,
        num_internal=num_internal,
        total_nodes=total,
        deterministic=True,
        idx=jnp.int32,
    )
    far = TargetSortedFarPairs(*out[:4])
    src, offsets, n_live = (np.asarray(x) for x in (out[0], out[1], out[3]))
    tgt = np.asarray(far_pair_targets(far))
    live = min(count, width)
    a = np.asarray(fa)[:width][:live]
    b = np.asarray(fb)[:width][:live]
    ref_t = np.concatenate([a, b])
    ref_s = np.concatenate([b, a])
    order = np.lexsort((ref_s, ref_t))
    k = 2 * live
    assert int(n_live) == 2 * min(count, width) or count > width
    assert src.shape == tgt.shape == (2 * width,)
    assert np.array_equal(tgt[:k], ref_t[order]) and np.array_equal(
        src[:k], ref_s[order]
    )
    assert np.all(src[k:] == -1) and np.all(tgt[k:] == -1)
    assert np.array_equal(
        offsets, np.searchsorted(ref_t[order], np.arange(total + 1), side="left")
    )
    # the M2L's CSR: presorted read of (sources, offsets) == sorting read of COO
    # jax arrays: the runtime type-check job holds csr_by_target to its annotations
    src_j, tgt_j, off_j = (jnp.asarray(x) for x in (src, tgt, offsets))
    srt = csr_by_target(
        src_j, tgt_j, total_nodes=total, active_pair_count=jnp.asarray(k)
    )
    pre = csr_by_target(src_j, off_j, total_nodes=total, presorted=True)
    assert np.array_equal(np.asarray(pre[0])[:k], np.asarray(srt[0])[:k])
    for x, y in zip(srt[1:], pre[1:]):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_targets_from_csr_offsets_inverts_the_row_offsets():
    """Empty rows anywhere (first, inner, last), a full list, and padding."""
    from jaccpot.pallas.m2l_real_csr import targets_from_csr_offsets

    counts = np.array([0, 3, 0, 0, 1, 2, 0], np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    want = np.repeat(np.arange(counts.size), counts)
    for pad in (0, 5):
        got = np.asarray(
            targets_from_csr_offsets(jnp.asarray(offsets), want.size + pad)
        )
        assert np.array_equal(got[: want.size], want)
        assert np.all(got[want.size :] == -1)
    empty = np.asarray(
        targets_from_csr_offsets(jnp.zeros((4,), jnp.int32), 3)
    )  # no live entry at all
    assert np.all(empty == -1)


def test_pallas_route_equals_the_sorted_route_on_gpu(monkeypatch):
    """Native atomics + ranks == the two sorts, bit for bit, at a realistic size.

    Long rows (several rank tiles), empty rows, a live prefix and padding; the
    atomics leave the rows in arbitrary order and the ranks must undo it.
    """
    from jaccpot.pallas.csr_place import pallas_directed_csr_supported

    if not pallas_directed_csr_supported():
        pytest.skip("native Pallas GPU lowering not available here")
    rng = np.random.default_rng(7)
    R, n_pairs, width = 20000, 300000, 400000
    a = rng.integers(0, R, 2 * n_pairs)
    b = rng.integers(0, R, 2 * n_pairs)
    hub = rng.random(2 * n_pairs) < 0.01  # a few rows thousands long
    a = np.where(hub, 17, a)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keep = lo != hi
    pairs = np.unique(np.stack([lo[keep], hi[keep]], 1), axis=0)[:n_pairs]
    pairs = pairs[rng.permutation(pairs.shape[0])]
    n = pairs.shape[0]
    pad = width - n
    fa = jnp.asarray(np.concatenate([pairs[:, 0], np.full(pad, 5)]), jnp.int32)
    fb = jnp.asarray(np.concatenate([pairs[:, 1], np.full(pad, 9)]), jnp.int32)
    live = jnp.arange(width, dtype=jnp.int32) < n

    def build():
        return _directed_csr_from_canonical(
            fa,
            fb,
            live,
            row_offset=0,
            num_rows=R,
            idx=jnp.int32,
            pad_source=-1,
            with_targets=True,
        )

    monkeypatch.setenv("JACCPOT_LIST_CSR_KERNEL", "xla")
    want = [np.asarray(x) for x in build()]
    monkeypatch.setenv("JACCPOT_LIST_CSR_KERNEL", "pallas")
    got = [np.asarray(x) for x in build()]
    assert int(want[3].max()) > 1000  # the hub row spans many tiles
    for g, w in zip(got, want):
        assert np.array_equal(g, w)
