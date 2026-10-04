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

from jaccpot.runtime._interaction_cache import _flat_walk_lists


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
def test_canonical_build_equals_the_directed_sort(num_leaves, n_pairs, width, count):
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
def test_far_list_is_the_directed_target_sort(width, n_pairs, count):
    """The deterministic far list: (target, source)-sorted directed pairs, -1 padded.

    And the M2L's CSR read of it without a sort equals its sorting read.
    """
    from jaccpot.pallas.m2l_real_csr import csr_by_target

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
    src, tgt, _tags, n_live = (np.asarray(x) for x in out[:4])
    live = min(count, width)
    a = np.asarray(fa)[:width][:live]
    b = np.asarray(fb)[:width][:live]
    ref_t = np.concatenate([a, b])
    ref_s = np.concatenate([b, a])
    order = np.lexsort((ref_s, ref_t))
    k = 2 * live
    assert int(n_live) == 2 * min(count, width) or count > width
    assert np.array_equal(tgt[:k], ref_t[order]) and np.array_equal(
        src[:k], ref_s[order]
    )
    assert np.all(src[k:] == -1) and np.all(tgt[k:] == -1)
    # the M2L's CSR: presorted read == sorting read
    srt = csr_by_target(src, tgt, total_nodes=total, active_pair_count=jnp.asarray(k))
    pre = csr_by_target(
        src, tgt, total_nodes=total, active_pair_count=jnp.asarray(k), presorted=True
    )
    for x, y in zip(srt, pre):
        assert np.array_equal(np.asarray(x), np.asarray(y))
