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
