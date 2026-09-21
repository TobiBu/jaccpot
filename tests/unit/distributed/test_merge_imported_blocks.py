"""`merge_imported_blocks`: the near import rides BEHIND the far import, shifted by its capacity.

Two capacity-padded payloads become one imported block for the M2L. The one thing
that can go quietly wrong is the shift of the near block's source rows: it must be
the far block's PADDED length, since the far payload's dead rows remain rows of the
concatenated array. A shift by the live count instead would alias near-leaf pairs
onto unrelated far nodes -- right shapes, plausible force, wrong answer -- so the
test pins the shift by reading the multipole each pair points at.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from jaccpot.distributed.cross import merge_imported_blocks

N_LOCAL = 37
FAR_CAP, NEAR_CAP = 8, 6  # payload capacities (rows), deliberately unequal
NC = 5  # coefficients


def _blocks(rng, far_live, near_live, far_pairs, near_pairs):
    far_mp = rng.normal(size=(FAR_CAP, NC))
    far_mp[far_live:] = 0.0
    near_mp = rng.normal(size=(NEAR_CAP, NC))
    near_mp[near_live:] = 0.0
    far_cen = rng.normal(size=(FAR_CAP, 3))
    near_cen = rng.normal(size=(NEAR_CAP, 3))
    P = 10  # pair capacity
    fs = np.full(P, 99, np.int32)
    ft = np.full(P, 99, np.int32)
    ns = np.full(P, 99, np.int32)
    nt = np.full(P, 99, np.int32)
    for i, (t, r) in enumerate(far_pairs):
        ft[i], fs[i] = t, r
    for i, (t, r) in enumerate(near_pairs):
        nt[i], ns[i] = t, r
    return (
        jnp.asarray(far_mp), jnp.asarray(far_cen), jnp.asarray(fs), jnp.asarray(ft),
        jnp.asarray(len(far_pairs), jnp.int32),
        jnp.asarray(near_mp), jnp.asarray(near_cen), jnp.asarray(ns), jnp.asarray(nt),
        jnp.asarray(len(near_pairs), jnp.int32),
    )


def test_every_pair_points_at_its_own_multipole():
    rng = np.random.default_rng(3)
    far_pairs = [(4, 0), (4, 2), (9, 1)]
    near_pairs = [(4, 0), (11, 3), (12, 3), (12, 5)]
    args = _blocks(rng, 3, 6, far_pairs, near_pairs)
    mp, cen, src, tgt = merge_imported_blocks(*args, n_local=N_LOCAL)
    mp, cen, src, tgt = map(np.asarray, (mp, cen, src, tgt))
    far_mp, far_cen, *_ = args
    near_mp, near_cen = args[5], args[6]

    assert mp.shape == (FAR_CAP + NEAR_CAP, NC)
    assert cen.shape == (FAR_CAP + NEAR_CAP, 3)
    live = src >= 0
    assert live.sum() == len(far_pairs) + len(near_pairs)
    assert np.all(tgt[live] >= 0) and np.all(tgt[~live] == -1)
    # every live source sits ABOVE the local block
    assert np.all(src[live] >= N_LOCAL)
    # the far pairs read the far multipoles ...
    for i, (t, r) in enumerate(far_pairs):
        assert tgt[i] == t
        np.testing.assert_array_equal(mp[src[i] - N_LOCAL], np.asarray(far_mp)[r])
        np.testing.assert_array_equal(cen[src[i] - N_LOCAL], np.asarray(far_cen)[r])
    # ... and the near pairs the near multipoles, behind the far block's CAPACITY
    for j, (t, r) in enumerate(near_pairs):
        k = 10 + j  # the near pair capacity follows the far pair capacity
        assert tgt[k] == t
        assert src[k] == N_LOCAL + FAR_CAP + r
        np.testing.assert_array_equal(mp[src[k] - N_LOCAL], np.asarray(near_mp)[r])
        np.testing.assert_array_equal(cen[src[k] - N_LOCAL], np.asarray(near_cen)[r])


def test_shift_is_the_far_capacity_not_the_live_count():
    """CONTROL: with only 3 live far rows, a live-count shift would land near
    row 0 on far row 3 -- a dead but existing row. The correct shift does not."""
    rng = np.random.default_rng(5)
    args = _blocks(rng, 3, 6, [(1, 0)], [(2, 0)])
    mp, _, src, _ = merge_imported_blocks(*args, n_local=N_LOCAL)
    near_mp = np.asarray(args[5])
    wrong = N_LOCAL + 3 + 0
    right = N_LOCAL + FAR_CAP + 0
    assert int(src[10]) == right != wrong
    assert np.all(np.asarray(mp)[wrong - N_LOCAL] == 0.0)  # what the wrong shift reads
    np.testing.assert_array_equal(np.asarray(mp)[right - N_LOCAL], near_mp[0])


def test_dead_slots_are_minus_one_whatever_they_held():
    """Capacity beyond `count` carries garbage (99 here) and must come out as -1."""
    rng = np.random.default_rng(7)
    args = _blocks(rng, 2, 2, [(0, 0)], [])
    _, _, src, tgt = merge_imported_blocks(*args, n_local=N_LOCAL)
    src, tgt = np.asarray(src), np.asarray(tgt)
    assert src.shape == (20,)
    assert (src >= 0).sum() == 1 and (tgt >= 0).sum() == 1
    assert np.all(src[1:] == -1) and np.all(tgt[1:] == -1)


def test_near_block_dtype_follows_far_block():
    rng = np.random.default_rng(9)
    args = list(_blocks(rng, 1, 1, [(0, 0)], [(0, 0)]))
    args[0] = jnp.asarray(args[0], jnp.float32)
    args[5] = jnp.asarray(args[5], jnp.float64)
    mp, cen, src, tgt = merge_imported_blocks(*args, n_local=N_LOCAL)
    assert mp.dtype == jnp.float32
    assert src.dtype == jnp.asarray(args[2]).dtype
