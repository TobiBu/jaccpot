"""The CSR near-field kernels reading sorted particle ranges equal the table kernel.

``nearfield_leafpair_csr_sorted_pallas`` reads a leaf's particles as the range
``[start, start + count)`` of the sorted array instead of gathered ``(L, W)``
tables. Same lane body, loop bounds and summation order, so the result must be
the same to the bit: on leaves of every occupancy, empty padding leaves at the
end, rows past every leaf's particles (a shard's padding), a subtile that pads
``W``, and chunks that split rows.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.nearfield_leafpair_csr import (
    build_leafpair_chunk_table,
    leafpair_chunk_capacity,
    nearfield_leafpair_csr_pallas,
    nearfield_leafpair_csr_sorted_direct_pallas,
    nearfield_leafpair_csr_sorted_pallas,
)


def _case(seed, *, num_live, num_pad, W, n_dead, max_row, empty_rows=()):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, W + 1, size=num_live)
    counts[0] = W  # one full leaf
    counts = np.concatenate([counts, np.zeros(num_pad, np.int64)])
    L = counts.size
    n_live = int(counts.sum())
    n = n_live + n_dead
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    starts[num_live:] = n_live  # empty padding leaves start past the live rows
    pos = rng.standard_normal((n, 3)).astype(np.float32)
    mass = (np.abs(rng.standard_normal(n)) + 0.1).astype(np.float32)
    # the table path's gathered (L, W) tables, live slots a prefix
    slot = np.arange(W)[None, :]
    mask = slot < counts[:, None]
    ids = np.where(mask, starts[:, None] + slot, 0)
    # neighbour rows over the live leaves, never the row's own leaf
    row_counts = rng.integers(0, max_row + 1, size=L)
    row_counts[num_live:] = 0
    for r in empty_rows:
        row_counts[r] = 0
    rows = [
        rng.choice(np.setdiff1d(np.arange(num_live), [leaf]), size=row_counts[leaf])
        for leaf in range(L)
    ]
    nbr = np.concatenate(rows).astype(np.int32)
    nbr = np.concatenate([nbr, np.zeros(5, np.int32)])  # padded capacity
    offsets = np.concatenate([[0], np.cumsum(row_counts)]).astype(np.int32)
    return dict(
        pos=jnp.asarray(pos),
        mass=jnp.asarray(mass),
        leaf_pos=jnp.asarray(pos[ids]),
        leaf_mass=jnp.asarray(mass[ids]),
        mask=jnp.asarray(mask),
        starts=jnp.asarray(starts, jnp.int32),
        counts=jnp.asarray(counts, jnp.int32),
        nbr=jnp.asarray(nbr),
        offsets=jnp.asarray(offsets),
        row_counts=jnp.asarray(row_counts, jnp.int32),
        L=L,
    )


@pytest.mark.parametrize(
    "W, chunk, subtile, n_dead, accum",
    [
        (8, 3, None, 0, "input"),
        (8, 1, 4, 5, "input"),
        (6, 2, 4, 3, "input"),  # W padded to 8 lanes
        (8, 4, None, 2, "wide"),
    ],
)
def test_sorted_ranges_equal_the_table_kernel(W, chunk, subtile, n_dead, accum):
    c = _case(7, num_live=9, num_pad=3, W=W, n_dead=n_dead, max_row=6, empty_rows=(2,))
    cap = leafpair_chunk_capacity(int(c["nbr"].shape[0]), c["L"], chunk)
    tab = build_leafpair_chunk_table(
        c["offsets"], c["row_counts"], chunk=chunk, capacity=cap
    )
    common = dict(
        softening_sq=jnp.float32(0.05**2),
        G=jnp.float32(1.3),
        chunk=chunk,
        target_subtile=subtile,
        interpret=True,
        accum=accum,
        include_self=True,
    )
    ref = nearfield_leafpair_csr_pallas(
        c["leaf_pos"], c["leaf_mass"], c["mask"], c["nbr"], tab, **common
    )
    got = nearfield_leafpair_csr_sorted_pallas(
        c["pos"],
        c["mass"],
        c["starts"],
        c["counts"],
        c["nbr"],
        tab,
        leaf_width=W,
        **common,
    )
    ref = np.asarray(ref)
    got = np.asarray(got)
    assert got.shape == ref.shape == (c["L"], W, 4)
    assert np.any(ref[:9]) and not np.any(ref[9:])  # padding leaves stay zero
    assert np.array_equal(got, ref)


@pytest.mark.parametrize(
    "W, chunk, subtile, n_dead, with_potential",
    [
        (8, 3, None, 0, False),
        (8, 1, 4, 5, True),  # rows of up to 6 chunks
        (6, 2, 4, 3, False),  # W padded to 8 lanes
        (8, 64, None, 2, True),  # every row one chunk: no second pass at all
    ],
)
@pytest.mark.parametrize("rows", ["chunked", "whole"])
def test_direct_equals_the_table_kernel_in_particle_order(
    W, chunk, subtile, n_dead, with_potential, rows, monkeypatch
):
    """The direct lane, gathered back: the table kernel's values.

    ``chunked``: on CPU the scatter of pass 2 adds a row's chunks in chunk
    order, which is the segment sum's order, so even rows of three or more
    chunks agree to the bit here (on GPU both are unordered atomics).
    ``whole``: one running sum per row, so rows of one chunk agree to the bit
    and longer rows to single-precision round-off.
    """
    monkeypatch.setenv("JACCPOT_NEARFIELD_DIRECT_ROWS", rows)
    c = _case(7, num_live=9, num_pad=3, W=W, n_dead=n_dead, max_row=6, empty_rows=(2,))
    cap = leafpair_chunk_capacity(int(c["nbr"].shape[0]), c["L"], chunk)
    tab = build_leafpair_chunk_table(
        c["offsets"], c["row_counts"], chunk=chunk, capacity=cap
    )
    common = dict(
        softening_sq=jnp.float32(0.05**2),
        G=jnp.float32(1.3),
        chunk=chunk,
        target_subtile=subtile,
        interpret=True,
    )
    ref = np.asarray(
        nearfield_leafpair_csr_pallas(
            c["leaf_pos"], c["leaf_mass"], c["mask"], c["nbr"], tab, **common
        )
    )
    acc, pot = nearfield_leafpair_csr_sorted_direct_pallas(
        c["pos"],
        c["mass"],
        c["starts"],
        c["counts"],
        c["nbr"],
        c["offsets"],
        c["row_counts"],
        leaf_width=W,
        with_potential=with_potential,
        source_tile=0,  # the scalar loop: the table kernel's sums, op for op
        **common,
    )
    n = c["pos"].shape[0]
    counts = np.asarray(c["counts"])
    starts = np.asarray(c["starts"])
    want = np.zeros((n, 4), np.float32)
    for leaf in range(c["L"]):
        want[starts[leaf] : starts[leaf] + counts[leaf]] = ref[leaf, : counts[leaf]]
    acc = np.asarray(acc)
    assert acc.shape == (n, 3)
    assert np.any(acc) and not np.any(acc[n - n_dead :])
    got = np.concatenate(
        [acc, np.asarray(pot)[:, None] if with_potential else want[:, 3:]], axis=1
    )
    if rows == "chunked":
        exact = np.ones(n, bool)
    else:  # particles of leaves whose row is one chunk keep the table's bits
        row_counts = np.asarray(c["row_counts"])
        exact = np.zeros(n, bool)
        for leaf in range(c["L"]):
            if row_counts[leaf] <= chunk:
                exact[starts[leaf] : starts[leaf] + counts[leaf]] = True
        np.testing.assert_allclose(got, want, rtol=2e-6, atol=1e-6)
    assert np.array_equal(got[exact], want[exact])
    if not with_potential:
        assert pot is None
    if chunk < 6:  # some row really is split (non-vacuity of pass 2)
        assert int(np.asarray(c["row_counts"]).max()) > chunk


@pytest.mark.parametrize("row_limit", [1, 2, 4])
def test_whole_rows_past_the_limit_go_in_pieces(row_limit, monkeypatch):
    """``whole`` mode runs each row up to ``row_limit`` entries in its own program
    and the rest in pieces: the unlimited run's values to single-precision
    round-off, and the bits of every row that fits the limit."""
    monkeypatch.setenv("JACCPOT_NEARFIELD_DIRECT_ROWS", "whole")
    c = _case(7, num_live=9, num_pad=3, W=8, n_dead=2, max_row=6, empty_rows=(2,))
    common = dict(
        softening_sq=jnp.float32(0.05**2),
        G=jnp.float32(1.3),
        chunk=1,
        target_subtile=None,
        interpret=True,
        leaf_width=8,
    )
    args = (
        c["pos"],
        c["mass"],
        c["starts"],
        c["counts"],
        c["nbr"],
        c["offsets"],
        c["row_counts"],
    )
    full, _ = nearfield_leafpair_csr_sorted_direct_pallas(
        *args, row_limit=1 << 20, source_tile=0, **common
    )
    lim, _ = nearfield_leafpair_csr_sorted_direct_pallas(
        *args, row_limit=row_limit, source_tile=0, **common
    )
    full, lim = np.asarray(full), np.asarray(lim)
    np.testing.assert_allclose(lim, full, rtol=2e-6, atol=1e-6)
    row_counts = np.asarray(c["row_counts"])
    counts = np.asarray(c["counts"])
    starts = np.asarray(c["starts"])
    fits = np.zeros(full.shape[0], bool)
    for leaf in range(c["L"]):
        if row_counts[leaf] <= row_limit:
            fits[starts[leaf] : starts[leaf] + counts[leaf]] = True
    assert np.array_equal(lim[fits], full[fits])
    assert int(row_counts.max()) > row_limit  # some row really goes in pieces


def _consecutive_case(seed, *, W):
    """``_case`` with rows of CONSECUTIVE leaves, so source runs really merge."""
    c = _case(seed, num_live=9, num_pad=3, W=W, n_dead=2, max_row=6, empty_rows=(2,))
    row_counts = np.asarray(c["row_counts"])
    rows = []
    for leaf in range(c["L"]):
        others = [x for x in range(9) if x != leaf]
        k = int(row_counts[leaf])
        lo = (leaf * 3) % max(1, len(others) - k + 1)
        rows.append(np.asarray(others[lo : lo + k], np.int32))
    nbr = np.concatenate(rows + [np.zeros(5, np.int32)])
    assert int(np.sum(np.diff(nbr[: int(row_counts.sum())]) == 1)) >= 10
    return dict(c, nbr=jnp.asarray(nbr))


@pytest.mark.parametrize(
    "W, subtile, source_tile, flags, with_potential, classes",
    [
        (8, None, 4, "", True, ()),
        (8, 4, 8, "p", False, ()),
        (6, 4, 2, "a", True, ()),  # W padded to 8 lanes
        (16, 8, 4, "r", True, ()),
        (16, 16, 32, "apr", True, ()),  # source tile wider than a leaf
        (8, 8, 4, "ar", False, ()),
        (16, None, 8, "al", True, (4, 8, 16)),  # one launch per occupancy class
        (16, None, 4, "alr", False, (2, 8)),  # the widest class in two subtiles
        (8, None, 8, "l", True, (8,)),
        (16, 8, 8, "alg", True, ()),  # 2D-indexed operands
        (16, None, 4, "aglr", True, (4, 16)),
    ],
)
@pytest.mark.parametrize("row_limit", [1 << 20, 2])
def test_source_tiles_equal_the_scalar_loop(
    W, subtile, source_tile, flags, with_potential, classes, row_limit
):
    """Vector source tiles: the scalar loop's values to single-precision round-off
    (a tile is summed as a tree), with and without the pieces of long rows, on rows
    whose leaves are consecutive (runs merge) and on random rows; per-class launches
    the one-launch tiled kernel's values exactly."""
    for c in (
        _case(7, num_live=9, num_pad=3, W=W, n_dead=2, max_row=6, empty_rows=(2,)),
        _consecutive_case(11, W=W),
    ):
        args = (
            c["pos"],
            c["mass"],
            c["starts"],
            c["counts"],
            c["nbr"],
            c["offsets"],
            c["row_counts"],
        )
        common = dict(
            leaf_width=W,
            softening_sq=jnp.float32(0.05**2),
            G=jnp.float32(1.3),
            chunk=1,
            target_subtile=subtile,
            interpret=True,
            with_potential=with_potential,
            row_limit=row_limit,
        )
        acc0, pot0 = nearfield_leafpair_csr_sorted_direct_pallas(
            *args, source_tile=0, **common
        )
        acc1, pot1 = nearfield_leafpair_csr_sorted_direct_pallas(
            *args,
            source_tile=source_tile,
            source_flags=flags,
            target_classes=classes,
            **common,
        )
        if classes:
            one, _ = nearfield_leafpair_csr_sorted_direct_pallas(
                *args,
                source_tile=source_tile,
                source_flags=flags,
                target_classes=(),
                **common,
            )
            assert np.array_equal(np.asarray(acc1), np.asarray(one))
        acc0, acc1 = np.asarray(acc0), np.asarray(acc1)
        assert np.any(acc0)
        n_dead = 2
        assert not np.any(acc1[-n_dead:])
        # absolute round-off on the scale of the largest force: a component can
        # be a small difference of large terms
        np.testing.assert_allclose(
            acc1, acc0, rtol=2e-6, atol=2e-6 * float(np.abs(acc0).max())
        )
        if with_potential:
            pot0 = np.asarray(pot0)
            np.testing.assert_allclose(
                np.asarray(pot1), pot0, rtol=2e-6, atol=2e-6 * float(np.abs(pot0).max())
            )
        else:
            assert pot1 is None


def test_source_flags_are_validated():
    """Bad tile options are refused, naming the option."""
    c = _case(7, num_live=4, num_pad=0, W=4, n_dead=0, max_row=2)
    with pytest.raises(ValueError, match="source_flags"):
        nearfield_leafpair_csr_sorted_direct_pallas(
            c["pos"],
            c["mass"],
            c["starts"],
            c["counts"],
            c["nbr"],
            c["offsets"],
            c["row_counts"],
            leaf_width=4,
            softening_sq=jnp.float32(0.01),
            G=jnp.float32(1.0),
            chunk=2,
            interpret=True,
            source_tile=4,
            source_flags="x",
        )
    with pytest.raises(ValueError, match="power of two"):
        nearfield_leafpair_csr_sorted_direct_pallas(
            c["pos"],
            c["mass"],
            c["starts"],
            c["counts"],
            c["nbr"],
            c["offsets"],
            c["row_counts"],
            leaf_width=4,
            softening_sq=jnp.float32(0.01),
            G=jnp.float32(1.0),
            chunk=2,
            interpret=True,
            source_tile=6,
        )
    with pytest.raises(ValueError, match="powers of two"):
        nearfield_leafpair_csr_sorted_direct_pallas(
            c["pos"],
            c["mass"],
            c["starts"],
            c["counts"],
            c["nbr"],
            c["offsets"],
            c["row_counts"],
            leaf_width=4,
            softening_sq=jnp.float32(0.01),
            G=jnp.float32(1.0),
            chunk=2,
            interpret=True,
            source_tile=4,
            target_classes=(3,),
        )


def test_the_default_is_the_tiled_kernel(monkeypatch):
    """No options and no environment: 8-source tiles, flags ``alr``, 16-lane
    targets -- the same bits as asking for them."""
    from jaccpot.pallas import nearfield_leafpair_csr as mod

    for var in (
        "JACCPOT_NEARFIELD_SOURCE_TILE",
        "JACCPOT_NEARFIELD_SOURCE_FLAGS",
        "JACCPOT_NEARFIELD_TARGET_CLASSES",
    ):
        monkeypatch.delenv(var, raising=False)
    c = _case(7, num_live=9, num_pad=3, W=16, n_dead=2, max_row=6, empty_rows=(2,))
    args = (
        c["pos"],
        c["mass"],
        c["starts"],
        c["counts"],
        c["nbr"],
        c["offsets"],
        c["row_counts"],
    )
    common = dict(
        leaf_width=16,
        softening_sq=jnp.float32(0.05**2),
        G=jnp.float32(1.3),
        chunk=4,
        interpret=True,
        with_potential=True,
    )
    got = nearfield_leafpair_csr_sorted_direct_pallas(*args, **common)
    want = nearfield_leafpair_csr_sorted_direct_pallas(
        *args,
        source_tile=mod.DIRECT_SOURCE_TILE,
        source_flags=mod.DIRECT_SOURCE_FLAGS,
        target_subtile=mod.DIRECT_TILED_TARGET_SUBTILE,
        **common,
    )
    scalar = nearfield_leafpair_csr_sorted_direct_pallas(*args, source_tile=0, **common)
    assert (mod.DIRECT_SOURCE_TILE, mod.DIRECT_SOURCE_FLAGS) == (8, "alr")
    for g, w, s in zip(got, want, scalar):
        assert np.array_equal(np.asarray(g), np.asarray(w))
        assert not np.array_equal(np.asarray(g), np.asarray(s))  # non-vacuous
