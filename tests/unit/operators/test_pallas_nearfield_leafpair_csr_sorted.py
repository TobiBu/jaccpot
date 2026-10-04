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
