"""The direct CSR near-field kernel, reading sorted particle ranges, equals the table kernel.

``nearfield_leafpair_csr_sorted_direct_pallas`` reads a leaf's particles as the
range ``[start, start + count)`` of the sorted array instead of gathered
``(L, W)`` tables, and stores straight into particle order. With the scalar
source loop it has the table kernel's lane body, loop bounds and summation order,
so rows of one chunk come out the same to the bit: on leaves of every occupancy,
empty padding leaves at the end, rows past every leaf's particles (a shard's
padding) and a subtile that pads ``W``. The vector source tiles agree with the
scalar loop to single-precision round-off.

The ``sorted`` layout (ranges in, per-chunk partials out), the chunked rows, the
per-class target launches and the source flags ``g`` and ``p`` without ``r``
were removed in the 2026-10 cleanup (X5).
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
    "W, chunk, subtile, n_dead, with_potential",
    [
        (8, 3, None, 0, False),
        (8, 1, 4, 5, True),  # rows of up to 6 chunks
        (6, 2, 4, 3, False),  # W padded to 8 lanes
        (8, 64, None, 2, True),  # every row one chunk
    ],
)
def test_direct_equals_the_table_kernel_in_particle_order(
    W, chunk, subtile, n_dead, with_potential
):
    """The direct lane, gathered back: the table kernel's values.

    One running sum per row, so rows of one chunk agree to the bit and longer
    rows to single-precision round-off. (The ``chunked`` rows that kept the
    table's chunk partials were removed in the 2026-10 cleanup, X5.)
    """
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
    # particles of the leaves that keep the table's bits: rows of one chunk
    row_counts = np.asarray(c["row_counts"])
    bitwise = row_counts <= chunk
    exact = np.zeros(n, bool)
    for leaf in range(c["L"]):
        if bitwise[leaf]:
            exact[starts[leaf] : starts[leaf] + counts[leaf]] = True
    np.testing.assert_allclose(got, want, rtol=2e-6, atol=1e-6)
    assert np.array_equal(got[exact], want[exact])
    if not with_potential:
        assert pot is None
    if chunk < 6:  # some row really is longer than a chunk
        assert int(np.asarray(c["row_counts"]).max()) > chunk


@pytest.mark.parametrize("row_limit", [1, 2, 4])
def test_whole_rows_past_the_limit_go_in_pieces(row_limit):
    """Each row runs up to ``row_limit`` entries in its own program and the rest
    in pieces: the unlimited run's values to single-precision round-off, and the
    bits of every row that fits the limit."""
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
    "W, subtile, source_tile, flags, with_potential",
    [
        (8, None, 4, "", True),
        (6, 4, 2, "a", True),  # W padded to 8 lanes
        (16, 8, 4, "r", True),
        (16, 16, 32, "apr", True),  # source tile wider than a leaf; p rides on r
        (8, 8, 4, "ar", False),
        (16, None, 8, "al", True),
        (16, None, 4, "alr", False),  # the default flags
        (8, None, 8, "l", True),
    ],
)
@pytest.mark.parametrize("row_limit", [1 << 20, 2])
def test_source_tiles_equal_the_scalar_loop(
    W, subtile, source_tile, flags, with_potential, row_limit
):
    """Vector source tiles: the scalar loop's values to single-precision round-off
    (a tile is summed as a tree), with and without the pieces of long rows, on rows
    whose leaves are consecutive (runs merge) and on random rows. (The cases of the
    per-class launches and of the removed ``g`` and lone ``p`` flags went in the
    2026-10 cleanup, X5; the class cases' flag sets stay, in one launch.)"""
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
            **common,
        )
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
    """Bad tile options are refused, naming the option; so are the removed ones."""
    c = _case(7, num_live=4, num_pad=0, W=4, n_dead=0, max_row=2)
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
        leaf_width=4,
        softening_sq=jnp.float32(0.01),
        G=jnp.float32(1.0),
        chunk=2,
        interpret=True,
    )
    with pytest.raises(ValueError, match="source_flags"):
        nearfield_leafpair_csr_sorted_direct_pallas(
            *args, source_tile=4, source_flags="x", **common
        )
    with pytest.raises(ValueError, match="power of two"):
        nearfield_leafpair_csr_sorted_direct_pallas(*args, source_tile=6, **common)
    # `g` (the 2D-indexed operands, removed in X5) is no letter of the flags now
    for flags in ("alg", "g"):
        with pytest.raises(ValueError, match="takes the letters a, l, p, r"):
            nearfield_leafpair_csr_sorted_direct_pallas(
                *args, source_tile=4, source_flags=flags, **common
            )


def test_the_default_is_the_tiled_kernel(monkeypatch):
    """No options and no environment: 8-source tiles, flags ``alr``, 16-lane
    targets -- the same bits as asking for them."""
    from jaccpot.pallas import nearfield_leafpair_csr as mod

    for var in (
        "JACCPOT_NEARFIELD_SOURCE_TILE",
        "JACCPOT_NEARFIELD_SOURCE_FLAGS",
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


# ------------------------------------------------------------ the ferrers3 tile skip

_SKIP_EPS = 0.05  # h = 0.123 against the unit-normal positions of ``_case``


def _ferrers3_params(eps=_SKIP_EPS):
    from jaccpot.softening import softening_params_from_sq

    return softening_params_from_sq("ferrers3", jnp.float32(eps**2), jnp.float32)


def _first_beyond(params):
    """The smallest float32 ``r^2`` whose rounded ``r^2 / h^2`` is 1."""
    p0, p1 = (np.float32(v) for v in np.asarray(params))
    v = min(p0, np.float32(1.0) / p1)
    for _ in range(8):
        v = np.nextafter(v, np.float32(0.0))
    while np.float32(v * p1) < np.float32(1.0):
        v = np.nextafter(v, np.float32(np.inf))
    return v


def test_the_newtonian_factors_are_the_polynomials_bits_beyond_the_support():
    """On pairs whose rounded ``r^2 / h^2`` reaches 1 -- from the first float that
    does, up -- the Newtonian factors are ``masked_pair_factors``' force and potential
    factors to the bit; dead pairs (some inside the support) are zero in both."""
    import jax

    from jaccpot.pallas.nearfield_leafpair_csr import _ferrers3_newton_factors
    from jaccpot.softening import masked_pair_factors

    params = _ferrers3_params()
    newton = jax.jit(
        lambda r, a: _ferrers3_newton_factors(r, a, params, "ferrers3", potential=True)
    )
    plain = jax.jit(
        lambda r, a: masked_pair_factors(r, a, params, "ferrers3", potential=True)
    )
    first = _first_beyond(params)
    walk = [first]
    for _ in range(31):
        walk.append(np.nextafter(walk[-1], np.float32(np.inf)))
    rng = np.random.default_rng(0)
    far = (first * np.exp(rng.uniform(0.0, 12.0, 64))).astype(np.float32)
    r = np.concatenate([np.asarray(walk, np.float32), far, np.zeros(32, np.float32)])
    live = np.arange(r.size) < 96  # the last 32: dead, at r = 0
    r, live = r.reshape(16, 8), live.reshape(16, 8)
    got = newton(jnp.asarray(r), jnp.asarray(live))
    want = plain(jnp.asarray(r), jnp.asarray(live))
    assert got[2] is None and want[2] is None
    for g, w in zip(got[:2], want[:2]):
        assert np.array_equal(np.asarray(g), np.asarray(w))
    # the walk starts at the boundary: one rounding below, c < 1
    p1 = np.float32(np.asarray(params)[1])
    assert np.float32(np.nextafter(first, np.float32(0.0)) * p1) < np.float32(1.0)


def _leaf_table(leaves):
    """``(pm, start, count)`` of consecutive leaves, the kernel's sorted layout."""
    pos = np.concatenate(leaves).astype(np.float32)
    counts = np.array([len(x) for x in leaves], np.int32)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)
    pm = np.concatenate([pos, np.ones((len(pos), 1), np.float32)], axis=1)
    pm = np.pad(pm, ((0, 64), (0, 0)))
    return jnp.asarray(pm), jnp.asarray(starts), jnp.asarray(counts)


def test_the_leaf_boxes_are_the_particles_extremes():
    """Each leaf's box is its particles' min and max per axis; an empty leaf's is
    ``(inf, -inf)``; leaves past ``L`` pad the table to the kernel's block."""
    from jaccpot.pallas.nearfield_leafpair_csr import _leaf_boxes

    rng = np.random.default_rng(3)
    leaves = [rng.standard_normal((n, 3)) for n in (1, 7, 16, 0, 33, 64, 2)]
    pm, starts, counts = _leaf_table(leaves)
    boxes = np.asarray(_leaf_boxes(pm, starts, counts, interpret=True))
    assert boxes.shape == (128, 8)
    for leaf, x in enumerate(leaves):
        x = x.astype(np.float32)
        lo = x.min(0) if len(x) else np.full(3, np.inf, np.float32)
        hi = x.max(0) if len(x) else np.full(3, -np.inf, np.float32)
        assert np.array_equal(boxes[leaf, 0:6:2], lo)
        assert np.array_equal(boxes[leaf, 1:6:2], hi)
        assert not np.any(boxes[leaf, 6:])


def test_the_box_test_admits_no_pair_inside_the_support():
    """Whenever ``_beyond_support`` clears a source leaf, every pair with the target
    leaf has a rounded ``r^2 / h^2`` of at least 1 (the clipped ``c`` is 1, so the
    Newtonian factors are the polynomial's); and it clears leaves within 0.1 % of
    the support as well as refusing nearer ones."""
    import jax

    from jaccpot.pallas.nearfield_leafpair_csr import (
        _BEYOND_SUPPORT_MARGIN,
        _beyond_support,
    )

    params = _ferrers3_params()
    p1 = np.float32(np.asarray(params)[1])
    h = float(np.sqrt(np.asarray(params)[0]))
    rng = np.random.default_rng(5)
    test = jax.jit(
        lambda boxes, t_box: _beyond_support(boxes, 1, t_box, params[1]),
    )
    cleared, refused, tight = 0, 0, 0
    for trial in range(300):
        nt, ns = rng.integers(1, 17, 2)
        centre = rng.uniform(-50.0, 50.0, 3)  # off the origin: the roundings matter
        t = (centre + 0.05 * rng.standard_normal((nt, 3))).astype(np.float32)
        s = (centre + 0.05 * rng.standard_normal((ns, 3))).astype(np.float32)
        # the source cloud past the target box along one axis, by a gap around
        # the cleared edge, its nearest particle facing the target's farthest
        k = int(rng.integers(3))
        gap = h * np.sqrt(_BEYOND_SUPPORT_MARGIN) * (1.0 + rng.uniform(-4e-4, 4e-4))
        s[:, k] += np.float32(t[:, k].max() - s[:, k].min() + gap)
        ia, ib = int(np.argmax(t[:, k])), int(np.argmin(s[:, k]))
        s[ib, [a for a in range(3) if a != k]] = t[ia, [a for a in range(3) if a != k]]
        # the boxes as _leaf_boxes builds them (pinned above), rows 0 and 1
        boxes = np.zeros((2, 8), np.float32)
        boxes[:, 0:6:2] = [t.min(0), s.min(0)]
        boxes[:, 1:6:2] = [t.max(0), s.max(0)]
        t_box = tuple((boxes[0, 2 * a], boxes[0, 2 * a + 1]) for a in range(3))
        ok = bool(test(jnp.asarray(boxes), t_box))
        d = t[:, None, :] - s[None, :, :]
        r_sq = d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]
        c_one = np.all((r_sq * p1).astype(np.float32) >= np.float32(1.0))
        if ok:
            cleared += 1
            assert c_one, trial
            tight += bool(np.min(r_sq) * p1 < 1.001)
        else:
            refused += 1
    assert cleared > 50 and refused > 50 and tight > 10, (cleared, refused, tight)


def _boundary_case(params):
    """Leaf pairs ``(2k, 2k+1)``, each the other's only neighbour, whose closest pair
    is ``d`` apart (``dx = -d`` exactly, ``dy = dz = 0``) and whose boxes are ``d``
    apart, with ``d`` a few roundings either side of the box test's edge; every other
    pair is far beyond the support."""
    from jaccpot.pallas.nearfield_leafpair_csr import _BEYOND_SUPPORT_MARGIN

    p0 = float(np.asarray(params)[0])
    h = np.sqrt(p0)
    W, pairs = 8, 7
    pos, d = [], np.float32(h * np.sqrt(_BEYOND_SUPPORT_MARGIN))
    for _ in range(3):
        d = np.nextafter(d, np.float32(0.0))
    for k in range(pairs):
        lane = np.arange(W, dtype=np.float32) * np.float32(20.0 * h)
        a = np.stack(
            [np.zeros(W, np.float32), lane, np.full(W, 3.0 * k, np.float32)], 1
        )
        b = a.copy()
        b[:, 0] += d
        b[1:, 1] += np.float32(10.0 * h)  # only lane 0 of b sits next to a
        pos += [a, b]
        d = np.nextafter(d, np.float32(np.inf))
    pos = np.concatenate(pos).astype(np.float32)
    L = 2 * pairs
    counts = np.full(L, W, np.int32)
    starts = (np.arange(L) * W).astype(np.int32)
    nbr = np.arange(L, dtype=np.int32) ^ 1
    offsets = np.arange(L + 1, dtype=np.int32)
    mass = np.linspace(0.5, 1.5, pos.shape[0]).astype(np.float32)
    return (
        jnp.asarray(pos),
        jnp.asarray(mass),
        jnp.asarray(starts),
        jnp.asarray(counts),
        jnp.asarray(np.concatenate([nbr, np.zeros(5, np.int32)])),
        jnp.asarray(offsets),
        jnp.asarray(np.ones(L, np.int32)),
    ), W


def _skip_case(regime, eps):
    """``(args, W, chunk, n_dead)``: ``_case``'s leaves as compact clusters on a unit
    grid, unless the regime is ``"overlapping"`` (``_case``'s own leaves, each spread
    over the whole cloud, so no two boxes are apart)."""
    if regime == "boundary":
        args, W = _boundary_case(_ferrers3_params(eps))
        return args, W, 4, 0
    c = _case(7, num_live=9, num_pad=3, W=16, n_dead=2, max_row=6, empty_rows=(2,))
    pos = np.asarray(c["pos"]).copy()
    if regime != "overlapping":
        rng = np.random.default_rng(8)
        starts, counts = np.asarray(c["starts"]), np.asarray(c["counts"])
        for leaf in range(9):
            centre = np.array([leaf % 3, leaf // 3, 0.0])
            rows = slice(starts[leaf], starts[leaf] + counts[leaf])
            pos[rows] = centre + 0.05 * rng.standard_normal((counts[leaf], 3))
    c = dict(c, pos=jnp.asarray(pos.astype(np.float32)))
    args = tuple(
        c[k]
        for k in ("pos", "mass", "starts", "counts", "nbr", "offsets", "row_counts")
    )
    return args, 16, 4, 2


_REGIMES = [
    ("beyond", 1e-3),  # every leaf but a target's own lies beyond the support
    ("straddling", 0.35),  # h = 0.86 on the unit grid: leaves of both kinds
    ("inside", 5.0),  # every pair inside the support
    ("boundary", _SKIP_EPS),  # leaves a few roundings either side of the box edge
    ("overlapping", 1e-3),  # every pair beyond the support, no two boxes apart
]


@pytest.mark.parametrize("interpret", [True, False])
@pytest.mark.parametrize("flags", ["alr", "a"])  # merged runs, and leaf by leaf
@pytest.mark.parametrize("with_potential", [False, True])
@pytest.mark.parametrize("regime, eps", _REGIMES)
def test_the_direct_kernel_with_the_tile_skip(
    regime, eps, with_potential, flags, interpret
):
    """The direct kernel with and without the tile skip.

    On the Triton kernel the contract is the same bits (forces and potentials). In
    interpret mode the two agree to float32 round-off (XLA may fuse around the
    conditional differently; the default keeps the polynomial there).
    """
    from jaccpot.pallas.m2l_real_csr import pallas_m2l_real_csr_supported

    if not interpret and not pallas_m2l_real_csr_supported():
        pytest.skip("native Pallas GPU lowering not available here")
    args, W, chunk, n_dead = _skip_case(regime, eps)
    common = dict(
        leaf_width=W,
        softening_sq=jnp.float32(eps**2),
        G=jnp.float32(1.3),
        chunk=chunk,
        interpret=interpret,
        with_potential=with_potential,
        softening_kernel="ferrers3",
        source_tile=8,
        source_flags=flags,
    )
    plain = nearfield_leafpair_csr_sorted_direct_pallas(
        *args, tile_skip=False, **common
    )
    skip = nearfield_leafpair_csr_sorted_direct_pallas(*args, tile_skip=True, **common)
    acc0 = np.asarray(plain[0])
    assert np.any(acc0) and np.all(np.isfinite(acc0))
    if n_dead:
        assert not np.any(np.asarray(skip[0])[-n_dead:])
    for got, want in zip(skip, plain):
        if want is None:
            assert got is None
            continue
        got, want = np.asarray(got), np.asarray(want)
        if interpret:
            np.testing.assert_allclose(
                got, want, rtol=2e-6, atol=2e-6 * float(np.abs(want).max())
            )
        else:
            assert np.array_equal(got, want)
    # the regime is what it says: pair distances against the support
    pos = np.asarray(args[0], np.float64)
    h = float(np.sqrt(np.asarray(_ferrers3_params(eps))[0]))
    r = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    off = r[~np.eye(len(pos), dtype=bool)]
    if regime in ("beyond", "overlapping"):
        assert off.min() > h
    elif regime == "inside":
        assert off.max() < h
    elif regime == "boundary":
        # every pair beyond the support, the closest at the box test's edge
        from jaccpot.pallas.nearfield_leafpair_csr import _BEYOND_SUPPORT_MARGIN

        assert abs(off.min() / (h * np.sqrt(_BEYOND_SUPPORT_MARGIN)) - 1) < 1e-6
    else:
        assert off.min() < h < off.max()


@pytest.mark.parametrize("flags", ["alr", "a"])
def test_the_skip_takes_the_newtonian_pass_where_it_may(monkeypatch, flags):
    """With the Newtonian factors doubled (a marker), the skip doubles exactly the
    neighbour leaves' share when they all lie beyond the support -- a leaf's own
    pass keeps the polynomial -- and nothing when every pair lies inside it."""
    from jaccpot.pallas import nearfield_leafpair_csr as mod

    real = mod._ferrers3_newton_factors

    def doubled(*args, **kwargs):
        g, psi, d = real(*args, **kwargs)
        return 2.0 * g, (None if psi is None else 2.0 * psi), d

    monkeypatch.setattr(mod, "_ferrers3_newton_factors", doubled)
    for regime, eps in (("beyond", 1e-3), ("inside", 5.0)):
        args, W, chunk, _ = _skip_case(regime, eps)
        common = dict(
            leaf_width=W,
            softening_sq=jnp.float32(eps**2),
            G=jnp.float32(1.3),
            chunk=chunk,
            interpret=True,
            softening_kernel="ferrers3",
            source_tile=8,
            source_flags=flags,
        )
        plain = np.asarray(
            nearfield_leafpair_csr_sorted_direct_pallas(
                *args, tile_skip=False, **common
            )[0]
        )
        skip = np.asarray(
            nearfield_leafpair_csr_sorted_direct_pallas(
                *args, tile_skip=True, **common
            )[0]
        )
        no_rows = args[:6] + (jnp.zeros_like(args[6]),)
        own = np.asarray(
            nearfield_leafpair_csr_sorted_direct_pallas(
                *no_rows, tile_skip=False, **common
            )[0]
        )
        scale = 2e-6 * float(np.abs(plain).max())
        if regime == "beyond":
            np.testing.assert_allclose(skip, 2.0 * plain - own, rtol=2e-6, atol=scale)
            assert np.abs(plain - own).max() > 1e3 * scale  # the neighbours count
        else:
            np.testing.assert_allclose(skip, plain, rtol=2e-6, atol=scale)


def test_the_tile_skip_default(monkeypatch):
    """Off by default (``DIRECT_TILE_SKIP``: it slows the production disc);
    ``JACCPOT_NEARFIELD_TILE_SKIP=1`` turns it on for the Triton kernel, not in
    interpret mode; an explicit argument wins; only ``"ferrers3"`` with a source tile
    skips."""
    from jaccpot.pallas import nearfield_leafpair_csr as mod

    seen = []

    class _Stop(Exception):
        pass

    def _spy(*args, tile_skip, boxes, **kwargs):
        seen.append((tile_skip, boxes is not None))
        raise _Stop

    monkeypatch.setattr(mod, "_sorted_pallas_call", _spy)
    monkeypatch.setattr(mod, "_leaf_boxes", lambda pm, *a, **k: pm)
    monkeypatch.delenv("JACCPOT_NEARFIELD_TILE_SKIP", raising=False)
    c = _case(7, num_live=4, num_pad=0, W=8, n_dead=0, max_row=2)
    args = tuple(
        c[k]
        for k in ("pos", "mass", "starts", "counts", "nbr", "offsets", "row_counts")
    )
    common = dict(leaf_width=8, softening_sq=jnp.float32(0.01), G=jnp.float32(1.0))

    def _resolved(**kw):
        kw.setdefault("softening_kernel", "ferrers3")
        with pytest.raises(_Stop):
            nearfield_leafpair_csr_sorted_direct_pallas(*args, chunk=2, **common, **kw)
        skip, with_boxes = seen.pop()
        assert skip is with_boxes
        return skip

    assert mod.DIRECT_TILE_SKIP is False
    assert _resolved(interpret=False) is False
    assert _resolved(interpret=True, tile_skip=True) is True
    monkeypatch.setenv("JACCPOT_NEARFIELD_TILE_SKIP", "1")
    assert _resolved(interpret=False) is True
    assert _resolved(interpret=True) is False
    assert _resolved(interpret=False, tile_skip=False) is False
    assert _resolved(interpret=False, softening_kernel="plummer") is False
    assert _resolved(interpret=False, softening_kernel="wendland_c2") is False
    assert _resolved(interpret=False, source_tile=0) is False
