"""The cross NEAR term on the direct layout against an independent direct sum.

``cross_near_acceleration_direct`` is the mesh force's cross near term since S11
session 2: the local near field's direct kernel over ONE sorted table -- this
device's sorted particles, then the flat receive buffer -- with the local leaves as
the only targets and no self term, stored into particle order and permuted back to
input rows. The table layout (``cross_near_acceleration``, the leaf-major pool) is
checked against the same reference in ``test_cross_near_acceleration.py``; here the
two layouts are built from the SAME particles, so they also agree with each other.

Everything runs on CPU: the Pallas kernel under interpret mode, float64 inputs, so
the sums agree with NumPy's to round-off. The reference is written out here, as in
the table layout's test: ``a = +G sum_j m_j (x_j - x_t) / (|x_j - x_t|^2 +
eps^2)^{3/2}`` over exactly the listed pairs.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from jaccpot.distributed.cross import (
    _near_import_tiles,
    cross_near_acceleration,
    cross_near_acceleration_direct,
)
from jaccpot.distributed.fused import _cross_near_layout

G = 1.3
SOFT = 0.07
NUM_INTERNAL = 5  # deliberately NOT L-1: a balanced tree would hide a row/node mixup
LOCAL_LIVE = [4, 6, 1, 0, 3]  # leaf 3 empty, as a shard's padding leaves are
IMPORTED_LIVE = [5, 2, 6]
WIDTH = 6


def _case(rng, pairs, *, spare=4, dead_rows=3):
    """The same particles in both layouts.

    Local: leaf r holds ``LOCAL_LIVE[r]`` consecutive SORTED rows; ``dead_rows``
    padding rows sit past them (zero mass, in no leaf). ``perm`` maps a sorted row
    to its input row. Imported: rows of ``IMPORTED_LIVE`` particles, flat in the
    receive buffer (``spare`` zero rows past them), and as W-wide tiles.
    """
    n_live = int(sum(LOCAL_LIVE))
    n = n_live + dead_rows
    pos = rng.normal(size=(n, 3))
    mass = np.concatenate([rng.uniform(0.5, 1.5, n_live), np.zeros(dead_rows)])
    starts = np.concatenate([[0], np.cumsum(LOCAL_LIVE)[:-1]])
    ends = starts + np.asarray(LOCAL_LIVE) - 1  # inclusive; empty: end < start
    perm = rng.permutation(n).astype(np.int32)

    total = int(sum(IMPORTED_LIVE))
    flat = np.zeros((total + spare, 4))
    flat[:total, :3] = rng.normal(size=(total, 3))
    flat[:total, 3] = rng.uniform(0.5, 1.5, total)
    r_count = np.asarray(IMPORTED_LIVE, np.int32)
    r_start = (np.cumsum(r_count) - r_count).astype(np.int32)

    P = 16  # pair capacity, deliberately larger than len(pairs)
    tgt = np.full(P, -1, np.int32)
    src = np.full(P, -1, np.int32)
    for i, (t, s) in enumerate(pairs):
        tgt[i] = t + NUM_INTERNAL  # targets arrive as NODE ids
        src[i] = s
    near = {
        "particles": jnp.asarray(flat),
        "row_start": jnp.asarray(r_start),
        "row_count": jnp.asarray(r_count),
        "leaf_width": WIDTH,
        "target_node": jnp.asarray(tgt),
        "source_row": jnp.asarray(src),
        "count": jnp.asarray(len(pairs), jnp.int32),
    }
    local = dict(
        pos=pos,
        mass=mass,
        ranges=np.stack([starts, ends], axis=1).astype(np.int32),
        perm=perm,
    )
    return near, local


def _run(near, local, **kw):
    out = cross_near_acceleration_direct(
        near,
        jnp.asarray(local["pos"]),
        jnp.asarray(local["mass"]),
        jnp.asarray(local["ranges"]),
        jnp.asarray(local["perm"]),
        NUM_INTERNAL,
        leaf_width=WIDTH,
        softening_sq=jnp.asarray(SOFT**2),
        G=jnp.asarray(G),
        chunk=8,
        interpret=True,
        softening_kernel=kw.pop("kernel", "plummer"),
        **kw,
    )
    if isinstance(out, tuple):
        return tuple(np.asarray(o, np.float64) for o in out)
    return np.asarray(out, np.float64)


def _pair_terms(pairs, near, local):
    """Per listed pair: the target's input rows, its positions, the sources'."""
    flat = np.asarray(near["particles"])
    r_start = np.asarray(near["row_start"])
    r_count = np.asarray(near["row_count"])
    for t, s in pairs:
        lo, hi = local["ranges"][t]
        rows = np.arange(lo, hi + 1)
        src = flat[r_start[s] : r_start[s] + r_count[s]]
        yield local["perm"][rows], local["pos"][rows], src[:, :3], src[:, 3]


def _reference(pairs, near, local):
    ref = np.zeros((local["pos"].shape[0], 3))
    for rows, tp, sp, sm in _pair_terms(pairs, near, local):
        d = sp[None, :, :] - tp[:, None, :]
        inv3 = (np.sum(d * d, axis=-1) + SOFT**2) ** -1.5
        ref[rows] += G * np.einsum("ij,ijk->ik", sm[None, :] * inv3, d)
    return ref


PAIRS = [
    [(0, 0)],
    [(1, 2), (4, 0)],
    [(0, 0), (0, 1), (0, 2), (2, 1), (4, 2)],  # several sources per target
]


@pytest.mark.parametrize("pairs", PAIRS, ids=["one", "two", "multi-source"])
def test_matches_direct_sum(pairs):
    rng = np.random.default_rng(7)
    near, local = _case(rng, pairs)
    got = _run(near, local)
    ref = _reference(pairs, near, local)
    assert np.any(ref != 0.0)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("pairs", PAIRS, ids=["one", "two", "multi-source"])
def test_agrees_with_the_table_layout(pairs):
    """The table layout on the same particles (its pool gathered from the sorted
    rows, its tiles from the flat import): the same sum to round-off."""
    rng = np.random.default_rng(11)
    near, local = _case(rng, pairs)
    ranges, perm = local["ranges"], local["perm"]
    L = ranges.shape[0]
    slot = np.arange(WIDTH)
    counts = np.maximum(ranges[:, 1] - ranges[:, 0] + 1, 0)
    mask = slot[None, :] < counts[:, None]
    sorted_rows = np.where(mask, ranges[:, :1] + slot[None, :], 0)
    pool_pos = np.where(mask[..., None], local["pos"][sorted_rows], 0.0)
    pool_mass = np.where(mask, local["mass"][sorted_rows], 0.0)
    table = np.asarray(
        cross_near_acceleration(
            near,
            jnp.asarray(pool_pos),
            jnp.asarray(pool_mass),
            jnp.asarray(mask),
            jnp.asarray(np.where(mask, perm[sorted_rows], 0)),
            local["pos"].shape[0],
            NUM_INTERNAL,
            softening_sq=jnp.asarray(SOFT**2),
            G=jnp.asarray(G),
            chunk=8,
            interpret=True,
            accum="input",
            softening_kernel="plummer",
        ),
        np.float64,
    )
    assert L == len(LOCAL_LIVE)
    np.testing.assert_allclose(_run(near, local), table, rtol=1e-10, atol=1e-12)


def test_the_import_tiles_are_the_flat_rows():
    """The table layout's tiles, laid out from the flat import, hold each row's
    particles in its leading slots and zero elsewhere."""
    rng = np.random.default_rng(13)
    near, _ = _case(rng, [(0, 0)])
    pos, mass, mask = (np.asarray(a) for a in _near_import_tiles(near))
    flat = np.asarray(near["particles"])
    for r, k in enumerate(IMPORTED_LIVE):
        lo = int(np.asarray(near["row_start"])[r])
        np.testing.assert_array_equal(pos[r, :k], flat[lo : lo + k, :3])
        np.testing.assert_array_equal(mass[r, :k], flat[lo : lo + k, 3])
        assert mask[r, :k].all() and not mask[r, k:].any()
        assert not pos[r, k:].any() and not mass[r, k:].any()


def test_untouched_and_padding_rows_stay_zero():
    """A local leaf in no pair, and a shard's padding rows, receive nothing -- and
    a target's own leaf is no source (no self term)."""
    rng = np.random.default_rng(17)
    pairs = [(1, 0)]
    near, local = _case(rng, pairs)
    got = _run(near, local)
    lo, hi = local["ranges"][1]
    touched = np.zeros(local["pos"].shape[0], bool)
    touched[local["perm"][lo : hi + 1]] = True
    assert np.all(got[~touched] == 0.0) and np.all(got[touched] != 0.0)
    np.testing.assert_allclose(
        got, _reference(pairs, near, local), rtol=1e-10, atol=1e-12
    )


def test_dropped_pairs_contribute_nothing():
    """Dead CSR slots past ``count``, a target past the local leaves and a source
    row past the import (an overflowed row buffer) are dropped, never clipped onto
    a real row or read past the import."""
    rng = np.random.default_rng(19)
    pairs = [(0, 1), (2, 2)]
    near, local = _case(rng, pairs)
    base = _run(near, local)
    tgt = np.asarray(near["target_node"]).copy()
    src = np.asarray(near["source_row"]).copy()
    # poison every dead slot with a pair that WOULD contribute if read
    tgt[len(pairs) :] = NUM_INTERNAL + 4
    src[len(pairs) :] = 0
    poisoned = dict(near, target_node=jnp.asarray(tgt), source_row=jnp.asarray(src))
    np.testing.assert_array_equal(_run(poisoned, local), base)
    # and live slots naming a target or a source outside their range
    tgt[2], src[2] = NUM_INTERNAL + 99, 1
    tgt[3], src[3] = NUM_INTERNAL + 0, len(IMPORTED_LIVE)
    tgt[4], src[4] = NUM_INTERNAL - 1, 1
    outside = dict(
        near,
        target_node=jnp.asarray(tgt),
        source_row=jnp.asarray(src),
        count=jnp.asarray(5, jnp.int32),
    )
    np.testing.assert_array_equal(_run(outside, local), base)


def test_a_row_past_the_receive_buffer_is_cut_to_it():
    """An overflowed receive buffer holds fewer particles than its last row claims
    (the force is flagged then): the row is cut to what lies inside the buffer."""
    rng = np.random.default_rng(23)
    pairs = [(0, 2), (4, 2)]
    near, local = _case(rng, pairs, spare=0)
    want = _reference(pairs, near, local)
    r_count = np.asarray(near["row_count"]).copy()
    r_count[-1] += 3  # three particles past the buffer's end
    claimed = dict(near, row_count=jnp.asarray(r_count))
    np.testing.assert_allclose(_run(claimed, local), want, rtol=1e-10, atol=1e-12)


def _reference_lane(pairs, near, local, lane, kernel):
    from jaccpot.softening import pair_factors, softening_params_np

    ref = np.zeros(local["pos"].shape[0])
    for rows, tp, sp, sm in _pair_terms(pairs, near, local):
        d2 = np.sum((sp[None, :, :] - tp[:, None, :]) ** 2, axis=-1)
        if lane == "force_scale":
            # eq (16b)'s near share, G sum m / (r^2 + eps^2), whatever the kernel
            ref[rows] += G * np.sum(sm[None, :] / (d2 + SOFT**2), axis=1)
        else:
            _, psi, _ = pair_factors(
                d2, softening_params_np(kernel, SOFT), kernel, potential=True, xp=np
            )
            ref[rows] -= G * np.sum(sm[None, :] * psi, axis=1)
    return ref


@pytest.mark.parametrize("lane", ["force_scale", "potential"])
@pytest.mark.parametrize("kernel", ["plummer", "ferrers3"])
def test_the_fourth_lane_is_the_near_pair_sum(lane, kernel):
    """``with_force_scale`` (eq 16b's ``f_b`` under ``dehnen_error``) and
    ``with_potential`` (the mesh potential): the same launch returns the imported
    leaves' share over exactly the pairs, and the acceleration is the plain
    call's, bitwise. Both at once is refused (one lane)."""
    rng = np.random.default_rng(29)
    pairs = PAIRS[-1]
    near, local = _case(rng, pairs)
    plain = _run(near, local, kernel=kernel)
    acc, got = _run(near, local, kernel=kernel, **{f"with_{lane}": True})
    np.testing.assert_array_equal(acc, plain)
    ref = _reference_lane(pairs, near, local, lane, kernel)
    assert np.any(ref != 0) and np.any(ref == 0)  # untouched leaves stay zero
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-14)
    with pytest.raises(ValueError, match="fourth lane"):
        _run(near, local, with_potential=True, with_force_scale=True)


def test_the_layout_follows_the_local_near_field(monkeypatch):
    """The mesh force's cross near term takes the direct layout on the default
    exchange's flat import, the table under ``JACCPOT_NEARFIELD_ACCUM=wide`` (the
    direct kernel accumulates in the input dtype) or ``JACCPOT_NEARFIELD_LAYOUT=
    table`` -- the local near field's switches -- and on the controls' tiles."""
    flat = {"particles": None}
    tiles = {"positions": None, "masses": None, "mask": None}
    monkeypatch.delenv("JACCPOT_NEARFIELD_ACCUM", raising=False)
    monkeypatch.delenv("JACCPOT_NEARFIELD_LAYOUT", raising=False)
    assert _cross_near_layout(flat) == "direct"
    assert _cross_near_layout(tiles) == "table"
    monkeypatch.setenv("JACCPOT_NEARFIELD_ACCUM", "wide")
    assert _cross_near_layout(flat) == "table"
    monkeypatch.delenv("JACCPOT_NEARFIELD_ACCUM")
    monkeypatch.setenv("JACCPOT_NEARFIELD_LAYOUT", "table")
    assert _cross_near_layout(flat) == "table"
