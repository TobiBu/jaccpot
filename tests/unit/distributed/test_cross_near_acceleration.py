"""The cross NEAR half against an independent direct sum.

`cross_near_acceleration` is the one piece of the cross field that is a plain
direct sum, so it can be checked exactly rather than by a ratio: build a pool by
hand, hand it a CSR that pairs known local leaves with known imported ones, and
require the answer to equal a plain numpy Plummer sum over precisely those
sources. Nothing here needs a GPU -- the Pallas kernel runs under interpret.

The reference is written out here rather than imported so that it is independent
of the code under test; it uses the same convention as the lane's own direct sum,
a = +G sum_j m_j (x_j - x_t) / (|x_j - x_t|^2 + eps^2)^{3/2}.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from jaccpot.distributed.cross import cross_near_acceleration

G = 1.3
SOFT = 0.07
NUM_INTERNAL = 5  # deliberately NOT L-1: a balanced tree would hide a row/node mixup


def _pool(rng, n_leaves, width, live_per_leaf, first_index):
    """A leaf-major pool with ragged occupancy and the particle indices behind it."""
    pos = np.zeros((n_leaves, width, 3))
    mass = np.zeros((n_leaves, width))
    mask = np.zeros((n_leaves, width), bool)
    idx = np.zeros((n_leaves, width), np.int32)
    nxt = first_index
    for r in range(n_leaves):
        k = live_per_leaf[r]
        pos[r, :k] = rng.normal(size=(k, 3))
        mass[r, :k] = rng.uniform(0.5, 1.5, size=k)
        mask[r, :k] = True
        idx[r, :k] = np.arange(nxt, nxt + k)
        nxt += k
    return pos, mass, mask, idx, nxt


def _build(rng, pairs, n_local=4, n_imp=3, width=6):
    local_live = [4, 6, 1, 3]
    imp_live = [5, 2, 6]
    lp, lm, lk, lidx, n_local_particles = _pool(rng, n_local, width, local_live, 0)
    ip, im, ik, iidx, total = _pool(rng, n_imp, width, imp_live, n_local_particles)

    P = 16  # pair capacity, deliberately larger than len(pairs)
    tgt = np.full(P, -1, np.int32)
    src = np.zeros(P, np.int32)
    for i, (t, s) in enumerate(pairs):
        tgt[i] = t + NUM_INTERNAL  # targets arrive as NODE ids
        src[i] = s
    near = {
        "positions": jnp.asarray(ip),
        "masses": jnp.asarray(im),
        "mask": jnp.asarray(ik),
        "target_node": jnp.asarray(tgt),
        "source_row": jnp.asarray(src),
        "count": jnp.asarray(len(pairs), jnp.int32),
    }
    return near, (lp, lm, lk, lidx), (ip, im, ik, iidx), n_local_particles, total


def _run(near, local, n_local_particles):
    lp, lm, lk, lidx = local
    return np.asarray(
        cross_near_acceleration(
            near,
            jnp.asarray(lp),
            jnp.asarray(lm),
            jnp.asarray(lk),
            jnp.asarray(lidx),
            n_local_particles,
            NUM_INTERNAL,
            softening_sq=jnp.asarray(SOFT**2),
            G=jnp.asarray(G),
            chunk=8,
            interpret=True,
        ),
        np.float64,
    )


def _reference(pairs, local, imported, n_local_particles, total):
    """Direct sum, one target leaf at a time, over exactly that leaf's sources."""
    lp, lm, lk, lidx = local
    ip, im, ik, iidx = imported
    ref = np.zeros((n_local_particles, 3))
    for t, s in pairs:
        tp = lp[t][lk[t]]
        ti = lidx[t][lk[t]]
        sp = ip[s][ik[s]]
        sm = im[s][ik[s]]
        d = sp[None, :, :] - tp[:, None, :]
        inv3 = (np.sum(d * d, axis=-1) + SOFT**2) ** -1.5
        ref[ti] += G * np.einsum("ij,ijk->ik", sm[None, :] * inv3, d)
    return ref


@pytest.mark.parametrize(
    "pairs",
    [
        [(0, 0)],
        [(1, 2), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (2, 1), (3, 2)],  # several sources per target
    ],
    ids=["one", "two", "multi-source"],
)
def test_matches_direct_sum(pairs):
    rng = np.random.default_rng(7)
    near, local, imported, n_lp, total = _build(rng, pairs)
    got = _run(near, local, n_lp)
    ref = _reference(pairs, local, imported, n_lp, total)
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-12), np.abs(got - ref).max()


def test_untouched_leaves_stay_zero():
    """A local leaf in no pair must receive nothing -- the CSR is not a broadcast."""
    rng = np.random.default_rng(11)
    pairs = [(1, 0)]
    near, local, imported, n_lp, _ = _build(rng, pairs)
    got = _run(near, local, n_lp)
    lp, lm, lk, lidx = local
    for r in (0, 2, 3):
        assert np.all(got[lidx[r][lk[r]]] == 0.0)
    assert np.any(got[lidx[1][lk[1]]] != 0.0)


def test_dead_csr_rows_contribute_nothing():
    """Capacity beyond `count` is padding; raising the cap must not change the answer."""
    rng = np.random.default_rng(13)
    pairs = [(0, 1), (2, 2)]
    near, local, imported, n_lp, _ = _build(rng, pairs)
    base = _run(near, local, n_lp)
    # poison every dead slot with a target/source that WOULD contribute if read
    tgt = np.asarray(near["target_node"]).copy()
    src = np.asarray(near["source_row"]).copy()
    tgt[len(pairs) :] = NUM_INTERNAL + 3
    src[len(pairs) :] = 0
    near["target_node"] = jnp.asarray(tgt)
    near["source_row"] = jnp.asarray(src)
    assert np.array_equal(_run(near, local, n_lp), base)


def test_out_of_range_target_is_dropped_not_folded():
    """A target past the local pool must not be clipped onto a real row."""
    rng = np.random.default_rng(17)
    near, local, imported, n_lp, _ = _build(rng, [(0, 0)])
    base = _run(near, local, n_lp)
    tgt = np.asarray(near["target_node"]).copy()
    tgt[1] = NUM_INTERNAL + 99  # beyond L
    src = np.asarray(near["source_row"]).copy()
    src[1] = 1
    near["target_node"] = jnp.asarray(tgt)
    near["source_row"] = jnp.asarray(src)
    near["count"] = jnp.asarray(2, jnp.int32)
    assert np.array_equal(_run(near, local, n_lp), base)


def test_mass_scaling_is_linear():
    """Doubling every imported mass doubles the term -- it is a source-linear sum."""
    rng = np.random.default_rng(19)
    pairs = [(0, 0), (1, 1)]
    near, local, imported, n_lp, _ = _build(rng, pairs)
    base = _run(near, local, n_lp)
    near["masses"] = near["masses"] * 2.0
    assert np.allclose(_run(near, local, n_lp), 2.0 * base, rtol=1e-12)


@pytest.mark.parametrize(
    "pad_local,pad_imported",
    [(5, 0), (0, 0), (0, 5)],
    ids=["imported-narrower", "equal", "imported-wider"],
)
def test_imported_tile_width_need_not_match_the_local_pool(pad_local, pad_imported):
    """W is a CAPACITY on the import side and a tree property locally; they differ.

    Both are padded up to the wider, which must not change the answer -- the extra
    slots are mask-false. Without this the concatenate is a trace-time shape error.
    """
    rng = np.random.default_rng(23)
    pairs = [(0, 0), (2, 1)]
    near, local, imported, n_lp, _ = _build(rng, pairs, width=6)
    ref = _reference(pairs, local, imported, n_lp, None)

    def _pad(pos, mass, mask, idx, n):
        if n == 0:
            return pos, mass, mask, idx
        return (
            np.pad(pos, ((0, 0), (0, n), (0, 0))),
            np.pad(mass, ((0, 0), (0, n))),
            np.pad(mask, ((0, 0), (0, n))),
            np.pad(idx, ((0, 0), (0, n))),
        )

    lp, lm, lk, lidx = _pad(*local, pad_local)
    ip, im, ik, _ = _pad(*imported, pad_imported)
    near["positions"] = jnp.asarray(ip)
    near["masses"] = jnp.asarray(im)
    near["mask"] = jnp.asarray(ik)

    got = _run(near, (lp, lm, lk, lidx), n_lp)
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-12), np.abs(got - ref).max()
