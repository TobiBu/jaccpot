"""Every near-field path evaluates the compact softening kernels it is handed.

The kernels' own polynomials are pinned in ``tests/unit/test_softening.py``. Here
each lane that evaluates pairs is run with ``"ferrers3"`` and ``"wendland_c2"``
(fp64, Pallas in interpret mode) against an independent reference:

* the dense JAX twin against a NumPy loop over the same pairs;
* the generic jnp kernels (pair, batched, componentwise, self) against the twin's
  arithmetic, and the batched pair rule's analytic reverse against autodiff;
* the CSR kernels (table, sorted direct in every source-tile mode), the rectangle
  kernel and the pairs kernel against the twin;
* the CSR reverse and the fast lane's analytic reverse against ``jax.vjp`` of the
  twins, positions, masses, the softening and ``G``;
* the targeted near field's jerk, snap and crackle against finite differences.

A lane that silently kept Plummer would differ from the twin by the softening
itself: every case puts pairs inside the support (``h`` comparable to the spacing).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.nearfield._kernels import (
    _pair_contributions,
    _pair_contributions_batched,
    _pair_contributions_batched_componentwise,
    _self_contributions,
)
from jaccpot.nearfield.grad import _pair_accel_cvjp, _pair_accel_masked_accels
from jaccpot.pallas.nearfield_fused_leaf import (
    nearfield_fused_leaf_jax,
    nearfield_fused_leaf_pallas,
    nearfield_leafpair_jax,
    nearfield_leafpair_pallas,
)
from jaccpot.pallas.nearfield_leafpair_csr import (
    build_leafpair_chunk_table,
    leafpair_chunk_capacity,
    nearfield_leafpair_csr_jax,
    nearfield_leafpair_csr_pallas,
    nearfield_leafpair_csr_pallas_cvjp,
    nearfield_leafpair_csr_sorted_direct_pallas,
)
from jaccpot.softening import pair_factors, softening_params_np, support_radius

jax.config.update("jax_enable_x64", True)

_COMPACT = ("ferrers3", "wendland_c2")
_EPS = 0.35  # Plummer-equivalent; h = 0.86 / 1.05 against unit-normal positions
_G = 1.3


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _np_pair(d, m, kernel):
    """Acceleration and potential of sources at offsets ``d`` (targets minus sources)."""
    r2 = np.sum(d * d, axis=-1)
    g, psi, _ = pair_factors(
        r2, softening_params_np(kernel, _EPS), kernel, potential=True, xp=np
    )
    return -_G * np.sum((g * m)[..., None] * d, axis=-2), -_G * np.sum(psi * m, axis=-1)


def _leaf_case(seed=0, L=6, W=8, S=4):
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((L, W, 3))
    mass = np.abs(rng.standard_normal((L, W))) + 0.1
    mask = rng.random((L, W)) > 0.2
    mask[:, 0] = True
    ids = np.stack([rng.choice(np.setdiff1d(np.arange(L), [l]), S) for l in range(L)])
    valid = rng.random((L, S)) > 0.2
    return pos, mass, mask, ids, valid


def _twin_reference(pos, mass, mask, ids, valid, kernel, include_self):
    L, W, _ = pos.shape
    out = np.zeros((L, W, 4))
    for l in range(L):
        for t in range(W):
            if not mask[l, t]:
                continue
            srcs = []
            for k in range(ids.shape[1]):
                if valid[l, k]:
                    s = ids[l, k]
                    srcs += [(s, j) for j in range(W) if mask[s, j]]
            if include_self:
                srcs += [(l, j) for j in range(W) if mask[l, j] and j != t]
            if not srcs:
                continue
            sp = np.array([pos[s, j] for s, j in srcs])
            sm = np.array([mass[s, j] for s, j in srcs])
            a, p = _np_pair(pos[l, t] - sp, sm, kernel)
            out[l, t, :3], out[l, t, 3] = a, p
    return out


@pytest.mark.parametrize("kernel", _COMPACT)
@pytest.mark.parametrize("include_self", [False, True])
def test_dense_twin_matches_a_numpy_loop(kernel, include_self):
    pos, mass, mask, ids, valid = _leaf_case(1)
    got = nearfield_leafpair_jax(
        jnp.asarray(pos),
        jnp.asarray(mass),
        jnp.asarray(mask),
        jnp.asarray(ids, jnp.int32),
        jnp.asarray(valid),
        softening_sq=jnp.asarray(_EPS**2),
        G=jnp.asarray(_G),
        include_self=include_self,
        softening_kernel=kernel,
    )
    want = _twin_reference(pos, mass, mask, ids, valid, kernel, include_self)
    assert _rel(got, want) < 1e-12
    # non-vacuity: some pairs lie inside the support
    h = support_radius(kernel, _EPS)
    d = pos[:, :, None, :] - pos[:, None, :, :]
    assert np.any((np.linalg.norm(d, axis=-1) < h) & (np.linalg.norm(d, axis=-1) > 0))
    plummer = nearfield_leafpair_jax(
        jnp.asarray(pos),
        jnp.asarray(mass),
        jnp.asarray(mask),
        jnp.asarray(ids, jnp.int32),
        jnp.asarray(valid),
        softening_sq=jnp.asarray(_EPS**2),
        G=jnp.asarray(_G),
        include_self=include_self,
        softening_kernel="plummer",
    )
    assert _rel(plummer, want) > 1e-3


@pytest.mark.parametrize("kernel", _COMPACT)
def test_generic_jnp_kernels_and_their_reverse(kernel):
    rng = np.random.default_rng(3)
    B, Wt, Ws = 4, 5, 7
    tp = jnp.asarray(rng.standard_normal((B, Wt, 3)))
    sp = jnp.asarray(rng.standard_normal((B, Ws, 3)))
    sm = jnp.asarray(np.abs(rng.standard_normal((B, Ws))) + 0.1)
    tmask = jnp.asarray(rng.random((B, Wt)) > 0.2)
    smask = jnp.asarray(rng.random((B, Ws)) > 0.2)
    soft, G = jnp.asarray(_EPS**2), jnp.asarray(_G)
    d = np.asarray(tp)[:, :, None, :] - np.asarray(sp)[:, None, :, :]
    m = np.where(np.asarray(smask), np.asarray(sm), 0.0)[:, None, :]
    want_a, want_p = _np_pair(d, np.broadcast_to(m, d.shape[:-1]), kernel)
    tm = np.asarray(tmask)
    want_a, want_p = np.where(tm[..., None], want_a, 0.0), np.where(tm, want_p, 0.0)
    kw = dict(softening_sq=soft, softening_kernel=kernel, G=G, compute_potential=True)
    a1, p1 = _pair_contributions_batched(tp, tmask, sp, sm, smask, **kw)
    a2, p2 = _pair_contributions_batched_componentwise(tp, tmask, sp, sm, smask, **kw)
    a3, p3 = _pair_contributions(tp[0], tmask[0], sp[0], sm[0], smask[0], **kw)
    for a, p in ((a1, p1), (a2, p2)):
        assert _rel(a, want_a) < 1e-12 and _rel(p, want_p) < 1e-12
    assert _rel(a3, want_a[0]) < 1e-12 and _rel(p3, want_p[0]) < 1e-12
    # the self block: one leaf against itself, diagonal out
    pos_l = jnp.asarray(rng.standard_normal((3, 6, 3)))
    m_l = jnp.asarray(np.abs(rng.standard_normal((3, 6))) + 0.1)
    mk_l = jnp.asarray(rng.random((3, 6)) > 0.2)
    sa, spot = _self_contributions(pos_l, m_l, mk_l, **kw)
    ids = np.zeros((3, 1), np.int32)
    want = _twin_reference(
        np.asarray(pos_l), np.asarray(m_l), np.asarray(mk_l), ids,
        np.zeros((3, 1), bool), kernel, True,
    )
    assert _rel(sa, want[..., :3]) < 1e-12 and _rel(spot, want[..., 3]) < 1e-12
    # the batched rule's analytic reverse (positions, masses, softening, G)
    tmf, smf = tmask.astype(jnp.float64), smask.astype(jnp.float64)
    cot = jnp.asarray(rng.standard_normal((B, Wt, 3)))

    def custom(t, s, ms, so, g):
        return _pair_accel_cvjp(t, s, ms, tmf, smf, so, g, kernel)

    def auto(t, s, ms, so, g):
        return _pair_accel_masked_accels(t, s, ms, tmask, smask, so, g, kernel)

    args = (tp, sp, sm, soft, G)
    _, vc = jax.vjp(custom, *args)
    _, va = jax.vjp(auto, *args)
    for got, want in zip(vc(cot), va(cot)):
        assert _rel(got, want) < 1e-10


def _csr_case(seed, L=7, W=6, p_edge=0.5):
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((L, W, 3))
    mass = np.abs(rng.standard_normal((L, W))) + 0.1
    mask = rng.random((L, W)) > 0.25
    mask[:, 0] = True
    adj = np.triu(rng.random((L, L)) < p_edge, 1)
    adj = adj | adj.T
    rows = [np.flatnonzero(adj[i]) for i in range(L)]
    counts = np.array([r.size for r in rows])
    nbr = np.zeros(int(counts.sum()) + 5, np.int32)
    nbr[: counts.sum()] = np.concatenate(rows)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    return (
        jnp.asarray(pos),
        jnp.asarray(mass),
        jnp.asarray(mask),
        jnp.asarray(nbr),
        jnp.asarray(offsets, jnp.int32),
        jnp.asarray(counts, jnp.int32),
    )


@pytest.mark.parametrize("kernel", _COMPACT)
@pytest.mark.parametrize("chunk", [2, 5])
def test_csr_table_kernel_and_its_reverse(kernel, chunk):
    pos, mass, mask, nbr, offsets, counts = _csr_case(2)
    L = int(pos.shape[0])
    tab = build_leafpair_chunk_table(
        offsets,
        counts,
        chunk=chunk,
        capacity=leafpair_chunk_capacity(int(nbr.shape[0]), L, chunk),
    )
    soft, G = jnp.asarray(_EPS**2), jnp.asarray(_G)
    want = nearfield_leafpair_csr_jax(
        pos, mass, mask, nbr, offsets, counts,
        softening_sq=soft, G=G, softening_kernel=kernel,
    )
    got = nearfield_leafpair_csr_pallas(
        pos, mass, mask, nbr, tab, softening_sq=soft, G=G, chunk=chunk,
        interpret=True, softening_kernel=kernel,
    )
    assert _rel(got, want) < 1e-12

    rng = np.random.default_rng(9)
    cot = jnp.asarray(rng.standard_normal(want.shape)).at[..., 3].set(0.0)

    def ref(p, m, s, g):
        return nearfield_leafpair_csr_jax(
            p, m, mask, nbr, offsets, counts, softening_sq=s, G=g,
            softening_kernel=kernel,
        )

    def cvjp(p, m, s, g):
        return nearfield_leafpair_csr_pallas_cvjp(
            p, m, mask, nbr, tab, s, g, chunk, None, 1, None, True, "input", True,
            None, kernel,
        )

    _, vr = jax.vjp(ref, pos, mass, soft, G)
    _, vg = jax.vjp(cvjp, pos, mass, soft, G)
    for got_b, want_b in zip(vg(cot), vr(cot)):
        assert _rel(got_b, want_b) < 1e-9
    assert abs(float(vr(cot)[2])) > 0  # the softening cotangent is live


@pytest.mark.parametrize("kernel", _COMPACT)
@pytest.mark.parametrize(
    "source_tile, flags", [(0, None), (4, ""), (4, "l"), (8, "alr"), (4, "alg")]
)
def test_csr_sorted_direct_in_every_source_tile_mode(kernel, source_tile, flags):
    from tests.unit.operators.test_pallas_nearfield_leafpair_csr_sorted import _case

    W = 8
    c = _case(7, num_live=9, num_pad=3, W=W, n_dead=2, max_row=6, empty_rows=(2,))
    pos64 = jnp.asarray(c["pos"], jnp.float64)
    mass64 = jnp.asarray(c["mass"], jnp.float64)
    soft, G = jnp.asarray(_EPS**2), jnp.asarray(_G)
    acc, pot = nearfield_leafpair_csr_sorted_direct_pallas(
        pos64, mass64, c["starts"], c["counts"], c["nbr"], c["offsets"],
        c["row_counts"], leaf_width=W, softening_sq=soft, G=G, chunk=1,
        interpret=True, with_potential=True, source_tile=source_tile,
        source_flags=flags, softening_kernel=kernel,
    )
    want = nearfield_leafpair_csr_jax(
        jnp.asarray(c["leaf_pos"], jnp.float64),
        jnp.asarray(c["leaf_mass"], jnp.float64),
        c["mask"], c["nbr"], c["offsets"], c["row_counts"],
        softening_sq=soft, G=G, softening_kernel=kernel,
    )
    want = np.asarray(want)
    counts, starts = np.asarray(c["counts"]), np.asarray(c["starts"])
    n = int(pos64.shape[0])
    flat = np.zeros((n, 4))
    for leaf in range(c["L"]):
        flat[starts[leaf] : starts[leaf] + counts[leaf]] = want[leaf, : counts[leaf]]
    assert _rel(acc, flat[:, :3]) < 1e-12
    assert _rel(pot, flat[:, 3]) < 1e-12


@pytest.mark.parametrize("kernel", _COMPACT)
@pytest.mark.parametrize("include_self", [False, True])
def test_rectangle_and_pairs_kernels(kernel, include_self):
    pos, mass, mask, ids, valid = _leaf_case(4)
    args = (
        jnp.asarray(pos),
        jnp.asarray(mass),
        jnp.asarray(mask),
        jnp.asarray(ids, jnp.int32),
        jnp.asarray(valid),
    )
    kw = dict(softening_sq=jnp.asarray(_EPS**2), G=jnp.asarray(_G), softening_kernel=kernel)
    want = nearfield_leafpair_jax(*args, include_self=include_self, **kw)
    got = nearfield_leafpair_pallas(*args, interpret=True, include_self=include_self, **kw)
    assert _rel(got, want) < 1e-12
    got_c = nearfield_leafpair_pallas(
        *args, interpret=True, include_self=include_self, source_chunk=2, **kw
    )
    assert _rel(got_c, want) < 1e-12
    # the pairs lane: materialised source blocks
    rng = np.random.default_rng(6)
    tp = jnp.asarray(rng.standard_normal((5, 8, 3)))
    tm = jnp.asarray(rng.random((5, 8)) > 0.2)
    sp = jnp.asarray(rng.standard_normal((5, 9, 3)))
    sm = jnp.asarray(np.abs(rng.standard_normal((5, 9))) + 0.1)
    smk = jnp.asarray(rng.random((5, 9)) > 0.2)
    want_p = nearfield_fused_leaf_jax(tp, tm, sp, sm, smk, **kw)
    got_p = nearfield_fused_leaf_pallas(tp, tm, sp, sm, smk, interpret=True, **kw)
    assert _rel(got_p, want_p) < 1e-12


@pytest.mark.parametrize("kernel", _COMPACT)
def test_fast_lane_analytic_reverse_matches_its_tiled_twin(kernel):
    from jaccpot.nearfield import _fast_lane as fast_lane
    from jaccpot.nearfield import near_field as nf

    rng = np.random.default_rng(0)
    num_leaves, width, max_blocks, block_size = 6, 8, 2, 3
    n = num_leaves * width
    f8 = jnp.float64
    leaf_particle_idx = jnp.asarray(
        np.arange(n).reshape(num_leaves, width), nf.INDEX_DTYPE
    )
    leaf_mask = jnp.ones((num_leaves, width), bool)
    positions = jnp.asarray(rng.normal(size=(n, 3)), f8)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=n), f8)
    leaf_positions, leaf_masses = positions[leaf_particle_idx], masses[leaf_particle_idx]
    sids = jnp.asarray(
        rng.integers(0, num_leaves, size=(num_leaves, max_blocks, block_size)),
        nf.INDEX_DTYPE,
    )
    svalid = jnp.asarray(rng.random((num_leaves, max_blocks, block_size)) > 0.3)
    soft, G = jnp.asarray(_EPS**2, f8), jnp.asarray(_G, f8)

    def custom(lp, lm, s, g):
        return fast_lane._radix_fast_lane_prepacked_accel_cvjp(
            lp, lm, positions, sids.astype(f8), svalid.astype(f8),
            leaf_mask.astype(f8), leaf_particle_idx.astype(f8), s, g,
            None, 1, None, True, 2, 2, False, None, kernel,
        )

    def ref(lp, lm, s, g):
        return nf._compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl(
            positions, sids, svalid, lp, lm, leaf_mask, leaf_particle_idx,
            G=g, softening_sq=s, softening_kernel=kernel,
            target_leaf_batch_size=2, target_block_tile_size=2,
            target_block_tile_scan_unroll=1, target_block_batch_scan_unroll=1,
            occupancy_sort=False, skip_empty_tiles=False, componentwise_pairs=False,
        )

    args = (leaf_positions, leaf_masses, soft, G)
    out_c, vc = jax.vjp(custom, *args)
    out_r, vr = jax.vjp(ref, *args)
    assert _rel(out_c, out_r) < 1e-10
    cot = jnp.asarray(rng.standard_normal(out_r.shape), f8)
    for got, want in zip(vc(cot), vr(cot)):
        assert _rel(got, want) < 1e-9


@pytest.mark.parametrize("kernel", _COMPACT)
def test_targeted_time_derivatives_match_finite_differences(kernel):
    from jaccpot.runtime.kernels._evaluate import _compute_targeted_nearfield

    rng = np.random.default_rng(8)
    n, t = 40, 6
    pos = jnp.asarray(rng.standard_normal((n, 3)) * 0.6)
    vel = jnp.asarray(rng.standard_normal((n, 3)))
    mass = jnp.asarray(np.abs(rng.standard_normal(n)) + 0.1)
    tgt = jnp.arange(t, dtype=jnp.int32)
    src_np = np.stack([np.setdiff1d(np.arange(n), [i]) for i in range(t)])
    src = jnp.asarray(src_np, jnp.int32)
    # A compact kernel's force is only C^3 at the support (C^1 is pinned in
    # test_softening.py): a pair that crosses h within the stencil puts a kink into
    # the finite differences, not into the derivatives. Keep pairs clear of it,
    # inside and outside the support.
    p_np = np.asarray(pos)
    r = np.linalg.norm(p_np[:t, None, :] - p_np[src_np], axis=-1)
    hk = support_radius(kernel, _EPS)
    smask_np = np.abs(r - hk) > 0.08
    assert np.any(smask_np & (r < hk)) and np.any(smask_np & (r > hk))
    smask = jnp.asarray(smask_np)

    def acc_at(dt):
        out = _compute_targeted_nearfield(
            positions_sorted=pos + dt * vel, masses_sorted=mass,
            target_sorted_indices=tgt, source_indices=src, source_mask=smask,
            G=_G, softening=_EPS, return_potential=False, softening_kernel=kernel,
        )
        return np.asarray(out[0])

    a, _, jerk, snap, crackle = _compute_targeted_nearfield(
        positions_sorted=pos, masses_sorted=mass, target_sorted_indices=tgt,
        source_indices=src, source_mask=smask, G=_G, softening=_EPS,
        return_potential=False, velocities_sorted=vel, return_jerk=True,
        return_snap=True, return_crackle=True, softening_kernel=kernel,
    )
    def fds(h):
        f = {k: acc_at(k * h) for k in (-2, -1, 0, 1, 2)}
        return (
            (f[1] - f[-1]) / (2 * h),
            (f[1] - 2 * f[0] + f[-1]) / h**2,
            (f[2] - 2 * f[1] + 2 * f[-1] - f[-2]) / (2 * h**3),
        )

    # central differences carry an h^2 error; Richardson removes it (2026-10-08:
    # jerk 1.4e-5 -> 2.3e-7). Plummer's closed forms on these pairs would miss by
    # 1e-2 and more.
    coarse, fine = fds(4e-3), fds(2e-3)
    fd1, fd2, fd3 = ((4 * b - a) / 3 for a, b in zip(coarse, fine))
    assert _rel(a, acc_at(0.0)) < 1e-14
    assert _rel(jerk, fd1) < 1e-7
    assert _rel(snap, fd2) < 1e-6
    assert _rel(crackle, fd3) < 1e-5
