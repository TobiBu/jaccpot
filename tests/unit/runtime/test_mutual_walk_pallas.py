"""One-launch-per-round Pallas walk (plan sub-10ms, Phase 2): same pair SETS as the flat walk."""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax._tree_impl import build_static_cells_tree, build_static_radix_tree
from yggdrax.bounds import infer_bounds
from yggdrax.interactions import dual_tree_walk_mutual
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.pallas.mutual_walk_pallas import mutual_walk_pallas
from jaccpot.runtime._mac_geometry import com_mac_geometry
from tests.unit._typecheck_budget import trim

_KINDS = trim(["cells", "buckets"])
_THETAS = trim([0.8, 0.5])


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _sets(res):
    nf, nn = int(res.far_count), int(res.near_count)
    fa, fb = np.asarray(res.far_a)[:nf], np.asarray(res.far_b)[:nf]
    na, nb = np.asarray(res.near_a)[:nn], np.asarray(res.near_b)[:nn]
    return set(zip(fa.tolist(), fb.tolist())), set(zip(na.tolist(), nb.tolist()))


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("theta", _THETAS)
@pytest.mark.parametrize(
    "node_layout, fused_emit", [("soa", False), ("record", False), ("record", True)]
)
def test_pallas_walk_lists_equal_the_flat_walk_as_sets(
    kind, theta, node_layout, fused_emit
):
    # Interpret-mode cost tracks the PAIR count, not n: 800 particles at leaf 4 ran
    # SLOWER than 4000 at leaf 16 (more leaves -> 43k far pairs). The typecheck job is
    # handled by trimming the grid above, not by shrinking here.
    n, leaf = 4000, 16
    P = jnp.asarray(_plummer(n, 3), jnp.float32)
    M = jnp.ones((n,), jnp.float32)
    bounds = infer_bounds(P)
    if kind == "buckets":
        topo, ps, ms, inv = build_static_radix_tree(
            P, M, bounds, leaf_size=leaf, return_reordered=True
        )
    else:
        topo, ps, ms, inv = build_static_cells_tree(
            P, M, bounds, leaf_size=leaf, leaf_capacity=1024, return_reordered=True
        )
    com = compute_tree_mass_moments(topo, ps, ms).center_of_mass
    geom = com_mac_geometry(topo, ps, com, leaf_cap=leaf)
    ni = int(topo.left_child.shape[0])
    tot = int(topo.parent.shape[0])
    idx = topo.parent.dtype
    left = jnp.concatenate([topo.left_child, jnp.full((tot - ni,), -1, idx)])
    right = jnp.concatenate([topo.right_child, jnp.full((tot - ni,), -1, idx)])
    ranges = np.asarray(topo.node_ranges)
    active = jnp.asarray(ranges[:, 1] >= ranges[:, 0])
    root = jnp.argmin(topo.parent).astype(idx)
    ref = dual_tree_walk_mutual(
        left,
        right,
        geom.center,
        geom.radius,
        theta,
        root,
        max_pair_queue=1 << 15,
        far_cap=1 << 17,
        near_cap=1 << 17,
        mac_type="dehnen",
        node_active=active,
    )
    assert not (
        bool(ref.queue_overflow) or bool(ref.far_overflow) or bool(ref.near_overflow)
    )
    got = mutual_walk_pallas(
        left,
        right,
        geom.center,
        geom.radius,
        theta,
        root,
        max_pair_queue=1 << 15,
        far_cap=1 << 17,
        near_cap=1 << 17,
        node_active=active,
        block=64,
        interpret=True,
        node_layout=node_layout,
        fused_emit=fused_emit,
    )
    assert not (
        bool(got.queue_overflow) or bool(got.far_overflow) or bool(got.near_overflow)
    )
    assert (
        int(ref.far_count) > 0
    ), "vacuous: no far pairs at this size/theta, the MAC never fires"
    assert int(ref.near_count) > 0, "vacuous: no near pairs"
    far_r, near_r = _sets(ref)
    far_g, near_g = _sets(got)
    assert len(far_g) == int(got.far_count) and len(near_g) == int(
        got.near_count
    ), "duplicate emission"
    assert far_g == far_r
    assert near_g == near_r
    assert int(got.rounds) == int(ref.rounds)
    assert int(got.peak_wavefront) == int(ref.peak_wavefront)


@pytest.mark.parametrize("fused_emit", [False, True])
def test_pallas_walk_flags_overflow(fused_emit):
    n = 3000
    P = jnp.asarray(_plummer(n, 5), jnp.float32)
    M = jnp.ones((n,), jnp.float32)
    topo, ps, ms, inv = build_static_radix_tree(
        P, M, infer_bounds(P), leaf_size=8, return_reordered=True
    )
    com = compute_tree_mass_moments(topo, ps, ms).center_of_mass
    geom = com_mac_geometry(topo, ps, com, leaf_cap=8)
    ni = int(topo.left_child.shape[0])
    tot = int(topo.parent.shape[0])
    idx = topo.parent.dtype
    left = jnp.concatenate([topo.left_child, jnp.full((tot - ni,), -1, idx)])
    right = jnp.concatenate([topo.right_child, jnp.full((tot - ni,), -1, idx)])
    root = jnp.argmin(topo.parent).astype(idx)
    small = mutual_walk_pallas(
        left,
        right,
        geom.center,
        geom.radius,
        0.6,
        root,
        max_pair_queue=1 << 14,
        far_cap=256,
        near_cap=1 << 16,
        block=64,
        interpret=True,
        fused_emit=fused_emit,
    )
    assert bool(small.far_overflow) and not bool(small.near_overflow)
    # A full far list does not stop the walk: it reports what the list needed,
    # which is the count of a walk that fits, and the near list is complete.
    fits = mutual_walk_pallas(
        left,
        right,
        geom.center,
        geom.radius,
        0.6,
        root,
        max_pair_queue=1 << 14,
        far_cap=1 << 16,
        near_cap=1 << 16,
        block=64,
        interpret=True,
        fused_emit=fused_emit,
    )
    assert not bool(fits.far_overflow) and int(fits.far_count) > 256
    assert int(small.far_needed) == int(fits.far_count)
    assert int(small.far_count) == 256
    assert int(small.near_needed) == int(small.near_count) == int(fits.near_count)
    assert _sets(small)[1] == _sets(fits)[1]
    tiny_q = mutual_walk_pallas(
        left,
        right,
        geom.center,
        geom.radius,
        0.6,
        root,
        max_pair_queue=64,
        far_cap=1 << 16,
        near_cap=1 << 16,
        block=64,
        interpret=True,
        fused_emit=fused_emit,
    )
    assert bool(tiny_q.queue_overflow)


def test_lex_sorted_orders_by_target_then_source_in_both_branches():
    from jaccpot.runtime._interaction_cache import _lex_sorted

    rng = np.random.default_rng(7)
    prim = jnp.asarray(rng.integers(0, 50, 500), jnp.int32)
    sec = jnp.asarray(rng.integers(0, 70000, 500), jnp.int32)
    expect = np.lexsort((np.asarray(sec), np.asarray(prim)))
    for primary_bound in (50, 50000):  # int32 composite / int64 composite
        p, s = _lex_sorted(
            prim, sec, primary_bound=primary_bound, secondary_bound=70000
        )
        assert p.dtype == prim.dtype and s.dtype == sec.dtype
        assert np.array_equal(np.asarray(p), np.asarray(prim)[expect])
        assert np.array_equal(np.asarray(s), np.asarray(sec)[expect])


def test_walk_backend_flag_parsing(monkeypatch):
    from jaccpot.runtime._interaction_cache import (
        strict_walk_backend,
        strict_walk_deterministic_rows,
    )

    monkeypatch.delenv("JACCPOT_STATIC_STRICT_FUSED_WALK", raising=False)
    assert strict_walk_backend() == "flat"
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_WALK", "Pallas")
    assert strict_walk_backend() == "pallas"
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_WALK", "cuda")
    with pytest.raises(ValueError):
        strict_walk_backend()
    monkeypatch.delenv("JACCPOT_STATIC_STRICT_FUSED_WALK_DETERMINISTIC", raising=False)
    assert strict_walk_deterministic_rows()
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_WALK_DETERMINISTIC", "off")
    assert not strict_walk_deterministic_rows()


@pytest.mark.parametrize("floor", [0.05, 0.3])
def test_separation_floor_keeps_every_far_pair_apart_in_both_walks(floor):
    # The softening floor: an accepted far pair also needs |c_b - c_a| >= r_a + r_b +
    # floor (exact COM radii, so no two particles of the pair are closer than the
    # floor). The Pallas walk and yggdrax's flat walk keep the same SETS with it,
    # every accepted pair honours it, and it moves work from the far to the near
    # list (a floor that changed nothing would make the test vacuous).
    n, leaf, theta = 4000, 16, 0.8
    P = jnp.asarray(_plummer(n, 5), jnp.float32)
    M = jnp.ones((n,), jnp.float32)
    topo, ps, ms, inv = build_static_cells_tree(
        P, M, infer_bounds(P), leaf_size=leaf, leaf_capacity=1024, return_reordered=True
    )
    com = compute_tree_mass_moments(topo, ps, ms).center_of_mass
    geom = com_mac_geometry(topo, ps, com, leaf_cap=leaf)
    ni = int(topo.left_child.shape[0])
    tot = int(topo.parent.shape[0])
    idx = topo.parent.dtype
    left = jnp.concatenate([topo.left_child, jnp.full((tot - ni,), -1, idx)])
    right = jnp.concatenate([topo.right_child, jnp.full((tot - ni,), -1, idx)])
    ranges = np.asarray(topo.node_ranges)
    active = jnp.asarray(ranges[:, 1] >= ranges[:, 0])
    root = jnp.argmin(topo.parent).astype(idx)
    kw = dict(max_pair_queue=1 << 15, far_cap=1 << 17, near_cap=1 << 17, node_active=active)
    base = dual_tree_walk_mutual(
        left, right, geom.center, geom.radius, theta, root, mac_type="dehnen", **kw
    )
    ref = dual_tree_walk_mutual(
        left,
        right,
        geom.center,
        geom.radius,
        theta,
        root,
        mac_type="dehnen",
        separation_floor=floor,
        **kw,
    )
    got = mutual_walk_pallas(
        left,
        right,
        geom.center,
        geom.radius,
        theta,
        root,
        block=64,
        interpret=True,
        separation_floor=floor,
        **kw,
    )
    for res in (base, ref, got):
        assert not (
            bool(res.queue_overflow) or bool(res.far_overflow) or bool(res.near_overflow)
        )
    far_r, near_r = _sets(ref)
    far_g, near_g = _sets(got)
    assert far_g == far_r and near_g == near_r
    assert int(ref.far_count) < int(base.far_count), "vacuous: the floor changed nothing"
    assert int(ref.near_count) > int(base.near_count)
    a = np.asarray([p[0] for p in far_r]); b = np.asarray([p[1] for p in far_r])
    c = np.asarray(geom.center, np.float64); r = np.asarray(geom.radius, np.float64)
    gap = np.linalg.norm(c[b] - c[a], axis=1) - r[a] - r[b]
    assert gap.min() >= floor * (1 - 1e-5)
    # and, with the exact radii, the closest particles of every far pair
    pos = np.asarray(ps, np.float64)
    worst = np.inf
    for i in np.argsort(gap)[:50]:
        pa = pos[ranges[a[i], 0] : ranges[a[i], 1] + 1]
        pb = pos[ranges[b[i], 0] : ranges[b[i], 1] + 1]
        dmin = np.min(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1))
        worst = min(worst, dmin)
    assert worst >= floor * (1 - 1e-5)
