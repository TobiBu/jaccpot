"""One-launch-per-round Pallas walk (plan sub-10ms, Phase 2): same pair SETS as the flat walk."""

from __future__ import annotations

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


@pytest.mark.parametrize("kind", ["buckets", "cells"])
@pytest.mark.parametrize("theta", [0.5, 0.8])
def test_pallas_walk_lists_equal_the_flat_walk_as_sets(kind, theta):
    n = 4000
    P = jnp.asarray(_plummer(n, 3), jnp.float32)
    M = jnp.ones((n,), jnp.float32)
    bounds = infer_bounds(P)
    if kind == "buckets":
        topo, ps, ms, inv = build_static_radix_tree(
            P, M, bounds, leaf_size=16, return_reordered=True
        )
    else:
        topo, ps, ms, inv = build_static_cells_tree(
            P, M, bounds, leaf_size=16, leaf_capacity=1024, return_reordered=True
        )
    com = compute_tree_mass_moments(topo, ps, ms).center_of_mass
    geom = com_mac_geometry(topo, ps, com, leaf_cap=16)
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
    )
    assert not (
        bool(got.queue_overflow) or bool(got.far_overflow) or bool(got.near_overflow)
    )
    far_r, near_r = _sets(ref)
    far_g, near_g = _sets(got)
    assert len(far_g) == int(got.far_count) and len(near_g) == int(
        got.near_count
    ), "duplicate emission"
    assert far_g == far_r
    assert near_g == near_r
    assert int(got.rounds) == int(ref.rounds)
    assert int(got.peak_wavefront) == int(ref.peak_wavefront)


def test_pallas_walk_flags_overflow():
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
    )
    assert bool(small.far_overflow) and not bool(small.near_overflow)
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
    )
    assert bool(tiny_q.queue_overflow)


def test_lex_perm_orders_by_target_then_source_in_both_branches():
    from jaccpot.runtime._interaction_cache import _lex_perm

    rng = np.random.default_rng(7)
    prim = jnp.asarray(rng.integers(0, 50, 500), jnp.int32)
    sec = jnp.asarray(rng.integers(0, 70000, 500), jnp.int32)
    expect = np.lexsort((np.asarray(sec), np.asarray(prim)))
    small = np.asarray(
        _lex_perm(prim, sec, primary_bound=50, secondary_bound=70000)
    )  # composite fits int32
    big = np.asarray(
        _lex_perm(prim, sec, primary_bound=50000, secondary_bound=70000)
    )  # two stable sorts
    for perm in (small, big):
        p, s = np.asarray(prim)[perm], np.asarray(sec)[perm]
        assert np.all(np.diff(p) >= 0)
        assert np.all((np.diff(s) >= 0) | (np.diff(p) > 0))
        assert np.array_equal(p, np.asarray(prim)[expect]) and np.array_equal(
            s, np.asarray(sec)[expect]
        )


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
