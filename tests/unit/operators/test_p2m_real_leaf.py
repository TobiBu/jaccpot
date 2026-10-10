"""Pallas leaf P2M (plan sub-10ms, Phase 3), interpret mode on CPU.

The forward is the blocked kernel (several leaves per program); the per-leaf
forward it replaced was removed in the 2026-10 cleanup (X5), and the blocked
variants are anchored to the pure-JAX batched reference instead.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax._tree_impl import build_static_cells_tree
from yggdrax.bounds import infer_bounds
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.pallas.p2m_real_leaf import p2m_real_leaves_pallas
from jaccpot.upward.real_tree_expansions import _p2m_leaves_real
from tests.unit._typecheck_budget import trim


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


@pytest.mark.parametrize("order", trim([5, 2, 4]))
@pytest.mark.parametrize("dtype", trim([jnp.float64, jnp.float32]))
def test_pallas_leaf_p2m_matches_the_batched_reference(order, dtype):
    n, leaf = 3000, 16
    P = jnp.asarray(_plummer(n, 1), dtype)
    M = jnp.asarray(np.random.default_rng(2).uniform(0.5, 1.5, n), dtype)
    # a cell tree with padding leaves (empty) and a few tiny leaves (delta ~ 0 lanes)
    topo, ps, ms, inv = build_static_cells_tree(
        P, M, infer_bounds(P), leaf_size=leaf, leaf_capacity=1024, return_reordered=True
    )
    ni = int(topo.left_child.shape[0])
    tot = int(topo.parent.shape[0])
    com = jnp.asarray(compute_tree_mass_moments(topo, ps, ms).center_of_mass, dtype)
    ref = _p2m_leaves_real(
        topo.node_ranges,
        ps,
        ms,
        com,
        order=order,
        max_leaf_size=leaf,
        num_internal=ni,
        total_nodes=tot,
        leaf_batch_size=256,
    )
    got = p2m_real_leaves_pallas(
        ps,
        ms,
        com[ni:],
        topo.node_ranges[ni:],
        order=order,
        num_internal=ni,
        total_nodes=tot,
        leaf_width=leaf,
        interpret=True,
    )
    r, g = np.asarray(ref), np.asarray(got)
    assert g.shape == r.shape
    assert np.all(g[:ni] == 0)
    tol = 2e-5 if dtype == jnp.float32 else 1e-12
    scale = np.maximum(np.abs(r).max(axis=1, keepdims=True), 1e-12)
    assert np.allclose(g / scale, r / scale, rtol=0, atol=tol)
    ranges = np.asarray(topo.node_ranges)[ni:]
    empty = ranges[:, 1] < ranges[:, 0]
    assert empty.any() and np.all(g[ni:][empty] == 0)
    # monopole = leaf mass
    mass_np = np.asarray(ms)
    for l in np.flatnonzero(~empty)[:20]:
        a, b = ranges[l]
        assert np.isclose(g[ni + l, 0], mass_np[a : b + 1].sum(), rtol=tol * 10)


@pytest.mark.parametrize("order", trim([5, 4]))
@pytest.mark.parametrize("block, chunk", [(4, 4), (3, 8), (8, 16)])
def test_blocked_p2m_matches_the_batched_reference(order, block, chunk):
    # several leaves per program, ``chunk`` lanes per leaf at a time: the pure-JAX
    # batched P2M's coefficients up to float32 summation order, internal and empty
    # rows zero; a block of 3 is rounded to 4 and the leaf count is not a multiple
    # of it. (Until the 2026-10 cleanup the reference was the per-leaf kernel.)
    n, leaf = 3000, 16
    P = jnp.asarray(_plummer(n, 3), jnp.float32)
    M = jnp.asarray(np.random.default_rng(4).uniform(0.5, 1.5, n), jnp.float32)
    topo, ps, ms, inv = build_static_cells_tree(
        P, M, infer_bounds(P), leaf_size=leaf, leaf_capacity=1021, return_reordered=True
    )
    ni = int(topo.left_child.shape[0])
    tot = int(topo.parent.shape[0])
    com = jnp.asarray(compute_tree_mass_moments(topo, ps, ms).center_of_mass)
    kw = dict(
        order=order,
        num_internal=ni,
        total_nodes=tot,
        leaf_width=leaf,
        interpret=True,
    )
    ref = np.asarray(
        _p2m_leaves_real(
            topo.node_ranges,
            ps,
            ms,
            com,
            order=order,
            max_leaf_size=leaf,
            num_internal=ni,
            total_nodes=tot,
            leaf_batch_size=256,
        )
    )
    got = np.asarray(
        p2m_real_leaves_pallas(
            ps, ms, com[ni:], topo.node_ranges[ni:], block=block, chunk=chunk, **kw
        )
    )
    assert got.shape == ref.shape
    assert np.all(got[:ni] == 0)
    ranges = np.asarray(topo.node_ranges)[ni:]
    empty = ranges[:, 1] < ranges[:, 0]
    assert empty.any() and np.all(got[ni:][empty] == 0)
    scale = np.maximum(np.abs(ref).max(axis=1, keepdims=True), 1e-12)
    assert np.allclose(got / scale, ref / scale, rtol=0, atol=2e-6)


@pytest.mark.parametrize("block", [0, -1])
def test_the_removed_per_leaf_forward_raises(block, monkeypatch):
    """``block=0`` (and ``JACCPOT_P2M_BLOCK=0``) selected one program per leaf,
    removed in the 2026-10 cleanup (X5): refused by name, not run blocked."""
    n, leaf = 200, 16
    P = jnp.asarray(_plummer(n, 5), jnp.float32)
    M = jnp.ones((n,), jnp.float32)
    topo, ps, ms, inv = build_static_cells_tree(
        P, M, infer_bounds(P), leaf_size=leaf, leaf_capacity=64, return_reordered=True
    )
    ni = int(topo.left_child.shape[0])
    tot = int(topo.parent.shape[0])
    com = jnp.asarray(compute_tree_mass_moments(topo, ps, ms).center_of_mass)
    kw = dict(order=3, num_internal=ni, total_nodes=tot, leaf_width=leaf)
    args = (ps, ms, com[ni:], topo.node_ranges[ni:])
    with pytest.raises(ValueError, match="removed in the 2026-10 cleanup"):
        p2m_real_leaves_pallas(*args, block=block, interpret=True, **kw)
    monkeypatch.setenv("JACCPOT_P2M_BLOCK", str(block))
    with pytest.raises(ValueError, match="JACCPOT_P2M_BLOCK"):
        p2m_real_leaves_pallas(*args, interpret=True, **kw)
