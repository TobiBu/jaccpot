"""Per-leaf Pallas P2M (plan sub-10ms, Phase 3), interpret mode on CPU."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.bounds import infer_bounds
from yggdrax._tree_impl import build_static_cells_tree
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.pallas.p2m_real_leaf import p2m_real_leaves_pallas
from jaccpot.upward.real_tree_expansions import _p2m_leaves_real


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


@pytest.mark.parametrize("order", [2, 4, 5])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pallas_leaf_p2m_matches_the_batched_reference(order, dtype):
    n, leaf = 3000, 16
    P = jnp.asarray(_plummer(n, 1), dtype)
    M = jnp.asarray(np.random.default_rng(2).uniform(0.5, 1.5, n), dtype)
    # a cell tree with padding leaves (empty) and a few tiny leaves (delta ~ 0 lanes)
    topo, ps, ms, inv = build_static_cells_tree(P, M, infer_bounds(P), leaf_size=leaf, leaf_capacity=1024, return_reordered=True)
    ni = int(topo.left_child.shape[0]); tot = int(topo.parent.shape[0])
    com = jnp.asarray(compute_tree_mass_moments(topo, ps, ms).center_of_mass, dtype)
    ref = _p2m_leaves_real(topo.node_ranges, ps, ms, com, order=order, max_leaf_size=leaf,
                           num_internal=ni, total_nodes=tot, leaf_batch_size=256)
    got = p2m_real_leaves_pallas(ps, ms, com[ni:], topo.node_ranges[ni:], order=order, num_internal=ni,
                                 total_nodes=tot, leaf_width=leaf, interpret=True)
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
