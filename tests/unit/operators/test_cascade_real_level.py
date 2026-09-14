"""Per-level Pallas M2M / L2L cascades (plan sub-10ms, Phase 3), interpret mode on CPU."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.bounds import infer_bounds
from yggdrax.tree import Tree

from jaccpot.operators.real_translations import l2l_real, m2m_real
from jaccpot.pallas.cascade_real_level import (
    l2l_real_centred_pair_jax,
    l2l_real_levels_pallas,
    m2m_real_centred_pair_jax,
    m2m_real_levels_pallas,
)
from jaccpot.runtime.kernels._l2l import _propagate_solidfmm_locals_by_level
from jaccpot.upward.real_tree_expansions import aggregate_m2m_real_by_level
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
@pytest.mark.parametrize("seed", trim([0, 3]))
def test_pair_twins_match_the_reference_operators(order, seed):
    rng = np.random.default_rng(seed)
    C = (order + 1) ** 2
    coeffs = jnp.asarray(rng.standard_normal(C), jnp.float64)
    delta = jnp.asarray(rng.standard_normal(3) * 0.7, jnp.float64)
    m_ref = m2m_real(coeffs, delta, order=order)
    m_got = m2m_real_centred_pair_jax(coeffs, delta, order=order)
    assert np.allclose(np.asarray(m_got), np.asarray(m_ref), rtol=1e-10, atol=1e-12)
    l_ref = l2l_real(coeffs, delta, order=order)
    l_got = l2l_real_centred_pair_jax(coeffs, delta, order=order)
    assert np.allclose(np.asarray(l_got), np.asarray(l_ref), rtol=1e-10, atol=1e-12)
    # axis-aligned and zero displacements (rho == 0 branch of the alignment)
    for d in (
        jnp.asarray([0.0, 0.0, 0.4]),
        jnp.asarray([0.0, 0.0, -0.4]),
        jnp.zeros(3),
    ):
        d = d.astype(jnp.float64)
        assert np.allclose(
            np.asarray(m2m_real_centred_pair_jax(coeffs, d, order=order)),
            np.asarray(m2m_real(coeffs, d, order=order)),
            rtol=1e-10,
            atol=1e-12,
        )
        assert np.allclose(
            np.asarray(l2l_real_centred_pair_jax(coeffs, d, order=order)),
            np.asarray(l2l_real(coeffs, d, order=order)),
            rtol=1e-10,
            atol=1e-12,
        )


@pytest.fixture(scope="module")
def tree_data():
    n, leaf = 3000, 16
    P = jnp.asarray(_plummer(n, 1), jnp.float32)
    M = jnp.asarray(np.random.default_rng(2).uniform(0.5, 1.5, n), jnp.float32)
    tree = Tree.from_particles(
        P, M, tree_type="radix", build_mode="static_radix", leaf_size=leaf
    )
    topo = tree.topology
    from yggdrax.tree_moments import compute_tree_mass_moments

    com = compute_tree_mass_moments(
        topo, tree.positions_sorted, tree.masses_sorted
    ).center_of_mass
    return topo, jnp.asarray(com, jnp.float32)


@pytest.mark.parametrize("order", trim([5, 3]))
def test_m2m_levels_interpret_matches_the_level_loop(tree_data, order):
    topo, com = tree_data
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    C = (order + 1) ** 2
    rng = np.random.default_rng(4)
    leaves = jnp.asarray(rng.standard_normal((total, C)), jnp.float32)
    leaves = leaves.at[:num_internal].set(0.0)  # only leaves carry P2M
    num_levels = int(jnp.max(topo.node_level)) + 1
    offs = topo.level_offsets
    width = int(jnp.max(offs[1:] - offs[:-1]))
    ref = aggregate_m2m_real_by_level(
        leaves,
        com,
        topo.left_child,
        topo.right_child,
        topo.nodes_by_level,
        offs,
        order=order,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=width,
    )
    got = m2m_real_levels_pallas(
        leaves,
        com,
        topo.left_child,
        topo.right_child,
        topo.nodes_by_level,
        offs,
        order=order,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=width,
        interpret=True,
    )
    assert np.array_equal(
        np.asarray(got)[num_internal:], np.asarray(leaves)[num_internal:]
    )
    r, g = np.asarray(ref), np.asarray(got)
    scale = np.abs(r).max()
    assert np.allclose(g, r, rtol=2e-5, atol=2e-5 * scale)
    assert np.abs(g[:num_internal]).max() > 0  # non-vacuous: the root got something


@pytest.mark.parametrize("order", trim([5, 3]))
def test_l2l_levels_interpret_matches_the_cascade(tree_data, order):
    topo, com = tree_data
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    C = (order + 1) ** 2
    rng = np.random.default_rng(5)
    locals0 = jnp.asarray(rng.standard_normal((total, C)), jnp.float32)
    num_levels = int(jnp.max(topo.node_level)) + 1
    offs = topo.level_offsets
    width = int(jnp.max(offs[1:] - offs[:-1]))
    ref = _propagate_solidfmm_locals_by_level(
        locals0 + 0.0,
        com,
        topo.left_child,
        topo.right_child,
        topo.node_level,  # donated by its jit
        order=order,
        rotation="solidfmm",
        total_nodes=total,
        basis_mode="real",
        num_levels=num_levels - 1,
    )
    got = l2l_real_levels_pallas(
        locals0,
        com,
        topo.parent,
        topo.nodes_by_level,
        offs,
        order=order,
        num_levels=num_levels,
        level_batch_width=width,
        interpret=True,
    )
    r, g = np.asarray(ref), np.asarray(got)
    scale = np.abs(r).max()
    assert np.allclose(g, r, rtol=2e-5, atol=2e-5 * scale)
    assert not np.allclose(g, np.asarray(locals0))  # non-vacuous


def test_levels_are_jittable(tree_data):
    topo, com = tree_data
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    order = 3
    C = (order + 1) ** 2
    x = jnp.asarray(np.random.default_rng(6).standard_normal((total, C)), jnp.float32)
    num_levels = int(jnp.max(topo.node_level)) + 1
    offs = topo.level_offsets
    width = int(jnp.max(offs[1:] - offs[:-1]))
    f = jax.jit(
        lambda x, com: m2m_real_levels_pallas(
            x,
            com,
            topo.left_child,
            topo.right_child,
            topo.nodes_by_level,
            offs,
            order=order,
            num_internal=num_internal,
            num_levels=num_levels,
            level_batch_width=width,
            interpret=True,
        )
    )
    g = jax.jit(
        lambda x, com: l2l_real_levels_pallas(
            x,
            com,
            topo.parent,
            topo.nodes_by_level,
            offs,
            order=order,
            num_levels=num_levels,
            level_batch_width=width,
            interpret=True,
        )
    )
    assert np.all(np.isfinite(np.asarray(f(x, com)))) and np.all(
        np.isfinite(np.asarray(g(x, com)))
    )
