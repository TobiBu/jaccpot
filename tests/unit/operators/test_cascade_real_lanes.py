"""One-node-per-lane M2M / L2L cascades against the level kernels (interpret mode, CPU).

Same arguments, same result to round-off (different operation order: the lane
kernels bake the rotation blocks into straight-line code and build ``r^k`` by
repeated multiplication). Pinned on a real static-radix tree with lane counts
that do and do not divide the level widths, on a degenerate edge (a child at its
parent's centre: the identity translation), and under ``jax.jit``; plus the
custom VJP's forward switch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.tree import Tree

from jaccpot.pallas.cascade_real_lanes import (
    l2l_real_levels_lanes_pallas,
    m2m_real_levels_lanes_pallas,
)
from jaccpot.pallas.cascade_real_level import (
    l2l_real_levels_pallas,
    l2l_real_levels_pallas_cvjp,
    m2m_real_levels_pallas,
)
from tests.unit._typecheck_budget import trim


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


@pytest.fixture(scope="module")
def tree_data():
    n, leaf = 2000, 16
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
    num_levels = int(jnp.max(topo.node_level)) + 1
    offs = topo.level_offsets
    width = int(jnp.max(offs[1:] - offs[:-1]))
    return topo, jnp.asarray(com, jnp.float32), num_levels, width


def _close(got, ref):
    g, r = np.asarray(got), np.asarray(ref)
    scale = np.abs(r).max()
    np.testing.assert_allclose(g, r, rtol=2e-5, atol=2e-5 * scale)


@pytest.mark.parametrize("order", trim([5, 3]))
@pytest.mark.parametrize("k_lanes", trim([32, 7]))
def test_m2m_lanes_match_the_level_kernel(tree_data, order, k_lanes):
    topo, com, num_levels, width = tree_data
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    C = (order + 1) ** 2
    leaves = jnp.asarray(
        np.random.default_rng(4).standard_normal((total, C)), jnp.float32
    )
    leaves = leaves.at[:num_internal].set(0.0)
    # a degenerate edge: the root's left child sits at the root's centre
    com = com.at[int(topo.left_child[0])].set(com[0])
    kw = dict(
        order=order,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=width,
        interpret=True,
    )
    args = (leaves, com, topo.left_child, topo.right_child, topo.nodes_by_level)
    ref = m2m_real_levels_pallas(*args, topo.level_offsets, **kw)
    got = m2m_real_levels_lanes_pallas(*args, topo.level_offsets, k_lanes=k_lanes, **kw)
    assert np.array_equal(
        np.asarray(got)[num_internal:], np.asarray(leaves)[num_internal:]
    )
    _close(got, ref)
    assert np.abs(np.asarray(got)[0]).max() > 0  # the root received


@pytest.mark.parametrize("order", trim([5, 3]))
@pytest.mark.parametrize("k_lanes", trim([32, 7]))
def test_l2l_lanes_match_the_level_kernel(tree_data, order, k_lanes):
    topo, com, num_levels, width = tree_data
    total = int(topo.parent.shape[0])
    C = (order + 1) ** 2
    locals0 = jnp.asarray(
        np.random.default_rng(5).standard_normal((total, C)), jnp.float32
    )
    com = com.at[int(topo.left_child[0])].set(com[0])
    kw = dict(
        order=order, num_levels=num_levels, level_batch_width=width, interpret=True
    )
    args = (locals0, com, topo.parent, topo.nodes_by_level, topo.level_offsets)
    ref = l2l_real_levels_pallas(*args, **kw)
    got = l2l_real_levels_lanes_pallas(*args, k_lanes=k_lanes, **kw)
    _close(got, ref)
    assert np.array_equal(np.asarray(got)[0], np.asarray(locals0)[0])  # root kept
    assert not np.allclose(np.asarray(got), np.asarray(locals0))


def test_lanes_jit_and_the_vjp_forward_switch(tree_data, monkeypatch):
    topo, com, num_levels, width = tree_data
    total = int(topo.parent.shape[0])
    order = 3
    C = (order + 1) ** 2
    x = jnp.asarray(np.random.default_rng(6).standard_normal((total, C)), jnp.float32)
    f = jax.jit(
        lambda x, com: l2l_real_levels_lanes_pallas(
            x,
            com,
            topo.parent,
            topo.nodes_by_level,
            topo.level_offsets,
            order=order,
            num_levels=num_levels,
            level_batch_width=width,
            interpret=True,
        )
    )
    lanes = np.asarray(f(x, com))
    assert np.all(np.isfinite(lanes))

    def cvjp():
        return np.asarray(
            l2l_real_levels_pallas_cvjp(
                x,
                com,
                topo.parent,
                topo.left_child,
                topo.right_child,
                topo.nodes_by_level,
                topo.level_offsets,
                order,
                num_levels,
                width,
                True,
                "triton",
                4,
            )
        )

    monkeypatch.setenv("JACCPOT_CASCADE_KERNEL", "lanes")
    assert np.array_equal(cvjp(), lanes)
    monkeypatch.setenv("JACCPOT_CASCADE_KERNEL", "level")
    level = cvjp()
    assert not np.array_equal(level, lanes)  # the switch selects a different kernel
    _close(lanes, level)
