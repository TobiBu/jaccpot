"""One-node-per-lane M2M / L2L cascades against the pure-JAX level loops (interpret mode, CPU).

The same result to round-off (different operation order: the lane kernels bake
the rotation blocks into straight-line code and build ``r^k`` by repeated
multiplication). Pinned on a real static-radix tree with lane counts that do and
do not divide the level widths, on a degenerate edge (a child at its parent's
centre: the identity translation), and under ``jax.jit``; plus the custom VJP's
forward, which is this kernel. Until the 2026-10 cleanup (X5) the reference here
was the level forward kernel, which ``JACCPOT_CASCADE_KERNEL=level`` selected (that
variable is read nowhere since Z-b).
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
from jaccpot.pallas.cascade_real_level import l2l_real_levels_pallas_cvjp
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
def test_m2m_lanes_match_the_level_loop(tree_data, order, k_lanes):
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
    loop_kw = {k: v for k, v in kw.items() if k != "interpret"}
    ref = aggregate_m2m_real_by_level(*args, topo.level_offsets, **loop_kw)
    got = m2m_real_levels_lanes_pallas(*args, topo.level_offsets, k_lanes=k_lanes, **kw)
    assert np.array_equal(
        np.asarray(got)[num_internal:], np.asarray(leaves)[num_internal:]
    )
    _close(got, ref)
    assert np.abs(np.asarray(got)[0]).max() > 0  # the root received


@pytest.mark.parametrize("order", trim([5, 3]))
@pytest.mark.parametrize("k_lanes", trim([32, 7]))
def test_l2l_lanes_match_the_cascade(tree_data, order, k_lanes):
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
    ref = _propagate_solidfmm_locals_by_level(
        locals0 + 0.0,
        com,
        topo.left_child,
        topo.right_child,
        topo.node_level,  # donated by its jit
        order=order,
        total_nodes=total,
        num_levels=num_levels - 1,
    )
    got = l2l_real_levels_lanes_pallas(*args, k_lanes=k_lanes, **kw)
    _close(got, ref)
    assert np.array_equal(np.asarray(got)[0], np.asarray(locals0)[0])  # root kept
    assert not np.allclose(np.asarray(got), np.asarray(locals0))


def test_lanes_jit_and_the_vjp_forward(tree_data):
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

    assert np.array_equal(cvjp(), lanes)


def test_eager_calls_compile_once_and_trace_the_same_program(tree_data):
    """An eager call (the caps-only prepare's upward pass is one, at every fresh
    run) goes through one program per static configuration: a second eager call
    compiles nothing. Before, ``lax.fori_loop`` recompiled its level loop at every
    eager call (its body is a closure made per call): ~30 s of Pallas compile per
    ``ProductionRunner.run`` at 1e6 on an A100 (S4). The eager result is the traced
    call's, bitwise, for both cascades."""
    from jax import monitoring

    topo, com, num_levels, width = tree_data
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    order = 3
    C = (order + 1) ** 2
    x = jnp.asarray(np.random.default_rng(7).standard_normal((total, C)), jnp.float32)
    m2m_args = (x, com, topo.left_child, topo.right_child, topo.nodes_by_level)
    m2m_kw = dict(
        order=order,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=width,
        interpret=True,
    )
    l2l_args = (x, com, topo.parent, topo.nodes_by_level, topo.level_offsets)
    l2l_kw = dict(
        order=order, num_levels=num_levels, level_batch_width=width, interpret=True
    )

    def m2m(*a):
        return m2m_real_levels_lanes_pallas(*a, topo.level_offsets, **m2m_kw)

    def l2l(*a):
        return l2l_real_levels_lanes_pallas(*a, **l2l_kw)

    compiles = []

    def listen(name, duration, **_):
        if name == "/jax/core/compile/backend_compile_duration":
            compiles.append(name)

    monitoring.register_event_duration_secs_listener(listen)
    try:
        first = [np.asarray(m2m(*m2m_args)), np.asarray(l2l(*l2l_args))]
        before = len(compiles)
        again = [np.asarray(m2m(*m2m_args)), np.asarray(l2l(*l2l_args))]
        assert len(compiles) == before  # nothing compiled for the second calls
    finally:
        monitoring.unregister_event_duration_listener(listen)
    traced = [np.asarray(jax.jit(m2m)(*m2m_args)), np.asarray(jax.jit(l2l)(*l2l_args))]
    for a, b, c in zip(first, again, traced):
        assert np.array_equal(a, b)
        assert np.array_equal(a, c)
