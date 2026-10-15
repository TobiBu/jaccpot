"""Reverse of the per-level Pallas M2M / L2L cascades vs ``jax.vjp`` of the pure-JAX level loops.

Plan ``fast-gradients-for-the-sub10ms-fmm``, Phase 4: both halves of every rule --
w.r.t. the coefficients AND w.r.t. the centres -- elementwise, in fp64, interpret
mode. A rule that is right in the coefficients and wrong in the geometry passes most
naive tests, so the centre half is asserted separately and must be non-vacuous.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.tree import Tree

from jaccpot.pallas.cascade_real_level import (
    l2l_real_levels_pallas_cvjp,
    m2m_real_levels_pallas_cvjp,
    pallas_cascade_level_supported,
)
from jaccpot.runtime.kernels._l2l import _propagate_solidfmm_locals_by_level
from jaccpot.upward.real_tree_expansions import aggregate_m2m_real_by_level

jax.config.update("jax_enable_x64", True)


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
    # 400, not 1500: these run in Pallas INTERPRET mode, which simulates the
    # grid in Python, so cost is linear in the node count and this fixture is
    # the dominant term in `test-runtime-typecheck` (whole of tests/unit under
    # jaxtyping+beartype, 60 min cap). 400 particles at leaf 16 still give a
    # multi-level tree with internal nodes at several depths, which is all the
    # cascade adjoints need to be exercised; the order sweep is what carries
    # the coverage here, and it is unchanged.
    n, leaf = 400, 16
    P = jnp.asarray(_plummer(n, 1), jnp.float64)
    M = jnp.asarray(np.random.default_rng(2).uniform(0.5, 1.5, n), jnp.float64)
    tree = Tree.from_particles(
        P, M, tree_type="radix", build_mode="static_radix", leaf_size=leaf
    )
    topo = tree.topology
    from yggdrax.tree_moments import compute_tree_mass_moments

    com = compute_tree_mass_moments(
        topo, tree.positions_sorted, tree.masses_sorted
    ).center_of_mass
    return topo, jnp.asarray(com, jnp.float64)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


# Interpret mode (CPU CI) runs orders 2 and 4 -- the low bracket and the production
# default (`max_order=4`); order 5 was the costliest interpret case (45 s for M2M in
# CI) and repeated what they cover (2026-10-12). The native Triton cases, which skip
# without an Ampere+ GPU and cost little where they run, keep all three orders.
_ORDER_CASES = [
    (2, True),
    (4, True),
    (2, False),
    (4, False),
    (5, False),
]


def _native_or_skip(interpret: bool) -> None:
    """Native Triton lowering needs an Ampere+ GPU; interpret mode runs anywhere."""
    if not interpret and not pallas_cascade_level_supported():
        pytest.skip("native Pallas GPU lowering not available here")


@pytest.mark.parametrize("order,interpret", _ORDER_CASES)
def test_m2m_reverse_matches_vjp_of_the_level_loop_in_both_halves(
    tree_data, order, interpret
):
    _native_or_skip(interpret)
    topo, com = tree_data
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    C = (order + 1) ** 2
    rng = np.random.default_rng(4 + order)
    leaves = (
        jnp.asarray(rng.standard_normal((total, C)), jnp.float64)
        .at[:num_internal]
        .set(0.0)
    )
    cot = jnp.asarray(rng.standard_normal((total, C)), jnp.float64)
    num_levels = int(jnp.max(topo.node_level)) + 1
    offs = topo.level_offsets
    width = int(jnp.max(offs[1:] - offs[:-1]))

    def ref(x, c):
        return aggregate_m2m_real_by_level(
            x,
            c,
            topo.left_child,
            topo.right_child,
            topo.nodes_by_level,
            offs,
            order=order,
            num_internal=num_internal,
            num_levels=num_levels,
            level_batch_width=width,
        )

    def got(x, c):
        return m2m_real_levels_pallas_cvjp(
            x,
            c,
            topo.left_child,
            topo.right_child,
            topo.parent,
            topo.nodes_by_level,
            offs,
            order,
            num_internal,
            num_levels,
            width,
            interpret,
            "triton",
            4,
        )

    out_r, vjp_r = jax.vjp(ref, leaves, com)
    out_g, vjp_g = jax.vjp(got, leaves, com)
    assert _rel(out_g, out_r) < 1e-10  # forward untouched
    xb_r, cb_r = vjp_r(cot)
    xb_g, cb_g = vjp_g(cot)
    assert np.all(np.isfinite(np.asarray(xb_g))) and np.all(
        np.isfinite(np.asarray(cb_g))
    )
    assert np.linalg.norm(np.asarray(cb_r)) > 0  # the geometry half is non-vacuous
    assert np.all(
        np.asarray(xb_g)[:num_internal] == 0
    )  # overwritten input rows get zero
    assert _rel(xb_g, xb_r) < 1e-9, f"coefficient half rel-L2 {_rel(xb_g, xb_r):.3e}"
    assert _rel(cb_g, cb_r) < 1e-9, f"centre half rel-L2 {_rel(cb_g, cb_r):.3e}"


@pytest.mark.parametrize("order,interpret", _ORDER_CASES)
def test_l2l_reverse_matches_vjp_of_the_cascade_in_both_halves(
    tree_data, order, interpret
):
    _native_or_skip(interpret)
    topo, com = tree_data
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    C = (order + 1) ** 2
    rng = np.random.default_rng(9 + order)
    locals0 = jnp.asarray(rng.standard_normal((total, C)), jnp.float64)
    cot = jnp.asarray(rng.standard_normal((total, C)), jnp.float64)
    num_levels = int(jnp.max(topo.node_level)) + 1
    offs = topo.level_offsets
    width = int(jnp.max(offs[1:] - offs[:-1]))

    def ref(x, c):
        return _propagate_solidfmm_locals_by_level(
            x + 0.0,
            c,
            topo.left_child,
            topo.right_child,
            topo.node_level,
            order=order,
            total_nodes=total,
            num_levels=num_levels - 1,
        )

    def got(x, c):
        return l2l_real_levels_pallas_cvjp(
            x,
            c,
            topo.parent,
            topo.left_child,
            topo.right_child,
            topo.nodes_by_level,
            offs,
            order,
            num_levels,
            width,
            interpret,
            "triton",
            4,
        )

    out_r, vjp_r = jax.vjp(ref, locals0, com)
    out_g, vjp_g = jax.vjp(got, locals0, com)
    assert _rel(out_g, out_r) < 1e-10
    xb_r, cb_r = vjp_r(cot)
    xb_g, cb_g = vjp_g(cot)
    assert np.all(np.isfinite(np.asarray(xb_g))) and np.all(
        np.isfinite(np.asarray(cb_g))
    )
    assert np.linalg.norm(np.asarray(cb_r)) > 0
    assert _rel(xb_g, xb_r) < 1e-9, f"coefficient half rel-L2 {_rel(xb_g, xb_r):.3e}"
    assert _rel(cb_g, cb_r) < 1e-9, f"centre half rel-L2 {_rel(cb_g, cb_r):.3e}"


@pytest.fixture(scope="module")
def axis_tree_data():
    """A tree whose edges include on-axis ones: four clusters stacked along z.

    The clusters share one ``(x, y)`` point set and masses, so the parent of two
    clusters sits on their common axis: of the 14 edges 6 are inside the band
    ``rho^2 <= eps r^2`` (5 at ``rho == 0`` exactly, one at ``rho/r = 1.9e-17``,
    the ulp two summation orders leave between equal centres).
    """
    rng = np.random.default_rng(3)
    xy = rng.normal(size=(8, 2)) * 0.35
    masses = np.tile(rng.uniform(0.6, 1.4, size=8), 4)
    positions = np.concatenate(
        [np.column_stack([xy, np.full(8, (i - 1.5) * 6.0)]) for i in range(4)]
    )
    tree = Tree.from_particles(
        jnp.asarray(positions),
        jnp.asarray(masses),
        tree_type="radix",
        build_mode="static_radix",
        leaf_size=4,
    )
    topo = tree.topology
    from yggdrax.tree_moments import compute_tree_mass_moments

    com = compute_tree_mass_moments(
        topo, tree.positions_sorted, tree.masses_sorted
    ).center_of_mass
    return topo, jnp.asarray(com, jnp.float64)


def _on_axis_edge_count(topo, com) -> int:
    parent, centres = np.asarray(topo.parent), np.asarray(com)
    has_parent = parent >= 0
    d = centres[has_parent] - centres[parent[has_parent]]
    rho2 = d[:, 0] ** 2 + d[:, 1] ** 2
    band = rho2 <= np.finfo(np.float64).eps * (rho2 + d[:, 2] ** 2)
    return int(np.sum(band & (d[:, 2] != 0)))


@pytest.mark.parametrize("order,interpret", _ORDER_CASES)
def test_reverse_matches_the_operator_rule_on_axis_aligned_edges(
    axis_tree_data, order, interpret
):
    """On-axis edges: the centre half equals the pure-JAX cascades', not zero.

    The level loops differentiate operators that carry the analytic transverse
    derivative (``operators/_transverse_degeneracy_jvp``); the reverse kernels
    differentiate the polar route, whose guards hand back zero on the axis and
    a cancellation-dominated value just off it. Before the kernels took the
    analytic limit, the centre half missed by 0.30-2.1 relative here (interpret
    mode, orders 2 and 4) while the coefficient half matched to 1e-15 -- and the
    tests above, on a Plummer tree with no edge in the band, could not see it.
    """
    _native_or_skip(interpret)
    topo, com = axis_tree_data
    assert _on_axis_edge_count(topo, com) >= 6
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    C = (order + 1) ** 2
    rng = np.random.default_rng(40 + order)
    num_levels = int(jnp.max(topo.node_level)) + 1
    offs = topo.level_offsets
    width = int(jnp.max(offs[1:] - offs[:-1]))
    leaves = (
        jnp.asarray(rng.standard_normal((total, C)), jnp.float64)
        .at[:num_internal]
        .set(0.0)
    )
    locals0 = jnp.asarray(rng.standard_normal((total, C)), jnp.float64)
    cot = jnp.asarray(rng.standard_normal((total, C)), jnp.float64)

    def m2m_ref(x, c):
        return aggregate_m2m_real_by_level(
            x,
            c,
            topo.left_child,
            topo.right_child,
            topo.nodes_by_level,
            offs,
            order=order,
            num_internal=num_internal,
            num_levels=num_levels,
            level_batch_width=width,
        )

    def m2m_got(x, c):
        return m2m_real_levels_pallas_cvjp(
            x,
            c,
            topo.left_child,
            topo.right_child,
            topo.parent,
            topo.nodes_by_level,
            offs,
            order,
            num_internal,
            num_levels,
            width,
            interpret,
            "triton",
            4,
        )

    def l2l_ref(x, c):
        return _propagate_solidfmm_locals_by_level(
            x + 0.0,
            c,
            topo.left_child,
            topo.right_child,
            topo.node_level,
            order=order,
            total_nodes=total,
            num_levels=num_levels - 1,
        )

    def l2l_got(x, c):
        return l2l_real_levels_pallas_cvjp(
            x,
            c,
            topo.parent,
            topo.left_child,
            topo.right_child,
            topo.nodes_by_level,
            offs,
            order,
            num_levels,
            width,
            interpret,
            "triton",
            4,
        )

    for name, ref, got, x in (
        ("m2m", m2m_ref, m2m_got, leaves),
        ("l2l", l2l_ref, l2l_got, locals0),
    ):
        _, vjp_r = jax.vjp(ref, x, com)
        _, vjp_g = jax.vjp(got, x, com)
        xb_r, cb_r = vjp_r(cot)
        xb_g, cb_g = vjp_g(cot)
        assert np.all(np.isfinite(np.asarray(cb_g)))
        assert (
            _rel(xb_g, xb_r) < 1e-9
        ), f"{name} coefficient half {_rel(xb_g, xb_r):.3e}"
        assert _rel(cb_g, cb_r) < 1e-9, f"{name} centre half {_rel(cb_g, cb_r):.3e}"


def test_reverse_is_finite_at_zero_and_on_axis_displacements():
    """delta == 0 (a single-child parent) and rho == 0 must give finite cotangents, not NaN."""
    from jaccpot.pallas.cascade_real_level import (
        _core_tables_to_jnp,
        _translate_rows,
        pack_centred,
    )

    order = 3
    t = _core_tables_to_jnp(order, jnp.float64)
    Bp = t["invfact"].shape[0]
    rows = pack_centred(
        jnp.asarray(np.random.default_rng(0).standard_normal((1, 16))), order=order
    )[0].reshape(Bp, -1)
    cot = jnp.ones_like(rows)
    for which in ("m2m", "l2l"):
        for d in ([0.0, 0.0, 0.0], [0.0, 0.0, 0.7], [0.0, 0.0, -0.7]):
            f = lambda r, dd: _translate_rows(r, dd, t, which, safe=True)  # noqa: E731
            out, vjp = jax.vjp(f, rows, tuple(jnp.asarray(v) for v in d))
            rb, db = vjp(cot)
            assert np.all(np.isfinite(np.asarray(out)))
            assert np.all(np.isfinite(np.asarray(rb))) and all(
                np.isfinite(float(v)) for v in db
            ), (which, d)
            # the primal is the unguarded one
            ref = _translate_rows(rows, tuple(jnp.asarray(v) for v in d), t, which)
            assert np.array_equal(np.asarray(out), np.asarray(ref))
