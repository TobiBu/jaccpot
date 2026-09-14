"""Reverse of the per-leaf Pallas P2M vs ``jax.vjp`` of the batched pure-JAX P2M (fp64, interpret).

Both halves: positions AND masses AND the leaf centres, on a cell tree that carries
empty padding leaves and single-particle leaves (``delta == 0`` exactly, where the
floored radii carry the derivative).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax._tree_impl import build_static_cells_tree
from yggdrax.bounds import infer_bounds
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.pallas.p2m_real_leaf import (
    p2m_real_leaves_pallas,
    p2m_real_leaves_pallas_cvjp,
    pallas_p2m_real_leaf_supported,
)
from jaccpot.upward.real_tree_expansions import _p2m_leaves_real

jax.config.update("jax_enable_x64", True)


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _native_or_skip(interpret: bool) -> None:
    """Native Triton lowering needs an Ampere+ GPU; interpret mode runs anywhere."""
    if not interpret and not pallas_p2m_real_leaf_supported():
        pytest.skip("native Pallas GPU lowering not available here")


@pytest.mark.parametrize("interpret", [True, False])
@pytest.mark.parametrize("order", [2, 4, 5])
def test_p2m_reverse_matches_vjp_of_the_batched_reference(order, interpret):
    _native_or_skip(interpret)
    n, leaf = 2000, 16
    dtype = jnp.float64
    P = jnp.asarray(_plummer(n, 1), dtype)
    M = jnp.asarray(np.random.default_rng(2).uniform(0.5, 1.5, n), dtype)
    topo, ps, ms, inv = build_static_cells_tree(
        P, M, infer_bounds(P), leaf_size=leaf, leaf_capacity=1024, return_reordered=True
    )
    ni = int(topo.left_child.shape[0])
    tot = int(topo.parent.shape[0])
    ranges = np.asarray(topo.node_ranges)[ni:]
    counts = ranges[:, 1] - ranges[:, 0] + 1
    assert (counts <= 0).any() and (
        counts == 1
    ).any()  # padding leaves and single-particle leaves
    com = jnp.asarray(compute_tree_mass_moments(topo, ps, ms).center_of_mass, dtype)
    C = (order + 1) ** 2
    cot = jnp.asarray(np.random.default_rng(3 + order).standard_normal((tot, C)), dtype)

    def ref(p, m, c):
        return _p2m_leaves_real(
            topo.node_ranges,
            p,
            m,
            c,
            order=order,
            max_leaf_size=leaf,
            num_internal=ni,
            total_nodes=tot,
            leaf_batch_size=256,
        )

    def got(p, m, c_leaf):
        return p2m_real_leaves_pallas_cvjp(
            p,
            m,
            c_leaf,
            topo.node_ranges[ni:],
            order,
            ni,
            tot,
            leaf,
            interpret,
            "triton",
            None,
        )

    out_r, vjp_r = jax.vjp(ref, ps, ms, com)
    out_g, vjp_g = jax.vjp(got, ps, ms, com[ni:])
    assert _rel(out_g, out_r) < 1e-12
    pb_r, mb_r, cb_r = vjp_r(cot)
    pb_g, mb_g, cb_g = vjp_g(cot)
    for name, g in (("positions", pb_g), ("masses", mb_g), ("centres", cb_g)):
        assert np.all(np.isfinite(np.asarray(g))), name
    assert (
        np.linalg.norm(np.asarray(cb_r[ni:])) > 0
        and np.linalg.norm(np.asarray(pb_r)) > 0
    )
    assert np.all(np.asarray(cb_r[:ni]) == 0)  # the reference only touches leaf centres
    assert _rel(pb_g, pb_r) < 1e-9, f"positions rel-L2 {_rel(pb_g, pb_r):.3e}"
    assert _rel(mb_g, mb_r) < 1e-9, f"masses rel-L2 {_rel(mb_g, mb_r):.3e}"
    assert _rel(cb_g, cb_r[ni:]) < 1e-9, f"centres rel-L2 {_rel(cb_g, cb_r[ni:]):.3e}"
    # the cvjp forward is the plain forward
    plain = p2m_real_leaves_pallas(
        ps,
        ms,
        com[ni:],
        topo.node_ranges[ni:],
        order=order,
        num_internal=ni,
        total_nodes=tot,
        leaf_width=leaf,
        interpret=interpret,
    )
    assert np.array_equal(np.asarray(plain), np.asarray(out_g))
