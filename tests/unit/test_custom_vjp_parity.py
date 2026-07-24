"""Bit-for-bit verification harness for analytic custom_vjp kernels.

Every analytic reverse rule installed for the differentiable FMM must produce
gradients identical (to round-off) to autodiff through the pure-JAX primal. This
module provides a reusable checker and the per-rule parity tests. A custom rule
can never silently encode a wrong gradient if these stay green.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def assert_vjp_matches(f_custom, f_ref, primals, *, seed=0, rtol=1e-11, atol=1e-11):
    """Assert f_custom and f_ref agree in value AND vjp for random cotangents.

    ``f_custom``/``f_ref`` map ``*primals -> pytree of arrays``; they must be the
    same mathematical function (f_ref = autodiff reference, f_custom = custom_vjp
    wrapped). Checks the forward outputs match and the cotangents w.r.t. every
    primal match for a random output cotangent.
    """
    out_c, vjp_c = jax.vjp(f_custom, *primals)
    out_r, vjp_r = jax.vjp(f_ref, *primals)

    leaves_c = jax.tree_util.tree_leaves(out_c)
    leaves_r = jax.tree_util.tree_leaves(out_r)
    for a, b in zip(leaves_c, leaves_r):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)

    key = jax.random.PRNGKey(seed)
    cot = []
    for leaf in leaves_c:
        key, sub = jax.random.split(key)
        cot.append(jax.random.normal(sub, leaf.shape, dtype=leaf.dtype))
    cot = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(out_c), cot)

    grads_c = vjp_c(cot)
    grads_r = vjp_r(cot)
    for gc, gr in zip(
        jax.tree_util.tree_leaves(grads_c), jax.tree_util.tree_leaves(grads_r)
    ):
        np.testing.assert_allclose(np.asarray(gc), np.asarray(gr), rtol=rtol, atol=atol)


# --------------------------------------------------------------------------
# Real-basis L2P custom_vjp
# --------------------------------------------------------------------------
from jaccpot.operators.real_harmonics import (  # noqa: E402
    evaluate_local_real,
    evaluate_local_real_with_grad,
)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_real_l2p_custom_vjp_matches_autodiff(order):
    rng = np.random.default_rng(order)
    ncoeff = (order + 1) ** 2
    local = jnp.asarray(rng.normal(size=(ncoeff,)), dtype=jnp.float64)
    delta = jnp.asarray(rng.normal(size=(3,)), dtype=jnp.float64)

    def f_custom(c, d):
        return evaluate_local_real_with_grad(c, d, order=order)

    def f_ref(c, d):
        def phi(dd):
            return evaluate_local_real(c, dd, order=order)

        potential, grad = jax.value_and_grad(phi)(d)
        return grad, potential

    assert_vjp_matches(f_custom, f_ref, (local, delta))


@pytest.mark.parametrize("order", [2, 4])
def test_real_l2p_custom_vjp_batched_matches_autodiff(order):
    """The rule must compose through vmap (as it does in the assembled L2P)."""
    rng = np.random.default_rng(100 + order)
    ncoeff = (order + 1) ** 2
    n = 8
    local = jnp.asarray(rng.normal(size=(ncoeff,)), dtype=jnp.float64)
    deltas = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)

    def f_custom(c, d):
        return jax.vmap(
            lambda dd: evaluate_local_real_with_grad(c, dd, order=order)[0]
        )(d)

    def f_ref(c, d):
        def one(dd):
            return jax.grad(lambda x: evaluate_local_real(c, x, order=order))(dd)

        return jax.vmap(one)(d)

    assert_vjp_matches(f_custom, f_ref, (local, deltas))
