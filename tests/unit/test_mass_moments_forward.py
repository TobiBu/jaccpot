"""Forward-mode mass moments: yggdrax's primal bitwise, a one-direction-at-a-time tangent.

Under ``ad_mode="forward"`` :func:`~jaccpot.upward._mass_moments_vjp.tree_mass_moments`
is a ``custom_jvp`` whose tangent op ``lax.map``s over the directions (JAX's own jvp
of the float64 prefix sums kept ``f64[K, N, 3]`` rows alive). Checked against JAX's
jvp of :func:`~yggdrax.tree_moments.compute_tree_mass_moments`: the primal bitwise,
the tangents to round-off, one ``jacfwd`` column per direction, and dead (zero)
mass tangents skipped.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")

from jaccpot.runtime.grad_options import ad_mode_override  # noqa: E402
from jaccpot.upward._mass_moments_vjp import tree_mass_moments  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


@pytest.fixture(scope="module")
def system():
    from yggdrax._tree_impl import (
        build_static_cells_tree,  # pyright: ignore[reportAttributeAccessIssue]
    )
    from yggdrax.bounds import infer_bounds

    rng = np.random.default_rng(3)
    pos = jnp.asarray(rng.normal(size=(300, 3)))
    mass = jnp.asarray(rng.uniform(0.5, 1.5, 300))
    topo, ps, ms, _ = build_static_cells_tree(
        pos,
        mass,
        infer_bounds(pos),
        leaf_size=8,
        leaf_capacity=256,
        return_reordered=True,
    )
    return topo, ps, ms


def _reference(topo, ps, ms):
    from yggdrax.tree_moments import compute_tree_mass_moments

    out = compute_tree_mass_moments(topo, ps, ms)
    return out.mass, out.center_of_mass


def _forward(topo, ps, ms):
    with ad_mode_override("forward"):
        out = tree_mass_moments(topo, ps, ms)
    return out.mass, out.center_of_mass


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_the_jvp_primal_is_bitwise_and_the_tangent_matches(system, dtype):
    topo, ps, ms = system
    ps, ms = ps.astype(dtype), ms.astype(dtype)
    rng = np.random.default_rng(5)
    tp = jnp.asarray(rng.normal(size=ps.shape), dtype)
    tm = jnp.asarray(rng.normal(size=ms.shape), dtype)

    ref, ref_t = jax.jvp(lambda p, m: _reference(topo, p, m), (ps, ms), (tp, tm))
    out, out_t = jax.jit(
        lambda p, m, a, b: jax.jvp(lambda p, m: _forward(topo, p, m), (p, m), (a, b))
    )(ps, ms, tp, tm)
    for a, b in zip(out, ref):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    rtol = 1e-5 if dtype == jnp.float32 else 1e-12
    for a, b in zip(out_t, ref_t):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=rtol)


@pytest.mark.parametrize("k", [1, 3, 8, 11])
def test_jacfwd_columns_are_separate_jvps(system, k):
    topo, ps, ms = system
    rng = np.random.default_rng(k)
    dirs = jnp.asarray(rng.normal(size=(k,) + ps.shape))

    def com_of(s):
        p = ps + jnp.tensordot(s, dirs, axes=1)
        return _forward(topo, p, ms)[1]

    jac = jax.jit(jax.jacfwd(com_of))(jnp.zeros((k,)))
    for j in range(k):
        _, col = jax.jvp(lambda p: _reference(topo, p, ms)[1], (ps,), (dirs[j],))
        np.testing.assert_allclose(
            np.asarray(jac[..., j]), np.asarray(col), rtol=1e-12, atol=1e-12
        )


def test_a_dead_mass_tangent_is_skipped(system):
    topo, ps, ms = system
    tp = jnp.asarray(np.random.default_rng(9).normal(size=ps.shape))
    _, out_t = jax.jvp(lambda p: _forward(topo, p, ms), (ps,), (tp,))
    _, ref_t = jax.jvp(lambda p: _reference(topo, p, ms), (ps,), (tp,))
    np.testing.assert_allclose(
        np.asarray(out_t[1]), np.asarray(ref_t[1]), rtol=1e-12, atol=1e-12
    )
    assert not np.any(np.asarray(out_t[0]))


def test_reverse_mode_is_untouched(system):
    topo, ps, ms = system
    out = tree_mass_moments(topo, ps, ms)
    ref = _reference(topo, ps, ms)
    assert np.array_equal(np.asarray(out.mass), np.asarray(ref[0]))
    g = jax.grad(lambda p: jnp.sum(tree_mass_moments(topo, p, ms).center_of_mass ** 2))(
        ps
    )
    assert np.all(np.isfinite(np.asarray(g)))
