"""The softening kernels against their definitions, re-derived numerically.

``jaccpot.softening`` carries polynomials derived symbolically from each kernel's
density. Here every one is checked against quadrature of that density, so a
mistyped coefficient cannot hide: enclosed mass, force, potential, the radial
derivative, Newton past the support (bitwise), continuity at the edge, the
Plummer-equivalent support factor, finite values at ``r = 0`` and Newton at zero
softening.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from jaccpot.softening import (
    DEFAULT_SOFTENING_KERNEL,
    SOFTENING_KERNELS,
    masked_pair_factors,
    pair_factors,
    pair_softening_sq_derivative,
    resolve_softening_kernel,
    softening_params,
    softening_params_np,
    support_factor,
    support_radius,
)

_DENSITY = {
    "ferrers3": lambda q: 315.0 / (64.0 * np.pi) * (1.0 - q * q) ** 3,
    "wendland_c2": lambda q: 21.0 / (2.0 * np.pi) * (1.0 - q) ** 4 * (1.0 + 4.0 * q),
}
_COMPACT = tuple(_DENSITY)


def _mass(kernel, q):
    return quad(lambda s: 4.0 * np.pi * s * s * _DENSITY[kernel](s), 0.0, min(q, 1.0))[0]


def _psi(kernel, q):
    """psi(q) = m(q)/q + int_q^1 4 pi s W ds, in units h = G M = 1."""
    if q >= 1.0:
        return 1.0 / q
    outer = quad(lambda s: 4.0 * np.pi * s * _DENSITY[kernel](s), q, 1.0)[0]
    return (_mass(kernel, q) / q if q > 0 else 0.0) + outer


def _np_factors(kernel, r, eps=1.0):
    params = softening_params_np(kernel, eps)
    return pair_factors(
        np.asarray(r, np.float64) ** 2,
        params,
        kernel,
        potential=True,
        derivative=True,
        xp=np,
    )


def test_names_default_and_support_factors():
    assert DEFAULT_SOFTENING_KERNEL == "ferrers3"
    assert resolve_softening_kernel(None) == "ferrers3"
    assert resolve_softening_kernel(" Wendland_C2 ") == "wendland_c2"
    with pytest.raises(ValueError, match="softening_kernel"):
        resolve_softening_kernel("spline")
    assert set(SOFTENING_KERNELS) == {"ferrers3", "wendland_c2", "plummer"}
    assert support_factor("plummer") == 0.0
    assert support_radius("wendland_c2", 0.5) == 1.5
    # equal central potential to Plummer's -G m / eps: psi(0) h^-1 = 1 / eps
    for kernel in _COMPACT:
        assert np.isclose(_psi(kernel, 0.0), support_factor(kernel), rtol=1e-12)


@pytest.mark.parametrize("kernel", _COMPACT)
def test_force_potential_and_derivative_match_the_density(kernel):
    h = support_factor(kernel)  # eps = 1, so r is in units of eps
    q = np.concatenate([np.linspace(0.02, 0.98, 25), [1.0, 1.3, 2.0, 5.0]])
    r = q * h
    g, psi, dg = _np_factors(kernel, r)
    m = np.array([_mass(kernel, x) for x in q])
    np.testing.assert_allclose(g * r**3, m, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(psi * h, [_psi(kernel, x) for x in q], rtol=1e-12)
    # potential and force are one field: d psi / dr = -r g
    step = 1e-6
    _, psi_p, _ = _np_factors(kernel, r + step)
    _, psi_m, _ = _np_factors(kernel, r - step)
    np.testing.assert_allclose((psi_p - psi_m) / (2 * step), -r * g, rtol=1e-6)
    # (1/r) dg/dr against central differences
    g_p, _, _ = _np_factors(kernel, r + step)
    g_m, _, _ = _np_factors(kernel, r - step)
    np.testing.assert_allclose((g_p - g_m) / (2 * step) / r, dg, rtol=1e-5)


@pytest.mark.parametrize("kernel", _COMPACT)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_newton_past_the_support_bitwise_and_finite_at_zero(kernel, dtype):
    if dtype == jnp.float64 and not jax.config.jax_enable_x64:
        pytest.skip("x64 disabled")
    eps = 0.37
    h = support_radius(kernel, eps)
    # strictly past h; AT h, sqrt(h^2)/h may round below 1 and leave a 1-ulp tail
    r2 = jnp.asarray([(1.0001 * h) ** 2, (1.5 * h) ** 2, 40.0 * h * h, 0.0, h * h], dtype)
    params = softening_params(kernel, eps, dtype)
    g, psi, dg = pair_factors(r2, params, kernel, potential=True, derivative=True)
    s = jax.lax.rsqrt(r2[:3])
    s3 = s * s * s
    assert np.array_equal(np.asarray(g[:3]), np.asarray(s3))
    assert np.array_equal(np.asarray(psi[:3]), np.asarray(s))
    assert np.array_equal(np.asarray(dg[:3]), np.asarray(-3.0 * s3 * s * s))
    assert np.all(np.isfinite(np.asarray(g))) and np.all(np.isfinite(np.asarray(psi)))
    # r = 0: the centre of the kernel, Plummer-equivalent depth -1/eps
    np.testing.assert_allclose(float(psi[3]), 1.0 / eps, rtol=1e-6)
    # r = h: Newton to rounding
    np.testing.assert_allclose(float(g[4]), float(h) ** -3, rtol=1e-6)


@pytest.mark.parametrize("kernel", _COMPACT)
def test_force_is_continuously_differentiable_at_the_edge(kernel):
    h = support_factor(kernel)
    inside = _np_factors(kernel, h * (1 - 1e-9))
    outside = _np_factors(kernel, h * (1 + 1e-9))
    for a, b in zip(inside, outside):
        np.testing.assert_allclose(a, b, rtol=1e-7)


def test_plummer_is_the_historical_form():
    eps = 0.21
    r2 = np.array([0.0, 1e-4, 0.05, 2.0])
    g, psi, dg = pair_factors(
        r2, softening_params_np("plummer", eps), "plummer",
        potential=True, derivative=True, xp=np,
    )
    d2 = r2 + eps * eps
    np.testing.assert_allclose(g, d2**-1.5, rtol=1e-14)
    np.testing.assert_allclose(psi, d2**-0.5, rtol=1e-14)
    np.testing.assert_allclose(dg, -3.0 * d2**-2.5, rtol=1e-14)


@pytest.mark.parametrize("kernel", SOFTENING_KERNELS)
def test_zero_softening_is_newton(kernel):
    r2 = jnp.asarray([1e-6, 0.3, 7.0], jnp.float32)
    g, psi, _ = pair_factors(
        r2, softening_params(kernel, 0.0, jnp.float32), kernel, potential=True
    )
    s = jax.lax.rsqrt(r2)
    np.testing.assert_allclose(np.asarray(g), np.asarray(s**3), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(psi), np.asarray(s), rtol=1e-6)


@pytest.mark.parametrize("kernel", SOFTENING_KERNELS)
def test_autodiff_matches_the_analytic_derivative(kernel):
    eps = 0.5
    params = softening_params(kernel, eps, jnp.float32)
    r = jnp.linspace(0.05, 3.0, 40, dtype=jnp.float32)

    def g_of_r(x):
        return pair_factors(x * x, params, kernel)[0]

    auto = jax.vmap(jax.grad(g_of_r))(r) / r
    _, _, dg = pair_factors(r * r, params, kernel, derivative=True)
    np.testing.assert_allclose(np.asarray(auto), np.asarray(dg), rtol=2e-4)


@pytest.mark.parametrize("kernel", SOFTENING_KERNELS)
def test_softening_derivative_matches_finite_differences(kernel):
    eps = 0.4
    r2 = np.linspace(0.01, 4.0, 60) ** 2

    def g_at(e):
        return pair_factors(r2, softening_params_np(kernel, e), kernel, xp=np)[0]

    de = 1e-6
    fd = (g_at(eps + de) - g_at(eps - de)) / (2 * de) / (2 * eps)  # d/d(eps^2)
    got = pair_softening_sq_derivative(
        r2, softening_params_np(kernel, eps), kernel, xp=np
    )
    np.testing.assert_allclose(got, fd, rtol=1e-6, atol=1e-9 * np.abs(fd).max())
    if kernel != "plummer":
        past = r2 > (support_factor(kernel) * eps) ** 2
        assert np.all(got[past] == 0.0)


@pytest.mark.parametrize("kernel", SOFTENING_KERNELS)
def test_masked_factors_zero_inactive_and_match_unmasked(kernel):
    eps = 0.3
    r2 = jnp.asarray([0.0, 0.01, 0.2, 3.0], jnp.float32)
    active = jnp.asarray([False, True, True, False])
    params = softening_params(kernel, eps, jnp.float32)
    g, psi, dg = masked_pair_factors(
        r2, active, params, kernel, potential=True, derivative=True
    )
    g0, psi0, dg0 = pair_factors(r2, params, kernel, potential=True, derivative=True)
    for a, b in ((g, g0), (psi, psi0), (dg, dg0)):
        a, b = np.asarray(a), np.asarray(b)
        assert np.all(a[~np.asarray(active)] == 0.0)
        np.testing.assert_allclose(a[np.asarray(active)], b[np.asarray(active)], rtol=1e-6)
