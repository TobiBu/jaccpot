import jax.numpy as jnp
import pytest

from jaccpot.operators.spherical_harmonics import p2m_point_real_sh, translate_along_z_m2m


@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("z", [0.0, 0.1, 0.7, -0.5])
def test_p2m_z_m2m_regression(order, z):
    """Regression: for points on z-axis, direct P2M at shifted center must
    equal translate_along_z_m2m of P2M at original center.

    This test checks multiple orders and z-values to catch normalization
    regressions.
    """

    mass = jnp.asarray(1.3, dtype=jnp.float64)
    pos = jnp.array([0.0, 0.0, z], dtype=jnp.float64)
    s = jnp.asarray(0.2, dtype=jnp.float64)

    M_src = p2m_point_real_sh(pos, mass, order=order)
    M_direct = p2m_point_real_sh(pos + jnp.array([0.0, 0.0, s]), mass, order=order)
    M_via = translate_along_z_m2m(M_src, s, order=order)

    assert jnp.allclose(M_via, M_direct, atol=1e-12, rtol=1e-12)
