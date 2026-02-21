"""Tests for complex solid harmonics ported from solidfmm."""

import jax.numpy as jnp
import numpy as np

from jaccpot.operators.complex_harmonics import complex_R_solidfmm, complex_S_solidfmm
from jaccpot.operators.real_harmonics import sh_index


def test_complex_R_degree_one_values():
    dtype = jnp.float64
    x, y, z = 1.2, -0.7, 2.5
    coeffs = complex_R_solidfmm(jnp.array([x, y, z], dtype=dtype), order=1)

    m0 = coeffs[sh_index(1, 0)]
    m1 = coeffs[sh_index(1, 1)]
    mneg1 = coeffs[sh_index(1, -1)]

    assert np.allclose(m0, z)
    assert np.allclose(m1, (x + 1j * y) * 0.5)
    assert np.allclose(mneg1, (-x + 1j * y) * 0.5)


def test_complex_S_degree_one_values():
    dtype = jnp.float64
    x, y, z = 1.2, -0.7, 2.5
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    coeffs = complex_S_solidfmm(jnp.array([x, y, z], dtype=dtype), order=1)

    s00 = coeffs[sh_index(0, 0)]
    s10 = coeffs[sh_index(1, 0)]
    s11 = coeffs[sh_index(1, 1)]

    assert np.allclose(s00, 1.0 / r)
    assert np.allclose(s10, z / (r2 * r))
    assert np.allclose(s11, (x + 1j * y) / (r2 * r))
