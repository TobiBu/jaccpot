import pytest

pytest.skip(
    "Complex/real transform tests removed (real-only pipeline).",
    allow_module_level=True,
)

import jax
import jax.numpy as jnp

from jaccpot.operators.spherical_harmonics import (
    _complex_to_real_tesseral_transform,
    _real_tesseral_to_complex_transform,
)


def test_real_complex_transform_roundtrip_per_degree():
    """T and U must be exact inverses (up to fp error) for each degree.

    This is the minimal algebraic property we need so that rotate→translate
    compositions are well-defined for coefficient vectors.
    """

    for ell in range(0, 6):
        n = 2 * ell + 1
        cdtype = jnp.dtype(jnp.complex128)
        T = _complex_to_real_tesseral_transform(ell, dtype=cdtype)
        U = _real_tesseral_to_complex_transform(ell, dtype=cdtype)

        key = jax.random.key(ell + 1)
        c_re = jax.random.normal(key, (n,), dtype=jnp.float64)
        c_im = jax.random.normal(key, (n,), dtype=jnp.float64)
        c = c_re + 1j * c_im
        c = c.astype(cdtype)

        r = T @ c
        c2 = U @ r

        assert jnp.allclose(c2, c, atol=1e-12, rtol=1e-12)
