"""Regression checks for the optional Pallas real z-translation kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.operators.m2l_real_rot_scale import m2l_core_z_real


def test_pallas_core_z_matches_pure_jax():
    order = 4
    coeff_count = (order + 1) ** 2
    key = jax.random.PRNGKey(99)
    key_mult, key_r = jax.random.split(key)
    multipoles = jax.random.normal(key_mult, (6, coeff_count), dtype=jnp.float32)
    radii = jax.random.uniform(
        key_r,
        (6,),
        minval=jnp.asarray(0.25, dtype=jnp.float32),
        maxval=jnp.asarray(1.25, dtype=jnp.float32),
    )

    pure = np.asarray(m2l_core_z_real(multipoles, radii, order=order, use_pallas=False))
    pallas = np.asarray(
        m2l_core_z_real(multipoles, radii, order=order, use_pallas=True)
    )

    assert np.allclose(pallas, pure, rtol=1.0e-5, atol=1.0e-5)
