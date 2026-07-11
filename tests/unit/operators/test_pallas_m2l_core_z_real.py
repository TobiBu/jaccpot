"""Regression checks for the optional Pallas real z-translation kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.m2l_real_rot_scale import m2l_core_z_real
from jaccpot.operators.real_harmonics import (
    sh_size,
    translate_along_z_m2l_real,
)
from jaccpot.pallas.m2l_core_z_real import m2l_core_z_real_pallas


@pytest.mark.filterwarnings(
    "ignore:scatter inputs have incompatible types:FutureWarning"
)
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
    try:
        pallas = np.asarray(
            m2l_core_z_real(multipoles, radii, order=order, use_pallas=True)
        )
    except Exception as exc:  # pragma: no cover - backend/hardware dependent
        msg = str(exc).lower()
        if "warpgroup" in msg or "ptx" in msg or "triton" in msg:
            pytest.skip(f"Pallas kernel unavailable on this GPU/runtime: {exc}")
        raise

    assert np.allclose(pallas, pure, rtol=1.0e-5, atol=1.0e-5)


@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_pallas_core_z_interpret_matches_pure_jax(order):
    """The Pallas kernel LOGIC must equal the pure-JAX z-M2L recurrence.

    Runs the actual Pallas kernel in interpret mode so this executes on CPU CI
    (the ``use_pallas=True`` dispatch silently falls back to pure-JAX off GPU,
    so it cannot catch kernel drift). This is the guard that keeps the two
    independent encodings of the recurrence -- the inline loops in
    translate_along_z_m2l_real and the static tables in
    jaccpot.pallas.m2l_core_z_real -- in sync.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for a tight tolerance")

    coeff_count = sh_size(order)
    key = jax.random.PRNGKey(order + 1)
    key_mult, key_r = jax.random.split(key)
    multipoles = jax.random.normal(key_mult, (8, coeff_count), dtype=jnp.float64)
    radii = jax.random.uniform(
        key_r,
        (8,),
        minval=jnp.asarray(2.0, dtype=jnp.float64),
        maxval=jnp.asarray(5.0, dtype=jnp.float64),
    )

    pure = jax.vmap(lambda m, r: translate_along_z_m2l_real(m, r, order=order))(
        multipoles, radii
    )
    pallas = m2l_core_z_real_pallas(multipoles, radii, order=order, interpret=True)
    assert np.allclose(np.asarray(pallas), np.asarray(pure), rtol=1.0e-12, atol=1.0e-12)
