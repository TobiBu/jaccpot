"""Regression checks for the optional Pallas real z-translation kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.m2l_real_rot_scale import m2l_core_z_real


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
