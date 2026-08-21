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
from jaccpot.pallas.m2l_core_z_real import (
    m2l_core_z_real_pallas,
    pallas_m2l_real_supported,
)


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

    # Calls the kernel DIRECTLY. This used to go through
    # `m2l_core_z_real(use_pallas=True)`, and that dispatch is gone -- `operators/`
    # no longer imports `pallas/` (audit G.3). The change also makes the skip
    # honest: the old route fell back to pure-JAX off GPU, so on CPU this test
    # compared pure against pure and passed while asserting nothing. It is now
    # gated on `pallas_m2l_real_supported()` and skips instead.
    #
    # `m2l_core_z_real` floors radii at 1e-30 before translating, so the kernel is
    # given the same treatment. Immaterial for the [0.25, 1.25] draws here, and
    # correct if that range ever changes.
    if not pallas_m2l_real_supported():
        pytest.skip("Pallas real z-M2L unsupported on this backend")

    pure = np.asarray(m2l_core_z_real(multipoles, radii, order=order))
    floored = jnp.maximum(radii, jnp.asarray(1.0e-30, dtype=radii.dtype))
    try:
        pallas = np.asarray(m2l_core_z_real_pallas(multipoles, floored, order=order))
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
    (the sibling test above needs real hardware and skips without it, so it
    cannot catch kernel drift on CI). This is the guard that keeps the two
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
