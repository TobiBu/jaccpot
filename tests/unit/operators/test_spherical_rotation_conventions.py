import pytest

pytest.skip(
    "Complex-basis rotation convention tests removed (real-only pipeline).",
    allow_module_level=True,
)

import jax.numpy as jnp

from jaccpot.operators.spherical_harmonics import (
    ZAxisRotation,
    _complex_to_real_tesseral_transform,
    _real_tesseral_rotation_block,
    _real_tesseral_to_complex_transform,
    p2m_point_real_sh,
    rotate_real_sh,
    sh_offset,
    sh_size,
)


def _slice_degree(p: int, ell: int) -> slice:
    start = sh_offset(ell)
    end = sh_offset(ell + 1)
    assert end - start == 2 * ell + 1
    return slice(start, end)


def test_real_rotation_roundtrip_per_degree():
    """Rotation should be invertible per degree (roundtrip check)."""

    p = 3

    # A point mass expansion about origin.
    delta = jnp.array([0.23, -0.31, 0.41], dtype=jnp.float64)
    m = jnp.array(1.7, dtype=jnp.float64)
    coeffs = p2m_point_real_sh(delta, m, order=p)
    assert coeffs.shape == (sh_size(p),)

    # Some arbitrary rotation.
    alpha = jnp.asarray(0.3, dtype=jnp.float64)
    beta = jnp.asarray(0.7, dtype=jnp.float64)
    gamma = jnp.asarray(-0.2, dtype=jnp.float64)

    rotated_blocks = []
    for ell in range(p + 1):
        blk = coeffs[_slice_degree(p, ell)]
        R = _real_tesseral_rotation_block(
            ell,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            real_dtype=jnp.dtype(jnp.float64),
        )
        rotated_blocks.append(R @ blk)

    rotated = jnp.concatenate(rotated_blocks)
    assert rotated.shape == coeffs.shape

    # Also check the full packed helper agrees via rotate_real_sh.
    rotated2 = rotate_real_sh(coeffs, alpha, beta, gamma, order=p)
    assert jnp.allclose(rotated2, rotated, rtol=1e-12, atol=1e-12)

    # Roundtrip using inverse rotation (ZYZ: inverse is -gamma,-beta,-alpha).
    rot_inv = ZAxisRotation(alpha=-gamma, beta=-beta, gamma=-alpha)
    roundtrip = rotate_real_sh(
        rotated2, rot_inv.alpha, rot_inv.beta, rot_inv.gamma, order=p
    )
    assert jnp.allclose(roundtrip, coeffs, rtol=1e-12, atol=1e-12)


def test_complex_to_real_transform_inverse_per_degree():
    """T and U should be exact inverses per degree."""

    for ell in range(0, 6):
        cdtype = jnp.dtype(jnp.complex128)
        T = _complex_to_real_tesseral_transform(ell, dtype=cdtype)
        U = _real_tesseral_to_complex_transform(ell, dtype=cdtype)
        ident = T @ U
        assert jnp.allclose(
            ident,
            jnp.eye(2 * ell + 1, dtype=cdtype),
            rtol=1e-12,
            atol=1e-12,
        )
