import pytest

pytest.skip(
    "Complex-basis diagnostics removed (real-only pipeline).",
    allow_module_level=True,
)

import jax
import jax.numpy as jnp

from jaccpot.operators.spherical_harmonics import (
    _complex_to_real_tesseral_transform,
    _real_tesseral_to_complex_transform,
    sh_offset,
    sh_size,
    translate_along_z_m2m,
)


def _slice_degree(ell: int) -> slice:
    return slice(sh_offset(ell), sh_offset(ell + 1))


def test_z_m2m_translation_couples_only_same_m_in_complex_basis():
    """Translation along z must not mix different m in complex SH basis.

    This is a convention-level diagnostic: if this fails, our real<->complex
    conversions around translate_along_z_m2m are inconsistent.

    Strategy:
    - Create a single complex coefficient M_{ell0,m0} with m0 != 0.
    - Convert it to real tesseral using T.
    - Run translate_along_z_m2m (which does real->complex, translate, back).
    - Convert the output back to complex and check that for each output degree
      ell', all m != m0 are ~0.
    """

    p = 5
    ell0 = 4
    m0 = 1

    cdtype = jnp.dtype(jnp.complex128)
    rdtype = jnp.dtype(jnp.float64)

    # Build complex blocks with a single nonzero coefficient.
    M_complex_blocks = []
    for ell in range(p + 1):
        blk = jnp.zeros((2 * ell + 1,), dtype=cdtype)
        if ell == ell0:
            blk = blk.at[m0 + ell].set(1.0 + 0.0j)
        M_complex_blocks.append(blk)

    # Convert to packed real tesseral.
    real_blocks = []
    for ell in range(p + 1):
        T = _complex_to_real_tesseral_transform(ell, dtype=cdtype)
        real_blocks.append(jnp.real(T @ M_complex_blocks[ell]).astype(rdtype))
    M_real = jnp.concatenate(real_blocks)
    assert M_real.shape == (sh_size(p),)

    # Translate along z.
    r = jnp.asarray(0.37, dtype=rdtype)
    out_real = translate_along_z_m2m(M_real, r, order=p)
    assert out_real.shape == M_real.shape

    # Convert back to complex and check m-coupling.
    for ell in range(p + 1):
        U = _real_tesseral_to_complex_transform(ell, dtype=cdtype)
        out_blk = (U @ out_real[_slice_degree(ell)].astype(cdtype)).astype(cdtype)

        for m in range(-ell, ell + 1):
            # For the real tesseral basis, a pure complex +m input generally
            # corresponds to a real combination of +m and -m. So we expect the
            # translated output to preserve the *pair* {+m0,-m0} and not
            # introduce other |m|.
            if abs(m) == abs(m0) and ell >= abs(m0):
                continue
            assert jnp.allclose(out_blk[m + ell], 0.0 + 0.0j, atol=1e-12)


def test_z_m2m_translation_matches_dehnen_series_for_each_m_channel():
    """Check z-M2M implements the per-m Dehnen series in complex basis.

    For a fixed m, Dehnen's z-shift reduces to:
      M'_{ell,m} = sum_{k=0..ell-|m|} (r^k/(k!)^2) M_{ell-k,m}

    We test that relationship directly in complex space by constructing a
    random-ish (but deterministic) complex multipole vector supported only on
    a single m0, passing through translate_along_z_m2m, and comparing.
    """

    p = 6
    m0 = 1
    r = jnp.asarray(0.21, dtype=jnp.float64)

    cdtype = jnp.dtype(jnp.complex128)
    rdtype = jnp.dtype(jnp.float64)

    # Deterministic coefficients per degree for just one m-channel.
    M_complex_blocks = []
    for ell in range(p + 1):
        blk = jnp.zeros((2 * ell + 1,), dtype=cdtype)
        if ell >= abs(m0):
            val = (0.3 + 0.1 * ell) + 1j * (0.07 * ell)
            blk = blk.at[m0 + ell].set(val)
        M_complex_blocks.append(blk)

    # Pack into real tesseral.
    real_blocks = []
    for ell in range(p + 1):
        T = _complex_to_real_tesseral_transform(ell, dtype=cdtype)
        real_blocks.append(jnp.real(T @ M_complex_blocks[ell]).astype(rdtype))
    M_real = jnp.concatenate(real_blocks)

    out_real = translate_along_z_m2m(M_real, r, order=p)

    # Unpack input/output to complex.
    in_complex = []
    out_complex = []
    for ell in range(p + 1):
        U = _real_tesseral_to_complex_transform(ell, dtype=cdtype)
        in_complex.append(
            (U @ M_real[_slice_degree(ell)].astype(cdtype)).astype(cdtype)
        )
        out_complex.append(
            (U @ out_real[_slice_degree(ell)].astype(cdtype)).astype(cdtype)
        )

    # Compare against Dehnen series per degree.
    # Precompute factorials.
    n = jnp.arange(0, 2 * p + 1)
    fact = jnp.exp(jax.lax.lgamma(n + 1.0)).astype(rdtype)

    for ell in range(abs(m0), p + 1):
        expected = 0.0 + 0.0j
        for k in range(0, ell - abs(m0) + 1):
            src_ell = ell - k
            src = in_complex[src_ell][m0 + src_ell]
            # Using the Dehnen-style Υ normalization adopted in the code,
            # the z-specialised polynomial factor is r^k / k! (not (k!)^2).
            expected = expected + (r**k) * src / fact[k]

        got = out_complex[ell][m0 + ell]
        assert jnp.allclose(got, expected, rtol=1e-12, atol=1e-12)
