"""Roundtrip checks for complex<->real spherical-harmonic conversions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.basis.real_sh import complex_to_real_coeffs, real_to_complex_coeffs


def _idx_nm(n: int, m: int) -> int:
    return n * n + (m + n)


def _random_conjugate_symmetric_complex_coeffs(
    order: int, *, key: jax.Array
) -> jax.Array:
    """Generate packed complex SH coefficients satisfying C_{n,-m}=(-1)^m conj(C_{n,m})."""
    n_coeffs = (order + 1) * (order + 1)
    coeffs = np.zeros((n_coeffs,), dtype=np.complex128)

    key_r, key_i = jax.random.split(key)
    draws_r = np.asarray(jax.random.normal(key_r, (n_coeffs,), dtype=jnp.float64))
    draws_i = np.asarray(jax.random.normal(key_i, (n_coeffs,), dtype=jnp.float64))
    cursor = 0

    for n in range(order + 1):
        idx0 = _idx_nm(n, 0)
        coeffs[idx0] = draws_r[cursor]
        cursor += 1
        for m in range(1, n + 1):
            idx_p = _idx_nm(n, m)
            idx_n = _idx_nm(n, -m)
            c_p = draws_r[cursor] + 1j * draws_i[cursor]
            cursor += 1
            sign = -1.0 if (m % 2) else 1.0
            coeffs[idx_p] = c_p
            coeffs[idx_n] = sign * np.conj(c_p)

    return jnp.asarray(coeffs)


def test_real_sh_complex_roundtrip_preserves_coefficients():
    """complex->real->complex should recover conjugate-symmetric coefficients."""
    order = 6
    coeffs_complex = _random_conjugate_symmetric_complex_coeffs(
        order,
        key=jax.random.PRNGKey(42),
    )
    coeffs_real = complex_to_real_coeffs(coeffs_complex, order=order)
    coeffs_complex_rt = real_to_complex_coeffs(coeffs_real, order=order)

    err = np.max(np.abs(np.asarray(coeffs_complex_rt - coeffs_complex)))
    assert err < 1.0e-10


def test_the_two_real_conversions_are_not_interchangeable():
    """``real_sh`` and the Dehnen converter target DIFFERENT real bases.

    ``basis.real_sh.complex_to_real_coeffs`` produces the unitary sqrt(2)
    tesseral basis; ``operators.real_dehnen_q.complex_to_dehnen_real_coeffs``
    produces Dehnen's no-sqrt2 real solid harmonics, which is what the real
    M2L/M2M/L2L/L2P operators consume. Substituting one for the other is
    silently wrong rather than an error, so this pins that they are genuinely
    distinct -- a guard against someone reading two same-shaped converters in
    neighbouring modules as duplicates and collapsing them.

    The ``m == 0`` entries agree (no sqrt2 enters there), which is exactly what
    makes a spot check on a monopole miss the difference; the disagreement lives
    in ``m != 0``.
    """
    from jaccpot.operators.real_harmonics import complex_to_dehnen_real_coeffs

    order = 4
    coeffs = _random_conjugate_symmetric_complex_coeffs(
        order, key=jax.random.PRNGKey(0)
    )

    tesseral = np.asarray(complex_to_real_coeffs(coeffs, order=order))
    dehnen = np.asarray(complex_to_dehnen_real_coeffs(coeffs, order=order))

    assert tesseral.shape == dehnen.shape

    m_nonzero = [
        _idx_nm(n, m) for n in range(order + 1) for m in range(-n, n + 1) if m != 0
    ]
    # Not merely "not allclose": the two must differ on essentially every
    # non-zero-m entry, so a partial refactor cannot pass this by accident.
    differing = np.abs(tesseral[m_nonzero] - dehnen[m_nonzero]) > 1e-12
    assert differing.mean() > 0.9, (differing.mean(), len(m_nonzero))
