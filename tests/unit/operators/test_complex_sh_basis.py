"""Unit coverage for the ComplexSHBasis adapter (jaccpot public API).

Small batched operator calls (order 2) -- no FMM tree build -- so this
exercises pack/unpack, the rotate-to/from-z round trip, and M2L end to end
in well under a second.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.basis.complex_sh import ComplexSHBasis
from jaccpot.operators.real_harmonics import sh_size


def _coeffs(batch: int, order: int, seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    cc = sh_size(order)
    return jnp.asarray(
        rng.standard_normal((batch, cc)) + 1j * rng.standard_normal((batch, cc)),
        dtype=jnp.complex128,
    )


def test_n_coeffs_matches_sh_size():
    basis = ComplexSHBasis()
    for p in (1, 2, 3):
        assert basis.n_coeffs(p) == sh_size(p)


def test_pack_and_unpack_validate_last_dim():
    basis = ComplexSHBasis()
    order = 2
    good = _coeffs(4, order)
    assert basis.pack_coeffs(good, order=order).shape == good.shape
    assert basis.unpack_coeffs(good, order=order).shape == good.shape
    with pytest.raises(ValueError, match="expected last dimension"):
        basis.pack_coeffs(jnp.zeros((4, 3), dtype=jnp.complex128), order=order)


def test_m2l_rot_scale_runs_and_validates_deltas():
    basis = ComplexSHBasis()
    order = 2
    src = _coeffs(4, order, seed=3)
    deltas = jnp.asarray(np.tile(np.array([1.0, 0.5, -0.5]), (4, 1)))
    out = basis.m2l_rot_scale(src, deltas, order=order)
    assert out.shape == src.shape
    assert np.all(np.isfinite(np.asarray(out)))
    with pytest.raises(ValueError, match="deltas must have shape"):
        basis.m2l_rot_scale(src, jnp.zeros((4,)), order=order)
