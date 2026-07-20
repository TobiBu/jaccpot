"""Unit coverage for the real rotate+scale M2L helpers (pure-JAX path).

Small batched operator calls (order 3, a handful of pairs) -- no FMM tree.
The headline test is an equivalence check: the cached-rotation-block M2L must
reproduce the direct per-pair rotate+scale M2L, which also exercises the block
builders and pack/unpack helpers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.m2l_real_rot_scale import (
    _real_rotation_blocks_padded,
    m2l_rot_scale_real_batch,
    m2l_rot_scale_real_batch_cached_blocks,
    real_rotation_blocks_from_z_local_batch,
    real_rotation_blocks_to_z_multipole_batch,
)
from jaccpot.operators.real_harmonics import sh_size


def _inputs(order: int, batch: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    mult = jnp.asarray(rng.standard_normal((batch, sh_size(order))), dtype=jnp.float64)
    # Offset the deltas so radii are comfortably > 0 (well-conditioned M2L).
    deltas = jnp.asarray(
        rng.standard_normal((batch, 3)) + np.array([3.0, 2.0, 4.0]), dtype=jnp.float64
    )
    return mult, deltas


def test_cached_blocks_m2l_matches_direct_batch():
    order, batch = 3, 5
    mult, deltas = _inputs(order, batch, seed=1)

    direct = np.asarray(m2l_rot_scale_real_batch(mult, deltas, order=order))

    blocks_to = real_rotation_blocks_to_z_multipole_batch(
        deltas, order=order, dtype=mult.dtype
    )
    blocks_from = real_rotation_blocks_from_z_local_batch(
        deltas, order=order, dtype=mult.dtype
    )
    cached = np.asarray(
        m2l_rot_scale_real_batch_cached_blocks(
            mult, deltas, blocks_to, blocks_from, order=order
        )
    )

    assert cached.shape == direct.shape == (batch, sh_size(order))
    assert np.allclose(cached, direct, atol=1e-9)


def test_m2l_rot_scale_real_batch_validates_shapes():
    order = 2
    mult, deltas = _inputs(order, 3)
    with pytest.raises(ValueError, match="multipoles must have shape"):
        m2l_rot_scale_real_batch(mult[0], deltas, order=order)  # 1-D multipoles
    with pytest.raises(ValueError, match="deltas must have shape"):
        m2l_rot_scale_real_batch(mult, jnp.zeros((3,)), order=order)


def test_real_rotation_blocks_padded_rejects_bad_which():
    with pytest.raises(ValueError, match="which must be"):
        _real_rotation_blocks_padded(
            jnp.zeros((2, 3)), order=2, dtype=jnp.float64, which="bogus"
        )
