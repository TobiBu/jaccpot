"""Grouped/cached real-basis M2M and L2L == the ungrouped rotate-scale operators.

The grouped/cached kernels precompute the per-degree real rotation blocks once per
interaction class (shared displacement) and apply them with a cached einsum, running only
the z-axis translation per node. They must be bit-identical (up to fp) to the reference
``m2m_real`` / ``l2l_real``, which are the force-validated Dehnen rotate-scale operators
(off-axis field correct upstream). This is the Phase-1 gate for the M2M/L2L launch-count
refactor: correctness is guaranteed at the operator level before any cascade wiring.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaccpot.operators.m2l_real_rot_scale import (
    l2l_rot_scale_real_batch_cached_blocks,
    m2m_rot_scale_real_batch_cached_blocks,
    real_rotation_blocks_from_z_local_batch,
    real_rotation_blocks_from_z_multipole_batch,
    real_rotation_blocks_to_z_local_batch,
    real_rotation_blocks_to_z_multipole_batch,
)
from jaccpot.operators.real_harmonics import l2l_real, m2m_real, sh_size

_TOL = 1e-10


def _rand(order, n, seed):
    rng = np.random.default_rng(seed)
    coeffs = jnp.asarray(rng.standard_normal((n, sh_size(order))))
    deltas = jnp.asarray(rng.standard_normal((n, 3)))  # generic off-axis displacements
    return coeffs, deltas


@pytest.mark.parametrize("order", [2, 4, 6])
def test_m2m_cached_matches_m2m_real(order):
    coeffs, deltas = _rand(order, 24, 11)
    ref = jax.vmap(lambda c, d: m2m_real(c, d, order=order))(coeffs, deltas)
    bt = real_rotation_blocks_to_z_multipole_batch(
        deltas, order=order, dtype=coeffs.dtype
    )
    bf = real_rotation_blocks_from_z_multipole_batch(
        deltas, order=order, dtype=coeffs.dtype
    )
    got = m2m_rot_scale_real_batch_cached_blocks(coeffs, deltas, bt, bf, order=order)
    assert float(np.max(np.abs(np.asarray(got - ref)))) < _TOL


@pytest.mark.parametrize("order", [2, 4, 6])
def test_l2l_cached_matches_l2l_real(order):
    coeffs, deltas = _rand(order, 24, 22)
    ref = jax.vmap(lambda c, d: l2l_real(c, d, order=order))(coeffs, deltas)
    bt = real_rotation_blocks_to_z_local_batch(deltas, order=order, dtype=coeffs.dtype)
    bf = real_rotation_blocks_from_z_local_batch(
        deltas, order=order, dtype=coeffs.dtype
    )
    got = l2l_rot_scale_real_batch_cached_blocks(coeffs, deltas, bt, bf, order=order)
    assert float(np.max(np.abs(np.asarray(got - ref)))) < _TOL


@pytest.mark.parametrize("op", ["m2m", "l2l"])
def test_class_grouping_matches_ungrouped(op):
    """Mirror the octree wiring: jnp.unique -> per-class blocks -> blocks[cls_id]."""
    order, n_classes, n = 4, 8, 50
    rng = np.random.default_rng(33)
    class_deltas = rng.standard_normal((n_classes, 3))
    idx = rng.integers(0, n_classes, size=n)
    deltas = jnp.asarray(class_deltas[idx])
    coeffs = jnp.asarray(rng.standard_normal((n, sh_size(order))))

    if op == "m2m":
        ref = jax.vmap(lambda c, d: m2m_real(c, d, order=order))(coeffs, deltas)
        to_fn, from_fn = (
            real_rotation_blocks_to_z_multipole_batch,
            real_rotation_blocks_from_z_multipole_batch,
        )
        cached = m2m_rot_scale_real_batch_cached_blocks
    else:
        ref = jax.vmap(lambda c, d: l2l_real(c, d, order=order))(coeffs, deltas)
        to_fn, from_fn = (
            real_rotation_blocks_to_z_local_batch,
            real_rotation_blocks_from_z_local_batch,
        )
        cached = l2l_rot_scale_real_batch_cached_blocks

    uniq, cls_id = jnp.unique(
        deltas, axis=0, size=n_classes, return_inverse=True, fill_value=0.0
    )
    cls_id = cls_id.reshape(-1)
    bt = to_fn(uniq, order=order, dtype=coeffs.dtype)
    bf = from_fn(uniq, order=order, dtype=coeffs.dtype)
    got = cached(coeffs, deltas, bt[cls_id], bf[cls_id], order=order)
    assert float(np.max(np.abs(np.asarray(got - ref)))) < _TOL
