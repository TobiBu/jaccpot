"""Runtime validation for real-basis rotate+scale M2L path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, RuntimePolicyConfig


def _direct_sum_accelerations(
    positions: np.ndarray, masses: np.ndarray, *, softening: float
) -> np.ndarray:
    n = int(positions.shape[0])
    out = np.zeros_like(positions)
    eps = np.finfo(positions.dtype).eps
    soft_sq = softening * softening
    for i in range(n):
        delta = positions[i] - positions
        dist_sq = np.sum(delta * delta, axis=1) + soft_sq
        dist = np.sqrt(dist_sq)
        inv_dist3 = 1.0 / (dist_sq * dist + eps)
        inv_dist3[i] = 0.0
        out[i] = -np.sum((masses[:, None] * inv_dist3[:, None]) * delta, axis=0)
    return out


def _solver_with_capacity(*, basis: str) -> FastMultipoleMethod:
    return FastMultipoleMethod(
        preset="accurate",
        basis=basis,
        theta=0.5,
        softening=1.0e-2,
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(
                traversal_config=DualTreeTraversalConfig(
                    max_pair_queue=131072,
                    process_block=512,
                    max_interactions_per_node=65536,
                    max_neighbors_per_leaf=65536,
                )
            )
        ),
    )


def test_real_basis_matches_direct_sum_small_n():
    """Real-basis FMM should agree with direct summation at small N."""
    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for stable tolerance")

    n = 96
    dtype = jnp.float64
    key = jax.random.PRNGKey(123)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )

    fmm_real = _solver_with_capacity(basis="real")
    acc_real = np.asarray(
        fmm_real.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )
    acc_ref = _direct_sum_accelerations(
        np.asarray(positions), np.asarray(masses), softening=1.0e-2
    )

    rel_l2 = np.linalg.norm(acc_real - acc_ref) / (np.linalg.norm(acc_ref) + 1.0e-12)
    assert rel_l2 < 5.0e-2


def test_real_basis_tracks_complex_basis():
    """Real and complex basis outputs should stay close at equal order."""
    n = 96
    dtype = jnp.float32
    key = jax.random.PRNGKey(321)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )

    fmm_real = _solver_with_capacity(basis="real")
    fmm_complex = _solver_with_capacity(basis="complex")
    acc_real = np.asarray(
        fmm_real.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )
    acc_complex = np.asarray(
        fmm_complex.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )

    rel_l2 = np.linalg.norm(acc_real - acc_complex) / (
        np.linalg.norm(acc_complex) + 1.0e-12
    )
    assert rel_l2 < 7.0e-2
