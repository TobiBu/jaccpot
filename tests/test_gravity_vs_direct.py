"""Numerical correctness checks against direct summation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, RuntimePolicyConfig


def _direct_sum_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    G: float,
    softening: float,
) -> np.ndarray:
    """Reference O(N^2) accelerations with self-interaction removed."""
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
        out[i] = -G * np.sum((masses[:, None] * inv_dist3[:, None]) * delta, axis=0)

    return out


@pytest.mark.parametrize(
    ("dtype", "rel_tol", "abs_tol"),
    [
        pytest.param(jnp.float32, 9.0e-2, 3.0e-1, id="float32"),
        pytest.param(jnp.float64, 2.0e-2, 6.0e-2, id="float64"),
    ],
)
def test_fmm_acceleration_matches_direct_sum(dtype, rel_tol: float, abs_tol: float):
    """FMM accelerations should agree with direct sum for small systems."""
    if dtype == jnp.float64 and not jax.config.jax_enable_x64:
        pytest.skip("float64 disabled in this JAX runtime")

    n = 128
    key = jax.random.PRNGKey(7)
    key_pos, key_mass = jax.random.split(key)

    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        dtype=dtype,
        minval=jnp.asarray(-1.0, dtype=dtype),
        maxval=jnp.asarray(1.0, dtype=dtype),
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )

    g_const = 1.0
    softening = 1.0e-2

    fmm = FastMultipoleMethod(
        preset="accurate",
        basis="solidfmm",
        theta=0.5,
        G=g_const,
        softening=softening,
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

    acc_fmm = np.asarray(
        fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=8,
            max_order=4,
        )
    )
    acc_ref = _direct_sum_accelerations(
        np.asarray(positions),
        np.asarray(masses),
        G=g_const,
        softening=softening,
    )

    diff = acc_fmm - acc_ref
    rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(acc_ref) + 1.0e-12)
    max_abs = float(np.max(np.abs(diff)))

    assert rel_l2 < rel_tol
    assert max_abs < abs_tol
