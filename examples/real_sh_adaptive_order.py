"""Run the real-basis solver with adaptive-order far-field evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, RuntimePolicyConfig


def _sample_problem(n: int = 256, dtype=jnp.float64):
    key = jax.random.PRNGKey(1234)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )
    return positions, masses


def _direct_sum_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    softening: float,
) -> np.ndarray:
    out = np.zeros_like(positions)
    soft_sq = softening * softening
    eps = np.finfo(positions.dtype).eps
    for i in range(positions.shape[0]):
        delta = positions[i] - positions
        dist_sq = np.sum(delta * delta, axis=1) + soft_sq
        dist = np.sqrt(dist_sq)
        inv_dist3 = 1.0 / (dist_sq * dist + eps)
        inv_dist3[i] = 0.0
        out[i] = -np.sum((masses[:, None] * inv_dist3[:, None]) * delta, axis=0)
    return out


def main():
    positions, masses = _sample_problem()
    softening = 1.0e-2
    advanced = FMMAdvancedConfig(
        runtime=RuntimePolicyConfig(
            traversal_config=DualTreeTraversalConfig(
                max_pair_queue=131072,
                process_block=512,
                max_interactions_per_node=65536,
                max_neighbors_per_leaf=65536,
            )
        )
    )

    solver = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        adaptive_order=True,
        p_gears=(2, 3, 4),
        theta=0.5,
        softening=softening,
        advanced=advanced,
    )
    accelerations = np.asarray(
        solver.compute_accelerations(positions, masses, leaf_size=16, max_order=4)
    )
    reference = _direct_sum_accelerations(
        np.asarray(positions),
        np.asarray(masses),
        softening=softening,
    )

    rel_l2 = np.linalg.norm(accelerations - reference) / (
        np.linalg.norm(reference) + 1.0e-12
    )
    print(f"backend={jax.default_backend()}")
    print("basis=real adaptive_order=True p_gears=(2, 3, 4)")
    print(f"rel_l2_vs_direct={rel_l2:.6e}")
    print(f"far_pairs_by_gear_counts={solver._impl._recent_far_pairs_by_gear_counts}")


if __name__ == "__main__":
    main()
