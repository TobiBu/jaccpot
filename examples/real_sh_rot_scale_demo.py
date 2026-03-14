"""Demo: real SH rotate+scale FMM vs direct-sum and complex basis."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, RuntimePolicyConfig


def _direct_sum(
    positions: np.ndarray, masses: np.ndarray, softening: float
) -> np.ndarray:
    n = positions.shape[0]
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


def main() -> None:
    n = 128
    order = 4
    softening = 1.0e-2
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    key = jax.random.PRNGKey(0)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )

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
    fmm_real = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=0.5,
        softening=softening,
        advanced=advanced,
    )
    fmm_complex = FastMultipoleMethod(
        preset="accurate",
        basis="complex",
        theta=0.5,
        softening=softening,
        advanced=advanced,
    )

    acc_real = np.asarray(
        fmm_real.compute_accelerations(positions, masses, leaf_size=8, max_order=order)
    )
    acc_complex = np.asarray(
        fmm_complex.compute_accelerations(
            positions, masses, leaf_size=8, max_order=order
        )
    )
    acc_ref = _direct_sum(np.asarray(positions), np.asarray(masses), softening)

    err_real = np.linalg.norm(acc_real - acc_ref) / (np.linalg.norm(acc_ref) + 1.0e-12)
    err_complex = np.linalg.norm(acc_complex - acc_ref) / (
        np.linalg.norm(acc_ref) + 1.0e-12
    )
    real_vs_complex = np.linalg.norm(acc_real - acc_complex) / (
        np.linalg.norm(acc_complex) + 1.0e-12
    )

    print(f"dtype={dtype} n={n} order={order}")
    print(f"real_vs_direct_rel_l2={err_real:.6e}")
    print(f"complex_vs_direct_rel_l2={err_complex:.6e}")
    print(f"real_vs_complex_rel_l2={real_vs_complex:.6e}")


if __name__ == "__main__":
    main()
