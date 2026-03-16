"""Compare pure-JAX and optional Pallas real-basis M2L performance."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, RuntimePolicyConfig
from jaccpot.pallas import pallas_m2l_real_supported


def _sample_problem(n: int = 1024, dtype=jnp.float32):
    key = jax.random.PRNGKey(7)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )
    return positions, masses


def _time_solver(*, use_pallas: bool) -> float:
    positions, masses = _sample_problem()
    solver = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=0.5,
        softening=1.0e-2,
        use_pallas=use_pallas,
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
    warmup = solver.compute_accelerations(positions, masses, leaf_size=16, max_order=4)
    jax.block_until_ready(warmup)
    t0 = time.perf_counter()
    acc = solver.compute_accelerations(positions, masses, leaf_size=16, max_order=4)
    jax.block_until_ready(acc)
    return time.perf_counter() - t0


def main():
    pure_time = _time_solver(use_pallas=False)
    pallas_time = _time_solver(use_pallas=True)

    print("backend             :", jax.default_backend())
    print("pallas_supported    :", pallas_m2l_real_supported())
    print("pure_jax_time_s     :", f"{pure_time:.4f}")
    print("use_pallas_time_s   :", f"{pallas_time:.4f}")
    if pallas_m2l_real_supported():
        print("speedup_x           :", f"{pure_time / max(pallas_time, 1.0e-12):.3f}")
    else:
        print("speedup_x           : fallback (unsupported backend)")


if __name__ == "__main__":
    main()
