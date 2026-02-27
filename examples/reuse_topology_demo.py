"""Demonstrate topology/pair reuse across small particle motions."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, RuntimePolicyConfig


def _sample_problem(n: int = 512, dtype=jnp.float64):
    key = jax.random.PRNGKey(2026)
    key_pos, key_mass, key_vel = jax.random.split(key, 3)
    positions = jax.random.uniform(
        key_pos, (n, 3), minval=-0.5, maxval=0.5, dtype=dtype
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )
    velocity = 1.0e-12 * jax.random.normal(key_vel, (n, 3), dtype=dtype)
    bounds = (
        jnp.asarray([-1.0, -1.0, -1.0], dtype=dtype),
        jnp.asarray([1.0, 1.0, 1.0], dtype=dtype),
    )
    return positions, masses, velocity, bounds


def _run_sequence(*, reuse_topology: bool, steps: int = 6):
    positions, masses, velocity, bounds = _sample_problem()
    fmm = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        adaptive_order=True,
        p_gears=(2, 3, 4),
        theta=0.5,
        softening=1.0e-2,
        reuse_topology=reuse_topology,
        rebuild_every=3,
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

    timings = []
    reused_flags = []
    for _step in range(steps):
        t0 = time.perf_counter()
        _ = fmm.compute_accelerations(
            positions,
            masses,
            bounds=bounds,
            leaf_size=16,
            max_order=4,
        )
        timings.append(time.perf_counter() - t0)
        reused_flags.append(bool(fmm.recent_topology_reused))
        positions = positions + velocity

    return np.asarray(timings), reused_flags


def main():
    timings_baseline, _ = _run_sequence(reuse_topology=False)
    timings_reuse, reused_flags = _run_sequence(reuse_topology=True)

    print("baseline_step_times_s:", np.array2string(timings_baseline, precision=4))
    print("reuse_step_times_s   :", np.array2string(timings_reuse, precision=4))
    print("topology_reused      :", reused_flags)
    print(
        "mean_speedup_x       :",
        f"{timings_baseline.mean() / max(timings_reuse.mean(), 1.0e-12):.3f}",
    )


if __name__ == "__main__":
    main()
