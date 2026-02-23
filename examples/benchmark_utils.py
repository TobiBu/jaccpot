"""Utility helpers for benchmarking Expanse FMM routines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


def _block_until_ready(value: Any) -> Any:
    """Recursively block on JAX arrays to ensure accurate timing."""

    def _maybe_block(x: Any) -> Any:
        if hasattr(x, "block_until_ready"):
            return x.block_until_ready()
        return x

    return jax.tree_util.tree_map(_maybe_block, value)


@dataclass(frozen=True)
class TimingResult:
    """Container summarising repeated runtime measurements."""

    wall_times: Tuple[float, ...]
    mean: float
    std: float
    result: Any

    @property
    def samples(self) -> Tuple[float, ...]:
        return self.wall_times


def time_callable(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 1,
    runs: int = 5,
    **kwargs: Any,
) -> TimingResult:
    """Measure execution time for ``fn`` with optional warmup passes."""

    if runs <= 0:
        raise ValueError("runs must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    for _ in range(warmup):
        _block_until_ready(fn(*args, **kwargs))

    samples = []
    result: Any = None
    for _ in range(runs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        result = _block_until_ready(result)
        end = time.perf_counter()
        samples.append(end - start)

    wall_times = tuple(samples)
    mean = float(jnp.mean(jnp.asarray(wall_times, dtype=jnp.float64)))
    std = float(jnp.std(jnp.asarray(wall_times, dtype=jnp.float64)))

    return TimingResult(
        wall_times=wall_times,
        mean=mean,
        std=std,
        result=result,
    )


def generate_random_distribution(
    num_particles: int,
    *,
    key: Optional[jax.Array] = None,
    box_half_extent: float = 1.0,
    mass_scale: float = 1.0,
    dtype: jnp.dtype = jnp.float64,
) -> Tuple[Array, Array, jax.Array]:
    """Create synthetic point masses inside a cube centred at the origin."""

    if num_particles <= 0:
        raise ValueError("num_particles must be positive")

    if key is None:
        key = jax.random.PRNGKey(0)

    dtype = jnp.asarray(0, dtype=dtype).dtype
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (num_particles, 3),
        dtype=dtype,
        minval=-box_half_extent,
        maxval=box_half_extent,
    )

    # Ensure strictly positive masses.
    raw_masses = jax.random.normal(key_mass, (num_particles,), dtype=dtype)
    masses = jnp.abs(raw_masses) + jnp.asarray(mass_scale, dtype=dtype)

    return positions, masses, key


__all__ = [
    "TimingResult",
    "generate_random_distribution",
    "time_callable",
]
