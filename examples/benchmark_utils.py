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


def apply_runtime_path(
    kwargs: dict[str, Any],
    *,
    runtime_path: str = "auto",
) -> dict[str, Any]:
    """Return solver kwargs with an explicit runtime-path selection."""

    updated = dict(kwargs)
    updated["runtime_path"] = str(runtime_path).strip().lower()
    return updated


def canonical_large_n_production_config(
    *,
    leaf_target: int = 256,
    theta: float = 0.6,
    softening: float = 1e-3,
    working_dtype: str = "float32",
) -> dict[str, Any]:
    """Return canonical benchmark config for large-N radix fast memory path."""

    return {
        "preset": "large_n_gpu",
        "basis": "solidfmm",
        "theta": float(theta),
        "softening": float(softening),
        "working_dtype": str(working_dtype),
        "tree_type": "radix",
        "leaf_target": int(leaf_target),
        "farfield_rotation": "solidfmm",
        "farfield_mode": "pair_grouped",
        "grouped_interactions": False,
        "streamed_far_pairs": True,
        "mixed_order": False,
        "mixed_order_min_order": None,
        "nearfield_mode": "bucketed",
        "nearfield_edge_chunk_size": 256,
        "precompute_scatter_schedules": False,
        "memory_objective": "minimum_memory",
        "fail_fast": True,
        "jit_traversal": True,
        "enable_interaction_cache": False,
        "retain_traversal_result": False,
        "retain_interactions": False,
        "autotune_m2l_chunk": False,
        "mac_type": "dehnen",
    }


def resolved_large_n_memory_path_report(fmm: Any) -> dict[str, Any]:
    """Return the resolved runtime flags relevant to the lean large-N path."""

    impl = getattr(fmm, "_impl", None)
    if impl is None:
        return {
            "resolved_runtime_path": None,
            "resolved_memory_objective": None,
            "resolved_streamed_far_pairs": None,
            "resolved_grouped_interactions": None,
            "resolved_retain_traversal_result": None,
            "resolved_retain_interactions": None,
            "resolved_large_n_memory_path_active": None,
        }

    memory_objective = str(getattr(impl, "memory_objective", "")).strip().lower()
    runtime_path = (
        str(getattr(fmm, "runtime_path", getattr(impl, "runtime_path", "auto")))
        .strip()
        .lower()
    )
    streamed_far_pairs = bool(getattr(impl, "streamed_far_pairs", False))
    grouped_interactions = getattr(impl, "grouped_interactions", None)
    grouped_interactions_bool = (
        None if grouped_interactions is None else bool(grouped_interactions)
    )
    retain_traversal_result = bool(getattr(impl, "retain_traversal_result", True))
    retain_interactions = bool(getattr(impl, "retain_interactions", True))
    active = (
        memory_objective == "minimum_memory"
        and streamed_far_pairs
        and not retain_traversal_result
        and not retain_interactions
        and not bool(grouped_interactions_bool)
    )
    return {
        "resolved_runtime_path": runtime_path,
        "resolved_memory_objective": memory_objective,
        "resolved_streamed_far_pairs": streamed_far_pairs,
        "resolved_grouped_interactions": grouped_interactions_bool,
        "resolved_retain_traversal_result": retain_traversal_result,
        "resolved_retain_interactions": retain_interactions,
        "resolved_large_n_memory_path_active": bool(active),
    }


__all__ = [
    "TimingResult",
    "apply_runtime_path",
    "canonical_large_n_production_config",
    "generate_random_distribution",
    "resolved_large_n_memory_path_report",
    "time_callable",
]
