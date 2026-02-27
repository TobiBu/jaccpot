"""Lightweight performance-metric utilities for regression tracking.

These helpers intentionally avoid strict wall-clock thresholds so they can run
across varied CI/dev hardware while still surfacing throughput drift.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, List, Sequence

import jax
import jax.numpy as jnp

from examples import benchmark_utils as bench_utils
from jaccpot import FastMultipoleMethod


def _benchmark_like_distribution(
    num_particles: int,
    *,
    key: jax.Array,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (num_particles, 3),
        dtype=dtype,
        minval=-1.0,
        maxval=1.0,
    )
    masses = jnp.abs(jax.random.normal(key_mass, (num_particles,), dtype=dtype)) + 1.0
    return positions, masses


def collect_prepare_eval_split_metrics(
    particle_counts: Sequence[int],
    *,
    leaf_size: int,
    max_order: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    key: jax.Array,
    fmm_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Collect full/prepare/evaluate timings over particle-count sweep."""
    rows: List[Dict[str, Any]] = []
    fmm = FastMultipoleMethod(**fmm_kwargs)
    current_key = key

    for n in particle_counts:
        positions, masses = _benchmark_like_distribution(
            int(n),
            key=current_key,
            dtype=dtype,
        )
        current_key, _ = jax.random.split(current_key)

        full_timing = bench_utils.time_callable(
            fmm.compute_accelerations,
            positions,
            masses,
            leaf_size=leaf_size,
            max_order=max_order,
            reuse_prepared_state=True,
            warmup=warmup,
            runs=runs,
        )
        prep_timing = bench_utils.time_callable(
            fmm.prepare_state,
            positions,
            masses,
            leaf_size=leaf_size,
            max_order=max_order,
            warmup=0,
            runs=1,
        )
        state = prep_timing.result
        eval_kwargs: Dict[str, Any] = {}
        if "jit_traversal" in inspect.signature(fmm.evaluate_prepared_state).parameters:
            eval_kwargs["jit_traversal"] = True
        eval_timing = bench_utils.time_callable(
            fmm.evaluate_prepared_state,
            state,
            warmup=warmup,
            runs=runs,
            **eval_kwargs,
        )

        rows.append(
            {
                "num_particles": int(n),
                "mean_seconds": float(full_timing.mean),
                "std_seconds": float(full_timing.std),
                "prepare_mean_seconds": float(prep_timing.mean),
                "evaluate_mean_seconds": float(eval_timing.mean),
            }
        )

    return rows


def collect_mode_comparison_metrics(
    particle_counts: Sequence[int],
    *,
    leaf_size: int,
    max_order: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    key: jax.Array,
    base_fmm_kwargs: Dict[str, Any],
    mode_variants: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collect full runtime metrics for a list of mode variants."""
    frames: List[List[Dict[str, Any]]] = []
    for mode_cfg in mode_variants:
        mode_name = str(mode_cfg["name"])
        kwargs = dict(base_fmm_kwargs)
        kwargs.update(mode_cfg.get("fmm_overrides", {}))
        rows = collect_prepare_eval_split_metrics(
            particle_counts,
            leaf_size=leaf_size,
            max_order=max_order,
            runs=runs,
            warmup=warmup,
            dtype=dtype,
            key=key,
            fmm_kwargs=kwargs,
        )
        renamed_rows: List[Dict[str, Any]] = []
        for row in rows:
            renamed_rows.append(
                {
                    "num_particles": row["num_particles"],
                    f"mean_seconds_{mode_name}": row["mean_seconds"],
                    f"std_seconds_{mode_name}": row["std_seconds"],
                    f"prepare_mean_seconds_{mode_name}": row["prepare_mean_seconds"],
                    f"evaluate_mean_seconds_{mode_name}": row["evaluate_mean_seconds"],
                }
            )
        frames.append(renamed_rows)

    merged_by_n: Dict[int, Dict[str, Any]] = {}
    for frame in frames:
        for row in frame:
            n = int(row["num_particles"])
            merged = merged_by_n.setdefault(n, {"num_particles": n})
            merged.update(row)

    return [merged_by_n[k] for k in sorted(merged_by_n.keys())]


def geometric_mean_speedup(
    numerator: Sequence[float], denominator: Sequence[float]
) -> float:
    values = jnp.asarray(numerator, dtype=jnp.float64) / jnp.asarray(
        denominator,
        dtype=jnp.float64,
    )
    return float(jnp.exp(jnp.mean(jnp.log(values))))


def collect_active_subset_evaluation_metrics(
    particle_counts: Sequence[int],
    *,
    active_fractions: Sequence[float],
    leaf_size: int,
    max_order: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    key: jax.Array,
    fmm_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Collect prepared-state full vs active-subset evaluation timings."""
    rows: List[Dict[str, Any]] = []
    fmm = FastMultipoleMethod(**fmm_kwargs)
    current_key = key

    for n in particle_counts:
        positions, masses = _benchmark_like_distribution(
            int(n),
            key=current_key,
            dtype=dtype,
        )
        current_key, key_subset = jax.random.split(current_key)

        prep_timing = bench_utils.time_callable(
            fmm.prepare_state,
            positions,
            masses,
            leaf_size=leaf_size,
            max_order=max_order,
            warmup=0,
            runs=1,
        )
        state = prep_timing.result

        eval_full_timing = bench_utils.time_callable(
            fmm.evaluate_prepared_state,
            state,
            warmup=warmup,
            runs=runs,
        )

        for frac in active_fractions:
            fraction = float(frac)
            if fraction <= 0.0 or fraction > 1.0:
                raise ValueError("active_fractions must satisfy 0 < f <= 1")
            active_count = max(1, int(round(fraction * int(n))))
            subset_idx = jax.random.permutation(
                key_subset,
                int(n),
                independent=True,
            )[
                :active_count
            ].astype(jnp.int32)
            key_subset, _ = jax.random.split(key_subset)

            eval_active_timing = bench_utils.time_callable(
                fmm.evaluate_prepared_state,
                state,
                target_indices=subset_idx,
                warmup=warmup,
                runs=runs,
            )

            rows.append(
                {
                    "num_particles": int(n),
                    "active_fraction": fraction,
                    "active_count": int(active_count),
                    "prepare_mean_seconds": float(prep_timing.mean),
                    "evaluate_full_mean_seconds": float(eval_full_timing.mean),
                    "evaluate_active_mean_seconds": float(eval_active_timing.mean),
                    "evaluate_speedup_full_over_active": float(
                        eval_full_timing.mean / eval_active_timing.mean
                    ),
                }
            )

    return rows
