"""Smoke coverage for performance-metric collection utilities.

These tests do not enforce strict runtime thresholds; they validate that metric
collection executes and produces sane, finite values for drift tracking.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot import (
    FarFieldConfig,
    FMMAdvancedConfig,
    FMMPreset,
    NearFieldConfig,
    TreeConfig,
)

from .perf_metrics import (
    collect_active_subset_evaluation_metrics,
    collect_mode_comparison_metrics,
    collect_prepare_eval_split_metrics,
    geometric_mean_speedup,
)


def _base_runtime_kwargs():
    advanced = FMMAdvancedConfig(
        tree=TreeConfig(leaf_target=16),
        farfield=FarFieldConfig(rotation="solidfmm", mode="pair_grouped"),
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=256),
        mac_type="dehnen",
    )
    return dict(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        advanced=advanced,
    )


def test_collect_prepare_eval_split_metrics_smoke():
    rows = collect_prepare_eval_split_metrics(
        [256, 512],
        leaf_size=16,
        max_order=2,
        runs=1,
        warmup=0,
        dtype=jnp.float32,
        key=jax.random.PRNGKey(0),
        fmm_kwargs=_base_runtime_kwargs(),
    )

    required = {
        "num_particles",
        "mean_seconds",
        "std_seconds",
        "prepare_mean_seconds",
        "evaluate_mean_seconds",
    }
    assert len(rows) == 2
    for row in rows:
        assert required.issubset(row.keys())
    numeric = np.asarray(
        [
            [
                row["mean_seconds"],
                row["std_seconds"],
                row["prepare_mean_seconds"],
                row["evaluate_mean_seconds"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    assert np.all(np.isfinite(numeric))
    assert np.all(numeric >= 0.0)


def test_collect_mode_comparison_metrics_smoke():
    base_kwargs = _base_runtime_kwargs()
    base_advanced = base_kwargs["advanced"]
    variants = [
        {
            "name": "pair_grouped",
            "fmm_overrides": {
                "advanced": replace(
                    base_advanced,
                    farfield=replace(base_advanced.farfield, mode="pair_grouped"),
                    nearfield=replace(base_advanced.nearfield, mode="baseline"),
                ),
            },
        },
        {
            "name": "class_major",
            "fmm_overrides": {
                "advanced": replace(
                    base_advanced,
                    farfield=replace(base_advanced.farfield, mode="class_major"),
                    nearfield=replace(base_advanced.nearfield, mode="bucketed"),
                ),
            },
        },
    ]
    rows = collect_mode_comparison_metrics(
        [256],
        leaf_size=16,
        max_order=2,
        runs=1,
        warmup=0,
        dtype=jnp.float32,
        key=jax.random.PRNGKey(1),
        base_fmm_kwargs=base_kwargs,
        mode_variants=variants,
    )

    assert len(rows) == 1
    row = rows[0]
    assert "mean_seconds_pair_grouped" in row
    assert "mean_seconds_class_major" in row
    speedup = geometric_mean_speedup(
        [row["mean_seconds_pair_grouped"]],
        [row["mean_seconds_class_major"]],
    )
    assert np.isfinite(speedup)
    assert speedup > 0.0


def test_collect_active_subset_evaluation_metrics_smoke():
    rows = collect_active_subset_evaluation_metrics(
        [512],
        active_fractions=[0.125, 0.5],
        leaf_size=16,
        max_order=2,
        runs=1,
        warmup=0,
        dtype=jnp.float32,
        key=jax.random.PRNGKey(7),
        fmm_kwargs=_base_runtime_kwargs(),
    )

    required = {
        "num_particles",
        "active_fraction",
        "active_count",
        "prepare_mean_seconds",
        "evaluate_full_mean_seconds",
        "evaluate_active_mean_seconds",
        "evaluate_speedup_full_over_active",
    }
    assert len(rows) == 2
    for row in rows:
        assert required.issubset(row.keys())
        assert row["active_count"] >= 1

    numeric = np.asarray(
        [
            [
                row["prepare_mean_seconds"],
                row["evaluate_full_mean_seconds"],
                row["evaluate_active_mean_seconds"],
                row["evaluate_speedup_full_over_active"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    assert np.all(np.isfinite(numeric))
    assert np.all(numeric > 0.0)
