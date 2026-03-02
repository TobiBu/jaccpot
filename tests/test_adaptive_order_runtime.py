"""Adaptive-order gear-bucket runtime checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, RuntimePolicyConfig


def _sample_problem(n: int, dtype=jnp.float32):
    key = jax.random.PRNGKey(202)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )
    return positions, masses


def _advanced_cfg() -> FMMAdvancedConfig:
    return FMMAdvancedConfig(
        runtime=RuntimePolicyConfig(
            traversal_config=DualTreeTraversalConfig(
                max_pair_queue=131072,
                process_block=512,
                max_interactions_per_node=65536,
                max_neighbors_per_leaf=65536,
            )
        )
    )


def test_adaptive_order_false_matches_baseline():
    positions, masses = _sample_problem(80)
    base = FastMultipoleMethod(
        preset="accurate",
        basis="complex",
        theta=0.5,
        softening=1.0e-2,
        advanced=_advanced_cfg(),
    )
    adaptive_off = FastMultipoleMethod(
        preset="accurate",
        basis="complex",
        theta=0.5,
        softening=1.0e-2,
        adaptive_order=False,
        p_gears=(4, 6, 8, 10),
        advanced=_advanced_cfg(),
    )

    acc_base = np.asarray(
        base.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )
    acc_off = np.asarray(
        adaptive_off.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )
    assert np.allclose(acc_base, acc_off, rtol=1.0e-5, atol=1.0e-5)


def test_adaptive_order_true_runs_and_matches_fixed_order():
    positions, masses = _sample_problem(80)
    fixed = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=0.5,
        softening=1.0e-2,
        advanced=_advanced_cfg(),
    )
    adaptive = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=0.5,
        softening=1.0e-2,
        adaptive_order=True,
        p_gears=(2, 3, 4),
        advanced=_advanced_cfg(),
    )

    acc_fixed = np.asarray(
        fixed.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )
    acc_adaptive = np.asarray(
        adaptive.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )
    rel_l2 = np.linalg.norm(acc_adaptive - acc_fixed) / (
        np.linalg.norm(acc_fixed) + 1.0e-12
    )
    assert rel_l2 < 2.0e-2
    assert len(adaptive._impl._recent_far_pairs_by_gear_counts) == 3


def test_dehnen_degree_adaptive_order_runs():
    positions, masses = _sample_problem(80)
    adaptive = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=0.7,
        softening=1.0e-2,
        adaptive_order=True,
        p_gears=(2, 3, 4),
        adaptive_error_model="dehnen_degree",
        adaptive_eps=0.005,
        advanced=_advanced_cfg(),
    )

    acc = np.asarray(
        adaptive.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )

    assert acc.shape == positions.shape
    assert np.all(np.isfinite(acc))
    assert len(adaptive._impl._recent_far_pairs_by_gear_counts) == 3


def test_dehnen_paper_adaptive_order_runs():
    positions, masses = _sample_problem(80)
    adaptive = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=0.7,
        softening=1.0e-2,
        adaptive_order=True,
        p_gears=(2, 3, 4),
        adaptive_error_model="dehnen_paper",
        adaptive_eps=1.0e-2,
        advanced=_advanced_cfg(),
    )

    acc = np.asarray(
        adaptive.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )

    assert acc.shape == positions.shape
    assert np.all(np.isfinite(acc))
    assert len(adaptive._impl._recent_far_pairs_by_gear_counts) == 3
