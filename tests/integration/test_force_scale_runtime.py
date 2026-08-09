"""Runtime checks for solver-side force-scale estimation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, RuntimePolicyConfig


def _sample_problem(n: int, dtype=jnp.float32):
    key = jax.random.PRNGKey(404)
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
        mac_type="dehnen",
        runtime=RuntimePolicyConfig(
            traversal_config=DualTreeTraversalConfig(
                max_pair_queue=131072,
                process_block=512,
                max_interactions_per_node=65536,
                max_neighbors_per_leaf=65536,
            )
        ),
    )


def _solver(*, mode: str) -> FastMultipoleMethod:
    return FastMultipoleMethod(
        preset="accurate",
        basis="real",
        adaptive_order=True,
        p_gears=(2, 3, 4),
        mac_force_scale_mode=mode,
        theta=5.0e-2,
        softening=1.0e-2,
        advanced=_advanced_cfg(),
    )


def test_force_scale_modes_build_solver_state():
    positions, masses = _sample_problem(72)

    prev_solver = _solver(mode="prev")
    _ = prev_solver.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    state_prev = prev_solver.prepare_state(positions, masses, leaf_size=8, max_order=4)

    assert state_prev.force_scale_nodes is not None
    assert state_prev.force_scale_nodes.shape[0] == state_prev.tree.parent.shape[0]
    assert np.all(np.isfinite(np.asarray(state_prev.force_scale_nodes)))

    prepass_solver = _solver(mode="prepass")
    state_prepass = prepass_solver.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=4,
    )
    assert state_prepass.force_scale_nodes is not None
    assert np.all(np.isfinite(np.asarray(state_prepass.force_scale_nodes)))
    assert not np.allclose(np.asarray(state_prepass.force_scale_nodes), 1.0)


def test_force_scale_prepass_is_stable_across_repeated_runs():
    positions, masses = _sample_problem(72)

    prev_solver = _solver(mode="prev")
    acc_prev = np.asarray(
        prev_solver.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )
    state_prev = prev_solver.prepare_state(positions, masses, leaf_size=8, max_order=4)
    assert state_prev.force_scale_nodes is not None

    prepass_solver = _solver(mode="prepass")
    acc_prepass_first = np.asarray(
        prepass_solver.compute_accelerations(
            positions, masses, leaf_size=8, max_order=4
        )
    )
    acc_prepass_second = np.asarray(
        prepass_solver.compute_accelerations(
            positions, masses, leaf_size=8, max_order=4
        )
    )

    assert np.all(np.isfinite(acc_prepass_first))
    assert np.linalg.norm(acc_prev) > 0.0
    rel_l2 = np.linalg.norm(acc_prepass_second - acc_prepass_first) / (
        np.linalg.norm(acc_prepass_first) + 1.0e-12
    )
    assert rel_l2 < 1.0e-5
