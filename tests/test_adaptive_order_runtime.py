"""Adaptive-order gear-bucket runtime checks."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.interactions import DualTreeTraversalConfig

import jaccpot.runtime._fmm_impl as fmm_impl_private
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


def _advanced_cfg(*, mac_type: str | None = None) -> FMMAdvancedConfig:
    return FMMAdvancedConfig(
        mac_type=mac_type,
        runtime=RuntimePolicyConfig(
            traversal_config=DualTreeTraversalConfig(
                max_pair_queue=131072,
                process_block=512,
                max_interactions_per_node=65536,
                max_neighbors_per_leaf=65536,
            )
        ),
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


def test_dehnen_paper_fixed_order_runs():
    positions, masses = _sample_problem(80)
    fixed = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=0.6,
        softening=1.0e-2,
        adaptive_order=False,
        adaptive_error_model="dehnen_paper",
        adaptive_eps=1.0e-3,
        advanced=_advanced_cfg(),
    )

    acc = np.asarray(
        fixed.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )

    assert acc.shape == positions.shape
    assert np.all(np.isfinite(acc))


def test_dehnen_paper_fixed_order_runs_with_paper_force_scale_mode():
    positions, masses = _sample_problem(80)
    fixed = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=0.6,
        softening=1.0e-2,
        adaptive_order=False,
        adaptive_error_model="dehnen_paper",
        adaptive_eps=1.0e-3,
        mac_force_scale_mode="paper",
        advanced=_advanced_cfg(),
    )

    acc = np.asarray(
        fixed.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )

    assert acc.shape == positions.shape
    assert np.all(np.isfinite(acc))


def test_dehnen_paper_fixed_order_runs_with_tree_geometry_mode():
    positions, masses = _sample_problem(80)
    fixed = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=0.6,
        softening=1.0e-2,
        adaptive_order=False,
        adaptive_error_model="dehnen_paper",
        dehnen_geometry_mode="tree",
        adaptive_eps=1.0e-3,
        mac_force_scale_mode="paper",
        advanced=_advanced_cfg(),
    )

    acc = np.asarray(
        fixed.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )

    assert acc.shape == positions.shape
    assert np.all(np.isfinite(acc))


def test_dehnen_paper_fixed_order_runs_with_tree_approx_geometry_mode():
    positions, masses = _sample_problem(80)
    fixed = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=0.6,
        softening=1.0e-2,
        adaptive_order=False,
        adaptive_error_model="dehnen_paper",
        dehnen_geometry_mode="tree_approx",
        adaptive_eps=1.0e-3,
        mac_force_scale_mode="paper",
        advanced=_advanced_cfg(),
    )

    acc = np.asarray(
        fixed.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )

    assert acc.shape == positions.shape
    assert np.all(np.isfinite(acc))


def test_dehnen_error_fixed_order_runs():
    positions, masses = _sample_problem(80)
    fixed = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=0.6,
        softening=1.0e-2,
        adaptive_order=False,
        advanced=_advanced_cfg(mac_type="dehnen_error"),
    )

    acc = np.asarray(
        fixed.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )

    assert acc.shape == positions.shape
    assert np.all(np.isfinite(acc))


def test_dehnen_error_uses_adaptive_pair_policy(monkeypatch):
    fmm = FastMultipoleMethod(
        preset="accurate",
        basis="real",
        theta=0.6,
        softening=1.0e-2,
        adaptive_order=False,
        advanced=_advanced_cfg(mac_type="dehnen_error"),
    )

    sentinel_policy_state = object()
    seen = {}

    def fake_build_adaptive_policy_state(**kwargs):
        seen["build_policy_kwargs"] = kwargs
        return sentinel_policy_state

    class _StopTraversal(RuntimeError):
        pass

    def fake_build_dual_tree_artifacts(*args, **kwargs):
        seen["pair_policy"] = kwargs.get("pair_policy")
        seen["policy_state"] = kwargs.get("policy_state")
        raise _StopTraversal("stop after recording traversal policy")

    monkeypatch.setattr(
        fmm._impl,
        "_build_adaptive_policy_state",
        fake_build_adaptive_policy_state,
    )
    monkeypatch.setattr(
        fmm_impl_private,
        "_build_dual_tree_artifacts",
        fake_build_dual_tree_artifacts,
    )

    upward = SimpleNamespace(
        geometry=SimpleNamespace(),
        multipoles=SimpleNamespace(
            order=4,
            packed=jnp.zeros((1, 25), dtype=jnp.float32),
        ),
    )
    tree_artifacts = fmm_impl_private._PrepareStateTreeUpwardArtifacts(
        tree_mode="lbvh",
        tree=SimpleNamespace(),
        positions_sorted=jnp.zeros((1, 3), dtype=jnp.float32),
        masses_sorted=jnp.ones((1,), dtype=jnp.float32),
        inverse_permutation=jnp.asarray([0], dtype=jnp.int32),
        leaf_cap=8,
        leaf_parameter=8,
        topology_key=None,
        upward=upward,
        locals_template=None,
    )

    with pytest.raises(_StopTraversal, match="recording traversal policy"):
        fmm._impl._prepare_state_dual_and_downward(
            tree_artifacts=tree_artifacts,
            force_scale_nodes=jnp.ones((1,), dtype=jnp.float32),
            upward_center_mode="com",
            theta_val=0.6,
            mac_type_val="dehnen",
            dehnen_radius_scale=1.0,
            runtime_traversal_config=_advanced_cfg().runtime.traversal_config,
            runtime_m2l_chunk_size=None,
            runtime_l2l_chunk_size=None,
            grouped_interactions=False,
            farfield_mode="pair_grouped",
            record_retry=lambda event: None,
            refine_local_val=False,
            max_refine_levels_val=0,
            aspect_threshold_val=16.0,
            allow_stateful_cache=False,
        )

    assert seen["pair_policy"] is fmm_impl_private.adaptive_pair_policy
    assert seen["policy_state"] is sentinel_policy_state
    assert int(seen["build_policy_kwargs"]["error_model_code"]) == 2
    assert tuple(int(v) for v in seen["build_policy_kwargs"]["p_gears"]) == (4,)
