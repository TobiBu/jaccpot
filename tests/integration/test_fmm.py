"""Tests for Fast Multipole Method."""

import json
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yggdrax.interactions as tree_interactions_module
from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.geometry import compute_tree_geometry
from yggdrax.interactions import DualTreeTraversalConfig, build_leaf_neighbor_lists
from yggdrax.tree import build_tree

import jaccpot.runtime._fmm_impl as fmm_impl_private
import jaccpot.runtime.fmm as fmm_module
import jaccpot.runtime.fmm_prepare as fmm_prepare_private
import jaccpot.runtime.kernels.core as kernels_core
from jaccpot import FMMPreset
from jaccpot.config import (
    FarFieldConfig,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)
from jaccpot.downward.local_expansions import (
    TreeDownwardData,
)
from jaccpot.downward.local_expansions import (
    prepare_downward_sweep as prepare_local_downward_sweep,
)
from jaccpot.downward.local_expansions import (
    run_downward_sweep as run_local_downward_sweep,
)
from jaccpot.nearfield.near_field import (
    compute_leaf_p2p_accelerations,
    compute_leaf_p2p_accelerations_large_n_accel_only,
    prepare_bucketed_scatter_schedules,
    prepare_leaf_neighbor_pairs,
)
from jaccpot.runtime.fmm import (
    FMMEngine,
    compute_gravitational_acceleration,
    compute_gravitational_potential,
)

DEFAULT_TEST_LEAF_SIZE = 1


def _direct_sum(positions, masses, *, G: float, softening: float):
    positions_np = np.asarray(positions)
    masses_np = np.asarray(masses)
    n = positions_np.shape[0]
    accelerations = np.zeros_like(positions_np)
    potentials = np.zeros((n,), dtype=positions_np.dtype)
    eps = np.finfo(positions_np.dtype).eps
    soft_sq = softening**2

    for i in range(n):
        diff = positions_np[i] - positions_np
        dist_sq = np.sum(diff * diff, axis=1) + soft_sq
        dist = np.sqrt(dist_sq)
        denom = dist_sq * dist + eps
        inv_dist3 = 1.0 / denom
        inv_dist3[i] = 0.0
        weighted = masses_np[:, None] * inv_dist3[:, None] * diff
        accelerations[i] = -G * np.sum(weighted, axis=0)

        inv_r = 1.0 / (dist + eps)
        inv_r[i] = 0.0
        potentials[i] = -G * np.sum(masses_np * inv_r)

    return accelerations, potentials


def test_compute_expansion_orders():
    """Multipole coefficients reflect requested order."""
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([2.0, 1.0])

    # Order 0: only monopole, dipole/quadrupole zero
    exp0 = FMMEngine.compute_expansion(positions, masses, order=0)
    assert jnp.isclose(exp0.monopole, 3.0)
    assert jnp.allclose(exp0.dipole, jnp.zeros(3))
    assert jnp.allclose(exp0.quadrupole, jnp.zeros((3, 3)))

    # Order 1: dipole enabled, quad still zero
    exp1 = FMMEngine.compute_expansion(positions, masses, order=1)
    assert jnp.isclose(exp1.monopole, exp0.monopole)
    assert jnp.all(jnp.isfinite(exp1.dipole))
    assert jnp.allclose(exp1.quadrupole, jnp.zeros((3, 3)))

    # Order 2: quadrupole enabled
    exp2 = FMMEngine.compute_expansion(positions, masses, order=2)
    assert jnp.isclose(exp2.monopole, exp0.monopole)
    assert jnp.all(jnp.isfinite(exp2.dipole))
    assert jnp.all(jnp.isfinite(exp2.quadrupole))


def test_evaluate_expansion_consistency():
    """evaluate_expansion gives consistent results across orders."""
    fmm = FMMEngine(G=1.0, softening=0.0)
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 2.0])
    point = jnp.array([2.0, 0.5, -1.0])

    exp0 = FMMEngine.compute_expansion(positions, masses, order=0)
    exp2 = FMMEngine.compute_expansion(positions, masses, order=2)

    # Monopole evaluation should match regardless of expansion order
    a0 = fmm.evaluate_expansion(exp0, order=0, eval_point=point)
    a0_ref = fmm.evaluate_expansion(exp2, order=0, eval_point=point)
    assert jnp.allclose(a0, a0_ref)

    # Order-2 evaluation should be finite
    a2 = fmm.evaluate_expansion(exp2, order=2, eval_point=point)
    assert jnp.all(jnp.isfinite(a2))


def test_multipole_accuracy_improves_with_order():
    """Higher order multipoles should not worsen far-field accuracy."""
    key = jax.random.PRNGKey(0)
    n = 20
    # Cluster within small radius around origin
    pos = 0.1 * jax.random.normal(key, (n, 3))
    mass = jnp.ones((n,))

    fmm = FMMEngine(G=1.0, softening=0.0)
    eval_point = jnp.array([3.0, 0.5, -1.0])

    # Reference direct sum at eval point
    a_ref = fmm.direct_sum(pos, mass, eval_point)

    # Expansions around CoM
    exp0 = FMMEngine.compute_expansion(pos, mass, order=0)
    exp1 = FMMEngine.compute_expansion(pos, mass, order=1)
    exp2 = FMMEngine.compute_expansion(pos, mass, order=2)

    a0 = fmm.evaluate_expansion(exp0, order=0, eval_point=eval_point)
    a1 = fmm.evaluate_expansion(exp1, order=1, eval_point=eval_point)
    a2 = fmm.evaluate_expansion(exp2, order=2, eval_point=eval_point)

    def err(a):
        return jnp.linalg.norm(a - a_ref)

    e0 = err(a0)
    e1 = err(a1)
    e2 = err(a2)

    # Non-increasing error with order
    assert e0 >= e1 - 1e-7
    assert e1 >= e2 - 1e-7


def test_zero_total_mass_expansion():
    """Zero total mass yields zero multipole moments (finite center)."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    masses = jnp.array([0.0, 0.0])

    exp = FMMEngine.compute_expansion(positions, masses, order=2)
    assert jnp.isclose(exp.monopole, 0.0)
    assert jnp.allclose(exp.dipole, 0.0)
    assert jnp.allclose(exp.quadrupole, 0.0)
    # Center should be finite numbers
    assert jnp.all(jnp.isfinite(exp.center))


def test_monopole_expansion():
    """Test monopole expansion calculation."""
    # Two particles with known center of mass
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 1.0])

    expansion = FMMEngine.compute_expansion(
        positions,
        masses,
        order=0,
    )

    # Total mass should be 2.0
    assert jnp.isclose(expansion.monopole, 2.0)

    # Center of mass should be at (0.5, 0.0, 0.0)
    assert jnp.allclose(expansion.center, jnp.array([0.5, 0.0, 0.0]))


def test_direct_acceleration():
    """Test direct summation of gravitational acceleration."""
    # Two particles on x-axis
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 1.0])

    G = 1.0
    softening = 0.0

    # Compute acceleration
    accelerations = compute_gravitational_acceleration(
        positions, masses, G=G, softening=softening
    )

    # First particle should be accelerated in +x direction
    # Second particle should be accelerated in -x direction
    assert accelerations[0, 0] > 0  # +x
    assert accelerations[1, 0] < 0  # -x


def test_prepare_state_fixed_depth_tree():
    n = 64
    positions = jnp.stack(
        [jnp.linspace(-1.0, 1.0, n), jnp.zeros((n,)), jnp.zeros((n,))],
        axis=1,
    )
    masses = jnp.ones((n,))
    fmm = FMMEngine(
        theta=0.6,
        tree=TreeConfig(mode="fixed_depth", leaf_target=8),
    )

    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=1,
        jit_tree=False,
    )

    assert state.tree.num_particles == n

    leaf_ranges = state.tree.node_ranges[state.tree.num_internal_nodes :]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    assert state.max_leaf_size == int(jnp.max(counts))


def test_prepare_refresh_static_radix_tree_preserves_static_shape(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    # This test profiles the strict cap on the fly (there is no pre-recorded
    # profile for its key), so do not require an exact cap-profile match --
    # otherwise it depends on another test having recorded one first (order
    # dependence). Matches the sibling strict-lane tests below.
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")

    key = jax.random.PRNGKey(123)
    core = 0.01 * jax.random.normal(key, (160, 3), dtype=jnp.float32)
    halo = jax.random.uniform(
        key,
        (64, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    positions = jnp.concatenate([core, halo], axis=0)
    masses = jnp.ones((positions.shape[0],), dtype=jnp.float32)

    fmm = FMMEngine(
        preset="large_n_gpu",
        runtime_path="large_n",
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
    )
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=32,
        max_order=2,
    )
    moved = positions.at[:8].add(jnp.array([1.0e-5, 0.0, 0.0], dtype=jnp.float32))
    refreshed = fmm.refresh_prepared_state(
        state,
        moved,
        masses,
        leaf_size=32,
        max_order=2,
    )
    diagnostics = fmm.get_runtime_diagnostics()

    assert state.tree.build_mode == "static_radix"
    assert refreshed.tree.build_mode == "static_radix"
    assert state.max_leaf_size <= 32
    assert refreshed.max_leaf_size <= 32
    assert refreshed.tree.num_leaves == state.tree.num_leaves
    assert state.tree.parent.shape == refreshed.tree.parent.shape
    assert (
        state.neighbor_list.neighbors.shape == refreshed.neighbor_list.neighbors.shape
    )
    assert diagnostics["large_n_same_topology_refresh_hits"] >= 1
    assert diagnostics["static_radix_refresh_hits"] >= 1


def test_static_radix_refresh_rebuilds_current_large_n_payloads(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    # Profile the strict cap on the fly rather than requiring a pre-recorded
    # profile match (which would make this test depend on run order).
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "4")
    monkeypatch.setenv("JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "16")

    key = jax.random.PRNGKey(20260507)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (2048, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (2048,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )
    displacement = 0.02 * jnp.stack(
        [
            jnp.sin(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.13),
            jnp.cos(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.17),
            jnp.sin(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.19),
        ],
        axis=1,
    )
    moved = positions + displacement

    kwargs = dict(
        preset="large_n_gpu",
        runtime_path="large_n",
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm", grouped_interactions=False),
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
    )

    fmm = FMMEngine(**kwargs)
    state = fmm.prepare_state(positions, masses, leaf_size=128, max_order=2)
    refreshed = fmm.refresh_prepared_state(
        state,
        moved,
        masses,
        leaf_size=128,
        max_order=2,
    )
    diagnostics = fmm.get_runtime_diagnostics()

    fresh_fmm = FMMEngine(**kwargs)
    fresh = fresh_fmm.prepare_state(moved, masses, leaf_size=128, max_order=2)

    assert diagnostics["large_n_same_topology_refresh_hits"] >= 1
    assert diagnostics["static_radix_refresh_hits"] >= 1
    assert refreshed.tree.build_mode == "static_radix"
    assert fresh.tree.build_mode == "static_radix"

    refreshed_acc = np.asarray(fmm.evaluate_prepared_state(refreshed))
    fresh_acc = np.asarray(fresh_fmm.evaluate_prepared_state(fresh))
    assert np.allclose(refreshed_acc, fresh_acc, rtol=1e-5, atol=1e-5)

    def assert_array_equal(left, right):
        if left is None or right is None:
            assert left is None and right is None
            return
        assert np.array_equal(np.asarray(left), np.asarray(right))

    assert_array_equal(
        refreshed.nearfield_leaf_particle_indices,
        fresh.nearfield_leaf_particle_indices,
    )
    assert_array_equal(
        refreshed.nearfield_leaf_particle_mask,
        fresh.nearfield_leaf_particle_mask,
    )
    assert_array_equal(
        refreshed.nearfield_target_block_source_leaf_ids_padded,
        fresh.nearfield_target_block_source_leaf_ids_padded,
    )
    assert_array_equal(
        refreshed.nearfield_target_block_valid_mask_padded,
        fresh.nearfield_target_block_valid_mask_padded,
    )
    assert_array_equal(
        refreshed.nearfield_target_block_source_leaf_ids,
        fresh.nearfield_target_block_source_leaf_ids,
    )
    assert_array_equal(
        refreshed.nearfield_target_block_valid_mask,
        fresh.nearfield_target_block_valid_mask,
    )
    assert_array_equal(
        refreshed.nearfield_target_block_offsets,
        fresh.nearfield_target_block_offsets,
    )

    assert refreshed.radix_fast_payload is not None
    assert fresh.radix_fast_payload is not None
    for attr in (
        "target_particle_ids",
        "target_particle_mask",
        "source_leaf_ids",
        "source_leaf_valid_mask",
        "source_particle_ids",
        "source_particle_mask",
    ):
        assert_array_equal(
            getattr(refreshed.radix_fast_payload, attr),
            getattr(fresh.radix_fast_payload, attr),
        )


def test_static_radix_refresh_dual_planner_mode_parity_and_diagnostics(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")

    key = jax.random.PRNGKey(20260512)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (1024, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (1024,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )
    displacement = 0.01 * jnp.stack(
        [
            jnp.sin(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.11),
            jnp.cos(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.07),
            jnp.sin(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.05),
        ],
        axis=1,
    )
    moved = positions + displacement
    moved_2 = moved + 0.5 * displacement

    kwargs = dict(
        preset="large_n_gpu",
        runtime_path="large_n",
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm", grouped_interactions=False),
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
    )

    monkeypatch.setenv("JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_MODE", "off")
    fmm_off = FMMEngine(**kwargs)
    state_off = fmm_off.prepare_state(positions, masses, leaf_size=128, max_order=2)
    refreshed_off = fmm_off.refresh_prepared_state(
        state_off,
        moved,
        masses,
        leaf_size=128,
        max_order=2,
    )
    acc_off = np.asarray(fmm_off.evaluate_prepared_state(refreshed_off))

    monkeypatch.setenv("JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "on")
    fmm_on = FMMEngine(**kwargs)
    state_on = fmm_on.prepare_state(positions, masses, leaf_size=128, max_order=2)
    refreshed_on = fmm_on.refresh_prepared_state(
        state_on,
        moved,
        masses,
        leaf_size=128,
        max_order=2,
    )
    _ = fmm_on.refresh_prepared_state(
        refreshed_on,
        moved_2,
        masses,
        leaf_size=128,
        max_order=2,
    )
    acc_on = np.asarray(fmm_on.evaluate_prepared_state(refreshed_on))
    diagnostics_on = fmm_on.get_runtime_diagnostics()

    assert np.allclose(acc_on, acc_off, rtol=1e-5, atol=1e-5)
    assert diagnostics_on["refresh_dual_planner_execute_count"] >= 2
    assert diagnostics_on["refresh_dual_planner_cache_hits"] >= 1
    assert diagnostics_on["refresh_dual_planner_steady_timing_bypass_count"] >= 0
    assert diagnostics_on["refresh_strict_mode_active_count"] >= 2
    # Strict static fast-lane can bypass compiled route probing entirely.
    if diagnostics_on["refresh_dual_planner_compile_count"] == 0:
        assert diagnostics_on["refresh_dual_planner_compiled_route_count"] == 0
    else:
        assert diagnostics_on["refresh_dual_planner_compile_count"] >= 1
        assert diagnostics_on["refresh_dual_planner_compiled_route_count"] >= 1


def test_strict_prepare_refresh_and_evaluate_api_and_diagnostics(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")

    key = jax.random.PRNGKey(20260513)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (1024, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (1024,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )
    moved = positions + 0.01 * jnp.stack(
        [
            jnp.sin(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.11),
            jnp.cos(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.07),
            jnp.sin(jnp.arange(positions.shape[0], dtype=jnp.float32) * 0.05),
        ],
        axis=1,
    )

    fmm = FMMEngine(
        preset="large_n_gpu",
        runtime_path="large_n",
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm", grouped_interactions=False),
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
    )
    state0, acc0 = fmm.strict_prepare_refresh_and_evaluate(
        None,
        positions,
        masses,
        leaf_size=128,
        max_order=2,
        theta=0.6,
    )
    state1, acc1 = fmm.strict_prepare_refresh_and_evaluate(
        state0,
        moved,
        masses,
        leaf_size=128,
        max_order=2,
        theta=0.6,
    )

    assert state0.tree.build_mode == "static_radix"
    assert state1.tree.build_mode == "static_radix"
    assert np.asarray(acc0).shape == (positions.shape[0], 3)
    assert np.asarray(acc1).shape == (positions.shape[0], 3)

    diagnostics = fmm.get_runtime_diagnostics()
    # Two strict_prepare_refresh_and_evaluate calls -> two strict-runner
    # executions (the increment is +1 per call at _fmm_impl.py:3732).
    assert diagnostics["strict_runner_execute_count"] >= 2
    assert diagnostics["strict_runner_compile_count"] >= 1
    assert diagnostics["strict_runner_profile_key_misses"] >= 1
    assert diagnostics["strict_runner_profile_key_hits"] >= 1


def test_strict_exact_cap_profile_match_fail_fast(monkeypatch, tmp_path):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "1")
    profile_path = tmp_path / "strict_caps.json"
    profile_path.write_text(
        json.dumps(
            {
                "version": 2,
                "active_context_key": "tree_mode=static_radix|leaf=64|n=999",
                "profiles": {
                    "tree_mode=static_radix|leaf=64|n=999": {
                        "max_pair_queue": 16384,
                        "pair_process_block": 1024,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH", str(profile_path))

    key = jax.random.PRNGKey(20260513)
    positions = jax.random.uniform(
        key,
        (1024, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.ones((1024,), dtype=jnp.float32)
    moved = positions + 1e-3

    fmm = FMMEngine(
        preset="large_n_gpu",
        runtime_path="large_n",
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm", grouped_interactions=False),
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
    )
    with pytest.raises(RuntimeError, match="exact cap profile key match"):
        _ = fmm.prepare_state(positions, masses, leaf_size=128, max_order=2)
    diagnostics = fmm.get_runtime_diagnostics()
    assert diagnostics["strict_runner_fail_fast_reject_count"] >= 1


def test_strict_run_v2_api(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")
    # Pin the fused static-sizing caps to fixed adequate values (mirroring the
    # sibling strict-fused tests below). Without this the fused lane sizes the
    # static neighbour-edge cap from the *first* build's active-edge count, which
    # is sensitive to leaked process-global tree/neighbour construction state
    # from earlier tests on the same xdist worker; an undersized first-build cap
    # then overflows on a later scan step ("neighbor-edge cap exceeded"). Fixed
    # caps make the run deterministic and order-independent.
    monkeypatch.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")

    key = jax.random.PRNGKey(20260513)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (2048, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (2048,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )
    state0 = jnp.stack(
        [
            positions,
            jnp.zeros_like(positions),
        ],
        axis=1,
    )

    fmm = FMMEngine(
        preset="large_n_gpu",
        runtime_path="large_n",
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm", grouped_interactions=False),
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
    )

    state_out, prepared_out, history = fmm.strict_run_v2(
        state=state0,
        masses=masses,
        dt=1e-3,
        num_steps=3,
        refresh_every=1,
        leaf_size=128,
        max_order=2,
        theta=0.6,
        return_history=False,
    )
    assert prepared_out.tree.build_mode == "static_radix"
    assert history is None
    assert np.asarray(state_out).shape == np.asarray(state0).shape

    diagnostics = fmm.get_runtime_diagnostics()
    assert diagnostics["strict_runner_execute_count"] >= 2


def test_strict_fused_moved_endpoint_matches_fresh_prepare(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY", "1")
    monkeypatch.setenv(
        "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK", "1"
    )
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS", "1")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP", "32768")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")
    monkeypatch.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")

    key = jax.random.PRNGKey(20260620)
    key_pos, key_vel, key_mass = jax.random.split(key, 3)
    positions = jax.random.uniform(
        key_pos,
        (512, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    velocities = 0.01 * jax.random.normal(key_vel, (512, 3), dtype=jnp.float32)
    masses = jax.random.uniform(
        key_mass,
        (512,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )
    state0 = jnp.stack([positions, velocities], axis=1)

    kwargs = dict(
        preset="large_n_gpu",
        runtime_path="large_n",
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm", grouped_interactions=False),
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
    )
    fmm = FMMEngine(**kwargs)
    state_out, prepared_out, history = fmm.strict_run_v2(
        state=state0,
        masses=masses,
        dt=2.0e-4,
        num_steps=1,
        refresh_every=1,
        leaf_size=128,
        max_order=2,
        theta=0.6,
        return_history=False,
    )
    assert history is None
    diagnostics = fmm.get_runtime_diagnostics()
    assert diagnostics["strict_fused_mode_active"] is True
    assert diagnostics["strict_fused_fallback_count"] == 0
    assert diagnostics["strict_self_force_endpoint_evaluations"] == 1

    fresh_fmm = FMMEngine(**kwargs)
    fresh = fresh_fmm.prepare_state(
        state_out[:, 0, :], masses, leaf_size=128, max_order=2
    )
    endpoint_acc = np.asarray(fmm.evaluate_prepared_state(prepared_out))
    fresh_acc = np.asarray(fresh_fmm.evaluate_prepared_state(fresh))
    diff = endpoint_acc - fresh_acc
    rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(fresh_acc), 1.0e-12)
    assert rel_l2 <= 1.0e-4
    assert np.max(np.abs(diff)) <= 5.0e-3


def test_strict_fused_compact_far_pair_cap_fails(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY", "1")
    monkeypatch.setenv(
        "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK", "0"
    )
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS", "1")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")
    monkeypatch.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")

    key = jax.random.PRNGKey(20260621)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (2048, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (2048,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )
    state0 = jnp.stack([positions, jnp.zeros_like(positions)], axis=1)

    fmm = FMMEngine(
        preset="large_n_gpu",
        runtime_path="large_n",
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm", grouped_interactions=False),
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
    )
    with pytest.raises(
        Exception,
        match="JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP|compact far-pair cap exceeded",
    ):
        fmm.strict_run_v2(
            state=state0,
            masses=masses,
            dt=1.0e-4,
            num_steps=1,
            refresh_every=1,
            leaf_size=64,
            max_order=2,
            theta=0.6,
            return_history=False,
        )


def test_capacity_fixed_depth_tree_mode_is_removed():
    with pytest.raises(ValueError, match="tree_build_mode"):
        FMMEngine(
            tree=TreeConfig(mode="capacity_fixed_depth"),
        )


def _fixed_depth_sample():
    positions = jnp.array(
        [
            [-0.6, -0.2, 0.1],
            [-0.1, 0.4, -0.3],
            [0.3, -0.5, 0.2],
            [0.7, 0.1, -0.4],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 0.8, 1.2, 0.9], dtype=jnp.float64)
    return positions, masses


def _line_cluster_sample():
    xs = jnp.linspace(-0.9, 0.9, 16)
    positions = jnp.stack(
        [xs, jnp.zeros_like(xs), jnp.zeros_like(xs)],
        axis=1,
        dtype=jnp.float64,
    )
    masses = jnp.ones((xs.shape[0],), dtype=jnp.float64)
    return positions, masses


def test_compute_accelerations_fixed_depth_matches_direct():
    positions, masses = _fixed_depth_sample()
    theta = 0.7
    G = 1.1
    softening = 0.02

    fmm = FMMEngine(
        theta=theta,
        G=G,
        softening=softening,
        tree=TreeConfig(mode="fixed_depth", leaf_target=2),
    )
    acc, pot = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=2,
        return_potential=True,
        jit_tree=False,
        jit_traversal=False,
    )

    direct_acc, direct_pot = _direct_sum(
        np.asarray(positions),
        np.asarray(masses),
        G=G,
        softening=softening,
    )

    assert np.allclose(np.asarray(acc), direct_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(pot), direct_pot, rtol=1e-6, atol=1e-6)


def test_compute_accelerations_refined_tree_matches_non_refined():
    """Ensure both refine_local modes stay accurate when leaves retain
    multiple particles."""
    positions, masses = _line_cluster_sample()
    theta = 0.7
    G = 1.1
    softening = 0.02
    leaf_size = 8
    target_leaf_particles = 8
    max_refine_levels = 1
    aspect_threshold = 4.0

    def run(refine_local_flag: bool):
        fmm = FMMEngine(
            theta=theta,
            G=G,
            softening=softening,
            tree=TreeConfig(mode="fixed_depth", leaf_target=target_leaf_particles),
        )
        acc, pot = fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=leaf_size,
            return_potential=True,
            refine_local=refine_local_flag,
            jit_tree=False,
            jit_traversal=False,
            max_refine_levels=max_refine_levels,
            aspect_threshold=aspect_threshold,
        )
        return np.asarray(acc), np.asarray(pot)

    acc_no, pot_no = run(False)
    acc_ref, pot_ref = run(True)
    direct_acc, direct_pot = _direct_sum(
        np.asarray(positions),
        np.asarray(masses),
        G=G,
        softening=softening,
    )

    assert np.allclose(acc_ref, direct_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(pot_ref, direct_pot, rtol=1e-6, atol=1e-6)
    assert np.allclose(acc_no, direct_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(pot_no, direct_pot, rtol=1e-6, atol=1e-6)


def test_compute_accelerations_fixed_depth_jitted_matches_eager():
    positions, masses = _fixed_depth_sample()

    def run(jit_tree: bool, jit_traversal: bool):
        fmm = FMMEngine(
            theta=0.7,
            G=1.1,
            softening=0.02,
            tree=TreeConfig(mode="fixed_depth", leaf_target=2),
        )
        acc, pot = fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=2,
            return_potential=True,
            jit_tree=jit_tree,
            jit_traversal=jit_traversal,
        )
        return np.asarray(acc), np.asarray(pot)

    eager_acc, eager_pot = run(False, False)
    jit_acc, jit_pot = run(True, True)

    assert np.allclose(eager_acc, jit_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(eager_pot, jit_pot, rtol=1e-6, atol=1e-6)


def test_acceleration_magnitude():
    """Test that acceleration magnitude follows inverse square law."""
    # Single particle at origin, evaluate at different distances
    positions = jnp.array([[0.0, 0.0, 0.0]])
    masses = jnp.array([1.0])

    fmm = FMMEngine(G=1.0, softening=0.0)

    # Points at distance 1 and 2
    point1 = jnp.array([1.0, 0.0, 0.0])
    point2 = jnp.array([2.0, 0.0, 0.0])

    expansion = FMMEngine.compute_expansion(
        positions,
        masses,
        order=0,
    )
    accel1 = fmm.evaluate_expansion(expansion, order=0, eval_point=point1)
    accel2 = fmm.evaluate_expansion(expansion, order=0, eval_point=point2)

    mag1 = jnp.sqrt(jnp.sum(accel1**2))
    mag2 = jnp.sqrt(jnp.sum(accel2**2))

    # Acceleration at distance 2 should be 1/4 of acceleration at distance 1
    assert jnp.isclose(mag2, mag1 / 4.0, rtol=1e-5)


def test_gravitational_potential():
    """Test gravitational potential calculation."""
    # Single particle at origin
    positions = jnp.array([[0.0, 0.0, 0.0]])
    masses = jnp.array([1.0])

    # Evaluate potential at distance 1
    eval_points = jnp.array([[1.0, 0.0, 0.0]])

    G = 1.0
    softening = 0.0

    potential = compute_gravitational_potential(
        positions, masses, eval_points, G=G, softening=softening
    )

    # Potential should be -G*M/r = -1.0
    assert jnp.isclose(potential[0], -1.0, rtol=1e-5)


def test_softening():
    """Test that softening prevents singularities."""
    # Particle very close to itself (should not diverge with softening)
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1e-10, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 1.0])

    # With softening
    accelerations = compute_gravitational_acceleration(
        positions, masses, G=1.0, softening=0.1
    )

    # Should not have NaN or Inf
    assert jnp.all(jnp.isfinite(accelerations))

    # Acceleration magnitudes should be reasonable
    mags = jnp.sqrt(jnp.sum(accelerations**2, axis=1))
    assert jnp.all(mags < 1000.0)


def test_zero_mass():
    """Test handling of zero mass particles."""
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    masses = jnp.array([1.0, 0.0])

    accelerations = compute_gravitational_acceleration(
        positions, masses, G=1.0, softening=0.0
    )

    # Should not have NaN or Inf
    assert jnp.all(jnp.isfinite(accelerations))

    # Zero mass particle should experience acceleration
    # but should not contribute to acceleration of other particle
    assert jnp.abs(accelerations[1, 0]) > 0  # Experiences acceleration


def test_stf_symmetry_and_trace_free():
    """Quadrupole, octupole, hexadecapole are symmetric and trace-free."""
    key = jax.random.PRNGKey(1)
    n = 10
    pos = 0.2 * jax.random.normal(key, (n, 3))
    mass = jax.random.uniform(key, (n,), minval=0.1, maxval=2.0)

    exp = FMMEngine.compute_expansion(pos, mass, order=4)

    # Quadrupole symmetry and trace-free
    Q = exp.quadrupole
    assert jnp.allclose(Q, jnp.swapaxes(Q, 0, 1), atol=1e-6)
    tr_Q = jnp.trace(Q)
    assert jnp.allclose(tr_Q, 0.0, atol=1e-5)

    # Octupole: symmetric under pairwise swaps, single traces vanish
    oct_t = exp.octupole
    assert jnp.allclose(oct_t, jnp.transpose(oct_t, (1, 0, 2)), atol=1e-6)
    assert jnp.allclose(oct_t, jnp.transpose(oct_t, (2, 1, 0)), atol=1e-6)
    # single traces
    tr1 = jnp.trace(oct_t, axis1=0, axis2=1)  # shape (3,)
    tr2 = jnp.trace(oct_t, axis1=0, axis2=2)  # shape (3,)
    tr3 = jnp.trace(oct_t, axis1=1, axis2=2)  # shape (3,)
    assert jnp.allclose(tr1, 0.0, atol=1e-5)
    assert jnp.allclose(tr2, 0.0, atol=1e-5)
    assert jnp.allclose(tr3, 0.0, atol=1e-5)

    # Hexadecapole: symmetric under swaps, single traces vanish
    hex_t = exp.hexadecapole
    assert jnp.allclose(hex_t, jnp.transpose(hex_t, (1, 0, 2, 3)), atol=1e-6)
    assert jnp.allclose(hex_t, jnp.transpose(hex_t, (0, 2, 1, 3)), atol=1e-6)
    assert jnp.allclose(hex_t, jnp.transpose(hex_t, (0, 1, 3, 2)), atol=1e-6)
    # single traces along different pairs
    tr_01 = jnp.trace(hex_t, axis1=0, axis2=1)  # (3,3)
    tr_02 = jnp.trace(hex_t, axis1=0, axis2=2)  # (3,3)
    tr_03 = jnp.trace(hex_t, axis1=0, axis2=3)  # (3,3)
    assert jnp.allclose(tr_01, 0.0, atol=1e-4)
    assert jnp.allclose(tr_02, 0.0, atol=1e-4)
    assert jnp.allclose(tr_03, 0.0, atol=1e-4)
    # double trace ~ 0
    dt = jnp.trace(tr_01)
    assert jnp.allclose(dt, 0.0, atol=1e-4)


def test_prepare_downward_sweep_matches_module_helper():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.9, -0.1, 0.05],
            [0.7, 0.0, -0.05],
            [0.9, -0.1, 0.1],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 0.8, 1.2, 0.9], dtype=jnp.float64)
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )

    fmm = FMMEngine(theta=0.4)
    upward = fmm.prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=2,
    )

    via_method = fmm.prepare_downward_sweep(tree, upward)
    via_module = prepare_local_downward_sweep(
        tree,
        upward,
        theta=0.4,
    )

    assert isinstance(via_method, TreeDownwardData)
    assert jnp.array_equal(
        via_method.interactions.offsets,
        via_module.interactions.offsets,
    )
    assert jnp.array_equal(
        via_method.interactions.sources,
        via_module.interactions.sources,
    )
    assert jnp.allclose(
        via_method.locals.coefficients,
        via_module.locals.coefficients,
    )
    assert jnp.allclose(via_method.locals.centers, via_module.locals.centers)

    theta_override = 0.3
    alt_method = fmm.prepare_downward_sweep(
        tree,
        upward,
        theta=theta_override,
    )
    alt_module = prepare_local_downward_sweep(
        tree,
        upward,
        theta=theta_override,
    )
    assert jnp.array_equal(
        alt_method.interactions.offsets,
        alt_module.interactions.offsets,
    )
    assert jnp.array_equal(
        alt_method.interactions.sources,
        alt_module.interactions.sources,
    )

    run_result = fmm.run_downward_sweep(
        tree,
        upward.multipoles,
        via_module.interactions,
    )
    module_result = run_local_downward_sweep(
        tree,
        upward.multipoles,
        via_module.interactions,
    )
    assert jnp.allclose(run_result.coefficients, module_result.coefficients)
    assert jnp.allclose(run_result.centers, module_result.centers)


def test_fmm_dense_downward_matches_sparse_path():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.4, -0.2, 0.25],
            [0.2, 0.3, -0.1],
            [0.6, -0.15, 0.15],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 0.7, 1.3, 0.9], dtype=jnp.float64)
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )

    kwargs = dict(bounds=bounds, leaf_size=2, max_order=2, theta=0.6)
    fmm_sparse = FMMEngine(theta=0.6, use_dense_interactions=False)
    fmm_dense = FMMEngine(theta=0.6, use_dense_interactions=True)

    state_sparse = fmm_sparse.prepare_state(positions, masses, **kwargs)
    state_dense = fmm_dense.prepare_state(positions, masses, **kwargs)

    acc_sparse = fmm_sparse.evaluate_prepared_state(state_sparse)
    acc_dense = fmm_dense.evaluate_prepared_state(state_dense)

    assert jnp.allclose(acc_sparse, acc_dense, rtol=1e-10, atol=1e-10)


def test_far_field_accuracy_order3_vs_order4():
    """Order 4 should be at least as accurate as order 3 in far field."""
    key = jax.random.PRNGKey(2)
    n = 30
    pos = 0.1 * jax.random.normal(key, (n, 3))
    mass = jax.random.uniform(key, (n,), minval=0.5, maxval=1.5)

    fmm = FMMEngine(G=1.0, softening=0.0)
    eval_point = jnp.array([6.0, -3.0, 2.0])

    a_ref = fmm.direct_sum(pos, mass, eval_point)

    exp3 = FMMEngine.compute_expansion(pos, mass, order=3)
    exp4 = FMMEngine.compute_expansion(pos, mass, order=4)

    a3 = fmm.evaluate_expansion(exp3, order=3, eval_point=eval_point)
    a4 = fmm.evaluate_expansion(exp4, order=4, eval_point=eval_point)

    e3 = jnp.linalg.norm(a3 - a_ref)
    e4 = jnp.linalg.norm(a4 - a_ref)

    assert e4 <= e3 + 1e-6


def test_evaluate_tree_matches_direct_sum_all_near_field():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.6, -0.2, 0.1],
            [0.2, 0.3, -0.2],
            [0.4, -0.1, 0.2],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 1.2, 0.9, 1.1], dtype=jnp.float64)
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )

    tree, pos_sorted, mass_sorted, inv = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=2,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=5.0)

    fmm = FMMEngine(theta=5.0, G=1.3, softening=0.05)
    upward = fmm.prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=2,
    )
    downward = fmm.prepare_downward_sweep(tree, upward, theta=5.0)

    accelerations, potentials = fmm.evaluate_tree(
        tree,
        pos_sorted,
        mass_sorted,
        downward,
        neighbor_list,
        return_potential=True,
    )

    inv_idx = np.asarray(inv)
    accel_orig = np.asarray(accelerations)[inv_idx]
    pot_orig = np.asarray(potentials)[inv_idx]

    direct_acc, direct_pot = _direct_sum(
        positions,
        masses,
        G=1.3,
        softening=0.05,
    )

    assert np.allclose(accel_orig, direct_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(pot_orig, direct_pot, rtol=1e-6, atol=1e-6)


def test_evaluate_tree_far_field_accuracy():
    positions = jnp.array(
        [
            [-0.9, -0.2, 0.1],
            [-0.7, 0.1, -0.1],
            [-0.6, 0.3, 0.05],
            [0.6, -0.1, -0.2],
            [0.75, 0.2, 0.0],
            [0.9, -0.05, 0.2],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 0.8, 1.1, 0.9, 1.2, 0.7], dtype=jnp.float64)
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )

    tree, pos_sorted, mass_sorted, inv = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=2,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbor_list = build_leaf_neighbor_lists(tree, geometry, theta=0.4)

    fmm = FMMEngine(theta=0.4, G=1.0, softening=0.01)
    upward = fmm.prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=2,
    )
    downward = fmm.prepare_downward_sweep(tree, upward, theta=0.4)

    max_leaf_size = int(
        np.max(
            np.asarray(
                tree.node_ranges[neighbor_list.leaf_indices, 1]
                - tree.node_ranges[neighbor_list.leaf_indices, 0]
                + 1,
                dtype=np.int64,
            )
        )
    )

    accelerations, potentials = fmm.evaluate_tree(
        tree,
        pos_sorted,
        mass_sorted,
        downward,
        neighbor_list,
        max_leaf_size=max_leaf_size,
        return_potential=True,
    )

    inv_idx = np.asarray(inv)
    accel_orig = np.asarray(accelerations)[inv_idx]
    pot_orig = np.asarray(potentials)[inv_idx]

    direct_acc, direct_pot = _direct_sum(
        positions,
        masses,
        G=1.0,
        softening=0.01,
    )

    assert np.allclose(accel_orig, direct_acc, rtol=5e-3, atol=5e-3)
    assert np.allclose(pot_orig, direct_pot, rtol=5e-3, atol=5e-3)


def test_fmm_pipeline_matches_direct_sum():
    positions = jnp.array(
        [
            [-0.8, 0.1, 0.0],
            [-0.6, -0.2, 0.1],
            [0.2, 0.3, -0.2],
            [0.4, -0.1, 0.2],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 1.2, 0.9, 1.1], dtype=jnp.float64)

    fmm = FMMEngine(theta=5.0, G=1.3, softening=0.05)
    acc_class, pot_class = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=2,
        return_potential=True,
    )

    acc_func, pot_func = compute_gravitational_acceleration(
        positions,
        masses,
        theta=5.0,
        G=1.3,
        softening=0.05,
        leaf_size=2,
        return_potential=True,
    )

    direct_acc, direct_pot = _direct_sum(
        np.asarray(positions),
        np.asarray(masses),
        G=1.3,
        softening=0.05,
    )

    acc_class_np = np.asarray(acc_class)
    pot_class_np = np.asarray(pot_class)
    acc_func_np = np.asarray(acc_func)
    pot_func_np = np.asarray(pot_func)

    assert np.allclose(acc_class_np, direct_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(pot_class_np, direct_pot, rtol=1e-6, atol=1e-6)
    assert np.allclose(acc_func_np, direct_acc, rtol=1e-6, atol=1e-6)
    assert np.allclose(pot_func_np, direct_pot, rtol=1e-6, atol=1e-6)


def test_evaluate_tree_compiled_matches_eager():
    positions = jnp.array(
        [
            [0.3, -0.2, 0.1],
            [-0.4, 0.5, -0.6],
            [0.7, 0.1, -0.3],
            [-0.2, -0.4, 0.8],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.1, 0.9, 1.3, 0.8], dtype=jnp.float64)

    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=2,
        return_reordered=True,
    )

    fmm = FMMEngine(theta=0.7, G=1.0, softening=0.02)
    upward = fmm.prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=2,
    )
    downward = fmm.prepare_downward_sweep(
        tree,
        upward,
        theta=fmm.theta,
    )
    neighbor_list = build_leaf_neighbor_lists(
        tree,
        upward.geometry,
        theta=fmm.theta,
    )

    kwargs = dict(max_leaf_size=2, return_potential=True)

    eager_acc, eager_pot = fmm.evaluate_tree(
        tree,
        pos_sorted,
        mass_sorted,
        downward,
        neighbor_list,
        **kwargs,
    )
    jit_acc, jit_pot = fmm.evaluate_tree_compiled(
        tree,
        pos_sorted,
        mass_sorted,
        downward,
        neighbor_list,
        **kwargs,
    )

    assert np.allclose(np.asarray(eager_acc), np.asarray(jit_acc))
    assert np.allclose(np.asarray(eager_pot), np.asarray(jit_pot))


@pytest.mark.parametrize(
    ("num_particles", "dtype", "chunk_size", "tolerance"),
    [
        # (96, float32, 128) is the configuration this test has always used.
        pytest.param(96, jnp.float32, 128, 1e-6, id="n96-f32-chunk128"),
        pytest.param(96, jnp.float64, 128, 1e-13, id="n96-f64-chunk128"),
        pytest.param(256, jnp.float32, 64, 1e-6, id="n256-f32-chunk64"),
        pytest.param(256, jnp.float64, 64, 1e-13, id="n256-f64-chunk64"),
    ],
)
def test_nearfield_bucketed_matches_baseline(
    num_particles, dtype, chunk_size, tolerance
):
    """``nearfield_mode`` "bucketed" and "baseline" agree to round-off.

    The two modes deliberately differ in edge order (``sort_by_source``), so they
    differ in float accumulation order and are **not** expected to be bit-equal --
    measured, they never are. What they must agree to is round-off, and this pins
    that at a tolerance derived from measurement rather than assumed.

    Measured relative L2 across 5 PRNG seeds x N in {96, 256}:

        float32   6.9e-08 .. 1.03e-07     (~1 eps_f32)
        float64   1.4e-16 .. 2.33e-16     (~1 eps_f64)

    Bounds are 1e-6 (fp32, ~8 eps) and 1e-13 (fp64), leaving roughly 10x and a
    wide margin respectively. Both are far tighter than the 1e-5 this test used
    before, which at fp32/N=96 was ~100x looser than the round-off it was meant to
    bound and would have admitted a real algorithmic divergence.

    The **float64 cases are the sharp instrument here**, and adding them matters
    more than tightening fp32. Mutation check: perturbing the bucketed path's
    softening by 3e-6 relative induces a ~1e-10 relative divergence in the force.
    The fp64 cases fail on it (1.3e-10 and 7.2e-10 against a 1e-13 bound); the fp32
    cases cannot see it at all, because fp32 round-off is already ~1e-7. So fp32
    can only ever catch a divergence above ~1e-6, and before this parametrisation
    there was no fp64 case at all.

    ``fixed_order`` stays at 3 rather than being parametrised: measured at orders
    2, 3 and 4 the near-field agreement is bit-for-bit unchanged, which is expected
    -- this exercises a near-field traversal, not the expansion. Adding that axis
    would cost compile time and measure nothing.

    The fp32/chunk-128 cases run in the CI smoke leg; the N=256 cases are listed in
    ``slow_tests.txt``. Before parametrisation the single case was slow-only, so no
    fast leg checked this equivalence at all.
    """
    key = jax.random.PRNGKey(515)
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=dtype,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=dtype)) + 1.0

    base_kwargs = dict(
        theta=0.6,
        softening=1e-3,
        working_dtype=dtype,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(
            rotation="solidfmm", grouped_interactions=True, mode="class_major"
        ),
        mac_type="dehnen",
        fixed_order=3,
        fixed_max_leaf_size=16,
    )

    fmm_baseline = FMMEngine(
        nearfield=NearFieldConfig(mode="baseline"),
        **base_kwargs,
    )
    fmm_bucketed = FMMEngine(
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=chunk_size),
        **base_kwargs,
    )

    acc_baseline = fmm_baseline.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
        jit_tree=False,
    )
    acc_bucketed = fmm_bucketed.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
        jit_tree=False,
    )

    baseline_np = np.asarray(acc_baseline, dtype=np.float64)
    bucketed_np = np.asarray(acc_bucketed, dtype=np.float64)
    rel_l2 = float(
        np.linalg.norm(bucketed_np - baseline_np)
        / max(float(np.linalg.norm(baseline_np)), 1e-300)
    )
    assert rel_l2 < tolerance, (
        f"bucketed vs baseline near-field disagree beyond round-off: rel-L2 "
        f"{rel_l2:.3e} > {tolerance:.0e} (N={num_particles}, {np.dtype(dtype).name}, "
        f"chunk={chunk_size})"
    )


def test_nearfield_precomputed_leaf_pairs_matches_inline_mapping():
    key = jax.random.PRNGKey(516)
    num_particles = 96
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbors = build_leaf_neighbor_lists(tree, geometry, theta=0.6)
    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    leaf_nodes = jnp.asarray(neighbors.leaf_indices, dtype=INDEX_DTYPE)
    offsets = jnp.asarray(neighbors.offsets, dtype=INDEX_DTYPE)
    neighbor_ids = jnp.asarray(neighbors.neighbors, dtype=INDEX_DTYPE)
    tgt_leaf, src_leaf, valid = prepare_leaf_neighbor_pairs(
        node_ranges,
        leaf_nodes,
        offsets,
        neighbor_ids,
    )

    acc_inline = compute_leaf_p2p_accelerations(
        tree,
        neighbors,
        pos_sorted,
        mass_sorted,
        softening=1e-3,
        max_leaf_size=16,
        nearfield_mode="bucketed",
        edge_chunk_size=64,
    )
    acc_precomputed = compute_leaf_p2p_accelerations(
        tree,
        neighbors,
        pos_sorted,
        mass_sorted,
        softening=1e-3,
        max_leaf_size=16,
        nearfield_mode="bucketed",
        edge_chunk_size=64,
        precomputed_target_leaf_ids=tgt_leaf,
        precomputed_source_leaf_ids=src_leaf,
        precomputed_valid_pairs=valid,
    )

    assert np.allclose(
        np.asarray(acc_precomputed), np.asarray(acc_inline), rtol=1e-5, atol=1e-5
    )


def test_nearfield_precomputed_bucketed_scatter_matches_inline():
    key = jax.random.PRNGKey(615)
    num_particles = 96
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=16,
        return_reordered=True,
    )
    geometry = compute_tree_geometry(tree, pos_sorted)
    neighbors = build_leaf_neighbor_lists(tree, geometry, theta=0.6)
    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    leaf_nodes = jnp.asarray(neighbors.leaf_indices, dtype=INDEX_DTYPE)
    offsets = jnp.asarray(neighbors.offsets, dtype=INDEX_DTYPE)
    neighbor_ids = jnp.asarray(neighbors.neighbors, dtype=INDEX_DTYPE)
    tgt_leaf, src_leaf, valid = prepare_leaf_neighbor_pairs(
        node_ranges,
        leaf_nodes,
        offsets,
        neighbor_ids,
    )
    sort_idx, group_ids, unique_indices = prepare_bucketed_scatter_schedules(
        node_ranges,
        leaf_nodes,
        tgt_leaf,
        valid,
        max_leaf_size=16,
        edge_chunk_size=64,
    )

    acc_inline = compute_leaf_p2p_accelerations(
        tree,
        neighbors,
        pos_sorted,
        mass_sorted,
        softening=1e-3,
        max_leaf_size=16,
        nearfield_mode="bucketed",
        edge_chunk_size=64,
        precomputed_target_leaf_ids=tgt_leaf,
        precomputed_source_leaf_ids=src_leaf,
        precomputed_valid_pairs=valid,
    )
    acc_precomputed = compute_leaf_p2p_accelerations(
        tree,
        neighbors,
        pos_sorted,
        mass_sorted,
        softening=1e-3,
        max_leaf_size=16,
        nearfield_mode="bucketed",
        edge_chunk_size=64,
        precomputed_target_leaf_ids=tgt_leaf,
        precomputed_source_leaf_ids=src_leaf,
        precomputed_valid_pairs=valid,
        precomputed_chunk_sort_indices=sort_idx,
        precomputed_chunk_group_ids=group_ids,
        precomputed_chunk_unique_indices=unique_indices,
    )

    assert np.allclose(
        np.asarray(acc_precomputed), np.asarray(acc_inline), rtol=1e-5, atol=1e-5
    )


def test_radix_fast_lane_prepared_state_matches_large_n_baseline(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "8")
    monkeypatch.setenv("JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT", "1")

    key = jax.random.PRNGKey(1234)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (1280, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (1280,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )

    kwargs = dict(
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        farfield=FarFieldConfig(grouped_interactions=False),
        working_dtype=jnp.float32,
    )

    fmm = FMMEngine(**kwargs)
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=256,
        max_order=4,
    )
    assert bool(getattr(state, "radix_fast_lane", False))
    assert getattr(state, "radix_fast_payload", None) is not None

    payload = state.radix_fast_payload
    assert payload.target_particle_ids.ndim == 2
    assert payload.source_particle_ids.ndim == 3
    assert payload.target_particle_mask.shape == payload.target_particle_ids.shape
    assert payload.source_particle_mask.shape == payload.source_particle_ids.shape

    fast_acc = np.asarray(fmm.evaluate_prepared_state(state))
    baseline_acc, _ = fmm.evaluate_prepared_state(state, return_potential=True)
    baseline_acc = np.asarray(baseline_acc)
    abs_err = np.abs(fast_acc - baseline_acc)
    max_abs_err = float(abs_err.max(initial=0.0))
    denom = np.maximum(np.abs(baseline_acc), 1e-8)
    max_rel_err = float((abs_err / denom).max(initial=0.0))
    assert np.allclose(
        fast_acc,
        baseline_acc,
        rtol=5e-4,
        atol=5e-4,
    ), (
        "radix fast-lane acceleration drift exceeded tolerance: "
        f"max_abs_err={max_abs_err:.6e}, max_rel_err={max_rel_err:.6e}"
    )


def test_radix_fast_lane_includes_overflow_target_blocks(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_SPEED_PREPARED_AUTO_FULL_BLOCKS", "0")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS", "0")

    key = jax.random.PRNGKey(20260506)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (1280, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (1280,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )

    fmm = FMMEngine(
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        farfield=FarFieldConfig(grouped_interactions=False),
        working_dtype=jnp.float32,
    )
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=256,
        max_order=4,
    )

    assert bool(getattr(state, "radix_fast_lane", False))
    assert state.nearfield_target_block_source_leaf_ids is not None
    assert int(state.nearfield_target_block_source_leaf_ids.shape[0]) > 0
    assert getattr(state, "radix_overflow_payload", None) is not None
    assert int(state.radix_overflow_payload.source_particle_ids.size) > 0

    fast_acc = np.asarray(fmm.evaluate_prepared_state(state))
    baseline_acc, _ = fmm.evaluate_prepared_state(state, return_potential=True)
    baseline_acc = np.asarray(baseline_acc)
    abs_err = np.abs(fast_acc - baseline_acc)
    max_abs_err = float(abs_err.max(initial=0.0))
    denom = np.maximum(np.abs(baseline_acc), 1e-8)
    max_rel_err = float((abs_err / denom).max(initial=0.0))
    assert np.allclose(
        fast_acc,
        baseline_acc,
        rtol=5e-4,
        atol=5e-4,
    ), (
        "radix fast-lane overflow acceleration drift exceeded tolerance: "
        f"max_abs_err={max_abs_err:.6e}, max_rel_err={max_rel_err:.6e}"
    )


def test_radix_fast_lane_auto_full_prefix_eliminates_overflow(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS", "0")

    key = jax.random.PRNGKey(20260507)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (1280, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (1280,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )

    fmm = FMMEngine(
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        farfield=FarFieldConfig(grouped_interactions=False),
        working_dtype=jnp.float32,
    )
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=256,
        max_order=4,
    )

    assert bool(getattr(state, "radix_fast_lane", False))
    assert state.nearfield_target_block_source_leaf_ids_padded is not None
    assert int(state.nearfield_target_block_source_leaf_ids_padded.shape[1]) > 1
    assert int(state.nearfield_target_block_overflow_active_blocks) == 0
    assert getattr(state, "radix_overflow_payload", None) is None


def test_large_n_prepacked_overflow_fallback_matches_tiled_overflow(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_SPEED_PREPARED_AUTO_FULL_BLOCKS", "0")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS", "0")

    key = jax.random.PRNGKey(20260417)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (1280, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (1280,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )

    fmm = FMMEngine(
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        farfield=FarFieldConfig(grouped_interactions=False),
        working_dtype=jnp.float32,
    )
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=256,
        max_order=4,
    )

    assert state.nearfield_target_block_source_leaf_ids_padded is not None
    assert state.nearfield_target_block_valid_mask_padded is not None
    overflow_blocks = int(state.nearfield_target_block_source_leaf_ids.shape[0])
    assert overflow_blocks > 1

    tiled_overflow_acc = np.asarray(
        compute_leaf_p2p_accelerations_large_n_accel_only(
            state.tree,
            state.neighbor_list,
            state.positions_sorted,
            state.masses_sorted,
            G=float(getattr(fmm, "G")),
            softening=float(getattr(fmm, "softening")),
            edge_chunk_size=int(state.nearfield_edge_chunk_size),
            precomputed_target_leaf_ids=state.nearfield_target_leaf_ids,
            precomputed_source_leaf_ids=state.nearfield_source_leaf_ids,
            precomputed_valid_pairs=state.nearfield_valid_pairs,
            leaf_particle_indices=state.nearfield_leaf_particle_indices,
            leaf_particle_mask=state.nearfield_leaf_particle_mask,
            precomputed_target_block_leaf_ids=state.nearfield_target_block_leaf_ids,
            precomputed_target_block_source_leaf_ids=state.nearfield_target_block_source_leaf_ids,
            precomputed_target_block_valid_mask=state.nearfield_target_block_valid_mask,
            precomputed_target_block_offsets=state.nearfield_target_block_offsets,
            precomputed_target_block_source_leaf_ids_padded=(
                state.nearfield_target_block_source_leaf_ids_padded
            ),
            precomputed_target_block_valid_mask_padded=(
                state.nearfield_target_block_valid_mask_padded
            ),
            delayed_scatter_chunks_per_superchunk=int(
                state.nearfield_delayed_scatter_chunks_per_superchunk
            ),
            chunk_scan_batch_size=int(state.nearfield_chunk_scan_batch_size),
            chunk_scan_unroll=int(state.nearfield_chunk_scan_unroll),
            superchunk_scan_unroll=int(state.nearfield_superchunk_scan_unroll),
            sorted_scatter_hint=bool(state.nearfield_sorted_scatter_hint),
            grouped_sorted_scatter=bool(state.nearfield_grouped_sorted_scatter),
            superchunk_target_reduce=bool(state.nearfield_superchunk_target_reduce),
            disable_chunk_cond=bool(state.nearfield_disable_chunk_cond),
            target_leaf_batch_size=int(state.nearfield_target_leaf_batch_size),
            target_block_tile_size=int(state.nearfield_target_block_tile_size),
            target_block_tile_scan_unroll=int(
                state.nearfield_target_block_tile_scan_unroll
            ),
            target_block_batch_scan_unroll=int(
                state.nearfield_target_block_batch_scan_unroll
            ),
            target_block_overflow_fast_max_blocks=131072,
        )
    )
    fallback_overflow_acc = np.asarray(
        compute_leaf_p2p_accelerations_large_n_accel_only(
            state.tree,
            state.neighbor_list,
            state.positions_sorted,
            state.masses_sorted,
            G=float(getattr(fmm, "G")),
            softening=float(getattr(fmm, "softening")),
            edge_chunk_size=int(state.nearfield_edge_chunk_size),
            precomputed_target_leaf_ids=state.nearfield_target_leaf_ids,
            precomputed_source_leaf_ids=state.nearfield_source_leaf_ids,
            precomputed_valid_pairs=state.nearfield_valid_pairs,
            leaf_particle_indices=state.nearfield_leaf_particle_indices,
            leaf_particle_mask=state.nearfield_leaf_particle_mask,
            precomputed_target_block_leaf_ids=state.nearfield_target_block_leaf_ids,
            precomputed_target_block_source_leaf_ids=state.nearfield_target_block_source_leaf_ids,
            precomputed_target_block_valid_mask=state.nearfield_target_block_valid_mask,
            precomputed_target_block_offsets=state.nearfield_target_block_offsets,
            precomputed_target_block_source_leaf_ids_padded=(
                state.nearfield_target_block_source_leaf_ids_padded
            ),
            precomputed_target_block_valid_mask_padded=(
                state.nearfield_target_block_valid_mask_padded
            ),
            delayed_scatter_chunks_per_superchunk=int(
                state.nearfield_delayed_scatter_chunks_per_superchunk
            ),
            chunk_scan_batch_size=int(state.nearfield_chunk_scan_batch_size),
            chunk_scan_unroll=int(state.nearfield_chunk_scan_unroll),
            superchunk_scan_unroll=int(state.nearfield_superchunk_scan_unroll),
            sorted_scatter_hint=bool(state.nearfield_sorted_scatter_hint),
            grouped_sorted_scatter=bool(state.nearfield_grouped_sorted_scatter),
            superchunk_target_reduce=bool(state.nearfield_superchunk_target_reduce),
            disable_chunk_cond=bool(state.nearfield_disable_chunk_cond),
            target_leaf_batch_size=int(state.nearfield_target_leaf_batch_size),
            target_block_tile_size=int(state.nearfield_target_block_tile_size),
            target_block_tile_scan_unroll=int(
                state.nearfield_target_block_tile_scan_unroll
            ),
            target_block_batch_scan_unroll=int(
                state.nearfield_target_block_batch_scan_unroll
            ),
            target_block_overflow_fast_max_blocks=1,
        )
    )

    abs_err = np.abs(fallback_overflow_acc - tiled_overflow_acc)
    max_abs_err = float(abs_err.max(initial=0.0))
    denom = np.maximum(np.abs(tiled_overflow_acc), 1e-8)
    max_rel_err = float((abs_err / denom).max(initial=0.0))
    assert np.allclose(
        fallback_overflow_acc,
        tiled_overflow_acc,
        rtol=2e-4,
        atol=3e-4,
    ), (
        "overflow fallback drift exceeded tolerance: "
        f"max_abs_err={max_abs_err:.6e}, max_rel_err={max_rel_err:.6e}, "
        f"overflow_blocks={overflow_blocks}"
    )


def test_radix_fast_lane_fixed_seed_repeatability(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "8")
    monkeypatch.setenv("JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT", "1")

    seed = 20260417
    key_a = jax.random.PRNGKey(seed)
    key_a_pos, key_a_mass = jax.random.split(key_a)
    positions_a = jax.random.uniform(
        key_a_pos,
        (1280, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses_a = jax.random.uniform(
        key_a_mass,
        (1280,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )

    key_b = jax.random.PRNGKey(seed)
    key_b_pos, key_b_mass = jax.random.split(key_b)
    positions_b = jax.random.uniform(
        key_b_pos,
        (1280, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses_b = jax.random.uniform(
        key_b_mass,
        (1280,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )
    assert np.array_equal(np.asarray(positions_a), np.asarray(positions_b))
    assert np.array_equal(np.asarray(masses_a), np.asarray(masses_b))

    kwargs = dict(
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        farfield=FarFieldConfig(grouped_interactions=False),
        working_dtype=jnp.float32,
    )
    fmm_a = FMMEngine(**kwargs)
    fmm_b = FMMEngine(**kwargs)

    state_a = fmm_a.prepare_state(
        positions_a,
        masses_a,
        leaf_size=256,
        max_order=4,
    )
    state_b = fmm_b.prepare_state(
        positions_b,
        masses_b,
        leaf_size=256,
        max_order=4,
    )
    assert bool(getattr(state_a, "radix_fast_lane", False))
    assert bool(getattr(state_b, "radix_fast_lane", False))

    payload_a = state_a.radix_fast_payload
    payload_b = state_b.radix_fast_payload
    assert payload_a is not None
    assert payload_b is not None
    assert np.array_equal(
        np.asarray(payload_a.target_particle_ids),
        np.asarray(payload_b.target_particle_ids),
    )
    assert np.array_equal(
        np.asarray(payload_a.target_particle_mask),
        np.asarray(payload_b.target_particle_mask),
    )
    assert np.array_equal(
        np.asarray(payload_a.source_particle_ids),
        np.asarray(payload_b.source_particle_ids),
    )
    assert np.array_equal(
        np.asarray(payload_a.source_particle_mask),
        np.asarray(payload_b.source_particle_mask),
    )

    acc_a_0 = np.asarray(fmm_a.evaluate_prepared_state(state_a))
    acc_b_0 = np.asarray(fmm_b.evaluate_prepared_state(state_b))
    abs_err = np.abs(acc_b_0 - acc_a_0)
    max_abs_err = float(abs_err.max(initial=0.0))
    denom = np.maximum(np.abs(acc_a_0), 1e-8)
    max_rel_err = float((abs_err / denom).max(initial=0.0))
    assert np.allclose(
        acc_b_0,
        acc_a_0,
        rtol=5e-6,
        atol=5e-6,
    ), (
        "fixed-seed repeatability drift exceeded tolerance: "
        f"max_abs_err={max_abs_err:.6e}, max_rel_err={max_rel_err:.6e}"
    )


def test_prepare_state_reuses_cached_interactions_when_inputs_match():
    key = jax.random.PRNGKey(123)
    num_particles = 32
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
    )
    masses = jnp.linspace(0.5, 1.5, num_particles, dtype=jnp.float32)

    fmm = FMMEngine(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
    )
    state_first = fmm.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=2,
        jit_tree=False,
    )

    with mock.patch.object(
        fmm_module,
        "build_interactions_and_neighbors",
        side_effect=AssertionError("should not rebuild interactions"),
    ):
        state_second = fmm.prepare_state(
            positions,
            masses,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
        )

    assert state_first.interactions.sources is state_second.interactions.sources
    assert jnp.array_equal(
        state_first.neighbor_list.neighbors,
        state_second.neighbor_list.neighbors,
    )


def test_compute_accelerations_reuses_prepared_state_when_enabled():
    key = jax.random.PRNGKey(211)
    num_particles = 32
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.linspace(0.75, 1.25, num_particles, dtype=jnp.float32)

    fmm = FMMEngine(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
    )
    with mock.patch.object(
        fmm, "prepare_state", wraps=fmm.prepare_state
    ) as spy_prepare:
        acc_first = fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
            reuse_prepared_state=True,
        )
        acc_second = fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
            reuse_prepared_state=True,
        )

    assert spy_prepare.call_count == 1
    assert np.allclose(
        np.asarray(acc_second), np.asarray(acc_first), rtol=1e-6, atol=1e-6
    )


def test_compute_accelerations_reuse_cache_invalidates_on_parameter_and_value_change():
    key = jax.random.PRNGKey(311)
    num_particles = 28
    positions = jax.random.normal(key, (num_particles, 3), dtype=jnp.float32)
    masses = jnp.ones((num_particles,), dtype=jnp.float32)
    masses_changed = masses.at[0].set(jnp.float32(1.5))

    fmm = FMMEngine(
        theta=0.55,
        softening=1e-3,
        working_dtype=jnp.float32,
    )
    with mock.patch.object(
        fmm, "prepare_state", wraps=fmm.prepare_state
    ) as spy_prepare:
        fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
            reuse_prepared_state=True,
        )
        fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=8,
            max_order=3,
            jit_tree=False,
            reuse_prepared_state=True,
        )
        fmm.compute_accelerations(
            positions,
            masses_changed,
            leaf_size=8,
            max_order=3,
            jit_tree=False,
            reuse_prepared_state=True,
        )

    assert spy_prepare.call_count == 3


def test_compute_accelerations_reuses_prepared_state_for_value_equal_copies():
    key = jax.random.PRNGKey(312)
    num_particles = 28
    positions = jax.random.normal(key, (num_particles, 3), dtype=jnp.float32)
    masses = jnp.ones((num_particles,), dtype=jnp.float32)
    positions_copy = jnp.array(np.asarray(positions))
    masses_copy = jnp.array(np.asarray(masses))

    fmm = FMMEngine(
        theta=0.55,
        softening=1e-3,
        working_dtype=jnp.float32,
    )
    with mock.patch.object(
        fmm, "prepare_state", wraps=fmm.prepare_state
    ) as spy_prepare:
        acc_first = fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
            reuse_prepared_state=True,
        )
        acc_second = fmm.compute_accelerations(
            positions_copy,
            masses_copy,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
            reuse_prepared_state=True,
        )

    assert spy_prepare.call_count == 1
    assert np.allclose(
        np.asarray(acc_second), np.asarray(acc_first), rtol=1e-6, atol=1e-6
    )


def test_prepare_state_precomputes_bucketed_scatter_schedule():
    key = jax.random.PRNGKey(911)
    num_particles = 128
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.ones((num_particles,), dtype=jnp.float32)

    fmm = FMMEngine(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
    )
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=2,
        jit_tree=False,
    )
    assert state.nearfield_chunk_sort_indices is not None
    assert state.nearfield_chunk_group_ids is not None
    assert state.nearfield_chunk_unique_indices is not None

    acc_state = fmm.evaluate_prepared_state(state, jit_traversal=True)
    acc_full = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=2,
        jit_tree=False,
    )
    assert np.allclose(
        np.asarray(acc_state), np.asarray(acc_full), rtol=1e-5, atol=1e-5
    )


def test_prepare_state_cache_respects_theta_changes():
    key = jax.random.PRNGKey(321)
    num_particles = 24
    positions = jax.random.normal(key, (num_particles, 3), dtype=jnp.float32)
    masses = jnp.ones((num_particles,), dtype=jnp.float32)

    fmm = FMMEngine(
        theta=0.5,
        softening=1e-3,
        working_dtype=jnp.float32,
    )
    fmm.prepare_state(
        positions,
        masses,
        leaf_size=6,
        max_order=2,
        jit_tree=False,
    )

    theta_override = 0.65
    with mock.patch.object(
        fmm_module,
        "build_interactions_and_neighbors",
        wraps=fmm_module.build_interactions_and_neighbors,
    ) as spy_build:
        fmm.prepare_state(
            positions,
            masses,
            leaf_size=6,
            max_order=2,
            theta=theta_override,
            jit_tree=False,
        )

    assert spy_build.call_count == 1


def test_prepare_state_cache_respects_dehnen_radius_scale_changes():
    key = jax.random.PRNGKey(411)
    num_particles = 32
    positions = jax.random.normal(key, (num_particles, 3), dtype=jnp.float32)
    masses = jnp.ones((num_particles,), dtype=jnp.float32)

    fmm = FMMEngine(
        theta=0.55,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        mac_type="dehnen",
        dehnen_radius_scale=1.0,
    )
    fmm.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
        jit_tree=False,
    )

    fmm.dehnen_radius_scale = 0.9
    with mock.patch.object(
        fmm_module,
        "build_interactions_and_neighbors",
        wraps=fmm_module.build_interactions_and_neighbors,
    ) as spy_build:
        fmm.prepare_state(
            positions,
            masses,
            leaf_size=8,
            max_order=3,
            jit_tree=False,
        )

    assert spy_build.call_count == 1
    assert float(spy_build.call_args.kwargs["dehnen_radius_scale"]) == pytest.approx(
        0.9
    )


def test_prepare_state_cache_respects_traversal_config_changes():
    key = jax.random.PRNGKey(654)
    num_particles = 28
    positions = jax.random.normal(key, (num_particles, 3), dtype=jnp.float32)
    masses = jnp.linspace(0.25, 1.0, num_particles, dtype=jnp.float32)

    config_a = DualTreeTraversalConfig(
        max_pair_queue=4096,
        process_block=256,
        max_interactions_per_node=4096,
    )
    config_b = DualTreeTraversalConfig(
        max_pair_queue=8192,
        process_block=512,
        max_interactions_per_node=8192,
        max_neighbors_per_leaf=4096,
    )

    fmm = FMMEngine(
        theta=0.55,
        softening=5e-4,
        working_dtype=jnp.float32,
        runtime_policy=RuntimePolicyConfig(traversal_config=config_a),
    )
    fmm.prepare_state(
        positions,
        masses,
        leaf_size=6,
        max_order=2,
        jit_tree=False,
    )

    fmm.traversal_config = config_b
    with mock.patch.object(
        fmm_module,
        "build_interactions_and_neighbors",
        wraps=fmm_module.build_interactions_and_neighbors,
    ) as spy_build:
        fmm.prepare_state(
            positions,
            masses,
            leaf_size=6,
            max_order=2,
            jit_tree=False,
        )

    assert spy_build.call_count == 1


def test_prepare_state_reuses_topology_when_morton_order_stable():
    key = jax.random.PRNGKey(777)
    num_particles = 48
    base_positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-0.4,
        maxval=0.4,
        dtype=jnp.float32,
    )
    moved_positions = jnp.array(np.asarray(base_positions))
    masses = jnp.linspace(0.5, 1.5, num_particles, dtype=jnp.float32)
    bounds = (
        jnp.asarray([-1.0, -1.0, -1.0], dtype=jnp.float32),
        jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32),
    )

    fmm = FMMEngine(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        reuse_topology=True,
        rebuild_every=3,
    )
    state_first = fmm.prepare_state(
        base_positions,
        masses,
        bounds=bounds,
        leaf_size=8,
        max_order=2,
        jit_tree=False,
    )

    with mock.patch.object(
        fmm_impl_private,
        "_build_tree_with_config",
        side_effect=AssertionError("should reuse cached topology"),
    ):
        state_second = fmm.prepare_state(
            moved_positions,
            masses,
            bounds=bounds,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
        )

    assert state_first.topology_key == state_second.topology_key
    assert fmm.recent_topology_reused is True


def test_prepare_state_rebuilds_topology_after_rebuild_every_steps():
    key = jax.random.PRNGKey(778)
    num_particles = 40
    base_positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-0.35,
        maxval=0.35,
        dtype=jnp.float32,
    )
    moved_a = jnp.array(np.asarray(base_positions))
    moved_b = jnp.array(np.asarray(base_positions))
    masses = jnp.ones((num_particles,), dtype=jnp.float32)
    bounds = (
        jnp.asarray([-1.0, -1.0, -1.0], dtype=jnp.float32),
        jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32),
    )

    fmm = FMMEngine(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        reuse_topology=True,
        rebuild_every=2,
    )
    state_first = fmm.prepare_state(
        base_positions,
        masses,
        bounds=bounds,
        leaf_size=8,
        max_order=2,
        jit_tree=False,
    )
    state_second = fmm.prepare_state(
        moved_a,
        masses,
        bounds=bounds,
        leaf_size=8,
        max_order=2,
        jit_tree=False,
    )
    assert state_first.topology_key == state_second.topology_key
    assert fmm.recent_topology_reused is True

    with mock.patch.object(
        fmm_prepare_private,
        "_build_tree_with_config",
        wraps=fmm_prepare_private._build_tree_with_config,
    ) as spy_build:
        state_third = fmm.prepare_state(
            moved_b,
            masses,
            bounds=bounds,
            leaf_size=8,
            max_order=2,
            jit_tree=False,
        )

    assert state_third.topology_key == state_second.topology_key
    assert spy_build.call_count == 1
    assert fmm.recent_topology_reused is False


def test_prepare_state_reuses_grouped_buffers_from_cache():
    key = jax.random.PRNGKey(808)
    num_particles = 64
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    fmm = FMMEngine(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm", grouped_interactions=True),
        mac_type="dehnen",
    )
    fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
        jit_tree=False,
    )

    with mock.patch.object(
        tree_interactions_module,
        "build_grouped_interactions_from_pairs",
        side_effect=AssertionError("should reuse grouped buffers from cache"),
    ):
        fmm.prepare_state(
            positions,
            masses,
            leaf_size=16,
            max_order=4,
            jit_tree=False,
        )


def test_prepare_state_reuses_grouped_class_segments_from_cache():
    key = jax.random.PRNGKey(810)
    num_particles = 96
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    fmm = FMMEngine(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(
            rotation="solidfmm",
            grouped_interactions=True,
            mode="class_major",
            m2l_chunk_size=128,
        ),
        mac_type="dehnen",
    )
    fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
        jit_tree=False,
    )

    with mock.patch.object(
        fmm_module,
        "_build_grouped_class_segments",
        side_effect=AssertionError("should reuse grouped class segments from cache"),
    ):
        fmm.prepare_state(
            positions,
            masses,
            leaf_size=16,
            max_order=4,
            jit_tree=False,
        )


def test_prepare_state_cache_key_respects_center_mode():
    key = jax.random.PRNGKey(809)
    num_particles = 64
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    fmm = FMMEngine(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm", grouped_interactions=False),
        mac_type="dehnen",
    )
    # The cache-key-vs-center_mode behaviour depends on the adaptive resolver
    # setting center_mode="aabb" when grouped_interactions flips on. That is an
    # adaptive rewrite which the production-default static fixed sizing skips, so
    # disable it here to exercise the center_mode-sensitive cache-key path.
    fmm._static_runtime_fixed_sizing = False
    fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
        jit_tree=False,
    )

    fmm.grouped_interactions = True
    fmm._explicit_grouped_interactions = True
    with mock.patch.object(
        fmm_module,
        "build_interactions_and_neighbors",
        wraps=fmm_module.build_interactions_and_neighbors,
    ) as spy_build:
        fmm.prepare_state(
            positions,
            masses,
            leaf_size=16,
            max_order=4,
            jit_tree=False,
        )

    assert spy_build.call_count == 1


def test_fast_preset_sets_lbvh_defaults():
    key = jax.random.PRNGKey(111)
    num_particles = 48
    positions = jax.random.normal(key, (num_particles, 3), dtype=jnp.float32)
    masses = jnp.ones((num_particles,), dtype=jnp.float32)

    fmm = FMMEngine(preset="fast", theta=0.6, softening=1e-3)

    assert fmm.tree_build_mode == "lbvh"
    assert fmm.target_leaf_particles == 64
    assert fmm.refine_local is False
    assert isinstance(fmm.traversal_config, DualTreeTraversalConfig)
    assert fmm.m2l_chunk_size == 512

    accelerations = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=8,
        max_order=2,
    )

    assert accelerations.shape == (num_particles, 3)


def test_fast_preset_allows_explicit_overrides():
    fmm = FMMEngine(
        preset=FMMPreset.FAST,
        tree=TreeConfig(mode="lbvh", leaf_target=12, refine_local=True),
    )

    assert fmm.tree_build_mode == "lbvh"
    assert fmm.target_leaf_particles == 12
    assert fmm.refine_local is True
    assert isinstance(fmm.traversal_config, DualTreeTraversalConfig)


def test_fast_preset_defaults_to_auto_jit_tree_policy():
    fmm = FMMEngine(preset=FMMPreset.FAST)
    assert fmm._jit_tree_default == "auto"


def test_solidfmm_float32_uses_complex64_locals():
    key = jax.random.PRNGKey(7)
    num_particles = 64
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    fmm = FMMEngine(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        mac_type="dehnen",
    )
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
        jit_tree=False,
    )
    assert state.downward.locals.coefficients.dtype == jnp.complex64


def test_solidfmm_float64_uses_complex128_locals():
    with jax.enable_x64(True):
        key = jax.random.PRNGKey(9)
        num_particles = 64
        positions = jax.random.uniform(
            key,
            (num_particles, 3),
            minval=-1.0,
            maxval=1.0,
            dtype=jnp.float64,
        )
        masses = (
            jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float64)) + 1.0
        )

        fmm = FMMEngine(
            theta=0.6,
            softening=1e-3,
            working_dtype=jnp.float64,
            expansion_basis="solidfmm",
            farfield=FarFieldConfig(rotation="solidfmm"),
            mac_type="dehnen",
        )
        state = fmm.prepare_state(
            positions,
            masses,
            leaf_size=16,
            max_order=4,
            jit_tree=False,
        )
        assert state.downward.locals.coefficients.dtype == jnp.complex128


def _m2l_far_pair_count(state) -> int:
    """M2L interactions in a prepared topology; 0 means the far field never runs."""
    interactions = getattr(state, "interactions", None)
    if interactions is not None:
        return int(jnp.sum(interactions.counts))
    dual = getattr(state, "dual_tree_result", None)
    if dual is not None:
        return int(jnp.sum(dual.far_pair_count))
    return 0


# leaf_size=4 is load-bearing: see the docstring's first paragraph.
_CHUNKED_M2L_LEAF_SIZE = 4


def test_solidfmm_chunked_m2l_matches_fullbatch():
    """Chunking the M2L reduction must not change the force.

    Two things about this test were previously not true of it.

    First, it ran at ``leaf_size=16``, where this system accepts **zero** M2L
    pairs -- so ``m2l_chunk_size=4096`` and ``m2l_chunk_size=32`` had nothing to
    chunk and could not possibly differ. Measured far-pair counts at N=128,
    ``theta=0.6``, ``mac_type="dehnen"``::

        leaf_size    16      8      4      2
        far_pairs     0      0    158   1412

    The two fp32 outputs were bit-identical (difference exactly 0.0) because the
    chunked M2L never executed, not because chunking is correct. This is the same
    vacuity failure `NUMERICS_AND_JAX.md` section 3 describes for gradient tests.
    ``leaf_size=4`` is now used and the pair count is asserted, so the test cannot
    silently return to testing nothing.

    Second, it compared the two **fp32** paths to each other. Two fp32 paths
    agreeing proves nothing -- they can be identically wrong, which is exactly why
    the neighbouring ``test_solidfmm_m2l_ignores_padded_compact_far_pairs`` was
    rewritten to use a float64 reference. Each path is now checked against a
    float64 reference built from the *same* particles cast up. (Re-drawing the
    particles in float64 instead would silently compare a different physical
    system: ``jax.random.uniform`` with the same key returns different values per
    dtype, measured rel-L2 1.2-1.4, which looks like catastrophic error.)

    Bounds, measured over 4 seeds on both backends (rel-L2 against the float64
    reference, and the two fp32 paths against each other)::

                        vs float64 ref     chunked vs fullbatch
        CPU             1.06e-07 .. 1.79e-07     0.0 .. 1.29e-08
        A100 (sm_80)    1.30e-07 .. 2.27e-07   3.80e-08 .. 5.89e-08

    So ~1-2 eps_float32 against truth, and the two paths differ at round-off
    because chunk boundaries reorder the M2L sum -- exact agreement is not the
    claim and never was achievable once the far field actually runs. The bounds
    below carry ~9x headroom over the worst measurement. They are not slack: a
    dropped or double-counted chunk moves the result by order 1, five orders
    above these.

    The direct fp32-vs-fp32 comparison is kept as a secondary check, bounded by
    reordering round-off rather than by the old elementwise
    ``allclose(rtol=1e-6, atol=1e-6)``. That form was a knife edge on GPU: it is
    elementwise, so near-zero acceleration components made ``atol`` the binding
    term, and the test failed ~1 run in 3 on an A100 from reduction
    nondeterminism alone (`ARCHITECTURE.md` section 10).
    """
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("float64 reference needs JAX_ENABLE_X64=1")

    key = jax.random.PRNGKey(13)
    num_particles = 128
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    def solver(*, chunk_size, dtype):
        return FMMEngine(
            theta=0.6,
            softening=1e-3,
            working_dtype=dtype,
            expansion_basis="solidfmm",
            farfield=FarFieldConfig(rotation="solidfmm", m2l_chunk_size=chunk_size),
            mac_type="dehnen",
            tree=TreeConfig(mode="lbvh"),
            fixed_order=4,
            fixed_max_leaf_size=_CHUNKED_M2L_LEAF_SIZE,
        )

    def accelerations(*, chunk_size, dtype):
        return np.asarray(
            solver(chunk_size=chunk_size, dtype=dtype).compute_accelerations(
                positions.astype(dtype),
                masses.astype(dtype),
                leaf_size=_CHUNKED_M2L_LEAF_SIZE,
                max_order=4,
                jit_tree=False,
            )
        )

    # Vacuity gate: without far pairs there is no M2L to chunk and both branches
    # collapse to the same near-field-only sum.
    far_pairs = _m2l_far_pair_count(
        solver(chunk_size=4096, dtype=jnp.float32).prepare_state(
            positions,
            masses,
            leaf_size=_CHUNKED_M2L_LEAF_SIZE,
            max_order=4,
        )
    )
    assert far_pairs > 0, (
        f"no M2L pairs accepted at leaf_size={_CHUNKED_M2L_LEAF_SIZE}: the chunked "
        "and fullbatch M2L cannot differ, so this test would pass vacuously"
    )
    # 32 < 158 so the chunked path really does take more than one chunk.
    assert far_pairs > 32, (
        f"only {far_pairs} far pairs: m2l_chunk_size=32 would fit them in a single "
        "chunk and the chunked reduction would never loop"
    )

    reference = accelerations(chunk_size=4096, dtype=jnp.float64)
    acc_full = accelerations(chunk_size=4096, dtype=jnp.float32)
    acc_chunked = accelerations(chunk_size=32, dtype=jnp.float32)

    assert np.isfinite(acc_full).all()
    assert np.isfinite(acc_chunked).all()

    reference_norm = np.linalg.norm(reference)

    def rel_l2(candidate, other):
        return float(np.linalg.norm(candidate - other) / reference_norm)

    # 2e-6 is ~9x the worst measured 2.27e-07; a leaked or dropped chunk is order 1.
    for label, candidate in (("fullbatch", acc_full), ("chunked", acc_chunked)):
        error = rel_l2(candidate, reference)
        assert error < 2.0e-6, (
            f"{label} M2L drifted from the float64 reference: rel-L2 {error:.3e} "
            f"> 2e-6 ({far_pairs} far pairs, leaf={_CHUNKED_M2L_LEAF_SIZE})"
        )

    # 5e-7 is ~8x the worst measured 5.89e-08 chunk-boundary reordering difference.
    reorder = rel_l2(acc_chunked, acc_full)
    assert reorder < 5.0e-7, (
        f"chunked and fullbatch M2L disagree by more than reordering round-off: "
        f"rel-L2 {reorder:.3e} > 5e-7 ({far_pairs} far pairs)"
    )


def test_solidfmm_m2l_ignores_padded_compact_far_pairs():
    """Sentinel (-1) padding in the far-pair list must not corrupt the M2L.

    Both the exact-length and the padded list are checked against a **float64
    reference**, not against each other. The earlier padded-vs-exact form was a
    false negative: on JAX 0.9.x the two fp32 paths were *identically* wrong
    (difference exactly 0.0) while both sat 3.7e-04 from the true value, so the
    assertion passed on a coincidence. JAX 0.11.0 fuses the padded case
    differently, the two errors stopped cancelling, and the test failed even
    though nothing had become less accurate -- the padded path was in fact
    *closer* to truth there (1.1e-04 vs 3.7e-04). Comparing to a reference also
    tests the actual intent better: a leaking sentinel would move the result by
    order 1, not by 1e-04.

    The tolerance is set by TF32, not by this kernel's algebra. XLA lowers fp32
    matmuls on Ampere to TF32 (~10-bit mantissa) by default, which caps M2L
    relative accuracy at ~6e-04 from order 4 up, *regardless of expansion order*
    (real basis, jax 0.9.0.1, max rel err vs float64)::

        order    default matmul    jax.default_matmul_precision("highest")
          2         2.1e-06                     2.1e-06
          4         5.7e-04                     1.5e-06
          6         5.7e-04                     2.4e-06
          8         5.6e-04                     1.8e-06

    So the bound below is what is achievable, not slack. Tightening it means
    setting the matmul precision in the M2L kernels -- jaccpot does that in the
    L2P path (``jaccpot/downward/local_expansions.py``, ``Precision.HIGHEST``)
    but not here. Left as-is deliberately; see ARCHITECTURE.md section 7.

    What this does and does not discriminate, measured by mutation:

    * A ``-1``-sentinel row is dropped by the **index masking** (``tgt >= 0``),
      not by ``active_count`` -- forcing ``active_count=4`` on the padded list
      returns a byte-identical result. So this test verifies the *outcome* for
      sentinel padding; it cannot tell which of the two guards did the work.
    * The leak it does catch is the ``0``-padded form (what the treecode compact
      far-pair list produces, with the true count in ``far_pair_count``), where
      ``active_count`` is the only guard: honest ``active_count=1`` lands at
      3.7e-04, while ``active_count=4`` returns NaN and fails both the isfinite
      assertion and the bound below.
    """
    order = 2
    src_exact = jnp.array([1], dtype=INDEX_DTYPE)
    tgt_exact = jnp.array([0], dtype=INDEX_DTYPE)
    src_padded = jnp.array([1, -1, -1, -1], dtype=INDEX_DTYPE)
    tgt_padded = jnp.array([0, -1, -1, -1], dtype=INDEX_DTYPE)
    active_count = jnp.array(1, dtype=INDEX_DTYPE)

    def accumulate(src, tgt, complex_dtype, real_dtype):
        coeff_count = fmm_impl_private.sh_size(order)
        centers = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.25, -0.5],
                [-0.75, 0.5, 0.25],
                [0.5, -0.5, 1.0],
            ],
            dtype=real_dtype,
        )
        multipoles = jnp.arange(
            centers.shape[0] * coeff_count, dtype=real_dtype
        ).reshape((centers.shape[0], coeff_count)).astype(complex_dtype) * jnp.array(
            0.01 + 0.02j, dtype=complex_dtype
        )
        return kernels_core._accumulate_m2l_fullbatch(
            jnp.zeros_like(multipoles),
            multipoles,
            centers,
            src,
            tgt,
            active_count,
            order=order,
            basis_mode="complex",
            rotation="solidfmm",
            total_nodes=int(centers.shape[0]),
        )

    # The reference is only a reference if x64 is actually on; without it
    # complex128 silently degrades to complex64 and the comparison is vacuous.
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("float64 reference needs JAX_ENABLE_X64=1")
    reference = np.asarray(
        accumulate(src_exact, tgt_exact, jnp.complex128, jnp.float64)
    )

    exact_np = np.asarray(accumulate(src_exact, tgt_exact, jnp.complex64, jnp.float32))
    padded_full_np = np.asarray(
        accumulate(src_padded, tgt_padded, jnp.complex64, jnp.float32)
    )
    padded_chunked_np = np.asarray(
        kernels_core._accumulate_m2l_chunked_scan(
            jnp.zeros((4, fmm_impl_private.sh_size(order)), dtype=jnp.complex64),
            jnp.arange(4 * fmm_impl_private.sh_size(order), dtype=jnp.float32)
            .reshape((4, fmm_impl_private.sh_size(order)))
            .astype(jnp.complex64)
            * jnp.array(0.01 + 0.02j, dtype=jnp.complex64),
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.25, -0.5],
                    [-0.75, 0.5, 0.25],
                    [0.5, -0.5, 1.0],
                ],
                dtype=jnp.float32,
            ),
            src_padded,
            tgt_padded,
            active_count,
            order=order,
            basis_mode="complex",
            rotation="solidfmm",
            total_nodes=4,
            chunk_size=2,
        )
    )

    assert np.isfinite(padded_full_np).all()
    assert np.isfinite(padded_chunked_np).all()

    # 2e-3 leaves headroom over the ~6e-4 TF32 floor documented above without
    # admitting a leaked sentinel contribution, which would be order 1.
    scale = np.max(np.abs(reference))
    for name, got in (
        ("exact", exact_np),
        ("padded fullbatch", padded_full_np),
        ("padded chunked", padded_chunked_np),
    ):
        err = float(np.max(np.abs(got - reference)) / scale)
        assert err < 2e-3, (
            f"{name} M2L differs from the float64 reference by {err:.3e} "
            "(rel); a leaked sentinel pair would be O(1), while TF32 alone "
            "accounts for ~6e-4"
        )


def test_fast_preset_adaptive_large_cpu_policy_applies():
    fmm = FMMEngine(
        preset=FMMPreset.FAST,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        mac_type="dehnen",
    )
    # This test exercises the adaptive large-CPU runtime policy, which is the
    # non-default opt-out path: the production default is static fixed sizing
    # (JACCPOT_STATIC_RUNTIME_FIXED_SIZING=1), which deliberately skips adaptive
    # runtime rewrites. Disable it so the adaptive policy is resolved and asserted.
    fmm._static_runtime_fixed_sizing = False

    overrides = fmm._resolve_runtime_execution_overrides(
        num_particles=131072,
        backend="cpu",
    )

    assert overrides.adaptive_applied is True
    assert overrides.m2l_chunk_size == 32768
    assert overrides.traversal_config is not None
    assert overrides.traversal_config.process_block == 4096
    assert overrides.traversal_config.max_interactions_per_node == 65536
    assert overrides.grouped_interactions is True
    assert overrides.farfield_mode == "pair_grouped"
    assert overrides.center_mode == "aabb"
    assert overrides.refine_local_override is False


def test_fast_preset_adaptive_class_major_threshold():
    fmm = FMMEngine(
        preset=FMMPreset.FAST,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        mac_type="dehnen",
    )
    # Adaptive class-major farfield policy is the non-default opt-out path;
    # static fixed sizing (the production default) skips it. Disable it here so
    # the adaptive threshold behaviour is resolved and asserted.
    fmm._static_runtime_fixed_sizing = False

    overrides = fmm._resolve_runtime_execution_overrides(
        num_particles=262144,
        backend="cpu",
    )

    assert overrides.grouped_interactions is True
    assert overrides.farfield_mode == "class_major"


def test_adaptive_nearfield_edge_chunk_size_auto_policy(monkeypatch):
    fmm_cpu = FMMEngine(
        preset=FMMPreset.FAST,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        mac_type="dehnen",
        nearfield=NearFieldConfig(mode="auto", edge_chunk_size=256),
    )
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")

    assert (
        fmm_cpu._resolve_nearfield_edge_chunk_size(
            num_particles=131072,
            nearfield_mode="baseline",
        )
        == 256
    )
    assert (
        fmm_cpu._resolve_nearfield_edge_chunk_size(
            num_particles=262144,
            nearfield_mode="bucketed",
        )
        == 1024
    )
    assert (
        fmm_cpu._resolve_nearfield_edge_chunk_size(
            num_particles=1000000,
            nearfield_mode="bucketed",
        )
        == 2048
    )
    assert (
        fmm_cpu._resolve_nearfield_edge_chunk_size(
            num_particles=2000000,
            nearfield_mode="bucketed",
        )
        == 4096
    )

    fmm_gpu = FMMEngine(
        preset=FMMPreset.LARGE_N_GPU,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        nearfield=NearFieldConfig(mode="auto", edge_chunk_size=128),
    )
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    assert fmm_gpu._resolve_nearfield_mode(num_particles=131072) == "baseline"
    assert fmm_gpu._resolve_nearfield_mode(num_particles=262144) == "bucketed"
    assert (
        fmm_gpu._resolve_nearfield_edge_chunk_size(
            num_particles=131072,
            nearfield_mode="baseline",
        )
        == 128
    )
    assert (
        fmm_gpu._resolve_nearfield_edge_chunk_size(
            num_particles=262144,
            nearfield_mode="bucketed",
        )
        == 256
    )
    assert (
        fmm_gpu._resolve_nearfield_edge_chunk_size(
            num_particles=1000000,
            nearfield_mode="bucketed",
        )
        == 256
    )


def test_fast_preset_adaptive_policy_respects_explicit_overrides():
    cfg = DualTreeTraversalConfig(
        max_pair_queue=4096,
        process_block=256,
        max_interactions_per_node=4096,
        max_neighbors_per_leaf=4096,
    )
    fmm = FMMEngine(
        preset=FMMPreset.FAST,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm", m2l_chunk_size=2048),
        mac_type="dehnen",
        runtime_policy=RuntimePolicyConfig(traversal_config=cfg),
    )

    overrides = fmm._resolve_runtime_execution_overrides(
        num_particles=131072,
        backend="cpu",
    )

    assert overrides.adaptive_applied is False
    assert overrides.traversal_config is cfg
    assert overrides.m2l_chunk_size == 2048


def test_solidfmm_grouped_interactions_matches_sparse_path():
    key = jax.random.PRNGKey(23)
    num_particles = 128
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    fmm = FMMEngine(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        mac_type="dehnen",
        fixed_order=3,
    )

    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=16,
        return_reordered=True,
    )
    upward = fmm.prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=3,
        center_mode="aabb",
    )
    downward_sparse = fmm.prepare_downward_sweep(
        tree,
        upward,
        theta=0.6,
        grouped_interactions=False,
    )
    downward_grouped = fmm.prepare_downward_sweep(
        tree,
        upward,
        theta=0.6,
        interactions=downward_sparse.interactions,
        grouped_interactions=True,
    )

    assert np.allclose(
        np.asarray(downward_grouped.locals.coefficients),
        np.asarray(downward_sparse.locals.coefficients),
        rtol=1e-5,
        atol=1e-5,
    )


def test_solidfmm_grouped_class_major_matches_pair_grouped():
    """The two grouped far-field modes are one computation, differently batched.

    They share the class rotation blocks and the per-pair displacement, so they
    must agree to reassociation. This used to be asserted at 112 particles with
    ``leaf_size=16`` and ``theta=0.6``, which produces **zero** far pairs -- the
    assertion compared two all-zero local arrays and could not fail. It missed
    G.11 (``docs/refactor_audit_2026-08.md``): ``pair_grouped`` gathered its
    rotation with ``class_ids``, which yggdrax stores in the original rather than
    the class-sorted pair order, so ~70% of pairs were rotated by another class.
    At the sizes below that put the two modes a relative L2 of ~1.0 apart.

    The far-pair count is asserted explicitly so it can never go vacuous again.
    """
    key = jax.random.PRNGKey(31)
    num_particles = 512
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.abs(jax.random.normal(key, (num_particles,), dtype=jnp.float32)) + 1.0

    fmm = FMMEngine(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(
            rotation="solidfmm", grouped_interactions=True, mode="pair_grouped"
        ),
        mac_type="dehnen",
        fixed_order=3,
    )

    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
    )
    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        leaf_size=8,
        return_reordered=True,
    )
    upward = fmm.prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=3,
        center_mode="aabb",
    )
    downward_pair = fmm.prepare_downward_sweep(
        tree,
        upward,
        theta=0.6,
        grouped_interactions=True,
        farfield_mode="pair_grouped",
    )
    downward_class = fmm.prepare_downward_sweep(
        tree,
        upward,
        theta=0.6,
        interactions=downward_pair.interactions,
        grouped_interactions=True,
        farfield_mode="class_major",
    )

    far_pairs = int(downward_pair.interactions.sources.shape[0])
    assert far_pairs > 0, "vacuous: no far pairs, so the M2L modes ran on nothing"

    pair_locals = np.asarray(downward_pair.locals.coefficients)
    class_locals = np.asarray(downward_class.locals.coefficients)
    # Relative, not absolute: the local coefficients here are O(1e2), so the old
    # atol=1e-5 would have been the only binding term.
    rel_l2 = np.linalg.norm(pair_locals - class_locals) / np.linalg.norm(class_locals)
    assert rel_l2 < 1e-6, (
        f"grouped far-field modes disagree at rel-L2 {rel_l2:.3e} over "
        f"{far_pairs} far pairs; they are the same computation, differently batched"
    )


def _benchmark_like_distribution(
    num_particles: int,
    *,
    key: jax.Array,
    dtype: jnp.dtype,
):
    """Match the benchmark notebook's synthetic distribution."""
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


def _direct_accelerations_vectorized(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    softening: float,
):
    diff = positions[:, None, :] - positions[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1) + softening**2
    mask = ~np.eye(positions.shape[0], dtype=bool)
    inv_r3 = np.where(mask, 1.0 / (dist_sq * np.sqrt(dist_sq)), 0.0)
    weighted = diff * masses[None, :, None] * inv_r3[..., None]
    return -np.sum(weighted, axis=1)


def test_solidfmm_basis_rejects_non_solidfmm_rotation():
    with pytest.raises(
        ValueError,
        match="complex_rotation must be 'solidfmm'",
    ):
        FMMEngine(
            expansion_basis="solidfmm",
            farfield=FarFieldConfig(rotation="cached"),
        )


def test_dehnen_radius_scale_must_be_positive():
    with pytest.raises(ValueError, match="dehnen_radius_scale must be > 0"):
        FMMEngine(dehnen_radius_scale=0.0)


def test_nearfield_mode_validation():
    with pytest.raises(
        ValueError, match="nearfield_mode must be 'auto', 'baseline', or 'bucketed'"
    ):
        FMMEngine(
            nearfield=NearFieldConfig(mode="unknown"),
        )
    with pytest.raises(ValueError, match="nearfield_edge_chunk_size must be positive"):
        FMMEngine(
            nearfield=NearFieldConfig(edge_chunk_size=0),
        )


def test_solidfmm_dehnen_accuracy_improves_with_order():
    """Regression: solidfmm+dehnen should improve strongly with expansion order."""
    with jax.enable_x64(True):
        num_particles = 224
        softening = 1e-3
        positions, masses = _benchmark_like_distribution(
            num_particles,
            key=jax.random.PRNGKey(2),
            dtype=jnp.float64,
        )

        reference = _direct_accelerations_vectorized(
            np.asarray(positions),
            np.asarray(masses),
            softening=softening,
        )
        ref_norm = np.linalg.norm(reference)

        traversal = DualTreeTraversalConfig(
            max_pair_queue=65536,
            process_block=512,
            max_interactions_per_node=16384,
            max_neighbors_per_leaf=8192,
        )

        errors = []
        for order in (1, 2, 4):
            fmm = FMMEngine(
                theta=0.9,
                softening=softening,
                working_dtype=jnp.float64,
                runtime_policy=RuntimePolicyConfig(traversal_config=traversal),
                expansion_basis="solidfmm",
                farfield=FarFieldConfig(rotation="solidfmm"),
                mac_type="dehnen",
                fixed_order=order,
                fixed_max_leaf_size=16,
            )
            accelerations = np.asarray(
                fmm.compute_accelerations(
                    positions,
                    masses,
                    leaf_size=16,
                    max_order=order,
                )
            )
            rel_l2 = np.linalg.norm(accelerations - reference) / ref_norm
            errors.append(rel_l2)

    assert errors[0] > errors[1] > errors[2]
    # Keep a strong-margin guard against accidental convention/sign regressions.
    assert errors[0] / errors[2] > 12.0
