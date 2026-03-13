"""Jaccpot package-local regression tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import (
    FarFieldConfig,
)
from jaccpot import FastMultipoleMethod
from jaccpot import FastMultipoleMethod as ExpanseFMM
from jaccpot import (
    FMMAdvancedConfig,
)
from jaccpot import FMMPreset
from jaccpot import FMMPreset as ExpansePreset
from jaccpot import (
    NearFieldConfig,
    TreeConfig,
)


def _sample_problem(n: int = 64):
    key = jax.random.PRNGKey(11)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (n,),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.float32,
    )
    return positions, masses


def _sample_velocities(n: int = 64):
    key = jax.random.PRNGKey(17)
    return jax.random.uniform(
        key,
        (n, 3),
        minval=-0.2,
        maxval=0.2,
        dtype=jnp.float32,
    )


def _direct_sum_jerk(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    velocities: jnp.ndarray,
    *,
    G: float,
    softening: float,
) -> jnp.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    vdiff = velocities[:, None, :] - velocities[None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening**2
    eps = jnp.finfo(positions.dtype).eps
    inv_r = jnp.where(dist_sq > 0, 1.0 / (jnp.sqrt(dist_sq) + eps), 0.0)
    inv_r3 = inv_r / dist_sq
    inv_r5 = inv_r3 / dist_sq
    eye = jnp.eye(positions.shape[0], dtype=bool)
    inv_r3 = jnp.where(eye, 0.0, inv_r3)
    inv_r5 = jnp.where(eye, 0.0, inv_r5)
    rv = jnp.sum(diff * vdiff, axis=-1)
    term = vdiff * inv_r3[..., None] - 3.0 * rv[..., None] * diff * inv_r5[..., None]
    weighted = masses[None, :, None] * term
    return -G * jnp.sum(weighted, axis=1)


def test_solver_matches_expanse_fast_path():
    positions, masses = _sample_problem(n=96)

    jaccpot_fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            farfield=FarFieldConfig(mode="pair_grouped"),
            nearfield=NearFieldConfig(mode="bucketed"),
        ),
    )
    expanse_fmm = ExpanseFMM(
        preset=ExpansePreset.FAST,
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
        farfield_mode="pair_grouped",
        nearfield_mode="bucketed",
    )

    acc_jaccpot = jaccpot_fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
    )
    acc_expanse = expanse_fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
    )
    assert np.allclose(
        np.asarray(acc_jaccpot), np.asarray(acc_expanse), rtol=1e-5, atol=1e-5
    )


def test_advanced_config_applies_to_runtime():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.BALANCED,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            farfield=FarFieldConfig(mode="class_major", grouped_interactions=True),
            nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=512),
            mac_type="engblom",
        ),
    )
    assert fmm.farfield_mode == "class_major"
    assert bool(fmm.grouped_interactions) is True
    assert fmm.nearfield_mode == "bucketed"
    assert int(fmm.nearfield_edge_chunk_size) == 512
    assert fmm.mac_type == "engblom"


def test_tree_type_flows_from_advanced_config():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(tree=TreeConfig(tree_type="radix")),
    )
    assert fmm._impl.tree_type == "radix"


def test_invalid_tree_type_raises():
    with pytest.raises(ValueError, match="tree_type must be one of"):
        FastMultipoleMethod(
            preset=FMMPreset.FAST,
            basis="solidfmm",
            advanced=FMMAdvancedConfig(tree=TreeConfig(tree_type="invalid-tree")),
        )


def test_kdtree_tree_type_runs_compute_accelerations():
    positions, masses = _sample_problem(n=64)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(tree=TreeConfig(tree_type="kdtree")),
    )
    acc = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    assert acc.shape == positions.shape
    assert np.isfinite(np.asarray(acc)).all()


def test_compute_accelerations_target_indices_matches_full_slice():
    positions, masses = _sample_problem(n=80)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    target_indices = jnp.asarray([1, 7, 11, 29, 63], dtype=jnp.int32)

    acc_full = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    acc_target = fmm.compute_accelerations(
        positions,
        masses,
        target_indices=target_indices,
        leaf_size=16,
        max_order=3,
    )

    assert acc_target.shape == (target_indices.shape[0], 3)
    assert np.allclose(
        np.asarray(acc_target),
        np.asarray(acc_full)[np.asarray(target_indices)],
        rtol=1e-5,
        atol=1e-5,
    )


def test_evaluate_prepared_state_target_indices_with_potential():
    positions, masses = _sample_problem(n=72)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    target_indices = jnp.asarray([0, 5, 9, 10, 33], dtype=jnp.int32)
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )

    full_acc, full_pot = fmm.evaluate_prepared_state(state, return_potential=True)
    target_acc, target_pot = fmm.evaluate_prepared_state(
        state,
        target_indices=target_indices,
        return_potential=True,
    )

    np_idx = np.asarray(target_indices)
    assert target_acc.shape == (target_indices.shape[0], 3)
    assert target_pot.shape == (target_indices.shape[0],)
    assert np.allclose(np.asarray(target_acc), np.asarray(full_acc)[np_idx])
    assert np.allclose(np.asarray(target_pot), np.asarray(full_pot)[np_idx])


def test_target_indices_out_of_range_raises():
    positions, masses = _sample_problem(n=16)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    with pytest.raises(ValueError, match="out-of-range"):
        fmm.compute_accelerations(
            positions,
            masses,
            target_indices=jnp.asarray([0, 16], dtype=jnp.int32),
            leaf_size=8,
            max_order=2,
        )


def test_target_indices_preserve_order_and_duplicates():
    positions, masses = _sample_problem(n=64)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    target_indices = jnp.asarray([9, 3, 9, 0], dtype=jnp.int32)
    full_acc = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    subset_acc = fmm.compute_accelerations(
        positions,
        masses,
        target_indices=target_indices,
        leaf_size=16,
        max_order=3,
    )
    assert subset_acc.shape == (4, 3)
    assert np.allclose(
        np.asarray(subset_acc),
        np.asarray(full_acc)[np.asarray(target_indices)],
        rtol=1e-5,
        atol=1e-5,
    )


def test_evaluate_prepared_state_can_run_inside_jit_with_targets():
    positions, masses = _sample_problem(n=64)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    target_indices = jnp.asarray([0, 7, 11, 23, 31], dtype=jnp.int32)

    jit_eval = jax.jit(
        lambda st, idx: fmm.evaluate_prepared_state(st, target_indices=idx)
    )
    acc_jit = jit_eval(state, target_indices)
    acc_ref = fmm.evaluate_prepared_state(state, target_indices=target_indices)

    assert acc_jit.shape == (target_indices.shape[0], 3)
    assert np.allclose(np.asarray(acc_jit), np.asarray(acc_ref), rtol=1e-5, atol=1e-5)


def test_compute_accelerations_returns_acc_derivatives_when_requested():
    positions, masses = _sample_problem(n=48)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    acc, derivatives = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
        max_acc_derivative_order=1,
    )
    assert acc.shape == positions.shape
    assert len(derivatives) == 1
    assert derivatives[0].shape == (positions.shape[0], 3, 3)


def test_compute_accelerations_acc_derivatives_target_indices_match_slice():
    positions, masses = _sample_problem(n=56)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    target_indices = jnp.asarray([2, 8, 13, 21, 34], dtype=jnp.int32)
    acc_full, deriv_full = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
        max_acc_derivative_order=1,
    )
    acc_sub, deriv_sub = fmm.compute_accelerations(
        positions,
        masses,
        target_indices=target_indices,
        leaf_size=16,
        max_order=3,
        max_acc_derivative_order=1,
    )
    np_idx = np.asarray(target_indices)
    assert np.allclose(np.asarray(acc_sub), np.asarray(acc_full)[np_idx])
    assert np.allclose(np.asarray(deriv_sub[0]), np.asarray(deriv_full[0])[np_idx])


def test_compute_accelerations_and_jerk_target_indices_match_full_slice():
    positions, masses = _sample_problem(n=52)
    velocities = _sample_velocities(n=52)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    target_indices = jnp.asarray([1, 6, 14, 23, 37], dtype=jnp.int32)
    acc_full, jerk_full = fmm.compute_accelerations_and_jerk(
        positions,
        masses,
        velocities,
        leaf_size=16,
        max_order=3,
    )
    acc_sub, jerk_sub = fmm.compute_accelerations_and_jerk(
        positions,
        masses,
        velocities,
        target_indices=target_indices,
        leaf_size=16,
        max_order=3,
    )
    np_idx = np.asarray(target_indices)
    assert np.allclose(np.asarray(acc_sub), np.asarray(acc_full)[np_idx])
    assert np.allclose(np.asarray(jerk_sub), np.asarray(jerk_full)[np_idx])


def test_compute_accelerations_and_jerk_matches_direct_sum_small_n():
    n = 24
    positions, masses = _sample_problem(n=n)
    velocities = _sample_velocities(n=n)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.ACCURATE,
        basis="solidfmm",
    )
    acc_fmm, jerk_fmm = fmm.compute_accelerations_and_jerk(
        positions,
        masses,
        velocities,
        leaf_size=12,
        max_order=4,
        theta=1e-4,
    )
    jerk_ref = _direct_sum_jerk(
        positions,
        masses,
        velocities,
        G=1.0,
        softening=1e-3,
    )
    rel = np.linalg.norm(np.asarray(jerk_fmm - jerk_ref)) / (
        np.linalg.norm(np.asarray(jerk_ref)) + 1e-12
    )
    assert acc_fmm.shape == positions.shape
    assert jerk_fmm.shape == positions.shape
    assert rel < 5e-2


def test_jitted_compute_does_not_leak_tracers_into_solver_caches():
    positions, masses = _sample_problem(n=64)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )

    jit_full = jax.jit(
        lambda p, m: fmm.compute_accelerations(
            p,
            m,
            leaf_size=16,
            max_order=3,
        )
    )
    _ = jit_full(positions, masses)

    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    acc = fmm.evaluate_prepared_state(state)
    assert acc.shape == positions.shape
