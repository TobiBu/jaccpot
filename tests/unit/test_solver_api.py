"""The public facade, ``FastMultipoleMethod``, end to end.

Solving, ``target_indices`` across the acceleration APIs, what a prepared state
retains, and the runtime caches. Split by subject in the 2026-10 cleanup (Z-a):
the GPU memory and traversal caps went to ``test_large_n_config_thresholds.py``,
the ``large_n_gpu`` preset to ``test_large_n_fast_path_policy.py``, jerk and the
time derivatives to ``core/test_time_derivatives.py``, configuration flow to
``runtime/test_engine_config_resolution.py``, the upward's default leaf batch to
``core/test_real_upward_sweep.py``, and the removed options and shims to
``test_removed_options.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.interactions import NodeInteractionList

import jaccpot.runtime._fmm_impl as fmm_impl_private
from jaccpot import (
    FarFieldConfig,
)
from jaccpot import FastMultipoleMethod
from jaccpot import FastMultipoleMethod as ExpanseFMM
from jaccpot import FMMAdvancedConfig
from jaccpot import FMMPreset
from jaccpot import FMMPreset as ExpansePreset
from jaccpot import NearFieldConfig, RealSHBasis, RuntimePolicyConfig, TreeConfig


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


def test_solver_matches_expanse_fast_path():
    positions, masses = _sample_problem(n=48)

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
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            farfield=FarFieldConfig(mode="pair_grouped", rotation="solidfmm"),
            nearfield=NearFieldConfig(mode="bucketed"),
            mac_type="dehnen",
        ),
    )

    acc_jaccpot = jaccpot_fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    acc_expanse = expanse_fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    assert np.allclose(
        np.asarray(acc_jaccpot), np.asarray(acc_expanse), rtol=1e-5, atol=1e-5
    )


def test_basis_object_is_accepted():
    positions, masses = _sample_problem(n=48)
    basis = RealSHBasis()
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis=basis,
    )
    acc = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    assert fmm.basis == "real"
    assert acc.shape == positions.shape


def test_real_basis_selects_rot_scale_m2l_impl():
    positions, masses = _sample_problem(n=40)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
    )
    acc = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
    )
    assert fmm.basis == "real"
    assert fmm._impl.m2l_impl == "rot_scale"
    assert acc.shape == positions.shape


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


def test_target_indices_match_slice_across_acceleration_apis():
    positions, masses = _sample_problem(n=64)
    velocities = _sample_velocities(n=64)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    acc_full = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    for target_indices in (
        jnp.asarray([1, 7, 11, 29, 63], dtype=jnp.int32),
        jnp.asarray([9, 3, 9, 0], dtype=jnp.int32),
    ):
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

    deriv_target_indices = jnp.asarray([2, 8, 13, 21, 34], dtype=jnp.int32)
    acc_full_deriv, deriv_full = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
        max_acc_derivative_order=1,
    )
    acc_sub_deriv, deriv_sub = fmm.compute_accelerations(
        positions,
        masses,
        target_indices=deriv_target_indices,
        leaf_size=16,
        max_order=3,
        max_acc_derivative_order=1,
    )
    np_idx_deriv = np.asarray(deriv_target_indices)
    assert np.allclose(
        np.asarray(acc_sub_deriv),
        np.asarray(acc_full_deriv)[np_idx_deriv],
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.allclose(
        np.asarray(deriv_sub[0]),
        np.asarray(deriv_full[0])[np_idx_deriv],
        rtol=1e-5,
        atol=1e-5,
    )

    jerk_target_indices = jnp.asarray([1, 6, 14, 23, 37], dtype=jnp.int32)
    acc_full_jerk, jerk_full = fmm.compute_accelerations_and_jerk(
        positions,
        masses,
        velocities,
        leaf_size=16,
        max_order=3,
    )
    acc_sub_jerk, jerk_sub = fmm.compute_accelerations_and_jerk(
        positions,
        masses,
        velocities,
        target_indices=jerk_target_indices,
        leaf_size=16,
        max_order=3,
    )
    np_idx_jerk = np.asarray(jerk_target_indices)
    assert np.allclose(
        np.asarray(acc_sub_jerk),
        np.asarray(acc_full_jerk)[np_idx_jerk],
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.allclose(
        np.asarray(jerk_sub),
        np.asarray(jerk_full)[np_idx_jerk],
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


def test_clear_runtime_caches_resets_runtime_state():
    positions, masses = _sample_problem(n=64)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    _ = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
        reuse_prepared_state=True,
    )

    assert fmm._impl._prepared_state_cache_value is not None
    fmm.clear_runtime_caches(clear_jax_compilation=False)
    assert fmm._impl._prepared_state_cache_value is None


def test_bucket_far_pairs_by_level_split_returns_two_gears():
    # A real `NodeInteractionList`, not a one-attribute stub. The function only
    # reads `level_offsets`, so a stub passed -- but it also made the test assert
    # nothing about the declared contract, and it was one of the things that made
    # `JACCPOT_RUNTIME_TYPECHECK=1` red (F40). Building the real NamedTuple costs
    # five more lines and means the test breaks if the contract changes.
    src = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)
    tgt = jnp.asarray([4, 5, 6, 7], dtype=jnp.int32)
    interactions = NodeInteractionList(
        offsets=jnp.asarray([0, 4], dtype=jnp.int32),
        sources=src,
        targets=tgt,
        counts=jnp.asarray([4], dtype=jnp.int32),
        level_offsets=jnp.asarray([0, 2, 4], dtype=jnp.int32),
        target_levels=jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
    )
    gears, buckets = fmm_impl_private._bucket_far_pairs_by_level_split(
        interactions=interactions,
        src_far=src,
        tgt_far=tgt,
        max_order=4,
        min_order=3,
    )
    assert gears == (3, 4)
    assert int(buckets[0][0].shape[0]) == 2
    assert int(buckets[1][0].shape[0]) == 2


def test_prepare_state_streamed_can_drop_interaction_storage():
    positions, masses = _sample_problem(n=128)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            farfield=FarFieldConfig(
                mode="auto",
                grouped_interactions=False,
                streamed_far_pairs=True,
            ),
            runtime=RuntimePolicyConfig(
                enable_interaction_cache=False,
                retain_traversal_result=False,
                retain_interactions=False,
            ),
        ),
    )

    state = fmm.prepare_state(positions, masses, leaf_size=16, max_order=3)
    assert state.interactions is None
    assert int(state.downward.interactions.sources.shape[0]) == 0
    acc = fmm.evaluate_prepared_state(state)
    assert acc.shape == positions.shape


def test_prepare_state_streamed_uses_compact_far_pairs_without_node_interactions():
    positions, masses = _sample_problem(n=128)
    call_kwargs: list[dict[str, object]] = []
    original = fmm_impl_private.build_interactions_and_neighbors

    def _recording_builder(*args, **kwargs):
        call_kwargs.append(dict(kwargs))
        return original(*args, **kwargs)

    fmm = FastMultipoleMethod(
        preset=FMMPreset.LARGE_N_GPU,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            farfield=FarFieldConfig(
                mode="pair_grouped",
                grouped_interactions=False,
                streamed_far_pairs=True,
                mixed_order=False,
            ),
            runtime=RuntimePolicyConfig(
                memory_objective="minimum_memory",
                enable_interaction_cache=False,
                retain_traversal_result=False,
                retain_interactions=False,
                traversal_config=None,
            ),
        ),
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            fmm_impl_private, "build_interactions_and_neighbors", _recording_builder
        )
        state = fmm.prepare_state(positions, masses, leaf_size=16, max_order=3)

    if call_kwargs:
        assert call_kwargs[-1]["return_compact_far_pairs"] is True
        assert call_kwargs[-1]["return_interactions"] is False
    else:
        # Dedicated large-N prepare path bypasses generic interaction building.
        backend = str(getattr(state, "execution_backend", "")).strip().lower()
        # CPU runs may resolve LARGE_N_GPU preset to radix while still bypassing
        # generic interaction building.
        assert backend in {"large_n", "radix"}
    assert state.interactions is None
    assert int(state.downward.interactions.sources.shape[0]) == 0


def test_prepare_state_non_streamed_without_retention_omits_interactions():
    positions, masses = _sample_problem(n=64)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            farfield=FarFieldConfig(streamed_far_pairs=False),
            runtime=RuntimePolicyConfig(
                enable_interaction_cache=False,
                retain_interactions=False,
            ),
        ),
    )

    state = fmm.prepare_state(positions, masses, leaf_size=16, max_order=2)
    assert state.interactions is None
    assert int(state.downward.interactions.sources.shape[0]) == 0
    acc = fmm.evaluate_prepared_state(state)
    assert acc.shape == positions.shape


def test_prepare_state_does_not_retain_duplicate_component_matrix():
    positions, masses = _sample_problem(n=64)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )

    state = fmm.prepare_state(positions, masses, leaf_size=16, max_order=2)
    assert state.upward.multipoles.component_matrix is None
    acc = fmm.evaluate_prepared_state(state)
    assert acc.shape == positions.shape


def test_prepare_state_streamed_without_adaptive_skips_traversal_result_build():
    positions, masses = _sample_problem(n=64)
    fmm = ExpanseFMM(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            farfield=FarFieldConfig(
                grouped_interactions=False,
                streamed_far_pairs=True,
                rotation="solidfmm",
            ),
            runtime=RuntimePolicyConfig(
                retain_traversal_result=False,
                retain_interactions=False,
            ),
            mac_type="dehnen",
        ),
    )

    seen = []
    original = fmm_impl_private.build_interactions_and_neighbors

    def spy_build(*args, **kwargs):
        seen.append(dict(kwargs))
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "jaccpot.runtime.fmm.build_interactions_and_neighbors",
            spy_build,
        )
        state = fmm.prepare_state(positions, masses, leaf_size=8, max_order=2)

    assert seen
    assert all(bool(item.get("return_result", True)) is False for item in seen)
    assert state.dual_tree_result is None


def test_prepare_state_adaptive_order_requests_compact_far_pairs():
    positions, masses = _sample_problem(n=64)
    fmm = ExpanseFMM(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        basis="solidfmm",
        adaptive_order=True,
        p_gears=(2, 3),
        advanced=FMMAdvancedConfig(
            farfield=FarFieldConfig(rotation="solidfmm"),
            runtime=RuntimePolicyConfig(retain_traversal_result=False),
            mac_type="dehnen",
        ),
    )

    seen = []
    original = fmm_impl_private.build_interactions_and_neighbors

    def spy_build(*args, **kwargs):
        seen.append(dict(kwargs))
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "jaccpot.runtime.fmm.build_interactions_and_neighbors",
            spy_build,
        )
        state = fmm.prepare_state(positions, masses, leaf_size=8, max_order=3)

    assert seen
    assert all(bool(item.get("return_result", True)) is False for item in seen)
    assert all(
        bool(item.get("return_compact_far_pairs", False)) is True for item in seen
    )
    assert state.dual_tree_result is None
