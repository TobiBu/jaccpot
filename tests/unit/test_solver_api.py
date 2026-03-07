"""Jaccpot package-local regression tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import (
    ComplexSHBasis,
    FarFieldConfig,
)
import jaccpot.runtime._fmm_impl as fmm_impl_private
from jaccpot import FastMultipoleMethod
from jaccpot import FastMultipoleMethod as ExpanseFMM
from jaccpot import (
    FMMAdvancedConfig,
)
from jaccpot import FMMPreset
from jaccpot import FMMPreset as ExpansePreset
from jaccpot import (
    NearFieldConfig,
    RuntimePolicyConfig,
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


def test_basis_complex_alias_matches_solidfmm():
    positions, masses = _sample_problem(n=64)
    fmm_alias = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="complex",
    )
    fmm_solidfmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    acc_alias = fmm_alias.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    acc_solidfmm = fmm_solidfmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    assert np.allclose(
        np.asarray(acc_alias), np.asarray(acc_solidfmm), rtol=1e-5, atol=1e-5
    )


def test_basis_object_is_accepted():
    positions, masses = _sample_problem(n=48)
    basis = ComplexSHBasis()
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
    assert fmm.basis == "complex"
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


def test_topology_reuse_options_flow_to_runtime():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        reuse_topology=True,
        rebuild_every=3,
    )
    assert fmm._impl.reuse_topology is True
    assert fmm._impl.rebuild_every == 3
    assert fmm.recent_topology_reused is False


def test_use_pallas_option_flows_to_runtime():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        use_pallas=True,
    )
    assert fmm._impl.use_pallas is True


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


def test_gpu_runtime_overrides_cap_traversal_capacities_for_large_n():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    overrides = fmm._impl._resolve_runtime_execution_overrides(
        num_particles=131072,
        backend="gpu",
    )
    cfg = overrides.traversal_config
    assert cfg is not None
    assert int(cfg.max_neighbors_per_leaf) <= 1024
    assert int(cfg.max_interactions_per_node) <= 4096
    assert int(cfg.max_pair_queue) >= 131072


def test_precision_fp32_sets_working_dtype():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        precision="fp32",
    )
    assert fmm._impl.working_dtype == jnp.float32


def test_precision_conflicts_with_working_dtype():
    with pytest.raises(ValueError, match="precision conflicts with explicit"):
        FastMultipoleMethod(
            preset=FMMPreset.FAST,
            basis="solidfmm",
            precision="fp32",
            working_dtype=jnp.float64,
        )


def test_precision_fp64_requires_x64_enabled():
    if bool(jax.config.jax_enable_x64):
        fmm = FastMultipoleMethod(
            preset=FMMPreset.FAST,
            basis="solidfmm",
            precision="fp64",
        )
        assert fmm._impl.working_dtype == jnp.float64
        return
    with pytest.raises(ValueError, match="requires jax_enable_x64=True"):
        FastMultipoleMethod(
            preset=FMMPreset.FAST,
            basis="solidfmm",
            precision="fp64",
        )


def test_nearfield_precompute_scatter_flag_flows_to_runtime():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            nearfield=NearFieldConfig(
                mode="bucketed",
                edge_chunk_size=256,
                precompute_scatter_schedules=False,
            )
        ),
    )
    assert fmm._impl.precompute_nearfield_scatter_schedules is False


def test_large_n_gpu_preset_applies_memory_safe_gpu_defaults():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.LARGE_N_GPU,
        basis="solidfmm",
    )
    assert fmm._impl.tree_type == "radix"
    assert fmm._impl.tree_build_mode == "lbvh"
    assert fmm._impl.grouped_interactions is False
    assert fmm._impl.nearfield_mode == "bucketed"
    assert fmm._impl.precompute_nearfield_scatter_schedules is False
    assert fmm._impl.streamed_far_pairs is True
    assert fmm._impl.mixed_order_farfield is False
    assert fmm._impl.enable_interaction_cache is False
    assert fmm._impl.retain_traversal_result is False
    assert fmm._impl.retain_interactions is False
    assert fmm._impl.mac_type == "dehnen"


def test_large_n_gpu_preset_accepts_string_alias():
    fmm = FastMultipoleMethod(
        preset="large_n_gpu",
        basis="solidfmm",
    )
    assert fmm.preset is FMMPreset.LARGE_N_GPU


def test_bucket_far_pairs_by_level_split_returns_two_gears():
    interactions = type(
        "DummyInteractions",
        (),
        {"level_offsets": jnp.asarray([0, 2, 4], dtype=jnp.int32)},
    )()
    src = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)
    tgt = jnp.asarray([4, 5, 6, 7], dtype=jnp.int32)
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
