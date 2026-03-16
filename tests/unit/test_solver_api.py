"""Jaccpot package-local regression tests."""

import json
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.interactions import DualTreeTraversalConfig

import jaccpot.runtime._fmm_impl as fmm_impl_private
import jaccpot.runtime._interaction_cache as interaction_cache_private
import jaccpot.runtime.fmm as runtime_fmm
import jaccpot.upward.solidfmm_complex_tree_expansions as upward_private
from jaccpot import (
    ComplexSHBasis,
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


def test_dehnen_error_defaults_to_paper_policy_settings():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(mac_type="dehnen_error"),
    )
    assert fmm.mac_type == "dehnen_error"
    assert fmm._impl.adaptive_error_model == "dehnen_paper"
    assert fmm._impl.mac_force_scale_mode == "paper"


def test_dehnen_error_preserves_explicit_policy_overrides():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        adaptive_error_model="dehnen_degree",
        mac_force_scale_mode="prepass",
        advanced=FMMAdvancedConfig(mac_type="dehnen_error"),
    )
    assert fmm.mac_type == "dehnen_error"
    assert fmm._impl.adaptive_error_model == "dehnen_degree"
    assert fmm._impl.mac_force_scale_mode == "prepass"


def test_tree_type_flows_from_advanced_config():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(tree=TreeConfig(tree_type="radix")),
    )
    assert fmm._impl.tree_type == "radix"
    assert fmm._impl.execution_backend == "auto"


def test_execution_backend_flows_from_advanced_config():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="radix"),
        ),
    )
    assert fmm._impl.tree_type == "octree"
    assert fmm._impl.execution_backend == "radix"


def test_prepare_state_records_resolved_execution_backend():
    positions, masses = _sample_problem(n=32)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="auto"),
        ),
    )

    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
    )

    assert state.execution_backend == "radix"


def test_explicit_octree_execution_backend_prepares_state():
    positions, masses = _sample_problem(n=32)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="octree"),
        ),
    )

    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
    )

    assert state.execution_backend == "octree"
    assert state.octree is not None
    assert state.octree_upward is not None
    assert state.octree_downward is not None


def test_octree_prepare_state_exposes_octree_execution_view():
    positions, masses = _sample_problem(n=48)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(tree=TreeConfig(tree_type="octree")),
    )

    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
    )

    assert state.tree.tree_type == "octree"
    assert state.octree is not None
    assert int(state.octree.num_valid_nodes) > 0
    assert state.octree.radix_node_to_oct.shape[0] == state.tree.parent.shape[0]
    assert state.octree.radix_leaf_to_oct.shape[0] == state.tree.num_leaves
    assert state.octree.box_centers.shape == (state.octree.valid_mask.shape[0], 3)
    assert state.octree.box_half_extents.shape == (state.octree.valid_mask.shape[0], 3)
    assert state.nearfield_interop is not None
    assert np.array_equal(
        np.asarray(state.nearfield_interop.leaf_nodes),
        np.asarray(state.neighbor_list.leaf_indices),
    )
    assert state.nearfield_interop.node_ranges.shape[0] == state.tree.parent.shape[0]
    assert int(np.asarray(state.nearfield_interop.counts).sum()) == int(
        np.asarray(state.neighbor_list.counts).sum()
    )
    native_map = np.asarray(state.nearfield_interop.particle_order_to_native_leaf)
    assert native_map.shape == (state.tree.num_leaves,)
    assert np.array_equal(np.sort(native_map), np.arange(state.tree.num_leaves))


def test_octree_solver_matches_radix_prepare_path():
    positions, masses = _sample_problem(n=64)
    radix = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(tree=TreeConfig(tree_type="radix")),
    )
    octree = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(tree=TreeConfig(tree_type="octree")),
    )

    acc_radix = radix.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    acc_octree = octree.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )

    assert np.allclose(
        np.asarray(acc_octree), np.asarray(acc_radix), rtol=1e-5, atol=1e-5
    )


def test_octree_execution_backend_matches_radix_on_octree_tree():
    positions, masses = _sample_problem(n=64)
    radix = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="radix"),
        ),
    )
    octree = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="octree"),
        ),
    )

    acc_radix = radix.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    acc_octree = octree.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )

    assert np.allclose(
        np.asarray(acc_octree), np.asarray(acc_radix), rtol=1e-5, atol=1e-5
    )


def test_octree_execution_backend_supports_baseline_nearfield_mode():
    positions, masses = _sample_problem(n=64)
    radix = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="radix"),
            nearfield=NearFieldConfig(mode="baseline"),
        ),
    )
    octree = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="octree"),
            nearfield=NearFieldConfig(mode="baseline"),
        ),
    )

    acc_radix = radix.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    acc_octree = octree.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )

    assert np.allclose(
        np.asarray(acc_octree), np.asarray(acc_radix), rtol=1e-5, atol=1e-5
    )


def test_octree_execution_backend_supports_class_major_farfield_mode():
    positions, masses = _sample_problem(n=64)
    radix = FastMultipoleMethod(
        preset=FMMPreset.BALANCED,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="radix"),
            farfield=FarFieldConfig(mode="class_major", grouped_interactions=True),
            nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=256),
        ),
    )
    octree = FastMultipoleMethod(
        preset=FMMPreset.BALANCED,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="octree"),
            farfield=FarFieldConfig(mode="class_major", grouped_interactions=True),
            nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=256),
        ),
    )

    acc_radix = radix.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )
    acc_octree = octree.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=3,
    )

    assert np.allclose(
        np.asarray(acc_octree), np.asarray(acc_radix), rtol=1e-5, atol=1e-5
    )


def test_octree_execution_backend_exposes_native_nearfield_view():
    positions, masses = _sample_problem(n=72)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="octree"),
        ),
    )

    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
    )

    assert state.octree is not None
    assert state.nearfield_interop is not None
    leaf_nodes = np.asarray(state.nearfield_interop.leaf_nodes)
    native_map = np.asarray(state.nearfield_interop.particle_order_to_native_leaf)
    carrier_nodes = np.unique(np.asarray(state.octree.radix_leaf_to_oct))
    assert state.nearfield_interop.node_ranges.shape[0] == state.octree.parent.shape[0]
    assert np.array_equal(np.sort(leaf_nodes), np.sort(carrier_nodes))
    assert native_map.shape == leaf_nodes.shape
    assert np.array_equal(np.sort(native_map), np.arange(leaf_nodes.shape[0]))
    assert state.nearfield_interop.leaf_particle_indices is not None
    assert state.nearfield_interop.leaf_particle_mask is not None
    assert state.nearfield_interop.particle_to_leaf_position is not None


def test_octree_execution_backend_target_indices_match_full_prepared_state():
    positions, masses = _sample_problem(n=72)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="octree"),
        ),
    )
    target_indices = jnp.asarray([0, 5, 9, 10, 33], dtype=jnp.int32)

    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
    )

    full_acc, full_pot = fmm.evaluate_prepared_state(state, return_potential=True)
    target_acc, target_pot = fmm.evaluate_prepared_state(
        state,
        target_indices=target_indices,
        return_potential=True,
    )

    np_idx = np.asarray(target_indices)
    assert state.nearfield_interop is not None
    assert target_acc.shape == (target_indices.shape[0], 3)
    assert target_pot.shape == (target_indices.shape[0],)
    assert np.allclose(np.asarray(target_acc), np.asarray(full_acc)[np_idx])
    assert np.allclose(np.asarray(target_pot), np.asarray(full_pot)[np_idx])


def test_octree_execution_backend_prepared_state_jit_targets_match_eager():
    positions, masses = _sample_problem(n=72)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="octree"),
        ),
    )
    target_indices = jnp.asarray([0, 7, 11, 23, 31], dtype=jnp.int32)
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
    )

    jit_eval = jax.jit(
        lambda st, idx: fmm.evaluate_prepared_state(st, target_indices=idx)
    )
    acc_jit = jit_eval(state, target_indices)
    acc_ref = fmm.evaluate_prepared_state(state, target_indices=target_indices)

    assert acc_jit.shape == (target_indices.shape[0], 3)
    assert np.allclose(np.asarray(acc_jit), np.asarray(acc_ref), rtol=1e-5, atol=1e-5)


def test_octree_execution_backend_prepared_state_eager_matches_compiled():
    positions, masses = _sample_problem(n=72)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="octree"),
        ),
    )
    target_indices = jnp.asarray([0, 5, 9, 10, 33], dtype=jnp.int32)
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
    )

    full_acc_compiled, full_pot_compiled = fmm.evaluate_prepared_state(
        state,
        return_potential=True,
        jit_traversal=True,
    )
    full_acc_eager, full_pot_eager = fmm.evaluate_prepared_state(
        state,
        return_potential=True,
        jit_traversal=False,
    )
    target_acc_compiled, target_pot_compiled = fmm.evaluate_prepared_state(
        state,
        target_indices=target_indices,
        return_potential=True,
        jit_traversal=True,
    )
    target_acc_eager, target_pot_eager = fmm.evaluate_prepared_state(
        state,
        target_indices=target_indices,
        return_potential=True,
        jit_traversal=False,
    )

    assert np.allclose(
        np.asarray(full_acc_compiled),
        np.asarray(full_acc_eager),
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.allclose(
        np.asarray(full_pot_compiled),
        np.asarray(full_pot_eager),
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.allclose(
        np.asarray(target_acc_compiled),
        np.asarray(target_acc_eager),
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.allclose(
        np.asarray(target_pot_compiled),
        np.asarray(target_pot_eager),
        rtol=1e-5,
        atol=1e-5,
    )


def test_octree_execution_backend_target_indices_preserve_order_and_duplicates():
    positions, masses = _sample_problem(n=72)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="octree"),
        ),
    )
    target_indices = jnp.asarray([9, 3, 9, 0], dtype=jnp.int32)
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
    )

    full_acc, full_pot = fmm.evaluate_prepared_state(state, return_potential=True)
    subset_acc, subset_pot = fmm.evaluate_prepared_state(
        state,
        target_indices=target_indices,
        return_potential=True,
    )

    np_idx = np.asarray(target_indices)
    assert subset_acc.shape == (target_indices.shape[0], 3)
    assert subset_pot.shape == (target_indices.shape[0],)
    assert np.allclose(
        np.asarray(subset_acc),
        np.asarray(full_acc)[np_idx],
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.allclose(
        np.asarray(subset_pot),
        np.asarray(full_pot)[np_idx],
        rtol=1e-5,
        atol=1e-5,
    )


def test_octree_execution_backend_prepared_state_jit_targets_with_potential():
    positions, masses = _sample_problem(n=72)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="octree"),
        ),
    )
    target_indices = jnp.asarray([0, 7, 11, 23, 31], dtype=jnp.int32)
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
    )

    jit_eval = jax.jit(
        lambda st, idx: fmm.evaluate_prepared_state(
            st,
            target_indices=idx,
            return_potential=True,
        )
    )
    acc_jit, pot_jit = jit_eval(state, target_indices)
    acc_ref, pot_ref = fmm.evaluate_prepared_state(
        state,
        target_indices=target_indices,
        return_potential=True,
    )

    assert acc_jit.shape == (target_indices.shape[0], 3)
    assert pot_jit.shape == (target_indices.shape[0],)
    assert np.allclose(np.asarray(acc_jit), np.asarray(acc_ref), rtol=1e-5, atol=1e-5)
    assert np.allclose(np.asarray(pot_jit), np.asarray(pot_ref), rtol=1e-5, atol=1e-5)


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


def test_octree_reuse_prepared_state_uses_cache_without_topology_reuse(monkeypatch):
    positions, masses = _sample_problem(n=72)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="octree"),
        ),
    )

    acc_first = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
        reuse_prepared_state=True,
    )

    cached_state = fmm._impl._prepared_state_cache_value
    assert cached_state is not None
    assert cached_state.execution_backend == "octree"
    assert fmm.recent_topology_reused is False

    def fail_prepare_state(*args, **kwargs):
        raise AssertionError("prepare_state should not run when octree cache is reused")

    monkeypatch.setattr(fmm._impl, "prepare_state", fail_prepare_state)
    acc_second = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
        reuse_prepared_state=True,
    )

    assert fmm._impl._prepared_state_cache_value is cached_state
    assert fmm.recent_topology_reused is False
    assert np.allclose(
        np.asarray(acc_second), np.asarray(acc_first), rtol=1e-5, atol=1e-5
    )


def test_octree_clear_runtime_caches_resets_prepared_state_cache():
    positions, masses = _sample_problem(n=72)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(tree_type="octree"),
            runtime=RuntimePolicyConfig(execution_backend="octree"),
        ),
    )
    _ = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=8,
        max_order=3,
        reuse_prepared_state=True,
    )

    assert fmm._impl._prepared_state_cache_value is not None
    assert fmm._impl._prepared_state_cache_value.execution_backend == "octree"
    fmm.clear_runtime_caches(clear_jax_compilation=False)
    assert fmm._impl._prepared_state_cache_value is None
    assert fmm.recent_topology_reused is False


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
    assert int(cfg.max_neighbors_per_leaf) == 2048
    assert int(cfg.max_interactions_per_node) == 8192
    assert int(cfg.max_pair_queue) >= 131072


def test_large_gpu_runs_skip_nearfield_scatter_precompute(monkeypatch):
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    monkeypatch.setattr(fmm_impl_private.jax, "default_backend", lambda: "gpu")
    assert (
        fmm._impl._should_precompute_nearfield_scatter_schedules(num_particles=131072)
        is False
    )


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
    assert fmm._impl.m2l_chunk_size is None
    assert fmm._impl.enable_interaction_cache is False
    assert fmm._impl.retain_traversal_result is False
    assert fmm._impl.retain_interactions is False
    assert fmm._impl.autotune_m2l_chunk is True
    assert fmm._impl.memory_objective == "minimum_memory"
    assert fmm._impl.precompute_grouped_class_segments is False
    assert fmm._impl.upward_leaf_batch_size == 2048
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


def test_prepare_state_streamed_uses_compact_far_pairs_without_node_interactions():
    positions, masses = _sample_problem(n=128)
    call_kwargs: list[dict[str, object]] = []
    original = runtime_fmm.build_interactions_and_neighbors

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
        mp.setattr(runtime_fmm, "build_interactions_and_neighbors", _recording_builder)
        state = fmm.prepare_state(positions, masses, leaf_size=16, max_order=3)

    assert call_kwargs
    assert call_kwargs[-1]["return_compact_far_pairs"] is True
    assert call_kwargs[-1]["return_interactions"] is False
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


def test_prepare_state_solidfmm_skips_prebuilt_locals_template():
    positions, masses = _sample_problem(n=64)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )

    seen_templates = []
    original = fmm._impl._prepare_downward_with_artifacts

    def spy_prepare_downward(*args, **kwargs):
        seen_templates.append(kwargs.get("locals_template"))
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fmm._impl, "_prepare_downward_with_artifacts", spy_prepare_downward)
        state = fmm.prepare_state(positions, masses, leaf_size=16, max_order=2)

    assert seen_templates == [None]
    acc = fmm.evaluate_prepared_state(state)
    assert acc.shape == positions.shape


def test_prepare_state_solidfmm_complex_locals_remain_conjugate_symmetric():
    positions, masses = _sample_problem(n=64)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )

    state = fmm.prepare_state(positions, masses, leaf_size=16, max_order=2)
    coeffs = state.downward.locals.coefficients
    coeffs_sym = fmm_impl_private.enforce_conjugate_symmetry_batch(
        coeffs,
        order=int(state.downward.locals.order),
    )
    assert np.allclose(np.asarray(coeffs), np.asarray(coeffs_sym))
    acc = fmm.evaluate_prepared_state(state)
    assert acc.shape == positions.shape


def test_prepare_state_streamed_without_adaptive_skips_traversal_result_build():
    positions, masses = _sample_problem(n=64)
    fmm = ExpanseFMM(
        theta=0.6,
        softening=1e-3,
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
        grouped_interactions=False,
        streamed_far_pairs=True,
        retain_traversal_result=False,
        retain_interactions=False,
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
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        mac_type="dehnen",
        adaptive_order=True,
        p_gears=(2, 3),
        retain_traversal_result=False,
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


def test_runtime_autotune_m2l_chunk_flag_flows_to_runtime():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(autotune_m2l_chunk=True),
        ),
    )
    assert bool(fmm._impl.autotune_m2l_chunk) is True


def test_runtime_fail_fast_disables_autotune_and_host_refine():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(
                fail_fast=True,
                autotune_m2l_chunk=True,
                host_refine_mode="on",
            ),
        ),
    )
    assert bool(fmm._impl.fail_fast) is True
    assert bool(fmm._impl.autotune_m2l_chunk) is False
    assert fmm._impl.host_refine_mode == "off"


def test_runtime_memory_policy_fields_flow_to_runtime():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(
                memory_objective="minimum_memory",
                memory_budget_bytes=123456,
                precompute_grouped_class_segments=False,
                grouped_schedule_budget_bytes=4096,
                nearfield_schedule_item_cap=2048,
                upward_leaf_batch_size=128,
            ),
        ),
    )
    assert fmm._impl.memory_objective == "minimum_memory"
    assert fmm._impl.memory_budget_bytes == 123456
    assert fmm._impl.precompute_grouped_class_segments is False
    assert fmm._impl.grouped_schedule_budget_bytes == 4096
    assert fmm._impl.nearfield_schedule_item_cap == 2048
    assert fmm._impl.upward_leaf_batch_size == 128


def test_solidfmm_upward_defaults_to_bounded_leaf_batch_size():
    positions, masses = _sample_problem(n=4096)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(upward_leaf_batch_size=None),
        ),
    )

    build = fmm._impl._prepare_state_tree_and_upward(
        positions_arr=positions,
        masses_arr=masses,
        bounds=None,
        leaf_size=16,
        max_order=2,
        refine_local_val=False,
        max_refine_levels_val=0,
        aspect_threshold_val=16.0,
        jit_tree_override=None,
        upward_center_mode="com",
        allow_stateful_cache=False,
    )
    num_internal = int(build.tree.num_internal_nodes)
    num_leaves = int(build.tree.parent.shape[0]) - num_internal
    expected_batch = min(num_leaves, upward_private._DEFAULT_LEAF_BATCH_SIZE)

    recorded: list[int] = []
    original = upward_private._p2m_leaves_complex

    def _recording_p2m(*args, **kwargs):
        recorded.append(int(kwargs["leaf_batch_size"]))
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(upward_private, "_p2m_leaves_complex", _recording_p2m)
        upward_private.prepare_solidfmm_complex_upward_sweep(
            build.tree,
            build.positions_sorted,
            build.masses_sorted,
            max_order=2,
            center_mode="com",
            max_leaf_size=int(build.leaf_cap),
            leaf_batch_size=None,
            rotation="solidfmm",
        )

    assert recorded == [expected_batch]


def test_pair_grouped_mode_skips_class_major_schedule_precompute():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            farfield=FarFieldConfig(mode="pair_grouped", grouped_interactions=True),
            runtime=RuntimePolicyConfig(precompute_grouped_class_segments=True),
        ),
    )

    assert (
        fmm._impl._should_precompute_grouped_class_segments(
            grouped_chunk_size=4096,
            farfield_mode="pair_grouped",
        )
        is False
    )
    assert (
        fmm._impl._should_precompute_grouped_class_segments(
            grouped_chunk_size=4096,
            farfield_mode="class_major",
        )
        is True
    )


def test_minimum_memory_gpu_runtime_does_not_auto_enable_grouped_interactions():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(memory_objective="minimum_memory"),
        ),
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax, "default_backend", lambda: "gpu")
        overrides = fmm._impl._resolve_runtime_execution_overrides(
            num_particles=131072,
        )

    assert overrides.grouped_interactions is False
    assert overrides.farfield_mode == "pair_grouped"


def test_minimum_memory_gpu_runtime_starts_with_smaller_traversal_capacities():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.LARGE_N_GPU,
        basis="solidfmm",
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax, "default_backend", lambda: "gpu")
        overrides = fmm._impl._resolve_runtime_execution_overrides(
            num_particles=524288,
        )

    assert overrides.traversal_config is not None
    assert int(overrides.traversal_config.max_pair_queue) == 32768
    assert int(overrides.traversal_config.process_block) == 64
    assert int(overrides.traversal_config.max_interactions_per_node) == 1024
    assert int(overrides.traversal_config.max_neighbors_per_leaf) == 256


def test_tracing_traversal_config_caps_queue_and_process_block():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )

    capped = fmm._impl._resolve_tracing_traversal_config(
        traversal_config=DualTreeTraversalConfig(
            max_pair_queue=524288,
            process_block=512,
            max_interactions_per_node=8192,
            max_neighbors_per_leaf=2048,
        )
    )

    assert capped is not None
    assert int(capped.max_pair_queue) == 65536
    assert int(capped.process_block) == 128
    assert int(capped.max_interactions_per_node) == 512
    assert int(capped.max_neighbors_per_leaf) == 512


def test_streamed_far_pairs_disables_grouped_interactions_runtime_override():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            farfield=FarFieldConfig(
                grouped_interactions=True,
                mode="class_major",
                streamed_far_pairs=True,
            ),
        ),
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax, "default_backend", lambda: "gpu")
        overrides = fmm._impl._resolve_runtime_execution_overrides(
            num_particles=524288,
        )

    assert overrides.grouped_interactions is False
    assert overrides.farfield_mode == "pair_grouped"


def test_capacity_retry_traversal_settings_clamp_interaction_growth():
    next_cfg, next_queue, next_block = (
        interaction_cache_private._next_retry_traversal_settings(
            traversal_config=DualTreeTraversalConfig(
                max_pair_queue=2_097_152,
                process_block=2048,
                max_interactions_per_node=16384,
                max_neighbors_per_leaf=4096,
            ),
            max_pair_queue=None,
            pair_process_block=None,
        )
    )

    assert next_queue == 4_194_304
    assert next_block == 4096
    assert int(next_cfg.max_interactions_per_node) == 16_384


def test_without_grouped_class_segments_clears_cached_schedule_arrays():
    dummy = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    entry = interaction_cache_private._InteractionCacheEntry(
        key="abc",
        interactions="interactions",
        neighbor_list="neighbors",
        dual_tree_result="walk",
        compact_far_pairs="compact",
        grouped_buffers="grouped",
        grouped_segment_starts=dummy,
        grouped_segment_lengths=dummy,
        grouped_segment_class_ids=dummy,
        grouped_segment_sort_permutation=dummy,
        grouped_segment_group_ids=dummy,
        grouped_segment_unique_targets=dummy,
        grouped_chunk_size=4096,
        nearfield_target_leaf_ids=None,
        nearfield_source_leaf_ids=None,
        nearfield_valid_pairs=None,
        nearfield_chunk_sort_indices=None,
        nearfield_chunk_group_ids=None,
        nearfield_chunk_unique_indices=None,
        nearfield_mode=None,
        nearfield_edge_chunk_size=None,
        nearfield_leaf_cap=None,
    )

    trimmed = interaction_cache_private._without_grouped_class_segments(entry)

    assert trimmed.grouped_buffers == "grouped"
    assert trimmed.grouped_segment_starts is None
    assert trimmed.grouped_segment_lengths is None
    assert trimmed.grouped_segment_class_ids is None
    assert trimmed.grouped_segment_sort_permutation is None
    assert trimmed.grouped_segment_group_ids is None
    assert trimmed.grouped_segment_unique_targets is None
    assert trimmed.grouped_chunk_size is None


def test_minimum_memory_objective_reduces_default_nearfield_schedule_cap():
    balanced = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(memory_objective="balanced"),
        ),
    )
    minimum = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(memory_objective="minimum_memory"),
        ),
    )

    balanced_cap = balanced._impl._resolve_nearfield_schedule_item_cap(
        edge_count=4096,
        leaf_cap=32,
        edge_chunk_size=256,
    )
    minimum_cap = minimum._impl._resolve_nearfield_schedule_item_cap(
        edge_count=4096,
        leaf_cap=32,
        edge_chunk_size=256,
    )

    assert minimum_cap < balanced_cap


def test_memory_budget_limits_default_nearfield_schedule_cap():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(memory_budget_bytes=96),
        ),
    )

    cap = fmm._impl._resolve_nearfield_schedule_item_cap(
        edge_count=1024,
        leaf_cap=32,
        edge_chunk_size=256,
    )

    assert cap == 8


def test_m2l_autotune_cache_roundtrip_api():
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )
    payload = [
        {
            "key": ["gpu", "complex", "float32", 4, "solidfmm", "", 0, 2],
            "chunk_size": 2048,
        }
    ]
    restored = fmm.import_m2l_autotune_cache(payload, merge=False)
    assert restored == 1
    exported = fmm.export_m2l_autotune_cache()
    assert any(int(item.get("chunk_size", -1)) == 2048 for item in exported)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=True) as handle:
        saved = fmm.save_m2l_autotune_cache(handle.name)
        assert saved >= 1
        handle.seek(0)
        raw = json.load(handle)
        assert isinstance(raw, list)
        loaded = fmm.load_m2l_autotune_cache(handle.name, merge=False)
        assert loaded >= 1
