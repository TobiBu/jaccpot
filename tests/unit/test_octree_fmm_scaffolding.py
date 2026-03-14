import jax.numpy as jnp
import numpy as np

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, FMMPreset, TreeConfig
from jaccpot.operators.complex_ops import l2l_complex_batch
from jaccpot.runtime._fmm_impl import _prepare_solidfmm_downward_sweep
from jaccpot.runtime._octree_fmm import (
    accumulate_octree_solidfmm_m2l,
    build_octree_downward_plan,
    build_octree_interaction_plan,
    build_octree_upward_plan,
    prepare_octree_solidfmm_complex_multipoles,
    propagate_octree_solidfmm_l2l,
)
from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_upward_sweep,
)


def _sample_problem(n: int = 48):
    positions = jnp.linspace(-1.0, 1.0, n * 3, dtype=jnp.float32).reshape(n, 3)
    masses = jnp.linspace(1.0, 2.0, n, dtype=jnp.float32)
    return positions, masses


def test_octree_upward_plan_exposes_level_major_metadata():
    positions, masses = _sample_problem()
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

    assert state.octree is not None
    plan = build_octree_upward_plan(state.octree)

    assert int(plan.num_levels) >= 1
    assert plan.nodes_by_level.shape == plan.valid_mask.shape
    assert plan.level_offsets.shape[0] >= int(plan.num_levels) + 1
    assert plan.children.shape[1] == 8
    assert int(plan.num_valid_nodes) >= int(plan.num_leaf_nodes) >= 1
    assert plan.box_centers.shape == (plan.valid_mask.shape[0], 3)
    assert plan.box_half_extents.shape == (plan.valid_mask.shape[0], 3)
    assert plan.box_radii.shape == plan.valid_mask.shape
    assert plan.box_max_extents.shape == plan.valid_mask.shape


def test_octree_execution_view_exposes_native_box_geometry():
    positions, masses = _sample_problem(n=64)
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

    assert state.octree is not None
    valid_mask = np.asarray(state.octree.valid_mask)
    box_centers = np.asarray(state.octree.box_centers)
    box_half_extents = np.asarray(state.octree.box_half_extents)
    box_radii = np.asarray(state.octree.box_radii)
    box_max_extents = np.asarray(state.octree.box_max_extents)

    assert box_centers.shape == (valid_mask.shape[0], 3)
    assert box_half_extents.shape == (valid_mask.shape[0], 3)
    assert box_radii.shape == valid_mask.shape
    assert box_max_extents.shape == valid_mask.shape
    assert np.all(box_radii[valid_mask] > 0.0)
    assert np.all(box_max_extents[valid_mask] > 0.0)


def test_octree_complex_multipoles_match_radix_upward_on_mapped_nodes():
    positions, masses = _sample_problem(n=64)
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

    assert state.octree is not None
    plan = build_octree_upward_plan(state.octree)
    octree_upward = prepare_octree_solidfmm_complex_multipoles(
        plan,
        state.positions_sorted,
        state.masses_sorted,
        max_order=3,
    )
    radix_upward = prepare_solidfmm_complex_upward_sweep(
        state.tree,
        state.positions_sorted,
        state.masses_sorted,
        max_order=3,
        max_leaf_size=8,
    )

    root_oct = int(np.asarray(state.octree.radix_node_to_oct)[0])
    radix_nodes = np.asarray(state.octree.radix_node_to_oct)
    unique_oct, inverse, counts = np.unique(
        radix_nodes,
        return_inverse=True,
        return_counts=True,
    )
    del unique_oct
    unique_radix_mask = counts[inverse] == 1
    radix_ranges = np.asarray(state.tree.node_ranges)
    oct_ranges = np.asarray(state.octree.node_ranges)[radix_nodes]
    matching_range_mask = np.all(radix_ranges == oct_ranges, axis=1)
    comparable_mask = unique_radix_mask & matching_range_mask

    assert np.allclose(
        np.asarray(octree_upward.centers)[root_oct],
        np.asarray(radix_upward.multipoles.centers)[0],
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.allclose(
        np.asarray(octree_upward.packed)[root_oct],
        np.asarray(radix_upward.multipoles.packed)[0],
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.allclose(
        np.asarray(octree_upward.centers)[radix_nodes[comparable_mask]],
        np.asarray(radix_upward.multipoles.centers)[comparable_mask],
        rtol=1e-6,
        atol=1e-6,
    )


def test_prepare_state_attaches_octree_native_upward_artifacts():
    positions, masses = _sample_problem(n=64)
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

    assert state.octree is not None
    assert state.octree_upward is not None

    plan = build_octree_upward_plan(state.octree)
    expected = prepare_octree_solidfmm_complex_multipoles(
        plan,
        state.positions_sorted,
        state.masses_sorted,
        max_order=3,
    )
    root_oct = int(np.asarray(state.octree.radix_node_to_oct)[0])

    assert np.allclose(
        np.asarray(state.octree_upward.centers),
        np.asarray(expected.centers),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.allclose(
        np.asarray(state.octree_upward.packed)[root_oct],
        np.asarray(expected.packed)[root_oct],
        rtol=1e-5,
        atol=1e-5,
    )


def test_octree_interaction_plan_remaps_farfield_pairs_by_level():
    positions, masses = _sample_problem(n=64)
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

    assert state.octree is not None
    assert state.interactions is not None

    plan = build_octree_interaction_plan(state.octree, state.interactions)
    num_pairs = int(np.asarray(plan.num_pairs))
    valid_mask = np.asarray(plan.valid_mask)
    target_nodes = np.asarray(plan.target_nodes)
    target_levels = np.asarray(plan.target_levels)
    counts = np.asarray(plan.counts)
    level_offsets = np.asarray(plan.level_offsets)

    assert num_pairs > 0
    assert valid_mask.shape == target_nodes.shape
    assert counts.sum() == num_pairs
    assert level_offsets[-1] == num_pairs
    assert np.all(target_levels[:num_pairs][1:] >= target_levels[:num_pairs][:-1])
    assert np.all(target_nodes[:num_pairs] >= 0)


def test_prepare_state_attaches_octree_native_downward_scaffold():
    positions, masses = _sample_problem(n=64)
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

    assert state.octree is not None
    assert state.octree_upward is not None
    assert state.octree_downward is not None

    interaction_plan = build_octree_interaction_plan(state.octree, state.interactions)
    expected = build_octree_downward_plan(
        state.octree,
        state.octree_upward,
        interaction_plan,
    )

    root_oct = int(np.asarray(state.octree.radix_node_to_oct)[0])
    parent = np.asarray(state.octree_downward.parent)
    children = np.asarray(state.octree.children)

    assert parent[root_oct] == -1
    assert state.octree_downward.locals_packed.shape == state.octree_upward.packed.shape
    assert np.allclose(
        np.asarray(state.octree_downward.centers),
        np.asarray(state.octree_upward.centers),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.array_equal(
        np.asarray(state.octree_downward.valid_interactions),
        np.asarray(expected.valid_interactions),
    )
    assert int(np.asarray(state.octree_downward.num_pairs)) == int(
        np.asarray(expected.num_pairs)
    )

    valid_children = children[root_oct][children[root_oct] >= 0]
    assert valid_children.size > 0
    assert np.all(parent[valid_children] == root_oct)


def test_octree_m2l_matches_radix_root_local():
    positions, masses = _sample_problem(n=64)
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

    assert state.octree is not None
    assert state.octree_upward is not None
    assert state.octree_downward is not None

    octree_m2l = accumulate_octree_solidfmm_m2l(
        state.octree_downward,
        state.octree_upward,
        chunk_size=128,
    )
    radix_downward = _prepare_solidfmm_downward_sweep(
        state.tree,
        state.upward,
        theta=float(state.theta),
        mac_type="bh",
        interactions=state.interactions,
        m2l_chunk_size=128,
        basis_mode="complex",
        complex_rotation="solidfmm",
    )

    root_oct = int(np.asarray(state.octree.radix_node_to_oct)[0])

    assert np.allclose(
        np.asarray(octree_m2l.locals_packed)[root_oct],
        np.asarray(radix_downward.locals.coefficients)[0],
        rtol=1e-5,
        atol=1e-5,
    )


def test_octree_l2l_propagation_updates_children_consistently():
    positions, masses = _sample_problem(n=64)
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

    assert state.octree is not None
    assert state.octree_upward is not None
    assert state.octree_downward is not None
    assert state.interactions is not None

    octree_downward = accumulate_octree_solidfmm_m2l(
        state.octree_downward,
        state.octree_upward,
        chunk_size=128,
    )
    octree_downward = propagate_octree_solidfmm_l2l(
        octree_downward,
        state.octree,
    )
    m2l_only = np.asarray(octree_downward.locals_packed)
    propagated = propagate_octree_solidfmm_l2l(
        octree_downward,
        state.octree,
    )

    children = np.asarray(state.octree.children)
    root_oct = int(np.asarray(state.octree.radix_node_to_oct)[0])
    valid_children = children[root_oct][children[root_oct] >= 0]

    assert np.allclose(
        np.asarray(propagated.locals_packed)[root_oct],
        m2l_only[root_oct],
        rtol=1e-6,
        atol=1e-6,
    )
    assert valid_children.size > 0

    parent_coeffs = np.broadcast_to(
        np.asarray(propagated.locals_packed)[root_oct][None, :],
        (valid_children.shape[0], np.asarray(propagated.locals_packed).shape[1]),
    )
    deltas = (
        np.asarray(state.octree_upward.centers)[valid_children]
        - np.asarray(state.octree_upward.centers)[root_oct]
    )
    translated = np.asarray(
        l2l_complex_batch(
            jnp.asarray(parent_coeffs),
            jnp.asarray(deltas),
            order=3,
            rotation="solidfmm",
        )
    )

    child_delta = (
        np.asarray(propagated.locals_packed)[valid_children] - m2l_only[valid_children]
    )
    assert np.allclose(
        child_delta,
        translated,
        rtol=1e-5,
        atol=1e-5,
    )
