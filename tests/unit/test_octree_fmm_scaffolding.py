import jax.numpy as jnp

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, FMMPreset, TreeConfig
from jaccpot.runtime._octree_fmm import build_octree_upward_plan


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
