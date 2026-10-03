"""The fused lane's static shapes must not depend on an eager visit having happened.

On one device an eager ``prepare_state`` always precedes the traced refresh, so the
level-shape registry is warm. Inside ``shard_map`` it never does, and the failure is
silent: the level width falls back to ``num_internal`` and the per-level Pallas
cascades are deselected with nothing raised. These tests pin both the failure and
the fix.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime import _level_shapes as level_shapes
from jaccpot.runtime.capacity_plan import (
    FusedCapacityPlan,
    fused_capacity_plan,
    fused_capacity_plan_overrides,
    merge_plans,
    plan_from_registry,
)
from jaccpot.runtime.kernels._l2l import _l2l_level_compact_kwargs


@pytest.fixture
def cold_registry(monkeypatch):
    """A process-level registry with nothing recorded, as under ``shard_map``."""
    monkeypatch.setattr(level_shapes, "_WIDTHS", {})
    monkeypatch.setattr(level_shapes, "_LEVELS", {})
    yield


def _cell_tree(n, seed, leaf_size=64, leaf_capacity=512):
    from yggdrax._tree_impl import build_static_cells_tree
    from yggdrax.bounds import infer_bounds

    rng = np.random.default_rng(seed)
    r = 1.0 / np.sqrt(rng.random(n) ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    s = np.sqrt(1.0 - mu * mu)
    pos = jnp.asarray(
        np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1),
        jnp.float32,
    )
    mass = jnp.full((n,), 1.0 / n, jnp.float32)
    topology = build_static_cells_tree(
        pos, mass, infer_bounds(pos), leaf_size=leaf_size, leaf_capacity=leaf_capacity
    )
    # Wrap the impl topology in the PUBLIC tree, as every production caller does:
    # `_l2l_level_compact_kwargs` is annotated `tree: yggdrax.tree.Tree`, and the
    # runtime type-check job (JACCPOT_RUNTIME_TYPECHECK=1) rejects the bare impl
    # RadixTree the helper used to return. The wrapper delegates every field.
    from yggdrax.tree import RadixTree

    return RadixTree(topology=topology, build_mode="static_radix")


def _shape_of(tree):
    return int(np.asarray(tree.node_ranges).shape[0]), int(tree.left_child.shape[0])


def _traced_kwargs(tree, total_nodes, num_internal):
    """``_l2l_level_compact_kwargs`` as the traced refresh sees it."""
    captured: dict = {}

    def body(t):
        captured.update(
            _l2l_level_compact_kwargs(
                t, total_nodes=total_nodes, num_internal=num_internal
            )
        )
        return jnp.zeros(())

    jax.eval_shape(body, tree)
    return captured


@pytest.mark.usefixtures("cold_registry")
def test_a_cold_registry_silently_deselects_the_pallas_cascade(monkeypatch):
    """The defect, as a negative control.

    If this ever stops failing the way it does, the capacity plan may be
    reconsidered -- until then it is the only thing standing between
    ``shard_map`` and the pre-Pallas level loops.
    """
    monkeypatch.setenv("JACCPOT_CASCADE_PALLAS", "1")
    monkeypatch.setenv("JACCPOT_CASCADE_PALLAS_INTERPRET", "1")
    tree = _cell_tree(4000, 0)
    total_nodes, num_internal = _shape_of(tree)

    cold = _traced_kwargs(tree, total_nodes, num_internal)

    assert "pallas_levels" not in cold, "cold registry unexpectedly kept the cascade"
    assert (
        cold["level_batch_width"] == num_internal
    ), "cold registry should fall back to the nodes x depth width"


@pytest.mark.usefixtures("cold_registry")
def test_an_installed_plan_keeps_the_pallas_cascade_on_a_cold_registry(monkeypatch):
    """The gate. Assert the cascade is SELECTED -- timing alone would not notice."""
    monkeypatch.setenv("JACCPOT_CASCADE_PALLAS", "1")
    monkeypatch.setenv("JACCPOT_CASCADE_PALLAS_INTERPRET", "1")
    tree = _cell_tree(4000, 0)
    total_nodes, num_internal = _shape_of(tree)

    # Plan the way the distributed driver will: one eager pass, then read it back.
    _l2l_level_compact_kwargs(tree, total_nodes=total_nodes, num_internal=num_internal)
    plan = plan_from_registry(total_nodes=total_nodes, num_internal=num_internal)
    assert plan is not None
    level_shapes._WIDTHS.clear()
    level_shapes._LEVELS.clear()

    with fused_capacity_plan_overrides(plan):
        planned = _traced_kwargs(tree, total_nodes, num_internal)

    assert "pallas_levels" in planned
    assert planned["pallas_levels"] == plan.num_levels
    assert planned["level_batch_width"] == plan.level_batch_width
    assert plan.level_batch_width < num_internal


@pytest.mark.usefixtures("cold_registry")
def test_the_eager_and_traced_arms_compile_one_width_under_a_plan(monkeypatch):
    """A planning pass and the refresh it plans for must agree, or they retrace apart."""
    monkeypatch.setenv("JACCPOT_CASCADE_PALLAS", "1")
    monkeypatch.setenv("JACCPOT_CASCADE_PALLAS_INTERPRET", "1")
    tree = _cell_tree(4000, 0)
    total_nodes, num_internal = _shape_of(tree)

    # A plan deliberately WIDER than this tree needs: the eager arm must not
    # narrow it back to the measured width.
    plan = FusedCapacityPlan(
        total_nodes=total_nodes,
        num_internal=num_internal,
        level_batch_width=num_internal - 1,
        num_levels=40,
        upward_num_levels=40,
    )
    with fused_capacity_plan_overrides(plan):
        eager = _l2l_level_compact_kwargs(
            tree, total_nodes=total_nodes, num_internal=num_internal
        )
        traced = _traced_kwargs(tree, total_nodes, num_internal)

    assert eager["level_batch_width"] == plan.level_batch_width
    assert traced["level_batch_width"] == plan.level_batch_width


@pytest.mark.usefixtures("cold_registry")
def test_the_overflow_guard_is_live_under_a_plan_and_dead_without_one():
    """``level_width_overflow`` returns a constant ``False`` on a cold registry."""
    tree = _cell_tree(4000, 0)
    total_nodes, num_internal = _shape_of(tree)
    offsets = tree.level_offsets

    dead = level_shapes.level_width_overflow(
        offsets, total_nodes=total_nodes, num_internal=num_internal
    )
    assert not bool(dead)  # vacuously false: nothing registered to compare against

    narrow = FusedCapacityPlan(
        total_nodes=total_nodes,
        num_internal=num_internal,
        level_batch_width=1,
        num_levels=8,
        upward_num_levels=8,
    )
    with fused_capacity_plan_overrides(narrow):
        live = level_shapes.level_width_overflow(
            offsets, total_nodes=total_nodes, num_internal=num_internal
        )
    assert bool(live), "a plan narrower than the tree must trip the guard"


def test_merge_plans_covers_the_worst_device():
    """One program serves every device, so each shape covers the widest shard."""
    shards = [_cell_tree(4000, seed) for seed in (0, 1, 2)]
    total_nodes, num_internal = _shape_of(shards[0])

    plans = []
    for tree in shards:
        level_shapes._WIDTHS.clear()
        level_shapes._LEVELS.clear()
        _l2l_level_compact_kwargs(
            tree, total_nodes=total_nodes, num_internal=num_internal
        )
        plans.append(
            plan_from_registry(total_nodes=total_nodes, num_internal=num_internal)
        )
    level_shapes._WIDTHS.clear()
    level_shapes._LEVELS.clear()

    merged = merge_plans(plans)
    assert merged.level_batch_width == max(p.level_batch_width for p in plans)
    assert merged.num_levels == max(p.num_levels for p in plans)
    # The shards genuinely differ -- which is why a single-device plan under-sizes.
    assert len({p.level_batch_width for p in plans}) > 1


def test_merge_plans_refuses_mismatched_tree_shapes():
    """Devices disagreeing about leaf_capacity cannot share one compiled program."""
    a = FusedCapacityPlan(
        total_nodes=1023,
        num_internal=511,
        level_batch_width=250,
        num_levels=33,
        upward_num_levels=33,
    )
    b = FusedCapacityPlan(
        total_nodes=2047,
        num_internal=1023,
        level_batch_width=250,
        num_levels=33,
        upward_num_levels=33,
    )
    with pytest.raises(ValueError, match="different tree shapes"):
        merge_plans([a, b])
    with pytest.raises(ValueError, match="at least one plan"):
        merge_plans([])


def test_a_plan_applies_only_to_its_own_tree_shape():
    """Installing a plan must not reshape an unrelated tree."""
    plan = FusedCapacityPlan(
        total_nodes=1023,
        num_internal=511,
        level_batch_width=250,
        num_levels=33,
        upward_num_levels=33,
    )
    with fused_capacity_plan_overrides(plan):
        assert fused_capacity_plan() is plan
        assert (
            level_shapes.registered_num_levels(total_nodes=1023, num_internal=511) == 33
        )
        assert (
            level_shapes.registered_num_levels(total_nodes=99, num_internal=49) is None
        )
    assert fused_capacity_plan() is None


def test_the_fingerprint_changes_with_the_shapes():
    """A ContextVar does not retrace, so the plan has to reach the cache key."""
    plan = FusedCapacityPlan(
        total_nodes=1023,
        num_internal=511,
        level_batch_width=250,
        num_levels=33,
        upward_num_levels=36,
    )
    assert plan.fingerprint() == (1023, 511, 250, 33, 36)
    assert plan.widened_to(level_batch_width=300).fingerprint() != plan.fingerprint()
    assert plan.widened_to(level_batch_width=100) is plan  # never narrows


def test_the_upward_depth_bound_comes_from_the_plan_when_one_is_installed():
    """``_resolve_upward_num_levels`` has the same cold-stash problem as the registry.

    Its stash degrades to ``None`` rather than to a wrong kernel, so callers fall
    back to the padded shape-derived depth: correct but slower. Under ``shard_map``
    the bound must still cover every device, so a plan overrides it.
    """
    from jaccpot.runtime.fmm_sweeps import _planned_upward_num_levels

    tree = _cell_tree(4000, 0)
    total_nodes, num_internal = _shape_of(tree)

    assert _planned_upward_num_levels(tree) is None

    plan = FusedCapacityPlan(
        total_nodes=total_nodes,
        num_internal=num_internal,
        level_batch_width=250,
        num_levels=33,
        upward_num_levels=41,
    )
    with fused_capacity_plan_overrides(plan):
        assert _planned_upward_num_levels(tree) == 41

    # A plan for another tree shape must not answer for this one.
    other = FusedCapacityPlan(
        total_nodes=total_nodes * 2 + 1,
        num_internal=num_internal * 2 + 1,
        level_batch_width=250,
        num_levels=33,
        upward_num_levels=41,
    )
    with fused_capacity_plan_overrides(other):
        assert _planned_upward_num_levels(tree) is None


def test_the_two_level_counts_are_distinct_fields():
    """``num_levels`` (level table) and ``upward_num_levels`` (tree depth) differ.

    Substituting one for the other truncates a sweep silently, so the plan keeps
    them apart and the merge maxes each independently.
    """
    a = FusedCapacityPlan(
        total_nodes=1023,
        num_internal=511,
        level_batch_width=100,
        num_levels=30,
        upward_num_levels=44,
    )
    b = FusedCapacityPlan(
        total_nodes=1023,
        num_internal=511,
        level_batch_width=120,
        num_levels=35,
        upward_num_levels=40,
    )
    merged = merge_plans([a, b])
    assert merged.level_batch_width == 120
    assert merged.num_levels == 35
    assert merged.upward_num_levels == 44


def test_merged_walk_caps_cover_the_worst_shard():
    """Each eager prepare overwrites the engine's record; the merge keeps the maxima."""
    from types import SimpleNamespace

    from jaccpot.runtime.capacity_plan import install_walk_caps, merge_walk_caps

    a = {
        "flat_walk": True,
        "queue_capacity": 1 << 17,
        "peak_wavefront": 85_000,
        "compact_far_pair_capacity": 1 << 21,
        "near_edge_capacity": 1 << 22,
    }
    b = {
        "flat_walk": True,
        "queue_capacity": 1 << 16,
        "peak_wavefront": 120_000,
        "compact_far_pair_capacity": 1 << 21,
        "near_edge_capacity": 1 << 23,
    }
    merged = merge_walk_caps([a, None, b])
    assert merged["peak_wavefront"] == 120_000
    assert merged["queue_capacity"] == 1 << 17
    assert merged["near_edge_capacity"] == 1 << 23
    assert merge_walk_caps([None]) is None
    with pytest.raises(ValueError, match="flat_walk"):
        merge_walk_caps([a, dict(b, flat_walk=False)])
    engine = SimpleNamespace(_strict_fused_validated_caps=b)
    install_walk_caps(SimpleNamespace(_impl=engine), merged)
    assert engine._strict_fused_validated_caps["peak_wavefront"] == 120_000


def _plan_for_tree(tree, *, width, levels):
    total_nodes, num_internal = _shape_of(tree)
    return FusedCapacityPlan(
        total_nodes=total_nodes,
        num_internal=num_internal,
        level_batch_width=width,
        num_levels=levels,
        upward_num_levels=levels,
    )


def _widest_and_depth(tree):
    offs = np.asarray(tree.level_offsets)
    return int(np.max(np.diff(offs))), int(np.max(np.asarray(tree.node_level))) + 1


def test_the_mesh_flag_sees_a_tree_that_outgrew_the_plan():
    """``plan_level_overflow``: what the mesh evaluator ORs into its local flag.

    On two A100s a plan 39 nodes narrower than one level of a shard's tree lost that
    level from both cascades -- one particle with its near field and no far field,
    rel-L2 4.2e-2 -- and every capacity flag read False.
    """
    from jaccpot.runtime.capacity_plan import plan_level_overflow

    tree = _cell_tree(4000, 0)
    widest, depth = _widest_and_depth(tree)
    assert not bool(plan_level_overflow(tree)), "no plan installed: nothing to outgrow"
    fits = _plan_for_tree(tree, width=widest, levels=depth)
    with fused_capacity_plan_overrides(fits):
        assert not bool(plan_level_overflow(tree)), "exactly covered is not an overflow"
    for narrow in (
        _plan_for_tree(tree, width=widest - 1, levels=depth),
        _plan_for_tree(tree, width=widest, levels=depth - 1),
    ):
        with fused_capacity_plan_overrides(narrow):
            assert bool(plan_level_overflow(tree)), narrow
            # and under trace, where the mesh evaluator calls it
            traced = jax.jit(plan_level_overflow)(tree)
        assert bool(traced)


def test_the_plan_is_measured_on_the_tree_the_refresh_builds(cold_registry):
    """The eager tree lives in the shard's own box; the refresh rebuilds it in the
    GLOBAL box. Rebuilt the way the refresh will, the registry must cover that tree."""
    from types import SimpleNamespace

    from yggdrax._tree_impl import build_static_cells_tree
    from yggdrax.tree import rebuild_static_radix_tree_from_template

    from jaccpot.distributed.fused import cube_bounds
    from jaccpot.runtime.capacity_plan import _lift_registry_to_refresh_tree

    rng = np.random.default_rng(3)
    n = 3000
    pos = jnp.asarray(rng.standard_normal((n, 3)), jnp.float32)
    mass = jnp.full((n,), 1.0 / n, jnp.float32)
    own = cube_bounds(jnp.min(pos, 0), jnp.max(pos, 0), pad=1e-6)
    # the mesh box: the other device's particles reach much further out
    mesh = cube_bounds(jnp.min(pos, 0) - 40.0, jnp.max(pos, 0) + 3.0, pad=1e-6)
    template = build_static_cells_tree(pos, mass, own, leaf_size=16, leaf_capacity=1024)
    total_nodes, num_internal = _shape_of(template)
    rebuilt = rebuild_static_radix_tree_from_template(
        pos,
        mass,
        template,
        bounds=mesh,
        return_reordered=True,
        leaf_partition="cells",
        return_overflow=True,
    )[0]
    w_own, d_own = _widest_and_depth(template)
    w_mesh, d_mesh = _widest_and_depth(rebuilt)
    assert (w_mesh, d_mesh) != (w_own, d_own), "vacuous: the box changed nothing"

    depths = []
    impl = SimpleNamespace(
        _tree_leaf_partition="cells",
        _tree_cell_min_level=0,
        _resolve_upward_num_levels=lambda t: depths.append(_widest_and_depth(t)[1]),
    )
    level_shapes.level_batch_width(
        template.level_offsets, total_nodes=total_nodes, num_internal=num_internal
    )  # what the eager prepare records
    _lift_registry_to_refresh_tree(
        impl, template, pos, mass, bounds=mesh, num_valid=None
    )
    width = level_shapes.registered_level_batch_width(
        total_nodes=total_nodes, num_internal=num_internal
    )
    levels = level_shapes.registered_num_levels(
        total_nodes=total_nodes, num_internal=num_internal
    )
    assert width >= max(w_own, w_mesh), (width, w_own, w_mesh)
    assert levels >= max(d_own, d_mesh)
    assert depths == [d_mesh], "the upward depth stash must see the refresh tree"


def test_the_host_mesh_box_ignores_padding_and_matches_the_traced_reduction():
    from jaccpot.distributed.fused import cube_bounds, host_mesh_bounds

    rng = np.random.default_rng(0)
    a = rng.standard_normal((50, 3)).astype(np.float32)
    b = (rng.standard_normal((50, 3)) + 5.0).astype(np.float32)
    a[40:] = 1e6  # padding rows: never part of the box
    lo, hi = host_mesh_bounds([a, b], [40, 50])
    live = np.concatenate([a[:40], b])
    ref_lo, ref_hi = cube_bounds(
        jnp.asarray(live.min(0)), jnp.asarray(live.max(0)), pad=1e-6
    )
    np.testing.assert_array_equal(np.asarray(lo), np.asarray(ref_lo))
    np.testing.assert_array_equal(np.asarray(hi), np.asarray(ref_hi))
