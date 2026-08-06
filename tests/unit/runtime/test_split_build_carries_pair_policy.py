"""The split/streamed dual-tree build must carry the pair policy, not route around it.

``_can_split_dual_tree_build`` used to return false whenever a ``pair_policy`` or
``policy_state`` was installed, so the Dehnen mass-dependent MAC was shut out of
the minimum-memory streamed build and could only ever run the monolithic one.
That matters at N >= 10^7, where the monolithic build's node-interaction buffers
(``num_nodes x max_interactions_per_node``) are the binding memory constraint.

All three yggdrax entry points the split build calls
(``build_compact_far_pairs_and_leaf_neighbor_lists``,
``build_interactions_and_neighbors_split``, ``build_leaf_neighbor_lists``)
already accepted ``pair_policy``; jaccpot never passed it. The failure mode of
getting that plumbing wrong is silent: the split build would answer the plain
geometric MAC while the caller asked for the criterion, producing a *cheaper*
run with a different force and no signal at all. So the two assertions here are

* the split build's accept mask is **bit-identical** to the monolithic build's on
  the same inputs -- a differing mask means the two builds disagree about the
  criterion, and

* the split build's mask **differs from the same split build with no policy** --
  without this, a dropped policy would pass the first assertion trivially by
  making both builds geometric.

The far/near partition invariant is re-checked through the split path as well,
reusing :mod:`tests.unit.runtime.test_far_near_partition`'s own coverage counter
so the two cannot drift apart.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot.config import FMMAdvancedConfig
from jaccpot.runtime._adaptive_policy import adaptive_pair_policy
from jaccpot.runtime._interaction_cache import (
    _build_dual_tree_artifacts,
    _can_split_dual_tree_build,
)
from jaccpot.solver import FastMultipoleMethod

from ._reproducibility import assert_same_accept_mask
from .test_far_near_partition import LEAF, ORDER, _coverage_counts, _problem

# Compile-bound: every test here builds solvers and runs full traversals, so it
# belongs behind `-m "not slow"` in the version-compatibility matrix for the same
# reason the sibling MAC tests do. See test_dehnen_mac_gradients.py.
pytestmark = pytest.mark.slow

EPS = 3e-3
THETA = 0.6

_TRAVERSAL = DualTreeTraversalConfig(
    max_pair_queue=131072,
    process_block=512,
    max_interactions_per_node=8192,
    max_neighbors_per_leaf=4096,
)


class _StateShim:
    """Just enough of a prepared state for ``_coverage_counts``."""

    def __init__(self, tree, interactions, neighbor_list):
        self.tree = tree
        self.interactions = interactions
        self.neighbor_list = neighbor_list


def _deep_problem(n: int = 2048, seed: int = 0):
    """A configuration with a real far field.

    ``test_far_near_partition``'s N=512 / leaf 16 gives 32 leaves, where eq (16a)
    accepts a handful of pairs at most (trap 5: the criterion is very
    conservative at small N). That is enough for the coverage invariant, which is
    about bookkeeping, but far too thin to conclude that a mask matches for the
    right reason -- so the mask comparison runs here as well, at 256 leaves.
    """

    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(n, 3))
    mass = rng.uniform(0.5, 1.5, size=n)
    return (
        jnp.asarray(pos, dtype=jnp.float64),
        jnp.asarray(mass, dtype=jnp.float64),
    )


def _criterion_solver():
    advanced = FMMAdvancedConfig()
    advanced = replace(
        advanced,
        mac_type="dehnen_error",
        runtime=replace(
            advanced.runtime,
            retain_traversal_result=True,
            retain_interactions=True,
        ),
    )
    return FastMultipoleMethod(theta=THETA, adaptive_eps=EPS, advanced=advanced)


def _policy_inputs(positions, masses, *, leaf: int = LEAF):
    """Return (engine, tree_artifacts, policy_state) for a real criterion run.

    The force scale comes from the solver's own paper prepass rather than an
    invented constant, so the policy under test is the one production installs.
    """

    fmm = _criterion_solver()
    fmm.prepare_state(positions, masses, leaf_size=leaf, max_order=ORDER)
    engine = getattr(fmm, "_impl", fmm)
    force_scale = engine._last_force_scale_nodes
    assert force_scale is not None, "the paper prepass produced no force scale"

    tree_artifacts = engine._prepare_state_tree_and_upward(
        positions_arr=positions,
        masses_arr=masses,
        bounds=None,
        leaf_size=leaf,
        max_order=ORDER,
        refine_local_val=False,
        max_refine_levels_val=0,
        aspect_threshold_val=16.0,
        jit_tree_override=None,
        upward_center_mode="com",
        allow_stateful_cache=False,
    )
    policy_state = engine._build_adaptive_policy_state(
        upward=tree_artifacts.upward,
        tree=tree_artifacts.tree,
        positions_sorted=tree_artifacts.positions_sorted,
        p_gears=(int(tree_artifacts.upward.multipoles.order),),
        force_scale_nodes=jnp.asarray(
            force_scale, dtype=tree_artifacts.positions_sorted.dtype
        ),
        eps=jnp.asarray(EPS, dtype=tree_artifacts.positions_sorted.dtype),
        theta=jnp.asarray(THETA, dtype=tree_artifacts.positions_sorted.dtype),
        error_model_code=jnp.asarray(
            engine._traversal_policy_error_model_code(), dtype=jnp.int32
        ),
        dehnen_geometry_mode=engine.dehnen_geometry_mode,
    )
    return engine, tree_artifacts, policy_state


def _build(engine, tree_artifacts, *, split: bool, policy_state):
    artifacts, _ = _build_dual_tree_artifacts(
        tree_artifacts.tree,
        tree_artifacts.upward.geometry,
        theta=THETA,
        mac_type=engine._base_mac_type(),
        dehnen_radius_scale=engine.dehnen_radius_scale,
        cache_key=None,
        cache_entry=None,
        max_pair_queue=None,
        pair_process_block=None,
        traversal_config=_TRAVERSAL,
        retry_logger=None,
        fail_fast=False,
        use_dense_interactions=False,
        grouped_interactions=False,
        grouped_chunk_size=None,
        need_traversal_result=False,
        need_compact_far_pairs=False,
        need_node_interactions=True,
        precompute_grouped_class_segments=False,
        grouped_schedule_budget_bytes=None,
        allow_split_build=bool(split),
        pair_policy=None if policy_state is None else adaptive_pair_policy,
        policy_state=policy_state,
        jit_traversal=False,
    )
    return artifacts


def _mask(artifacts) -> tuple[tuple[int, int], ...]:
    interactions = artifacts.interactions
    assert interactions is not None, "build returned no node interaction list"
    src = np.asarray(interactions.sources).ravel()
    tgt = np.asarray(interactions.targets).ravel()
    keep = (src >= 0) & (tgt >= 0)
    return tuple(sorted(zip(tgt[keep].tolist(), src[keep].tolist())))


def test_the_split_build_is_eligible_under_a_pair_policy():
    """The gate itself: a policy must no longer route away from the split build."""

    assert _can_split_dual_tree_build(
        split_enabled=True,
        grouped_interactions=False,
        need_traversal_result=False,
    )


@pytest.mark.parametrize(
    "case",
    ["uniform", "clustered", "deep"],
)
def test_split_and_monolithic_builds_agree_on_the_accept_mask(case):
    if case == "deep":
        positions, masses = _deep_problem()
        leaf = 8
    else:
        positions, masses = _problem(clustered=(case == "clustered"))
        leaf = LEAF
    engine, tree_artifacts, policy_state = _policy_inputs(positions, masses, leaf=leaf)

    monolithic = _mask(
        _build(engine, tree_artifacts, split=False, policy_state=policy_state)
    )
    split = _mask(_build(engine, tree_artifacts, split=True, policy_state=policy_state))
    geometric_split = _mask(
        _build(engine, tree_artifacts, split=True, policy_state=None)
    )

    assert len(split) > 0, (
        "the criterion accepted nothing, so this configuration cannot see a "
        "dropped policy (trap 5: eq (16a) is very conservative at small N)"
    )
    assert split != geometric_split, (
        f"the split build with a policy accepted the same {len(split)} pairs as "
        "the split build without one: the policy is being dropped, not carried"
    )
    assert_same_accept_mask(
        split,
        monolithic,
        err_msg=(
            f"split build accepted {len(split)} far pairs, monolithic accepted "
            f"{len(monolithic)}: the two builds disagree about the criterion."
        ),
    )


@pytest.mark.parametrize("clustered", [False, True])
def test_the_far_near_partition_survives_the_split_build_with_a_policy(clustered):
    """Every source particle covered exactly once, per target leaf.

    The invariant that catches a dropped pair -- and eq (16a) is asymmetric in
    target<->source, so a policy-driven leaf-leaf disagreement is exactly how
    pairs go missing.
    """

    positions, masses = _problem(clustered=clustered)
    engine, tree_artifacts, policy_state = _policy_inputs(positions, masses)
    artifacts = _build(engine, tree_artifacts, split=True, policy_state=policy_state)

    counts = _coverage_counts(
        _StateShim(tree_artifacts.tree, artifacts.interactions, artifacts.neighbor_list)
    )
    missing = int(np.sum(counts == 0))
    doubled = int(np.sum(counts > 1))
    assert missing == 0, (
        f"{missing} (target leaf, source particle) incidences receive neither an "
        "M2L nor a P2P contribution through the split build"
    )
    assert doubled == 0, f"{doubled} incidences are counted more than once"
