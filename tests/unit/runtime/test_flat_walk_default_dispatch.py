"""The strict streamed seam takes the flat walk BY DEFAULT and falls back sanely.

``_build_dual_tree_artifacts_split_strict_streamed`` (``runtime/_interaction_cache.py``)
routes to the flat-emission walk unless ``JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK=0``.
A configuration the flat walk cannot carry -- the treecode walk requested, a MAC
other than bh/dehnen, a solver-owned pair policy, the non-flat far-pair layout --
falls back to the dual walk QUIETLY while the flag is merely defaulted, and RAISES
when the flag was set to ``1`` explicitly (the caller asked for a walk it cannot
have). Capacities the caller did not name grow eagerly from their floor and the
floor handed over from an earlier eager pass is honoured; a named cap is exact.
Pinned on CPU on a real tree, every commit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("yggdrax")
from yggdrax import DualTreeTraversalConfig
from yggdrax._geometry_impl import compute_tree_geometry
from yggdrax.tree import Tree

import jaccpot.runtime._interaction_cache as ic

_LEAF = 8
_FLAG = "JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK"
_NEAR_CAP = "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP"
_FAR_CAP = "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP"


@pytest.fixture(scope="module")
def tree_and_geometry():
    points = jax.random.uniform(jax.random.PRNGKey(5), (512, 3), dtype=jnp.float64)
    masses = jnp.ones((512,), dtype=jnp.float64)
    tree = Tree.from_particles(points, masses, leaf_size=_LEAF, tree_type="radix")
    geometry = compute_tree_geometry(
        tree.topology, tree.positions_sorted, max_leaf_size=_LEAF
    )
    return tree, geometry


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        _FLAG,
        _NEAR_CAP,
        _FAR_CAP,
        "JACCPOT_STATIC_STRICT_FUSED_TREECODE_WALK",
        "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
    ):
        monkeypatch.delenv(k, raising=False)


def _seam(tree, geometry, *, mac_type="dehnen", report=None, floor=None):
    return ic._build_dual_tree_artifacts_split_strict_streamed(
        tree=tree,
        geometry=geometry,
        theta=0.5,
        mac_type=mac_type,
        dehnen_radius_scale=1.0,
        max_pair_queue=None,
        pair_process_block=None,
        traversal_config=DualTreeTraversalConfig(
            max_pair_queue=1 << 14,
            process_block=256,
            max_interactions_per_node=1024,
            max_neighbors_per_leaf=512,
        ),
        pair_policy=None,
        policy_state=None,
        capacity_report=report,
        flat_walk_capacity_floor=floor,
    )


def _count(monkeypatch, name):
    calls = {"n": 0}
    real = getattr(ic, name)

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ic, name, counting)
    return calls


def test_flat_walk_is_the_default(monkeypatch, tree_and_geometry):
    tree, geometry = tree_and_geometry
    flat = _count(monkeypatch, "_build_flat_walk_artifacts_strict_streamed")
    reports = []
    art = _seam(tree, geometry, report=reports.append)
    assert flat["n"] == 1
    (report,) = reports
    assert report["flat_walk"] is True and report["peak_wavefront"] > 0
    assert report["far_named"] is False and report["near_edge_named"] is False
    # unnamed caps sit at their floors (the 512-particle tree fits them)
    assert report["near_edge_capacity"] == ic._FLAT_WALK_NEAR_EDGE_FLOOR
    assert report["compact_far_pair_capacity"] == ic._STRICT_STREAMED_FAR_PAIR_FLOOR
    assert int(art.compact_far_pairs.far_pair_count) > 0


def test_flag_zero_takes_the_dual_walk(monkeypatch, tree_and_geometry):
    tree, geometry = tree_and_geometry
    flat = _count(monkeypatch, "_build_flat_walk_artifacts_strict_streamed")
    monkeypatch.setenv(_FLAG, "0")
    reports = []
    art = _seam(tree, geometry, report=reports.append)
    assert flat["n"] == 0
    (report,) = reports
    assert not report.get("flat_walk")
    assert int(art.compact_far_pairs.far_pair_count) > 0


@pytest.mark.parametrize(
    "blocker",
    ["treecode", "mac", "layout"],
)
def test_defaulted_flag_falls_back_and_explicit_flag_raises(
    monkeypatch, tree_and_geometry, blocker
):
    tree, geometry = tree_and_geometry
    mac_type = "dehnen"
    if blocker == "treecode":
        monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_TREECODE_WALK", "1")
        # the treecode graft is exercised elsewhere; here only the routing matters
        sentinel = object()
        monkeypatch.setattr(
            ic, "_build_treecode_artifacts_strict_streamed", lambda **k: sentinel
        )
    elif blocker == "mac":
        mac_type = "engblom"
    else:
        monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS", "0")
    flat = _count(monkeypatch, "_build_flat_walk_artifacts_strict_streamed")

    # defaulted: quiet fallback, the flat builder is never entered
    out = _seam(tree, geometry, mac_type=mac_type)
    assert flat["n"] == 0
    if blocker == "treecode":
        assert out is sentinel
    else:
        assert int(out.compact_far_pairs.far_pair_count) > 0 or blocker == "layout"

    # explicit: the caller asked for a walk it cannot have
    monkeypatch.setenv(_FLAG, "1")
    with pytest.raises(RuntimeError, match="JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK"):
        _seam(tree, geometry, mac_type=mac_type)
    assert flat["n"] == 0


def test_named_near_cap_is_exact_and_unnamed_grows(monkeypatch, tree_and_geometry):
    tree, geometry = tree_and_geometry
    monkeypatch.setenv(_NEAR_CAP, "16")
    monkeypatch.setenv(_FAR_CAP, "16")
    with pytest.raises(RuntimeError, match="named it"):
        _seam(tree, geometry)
    monkeypatch.delenv(_NEAR_CAP)
    monkeypatch.delenv(_FAR_CAP)
    monkeypatch.setattr(ic, "_FLAT_WALK_NEAR_EDGE_FLOOR", 16)
    monkeypatch.setattr(ic, "_STRICT_STREAMED_FAR_PAIR_FLOOR", 16)
    reports = []
    _seam(tree, geometry, report=reports.append)
    (report,) = reports
    assert (
        report["near_edge_capacity"] > 16 and report["compact_far_pair_capacity"] > 16
    )
    assert report["grew"], report


def test_capacity_floor_from_an_earlier_pass_is_honoured(
    monkeypatch, tree_and_geometry
):
    tree, geometry = tree_and_geometry
    reports = []
    _seam(
        tree,
        geometry,
        report=reports.append,
        floor={"near_edge_capacity": 1 << 23, "compact_far_pair_capacity": 1 << 19},
    )
    (report,) = reports
    assert report["near_edge_capacity"] == 1 << 23
    assert report["compact_far_pair_capacity"] == 1 << 19
    # a NAMED cap ignores the floor
    monkeypatch.setenv(_NEAR_CAP, str(1 << 15))
    reports.clear()
    _seam(tree, geometry, report=reports.append, floor={"near_edge_capacity": 1 << 23})
    assert reports[0]["near_edge_capacity"] == 1 << 15
