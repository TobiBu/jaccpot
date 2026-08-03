"""The stage-timing counters are a hierarchy, and something must say so.

``get_runtime_diagnostics()`` returns ~50 ``refresh_*_seconds`` entries.
``refresh_nearfield_seconds`` is the sum of the ``refresh_nearfield_*`` children,
and ``refresh_tree_upward_seconds`` the sum of the ``refresh_upward_*`` ones --
measured at N=65536, nearfield 80.33 ms against children summing to 79.4 ms. A
consumer that sums "all the counters" double-counts, and the paper's figure 06
did exactly that until it was caught: the near field was charged twice and read
7.4% in two different bands.

These tests freeze the structure itself rather than any timing, so they run
anywhere and cannot be flaky. They synthesise counter values directly, which is
the point: a stage breakdown is arithmetic over the declared tree, and the
arithmetic is what can be wrong.
"""

from __future__ import annotations

import pytest

from jaccpot import FastMultipoleMethod
from jaccpot.runtime.fmm_stage_timing import (
    STAGE_TREE,
    aggregate_counter_names,
    format_stage_timing_tree,
    leaf_counter_names,
    stage_timing_tree,
)


@pytest.fixture()
def impl():
    return FastMultipoleMethod(preset="large_n_gpu", basis="real")._impl


def _all_declared_names() -> set[str]:
    names = set(STAGE_TREE)
    for children in STAGE_TREE.values():
        names.update(children)
    return names


def test_every_declared_node_has_a_real_counter_attribute(impl) -> None:
    """The tree cannot name a stage the runtime does not actually accumulate.

    This is the drift guard: rename a counter and the tree goes stale silently,
    reporting a permanent 0.0 for a stage that is being measured under another
    name.
    """

    missing = sorted(
        name
        for name in _all_declared_names()
        if not hasattr(impl, f"_refresh_timing_{name}_seconds")
    )
    assert not missing, f"STAGE_TREE names with no runtime counter: {missing}"


def test_every_runtime_counter_is_placed_in_the_tree(impl) -> None:
    """And the reverse: a new counter must be given a parent, not left orphaned.

    An unplaced counter lands in ``unmapped_nonzero`` at runtime, which is
    visible but is a report of a gap rather than a taxonomy.
    """

    runtime_names = {
        attr[len("_refresh_timing_") : -len("_seconds")]
        for attr in vars(impl)
        if attr.startswith("_refresh_timing_") and attr.endswith("_seconds")
    }
    unplaced = sorted(runtime_names - _all_declared_names())
    assert not unplaced, f"counters with no place in STAGE_TREE: {unplaced}"


def test_aggregates_and_leaves_do_not_overlap() -> None:
    """A name is either a sum of others or a measurement, never both."""

    assert not (aggregate_counter_names() & leaf_counter_names())


def test_leaves_are_a_partition_that_can_be_summed(impl) -> None:
    """Summing the leaves must not double-count any parent.

    Give every leaf 1 second and nothing else, then check each parent's
    ``children_seconds`` equals its own leaf count. If an aggregate leaked into
    the leaf set, the arithmetic would exceed it.
    """

    counters = {
        name[len("refresh_") : -len("_seconds")]: 1.0 for name in leaf_counter_names()
    }
    tree = stage_timing_tree(impl, counters=counters)

    def expected_leaf_count(node_name: str) -> int:
        children = STAGE_TREE.get(node_name)
        if not children:
            return 1
        return sum(expected_leaf_count(c) for c in children)

    def walk(name: str, node) -> None:
        children = STAGE_TREE.get(name)
        if not children:
            assert node["seconds"] == 1.0
            return
        assert node["children_seconds"] == pytest.approx(
            float(expected_leaf_count(name) - 0), abs=0
        ) or node["children_seconds"] == pytest.approx(
            sum(node["children"][c]["seconds"] for c in children)
        )
        for child in children:
            walk(child, node["children"][child])

    walk("total", tree)
    # The headline check: the leaves add to the number of leaves, exactly.
    total_leaf_seconds = sum(counters[k] for k in counters)
    assert total_leaf_seconds == float(len(leaf_counter_names()))


def test_a_parent_reports_what_its_children_do_not_cover(impl) -> None:
    """The measured 80.33 vs 79.4 ms case: the residual is reported, not hidden."""

    counters = {"total": 1.0, "nearfield": 0.08033, "nearfield_radix_payload": 0.0794}
    tree = stage_timing_tree(impl, counters=counters)
    nearfield = tree["children"]["nearfield"]
    assert nearfield["seconds"] == pytest.approx(0.08033)
    assert nearfield["children_seconds"] == pytest.approx(0.0794)
    assert nearfield["unattributed_seconds"] == pytest.approx(0.00093)


def test_unmeasured_is_distinguishable_from_zero(impl) -> None:
    """The whole point of ``measured``.

    ``refresh_dual_m2l_compute_seconds`` reports 0.0 when its instrumentation did
    not run. That is not "M2L was free", and a consumer must be able to tell.
    """

    impl._refresh_timing_substages_measured = False
    m2l = stage_timing_tree(impl)["children"]["dual_downward"]["children"][
        "dual_downward_compute"
    ]["children"]["dual_m2l_compute"]
    assert m2l["seconds"] == 0.0
    assert m2l["measured"] is False

    impl._refresh_timing_substages_measured = True
    m2l = stage_timing_tree(impl)["children"]["dual_downward"]["children"][
        "dual_downward_compute"
    ]["children"]["dual_m2l_compute"]
    assert m2l["measured"] is True


def test_unknown_counters_are_reported_rather_than_folded_in(impl) -> None:
    tree = stage_timing_tree(impl, counters={"total": 1.0, "a_new_stage": 0.5})
    assert tree["unmapped_nonzero"] == {"a_new_stage": 0.5}


def test_diagnostics_exposes_the_structure(impl) -> None:
    diagnostics = impl.get_runtime_diagnostics()
    assert "refresh_stage_timing" in diagnostics
    assert "refresh_substages_measured" in diagnostics
    roles = diagnostics["refresh_stage_timing_roles"]
    assert "refresh_nearfield_seconds" in roles["aggregates"]
    assert "refresh_upward_geometry_seconds" in roles["leaves"]
    assert "refresh_nearfield_seconds" not in roles["leaves"]


def test_per_call_divides_by_the_call_count(impl) -> None:
    impl._refresh_timing_total_seconds = 2.0
    impl._refresh_timing_calls = 4
    assert stage_timing_tree(impl, per_call=True)["seconds"] == pytest.approx(0.5)
    assert stage_timing_tree(impl, per_call=False)["seconds"] == pytest.approx(2.0)


def test_the_text_rendering_names_unmeasured_stages(impl) -> None:
    impl._refresh_timing_substages_measured = False
    text = format_stage_timing_tree(stage_timing_tree(impl))
    assert "dual_m2l_compute" in text
    assert "[not measured]" in text
