"""The refresh stage-timing counters, and which of them are sums of others.

``get_runtime_diagnostics()`` returns ~50 ``refresh_*_seconds`` counters. They
are a **hierarchy, not a partition**: ``refresh_nearfield_seconds`` is the sum of
the ``refresh_nearfield_*`` children, and ``refresh_tree_upward_seconds`` the sum
of the ``refresh_upward_*`` ones. Measured at N=65536: nearfield 80.33 ms against
its children summing to 79.4 ms.

Any consumer that sums "all the counters" therefore double-counts, and the first
version of the paper's figure 06 did exactly that -- charging the near field
twice and reading 7.4% in two different bands. Nothing in the flat dict said
which names were aggregates, so the only way to find out was to read the runtime.

This module makes the structure data rather than folklore. It is the single
source of truth for:

* which counter is a parent of which (``STAGE_TREE``),
* therefore which counters form a partition (the leaves),
* and whether a given counter was actually measured on this run, as opposed to
  reported as ``0.0`` because its instrumentation was switched off.

That last distinction is the reason ``measured`` exists at all. A stage that was
not instrumented and a stage that took no time both used to serialise as
``0.0``.

Naming rule
-----------
Every counter is ``refresh_<path>_seconds``, where ``<path>`` is the
underscore-joined node path in ``STAGE_TREE``. A counter whose name is a strict
prefix of other counters' names is an aggregate of exactly those; a counter with
no such children is a leaf. ``refresh_total_seconds`` is the root and is a sum of
everything below it *plus* whatever the refresh spends outside any instrumented
stage.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

__all__ = [
    "STAGE_TREE",
    "aggregate_counter_names",
    "format_stage_timing_tree",
    "leaf_counter_names",
    "stage_timing_tree",
]


#: Parent -> children, by the ``<path>`` part of ``refresh_<path>_seconds``.
#: A name absent from the keys here is a leaf. Children are listed in execution
#: order where that is meaningful, so a printed tree reads like a step.
STAGE_TREE: dict[str, tuple[str, ...]] = {
    "total": (
        "input",
        "tree_upward",
        "dual_downward",
        "nearfield",
        # The evaluate is part of a step even though it is not part of a
        # refresh: strict_prepare_refresh_and_evaluate does both, and leaving
        # this out made the evaluate the largest single term inside
        # "unattributed". Populated only on that entry point.
        "evaluate",
        "profile_accounting",
        "compile_or_sync_suspect",
    ),
    "tree_upward": ("tree_build", "upward_compute"),
    "upward_compute": (
        "upward_geometry",
        "upward_mass_moments",
        "upward_p2m",
        "upward_m2m",
        "upward_source_motion",
    ),
    "dual_downward": (
        "dual_setup",
        "dual_artifact_build",
        "dual_split_combined",
        "dual_raw_combined",
        "dual_split_dense_buffers",
        "dual_far_pair_plan",
        "dual_m2l_autotune",
        "dual_select_interactions",
        "dual_downward_compute",
        "dual_finalize",
        "dual_residual",
    ),
    "dual_split_combined": (
        "dual_split_shared_far_near",
        "dual_split_far_pairs",
        "dual_split_leaf_neighbors",
    ),
    "dual_split_shared_far_near": (
        "dual_split_shared_count",
        "dual_split_shared_combined_fill",
        "dual_split_shared_far_fill",
        "dual_split_shared_near_fill",
    ),
    "dual_downward_compute": (
        "dual_m2l_compute",
        "dual_l2l_compute",
        "dual_final_symmetry",
        "dual_source_motion",
    ),
    "nearfield": (
        "nearfield_leaf_groups",
        "nearfield_precompute",
        "nearfield_target_blocks",
        "nearfield_block_sort",
        "nearfield_speed_layout",
        "nearfield_overflow_profile",
        "nearfield_radix_payload",
        "nearfield_neighbor_padding",
        "nearfield_state_pack",
        "nearfield_residual",
    ),
}

#: Counters whose instrumentation is conditional, and the runtime flag that says
#: whether it ran. Reported as ``measured: False`` rather than as ``0.0`` when
#: the flag is off, so a consumer cannot read "switched off" as "free".
_CONDITIONAL: dict[str, str] = {
    "dual_m2l_compute": "_refresh_timing_substages_measured",
    "dual_l2l_compute": "_refresh_timing_substages_measured",
    "dual_final_symmetry": "_refresh_timing_substages_measured",
    "dual_source_motion": "_refresh_timing_substages_measured",
}


def aggregate_counter_names() -> frozenset[str]:
    """``refresh_*_seconds`` names that are sums of other counters.

    Returns
    -------
    frozenset[str]
        The counter name of every stage that has children, fully prefixed. These
        overlap their children by construction, so adding an aggregate to a leaf
        double-counts -- see :func:`leaf_counter_names` for the set that does
        not. A name here can also appear in the leaf set of no other stage; the
        two sets are disjoint.
    """

    return frozenset(f"refresh_{name}_seconds" for name in STAGE_TREE)


def leaf_counter_names() -> frozenset[str]:
    """``refresh_*_seconds`` names that form a partition (no double counting).

    Returns
    -------
    frozenset[str]
        Counter names for stages that appear as somebody's child and have no
        children themselves. Safe to sum: no leaf contains another. Stages whose
        measurement is conditional are included regardless of whether they were
        measured on a given run, so a leaf may legitimately read 0.0 rather than
        be absent.
    """

    children: set[str] = set()
    for names in STAGE_TREE.values():
        children.update(names)
    leaves = {name for name in children if name not in STAGE_TREE}
    return frozenset(f"refresh_{name}_seconds" for name in sorted(leaves))


def _node(
    name: str,
    values: Mapping[str, float],
    measured: Mapping[str, bool],
) -> dict[str, Any]:
    seconds = float(values.get(name, 0.0))
    node: dict[str, Any] = {
        "seconds": seconds,
        "measured": bool(measured.get(name, True)),
        "counter": f"refresh_{name}_seconds",
    }
    children = STAGE_TREE.get(name)
    if children:
        node["children"] = {child: _node(child, values, measured) for child in children}
        # How much of this parent its instrumented children account for. A large
        # residual is a real finding (uninstrumented work inside the stage), not
        # a rounding artefact, so it is reported rather than distributed.
        child_sum = sum(node["children"][c]["seconds"] for c in children)
        node["children_seconds"] = float(child_sum)
        node["unattributed_seconds"] = float(max(seconds - child_sum, 0.0))
    return node


def stage_timing_tree(
    impl: Any,
    *,
    counters: Optional[Mapping[str, float]] = None,
    per_call: bool = False,
) -> dict[str, Any]:
    """Return the refresh stage timings as a nested, self-describing tree.

    Parameters
    ----------
    impl : Any
        The runtime engine (``FMMEngine._impl``) holding the
        ``_refresh_timing_*_seconds`` accumulators. Ignored except as the counter
        source when ``counters`` is supplied.
    counters : Optional[Mapping[str, float]]
        Pre-read counter values keyed by ``<path>``. Defaults to reading them off
        ``impl``. Supplying them lets a caller build the tree from a snapshot.
    per_call : bool
        Divide every value by ``refresh_timing_calls``, giving per-step seconds.
        Returns the accumulated totals when the call count is zero.

    Returns
    -------
    dict[str, Any]
        The stage tree, keyed by top-level stage name. Host-side bookkeeping
        only -- these are Python floats read off the engine, never traced values,
        so this must not be called from inside a jitted path.

    Notes
    -----
    Every node carries ``seconds``, ``measured``, and the flat ``counter`` name it
    corresponds to; parents additionally carry ``children``, ``children_seconds``
    and ``unattributed_seconds``. Summing ``children_seconds`` and ``seconds`` of
    the same node is the double count this structure exists to prevent -- sum one
    level, or sum the leaves.
    """

    if counters is None:
        values = {
            attr[len("_refresh_timing_") : -len("_seconds")]: float(getattr(impl, attr))
            for attr in vars(impl)
            if attr.startswith("_refresh_timing_") and attr.endswith("_seconds")
        }
    else:
        values = {str(k): float(v) for k, v in counters.items()}

    calls = int(getattr(impl, "_refresh_timing_calls", 0) or 0)
    if per_call and calls > 0:
        values = {k: v / calls for k, v in values.items()}

    measured = {
        name: bool(getattr(impl, flag, False)) for name, flag in _CONDITIONAL.items()
    }

    tree = _node("total", values, measured)
    known = _known_names()
    tree["unmapped_nonzero"] = {
        name: value
        for name, value in sorted(values.items())
        if value > 0.0 and name not in known
    }
    tree["calls"] = calls
    tree["per_call"] = bool(per_call and calls > 0)
    return tree


def _known_names() -> frozenset[str]:
    names: set[str] = set(STAGE_TREE)
    for children in STAGE_TREE.values():
        names.update(children)
    return frozenset(names)


def format_stage_timing_tree(tree: Mapping[str, Any]) -> str:
    """Render :func:`stage_timing_tree` output as indented text, for a log.

    Presentation only -- it derives no timings, and unmeasured stages are marked
    ``[not measured]`` rather than dropped, so a zero that means "we did not look"
    is distinguishable from a zero that means "it was free".

    Parameters
    ----------
    tree : Mapping[str, Any]
        A :func:`stage_timing_tree` result. Indexed by the layout that function
        produces: ``seconds`` is required on every node, while ``children``,
        ``measured``, ``unattributed_seconds`` and the top-level
        ``unmapped_nonzero`` are optional. A mapping missing ``seconds`` raises
        ``KeyError``; the rest degrade quietly.

    Returns
    -------
    str
        Newline-joined lines, no trailing newline. Times are milliseconds.
        Per-node residuals appear as ``(unattributed in <stage>)`` and counters
        the tree does not place appear as ``[unmapped]`` -- both are there so a
        reader can see that the children do not account for the parent, which is
        the failure mode this format exists to expose.
    """

    lines: list[str] = []

    def walk(name: str, node: Mapping[str, Any], depth: int) -> None:
        seconds = float(node["seconds"])
        flag = "" if node.get("measured", True) else "   [not measured]"
        lines.append(f"{'  ' * depth}{name:<32s}{seconds * 1e3:9.2f} ms{flag}")
        for child_name, child in (node.get("children") or {}).items():
            walk(child_name, child, depth + 1)
        residual = node.get("unattributed_seconds")
        if residual:
            lines.append(
                f"{'  ' * (depth + 1)}{'(unattributed in ' + name + ')':<32s}"
                f"{float(residual) * 1e3:9.2f} ms"
            )

    walk("total", tree, 0)
    unmapped: Iterable[tuple[str, float]] = (tree.get("unmapped_nonzero") or {}).items()
    for name, value in unmapped:
        lines.append(f"[unmapped] {name:<28s}{float(value) * 1e3:9.2f} ms")
    return "\n".join(lines)
