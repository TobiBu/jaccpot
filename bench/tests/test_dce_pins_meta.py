"""`bench/dce_pins.py compare`: the metadata check, at any nesting depth.

Pin M2's A-vs-A control is not bitwise (ulp-level positions, so its probe errors move in
the 6th digit), and its probes are a dict of dicts. A nested float has to get the same
2-significant-digit tolerance as a top-level one, or M2 fails against its own control.
"""

from __future__ import annotations

from bench.dce_pins import _meta_matches

# recorded 2026-10-12: tags fix-27372d6 and fix-27372d6-b
_M2_REF = {"n": 100000, "probes": {"1": {"err": 0.0008598346906136312}}}
_M2_CONTROL = {"n": 100000, "probes": {"1": {"err": 0.0008598315284953285}}}


def test_nested_floats_get_the_top_level_tolerance():
    assert _meta_matches(_M2_REF, _M2_CONTROL)
    assert _meta_matches({"rel_l2": 1.853e-4}, {"rel_l2": 1.8531e-4})


def test_real_changes_still_fail():
    moved = {"n": 100000, "probes": {"1": {"err": 0.00093}}}
    assert not _meta_matches(_M2_REF, moved)
    assert not _meta_matches({"n": 100000}, {"n": 100001})
    assert not _meta_matches({"overflow": {"cross": 0.0}}, {"overflow": {"cross": 1.0}})
    # a key the new record lacks
    assert not _meta_matches(_M2_REF, {"n": 100000, "probes": {}})
