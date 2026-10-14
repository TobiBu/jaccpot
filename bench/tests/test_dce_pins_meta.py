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


# --------------------------------------------------------------------- the L pins
# Rows measured 2026-10-13 on GPU 7 (8e6 clipped Plummer, particle carry): main with
# plummer twice (the A-vs-A control) and main with the ferrers3 default.
from bench.dce_pins import _compare_timed  # noqa: E402


def _row(
    ms: float, peak: float, contaminated: bool = False, far: int = 39093974
) -> dict:
    return {
        "timing": True,
        "n": 8_000_000,
        "step": {
            "ms_min": ms,
            "peak_gib": peak,
            "far_pair_count": far,
            "contaminated": contaminated,
        },
    }


_PLUMMER, _PLUMMER_B, _FERRERS3 = (
    _row(72.81, 2.3589),
    _row(72.80, 2.3589),
    _row(78.24, 2.3733),
)


def test_timing_pin_passes_against_its_own_control():
    assert _compare_timed("L1", _PLUMMER, _PLUMMER_B, _PLUMMER_B)["pass"]


def test_timing_pin_catches_the_ferrers3_step_cost():
    # +7.5 % step time: outside the 3 % floor (the control's spread is ~0.01 %)
    entry = _compare_timed("L1", _PLUMMER, _FERRERS3, _PLUMMER_B)
    assert not entry["pass"]
    assert entry["rows"]["step"]["lists_equal"]


def test_timing_pin_refuses_a_contaminated_row():
    assert not _compare_timed(
        "L1", _PLUMMER, _row(72.8, 2.3589, contaminated=True), None
    )["pass"]


def test_timing_pin_envelope_widens_with_a_noisy_control():
    noisy = _row(75.0, 2.3589)  # a 3 % A-vs-A spread -> a 6 % envelope
    assert _compare_timed("L1", _PLUMMER, _row(76.5, 2.3589), noisy)["pass"]
    assert not _compare_timed("L1", _PLUMMER, _row(78.24, 2.3589), noisy)["pass"]


def test_timing_pin_accuracy_must_match_to_two_digits():
    ref = {**_PLUMMER, "force": {"ms_min": 100.6, "peak_gib": 4.82, "rel_l2": 4.179e-4}}
    same = {
        **_PLUMMER,
        "force": {"ms_min": 100.7, "peak_gib": 4.82, "rel_l2": 4.181e-4},
    }
    worse = {**_PLUMMER, "force": {"ms_min": 100.7, "peak_gib": 4.82, "rel_l2": 4.6e-4}}
    assert _compare_timed("L3", ref, same, None)["pass"]
    assert not _compare_timed("L3", ref, worse, None)["pass"]
