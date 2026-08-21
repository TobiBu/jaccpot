"""Persisting and extrapolating ``MutualCapacities``.

The point of this table is that one cap -- the wavefront ``queue`` -- cannot be
derived from a finished topology, only found by trial, and that trial dominates
`freeze_template`. So the table has to be right about two things a naive copy of the
target-centric version would get wrong: what identifies a profile, and how each cap
extrapolates.
"""

from __future__ import annotations

import pytest

from jaccpot.mutual.cap_presets import (
    MUTUAL_CAP_FIELDS,
    apply_caps,
    caps_of,
    load_presets,
    lookup,
    record,
    save_presets,
)
from jaccpot.mutual.force import MutualCapacities

# Measured on an A100, theta 0.7, order 4, leaf 64 (see the module docstring).
CAPS_1E5 = dict(near=524_288, far=262_144, depth=16, width=1_536, queue=32_768)
CAPS_1E6 = dict(near=6_291_456, far=3_145_728, depth=24, width=24_576, queue=4_194_304)
PROFILE = dict(ndev=1, leaf_size=64, theta=0.7, order=4)


def _table():
    return record({}, per_gpu_n=100_000, total_n=100_000, caps=CAPS_1E5, **PROFILE)


def test_roundtrip_through_a_file(tmp_path):
    path = str(tmp_path / "presets.json")
    save_presets(path, _table())
    got = load_presets(path)
    assert lookup(got, 100_000, **PROFILE) == CAPS_1E5


def test_missing_file_is_an_empty_table():
    assert load_presets(None) == {}
    assert load_presets("/nonexistent/presets.json") == {}


def test_exact_hit_is_returned_unchanged():
    assert lookup(_table(), 100_000, **PROFILE) == CAPS_1E5


def test_a_larger_measured_profile_is_preferred_over_scaling_a_smaller_one():
    """A measured over-estimate beats an extrapolation."""
    t = record(
        _table(), per_gpu_n=1_000_000, total_n=1_000_000, caps=CAPS_1E6, **PROFILE
    )
    got = lookup(t, 500_000, **PROFILE)
    assert got == CAPS_1E6, "should take the measured larger profile, not scale up"


def test_depth_does_not_scale_linearly():
    """The distinction this module exists for.

    Pair lists track N; a tree DEPTH does not. Scaling depth linearly would inflate
    the dense (depth, width) level schedule every M2M/L2L cascade scans -- the exact
    waste the quarter-octave capacity ladder was introduced to remove.
    """
    got = lookup(_table(), 1_000_000, **PROFILE)  # 10x up from the 1e5 entry
    assert got is not None
    # linear caps scale by ~10x
    assert got["near"] == pytest.approx(CAPS_1E5["near"] * 10, rel=0.01)
    assert got["far"] == pytest.approx(CAPS_1E5["far"] * 10, rel=0.01)
    assert got["queue"] == pytest.approx(CAPS_1E5["queue"] * 10, rel=0.01)
    # depth grows by a small additive margin, nowhere near 10x
    assert (
        got["depth"] < CAPS_1E5["depth"] * 2
    ), f"depth scaled to {got['depth']} from {CAPS_1E5['depth']} -- looks linear"
    # and it lands just under the measured 24 for this decade. Being one shallow is
    # the intended direction: a shallow depth raises a loud overflow and costs a
    # retry, a deep one silently inflates the level schedule for the whole run.
    assert 20 <= got["depth"] <= 24, got["depth"]


def test_scaling_up_never_shrinks_a_cap():
    got = lookup(_table(), 400_000, **PROFILE)
    for f in MUTUAL_CAP_FIELDS:
        assert got[f] >= CAPS_1E5[f], f"{f} shrank while N grew"


def test_a_different_leaf_size_is_never_matched():
    """Caps move ~4x between leaf 64 and 256, so cross-leaf reuse would be wrong."""
    t = _table()
    assert lookup(t, 100_000, ndev=1, leaf_size=256, theta=0.7, order=4) is None


def test_a_different_theta_or_order_is_never_matched():
    t = _table()
    assert lookup(t, 100_000, ndev=1, leaf_size=64, theta=0.5, order=4) is None
    assert lookup(t, 100_000, ndev=1, leaf_size=64, theta=0.7, order=6) is None


def test_a_different_device_count_is_never_matched():
    t = _table()
    assert lookup(t, 100_000, ndev=8, leaf_size=64, theta=0.7, order=4) is None


def test_theta_is_rounded_so_float_noise_shares_an_entry():
    t = _table()
    assert lookup(t, 100_000, ndev=1, leaf_size=64, theta=0.70000001, order=4) == (
        CAPS_1E5
    )


def test_caps_of_and_apply_caps_round_trip_a_real_MutualCapacities():
    caps = MutualCapacities(near=1, far=2, depth=3, width=4, queue=5)
    d = caps_of(caps)
    assert d == dict(near=1, far=2, depth=3, width=4, queue=5)
    updated = apply_caps(caps, dict(near=99, queue=None))
    assert updated.near == 99
    assert updated.queue == 5, "None must leave the existing value alone"
    assert updated.far == 2


def test_record_keeps_provenance():
    t = record(
        {},
        per_gpu_n=1_000,
        total_n=8_000,
        ndev=8,
        leaf_size=32,
        theta=0.6,
        order=5,
        caps=CAPS_1E5,
    )
    (entry,) = t.values()
    assert entry["per_gpu_n"] == 1_000
    assert entry["total_n"] == 8_000
    assert entry["ndev"] == 8
    assert entry["leaf_size"] == 32
    assert entry["order"] == 5
