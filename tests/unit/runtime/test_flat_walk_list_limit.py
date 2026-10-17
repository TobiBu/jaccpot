"""The flat-walk lists stop at the index range instead of re-planning past it.

The S3 production gate (2026-10-17, 1e8 clipped Plummer, dt 1e-2) outgrew its far
list twice. Each segment retry sized the list from the walk's EXACT need with the
1.5x headroom: 603,979,776 -> 948,961,280 -> 1,508,900,864 directed entries for
needs of about 632.6M and 1,005.9M. The second width is a canonical
754,450,432 pairs, and the CSR build (``_directed_csr_from_canonical``, whose XLA
placement addresses ``3 W`` slots) raised ``pair width 754450432 overflows int32``
deep inside the re-prepare's template trace -- although the need itself fit.

Pinned here, on CPU without a tree:

* the index range: 1,431,306,240 directed entries for the int32 CSRs (2/3 of the
  range, granule-aligned), 2,146,435,072 for the flat walk's interleaved lists;
* ``_tight_list_capacity(limit=...)`` stops the headroom at the limit, keeps a
  larger floor, and is the old capacity below it;
* the segment re-plan clamps a list whose need fits, and raises the index-range
  error -- before any re-prepare -- for a need past it, a lower bound already at
  it, or a count that wrapped negative; below the limit it re-plans as before.

The eager ladder's clamp is pinned on a real tree in
``test_flat_walk_production_seam.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaccpot.runtime.fmm_strict_run as fsr
from jaccpot.runtime._interaction_cache import _tight_list_capacity
from jaccpot.runtime.capacity_guard import WALK_NEEDS_FIELDS

#: int32, CSR lists: 1365 granules of 2^20 directed entries (W = 715,653,120)
_CSR_INT32 = 1_431_306_240


def test_the_limit_is_the_csr_builds_index_range():
    from jaccpot.runtime._interaction_cache import flat_walk_list_limit

    top = int(np.iinfo(np.int32).max)
    csr = flat_walk_list_limit(jnp.int32, csr=True)
    assert csr == _CSR_INT32
    # the CSR build's own check passes at the limit, fails one granule past it
    assert 3 * (csr // 2) < top
    assert 3 * ((csr + (1 << 20)) // 2) >= top
    flat = flat_walk_list_limit(jnp.int32, csr=False)
    assert flat == 2_146_435_072 and flat < top
    assert flat_walk_list_limit(jnp.int64, csr=True) > 1 << 62


def test_the_headroom_stops_at_the_limit():
    # the crash's second re-plan: a need that fits, a 1.5x width that does not
    assert _tight_list_capacity(1_005_900_000, headroom=1.5) == 1_508_900_864
    got = _tight_list_capacity(
        1_005_900_000, headroom=1.5, floor=948_961_280, limit=_CSR_INT32
    )
    assert got == _CSR_INT32
    # below the limit nothing changes, and a floor past it is kept
    for count, floor in ((632_500_000, 603_979_776), (1000, 0), (3, 64)):
        assert _tight_list_capacity(
            count, headroom=1.5, floor=floor, limit=_CSR_INT32
        ) == _tight_list_capacity(count, headroom=1.5, floor=floor)
    assert (
        _tight_list_capacity(10, headroom=1.5, floor=_CSR_INT32 + 2, limit=_CSR_INT32)
        == _CSR_INT32 + 2
    )


def _needs(far, near=1000, *, far_ovf=1, near_ovf=0, queue_ovf=0, lower=0):
    n = dict(
        far_needed=far,
        near_needed=near,
        peak_wavefront=39_370_708,
        far_overflow=far_ovf,
        near_overflow=near_ovf,
        queue_overflow=queue_ovf,
        lower_bound=lower,
    )
    # the walk entries, then the tree's (fused_topology.NEEDS_FIELDS)
    return np.asarray([n[k] for k in WALK_NEEDS_FIELDS] + [0, 0, 0, 0], np.int64)


@pytest.fixture
def engine(monkeypatch):
    # the GPU lane's lists: Pallas walk (CSRs), int32 indices
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_WALK", "pallas")
    monkeypatch.setattr(fsr, "INDEX_DTYPE", jnp.int32)
    eng = fsr.StrictRunMixin()
    eng._strict_fused_validated_caps = dict(
        flat_walk=True,
        compact_far_pair_capacity=948_961_280,
        near_edge_capacity=253_755_392,
        peak_wavefront=39_370_708,
    )
    eng._strict_fused_traced_caps = {"queue_capacity": 67_108_864}
    return eng


def _caps(eng):
    v = eng._strict_fused_validated_caps
    return v["compact_far_pair_capacity"], v["near_edge_capacity"]


def test_a_need_that_fits_is_clamped_at_the_limit(engine):
    assert engine._replan_walk_caps_from_needs(_needs(1_005_900_000)) is True
    assert _caps(engine) == (_CSR_INT32, 253_755_392)


def test_below_the_limit_the_re_plan_is_unchanged(engine):
    engine._strict_fused_validated_caps["compact_far_pair_capacity"] = 603_979_776
    assert engine._replan_walk_caps_from_needs(_needs(632_500_000)) is True
    assert _caps(engine) == (948_961_280, 253_755_392)


@pytest.mark.parametrize(
    "needs, what",
    [
        # an exact need past the range
        (_needs(1_600_000_000), "needs 1600000000 directed entries"),
        # a lower bound (the flat walk stops at its first overflow) at the limit
        (_needs(_CSR_INT32, lower=1), f"overflowed {_CSR_INT32} directed"),
        # a canonical count past 2^30 wraps the traced int32 need negative
        (_needs(-123_456), f"overflowed {_CSR_INT32} directed"),
    ],
)
def test_a_list_past_the_index_range_raises_before_any_re_prepare(engine, needs, what):
    engine._strict_fused_validated_caps["compact_far_pair_capacity"] = _CSR_INT32
    before = dict(engine._strict_fused_validated_caps)
    with pytest.raises(RuntimeError, match="past the largest list") as err:
        engine._replan_walk_caps_from_needs(needs)
    assert what in str(err.value) and str(_CSR_INT32) in str(err.value)
    assert "far list" in str(err.value)
    # the remedies: fewer pairs, more cards, and the box a growing list points at
    for remedy in ("theta", "leaf_size", "jaccpot.distributed", "JACCPOT_CUBIC_BOUNDS"):
        assert remedy in str(err.value)
    # nothing re-planned: the caller never reaches the re-prepare
    assert engine._strict_fused_validated_caps == before


def test_a_doubling_lower_bound_stops_at_the_limit(engine):
    engine._strict_fused_validated_caps["compact_far_pair_capacity"] = 948_961_280
    needs = _needs(948_961_280, lower=1)
    assert engine._replan_walk_caps_from_needs(needs) is True
    assert _caps(engine)[0] == _CSR_INT32
