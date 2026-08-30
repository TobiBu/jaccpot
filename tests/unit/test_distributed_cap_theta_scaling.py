"""The derived walk caps must cover every capacity floor we have measured.

The asymmetry these tests exist for: an OVER-provisioned queue costs time linearly,
but an UNDER-provisioned one truncates the walk SILENTLY -- the run reads *faster*
and only ``self_near_pairs`` witnesses it. So the meaningful guard is a lower bound
against measurement, not a bit-identity pin against the previous rule (which would
forbid the rule ever getting tighter, and which the fix deliberately changes).

Floors below were measured on 5 x A100, per-device 2 097 152, leaf 512, order 6, by
walking each queue down until ``self_near_pairs`` collapsed. See
``_QUEUE_THETA_EXPONENT`` for the table and the fit.
"""

import math

import pytest

from jaccpot.distributed.cap_presets import _key
from jaccpot.distributed.fmm import DistributedFMMConfig

_PER_DEVICE_N = 2_097_152
_LEAF = 512

#: theta -> smallest self wavefront queue that did NOT truncate.
_SELF_FLOORS = {
    0.3: 2_097_152,
    0.4: 1_048_576,
    0.5: 524_288,
    0.6: 524_288,
    0.7: 524_288,
    0.8: 262_144,
}

#: theta -> smallest cross wavefront queue that did NOT truncate, at ndev 5.
_CROSS_FLOORS = {
    0.4: 4_194_304,
    0.5: 2_097_152,
    0.6: 2_097_152,
    0.7: 1_048_576,
    0.8: 1_048_576,
}

#: ndev -> smallest cross queue that did not truncate, at theta 0.7, per-device N fixed.
#: A LINEAR remote factor predicts 2 097 152 at ndev 4 and is refuted by this row.
_CROSS_BY_NDEV = {2: 524_288, 4: 1_048_576, 5: 1_048_576}


def _resolved(theta: float, ndev: int = 5) -> DistributedFMMConfig:
    return DistributedFMMConfig(leaf_size=_LEAF, theta=theta).resolved_for(
        _PER_DEVICE_N, ndev
    )


@pytest.mark.parametrize("theta,floor", sorted(_SELF_FLOORS.items()))
def test_self_queue_covers_the_measured_floor(theta: float, floor: int) -> None:
    """The self wavefront queue must never be derived below what the walk needs."""
    assert _resolved(theta).max_pair_queue >= floor


@pytest.mark.parametrize("theta,floor", sorted(_CROSS_FLOORS.items()))
def test_cross_queue_covers_the_measured_floor(theta: float, floor: int) -> None:
    """The cross wavefront queue must never be derived below what the walk needs."""
    assert _resolved(theta).cross_max_pair_queue >= floor


@pytest.mark.parametrize("ndev,floor", sorted(_CROSS_BY_NDEV.items()))
def test_cross_queue_covers_the_floor_at_every_measured_device_count(
    ndev: int, floor: int
) -> None:
    """The remote factor must cover ndev 2, 4 and 5 -- not just the 5 it was fitted at."""
    assert _resolved(0.7, ndev).cross_max_pair_queue >= floor


def test_tightening_theta_grows_the_queues() -> None:
    """A stricter MAC needs a bigger wavefront, and the rule must know it.

    This is the regression the whole change exists for: the rule used to be
    theta-blind, so theta 0.3 derived the same 1 048 576 as theta 0.4 and truncated
    (``self_queue_overflow``, rel_l2 5.5e-01).
    """
    tight, loose = _resolved(0.3), _resolved(0.8)
    assert tight.max_pair_queue > loose.max_pair_queue
    assert tight.cross_max_pair_queue > loose.cross_max_pair_queue


def test_the_shipped_default_is_not_on_its_cliff_edge() -> None:
    """theta 0.4 must clear its floor with margin, not sit exactly on it.

    Before this change the derived self queue at the SHIPPED default was 1 048 576
    against a measured floor of 1 048 576 -- zero margin, one ladder step from a
    silent truncation that collapses self_near_pairs 28 110 976 -> 3 502 872.
    """
    assert _resolved(0.4).max_pair_queue >= 2 * _SELF_FLOORS[0.4]


def test_queues_never_fall_below_their_floors_at_a_loose_mac() -> None:
    """A very loose MAC must still respect the hard minimums."""
    r = _resolved(2.0)
    assert r.max_pair_queue >= 1024
    assert r.cross_max_pair_queue >= 1024


def test_cross_remote_factor_is_sublinear() -> None:
    """Going from 1 to 4 remote domains must not quadruple the queue.

    Measured: the floor moves 524 288 -> 1 048 576 for remote 1 -> 4, i.e. 2x for a
    4x change. A linear factor would over-provision by 2x at ndev 5 and worse above.
    """
    one, four = _resolved(0.7, 2), _resolved(0.7, 5)
    assert four.cross_max_pair_queue < 4 * one.cross_max_pair_queue


def test_presets_key_separates_theta_and_leaf() -> None:
    """A preset recorded at one theta must not be served at another.

    The caps scale as theta**-1.5, so a theta 0.7 preset is a 2.8x under-estimate at
    theta 0.4 -- and under-sizing truncates silently.
    """
    assert _key(1, 2, 0.4, 512) != _key(1, 2, 0.7, 512)
    assert _key(1, 2, 0.7, 256) != _key(1, 2, 0.7, 512)
    assert _key(1, 2) == "1:2", "legacy two-part keys must still be readable"


def test_theta_scale_is_monotone_and_finite() -> None:
    """The multiplier must fall with theta and survive a nonsense value."""
    from jaccpot.distributed.fmm import _queue_theta_scale

    vals = [_queue_theta_scale(t) for t in (0.2, 0.4, 0.8, 1.6)]
    assert vals == sorted(vals, reverse=True)
    assert math.isfinite(_queue_theta_scale(0.0))
