"""Persisting and extrapolating the distributed traversal caps.

``jaccpot/distributed/cap_presets.py`` measured **0%** in the CPU coverage run --
41 statements, entirely untouched. It is host-side (one ``jax`` reference in the
whole file), so this is an F27-shaped gap: it needed a test, not a second card.
Its sibling ``jaccpot/mutual/cap_presets.py`` already sits at 97%, and
``tests/unit/mutual/test_cap_presets.py`` served as the template.

The table exists because the caps are *discovered* by a retry loop, and every
retry costs a recompile. So the behaviour worth pinning is not "does it round
trip" but the **direction of error** in :func:`lookup`, which its docstring is
explicit about:

* an exact match is used as-is;
* otherwise the nearest **larger** per-GPU N at the same ``ndev`` -- deliberately
  a safe over-estimate, since over-sizing costs memory while under-sizing costs
  a retry storm;
* otherwise the largest **smaller** preset scaled up by the N ratio -- explicitly
  only a starting point, which ``auto_scale`` then refines.

Getting the middle rule subtly wrong -- taking *any* larger preset rather than
the nearest, or preferring a scaled-up smaller one over an available larger one
-- would still round trip, still look correct, and quietly return caps that need
growing again. The ordering tests below are what separate those cases.
"""

from __future__ import annotations

import json
import os

import pytest

from jaccpot.distributed.cap_presets import (
    CAP_FIELDS,
    apply_caps,
    caps_of,
    load_presets,
    lookup,
    record,
    save_presets,
)
from jaccpot.distributed.fmm import DistributedFMMConfig


def _caps(scale: int = 1) -> dict:
    """Caps carrying a distinct value per field, so a field mix-up is visible."""
    return {f: (idx + 1) * 100 * scale for idx, f in enumerate(CAP_FIELDS)}


def test_caps_of_reads_every_cap_field():
    """The extract half of the round trip covers all of ``CAP_FIELDS``."""
    config = DistributedFMMConfig()
    caps = caps_of(config)
    assert set(caps) == set(CAP_FIELDS)
    for field in CAP_FIELDS:
        assert caps[field] == getattr(config, field)


def test_apply_caps_round_trips_through_a_config():
    """Applying extracted caps reproduces them, field for field."""
    applied = apply_caps(DistributedFMMConfig(), _caps())
    assert caps_of(applied) == _caps()


def test_apply_caps_ignores_fields_the_dict_omits():
    """A partial cap dict is a partial update, not a reset to defaults.

    The retry loop grows some caps and leaves others alone, so the fields it does
    not mention have to survive untouched.
    """
    base = DistributedFMMConfig()
    applied = apply_caps(base, {"max_pair_queue": 999_999})
    assert applied.max_pair_queue == 999_999
    for field in CAP_FIELDS:
        if field != "max_pair_queue":
            assert getattr(applied, field) == getattr(base, field)


def test_record_then_lookup_returns_the_exact_entry():
    presets: dict = {}
    record(presets, per_gpu_n=1000, ndev=2, total_n=2000, caps=_caps())
    assert lookup(presets, 1000, 2) == _caps()


def test_record_normalises_to_int_and_keeps_none():
    """``None`` means "driver right-sizes it" and must survive the round trip.

    Coercing it to ``int`` would silently turn "no opinion" into a cap of zero.
    """
    presets: dict = {}
    caps = dict(_caps())
    caps[CAP_FIELDS[0]] = None
    caps[CAP_FIELDS[1]] = 512.0  # converged values can arrive as floats
    record(presets, per_gpu_n=10, ndev=1, total_n=20, caps=caps)
    entry = presets["10:1"]
    assert entry["caps"][CAP_FIELDS[0]] is None
    assert entry["caps"][CAP_FIELDS[1]] == 512
    assert isinstance(entry["caps"][CAP_FIELDS[1]], int)
    assert entry["per_gpu_n"] == 10 and entry["ndev"] == 1 and entry["total_n"] == 20


def test_lookup_returns_none_when_no_preset_exists_for_that_ndev():
    presets: dict = {}
    record(presets, per_gpu_n=1000, ndev=2, total_n=2000, caps=_caps())
    assert lookup(presets, 1000, 4) is None


def test_presets_do_not_leak_across_device_counts():
    """``ndev`` partitions the table, because the caps are per-device.

    A 2-device preset describes a different traversal than a 4-device one at the
    same per-GPU N, so serving one for the other would be silently wrong rather
    than merely mis-sized.
    """
    presets: dict = {}
    record(presets, per_gpu_n=1000, ndev=2, total_n=2000, caps=_caps(scale=1))
    record(presets, per_gpu_n=1000, ndev=4, total_n=4000, caps=_caps(scale=7))
    assert lookup(presets, 1000, 2) == _caps(scale=1)
    assert lookup(presets, 1000, 4) == _caps(scale=7)


def test_lookup_prefers_the_nearest_larger_preset():
    """Not merely *a* larger preset -- the nearest one.

    With presets at 2000 and 8000, a query at 1000 must take 2000. Returning
    8000 would still be "safe", would still pass a laxer test, and would size
    every buffer four times too big.
    """
    presets: dict = {}
    record(presets, per_gpu_n=2000, ndev=2, total_n=4000, caps=_caps(scale=2))
    record(presets, per_gpu_n=8000, ndev=2, total_n=16000, caps=_caps(scale=8))
    assert lookup(presets, 1000, 2) == _caps(scale=2)


def test_a_larger_preset_beats_scaling_up_a_smaller_one():
    """A measured over-estimate is preferred to extrapolation, which is only a seed."""
    presets: dict = {}
    record(presets, per_gpu_n=500, ndev=2, total_n=1000, caps=_caps(scale=1))
    record(presets, per_gpu_n=4000, ndev=2, total_n=8000, caps=_caps(scale=4))
    assert lookup(presets, 1000, 2) == _caps(scale=4)


def test_lookup_scales_the_largest_smaller_preset_when_nothing_larger_exists():
    """The last resort scales from the *largest* smaller preset.

    Extrapolating from a more distant one would stretch the ratio further than
    it has to be stretched.
    """
    presets: dict = {}
    record(presets, per_gpu_n=100, ndev=2, total_n=200, caps=_caps(scale=1))
    record(
        presets, per_gpu_n=1000, ndev=2, total_n=2000, caps={f: 10 for f in CAP_FIELDS}
    )
    assert lookup(presets, 2000, 2) == {f: 20 for f in CAP_FIELDS}


def test_scaling_rounds_up_and_preserves_none():
    """Ceil, not floor -- rounding a cap down is how a retry storm starts."""
    presets: dict = {}
    caps: dict = {f: 3 for f in CAP_FIELDS}
    caps[CAP_FIELDS[0]] = None
    record(presets, per_gpu_n=2, ndev=1, total_n=2, caps=caps)
    got = lookup(presets, 3, 1)  # ratio 3/2, so 3 * 3 / 2 = 4.5 must land on 5
    assert got is not None
    assert got[CAP_FIELDS[0]] is None
    for field in CAP_FIELDS[1:]:
        assert got[field] == 5, f"{field} rounded down"


def test_save_and_load_round_trip(tmp_path):
    presets: dict = {}
    record(presets, per_gpu_n=1000, ndev=2, total_n=2000, caps=_caps())
    path = str(tmp_path / "presets.json")
    save_presets(path, presets)
    assert load_presets(path) == presets
    assert lookup(load_presets(path), 1000, 2) == _caps()


def test_save_is_atomic_and_leaves_no_temporary_behind(tmp_path):
    """Written to a ``.tmp`` and moved, so a concurrent reader never sees a partial file."""
    path = str(tmp_path / "presets.json")
    presets: dict = {}
    record(presets, per_gpu_n=1, ndev=1, total_n=1, caps=_caps())
    save_presets(path, presets)
    assert os.path.exists(path)
    assert not os.path.exists(f"{path}.tmp")
    with open(path) as fh:
        assert json.load(fh) == presets


def test_save_replaces_an_existing_table_rather_than_merging(tmp_path):
    path = str(tmp_path / "presets.json")
    first: dict = {}
    record(first, per_gpu_n=1, ndev=1, total_n=1, caps=_caps(scale=1))
    save_presets(path, first)
    second: dict = {}
    record(second, per_gpu_n=2, ndev=1, total_n=2, caps=_caps(scale=2))
    save_presets(path, second)
    assert load_presets(path) == second


@pytest.mark.parametrize("path", [None, ""])
def test_load_presets_without_a_path_is_an_empty_table(path):
    """An unset path means "no presets", not an error -- the caller then calibrates."""
    assert load_presets(path) == {}


def test_load_presets_from_a_missing_file_is_an_empty_table(tmp_path):
    assert load_presets(str(tmp_path / "does-not-exist.json")) == {}


def test_the_criterion_does_not_share_a_preset_slot_with_the_geometric_mac():
    """A ``dehnen_error`` run must not read back caps a geometric run recorded.

    The two derive DIFFERENT self queues at the same ``(N, ndev, theta, leaf)``: the
    criterion's is floored rather than theta-scaled, because a pair policy decides
    its acceptance and theta gates nothing. Sharing a slot means the geometric run
    records the smaller caps and the next criterion run applies them -- and an
    under-sized queue truncates the walk SILENTLY, reading faster with only
    ``self_near_pairs`` as the witness. Same failure the theta component of the key
    already exists to prevent, reached from a different direction.
    """

    from jaccpot.distributed import cap_presets as cp

    presets: dict = {}
    geometric_caps = {f: 111 for f in cp.CAP_FIELDS}
    cp.record(presets, 65536, 4, 262144, geometric_caps, 0.8, 64, "dehnen")

    assert cp.lookup(presets, 65536, 4, 0.8, 64, "dehnen") == geometric_caps
    assert (
        cp.lookup(presets, 65536, 4, 0.8, 64, "dehnen_error") is None
    ), "the criterion read back the geometric arm's caps"

    criterion_caps = {f: 222 for f in cp.CAP_FIELDS}
    cp.record(presets, 65536, 4, 262144, criterion_caps, 0.8, 64, "dehnen_error")
    assert cp.lookup(presets, 65536, 4, 0.8, 64, "dehnen_error") == criterion_caps
    assert (
        cp.lookup(presets, 65536, 4, 0.8, 64, "dehnen") == geometric_caps
    ), "recording the criterion overwrote the geometric entry"


def test_geometric_preset_keys_are_unchanged_by_the_criterion_component():
    """An existing presets file must keep working, byte for byte.

    Only a non-geometric ``mac_type`` joins the key, so every key already on disk --
    written before this component existed -- still resolves.
    """

    from jaccpot.distributed import cap_presets as cp

    presets: dict = {}
    caps = {f: 7 for f in cp.CAP_FIELDS}
    cp.record(presets, 1024, 2, 2048, caps, 0.4, 32)
    written = next(iter(presets))

    assert written == "1024:2:t0.4:l32"
    for mac in ("", "bh", "engblom", "dehnen"):
        assert cp.lookup(presets, 1024, 2, 0.4, 32, mac) == caps, mac
