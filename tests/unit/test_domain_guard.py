"""The IC guard and the loud tree-depth bound (A2), without a force.

* :func:`jaccpot.domain.domain_check` names the Morton box, the level-21 cell, the
  farthest particle and the bulk's radius; :func:`~jaccpot.domain.domain_guard`
  warns (``"warn"``) or refuses (``"error"``) when a few far particles set the box,
  and says nothing about a compact system;
* ``SegmentConfig.domain_guard`` takes ``warn`` (default), ``error`` or ``off``;
* the fused scan never re-plans a tree deeper than yggdrax's ``MAX_TREE_LEVELS``
  (no bound it grows covers it) and its error names the box and the outliers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from jaxtyping import TypeCheckError

from jaccpot.domain import DomainWarning, domain_check, domain_guard
from jaccpot.production import SegmentConfig
from jaccpot.runtime.fmm_strict_run import StrictRunMixin
from jaccpot.runtime.fused_topology import NEEDS_FIELDS

#: an unknown mode: the guard's own ValueError, or -- under
#: JACCPOT_RUNTIME_TYPECHECK=1, as CI's unit shard runs -- the type checker's
#: refusal of the Literal first
_BAD_MODE = (ValueError, TypeCheckError)


def _bulk_and_tails(n=20_000, tails=3, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3))
    if tails:
        far = np.zeros((tails, 3))
        far[:, 0] = 1e4 * (1.0 + np.arange(tails))
        pos = np.concatenate([pos, far])
    vel = np.zeros_like(pos)
    return np.stack([pos, vel], axis=1)


def test_the_check_names_the_box_and_the_outliers():
    state = _bulk_and_tails()
    report = domain_check(state, leaf_size=64, sample=4096)
    assert report["n"] == 20_003
    assert report["farthest"]["index"] == 20_002
    assert report["r_max"] == pytest.approx(3e4, rel=1e-3)
    assert report["cell21"] == pytest.approx(report["box_extent"] / 2.0**21)
    assert report["r_quantile"] < 10.0 and report["dominance"] > 1e3
    assert report["n_beyond"] == 3
    # a level-21 cell of ~0.014 at the centre of a unit Gaussian holds no one
    assert report["central_count"] <= 64 and report["depth_bound"] == 64
    assert not report["depth_overflow_possible"]


def test_the_guard_warns_refuses_or_stays_quiet():
    state = _bulk_and_tails()
    with pytest.warns(DomainWarning, match=r"dominated.*farthest particle #20002"):
        domain_guard(state, mode="warn", sample=4096)
    with pytest.raises(ValueError, match="dominated by a few far particles"):
        domain_guard(state, mode="error", sample=4096)
    assert domain_guard(state, mode="off") is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = domain_guard(_bulk_and_tails(tails=0), mode="error", sample=4096)
    assert report is not None and report["dominance"] < 20.0
    with pytest.raises(_BAD_MODE):
        domain_guard(state, mode="loud")


def test_segment_config_takes_the_guard_mode():
    assert SegmentConfig().domain_guard == "warn"
    assert SegmentConfig(domain_guard="off").domain_guard == "off"
    with pytest.raises(_BAD_MODE):
        SegmentConfig(domain_guard="sometimes")


def _needs(**named):
    out = np.zeros(len(NEEDS_FIELDS), np.int64)
    for key, value in named.items():
        out[NEEDS_FIELDS.index(key)] = value
    return out


def test_a_tree_past_64_levels_is_never_replanned_and_says_why():
    deep = _needs(tree_overflow=1, depth_needed=68, leaf_needed=10)
    shallow = _needs(tree_overflow=1, depth_needed=40, leaf_needed=10)
    assert StrictRunMixin._segment_retry_mode(None, deep, 0, False) is None
    assert StrictRunMixin._segment_retry_mode(None, shallow, 0, False) == "full"
    named = dict(zip(NEEDS_FIELDS, (int(v) for v in deep)))
    positions = _bulk_and_tails()[:, 0]
    with pytest.raises(RuntimeError) as err:
        StrictRunMixin._raise_if_too_deep(None, named, positions, "")
    message = str(err.value)
    assert "68 levels deep" in message and "MAX_TREE_LEVELS (64)" in message
    assert "farthest particle #20002" in message and "level-21 cell" in message
    # within the bound: nothing to say
    StrictRunMixin._raise_if_too_deep(
        None, dict(zip(NEEDS_FIELDS, (int(v) for v in shallow))), positions, ""
    )
