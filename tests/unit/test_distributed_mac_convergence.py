"""Distributed-path accuracy vs. direct summation, at whatever basis/MAC
config is shipped for the paper.

Extends the existing tests/test_distributed_fmm_driver.py (repo root) rather
than duplicating its fixtures. See PROJECT_PLAN.md Phase 0 -- this test's
tolerance should match whichever of items 5a-5c have landed by the time this
is written (0.19-0.24% vs. direct at 4xGPU as of the last engineering-log
update; recheck before hardcoding a tolerance).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Fill in once Phase 0's basis/MAC status is confirmed.")
def test_distributed_matches_direct_within_tolerance():
    raise NotImplementedError


@pytest.mark.skip(reason="Fill in once Phase 0's basis/MAC status is confirmed.")
def test_distributed_config_matches_documented_status():
    """Guards against the paper text and the actual shipped config drifting
    apart -- assert the driver's default basis/MAC matches whatever
    docs/multigpu_differentiability_model.md claims."""
    raise NotImplementedError
