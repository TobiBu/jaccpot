"""Finite-difference vs. jax.grad comparison for force outputs, swept over
theta (MAC tightness), single-device and (if Phase 0 lands) distributed.

Mirrors the equivalent test in the yggdrax paper's PROJECT_PLAN.md.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Wire up jaccpot/runtime/fmm_evaluate.py's forward call.")
@pytest.mark.parametrize("theta", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_grad_matches_finite_difference_single_device(theta):
    raise NotImplementedError


@pytest.mark.skip(
    reason="Only relevant once the distributed path is differentiable end-to-end."
)
@pytest.mark.parametrize("theta", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_grad_matches_finite_difference_distributed(theta):
    raise NotImplementedError
