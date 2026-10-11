"""Shared setup for the strict lane's tests.

``strict_env`` was defined identically in ``test_strict_fused_eval_fn.py``,
``test_strict_refresh_faces.py`` and ``test_same_topology_refresh_modes.py``; it
lives here since those moved into this directory (2026-10 cleanup, Z-a, rule R5).
``test_strict_lane_removed_switches.py`` defines its own, which takes precedence
there.
"""

from __future__ import annotations

import jax
import pytest


@pytest.fixture
def strict_env(monkeypatch):
    """Make the large-N production profile reachable on CPU."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")
    monkeypatch.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")
