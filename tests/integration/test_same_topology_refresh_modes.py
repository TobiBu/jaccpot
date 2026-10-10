"""An opt-in path inside ``_refresh_large_n_same_topology`` -- audit **F33**.

The **upward-only diagnostic short-circuit** returns after the upward sweep so a
profiler can attribute stage cost. It is gated behind a flag that defaults off,
which is why nothing reached it before this file. It carries a ``dep`` term built
from sums multiplied by ``0.0`` -- a data dependency that keeps the compiler from
eliding the work being measured while contributing nothing numerically.

This file also covered the **compact far-pair reuse** path, which skipped
re-walking the far list after the drift, and its refusal without the unsafe
opt-in. The path went in the 2026-10 cleanup (X6): the fused refresh always
rebuilds its far list, and the opt-in now raises
(``test_strict_lane_removed_switches.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaccpot.config import FarFieldConfig, TreeConfig
from jaccpot.runtime._fmm_impl import FMMEngine
from jaccpot.runtime._large_n_types import LargeNPreparedState

N_PARTICLES = 512
LEAF_SIZE = 64
MAX_ORDER = 2


@pytest.fixture
def strict_env(monkeypatch):
    """Make the large-N production profile reachable on CPU."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")
    monkeypatch.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")


def _engine():
    return FMMEngine(
        preset="large_n_gpu",
        runtime_path="large_n",
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        theta=0.6,
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=MAX_ORDER,
    )


def _particles(seed: int = 3):
    key = jax.random.PRNGKey(seed)
    key_pos, key_mass, key_move = jax.random.split(key, 3)
    positions = jax.random.uniform(
        key_pos, (N_PARTICLES, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
    )
    masses = jax.random.uniform(
        key_mass, (N_PARTICLES,), minval=0.1, maxval=1.1, dtype=jnp.float32
    )
    moved = positions + 1e-4 * jax.random.normal(
        key_move, positions.shape, dtype=positions.dtype
    )
    return positions, masses, moved


@pytest.mark.slow
def test_upward_only_diagnostic_returns_after_the_upward_sweep(strict_env, monkeypatch):
    """The stage-attribution mode still returns a usable state.

    ``fused_device_mode=True`` is passed explicitly because
    ``static_fused_refresh`` requires it, and ``refresh_prepared_state`` defaults
    it to ``False``.
    """
    monkeypatch.setenv("JACCPOT_STRICT_REFRESH_DIAG_MODE", "upward_only")
    fmm = _engine()
    assert fmm._strict_refresh_diag_mode == "upward_only"
    assert fmm._strict_refresh_diag_upward_active
    assert not fmm._strict_refresh_diag_downward_active, (
        "upward_only must switch the downward stage off, or it is not the mode "
        "under test"
    )

    positions, masses, moved = _particles()
    prepared = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )
    hits_before = int(fmm._large_n_same_topology_refresh_hits)

    refreshed = fmm.refresh_prepared_state(
        prepared,
        moved,
        masses,
        leaf_size=LEAF_SIZE,
        max_order=MAX_ORDER,
        fused_device_mode=True,
    )

    assert isinstance(refreshed, LargeNPreparedState)
    assert int(fmm._large_n_same_topology_refresh_hits) == hits_before + 1
    assert refreshed.tree.positions_sorted.shape == (N_PARTICLES, 3)
