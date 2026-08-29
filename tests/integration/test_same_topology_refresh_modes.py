"""Two opt-in paths inside ``_refresh_large_n_same_topology`` -- audit **F33**.

Both are gated behind flags that default off, which is why nothing reached them:
lines 1442-1463 and 1542-1579 of ``fmm_strict_run.py``, 60 statements between
them, unexercised on CPU and on the A100 run #193 recorded.

* the **upward-only diagnostic short-circuit**, which returns after the upward
  sweep so a profiler can attribute stage cost. It carries a ``dep`` term built
  from sums multiplied by ``0.0`` -- a data dependency that keeps the compiler
  from eliding the work being measured while contributing nothing numerically.
* the **compact far-pair reuse** fast path, which skips re-walking the far list
  and is gated behind an env var whose name says it is unsafe. The refusal when
  reuse is requested *without* that opt-in is covered too, and is the more
  valuable of the two: it is what stands between a moved static-radix tree and
  silently stale M2L pairs.

Reaching the second needs ``_strict_fused_mode_active``, which only
``strict_run_v2`` sets -- ``refresh_prepared_state`` never does, so these go
through the runner rather than the refresh method directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaccpot.config import TreeConfig
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
        complex_rotation="solidfmm",
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


def _state(positions):
    return jnp.stack([positions, jnp.zeros_like(positions)], axis=1)


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


@pytest.mark.slow
def test_compact_pair_reuse_refuses_without_the_unsafe_opt_in(strict_env, monkeypatch):
    """The refusal is the safety property, so it is asserted directly.

    With reuse enabled, the unsafe opt-in withheld, and the fresh rebuild that
    would otherwise make it safe turned off, the only correct outcome is a
    refusal: cached M2L pairs can change once static-radix positions move, and
    reusing them would be silently wrong rather than loudly broken.
    """
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_REUSE_COMPACT_PAIRS", "1")
    monkeypatch.setenv(
        "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE", "0"
    )
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD", "0")

    fmm = _engine()
    positions, masses, _ = _particles()
    with pytest.raises(RuntimeError, match="unsafe"):
        fmm.strict_run_v2(
            state=_state(positions),
            masses=masses,
            dt=1e-3,
            num_steps=2,
            refresh_every=1,
            leaf_size=LEAF_SIZE,
            max_order=MAX_ORDER,
            theta=0.6,
            return_history=False,
        )


@pytest.mark.slow
def test_compact_pair_reuse_is_taken_when_explicitly_allowed(strict_env, monkeypatch):
    """The unsafe path runs when asked for by name, and is counted.

    Asserting the reuse counter rather than a numerical result on purpose: this
    path reuses far pairs that the moved positions may have invalidated, so its
    output is not something to pin. What matters is that opting in reaches it and
    that the diagnostics say so.
    """
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_REUSE_COMPACT_PAIRS", "1")
    monkeypatch.setenv(
        "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE", "1"
    )

    fmm = _engine()
    positions, masses, _ = _particles()
    reuse_before = int(getattr(fmm, "_static_radix_compact_pair_reuse_hits", 0))

    fmm.strict_run_v2(
        state=_state(positions),
        masses=masses,
        dt=1e-3,
        num_steps=2,
        refresh_every=1,
        leaf_size=LEAF_SIZE,
        max_order=MAX_ORDER,
        theta=0.6,
        return_history=False,
    )

    reuse_after = int(getattr(fmm, "_static_radix_compact_pair_reuse_hits", 0))
    assert reuse_after > reuse_before, (
        "the unsafe reuse path was not taken despite being opted into; without "
        "this the test silently exercises the ordinary rebuild instead"
    )
