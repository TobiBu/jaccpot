"""Tests for the two guards D-012 moved here from ODISSEO's block-step lane.

Each guards a way the mutual FMM can be *silently* wrong -- a configuration with
no far pairs (a direct sum whose far-field numbers are vacuous), and a device
topology that overflowed a capacity (interactions dropped with momentum still
exact). Neither is caught by any correctness assertion, which is why they are
guards rather than tests. See ``jaccpot.nornax_adapter``.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.mutual.force import OVERFLOW_CAUSES
from jaccpot.nornax_adapter import (
    BlockStepFMM,
    assert_far_field_is_exercised,
    raise_on_overflow,
)

SOFTENING = 1.0e-2


def _two_clumps(n=256, seed=23, separation=8.0, sigma=0.6):
    """Two well-separated clumps: 15 far pairs at theta = 0.6 for this seed."""
    rng = np.random.default_rng(seed)
    half = n // 2
    centres = ([-separation / 2.0, 0.0, 0.0], [separation / 2.0, 0.0, 0.0])
    positions = np.concatenate([rng.normal(c, sigma, (half, 3)) for c in centres])
    masses = rng.uniform(0.5, 1.5, n)
    return jnp.asarray(positions, jnp.float64), jnp.asarray(masses, jnp.float64)


def _one_blob(n=256, seed=0):
    """A single Gaussian blob: no far pairs at N = 256 for theta <= 0.9."""
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(0.0, 1.0, (n, 3)), jnp.float64),
        jnp.asarray(rng.uniform(0.5, 1.5, n), jnp.float64),
    )


def _device_fmm(theta):
    return BlockStepFMM(
        softening=SOFTENING,
        k_max=2,
        theta=theta,
        max_order=4,
        leaf_size=16,
        topology_backend="device",
    )


# --- far field ---------------------------------------------------------------------


def test_far_field_guard_returns_the_occupancy_on_a_system_with_far_pairs():
    """Two clumps at theta = 0.6 have far pairs; the count is the live occupancy."""
    positions, masses = _two_clumps()
    fmm = _device_fmm(theta=0.6)
    state = fmm.prepare(positions, masses)
    count = assert_far_field_is_exercised(fmm)
    assert count > 0
    assert count == int(state.num_far_pairs)
    # Occupancy, not capacity: the padded list is wider than the live count.
    assert count <= int(state.far_a.shape[0])
    # A state may be passed directly, as a rollout's carried topology would be.
    assert assert_far_field_is_exercised(state) == count


def test_far_field_guard_raises_on_the_vacuous_single_blob():
    """The cross-repo file's own standard system is a direct sum in disguise."""
    positions, masses = _one_blob()
    fmm = _device_fmm(theta=0.9)
    fmm.prepare(positions, masses)
    with pytest.raises(RuntimeError, match="no far pairs"):
        assert_far_field_is_exercised(fmm)
    assert assert_far_field_is_exercised(fmm, require=False) == 0


def test_far_field_guard_needs_a_prepared_model():
    """Without a topology there is nothing to count; say so rather than guess."""
    with pytest.raises(RuntimeError, match="prepare"):
        assert_far_field_is_exercised(_device_fmm(theta=0.6))


# --- overflow ----------------------------------------------------------------------


def test_overflow_guard_passes_a_healthy_state_and_names_the_blamed_caps():
    """No flag: silent. Flag with a cause bitmask: the blamed caps and the profile."""
    positions, masses = _two_clumps()
    fmm = _device_fmm(theta=0.6)
    state = fmm.prepare(positions, masses)
    raise_on_overflow(state, fmm)  # healthy: no raise

    far_bit = 1 << OVERFLOW_CAUSES.index("far")
    queue_bit = 1 << OVERFLOW_CAUSES.index("pair_queue")
    blown = dataclasses.replace(
        state,
        topology_overflow=jnp.asarray(True),
        overflow_causes=jnp.asarray(far_bit | queue_bit, dtype=jnp.int32),
    )
    with pytest.raises(RuntimeError) as excinfo:
        raise_on_overflow(blown, fmm)
    message = str(excinfo.value)
    assert "far, pair_queue exceeded" in message
    assert f"far={fmm.capacities.far}" in message
    assert "near" not in message.split("exceeded")[0]  # only the blamed caps are named


def test_overflow_guard_accepts_any_record_with_the_two_fields():
    """A reduction over per-step flags is not a MutualFMMState; it still works."""

    class _Record:
        topology_overflow = True
        overflow_causes = 1 << OVERFLOW_CAUSES.index("tree_depth")

    with pytest.raises(RuntimeError, match="tree_depth exceeded.*unknown profile"):
        raise_on_overflow(_Record(), object())


def test_overflow_guard_is_silent_under_a_trace():
    """Inside jit the flag is a tracer; the guard has nothing concrete to raise on."""
    positions, masses = _two_clumps()
    fmm = _device_fmm(theta=0.6)
    fmm.prepare(positions, masses)

    def traced(p, m):
        state = fmm.rebuild_state(p, m)
        raise_on_overflow(state, fmm)
        return state.num_far_pairs

    assert int(jax.jit(traced)(positions, masses)) > 0


def test_prepare_raises_through_the_shared_guard_on_a_starved_profile():
    """``prepare`` still refuses a topology that overflowed -- via ``raise_on_overflow``."""
    from jaccpot.mutual.force import MutualCapacities

    positions, masses = _two_clumps()
    healthy = _device_fmm(theta=0.6)
    healthy.prepare(positions, masses)
    caps = healthy.capacities
    starved = BlockStepFMM(
        softening=SOFTENING,
        k_max=2,
        theta=0.6,
        max_order=4,
        leaf_size=16,
        topology_backend="device",
        caps=caps._replace(far=1),
    )
    with pytest.raises(RuntimeError, match="overflowed its capacity profile: far"):
        starved.prepare(positions, masses)
