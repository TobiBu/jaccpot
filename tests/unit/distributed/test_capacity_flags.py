"""The fused lane's capacity flags must be able to fire.

Under ``jit`` a saturated walk or leaf partition cannot raise; it saturates
``compact_far_pairs.far_pair_count`` to the full far-pair buffer length, and the
traced guard reads that. The multi-GPU lane's local flag used to read three
attributes ``LargeNPreparedState`` does not have, so it was the constant ``False``.
These tests pin that it now depends on the state, using a constructed state (the
fused pipeline does not engage on CPU) -- and that the OLD reading would have
stayed ``False`` on the same saturated input, so the test is not vacuous.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.distributed.fused import _fold_cross_flags, _local_overflow
from jaccpot.runtime.capacity_guard import fused_state_capacity_ok

FAR_BUFFER = 64  # directed far-pair buffer length (2 x the walk's far cap)


def _state(far_pair_count, *, neighbor_counts=(3, 0, 5, 2)):
    offsets = np.concatenate([[0], np.cumsum(neighbor_counts)]).astype(np.int32)
    return SimpleNamespace(
        neighbor_list=SimpleNamespace(offsets=jnp.asarray(offsets)),
        compact_far_pairs=SimpleNamespace(
            sources=jnp.full((FAR_BUFFER,), -1, jnp.int32),
            far_pair_count=jnp.asarray(far_pair_count, jnp.int32),
        ),
        nearfield_target_block_source_leaf_ids_padded=None,
    )


def test_a_saturated_far_pair_count_fails_the_guard():
    assert bool(fused_state_capacity_ok(_state(10)))
    # an overflow anywhere in the walk saturates the count to the buffer length
    assert not bool(fused_state_capacity_ok(_state(FAR_BUFFER)))


def test_the_recorded_caps_still_refine_the_check():
    caps = {"compact_far_pair_capacity": 32, "max_neighbors_per_leaf_used": 6}
    assert bool(fused_state_capacity_ok(_state(10), traced_caps=caps))
    assert not bool(fused_state_capacity_ok(_state(40), traced_caps=caps))
    # a neighbour row that reaches its cap means entries were dropped
    assert not bool(
        fused_state_capacity_ok(_state(10, neighbor_counts=(6, 1)), traced_caps=caps)
    )


def test_the_mesh_local_flag_fires_and_the_old_reading_would_not():
    saturated = _state(FAR_BUFFER)
    assert bool(_local_overflow(saturated))
    assert not bool(_local_overflow(_state(10)))
    # NEGATIVE CONTROL: the attributes the old implementation read do not exist on
    # the state, so it returned False on this very input
    old_reading = any(
        getattr(saturated, name, None) is not None
        for name in ("leaf_capacity_overflow", "walk_overflow", "capacity_overflow")
    )
    assert not old_reading


def test_the_flag_is_a_function_of_the_state_under_trace():
    """Under jit the flag must depend on the traced count, not be a constant."""

    def flag(count):
        return _local_overflow(_state(count))

    jaxpr = jax.make_jaxpr(flag)(jnp.asarray(10, jnp.int32)).jaxpr
    out = jaxpr.outvars[0]
    # a constant flag would be a literal output, produced by no equation
    assert any(out in eqn.outvars for eqn in jaxpr.eqns), "flag folded to a constant"
    assert bool(jax.jit(flag)(jnp.asarray(FAR_BUFFER, jnp.int32)))
    assert not bool(jax.jit(flag)(jnp.asarray(3, jnp.int32)))


@pytest.mark.parametrize("channel", ["flag_sink", "near_sink", "record"])
def test_every_cross_channel_reaches_the_device_flag(channel):
    hook = SimpleNamespace(flag_sink={})
    near, record = {}, {}
    true = jnp.asarray(True)
    if channel == "flag_sink":
        hook.flag_sink["far"] = true
    elif channel == "near_sink":
        near["overflow"] = true
    else:
        record["overflow"] = true
    assert bool(_fold_cross_flags(jnp.asarray(False), hook, near, record))
    # and nothing raised -> nothing reported
    assert not bool(
        _fold_cross_flags(jnp.asarray(False), SimpleNamespace(flag_sink={}), {}, {})
    )


def test_no_hook_and_no_sinks_is_the_local_flag():
    assert not bool(_fold_cross_flags(jnp.asarray(False), None, None, None))
    assert bool(_fold_cross_flags(jnp.asarray(True), None, None, None))
