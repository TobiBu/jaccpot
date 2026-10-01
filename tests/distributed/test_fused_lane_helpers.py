"""The mesh-facing helpers of the distributed fused lane.

These run without a GPU and without the fused lane itself: they cover the
bookkeeping the mesh forces -- stacking per-device states, the box all-reduce and
the overflow OR -- on forced CPU devices. The fused pipeline itself needs the
large-N production profile, which does not engage on CPU, so its own gate is a
bench probe (``bench/multigpu_fused_shardmap_probe.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from jaccpot.distributed.fused import (
    assemble_prepared_states,
    global_mesh_bounds,
    reduce_flag_across_mesh,
    stack_prepared_states,
)

AXIS = "gpus"


def _mesh(n):
    devices = jax.devices()
    if len(devices) < n:
        pytest.skip(f"needs {n} devices, have {len(devices)}")
    return jax.sharding.Mesh(
        np.asarray(devices[:n]), (AXIS,), axis_types=(jax.sharding.AxisType.Auto,)
    )


def test_stacking_gives_every_leaf_a_device_axis():
    states = [
        {"a": jnp.arange(4.0) + d, "b": {"c": jnp.full((2, 3), float(d))}}
        for d in range(3)
    ]
    stacked = stack_prepared_states(states)
    assert stacked["a"].shape == (3, 4)
    assert stacked["b"]["c"].shape == (3, 2, 3)
    assert float(stacked["a"][2][0]) == 2.0


def test_stacking_names_the_leaf_that_disagrees():
    """A bare "cannot stack" over 55 anonymous leaves is not actionable."""
    good = {"a": jnp.zeros((4,)), "caps": {"far": jnp.zeros((8,))}}
    bad = {"a": jnp.zeros((4,)), "caps": {"far": jnp.zeros((16,))}}
    with pytest.raises(ValueError, match=r"far"):
        stack_prepared_states([good, bad])
    with pytest.raises(ValueError, match="capacity plan"):
        stack_prepared_states([good, bad])


def test_stacking_rejects_mismatched_structures_and_empty_input():
    with pytest.raises(ValueError, match="at least one state"):
        stack_prepared_states([])
    with pytest.raises(ValueError, match="pytree structure"):
        stack_prepared_states([{"a": jnp.zeros((2,))}, {"b": jnp.zeros((2,))}])


def test_the_box_is_global_and_identical_on_every_device():
    """Every device must encode into one Morton frame or its codes travel badly."""
    mesh = _mesh(2)
    rng = np.random.default_rng(0)
    left = rng.uniform(-3.0, -1.0, (8, 3)).astype(np.float32)
    right = rng.uniform(1.0, 4.0, (8, 3)).astype(np.float32)
    # FLAT (ndev*n, 3): shard_map keeps the mapped axis, so a (ndev, n, 3) input
    # would arrive as (1, n, 3) and min(axis=0) would reduce the DEVICE axis.
    positions = jnp.asarray(np.concatenate([left, right]))

    fn = jax.jit(
        jax.shard_map(
            lambda p: global_mesh_bounds(p, axis_name=AXIS),
            mesh=mesh,
            in_specs=(P(AXIS),),
            out_specs=(P(), P()),
            check_vma=False,
        )
    )
    lo, hi = fn(positions)
    lo, hi = np.asarray(lo), np.asarray(hi)

    everything = np.concatenate([left, right])
    assert np.all(lo <= everything.min(axis=0))
    assert np.all(hi >= everything.max(axis=0))
    # A per-shard box would stop near -1 on the left device; the global one must not.
    assert hi.min() > 0.0


def test_a_fully_padded_shard_does_not_poison_the_box():
    """Masking dead rows with +-inf would reduce to inf and take the mesh with it."""
    mesh = _mesh(2)
    live = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], np.float32)
    positions = jnp.asarray(np.concatenate([live, np.full((2, 3), 7.0, np.float32)]))
    num_valid = jnp.asarray([2, 0], jnp.int32)

    fn = jax.jit(
        jax.shard_map(
            lambda p, n: global_mesh_bounds(p, num_valid=n[0], axis_name=AXIS),
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS)),
            out_specs=(P(), P()),
            check_vma=False,
        )
    )
    lo, hi = fn(positions, num_valid)
    assert np.all(np.isfinite(np.asarray(lo)))
    assert np.all(np.isfinite(np.asarray(hi)))


def test_one_devices_overflow_is_every_devices():
    """A guard read per device lets one card truncate while its peers look clean."""
    mesh = _mesh(2)

    fn = jax.jit(
        jax.shard_map(
            lambda f: reduce_flag_across_mesh(f[0], axis_name=AXIS),
            mesh=mesh,
            in_specs=(P(AXIS),),
            out_specs=P(),
            check_vma=False,
        )
    )
    assert bool(fn(jnp.asarray([False, True])))
    assert not bool(fn(jnp.asarray([False, False])))


def test_the_mapped_axis_is_not_removed_by_shard_map():
    """The convention this module depends on, pinned.

    ``in_specs=P("gpus")`` gives a device ``(1, n, ...)``, not ``(n, ...)``. A
    body written for the squeezed shape reduces the DEVICE axis instead of the
    particles, and every collective after it compares whole shards elementwise --
    with plausible shapes and no error. Passing particle arrays FLAT is what
    keeps that from happening.
    """
    mesh = _mesh(2)
    stacked = jnp.asarray(np.zeros((2, 8, 3), np.float32))
    flat = jnp.asarray(np.zeros((16, 3), np.float32))

    seen = {}

    def record(key):
        def body(x):
            seen[key] = x.shape
            return jnp.zeros(())

        return jax.jit(
            jax.shard_map(
                body, mesh=mesh, in_specs=(P(AXIS),), out_specs=P(), check_vma=False
            )
        )

    record("stacked")(stacked)
    record("flat")(flat)

    assert seen["stacked"] == (1, 8, 3), "shard_map unexpectedly squeezed the axis"
    assert seen["flat"] == (8, 3)


def test_assembled_states_equal_the_stacked_ones_and_are_sharded():
    """Per-device assembly is the stack, placed: same values, one shard per device."""
    mesh = _mesh(2)
    states = [
        {"a": jnp.arange(4.0) + d, "b": {"c": jnp.full((2, 3), float(d))}}
        for d in range(2)
    ]
    stacked = stack_prepared_states(states)
    assembled = assemble_prepared_states(states, mesh, axis_name=AXIS)
    for got, want in zip(
        jax.tree_util.tree_leaves(assembled), jax.tree_util.tree_leaves(stacked)
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        assert got.sharding.spec == P(AXIS)
        # each device holds exactly its own slice
        for shard in got.addressable_shards:
            assert shard.data.shape[0] == 1


def test_assembly_needs_one_state_per_device():
    mesh = _mesh(2)
    with pytest.raises(ValueError, match="for a mesh of"):
        assemble_prepared_states([{"a": jnp.zeros(3)}], mesh, axis_name=AXIS)
