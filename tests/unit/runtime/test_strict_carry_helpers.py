"""Helpers of ``strict_run_v2``'s carry: unaliasing a donated pytree, and the
segment retry's (and the runner's) host copy of a state about to be donated.

(The fresh-rebuild flag it also pinned was always on, and went in the 2026-10
cleanup, Z-b.)
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime.fmm_strict_run import _unaliased
from jaccpot.runtime.strict_carry import host_copy


def test_a_repeated_array_is_copied_once_seen():
    shared = jnp.arange(5)
    other = jnp.ones(3)
    tree = {"a": shared, "b": (shared, other), "c": None, "d": 7}
    out = _unaliased(tree)
    leaves = [out["a"], out["b"][0], out["b"][1]]
    assert out["a"] is shared  # the first occurrence is kept
    assert out["b"][0] is not shared  # the repeat is a new buffer ...
    assert bool(jnp.array_equal(out["b"][0], shared))  # ... with the same values
    assert out["b"][1] is other
    assert len({id(x) for x in leaves}) == 3
    assert out["c"] is None and out["d"] == 7


def test_an_unaliased_tree_is_returned_as_is():
    tree = (jnp.zeros(2), jnp.ones(2))
    out = _unaliased(tree)
    assert out[0] is tree[0] and out[1] is tree[1]


def test_the_host_copy_off_a_gpu_is_a_numpy_copy():
    if jax.devices()[0].platform == "gpu":
        pytest.skip("a GPU copies into pinned host memory (the next test)")
    state = jnp.arange(24.0, dtype=jnp.float32).reshape(4, 2, 3)
    copy = host_copy(state)
    assert isinstance(copy, np.ndarray)
    assert np.array_equal(copy, np.asarray(state))


@pytest.mark.skipif(
    jax.devices()[0].platform != "gpu", reason="pinned host memory is the GPU's"
)
def test_the_host_copy_on_a_gpu_survives_the_donation_it_overlaps():
    """The asynchronous pinned copy is issued, then the state is donated to a program
    that overwrites it at once: PjRt orders the donation after the copy, so the copy
    restores the start bitwise."""

    @partial(jax.jit, donate_argnums=(0,))
    def overwrite(state):
        return state * 2.0 + 1.0

    state = jax.random.normal(jax.random.PRNGKey(0), (1 << 20, 2, 3), jnp.float32)
    reference = np.asarray(state)
    copy = host_copy(state)
    assert isinstance(copy, jax.Array)
    assert copy.sharding.memory_kind == "pinned_host"
    out = overwrite(state)
    assert state.is_deleted()
    restored = jax.device_put(copy, out.sharding)
    assert np.array_equal(np.asarray(restored), reference)
    assert np.array_equal(np.asarray(out), reference * 2.0 + 1.0)
