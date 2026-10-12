"""A helper of ``strict_run_v2``'s carry: unaliasing a donated pytree.

(The fresh-rebuild flag it also pinned was always on, and went in the 2026-10
cleanup, Z-b.)
"""

from __future__ import annotations

import jax.numpy as jnp

from jaccpot.runtime.fmm_strict_run import _unaliased


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
