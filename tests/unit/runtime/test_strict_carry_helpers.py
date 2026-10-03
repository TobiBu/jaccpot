"""Helpers of ``strict_run_v2``'s carry: unaliasing a donated pytree, the fresh-rebuild flag."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaccpot.runtime.fmm_strict_run import (
    _fresh_compact_pair_rebuild_enabled,
    _unaliased,
)


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


@pytest.mark.parametrize(
    "fresh, unsafe, expected",
    [(None, None, True), ("0", None, False), ("1", "1", False), ("on", "0", True)],
)
def test_fresh_rebuild_flag(monkeypatch, fresh, unsafe, expected):
    for name, value in (
        ("JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD", fresh),
        ("JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE", unsafe),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert _fresh_compact_pair_rebuild_enabled() is expected
