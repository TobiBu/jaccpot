"""jaccpot's index dtype follows yggdrax's unless set; int32 sizes past their range are refused."""

from __future__ import annotations

import os

import jax.numpy as jnp
import pytest

from jaccpot.runtime import dtypes


def test_the_default_follows_yggdrax(monkeypatch):
    from yggdrax import dtypes as ydtypes

    monkeypatch.delenv("JACCPOT_INDEX_PRECISION", raising=False)
    assert jnp.dtype(dtypes._resolve_index_dtype()) == jnp.dtype(ydtypes.INDEX_DTYPE)
    if not os.environ.get("JACCPOT_INDEX_PRECISION"):
        assert jnp.dtype(dtypes.INDEX_DTYPE) == jnp.dtype(ydtypes.INDEX_DTYPE)


def test_the_variable_overrides(monkeypatch):
    from yggdrax import dtypes as ydtypes

    monkeypatch.setenv("JACCPOT_INDEX_PRECISION", "int64")
    assert jnp.dtype(dtypes._resolve_index_dtype()) == jnp.int64
    monkeypatch.setenv("JACCPOT_INDEX_PRECISION", "int32")
    assert jnp.dtype(dtypes._resolve_index_dtype()) == jnp.int32
    monkeypatch.setenv("JACCPOT_INDEX_PRECISION", "nonsense")
    assert jnp.dtype(dtypes._resolve_index_dtype()) == jnp.dtype(ydtypes.INDEX_DTYPE)


def test_sizes_past_the_int32_range_are_refused(monkeypatch):
    monkeypatch.setattr(dtypes, "INDEX_DTYPE", jnp.int32)
    dtypes.require_index_capacity(particles=10**9, near_edges=2**31 - 2)
    with pytest.raises(ValueError, match="JACCPOT_INDEX_PRECISION=int64") as err:
        dtypes.require_index_capacity(particles=10, node_coefficients=2**31)
    assert "node_coefficients" in str(err.value)
    monkeypatch.setattr(dtypes, "INDEX_DTYPE", jnp.int64)
    dtypes.require_index_capacity(node_coefficients=2**40)  # int64: no limit here
