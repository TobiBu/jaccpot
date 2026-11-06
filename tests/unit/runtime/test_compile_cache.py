"""The opt-in persistent compilation cache (``JACCPOT_COMPILATION_CACHE_DIR``).

Unset, nothing changes; set, a compile writes an entry into the directory (CPU, a
tiny program, the minimum compile time lowered to 0), and a second call is a no-op.
The process's cache state is restored after each test.
"""

from __future__ import annotations

import os
from typing import Iterator

import jax
import jax.numpy as jnp
import pytest

from jaccpot.runtime import compile_cache


@pytest.fixture(autouse=True)
def _restore() -> Iterator[None]:
    from jax._src import compilation_cache as cc

    keys = (
        "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
        "jax_compilation_cache_max_size",
    )
    saved = {k: getattr(jax.config, k) for k in keys}
    yield
    for k, v in saved.items():
        jax.config.update(k, v)
    cc.reset_cache()
    compile_cache._applied = None


def test_unset_is_a_no_op(monkeypatch):
    monkeypatch.delenv(compile_cache.COMPILATION_CACHE_ENV, raising=False)
    before = jax.config.jax_compilation_cache_dir
    assert compile_cache.maybe_enable_compilation_cache() is None
    assert jax.config.jax_compilation_cache_dir == before


def test_set_populates_the_directory(monkeypatch, tmp_path):
    cache = tmp_path / "jax-cache"
    monkeypatch.setenv(compile_cache.COMPILATION_CACHE_ENV, str(cache))
    monkeypatch.setenv("JACCPOT_COMPILATION_CACHE_MIN_COMPILE_S", "0")
    # something compiled before the cache is set (jax fixes it at the first
    # compile): the helper must re-arm it
    jax.jit(lambda x: x - 1.0)(jnp.ones(3)).block_until_ready()
    got = compile_cache.maybe_enable_compilation_cache()
    assert got == os.path.abspath(cache)
    assert jax.config.jax_compilation_cache_dir == got
    assert compile_cache.maybe_enable_compilation_cache() == got  # idempotent

    @jax.jit
    def f(x):
        return jnp.sin(x) * 3.0 + jnp.cumsum(x)

    f(jnp.arange(17.0)).block_until_ready()
    entries = [p for p in cache.rglob("*") if p.is_file()]
    assert entries, "the compile wrote no cache entry"


def test_the_runner_and_the_plan_apply_it(monkeypatch):
    """``ProductionRunner`` and ``GradientPlan`` call the helper on construction."""
    from jaccpot import FastMultipoleMethod, ProductionRunner
    from jaccpot.production_grad import GradientPlan

    calls = []
    monkeypatch.setattr(
        compile_cache, "maybe_enable_compilation_cache", lambda: calls.append(1)
    )
    ProductionRunner(FastMultipoleMethod(preset="galaxy"))
    assert len(calls) == 1
    GradientPlan(
        engine=None,
        segments=(),
        start={},
        params={},
        boundary=None,
        external_fn=None,
        forward=None,
    )
    assert len(calls) == 2
