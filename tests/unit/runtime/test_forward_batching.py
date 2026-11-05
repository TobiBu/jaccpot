"""Direction batching of forward-mode Jacobians (``jaccpot.runtime.forward_batching``).

On plain ``jnp`` functions (no FMM): :func:`jacfwd_batched` with
``directions_per_pass`` in {1, 3, D, > D} returns ``jax.jacfwd``'s Jacobian (the
same pytree structure, shapes and dtypes; values to round-off) for pytree inputs
and outputs, with and without ``has_aux``; ``None`` IS ``jax.jacfwd`` (the lowered
program identical); :func:`apply_in_chunks` equals one ``vmap``; the memory model
is monotone and suggests ``k = 1`` at 1e8 on an A100-40. The FMM gate (G-batch) is
``tests/integration/strict/test_forward_mode.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime.forward_batching import (
    apply_in_chunks,
    forward_memory_estimate,
    jacfwd_batched,
)


def _fun(x: dict) -> tuple:
    a, b = x["a"], x["b"]
    y = jnp.sin(a) * jnp.sum(b**2) + jnp.cos(a).sum()
    z = jnp.stack([jnp.prod(b), jnp.sum(a * a[::-1])])
    return {"y": y, "z": z}, jnp.sum(a)


def _point() -> dict:
    rng = np.random.default_rng(0)
    return {
        "a": jnp.asarray(rng.normal(size=(5,))),
        "b": jnp.asarray(rng.normal(size=(2, 3))),
    }


@pytest.mark.parametrize("k", [1, 3, 11, 20])
def test_jacfwd_batched_is_jacfwd(k):
    x = _point()
    want, aux_want = jax.jit(jax.jacfwd(_fun, has_aux=True))(x)
    got, aux_got = jax.jit(jacfwd_batched(_fun, directions_per_pass=k, has_aux=True))(x)
    assert jax.tree_util.tree_structure(got) == jax.tree_util.tree_structure(want)
    for g, w in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
        assert g.shape == w.shape and g.dtype == w.dtype
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-13, atol=0)
    assert float(aux_got) == float(aux_want)


def test_jacfwd_batched_without_aux_and_float32():
    x = jnp.linspace(0.1, 1.0, 7, dtype=jnp.float32)

    def f(v):
        return jnp.outer(jnp.sin(v), v)

    want = jax.jacfwd(f)(x)
    got = jax.jit(jacfwd_batched(f, directions_per_pass=2))(x)
    assert got.shape == want.shape == (7, 7, 7) and got.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6)


def test_none_is_jax_jacfwd():
    x = _point()
    a = jax.jit(jacfwd_batched(_fun, has_aux=True)).lower(x).as_text()
    b = jax.jit(jax.jacfwd(_fun, has_aux=True)).lower(x).as_text()
    assert a == b


def test_the_tangent_axis_is_k():
    """Inside the loop the tangents carry ``k`` columns: the pushed function sees
    a batch of ``k`` (here counted by a vmap of a primitive-free probe)."""
    x = jnp.arange(10.0)
    seen = []

    def f(v):
        seen.append(v.shape)
        return v * v

    jax.jit(jacfwd_batched(f, directions_per_pass=4)).lower(x)
    assert seen  # traced
    jaxpr = jax.make_jaxpr(jacfwd_batched(f, directions_per_pass=4))(x)
    text = str(jaxpr)
    assert "while" in text or "scan" in text
    assert "f64[3,4,10]" in text or "f32[3,4,10]" in text  # 3 chunks of 4


@pytest.mark.parametrize("k", [None, 1, 2, 5])
def test_apply_in_chunks_is_vmap(k):
    rng = np.random.default_rng(1)
    m = jnp.asarray(rng.normal(size=(4, 6)))
    d = {"u": jnp.asarray(rng.normal(size=(5, 6))), "s": jnp.arange(5.0)}

    def lin(t):
        return m @ t["u"] + t["s"]

    np.testing.assert_allclose(
        np.asarray(apply_in_chunks(lin, d, k)),
        np.asarray(jax.vmap(lin)(d)),
        rtol=1e-14,
    )


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_bad_directions_per_pass(bad):
    # ValueError; a float is a TypeError under JACCPOT_RUNTIME_TYPECHECK=1
    with pytest.raises((ValueError, TypeError)):
        jacfwd_batched(_fun, directions_per_pass=bad)


def test_memory_model():
    one = forward_memory_estimate(100_000_000, 1)
    assert abs(one["bytes_per_particle"] - 384.1) < 0.5
    assert one["fits"] and one["largest_k_that_fits"] == 1
    eight = forward_memory_estimate(100_000_000, 8)
    assert not eight["fits"]
    mid = forward_memory_estimate(25_000_000, 8)
    assert abs(mid["bytes_per_particle"] - (266.1 + 8 * 143.1)) < 0.5
    per = [
        forward_memory_estimate(n, 1)["bytes_per_particle"]
        for n in (10**6, 10**7, 10**9)
    ]
    assert per[0] > per[1] > per[2] > 0
    small = forward_memory_estimate(2_000_000, 1)
    assert small["largest_k_that_fits"] == 8
