"""Forward-mode tangents in groups of directions (:mod:`jaccpot.runtime._forward_group`).

A toy force -- a gather, a nonlinear map with a tangent op (the
:func:`~jaccpot.pallas._tangent.make_tangent_op` contract, a jnp implementation that
records the ``K`` it is called with), traced parameters, a stopped output -- whose
``jax.jacfwd`` through :func:`grouped_jvp` must equal the ungrouped one to round-off
for ``K`` in {3, 8, 11} and ``g`` in {1, 2, 4} (a partial last group included), with
the tangent op launched at ``K = g`` (plus the remainder) and the primal once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas._tangent import (
    drop_tangent_axis,
    make_tangent_op,
    with_tangent_axis,
)
from jaccpot.runtime._forward_group import (
    ENV_FORWARD_TANGENT_GROUP,
    forward_tangent_group,
    grouped_jvp,
)
from jaccpot.runtime.fused_topology import ForceParams

_N = 6
_PERM = np.array([3, 1, 0, 2, 5, 4])
_LAUNCHES: list[int] = []
_PRIMALS: list[int] = []


def _tangent_impl(x, m, tx, tm, *, live):
    del live
    _LAUNCHES.append(int(tx.shape[0]))
    return [jnp.cos(x)[None] * m[None, :, None] * tx + jnp.sin(x)[None] * tm[..., None]]


def _tangent_shapes(x, m, tx, tm, **_):
    return [jax.ShapeDtypeStruct(tx.shape, tx.dtype)]


# a closed-form abstract eval: _LAUNCHES then counts the lowerings only
_OP = make_tangent_op(
    "grouped_toy",
    _tangent_impl,
    num_primals=2,
    any_k=True,
    abstract_eval=_tangent_shapes,
)


@jax.custom_jvp
def _seam(x, m):
    _PRIMALS.append(1)
    return jnp.sin(x) * m[:, None]


@_seam.defjvp
def _seam_jvp(primals, tangents):
    out = _seam(*primals)
    tk, live = with_tangent_axis(tangents, primals)
    (t,) = drop_tangent_axis(_OP(*primals, *tk, live=live))
    return out, t


def _force(x, m, fp):
    perm = jnp.asarray(_PERM)
    y = _seam(x[perm], m[perm]) * fp.G + jnp.cos(x[perm]) * fp.eps2
    return y[jnp.argsort(perm)], jax.lax.stop_gradient(2.0 * m)


def _map(directions, group):
    x0 = jnp.linspace(0.1, 1.0, 3 * _N).reshape(_N, 3)
    m0 = jnp.linspace(1.0, 2.0, _N)

    def f(c):
        x = x0 + jnp.einsum("k,knd->nd", c, directions[:, :, :3])
        m = m0 + jnp.einsum("k,kn->n", c, directions[:, :, 3])
        fp = ForceParams(G=jnp.asarray(1.3) + c[0], eps2=0.01)
        if group is None:
            return _force(x, m, fp)
        return grouped_jvp(_force, (x, m, fp), group=group)

    return f


@pytest.mark.parametrize("k", [3, 8, 11])
@pytest.mark.parametrize("group", [1, 2, 4])
def test_grouped_jacfwd_is_the_single_pass(k, group):
    """Grouped ``jacfwd`` equals the ungrouped one; the op runs at ``K = g``."""
    directions = jnp.asarray(np.random.default_rng(k).normal(size=(k, _N, 4)))
    c0 = jnp.zeros(k)
    want = jax.jit(jax.jacfwd(_map(directions, None)))(c0)
    _LAUNCHES.clear()
    _PRIMALS.clear()
    got = jax.jit(jax.jacfwd(_map(directions, group)))(c0)
    full, rem = divmod(k, group)
    assert sorted(_LAUNCHES) == sorted([group] * min(full, 1) + [rem] * (rem > 0))
    # the primal is traced once (linearize), never per group
    assert len(_PRIMALS) == 1
    for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-13, atol=1e-13)
    # the stopped output's tangent is zero
    assert not np.any(np.asarray(got[1]))


def test_grouped_jvp_primal_and_one_direction():
    """``jax.jvp``: the primal bitwise the plain function's, one group of one."""
    directions = jnp.asarray(np.random.default_rng(0).normal(size=(2, _N, 4)))
    c0, t = jnp.zeros(2), jnp.asarray([1.0, -0.5])
    want = jax.jvp(_map(directions, None), (c0,), (t,))
    got = jax.jvp(_map(directions, 4), (c0,), (t,))
    for a, b in zip(
        jax.tree_util.tree_leaves(got[0]), jax.tree_util.tree_leaves(want[0])
    ):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    for a, b in zip(
        jax.tree_util.tree_leaves(got[1]), jax.tree_util.tree_leaves(want[1])
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-13, atol=1e-13)


def test_grouped_refuses_reverse_mode():
    """``jax.grad`` of a grouped forward-mode function reaches the refusal."""
    directions = jnp.asarray(np.random.default_rng(1).normal(size=(2, _N, 4)))
    f = _map(directions, 2)
    with pytest.raises(NotImplementedError):
        jax.grad(lambda c: jnp.sum(f(c)[0]))(jnp.zeros(2))


def test_the_knob(monkeypatch):
    """Off unless forward mode is on and the variable holds g >= 1."""
    from jaccpot.runtime.grad_options import ad_mode_override

    monkeypatch.delenv(ENV_FORWARD_TANGENT_GROUP, raising=False)
    with ad_mode_override("forward"):
        assert forward_tangent_group() is None
    monkeypatch.setenv(ENV_FORWARD_TANGENT_GROUP, "2")
    assert forward_tangent_group() is None  # reverse (default) mode
    with ad_mode_override("forward"):
        assert forward_tangent_group() == 2
        monkeypatch.setenv(ENV_FORWARD_TANGENT_GROUP, "0")
        assert forward_tangent_group() is None
        monkeypatch.setenv(ENV_FORWARD_TANGENT_GROUP, "two")
        with pytest.raises(ValueError):
            forward_tangent_group()
