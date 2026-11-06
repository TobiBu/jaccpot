"""The forward-mode tangent-op contract (:mod:`jaccpot.pallas._tangent`) on a toy kernel.

The toy kernel indexes its block with ``pl.program_id``, which is exactly what makes
``pallas_call``'s generic JVP rule unusable on the production kernels; the first
test pins that, so the rest are not vacuous. Everything runs in Pallas interpret
mode on CPU, in float64.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pl = pytest.importorskip("jax.experimental.pallas")

from jaccpot.pallas._tangent import (  # noqa: E402
    MAX_TANGENT_K,
    TANGENT_K_BUCKETS,
    drop_tangent_axis,
    linearize_lanes,
    make_tangent_op,
    tangent_k_bucket,
    with_tangent_axis,
)

_BLOCK = 8
_N = 32


def _body(x, m):
    return m * jnp.sin(x) + x * x * m


def _primal_kernel(x_ref, m_ref, o_ref):
    rows = pl.ds(pl.program_id(0) * _BLOCK, _BLOCK)
    o_ref[rows] = _body(x_ref[rows], m_ref[rows])


def _whole(a):
    return pl.BlockSpec(a.shape, lambda i: (0,) * a.ndim)


def _primal(x, m):
    return pl.pallas_call(
        _primal_kernel,
        grid=(x.shape[0] // _BLOCK,),
        in_specs=[_whole(x), _whole(m)],
        out_specs=_whole(x),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=True,
    )(x, m)


def _tangent_kernel(x_ref, m_ref, tx_ref, tm_ref, o_ref, *, k):
    rows = pl.ds(pl.program_id(0) * _BLOCK, _BLOCK)
    _, lin = linearize_lanes(_body, (x_ref[rows], m_ref[rows]))
    for j in range(k):
        o_ref[j, rows] = lin(tx_ref[j, rows], tm_ref[j, rows])


_LAUNCHES: list[int] = []


def _tangent_impl(x, m, tx, tm, *, live):
    del live
    k = int(tx.shape[0])
    _LAUNCHES.append(k)
    return [
        pl.pallas_call(
            functools.partial(_tangent_kernel, k=k),
            grid=(x.shape[0] // _BLOCK,),
            in_specs=[_whole(x), _whole(m), _whole(tx), _whole(tm)],
            out_specs=_whole(tx),
            out_shape=jax.ShapeDtypeStruct(tx.shape, tx.dtype),
            interpret=True,
        )(x, m, tx, tm)
    ]


_toy_tangent = make_tangent_op("toy_test", _tangent_impl, num_primals=2)


@jax.custom_jvp
def _seam(x, m):
    return _primal(x, m)


@_seam.defjvp
def _seam_jvp(primals, tangents):
    out = _seam(*primals)
    tk, live = with_tangent_axis(tangents, primals)
    (t,) = drop_tangent_axis(_toy_tangent(*primals, *tk, live=live))
    return out, t


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    with jax.enable_x64(True):
        return {
            "x": jnp.asarray(rng.normal(size=_N)),
            "m": jnp.asarray(rng.uniform(0.5, 1.5, size=_N)),
            "dx": jnp.asarray(rng.normal(size=_N)),
            "dm": jnp.asarray(rng.normal(size=_N)),
            "A": jnp.asarray(rng.normal(size=(_N, 11))),
            "B": jnp.asarray(rng.normal(size=(_N, 11))),
        }


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


def test_pallas_generic_jvp_dies_on_program_id(data):
    """The premise: without a tangent op, forward mode cannot reach these kernels."""
    with pytest.raises(AssertionError):
        jax.jvp(_primal, (data["x"], data["m"]), (data["dx"], data["dm"]))


def test_jvp_primal_is_bitwise_and_tangent_matches_the_reference(data):
    x, m, dx, dm = data["x"], data["m"], data["dx"], data["dm"]
    y, t = jax.jvp(_seam, (x, m), (dx, dm))
    _, t_ref = jax.jvp(_body, (x, m), (dx, dm))
    assert bool(jnp.all(y == _primal(x, m)))
    np.testing.assert_allclose(t, t_ref, rtol=1e-13, atol=1e-13)
    yj, tj = jax.jit(lambda *a: jax.jvp(_seam, a[:2], a[2:]))(x, m, dx, dm)
    assert bool(jnp.all(yj == y))
    np.testing.assert_allclose(tj, t_ref, rtol=1e-13, atol=1e-13)


def test_a_zero_tangent_input_is_dense_zeros(data):
    x, m, dx = data["x"], data["m"], data["dx"]
    _, t = jax.jvp(lambda a: _seam(a, m), (x,), (dx,))
    _, t_ref = jax.jvp(lambda a: _body(a, m), (x,), (dx,))
    np.testing.assert_allclose(t, t_ref, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("columns", [1, 3, 8, 11])
def test_jacfwd_folds_columns_into_bucketed_launches(data, columns):
    x, m, A, B = data["x"], data["m"], data["A"][:, :columns], data["B"][:, :columns]

    def f(theta, fn):
        return fn(x + A @ theta, m + 0.01 * (B @ theta))

    theta = jnp.zeros(columns)
    _LAUNCHES.clear()
    jac = jax.jacfwd(lambda th: f(th, _seam))(theta)
    jac_ref = jax.jacfwd(lambda th: f(th, _body))(theta)
    np.testing.assert_allclose(jac, jac_ref, rtol=1e-13, atol=1e-13)
    expected = []
    for start in range(0, columns, MAX_TANGENT_K):
        expected.append(tangent_k_bucket(min(MAX_TANGENT_K, columns - start)))
    assert _LAUNCHES == expected
    jac_jit = jax.jit(jax.jacfwd(lambda th: f(th, _seam)))(theta)
    np.testing.assert_allclose(jac_jit, jac_ref, rtol=1e-13, atol=1e-13)


def test_columns_one_at_a_time_equal_the_batched_jacobian(data):
    """K-batched tangents are K single-tangent jvps, to round-off.

    Not bitwise: the K = 8 and K = 1 kernels are different XLA programs, and on CPU
    they contract different multiply-adds (measured: 1 ulp, 2.2e-16 absolute).
    """
    x, m, A, B = data["x"], data["m"], data["A"], data["B"]
    jac = jax.jacfwd(lambda th: _seam(x + A @ th, m + 0.01 * (B @ th)))(jnp.zeros(11))
    for j in range(11):
        _, col = jax.jvp(_seam, (x, m), (A[:, j], 0.01 * B[:, j]))
        np.testing.assert_allclose(jac[:, j], col, rtol=1e-14, atol=1e-15)


def test_linearize_splits_the_primal_and_reuses_it(data):
    x, m, dx, dm = data["x"], data["m"], data["dx"], data["dm"]
    y, lin = jax.linearize(_seam, x, m)
    assert bool(jnp.all(y == _primal(x, m)))
    for a, b in [(dx, dm), (dm, dx), (dx, jnp.zeros_like(dm))]:
        _, ref = jax.jvp(_seam, (x, m), (a, b))
        np.testing.assert_allclose(lin(a, b), ref, rtol=1e-14, atol=1e-14)


def test_vmap_over_primals_falls_back_to_one_member_at_a_time(data):
    x, m, dx, dm = data["x"], data["m"], data["dx"], data["dm"]
    xs = jnp.stack([x, x + 0.1, x - 0.2])
    dxs = jnp.stack([dx, dm, dx])

    def tan(fn):
        return jax.vmap(lambda a, b: jax.jvp(fn, (a, m), (b, dm))[1])(xs, dxs)

    np.testing.assert_allclose(tan(_seam), tan(_body), rtol=1e-13, atol=1e-13)


def test_grad_of_a_forward_seam_refuses_with_the_reverse_hint(data):
    with pytest.raises(NotImplementedError, match="ad_mode='reverse'"):
        jax.grad(lambda a: jnp.sum(_seam(a, data["m"])))(data["x"])


def test_second_order_refuses_clearly(data):
    x, m, A = data["x"], data["m"], data["A"]
    with pytest.raises(NotImplementedError, match="second-order"):
        jax.jacfwd(jax.jacfwd(lambda th: jnp.sum(_seam(x + A @ th, m))))(jnp.zeros(11))


def test_buckets():
    assert TANGENT_K_BUCKETS == (1, 2, 4, 8)
    assert [tangent_k_bucket(k) for k in range(1, 9)] == [1, 2, 4, 4, 8, 8, 8, 8]
    for bad in (0, 9):
        with pytest.raises(ValueError):
            tangent_k_bucket(bad)


def test_mismatched_tangent_axes_are_refused(data):
    x, m = data["x"], data["m"]
    with pytest.raises(ValueError, match="same leading tangent axis"):
        _toy_tangent(x, m, jnp.zeros((2, _N)), jnp.zeros((3, _N)), live=(True, True))


# ---------------------------------------------------------------------------
# Dead tangents (round 2): symbolic zeros are (1, ...) placeholders, never K copies.
# ---------------------------------------------------------------------------

_SEEN: list[tuple] = []


def _tangent_kernel_live(x_ref, m_ref, tx_ref, tm_ref, o_ref, *, k, live):
    rows = pl.ds(pl.program_id(0) * _BLOCK, _BLOCK)
    _, lin = linearize_lanes(_body, (x_ref[rows], m_ref[rows]))
    zero = jnp.zeros((_BLOCK,), x_ref.dtype)
    for j in range(k):
        tx = tx_ref[j, rows] if live[0] else zero
        tm = tm_ref[j, rows] if live[1] else zero
        o_ref[j, rows] = lin(tx, tm)


def _tangent_impl_placeholders(x, m, tx, tm, *, live):
    """A ``dead_placeholders=True`` implementation: K from the live operands."""
    k = max(int(tx.shape[0]), int(tm.shape[0]))
    _SEEN.append((tuple(tx.shape), tuple(tm.shape), tuple(live)))
    return [
        pl.pallas_call(
            functools.partial(_tangent_kernel_live, k=k, live=tuple(live)),
            grid=(x.shape[0] // _BLOCK,),
            in_specs=[_whole(x), _whole(m), _whole(tx), _whole(tm)],
            out_specs=_whole(jax.ShapeDtypeStruct((k, _N), x.dtype)),
            out_shape=jax.ShapeDtypeStruct((k, _N), x.dtype),
            interpret=True,
        )(x, m, tx, tm)
    ]


def _recording_impl(x, m, tx, tm, *, live):
    _SEEN.append((tuple(tx.shape), tuple(tm.shape), tuple(live)))
    return _tangent_impl(x, m, tx, tm, live=live)


_toy_dense = make_tangent_op("toy_dense_test", _recording_impl, num_primals=2)
_toy_placeholders = make_tangent_op(
    "toy_placeholder_test",
    _tangent_impl_placeholders,
    num_primals=2,
    dead_placeholders=True,
    abstract_eval=lambda x, m, tx, tm, **_: [
        jax.ShapeDtypeStruct((tx.shape[0], _N), x.dtype)
    ],
)


def _make_seam(op):
    @jax.custom_jvp
    def seam(x, m):
        return _primal(x, m)

    def rule(primals, tangents):
        out = seam(*primals)
        tk, live = with_tangent_axis(tangents, primals)
        (t,) = drop_tangent_axis(op(*primals, *tk, live=live))
        return out, t

    seam.defjvp(rule, symbolic_zeros=True)
    return seam


@pytest.mark.parametrize("columns", [3, 8, 11])
@pytest.mark.parametrize("op", ["dense", "placeholders"])
def test_a_dead_tangent_is_never_broadcast_to_the_columns(data, columns, op):
    """``jacfwd`` over ``x`` only (``m`` constant: a symbolic-zero tangent): the
    live tangent is folded to ``K``, the dead one reaches the implementation as a
    ``(1, 1)`` placeholder (``dead_placeholders=True``) or as a dense zero of the
    chunk's bucket (the default) -- never as the vmap's broadcast -- and the
    Jacobian is the reference's."""
    x, m, A = data["x"], data["m"], data["A"][:, :columns]
    seam = _make_seam(_toy_placeholders if op == "placeholders" else _toy_dense)
    _SEEN.clear()
    jac = jax.jit(jax.jacfwd(lambda th: seam(x + A @ th, m)))(jnp.zeros(columns))
    ref = jax.jacfwd(lambda th: _body(x + A @ th, m))(jnp.zeros(columns))
    np.testing.assert_allclose(jac, ref, rtol=1e-13, atol=1e-13)
    assert _SEEN, "the implementation was not traced"
    buckets = [
        tangent_k_bucket(min(MAX_TANGENT_K, columns - s))
        for s in range(0, columns, MAX_TANGENT_K)
    ]
    for (tx_shape, tm_shape, live), kb in zip(_SEEN, buckets):
        assert live == (True, False)
        assert tx_shape == (kb, _N)
        assert tm_shape == ((1, 1) if op == "placeholders" else (kb, _N))
    # the program binds the primitive with the placeholder, not a [K, N] zero
    text = str(
        jax.make_jaxpr(jax.jacfwd(lambda th: seam(x + A @ th, m)))(jnp.zeros(columns))
    )
    assert f"f64[{columns},{_N}]" in text
    assert "dead_shapes=(None, (32,))" in text


def test_a_dead_mass_and_a_live_mass_agree(data):
    """The same Jacobian with the mass tangent dead (symbolic zero) and live but
    zero (a dense zero direction): the placeholder path is exact."""
    x, m, A, B = data["x"], data["m"], data["A"][:, :5], data["B"][:, :5]
    seam = _make_seam(_toy_placeholders)
    dead = jax.jacfwd(lambda th: seam(x + A @ th, m))(jnp.zeros(5))
    live = jax.jacfwd(lambda th: seam(x + A @ th, m + 0.0 * (B @ th)))(jnp.zeros(5))
    np.testing.assert_allclose(dead, live, rtol=1e-14, atol=1e-15)


def test_all_dead_tangents_return_zeros_without_a_launch(data):
    x, m = data["x"], data["m"]
    _SEEN.clear()
    # (a closed-form abstract_eval: the implementation is never traced)
    (out,) = _toy_placeholders(
        x, m, jnp.zeros((1, _N)), jnp.zeros((1, _N)), live=(False, False)
    )
    assert out.shape == (1, _N) and bool(jnp.all(out == 0.0))
    assert _SEEN == []
