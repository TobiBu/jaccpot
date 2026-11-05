"""Forward-mode tangent of the pair-per-lane M2L CSR kernel (interpret mode, CPU).

:data:`~jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_pallas_tangent` is the
tangent op of the M2L seam (plan forward-mode, track C): the forward's target-CSR grid
with the pair body linearised in-kernel, ``K`` tangents per launch, and the analytic
transverse term on on-axis lanes (``rho = 0`` between the pair's centres). Pinned
here, in float64 and float32:

* ``K = 1`` and ``K = 8`` tangents equal its pure-JAX reference
  (:func:`~jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_pallas_tangent_jnp`)
  to round-off, on ragged rows with an empty row, rows of one and of several tiles,
  padding behind a live prefix, on-axis pairs (exactly on the axis and inside the
  ``rho^2 <= eps r^2`` band), a COO and a presorted (row-offset) list, and
  ``n_targets < n``; each tangent alone (``live``) too;
* the reference's primal is the Pallas forward to round-off;
* ``jax.jacfwd`` through a ``custom_jvp`` seam (primal: the production forward;
  tangent: the op) over 3 and 11 columns equals ``jax.jacfwd`` of the reference;
* central differences of the PRODUCTION forward anchor the reference, and the
  independent pure-JAX operator :func:`m2l_rot_scale_real_batch` (whose analytic
  transverse rule is validated in ``test_transverse_degeneracy_jvp.py``) agrees with
  it on every pair, on-axis ones included.

Every case asserts it has live far pairs.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from jaccpot.operators.m2l_real_rot_scale import m2l_rot_scale_real_batch
from jaccpot.pallas._tangent import drop_tangent_axis, with_tangent_axis
from jaccpot.pallas.m2l_real_csr import pallas_m2l_real_csr_supported
from jaccpot.pallas.m2l_real_csr_lanes import (
    _m2l_tangent_twin,
    m2l_real_csr_lanes_pallas,
    m2l_real_csr_lanes_pallas_tangent,
    m2l_real_csr_lanes_pallas_tangent_jnp,
)

jax.config.update("jax_enable_x64", True)

_K_LANES = 8
#: float64: sums of O(100) pair terms in another order; float32: the same, at 2^-23
_TOL = {np.float64: 1e-12, np.float32: 2e-5}


def _native_or_skip(interpret: bool) -> None:
    """Native Triton lowering needs an Ampere+ GPU; interpret mode runs anywhere."""
    if not interpret and not pallas_m2l_real_csr_supported():
        pytest.skip("native Pallas GPU lowering not available here")


def _rel(a, b) -> float:
    """Max-norm difference relative to the reference's max-norm."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


class _Case(NamedTuple):
    """One test system: nodes, their far list (COO and CSR) and its live pairs."""

    mult: Array
    cent: Array
    src: Array
    tgt: Array
    offsets: Array
    live: Array
    n: int
    nt: int
    C: int
    order: int
    pairs: tuple[np.ndarray, np.ndarray]


def _case(seed: int, order: int, dtype, *, n: int = 24, n_targets=None) -> _Case:
    """Nodes with on-axis partners and a ragged, padded far list by target.

    Nodes 0-5 sit on one vertical line (``rho == 0`` between them); 7 and 8 on
    another, and 6 one band-width off it (``rho / r ~ 1e-12``, inside ``rho^2 <= eps
    r^2`` in float64); the rest scattered. Target rows: an empty one, one of exactly a tile,
    one of two tiles and a lane, the others ragged; every on-axis node lists its
    line partners. ``n_targets`` (< n): only the leading nodes are targets, sources
    may be any node.
    """
    rng = np.random.default_rng(seed)
    C = (order + 1) ** 2
    nt = n if n_targets is None else int(n_targets)
    cent = rng.uniform(-1.0, 1.0, (n, 3)) * 4.0
    cent[0:6, :2] = [0.7, -0.3]
    cent[0:6, 2] = np.arange(6) * 1.7 - 4.0
    cent[6:9, :2] = [-1.1 + 1e-12, 0.5]
    cent[6, 0] = -1.1
    cent[6:9, 2] = np.arange(3) * 2.1
    rows = []
    for t in range(nt):
        if t == 9:
            row = []
        elif t < 6:
            row = [s for s in range(6) if s != t] + list(rng.integers(9, n, 3))
        elif t < 9:
            row = [s for s in range(6, 9) if s != t] + list(rng.integers(9, n, 2))
        elif t == 10:
            row = list(rng.integers(0, n, _K_LANES))
        elif t == 11:
            row = list(rng.integers(0, n, 2 * _K_LANES + 1))
        else:
            row = list(rng.integers(0, n, rng.integers(0, 12)))
        rows.append([s if s != t else (s + 1) % n for s in row])
    tgt = np.concatenate([np.full(len(r), t) for t, r in enumerate(rows)])
    src = np.concatenate([np.asarray(r, np.int64) for r in rows])
    pad = 5
    live = int(src.size)
    counts = np.asarray([len(r) for r in rows])
    return _Case(
        mult=jnp.asarray(rng.standard_normal((n, C)).astype(dtype)),
        cent=jnp.asarray(cent.astype(dtype)),
        src=jnp.asarray(np.concatenate([src, np.full(pad, -1)]), jnp.int32),
        tgt=jnp.asarray(np.concatenate([tgt, np.full(pad, -1)]), jnp.int32),
        offsets=jnp.asarray(np.concatenate([[0], np.cumsum(counts)]), jnp.int32),
        live=jnp.asarray(live, jnp.int32),
        n=n,
        nt=nt,
        C=C,
        order=order,
        pairs=(src, tgt),
    )


def _far_and_axis_pairs(c) -> tuple[int, int, int]:
    """Live far pairs, those exactly on the axis, those in the band off it."""
    src, tgt = c.pairs
    cent = np.asarray(c.cent, np.float64)
    d = cent[tgt] - cent[src]
    rho2 = d[:, 0] ** 2 + d[:, 1] ** 2
    eps = float(np.finfo(np.asarray(c.cent).dtype).eps)
    band = (rho2 <= eps * (rho2 + d[:, 2] ** 2)) & (d[:, 2] != 0)
    return (
        int(src.size),
        int(np.sum(band & (rho2 == 0))),
        int(np.sum(band & (rho2 > 0))),
    )


def _list(c, presorted: bool):
    """``(sources, targets-or-offsets, active_pair_count)`` of the case's list."""
    return (c.src, c.offsets if presorted else c.tgt, c.live)


def _params(c, presorted: bool, *, live=(True, True)) -> dict:
    return dict(
        order=c.order,
        k_lanes=_K_LANES,
        interpret=True,
        backend="triton",
        num_warps=1,
        n_targets=None if c.nt == c.n else c.nt,
        presorted=presorted,
        symmetric=False,
        has_active_count=True,
        live=live,
    )


def _tangents(c, k: int, seed: int = 0):
    rng = np.random.default_rng(100 + seed)
    dtype = np.asarray(c.mult).dtype
    return (
        jnp.asarray(rng.standard_normal((k, c.n, c.C)).astype(dtype)),
        jnp.asarray(rng.standard_normal((k, c.n, 3)).astype(dtype)),
    )


@pytest.mark.parametrize("variant", ["coo", "presorted", "n_targets"])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_tangent_op_matches_the_jnp_reference(variant, k, dtype):
    c = _case(1, 3, dtype, n_targets=20 if variant == "n_targets" else None)
    n_far, exact, band = _far_and_axis_pairs(c)
    assert n_far > 0 and exact > 0
    if dtype == np.float64:
        assert band > 0
    presorted = variant == "presorted"
    tm, tc = _tangents(c, k)
    args = (c.mult, c.cent, *_list(c, presorted), tm, tc)
    (got,) = m2l_real_csr_lanes_pallas_tangent(*args, **_params(c, presorted))
    (want,) = m2l_real_csr_lanes_pallas_tangent_jnp(*args, **_params(c, presorted))
    assert got.shape == (k, c.nt, c.C) and got.dtype == jnp.dtype(dtype)
    assert np.all(np.isfinite(np.asarray(got)))
    assert np.max(np.abs(np.asarray(want))) > 0
    # measured (order 3): float64 <= 7.6e-16, float32 <= 2.3e-7
    assert _rel(got, want) < _TOL[dtype], _rel(got, want)
    # the empty row gets nothing
    assert not np.any(np.asarray(got)[:, 9])


@pytest.mark.parametrize("live", [(True, False), (False, True)])
def test_each_tangent_alone(live):
    """A known-zero tangent (``live``) skips its terms; the rest is unchanged."""
    c = _case(2, 2, np.float64)
    tm, tc = _tangents(c, 2, seed=1)
    tm = tm if live[0] else jnp.zeros_like(tm)
    tc = tc if live[1] else jnp.zeros_like(tc)
    args = (c.mult, c.cent, *_list(c, False), tm, tc)
    (got,) = m2l_real_csr_lanes_pallas_tangent(*args, **_params(c, False, live=live))
    (full,) = m2l_real_csr_lanes_pallas_tangent(*args, **_params(c, False))
    (want,) = m2l_real_csr_lanes_pallas_tangent_jnp(*args, **_params(c, False))
    assert np.max(np.abs(np.asarray(want))) > 0
    assert _rel(got, want) < 1e-12
    assert _rel(full, want) < 1e-12


def test_no_live_tangent_and_an_empty_list_give_zeros():
    c = _case(3, 2, np.float64)
    tm, tc = _tangents(c, 2)
    args = (c.mult, c.cent, *_list(c, False), tm, tc)
    (got,) = m2l_real_csr_lanes_pallas_tangent(
        *args, **_params(c, False, live=(False, False))
    )
    assert got.shape == (2, c.n, c.C) and not np.any(np.asarray(got))
    empty = jnp.zeros((0,), jnp.int32)
    (got,) = m2l_real_csr_lanes_pallas_tangent(
        c.mult, c.cent, empty, empty, c.live, tm, tc, **_params(c, False)
    )
    assert got.shape == (2, c.n, c.C) and not np.any(np.asarray(got))


@pytest.mark.parametrize("presorted", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_the_reference_primal_is_the_pallas_forward(dtype, presorted):
    c = _case(4, 3, dtype)
    src, tgt, live = _list(c, presorted)
    twin = _m2l_tangent_twin(
        c.mult,
        c.cent,
        src,
        tgt,
        live,
        order=c.order,
        n_targets=None,
        presorted=presorted,
        has_active_count=True,
    )
    want = m2l_real_csr_lanes_pallas(
        c.mult,
        c.cent,
        src,
        tgt,
        order=c.order,
        active_pair_count=live,
        k_lanes=_K_LANES,
        interpret=True,
        presorted=presorted,
    )
    got = twin(c.mult, c.cent)
    # measured: float64 1.8e-16, float32 1.3e-7 (another summation order)
    assert _rel(got, want) < (1e-14 if dtype == np.float64 else 2e-6)


@pytest.mark.parametrize("columns", [3, 11])
@pytest.mark.parametrize("presorted", [False, True])
def test_jacfwd_through_a_custom_jvp_seam(columns, presorted):
    c = _case(5, 2, np.float64)
    assert _far_and_axis_pairs(c)[0] > 0
    src, tgt, live = _list(c, presorted)
    rng = np.random.default_rng(columns)
    A = jnp.asarray(rng.standard_normal((c.n * c.C, columns)))
    B = jnp.asarray(rng.standard_normal((c.n * 3, columns)))
    params = _params(c, presorted)

    @jax.custom_jvp
    def seam(m, x):
        return m2l_real_csr_lanes_pallas(
            m,
            x,
            src,
            tgt,
            order=c.order,
            active_pair_count=live,
            k_lanes=_K_LANES,
            interpret=True,
            presorted=presorted,
        )

    @seam.defjvp
    def _rule(primals, tangents):
        out = seam(*primals)
        tk, flags = with_tangent_axis(tangents, primals)
        m, x = primals
        (t,) = drop_tangent_axis(
            m2l_real_csr_lanes_pallas_tangent(
                m, x, src, tgt, live, *tk, **{**params, "live": flags}
            )
        )
        return out, t

    twin = _m2l_tangent_twin(
        c.mult,
        c.cent,
        src,
        tgt,
        live,
        order=c.order,
        n_targets=None,
        presorted=presorted,
        has_active_count=True,
    )

    def f(theta, fn):
        m = c.mult + (A @ theta).reshape(c.n, c.C)
        x = c.cent + 0.1 * (B @ theta).reshape(c.n, 3)
        return fn(m, x)

    theta = jnp.zeros(columns)
    jac = jax.jacfwd(lambda th: f(th, seam))(theta)
    want = jax.jacfwd(lambda th: f(th, twin))(theta)
    assert jac.shape == (c.n, c.C, columns)
    # measured: 1.6e-16
    assert _rel(jac, want) < 1e-12
    # the seam's primal is the production forward, bitwise
    y, _ = jax.jvp(seam, (c.mult, c.cent), (c.mult, c.cent))
    assert np.array_equal(np.asarray(y), np.asarray(seam(c.mult, c.cent)))
    with pytest.raises(NotImplementedError, match="ad_mode='reverse'"):
        jax.grad(lambda th: jnp.sum(f(th, seam)))(theta)


def test_central_differences_and_the_operator_rule_anchor_the_reference():
    """Central differences of the production forward along random directions, and
    ``jax.jvp`` of :func:`m2l_rot_scale_real_batch` (its own analytic transverse rule)
    over the same pairs, both agree with the reference -- on-axis pairs included."""
    c = _case(6, 3, np.float64)
    n_far, exact, band = _far_and_axis_pairs(c)
    assert n_far > 0 and exact > 0 and band > 0
    src, tgt, live = _list(c, False)
    tm, tc = _tangents(c, 3, seed=6)
    (want,) = m2l_real_csr_lanes_pallas_tangent_jnp(
        c.mult, c.cent, src, tgt, live, tm, tc, **_params(c, False)
    )
    prod = jax.jit(
        lambda m, x: m2l_real_csr_lanes_pallas(
            m,
            x,
            src,
            tgt,
            order=c.order,
            active_pair_count=live,
            k_lanes=_K_LANES,
            interpret=True,
        )
    )
    h = 1e-6
    for k in range(3):
        up = prod(c.mult + h * tm[k], c.cent + h * tc[k])
        dn = prod(c.mult - h * tm[k], c.cent - h * tc[k])
        fd = (np.asarray(up) - np.asarray(dn)) / (2.0 * h)
        # measured: <= 6.9e-11 (round-off of the differences); without the on-axis
        # term the tangent misses by 6e-2
        assert _rel(want[k], fd) < 1e-7, _rel(want[k], fd)

    psrc, ptgt = (jnp.asarray(a) for a in c.pairs)

    def operator(m, x):
        contrib = m2l_rot_scale_real_batch(m[psrc], x[ptgt] - x[psrc], order=c.order)
        return jax.ops.segment_sum(contrib, ptgt, num_segments=c.n)

    ind = jax.vmap(lambda a, b: jax.jvp(operator, (c.mult, c.cent), (a, b))[1])(tm, tc)
    # measured: 6.4e-16
    assert _rel(want, ind) < 1e-12


@pytest.mark.parametrize("k", [1, 8])
def test_native_gpu_parity(k):
    """The Triton tangent kernel against the reference, natively (skips without an
    Ampere card)."""
    _native_or_skip(False)
    c = _case(7, 3, np.float32)
    tm, tc = _tangents(c, k)
    args = (c.mult, c.cent, *_list(c, False), tm, tc)
    params = {**_params(c, False), "interpret": False}
    (got,) = m2l_real_csr_lanes_pallas_tangent(*args, **params)
    (want,) = m2l_real_csr_lanes_pallas_tangent_jnp(*args, **params)
    assert _rel(got, want) < _TOL[np.float32]
