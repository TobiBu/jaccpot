"""Forward mode of the blocked Pallas leaf P2M: the tangent op vs ``jax.jvp`` of its twin.

``p2m_real_leaves_pallas_tangent`` (:mod:`jaccpot.pallas.p2m_real_leaf`) is the
tangent op of the forward-mode seam of ``p2m_real_leaves_pallas_cvjp``. In Pallas
interpret mode on CPU it is checked against ``p2m_real_leaves_pallas_tangent_jnp``
(``vmap`` over ``K`` of ``jax.jvp`` of the twin ``p2m_real_leaves_jnp``), whose
primal is checked against the Pallas forward and whose tangent is anchored twice:
by finite differences and by ``jax.jvp`` of the batched production reference
``_p2m_leaves_real``. The trees carry empty padding leaves, single-particle leaves
(``delta == 0`` exactly, where the floored radii carry the derivative) and leaves
wider than one chunk.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax.experimental.pallas")
pytest.importorskip("yggdrax")

from jaccpot.pallas._tangent import drop_tangent_axis, with_tangent_axis  # noqa: E402
from jaccpot.pallas.p2m_real_leaf import (  # noqa: E402
    _p2m_tangent_impl,
    p2m_real_leaves_jnp,
    p2m_real_leaves_pallas,
    p2m_real_leaves_pallas_cvjp,
    p2m_real_leaves_pallas_tangent,
    p2m_real_leaves_pallas_tangent_jnp,
    pallas_p2m_real_leaf_supported,
)
from jaccpot.upward.real_tree_expansions import _p2m_leaves_real  # noqa: E402

_TOL = {jnp.float64: 1e-12, jnp.float32: 2e-5}


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


def _plummer(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


class _Tree:
    """A cell tree's P2M operands (the seam's) and its node layout."""

    def __init__(self, n, leaf, seed):
        from yggdrax._tree_impl import (
            build_static_cells_tree,  # pyright: ignore[reportAttributeAccessIssue]
        )
        from yggdrax.bounds import infer_bounds
        from yggdrax.tree_moments import compute_tree_mass_moments

        P = jnp.asarray(_plummer(n, seed))
        M = jnp.asarray(np.random.default_rng(seed + 1).uniform(0.5, 1.5, n))
        topo, ps, ms, _ = build_static_cells_tree(
            P,
            M,
            infer_bounds(P),
            leaf_size=leaf,
            leaf_capacity=1024,
            return_reordered=True,
        )
        self.topo = topo
        self.ni = int(topo.left_child.shape[0])
        self.tot = int(topo.parent.shape[0])
        self.leaf = leaf
        ranges = np.asarray(topo.node_ranges)[self.ni :]
        counts = ranges[:, 1] - ranges[:, 0] + 1
        # padding leaves, single-particle leaves and leaves wider than a chunk (8)
        assert (counts <= 0).any() and (counts == 1).any() and (counts > 8).any()
        com = compute_tree_mass_moments(topo, ps, ms).center_of_mass
        self.positions, self.masses = ps, ms
        self.com = jnp.asarray(com)
        self.leaf_centers = self.com[self.ni :]
        self.leaf_ranges = topo.node_ranges[self.ni :]

    def args(self, dtype):
        return (
            self.positions.astype(dtype),
            self.masses.astype(dtype),
            self.leaf_centers.astype(dtype),
            self.leaf_ranges,
        )

    def static(self, order):
        return dict(order=order, num_internal=self.ni, total_nodes=self.tot)


@pytest.fixture(scope="module")
def tree():
    with jax.enable_x64(True):
        return _Tree(600, 16, 1)


def _tangents(args, k, seed):
    rng = np.random.default_rng(100 + seed)
    return tuple(
        jnp.asarray(rng.normal(size=(k,) + a.shape), a.dtype) for a in args[:3]
    )


def _op(tree, args, tangents, order, live=(True, True, True)):
    return p2m_real_leaves_pallas_tangent(
        *args,
        *tangents,
        **tree.static(order),
        leaf_width=tree.leaf,
        interpret=True,
        backend="triton",
        num_warps=None,
        live=live,
    )


def _ref(tree, args, tangents, order):
    return p2m_real_leaves_pallas_tangent_jnp(*args, *tangents, **tree.static(order))


def _rel_max(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-300))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("order", [2, 5])
def test_jnp_twin_primal_is_the_pallas_forward(tree, order, dtype):
    args = tree.args(dtype)
    want = p2m_real_leaves_pallas(
        *args, **tree.static(order), leaf_width=tree.leaf, interpret=True
    )
    got = p2m_real_leaves_jnp(*args, **tree.static(order))
    assert got.dtype == want.dtype and got.shape == want.shape
    assert _rel_max(got, want) < _TOL[dtype]
    assert not np.any(np.asarray(got)[: tree.ni])


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("order", [2, 5])
def test_tangent_op_equals_the_jvp_of_the_twin(tree, order, k, dtype):
    """Measured max rel error (2026-11-03, CPU interpret, p2/p5): fp64 2.2e-16, fp32 2.0e-7."""
    args = tree.args(dtype)
    tangents = _tangents(args, k, order)
    (got,) = _op(tree, args, tangents, order)
    (want,) = _ref(tree, args, tangents, order)
    assert got.shape == (k, tree.tot, (order + 1) ** 2) and got.dtype == dtype
    assert np.linalg.norm(np.asarray(want)) > 0
    assert _rel_max(got, want) < _TOL[dtype], _rel_max(got, want)
    # internal rows and empty leaves carry no tangent
    assert not np.any(np.asarray(got)[:, : tree.ni])
    counts = np.asarray(tree.leaf_ranges[:, 1] - tree.leaf_ranges[:, 0] + 1)
    assert not np.any(np.asarray(got)[:, tree.ni + np.flatnonzero(counts <= 0)])


@pytest.mark.parametrize(
    "live",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_dead_tangents_are_skipped_not_wrong(tree, live):
    args = tree.args(jnp.float64)
    tangents = tuple(
        t if on else jnp.zeros_like(t) for t, on in zip(_tangents(args, 2, 0), live)
    )
    (got,) = _op(tree, args, tangents, 4, live=live)
    (want,) = _ref(tree, args, tangents, 4)
    if not any(live):
        assert not np.any(np.asarray(got))
        return
    assert _rel_max(got, want) < 1e-12


def test_unbucketed_k_pads_and_chunks(tree):
    args = tree.args(jnp.float64)
    tangents = _tangents(args, 11, 3)
    (got,) = _op(tree, args, tangents, 3)
    (want,) = _ref(tree, args, tangents, 3)
    assert got.shape[0] == 11
    assert _rel_max(got, want) < 1e-12


_TPP_ONE: dict = {}


def _tpp_grid(tree, tpp, k, *, interpret=True, dtype=jnp.float64):
    order = 3
    args = tree.args(dtype)
    tangents = _tangents(args, k, 20 + k)

    def run(t):
        (out,) = _p2m_tangent_impl(
            *args,
            *tangents,
            **tree.static(order),
            leaf_width=tree.leaf,
            interpret=interpret,
            backend="triton",
            num_warps=None,
            live=(True, True, True),
            tangents_per_program=t,
        )
        return np.asarray(out)

    got = run(tpp)
    key = (k, interpret, dtype)
    if key not in _TPP_ONE:
        _TPP_ONE[key] = run(1)
    (want,) = _ref(tree, args, tangents, order)
    assert got.shape == (k, tree.tot, (order + 1) ** 2)
    return _rel_max(got, want), _rel_max(got, _TPP_ONE[key])


@pytest.mark.parametrize("k", [1, 3, 8, 11])
@pytest.mark.parametrize("tpp", [1, 2, 4, 8])
def test_tangents_per_program_grid(tree, tpp, k):
    """Tangents on the grid: any ``K`` (a partial last group masked in-kernel, no
    padding) at every tangents-per-program equals the twin's jvp, and equals
    ``tpp = 1`` to round-off (G-tpp)."""
    err, err_one = _tpp_grid(tree, tpp, k)
    assert err < 1e-12, err
    assert err_one < 1e-14, err_one


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("k", [3, 8])
@pytest.mark.parametrize("tpp", [1, 2, 4, 8])
def test_native_tangents_per_program_grid(tree, tpp, k, dtype):
    """G-tpp natively (skips without an Ampere card)."""
    _native_or_skip(False)
    err, err_one = _tpp_grid(tree, tpp, k, interpret=False, dtype=dtype)
    assert err < _NATIVE_TOL[dtype], err
    assert err_one < (1e-14 if dtype == jnp.float64 else 1e-6), err_one


@pytest.mark.parametrize("live", [(True, False, True), (False, True, False)])
def test_dead_placeholders_are_never_read(tree, live):
    """A dead tangent bound as its ``(1, ...)`` placeholder (track G's
    ``dead_placeholders``; NaN here, so a read would show): ``K`` comes from the
    live tangents and the result equals the dense-zero call's reference."""
    order = 3
    args = tree.args(jnp.float64)
    dense = tuple(
        t if on else jnp.zeros_like(t) for t, on in zip(_tangents(args, 3, 5), live)
    )
    sparse = tuple(
        t if on else jnp.full((1,) * t.ndim, jnp.nan) for t, on in zip(dense, live)
    )
    (got,) = _p2m_tangent_impl(
        *args,
        *sparse,
        **tree.static(order),
        leaf_width=tree.leaf,
        interpret=True,
        backend="triton",
        num_warps=None,
        live=live,
        tangents_per_program=2,
    )
    (want,) = _ref(tree, args, dense, order)
    assert got.shape == want.shape
    assert _rel_max(got, want) < 1e-12


def test_twin_tangent_equals_the_jvp_of_the_production_reference(tree):
    """``_p2m_leaves_real`` (the batched XLA P2M) is an independent second anchor."""
    order = 4
    args = tree.args(jnp.float64)
    dx, dm, dc = (t[0] for t in _tangents(args, 1, 7))

    def ref(p, m, c_leaf):
        c = tree.com.at[tree.ni :].set(c_leaf)
        return _p2m_leaves_real(
            tree.topo.node_ranges,
            p,
            m,
            c,
            order=order,
            max_leaf_size=tree.leaf,
            num_internal=tree.ni,
            total_nodes=tree.tot,
            leaf_batch_size=256,
        )

    _, want = jax.jvp(ref, args[:3], (dx, dm, dc))
    (got,) = _ref(tree, args, (dx[None], dm[None], dc[None]), order)
    assert _rel_max(got[0], want) < 1e-12


def _seam_for(tree, order):
    ranges = tree.leaf_ranges
    st = (order, tree.ni, tree.tot, tree.leaf, True, "triton", None)

    @jax.custom_jvp
    def seam(p, m, c):
        return p2m_real_leaves_pallas_cvjp(p, m, c, ranges, *st)

    @seam.defjvp
    def _seam_jvp(primals, tangents):
        out = seam(*primals)
        tk, live = with_tangent_axis(tangents, primals)
        (t,) = drop_tangent_axis(
            p2m_real_leaves_pallas_tangent(
                *primals,
                ranges,
                *tk,
                order=order,
                num_internal=tree.ni,
                total_nodes=tree.tot,
                leaf_width=tree.leaf,
                interpret=True,
                backend="triton",
                num_warps=None,
                live=live,
            )
        )
        return out, t

    return seam


@pytest.mark.parametrize("columns", [3, 11])
def test_jacfwd_through_a_forward_seam(tree, columns):
    order = 3
    pos, mass, cent, ranges = tree.args(jnp.float64)
    rng = np.random.default_rng(columns)
    dirs = [
        jnp.asarray(rng.normal(size=a.shape + (columns,))) for a in (pos, mass, cent)
    ]
    seam = _seam_for(tree, order)

    def twin(p, m, c):
        return p2m_real_leaves_jnp(p, m, c, ranges, **tree.static(order))

    def f(theta, fn):
        return fn(
            pos + dirs[0] @ theta,
            mass + 0.01 * (dirs[1] @ theta),
            cent + dirs[2] @ theta,
        )

    theta = jnp.zeros(columns)
    jac = jax.jacfwd(lambda th: f(th, seam))(theta)
    jac_ref = jax.jacfwd(lambda th: f(th, twin))(theta)
    assert jac.shape == (tree.tot, (order + 1) ** 2, columns)
    assert _rel_max(jac, jac_ref) < 1e-12
    y, _ = jax.jvp(seam, (pos, mass, cent), (pos, mass, cent))
    plain = p2m_real_leaves_pallas(
        pos, mass, cent, ranges, **tree.static(order), leaf_width=16, interpret=True
    )
    assert np.array_equal(np.asarray(y), np.asarray(plain))


def test_finite_differences_anchor_the_twin(tree):
    order = 4
    args = tree.args(jnp.float64)
    rng = np.random.default_rng(9)
    d = tuple(jnp.asarray(rng.normal(size=a.shape)) for a in args[:3])

    def f(h):
        p, m, c = (a + h * da for a, da in zip(args[:3], d))
        return p2m_real_leaves_jnp(p, m, c, args[3], **tree.static(order))

    (t,) = _ref(tree, args, tuple(x[None] for x in d), order)
    h = 1e-5
    fd = (f(h) - f(-h)) / (2 * h)
    assert _rel_max(t[0], fd) < 1e-7


def test_dead_rows_past_the_leaves_carry_nothing():
    """Particles outside every leaf range (a shard's padding) contribute nothing."""
    rng = np.random.default_rng(3)
    n, order = 90, 3
    # five leaves over rows 0..69 (one of 25 > chunk), two empty ones, 20 dead rows
    bounds = [0, 1, 26, 40, 41, 70]
    ranges = [(bounds[i], bounds[i + 1] - 1) for i in range(5)] + [(70, 69), (70, 69)]
    ranges = jnp.asarray(ranges, jnp.int32)
    L, ni = 7, 6
    args = (
        jnp.asarray(rng.normal(size=(n, 3))),
        jnp.asarray(rng.uniform(0.5, 1.5, n)),
        jnp.asarray(rng.normal(size=(L, 3)) * 0.1),
        ranges,
    )
    static = dict(order=order, num_internal=ni, total_nodes=ni + L)
    tangents = _tangents(args, 4, 5)
    (got,) = p2m_real_leaves_pallas_tangent(
        *args,
        *tangents,
        **static,
        leaf_width=32,
        interpret=True,
        backend="triton",
        num_warps=None,
        live=(True, True, True),
    )
    (want,) = p2m_real_leaves_pallas_tangent_jnp(*args, *tangents, **static)
    assert _rel_max(got, want) < 1e-12
    # moving the dead rows changes nothing
    dead = tuple(t.at[:, 70:].set(0.0) if t.shape[1] == n else t for t in tangents)
    (got_dead,) = p2m_real_leaves_pallas_tangent(
        *args,
        *dead,
        **static,
        leaf_width=32,
        interpret=True,
        backend="triton",
        num_warps=None,
        live=(True, True, True),
    )
    np.testing.assert_allclose(got_dead, got, rtol=1e-14, atol=1e-14)


def _native_or_skip(interpret: bool) -> None:
    """Native Triton lowering needs an Ampere+ GPU; interpret mode runs anywhere."""
    if not interpret and not pallas_p2m_real_leaf_supported():
        pytest.skip("native Pallas GPU lowering not available here")


#: Native tolerances: the CPU-interpret measurements (fp64 2.2e-16, fp32 2.0e-7)
#: with headroom for the GPU's fused multiply-adds and reduction order.
_NATIVE_TOL = {jnp.float64: 1e-12, jnp.float32: 2e-6}


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("k", [1, 8])
def test_native_gpu_parity(tree, k, dtype):
    """The Triton tangent kernel against the reference, natively, at p5 (skips
    without an Ampere card)."""
    _native_or_skip(False)
    order = 5
    args = tree.args(dtype)
    tangents = _tangents(args, k, order)
    (got,) = p2m_real_leaves_pallas_tangent(
        *args,
        *tangents,
        **tree.static(order),
        leaf_width=tree.leaf,
        interpret=False,
        backend="triton",
        num_warps=None,
        live=(True, True, True),
    )
    (want,) = _ref(tree, args, tangents, order)
    assert np.all(np.isfinite(np.asarray(got)))
    assert _rel_max(got, want) < _NATIVE_TOL[dtype], _rel_max(got, want)
