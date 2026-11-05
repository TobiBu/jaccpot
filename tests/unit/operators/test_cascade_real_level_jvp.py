"""Forward mode of the lane M2M / L2L cascades: the tangent ops vs ``jax.jvp`` of the level loops.

``m2m_real_levels_pallas_tangent`` and ``l2l_real_levels_pallas_tangent``
(:mod:`jaccpot.pallas.cascade_real_level`) are the tangent ops of the forward-mode
seams of ``m2m_real_levels_pallas_cvjp`` / ``l2l_real_levels_pallas_cvjp``. In
Pallas interpret mode on CPU they are checked against their references
``*_pallas_tangent_jnp`` (``vmap`` over ``K`` of ``jax.jvp`` of the XLA level loops
``m2m_real_levels_jnp`` / ``l2l_real_levels_jnp``), in both halves -- the
coefficient AND the centre tangent, each asserted separately and non-vacuous -- on
a Plummer radix tree (ragged levels narrower and wider than a program's 32 lanes,
so masked lanes everywhere) and on a tree with on-axis edges (child centre exactly
above its parent, ``rho == 0``), where the kernels' ladder identity must agree with
the operators' analytic transverse derivative.
"""

from __future__ import annotations

import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax.experimental.pallas")
pytest.importorskip("yggdrax")

from jaccpot.pallas._tangent import drop_tangent_axis, with_tangent_axis  # noqa: E402
from jaccpot.pallas.cascade_real_level import (  # noqa: E402
    _l2l_tangent_impl,
    _m2m_tangent_impl,
    l2l_real_levels_jnp,
    l2l_real_levels_pallas_cvjp,
    l2l_real_levels_pallas_tangent,
    l2l_real_levels_pallas_tangent_jnp,
    l2l_real_levels_pallas_tangent_out,
    m2m_real_levels_jnp,
    m2m_real_levels_pallas_cvjp,
    m2m_real_levels_pallas_tangent,
    m2m_real_levels_pallas_tangent_jnp,
    m2m_real_levels_pallas_tangent_out,
    pallas_cascade_level_supported,
)

_TOL = {jnp.float64: 1e-11, jnp.float32: 5e-5}


#: Nodes per program of the lane kernels in these tests (production: 32). The
#: Plummer tree's levels hold 1..18 nodes, so 8 makes them both narrower and wider
#: than a program: partly masked programs AND levels spanning several programs.
_LANES_K = 8


@pytest.fixture(autouse=True)
def _x64(monkeypatch):
    """x64 and the test lane width; after each test, drop its compiled programs.

    Kept across tests, the K = 8 cascade programs (interpret mode) took this file to
    16.2 GB RSS in one process -- more than the 16 GB CI runner, which shut the
    `unit` shard down on #500/#502 (2026-11-04).
    """
    monkeypatch.setenv("JACCPOT_CASCADE_LANES_K", str(_LANES_K))
    with jax.enable_x64(True):
        yield
    jax.clear_caches()
    gc.collect()


def _plummer(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


class _Tree:
    """A radix tree's cascade operands (the seams') and statics."""

    def __init__(self, positions, masses, leaf):
        from yggdrax.tree import Tree
        from yggdrax.tree_moments import compute_tree_mass_moments

        tree = Tree.from_particles(
            jnp.asarray(positions),
            jnp.asarray(masses),
            tree_type="radix",
            build_mode="static_radix",
            leaf_size=leaf,
        )
        topo = tree.topology
        self.topo = topo
        self.ni = int(topo.left_child.shape[0])
        self.tot = int(topo.parent.shape[0])
        self.num_levels = int(jnp.max(topo.node_level)) + 1
        offs = np.asarray(topo.level_offsets)
        self.widths = offs[1:] - offs[:-1]
        self.width = int(self.widths.max())
        self.com = jnp.asarray(
            compute_tree_mass_moments(
                topo, tree.positions_sorted, tree.masses_sorted
            ).center_of_mass,
            jnp.float64,
        )

    def m2m_topo(self):
        t = self.topo
        return (
            t.left_child,
            t.right_child,
            t.parent,
            t.nodes_by_level,
            t.level_offsets,
        )

    def l2l_topo(self):
        t = self.topo
        return (
            t.parent,
            t.left_child,
            t.right_child,
            t.nodes_by_level,
            t.level_offsets,
        )

    def m2m_static(self, order):
        return dict(
            order=order,
            num_internal=self.ni,
            num_levels=self.num_levels,
            level_batch_width=self.width,
        )

    def l2l_static(self, order):
        return dict(
            order=order, num_levels=self.num_levels, level_batch_width=self.width
        )

    def on_axis_edges(self):
        parent, centres = np.asarray(self.topo.parent), np.asarray(self.com)
        has = parent >= 0
        d = centres[has] - centres[parent[has]]
        rho2 = d[:, 0] ** 2 + d[:, 1] ** 2
        return int(np.sum((rho2 == 0) & (d[:, 2] != 0)))


@pytest.fixture(scope="module")
def plummer_tree():
    with jax.enable_x64(True):
        rng = np.random.default_rng(2)
        tree = _Tree(_plummer(400, 1), rng.uniform(0.5, 1.5, 400), 16)
    # ragged levels: narrower and wider than one program's lanes
    live = tree.widths[tree.widths > 0]
    assert live.min() < _LANES_K < live.max()
    return tree


@pytest.fixture(scope="module")
def axis_tree():
    """Four clusters stacked along z sharing one ``(x, y)`` set (the vjp test's)."""
    rng = np.random.default_rng(3)
    xy = rng.normal(size=(8, 2)) * 0.35
    masses = np.tile(rng.uniform(0.6, 1.4, size=8), 4)
    positions = np.concatenate(
        [np.column_stack([xy, np.full(8, (i - 1.5) * 6.0)]) for i in range(4)]
    )
    with jax.enable_x64(True):
        tree = _Tree(positions, masses, 4)
    assert tree.on_axis_edges() >= 5
    return tree


def _inputs(tree, which, order, dtype, k, seed):
    rng = np.random.default_rng(seed + order)
    C = (order + 1) ** 2
    coef = jnp.asarray(rng.standard_normal((tree.tot, C)), dtype)
    if which == "m2m":
        coef = coef.at[: tree.ni].set(0.0)
    cent = tree.com.astype(dtype)
    t_coef = jnp.asarray(rng.standard_normal((k, tree.tot, C)), dtype)
    t_cent = jnp.asarray(rng.standard_normal((k, tree.tot, 3)), dtype)
    return coef, cent, t_coef, t_cent


_OPS = {
    "m2m": (m2m_real_levels_pallas_tangent, m2m_real_levels_pallas_tangent_jnp),
    "l2l": (l2l_real_levels_pallas_tangent, l2l_real_levels_pallas_tangent_jnp),
}


def _call(
    tree,
    which,
    order,
    coef,
    cent,
    t_coef,
    t_cent,
    live=(True, True),
    ref=False,
    interpret=True,
):
    op = _OPS[which][1 if ref else 0]
    topo = tree.m2m_topo() if which == "m2m" else tree.l2l_topo()
    static = tree.m2m_static(order) if which == "m2m" else tree.l2l_static(order)
    (out,) = op(
        coef,
        cent,
        *topo,
        t_coef,
        t_cent,
        **static,
        interpret=interpret,
        backend="triton",
        num_warps=4,
        live=live,
    )
    return out


def _forward(tree, which, order, coef, cent):
    if which == "m2m":
        s = tree.m2m_static(order)
        return m2m_real_levels_pallas_cvjp(
            coef,
            cent,
            *tree.m2m_topo(),
            order,
            s["num_internal"],
            s["num_levels"],
            s["level_batch_width"],
            True,
            "triton",
            4,
        )
    s = tree.l2l_static(order)
    return l2l_real_levels_pallas_cvjp(
        coef,
        cent,
        *tree.l2l_topo(),
        order,
        s["num_levels"],
        s["level_batch_width"],
        True,
        "triton",
        4,
    )


def _twin(tree, which, order, coef, cent):
    if which == "m2m":
        return m2m_real_levels_jnp(
            coef, cent, *tree.m2m_topo(), **tree.m2m_static(order)
        )
    return l2l_real_levels_jnp(coef, cent, *tree.l2l_topo(), **tree.l2l_static(order))


def _rel_max(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-300))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("which", ["m2m", "l2l"])
def test_jnp_twin_primal_is_the_pallas_forward(plummer_tree, which, dtype):
    order = 4
    coef, cent, _, _ = _inputs(plummer_tree, which, order, dtype, 1, 0)
    want = _forward(plummer_tree, which, order, coef, cent)
    got = _twin(plummer_tree, which, order, coef, cent)
    assert jnp.asarray(got).dtype == jnp.asarray(want).dtype
    assert _rel_max(got, want) < _TOL[dtype]


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("which", ["m2m", "l2l"])
def test_tangent_op_equals_the_jvp_of_the_level_loop(
    plummer_tree, which, order, k, dtype
):
    """Both halves separately. Measured max rel error (2026-11-03, CPU interpret, p2/p4,
    K=1/8): fp64 <= 1.2e-14, fp32 <= 4.7e-6 (either half, either cascade)."""
    tree = plummer_tree
    coef, cent, t_coef, t_cent = _inputs(tree, which, order, dtype, k, k)
    for half, (tc, tx) in (
        ("coefficients", (t_coef, jnp.zeros_like(t_cent))),
        ("centres", (jnp.zeros_like(t_coef), t_cent)),
        ("both", (t_coef, t_cent)),
    ):
        got = _call(tree, which, order, coef, cent, tc, tx)
        want = _call(tree, which, order, coef, cent, tc, tx, ref=True)
        assert got.shape == (k, tree.tot, (order + 1) ** 2) and got.dtype == dtype
        assert np.linalg.norm(np.asarray(want)) > 0, half
        err = _rel_max(got, want)
        assert err < _TOL[dtype], f"{which} {half}: {err:.3e}"


@pytest.mark.parametrize("which", ["m2m", "l2l"])
def test_on_axis_edges_take_the_analytic_limit(axis_tree, which):
    """``rho == 0`` edges: the ladder identity equals the operators' analytic rule."""
    for order in (2, 4):
        coef, cent, t_coef, t_cent = _inputs(axis_tree, which, order, jnp.float64, 2, 5)
        got = _call(axis_tree, which, order, coef, cent, jnp.zeros_like(t_coef), t_cent)
        want = _call(
            axis_tree,
            which,
            order,
            coef,
            cent,
            jnp.zeros_like(t_coef),
            t_cent,
            ref=True,
        )
        assert np.all(np.isfinite(np.asarray(got)))
        assert np.linalg.norm(np.asarray(want)) > 0
        assert _rel_max(got, want) < 1e-10, f"p{order}: {_rel_max(got, want):.3e}"


@pytest.mark.parametrize("which", ["m2m", "l2l"])
def test_dead_centre_tangent_is_skipped_not_wrong(plummer_tree, which):
    order = 2
    coef, cent, t_coef, t_cent = _inputs(plummer_tree, which, order, jnp.float64, 2, 1)
    zero = jnp.zeros_like(t_cent)
    got = _call(
        plummer_tree, which, order, coef, cent, t_coef, zero, live=(True, False)
    )
    want = _call(plummer_tree, which, order, coef, cent, t_coef, zero, ref=True)
    assert _rel_max(got, want) < 1e-12
    # a dead coefficient tangent still receives the centre terms of other levels
    zc = jnp.zeros_like(t_coef)
    got = _call(plummer_tree, which, order, coef, cent, zc, t_cent, live=(False, True))
    want = _call(plummer_tree, which, order, coef, cent, zc, t_cent, ref=True)
    assert _rel_max(got, want) < 1e-11
    none = _call(plummer_tree, which, order, coef, cent, zc, zero, live=(False, False))
    assert not np.any(np.asarray(none))


@pytest.mark.parametrize("which", ["m2m", "l2l"])
def test_unbucketed_k_pads_and_chunks(plummer_tree, which):
    order = 2
    coef, cent, t_coef, t_cent = _inputs(plummer_tree, which, order, jnp.float64, 11, 2)
    got = _call(plummer_tree, which, order, coef, cent, t_coef, t_cent)
    want = _call(plummer_tree, which, order, coef, cent, t_coef, t_cent, ref=True)
    assert got.shape[0] == 11
    assert _rel_max(got, want) < 1e-11


_TPP_ONE: dict = {}


@pytest.mark.parametrize("k", [1, 3, 8, 11])
@pytest.mark.parametrize("tpp", [1, 2, 4, 8])
@pytest.mark.parametrize("which", ["m2m", "l2l"])
def test_tangents_per_program_grid(plummer_tree, which, tpp, k):
    """Tangents on the grid: any ``K`` (a partial last group masked in-kernel, no
    padding) at every tangents-per-program equals the level loop's jvp, and equals
    ``tpp = 1`` to round-off (G-tpp)."""
    err, err_one = _tpp_grid(plummer_tree, which, tpp, k)
    assert err < 1e-11, err
    assert err_one < 1e-14, err_one


@pytest.mark.parametrize("k", [3, 8])
@pytest.mark.parametrize("tpp", [1, 2, 4, 8])
@pytest.mark.parametrize("which", ["m2m", "l2l"])
def test_native_tangents_per_program_grid(plummer_tree, which, tpp, k):
    """G-tpp natively, float64 (skips without an Ampere card)."""
    _native_or_skip(False)
    err, err_one = _tpp_grid(plummer_tree, which, tpp, k, interpret=False)
    assert err < 1e-11, err
    assert err_one < 1e-14, err_one


def _tpp_grid(tree, which, tpp, k, *, interpret=True):
    order = 2
    coef, cent, t_coef, t_cent = _inputs(tree, which, order, jnp.float64, k, 40 + k)
    topo = tree.m2m_topo() if which == "m2m" else tree.l2l_topo()
    static = tree.m2m_static(order) if which == "m2m" else tree.l2l_static(order)
    impl = _m2m_tangent_impl if which == "m2m" else _l2l_tangent_impl

    def run(t):
        (out,) = impl(
            coef,
            cent,
            *topo,
            t_coef,
            t_cent,
            **static,
            interpret=interpret,
            backend="triton",
            num_warps=4,
            live=(True, True),
            tangents_per_program=t,
        )
        return np.asarray(out)

    got = run(tpp)
    key = (which, k, interpret)
    if key not in _TPP_ONE:
        _TPP_ONE[key] = run(1)
    want = _call(tree, which, order, coef, cent, t_coef, t_cent, ref=True)
    assert got.shape == (k, tree.tot, (order + 1) ** 2)
    return _rel_max(got, want), _rel_max(got, _TPP_ONE[key])


@pytest.mark.parametrize("which", ["m2m", "l2l"])
def test_the_output_reading_op_equals_the_reference(plummer_tree, which):
    """``*_pallas_tangent_out`` reads the seam's OUTPUT as an eighth primal and
    neither recomputes nor stores the primal cascade: both halves, ``K = 3``."""
    order = 3
    tree = plummer_tree
    coef, cent, t_coef, t_cent = _inputs(tree, which, order, jnp.float64, 3, 9)
    out = _forward(tree, which, order, coef, cent)
    op = (
        m2m_real_levels_pallas_tangent_out
        if which == "m2m"
        else l2l_real_levels_pallas_tangent_out
    )
    topo = tree.m2m_topo() if which == "m2m" else tree.l2l_topo()
    static = tree.m2m_static(order) if which == "m2m" else tree.l2l_static(order)
    for tc, tx in (
        (t_coef, jnp.zeros_like(t_cent)),
        (jnp.zeros_like(t_coef), t_cent),
    ):
        (got,) = op(
            coef,
            cent,
            *topo,
            out,
            tc,
            tx,
            **static,
            interpret=True,
            backend="triton",
            num_warps=4,
            live=(True, True),
        )
        want = _call(tree, which, order, coef, cent, tc, tx, ref=True)
        assert np.linalg.norm(np.asarray(want)) > 0
        assert _rel_max(got, want) < 1e-11, _rel_max(got, want)


@pytest.mark.parametrize("live", [(True, False), (False, True)])
@pytest.mark.parametrize("which", ["m2m", "l2l"])
def test_dead_placeholders_are_never_read(plummer_tree, which, live):
    """A dead tangent bound as its ``(1, ...)`` placeholder (NaN, so a read would
    show): ``K`` comes from the live one; a dead coefficient tangent becomes the
    zero table the levels' centre terms are written into."""
    order = 2
    tree = plummer_tree
    coef, cent, t_coef, t_cent = _inputs(tree, which, order, jnp.float64, 3, 11)
    dense = (
        t_coef if live[0] else jnp.zeros_like(t_coef),
        t_cent if live[1] else jnp.zeros_like(t_cent),
    )
    sparse = tuple(
        t if on else jnp.full((1,) * t.ndim, jnp.nan) for t, on in zip(dense, live)
    )
    impl = _m2m_tangent_impl if which == "m2m" else _l2l_tangent_impl
    topo = tree.m2m_topo() if which == "m2m" else tree.l2l_topo()
    static = tree.m2m_static(order) if which == "m2m" else tree.l2l_static(order)
    (got,) = impl(
        coef,
        cent,
        *topo,
        *sparse,
        **static,
        interpret=True,
        backend="triton",
        num_warps=4,
        live=live,
        tangents_per_program=2,
    )
    want = _call(tree, which, order, coef, cent, *dense, ref=True)
    assert got.shape == want.shape
    assert np.linalg.norm(np.asarray(want)) > 0
    assert _rel_max(got, want) < 1e-11, _rel_max(got, want)


def _seam_for(tree, which, order):
    op = _OPS[which][0]
    topo = tree.m2m_topo() if which == "m2m" else tree.l2l_topo()
    static = tree.m2m_static(order) if which == "m2m" else tree.l2l_static(order)

    @jax.custom_jvp
    def seam(x, c):
        return _forward(tree, which, order, x, c)

    @seam.defjvp
    def _seam_jvp(primals, tangents):
        out = seam(*primals)
        tk, live = with_tangent_axis(tangents, primals)
        (t,) = drop_tangent_axis(
            op(
                *primals,
                *topo,
                *tk,
                **static,
                interpret=True,
                backend="triton",
                num_warps=4,
                live=live,
            )
        )
        return out, t

    return seam


@pytest.mark.parametrize("columns", [3, 11])
@pytest.mark.parametrize("which", ["m2m", "l2l"])
def test_jacfwd_through_a_forward_seam(plummer_tree, which, columns):
    order = 2
    tree = plummer_tree
    coef, cent, _, _ = _inputs(tree, which, order, jnp.float64, 1, 3)
    rng = np.random.default_rng(columns)
    a = jnp.asarray(rng.normal(size=coef.shape + (columns,)))
    b = jnp.asarray(rng.normal(size=cent.shape + (columns,))) * 0.01
    if which == "m2m":
        a = a.at[: tree.ni].set(0.0)
    seam = _seam_for(tree, which, order)

    def f(theta, fn):
        return fn(coef + a @ theta, cent + b @ theta)

    theta = jnp.zeros(columns)
    jac = jax.jacfwd(lambda th: f(th, seam))(theta)
    jac_ref = jax.jacfwd(
        lambda th: f(th, lambda x, c: _twin(tree, which, order, x, c))
    )(theta)
    assert jac.shape == coef.shape + (columns,)
    assert _rel_max(jac, jac_ref) < 1e-11
    y, _ = jax.jvp(seam, (coef, cent), (coef, cent))
    assert np.array_equal(
        np.asarray(y), np.asarray(_forward(tree, which, order, coef, cent))
    )


@pytest.mark.parametrize("which", ["m2m", "l2l"])
def test_finite_differences_anchor_the_twin(plummer_tree, which):
    order = 3
    coef, cent, t_coef, t_cent = _inputs(plummer_tree, which, order, jnp.float64, 1, 9)
    dc, dx = t_coef[0], t_cent[0] * 0.01

    def f(h):
        return _twin(plummer_tree, which, order, coef + h * dc, cent + h * dx)

    want = _call(plummer_tree, which, order, coef, cent, dc[None], dx[None], ref=True)
    h = 1e-5
    fd = (f(h) - f(-h)) / (2 * h)
    assert _rel_max(want[0], fd) < 1e-7


@pytest.mark.parametrize("which", ["m2m", "l2l"])
def test_production_lane_count(plummer_tree, which, monkeypatch):
    """The production 32 nodes per program: every level in one masked program."""
    monkeypatch.setenv("JACCPOT_CASCADE_LANES_K", "32")
    order = 2
    coef, cent, t_coef, t_cent = _inputs(plummer_tree, which, order, jnp.float64, 4, 6)
    got = _call(plummer_tree, which, order, coef, cent, t_coef, t_cent)
    want = _call(plummer_tree, which, order, coef, cent, t_coef, t_cent, ref=True)
    assert _rel_max(got, want) < 1e-11


def _native_or_skip(interpret: bool) -> None:
    """Native Triton lowering needs an Ampere+ GPU; interpret mode runs anywhere."""
    if not interpret and not pallas_cascade_level_supported():
        pytest.skip("native Pallas GPU lowering not available here")


#: Native tolerances: the CPU-interpret measurements (fp64 1.2e-14, fp32 4.7e-6 on
#: the Plummer tree; the on-axis tree asserted < 1e-10 in fp64) with headroom for
#: the GPU's fused multiply-adds.
_NATIVE_TOL = {jnp.float64: 1e-11, jnp.float32: 5e-5}


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("which", ["m2m", "l2l"])
@pytest.mark.parametrize("tree_name", ["plummer", "on_axis"])
def test_native_gpu_parity(request, tree_name, which, k, dtype, monkeypatch):
    """The Triton tangent kernels at the production 32 lanes against the reference,
    natively, in both halves at once, on the Plummer and the on-axis trees (skips
    without an Ampere card)."""
    _native_or_skip(False)
    monkeypatch.setenv("JACCPOT_CASCADE_LANES_K", "32")
    tree = request.getfixturevalue(
        "plummer_tree" if tree_name == "plummer" else "axis_tree"
    )
    order = 4
    coef, cent, t_coef, t_cent = _inputs(tree, which, order, dtype, k, k)
    got = _call(tree, which, order, coef, cent, t_coef, t_cent, interpret=False)
    want = _call(tree, which, order, coef, cent, t_coef, t_cent, ref=True)
    assert np.all(np.isfinite(np.asarray(got)))
    assert np.linalg.norm(np.asarray(want)) > 0
    tol = (
        _NATIVE_TOL[dtype] if tree_name == "plummer" else max(_NATIVE_TOL[dtype], 1e-10)
    )
    assert _rel_max(got, want) < tol, _rel_max(got, want)
