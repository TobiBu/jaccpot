"""Forward mode of the particle-major Pallas L2P: the tangent op vs ``jax.jvp`` of its twin.

``l2p_real_particles_pallas_tangent`` (:mod:`jaccpot.pallas.l2p_real`) is the
tangent op the forward-mode seam of ``l2p_real_particles_pallas_cvjp`` calls. It is
checked, in Pallas interpret mode on CPU, against
``l2p_real_particles_pallas_tangent_jnp`` (``vmap`` over ``K`` of ``jax.jvp`` of the
pure-``jnp`` twin ``l2p_real_particles_jnp``), whose primal is checked against the
Pallas forward, and whose tangent is anchored by finite differences. The trees
carry leaves of every occupancy, empty padding leaves, dead rows past the leaves'
particles (masked lanes) and a real yggdrax cell tree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax.experimental.pallas")

from jaccpot.pallas._tangent import drop_tangent_axis, with_tangent_axis  # noqa: E402
from jaccpot.pallas.l2p_real import (  # noqa: E402
    _l2p_tangent_impl,
    l2p_real_particles_jnp,
    l2p_real_particles_pallas,
    l2p_real_particles_pallas_cvjp,
    l2p_real_particles_pallas_tangent,
    l2p_real_particles_pallas_tangent_jnp,
)
from jaccpot.pallas.m2l_real_csr import pallas_m2l_real_csr_supported  # noqa: E402
from jaccpot.runtime.dtypes import INDEX_DTYPE  # noqa: E402

_ORDER = 5
_TOL = {jnp.float64: 1e-12, jnp.float32: 2e-5}


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


def _synthetic(seed, n_live, n_rows, num_pad, width, order):
    """Leaves of 1..``width`` particles in order, ``num_pad`` empty ones at the end."""
    rng = np.random.default_rng(seed)
    counts = []
    left = n_live
    while left > 0:
        c = int(min(left, rng.integers(1, width + 1)))
        counts.append(c)
        left -= c
    num_leaves = len(counts) + num_pad
    num_internal = num_leaves - 1
    num_nodes = num_internal + num_leaves
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    starts = np.concatenate([starts, np.full(num_pad, n_live)])
    counts = np.concatenate([counts, np.zeros(num_pad, np.int64)])
    return (
        jnp.asarray(rng.normal(size=(num_nodes, (order + 1) ** 2))),
        jnp.asarray(rng.normal(size=(num_nodes, 3)) * 0.3),
        jnp.asarray(rng.normal(size=(n_rows, 3)) * 0.3),
        jnp.arange(num_internal, num_nodes, dtype=INDEX_DTYPE),
        jnp.asarray(starts, INDEX_DTYPE),
        jnp.asarray(counts, INDEX_DTYPE),
    )


def _cell_tree(order):
    """A real yggdrax cell tree: padding and single-particle leaves."""
    pytest.importorskip("yggdrax")
    from yggdrax._tree_impl import (
        build_static_cells_tree,  # pyright: ignore[reportAttributeAccessIssue]
    )
    from yggdrax.bounds import infer_bounds
    from yggdrax.tree_moments import compute_tree_mass_moments

    rng = np.random.default_rng(5)
    pos = jnp.asarray(rng.normal(size=(300, 3)))
    mass = jnp.asarray(rng.uniform(0.5, 1.5, 300))
    topo, ps, ms, _ = build_static_cells_tree(
        pos,
        mass,
        infer_bounds(pos),
        leaf_size=16,
        leaf_capacity=512,
        return_reordered=True,
    )
    ni = int(topo.left_child.shape[0])
    tot = int(topo.parent.shape[0])
    ranges = np.asarray(topo.node_ranges)[ni:]
    counts = np.maximum(ranges[:, 1] - ranges[:, 0] + 1, 0)
    assert (counts == 0).any() and (counts == 1).any()
    com = jnp.asarray(compute_tree_mass_moments(topo, ps, ms).center_of_mass)
    return (
        jnp.asarray(rng.normal(size=(tot, (order + 1) ** 2))),
        com,
        ps,
        jnp.arange(ni, tot, dtype=INDEX_DTYPE),
        jnp.asarray(ranges[:, 0], INDEX_DTYPE),
        jnp.asarray(counts, INDEX_DTYPE),
    )


# (n_live, n_rows, padding leaves, widest leaf): dead rows, leaves of up to 70 that
# span blocks, single-particle-heavy leaves
_CASES = [(187, 200, 2, 8), (300, 300, 1, 70), (150, 160, 0, 3)]


def _case(i, dtype, order=_ORDER):
    if i == len(_CASES):
        args = _cell_tree(order)
    else:
        n_live, n_rows, pad, width = _CASES[i]
        args = _synthetic(i, n_live, n_rows, pad, width, order)
    return tuple(a.astype(dtype) for a in args[:3]) + tuple(args[3:])


_ALL = list(range(len(_CASES) + 1))


def _tangents(args, k, seed):
    rng = np.random.default_rng(100 + seed)
    return tuple(
        jnp.asarray(rng.normal(size=(k,) + a.shape), a.dtype) for a in args[:3]
    )


def _op(args, tangents, live=(True, True, True), block=32):
    return l2p_real_particles_pallas_tangent(
        *args,
        *tangents,
        order=_ORDER,
        block=block,
        interpret=True,
        backend="triton",
        num_warps=4,
        live=live,
    )


def _ref(args, tangents):
    return l2p_real_particles_pallas_tangent_jnp(*args, *tangents, order=_ORDER)


def _rel_max(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-300))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("case", _ALL)
def test_jnp_twin_primal_is_the_pallas_forward(case, dtype):
    args = _case(case, dtype)
    want = l2p_real_particles_pallas(*args, order=_ORDER, block=32, interpret=True)
    got = l2p_real_particles_jnp(*args, order=_ORDER)
    assert got.dtype == want.dtype
    assert _rel_max(got, want) < _TOL[dtype]
    # rows past the leaves' particles are zero in both
    n_live = int(jnp.sum(args[5]))
    assert not np.any(np.asarray(got)[n_live:])


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("case", _ALL)
def test_tangent_op_equals_the_jvp_of_the_twin(case, k, dtype):
    """Measured max rel error (2026-11-03, CPU interpret, p5, 4 trees): fp64 3.7e-16, fp32 1.2e-7."""
    args = _case(case, dtype)
    tangents = _tangents(args, k, case)
    (got,) = _op(args, tangents)
    (want,) = _ref(args, tangents)
    assert got.shape == (k,) + args[2].shape and got.dtype == args[2].dtype
    assert np.linalg.norm(np.asarray(want)) > 0
    assert _rel_max(got, want) < _TOL[dtype], _rel_max(got, want)
    n_live = int(jnp.sum(args[5]))
    assert not np.any(np.asarray(got)[:, n_live:])


@pytest.mark.parametrize(
    "live",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_dead_tangents_are_skipped_not_wrong(live):
    """A dead tangent is zero: the skipped terms change nothing."""
    args = _case(0, jnp.float64)
    tangents = tuple(
        t if on else jnp.zeros_like(t) for t, on in zip(_tangents(args, 2, 0), live)
    )
    (got,) = _op(args, tangents, live=live)
    (want,) = _ref(args, tangents)
    if not any(live):
        assert not np.any(np.asarray(got))
        return
    assert _rel_max(got, want) < 1e-12


@pytest.mark.parametrize("k", [3, 11])
def test_unbucketed_k_pads_and_chunks(k):
    args = _case(1, jnp.float64)
    tangents = _tangents(args, k, 1)
    (got,) = _op(args, tangents)
    (want,) = _ref(args, tangents)
    assert got.shape[0] == k
    assert _rel_max(got, want) < 1e-12


_TPP_ONE: dict = {}


@pytest.mark.parametrize("k", [1, 3, 8, 11])
@pytest.mark.parametrize("tpp", [1, 2, 4, 8])
def test_tangents_per_program_grid(tpp, k):
    """Tangents on the grid: any ``K`` (a partial last group masked in-kernel, no
    padding) at every tangents-per-program equals the twin's jvp, and equals
    ``tpp = 1`` to round-off (G-tpp)."""
    err, err_one = _tpp_grid(tpp, k)
    assert err < 1e-12, err
    assert err_one < 1e-14, err_one


@pytest.mark.parametrize("k", [3, 8])
@pytest.mark.parametrize("tpp", [1, 2, 4, 8])
def test_native_tangents_per_program_grid(tpp, k):
    """G-tpp natively, float64 (skips without an Ampere card)."""
    _native_or_skip(False)
    err, err_one = _tpp_grid(tpp, k, interpret=False)
    assert err < 1e-12, err
    assert err_one < 1e-14, err_one


def _tpp_grid(tpp, k, *, interpret=True):
    args = _case(0, jnp.float64)
    tangents = _tangents(args, k, 30 + k)

    def run(t):
        (out,) = _l2p_tangent_impl(
            *args,
            *tangents,
            order=_ORDER,
            block=32,
            interpret=interpret,
            backend="triton",
            num_warps=4,
            live=(True, True, True),
            tangents_per_program=t,
        )
        return np.asarray(out)

    got = run(tpp)
    key = (k, interpret)
    if key not in _TPP_ONE:
        _TPP_ONE[key] = run(1)
    (want,) = _ref(args, tangents)
    assert got.shape == (k,) + args[2].shape
    return _rel_max(got, want), _rel_max(got, _TPP_ONE[key])


@pytest.mark.parametrize(
    "live", [(True, False, True), (False, True, False), (True, False, False)]
)
def test_dead_placeholders_are_never_read(live):
    """A dead tangent bound as its ``(1, ...)`` placeholder (NaN, so a read would
    show): ``K`` comes from the live tangents; per-operand liveness of the centre
    and position tangents."""
    args = _case(0, jnp.float64)
    dense = tuple(
        t if on else jnp.zeros_like(t) for t, on in zip(_tangents(args, 3, 6), live)
    )
    sparse = tuple(
        t if on else jnp.full((1,) * t.ndim, jnp.nan) for t, on in zip(dense, live)
    )
    (got,) = _l2p_tangent_impl(
        *args,
        *sparse,
        order=_ORDER,
        block=32,
        interpret=True,
        backend="triton",
        num_warps=4,
        live=live,
        tangents_per_program=2,
    )
    (want,) = _ref(args, dense)
    assert got.shape == want.shape
    assert _rel_max(got, want) < 1e-12


def _seam_for(args):
    leaf_nodes, leaf_start, leaf_count = args[3:]

    @jax.custom_jvp
    def seam(c, x, q):
        return l2p_real_particles_pallas_cvjp(
            c,
            x,
            q,
            leaf_nodes,
            leaf_start,
            leaf_count,
            _ORDER,
            32,
            True,
            "triton",
            4,
        )

    @seam.defjvp
    def _seam_jvp(primals, tangents):
        out = seam(*primals)
        tk, live = with_tangent_axis(tangents, primals)
        (t,) = drop_tangent_axis(
            l2p_real_particles_pallas_tangent(
                *primals,
                leaf_nodes,
                leaf_start,
                leaf_count,
                *tk,
                order=_ORDER,
                block=32,
                interpret=True,
                backend="triton",
                num_warps=4,
                live=live,
            )
        )
        return out, t

    return seam


@pytest.mark.parametrize("columns", [3, 11])
def test_jacfwd_through_a_forward_seam(columns):
    args = _case(len(_CASES), jnp.float64)
    coef, cent, pos = args[:3]
    rng = np.random.default_rng(columns)
    dirs = [
        jnp.asarray(rng.normal(size=a.shape + (columns,))) for a in (coef, cent, pos)
    ]
    seam = _seam_for(args)

    def twin(c, x, q):
        return l2p_real_particles_jnp(c, x, q, *args[3:], order=_ORDER)

    def f(theta, fn):
        return fn(coef + dirs[0] @ theta, cent + dirs[1] @ theta, pos + dirs[2] @ theta)

    theta = jnp.zeros(columns)
    jac = jax.jacfwd(lambda th: f(th, seam))(theta)
    jac_ref = jax.jacfwd(lambda th: f(th, twin))(theta)
    assert jac.shape == pos.shape + (columns,)
    assert _rel_max(jac, jac_ref) < 1e-12
    # the primal through the seam is the production forward, bitwise
    y, _ = jax.jvp(seam, (coef, cent, pos), (coef, cent, pos))
    plain = l2p_real_particles_pallas(*args, order=_ORDER, block=32, interpret=True)
    assert np.array_equal(np.asarray(y), np.asarray(plain))


def test_finite_differences_anchor_the_twin():
    args = _case(0, jnp.float64)
    rng = np.random.default_rng(9)
    d = tuple(jnp.asarray(rng.normal(size=a.shape)) for a in args[:3])

    def f(h):
        return l2p_real_particles_jnp(
            *(a + h * da for a, da in zip(args[:3], d)), *args[3:], order=_ORDER
        )

    (t,) = _ref(args, tuple(x[None] for x in d))
    h = 1e-5
    fd = (f(h) - f(-h)) / (2 * h)
    assert _rel_max(t[0], fd) < 1e-7


def _native_or_skip(interpret: bool) -> None:
    """Native Triton lowering needs an Ampere+ GPU; interpret mode runs anywhere."""
    if not interpret and not pallas_m2l_real_csr_supported():
        pytest.skip("native Pallas GPU lowering not available here")


#: Native tolerances: the CPU-interpret measurements (fp64 3.7e-16, fp32 1.2e-7)
#: with headroom for the GPU's fused multiply-adds and reduction order.
_NATIVE_TOL = {jnp.float64: 1e-12, jnp.float32: 2e-6}


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("k", [1, 8])
def test_native_gpu_parity(k, dtype):
    """The Triton tangent kernel (production block 128) against the reference,
    natively, on the cell tree (skips without an Ampere card)."""
    _native_or_skip(False)
    args = _case(len(_CASES), dtype)
    tangents = _tangents(args, k, 0)
    (got,) = l2p_real_particles_pallas_tangent(
        *args,
        *tangents,
        order=_ORDER,
        block=128,
        interpret=False,
        backend="triton",
        num_warps=4,
        live=(True, True, True),
    )
    (want,) = _ref(args, tangents)
    assert np.all(np.isfinite(np.asarray(got)))
    assert _rel_max(got, want) < _NATIVE_TOL[dtype], _rel_max(got, want)
