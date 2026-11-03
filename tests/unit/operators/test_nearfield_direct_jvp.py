"""Forward-mode tangent of the direct CSR near field (interpret mode, CPU).

:data:`~jaccpot.pallas.nearfield_leafpair_csr_direct_vjp.nearfield_leafpair_csr_sorted_direct_pallas_tangent`
is the tangent op of the near-field seam (plan forward-mode, track C): the forward's
target-leaf programs and merged source runs, its row pieces past ``row_limit``, the
analytic pair tangent (relative displacement per pair, the kernels' own ``(1/r)
dg/dr`` and ``dg/d(eps^2)``), ``K`` tangents per launch. Pinned here, for every
softening kernel, in float64 and float32:

* ``K = 1`` and ``K = 8`` tangents in the positions (sink and source), the masses,
  ``softening_sq`` and ``G`` equal the pure-JAX reference
  (:func:`~jaccpot.pallas.nearfield_leafpair_csr_direct_vjp.nearfield_leafpair_csr_sorted_direct_pallas_tangent_jnp`)
  to round-off -- leaves of every occupancy (a single particle, a full leaf), empty
  padding leaves, rows past every leaf's particles, an empty row, merged source runs,
  rows of three or more pieces (the rest-pieces launch), the self terms; each tangent
  alone (``live``) too;
* the reference's primal is the production forward to round-off, and ``jax.jvp`` of
  the independent pair sum of ``test_nearfield_direct_vjp.py`` agrees with it;
* ``jax.jacfwd`` through a ``custom_jvp`` seam (primal: the production forward;
  tangent: the op) over 3 and 11 columns equals ``jax.jacfwd`` of the reference;
* central differences of the PRODUCTION forward anchor the reference;
* coincident particles (softened, and unsoftened) give finite tangents, never NaN;
* the opt-in wide accumulator is within float32 round-off of the float64 truth.

The neighbour CSR is the symmetric one of ``test_nearfield_direct_vjp.py``; the
tangent does not need the symmetry (it has no source-side half).
"""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from jaccpot.pallas._tangent import drop_tangent_axis, with_tangent_axis
from jaccpot.pallas.m2l_real_csr import pallas_m2l_real_csr_supported
from jaccpot.pallas.nearfield_leafpair_csr import (
    nearfield_leafpair_csr_sorted_direct_pallas,
)
from jaccpot.pallas.nearfield_leafpair_csr_direct_vjp import (
    _DirectStatic,
    _near_tangent_twin,
    nearfield_leafpair_csr_sorted_direct_pallas_tangent,
    nearfield_leafpair_csr_sorted_direct_pallas_tangent_jnp,
)
from tests.unit.operators.test_nearfield_direct_vjp import (
    _pair_sum,
    _symmetric_case,
    _topology,
)

jax.config.update("jax_enable_x64", True)

KERNELS = ("ferrers3", "plummer", "wendland_c2")
SOFTENING = 0.5  # a compact support of 1.23 (ferrers3) / 1.5 (wendland_c2)
_LIVE = (True, True, True, True)
#: float64: sums of O(100) pair terms in another order; float32: the same, at 2^-23
_TOL = {np.float64: 1e-12, np.float32: 2e-5}


class _Case(NamedTuple):
    """``test_nearfield_direct_vjp._symmetric_case``'s system, typed."""

    pos: Array
    mass: Array
    starts: Array
    counts: Array
    nbr: Array
    offsets: Array
    row_counts: Array
    pairs: tuple[np.ndarray, np.ndarray]
    n: int
    n_live: int
    W: int
    L: int


def _case(seed: int, **kw: Any) -> _Case:
    """:func:`_symmetric_case` (same keywords) as a :class:`_Case`."""
    build = cast(Any, _symmetric_case)
    return _Case(**build(seed, **kw))


def _topo(c: _Case) -> tuple[Array, Array, Array, Array, Array]:
    """The leaf ranges and the CSR, in the seam's operand order."""
    return (c.starts, c.counts, c.nbr, c.offsets, c.row_counts)


def _native_or_skip(interpret: bool) -> None:
    """Native Triton lowering needs an Ampere+ GPU; interpret mode runs anywhere."""
    if not interpret and not pallas_m2l_real_csr_supported():
        pytest.skip("native Pallas GPU lowering not available here")


def _rel(a, b) -> float:
    """Max-norm difference relative to the reference's max-norm."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _static(
    c: _Case, kernel: str, *, row_limit=None, interpret: bool = True
) -> _DirectStatic:
    return _DirectStatic(
        leaf_width=c.W,
        chunk=1,
        num_warps=None,
        num_stages=1,
        target_subtile=None,
        interpret=interpret,
        with_force_scale=False,
        row_limit=row_limit,
        source_tile=None,
        source_flags=None,
        softening_kernel=kernel,
        tile_skip=None,
    )


def _scalars(dtype):
    return jnp.asarray(SOFTENING**2, dtype), jnp.asarray(1.7, dtype)


def _tangents(c: _Case, k: int, dtype, seed: int = 0):
    rng = np.random.default_rng(200 + seed)
    return (
        jnp.asarray(rng.standard_normal((k, c.n, 3)).astype(dtype)),
        jnp.asarray(rng.standard_normal((k, c.n)).astype(dtype)),
        jnp.asarray(rng.standard_normal(k).astype(dtype)),
        jnp.asarray(rng.standard_normal(k).astype(dtype)),
    )


def _operands(c: _Case, dtype, k: int, seed: int = 0):
    soft, G = _scalars(dtype)
    return (c.pos, c.mass, *_topo(c), soft, G, *_tangents(c, k, dtype, seed))


def _both(c: _Case, kernel, k, dtype, *, row_limit=None, live=_LIVE, seed=0, **kw):
    args = _operands(c, dtype, k, seed)
    if not all(live):
        args = args[:9] + tuple(
            t if on else jnp.zeros_like(t) for t, on in zip(args[9:], live)
        )
    st = _static(c, kernel, row_limit=row_limit)
    (got,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent(
        *args, static=st, live=live, **kw
    )
    (want,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent_jnp(
        *args, static=st, live=live
    )
    return got, want


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize("kernel", KERNELS)
def test_tangent_op_matches_the_jnp_reference(kernel, k, dtype):
    c = _case(3, dtype=dtype)
    got, want = _both(c, kernel, k, dtype)
    assert got.shape == (k, c.n, 3) and got.dtype == jnp.dtype(dtype)
    assert np.all(np.isfinite(np.asarray(got)))
    assert np.max(np.abs(np.asarray(want))) > 0
    # measured: float64 <= 7.0e-16, float32 <= 4.1e-7 (runs, rest pieces: 5.4e-16,
    # 3.5e-7)
    assert _rel(got, want) < _TOL[dtype], _rel(got, want)
    # rows no leaf holds get nothing
    assert not np.any(np.asarray(got)[:, c.n_live :])


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("k", [1, 8])
@pytest.mark.parametrize(
    "case, row_limit",
    [
        (dict(seed=5, consecutive=True), None),  # every row's sources merge into runs
        (dict(seed=7, W=6), 2),  # W padded to 8 target lanes; rows in pieces of 2
    ],
)
def test_runs_and_rest_pieces(case, row_limit, k, dtype):
    c = _case(**case, dtype=dtype)
    if row_limit is not None:
        assert int(np.max(np.asarray(c.row_counts))) > 2 * row_limit  # 3+ pieces
    got, want = _both(c, "ferrers3", k, dtype, row_limit=row_limit)
    assert np.max(np.abs(np.asarray(want))) > 0
    assert _rel(got, want) < _TOL[dtype], _rel(got, want)


@pytest.mark.parametrize(
    "live",
    [
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
)
@pytest.mark.parametrize("kernel", ["ferrers3", "plummer"])
def test_each_tangent_alone(kernel, live):
    """A known-zero tangent (``live``) skips its terms; each alone is the reference."""
    c = _case(11)
    got, want = _both(c, kernel, 2, np.float64, live=live, seed=3)
    assert np.max(np.abs(np.asarray(want))) > 0
    assert _rel(got, want) < 1e-12


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("kernel", KERNELS)
def test_the_reference_primal_is_the_production_forward(kernel, dtype):
    """And ``jax.jvp`` of the independent pair sum agrees with the reference."""
    c = _case(13, dtype=dtype)
    soft, G = _scalars(dtype)
    twin = _near_tangent_twin(*_topo(c), num_particles=c.n, static=_static(c, kernel))
    acc, _ = nearfield_leafpair_csr_sorted_direct_pallas(
        c.pos,
        c.mass,
        *_topo(c),
        leaf_width=c.W,
        softening_sq=soft,
        G=G,
        chunk=1,
        softening_kernel=kernel,
        interpret=True,
    )
    # measured: float64 <= 2.7e-16, float32 <= 2.2e-7
    assert _rel(twin(c.pos, c.mass, soft, G), acc) < (
        1e-14 if dtype == np.float64 else 2e-6
    )
    if dtype == np.float64:
        args = _operands(c, dtype, 3, seed=5)
        (want,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent_jnp(
            *args, static=_static(c, kernel)
        )
        pair_sum = _pair_sum(c._asdict(), kernel)
        ind = jax.vmap(
            lambda a, b, s, g: jax.jvp(
                pair_sum, (c.pos, c.mass, soft, G), (a, b, s, g)
            )[1]
        )(*args[9:])
        # measured: <= 6.4e-16
        assert _rel(want, ind) < 1e-12


@pytest.mark.parametrize("columns", [3, 11])
@pytest.mark.parametrize("kernel", ["ferrers3", "plummer"])
def test_jacfwd_through_a_custom_jvp_seam(kernel, columns):
    c = _case(17)
    topo = _topo(c)
    st = _static(c, kernel)
    soft0, G0 = _scalars(np.float64)
    rng = np.random.default_rng(columns)
    A = jnp.asarray(rng.standard_normal((c.n * 3, columns)))
    B = jnp.asarray(rng.standard_normal((c.n, columns)))
    s_dir = jnp.asarray(rng.standard_normal(columns))
    g_dir = jnp.asarray(rng.standard_normal(columns))

    @jax.custom_jvp
    def seam(x, m, s, g):
        acc, _ = nearfield_leafpair_csr_sorted_direct_pallas(
            x,
            m,
            *topo,
            leaf_width=c.W,
            softening_sq=s,
            G=g,
            chunk=1,
            softening_kernel=kernel,
            interpret=True,
        )
        return acc

    @seam.defjvp
    def _rule(primals, tangents):
        out = seam(*primals)
        tk, flags = with_tangent_axis(tangents, primals)
        x, m, s, g = primals
        (t,) = drop_tangent_axis(
            nearfield_leafpair_csr_sorted_direct_pallas_tangent(
                x, m, *topo, s, g, *tk, static=st, live=flags
            )
        )
        return out, t

    twin = _near_tangent_twin(*topo, num_particles=c.n, static=st)

    def f(theta, fn):
        x = c.pos + 0.1 * (A @ theta).reshape(c.n, 3)
        m = c.mass * (1.0 + 0.1 * (B @ theta))
        return fn(x, m, soft0 * (1.0 + 0.1 * s_dir @ theta), G0 + g_dir @ theta)

    theta = jnp.zeros(columns)
    jac = jax.jacfwd(lambda th: f(th, seam))(theta)
    want = jax.jacfwd(lambda th: f(th, twin))(theta)
    assert jac.shape == (c.n, 3, columns)
    # measured: <= 5.4e-16
    assert _rel(jac, want) < 1e-12
    y, _ = jax.jvp(seam, (c.pos, c.mass, soft0, G0), (c.pos, c.mass, soft0, G0))
    assert np.array_equal(np.asarray(y), np.asarray(seam(c.pos, c.mass, soft0, G0)))
    with pytest.raises(NotImplementedError, match="ad_mode='reverse'"):
        jax.grad(lambda th: jnp.sum(f(th, seam)))(theta)


@pytest.mark.parametrize("kernel", KERNELS)
def test_central_differences_of_the_production_forward_anchor_the_reference(kernel):
    """Directional central differences in each input (float64, interpret), the
    forward being the production kernel itself."""
    c = _case(19)
    soft, G = _scalars(np.float64)
    args = _operands(c, np.float64, 4, seed=7)
    # one direction per input: positions, masses, softening, G
    tans = [jnp.zeros_like(t) for t in args[9:]]
    directions = []
    for i in range(4):
        d = [t for t in tans]
        d[i] = args[9 + i]
        directions.append(d)
    stacked = [jnp.stack([directions[k][i][k] for k in range(4)]) for i in range(4)]
    (want,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent_jnp(
        *args[:9], *stacked, static=_static(c, kernel)
    )

    def prod(x, m, s, g):
        acc, _ = nearfield_leafpair_csr_sorted_direct_pallas(
            x,
            m,
            *_topo(c),
            leaf_width=c.W,
            softening_sq=s,
            G=g,
            chunk=1,
            softening_kernel=kernel,
            interpret=True,
        )
        return acc

    prod = jax.jit(prod)
    primals = [c.pos, c.mass, soft, G]
    # steps: second-order truncation ~h^2 against round-off ~1e-16 |a| / h
    for k, h in enumerate((1e-5, 1e-5, 1e-6, 1e-4)):
        v = stacked[k][k]
        up = list(primals)
        dn = list(primals)
        up[k] = primals[k] + h * v
        dn[k] = primals[k] - h * v
        fd = (np.asarray(prod(*up)) - np.asarray(prod(*dn))) / (2.0 * h)
        assert np.max(np.abs(fd)) > 1e-3
        # measured: <= 1.8e-9 (positions), <= 1.4e-10 for the rest
        assert _rel(want[k], fd) < 1e-7, (k, _rel(want[k], fd))


@pytest.mark.parametrize("kernel", ["ferrers3", "plummer"])
@pytest.mark.parametrize("softening", [SOFTENING, 0.0])
def test_coincident_particles_and_self_pairs_are_finite(kernel, softening):
    """Two coincident particles in one leaf and one across leaves: with softening the
    pair's tangent is the kernel's ``-G m g(0) dd`` (the reference's), without it the
    pair adds nothing; never NaN. The self pass's diagonal is always excluded. (A
    compact kernel at ``eps = 0`` is Newton; its ``d/d(eps^2)`` is zero there in
    the kernel and the reference.)"""
    c = _case(23)
    pos = np.asarray(c.pos).copy()
    starts = np.asarray(c.starts)
    pos[starts[0] + 1] = pos[starts[0]]  # leaf 0 is full: two coincident particles
    nbr_of_0 = np.asarray(c.nbr)[int(c.offsets[0]) : int(c.offsets[1])]
    assert nbr_of_0.size > 0
    pos[starts[int(nbr_of_0[0])]] = pos[starts[0] + 2]  # one across a row entry
    c = c._replace(pos=jnp.asarray(pos))
    soft = jnp.asarray(softening**2)
    G = jnp.asarray(1.3)
    tx, tm, ts, tg = _tangents(c, 2, np.float64, seed=9)
    args = (c.pos, c.mass, *_topo(c), soft, G, tx, tm, ts, tg)
    st = _static(c, kernel)
    live = _LIVE
    (got,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent(
        *args, static=st, live=live
    )
    (want,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent_jnp(
        *args, static=st, live=live
    )
    assert np.all(np.isfinite(np.asarray(got)))
    assert np.all(np.isfinite(np.asarray(want)))
    assert _rel(got, want) < 1e-12


def test_the_wide_accumulator_is_within_float32_round_off():
    """``accum="wide"`` (opt-in; also ``JACCPOT_NEARFIELD_TANGENT_ACCUM=wide``):
    float32 inputs, float64 across source runs; against the float64 reference."""
    c32 = _case(29, consecutive=True, dtype=np.float32)
    c64 = _case(29, consecutive=True)
    st = _static(c32, "ferrers3", row_limit=2)
    args32 = _operands(c32, np.float32, 2, seed=11)
    args64 = tuple(
        jnp.asarray(a, jnp.float64) if a.dtype == jnp.float32 else a for a in args32
    )
    (truth,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent_jnp(
        *args64, static=_static(c64, "ferrers3", row_limit=2)
    )
    (wide,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent(
        *args32, static=st, live=_LIVE, accum="wide"
    )
    (plain,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent(
        *args32, static=st, live=_LIVE, accum="input"
    )
    assert wide.dtype == jnp.float32
    # measured: wide 1.4e-7, input 1.6e-7 (too few pairs here to separate them)
    assert _rel(wide, truth) < 2e-6 and _rel(plain, truth) < 2e-6
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("JACCPOT_NEARFIELD_TANGENT_ACCUM", "wide")
        (env_wide,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent(
            *args32, static=st, live=_LIVE
        )
    assert np.array_equal(np.asarray(env_wide), np.asarray(wide))
    with pytest.raises(ValueError, match="accum must be one of"):
        nearfield_leafpair_csr_sorted_direct_pallas_tangent(
            *args32, static=st, live=_LIVE, accum="double"
        )


def test_self_terms_only_an_empty_neighbour_array_and_an_unknown_kernel():
    c = _case(31, p_edge=0.0)
    assert int(np.asarray(c.row_counts).sum()) == 0
    args = _operands(c, np.float64, 2)
    st = _static(c, "plummer")
    (want,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent_jnp(*args, static=st)
    for nbr in (c.nbr, jnp.zeros((0,), jnp.int32)):
        a = args[:4] + (nbr,) + args[5:]
        (got,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent(
            *a, static=st, live=_LIVE
        )
        assert np.max(np.abs(np.asarray(want))) > 0
        assert _rel(got, want) < 1e-12
    with pytest.raises(ValueError, match="softening_kernel"):
        nearfield_leafpair_csr_sorted_direct_pallas_tangent(
            *args, static=_static(c, "gaussian"), live=_LIVE
        )
    (zero,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent(
        *args, static=st, live=(False, False, False, False)
    )
    assert zero.shape == (2, c.n, 3) and not np.any(np.asarray(zero))


@pytest.mark.parametrize("k", [1, 8])
def test_native_gpu_parity(k):
    """The Triton tangent kernel against the reference, natively (skips without an
    Ampere card)."""
    _native_or_skip(False)
    c = _case(7, W=6, dtype=np.float32)
    args = _operands(c, np.float32, k)
    st = _static(c, "ferrers3", row_limit=2, interpret=False)
    (got,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent(
        *args, static=st, live=_LIVE
    )
    (want,) = nearfield_leafpair_csr_sorted_direct_pallas_tangent_jnp(*args, static=st)
    assert _rel(got, want) < _TOL[np.float32]
