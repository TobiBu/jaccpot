"""The direct CSR near field's reverse (the production layout) vs autodiff and differences.

:func:`nearfield_leafpair_csr_sorted_direct_pallas_cvjp` is the production near-field
call under a ``custom_vjp`` whose reverse runs on the same sorted layout. Pinned here,
for every softening kernel, in float64 on the Pallas interpreter (and natively on an
Ampere card):

* its cotangents (positions, masses, ``softening_sq``, ``G``) equal ``jax.vjp`` of a
  pure-JAX pair sum over the same pairs (rows + self, diagonal excluded) to float64
  round-off -- on leaves of every occupancy, empty padding leaves, rows past every
  leaf's particles, an empty row, rows of consecutive leaves (the merged source runs)
  and rows longer than the row limit (the reverse's in-place pieces);
* central finite differences of the PRODUCTION forward agree with them;
* its forward IS the production call: the same bits, the same lowered program for
  traced operands; the force-scale lane passes through and gets no cotangent, and the
  potential lane is refused;
* in float32 it and the table layout's reverse are both within single-precision
  round-off of the float64 truth (the two float32 layouts are not compared with each
  other: the scalars are sums with cancellation, and two float32 paths agreeing proves
  nothing).

The neighbour CSR is built SYMMETRIC, as the walk emits it: the reverse takes each
source-side cotangent from the other end of the pair.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.m2l_real_csr import pallas_m2l_real_csr_supported
from jaccpot.pallas.nearfield_leafpair_csr import (
    build_leafpair_chunk_table,
    leafpair_chunk_capacity,
    nearfield_leafpair_csr_pallas_cvjp,
    nearfield_leafpair_csr_sorted_direct_pallas,
)
from jaccpot.pallas.nearfield_leafpair_csr_direct_vjp import (
    nearfield_leafpair_csr_sorted_direct_pallas_cvjp,
    nearfield_leafpair_csr_sorted_direct_reverse_pallas,
)
from jaccpot.softening import pair_factors, softening_params_from_sq, support_factor

KERNELS = ("plummer", "ferrers3", "wendland_c2")
SOFTENING = 0.5  # a compact support of 1.23 (ferrers3) / 1.5 (wendland_c2)


def _native_or_skip(interpret: bool) -> None:
    """Native Triton lowering needs an Ampere+ GPU; interpret mode runs anywhere."""
    if not interpret and not pallas_m2l_real_csr_supported():
        pytest.skip("native Pallas GPU lowering not available here")


def _symmetric_case(
    seed,
    *,
    num_live=10,
    num_pad=2,
    W=8,
    n_dead=3,
    p_edge=0.45,
    consecutive=False,
    dtype=np.float64,
):
    """Sorted particles in leaf ranges and a SYMMETRIC leaf CSR over the live leaves.

    Leaf occupancies 1..W (leaf 0 full, leaf 1 a single particle), ``num_pad`` empty
    padding leaves starting past the live rows, ``n_dead`` rows past every leaf's
    particles, and leaf 3's row empty. Rows are sorted, so neighbouring entries are
    often adjacent particle ranges (the kernel's merged runs); ``consecutive`` lists
    each leaf's three nearest leaves on either side, so they always are.
    """
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, W + 1, size=num_live)
    counts[0], counts[1] = W, 1
    counts = np.concatenate([counts, np.zeros(num_pad, np.int64)])
    L = counts.size
    n_live = int(counts.sum())
    n = n_live + n_dead
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    starts[num_live:] = n_live
    if consecutive:
        gap = np.abs(np.arange(num_live)[:, None] - np.arange(num_live)[None, :])
        adj = gap <= 3
    else:
        adj = rng.random((num_live, num_live)) < p_edge
    adj = np.triu(adj, 1)
    adj = adj | adj.T
    adj[3, :] = adj[:, 3] = False
    rows = [np.flatnonzero(adj[i]) for i in range(num_live)]
    rows += [np.zeros(0, np.int64) for _ in range(num_pad)]
    row_counts = np.array([r.size for r in rows])
    nbr = np.concatenate(rows + [np.zeros(4, np.int64)]).astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum(row_counts)]).astype(np.int32)
    pos = rng.standard_normal((n, 3))
    mass = np.abs(rng.standard_normal(n)) + 0.1
    # the (target, source) pairs the kernel sums: rows + self, diagonal excluded
    pi, pj = [], []
    for t in range(L):
        for s in list(rows[t]) + [t]:
            for a in range(starts[t], starts[t] + counts[t]):
                for b in range(starts[s], starts[s] + counts[s]):
                    if a != b:
                        pi.append(a)
                        pj.append(b)
    return dict(
        pos=jnp.asarray(pos.astype(dtype)),
        mass=jnp.asarray(mass.astype(dtype)),
        starts=jnp.asarray(starts, jnp.int32),
        counts=jnp.asarray(counts, jnp.int32),
        nbr=jnp.asarray(nbr),
        offsets=jnp.asarray(offsets),
        row_counts=jnp.asarray(row_counts, jnp.int32),
        pairs=(np.asarray(pi), np.asarray(pj)),
        n=n,
        n_live=n_live,
        W=W,
        L=L,
    )


def _topology(c):
    return (c["starts"], c["counts"], c["nbr"], c["offsets"], c["row_counts"])


def _pair_sum(c, kernel):
    """Pure-JAX pair sum over the case's pairs: ``a_i = sum_j -G m_j g(r_ij^2) d_ij``."""
    pi, pj = (jnp.asarray(p) for p in c["pairs"])

    def acc(pos, mass, soft, g):
        params = softening_params_from_sq(kernel, soft, pos.dtype)
        d = pos[pi] - pos[pj]
        gk, _, _ = pair_factors(jnp.sum(d * d, axis=1), params, kernel)
        contrib = (-g * mass[pj] * gk)[:, None] * d
        return jax.ops.segment_sum(contrib, pi, num_segments=c["n"])

    return acc


def _rel(a, b):
    """Max-norm difference relative to the reference's max-norm."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


@pytest.mark.parametrize("interpret", [True, False])
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "case, row_limit",
    [
        (dict(seed=3), None),
        (dict(seed=5, consecutive=True), None),  # every row's sources merge into runs
        (dict(seed=7, W=6), 2),  # W padded to 8 target lanes; rows in pieces of 2
    ],
)
def test_reverse_matches_vjp_of_the_pure_jax_pair_sum(
    case, row_limit, kernel, interpret
):
    _native_or_skip(interpret)
    c = _symmetric_case(**case)
    soft = jnp.asarray(SOFTENING**2)
    G = jnp.asarray(1.7)
    if row_limit is not None:
        assert int(np.max(np.asarray(c["row_counts"]))) > 2 * row_limit  # 3+ pieces
    if kernel != "plummer":
        # both sides of the support are exercised: polynomial and Newtonian pairs
        pi, pj = c["pairs"]
        pos = np.asarray(c["pos"])
        r = np.linalg.norm(pos[pi] - pos[pj], axis=1)
        h = support_factor(kernel) * SOFTENING
        assert 0.05 < np.mean(r < h) < 0.95

    common = dict(
        leaf_width=c["W"],
        softening_sq=soft,
        G=G,
        chunk=1,
        row_limit=row_limit,
        softening_kernel=kernel,
        interpret=interpret,
    )
    out_r, vjp_r = jax.vjp(_pair_sum(c, kernel), c["pos"], c["mass"], soft, G)
    acc, _ = nearfield_leafpair_csr_sorted_direct_pallas(
        c["pos"], c["mass"], *_topology(c), **common
    )
    # float64 round-off: the kernel's tiled sums against a segment sum
    assert _rel(acc, out_r) < 1e-12
    w = jnp.asarray(np.random.default_rng(11).standard_normal((c["n"], 3)))
    got = nearfield_leafpair_csr_sorted_direct_reverse_pallas(
        c["pos"], c["mass"], *_topology(c), w, **common
    )
    names = ("positions", "masses", "softening_sq", "G")
    for name, a, b in zip(names, got, vjp_r(w)):
        assert np.all(np.isfinite(np.asarray(a))), name
        assert np.max(np.abs(np.asarray(b))) > 0, name
        # float64 round-off of sums of O(100) pair terms (measured <= 6.8e-15)
        assert _rel(a, b) < 1e-12, (name, _rel(a, b))
    pos_bar, mass_bar = got[:2]
    # rows no leaf holds get nothing
    assert not np.any(np.asarray(pos_bar)[c["n_live"] :])
    assert not np.any(np.asarray(mass_bar)[c["n_live"] :])


@pytest.mark.parametrize("kernel", KERNELS)
def test_reverse_matches_central_differences_of_the_production_forward(kernel):
    """d/dx, d/dm, d/d(eps^2), d/dG of ``sum(acc . w)``, the forward being the
    production kernel itself (float64, interpret), along random directions."""
    c = _symmetric_case(13)
    soft = jnp.asarray(SOFTENING**2)
    G = jnp.asarray(0.9)
    w = jnp.asarray(np.random.default_rng(2).standard_normal((c["n"], 3)))

    def loss(fn, p, m, s, g):
        acc, _ = fn(
            p,
            m,
            *_topology(c),
            leaf_width=c["W"],
            softening_sq=s,
            G=g,
            chunk=1,
            softening_kernel=kernel,
            interpret=True,
        )
        return jnp.sum(acc * w)

    grads = jax.grad(
        lambda *a: loss(nearfield_leafpair_csr_sorted_direct_pallas_cvjp, *a),
        argnums=(0, 1, 2, 3),
    )(c["pos"], c["mass"], soft, G)
    prod = jax.jit(lambda *a: loss(nearfield_leafpair_csr_sorted_direct_pallas, *a))
    rng = np.random.default_rng(4)
    primals = [c["pos"], c["mass"], soft, G]
    # steps: second-order truncation ~h^2 against round-off ~1e-16 |L| / h
    for k, h in enumerate((1e-5, 1e-5, 1e-6, 1e-4)):
        v = jnp.asarray(rng.standard_normal(np.shape(primals[k])))
        up = list(primals)
        dn = list(primals)
        up[k] = primals[k] + h * v
        dn[k] = primals[k] - h * v
        fd = (float(prod(*up)) - float(prod(*dn))) / (2.0 * h)
        ad = float(jnp.sum(grads[k] * v))
        assert abs(ad) > 1e-3
        # measured: <= 1.4e-8 relative (positions, round-off of the differences),
        # <= 7e-10 for the rest; an algebra slip shows at 1e-3 or more
        assert abs(fd - ad) <= 1e-6 * abs(ad), (k, fd, ad)


@pytest.mark.parametrize("interpret", [True, False])
@pytest.mark.parametrize("with_force_scale", [False, True])
def test_the_forward_is_the_production_call(with_force_scale, interpret):
    """float32 at the production defaults (source tiles, flags, row limit from the
    environment's defaults): the primal of ``jax.vjp`` has the production call's
    bits, and for traced operands the jitted programs are the same text."""
    _native_or_skip(interpret)
    c = _symmetric_case(17, dtype=np.float32)
    soft = jnp.asarray(0.05**2, jnp.float32)
    G = jnp.asarray(1.3, jnp.float32)
    kw = dict(
        leaf_width=c["W"],
        chunk=64,
        softening_kernel="ferrers3",
        interpret=interpret,
        with_force_scale=with_force_scale,
    )

    def _program(fn):
        def near(p, m, ls, lc, nb, off, cnt, s, g):
            return fn(p, m, ls, lc, nb, off, cnt, softening_sq=s, G=g, **kw)

        return near

    prod = _program(nearfield_leafpair_csr_sorted_direct_pallas)
    diff = _program(nearfield_leafpair_csr_sorted_direct_pallas_cvjp)
    args = (c["pos"], c["mass"], *_topology(c), soft, G)
    want = jax.jit(prod)(*args)
    got, _ = jax.vjp(lambda p, m: jax.jit(diff)(p, m, *args[2:]), *args[:2])
    assert np.array_equal(np.asarray(got[0]), np.asarray(want[0]))
    if with_force_scale:
        assert np.array_equal(np.asarray(got[1]), np.asarray(want[1]))
    else:
        assert got[1] is None and want[1] is None
    text_prod = jax.jit(prod).lower(*args).as_text(debug_info=False)
    text_diff = jax.jit(diff).lower(*args).as_text(debug_info=False)
    assert text_prod == text_diff


def test_the_force_scale_lane_gets_no_cotangent():
    """With ``with_force_scale`` the accelerations' gradient is the one without it,
    and a cotangent on the force-scale lane is dropped (it only sets thresholds)."""
    c = _symmetric_case(19)
    soft = jnp.asarray(SOFTENING**2)
    G = jnp.asarray(1.1)
    w = jnp.asarray(np.random.default_rng(6).standard_normal((c["n"], 3)))
    u = jnp.asarray(np.random.default_rng(7).standard_normal(c["n"]))

    def loss(p, m, s, g, *, with_force_scale, scale_weight):
        acc, fb = nearfield_leafpair_csr_sorted_direct_pallas_cvjp(
            p,
            m,
            *_topology(c),
            leaf_width=c["W"],
            softening_sq=s,
            G=g,
            chunk=1,
            softening_kernel="ferrers3",
            interpret=True,
            with_force_scale=with_force_scale,
        )
        out = jnp.sum(acc * w)
        if with_force_scale:
            assert fb is not None and fb.shape == (c["n"],)
            out = out + scale_weight * jnp.sum(fb * u)
        return out

    def grads(**kw):
        g = jax.grad(lambda *a: loss(*a, **kw), argnums=(0, 1, 2, 3))
        return [np.asarray(x) for x in g(c["pos"], c["mass"], soft, G)]

    plain = grads(with_force_scale=False, scale_weight=0.0)
    for weight in (0.0, 1.0):
        for a, b in zip(grads(with_force_scale=True, scale_weight=weight), plain):
            assert np.array_equal(a, b)


def test_the_potential_lane_is_refused():
    c = _symmetric_case(19)
    with pytest.raises(ValueError, match="potential lane has no reverse"):
        nearfield_leafpair_csr_sorted_direct_pallas_cvjp(
            c["pos"],
            c["mass"],
            *_topology(c),
            leaf_width=c["W"],
            softening_sq=jnp.asarray(0.01),
            G=jnp.asarray(1.0),
            chunk=1,
            interpret=True,
            with_potential=True,
        )


def test_self_terms_only_and_an_empty_neighbour_array():
    """No neighbour entries at all (the self terms alone): the reverse is the pure-JAX
    sum's, and an empty ``neighbors`` array is accepted."""
    c = _symmetric_case(23, p_edge=0.0)
    assert int(np.asarray(c["row_counts"]).sum()) == 0
    soft = jnp.asarray(SOFTENING**2)
    G = jnp.asarray(1.0)
    w = jnp.asarray(np.random.default_rng(8).standard_normal((c["n"], 3)))
    _, vjp_r = jax.vjp(_pair_sum(c, "plummer"), c["pos"], c["mass"], soft, G)
    want = vjp_r(w)
    for nbr in (c["nbr"], jnp.zeros((0,), jnp.int32)):
        got = nearfield_leafpair_csr_sorted_direct_reverse_pallas(
            c["pos"],
            c["mass"],
            c["starts"],
            c["counts"],
            nbr,
            c["offsets"],
            c["row_counts"],
            w,
            leaf_width=c["W"],
            softening_sq=soft,
            G=G,
            chunk=1,
            softening_kernel="plummer",
            interpret=True,
        )
        for a, b in zip(got, want):
            assert _rel(a, b) < 1e-12


@pytest.mark.parametrize("kernel", KERNELS)
def test_float32_agrees_with_the_table_layouts_reverse(kernel):
    """The table layout's rule (:func:`nearfield_leafpair_csr_pallas_cvjp`, the
    differentiable lane until now) on the same pairs, gathered into ``(L, W)``
    tables, and the direct rule: each within single-precision round-off of the
    float64 pair sum's cotangents."""
    c = _symmetric_case(29, consecutive=True, dtype=np.float32)
    soft = jnp.asarray(SOFTENING**2, jnp.float32)
    G = jnp.asarray(1.3, jnp.float32)
    W, L = c["W"], c["L"]
    starts, counts = np.asarray(c["starts"]), np.asarray(c["counts"])
    slot = np.arange(W)[None, :]
    mask = slot < counts[:, None]
    ids = np.where(mask, starts[:, None] + slot, 0)
    cap = leafpair_chunk_capacity(int(c["nbr"].shape[0]), L, 4)
    table = build_leafpair_chunk_table(
        c["offsets"], c["row_counts"], chunk=4, capacity=cap
    )
    w = np.random.default_rng(9).standard_normal((c["n"], 3)).astype(np.float32)

    def table_layout(p, m, s, g):
        out = nearfield_leafpair_csr_pallas_cvjp(
            p[ids],
            m[ids],
            jnp.asarray(mask),
            c["nbr"],
            table,
            s,
            g,
            4,
            None,
            1,
            None,
            True,
            "input",
            True,
            None,
            kernel,
        )
        return out[..., :3]

    def direct_layout(p, m, s, g):
        return nearfield_leafpair_csr_sorted_direct_pallas_cvjp(
            p,
            m,
            *_topology(c),
            leaf_width=W,
            softening_sq=s,
            G=g,
            chunk=4,
            softening_kernel=kernel,
            interpret=True,
        )[0]

    leaf_w = np.where(mask[..., None], w[ids], 0.0).astype(np.float32)
    _, vjp_t = jax.vjp(table_layout, c["pos"], c["mass"], soft, G)
    _, vjp_d = jax.vjp(direct_layout, c["pos"], c["mass"], soft, G)
    table = vjp_t(jnp.asarray(leaf_w))
    direct = vjp_d(jnp.asarray(w))

    # the float64 truth on the same (float32) inputs, and the scale each cotangent's
    # round-off is measured against: its largest element for the per-particle ones;
    # for the two scalars, sums over every particle with cancellation, the sum of the
    # particles' absolute contributions (sum |w . acc| / |G| and sum |w . dacc/deps^2|)
    ref = _pair_sum(c, kernel)
    primals64 = [jnp.asarray(x, jnp.float64) for x in (c["pos"], c["mass"], soft, G)]
    w64 = jnp.asarray(w, jnp.float64)
    acc64, vjp_r = jax.vjp(ref, *primals64)
    truth = vjp_r(w64)
    _, dacc_ds = jax.jvp(
        lambda s_: ref(primals64[0], primals64[1], s_, primals64[3]),
        (primals64[2],),
        (jnp.ones_like(primals64[2]),),
    )
    scales = (
        np.max(np.abs(np.asarray(truth[0]))),
        np.max(np.abs(np.asarray(truth[1]))),
        float(jnp.sum(jnp.abs(jnp.sum(w64 * dacc_ds, axis=1)))),
        float(jnp.sum(jnp.abs(jnp.sum(w64 * acc64, axis=1))) / abs(primals64[3])),
    )
    names = ("positions", "masses", "softening_sq", "G")
    for name, a, b, t, scale in zip(names, direct, table, truth, scales):
        assert scale > 0, name
        err_d = float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(t))) / scale)
        err_t = float(np.max(np.abs(np.asarray(b, np.float64) - np.asarray(t))) / scale)
        # float32 round-off of sums of a few hundred pair terms on either layout
        # (measured <= 3.9e-7 on CPU and A100)
        assert err_d < 4e-6, (name, err_d, err_t)
        assert err_t < 4e-6, (name, err_d, err_t)
