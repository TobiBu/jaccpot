"""Reverse of the pair-per-lane M2L CSR kernel vs ``jax.vjp`` of the per-pair reference (fp64, interpret).

Both halves -- multipoles AND centres -- on ragged rows with padding and a live
prefix, including axis-aligned pairs (``rho == 0``) where the unguarded transpose
would be NaN.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.m2l_real_rot_scale import m2l_rot_scale_real_batch
from jaccpot.pallas.m2l_real_csr import m2l_real_csr_jax, pallas_m2l_real_csr_supported
from jaccpot.pallas.m2l_real_csr_lanes import (
    m2l_real_csr_lanes_pallas,
    m2l_real_csr_lanes_pallas_cvjp,
)

jax.config.update("jax_enable_x64", True)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _native_or_skip(interpret: bool) -> None:
    """Native Triton lowering needs an Ampere+ GPU; interpret mode runs anywhere."""
    if not interpret and not pallas_m2l_real_csr_supported():
        pytest.skip("native Pallas GPU lowering not available here")


def _case(seed, n, rows, order):
    rng = np.random.default_rng(seed)
    C = (order + 1) ** 2
    mult = jnp.asarray(rng.standard_normal((n, C)), jnp.float64)
    cent = jnp.asarray(rng.uniform(-1.0, 1.0, (n, 3)) * 4.0, jnp.float64)
    tgt = np.repeat(np.arange(n), rows)
    src = rng.integers(0, n, tgt.size)
    src = np.where(src == tgt, (src + 1) % n, src)
    return mult, cent, jnp.asarray(src, jnp.int32), jnp.asarray(tgt, jnp.int32)


@pytest.mark.parametrize("interpret", [True, False])
@pytest.mark.parametrize("order", [2, 3, 5])
@pytest.mark.parametrize("k_lanes", [8, 32])
def test_lanes_reverse_matches_vjp_of_the_reference_in_both_halves(
    order, k_lanes, interpret
):
    _native_or_skip(interpret)
    rng = np.random.default_rng(order)
    n = 40
    rows = rng.integers(0, 40, n)
    rows[3] = 0
    rows[5] = k_lanes
    rows[7] = 2 * k_lanes + 1
    mult, cent, src, tgt = _case(order, n, rows, order)
    C = (order + 1) ** 2
    cot = jnp.asarray(rng.standard_normal((n, C)), jnp.float64)

    def ref(m, c):
        return m2l_real_csr_jax(m, c, src, tgt, order=order)

    def got(m, c):
        return m2l_real_csr_lanes_pallas_cvjp(
            m, c, src, tgt, None, order, k_lanes, interpret, "triton", 1
        )

    out_r, vjp_r = jax.vjp(ref, mult, cent)
    out_g, vjp_g = jax.vjp(got, mult, cent)
    assert _rel(out_g, out_r) < 1e-10
    mb_r, cb_r = vjp_r(cot)
    mb_g, cb_g = vjp_g(cot)
    assert np.all(np.isfinite(np.asarray(mb_g))) and np.all(
        np.isfinite(np.asarray(cb_g))
    )
    assert np.linalg.norm(np.asarray(cb_r)) > 0
    assert _rel(mb_g, mb_r) < 1e-9, f"multipole half rel-L2 {_rel(mb_g, mb_r):.3e}"
    assert _rel(cb_g, cb_r) < 1e-9, f"centre half rel-L2 {_rel(cb_g, cb_r):.3e}"


@pytest.mark.parametrize("interpret", [True, False])
def test_lanes_reverse_axis_aligned_pairs_with_padding_and_prefix(interpret):
    _native_or_skip(interpret)
    order = 4
    n = 12
    rng = np.random.default_rng(9)
    C = (order + 1) ** 2
    mult = jnp.asarray(rng.standard_normal((n, C)), jnp.float64)
    cent = np.zeros((n, 3))
    cent[: n // 2, 2] = np.arange(n // 2) * 1.5  # rho == 0 pairs along z
    cent[n // 2 :, 0] = np.arange(n // 2) * 1.5 + 0.25  # dz == 0 pairs along x
    cent = jnp.asarray(cent)
    tgt = np.repeat(np.arange(n), n - 1)
    src = np.concatenate([np.delete(np.arange(n), t) for t in range(n)])
    src_p = jnp.concatenate(
        [jnp.asarray(src, jnp.int32), jnp.full((7,), -1, jnp.int32)]
    )
    tgt_p = jnp.concatenate(
        [jnp.asarray(tgt, jnp.int32), jnp.full((7,), -1, jnp.int32)]
    )
    live = jnp.asarray(src.size // 2)
    cot = jnp.asarray(rng.standard_normal((n, C)), jnp.float64)

    def ref(m, c):
        return m2l_real_csr_jax(m, c, src_p, tgt_p, order=order, active_pair_count=live)

    def got(m, c):
        return m2l_real_csr_lanes_pallas_cvjp(
            m, c, src_p, tgt_p, live, order, 32, interpret, "triton", 1
        )

    _, vjp_r = jax.vjp(ref, mult, cent)
    _, vjp_g = jax.vjp(got, mult, cent)
    mb_r, cb_r = vjp_r(cot)
    mb_g, cb_g = vjp_g(cot)
    assert np.all(np.isfinite(np.asarray(mb_g))) and np.all(
        np.isfinite(np.asarray(cb_g))
    )
    assert _rel(mb_g, mb_r) < 1e-9
    # The unguarded per-pair twin is singular at rho == 0 (0/0 in arctan2's
    # transpose): every node with an on-axis pair gets a NaN centre cotangent
    # from the reference, which is exactly what the guarded reverse avoids.
    # Compare where the reference is finite, and require that to be non-empty.
    cb_r_np, cb_g_np = np.asarray(cb_r), np.asarray(cb_g)
    finite = np.all(np.isfinite(cb_r_np), axis=1)
    assert finite.any() and not finite.all()
    assert _rel(cb_g_np[finite], cb_r_np[finite]) < 1e-9
    plain = m2l_real_csr_lanes_pallas(
        mult,
        cent,
        src_p,
        tgt_p,
        order=order,
        active_pair_count=live,
        interpret=interpret,
    )
    assert np.array_equal(np.asarray(plain), np.asarray(got(mult, cent)))


@pytest.mark.parametrize("order", [2, 4])
def test_presorted_row_offsets_equal_the_coo_read_forward_and_reverse(order):
    """``presorted``: ``targets`` are the row offsets of a CSR-ordered list.

    The forward reads (sources, offsets) as is; the reverse expands one target
    per entry. Both equal the COO call on the same (already target-sorted) list,
    padding behind the live prefix included (interpret mode).
    """
    rng = np.random.default_rng(10 + order)
    n, K = 30, 8
    rows = rng.integers(0, 3 * K, n)
    rows[0] = 0
    rows[-1] = 0
    rows[4] = 2 * K + 1
    mult, cent, src, tgt = _case(order, n, rows, order)
    pad = 7
    src_p = jnp.concatenate([src, jnp.full((pad,), -1, jnp.int32)])
    tgt_p = jnp.concatenate([tgt, jnp.full((pad,), -1, jnp.int32)])
    offsets = jnp.asarray(np.concatenate([[0], np.cumsum(rows)]), jnp.int32)
    live = jnp.asarray(int(src.shape[0]), jnp.int32)
    C = (order + 1) ** 2
    cot = jnp.asarray(rng.standard_normal((n, C)), jnp.float64)

    def coo(m, c):
        return m2l_real_csr_lanes_pallas_cvjp(
            m, c, src_p, tgt_p, live, order, K, True, "triton", 1
        )

    def csr(m, c):
        return m2l_real_csr_lanes_pallas_cvjp(
            m, c, src_p, offsets, live, order, K, True, "triton", 1, None, True
        )

    out_c, vjp_c = jax.vjp(coo, mult, cent)
    out_s, vjp_s = jax.vjp(csr, mult, cent)
    assert np.array_equal(np.asarray(out_s), np.asarray(out_c))
    for x, y in zip(vjp_s(cot), vjp_c(cot)):
        assert np.array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("interpret", [True, False])
@pytest.mark.parametrize("order", [2, 4])
def test_lanes_reverse_matches_the_operator_rule_on_the_axis(order, interpret):
    """On-axis pairs: the centre half equals the pure-JAX operator's, not zero.

    The test above can only compare where the unguarded twin is finite, which
    leaves out exactly the nodes with an on-axis pair. The reference here is
    :func:`m2l_rot_scale_real_batch`, which carries the analytic transverse
    derivative (``operators/_transverse_degeneracy_jvp``), so the comparison
    covers every node -- pairs at ``rho == 0`` and pairs one band-width off it
    (``rho/r = 1e-12``, inside ``rho^2 <= eps r^2``), where the polar route's
    guards give zero and a cancellation-dominated value respectively. Before the
    kernel took that limit the centre half missed by O(1) here.
    """
    _native_or_skip(interpret)
    n = 10
    rng = np.random.default_rng(31 + order)
    C = (order + 1) ** 2
    mult = jnp.asarray(rng.standard_normal((n, C)), jnp.float64)
    cent = np.zeros((n, 3))
    cent[:, 2] = np.arange(n) * 1.5
    cent[n // 2 :, 0] = 1.0e-12  # in the band, not exactly on the axis
    cent[-1, :2] = [0.3, -0.2]  # one node off the axis, so both kinds of pair meet
    cent = jnp.asarray(cent)
    tgt = np.repeat(np.arange(n), n - 1)
    src = np.concatenate([np.delete(np.arange(n), t) for t in range(n)])
    src, tgt = jnp.asarray(src, jnp.int32), jnp.asarray(tgt, jnp.int32)
    cot = jnp.asarray(rng.standard_normal((n, C)), jnp.float64)

    deltas = np.asarray(cent)[np.asarray(tgt)] - np.asarray(cent)[np.asarray(src)]
    rho2 = deltas[:, 0] ** 2 + deltas[:, 1] ** 2
    on_axis = rho2 <= np.finfo(np.float64).eps * (rho2 + deltas[:, 2] ** 2)
    assert np.any(on_axis & (rho2 == 0)) and np.any(on_axis & (rho2 > 0))
    assert not np.all(on_axis)

    def ref(m, c):
        contrib = m2l_rot_scale_real_batch(m[src], c[tgt] - c[src], order=order)
        return jax.ops.segment_sum(contrib, tgt, num_segments=n)

    def got(m, c):
        return m2l_real_csr_lanes_pallas_cvjp(
            m, c, src, tgt, None, order, 8, interpret, "triton", 1
        )

    out_r, vjp_r = jax.vjp(ref, mult, cent)
    out_g, vjp_g = jax.vjp(got, mult, cent)
    assert _rel(out_g, out_r) < 1e-10
    mb_r, cb_r = vjp_r(cot)
    mb_g, cb_g = vjp_g(cot)
    assert np.all(np.isfinite(np.asarray(cb_r)))
    assert _rel(mb_g, mb_r) < 1e-9, f"multipole half rel-L2 {_rel(mb_g, mb_r):.3e}"
    assert _rel(cb_g, cb_r) < 1e-9, f"centre half rel-L2 {_rel(cb_g, cb_r):.3e}"


def test_a_presorted_reverse_sorts_the_pairs_once():
    """On a target-sorted list the reverse's by-target pass reads the forward's CSR:
    one sort of the ``P`` pairs (by source), not two (each a ``P``-sized key/value
    sort and its scratch, the far reverse's largest temporaries)."""
    order, n, K = 2, 30, 8
    rows = np.random.default_rng(3).integers(0, 3 * K, n)
    mult, cent, src, tgt = _case(order, n, rows, order)
    offsets = jnp.asarray(np.concatenate([[0], np.cumsum(rows)]), jnp.int32)
    live = jnp.asarray(int(src.shape[0]), jnp.int32)
    cot = jnp.ones((n, (order + 1) ** 2), jnp.float64)

    def reverse(m, c, g):
        return jax.vjp(
            lambda m, c: m2l_real_csr_lanes_pallas_cvjp(
                m, c, src, offsets, live, order, K, True, "triton", 1, None, True
            ),
            m,
            c,
        )[1](g)

    jaxpr = jax.make_jaxpr(reverse)(mult, cent, cot)

    def sorts(j) -> int:
        count = 0
        for eqn in j.eqns:
            count += eqn.primitive.name == "sort"
            for p in eqn.params.values():
                for sub in p if isinstance(p, (list, tuple)) else (p,):
                    inner = getattr(sub, "jaxpr", sub)
                    if hasattr(inner, "eqns"):
                        count += sorts(inner)
        return count

    assert sorts(jaxpr.jaxpr) == 1
