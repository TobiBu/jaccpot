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
