"""Tiled real M2L CSR kernel (plan sub-10ms Phase 5) against the per-pair reference, interpret mode."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.m2l_real_csr import m2l_real_csr_jax
from jaccpot.pallas.m2l_real_csr_tiled import (
    m2l_real_csr_tiled_pallas,
    m2l_real_csr_tiled_supported,
)


def _case(seed, n, rows, order):
    rng = np.random.default_rng(seed)
    C = (order + 1) ** 2
    mult = jnp.asarray(rng.standard_normal((n, C)), jnp.float32)
    cent = jnp.asarray(rng.uniform(-1.0, 1.0, (n, 3)) * 4.0, jnp.float32)
    tgt = np.repeat(np.arange(n), rows)
    src = rng.integers(0, n, tgt.size)
    # never a pair with itself (r = 0)
    src = np.where(src == tgt, (src + 1) % n, src)
    return mult, cent, jnp.asarray(src, jnp.int32), jnp.asarray(tgt, jnp.int32)


@pytest.mark.parametrize("order", [4, 5, 6])
def test_tiled_matches_the_reference_on_ragged_rows(order):
    rng = np.random.default_rng(order)
    n = 40
    rows = rng.integers(0, 40, n)  # empty rows, rows shorter / longer than a tile
    rows[3] = 0
    rows[5] = 16
    rows[7] = 33
    mult, cent, src, tgt = _case(order, n, rows, order)
    ref = m2l_real_csr_jax(mult, cent, src, tgt, order=order)
    got = m2l_real_csr_tiled_pallas(mult, cent, src, tgt, order=order, interpret=True)
    r, g = np.asarray(ref), np.asarray(got)
    scale = np.abs(r).max()
    assert np.allclose(g, r, rtol=1e-4, atol=1e-4 * scale)
    assert np.all(g[3] == 0.0)  # empty row


def test_tiled_honours_the_active_prefix_and_padding():
    order = 5
    n = 24
    mult, cent, src, tgt = _case(11, n, np.full(n, 9), order)
    P = int(src.shape[0])
    # pad with -1 pairs and mark only the first half live
    src_p = jnp.concatenate([src, jnp.full((16,), -1, jnp.int32)])
    tgt_p = jnp.concatenate([tgt, jnp.full((16,), -1, jnp.int32)])
    live = P // 2
    ref = m2l_real_csr_jax(mult, cent, src_p, tgt_p, order=order, active_pair_count=jnp.asarray(live))
    got = m2l_real_csr_tiled_pallas(
        mult, cent, src_p, tgt_p, order=order, active_pair_count=jnp.asarray(live), interpret=True
    )
    r, g = np.asarray(ref), np.asarray(got)
    assert np.allclose(g, r, rtol=1e-4, atol=1e-4 * np.abs(r).max())


def test_tiled_is_jittable_and_rejects_low_orders():
    order = 5
    mult, cent, src, tgt = _case(3, 20, np.full(20, 5), order)
    f = jax.jit(lambda m, c: m2l_real_csr_tiled_pallas(m, c, src, tgt, order=order, interpret=True))
    assert np.all(np.isfinite(np.asarray(f(mult, cent))))
    assert not m2l_real_csr_tiled_supported(3)
    with pytest.raises(ValueError):
        m2l_real_csr_tiled_pallas(mult[:, :16], cent, src, tgt, order=3, interpret=True)
