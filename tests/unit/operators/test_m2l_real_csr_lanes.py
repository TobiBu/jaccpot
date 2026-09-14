"""Pair-per-lane real M2L CSR kernel (plan sub-10ms Phase 5) against the per-pair reference, interpret mode."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.m2l_real_csr import m2l_real_csr_jax
from jaccpot.pallas.m2l_real_csr_lanes import m2l_real_csr_lanes_pallas
from tests.unit._typecheck_budget import trim


def _case(seed, n, rows, order):
    rng = np.random.default_rng(seed)
    C = (order + 1) ** 2
    mult = jnp.asarray(rng.standard_normal((n, C)), jnp.float32)
    cent = jnp.asarray(rng.uniform(-1.0, 1.0, (n, 3)) * 4.0, jnp.float32)
    tgt = np.repeat(np.arange(n), rows)
    src = rng.integers(0, n, tgt.size)
    src = np.where(src == tgt, (src + 1) % n, src)
    return mult, cent, jnp.asarray(src, jnp.int32), jnp.asarray(tgt, jnp.int32)


@pytest.mark.parametrize("order", trim([5, 2, 3, 6]))
@pytest.mark.parametrize("k_lanes", trim([32, 8]))
def test_lanes_match_the_reference_on_ragged_rows(order, k_lanes):
    rng = np.random.default_rng(order)
    n = 40
    rows = rng.integers(0, 40, n)
    rows[3] = 0
    rows[5] = k_lanes
    rows[7] = 2 * k_lanes + 1
    mult, cent, src, tgt = _case(order, n, rows, order)
    ref = m2l_real_csr_jax(mult, cent, src, tgt, order=order)
    got = m2l_real_csr_lanes_pallas(
        mult, cent, src, tgt, order=order, k_lanes=k_lanes, interpret=True
    )
    r, g = np.asarray(ref), np.asarray(got)
    scale = np.abs(r).max()
    assert np.allclose(g, r, rtol=1e-4, atol=1e-4 * scale)
    assert np.all(g[3] == 0.0)


def test_lanes_axis_aligned_displacements_and_prefix():
    order = 4
    n = 12
    rng = np.random.default_rng(9)
    C = (order + 1) ** 2
    mult = jnp.asarray(rng.standard_normal((n, C)), jnp.float32)
    # centres on a line along z and along x: rho = 0 and dz = 0 branches
    cent = np.zeros((n, 3), np.float32)
    cent[: n // 2, 2] = np.arange(n // 2) * 1.5
    cent[n // 2 :, 0] = np.arange(n // 2) * 1.5 + 0.25
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
    ref = m2l_real_csr_jax(
        mult, cent, src_p, tgt_p, order=order, active_pair_count=live
    )
    got = m2l_real_csr_lanes_pallas(
        mult, cent, src_p, tgt_p, order=order, active_pair_count=live, interpret=True
    )
    r, g = np.asarray(ref), np.asarray(got)
    assert np.all(np.isfinite(g))
    assert np.allclose(g, r, rtol=1e-4, atol=1e-4 * np.abs(r).max())


def test_lanes_is_jittable():
    order = 5
    mult, cent, src, tgt = _case(3, 20, np.full(20, 5), order)
    f = jax.jit(
        lambda m, c: m2l_real_csr_lanes_pallas(
            m, c, src, tgt, order=order, interpret=True
        )
    )
    assert np.all(np.isfinite(np.asarray(f(mult, cent))))
