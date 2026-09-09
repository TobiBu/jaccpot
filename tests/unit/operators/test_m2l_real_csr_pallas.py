"""Parity tests for the target-tiled (CSR) real-basis M2L Pallas kernel.

``jaccpot.pallas.m2l_real_csr`` builds the rotations on chip from two angles and
sums each target's far pairs inside one program. Pinned here, in interpret mode
so it runs on CPU CI, against

* ``m2l_rot_scale_real_batch`` -- the pure-JAX rotate/scale M2L that is THE
  reference for every real M2L lane (rel err < 1e-10 at fp64, < 3e-4 at fp32,
  the tolerances of ``test_m2l_real_fused_pallas.py``), reduced per target with
  ``segment_sum``;
* the kernel's own pure-jnp twin (``m2l_real_csr_jax``), a literal port.

Cases: random CSR with empty targets and a padded ``-1`` tail, an
``active_pair_count`` shorter than the live prefix, on-axis deltas (``rho = 0``,
where the alignment azimuth is undefined and the forward must still be exact),
and orders 2..6 (``Cp`` switches 16 -> 32 -> 64 across that range).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.m2l_real_rot_scale import m2l_rot_scale_real_batch
from jaccpot.operators.real_harmonics import sh_size
from jaccpot.pallas.m2l_real_csr import (
    csr_by_target,
    m2l_real_csr_jax,
    m2l_real_csr_pair_jax,
    m2l_real_csr_pallas,
    pallas_m2l_real_csr_supported,
)


def _case(order, dtype, *, n=12, pairs=40, seed=0, on_axis=False, pad=7):
    rng = np.random.default_rng(seed)
    c = sh_size(order)
    mult = rng.standard_normal((n, c)).astype(dtype)
    centers = (rng.standard_normal((n, 3)) * 3.0).astype(dtype)
    if on_axis:
        centers[:, :2] = 0.0  # every delta along z: rho == 0
        centers[:, 2] = np.arange(n) * 2.5
    # random pairs, targets 0..n-3 so the last two targets are EMPTY
    tgt = rng.integers(0, n - 2, size=pairs).astype(np.int32)
    src = rng.integers(0, n, size=pairs).astype(np.int32)
    src = np.where(src == tgt, (src + 1) % n, src).astype(np.int32)
    if on_axis:
        src = np.where(src == tgt, (src + 2) % n, src).astype(np.int32)
    tgt_p = np.concatenate([tgt, -np.ones(pad, np.int32)])
    src_p = np.concatenate([src, -np.ones(pad, np.int32)])
    return mult, centers, src_p, tgt_p


def _reference(mult, centers, src, tgt, order, active=None):
    """pure-JAX rot-scale M2L per pair + segment_sum, on the live pairs only."""
    n = mult.shape[0]
    valid = (src >= 0) & (tgt >= 0)
    if active is not None:
        valid &= np.arange(src.shape[0]) < active
    s, t = src[valid], tgt[valid]
    deltas = centers[t] - centers[s]
    contrib = np.asarray(
        m2l_rot_scale_real_batch(jnp.asarray(mult[s]), jnp.asarray(deltas), order=order)
    )
    out = np.zeros_like(mult, dtype=np.float64)
    np.add.at(out, t, contrib.astype(np.float64))
    return out


def _relerr(a, ref):
    return float(
        np.linalg.norm(np.asarray(a, np.float64) - ref) / (np.linalg.norm(ref) + 1e-30)
    )


def test_csr_by_target_partitions_the_live_pairs():
    _, _, src, tgt = _case(3, np.float32, n=10, pairs=33, seed=2)
    n = 10
    ss, off, cnt = csr_by_target(jnp.asarray(src), jnp.asarray(tgt), total_nodes=n)
    ss, off, cnt = np.asarray(ss), np.asarray(off), np.asarray(cnt)
    live = tgt >= 0
    assert int(cnt.sum()) == int(live.sum())
    assert cnt[-2:].sum() == 0  # the two empty targets
    np.testing.assert_array_equal(off, np.cumsum(cnt) - cnt)
    for t in range(n):
        got = sorted(ss[off[t] : off[t] + cnt[t]].tolist())
        assert got == sorted(src[live & (tgt == t)].tolist())


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
def test_csr_pair_twin_matches_rot_scale_f64(order):
    """The on-chip rotation assembly equals the pure-JAX rotate/scale per pair."""
    if not jax.config.jax_enable_x64:
        pytest.skip("float64 disabled in this JAX runtime")
    rng = np.random.default_rng(order)
    c = sh_size(order)
    mult = jnp.asarray(rng.standard_normal((25, c)))
    deltas = rng.standard_normal((25, 3)) * 2.0
    deltas[:, 2] += 3.0
    deltas = jnp.asarray(deltas)
    ref = np.asarray(m2l_rot_scale_real_batch(mult, deltas, order=order))
    got = m2l_real_csr_pair_jax(mult, deltas, order=order)
    assert _relerr(got, ref) < 1e-10


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
def test_csr_pallas_interpret_matches_rot_scale_f64(order):
    if not jax.config.jax_enable_x64:
        pytest.skip("float64 disabled in this JAX runtime")
    mult, centers, src, tgt = _case(order, np.float64, seed=order)
    ref = _reference(mult, centers, src, tgt, order)
    got = m2l_real_csr_pallas(
        jnp.asarray(mult),
        jnp.asarray(centers),
        jnp.asarray(src),
        jnp.asarray(tgt),
        order=order,
        interpret=True,
    )
    assert got.shape == mult.shape
    assert _relerr(got, ref) < 1e-10
    # empty targets stay exactly zero
    assert np.all(np.asarray(got)[-2:] == 0.0)


@pytest.mark.parametrize("order", [2, 4])
def test_csr_pallas_interpret_matches_twin_and_rot_scale_f32(order):
    mult, centers, src, tgt = _case(order, np.float32, seed=10 + order)
    ref = _reference(mult, centers, src, tgt, order)
    got = m2l_real_csr_pallas(
        jnp.asarray(mult),
        jnp.asarray(centers),
        jnp.asarray(src),
        jnp.asarray(tgt),
        order=order,
        interpret=True,
    )
    twin = m2l_real_csr_jax(
        jnp.asarray(mult),
        jnp.asarray(centers),
        jnp.asarray(src),
        jnp.asarray(tgt),
        order=order,
    )
    assert _relerr(got, ref) < 3e-4
    assert _relerr(got, np.asarray(twin, np.float64)) < 1e-5


def test_csr_pallas_interpret_active_pair_count_truncates():
    if not jax.config.jax_enable_x64:
        pytest.skip("float64 disabled in this JAX runtime")
    order = 3
    mult, centers, src, tgt = _case(order, np.float64, pairs=40, seed=5)
    active = 23
    ref = _reference(mult, centers, src, tgt, order, active=active)
    got = m2l_real_csr_pallas(
        jnp.asarray(mult),
        jnp.asarray(centers),
        jnp.asarray(src),
        jnp.asarray(tgt),
        order=order,
        active_pair_count=jnp.asarray(active, jnp.int32),
        interpret=True,
    )
    assert _relerr(got, ref) < 1e-10


def test_csr_pallas_interpret_on_axis_deltas_are_exact():
    """rho == 0: atan2(0, 0) = 0 must give the exact forward, no NaN."""
    if not jax.config.jax_enable_x64:
        pytest.skip("float64 disabled in this JAX runtime")
    order = 4
    mult, centers, src, tgt = _case(order, np.float64, seed=8, on_axis=True)
    ref = _reference(mult, centers, src, tgt, order)
    got = np.asarray(
        m2l_real_csr_pallas(
            jnp.asarray(mult),
            jnp.asarray(centers),
            jnp.asarray(src),
            jnp.asarray(tgt),
            order=order,
            interpret=True,
        )
    )
    assert np.all(np.isfinite(got))
    assert _relerr(got, ref) < 1e-10


def test_csr_pallas_under_jit_with_traced_active_count():
    """The wrapper (sort + CSR + pallas_call) must trace: caps are static, counts traced."""
    order = 3
    mult, centers, src, tgt = _case(order, np.float32, seed=11)
    fn = jax.jit(
        lambda m, c, s, t, a: m2l_real_csr_pallas(
            m, c, s, t, order=order, active_pair_count=a, interpret=True
        )
    )
    got = fn(
        jnp.asarray(mult),
        jnp.asarray(centers),
        jnp.asarray(src),
        jnp.asarray(tgt),
        jnp.asarray(40, jnp.int32),
    )
    ref = _reference(mult, centers, src, tgt, order)
    assert _relerr(got, ref) < 3e-4


@pytest.mark.skipif(
    not pallas_m2l_real_csr_supported(),
    reason="CSR M2L Pallas kernel needs an Ampere+ (sm_80) GPU",
)
@pytest.mark.parametrize("order", [2, 4, 6])
def test_csr_pallas_gpu_matches_rot_scale(order):
    """The Triton lowering: dynamic-trip loop, row gathers by id, atan2, pow2 tiles."""
    mult, centers, src, tgt = _case(order, np.float32, n=40, pairs=400, seed=20 + order)
    ref = _reference(mult, centers, src, tgt, order)
    got = m2l_real_csr_pallas(
        jnp.asarray(mult),
        jnp.asarray(centers),
        jnp.asarray(src),
        jnp.asarray(tgt),
        order=order,
        interpret=False,
    )
    assert np.all(np.isfinite(np.asarray(got)))
    assert _relerr(got, ref) < 3e-4


# ---------------------------------------------------------------- axis contracts


def test_csr_pallas_rejects_a_coefficient_count_of_another_order():
    mult, centers, src, tgt = _case(3, np.float32, seed=30)
    with pytest.raises(ValueError, match="coefficients"):
        m2l_real_csr_pallas(
            jnp.asarray(mult),
            jnp.asarray(centers),
            jnp.asarray(src),
            jnp.asarray(tgt),
            order=4,
            interpret=True,
        )


def test_csr_pallas_rejects_misaligned_centers():
    mult, centers, src, tgt = _case(3, np.float32, seed=31)
    with pytest.raises(ValueError, match="centers"):
        m2l_real_csr_pallas(
            jnp.asarray(mult),
            jnp.asarray(centers[:-1]),
            jnp.asarray(src),
            jnp.asarray(tgt),
            order=3,
            interpret=True,
        )


def test_pack_unpack_centred_round_trip():
    from jaccpot.pallas.m2l_real_csr import pack_centred, unpack_centred

    for order in (2, 4, 6):
        c = sh_size(order)
        x = jnp.asarray(
            np.random.default_rng(order).standard_normal((5, c)).astype(np.float32)
        )
        rows = pack_centred(x, order=order)
        assert rows.shape[1] & (rows.shape[1] - 1) == 0  # pow2 row width
        np.testing.assert_array_equal(
            np.asarray(unpack_centred(rows, order=order)), np.asarray(x)
        )
