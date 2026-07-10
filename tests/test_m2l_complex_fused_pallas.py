"""Parity tests for the fused complex-basis M2L Pallas prototype (Phase 5).

Validates both the pure-jnp arithmetic (`m2l_complex_fused_jax`) and the Pallas
port under `interpret=True` (`m2l_complex_fused_pallas`) against the existing
reference `m2l_complex_reference_batch_cached_blocks`.

These run via Pallas `interpret=True`, so they execute on CPU (CI) and on any GPU
regardless of compute capability -- real Pallas GPU execution needs Ampere+
(sm_80), which the dev box (RTX 2080 Ti, sm_75) does not have. See
docs/phase5_pallas_plan.md.

    pytest tests/test_m2l_complex_fused_pallas.py -q
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.complex_ops import (
    complex_rotation_blocks_from_z_batch,
    complex_rotation_blocks_to_z_batch,
    m2l_complex_reference_batch_cached_blocks,
)
from jaccpot.operators.real_harmonics import sh_size
from jaccpot.pallas.m2l_complex_fused import (
    m2l_complex_fused_jax,
    m2l_complex_fused_pallas,
)


def _build_case(order, dtype, n=17, seed=0):
    rng = np.random.default_rng(seed)
    c = sh_size(order)
    cdtype = np.complex128 if dtype == np.float64 else np.complex64
    mult = (rng.standard_normal((n, c)) + 1j * rng.standard_normal((n, c))).astype(
        cdtype
    )
    deltas = (rng.standard_normal((n, 3)) * 2.0).astype(dtype)
    deltas[:, 2] += 3.0  # keep |delta| well away from 0 (well-separated pairs)
    mult = jnp.asarray(mult)
    deltas = jnp.asarray(deltas)
    r = jnp.sqrt(jnp.sum(deltas * deltas, axis=1))
    jdt = jnp.float64 if dtype == np.float64 else jnp.float32
    bto = complex_rotation_blocks_to_z_batch(
        deltas, order=order, basis="multipole", dtype=jdt
    )
    bfr = complex_rotation_blocks_from_z_batch(
        deltas, order=order, basis="local", dtype=jdt
    )
    ref = np.asarray(
        m2l_complex_reference_batch_cached_blocks(mult, deltas, bto, bfr, order=order)
    )
    return mult, bto, bfr, r, ref


def _relerr(a, ref):
    return float(np.linalg.norm(np.asarray(a) - ref) / (np.linalg.norm(ref) + 1e-30))


@pytest.mark.parametrize("order", [2, 3, 4])
def test_fused_m2l_jax_matches_reference_f64(order):
    if not jax.config.jax_enable_x64:
        pytest.skip("float64 disabled in this JAX runtime")
    mult, bto, bfr, r, ref = _build_case(order, np.float64)
    got = m2l_complex_fused_jax(mult, bto, bfr, r, order=order)
    assert _relerr(got, ref) < 1e-10


@pytest.mark.parametrize("order", [2, 3, 4])
def test_fused_m2l_pallas_interpret_matches_reference_f64(order):
    if not jax.config.jax_enable_x64:
        pytest.skip("float64 disabled in this JAX runtime")
    mult, bto, bfr, r, ref = _build_case(order, np.float64)
    got = m2l_complex_fused_pallas(mult, bto, bfr, r, order=order, interpret=True)
    assert _relerr(got, ref) < 1e-10


@pytest.mark.parametrize("order", [2, 3, 4])
def test_fused_m2l_pallas_interpret_matches_reference_f32(order):
    mult, bto, bfr, r, ref = _build_case(order, np.float32)
    got = m2l_complex_fused_pallas(mult, bto, bfr, r, order=order, interpret=True)
    # fp32 rotate -> z-core -> rotate accumulates rounding; loose but tight enough
    # to catch structural errors.
    assert _relerr(got, ref) < 3e-4
