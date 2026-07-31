"""Parity tests for the fully-fused real-basis M2L Pallas kernel.

Real analogue of ``tests/test_m2l_complex_fused_pallas.py`` for the real (Dehnen
no-sqrt2) fully-fused kernel (rotate -> z-translate -> rotate-back in one launch).
Closes a coverage gap: ``m2l_real_fused`` previously had no forward parity test.

Validates ``m2l_real_fused_pallas`` (``interpret=True``, so it runs on CPU CI and
any GPU) against:

* ``m2l_real_fused_jax`` -- the pure-jnp twin the Pallas kernel is a literal port
  of (the gradient reference for the ``custom_vjp``), and
* ``m2l_rot_scale_real_batch`` (``use_pallas=False``) -- the pure-JAX rot-scale
  M2L path the fused kernel is the runtime *accelerator* for, fed the SAME real
  rotation blocks the runtime builds (``_m2l_real_batch_kernel_fused_pallas``).
  This is the equivalence runtime dispatch relies on.

Real Pallas GPU execution needs Ampere+ (sm_80); the interpret-mode checks here
exercise the kernel arithmetic without a GPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.m2l_real_rot_scale import (
    m2l_rot_scale_real_batch,
    real_rotation_blocks_from_z_local_batch,
    real_rotation_blocks_to_z_multipole_batch,
)
from jaccpot.operators.real_harmonics import sh_size
from jaccpot.pallas.m2l_real_fused import (
    m2l_real_fused_jax,
    m2l_real_fused_pallas,
)


def _build_case(order, dtype, n=17, seed=0):
    """Well-separated real M2L batch + the SAME blocks the runtime feeds the kernel."""
    rng = np.random.default_rng(seed)
    c = sh_size(order)
    mult = jnp.asarray(rng.standard_normal((n, c)).astype(dtype))
    deltas = (rng.standard_normal((n, 3)) * 2.0).astype(dtype)
    deltas[:, 2] += 3.0  # keep |delta| well away from 0 (well-separated pairs)
    deltas = jnp.asarray(deltas)
    r = jnp.linalg.norm(deltas, axis=1)
    jdt = jnp.float64 if dtype == np.float64 else jnp.float32
    bto = real_rotation_blocks_to_z_multipole_batch(deltas, order=order, dtype=jdt)
    bfr = real_rotation_blocks_from_z_local_batch(deltas, order=order, dtype=jdt)
    ref = np.asarray(
        m2l_rot_scale_real_batch(mult, deltas, order=order, use_pallas=False)
    )
    return mult, bto, bfr, r, ref


def _relerr(a, ref):
    return float(np.linalg.norm(np.asarray(a) - ref) / (np.linalg.norm(ref) + 1e-30))


@pytest.mark.parametrize("order", [2, 3, 4])
def test_real_fused_m2l_jax_matches_rot_scale_f64(order):
    """The pure-jnp twin must equal the pure-JAX rot-scale reference."""
    if not jax.config.jax_enable_x64:
        pytest.skip("float64 disabled in this JAX runtime")
    mult, bto, bfr, r, ref = _build_case(order, np.float64)
    got = m2l_real_fused_jax(mult, bto, bfr, r, order=order)
    assert _relerr(got, ref) < 1e-10


@pytest.mark.parametrize("order", [2, 3, 4])
def test_real_fused_m2l_pallas_interpret_matches_jax_f64(order):
    """The Pallas kernel LOGIC must equal its pure-jnp twin (literal port)."""
    if not jax.config.jax_enable_x64:
        pytest.skip("float64 disabled in this JAX runtime")
    mult, bto, bfr, r, _ = _build_case(order, np.float64)
    jax_out = np.asarray(m2l_real_fused_jax(mult, bto, bfr, r, order=order))
    pallas = m2l_real_fused_pallas(mult, bto, bfr, r, order=order, interpret=True)
    assert _relerr(pallas, jax_out) < 1e-12


@pytest.mark.parametrize("order", [2, 3, 4])
def test_real_fused_m2l_pallas_interpret_matches_rot_scale_f64(order):
    """Adapter parity: fused kernel fed runtime blocks == pure-JAX rot-scale M2L."""
    if not jax.config.jax_enable_x64:
        pytest.skip("float64 disabled in this JAX runtime")
    mult, bto, bfr, r, ref = _build_case(order, np.float64)
    pallas = m2l_real_fused_pallas(mult, bto, bfr, r, order=order, interpret=True)
    assert _relerr(pallas, ref) < 1e-10


@pytest.mark.parametrize("order", [2, 3, 4])
def test_real_fused_m2l_pallas_interpret_matches_jax_f32(order):
    """fp32 accumulates rounding; loose but tight enough to catch structural errors."""
    mult, bto, bfr, r, _ = _build_case(order, np.float32)
    jax_out = np.asarray(m2l_real_fused_jax(mult, bto, bfr, r, order=order))
    pallas = m2l_real_fused_pallas(mult, bto, bfr, r, order=order, interpret=True)
    assert _relerr(pallas, jax_out) < 3e-4
