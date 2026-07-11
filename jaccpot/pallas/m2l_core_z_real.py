"""Optional Pallas kernel for batched real-basis z-axis M2L translation."""

from __future__ import annotations

import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from jaccpot.operators.real_harmonics import (
    sh_index,
    sh_size,
    translate_along_z_m2l_real,
)

try:
    import jax.experimental.pallas as pl
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None


def pallas_m2l_real_supported() -> bool:
    """Return whether the active JAX backend can run the Pallas real M2L kernel."""

    if pl is None:
        return False
    return jax.default_backend() in ("gpu", "tpu")


@lru_cache(maxsize=32)
def _m2l_translation_tables(order: int) -> tuple[np.ndarray, ...]:
    """Build static lookup tables for one fixed spherical-harmonic order."""

    p = int(order)
    coeff_count = sh_size(p)
    src_index = np.zeros((coeff_count, p + 1), dtype=np.int32)
    valid = np.zeros((coeff_count, p + 1), dtype=np.bool_)
    power = np.zeros((coeff_count, p + 1), dtype=np.int32)
    fact_index = np.zeros((coeff_count, p + 1), dtype=np.int32)
    sign = np.ones((coeff_count,), dtype=np.float64)

    # This MUST stay identical to the reference recurrence in
    # jaccpot.operators.real_harmonics.translate_along_z_m2l_real:
    #   F_n^m = sum_k  sign * (n+k)! / r^(n+k+1) * M_k^m
    # so the factorial NUMERATOR index is (n+k) while the radius exponent is
    # (n+k+1). They are distinct tables on purpose -- conflating them (using the
    # radius exponent as the factorial index) scales every term by (n+k+1).
    # tests/unit/operators/test_pallas_m2l_core_z_real.py enforces parity with
    # the pure-JAX kernel in interpret mode.
    for n in range(p + 1):
        for m in range(-n, n + 1):
            idx = sh_index(n, m)
            # (-1)^m parity times the no-sqrt2 real-basis channel factor: every
            # m != 0 channel carries an extra factor of 2.
            parity = -1.0 if (m % 2) else 1.0
            sign[idx] = parity * (2.0 if m != 0 else 1.0)
            m_abs = abs(m)
            for k in range(m_abs, p - n + 1):
                src_index[idx, k] = sh_index(k, m)
                valid[idx, k] = True
                power[idx, k] = n + k + 1
                fact_index[idx, k] = n + k

    factorial = np.asarray(
        [math.factorial(k) for k in range(2 * p + 1)],
        dtype=np.float64,
    )
    return src_index, valid, power, fact_index, sign, factorial


def _m2l_core_z_real_kernel(
    multipole_ref,
    radius_ref,
    src_index_ref,
    valid_ref,
    power_ref,
    fact_index_ref,
    sign_ref,
    factorial_ref,
    out_ref,
):
    """Compute one output coefficient for one pair."""

    coeff_idx = pl.program_id(axis=1)
    dtype = multipole_ref.dtype
    radius = jnp.maximum(
        radius_ref[0],
        jnp.asarray(1.0e-30, dtype=radius_ref.dtype),
    )
    acc0 = jnp.asarray(0.0, dtype=dtype)

    def body(slot: int, acc: Array) -> Array:
        is_valid = valid_ref[coeff_idx, slot]
        src_idx = src_index_ref[coeff_idx, slot]
        exponent = power_ref[coeff_idx, slot]
        fact_num = factorial_ref[fact_index_ref[coeff_idx, slot]]
        coeff = sign_ref[coeff_idx] * fact_num / jnp.power(radius, exponent)
        contrib = coeff.astype(dtype) * multipole_ref[0, src_idx]
        return jnp.where(is_valid, acc + contrib, acc)

    acc = lax.fori_loop(0, src_index_ref.shape[1], body, acc0)
    out_ref[0, 0] = acc


def m2l_core_z_real_pallas(
    multipole_rot: Array, radii: Array, *, order: int, interpret: bool = False
) -> Array:
    """Apply batched z-axis M2L translation with Pallas when supported.

    The kernel is only enabled on supported accelerators. On other backends,
    callers should fall back to the pure-JAX implementation. ``interpret=True``
    runs the kernel in Pallas interpret mode (works on CPU) -- used by the
    parity test to exercise the kernel logic without a GPU.
    """

    if pl is None:
        raise RuntimeError("jax.experimental.pallas is not available")

    multipole_rot = jnp.asarray(multipole_rot)
    radii = jnp.asarray(radii)
    if multipole_rot.ndim != 2:
        raise ValueError("multipole_rot must have shape (batch, coeffs)")
    if radii.ndim != 1:
        raise ValueError("radii must have shape (batch,)")

    batch_size, coeff_count = multipole_rot.shape
    if batch_size == 0:
        return jnp.zeros_like(multipole_rot)
    if coeff_count != sh_size(int(order)):
        raise ValueError("multipole_rot coefficient count does not match order")

    (
        src_index_np,
        valid_np,
        power_np,
        fact_index_np,
        sign_np,
        factorial_np,
    ) = _m2l_translation_tables(int(order))
    src_index = jnp.asarray(src_index_np, dtype=jnp.int32)
    valid = jnp.asarray(valid_np, dtype=jnp.bool_)
    power = jnp.asarray(power_np, dtype=jnp.int32)
    fact_index = jnp.asarray(fact_index_np, dtype=jnp.int32)
    sign = jnp.asarray(sign_np, dtype=multipole_rot.dtype)
    factorial = jnp.asarray(factorial_np, dtype=multipole_rot.dtype)

    kernel = pl.pallas_call(
        _m2l_core_z_real_kernel,
        out_shape=jax.ShapeDtypeStruct(multipole_rot.shape, multipole_rot.dtype),
        grid=(batch_size, coeff_count),
        in_specs=[
            pl.BlockSpec(
                (1, coeff_count), lambda batch_idx, _coeff_idx: (batch_idx, 0)
            ),
            pl.BlockSpec((1,), lambda batch_idx, _coeff_idx: (batch_idx,)),
            pl.BlockSpec(src_index.shape, lambda _batch_idx, _coeff_idx: (0, 0)),
            pl.BlockSpec(valid.shape, lambda _batch_idx, _coeff_idx: (0, 0)),
            pl.BlockSpec(power.shape, lambda _batch_idx, _coeff_idx: (0, 0)),
            pl.BlockSpec(fact_index.shape, lambda _batch_idx, _coeff_idx: (0, 0)),
            pl.BlockSpec(sign.shape, lambda _batch_idx, _coeff_idx: (0,)),
            pl.BlockSpec(factorial.shape, lambda _batch_idx, _coeff_idx: (0,)),
        ],
        out_specs=pl.BlockSpec(
            (1, 1), lambda batch_idx, coeff_idx: (batch_idx, coeff_idx)
        ),
        interpret=bool(interpret),
        name=f"m2l_core_z_real_p{int(order)}",
    )
    return kernel(
        multipole_rot, radii, src_index, valid, power, fact_index, sign, factorial
    )


__all__ = ["m2l_core_z_real_pallas", "pallas_m2l_real_supported"]
