"""Complex-basis operators (solidfmm-style reference) in JAX."""

from __future__ import annotations

import math
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from .complex_harmonics import complex_R_solidfmm
from .dtypes import complex_dtype_for_real
from .real_harmonics import (
    _compute_dehnen_B_matrix_complex,
    _rotation_to_z_angles,
    sh_offset,
    sh_size,
)
from .symmetric_tensors import (
    contract_symmetric_one_axis_3d,
    symmetric_component_count,
    symmetric_multi_indices_3d,
)

Array = jnp.ndarray


@lru_cache(maxsize=None)
def _conjugate_symmetry_metadata(
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return packed-index metadata for conjugate-symmetry projection."""

    p = int(order)
    center_idx: list[int] = []
    pos_idx: list[int] = []
    neg_idx: list[int] = []
    signs: list[float] = []
    for ell in range(p + 1):
        base = sh_offset(ell)
        center_idx.append(base + ell)
        for m in range(1, ell + 1):
            pos_idx.append(base + ell + m)
            neg_idx.append(base + ell - m)
            signs.append(-1.0 if (m % 2) else 1.0)
    return (
        np.asarray(center_idx, dtype=np.int32),
        np.asarray(pos_idx, dtype=np.int32),
        np.asarray(neg_idx, dtype=np.int32),
        np.asarray(signs, dtype=np.float64),
    )


def enforce_conjugate_symmetry(
    coeffs: Array,
    *,
    order: int,
) -> Array:
    """Project coefficients onto conjugate-symmetric form.

    Enforces C_n^{-m} = (-1)^m * conj(C_n^{m}) and Im(C_n^0)=0.
    """
    coeffs_arr = jnp.asarray(coeffs)
    return enforce_conjugate_symmetry_batch(coeffs_arr[None, :], order=order)[0]


@partial(jax.jit, static_argnames=("order",))
def enforce_conjugate_symmetry_batch(
    coeffs: Array,
    *,
    order: int,
) -> Array:
    """Batch projection onto conjugate-symmetric form."""

    coeffs_arr = jnp.asarray(coeffs)
    center_idx_np, pos_idx_np, neg_idx_np, signs_np = _conjugate_symmetry_metadata(
        int(order)
    )
    center_idx = jnp.asarray(center_idx_np, dtype=jnp.int32)
    pos_idx = jnp.asarray(pos_idx_np, dtype=jnp.int32)
    neg_idx = jnp.asarray(neg_idx_np, dtype=jnp.int32)
    real_dtype = jnp.real(jnp.zeros((), dtype=coeffs_arr.dtype)).dtype
    signs = jnp.asarray(signs_np, dtype=real_dtype).astype(coeffs_arr.dtype)

    out = coeffs_arr
    center_vals = jnp.real(out[..., center_idx]).astype(coeffs_arr.dtype)
    out = out.at[..., center_idx].set(center_vals)
    if pos_idx_np.size == 0:
        return out
    mirrored = signs * jnp.conjugate(out[..., pos_idx])
    out = out.at[..., neg_idx].set(mirrored)
    return out


def _wigner_d_small_jax(ell: int, beta: Array, *, dtype: jnp.dtype) -> Array:
    """Compute Wigner small-d matrix d^ell(beta) in JAX."""
    l = int(ell)
    beta = jnp.asarray(beta)
    rdtype = jnp.real(jnp.zeros((), dtype=dtype)).dtype

    m_vals = range(-l, l + 1)
    n = 2 * l + 1
    dmat = jnp.zeros((n, n), dtype=dtype)

    cb2 = jnp.cos(beta / 2.0).astype(rdtype)
    sb2 = jnp.sin(beta / 2.0).astype(rdtype)

    def lgamma(x: Array) -> Array:
        return jax.lax.lgamma(x)

    for i, m in enumerate(m_vals):
        for j, mp in enumerate(m_vals):
            # k range
            k_min = max(0, m - mp)
            k_max = min(l + m, l - mp)
            if k_min > k_max:
                continue

            pref = jnp.exp(
                0.5
                * (
                    lgamma(l + m + 1.0)
                    + lgamma(l - m + 1.0)
                    + lgamma(l + mp + 1.0)
                    + lgamma(l - mp + 1.0)
                )
            ).astype(rdtype)

            acc = jnp.asarray(0.0, dtype=rdtype)
            for k in range(k_min, k_max + 1):
                denom = (
                    lgamma(l + m - k + 1.0)
                    + lgamma(k + 1.0)
                    + lgamma(mp - m + k + 1.0)
                    + lgamma(l - mp - k + 1.0)
                )
                sign = (-1.0) ** k
                pow_c = 2 * l + m - mp - 2 * k
                pow_s = mp - m + 2 * k
                term = sign * jnp.exp(-denom) * (cb2**pow_c) * (sb2**pow_s)
                acc = acc + term

            dmat = dmat.at[i, j].set(pref * acc)

    return dmat.astype(dtype)


def wigner_D_complex_jax(
    ell: int,
    alpha: Array,
    beta: Array,
    gamma: Array,
    *,
    dtype: jnp.dtype,
    no_condon_shortley: bool = True,
) -> Array:
    """Compute complex Wigner D^ell(alpha,beta,gamma) in JAX."""
    l = int(ell)
    alpha = jnp.asarray(alpha)
    beta = jnp.asarray(beta)
    gamma = jnp.asarray(gamma)

    dsmall = _wigner_d_small_jax(l, beta, dtype=dtype)
    m_vals = jnp.arange(-l, l + 1, dtype=jnp.int32)
    phase_left = jnp.exp(-1j * m_vals * alpha).astype(dtype)
    phase_right = jnp.exp(-1j * m_vals * gamma).astype(dtype)
    D = jnp.diag(phase_left) @ dsmall @ jnp.diag(phase_right)

    if no_condon_shortley:
        S = jnp.diag(((-1.0) ** m_vals).astype(dtype))
        D = S @ D @ S

    return D


@lru_cache(maxsize=None)
def _factorial_table_cached_impl(max_n: int, dtype_key: str) -> np.ndarray:
    dtype = np.dtype(dtype_key)
    if max_n < 0:
        raise ValueError("max_n must be >= 0")
    if max_n == 0:
        return np.ones((1,), dtype=dtype)
    n = np.arange(1, max_n + 1, dtype=dtype)
    return np.concatenate([np.ones((1,), dtype=dtype), np.cumprod(n)])


def _factorial_table_cached(max_n: int, dtype: jnp.dtype) -> Array:
    dtype_key = str(jnp.dtype(dtype))
    return jnp.asarray(_factorial_table_cached_impl(max_n, dtype_key), dtype=dtype)


def complex_dot(
    left: Array,
    right: Array,
    *,
    order: int,
    conjugate_left: bool = True,
) -> Array:
    """Complex dot product for packed solid-harmonic coefficients.

    When `conjugate_left` is True, computes sum(conj(left) * right),
    which matches the standard complex inner product used in solidfmm.
    """
    ncoeff = sh_size(int(order))
    left = jnp.asarray(left)[:ncoeff]
    right = jnp.asarray(right)[:ncoeff]
    if conjugate_left:
        left = jnp.conjugate(left)
    return jnp.sum(left * right)


def evaluate_local_complex(
    local: Array,
    delta: Array,
    *,
    order: int,
    conjugate_left: bool = True,
) -> Array:
    """Evaluate complex local expansion at a displacement.

    Returns the real-valued potential (solidfmm normalization).
    """
    regular = complex_R_solidfmm(delta, order=order)
    pot = complex_dot(local, regular, order=order, conjugate_left=conjugate_left)
    return jnp.real(pot)


def evaluate_local_complex_with_grad(
    local: Array,
    delta: Array,
    *,
    order: int,
    conjugate_left: bool = True,
) -> tuple[Array, Array]:
    """Evaluate complex local expansion and gradient at a displacement."""
    p = int(order)

    def phi_fn(d: Array) -> Array:
        return evaluate_local_complex(local, d, order=p, conjugate_left=conjugate_left)

    potential, grad = jax.value_and_grad(phi_fn)(delta)
    return grad, potential


@partial(jax.jit, static_argnames=("order", "conjugate_left"))
def evaluate_local_complex_with_grad_batch(
    local: Array,
    deltas: Array,
    *,
    order: int,
    conjugate_left: bool = True,
) -> tuple[Array, Array]:
    """Batch evaluate complex local expansion and gradients."""
    return jax.vmap(
        lambda d: evaluate_local_complex_with_grad(
            local,
            d,
            order=order,
            conjugate_left=conjugate_left,
        )
    )(deltas)


def _lower_complex_harmonics_one_axis(
    coeffs: Array,
    *,
    order: int,
    axis: int,
) -> Array:
    """Apply one Cartesian derivative to packed complex-harmonic coefficients.

    If ``coeffs`` represents ``f_n^m`` over ``0 <= n <= order``, this returns
    coefficients representing ``∂_{axis} f_n^m`` in the same packed layout.
    """
    p = int(order)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    coeffs = jnp.asarray(coeffs)[: sh_size(p)]
    idx_a, idx_b, fac_a, fac_b = _lower_complex_harmonics_axis_maps(p, axis)
    gathered_a = coeffs[jnp.asarray(idx_a, dtype=jnp.int32)]
    gathered_b = coeffs[jnp.asarray(idx_b, dtype=jnp.int32)]
    fac_a_arr = jnp.asarray(fac_a, dtype=coeffs.dtype)
    fac_b_arr = jnp.asarray(fac_b, dtype=coeffs.dtype)
    return fac_a_arr * gathered_a + fac_b_arr * gathered_b


@lru_cache(maxsize=None)
def _lower_complex_harmonics_axis_maps(
    order: int,
    axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute gather/scale maps for one Cartesian derivative axis."""
    p = int(order)
    ncoeff = sh_size(p)
    idx_a = np.zeros((ncoeff,), dtype=np.int32)
    idx_b = np.zeros((ncoeff,), dtype=np.int32)
    fac_a = np.zeros((ncoeff,), dtype=np.complex128)
    fac_b = np.zeros((ncoeff,), dtype=np.complex128)

    def _src_index(n: int, m: int) -> tuple[int, bool]:
        if n < 0 or abs(m) > n:
            return 0, False
        return sh_offset(n) + (m + n), True

    for n in range(0, p + 1):
        for m in range(-n, n + 1):
            out_idx = sh_offset(n) + (m + n)
            if n == 0:
                continue
            if axis == 0:
                idx_a_val, valid_a = _src_index(n - 1, m - 1)
                idx_b_val, valid_b = _src_index(n - 1, m + 1)
                idx_a[out_idx] = idx_a_val
                idx_b[out_idx] = idx_b_val
                fac_a[out_idx] = 0.5 if valid_a else 0.0
                fac_b[out_idx] = -0.5 if valid_b else 0.0
            elif axis == 1:
                idx_a_val, valid_a = _src_index(n - 1, m - 1)
                idx_b_val, valid_b = _src_index(n - 1, m + 1)
                idx_a[out_idx] = idx_a_val
                idx_b[out_idx] = idx_b_val
                fac_a[out_idx] = 0.5j if valid_a else 0.0
                fac_b[out_idx] = 0.5j if valid_b else 0.0
            else:
                idx_a_val, valid_a = _src_index(n - 1, m)
                idx_a[out_idx] = idx_a_val
                idx_b[out_idx] = 0
                fac_a[out_idx] = 1.0 if valid_a else 0.0
                fac_b[out_idx] = 0.0

    return idx_a, idx_b, fac_a, fac_b


def _build_complex_harmonic_derivative_coefficients(
    delta: Array,
    *,
    order: int,
    max_derivative_order: int,
) -> tuple[Array, ...]:
    """Build packed coefficient vectors for ``D^k R`` (k=0..K)."""
    p = int(order)
    k_max = int(max_derivative_order)
    if k_max < 0:
        raise ValueError("max_derivative_order must be non-negative")

    base = jnp.asarray(complex_R_solidfmm(delta, order=p))
    levels: list[Array] = [base[jnp.newaxis, :]]
    if k_max == 0:
        return tuple(levels)

    for deriv_order in range(1, k_max + 1):
        combos = symmetric_multi_indices_3d(deriv_order)
        prev_combos = symmetric_multi_indices_3d(deriv_order - 1)
        prev = levels[-1]
        prev_index = {combo: idx for idx, combo in enumerate(prev_combos)}
        current = jnp.zeros(
            (symmetric_component_count(deriv_order, dim=3), sh_size(p)),
            dtype=base.dtype,
        )
        for idx, combo in enumerate(combos):
            if combo[0] > 0:
                parent = (combo[0] - 1, combo[1], combo[2])
                axis = 0
            elif combo[1] > 0:
                parent = (combo[0], combo[1] - 1, combo[2])
                axis = 1
            else:
                parent = (combo[0], combo[1], combo[2] - 1)
                axis = 2
            parent_coeff = prev[prev_index[parent]]
            derived = _lower_complex_harmonics_one_axis(
                parent_coeff,
                order=p,
                axis=axis,
            )
            current = current.at[idx].set(derived)
        levels.append(current)

    return tuple(levels)


def evaluate_local_complex_derivative_tower(
    local: Array,
    delta: Array,
    *,
    order: int,
    max_derivative_order: int,
    conjugate_left: bool = True,
) -> tuple[Array, ...]:
    """Evaluate potential and packed spatial derivatives ``D0..DK``.

    Notes
    -----
    This is an order-generic API scaffold for derivative towers. It uses
    autodiff internally today; hot-path contraction kernels can replace the
    internals without changing downstream code.
    """
    p = int(order)
    k_max = int(max_derivative_order)
    if k_max < 0:
        raise ValueError("max_derivative_order must be non-negative")

    local = jnp.asarray(local)[: sh_size(p)]
    deriv_coeffs = _build_complex_harmonic_derivative_coefficients(
        delta,
        order=p,
        max_derivative_order=k_max,
    )
    out: list[Array] = []
    for deriv_order, coeff_level in enumerate(deriv_coeffs):
        vals = jax.vmap(
            lambda coeff: jnp.real(
                complex_dot(local, coeff, order=p, conjugate_left=conjugate_left)
            )
        )(coeff_level)
        if deriv_order == 0:
            out.append(vals)
        else:
            out.append(vals[: symmetric_component_count(deriv_order, dim=3)])
    return tuple(out)


@partial(
    jax.jit,
    static_argnames=("order", "max_derivative_order", "conjugate_left"),
)
def evaluate_local_complex_derivative_tower_batch(
    local: Array,
    deltas: Array,
    *,
    order: int,
    max_derivative_order: int,
    conjugate_left: bool = True,
) -> tuple[Array, ...]:
    """Batch evaluate packed derivative towers for one local expansion."""
    return jax.vmap(
        lambda d: evaluate_local_complex_derivative_tower(
            local,
            d,
            order=order,
            max_derivative_order=max_derivative_order,
            conjugate_left=conjugate_left,
        )
    )(deltas)


@partial(jax.jit, static_argnames=("order",))
def contract_spatial_derivative_with_velocity(
    packed: Array,
    velocity: Array,
    *,
    order: int,
) -> Array:
    """Contract packed order-``order`` spatial derivatives with velocity."""
    return contract_symmetric_one_axis_3d(packed, velocity, order=order)


@partial(jax.jit, static_argnames=("order",))
def regular_solid_harmonic_gradient_coefficients(
    delta: Array,
    *,
    order: int,
) -> Array:
    """Return packed ``(d/dx, d/dy, d/dz)`` coefficients of ``R_n^m(delta)``."""
    p = int(order)
    base = jnp.asarray(complex_R_solidfmm(delta, order=p))
    grad_x = _lower_complex_harmonics_one_axis(base, order=p, axis=0)
    grad_y = _lower_complex_harmonics_one_axis(base, order=p, axis=1)
    grad_z = _lower_complex_harmonics_one_axis(base, order=p, axis=2)
    return jnp.stack((grad_x, grad_y, grad_z), axis=0)


@partial(jax.jit, static_argnames=("order",))
def regular_solid_harmonic_directional_derivative(
    delta: Array,
    direction: Array,
    *,
    order: int,
) -> Array:
    """Directional derivative of packed regular harmonics along ``direction``."""
    return regular_solid_harmonic_directional_derivative_order(
        delta,
        direction,
        order=order,
        derivative_order=1,
    )


@partial(jax.jit, static_argnames=("order", "derivative_order"))
def regular_solid_harmonic_directional_derivative_order(
    delta: Array,
    direction: Array,
    *,
    order: int,
    derivative_order: int,
) -> Array:
    """Order-``k`` directional derivative ``(v·∇)^k R`` in packed form."""
    p = int(order)
    k = int(derivative_order)
    if k < 0:
        raise ValueError("derivative_order must be non-negative")
    if k == 0:
        return jnp.asarray(complex_R_solidfmm(delta, order=p))

    base = jnp.asarray(complex_R_solidfmm(delta, order=p))
    direction_arr = jnp.asarray(direction, dtype=jnp.real(base).dtype)

    def body(_i: int, coeffs: Array) -> Array:
        dx = _lower_complex_harmonics_one_axis(coeffs, order=p, axis=0)
        dy = _lower_complex_harmonics_one_axis(coeffs, order=p, axis=1)
        dz = _lower_complex_harmonics_one_axis(coeffs, order=p, axis=2)
        return direction_arr[0] * dx + direction_arr[1] * dy + direction_arr[2] * dz

    return jax.lax.fori_loop(0, k, body, base)


@partial(jax.jit, static_argnames=("order",))
def regular_solid_harmonic_directional_derivative_batch(
    deltas: Array,
    directions: Array,
    *,
    order: int,
) -> Array:
    """Batch directional derivatives of packed regular harmonics."""
    return jax.vmap(
        lambda d, v: regular_solid_harmonic_directional_derivative_order(
            d,
            v,
            order=order,
            derivative_order=1,
        ),
        in_axes=(0, 0),
        out_axes=0,
    )(deltas, directions)


@partial(jax.jit, static_argnames=("order", "derivative_order"))
def regular_solid_harmonic_directional_derivative_order_batch(
    deltas: Array,
    directions: Array,
    *,
    order: int,
    derivative_order: int,
) -> Array:
    """Batch order-``k`` directional derivatives of packed regular harmonics."""
    return jax.vmap(
        lambda d, v: regular_solid_harmonic_directional_derivative_order(
            d,
            v,
            order=order,
            derivative_order=derivative_order,
        ),
        in_axes=(0, 0),
        out_axes=0,
    )(deltas, directions)


def translate_along_z_m2l_complex(
    multipole: Array,
    r: Array,
    *,
    order: int,
) -> Array:
    """Translate complex multipole to local along +z (Dehnen series)."""
    p = int(order)
    multipole = jnp.asarray(multipole)
    r = jnp.asarray(r).reshape(())
    dtype = multipole.real.dtype

    ncoeff = sh_size(p)
    out = jnp.zeros((ncoeff,), dtype=multipole.dtype)
    fact = _factorial_table_cached(2 * p, dtype)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            m_abs = abs(m)
            acc = jnp.asarray(0.0 + 0.0j, dtype=multipole.dtype)
            for k in range(m_abs, p - n + 1):
                src_idx = sh_offset(k) + (m + k)
                coeff = ((-1.0) ** m) * fact[n + k] / (r ** (n + k + 1))
                acc = acc + coeff * multipole[src_idx]
            out = out.at[sh_offset(n) + (m + n)].set(acc)

    return out


def translate_along_z_m2m_complex(
    multipole: Array,
    dz: Array,
    *,
    order: int,
) -> Array:
    """Translate complex multipole along +z (Dehnen series)."""
    p = int(order)
    multipole = jnp.asarray(multipole)
    dz = jnp.asarray(dz).reshape(())
    dtype = multipole.real.dtype

    ncoeff = sh_size(p)
    out = jnp.zeros((ncoeff,), dtype=multipole.dtype)
    fact = _factorial_table_cached(p, dtype)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            m_abs = abs(m)
            acc = jnp.asarray(0.0 + 0.0j, dtype=multipole.dtype)
            for k in range(0, n - m_abs + 1):
                src_n = n - k
                if m_abs > src_n:
                    continue
                src_idx = sh_offset(src_n) + (m + src_n)
                coeff = (dz**k) / fact[k]
                acc = acc + coeff * multipole[src_idx]
            out = out.at[sh_offset(n) + (m + n)].set(acc)

    return out


def translate_along_z_m2m_complex_solidfmm(
    multipole: Array,
    dz: Array,
    *,
    order: int,
) -> Array:
    """Translate complex multipole along +z (solidfmm zm2m)."""
    p = int(order)
    multipole = jnp.asarray(multipole)
    dz = jnp.asarray(dz).reshape(())
    dtype = multipole.real.dtype

    ncoeff = sh_size(p)
    out = jnp.zeros((ncoeff,), dtype=multipole.dtype)
    fact = _factorial_table_cached(p, dtype)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            m_abs = abs(m)
            acc = jnp.asarray(0.0 + 0.0j, dtype=multipole.dtype)
            for k in range(0, n - m_abs + 1):
                src_n = n - k
                if m_abs > src_n:
                    continue
                src_idx = sh_offset(src_n) + (m + src_n)
                coeff = (dz**k) / fact[k]
                acc = acc + coeff * multipole[src_idx]
            out = out.at[sh_offset(n) + (m + n)].set(acc)

    return out


def translate_along_z_l2l_complex(
    local: Array,
    dz: Array,
    *,
    order: int,
) -> Array:
    """Translate complex local expansion along +z (Dehnen series)."""
    p = int(order)
    local = jnp.asarray(local)
    dz = jnp.asarray(dz).reshape(())
    dtype = local.real.dtype

    ncoeff = sh_size(p)
    out = jnp.zeros((ncoeff,), dtype=local.dtype)
    fact = _factorial_table_cached(p + 1, dtype)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            acc = jnp.asarray(0.0 + 0.0j, dtype=local.dtype)
            for k in range(0, p - n + 1):
                src_n = n + k
                if src_n > p:
                    continue
                src_idx = sh_offset(src_n) + (m + src_n)
                coeff = (dz**k) / fact[k]
                acc = acc + coeff * local[src_idx]
            out = out.at[sh_offset(n) + (m + n)].set(acc)

    return out


def _complex_Dz(ell: int, angle: Array, *, dtype: jnp.dtype) -> Array:
    m_vals = jnp.arange(-ell, ell + 1, dtype=dtype)
    diag = jnp.exp(1j * m_vals * angle)
    return jnp.diag(diag)


@lru_cache(maxsize=None)
def _complex_swap_matrices_cached(
    ell: int, dtype_key: str
) -> tuple[np.ndarray, np.ndarray]:
    B = _compute_dehnen_B_matrix_complex(ell, dtype_key)
    return B, B.T


def _complex_swap_matrices(ell: int, *, dtype: jnp.dtype) -> tuple[Array, Array]:
    dtype_key = str(jnp.dtype(dtype))
    B, Bt = _complex_swap_matrices_cached(ell, dtype_key)
    return jnp.asarray(B, dtype=dtype), jnp.asarray(Bt, dtype=dtype)


def _solidfmm_pack_m_nonneg(block: Array, *, ell: int) -> tuple[Array, Array]:
    """Extract m>=0 coefficients as (re, im) arrays.

    Used for solidfmm-style swap/rotscale operations.
    """
    block = jnp.asarray(block)
    start = ell
    re = jnp.real(block[start:])
    im = jnp.imag(block[start:])
    return re, im


def _solidfmm_unpack_m_nonneg(re: Array, im: Array, *, ell: int) -> Array:
    """Reconstruct full m in [-ell, ell] block from m>=0 real/imag arrays."""
    re = jnp.asarray(re)
    im = jnp.asarray(im)
    dtype = complex_dtype_for_real(jnp.result_type(re, im))
    block = jnp.zeros((2 * ell + 1,), dtype=dtype)

    m_vals = jnp.arange(0, ell + 1)
    pos = ell + m_vals
    block = block.at[pos].set(re + 1j * im)

    neg_m = jnp.arange(1, ell + 1)
    neg_pos = ell - neg_m
    pos_m = ell + neg_m
    signs = (-1.0) ** neg_m
    block = block.at[neg_pos].set(signs * jnp.conjugate(block[pos_m]))
    return block


def _solidfmm_swap_mats(
    B_swap: Array,
    *,
    ell: int,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Build real/imag swap matrices for solidfmm's m>=0 storage.

    These implement the real-linear map induced by B on coefficients with
    conjugate symmetry.
    """
    m_vals = jnp.arange(0, ell + 1)
    l_vals = jnp.arange(0, ell + 1)
    row_idx = ell + m_vals[:, None]
    col_pos = ell + l_vals[None, :]
    col_neg = ell - l_vals[None, :]

    B = jnp.asarray(B_swap, dtype=dtype)
    B_pos = B[row_idx, col_pos]
    B_neg = B[row_idx, col_neg]

    signs = (-1.0) ** l_vals
    real_mat = B_pos + signs * B_neg
    imag_mat = B_pos - signs * B_neg

    real_mat = real_mat.at[:, 0].set(B_pos[:, 0])
    imag_mat = imag_mat.at[:, 0].set(jnp.zeros((ell + 1,), dtype=dtype))
    return real_mat, imag_mat


def _solidfmm_swap_apply(
    re: Array,
    im: Array,
    B_swap: Array,
    *,
    ell: int,
) -> tuple[Array, Array]:
    """Apply solidfmm-style swap to m>=0 real/imag arrays."""
    dtype = jnp.result_type(re, im)
    real_mat, imag_mat = _solidfmm_swap_mats(B_swap, ell=ell, dtype=dtype)
    re_out = real_mat @ re
    im_out = imag_mat @ im
    return re_out, im_out


def _solidfmm_rotscale(
    re: Array,
    im: Array,
    *,
    angle: Array,
    scale: Array,
    ell: int,
    forward: bool,
) -> tuple[Array, Array]:
    """Solidfmm rotscale for m>=0 coefficients."""
    m_vals = jnp.arange(0, ell + 1, dtype=jnp.result_type(re, im, angle))
    cos_m = jnp.cos(m_vals * angle)
    sin_m = jnp.sin(m_vals * angle)
    scale = jnp.asarray(scale)

    if forward:
        re_out = scale * (cos_m * re - sin_m * im)
        im_out = scale * (sin_m * re + cos_m * im)
    else:
        re_out = scale * (cos_m * re + sin_m * im)
        im_out = scale * (-sin_m * re + cos_m * im)
    return re_out, im_out


def _angles_from_delta(delta: Array) -> tuple[Array, Array]:
    x, y, z = delta[0], delta[1], delta[2]
    rho = jnp.sqrt(x * x + y * y)
    alpha = jnp.arctan2(y, x)
    beta = jnp.arctan2(rho, z)
    return alpha, beta


def _angles_from_delta_solidfmm(delta: Array) -> tuple[Array, Array]:
    """Angles matching solidfmm's euler() convention.

    solidfmm defines:
        cos(alpha)=y/rxy, sin(alpha)=x/rxy
        cos(beta)=z/r, sin(beta)=-rxy/r
    so alpha=atan2(x,y), beta=atan2(-rxy,z).
    """
    x, y, z = delta[0], delta[1], delta[2]
    rho = jnp.sqrt(x * x + y * y)
    alpha = jnp.arctan2(x, y)
    beta = jnp.arctan2(-rho, z)
    return alpha, beta


def _rotate_multipole_to_z(block_complex: Array, delta: Array, ell: int) -> Array:
    alpha, beta = _angles_from_delta(delta)
    B_T, B_U = _complex_swap_matrices(ell, dtype=block_complex.dtype)
    Dz_alpha = _complex_Dz(ell, -alpha, dtype=block_complex.dtype)
    Dz_beta = _complex_Dz(ell, -beta, dtype=block_complex.dtype)
    D = B_U @ Dz_beta @ B_U @ Dz_alpha
    return D @ block_complex


def _rotate_local_from_z(block_complex: Array, delta: Array, ell: int) -> Array:
    alpha, beta = _angles_from_delta(delta)
    B_T, _ = _complex_swap_matrices(ell, dtype=block_complex.dtype)
    Dz_alpha = _complex_Dz(ell, alpha, dtype=block_complex.dtype)
    Dz_beta = _complex_Dz(ell, beta, dtype=block_complex.dtype)
    D = Dz_alpha @ B_T @ Dz_beta @ B_T
    return D @ block_complex


@partial(jax.jit, static_argnames=("order",))
def rotate_complex_multipole_to_z(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate packed complex multipoles into the z-aligned frame."""
    p = int(order)
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)

    blocks = _complex_rotation_blocks_to_z(
        delta, order=p, basis="multipole", dtype=multipole.dtype
    )
    return _apply_complex_rotation_blocks_batched(multipole, blocks, order=p)


@partial(jax.jit, static_argnames=("order",))
def rotate_complex_multipole_from_z(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate packed complex multipoles back from the z-aligned frame."""
    p = int(order)
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)

    blocks = _complex_rotation_blocks_from_z(
        delta, order=p, basis="multipole", dtype=multipole.dtype
    )
    return _apply_complex_rotation_blocks_batched(multipole, blocks, order=p)


@partial(jax.jit, static_argnames=("order",))
def rotate_complex_local_from_z(
    local: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate packed complex locals back from the z-aligned frame."""
    p = int(order)
    local = jnp.asarray(local)
    delta = jnp.asarray(delta)

    blocks = _complex_rotation_blocks_from_z(
        delta, order=p, basis="local", dtype=local.dtype
    )
    return _apply_complex_rotation_blocks_batched(local, blocks, order=p)


@partial(jax.jit, static_argnames=("order",))
def rotate_complex_local_to_z(
    local: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate packed complex locals into the z-aligned frame."""
    p = int(order)
    local = jnp.asarray(local)
    delta = jnp.asarray(delta)

    blocks = _complex_rotation_blocks_to_z(
        delta, order=p, basis="local", dtype=local.dtype
    )
    return _apply_complex_rotation_blocks_batched(local, blocks, order=p)


def _complex_rotation_blocks_to_z(
    delta: Array,
    *,
    order: int,
    basis: str,
    dtype: jnp.dtype,
) -> tuple[Array, ...]:
    """Precompute complex rotation blocks mapping to the z-aligned frame."""
    if basis not in ("multipole", "local"):
        raise ValueError("basis must be 'multipole' or 'local'")
    p = int(order)
    delta = jnp.asarray(delta)
    alpha, beta = _angles_from_delta(delta)

    blocks = []
    for ell in range(p + 1):
        B_T, B_U = _complex_swap_matrices(ell, dtype=dtype)
        Dz_alpha = _complex_Dz(ell, -alpha, dtype=dtype)
        Dz_beta = _complex_Dz(ell, -beta, dtype=dtype)
        if basis == "multipole":
            D = B_U @ Dz_beta @ B_U @ Dz_alpha
        else:
            D = B_T @ Dz_beta @ B_T @ Dz_alpha
        blocks.append(D)
    return tuple(blocks)


def _complex_rotation_blocks_to_z_solidfmm(
    delta: Array,
    *,
    order: int,
    basis: str,
    dtype: jnp.dtype,
) -> tuple[Array, ...]:
    """Rotation blocks to z using solidfmm's swap+z-rotation convention."""
    if basis not in ("multipole", "local"):
        raise ValueError("basis must be 'multipole' or 'local'")
    p = int(order)
    delta = jnp.asarray(delta)
    alpha, beta = _angles_from_delta_solidfmm(delta)

    blocks = []
    for ell in range(p + 1):
        B_T, B_U = _complex_swap_matrices(ell, dtype=dtype)
        Dz_alpha = _complex_Dz(ell, alpha, dtype=dtype)
        Dz_beta = _complex_Dz(ell, beta, dtype=dtype)
        if basis == "multipole":
            D = B_U @ Dz_beta @ B_U @ Dz_alpha
        else:
            D = B_T @ Dz_beta @ B_T @ Dz_alpha
        blocks.append(D)
    return tuple(blocks)


def _complex_rotation_blocks_from_z(
    delta: Array,
    *,
    order: int,
    basis: str,
    dtype: jnp.dtype,
) -> tuple[Array, ...]:
    """Precompute complex rotation blocks mapping from the z-aligned frame."""
    if basis not in ("multipole", "local"):
        raise ValueError("basis must be 'multipole' or 'local'")
    p = int(order)
    delta = jnp.asarray(delta)
    alpha, beta = _angles_from_delta(delta)

    blocks = []
    for ell in range(p + 1):
        B_T, B_U = _complex_swap_matrices(ell, dtype=dtype)
        Dz_alpha = _complex_Dz(ell, alpha, dtype=dtype)
        Dz_beta = _complex_Dz(ell, beta, dtype=dtype)
        if basis == "multipole":
            D = Dz_alpha @ B_U @ Dz_beta @ B_U
        else:
            D = Dz_alpha @ B_T @ Dz_beta @ B_T
        blocks.append(D)
    return tuple(blocks)


def _complex_rotation_blocks_from_z_solidfmm(
    delta: Array,
    *,
    order: int,
    basis: str,
    dtype: jnp.dtype,
) -> tuple[Array, ...]:
    """Rotation blocks from z using solidfmm's swap+z-rotation convention."""
    if basis not in ("multipole", "local"):
        raise ValueError("basis must be 'multipole' or 'local'")
    p = int(order)
    delta = jnp.asarray(delta)
    alpha, beta = _angles_from_delta_solidfmm(delta)

    blocks = []
    for ell in range(p + 1):
        B_T, B_U = _complex_swap_matrices(ell, dtype=dtype)
        Dz_alpha = _complex_Dz(ell, -alpha, dtype=dtype)
        Dz_beta = _complex_Dz(ell, -beta, dtype=dtype)
        if basis == "multipole":
            D = Dz_alpha @ B_U @ Dz_beta @ B_U
        else:
            D = Dz_alpha @ B_T @ Dz_beta @ B_T
        blocks.append(D)
    return tuple(blocks)


def _pack_coeffs_by_ell(
    coeffs: Array,
    *,
    order: int,
) -> Array:
    """Pack coefficients into (p+1, 2p+1) array with zero padding."""
    p = int(order)
    coeffs = jnp.asarray(coeffs)
    max_m = 2 * p + 1
    out = jnp.zeros((p + 1, max_m), dtype=coeffs.dtype)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        out = out.at[ell, : 2 * ell + 1].set(coeffs[sl])
    return out


def _unpack_coeffs_by_ell(
    packed: Array,
    *,
    order: int,
) -> Array:
    """Unpack (p+1, 2p+1) coefficients back into packed layout."""
    p = int(order)
    dtype = jnp.asarray(packed).dtype
    out = jnp.zeros((sh_size(p),), dtype=dtype)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        out = out.at[sl].set(packed[ell, : 2 * ell + 1])
    return out


def _blocks_to_padded_array(
    blocks: tuple[Array, ...],
    *,
    order: int,
    dtype: jnp.dtype,
) -> Array:
    """Pad rotation blocks to (p+1, 2p+1, 2p+1)."""
    p = int(order)
    max_m = 2 * p + 1
    out = jnp.zeros((p + 1, max_m, max_m), dtype=dtype)
    for ell in range(p + 1):
        size = 2 * ell + 1
        out = out.at[ell, :size, :size].set(blocks[ell])
    return out


def _complex_rotation_blocks_to_z_padded(
    delta: Array,
    *,
    order: int,
    basis: str,
    dtype: jnp.dtype,
) -> Array:
    blocks = _complex_rotation_blocks_to_z(delta, order=order, basis=basis, dtype=dtype)
    return _blocks_to_padded_array(blocks, order=order, dtype=dtype)


def _complex_rotation_blocks_from_z_padded(
    delta: Array,
    *,
    order: int,
    basis: str,
    dtype: jnp.dtype,
) -> Array:
    blocks = _complex_rotation_blocks_from_z(
        delta, order=order, basis=basis, dtype=dtype
    )
    return _blocks_to_padded_array(blocks, order=order, dtype=dtype)


def _complex_rotation_blocks_to_z_solidfmm_padded(
    delta: Array,
    *,
    order: int,
    basis: str,
    dtype: jnp.dtype,
) -> Array:
    blocks = _complex_rotation_blocks_to_z_solidfmm(
        delta,
        order=order,
        basis=basis,
        dtype=dtype,
    )
    return _blocks_to_padded_array(blocks, order=order, dtype=dtype)


def _complex_rotation_blocks_from_z_solidfmm_padded(
    delta: Array,
    *,
    order: int,
    basis: str,
    dtype: jnp.dtype,
) -> Array:
    blocks = _complex_rotation_blocks_from_z_solidfmm(
        delta,
        order=order,
        basis=basis,
        dtype=dtype,
    )
    return _blocks_to_padded_array(blocks, order=order, dtype=dtype)


@partial(jax.jit, static_argnames=("order", "basis", "dtype"))
def complex_rotation_blocks_to_z_batch(
    deltas: Array,
    *,
    order: int,
    basis: str,
    dtype: jnp.dtype,
) -> Array:
    """Batch padded rotation blocks to z-aligned frame."""
    return jax.vmap(
        lambda d: _complex_rotation_blocks_to_z_padded(
            d,
            order=order,
            basis=basis,
            dtype=dtype,
        )
    )(deltas)


@partial(jax.jit, static_argnames=("order", "basis", "dtype"))
def complex_rotation_blocks_to_z_solidfmm_batch(
    deltas: Array,
    *,
    order: int,
    basis: str,
    dtype: jnp.dtype,
) -> Array:
    """Batch padded rotation blocks to z using solidfmm convention."""
    return jax.vmap(
        lambda d: _complex_rotation_blocks_to_z_solidfmm_padded(
            d,
            order=order,
            basis=basis,
            dtype=dtype,
        )
    )(deltas)


@partial(jax.jit, static_argnames=("order", "basis", "dtype"))
def complex_rotation_blocks_from_z_batch(
    deltas: Array,
    *,
    order: int,
    basis: str,
    dtype: jnp.dtype,
) -> Array:
    """Batch padded rotation blocks from z-aligned frame."""
    return jax.vmap(
        lambda d: _complex_rotation_blocks_from_z_padded(
            d,
            order=order,
            basis=basis,
            dtype=dtype,
        )
    )(deltas)


@partial(jax.jit, static_argnames=("order", "basis", "dtype"))
def complex_rotation_blocks_from_z_solidfmm_batch(
    deltas: Array,
    *,
    order: int,
    basis: str,
    dtype: jnp.dtype,
) -> Array:
    """Batch padded rotation blocks from z using solidfmm convention."""
    return jax.vmap(
        lambda d: _complex_rotation_blocks_from_z_solidfmm_padded(
            d,
            order=order,
            basis=basis,
            dtype=dtype,
        )
    )(deltas)


def _apply_complex_rotation_blocks_batched(
    coeffs: Array,
    blocks: tuple[Array, ...],
    *,
    order: int,
) -> Array:
    """Apply rotation blocks using per-ell batched matvecs."""
    p = int(order)
    coeffs = jnp.asarray(coeffs)
    dtype = coeffs.dtype
    blocks_array = _blocks_to_padded_array(blocks, order=p, dtype=dtype)
    packed = _pack_coeffs_by_ell(coeffs, order=p)
    rotated = jnp.einsum("bij,bj->bi", blocks_array, packed)
    return _unpack_coeffs_by_ell(rotated, order=p)


@partial(jax.jit, static_argnames=("order",))
def _apply_complex_rotation_blocks_padded_batch(
    coeffs: Array,
    blocks_array: Array,
    *,
    order: int,
) -> Array:
    """Apply padded rotation blocks to a batch of coefficients."""
    packed = jax.vmap(lambda c: _pack_coeffs_by_ell(c, order=order))(coeffs)
    rotated = jnp.einsum("nbij,nbj->nbi", blocks_array, packed)
    return jax.vmap(lambda c: _unpack_coeffs_by_ell(c, order=order))(rotated)


def rotate_complex_multipole_to_z_cached(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate packed complex multipoles using precomputed blocks."""
    blocks = _complex_rotation_blocks_to_z(
        delta, order=order, basis="multipole", dtype=jnp.asarray(multipole).dtype
    )
    return _apply_complex_rotation_blocks_batched(multipole, blocks, order=order)


def rotate_complex_multipole_to_z_solidfmm(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate multipoles to z using solidfmm's swap+z-rotation convention."""
    blocks = _complex_rotation_blocks_to_z_solidfmm(
        delta, order=order, basis="multipole", dtype=jnp.asarray(multipole).dtype
    )
    return _apply_complex_rotation_blocks_batched(multipole, blocks, order=order)


def rotate_complex_multipole_from_z_cached(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate packed complex multipoles back using precomputed blocks."""
    blocks = _complex_rotation_blocks_from_z(
        delta, order=order, basis="multipole", dtype=jnp.asarray(multipole).dtype
    )
    return _apply_complex_rotation_blocks_batched(multipole, blocks, order=order)


def rotate_complex_multipole_from_z_solidfmm(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate multipoles from z using solidfmm's swap+z-rotation convention."""
    blocks = _complex_rotation_blocks_from_z_solidfmm(
        delta, order=order, basis="multipole", dtype=jnp.asarray(multipole).dtype
    )
    return _apply_complex_rotation_blocks_batched(multipole, blocks, order=order)


def rotate_complex_local_to_z_cached(
    local: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate packed complex locals using precomputed blocks."""
    blocks = _complex_rotation_blocks_to_z(
        delta, order=order, basis="local", dtype=jnp.asarray(local).dtype
    )
    return _apply_complex_rotation_blocks_batched(local, blocks, order=order)


def rotate_complex_local_to_z_solidfmm(
    local: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate locals to z using solidfmm's swap+z-rotation convention."""
    blocks = _complex_rotation_blocks_to_z_solidfmm(
        delta, order=order, basis="local", dtype=jnp.asarray(local).dtype
    )
    return _apply_complex_rotation_blocks_batched(local, blocks, order=order)


def rotate_complex_local_from_z_cached(
    local: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate packed complex locals back using precomputed blocks."""
    blocks = _complex_rotation_blocks_from_z(
        delta, order=order, basis="local", dtype=jnp.asarray(local).dtype
    )
    return _apply_complex_rotation_blocks_batched(local, blocks, order=order)


def rotate_complex_local_from_z_solidfmm(
    local: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate locals from z using solidfmm's swap+z-rotation convention."""
    blocks = _complex_rotation_blocks_from_z_solidfmm(
        delta, order=order, basis="local", dtype=jnp.asarray(local).dtype
    )
    return _apply_complex_rotation_blocks_batched(local, blocks, order=order)


def rotate_complex_multipole_to_z_wigner(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate multipoles to z-axis using Wigner D matrices (JAX)."""
    p = int(order)
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)
    alpha, beta, gamma = _rotation_to_z_angles(delta[0], delta[1], delta[2])

    out = jnp.zeros_like(multipole)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D = wigner_D_complex_jax(
            ell,
            alpha,
            beta,
            gamma,
            dtype=multipole.dtype,
            no_condon_shortley=True,
        )
        out = out.at[sl].set(D @ multipole[sl])
    return out


def rotate_complex_multipole_from_z_wigner(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate multipoles back from z-axis using Wigner D matrices (JAX)."""
    p = int(order)
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)
    alpha, beta, gamma = _rotation_to_z_angles(delta[0], delta[1], delta[2])

    out = jnp.zeros_like(multipole)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D = wigner_D_complex_jax(
            ell,
            -gamma,
            -beta,
            -alpha,
            dtype=multipole.dtype,
            no_condon_shortley=True,
        )
        out = out.at[sl].set(D @ multipole[sl])
    return out


def rotate_complex_local_to_z_wigner(
    local: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate locals to z-axis using Wigner D matrices (JAX)."""
    p = int(order)
    local = jnp.asarray(local)
    delta = jnp.asarray(delta)
    alpha, beta, gamma = _rotation_to_z_angles(delta[0], delta[1], delta[2])

    out = jnp.zeros_like(local)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D = wigner_D_complex_jax(
            ell,
            alpha,
            beta,
            gamma,
            dtype=local.dtype,
            no_condon_shortley=True,
        )
        out = out.at[sl].set(D @ local[sl])
    return out


def rotate_complex_local_from_z_wigner(
    local: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Rotate locals back from z-axis using Wigner D matrices (JAX)."""
    p = int(order)
    local = jnp.asarray(local)
    delta = jnp.asarray(delta)
    alpha, beta, gamma = _rotation_to_z_angles(delta[0], delta[1], delta[2])

    out = jnp.zeros_like(local)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        D = wigner_D_complex_jax(
            ell,
            -gamma,
            -beta,
            -alpha,
            dtype=local.dtype,
            no_condon_shortley=True,
        )
        out = out.at[sl].set(D @ local[sl])
    return out


@partial(jax.jit, static_argnames=("order",))
def rotate_complex_multipole_to_z_batch(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
) -> Array:
    """Batch rotate complex multipoles into z-aligned frame."""
    return jax.vmap(
        lambda m, d: rotate_complex_multipole_to_z_cached(m, d, order=order),
        in_axes=(0, 0),
        out_axes=0,
    )(multipoles, deltas)


@partial(jax.jit, static_argnames=("order",))
def rotate_complex_multipole_from_z_batch(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
) -> Array:
    """Batch rotate complex multipoles back from z-aligned frame."""
    return jax.vmap(
        lambda m, d: rotate_complex_multipole_from_z_cached(m, d, order=order),
        in_axes=(0, 0),
        out_axes=0,
    )(multipoles, deltas)


@partial(jax.jit, static_argnames=("order",))
def rotate_complex_local_to_z_batch(
    locals: Array,
    deltas: Array,
    *,
    order: int,
) -> Array:
    """Batch rotate complex locals into z-aligned frame."""
    return jax.vmap(
        lambda m, d: rotate_complex_local_to_z_cached(m, d, order=order),
        in_axes=(0, 0),
        out_axes=0,
    )(locals, deltas)


@partial(jax.jit, static_argnames=("order",))
def rotate_complex_local_from_z_batch(
    locals: Array,
    deltas: Array,
    *,
    order: int,
) -> Array:
    """Batch rotate complex locals back from z-aligned frame."""
    return jax.vmap(
        lambda m, d: rotate_complex_local_from_z_cached(m, d, order=order),
        in_axes=(0, 0),
        out_axes=0,
    )(locals, deltas)


@partial(jax.jit, static_argnames=("order", "rotation"))
def m2m_complex(
    multipole: Array,
    delta: Array,
    *,
    order: int,
    rotation: str = "bdz",
) -> Array:
    """Complex M2M using A6: rotate → z-translate → rotate back."""
    p = int(order)
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)
    delta_m2m = -delta
    if rotation == "solidfmm":
        delta_m2m = delta

    r = jnp.sqrt(jnp.maximum(jnp.dot(delta, delta), 1e-60))
    if rotation == "wigner":
        M_rot = rotate_complex_multipole_to_z_wigner(multipole, delta_m2m, order=p)
    elif rotation == "cached":
        M_rot = rotate_complex_multipole_to_z_cached(multipole, delta_m2m, order=p)
    elif rotation == "solidfmm":
        M_rot = rotate_complex_multipole_to_z_solidfmm(multipole, delta_m2m, order=p)
    elif rotation == "bdz":
        M_rot = rotate_complex_multipole_to_z(multipole, delta_m2m, order=p)
    else:
        raise ValueError("rotation must be 'bdz', 'cached', 'solidfmm', or 'wigner'")
    if rotation == "solidfmm":
        M_z = translate_along_z_m2m_complex_solidfmm(M_rot, r, order=p)
    else:
        M_z = translate_along_z_m2m_complex(M_rot, r, order=p)
    if rotation == "wigner":
        return rotate_complex_multipole_from_z_wigner(M_z, delta_m2m, order=p)
    if rotation == "cached":
        return rotate_complex_multipole_from_z_cached(M_z, delta_m2m, order=p)
    if rotation == "solidfmm":
        return rotate_complex_multipole_from_z_solidfmm(M_z, delta_m2m, order=p)
    return rotate_complex_multipole_from_z(M_z, delta_m2m, order=p)


@partial(jax.jit, static_argnames=("order", "rotation"))
def l2l_complex(
    local: Array,
    delta: Array,
    *,
    order: int,
    rotation: str = "bdz",
) -> Array:
    """Complex L2L using A6: rotate → z-translate → rotate back."""
    p = int(order)
    local = jnp.asarray(local)
    delta = jnp.asarray(delta)

    r = jnp.sqrt(jnp.maximum(jnp.dot(delta, delta), 1e-60))
    if rotation == "wigner":
        L_rot = rotate_complex_local_to_z_wigner(local, delta, order=p)
    elif rotation == "cached":
        L_rot = rotate_complex_local_to_z_cached(local, delta, order=p)
    elif rotation == "solidfmm":
        L_rot = rotate_complex_local_to_z_solidfmm(local, delta, order=p)
    elif rotation == "bdz":
        L_rot = rotate_complex_local_to_z(local, delta, order=p)
    else:
        raise ValueError("rotation must be 'bdz', 'cached', 'solidfmm', or 'wigner'")
    L_z = translate_along_z_l2l_complex(L_rot, r, order=p)
    if rotation == "wigner":
        return rotate_complex_local_from_z_wigner(L_z, delta, order=p)
    if rotation == "cached":
        return rotate_complex_local_from_z_cached(L_z, delta, order=p)
    if rotation == "solidfmm":
        return rotate_complex_local_from_z_solidfmm(L_z, delta, order=p)
    return rotate_complex_local_from_z(L_z, delta, order=p)


def m2l_complex_reference(
    multipole: Array,
    delta: Array,
    *,
    order: int,
    rotation: str = "bdz",
) -> Array:
    """Reference M2L in complex basis (rotate → z-translate → rotate back)."""
    p = int(order)
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)

    ncoeff = sh_size(p)
    multipole = multipole[:ncoeff]

    if rotation == "wigner":
        M_rotated = rotate_complex_multipole_to_z_wigner(multipole, delta, order=p)
    elif rotation == "cached":
        M_rotated = rotate_complex_multipole_to_z_cached(multipole, delta, order=p)
    elif rotation == "solidfmm":
        M_rotated = rotate_complex_multipole_to_z_solidfmm(multipole, delta, order=p)
    elif rotation == "bdz":
        M_rotated = rotate_complex_multipole_to_z(multipole, delta, order=p)
    else:
        raise ValueError("rotation must be 'bdz', 'cached', 'solidfmm', or 'wigner'")

    r = jnp.sqrt(jnp.maximum(jnp.dot(delta, delta), 1e-60))
    local_z = translate_along_z_m2l_complex(M_rotated, r, order=p)

    if rotation == "wigner":
        return rotate_complex_local_from_z_wigner(local_z, delta, order=p)
    if rotation == "cached":
        return rotate_complex_local_from_z_cached(local_z, delta, order=p)
    if rotation == "solidfmm":
        return rotate_complex_local_from_z_solidfmm(local_z, delta, order=p)
    return rotate_complex_local_from_z(local_z, delta, order=p)


@partial(jax.jit, static_argnames=("order", "rotation"))
def m2l_complex_reference_batch(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    rotation: str = "bdz",
) -> Array:
    """Batch M2L in complex basis (rotate → z-translate → rotate back)."""
    return jax.vmap(
        lambda m, d: m2l_complex_reference(m, d, order=order, rotation=rotation),
        in_axes=(0, 0),
        out_axes=0,
    )(multipoles, deltas)


@partial(jax.jit, static_argnames=("order",))
def m2l_complex_reference_batch_cached_blocks(
    multipoles: Array,
    deltas: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    *,
    order: int,
) -> Array:
    """Batch M2L using precomputed rotation blocks for each pair."""
    p = int(order)
    M_rot = _apply_complex_rotation_blocks_padded_batch(
        multipoles,
        blocks_to_z,
        order=p,
    )
    r = jnp.sqrt(jnp.maximum(jnp.sum(deltas * deltas, axis=1), 1e-60))
    local_z = translate_along_z_m2l_complex_batch(M_rot, r, order=p)
    return _apply_complex_rotation_blocks_padded_batch(
        local_z,
        blocks_from_z,
        order=p,
    )


@partial(jax.jit, static_argnames=("order",))
def translate_along_z_m2l_complex_batch(
    multipoles: Array,
    r: Array,
    *,
    order: int,
) -> Array:
    """Batch translate complex multipoles to locals along +z."""
    return jax.vmap(
        lambda m, rr: translate_along_z_m2l_complex(m, rr, order=order),
        in_axes=(0, 0),
        out_axes=0,
    )(multipoles, r)


@partial(jax.jit, static_argnames=("order",))
def translate_along_z_m2m_complex_batch(
    multipoles: Array,
    dz: Array,
    *,
    order: int,
) -> Array:
    """Batch translate complex multipoles along +z."""
    return jax.vmap(
        lambda m, rr: translate_along_z_m2m_complex(m, rr, order=order),
        in_axes=(0, 0),
        out_axes=0,
    )(multipoles, dz)


@partial(jax.jit, static_argnames=("order",))
def translate_along_z_l2l_complex_batch(
    locals: Array,
    dz: Array,
    *,
    order: int,
) -> Array:
    """Batch translate complex locals along +z."""
    return jax.vmap(
        lambda m, rr: translate_along_z_l2l_complex(m, rr, order=order),
        in_axes=(0, 0),
        out_axes=0,
    )(locals, dz)


@partial(jax.jit, static_argnames=("order", "rotation"))
def l2l_complex_batch(
    locals: Array,
    deltas: Array,
    *,
    order: int,
    rotation: str = "bdz",
) -> Array:
    """Batch L2L in complex basis."""
    return jax.vmap(
        lambda l, d: l2l_complex(l, d, order=order, rotation=rotation),
        in_axes=(0, 0),
        out_axes=0,
    )(locals, deltas)
