"""Complex-basis operators (solidfmm-style reference) in JAX."""

from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ._precision import highest_matmul_precision
from ._transverse_degeneracy_jvp import (
    TransverseGenerators,
    with_transverse_degeneracy_jvp,
    with_transverse_degeneracy_tangent,
    withdraw_unresolvable_transverse,
    without_unresolvable_transverse_jvp,
)
from .complex_harmonics import complex_R_solidfmm, complex_R_solidfmm_preserve_dtype
from .dtypes import complex_dtype_for_real, floor_squared_radius
from .real_harmonics import (
    _compute_dehnen_B_matrix_complex,
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
def regular_solid_harmonic_gradient_coefficients_preserve_dtype(
    delta: Array,
    *,
    order: int,
) -> Array:
    """Return local-gradient coefficients without widening float32 deltas."""
    p = int(order)
    base = jnp.asarray(complex_R_solidfmm_preserve_dtype(delta, order=p))
    grad_x = _lower_complex_harmonics_one_axis(base, order=p, axis=0)
    grad_y = _lower_complex_harmonics_one_axis(base, order=p, axis=1)
    grad_z = _lower_complex_harmonics_one_axis(base, order=p, axis=2)
    return jnp.stack((grad_x, grad_y, grad_z), axis=0)


def evaluate_local_complex_grad_analytic_preserve_dtype(
    local: Array,
    delta: Array,
    *,
    order: int,
    conjugate_left: bool = True,
) -> Array:
    """Evaluate the analytic local gradient without float32->complex128 widening."""
    p = int(order)
    ncoeff = sh_size(p)
    local_coeffs = jnp.asarray(local)[:ncoeff]
    if conjugate_left:
        local_coeffs = jnp.conjugate(local_coeffs)
    grad_coeffs = regular_solid_harmonic_gradient_coefficients_preserve_dtype(
        delta,
        order=p,
    )[:, :ncoeff]
    return jnp.real(jnp.sum(local_coeffs[None, :] * grad_coeffs, axis=-1))


def _regular_solid_harmonic_order4_scalars(delta: Array) -> tuple[Array, ...]:
    """Return packed order-4 regular harmonics as scalar expressions."""
    delta_arr = jnp.asarray(delta)
    real_dtype = (
        delta_arr.dtype
        if jnp.issubdtype(delta_arr.dtype, jnp.floating)
        else jnp.float32
    )
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64

    d = jnp.asarray(delta, dtype=real_dtype)
    x, y, z = d[0], d[1], d[2]
    xy = x.astype(complex_dtype) + jnp.asarray(1j, dtype=complex_dtype) * y.astype(
        complex_dtype
    )
    zc = z.astype(complex_dtype)
    r2c = (x * x + y * y + z * z).astype(complex_dtype)
    one = jnp.asarray(1.0, dtype=real_dtype).astype(complex_dtype)

    pos: dict[tuple[int, int], Array] = {}
    pos[(0, 0)] = one
    pos[(1, 0)] = zc
    pos[(1, 1)] = xy * jnp.asarray(0.5, dtype=real_dtype).astype(complex_dtype)
    pos[(2, 2)] = (
        pos[(1, 1)] * xy * jnp.asarray(0.25, dtype=real_dtype).astype(complex_dtype)
    )
    pos[(3, 3)] = (
        pos[(2, 2)]
        * xy
        * jnp.asarray(1.0 / 6.0, dtype=real_dtype).astype(complex_dtype)
    )
    pos[(4, 4)] = (
        pos[(3, 3)] * xy * jnp.asarray(0.125, dtype=real_dtype).astype(complex_dtype)
    )
    pos[(2, 1)] = zc * pos[(1, 1)]
    pos[(3, 2)] = zc * pos[(2, 2)]
    pos[(4, 3)] = zc * pos[(3, 3)]
    pos[(2, 0)] = (
        jnp.asarray(3.0, dtype=real_dtype).astype(complex_dtype) * zc * pos[(1, 0)]
        - r2c * pos[(0, 0)]
    ) * jnp.asarray(0.25, dtype=real_dtype).astype(complex_dtype)
    pos[(3, 0)] = (
        jnp.asarray(5.0, dtype=real_dtype).astype(complex_dtype) * zc * pos[(2, 0)]
        - r2c * pos[(1, 0)]
    ) * jnp.asarray(1.0 / 9.0, dtype=real_dtype).astype(complex_dtype)
    pos[(4, 0)] = (
        jnp.asarray(7.0, dtype=real_dtype).astype(complex_dtype) * zc * pos[(3, 0)]
        - r2c * pos[(2, 0)]
    ) * jnp.asarray(1.0 / 16.0, dtype=real_dtype).astype(complex_dtype)
    pos[(3, 1)] = (
        jnp.asarray(5.0, dtype=real_dtype).astype(complex_dtype) * zc * pos[(2, 1)]
        - r2c * pos[(1, 1)]
    ) * jnp.asarray(0.125, dtype=real_dtype).astype(complex_dtype)
    pos[(4, 1)] = (
        jnp.asarray(7.0, dtype=real_dtype).astype(complex_dtype) * zc * pos[(3, 1)]
        - r2c * pos[(2, 1)]
    ) * jnp.asarray(1.0 / 15.0, dtype=real_dtype).astype(complex_dtype)
    pos[(4, 2)] = (
        jnp.asarray(7.0, dtype=real_dtype).astype(complex_dtype) * zc * pos[(3, 2)]
        - r2c * pos[(2, 2)]
    ) * jnp.asarray(1.0 / 12.0, dtype=real_dtype).astype(complex_dtype)

    def get(n: int, m: int) -> Array:
        if m >= 0:
            return pos[(n, m)]
        m_abs = -m
        sign = jnp.asarray(-1.0 if (m_abs % 2) else 1.0, dtype=real_dtype).astype(
            complex_dtype
        )
        return sign * jnp.conjugate(pos[(n, m_abs)])

    return tuple(get(n, m) for n in range(5) for m in range(-n, n + 1))


def evaluate_local_complex_grad_order4_unrolled(
    local: Array,
    delta: Array,
    *,
    order: int,
    conjugate_left: bool = True,
) -> Array:
    """Evaluate order-4 local gradient with scalar recurrence/contraction."""
    if int(order) != 4:
        return evaluate_local_complex_grad_analytic_preserve_dtype(
            local,
            delta,
            order=order,
            conjugate_left=conjugate_left,
        )

    r = _regular_solid_harmonic_order4_scalars(delta)
    local_coeffs = jnp.asarray(local)[:25]
    if conjugate_left:
        local_coeffs = jnp.conjugate(local_coeffs)
    cdtype = jnp.result_type(local_coeffs.dtype, r[0].dtype)
    half = jnp.asarray(0.5, dtype=jnp.real(jnp.zeros((), dtype=cdtype)).dtype).astype(
        cdtype
    )
    half_i = jnp.asarray(0.5j, dtype=cdtype)
    zero = jnp.asarray(0.0, dtype=cdtype)

    def ridx(n: int, m: int) -> int:
        return n * n + (m + n)

    def src(n: int, m: int) -> Array:
        if n < 0 or m < -n or m > n:
            return zero
        return r[ridx(n, m)]

    acc_x = zero
    acc_y = zero
    acc_z = zero
    for n in range(1, 5):
        for m in range(-n, n + 1):
            coeff = local_coeffs[ridx(n, m)].astype(cdtype)
            left = src(n - 1, m - 1)
            right = src(n - 1, m + 1)
            acc_x = acc_x + coeff * (half * left - half * right)
            acc_y = acc_y + coeff * (half_i * left + half_i * right)
            acc_z = acc_z + coeff * src(n - 1, m)
    return jnp.real(jnp.stack((acc_x, acc_y, acc_z), axis=0))


def evaluate_local_complex_with_grad_analytic(
    local: Array,
    delta: Array,
    *,
    order: int,
    conjugate_left: bool = True,
) -> tuple[Array, Array]:
    """Evaluate complex local expansion and gradient without autodiff."""
    p = int(order)
    ncoeff = sh_size(p)
    local_coeffs = jnp.asarray(local)[:ncoeff]
    if conjugate_left:
        local_coeffs = jnp.conjugate(local_coeffs)
    regular = jnp.asarray(complex_R_solidfmm(delta, order=p))[:ncoeff]
    grad_coeffs = regular_solid_harmonic_gradient_coefficients(delta, order=p)[
        :, :ncoeff
    ]
    potential = jnp.real(jnp.sum(local_coeffs * regular))
    grad = jnp.real(jnp.sum(local_coeffs[None, :] * grad_coeffs, axis=-1))
    return grad, potential


def evaluate_local_complex_grad_analytic(
    local: Array,
    delta: Array,
    *,
    order: int,
    conjugate_left: bool = True,
) -> Array:
    """Evaluate only the complex local-expansion gradient without autodiff."""
    p = int(order)
    ncoeff = sh_size(p)
    local_coeffs = jnp.asarray(local)[:ncoeff]
    if conjugate_left:
        local_coeffs = jnp.conjugate(local_coeffs)
    grad_coeffs = regular_solid_harmonic_gradient_coefficients(delta, order=p)[
        :, :ncoeff
    ]
    return jnp.real(jnp.sum(local_coeffs[None, :] * grad_coeffs, axis=-1))


@partial(jax.jit, static_argnames=("order", "conjugate_left"))
def evaluate_local_complex_grad_analytic_batch(
    local: Array,
    deltas: Array,
    *,
    order: int,
    conjugate_left: bool = True,
) -> Array:
    """Batch evaluate only complex local-expansion gradients."""
    return jax.vmap(
        lambda d: evaluate_local_complex_grad_analytic(
            local,
            d,
            order=order,
            conjugate_left=conjugate_left,
        )
    )(deltas)


@partial(jax.jit, static_argnames=("order", "conjugate_left"))
def evaluate_local_complex_with_grad_analytic_batch(
    local: Array,
    deltas: Array,
    *,
    order: int,
    conjugate_left: bool = True,
) -> tuple[Array, Array]:
    """Batch evaluate complex local expansion gradients without autodiff."""
    return jax.vmap(
        lambda d: evaluate_local_complex_with_grad_analytic(
            local,
            d,
            order=order,
            conjugate_left=conjugate_left,
        )
    )(deltas)


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
    # This is the complex-basis M2L: always accumulate in a complex dtype even
    # if a real-typed multipole array is passed in (defensive; a real input
    # would otherwise raise when constructing a complex accumulator).
    cdtype = jnp.result_type(multipole.dtype, jnp.complex64)
    out = jnp.zeros((ncoeff,), dtype=cdtype)
    fact = _factorial_table_cached(2 * p, dtype)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            m_abs = abs(m)
            acc = jnp.asarray(0.0 + 0.0j, dtype=cdtype)
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


# --------------------------------------------------------------------------
# Rotation generators, for the analytic transverse derivative at rho == 0.
# --------------------------------------------------------------------------
#
# The complex-basis counterpart of ``real_harmonics``'s generators. See
# :mod:`jaccpot.operators._transverse_degeneracy_jvp` for what they are for and
# ``docs/rotation_degeneracy_derivative.md`` for the derivation.
#
# Calibrated exactly as the real-basis ones were -- against central differences of
# the operator itself, not on paper -- and the two sign choices were searched rather
# than assumed. At order 4, ``z = 2.5``, ``eps = 1e-5`` on
# :func:`m2l_complex_reference`, the four combinations of the two signs give
# 2.0e+00, 1.5e+00, 9.6e-01 and **7.4e-11**; the surviving one is the ``-swap Lambda
# swap`` coded below, matching the real basis' sign.
#
# One structural difference from the real basis: there the local rotation *is* the
# transpose of the multipole one, so ``G^L = -(G^M)^T``. Here the local rotation is
# built from its own swap matrix (``B_T`` rather than ``B_U``), so each
# representation gets its generator from the swap matrix the rotation blocks
# actually use. Deriving one from the other would be a coincidence to rely on.


@lru_cache(maxsize=None)
def _complex_z_rotation_generator(ell: int) -> np.ndarray:
    """``d/dangle`` of :func:`_complex_Dz` at ``angle == 0``, degree ``ell``.

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.

    Returns
    -------
    np.ndarray
        ``[2*ell+1, 2*ell+1]`` complex128, the diagonal ``i * m`` for
        ``m = -ell .. ell``.
    """
    return np.diag(1j * np.arange(-ell, ell + 1)).astype(np.complex128)


@lru_cache(maxsize=None)
@highest_matmul_precision
def _complex_rotation_generator_block(ell: int, axis: str, basis: str) -> np.ndarray:
    """Generator of rotation about ``axis`` for one degree, in one representation.

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.
    axis : str
        ``'x'`` or ``'y'``. The x-generator conjugates the z-generator with the
        involutory x<->z swap this basis' rotation blocks use; the y-generator is the
        x-generator conjugated by a quarter turn about z.
    basis : str
        ``'multipole'`` (swap ``B_U``) or ``'local'`` (swap ``B_T``), matching
        :func:`_complex_rotation_blocks_to_z_solidfmm`.

    Returns
    -------
    np.ndarray
        ``[2*ell+1, 2*ell+1]`` complex128.

    Raises
    ------
    ValueError
        If ``axis`` or ``basis`` is not one of the listed values.
    """
    # The matmuls below are numpy, in float64, so the pinned precision is inert
    # here -- the decorator is on for policy conformance
    # (tests/unit/operators/test_matmul_precision_pinned.py) rather than because
    # this function could drop to TF32.
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    if basis not in ("multipole", "local"):
        raise ValueError(f"basis must be 'multipole' or 'local', got {basis!r}")
    B_T, B_U = _complex_swap_matrices_cached(ell, "complex128")
    swap = B_U if basis == "multipole" else B_T
    generator = -swap @ _complex_z_rotation_generator(ell) @ swap
    if axis == "y":
        # Built from this module's own ``_complex_Dz`` so it cannot drift from the
        # rotation blocks. That returns a ``jnp`` array, and under ``jax.jit``
        # ``jnp.asarray`` of a constant is a tracer, so pulling it back to numpy --
        # which is what lets this be ``lru_cache``d into a compile-time constant --
        # needs the constant-folding context. This runs inside a ``custom_jvp`` rule,
        # i.e. always inside a trace.
        with jax.ensure_compile_time_eval():
            quarter, quarter_back = (
                np.asarray(
                    _complex_Dz(
                        ell,
                        jnp.asarray(sign * np.pi / 2.0, dtype=jnp.float64),
                        # ``np.dtype(...)``, not the ``jnp.complex128`` scalar type:
                        # this parameter is annotated ``jnp.dtype`` and the runtime
                        # typechecker (JACCPOT_RUNTIME_TYPECHECK=1) rejects the latter.
                        dtype=np.dtype(np.complex128),
                    )
                )
                for sign in (+1.0, -1.0)
            )
        generator = quarter @ generator @ quarter_back
    return generator


@lru_cache(maxsize=None)
def _complex_transverse_generator_packed(
    order: int, axis: str, basis: str
) -> np.ndarray:
    """Per-degree generator blocks assembled into one packed square matrix.

    Parameters
    ----------
    order : int
        Maximum SH degree ``p``.
    axis : str
        ``'x'`` or ``'y'``, as in :func:`_complex_rotation_generator_block`.
    basis : str
        ``'multipole'`` or ``'local'``, as in :func:`_complex_rotation_generator_block`.

    Returns
    -------
    np.ndarray
        ``[(p+1)^2, (p+1)^2]`` complex128, block-diagonal in ``ell`` with the packing
        of :func:`~jaccpot.operators.real_harmonics.sh_offset`.
    """
    p = int(order)
    packed = np.zeros((sh_size(p), sh_size(p)), dtype=np.complex128)
    for ell in range(p + 1):
        block = slice(sh_offset(ell), sh_offset(ell + 1))
        packed[block, block] = _complex_rotation_generator_block(ell, axis, basis)
    return packed


def complex_transverse_generators(
    order: int,
    dtype: jnp.dtype,
    *,
    in_representation: str,
    out_representation: str,
) -> TransverseGenerators:
    """Complex-basis generators for the ``rho == 0`` transverse derivative.

    Feeds :func:`~jaccpot.operators._transverse_degeneracy_jvp.with_transverse_degeneracy_jvp`,
    which documents what the four matrices are for.

    Parameters
    ----------
    order : int
        Maximum SH degree ``p``.
    dtype : jnp.dtype
        Working complex dtype of the coefficients. The generators are built in
        complex128 and cast down, so the generator matmuls run in the working dtype
        rather than promoting complex64 coefficients.
    in_representation : str
        ``'multipole'`` or ``'local'`` -- which slot the operator's input occupies.
    out_representation : str
        Likewise for its output. M2L is multipole in, local out; M2M is multipole to
        multipole; L2L is local to local.

    Returns
    -------
    TransverseGenerators
        The four ``[(p+1)^2, (p+1)^2]`` packed generators, in ``dtype``.
    """
    return TransverseGenerators(
        in_x=jnp.asarray(
            _complex_transverse_generator_packed(order, "x", in_representation),
            dtype=dtype,
        ),
        in_y=jnp.asarray(
            _complex_transverse_generator_packed(order, "y", in_representation),
            dtype=dtype,
        ),
        out_x=jnp.asarray(
            _complex_transverse_generator_packed(order, "x", out_representation),
            dtype=dtype,
        ),
        out_y=jnp.asarray(
            _complex_transverse_generator_packed(order, "y", out_representation),
            dtype=dtype,
        ),
    )


#: :func:`complex_transverse_generators` bound to each cascade operator's pair of
#: representations, ready to hand to
#: :func:`~jaccpot.operators._transverse_degeneracy_jvp.with_transverse_degeneracy_jvp`.
_M2L_TRANSVERSE_GENERATORS = partial(
    complex_transverse_generators,
    in_representation="multipole",
    out_representation="local",
)
_M2M_TRANSVERSE_GENERATORS = partial(
    complex_transverse_generators,
    in_representation="multipole",
    out_representation="multipole",
)
_L2L_TRANSVERSE_GENERATORS = partial(
    complex_transverse_generators,
    in_representation="local",
    out_representation="local",
)


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


@highest_matmul_precision
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


def _angles_from_delta_solidfmm(delta: Array) -> tuple[Array, Array]:
    """Angles matching solidfmm's euler() convention.

    solidfmm defines:
        cos(alpha)=y/rxy, sin(alpha)=x/rxy
        cos(beta)=z/r, sin(beta)=-rxy/r
    so alpha=atan2(x,y), beta=atan2(-rxy,z).
    """
    x, y, z = delta[0], delta[1], delta[2]
    # NaN-safe (double-where) angle computation. The forward values are
    # unchanged for every displacement, but the reverse-mode cotangents stay
    # finite at the degenerate directions the fixed-topology FMM genuinely hits:
    #   * zero displacement (COM of a single-child internal node == its child),
    #   * z-axis-aligned displacement (rho == 0), e.g. lattice-aligned M2L pairs.
    # At those points ``sqrt`` (infinite grad at 0) and ``arctan2`` (0/0 grad at
    # the origin) would otherwise inject NaNs into the M2L/L2L reverse pass. The
    # azimuth is undefined there and the rotation reduces to a pure polar turn /
    # identity, so returning 0 keeps the forward exact.
    #
    # THE ZERO COTANGENT IS NOT CORRECT, and this is deliberate -- the missing
    # derivative is supplied one level up rather than here. Do not try to fix it at
    # this site. The retracted history is kept because its measurement was real and
    # its scope is the instructive part: on four z-stacked clusters sharing one (x,y)
    # point set (COM displacements exactly (0, 0, +-8), so rho == 0 on every M2L
    # pair), reverse-mode AD agreed with finite differences to ~10 significant
    # digits, and the transverse one-sided derivatives converged to the same value
    # from both sides -- read at the time as "the loss is smooth and nothing is
    # dropped".
    #
    # What that construction could not see is that it is symmetric: the same (x,y)
    # set and the same intra-cluster masses in every cluster, so the per-pair
    # transverse errors cancel in the sum. Re-measured with an ASYMMETRIC random
    # cotangent and an asymmetric purely-transverse perturbation direction, on a
    # system with 6 of 24 M2L pairs at rho == 0 exactly, FD and AD disagreed by
    # 1.8e-05 relative in this (complex) basis and 1.9e-03 in the real one --
    # stable across step size, so not finite-difference noise.
    #
    # The forward argument still holds and is worth keeping: the m != 0 terms carry
    # sin^|m|(theta) = (rho/r)^|m| factors that annihilate the arbitrary azimuth, so
    # the VALUE at rho == 0 is exact. It is the derivative that is not -- the
    # cascade is differentiable there (the limit is direction-independent) and the
    # true transverse derivative is nonzero. It cannot be recovered at this site: the
    # code reaches (x, y) only through rho and the azimuth, so every chain-rule route
    # carries x/rho or y/rho^2 and the O(rho) coefficient the derivative needs has
    # already been divided out.
    #
    # RESOLVED at the cascade level (G.10). ``m2l_complex_reference``, ``m2m_complex``
    # and ``l2l_complex`` each carry a ``custom_jvp`` that supplies the transverse
    # derivative analytically, from the rotational covariance of the assembled
    # operator -- which is available there and not here, because an individual
    # alignment block is genuinely direction-dependent at rho == 0 while the product
    # is not. See :mod:`jaccpot.operators._transverse_degeneracy_jvp` and
    # ``docs/rotation_degeneracy_derivative.md``. Asserted by
    # ``tests/unit/operators/test_complex_ops.py::test_complex_cascade_transverse_gradient_at_rho_zero``.
    #
    # WARNING for anyone extending the coverage: a *uniform lattice* does not
    # exercise this guard, despite what
    # ``tests/unit/test_gradient_correctness.py::test_no_nan_axis_aligned_grid``
    # says. Centres are COM, not geometric box centres, so lattice leaf COMs are
    # generically off-axis relative to each other -- measured on that test's own
    # 5^3 grid, zero of its 22 M2L pairs have rho == 0 and the minimum rho^2 is
    # 5.64. Use the z-stacked-cluster construction above to reach the degeneracy.
    # Also note a central difference cannot validate a subgradient choice at a
    # symmetric kink: it returns 0 there whether or not 0 is correct.
    rho_sq = x * x + y * y
    rho_pos = rho_sq > 0
    rho = jnp.where(rho_pos, jnp.sqrt(jnp.where(rho_pos, rho_sq, 1.0)), 0.0)
    alpha = jnp.where(rho_pos, jnp.arctan2(jnp.where(rho_pos, x, 1.0), y), 0.0)
    r_pos = (rho_sq + z * z) > 0
    beta = jnp.where(r_pos, jnp.arctan2(-rho, jnp.where(r_pos, z, 1.0)), 0.0)
    return alpha, beta


@highest_matmul_precision
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


@highest_matmul_precision
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


# Withdraws its in-band transverse tangent: an individual alignment block has no
# transverse derivative near rho == 0 (its limit is approach-dependent, unlike the
# assembled cascade's), so it hands the caller nothing there rather than something
# wrong, and the caller supplies the cascade-level term. Applied to the *padded*
# per-delta builder, which is what the batch (precomputed-block) lanes go through; the
# unpadded one feeds m2l_complex_reference / m2m_complex / l2l_complex, which already
# carry the cascade-level rule themselves.
@without_unresolvable_transverse_jvp
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


# Withdraws its in-band transverse tangent: an individual alignment block has no
# transverse derivative near rho == 0 (its limit is approach-dependent, unlike the
# assembled cascade's), so it hands the caller nothing there rather than something
# wrong, and the caller supplies the cascade-level term. Applied to the *padded*
# per-delta builder, which is what the batch (precomputed-block) lanes go through; the
# unpadded one feeds m2l_complex_reference / m2m_complex / l2l_complex, which already
# carry the cascade-level rule themselves.
@without_unresolvable_transverse_jvp
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
    rotated = jnp.einsum(
        "bij,bj->bi", blocks_array, packed, precision=lax.Precision.HIGHEST
    )
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
    rotated = jnp.einsum(
        "nbij,nbj->nbi", blocks_array, packed, precision=lax.Precision.HIGHEST
    )
    return jax.vmap(lambda c: _unpack_coeffs_by_ell(c, order=order))(rotated)


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


# The `with_transverse_degeneracy_jvp` layer on the three cascade operators below
# sits *inside* the `jax.jit`, so it adds no dispatch boundary. It leaves the primal
# bit-identical and supplies only the transverse derivative on the `rho == 0` axis,
# where `_angles_from_delta_solidfmm`'s guards return a zero cotangent; see
# :mod:`jaccpot.operators._transverse_degeneracy_jvp`.
@partial(jax.jit, static_argnames=("order", "rotation"))
@partial(with_transverse_degeneracy_jvp, generators=_M2M_TRANSVERSE_GENERATORS)
def m2m_complex(
    multipole: Array,
    delta: Array,
    *,
    order: int,
    rotation: str = "solidfmm",
) -> Array:
    """Complex M2M using A6: rotate → z-translate → rotate back."""
    if rotation != "solidfmm":
        raise ValueError("rotation must be 'solidfmm'")
    p = int(order)
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)

    r = jnp.sqrt(
        floor_squared_radius(jnp.dot(delta, delta, precision=lax.Precision.HIGHEST))
    )
    M_rot = rotate_complex_multipole_to_z_solidfmm(multipole, delta, order=p)
    M_z = translate_along_z_m2m_complex_solidfmm(M_rot, r, order=p)
    return rotate_complex_multipole_from_z_solidfmm(M_z, delta, order=p)


@partial(jax.jit, static_argnames=("order", "rotation"))
@partial(with_transverse_degeneracy_jvp, generators=_L2L_TRANSVERSE_GENERATORS)
def l2l_complex(
    local: Array,
    delta: Array,
    *,
    order: int,
    rotation: str = "solidfmm",
) -> Array:
    """Complex L2L using A6: rotate → z-translate → rotate back."""
    if rotation != "solidfmm":
        raise ValueError("rotation must be 'solidfmm'")
    p = int(order)
    local = jnp.asarray(local)
    delta = jnp.asarray(delta)

    r = jnp.sqrt(
        floor_squared_radius(jnp.dot(delta, delta, precision=lax.Precision.HIGHEST))
    )
    L_rot = rotate_complex_local_to_z_solidfmm(local, delta, order=p)
    L_z = translate_along_z_l2l_complex(L_rot, r, order=p)
    return rotate_complex_local_from_z_solidfmm(L_z, delta, order=p)


@partial(with_transverse_degeneracy_jvp, generators=_M2L_TRANSVERSE_GENERATORS)
def m2l_complex_reference(
    multipole: Array,
    delta: Array,
    *,
    order: int,
    rotation: str = "solidfmm",
) -> Array:
    """Reference M2L in complex basis (rotate → z-translate → rotate back).

    Differentiable in both arguments, forward and reverse. Near the ``rho == 0`` axis
    the ``d/dx`` and ``d/dy`` cotangents come from a ``custom_jvp`` rather than from
    differentiating ``_angles_from_delta_solidfmm``'s guarded azimuth; the analytic
    branch applies inside exactly zero outside a narrow band around that axis (``rho <= sqrt(eps) * |delta|``, the measured crossover between the two routes' errors).
    """
    if rotation != "solidfmm":
        raise ValueError("rotation must be 'solidfmm'")
    p = int(order)
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)

    ncoeff = sh_size(p)
    multipole = multipole[:ncoeff]

    M_rotated = rotate_complex_multipole_to_z_solidfmm(multipole, delta, order=p)

    r = jnp.sqrt(
        floor_squared_radius(jnp.dot(delta, delta, precision=lax.Precision.HIGHEST))
    )
    local_z = translate_along_z_m2l_complex(M_rotated, r, order=p)

    return rotate_complex_local_from_z_solidfmm(local_z, delta, order=p)


@partial(jax.jit, static_argnames=("order", "rotation"))
def m2l_complex_reference_batch(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    rotation: str = "solidfmm",
) -> Array:
    """Batch M2L in complex basis (rotate → z-translate → rotate back)."""
    return jax.vmap(
        lambda m, d: m2l_complex_reference(m, d, order=order, rotation=rotation),
        in_axes=(0, 0),
        out_axes=0,
    )(multipoles, deltas)


@partial(jax.jit, static_argnames=("order",))
@partial(with_transverse_degeneracy_jvp, generators=_M2L_TRANSVERSE_GENERATORS)
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
    r = jnp.sqrt(floor_squared_radius(jnp.sum(deltas * deltas, axis=1)))
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
    rotation: str = "solidfmm",
) -> Array:
    """Batch L2L in complex basis."""
    return jax.vmap(
        lambda l, d: l2l_complex(l, d, order=order, rotation=rotation),
        in_axes=(0, 0),
        out_axes=0,
    )(locals, deltas)


def _m2l_complex_fused_twin(
    multipoles: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    radii: Array,
    *,
    order: int,
) -> Array:
    """Pure-JAX twin of the fused Pallas complex M2L, for the transverse correction only.

    Deferred import: :mod:`jaccpot.pallas.m2l_complex_fused` reaches back into
    :mod:`jaccpot.operators`, so a module-scope import is a cycle. This is the same twin
    the fused kernel's own ``custom_vjp`` uses as its correctness reference, which is why
    it is the right thing to compute the correction with -- the kernel itself cannot be
    used, because the correction lives inside a JVP rule and a ``custom_vjp`` is not
    forward-differentiable.

    Parameters
    ----------
    multipoles : Array
        Source multipole coefficients ``[N, (p+1)^2]``, complex.
    blocks_to_z : Array
        Multipole world->z blocks ``[N, p+1, 2p+1, 2p+1]``, complex.
    blocks_from_z : Array
        Local z->world blocks, same shape.
    radii : Array
        Translation distances ``[N]``.
    order : int
        Maximum SH degree ``p``. Static under ``jit``.

    Returns
    -------
    Array
        Complex local expansion contributions ``[N, (p+1)^2]``.
    """
    from jaccpot.pallas.m2l_complex_fused import m2l_complex_fused_jax

    return m2l_complex_fused_jax(
        multipoles, blocks_to_z, blocks_from_z, radii, order=int(order)
    )


#: Re-exported so the complex fused lane's two halves come from one module and cannot be
#: reached apart, exactly as
#: :data:`~jaccpot.operators.m2l_real_rot_scale.m2l_real_fused_align_deltas` is for the
#: real one. This withdraws the unresolvable transverse tangent from the displacement
#: *before* anything is built from it.
#:
#: This lane needs it stated explicitly even though its block builders already carry
#: :func:`~jaccpot.operators._transverse_degeneracy_jvp.without_unresolvable_transverse_jvp`:
#: the withdrawal is idempotent, and naming it here is what makes the pairing visible at
#: the call site instead of being an accident of which builder happens to be used.
m2l_complex_fused_align_deltas = withdraw_unresolvable_transverse

#: Puts the analytic transverse derivative back onto the fused Pallas complex M2L's
#: output. Signature
#: ``(out, multipoles, deltas, blocks_to_z, blocks_from_z, radii, order=p)``; the primal
#: returns ``out`` untouched, so there is no forward footprint. Pair it with
#: :data:`m2l_complex_fused_align_deltas` on the displacement *before* the blocks and
#: radius are built from it -- see
#: ``runtime/kernels/core.py::_m2l_complex_batch_kernel_fused_pallas``, its only caller.
#: Using either half alone gives a wrong gradient: without the carrier this lane's
#: on-axis ``d/dx`` and ``d/dy`` come back exactly zero, measured 5.1e-01 from the
#: pure-JAX reference.
m2l_complex_fused_carry_axis_derivative = with_transverse_degeneracy_tangent(
    _m2l_complex_fused_twin,
    generators=_M2L_TRANSVERSE_GENERATORS,
)
