import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.complex_harmonics import complex_R_solidfmm
from jaccpot.operators.complex_ops import (
    complex_dot,
    contract_spatial_derivative_with_velocity,
    evaluate_local_complex_derivative_tower,
    evaluate_local_complex_with_grad,
    l2l_complex,
    m2m_complex,
    regular_solid_harmonic_directional_derivative,
    regular_solid_harmonic_directional_derivative_batch,
    regular_solid_harmonic_directional_derivative_order,
    regular_solid_harmonic_directional_derivative_order_batch,
    translate_along_z_l2l_complex,
    translate_along_z_l2l_complex_batch,
    translate_along_z_m2l_complex,
    translate_along_z_m2l_complex_batch,
    translate_along_z_m2m_complex,
    translate_along_z_m2m_complex_batch,
)
from jaccpot.operators.real_harmonics import sh_size
from jaccpot.operators.solidfmm_reference import (
    translate_along_z_m2l_complex as ref_translate_z,
)
from jaccpot.operators.symmetric_tensors import symmetric_multi_indices_3d


def _complex_coeffs(order: int, seed: int) -> np.ndarray:
    """Build a complex coefficient vector of the given order from a fixed seed."""
    rng = np.random.default_rng(seed)
    ncoeff = sh_size(order)
    return rng.normal(size=(ncoeff,)) + 1j * rng.normal(size=(ncoeff,))


def test_translate_along_z_m2l_complex_matches_reference() -> None:
    order = 5
    multipole = _complex_coeffs(order, 0)
    r = 3.7

    ref = ref_translate_z(multipole, r, order=order)
    got = translate_along_z_m2l_complex(
        jnp.asarray(multipole), jnp.asarray(r), order=order
    )

    assert np.allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)


def test_complex_dot_conjugate_left() -> None:
    order = 3
    rng = np.random.default_rng(1)
    ncoeff = sh_size(order)
    left = rng.normal(size=(ncoeff,)) + 1j * rng.normal(size=(ncoeff,))
    right = rng.normal(size=(ncoeff,)) + 1j * rng.normal(size=(ncoeff,))

    ref = np.sum(np.conjugate(left) * right)
    got = complex_dot(
        jnp.asarray(left), jnp.asarray(right), order=order, conjugate_left=True
    )

    assert np.allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)


def _translate_z_m2m_reference(
    multipole: np.ndarray, dz: float, order: int
) -> np.ndarray:
    ncoeff = (order + 1) * (order + 1)
    out = np.zeros((ncoeff,), dtype=np.complex128)
    fact = np.array([math.factorial(k) for k in range(order + 1)], dtype=np.float64)
    for n in range(order + 1):
        for m in range(-n, n + 1):
            m_abs = abs(m)
            acc = 0.0 + 0.0j
            for k in range(0, n - m_abs + 1):
                src_n = n - k
                if m_abs > src_n:
                    continue
                src_idx = src_n * src_n + (m + src_n)
                acc += (dz**k) * multipole[src_idx] / fact[k]
            out[n * n + (m + n)] = acc
    return out


def _translate_z_l2l_reference(local: np.ndarray, dz: float, order: int) -> np.ndarray:
    ncoeff = (order + 1) * (order + 1)
    out = np.zeros((ncoeff,), dtype=np.complex128)
    fact = np.array([math.factorial(k) for k in range(order + 1)], dtype=np.float64)
    for n in range(order + 1):
        for m in range(-n, n + 1):
            acc = 0.0 + 0.0j
            for k in range(0, order - n + 1):
                src_n = n + k
                if src_n > order:
                    continue
                src_idx = src_n * src_n + (m + src_n)
                acc += (dz**k) * local[src_idx] / fact[k]
            out[n * n + (m + n)] = acc
    return out


def _l2l_reference_complex(
    local: np.ndarray, delta: np.ndarray, order: int
) -> np.ndarray:
    p = int(order)
    R = np.asarray(complex_R_solidfmm(jnp.asarray(delta), order=p))
    out = np.zeros((sh_size(p),), dtype=np.complex128)
    for n in range(p + 1):
        for m in range(-n, n + 1):
            acc = 0.0 + 0.0j
            for k in range(n, p + 1):
                for l in range(-k, k + 1):
                    q = k - n
                    t = l - m
                    if abs(t) > q:
                        continue
                    acc += local[k * k + (l + k)] * np.conjugate(R[q * q + (t + q)])
            out[n * n + (m + n)] = acc
    return out


def test_translate_along_z_m2m_complex_matches_reference() -> None:
    order = 4
    multipole = _complex_coeffs(order, 2)
    dz = 0.37

    ref = _translate_z_m2m_reference(multipole, dz, order)
    got = translate_along_z_m2m_complex(
        jnp.asarray(multipole), jnp.asarray(dz), order=order
    )

    assert np.allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)


def test_translate_along_z_l2l_complex_matches_reference() -> None:
    order = 4
    local = _complex_coeffs(order, 3)
    dz = -0.21

    ref = _translate_z_l2l_reference(local, dz, order)
    got = translate_along_z_l2l_complex(
        jnp.asarray(local), jnp.asarray(dz), order=order
    )

    assert np.allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "single_fn, batch_fn, seed, lo, hi",
    [
        (
            translate_along_z_m2l_complex,
            translate_along_z_m2l_complex_batch,
            4,
            0.5,
            2.5,
        ),
        (
            translate_along_z_m2m_complex,
            translate_along_z_m2m_complex_batch,
            5,
            -1.5,
            1.5,
        ),
        (
            translate_along_z_l2l_complex,
            translate_along_z_l2l_complex_batch,
            6,
            -1.0,
            1.0,
        ),
    ],
)
def test_translate_along_z_complex_batch_matches_single(
    single_fn, batch_fn, seed, lo, hi
) -> None:
    order = 4
    rng = np.random.default_rng(seed)
    ncoeff = sh_size(order)
    batch = 3
    coeffs = rng.normal(size=(batch, ncoeff)) + 1j * rng.normal(size=(batch, ncoeff))
    scalars = rng.uniform(lo, hi, size=(batch,))

    ref = np.stack(
        [
            np.asarray(single_fn(jnp.asarray(m), jnp.asarray(rr), order=order))
            for m, rr in zip(coeffs, scalars)
        ],
        axis=0,
    )
    got = batch_fn(jnp.asarray(coeffs), jnp.asarray(scalars), order=order)

    assert np.allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)


def test_m2m_complex_matches_z_axis_translation() -> None:
    order = 4
    multipole = _complex_coeffs(order, 7)
    dz = 1.7
    delta = jnp.array([0.0, 0.0, dz], dtype=jnp.float64)

    ref = translate_along_z_m2m_complex(
        jnp.asarray(multipole), jnp.asarray(dz), order=order
    )
    got_solidfmm = m2m_complex(
        jnp.asarray(multipole), delta, order=order, rotation="solidfmm"
    )

    assert np.allclose(
        np.asarray(got_solidfmm), np.asarray(ref), rtol=1e-12, atol=1e-12
    )


def test_l2l_complex_matches_z_axis_translation() -> None:
    order = 4
    local = _complex_coeffs(order, 8)
    dz = -0.9
    delta = jnp.array([0.0, 0.0, dz], dtype=jnp.float64)

    ref = translate_along_z_l2l_complex(
        jnp.asarray(local), jnp.asarray(dz), order=order
    )
    got = l2l_complex(jnp.asarray(local), delta, order=order, rotation="solidfmm")

    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_l2l_complex_solidfmm_matches_direct_reference() -> None:
    order = 5
    local = _complex_coeffs(order, 12)
    delta = np.array([0.35, -0.25, 0.45], dtype=np.float64)

    ref = _l2l_reference_complex(local, delta, order)
    got = l2l_complex(
        jnp.asarray(local),
        jnp.asarray(delta),
        order=order,
        rotation="solidfmm",
    )

    assert np.allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)


def _pack_dense_reference(dense: np.ndarray, order: int) -> np.ndarray:
    flat = dense.reshape(-1)
    gather = []
    for nx, ny, nz in symmetric_multi_indices_3d(order):
        axis = (0,) * nx + (1,) * ny + (2,) * nz
        idx = 0
        for a in axis:
            idx = idx * 3 + a
        gather.append(idx)
    return flat[np.asarray(gather, dtype=np.int32)]


def test_evaluate_local_complex_derivative_tower_matches_grad_and_hessian() -> None:
    order = 4
    local = _complex_coeffs(order, 21)
    delta = jnp.array([0.31, -0.27, 0.58], dtype=jnp.float64)

    d0, d1, d2 = evaluate_local_complex_derivative_tower(
        jnp.asarray(local),
        delta,
        order=order,
        max_derivative_order=2,
    )

    grad_ref, pot_ref = evaluate_local_complex_with_grad(
        jnp.asarray(local),
        delta,
        order=order,
    )
    hessian_ref = jax.hessian(
        lambda d: complex_dot(
            jnp.asarray(local),
            complex_R_solidfmm(d, order=order),
            order=order,
            conjugate_left=True,
        ).real
    )(delta)
    d2_ref = _pack_dense_reference(np.asarray(hessian_ref), order=2)

    assert np.allclose(np.asarray(d0), np.asarray([pot_ref]), rtol=1e-12, atol=1e-12)
    assert np.allclose(np.asarray(d1), np.asarray(grad_ref), rtol=1e-12, atol=1e-12)
    assert np.allclose(np.asarray(d2), d2_ref, rtol=1e-12, atol=1e-12)


def test_evaluate_local_complex_derivative_tower_matches_third_order_autodiff() -> None:
    order = 5
    local = _complex_coeffs(order, 22)
    delta = jnp.array([0.19, -0.41, 0.63], dtype=jnp.float64)

    _, _, _, d3 = evaluate_local_complex_derivative_tower(
        jnp.asarray(local),
        delta,
        order=order,
        max_derivative_order=3,
    )

    phi = lambda d: complex_dot(
        jnp.asarray(local),
        complex_R_solidfmm(d, order=order),
        order=order,
        conjugate_left=True,
    ).real
    d3_dense = jax.jacfwd(jax.hessian(phi))(delta)
    d3_ref = _pack_dense_reference(np.asarray(d3_dense), order=3)
    assert np.allclose(np.asarray(d3), d3_ref, rtol=1e-11, atol=1e-11)


def test_contract_spatial_derivative_with_velocity_matches_hessian_times_v() -> None:
    # Packed Hessian layout: xx, xy, xz, yy, yz, zz
    hessian_packed = jnp.array([2.0, -1.0, 4.0, 3.0, -2.0, 5.0], dtype=jnp.float64)
    velocity = jnp.array([0.7, -1.1, 0.4], dtype=jnp.float64)

    got = contract_spatial_derivative_with_velocity(hessian_packed, velocity, order=2)
    expected = jnp.array(
        [
            2.0 * 0.7 + (-1.0) * (-1.1) + 4.0 * 0.4,
            (-1.0) * 0.7 + 3.0 * (-1.1) + (-2.0) * 0.4,
            4.0 * 0.7 + (-2.0) * (-1.1) + 5.0 * 0.4,
        ],
        dtype=jnp.float64,
    )
    assert jnp.allclose(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "deriv_fn, kwargs, eps, second_order, rtol, atol",
    [
        (regular_solid_harmonic_directional_derivative, {}, 1e-6, False, 1e-6, 1e-7),
        (
            regular_solid_harmonic_directional_derivative_order,
            {"derivative_order": 2},
            1e-5,
            True,
            4e-5,
            2e-6,
        ),
    ],
)
def test_regular_harmonic_directional_derivative_matches_finite_difference(
    deriv_fn, kwargs, eps, second_order, rtol, atol
) -> None:
    order = 5
    delta = jnp.asarray([0.37, -0.22, 0.58], dtype=jnp.float64)
    direction = jnp.asarray([0.31, -0.44, 0.21], dtype=jnp.float64)
    eps = jnp.asarray(eps, dtype=jnp.float64)

    r_plus = complex_R_solidfmm(delta + eps * direction, order=order)
    r_minus = complex_R_solidfmm(delta - eps * direction, order=order)
    if second_order:
        r_zero = complex_R_solidfmm(delta, order=order)
        ref = (r_plus - 2.0 * r_zero + r_minus) / (eps * eps)
    else:
        ref = (r_plus - r_minus) / (2.0 * eps)
    got = deriv_fn(delta, direction, order=order, **kwargs)
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "single_fn, batch_fn, kwargs, seed",
    [
        (
            regular_solid_harmonic_directional_derivative,
            regular_solid_harmonic_directional_derivative_batch,
            {},
            17,
        ),
        (
            regular_solid_harmonic_directional_derivative_order,
            regular_solid_harmonic_directional_derivative_order_batch,
            {"derivative_order": 2},
            23,
        ),
    ],
)
def test_regular_harmonic_directional_derivative_batch_matches_single(
    single_fn, batch_fn, kwargs, seed
) -> None:
    order = 4
    rng = np.random.default_rng(seed)
    deltas = jnp.asarray(rng.normal(size=(6, 3)), dtype=jnp.float64)
    directions = jnp.asarray(rng.normal(size=(6, 3)), dtype=jnp.float64)
    got = batch_fn(deltas, directions, order=order, **kwargs)
    ref = jnp.stack(
        [single_fn(d, v, order=order, **kwargs) for d, v in zip(deltas, directions)],
        axis=0,
    )
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-12)
