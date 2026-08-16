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
    m2l_complex_reference,
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


@pytest.mark.parametrize(
    "delta",
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.37], [0.0, 0.0, -0.37]],
    ids=["origin", "plus_z", "minus_z"],
)
@pytest.mark.parametrize("order", [1, 2, 4])
def test_evaluate_local_complex_grad_at_rho_zero_matches_limit(delta, order) -> None:
    """The complex L2P has no azimuthal degeneracy -- keep it that way.

    ``complex_R_solidfmm`` is a pure polynomial recursion in (x, y, z): it never
    forms an azimuth, so it is smooth at the expansion centre and on its z axis
    where the real-basis L2P used to lose both transverse gradient components.
    """
    local = jnp.asarray(_complex_coeffs(order, 23))
    delta = jnp.asarray(delta, dtype=jnp.float64)

    grad, pot = evaluate_local_complex_with_grad(local, delta, order=order)
    grad_off, pot_off = evaluate_local_complex_with_grad(
        local, delta + jnp.asarray([3e-15, -4e-15, 0.0]), order=order
    )

    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.allclose(np.asarray(grad), np.asarray(grad_off), rtol=1e-9, atol=1e-12)
    assert np.allclose(float(pot), float(pot_off), rtol=1e-9, atol=1e-12)
    assert abs(float(grad[0])) > 1e-6
    assert abs(float(grad[1])) > 1e-6


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


# The rotate -> z-translate -> rotate-back cascade at the degenerate separation
# rho == 0, where the solidfmm alignment azimuth is undefined. The complex-basis
# counterpart of
# tests/unit/operators/test_real_harmonics.py::test_rotation_cascade_transverse_gradient_at_rho_zero
# -- same defect (G.10), same fix, and worth asserting separately because the
# generators are built from this basis' own swap matrices rather than derived from
# the real ones.
_COMPLEX_CASCADE_OPERATORS = [
    pytest.param(m2l_complex_reference, id="m2l_complex_reference"),
    pytest.param(m2m_complex, id="m2m_complex"),
    pytest.param(l2l_complex, id="l2l_complex"),
]

_COMPLEX_CASCADE_ORDER = 4
_COMPLEX_CASCADE_Z = 2.5
# Probe radius for the rho -> 0 limit. NOT smaller: the polar route loses transverse
# gradient accuracy like eps / rho, so at rho = 1e-9 the eight-direction spread is
# round-off (4.6e-05) rather than the O(rho) structure being measured, and at 1e-7 it
# is 1.2e-06 to 2.4e-06 and scales linearly in rho as it should.
_COMPLEX_PROBE_RHO = 1.0e-7


def _complex_cascade_gradient(operator, delta):
    """``grad`` of a fixed-cotangent real scalar loss on ``operator`` w.r.t. ``delta``."""
    coeffs = jnp.asarray(
        _complex_coeffs(_COMPLEX_CASCADE_ORDER, 5), dtype=jnp.complex128
    )
    weights = jnp.asarray(
        _complex_coeffs(_COMPLEX_CASCADE_ORDER, 6), dtype=jnp.complex128
    )

    def loss(d):
        out = operator(coeffs, d, order=_COMPLEX_CASCADE_ORDER)
        return jnp.real(jnp.sum(weights * out))

    return np.asarray(
        jax.grad(loss)(jnp.asarray(delta, dtype=jnp.float64)), dtype=np.float64
    )


def _complex_cascade_offaxis_limit(operator, num_directions=8, rho=_COMPLEX_PROBE_RHO):
    """The ``rho -> 0`` gradient limit, averaged over approach directions.

    Averaging is only meaningful because the limit is direction-independent, which
    :func:`test_complex_cascade_gradient_limit_is_direction_independent` asserts
    separately rather than leaving to assumption.
    """
    grads = [
        _complex_cascade_gradient(
            operator,
            [
                rho * np.cos(2.0 * np.pi * k / num_directions),
                rho * np.sin(2.0 * np.pi * k / num_directions),
                _COMPLEX_CASCADE_Z,
            ],
        )
        for k in range(num_directions)
    ]
    return np.mean(np.array(grads), axis=0)


@pytest.mark.parametrize("operator", _COMPLEX_CASCADE_OPERATORS)
def test_complex_cascade_gradient_limit_is_direction_independent(operator) -> None:
    """The complex cascade is genuinely differentiable at ``rho == 0``.

    The premise the next test rests on: if the gradient limit depended on the approach
    direction there would be no derivative there, and a zero cotangent would be a
    defensible subgradient rather than a defect.

    The measured spread across eight directions at ``rho = 1e-7`` is 1.2e-06 (M2L) to
    2.4e-06 (L2L) against gradient components of 1.9 to 18.5, i.e. ~1.3e-07 relative,
    and it scales as ``O(rho)`` -- it is the genuine second-order variation of the
    gradient with position, not structure in the limit. The bound below is 1e-4
    relative: ~1000x above that measurement, and ~1e4 below the ``O(1)`` relative
    spread a genuinely direction-dependent limit would produce.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("gradient limits require float64 (JAX_ENABLE_X64=1)")

    grads = np.array(
        [
            _complex_cascade_gradient(
                operator,
                [
                    _COMPLEX_PROBE_RHO * np.cos(2.0 * np.pi * k / 8),
                    _COMPLEX_PROBE_RHO * np.sin(2.0 * np.pi * k / 8),
                    _COMPLEX_CASCADE_Z,
                ],
            )
            for k in range(8)
        ]
    )
    spread = float(np.max(grads.max(axis=0) - grads.min(axis=0)))
    scale = max(float(np.max(np.abs(grads))), 1.0)
    assert spread <= 1.0e-4 * scale, (
        "the rho -> 0 gradient limit must be direction-independent for the derivative "
        f"to exist; spread across 8 directions is {spread:.3e} against a gradient "
        f"scale of {scale:.3f}"
    )


@pytest.mark.parametrize("operator", _COMPLEX_CASCADE_OPERATORS)
def test_complex_cascade_transverse_gradient_at_rho_zero(operator) -> None:
    """``d/dx`` and ``d/dy`` at ``rho == 0`` must equal the off-axis limit.

    ``_angles_from_delta_solidfmm`` guards its azimuth at ``rho == 0`` and so returns a
    zero transverse cotangent; the cascade carries a ``custom_jvp`` that supplies the
    analytic derivative instead (G.10 -- see
    :mod:`jaccpot.operators._transverse_degeneracy_jvp`). Without it these two
    components come out as exactly 0.0 against limits of 1.9 to 18.5, so the test does
    not depend on the tolerance being tight; the measured agreement is <= 6e-08
    absolute, and the 1e-5 relative bound leaves the O(rho) offset of the probe radius
    plenty of room.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("gradient limits require float64 (JAX_ENABLE_X64=1)")

    at_zero = _complex_cascade_gradient(operator, [0.0, 0.0, _COMPLEX_CASCADE_Z])
    limit = _complex_cascade_offaxis_limit(operator)

    assert np.all(np.isfinite(at_zero)), f"gradient is not finite: {at_zero}"
    for component, axis in ((0, "x"), (1, "y"), (2, "z")):
        assert abs(at_zero[component] - limit[component]) <= 1.0e-5 * max(
            abs(limit[component]), 1.0
        ), (
            f"d/d{axis} at rho == 0 is {at_zero[component]:.9f}, but the off-axis "
            f"limit is {limit[component]:.9f}"
        )
