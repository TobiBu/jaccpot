import math

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.complex_harmonics import complex_R_solidfmm
from jaccpot.operators.complex_ops import (
    complex_dot,
    l2l_complex,
    m2m_complex,
    rotate_complex_local_from_z,
    rotate_complex_local_from_z_batch,
    rotate_complex_local_from_z_cached,
    rotate_complex_local_to_z,
    rotate_complex_local_to_z_batch,
    rotate_complex_local_to_z_cached,
    rotate_complex_multipole_from_z,
    rotate_complex_multipole_from_z_batch,
    rotate_complex_multipole_from_z_cached,
    rotate_complex_multipole_to_z,
    rotate_complex_multipole_to_z_batch,
    rotate_complex_multipole_to_z_cached,
    translate_along_z_l2l_complex,
    translate_along_z_l2l_complex_batch,
    translate_along_z_m2l_complex,
    translate_along_z_m2l_complex_batch,
    translate_along_z_m2m_complex,
    translate_along_z_m2m_complex_batch,
    wigner_D_complex_jax,
)
from jaccpot.operators.real_harmonics import sh_size
from jaccpot.operators.solidfmm_reference import (
    translate_along_z_m2l_complex as ref_translate_z,
)


def test_translate_along_z_m2l_complex_matches_reference() -> None:
    order = 5
    rng = np.random.default_rng(0)
    ncoeff = sh_size(order)
    multipole = rng.normal(size=(ncoeff,)) + 1j * rng.normal(size=(ncoeff,))
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
    rng = np.random.default_rng(2)
    ncoeff = sh_size(order)
    multipole = rng.normal(size=(ncoeff,)) + 1j * rng.normal(size=(ncoeff,))
    dz = 0.37

    ref = _translate_z_m2m_reference(multipole, dz, order)
    got = translate_along_z_m2m_complex(
        jnp.asarray(multipole), jnp.asarray(dz), order=order
    )

    assert np.allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)


def test_translate_along_z_l2l_complex_matches_reference() -> None:
    order = 4
    rng = np.random.default_rng(3)
    ncoeff = sh_size(order)
    local = rng.normal(size=(ncoeff,)) + 1j * rng.normal(size=(ncoeff,))
    dz = -0.21

    ref = _translate_z_l2l_reference(local, dz, order)
    got = translate_along_z_l2l_complex(
        jnp.asarray(local), jnp.asarray(dz), order=order
    )

    assert np.allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)


def test_translate_along_z_m2l_complex_batch_matches_single() -> None:
    order = 4
    rng = np.random.default_rng(4)
    ncoeff = sh_size(order)
    batch = 3
    multipoles = rng.normal(size=(batch, ncoeff)) + 1j * rng.normal(
        size=(batch, ncoeff)
    )
    r = rng.uniform(0.5, 2.5, size=(batch,))

    ref = np.stack(
        [
            np.asarray(
                translate_along_z_m2l_complex(
                    jnp.asarray(m), jnp.asarray(rr), order=order
                )
            )
            for m, rr in zip(multipoles, r)
        ],
        axis=0,
    )
    got = translate_along_z_m2l_complex_batch(
        jnp.asarray(multipoles), jnp.asarray(r), order=order
    )

    assert np.allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)


def test_translate_along_z_m2m_complex_batch_matches_single() -> None:
    order = 4
    rng = np.random.default_rng(5)
    ncoeff = sh_size(order)
    batch = 3
    multipoles = rng.normal(size=(batch, ncoeff)) + 1j * rng.normal(
        size=(batch, ncoeff)
    )
    dz = rng.uniform(-1.5, 1.5, size=(batch,))

    ref = np.stack(
        [
            np.asarray(
                translate_along_z_m2m_complex(
                    jnp.asarray(m), jnp.asarray(rr), order=order
                )
            )
            for m, rr in zip(multipoles, dz)
        ],
        axis=0,
    )
    got = translate_along_z_m2m_complex_batch(
        jnp.asarray(multipoles), jnp.asarray(dz), order=order
    )

    assert np.allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)


def test_translate_along_z_l2l_complex_batch_matches_single() -> None:
    order = 4
    rng = np.random.default_rng(6)
    ncoeff = sh_size(order)
    batch = 3
    locals_ = rng.normal(size=(batch, ncoeff)) + 1j * rng.normal(size=(batch, ncoeff))
    dz = rng.uniform(-1.0, 1.0, size=(batch,))

    ref = np.stack(
        [
            np.asarray(
                translate_along_z_l2l_complex(
                    jnp.asarray(m), jnp.asarray(rr), order=order
                )
            )
            for m, rr in zip(locals_, dz)
        ],
        axis=0,
    )
    got = translate_along_z_l2l_complex_batch(
        jnp.asarray(locals_), jnp.asarray(dz), order=order
    )

    assert np.allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)


def test_m2m_complex_matches_z_axis_translation() -> None:
    order = 4
    rng = np.random.default_rng(7)
    ncoeff = sh_size(order)
    multipole = rng.normal(size=(ncoeff,)) + 1j * rng.normal(size=(ncoeff,))
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
    rng = np.random.default_rng(8)
    ncoeff = sh_size(order)
    local = rng.normal(size=(ncoeff,)) + 1j * rng.normal(size=(ncoeff,))
    dz = -0.9
    delta = jnp.array([0.0, 0.0, dz], dtype=jnp.float64)

    ref = translate_along_z_l2l_complex(
        jnp.asarray(local), jnp.asarray(dz), order=order
    )
    got = l2l_complex(jnp.asarray(local), delta, order=order)
    got_wigner = l2l_complex(jnp.asarray(local), delta, order=order, rotation="wigner")

    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-12)
    assert np.allclose(np.asarray(got_wigner), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_l2l_complex_solidfmm_matches_direct_reference() -> None:
    order = 5
    rng = np.random.default_rng(12)
    ncoeff = sh_size(order)
    local = rng.normal(size=(ncoeff,)) + 1j * rng.normal(size=(ncoeff,))
    delta = np.array([0.35, -0.25, 0.45], dtype=np.float64)

    ref = _l2l_reference_complex(local, delta, order)
    got = l2l_complex(
        jnp.asarray(local),
        jnp.asarray(delta),
        order=order,
        rotation="solidfmm",
    )

    assert np.allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)


def test_cached_rotation_blocks_match_direct_multipole() -> None:
    order = 4
    rng = np.random.default_rng(9)
    ncoeff = sh_size(order)
    multipole = rng.normal(size=(ncoeff,)) + 1j * rng.normal(size=(ncoeff,))
    delta = jnp.array([0.3, -0.2, 0.7], dtype=jnp.float64)

    ref_to = rotate_complex_multipole_to_z(jnp.asarray(multipole), delta, order=order)
    got_to = rotate_complex_multipole_to_z_cached(
        jnp.asarray(multipole), delta, order=order
    )
    assert np.allclose(np.asarray(got_to), np.asarray(ref_to), rtol=1e-12, atol=1e-12)

    ref_from = rotate_complex_multipole_from_z(
        jnp.asarray(multipole), delta, order=order
    )
    got_from = rotate_complex_multipole_from_z_cached(
        jnp.asarray(multipole), delta, order=order
    )
    assert np.allclose(
        np.asarray(got_from), np.asarray(ref_from), rtol=1e-12, atol=1e-12
    )


def test_cached_rotation_blocks_match_direct_local() -> None:
    order = 4
    rng = np.random.default_rng(10)
    ncoeff = sh_size(order)
    local = rng.normal(size=(ncoeff,)) + 1j * rng.normal(size=(ncoeff,))
    delta = jnp.array([-0.4, 0.1, 1.2], dtype=jnp.float64)

    ref_to = rotate_complex_local_to_z(jnp.asarray(local), delta, order=order)
    got_to = rotate_complex_local_to_z_cached(jnp.asarray(local), delta, order=order)
    assert np.allclose(np.asarray(got_to), np.asarray(ref_to), rtol=1e-12, atol=1e-12)

    ref_from = rotate_complex_local_from_z(jnp.asarray(local), delta, order=order)
    got_from = rotate_complex_local_from_z_cached(
        jnp.asarray(local), delta, order=order
    )
    assert np.allclose(
        np.asarray(got_from), np.asarray(ref_from), rtol=1e-12, atol=1e-12
    )


def test_batched_rotation_matches_single() -> None:
    order = 3
    rng = np.random.default_rng(11)
    ncoeff = sh_size(order)
    batch = 4
    multipoles = rng.normal(size=(batch, ncoeff)) + 1j * rng.normal(
        size=(batch, ncoeff)
    )
    locals_ = rng.normal(size=(batch, ncoeff)) + 1j * rng.normal(size=(batch, ncoeff))
    deltas = rng.normal(size=(batch, 3))

    ref_m_to = np.stack(
        [
            np.asarray(
                rotate_complex_multipole_to_z_cached(
                    jnp.asarray(m), jnp.asarray(d), order=order
                )
            )
            for m, d in zip(multipoles, deltas)
        ],
        axis=0,
    )
    got_m_to = rotate_complex_multipole_to_z_batch(
        jnp.asarray(multipoles), jnp.asarray(deltas), order=order
    )
    assert np.allclose(np.asarray(got_m_to), ref_m_to, rtol=1e-12, atol=1e-12)

    ref_m_from = np.stack(
        [
            np.asarray(
                rotate_complex_multipole_from_z_cached(
                    jnp.asarray(m), jnp.asarray(d), order=order
                )
            )
            for m, d in zip(multipoles, deltas)
        ],
        axis=0,
    )
    got_m_from = rotate_complex_multipole_from_z_batch(
        jnp.asarray(multipoles), jnp.asarray(deltas), order=order
    )
    assert np.allclose(np.asarray(got_m_from), ref_m_from, rtol=1e-12, atol=1e-12)

    ref_l_to = np.stack(
        [
            np.asarray(
                rotate_complex_local_to_z_cached(
                    jnp.asarray(m), jnp.asarray(d), order=order
                )
            )
            for m, d in zip(locals_, deltas)
        ],
        axis=0,
    )
    got_l_to = rotate_complex_local_to_z_batch(
        jnp.asarray(locals_), jnp.asarray(deltas), order=order
    )
    assert np.allclose(np.asarray(got_l_to), ref_l_to, rtol=1e-12, atol=1e-12)

    ref_l_from = np.stack(
        [
            np.asarray(
                rotate_complex_local_from_z_cached(
                    jnp.asarray(m), jnp.asarray(d), order=order
                )
            )
            for m, d in zip(locals_, deltas)
        ],
        axis=0,
    )
    got_l_from = rotate_complex_local_from_z_batch(
        jnp.asarray(locals_), jnp.asarray(deltas), order=order
    )
    assert np.allclose(np.asarray(got_l_from), ref_l_from, rtol=1e-12, atol=1e-12)


def test_wigner_d_jax_matches_sympy_small_ell() -> None:
    sympy = pytest.importorskip("sympy")
    from jaccpot.operators.real_harmonics import _wigner_D_complex as wigner_sympy

    alpha = 0.3
    beta = 0.7
    gamma = -0.2

    for ell in range(0, 4):
        D_ref = wigner_sympy(ell, alpha, beta, gamma)
        # Apply same no-Condon-Shortley adjustment used in real_harmonics
        m_vals = np.arange(-ell, ell + 1)
        S = np.diag(((-1.0) ** m_vals).astype(np.complex128))
        D_ref = S @ D_ref @ S

        D_jax = wigner_D_complex_jax(
            ell,
            jnp.asarray(alpha),
            jnp.asarray(beta),
            jnp.asarray(gamma),
            dtype=jnp.complex128,
            no_condon_shortley=True,
        )

        assert np.allclose(np.asarray(D_jax), D_ref, rtol=1e-10, atol=1e-10)
