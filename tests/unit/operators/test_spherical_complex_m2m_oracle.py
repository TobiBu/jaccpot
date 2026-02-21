import pytest

pytest.skip(
    "Complex-basis reference tests removed (real-only pipeline).",
    allow_module_level=True,
)

import jax
import jax.numpy as jnp

from jaccpot.operators.spherical_harmonics import _wigner_D_complex


def _slice_degree(ell: int) -> slice:
    return slice(ell * ell, (ell + 1) * (ell + 1))


def _unpack_blocks(vec: jnp.ndarray, order: int) -> list[jnp.ndarray]:
    return [vec[_slice_degree(ell)] for ell in range(order + 1)]


def _pack_blocks(blocks: list[jnp.ndarray]) -> jnp.ndarray:
    return jnp.concatenate(blocks, axis=0)


def _rotation_to_z_angles(
    delta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x, y, z = delta
    eps = jnp.asarray(1e-30, dtype=delta.dtype)
    rho = jnp.sqrt(jnp.maximum(x * x + y * y, eps))
    alpha = jnp.arctan2(y, x)
    beta = jnp.arctan2(rho, z)
    gamma = -alpha
    return alpha, beta, gamma


def _rotate_complex_blocks(
    blocks: list[jnp.ndarray],
    *,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    gamma: jnp.ndarray,
    use_inverse_angles: bool,
) -> list[jnp.ndarray]:
    """Rotate complex multipole blocks using D @ c.

    If use_inverse_angles=True, apply the inverse rotation by passing the
    inverse ZYZ angles (passive transform).
    """

    if use_inverse_angles:
        alpha, beta, gamma = (-gamma, -beta, -alpha)

    out = []
    for ell, blk in enumerate(blocks):
        D = _wigner_D_complex(
            ell,
            alpha,
            beta,
            gamma,
            dtype=jnp.dtype(jnp.complex128),
        )
        out.append(D @ blk)
    return out


def translate_z_m2m_oracle(
    blocks: list[jnp.ndarray],
    r: jnp.ndarray,
) -> list[jnp.ndarray]:
    """Z-translation oracle for complex multipoles.

    This uses the same series currently used in our code path:
            M'_{ell,m} = sum_k r^k/k! * M_{ell-k,m}

    It is intentionally small-order and slow.
    """

    p = len(blocks) - 1
    rdtype = jnp.float64
    r = jnp.asarray(r, dtype=rdtype)

    # Precompute factorials 0..2p
    n = jnp.arange(0, 2 * p + 1, dtype=rdtype)
    fact = jnp.exp(jax.lax.lgamma(n + 1.0))

    out = []
    for ell in range(p + 1):
        o = jnp.zeros((2 * ell + 1,), dtype=jnp.complex128)
        for m in range(-ell, ell + 1):
            mm = abs(m)
            acc = 0.0 + 0.0j
            for k in range(0, ell + 1):
                src_ell = ell - k
                if mm > src_ell:
                    continue
                src = blocks[src_ell][m + src_ell]
                # For the Dehnen-style Υ normalization used in the codebase
                # the specialised z-translation polynomial is r^k / k!.
                acc = acc + (r**k) * src / fact[k]
            o = o.at[m + ell].set(acc)
        out.append(o)
    return out


def m2m_a6_complex_oracle(
    vec: jnp.ndarray,
    delta: jnp.ndarray,
    order: int,
) -> jnp.ndarray:
    """A6 oracle in complex basis.

    This mirrors the *production* A6 composition:
      rotate (forward) -> zshift -> rotate back (inverse).
    """

    blocks = _unpack_blocks(vec, order)
    alpha, beta, gamma = _rotation_to_z_angles(delta)
    r = jnp.linalg.norm(delta)

    # Rotate into z-aligned frame (forward angles).
    rot_in = _rotate_complex_blocks(
        blocks,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        use_inverse_angles=False,
    )
    shifted = translate_z_m2m_oracle(rot_in, r)

    # Back to original frame (inverse angles).
    rot_out = _rotate_complex_blocks(
        shifted,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        use_inverse_angles=True,
    )
    return _pack_blocks(rot_out)


def test_a6_oracle_is_identity_for_zero_delta():
    """Sanity: translating by 0 should be the identity map."""

    p = 4
    n = (p + 1) * (p + 1)

    key = jax.random.key(0)
    v_re = jax.random.normal(key, (n,), dtype=jnp.float64)
    v_im = jax.random.normal(key, (n,), dtype=jnp.float64)
    v = (v_re + 1j * v_im).astype(jnp.complex128)

    delta0 = jnp.zeros((3,), dtype=jnp.float64)
    v1 = m2m_a6_complex_oracle(v, delta0, p)
    assert jnp.allclose(v1, v, atol=1e-12, rtol=1e-12)


def test_a6_oracle_roundtrip_is_small_for_tiny_delta():
    """Truncated translations aren't exactly invertible.

    The roundtrip error should be small for tiny |delta|.
    """

    p = 4
    n = (p + 1) * (p + 1)

    key = jax.random.key(1)
    v_re = jax.random.normal(key, (n,), dtype=jnp.float64)
    v_im = jax.random.normal(key, (n,), dtype=jnp.float64)
    v = (v_re + 1j * v_im).astype(jnp.complex128)

    delta = jnp.array([1e-3, -2e-3, 1.5e-3], dtype=jnp.float64)
    v1 = m2m_a6_complex_oracle(v, delta, p)
    v2 = m2m_a6_complex_oracle(v1, -delta, p)

    rel = jnp.linalg.norm(v2 - v) / jnp.linalg.norm(v)
    # NOTE: Even for tiny |delta|, composing +delta then -delta is not exact at
    # fixed truncation order p because intermediates couple into degrees > p.
    assert rel < 1e-5
