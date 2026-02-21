import pytest

pytest.skip(
    "Wigner-D complex convention tests removed (real-only pipeline).",
    allow_module_level=True,
)

import jax.numpy as jnp

from jaccpot.operators.spherical_harmonics import _wigner_D_complex


def _expected_z_phases(ell: int, alpha: float, gamma: float) -> jnp.ndarray:
    """Expected diagonal of D^ell for a pure z-rotation (beta=0).

    For ZYZ with beta=0, the D-matrix is diagonal with entries
      exp(-i m alpha) * exp(-i m gamma) = exp(-i m (alpha + gamma))
    when using the common convention D_{m m'} = e^{-im alpha} d e^{-im' gamma}
    and d=I.
    """

    m = jnp.arange(-ell, ell + 1, dtype=jnp.float64)
    return jnp.exp(-1j * m * (alpha + gamma)).astype(jnp.complex128)


def test_wigner_D_is_diagonal_for_beta_zero():
    """Sanity: for beta=0, d^ell=I so D^ell must be diagonal."""

    alpha = 0.37
    gamma = -0.12
    beta = 0.0

    for ell in range(0, 7):
        D = _wigner_D_complex(
            ell,
            jnp.asarray(alpha),
            jnp.asarray(beta),
            jnp.asarray(gamma),
            dtype=jnp.dtype(jnp.complex128),
        )
        # Off-diagonal should be ~0.
        off = D - jnp.diag(jnp.diag(D))
        assert jnp.allclose(off, 0.0, atol=1e-12, rtol=1e-12)


def test_wigner_D_z_rotation_phase_matches_some_action():
    """Determine which matrix action matches the expected z-rotation phases.

    Depending on whether coefficient vectors are treated as column vectors of
    c_m or c_{m'}, and whether D is defined as D_{m m'} or D_{m' m}, the
    correct action on a coeff vector can be D@c, D.T@c, D^H@c, or D* @ c.

    This test asserts that for at least one of these four actions, applying it
    to a single basis vector e_m produces exactly the expected phase for that m
    and does not excite other m'.
    """

    ell = 5
    alpha = 0.41
    gamma = -0.23
    beta = 0.0

    D = _wigner_D_complex(
        ell,
        jnp.asarray(alpha),
        jnp.asarray(beta),
        jnp.asarray(gamma),
        dtype=jnp.dtype(jnp.complex128),
    )
    phases = _expected_z_phases(ell, alpha, gamma)

    actions = {
        "D": lambda M: M,
        "DT": lambda M: M.T,
        "DH": lambda M: jnp.conj(M).T,
        "Dconj": lambda M: jnp.conj(M),
    }

    matched = None
    for name, action in actions.items():
        A = action(D)
        for mi, m in enumerate(range(-ell, ell + 1)):
            e = jnp.zeros((2 * ell + 1,), dtype=jnp.complex128).at[mi].set(1.0)
            out = A @ e
            target = jnp.zeros_like(out).at[mi].set(phases[mi])
            if not jnp.allclose(out, target, atol=1e-12, rtol=1e-12):
                break
        else:
            matched = name
            break

    # This locks in the convention used by _wigner_D_complex.
    assert matched == "D"
