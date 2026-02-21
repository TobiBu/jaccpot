import jax.numpy as jnp

from jaccpot.operators.spherical_harmonics import rotation_to_z


def _Rz(a: jnp.ndarray) -> jnp.ndarray:
    ca, sa = jnp.cos(a), jnp.sin(a)
    return jnp.array(
        [
            [ca, -sa, 0.0],
            [sa, ca, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=a.dtype,
    )


def _Ry(b: jnp.ndarray) -> jnp.ndarray:
    cb, sb = jnp.cos(b), jnp.sin(b)
    return jnp.array(
        [
            [cb, 0.0, sb],
            [0.0, 1.0, 0.0],
            [-sb, 0.0, cb],
        ],
        dtype=b.dtype,
    )


def _zyz_matrix(
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    gamma: jnp.ndarray,
) -> jnp.ndarray:
    """Active ZYZ rotation matrix R = Rz(alpha) Ry(beta) Rz(gamma)."""

    return _Rz(alpha) @ _Ry(beta) @ _Rz(gamma)


def test_rotation_to_z_aligns_delta_under_passive_convention():
    """rotation_to_z(delta) must align delta with +z.

    We treat rotation_to_z(delta) as defining a *frame rotation* that makes the
    displacement lie on +z. In terms of an active rotation matrix R built from
    the returned ZYZ angles, the corresponding passive coordinate transform is
    applying R^{-1} to vectors.

    Therefore we must have:
      R^{-1} delta == (0, 0, ||delta||)
    """

    delta = jnp.array([0.23, -0.11, 0.47], dtype=jnp.float64)
    rot = rotation_to_z(delta)
    R = _zyz_matrix(rot.alpha, rot.beta, rot.gamma)

    d_passive = R.T @ delta  # R^{-1} for an orthogonal rotation matrix
    r = jnp.linalg.norm(delta)

    # x,y should be ~0, z should be +||delta||.
    assert jnp.allclose(d_passive[0], 0.0, atol=1e-12, rtol=1e-12)
    assert jnp.allclose(d_passive[1], 0.0, atol=1e-12, rtol=1e-12)
    assert jnp.allclose(d_passive[2], r, atol=1e-12, rtol=1e-12)
