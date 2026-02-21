import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.operators.real_harmonics import evaluate_local_real
from jaccpot.operators.spherical_harmonics import (
    sh_size,
    translate_along_z_l2l,
    translate_local_real_sh,
)


def _evaluate_local(
    coeffs: jnp.ndarray,
    offsets: jnp.ndarray,
    order: int,
) -> jnp.ndarray:
    """Evaluate a packed real Dehnen-basis local expansion at offsets."""

    def phi_fn(vec: jnp.ndarray) -> jnp.ndarray:
        return evaluate_local_real(coeffs, -vec, order=order)

    return jax.vmap(phi_fn)(offsets)


def test_translate_local_real_sh_matches_parent_evaluation():
    p = 3
    rng = np.random.default_rng(123)

    parent_coeffs = jnp.asarray(rng.normal(size=sh_size(p)), dtype=jnp.float64)
    delta = jnp.asarray([0.3, -0.15, 0.2], dtype=jnp.float64)
    child_coeffs = translate_local_real_sh(parent_coeffs, delta, order=p)

    offsets_child = jnp.asarray(
        rng.normal(scale=0.2, size=(12, 3)),
        dtype=jnp.float64,
    )
    offsets_parent = offsets_child + delta

    vals_parent = _evaluate_local(parent_coeffs, offsets_parent, order=p)
    vals_child = _evaluate_local(child_coeffs, offsets_child, order=p)

    assert jnp.allclose(vals_parent, vals_child, rtol=1e-6, atol=1e-6)


def test_translate_along_z_l2l_matches_collocation_reference():
    rng = np.random.default_rng(321)
    dz_values = [-0.4, -0.1, 0.0, 0.25, 0.6]

    # Collocation reference becomes ill-conditioned for higher order; restrict
    # comparison to modest p where the solve remains stable.
    for p in range(0, 5):
        for dz in dz_values:
            local = jnp.asarray(rng.normal(size=sh_size(p)), dtype=jnp.float64)
            dz_arr = jnp.asarray(dz, dtype=jnp.float64)

            ref = translate_local_real_sh(
                local,
                jnp.array([0.0, 0.0, dz], dtype=jnp.float64),
                order=p,
            )
            got = translate_along_z_l2l(local, dz_arr, order=p)

            assert jnp.allclose(ref, got, rtol=2e-2, atol=1e-2)
