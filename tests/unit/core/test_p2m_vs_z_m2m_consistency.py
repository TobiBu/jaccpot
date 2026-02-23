import jax.numpy as jnp
import pytest

from jaccpot.operators.spherical_harmonics import (
    p2m_point_real_sh,
    translate_along_z_m2m,
)

"""@pytest.mark.xfail(
    reason=(
        "Known mismatch: current p2m_point_real_sh coefficients are not yet "
        "in the same convention as translate_along_z_m2m. This is the next "
        "thing to reconcile."
    ),
    strict=False,
)
@pytest.mark.xfail(
    reason=(
        "Known mismatch: current p2m_point_real_sh coefficients are not yet "
        "in the same convention as translate_along_z_m2m. This is the next "
        "thing to reconcile."
    ),
    strict=False,
)"""


def test_p2m_matches_z_m2m_for_z_axis_displacement():
    """For a point on the z-axis, M2M only needs z-translation.

    This isolates the P2M coefficient convention from any rotation/A6 logic.
    """

    p = 4
    mass = jnp.asarray(1.3, dtype=jnp.float64)

    # Source point location (relative to source expansion center).
    z = jnp.asarray(0.7, dtype=jnp.float64)
    pos = jnp.array([0.0, 0.0, z], dtype=jnp.float64)

    # Translate expansion center by +s along z: dest = src + (0,0,s)
    s = jnp.asarray(0.2, dtype=jnp.float64)

    # Direct P2M about source center (at origin) and about destination center.
    M_src = p2m_point_real_sh(pos, mass, order=p)
    M_dest_direct = p2m_point_real_sh(
        pos + jnp.array([0.0, 0.0, s]),
        mass,
        order=p,
    )

    # M2M: shift multipole center by +s along z.
    M_dest_via_m2m = translate_along_z_m2m(M_src, s, order=p)

    # Print per-degree m=0 coefficients for both methods to diagnose normalization.
    print("l | direct   | via_m2m")
    for ell in range(p + 1):
        idx = ell * ell + ell  # m=0 index in packed layout
        print(f"{ell} | {M_dest_direct[idx]: .8f} | {M_dest_via_m2m[idx]: .8f}")

    assert jnp.allclose(M_dest_via_m2m, M_dest_direct, atol=1e-10, rtol=1e-10)
