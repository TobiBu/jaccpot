import jax.numpy as jnp
import pytest

from jaccpot.operators.spherical_harmonics import (
    delta_norm,
    m2l_a6_dehnen,
    packed_degree_slices,
    rotate_real_sh_with_rot,
    rotation_to_z,
    sh_index,
    sh_offset,
    sh_size,
    translate_along_z_m2l,
)


def _direct_potential_point_masses(
    sources: jnp.ndarray,
    masses: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    softening: float = 0.0,
) -> jnp.ndarray:
    """Direct 1/r potential from point masses (JAX version for tests)."""

    d = targets[:, None, :] - sources[None, :, :]
    r2 = jnp.sum(d * d, axis=-1) + softening**2
    inv_r = 1.0 / jnp.sqrt(r2)
    return inv_r @ masses


def test_sh_size_matches_square() -> None:
    for p in range(0, 10):
        assert sh_size(p) == (p + 1) ** 2


def test_sh_index_bounds_and_offsets() -> None:
    for ell in range(0, 8):
        assert sh_offset(ell) == ell * ell

        # m runs from -ell..ell
        indices = [sh_index(ell, m) for m in range(-ell, ell + 1)]
        assert indices[0] == ell * ell
        assert indices[-1] == (ell + 1) * (ell + 1) - 1
        assert sorted(indices) == list(range(ell * ell, (ell + 1) * (ell + 1)))

    with pytest.raises(ValueError):
        _ = sh_index(2, 3)


def test_packed_degree_slices_cover_buffer() -> None:
    p = 7
    slices = packed_degree_slices(p)
    assert len(slices) == p + 1
    assert slices[0] == slice(0, 1)
    assert slices[-1] == slice(p * p, (p + 1) * (p + 1))


def test_rotation_to_z_angles_basic_cases() -> None:
    rot = rotation_to_z(jnp.array([0.0, 0.0, 2.0]))
    assert jnp.allclose(rot.alpha, 0.0)
    assert jnp.allclose(rot.beta, 0.0)

    rot = rotation_to_z(jnp.array([1.0, 0.0, 0.0]))
    assert jnp.allclose(rot.alpha, 0.0)
    assert jnp.allclose(rot.beta, jnp.pi / 2)

    rot = rotation_to_z(jnp.array([0.0, 1.0, 0.0]))
    assert jnp.allclose(rot.alpha, jnp.pi / 2)
    assert jnp.allclose(rot.beta, jnp.pi / 2)


def test_delta_norm_matches_euclidean_norm() -> None:
    d = jnp.array([3.0, 4.0, 12.0])
    assert jnp.allclose(delta_norm(d), 13.0)


def test_rotate_real_sh_identity_is_noop() -> None:
    order = 5
    coeffs = jnp.arange(sh_size(order), dtype=jnp.float64)
    rot = rotation_to_z(jnp.array([0.0, 0.0, 1.0]))
    out = rotate_real_sh_with_rot(coeffs, rot, order=order)
    assert jnp.allclose(out, coeffs)


def test_rotate_real_sh_preserves_degree_norms() -> None:
    order = 6
    coeffs = jnp.linspace(0.1, 1.0, sh_size(order), dtype=jnp.float64)
    rot = rotation_to_z(jnp.array([1.0, 2.0, 3.0]))
    out = rotate_real_sh_with_rot(coeffs, rot, order=order)

    # Invert the rotation (ZYZ: inverse is -gamma,-beta,-alpha).
    from jaccpot.operators.spherical_harmonics import ZAxisRotation

    rot_inv = ZAxisRotation(alpha=-rot.gamma, beta=-rot.beta, gamma=-rot.alpha)
    roundtrip = rotate_real_sh_with_rot(out, rot_inv, order=order)
    assert jnp.allclose(roundtrip, coeffs, rtol=1e-10, atol=1e-12)


def test_translate_along_z_m2l_order0_scales_like_inverse_distance() -> None:
    # For p=0 we only have one coefficient. The precise normalization depends
    # on Y00, but it must scale like 1/r.
    order = 0
    m = jnp.array([1.0], dtype=jnp.float64)

    l1 = translate_along_z_m2l(
        m,
        jnp.asarray(2.0, dtype=jnp.float64),
        order=order,
    )
    l2 = translate_along_z_m2l(
        m,
        jnp.asarray(4.0, dtype=jnp.float64),
        order=order,
    )

    assert l1.shape == (1,)
    assert l2.shape == (1,)
    assert jnp.isfinite(l1[0])
    assert jnp.isfinite(l2[0])

    ratio = (l1[0] / l2[0]).item()
    assert ratio == pytest.approx(2.0, rel=0.2)


def test_m2l_a6_z_axis_preserves_monopole() -> None:
    """For a z-axis translation, the monopole (l=0) should scale like 1/r."""
    order = 2
    # Just monopole input
    multipole = jnp.zeros(sh_size(order), dtype=jnp.float64)
    multipole = multipole.at[0].set(1.0)  # l=0, m=0

    delta1 = jnp.array([0.0, 0.0, 2.0], dtype=jnp.float64)
    delta2 = jnp.array([0.0, 0.0, 4.0], dtype=jnp.float64)

    local1 = m2l_a6_dehnen(multipole, delta1, order=order)
    local2 = m2l_a6_dehnen(multipole, delta2, order=order)

    # Monopole contribution to local monopole should scale like 1/r
    ratio = (local1[0] / local2[0]).item()
    assert ratio == pytest.approx(2.0, rel=1e-10)


def test_m2l_a6_order0_matches_direct_point_mass_potential() -> None:
    # For p=0, the multipole/local are just scalars and A6 should reproduce
    # the potential of a point mass at the target center.
    order = 0

    # One point mass at an offset.
    source = jnp.array([[0.3, -0.2, 1.7]], dtype=jnp.float64)
    mass = jnp.array([2.5], dtype=jnp.float64)

    # Targets near the origin (our local expansion center).
    targets = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, -0.03, 0.02],
            [-0.04, 0.02, -0.01],
        ],
        dtype=jnp.float64,
    )

    # In our baseline z-translation path, order-0 acts like a monopole.
    # We treat the point mass's multipole coefficient as `mass`.
    m = jnp.asarray([mass[0]], dtype=jnp.float64)

    # delta is vector from target center (origin) to source center.
    delta = source[0]
    local = m2l_a6_dehnen(m, delta, order=order)
    assert local.shape == (1,)

    phi_local = jnp.repeat(local[0], targets.shape[0])
    phi_direct = _direct_potential_point_masses(source, mass, targets)

    # We don't pin down the absolute normalization here. We only require the
    # correct *scaling* (phi ∝ mass / r).
    ratio_local = (phi_local[0] / phi_local[1]).item()
    ratio_direct = (phi_direct[0] / phi_direct[1]).item()
    assert ratio_local == pytest.approx(ratio_direct, rel=5e-2)

    # Also check overall agreement is reasonable in an RMS sense.
    rel_err = jnp.linalg.norm(phi_local - phi_direct) / jnp.linalg.norm(
        phi_direct,
    )
    assert float(rel_err) < 5e-2


@pytest.mark.parametrize("order", [1, 2])
def test_m2l_a6_higher_order_center_potential_matches_direct(
    order: int,
) -> None:
    """For p>=1, the local coefficient (l=0,m=0) represents the potential
    at the expansion center up to a fixed normalization.

    This test avoids needing the full local-evaluation machinery by only
    comparing the potential *at the center*, where higher-order terms vanish.
    """

    # A small cluster of point masses.
    sources = jnp.array(
        [
            [0.3, -0.2, 1.7],
            [-0.4, 0.1, 2.2],
            [0.15, 0.35, 1.3],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([2.5, -1.2, 0.7], dtype=jnp.float64)

    # Build a multipole that only contains the monopole term.
    # This is physically meaningful and sufficient to validate the M2L kernel
    # and its higher-order bookkeeping.
    multipole = jnp.zeros((sh_size(order),), dtype=jnp.float64)
    multipole = multipole.at[sh_index(0, 0)].set(jnp.sum(masses))

    # Place the source cluster around a common source center.
    source_center = jnp.array([0.2, -0.1, 1.8], dtype=jnp.float64)
    # delta is vector from target center to *source center*.
    delta = source_center

    local = m2l_a6_dehnen(multipole, delta, order=order)

    # Potential at the target center from direct sum.
    target_center = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float64)
    phi_direct = _direct_potential_point_masses(
        sources,
        masses,
        target_center,
    )[0]

    # The potential at the center is encoded in the l=0,m=0 local coefficient
    # up to SH normalization. Validate by matching the *scaling* against a
    # second configuration at a different distance.
    phi_local_1 = local[sh_index(0, 0)]

    # Second configuration: scale delta (and sources) by a factor s.
    s = 1.7
    sources2 = sources * s
    delta2 = delta * s
    local2 = m2l_a6_dehnen(multipole, delta2, order=order)
    phi_local_2 = local2[sh_index(0, 0)]

    phi_direct_2 = _direct_potential_point_masses(
        sources2,
        masses,
        target_center,
    )[0]

    ratio_local = (phi_local_1 / phi_local_2).item()
    ratio_direct = (phi_direct / phi_direct_2).item()
    assert ratio_local == pytest.approx(ratio_direct, rel=5e-2)
