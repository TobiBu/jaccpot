import jax
import jax.numpy as jnp

from jaccpot import differentiable_gravitational_acceleration


def _direct_sum_accelerations(positions, masses, *, softening, G):
    diffs = positions[:, None, :] - positions[None, :, :]
    dist2 = jnp.sum(diffs * diffs, axis=-1) + softening**2
    inv_dist = jnp.where(dist2 > 0, dist2**-0.5, 0.0)
    inv_dist3 = inv_dist**3
    weights = masses[None, :] * inv_dist3
    # remove self-interaction contributions explicitly
    weights = weights * (1.0 - jnp.eye(positions.shape[0]))
    acc = -G * jnp.einsum("ij,ijk->ik", weights, diffs)
    return acc


def test_autodiff_matches_direct_sum():
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, -0.4, 0.1],
            [-0.5, 0.2, 0.6],
            [0.7, 0.1, -0.3],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.array([1.0, 0.8, 1.2, 0.5], dtype=jnp.float64)
    probe = jnp.linspace(0.1, 0.4, positions.size).reshape(positions.shape)
    softening = 1e-2
    G = 2.0

    def fmm_loss(pos):
        acc = differentiable_gravitational_acceleration(
            pos,
            masses,
            softening=softening,
            G=G,
            theta=0.6,
            max_order=2,
        )
        return jnp.sum(acc * probe)

    def dense_loss(pos):
        acc = _direct_sum_accelerations(pos, masses, softening=softening, G=G)
        return jnp.sum(acc * probe)

    grad_fmm = jax.grad(fmm_loss)(positions)
    grad_dense = jax.grad(dense_loss)(positions)
    assert jnp.allclose(grad_fmm, grad_dense, rtol=5e-4, atol=5e-4)


def test_autodiff_repeated_calls_are_stable():
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.1, -0.2],
            [-0.3, 0.4, 0.5],
        ],
        dtype=jnp.float32,
    )
    masses = jnp.array([1.0, 0.6, 0.9], dtype=jnp.float32)

    acc1 = differentiable_gravitational_acceleration(
        positions,
        masses,
        theta=0.5,
        max_order=2,
    )
    acc2 = differentiable_gravitational_acceleration(
        positions,
        masses,
        theta=0.5,
        max_order=2,
    )
    acc3 = differentiable_gravitational_acceleration(
        positions,
        masses,
        theta=0.7,
        max_order=2,
    )
    assert acc1.shape == positions.shape
    assert jnp.allclose(acc1, acc2, rtol=1e-5, atol=1e-5)
    assert acc3.shape == positions.shape
