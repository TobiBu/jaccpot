"""Primary correctness gate for the differentiable FMM force.

The direct O(N^2) sum (:func:`differentiable_gravitational_acceleration`) is
exactly differentiable, so ``jax.grad`` of the fixed-topology FMM
(:meth:`FastMultipoleMethod.differentiable_accelerations`) must agree with
``jax.grad`` of the direct sum to the FMM's own *force* accuracy -- not to
machine precision. This is the strongest cross-check that the reverse pass is
wired correctly (see ``docs/differentiable_fmm_design.md``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import FastMultipoleMethod


def _direct_sum_accelerations(positions, masses, *, softening, G):
    diffs = positions[:, None, :] - positions[None, :, :]
    dist2 = jnp.sum(diffs * diffs, axis=-1) + softening**2
    inv_dist = jnp.where(dist2 > 0, dist2**-0.5, 0.0)
    inv_dist3 = inv_dist**3
    weights = masses[None, :] * inv_dist3
    weights = weights * (1.0 - jnp.eye(positions.shape[0], dtype=positions.dtype))
    return -G * jnp.einsum("ij,ijk->ik", weights, diffs)


def _system(n, seed=0):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(n,)), dtype=jnp.float64)
    probe = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    return positions, masses, probe


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


# theta tight enough that the far field is accurate; both solidfmm submodes.
@pytest.mark.parametrize("basis", ["complex", "real"])
@pytest.mark.parametrize("theta,order,tol", [(0.3, 6, 5e-3), (0.5, 6, 2e-2)])
def test_grad_fmm_matches_grad_directsum_positions(basis, theta, order, tol):
    n = 128
    positions, masses, probe = _system(n)
    softening, G = 1e-2, 1.5
    fmm = FastMultipoleMethod(
        basis=basis, use_pallas=False, theta=theta, G=G, softening=softening
    )
    state = fmm.prepare_state(positions, masses, max_order=order, leaf_size=16)

    def fmm_loss(pos):
        return jnp.sum(fmm.differentiable_accelerations(state, pos, masses) * probe)

    def dense_loss(pos):
        return jnp.sum(
            _direct_sum_accelerations(pos, masses, softening=softening, G=G) * probe
        )

    g_fmm = jax.grad(fmm_loss)(positions)
    g_dense = jax.grad(dense_loss)(positions)
    assert jnp.all(jnp.isfinite(g_fmm))
    err = _rel_l2(g_fmm, g_dense)
    assert (
        err < tol
    ), f"position grad rel-L2 {err:.3e} exceeds {tol:.1e} (basis={basis}, theta={theta})"


@pytest.mark.parametrize("basis", ["complex", "real"])
def test_grad_fmm_matches_grad_directsum_masses(basis):
    n = 128
    positions, masses, probe = _system(n, seed=1)
    softening, G, theta, order = 1e-2, 1.5, 0.3, 6
    fmm = FastMultipoleMethod(
        basis=basis, use_pallas=False, theta=theta, G=G, softening=softening
    )
    state = fmm.prepare_state(positions, masses, max_order=order, leaf_size=16)

    def fmm_loss(m):
        return jnp.sum(fmm.differentiable_accelerations(state, positions, m) * probe)

    def dense_loss(m):
        return jnp.sum(
            _direct_sum_accelerations(positions, m, softening=softening, G=G) * probe
        )

    g_fmm = jax.grad(fmm_loss)(masses)
    g_dense = jax.grad(dense_loss)(masses)
    assert jnp.all(jnp.isfinite(g_fmm))
    err = _rel_l2(g_fmm, g_dense)
    assert err < 5e-3, f"mass grad rel-L2 {err:.3e} (basis={basis})"


def test_differentiable_matches_forward_compute_accelerations():
    """The differentiable path must reproduce the forward FMM force it differentiates."""
    n = 128
    positions, masses, _ = _system(n, seed=2)
    fmm = FastMultipoleMethod(
        basis="complex", use_pallas=False, theta=0.4, G=1.0, softening=1e-2
    )
    state = fmm.prepare_state(positions, masses, max_order=4, leaf_size=16)
    a_diff = fmm.differentiable_accelerations(state, positions, masses)
    a_fwd = fmm.compute_accelerations(positions, masses, max_order=4, theta=0.4)
    assert a_diff.shape == positions.shape
    assert _rel_l2(a_diff, a_fwd) < 1e-6
