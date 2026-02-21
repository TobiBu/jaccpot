import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime.fmm import FastMultipoleMethod
from jaccpot.runtime.reference import direct_sum


@pytest.mark.parametrize("max_order", [2, 3])
def test_spherical_matches_direct_small_system(max_order: int):
    # Small random system to keep runtime low
    rng = np.random.default_rng(42)
    n = 12
    positions = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.1, 1.0, size=(n,)), dtype=jnp.float64)

    # Spherical FMM
    fmm = FastMultipoleMethod(expansion_basis="spherical", theta=0.7)
    state = fmm.prepare_state(positions, masses, max_order=max_order)
    acc_fmm = fmm.evaluate_prepared_state(state)

    # Direct reference
    acc_direct = jax.vmap(
        lambda p: direct_sum(
            positions,
            masses,
            p,
            G=fmm.G,
            softening=fmm.softening,
        )
    )(positions)

    # Modest tolerance: spherical path uses correctness-first collocation L2L
    assert jnp.allclose(acc_fmm, acc_direct, rtol=5e-3, atol=5e-3)


def test_spherical_potential_matches_direct():
    rng = np.random.default_rng(7)
    n = 10
    positions = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.2, 1.2, size=(n,)), dtype=jnp.float64)

    fmm = FastMultipoleMethod(expansion_basis="spherical", theta=0.7)
    state = fmm.prepare_state(positions, masses, max_order=3)
    acc_fmm, pot_fmm = fmm.evaluate_prepared_state(
        state,
        return_potential=True,
    )

    acc_direct = jax.vmap(
        lambda p: direct_sum(
            positions,
            masses,
            p,
            G=fmm.G,
            softening=fmm.softening,
        )
    )(positions)
    # Direct potentials excluding self-interaction

    def potentials_excluding_self(pos, mass, G, soft):
        r_vec = pos[:, None, :] - pos[None, :, :]
        r = jnp.sqrt(jnp.sum(r_vec**2, axis=-1) + soft**2)
        mask = ~jnp.eye(pos.shape[0], dtype=bool)
        contrib = jnp.where(mask, mass[None, :] / (r + 1e-10), 0.0)
        return -G * jnp.sum(contrib, axis=1)

    pot_direct = potentials_excluding_self(
        positions,
        masses,
        fmm.G,
        fmm.softening,
    )

    assert jnp.allclose(acc_fmm, acc_direct, rtol=5e-3, atol=5e-3)
    assert jnp.allclose(pot_fmm, pot_direct, rtol=5e-3, atol=5e-3)
