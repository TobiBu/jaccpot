import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import FastMultipoleMethod, OdisseoFMMCoupler


def _sample_state(n: int = 48):
    key = jax.random.PRNGKey(17)
    key_pos, key_vel, key_mass = jax.random.split(key, 3)
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    velocities = jax.random.normal(key_vel, (n, 3), dtype=jnp.float32) * 0.1
    masses = jax.random.uniform(
        key_mass,
        (n,),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.float32,
    )
    state = jnp.stack((positions, velocities), axis=1)
    return state, masses


def test_coupler_prepare_and_full_accelerations():
    state, masses = _sample_state()
    solver = FastMultipoleMethod(preset="fast", basis="solidfmm")
    coupler = OdisseoFMMCoupler(solver=solver, leaf_size=16, max_order=3)
    coupler.prepare(state, masses)
    acc = coupler.accelerations(state)
    assert acc.shape == (state.shape[0], 3)
    assert np.isfinite(np.asarray(acc)).all()


def test_coupler_active_subset_matches_solver_prepared_subset():
    state, masses = _sample_state(n=56)
    solver = FastMultipoleMethod(preset="fast", basis="solidfmm")
    coupler = OdisseoFMMCoupler(solver=solver, leaf_size=16, max_order=3)
    active = jnp.asarray([0, 3, 9, 22, 47], dtype=jnp.int32)

    coupler.prepare(state, masses)
    subset_acc = coupler.accelerations(state, active_indices=active)

    positions = state[:, 0, :]
    prepared = solver.prepare_state(positions, masses, leaf_size=16, max_order=3)
    expected = solver.evaluate_prepared_state(prepared, target_indices=active)

    assert subset_acc.shape == (active.shape[0], 3)
    assert np.allclose(
        np.asarray(subset_acc), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def test_coupler_requires_masses_on_first_call():
    state, _ = _sample_state(n=12)
    solver = FastMultipoleMethod(preset="fast", basis="solidfmm")
    coupler = OdisseoFMMCoupler(solver=solver)
    with pytest.raises(ValueError, match="masses must be provided"):
        coupler.accelerations(state)
