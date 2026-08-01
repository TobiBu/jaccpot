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


@pytest.fixture(scope="module")
def prepared_coupler():
    state, masses = _sample_state(n=56)
    solver = FastMultipoleMethod(preset="fast", basis="solidfmm")
    coupler = OdisseoFMMCoupler(solver=solver, leaf_size=16, max_order=3)
    coupler.prepare(state, masses)
    return coupler, state, masses


def test_coupler_prepare_and_full_accelerations(prepared_coupler):
    coupler, state, _ = prepared_coupler
    acc = coupler.accelerations(state)
    assert acc.shape == (state.shape[0], 3)
    assert np.isfinite(np.asarray(acc)).all()


def test_coupler_active_subset_matches_solver_prepared_subset(prepared_coupler):
    coupler, state, _ = prepared_coupler
    active = jnp.asarray([0, 3, 9, 22, 47], dtype=jnp.int32)

    subset_acc = coupler.accelerations(state, active_indices=active)

    expected = coupler.solver.evaluate_prepared_state(
        coupler._prepared_state,
        target_indices=active,
    )

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


def test_coupler_forward_path_rejects_traced_inputs(prepared_coupler):
    """The forward path evaluates PREBAKED expansions and reads no live input.

    Differentiating it therefore used to return exactly zero -- a silently wrong
    gradient, which is worse than any error. It must reject tracers instead.
    """
    coupler, state, masses = prepared_coupler

    with pytest.raises(NotImplementedError, match="differentiable=True"):
        jax.grad(lambda s: jnp.sum(coupler.accelerations(s, masses) ** 2))(state)


def test_coupler_differentiable_path_gives_nonzero_gradients(prepared_coupler):
    """``differentiable=True`` routes to the fixed-topology re-evaluation."""
    coupler, state, masses = prepared_coupler

    def loss(s, m):
        return jnp.sum(coupler.accelerations(s, m, differentiable=True) ** 2)

    grad_state, grad_masses = jax.grad(loss, argnums=(0, 1))(state, masses)

    assert jnp.all(jnp.isfinite(grad_state))
    assert jnp.all(jnp.isfinite(grad_masses))
    # Positions live in slot 0 and carry the whole sensitivity; velocities are
    # not an input to the force, so slot 1 must stay exactly zero.
    assert float(jnp.linalg.norm(grad_state[:, 0, :])) > 0.0
    assert float(jnp.linalg.norm(grad_masses)) > 0.0
    assert jnp.all(grad_state[:, 1, :] == 0)


def test_coupler_differentiable_path_matches_the_solver_entry_point(prepared_coupler):
    """The coupler is a thin adapter; it must not perturb the force it wraps."""
    coupler, state, masses = prepared_coupler

    via_coupler = coupler.accelerations(state, masses, differentiable=True)
    via_solver = coupler.solver.differentiable_accelerations(
        coupler._prepared_state, state[:, 0, :], masses
    )
    assert np.allclose(np.asarray(via_coupler), np.asarray(via_solver))


def test_coupler_differentiable_path_rejects_potential(prepared_coupler):
    coupler, state, masses = prepared_coupler
    with pytest.raises(NotImplementedError, match="acceleration-only"):
        coupler.accelerations(state, masses, differentiable=True, return_potential=True)
