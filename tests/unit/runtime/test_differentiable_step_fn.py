"""The jittable step seam for differentiating through the force.

``differentiable_accelerations`` re-traces its whole pipeline on every eager
call, and the re-tracing dominates. Measured on an idle A100 at N=4096, leaf 32,
p=4, real basis, ``preset="accurate"``, fp64: 5.893 s eager against 0.0224 s
through this seam after a 19.1 s one-time compile -- **264x** -- with the forward
agreeing to 3.0e-16 rel-L2 and the gradient to 3.0e-17 (positions) and 7.5e-17
(masses) against the eager gradient. The forward+backward ratio is 1.82x, which
is the bounded multiple figure 12 reports.

The old advice ("wrap the whole call in jax.jit") failed in a way no
``try``/``except`` could catch: at N=256 on an A100 the compile had not finished
after 18 minutes, so the fallback never fired because slowness is not an
exception. These tests are small and run anywhere; the seam's job is to make the
compile a one-time cost you pay deliberately, at a point you chose.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaccpot import FastMultipoleMethod

N = 512
LEAF = 32
ORDER = 3
THETA = 0.5


@pytest.fixture(scope="module")
def setup():
    key = jax.random.PRNGKey(N)
    k1, k2, k3 = jax.random.split(key, 3)
    positions = jax.random.normal(k1, (N, 3), dtype=jnp.float64)
    masses = jax.random.uniform(k2, (N,), minval=0.5, maxval=1.5, dtype=jnp.float64)
    probe = jax.random.normal(k3, (N, 3), dtype=jnp.float64)
    solver = FastMultipoleMethod(
        preset="fast", basis="real", theta=THETA, softening=1e-2
    )
    state = solver.prepare_state(
        positions, masses, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    return solver, state, positions, masses, probe


def test_it_returns_the_same_force_as_the_eager_entry_point(setup) -> None:
    solver, state, positions, masses, _ = setup
    step = solver.differentiable_step_fn(state)
    eager = solver.differentiable_accelerations(state, positions, masses)
    got = step(positions, masses)
    assert jnp.allclose(got, eager, rtol=1e-12, atol=0.0)


def test_it_stays_differentiable(setup) -> None:
    """The point of the seam. An AOT-compiled object would fail here.

    Returning ``jit(f).lower(...).compile()`` instead of ``jit(f)`` raises
    "Cannot apply JAX transformations to a function lowered and compiled for a
    particular signature" -- silently turning the seam into a dead end for the
    training loops it exists to serve.
    """

    solver, state, positions, masses, probe = setup
    step = solver.differentiable_step_fn(state)

    def loss(pos, mass):
        return jnp.sum(probe * step(pos, mass))

    grad_pos, grad_mass = jax.grad(loss, argnums=(0, 1))(positions, masses)
    assert grad_pos.shape == positions.shape
    assert grad_mass.shape == masses.shape
    assert bool(jnp.all(jnp.isfinite(grad_pos)))
    assert bool(jnp.all(jnp.isfinite(grad_mass)))


def test_the_gradient_matches_the_eager_gradient(setup) -> None:
    """Faster is worthless if it is a different derivative."""

    solver, state, positions, masses, probe = setup
    step = solver.differentiable_step_fn(state)

    def loss_seam(pos, mass):
        return jnp.sum(probe * step(pos, mass))

    def loss_eager(pos, mass):
        return jnp.sum(probe * solver.differentiable_accelerations(state, pos, mass))

    seam = jax.grad(loss_seam, argnums=(0, 1))(positions, masses)
    eager = jax.grad(loss_eager, argnums=(0, 1))(positions, masses)
    for got, want in zip(seam, eager):
        assert jnp.allclose(got, want, rtol=1e-10, atol=0.0)


def test_compile_now_does_not_change_the_answer(setup) -> None:
    """Warming the cache is not allowed to change the result.

    Compared to a tolerance rather than bit-for-bit: the near field and M2L
    accumulate via scatter-add, which XLA lowers to atomics on a GPU, so two runs
    of the SAME graph differ by a few ulps (measured 0 of 7 repeats bit-identical
    on an A100, worst 3.8 eps). A bit-equality assertion here passes on CPU and
    fails on GPU -- which is exactly what it did before this comment existed.
    """

    solver, state, positions, masses, _ = setup
    warm = solver.differentiable_step_fn(state, compile_now=(positions, masses))
    cold = solver.differentiable_step_fn(state)
    assert jnp.allclose(
        warm(positions, masses), cold(positions, masses), rtol=1e-12, atol=0.0
    )


def test_compile_now_still_returns_something_differentiable(setup) -> None:
    """The specific regression: warming must not hand back an AOT artifact."""

    solver, state, positions, masses, probe = setup
    step = solver.differentiable_step_fn(state, compile_now=(positions, masses))
    grad = jax.grad(lambda p: jnp.sum(probe * step(p, masses)))(positions)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_a_traced_state_fails_fast_and_says_why(setup) -> None:
    """Rather than surfacing later as a leaked-tracer error naming a cache.

    Tree construction is host-side and not traceable, so a state built inside a
    transform cannot be a compile-time constant. The old failure mode was an
    error deep in the trace naming an internal stateful cache, which gave the
    caller nothing to act on.
    """

    solver, state, positions, _masses, _probe = setup

    @jax.jit
    def ask_inside_a_trace(pos):
        # Substituting the traced positions into the state is exactly what a
        # caller doing this by accident ends up with.
        traced = jax.tree_util.tree_map(lambda leaf: pos.sum() + 0 * leaf, state)
        return solver.differentiable_step_fn(traced)

    with pytest.raises(TypeError, match="concrete prepared state"):
        ask_inside_a_trace(positions)
