"""Energy/angular-momentum conservation check for a long integration using
jaccpot forces -- supports bench/payoff/energy_conservation.py.

Check nornax's integrator examples/tests first; only implement here if
nornax has nothing suitable (see PROJECT_PLAN.md Phase 4).
"""

from __future__ import annotations

import jax.numpy as jnp


def integrate_and_track_conservation(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    n_steps: int,
    dt: float,
) -> dict[str, jnp.ndarray]:
    """Integrate an N-body system under jaccpot forces and track energy and
    angular momentum at each step.

    TODO: check nornax for an existing Hermite-integrator-based version of
    this before implementing a new integration loop here.
    """
    raise NotImplementedError("Check nornax repo for an existing implementation first.")
