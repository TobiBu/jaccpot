"""Parametric gravitational potential model + synthetic IFU-like kinematic
data generator, for the gradient-based potential-recovery payoff case study.

Lives under jaccpot/applications/ (not the core runtime/ public API) since
it's paper-specific rather than maintained-library surface -- see
PROJECT_PLAN.md's repo conventions.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

__all__ = ["ParametricPotential", "generate_synthetic_ifu_kinematics"]


@dataclasses.dataclass
class ParametricPotential:
    """A small number of free parameters (e.g. scale radius, mass, flattening)
    defining a parametric potential to be recovered from synthetic kinematics.

    TODO: choose the concrete parametric family (e.g. a simple flattened
    NFW-like or Miyamoto-Nagai-like potential) and implement its evaluation
    here, wired so jaccpot's FMM forward pass can be combined with (or
    validated against) this analytic potential's known gradient.

    Attributes
    ----------
    params : jnp.ndarray
        The free parameters, ``[n_params]``. Their meaning is fixed by whichever
        parametric family the TODO above settles on, so it is deliberately not
        described here yet.
    """

    params: jnp.ndarray

    def potential(self: "ParametricPotential", positions: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(
            "Choose and implement the parametric potential family."
        )

    def acceleration(
        self: "ParametricPotential", positions: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError("Analytic or autodiff gradient of potential().")


def generate_synthetic_ifu_kinematics(
    potential: ParametricPotential,
    n_particles: int,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Generate a synthetic IFU-like kinematic dataset (positions + line-of-
    sight velocities on a projected grid) from a known ground-truth potential,
    to be recovered by recover.py.

    TODO: sample a tracer population in the given potential (e.g. via
    distribution-function sampling or a short N-body realization using
    jaccpot's own FMM), project onto a mock IFU field of view, add
    realistic-ish observational noise.

    Parameters
    ----------
    potential : ParametricPotential
        Ground-truth potential the tracers are sampled in, and the thing
        ``recover.py`` is then asked to reconstruct.
    n_particles : int
        Number of tracer particles to sample.
    seed : int
        RNG seed for the sampling and the observational noise.

    Returns
    -------
    dict[str, jnp.ndarray]
        The mock dataset. Its keys are fixed by the field-of-view projection the
        TODO above settles on, so they are not enumerated yet.

    Raises
    ------
    NotImplementedError
        Always: this is scaffolding for PROJECT_PLAN.md Phase 4.
    """
    raise NotImplementedError("Implement synthetic IFU-like kinematics generation.")
