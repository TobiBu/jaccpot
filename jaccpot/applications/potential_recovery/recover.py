"""Gradient-based (and optionally HMC/VI) recovery of parametric potential
parameters from synthetic kinematics, using jaccpot's FMM forward pass +
autodiff end-to-end.

This is the code bench/payoff/parameter_recovery_demo.py calls into.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from .model import ParametricPotential

__all__ = ["RecoveryResult", "recover_grad_descent", "recover_hmc"]


@dataclasses.dataclass
class RecoveryResult:
    params_history: jnp.ndarray  # (n_iterations, n_params)
    loss_history: jnp.ndarray  # (n_iterations,)


def recover_grad_descent(
    observed_kinematics: dict[str, jnp.ndarray],
    initial_params: jnp.ndarray,
    n_iterations: int = 500,
    learning_rate: float = 1e-2,
) -> RecoveryResult:
    """Recover potential parameters by gradient descent through jaccpot's FMM
    forward pass.

    TODO: define the forward model (predict kinematics from params via
    jaccpot's FMM), the loss (e.g. chi^2 vs. observed_kinematics), and the
    optimization loop (plain SGD/Adam via optax, or hand-rolled -- match
    whatever the rest of jaccpot's examples use).

    Parameters
    ----------
    observed_kinematics : dict[str, jnp.ndarray]
        The mock dataset to fit, as produced by
        :func:`~jaccpot.applications.potential_recovery.model.generate_synthetic_ifu_kinematics`.
    initial_params : jnp.ndarray
        Starting point for the optimisation, ``[n_params]``.
    n_iterations : int
        Number of descent steps.
    learning_rate : float
        Step size.

    Returns
    -------
    RecoveryResult
        The parameter and loss traces over the optimisation.

    Raises
    ------
    NotImplementedError
        Always: this is scaffolding for PROJECT_PLAN.md Phase 4.
    """
    raise NotImplementedError(
        "Implement the forward model, loss, and optimization loop."
    )


def recover_hmc(
    observed_kinematics: dict[str, jnp.ndarray],
    initial_params: jnp.ndarray,
    n_samples: int = 1000,
) -> RecoveryResult:
    """Optional: HMC/NUTS-based posterior recovery instead of point-estimate
    gradient descent, using the same differentiable forward model.

    TODO: only implement this if the paper wants the full posterior, not
    just a point estimate -- see PROJECT_PLAN.md Phase 4 for sequencing.

    Parameters
    ----------
    observed_kinematics : dict[str, jnp.ndarray]
        The mock dataset to fit, as for :func:`recover_grad_descent`.
    initial_params : jnp.ndarray
        Chain starting point, ``[n_params]``.
    n_samples : int
        Number of posterior samples to draw.

    Returns
    -------
    RecoveryResult
        The sampled chain, in the same container the point-estimate path uses.

    Raises
    ------
    NotImplementedError
        Always: this is scaffolding for PROJECT_PLAN.md Phase 4.
    """
    raise NotImplementedError(
        "Optional HMC path -- implement only if needed for the paper."
    )
