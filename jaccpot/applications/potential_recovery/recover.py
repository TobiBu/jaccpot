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
    """
    raise NotImplementedError(
        "Optional HMC path -- implement only if needed for the paper."
    )
