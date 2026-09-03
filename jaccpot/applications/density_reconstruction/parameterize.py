"""Parameter pytrees and the mapping to source positions.

Hard decision 2: positions are the only free parameters, ``3N`` of them. The
``positions`` parameterisation is therefore the identity, and exists as a
module so the ``parametric`` alternative (5-12 scale/orientation parameters,
masses still fixed) can share the same ``to_positions`` seam when it lands.
Masses appear nowhere in any pytree here, and
:func:`~jaccpot.applications.density_reconstruction.forward.assert_masses_frozen_and_equal`
checks that.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

import jax.numpy as jnp
import numpy as np

__all__ = ["InitialGuessMode", "initial_positions", "positions_params", "to_positions"]

InitialGuessMode = Literal["perturbed_truth", "isotropized_truth", "uniform_sphere"]


def positions_params(source_positions: Any) -> Dict[str, jnp.ndarray]:
    """Wrap positions as the free-parameter pytree.

    Parameters
    ----------
    source_positions : Any
        ``(N, 3)`` positions.

    Returns
    -------
    Dict[str, jnp.ndarray]
        ``{"positions": (N, 3) float64}``. The single-key dict is the seam the
        parametric case will share; nothing else belongs in here.
    """
    return {"positions": jnp.asarray(source_positions, dtype=jnp.float64)}


def to_positions(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Map a parameter pytree to source positions.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Output of :func:`positions_params`.

    Returns
    -------
    jnp.ndarray
        ``(N, 3)`` positions. The identity for this parameterisation.
    """
    return params["positions"]


def initial_positions(
    truth_positions: np.ndarray,
    *,
    mode: InitialGuessMode,
    seed: int,
    perturbation: float = 0.1,
) -> np.ndarray:
    """Construct an initial guess (section 2: the starting point is a result).

    Parameters
    ----------
    truth_positions : np.ndarray
        ``(N, 3)`` true positions. Never returned unmodified.
    mode : InitialGuessMode
        ``"perturbed_truth"`` adds isotropic Gaussian displacement of scale
        ``perturbation``; ``"isotropized_truth"`` keeps each radius and
        redraws its direction; ``"uniform_sphere"`` ignores the truth beyond
        its outer radius.
    seed : int
        Seed for the draw.
    perturbation : float
        Displacement scale for ``"perturbed_truth"``.

    Returns
    -------
    np.ndarray
        ``(N, 3)`` float64 initial positions.

    Raises
    ------
    ValueError
        If ``mode`` is not recognised.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(truth_positions, dtype=np.float64)
    n = x.shape[0]
    if mode == "perturbed_truth":
        return x + perturbation * rng.standard_normal(x.shape)
    radii = np.linalg.norm(x, axis=1)
    cos_t = rng.uniform(-1.0, 1.0, size=n)
    sin_t = np.sqrt(np.maximum(1.0 - cos_t**2, 0.0))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    directions = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)
    if mode == "isotropized_truth":
        return radii[:, None] * directions
    if mode == "uniform_sphere":
        r_max = float(radii.max())
        r = r_max * np.cbrt(rng.uniform(0.0, 1.0, size=n))
        return r[:, None] * directions
    raise ValueError(f"unknown initial-guess mode {mode!r}")
