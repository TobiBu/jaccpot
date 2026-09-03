"""Data misfit for the reconstruction. Regularisers are added, never hidden.

Only the data term lives here for now; the regularisers hard decision 9
requires (nearest-neighbour / smoothness, and a softening-floor penalty) land
with ``fit.py``. Every term is individually toggleable and weighted, and the
weights are recorded in the results JSON -- so the loss returns its components
as well as the total.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = ["data_misfit", "total_loss"]


def data_misfit(
    predicted: ArrayLike,
    observed: ArrayLike,
    *,
    scale: float = 1.0,
) -> jnp.ndarray:
    """Mean squared acceleration residual over tracers.

    Parameters
    ----------
    predicted : ArrayLike
        ``(M, 3)`` model accelerations at the tracers. NumPy or JAX.
    observed : ArrayLike
        ``(M, 3)`` observed accelerations. NumPy or JAX.
    scale : float
        Divides the residual before squaring, so the loss is O(1) whatever the
        field magnitude. Use a characteristic acceleration; ``1.0`` leaves the
        raw units.

    Returns
    -------
    jnp.ndarray
        Scalar. This is the **field-space residual**, the primary metric of
        the section.
    """
    residual = (jnp.asarray(predicted) - jnp.asarray(observed)) / scale
    return jnp.mean(jnp.sum(residual**2, axis=-1))


def total_loss(
    predicted: ArrayLike,
    observed: ArrayLike,
    *,
    weights: Dict[str, float],
    scale: float = 1.0,
    extra_terms: Dict[str, Any] | None = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Weighted sum of the data misfit and any supplied regulariser values.

    Parameters
    ----------
    predicted : ArrayLike
        ``(M, 3)`` model accelerations. NumPy or JAX.
    observed : ArrayLike
        ``(M, 3)`` observed accelerations. NumPy or JAX.
    weights : Dict[str, float]
        Term name to weight. ``"data"`` weights the misfit; other keys weight
        matching entries of ``extra_terms``. A weight of ``0.0`` disables a
        term but still reports it.
    scale : float
        Passed to :func:`data_misfit`.
    extra_terms : Dict[str, Any] | None
        Already-evaluated regulariser scalars by name.

    Returns
    -------
    Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]
        The total, and every component (unweighted) so the results JSON can
        report them alongside the weights.
    """
    components: Dict[str, jnp.ndarray] = {
        "data": data_misfit(predicted, observed, scale=scale)
    }
    for name, value in (extra_terms or {}).items():
        components[name] = jnp.asarray(value)
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for name, value in components.items():
        total = total + float(weights.get(name, 0.0)) * value
    return total, components
