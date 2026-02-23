"""Autodiff helpers for Jaccpot."""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

def differentiable_gravitational_acceleration(
    positions: Array,
    masses: Array,
    *,
    theta: float = 0.6,
    G: float = 1.0,
    softening: float = 1e-3,
    bounds: Optional[Tuple[Array, Array]] = None,
    leaf_size: int = 16,
    max_order: int = 4,
) -> Array:
    """Differentiable gravitational accelerations via direct summation."""

    del theta, bounds, leaf_size, max_order

    diffs = positions[:, None, :] - positions[None, :, :]
    dist2 = jnp.sum(diffs * diffs, axis=-1) + softening**2
    inv_dist = jnp.where(dist2 > 0, jnp.power(dist2, -0.5), 0.0)
    inv_dist3 = inv_dist**3
    weights = masses[None, :] * inv_dist3
    weights = weights * (1.0 - jnp.eye(positions.shape[0], dtype=positions.dtype))
    return -G * jnp.einsum("ij,ijk->ik", weights, diffs)


__all__ = ["differentiable_gravitational_acceleration"]
