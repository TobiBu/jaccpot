"""Autodiff helpers for Jaccpot."""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import lax
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
    """Differentiable gravitational accelerations via direct O(N^2) summation.

    A plain, fully-differentiable direct sum. The FMM force *itself* is now
    end-to-end differentiable at fixed topology via
    :meth:`jaccpot.FastMultipoleMethod.differentiable_accelerations` (exact
    gradients w.r.t. positions and masses; see ``docs/differentiable_fmm_design.md``).
    This direct sum is retained as the simple exact-gradient reference and as the
    **gradient oracle** for tests: ``jax.grad`` of the FMM must match ``jax.grad``
    of this sum to the FMM's own force accuracy.

    It deliberately accepts the same FMM-shaped keyword arguments (``theta``,
    ``bounds``, ``leaf_size``, ``max_order``) purely for drop-in signature
    compatibility with the FMM call -- they do not apply to direct summation and
    are ignored. Only ``G`` and ``softening`` affect the result.

    Parameters
    ----------
    positions : Array
        Particle positions ``[N, 3]``. Any order; unlike the FMM there is no Morton
        sort, so the output is aligned with the input.
    masses : Array
        Particle masses ``[N]``.
    theta : float
        **Ignored.** Accepted for signature compatibility with the FMM.
    G : float
        Gravitational constant, applied as a plain multiplier.
    softening : float
        Plummer softening length. Added in quadrature, so a coincident pair is
        finite for any nonzero value; at ``softening == 0`` a coincident pair is a
        genuine singularity that this function does not guard.
    bounds : Optional[Tuple[Array, Array]]
        **Ignored.** Accepted for signature compatibility with the FMM.
    leaf_size : int
        **Ignored.** Accepted for signature compatibility with the FMM.
    max_order : int
        **Ignored.** There is no expansion here, so there is no truncation error --
        which is precisely what makes this a usable oracle.

    Returns
    -------
    Array
        Accelerations ``[N, 3]``, in the input particle order.

    Notes
    -----
    Differentiable in ``positions``, ``masses`` and ``G``; not in ``softening``
    (it is squared into a constant) and not in the ignored arguments.

    ``O(N^2)`` in time and memory -- the ``[N, N, 3]`` pairwise difference tensor is
    materialised, so this is for gradient work at modest ``N``, not evaluation at
    scale. Self-interaction is removed by a diagonal mask rather than by skipping,
    which keeps the computation dense and jittable.

    fp32 matmul precision is pinned to ``lax.Precision.HIGHEST`` in the final
    contraction, for the same reason it is pinned across the operators: XLA
    otherwise lowers fp32 matmuls to TF32 on Ampere+ and an oracle with a ~6e-04
    accuracy floor cannot validate anything below that floor.
    """

    # FMM-shaped args accepted for API parity only; direct sum ignores them.
    del theta, bounds, leaf_size, max_order

    diffs = positions[:, None, :] - positions[None, :, :]
    dist2 = jnp.sum(diffs * diffs, axis=-1) + softening**2
    inv_dist = jnp.where(dist2 > 0, jnp.power(dist2, -0.5), 0.0)
    inv_dist3 = inv_dist**3
    weights = masses[None, :] * inv_dist3
    weights = weights * (1.0 - jnp.eye(positions.shape[0], dtype=positions.dtype))
    return -G * jnp.einsum(
        "ij,ijk->ik", weights, diffs, precision=lax.Precision.HIGHEST
    )


__all__ = ["differentiable_gravitational_acceleration"]
