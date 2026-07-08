"""Pure-JAX real SH rotate+scale-to-z M2L kernels.

This module provides batched helpers for a solidFMM-inspired pipeline:
rotate multipoles into a z-aligned frame, apply z-axis M2L translation,
then rotate local coefficients back.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from jaccpot.operators.real_harmonics import (
    real_rotation_from_z_axis_multipole,
    real_rotation_to_z_axis_local,
    sh_offset,
    translate_along_z_m2l_real,
)

# NOTE: ``jaccpot.pallas.m2l_core_z_real`` is imported lazily inside
# ``m2l_core_z_real`` below. A top-level import creates a circular import
# (``jaccpot.pallas.__init__`` -> ``m2l_core_z_real`` -> ``jaccpot.operators``
# -> this module -> ``jaccpot.pallas.m2l_core_z_real``), which breaks
# ``from jaccpot.pallas import ...`` when it is the first jaccpot import.


def _rotate_multipole_to_z_single(
    multipole: Array, delta: Array, *, order: int
) -> Array:
    """Rotate one real multipole expansion into the z-aligned frame."""
    x, y, z = delta[0], delta[1], delta[2]
    out = jnp.zeros_like(multipole)
    for ell in range(int(order) + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        block = real_rotation_from_z_axis_multipole(x, y, z, ell, dtype=multipole.dtype)
        out = out.at[sl].set(block @ multipole[sl])
    return out


def _rotate_local_from_z_single(local_z: Array, delta: Array, *, order: int) -> Array:
    """Rotate one real local expansion from z-frame back to world frame."""
    x, y, z = delta[0], delta[1], delta[2]
    out = jnp.zeros_like(local_z)
    for ell in range(int(order) + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        block = real_rotation_to_z_axis_local(x, y, z, ell, dtype=local_z.dtype)
        out = out.at[sl].set(block @ local_z[sl])
    return out


def m2l_core_z_real(
    multipole_rot: Array,
    radii: Array,
    *,
    order: int,
    use_pallas: bool = False,
) -> Array:
    """Apply z-axis real M2L translation to a batch of rotated multipoles.

    When ``use_pallas=True``, the function dispatches to the optional Pallas
    kernel on supported accelerator backends and otherwise falls back to the
    pure-JAX recurrence.
    """
    from jaccpot.pallas.m2l_core_z_real import (
        m2l_core_z_real_pallas,
        pallas_m2l_real_supported,
    )

    r = jnp.maximum(jnp.asarray(radii), jnp.asarray(1.0e-30, dtype=radii.dtype))
    if bool(use_pallas) and pallas_m2l_real_supported():
        return m2l_core_z_real_pallas(multipole_rot, r, order=int(order))
    return jax.vmap(lambda m, rr: translate_along_z_m2l_real(m, rr, order=int(order)))(
        multipole_rot,
        r,
    )


def m2l_rot_scale_real_batch(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    use_pallas: bool = False,
) -> Array:
    """Batched rotate+scale real-basis M2L translation.

    Parameters
    ----------
    multipoles:
        Source multipoles with shape ``(batch, (order+1)^2)``.
    deltas:
        Source-to-target vectors with shape ``(batch, 3)``.
    order:
        Maximum SH order.
    use_pallas:
        Enable the optional Pallas z-translation kernel when supported.
    """
    mult = jnp.asarray(multipoles)
    delta = jnp.asarray(deltas)
    if mult.ndim != 2:
        raise ValueError("multipoles must have shape (batch, coeffs)")
    if delta.ndim != 2 or int(delta.shape[1]) != 3:
        raise ValueError("deltas must have shape (batch, 3)")

    radii = jnp.linalg.norm(delta, axis=1)
    mult_rot = jax.vmap(
        lambda m, d: _rotate_multipole_to_z_single(m, d, order=int(order))
    )(
        mult,
        delta,
    )
    locals_z = m2l_core_z_real(
        mult_rot,
        radii,
        order=int(order),
        use_pallas=bool(use_pallas),
    )
    return jax.vmap(lambda l, d: _rotate_local_from_z_single(l, d, order=int(order)))(
        locals_z,
        delta,
    )


__all__ = ["m2l_core_z_real", "m2l_rot_scale_real_batch"]
