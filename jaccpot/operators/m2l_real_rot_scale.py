"""Pure-JAX real SH rotate+scale-to-z M2L kernels.

This module provides batched helpers for a solidFMM-inspired pipeline:
rotate multipoles into a z-aligned frame, apply z-axis M2L translation,
then rotate local coefficients back.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

from jaccpot.operators.real_harmonics import (
    real_rotation_from_z_axis_local,
    real_rotation_to_z_axis_multipole,
    sh_offset,
    sh_size,
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
        block = real_rotation_to_z_axis_multipole(x, y, z, ell, dtype=multipole.dtype)
        out = out.at[sl].set(block @ multipole[sl])
    return out


def _rotate_local_from_z_single(local_z: Array, delta: Array, *, order: int) -> Array:
    """Rotate one real local expansion from z-frame back to world frame."""
    x, y, z = delta[0], delta[1], delta[2]
    out = jnp.zeros_like(local_z)
    for ell in range(int(order) + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        block = real_rotation_from_z_axis_local(x, y, z, ell, dtype=local_z.dtype)
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


# ---------------------------------------------------------------------------
# Grouped (per-interaction-class) real M2L: precompute the real rotation blocks
# once per class and reuse them across all pairs that share the class geometry.
# This mirrors the complex cached-blocks path and removes the per-pair rotation
# construction cost, which dominates the rotate/scale M2L.
# ---------------------------------------------------------------------------


def _pack_by_ell(coeffs: Array, *, order: int) -> Array:
    """Pack (p+1)^2 coefficients into a padded (p+1, 2p+1) array."""
    p = int(order)
    max_m = 2 * p + 1
    out = jnp.zeros((p + 1, max_m), dtype=coeffs.dtype)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        out = out.at[ell, : 2 * ell + 1].set(coeffs[sl])
    return out


def _unpack_by_ell(packed: Array, *, order: int) -> Array:
    """Inverse of :func:`_pack_by_ell`."""
    p = int(order)
    out = jnp.zeros((sh_size(p),), dtype=packed.dtype)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        out = out.at[sl].set(packed[ell, : 2 * ell + 1])
    return out


@partial(jax.jit, static_argnames=("order",))
def _apply_real_rotation_blocks_padded_batch(
    coeffs: Array, blocks_array: Array, *, order: int
) -> Array:
    """Apply padded per-degree real rotation blocks to a batch of coefficients.

    ``blocks_array`` has shape ``(batch, p+1, 2p+1, 2p+1)`` (block-diagonal per
    degree, zero-padded); ``coeffs`` has shape ``(batch, (p+1)^2)``.
    """
    packed = jax.vmap(lambda c: _pack_by_ell(c, order=order))(coeffs)
    rotated = jnp.einsum("nbij,nbj->nbi", blocks_array, packed)
    return jax.vmap(lambda c: _unpack_by_ell(c, order=order))(rotated)


def _real_rotation_blocks_padded(
    deltas: Array, *, order: int, dtype: Any, which: str
) -> Array:
    """Padded real rotation blocks for a batch of displacement vectors.

    ``which='to_z_multipole'`` builds the multipole world->z blocks;
    ``which='from_z_local'`` builds the local z->world blocks.
    """
    p = int(order)
    max_m = 2 * p + 1
    if which == "to_z_multipole":
        rot_fn = real_rotation_to_z_axis_multipole
    elif which == "from_z_local":
        rot_fn = real_rotation_from_z_axis_local
    else:
        raise ValueError("which must be 'to_z_multipole' or 'from_z_local'")

    def one(delta: Array) -> Array:
        x, y, z = delta[0], delta[1], delta[2]
        out = jnp.zeros((p + 1, max_m, max_m), dtype=dtype)
        for ell in range(p + 1):
            size = 2 * ell + 1
            block = rot_fn(x, y, z, ell, dtype=dtype)
            out = out.at[ell, :size, :size].set(block)
        return out

    return jax.vmap(one)(jnp.asarray(deltas))


def real_rotation_blocks_to_z_multipole_batch(
    deltas: Array, *, order: int, dtype: Any
) -> Array:
    """Padded real multipole world->z rotation blocks, one set per delta."""
    return _real_rotation_blocks_padded(
        deltas, order=order, dtype=dtype, which="to_z_multipole"
    )


def real_rotation_blocks_from_z_local_batch(
    deltas: Array, *, order: int, dtype: Any
) -> Array:
    """Padded real local z->world rotation blocks, one set per delta."""
    return _real_rotation_blocks_padded(
        deltas, order=order, dtype=dtype, which="from_z_local"
    )


@partial(jax.jit, static_argnames=("order",))
def m2l_rot_scale_real_batch_cached_blocks(
    multipoles: Array,
    deltas: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    *,
    order: int,
) -> Array:
    """Batched real-basis M2L using precomputed per-pair rotation blocks.

    Equivalent to :func:`m2l_rot_scale_real_batch` but the (expensive) real
    rotation matrices are supplied precomputed (typically shared across an
    interaction class), so only the z-axis translation runs per pair.

    Parameters
    ----------
    multipoles:
        Source multipoles ``(batch, (order+1)^2)``.
    deltas:
        Source-to-target vectors ``(batch, 3)`` (used for the translation radius).
    blocks_to_z:
        Multipole world->z blocks ``(batch, order+1, 2*order+1, 2*order+1)``.
    blocks_from_z:
        Local z->world blocks with the same shape.
    order:
        Maximum SH order.
    """
    p = int(order)
    mult_rot = _apply_real_rotation_blocks_padded_batch(
        jnp.asarray(multipoles), jnp.asarray(blocks_to_z), order=p
    )
    radii = jnp.sqrt(
        jnp.maximum(jnp.sum(jnp.asarray(deltas) * jnp.asarray(deltas), axis=1), 1.0e-60)
    )
    locals_z = jax.vmap(lambda m, rr: translate_along_z_m2l_real(m, rr, order=p))(
        mult_rot, radii
    )
    return _apply_real_rotation_blocks_padded_batch(
        locals_z, jnp.asarray(blocks_from_z), order=p
    )


__all__ = [
    "m2l_core_z_real",
    "m2l_rot_scale_real_batch",
    "m2l_rot_scale_real_batch_cached_blocks",
    "real_rotation_blocks_to_z_multipole_batch",
    "real_rotation_blocks_from_z_local_batch",
]
