"""Operator dtype helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, DTypeLike
from yggdrax.dtypes import INDEX_DTYPE, as_index

# Historical squared-radius floor. Load-bearing for REVERSE-mode safety, not just
# for the forward: the translation operators take ``r = sqrt(max(r2, FLOOR))`` and
# then divide by ``r`` (or test ``rho > 1e-30``), so a floor that keeps ``r``
# bounded away from zero is what makes ``d sqrt / dr2 = 1/(2r)`` finite at the
# genuinely-degenerate displacements the fixed-topology FMM hits (a single-child
# internal node shares its child's centre of mass => an exact-zero L2L
# translation).
#
# 1e-60 is fine in float64 but **underflows to exactly 0.0 in float32** (smallest
# subnormal ~1.4e-45), which silently disabled the guard: the float32 reverse pass
# then evaluated ``sqrt(0)`` / ``x / 0`` and produced all-NaN gradients as soon as
# the far field was active, while float64 stayed clean. See
# :func:`squared_radius_floor`.
#
# A normal floor is not enough either: the L2P's reverse is SECOND order in the
# offset (its custom VJP is a Hessian-vector product, and plain autodiff is
# reverse-over-reverse), and ``d^2 sqrt / dr2^2 = -r2^(-3/2) / 4`` at the floor.
# float32's smallest normal (1.2e-38) gives 7.8e56 there, past float32's max, so
# a particle exactly at its leaf's expansion centre or on the centre's z axis
# (one-particle leaves; common at any N) turned ``inf * 0`` into NaN in the
# position gradient. The floor has to keep ``floor^(-3/2)`` finite as well.
_LEGACY_SQUARED_RADIUS_FLOOR = 1e-60


def squared_radius_floor(dtype: DTypeLike) -> float:
    """Squared-radius floor that keeps ``sqrt`` twice differentiable in ``dtype``.

    Returns the historical ``1e-60`` wherever it and ``1e-60 ** -1.5`` are both
    finite normal numbers of ``dtype`` (float64, whose behaviour is preserved
    bit-for-bit), and ``sqrt(tiny)`` of the dtype otherwise -- 1.1e-19 for
    float32, so ``r >= 3.3e-10`` and ``floor ** -1.5 = 2.8e28``, ten orders of
    magnitude inside float32's range. ``1e-60`` flushes to zero in float32, and
    the dtype's smallest normal still overflows the second derivative the L2P
    reverse takes (see the comment above ``_LEGACY_SQUARED_RADIUS_FLOOR``). The
    floor binds only at displacements below ``sqrt(floor)``, which for float32
    coordinates of order one are exact zeros.

    Parameters
    ----------
    dtype : DTypeLike
        Real floating dtype the floor will be compared against. Passing an
        integer dtype raises from :func:`numpy.finfo`, which is the intended
        failure — there is no meaningful squared-radius floor for one.

    Returns
    -------
    float
        A Python float, not an array: callers embed it as a constant so it stays
        out of the traced graph. Cast it to the working dtype at the use site,
        as :func:`floor_squared_radius` does.
    """
    info = np.finfo(jnp.dtype(dtype))
    tiny, huge = float(info.tiny), float(info.max)
    legacy = _LEGACY_SQUARED_RADIUS_FLOOR
    if legacy >= tiny and legacy**-1.5 < huge:
        return legacy
    return float(np.sqrt(tiny))


def floor_squared_radius(r2: Array) -> Array:
    """Clamp a squared radius to :func:`squared_radius_floor` for ``r2``'s dtype.

    Replaces bare ``jnp.maximum(r2, 1e-60)`` so the clamp survives float32.

    Parameters
    ----------
    r2 : Array
        Squared radii, any shape. The floor is chosen from ``r2.dtype``, so a
        float32 array clamped here and a float64 array clamped there are floored
        at different values on purpose — passing a weakly-typed Python scalar
        instead of an array therefore picks up whatever dtype JAX promotes it to.

    Returns
    -------
    Array
        ``r2`` with every entry raised to at least the dtype's floor, same shape
        and dtype. Elementwise and differentiable; the gradient is zero wherever
        the floor bound, which is what keeps ``d sqrt/d r2`` finite downstream.
    """
    return jnp.maximum(r2, jnp.asarray(squared_radius_floor(r2.dtype), r2.dtype))


__all__ = [
    "INDEX_DTYPE",
    "as_index",
    "floor_squared_radius",
    "squared_radius_floor",
]
