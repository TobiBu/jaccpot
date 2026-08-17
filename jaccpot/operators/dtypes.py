"""Operator dtype helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, DTypeLike
from yggdrax.dtypes import INDEX_DTYPE, as_index, complex_dtype_for_real

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
_LEGACY_SQUARED_RADIUS_FLOOR = 1e-60


def squared_radius_floor(dtype: DTypeLike) -> float:
    """Smallest squared radius representable enough to keep ``sqrt`` differentiable.

    Returns the historical ``1e-60`` wherever it is a normal number of ``dtype``
    (so float64 behaviour is preserved bit-for-bit) and the dtype's smallest
    normal otherwise (float32/float16, where ``1e-60`` would flush to zero and
    disable the guard entirely).

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
    tiny = float(np.finfo(jnp.dtype(dtype)).tiny)
    return max(_LEGACY_SQUARED_RADIUS_FLOOR, tiny)


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
    "complex_dtype_for_real",
    "floor_squared_radius",
    "squared_radius_floor",
]
