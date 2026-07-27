"""Operator dtype helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
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


def squared_radius_floor(dtype) -> float:
    """Smallest squared radius representable enough to keep ``sqrt`` differentiable.

    Returns the historical ``1e-60`` wherever it is a normal number of ``dtype``
    (so float64 behaviour is preserved bit-for-bit) and the dtype's smallest
    normal otherwise (float32/float16, where ``1e-60`` would flush to zero and
    disable the guard entirely).
    """
    tiny = float(np.finfo(jnp.dtype(dtype)).tiny)
    return max(_LEGACY_SQUARED_RADIUS_FLOOR, tiny)


def floor_squared_radius(r2: Array) -> Array:
    """Clamp a squared radius to :func:`squared_radius_floor` for ``r2``'s dtype.

    Replaces bare ``jnp.maximum(r2, 1e-60)`` so the clamp survives float32.
    """
    return jnp.maximum(r2, jnp.asarray(squared_radius_floor(r2.dtype), r2.dtype))


__all__ = [
    "INDEX_DTYPE",
    "as_index",
    "complex_dtype_for_real",
    "floor_squared_radius",
    "squared_radius_floor",
]
