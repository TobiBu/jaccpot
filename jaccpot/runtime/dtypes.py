"""Centralized dtypes for integer indices.

Keep a single source of truth for index dtype so the codebase can be
switched between 32-bit and 64-bit indices easily.
"""

import jax.numpy as jnp
from jaxtyping import DTypeLike

# Use 64-bit indices by default to avoid overflow on large problems.
INDEX_DTYPE = jnp.int64


def as_index(x: object) -> jnp.ndarray:
    """Convert a Python or JAX scalar/array to INDEX_DTYPE.

    This helper ensures we consistently produce the configured integer
    dtype for indices and small scalar constants used as indices.
    """
    return jnp.asarray(x, dtype=INDEX_DTYPE)


def complex_dtype_for_real(real_dtype: DTypeLike) -> jnp.dtype:
    """Return complex dtype paired with a real floating dtype."""

    dtype = jnp.asarray(0, dtype=real_dtype).dtype
    if dtype == jnp.float64:
        return jnp.complex128
    return jnp.complex64


__all__ = ["INDEX_DTYPE", "as_index", "complex_dtype_for_real"]
