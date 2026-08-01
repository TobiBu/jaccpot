"""Centralized dtypes for integer indices.

Keep a single source of truth for index dtype so the codebase can be
switched between 32-bit and 64-bit indices easily.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
from jaxtyping import DTypeLike


def _resolve_index_dtype() -> DTypeLike:
    """Resolve index dtype from environment.

    Supported values:
    - ``JACCPOT_INDEX_PRECISION=int32`` (lower memory, faster on many GPUs)
    - ``JACCPOT_INDEX_PRECISION=int64`` (default; safest for very large indices)
    """
    raw = str(os.environ.get("JACCPOT_INDEX_PRECISION", "int64")).strip().lower()
    if raw in ("int32", "i32", "32"):
        return jnp.int32
    if raw in ("int64", "i64", "64"):
        return jnp.int64
    # Defensive fallback for unknown user input.
    return jnp.int64


INDEX_DTYPE = _resolve_index_dtype()


def as_index(x: object) -> jnp.ndarray:
    """Convert a Python or JAX scalar/array to INDEX_DTYPE.

    This helper ensures we consistently produce the configured integer
    dtype for indices and small scalar constants used as indices.
    """
    return jnp.asarray(x, dtype=INDEX_DTYPE)


def complex_dtype_for_real(real_dtype: DTypeLike) -> DTypeLike:
    """Return complex dtype paired with a real floating dtype."""

    dtype = jnp.asarray(0, dtype=real_dtype).dtype
    if dtype == jnp.float64:
        return jnp.complex128
    return jnp.complex64


__all__ = ["INDEX_DTYPE", "as_index", "complex_dtype_for_real"]
