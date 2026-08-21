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

    Called exactly once, at import, to initialise ``INDEX_DTYPE``. Setting the
    variable after :mod:`jaccpot` is imported therefore does nothing.

    An unrecognised value falls back to ``int64`` **silently** rather than
    raising -- a deliberate exception to the fail-loudly policy for a
    diagnostics-adjacent knob, but it does mean a typo such as
    ``JACCPOT_INDEX_PRECISION=in32`` gives 64-bit indices with no warning.

    Returns
    -------
    DTypeLike
        ``jnp.int32`` or ``jnp.int64``. The ``int64`` request is honoured in
        practice because importing yggdrax sets ``jax_enable_x64``; without that
        JAX would quietly demote it to int32.
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

    Parameters
    ----------
    x : object
        Anything ``jnp.asarray`` accepts: a Python int, a NumPy or JAX array, or
        a tracer. Deliberately widest-possible, because this is called on both
        host constants and traced values. Floating input is **truncated**, not
        rejected -- ``as_index(2.7)`` is 2.

    Returns
    -------
    jnp.ndarray
        ``x`` as ``INDEX_DTYPE``. Safe under ``jit``: a traced argument stays
        traced, so this is a cast, not a host sync.
    """
    return jnp.asarray(x, dtype=INDEX_DTYPE)


def complex_dtype_for_real(real_dtype: DTypeLike) -> DTypeLike:
    """Return complex dtype paired with a real floating dtype.

    Parameters
    ----------
    real_dtype : DTypeLike
        A real dtype. Resolved through ``jnp.asarray(0, dtype=...)``, so it picks
        up JAX's own canonicalisation -- with x64 disabled a ``float64`` request
        resolves to float32 here and pairs with complex64, matching what the
        arrays will actually be.

    Returns
    -------
    DTypeLike
        ``jnp.complex128`` for float64, ``jnp.complex64`` for **everything
        else**. That is a floor as well as a mapping: float16 and bfloat16 both
        widen to complex64 rather than to a half-precision complex, which JAX has
        no type for.
    """

    dtype = jnp.asarray(0, dtype=real_dtype).dtype
    if dtype == jnp.float64:
        return jnp.complex128
    return jnp.complex64


__all__ = ["INDEX_DTYPE", "as_index", "complex_dtype_for_real"]
