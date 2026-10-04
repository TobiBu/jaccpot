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
    - ``JACCPOT_INDEX_PRECISION=int32`` (default since 2026-10: half the bytes of
      every index array, list and sort key; ample for one GPU)
    - ``JACCPOT_INDEX_PRECISION=int64`` (opt-in, for index spaces past 2^31)

    Called exactly once, at import, to initialise ``INDEX_DTYPE``. Setting the
    variable after :mod:`jaccpot` is imported therefore does nothing. With int32,
    :func:`require_index_capacity` refuses a problem whose particle count, node x
    coefficient count or list capacities reach 2^31, naming the switch.

    An unrecognised value falls back to the default **silently** rather than
    raising -- a deliberate exception to the fail-loudly policy for a
    diagnostics-adjacent knob.

    Returns
    -------
    DTypeLike
        ``jnp.int32`` or ``jnp.int64``. The ``int64`` request is honoured in
        practice because importing yggdrax sets ``jax_enable_x64``; without that
        JAX would quietly demote it to int32.
    """
    raw = str(os.environ.get("JACCPOT_INDEX_PRECISION", "int32")).strip().lower()
    if raw in ("int64", "i64", "64"):
        return jnp.int64
    return jnp.int32


INDEX_DTYPE = _resolve_index_dtype()

#: Sizes at or past this need int64 indices.
_INT32_INDEX_LIMIT = 2**31 - 1


def require_index_capacity(**sizes: int) -> None:
    """Refuse sizes the index dtype cannot address (int32: ``2^31 - 1``).

    Parameters
    ----------
    **sizes : int
        Named element counts that index arithmetic reaches (particle count,
        nodes x coefficients, list and queue capacities). Static.

    Raises
    ------
    ValueError
        With int32 indices, if any size reaches ``2^31 - 1``; the message names
        the sizes and the switch to int64.
    """
    if jnp.dtype(INDEX_DTYPE).itemsize >= 8:
        return
    over = {k: int(v) for k, v in sizes.items() if int(v) >= _INT32_INDEX_LIMIT}
    if over:
        raise ValueError(
            f"{over} reach the int32 index range (2^31 - 1); set "
            "JACCPOT_INDEX_PRECISION=int64 (and YGGDRAX_INDEX_PRECISION=int64) "
            "before importing jaccpot"
        )


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


__all__ = [
    "INDEX_DTYPE",
    "as_index",
    "complex_dtype_for_real",
    "require_index_capacity",
]
