"""Centralized dtypes for integer indices, and the ``jax_enable_x64`` requirement.

Keep a single source of truth for index dtype so the codebase can be
switched between 32-bit and 64-bit indices easily.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jaxtyping import DTypeLike


def _resolve_index_dtype() -> DTypeLike:
    """Resolve index dtype: ``JACCPOT_INDEX_PRECISION``, else yggdrax's.

    Supported values:
    - ``JACCPOT_INDEX_PRECISION=int32``: half the bytes of every index array,
      list and sort key; ample for one GPU.
    - ``JACCPOT_INDEX_PRECISION=int64``: for index spaces past 2^31.
    - unset: **yggdrax's** ``INDEX_DTYPE`` (``YGGDRAX_INDEX_PRECISION``, else this
      variable, else yggdrax's default). jaccpot hands its index arrays to
      yggdrax and some of its modules use yggdrax's dtype directly, so the two
      must agree; a jaccpot default of its own could disagree with an older or
      newer yggdrax.

    Called exactly once, at import, to initialise ``INDEX_DTYPE``. Setting the
    variable after :mod:`jaccpot` is imported therefore does nothing. With int32,
    :func:`require_index_capacity` refuses a problem whose particle count, node x
    coefficient count or list capacities reach 2^31, naming the switch.

    An unrecognised value falls back to yggdrax's dtype **silently** rather than
    raising -- a deliberate exception to the fail-loudly policy for a
    diagnostics-adjacent knob.

    Returns
    -------
    DTypeLike
        ``jnp.int32`` or ``jnp.int64``. The ``int64`` request is honoured because
        jaccpot requires ``jax_enable_x64`` (:func:`require_x64`); without it JAX
        would quietly demote it to int32.
    """
    raw = str(os.environ.get("JACCPOT_INDEX_PRECISION", "")).strip().lower()
    if raw in ("int64", "i64", "64"):
        return jnp.int64
    if raw in ("int32", "i32", "32"):
        return jnp.int32
    from yggdrax.dtypes import INDEX_DTYPE as yggdrax_index_dtype

    return jnp.int64 if jnp.dtype(yggdrax_index_dtype).itemsize >= 8 else jnp.int32


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


def require_x64(what: str) -> None:
    """Raise unless ``jax_enable_x64`` is on; jaccpot needs JAX's 64-bit types.

    The trees are built on 63-bit ``uint64`` Morton codes, and the large-N centre of
    mass prefix sums, the ``accum="wide"`` near-field accumulator, the precomputed
    rotation constants, the multi-GPU global ids and ``precision="fp64"`` are 64-bit.
    With x64 off, JAX narrows those requests to 32 bits with at most a warning: the
    trees come out wrong and the forces with them (rel-L2 0.19 on the float32 lanes
    when this was tried), with no error. yggdrax switched x64 on at import until
    2026-10; jaccpot now asks for it instead of relying on that.

    x64 does not make jaccpot compute in float64: the working precision follows the
    input dtype (``working_dtype`` / ``precision``), so float32 inputs stay float32.

    Parameters
    ----------
    what : str
        The class or function that needs it, named in the message.

    Raises
    ------
    RuntimeError
        If ``jax_enable_x64`` is off.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            f"{what} needs jax_enable_x64=True. Enable it at the top of your program, "
            'before creating arrays: jax.config.update("jax_enable_x64", True), or '
            "set JAX_ENABLE_X64=1. It does not switch jaccpot to float64: the input "
            "dtype (or working_dtype / precision) still sets the working precision."
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


__all__ = [
    "INDEX_DTYPE",
    "as_index",
    "require_index_capacity",
    "require_x64",
]
