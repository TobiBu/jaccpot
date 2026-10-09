"""Process-level runtime caches + byte accounting for the FMM runtime.

Leaf module extracted from ``_fmm_impl.py`` (Phase 2 of the runtime refactor).
It owns the helpers that size runtime payloads and clear the process-level
caches. (The grouped operator and segment caches went with the grouped far
field in the 2026-10 cleanup, X3; the M2L chunk autotune cache went with the
autotune, X4.) Depends only on stdlib + jax + numpy, so both the orchestrator
and the kernel library import it without cycles.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from jaxtyping import Array

from jaccpot._jax_compat import Tracer

__all__: list[str] = []


def _array_nbytes(arr: Array) -> int:
    """Return approximate storage size in bytes for one array leaf.

    Approximate because it is ``prod(shape) * itemsize`` -- the logical size, not
    whatever the device actually allocated after padding or layout choices. Good
    enough to drive the cache budgets, which are advisory ceilings.

    Parameters
    ----------
    arr : Array
        Array to size. A missing ``shape`` is treated as scalar.

    Returns
    -------
    int
        Logical byte count.
    """
    shape = tuple(int(dim) for dim in getattr(arr, "shape", ()))
    if len(shape) == 0:
        return int(np.dtype(arr.dtype).itemsize)
    return int(np.prod(np.asarray(shape, dtype=np.int64))) * int(
        np.dtype(arr.dtype).itemsize
    )


def _format_nbytes(count: int) -> str:
    """Render a byte count in binary units, for diagnostics.

    Parameters
    ----------
    count : int
        Byte count; negatives are clamped to zero.

    Returns
    -------
    str
        Two-decimal value with a ``B``/``KiB``/``MiB``/``GiB``/``TiB`` suffix,
        saturating at ``TiB``.
    """
    value = float(max(int(count), 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TiB"


def _estimate_payload_nbytes(value: Any) -> int:
    """Best-effort recursive byte estimate for array-centric payloads.

    Walks arrays, dicts, tuples/lists, ``NamedTuple``s and plain objects with a
    ``__dict__``. Anything else contributes **zero** rather than raising, so this
    under-reports rather than failing -- appropriate for a diagnostic, and the
    reason it must not be used as a hard memory bound.

    Parameters
    ----------
    value : Any
        Payload to size. ``None`` is zero.

    Returns
    -------
    int
        Estimated bytes.
    """
    if value is None:
        return 0
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _array_nbytes(value)
    if isinstance(value, dict):
        return int(sum(_estimate_payload_nbytes(v) for v in value.values()))
    if isinstance(value, (tuple, list)):
        return int(sum(_estimate_payload_nbytes(v) for v in value))
    if hasattr(value, "_asdict"):
        return _estimate_payload_nbytes(value._asdict())
    if hasattr(value, "__dict__"):
        return _estimate_payload_nbytes(vars(value))
    return 0


def _clear_global_runtime_caches(*, clear_jax_compilation: bool = False) -> None:
    """Drop process-level runtime caches that can retain large array payloads.

    Since the grouped operator and segment caches went with the grouped far field
    (2026-10 cleanup, X3) no process-level cache here holds arrays, so only the
    optional JAX compilation clear is left.

    Parameters
    ----------
    clear_jax_compilation : bool
        Also call ``jax.clear_caches()``. Off by default because that cache is
        process-global and shared with everything else in the process.

    Returns
    -------
    None
        Mutates the process-level caches in place.
    """
    if clear_jax_compilation:
        jax.clear_caches()


def _contains_tracer(value: Any) -> bool:
    """Return ``True`` when a pytree contains JAX tracer values.

    The gate on host-side caching: a tracer has no concrete bytes to digest, so a
    traced payload must not be keyed.

    Parameters
    ----------
    value : Any
        Pytree to inspect.

    Returns
    -------
    bool
        ``True`` if any leaf is a tracer.
    """
    return any(isinstance(leaf, Tracer) for leaf in jax.tree_util.tree_leaves(value))
