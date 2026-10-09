"""Process-level runtime caches + byte accounting for the FMM runtime.

Leaf module extracted from ``_fmm_impl.py`` (Phase 2 of the runtime refactor).
It owns the mutable M2L autotune cache and every helper that keys, sizes,
evicts, serializes, or clears runtime caches, so the single source of truth for
this shared state lives in one place. (The grouped operator and segment caches
went with the grouped far field in the 2026-10 cleanup, X3.) Depends only on
stdlib + jax + numpy, so both the orchestrator and the kernel library import it
without cycles.

The cache objects are module-level singletons mutated in place (never
reassigned); importers get a shared reference.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

import jax
import numpy as np
from jaxtyping import Array

from jaccpot._jax_compat import Tracer

__all__: list[str] = []

_M2L_CHUNK_AUTOTUNE_CACHE_MAX = 64
_m2l_chunk_autotune_cache: "OrderedDict[tuple[Any, ...], int]" = OrderedDict()
_GPU_M2L_AUTOTUNE_PAIR_BINS = (
    65_536,
    262_144,
    1_048_576,
    4_194_304,
)
_GPU_M2L_AUTOTUNE_SMALL_CANDIDATES = (512, 1024)
_GPU_M2L_AUTOTUNE_MEDIUM_CANDIDATES = (1024, 2048)
_GPU_M2L_AUTOTUNE_LARGE_CANDIDATES = (2048, 4096)
_GPU_M2L_AUTOTUNE_XL_CANDIDATES = (4096, 8192)
_GPU_M2L_AUTOTUNE_MAX_SAMPLE_PAIRS = 65_536
_GPU_M2L_AUTOTUNE_MAX_SAMPLE_NODES = 8_192


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


def _m2l_autotune_lookup(key: tuple[Any, ...]) -> Optional[int]:
    """Return cached M2L chunk size for an autotune signature.

    A hit refreshes the entry's LRU position, so lookups are not read-only with
    respect to eviction order.

    Parameters
    ----------
    key : tuple[Any, ...]
        Autotune signature.

    Returns
    -------
    Optional[int]
        Cached chunk size, or ``None`` on a miss.
    """

    cached = _m2l_chunk_autotune_cache.get(key)
    if cached is None:
        return None
    _m2l_chunk_autotune_cache.move_to_end(key)
    return int(cached)


def _m2l_autotune_store(key: tuple[Any, ...], chunk_size: int) -> None:
    """Store one autotuned M2L chunk size with LRU eviction.

    Parameters
    ----------
    key : tuple[Any, ...]
        Autotune signature.
    chunk_size : int
        Measured chunk size to remember.

    Returns
    -------
    None
        Mutates the process-level cache in place.
    """

    _m2l_chunk_autotune_cache[key] = int(chunk_size)
    _m2l_chunk_autotune_cache.move_to_end(key)
    while len(_m2l_chunk_autotune_cache) > _M2L_CHUNK_AUTOTUNE_CACHE_MAX:
        _m2l_chunk_autotune_cache.popitem(last=False)


def _m2l_autotune_payload() -> list[dict[str, Any]]:
    """Return a JSON-serializable snapshot of the global M2L autotune cache.

    Returns
    -------
    list[dict[str, Any]]
        One ``{"key": [...], "chunk_size": int}`` entry per cached signature, in
        LRU order (least recently used first).
    """

    payload: list[dict[str, Any]] = []
    for key, chunk in _m2l_chunk_autotune_cache.items():
        payload.append({"key": list(key), "chunk_size": int(chunk)})
    return payload


def _restore_m2l_autotune_payload(
    payload: list[dict[str, Any]],
    *,
    merge: bool = True,
) -> int:
    """Restore global M2L autotune cache entries from serialized payload.

    Deliberately lenient: malformed entries, non-list keys, unparseable chunk
    sizes and non-positive chunk sizes are **skipped silently** rather than
    raising, so one bad record in a cache file cannot break a run. The return
    value is how a caller detects that something was dropped.

    Parameters
    ----------
    payload : list[dict[str, Any]]
        Entries as produced by :func:`_m2l_autotune_payload`.
    merge : bool
        Merge into the existing cache rather than clearing it first.

    Returns
    -------
    int
        Number of entries actually restored, which may be fewer than were
        supplied.
    """

    if not merge:
        _m2l_chunk_autotune_cache.clear()
    restored = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        key_raw = item.get("key")
        chunk_raw = item.get("chunk_size")
        if not isinstance(key_raw, list):
            continue
        try:
            key = tuple(key_raw)
            chunk_size = int(chunk_raw)
        except Exception:
            continue
        if chunk_size <= 0:
            continue
        _m2l_autotune_store(key, chunk_size)
        restored += 1
    return int(restored)


def _clear_global_runtime_caches(*, clear_jax_compilation: bool = False) -> None:
    """Drop process-level runtime caches that can retain large array payloads.

    Since the grouped operator and segment caches went with the grouped far field
    (2026-10 cleanup, X3) no process-level cache here holds arrays, so only the
    optional JAX compilation clear is left. Note it does **not** clear the M2L
    autotune cache -- that holds only integers, so it costs nothing to keep and is
    worth preserving across a memory-pressure event.

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
