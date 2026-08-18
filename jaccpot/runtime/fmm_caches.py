"""Process-level runtime caches + byte accounting for the FMM runtime.

Leaf module extracted from ``_fmm_impl.py`` (Phase 2 of the runtime refactor).
It owns the mutable operator/segment/autotune caches and every helper that
keys, sizes, evicts, serializes, or clears them, so the single source of truth
for this shared state lives in one place. Depends only on stdlib + jax + numpy
+ ``fmm_constants``, so both the orchestrator and the kernel library import it
without cycles.

The cache objects are module-level singletons mutated in place (never
reassigned); importers get a shared reference.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .fmm_constants import _env_int

__all__: list[str] = []

_GROUPED_OPERATOR_CACHE_MAX = _env_int("JACCPOT_GROUPED_OPERATOR_CACHE_MAX", 32)
_grouped_operator_blocks_cache: "OrderedDict[tuple, tuple[Array, Array]]" = (
    OrderedDict()
)
_GROUPED_SEGMENT_CACHE_MAX = _env_int("JACCPOT_GROUPED_SEGMENT_CACHE_MAX", 32)
_grouped_segment_cache: "OrderedDict[tuple, tuple[Array, Array, Array]]" = OrderedDict()
_GROUPED_OPERATOR_CACHE_ENTRY_MAX_BYTES = _env_int(
    "JACCPOT_GROUPED_OPERATOR_CACHE_ENTRY_MAX_BYTES",
    64 * 1024 * 1024,
)
_GROUPED_OPERATOR_CACHE_TOTAL_MAX_BYTES = _env_int(
    "JACCPOT_GROUPED_OPERATOR_CACHE_TOTAL_MAX_BYTES",
    256 * 1024 * 1024,
)
_GROUPED_SEGMENT_CACHE_ENTRY_MAX_BYTES = _env_int(
    "JACCPOT_GROUPED_SEGMENT_CACHE_ENTRY_MAX_BYTES",
    32 * 1024 * 1024,
)
_GROUPED_SEGMENT_CACHE_TOTAL_MAX_BYTES = _env_int(
    "JACCPOT_GROUPED_SEGMENT_CACHE_TOTAL_MAX_BYTES",
    128 * 1024 * 1024,
)
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
# Keep full-batch M2L kernels for genuinely small pair sets only; larger sets use
# chunked reduction to avoid pair_count-scaled temporaries on GPU.
_M2L_FULLBATCH_MAX_PAIRS = 2_048


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


def _tuple_array_nbytes(value: tuple[Array, ...]) -> int:
    """Return total bytes for a tuple of array leaves.

    Parameters
    ----------
    value : tuple[Array, ...]
        Cache entry as stored by the grouped caches.

    Returns
    -------
    int
        Sum of :func:`_array_nbytes` over the tuple.
    """
    return int(sum(_array_nbytes(arr) for arr in value))


def _ordered_dict_values_nbytes(cache: OrderedDict) -> int:
    """Return cumulative bytes of array-valued OrderedDict entries.

    Recomputed from scratch on every call, and every ``*_cache_put`` calls it in
    its eviction loop, so it is O(entries x leaves) per insertion. That is fine at
    the current cache caps (32 entries) and is the thing to revisit if they grow.

    Parameters
    ----------
    cache : OrderedDict
        Cache whose values are tuples of arrays.

    Returns
    -------
    int
        Total bytes across all entries.
    """
    total = 0
    for value in cache.values():
        total += _tuple_array_nbytes(value)
    return int(total)


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

    Clears the grouped operator and segment caches. Note it does **not** clear the
    M2L autotune cache -- that holds only integers, so it costs nothing to keep and
    is worth preserving across a memory-pressure event.

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
    _grouped_operator_blocks_cache.clear()
    _grouped_segment_cache.clear()
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
    return any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(value)
    )


def _array_digest(arr: Array) -> Optional[tuple[tuple[int, ...], str, bytes]]:
    """Return (shape, dtype, digest) for stable host-side cache keys.

    Digests the array's **contents**, so two structurally identical arrays with
    different values get different keys. This forces a device-to-host transfer and
    a full hash of the buffer, which is why it is only used to build prepare-time
    keys and never on a per-evaluation path.

    Parameters
    ----------
    arr : Array
        Array to digest.

    Returns
    -------
    Optional[tuple[tuple[int, ...], str, bytes]]
        ``(shape, dtype, blake2b-128 digest)``, or ``None`` when the array is
        traced and therefore uncacheable.
    """
    if _contains_tracer(arr):
        return None
    arr_np = np.asarray(jax.device_get(arr))
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(arr_np.tobytes())
    return tuple(int(v) for v in arr_np.shape), str(arr_np.dtype), hasher.digest()


def _grouped_operator_cache_key(
    *,
    order: int,
    rotation: str,
    dtype: jnp.dtype,
    class_keys: Array,
    class_deltas: Array,
) -> Optional[tuple]:
    """Build a content-addressed key for the grouped rotation-block cache.

    Parameters
    ----------
    order : int
        Expansion order.
    rotation : str
        Rotation lane name.
    dtype : jnp.dtype
        Working dtype of the blocks.
    class_keys : Array
        Translation-class keys; digested by content.
    class_deltas : Array
        Per-class displacement vectors; digested by content.

    Returns
    -------
    Optional[tuple]
        The key, or ``None`` if either array is traced -- in which case the
        caller must skip caching rather than fall back to a weaker key.
    """
    keys_sig = _array_digest(class_keys)
    deltas_sig = _array_digest(class_deltas)
    if keys_sig is None or deltas_sig is None:
        return None
    return (
        int(order),
        str(rotation),
        str(dtype),
        keys_sig,
        deltas_sig,
    )


def _grouped_segment_cache_key(
    *,
    class_offsets: Array,
    class_targets: Array,
    chunk_size: int,
) -> Optional[tuple]:
    """Build a content-addressed key for the grouped segment-table cache.

    Parameters
    ----------
    class_offsets : Array
        Per-class segment offsets; digested by content.
    class_targets : Array
        Per-class target nodes; digested by content.
    chunk_size : int
        Chunk size the tables were built for; part of the key because the
        segmentation depends on it.

    Returns
    -------
    Optional[tuple]
        The key, or ``None`` if either array is traced.
    """
    offsets_sig = _array_digest(class_offsets)
    targets_sig = _array_digest(class_targets)
    if offsets_sig is None or targets_sig is None:
        return None
    return (int(chunk_size), offsets_sig, targets_sig)


def _grouped_operator_cache_get(key: tuple) -> Optional[tuple[Array, Array]]:
    """Look up cached grouped rotation blocks, refreshing their LRU position.

    Parameters
    ----------
    key : tuple
        Key from :func:`_grouped_operator_cache_key`.

    Returns
    -------
    Optional[tuple[Array, Array]]
        The cached blocks, or ``None`` on a miss.
    """
    blocks = _grouped_operator_blocks_cache.get(key)
    if blocks is None:
        return None
    _grouped_operator_blocks_cache.move_to_end(key)
    return blocks


def _grouped_operator_cache_put(key: tuple, value: tuple[Array, Array]) -> None:
    """Insert grouped rotation blocks, honouring the entry and total byte caps.

    An entry larger than the per-entry cap is **dropped rather than stored**, so a
    single oversized payload cannot evict the whole cache. Insertion then evicts
    least-recently-used entries until both the count and total-byte caps hold.

    Parameters
    ----------
    key : tuple
        Key from :func:`_grouped_operator_cache_key`.
    value : tuple[Array, Array]
        Blocks to cache.

    Returns
    -------
    None
        Mutates the process-level cache in place.
    """
    if _tuple_array_nbytes(value) > _GROUPED_OPERATOR_CACHE_ENTRY_MAX_BYTES:
        return
    _grouped_operator_blocks_cache[key] = value
    _grouped_operator_blocks_cache.move_to_end(key)
    while (
        len(_grouped_operator_blocks_cache) > _GROUPED_OPERATOR_CACHE_MAX
        or _ordered_dict_values_nbytes(_grouped_operator_blocks_cache)
        > _GROUPED_OPERATOR_CACHE_TOTAL_MAX_BYTES
    ):
        _grouped_operator_blocks_cache.popitem(last=False)


def _grouped_segment_cache_get(
    key: tuple,
) -> Optional[tuple[Array, Array, Array]]:
    """Look up cached grouped segment tables, refreshing their LRU position.

    Parameters
    ----------
    key : tuple
        Key from :func:`_grouped_segment_cache_key`.

    Returns
    -------
    Optional[tuple[Array, Array, Array]]
        The cached tables, or ``None`` on a miss.
    """
    cached = _grouped_segment_cache.get(key)
    if cached is None:
        return None
    _grouped_segment_cache.move_to_end(key)
    return cached


def _grouped_segment_cache_put(
    key: tuple,
    value: tuple[Array, Array, Array],
) -> None:
    """Insert grouped segment tables, honouring the entry and total byte caps.

    Same drop-if-oversized and LRU-evict-to-fit policy as
    :func:`_grouped_operator_cache_put`, against the segment cache's own caps.

    Parameters
    ----------
    key : tuple
        Key from :func:`_grouped_segment_cache_key`.
    value : tuple[Array, Array, Array]
        Tables to cache.

    Returns
    -------
    None
        Mutates the process-level cache in place.
    """
    if _tuple_array_nbytes(value) > _GROUPED_SEGMENT_CACHE_ENTRY_MAX_BYTES:
        return
    _grouped_segment_cache[key] = value
    _grouped_segment_cache.move_to_end(key)
    while (
        len(_grouped_segment_cache) > _GROUPED_SEGMENT_CACHE_MAX
        or _ordered_dict_values_nbytes(_grouped_segment_cache)
        > _GROUPED_SEGMENT_CACHE_TOTAL_MAX_BYTES
    ):
        _grouped_segment_cache.popitem(last=False)
