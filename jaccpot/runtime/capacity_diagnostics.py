"""Turn an out-of-memory or capacity failure into an actionable message.

Two failures from the tranche-1 measurements said nothing a caller could act on:

* ``preset="accurate"``, leaf 32, N=1048576: ``RESOURCE_EXHAUSTED: Out of memory
  while trying to allocate 8.00GiB`` on a 40 GB card. An 8 GiB *single*
  allocation on a 40 GB device is a capacity estimate, not a hard limit, and the
  message named neither the buffer nor a configuration that would fit. Cost
  figure 05 its top ladder point.
* ``preset="large_n_gpu"``, leaf 64, same N: the same thing at 4.00 GiB. Cost
  figure 07 its top point.

Rather than guess which buffer XLA was allocating, this module computes the
handful of allocations that are closed-form in (N, leaf, order, traversal caps),
reports them largest-first alongside the device's own free memory, and names the
knob for each. The re-raise chains the original error, so nothing is hidden.
"""

from __future__ import annotations

from typing import Any, Optional

from jaxtyping import DTypeLike

__all__ = [
    "capacity_report",
    "is_capacity_failure",
    "reraise_with_capacity_report",
]

_GIB = 1024.0**3


def _bytes_per_element(dtype_name: DTypeLike) -> int:
    # Stringifies whatever it is given, so the hint must admit JAX's scalar
    # types too -- `jnp.float32` is a `_ScalarMeta`, not a `str` (F40).
    return 8 if "64" in str(dtype_name) else 4


def capacity_report(
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    working_dtype: Any,
    preset: Optional[str],
    traversal_config: Any = None,
    nearfield_mode: Optional[str] = None,
) -> list[tuple[str, float, str]]:
    """Return ``[(buffer name, GiB, the knob that sizes it)]``, largest first.

    Every entry is closed-form in the arguments -- no device work -- so this is
    safe to call from an exception handler on a device that has just failed to
    allocate.

    Parameters
    ----------
    num_particles : int
        Particle count.
    leaf_size : int
        Leaf occupancy target.
    max_order : int
        Expansion order.
    working_dtype : Any
        Working dtype, for the per-element size.
    preset : Optional[str]
        Preset name, when one was used.
    traversal_config : Any
        Traversal capacities, which size the pair buffers.
    nearfield_mode : Optional[str]
        Near-field mode, which decides whether the bucketed buffers exist.

    Returns
    -------
    list[tuple[str, float, str]]
        ``(buffer name, GiB, sizing knob)`` per buffer, largest first. Estimates,
        not measurements -- they say what the configuration ASKS for, which is the
        useful thing when the allocation has already failed.
    """

    n = max(1, int(num_particles))
    leaf = max(1, int(leaf_size))
    p = max(0, int(max_order))
    itemsize = _bytes_per_element(getattr(working_dtype, "name", working_dtype))
    num_leaves = max(1, -(-n // leaf))
    num_nodes = 2 * num_leaves
    coeffs = (p + 1) * (p + 1)

    queue = block = per_node = per_leaf = 0
    if traversal_config is not None:
        queue = int(getattr(traversal_config, "max_pair_queue", 0) or 0)
        block = int(getattr(traversal_config, "process_block", 0) or 0)
        per_node = int(getattr(traversal_config, "max_interactions_per_node", 0) or 0)
        per_leaf = int(getattr(traversal_config, "max_neighbors_per_leaf", 0) or 0)

    entries: list[tuple[str, float, str]] = [
        (
            "far-field interaction buffer (total_nodes x max_interactions_per_node)",
            num_nodes * per_node * 4 / _GIB,
            "TraversalOverrides(max_interactions_per_node=...)",
        ),
        (
            "near-field neighbour buffer (num_leaves x max_neighbors_per_leaf)",
            num_leaves * per_leaf * 4 / _GIB,
            "TraversalOverrides(max_neighbors_per_leaf=...)",
        ),
        (
            "fast-lane source payload (num_leaves x max_neighbors_per_leaf x leaf)",
            num_leaves * per_leaf * leaf * (itemsize + 1) / _GIB,
            "leaf_size, or JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB to decline it",
        ),
        (
            "pair queue (max_pair_queue x 2 indices)",
            queue * 8 / _GIB,
            "TraversalOverrides(max_pair_queue=...)",
        ),
        (
            "multipole + local coefficients (2 x total_nodes x (p+1)^2)",
            2 * num_nodes * coeffs * itemsize / _GIB,
            "max_order",
        ),
        (
            "leaf-major particle payload (num_leaves x leaf x 4)",
            num_leaves * leaf * 4 * itemsize / _GIB,
            "leaf_size",
        ),
    ]
    if str(block or 0) and block:
        entries.append(
            (
                "traversal process block scratch (process_block x max_pair_queue)",
                0.0,  # not a materialised buffer; listed so the knob is visible
                "TraversalOverrides(process_block=...)",
            )
        )
    return sorted((e for e in entries if e[1] > 0.0), key=lambda e: e[1], reverse=True)


def _device_memory_gib() -> Optional[tuple[float, float]]:
    """``(bytes_in_use, bytes_limit)`` in GiB for the default device, if exposed.

    Returns
    -------
    Optional[tuple[float, float]]
        ``(in_use, limit)`` in GiB, or ``None`` when the backend does not expose
        memory stats -- CPU, for instance. Callers must treat ``None`` as "unknown",
        not as zero.
    """

    try:
        import jax

        stats = jax.devices()[0].memory_stats()
    except Exception:  # pragma: no cover - not all backends expose this
        return None
    if not stats:
        return None
    limit = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
    in_use = stats.get("bytes_in_use")
    if limit is None or in_use is None:
        return None
    return float(in_use) / _GIB, float(limit) / _GIB


def reraise_with_capacity_report(
    exc: BaseException,
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    working_dtype: Any,
    preset: Optional[str],
    traversal_config: Any = None,
    nearfield_mode: Optional[str] = None,
) -> None:
    """Re-raise ``exc`` with the buffer sizes and a configuration that fits.

    Only for allocation and capacity failures -- the caller checks. Chains the
    original with ``from exc`` so the XLA message and its allocation size stay
    visible.

    Parameters
    ----------
    exc : BaseException
        The original failure. Chained with ``from exc``.
    num_particles : int
        Particle count.
    leaf_size : int
        Leaf occupancy target.
    max_order : int
        Expansion order.
    working_dtype : Any
        Working dtype, for the per-element size.
    preset : Optional[str]
        Preset name, when one was used.
    traversal_config : Any
        Traversal capacities, which size the pair buffers.
    nearfield_mode : Optional[str]
        Near-field mode, which decides whether the bucketed buffers exist.

    Returns
    -------
    None
        Never returns normally.

    Raises
    ------
    RuntimeError
        Always -- carrying the buffer table and a suggested configuration, with
        ``exc`` chained so the XLA message and its allocation size stay visible.
    """

    entries = capacity_report(
        num_particles=num_particles,
        leaf_size=leaf_size,
        max_order=max_order,
        working_dtype=working_dtype,
        preset=preset,
        traversal_config=traversal_config,
        nearfield_mode=nearfield_mode,
    )
    lines = [
        f"jaccpot could not fit N={int(num_particles)} with preset={preset!r}, "
        f"leaf_size={int(leaf_size)}, max_order={int(max_order)}, "
        f"dtype={getattr(working_dtype, 'name', working_dtype)}."
    ]
    memory = _device_memory_gib()
    if memory is not None:
        lines.append(f"Device: {memory[0]:.2f} GiB in use of {memory[1]:.2f} GiB.")
    lines.append("Largest statically-sized buffers for this configuration:")
    for name, gib, knob in entries[:5]:
        lines.append(f"  {gib:8.2f} GiB  {name}")
        lines.append(f"           sized by: {knob}")
    lines.append(
        "What usually fits: preset='large_n_gpu' with leaf_size=256 keeps the "
        "fast-lane payload and the near-field neighbour buffer bounded (it is the "
        "path built for this N); raising leaf_size shrinks num_leaves and every "
        "per-leaf buffer with it, at some accuracy cost. To change ONE traversal "
        "capacity without disturbing the preset's other tuning, pass "
        "jaccpot.TraversalOverrides(...) rather than a full DualTreeTraversalConfig."
    )
    raise RuntimeError("\n".join(lines)) from exc


def is_capacity_failure(exc: BaseException) -> bool:
    """Whether ``exc`` is an allocation or a traversal-capacity overflow.
    Matched on message text, not exception type: XLA reports OOM as a generic
    error, so there is no class to test. That makes this a heuristic -- a false
    negative just means the caller re-raises without the buffer table.

    Parameters
    ----------
    exc : BaseException
        Exception to classify.

    Returns
    -------
    bool
        ``True`` when the message looks like an allocation or capacity failure.
    """

    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "resource_exhausted",
            "out of memory",
            "capacity exceeded",
            "cap exceeded",
            "overflowed",
        )
    )
