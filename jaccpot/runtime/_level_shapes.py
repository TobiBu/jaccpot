"""Static per-level batch widths for the level cascades (plan sub-10ms, Phase 3 in pure JAX).

The M2M upward sweep (``aggregate_m2m_real_by_level``) and the L2L cascade
(``_propagate_solidfmm_locals_by_level``) run one iteration per tree level.
Their per-level work has to have a STATIC shape, and both used a batch of
``num_internal`` nodes (M2M) or every internal node masked by level (L2L), so
each level cost the whole tree: ``nodes x depth``. On the balanced bucket tree
(depth 12) that was tolerable; the radix tree over Morton-cell leaves is 39
levels deep at N = 2x10^5 and the L2L alone took 44 ms per step in its
full-array scatter.

The right static width is the widest LEVEL, not the node count. Under trace the
level table is a tracer, so the width comes from a process-level registry keyed
by the tree's static shape ``(total_nodes, num_internal)``: the eager prepare
that always precedes a traced refresh fills it (with headroom, never
shrinking), and the traced rebuild checks the live tree against it
(:func:`level_width_overflow`) so a wider level trips the capacity guard rather
than dropping nodes.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaccpot._jax_compat import Tracer

__all__ = [
    "level_batch_width",
    "level_width_overflow",
    "registered_level_batch_width",
    "registered_num_levels",
    "pallas_cascades_enabled",
]

_WIDTHS: dict[tuple[int, int], int] = {}
_LEVELS: dict[tuple[int, int], int] = {}
_HEADROOM = 1.25
_LEVEL_HEADROOM = 8


def _key(total_nodes: int, num_internal: int) -> tuple[int, int]:
    return (int(total_nodes), int(num_internal))


def registered_level_batch_width(*, total_nodes: int, num_internal: int) -> int | None:
    """The registered width for this tree shape, or ``None`` before any eager visit.

    Parameters
    ----------
    total_nodes : int
        Node count of the tree.
    num_internal : int
        Internal node count.

    Returns
    -------
    int | None
        The stashed static width.
    """
    return _WIDTHS.get(_key(total_nodes, num_internal))


def level_batch_width(level_offsets: Array, *, total_nodes: int, num_internal: int) -> int:
    """Static batch width for the level loops: the widest level, with headroom.

    Concrete ``level_offsets`` (eager) measure the tree and raise the registry
    entry to ``ceil(1.25 x widest level)``, never lowering it; a traced table
    returns the registry entry, or ``num_internal`` (the historical width) when
    nothing eager has been seen for this shape.

    Parameters
    ----------
    level_offsets : Array
        ``(levels + 1,)`` start offsets into ``nodes_by_level``.
    total_nodes : int
        Node count (registry key).
    num_internal : int
        Internal node count (registry key and fallback width).

    Returns
    -------
    int
        A Python int in ``[1, total_nodes]``.
    """
    key = _key(total_nodes, num_internal)
    if isinstance(level_offsets, Tracer):
        return int(_WIDTHS.get(key) or max(int(num_internal), 1))
    offs = np.asarray(jax.device_get(level_offsets)).astype(np.int64)
    counts = np.diff(offs) if offs.size >= 2 else np.ones((1,), np.int64)
    widest = int(np.max(counts))
    live = int(np.count_nonzero(counts))
    levels = min(int(level_offsets.shape[0]) - 1, live + _LEVEL_HEADROOM)
    _LEVELS[key] = max(levels, _LEVELS.get(key, 0))
    width = max(1, int(math.ceil(widest * _HEADROOM)))
    width = min(width, max(int(total_nodes), 1))
    width = max(width, _WIDTHS.get(key, 0))
    _WIDTHS[key] = width
    return width


def registered_num_levels(*, total_nodes: int, num_internal: int) -> int | None:
    """Static level-loop trip count for this tree shape: live levels + 8 headroom.

    Filled by :func:`level_batch_width` on an eager visit (never lowered);
    ``None`` before any. The traced rebuild guards ``depth > bound`` elsewhere.

    Parameters
    ----------
    total_nodes : int
        Node count (registry key).
    num_internal : int
        Internal node count (registry key).

    Returns
    -------
    int | None
        The stashed static level count.
    """
    return _LEVELS.get(_key(total_nodes, num_internal))


def pallas_cascades_enabled() -> bool:
    """Whether the M2M/L2L cascades run as one Pallas launch per level (plan sub-10ms 3).

    ``JACCPOT_CASCADE_PALLAS=1`` selects :mod:`jaccpot.pallas.cascade_real_level`
    on the real basis where its Triton lowering runs (or under
    ``JACCPOT_CASCADE_PALLAS_INTERPRET=1`` anywhere); default off.

    Returns
    -------
    bool
        ``True`` when the Pallas cascades are selected.
    """
    from jaccpot._env import env_flag

    if not env_flag("JACCPOT_CASCADE_PALLAS", True):  # default on since Phase 6 (2026-09-11)
        return False
    if env_flag("JACCPOT_CASCADE_PALLAS_INTERPRET", False):
        return True
    from jaccpot.pallas.cascade_real_level import pallas_cascade_level_supported

    return bool(pallas_cascade_level_supported())


def level_width_overflow(level_offsets: Array, *, total_nodes: int, num_internal: int) -> Array:
    """Traced check that no level of a rebuilt tree exceeds the registered width.

    Parameters
    ----------
    level_offsets : Array
        ``(levels + 1,)`` start offsets of the rebuilt tree.
    total_nodes : int
        Node count (registry key).
    num_internal : int
        Internal node count (registry key).

    Returns
    -------
    Array
        Boolean scalar; ``True`` when a level is wider than the loops' batch.
    """
    width = registered_level_batch_width(total_nodes=total_nodes, num_internal=num_internal)
    if width is None:
        return jnp.asarray(False)
    offs = jnp.asarray(level_offsets)
    return jnp.max(offs[1:] - offs[:-1]) > jnp.asarray(int(width), offs.dtype)
