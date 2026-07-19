"""Shared tuning constants and environment helpers for the FMM runtime.

Leaf module extracted from ``_fmm_impl.py`` (Phase 2 of the runtime refactor).
It holds the dtype/GPU/CPU tuning thresholds, the prebuilt traversal-config
templates, the minimum-memory streamed-GPU traversal sizing helpers, and the
env-flag readers. It depends only on stdlib + numpy + yggdrax, so both the
orchestrator (``runtime.fmm``) and the kernel library (``runtime.kernels``) can
import it without cycles.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from yggdrax.interactions import DualTreeTraversalConfig

_MINIMUM_MEMORY_GPU_M2L_CHUNK_SIZE = 1024
_MINIMUM_MEMORY_CPU_M2L_CHUNK_SIZE = 4096
_GROUPED_SCHEDULE_BUDGET_DEFAULT = 32 * 1024 * 1024


_LARGE_CPU_PARTICLE_THRESHOLD = 65536
_CLASS_MAJOR_CPU_PARTICLE_THRESHOLD = 262144
# Bucketed near-field becomes beneficial on CPU at moderate N for the
# current fast/solidfmm path; keep threshold above tiny-N crossover noise.
_NEARFIELD_BUCKETED_CPU_PARTICLE_THRESHOLD = 1024
_NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_MEDIUM = 1024
_NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_LARGE = 2048
_NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_XL = 4096
_NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP = 16_000_000
_NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP_GPU = 4_000_000
_NEARFIELD_GPU_PRECOMPUTE_MAX_PARTICLES = 65_536
_NEARFIELD_SCATTER_SCHEDULE_INT32_ITEM_LIMIT = np.iinfo(np.int32).max
_LARGE_CPU_M2L_CHUNK_SIZE = 32768
_TRACING_MAX_NEIGHBORS_PER_LEAF = 512
_TRACING_MAX_PAIR_QUEUE = 65_536
_TRACING_MAX_PROCESS_BLOCK = 128
# Traced prepare_state uses static-capacity interaction buffers. This cap limits
# max_interactions_per_node only in traced mode (outer jax.jit prepare path) to
# keep padded far-field buffers from dominating runtime. Lower is faster but can
# trigger traversal overflow/retry on harder particle configurations.
_TRACING_MAX_INTERACTIONS_PER_NODE = 512
_GPU_LARGE_PARTICLE_THRESHOLD = 65_536
_GPU_MIN_NEIGHBORS_PER_LEAF = 2048
_GPU_MIN_INTERACTIONS_PER_NODE = 8192
_GPU_MAX_NEIGHBORS_PER_LEAF = 2048
_GPU_MAX_INTERACTIONS_PER_NODE = 8192
_GPU_MIN_PAIR_QUEUE_MEDIUM = 131_072
_GPU_MIN_PAIR_QUEUE_LARGE = 262_144
_GPU_MIN_PAIR_QUEUE_XL = 524_288
_GPU_MINIMUM_MEMORY_PAIR_QUEUE = 32_768
_GPU_MINIMUM_MEMORY_PROCESS_BLOCK = 1024
_GPU_MINIMUM_MEMORY_INTERACTIONS_PER_NODE = 1_024
_GPU_MINIMUM_MEMORY_NEIGHBORS_PER_LEAF = 256
_GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_PAIR_QUEUE_LARGE = 262_144
_GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_PAIR_QUEUE_XL = 524_288
_GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_PROCESS_BLOCK = 256
_GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_INTERACTIONS_PER_NODE = 8_192
_GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_NEIGHBORS_PER_LEAF = 4_096
_LEGACY_STATIC_TRAVERSAL_INT32_ITEM_LIMIT = np.iinfo(np.int32).max
_LARGE_CPU_TRAVERSAL_CONFIG = DualTreeTraversalConfig(
    max_pair_queue=131072,
    process_block=4096,
    max_interactions_per_node=65536,
    max_neighbors_per_leaf=32768,
)
_KDTREE_DEFAULT_TRAVERSAL_CONFIG = DualTreeTraversalConfig(
    max_pair_queue=65536,
    process_block=64,
    max_interactions_per_node=512,
    max_neighbors_per_leaf=2048,
)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a positive integer from env with a defensive fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        val = int(str(raw).strip())
    except Exception:
        return int(default)
    return int(max(val, int(minimum)))


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean env flag with a defensive fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _minimum_memory_streamed_gpu_traversal_ceiling(
    *, num_particles: int
) -> DualTreeTraversalConfig:
    """Return explicit traversal ceilings for large-N streamed GPU runs.

    These ceilings mirror the lean engblom/streamed production profile that has
    been substantially more memory-efficient than oversized explicit traversal
    caps in large-N minimum-memory benchmarks.
    """

    n = int(num_particles)
    pair_queue = (
        _GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_PAIR_QUEUE_XL
        if n >= 4_194_304
        else _GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_PAIR_QUEUE_LARGE
    )
    return DualTreeTraversalConfig(
        max_pair_queue=int(pair_queue),
        process_block=int(_GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_PROCESS_BLOCK),
        max_interactions_per_node=int(
            _GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_INTERACTIONS_PER_NODE
        ),
        max_neighbors_per_leaf=int(
            _GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_NEIGHBORS_PER_LEAF
        ),
    )


def _minimum_memory_streamed_gpu_traversal_seed(
    *, num_particles: int
) -> DualTreeTraversalConfig:
    """Return deterministic minimum-memory traversal seed for production GPU runs.

    Keep a small queue seed for sub-million workloads, but use the streamed
    process-block floor to avoid underfilled count-pass kernels. Multi-million
    particle runs use a larger fixed seed to avoid early fail-fast traversal
    overflow.
    """

    n = int(num_particles)
    if n >= 4_194_304:
        default_config = _minimum_memory_streamed_gpu_traversal_ceiling(num_particles=n)
        return DualTreeTraversalConfig(
            max_pair_queue=_env_int(
                "JACCPOT_LARGE_N_GPU_MIN_MEMORY_PAIR_QUEUE",
                int(default_config.max_pair_queue),
                minimum=4,
            ),
            process_block=_env_int(
                "JACCPOT_LARGE_N_GPU_MIN_MEMORY_PROCESS_BLOCK",
                int(default_config.process_block),
                minimum=1,
            ),
            max_interactions_per_node=_env_int(
                "JACCPOT_LARGE_N_GPU_MIN_MEMORY_INTERACTIONS_PER_NODE",
                int(default_config.max_interactions_per_node),
                minimum=1,
            ),
            max_neighbors_per_leaf=_env_int(
                "JACCPOT_LARGE_N_GPU_MIN_MEMORY_NEIGHBORS_PER_LEAF",
                int(default_config.max_neighbors_per_leaf),
                minimum=1,
            ),
        )
    if n >= 1_048_576:
        default_config = DualTreeTraversalConfig(
            max_pair_queue=int(_GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_PAIR_QUEUE_LARGE),
            process_block=int(_GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_PROCESS_BLOCK),
            max_interactions_per_node=int(
                _GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_INTERACTIONS_PER_NODE
            ),
            max_neighbors_per_leaf=int(
                _GPU_STREAMED_MINIMUM_MEMORY_EXPLICIT_NEIGHBORS_PER_LEAF
            ),
        )
    else:
        default_config = DualTreeTraversalConfig(
            max_pair_queue=int(_GPU_MINIMUM_MEMORY_PAIR_QUEUE),
            process_block=int(_GPU_MINIMUM_MEMORY_PROCESS_BLOCK),
            max_interactions_per_node=int(_GPU_MINIMUM_MEMORY_INTERACTIONS_PER_NODE),
            max_neighbors_per_leaf=int(_GPU_MINIMUM_MEMORY_NEIGHBORS_PER_LEAF),
        )
    return DualTreeTraversalConfig(
        max_pair_queue=_env_int(
            "JACCPOT_LARGE_N_GPU_MIN_MEMORY_PAIR_QUEUE",
            int(default_config.max_pair_queue),
            minimum=4,
        ),
        process_block=_env_int(
            "JACCPOT_LARGE_N_GPU_MIN_MEMORY_PROCESS_BLOCK",
            int(default_config.process_block),
            minimum=1,
        ),
        max_interactions_per_node=_env_int(
            "JACCPOT_LARGE_N_GPU_MIN_MEMORY_INTERACTIONS_PER_NODE",
            int(default_config.max_interactions_per_node),
            minimum=1,
        ),
        max_neighbors_per_leaf=_env_int(
            "JACCPOT_LARGE_N_GPU_MIN_MEMORY_NEIGHBORS_PER_LEAF",
            int(default_config.max_neighbors_per_leaf),
            minimum=1,
        ),
    )


def _cap_minimum_memory_streamed_gpu_traversal_config_for_tree(
    *,
    traversal_config: Optional[DualTreeTraversalConfig],
    total_nodes: int,
    num_leaves: int,
    num_particles: int,
) -> Optional[DualTreeTraversalConfig]:
    """Clamp impossible explicit traversal seeds for legacy large-N GPU walks.

    Yggdrax's legacy static-capacity traversal path materializes far/near
    buffers sized by:
    - ``total_nodes * max_interactions_per_node``
    - ``num_leaves * max_neighbors_per_leaf``

    For very large radix trees, oversized explicit seeds can overflow signed
    int32 shape scalars before traversal starts, or force enormous flat buffers
    that are guaranteed to exhaust device memory. Cap only the impossible cases
    to the existing lean streamed-GPU ceiling, while preserving smaller explicit
    configs unchanged.
    """

    if traversal_config is None:
        return None

    safe_total_nodes = max(1, int(total_nodes))
    safe_num_leaves = max(1, int(num_leaves))
    current_queue = int(traversal_config.max_pair_queue)
    current_block = int(traversal_config.process_block)
    current_interactions = int(traversal_config.max_interactions_per_node)
    current_neighbors = int(traversal_config.max_neighbors_per_leaf)

    far_slots = safe_total_nodes * current_interactions
    near_slots = safe_num_leaves * current_neighbors
    if far_slots <= int(
        _LEGACY_STATIC_TRAVERSAL_INT32_ITEM_LIMIT
    ) and near_slots <= int(_LEGACY_STATIC_TRAVERSAL_INT32_ITEM_LIMIT):
        return traversal_config

    explicit_ceiling = _minimum_memory_streamed_gpu_traversal_ceiling(
        num_particles=int(num_particles)
    )
    int32_far_cap = max(
        1, int(_LEGACY_STATIC_TRAVERSAL_INT32_ITEM_LIMIT) // safe_total_nodes
    )
    int32_near_cap = max(
        1, int(_LEGACY_STATIC_TRAVERSAL_INT32_ITEM_LIMIT) // safe_num_leaves
    )
    capped = DualTreeTraversalConfig(
        max_pair_queue=int(min(current_queue, int(explicit_ceiling.max_pair_queue))),
        process_block=int(min(current_block, int(explicit_ceiling.process_block))),
        max_interactions_per_node=int(
            min(
                current_interactions,
                int(explicit_ceiling.max_interactions_per_node),
                int32_far_cap,
            )
        ),
        max_neighbors_per_leaf=int(
            min(
                current_neighbors,
                int(explicit_ceiling.max_neighbors_per_leaf),
                int32_near_cap,
            )
        ),
    )
    return capped


_PREPARE_DIAGNOSTICS = _env_flag("JACCPOT_PREPARE_DIAGNOSTICS", False)


def _prepare_diag(message: str) -> None:
    """Emit opt-in prepare diagnostics to stdout."""
    if _PREPARE_DIAGNOSTICS:
        print(f"[jaccpot.prepare] {message}", flush=True)
