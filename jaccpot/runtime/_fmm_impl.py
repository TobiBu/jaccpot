"""
Fast Multipole Method (FMM) for computing gravitational accelerations.

This implementation uses multipole and local expansions to compute
gravitational forces in O(N) time instead of O(N^2) for direct summation.
"""

import hashlib
import json
import math
import os
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from math import comb
from typing import Any, Literal, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Callable, Tuple
from jaxtyping import Array, DTypeLike, jaxtyped
from yggdrax import build_tree
from yggdrax.dense_interactions import DenseInteractionBuffers
from yggdrax.geometry import compute_tree_geometry
from yggdrax.grouped_interactions import GroupedInteractionBuffers
from yggdrax.interactions import (
    CompactTaggedFarPairs,
    CompactTaggedOctreeFarPairs,
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    DualTreeWalkResult,
    MACType,
    NodeInteractionList,
    NodeNeighborList,
    OctreeNativeNeighborList,
    build_interactions_and_neighbors,
    build_octree_native_far_pairs,
    build_octree_native_neighbor_lists,
    build_well_separated_interactions,
)
from yggdrax.morton import morton_encode
from yggdrax.tree import (
    RadixTree,
    Tree,
    TreeType,
    available_tree_types,
    get_node_levels,
    rebuild_static_radix_tree_from_template,
    reorder_particles_by_indices,
)
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.basis.real_sh import complex_to_real_coeffs
from jaccpot.config import FMMExecutionBackend, MemoryObjective
from jaccpot.downward.local_expansions import (
    LocalExpansionData,
    TreeDownwardData,
    initialize_local_expansions,
)
from jaccpot.downward.local_expansions import (
    prepare_downward_sweep as prepare_tree_downward_sweep,
)
from jaccpot.downward.local_expansions import (
    run_downward_sweep as run_tree_downward_sweep,
)
from jaccpot.downward.local_expansions import translate_local_expansion
from jaccpot.nearfield.near_field import (
    compute_leaf_p2p_accelerations,
    compute_leaf_p2p_accelerations_large_n_accel_only,
    prepare_bucketed_scatter_schedules,
    prepare_bucketed_scatter_schedules_from_groups,
    prepare_leaf_neighbor_pairs,
)
from jaccpot.operators.complex_ops import (
    complex_rotation_blocks_from_z_solidfmm_batch,
    complex_rotation_blocks_to_z_solidfmm_batch,
    enforce_conjugate_symmetry_batch,
    evaluate_local_complex_derivative_tower_batch,
    evaluate_local_complex_grad_analytic,
    evaluate_local_complex_grad_analytic_batch,
    evaluate_local_complex_grad_analytic_preserve_dtype,
    evaluate_local_complex_grad_order4_unrolled,
    evaluate_local_complex_with_grad,
    evaluate_local_complex_with_grad_analytic_batch,
    evaluate_local_complex_with_grad_batch,
    l2l_complex,
    l2l_complex_batch,
    m2l_complex_reference,
    m2l_complex_reference_batch,
    m2l_complex_reference_batch_cached_blocks,
)
from jaccpot.operators.m2l_real_rot_scale import (
    m2l_rot_scale_real_batch,
    m2l_rot_scale_real_batch_cached_blocks,
    real_rotation_blocks_from_z_local_batch,
    real_rotation_blocks_to_z_multipole_batch,
)
from jaccpot.operators.multipole_utils import (
    LOCAL_LEVEL_COMBOS,
    MAX_MULTIPOLE_ORDER,
    level_offset,
    total_coefficients,
)
from jaccpot.operators.real_harmonics import (
    complex_to_dehnen_real_coeffs,
    evaluate_local_real_derivative_tower_batch,
    evaluate_local_real_with_grad,
    l2l_real,
    sh_size,
)
from jaccpot.operators.symmetric_tensors import (
    component_lift_index_map_3d,
    contract_symmetric_one_axis_3d,
)
from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_source_motion_multipoles,
    prepare_solidfmm_complex_upward_sweep,
)
from jaccpot.upward.tree_expansions import (
    NodeMultipoleData,
    TreeUpwardData,
)
from jaccpot.upward.tree_expansions import (
    prepare_upward_sweep as prepare_tree_upward_sweep,
)

from ._adaptive_policy import (
    adaptive_pair_policy,
    adaptive_policy_tolerance,
    bucket_far_pairs_by_tag,
    build_adaptive_policy_state,
    compute_node_force_scale_from_sorted_acc,
    source_error_proxy_by_order_from_multipoles,
)
from ._interaction_cache import (
    _build_dual_tree_artifacts,
    _compiled_refresh_dual_planner_route,
    _DualTreeArtifacts,
    _interaction_cache_key,
    _InteractionCacheEntry,
    _RefreshDualPlannerHint,
)
from ._large_n_pipeline import (
    can_use_large_n_prepare_path,
    evaluate_large_n_state,
    prepare_large_n_state,
)
from ._large_n_types import LargeNPreparedState
from ._nearfield_cache import (
    NearfieldPrecomputeArtifacts,
    nearfield_cache_matches,
    nearfield_from_cache,
    with_nearfield_cache_artifacts,
)
from ._octree_adapter import (
    OctreeExecutionData,
    build_octree_execution_data,
    build_octree_execution_data_with_status,
)
from ._octree_fmm import (
    OctreeSolidFMMComplexMultipoles,
    OctreeSolidFMMDownwardPlan,
    accumulate_octree_solidfmm_m2l,
    build_octree_downward_plan,
    build_octree_interaction_plan,
    build_octree_interaction_plan_from_native_pairs,
    build_octree_upward_plan,
    prepare_octree_solidfmm_complex_multipoles,
    propagate_octree_solidfmm_l2l,
)
from .dtypes import INDEX_DTYPE, as_index, complex_dtype_for_real
from .fmm_presets import FMMPreset, FMMPresetConfig, get_preset_config
from .reference import MultipoleExpansion
from .reference import compute_expansion as reference_compute_expansion
from .reference import compute_gravitational_potential as reference_compute_potential
from .reference import direct_sum as reference_direct_sum
from .reference import evaluate_expansion as reference_evaluate_expansion

ExpansionBasis = Literal["cartesian", "solidfmm", "complex"]
FarFieldMode = Literal["auto", "pair_grouped", "class_major"]
NearFieldMode = Literal["auto", "baseline", "bucketed"]
JerkMode = Literal["fast_approx", "accurate"]
PackedAccelerationDerivatives = tuple[Array, ...]
PreparedStateLike = Union["FMMPreparedState", LargeNPreparedState]

_STRICT_REFRESH_DIAG_MODES = frozenset(
    {
        "full",
        "tree_only",
        "upward_only",
        "downward_only",
        "eval_only",
        "integrator_only",
    }
)

_STRICT_REFRESH_DETAIL_DIAG_MODES = frozenset(
    {
        "full",
        "tree_sort_only",
        "tree_metadata_only",
        "p2m_only",
        "m2m_only",
        "m2l_only",
        "l2l_only",
        "downward_artifacts_only",
    }
)


def _velocity_verlet_state_update(
    state: Array,
    acceleration_current: Array,
    acceleration_new: Array,
    dt: Array,
) -> Array:
    """Complete a velocity-Verlet step after the endpoint force is known."""
    state_arr = jnp.asarray(state)
    dt_arr = jnp.asarray(dt, dtype=state_arr.dtype)
    position_new = (
        state_arr[:, 0]
        + state_arr[:, 1] * dt_arr
        + 0.5 * jnp.asarray(acceleration_current, dtype=state_arr.dtype) * dt_arr**2
    )
    velocity_new = (
        state_arr[:, 1]
        + 0.5
        * (
            jnp.asarray(acceleration_current, dtype=state_arr.dtype)
            + jnp.asarray(acceleration_new, dtype=state_arr.dtype)
        )
        * dt_arr
    )
    return state_arr.at[:, 0].set(position_new).at[:, 1].set(velocity_new)


def _normalize_strict_refresh_diag_mode(raw: object) -> str:
    mode = str(raw if raw is not None else "full").strip().lower()
    if mode not in _STRICT_REFRESH_DIAG_MODES:
        return "full"
    return mode


def _normalize_strict_refresh_detail_diag_mode(raw: object) -> str:
    mode = str(raw if raw is not None else "full").strip().lower()
    if mode not in _STRICT_REFRESH_DETAIL_DIAG_MODES:
        return "full"
    return mode


def _strict_refresh_diag_stage_flags(mode: str) -> tuple[bool, bool, bool, bool]:
    mode = _normalize_strict_refresh_diag_mode(mode)
    if mode == "integrator_only":
        return False, False, False, False
    if mode == "eval_only":
        return False, False, False, True
    if mode == "tree_only":
        return True, False, False, False
    if mode == "upward_only":
        return True, True, False, False
    if mode == "downward_only":
        return True, True, True, False
    return True, True, True, True


@dataclass(frozen=True)
class TreeBuilderConfig:
    """Resolved configuration controlling tree construction."""

    mode: str
    target_leaf_particles: int
    refine_local: bool
    max_refine_levels: int
    aspect_threshold: float


@dataclass(frozen=True)
class TraversalExecutionConfig:
    """Resolved configuration for traversal, batching, and dense buffers."""

    m2l_chunk_size: Optional[int]
    l2l_chunk_size: Optional[int]
    max_pair_queue: Optional[int]
    pair_process_block: Optional[int]
    traversal_config: Optional[DualTreeTraversalConfig]
    use_dense_interactions: bool
    jit_tree: Union[bool, Literal["auto"]]
    jit_traversal: bool


@dataclass(frozen=True)
class FMMResolvedConfig:
    """Container bundling all resolved FastMultipoleMethod options."""

    theta: float
    G: float
    softening: float
    working_dtype: Optional[DTypeLike]
    tree: TreeBuilderConfig
    traversal: TraversalExecutionConfig
    preset: Optional[str]


@dataclass(frozen=True)
class _TreeBuildArtifacts:
    """Outputs from a tree construction pass used by the FMM pipeline."""

    tree: Tree
    positions_sorted: Array
    masses_sorted: Array
    inverse_permutation: Array
    workspace: Optional[object]
    max_leaf_size: int
    cache_leaf_parameter: int


@dataclass(frozen=True)
class _TopologyReuseCandidate:
    """Candidate topology signature derived from current particle Morton order."""

    key: str
    sorted_indices: Array
    sorted_codes: Optional[Array] = None
    bounds: Optional[Tuple[Array, Array]] = None


@dataclass(frozen=True)
class _TopologyReuseEntry:
    """Cached topology metadata for bounded multi-step reuse."""

    key: str
    tree: Tree
    max_leaf_size: int
    cache_leaf_parameter: int
    reuse_count: int


@dataclass(frozen=True)
class _GeometryReuseEntry:
    """Cached tree geometry keyed by topology signature and input identity."""

    key: tuple[Any, ...]
    geometry: Any


class _RuntimeExecutionOverrides(NamedTuple):
    """Resolved runtime execution knobs after adaptive policy decisions."""

    traversal_config: Optional[DualTreeTraversalConfig]
    m2l_chunk_size: Optional[int]
    l2l_chunk_size: Optional[int]
    grouped_interactions: bool
    farfield_mode: str
    center_mode: str
    refine_local_override: Optional[bool]
    adaptive_applied: bool


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


_OPERATOR_CACHE_MAX = _env_int("JACCPOT_OPERATOR_CACHE_MAX", 512)
_operator_blocks_cache: "OrderedDict[tuple, tuple[Array, Array]]" = OrderedDict()
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
    """Return approximate storage size in bytes for one array leaf."""
    shape = tuple(int(dim) for dim in getattr(arr, "shape", ()))
    if len(shape) == 0:
        return int(np.dtype(arr.dtype).itemsize)
    return int(np.prod(np.asarray(shape, dtype=np.int64))) * int(
        np.dtype(arr.dtype).itemsize
    )


def _tuple_array_nbytes(value: tuple[Array, ...]) -> int:
    """Return total bytes for a tuple of array leaves."""
    return int(sum(_array_nbytes(arr) for arr in value))


def _ordered_dict_values_nbytes(cache: OrderedDict) -> int:
    """Return cumulative bytes of array-valued OrderedDict entries."""
    total = 0
    for value in cache.values():
        total += _tuple_array_nbytes(value)
    return int(total)


def _format_nbytes(count: int) -> str:
    value = float(max(int(count), 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TiB"


def _estimate_payload_nbytes(value: Any) -> int:
    """Best-effort recursive byte estimate for array-centric payloads."""
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
    """Return cached M2L chunk size for an autotune signature."""

    cached = _m2l_chunk_autotune_cache.get(key)
    if cached is None:
        return None
    _m2l_chunk_autotune_cache.move_to_end(key)
    return int(cached)


def _m2l_autotune_store(key: tuple[Any, ...], chunk_size: int) -> None:
    """Store one autotuned M2L chunk size with LRU eviction."""

    _m2l_chunk_autotune_cache[key] = int(chunk_size)
    _m2l_chunk_autotune_cache.move_to_end(key)
    while len(_m2l_chunk_autotune_cache) > _M2L_CHUNK_AUTOTUNE_CACHE_MAX:
        _m2l_chunk_autotune_cache.popitem(last=False)


def _m2l_autotune_payload() -> list[dict[str, Any]]:
    """Return a JSON-serializable snapshot of the global M2L autotune cache."""

    payload: list[dict[str, Any]] = []
    for key, chunk in _m2l_chunk_autotune_cache.items():
        payload.append({"key": list(key), "chunk_size": int(chunk)})
    return payload


def _restore_m2l_autotune_payload(
    payload: list[dict[str, Any]],
    *,
    merge: bool = True,
) -> int:
    """Restore global M2L autotune cache entries from serialized payload."""

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
    """Drop process-level runtime caches that can retain large array payloads."""
    _operator_blocks_cache.clear()
    _grouped_operator_blocks_cache.clear()
    _grouped_segment_cache.clear()
    if clear_jax_compilation:
        jax.clear_caches()


def _array_digest(arr: Array) -> Optional[tuple[tuple[int, ...], str, bytes]]:
    """Return (shape, dtype, digest) for stable host-side cache keys."""
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
    offsets_sig = _array_digest(class_offsets)
    targets_sig = _array_digest(class_targets)
    if offsets_sig is None or targets_sig is None:
        return None
    return (int(chunk_size), offsets_sig, targets_sig)


def _grouped_operator_cache_get(key: tuple) -> Optional[tuple[Array, Array]]:
    blocks = _grouped_operator_blocks_cache.get(key)
    if blocks is None:
        return None
    _grouped_operator_blocks_cache.move_to_end(key)
    return blocks


def _grouped_operator_cache_put(key: tuple, value: tuple[Array, Array]) -> None:
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
    cached = _grouped_segment_cache.get(key)
    if cached is None:
        return None
    _grouped_segment_cache.move_to_end(key)
    return cached


def _grouped_segment_cache_put(
    key: tuple,
    value: tuple[Array, Array, Array],
) -> None:
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


def _resolve_optional(value, preset_value, fallback):
    """Pick explicit value, then preset value, then fallback."""
    if value is not None:
        return value
    if preset_value is not None:
        return preset_value
    return fallback


def _resolve_fmm_config(
    *,
    theta: float,
    G: float,
    softening: float,
    working_dtype: Optional[DTypeLike],
    tree_build_mode: Optional[str],
    target_leaf_particles: Optional[int],
    refine_local: Optional[bool],
    max_refine_levels: Optional[int],
    aspect_threshold: Optional[float],
    m2l_chunk_size: Optional[int],
    l2l_chunk_size: Optional[int],
    max_pair_queue: Optional[int],
    pair_process_block: Optional[int],
    traversal_config: Optional[DualTreeTraversalConfig],
    use_dense_interactions: Optional[bool],
    preset_config: Optional[FMMPresetConfig],
) -> FMMResolvedConfig:
    """Normalize constructor inputs into a validated runtime configuration."""
    preset_name = preset_config.name if preset_config is not None else None
    preset_use_dense_interactions = (
        preset_config.use_dense_interactions if preset_config else None
    )

    tree_mode = _resolve_optional(
        tree_build_mode,
        preset_config.tree_build_mode if preset_config else None,
        "lbvh",
    )
    valid_tree_modes = {"lbvh", "fixed_depth", "adaptive", "static_radix"}
    if tree_mode not in valid_tree_modes:
        allowed_modes = sorted(valid_tree_modes)
        raise ValueError(f"tree_build_mode must be one of {allowed_modes}")

    leaf_target = _resolve_optional(
        target_leaf_particles,
        preset_config.target_leaf_particles if preset_config else None,
        32,
    )
    if int(leaf_target) < 1:
        raise ValueError("target_leaf_particles must be >= 1")

    tree_config = TreeBuilderConfig(
        mode=str(tree_mode),
        target_leaf_particles=int(leaf_target),
        refine_local=bool(
            _resolve_optional(
                refine_local,
                preset_config.refine_local if preset_config else None,
                False,
            )
        ),
        max_refine_levels=int(
            _resolve_optional(
                max_refine_levels,
                preset_config.max_refine_levels if preset_config else None,
                2,
            )
        ),
        aspect_threshold=float(
            _resolve_optional(
                aspect_threshold,
                preset_config.aspect_threshold if preset_config else None,
                8.0,
            )
        ),
    )

    jit_tree_cfg = _resolve_optional(
        None,
        preset_config.jit_tree if preset_config else None,
        "auto",
    )
    if jit_tree_cfg not in (True, False, "auto"):
        raise ValueError("jit_tree must be True, False, or 'auto'")

    traversal_cfg = TraversalExecutionConfig(
        m2l_chunk_size=_resolve_optional(
            m2l_chunk_size,
            preset_config.m2l_chunk_size if preset_config else None,
            None,
        ),
        l2l_chunk_size=_resolve_optional(
            l2l_chunk_size,
            preset_config.l2l_chunk_size if preset_config else None,
            None,
        ),
        max_pair_queue=_resolve_optional(
            max_pair_queue,
            preset_config.max_pair_queue if preset_config else None,
            None,
        ),
        pair_process_block=_resolve_optional(
            pair_process_block,
            preset_config.pair_process_block if preset_config else None,
            None,
        ),
        traversal_config=_resolve_optional(
            traversal_config,
            preset_config.traversal_config if preset_config else None,
            None,
        ),
        use_dense_interactions=bool(
            _resolve_optional(
                use_dense_interactions,
                preset_use_dense_interactions,
                False,
            )
        ),
        jit_tree=jit_tree_cfg,
        jit_traversal=bool(
            _resolve_optional(
                None,
                preset_config.jit_traversal if preset_config else None,
                True,
            )
        ),
    )

    preset_name = preset_config.name if preset_config is not None else None

    return FMMResolvedConfig(
        theta=float(theta),
        G=float(G),
        softening=float(softening),
        working_dtype=working_dtype,
        tree=tree_config,
        traversal=traversal_cfg,
        preset=(
            preset_name.value if isinstance(preset_name, FMMPreset) else preset_name
        ),
    )


def _build_tree_with_config(
    positions: Array,
    masses: Array,
    bounds: Tuple[Array, Array],
    *,
    tree_type: str,
    tree_config: TreeBuilderConfig,
    leaf_size: int,
    workspace: Optional[object],
    jit_tree: bool,
    refine_local: bool,
    max_refine_levels: int,
    aspect_threshold: float,
) -> _TreeBuildArtifacts:
    """Construct a tree according to the resolved builder configuration."""

    mode = tree_config.mode
    use_fast_lbvh_path = (
        bool(jit_tree)
        and tree_type == "radix"
        and mode == "lbvh"
        and not bool(refine_local)
    )
    if use_fast_lbvh_path:
        tree, pos_sorted, mass_sorted, inverse = _jit_radix_lbvh_builder(
            int(leaf_size)
        )(positions, masses, bounds)
        tree.require_fmm_topology()
        workspace_out = None
    else:
        build_mode = (
            "fixed_depth"
            if mode == "fixed_depth"
            else "static_radix" if mode == "static_radix" else "adaptive"
        )
        supports_workspace = tree_type == "radix" and mode != "static_radix"
        built_tree = Tree.from_particles(
            positions,
            masses,
            tree_type=tree_type,
            build_mode=build_mode,
            bounds=bounds,
            return_reordered=True,
            workspace=workspace if supports_workspace else None,  # type: ignore[arg-type]
            return_workspace=supports_workspace,
            leaf_size=int(leaf_size),
            target_leaf_particles=tree_config.target_leaf_particles,
            refine_local=refine_local,
            max_refine_levels=max_refine_levels,
            aspect_threshold=aspect_threshold,
        )
        built_tree.require_fmm_topology()
        tree = built_tree
        pos_sorted = built_tree.positions_sorted
        mass_sorted = built_tree.masses_sorted
        inverse = built_tree.inverse_permutation
        workspace_out = built_tree.workspace if tree_type == "radix" else None
    if pos_sorted is None or mass_sorted is None or inverse is None:
        raise ValueError(
            "Tree.from_particles must return reordered arrays for FMM runtime."
        )
    # Under outer jax.jit, converting a value-dependent leaf max to Python int
    # can trigger ConcretizationTypeError. Use the configured leaf-size contract
    # instead of inflating to N, so traced mode matches eager semantics.
    try:
        max_leaf_size = _max_leaf_size_from_tree(tree)
    except jax.errors.ConcretizationTypeError:
        max_leaf_size = int(leaf_size)
    cache_leaf_parameter = (
        int(leaf_size)
        if mode in {"lbvh", "static_radix"}
        else tree_config.target_leaf_particles
    )
    if mode != "fixed_depth" and int(max_leaf_size) > int(leaf_size):
        raise ValueError(
            "configured leaf_size is too small for built tree: "
            f"max_leaf_size={int(max_leaf_size)} > leaf_size={int(leaf_size)}"
        )

    return _TreeBuildArtifacts(
        tree=tree,
        positions_sorted=pos_sorted,
        masses_sorted=mass_sorted,
        inverse_permutation=inverse,
        workspace=workspace_out,
        max_leaf_size=int(max_leaf_size),
        cache_leaf_parameter=int(cache_leaf_parameter),
    )


@lru_cache(maxsize=16)
def _jit_radix_lbvh_builder(leaf_size: int):
    """Return cached jitted radix LBVH builder for a fixed leaf size."""

    leaf_size_int = int(leaf_size)
    return jax.jit(
        lambda p, m, b: build_tree(
            p,
            m,
            bounds=b,
            return_reordered=True,
            leaf_size=leaf_size_int,
        )
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FMMPreparedState:
    """Keep prepared tree artifacts resident as a JAX pytree payload.

    The array/tree payload is carried as pytree children so callers can pass
    this state through ``jax.jit``. Non-array metadata is tracked as static
    auxiliary data to avoid tracing errors on dtype/string objects.
    """

    tree: Tree
    upward: Optional[TreeUpwardData]
    downward: TreeDownwardData
    neighbor_list: NodeNeighborList
    max_leaf_size: int
    input_dtype: jnp.dtype
    working_dtype: jnp.dtype
    expansion_basis: ExpansionBasis
    theta: float
    topology_key: Optional[str]
    interactions: Optional[NodeInteractionList]
    dual_tree_result: Optional[DualTreeWalkResult]
    retry_events: Tuple[DualTreeRetryEvent, ...]
    nearfield_interop: Optional["NearfieldInteropData"]
    nearfield_target_leaf_ids: Optional[Array]
    nearfield_source_leaf_ids: Optional[Array]
    nearfield_valid_pairs: Optional[Array]
    nearfield_chunk_sort_indices: Optional[Array]
    nearfield_chunk_group_ids: Optional[Array]
    nearfield_chunk_unique_indices: Optional[Array]
    force_scale_nodes: Optional[Array]
    execution_backend: str = "radix"
    octree: Optional[OctreeExecutionData] = None
    octree_upward: Optional[OctreeSolidFMMComplexMultipoles] = None
    octree_downward: Optional[OctreeSolidFMMDownwardPlan] = None

    @property
    def positions_sorted(self: "FMMPreparedState") -> Array:
        """Canonical sorted particle positions owned by ``tree``."""
        value = getattr(self.tree, "positions_sorted", None)
        if value is None:
            raise ValueError("prepared tree is missing positions_sorted")
        return jnp.asarray(value)

    @property
    def masses_sorted(self: "FMMPreparedState") -> Array:
        """Canonical sorted particle masses owned by ``tree``."""
        value = getattr(self.tree, "masses_sorted", None)
        if value is None:
            raise ValueError("prepared tree is missing masses_sorted")
        return jnp.asarray(value)

    @property
    def inverse_permutation(self: "FMMPreparedState") -> Array:
        """Canonical inverse permutation owned by ``tree``."""
        value = getattr(self.tree, "inverse_permutation", None)
        if value is None:
            raise ValueError("prepared tree is missing inverse_permutation")
        return jnp.asarray(value, dtype=INDEX_DTYPE)

    def tree_flatten(
        self: "FMMPreparedState",
    ) -> tuple[
        tuple[Any, ...],
        tuple[
            int,
            str,
            str,
            str,
            float,
            Optional[str],
            Tuple[DualTreeRetryEvent, ...],
            str,
        ],
    ]:
        children = (
            self.tree,
            self.upward,
            self.downward,
            self.neighbor_list,
            self.interactions,
            self.dual_tree_result,
            self.nearfield_interop,
            self.nearfield_target_leaf_ids,
            self.nearfield_source_leaf_ids,
            self.nearfield_valid_pairs,
            self.nearfield_chunk_sort_indices,
            self.nearfield_chunk_group_ids,
            self.nearfield_chunk_unique_indices,
            self.force_scale_nodes,
            self.octree,
            self.octree_upward,
            self.octree_downward,
        )
        aux = (
            int(self.max_leaf_size),
            str(jnp.dtype(self.input_dtype)),
            str(jnp.dtype(self.working_dtype)),
            str(self.expansion_basis),
            float(self.theta),
            self.topology_key,
            self.retry_events,
            str(self.execution_backend),
        )
        return children, aux

    @classmethod
    def tree_unflatten(
        cls: type["FMMPreparedState"], aux: tuple[Any, ...], children: tuple[Any, ...]
    ) -> "FMMPreparedState":
        (
            max_leaf_size,
            input_dtype_name,
            working_dtype_name,
            expansion_basis,
            theta,
            topology_key,
            retry_events,
            execution_backend,
        ) = aux
        (
            tree,
            upward,
            downward,
            neighbor_list,
            interactions,
            dual_tree_result,
            nearfield_interop,
            nearfield_target_leaf_ids,
            nearfield_source_leaf_ids,
            nearfield_valid_pairs,
            nearfield_chunk_sort_indices,
            nearfield_chunk_group_ids,
            nearfield_chunk_unique_indices,
            force_scale_nodes,
            octree,
            octree_upward,
            octree_downward,
        ) = children
        return cls(
            tree=tree,
            upward=upward,
            downward=downward,
            neighbor_list=neighbor_list,
            max_leaf_size=int(max_leaf_size),
            input_dtype=jnp.dtype(input_dtype_name),
            working_dtype=jnp.dtype(working_dtype_name),
            expansion_basis=expansion_basis,
            theta=float(theta),
            topology_key=topology_key,
            interactions=interactions,
            dual_tree_result=dual_tree_result,
            retry_events=retry_events,
            nearfield_interop=nearfield_interop,
            nearfield_target_leaf_ids=nearfield_target_leaf_ids,
            nearfield_source_leaf_ids=nearfield_source_leaf_ids,
            nearfield_valid_pairs=nearfield_valid_pairs,
            nearfield_chunk_sort_indices=nearfield_chunk_sort_indices,
            nearfield_chunk_group_ids=nearfield_chunk_group_ids,
            nearfield_chunk_unique_indices=nearfield_chunk_unique_indices,
            force_scale_nodes=force_scale_nodes,
            execution_backend=str(execution_backend),
            octree=octree,
            octree_upward=octree_upward,
            octree_downward=octree_downward,
        )


class _PrepareStateTreeUpwardArtifacts(NamedTuple):
    """Tree/upward artifacts produced during prepare_state orchestration."""

    tree_mode: str
    tree: Tree
    positions_sorted: Array
    masses_sorted: Array
    inverse_permutation: Array
    leaf_cap: int
    leaf_parameter: int
    topology_key: Optional[str]
    upward: TreeUpwardData
    locals_template: Optional[LocalExpansionData]


class _PrepareStateDualDownwardArtifacts(NamedTuple):
    """Dual-tree and downward artifacts produced during prepare_state."""

    interactions: Optional[NodeInteractionList]
    neighbor_list: NodeNeighborList
    traversal_result: Optional[DualTreeWalkResult]
    compact_far_pairs: Optional[CompactTaggedFarPairs]
    downward: TreeDownwardData
    cache_entry: Optional[_InteractionCacheEntry]


class NearfieldInteropData(NamedTuple):
    """Explicit shared leaf/node view used to interoperate with nearfield code."""

    leaf_nodes: Array
    node_ranges: Array
    offsets: Array
    neighbors: Array
    counts: Array
    particle_order_node_ranges: Array
    particle_order_leaf_indices: Array
    particle_order_to_native_leaf: Array
    leaf_particle_indices: Optional[Array] = None
    leaf_particle_mask: Optional[Array] = None
    particle_to_leaf_position: Optional[Array] = None
    neighbor_leaf_positions: Optional[Array] = None


def _build_octree_upward_artifacts(
    *,
    octree: Optional[OctreeExecutionData],
    positions_sorted: Array,
    masses_sorted: Array,
    expansion_basis: ExpansionBasis,
    max_order: int,
) -> Optional[OctreeSolidFMMComplexMultipoles]:
    """Build octree-native upward artifacts when the execution tree exposes them."""

    if octree is None or expansion_basis != "solidfmm":
        return None
    plan = build_octree_upward_plan(octree)
    return prepare_octree_solidfmm_complex_multipoles(
        plan,
        positions_sorted,
        masses_sorted,
        max_order=int(max_order),
    )


def _prepared_state_upward_payload(
    *,
    upward: TreeUpwardData,
    memory_objective: str,
) -> Optional[TreeUpwardData]:
    """Return the upward payload to retain in prepared state.

    The plain prepared evaluation path uses `downward`, `tree`, and near-field
    metadata, but does not consume the original upward bundle. In
    minimum-memory mode we can therefore avoid retaining this large payload and
    reconstruct any advanced source-motion data later from the canonical sorted
    particle arrays if needed.
    """

    if str(memory_objective).strip().lower() == "minimum_memory":
        return None
    return upward


def _prepared_state_octree_upward_payload(
    *,
    octree_upward: Optional[OctreeSolidFMMComplexMultipoles],
    memory_objective: str,
) -> Optional[OctreeSolidFMMComplexMultipoles]:
    """Return the octree-upward payload to retain in prepared state."""

    if str(memory_objective).strip().lower() == "minimum_memory":
        return None
    return octree_upward


def _build_octree_downward_artifacts(
    *,
    octree: Optional[OctreeExecutionData],
    octree_upward: Optional[OctreeSolidFMMComplexMultipoles],
    interactions: Optional[NodeInteractionList],
    native_far_pairs: Optional[CompactTaggedOctreeFarPairs],
    execution_backend: str,
) -> Optional[OctreeSolidFMMDownwardPlan]:
    """Build octree-native downward scaffolding when prepared octree data exists."""

    if octree is None or octree_upward is None:
        return None
    if execution_backend == "octree" and native_far_pairs is not None:
        interaction_plan = build_octree_interaction_plan_from_native_pairs(
            octree,
            native_far_pairs,
        )
    elif interactions is not None:
        interaction_plan = build_octree_interaction_plan(octree, interactions)
    else:
        return None
    return build_octree_downward_plan(octree, octree_upward, interaction_plan)


def _finalize_octree_downward_artifacts(
    *,
    octree: Optional[OctreeExecutionData],
    octree_upward: Optional[OctreeSolidFMMComplexMultipoles],
    octree_downward: Optional[OctreeSolidFMMDownwardPlan],
    expansion_basis: ExpansionBasis,
    execution_backend: str,
    m2l_chunk_size: Optional[int],
) -> Optional[OctreeSolidFMMDownwardPlan]:
    """Run octree-native M2L/L2L when the narrow octree backend is active."""

    if (
        execution_backend != "octree"
        or expansion_basis != "solidfmm"
        or octree is None
        or octree_upward is None
        or octree_downward is None
    ):
        return octree_downward
    accumulated = accumulate_octree_solidfmm_m2l(
        octree_downward,
        octree_upward,
        chunk_size=4096 if m2l_chunk_size is None else int(m2l_chunk_size),
    )
    return propagate_octree_solidfmm_l2l(accumulated, octree)


def _octree_farfield_eval_inputs(state):
    """Far-field eval overrides that make the octree backend evaluate its OWN locals.

    For ``execution_backend == "octree"`` the octree upward/M2L/L2L pass fills octree-node-
    space local expansions (``state.octree_downward``), but the default far-field eval
    evaluates the radix locals. Passing these three overrides into the full-particle eval
    path evaluates the OCTREE locals at each particle instead. The near-field is already
    octree-native (``state.nearfield_interop``) and needs no override.

    The three outputs share the octree node-id space, and ``state.octree.node_ranges`` index
    into ``state.positions_sorted`` in the same (radix-Morton) order -- ``state.octree`` is
    derived from ``state.tree`` via ``build_octree_execution_data`` (which asserts root-range
    equality) -- so no re-permutation is needed. Returns ``(None, None, None)`` for non-octree
    backends or when the octree downward pass was not run.
    """
    if (
        str(getattr(state, "execution_backend", "radix")).strip().lower() != "octree"
        or getattr(state, "octree", None) is None
        or getattr(state, "octree_downward", None) is None
    ):
        return None, None, None
    downward = state.octree_downward
    coefficients = jnp.asarray(downward.locals_packed)
    farfield_local_data = LocalExpansionData(
        # Infer order from the (static) coefficient width. downward.order can be a
        # traced pytree leaf when compute_accelerations is jitted, so concretizing it
        # with int(...) raises ConcretizationTypeError; coefficients.shape[-1] is static.
        order=_infer_order_from_coeff_count(
            coeff_count=int(coefficients.shape[-1]),
            expansion_basis="solidfmm",
        ),
        centers=jnp.asarray(downward.centers),
        coefficients=coefficients,
    )
    farfield_leaf_nodes = jnp.asarray(state.octree.leaf_nodes, dtype=INDEX_DTYPE)
    farfield_node_ranges = jnp.asarray(state.octree.node_ranges, dtype=INDEX_DTYPE)
    return farfield_local_data, farfield_leaf_nodes, farfield_node_ranges


class _FarPairCOO(NamedTuple):
    """Compact COO-style far-pair representation for streamed M2L execution."""

    sources: Array
    targets: Array
    active_count: Optional[Array] = None


class _PrepareStateFarPairPlan(NamedTuple):
    """Far-pair payloads prepared for the downward sweep."""

    far_pairs_by_gear: Optional[tuple[tuple[Array, Array], ...]]
    far_pairs_coo: Optional[_FarPairCOO]
    adaptive_order_for_downward: bool
    p_gears_for_downward: tuple[int, ...]
    recent_far_pairs_by_gear_counts: tuple[int, ...]


class _SolidFMMDownwardInit(NamedTuple):
    """Resolved local-buffer initialization for solidfmm downward prep."""

    centers: Array
    locals_coeffs: Array
    total_nodes: int
    coeff_count: int
    dtype: Any


class _SolidFMMDownwardInteractionInputs(NamedTuple):
    """Resolved far-pair arrays for solidfmm downward prep."""

    interactions: NodeInteractionList
    src: Array
    tgt: Array
    pair_count: int
    active_pair_count: Array


class _SolidFMMDownwardMultipoleInputs(NamedTuple):
    """Resolved multipole coefficient payloads for downward accumulation."""

    multip_packed: Array
    source_motion_multip_packed: Optional[Array]
    multip_packed_kernel: Array
    rotation_mode: str


class _SolidFMMDownwardChildInputs(NamedTuple):
    """Resolved child-index arrays for L2L propagation."""

    num_internal_nodes: int
    left_child: Optional[Array]
    right_child: Optional[Array]


def _empty_interaction_storage_for_tree(
    tree: Tree,
    *,
    index_dtype: Any = INDEX_DTYPE,
) -> NodeInteractionList:
    """Construct a minimal zero-pair interaction list for a given tree."""

    total_nodes = int(jnp.asarray(tree.parent).shape[0])
    return NodeInteractionList(
        offsets=jnp.zeros((total_nodes + 1,), dtype=index_dtype),
        sources=jnp.zeros((0,), dtype=index_dtype),
        targets=jnp.zeros((0,), dtype=index_dtype),
        counts=jnp.zeros((total_nodes,), dtype=index_dtype),
        level_offsets=jnp.zeros((1,), dtype=index_dtype),
        target_levels=jnp.zeros((0,), dtype=index_dtype),
    )


def _empty_interaction_storage_like(
    interactions: Optional[NodeInteractionList],
) -> NodeInteractionList:
    """Return zero-pair interaction storage while preserving node-shaped metadata."""

    if interactions is None:
        raise ValueError("interactions must be present to derive empty storage")
    offsets = jnp.asarray(interactions.offsets)
    counts = jnp.asarray(interactions.counts)
    level_offsets = jnp.asarray(interactions.level_offsets)
    sources = jnp.zeros((0,), dtype=jnp.asarray(interactions.sources).dtype)
    targets = jnp.zeros((0,), dtype=jnp.asarray(interactions.targets).dtype)
    target_levels = jnp.zeros((0,), dtype=jnp.asarray(interactions.target_levels).dtype)
    return NodeInteractionList(
        offsets=jnp.zeros_like(offsets),
        sources=sources,
        targets=targets,
        counts=jnp.zeros_like(counts),
        level_offsets=jnp.zeros_like(level_offsets),
        target_levels=target_levels,
    )


def _prepare_solidfmm_downward_interaction_inputs(
    *,
    tree: Tree,
    upward: TreeUpwardData,
    theta: float,
    mac_type: MACType,
    interactions: Optional[NodeInteractionList],
    far_pairs_coo: Optional[_FarPairCOO],
    traversal_config: Optional[DualTreeTraversalConfig],
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]],
    dehnen_radius_scale: float,
) -> _SolidFMMDownwardInteractionInputs:
    """Resolve interaction storage and far-pair arrays for downward prep."""

    resolved_interactions = interactions
    if resolved_interactions is None and far_pairs_coo is None:
        resolved_interactions = build_well_separated_interactions(
            tree,
            upward.geometry,
            theta=theta,
            mac_type=mac_type,
            dehnen_radius_scale=dehnen_radius_scale,
            traversal_config=traversal_config,
            retry_logger=retry_logger,
        )
    if resolved_interactions is None:
        resolved_interactions = _empty_interaction_storage_for_tree(tree)

    if far_pairs_coo is not None:
        src = jnp.asarray(far_pairs_coo.sources, dtype=INDEX_DTYPE)
        tgt = jnp.asarray(far_pairs_coo.targets, dtype=INDEX_DTYPE)
        active_pair_count = (
            jnp.asarray(far_pairs_coo.active_count, dtype=INDEX_DTYPE)
            if far_pairs_coo.active_count is not None
            else jnp.asarray(src.shape[0], dtype=INDEX_DTYPE)
        )
    else:
        src = jnp.asarray(resolved_interactions.sources, dtype=INDEX_DTYPE)
        tgt = jnp.asarray(resolved_interactions.targets, dtype=INDEX_DTYPE)
        active_pair_count = jnp.asarray(src.shape[0], dtype=INDEX_DTYPE)
    return _SolidFMMDownwardInteractionInputs(
        interactions=resolved_interactions,
        src=src,
        tgt=tgt,
        pair_count=int(src.shape[0]),
        active_pair_count=active_pair_count,
    )


def _prepare_solidfmm_downward_init(
    *,
    upward: TreeUpwardData,
    initial_locals: Optional[LocalExpansionData],
    basis_mode: str,
) -> _SolidFMMDownwardInit:
    """Resolve centers and local-expansion buffers for downward prep."""

    p = int(upward.multipoles.order)
    centers = jnp.asarray(upward.multipoles.centers)
    total_nodes = int(centers.shape[0])
    coeff_count = sh_size(p)
    basis_mode_norm = str(basis_mode).strip().lower()
    if basis_mode_norm not in ("complex", "real"):
        raise ValueError("basis_mode must be 'complex' or 'real'")
    dtype = (
        complex_dtype_for_real(centers.dtype)
        if basis_mode_norm == "complex"
        else centers.dtype
    )
    if initial_locals is not None:
        locals_coeffs = jnp.asarray(initial_locals.coefficients)
        if locals_coeffs.shape != (total_nodes, coeff_count):
            raise ValueError("initial_locals must match solidfmm layout")
    else:
        locals_coeffs = jnp.zeros((total_nodes, coeff_count), dtype=dtype)
    return _SolidFMMDownwardInit(
        centers=centers,
        locals_coeffs=locals_coeffs,
        total_nodes=total_nodes,
        coeff_count=coeff_count,
        dtype=dtype,
    )


def _prepare_solidfmm_downward_multipole_inputs(
    *,
    upward: TreeUpwardData,
    dtype: Any,
    basis_mode: str,
    complex_rotation: str,
) -> _SolidFMMDownwardMultipoleInputs:
    """Resolve multipole coefficient payloads for downward accumulation."""

    p = int(upward.multipoles.order)
    basis_mode_norm = str(basis_mode).strip().lower()
    rotation_mode = str(complex_rotation).strip().lower()
    # The upward sweep always produces packed COMPLEX solidfmm multipoles.
    packed_raw = jnp.asarray(upward.multipoles.packed)
    source_motion_raw = (
        jnp.asarray(upward.multipoles.source_motion_packed)
        if upward.multipoles.source_motion_packed is not None
        else None
    )
    if basis_mode_norm == "complex":
        if rotation_mode != "solidfmm":
            raise ValueError("complex_rotation must be 'solidfmm'")
        multip_packed = packed_raw.astype(dtype)
        source_motion_multip_packed = (
            source_motion_raw.astype(dtype) if source_motion_raw is not None else None
        )
        multip_packed_kernel = multip_packed
    else:
        # Real (Dehnen no-sqrt2) basis: convert the FULL complex multipoles with
        # the Dehnen Q transform (consistent with the real M2L/L2L/L2P operators).
        # NOTE: do NOT use jaccpot.basis.real_sh.complex_to_real_coeffs here --
        # that is the unitary sqrt(2) tesseral basis and is incompatible with the
        # Dehnen operators (it silently caps far-field accuracy). Also must NOT
        # cast the complex packed to a real dtype first (that would drop the
        # imaginary part / sin channels).
        multip_packed = packed_raw
        multip_packed_kernel = complex_to_dehnen_real_coeffs(
            packed_raw, order=p
        ).astype(dtype)
        source_motion_multip_packed = (
            complex_to_dehnen_real_coeffs(source_motion_raw, order=p).astype(dtype)
            if source_motion_raw is not None
            else None
        )
    return _SolidFMMDownwardMultipoleInputs(
        multip_packed=multip_packed,
        source_motion_multip_packed=source_motion_multip_packed,
        multip_packed_kernel=multip_packed_kernel,
        rotation_mode=rotation_mode,
    )


def _prepare_solidfmm_downward_child_inputs(
    tree: Tree,
) -> _SolidFMMDownwardChildInputs:
    """Resolve child-index arrays for L2L propagation."""

    num_internal_nodes = int(jnp.asarray(tree.left_child).shape[0])
    if num_internal_nodes <= 0:
        return _SolidFMMDownwardChildInputs(
            num_internal_nodes=0,
            left_child=None,
            right_child=None,
        )
    return _SolidFMMDownwardChildInputs(
        num_internal_nodes=num_internal_nodes,
        left_child=jnp.asarray(tree.left_child[:num_internal_nodes], dtype=INDEX_DTYPE),
        right_child=jnp.asarray(
            tree.right_child[:num_internal_nodes], dtype=INDEX_DTYPE
        ),
    )


def _solidfmm_downward_accumulate_from_multipoles(
    initial_locals_coeffs: Array,
    multipoles_coeffs: Array,
    *,
    tree: Tree,
    upward: TreeUpwardData,
    interactions: NodeInteractionList,
    centers: Array,
    src: Array,
    tgt: Array,
    pair_count: int,
    active_pair_count: Array,
    order: int,
    rotation_mode: str,
    total_nodes: int,
    chunk_size: int,
    grouped_interactions: bool,
    grouped_buffers: Optional[GroupedInteractionBuffers],
    grouped_segment_starts: Optional[Array],
    grouped_segment_lengths: Optional[Array],
    grouped_segment_class_ids: Optional[Array],
    grouped_segment_sort_permutation: Optional[Array],
    grouped_segment_group_ids: Optional[Array],
    grouped_segment_unique_targets: Optional[Array],
    farfield_mode: str,
    basis_mode: str = "complex",
    m2l_impl: str = "rot_scale",
) -> Array:
    """Run one solidfmm M2L accumulation pass plus symmetry enforcement.

    Both the complex and real (Dehnen no-sqrt2) bases share the grouped /
    class-major / flat dispatch; ``basis_mode`` selects the cached rotation
    blocks and translation kernel (see :func:`_m2l_cached_kernel_dispatch`). The
    non-grouped path uses the dedicated real flat kernels for ``basis_mode ==
    "real"``. Real coefficients carry no conjugate symmetry, so the complex
    symmetry-enforcement step is skipped for the real basis.
    """

    real_basis = str(basis_mode).strip().lower() == "real"

    if grouped_interactions:
        grouped = (
            grouped_buffers
            if grouped_buffers is not None
            else build_grouped_interactions(tree, upward.geometry, interactions)
        )
        mode = str(farfield_mode).strip().lower()
        if mode not in ("pair_grouped", "class_major"):
            raise ValueError("farfield_mode must be 'pair_grouped' or 'class_major'")
        if mode == "class_major":
            locals_updated = _accumulate_solidfmm_m2l_grouped_class_major(
                initial_locals_coeffs,
                multipoles_coeffs,
                centers,
                grouped,
                grouped_segment_starts=grouped_segment_starts,
                grouped_segment_lengths=grouped_segment_lengths,
                grouped_segment_class_ids=grouped_segment_class_ids,
                grouped_segment_sort_permutation=grouped_segment_sort_permutation,
                grouped_segment_group_ids=grouped_segment_group_ids,
                grouped_segment_unique_targets=grouped_segment_unique_targets,
                order=order,
                rotation=rotation_mode,
                total_nodes=total_nodes,
                chunk_size=chunk_size,
                basis_mode=basis_mode,
            )
        else:
            locals_updated = _accumulate_solidfmm_m2l_grouped(
                initial_locals_coeffs,
                multipoles_coeffs,
                centers,
                grouped,
                order=order,
                rotation=rotation_mode,
                total_nodes=total_nodes,
                chunk_size=chunk_size,
                basis_mode=basis_mode,
            )
    elif real_basis:
        if pair_count <= chunk_size:
            locals_updated = _accumulate_real_m2l_fullbatch(
                initial_locals_coeffs,
                multipoles_coeffs,
                centers,
                src,
                tgt,
                active_pair_count,
                order=order,
                m2l_impl=m2l_impl,
                total_nodes=total_nodes,
            )
        else:
            locals_updated = _accumulate_real_m2l_chunked_scan(
                initial_locals_coeffs,
                multipoles_coeffs,
                centers,
                src,
                tgt,
                active_pair_count,
                order=order,
                m2l_impl=m2l_impl,
                total_nodes=total_nodes,
                chunk_size=chunk_size,
            )
    else:
        if pair_count <= chunk_size:
            locals_updated = _accumulate_solidfmm_m2l_fullbatch(
                initial_locals_coeffs,
                multipoles_coeffs,
                centers,
                src,
                tgt,
                active_pair_count,
                order=order,
                rotation=rotation_mode,
                total_nodes=total_nodes,
            )
        else:
            locals_updated = _accumulate_solidfmm_m2l_chunked_scan(
                initial_locals_coeffs,
                multipoles_coeffs,
                centers,
                src,
                tgt,
                active_pair_count,
                order=order,
                rotation=rotation_mode,
                total_nodes=total_nodes,
                chunk_size=chunk_size,
            )

    if real_basis:
        return locals_updated
    return enforce_conjugate_symmetry_batch(locals_updated, order=order)


def _contains_tracer(value: Any) -> bool:
    """Return ``True`` when a pytree contains JAX tracer values."""
    return any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(value)
    )


def _bucket_far_pairs_by_level_split(
    *,
    interactions: NodeInteractionList,
    src_far: Array,
    tgt_far: Array,
    max_order: int,
    min_order: int,
) -> tuple[tuple[int, ...], tuple[tuple[Array, Array], ...]]:
    """Split far pairs into two orders using interaction level offsets.

    Coarser levels use ``max_order`` and deeper levels use ``min_order``.
    """
    min_order_int = int(min_order)
    max_order_int = int(max_order)
    if min_order_int >= max_order_int:
        return (max_order_int,), ((src_far, tgt_far),)

    level_offsets = getattr(interactions, "level_offsets", None)
    if level_offsets is None:
        return (max_order_int,), ((src_far, tgt_far),)

    try:
        offsets_np = np.asarray(jax.device_get(level_offsets), dtype=np.int64)
    except Exception:
        return (max_order_int,), ((src_far, tgt_far),)
    if offsets_np.size <= 2:
        return (max_order_int,), ((src_far, tgt_far),)

    levels = int(offsets_np.size - 1)
    split_level = max(1, levels // 2)
    coarse_end = int(offsets_np[min(split_level, levels)])
    fine_start = coarse_end
    pair_count = int(src_far.shape[0])
    coarse_end = max(0, min(coarse_end, pair_count))
    fine_start = max(0, min(fine_start, pair_count))

    src_hi = jnp.asarray(src_far[:coarse_end], dtype=INDEX_DTYPE)
    tgt_hi = jnp.asarray(tgt_far[:coarse_end], dtype=INDEX_DTYPE)
    src_lo = jnp.asarray(src_far[fine_start:], dtype=INDEX_DTYPE)
    tgt_lo = jnp.asarray(tgt_far[fine_start:], dtype=INDEX_DTYPE)
    return (
        (min_order_int, max_order_int),
        ((src_lo, tgt_lo), (src_hi, tgt_hi)),
    )


@partial(jax.jit, static_argnames=("order",))
def _evaluate_local_cartesian_with_grad_batch(
    coeffs: Array,
    offsets: Array,
    *,
    order: int,
) -> tuple[Array, Array]:
    """Evaluate cartesian local expansions and gradients at batch offsets."""
    leading_shape = coeffs.shape[:-1]
    coeffs_flat = jnp.reshape(coeffs, (-1, coeffs.shape[-1]))
    offsets_flat = jnp.reshape(offsets, (-1, offsets.shape[-1]))

    translated_flat = jax.vmap(
        lambda coeff_row, offset_row: translate_local_expansion(
            coeff_row,
            offset_row,
            order=order,
        )
    )(coeffs_flat, offsets_flat)

    translated = jnp.reshape(
        translated_flat,
        leading_shape + (translated_flat.shape[-1],),
    )

    potentials = translated[..., level_offset(0)]
    if order <= 0:
        gradients = jnp.zeros(leading_shape + (3,), dtype=translated.dtype)
    else:
        first = translated[..., level_offset(1) : level_offset(1) + 3]
        gradients = jnp.stack([first[..., 2], first[..., 1], first[..., 0]], axis=-1)
    return gradients, potentials


class FastMultipoleMethod:
    """
    Fast Multipole Method for gravitational N-body simulations.

    Args:
        theta: Opening angle criterion (typically 0.5-1.0)
        G: Gravitational constant (default: 1.0)
        softening: Softening length to avoid singularities (default: 0.0)
        tree_build_mode:
            Choose between "lbvh" and "fixed_depth" builders.
        tree_type:
            Yggdrax tree family selector (e.g. ``"radix"`` or ``"kdtree"``).
        target_leaf_particles:
            Desired particle count per leaf for fixed-depth trees.
        refine_local:
            Enable host-side refinement of elongated leaves.
        max_refine_levels:
            Maximum refinement depth for the local refinement pass.
        aspect_threshold:
            Aspect ratio threshold that triggers extra splits.
    """

    def __init__(
        self: "FastMultipoleMethod",
        theta: float = 0.5,
        G: float = 1.0,
        softening: float = 1e-12,
        working_dtype: Optional[DTypeLike] = None,
        *,
        expansion_basis: ExpansionBasis = "cartesian",
        basis_impl: Optional[Any] = None,
        m2l_impl: Optional[str] = None,
        adaptive_order: bool = False,
        p_gears: Optional[tuple[int, ...]] = None,
        use_pallas: Optional[bool] = None,
        reuse_topology: bool = False,
        rebuild_every: int = 1,
        mac_force_scale_mode: str = "prev",
        adaptive_error_model: str = "tail_proxy",
        adaptive_eps: Optional[float] = None,
        dehnen_geometry_mode: str = "tree",
        mac_type: MACType = "bh",
        complex_rotation: str = "solidfmm",  # "cached",
        dehnen_radius_scale: float = 1.0,
        m2l_chunk_size: Optional[int] = None,
        l2l_chunk_size: Optional[int] = None,
        max_pair_queue: Optional[int] = None,
        pair_process_block: Optional[int] = None,
        traversal_config: Optional[DualTreeTraversalConfig] = None,
        tree_build_mode: Optional[str] = None,
        tree_type: str = "radix",
        target_leaf_particles: Optional[int] = None,
        refine_local: Optional[bool] = None,
        max_refine_levels: Optional[int] = None,
        aspect_threshold: Optional[float] = None,
        interaction_retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
        use_dense_interactions: Optional[bool] = None,
        grouped_interactions: Optional[bool] = None,
        farfield_mode: FarFieldMode = "auto",
        streamed_far_pairs: Optional[bool] = None,
        mixed_order_farfield: bool = False,
        mixed_order_min_order: Optional[int] = None,
        nearfield_mode: NearFieldMode = "auto",
        runtime_path: Literal["auto", "legacy", "large_n"] = "auto",
        execution_backend: FMMExecutionBackend = "auto",
        nearfield_edge_chunk_size: int = 256,
        precompute_nearfield_scatter_schedules: bool = True,
        memory_objective: MemoryObjective = "balanced",
        memory_budget_bytes: Optional[int] = None,
        enable_interaction_cache: bool = True,
        retain_traversal_result: bool = True,
        retain_interactions: bool = True,
        prepare_stage_memory_split_enabled: Optional[bool] = None,
        autotune_m2l_chunk: bool = False,
        precompute_grouped_class_segments: Optional[bool] = None,
        grouped_schedule_budget_bytes: Optional[int] = None,
        nearfield_schedule_item_cap: Optional[int] = None,
        upward_leaf_batch_size: Optional[int] = None,
        host_refine_mode: str = "auto",
        fail_fast: bool = False,
        preset: Optional[Union[str, FMMPreset]] = None,
        fixed_order: Optional[int] = None,
        fixed_max_leaf_size: Optional[int] = None,
    ):
        basis_norm = str(expansion_basis).strip().lower()
        if basis_norm == "complex":
            basis_norm = "solidfmm"
        if basis_norm not in ("cartesian", "solidfmm"):
            raise ValueError(
                "expansion_basis must be 'cartesian', 'solidfmm', or 'complex'",
            )
        self.expansion_basis = basis_norm  # type: ignore[assignment]
        self.basis_impl = basis_impl
        self.m2l_impl = None if m2l_impl is None else str(m2l_impl).strip().lower()
        if self.m2l_impl is None and self._solidfmm_basis_mode() == "real":
            self.m2l_impl = "rot_scale"
        self.adaptive_order = bool(adaptive_order)
        self.p_gears = tuple(int(v) for v in (p_gears or ()))
        # Default the Pallas near-field ON wherever it can run (Ampere sm_80+),
        # and fall back to the pure-JAX near-field ONLY on hardware that cannot
        # run Pallas (e.g. RTX 2080 / sm_75) or CPU. Leaving it off by default
        # silently ran the ~10x-slower launch-bound jnp near-field on capable
        # GPUs; the non-Pallas path is retained solely as the sm_75/CPU lane.
        # An explicit use_pallas=True/False still overrides this resolution.
        if use_pallas is None:
            try:
                from jaccpot.pallas.nearfield_fused_leaf import (
                    pallas_nearfield_fused_supported,
                )

                resolved_use_pallas = bool(pallas_nearfield_fused_supported())
            except Exception:
                resolved_use_pallas = False
        else:
            resolved_use_pallas = bool(use_pallas)
        self.use_pallas = resolved_use_pallas
        self.reuse_topology = bool(reuse_topology)
        if int(rebuild_every) <= 0:
            raise ValueError("rebuild_every must be positive")
        self.rebuild_every = int(rebuild_every)
        self._recent_far_pairs_by_gear_counts: tuple[int, ...] = tuple()
        self._recent_dual_node_count: int = 0
        self._recent_dual_leaf_count: int = 0
        self._recent_dual_neighbor_count: int = 0
        self._recent_dual_far_pair_count: int = 0
        self._recent_dual_m2l_chunk_size: int = 0
        self._static_radix_tree_leaf_count: int = 0
        self._static_radix_tree_node_count: int = 0
        # Concrete tree depth (unpadded), stashed at build so the traced refresh
        # can pass it as a static arg to the upward M2M level loop. Radix trees
        # pad level_offsets to full Morton depth; using the padded shape makes the
        # M2M loop iterate many empty levels. See _resolve_upward_num_levels.
        self._static_upward_num_levels: Optional[int] = None
        self._static_radix_far_pair_count: int = 0
        self._static_radix_m2l_chunk_count: int = 0
        self._static_radix_l2l_edge_count: int = 0
        force_scale_mode_norm = str(mac_force_scale_mode).strip().lower()
        if force_scale_mode_norm not in ("prev", "prepass", "paper"):
            raise ValueError(
                "mac_force_scale_mode must be 'prev', 'prepass', or 'paper'"
            )
        self.mac_force_scale_mode = force_scale_mode_norm
        adaptive_error_model_norm = str(adaptive_error_model).strip().lower()
        if adaptive_error_model_norm not in (
            "tail_proxy",
            "dehnen_degree",
            "dehnen_paper",
        ):
            raise ValueError(
                "adaptive_error_model must be 'tail_proxy', 'dehnen_degree', or 'dehnen_paper'"
            )
        self.adaptive_error_model = adaptive_error_model_norm
        dehnen_geometry_mode_norm = str(dehnen_geometry_mode).strip().lower()
        if dehnen_geometry_mode_norm not in ("exact", "tree", "tree_approx", "runtime"):
            raise ValueError(
                "dehnen_geometry_mode must be 'exact', 'tree', 'tree_approx', or 'runtime'"
            )
        self.dehnen_geometry_mode = dehnen_geometry_mode_norm
        self.adaptive_eps = None if adaptive_eps is None else float(adaptive_eps)
        if self.adaptive_eps is not None and self.adaptive_eps <= 0.0:
            raise ValueError("adaptive_eps must be > 0 when provided")
        self._last_force_scale_nodes: Optional[Array] = None
        self._in_force_scale_prepass = False

        rotation_norm = str(complex_rotation).strip().lower()
        if rotation_norm != "solidfmm":
            raise ValueError("complex_rotation must be 'solidfmm'")
        self.complex_rotation = rotation_norm
        farfield_mode_norm = str(farfield_mode).strip().lower()
        if farfield_mode_norm not in ("auto", "pair_grouped", "class_major"):
            raise ValueError(
                "farfield_mode must be 'auto', 'pair_grouped', or 'class_major'"
            )
        self.farfield_mode = farfield_mode_norm
        self._explicit_streamed_far_pairs = streamed_far_pairs is not None
        self.streamed_far_pairs = bool(streamed_far_pairs)
        self.mixed_order_farfield = bool(mixed_order_farfield)
        self.mixed_order_min_order = (
            None if mixed_order_min_order is None else int(mixed_order_min_order)
        )
        if (
            self.mixed_order_min_order is not None
            and int(self.mixed_order_min_order) < 0
        ):
            raise ValueError("mixed_order_min_order must be >= 0")
        nearfield_mode_norm = str(nearfield_mode).strip().lower()
        if nearfield_mode_norm not in ("auto", "baseline", "bucketed"):
            raise ValueError("nearfield_mode must be 'auto', 'baseline', or 'bucketed'")
        runtime_path_norm = str(runtime_path).strip().lower()
        if runtime_path_norm not in ("auto", "legacy", "large_n"):
            raise ValueError("runtime_path must be 'auto', 'legacy', or 'large_n'")
        if runtime_path_norm == "legacy":
            warnings.warn(
                "runtime_path='legacy' is deprecated and will be removed; "
                "use runtime_path='large_n' or runtime_path='auto'.",
                FutureWarning,
                stacklevel=2,
            )
        execution_backend_norm = str(execution_backend).strip().lower()
        if execution_backend_norm not in ("auto", "radix", "octree"):
            raise ValueError("execution_backend must be 'auto', 'radix', or 'octree'")
        if int(nearfield_edge_chunk_size) <= 0:
            raise ValueError("nearfield_edge_chunk_size must be positive")
        self.nearfield_mode = nearfield_mode_norm
        self._explicit_nearfield_mode = nearfield_mode_norm != "auto"
        self.runtime_path = runtime_path_norm
        self.execution_backend = execution_backend_norm
        self.nearfield_edge_chunk_size = int(nearfield_edge_chunk_size)
        self.precompute_nearfield_scatter_schedules = bool(
            precompute_nearfield_scatter_schedules
        )
        objective_norm = str(memory_objective).strip().lower()
        if objective_norm not in ("balanced", "throughput", "minimum_memory"):
            raise ValueError(
                "memory_objective must be 'balanced', 'throughput', or 'minimum_memory'"
            )
        self.memory_objective: MemoryObjective = objective_norm  # type: ignore[assignment]
        self._explicit_memory_objective = objective_norm != "balanced"
        self.memory_budget_bytes = (
            None if memory_budget_bytes is None else int(memory_budget_bytes)
        )
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be > 0 when provided")
        self.enable_interaction_cache = bool(enable_interaction_cache)
        self.retain_traversal_result = bool(retain_traversal_result)
        self.retain_interactions = bool(retain_interactions)
        self.prepare_stage_memory_split_enabled = (
            None
            if prepare_stage_memory_split_enabled is None
            else bool(prepare_stage_memory_split_enabled)
        )
        self.fail_fast = bool(fail_fast)
        self.autotune_m2l_chunk = bool(autotune_m2l_chunk) and not self.fail_fast
        self.precompute_grouped_class_segments = (
            None
            if precompute_grouped_class_segments is None
            else bool(precompute_grouped_class_segments)
        )
        self.grouped_schedule_budget_bytes = (
            _GROUPED_SCHEDULE_BUDGET_DEFAULT
            if grouped_schedule_budget_bytes is None
            else int(grouped_schedule_budget_bytes)
        )
        if self.grouped_schedule_budget_bytes <= 0:
            raise ValueError("grouped_schedule_budget_bytes must be positive")
        self.nearfield_schedule_item_cap = (
            None
            if nearfield_schedule_item_cap is None
            else int(nearfield_schedule_item_cap)
        )
        if (
            self.nearfield_schedule_item_cap is not None
            and self.nearfield_schedule_item_cap <= 0
        ):
            raise ValueError("nearfield_schedule_item_cap must be > 0 when provided")
        self.upward_leaf_batch_size = (
            None if upward_leaf_batch_size is None else int(upward_leaf_batch_size)
        )
        if self.upward_leaf_batch_size is not None and self.upward_leaf_batch_size <= 0:
            raise ValueError("upward_leaf_batch_size must be > 0 when provided")
        dehnen_scale_val = float(dehnen_radius_scale)
        if dehnen_scale_val <= 0.0:
            raise ValueError("dehnen_radius_scale must be > 0")
        self.dehnen_radius_scale = dehnen_scale_val

        refine_mode_norm = str(host_refine_mode).strip().lower()
        if refine_mode_norm not in ("auto", "on", "off"):
            raise ValueError("host_refine_mode must be 'auto', 'on', or 'off'")
        if self.fail_fast:
            refine_mode_norm = "off"
        self.host_refine_mode = refine_mode_norm
        tree_type_norm = str(tree_type).strip().lower()
        supported_tree_types = set(available_tree_types())
        if tree_type_norm not in supported_tree_types:
            supported_txt = ", ".join(sorted(supported_tree_types))
            raise ValueError(
                f"tree_type must be one of ({supported_txt}), got '{tree_type}'"
            )
        self.tree_type: TreeType = tree_type_norm  # type: ignore[assignment]

        preset_config = get_preset_config(preset) if preset is not None else None

        resolved = _resolve_fmm_config(
            theta=theta,
            G=G,
            softening=softening,
            working_dtype=working_dtype,
            tree_build_mode=tree_build_mode,
            target_leaf_particles=target_leaf_particles,
            refine_local=refine_local,
            max_refine_levels=max_refine_levels,
            aspect_threshold=aspect_threshold,
            m2l_chunk_size=m2l_chunk_size,
            l2l_chunk_size=l2l_chunk_size,
            max_pair_queue=max_pair_queue,
            pair_process_block=pair_process_block,
            traversal_config=traversal_config,
            use_dense_interactions=use_dense_interactions,
            preset_config=preset_config,
        )

        self.config = resolved
        self.preset = resolved.preset
        self.theta = resolved.theta
        self.mac_type = mac_type
        if self._uses_dehnen_error_policy():
            if self.adaptive_error_model == "tail_proxy":
                self.adaptive_error_model = "dehnen_paper"
            if self.mac_force_scale_mode == "prev":
                self.mac_force_scale_mode = "paper"
        self.G = resolved.G
        self.softening = resolved.softening
        self.working_dtype = resolved.working_dtype
        self._preset_config = preset_config
        self._jit_tree_default = resolved.traversal.jit_tree
        self._jit_traversal_default = resolved.traversal.jit_traversal
        self.m2l_chunk_size = resolved.traversal.m2l_chunk_size
        self.l2l_chunk_size = resolved.traversal.l2l_chunk_size
        self.max_pair_queue = resolved.traversal.max_pair_queue
        self.pair_process_block = resolved.traversal.pair_process_block
        self.traversal_config = resolved.traversal.traversal_config
        self.tree_build_mode = resolved.tree.mode
        self.target_leaf_particles = resolved.tree.target_leaf_particles
        self.refine_local = resolved.tree.refine_local
        self.max_refine_levels = resolved.tree.max_refine_levels
        self.aspect_threshold = resolved.tree.aspect_threshold
        self.interaction_retry_logger = interaction_retry_logger
        self.use_dense_interactions = resolved.traversal.use_dense_interactions
        self._tree_workspace: Optional[object] = None
        self._locals_template: Optional[LocalExpansionData] = None
        self._interaction_cache: Optional[_InteractionCacheEntry] = None
        self._interaction_cache_hits: int = 0
        self._interaction_cache_misses: int = 0
        self._prepared_state_cache_key: Optional[tuple[Any, ...]] = None
        self._prepared_state_cache_value: Optional[PreparedStateLike] = None
        self._prepared_state_cache_positions: Optional[Array] = None
        self._prepared_state_cache_masses: Optional[Array] = None
        self._topology_reuse_entry: Optional[_TopologyReuseEntry] = None
        self._geometry_reuse_entry: Optional[_GeometryReuseEntry] = None
        self._recent_topology_reused: bool = False
        self._recent_retry_events: Tuple[DualTreeRetryEvent, ...] = tuple()
        self._compiled_profile_fingerprint_last: Optional[str] = None
        self._compiled_profile_transitions: int = 0
        self._large_n_eval_leaf_nodes_shape: tuple[int, ...] = ()
        self._large_n_eval_local_coefficients_shape: tuple[int, ...] = ()
        self._large_n_eval_local_centers_shape: tuple[int, ...] = ()
        self._large_n_eval_active_leaf_count: int = 0
        self._large_n_eval_max_leaf_size: int = 0
        self._large_n_eval_leaf_particle_slots: int = 0
        self._large_n_radix_payload_present: bool = False
        self._large_n_radix_payload_source_particle_shape: tuple[int, ...] = ()
        self._large_n_radix_payload_source_particle_slots: int = 0
        self._large_n_radix_payload_source_leaf_shape: tuple[int, ...] = ()
        self._large_n_radix_payload_source_leaf_slots: int = 0
        self._large_n_target_block_source_leaf_padded_shape: tuple[int, ...] = ()
        self._compiled_profile_refresh_calls: int = 0
        self._compiled_profile_refresh_reuse_tier_full: int = 0
        self._compiled_profile_refresh_reuse_tier_topology: int = 0
        self._compiled_profile_refresh_reuse_tier_overflow: int = 0
        self._large_n_same_topology_refresh_attempts: int = 0
        self._large_n_same_topology_refresh_hits: int = 0
        self._large_n_same_topology_refresh_misses: int = 0
        self._large_n_same_topology_refresh_miss_no_key: int = 0
        self._large_n_same_topology_refresh_miss_topology: int = 0
        self._large_n_same_topology_refresh_miss_neighbor: int = 0
        self._large_n_same_topology_refresh_miss_traced: int = 0
        self._large_n_same_topology_refresh_last_error: str = ""
        self._static_radix_refresh_hits: int = 0
        self._static_radix_refresh_misses: int = 0
        self._static_radix_profile_overflows: int = 0
        self._static_radix_compact_pair_reuse_hits: int = 0
        self._static_radix_compact_pair_reuse_misses: int = 0
        self._compiled_profile_multipoles_only_calls: int = 0
        self._compiled_profile_topology_rebuild_calls: int = 0
        self._large_n_overflow_profile_cap: int = 0
        self._large_n_overflow_profile_reprofiles: int = 0
        self._large_n_neighbor_edges_profile_cap: int = 0
        self._large_n_neighbor_edges_profile_reprofiles: int = 0
        self._refresh_timing_total_seconds: float = 0.0
        self._refresh_timing_input_seconds: float = 0.0
        self._refresh_timing_tree_upward_seconds: float = 0.0
        self._refresh_timing_tree_build_seconds: float = 0.0
        self._refresh_timing_upward_compute_seconds: float = 0.0
        self._refresh_timing_upward_geometry_seconds: float = 0.0
        self._refresh_timing_upward_mass_moments_seconds: float = 0.0
        self._refresh_timing_upward_p2m_seconds: float = 0.0
        self._refresh_timing_upward_m2m_seconds: float = 0.0
        self._refresh_timing_upward_source_motion_seconds: float = 0.0
        self._refresh_timing_dual_downward_seconds: float = 0.0
        self._refresh_timing_nearfield_seconds: float = 0.0
        self._refresh_timing_profile_accounting_seconds: float = 0.0
        self._refresh_timing_compile_or_sync_suspect_seconds: float = 0.0
        self._refresh_timing_dual_setup_seconds: float = 0.0
        self._refresh_timing_dual_artifact_build_seconds: float = 0.0
        self._refresh_timing_dual_split_shared_far_near_seconds: float = 0.0
        self._refresh_timing_dual_split_shared_count_seconds: float = 0.0
        self._refresh_timing_dual_split_shared_combined_fill_seconds: float = 0.0
        self._refresh_timing_dual_split_shared_far_fill_seconds: float = 0.0
        self._refresh_timing_dual_split_shared_near_fill_seconds: float = 0.0
        self._refresh_timing_dual_split_far_pairs_seconds: float = 0.0
        self._refresh_timing_dual_split_leaf_neighbors_seconds: float = 0.0
        self._refresh_timing_dual_split_combined_seconds: float = 0.0
        self._refresh_timing_dual_raw_combined_seconds: float = 0.0
        self._refresh_timing_dual_split_dense_buffers_seconds: float = 0.0
        self._refresh_timing_dual_far_pair_plan_seconds: float = 0.0
        self._refresh_timing_dual_m2l_autotune_seconds: float = 0.0
        self._refresh_timing_dual_select_interactions_seconds: float = 0.0
        self._refresh_timing_dual_downward_compute_seconds: float = 0.0
        self._refresh_timing_dual_m2l_compute_seconds: float = 0.0
        self._refresh_timing_dual_l2l_compute_seconds: float = 0.0
        self._refresh_timing_dual_final_symmetry_seconds: float = 0.0
        self._refresh_timing_dual_source_motion_seconds: float = 0.0
        self._refresh_timing_dual_finalize_seconds: float = 0.0
        self._refresh_timing_dual_residual_seconds: float = 0.0
        self._refresh_timing_nearfield_leaf_groups_seconds: float = 0.0
        self._refresh_timing_nearfield_precompute_seconds: float = 0.0
        self._refresh_timing_nearfield_target_blocks_seconds: float = 0.0
        self._refresh_timing_nearfield_block_sort_seconds: float = 0.0
        self._refresh_timing_nearfield_speed_layout_seconds: float = 0.0
        self._refresh_timing_nearfield_overflow_profile_seconds: float = 0.0
        self._refresh_timing_nearfield_radix_payload_seconds: float = 0.0
        self._refresh_timing_nearfield_neighbor_padding_seconds: float = 0.0
        self._refresh_timing_nearfield_state_pack_seconds: float = 0.0
        self._refresh_timing_nearfield_residual_seconds: float = 0.0
        self._refresh_timing_calls: int = 0
        self._refresh_timing_active: bool = False
        self._refresh_timing_enabled: bool = str(
            os.environ.get("JACCPOT_REFRESH_TIMING_ENABLE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._refresh_dual_planner_mode: str = (
            str(os.environ.get("JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_MODE", "auto"))
            .strip()
            .lower()
        )
        self._refresh_dual_planner_mode_on: bool = (
            self._refresh_dual_planner_mode == "on"
        )
        self._refresh_dual_planner_mode_auto: bool = (
            self._refresh_dual_planner_mode == "auto"
        )
        self._strict_gpu_mode: str = (
            str(os.environ.get("JACCPOT_STATIC_STRICT_GPU_MODE", "auto"))
            .strip()
            .lower()
        )
        self._strict_gpu_mode_on: bool = self._strict_gpu_mode == "on"
        self._strict_gpu_mode_auto: bool = self._strict_gpu_mode == "auto"
        self._strict_cap_record_enabled: bool = str(
            os.environ.get("JACCPOT_STATIC_STRICT_CAP_RECORD", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_cap_require_exact_profile_match: bool = str(
            os.environ.get("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        split_build_env_raw = os.environ.get(
            "JACCPOT_PREPARE_STAGE_MEMORY_SPLIT_ENABLED"
        )
        self._prepare_stage_memory_split_env_override: Optional[bool] = (
            None
            if split_build_env_raw is None
            else str(split_build_env_raw).strip().lower() in {"1", "true", "yes", "on"}
        )
        self._planner_steady_timing_bypass_enabled: bool = str(
            os.environ.get(
                "JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_STEADY_NO_SUBSTAGE_TIMING",
                "1",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_shared_env_applied: bool = False
        self._refresh_dual_planner_cache: dict[str, _RefreshDualPlannerHint] = {}
        self._refresh_dual_planner_cache_hits: int = 0
        self._refresh_dual_planner_cache_misses: int = 0
        self._refresh_dual_planner_compile_count: int = 0
        self._refresh_dual_planner_execute_count: int = 0
        self._refresh_dual_planner_steady_timing_bypass_count: int = 0
        self._refresh_dual_planner_compiled_route_count: int = 0
        self._refresh_strict_mode_active_count: int = 0
        self._strict_runner_compile_count: int = 0
        self._strict_runner_execute_count: int = 0
        self._strict_runner_profile_key_hits: int = 0
        self._strict_runner_profile_key_misses: int = 0
        self._strict_runner_fail_fast_reject_count: int = 0
        self._strict_runner_seen_profile_keys: set[str] = set()
        self._strict_v2_compile_count: int = 0
        self._strict_v2_execute_count: int = 0
        self._strict_v2_profile_key_hits: int = 0
        self._strict_v2_profile_key_misses: int = 0
        self._strict_v2_fail_fast_reject_count: int = 0
        self._strict_v2_seen_profile_keys: set[str] = set()
        self._strict_fused_mode_raw: str = (
            str(os.environ.get("JACCPOT_STATIC_STRICT_FUSED_MODE", "off"))
            .strip()
            .lower()
        )
        self._strict_fused_mode_enabled: bool = self._strict_fused_mode_raw in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._strict_fused_profile_set_raw: str = str(
            os.environ.get("JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET", "")
        ).strip()
        self._strict_fused_disable_hot_timing: bool = str(
            os.environ.get("JACCPOT_STATIC_STRICT_FUSED_DISABLE_HOT_TIMING", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_fused_disable_rematerialize: bool = str(
            os.environ.get("JACCPOT_STATIC_STRICT_FUSED_DISABLE_REMATERIALIZE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_fused_disallow_host_segment_fallback: bool = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK",
                "0",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Default ON: the device-only fused hot path enables the streamed
        # fast-lane (_prepare_state_dual_and_downward_strict_streamed_fast),
        # which is ~10x faster than the host-routed path for the strict fused
        # static-radix lane (200k particles: ~1224 -> ~119 ms/step on an A100)
        # with bit-identical energy / angular-momentum conservation
        # (max|dE/E0| = 8.415e-04 either way, verified over 400 steps). Set the
        # env var to "0" to opt back into the slower host-routed path, which is
        # retained only as a fallback.
        self._strict_fused_device_only: bool = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY",
                "1",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_fused_compiled_segment_loop: bool = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_COMPILED_SEGMENT_LOOP",
                "1",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_fused_jit_refresh_eval: bool = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_JIT_REFRESH_EVAL",
                "1",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._large_n_eval_diag_mode: str = (
            str(os.environ.get("JACCPOT_LARGE_N_EVAL_DIAG_MODE", "full"))
            .strip()
            .lower()
        )
        if self._large_n_eval_diag_mode not in {
            "full",
            "near_only",
            "far_only",
            "local_only",
            "near_zero",
            "far_zero",
            "permutation_only",
            "zero",
        }:
            self._large_n_eval_diag_mode = "full"
        self._large_n_nearfield_diag_mode: str = (
            str(os.environ.get("JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE", "full"))
            .strip()
            .lower()
        )
        if self._large_n_nearfield_diag_mode not in {
            "full",
            "self_only",
            "pairs_only",
            "overflow_only",
            "zero",
        }:
            self._large_n_nearfield_diag_mode = "full"
        self._strict_refresh_diag_mode: str = _normalize_strict_refresh_diag_mode(
            os.environ.get("JACCPOT_STRICT_REFRESH_DIAG_MODE", "full")
        )
        self._strict_refresh_detail_diag_mode: str = (
            _normalize_strict_refresh_detail_diag_mode(
                os.environ.get("JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE", "full")
            )
        )
        (
            self._strict_refresh_diag_tree_active,
            self._strict_refresh_diag_upward_active,
            self._strict_refresh_diag_downward_active,
            self._strict_refresh_diag_eval_active,
        ) = _strict_refresh_diag_stage_flags(self._strict_refresh_diag_mode)
        self._strict_fused_mode_active: bool = False
        self._strict_fused_compile_count: int = 0
        self._strict_fused_execute_count: int = 0
        self._strict_fused_profile_key_hits: int = 0
        self._strict_fused_profile_key_misses: int = 0
        self._strict_fused_fallback_count: int = 0
        self._strict_fused_last_fallback_reason: str = ""
        self._strict_fused_device_refresh_route_count: int = 0
        self._strict_fused_planner_bypassed_count: int = 0
        self._strict_velocity_verlet_acceleration_carry_active: bool = False
        self._strict_self_force_bootstrap_evaluations: int = 0
        self._strict_self_force_endpoint_evaluations: int = 0
        self._strict_external_bootstrap_evaluations: int = 0
        self._strict_external_endpoint_evaluations: int = 0
        self._strict_static_target_block_capacity_ok: bool = True
        self._large_n_radix_fast_occupancy_sort: bool = str(
            os.environ.get("JACCPOT_LARGE_N_RADIX_FAST_OCCUPANCY_SORT", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._large_n_radix_fast_skip_empty_tiles: bool = str(
            os.environ.get("JACCPOT_LARGE_N_RADIX_FAST_SKIP_EMPTY_TILES", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_fused_seen_profile_keys: set[str] = set()
        self._strict_fused_fastlane_diag_enabled: bool = str(
            os.environ.get("JACCPOT_STATIC_STRICT_FUSED_FASTLANE_DIAG", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._strict_fused_fastlane_attempts: int = 0
        self._strict_fused_fastlane_hits: int = 0
        self._strict_fused_fastlane_misses: int = 0
        self._strict_fused_fastlane_last_blockers: tuple[str, ...] = tuple()
        self._strict_fused_fastlane_block_counts: dict[str, int] = {}
        self._strict_fused_jit_function_cache: dict[
            tuple[Any, ...], tuple[Any, ...]
        ] = {}
        self._strict_profiled_max_pair_queue: int = 0
        self._strict_profiled_pair_process_block: int = 0
        self._strict_profiled_context_key: str = ""
        self._strict_profile_catalog: dict[str, dict[str, int]] = {}
        self._strict_profile_loaded_once: bool = False
        self.fixed_order = fixed_order
        self.fixed_max_leaf_size = fixed_max_leaf_size
        self._explicit_m2l_chunk_size = m2l_chunk_size is not None
        self._explicit_l2l_chunk_size = l2l_chunk_size is not None
        self._explicit_traversal_config = traversal_config is not None
        self._explicit_max_pair_queue = max_pair_queue is not None
        self._explicit_pair_process_block = pair_process_block is not None
        self._explicit_grouped_interactions = grouped_interactions is not None
        self.grouped_interactions = grouped_interactions
        self._streamed_minimum_memory_gpu_default_split_build: bool = bool(
            self.memory_objective == "minimum_memory"
            and jax.default_backend() == "gpu"
            and self.tree_type == "radix"
            and self.expansion_basis == "solidfmm"
            and bool(self.streamed_far_pairs)
        )
        self._large_n_gpu_production_profile_cached: bool = (
            str(self.preset).strip().lower() == "large_n_gpu"
            and str(self.tree_type).strip().lower() == "radix"
            and str(self.expansion_basis).strip().lower() == "solidfmm"
            and str(self.execution_backend).strip().lower() != "octree"
        )
        self._static_runtime_fixed_sizing: bool = str(
            os.environ.get("JACCPOT_STATIC_RUNTIME_FIXED_SIZING", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._apply_large_n_gpu_production_contract()

    def _is_large_n_gpu_production_profile(self) -> bool:
        """Whether this solver should run the canonical large-N GPU contract."""
        return bool(
            getattr(
                self,
                "_large_n_gpu_production_profile_cached",
                (
                    str(self.preset).strip().lower() == "large_n_gpu"
                    and str(self.tree_type).strip().lower() == "radix"
                    and str(self.expansion_basis).strip().lower() == "solidfmm"
                    and str(self.execution_backend).strip().lower() != "octree"
                ),
            )
        )

    def _apply_large_n_gpu_production_contract(self) -> None:
        """Normalize large-N GPU runtime knobs to the canonical fast memory path."""

        if not self._is_large_n_gpu_production_profile():
            return

        if (
            self._explicit_memory_objective
            and self.memory_objective != "minimum_memory"
        ):
            warnings.warn(
                "large_n_gpu production profile coerces memory_objective to "
                "'minimum_memory' for memory-stable performance.",
                FutureWarning,
                stacklevel=2,
            )
        if (
            self._explicit_nearfield_mode
            and str(self.nearfield_mode).strip().lower() != "bucketed"
        ):
            warnings.warn(
                "large_n_gpu production profile coerces nearfield_mode to "
                "'bucketed' to keep radix fast-lane active.",
                FutureWarning,
                stacklevel=2,
            )
        if bool(self.grouped_interactions):
            warnings.warn(
                "large_n_gpu production profile disables grouped_interactions "
                "to keep streamed pair_grouped execution.",
                FutureWarning,
                stacklevel=2,
            )
        if self._explicit_streamed_far_pairs and not bool(self.streamed_far_pairs):
            warnings.warn(
                "large_n_gpu production profile enables streamed_far_pairs "
                "for low-memory execution.",
                FutureWarning,
                stacklevel=2,
            )

        # Keep large-N production on one stable runtime lane.
        self.runtime_path = "large_n"
        self.memory_objective = "minimum_memory"  # type: ignore[assignment]
        self.streamed_far_pairs = True
        self.grouped_interactions = False
        self._explicit_grouped_interactions = False
        self.farfield_mode = "pair_grouped"

        # Keep near-field on the radix fast-lane compatible bucketed path.
        self.nearfield_mode = "bucketed"
        self.precompute_nearfield_scatter_schedules = False
        self.mixed_order_farfield = False
        self.mixed_order_min_order = None

        # Keep the topology-derived interaction scaffold resident so fixed-shape
        # refreshes can reuse it instead of rebuilding the dual-tree artifacts.
        self.enable_interaction_cache = True
        self.retain_traversal_result = False
        self.retain_interactions = False
        self.precompute_grouped_class_segments = False
        if self.upward_leaf_batch_size is None:
            self.upward_leaf_batch_size = 2048

    def _resolve_execution_backend(self) -> str:
        """Resolve the active FMM execution backend without altering tree choice."""
        if self.execution_backend == "auto":
            return "radix"
        return self.execution_backend

    def _ensure_execution_backend_supported(
        self, *, tree: Optional[Tree] = None
    ) -> str:
        """Validate execution backends that are available for the current tree."""
        backend = self._resolve_execution_backend()
        if backend != "octree":
            return backend

        tree_type = getattr(tree, "tree_type", self.tree_type)
        if str(tree_type).strip().lower() != "octree":
            raise ValueError("execution_backend='octree' requires an octree tree_type")
        if self.expansion_basis != "solidfmm":
            raise NotImplementedError(
                "execution_backend='octree' currently supports basis='solidfmm' only"
            )
        return backend

    @property
    def recent_retry_events(
        self: "FastMultipoleMethod",
    ) -> Tuple[DualTreeRetryEvent, ...]:
        """Return retry telemetry collected during the latest build."""

        return self._recent_retry_events

    @property
    def recent_topology_reused(self: "FastMultipoleMethod") -> bool:
        """Whether the most recent prepare/evaluate path reused cached topology."""

        return bool(self._recent_topology_reused)

    def clear_prepared_state_cache(self: "FastMultipoleMethod") -> None:
        """Clear cached prepared-state payloads used by reuse mode."""

        self._prepared_state_cache_key = None
        self._prepared_state_cache_value = None
        self._prepared_state_cache_positions = None
        self._prepared_state_cache_masses = None
        self._topology_reuse_entry = None
        self._recent_topology_reused = False

    def clear_runtime_caches(
        self: "FastMultipoleMethod", *, clear_jax_compilation: bool = False
    ) -> None:
        """Release solver/runtime caches to reduce memory pressure."""

        self.clear_prepared_state_cache()
        self._locals_template = None
        self._interaction_cache = None
        self._interaction_cache_hits = 0
        self._interaction_cache_misses = 0
        self._tree_workspace = None
        self._last_force_scale_nodes = None
        self._recent_retry_events = tuple()
        self._recent_far_pairs_by_gear_counts = tuple()
        self._recent_dual_node_count = 0
        self._recent_dual_leaf_count = 0
        self._recent_dual_neighbor_count = 0
        self._recent_dual_far_pair_count = 0
        self._recent_dual_m2l_chunk_size = 0
        self._compiled_profile_fingerprint_last = None
        self._compiled_profile_transitions = 0
        self._large_n_eval_leaf_nodes_shape = ()
        self._large_n_eval_local_coefficients_shape = ()
        self._large_n_eval_local_centers_shape = ()
        self._large_n_eval_active_leaf_count = 0
        self._large_n_eval_max_leaf_size = 0
        self._large_n_eval_leaf_particle_slots = 0
        self._large_n_radix_payload_present = False
        self._large_n_radix_payload_source_particle_shape = ()
        self._large_n_radix_payload_source_particle_slots = 0
        self._large_n_radix_payload_source_leaf_shape = ()
        self._large_n_radix_payload_source_leaf_slots = 0
        self._large_n_target_block_source_leaf_padded_shape = ()
        self._compiled_profile_refresh_calls = 0
        self._compiled_profile_refresh_reuse_tier_full = 0
        self._compiled_profile_refresh_reuse_tier_topology = 0
        self._compiled_profile_refresh_reuse_tier_overflow = 0
        self._large_n_same_topology_refresh_attempts = 0
        self._large_n_same_topology_refresh_hits = 0
        self._large_n_same_topology_refresh_misses = 0
        self._large_n_same_topology_refresh_miss_no_key = 0
        self._large_n_same_topology_refresh_miss_topology = 0
        self._large_n_same_topology_refresh_miss_neighbor = 0
        self._large_n_same_topology_refresh_miss_traced = 0
        self._large_n_same_topology_refresh_last_error = ""
        self._static_radix_refresh_hits = 0
        self._static_radix_refresh_misses = 0
        self._static_radix_profile_overflows = 0
        self._static_radix_compact_pair_reuse_hits = 0
        self._static_radix_compact_pair_reuse_misses = 0
        self._compiled_profile_multipoles_only_calls = 0
        self._compiled_profile_topology_rebuild_calls = 0
        self._large_n_overflow_profile_cap = 0
        self._large_n_overflow_profile_reprofiles = 0
        self._large_n_neighbor_edges_profile_cap = 0
        self._large_n_neighbor_edges_profile_reprofiles = 0
        self._refresh_timing_total_seconds = 0.0
        self._refresh_timing_input_seconds = 0.0
        self._refresh_timing_tree_upward_seconds = 0.0
        self._refresh_timing_tree_build_seconds = 0.0
        self._refresh_timing_upward_compute_seconds = 0.0
        self._refresh_timing_upward_geometry_seconds = 0.0
        self._refresh_timing_upward_mass_moments_seconds = 0.0
        self._refresh_timing_upward_p2m_seconds = 0.0
        self._refresh_timing_upward_m2m_seconds = 0.0
        self._refresh_timing_upward_source_motion_seconds = 0.0
        self._refresh_timing_dual_downward_seconds = 0.0
        self._refresh_timing_nearfield_seconds = 0.0
        self._refresh_timing_profile_accounting_seconds = 0.0
        self._refresh_timing_compile_or_sync_suspect_seconds = 0.0
        self._refresh_timing_dual_setup_seconds = 0.0
        self._refresh_timing_dual_artifact_build_seconds = 0.0
        self._refresh_timing_dual_split_shared_far_near_seconds = 0.0
        self._refresh_timing_dual_split_shared_count_seconds = 0.0
        self._refresh_timing_dual_split_shared_combined_fill_seconds = 0.0
        self._refresh_timing_dual_split_shared_far_fill_seconds = 0.0
        self._refresh_timing_dual_split_shared_near_fill_seconds = 0.0
        self._refresh_timing_dual_split_far_pairs_seconds = 0.0
        self._refresh_timing_dual_split_leaf_neighbors_seconds = 0.0
        self._refresh_timing_dual_split_combined_seconds = 0.0
        self._refresh_timing_dual_raw_combined_seconds = 0.0
        self._refresh_timing_dual_split_dense_buffers_seconds = 0.0
        self._refresh_timing_dual_far_pair_plan_seconds = 0.0
        self._refresh_timing_dual_m2l_autotune_seconds = 0.0
        self._refresh_timing_dual_select_interactions_seconds = 0.0
        self._refresh_timing_dual_downward_compute_seconds = 0.0
        self._refresh_timing_dual_m2l_compute_seconds = 0.0
        self._refresh_timing_dual_l2l_compute_seconds = 0.0
        self._refresh_timing_dual_final_symmetry_seconds = 0.0
        self._refresh_timing_dual_source_motion_seconds = 0.0
        self._refresh_timing_dual_finalize_seconds = 0.0
        self._refresh_timing_dual_residual_seconds = 0.0
        self._refresh_timing_nearfield_leaf_groups_seconds = 0.0
        self._refresh_timing_nearfield_precompute_seconds = 0.0
        self._refresh_timing_nearfield_target_blocks_seconds = 0.0
        self._refresh_timing_nearfield_block_sort_seconds = 0.0
        self._refresh_timing_nearfield_speed_layout_seconds = 0.0
        self._refresh_timing_nearfield_overflow_profile_seconds = 0.0
        self._refresh_timing_nearfield_radix_payload_seconds = 0.0
        self._refresh_timing_nearfield_neighbor_padding_seconds = 0.0
        self._refresh_timing_nearfield_state_pack_seconds = 0.0
        self._refresh_timing_nearfield_residual_seconds = 0.0
        self._refresh_timing_calls = 0
        self._refresh_timing_active = False
        self._refresh_dual_planner_cache = {}
        self._refresh_dual_planner_cache_hits = 0
        self._refresh_dual_planner_cache_misses = 0
        self._refresh_dual_planner_compile_count = 0
        self._refresh_dual_planner_execute_count = 0
        self._refresh_dual_planner_steady_timing_bypass_count = 0
        self._refresh_dual_planner_compiled_route_count = 0
        self._refresh_strict_mode_active_count = 0
        self._strict_runner_compile_count = 0
        self._strict_runner_execute_count = 0
        self._strict_runner_profile_key_hits = 0
        self._strict_runner_profile_key_misses = 0
        self._strict_runner_fail_fast_reject_count = 0
        self._strict_runner_seen_profile_keys = set()
        self._strict_v2_compile_count = 0
        self._strict_v2_execute_count = 0
        self._strict_v2_profile_key_hits = 0
        self._strict_v2_profile_key_misses = 0
        self._strict_v2_fail_fast_reject_count = 0
        self._strict_v2_seen_profile_keys = set()
        self._strict_fused_mode_active = False
        self._strict_fused_compile_count = 0
        self._strict_fused_execute_count = 0
        self._strict_fused_profile_key_hits = 0
        self._strict_fused_profile_key_misses = 0
        self._strict_fused_fallback_count = 0
        self._strict_fused_last_fallback_reason = ""
        self._strict_fused_device_refresh_route_count = 0
        self._strict_fused_planner_bypassed_count = 0
        self._strict_velocity_verlet_acceleration_carry_active = False
        self._strict_self_force_bootstrap_evaluations = 0
        self._strict_self_force_endpoint_evaluations = 0
        self._strict_external_bootstrap_evaluations = 0
        self._strict_external_endpoint_evaluations = 0
        self._strict_static_target_block_capacity_ok = True
        self._strict_fused_seen_profile_keys = set()
        self._strict_fused_fastlane_attempts = 0
        self._strict_fused_fastlane_hits = 0
        self._strict_fused_fastlane_misses = 0
        self._strict_fused_fastlane_last_blockers = tuple()
        self._strict_fused_fastlane_block_counts = {}
        self._strict_fused_jit_function_cache = {}
        self._strict_profiled_max_pair_queue = 0
        self._strict_profiled_pair_process_block = 0
        self._strict_profiled_context_key = ""
        self._strict_profile_catalog = {}
        self._strict_profile_loaded_once = False
        _clear_global_runtime_caches(clear_jax_compilation=bool(clear_jax_compilation))

    def _strict_cap_profile_path(self) -> str:
        return str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH",
                "/tmp/jaccpot_static_strict_caps.json",
            )
        )

    def _strict_cap_profile_context_key(
        self,
        *,
        tree_mode: str,
        leaf_parameter: int,
        particle_count: int,
    ) -> str:
        return (
            f"tree_mode={str(tree_mode).strip().lower()}|"
            f"leaf={int(leaf_parameter)}|n={int(particle_count)}"
        )

    def _maybe_load_strict_cap_profile(
        self, *, context_key: Optional[str] = None
    ) -> None:
        if self._strict_profile_loaded_once:
            if context_key is not None:
                self._apply_strict_cap_profile_for_key(context_key=context_key)
            return
        self._strict_profile_loaded_once = True
        try:
            path = self._strict_cap_profile_path()
            if not os.path.exists(path):
                return
            payload = json.load(open(path, "r", encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("profiles"), dict):
                self._strict_profile_catalog = {
                    str(k): {
                        "max_pair_queue": int(v.get("max_pair_queue", 0) or 0),
                        "pair_process_block": int(v.get("pair_process_block", 0) or 0),
                    }
                    for k, v in payload["profiles"].items()
                    if isinstance(v, dict)
                }
            else:
                # Backward compatibility with the original single-profile payload.
                q = int(payload.get("max_pair_queue", 0) or 0)
                b = int(payload.get("pair_process_block", 0) or 0)
                self._strict_profile_catalog = {
                    "legacy_default": {
                        "max_pair_queue": q,
                        "pair_process_block": b,
                    }
                }
            if context_key is not None:
                self._apply_strict_cap_profile_for_key(context_key=context_key)
            elif len(self._strict_profile_catalog) > 0:
                # Preserve previous behavior when no context is supplied.
                self._apply_strict_cap_profile_for_key(context_key="legacy_default")
        except Exception:
            return

    def _apply_strict_cap_profile_for_key(self, *, context_key: str) -> None:
        selected_key = ""
        selected = self._strict_profile_catalog.get(context_key)
        if selected is not None:
            selected_key = context_key
        else:
            # Conservative fallback: keep same tree_mode+leaf and pick the largest queue.
            prefix = "|".join(str(context_key).split("|")[:2])
            best_q = 0
            best_entry: Optional[dict[str, int]] = None
            best_key = ""
            for key, entry in self._strict_profile_catalog.items():
                if not str(key).startswith(prefix):
                    continue
                q = int(entry.get("max_pair_queue", 0) or 0)
                if q >= best_q:
                    best_q = q
                    best_entry = entry
                    best_key = str(key)
            if best_entry is not None:
                selected = best_entry
                selected_key = best_key
            else:
                selected = self._strict_profile_catalog.get("legacy_default")
                selected_key = "legacy_default" if selected is not None else ""
        if selected is None:
            return
        q = int(selected.get("max_pair_queue", 0) or 0)
        b = int(selected.get("pair_process_block", 0) or 0)
        if q > 0:
            self._strict_profiled_max_pair_queue = q
        if b > 0:
            self._strict_profiled_pair_process_block = b
        if selected_key:
            self._strict_profiled_context_key = selected_key

    def _record_strict_cap_profile_from_retries(
        self,
        retry_events: Tuple[DualTreeRetryEvent, ...],
        *,
        context_key: Optional[str] = None,
    ) -> None:
        if len(retry_events) == 0:
            return
        max_queue = int(self._strict_profiled_max_pair_queue)
        max_block = int(self._strict_profiled_pair_process_block)
        for ev in retry_events:
            try:
                q = int(getattr(ev, "queue_capacity", 0) or 0)
            except Exception:
                q = 0
            if q > max_queue:
                max_queue = q
        block_hint = int(self.pair_process_block or 0)
        if block_hint > max_block:
            max_block = block_hint
        if max_queue <= 0 and max_block <= 0:
            return
        self._strict_profiled_max_pair_queue = max_queue
        self._strict_profiled_pair_process_block = max_block
        if context_key is not None:
            self._strict_profiled_context_key = str(context_key)
            existing = self._strict_profile_catalog.get(str(context_key), {})
            self._strict_profile_catalog[str(context_key)] = {
                "max_pair_queue": max(
                    int(existing.get("max_pair_queue", 0) or 0),
                    int(max_queue),
                ),
                "pair_process_block": max(
                    int(existing.get("pair_process_block", 0) or 0),
                    int(max_block),
                ),
            }
        if not bool(getattr(self, "_strict_cap_record_enabled", True)):
            return
        try:
            path = self._strict_cap_profile_path()
            payload = {
                "version": 2,
                "active_context_key": str(self._strict_profiled_context_key),
                "profiles": self._strict_profile_catalog,
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except Exception:
            return

    def _compiled_profile_from_prepared_state(
        self: "FastMultipoleMethod",
        state: PreparedStateLike,
    ) -> dict[str, Any]:
        """Build a stable-shape profile summary for compile-reuse diagnostics."""

        def _shape0(value: Any) -> int:
            if value is None:
                return 0
            return int(jnp.asarray(value).shape[0])

        def _shape_last(value: Any) -> int:
            if value is None:
                return 0
            arr = jnp.asarray(value)
            return int(arr.shape[-1]) if arr.ndim >= 1 else 0

        leaves, _ = jax.tree_util.tree_flatten(state)
        leaf_shapes: list[tuple[int, ...]] = []
        for leaf in leaves:
            shape = getattr(leaf, "shape", None)
            if shape is None:
                continue
            leaf_shapes.append(tuple(int(v) for v in shape))

        tree_parent = getattr(state.tree, "parent", None)
        neighbor_leaf_indices = getattr(state.neighbor_list, "leaf_indices", None)
        node_count = (
            int(jnp.asarray(tree_parent).shape[0]) if tree_parent is not None else 0
        )
        leaf_count = (
            int(jnp.asarray(neighbor_leaf_indices).shape[0])
            if neighbor_leaf_indices is not None
            else 0
        )
        nearfield_blocks = _shape0(getattr(state, "nearfield_target_leaf_ids", None))
        nearfield_target_block_slots = _shape0(
            getattr(state, "nearfield_target_block_source_leaf_ids", None)
        )
        leaf_particle_slots = _shape_last(
            getattr(state, "nearfield_leaf_particle_indices", None)
        )
        order = 0
        local_data = getattr(state, "local_data", None)
        if local_data is not None:
            order = int(getattr(local_data, "order", 0))
        else:
            downward = getattr(state, "downward", None)
            locals_view = (
                getattr(downward, "locals", None) if downward is not None else None
            )
            order = (
                int(getattr(locals_view, "order", 0)) if locals_view is not None else 0
            )

        return {
            "preset": str(self.preset),
            "runtime_path": str(self.runtime_path),
            "tree_type": str(self.tree_type),
            "execution_backend": str(getattr(state, "execution_backend", "unknown")),
            "expansion_basis": str(
                getattr(state, "expansion_basis", self.expansion_basis)
            ),
            "working_dtype": str(
                jnp.dtype(getattr(state, "working_dtype", self.working_dtype))
            ),
            "max_leaf_size": int(getattr(state, "max_leaf_size", 0)),
            "max_order": int(order),
            "max_nodes": int(node_count),
            "max_leaves": int(leaf_count),
            "max_nearfield_blocks": int(nearfield_blocks),
            "max_nearfield_target_block_slots": int(nearfield_target_block_slots),
            "max_leaf_particle_slots": int(leaf_particle_slots),
            "leaf_shapes": tuple(leaf_shapes),
        }

    def _record_large_n_eval_shape_diagnostics(
        self: "FastMultipoleMethod",
        state: PreparedStateLike,
    ) -> None:
        """Record shape-only local-eval diagnostics outside compiled hot loops."""
        neighbor_list = getattr(state, "neighbor_list", None)
        leaf_nodes = (
            getattr(neighbor_list, "leaf_indices", None)
            if neighbor_list is not None
            else None
        )
        local_data = getattr(state, "local_data", None)
        coefficients = getattr(local_data, "coefficients", None)
        centers = getattr(local_data, "centers", None)
        leaf_particles = getattr(state, "nearfield_leaf_particle_indices", None)
        radix_payload = getattr(state, "radix_fast_payload", None)
        radix_source_particles = (
            getattr(radix_payload, "source_particle_ids", None)
            if radix_payload is not None
            else None
        )
        radix_source_leaves = (
            getattr(radix_payload, "source_leaf_ids", None)
            if radix_payload is not None
            else None
        )
        padded_source_leaves = getattr(
            state, "nearfield_target_block_source_leaf_ids_padded", None
        )

        def _shape_tuple(value: Any) -> tuple[int, ...]:
            shape = getattr(value, "shape", None)
            if shape is None:
                return ()
            return tuple(int(v) for v in shape)

        leaf_shape = _shape_tuple(leaf_nodes)
        leaf_particle_shape = _shape_tuple(leaf_particles)
        radix_source_particle_shape = _shape_tuple(radix_source_particles)
        radix_source_leaf_shape = _shape_tuple(radix_source_leaves)
        padded_source_leaf_shape = _shape_tuple(padded_source_leaves)
        self._large_n_eval_leaf_nodes_shape = leaf_shape
        self._large_n_eval_local_coefficients_shape = _shape_tuple(coefficients)
        self._large_n_eval_local_centers_shape = _shape_tuple(centers)
        self._large_n_eval_active_leaf_count = int(leaf_shape[0]) if leaf_shape else 0
        self._large_n_eval_max_leaf_size = int(getattr(state, "max_leaf_size", 0))
        self._large_n_eval_leaf_particle_slots = (
            int(leaf_particle_shape[0] * leaf_particle_shape[1])
            if len(leaf_particle_shape) >= 2
            else 0
        )
        self._large_n_radix_payload_present = radix_payload is not None
        self._large_n_radix_payload_source_particle_shape = radix_source_particle_shape
        self._large_n_radix_payload_source_particle_slots = (
            int(
                radix_source_particle_shape[0]
                * radix_source_particle_shape[1]
                * radix_source_particle_shape[2]
            )
            if len(radix_source_particle_shape) >= 3
            else 0
        )
        self._large_n_radix_payload_source_leaf_shape = radix_source_leaf_shape
        self._large_n_radix_payload_source_leaf_slots = (
            int(
                radix_source_leaf_shape[0]
                * radix_source_leaf_shape[1]
                * radix_source_leaf_shape[2]
            )
            if len(radix_source_leaf_shape) >= 3
            else 0
        )
        self._large_n_target_block_source_leaf_padded_shape = padded_source_leaf_shape

    def _compiled_profile_fingerprint(
        self: "FastMultipoleMethod", profile: dict[str, Any]
    ) -> str:
        payload = json.dumps(profile, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _compiled_profile_capacity_compatible(
        self: "FastMultipoleMethod",
        base_profile: dict[str, Any],
        candidate_profile: dict[str, Any],
    ) -> bool:
        """Return True when candidate usage fits within base profile capacities."""
        capacity_fields = (
            "max_nodes",
            "max_leaves",
            "max_nearfield_blocks",
            "max_nearfield_target_block_slots",
            "max_leaf_particle_slots",
        )
        return all(
            int(candidate_profile.get(name, 0)) <= int(base_profile.get(name, 0))
            for name in capacity_fields
        )

    def _compiled_profile_record_transition(
        self: "FastMultipoleMethod",
        profile_fingerprint: str,
    ) -> None:
        prev = self._compiled_profile_fingerprint_last
        if prev is not None and profile_fingerprint != prev:
            self._compiled_profile_transitions += 1
        self._compiled_profile_fingerprint_last = profile_fingerprint

    def _strict_fused_profile_allows_n(self: "FastMultipoleMethod", n: int) -> bool:
        raw = str(getattr(self, "_strict_fused_profile_set_raw", "")).strip()
        if raw == "":
            return True
        allowed: set[int] = set()
        for token in raw.split(","):
            t = token.strip()
            if not t:
                continue
            try:
                allowed.add(int(t))
            except Exception:
                continue
        if not allowed:
            return True
        return int(n) in allowed

    def get_runtime_diagnostics(self: "FastMultipoleMethod") -> dict[str, Any]:
        """Return read-only runtime diagnostics for compile/profile reuse audits."""
        return {
            "compiled_profile_fingerprint_last": self._compiled_profile_fingerprint_last,
            "compiled_profile_transitions": int(self._compiled_profile_transitions),
            "refresh_prepare_calls": int(self._compiled_profile_refresh_calls),
            "refresh_prepare_reuse_tier_full": int(
                self._compiled_profile_refresh_reuse_tier_full
            ),
            "refresh_prepare_reuse_tier_topology": int(
                self._compiled_profile_refresh_reuse_tier_topology
            ),
            "refresh_prepare_reuse_tier_overflow": int(
                self._compiled_profile_refresh_reuse_tier_overflow
            ),
            "large_n_same_topology_refresh_attempts": int(
                self._large_n_same_topology_refresh_attempts
            ),
            "large_n_same_topology_refresh_hits": int(
                self._large_n_same_topology_refresh_hits
            ),
            "large_n_same_topology_refresh_misses": int(
                self._large_n_same_topology_refresh_misses
            ),
            "large_n_same_topology_refresh_miss_no_key": int(
                self._large_n_same_topology_refresh_miss_no_key
            ),
            "large_n_same_topology_refresh_miss_topology": int(
                self._large_n_same_topology_refresh_miss_topology
            ),
            "large_n_same_topology_refresh_miss_neighbor": int(
                self._large_n_same_topology_refresh_miss_neighbor
            ),
            "large_n_same_topology_refresh_miss_traced": int(
                self._large_n_same_topology_refresh_miss_traced
            ),
            "large_n_same_topology_refresh_last_error": str(
                self._large_n_same_topology_refresh_last_error
            ),
            "static_radix_refresh_hits": int(self._static_radix_refresh_hits),
            "static_radix_refresh_misses": int(self._static_radix_refresh_misses),
            "static_radix_profile_overflows": int(self._static_radix_profile_overflows),
            "static_radix_compact_pair_reuse_hits": int(
                self._static_radix_compact_pair_reuse_hits
            ),
            "static_radix_compact_pair_reuse_misses": int(
                self._static_radix_compact_pair_reuse_misses
            ),
            "update_multipoles_only_calls": int(
                self._compiled_profile_multipoles_only_calls
            ),
            "rebuild_topology_in_place_calls": int(
                self._compiled_profile_topology_rebuild_calls
            ),
            "large_n_overflow_profile_cap": int(self._large_n_overflow_profile_cap),
            "large_n_overflow_profile_reprofiles": int(
                self._large_n_overflow_profile_reprofiles
            ),
            "large_n_neighbor_edges_profile_cap": int(
                self._large_n_neighbor_edges_profile_cap
            ),
            "large_n_neighbor_edges_profile_reprofiles": int(
                self._large_n_neighbor_edges_profile_reprofiles
            ),
            "interaction_cache_hits": int(self._interaction_cache_hits),
            "interaction_cache_misses": int(self._interaction_cache_misses),
            "refresh_dual_planner_cache_hits": int(
                self._refresh_dual_planner_cache_hits
            ),
            "refresh_dual_planner_cache_misses": int(
                self._refresh_dual_planner_cache_misses
            ),
            "refresh_dual_planner_compile_count": int(
                self._refresh_dual_planner_compile_count
            ),
            "refresh_dual_planner_execute_count": int(
                self._refresh_dual_planner_execute_count
            ),
            "refresh_dual_planner_steady_timing_bypass_count": int(
                self._refresh_dual_planner_steady_timing_bypass_count
            ),
            "refresh_dual_planner_compiled_route_count": int(
                self._refresh_dual_planner_compiled_route_count
            ),
            "refresh_strict_mode_active_count": int(
                self._refresh_strict_mode_active_count
            ),
            "strict_runner_compile_count": int(self._strict_runner_compile_count),
            "strict_runner_execute_count": int(self._strict_runner_execute_count),
            "strict_runner_profile_key_hits": int(self._strict_runner_profile_key_hits),
            "strict_runner_profile_key_misses": int(
                self._strict_runner_profile_key_misses
            ),
            "strict_runner_fail_fast_reject_count": int(
                self._strict_runner_fail_fast_reject_count
            ),
            "strict_v2_compile_count": int(self._strict_v2_compile_count),
            "strict_v2_execute_count": int(self._strict_v2_execute_count),
            "strict_v2_profile_key_hits": int(self._strict_v2_profile_key_hits),
            "strict_v2_profile_key_misses": int(self._strict_v2_profile_key_misses),
            "strict_v2_fail_fast_reject_count": int(
                self._strict_v2_fail_fast_reject_count
            ),
            "large_n_eval_diag_mode": str(
                getattr(self, "_large_n_eval_diag_mode", "full")
            ),
            "large_n_nearfield_diag_mode": str(
                getattr(self, "_large_n_nearfield_diag_mode", "full")
            ),
            "large_n_eval_leaf_nodes_shape": tuple(
                getattr(self, "_large_n_eval_leaf_nodes_shape", ())
            ),
            "large_n_eval_local_coefficients_shape": tuple(
                getattr(self, "_large_n_eval_local_coefficients_shape", ())
            ),
            "large_n_eval_local_centers_shape": tuple(
                getattr(self, "_large_n_eval_local_centers_shape", ())
            ),
            "large_n_eval_active_leaf_count": int(
                getattr(self, "_large_n_eval_active_leaf_count", 0)
            ),
            "large_n_eval_max_leaf_size": int(
                getattr(self, "_large_n_eval_max_leaf_size", 0)
            ),
            "large_n_eval_leaf_particle_slots": int(
                getattr(self, "_large_n_eval_leaf_particle_slots", 0)
            ),
            "large_n_radix_payload_present": bool(
                getattr(self, "_large_n_radix_payload_present", False)
            ),
            "large_n_radix_payload_source_particle_shape": tuple(
                getattr(self, "_large_n_radix_payload_source_particle_shape", ())
            ),
            "large_n_radix_payload_source_particle_slots": int(
                getattr(self, "_large_n_radix_payload_source_particle_slots", 0)
            ),
            "large_n_radix_payload_source_leaf_shape": tuple(
                getattr(self, "_large_n_radix_payload_source_leaf_shape", ())
            ),
            "large_n_radix_payload_source_leaf_slots": int(
                getattr(self, "_large_n_radix_payload_source_leaf_slots", 0)
            ),
            "large_n_target_block_source_leaf_padded_shape": tuple(
                getattr(self, "_large_n_target_block_source_leaf_padded_shape", ())
            ),
            "strict_refresh_diag_mode": str(
                getattr(self, "_strict_refresh_diag_mode", "full")
            ),
            "strict_refresh_diag_tree_active": bool(
                getattr(self, "_strict_refresh_diag_tree_active", True)
            ),
            "strict_refresh_diag_upward_active": bool(
                getattr(self, "_strict_refresh_diag_upward_active", True)
            ),
            "strict_refresh_diag_downward_active": bool(
                getattr(self, "_strict_refresh_diag_downward_active", True)
            ),
            "strict_refresh_diag_eval_active": bool(
                getattr(self, "_strict_refresh_diag_eval_active", True)
            ),
            "strict_refresh_detail_diag_mode": str(
                getattr(self, "_strict_refresh_detail_diag_mode", "full")
            ),
            "static_radix_tree_leaf_count": int(
                getattr(self, "_static_radix_tree_leaf_count", 0)
            ),
            "static_radix_tree_node_count": int(
                getattr(self, "_static_radix_tree_node_count", 0)
            ),
            "static_radix_far_pair_count": int(
                getattr(self, "_static_radix_far_pair_count", 0)
            ),
            "static_radix_compact_pair_reuse_hits": int(
                getattr(self, "_static_radix_compact_pair_reuse_hits", 0)
            ),
            "static_radix_compact_pair_reuse_misses": int(
                getattr(self, "_static_radix_compact_pair_reuse_misses", 0)
            ),
            "static_radix_m2l_chunk_count": int(
                getattr(self, "_static_radix_m2l_chunk_count", 0)
            ),
            "static_radix_l2l_edge_count": int(
                getattr(self, "_static_radix_l2l_edge_count", 0)
            ),
            "strict_fused_mode_active": bool(self._strict_fused_mode_active),
            "strict_fused_compile_count": int(self._strict_fused_compile_count),
            "strict_fused_execute_count": int(self._strict_fused_execute_count),
            "strict_fused_profile_key_hits": int(self._strict_fused_profile_key_hits),
            "strict_fused_profile_key_misses": int(
                self._strict_fused_profile_key_misses
            ),
            "strict_fused_fallback_count": int(self._strict_fused_fallback_count),
            "strict_fused_last_fallback_reason": str(
                self._strict_fused_last_fallback_reason
            ),
            "strict_fused_device_refresh_route_count": int(
                self._strict_fused_device_refresh_route_count
            ),
            "strict_fused_planner_bypassed_count": int(
                self._strict_fused_planner_bypassed_count
            ),
            "strict_velocity_verlet_acceleration_carry_active": bool(
                self._strict_velocity_verlet_acceleration_carry_active
            ),
            "strict_self_force_bootstrap_evaluations": int(
                self._strict_self_force_bootstrap_evaluations
            ),
            "strict_self_force_initial_full_fmm_evaluations": int(
                self._strict_self_force_bootstrap_evaluations
            ),
            "strict_self_force_endpoint_evaluations": int(
                self._strict_self_force_endpoint_evaluations
            ),
            "strict_external_bootstrap_evaluations": int(
                self._strict_external_bootstrap_evaluations
            ),
            "strict_external_endpoint_evaluations": int(
                self._strict_external_endpoint_evaluations
            ),
            "strict_static_target_block_capacity_ok": bool(
                self._strict_static_target_block_capacity_ok
            ),
            "large_n_radix_fast_occupancy_sort": bool(
                self._large_n_radix_fast_occupancy_sort
            ),
            "large_n_radix_fast_skip_empty_tiles": bool(
                self._large_n_radix_fast_skip_empty_tiles
            ),
            "strict_fused_fastlane_diag_enabled": bool(
                self._strict_fused_fastlane_diag_enabled
            ),
            "strict_fused_fastlane_attempts": int(self._strict_fused_fastlane_attempts),
            "strict_fused_fastlane_hits": int(self._strict_fused_fastlane_hits),
            "strict_fused_fastlane_misses": int(self._strict_fused_fastlane_misses),
            "strict_fused_fastlane_last_blockers": tuple(
                str(v) for v in self._strict_fused_fastlane_last_blockers
            ),
            "strict_fused_fastlane_block_counts": {
                str(k): int(v)
                for k, v in dict(self._strict_fused_fastlane_block_counts).items()
            },
            "strict_profiled_max_pair_queue": int(self._strict_profiled_max_pair_queue),
            "strict_profiled_pair_process_block": int(
                self._strict_profiled_pair_process_block
            ),
            "strict_profiled_context_key": str(self._strict_profiled_context_key),
            "recent_dual_node_count": int(self._recent_dual_node_count),
            "recent_dual_leaf_count": int(self._recent_dual_leaf_count),
            "recent_dual_neighbor_count": int(self._recent_dual_neighbor_count),
            "recent_dual_far_pair_count": int(self._recent_dual_far_pair_count),
            "recent_dual_far_pairs_by_gear_counts": tuple(
                int(v) for v in self._recent_far_pairs_by_gear_counts
            ),
            "recent_dual_m2l_chunk_size": int(self._recent_dual_m2l_chunk_size),
            "refresh_total_seconds": float(self._refresh_timing_total_seconds),
            "refresh_input_seconds": float(self._refresh_timing_input_seconds),
            "refresh_tree_upward_seconds": float(
                self._refresh_timing_tree_upward_seconds
            ),
            "refresh_tree_build_seconds": float(
                self._refresh_timing_tree_build_seconds
            ),
            "refresh_upward_compute_seconds": float(
                self._refresh_timing_upward_compute_seconds
            ),
            "refresh_upward_geometry_seconds": float(
                self._refresh_timing_upward_geometry_seconds
            ),
            "refresh_upward_mass_moments_seconds": float(
                self._refresh_timing_upward_mass_moments_seconds
            ),
            "refresh_upward_p2m_seconds": float(
                self._refresh_timing_upward_p2m_seconds
            ),
            "refresh_upward_m2m_seconds": float(
                self._refresh_timing_upward_m2m_seconds
            ),
            "refresh_upward_source_motion_seconds": float(
                self._refresh_timing_upward_source_motion_seconds
            ),
            "refresh_dual_downward_seconds": float(
                self._refresh_timing_dual_downward_seconds
            ),
            "refresh_nearfield_seconds": float(self._refresh_timing_nearfield_seconds),
            "refresh_profile_accounting_seconds": float(
                self._refresh_timing_profile_accounting_seconds
            ),
            "refresh_compile_or_sync_suspect_seconds": float(
                self._refresh_timing_compile_or_sync_suspect_seconds
            ),
            "refresh_dual_setup_seconds": float(
                self._refresh_timing_dual_setup_seconds
            ),
            "refresh_dual_artifact_build_seconds": float(
                self._refresh_timing_dual_artifact_build_seconds
            ),
            "refresh_dual_split_shared_far_near_seconds": float(
                self._refresh_timing_dual_split_shared_far_near_seconds
            ),
            "refresh_dual_split_shared_count_seconds": float(
                self._refresh_timing_dual_split_shared_count_seconds
            ),
            "refresh_dual_split_shared_combined_fill_seconds": float(
                self._refresh_timing_dual_split_shared_combined_fill_seconds
            ),
            "refresh_dual_split_shared_far_fill_seconds": float(
                self._refresh_timing_dual_split_shared_far_fill_seconds
            ),
            "refresh_dual_split_shared_near_fill_seconds": float(
                self._refresh_timing_dual_split_shared_near_fill_seconds
            ),
            "refresh_dual_split_far_pairs_seconds": float(
                self._refresh_timing_dual_split_far_pairs_seconds
            ),
            "refresh_dual_split_leaf_neighbors_seconds": float(
                self._refresh_timing_dual_split_leaf_neighbors_seconds
            ),
            "refresh_dual_split_combined_seconds": float(
                self._refresh_timing_dual_split_combined_seconds
            ),
            "refresh_dual_raw_combined_seconds": float(
                self._refresh_timing_dual_raw_combined_seconds
            ),
            "refresh_dual_split_dense_buffers_seconds": float(
                self._refresh_timing_dual_split_dense_buffers_seconds
            ),
            "refresh_dual_far_pair_plan_seconds": float(
                self._refresh_timing_dual_far_pair_plan_seconds
            ),
            "refresh_dual_m2l_autotune_seconds": float(
                self._refresh_timing_dual_m2l_autotune_seconds
            ),
            "refresh_dual_select_interactions_seconds": float(
                self._refresh_timing_dual_select_interactions_seconds
            ),
            "refresh_dual_downward_compute_seconds": float(
                self._refresh_timing_dual_downward_compute_seconds
            ),
            "refresh_dual_m2l_compute_seconds": float(
                self._refresh_timing_dual_m2l_compute_seconds
            ),
            "refresh_dual_l2l_compute_seconds": float(
                self._refresh_timing_dual_l2l_compute_seconds
            ),
            "refresh_dual_final_symmetry_seconds": float(
                self._refresh_timing_dual_final_symmetry_seconds
            ),
            "refresh_dual_source_motion_seconds": float(
                self._refresh_timing_dual_source_motion_seconds
            ),
            "refresh_dual_finalize_seconds": float(
                self._refresh_timing_dual_finalize_seconds
            ),
            "refresh_dual_residual_seconds": float(
                self._refresh_timing_dual_residual_seconds
            ),
            "refresh_nearfield_leaf_groups_seconds": float(
                self._refresh_timing_nearfield_leaf_groups_seconds
            ),
            "refresh_nearfield_precompute_seconds": float(
                self._refresh_timing_nearfield_precompute_seconds
            ),
            "refresh_nearfield_target_blocks_seconds": float(
                self._refresh_timing_nearfield_target_blocks_seconds
            ),
            "refresh_nearfield_block_sort_seconds": float(
                self._refresh_timing_nearfield_block_sort_seconds
            ),
            "refresh_nearfield_speed_layout_seconds": float(
                self._refresh_timing_nearfield_speed_layout_seconds
            ),
            "refresh_nearfield_overflow_profile_seconds": float(
                self._refresh_timing_nearfield_overflow_profile_seconds
            ),
            "refresh_nearfield_radix_payload_seconds": float(
                self._refresh_timing_nearfield_radix_payload_seconds
            ),
            "refresh_nearfield_neighbor_padding_seconds": float(
                self._refresh_timing_nearfield_neighbor_padding_seconds
            ),
            "refresh_nearfield_state_pack_seconds": float(
                self._refresh_timing_nearfield_state_pack_seconds
            ),
            "refresh_nearfield_residual_seconds": float(
                self._refresh_timing_nearfield_residual_seconds
            ),
            "refresh_timing_calls": int(self._refresh_timing_calls),
        }

    def refresh_prepared_state(
        self: "FastMultipoleMethod",
        prepared_state: PreparedStateLike,
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
        fused_device_mode: bool = False,
    ) -> PreparedStateLike:
        """Refresh prepared state under large-N/radix profile constraints."""
        if not self._is_large_n_gpu_production_profile():
            raise NotImplementedError(
                "refresh_prepared_state is currently supported only for "
                "preset='large_n_gpu', tree_type='radix', expansion_basis='solidfmm'."
            )
        if not isinstance(prepared_state, LargeNPreparedState):
            raise NotImplementedError(
                "refresh_prepared_state currently supports LargeNPreparedState only."
            )

        self._compiled_profile_refresh_calls += 1
        refresh_timing_enabled = bool(getattr(self, "_refresh_timing_enabled", False))
        if not refresh_timing_enabled:
            next_state = self._refresh_large_n_same_topology(
                prepared_state,
                positions,
                masses,
                bounds=bounds,
                leaf_size=int(
                    prepared_state.max_leaf_size if leaf_size is None else leaf_size
                ),
                max_order=(
                    int(prepared_state.local_data.order)
                    if max_order is None
                    else int(max_order)
                ),
                theta=theta,
                fused_device_mode=bool(fused_device_mode),
            )
            if next_state is None:
                next_state = self.prepare_state(
                    positions,
                    masses,
                    bounds=bounds,
                    leaf_size=int(
                        prepared_state.max_leaf_size if leaf_size is None else leaf_size
                    ),
                    max_order=(
                        int(prepared_state.local_data.order)
                        if max_order is None
                        else int(max_order)
                    ),
                    theta=theta,
                    fused_device_mode=bool(fused_device_mode),
                )
            return next_state

        refresh_t0 = time.perf_counter()
        input_before = float(getattr(self, "_refresh_timing_input_seconds", 0.0))
        tree_before = float(getattr(self, "_refresh_timing_tree_upward_seconds", 0.0))
        dual_before = float(getattr(self, "_refresh_timing_dual_downward_seconds", 0.0))
        nearfield_before = float(
            getattr(self, "_refresh_timing_nearfield_seconds", 0.0)
        )
        profile_t0 = time.perf_counter()
        prev_profile = self._compiled_profile_from_prepared_state(prepared_state)
        prev_fingerprint = self._compiled_profile_fingerprint(prev_profile)
        profile_seconds = time.perf_counter() - profile_t0

        was_refresh_timing_active = bool(getattr(self, "_refresh_timing_active", False))
        self._refresh_timing_active = True
        try:
            next_state = self._refresh_large_n_same_topology(
                prepared_state,
                positions,
                masses,
                bounds=bounds,
                leaf_size=int(
                    prepared_state.max_leaf_size if leaf_size is None else leaf_size
                ),
                max_order=(
                    int(prepared_state.local_data.order)
                    if max_order is None
                    else int(max_order)
                ),
                theta=theta,
                fused_device_mode=bool(fused_device_mode),
            )
            if next_state is None:
                next_state = self.prepare_state(
                    positions,
                    masses,
                    bounds=bounds,
                    leaf_size=int(
                        prepared_state.max_leaf_size if leaf_size is None else leaf_size
                    ),
                    max_order=(
                        int(prepared_state.local_data.order)
                        if max_order is None
                        else int(max_order)
                    ),
                    theta=theta,
                    fused_device_mode=bool(fused_device_mode),
                )
        finally:
            self._refresh_timing_active = was_refresh_timing_active
        prepare_elapsed = time.perf_counter() - refresh_t0

        profile_t0 = time.perf_counter()
        next_profile = self._compiled_profile_from_prepared_state(next_state)
        next_fingerprint = self._compiled_profile_fingerprint(next_profile)
        self._compiled_profile_record_transition(next_fingerprint)

        if next_fingerprint == prev_fingerprint:
            self._compiled_profile_refresh_reuse_tier_full += 1
        elif self._compiled_profile_capacity_compatible(prev_profile, next_profile):
            self._compiled_profile_refresh_reuse_tier_topology += 1
        else:
            self._compiled_profile_refresh_reuse_tier_overflow += 1
        profile_seconds += time.perf_counter() - profile_t0
        total_elapsed = time.perf_counter() - refresh_t0
        input_delta = (
            float(getattr(self, "_refresh_timing_input_seconds", 0.0)) - input_before
        )
        tree_delta = (
            float(getattr(self, "_refresh_timing_tree_upward_seconds", 0.0))
            - tree_before
        )
        dual_delta = (
            float(getattr(self, "_refresh_timing_dual_downward_seconds", 0.0))
            - dual_before
        )
        nearfield_delta = (
            float(getattr(self, "_refresh_timing_nearfield_seconds", 0.0))
            - nearfield_before
        )
        stage_sum = (
            input_delta
            + tree_delta
            + dual_delta
            + nearfield_delta
            + float(profile_seconds)
        )
        # prepare_large_n_state records cumulative stage timings directly on
        # the solver. Attribute the unaccounted part of this refresh to Python
        # overhead, sync, compilation, or other work outside the explicit
        # large-N stage timers.
        self._refresh_timing_profile_accounting_seconds += float(profile_seconds)
        self._refresh_timing_total_seconds += float(total_elapsed)
        self._refresh_timing_compile_or_sync_suspect_seconds += max(
            0.0,
            float(total_elapsed) - float(stage_sum),
        )
        self._refresh_timing_calls += 1
        return next_state

    def strict_prepare_refresh_and_evaluate(
        self: "FastMultipoleMethod",
        prepared_state: Optional[PreparedStateLike],
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        theta: Optional[float] = None,
        jit_traversal: Optional[bool] = True,
        runtime_overrides: Optional[_RuntimeExecutionOverrides] = None,
        fused_device_mode: Optional[bool] = None,
    ) -> tuple[PreparedStateLike, Array]:
        """Strict static-radix helper: prepare/refresh once, then evaluate."""
        if not self._is_large_n_gpu_production_profile():
            self._strict_runner_fail_fast_reject_count += 1
            raise RuntimeError(
                "strict_prepare_refresh_and_evaluate requires large_n_gpu production profile."
            )

        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        profile_key = (
            f"n={int(positions_arr.shape[0])}|"
            f"leaf={int(leaf_size)}|"
            f"order={int(max_order)}|"
            f"theta={float(self.theta if theta is None else theta):.12g}"
        )
        if profile_key in self._strict_runner_seen_profile_keys:
            self._strict_runner_profile_key_hits += 1
        else:
            self._strict_runner_profile_key_misses += 1
            self._strict_runner_compile_count += 1
            self._strict_runner_seen_profile_keys.add(profile_key)
        self._strict_runner_execute_count += 1

        if prepared_state is None:
            next_state = self.prepare_state(
                positions_arr,
                masses_arr,
                bounds=bounds,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_tree=self._jit_tree_default,
                runtime_overrides_override=runtime_overrides,
                fused_device_mode=bool(
                    self._strict_fused_mode_active
                    if fused_device_mode is None
                    else fused_device_mode
                ),
            )
        else:
            if not isinstance(prepared_state, LargeNPreparedState):
                self._strict_runner_fail_fast_reject_count += 1
                raise RuntimeError(
                    "strict_prepare_refresh_and_evaluate requires LargeNPreparedState input."
                )
            next_state_try = self._refresh_large_n_same_topology(
                prepared_state,
                positions_arr,
                masses_arr,
                bounds=bounds,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                runtime_overrides_override=runtime_overrides,
                fused_device_mode=bool(
                    self._strict_fused_mode_active
                    if fused_device_mode is None
                    else fused_device_mode
                ),
            )
            if next_state_try is None:
                self._strict_runner_fail_fast_reject_count += 1
                raise RuntimeError(
                    "strict_prepare_refresh_and_evaluate fail-fast: "
                    "refresh miss (profile/topology mismatch)."
                )
            next_state = next_state_try

        acc = self.evaluate_prepared_state(
            next_state,
            target_indices=None,
            return_potential=False,
            jit_traversal=(
                self._jit_traversal_default
                if jit_traversal is None
                else bool(jit_traversal)
            ),
            max_acc_derivative_order=0,
        )
        return next_state, jnp.asarray(acc)

    def strict_run_segmented(
        self: "FastMultipoleMethod",
        *,
        state: Any,
        masses: Array,
        num_steps: int,
        refresh_every: int,
        segment_runner: Callable[[Any, Array, int], tuple[Any, Any]],
        positions_getter: Callable[[Any], Array],
        prepared_state: Optional[PreparedStateLike] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        theta: Optional[float] = None,
        jit_traversal: Optional[bool] = True,
        rematerialize_fn: Optional[Callable[[Any], Any]] = None,
        collect_history: bool = False,
    ) -> tuple[Any, PreparedStateLike, Optional[list[Any]]]:
        """Run strict refresh/evaluate cadence with caller-provided segment runner."""
        if int(num_steps) <= 0:
            raise ValueError("num_steps must be positive")
        if int(refresh_every) <= 0:
            raise ValueError("refresh_every must be positive")

        num_steps_i = int(num_steps)
        refresh_every_i = int(refresh_every)
        full_segments = num_steps_i // refresh_every_i
        tail_segment = num_steps_i % refresh_every_i

        state_curr = state
        prepared_curr = prepared_state
        history: Optional[list[Any]] = [] if collect_history else None
        runtime_overrides_cached = self._resolve_runtime_execution_overrides(
            num_particles=int(jnp.asarray(masses).shape[0]),
        )

        for _ in range(full_segments):
            positions_curr = positions_getter(state_curr)
            prepared_curr, acc_self = self.strict_prepare_refresh_and_evaluate(
                prepared_curr,
                positions_curr,
                masses,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_traversal=jit_traversal,
                runtime_overrides=runtime_overrides_cached,
                fused_device_mode=bool(self._strict_fused_mode_active),
            )
            state_curr, seg_hist = segment_runner(
                state_curr,
                jnp.asarray(acc_self),
                int(refresh_every_i),
            )
            if rematerialize_fn is not None:
                state_curr = rematerialize_fn(state_curr)
            if history is not None:
                history.append(seg_hist)

        if tail_segment > 0:
            positions_curr = positions_getter(state_curr)
            prepared_curr, acc_self = self.strict_prepare_refresh_and_evaluate(
                prepared_curr,
                positions_curr,
                masses,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_traversal=jit_traversal,
                runtime_overrides=runtime_overrides_cached,
                fused_device_mode=bool(self._strict_fused_mode_active),
            )
            state_curr, seg_hist = segment_runner(
                state_curr,
                jnp.asarray(acc_self),
                int(tail_segment),
            )
            if rematerialize_fn is not None:
                state_curr = rematerialize_fn(state_curr)
            if history is not None:
                history.append(seg_hist)

        return state_curr, prepared_curr, history

    def strict_run_v2(
        self: "FastMultipoleMethod",
        *,
        state: Array,
        masses: Array,
        dt: float,
        num_steps: int,
        refresh_every: int,
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
        prepared_state: Optional[PreparedStateLike] = None,
        initial_self_acceleration: Optional[Array] = None,
        jit_traversal: Optional[bool] = True,
        add_external: bool = False,
        external_acceleration_fn: Optional[Callable[[Array], Array]] = None,
        rematerialize_between_refresh: bool = True,
        return_history: bool = False,
        return_prepared_state: bool = True,
        step_callback: Optional[Callable[[Array, Array], None]] = None,
        step_callback_stride: int = 1,
    ) -> tuple[Array, Optional[PreparedStateLike], Optional[Array]]:
        """Run endpoint-correct velocity Verlet with strict prepared-state refresh.

        ``step_callback`` is an optional traced, side-effecting hook called inside
        the device-resident scan as ``step_callback(step_index, state)`` every
        ``step_callback_stride`` steps (``step_index`` and ``state`` are traced
        device values). It must be fire-and-forget (return nothing) and should use
        ``jax.debug.callback`` internally to ship only small, on-device-reduced
        data to the host (e.g. a projected density grid), so the GPU is not
        stalled. It does not touch the scan carry and is independent of
        ``return_history``."""
        state_arr = jnp.asarray(state)
        masses_arr = jnp.asarray(masses)
        dt_arr = jnp.asarray(float(dt), dtype=state_arr.dtype)
        num_steps_i = int(num_steps)

        if not self._is_large_n_gpu_production_profile():
            self._strict_v2_fail_fast_reject_count += 1
            raise RuntimeError("strict_run_v2 requires large_n_gpu production profile.")
        if num_steps_i <= 0:
            raise ValueError("num_steps must be positive")
        if int(refresh_every) != 1:
            self._strict_v2_fail_fast_reject_count += 1
            raise ValueError(
                "strict_run_v2 requires refresh_every=1 for endpoint-correct "
                "velocity-Verlet self gravity"
            )

        profile_key = (
            f"n={int(state_arr.shape[0])}|leaf={int(leaf_size)}|"
            f"order={int(max_order)}|refresh=1|"
            f"dt={float(dt):.12g}|external={int(bool(add_external))}|"
            f"theta={float(self.theta if theta is None else theta):.12g}"
        )
        if profile_key in self._strict_v2_seen_profile_keys:
            self._strict_v2_profile_key_hits += 1
        else:
            self._strict_v2_profile_key_misses += 1
            self._strict_v2_compile_count += 1
            self._strict_v2_seen_profile_keys.add(profile_key)
        self._strict_v2_execute_count += 1

        fused_mode_requested = bool(getattr(self, "_strict_fused_mode_enabled", False))
        fused_mode_allowed = self._strict_fused_profile_allows_n(
            int(state_arr.shape[0])
        )
        self._strict_fused_mode_active = bool(
            fused_mode_requested and fused_mode_allowed
        )
        if self._strict_fused_mode_active:
            if profile_key in self._strict_fused_seen_profile_keys:
                self._strict_fused_profile_key_hits += 1
            else:
                self._strict_fused_profile_key_misses += 1
                self._strict_fused_compile_count += 1
                self._strict_fused_seen_profile_keys.add(profile_key)
            self._strict_fused_execute_count += 1
            self._strict_fused_device_refresh_route_count += num_steps_i
            self._strict_fused_planner_bypassed_count += num_steps_i
        elif fused_mode_requested and not fused_mode_allowed:
            self._strict_fused_fallback_count += 1
            self._strict_fused_last_fallback_reason = (
                "particle_count_not_in_JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET"
            )
        else:
            self._strict_fused_last_fallback_reason = ""

        self._strict_velocity_verlet_acceleration_carry_active = True
        diag_mode = str(getattr(self, "_strict_refresh_diag_mode", "full"))
        eval_diag_mode = str(getattr(self, "_large_n_eval_diag_mode", "full"))
        detail_diag_mode = str(
            getattr(self, "_strict_refresh_detail_diag_mode", "full")
        )
        self_eval_active = (
            bool(getattr(self, "_strict_refresh_diag_eval_active", True))
            and detail_diag_mode == "full"
            and eval_diag_mode != "zero"
        )
        self._strict_self_force_bootstrap_evaluations = int(self_eval_active)
        self._strict_self_force_endpoint_evaluations = (
            num_steps_i if self_eval_active else 0
        )
        self._strict_external_bootstrap_evaluations = int(
            bool(add_external) and external_acceleration_fn is not None
        )
        self._strict_external_endpoint_evaluations = (
            num_steps_i
            if bool(add_external) and external_acceleration_fn is not None
            else 0
        )

        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(state_arr.shape[0])
        )
        prepared_curr = prepared_state
        if prepared_curr is None:
            prepared_curr = self.prepare_state(
                state_arr[:, 0, :],
                masses_arr,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_tree=self._jit_tree_default,
                runtime_overrides_override=runtime_overrides,
                fused_device_mode=bool(self._strict_fused_mode_active),
            )
        if self._strict_fused_mode_active and not isinstance(
            prepared_curr, LargeNPreparedState
        ):
            self._strict_runner_fail_fast_reject_count += 1
            raise RuntimeError(
                "strict fused velocity-Verlet requires LargeNPreparedState input."
            )
        if isinstance(prepared_curr, LargeNPreparedState):
            self._record_large_n_eval_shape_diagnostics(prepared_curr)

        def _evaluate_self(prepared_in: PreparedStateLike, state_in: Array) -> Array:
            if not self_eval_active:
                return jnp.zeros_like(state_in[:, 0, :])
            if eval_diag_mode == "permutation_only":
                return jnp.asarray(prepared_in.positions_sorted)[
                    prepared_in.inverse_permutation
                ] * jnp.asarray(0.0, dtype=state_in.dtype)
            return jnp.asarray(
                evaluate_large_n_state(
                    self,
                    prepared_in,
                    target_indices=None,
                    return_potential=False,
                    max_acc_derivative_order=0,
                ),
                dtype=state_in.dtype,
            )

        if not self_eval_active:
            acceleration_self_current = jnp.zeros_like(state_arr[:, 0, :])
        elif initial_self_acceleration is None:
            acceleration_self_current = _evaluate_self(prepared_curr, state_arr)
        else:
            acceleration_self_current = jnp.asarray(
                initial_self_acceleration, dtype=state_arr.dtype
            )
        if add_external and external_acceleration_fn is not None:
            acceleration_current = acceleration_self_current + jnp.asarray(
                external_acceleration_fn(state_arr), dtype=state_arr.dtype
            )
        else:
            acceleration_current = acceleration_self_current

        def _static_target_block_capacity_ok(
            prepared_in: PreparedStateLike,
        ) -> Array:
            padded = getattr(
                prepared_in,
                "nearfield_target_block_source_leaf_ids_padded",
                None,
            )
            if padded is None:
                return jnp.asarray(True)
            padded_arr = jnp.asarray(padded)
            if padded_arr.ndim != 3 or int(padded_arr.shape[1]) == 0:
                return jnp.asarray(True)
            offsets = jnp.asarray(prepared_in.neighbor_list.offsets)
            counts = offsets[1:] - offsets[:-1]
            capacity = int(padded_arr.shape[1]) * int(padded_arr.shape[2])
            return jnp.all(counts <= jnp.asarray(capacity, dtype=counts.dtype))

        def _refresh_and_evaluate_endpoint(
            prepared_in: PreparedStateLike,
            state_position: Array,
        ) -> tuple[PreparedStateLike, Array]:
            if diag_mode in {"integrator_only", "eval_only"}:
                prepared_new = prepared_in
            else:
                prepared_new = self._refresh_large_n_same_topology(
                    prepared_in,
                    state_position[:, 0, :],
                    masses_arr,
                    bounds=None,
                    leaf_size=int(leaf_size),
                    max_order=int(max_order),
                    theta=theta,
                    runtime_overrides_override=None,
                    fused_device_mode=bool(self._strict_fused_mode_active),
                )
                if prepared_new is None:
                    raise RuntimeError(
                        "strict velocity-Verlet refresh failed: topology/profile mismatch"
                    )
            return prepared_new, _evaluate_self(prepared_new, state_position)

        if self._strict_fused_mode_active:
            # Stash the concrete tree depth now, while prepared_curr is concrete,
            # so the traced refresh inside the compiled runner passes it as the
            # M2M level-loop static arg. Keyed into cache_key so a topology with a
            # different depth compiles its own runner.
            static_upward_num_levels = self._resolve_upward_num_levels(
                getattr(prepared_curr, "tree", None)
            )
            cache_key = (
                "strict_velocity_verlet",
                tuple(int(v) for v in state_arr.shape),
                str(state_arr.dtype),
                tuple(int(v) for v in masses_arr.shape),
                str(masses_arr.dtype),
                float(dt),
                num_steps_i,
                int(leaf_size),
                int(max_order),
                float(self.theta if theta is None else theta),
                bool(add_external),
                (
                    id(external_acceleration_fn)
                    if external_acceleration_fn is not None
                    else 0
                ),
                bool(rematerialize_between_refresh),
                bool(return_history),
                diag_mode,
                detail_diag_mode,
                eval_diag_mode,
                str(getattr(self, "_large_n_nearfield_diag_mode", "full")),
                static_upward_num_levels,
                id(step_callback) if step_callback is not None else 0,
                int(step_callback_stride),
            )
            jit_cache = getattr(self, "_strict_fused_jit_function_cache", {})
            compiled_runner = jit_cache.get(cache_key)
            if compiled_runner is None:

                @jax.jit
                def _compiled_runner(
                    prepared_initial: LargeNPreparedState,
                    state_initial: Array,
                    acceleration_initial: Array,
                ) -> tuple[
                    tuple[LargeNPreparedState, Array, Array, Array], Optional[Array]
                ]:
                    def _step(carry, scan_x):
                        (
                            prepared_now,
                            state_now,
                            acceleration_now,
                            capacity_ok_now,
                        ) = carry
                        position_new = (
                            state_now[:, 0]
                            + state_now[:, 1] * dt_arr
                            + 0.5 * acceleration_now * dt_arr**2
                        )
                        state_position = state_now.at[:, 0].set(position_new)
                        prepared_new, acceleration_self_new = (
                            _refresh_and_evaluate_endpoint(prepared_now, state_position)
                        )
                        if add_external and external_acceleration_fn is not None:
                            acceleration_new = acceleration_self_new + jnp.asarray(
                                external_acceleration_fn(state_position),
                                dtype=state_now.dtype,
                            )
                        else:
                            acceleration_new = acceleration_self_new
                        state_new = _velocity_verlet_state_update(
                            state_now,
                            acceleration_now,
                            acceleration_new,
                            dt_arr,
                        )
                        if rematerialize_between_refresh:
                            state_new = jnp.asarray(state_new, dtype=state_now.dtype)
                        if step_callback is not None:
                            # Fire-and-forget streaming hook (e.g. render). Gated by
                            # stride via lax.cond so it only fires + only computes its
                            # on-device reduction on emit steps. Returns a dummy int so
                            # both cond branches match; the result is discarded and the
                            # scan carry is untouched.
                            def _emit(_):
                                step_callback(scan_x, state_new)
                                return jnp.int32(0)

                            def _skip(_):
                                return jnp.int32(0)

                            jax.lax.cond(
                                (scan_x % jnp.int32(step_callback_stride))
                                == jnp.int32(0),
                                _emit,
                                _skip,
                                operand=None,
                            )
                        capacity_ok_new = capacity_ok_now & (
                            _static_target_block_capacity_ok(prepared_new)
                        )
                        return (
                            prepared_new,
                            state_new,
                            acceleration_new,
                            capacity_ok_new,
                        ), (state_new if return_history else None)

                    # Feed a per-step index only when a streaming callback needs it
                    # (keeps the no-callback path byte-for-byte unchanged).
                    scan_xs = (
                        jnp.arange(num_steps_i, dtype=jnp.int32)
                        if step_callback is not None
                        else None
                    )
                    return jax.lax.scan(
                        _step,
                        (
                            prepared_initial,
                            state_initial,
                            acceleration_initial,
                            _static_target_block_capacity_ok(prepared_initial),
                        ),
                        xs=scan_xs,
                        length=num_steps_i,
                    )

                compiled_runner = _compiled_runner
                jit_cache[cache_key] = compiled_runner
                self._strict_fused_jit_function_cache = jit_cache

            try:
                (
                    prepared_curr,
                    state_curr,
                    _,
                    capacity_ok_all,
                ), history_out = compiled_runner(
                    prepared_curr,
                    state_arr,
                    jnp.asarray(acceleration_current, dtype=state_arr.dtype),
                )
                self._strict_static_target_block_capacity_ok = bool(
                    np.asarray(jax.device_get(capacity_ok_all))
                )
                if not self._strict_static_target_block_capacity_ok:
                    max_blocks = os.environ.get(
                        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
                        "32",
                    )
                    raise RuntimeError(
                        "fused payload static target-block cap exceeded during "
                        "compiled velocity-Verlet scan: max_blocks_per_leaf="
                        f"{max_blocks}. Increase "
                        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF."
                    )
            except Exception as exc:
                if bool(
                    getattr(self, "_strict_fused_disallow_host_segment_fallback", False)
                ):
                    raise RuntimeError(
                        "strict fused velocity-Verlet scan failed while host fallback "
                        "is disallowed"
                    ) from exc
                raise
        else:
            state_curr = state_arr
            history_parts: list[Array] = []
            acceleration_now = jnp.asarray(acceleration_current, dtype=state_arr.dtype)
            for _ in range(num_steps_i):
                position_new = (
                    state_curr[:, 0]
                    + state_curr[:, 1] * dt_arr
                    + 0.5 * acceleration_now * dt_arr**2
                )
                state_position = state_curr.at[:, 0].set(position_new)
                prepared_curr, acceleration_self_new = _refresh_and_evaluate_endpoint(
                    prepared_curr, state_position
                )
                if add_external and external_acceleration_fn is not None:
                    acceleration_new = acceleration_self_new + jnp.asarray(
                        external_acceleration_fn(state_position),
                        dtype=state_curr.dtype,
                    )
                else:
                    acceleration_new = acceleration_self_new
                state_curr = _velocity_verlet_state_update(
                    state_curr, acceleration_now, acceleration_new, dt_arr
                )
                acceleration_now = acceleration_new
                if return_history:
                    history_parts.append(state_curr)
            history_out = jnp.stack(history_parts, axis=0) if return_history else None

        self._strict_runner_execute_count += num_steps_i
        if profile_key in self._strict_runner_seen_profile_keys:
            self._strict_runner_profile_key_hits += num_steps_i
        else:
            self._strict_runner_seen_profile_keys.add(profile_key)
            self._strict_runner_compile_count += 1
            self._strict_runner_profile_key_misses += 1
            self._strict_runner_profile_key_hits += max(0, num_steps_i - 1)
        prepared_out = prepared_curr if return_prepared_state else None
        return state_curr, prepared_out, history_out

    def strict_fused_prepared_eval_fn(
        self: "FastMultipoleMethod",
        *,
        positions: Array,
        masses: Array,
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
    ) -> tuple[PreparedStateLike, Callable[[PreparedStateLike], Array]]:
        """Build a fused-lane prepared state and return a jitted eval-only closure.

        Isolates the *evaluate* cost of the strict fused static-radix lane for
        apples-to-apples benchmarking against functional FMM eval APIs (e.g.
        jaxfmm ``eval_potential``): the prepared state is built eagerly with the
        fused device-mode layout (optimized flat compact far-pairs + static
        target-block near-field), exactly as ``strict_run_v2`` bootstraps it, and
        the returned closure runs the same self-force evaluation the fused step
        runs per endpoint (``evaluate_large_n_state``) with **no refresh and no
        velocity-Verlet update**.

        Returns ``(prepared_state, eval_fn)``; time ``eval_fn(prepared_state)``.
        """
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        if not self._is_large_n_gpu_production_profile():
            raise RuntimeError(
                "strict_fused_prepared_eval_fn requires large_n_gpu production profile."
            )
        fused_mode_requested = bool(getattr(self, "_strict_fused_mode_enabled", False))
        fused_mode_allowed = self._strict_fused_profile_allows_n(
            int(positions_arr.shape[0])
        )
        self._strict_fused_mode_active = bool(
            fused_mode_requested and fused_mode_allowed
        )
        if not self._strict_fused_mode_active:
            raise RuntimeError(
                "strict fused mode is not active for this particle count/config; "
                "enable JACCPOT_STATIC_STRICT_FUSED_MODE and include N in "
                "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET."
            )
        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(positions_arr.shape[0])
        )
        prepared = self.prepare_state(
            positions_arr,
            masses_arr,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            theta=theta,
            jit_tree=self._jit_tree_default,
            runtime_overrides_override=runtime_overrides,
            fused_device_mode=True,
        )
        if not isinstance(prepared, LargeNPreparedState):
            raise RuntimeError("strict fused eval-only requires a LargeNPreparedState.")
        self._record_large_n_eval_shape_diagnostics(prepared)

        @jax.jit
        def _eval(prepared_in: PreparedStateLike) -> Array:
            return jnp.asarray(
                evaluate_large_n_state(
                    self,
                    prepared_in,
                    target_indices=None,
                    return_potential=False,
                    max_acc_derivative_order=0,
                )
            )

        return prepared, _eval

    def _refresh_large_n_same_topology(
        self: "FastMultipoleMethod",
        prepared_state: LargeNPreparedState,
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]],
        leaf_size: int,
        max_order: int,
        theta: Optional[float],
        runtime_overrides_override: Optional[_RuntimeExecutionOverrides] = None,
        fused_device_mode: bool = False,
    ) -> Optional[LargeNPreparedState]:
        """Refresh large-N numeric payloads when the radix topology is unchanged."""

        self._large_n_same_topology_refresh_attempts += 1
        if not isinstance(prepared_state.tree, RadixTree):
            self._large_n_same_topology_refresh_misses += 1
            self._large_n_same_topology_refresh_miss_no_key += 1
            return None

        refresh_timing_active = bool(
            getattr(self, "_refresh_timing_active", False)
        ) and not (
            bool(fused_device_mode)
            and bool(getattr(self, "_strict_fused_disable_hot_timing", False))
        )

        input_t0 = time.perf_counter() if refresh_timing_active else 0.0
        positions_arr, masses_arr, input_dtype = self._prepare_state_input_arrays(
            positions,
            masses,
        )
        if refresh_timing_active:
            self._refresh_timing_input_seconds += time.perf_counter() - input_t0
        traced_refresh = bool(_contains_tracer((positions_arr, masses_arr)))
        allow_stateful_cache = bool(fused_device_mode) or (not traced_refresh)
        if (not allow_stateful_cache) and (not bool(fused_device_mode)):
            self._large_n_same_topology_refresh_misses += 1
            self._large_n_same_topology_refresh_miss_traced += 1
            return None

        self._validate_prepare_state_request(
            leaf_size=int(leaf_size),
            max_order=int(max_order),
        )
        runtime_overrides = runtime_overrides_override
        if runtime_overrides is None:
            runtime_overrides = self._resolve_runtime_execution_overrides(
                num_particles=int(positions_arr.shape[0]),
            )
        runtime_traversal_config = runtime_overrides.traversal_config
        runtime_m2l_chunk_size = runtime_overrides.m2l_chunk_size
        runtime_l2l_chunk_size = runtime_overrides.l2l_chunk_size
        upward_center_mode = runtime_overrides.center_mode
        refine_local_val = self.refine_local
        if runtime_overrides.refine_local_override is not None:
            refine_local_val = bool(runtime_overrides.refine_local_override)
        max_refine_levels_val = self.max_refine_levels
        aspect_threshold_val = self.aspect_threshold
        theta_val = float(self.theta if theta is None else theta)
        mac_type_val = self._base_mac_type()

        tree_config = self.config.tree
        if self.tree_type != "radix" and tree_config.mode in (
            "fixed_depth",
            "static_radix",
        ):
            tree_config = TreeBuilderConfig(
                mode="lbvh",
                target_leaf_particles=tree_config.target_leaf_particles,
                refine_local=tree_config.refine_local,
                max_refine_levels=tree_config.max_refine_levels,
                aspect_threshold=tree_config.aspect_threshold,
            )
        static_fused_refresh = bool(fused_device_mode) and (
            str(tree_config.mode).strip().lower() == "static_radix"
        )
        inferred_bounds = self._resolve_prepare_state_bounds(
            positions=positions_arr,
            bounds=bounds,
        )

        tree_t0 = time.perf_counter() if refresh_timing_active else 0.0
        refresh_topology_key = getattr(prepared_state, "topology_key", None)
        topology_candidate = None

        if static_fused_refresh:
            build_artifacts = self._rebuild_tree_artifacts_from_static_template(
                template_tree=prepared_state.tree,
                positions=positions_arr,
                masses=masses_arr,
                bounds=inferred_bounds,
                max_leaf_size=int(prepared_state.max_leaf_size),
                cache_leaf_parameter=int(leaf_size),
            )
            if refresh_topology_key is None:
                refresh_topology_key = "static_fused_template"
        else:
            previous_topology_key = refresh_topology_key
            if previous_topology_key is None:
                if tree_config.mode == "static_radix" and isinstance(
                    prepared_state.tree, RadixTree
                ):
                    previous_topology_key = self._static_radix_topology_key_from_tree(
                        prepared_state.tree,
                        leaf_size=int(leaf_size),
                    )
                else:
                    previous_codes = getattr(prepared_state.tree, "morton_codes", None)
                    if previous_codes is not None:
                        previous_topology_key = (
                            self._topology_reuse_key_from_sorted_codes(
                                sorted_codes=jnp.asarray(previous_codes),
                                tree_config=tree_config,
                                leaf_size=int(leaf_size),
                                refine_local=refine_local_val,
                                max_refine_levels=max_refine_levels_val,
                                aspect_threshold=aspect_threshold_val,
                            )
                        )
            if previous_topology_key is None:
                self._large_n_same_topology_refresh_misses += 1
                self._large_n_same_topology_refresh_miss_no_key += 1
                if tree_config.mode == "static_radix":
                    self._static_radix_refresh_misses += 1
                return None

            topology_candidate = self._topology_reuse_candidate(
                positions=positions_arr,
                bounds=inferred_bounds,
                tree_config=tree_config,
                leaf_size=int(leaf_size),
                refine_local=refine_local_val,
                max_refine_levels=max_refine_levels_val,
                aspect_threshold=aspect_threshold_val,
                allow_stateful_cache=allow_stateful_cache,
            )
            if (
                topology_candidate is None
                or topology_candidate.key != previous_topology_key
            ):
                self._large_n_same_topology_refresh_misses += 1
                self._large_n_same_topology_refresh_miss_topology += 1
                if tree_config.mode == "static_radix":
                    self._static_radix_refresh_misses += 1
                return None

            topology_entry = _TopologyReuseEntry(
                key=str(previous_topology_key),
                tree=prepared_state.tree,
                max_leaf_size=int(prepared_state.max_leaf_size),
                cache_leaf_parameter=int(leaf_size),
                reuse_count=0,
            )
            build_artifacts = self._rebuild_tree_artifacts_from_topology(
                candidate=topology_candidate,
                entry=topology_entry,
                positions=positions_arr,
                masses=masses_arr,
            )
            refresh_topology_key = topology_candidate.key

        strict_refresh_diag_mode = str(
            getattr(self, "_strict_refresh_diag_mode", "full")
        )
        strict_refresh_detail_diag_mode = str(
            getattr(self, "_strict_refresh_detail_diag_mode", "full")
        )
        strict_refresh_tree_detail_only = strict_refresh_detail_diag_mode in {
            "tree_sort_only",
            "tree_metadata_only",
        }
        strict_refresh_upward_detail_only = strict_refresh_detail_diag_mode in {
            "p2m_only",
            "m2m_only",
        }
        if bool(static_fused_refresh) and (
            strict_refresh_diag_mode == "tree_only" or strict_refresh_tree_detail_only
        ):
            self._large_n_same_topology_refresh_hits += 1
            if tree_config.mode == "static_radix":
                self._static_radix_refresh_hits += 1
            return replace(
                prepared_state,
                tree=build_artifacts.tree,
                topology_key=refresh_topology_key,
            )

        defer_geometry = False
        upward = self.prepare_upward_sweep(
            build_artifacts.tree,
            build_artifacts.positions_sorted,
            build_artifacts.masses_sorted,
            max_order=int(max_order),
            center_mode=upward_center_mode,
            max_leaf_size=int(build_artifacts.max_leaf_size),
            defer_geometry=defer_geometry,
        )
        locals_template = self._build_locals_template_for_prepare_state(
            tree=build_artifacts.tree,
            upward=upward,
            max_order=int(max_order),
            pos_sorted=build_artifacts.positions_sorted,
        )
        tree_artifacts = _PrepareStateTreeUpwardArtifacts(
            tree_mode=tree_config.mode,
            tree=build_artifacts.tree,
            positions_sorted=build_artifacts.positions_sorted,
            masses_sorted=build_artifacts.masses_sorted,
            inverse_permutation=build_artifacts.inverse_permutation,
            leaf_cap=int(build_artifacts.max_leaf_size),
            leaf_parameter=int(build_artifacts.cache_leaf_parameter),
            topology_key=refresh_topology_key,
            upward=upward,
            locals_template=locals_template,
        )
        if refresh_timing_active:
            self._refresh_timing_tree_upward_seconds += time.perf_counter() - tree_t0

        if bool(static_fused_refresh) and (
            strict_refresh_diag_mode == "upward_only"
            or strict_refresh_upward_detail_only
        ):
            dep = jnp.asarray(0.0, dtype=tree_artifacts.positions_sorted.dtype)
            multipoles = getattr(tree_artifacts.upward, "multipoles", None)
            packed = getattr(multipoles, "packed", None)
            centers = getattr(multipoles, "centers", None)
            if packed is not None:
                dep = dep + jnp.asarray(
                    jnp.real(jnp.sum(jnp.asarray(packed))),
                    dtype=dep.dtype,
                ) * jnp.asarray(0.0, dtype=dep.dtype)
            if centers is not None:
                dep = dep + jnp.asarray(
                    jnp.sum(jnp.asarray(centers)),
                    dtype=dep.dtype,
                ) * jnp.asarray(0.0, dtype=dep.dtype)
            diag_tree = replace(
                tree_artifacts.tree,
                positions_sorted=tree_artifacts.positions_sorted + dep,
            )
            self._large_n_same_topology_refresh_hits += 1
            if tree_config.mode == "static_radix":
                self._static_radix_refresh_hits += 1
            return replace(
                prepared_state,
                tree=diag_tree,
                topology_key=refresh_topology_key,
            )

        collected_retries: list[DualTreeRetryEvent] = []

        def record_retry(event: DualTreeRetryEvent) -> None:
            collected_retries.append(event)
            if self.interaction_retry_logger is not None:
                self.interaction_retry_logger(event)

        dual_t0 = time.perf_counter() if refresh_timing_active else 0.0
        strict_fused_traced_hot_path = bool(fused_device_mode) and bool(
            getattr(self, "_strict_fused_mode_active", False)
        )
        cached_compact_far_pairs = getattr(prepared_state, "compact_far_pairs", None)
        compact_far_pairs_carry_placeholder = cached_compact_far_pairs
        reuse_static_compact_pairs_enabled = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_REUSE_COMPACT_PAIRS",
                "1",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        allow_unsafe_compact_pair_reuse = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE",
                "0",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        safe_fresh_compact_pair_rebuild = (
            bool(strict_fused_traced_hot_path)
            and str(tree_config.mode).strip().lower() == "static_radix"
            and str(
                os.environ.get(
                    "JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD",
                    "1",
                )
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            and not bool(allow_unsafe_compact_pair_reuse)
        )
        reuse_static_compact_pairs = (
            bool(strict_fused_traced_hot_path)
            and bool(cached_compact_far_pairs is not None)
            and str(tree_config.mode).strip().lower() == "static_radix"
            and bool(reuse_static_compact_pairs_enabled)
            and bool(allow_unsafe_compact_pair_reuse)
        )
        if (
            bool(strict_fused_traced_hot_path)
            and bool(cached_compact_far_pairs is not None)
            and str(tree_config.mode).strip().lower() == "static_radix"
            and bool(reuse_static_compact_pairs_enabled)
            and not bool(allow_unsafe_compact_pair_reuse)
            and not bool(safe_fresh_compact_pair_rebuild)
        ):
            raise RuntimeError(
                "strict fused compact far-pair reuse is unsafe for moved "
                "static-radix positions: cached M2L pairs can change after "
                "the drift and corrupt endpoint forces. A production fix needs "
                "fresh fixed-cap compact pairs with an active mask/count, or a "
                "proven far-pair validity key. Set "
                "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE=1 "
                "only for legacy performance experiments."
            )
        if bool(safe_fresh_compact_pair_rebuild):
            cached_compact_far_pairs = None
        if bool(strict_fused_traced_hot_path) and (
            str(tree_config.mode).strip().lower() == "static_radix"
        ):
            if reuse_static_compact_pairs:
                self._static_radix_compact_pair_reuse_hits += 1
            else:
                self._static_radix_compact_pair_reuse_misses += 1
        if reuse_static_compact_pairs:
            src_far = jnp.asarray(cached_compact_far_pairs.sources, dtype=INDEX_DTYPE)
            tgt_far = jnp.asarray(cached_compact_far_pairs.targets, dtype=INDEX_DTYPE)
            far_pairs_by_gear = ((src_far, tgt_far),)
            downward = self._prepare_downward_with_artifacts(
                tree=tree_artifacts.tree,
                upward=tree_artifacts.upward,
                theta_val=theta_val,
                locals_template=tree_artifacts.locals_template,
                interactions=None,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                runtime_traversal_config=runtime_traversal_config,
                record_retry=record_retry,
                dense_buffers=None,
                grouped_interactions=False,
                grouped_buffers=None,
                grouped_segment_starts=None,
                grouped_segment_lengths=None,
                grouped_segment_class_ids=None,
                grouped_segment_sort_permutation=None,
                grouped_segment_group_ids=None,
                grouped_segment_unique_targets=None,
                farfield_mode="pair_grouped",
                far_pairs_coo=_FarPairCOO(
                    sources=src_far,
                    targets=tgt_far,
                    active_count=getattr(
                        cached_compact_far_pairs, "far_pair_count", None
                    ),
                ),
                far_pairs_by_gear=far_pairs_by_gear,
                adaptive_order=True,
                p_gears=(int(tree_artifacts.upward.multipoles.order),),
            )
            downward = downward._replace(
                interactions=_empty_interaction_storage_for_tree(tree_artifacts.tree)
            )
            dual_downward_artifacts = _PrepareStateDualDownwardArtifacts(
                interactions=None,
                neighbor_list=prepared_state.neighbor_list,
                traversal_result=None,
                compact_far_pairs=cached_compact_far_pairs,
                downward=downward,
                cache_entry=None,
            )
        else:
            dual_downward_artifacts = self._prepare_state_dual_and_downward(
                tree_artifacts=tree_artifacts,
                force_scale_nodes=prepared_state.force_scale_nodes,
                upward_center_mode=upward_center_mode,
                theta_val=theta_val,
                mac_type_val=mac_type_val,
                dehnen_radius_scale=self.dehnen_radius_scale,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                grouped_interactions=False,
                farfield_mode="pair_grouped",
                record_retry=record_retry,
                refine_local_val=refine_local_val,
                max_refine_levels_val=max_refine_levels_val,
                aspect_threshold_val=aspect_threshold_val,
                allow_stateful_cache=True,
                suppress_host_side_effects=strict_fused_traced_hot_path,
            )
        if str(tree_config.mode).strip().lower() == "static_radix":
            tree_now = build_artifacts.tree
            leaf_codes = getattr(tree_now, "leaf_codes", None)
            parent = getattr(tree_now, "parent", None)
            left_child = getattr(tree_now, "left_child", None)
            compact_pairs = getattr(dual_downward_artifacts, "compact_far_pairs", None)
            compact_sources = (
                getattr(compact_pairs, "sources", None)
                if compact_pairs is not None
                else None
            )
            far_pair_count = (
                int(getattr(compact_sources, "shape", (0,))[0])
                if compact_sources is not None
                else int(getattr(self, "_recent_dual_far_pair_count", 0))
            )
            chunk_size = (
                4096 if runtime_m2l_chunk_size is None else int(runtime_m2l_chunk_size)
            )
            self._static_radix_tree_leaf_count = (
                int(getattr(leaf_codes, "shape", (0,))[0])
                if leaf_codes is not None
                else int(getattr(self, "_recent_dual_leaf_count", 0))
            )
            self._static_radix_tree_node_count = (
                int(getattr(parent, "shape", (0,))[0])
                if parent is not None
                else int(getattr(self, "_recent_dual_node_count", 0))
            )
            self._static_radix_far_pair_count = int(far_pair_count)
            self._static_radix_m2l_chunk_count = (
                0
                if chunk_size <= 0 or far_pair_count <= 0
                else int((far_pair_count + chunk_size - 1) // chunk_size)
            )
            self._static_radix_l2l_edge_count = (
                2 * int(getattr(left_child, "shape", (0,))[0])
                if left_child is not None
                else 0
            )

        if refresh_timing_active:
            elapsed = time.perf_counter() - dual_t0
            recorded = float(
                getattr(self, "_refresh_timing_dual_downward_seconds", 0.0)
            )
            # _prepare_state_dual_and_downward records detailed timing itself.
            # Keep this branch intentionally empty except to make the elapsed
            # value visible while avoiding double accounting.
            _ = (elapsed, recorded)

        if (
            tree_config.mode != "static_radix"
            and not self._large_n_neighbor_list_matches(
                prepared_state.neighbor_list,
                dual_downward_artifacts.neighbor_list,
            )
        ):
            self._large_n_same_topology_refresh_misses += 1
            self._large_n_same_topology_refresh_miss_neighbor += 1
            return None

        self._large_n_same_topology_refresh_hits += 1
        if tree_config.mode == "static_radix":
            self._static_radix_refresh_hits += 1

        if allow_stateful_cache and (not traced_refresh):
            self._update_locals_template_cache_after_prepare(
                locals_template=tree_artifacts.locals_template,
                upward=tree_artifacts.upward,
                max_order=int(max_order),
            )
            self._recent_retry_events = tuple(collected_retries)
            self._record_strict_cap_profile_from_retries(
                self._recent_retry_events,
                context_key=self._strict_cap_profile_context_key(
                    tree_mode=str(tree_artifacts.tree_mode),
                    leaf_parameter=int(tree_artifacts.leaf_parameter),
                    particle_count=int(
                        jnp.asarray(tree_artifacts.positions_sorted).shape[0]
                    ),
                ),
            )
            self._topology_reuse_entry = _TopologyReuseEntry(
                key=str(refresh_topology_key),
                tree=tree_artifacts.tree,
                max_leaf_size=int(tree_artifacts.leaf_cap),
                cache_leaf_parameter=int(tree_artifacts.leaf_parameter),
                reuse_count=0,
            )

        refreshed_state = prepare_large_n_state(
            self,
            positions_arr=positions_arr,
            masses_arr=masses_arr,
            input_dtype=input_dtype,
            bounds=bounds,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            theta_val=theta_val,
            mac_type_val=mac_type_val,
            refine_local_val=refine_local_val,
            max_refine_levels_val=max_refine_levels_val,
            aspect_threshold_val=aspect_threshold_val,
            jit_tree_override=None,
            allow_stateful_cache=allow_stateful_cache,
            runtime_traversal_config=runtime_traversal_config,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            upward_center_mode=upward_center_mode,
            record_retry=record_retry,
            collected_retries=collected_retries,
            tree_artifacts=tree_artifacts,
            dual_downward_artifacts=dual_downward_artifacts,
            fused_device_mode=bool(fused_device_mode),
        )
        if bool(safe_fresh_compact_pair_rebuild):
            return replace(
                refreshed_state,
                compact_far_pairs=compact_far_pairs_carry_placeholder,
            )
        return refreshed_state

    def _large_n_neighbor_list_matches(
        self: "FastMultipoleMethod",
        previous: NodeNeighborList,
        current: NodeNeighborList,
    ) -> bool:
        """Return True when current active neighbor edges match previous state."""

        try:
            prev_offsets = np.asarray(jax.device_get(previous.offsets))
            cur_offsets = np.asarray(jax.device_get(current.offsets))
            prev_counts = np.asarray(jax.device_get(previous.counts))
            cur_counts = np.asarray(jax.device_get(current.counts))
            prev_leaf = np.asarray(jax.device_get(previous.leaf_indices))
            cur_leaf = np.asarray(jax.device_get(current.leaf_indices))
            if (
                prev_offsets.shape != cur_offsets.shape
                or prev_counts.shape != cur_counts.shape
                or prev_leaf.shape != cur_leaf.shape
            ):
                return False
            if (
                not np.array_equal(prev_offsets, cur_offsets)
                or not np.array_equal(prev_counts, cur_counts)
                or not np.array_equal(prev_leaf, cur_leaf)
            ):
                return False
            active_edges = int(cur_offsets[-1]) if cur_offsets.size > 0 else 0
            prev_neighbors = np.asarray(jax.device_get(previous.neighbors))
            cur_neighbors = np.asarray(jax.device_get(current.neighbors))
            if int(prev_neighbors.shape[0]) < active_edges:
                return False
            if int(cur_neighbors.shape[0]) < active_edges:
                return False
            return bool(
                np.array_equal(
                    prev_neighbors[:active_edges],
                    cur_neighbors[:active_edges],
                )
            )
        except Exception:
            return False

    def update_multipoles_only(
        self: "FastMultipoleMethod",
        prepared_state: PreparedStateLike,
        positions: Array,
        masses: Array,
        *,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
    ) -> PreparedStateLike:
        """Refresh multipole/local payloads when topology key remains unchanged."""
        if not self._is_large_n_gpu_production_profile():
            raise NotImplementedError(
                "update_multipoles_only is currently supported only for "
                "preset='large_n_gpu', tree_type='radix', expansion_basis='solidfmm'."
            )
        if not isinstance(prepared_state, LargeNPreparedState):
            raise NotImplementedError(
                "update_multipoles_only currently supports LargeNPreparedState only."
            )
        self._compiled_profile_multipoles_only_calls += 1
        refreshed = self.refresh_prepared_state(
            prepared_state,
            positions,
            masses,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
        )
        if getattr(refreshed, "topology_key", None) != getattr(
            prepared_state, "topology_key", None
        ):
            raise RuntimeError(
                "Topology changed during update_multipoles_only; "
                "use rebuild_topology_in_place for topology updates."
            )
        return refreshed

    def rebuild_topology_in_place(
        self: "FastMultipoleMethod",
        prepared_state: PreparedStateLike,
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
    ) -> PreparedStateLike:
        """Rebuild topology while attempting to remain profile-capacity compatible."""
        if not self._is_large_n_gpu_production_profile():
            raise NotImplementedError(
                "rebuild_topology_in_place is currently supported only for "
                "preset='large_n_gpu', tree_type='radix', expansion_basis='solidfmm'."
            )
        if not isinstance(prepared_state, LargeNPreparedState):
            raise NotImplementedError(
                "rebuild_topology_in_place currently supports LargeNPreparedState only."
            )
        self._compiled_profile_topology_rebuild_calls += 1
        return self.refresh_prepared_state(
            prepared_state,
            positions,
            masses,
            bounds=bounds,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
        )

    def export_m2l_autotune_cache(self: "FastMultipoleMethod") -> list[dict[str, Any]]:
        """Return a JSON-serializable snapshot of global M2L autotune results."""

        return _m2l_autotune_payload()

    def import_m2l_autotune_cache(
        self: "FastMultipoleMethod",
        payload: list[dict[str, Any]],
        *,
        merge: bool = True,
    ) -> int:
        """Restore global M2L autotune results from serialized payload."""

        return _restore_m2l_autotune_payload(payload, merge=bool(merge))

    def save_m2l_autotune_cache(self: "FastMultipoleMethod", path: str) -> int:
        """Write global M2L autotune cache to a JSON file."""

        payload = self.export_m2l_autotune_cache()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return int(len(payload))

    def load_m2l_autotune_cache(
        self: "FastMultipoleMethod",
        path: str,
        *,
        merge: bool = True,
    ) -> int:
        """Load global M2L autotune cache from a JSON file."""

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("autotune cache JSON must contain a list payload")
        return self.import_m2l_autotune_cache(payload, merge=bool(merge))

    def _solidfmm_basis_mode(self: "FastMultipoleMethod") -> str:
        """Return active solidfmm coefficient family ('complex' or 'real')."""
        basis_obj = self.basis_impl
        name = str(getattr(basis_obj, "name", "")).strip().lower()
        if name == "real":
            return "real"
        return "complex"

    def _compute_node_force_scale_from_sorted_acc(
        self: "FastMultipoleMethod",
        *,
        tree: Tree,
        accelerations_sorted: Array,
        reduction: str = "max",
    ) -> Array:
        """Estimate per-node force scales from sorted per-particle accelerations."""

        return compute_node_force_scale_from_sorted_acc(
            tree=tree,
            accelerations_sorted=accelerations_sorted,
            reduction=reduction,
        )

    def _source_error_proxy_by_order_from_multipoles(
        self: "FastMultipoleMethod",
        *,
        multipole_packed: Array,
        p_gears: tuple[int, ...],
    ) -> Array:
        """Compute a conservative per-node residual proxy for each candidate order."""

        return source_error_proxy_by_order_from_multipoles(
            multipole_packed=multipole_packed,
            p_gears=p_gears,
        )

    def _adaptive_error_model_code(self: "FastMultipoleMethod") -> int:
        """Return the integer policy code for the active adaptive error model."""

        if self.adaptive_error_model == "dehnen_paper":
            return 2
        if self.adaptive_error_model == "dehnen_degree":
            return 1
        return 0

    def _uses_dehnen_error_policy(self: "FastMultipoleMethod") -> bool:
        """Return whether traversal should use the Dehnen paper policy hook."""

        return str(self.mac_type) == "dehnen_error"

    def _uses_dehnen_paper_error_model(self: "FastMultipoleMethod") -> bool:
        """Return whether the active adaptive error model is the paper estimator."""

        return self.adaptive_error_model == "dehnen_paper"

    def _uses_paper_style_traversal_policy(self: "FastMultipoleMethod") -> bool:
        """Return whether traversal should use the paper-style error policy."""

        return self._uses_dehnen_paper_error_model() or self._uses_dehnen_error_policy()

    def _traversal_policy_error_model_code(self: "FastMultipoleMethod") -> int:
        """Return the policy error model code used during traversal."""

        if self._uses_dehnen_error_policy():
            return 2
        return self._adaptive_error_model_code()

    def _force_scale_reduction_mode(self: "FastMultipoleMethod") -> str:
        """Return the node reduction mode used for adaptive force scales."""

        return "min" if self._uses_dehnen_paper_error_model() else "max"

    def _uses_paper_style_force_scale(self: "FastMultipoleMethod") -> bool:
        """Return whether prepare_state needs paper-style force-scale handling."""

        return self.adaptive_order or self._uses_paper_style_traversal_policy()

    def _base_mac_type(self: "FastMultipoleMethod") -> MACType:
        """Return the Yggdrax-facing geometric MAC for the active solver mode."""

        return "dehnen" if self._uses_dehnen_error_policy() else self.mac_type

    def _policy_orders_for_prepare_state(
        self: "FastMultipoleMethod", *, max_order: int
    ) -> tuple[int, ...]:
        """Return candidate orders used to build adaptive traversal policy state."""

        if (not self.adaptive_order) and self._uses_dehnen_paper_error_model():
            return (int(max_order),)
        return self.p_gears

    def _build_adaptive_policy_state(
        self: "FastMultipoleMethod",
        *,
        upward: TreeUpwardData,
        tree: Tree,
        positions_sorted: Array,
        p_gears: tuple[int, ...],
        force_scale_nodes: Optional[Array],
        eps: Array,
        theta: Array,
        error_model_code: Array,
        dehnen_geometry_mode: str,
    ):
        """Build the solver-owned adaptive policy state from upward data."""

        return build_adaptive_policy_state(
            upward=upward,
            tree=tree,
            positions_sorted=positions_sorted,
            p_gears=p_gears,
            force_scale_nodes=force_scale_nodes,
            eps=eps,
            theta=theta,
            error_model_code=error_model_code,
            dehnen_geometry_mode=dehnen_geometry_mode,
        )

    def _compute_force_scale_paper_prepass_from_tree_artifacts(
        self: "FastMultipoleMethod",
        *,
        tree_artifacts: _PrepareStateTreeUpwardArtifacts,
        low_order: int,
        theta_val: float,
        upward_center_mode: str,
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        grouped_interactions: bool,
        farfield_mode: str,
        record_retry: Callable[[DualTreeRetryEvent], None],
        refine_local_val: bool,
        max_refine_levels_val: int,
        aspect_threshold_val: float,
    ) -> Array:
        """Compute paper-mode force scales via a low-order prepass on the current tree."""

        low_upward = self.prepare_upward_sweep(
            tree_artifacts.tree,
            tree_artifacts.positions_sorted,
            tree_artifacts.masses_sorted,
            max_order=int(low_order),
            center_mode=upward_center_mode,
            max_leaf_size=tree_artifacts.leaf_cap,
        )
        low_locals_template = self._build_locals_template_for_prepare_state(
            tree=tree_artifacts.tree,
            upward=low_upward,
            max_order=int(low_order),
            pos_sorted=tree_artifacts.positions_sorted,
        )
        low_tree_artifacts = _PrepareStateTreeUpwardArtifacts(
            tree_mode=tree_artifacts.tree_mode,
            tree=tree_artifacts.tree,
            positions_sorted=tree_artifacts.positions_sorted,
            masses_sorted=tree_artifacts.masses_sorted,
            inverse_permutation=tree_artifacts.inverse_permutation,
            leaf_cap=tree_artifacts.leaf_cap,
            leaf_parameter=tree_artifacts.leaf_parameter,
            topology_key=tree_artifacts.topology_key,
            upward=low_upward,
            locals_template=low_locals_template,
        )

        saved_p_gears = self.p_gears
        saved_adaptive_order = self.adaptive_order
        saved_adaptive_error_model = self.adaptive_error_model
        saved_mac_type = self.mac_type
        saved_recent_counts = self._recent_far_pairs_by_gear_counts
        self._in_force_scale_prepass = True
        try:
            self.p_gears = (int(low_order),)
            self.adaptive_order = False
            self.adaptive_error_model = "tail_proxy"
            self.mac_type = "dehnen"
            dual_downward_artifacts = self._prepare_state_dual_and_downward(
                tree_artifacts=low_tree_artifacts,
                force_scale_nodes=None,
                upward_center_mode=upward_center_mode,
                theta_val=theta_val,
                mac_type_val=self.mac_type,
                dehnen_radius_scale=self.dehnen_radius_scale,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                grouped_interactions=grouped_interactions,
                farfield_mode=farfield_mode,
                record_retry=record_retry,
                refine_local_val=refine_local_val,
                max_refine_levels_val=max_refine_levels_val,
                aspect_threshold_val=aspect_threshold_val,
                allow_stateful_cache=False,
            )
            prepass_execution_backend = self._resolve_execution_backend()
            if prepass_execution_backend == "octree":
                prepass_octree, prepass_octree_native = (
                    build_octree_execution_data_with_status(low_tree_artifacts.tree)
                )
            else:
                prepass_octree, prepass_octree_native = None, False
            # See the main prepared-state path: only build native-octree interaction
            # lists when the octree view is non-degenerate; otherwise far/near come from
            # the consistent compat lists on the fallback (binary) tree.
            prepass_octree_native_neighbors = None
            if (
                prepass_execution_backend == "octree"
                and prepass_octree is not None
                and prepass_octree_native
            ):
                prepass_octree_native_neighbors = build_octree_native_neighbor_lists(
                    low_tree_artifacts.tree,
                    low_tree_artifacts.upward.geometry,
                    theta=theta_val,
                    mac_type=self.mac_type,
                    dehnen_radius_scale=self.dehnen_radius_scale,
                    max_pair_queue=self.max_pair_queue,
                    process_block=self.pair_process_block,
                    traversal_config=runtime_traversal_config,
                )
            prepass_nearfield_interop = _build_nearfield_interop_data(
                low_tree_artifacts.tree,
                dual_downward_artifacts.neighbor_list,
                octree=prepass_octree,
                native_neighbors=prepass_octree_native_neighbors,
            )
            nearfield_artifacts = self._prepare_state_nearfield_artifacts(
                neighbor_list=dual_downward_artifacts.neighbor_list,
                nearfield_interop=prepass_nearfield_interop,
                leaf_cap=low_tree_artifacts.leaf_cap,
                num_particles=int(low_tree_artifacts.positions_sorted.shape[0]),
                cache_entry=dual_downward_artifacts.cache_entry,
                allow_stateful_cache=False,
            )
            prepass_octree_upward = _build_octree_upward_artifacts(
                octree=prepass_octree,
                positions_sorted=low_tree_artifacts.positions_sorted,
                masses_sorted=low_tree_artifacts.masses_sorted,
                expansion_basis=self.expansion_basis,
                max_order=int(low_order),
            )
            prepass_octree_native_far_pairs = None
            if (
                prepass_execution_backend == "octree"
                and prepass_octree is not None
                and prepass_octree_native
            ):
                prepass_octree_native_far_pairs = build_octree_native_far_pairs(
                    low_tree_artifacts.tree,
                    low_tree_artifacts.upward.geometry,
                    theta=theta_val,
                    mac_type=self.mac_type,
                    dehnen_radius_scale=self.dehnen_radius_scale,
                    max_pair_queue=self.max_pair_queue,
                    process_block=self.pair_process_block,
                    traversal_config=runtime_traversal_config,
                )
            prepass_octree_downward = _build_octree_downward_artifacts(
                octree=prepass_octree,
                octree_upward=prepass_octree_upward,
                interactions=dual_downward_artifacts.interactions,
                native_far_pairs=prepass_octree_native_far_pairs,
                execution_backend=prepass_execution_backend,
            )
            prepass_state = FMMPreparedState(
                tree=low_tree_artifacts.tree,
                upward=_prepared_state_upward_payload(
                    upward=low_tree_artifacts.upward,
                    memory_objective=self.memory_objective,
                ),
                downward=dual_downward_artifacts.downward,
                neighbor_list=dual_downward_artifacts.neighbor_list,
                max_leaf_size=low_tree_artifacts.leaf_cap,
                input_dtype=low_tree_artifacts.positions_sorted.dtype,
                working_dtype=low_tree_artifacts.positions_sorted.dtype,
                expansion_basis=self.expansion_basis,
                theta=theta_val,
                topology_key=low_tree_artifacts.topology_key,
                interactions=dual_downward_artifacts.interactions,
                dual_tree_result=dual_downward_artifacts.traversal_result,
                retry_events=tuple(),
                nearfield_interop=prepass_nearfield_interop,
                nearfield_target_leaf_ids=nearfield_artifacts.target_leaf_ids,
                nearfield_source_leaf_ids=nearfield_artifacts.source_leaf_ids,
                nearfield_valid_pairs=nearfield_artifacts.valid_pairs,
                nearfield_chunk_sort_indices=nearfield_artifacts.chunk_sort_indices,
                nearfield_chunk_group_ids=nearfield_artifacts.chunk_group_ids,
                nearfield_chunk_unique_indices=nearfield_artifacts.chunk_unique_indices,
                force_scale_nodes=None,
                execution_backend=prepass_execution_backend,
                octree=prepass_octree,
                octree_upward=_prepared_state_octree_upward_payload(
                    octree_upward=prepass_octree_upward,
                    memory_objective=self.memory_objective,
                ),
                octree_downward=_finalize_octree_downward_artifacts(
                    octree=prepass_octree,
                    octree_upward=prepass_octree_upward,
                    octree_downward=prepass_octree_downward,
                    expansion_basis=self.expansion_basis,
                    execution_backend=prepass_execution_backend,
                    m2l_chunk_size=runtime_m2l_chunk_size,
                ),
            )
            prepass_acc = self.evaluate_prepared_state(
                prepass_state,
                return_potential=False,
                jit_traversal=False,
            )
        finally:
            self.p_gears = saved_p_gears
            self.adaptive_order = saved_adaptive_order
            self.adaptive_error_model = saved_adaptive_error_model
            self.mac_type = saved_mac_type
            self._recent_far_pairs_by_gear_counts = saved_recent_counts
            self._in_force_scale_prepass = False

        sorted_idx = jnp.argsort(low_tree_artifacts.inverse_permutation)
        return jnp.asarray(prepass_acc)[sorted_idx]

    def _prepared_state_cache_lookup(
        self,
        *,
        key: tuple[Any, ...],
        positions: Array,
        masses: Array,
    ) -> Optional[PreparedStateLike]:
        """Return cached prepared state when key and inputs exactly match."""
        cached_key = self._prepared_state_cache_key
        cached_value = self._prepared_state_cache_value
        cached_positions = self._prepared_state_cache_positions
        cached_masses = self._prepared_state_cache_masses
        if cached_key is None or cached_value is None:
            return None
        if cached_positions is None or cached_masses is None:
            return None
        if cached_key != key:
            return None
        if (
            cached_positions.shape != positions.shape
            or cached_positions.dtype != positions.dtype
            or cached_masses.shape != masses.shape
            or cached_masses.dtype != masses.dtype
        ):
            return None
        if not bool(jnp.array_equal(positions, cached_positions)):
            return None
        if not bool(jnp.array_equal(masses, cached_masses)):
            return None
        return cached_value

    def _prepared_state_cache_store(
        self,
        *,
        key: tuple[Any, ...],
        positions: Array,
        masses: Array,
        state: PreparedStateLike,
    ) -> None:
        """Store prepared-state payload and the exact input arrays used."""
        if _contains_tracer((positions, masses, state)):
            return
        self._prepared_state_cache_key = key
        self._prepared_state_cache_value = state
        self._prepared_state_cache_positions = positions
        self._prepared_state_cache_masses = masses

    def _resolve_jit_tree_flag(
        self,
        positions: Array,
        *,
        jit_tree_override: Optional[bool],
    ) -> bool:
        """Resolve tree-build JIT mode with a CPU-friendly auto heuristic."""

        if self.tree_type != "radix":
            return False

        if jit_tree_override is not None:
            return bool(jit_tree_override)

        default_mode = self._jit_tree_default
        if default_mode != "auto":
            return bool(default_mode)

        backend = jax.default_backend()
        num_particles = int(jnp.asarray(positions).shape[0])
        # CPU tree build often performs better without JIT for small/medium N.
        if backend == "cpu" and num_particles <= 8192:
            return False
        return True

    def _resolve_runtime_execution_overrides(
        self,
        *,
        num_particles: int,
        backend: Optional[str] = None,
    ) -> _RuntimeExecutionOverrides:
        """Resolve adaptive runtime traversal/chunk settings."""

        traversal_config = self.traversal_config
        m2l_chunk_size = self.m2l_chunk_size
        l2l_chunk_size = self.l2l_chunk_size
        grouped_interactions = (
            False
            if self.grouped_interactions is None
            else bool(self.grouped_interactions)
        )
        farfield_mode = self.farfield_mode
        center_mode = "com"
        refine_local_override: Optional[bool] = None
        adaptive_applied = False

        backend_name = jax.default_backend() if backend is None else str(backend)
        n_particles = int(num_particles)
        production_large_n = self._is_large_n_gpu_production_profile()
        static_runtime_fixed_sizing = bool(
            getattr(self, "_static_runtime_fixed_sizing", True)
        )
        minimum_memory = self.memory_objective == "minimum_memory" or production_large_n
        large_cpu = (
            backend_name == "cpu" and n_particles >= _LARGE_CPU_PARTICLE_THRESHOLD
        )
        class_major_cpu = (
            backend_name == "cpu" and n_particles >= _CLASS_MAJOR_CPU_PARTICLE_THRESHOLD
        )
        class_major_gpu = (
            backend_name == "gpu" and n_particles >= _GPU_LARGE_PARTICLE_THRESHOLD
        )

        if self.host_refine_mode == "off":
            refine_local_override = False
        elif self.host_refine_mode == "on":
            refine_local_override = True
        elif (
            large_cpu
            and self.tree_type == "radix"
            and self.preset == "fast"
            and self.expansion_basis == "solidfmm"
            and self.mac_type == "dehnen"
        ):
            refine_local_override = False

        if (
            self.tree_type == "kdtree"
            and not self._explicit_traversal_config
            and not self._explicit_max_pair_queue
            and not self._explicit_pair_process_block
        ):
            traversal_config = _KDTREE_DEFAULT_TRAVERSAL_CONFIG

        if (
            not self._explicit_grouped_interactions
            and self.preset == "fast"
            and self.expansion_basis == "solidfmm"
            and self.mac_type == "dehnen"
            and self.tree_type == "radix"
            and large_cpu
            and not minimum_memory
        ):
            grouped_interactions = True
        if (
            not self._explicit_grouped_interactions
            and self.preset in ("fast", "large_n_gpu")
            and self.expansion_basis == "solidfmm"
            and self.mac_type == "dehnen"
            and self.tree_type == "radix"
            and backend_name == "gpu"
            and n_particles >= _GPU_LARGE_PARTICLE_THRESHOLD
            and not minimum_memory
        ):
            grouped_interactions = True

        if production_large_n:
            grouped_interactions = False
            farfield_mode = "pair_grouped"

        if static_runtime_fixed_sizing:
            # Static sizing mode: keep traversal/chunk execution knobs fixed to
            # constructor/global-input values and skip adaptive runtime rewrites.
            if self.streamed_far_pairs and grouped_interactions:
                grouped_interactions = False
                farfield_mode = "pair_grouped"
            if not grouped_interactions:
                farfield_mode = "pair_grouped"
            return _RuntimeExecutionOverrides(
                traversal_config=traversal_config,
                m2l_chunk_size=m2l_chunk_size,
                l2l_chunk_size=l2l_chunk_size,
                grouped_interactions=grouped_interactions,
                farfield_mode=farfield_mode,
                center_mode=center_mode,
                refine_local_override=refine_local_override,
                adaptive_applied=False,
            )

        if self.streamed_far_pairs and grouped_interactions:
            # Streamed far-pair execution and grouped/class-major M2L are
            # competing strategies. The grouped path overrides streaming in the
            # downward sweep, so keeping both enabled only pays the grouped
            # traversal/materialization cost while defeating the user's request
            # for streamed execution.
            grouped_interactions = False
            farfield_mode = "pair_grouped"

        if (
            self.preset == "fast"
            and self.expansion_basis == "solidfmm"
            and self.mac_type == "dehnen"
            and self.tree_type == "radix"
            and large_cpu
            and not self._explicit_traversal_config
            and not self._explicit_max_pair_queue
            and not self._explicit_pair_process_block
        ):
            traversal_config = _LARGE_CPU_TRAVERSAL_CONFIG
            adaptive_applied = True

            if not self._explicit_m2l_chunk_size:
                m2l_chunk_size = _LARGE_CPU_M2L_CHUNK_SIZE
            if not self._explicit_l2l_chunk_size:
                l2l_chunk_size = self.l2l_chunk_size
        if (
            backend_name == "gpu"
            and self.tree_type == "radix"
            and traversal_config is not None
            and not self._explicit_traversal_config
            and n_particles >= _GPU_LARGE_PARTICLE_THRESHOLD
        ):
            current_queue = int(traversal_config.max_pair_queue)
            current_block = int(traversal_config.process_block)
            current_interactions = int(traversal_config.max_interactions_per_node)
            current_neighbors = int(traversal_config.max_neighbors_per_leaf)

            if minimum_memory:
                target_queue = _GPU_MINIMUM_MEMORY_PAIR_QUEUE
                target_block = _GPU_MINIMUM_MEMORY_PROCESS_BLOCK
                target_interactions = _GPU_MINIMUM_MEMORY_INTERACTIONS_PER_NODE
                target_neighbors = _GPU_MINIMUM_MEMORY_NEIGHBORS_PER_LEAF
            elif n_particles >= 4_194_304:
                target_queue = _GPU_MIN_PAIR_QUEUE_XL
                target_block = current_block
                target_interactions = _GPU_MIN_INTERACTIONS_PER_NODE
                target_neighbors = _GPU_MIN_NEIGHBORS_PER_LEAF
            elif n_particles >= 1_048_576:
                target_queue = _GPU_MIN_PAIR_QUEUE_LARGE
                target_block = current_block
                target_interactions = _GPU_MIN_INTERACTIONS_PER_NODE
                target_neighbors = _GPU_MIN_NEIGHBORS_PER_LEAF
            else:
                target_queue = _GPU_MIN_PAIR_QUEUE_MEDIUM
                target_block = current_block
                target_interactions = _GPU_MIN_INTERACTIONS_PER_NODE
                target_neighbors = _GPU_MIN_NEIGHBORS_PER_LEAF

            if minimum_memory:
                next_queue = min(current_queue, int(target_queue))
                next_block = min(current_block, int(target_block))
                next_interactions = min(current_interactions, int(target_interactions))
                next_neighbors = min(current_neighbors, int(target_neighbors))
            else:
                next_queue = max(current_queue, int(target_queue))
                next_block = current_block
                next_interactions = min(
                    max(current_interactions, int(target_interactions)),
                    int(_GPU_MAX_INTERACTIONS_PER_NODE),
                )
                next_neighbors = min(
                    max(current_neighbors, int(target_neighbors)),
                    int(_GPU_MAX_NEIGHBORS_PER_LEAF),
                )
            if (
                next_queue != current_queue
                or next_block != current_block
                or next_interactions != current_interactions
                or next_neighbors != current_neighbors
            ):
                traversal_config = DualTreeTraversalConfig(
                    max_pair_queue=int(next_queue),
                    process_block=int(next_block),
                    max_interactions_per_node=int(next_interactions),
                    max_neighbors_per_leaf=int(next_neighbors),
                )
        if (
            minimum_memory
            and backend_name == "gpu"
            and self.tree_type == "radix"
            and not self._explicit_traversal_config
            and not self._explicit_max_pair_queue
            and not self._explicit_pair_process_block
        ):
            # The Yggdrax count-pass auto-sizing path is still too expensive on
            # large GPU radix trees. Keep the large-N minimum-memory route on a
            # bounded explicit traversal config so host-side retry can grow from
            # a safe baseline without compiling the count-pass kernel.
            traversal_config = _minimum_memory_streamed_gpu_traversal_seed(
                num_particles=n_particles
            )
        if (
            production_large_n
            and backend_name == "gpu"
            and traversal_config is not None
        ):
            # Production large-N radix path should not allow oversized explicit
            # traversal seeds to inflate memory footprint. Keep user overrides
            # only within the bounded streamed minimum-memory ceiling.
            explicit_ceiling = _minimum_memory_streamed_gpu_traversal_ceiling(
                num_particles=n_particles
            )
            traversal_config = DualTreeTraversalConfig(
                max_pair_queue=int(
                    min(
                        int(traversal_config.max_pair_queue),
                        int(explicit_ceiling.max_pair_queue),
                    )
                ),
                process_block=int(
                    min(
                        int(traversal_config.process_block),
                        int(explicit_ceiling.process_block),
                    )
                ),
                max_interactions_per_node=int(
                    min(
                        int(traversal_config.max_interactions_per_node),
                        int(explicit_ceiling.max_interactions_per_node),
                    )
                ),
                max_neighbors_per_leaf=int(
                    min(
                        int(traversal_config.max_neighbors_per_leaf),
                        int(explicit_ceiling.max_neighbors_per_leaf),
                    )
                ),
            )
        if (
            minimum_memory
            and backend_name == "gpu"
            and self.tree_type == "radix"
            and self.expansion_basis == "solidfmm"
            and bool(self.streamed_far_pairs)
            and not bool(grouped_interactions)
            and bool(self.fail_fast)
            and not self._explicit_traversal_config
            and not self._explicit_max_pair_queue
            and not self._explicit_pair_process_block
            and traversal_config is not None
            and n_particles >= 1_048_576
        ):
            explicit_ceiling = _minimum_memory_streamed_gpu_traversal_ceiling(
                num_particles=n_particles
            )
            capped_queue = min(
                int(traversal_config.max_pair_queue),
                int(explicit_ceiling.max_pair_queue),
            )
            capped_block = min(
                int(traversal_config.process_block),
                int(explicit_ceiling.process_block),
            )
            capped_interactions = min(
                int(traversal_config.max_interactions_per_node),
                int(explicit_ceiling.max_interactions_per_node),
            )
            capped_neighbors = min(
                int(traversal_config.max_neighbors_per_leaf),
                int(explicit_ceiling.max_neighbors_per_leaf),
            )
            traversal_config = DualTreeTraversalConfig(
                max_pair_queue=int(capped_queue),
                process_block=int(capped_block),
                max_interactions_per_node=int(capped_interactions),
                max_neighbors_per_leaf=int(capped_neighbors),
            )
        if grouped_interactions:
            center_mode = "aabb"
            if farfield_mode == "auto":
                if minimum_memory:
                    farfield_mode = "pair_grouped"
                else:
                    farfield_mode = (
                        "class_major"
                        if (class_major_cpu or class_major_gpu)
                        else "pair_grouped"
                    )
        else:
            farfield_mode = "pair_grouped"

        if minimum_memory and not self._explicit_m2l_chunk_size:
            m2l_chunk_size = (
                _MINIMUM_MEMORY_GPU_M2L_CHUNK_SIZE
                if backend_name == "gpu"
                else _MINIMUM_MEMORY_CPU_M2L_CHUNK_SIZE
            )

        return _RuntimeExecutionOverrides(
            traversal_config=traversal_config,
            m2l_chunk_size=m2l_chunk_size,
            l2l_chunk_size=l2l_chunk_size,
            grouped_interactions=grouped_interactions,
            farfield_mode=farfield_mode,
            center_mode=center_mode,
            refine_local_override=refine_local_override,
            adaptive_applied=adaptive_applied,
        )

    def _resolve_tracing_traversal_config(
        self,
        *,
        traversal_config: Optional[DualTreeTraversalConfig],
    ) -> Optional[DualTreeTraversalConfig]:
        """Clamp traced traversal capacities to avoid pathological padding.

        Applies only when prepare_state runs under tracing and the user did not
        explicitly provide traversal_config overrides.
        """

        if traversal_config is None or self._explicit_traversal_config:
            return traversal_config

        current_queue = int(traversal_config.max_pair_queue)
        capped_queue = min(current_queue, _TRACING_MAX_PAIR_QUEUE)
        current_block = int(traversal_config.process_block)
        capped_block = min(current_block, _TRACING_MAX_PROCESS_BLOCK)
        current_neighbors = int(traversal_config.max_neighbors_per_leaf)
        capped_neighbors = min(current_neighbors, _TRACING_MAX_NEIGHBORS_PER_LEAF)
        current_interactions = int(traversal_config.max_interactions_per_node)
        capped_interactions = min(
            current_interactions, _TRACING_MAX_INTERACTIONS_PER_NODE
        )
        if (
            capped_queue == current_queue
            and capped_block == current_block
            and capped_neighbors == current_neighbors
            and capped_interactions == current_interactions
        ):
            return traversal_config

        return DualTreeTraversalConfig(
            max_pair_queue=int(capped_queue),
            process_block=int(capped_block),
            max_interactions_per_node=int(capped_interactions),
            max_neighbors_per_leaf=int(capped_neighbors),
        )

    def _select_autotune_m2l_candidates(self, *, pair_count: int) -> tuple[int, ...]:
        """Return candidate chunk sizes for one pair-count regime."""

        pairs = int(pair_count)
        if pairs < _GPU_M2L_AUTOTUNE_PAIR_BINS[0]:
            return _GPU_M2L_AUTOTUNE_SMALL_CANDIDATES
        if pairs < _GPU_M2L_AUTOTUNE_PAIR_BINS[1]:
            return _GPU_M2L_AUTOTUNE_MEDIUM_CANDIDATES
        if pairs < _GPU_M2L_AUTOTUNE_PAIR_BINS[2]:
            return _GPU_M2L_AUTOTUNE_LARGE_CANDIDATES
        return _GPU_M2L_AUTOTUNE_XL_CANDIDATES

    def _sample_and_remap_far_pairs_for_autotune(
        self,
        *,
        src: Array,
        tgt: Array,
        max_pairs: int = _GPU_M2L_AUTOTUNE_MAX_SAMPLE_PAIRS,
        max_nodes: int = _GPU_M2L_AUTOTUNE_MAX_SAMPLE_NODES,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample far pairs and remap node ids to a compact contiguous range."""

        src_np = np.asarray(jax.device_get(src), dtype=np.int64).reshape(-1)
        tgt_np = np.asarray(jax.device_get(tgt), dtype=np.int64).reshape(-1)
        pair_count = int(src_np.shape[0])
        if pair_count == 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int64),
            )

        stride = max(1, pair_count // int(max_pairs))
        src_view = src_np[::stride]
        tgt_view = tgt_np[::stride]
        if src_view.shape[0] > int(max_pairs):
            src_view = src_view[: int(max_pairs)]
            tgt_view = tgt_view[: int(max_pairs)]

        node_to_local: dict[int, int] = {}
        local_to_global: list[int] = []
        src_local: list[int] = []
        tgt_local: list[int] = []
        max_nodes_int = int(max_nodes)
        for src_i, tgt_i in zip(src_view.tolist(), tgt_view.tolist()):
            src_g = int(src_i)
            tgt_g = int(tgt_i)
            src_local_id = node_to_local.get(src_g)
            if src_local_id is None:
                if len(local_to_global) >= max_nodes_int:
                    continue
                src_local_id = len(local_to_global)
                node_to_local[src_g] = src_local_id
                local_to_global.append(src_g)
            tgt_local_id = node_to_local.get(tgt_g)
            if tgt_local_id is None:
                if len(local_to_global) >= max_nodes_int:
                    continue
                tgt_local_id = len(local_to_global)
                node_to_local[tgt_g] = tgt_local_id
                local_to_global.append(tgt_g)
            src_local.append(src_local_id)
            tgt_local.append(tgt_local_id)

        return (
            np.asarray(src_local, dtype=np.int32),
            np.asarray(tgt_local, dtype=np.int32),
            np.asarray(local_to_global, dtype=np.int64),
        )

    def _autotune_runtime_m2l_chunk_size(
        self,
        *,
        upward: TreeUpwardData,
        src: Array,
        tgt: Array,
        order: int,
        pair_count: int,
    ) -> Optional[int]:
        """Auto-select M2L chunk size on GPU for streamed far-pair execution."""

        if (
            not bool(self.autotune_m2l_chunk)
            or self.expansion_basis != "solidfmm"
            or jax.default_backend() != "gpu"
            or int(pair_count) <= 0
        ):
            return None

        basis_mode = self._solidfmm_basis_mode()
        order_int = int(order)
        pair_count_int = int(pair_count)
        dtype_name = str(jnp.asarray(upward.multipoles.centers).dtype)
        pair_bin = 0
        for idx, upper in enumerate(_GPU_M2L_AUTOTUNE_PAIR_BINS):
            if pair_count_int < int(upper):
                pair_bin = idx
                break
        else:
            pair_bin = len(_GPU_M2L_AUTOTUNE_PAIR_BINS)
        key = (
            "gpu",
            str(basis_mode),
            dtype_name,
            order_int,
            str(self.complex_rotation),
            "" if self.m2l_impl is None else str(self.m2l_impl),
            int(bool(self.use_pallas)),
            int(pair_bin),
        )
        cached = _m2l_autotune_lookup(key)
        if cached is not None:
            return int(cached)

        candidates = self._select_autotune_m2l_candidates(pair_count=pair_count_int)
        (
            src_sample_np,
            tgt_sample_np,
            local_to_global_np,
        ) = self._sample_and_remap_far_pairs_for_autotune(src=src, tgt=tgt)
        if src_sample_np.size == 0 or local_to_global_np.size == 0:
            return None

        local_to_global = jnp.asarray(local_to_global_np, dtype=INDEX_DTYPE)
        src_sample = jnp.asarray(src_sample_np, dtype=INDEX_DTYPE)
        tgt_sample = jnp.asarray(tgt_sample_np, dtype=INDEX_DTYPE)
        centers = jnp.asarray(upward.multipoles.centers)[local_to_global]
        coeff_count = sh_size(order_int)
        if basis_mode == "complex":
            coeff_dtype = complex_dtype_for_real(centers.dtype)
            multip_all = jnp.asarray(upward.multipoles.packed).astype(coeff_dtype)
        else:
            coeff_dtype = centers.dtype
            multip_all = complex_to_real_coeffs(
                jnp.asarray(upward.multipoles.packed), order=order_int
            ).astype(coeff_dtype)
        multip = multip_all[local_to_global, :coeff_count]
        locals0 = jnp.zeros(
            (int(local_to_global_np.shape[0]), int(coeff_count)),
            dtype=coeff_dtype,
        )
        total_nodes = int(local_to_global_np.shape[0])
        best_chunk: Optional[int] = None
        best_time = math.inf

        for chunk in candidates:
            chunk_int = int(chunk)
            if chunk_int <= 0:
                continue
            try:
                if basis_mode == "complex":
                    _ = _accumulate_solidfmm_m2l_chunked_scan(
                        locals0,
                        multip,
                        centers,
                        src_sample,
                        tgt_sample,
                        jnp.asarray(src_sample.shape[0], dtype=INDEX_DTYPE),
                        order=order_int,
                        rotation=str(self.complex_rotation),
                        total_nodes=total_nodes,
                        chunk_size=chunk_int,
                    ).block_until_ready()
                    t0 = time.perf_counter()
                    _ = _accumulate_solidfmm_m2l_chunked_scan(
                        locals0,
                        multip,
                        centers,
                        src_sample,
                        tgt_sample,
                        jnp.asarray(src_sample.shape[0], dtype=INDEX_DTYPE),
                        order=order_int,
                        rotation=str(self.complex_rotation),
                        total_nodes=total_nodes,
                        chunk_size=chunk_int,
                    ).block_until_ready()
                else:
                    m2l_impl = (
                        "rot_scale" if self.m2l_impl is None else str(self.m2l_impl)
                    )
                    if self.use_pallas:
                        _ = _accumulate_real_m2l_chunked_scan_pallas(
                            locals0,
                            multip,
                            centers,
                            src_sample,
                            tgt_sample,
                            order=order_int,
                            m2l_impl=m2l_impl,
                            total_nodes=total_nodes,
                            chunk_size=chunk_int,
                        ).block_until_ready()
                        t0 = time.perf_counter()
                        _ = _accumulate_real_m2l_chunked_scan_pallas(
                            locals0,
                            multip,
                            centers,
                            src_sample,
                            tgt_sample,
                            order=order_int,
                            m2l_impl=m2l_impl,
                            total_nodes=total_nodes,
                            chunk_size=chunk_int,
                        ).block_until_ready()
                    else:
                        _ = _accumulate_real_m2l_chunked_scan(
                            locals0,
                            multip,
                            centers,
                            src_sample,
                            tgt_sample,
                            jnp.asarray(src_sample.shape[0], dtype=INDEX_DTYPE),
                            order=order_int,
                            m2l_impl=m2l_impl,
                            total_nodes=total_nodes,
                            chunk_size=chunk_int,
                        ).block_until_ready()
                        t0 = time.perf_counter()
                        _ = _accumulate_real_m2l_chunked_scan(
                            locals0,
                            multip,
                            centers,
                            src_sample,
                            tgt_sample,
                            jnp.asarray(src_sample.shape[0], dtype=INDEX_DTYPE),
                            order=order_int,
                            m2l_impl=m2l_impl,
                            total_nodes=total_nodes,
                            chunk_size=chunk_int,
                        ).block_until_ready()
                elapsed = float(time.perf_counter() - t0)
            except Exception:
                continue
            if elapsed < best_time:
                best_time = elapsed
                best_chunk = chunk_int

        if best_chunk is not None:
            _m2l_autotune_store(key, int(best_chunk))
        return best_chunk

    def _validate_prepare_state_request(
        self,
        *,
        leaf_size: int,
        max_order: int,
    ) -> None:
        """Validate order/leaf-size constraints for state preparation."""
        if leaf_size < 1:
            raise ValueError("leaf_size must be >= 1")
        if self.fixed_order is not None and int(self.fixed_order) != int(max_order):
            raise ValueError("fixed_order must match max_order")
        if max_order > MAX_MULTIPOLE_ORDER and self.expansion_basis != "solidfmm":
            raise NotImplementedError(
                "orders above 4 require expansion_basis='solidfmm'",
            )

    def _prepare_state_input_arrays(
        self,
        positions: Array,
        masses: Array,
    ) -> tuple[Array, Array, Any]:
        """Validate prepare-state inputs and cast them to the working dtype."""
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        input_dtype = positions_arr.dtype

        if positions_arr.ndim != 2 or positions_arr.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if masses_arr.shape != (positions_arr.shape[0],):
            raise ValueError("masses must have shape (N,)")
        if positions_arr.shape[0] == 0:
            raise ValueError("need at least one particle")

        working_dtype = self.working_dtype
        if working_dtype is not None and positions_arr.dtype != working_dtype:
            positions_arr = positions_arr.astype(working_dtype)
        if working_dtype is not None and masses_arr.dtype != working_dtype:
            masses_arr = masses_arr.astype(working_dtype)
        return positions_arr, masses_arr, input_dtype

    def _resolve_prepare_state_bounds(
        self,
        *,
        positions: Array,
        bounds: Optional[Tuple[Array, Array]],
    ) -> tuple[Array, Array]:
        """Return bounds converted to the working dtype or infer them."""
        if bounds is None:
            min_corner, max_corner = _infer_bounds(positions)
            return min_corner, max_corner
        min_corner, max_corner = bounds
        return (
            jnp.asarray(min_corner, dtype=positions.dtype),
            jnp.asarray(max_corner, dtype=positions.dtype),
        )

    def _topology_reuse_candidate(
        self,
        *,
        positions: Array,
        bounds: Tuple[Array, Array],
        tree_config: TreeBuilderConfig,
        leaf_size: int,
        refine_local: bool,
        max_refine_levels: int,
        aspect_threshold: float,
        allow_stateful_cache: bool,
    ) -> Optional[_TopologyReuseCandidate]:
        """Return a radix-topology reuse signature when host-side caching is safe."""

        if (
            (not self.reuse_topology and tree_config.mode != "static_radix")
            or not allow_stateful_cache
            or self.tree_type != "radix"
        ):
            return None
        try:
            morton_codes = morton_encode(positions, bounds)
            orig_idx = jnp.arange(positions.shape[0], dtype=INDEX_DTYPE)
            sorted_indices = jnp.lexsort((orig_idx, morton_codes))
            sorted_codes = morton_codes[sorted_indices]
            if tree_config.mode == "static_radix":
                num_leaves = (int(positions.shape[0]) + int(leaf_size) - 1) // int(
                    leaf_size
                )
                hasher = hashlib.sha256()
                hasher.update(b"static_radix_topology_v1")
                hasher.update(
                    np.asarray(int(positions.shape[0]), dtype=np.int64).tobytes()
                )
                hasher.update(
                    np.asarray(max(2 * num_leaves - 1, 1), dtype=np.int64).tobytes()
                )
                hasher.update(
                    np.asarray(max(num_leaves - 1, 0), dtype=np.int64).tobytes()
                )
                hasher.update(np.asarray(int(leaf_size), dtype=np.int64).tobytes())
                key = hasher.hexdigest()
            else:
                key = self._topology_reuse_key_from_sorted_codes(
                    sorted_codes=sorted_codes,
                    tree_config=tree_config,
                    leaf_size=leaf_size,
                    refine_local=refine_local,
                    max_refine_levels=max_refine_levels,
                    aspect_threshold=aspect_threshold,
                )
        except Exception:
            return None
        if key is None:
            return None
        return _TopologyReuseCandidate(
            key=key,
            sorted_indices=jnp.asarray(sorted_indices, dtype=INDEX_DTYPE),
            sorted_codes=jnp.asarray(sorted_codes),
            bounds=bounds,
        )

    def _topology_reuse_key_from_sorted_codes(
        self,
        *,
        sorted_codes: Array,
        tree_config: TreeBuilderConfig,
        leaf_size: int,
        refine_local: bool,
        max_refine_levels: int,
        aspect_threshold: float,
    ) -> Optional[str]:
        """Build a stable topology key from sorted Morton codes and tree options."""

        try:
            hasher = hashlib.sha256()
            hasher.update(
                np.asarray(jax.device_get(sorted_codes), dtype=np.uint64).tobytes()
            )
            hasher.update(str(tree_config.mode).encode("utf8"))
            leaf_param = (
                int(leaf_size)
                if tree_config.mode == "lbvh"
                else int(tree_config.target_leaf_particles)
            )
            hasher.update(np.asarray(leaf_param, dtype=np.int64).tobytes())
            hasher.update(np.asarray(int(bool(refine_local)), dtype=np.int64).tobytes())
            hasher.update(np.asarray(int(max_refine_levels), dtype=np.int64).tobytes())
            hasher.update(
                np.asarray(float(aspect_threshold), dtype=np.float64).tobytes()
            )
        except Exception:
            return None
        return hasher.hexdigest()

    def _static_radix_topology_key_from_tree(
        self,
        tree: RadixTree,
        *,
        leaf_size: int,
    ) -> Optional[str]:
        """Return a stable key for a static-radix data-structure shape."""

        try:
            hasher = hashlib.sha256()
            hasher.update(b"static_radix_topology_v1")
            hasher.update(np.asarray(int(tree.num_particles), dtype=np.int64).tobytes())
            hasher.update(
                np.asarray(int(tree.parent.shape[0]), dtype=np.int64).tobytes()
            )
            hasher.update(
                np.asarray(int(tree.num_internal_nodes), dtype=np.int64).tobytes()
            )
            hasher.update(np.asarray(int(leaf_size), dtype=np.int64).tobytes())
        except Exception:
            return None
        return hasher.hexdigest()

    def _rebuild_tree_artifacts_from_topology(
        self,
        *,
        candidate: _TopologyReuseCandidate,
        entry: _TopologyReuseEntry,
        positions: Array,
        masses: Array,
    ) -> _TreeBuildArtifacts:
        """Reorder particles and attach them to a cached radix topology."""

        positions_sorted, masses_sorted, inverse = reorder_particles_by_indices(
            positions,
            masses,
            candidate.sorted_indices,
        )
        cached_tree = entry.tree
        if not isinstance(cached_tree, RadixTree):
            raise ValueError("topology reuse currently supports radix trees only")
        topology = cached_tree.topology
        if (
            cached_tree.build_mode == "static_radix"
            and candidate.sorted_codes is not None
            and candidate.bounds is not None
        ):
            num_internal = int(cached_tree.num_internal_nodes)
            leaf_starts = jnp.asarray(
                cached_tree.node_ranges[num_internal:, 0],
                dtype=INDEX_DTYPE,
            )
            topology = topology._replace(
                particle_indices=jnp.asarray(
                    candidate.sorted_indices,
                    dtype=INDEX_DTYPE,
                ),
                morton_codes=jnp.asarray(candidate.sorted_codes),
                bounds_min=jnp.asarray(candidate.bounds[0], dtype=positions.dtype),
                bounds_max=jnp.asarray(candidate.bounds[1], dtype=positions.dtype),
                leaf_codes=jnp.asarray(candidate.sorted_codes)[leaf_starts],
                leaf_depths=jnp.full(
                    (leaf_starts.shape[0],),
                    -1,
                    dtype=INDEX_DTYPE,
                ),
                use_morton_geometry=jnp.asarray(False, dtype=jnp.bool_),
            )
        rebuilt_tree = RadixTree(
            topology=topology,
            build_mode=cached_tree.build_mode,
            positions_sorted=positions_sorted,
            masses_sorted=masses_sorted,
            inverse_permutation=inverse,
            workspace=cached_tree.workspace,
        )
        return _TreeBuildArtifacts(
            tree=rebuilt_tree,
            positions_sorted=positions_sorted,
            masses_sorted=masses_sorted,
            inverse_permutation=inverse,
            workspace=cached_tree.workspace,
            max_leaf_size=int(entry.max_leaf_size),
            cache_leaf_parameter=int(entry.cache_leaf_parameter),
        )

    def _rebuild_tree_artifacts_from_static_template(
        self,
        *,
        template_tree: RadixTree,
        positions: Array,
        masses: Array,
        bounds: Optional[Tuple[Array, Array]],
        max_leaf_size: int,
        cache_leaf_parameter: int,
    ) -> _TreeBuildArtifacts:
        """Refresh static-radix tree artifacts from a fixed template topology."""

        rebuilt_result = rebuild_static_radix_tree_from_template(
            positions,
            masses,
            template_tree,
            bounds=bounds,
            return_reordered=True,
        )
        if not isinstance(rebuilt_result, tuple) or len(rebuilt_result) != 4:
            raise RuntimeError(
                "static radix template rebuild must return tree and reordered arrays"
            )
        rebuilt_tree, positions_sorted, masses_sorted, inverse = rebuilt_result
        if not isinstance(rebuilt_tree, RadixTree):
            raise ValueError("static radix template rebuild returned non-radix tree")
        return _TreeBuildArtifacts(
            tree=rebuilt_tree,
            positions_sorted=jnp.asarray(positions_sorted, dtype=positions.dtype),
            masses_sorted=jnp.asarray(masses_sorted, dtype=masses.dtype),
            inverse_permutation=jnp.asarray(inverse, dtype=INDEX_DTYPE),
            workspace=template_tree.workspace,
            max_leaf_size=int(max_leaf_size),
            cache_leaf_parameter=int(cache_leaf_parameter),
        )

    def _build_locals_template_for_prepare_state(
        self,
        *,
        tree: Tree,
        upward: TreeUpwardData,
        max_order: int,
        pos_sorted: Array,
    ) -> Optional[LocalExpansionData]:
        """Build initial local-expansion buffers matching the active basis."""
        if self.expansion_basis == "solidfmm":
            # Solid-FMM does not reuse a cached locals template across prepare calls.
            # Let the downward pass allocate its accumulator on demand so we do not
            # hold a redundant zero buffer alive across dual-tree staging.
            return None

        template = self._locals_template
        total_nodes = int(tree.parent.shape[0])
        coeff_count = total_coefficients(max_order)
        reuse_template = (
            template is not None
            and int(template.order) == max_order
            and template.coefficients.shape == (total_nodes, coeff_count)
            and template.coefficients.dtype == pos_sorted.dtype
        )

        if reuse_template:
            return LocalExpansionData(
                order=max_order,
                centers=upward.multipoles.centers,
                coefficients=jnp.zeros_like(template.coefficients),
            )
        return initialize_local_expansions(
            tree,
            upward.multipoles.centers,
            max_order=max_order,
        )

    def _prepare_state_tree_and_upward(
        self,
        *,
        positions_arr: Array,
        masses_arr: Array,
        bounds: Optional[Tuple[Array, Array]],
        leaf_size: int,
        max_order: int,
        refine_local_val: bool,
        max_refine_levels_val: int,
        aspect_threshold_val: float,
        jit_tree_override: Optional[bool],
        upward_center_mode: str,
        allow_stateful_cache: bool,
    ) -> _PrepareStateTreeUpwardArtifacts:
        """Build tree artifacts and run upward preparation for prepare_state."""
        tree_config = self.config.tree
        if self.tree_type != "radix" and tree_config.mode in (
            "fixed_depth",
            "static_radix",
        ):
            tree_config = TreeBuilderConfig(
                mode="lbvh",
                target_leaf_particles=tree_config.target_leaf_particles,
                refine_local=tree_config.refine_local,
                max_refine_levels=tree_config.max_refine_levels,
                aspect_threshold=tree_config.aspect_threshold,
            )

        inferred_bounds = self._resolve_prepare_state_bounds(
            positions=positions_arr,
            bounds=bounds,
        )
        jit_tree_flag = self._resolve_jit_tree_flag(
            positions_arr,
            jit_tree_override=jit_tree_override,
        )

        self._recent_topology_reused = False
        cached_topology = self._topology_reuse_entry
        topology_candidate = None
        if cached_topology is not None:
            topology_candidate = self._topology_reuse_candidate(
                positions=positions_arr,
                bounds=inferred_bounds,
                tree_config=tree_config,
                leaf_size=int(leaf_size),
                refine_local=refine_local_val,
                max_refine_levels=max_refine_levels_val,
                aspect_threshold=aspect_threshold_val,
                allow_stateful_cache=allow_stateful_cache,
            )
        can_reuse_cached_topology = (
            topology_candidate is not None
            and cached_topology is not None
            and topology_candidate.key == cached_topology.key
            and cached_topology.reuse_count < (self.rebuild_every - 1)
        )
        topology_key_for_state: Optional[str] = None

        tree_build_t0 = time.perf_counter()
        if can_reuse_cached_topology:
            build_artifacts = self._rebuild_tree_artifacts_from_topology(
                candidate=topology_candidate,
                entry=cached_topology,
                positions=positions_arr,
                masses=masses_arr,
            )
            self._recent_topology_reused = True
            topology_key_for_state = topology_candidate.key
            self._topology_reuse_entry = _TopologyReuseEntry(
                key=cached_topology.key,
                tree=build_artifacts.tree,
                max_leaf_size=int(cached_topology.max_leaf_size),
                cache_leaf_parameter=int(cached_topology.cache_leaf_parameter),
                reuse_count=int(cached_topology.reuse_count) + 1,
            )
        else:
            build_artifacts = _build_tree_with_config(
                positions_arr,
                masses_arr,
                inferred_bounds,
                tree_type=self.tree_type,
                tree_config=tree_config,
                leaf_size=int(leaf_size),
                workspace=(self._tree_workspace if allow_stateful_cache else None),
                jit_tree=jit_tree_flag,
                refine_local=refine_local_val,
                max_refine_levels=max_refine_levels_val,
                aspect_threshold=aspect_threshold_val,
            )
            if allow_stateful_cache:
                self._tree_workspace = build_artifacts.workspace
                if tree_config.mode == "static_radix":
                    topology_key_for_state = self._static_radix_topology_key_from_tree(
                        build_artifacts.tree,
                        leaf_size=int(leaf_size),
                    )
                if self.reuse_topology:
                    if topology_candidate is not None:
                        topology_key_for_state = topology_candidate.key
                    elif topology_key_for_state is None:
                        topology_key_for_state = (
                            self._topology_reuse_key_from_sorted_codes(
                                sorted_codes=build_artifacts.tree.morton_codes,
                                tree_config=tree_config,
                                leaf_size=int(leaf_size),
                                refine_local=refine_local_val,
                                max_refine_levels=max_refine_levels_val,
                                aspect_threshold=aspect_threshold_val,
                            )
                        )
                if topology_key_for_state is not None:
                    self._topology_reuse_entry = _TopologyReuseEntry(
                        key=topology_key_for_state,
                        tree=build_artifacts.tree,
                        max_leaf_size=int(build_artifacts.max_leaf_size),
                        cache_leaf_parameter=int(build_artifacts.cache_leaf_parameter),
                        reuse_count=0,
                    )
                elif self.reuse_topology:
                    self._topology_reuse_entry = None
        if bool(getattr(self, "_refresh_timing_active", False)):
            self._refresh_timing_tree_build_seconds += (
                time.perf_counter() - tree_build_t0
            )

        tree = build_artifacts.tree
        pos_sorted = build_artifacts.positions_sorted
        mass_sorted = build_artifacts.masses_sorted
        total_nodes = int(tree.parent.shape[0])
        num_internal = int(jnp.asarray(tree.left_child).shape[0])
        num_leaves = int(total_nodes - num_internal)
        _prepare_diag(
            "tree built "
            f"particles={int(pos_sorted.shape[0])} leaf_size={int(leaf_size)} "
            f"total_nodes={total_nodes} num_internal={num_internal} num_leaves={num_leaves} "
            f"tree_type={self.tree_type} tree_mode={tree_config.mode} jit_tree={bool(jit_tree_flag)}"
        )
        # Keep one leaf-size contract in eager and traced paths.
        leaf_cap_hint = int(build_artifacts.max_leaf_size)
        upward_leaf_batch = (
            int(self.upward_leaf_batch_size)
            if self.upward_leaf_batch_size is not None
            else None
        )
        _prepare_diag(
            "upward start "
            f"max_order={int(max_order)} center_mode={upward_center_mode} "
            f"leaf_cap={leaf_cap_hint} upward_leaf_batch_size={upward_leaf_batch}"
        )
        geometry_cache_key: Optional[tuple[Any, ...]] = None
        cached_geometry = None
        if allow_stateful_cache:
            if topology_key_for_state is not None:
                geometry_key_base: Any = topology_key_for_state
            else:
                bounds_key = (
                    None if bounds is None else (int(id(bounds[0])), int(id(bounds[1])))
                )
                geometry_key_base = (
                    self.tree_type,
                    tree_config.mode,
                    int(leaf_size),
                    bool(refine_local_val),
                    int(max_refine_levels_val),
                    float(aspect_threshold_val),
                    bounds_key,
                )
            geometry_cache_key = (
                geometry_key_base,
                int(id(positions_arr)),
                int(positions_arr.shape[0]),
                str(positions_arr.dtype),
            )
            geometry_entry = self._geometry_reuse_entry
            if geometry_entry is not None and geometry_entry.key == geometry_cache_key:
                cached_geometry = geometry_entry.geometry

        upward_t0 = time.perf_counter()
        defer_geometry = (
            bool(getattr(self, "_refresh_timing_active", False))
            and tree_config.mode == "static_radix"
            and str(upward_center_mode).strip().lower() == "com"
            and self._interaction_cache is not None
        )
        upward = self.prepare_upward_sweep(
            tree,
            pos_sorted,
            mass_sorted,
            max_order=max_order,
            center_mode=upward_center_mode,
            max_leaf_size=leaf_cap_hint,
            precomputed_geometry=cached_geometry,
            defer_geometry=defer_geometry,
        )
        if bool(getattr(self, "_refresh_timing_active", False)):
            self._refresh_timing_upward_compute_seconds += (
                time.perf_counter() - upward_t0
            )
        if (
            allow_stateful_cache
            and geometry_cache_key is not None
            and upward.geometry is not None
        ):
            self._geometry_reuse_entry = _GeometryReuseEntry(
                key=geometry_cache_key,
                geometry=upward.geometry,
            )
        _prepare_diag(
            "upward done "
            f"multipole_order={int(upward.multipoles.order)} "
            f"multipole_shape={tuple(int(v) for v in upward.multipoles.packed.shape)}"
        )
        locals_template = self._build_locals_template_for_prepare_state(
            tree=tree,
            upward=upward,
            max_order=max_order,
            pos_sorted=pos_sorted,
        )

        return _PrepareStateTreeUpwardArtifacts(
            tree_mode=tree_config.mode,
            tree=tree,
            positions_sorted=pos_sorted,
            masses_sorted=mass_sorted,
            inverse_permutation=build_artifacts.inverse_permutation,
            leaf_cap=leaf_cap_hint,
            leaf_parameter=build_artifacts.cache_leaf_parameter,
            topology_key=topology_key_for_state,
            upward=upward,
            locals_template=locals_template,
        )

    def _prepare_state_tree_upward_and_dual_downward(
        self,
        *,
        positions_arr: Array,
        masses_arr: Array,
        bounds: Optional[Tuple[Array, Array]],
        leaf_size: int,
        max_order: int,
        refine_local_val: bool,
        max_refine_levels_val: int,
        aspect_threshold_val: float,
        jit_tree_override: Optional[bool],
        upward_center_mode: str,
        allow_stateful_cache: bool,
        theta_val: float,
        mac_type_val: MACType,
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        record_retry: Callable[[DualTreeRetryEvent], None],
    ) -> tuple[_PrepareStateTreeUpwardArtifacts, _PrepareStateDualDownwardArtifacts]:
        """Build tree/upward and dual/downward artifacts in one helper call."""

        tree_artifacts = self._prepare_state_tree_and_upward(
            positions_arr=positions_arr,
            masses_arr=masses_arr,
            bounds=bounds,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            refine_local_val=bool(refine_local_val),
            max_refine_levels_val=int(max_refine_levels_val),
            aspect_threshold_val=float(aspect_threshold_val),
            jit_tree_override=jit_tree_override,
            upward_center_mode=upward_center_mode,
            allow_stateful_cache=bool(allow_stateful_cache),
        )
        dual_downward_artifacts = self._prepare_state_dual_and_downward(
            tree_artifacts=tree_artifacts,
            force_scale_nodes=None,
            upward_center_mode=upward_center_mode,
            theta_val=float(theta_val),
            mac_type_val=mac_type_val,
            dehnen_radius_scale=self.dehnen_radius_scale,
            runtime_traversal_config=runtime_traversal_config,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            grouped_interactions=False,
            farfield_mode=self.farfield_mode,
            record_retry=record_retry,
            refine_local_val=bool(refine_local_val),
            max_refine_levels_val=int(max_refine_levels_val),
            aspect_threshold_val=float(aspect_threshold_val),
            allow_stateful_cache=bool(allow_stateful_cache),
        )
        return tree_artifacts, dual_downward_artifacts

    def _prepare_state_dual_and_downward(
        self,
        *,
        tree_artifacts: _PrepareStateTreeUpwardArtifacts,
        force_scale_nodes: Optional[Array],
        upward_center_mode: str,
        theta_val: float,
        mac_type_val: MACType,
        dehnen_radius_scale: float,
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        grouped_interactions: bool,
        farfield_mode: str,
        record_retry: Callable[[DualTreeRetryEvent], None],
        refine_local_val: bool,
        max_refine_levels_val: int,
        aspect_threshold_val: float,
        allow_stateful_cache: bool,
        suppress_host_side_effects: bool = False,
    ) -> _PrepareStateDualDownwardArtifacts:
        """Build/reuse interactions and prepare downward artifacts.

        Memory note:
        The dominant warm prepare peak is usually the far-field payload that
        exists long enough to feed M2L: raw interactions or streamed COO pairs,
        optional grouped layouts, and the downward locals buffer. Only the
        returned ``TreeDownwardData`` is meant to survive into prepared state;
        grouped schedules and other M2L feed artifacts are transient and should
        stay scoped to this helper.
        """
        suppress_host_side_effects = bool(suppress_host_side_effects)
        refresh_timing_active = bool(
            getattr(self, "_refresh_timing_active", False)
        ) and (not suppress_host_side_effects)
        dual_total_t0 = time.perf_counter() if refresh_timing_active else 0.0
        dual_stage_sum = 0.0

        def _stage_now() -> float:
            return time.perf_counter() if refresh_timing_active else 0.0

        def _record_dual_stage(attr: str, start: float) -> None:
            nonlocal dual_stage_sum
            if not refresh_timing_active:
                return
            elapsed = float(time.perf_counter() - start)
            dual_stage_sum += elapsed
            setattr(
                self,
                attr,
                float(getattr(self, attr, 0.0)) + elapsed,
            )

        def _record_dual_artifact_substage(name: str, elapsed: float) -> None:
            if not refresh_timing_active:
                return
            attr_by_name = {
                "dual_split_shared_far_pairs_leaf_neighbors": "_refresh_timing_dual_split_shared_far_near_seconds",
                "dual_split_shared_count": "_refresh_timing_dual_split_shared_count_seconds",
                "dual_split_shared_combined_fill": "_refresh_timing_dual_split_shared_combined_fill_seconds",
                "dual_split_shared_far_fill": "_refresh_timing_dual_split_shared_far_fill_seconds",
                "dual_split_shared_near_fill": "_refresh_timing_dual_split_shared_near_fill_seconds",
                "dual_split_far_pairs": "_refresh_timing_dual_split_far_pairs_seconds",
                "dual_split_leaf_neighbors": "_refresh_timing_dual_split_leaf_neighbors_seconds",
                "dual_split_interactions_and_neighbors": "_refresh_timing_dual_split_combined_seconds",
                "dual_raw_interactions_and_neighbors": "_refresh_timing_dual_raw_combined_seconds",
                "dual_split_dense_buffers": "_refresh_timing_dual_split_dense_buffers_seconds",
            }
            attr = attr_by_name.get(str(name))
            if attr is None:
                return
            setattr(self, attr, float(getattr(self, attr, 0.0)) + float(elapsed))

        stage_t0 = _stage_now()
        pair_policy = None
        policy_state = None
        cache_key = None
        use_paper_fixed_policy = (not self.adaptive_order) and (
            self._uses_paper_style_traversal_policy()
        )
        if (
            bool(suppress_host_side_effects)
            and bool(getattr(self, "_strict_fused_mode_active", False))
            and bool(getattr(self, "_strict_fused_device_only", False))
        ):
            use_paper_fixed_policy = False

        # A static-radix topology key describes the capacity-fixed tree shape,
        # not the current leaf membership, geometry, or MAC decisions. Reusing
        # cached dual-tree artifacts across evolved positions can therefore
        # attach stale neighbor/far-field payloads to freshly sorted particles.
        stateful_cache_enabled = (
            bool(allow_stateful_cache)
            and bool(self.enable_interaction_cache)
            and str(tree_artifacts.tree_mode) != "static_radix"
            and (not suppress_host_side_effects)
        )
        grouped_interactions_active = bool(grouped_interactions)
        strict_fused_device_only_hot_path = (
            bool(suppress_host_side_effects)
            and bool(getattr(self, "_strict_fused_mode_active", False))
            and bool(getattr(self, "_strict_fused_device_only", False))
        )
        adaptive_order_active = bool(self.adaptive_order) and not bool(
            strict_fused_device_only_hot_path
        )
        mixed_order_farfield_active = bool(self.mixed_order_farfield) and not bool(
            strict_fused_device_only_hot_path
        )
        retain_interactions_active = bool(self.retain_interactions) and not bool(
            strict_fused_device_only_hot_path
        )
        need_traversal_result = (
            bool(self.retain_traversal_result)
            and not bool(strict_fused_device_only_hot_path)
        ) or bool(use_paper_fixed_policy)
        traced_prepare_inputs = bool(
            _contains_tracer(
                (tree_artifacts.positions_sorted, tree_artifacts.masses_sorted)
            )
        )
        strict_fused_node_interactions_safe_path = (
            bool(strict_fused_device_only_hot_path)
            and str(
                os.environ.get(
                    "JACCPOT_STATIC_STRICT_FUSED_NODE_INTERACTIONS_SAFE_PATH",
                    "0",
                )
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            and not (
                str(
                    os.environ.get(
                        "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE",
                        "0",
                    )
                )
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            )
        )
        use_compact_streamed_pairs = (
            (bool(self.streamed_far_pairs) or bool(strict_fused_device_only_hot_path))
            and not bool(strict_fused_node_interactions_safe_path)
            and not adaptive_order_active
            and not grouped_interactions_active
            and not mixed_order_farfield_active
            and not retain_interactions_active
            and not bool(need_traversal_result)
            and (
                not bool(traced_prepare_inputs)
                or bool(strict_fused_device_only_hot_path)
            )
        )
        need_compact_far_pairs = (
            bool(adaptive_order_active) and not bool(need_traversal_result)
        ) or bool(use_compact_streamed_pairs)
        need_node_interactions = not bool(use_compact_streamed_pairs)
        use_dense_interactions_for_prepare = bool(self.use_dense_interactions) and (
            self.expansion_basis != "solidfmm"
        )
        if not suppress_host_side_effects:
            _prepare_diag(
                "dual-tree start "
                f"theta={theta_val:.3f} mac_type={mac_type_val} "
                f"streamed={bool(self.streamed_far_pairs)} grouped={grouped_interactions_active} "
                f"farfield_mode={farfield_mode} memory_objective={self.memory_objective} "
                f"traversal_config={runtime_traversal_config} "
                f"need_compact_far_pairs={bool(need_compact_far_pairs)} "
                f"need_node_interactions={bool(need_node_interactions)} "
                f"dense_buffers={bool(use_dense_interactions_for_prepare)}"
            )
        if (
            runtime_traversal_config is not None
            and self.memory_objective == "minimum_memory"
            and jax.default_backend() == "gpu"
            and self.tree_type == "radix"
            and self.expansion_basis == "solidfmm"
            and bool(self.streamed_far_pairs)
            and not grouped_interactions_active
        ):
            total_nodes = int(tree_artifacts.tree.parent.shape[0])
            num_internal = int(jnp.asarray(tree_artifacts.tree.left_child).shape[0])
            num_leaves = max(1, total_nodes - num_internal)
            sanitized_traversal_config = (
                _cap_minimum_memory_streamed_gpu_traversal_config_for_tree(
                    traversal_config=runtime_traversal_config,
                    total_nodes=total_nodes,
                    num_leaves=num_leaves,
                    num_particles=int(tree_artifacts.positions_sorted.shape[0]),
                )
            )
            if sanitized_traversal_config != runtime_traversal_config:
                far_slots_before = total_nodes * int(
                    runtime_traversal_config.max_interactions_per_node
                )
                near_slots_before = num_leaves * int(
                    runtime_traversal_config.max_neighbors_per_leaf
                )
                if not suppress_host_side_effects:
                    _prepare_diag(
                        "capped explicit traversal_config for legacy streamed GPU walk "
                        f"total_nodes={total_nodes} num_leaves={num_leaves} "
                        f"far_slots={far_slots_before} near_slots={near_slots_before} "
                        f"from={runtime_traversal_config} "
                        f"to={sanitized_traversal_config}"
                    )
                runtime_traversal_config = sanitized_traversal_config
        jit_traversal_for_prepare = bool(
            self._jit_traversal_default
        ) and not _contains_tracer(
            (tree_artifacts.positions_sorted, tree_artifacts.masses_sorted)
        )
        if self.prepare_stage_memory_split_enabled is not None:
            allow_split_build = bool(self.prepare_stage_memory_split_enabled)
        elif self._prepare_stage_memory_split_env_override is None:
            # Default to the lower-peak split traversal build in the production
            # minimum-memory streamed GPU path; keep env opt-out for debugging.
            allow_split_build = bool(
                self._streamed_minimum_memory_gpu_default_split_build
                and not grouped_interactions_active
            )
        else:
            allow_split_build = bool(self._prepare_stage_memory_split_env_override)
        if bool(strict_fused_device_only_hot_path):
            allow_split_build = True
        if not suppress_host_side_effects:
            _prepare_diag(f"allow_split_build={bool(allow_split_build)}")
        tree_mode_static_radix = (
            str(tree_artifacts.tree_mode).strip().lower() == "static_radix"
        )
        strict_mode_active = bool(
            (
                self._strict_gpu_mode_on
                or (
                    self._strict_gpu_mode_auto
                    and self._large_n_gpu_production_profile_cached
                    and tree_mode_static_radix
                )
            )
        )
        if strict_mode_active:
            strict_context_key = self._strict_cap_profile_context_key(
                tree_mode=str(tree_artifacts.tree_mode),
                leaf_parameter=int(tree_artifacts.leaf_parameter),
                particle_count=int(
                    jnp.asarray(tree_artifacts.positions_sorted).shape[0]
                ),
            )
            strict_fused_hot_path = bool(
                getattr(self, "_strict_fused_mode_active", False)
            )
            strict_profile_key_stable = str(self._strict_profiled_context_key) == str(
                strict_context_key
            )
            if not (strict_fused_hot_path and strict_profile_key_stable):
                self._maybe_load_strict_cap_profile(context_key=strict_context_key)
            if bool(self._strict_cap_require_exact_profile_match):
                if str(self._strict_profiled_context_key) != str(strict_context_key):
                    self._strict_runner_fail_fast_reject_count += 1
                    raise RuntimeError(
                        "strict static lane requires exact cap profile key match: "
                        f"requested={strict_context_key} "
                        f"resolved={self._strict_profiled_context_key or 'none'}"
                    )
            profiled_q = int(self._strict_profiled_max_pair_queue)
            profiled_b = int(self._strict_profiled_pair_process_block)
            if bool(self._strict_cap_require_exact_profile_match) and profiled_q <= 0:
                self._strict_runner_fail_fast_reject_count += 1
                raise RuntimeError(
                    "strict static lane requires non-zero profiled max_pair_queue "
                    f"for key {strict_context_key}"
                )
            if profiled_q > 0:
                if runtime_traversal_config is not None:
                    runtime_traversal_config = DualTreeTraversalConfig(
                        max_pair_queue=max(
                            int(runtime_traversal_config.max_pair_queue),
                            int(profiled_q),
                        ),
                        process_block=(
                            int(profiled_b)
                            if profiled_b > 0
                            else int(runtime_traversal_config.process_block)
                        ),
                        max_interactions_per_node=int(
                            runtime_traversal_config.max_interactions_per_node
                        ),
                        max_neighbors_per_leaf=int(
                            runtime_traversal_config.max_neighbors_per_leaf
                        ),
                    )
                else:
                    runtime_traversal_config = DualTreeTraversalConfig(
                        max_pair_queue=int(profiled_q),
                        process_block=(
                            int(profiled_b)
                            if profiled_b > 0
                            else int(self.pair_process_block or 1024)
                        ),
                        max_interactions_per_node=8192,
                        max_neighbors_per_leaf=4096,
                    )
            if bool(traced_prepare_inputs) and not bool(
                strict_fused_device_only_hot_path
            ):
                allow_split_build = False
            if not suppress_host_side_effects:
                self._refresh_strict_mode_active_count += 1
                # Strict mode contract: one-shot shared count->fill and single queue.
                if not bool(getattr(self, "_strict_shared_env_applied", False)):
                    os.environ["YGGDRAX_DUAL_TREE_SHARED_COUNT_FILL_ONE_SHOT"] = "1"
                    os.environ[
                        "YGGDRAX_DUAL_TREE_SHARED_COUNT_FILL_STEADY_SINGLE_QUEUE"
                    ] = "1"
                    self._strict_shared_env_applied = True
        strict_streamed_fast_path = bool(
            strict_mode_active
            and bool(allow_split_build)
            and bool(use_compact_streamed_pairs)
            and not grouped_interactions_active
            and not bool(need_traversal_result)
            and not adaptive_order_active
            and not mixed_order_farfield_active
            and (
                not bool(traced_prepare_inputs)
                or bool(strict_fused_device_only_hot_path)
            )
        )
        if bool(strict_fused_device_only_hot_path) and bool(
            getattr(self, "_strict_fused_fastlane_diag_enabled", True)
        ):
            self._strict_fused_fastlane_attempts += 1
            blockers: list[str] = []
            if not bool(strict_mode_active):
                blockers.append("strict_mode_inactive")
            if not bool(allow_split_build):
                blockers.append("split_build_disabled")
            if not bool(use_compact_streamed_pairs):
                blockers.append("compact_streamed_pairs_disabled")
            if bool(grouped_interactions_active):
                blockers.append("grouped_interactions_active")
            if bool(need_traversal_result):
                blockers.append("traversal_result_required")
            if bool(adaptive_order_active):
                blockers.append("adaptive_order_active")
            if bool(mixed_order_farfield_active):
                blockers.append("mixed_order_farfield_active")
            if bool(traced_prepare_inputs) and not bool(strict_streamed_fast_path):
                blockers.append("compact_streamed_tracer_unsupported")
            if bool(strict_streamed_fast_path):
                self._strict_fused_fastlane_hits += 1
                self._strict_fused_fastlane_last_blockers = tuple()
            else:
                self._strict_fused_fastlane_misses += 1
                self._strict_fused_fastlane_last_blockers = tuple(blockers)
                counts = dict(getattr(self, "_strict_fused_fastlane_block_counts", {}))
                for key in blockers:
                    counts[str(key)] = int(counts.get(str(key), 0)) + 1
                self._strict_fused_fastlane_block_counts = counts
        if strict_streamed_fast_path:
            if not suppress_host_side_effects:
                self._refresh_dual_planner_cache_hits += 1
                self._refresh_dual_planner_execute_count += 1
                self._refresh_dual_planner_steady_timing_bypass_count += 1
            return self._prepare_state_dual_and_downward_strict_streamed_fast(
                tree_artifacts=tree_artifacts,
                theta_val=theta_val,
                mac_type_val=mac_type_val,
                dehnen_radius_scale=dehnen_radius_scale,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                record_retry=record_retry,
                farfield_mode=farfield_mode,
                retain_interactions=bool(retain_interactions_active),
                suppress_host_side_effects=suppress_host_side_effects,
            )

        if self.adaptive_order or use_paper_fixed_policy:
            policy_orders = self.p_gears
            if use_paper_fixed_policy:
                policy_orders = (int(tree_artifacts.upward.multipoles.order),)
            if len(policy_orders) == 0:
                raise ValueError("adaptive traversal policy requires non-empty orders")
            policy_state = self._build_adaptive_policy_state(
                upward=tree_artifacts.upward,
                tree=tree_artifacts.tree,
                positions_sorted=tree_artifacts.positions_sorted,
                p_gears=policy_orders,
                force_scale_nodes=force_scale_nodes,
                eps=jnp.asarray(
                    (
                        self.adaptive_eps
                        if self.adaptive_eps is not None
                        else adaptive_policy_tolerance(
                            theta=theta_val,
                            p_gears=policy_orders,
                            dtype=tree_artifacts.upward.multipoles.packed.real.dtype,
                        )
                    ),
                    dtype=tree_artifacts.upward.multipoles.packed.real.dtype,
                ),
                theta=jnp.asarray(
                    theta_val,
                    dtype=tree_artifacts.upward.multipoles.packed.real.dtype,
                ),
                error_model_code=jnp.asarray(
                    self._traversal_policy_error_model_code(),
                    dtype=jnp.int32,
                ),
                dehnen_geometry_mode=self.dehnen_geometry_mode,
            )
            pair_policy = adaptive_pair_policy
        else:
            cache_key = _interaction_cache_key(
                tree_artifacts.tree,
                topology_key=tree_artifacts.topology_key,
                tree_mode=tree_artifacts.tree_mode,
                leaf_parameter=tree_artifacts.leaf_parameter,
                theta=theta_val,
                mac_type=mac_type_val,
                dehnen_radius_scale=dehnen_radius_scale,
                expansion_basis=self.expansion_basis,
                center_mode=upward_center_mode,
                max_pair_queue=self.max_pair_queue,
                pair_process_block=self.pair_process_block,
                traversal_config=runtime_traversal_config,
                refine_local=refine_local_val,
                max_refine_levels=max_refine_levels_val,
                aspect_threshold=aspect_threshold_val,
            )
        has_pair_policy = pair_policy is not None
        has_policy_state = policy_state is not None
        planner_enabled = bool(
            (
                self._refresh_dual_planner_mode_on
                or (
                    self._refresh_dual_planner_mode_auto
                    and self._large_n_gpu_production_profile_cached
                    and tree_mode_static_radix
                )
            )
        )
        planner_hint: Optional[_RefreshDualPlannerHint] = None
        planner_cache_hit = False
        if planner_enabled:
            strict_split_fastlane = bool(
                (
                    bool(suppress_host_side_effects)
                    and not has_pair_policy
                    and not has_policy_state
                )
                or (
                    strict_mode_active
                    and bool(allow_split_build)
                    and not grouped_interactions_active
                    and not bool(need_traversal_result)
                    and not has_pair_policy
                    and not has_policy_state
                )
            )
            if strict_split_fastlane:
                # Strict/static production lane: keep routing fully on a
                # fixed host-side decision and skip compiled route probing +
                # device_get round-trips in the refresh hot path.
                planner_hint = _RefreshDualPlannerHint(
                    use_split_build=bool(allow_split_build),
                    suppress_substage_timing=True,
                )
                planner_cache_hit = True
                if not suppress_host_side_effects:
                    self._refresh_dual_planner_cache_hits += 1
                    self._refresh_dual_planner_execute_count += 1
            else:
                traversal_key = (
                    "none"
                    if runtime_traversal_config is None
                    else (
                        f"{int(runtime_traversal_config.max_pair_queue)}:"
                        f"{int(runtime_traversal_config.process_block)}:"
                        f"{int(runtime_traversal_config.max_interactions_per_node)}:"
                        f"{int(runtime_traversal_config.max_neighbors_per_leaf)}"
                    )
                )
                planner_key = "|".join(
                    (
                        str(tree_artifacts.topology_key),
                        str(tree_artifacts.tree_mode),
                        str(int(tree_artifacts.leaf_parameter)),
                        f"{float(theta_val):.12g}",
                        str(mac_type_val),
                        str(grouped_interactions_active),
                        str(bool(need_traversal_result)),
                        str(bool(need_compact_far_pairs)),
                        str(bool(need_node_interactions)),
                        str(bool(allow_split_build)),
                        str(traversal_key),
                    )
                )
                planner_hint = self._refresh_dual_planner_cache.get(planner_key)
                if planner_hint is None:
                    if not suppress_host_side_effects:
                        self._refresh_dual_planner_cache_misses += 1
                    total_nodes_planner = int(tree_artifacts.tree.parent.shape[0])
                    internal_nodes_planner = int(
                        jnp.asarray(tree_artifacts.tree.left_child).shape[0]
                    )
                    leaf_count_planner = max(
                        0, total_nodes_planner - internal_nodes_planner
                    )
                    (
                        use_split_build_compiled,
                        _use_compact_shared_far_near_compiled,
                        suppress_substage_timing_compiled,
                    ) = _compiled_refresh_dual_planner_route(
                        allow_split_build_flag=jnp.asarray(
                            bool(allow_split_build), dtype=jnp.bool_
                        ),
                        grouped_interactions_flag=jnp.asarray(
                            grouped_interactions_active, dtype=jnp.bool_
                        ),
                        need_traversal_result_flag=jnp.asarray(
                            bool(need_traversal_result), dtype=jnp.bool_
                        ),
                        has_pair_policy_flag=jnp.asarray(
                            has_pair_policy, dtype=jnp.bool_
                        ),
                        has_policy_state_flag=jnp.asarray(
                            has_policy_state, dtype=jnp.bool_
                        ),
                        leaf_count=jnp.asarray(leaf_count_planner, dtype=jnp.int32),
                        need_node_interactions_flag=jnp.asarray(
                            bool(need_node_interactions), dtype=jnp.bool_
                        ),
                        need_compact_far_pairs_flag=jnp.asarray(
                            bool(need_compact_far_pairs), dtype=jnp.bool_
                        ),
                        use_dense_interactions_flag=jnp.asarray(
                            bool(use_dense_interactions_for_prepare), dtype=jnp.bool_
                        ),
                    )
                    if suppress_host_side_effects:
                        use_split_build_compiled_bool = bool(allow_split_build)
                        suppress_substage_timing_compiled_bool = True
                    else:
                        use_split_build_compiled_bool = bool(use_split_build_compiled)
                        suppress_substage_timing_compiled_bool = bool(
                            suppress_substage_timing_compiled
                        )
                    if not suppress_host_side_effects:
                        self._refresh_dual_planner_compiled_route_count += 1
                    planner_hint = _RefreshDualPlannerHint(
                        use_split_build=use_split_build_compiled_bool,
                        suppress_substage_timing=suppress_substage_timing_compiled_bool,
                    )
                    if not suppress_host_side_effects:
                        self._refresh_dual_planner_cache[planner_key] = planner_hint
                        self._refresh_dual_planner_compile_count += 1
                else:
                    planner_cache_hit = True
                    if not suppress_host_side_effects:
                        self._refresh_dual_planner_cache_hits += 1
                if not suppress_host_side_effects:
                    self._refresh_dual_planner_execute_count += 1
        planner_allow_steady_timing_bypass = bool(
            self._planner_steady_timing_bypass_enabled
        )
        dual_artifact_timing_callback = _record_dual_artifact_substage
        if (
            planner_hint is not None
            and planner_cache_hit
            and bool(getattr(planner_hint, "suppress_substage_timing", False))
            and planner_allow_steady_timing_bypass
        ):
            dual_artifact_timing_callback = None
            if not suppress_host_side_effects:
                self._refresh_dual_planner_steady_timing_bypass_count += 1
        _record_dual_stage("_refresh_timing_dual_setup_seconds", stage_t0)

        stage_t0 = time.perf_counter()
        geometry_factory = (
            None
            if tree_artifacts.upward.geometry is not None
            else lambda: compute_tree_geometry(
                tree_artifacts.tree,
                tree_artifacts.positions_sorted,
                max_leaf_size=int(tree_artifacts.leaf_cap),
            )
        )
        dual_artifacts, cache_entry = _build_dual_tree_artifacts(
            tree_artifacts.tree,
            tree_artifacts.upward.geometry,
            geometry_factory=geometry_factory,
            theta=theta_val,
            mac_type=mac_type_val,
            dehnen_radius_scale=dehnen_radius_scale,
            cache_key=cache_key,
            cache_entry=(self._interaction_cache if stateful_cache_enabled else None),
            max_pair_queue=self.max_pair_queue,
            pair_process_block=self.pair_process_block,
            traversal_config=runtime_traversal_config,
            retry_logger=(
                None
                if strict_mode_active
                else (
                    record_retry
                    if bool(getattr(self, "_strict_cap_record_enabled", True))
                    else (None if jit_traversal_for_prepare else record_retry)
                )
            ),
            fail_fast=(self.fail_fast or strict_mode_active),
            use_dense_interactions=use_dense_interactions_for_prepare,
            grouped_interactions=grouped_interactions,
            grouped_chunk_size=runtime_m2l_chunk_size,
            need_traversal_result=need_traversal_result,
            need_compact_far_pairs=need_compact_far_pairs,
            need_node_interactions=need_node_interactions,
            precompute_grouped_class_segments=self._should_precompute_grouped_class_segments(
                grouped_chunk_size=runtime_m2l_chunk_size,
                farfield_mode=farfield_mode,
            ),
            grouped_schedule_budget_bytes=self._grouped_schedule_item_budget(),
            allow_split_build=allow_split_build,
            pair_policy=pair_policy,
            policy_state=policy_state,
            jit_traversal=jit_traversal_for_prepare,
            timing_callback=dual_artifact_timing_callback,
            planner_hint=planner_hint,
        )
        if stateful_cache_enabled:
            if bool(getattr(dual_artifacts, "cache_hit", False)):
                if not suppress_host_side_effects:
                    self._interaction_cache_hits += 1
            else:
                if not suppress_host_side_effects:
                    self._interaction_cache_misses += 1
        _record_dual_stage("_refresh_timing_dual_artifact_build_seconds", stage_t0)
        if stateful_cache_enabled:
            self._interaction_cache = cache_entry

        stage_t0 = _stage_now()
        (
            interactions,
            neighbor_list,
            traversal_result,
            compact_far_pairs,
            dense_buffers,
            grouped_buffers,
            grouped_segment_starts,
            grouped_segment_lengths,
            grouped_segment_class_ids,
            grouped_segment_sort_permutation,
            grouped_segment_group_ids,
            grouped_segment_unique_targets,
        ) = self._unpack_dual_tree_artifacts(dual_artifacts)
        if not suppress_host_side_effects:
            far_pair_count_diag = None
            if compact_far_pairs is not None:
                far_pair_count_diag = int(compact_far_pairs.sources.shape[0])
            elif interactions is not None:
                far_pair_count_diag = int(interactions.sources.shape[0])
            total_nodes_diag = int(tree_artifacts.tree.parent.shape[0])
            internal_nodes_diag = int(
                jnp.asarray(tree_artifacts.tree.left_child).shape[0]
            )
            leaf_count_diag = max(0, total_nodes_diag - internal_nodes_diag)
            self._recent_dual_node_count = int(total_nodes_diag)
            self._recent_dual_leaf_count = int(leaf_count_diag)
            self._recent_dual_neighbor_count = int(neighbor_list.neighbors.shape[0])
            self._recent_dual_far_pair_count = (
                0 if far_pair_count_diag is None else int(far_pair_count_diag)
            )
            _prepare_diag(
                "dual-tree done "
                f"neighbor_count={int(neighbor_list.neighbors.shape[0])} "
                f"far_pair_count={far_pair_count_diag} "
                f"compact_far_pairs={compact_far_pairs is not None} "
                f"interactions_present={interactions is not None}"
            )
            _prepare_diag(
                "dual-tree bytes "
                f"neighbors={_format_nbytes(_estimate_payload_nbytes(neighbor_list))} "
                f"compact_far_pairs={_format_nbytes(_estimate_payload_nbytes(compact_far_pairs))} "
                f"interactions={_format_nbytes(_estimate_payload_nbytes(interactions))} "
                f"dense_buffers={_format_nbytes(_estimate_payload_nbytes(dense_buffers))} "
                f"grouped_buffers={_format_nbytes(_estimate_payload_nbytes(grouped_buffers))}"
            )

        strict_streamed_direct_far_pairs = bool(
            strict_mode_active
            and bool(use_compact_streamed_pairs)
            and compact_far_pairs is not None
            and not bool(adaptive_order_active)
            and not bool(mixed_order_farfield_active)
        )
        if strict_streamed_direct_far_pairs:
            src_far = jnp.asarray(compact_far_pairs.sources, dtype=INDEX_DTYPE)
            tgt_far = jnp.asarray(compact_far_pairs.targets, dtype=INDEX_DTYPE)
            far_pairs_coo = _FarPairCOO(
                sources=src_far,
                targets=tgt_far,
                active_count=getattr(compact_far_pairs, "far_pair_count", None),
            )
            far_pairs_by_gear = ((src_far, tgt_far),)
            adaptive_order_for_downward = True
            p_gears_for_downward = (int(tree_artifacts.upward.multipoles.order),)
            if not suppress_host_side_effects:
                self._recent_far_pairs_by_gear_counts = (int(src_far.shape[0]),)
        else:
            far_pair_plan = self._prepare_state_plan_far_pairs_for_downward(
                interactions=interactions,
                traversal_result=traversal_result,
                compact_far_pairs=compact_far_pairs,
                upward=tree_artifacts.upward,
            )
            far_pairs_by_gear = far_pair_plan.far_pairs_by_gear
            far_pairs_coo = far_pair_plan.far_pairs_coo
            adaptive_order_for_downward = far_pair_plan.adaptive_order_for_downward
            p_gears_for_downward = far_pair_plan.p_gears_for_downward
            if not suppress_host_side_effects:
                self._recent_far_pairs_by_gear_counts = (
                    far_pair_plan.recent_far_pairs_by_gear_counts
                )
        _record_dual_stage("_refresh_timing_dual_far_pair_plan_seconds", stage_t0)

        stage_t0 = _stage_now()
        if suppress_host_side_effects and bool(
            getattr(self, "_static_runtime_fixed_sizing", True)
        ):
            runtime_m2l_chunk_size = runtime_m2l_chunk_size
        else:
            runtime_m2l_chunk_size = self._prepare_state_autotune_downward_chunk_size(
                upward=tree_artifacts.upward,
                far_pairs_by_gear=far_pairs_by_gear,
                p_gears_for_downward=p_gears_for_downward,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            )
        if not suppress_host_side_effects:
            self._recent_dual_m2l_chunk_size = (
                0 if runtime_m2l_chunk_size is None else int(runtime_m2l_chunk_size)
            )
        _record_dual_stage("_refresh_timing_dual_m2l_autotune_seconds", stage_t0)

        stage_t0 = _stage_now()
        interactions_for_downward = (
            None
            if strict_streamed_direct_far_pairs and not bool(retain_interactions_active)
            else self._prepare_state_select_interactions_for_downward(
                interactions=interactions,
                far_pairs_coo=far_pairs_coo,
            )
        )
        _record_dual_stage(
            "_refresh_timing_dual_select_interactions_seconds",
            stage_t0,
        )

        stage_t0 = _stage_now()
        downward = self._prepare_downward_with_artifacts(
            tree=tree_artifacts.tree,
            upward=tree_artifacts.upward,
            theta_val=theta_val,
            locals_template=tree_artifacts.locals_template,
            interactions=interactions_for_downward,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            runtime_traversal_config=runtime_traversal_config,
            record_retry=record_retry,
            dense_buffers=dense_buffers,
            grouped_interactions=grouped_interactions,
            grouped_buffers=grouped_buffers,
            grouped_segment_starts=grouped_segment_starts,
            grouped_segment_lengths=grouped_segment_lengths,
            grouped_segment_class_ids=grouped_segment_class_ids,
            grouped_segment_sort_permutation=grouped_segment_sort_permutation,
            grouped_segment_group_ids=grouped_segment_group_ids,
            grouped_segment_unique_targets=grouped_segment_unique_targets,
            farfield_mode=farfield_mode,
            far_pairs_coo=far_pairs_coo,
            far_pairs_by_gear=far_pairs_by_gear,
            adaptive_order=adaptive_order_for_downward,
            p_gears=p_gears_for_downward,
        )
        _record_dual_stage("_refresh_timing_dual_downward_compute_seconds", stage_t0)

        stage_t0 = _stage_now()
        if not suppress_host_side_effects:
            _prepare_diag(
                "downward done "
                f"locals_shape={tuple(int(v) for v in downward.locals.coefficients.shape)} "
                f"interactions_shape={tuple(int(v) for v in downward.interactions.sources.shape)}"
            )
            _prepare_diag(
                "downward bytes "
                f"locals={_format_nbytes(_estimate_payload_nbytes(downward.locals))} "
                f"stored_interactions={_format_nbytes(_estimate_payload_nbytes(downward.interactions))}"
            )
        if not bool(retain_interactions_active):
            # Prepared state only needs the locals; keep a shape-compatible
            # placeholder so downstream code does not accidentally pin the full
            # far-field pair payload after prepare_state completes.
            downward = downward._replace(
                interactions=(
                    _empty_interaction_storage_like(interactions)
                    if interactions is not None
                    else _empty_interaction_storage_for_tree(tree_artifacts.tree)
                )
            )
        interactions_out: Optional[NodeInteractionList]
        if bool(retain_interactions_active):
            interactions_out = interactions
        else:
            interactions_out = None
        _record_dual_stage("_refresh_timing_dual_finalize_seconds", stage_t0)
        if refresh_timing_active:
            residual = max(
                0.0,
                float(time.perf_counter() - dual_total_t0) - float(dual_stage_sum),
            )
            self._refresh_timing_dual_residual_seconds += residual
        return _PrepareStateDualDownwardArtifacts(
            interactions=interactions_out,
            neighbor_list=neighbor_list,
            traversal_result=(
                traversal_result if bool(need_traversal_result) else None
            ),
            compact_far_pairs=(
                compact_far_pairs
                if (
                    bool(adaptive_order_active)
                    or bool(strict_streamed_direct_far_pairs)
                )
                else None
            ),
            downward=downward,
            cache_entry=cache_entry,
        )

    def _prepare_state_extract_adaptive_far_pairs(
        self,
        *,
        traversal_result: Optional[DualTreeWalkResult],
        compact_far_pairs: Optional[CompactTaggedFarPairs],
    ) -> tuple[Array, Array, Array]:
        """Extract tagged far pairs for adaptive-order downward planning."""

        if len(self.p_gears) == 0:
            raise ValueError("adaptive_order=True requires non-empty p_gears")
        if traversal_result is not None:
            far_total = int(traversal_result.far_pair_count)
            far_sources = jnp.asarray(
                traversal_result.interaction_sources[:far_total], dtype=INDEX_DTYPE
            )
            far_targets = jnp.asarray(
                traversal_result.interaction_targets[:far_total], dtype=INDEX_DTYPE
            )
            far_tags = jnp.asarray(
                traversal_result.interaction_tags[:far_total], dtype=INDEX_DTYPE
            )
            return far_sources, far_targets, far_tags
        if compact_far_pairs is not None:
            return (
                jnp.asarray(compact_far_pairs.sources, dtype=INDEX_DTYPE),
                jnp.asarray(compact_far_pairs.targets, dtype=INDEX_DTYPE),
                jnp.asarray(compact_far_pairs.tags, dtype=INDEX_DTYPE),
            )
        raise RuntimeError("adaptive-order traversal requires tagged far-pair payload")

    def _prepare_state_build_streamed_far_pair_plan(
        self,
        *,
        interactions: Optional[NodeInteractionList],
        compact_far_pairs: Optional[CompactTaggedFarPairs],
        upward: TreeUpwardData,
    ) -> tuple[_FarPairCOO, tuple[tuple[Array, Array], ...], tuple[int, ...]]:
        """Build streamed far-pair payloads and per-gear buckets."""

        if compact_far_pairs is not None:
            src_far = jnp.asarray(compact_far_pairs.sources, dtype=INDEX_DTYPE)
            tgt_far = jnp.asarray(compact_far_pairs.targets, dtype=INDEX_DTYPE)
        else:
            if interactions is None:
                raise RuntimeError(
                    "streamed far-pair execution requires interactions or compact far pairs"
                )
            src_far = jnp.asarray(interactions.sources, dtype=INDEX_DTYPE)
            tgt_far = jnp.asarray(interactions.targets, dtype=INDEX_DTYPE)
        active_count = (
            getattr(compact_far_pairs, "far_pair_count", None)
            if compact_far_pairs is not None
            else None
        )
        far_pairs_coo = _FarPairCOO(
            sources=src_far,
            targets=tgt_far,
            active_count=active_count,
        )
        max_order_int = int(upward.multipoles.order)
        strict_fused_device_only_active = bool(
            getattr(self, "_strict_fused_mode_active", False)
        ) and bool(getattr(self, "_strict_fused_device_only", False))
        if (
            self.mixed_order_farfield
            and max_order_int >= 1
            and (not strict_fused_device_only_active)
        ):
            if interactions is None:
                raise RuntimeError(
                    "mixed-order streamed farfield requires node interaction lists"
                )
            min_order_candidate = (
                max_order_int - 1
                if self.mixed_order_min_order is None
                else int(self.mixed_order_min_order)
            )
            min_order_candidate = max(0, min(min_order_candidate, max_order_int))
            p_gears_for_downward, far_pairs_by_gear = _bucket_far_pairs_by_level_split(
                interactions=interactions,
                src_far=src_far,
                tgt_far=tgt_far,
                max_order=max_order_int,
                min_order=min_order_candidate,
            )
        else:
            p_gears_for_downward = (max_order_int,)
            far_pairs_by_gear = ((src_far, tgt_far),)
        return far_pairs_coo, far_pairs_by_gear, p_gears_for_downward

    def _prepare_state_plan_far_pairs_for_downward(
        self,
        *,
        interactions: Optional[NodeInteractionList],
        traversal_result: Optional[DualTreeWalkResult],
        compact_far_pairs: Optional[CompactTaggedFarPairs],
        upward: TreeUpwardData,
    ) -> _PrepareStateFarPairPlan:
        """Prepare far-pair payloads consumed by the downward sweep."""

        far_pairs_by_gear = None
        far_pairs_coo: Optional[_FarPairCOO] = None
        adaptive_order_for_downward = bool(self.adaptive_order)
        p_gears_for_downward = self.p_gears
        recent_counts: tuple[int, ...] = tuple()

        if self.adaptive_order:
            far_sources, far_targets, far_tags = (
                self._prepare_state_extract_adaptive_far_pairs(
                    traversal_result=traversal_result,
                    compact_far_pairs=compact_far_pairs,
                )
            )
            far_pairs_by_gear = bucket_far_pairs_by_tag(
                far_sources,
                far_targets,
                far_tags,
                num_tags=len(self.p_gears),
            )
            recent_counts = tuple(
                int(bucket_src.shape[0]) for bucket_src, _ in far_pairs_by_gear
            )
        elif self.streamed_far_pairs:
            (
                far_pairs_coo,
                far_pairs_by_gear,
                p_gears_for_downward,
            ) = self._prepare_state_build_streamed_far_pair_plan(
                interactions=interactions,
                compact_far_pairs=compact_far_pairs,
                upward=upward,
            )
            adaptive_order_for_downward = True
            recent_counts = tuple(
                int(bucket_src.shape[0]) for bucket_src, _ in far_pairs_by_gear
            )

        return _PrepareStateFarPairPlan(
            far_pairs_by_gear=far_pairs_by_gear,
            far_pairs_coo=far_pairs_coo,
            adaptive_order_for_downward=adaptive_order_for_downward,
            p_gears_for_downward=p_gears_for_downward,
            recent_far_pairs_by_gear_counts=recent_counts,
        )

    def _prepare_state_autotune_downward_chunk_size(
        self,
        *,
        upward: TreeUpwardData,
        far_pairs_by_gear: Optional[tuple[tuple[Array, Array], ...]],
        p_gears_for_downward: tuple[int, ...],
        runtime_m2l_chunk_size: Optional[int],
    ) -> Optional[int]:
        """Choose runtime M2L chunk size for the downward pass."""

        static_runtime_fixed_sizing = bool(
            getattr(self, "_static_runtime_fixed_sizing", True)
        )
        if static_runtime_fixed_sizing:
            return runtime_m2l_chunk_size

        if bool(getattr(self, "_strict_fused_mode_active", False)):
            return runtime_m2l_chunk_size

        if (
            runtime_m2l_chunk_size is not None
            or not bool(self.autotune_m2l_chunk)
            or self.expansion_basis != "solidfmm"
            or jax.default_backend() != "gpu"
            or far_pairs_by_gear is None
            or len(far_pairs_by_gear) == 0
        ):
            return runtime_m2l_chunk_size

        tune_idx = 0
        tune_pair_count = -1
        for idx, (src_bucket, _) in enumerate(far_pairs_by_gear):
            count_i = int(src_bucket.shape[0])
            if count_i > tune_pair_count:
                tune_idx = idx
                tune_pair_count = count_i
        if tune_pair_count <= 0:
            return runtime_m2l_chunk_size

        tune_src, tune_tgt = far_pairs_by_gear[tune_idx]
        tune_order = int(
            upward.multipoles.order
            if tune_idx >= len(p_gears_for_downward)
            else p_gears_for_downward[tune_idx]
        )
        tuned_chunk = self._autotune_runtime_m2l_chunk_size(
            upward=upward,
            src=tune_src,
            tgt=tune_tgt,
            order=tune_order,
            pair_count=tune_pair_count,
        )
        if tuned_chunk is not None:
            return int(tuned_chunk)
        return runtime_m2l_chunk_size

    def _prepare_state_select_interactions_for_downward(
        self,
        *,
        interactions: Optional[NodeInteractionList],
        far_pairs_coo: Optional[_FarPairCOO],
    ) -> Optional[NodeInteractionList]:
        """Choose whether downward should consume node interactions or COO pairs."""

        if (
            self.streamed_far_pairs
            and far_pairs_coo is not None
            and not bool(self.retain_interactions)
        ):
            # Streamed far-pair execution can feed the downward pass directly
            # from COO pairs, so avoid keeping a second node-structured
            # interaction list alive unless the caller asked to retain it.
            return None
        return interactions

    def _prepare_state_nearfield_artifacts(
        self,
        *,
        neighbor_list: NodeNeighborList,
        nearfield_interop: NearfieldInteropData,
        leaf_cap: int,
        num_particles: int,
        cache_entry: Optional[_InteractionCacheEntry],
        allow_stateful_cache: bool,
    ) -> NearfieldPrecomputeArtifacts:
        """Build/reuse near-field precompute artifacts for prepare_state."""
        effective_leaf_cap = (
            int(nearfield_interop.leaf_particle_indices.shape[1])
            if nearfield_interop.leaf_particle_indices is not None
            else int(leaf_cap)
        )
        nearfield_mode_resolved = self._resolve_nearfield_mode(
            num_particles=num_particles
        )
        retain_pair_vectors = (
            nearfield_mode_resolved == "bucketed"
            and self._should_retain_nearfield_pair_vectors(
                num_particles=int(num_particles)
            )
        )
        nearfield_edge_chunk_size_resolved = self._resolve_nearfield_edge_chunk_size(
            num_particles=num_particles,
            nearfield_mode=nearfield_mode_resolved,
        )
        if nearfield_cache_matches(
            cache_entry,
            nearfield_mode=nearfield_mode_resolved,
            nearfield_edge_chunk_size=nearfield_edge_chunk_size_resolved,
            leaf_cap=int(leaf_cap),
            require_pair_vectors=bool(retain_pair_vectors),
        ):
            return nearfield_from_cache(cache_entry)

        nearfield_artifacts = self._prepare_nearfield_precompute_artifacts(
            neighbor_list=neighbor_list,
            nearfield_interop=nearfield_interop,
            leaf_cap=effective_leaf_cap,
            num_particles=num_particles,
            nearfield_mode=nearfield_mode_resolved,
            nearfield_edge_chunk_size=nearfield_edge_chunk_size_resolved,
            retain_pair_vectors=retain_pair_vectors,
        )
        if allow_stateful_cache and cache_entry is not None:
            self._interaction_cache = with_nearfield_cache_artifacts(
                cache_entry,
                artifacts=nearfield_artifacts,
                nearfield_mode=nearfield_mode_resolved,
                nearfield_edge_chunk_size=nearfield_edge_chunk_size_resolved,
                leaf_cap=int(leaf_cap),
            )
        return nearfield_artifacts

    def _update_locals_template_cache_after_prepare(
        self,
        *,
        locals_template: Optional[LocalExpansionData],
        upward: TreeUpwardData,
        max_order: int,
    ) -> None:
        """Update reusable cartesian local-expansion template after prepare_state."""
        if locals_template is not None and self.expansion_basis == "cartesian":
            self._locals_template = LocalExpansionData(
                order=max_order,
                centers=upward.multipoles.centers,
                coefficients=jnp.zeros_like(locals_template.coefficients),
            )
            return
        self._locals_template = None

    def _prepare_nearfield_precompute_artifacts(
        self,
        *,
        neighbor_list: NodeNeighborList,
        nearfield_interop: NearfieldInteropData,
        leaf_cap: int,
        num_particles: int,
        nearfield_mode: Optional[str] = None,
        nearfield_edge_chunk_size: Optional[int] = None,
        retain_pair_vectors: Optional[bool] = None,
    ) -> NearfieldPrecomputeArtifacts:
        """Best-effort precompute of nearfield leaf-pair and scatter artifacts."""
        nearfield_chunk_sort_indices = None
        nearfield_chunk_group_ids = None
        nearfield_chunk_unique_indices = None

        resolved_nearfield_mode = (
            self._resolve_nearfield_mode(num_particles=num_particles)
            if nearfield_mode is None
            else str(nearfield_mode).strip().lower()
        )
        resolved_nearfield_edge_chunk_size = (
            self._resolve_nearfield_edge_chunk_size(
                num_particles=num_particles,
                nearfield_mode=resolved_nearfield_mode,
            )
            if nearfield_edge_chunk_size is None
            else int(nearfield_edge_chunk_size)
        )
        retain_pair_vectors_resolved = (
            self._should_retain_nearfield_pair_vectors(num_particles=int(num_particles))
            if retain_pair_vectors is None
            else bool(retain_pair_vectors)
        )

        should_precompute_scatter = self._should_precompute_nearfield_scatter_schedules(
            num_particles=int(num_particles)
        )
        if not bool(retain_pair_vectors_resolved) and not bool(
            should_precompute_scatter
        ):
            # In large-N minimum-memory GPU runs, prepared evaluation can derive
            # near-field pair vectors on demand from the neighbor list. Avoid
            # materializing enormous edge-index buffers during prepare_state
            # when we are neither retaining them nor building scatter schedules.
            return NearfieldPrecomputeArtifacts(
                target_leaf_ids=None,
                source_leaf_ids=None,
                valid_pairs=None,
                chunk_sort_indices=None,
                chunk_group_ids=None,
                chunk_unique_indices=None,
            )
        if resolved_nearfield_mode != "bucketed":
            return NearfieldPrecomputeArtifacts(
                target_leaf_ids=None,
                source_leaf_ids=None,
                valid_pairs=None,
                chunk_sort_indices=None,
                chunk_group_ids=None,
                chunk_unique_indices=None,
            )

        nearfield_target_leaf_ids = None
        nearfield_valid_pairs = None

        (
            nearfield_target_leaf_ids,
            nearfield_source_leaf_ids,
            nearfield_valid_pairs,
        ) = self._prepare_leaf_neighbor_pairs_safe(
            nearfield_interop=nearfield_interop,
        )

        traced_nearfield_pairs = False
        if nearfield_target_leaf_ids is not None and nearfield_valid_pairs is not None:
            traced_nearfield_pairs = _contains_tracer(
                (nearfield_target_leaf_ids, nearfield_valid_pairs)
            )

        if (
            nearfield_target_leaf_ids is not None
            and nearfield_valid_pairs is not None
            and not traced_nearfield_pairs
            and bool(should_precompute_scatter)
        ):
            (
                nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices,
            ) = self._prepare_bucketed_scatter_schedules_safe(
                nearfield_interop=nearfield_interop,
                target_leaf_ids=nearfield_target_leaf_ids,
                valid_pairs=nearfield_valid_pairs,
                leaf_cap=int(leaf_cap),
                edge_chunk_size=resolved_nearfield_edge_chunk_size,
            )

        return NearfieldPrecomputeArtifacts(
            target_leaf_ids=(
                nearfield_target_leaf_ids if retain_pair_vectors_resolved else None
            ),
            source_leaf_ids=(
                nearfield_source_leaf_ids if retain_pair_vectors_resolved else None
            ),
            valid_pairs=(
                nearfield_valid_pairs if retain_pair_vectors_resolved else None
            ),
            chunk_sort_indices=nearfield_chunk_sort_indices,
            chunk_group_ids=nearfield_chunk_group_ids,
            chunk_unique_indices=nearfield_chunk_unique_indices,
        )

    def _should_retain_nearfield_pair_vectors(
        self,
        *,
        num_particles: int,
    ) -> bool:
        """Decide whether prepared-state near-field pair vectors should be retained."""
        if self.memory_objective == "minimum_memory":
            return False
        if jax.default_backend() != "gpu":
            return True
        return int(num_particles) <= int(_NEARFIELD_GPU_PRECOMPUTE_MAX_PARTICLES)

    def _should_precompute_nearfield_scatter_schedules(
        self,
        *,
        num_particles: int,
    ) -> bool:
        """Decide whether to materialize near-field scatter schedules."""
        if not bool(self.precompute_nearfield_scatter_schedules):
            return False
        if jax.default_backend() != "gpu":
            return True
        # For large GPU runs, schedule materialization is often memory-dominant;
        # keep execution streamed/chunked instead.
        return int(num_particles) <= int(_NEARFIELD_GPU_PRECOMPUTE_MAX_PARTICLES)

    def _prepare_leaf_neighbor_pairs_safe(
        self,
        *,
        nearfield_interop: NearfieldInteropData,
    ) -> tuple[Optional[Array], Optional[Array], Optional[Array]]:
        """Best-effort leaf neighbor pair generation."""
        try:
            return prepare_leaf_neighbor_pairs(
                jnp.asarray(nearfield_interop.node_ranges, dtype=INDEX_DTYPE),
                jnp.asarray(nearfield_interop.leaf_nodes, dtype=INDEX_DTYPE),
                jnp.asarray(nearfield_interop.offsets, dtype=INDEX_DTYPE),
                jnp.asarray(nearfield_interop.neighbors, dtype=INDEX_DTYPE),
                # Keep edge order aligned with neighbor_list so source leaf ids can
                # be derived on demand without storing a second index vector.
                sort_by_source=False,
            )
        except Exception:
            return None, None, None

    def _prepare_bucketed_scatter_schedules_safe(
        self,
        *,
        nearfield_interop: NearfieldInteropData,
        target_leaf_ids: Array,
        valid_pairs: Array,
        leaf_cap: int,
        edge_chunk_size: int,
    ) -> tuple[Optional[Array], Optional[Array], Optional[Array]]:
        """Best-effort bucketed scatter schedule generation."""
        try:
            edge_count = int(target_leaf_ids.shape[0])
            chunk = int(edge_chunk_size)
            chunk_count = (edge_count + chunk - 1) // chunk if edge_count > 0 else 0
            schedule_items = int(chunk_count * chunk * int(leaf_cap))
            if schedule_items > int(_NEARFIELD_SCATTER_SCHEDULE_INT32_ITEM_LIMIT):
                return None, None, None
            schedule_item_cap = self._resolve_nearfield_schedule_item_cap(
                edge_count=edge_count,
                leaf_cap=int(leaf_cap),
                edge_chunk_size=chunk,
            )
            if schedule_items > int(schedule_item_cap):
                return None, None, None
            if nearfield_interop.leaf_particle_indices is not None:
                return prepare_bucketed_scatter_schedules_from_groups(
                    jnp.asarray(
                        nearfield_interop.leaf_particle_indices,
                        dtype=INDEX_DTYPE,
                    ),
                    jnp.asarray(nearfield_interop.leaf_particle_mask, dtype=bool),
                    target_leaf_ids,
                    valid_pairs,
                    edge_chunk_size=chunk,
                )
            return prepare_bucketed_scatter_schedules(
                jnp.asarray(nearfield_interop.node_ranges, dtype=INDEX_DTYPE),
                jnp.asarray(nearfield_interop.leaf_nodes, dtype=INDEX_DTYPE),
                target_leaf_ids,
                valid_pairs,
                max_leaf_size=int(leaf_cap),
                edge_chunk_size=chunk,
            )
        except Exception:
            return None, None, None

    def _resolve_nearfield_schedule_item_cap(
        self,
        *,
        edge_count: int,
        leaf_cap: int,
        edge_chunk_size: int,
    ) -> int:
        """Return the max near-field schedule items allowed for this workload."""
        del edge_count, leaf_cap, edge_chunk_size
        if self.nearfield_schedule_item_cap is not None:
            return int(self.nearfield_schedule_item_cap)

        backend_name = jax.default_backend()
        base_cap = (
            _NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP_GPU
            if backend_name == "gpu"
            else _NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP
        )
        if self.memory_objective == "minimum_memory":
            base_cap = max(1, base_cap // 4)
        elif self.memory_objective == "throughput":
            base_cap = base_cap * 2

        if self.memory_budget_bytes is not None:
            bytes_per_item = 3 * np.dtype(np.int32).itemsize
            budget_limited_cap = max(1, int(self.memory_budget_bytes) // bytes_per_item)
            base_cap = min(base_cap, budget_limited_cap)
        return int(base_cap)

    def _should_precompute_grouped_class_segments(
        self,
        *,
        grouped_chunk_size: Optional[int],
        farfield_mode: str,
    ) -> bool:
        """Decide whether grouped class-major schedules should be materialized."""
        if grouped_chunk_size is None:
            return False
        if str(farfield_mode).strip().lower() != "class_major":
            return False
        if self.precompute_grouped_class_segments is not None:
            return bool(self.precompute_grouped_class_segments)
        return self.memory_objective != "minimum_memory"

    def _grouped_schedule_item_budget(self) -> int:
        """Return max bytes allowed for cached grouped schedule matrices."""
        return int(self.grouped_schedule_budget_bytes)

    def _resolve_target_indices(
        self,
        *,
        target_indices: Optional[Array],
        num_particles: int,
    ) -> Optional[Array]:
        """Validate and normalize optional target particle indices."""
        if target_indices is None:
            return None
        indices = jnp.asarray(target_indices)
        if indices.ndim != 1:
            raise ValueError("target_indices must be a 1D array")
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise ValueError("target_indices must contain integer values")
        if indices.shape[0] == 0:
            return indices.astype(INDEX_DTYPE)
        # Under JAX tracing we cannot materialize min/max as Python ints.
        if isinstance(indices, jax.core.Tracer):
            return indices.astype(INDEX_DTYPE)
        min_idx = int(jnp.min(indices))
        max_idx = int(jnp.max(indices))
        if min_idx < 0 or max_idx >= int(num_particles):
            raise ValueError("target_indices contains out-of-range values")
        return indices.astype(INDEX_DTYPE)

    def _unpack_dual_tree_artifacts(
        self,
        dual_artifacts: _DualTreeArtifacts,
    ) -> tuple[
        NodeInteractionList,
        NodeNeighborList,
        Optional[DualTreeWalkResult],
        Optional[CompactTaggedFarPairs],
        Optional[DenseInteractionBuffers],
        Optional[GroupedInteractionBuffers],
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
    ]:
        """Unpack dual-tree artifacts for downward preparation and state export."""
        return (
            dual_artifacts.interactions,
            dual_artifacts.neighbor_list,
            dual_artifacts.traversal_result,
            dual_artifacts.compact_far_pairs,
            dual_artifacts.dense_buffers,
            dual_artifacts.grouped_buffers,
            dual_artifacts.grouped_segment_starts,
            dual_artifacts.grouped_segment_lengths,
            dual_artifacts.grouped_segment_class_ids,
            dual_artifacts.grouped_segment_sort_permutation,
            dual_artifacts.grouped_segment_group_ids,
            dual_artifacts.grouped_segment_unique_targets,
        )

    def _prepare_downward_with_artifacts(
        self,
        *,
        tree: Tree,
        upward: TreeUpwardData,
        theta_val: float,
        locals_template: Optional[LocalExpansionData],
        interactions: Optional[NodeInteractionList],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        record_retry: Callable[[DualTreeRetryEvent], None],
        dense_buffers: Optional[DenseInteractionBuffers],
        grouped_interactions: bool,
        grouped_buffers: Optional[GroupedInteractionBuffers],
        grouped_segment_starts: Optional[Array],
        grouped_segment_lengths: Optional[Array],
        grouped_segment_class_ids: Optional[Array],
        grouped_segment_sort_permutation: Optional[Array],
        grouped_segment_group_ids: Optional[Array],
        grouped_segment_unique_targets: Optional[Array],
        farfield_mode: str,
        far_pairs_coo: Optional[_FarPairCOO] = None,
        far_pairs_by_gear: Optional[tuple[tuple[Array, Array], ...]] = None,
        adaptive_order: bool = False,
        p_gears: tuple[int, ...] = tuple(),
    ) -> TreeDownwardData:
        """Prepare downward sweep using precomputed interaction artifacts."""
        return self.prepare_downward_sweep(
            tree,
            upward,
            theta=theta_val,
            initial_locals=locals_template,
            interactions=interactions,
            m2l_chunk_size=runtime_m2l_chunk_size,
            l2l_chunk_size=runtime_l2l_chunk_size,
            traversal_config=runtime_traversal_config,
            retry_logger=record_retry,
            dense_buffers=dense_buffers,
            grouped_interactions=grouped_interactions,
            grouped_buffers=grouped_buffers,
            grouped_segment_starts=grouped_segment_starts,
            grouped_segment_lengths=grouped_segment_lengths,
            grouped_segment_class_ids=grouped_segment_class_ids,
            grouped_segment_sort_permutation=grouped_segment_sort_permutation,
            grouped_segment_group_ids=grouped_segment_group_ids,
            grouped_segment_unique_targets=grouped_segment_unique_targets,
            farfield_mode=farfield_mode,
            far_pairs_coo=far_pairs_coo,
            far_pairs_by_gear=far_pairs_by_gear,
            adaptive_order=adaptive_order,
            p_gears=p_gears,
        )

    def _prepare_state_dual_and_downward_strict_streamed_fast(
        self,
        *,
        tree_artifacts: _PrepareStateTreeUpwardArtifacts,
        theta_val: float,
        mac_type_val: MACType,
        dehnen_radius_scale: float,
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        record_retry: Callable[[DualTreeRetryEvent], None],
        farfield_mode: str,
        retain_interactions: bool = False,
        suppress_host_side_effects: bool = False,
    ) -> _PrepareStateDualDownwardArtifacts:
        """Strict static fast path with compact streamed far-pairs only."""

        geometry_factory = (
            None
            if tree_artifacts.upward.geometry is not None
            else lambda: compute_tree_geometry(
                tree_artifacts.tree,
                tree_artifacts.positions_sorted,
                max_leaf_size=int(tree_artifacts.leaf_cap),
            )
        )
        dual_artifacts, cache_entry = _build_dual_tree_artifacts(
            tree_artifacts.tree,
            tree_artifacts.upward.geometry,
            geometry_factory=geometry_factory,
            theta=theta_val,
            mac_type=mac_type_val,
            dehnen_radius_scale=dehnen_radius_scale,
            cache_key=None,
            cache_entry=None,
            max_pair_queue=self.max_pair_queue,
            pair_process_block=self.pair_process_block,
            traversal_config=runtime_traversal_config,
            retry_logger=None,
            fail_fast=True,
            use_dense_interactions=False,
            grouped_interactions=False,
            grouped_chunk_size=runtime_m2l_chunk_size,
            need_traversal_result=False,
            need_compact_far_pairs=True,
            need_node_interactions=False,
            precompute_grouped_class_segments=False,
            grouped_schedule_budget_bytes=self._grouped_schedule_item_budget(),
            allow_split_build=True,
            pair_policy=None,
            policy_state=None,
            jit_traversal=True,
            timing_callback=None,
            planner_hint=_RefreshDualPlannerHint(
                use_split_build=True,
                suppress_substage_timing=True,
            ),
        )
        (
            interactions,
            neighbor_list,
            traversal_result,
            compact_far_pairs,
            dense_buffers,
            grouped_buffers,
            grouped_segment_starts,
            grouped_segment_lengths,
            grouped_segment_class_ids,
            grouped_segment_sort_permutation,
            grouped_segment_group_ids,
            grouped_segment_unique_targets,
        ) = self._unpack_dual_tree_artifacts(dual_artifacts)
        del (
            interactions,
            traversal_result,
            dense_buffers,
            grouped_buffers,
            grouped_segment_starts,
            grouped_segment_lengths,
            grouped_segment_class_ids,
            grouped_segment_sort_permutation,
            grouped_segment_group_ids,
            grouped_segment_unique_targets,
        )
        if compact_far_pairs is None:
            raise RuntimeError(
                "strict streamed fast path requires compact far-pair artifacts"
            )
        if not suppress_host_side_effects:
            total_nodes_diag = int(tree_artifacts.tree.parent.shape[0])
            internal_nodes_diag = int(
                jnp.asarray(tree_artifacts.tree.left_child).shape[0]
            )
            leaf_count_diag = max(0, total_nodes_diag - internal_nodes_diag)
            self._recent_dual_node_count = int(total_nodes_diag)
            self._recent_dual_leaf_count = int(leaf_count_diag)
            self._recent_dual_neighbor_count = int(neighbor_list.neighbors.shape[0])
            self._recent_dual_far_pair_count = int(compact_far_pairs.sources.shape[0])

        src_far = jnp.asarray(compact_far_pairs.sources, dtype=INDEX_DTYPE)
        tgt_far = jnp.asarray(compact_far_pairs.targets, dtype=INDEX_DTYPE)
        far_pairs_coo = _FarPairCOO(
            sources=src_far,
            targets=tgt_far,
            active_count=getattr(compact_far_pairs, "far_pair_count", None),
        )
        far_pairs_by_gear: tuple[tuple[Array, Array], ...] = ((src_far, tgt_far),)
        p_gears_for_downward = (int(tree_artifacts.upward.multipoles.order),)
        if not suppress_host_side_effects:
            self._recent_far_pairs_by_gear_counts = (int(src_far.shape[0]),)

        if suppress_host_side_effects and bool(
            getattr(self, "_static_runtime_fixed_sizing", True)
        ):
            runtime_m2l_chunk_size = runtime_m2l_chunk_size
        else:
            runtime_m2l_chunk_size = self._prepare_state_autotune_downward_chunk_size(
                upward=tree_artifacts.upward,
                far_pairs_by_gear=far_pairs_by_gear,
                p_gears_for_downward=p_gears_for_downward,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            )
        if not suppress_host_side_effects:
            self._recent_dual_m2l_chunk_size = (
                0 if runtime_m2l_chunk_size is None else int(runtime_m2l_chunk_size)
            )
        downward = self._prepare_downward_with_artifacts(
            tree=tree_artifacts.tree,
            upward=tree_artifacts.upward,
            theta_val=theta_val,
            locals_template=tree_artifacts.locals_template,
            interactions=None,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            runtime_traversal_config=runtime_traversal_config,
            record_retry=record_retry,
            dense_buffers=None,
            grouped_interactions=False,
            grouped_buffers=None,
            grouped_segment_starts=None,
            grouped_segment_lengths=None,
            grouped_segment_class_ids=None,
            grouped_segment_sort_permutation=None,
            grouped_segment_group_ids=None,
            grouped_segment_unique_targets=None,
            farfield_mode=farfield_mode,
            far_pairs_coo=far_pairs_coo,
            far_pairs_by_gear=far_pairs_by_gear,
            adaptive_order=True,
            p_gears=p_gears_for_downward,
        )
        if not bool(retain_interactions):
            downward = downward._replace(
                interactions=_empty_interaction_storage_for_tree(tree_artifacts.tree)
            )
        return _PrepareStateDualDownwardArtifacts(
            interactions=None,
            neighbor_list=neighbor_list,
            traversal_result=None,
            compact_far_pairs=compact_far_pairs,
            downward=downward,
            cache_entry=cache_entry,
        )

    def _resolve_nearfield_mode(self, *, num_particles: int) -> str:
        """Resolve near-field execution mode from configured policy."""
        if self._is_large_n_gpu_production_profile():
            if (
                not bool(self._explicit_nearfield_mode)
                and jax.default_backend() == "gpu"
                and int(num_particles) < 262_144
            ):
                return "baseline"
            return "bucketed"
        mode = str(self.nearfield_mode).strip().lower()
        if mode != "auto":
            return mode
        backend = jax.default_backend()
        large_gpu = (
            backend == "gpu"
            and int(num_particles) >= 262_144
            and str(self.preset).strip().lower() == "large_n_gpu"
            and str(self.expansion_basis).strip().lower() == "solidfmm"
        )
        large_cpu = (
            backend == "cpu"
            and int(num_particles) >= _NEARFIELD_BUCKETED_CPU_PARTICLE_THRESHOLD
        )
        if large_gpu:
            return "bucketed"
        if (
            large_cpu
            and self.preset == "fast"
            and self.expansion_basis == "solidfmm"
            and self.mac_type == "dehnen"
        ):
            return "bucketed"
        return "baseline"

    def _resolve_nearfield_edge_chunk_size(
        self,
        *,
        num_particles: int,
        nearfield_mode: str,
    ) -> int:
        """Resolve near-field edge chunk size with large-N auto policy."""
        base_chunk = int(self.nearfield_edge_chunk_size)
        if base_chunk <= 0:
            raise ValueError("nearfield_edge_chunk_size must be positive")
        mode = str(self.nearfield_mode).strip().lower()
        auto_policy_enabled = mode == "auto" or (
            self._is_large_n_gpu_production_profile()
            and not bool(self._explicit_nearfield_mode)
        )
        if (not auto_policy_enabled) or str(
            nearfield_mode
        ).strip().lower() != "bucketed":
            return base_chunk

        n = int(num_particles)
        if jax.default_backend() == "gpu":
            if (
                str(self.preset).strip().lower() == "large_n_gpu"
                and str(self.expansion_basis).strip().lower() == "solidfmm"
            ):
                if n >= 262_144:
                    return max(base_chunk, 256)
            return base_chunk

        if n >= 2_000_000:
            return max(base_chunk, _NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_XL)
        if n >= 1_000_000:
            return max(base_chunk, _NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_LARGE)
        if n >= _NEARFIELD_BUCKETED_CPU_PARTICLE_THRESHOLD:
            return max(base_chunk, _NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_MEDIUM)
        return base_chunk

    # ------------------------------------------------------------------
    # Expansion construction up to a given order
    # order=0: monopole, order=1: +dipole, order=2: +quadrupole
    # order=3: +octupole, order=4: +hexadecapole
    # ------------------------------------------------------------------

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def compute_expansion(
        positions: Array,
        masses: Array,
        order: int = 1,
    ) -> MultipoleExpansion:
        """Return the multipole expansion via the shared reference helper."""

        return reference_compute_expansion(positions, masses, order=order)

    # ------------------------------------------------------------------
    # Expansion evaluation up to a given order
    # ------------------------------------------------------------------

    @jaxtyped(typechecker=beartype)
    def evaluate_expansion(
        self: "FastMultipoleMethod",
        expansion: MultipoleExpansion,
        order: int = 1,
        eval_point: Optional[Array] = None,
    ) -> Array:
        """Evaluate multipole expansions via the shared reference helper."""

        return reference_evaluate_expansion(
            expansion,
            order=order,
            eval_point=eval_point,
            G=self.G,
            softening=self.softening,
        )

    # ------------------------------------------------------------------
    # Direct summation fallback (for validation / small N)
    # ------------------------------------------------------------------

    @jaxtyped(typechecker=beartype)
    def direct_sum(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        eval_point: Array,
        eval_mass: float = 0.0,
    ) -> Array:
        """Compute direct-sum accelerations for diagnostic comparisons."""

        _ = eval_mass  # Unused but preserved for backwards compatibility.
        return reference_direct_sum(
            positions,
            masses,
            eval_point,
            G=self.G,
            softening=self.softening,
        )

    def _resolve_upward_num_levels(self, tree: Tree) -> Optional[int]:
        """Return the concrete (unpadded) tree depth for the M2M level loop.

        When ``tree`` is concrete (full prepare / template build) this computes
        the actual depth and stashes it on ``self``. When ``tree`` is a JAX
        tracer (the fused device-resident refresh) its array values are
        unavailable, so we return the previously stashed value. The stash is
        populated by the eager full-prepare that always precedes the traced
        refresh, so the hot path gets the concrete depth. Returns ``None`` only
        if nothing concrete has been seen yet (callers then fall back to the
        padded shape-derived depth, which is correct, just slower)."""
        probe = getattr(tree, "parent", None)
        if probe is None or isinstance(probe, jax.core.Tracer):
            return self._static_upward_num_levels
        try:
            levels = get_node_levels(tree)
            if isinstance(levels, jax.core.Tracer):
                return self._static_upward_num_levels
            self._static_upward_num_levels = int(jnp.max(levels)) + 1
        except Exception:
            return self._static_upward_num_levels
        return self._static_upward_num_levels

    def prepare_upward_sweep(
        self: "FastMultipoleMethod",
        tree: Tree,
        positions_sorted: Array,
        masses_sorted: Array,
        *,
        max_order: int = 2,
        center_mode: str = "com",
        explicit_centers: Optional[Array] = None,
        max_leaf_size: Optional[int] = None,
        precomputed_geometry: Optional[Any] = None,
        defer_geometry: bool = False,
    ) -> TreeUpwardData:
        """Bundle geometry, raw moments, and packed expansions for a tree."""
        self._ensure_execution_backend_supported(tree=tree)

        if self.expansion_basis == "solidfmm":

            def _record_upward_stage(name: str, elapsed: float) -> None:
                if not bool(getattr(self, "_refresh_timing_active", False)):
                    return
                attr_by_name = {
                    "geometry": "_refresh_timing_upward_geometry_seconds",
                    "mass_moments": "_refresh_timing_upward_mass_moments_seconds",
                    "p2m": "_refresh_timing_upward_p2m_seconds",
                    "m2m": "_refresh_timing_upward_m2m_seconds",
                    "source_motion": "_refresh_timing_upward_source_motion_seconds",
                }
                attr = attr_by_name.get(str(name))
                if attr is None:
                    return
                setattr(self, attr, float(getattr(self, attr, 0.0)) + float(elapsed))

            complex_upward = prepare_solidfmm_complex_upward_sweep(
                tree,
                positions_sorted,
                masses_sorted,
                max_order=max_order,
                center_mode=center_mode,
                explicit_centers=explicit_centers,
                max_leaf_size=max_leaf_size,
                leaf_batch_size=self.upward_leaf_batch_size,
                rotation=self.complex_rotation,
                precomputed_geometry=precomputed_geometry,
                upward_timing_callback=_record_upward_stage,
                defer_geometry=bool(defer_geometry),
                static_num_levels=self._resolve_upward_num_levels(tree),
            )

            multipoles = NodeMultipoleData(
                order=int(complex_upward.multipoles.order),
                centers=complex_upward.multipoles.centers,
                moments=None,  # type: ignore[arg-type]
                packed=complex_upward.multipoles.packed,
                component_matrix=None,
                source_motion_packed=complex_upward.multipoles.source_motion_packed,
            )

            return TreeUpwardData(
                geometry=complex_upward.geometry,
                mass_moments=complex_upward.mass_moments,
                multipoles=multipoles,
            )

        return prepare_tree_upward_sweep(
            tree,
            positions_sorted,
            masses_sorted,
            max_order=max_order,
            center_mode=center_mode,
            explicit_centers=explicit_centers,
            precomputed_geometry=precomputed_geometry,
        )

    def run_downward_sweep(
        self: "FastMultipoleMethod",
        tree: Tree,
        multipoles: NodeMultipoleData,
        interactions: NodeInteractionList,
        *,
        initial_locals: Optional[LocalExpansionData] = None,
        m2l_chunk_size: Optional[int] = None,
        dense_buffers: Optional[DenseInteractionBuffers] = None,
    ) -> LocalExpansionData:
        """Execute an M2L+L2L pass for the provided multipoles."""

        return run_tree_downward_sweep(
            tree,
            multipoles,
            interactions,
            initial_locals=initial_locals,
            m2l_chunk_size=m2l_chunk_size,
            dense_buffers=dense_buffers,
        )

    def prepare_downward_sweep(
        self: "FastMultipoleMethod",
        tree: Tree,
        upward_data: TreeUpwardData,
        *,
        theta: Optional[float] = None,
        mac_type: Optional[MACType] = None,
        initial_locals: Optional[LocalExpansionData] = None,
        interactions: Optional[NodeInteractionList] = None,
        m2l_chunk_size: Optional[int] = None,
        l2l_chunk_size: Optional[int] = None,
        traversal_config: Optional[DualTreeTraversalConfig] = None,
        dense_buffers: Optional[DenseInteractionBuffers] = None,
        retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
        grouped_interactions: bool = False,
        grouped_buffers: Optional[GroupedInteractionBuffers] = None,
        grouped_segment_starts: Optional[Array] = None,
        grouped_segment_lengths: Optional[Array] = None,
        grouped_segment_class_ids: Optional[Array] = None,
        grouped_segment_sort_permutation: Optional[Array] = None,
        grouped_segment_group_ids: Optional[Array] = None,
        grouped_segment_unique_targets: Optional[Array] = None,
        farfield_mode: str = "pair_grouped",
        dehnen_radius_scale: Optional[float] = None,
        far_pairs_coo: Optional[_FarPairCOO] = None,
        far_pairs_by_gear: Optional[tuple[tuple[Array, Array], ...]] = None,
        adaptive_order: Optional[bool] = None,
        p_gears: Optional[tuple[int, ...]] = None,
    ) -> TreeDownwardData:
        """Build interactions and locals needed for the downward sweep."""
        self._ensure_execution_backend_supported(tree=tree)

        theta_val = float(self.theta if theta is None else theta)
        mac_type_val = self.mac_type if mac_type is None else mac_type
        dehnen_scale_val = float(
            self.dehnen_radius_scale
            if dehnen_radius_scale is None
            else dehnen_radius_scale
        )
        config = traversal_config if traversal_config is not None else None
        if config is None:
            config = self.traversal_config
        retry_callback = (
            retry_logger if retry_logger is not None else self.interaction_retry_logger
        )
        if self.expansion_basis == "solidfmm":
            adaptive_order_val = (
                self.adaptive_order if adaptive_order is None else bool(adaptive_order)
            )
            p_gears_val = (
                self.p_gears if p_gears is None else tuple(int(v) for v in p_gears)
            )
            timing_recorder = None
            sync_substage_timing = str(
                os.environ.get("JACCPOT_REFRESH_TIMING_SYNC_SUBSTAGES", "0")
            ).strip().lower() in {"1", "true", "yes", "on"}
            if bool(getattr(self, "_refresh_timing_active", False)) and bool(
                sync_substage_timing
            ):

                def timing_recorder(attr: str, elapsed: float) -> None:
                    setattr(self, attr, float(getattr(self, attr, 0.0)) + elapsed)

            return _prepare_solidfmm_downward_sweep(
                tree,
                upward_data,
                theta=theta_val,
                mac_type=mac_type_val,
                initial_locals=initial_locals,
                interactions=interactions,
                m2l_chunk_size=m2l_chunk_size,
                l2l_chunk_size=l2l_chunk_size,
                complex_rotation=self.complex_rotation,
                basis_mode=self._solidfmm_basis_mode(),
                m2l_impl=self.m2l_impl,
                retry_logger=retry_callback,
                traversal_config=config,
                dense_buffers=dense_buffers,
                grouped_interactions=grouped_interactions,
                grouped_buffers=grouped_buffers,
                grouped_segment_starts=grouped_segment_starts,
                grouped_segment_lengths=grouped_segment_lengths,
                grouped_segment_class_ids=grouped_segment_class_ids,
                grouped_segment_sort_permutation=grouped_segment_sort_permutation,
                grouped_segment_group_ids=grouped_segment_group_ids,
                grouped_segment_unique_targets=grouped_segment_unique_targets,
                farfield_mode=farfield_mode,
                far_pairs_coo=far_pairs_coo,
                far_pairs_by_gear=far_pairs_by_gear,
                adaptive_order=adaptive_order_val,
                p_gears=p_gears_val,
                dehnen_radius_scale=dehnen_scale_val,
                use_pallas=self.use_pallas,
                timing_recorder=timing_recorder,
            )

        return prepare_tree_downward_sweep(
            tree,
            upward_data,
            theta=theta_val,
            mac_type=mac_type_val,
            initial_locals=initial_locals,
            interactions=interactions,
            m2l_chunk_size=m2l_chunk_size,
            retry_logger=retry_callback,
            traversal_config=config,
            max_pair_queue=self.max_pair_queue,
            process_block=self.pair_process_block,
            dense_buffers=dense_buffers,
        )

    @jaxtyped(typechecker=beartype)
    def compute_accelerations(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        *,
        target_indices: Optional[Array] = None,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        return_potential: bool = False,
        theta: Optional[float] = None,
        jit_tree: Optional[bool] = None,
        refine_local: Optional[bool] = None,
        max_refine_levels: Optional[int] = None,
        aspect_threshold: Optional[float] = None,
        jit_traversal: Optional[bool] = None,
        reuse_prepared_state: bool = False,
        max_acc_derivative_order: int = 0,
    ) -> Union[
        Array,
        Tuple[Array, Array],
        Tuple[Array, PackedAccelerationDerivatives],
        Tuple[Array, Array, PackedAccelerationDerivatives],
    ]:
        """Run the full FMM pipeline for particle accelerations.

        Parameters
        ----------
        positions : Array
            Source and target particle positions.
        masses : Array
            Particle masses aligned with ``positions``.
        target_indices : Optional[Array]
            Optional 1D index array selecting which target-particle outputs to
            return. All particles are still used as source masses.
        bounds : Optional[Tuple[Array, Array]]
            Optional explicit domain bounds used during tree construction.
        leaf_size : int
            Target maximum particle count per leaf for the prepared tree.
        max_order : int
            Multipole/local expansion order used for the upward and downward
            passes.
        return_potential : bool
            When ``True``, return a tuple ``(accelerations, potentials)``.
        theta : Optional[float]
            Optional per-call MAC opening angle override.
        jit_tree : Optional[bool]
            When ``True``, specialise tree construction via JIT to amortise
            repeated builds for consistent tree sizes.
        refine_local : Optional[bool]
            Override the fixed-depth builder's local refinement toggle when
            ``tree_build_mode`` is ``"fixed_depth"``.
        max_refine_levels : Optional[int]
            Maximum local refinement iterations passed to the builder.
        aspect_threshold : Optional[float]
            Aspect ratio threshold that triggers additional splits in the
            refinement pass.
        jit_traversal : Optional[bool]
            When ``True``, evaluate the traversal/evaluation path with the
            compiled implementation for improved throughput.
        reuse_prepared_state : bool
            Reuse the most recent prepared state when identical array objects
            and preparation parameters are provided.

        Returns
        -------
        Union[Array, Tuple[Array, Array]]
            Accelerations for all particles or selected targets. When
            ``return_potential`` is ``True``, also returns the potential.
        """

        cache_key: Optional[tuple[Any, ...]] = None
        state: Optional[FMMPreparedState] = None
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        if _contains_tracer((positions_arr, masses_arr)):
            if bool(return_potential):
                raise NotImplementedError(
                    "traced compute_accelerations fallback does not support return_potential=True"
                )
            if int(max_acc_derivative_order) != 0:
                raise NotImplementedError(
                    "traced compute_accelerations fallback does not support acceleration derivatives"
                )
            target_positions = (
                positions_arr
                if target_indices is None
                else jnp.asarray(
                    positions_arr[jnp.asarray(target_indices, dtype=INDEX_DTYPE),],
                    dtype=positions_arr.dtype,
                )
            )
            return jax.vmap(
                lambda eval_point: reference_direct_sum(
                    positions_arr,
                    masses_arr,
                    eval_point,
                    G=self.G,
                    softening=self.softening,
                )
            )(target_positions)
        if reuse_prepared_state:
            if bounds is None:
                bounds_key: tuple[Any, ...] = ("none",)
            else:
                bounds_key = ("set", id(bounds[0]), id(bounds[1]))
            cache_key = (
                positions_arr.shape,
                str(positions_arr.dtype),
                masses_arr.shape,
                str(masses_arr.dtype),
                bounds_key,
                int(leaf_size),
                int(max_order),
                None if theta is None else float(theta),
                None if jit_tree is None else bool(jit_tree),
                None if refine_local is None else bool(refine_local),
                None if max_refine_levels is None else int(max_refine_levels),
                None if aspect_threshold is None else float(aspect_threshold),
            )
            state = self._prepared_state_cache_lookup(
                key=cache_key,
                positions=positions_arr,
                masses=masses_arr,
            )

        if state is None:
            state = self.prepare_state(
                positions,
                masses,
                bounds=bounds,
                leaf_size=leaf_size,
                max_order=max_order,
                theta=theta,
                jit_tree=jit_tree,
                refine_local=refine_local,
                max_refine_levels=max_refine_levels,
                aspect_threshold=aspect_threshold,
            )
            if reuse_prepared_state and cache_key is not None:
                self._prepared_state_cache_store(
                    key=cache_key,
                    positions=positions_arr,
                    masses=masses_arr,
                    state=state,
                )

        jit_traversal_flag = (
            self._jit_traversal_default
            if jit_traversal is None
            else bool(jit_traversal)
        )

        evaluation = self.evaluate_prepared_state(
            state,
            target_indices=target_indices,
            return_potential=return_potential,
            jit_traversal=jit_traversal_flag,
            max_acc_derivative_order=max_acc_derivative_order,
        )
        return evaluation

    @jaxtyped(typechecker=beartype)
    def compute_accelerations_and_jerk(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        theta: Optional[float] = None,
        jit_tree: Optional[bool] = None,
        refine_local: Optional[bool] = None,
        max_refine_levels: Optional[int] = None,
        aspect_threshold: Optional[float] = None,
        jit_traversal: Optional[bool] = None,
        reuse_prepared_state: bool = False,
        jerk_mode: str = "fast_approx",
        jerk_fd_dt: float = 1e-3,
    ) -> tuple[Array, Array]:
        """Run FMM and return accelerations plus jerk estimates.

        Jerk combines:
        - exact near-field pairwise jerk from source/target velocities,
        - far-field convective term from acceleration Jacobian times target velocity.
        """
        cache_key: Optional[tuple[Any, ...]] = None
        state: Optional[FMMPreparedState] = None
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        if reuse_prepared_state:
            if bounds is None:
                bounds_key: tuple[Any, ...] = ("none",)
            else:
                bounds_key = ("set", id(bounds[0]), id(bounds[1]))
            cache_key = (
                positions_arr.shape,
                str(positions_arr.dtype),
                masses_arr.shape,
                str(masses_arr.dtype),
                bounds_key,
                int(leaf_size),
                int(max_order),
                None if theta is None else float(theta),
                None if jit_tree is None else bool(jit_tree),
                None if refine_local is None else bool(refine_local),
                None if max_refine_levels is None else int(max_refine_levels),
                None if aspect_threshold is None else float(aspect_threshold),
            )
            state = self._prepared_state_cache_lookup(
                key=cache_key,
                positions=positions_arr,
                masses=masses_arr,
            )

        if state is None:
            state = self.prepare_state(
                positions,
                masses,
                bounds=bounds,
                leaf_size=leaf_size,
                max_order=max_order,
                theta=theta,
                jit_tree=jit_tree,
                refine_local=refine_local,
                max_refine_levels=max_refine_levels,
                aspect_threshold=aspect_threshold,
            )
            if reuse_prepared_state and cache_key is not None:
                self._prepared_state_cache_store(
                    key=cache_key,
                    positions=positions_arr,
                    masses=masses_arr,
                    state=state,
                )

        jit_traversal_flag = (
            self._jit_traversal_default
            if jit_traversal is None
            else bool(jit_traversal)
        )
        return self.evaluate_prepared_state_with_jerk(
            state,
            velocities,
            target_indices=target_indices,
            jit_traversal=jit_traversal_flag,
            jerk_mode=jerk_mode,
            jerk_fd_dt=jerk_fd_dt,
        )

    @jaxtyped(typechecker=beartype)
    def compute_accelerations_with_time_derivatives(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        theta: Optional[float] = None,
        jit_tree: Optional[bool] = None,
        refine_local: Optional[bool] = None,
        max_refine_levels: Optional[int] = None,
        aspect_threshold: Optional[float] = None,
        jit_traversal: Optional[bool] = None,
        reuse_prepared_state: bool = False,
        max_time_derivative_order: int = 1,
        mode: str = "accurate",
    ) -> tuple[Array, tuple[Array, ...]]:
        """Run FMM and return accelerations plus time derivatives up to order K."""
        cache_key: Optional[tuple[Any, ...]] = None
        state: Optional[FMMPreparedState] = None
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        if reuse_prepared_state:
            if bounds is None:
                bounds_key: tuple[Any, ...] = ("none",)
            else:
                bounds_key = ("set", id(bounds[0]), id(bounds[1]))
            cache_key = (
                positions_arr.shape,
                str(positions_arr.dtype),
                masses_arr.shape,
                str(masses_arr.dtype),
                bounds_key,
                int(leaf_size),
                int(max_order),
                None if theta is None else float(theta),
                None if jit_tree is None else bool(jit_tree),
                None if refine_local is None else bool(refine_local),
                None if max_refine_levels is None else int(max_refine_levels),
                None if aspect_threshold is None else float(aspect_threshold),
            )
            state = self._prepared_state_cache_lookup(
                key=cache_key,
                positions=positions_arr,
                masses=masses_arr,
            )

        if state is None:
            state = self.prepare_state(
                positions,
                masses,
                bounds=bounds,
                leaf_size=leaf_size,
                max_order=max_order,
                theta=theta,
                jit_tree=jit_tree,
                refine_local=refine_local,
                max_refine_levels=max_refine_levels,
                aspect_threshold=aspect_threshold,
            )
            if reuse_prepared_state and cache_key is not None:
                self._prepared_state_cache_store(
                    key=cache_key,
                    positions=positions_arr,
                    masses=masses_arr,
                    state=state,
                )

        jit_traversal_flag = (
            self._jit_traversal_default
            if jit_traversal is None
            else bool(jit_traversal)
        )
        return self.evaluate_prepared_state_with_time_derivatives(
            state,
            velocities,
            target_indices=target_indices,
            jit_traversal=jit_traversal_flag,
            max_time_derivative_order=max_time_derivative_order,
            mode=mode,
        )
        if target_indices is None and not _contains_tracer((state, evaluation)):
            accelerations_out = evaluation[0] if return_potential else evaluation
            inv = jnp.asarray(state.inverse_permutation)
            sorted_idx = jnp.argsort(inv)
            accelerations_sorted = jnp.asarray(accelerations_out)[sorted_idx]
            self._last_force_scale_nodes = (
                self._compute_node_force_scale_from_sorted_acc(
                    tree=state.tree,
                    accelerations_sorted=accelerations_sorted,
                    reduction=self._force_scale_reduction_mode(),
                )
            )
        return evaluation

    @jaxtyped(typechecker=beartype)
    def prepare_state(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        theta: Optional[float] = None,
        jit_tree: Optional[bool] = None,
        refine_local: Optional[bool] = None,
        max_refine_levels: Optional[int] = None,
        aspect_threshold: Optional[float] = None,
        runtime_overrides_override: Optional[_RuntimeExecutionOverrides] = None,
        fused_device_mode: bool = False,
    ) -> PreparedStateLike:
        """Precompute tree and interaction data for repeated evaluations.

        When ``tree_build_mode`` is ``"fixed_depth"`` the optional
        ``refine_local``, ``max_refine_levels``, and ``aspect_threshold``
        arguments control the host-side leaf refinement pass.
        """

        self._validate_prepare_state_request(
            leaf_size=int(leaf_size),
            max_order=int(max_order),
        )

        refine_local_val = (
            self.refine_local if refine_local is None else bool(refine_local)
        )
        max_refine_levels_val = (
            self.max_refine_levels
            if max_refine_levels is None
            else int(max_refine_levels)
        )
        aspect_threshold_val = (
            self.aspect_threshold
            if aspect_threshold is None
            else float(aspect_threshold)
        )

        collected_retries: list[DualTreeRetryEvent] = []

        def record_retry(event: DualTreeRetryEvent) -> None:
            collected_retries.append(event)
            if self.interaction_retry_logger is not None:
                self.interaction_retry_logger(event)

        input_t0 = time.perf_counter()
        positions_arr, masses_arr, input_dtype = self._prepare_state_input_arrays(
            positions,
            masses,
        )
        if bool(getattr(self, "_refresh_timing_active", False)):
            self._refresh_timing_input_seconds += time.perf_counter() - input_t0
        allow_stateful_cache = not _contains_tracer((positions_arr, masses_arr))

        runtime_overrides = runtime_overrides_override
        if runtime_overrides is None:
            runtime_overrides = self._resolve_runtime_execution_overrides(
                num_particles=int(positions_arr.shape[0]),
            )
        runtime_traversal_config = runtime_overrides.traversal_config
        runtime_m2l_chunk_size = runtime_overrides.m2l_chunk_size
        runtime_l2l_chunk_size = runtime_overrides.l2l_chunk_size
        grouped_interactions = runtime_overrides.grouped_interactions
        farfield_mode = runtime_overrides.farfield_mode
        upward_center_mode = runtime_overrides.center_mode
        if refine_local is None and runtime_overrides.refine_local_override is not None:
            refine_local_val = bool(runtime_overrides.refine_local_override)
        if not allow_stateful_cache:
            runtime_traversal_config = self._resolve_tracing_traversal_config(
                traversal_config=runtime_traversal_config,
            )

        theta_val = float(self.theta if theta is None else theta)
        mac_type_val = self._base_mac_type()

        if can_use_large_n_prepare_path(
            self,
            positions_arr=positions_arr,
            masses_arr=masses_arr,
            allow_stateful_cache=allow_stateful_cache,
        ):
            return prepare_large_n_state(
                self,
                positions_arr=positions_arr,
                masses_arr=masses_arr,
                input_dtype=input_dtype,
                bounds=bounds,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta_val=theta_val,
                mac_type_val=mac_type_val,
                refine_local_val=refine_local_val,
                max_refine_levels_val=max_refine_levels_val,
                aspect_threshold_val=aspect_threshold_val,
                jit_tree_override=jit_tree,
                allow_stateful_cache=allow_stateful_cache,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                upward_center_mode=upward_center_mode,
                record_retry=record_retry,
                collected_retries=collected_retries,
                fused_device_mode=bool(fused_device_mode),
            )

        tree_artifacts = self._prepare_state_tree_and_upward(
            positions_arr=positions_arr,
            masses_arr=masses_arr,
            bounds=bounds,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            refine_local_val=refine_local_val,
            max_refine_levels_val=max_refine_levels_val,
            aspect_threshold_val=aspect_threshold_val,
            jit_tree_override=jit_tree,
            upward_center_mode=upward_center_mode,
            allow_stateful_cache=allow_stateful_cache,
        )
        force_scale_nodes = None
        use_paper_force_scale = self._uses_paper_style_force_scale()
        if use_paper_force_scale:
            node_count = int(tree_artifacts.tree.parent.shape[0])
            previous_force_scale = self._last_force_scale_nodes
            reduction_mode = self._force_scale_reduction_mode()
            need_prepass = False
            policy_orders = self._policy_orders_for_prepare_state(
                max_order=int(max_order)
            )
            if self.mac_force_scale_mode == "paper":
                need_prepass = True
            elif self.mac_force_scale_mode == "prev" or self._in_force_scale_prepass:
                if (
                    previous_force_scale is not None
                    and int(previous_force_scale.shape[0]) == node_count
                ):
                    force_scale_nodes = jnp.asarray(
                        previous_force_scale,
                        dtype=positions_arr.dtype,
                    )
                elif self._uses_paper_style_traversal_policy() and (
                    not self._in_force_scale_prepass
                ):
                    need_prepass = True
                else:
                    force_scale_nodes = jnp.ones(
                        (node_count,),
                        dtype=positions_arr.dtype,
                    )
            else:
                need_prepass = True
            if need_prepass:
                if len(policy_orders) == 0:
                    raise ValueError(
                        "mac_force_scale_mode='prepass' requires non-empty orders"
                    )
                low_order = int(min(policy_orders))
                if self.mac_force_scale_mode == "paper" and (
                    self._uses_paper_style_traversal_policy()
                ):
                    low_order = 1 if int(max_order) >= 1 else 0
                if self.mac_force_scale_mode == "paper" and (
                    self._uses_paper_style_traversal_policy()
                ):
                    prepass_sorted = (
                        self._compute_force_scale_paper_prepass_from_tree_artifacts(
                            tree_artifacts=tree_artifacts,
                            low_order=low_order,
                            theta_val=theta_val,
                            upward_center_mode=upward_center_mode,
                            runtime_traversal_config=runtime_traversal_config,
                            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                            grouped_interactions=grouped_interactions,
                            farfield_mode=farfield_mode,
                            record_retry=record_retry,
                            refine_local_val=refine_local_val,
                            max_refine_levels_val=max_refine_levels_val,
                            aspect_threshold_val=aspect_threshold_val,
                        )
                    )
                else:
                    self._in_force_scale_prepass = True
                    saved_p_gears = self.p_gears
                    saved_adaptive_order = self.adaptive_order
                    saved_adaptive_error_model = self.adaptive_error_model
                    saved_mac_type = self.mac_type
                    try:
                        self.p_gears = (low_order,)
                        self.adaptive_order = bool(saved_adaptive_order)
                        prepass_acc = self.compute_accelerations(
                            positions_arr,
                            masses_arr,
                            bounds=bounds,
                            leaf_size=int(leaf_size),
                            max_order=low_order,
                            return_potential=False,
                            theta=theta_val,
                            reuse_prepared_state=False,
                            jit_tree=jit_tree,
                            jit_traversal=False,
                        )
                    finally:
                        self.p_gears = saved_p_gears
                        self.adaptive_order = saved_adaptive_order
                        self.adaptive_error_model = saved_adaptive_error_model
                        self.mac_type = saved_mac_type
                        self._in_force_scale_prepass = False
                    prepass_sorted = jnp.asarray(prepass_acc)[
                        jnp.argsort(tree_artifacts.inverse_permutation)
                    ]
                force_scale_nodes = self._compute_node_force_scale_from_sorted_acc(
                    tree=tree_artifacts.tree,
                    accelerations_sorted=prepass_sorted,
                    reduction=reduction_mode,
                ).astype(positions_arr.dtype)
                self._last_force_scale_nodes = force_scale_nodes
        dual_downward_artifacts = self._prepare_state_dual_and_downward(
            tree_artifacts=tree_artifacts,
            force_scale_nodes=force_scale_nodes,
            upward_center_mode=upward_center_mode,
            theta_val=theta_val,
            mac_type_val=mac_type_val,
            dehnen_radius_scale=self.dehnen_radius_scale,
            runtime_traversal_config=runtime_traversal_config,
            grouped_interactions=grouped_interactions,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            farfield_mode=farfield_mode,
            record_retry=record_retry,
            refine_local_val=refine_local_val,
            max_refine_levels_val=max_refine_levels_val,
            aspect_threshold_val=aspect_threshold_val,
            allow_stateful_cache=allow_stateful_cache,
        )

        if allow_stateful_cache:
            self._update_locals_template_cache_after_prepare(
                locals_template=tree_artifacts.locals_template,
                upward=tree_artifacts.upward,
                max_order=int(max_order),
            )

        retry_events_tuple = tuple(collected_retries)
        if allow_stateful_cache:
            self._recent_retry_events = retry_events_tuple
            self._record_strict_cap_profile_from_retries(
                retry_events_tuple,
                context_key=self._strict_cap_profile_context_key(
                    tree_mode=str(tree_artifacts.tree_mode),
                    leaf_parameter=int(tree_artifacts.leaf_parameter),
                    particle_count=int(
                        jnp.asarray(tree_artifacts.positions_sorted).shape[0]
                    ),
                ),
            )

        execution_backend = self._resolve_execution_backend()
        tree_type_norm = (
            str(getattr(tree_artifacts.tree, "tree_type", "")).strip().lower()
        )
        build_octree_payload = (
            execution_backend == "octree" or tree_type_norm == "octree"
        )
        if build_octree_payload:
            octree, octree_native = build_octree_execution_data_with_status(
                tree_artifacts.tree
            )
        else:
            octree, octree_native = None, False
        # Only build the native-octree interaction lists when the octree view is
        # actually native (non-degenerate). On a degenerate octree (build_octree_
        # execution_data fell back to the binary tree), the native walk would produce
        # far pairs in a node space inconsistent with `octree`/the near list -> gaps +
        # double-counts; leaving native_far_pairs=None routes far through the compat
        # interaction list on the same (fallback) tree, matching the near field.
        octree_native_neighbors = None
        if execution_backend == "octree" and octree is not None and octree_native:
            octree_native_neighbors = build_octree_native_neighbor_lists(
                tree_artifacts.tree,
                tree_artifacts.upward.geometry,
                theta=theta_val,
                mac_type=mac_type_val,
                dehnen_radius_scale=self.dehnen_radius_scale,
                max_pair_queue=self.max_pair_queue,
                process_block=self.pair_process_block,
                traversal_config=runtime_traversal_config,
            )
        nearfield_interop = _build_nearfield_interop_data(
            tree_artifacts.tree,
            dual_downward_artifacts.neighbor_list,
            octree=None,
            native_neighbors=None,
        )
        if (
            execution_backend == "octree"
            and nearfield_interop.leaf_particle_indices is None
        ):
            leaf_nodes_nf = jnp.asarray(nearfield_interop.leaf_nodes, dtype=INDEX_DTYPE)
            node_ranges_nf = jnp.asarray(
                nearfield_interop.particle_order_node_ranges,
                dtype=INDEX_DTYPE,
            )
            leaf_ranges_nf = node_ranges_nf[leaf_nodes_nf]
            counts_nf = leaf_ranges_nf[:, 1] - leaf_ranges_nf[:, 0] + 1
            width_nf = int(jnp.max(counts_nf)) if int(leaf_nodes_nf.shape[0]) > 0 else 0
            if width_nf > 0:
                offsets_nf = jnp.arange(width_nf, dtype=INDEX_DTYPE)
                leaf_particle_indices_nf = (
                    leaf_ranges_nf[:, 0][:, None] + offsets_nf[None, :]
                )
                leaf_particle_mask_nf = offsets_nf[None, :] < counts_nf[:, None]
                particle_to_leaf_position_nf = jnp.zeros(
                    (int(positions_arr.shape[0]),),
                    dtype=INDEX_DTYPE,
                )
                particle_to_leaf_position_nf = particle_to_leaf_position_nf.at[
                    leaf_particle_indices_nf[leaf_particle_mask_nf]
                ].set(
                    jnp.repeat(
                        jnp.arange(int(leaf_nodes_nf.shape[0]), dtype=INDEX_DTYPE),
                        counts_nf.astype(INDEX_DTYPE),
                    )
                )
            else:
                leaf_particle_indices_nf = jnp.zeros(
                    (int(leaf_nodes_nf.shape[0]), 0), dtype=INDEX_DTYPE
                )
                leaf_particle_mask_nf = jnp.zeros(
                    (int(leaf_nodes_nf.shape[0]), 0), dtype=bool
                )
                particle_to_leaf_position_nf = jnp.zeros(
                    (int(positions_arr.shape[0]),),
                    dtype=INDEX_DTYPE,
                )
            nearfield_interop = nearfield_interop._replace(
                leaf_particle_indices=leaf_particle_indices_nf,
                leaf_particle_mask=leaf_particle_mask_nf,
                particle_to_leaf_position=particle_to_leaf_position_nf,
            )
        nearfield_artifacts = self._prepare_state_nearfield_artifacts(
            neighbor_list=dual_downward_artifacts.neighbor_list,
            nearfield_interop=nearfield_interop,
            leaf_cap=tree_artifacts.leaf_cap,
            num_particles=int(positions_arr.shape[0]),
            cache_entry=dual_downward_artifacts.cache_entry,
            allow_stateful_cache=allow_stateful_cache,
        )
        _prepare_diag(
            "nearfield bytes "
            f"target_leaf_ids={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.target_leaf_ids))} "
            f"source_leaf_ids={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.source_leaf_ids))} "
            f"valid_pairs={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.valid_pairs))} "
            f"chunk_sort_indices={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.chunk_sort_indices))} "
            f"chunk_group_ids={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.chunk_group_ids))} "
            f"chunk_unique_indices={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.chunk_unique_indices))}"
        )
        octree_upward = _build_octree_upward_artifacts(
            octree=octree,
            positions_sorted=tree_artifacts.positions_sorted,
            masses_sorted=tree_artifacts.masses_sorted,
            expansion_basis=self.expansion_basis,
            max_order=int(max_order),
        )
        octree_native_far_pairs = None
        if execution_backend == "octree" and octree is not None and octree_native:
            octree_native_far_pairs = build_octree_native_far_pairs(
                tree_artifacts.tree,
                tree_artifacts.upward.geometry,
                theta=theta_val,
                mac_type=mac_type_val,
                dehnen_radius_scale=self.dehnen_radius_scale,
                max_pair_queue=self.max_pair_queue,
                process_block=self.pair_process_block,
                traversal_config=runtime_traversal_config,
            )
        octree_downward = _build_octree_downward_artifacts(
            octree=octree,
            octree_upward=octree_upward,
            interactions=dual_downward_artifacts.interactions,
            native_far_pairs=octree_native_far_pairs,
            execution_backend=execution_backend,
        )

        return FMMPreparedState(
            tree=tree_artifacts.tree,
            upward=_prepared_state_upward_payload(
                upward=tree_artifacts.upward,
                memory_objective=self.memory_objective,
            ),
            downward=dual_downward_artifacts.downward,
            neighbor_list=dual_downward_artifacts.neighbor_list,
            max_leaf_size=tree_artifacts.leaf_cap,
            input_dtype=input_dtype,
            working_dtype=positions_arr.dtype,
            expansion_basis=self.expansion_basis,
            theta=theta_val,
            topology_key=tree_artifacts.topology_key,
            interactions=dual_downward_artifacts.interactions,
            dual_tree_result=dual_downward_artifacts.traversal_result,
            retry_events=retry_events_tuple,
            nearfield_interop=nearfield_interop,
            nearfield_target_leaf_ids=nearfield_artifacts.target_leaf_ids,
            nearfield_source_leaf_ids=nearfield_artifacts.source_leaf_ids,
            nearfield_valid_pairs=nearfield_artifacts.valid_pairs,
            nearfield_chunk_sort_indices=nearfield_artifacts.chunk_sort_indices,
            nearfield_chunk_group_ids=nearfield_artifacts.chunk_group_ids,
            nearfield_chunk_unique_indices=nearfield_artifacts.chunk_unique_indices,
            force_scale_nodes=force_scale_nodes,
            execution_backend=execution_backend,
            octree=octree,
            octree_upward=_prepared_state_octree_upward_payload(
                octree_upward=octree_upward,
                memory_objective=self.memory_objective,
            ),
            octree_downward=_finalize_octree_downward_artifacts(
                octree=octree,
                octree_upward=octree_upward,
                octree_downward=octree_downward,
                expansion_basis=self.expansion_basis,
                execution_backend=execution_backend,
                m2l_chunk_size=runtime_m2l_chunk_size,
            ),
        )

    @jaxtyped(typechecker=beartype)
    def evaluate_prepared_state(
        self: "FastMultipoleMethod",
        state: PreparedStateLike,
        *,
        target_indices: Optional[Array] = None,
        return_potential: bool = False,
        jit_traversal: bool = True,
        max_acc_derivative_order: int = 0,
    ) -> Union[
        Array,
        Tuple[Array, Array],
        Tuple[Array, PackedAccelerationDerivatives],
        Tuple[Array, Array, PackedAccelerationDerivatives],
    ]:
        """Evaluate accelerations/potentials for all particles or targets."""

        if isinstance(state, LargeNPreparedState):
            return evaluate_large_n_state(
                self,
                state,
                target_indices=target_indices,
                return_potential=return_potential,
                max_acc_derivative_order=max_acc_derivative_order,
            )

        resolved_target_indices = self._resolve_target_indices(
            target_indices=target_indices,
            num_particles=int(state.inverse_permutation.shape[0]),
        )
        tracing_targets = isinstance(
            state.positions_sorted, jax.core.Tracer
        ) or isinstance(resolved_target_indices, jax.core.Tracer)
        derivative_order = int(max_acc_derivative_order)
        if derivative_order < 0:
            raise ValueError("max_acc_derivative_order must be non-negative")
        if derivative_order > 0 and state.expansion_basis != "solidfmm":
            raise NotImplementedError(
                "max_acc_derivative_order > 0 currently requires expansion_basis='solidfmm'"
            )

        use_full_eval_for_targets = bool(return_potential) and (
            resolved_target_indices is not None
        )
        # Octree backend: evaluate the octree-native far-field locals (the near-field is
        # already octree-native). Only the full-particle path honours these overrides.
        (
            octree_farfield_local_data,
            octree_farfield_leaf_nodes,
            octree_farfield_node_ranges,
        ) = _octree_farfield_eval_inputs(state)
        if (
            resolved_target_indices is None
            or tracing_targets
            or use_full_eval_for_targets
        ):
            evaluation = _evaluate_prepared_tree(
                fmm=self,
                tree=state.tree,
                positions_sorted=state.positions_sorted,
                masses_sorted=state.masses_sorted,
                downward=state.downward,
                neighbor_list=state.neighbor_list,
                nearfield_interop=state.nearfield_interop,
                farfield_local_data=octree_farfield_local_data,
                farfield_leaf_nodes=octree_farfield_leaf_nodes,
                farfield_node_ranges=octree_farfield_node_ranges,
                nearfield_target_leaf_ids=state.nearfield_target_leaf_ids,
                nearfield_source_leaf_ids=state.nearfield_source_leaf_ids,
                nearfield_valid_pairs=state.nearfield_valid_pairs,
                nearfield_chunk_sort_indices=state.nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids=state.nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices=state.nearfield_chunk_unique_indices,
                max_leaf_size=state.max_leaf_size,
                return_potential=return_potential,
                jit_traversal=jit_traversal,
                max_acc_derivative_order=derivative_order,
            )
        else:
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[resolved_target_indices],
                dtype=INDEX_DTYPE,
            )
            evaluation = _evaluate_prepared_tree_targets(
                fmm=self,
                tree=state.tree,
                positions_sorted=state.positions_sorted,
                masses_sorted=state.masses_sorted,
                downward=state.downward,
                neighbor_list=state.neighbor_list,
                nearfield_interop=state.nearfield_interop,
                farfield_local_data=None,
                farfield_leaf_nodes=None,
                farfield_node_ranges=None,
                target_sorted_indices=target_sorted_indices,
                return_potential=return_potential,
                max_acc_derivative_order=derivative_order,
            )

        if jnp.issubdtype(state.input_dtype, jnp.floating):
            output_dtype = state.input_dtype
        else:
            output_dtype = state.working_dtype

        if derivative_order > 0:
            if return_potential:
                acc_sorted, pot_sorted, deriv_sorted = evaluation
            else:
                acc_sorted, deriv_sorted = evaluation
            if resolved_target_indices is None:
                accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation]
                derivatives = tuple(
                    jnp.asarray(level)[state.inverse_permutation]
                    for level in deriv_sorted
                )
                if return_potential:
                    potentials = jnp.asarray(pot_sorted)[state.inverse_permutation]
            elif tracing_targets or use_full_eval_for_targets:
                accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation][
                    resolved_target_indices
                ]
                derivatives = tuple(
                    jnp.asarray(level)[state.inverse_permutation][
                        resolved_target_indices
                    ]
                    for level in deriv_sorted
                )
                if return_potential:
                    potentials = jnp.asarray(pot_sorted)[state.inverse_permutation][
                        resolved_target_indices
                    ]
            else:
                accelerations = jnp.asarray(acc_sorted)
                derivatives = tuple(jnp.asarray(level) for level in deriv_sorted)
                if return_potential:
                    potentials = jnp.asarray(pot_sorted)
            accelerations = accelerations.astype(output_dtype)
            derivatives = tuple(level.astype(output_dtype) for level in derivatives)
            if return_potential:
                return accelerations, potentials.astype(output_dtype), derivatives
            return accelerations, derivatives

        if return_potential:
            acc_sorted, pot_sorted = evaluation
            if resolved_target_indices is None:
                accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation]
                potentials = jnp.asarray(pot_sorted)[state.inverse_permutation]
            elif tracing_targets or use_full_eval_for_targets:
                accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation][
                    resolved_target_indices
                ]
                potentials = jnp.asarray(pot_sorted)[state.inverse_permutation][
                    resolved_target_indices
                ]
            else:
                accelerations = jnp.asarray(acc_sorted)
                potentials = jnp.asarray(pot_sorted)
            accelerations = accelerations.astype(output_dtype)
            potentials = potentials.astype(output_dtype)
            return accelerations, potentials

        if resolved_target_indices is None:
            accelerations = jnp.asarray(evaluation)[state.inverse_permutation]
        elif tracing_targets:
            accelerations = jnp.asarray(evaluation)[state.inverse_permutation][
                resolved_target_indices
            ]
        else:
            accelerations = jnp.asarray(evaluation)
        accelerations = accelerations.astype(output_dtype)
        return accelerations

    @jaxtyped(typechecker=beartype)
    def evaluate_prepared_state_with_jerk(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
        jit_traversal: bool = True,
        jerk_mode: str = "fast_approx",
        jerk_fd_dt: float = 1e-3,
    ) -> tuple[Array, Array]:
        """Evaluate accelerations and jerk for all particles or targets."""
        vel_arr = jnp.asarray(velocities, dtype=state.working_dtype)
        if vel_arr.shape != state.positions_sorted.shape:
            raise ValueError(
                "velocities must have shape "
                f"{tuple(state.positions_sorted.shape)}, got {tuple(vel_arr.shape)}"
            )

        resolved_target_indices = self._resolve_target_indices(
            target_indices=target_indices,
            num_particles=int(state.inverse_permutation.shape[0]),
        )
        mode = str(jerk_mode).strip().lower()
        if mode not in ("fast_approx", "accurate"):
            raise ValueError("jerk_mode must be 'fast_approx' or 'accurate'")

        if mode == "accurate":
            if state.expansion_basis != "solidfmm":
                dt = float(jerk_fd_dt)
                if dt <= 0.0:
                    raise ValueError("jerk_fd_dt must be positive")

                accelerations = self.evaluate_prepared_state(
                    state,
                    target_indices=resolved_target_indices,
                    return_potential=False,
                    jit_traversal=jit_traversal,
                    max_acc_derivative_order=0,
                )
                particle_indices = jnp.asarray(
                    state.tree.particle_indices, dtype=INDEX_DTYPE
                )
                vel_sorted = vel_arr[particle_indices]
                positions_plus_sorted = (
                    jnp.asarray(state.positions_sorted) + dt * vel_sorted
                )
                positions_minus_sorted = (
                    jnp.asarray(state.positions_sorted) - dt * vel_sorted
                )

                acc_plus = self._evaluate_prepared_state_at_positions_sorted(
                    state=state,
                    positions_sorted=positions_plus_sorted,
                    target_indices=resolved_target_indices,
                    jit_traversal=jit_traversal,
                )
                acc_minus = self._evaluate_prepared_state_at_positions_sorted(
                    state=state,
                    positions_sorted=positions_minus_sorted,
                    target_indices=resolved_target_indices,
                    jit_traversal=jit_traversal,
                )
                jerk = (acc_plus - acc_minus) / (2.0 * dt)
                if jnp.issubdtype(state.input_dtype, jnp.floating):
                    output_dtype = state.input_dtype
                else:
                    output_dtype = state.working_dtype
                return accelerations.astype(output_dtype), jerk.astype(output_dtype)

            acc_eval = self.evaluate_prepared_state(
                state,
                target_indices=resolved_target_indices,
                return_potential=False,
                jit_traversal=jit_traversal,
                max_acc_derivative_order=1,
            )
            accelerations, acc_derivs = acc_eval
            acc_jac = acc_derivs[0]
            vel_targets = (
                vel_arr
                if resolved_target_indices is None
                else vel_arr[resolved_target_indices]
            )
            far_convective_jerk = jnp.einsum("nij,nj->ni", acc_jac, vel_targets)
            far_source_motion_jerk = (
                self._evaluate_farfield_time_derivative_orders(
                    state=state,
                    velocities=vel_arr,
                    target_indices=resolved_target_indices,
                    max_time_derivative_order=1,
                )[0]
                - far_convective_jerk
            )
            near_jerk = self._evaluate_target_nearfield_jerk(
                state=state,
                velocities=vel_arr,
                target_indices=resolved_target_indices,
            )
            jerk = near_jerk + far_convective_jerk + far_source_motion_jerk
            if jnp.issubdtype(state.input_dtype, jnp.floating):
                output_dtype = state.input_dtype
            else:
                output_dtype = state.working_dtype
            return accelerations.astype(output_dtype), jerk.astype(output_dtype)

        acc_eval = self.evaluate_prepared_state(
            state,
            target_indices=resolved_target_indices,
            return_potential=False,
            jit_traversal=jit_traversal,
            max_acc_derivative_order=1,
        )
        accelerations, acc_derivs = acc_eval
        acc_jac = acc_derivs[0]
        vel_targets = (
            vel_arr
            if resolved_target_indices is None
            else vel_arr[resolved_target_indices]
        )
        far_jerk = jnp.einsum("nij,nj->ni", acc_jac, vel_targets)

        near_jerk = self._evaluate_target_nearfield_jerk(
            state=state,
            velocities=vel_arr,
            target_indices=resolved_target_indices,
        )

        jerk = near_jerk + far_jerk
        if jnp.issubdtype(state.input_dtype, jnp.floating):
            output_dtype = state.input_dtype
        else:
            output_dtype = state.working_dtype
        return accelerations.astype(output_dtype), jerk.astype(output_dtype)

    @jaxtyped(typechecker=beartype)
    def evaluate_prepared_state_with_time_derivatives(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
        jit_traversal: bool = True,
        max_time_derivative_order: int = 1,
        mode: str = "accurate",
    ) -> tuple[Array, tuple[Array, ...]]:
        """Evaluate accelerations and total time derivatives up to order K.

        Returns ``(accelerations, derivatives)`` where ``derivatives[n-1]``
        corresponds to ``D_t^n a``.
        """
        k_max = int(max_time_derivative_order)
        if k_max < 1:
            raise ValueError("max_time_derivative_order must be >= 1")
        if k_max > 3:
            raise NotImplementedError(
                "max_time_derivative_order > 3 is not implemented yet"
            )
        resolved_target_indices = self._resolve_target_indices(
            target_indices=target_indices,
            num_particles=int(state.inverse_permutation.shape[0]),
        )
        mode_norm = str(mode).strip().lower()
        if mode_norm not in ("accurate",):
            raise ValueError("mode must be 'accurate'")

        vel_arr = jnp.asarray(velocities, dtype=state.working_dtype)
        if vel_arr.shape != state.positions_sorted.shape:
            raise ValueError(
                "velocities must have shape "
                f"{tuple(state.positions_sorted.shape)}, got {tuple(vel_arr.shape)}"
            )

        accelerations = self.evaluate_prepared_state(
            state,
            target_indices=resolved_target_indices,
            return_potential=False,
            jit_traversal=jit_traversal,
            max_acc_derivative_order=0,
        )
        far_terms = self._evaluate_farfield_time_derivative_orders(
            state=state,
            velocities=vel_arr,
            target_indices=resolved_target_indices,
            max_time_derivative_order=k_max,
        )
        near_terms = self._evaluate_target_nearfield_time_derivatives(
            state=state,
            velocities=vel_arr,
            target_indices=resolved_target_indices,
            max_time_derivative_order=k_max,
        )
        derivatives = tuple(n + f for n, f in zip(near_terms, far_terms))
        if jnp.issubdtype(state.input_dtype, jnp.floating):
            output_dtype = state.input_dtype
        else:
            output_dtype = state.working_dtype
        return accelerations.astype(output_dtype), tuple(
            d.astype(output_dtype) for d in derivatives
        )

    @jaxtyped(typechecker=beartype)
    def _evaluate_target_nearfield_jerk(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
    ) -> Array:
        particle_indices = jnp.asarray(state.tree.particle_indices, dtype=INDEX_DTYPE)
        vel_sorted = velocities[particle_indices]

        if target_indices is None:
            target_sorted_indices = jnp.arange(
                state.positions_sorted.shape[0], dtype=INDEX_DTYPE
            )
        else:
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[target_indices], dtype=INDEX_DTYPE
            )
        leaf_nodes = jnp.asarray(state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
        node_ranges = jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE)
        target_leaf_positions = _map_targets_to_leaf_positions(
            target_sorted_indices=target_sorted_indices,
            leaf_nodes=leaf_nodes,
            node_ranges=node_ranges,
        )
        near_source_idx, near_source_mask = _build_target_nearfield_source_index_matrix(
            target_sorted_indices=target_sorted_indices,
            target_leaf_positions=target_leaf_positions,
            nearfield_interop=(
                _build_nearfield_interop_data(state.tree, state.neighbor_list)
                if state.nearfield_interop is None
                else state.nearfield_interop
            ),
        )
        _, _, near_jerk_sorted, _, _ = _compute_targeted_nearfield(
            positions_sorted=state.positions_sorted,
            masses_sorted=state.masses_sorted,
            target_sorted_indices=target_sorted_indices,
            source_indices=near_source_idx,
            source_mask=near_source_mask,
            G=jnp.asarray(self.G, dtype=state.positions_sorted.dtype),
            softening=float(self.softening),
            return_potential=False,
            velocities_sorted=vel_sorted,
            return_jerk=True,
        )
        if near_jerk_sorted is None:
            raise RuntimeError("expected near-field jerk values")
        if target_indices is None:
            return near_jerk_sorted[state.inverse_permutation]
        return near_jerk_sorted

    @jaxtyped(typechecker=beartype)
    def _evaluate_target_nearfield_time_derivatives(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
        max_time_derivative_order: int,
    ) -> tuple[Array, ...]:
        """Evaluate near-field time derivatives up to order K (currently K<=2)."""
        k_max = int(max_time_derivative_order)
        if k_max < 1:
            return tuple()
        if k_max > 3:
            raise NotImplementedError(
                "near-field time derivatives above order 3 are not implemented"
            )
        particle_indices = jnp.asarray(state.tree.particle_indices, dtype=INDEX_DTYPE)
        vel_sorted = velocities[particle_indices]
        if target_indices is None:
            target_sorted_indices = jnp.arange(
                state.positions_sorted.shape[0], dtype=INDEX_DTYPE
            )
        else:
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[target_indices], dtype=INDEX_DTYPE
            )
        leaf_nodes = jnp.asarray(state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
        node_ranges = jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE)
        target_leaf_positions = _map_targets_to_leaf_positions(
            target_sorted_indices=target_sorted_indices,
            leaf_nodes=leaf_nodes,
            node_ranges=node_ranges,
        )
        near_source_idx, near_source_mask = _build_target_nearfield_source_index_matrix(
            target_sorted_indices=target_sorted_indices,
            target_leaf_positions=target_leaf_positions,
            nearfield_interop=(
                _build_nearfield_interop_data(state.tree, state.neighbor_list)
                if state.nearfield_interop is None
                else state.nearfield_interop
            ),
        )
        _, _, near_jerk_sorted, near_snap_sorted, near_crackle_sorted = (
            _compute_targeted_nearfield(
                positions_sorted=state.positions_sorted,
                masses_sorted=state.masses_sorted,
                target_sorted_indices=target_sorted_indices,
                source_indices=near_source_idx,
                source_mask=near_source_mask,
                G=jnp.asarray(self.G, dtype=state.positions_sorted.dtype),
                softening=float(self.softening),
                return_potential=False,
                velocities_sorted=vel_sorted,
                return_jerk=True,
                return_snap=(k_max >= 2),
                return_crackle=(k_max >= 3),
            )
        )
        if near_jerk_sorted is None:
            raise RuntimeError("expected near-field jerk values")
        if k_max >= 2 and near_snap_sorted is None:
            raise RuntimeError("expected near-field snap values")
        if k_max >= 3 and near_crackle_sorted is None:
            raise RuntimeError("expected near-field crackle values")
        if target_indices is None:
            jerk = near_jerk_sorted[state.inverse_permutation]
            if k_max >= 2:
                snap = near_snap_sorted[state.inverse_permutation]  # type: ignore[index]
            if k_max >= 3:
                crackle = near_crackle_sorted[state.inverse_permutation]  # type: ignore[index]
        else:
            jerk = near_jerk_sorted
            if k_max >= 2:
                snap = near_snap_sorted  # type: ignore[assignment]
            if k_max >= 3:
                crackle = near_crackle_sorted  # type: ignore[assignment]
        if k_max == 1:
            return (jerk,)
        if k_max == 2:
            return (jerk, snap)  # type: ignore[possibly-undefined]
        return (jerk, snap, crackle)  # type: ignore[possibly-undefined]

    @jaxtyped(typechecker=beartype)
    def _evaluate_source_motion_farfield_jerk(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
    ) -> Array:
        """Evaluate source-motion far-field jerk using analytic dM->dL contraction."""
        if state.expansion_basis != "solidfmm":
            raise NotImplementedError(
                "source-motion far-field jerk currently requires expansion_basis='solidfmm'"
            )
        particle_indices = jnp.asarray(state.tree.particle_indices, dtype=INDEX_DTYPE)
        vel_sorted = velocities[particle_indices]
        centers = jnp.asarray(state.downward.locals.centers, dtype=state.working_dtype)
        source_motion_packed = prepare_solidfmm_complex_source_motion_multipoles(
            state.tree,
            state.positions_sorted,
            state.masses_sorted,
            vel_sorted,
            max_order=int(state.downward.locals.order),
            centers=centers,
            time_derivative_order=1,
            max_leaf_size=int(state.max_leaf_size),
            rotation=self.complex_rotation,
        )
        source_motion_multipoles = NodeMultipoleData(
            order=int(state.downward.locals.order),
            centers=centers,
            moments=None,  # type: ignore[arg-type]
            packed=jnp.asarray(source_motion_packed),
            component_matrix=None,
            source_motion_packed=None,
        )
        source_motion_upward = TreeUpwardData(
            geometry=compute_tree_geometry(state.tree, state.positions_sorted),
            mass_moments=compute_tree_mass_moments(
                state.tree,
                state.positions_sorted,
                state.masses_sorted,
            ),
            multipoles=source_motion_multipoles,
        )
        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(state.positions_sorted.shape[0]),
        )
        source_motion_downward = self.prepare_downward_sweep(
            state.tree,
            source_motion_upward,
            theta=float(state.theta),
            mac_type=self.mac_type,
            initial_locals=None,
            interactions=state.interactions,
            m2l_chunk_size=runtime_overrides.m2l_chunk_size,
            l2l_chunk_size=runtime_overrides.l2l_chunk_size,
            grouped_interactions=runtime_overrides.grouped_interactions,
            farfield_mode=runtime_overrides.farfield_mode,
            dehnen_radius_scale=self.dehnen_radius_scale,
        )
        tracing_targets = isinstance(
            state.positions_sorted, jax.core.Tracer
        ) or isinstance(target_indices, jax.core.Tracer)
        if target_indices is None or tracing_targets:
            far_grad_sorted, _, _ = _evaluate_local_expansions_for_particles(
                source_motion_downward.locals,
                state.positions_sorted,
                leaf_nodes=jnp.asarray(
                    state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE
                ),
                node_ranges=jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE),
                max_leaf_size=state.max_leaf_size,
                order=int(source_motion_downward.locals.order),
                expansion_basis=state.expansion_basis,
                return_potential=False,
                max_acc_derivative_order=0,
            )
            if target_indices is None:
                far_grad = jnp.asarray(far_grad_sorted)[state.inverse_permutation]
            else:
                far_grad = jnp.asarray(far_grad_sorted)[state.inverse_permutation][
                    target_indices
                ]
        else:
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[target_indices], dtype=INDEX_DTYPE
            )
            leaf_nodes = jnp.asarray(
                state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE
            )
            node_ranges = jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE)
            target_leaf_positions = _map_targets_to_leaf_positions(
                target_sorted_indices=target_sorted_indices,
                leaf_nodes=leaf_nodes,
                node_ranges=node_ranges,
            )
            far_grad, _, _ = _evaluate_local_expansions_for_target_particles(
                local_data=source_motion_downward.locals,
                positions_sorted=state.positions_sorted,
                target_sorted_indices=target_sorted_indices,
                target_leaf_positions=target_leaf_positions,
                leaf_nodes=leaf_nodes,
                order=int(source_motion_downward.locals.order),
                expansion_basis=state.expansion_basis,
                return_potential=False,
                max_acc_derivative_order=0,
            )
        return -jnp.asarray(self.G, dtype=state.positions_sorted.dtype) * far_grad

    @jaxtyped(typechecker=beartype)
    def _evaluate_farfield_time_derivative_orders(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
        max_time_derivative_order: int,
    ) -> tuple[Array, ...]:
        """Evaluate far-field total time derivatives up to order ``max_time_derivative_order``.

        Uses binomial expansion of ``(∂t + v·∇)^n a`` with analytic source-motion
        locals ``L_k = ∂t^k L`` and acceleration spatial derivatives.
        """
        if state.expansion_basis != "solidfmm":
            raise NotImplementedError(
                "far-field higher time derivatives currently require expansion_basis='solidfmm'"
            )
        k_max = int(max_time_derivative_order)
        if k_max <= 0:
            return tuple()

        def _contract_acc_tensor_with_velocity_power(
            tensor: Array,
            velocity: Array,
            *,
            order: int,
        ) -> Array:
            """Contract symmetric acceleration-derivative tensor ``order`` times."""
            if order <= 0:
                raise ValueError("order must be positive")

            def contract_row(row: Array, vrow: Array) -> Array:
                # row shape: (3, components(order))
                contracted = row
                for ord_i in range(order, 0, -1):
                    contracted = jax.vmap(
                        lambda comp: contract_symmetric_one_axis_3d(
                            comp,
                            vrow,
                            order=ord_i,
                        ),
                        in_axes=0,
                        out_axes=0,
                    )(contracted)
                return contracted[:, 0]

            return jax.vmap(contract_row, in_axes=(0, 0), out_axes=0)(tensor, velocity)

        particle_indices = jnp.asarray(state.tree.particle_indices, dtype=INDEX_DTYPE)
        vel_sorted = velocities[particle_indices]
        centers = jnp.asarray(state.downward.locals.centers, dtype=state.working_dtype)
        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(state.positions_sorted.shape[0]),
        )
        resolved_target_indices = self._resolve_target_indices(
            target_indices=target_indices,
            num_particles=int(state.inverse_permutation.shape[0]),
        )
        tracing_targets = isinstance(
            state.positions_sorted, jax.core.Tracer
        ) or isinstance(resolved_target_indices, jax.core.Tracer)
        vel_targets = (
            velocities
            if resolved_target_indices is None
            else velocities[resolved_target_indices]
        )

        # Build local coefficient streams L_k = ∂t^k L, including k=0.
        locals_by_k: list[LocalExpansionData] = [state.downward.locals]
        geometry = compute_tree_geometry(state.tree, state.positions_sorted)
        mass_moments = compute_tree_mass_moments(
            state.tree, state.positions_sorted, state.masses_sorted
        )
        for k in range(1, k_max + 1):
            source_motion_packed = prepare_solidfmm_complex_source_motion_multipoles(
                state.tree,
                state.positions_sorted,
                state.masses_sorted,
                vel_sorted,
                max_order=int(state.downward.locals.order),
                centers=centers,
                time_derivative_order=k,
                max_leaf_size=int(state.max_leaf_size),
                rotation=self.complex_rotation,
            )
            source_motion_upward = TreeUpwardData(
                geometry=geometry,
                mass_moments=mass_moments,
                multipoles=NodeMultipoleData(
                    order=int(state.downward.locals.order),
                    centers=centers,
                    moments=None,  # type: ignore[arg-type]
                    packed=jnp.asarray(source_motion_packed),
                    component_matrix=jnp.asarray(source_motion_packed),
                    source_motion_packed=None,
                ),
            )
            down_k = self.prepare_downward_sweep(
                state.tree,
                source_motion_upward,
                theta=float(state.theta),
                mac_type=self.mac_type,
                initial_locals=None,
                interactions=state.interactions,
                m2l_chunk_size=runtime_overrides.m2l_chunk_size,
                l2l_chunk_size=runtime_overrides.l2l_chunk_size,
                grouped_interactions=runtime_overrides.grouped_interactions,
                farfield_mode=runtime_overrides.farfield_mode,
                dehnen_radius_scale=self.dehnen_radius_scale,
            )
            locals_by_k.append(down_k.locals)

        def _evaluate_local_stream(
            local_data: LocalExpansionData,
            *,
            max_acc_deriv_order: int,
        ) -> tuple[Array, Optional[PackedAccelerationDerivatives]]:
            if resolved_target_indices is None or tracing_targets:
                far_grad_sorted, _, far_derivs = (
                    _evaluate_local_expansions_for_particles(
                        local_data,
                        state.positions_sorted,
                        leaf_nodes=jnp.asarray(
                            state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE
                        ),
                        node_ranges=jnp.asarray(
                            state.tree.node_ranges, dtype=INDEX_DTYPE
                        ),
                        max_leaf_size=state.max_leaf_size,
                        order=int(local_data.order),
                        expansion_basis=state.expansion_basis,
                        return_potential=False,
                        max_acc_derivative_order=max_acc_deriv_order,
                    )
                )
                if resolved_target_indices is None:
                    far_grad = jnp.asarray(far_grad_sorted)[state.inverse_permutation]
                    if far_derivs is None:
                        derivs = None
                    else:
                        derivs = tuple(
                            jnp.asarray(level)[state.inverse_permutation]
                            for level in far_derivs
                        )
                else:
                    far_grad = jnp.asarray(far_grad_sorted)[state.inverse_permutation][
                        resolved_target_indices
                    ]
                    if far_derivs is None:
                        derivs = None
                    else:
                        derivs = tuple(
                            jnp.asarray(level)[state.inverse_permutation][
                                resolved_target_indices
                            ]
                            for level in far_derivs
                        )
                return far_grad, derivs
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[resolved_target_indices], dtype=INDEX_DTYPE
            )
            leaf_nodes = jnp.asarray(
                state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE
            )
            node_ranges = jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE)
            target_leaf_positions = _map_targets_to_leaf_positions(
                target_sorted_indices=target_sorted_indices,
                leaf_nodes=leaf_nodes,
                node_ranges=node_ranges,
            )
            far_grad, _, far_derivs = _evaluate_local_expansions_for_target_particles(
                local_data=local_data,
                positions_sorted=state.positions_sorted,
                target_sorted_indices=target_sorted_indices,
                target_leaf_positions=target_leaf_positions,
                leaf_nodes=leaf_nodes,
                order=int(local_data.order),
                expansion_basis=state.expansion_basis,
                return_potential=False,
                max_acc_derivative_order=max_acc_deriv_order,
            )
            return far_grad, far_derivs

        g_const = jnp.asarray(self.G, dtype=state.positions_sorted.dtype)
        outputs: list[Array] = []
        for n in range(1, k_max + 1):
            accum = jnp.zeros_like(vel_targets)
            for k in range(0, n + 1):
                m = n - k
                far_grad_k, far_derivs_k = _evaluate_local_stream(
                    locals_by_k[k],
                    max_acc_deriv_order=m,
                )
                if m == 0:
                    term_vec = -g_const * far_grad_k
                else:
                    if far_derivs_k is None:
                        raise RuntimeError(
                            "expected far-field acceleration derivatives"
                        )
                    acc_deriv_tensor = g_const * far_derivs_k[m - 1]
                    term_vec = _contract_acc_tensor_with_velocity_power(
                        acc_deriv_tensor,
                        vel_targets,
                        order=m,
                    )
                accum = accum + float(comb(n, k)) * term_vec
            outputs.append(accum)
        return tuple(outputs)

    @jaxtyped(typechecker=beartype)
    def _evaluate_prepared_state_at_positions_sorted(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        positions_sorted: Array,
        *,
        target_indices: Optional[Array] = None,
        jit_traversal: bool = True,
    ) -> Array:
        """Evaluate accelerations for updated sorted positions on a fixed topology."""
        positions_sorted_arr = jnp.asarray(positions_sorted, dtype=state.working_dtype)
        if positions_sorted_arr.shape != state.positions_sorted.shape:
            raise ValueError(
                "positions_sorted must have shape "
                f"{tuple(state.positions_sorted.shape)}, got {tuple(positions_sorted_arr.shape)}"
            )

        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(positions_sorted_arr.shape[0]),
        )
        upward = self.prepare_upward_sweep(
            state.tree,
            positions_sorted_arr,
            state.masses_sorted,
            max_order=int(state.downward.locals.order),
            center_mode=runtime_overrides.center_mode,
            max_leaf_size=int(state.max_leaf_size),
        )
        downward = self.prepare_downward_sweep(
            state.tree,
            upward,
            theta=float(state.theta),
            mac_type=self.mac_type,
            initial_locals=None,
            interactions=state.interactions,
            m2l_chunk_size=runtime_overrides.m2l_chunk_size,
            l2l_chunk_size=runtime_overrides.l2l_chunk_size,
            grouped_interactions=runtime_overrides.grouped_interactions,
            farfield_mode=runtime_overrides.farfield_mode,
            dehnen_radius_scale=self.dehnen_radius_scale,
        )
        resolved_target_indices = self._resolve_target_indices(
            target_indices=target_indices,
            num_particles=int(state.inverse_permutation.shape[0]),
        )
        tracing_targets = isinstance(
            positions_sorted_arr, jax.core.Tracer
        ) or isinstance(resolved_target_indices, jax.core.Tracer)
        # Octree backend: evaluate octree-native far-field locals (full path only).
        (
            octree_farfield_local_data,
            octree_farfield_leaf_nodes,
            octree_farfield_node_ranges,
        ) = _octree_farfield_eval_inputs(state)
        if resolved_target_indices is None or tracing_targets:
            evaluation = _evaluate_prepared_tree(
                fmm=self,
                tree=state.tree,
                positions_sorted=positions_sorted_arr,
                masses_sorted=state.masses_sorted,
                downward=downward,
                neighbor_list=state.neighbor_list,
                nearfield_interop=state.nearfield_interop,
                farfield_local_data=octree_farfield_local_data,
                farfield_leaf_nodes=octree_farfield_leaf_nodes,
                farfield_node_ranges=octree_farfield_node_ranges,
                nearfield_target_leaf_ids=state.nearfield_target_leaf_ids,
                nearfield_source_leaf_ids=state.nearfield_source_leaf_ids,
                nearfield_valid_pairs=state.nearfield_valid_pairs,
                nearfield_chunk_sort_indices=state.nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids=state.nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices=state.nearfield_chunk_unique_indices,
                max_leaf_size=state.max_leaf_size,
                return_potential=False,
                jit_traversal=jit_traversal,
                max_acc_derivative_order=0,
            )
        else:
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[resolved_target_indices],
                dtype=INDEX_DTYPE,
            )
            evaluation = _evaluate_prepared_tree_targets(
                fmm=self,
                tree=state.tree,
                positions_sorted=positions_sorted_arr,
                masses_sorted=state.masses_sorted,
                downward=downward,
                neighbor_list=state.neighbor_list,
                nearfield_interop=state.nearfield_interop,
                farfield_local_data=None,
                farfield_leaf_nodes=None,
                farfield_node_ranges=None,
                target_sorted_indices=target_sorted_indices,
                return_potential=False,
                max_acc_derivative_order=0,
            )

        if jnp.issubdtype(state.input_dtype, jnp.floating):
            output_dtype = state.input_dtype
        else:
            output_dtype = state.working_dtype
        if resolved_target_indices is None:
            accelerations = jnp.asarray(evaluation)[state.inverse_permutation]
        elif tracing_targets:
            accelerations = jnp.asarray(evaluation)[state.inverse_permutation][
                resolved_target_indices
            ]
        else:
            accelerations = jnp.asarray(evaluation)
        return accelerations.astype(output_dtype)

    @jaxtyped(typechecker=beartype)
    def evaluate_tree(
        self: "FastMultipoleMethod",
        tree: Tree,
        positions_sorted: Array,
        masses_sorted: Array,
        locals_or_downward: Union[LocalExpansionData, TreeDownwardData],
        neighbor_list: NodeNeighborList,
        *,
        nearfield_interop: Optional[NearfieldInteropData] = None,
        farfield_local_data: Optional[LocalExpansionData] = None,
        farfield_leaf_nodes: Optional[Array] = None,
        farfield_node_ranges: Optional[Array] = None,
        precomputed_target_leaf_ids: Optional[Array] = None,
        precomputed_source_leaf_ids: Optional[Array] = None,
        precomputed_valid_pairs: Optional[Array] = None,
        precomputed_chunk_sort_indices: Optional[Array] = None,
        precomputed_chunk_group_ids: Optional[Array] = None,
        precomputed_chunk_unique_indices: Optional[Array] = None,
        max_leaf_size: Optional[int] = None,
        return_potential: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Combine far- and near-field effects for leaf particles."""

        setup = _prepare_tree_evaluation_inputs(
            tree,
            positions_sorted,
            masses_sorted,
            locals_or_downward,
            neighbor_list,
            farfield_local_data=farfield_local_data,
            farfield_leaf_nodes=farfield_leaf_nodes,
            farfield_node_ranges=farfield_node_ranges,
            max_leaf_size=max_leaf_size,
            return_potential=return_potential,
        )

        if setup.empty_output is not None:
            return setup.empty_output

        locals_data = setup.locals_data
        positions = setup.positions
        masses = setup.masses
        leaf_nodes = setup.leaf_nodes
        node_ranges = setup.node_ranges
        resolved_max_leaf = setup.max_leaf_size

        order = int(locals_data.order)
        nearfield_mode = self._resolve_nearfield_mode(
            num_particles=int(positions.shape[0])
        )
        nearfield_edge_chunk_size = self._resolve_nearfield_edge_chunk_size(
            num_particles=int(positions.shape[0]),
            nearfield_mode=nearfield_mode,
        )
        nearfield_view = (
            _build_nearfield_interop_data(tree, neighbor_list)
            if nearfield_interop is None
            else nearfield_interop
        )

        near = compute_leaf_p2p_accelerations(
            tree,
            neighbor_list,
            positions,
            masses,
            G=self.G,
            softening=self.softening,
            max_leaf_size=resolved_max_leaf,
            return_potential=return_potential,
            nearfield_mode=nearfield_mode,
            edge_chunk_size=nearfield_edge_chunk_size,
            precomputed_target_leaf_ids=precomputed_target_leaf_ids,
            precomputed_source_leaf_ids=precomputed_source_leaf_ids,
            precomputed_valid_pairs=precomputed_valid_pairs,
            precomputed_chunk_sort_indices=precomputed_chunk_sort_indices,
            precomputed_chunk_group_ids=precomputed_chunk_group_ids,
            precomputed_chunk_unique_indices=precomputed_chunk_unique_indices,
            node_ranges_override=nearfield_view.node_ranges,
            leaf_nodes_override=nearfield_view.leaf_nodes,
            neighbor_offsets_override=nearfield_view.offsets,
            neighbor_indices_override=nearfield_view.neighbors,
            neighbor_counts_override=nearfield_view.counts,
            leaf_particle_indices_override=nearfield_view.leaf_particle_indices,
            leaf_particle_mask_override=nearfield_view.leaf_particle_mask,
        )

        far_grad, far_potential_pre, _ = _evaluate_local_expansions_for_particles(
            locals_data,
            positions,
            leaf_nodes=leaf_nodes,
            node_ranges=node_ranges,
            max_leaf_size=resolved_max_leaf,
            order=order,
            expansion_basis=self.expansion_basis,
            return_potential=return_potential,
            max_acc_derivative_order=0,
        )

        # far_grad is d/d(delta) of +1/r with delta = center - eval_point.
        # Physical acceleration is d/d(eval_point)(+1/r) * G = -d/d(delta)(+1/r) * G.
        far_acc = -self.G * far_grad

        if return_potential:
            near_acc, near_pot = near
            far_pot = (
                -self.G * far_potential_pre
                if far_potential_pre is not None
                else jnp.zeros((positions.shape[0],), dtype=positions.dtype)
            )
            accelerations = near_acc + far_acc
            potentials = near_pot + far_pot
            return accelerations, potentials

        accelerations = near + far_acc
        return accelerations

    @jaxtyped(typechecker=beartype)
    def evaluate_tree_compiled(
        self: "FastMultipoleMethod",
        tree: Tree,
        positions_sorted: Array,
        masses_sorted: Array,
        locals_or_downward: Union[LocalExpansionData, TreeDownwardData],
        neighbor_list: NodeNeighborList,
        *,
        nearfield_interop: Optional[NearfieldInteropData] = None,
        farfield_local_data: Optional[LocalExpansionData] = None,
        farfield_leaf_nodes: Optional[Array] = None,
        farfield_node_ranges: Optional[Array] = None,
        precomputed_target_leaf_ids: Optional[Array] = None,
        precomputed_source_leaf_ids: Optional[Array] = None,
        precomputed_valid_pairs: Optional[Array] = None,
        precomputed_chunk_sort_indices: Optional[Array] = None,
        precomputed_chunk_group_ids: Optional[Array] = None,
        precomputed_chunk_unique_indices: Optional[Array] = None,
        max_leaf_size: Optional[int] = None,
        return_potential: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        """JIT-compiled variant of :meth:`evaluate_tree`."""

        resolved_max_leaf = (
            self.fixed_max_leaf_size
            if self.fixed_max_leaf_size is not None
            else max_leaf_size
        )

        setup = _prepare_tree_evaluation_inputs(
            tree,
            positions_sorted,
            masses_sorted,
            locals_or_downward,
            neighbor_list,
            farfield_local_data=farfield_local_data,
            farfield_leaf_nodes=farfield_leaf_nodes,
            farfield_node_ranges=farfield_node_ranges,
            max_leaf_size=resolved_max_leaf,
            return_potential=return_potential,
        )

        if setup.empty_output is not None:
            return setup.empty_output

        if self.fixed_order is not None:
            order = int(self.fixed_order)
        else:
            coeff_count = int(setup.locals_data.coefficients.shape[-1])
            order = _infer_order_from_coeff_count(
                coeff_count=coeff_count,
                expansion_basis=self.expansion_basis,
            )

        if self.fixed_max_leaf_size is not None and setup.max_leaf_size > int(
            self.fixed_max_leaf_size
        ):
            raise ValueError("fixed_max_leaf_size too small for prepared tree")
        nearfield_mode = self._resolve_nearfield_mode(
            num_particles=int(setup.positions.shape[0])
        )
        nearfield_edge_chunk_size = self._resolve_nearfield_edge_chunk_size(
            num_particles=int(setup.positions.shape[0]),
            nearfield_mode=nearfield_mode,
        )
        nearfield_view = (
            _build_nearfield_interop_data(tree, neighbor_list)
            if nearfield_interop is None
            else nearfield_interop
        )

        return _evaluate_tree_compiled_impl(
            tree,
            setup.positions,
            setup.masses,
            setup.locals_data,
            neighbor_list,
            jnp.asarray(nearfield_view.leaf_nodes, dtype=INDEX_DTYPE),
            jnp.asarray(nearfield_view.node_ranges, dtype=INDEX_DTYPE),
            jnp.asarray(nearfield_view.offsets, dtype=INDEX_DTYPE),
            jnp.asarray(nearfield_view.neighbors, dtype=INDEX_DTYPE),
            jnp.asarray(nearfield_view.counts, dtype=INDEX_DTYPE),
            (
                jnp.asarray(nearfield_view.leaf_particle_indices, dtype=INDEX_DTYPE)
                if nearfield_view.leaf_particle_indices is not None
                else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(nearfield_view.leaf_particle_mask, dtype=bool)
                if nearfield_view.leaf_particle_mask is not None
                else jnp.zeros((0, 0), dtype=bool)
            ),
            setup.leaf_nodes,
            setup.node_ranges,
            (
                jnp.asarray(precomputed_target_leaf_ids, dtype=INDEX_DTYPE)
                if precomputed_target_leaf_ids is not None
                else jnp.zeros((0,), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(precomputed_source_leaf_ids, dtype=INDEX_DTYPE)
                if precomputed_source_leaf_ids is not None
                else jnp.zeros((0,), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(precomputed_valid_pairs, dtype=bool)
                if precomputed_valid_pairs is not None
                else jnp.zeros((0,), dtype=bool)
            ),
            (
                jnp.asarray(precomputed_chunk_sort_indices, dtype=INDEX_DTYPE)
                if precomputed_chunk_sort_indices is not None
                else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(precomputed_chunk_group_ids, dtype=INDEX_DTYPE)
                if precomputed_chunk_group_ids is not None
                else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(precomputed_chunk_unique_indices, dtype=INDEX_DTYPE)
                if precomputed_chunk_unique_indices is not None
                else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
            ),
            jnp.zeros((setup.leaf_nodes.shape[0] + 1,), dtype=INDEX_DTYPE),
            jnp.zeros((0,), dtype=INDEX_DTYPE),
            jnp.zeros((0, 0), dtype=INDEX_DTYPE),
            jnp.zeros((0, 0), dtype=bool),
            jnp.zeros((setup.leaf_nodes.shape[0], 0, 0), dtype=INDEX_DTYPE),
            jnp.zeros((setup.leaf_nodes.shape[0], 0, 0), dtype=bool),
            G=self.G,
            softening=self.softening,
            order=order,
            expansion_basis=self.expansion_basis,
            max_leaf_size=setup.max_leaf_size,
            return_potential=return_potential,
            nearfield_mode=nearfield_mode,
            nearfield_edge_chunk_size=nearfield_edge_chunk_size,
        )


def _infer_bounds(positions: Array) -> tuple[Array, Array]:
    """Infer generous bounds for tree construction from particle positions."""

    minimum = jnp.min(positions, axis=0)
    maximum = jnp.max(positions, axis=0)
    span = maximum - minimum
    padding = jnp.maximum(span * 0.05, jnp.full_like(span, 1e-6))
    return minimum - padding, maximum + padding


def _max_leaf_size_from_tree(tree: Tree) -> int:
    """Compute maximum number of particles per leaf node."""
    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    leaf_ranges = tree.node_ranges[num_internal:]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + as_index(1)
    return int(jnp.max(counts))


class _TreeEvaluationSetup(NamedTuple):
    """Prevalidated inputs required by tree-evaluation entry points."""

    locals_data: LocalExpansionData
    positions: Array
    masses: Array
    leaf_nodes: Array
    node_ranges: Array
    max_leaf_size: int
    empty_output: Optional[Union[Array, Tuple[Array, Array]]]


class _EvaluationNodeViews(NamedTuple):
    """Resolved leaf/node metadata for shared nearfield and backend-specific farfield."""

    nearfield: NearfieldInteropData
    farfield_leaf_nodes: Array
    farfield_node_ranges: Array


def _infer_order_from_coeff_count(
    *,
    coeff_count: int,
    expansion_basis: ExpansionBasis,
) -> int:
    """Infer expansion order from static coefficient-array width."""
    if expansion_basis == "solidfmm":
        root = int(round(float(np.sqrt(coeff_count))))
        order = root - 1
        if (order + 1) ** 2 != int(coeff_count):
            raise ValueError(
                "Could not infer solidfmm order from coefficient shape; "
                f"got coeff_count={coeff_count}."
            )
        return order

    for order in range(MAX_MULTIPOLE_ORDER + 1):
        if int(total_coefficients(order)) == int(coeff_count):
            return int(order)
    raise ValueError(
        "Could not infer cartesian order from coefficient shape; "
        f"got coeff_count={coeff_count}."
    )


def _resolve_evaluation_node_views(
    tree: Tree,
    neighbor_list: NodeNeighborList,
    *,
    farfield_leaf_nodes: Optional[Array],
    farfield_node_ranges: Optional[Array],
) -> _EvaluationNodeViews:
    """Resolve shared nearfield views and optional backend-specific farfield views.

    Nearfield continues to use the shared radix-oriented neighbor/leaf layout.
    Farfield may override that view, which is how the octree backend evaluates
    octree-native locals without rewriting nearfield plumbing yet.
    """

    nearfield = _build_nearfield_interop_data(tree, neighbor_list)
    radix_leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    radix_node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    resolved_farfield_leaf_nodes = (
        radix_leaf_nodes
        if farfield_leaf_nodes is None
        else jnp.asarray(farfield_leaf_nodes, dtype=INDEX_DTYPE)
    )
    resolved_farfield_node_ranges = (
        radix_node_ranges
        if farfield_node_ranges is None
        else jnp.asarray(farfield_node_ranges, dtype=INDEX_DTYPE)
    )
    return _EvaluationNodeViews(
        nearfield=nearfield,
        farfield_leaf_nodes=resolved_farfield_leaf_nodes,
        farfield_node_ranges=resolved_farfield_node_ranges,
    )


def _build_nearfield_interop_data(
    tree: Tree,
    neighbor_list: NodeNeighborList,
    *,
    octree: Optional[OctreeExecutionData] = None,
    native_neighbors: Optional[OctreeNativeNeighborList] = None,
) -> NearfieldInteropData:
    """Build the explicit leaf/node view shared by current nearfield helpers.

    The source-of-truth leaf ordering comes from ``neighbor_list``. For octree
    trees, yggdrax now emits that neighbor list in octree-native order while
    still exposing the particle-order leaf mapping needed for target lookup.
    """
    if native_neighbors is not None:
        if octree is None:
            raise ValueError("native octree nearfield data requires octree metadata")
        leaf_nodes = jnp.asarray(native_neighbors.leaf_indices, dtype=INDEX_DTYPE)
        native_offsets = jnp.asarray(native_neighbors.offsets, dtype=INDEX_DTYPE)
        native_neighbors_flat = jnp.asarray(
            native_neighbors.neighbors, dtype=INDEX_DTYPE
        )
        native_counts = jnp.asarray(native_neighbors.counts, dtype=INDEX_DTYPE)
        leaf_count = int(leaf_nodes.shape[0])
        radix_leaf_nodes = jnp.asarray(
            getattr(
                neighbor_list,
                "particle_order_leaf_indices",
                neighbor_list.leaf_indices,
            ),
            dtype=INDEX_DTYPE,
        )
        radix_leaf_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)[
            radix_leaf_nodes
        ]
        radix_leaf_counts = radix_leaf_ranges[:, 1] - radix_leaf_ranges[:, 0] + 1
        carrier_lookup = jnp.full(
            (octree.parent.shape[0],),
            -1,
            dtype=INDEX_DTYPE,
        )
        carrier_lookup = carrier_lookup.at[leaf_nodes].set(
            jnp.arange(leaf_count, dtype=INDEX_DTYPE)
        )
        radix_carrier_pos = carrier_lookup[
            jnp.asarray(octree.radix_leaf_to_oct, dtype=INDEX_DTYPE)
        ]
        carrier_particle_counts = jax.ops.segment_sum(
            radix_leaf_counts.astype(INDEX_DTYPE),
            radix_carrier_pos,
            leaf_count,
        )
        max_particles = int(jnp.max(carrier_particle_counts)) if leaf_count > 0 else 0

        if max_particles > 0:
            max_radix_leaf_particles = int(jnp.max(radix_leaf_counts))
            local_offsets = jnp.arange(max_radix_leaf_particles, dtype=INDEX_DTYPE)
            radix_particle_idx = (
                radix_leaf_ranges[:, 0][:, None] + local_offsets[None, :]
            )
            radix_particle_valid = local_offsets[None, :] < radix_leaf_counts[:, None]
            flat_particle_idx = radix_particle_idx.reshape(-1)
            flat_valid = radix_particle_valid.reshape(-1)
            flat_carrier_pos = jnp.repeat(radix_carrier_pos, max_radix_leaf_particles)
            safe_carrier_pos = jnp.where(flat_valid, flat_carrier_pos, leaf_count)
            order = jnp.argsort(safe_carrier_pos, stable=True)
            sorted_valid = flat_valid[order]
            sorted_carrier = safe_carrier_pos[order]
            sorted_particle_idx = flat_particle_idx[order]
            valid_int = sorted_valid.astype(INDEX_DTYPE)
            running = jnp.cumsum(valid_int, dtype=INDEX_DTYPE) - valid_int
            changed = jnp.concatenate(
                [
                    jnp.ones((1,), dtype=bool),
                    sorted_carrier[1:] != sorted_carrier[:-1],
                ]
            )
            group_starts = jnp.where(
                sorted_valid & changed,
                running,
                jnp.zeros_like(running),
            )
            group_starts = jnp.maximum.accumulate(group_starts)
            sorted_slots = running - group_starts
            row = jnp.where(sorted_valid, sorted_carrier, leaf_count)
            col = jnp.where(sorted_valid, sorted_slots, 0)
            leaf_particle_indices = jnp.zeros(
                (leaf_count + 1, max_particles),
                dtype=INDEX_DTYPE,
            )
            leaf_particle_mask = jnp.zeros((leaf_count + 1, max_particles), dtype=bool)
            leaf_particle_indices = leaf_particle_indices.at[row, col].set(
                jnp.where(sorted_valid, sorted_particle_idx, 0),
                mode="drop",
            )
            leaf_particle_mask = leaf_particle_mask.at[row, col].set(
                sorted_valid,
                mode="drop",
            )
            leaf_particle_indices = leaf_particle_indices[:leaf_count]
            leaf_particle_mask = leaf_particle_mask[:leaf_count]
            particle_to_leaf_position = jnp.zeros(
                (tree.positions_sorted.shape[0],),
                dtype=INDEX_DTYPE,
            )
            particle_to_leaf_position = particle_to_leaf_position.at[
                flat_particle_idx[flat_valid]
            ].set(flat_carrier_pos[flat_valid])
        else:
            leaf_particle_indices = jnp.zeros((leaf_count, 0), dtype=INDEX_DTYPE)
            leaf_particle_mask = jnp.zeros((leaf_count, 0), dtype=bool)
            particle_to_leaf_position = jnp.zeros(
                (tree.positions_sorted.shape[0],),
                dtype=INDEX_DTYPE,
            )

        native_neighbor_leaf_positions = getattr(
            native_neighbors,
            "neighbor_leaf_positions",
            None,
        )
        if native_neighbor_leaf_positions is not None:
            neighbor_leaf_positions = jnp.asarray(
                native_neighbor_leaf_positions,
                dtype=INDEX_DTYPE,
            )
        else:
            if leaf_count > 0:
                max_nbr = int(jnp.max(native_counts))
            else:
                max_nbr = 0
            if max_nbr > 0:
                nbr_offsets = jnp.arange(max_nbr, dtype=INDEX_DTYPE)
                nbr_idx = native_offsets[:-1, None] + nbr_offsets[None, :]
                nbr_valid = nbr_offsets[None, :] < native_counts[:, None]
                nbr_safe_idx = jnp.where(nbr_valid, nbr_idx, 0)
                nbr_nodes = native_neighbors_flat[nbr_safe_idx]
                neighbor_leaf_positions = carrier_lookup[nbr_nodes]
                neighbor_leaf_positions = jnp.where(
                    nbr_valid,
                    neighbor_leaf_positions,
                    jnp.asarray(-1, dtype=INDEX_DTYPE),
                )
            else:
                neighbor_leaf_positions = jnp.zeros((leaf_count, 0), dtype=INDEX_DTYPE)

        oct_node_ranges = jnp.asarray(octree.node_ranges, dtype=INDEX_DTYPE)
        particle_order_leaf_indices = jnp.asarray(
            native_neighbors.particle_order_leaf_indices,
            dtype=INDEX_DTYPE,
        )
        return NearfieldInteropData(
            leaf_nodes=leaf_nodes,
            node_ranges=oct_node_ranges,
            offsets=native_offsets,
            neighbors=native_neighbors_flat,
            counts=native_counts,
            particle_order_node_ranges=oct_node_ranges,
            particle_order_leaf_indices=particle_order_leaf_indices,
            particle_order_to_native_leaf=jnp.asarray(
                native_neighbors.particle_order_to_native_leaf,
                dtype=INDEX_DTYPE,
            ),
            leaf_particle_indices=leaf_particle_indices,
            leaf_particle_mask=leaf_particle_mask,
            particle_to_leaf_position=particle_to_leaf_position,
            neighbor_leaf_positions=neighbor_leaf_positions,
        )

    del octree
    leaf_indices = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    particle_order_leaf_indices = jnp.asarray(
        getattr(
            neighbor_list, "particle_order_leaf_indices", neighbor_list.leaf_indices
        ),
        dtype=INDEX_DTYPE,
    )
    nbr_counts = jnp.asarray(neighbor_list.counts, dtype=INDEX_DTYPE)
    num_leaves = int(leaf_indices.shape[0])
    payload_neighbor_leaf_positions = getattr(
        neighbor_list,
        "neighbor_leaf_positions",
        None,
    )
    if payload_neighbor_leaf_positions is not None:
        neighbor_leaf_positions = jnp.asarray(
            payload_neighbor_leaf_positions,
            dtype=INDEX_DTYPE,
        )
    else:
        if num_leaves > 0:
            max_nbr = int(jnp.max(nbr_counts))
        else:
            max_nbr = 0
        if max_nbr > 0:
            total_nodes = int(tree.node_ranges.shape[0])
            leaf_lookup = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
            leaf_lookup = leaf_lookup.at[leaf_indices].set(
                jnp.arange(num_leaves, dtype=INDEX_DTYPE)
            )
            offsets = jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE)
            neighbors = jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE)
            nbr_offsets = jnp.arange(max_nbr, dtype=INDEX_DTYPE)
            nbr_idx = offsets[:-1, None] + nbr_offsets[None, :]
            nbr_valid = nbr_offsets[None, :] < nbr_counts[:, None]
            nbr_safe_idx = jnp.where(nbr_valid, nbr_idx, 0)
            nbr_nodes = neighbors[nbr_safe_idx]
            neighbor_leaf_positions = leaf_lookup[nbr_nodes]
            neighbor_leaf_positions = jnp.where(
                nbr_valid,
                neighbor_leaf_positions,
                jnp.asarray(-1, dtype=INDEX_DTYPE),
            )
        else:
            neighbor_leaf_positions = jnp.zeros((num_leaves, 0), dtype=INDEX_DTYPE)

    return NearfieldInteropData(
        leaf_nodes=leaf_indices,
        node_ranges=jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        offsets=jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE),
        neighbors=jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE),
        counts=jnp.asarray(neighbor_list.counts, dtype=INDEX_DTYPE),
        particle_order_node_ranges=jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        particle_order_leaf_indices=particle_order_leaf_indices,
        particle_order_to_native_leaf=jnp.asarray(
            getattr(
                neighbor_list,
                "particle_order_to_native_leaf",
                jnp.arange(leaf_indices.shape[0], dtype=INDEX_DTYPE),
            ),
            dtype=INDEX_DTYPE,
        ),
        leaf_particle_indices=None,
        leaf_particle_mask=None,
        particle_to_leaf_position=None,
        neighbor_leaf_positions=neighbor_leaf_positions,
    )


def _prepare_tree_evaluation_inputs(
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    locals_or_downward: Union[LocalExpansionData, TreeDownwardData],
    neighbor_list: NodeNeighborList,
    *,
    farfield_local_data: Optional[LocalExpansionData],
    farfield_leaf_nodes: Optional[Array],
    farfield_node_ranges: Optional[Array],
    max_leaf_size: Optional[int],
    return_potential: bool,
) -> _TreeEvaluationSetup:
    """Validate and normalize tree-evaluation inputs for eager/JIT paths."""
    locals_data = (
        locals_or_downward.locals
        if isinstance(locals_or_downward, TreeDownwardData)
        else locals_or_downward
    )
    farfield_locals = (
        locals_data if farfield_local_data is None else farfield_local_data
    )
    node_views = _resolve_evaluation_node_views(
        tree,
        neighbor_list,
        farfield_leaf_nodes=farfield_leaf_nodes,
        farfield_node_ranges=farfield_node_ranges,
    )

    if farfield_locals.centers.shape[0] != node_views.farfield_node_ranges.shape[0]:
        raise ValueError("local expansions must align with evaluation node ranges")
    if (
        farfield_locals.coefficients.shape[0]
        != node_views.farfield_node_ranges.shape[0]
    ):
        raise ValueError("local expansions must align with evaluation node ranges")

    positions = jnp.asarray(positions_sorted)
    masses = jnp.asarray(masses_sorted)
    leaf_nodes = node_views.farfield_leaf_nodes
    node_ranges = node_views.farfield_node_ranges

    if leaf_nodes.size == 0:
        zeros = jnp.zeros_like(positions)
        if return_potential:
            pot_zeros = jnp.zeros((positions.shape[0],), dtype=zeros.dtype)
            empty: Optional[Union[Array, Tuple[Array, Array]]] = (
                zeros,
                pot_zeros,
            )
        else:
            empty = zeros
        resolved_max_leaf = 0 if max_leaf_size is None else int(max_leaf_size)
        return _TreeEvaluationSetup(
            farfield_locals,
            positions,
            masses,
            leaf_nodes,
            node_ranges,
            resolved_max_leaf,
            empty,
        )
    if max_leaf_size is None:
        leaf_ranges = node_ranges[leaf_nodes]
        counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
        try:
            resolved_max_leaf = int(jnp.max(counts).item())
        except TypeError as exc:
            raise ValueError(
                "max_leaf_size must be provided when tracing or JIT-compiling"
            ) from exc
    else:
        resolved_max_leaf = int(max_leaf_size)

    return _TreeEvaluationSetup(
        farfield_locals,
        positions,
        masses,
        leaf_nodes,
        node_ranges,
        resolved_max_leaf,
        None,
    )


@partial(jax.jit, static_argnames=("order", "rotation"))
def _m2l_complex_batch_kernel(
    src_mult: Array,
    deltas: Array,
    *,
    order: int,
    rotation: str,
) -> Array:
    """Vectorized complex-basis M2L kernel for one interaction batch."""
    return m2l_complex_reference_batch(
        src_mult,
        deltas,
        order=order,
        rotation=rotation,
    )


@partial(jax.jit, static_argnames=("order",))
def _m2l_complex_batch_cached_kernel(
    src_mult: Array,
    deltas: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    *,
    order: int,
) -> Array:
    """Vectorized complex M2L kernel using precomputed rotation blocks."""
    return m2l_complex_reference_batch_cached_blocks(
        src_mult,
        deltas,
        blocks_to_z,
        blocks_from_z,
        order=order,
    )


def _m2l_cached_kernel_dispatch(
    src_mult: Array,
    deltas: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    *,
    order: int,
    basis_mode: str,
) -> Array:
    """Apply precomputed rotation blocks in the complex or real basis.

    ``basis_mode`` is a Python string (static under jit), so this branches at
    trace time. The real branch uses the Dehnen no-sqrt2 cached kernel; the
    complex branch is unchanged.
    """
    if str(basis_mode).strip().lower() == "real":
        return m2l_rot_scale_real_batch_cached_blocks(
            src_mult, deltas, blocks_to_z, blocks_from_z, order=order
        )
    return _m2l_complex_batch_cached_kernel(
        src_mult, deltas, blocks_to_z, blocks_from_z, order=order
    )


def _operator_cache_get(key: tuple) -> Optional[tuple[Array, Array]]:
    """Read cached grouped-rotation blocks and update LRU order."""
    blocks = _operator_blocks_cache.get(key)
    if blocks is None:
        return None
    _operator_blocks_cache.move_to_end(key)
    return blocks


def _operator_cache_put(key: tuple, value: tuple[Array, Array]) -> None:
    """Insert grouped-rotation blocks and evict stale entries by LRU policy."""
    _operator_blocks_cache[key] = value
    _operator_blocks_cache.move_to_end(key)
    while len(_operator_blocks_cache) > _OPERATOR_CACHE_MAX:
        _operator_blocks_cache.popitem(last=False)


def _rotation_blocks_for_grouped_classes(
    *,
    order: int,
    rotation: str,
    class_keys: Array,
    class_deltas: Array,
    dtype: jnp.dtype,
    basis_mode: str = "complex",
) -> tuple[Array, Array]:
    """Resolve rotation blocks for all grouped classes with cache reuse.

    For ``basis_mode == "real"`` the Dehnen no-sqrt2 real rotation blocks are
    built (multipole world->z and local z->world) and the ``rotation`` argument
    is ignored (the real path has a single rotation construction).
    """
    num_classes = int(class_deltas.shape[0])
    max_m = 2 * int(order) + 1
    empty_shape = (0, int(order) + 1, max_m, max_m)
    if num_classes == 0:
        empty = jnp.zeros(empty_shape, dtype=dtype)
        return empty, empty

    real_basis = str(basis_mode).strip().lower() == "real"
    cache_key = _grouped_operator_cache_key(
        order=order,
        rotation=("real" if real_basis else rotation),
        dtype=dtype,
        class_keys=class_keys,
        class_deltas=class_deltas,
    )
    if cache_key is not None:
        cached = _grouped_operator_cache_get(cache_key)
        if cached is not None:
            return cached

    deltas = jnp.asarray(class_deltas)
    if real_basis:
        blocks_to = real_rotation_blocks_to_z_multipole_batch(
            deltas, order=order, dtype=dtype
        )
        blocks_from = real_rotation_blocks_from_z_local_batch(
            deltas, order=order, dtype=dtype
        )
        if cache_key is not None:
            _grouped_operator_cache_put(cache_key, (blocks_to, blocks_from))
        return blocks_to, blocks_from

    if rotation == "solidfmm":
        blocks_to = complex_rotation_blocks_to_z_solidfmm_batch(
            deltas,
            order=order,
            basis="multipole",
            dtype=dtype,
        )
        blocks_from = complex_rotation_blocks_from_z_solidfmm_batch(
            deltas,
            order=order,
            basis="local",
            dtype=dtype,
        )
    else:
        raise ValueError(
            "grouped operator cache currently supports rotation='solidfmm'"
        )
    if cache_key is not None:
        _grouped_operator_cache_put(cache_key, (blocks_to, blocks_from))
    return blocks_to, blocks_from


def _chunk_segment_scatter_add(
    local_accum: Array,
    contribs: Array,
    tgt_chunk: Array,
    valid: Array,
    *,
    chunk_size: int,
) -> Array:
    """Reduce one fixed-width chunk by target index and scatter-add into locals."""
    masked_targets = jnp.where(valid, tgt_chunk, jnp.iinfo(INDEX_DTYPE).max)
    sort_idx = jnp.argsort(masked_targets)
    sorted_keys = masked_targets[sort_idx]
    tgt_sorted = tgt_chunk[sort_idx]
    contribs_sorted = contribs[sort_idx]
    valid_sorted = valid[sort_idx]

    contribs_sorted = jnp.where(valid_sorted[:, None], contribs_sorted, 0)
    new_group = jnp.concatenate(
        (
            jnp.asarray([True], dtype=bool),
            sorted_keys[1:] != sorted_keys[:-1],
        ),
        axis=0,
    )
    group_ids = jnp.cumsum(new_group.astype(INDEX_DTYPE)) - jnp.asarray(
        1,
        dtype=INDEX_DTYPE,
    )
    reduced = jax.ops.segment_sum(contribs_sorted, group_ids, chunk_size)

    unique_targets = jnp.zeros((chunk_size,), dtype=INDEX_DTYPE)
    unique_targets = unique_targets.at[group_ids].set(tgt_sorted)
    unique_valid = jnp.zeros((chunk_size,), dtype=bool)
    unique_valid = unique_valid.at[group_ids].set(valid_sorted)
    safe_targets = jnp.where(unique_valid, unique_targets, 0)
    reduced = jnp.where(unique_valid[:, None], reduced, 0)
    return local_accum.at[safe_targets].add(reduced)


@partial(
    jax.jit,
    static_argnames=("order", "total_nodes", "chunk_size", "basis_mode"),
    donate_argnums=(0,),
)
def _accumulate_solidfmm_m2l_grouped_chunked_scan(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    src_sorted: Array,
    tgt_sorted: Array,
    class_ids_sorted: Array,
    blocks_to_classes: Array,
    blocks_from_classes: Array,
    *,
    order: int,
    total_nodes: int,
    chunk_size: int,
    basis_mode: str = "complex",
) -> Array:
    """Accumulate grouped solidfmm M2L contributions via chunked scan."""
    pair_count = src_sorted.shape[0]
    starts = jnp.arange(0, pair_count, chunk_size, dtype=INDEX_DTYPE)

    def body(local_accum: Array, start_idx: Array) -> tuple[Array, None]:
        offset = jnp.arange(chunk_size, dtype=INDEX_DTYPE)
        idx = start_idx + offset
        valid = idx < pair_count
        safe_idx = jnp.where(valid, idx, 0)

        src_chunk = src_sorted[safe_idx]
        tgt_chunk = tgt_sorted[safe_idx]
        cls_chunk = class_ids_sorted[safe_idx]
        src_mult = multip_packed[src_chunk]
        deltas = centers[tgt_chunk] - centers[src_chunk]
        blocks_to = blocks_to_classes[cls_chunk]
        blocks_from = blocks_from_classes[cls_chunk]

        contribs = _m2l_cached_kernel_dispatch(
            src_mult,
            deltas,
            blocks_to,
            blocks_from,
            order=order,
            basis_mode=basis_mode,
        ).astype(locals_coeffs.dtype)
        local_accum = _chunk_segment_scatter_add(
            local_accum,
            contribs,
            tgt_chunk,
            valid,
            chunk_size=chunk_size,
        )
        return local_accum, None

    local_accum, _ = jax.lax.scan(body, locals_coeffs, starts)
    return local_accum


@partial(
    jax.jit,
    static_argnames=("order", "total_nodes", "basis_mode"),
    donate_argnums=(0,),
)
def _accumulate_solidfmm_m2l_grouped_fullbatch(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    src_sorted: Array,
    tgt_sorted: Array,
    class_ids_sorted: Array,
    blocks_to_classes: Array,
    blocks_from_classes: Array,
    *,
    order: int,
    total_nodes: int,
    basis_mode: str = "complex",
) -> Array:
    """Accumulate grouped solidfmm M2L contributions in one full batch."""
    src_mult = multip_packed[src_sorted]
    deltas = centers[tgt_sorted] - centers[src_sorted]
    blocks_to = blocks_to_classes[class_ids_sorted]
    blocks_from = blocks_from_classes[class_ids_sorted]
    contribs = _m2l_cached_kernel_dispatch(
        src_mult,
        deltas,
        blocks_to,
        blocks_from,
        order=order,
        basis_mode=basis_mode,
    ).astype(locals_coeffs.dtype)
    return locals_coeffs + jax.ops.segment_sum(contribs, tgt_sorted, total_nodes)


def _build_grouped_class_segments(
    grouped: GroupedInteractionBuffers,
    *,
    chunk_size: int,
) -> tuple[Array, Array, Array]:
    """Build compact class-major segment metadata for chunked execution."""
    cache_key = _grouped_segment_cache_key(
        class_offsets=grouped.class_offsets,
        class_targets=grouped.class_targets,
        chunk_size=int(chunk_size),
    )
    if cache_key is not None:
        cached = _grouped_segment_cache_get(cache_key)
        if cached is not None:
            return cached

    class_offsets = np.asarray(jax.device_get(grouped.class_offsets), dtype=np.int64)
    if class_offsets.size <= 1:
        empty = jnp.zeros((0,), dtype=INDEX_DTYPE)
        result = (empty, empty, empty)
        if cache_key is not None:
            _grouped_segment_cache_put(cache_key, result)
        return result

    starts: list[int] = []
    lengths: list[int] = []
    class_ids: list[int] = []
    for class_idx in range(class_offsets.shape[0] - 1):
        start = int(class_offsets[class_idx])
        end = int(class_offsets[class_idx + 1])
        while start < end:
            seg_len = min(int(chunk_size), end - start)
            starts.append(start)
            lengths.append(seg_len)
            class_ids.append(class_idx)
            start += seg_len

    if len(starts) == 0:
        result = (
            jnp.asarray(starts, dtype=INDEX_DTYPE),
            jnp.asarray(lengths, dtype=INDEX_DTYPE),
            jnp.asarray(class_ids, dtype=INDEX_DTYPE),
        )
        if cache_key is not None:
            _grouped_segment_cache_put(cache_key, result)
        return result

    result = (
        jnp.asarray(starts, dtype=INDEX_DTYPE),
        jnp.asarray(lengths, dtype=INDEX_DTYPE),
        jnp.asarray(class_ids, dtype=INDEX_DTYPE),
    )
    if cache_key is not None:
        _grouped_segment_cache_put(cache_key, result)
    return result


@partial(
    jax.jit,
    static_argnames=("order", "total_nodes", "chunk_size", "basis_mode"),
    donate_argnums=(0,),
)
def _accumulate_solidfmm_m2l_class_major_chunked_scan(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    src_sorted: Array,
    tgt_sorted: Array,
    segment_starts: Array,
    segment_lengths: Array,
    segment_class_ids: Array,
    blocks_to_classes: Array,
    blocks_from_classes: Array,
    *,
    order: int,
    total_nodes: int,
    chunk_size: int,
    basis_mode: str = "complex",
) -> Array:
    """Accumulate class-major grouped M2L contributions via chunked scan."""
    num_segments = segment_starts.shape[0]
    if num_segments == 0:
        return locals_coeffs

    offsets = jnp.arange(chunk_size, dtype=INDEX_DTYPE)

    def body(local_accum: Array, seg_idx: Array) -> tuple[Array, None]:
        start = segment_starts[seg_idx]
        seg_len = segment_lengths[seg_idx]
        cls = segment_class_ids[seg_idx]
        idx = start + offsets
        valid = offsets < seg_len
        safe_idx = jnp.where(valid, idx, 0)

        src_chunk = src_sorted[safe_idx]
        tgt_chunk = tgt_sorted[safe_idx]
        src_mult = multip_packed[src_chunk]
        deltas = centers[tgt_chunk] - centers[src_chunk]

        block_to = blocks_to_classes[cls]
        block_from = blocks_from_classes[cls]
        blocks_to = jnp.broadcast_to(block_to, (chunk_size,) + block_to.shape)
        blocks_from = jnp.broadcast_to(block_from, (chunk_size,) + block_from.shape)

        contribs = _m2l_cached_kernel_dispatch(
            src_mult,
            deltas,
            blocks_to,
            blocks_from,
            order=order,
            basis_mode=basis_mode,
        ).astype(locals_coeffs.dtype)
        contribs = jnp.where(valid[:, None], contribs, 0)
        masked_targets = jnp.where(valid, tgt_chunk, jnp.iinfo(INDEX_DTYPE).max)
        sort_idx = jnp.argsort(masked_targets)
        tgt_sorted_chunk = tgt_chunk[sort_idx]
        contribs_sorted = contribs[sort_idx]
        valid_sorted = valid[sort_idx]
        prev_targets = jnp.concatenate(
            [
                jnp.asarray([-1], dtype=INDEX_DTYPE),
                tgt_sorted_chunk[:-1],
            ]
        )
        group_starts = valid_sorted & (
            jnp.logical_not(jnp.roll(valid_sorted, 1))
            | (tgt_sorted_chunk != prev_targets)
        )
        group_ids = jnp.cumsum(group_starts.astype(INDEX_DTYPE)) - 1
        safe_group_ids = jnp.where(valid_sorted, group_ids, 0)
        reduced = jax.ops.segment_sum(contribs_sorted, safe_group_ids, chunk_size)
        unique_targets = jnp.where(valid_sorted, tgt_sorted_chunk, 0)
        return local_accum.at[unique_targets].add(reduced), None

    local_accum, _ = jax.lax.scan(
        body,
        locals_coeffs,
        jnp.arange(num_segments, dtype=INDEX_DTYPE),
    )
    return local_accum


def _accumulate_solidfmm_m2l_grouped_class_major(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    grouped: GroupedInteractionBuffers,
    grouped_segment_starts: Optional[Array],
    grouped_segment_lengths: Optional[Array],
    grouped_segment_class_ids: Optional[Array],
    grouped_segment_sort_permutation: Optional[Array],
    grouped_segment_group_ids: Optional[Array],
    grouped_segment_unique_targets: Optional[Array],
    *,
    order: int,
    rotation: str,
    total_nodes: int,
    chunk_size: int,
    basis_mode: str = "complex",
) -> Array:
    """Class-major grouped accumulation without per-pair operator gathers."""
    del (
        grouped_segment_sort_permutation,
        grouped_segment_group_ids,
        grouped_segment_unique_targets,
    )

    if rotation not in ("solidfmm",):
        src = grouped.class_sources
        tgt = grouped.class_targets
        return _accumulate_solidfmm_m2l_fullbatch(
            locals_coeffs,
            multip_packed,
            centers,
            src,
            tgt,
            jnp.asarray(src.shape[0], dtype=INDEX_DTYPE),
            order=order,
            rotation=rotation,
            total_nodes=total_nodes,
        )

    blocks_to_classes, blocks_from_classes = _rotation_blocks_for_grouped_classes(
        order=order,
        rotation=rotation,
        class_keys=jnp.asarray(grouped.class_keys, dtype=jnp.int32),
        class_deltas=jnp.asarray(grouped.class_displacements),
        dtype=multip_packed.dtype,
        basis_mode=basis_mode,
    )
    if (
        grouped_segment_starts is None
        or grouped_segment_lengths is None
        or grouped_segment_class_ids is None
    ):
        (
            segment_starts,
            segment_lengths,
            segment_class_ids,
        ) = _build_grouped_class_segments(
            grouped,
            chunk_size=int(chunk_size),
        )
    else:
        segment_starts = jnp.asarray(grouped_segment_starts, dtype=INDEX_DTYPE)
        segment_lengths = jnp.asarray(grouped_segment_lengths, dtype=INDEX_DTYPE)
        segment_class_ids = jnp.asarray(grouped_segment_class_ids, dtype=INDEX_DTYPE)
    return _accumulate_solidfmm_m2l_class_major_chunked_scan(
        locals_coeffs,
        multip_packed,
        centers,
        jnp.asarray(grouped.class_sources, dtype=INDEX_DTYPE),
        jnp.asarray(grouped.class_targets, dtype=INDEX_DTYPE),
        segment_starts,
        segment_lengths,
        segment_class_ids,
        blocks_to_classes,
        blocks_from_classes,
        order=order,
        total_nodes=total_nodes,
        chunk_size=int(chunk_size),
        basis_mode=basis_mode,
    )


def _accumulate_solidfmm_m2l_grouped(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    grouped: GroupedInteractionBuffers,
    *,
    order: int,
    rotation: str,
    total_nodes: int,
    chunk_size: int,
    basis_mode: str = "complex",
) -> Array:
    """Grouped M2L accumulation using cached class blocks and pair chunking."""

    if rotation not in ("solidfmm",):
        # Keep existing sparse path semantics for other conventions.
        src = grouped.class_sources
        tgt = grouped.class_targets
        return _accumulate_solidfmm_m2l_fullbatch(
            locals_coeffs,
            multip_packed,
            centers,
            src,
            tgt,
            jnp.asarray(src.shape[0], dtype=INDEX_DTYPE),
            order=order,
            rotation=rotation,
            total_nodes=total_nodes,
        )

    src_sorted = grouped.class_sources
    tgt_sorted = grouped.class_targets
    class_ids = jnp.asarray(grouped.class_ids, dtype=INDEX_DTYPE)
    class_keys = jnp.asarray(grouped.class_keys, dtype=jnp.int32)
    class_deltas = jnp.asarray(grouped.class_displacements)

    blocks_to_classes, blocks_from_classes = _rotation_blocks_for_grouped_classes(
        order=order,
        rotation=rotation,
        class_keys=class_keys,
        class_deltas=class_deltas,
        dtype=multip_packed.dtype,
        basis_mode=basis_mode,
    )
    if int(src_sorted.shape[0]) <= min(int(chunk_size), _M2L_FULLBATCH_MAX_PAIRS):
        return _accumulate_solidfmm_m2l_grouped_fullbatch(
            locals_coeffs,
            multip_packed,
            centers,
            src_sorted,
            tgt_sorted,
            class_ids,
            blocks_to_classes,
            blocks_from_classes,
            order=order,
            total_nodes=total_nodes,
        )
    return _accumulate_solidfmm_m2l_grouped_chunked_scan(
        locals_coeffs,
        multip_packed,
        centers,
        src_sorted,
        tgt_sorted,
        class_ids,
        blocks_to_classes,
        blocks_from_classes,
        order=order,
        total_nodes=total_nodes,
        chunk_size=int(chunk_size),
        basis_mode=basis_mode,
    )


@partial(jax.jit, static_argnames=("order", "rotation"))
def _l2l_complex_batch_kernel(
    coeffs: Array,
    deltas: Array,
    *,
    order: int,
    rotation: str,
) -> Array:
    """Vectorized complex-basis L2L translation kernel."""
    return l2l_complex_batch(coeffs, deltas, order=order, rotation=rotation)


@partial(jax.jit, static_argnames=("order", "m2l_impl"))
def _m2l_real_batch_kernel(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    m2l_impl: str,
) -> Array:
    """Vectorized real-basis M2L translation kernel."""
    mode = str(m2l_impl).strip().lower()
    if mode != "rot_scale":
        raise ValueError("real-basis m2l_impl must be 'rot_scale'")
    return m2l_rot_scale_real_batch(multipoles, deltas, order=order, use_pallas=False)


@partial(jax.jit, static_argnames=("order", "m2l_impl"))
def _m2l_real_batch_kernel_pallas(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    m2l_impl: str,
) -> Array:
    """Vectorized real-basis M2L translation kernel using the optional Pallas core."""
    mode = str(m2l_impl).strip().lower()
    if mode != "rot_scale":
        raise ValueError("real-basis m2l_impl must be 'rot_scale'")
    return m2l_rot_scale_real_batch(multipoles, deltas, order=order, use_pallas=True)


def _real_m2l_pallas_active() -> bool:
    """Whether to route the real-basis M2L z-core through the Pallas kernel.

    Gated by ``JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS`` and the sm_80+ real-M2L
    support check (falls back to the pure-JAX rot-scale otherwise). Trace-time;
    the flag does not change within a compiled run.
    """
    flag = (
        str(os.environ.get("JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS", "0"))
        .strip()
        .lower()
    )
    if flag not in {"1", "true", "yes", "on"}:
        return False
    try:
        from jaccpot.pallas.m2l_core_z_real import pallas_m2l_real_supported

        return bool(pallas_m2l_real_supported())
    except Exception:
        return False


def _m2l_real_batch_kernel_fused_pallas(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    m2l_impl: str,
) -> Array:
    """Real-basis M2L via the FULLY-fused Pallas kernel (rotate+z-translate+rotate
    in one launch). Builds the real rotation blocks + radii from deltas."""
    mode = str(m2l_impl).strip().lower()
    if mode != "rot_scale":
        raise ValueError("real-basis m2l_impl must be 'rot_scale'")
    from jaccpot.operators.m2l_real_rot_scale import (
        real_rotation_blocks_from_z_local_batch,
        real_rotation_blocks_to_z_multipole_batch,
    )
    from jaccpot.pallas.m2l_real_fused import m2l_real_fused_pallas

    r = jnp.linalg.norm(deltas, axis=1)
    bto = real_rotation_blocks_to_z_multipole_batch(
        deltas, order=order, dtype=multipoles.dtype
    )
    bfr = real_rotation_blocks_from_z_local_batch(
        deltas, order=order, dtype=multipoles.dtype
    )
    return m2l_real_fused_pallas(multipoles, bto, bfr, r, order=order)


def _apply_real_m2l(src_mult, deltas, *, order, m2l_impl):
    """Real-basis batched M2L: fully-fused Pallas kernel when enabled, else pure-JAX.

    When the fused-M2L Pallas flag is active, route through the single-launch fused
    kernel (rotate -> z-translate -> rotate-back on-chip), collapsing the per-pair
    JAX rotation launches. Otherwise the pure-JAX rot-scale path.
    """
    if _real_m2l_pallas_active():
        return _m2l_real_batch_kernel_fused_pallas(
            src_mult, deltas, order=order, m2l_impl=m2l_impl
        )
    return _m2l_real_batch_kernel(src_mult, deltas, order=order, m2l_impl=m2l_impl)


@partial(jax.jit, static_argnames=("order",))
def _l2l_real_batch_kernel(
    coeffs: Array,
    deltas: Array,
    *,
    order: int,
) -> Array:
    """Vectorized real-basis L2L translation kernel."""
    return jax.vmap(lambda c, d: l2l_real(c, d, order=order))(coeffs, deltas)


@partial(
    jax.jit,
    static_argnames=("order", "rotation", "total_nodes"),
    donate_argnums=(0,),
)
def _accumulate_solidfmm_m2l_fullbatch(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    src: Array,
    tgt: Array,
    active_pair_count: Array,
    *,
    order: int,
    rotation: str,
    total_nodes: int,
) -> Array:
    """Accumulate solidfmm M2L contributions in one full interaction batch."""
    idx = jnp.arange(src.shape[0], dtype=INDEX_DTYPE)
    raw_src = src
    raw_tgt = tgt
    valid = (idx < active_pair_count) & (raw_src >= 0) & (raw_tgt >= 0)
    safe_src = jnp.where(valid, raw_src, 0)
    safe_tgt = jnp.where(valid, raw_tgt, 0)
    src_mult = multip_packed[safe_src]
    deltas = centers[safe_tgt] - centers[safe_src]

    contribs = _m2l_complex_batch_kernel(
        src_mult,
        deltas,
        order=order,
        rotation=rotation,
    )
    contribs = contribs.astype(locals_coeffs.dtype)
    contribs = jnp.where(valid[:, None], contribs, 0)
    return locals_coeffs + jax.ops.segment_sum(contribs, safe_tgt, total_nodes)


@partial(
    jax.jit,
    static_argnames=("order", "rotation", "total_nodes", "chunk_size"),
    donate_argnums=(0,),
)
def _accumulate_solidfmm_m2l_chunked_scan(
    locals_coeffs: Array,
    multip_packed: Array,
    centers: Array,
    src: Array,
    tgt: Array,
    active_pair_count: Array,
    *,
    order: int,
    rotation: str,
    total_nodes: int,
    chunk_size: int,
) -> Array:
    """Accumulate solidfmm M2L contributions with chunked scan reduction."""
    pair_count = src.shape[0]
    starts = jnp.arange(0, pair_count, chunk_size, dtype=INDEX_DTYPE)

    def body(local_accum: Array, start_idx: Array) -> tuple[Array, None]:
        def active_chunk(accum: Array) -> Array:
            offset = jnp.arange(chunk_size, dtype=INDEX_DTYPE)
            idx = start_idx + offset
            valid = idx < pair_count
            safe_idx = jnp.where(valid, idx, 0)

            src_chunk_raw = src[safe_idx]
            tgt_chunk_raw = tgt[safe_idx]
            valid = (
                valid
                & (idx < active_pair_count)
                & (src_chunk_raw >= 0)
                & (tgt_chunk_raw >= 0)
            )
            src_chunk = jnp.where(valid, src_chunk_raw, 0)
            tgt_chunk = jnp.where(valid, tgt_chunk_raw, 0)
            src_mult = multip_packed[src_chunk]
            deltas = centers[tgt_chunk] - centers[src_chunk]

            contribs = _m2l_complex_batch_kernel(
                src_mult,
                deltas,
                order=order,
                rotation=rotation,
            )

            contribs = contribs.astype(locals_coeffs.dtype)
            return _chunk_segment_scatter_add(
                accum,
                contribs,
                tgt_chunk,
                valid,
                chunk_size=chunk_size,
            )

        local_accum = jax.lax.cond(
            start_idx < active_pair_count,
            active_chunk,
            lambda accum: accum,
            local_accum,
        )
        return local_accum, None

    local_accum, _ = jax.lax.scan(body, locals_coeffs, starts)
    return local_accum


@partial(
    jax.jit,
    static_argnames=("order", "rotation", "total_nodes"),
    donate_argnums=(0,),
)
def _propagate_solidfmm_locals_to_children(
    coeffs_local: Array,
    centers_local: Array,
    left_child: Array,
    right_child: Array,
    *,
    order: int,
    rotation: str,
    total_nodes: int,
) -> Array:
    """Apply solidfmm L2L translations from parents to their children."""
    num_internal_nodes = left_child.shape[0]
    parent_idx = jnp.arange(num_internal_nodes, dtype=INDEX_DTYPE)
    child_idx = jnp.concatenate(
        [left_child[:num_internal_nodes], right_child[:num_internal_nodes]],
        axis=0,
    )
    parent_rep = jnp.concatenate([parent_idx, parent_idx], axis=0)
    valid = child_idx >= 0
    safe_child_idx = jnp.where(valid, child_idx, 0)

    parent_coeffs = coeffs_local[parent_rep]
    deltas = centers_local[safe_child_idx] - centers_local[parent_rep]
    translated = _l2l_complex_batch_kernel(
        parent_coeffs,
        deltas,
        order=order,
        rotation=rotation,
    )
    translated = translated.astype(coeffs_local.dtype)
    translated = jnp.where(valid[:, None], translated, 0)
    updates = jax.ops.segment_sum(translated, safe_child_idx, total_nodes)
    return coeffs_local + updates


@partial(
    jax.jit,
    static_argnames=("order", "rotation", "total_nodes", "basis_mode"),
    donate_argnums=(0,),
)
def _propagate_solidfmm_locals_by_level(
    coeffs_local: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    node_levels: Array,
    *,
    order: int,
    rotation: str,
    total_nodes: int,
    basis_mode: str = "complex",
) -> Array:
    """Top-down, level-by-level L2L cascade over a binary tree.

    A single parent->child pass (``_propagate_solidfmm_locals_to_children``)
    moves each node's local expansion down exactly one level. That is only
    sufficient when every node already carries the far-field appropriate to its
    own level. A local expansion deposited high in the tree (a well-separated
    interaction accepted at a coarse node) must instead cascade through every
    intermediate level to reach the leaves, or the leaves never see it and the
    evaluated field degrades with tree depth.

    Iterate levels root->leaf and translate only the parents that live at the
    current level, so each node's fully-accumulated expansion (its own plus
    everything inherited from shallower ancestors) is propagated to its children
    before those children are used as parents in turn.
    """
    num_internal = int(left_child.shape[0])
    if num_internal <= 0:
        return coeffs_local

    real_basis = str(basis_mode).strip().lower() == "real"
    left_internal = left_child[:num_internal]
    right_internal = right_child[:num_internal]
    parent_levels = node_levels[:num_internal].astype(INDEX_DTYPE)
    parent_idx = jnp.arange(num_internal, dtype=INDEX_DTYPE)
    parent_rep = jnp.concatenate([parent_idx, parent_idx], axis=0)
    max_level = jnp.max(parent_levels)
    minus_one = jnp.asarray(-1, dtype=left_internal.dtype)

    def level_body(level: Array, state: Array) -> Array:
        active = parent_levels == level
        lc = jnp.where(active, left_internal, minus_one)
        rc = jnp.where(active, right_internal, minus_one)
        child_idx = jnp.concatenate([lc, rc], axis=0)
        valid = child_idx >= 0
        safe_child = jnp.where(valid, child_idx, 0)
        parent_coeffs = state[parent_rep]
        # L2L uses the old_center - new_center (parent - child) displacement in
        # BOTH bases. The complex path previously used child - parent here,
        # which is the wrong sign: the far field was left uncorrected in
        # proportion to the cascade depth, capping accuracy (~3e-3 at
        # theta>=0.5) regardless of expansion order, while looking fine at small
        # theta where the L2L cascade is shallow.
        deltas = centers[parent_rep] - centers[safe_child]
        if real_basis:
            translated = _l2l_real_batch_kernel(
                parent_coeffs, deltas, order=order
            ).astype(state.dtype)
        else:
            translated = _l2l_complex_batch_kernel(
                parent_coeffs, deltas, order=order, rotation=rotation
            ).astype(state.dtype)
        translated = jnp.where(valid[:, None], translated, 0)
        updates = jax.ops.segment_sum(translated, safe_child, total_nodes)
        return state + updates

    return jax.lax.fori_loop(0, max_level + 1, level_body, coeffs_local)


@partial(
    jax.jit,
    static_argnames=("order", "m2l_impl", "total_nodes"),
    donate_argnums=(0,),
)
def _accumulate_real_m2l_fullbatch(
    locals_coeffs: Array,
    multip_packed_real: Array,
    centers: Array,
    src: Array,
    tgt: Array,
    active_pair_count: Array,
    *,
    order: int,
    m2l_impl: str,
    total_nodes: int,
) -> Array:
    """Accumulate real-basis M2L contributions in one full interaction batch."""
    idx = jnp.arange(src.shape[0], dtype=INDEX_DTYPE)
    raw_src = src
    raw_tgt = tgt
    valid = (idx < active_pair_count) & (raw_src >= 0) & (raw_tgt >= 0)
    safe_src = jnp.where(valid, raw_src, 0)
    safe_tgt = jnp.where(valid, raw_tgt, 0)
    src_mult = multip_packed_real[safe_src]
    deltas = centers[safe_tgt] - centers[safe_src]
    contribs = _apply_real_m2l(
        src_mult,
        deltas,
        order=order,
        m2l_impl=m2l_impl,
    ).astype(locals_coeffs.dtype)
    contribs = jnp.where(valid[:, None], contribs, 0)
    return locals_coeffs + jax.ops.segment_sum(contribs, safe_tgt, total_nodes)


@partial(
    jax.jit,
    static_argnames=("order", "m2l_impl", "total_nodes"),
    donate_argnums=(0,),
)
def _accumulate_real_m2l_fullbatch_pallas(
    locals_coeffs: Array,
    multip_packed_real: Array,
    centers: Array,
    src: Array,
    tgt: Array,
    *,
    order: int,
    m2l_impl: str,
    total_nodes: int,
) -> Array:
    """Accumulate real-basis M2L contributions using the optional Pallas core."""
    src_mult = multip_packed_real[src]
    deltas = centers[tgt] - centers[src]
    contribs = _m2l_real_batch_kernel_pallas(
        src_mult,
        deltas,
        order=order,
        m2l_impl=m2l_impl,
    ).astype(locals_coeffs.dtype)
    return locals_coeffs + jax.ops.segment_sum(contribs, tgt, total_nodes)


@partial(
    jax.jit,
    static_argnames=("order", "m2l_impl", "total_nodes", "chunk_size"),
    donate_argnums=(0,),
)
def _accumulate_real_m2l_chunked_scan(
    locals_coeffs: Array,
    multip_packed_real: Array,
    centers: Array,
    src: Array,
    tgt: Array,
    active_pair_count: Array,
    *,
    order: int,
    m2l_impl: str,
    total_nodes: int,
    chunk_size: int,
) -> Array:
    """Accumulate real-basis M2L contributions with chunked scan reduction."""
    pair_count = src.shape[0]
    starts = jnp.arange(0, pair_count, chunk_size, dtype=INDEX_DTYPE)

    def body(local_accum: Array, start_idx: Array) -> tuple[Array, None]:
        def active_chunk(accum: Array) -> Array:
            offset = jnp.arange(chunk_size, dtype=INDEX_DTYPE)
            idx = start_idx + offset
            valid = idx < pair_count
            safe_idx = jnp.where(valid, idx, 0)
            src_chunk_raw = src[safe_idx]
            tgt_chunk_raw = tgt[safe_idx]
            valid = (
                valid
                & (idx < active_pair_count)
                & (src_chunk_raw >= 0)
                & (tgt_chunk_raw >= 0)
            )
            src_chunk = jnp.where(valid, src_chunk_raw, 0)
            tgt_chunk = jnp.where(valid, tgt_chunk_raw, 0)
            src_mult = multip_packed_real[src_chunk]
            deltas = centers[tgt_chunk] - centers[src_chunk]
            contribs = _apply_real_m2l(
                src_mult,
                deltas,
                order=order,
                m2l_impl=m2l_impl,
            ).astype(locals_coeffs.dtype)
            return _chunk_segment_scatter_add(
                accum,
                contribs,
                tgt_chunk,
                valid,
                chunk_size=chunk_size,
            )

        local_accum = jax.lax.cond(
            start_idx < active_pair_count,
            active_chunk,
            lambda accum: accum,
            local_accum,
        )
        return local_accum, None

    local_accum, _ = jax.lax.scan(body, locals_coeffs, starts)
    return local_accum


@partial(
    jax.jit,
    static_argnames=("order", "m2l_impl", "total_nodes", "chunk_size"),
    donate_argnums=(0,),
)
def _accumulate_real_m2l_chunked_scan_pallas(
    locals_coeffs: Array,
    multip_packed_real: Array,
    centers: Array,
    src: Array,
    tgt: Array,
    *,
    order: int,
    m2l_impl: str,
    total_nodes: int,
    chunk_size: int,
) -> Array:
    """Chunked real-basis M2L accumulation using the optional Pallas core."""
    pair_count = src.shape[0]
    starts = jnp.arange(0, pair_count, chunk_size, dtype=INDEX_DTYPE)

    def body(local_accum: Array, start_idx: Array) -> tuple[Array, None]:
        offset = jnp.arange(chunk_size, dtype=INDEX_DTYPE)
        idx = start_idx + offset
        valid = idx < pair_count
        safe_idx = jnp.where(valid, idx, 0)
        src_chunk = src[safe_idx]
        tgt_chunk = tgt[safe_idx]
        src_mult = multip_packed_real[src_chunk]
        deltas = centers[tgt_chunk] - centers[src_chunk]
        contribs = _m2l_real_batch_kernel_pallas(
            src_mult,
            deltas,
            order=order,
            m2l_impl=m2l_impl,
        ).astype(locals_coeffs.dtype)
        local_accum = _chunk_segment_scatter_add(
            local_accum,
            contribs,
            tgt_chunk,
            valid,
            chunk_size=chunk_size,
        )
        return local_accum, None

    local_accum, _ = jax.lax.scan(body, locals_coeffs, starts)
    return local_accum


@partial(
    jax.jit,
    static_argnames=("order", "total_nodes"),
    donate_argnums=(0,),
)
def _propagate_real_locals_to_children(
    coeffs_local: Array,
    centers_local: Array,
    left_child: Array,
    right_child: Array,
    *,
    order: int,
    total_nodes: int,
) -> Array:
    """Apply real-basis L2L translations from parents to their children."""
    num_internal_nodes = left_child.shape[0]
    parent_idx = jnp.arange(num_internal_nodes, dtype=INDEX_DTYPE)
    child_idx = jnp.concatenate(
        [left_child[:num_internal_nodes], right_child[:num_internal_nodes]],
        axis=0,
    )
    parent_rep = jnp.concatenate([parent_idx, parent_idx], axis=0)
    valid = child_idx >= 0
    safe_child_idx = jnp.where(valid, child_idx, 0)
    parent_coeffs = coeffs_local[parent_rep]
    deltas = centers_local[safe_child_idx] - centers_local[parent_rep]
    translated = _l2l_real_batch_kernel(parent_coeffs, deltas, order=order)
    translated = translated.astype(coeffs_local.dtype)
    translated = jnp.where(valid[:, None], translated, 0)
    updates = jax.ops.segment_sum(translated, safe_child_idx, total_nodes)
    return coeffs_local + updates


def _prepare_solidfmm_downward_sweep(
    tree: Tree,
    upward: TreeUpwardData,
    *,
    theta: float,
    mac_type: MACType,
    initial_locals: Optional[LocalExpansionData] = None,
    interactions: Optional[NodeInteractionList] = None,
    m2l_chunk_size: Optional[int] = None,
    l2l_chunk_size: Optional[int] = None,
    complex_rotation: str = "solidfmm",
    basis_mode: str = "complex",
    m2l_impl: Optional[str] = None,
    traversal_config: Optional[DualTreeTraversalConfig] = None,
    dense_buffers: Optional[DenseInteractionBuffers] = None,
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
    grouped_interactions: bool = False,
    grouped_buffers: Optional[GroupedInteractionBuffers] = None,
    grouped_segment_starts: Optional[Array] = None,
    grouped_segment_lengths: Optional[Array] = None,
    grouped_segment_class_ids: Optional[Array] = None,
    grouped_segment_sort_permutation: Optional[Array] = None,
    grouped_segment_group_ids: Optional[Array] = None,
    grouped_segment_unique_targets: Optional[Array] = None,
    farfield_mode: str = "pair_grouped",
    far_pairs_coo: Optional[_FarPairCOO] = None,
    far_pairs_by_gear: Optional[tuple[tuple[Array, Array], ...]] = None,
    adaptive_order: bool = False,
    p_gears: tuple[int, ...] = tuple(),
    dehnen_radius_scale: float = 1.0,
    use_pallas: bool = False,
    timing_recorder: Optional[Callable[[str, float], None]] = None,
) -> TreeDownwardData:
    """Prepare M2L accumulation for solidfmm-style complex or real locals.

    The returned value intentionally retains only the locals plus a minimal
    interaction handle. Grouped layouts, chunk schedules, and other M2L feed
    structures are execution inputs, not part of the long-lived downward state.
    """

    interaction_inputs = _prepare_solidfmm_downward_interaction_inputs(
        tree=tree,
        upward=upward,
        theta=theta,
        mac_type=mac_type,
        interactions=interactions,
        far_pairs_coo=far_pairs_coo,
        traversal_config=traversal_config,
        retry_logger=retry_logger,
        dehnen_radius_scale=dehnen_radius_scale,
    )
    interactions = interaction_inputs.interactions
    src = interaction_inputs.src
    tgt = interaction_inputs.tgt
    pair_count = interaction_inputs.pair_count
    active_pair_count = interaction_inputs.active_pair_count

    def _record_timed_array(attr: str, start: float, value: Array) -> Array:
        if timing_recorder is None:
            return value
        value = jax.block_until_ready(value)
        timing_recorder(attr, float(time.perf_counter() - start))
        return value

    p = int(upward.multipoles.order)
    downward_init = _prepare_solidfmm_downward_init(
        upward=upward,
        initial_locals=initial_locals,
        basis_mode=basis_mode,
    )
    centers = downward_init.centers
    locals_coeffs = downward_init.locals_coeffs
    total_nodes = downward_init.total_nodes
    coeff_count = downward_init.coeff_count
    dtype = downward_init.dtype

    if pair_count == 0:
        empty_locals = LocalExpansionData(
            order=p,
            centers=centers,
            coefficients=locals_coeffs,
        )
        empty_source_motion_locals: Optional[LocalExpansionData]
        if upward.multipoles.source_motion_packed is not None:
            empty_source_motion_locals = LocalExpansionData(
                order=p,
                centers=centers,
                coefficients=jnp.zeros_like(locals_coeffs),
            )
        else:
            empty_source_motion_locals = None
        return TreeDownwardData(
            interactions=interactions,
            locals=empty_locals,
            source_motion_locals=empty_source_motion_locals,
        )

    detail_diag_mode = _normalize_strict_refresh_detail_diag_mode(
        os.environ.get("JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE", "full")
    )

    def _detail_downward_data(
        coefficients: Array,
        source_motion_coefficients: Optional[Array] = None,
    ) -> TreeDownwardData:
        source_motion_locals: Optional[LocalExpansionData]
        if source_motion_coefficients is not None:
            source_motion_locals = LocalExpansionData(
                order=p,
                centers=centers,
                coefficients=source_motion_coefficients,
            )
        else:
            source_motion_locals = None
        return TreeDownwardData(
            interactions=interactions,
            locals=LocalExpansionData(
                order=p,
                centers=centers,
                coefficients=coefficients,
            ),
            source_motion_locals=source_motion_locals,
        )

    if detail_diag_mode == "downward_artifacts_only":
        source_motion_zeros = (
            jnp.zeros_like(locals_coeffs)
            if upward.multipoles.source_motion_packed is not None
            else None
        )
        return _detail_downward_data(locals_coeffs, source_motion_zeros)

    rotation_mode = str(complex_rotation).strip().lower()
    resolved_m2l_impl = (
        "rot_scale" if m2l_impl is None else str(m2l_impl).strip().lower()
    )
    source_motion_multip_packed = None
    if detail_diag_mode == "l2l_only":
        locals_updated = jnp.ones_like(locals_coeffs)
        chunk_size = 4096 if m2l_chunk_size is None else int(m2l_chunk_size)
        if chunk_size <= 0:
            raise ValueError("m2l_chunk_size must be positive")
    else:
        multipole_inputs = _prepare_solidfmm_downward_multipole_inputs(
            upward=upward,
            dtype=dtype,
            basis_mode=basis_mode,
            complex_rotation=complex_rotation,
        )
        multip_packed = multipole_inputs.multip_packed
        source_motion_multip_packed = multipole_inputs.source_motion_multip_packed
        multip_packed_kernel = multipole_inputs.multip_packed_kernel
        rotation_mode = multipole_inputs.rotation_mode

        chunk_size = 4096 if m2l_chunk_size is None else int(m2l_chunk_size)
        if chunk_size <= 0:
            raise ValueError("m2l_chunk_size must be positive")

        stage_t0 = time.perf_counter()
        locals_updated = _solidfmm_downward_accumulate_from_multipoles(
            locals_coeffs,
            multip_packed_kernel,
            tree=tree,
            upward=upward,
            interactions=interactions,
            centers=centers,
            src=src,
            tgt=tgt,
            pair_count=pair_count,
            active_pair_count=active_pair_count,
            order=p,
            rotation_mode=rotation_mode,
            total_nodes=total_nodes,
            chunk_size=chunk_size,
            grouped_interactions=grouped_interactions,
            grouped_buffers=grouped_buffers,
            grouped_segment_starts=grouped_segment_starts,
            grouped_segment_lengths=grouped_segment_lengths,
            grouped_segment_class_ids=grouped_segment_class_ids,
            grouped_segment_sort_permutation=grouped_segment_sort_permutation,
            grouped_segment_group_ids=grouped_segment_group_ids,
            grouped_segment_unique_targets=grouped_segment_unique_targets,
            farfield_mode=farfield_mode,
            basis_mode=basis_mode,
            m2l_impl=resolved_m2l_impl,
        )
        locals_updated = _record_timed_array(
            "_refresh_timing_dual_m2l_compute_seconds",
            stage_t0,
            locals_updated,
        )
        if detail_diag_mode == "m2l_only":
            return _detail_downward_data(locals_updated)

    if l2l_chunk_size is not None and int(l2l_chunk_size) <= 0:
        raise ValueError("l2l_chunk_size must be positive")

    child_inputs = _prepare_solidfmm_downward_child_inputs(tree)
    if child_inputs.num_internal_nodes > 0:
        left_child = child_inputs.left_child
        right_child = child_inputs.right_child
        node_levels = get_node_levels(tree)
        stage_t0 = time.perf_counter()
        locals_updated = _propagate_solidfmm_locals_by_level(
            locals_updated,
            centers,
            left_child,
            right_child,
            node_levels,
            order=p,
            rotation=rotation_mode,
            total_nodes=total_nodes,
            basis_mode=basis_mode,
        )
        locals_updated = _record_timed_array(
            "_refresh_timing_dual_l2l_compute_seconds",
            stage_t0,
            locals_updated,
        )
        source_motion_locals_updated: Optional[Array]
        if source_motion_multip_packed is not None:
            stage_t0 = time.perf_counter()
            source_motion_locals_updated = (
                _solidfmm_downward_accumulate_from_multipoles(
                    jnp.zeros_like(locals_coeffs),
                    source_motion_multip_packed,
                    tree=tree,
                    upward=upward,
                    interactions=interactions,
                    centers=centers,
                    src=src,
                    tgt=tgt,
                    pair_count=pair_count,
                    active_pair_count=active_pair_count,
                    order=p,
                    rotation_mode=rotation_mode,
                    total_nodes=total_nodes,
                    chunk_size=chunk_size,
                    grouped_interactions=grouped_interactions,
                    grouped_buffers=grouped_buffers,
                    grouped_segment_starts=grouped_segment_starts,
                    grouped_segment_lengths=grouped_segment_lengths,
                    grouped_segment_class_ids=grouped_segment_class_ids,
                    grouped_segment_sort_permutation=grouped_segment_sort_permutation,
                    grouped_segment_group_ids=grouped_segment_group_ids,
                    grouped_segment_unique_targets=grouped_segment_unique_targets,
                    farfield_mode=farfield_mode,
                    basis_mode=basis_mode,
                    m2l_impl=resolved_m2l_impl,
                )
            )
            source_motion_locals_updated = _propagate_solidfmm_locals_by_level(
                source_motion_locals_updated,
                centers,
                left_child,
                right_child,
                node_levels,
                order=p,
                rotation=rotation_mode,
                total_nodes=total_nodes,
                basis_mode=basis_mode,
            )
            source_motion_locals_updated = _record_timed_array(
                "_refresh_timing_dual_source_motion_seconds",
                stage_t0,
                source_motion_locals_updated,
            )
        else:
            source_motion_locals_updated = None
    else:
        if source_motion_multip_packed is not None:
            stage_t0 = time.perf_counter()
            source_motion_locals_updated = _accumulate_from_multipoles(
                jnp.zeros_like(locals_coeffs), source_motion_multip_packed
            )
            source_motion_locals_updated = _record_timed_array(
                "_refresh_timing_dual_source_motion_seconds",
                stage_t0,
                source_motion_locals_updated,
            )
        else:
            source_motion_locals_updated = None

    stage_t0 = time.perf_counter()
    locals_after = LocalExpansionData(
        order=p,
        centers=centers,
        coefficients=locals_updated,
    )

    # Conjugate symmetry is a property of the COMPLEX solidfmm coefficients
    # only. Real (Dehnen no-sqrt2) locals are not conjugate-symmetric, so
    # applying it there corrupts them (it silently caps far-field accuracy).
    real_basis = str(basis_mode).strip().lower() == "real"
    if not real_basis:
        coefficients_after = enforce_conjugate_symmetry_batch(
            jnp.asarray(locals_after.coefficients),
            order=p,
        )
        coefficients_after = _record_timed_array(
            "_refresh_timing_dual_final_symmetry_seconds",
            stage_t0,
            coefficients_after,
        )
        locals_after = locals_after._replace(coefficients=coefficients_after)
    source_motion_locals_after: Optional[LocalExpansionData]
    if source_motion_locals_updated is not None:
        source_motion_coefficients = (
            jnp.asarray(source_motion_locals_updated)
            if real_basis
            else enforce_conjugate_symmetry_batch(
                jnp.asarray(source_motion_locals_updated),
                order=p,
            )
        )
        source_motion_locals_after = LocalExpansionData(
            order=p,
            centers=centers,
            coefficients=source_motion_coefficients,
        )
    else:
        source_motion_locals_after = None

    return TreeDownwardData(
        interactions=interactions,
        locals=locals_after,
        source_motion_locals=source_motion_locals_after,
    )


@partial(
    jax.jit,
    static_argnames=(
        "return_potential",
        "max_leaf_size",
        "order",
        "G",
        "softening",
        "expansion_basis",
        "nearfield_mode",
        "nearfield_edge_chunk_size",
        "nearfield_delayed_scatter_chunks_per_superchunk",
        "nearfield_chunk_scan_batch_size",
        "nearfield_chunk_scan_unroll",
        "nearfield_superchunk_scan_unroll",
        "nearfield_sorted_scatter_hint",
        "nearfield_grouped_sorted_scatter",
        "nearfield_superchunk_target_reduce",
        "nearfield_disable_chunk_cond",
        "nearfield_target_leaf_batch_size",
        "nearfield_target_block_tile_size",
        "nearfield_target_block_tile_scan_unroll",
        "nearfield_target_block_batch_scan_unroll",
        "nearfield_target_block_overflow_fast_max_blocks",
        "disable_specialized_large_n_nearfield",
    ),
)
def _evaluate_tree_compiled_impl(
    tree: Tree,
    positions: Array,
    masses: Array,
    locals_data: LocalExpansionData,
    neighbor_list: NodeNeighborList,
    nearfield_leaf_nodes: Array,
    nearfield_node_ranges: Array,
    nearfield_offsets: Array,
    nearfield_neighbors: Array,
    nearfield_counts: Array,
    nearfield_leaf_particle_indices: Array,
    nearfield_leaf_particle_mask: Array,
    leaf_nodes: Array,
    node_ranges: Array,
    precomputed_target_leaf_ids: Array,
    precomputed_source_leaf_ids: Array,
    precomputed_valid_pairs: Array,
    precomputed_chunk_sort_indices: Array,
    precomputed_chunk_group_ids: Array,
    precomputed_chunk_unique_indices: Array,
    precomputed_target_block_offsets: Array,
    precomputed_target_block_leaf_ids: Array,
    precomputed_target_block_source_leaf_ids: Array,
    precomputed_target_block_valid_mask: Array,
    precomputed_target_block_source_leaf_ids_padded: Array,
    precomputed_target_block_valid_mask_padded: Array,
    *,
    G: float,
    softening: float,
    order: int,
    expansion_basis: ExpansionBasis,
    max_leaf_size: int,
    return_potential: bool,
    nearfield_mode: str,
    nearfield_edge_chunk_size: int,
    nearfield_delayed_scatter_chunks_per_superchunk: int = 1,
    nearfield_chunk_scan_batch_size: int = 1,
    nearfield_chunk_scan_unroll: int = 1,
    nearfield_superchunk_scan_unroll: int = 1,
    nearfield_sorted_scatter_hint: bool = False,
    nearfield_grouped_sorted_scatter: bool = False,
    nearfield_superchunk_target_reduce: bool = False,
    nearfield_disable_chunk_cond: bool = True,
    nearfield_target_leaf_batch_size: int = 32,
    nearfield_target_block_tile_size: int = 8,
    nearfield_target_block_tile_scan_unroll: int = 1,
    nearfield_target_block_batch_scan_unroll: int = 1,
    nearfield_target_block_overflow_fast_max_blocks: int = 65536,
    disable_specialized_large_n_nearfield: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """JIT core for far/near field evaluation on a prepared tree state."""
    disable_specialized_large_n = bool(disable_specialized_large_n_nearfield)
    use_precomputed = (
        precomputed_target_leaf_ids.shape[0] == neighbor_list.neighbors.shape[0]
        and precomputed_valid_pairs.shape[0] == neighbor_list.neighbors.shape[0]
    )
    use_precomputed_source = (
        precomputed_source_leaf_ids.shape[0] == neighbor_list.neighbors.shape[0]
    )
    edge_count = int(neighbor_list.neighbors.shape[0])
    chunk_count = (
        (edge_count + int(nearfield_edge_chunk_size) - 1)
        // int(nearfield_edge_chunk_size)
        if edge_count > 0
        else 0
    )
    chunk_flat_size = int(nearfield_edge_chunk_size) * int(max_leaf_size)
    use_precomputed_scatter = (
        precomputed_chunk_sort_indices.shape == (chunk_count, chunk_flat_size)
        and precomputed_chunk_group_ids.shape == (chunk_count, chunk_flat_size)
        and precomputed_chunk_unique_indices.shape == (chunk_count, chunk_flat_size)
    )
    use_specialized_large_n = (
        not disable_specialized_large_n
        and not bool(return_potential)
        and str(nearfield_mode).strip().lower() == "bucketed"
        and nearfield_leaf_particle_indices.shape[0] > 0
        and not use_precomputed_scatter
    )
    use_target_blocks = (
        precomputed_target_block_offsets.shape[0]
        == (neighbor_list.leaf_indices.shape[0] + 1)
        and precomputed_target_block_leaf_ids.shape[0] > 0
        and precomputed_target_block_source_leaf_ids.shape[0]
        == precomputed_target_block_leaf_ids.shape[0]
        and precomputed_target_block_valid_mask.shape
        == precomputed_target_block_source_leaf_ids.shape
    )
    use_target_blocks_padded = (
        precomputed_target_block_source_leaf_ids_padded.shape[0]
        == neighbor_list.leaf_indices.shape[0]
        and precomputed_target_block_source_leaf_ids_padded.shape[1] > 0
        and precomputed_target_block_source_leaf_ids_padded.shape[2] > 0
        and precomputed_target_block_valid_mask_padded.shape
        == precomputed_target_block_source_leaf_ids_padded.shape
    )
    if use_specialized_large_n:
        near = compute_leaf_p2p_accelerations_large_n_accel_only(
            tree,
            neighbor_list,
            positions,
            masses,
            G=G,
            softening=softening,
            edge_chunk_size=nearfield_edge_chunk_size,
            precomputed_target_leaf_ids=(
                precomputed_target_leaf_ids if use_precomputed else None
            ),
            precomputed_source_leaf_ids=(
                precomputed_source_leaf_ids
                if (use_precomputed and use_precomputed_source)
                else None
            ),
            precomputed_valid_pairs=(
                precomputed_valid_pairs if use_precomputed else None
            ),
            leaf_particle_indices=nearfield_leaf_particle_indices,
            leaf_particle_mask=nearfield_leaf_particle_mask,
            precomputed_target_block_leaf_ids=(
                precomputed_target_block_leaf_ids if use_target_blocks else None
            ),
            precomputed_target_block_source_leaf_ids=(
                precomputed_target_block_source_leaf_ids if use_target_blocks else None
            ),
            precomputed_target_block_valid_mask=(
                precomputed_target_block_valid_mask if use_target_blocks else None
            ),
            precomputed_target_block_offsets=(
                precomputed_target_block_offsets if use_target_blocks else None
            ),
            precomputed_target_block_source_leaf_ids_padded=(
                precomputed_target_block_source_leaf_ids_padded
                if use_target_blocks_padded
                else None
            ),
            precomputed_target_block_valid_mask_padded=(
                precomputed_target_block_valid_mask_padded
                if use_target_blocks_padded
                else None
            ),
            delayed_scatter_chunks_per_superchunk=(
                nearfield_delayed_scatter_chunks_per_superchunk
            ),
            chunk_scan_batch_size=nearfield_chunk_scan_batch_size,
            chunk_scan_unroll=nearfield_chunk_scan_unroll,
            superchunk_scan_unroll=nearfield_superchunk_scan_unroll,
            sorted_scatter_hint=nearfield_sorted_scatter_hint,
            grouped_sorted_scatter=nearfield_grouped_sorted_scatter,
            superchunk_target_reduce=nearfield_superchunk_target_reduce,
            disable_chunk_cond=nearfield_disable_chunk_cond,
            target_leaf_batch_size=nearfield_target_leaf_batch_size,
            target_block_tile_size=nearfield_target_block_tile_size,
            target_block_tile_scan_unroll=nearfield_target_block_tile_scan_unroll,
            target_block_batch_scan_unroll=nearfield_target_block_batch_scan_unroll,
            target_block_overflow_fast_max_blocks=(
                nearfield_target_block_overflow_fast_max_blocks
            ),
        )
    else:
        near = compute_leaf_p2p_accelerations(
            tree,
            neighbor_list,
            positions,
            masses,
            G=G,
            softening=softening,
            max_leaf_size=max_leaf_size,
            return_potential=return_potential,
            nearfield_mode=nearfield_mode,
            edge_chunk_size=nearfield_edge_chunk_size,
            precomputed_target_leaf_ids=(
                precomputed_target_leaf_ids if use_precomputed else None
            ),
            precomputed_source_leaf_ids=(
                precomputed_source_leaf_ids
                if (use_precomputed and use_precomputed_source)
                else None
            ),
            precomputed_valid_pairs=(
                precomputed_valid_pairs if use_precomputed else None
            ),
            precomputed_chunk_sort_indices=(
                precomputed_chunk_sort_indices if use_precomputed_scatter else None
            ),
            precomputed_chunk_group_ids=(
                precomputed_chunk_group_ids if use_precomputed_scatter else None
            ),
            precomputed_chunk_unique_indices=(
                precomputed_chunk_unique_indices if use_precomputed_scatter else None
            ),
            node_ranges_override=nearfield_node_ranges,
            leaf_nodes_override=nearfield_leaf_nodes,
            neighbor_offsets_override=nearfield_offsets,
            neighbor_indices_override=nearfield_neighbors,
            neighbor_counts_override=nearfield_counts,
            leaf_particle_indices_override=(
                nearfield_leaf_particle_indices
                if nearfield_leaf_particle_indices.shape[0] > 0
                else None
            ),
            leaf_particle_mask_override=(
                nearfield_leaf_particle_mask
                if nearfield_leaf_particle_mask.shape[0] > 0
                else None
            ),
        )

    far_grad, far_potential_pre, _ = _evaluate_local_expansions_for_particles(
        locals_data,
        positions,
        leaf_nodes=leaf_nodes,
        node_ranges=node_ranges,
        max_leaf_size=max_leaf_size,
        order=order,
        expansion_basis=expansion_basis,
        return_potential=return_potential,
        max_acc_derivative_order=0,
    )

    # far_grad is d/d(delta) of +1/r with delta = center - eval_point.
    # Physical acceleration is d/d(eval_point)(+1/r) * G = -d/d(delta)(+1/r) * G.
    far_acc = -G * far_grad

    if return_potential:
        near_acc, near_pot = near  # type: ignore[misc]
        far_pot = (
            -G * far_potential_pre
            if far_potential_pre is not None
            else jnp.zeros((positions.shape[0],), dtype=positions.dtype)
        )
        accelerations = near_acc + far_acc
        potentials = near_pot + far_pot
        return accelerations, potentials

    accelerations = near + far_acc  # type: ignore[operator]
    return accelerations


def _evaluate_prepared_tree(
    *,
    fmm: "FastMultipoleMethod",
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    downward: TreeDownwardData,
    neighbor_list: NodeNeighborList,
    nearfield_interop: Optional[NearfieldInteropData],
    farfield_local_data: Optional[LocalExpansionData],
    farfield_leaf_nodes: Optional[Array],
    farfield_node_ranges: Optional[Array],
    nearfield_target_leaf_ids: Optional[Array],
    nearfield_source_leaf_ids: Optional[Array],
    nearfield_valid_pairs: Optional[Array],
    nearfield_chunk_sort_indices: Optional[Array],
    nearfield_chunk_group_ids: Optional[Array],
    nearfield_chunk_unique_indices: Optional[Array],
    max_leaf_size: int,
    return_potential: bool,
    jit_traversal: bool,
    max_acc_derivative_order: int = 0,
) -> Union[
    Array,
    Tuple[Array, Array],
    Tuple[Array, PackedAccelerationDerivatives],
    Tuple[Array, Array, PackedAccelerationDerivatives],
]:
    """Run the prepared-tree evaluation returning Morton-sorted outputs."""

    if int(max_acc_derivative_order) > 0:
        nearfield_mode = fmm._resolve_nearfield_mode(
            num_particles=int(positions_sorted.shape[0])
        )
        nearfield_edge_chunk_size = fmm._resolve_nearfield_edge_chunk_size(
            num_particles=int(positions_sorted.shape[0]),
            nearfield_mode=nearfield_mode,
        )
        near = compute_leaf_p2p_accelerations(
            tree,
            neighbor_list,
            positions_sorted,
            masses_sorted,
            G=fmm.G,
            softening=fmm.softening,
            max_leaf_size=max_leaf_size,
            return_potential=return_potential,
            nearfield_mode=nearfield_mode,
            edge_chunk_size=nearfield_edge_chunk_size,
            precomputed_target_leaf_ids=nearfield_target_leaf_ids,
            precomputed_source_leaf_ids=nearfield_source_leaf_ids,
            precomputed_valid_pairs=nearfield_valid_pairs,
            precomputed_chunk_sort_indices=nearfield_chunk_sort_indices,
            precomputed_chunk_group_ids=nearfield_chunk_group_ids,
            precomputed_chunk_unique_indices=nearfield_chunk_unique_indices,
        )
        far_grad, far_potential_pre, far_derivatives = (
            _evaluate_local_expansions_for_particles(
                downward.locals,
                positions_sorted,
                leaf_nodes=jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE),
                node_ranges=jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
                max_leaf_size=max_leaf_size,
                order=int(downward.locals.order),
                expansion_basis=fmm.expansion_basis,
                return_potential=return_potential,
                max_acc_derivative_order=int(max_acc_derivative_order),
            )
        )
        far_acc = -fmm.G * far_grad
        if far_derivatives is None:
            raise RuntimeError("expected far-field acceleration derivatives")
        acc_derivatives = tuple(fmm.G * level for level in far_derivatives)

        if return_potential:
            near_acc, near_pot = near
            far_pot = (
                -fmm.G * far_potential_pre
                if far_potential_pre is not None
                else jnp.zeros(
                    (positions_sorted.shape[0],), dtype=positions_sorted.dtype
                )
            )
            return near_acc + far_acc, near_pot + far_pot, acc_derivatives
        return near + far_acc, acc_derivatives

    if jit_traversal:
        evaluate_fn = fmm.evaluate_tree_compiled
    else:
        evaluate_fn = fmm.evaluate_tree

    return evaluate_fn(
        tree,
        positions_sorted,
        masses_sorted,
        downward,
        neighbor_list,
        nearfield_interop=nearfield_interop,
        farfield_local_data=farfield_local_data,
        farfield_leaf_nodes=farfield_leaf_nodes,
        farfield_node_ranges=farfield_node_ranges,
        precomputed_target_leaf_ids=nearfield_target_leaf_ids,
        precomputed_source_leaf_ids=nearfield_source_leaf_ids,
        precomputed_valid_pairs=nearfield_valid_pairs,
        precomputed_chunk_sort_indices=nearfield_chunk_sort_indices,
        precomputed_chunk_group_ids=nearfield_chunk_group_ids,
        precomputed_chunk_unique_indices=nearfield_chunk_unique_indices,
        max_leaf_size=max_leaf_size,
        return_potential=return_potential,
    )


def _map_targets_to_leaf_positions(
    *,
    target_sorted_indices: Array,
    leaf_nodes: Array,
    node_ranges: Array,
) -> Array:
    """Map sorted particle indices to positions in the leaf-node array."""
    if int(target_sorted_indices.shape[0]) == 0:
        return jnp.zeros((0,), dtype=INDEX_DTYPE)
    leaf_ranges = node_ranges[leaf_nodes]
    starts = leaf_ranges[:, 0]
    ends = leaf_ranges[:, 1]
    leaf_pos = jnp.searchsorted(starts, target_sorted_indices, side="right") - 1
    leaf_pos = jnp.clip(leaf_pos, 0, leaf_nodes.shape[0] - 1)
    valid = (target_sorted_indices >= starts[leaf_pos]) & (
        target_sorted_indices <= ends[leaf_pos]
    )
    if not bool(jnp.all(valid)):
        raise ValueError("target_indices could not be mapped to prepared tree leaves")
    return leaf_pos.astype(INDEX_DTYPE)


def _build_target_nearfield_source_index_matrix(
    *,
    target_sorted_indices: Array,
    target_leaf_positions: Array,
    nearfield_interop: NearfieldInteropData,
) -> tuple[Array, Array]:
    """Build padded source-index lists for each target particle near-field eval."""
    targets = jnp.asarray(target_sorted_indices, dtype=INDEX_DTYPE)
    target_leaf_pos = jnp.asarray(target_leaf_positions, dtype=INDEX_DTYPE)
    node_ranges = jnp.asarray(nearfield_interop.node_ranges, dtype=INDEX_DTYPE)
    leaf_nodes = jnp.asarray(nearfield_interop.leaf_nodes, dtype=INDEX_DTYPE)
    offsets = jnp.asarray(nearfield_interop.offsets, dtype=INDEX_DTYPE)
    neighbors = jnp.asarray(nearfield_interop.neighbors, dtype=INDEX_DTYPE)

    num_targets = int(targets.shape[0])
    if num_targets == 0:
        empty_idx = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        empty_mask = jnp.zeros((0, 0), dtype=bool)
        return empty_idx, empty_mask

    num_leaves = int(leaf_nodes.shape[0])
    if num_leaves == 0:
        empty_idx = jnp.zeros((num_targets, 0), dtype=INDEX_DTYPE)
        empty_mask = jnp.zeros((num_targets, 0), dtype=bool)
        return empty_idx, empty_mask

    if (
        nearfield_interop.leaf_particle_indices is not None
        and nearfield_interop.leaf_particle_mask is not None
    ):
        leaf_particle_idx = jnp.asarray(
            nearfield_interop.leaf_particle_indices,
            dtype=INDEX_DTYPE,
        )
        leaf_particle_mask = jnp.asarray(
            nearfield_interop.leaf_particle_mask,
            dtype=bool,
        )
        max_leaf_particles = int(leaf_particle_idx.shape[1])
    else:
        leaf_ranges = node_ranges[leaf_nodes]
        leaf_counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
        max_leaf_particles = int(jnp.max(leaf_counts))
        if max_leaf_particles <= 0:
            empty_idx = jnp.zeros((num_targets, 0), dtype=INDEX_DTYPE)
            empty_mask = jnp.zeros((num_targets, 0), dtype=bool)
            return empty_idx, empty_mask

        particle_offsets = jnp.arange(max_leaf_particles, dtype=INDEX_DTYPE)
        leaf_particle_idx = leaf_ranges[:, 0][:, None] + particle_offsets[None, :]
        leaf_particle_mask = particle_offsets[None, :] < leaf_counts[:, None]

    if max_leaf_particles <= 0:
        empty_idx = jnp.zeros((num_targets, 0), dtype=INDEX_DTYPE)
        empty_mask = jnp.zeros((num_targets, 0), dtype=bool)
        return empty_idx, empty_mask

    if nearfield_interop.neighbor_leaf_positions is not None:
        nbr_leaf_pos = jnp.asarray(
            nearfield_interop.neighbor_leaf_positions,
            dtype=INDEX_DTYPE,
        )
    else:
        total_nodes = int(node_ranges.shape[0])
        leaf_lookup = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
        leaf_lookup = leaf_lookup.at[leaf_nodes].set(
            jnp.arange(num_leaves, dtype=INDEX_DTYPE)
        )
        nbr_counts = offsets[1:] - offsets[:-1]
        max_nbr = int(jnp.max(nbr_counts))
        if max_nbr > 0:
            nbr_offsets = jnp.arange(max_nbr, dtype=INDEX_DTYPE)
            nbr_idx = offsets[:-1, None] + nbr_offsets[None, :]
            nbr_valid = nbr_offsets[None, :] < nbr_counts[:, None]
            nbr_safe_idx = jnp.where(nbr_valid, nbr_idx, 0)
            nbr_nodes = neighbors[nbr_safe_idx]
            nbr_leaf_pos = leaf_lookup[nbr_nodes]
            nbr_leaf_pos = jnp.where(nbr_valid, nbr_leaf_pos, -1)
        else:
            nbr_leaf_pos = jnp.zeros((num_leaves, 0), dtype=INDEX_DTYPE)

    self_leaf = jnp.arange(num_leaves, dtype=INDEX_DTYPE)[:, None]
    source_leaf_positions = jnp.concatenate([self_leaf, nbr_leaf_pos], axis=1)
    source_leaf_valid = source_leaf_positions >= 0
    source_leaf_safe = jnp.where(source_leaf_valid, source_leaf_positions, 0)

    source_particle_idx_by_leaf = leaf_particle_idx[source_leaf_safe]
    source_particle_mask_by_leaf = (
        leaf_particle_mask[source_leaf_safe] & source_leaf_valid[..., None]
    )

    target_source_idx = source_particle_idx_by_leaf[target_leaf_pos]
    target_source_mask = source_particle_mask_by_leaf[target_leaf_pos]
    target_source_idx = target_source_idx.reshape((num_targets, -1))
    target_source_mask = target_source_mask.reshape((num_targets, -1))
    target_source_mask = target_source_mask & (target_source_idx != targets[:, None])

    sentinel = jnp.asarray(jnp.iinfo(INDEX_DTYPE).max, dtype=INDEX_DTYPE)
    sortable = jnp.where(target_source_mask, target_source_idx, sentinel)
    sorted_idx = jnp.sort(sortable, axis=1)
    non_sentinel = sorted_idx < sentinel
    first = jnp.ones((num_targets, 1), dtype=bool)
    changed = jnp.concatenate([first, sorted_idx[:, 1:] != sorted_idx[:, :-1]], axis=1)
    unique_mask = non_sentinel & changed
    padded = jnp.where(unique_mask, sorted_idx, 0)
    return padded, unique_mask


def _compute_targeted_nearfield(
    *,
    positions_sorted: Array,
    masses_sorted: Array,
    target_sorted_indices: Array,
    source_indices: Array,
    source_mask: Array,
    G: Union[float, Array],
    softening: float,
    return_potential: bool,
    velocities_sorted: Optional[Array] = None,
    return_jerk: bool = False,
    return_snap: bool = False,
    return_crackle: bool = False,
) -> tuple[Array, Optional[Array], Optional[Array], Optional[Array], Optional[Array]]:
    """Compute near-field contributions for target particles only."""
    if return_jerk and velocities_sorted is None:
        raise ValueError("velocities_sorted must be provided when return_jerk=True")
    if return_snap and velocities_sorted is None:
        raise ValueError("velocities_sorted must be provided when return_snap=True")
    if return_crackle and velocities_sorted is None:
        raise ValueError("velocities_sorted must be provided when return_crackle=True")
    target_positions = positions_sorted[target_sorted_indices]
    dtype = positions_sorted.dtype
    g_const = jnp.asarray(G, dtype=dtype)
    softening_sq = jnp.asarray(float(softening) ** 2, dtype=dtype)
    target_velocities = (
        velocities_sorted[target_sorted_indices]
        if velocities_sorted is not None
        else None
    )
    if int(source_indices.shape[1]) == 0:
        zeros = jnp.zeros((target_positions.shape[0], 3), dtype=positions_sorted.dtype)
        jerk_zeros = (
            jnp.zeros((target_positions.shape[0], 3), dtype=positions_sorted.dtype)
            if return_jerk
            else None
        )
        snap_zeros = (
            jnp.zeros((target_positions.shape[0], 3), dtype=positions_sorted.dtype)
            if return_snap
            else None
        )
        crackle_zeros = (
            jnp.zeros((target_positions.shape[0], 3), dtype=positions_sorted.dtype)
            if return_crackle
            else None
        )
        if return_potential:
            return (
                zeros,
                jnp.zeros((target_positions.shape[0],), dtype=zeros.dtype),
                jerk_zeros,
                snap_zeros,
                crackle_zeros,
            )
        return zeros, None, jerk_zeros, snap_zeros, crackle_zeros
    src_pos = positions_sorted[source_indices]
    src_mass = masses_sorted[source_indices]
    diff = target_positions[:, None, :] - src_pos
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
    eps = jnp.finfo(positions_sorted.dtype).eps
    one = jnp.asarray(1.0, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    three = jnp.asarray(3.0, dtype=dtype)
    six = jnp.asarray(6.0, dtype=dtype)
    nine = jnp.asarray(9.0, dtype=dtype)
    fifteen = jnp.asarray(15.0, dtype=dtype)
    forty_five = jnp.asarray(45.0, dtype=dtype)
    one_oh_five = jnp.asarray(105.0, dtype=dtype)

    inv_r = jnp.where(source_mask, one / (jnp.sqrt(dist_sq) + eps), zero)
    inv_dist3 = jnp.where(source_mask, inv_r / dist_sq, zero)
    weighted = inv_dist3 * src_mass
    near_acc = -g_const * jnp.sum(weighted[..., None] * diff, axis=1)
    near_jerk: Optional[Array]
    near_snap: Optional[Array]
    near_crackle: Optional[Array]
    if return_jerk:
        src_vel = velocities_sorted[source_indices]  # type: ignore[index]
        vel_diff = target_velocities[:, None, :] - src_vel  # type: ignore[index]
        inv_dist5 = jnp.where(source_mask, inv_dist3 / dist_sq, zero)
        rv = jnp.sum(diff * vel_diff, axis=-1)
        jerk_term = vel_diff * inv_dist3[..., None] - (
            three * rv[..., None] * diff * inv_dist5[..., None]
        )
        near_jerk = -g_const * jnp.sum(src_mass[..., None] * jerk_term, axis=1)
        if return_snap:
            inv_dist7 = jnp.where(source_mask, inv_dist5 / dist_sq, zero)
            vv = jnp.sum(vel_diff * vel_diff, axis=-1)
            snap_term = (
                six * rv[..., None] * vel_diff * inv_dist5[..., None]
                + three * vv[..., None] * diff * inv_dist5[..., None]
                - fifteen * (rv * rv)[..., None] * diff * inv_dist7[..., None]
            )
            near_snap = jnp.sum(src_mass[..., None] * snap_term, axis=1) * g_const
            if return_crackle:
                inv_dist9 = jnp.where(source_mask, inv_dist7 / dist_sq, zero)
                crackle_term = (
                    nine * vv[..., None] * vel_diff * inv_dist5[..., None]
                    - forty_five
                    * (rv * rv)[..., None]
                    * vel_diff
                    * inv_dist7[..., None]
                    - forty_five
                    * rv[..., None]
                    * vv[..., None]
                    * diff
                    * inv_dist7[..., None]
                    + one_oh_five
                    * (rv * rv * rv)[..., None]
                    * diff
                    * inv_dist9[..., None]
                )
                near_crackle = jnp.sum(src_mass[..., None] * crackle_term, axis=1) * (
                    g_const
                )
            else:
                near_crackle = None
        else:
            near_snap = None
            near_crackle = None
    else:
        near_jerk = None
        near_snap = None
        near_crackle = None
    if not return_potential:
        return near_acc, None, near_jerk, near_snap, near_crackle
    near_pot = -g_const * jnp.sum(inv_r * src_mass, axis=1)
    return near_acc, near_pot, near_jerk, near_snap, near_crackle


def _evaluate_local_expansions_for_target_particles(
    *,
    local_data: LocalExpansionData,
    positions_sorted: Array,
    target_sorted_indices: Array,
    target_leaf_positions: Array,
    leaf_nodes: Array,
    order: int,
    expansion_basis: ExpansionBasis,
    return_potential: bool,
    max_acc_derivative_order: int = 0,
) -> tuple[Array, Optional[Array], Optional[PackedAccelerationDerivatives]]:
    """Evaluate far-field local expansions for target particles only."""
    if order > MAX_MULTIPOLE_ORDER and expansion_basis != "solidfmm":
        raise NotImplementedError(
            "orders above 4 require expansion_basis='solidfmm'",
        )
    if int(target_sorted_indices.shape[0]) == 0:
        zeros = jnp.zeros((0, 3), dtype=positions_sorted.dtype)
        derivatives: Optional[PackedAccelerationDerivatives]
        if max_acc_derivative_order > 0:
            derivatives = tuple(
                jnp.zeros(
                    (
                        0,
                        3,
                        len(component_lift_index_map_3d(level)),
                    ),
                    dtype=positions_sorted.dtype,
                )
                for level in range(1, max_acc_derivative_order + 1)
            )
        else:
            derivatives = None
        if return_potential:
            return zeros, jnp.zeros((0,), dtype=positions_sorted.dtype), derivatives
        return zeros, None, derivatives

    target_leaf_nodes = leaf_nodes[target_leaf_positions]
    centers = local_data.centers[target_leaf_nodes]
    coeffs = local_data.coefficients[target_leaf_nodes]
    target_positions = positions_sorted[target_sorted_indices]

    if expansion_basis == "solidfmm":
        offsets_solid = centers - target_positions
        offsets_complex = offsets_solid

        # Real (Dehnen no-sqrt2) basis: real-typed locals, evaluated with the
        # real L2P operator (detected by coefficient dtype).
        if not jnp.iscomplexobj(coeffs):
            if int(max_acc_derivative_order) <= 0:
                grads, pots = jax.vmap(
                    lambda coeff_row, offset_row: evaluate_local_real_with_grad(
                        coeff_row, offset_row, order=int(order)
                    )
                )(coeffs, offsets_complex)
                if return_potential:
                    return grads, pots, None
                return grads, None, None

            tower = jax.vmap(
                lambda coeff_row, offset_row: (
                    evaluate_local_real_derivative_tower_batch(
                        coeff_row,
                        offset_row[jnp.newaxis, :],
                        order=int(order),
                        max_derivative_order=int(max_acc_derivative_order) + 1,
                    )
                ),
                in_axes=(0, 0),
            )(coeffs, offsets_complex)
            potentials = tower[0][:, 0, 0]
            gradients = tower[1][:, 0, :]
            derivatives_real: list[Array] = []
            for level in range(1, max_acc_derivative_order + 1):
                high = tower[level + 1][:, 0, :]
                gather = jnp.asarray(
                    component_lift_index_map_3d(level),
                    dtype=INDEX_DTYPE,
                )
                lifted = jnp.swapaxes(high[:, gather], 1, 2)
                sign = -1.0 if level % 2 == 0 else 1.0
                derivatives_real.append(sign * lifted)
            packed_real: PackedAccelerationDerivatives = tuple(derivatives_real)
            if return_potential:
                return gradients, potentials, packed_real
            return gradients, None, packed_real

        if max_acc_derivative_order <= 0:
            if return_potential:

                def eval_one(
                    coeff_row: Array, offset_row: Array
                ) -> tuple[Array, Array]:
                    grad, pot = evaluate_local_complex_with_grad_analytic(
                        coeff_row,
                        offset_row,
                        order=int(order),
                    )
                    return grad, pot

                gradients, potentials = jax.vmap(eval_one)(coeffs, offsets_complex)
                return gradients, potentials, None

            gradients = jax.vmap(
                lambda coeff_row, offset_row: evaluate_local_complex_grad_analytic(
                    coeff_row,
                    offset_row,
                    order=int(order),
                )
            )(coeffs, offsets_complex)
            return gradients, None, None

        tower = jax.vmap(
            lambda coeff_row, offset_row: evaluate_local_complex_derivative_tower_batch(
                coeff_row,
                offset_row[jnp.newaxis, :],
                order=int(order),
                max_derivative_order=int(max_acc_derivative_order) + 1,
            ),
            in_axes=(0, 0),
        )(coeffs, offsets_complex)

        potentials = tower[0][:, 0, 0]
        gradients = tower[1][:, 0, :]
        derivatives: list[Array] = []
        for level in range(1, max_acc_derivative_order + 1):
            high = tower[level + 1][:, 0, :]
            gather = jnp.asarray(
                component_lift_index_map_3d(level),
                dtype=INDEX_DTYPE,
            )
            # (targets, components(level), xyz) -> (targets, xyz, components(level))
            lifted = jnp.swapaxes(high[:, gather], 1, 2)
            sign = -1.0 if level % 2 == 0 else 1.0
            derivatives.append(sign * lifted)
        packed_derivatives: PackedAccelerationDerivatives = tuple(derivatives)
        if return_potential:
            return gradients, potentials, packed_derivatives
        return gradients, None, packed_derivatives

    offsets = target_positions - centers

    gradients, potentials = _evaluate_local_cartesian_with_grad_batch(
        coeffs,
        offsets,
        order=order,
    )
    if return_potential:
        return gradients, potentials, None
    return gradients, None, None


def _evaluate_prepared_tree_targets(
    *,
    fmm: "FastMultipoleMethod",
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    downward: TreeDownwardData,
    neighbor_list: NodeNeighborList,
    nearfield_interop: Optional[NearfieldInteropData],
    farfield_local_data: Optional[LocalExpansionData],
    farfield_leaf_nodes: Optional[Array],
    farfield_node_ranges: Optional[Array],
    target_sorted_indices: Array,
    return_potential: bool,
    max_acc_derivative_order: int = 0,
) -> Union[
    Array,
    Tuple[Array, Array],
    Tuple[Array, PackedAccelerationDerivatives],
    Tuple[Array, Array, PackedAccelerationDerivatives],
]:
    """Run prepared-tree evaluation for target particles only."""
    g_const = jnp.asarray(fmm.G, dtype=positions_sorted.dtype)
    nearfield_view = (
        _build_nearfield_interop_data(tree, neighbor_list)
        if nearfield_interop is None
        else nearfield_interop
    )
    node_views = _resolve_evaluation_node_views(
        tree,
        neighbor_list,
        farfield_leaf_nodes=farfield_leaf_nodes,
        farfield_node_ranges=farfield_node_ranges,
    )
    if nearfield_view.particle_to_leaf_position is not None:
        target_leaf_positions = jnp.asarray(
            nearfield_view.particle_to_leaf_position,
            dtype=INDEX_DTYPE,
        )[target_sorted_indices]
    else:
        target_leaf_positions = _map_targets_to_leaf_positions(
            target_sorted_indices=target_sorted_indices,
            leaf_nodes=nearfield_view.particle_order_leaf_indices,
            node_ranges=nearfield_view.particle_order_node_ranges,
        )
        target_leaf_positions = nearfield_view.particle_order_to_native_leaf[
            target_leaf_positions
        ]
    near_source_idx, near_source_mask = _build_target_nearfield_source_index_matrix(
        target_sorted_indices=target_sorted_indices,
        target_leaf_positions=target_leaf_positions,
        nearfield_interop=nearfield_view,
    )
    near_acc, near_pot, _, _, _ = _compute_targeted_nearfield(
        positions_sorted=positions_sorted,
        masses_sorted=masses_sorted,
        target_sorted_indices=target_sorted_indices,
        source_indices=near_source_idx,
        source_mask=near_source_mask,
        G=g_const,
        softening=float(fmm.softening),
        return_potential=return_potential,
    )
    far_grad, far_potential_pre, far_derivatives = (
        _evaluate_local_expansions_for_target_particles(
            local_data=downward.locals,
            positions_sorted=positions_sorted,
            target_sorted_indices=target_sorted_indices,
            target_leaf_positions=target_leaf_positions,
            leaf_nodes=node_views.farfield_leaf_nodes,
            order=int(downward.locals.order),
            expansion_basis=fmm.expansion_basis,
            return_potential=return_potential,
            max_acc_derivative_order=max_acc_derivative_order,
        )
    )
    far_acc = -g_const * far_grad
    acc_derivatives: Optional[PackedAccelerationDerivatives]
    if far_derivatives is not None:
        acc_derivatives = tuple(g_const * level for level in far_derivatives)
    else:
        acc_derivatives = None
    if return_potential:
        far_pot = (
            -g_const * far_potential_pre
            if far_potential_pre is not None
            else jnp.zeros(
                (target_sorted_indices.shape[0],), dtype=positions_sorted.dtype
            )
        )
        near_pot_resolved = (
            near_pot
            if near_pot is not None
            else jnp.zeros(
                (target_sorted_indices.shape[0],), dtype=positions_sorted.dtype
            )
        )
        if acc_derivatives is None:
            return near_acc + far_acc, near_pot_resolved + far_pot
        return near_acc + far_acc, near_pot_resolved + far_pot, acc_derivatives
    if acc_derivatives is None:
        return near_acc + far_acc
    return near_acc + far_acc, acc_derivatives


@partial(
    jax.jit,
    static_argnames=(
        "max_leaf_size",
        "return_potential",
        "order",
        "expansion_basis",
        "max_acc_derivative_order",
    ),
)
def _evaluate_local_expansions_for_particles(
    local_data: LocalExpansionData,
    positions: Array,
    *,
    leaf_nodes: Array,
    node_ranges: Array,
    max_leaf_size: int,
    order: int,
    expansion_basis: ExpansionBasis,
    return_potential: bool,
    max_acc_derivative_order: int = 0,
) -> tuple[Array, Optional[Array], Optional[PackedAccelerationDerivatives]]:
    """Evaluate node-local expansions at leaf particles and scatter results."""
    if order > MAX_MULTIPOLE_ORDER and expansion_basis != "solidfmm":
        raise NotImplementedError(
            "orders above 4 require expansion_basis='solidfmm'",
        )

    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1

    idx = jnp.arange(max_leaf_size, dtype=INDEX_DTYPE)
    starts = leaf_ranges[:, 0][:, None]
    particle_idx = starts + idx
    valid = idx[None, :] < counts[:, None]

    safe_idx = jnp.clip(
        particle_idx,
        min=0,
        max=positions.shape[0] - 1,
    )
    leaf_positions = positions[safe_idx]
    leaf_positions = jnp.where(valid[..., None], leaf_positions, 0.0)

    centers = local_data.centers[leaf_nodes]
    offsets = leaf_positions - centers[:, None, :]
    offsets = jnp.where(valid[..., None], offsets, 0.0)

    coeffs = local_data.coefficients[leaf_nodes]
    dtype = positions.dtype

    if expansion_basis == "solidfmm":
        p = int(order)

        # Complex solidfmm expects delta = center - eval_point (same as real)
        offsets_complex = centers[:, None, :] - leaf_positions
        offsets_complex = jnp.where(valid[..., None], offsets_complex, 0.0)

        # Real (Dehnen no-sqrt2) basis: locals are real-typed, evaluated with the
        # real L2P operator. Detected by coefficient dtype so no basis_mode needs
        # to be threaded through every caller.
        if not jnp.iscomplexobj(coeffs):
            # Real (Dehnen) branch: compute grad_field / potentials /
            # derivative_fields with the real L2P operators, then fall through
            # to the shared scatter below (identical to the complex path).
            if int(max_acc_derivative_order) <= 0:

                def evaluate_leaf_real(
                    coeffs_leaf: Array,
                    offsets_leaf: Array,
                    mask_leaf: Array,
                ) -> tuple[Array, Array]:
                    grads, values = jax.vmap(
                        lambda offset: evaluate_local_real_with_grad(
                            coeffs_leaf, offset, order=p
                        )
                    )(offsets_leaf)
                    # evaluate_local_real_with_grad returns d(phi)/d(delta) with
                    # delta = center - eval_point == the acceleration
                    # contribution consumed downstream.
                    grads = grads.astype(dtype)
                    values = values.astype(dtype)
                    grads = jnp.where(mask_leaf[..., None], grads, 0.0)
                    values = jnp.where(mask_leaf, values, 0.0)
                    return grads, values

                grad_field, potentials = jax.vmap(evaluate_leaf_real)(
                    coeffs,
                    offsets_complex,
                    valid,
                )
                derivative_fields = []
            else:

                def evaluate_leaf_real_with_derivatives(
                    coeffs_leaf: Array,
                    offsets_leaf: Array,
                    mask_leaf: Array,
                ) -> tuple[Array, Array, tuple[Array, ...]]:
                    tower = evaluate_local_real_derivative_tower_batch(
                        coeffs_leaf,
                        offsets_leaf,
                        order=p,
                        max_derivative_order=int(max_acc_derivative_order) + 1,
                    )
                    grads = tower[1].astype(dtype)
                    values = tower[0][:, 0].astype(dtype)
                    grads = jnp.where(mask_leaf[..., None], grads, 0.0)
                    values = jnp.where(mask_leaf, values, 0.0)
                    derivative_levels: list[Array] = []
                    for level in range(1, int(max_acc_derivative_order) + 1):
                        high = tower[level + 1]
                        gather = jnp.asarray(
                            component_lift_index_map_3d(level),
                            dtype=INDEX_DTYPE,
                        )
                        lifted = jnp.swapaxes(high[:, gather], 1, 2)
                        sign = -1.0 if level % 2 == 0 else 1.0
                        lifted = (sign * lifted).astype(dtype)
                        lifted = jnp.where(mask_leaf[:, None, None], lifted, 0.0)
                        derivative_levels.append(lifted)
                    return grads, values, tuple(derivative_levels)

                grad_field, potentials, derivative_fields_tuple = jax.vmap(
                    evaluate_leaf_real_with_derivatives
                )(
                    coeffs,
                    offsets_complex,
                    valid,
                )
                derivative_fields = list(derivative_fields_tuple)
        elif max_acc_derivative_order <= 0:
            if not bool(return_potential):
                flat_analytic = str(
                    os.environ.get(
                        "JACCPOT_LOCAL_EVAL_FLAT_ANALYTIC",
                        "0",
                    )
                ).strip().lower() in {"1", "true", "yes", "on"}

                dtype_preserve_analytic = str(
                    os.environ.get(
                        "JACCPOT_LOCAL_EVAL_DTYPE_PRESERVE",
                        "0",
                    )
                ).strip().lower() in {"1", "true", "yes", "on"}
                order4_unrolled_analytic = str(
                    os.environ.get(
                        "JACCPOT_LOCAL_EVAL_ORDER4_UNROLLED",
                        "0",
                    )
                ).strip().lower() in {"1", "true", "yes", "on"}
                eval_complex_grad = (
                    evaluate_local_complex_grad_order4_unrolled
                    if bool(order4_unrolled_analytic) and p == 4
                    else (
                        evaluate_local_complex_grad_analytic_preserve_dtype
                        if bool(dtype_preserve_analytic)
                        else evaluate_local_complex_grad_analytic
                    )
                )

                if bool(flat_analytic):
                    coeffs_flat = jnp.broadcast_to(
                        coeffs[:, None, :],
                        offsets_complex.shape[:-1] + (coeffs.shape[-1],),
                    ).reshape((-1, coeffs.shape[-1]))
                    offsets_flat = offsets_complex.reshape(
                        (-1, offsets_complex.shape[-1])
                    )
                    mask_flat = valid.reshape((-1,))
                    grad_flat = jax.vmap(
                        lambda coeff_row, offset_row: eval_complex_grad(
                            coeff_row,
                            offset_row,
                            order=p,
                        )
                    )(coeffs_flat, offsets_flat)
                    grad_flat = grad_flat.astype(dtype)
                    grad_flat = jnp.where(mask_flat[:, None], grad_flat, 0.0)
                    grad_field = grad_flat.reshape(valid.shape + (3,))
                else:

                    def evaluate_leaf_complex_grad_only(
                        coeffs_leaf: Array,
                        offsets_leaf: Array,
                        mask_leaf: Array,
                    ) -> Array:
                        grads = jax.vmap(
                            lambda offset: eval_complex_grad(
                                coeffs_leaf,
                                offset,
                                order=p,
                            )
                        )(offsets_leaf)
                        grads = grads.astype(dtype)
                        return jnp.where(mask_leaf[..., None], grads, 0.0)

                    grad_field = jax.vmap(evaluate_leaf_complex_grad_only)(
                        coeffs,
                        offsets_complex,
                        valid,
                    )
                potentials = None
            else:

                def evaluate_leaf_complex(
                    coeffs_leaf: Array,
                    offsets_leaf: Array,
                    mask_leaf: Array,
                ) -> tuple[Array, Array]:
                    grads, values = evaluate_local_complex_with_grad_analytic_batch(
                        coeffs_leaf,
                        offsets_leaf,
                        order=p,
                    )
                    grads = grads.astype(dtype)
                    values = values.astype(dtype)
                    grads = jnp.where(mask_leaf[..., None], grads, 0.0)
                    values = jnp.where(mask_leaf, values, 0.0)
                    return grads, values

                grad_field, potentials = jax.vmap(evaluate_leaf_complex)(
                    coeffs,
                    offsets_complex,
                    valid,
                )
            derivative_fields: list[Array] = []
        else:

            def evaluate_leaf_complex_with_derivatives(
                coeffs_leaf: Array,
                offsets_leaf: Array,
                mask_leaf: Array,
            ) -> tuple[Array, Array, tuple[Array, ...]]:
                tower = evaluate_local_complex_derivative_tower_batch(
                    coeffs_leaf,
                    offsets_leaf,
                    order=p,
                    max_derivative_order=int(max_acc_derivative_order) + 1,
                )
                grads = tower[1].astype(dtype)
                values = tower[0][:, 0].astype(dtype)
                grads = jnp.where(mask_leaf[..., None], grads, 0.0)
                values = jnp.where(mask_leaf, values, 0.0)
                derivative_levels: list[Array] = []
                for level in range(1, int(max_acc_derivative_order) + 1):
                    high = tower[level + 1]
                    gather = jnp.asarray(
                        component_lift_index_map_3d(level),
                        dtype=INDEX_DTYPE,
                    )
                    lifted = jnp.swapaxes(high[:, gather], 1, 2)
                    sign = -1.0 if level % 2 == 0 else 1.0
                    lifted = (sign * lifted).astype(dtype)
                    lifted = jnp.where(mask_leaf[:, None, None], lifted, 0.0)
                    derivative_levels.append(lifted)
                return grads, values, tuple(derivative_levels)

            grad_field, potentials, derivative_fields_tuple = jax.vmap(
                evaluate_leaf_complex_with_derivatives
            )(
                coeffs,
                offsets_complex,
                valid,
            )
            derivative_fields = list(derivative_fields_tuple)

        direct_leaf_flatten = str(
            os.environ.get(
                "JACCPOT_LOCAL_EVAL_DIRECT_LEAF_FLATTEN",
                "0",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        if bool(direct_leaf_flatten) and max_acc_derivative_order <= 0:
            gradients = grad_field.reshape((-1, grad_field.shape[-1]))[
                : positions.shape[0]
            ]
        else:
            gradients = _scatter_vectors(
                jnp.zeros_like(positions),
                safe_idx,
                grad_field,
                valid,
            )

        derivative_outputs: Optional[PackedAccelerationDerivatives]
        if max_acc_derivative_order > 0:
            derivative_outputs = []
            for level, deriv_field in enumerate(derivative_fields, start=1):
                scattered = _scatter_rank3(
                    jnp.zeros(
                        (
                            positions.shape[0],
                            3,
                            len(component_lift_index_map_3d(level)),
                        ),
                        dtype=positions.dtype,
                    ),
                    safe_idx,
                    deriv_field,
                    valid,
                )
                derivative_outputs.append(scattered)
            derivative_outputs = tuple(derivative_outputs)
        else:
            derivative_outputs = None

        if not return_potential:
            return gradients, None, derivative_outputs

        potentials_flat = _scatter_scalars(
            jnp.zeros((positions.shape[0],), dtype=dtype),
            safe_idx,
            potentials,
            valid,
        )
        return gradients, potentials_flat, derivative_outputs

    coeffs_broadcast = jnp.broadcast_to(
        coeffs[:, None, :],
        offsets.shape[:-1] + (coeffs.shape[-1],),
    )
    grad_field, potentials = _evaluate_local_cartesian_with_grad_batch(
        coeffs_broadcast,
        offsets,
        order=order,
    )
    grad_field = jnp.where(valid[..., None], grad_field, 0.0)
    potentials = jnp.where(valid, potentials, 0.0)

    gradients = _scatter_vectors(
        jnp.zeros_like(positions),
        safe_idx,
        grad_field,
        valid,
    )

    if not return_potential:
        return gradients, None, None

    potentials_flat = _scatter_scalars(
        jnp.zeros((positions.shape[0],), dtype=dtype),
        safe_idx,
        potentials,
        valid,
    )
    return gradients, potentials_flat, None


def _scatter_vectors(
    base: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Scatter-add vector values into a flat particle buffer with masking."""
    if values.size == 0:
        return base
    flat_idx = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_mask = mask.reshape(-1)
    zero = jnp.zeros((), dtype=base.dtype)
    masked = jnp.where(flat_mask[:, None], flat_values, zero)
    return base.at[flat_idx].add(masked)


def _scatter_scalars(
    base: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Scatter-add scalar values into a flat particle buffer with masking."""
    if values is None or values.size == 0:
        return base
    flat_idx = indices.reshape(-1)
    flat_values = values.reshape(-1)
    flat_mask = mask.reshape(-1)
    zero = jnp.zeros((), dtype=base.dtype)
    masked = jnp.where(flat_mask, flat_values, zero)
    return base.at[flat_idx].add(masked)


def _scatter_rank3(
    base: Array,
    indices: Array,
    values: Array,
    mask: Array,
) -> Array:
    """Scatter-add rank-3 values into a particle-major buffer."""
    if values.size == 0:
        return base
    flat_idx = indices.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-2], values.shape[-1])
    flat_mask = mask.reshape(-1)
    zero = jnp.zeros((), dtype=base.dtype)
    masked = jnp.where(flat_mask[:, None, None], flat_values, zero)
    return base.at[flat_idx].add(masked)


@jaxtyped(typechecker=beartype)
def compute_gravitational_acceleration(
    positions: Array,
    masses: Array,
    theta: float = 0.5,
    G: Union[float, Array] = 1.0,
    softening: Union[float, Array] = 0.0,
    *,
    bounds: Optional[Tuple[Array, Array]] = None,
    leaf_size: int = 16,
    max_order: int = 2,
    return_potential: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """Compute gravitational accelerations via the Fast Multipole Method."""

    fmm = FastMultipoleMethod(
        theta=theta,
        G=G,
        softening=softening,
    )
    return fmm.compute_accelerations(
        positions,
        masses,
        bounds=bounds,
        leaf_size=leaf_size,
        max_order=max_order,
        return_potential=return_potential,
    )


@jaxtyped(typechecker=beartype)
def compute_gravitational_potential(
    positions: Array,
    masses: Array,
    eval_points: Array,
    G: Union[float, Array] = 1.0,
    softening: Union[float, Array] = 0.0,
) -> Array:
    """Compute gravitational potential at evaluation points."""

    return reference_compute_potential(
        positions,
        masses,
        eval_points,
        G=G,
        softening=softening,
    )
