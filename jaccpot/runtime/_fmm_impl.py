"""
Fast Multipole Method (FMM) for computing gravitational accelerations.

This implementation uses multipole and local expansions to compute
gravitational forces in O(N) time instead of O(N^2) for direct summation.
"""

import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Callable, Tuple
from jaxtyping import Array, DTypeLike, jaxtyped
from yggdrax.dense_interactions import DenseInteractionBuffers
from yggdrax.grouped_interactions import (
    GroupedInteractionBuffers,
    build_grouped_interactions,
)
from yggdrax.interactions import (
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    DualTreeWalkResult,
    MACType,
    NodeInteractionList,
    NodeNeighborList,
    build_interactions_and_neighbors,
    build_well_separated_interactions,
)
from yggdrax.morton import morton_encode
from yggdrax.tree import (
    RadixTree,
    Tree,
    TreeType,
    available_tree_types,
    reorder_particles_by_indices,
)

from jaccpot.basis.real_sh import complex_to_real_coeffs
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
from jaccpot.nearfield.near_field import (
    compute_leaf_p2p_accelerations,
    prepare_bucketed_scatter_schedules,
    prepare_leaf_neighbor_pairs,
)
from jaccpot.operators.complex_ops import (
    complex_rotation_blocks_from_z_batch,
    complex_rotation_blocks_from_z_solidfmm_batch,
    complex_rotation_blocks_to_z_batch,
    complex_rotation_blocks_to_z_solidfmm_batch,
    enforce_conjugate_symmetry_batch,
    evaluate_local_complex_with_grad,
    evaluate_local_complex_with_grad_batch,
    l2l_complex,
    l2l_complex_batch,
    m2l_complex_reference,
    m2l_complex_reference_batch,
    m2l_complex_reference_batch_cached_blocks,
)
from jaccpot.operators.m2l_real_rot_scale import m2l_rot_scale_real_batch
from jaccpot.operators.multipole_utils import (
    LOCAL_LEVEL_COMBOS,
    MAX_MULTIPOLE_ORDER,
    level_offset,
    total_coefficients,
)
from jaccpot.operators.real_harmonics import (
    evaluate_local_real_with_grad,
    l2l_real,
    sh_size,
)
from jaccpot.upward.solidfmm_complex_tree_expansions import (
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
    _DualTreeArtifacts,
    _interaction_cache_key,
    _InteractionCacheEntry,
)
from ._nearfield_cache import (
    NearfieldPrecomputeArtifacts,
    nearfield_cache_matches,
    nearfield_from_cache,
    with_nearfield_cache_artifacts,
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

_cartesian_eval_table_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _cartesian_eval_tables(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened cartesian local-evaluation tables for one fixed order."""
    order_int = int(order)
    cached = _cartesian_eval_table_cache.get(order_int)
    if cached is not None:
        return cached

    coeff_indices: list[int] = []
    exponents: list[tuple[int, int, int]] = []
    inv_factorials: list[float] = []
    for level_idx in range(order_int + 1):
        start_idx = level_offset(level_idx)
        combos = LOCAL_LEVEL_COMBOS[level_idx]
        for combo_idx, combo in enumerate(combos):
            coeff_indices.append(start_idx + combo_idx)
            exponents.append((int(combo[0]), int(combo[1]), int(combo[2])))
            inv_factorials.append(
                float(
                    1.0
                    / (
                        math.factorial(int(combo[0]))
                        * math.factorial(int(combo[1]))
                        * math.factorial(int(combo[2]))
                    )
                )
            )

    coeff_idx_np = np.asarray(coeff_indices, dtype=np.int32)
    exponents_np = np.asarray(exponents, dtype=np.int32)
    inv_fact_np = np.asarray(inv_factorials, dtype=np.float64)
    tables = (coeff_idx_np, exponents_np, inv_fact_np)
    _cartesian_eval_table_cache[order_int] = tables
    return tables


@partial(jax.jit, static_argnames=("order",))
def _evaluate_local_cartesian_with_grad_batch(
    coeffs: Array,
    offsets: Array,
    *,
    order: int,
) -> tuple[Array, Array]:
    """Evaluate cartesian locals and gradients for a batch of offsets."""
    coeff_rows = jnp.asarray(coeffs)
    delta_rows = jnp.asarray(offsets, dtype=coeff_rows.dtype)
    coeff_shape = coeff_rows.shape[:-1]
    delta_shape = delta_rows.shape[:-1]
    if coeff_shape != delta_shape:
        raise ValueError("coeffs and offsets must share leading batch dimensions")

    coeff_flat = coeff_rows.reshape((-1, coeff_rows.shape[-1]))
    delta_flat = delta_rows.reshape((-1, 3))
    coeff_idx_np, exponents_np, inv_fact_np = _cartesian_eval_tables(order)
    coeff_idx = jnp.asarray(coeff_idx_np, dtype=INDEX_DTYPE)
    exponents = jnp.asarray(exponents_np, dtype=INDEX_DTYPE)
    inv_fact = jnp.asarray(inv_fact_np, dtype=coeff_flat.dtype)

    ex = exponents[:, 0]
    ey = exponents[:, 1]
    ez = exponents[:, 2]
    exm1 = jnp.maximum(ex - 1, 0)
    eym1 = jnp.maximum(ey - 1, 0)
    ezm1 = jnp.maximum(ez - 1, 0)

    x = delta_flat[:, 0:1]
    y = delta_flat[:, 1:2]
    z = delta_flat[:, 2:3]

    x_pow = jnp.power(x, ex[None, :])
    y_pow = jnp.power(y, ey[None, :])
    z_pow = jnp.power(z, ez[None, :])

    coeff_terms = coeff_flat[:, coeff_idx] * inv_fact[None, :]
    monomials = x_pow * y_pow * z_pow
    values = jnp.sum(coeff_terms * monomials, axis=1)

    ex_f = ex.astype(coeff_flat.dtype)
    ey_f = ey.astype(coeff_flat.dtype)
    ez_f = ez.astype(coeff_flat.dtype)
    grad_x = jnp.sum(
        coeff_terms * ex_f[None, :] * jnp.power(x, exm1[None, :]) * y_pow * z_pow,
        axis=1,
    )
    grad_y = jnp.sum(
        coeff_terms * ey_f[None, :] * x_pow * jnp.power(y, eym1[None, :]) * z_pow,
        axis=1,
    )
    grad_z = jnp.sum(
        coeff_terms * ez_f[None, :] * x_pow * y_pow * jnp.power(z, ezm1[None, :]),
        axis=1,
    )
    grads = jnp.stack((grad_x, grad_y, grad_z), axis=1)
    return grads.reshape(coeff_shape + (3,)), values.reshape(coeff_shape)


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


@dataclass(frozen=True)
class _TopologyReuseEntry:
    """Cached topology metadata for bounded multi-step reuse."""

    key: str
    tree: Tree
    max_leaf_size: int
    cache_leaf_parameter: int
    reuse_count: int


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
_NEARFIELD_GPU_PRECOMPUTE_MAX_PARTICLES = 262_144
_LARGE_CPU_M2L_CHUNK_SIZE = 32768
_TRACING_MAX_NEIGHBORS_PER_LEAF = 512
# Traced prepare_state uses static-capacity interaction buffers. This cap limits
# max_interactions_per_node only in traced mode (outer jax.jit prepare path) to
# keep padded far-field buffers from dominating runtime. Lower is faster but can
# trigger traversal overflow/retry on harder particle configurations.
_TRACING_MAX_INTERACTIONS_PER_NODE = 512
_GPU_LARGE_PARTICLE_THRESHOLD = 65_536
_GPU_MAX_NEIGHBORS_PER_LEAF = 1024
_GPU_MAX_INTERACTIONS_PER_NODE = 4096
_GPU_MIN_PAIR_QUEUE_MEDIUM = 131_072
_GPU_MIN_PAIR_QUEUE_LARGE = 262_144
_GPU_MIN_PAIR_QUEUE_XL = 524_288
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
_OPERATOR_CACHE_MAX = 512
_operator_blocks_cache: "OrderedDict[tuple, tuple[Array, Array]]" = OrderedDict()
_GROUPED_OPERATOR_CACHE_MAX = 32
_grouped_operator_blocks_cache: "OrderedDict[tuple, tuple[Array, Array]]" = (
    OrderedDict()
)
_GROUPED_SEGMENT_CACHE_MAX = 32
_grouped_segment_cache: (
    "OrderedDict[tuple, tuple[Array, Array, Array, Array, Array, Array]]"
) = OrderedDict()
_GROUPED_OPERATOR_CACHE_ENTRY_MAX_BYTES = 64 * 1024 * 1024
_GROUPED_OPERATOR_CACHE_TOTAL_MAX_BYTES = 256 * 1024 * 1024
_GROUPED_SEGMENT_CACHE_ENTRY_MAX_BYTES = 32 * 1024 * 1024
_GROUPED_SEGMENT_CACHE_TOTAL_MAX_BYTES = 128 * 1024 * 1024
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
) -> Optional[tuple[Array, Array, Array, Array, Array, Array]]:
    cached = _grouped_segment_cache.get(key)
    if cached is None:
        return None
    _grouped_segment_cache.move_to_end(key)
    return cached


def _grouped_segment_cache_put(
    key: tuple,
    value: tuple[Array, Array, Array, Array, Array, Array],
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
    valid_tree_modes = {"lbvh", "fixed_depth", "adaptive"}
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

    del jit_tree

    mode = tree_config.mode
    build_mode = "fixed_depth" if mode == "fixed_depth" else "adaptive"
    built_tree = Tree.from_particles(
        positions,
        masses,
        tree_type=tree_type,
        build_mode=build_mode,
        bounds=bounds,
        return_reordered=True,
        workspace=workspace if tree_type == "radix" else None,  # type: ignore[arg-type]
        return_workspace=(tree_type == "radix"),
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
        int(leaf_size) if mode == "lbvh" else tree_config.target_leaf_particles
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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FMMPreparedState:
    """Keep prepared tree artifacts resident as a JAX pytree payload.

    The array/tree payload is carried as pytree children so callers can pass
    this state through ``jax.jit``. Non-array metadata is tracked as static
    auxiliary data to avoid tracing errors on dtype/string objects.
    """

    tree: Tree
    upward: TreeUpwardData
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
    nearfield_target_leaf_ids: Optional[Array]
    nearfield_source_leaf_ids: Optional[Array]
    nearfield_valid_pairs: Optional[Array]
    nearfield_chunk_sort_indices: Optional[Array]
    nearfield_chunk_group_ids: Optional[Array]
    nearfield_chunk_unique_indices: Optional[Array]
    force_scale_nodes: Optional[Array]

    @property
    def positions_sorted(self) -> Array:
        """Canonical sorted particle positions owned by ``tree``."""
        value = getattr(self.tree, "positions_sorted", None)
        if value is None:
            raise ValueError("prepared tree is missing positions_sorted")
        return jnp.asarray(value)

    @property
    def masses_sorted(self) -> Array:
        """Canonical sorted particle masses owned by ``tree``."""
        value = getattr(self.tree, "masses_sorted", None)
        if value is None:
            raise ValueError("prepared tree is missing masses_sorted")
        return jnp.asarray(value)

    @property
    def inverse_permutation(self) -> Array:
        """Canonical inverse permutation owned by ``tree``."""
        value = getattr(self.tree, "inverse_permutation", None)
        if value is None:
            raise ValueError("prepared tree is missing inverse_permutation")
        return jnp.asarray(value, dtype=INDEX_DTYPE)

    def tree_flatten(
        self: "FMMPreparedState",
    ) -> tuple[
        tuple[Any, ...],
        tuple[int, str, str, str, float, Optional[str], Tuple[DualTreeRetryEvent, ...]],
    ]:
        children = (
            self.tree,
            self.upward,
            self.downward,
            self.neighbor_list,
            self.interactions,
            self.dual_tree_result,
            self.nearfield_target_leaf_ids,
            self.nearfield_source_leaf_ids,
            self.nearfield_valid_pairs,
            self.nearfield_chunk_sort_indices,
            self.nearfield_chunk_group_ids,
            self.nearfield_chunk_unique_indices,
            self.force_scale_nodes,
        )
        aux = (
            int(self.max_leaf_size),
            str(jnp.dtype(self.input_dtype)),
            str(jnp.dtype(self.working_dtype)),
            str(self.expansion_basis),
            float(self.theta),
            self.topology_key,
            self.retry_events,
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
        ) = aux
        (
            tree,
            upward,
            downward,
            neighbor_list,
            interactions,
            dual_tree_result,
            nearfield_target_leaf_ids,
            nearfield_source_leaf_ids,
            nearfield_valid_pairs,
            nearfield_chunk_sort_indices,
            nearfield_chunk_group_ids,
            nearfield_chunk_unique_indices,
            force_scale_nodes,
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
            nearfield_target_leaf_ids=nearfield_target_leaf_ids,
            nearfield_source_leaf_ids=nearfield_source_leaf_ids,
            nearfield_valid_pairs=nearfield_valid_pairs,
            nearfield_chunk_sort_indices=nearfield_chunk_sort_indices,
            nearfield_chunk_group_ids=nearfield_chunk_group_ids,
            nearfield_chunk_unique_indices=nearfield_chunk_unique_indices,
            force_scale_nodes=force_scale_nodes,
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
    downward: TreeDownwardData
    cache_entry: Optional[_InteractionCacheEntry]


class _FarPairCOO(NamedTuple):
    """Compact COO-style far-pair representation for streamed M2L execution."""

    sources: Array
    targets: Array


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
    interactions: NodeInteractionList,
) -> NodeInteractionList:
    """Return zero-pair interaction storage while preserving node-shaped metadata."""

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
        use_pallas: bool = False,
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
        nearfield_edge_chunk_size: int = 256,
        precompute_nearfield_scatter_schedules: bool = True,
        enable_interaction_cache: bool = True,
        retain_traversal_result: bool = True,
        retain_interactions: bool = True,
        autotune_m2l_chunk: bool = False,
        host_refine_mode: str = "auto",
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
        self.use_pallas = bool(use_pallas)
        self.reuse_topology = bool(reuse_topology)
        if int(rebuild_every) <= 0:
            raise ValueError("rebuild_every must be positive")
        self.rebuild_every = int(rebuild_every)
        self._recent_far_pairs_by_gear_counts: tuple[int, ...] = tuple()
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
        if rotation_norm not in ("bdz", "cached", "wigner", "solidfmm"):
            raise ValueError(
                "complex_rotation must be 'bdz', 'cached', 'wigner', or 'solidfmm'"
            )
        if basis_norm == "solidfmm" and rotation_norm != "solidfmm":
            raise ValueError(
                "expansion_basis='solidfmm' requires complex_rotation='solidfmm'"
            )
        self.complex_rotation = rotation_norm
        farfield_mode_norm = str(farfield_mode).strip().lower()
        if farfield_mode_norm not in ("auto", "pair_grouped", "class_major"):
            raise ValueError(
                "farfield_mode must be 'auto', 'pair_grouped', or 'class_major'"
            )
        self.farfield_mode = farfield_mode_norm
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
        if int(nearfield_edge_chunk_size) <= 0:
            raise ValueError("nearfield_edge_chunk_size must be positive")
        self.nearfield_mode = nearfield_mode_norm
        self.nearfield_edge_chunk_size = int(nearfield_edge_chunk_size)
        self.precompute_nearfield_scatter_schedules = bool(
            precompute_nearfield_scatter_schedules
        )
        self.enable_interaction_cache = bool(enable_interaction_cache)
        self.retain_traversal_result = bool(retain_traversal_result)
        self.retain_interactions = bool(retain_interactions)
        self.autotune_m2l_chunk = bool(autotune_m2l_chunk)
        dehnen_scale_val = float(dehnen_radius_scale)
        if dehnen_scale_val <= 0.0:
            raise ValueError("dehnen_radius_scale must be > 0")
        self.dehnen_radius_scale = dehnen_scale_val

        refine_mode_norm = str(host_refine_mode).strip().lower()
        if refine_mode_norm not in ("auto", "on", "off"):
            raise ValueError("host_refine_mode must be 'auto', 'on', or 'off'")
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
        self._prepared_state_cache_key: Optional[tuple[Any, ...]] = None
        self._prepared_state_cache_value: Optional[FMMPreparedState] = None
        self._prepared_state_cache_positions: Optional[Array] = None
        self._prepared_state_cache_masses: Optional[Array] = None
        self._topology_reuse_entry: Optional[_TopologyReuseEntry] = None
        self._recent_topology_reused: bool = False
        self._recent_retry_events: Tuple[DualTreeRetryEvent, ...] = tuple()
        self.fixed_order = fixed_order
        self.fixed_max_leaf_size = fixed_max_leaf_size
        self._explicit_m2l_chunk_size = m2l_chunk_size is not None
        self._explicit_l2l_chunk_size = l2l_chunk_size is not None
        self._explicit_traversal_config = traversal_config is not None
        self._explicit_max_pair_queue = max_pair_queue is not None
        self._explicit_pair_process_block = pair_process_block is not None
        self._explicit_grouped_interactions = grouped_interactions is not None
        self.grouped_interactions = grouped_interactions

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
        self._tree_workspace = None
        self._last_force_scale_nodes = None
        self._recent_retry_events = tuple()
        self._recent_far_pairs_by_gear_counts = tuple()
        _clear_global_runtime_caches(clear_jax_compilation=bool(clear_jax_compilation))

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

    def _uses_dehnen_paper_error_model(self: "FastMultipoleMethod") -> bool:
        """Return whether the active adaptive error model is the paper estimator."""

        return self.adaptive_error_model == "dehnen_paper"

    def _force_scale_reduction_mode(self: "FastMultipoleMethod") -> str:
        """Return the node reduction mode used for adaptive force scales."""

        return "min" if self._uses_dehnen_paper_error_model() else "max"

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
            nearfield_artifacts = self._prepare_state_nearfield_artifacts(
                tree=low_tree_artifacts.tree,
                neighbor_list=dual_downward_artifacts.neighbor_list,
                leaf_cap=low_tree_artifacts.leaf_cap,
                num_particles=int(low_tree_artifacts.positions_sorted.shape[0]),
                cache_entry=dual_downward_artifacts.cache_entry,
                allow_stateful_cache=False,
            )
            prepass_state = FMMPreparedState(
                tree=low_tree_artifacts.tree,
                upward=low_tree_artifacts.upward,
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
                nearfield_target_leaf_ids=nearfield_artifacts.target_leaf_ids,
                nearfield_source_leaf_ids=nearfield_artifacts.source_leaf_ids,
                nearfield_valid_pairs=nearfield_artifacts.valid_pairs,
                nearfield_chunk_sort_indices=nearfield_artifacts.chunk_sort_indices,
                nearfield_chunk_group_ids=nearfield_artifacts.chunk_group_ids,
                nearfield_chunk_unique_indices=nearfield_artifacts.chunk_unique_indices,
                force_scale_nodes=None,
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
    ) -> Optional[FMMPreparedState]:
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
        state: FMMPreparedState,
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
        ):
            grouped_interactions = True

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

            if n_particles >= 4_194_304:
                target_queue = _GPU_MIN_PAIR_QUEUE_XL
            elif n_particles >= 1_048_576:
                target_queue = _GPU_MIN_PAIR_QUEUE_LARGE
            else:
                target_queue = _GPU_MIN_PAIR_QUEUE_MEDIUM

            next_queue = max(current_queue, int(target_queue))
            next_interactions = min(
                current_interactions, int(_GPU_MAX_INTERACTIONS_PER_NODE)
            )
            next_neighbors = min(current_neighbors, int(_GPU_MAX_NEIGHBORS_PER_LEAF))
            if (
                next_queue != current_queue
                or next_interactions != current_interactions
                or next_neighbors != current_neighbors
            ):
                traversal_config = DualTreeTraversalConfig(
                    max_pair_queue=int(next_queue),
                    process_block=int(current_block),
                    max_interactions_per_node=int(next_interactions),
                    max_neighbors_per_leaf=int(next_neighbors),
                )
        if grouped_interactions:
            center_mode = "aabb"
            if farfield_mode == "auto":
                farfield_mode = (
                    "class_major"
                    if (class_major_cpu or class_major_gpu)
                    else "pair_grouped"
                )
        else:
            farfield_mode = "pair_grouped"

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

        current_neighbors = int(traversal_config.max_neighbors_per_leaf)
        capped_neighbors = min(current_neighbors, _TRACING_MAX_NEIGHBORS_PER_LEAF)
        current_interactions = int(traversal_config.max_interactions_per_node)
        capped_interactions = min(
            current_interactions, _TRACING_MAX_INTERACTIONS_PER_NODE
        )
        if (
            capped_neighbors == current_neighbors
            and capped_interactions == current_interactions
        ):
            return traversal_config

        return DualTreeTraversalConfig(
            max_pair_queue=int(traversal_config.max_pair_queue),
            process_block=int(traversal_config.process_block),
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
            return _infer_bounds(positions)
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
            not self.reuse_topology
            or not allow_stateful_cache
            or self.tree_type != "radix"
        ):
            return None
        try:
            morton_codes = morton_encode(positions, bounds)
            orig_idx = jnp.arange(positions.shape[0], dtype=INDEX_DTYPE)
            sorted_indices = jnp.lexsort((orig_idx, morton_codes))
            sorted_codes = morton_codes[sorted_indices]
            hasher = hashlib.sha256()
            hasher.update(
                np.asarray(jax.device_get(sorted_codes), dtype=np.uint64).tobytes()
            )
            hasher.update(
                np.asarray(jax.device_get(sorted_indices), dtype=np.int64).tobytes()
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
        return _TopologyReuseCandidate(
            key=hasher.hexdigest(),
            sorted_indices=jnp.asarray(sorted_indices, dtype=INDEX_DTYPE),
        )

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
        rebuilt_tree = RadixTree(
            topology=cached_tree.topology,
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
            total_nodes = int(tree.parent.shape[0])
            coeff_count = sh_size(max_order)
            if self._solidfmm_basis_mode() == "real":
                coeff_dtype = pos_sorted.dtype
            else:
                coeff_dtype = complex_dtype_for_real(pos_sorted.dtype)
            return LocalExpansionData(
                order=max_order,
                centers=upward.multipoles.centers,
                coefficients=jnp.zeros((total_nodes, coeff_count), dtype=coeff_dtype),
            )

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
        if self.tree_type != "radix" and tree_config.mode == "fixed_depth":
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
        self._recent_topology_reused = False
        cached_topology = self._topology_reuse_entry
        can_reuse_cached_topology = (
            topology_candidate is not None
            and cached_topology is not None
            and topology_candidate.key == cached_topology.key
            and cached_topology.reuse_count < (self.rebuild_every - 1)
        )

        if can_reuse_cached_topology:
            build_artifacts = self._rebuild_tree_artifacts_from_topology(
                candidate=topology_candidate,
                entry=cached_topology,
                positions=positions_arr,
                masses=masses_arr,
            )
            self._recent_topology_reused = True
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
                if topology_candidate is not None:
                    self._topology_reuse_entry = _TopologyReuseEntry(
                        key=topology_candidate.key,
                        tree=build_artifacts.tree,
                        max_leaf_size=int(build_artifacts.max_leaf_size),
                        cache_leaf_parameter=int(build_artifacts.cache_leaf_parameter),
                        reuse_count=0,
                    )
                elif self.reuse_topology:
                    self._topology_reuse_entry = None

        tree = build_artifacts.tree
        pos_sorted = build_artifacts.positions_sorted
        mass_sorted = build_artifacts.masses_sorted
        # Keep one leaf-size contract in eager and traced paths.
        leaf_cap_hint = int(build_artifacts.max_leaf_size)
        upward = self.prepare_upward_sweep(
            tree,
            pos_sorted,
            mass_sorted,
            max_order=max_order,
            center_mode=upward_center_mode,
            max_leaf_size=leaf_cap_hint,
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
            topology_key=None if topology_candidate is None else topology_candidate.key,
            upward=upward,
            locals_template=locals_template,
        )

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
    ) -> _PrepareStateDualDownwardArtifacts:
        """Build/reuse interactions and prepare downward artifacts."""
        pair_policy = None
        policy_state = None
        cache_key = None
        use_paper_fixed_policy = (
            not self.adaptive_order
        ) and self._uses_dehnen_paper_error_model()
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
                    self._adaptive_error_model_code(),
                    dtype=jnp.int32,
                ),
                dehnen_geometry_mode=self.dehnen_geometry_mode,
            )
            pair_policy = adaptive_pair_policy
        else:
            cache_key = _interaction_cache_key(
                tree_artifacts.tree,
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

        stateful_cache_enabled = bool(allow_stateful_cache) and bool(
            self.enable_interaction_cache
        )
        dual_artifacts, cache_entry = _build_dual_tree_artifacts(
            tree_artifacts.tree,
            tree_artifacts.upward.geometry,
            theta=theta_val,
            mac_type=mac_type_val,
            dehnen_radius_scale=dehnen_radius_scale,
            cache_key=cache_key,
            cache_entry=(self._interaction_cache if stateful_cache_enabled else None),
            max_pair_queue=self.max_pair_queue,
            pair_process_block=self.pair_process_block,
            traversal_config=runtime_traversal_config,
            retry_logger=record_retry,
            use_dense_interactions=self.use_dense_interactions,
            grouped_interactions=grouped_interactions,
            grouped_chunk_size=runtime_m2l_chunk_size,
            pair_policy=pair_policy,
            policy_state=policy_state,
        )
        if stateful_cache_enabled:
            self._interaction_cache = cache_entry

        (
            interactions,
            neighbor_list,
            traversal_result,
            dense_buffers,
            grouped_buffers,
            grouped_segment_starts,
            grouped_segment_lengths,
            grouped_segment_class_ids,
            grouped_segment_sort_permutation,
            grouped_segment_group_ids,
            grouped_segment_unique_targets,
        ) = self._unpack_dual_tree_artifacts(dual_artifacts)

        far_pairs_by_gear = None
        far_pairs_coo: Optional[_FarPairCOO] = None
        adaptive_order_for_downward = bool(self.adaptive_order)
        p_gears_for_downward = self.p_gears
        if self.adaptive_order:
            if len(self.p_gears) == 0:
                raise ValueError("adaptive_order=True requires non-empty p_gears")
            far_total = int(traversal_result.far_pair_count)
            far_pairs_by_gear = bucket_far_pairs_by_tag(
                jnp.asarray(
                    traversal_result.interaction_sources[:far_total], dtype=INDEX_DTYPE
                ),
                jnp.asarray(
                    traversal_result.interaction_targets[:far_total], dtype=INDEX_DTYPE
                ),
                jnp.asarray(
                    traversal_result.interaction_tags[:far_total], dtype=INDEX_DTYPE
                ),
                num_tags=len(self.p_gears),
            )
            self._recent_far_pairs_by_gear_counts = tuple(
                int(bucket_src.shape[0]) for bucket_src, _ in far_pairs_by_gear
            )
        elif self.streamed_far_pairs:
            far_total = int(traversal_result.far_pair_count)
            src_far = jnp.asarray(
                traversal_result.interaction_sources[:far_total], dtype=INDEX_DTYPE
            )
            tgt_far = jnp.asarray(
                traversal_result.interaction_targets[:far_total], dtype=INDEX_DTYPE
            )
            far_pairs_coo = _FarPairCOO(sources=src_far, targets=tgt_far)
            max_order_int = int(tree_artifacts.upward.multipoles.order)
            adaptive_order_for_downward = True
            if self.mixed_order_farfield and max_order_int >= 1:
                min_order_candidate = (
                    max_order_int - 1
                    if self.mixed_order_min_order is None
                    else int(self.mixed_order_min_order)
                )
                min_order_candidate = max(0, min(min_order_candidate, max_order_int))
                p_gears_for_downward, far_pairs_by_gear = (
                    _bucket_far_pairs_by_level_split(
                        interactions=interactions,
                        src_far=src_far,
                        tgt_far=tgt_far,
                        max_order=max_order_int,
                        min_order=min_order_candidate,
                    )
                )
            else:
                p_gears_for_downward = (max_order_int,)
                far_pairs_by_gear = ((src_far, tgt_far),)
            self._recent_far_pairs_by_gear_counts = tuple(
                int(bucket_src.shape[0]) for bucket_src, _ in far_pairs_by_gear
            )
        else:
            self._recent_far_pairs_by_gear_counts = tuple()

        if (
            runtime_m2l_chunk_size is None
            and bool(self.autotune_m2l_chunk)
            and self.expansion_basis == "solidfmm"
            and jax.default_backend() == "gpu"
            and far_pairs_by_gear is not None
            and len(far_pairs_by_gear) > 0
        ):
            tune_idx = 0
            tune_pair_count = -1
            for idx, (src_bucket, _) in enumerate(far_pairs_by_gear):
                count_i = int(src_bucket.shape[0])
                if count_i > tune_pair_count:
                    tune_idx = idx
                    tune_pair_count = count_i
            if tune_pair_count > 0:
                tune_src, tune_tgt = far_pairs_by_gear[tune_idx]
                tune_order = int(
                    tree_artifacts.upward.multipoles.order
                    if tune_idx >= len(p_gears_for_downward)
                    else p_gears_for_downward[tune_idx]
                )
                tuned_chunk = self._autotune_runtime_m2l_chunk_size(
                    upward=tree_artifacts.upward,
                    src=tune_src,
                    tgt=tune_tgt,
                    order=tune_order,
                    pair_count=tune_pair_count,
                )
                if tuned_chunk is not None:
                    runtime_m2l_chunk_size = int(tuned_chunk)

        interactions_for_downward: Optional[NodeInteractionList] = interactions
        if (
            self.streamed_far_pairs
            and far_pairs_coo is not None
            and not bool(self.retain_interactions)
        ):
            interactions_for_downward = None

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
        if not bool(self.retain_interactions):
            downward = downward._replace(
                interactions=_empty_interaction_storage_like(interactions)
            )
        interactions_out: Optional[NodeInteractionList]
        if bool(self.retain_interactions):
            interactions_out = interactions
        else:
            interactions_out = None
        return _PrepareStateDualDownwardArtifacts(
            interactions=interactions_out,
            neighbor_list=neighbor_list,
            traversal_result=(
                traversal_result if self.retain_traversal_result else None
            ),
            downward=downward,
            cache_entry=cache_entry,
        )

    def _prepare_state_nearfield_artifacts(
        self,
        *,
        tree: Tree,
        neighbor_list: NodeNeighborList,
        leaf_cap: int,
        num_particles: int,
        cache_entry: Optional[_InteractionCacheEntry],
        allow_stateful_cache: bool,
    ) -> NearfieldPrecomputeArtifacts:
        """Build/reuse near-field precompute artifacts for prepare_state."""
        nearfield_mode_resolved = self._resolve_nearfield_mode(
            num_particles=num_particles
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
        ):
            return nearfield_from_cache(cache_entry)

        nearfield_artifacts = self._prepare_nearfield_precompute_artifacts(
            tree=tree,
            neighbor_list=neighbor_list,
            leaf_cap=leaf_cap,
            num_particles=num_particles,
            nearfield_mode=nearfield_mode_resolved,
            nearfield_edge_chunk_size=nearfield_edge_chunk_size_resolved,
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
        tree: Tree,
        neighbor_list: NodeNeighborList,
        leaf_cap: int,
        num_particles: int,
        nearfield_mode: Optional[str] = None,
        nearfield_edge_chunk_size: Optional[int] = None,
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

        should_precompute_scatter = self._should_precompute_nearfield_scatter_schedules(
            num_particles=int(num_particles)
        )
        if resolved_nearfield_mode != "bucketed" or not should_precompute_scatter:
            # Keep prepared-state nearfield representation compact; derive leaf-edge
            # pair vectors on demand during evaluation.
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
            tree=tree,
            neighbor_list=neighbor_list,
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
        ):
            (
                nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices,
            ) = self._prepare_bucketed_scatter_schedules_safe(
                tree=tree,
                neighbor_list=neighbor_list,
                target_leaf_ids=nearfield_target_leaf_ids,
                valid_pairs=nearfield_valid_pairs,
                leaf_cap=int(leaf_cap),
                edge_chunk_size=resolved_nearfield_edge_chunk_size,
            )

        return NearfieldPrecomputeArtifacts(
            target_leaf_ids=None,
            source_leaf_ids=None,
            valid_pairs=None,
            chunk_sort_indices=nearfield_chunk_sort_indices,
            chunk_group_ids=nearfield_chunk_group_ids,
            chunk_unique_indices=nearfield_chunk_unique_indices,
        )

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
        tree: Tree,
        neighbor_list: NodeNeighborList,
    ) -> tuple[Optional[Array], Optional[Array], Optional[Array]]:
        """Best-effort leaf neighbor pair generation."""
        try:
            return prepare_leaf_neighbor_pairs(
                jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
                jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE),
                jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE),
                jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE),
                # Keep edge order aligned with neighbor_list so source leaf ids can
                # be derived on demand without storing a second index vector.
                sort_by_source=False,
            )
        except Exception:
            return None, None, None

    def _prepare_bucketed_scatter_schedules_safe(
        self,
        *,
        tree: Tree,
        neighbor_list: NodeNeighborList,
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
            schedule_item_cap = (
                _NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP_GPU
                if jax.default_backend() == "gpu"
                else _NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP
            )
            if schedule_items > int(schedule_item_cap):
                return None, None, None
            return prepare_bucketed_scatter_schedules(
                jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
                jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE),
                target_leaf_ids,
                valid_pairs,
                max_leaf_size=int(leaf_cap),
                edge_chunk_size=chunk,
            )
        except Exception:
            return None, None, None

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
        DualTreeWalkResult,
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

    def _resolve_nearfield_mode(self, *, num_particles: int) -> str:
        """Resolve near-field execution mode from configured policy."""
        mode = str(self.nearfield_mode).strip().lower()
        if mode != "auto":
            return mode
        backend = jax.default_backend()
        large_cpu = (
            backend == "cpu"
            and int(num_particles) >= _NEARFIELD_BUCKETED_CPU_PARTICLE_THRESHOLD
        )
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
        """Resolve near-field edge chunk size with large-N CPU auto policy."""
        base_chunk = int(self.nearfield_edge_chunk_size)
        if base_chunk <= 0:
            raise ValueError("nearfield_edge_chunk_size must be positive")
        mode = str(self.nearfield_mode).strip().lower()
        if mode != "auto" or str(nearfield_mode).strip().lower() != "bucketed":
            return base_chunk
        if jax.default_backend() != "cpu":
            return base_chunk

        n = int(num_particles)
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
    ) -> TreeUpwardData:
        """Bundle geometry, raw moments, and packed expansions for a tree."""

        if self.expansion_basis == "solidfmm":
            complex_upward = prepare_solidfmm_complex_upward_sweep(
                tree,
                positions_sorted,
                masses_sorted,
                max_order=max_order,
                center_mode=center_mode,
                explicit_centers=explicit_centers,
                max_leaf_size=max_leaf_size,
                rotation=self.complex_rotation,
            )

            multipoles = NodeMultipoleData(
                order=int(complex_upward.multipoles.order),
                centers=complex_upward.multipoles.centers,
                moments=None,  # type: ignore[arg-type]
                packed=complex_upward.multipoles.packed,
                component_matrix=complex_upward.multipoles.packed,
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
    ) -> Union[Array, Tuple[Array, Array]]:
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
    ) -> FMMPreparedState:
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

        positions_arr, masses_arr, input_dtype = self._prepare_state_input_arrays(
            positions,
            masses,
        )
        allow_stateful_cache = not _contains_tracer((positions_arr, masses_arr))

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
        mac_type_val = "dehnen" if self.mac_type == "dehnen_error" else self.mac_type

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
        use_paper_force_scale = self.adaptive_order or (
            self.adaptive_error_model == "dehnen_paper"
        )
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
                elif (
                    self._uses_dehnen_paper_error_model()
                    and not self._in_force_scale_prepass
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
                if (
                    self.mac_force_scale_mode == "paper"
                    and self._uses_dehnen_paper_error_model()
                ):
                    low_order = 1 if int(max_order) >= 1 else 0
                if (
                    self.mac_force_scale_mode == "paper"
                    and self._uses_dehnen_paper_error_model()
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

        nearfield_artifacts = self._prepare_state_nearfield_artifacts(
            tree=tree_artifacts.tree,
            neighbor_list=dual_downward_artifacts.neighbor_list,
            leaf_cap=tree_artifacts.leaf_cap,
            num_particles=int(positions_arr.shape[0]),
            cache_entry=dual_downward_artifacts.cache_entry,
            allow_stateful_cache=allow_stateful_cache,
        )

        return FMMPreparedState(
            tree=tree_artifacts.tree,
            upward=tree_artifacts.upward,
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
            nearfield_target_leaf_ids=nearfield_artifacts.target_leaf_ids,
            nearfield_source_leaf_ids=nearfield_artifacts.source_leaf_ids,
            nearfield_valid_pairs=nearfield_artifacts.valid_pairs,
            nearfield_chunk_sort_indices=nearfield_artifacts.chunk_sort_indices,
            nearfield_chunk_group_ids=nearfield_artifacts.chunk_group_ids,
            nearfield_chunk_unique_indices=nearfield_artifacts.chunk_unique_indices,
            force_scale_nodes=force_scale_nodes,
        )

    @jaxtyped(typechecker=beartype)
    def evaluate_prepared_state(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        *,
        target_indices: Optional[Array] = None,
        return_potential: bool = False,
        jit_traversal: bool = True,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Evaluate accelerations/potentials for all particles or targets."""

        resolved_target_indices = self._resolve_target_indices(
            target_indices=target_indices,
            num_particles=int(state.inverse_permutation.shape[0]),
        )
        tracing_targets = isinstance(
            state.positions_sorted, jax.core.Tracer
        ) or isinstance(resolved_target_indices, jax.core.Tracer)
        if resolved_target_indices is None or tracing_targets:
            evaluation = _evaluate_prepared_tree(
                fmm=self,
                tree=state.tree,
                positions_sorted=state.positions_sorted,
                masses_sorted=state.masses_sorted,
                downward=state.downward,
                neighbor_list=state.neighbor_list,
                nearfield_target_leaf_ids=state.nearfield_target_leaf_ids,
                nearfield_source_leaf_ids=state.nearfield_source_leaf_ids,
                nearfield_valid_pairs=state.nearfield_valid_pairs,
                nearfield_chunk_sort_indices=state.nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids=state.nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices=state.nearfield_chunk_unique_indices,
                max_leaf_size=state.max_leaf_size,
                return_potential=return_potential,
                jit_traversal=jit_traversal,
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
                target_sorted_indices=target_sorted_indices,
                return_potential=return_potential,
            )

        if jnp.issubdtype(state.input_dtype, jnp.floating):
            output_dtype = state.input_dtype
        else:
            output_dtype = state.working_dtype

        if return_potential:
            acc_sorted, pot_sorted = evaluation
            if resolved_target_indices is None:
                accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation]
                potentials = jnp.asarray(pot_sorted)[state.inverse_permutation]
            elif tracing_targets:
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
    def evaluate_tree(
        self: "FastMultipoleMethod",
        tree: Tree,
        positions_sorted: Array,
        masses_sorted: Array,
        locals_or_downward: Union[LocalExpansionData, TreeDownwardData],
        neighbor_list: NodeNeighborList,
        *,
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
        )

        far_grad, far_potential_pre = _evaluate_local_expansions_for_particles(
            locals_data,
            positions,
            leaf_nodes=leaf_nodes,
            node_ranges=node_ranges,
            max_leaf_size=resolved_max_leaf,
            order=order,
            expansion_basis=self.expansion_basis,
            return_potential=return_potential,
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

        return _evaluate_tree_compiled_impl(
            tree,
            setup.positions,
            setup.masses,
            setup.locals_data,
            neighbor_list,
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
    num_internal = int(tree.num_internal_nodes)
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


def _prepare_tree_evaluation_inputs(
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    locals_or_downward: Union[LocalExpansionData, TreeDownwardData],
    neighbor_list: NodeNeighborList,
    *,
    max_leaf_size: Optional[int],
    return_potential: bool,
) -> _TreeEvaluationSetup:
    """Validate and normalize tree-evaluation inputs for eager/JIT paths."""
    locals_data = (
        locals_or_downward.locals
        if isinstance(locals_or_downward, TreeDownwardData)
        else locals_or_downward
    )

    if locals_data.centers.shape[0] != tree.node_ranges.shape[0]:
        raise ValueError("local expansions must align with tree nodes")
    if locals_data.coefficients.shape[0] != tree.node_ranges.shape[0]:
        raise ValueError("local expansions must align with tree nodes")

    positions = jnp.asarray(positions_sorted)
    masses = jnp.asarray(masses_sorted)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)

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
            locals_data,
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
        locals_data,
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
) -> tuple[Array, Array]:
    """Resolve rotation blocks for all grouped classes with cache reuse."""
    num_classes = int(class_deltas.shape[0])
    max_m = 2 * int(order) + 1
    empty_shape = (0, int(order) + 1, max_m, max_m)
    if num_classes == 0:
        empty = jnp.zeros(empty_shape, dtype=dtype)
        return empty, empty

    cache_key = _grouped_operator_cache_key(
        order=order,
        rotation=rotation,
        dtype=dtype,
        class_keys=class_keys,
        class_deltas=class_deltas,
    )
    if cache_key is not None:
        cached = _grouped_operator_cache_get(cache_key)
        if cached is not None:
            return cached

    deltas = jnp.asarray(class_deltas)
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
    elif rotation == "cached":
        blocks_to = complex_rotation_blocks_to_z_batch(
            deltas,
            order=order,
            basis="multipole",
            dtype=dtype,
        )
        blocks_from = complex_rotation_blocks_from_z_batch(
            deltas,
            order=order,
            basis="local",
            dtype=dtype,
        )
    else:
        raise ValueError(
            "grouped operator cache currently supports rotation='cached' or 'solidfmm'"
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
    sort_idx = jnp.argsort(tgt_chunk)
    tgt_sorted = tgt_chunk[sort_idx]
    contribs_sorted = contribs[sort_idx]
    valid_sorted = valid[sort_idx]

    contribs_sorted = jnp.where(valid_sorted[:, None], contribs_sorted, 0)
    new_group = jnp.concatenate(
        (
            jnp.asarray([True], dtype=bool),
            tgt_sorted[1:] != tgt_sorted[:-1],
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
    static_argnames=("order", "total_nodes", "chunk_size"),
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
) -> Array:
    """Accumulate grouped solidfmm M2L contributions via chunked scan."""
    pair_count = src_sorted.shape[0]
    starts = jnp.arange(0, pair_count, chunk_size, dtype=INDEX_DTYPE)
    local_accum0 = jnp.zeros_like(locals_coeffs)

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

        contribs = _m2l_complex_batch_cached_kernel(
            src_mult,
            deltas,
            blocks_to,
            blocks_from,
            order=order,
        ).astype(locals_coeffs.dtype)
        local_accum = _chunk_segment_scatter_add(
            local_accum,
            contribs,
            tgt_chunk,
            valid,
            chunk_size=chunk_size,
        )
        return local_accum, None

    local_accum, _ = jax.lax.scan(body, local_accum0, starts)
    return locals_coeffs + local_accum


@partial(jax.jit, static_argnames=("order", "total_nodes"), donate_argnums=(0,))
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
) -> Array:
    """Accumulate grouped solidfmm M2L contributions in one full batch."""
    src_mult = multip_packed[src_sorted]
    deltas = centers[tgt_sorted] - centers[src_sorted]
    blocks_to = blocks_to_classes[class_ids_sorted]
    blocks_from = blocks_from_classes[class_ids_sorted]
    contribs = _m2l_complex_batch_cached_kernel(
        src_mult,
        deltas,
        blocks_to,
        blocks_from,
        order=order,
    ).astype(locals_coeffs.dtype)
    return locals_coeffs + jax.ops.segment_sum(contribs, tgt_sorted, total_nodes)


def _build_grouped_class_segments(
    grouped: GroupedInteractionBuffers,
    *,
    chunk_size: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Build class-major fixed-width segments for chunked class execution."""
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
    class_targets = np.asarray(jax.device_get(grouped.class_targets), dtype=np.int64)
    if class_offsets.size <= 1:
        empty = jnp.zeros((0,), dtype=INDEX_DTYPE)
        empty_matrix = jnp.zeros((0, int(chunk_size)), dtype=INDEX_DTYPE)
        result = (empty, empty, empty, empty_matrix, empty_matrix, empty_matrix)
        if cache_key is not None:
            _grouped_segment_cache_put(cache_key, result)
        return result

    starts: list[int] = []
    lengths: list[int] = []
    class_ids: list[int] = []
    sort_permutation: list[np.ndarray] = []
    group_ids: list[np.ndarray] = []
    unique_targets: list[np.ndarray] = []

    offsets = np.arange(int(chunk_size), dtype=np.int64)
    for class_idx in range(class_offsets.shape[0] - 1):
        start = int(class_offsets[class_idx])
        end = int(class_offsets[class_idx + 1])
        while start < end:
            seg_len = min(int(chunk_size), end - start)
            starts.append(start)
            lengths.append(seg_len)
            class_ids.append(class_idx)

            idx = start + offsets
            valid = offsets < seg_len
            safe_idx = np.where(valid, idx, 0)
            tgt_chunk = class_targets[safe_idx]

            perm = np.argsort(tgt_chunk, kind="stable")
            tgt_sorted = tgt_chunk[perm]
            grp = np.zeros((int(chunk_size),), dtype=np.int64)
            if int(chunk_size) > 1:
                grp[1:] = np.cumsum(
                    (tgt_sorted[1:] != tgt_sorted[:-1]).astype(np.int64)
                )

            unique = np.zeros((int(chunk_size),), dtype=np.int64)
            unique[grp] = tgt_sorted

            sort_permutation.append(perm.astype(np.int64))
            group_ids.append(grp)
            unique_targets.append(unique)
            start += seg_len

    if len(sort_permutation) == 0:
        empty_matrix = jnp.zeros((0, int(chunk_size)), dtype=INDEX_DTYPE)
        result = (
            jnp.asarray(starts, dtype=INDEX_DTYPE),
            jnp.asarray(lengths, dtype=INDEX_DTYPE),
            jnp.asarray(class_ids, dtype=INDEX_DTYPE),
            empty_matrix,
            empty_matrix,
            empty_matrix,
        )
        if cache_key is not None:
            _grouped_segment_cache_put(cache_key, result)
        return result

    result = (
        jnp.asarray(starts, dtype=INDEX_DTYPE),
        jnp.asarray(lengths, dtype=INDEX_DTYPE),
        jnp.asarray(class_ids, dtype=INDEX_DTYPE),
        jnp.asarray(np.stack(sort_permutation, axis=0), dtype=INDEX_DTYPE),
        jnp.asarray(np.stack(group_ids, axis=0), dtype=INDEX_DTYPE),
        jnp.asarray(np.stack(unique_targets, axis=0), dtype=INDEX_DTYPE),
    )
    if cache_key is not None:
        _grouped_segment_cache_put(cache_key, result)
    return result


@partial(
    jax.jit,
    static_argnames=("order", "total_nodes", "chunk_size"),
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
    segment_sort_permutation: Array,
    segment_group_ids: Array,
    segment_unique_targets: Array,
    blocks_to_classes: Array,
    blocks_from_classes: Array,
    *,
    order: int,
    total_nodes: int,
    chunk_size: int,
) -> Array:
    """Accumulate class-major grouped M2L contributions via chunked scan."""
    num_segments = segment_starts.shape[0]
    if num_segments == 0:
        return locals_coeffs

    offsets = jnp.arange(chunk_size, dtype=INDEX_DTYPE)
    local_accum0 = jnp.zeros_like(locals_coeffs)

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

        contribs = _m2l_complex_batch_cached_kernel(
            src_mult,
            deltas,
            blocks_to,
            blocks_from,
            order=order,
        ).astype(locals_coeffs.dtype)
        contribs = jnp.where(valid[:, None], contribs, 0)
        sort_idx = segment_sort_permutation[seg_idx]
        group_ids = segment_group_ids[seg_idx]
        unique_targets = segment_unique_targets[seg_idx]
        contribs_sorted = contribs[sort_idx]
        reduced = jax.ops.segment_sum(contribs_sorted, group_ids, chunk_size)
        return local_accum.at[unique_targets].add(reduced), None

    local_accum, _ = jax.lax.scan(
        body,
        local_accum0,
        jnp.arange(num_segments, dtype=INDEX_DTYPE),
    )
    return locals_coeffs + local_accum


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
) -> Array:
    """Class-major grouped accumulation without per-pair operator gathers."""

    if rotation not in ("cached", "solidfmm"):
        src = grouped.class_sources
        tgt = grouped.class_targets
        return _accumulate_solidfmm_m2l_fullbatch(
            locals_coeffs,
            multip_packed,
            centers,
            src,
            tgt,
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
    )
    if (
        grouped_segment_starts is None
        or grouped_segment_lengths is None
        or grouped_segment_class_ids is None
        or grouped_segment_sort_permutation is None
        or grouped_segment_group_ids is None
        or grouped_segment_unique_targets is None
    ):
        (
            segment_starts,
            segment_lengths,
            segment_class_ids,
            segment_sort_permutation,
            segment_group_ids,
            segment_unique_targets,
        ) = _build_grouped_class_segments(
            grouped,
            chunk_size=int(chunk_size),
        )
    else:
        segment_starts = jnp.asarray(grouped_segment_starts, dtype=INDEX_DTYPE)
        segment_lengths = jnp.asarray(grouped_segment_lengths, dtype=INDEX_DTYPE)
        segment_class_ids = jnp.asarray(grouped_segment_class_ids, dtype=INDEX_DTYPE)
        segment_sort_permutation = jnp.asarray(
            grouped_segment_sort_permutation,
            dtype=INDEX_DTYPE,
        )
        segment_group_ids = jnp.asarray(grouped_segment_group_ids, dtype=INDEX_DTYPE)
        segment_unique_targets = jnp.asarray(
            grouped_segment_unique_targets,
            dtype=INDEX_DTYPE,
        )
    return _accumulate_solidfmm_m2l_class_major_chunked_scan(
        locals_coeffs,
        multip_packed,
        centers,
        jnp.asarray(grouped.class_sources, dtype=INDEX_DTYPE),
        jnp.asarray(grouped.class_targets, dtype=INDEX_DTYPE),
        segment_starts,
        segment_lengths,
        segment_class_ids,
        segment_sort_permutation,
        segment_group_ids,
        segment_unique_targets,
        blocks_to_classes,
        blocks_from_classes,
        order=order,
        total_nodes=total_nodes,
        chunk_size=int(chunk_size),
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
) -> Array:
    """Grouped M2L accumulation using cached class blocks and pair chunking."""

    if rotation not in ("cached", "solidfmm"):
        # Keep existing sparse path semantics for other conventions.
        src = grouped.class_sources
        tgt = grouped.class_targets
        return _accumulate_solidfmm_m2l_fullbatch(
            locals_coeffs,
            multip_packed,
            centers,
            src,
            tgt,
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
    *,
    order: int,
    rotation: str,
    total_nodes: int,
) -> Array:
    """Accumulate solidfmm M2L contributions in one full interaction batch."""
    src_mult = multip_packed[src]
    deltas = centers[tgt] - centers[src]

    if rotation == "cached":
        blocks_to_z = complex_rotation_blocks_to_z_batch(
            deltas,
            order=order,
            basis="multipole",
            dtype=src_mult.dtype,
        )
        blocks_from_z = complex_rotation_blocks_from_z_batch(
            deltas,
            order=order,
            basis="local",
            dtype=src_mult.dtype,
        )
        contribs = _m2l_complex_batch_cached_kernel(
            src_mult,
            deltas,
            blocks_to_z,
            blocks_from_z,
            order=order,
        )
    else:
        contribs = _m2l_complex_batch_kernel(
            src_mult,
            deltas,
            order=order,
            rotation=rotation,
        )
    contribs = contribs.astype(locals_coeffs.dtype)
    return locals_coeffs + jax.ops.segment_sum(contribs, tgt, total_nodes)


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
    *,
    order: int,
    rotation: str,
    total_nodes: int,
    chunk_size: int,
) -> Array:
    """Accumulate solidfmm M2L contributions with chunked scan reduction."""
    pair_count = src.shape[0]
    starts = jnp.arange(0, pair_count, chunk_size, dtype=INDEX_DTYPE)
    local_accum0 = jnp.zeros_like(locals_coeffs)

    def body(local_accum: Array, start_idx: Array) -> tuple[Array, None]:
        offset = jnp.arange(chunk_size, dtype=INDEX_DTYPE)
        idx = start_idx + offset
        valid = idx < pair_count
        safe_idx = jnp.where(valid, idx, 0)

        src_chunk = src[safe_idx]
        tgt_chunk = tgt[safe_idx]
        src_mult = multip_packed[src_chunk]
        deltas = centers[tgt_chunk] - centers[src_chunk]

        if rotation == "cached":
            blocks_to_z = complex_rotation_blocks_to_z_batch(
                deltas,
                order=order,
                basis="multipole",
                dtype=src_mult.dtype,
            )
            blocks_from_z = complex_rotation_blocks_from_z_batch(
                deltas,
                order=order,
                basis="local",
                dtype=src_mult.dtype,
            )
            contribs = _m2l_complex_batch_cached_kernel(
                src_mult,
                deltas,
                blocks_to_z,
                blocks_from_z,
                order=order,
            )
        else:
            contribs = _m2l_complex_batch_kernel(
                src_mult,
                deltas,
                order=order,
                rotation=rotation,
            )

        contribs = contribs.astype(locals_coeffs.dtype)
        local_accum = _chunk_segment_scatter_add(
            local_accum,
            contribs,
            tgt_chunk,
            valid,
            chunk_size=chunk_size,
        )
        return local_accum, None

    local_accum, _ = jax.lax.scan(body, local_accum0, starts)
    return locals_coeffs + local_accum


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
    static_argnames=("order", "m2l_impl", "total_nodes"),
    donate_argnums=(0,),
)
def _accumulate_real_m2l_fullbatch(
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
    """Accumulate real-basis M2L contributions in one full interaction batch."""
    src_mult = multip_packed_real[src]
    deltas = centers[tgt] - centers[src]
    contribs = _m2l_real_batch_kernel(
        src_mult,
        deltas,
        order=order,
        m2l_impl=m2l_impl,
    ).astype(locals_coeffs.dtype)
    return locals_coeffs + jax.ops.segment_sum(contribs, tgt, total_nodes)


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
    *,
    order: int,
    m2l_impl: str,
    total_nodes: int,
    chunk_size: int,
) -> Array:
    """Accumulate real-basis M2L contributions with chunked scan reduction."""
    pair_count = src.shape[0]
    starts = jnp.arange(0, pair_count, chunk_size, dtype=INDEX_DTYPE)
    local_accum0 = jnp.zeros_like(locals_coeffs)

    def body(local_accum: Array, start_idx: Array) -> tuple[Array, None]:
        offset = jnp.arange(chunk_size, dtype=INDEX_DTYPE)
        idx = start_idx + offset
        valid = idx < pair_count
        safe_idx = jnp.where(valid, idx, 0)
        src_chunk = src[safe_idx]
        tgt_chunk = tgt[safe_idx]
        src_mult = multip_packed_real[src_chunk]
        deltas = centers[tgt_chunk] - centers[src_chunk]
        contribs = _m2l_real_batch_kernel(
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

    local_accum, _ = jax.lax.scan(body, local_accum0, starts)
    return locals_coeffs + local_accum


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
    local_accum0 = jnp.zeros_like(locals_coeffs)

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

    local_accum, _ = jax.lax.scan(body, local_accum0, starts)
    return locals_coeffs + local_accum


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
) -> TreeDownwardData:
    """Prepare M2L accumulation for solidfmm-style complex or real locals."""

    if interactions is None and far_pairs_coo is None:
        interactions = build_well_separated_interactions(
            tree,
            upward.geometry,
            theta=theta,
            mac_type=mac_type,
            dehnen_radius_scale=dehnen_radius_scale,
            traversal_config=traversal_config,
            retry_logger=retry_logger,
        )
    if interactions is None:
        interactions = _empty_interaction_storage_for_tree(tree)

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

    if far_pairs_coo is not None:
        src = jnp.asarray(far_pairs_coo.sources, dtype=INDEX_DTYPE)
        tgt = jnp.asarray(far_pairs_coo.targets, dtype=INDEX_DTYPE)
    else:
        src = jnp.asarray(interactions.sources, dtype=INDEX_DTYPE)
        tgt = jnp.asarray(interactions.targets, dtype=INDEX_DTYPE)

    pair_count = int(src.shape[0])

    multip_packed = jnp.asarray(upward.multipoles.packed)

    rotation_mode = str(complex_rotation).strip().lower()
    if basis_mode_norm == "complex":
        if rotation_mode not in ("bdz", "cached", "wigner", "solidfmm"):
            raise ValueError(
                "complex_rotation must be 'bdz', 'cached', 'wigner', or 'solidfmm'"
            )
        multip_packed_kernel = multip_packed.astype(dtype)
    else:
        multip_packed_kernel = complex_to_real_coeffs(multip_packed, order=p).astype(
            dtype
        )

    chunk_size = 4096 if m2l_chunk_size is None else int(m2l_chunk_size)
    if chunk_size <= 0:
        raise ValueError("m2l_chunk_size must be positive")

    if basis_mode_norm == "complex" and grouped_interactions and not adaptive_order:
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
                locals_coeffs,
                multip_packed_kernel,
                centers,
                grouped,
                grouped_segment_starts=grouped_segment_starts,
                grouped_segment_lengths=grouped_segment_lengths,
                grouped_segment_class_ids=grouped_segment_class_ids,
                grouped_segment_sort_permutation=grouped_segment_sort_permutation,
                grouped_segment_group_ids=grouped_segment_group_ids,
                grouped_segment_unique_targets=grouped_segment_unique_targets,
                order=p,
                rotation=rotation_mode,
                total_nodes=total_nodes,
                chunk_size=chunk_size,
            )
        else:
            locals_updated = _accumulate_solidfmm_m2l_grouped(
                locals_coeffs,
                multip_packed_kernel,
                centers,
                grouped,
                order=p,
                rotation=rotation_mode,
                total_nodes=total_nodes,
                chunk_size=chunk_size,
            )
    else:
        if adaptive_order:
            if len(p_gears) == 0:
                raise ValueError("adaptive_order=True requires non-empty p_gears")
            p_gears_int = tuple(int(v) for v in p_gears)
            gear_pairs = far_pairs_by_gear
            if gear_pairs is None:
                buckets: list[tuple[Array, Array]] = []
                for _ in p_gears_int:
                    buckets.append(
                        (
                            jnp.zeros((0,), dtype=INDEX_DTYPE),
                            jnp.zeros((0,), dtype=INDEX_DTYPE),
                        )
                    )
                buckets[-1] = (src, tgt)
                gear_pairs = tuple(buckets)
            if len(gear_pairs) != len(p_gears_int):
                raise ValueError("far_pairs_by_gear must align with p_gears")

            pairs_by_order: dict[int, list[tuple[Array, Array]]] = {}
            total_pairs_for_adaptive = 0
            for p_gear, pair_bucket in zip(p_gears_int, gear_pairs):
                if int(p_gear) < 0 or int(p_gear) > int(p):
                    raise ValueError(
                        "p_gears entries must satisfy 0 <= p_gear <= p_max"
                    )
                total_pairs_for_adaptive += int(pair_bucket[0].shape[0])
                pairs_by_order.setdefault(int(p_gear), []).append(pair_bucket)
            if total_pairs_for_adaptive == 0:
                empty_locals = LocalExpansionData(
                    order=p,
                    centers=centers,
                    coefficients=locals_coeffs,
                )
                return TreeDownwardData(
                    interactions=interactions,
                    locals=empty_locals,
                )

            locals_updated = locals_coeffs
            for p_gear, order_pairs in sorted(pairs_by_order.items()):
                src_parts: list[Array] = []
                tgt_parts: list[Array] = []
                for src_g, tgt_g in order_pairs:
                    if int(src_g.shape[0]) == 0:
                        continue
                    src_parts.append(jnp.asarray(src_g, dtype=INDEX_DTYPE))
                    tgt_parts.append(jnp.asarray(tgt_g, dtype=INDEX_DTYPE))
                if len(src_parts) == 0:
                    continue
                if len(src_parts) == 1:
                    src_g = src_parts[0]
                    tgt_g = tgt_parts[0]
                else:
                    src_g = jnp.concatenate(src_parts, axis=0)
                    tgt_g = jnp.concatenate(tgt_parts, axis=0)

                pair_count_g = int(src_g.shape[0])
                if pair_count_g == 0:
                    continue
                coeff_g = sh_size(int(p_gear))
                locals_slice = jnp.array(locals_updated[:, :coeff_g], copy=True)
                multip_slice = multip_packed_kernel[:, :coeff_g]
                if basis_mode_norm == "complex":
                    if pair_count_g <= min(chunk_size, _M2L_FULLBATCH_MAX_PAIRS):
                        locals_slice = _accumulate_solidfmm_m2l_fullbatch(
                            locals_slice,
                            multip_slice,
                            centers,
                            src_g,
                            tgt_g,
                            order=int(p_gear),
                            rotation=rotation_mode,
                            total_nodes=total_nodes,
                        )
                    else:
                        locals_slice = _accumulate_solidfmm_m2l_chunked_scan(
                            locals_slice,
                            multip_slice,
                            centers,
                            src_g,
                            tgt_g,
                            order=int(p_gear),
                            rotation=rotation_mode,
                            total_nodes=total_nodes,
                            chunk_size=chunk_size,
                        )
                else:
                    if pair_count_g <= min(chunk_size, _M2L_FULLBATCH_MAX_PAIRS):
                        if use_pallas:
                            locals_slice = _accumulate_real_m2l_fullbatch_pallas(
                                locals_slice,
                                multip_slice,
                                centers,
                                src_g,
                                tgt_g,
                                order=int(p_gear),
                                m2l_impl=(
                                    "rot_scale" if m2l_impl is None else str(m2l_impl)
                                ),
                                total_nodes=total_nodes,
                            )
                        else:
                            locals_slice = _accumulate_real_m2l_fullbatch(
                                locals_slice,
                                multip_slice,
                                centers,
                                src_g,
                                tgt_g,
                                order=int(p_gear),
                                m2l_impl=(
                                    "rot_scale" if m2l_impl is None else str(m2l_impl)
                                ),
                                total_nodes=total_nodes,
                            )
                    else:
                        if use_pallas:
                            locals_slice = _accumulate_real_m2l_chunked_scan_pallas(
                                locals_slice,
                                multip_slice,
                                centers,
                                src_g,
                                tgt_g,
                                order=int(p_gear),
                                m2l_impl=(
                                    "rot_scale" if m2l_impl is None else str(m2l_impl)
                                ),
                                total_nodes=total_nodes,
                                chunk_size=chunk_size,
                            )
                        else:
                            locals_slice = _accumulate_real_m2l_chunked_scan(
                                locals_slice,
                                multip_slice,
                                centers,
                                src_g,
                                tgt_g,
                                order=int(p_gear),
                                m2l_impl=(
                                    "rot_scale" if m2l_impl is None else str(m2l_impl)
                                ),
                                total_nodes=total_nodes,
                                chunk_size=chunk_size,
                            )
                locals_tail = locals_updated[:, coeff_g:]
                locals_updated = jnp.concatenate(
                    (locals_slice, locals_tail),
                    axis=1,
                )
        elif pair_count == 0:
            empty_locals = LocalExpansionData(
                order=p,
                centers=centers,
                coefficients=locals_coeffs,
            )
            return TreeDownwardData(
                interactions=interactions,
                locals=empty_locals,
            )
        elif basis_mode_norm == "complex" and pair_count <= min(
            chunk_size, _M2L_FULLBATCH_MAX_PAIRS
        ):
            locals_updated = _accumulate_solidfmm_m2l_fullbatch(
                locals_coeffs,
                multip_packed_kernel,
                centers,
                src,
                tgt,
                order=p,
                rotation=rotation_mode,
                total_nodes=total_nodes,
            )
        elif basis_mode_norm == "complex":
            locals_updated = _accumulate_solidfmm_m2l_chunked_scan(
                locals_coeffs,
                multip_packed_kernel,
                centers,
                src,
                tgt,
                order=p,
                rotation=rotation_mode,
                total_nodes=total_nodes,
                chunk_size=chunk_size,
            )
        elif pair_count <= min(chunk_size, _M2L_FULLBATCH_MAX_PAIRS):
            if use_pallas:
                locals_updated = _accumulate_real_m2l_fullbatch_pallas(
                    locals_coeffs,
                    multip_packed_kernel,
                    centers,
                    src,
                    tgt,
                    order=p,
                    m2l_impl=("rot_scale" if m2l_impl is None else str(m2l_impl)),
                    total_nodes=total_nodes,
                )
            else:
                locals_updated = _accumulate_real_m2l_fullbatch(
                    locals_coeffs,
                    multip_packed_kernel,
                    centers,
                    src,
                    tgt,
                    order=p,
                    m2l_impl=("rot_scale" if m2l_impl is None else str(m2l_impl)),
                    total_nodes=total_nodes,
                )
        else:
            if use_pallas:
                locals_updated = _accumulate_real_m2l_chunked_scan_pallas(
                    locals_coeffs,
                    multip_packed_kernel,
                    centers,
                    src,
                    tgt,
                    order=p,
                    m2l_impl=("rot_scale" if m2l_impl is None else str(m2l_impl)),
                    total_nodes=total_nodes,
                    chunk_size=chunk_size,
                )
            else:
                locals_updated = _accumulate_real_m2l_chunked_scan(
                    locals_coeffs,
                    multip_packed_kernel,
                    centers,
                    src,
                    tgt,
                    order=p,
                    m2l_impl=("rot_scale" if m2l_impl is None else str(m2l_impl)),
                    total_nodes=total_nodes,
                    chunk_size=chunk_size,
                )

    if basis_mode_norm == "complex":
        locals_updated = enforce_conjugate_symmetry_batch(locals_updated, order=p)

    if l2l_chunk_size is not None and int(l2l_chunk_size) <= 0:
        raise ValueError("l2l_chunk_size must be positive")

    num_internal_nodes = int(tree.num_internal_nodes)
    if num_internal_nodes > 0:
        left_child = jnp.asarray(
            tree.left_child[:num_internal_nodes], dtype=INDEX_DTYPE
        )
        right_child = jnp.asarray(
            tree.right_child[:num_internal_nodes],
            dtype=INDEX_DTYPE,
        )
        if basis_mode_norm == "complex":
            locals_updated = _propagate_solidfmm_locals_to_children(
                locals_updated,
                centers,
                left_child,
                right_child,
                order=p,
                rotation=rotation_mode,
                total_nodes=total_nodes,
            )
        else:
            locals_updated = _propagate_real_locals_to_children(
                locals_updated,
                centers,
                left_child,
                right_child,
                order=p,
                total_nodes=total_nodes,
            )

    locals_after = LocalExpansionData(
        order=p,
        centers=centers,
        coefficients=locals_updated,
    )

    if basis_mode_norm == "complex":
        locals_after = locals_after._replace(
            coefficients=enforce_conjugate_symmetry_batch(
                jnp.asarray(locals_after.coefficients),
                order=p,
            )
        )

    return TreeDownwardData(
        interactions=interactions,
        locals=locals_after,
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
    ),
)
def _evaluate_tree_compiled_impl(
    tree: Tree,
    positions: Array,
    masses: Array,
    locals_data: LocalExpansionData,
    neighbor_list: NodeNeighborList,
    leaf_nodes: Array,
    node_ranges: Array,
    precomputed_target_leaf_ids: Array,
    precomputed_source_leaf_ids: Array,
    precomputed_valid_pairs: Array,
    precomputed_chunk_sort_indices: Array,
    precomputed_chunk_group_ids: Array,
    precomputed_chunk_unique_indices: Array,
    *,
    G: float,
    softening: float,
    order: int,
    expansion_basis: ExpansionBasis,
    max_leaf_size: int,
    return_potential: bool,
    nearfield_mode: str,
    nearfield_edge_chunk_size: int,
) -> Union[Array, Tuple[Array, Array]]:
    """JIT core for far/near field evaluation on a prepared tree state."""
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
        precomputed_valid_pairs=precomputed_valid_pairs if use_precomputed else None,
        precomputed_chunk_sort_indices=(
            precomputed_chunk_sort_indices if use_precomputed_scatter else None
        ),
        precomputed_chunk_group_ids=(
            precomputed_chunk_group_ids if use_precomputed_scatter else None
        ),
        precomputed_chunk_unique_indices=(
            precomputed_chunk_unique_indices if use_precomputed_scatter else None
        ),
    )

    far_grad, far_potential_pre = _evaluate_local_expansions_for_particles(
        locals_data,
        positions,
        leaf_nodes=leaf_nodes,
        node_ranges=node_ranges,
        max_leaf_size=max_leaf_size,
        order=order,
        expansion_basis=expansion_basis,
        return_potential=return_potential,
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
    nearfield_target_leaf_ids: Optional[Array],
    nearfield_source_leaf_ids: Optional[Array],
    nearfield_valid_pairs: Optional[Array],
    nearfield_chunk_sort_indices: Optional[Array],
    nearfield_chunk_group_ids: Optional[Array],
    nearfield_chunk_unique_indices: Optional[Array],
    max_leaf_size: int,
    return_potential: bool,
    jit_traversal: bool,
) -> Union[Array, Tuple[Array, Array]]:
    """Run the prepared-tree evaluation returning Morton-sorted outputs."""

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
    tree: Tree,
    neighbor_list: NodeNeighborList,
) -> tuple[Array, Array]:
    """Build padded source-index lists for each target particle near-field eval."""
    targets = jnp.asarray(target_sorted_indices, dtype=INDEX_DTYPE)
    target_leaf_pos = jnp.asarray(target_leaf_positions, dtype=INDEX_DTYPE)
    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    offsets = jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE)
    neighbors = jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE)

    num_targets = int(targets.shape[0])
    if num_targets == 0:
        empty_idx = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        empty_mask = jnp.zeros((0, 0), dtype=bool)
        return empty_idx, empty_mask

    total_nodes = int(node_ranges.shape[0])
    num_leaves = int(leaf_nodes.shape[0])
    if num_leaves == 0:
        empty_idx = jnp.zeros((num_targets, 0), dtype=INDEX_DTYPE)
        empty_mask = jnp.zeros((num_targets, 0), dtype=bool)
        return empty_idx, empty_mask

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
) -> tuple[Array, Optional[Array]]:
    """Compute near-field contributions for target particles only."""
    target_positions = positions_sorted[target_sorted_indices]
    dtype = positions_sorted.dtype
    g_const = jnp.asarray(G, dtype=dtype)
    softening_sq = jnp.asarray(float(softening) ** 2, dtype=dtype)
    if int(source_indices.shape[1]) == 0:
        zeros = jnp.zeros((target_positions.shape[0], 3), dtype=positions_sorted.dtype)
        if return_potential:
            return zeros, jnp.zeros((target_positions.shape[0],), dtype=zeros.dtype)
        return zeros, None
    src_pos = positions_sorted[source_indices]
    src_mass = masses_sorted[source_indices]
    diff = target_positions[:, None, :] - src_pos
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
    eps = jnp.finfo(positions_sorted.dtype).eps
    inv_r = jnp.where(source_mask, 1.0 / (jnp.sqrt(dist_sq) + eps), 0.0)
    inv_dist3 = jnp.where(source_mask, inv_r / dist_sq, 0.0)
    weighted = inv_dist3 * src_mass
    near_acc = -g_const * jnp.sum(weighted[..., None] * diff, axis=1)
    if not return_potential:
        return near_acc, None
    near_pot = -g_const * jnp.sum(inv_r * src_mass, axis=1)
    return near_acc, near_pot


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
) -> tuple[Array, Optional[Array]]:
    """Evaluate far-field local expansions for target particles only."""
    if order > MAX_MULTIPOLE_ORDER and expansion_basis != "solidfmm":
        raise NotImplementedError(
            "orders above 4 require expansion_basis='solidfmm'",
        )
    if int(target_sorted_indices.shape[0]) == 0:
        zeros = jnp.zeros((0, 3), dtype=positions_sorted.dtype)
        if return_potential:
            return zeros, jnp.zeros((0,), dtype=positions_sorted.dtype)
        return zeros, None

    target_leaf_nodes = leaf_nodes[target_leaf_positions]
    centers = local_data.centers[target_leaf_nodes]
    coeffs = local_data.coefficients[target_leaf_nodes]
    target_positions = positions_sorted[target_sorted_indices]

    if expansion_basis == "solidfmm":
        offsets_solid = centers - target_positions
        if jnp.issubdtype(coeffs.dtype, jnp.complexfloating):

            def eval_one(coeff_row: Array, offset_row: Array) -> tuple[Array, Array]:
                grad, pot = evaluate_local_complex_with_grad(
                    coeff_row,
                    offset_row,
                    order=int(order),
                )
                return grad, pot

        else:

            def eval_one(coeff_row: Array, offset_row: Array) -> tuple[Array, Array]:
                grad, pot = evaluate_local_real_with_grad(
                    coeff_row,
                    offset_row,
                    order=int(order),
                )
                return grad, pot

        gradients, potentials = jax.vmap(eval_one)(coeffs, offsets_solid)
        if return_potential:
            return gradients, potentials
        return gradients, None

    offsets = target_positions - centers

    gradients, potentials = _evaluate_local_cartesian_with_grad_batch(
        coeffs,
        offsets,
        order=order,
    )
    if return_potential:
        return gradients, potentials
    return gradients, None


def _evaluate_prepared_tree_targets(
    *,
    fmm: "FastMultipoleMethod",
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    downward: TreeDownwardData,
    neighbor_list: NodeNeighborList,
    target_sorted_indices: Array,
    return_potential: bool,
) -> Union[Array, Tuple[Array, Array]]:
    """Run prepared-tree evaluation for target particles only."""
    g_const = jnp.asarray(fmm.G, dtype=positions_sorted.dtype)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    target_leaf_positions = _map_targets_to_leaf_positions(
        target_sorted_indices=target_sorted_indices,
        leaf_nodes=leaf_nodes,
        node_ranges=node_ranges,
    )
    near_source_idx, near_source_mask = _build_target_nearfield_source_index_matrix(
        target_sorted_indices=target_sorted_indices,
        target_leaf_positions=target_leaf_positions,
        tree=tree,
        neighbor_list=neighbor_list,
    )
    near_acc, near_pot = _compute_targeted_nearfield(
        positions_sorted=positions_sorted,
        masses_sorted=masses_sorted,
        target_sorted_indices=target_sorted_indices,
        source_indices=near_source_idx,
        source_mask=near_source_mask,
        G=g_const,
        softening=float(fmm.softening),
        return_potential=return_potential,
    )
    far_grad, far_potential_pre = _evaluate_local_expansions_for_target_particles(
        local_data=downward.locals,
        positions_sorted=positions_sorted,
        target_sorted_indices=target_sorted_indices,
        target_leaf_positions=target_leaf_positions,
        leaf_nodes=leaf_nodes,
        order=int(downward.locals.order),
        expansion_basis=fmm.expansion_basis,
        return_potential=return_potential,
    )
    far_acc = -g_const * far_grad
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
        return near_acc + far_acc, near_pot_resolved + far_pot
    return near_acc + far_acc


@partial(
    jax.jit,
    static_argnames=(
        "max_leaf_size",
        "return_potential",
        "order",
        "expansion_basis",
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
) -> Tuple[Array, Optional[Array]]:
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

        # Both complex and real solidfmm locals use delta = center - eval_point.
        offsets_solid = centers[:, None, :] - leaf_positions
        offsets_solid = jnp.where(valid[..., None], offsets_solid, 0.0)
        if jnp.issubdtype(coeffs.dtype, jnp.complexfloating):

            def evaluate_leaf_solid(
                coeffs_leaf: Array,
                offsets_leaf: Array,
                mask_leaf: Array,
            ) -> tuple[Array, Array]:
                grads, values = evaluate_local_complex_with_grad_batch(
                    coeffs_leaf,
                    offsets_leaf,
                    order=p,
                )
                grads = jnp.where(mask_leaf[..., None], grads, 0.0)
                values = jnp.where(mask_leaf, values, 0.0)
                return grads, values

        else:

            def evaluate_leaf_solid(
                coeffs_leaf: Array,
                offsets_leaf: Array,
                mask_leaf: Array,
            ) -> tuple[Array, Array]:
                grads, values = jax.vmap(
                    lambda d: evaluate_local_real_with_grad(coeffs_leaf, d, order=p)
                )(offsets_leaf)
                grads = jnp.where(mask_leaf[..., None], grads, 0.0)
                values = jnp.where(mask_leaf, values, 0.0)
                return grads, values

        grad_field, potentials = jax.vmap(evaluate_leaf_solid)(
            coeffs, offsets_solid, valid
        )

        gradients = _scatter_vectors(
            jnp.zeros_like(positions),
            safe_idx,
            grad_field,
            valid,
        )

        if not return_potential:
            return gradients, None

        potentials_flat = _scatter_scalars(
            jnp.zeros((positions.shape[0],), dtype=dtype),
            safe_idx,
            potentials,
            valid,
        )
        return gradients, potentials_flat

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
        return gradients, None

    potentials_flat = _scatter_scalars(
        jnp.zeros((positions.shape[0],), dtype=dtype),
        safe_idx,
        potentials,
        valid,
    )
    return gradients, potentials_flat


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
    masked = jnp.where(flat_mask[:, None], flat_values, 0.0)
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
    masked = jnp.where(flat_mask, flat_values, 0.0)
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
