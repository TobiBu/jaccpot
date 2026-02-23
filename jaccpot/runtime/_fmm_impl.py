"""
Fast Multipole Method (FMM) for computing gravitational accelerations.

This implementation uses multipole and local expansions to compute
gravitational forces in O(N) time instead of O(N^2) for direct summation.
"""

import hashlib
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
from yggdrax.dense_interactions import DenseInteractionBuffers, densify_interactions
from yggdrax.grouped_interactions import GroupedInteractionBuffers
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
from yggdrax.tree import (
    RadixTree,
    RadixTreeWorkspace,
    build_fixed_depth_tree,
    build_fixed_depth_tree_jit,
    build_tree,
    build_tree_jit,
)

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
from jaccpot.operators.multipole_utils import (
    LOCAL_COMBO_INV_FACTORIAL,
    LOCAL_LEVEL_COMBOS,
    MAX_MULTIPOLE_ORDER,
    level_offset,
    multi_power,
    total_coefficients,
)
from jaccpot.operators.real_harmonics import (
    evaluate_local_real_with_grad,
    l2l_real,
    m2l_real,
    sh_size,
)
from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_upward_sweep,
)
from jaccpot.upward.spherical_tree_expansions import (
    prepare_spherical_upward_sweep as prepare_spherical_tree_upward_sweep,
)
from jaccpot.upward.tree_expansions import (
    NodeMultipoleData,
    TreeUpwardData,
)
from jaccpot.upward.tree_expansions import (
    prepare_upward_sweep as prepare_tree_upward_sweep,
)

from .dtypes import INDEX_DTYPE, as_index, complex_dtype_for_real
from .fmm_presets import FMMPreset, FMMPresetConfig, get_preset_config
from .reference import MultipoleExpansion
from .reference import compute_expansion as reference_compute_expansion
from .reference import compute_gravitational_potential as reference_compute_potential
from .reference import direct_sum as reference_direct_sum
from .reference import evaluate_expansion as reference_evaluate_expansion

ExpansionBasis = Literal["cartesian", "spherical", "solidfmm"]
FarFieldMode = Literal["auto", "pair_grouped", "class_major"]
NearFieldMode = Literal["auto", "baseline", "bucketed"]


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

    tree: RadixTree
    positions_sorted: Array
    masses_sorted: Array
    inverse_permutation: Array
    workspace: Optional[RadixTreeWorkspace]
    max_leaf_size: int
    cache_leaf_parameter: int


@dataclass(frozen=True)
class _DualTreeArtifacts:
    """Artifacts emitted by the dual-tree traversal builder."""

    interactions: NodeInteractionList
    neighbor_list: NodeNeighborList
    traversal_result: DualTreeWalkResult
    dense_buffers: Optional[DenseInteractionBuffers]
    grouped_buffers: Optional[GroupedInteractionBuffers]
    grouped_segment_starts: Optional[Array]
    grouped_segment_lengths: Optional[Array]
    grouped_segment_class_ids: Optional[Array]
    grouped_segment_sort_permutation: Optional[Array]
    grouped_segment_group_ids: Optional[Array]
    grouped_segment_unique_targets: Optional[Array]
    grouped_chunk_size: Optional[int]


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
_NEARFIELD_BUCKETED_CPU_PARTICLE_THRESHOLD = 262144
_NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_MEDIUM = 1024
_NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_LARGE = 2048
_NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_XL = 4096
_NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP = 16_000_000
_LARGE_CPU_M2L_CHUNK_SIZE = 32768
_LARGE_CPU_TRAVERSAL_CONFIG = DualTreeTraversalConfig(
    max_pair_queue=131072,
    process_block=4096,
    max_interactions_per_node=65536,
    max_neighbors_per_leaf=32768,
)
_OPERATOR_CACHE_MAX = 4096
_operator_blocks_cache: "OrderedDict[tuple, tuple[Array, Array]]" = OrderedDict()


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
    valid_tree_modes = {"lbvh", "fixed_depth"}
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
    tree_config: TreeBuilderConfig,
    leaf_size: int,
    workspace: Optional[RadixTreeWorkspace],
    jit_tree: bool,
    refine_local: bool,
    max_refine_levels: int,
    aspect_threshold: float,
) -> _TreeBuildArtifacts:
    """Construct a tree according to the resolved builder configuration."""

    mode = tree_config.mode
    if mode == "fixed_depth":
        build_fn = build_fixed_depth_tree_jit if jit_tree else build_fixed_depth_tree
        (
            tree,
            pos_sorted,
            mass_sorted,
            inverse,
            workspace_out,
        ) = build_fn(
            positions,
            masses,
            bounds,
            target_leaf_particles=tree_config.target_leaf_particles,
            return_reordered=True,
            workspace=workspace,
            return_workspace=True,
            refine_local=refine_local,
            max_refine_levels=max_refine_levels,
            aspect_threshold=aspect_threshold,
        )
        max_leaf_size = _max_leaf_size_from_tree(tree)
        cache_leaf_parameter = tree_config.target_leaf_particles
    elif mode == "lbvh":
        leaf_cap = int(leaf_size)
        build_fn = build_tree_jit if jit_tree else build_tree
        tree, pos_sorted, mass_sorted, inverse, workspace_out = build_fn(
            positions,
            masses,
            bounds,
            leaf_size=leaf_cap,
            return_reordered=True,
            workspace=workspace,
            return_workspace=True,
        )
        max_leaf_size = leaf_cap
        cache_leaf_parameter = leaf_cap
    else:
        raise ValueError(f"unsupported tree build mode: {mode}")

    return _TreeBuildArtifacts(
        tree=tree,
        positions_sorted=pos_sorted,
        masses_sorted=mass_sorted,
        inverse_permutation=inverse,
        workspace=workspace_out,
        max_leaf_size=int(max_leaf_size),
        cache_leaf_parameter=int(cache_leaf_parameter),
    )


def _interaction_cache_key(
    tree: RadixTree,
    *,
    tree_mode: str,
    leaf_parameter: int,
    theta: float,
    mac_type: MACType,
    dehnen_radius_scale: float,
    expansion_basis: ExpansionBasis,
    center_mode: str,
    max_pair_queue: Optional[int],
    pair_process_block: Optional[int],
    traversal_config: Optional[DualTreeTraversalConfig],
    refine_local: Optional[bool],
    max_refine_levels: Optional[int],
    aspect_threshold: Optional[float],
) -> Optional[str]:
    """Return a hash for the interaction list of a tree/theta configuration.

    If any tree arrays are tracers (e.g., under grad/jit), return ``None`` to
    disable caching and avoid host round-trips on traced values.
    """

    hasher = hashlib.sha256()

    try:
        morton_codes = np.asarray(
            jax.device_get(tree.morton_codes),
            dtype=np.uint64,
        )
        node_ranges = np.asarray(
            jax.device_get(tree.node_ranges),
            dtype=np.int64,
        )
        bounds_min = np.asarray(
            jax.device_get(tree.bounds_min),
            dtype=np.float64,
        )
        bounds_max = np.asarray(
            jax.device_get(tree.bounds_max),
            dtype=np.float64,
        )
    except Exception:
        # Any tracer or host-conversion failure disables caching under
        # transformation (e.g., grad/jit).
        return None

    hasher.update(morton_codes.tobytes())
    hasher.update(node_ranges.tobytes())
    hasher.update(bounds_min.tobytes())
    hasher.update(bounds_max.tobytes())

    mode_bytes = tree_mode.encode("utf8")
    leaf_bytes = np.asarray(int(leaf_parameter), dtype=np.int64).tobytes()
    theta_bytes = np.asarray(float(theta), dtype=np.float64).tobytes()
    mac_bytes = str(mac_type).encode("utf8")
    dehnen_scale_bytes = np.asarray(
        float(dehnen_radius_scale), dtype=np.float64
    ).tobytes()
    basis_bytes = str(expansion_basis).encode("utf8")
    center_mode_bytes = str(center_mode).encode("utf8")
    if traversal_config is not None:
        queue_val = int(traversal_config.max_pair_queue)
        block_val = int(traversal_config.process_block)
        interaction_val = int(traversal_config.max_interactions_per_node)
        neighbor_val = int(traversal_config.max_neighbors_per_leaf)
    else:
        queue_val = -1 if max_pair_queue is None else int(max_pair_queue)
        block_val = -1 if pair_process_block is None else int(pair_process_block)
        interaction_val = -1
        neighbor_val = -1
    refine_val = -1 if refine_local is None else int(bool(refine_local))
    max_refine_val = -1 if max_refine_levels is None else int(max_refine_levels)
    aspect_val = -1.0 if aspect_threshold is None else float(aspect_threshold)
    hasher.update(mode_bytes)
    hasher.update(leaf_bytes)
    hasher.update(theta_bytes)
    hasher.update(mac_bytes)
    hasher.update(dehnen_scale_bytes)
    hasher.update(basis_bytes)
    hasher.update(center_mode_bytes)
    hasher.update(np.asarray(queue_val, dtype=np.int64).tobytes())
    hasher.update(np.asarray(block_val, dtype=np.int64).tobytes())
    hasher.update(np.asarray(interaction_val, dtype=np.int64).tobytes())
    hasher.update(np.asarray(neighbor_val, dtype=np.int64).tobytes())
    hasher.update(np.asarray(refine_val, dtype=np.int64).tobytes())
    hasher.update(np.asarray(max_refine_val, dtype=np.int64).tobytes())
    hasher.update(np.asarray(aspect_val, dtype=np.float64).tobytes())
    return hasher.hexdigest()


class FMMPreparedState(NamedTuple):
    """Keep tree data resident on device across repeated evaluations."""

    tree: RadixTree
    positions_sorted: Array
    masses_sorted: Array
    inverse_permutation: Array
    downward: TreeDownwardData
    neighbor_list: NodeNeighborList
    max_leaf_size: int
    input_dtype: jnp.dtype
    working_dtype: jnp.dtype
    expansion_basis: ExpansionBasis
    theta: float
    interactions: NodeInteractionList
    dual_tree_result: DualTreeWalkResult
    retry_events: Tuple[DualTreeRetryEvent, ...]
    nearfield_target_leaf_ids: Optional[Array]
    nearfield_source_leaf_ids: Optional[Array]
    nearfield_valid_pairs: Optional[Array]
    nearfield_chunk_sort_indices: Optional[Array]
    nearfield_chunk_group_ids: Optional[Array]
    nearfield_chunk_unique_indices: Optional[Array]


class _InteractionCacheEntry(NamedTuple):
    """Cache entry for dual-tree interaction artifacts keyed by build options."""

    key: str
    interactions: NodeInteractionList
    neighbor_list: NodeNeighborList
    dual_tree_result: DualTreeWalkResult
    grouped_buffers: Optional[GroupedInteractionBuffers]
    grouped_segment_starts: Optional[Array]
    grouped_segment_lengths: Optional[Array]
    grouped_segment_class_ids: Optional[Array]
    grouped_segment_sort_permutation: Optional[Array]
    grouped_segment_group_ids: Optional[Array]
    grouped_segment_unique_targets: Optional[Array]
    grouped_chunk_size: Optional[int]
    nearfield_target_leaf_ids: Optional[Array]
    nearfield_source_leaf_ids: Optional[Array]
    nearfield_valid_pairs: Optional[Array]
    nearfield_chunk_sort_indices: Optional[Array]
    nearfield_chunk_group_ids: Optional[Array]
    nearfield_chunk_unique_indices: Optional[Array]
    nearfield_mode: Optional[str]
    nearfield_edge_chunk_size: Optional[int]
    nearfield_leaf_cap: Optional[int]


def _build_dual_tree_artifacts(
    tree: RadixTree,
    geometry,
    *,
    theta: float,
    mac_type: MACType,
    dehnen_radius_scale: float,
    cache_key: Optional[str],
    cache_entry: Optional[_InteractionCacheEntry],
    max_pair_queue: Optional[int],
    pair_process_block: Optional[int],
    traversal_config: Optional[DualTreeTraversalConfig],
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]],
    use_dense_interactions: bool,
    grouped_interactions: bool,
    grouped_chunk_size: Optional[int],
) -> tuple[_DualTreeArtifacts, Optional[_InteractionCacheEntry]]:
    """Construct or reuse dual-tree traversal products for a tree."""

    cache_out = cache_entry
    if (
        cache_key is not None
        and cache_entry is not None
        and cache_entry.key == cache_key
    ):
        interactions = cache_entry.interactions
        neighbor_list = cache_entry.neighbor_list
        traversal_result = cache_entry.dual_tree_result
        grouped_buffers = cache_entry.grouped_buffers
        grouped_segment_starts = cache_entry.grouped_segment_starts
        grouped_segment_lengths = cache_entry.grouped_segment_lengths
        grouped_segment_class_ids = cache_entry.grouped_segment_class_ids
        grouped_segment_sort_permutation = cache_entry.grouped_segment_sort_permutation
        grouped_segment_group_ids = cache_entry.grouped_segment_group_ids
        grouped_segment_unique_targets = cache_entry.grouped_segment_unique_targets
        grouped_chunk_size_cached = cache_entry.grouped_chunk_size
    else:
        from . import fmm as _runtime_fmm

        build_out = _runtime_fmm.build_interactions_and_neighbors(
            tree,
            geometry,
            theta=theta,
            mac_type=mac_type,
            dehnen_radius_scale=dehnen_radius_scale,
            max_pair_queue=max_pair_queue,
            process_block=pair_process_block,
            traversal_config=traversal_config,
            retry_logger=retry_logger,
            return_result=True,
            return_grouped=grouped_interactions,
        )
        if grouped_interactions:
            (
                interactions,
                neighbor_list,
                traversal_result,
                grouped_buffers,
            ) = build_out
        else:
            (
                interactions,
                neighbor_list,
                traversal_result,
            ) = build_out
        cache_out = (
            _InteractionCacheEntry(
                key=cache_key,
                interactions=interactions,
                neighbor_list=neighbor_list,
                dual_tree_result=traversal_result,
                grouped_buffers=grouped_buffers if grouped_interactions else None,
                grouped_segment_starts=None,
                grouped_segment_lengths=None,
                grouped_segment_class_ids=None,
                grouped_segment_sort_permutation=None,
                grouped_segment_group_ids=None,
                grouped_segment_unique_targets=None,
                grouped_chunk_size=None,
                nearfield_target_leaf_ids=None,
                nearfield_source_leaf_ids=None,
                nearfield_valid_pairs=None,
                nearfield_chunk_sort_indices=None,
                nearfield_chunk_group_ids=None,
                nearfield_chunk_unique_indices=None,
                nearfield_mode=None,
                nearfield_edge_chunk_size=None,
                nearfield_leaf_cap=None,
            )
            if cache_key is not None
            else None
        )
        grouped_segment_starts = None
        grouped_segment_lengths = None
        grouped_segment_class_ids = None
        grouped_segment_sort_permutation = None
        grouped_segment_group_ids = None
        grouped_segment_unique_targets = None
        grouped_chunk_size_cached = None
        if not grouped_interactions:
            grouped_buffers = None

    if grouped_interactions and grouped_buffers is None:
        from yggdrax import interactions as _yggdrax_interactions

        grouped_buffers = _yggdrax_interactions.build_grouped_interactions_from_pairs(
            tree,
            geometry,
            interactions.sources,
            interactions.targets,
            level_offsets=getattr(interactions, "level_offsets", None),
        )
        if cache_out is not None:
            cache_out = _InteractionCacheEntry(
                key=cache_out.key,
                interactions=cache_out.interactions,
                neighbor_list=cache_out.neighbor_list,
                dual_tree_result=cache_out.dual_tree_result,
                grouped_buffers=grouped_buffers,
                grouped_segment_starts=cache_out.grouped_segment_starts,
                grouped_segment_lengths=cache_out.grouped_segment_lengths,
                grouped_segment_class_ids=cache_out.grouped_segment_class_ids,
                grouped_segment_sort_permutation=cache_out.grouped_segment_sort_permutation,
                grouped_segment_group_ids=cache_out.grouped_segment_group_ids,
                grouped_segment_unique_targets=cache_out.grouped_segment_unique_targets,
                grouped_chunk_size=cache_out.grouped_chunk_size,
                nearfield_target_leaf_ids=cache_out.nearfield_target_leaf_ids,
                nearfield_source_leaf_ids=cache_out.nearfield_source_leaf_ids,
                nearfield_valid_pairs=cache_out.nearfield_valid_pairs,
                nearfield_chunk_sort_indices=cache_out.nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids=cache_out.nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices=cache_out.nearfield_chunk_unique_indices,
                nearfield_mode=cache_out.nearfield_mode,
                nearfield_edge_chunk_size=cache_out.nearfield_edge_chunk_size,
                nearfield_leaf_cap=cache_out.nearfield_leaf_cap,
            )

    if (
        grouped_interactions
        and grouped_buffers is not None
        and grouped_chunk_size is not None
    ):
        needs_schedule = (
            grouped_segment_starts is None
            or grouped_segment_lengths is None
            or grouped_segment_class_ids is None
            or grouped_segment_sort_permutation is None
            or grouped_segment_group_ids is None
            or grouped_segment_unique_targets is None
            or grouped_chunk_size_cached != int(grouped_chunk_size)
        )
        if needs_schedule:
            from . import fmm as _runtime_fmm

            (
                grouped_segment_starts,
                grouped_segment_lengths,
                grouped_segment_class_ids,
                grouped_segment_sort_permutation,
                grouped_segment_group_ids,
                grouped_segment_unique_targets,
            ) = _runtime_fmm._build_grouped_class_segments(
                grouped_buffers,
                chunk_size=int(grouped_chunk_size),
            )
            grouped_chunk_size_cached = int(grouped_chunk_size)
            if cache_out is not None:
                cache_out = _InteractionCacheEntry(
                    key=cache_out.key,
                    interactions=cache_out.interactions,
                    neighbor_list=cache_out.neighbor_list,
                    dual_tree_result=cache_out.dual_tree_result,
                    grouped_buffers=grouped_buffers,
                    grouped_segment_starts=grouped_segment_starts,
                    grouped_segment_lengths=grouped_segment_lengths,
                    grouped_segment_class_ids=grouped_segment_class_ids,
                    grouped_segment_sort_permutation=grouped_segment_sort_permutation,
                    grouped_segment_group_ids=grouped_segment_group_ids,
                    grouped_segment_unique_targets=grouped_segment_unique_targets,
                    grouped_chunk_size=grouped_chunk_size_cached,
                    nearfield_target_leaf_ids=cache_out.nearfield_target_leaf_ids,
                    nearfield_source_leaf_ids=cache_out.nearfield_source_leaf_ids,
                    nearfield_valid_pairs=cache_out.nearfield_valid_pairs,
                    nearfield_chunk_sort_indices=cache_out.nearfield_chunk_sort_indices,
                    nearfield_chunk_group_ids=cache_out.nearfield_chunk_group_ids,
                    nearfield_chunk_unique_indices=cache_out.nearfield_chunk_unique_indices,
                    nearfield_mode=cache_out.nearfield_mode,
                    nearfield_edge_chunk_size=cache_out.nearfield_edge_chunk_size,
                    nearfield_leaf_cap=cache_out.nearfield_leaf_cap,
                )

    dense_buffers = (
        densify_interactions(tree, geometry, interactions)
        if use_dense_interactions
        else None
    )

    artifacts = _DualTreeArtifacts(
        interactions=interactions,
        neighbor_list=neighbor_list,
        traversal_result=traversal_result,
        dense_buffers=dense_buffers,
        grouped_buffers=grouped_buffers,
        grouped_segment_starts=grouped_segment_starts,
        grouped_segment_lengths=grouped_segment_lengths,
        grouped_segment_class_ids=grouped_segment_class_ids,
        grouped_segment_sort_permutation=grouped_segment_sort_permutation,
        grouped_segment_group_ids=grouped_segment_group_ids,
        grouped_segment_unique_targets=grouped_segment_unique_targets,
        grouped_chunk_size=grouped_chunk_size_cached,
    )
    return artifacts, cache_out


class FastMultipoleMethod:
    """
    Fast Multipole Method for gravitational N-body simulations.

    Args:
        theta: Opening angle criterion (typically 0.5-1.0)
        G: Gravitational constant (default: 1.0)
        softening: Softening length to avoid singularities (default: 0.0)
        tree_build_mode:
            Choose between "lbvh" and "fixed_depth" builders.
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
        mac_type: MACType = "bh",
        complex_rotation: str = "solidfmm",  # "cached",
        dehnen_radius_scale: float = 1.0,
        m2l_chunk_size: Optional[int] = None,
        l2l_chunk_size: Optional[int] = None,
        max_pair_queue: Optional[int] = None,
        pair_process_block: Optional[int] = None,
        traversal_config: Optional[DualTreeTraversalConfig] = None,
        tree_build_mode: Optional[str] = None,
        target_leaf_particles: Optional[int] = None,
        refine_local: Optional[bool] = None,
        max_refine_levels: Optional[int] = None,
        aspect_threshold: Optional[float] = None,
        interaction_retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
        use_dense_interactions: Optional[bool] = None,
        grouped_interactions: Optional[bool] = None,
        farfield_mode: FarFieldMode = "auto",
        nearfield_mode: NearFieldMode = "auto",
        nearfield_edge_chunk_size: int = 256,
        host_refine_mode: str = "auto",
        preset: Optional[Union[str, FMMPreset]] = None,
        fixed_order: Optional[int] = None,
        fixed_max_leaf_size: Optional[int] = None,
    ):
        """Initialize FMM runtime with validated policy and kernel settings."""
        basis_norm = str(expansion_basis).strip().lower()
        if basis_norm not in ("cartesian", "spherical", "solidfmm"):
            raise ValueError(
                "expansion_basis must be 'cartesian', 'spherical', or 'solidfmm'",
            )
        self.expansion_basis = basis_norm  # type: ignore[assignment]

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
        nearfield_mode_norm = str(nearfield_mode).strip().lower()
        if nearfield_mode_norm not in ("auto", "baseline", "bucketed"):
            raise ValueError("nearfield_mode must be 'auto', 'baseline', or 'bucketed'")
        if int(nearfield_edge_chunk_size) <= 0:
            raise ValueError("nearfield_edge_chunk_size must be positive")
        self.nearfield_mode = nearfield_mode_norm
        self.nearfield_edge_chunk_size = int(nearfield_edge_chunk_size)
        dehnen_scale_val = float(dehnen_radius_scale)
        if dehnen_scale_val <= 0.0:
            raise ValueError("dehnen_radius_scale must be > 0")
        self.dehnen_radius_scale = dehnen_scale_val

        refine_mode_norm = str(host_refine_mode).strip().lower()
        if refine_mode_norm not in ("auto", "on", "off"):
            raise ValueError("host_refine_mode must be 'auto', 'on', or 'off'")
        self.host_refine_mode = refine_mode_norm

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
        self._tree_workspace: Optional[RadixTreeWorkspace] = None
        self._locals_template: Optional[LocalExpansionData] = None
        self._interaction_cache: Optional[_InteractionCacheEntry] = None
        self._prepared_state_cache_key: Optional[tuple[Any, ...]] = None
        self._prepared_state_cache_value: Optional[FMMPreparedState] = None
        self._prepared_state_cache_positions: Optional[Array] = None
        self._prepared_state_cache_masses: Optional[Array] = None
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

    def clear_prepared_state_cache(self: "FastMultipoleMethod") -> None:
        """Clear cached prepared-state payloads used by reuse mode."""

        self._prepared_state_cache_key = None
        self._prepared_state_cache_value = None
        self._prepared_state_cache_positions = None
        self._prepared_state_cache_masses = None

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

        if self.host_refine_mode == "off":
            refine_local_override = False
        elif self.host_refine_mode == "on":
            refine_local_override = True
        elif large_cpu and self.tree_build_mode == "fixed_depth":
            refine_local_override = False

        if (
            not self._explicit_grouped_interactions
            and self.preset == "fast"
            and self.expansion_basis == "solidfmm"
            and self.mac_type == "dehnen"
            and self.tree_build_mode == "fixed_depth"
            and large_cpu
        ):
            grouped_interactions = True

        if (
            self.preset == "fast"
            and self.expansion_basis == "solidfmm"
            and self.mac_type == "dehnen"
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
        if grouped_interactions:
            center_mode = "aabb"
            if farfield_mode == "auto":
                farfield_mode = "class_major" if class_major_cpu else "pair_grouped"
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
        if max_order > MAX_MULTIPOLE_ORDER and self.expansion_basis not in (
            "spherical",
            "solidfmm",
        ):
            raise NotImplementedError(
                "orders above 4 require expansion_basis='spherical' or 'solidfmm'",
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

    def _build_locals_template_for_prepare_state(
        self,
        *,
        tree: RadixTree,
        upward: TreeUpwardData,
        max_order: int,
        pos_sorted: Array,
    ) -> Optional[LocalExpansionData]:
        """Build initial local-expansion buffers matching the active basis."""
        if self.expansion_basis == "spherical":
            return None
        if self.expansion_basis == "solidfmm":
            total_nodes = int(tree.parent.shape[0])
            coeff_count = sh_size(max_order)
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

    def _prepare_nearfield_precompute_artifacts(
        self,
        *,
        tree: RadixTree,
        neighbor_list: NodeNeighborList,
        leaf_cap: int,
        num_particles: int,
        nearfield_mode: Optional[str] = None,
        nearfield_edge_chunk_size: Optional[int] = None,
    ) -> tuple[
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
    ]:
        """Best-effort precompute of nearfield leaf-pair and scatter artifacts."""
        nearfield_target_leaf_ids = None
        nearfield_source_leaf_ids = None
        nearfield_valid_pairs = None
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

        try:
            (
                nearfield_target_leaf_ids,
                nearfield_source_leaf_ids,
                nearfield_valid_pairs,
            ) = prepare_leaf_neighbor_pairs(
                jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
                jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE),
                jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE),
                jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE),
            )
        except Exception:
            nearfield_target_leaf_ids = None
            nearfield_source_leaf_ids = None
            nearfield_valid_pairs = None

        if (
            resolved_nearfield_mode == "bucketed"
            and nearfield_target_leaf_ids is not None
            and nearfield_valid_pairs is not None
        ):
            try:
                edge_count = int(nearfield_target_leaf_ids.shape[0])
                chunk = int(resolved_nearfield_edge_chunk_size)
                chunk_count = (edge_count + chunk - 1) // chunk if edge_count > 0 else 0
                schedule_items = int(chunk_count * chunk * int(leaf_cap))
                if schedule_items <= _NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP:
                    (
                        nearfield_chunk_sort_indices,
                        nearfield_chunk_group_ids,
                        nearfield_chunk_unique_indices,
                    ) = prepare_bucketed_scatter_schedules(
                        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
                        jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE),
                        nearfield_target_leaf_ids,
                        nearfield_valid_pairs,
                        max_leaf_size=int(leaf_cap),
                        edge_chunk_size=chunk,
                    )
            except Exception:
                nearfield_chunk_sort_indices = None
                nearfield_chunk_group_ids = None
                nearfield_chunk_unique_indices = None

        return (
            nearfield_target_leaf_ids,
            nearfield_source_leaf_ids,
            nearfield_valid_pairs,
            nearfield_chunk_sort_indices,
            nearfield_chunk_group_ids,
            nearfield_chunk_unique_indices,
        )

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
        tree: RadixTree,
        upward: TreeUpwardData,
        theta_val: float,
        locals_template: Optional[LocalExpansionData],
        interactions: NodeInteractionList,
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
        tree: RadixTree,
        positions_sorted: Array,
        masses_sorted: Array,
        *,
        max_order: int = 2,
        center_mode: str = "com",
        explicit_centers: Optional[Array] = None,
    ) -> TreeUpwardData:
        """Bundle geometry, raw moments, and packed expansions for a tree."""

        if self.expansion_basis == "spherical":
            spherical_upward = prepare_spherical_tree_upward_sweep(
                tree,
                positions_sorted,
                masses_sorted,
                max_order=max_order,
                center_mode=center_mode,
                explicit_centers=explicit_centers,
            )

            # Adapter: reuse the TreeUpwardData shape expected by the existing
            # cartesian pipeline for now.
            #
            # NOTE: This only makes the *upward* data available for inspection
            # and experimentation. The current downward pass still expects the
            # cartesian/STF packed layout (total_coefficients(order)), so a
            # full spherical pipeline will need a dedicated TreeDownwardData
            # path.
            multipoles = NodeMultipoleData(
                order=int(spherical_upward.multipoles.order),
                centers=spherical_upward.multipoles.centers,
                moments=None,  # type: ignore[arg-type]
                packed=spherical_upward.multipoles.packed,
                component_matrix=spherical_upward.multipoles.packed,
            )

            return TreeUpwardData(
                geometry=spherical_upward.geometry,
                mass_moments=spherical_upward.mass_moments,
                multipoles=multipoles,
            )

        if self.expansion_basis == "solidfmm":
            complex_upward = prepare_solidfmm_complex_upward_sweep(
                tree,
                positions_sorted,
                masses_sorted,
                max_order=max_order,
                center_mode=center_mode,
                explicit_centers=explicit_centers,
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
        tree: RadixTree,
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
        tree: RadixTree,
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
        # If spherical basis requested, use spherical M2L -> spherical locals
        if self.expansion_basis == "spherical":
            return _prepare_spherical_downward_sweep(
                tree,
                upward_data,
                theta=theta_val,
                mac_type=mac_type_val,
                initial_locals=initial_locals,
                interactions=interactions,
                m2l_chunk_size=m2l_chunk_size,
                retry_logger=retry_callback,
                traversal_config=config,
                dense_buffers=dense_buffers,
                dehnen_radius_scale=dehnen_scale_val,
            )

        if self.expansion_basis == "solidfmm":
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
                dehnen_radius_scale=dehnen_scale_val,
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
        jit_tree:
            When ``True`` (default) specialise tree construction via JIT to
            amortise repeated builds for consistent tree sizes.
        jit_traversal:
            When ``True`` (default) evaluate the leaf traversal with the
            compiled implementation for improved throughput.
        reuse_prepared_state:
            Reuse the most recent prepared state when identical array objects
            and preparation parameters are provided.
        refine_local:
            Override the fixed-depth builder's local refinement toggle when
            ``tree_build_mode`` is ``"fixed_depth"``.
        max_refine_levels:
            Maximum local refinement iterations passed to the builder.
        aspect_threshold:
            Aspect ratio threshold that triggers additional splits in the
            refinement pass.
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

        return self.evaluate_prepared_state(
            state,
            return_potential=return_potential,
            jit_traversal=jit_traversal_flag,
        )

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

        theta_val = float(self.theta if theta is None else theta)
        mac_type_val = self.mac_type

        inferred_bounds = self._resolve_prepare_state_bounds(
            positions=positions_arr,
            bounds=bounds,
        )

        jit_tree_flag = self._resolve_jit_tree_flag(
            positions_arr,
            jit_tree_override=jit_tree,
        )

        build_artifacts = _build_tree_with_config(
            positions_arr,
            masses_arr,
            inferred_bounds,
            tree_config=self.config.tree,
            leaf_size=int(leaf_size),
            workspace=self._tree_workspace,
            jit_tree=jit_tree_flag,
            refine_local=refine_local_val,
            max_refine_levels=max_refine_levels_val,
            aspect_threshold=aspect_threshold_val,
        )
        self._tree_workspace = build_artifacts.workspace
        tree_mode = self.tree_build_mode
        tree = build_artifacts.tree
        pos_sorted = build_artifacts.positions_sorted
        mass_sorted = build_artifacts.masses_sorted
        inverse = build_artifacts.inverse_permutation
        leaf_cap = build_artifacts.max_leaf_size
        leaf_param = build_artifacts.cache_leaf_parameter

        upward = self.prepare_upward_sweep(
            tree,
            pos_sorted,
            mass_sorted,
            max_order=max_order,
            center_mode=upward_center_mode,
        )

        locals_template = self._build_locals_template_for_prepare_state(
            tree=tree,
            upward=upward,
            max_order=max_order,
            pos_sorted=pos_sorted,
        )

        cache_key = _interaction_cache_key(
            tree,
            tree_mode=tree_mode,
            leaf_parameter=leaf_param,
            theta=theta_val,
            mac_type=mac_type_val,
            dehnen_radius_scale=self.dehnen_radius_scale,
            expansion_basis=self.expansion_basis,
            center_mode=upward_center_mode,
            max_pair_queue=self.max_pair_queue,
            pair_process_block=self.pair_process_block,
            traversal_config=runtime_traversal_config,
            refine_local=refine_local_val,
            max_refine_levels=max_refine_levels_val,
            aspect_threshold=aspect_threshold_val,
        )

        dual_artifacts, cache_entry = _build_dual_tree_artifacts(
            tree,
            upward.geometry,
            theta=theta_val,
            mac_type=mac_type_val,
            dehnen_radius_scale=self.dehnen_radius_scale,
            cache_key=cache_key,
            cache_entry=self._interaction_cache,
            max_pair_queue=self.max_pair_queue,
            pair_process_block=self.pair_process_block,
            traversal_config=runtime_traversal_config,
            retry_logger=record_retry,
            use_dense_interactions=self.use_dense_interactions,
            grouped_interactions=grouped_interactions,
            grouped_chunk_size=runtime_m2l_chunk_size,
        )
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

        downward = self._prepare_downward_with_artifacts(
            tree=tree,
            upward=upward,
            theta_val=theta_val,
            locals_template=locals_template,
            interactions=interactions,
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
        )

        if locals_template is not None and self.expansion_basis == "cartesian":
            self._locals_template = LocalExpansionData(
                order=max_order,
                centers=upward.multipoles.centers,
                coefficients=jnp.zeros_like(locals_template.coefficients),
            )
        else:
            self._locals_template = None

        retry_events_tuple = tuple(collected_retries)
        self._recent_retry_events = retry_events_tuple

        nearfield_mode_resolved = self._resolve_nearfield_mode(
            num_particles=int(positions_arr.shape[0])
        )
        nearfield_edge_chunk_size_resolved = self._resolve_nearfield_edge_chunk_size(
            num_particles=int(positions_arr.shape[0]),
            nearfield_mode=nearfield_mode_resolved,
        )
        reuse_nearfield_cache = (
            cache_entry is not None
            and cache_entry.nearfield_mode == nearfield_mode_resolved
            and cache_entry.nearfield_edge_chunk_size
            == nearfield_edge_chunk_size_resolved
            and cache_entry.nearfield_leaf_cap == int(leaf_cap)
            and cache_entry.nearfield_target_leaf_ids is not None
            and cache_entry.nearfield_source_leaf_ids is not None
            and cache_entry.nearfield_valid_pairs is not None
        )
        if reuse_nearfield_cache:
            nearfield_target_leaf_ids = cache_entry.nearfield_target_leaf_ids
            nearfield_source_leaf_ids = cache_entry.nearfield_source_leaf_ids
            nearfield_valid_pairs = cache_entry.nearfield_valid_pairs
            nearfield_chunk_sort_indices = cache_entry.nearfield_chunk_sort_indices
            nearfield_chunk_group_ids = cache_entry.nearfield_chunk_group_ids
            nearfield_chunk_unique_indices = cache_entry.nearfield_chunk_unique_indices
        else:
            (
                nearfield_target_leaf_ids,
                nearfield_source_leaf_ids,
                nearfield_valid_pairs,
                nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices,
            ) = self._prepare_nearfield_precompute_artifacts(
                tree=tree,
                neighbor_list=neighbor_list,
                leaf_cap=leaf_cap,
                num_particles=int(positions_arr.shape[0]),
                nearfield_mode=nearfield_mode_resolved,
                nearfield_edge_chunk_size=nearfield_edge_chunk_size_resolved,
            )
            if cache_entry is not None:
                cache_entry = _InteractionCacheEntry(
                    key=cache_entry.key,
                    interactions=cache_entry.interactions,
                    neighbor_list=cache_entry.neighbor_list,
                    dual_tree_result=cache_entry.dual_tree_result,
                    grouped_buffers=cache_entry.grouped_buffers,
                    grouped_segment_starts=cache_entry.grouped_segment_starts,
                    grouped_segment_lengths=cache_entry.grouped_segment_lengths,
                    grouped_segment_class_ids=cache_entry.grouped_segment_class_ids,
                    grouped_segment_sort_permutation=cache_entry.grouped_segment_sort_permutation,
                    grouped_segment_group_ids=cache_entry.grouped_segment_group_ids,
                    grouped_segment_unique_targets=cache_entry.grouped_segment_unique_targets,
                    grouped_chunk_size=cache_entry.grouped_chunk_size,
                    nearfield_target_leaf_ids=nearfield_target_leaf_ids,
                    nearfield_source_leaf_ids=nearfield_source_leaf_ids,
                    nearfield_valid_pairs=nearfield_valid_pairs,
                    nearfield_chunk_sort_indices=nearfield_chunk_sort_indices,
                    nearfield_chunk_group_ids=nearfield_chunk_group_ids,
                    nearfield_chunk_unique_indices=nearfield_chunk_unique_indices,
                    nearfield_mode=nearfield_mode_resolved,
                    nearfield_edge_chunk_size=nearfield_edge_chunk_size_resolved,
                    nearfield_leaf_cap=int(leaf_cap),
                )
                self._interaction_cache = cache_entry

        return FMMPreparedState(
            tree=tree,
            positions_sorted=pos_sorted,
            masses_sorted=mass_sorted,
            inverse_permutation=jnp.asarray(inverse, dtype=INDEX_DTYPE),
            downward=downward,
            neighbor_list=neighbor_list,
            max_leaf_size=leaf_cap,
            input_dtype=input_dtype,
            working_dtype=positions_arr.dtype,
            expansion_basis=self.expansion_basis,
            theta=theta_val,
            interactions=interactions,
            dual_tree_result=traversal_result,
            retry_events=retry_events_tuple,
            nearfield_target_leaf_ids=nearfield_target_leaf_ids,
            nearfield_source_leaf_ids=nearfield_source_leaf_ids,
            nearfield_valid_pairs=nearfield_valid_pairs,
            nearfield_chunk_sort_indices=nearfield_chunk_sort_indices,
            nearfield_chunk_group_ids=nearfield_chunk_group_ids,
            nearfield_chunk_unique_indices=nearfield_chunk_unique_indices,
        )

    @jaxtyped(typechecker=beartype)
    def evaluate_prepared_state(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        *,
        return_potential: bool = False,
        jit_traversal: bool = True,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Evaluate accelerations (and potentials) for a prepared state."""

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

        if jnp.issubdtype(state.input_dtype, jnp.floating):
            output_dtype = state.input_dtype
        else:
            output_dtype = state.working_dtype

        if return_potential:
            acc_sorted, pot_sorted = evaluation
            accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation]
            accelerations = accelerations.astype(output_dtype)
            potentials = jnp.asarray(pot_sorted)[state.inverse_permutation]
            potentials = potentials.astype(output_dtype)
            return accelerations, potentials

        accelerations = jnp.asarray(evaluation)[state.inverse_permutation]
        accelerations = accelerations.astype(output_dtype)
        return accelerations

    @jaxtyped(typechecker=beartype)
    def evaluate_tree(
        self: "FastMultipoleMethod",
        tree: RadixTree,
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
        tree: RadixTree,
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

        order = (
            int(self.fixed_order)
            if self.fixed_order is not None
            else int(setup.locals_data.order)
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


def _max_leaf_size_from_tree(tree: RadixTree) -> int:
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


def _prepare_tree_evaluation_inputs(
    tree: RadixTree,
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
    # spherical downward preparation handled by top-level helper

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


@partial(jax.jit, static_argnames=("order",))
def _m2l_real_batch_kernel(src_mult: Array, deltas: Array, *, order: int) -> Array:
    """Vectorized real-basis M2L kernel for one interaction batch."""
    return jax.vmap(lambda m, d: m2l_real(m, d, order=order))(src_mult, deltas)


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

    keys_np = np.asarray(jax.device_get(class_keys), dtype=np.int64)
    blocks_to_host: list[Optional[Array]] = [None] * num_classes
    blocks_from_host: list[Optional[Array]] = [None] * num_classes
    missing_indices: list[int] = []

    for class_idx, key_vals in enumerate(keys_np):
        class_key = tuple(int(v) for v in key_vals.tolist())
        cache_key = (int(order), str(rotation), str(dtype), class_key)
        cached = _operator_cache_get(cache_key)
        if cached is None:
            missing_indices.append(class_idx)
            continue
        blocks_to_host[class_idx], blocks_from_host[class_idx] = cached

    if missing_indices:
        miss_idx = jnp.asarray(missing_indices, dtype=INDEX_DTYPE)
        deltas_miss = class_deltas[miss_idx]
        if rotation == "solidfmm":
            blocks_to_miss = complex_rotation_blocks_to_z_solidfmm_batch(
                deltas_miss,
                order=order,
                basis="multipole",
                dtype=dtype,
            )
            blocks_from_miss = complex_rotation_blocks_from_z_solidfmm_batch(
                deltas_miss,
                order=order,
                basis="local",
                dtype=dtype,
            )
        elif rotation == "cached":
            blocks_to_miss = complex_rotation_blocks_to_z_batch(
                deltas_miss,
                order=order,
                basis="multipole",
                dtype=dtype,
            )
            blocks_from_miss = complex_rotation_blocks_from_z_batch(
                deltas_miss,
                order=order,
                basis="local",
                dtype=dtype,
            )
        else:
            raise ValueError(
                "grouped operator cache currently supports rotation='cached' or 'solidfmm'"
            )

        for miss_pos, class_idx in enumerate(missing_indices):
            key_vals = keys_np[class_idx]
            class_key = tuple(int(v) for v in key_vals.tolist())
            blocks = (blocks_to_miss[miss_pos], blocks_from_miss[miss_pos])
            _operator_cache_put(
                (int(order), str(rotation), str(dtype), class_key), blocks
            )
            blocks_to_host[class_idx], blocks_from_host[class_idx] = blocks

    blocks_to = jnp.stack([b for b in blocks_to_host if b is not None], axis=0)
    blocks_from = jnp.stack([b for b in blocks_from_host if b is not None], axis=0)
    return blocks_to, blocks_from


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
        contribs = jnp.where(valid[:, None], contribs, 0)
        accum_chunk = jax.ops.segment_sum(contribs, tgt_chunk, total_nodes)
        return local_accum + accum_chunk, None

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
    class_offsets = np.asarray(jax.device_get(grouped.class_offsets), dtype=np.int64)
    class_targets = np.asarray(jax.device_get(grouped.class_targets), dtype=np.int64)
    if class_offsets.size <= 1:
        empty = jnp.zeros((0,), dtype=INDEX_DTYPE)
        empty_matrix = jnp.zeros((0, int(chunk_size)), dtype=INDEX_DTYPE)
        return empty, empty, empty, empty_matrix, empty_matrix, empty_matrix

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
        return (
            jnp.asarray(starts, dtype=INDEX_DTYPE),
            jnp.asarray(lengths, dtype=INDEX_DTYPE),
            jnp.asarray(class_ids, dtype=INDEX_DTYPE),
            empty_matrix,
            empty_matrix,
            empty_matrix,
        )

    return (
        jnp.asarray(starts, dtype=INDEX_DTYPE),
        jnp.asarray(lengths, dtype=INDEX_DTYPE),
        jnp.asarray(class_ids, dtype=INDEX_DTYPE),
        jnp.asarray(np.stack(sort_permutation, axis=0), dtype=INDEX_DTYPE),
        jnp.asarray(np.stack(group_ids, axis=0), dtype=INDEX_DTYPE),
        jnp.asarray(np.stack(unique_targets, axis=0), dtype=INDEX_DTYPE),
    )


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
    if int(src_sorted.shape[0]) <= int(chunk_size):
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


@partial(jax.jit, static_argnames=("order",))
def _l2l_real_batch_kernel(
    coeffs: Array,
    deltas: Array,
    *,
    order: int,
) -> Array:
    """Vectorized real-basis L2L translation kernel."""
    return jax.vmap(lambda c, d: l2l_real(c, d, order=order))(coeffs, deltas)


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
        contribs = jnp.where(valid[:, None], contribs, 0)
        accum_chunk = jax.ops.segment_sum(contribs, tgt_chunk, total_nodes)
        return local_accum + accum_chunk, None

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


@partial(jax.jit, static_argnames=("order", "total_nodes"))
def _propagate_spherical_locals_to_children(
    coeffs_local: Array,
    centers_local: Array,
    left_child: Array,
    right_child: Array,
    *,
    order: int,
    total_nodes: int,
) -> Array:
    """Apply spherical-basis L2L translations from parents to children."""
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
    translated = jnp.where(valid[:, None], translated, 0)
    updates = jax.ops.segment_sum(translated, safe_child_idx, total_nodes)
    return coeffs_local + updates


def _prepare_spherical_downward_sweep(
    tree: RadixTree,
    upward: TreeUpwardData,
    *,
    theta: float,
    mac_type: MACType,
    initial_locals: Optional[LocalExpansionData] = None,
    interactions: Optional[NodeInteractionList] = None,
    m2l_chunk_size: Optional[int] = None,
    traversal_config: Optional[DualTreeTraversalConfig] = None,
    dense_buffers: Optional[DenseInteractionBuffers] = None,
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
    dehnen_radius_scale: float = 1.0,
) -> TreeDownwardData:
    """Prepare M2L accumulation into spherical local buffers.

    This is a correctness-first implementation that computes per-pair
    spherical M2L contributions using :func:`m2l_a6_dehnen` and
    accumulates them into per-node packed real tesseral local buffers.
    """

    # Build interactions if missing.
    if interactions is None:
        interactions = build_well_separated_interactions(
            tree,
            upward.geometry,
            theta=theta,
            mac_type=mac_type,
            dehnen_radius_scale=dehnen_radius_scale,
            traversal_config=traversal_config,
            retry_logger=retry_logger,
        )

    p = int(upward.multipoles.order)
    centers = jnp.asarray(upward.multipoles.centers)
    total_nodes = int(centers.shape[0])
    coeff_count = sh_size(p)

    # Initialize local buffers (packed real tesseral layout)
    dtype = centers.dtype
    if initial_locals is not None:
        locals_coeffs = jnp.asarray(initial_locals.coefficients)
        if locals_coeffs.shape != (total_nodes, coeff_count):
            raise ValueError("initial_locals must match spherical layout")
    else:
        locals_coeffs = jnp.zeros((total_nodes, coeff_count), dtype=dtype)

    src = jnp.asarray(interactions.sources, dtype=INDEX_DTYPE)
    tgt = jnp.asarray(interactions.targets, dtype=INDEX_DTYPE)

    pair_count = int(src.shape[0])
    if pair_count == 0:
        empty_locals = LocalExpansionData(
            order=p,
            centers=centers,
            coefficients=locals_coeffs,
        )
        return TreeDownwardData(
            interactions=interactions,
            locals=empty_locals,
        )

    multip_packed = jnp.asarray(upward.multipoles.packed)

    # Gather per-pair multipoles and deltas
    src_mult = multip_packed[src]
    deltas = centers[tgt] - centers[src]  # delta: source->target

    contribs = _m2l_real_batch_kernel(src_mult, deltas, order=p)

    # Accumulate per-node local contributions using segment_sum for better
    # fusion.
    locals_accum = jax.ops.segment_sum(contribs, tgt, total_nodes)
    locals_updated = locals_coeffs + locals_accum

    num_internal_nodes = int(tree.num_internal_nodes)
    if num_internal_nodes > 0:
        left_child = jnp.asarray(
            tree.left_child[:num_internal_nodes], dtype=INDEX_DTYPE
        )
        right_child = jnp.asarray(
            tree.right_child[:num_internal_nodes],
            dtype=INDEX_DTYPE,
        )
        locals_updated = _propagate_spherical_locals_to_children(
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

    return TreeDownwardData(
        interactions=interactions,
        locals=locals_after,
    )


def _prepare_solidfmm_downward_sweep(
    tree: RadixTree,
    upward: TreeUpwardData,
    *,
    theta: float,
    mac_type: MACType,
    initial_locals: Optional[LocalExpansionData] = None,
    interactions: Optional[NodeInteractionList] = None,
    m2l_chunk_size: Optional[int] = None,
    l2l_chunk_size: Optional[int] = None,
    complex_rotation: str = "solidfmm",
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
    dehnen_radius_scale: float = 1.0,
) -> TreeDownwardData:
    """Prepare M2L accumulation into solidfmm complex local buffers."""

    if interactions is None:
        interactions = build_well_separated_interactions(
            tree,
            upward.geometry,
            theta=theta,
            mac_type=mac_type,
            dehnen_radius_scale=dehnen_radius_scale,
            traversal_config=traversal_config,
            retry_logger=retry_logger,
        )

    p = int(upward.multipoles.order)
    centers = jnp.asarray(upward.multipoles.centers)
    total_nodes = int(centers.shape[0])
    coeff_count = sh_size(p)

    dtype = complex_dtype_for_real(centers.dtype)
    if initial_locals is not None:
        locals_coeffs = jnp.asarray(initial_locals.coefficients)
        if locals_coeffs.shape != (total_nodes, coeff_count):
            raise ValueError("initial_locals must match solidfmm layout")
    else:
        locals_coeffs = jnp.zeros((total_nodes, coeff_count), dtype=dtype)

    src = jnp.asarray(interactions.sources, dtype=INDEX_DTYPE)
    tgt = jnp.asarray(interactions.targets, dtype=INDEX_DTYPE)

    pair_count = int(src.shape[0])
    if pair_count == 0:
        empty_locals = LocalExpansionData(
            order=p,
            centers=centers,
            coefficients=locals_coeffs,
        )
        return TreeDownwardData(
            interactions=interactions,
            locals=empty_locals,
        )

    multip_packed = jnp.asarray(upward.multipoles.packed, dtype=dtype)

    rotation_mode = str(complex_rotation).strip().lower()
    if rotation_mode not in ("bdz", "cached", "wigner", "solidfmm"):
        raise ValueError(
            "complex_rotation must be 'bdz', 'cached', 'wigner', or 'solidfmm'"
        )

    chunk_size = 4096 if m2l_chunk_size is None else int(m2l_chunk_size)
    if chunk_size <= 0:
        raise ValueError("m2l_chunk_size must be positive")

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
                locals_coeffs,
                multip_packed,
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
                multip_packed,
                centers,
                grouped,
                order=p,
                rotation=rotation_mode,
                total_nodes=total_nodes,
                chunk_size=chunk_size,
            )
    else:
        if pair_count <= chunk_size:
            locals_updated = _accumulate_solidfmm_m2l_fullbatch(
                locals_coeffs,
                multip_packed,
                centers,
                src,
                tgt,
                order=p,
                rotation=rotation_mode,
                total_nodes=total_nodes,
            )
        else:
            locals_updated = _accumulate_solidfmm_m2l_chunked_scan(
                locals_coeffs,
                multip_packed,
                centers,
                src,
                tgt,
                order=p,
                rotation=rotation_mode,
                total_nodes=total_nodes,
                chunk_size=chunk_size,
            )

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
        locals_updated = _propagate_solidfmm_locals_to_children(
            locals_updated,
            centers,
            left_child,
            right_child,
            order=p,
            rotation=rotation_mode,
            total_nodes=total_nodes,
        )

    locals_after = LocalExpansionData(
        order=p,
        centers=centers,
        coefficients=locals_updated,
    )

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
    tree: RadixTree,
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
            precomputed_source_leaf_ids if use_precomputed else None
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
    tree: RadixTree,
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
    if order > MAX_MULTIPOLE_ORDER and expansion_basis not in ("spherical", "solidfmm"):
        raise NotImplementedError(
            "orders above 4 require expansion_basis='spherical' or 'solidfmm'",
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

    # If spherical basis, evaluate using pure-real spherical-harmonic locals.
    if expansion_basis == "spherical":
        p = int(order)

        # real_harmonics expects delta = center - eval_point
        offsets_sph = centers[:, None, :] - leaf_positions
        offsets_sph = jnp.where(valid[..., None], offsets_sph, 0.0)

        def evaluate_leaf_sph(
            coeffs_leaf: Array,
            offsets_leaf: Array,
            mask_leaf: Array,
        ) -> tuple[Array, Array]:
            """Evaluate local expansion at particles using pure-real harmonics."""

            # Use the pure-real L2P evaluation from real_harmonics
            def eval_single(offset: Array) -> tuple[Array, Array]:
                return evaluate_local_real_with_grad(coeffs_leaf, offset, order=p)

            grads, values = jax.vmap(eval_single)(offsets_leaf)
            grads = jnp.where(mask_leaf[..., None], grads, 0.0)
            values = jnp.where(mask_leaf, values, 0.0)
            return grads, values

        grad_field, potentials = jax.vmap(evaluate_leaf_sph)(
            coeffs,
            offsets_sph,
            valid,
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

    if expansion_basis == "solidfmm":
        p = int(order)

        # Complex solidfmm expects delta = center - eval_point (same as real)
        offsets_complex = centers[:, None, :] - leaf_positions
        offsets_complex = jnp.where(valid[..., None], offsets_complex, 0.0)

        def evaluate_leaf_complex(
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

        grad_field, potentials = jax.vmap(evaluate_leaf_complex)(
            coeffs,
            offsets_complex,
            valid,
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

    def evaluate_leaf(
        coeffs_leaf: Array,
        offsets_leaf: Array,
        mask_leaf: Array,
    ) -> tuple[Array, Array]:
        coeff_dtype = coeffs_leaf.dtype

        def phi_fn(vec: Array) -> Array:
            total = coeff_dtype.type(0.0)
            for level_idx in range(order + 1):
                start_idx = level_offset(level_idx)
                combos = LOCAL_LEVEL_COMBOS[level_idx]
                end_idx = start_idx + len(combos)
                coeff_slice = coeffs_leaf[start_idx:end_idx]
                for combo_idx, combo in enumerate(combos):
                    factor = coeff_dtype.type(LOCAL_COMBO_INV_FACTORIAL[combo])
                    term = multi_power(vec, combo)
                    total = total + coeff_slice[combo_idx] * term * factor
            return total

        value_and_grad_fn = jax.value_and_grad(phi_fn)
        values, grads = jax.vmap(value_and_grad_fn)(offsets_leaf)
        grads = jnp.where(mask_leaf[..., None], grads, 0.0)
        values = jnp.where(mask_leaf, values, 0.0)
        return grads, values

    grad_field, potentials = jax.vmap(evaluate_leaf)(coeffs, offsets, valid)

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
