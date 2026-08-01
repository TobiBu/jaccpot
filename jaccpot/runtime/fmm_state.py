"""Orchestrator data + config scaffolding for the FMM runtime.

Extracted from _fmm_impl.py (Phase 2d): the resolved-config dataclasses and
resolution, tree-build artifacts + builders, FMMPreparedState (the pytree
passed between prepare/evaluate) + its artifact NamedTuples and octree
builders, and the strict-refresh diag helpers. Sibling of _fmm_impl at the
runtime level to avoid the fmm/ package-init cycle; depends on kernels +
fmm_constants/fmm_caches + tree/octree helpers, never the engine class.
To be subdivided into fmm/{resolved_config,tree_build,prepared_state}.py once
the engine class is dissolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Tuple
from jaxtyping import Array, DTypeLike
from yggdrax import build_tree
from yggdrax.interactions import (
    CompactTaggedFarPairs,
    CompactTaggedOctreeFarPairs,
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    DualTreeWalkResult,
    NodeInteractionList,
    NodeNeighborList,
)
from yggdrax.tree import Tree

from jaccpot.downward.local_expansions import LocalExpansionData, TreeDownwardData
from jaccpot.upward.tree_expansions import TreeUpwardData

from ..config import FMMPreset
from ._interaction_cache import _InteractionCacheEntry
from ._octree_adapter import OctreeExecutionData
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
from .dtypes import INDEX_DTYPE
from .fmm_presets import FMMPresetConfig
from .kernels.core import (
    ExpansionBasis,
    _FarPairCOO,
    _infer_order_from_coeff_count,
    _max_leaf_size_from_tree,
)

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


class _PrepareStateFarPairPlan(NamedTuple):
    """Far-pair payloads prepared for the downward sweep."""

    far_pairs_by_gear: Optional[tuple[tuple[Array, Array], ...]]
    far_pairs_coo: Optional[_FarPairCOO]
    adaptive_order_for_downward: bool
    p_gears_for_downward: tuple[int, ...]
    recent_far_pairs_by_gear_counts: tuple[int, ...]


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
