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
from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple, Optional, Union

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

if TYPE_CHECKING:  # pragma: no cover - annotations only, no runtime import
    # `FMMPreparedState.nearfield_interop` is annotated with a string forward
    # reference to keep this module off `kernels.core`'s import path at module scope.
    # The name was dangling before this block, so `typing.get_type_hints` on the
    # dataclass raised NameError.
    from .kernels.core import NearfieldInteropData

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
    """Complete a velocity-Verlet step after the endpoint force is known.

    The second half of a kick-drift-kick step: the position uses only the
    starting acceleration, while the velocity averages the two endpoints. That
    average is what makes the scheme second-order and symplectic, and it is why
    this cannot run until the force at the new position is known.

    Parameters
    ----------
    state : Array
        Packed state ``[N, 2, 3]``: positions on index 0 of axis 1, velocities on
        index 1. Its dtype governs the whole step -- the other arguments are cast
        to it.
    acceleration_current : Array
        Acceleration at the starting positions ``[N, 3]``.
    acceleration_new : Array
        Acceleration at the new positions ``[N, 3]``.
    dt : Array
        Timestep.

    Returns
    -------
    Array
        Updated state, same shape and dtype as ``state``.
    """
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
    """Resolved configuration controlling tree construction.

    Attributes
    ----------
    mode : str
        Build mode: ``"lbvh"``, ``"fixed_depth"``, ``"adaptive"`` or
        ``"static_radix"``.
    target_leaf_particles : int
        Leaf occupancy target. At least 1.
    refine_local : bool
        Run the local refinement pass after the initial build.
    max_refine_levels : int
        Extra refinement levels permitted; ``0`` disables refinement even when
        ``refine_local`` is set.
    aspect_threshold : float
        Node aspect ratio above which refinement is attempted.
    """

    mode: str
    target_leaf_particles: int
    refine_local: bool
    max_refine_levels: int
    aspect_threshold: float


@dataclass(frozen=True)
class TraversalExecutionConfig:
    """Resolved configuration for traversal, batching, and dense buffers.

    Attributes
    ----------
    m2l_chunk_size : Optional[int]
        Far-field pairs per M2L chunk; ``None`` lets the runtime choose, and is
        what the autotuner fills in.
    l2l_chunk_size : Optional[int]
        Nodes per L2L chunk; ``None`` as above.
    max_pair_queue : Optional[int]
        Cap on the traversal's pending-pair queue; ``None`` is unbounded.
    pair_process_block : Optional[int]
        Pairs drained from that queue per step; ``None`` lets the runtime choose.
    traversal_config : Optional[DualTreeTraversalConfig]
        Full dual-tree traversal override; ``None`` builds one from the fields
        above.
    use_dense_interactions : bool
        Materialize the interaction list densely rather than as a compact
        list -- faster for small trees, quadratic in memory for large ones.
    jit_tree : Union[bool, Literal['auto']]
        Compile the tree build. ``"auto"`` decides per call, which is why this is
        not a plain ``bool``.
    jit_traversal : bool
        Compile the traversal.
    """

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
    """Container bundling all resolved FMMEngine options.

    What :func:`_resolve_fmm_config` produces: every constructor input, the
    preset's contribution and the built-in defaults collapsed into one value with
    no ``None``-means-decide-later left in the scalars.

    Attributes
    ----------
    theta : float
        Opening angle.
    G : float
        Gravitational constant.
    softening : float
        Plummer softening length.
    working_dtype : Optional[DTypeLike]
        Working dtype; still ``None`` when the caller did not pin one, since that
        is resolved against the device later.
    tree : TreeBuilderConfig
        Resolved tree-construction settings.
    traversal : TraversalExecutionConfig
        Resolved traversal, batching and dense-buffer settings.
    preset : Optional[str]
        Name of the preset these came from, for diagnostics; ``None`` when no
        preset was used.
    """

    theta: float
    G: float
    softening: float
    working_dtype: Optional[DTypeLike]
    tree: TreeBuilderConfig
    traversal: TraversalExecutionConfig
    preset: Optional[str]


@dataclass(frozen=True)
class _TreeBuildArtifacts:
    """Outputs from a tree construction pass used by the FMM pipeline.

    Attributes
    ----------
    tree : Tree
        The built tree.
    positions_sorted : Array
        Particle positions ``[N, 3]`` in the tree's Morton order.
    masses_sorted : Array
        Particle masses ``[N]`` in the same order.
    inverse_permutation : Array
        Maps sorted back to original particle order.
    workspace : Optional[object]
        Reusable builder scratch, returned so a later build can skip
        reallocating. ``None`` on the fast LBVH path, which allocates none.
    max_leaf_size : int
        Widest leaf occupancy actually produced.
    cache_leaf_parameter : int
        The leaf-size *request* this build was made with, as opposed to
        ``max_leaf_size`` which is the outcome. Cache keys use this one, so two
        builds that asked for the same thing match even if the trees differ.
    """

    tree: Tree
    positions_sorted: Array
    masses_sorted: Array
    inverse_permutation: Array
    workspace: Optional[object]
    max_leaf_size: int
    cache_leaf_parameter: int


@dataclass(frozen=True)
class _TopologyReuseCandidate:
    """Candidate topology signature derived from current particle Morton order.

    The lookup side of topology reuse: built from the *current* particles and
    compared against the cached :class:`_TopologyReuseEntry`.

    Attributes
    ----------
    key : str
        Signature of the Morton ordering. Equal keys mean the particles bin the
        same way, which is what makes the cached tree still valid.
    sorted_indices : Array
        The Morton permutation this signature came from.
    sorted_codes : Optional[Array]
        The Morton codes themselves, when they were computed.
    bounds : Optional[Tuple[Array, Array]]
        Domain the codes were computed against. Part of the identity: the same
        permutation under different bounds is not the same topology.
    """

    key: str
    sorted_indices: Array
    sorted_codes: Optional[Array] = None
    bounds: Optional[Tuple[Array, Array]] = None


@dataclass(frozen=True)
class _TopologyReuseEntry:
    """Cached topology metadata for bounded multi-step reuse.

    The stored side of topology reuse; ``key`` is matched against a
    :class:`_TopologyReuseCandidate`.

    Attributes
    ----------
    key : str
        Topology signature this entry was stored under.
    tree : Tree
        The cached tree.
    max_leaf_size : int
        Widest leaf occupancy of ``tree``.
    cache_leaf_parameter : int
        Leaf-size request the tree was built with.
    reuse_count : int
        Times this entry has been reused. What makes the reuse *bounded*:
        ``rebuild_every`` compares against this, so a tree cannot be reused
        indefinitely as the particles drift away from it.
    """

    key: str
    tree: Tree
    max_leaf_size: int
    cache_leaf_parameter: int
    reuse_count: int


@dataclass(frozen=True)
class _GeometryReuseEntry:
    """Cached tree geometry keyed by topology signature and input identity.

    Attributes
    ----------
    key : tuple[Any, ...]
        Composite key: the topology signature plus the identity of the arrays the
        geometry was measured from. Identity, not value -- mutating an array in
        place and passing it again hits this entry.
    geometry : Any
        The cached node centres and radii. ``Any`` to keep this module from
        importing the geometry types.
    """

    key: tuple[Any, ...]
    geometry: Any


class _RuntimeExecutionOverrides(NamedTuple):
    """Resolved runtime execution knobs after adaptive policy decisions.

    What the adaptive policy hands back for one call. These override the
    engine's standing configuration for that call only.

    Attributes
    ----------
    traversal_config : Optional[DualTreeTraversalConfig]
        Traversal override; ``None`` keeps the engine's.
    m2l_chunk_size : Optional[int]
        M2L chunk override; ``None`` keeps the engine's.
    l2l_chunk_size : Optional[int]
        L2L chunk override; ``None`` keeps the engine's.
    grouped_interactions : bool
        Whether to group interactions during traversal.
    farfield_mode : str
        Resolved far-field interaction mode.
    center_mode : str
        How node centres are measured for the acceptance test.
    refine_local_override : Optional[bool]
        Forces local refinement on or off; ``None`` keeps the tree config's.
    adaptive_applied : bool
        Whether the policy actually changed anything. ``False`` means every field
        above is the engine's own value, so a caller can skip the override path
        entirely.
    """

    traversal_config: Optional[DualTreeTraversalConfig]
    m2l_chunk_size: Optional[int]
    l2l_chunk_size: Optional[int]
    grouped_interactions: bool
    farfield_mode: str
    center_mode: str
    refine_local_override: Optional[bool]
    adaptive_applied: bool


def _resolve_optional(value: Any, preset_value: Any, fallback: Any) -> Any:
    """Pick explicit value, then preset value, then fallback.

    The three-tier resolution rule used throughout :func:`_resolve_fmm_config`.
    ``None`` is the only "not supplied" signal, so this cannot resolve an option
    whose ``None`` is a meaningful value -- those are handled explicitly there.

    Typed ``Any`` rather than with a type variable: nothing requires the three
    arguments to share a type, and callers do pass a ``None`` fallback, so
    binding them together would assert an invariant the code does not enforce.

    Parameters
    ----------
    value : Any
        The explicitly requested value, or ``None``.
    preset_value : Any
        The preset's value, or ``None``.
    fallback : Any
        The built-in default. Returned as-is, including when it is ``None``.

    Returns
    -------
    Any
        The first of the three that is not ``None``, else ``fallback``.
    """
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
    """Normalize constructor inputs into a validated runtime configuration.

    Three-tier resolution throughout, via :func:`_resolve_optional`: an explicit
    argument wins, then the preset's value, then the built-in default. Validation
    happens after resolution, so an invalid *default* would be caught too.

    Parameters
    ----------
    theta : float
        Opening angle.
    G : float
        Gravitational constant.
    softening : float
        Plummer softening length.
    working_dtype : Optional[DTypeLike]
        Working dtype; passed through, not resolved here.
    tree_build_mode : Optional[str]
        Tree build mode; defaults to ``"lbvh"``.
    target_leaf_particles : Optional[int]
        Leaf occupancy target; defaults to 32.
    refine_local : Optional[bool]
        Run local refinement.
    max_refine_levels : Optional[int]
        Extra refinement levels permitted.
    aspect_threshold : Optional[float]
        Aspect ratio above which refinement is attempted.
    m2l_chunk_size : Optional[int]
        Far-field pairs per M2L chunk.
    l2l_chunk_size : Optional[int]
        Nodes per L2L chunk.
    max_pair_queue : Optional[int]
        Cap on the traversal's pending-pair queue.
    pair_process_block : Optional[int]
        Pairs drained from that queue per step.
    traversal_config : Optional[DualTreeTraversalConfig]
        Full traversal override.
    use_dense_interactions : Optional[bool]
        Materialize interactions densely.
    preset_config : Optional[FMMPresetConfig]
        The preset's contribution -- the middle tier. ``None`` means every
        resolution falls straight from the explicit argument to the default.

    Returns
    -------
    FMMResolvedConfig
        The validated configuration.

    Raises
    ------
    ValueError
        If ``tree_build_mode`` names no known mode, ``target_leaf_particles`` is
        below 1, or ``jit_tree`` is neither a bool nor ``"auto"``.
    """
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
    """Construct a tree according to the resolved builder configuration.

    Takes the cached jitted LBVH path only when every condition lines up --
    ``jit_tree``, a radix tree, ``"lbvh"`` mode, and no local refinement -- and
    falls back to the general builder otherwise. The fast path allocates no
    workspace, which is why it returns ``None`` for one.

    Parameters
    ----------
    positions : Array
        Particle positions ``[N, 3]`` in the caller's order.
    masses : Array
        Particle masses ``[N]`` in the same order.
    bounds : Tuple[Array, Array]
        Explicit ``(lower, upper)`` domain bounds.
    tree_type : str
        Tree family; only ``"radix"`` admits the fast path.
    tree_config : TreeBuilderConfig
        Resolved builder settings. Its ``mode`` selects the general builder's
        branch.
    leaf_size : int
        Leaf-size request; also the key of the jitted-builder cache, so distinct
        values each cost a compile.
    workspace : Optional[object]
        Reusable builder scratch from a previous build; ``None`` allocates fresh.
    jit_tree : bool
        Compile the build. Already resolved from ``"auto"`` by this point.
    refine_local : bool
        Run local refinement. Any truth here disqualifies the fast path.
    max_refine_levels : int
        Extra refinement levels permitted.
    aspect_threshold : float
        Aspect ratio above which refinement is attempted.

    Returns
    -------
    _TreeBuildArtifacts
        The tree and its sorted particle arrays, permutation, workspace and leaf
        sizes.

    Raises
    ------
    ValueError
        If the requested build mode is not supported for this tree type, or the
        builder returns a tree without the FMM topology the pipeline needs.
    """

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
def _jit_radix_lbvh_builder(leaf_size: int) -> Callable[..., Any]:
    """Return cached jitted radix LBVH builder for a fixed leaf size.

    The leaf size is closed over rather than passed, because it is static to the
    build. That is also why this is cached per leaf size: each distinct value
    costs its own compile, and the ``lru_cache`` is what stops a loop that varies
    the leaf size from recompiling every iteration. The cache holds 16, so more
    than 16 distinct leaf sizes in play will thrash it.

    Parameters
    ----------
    leaf_size : int
        Leaf-size request. Also the cache key, so pass it already normalized.

    Returns
    -------
    Callable[..., Any]
        ``f(positions, masses, bounds) -> (tree, positions_sorted,
        masses_sorted, inverse_permutation)``. Loosely typed because the value is
        a ``jax.jit`` wrapper, not a plain function.
    """

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

    That split is the whole design. Seventeen of the fields below are pytree
    **children**: they are traced, and changing one does not force a recompile.
    The other eight are marked *(static)* -- they are aux data, so they
    participate in the jit cache key and changing one recompiles.
    ``max_leaf_size`` being static is what lets the padded leaf axis have a
    concrete width. The two kinds are interleaved below because pydoclint
    requires declaration order.

    Producing one is not traceable -- tree construction is host-side -- so build
    it with ``prepare_state`` outside any transform, then pass it in.

    Attributes
    ----------
    tree : Tree
        The built tree. Owns the canonical sorted positions, masses and inverse
        permutation, which this class re-exposes as properties rather than
        duplicating.
    upward : Optional[TreeUpwardData]
        Upward-sweep bundle. ``None`` under ``memory_objective="minimum_memory"``,
        which drops it because the plain evaluation path never reads it -- see
        :func:`_prepared_state_upward_payload`.
    downward : TreeDownwardData
        Downward-sweep result carrying the local expansions. Never dropped: this
        is what evaluation contracts at the particles.
    neighbor_list : NodeNeighborList
        Leaf ordering and near-field neighbour lists.
    max_leaf_size : int
        Widest leaf occupancy. *(static)* -- it sets padded shapes.
    input_dtype : jnp.dtype
        Dtype of the caller's arrays. *(static)*, carried as its name.
    working_dtype : jnp.dtype
        Dtype the pipeline computes in. *(static)*, carried as its name.
    expansion_basis : ExpansionBasis
        Which expansion algebra the coefficients are in. *(static)*
    theta : float
        Opening angle this state was prepared at. *(static)*
    topology_key : Optional[str]
        Topology signature, for reuse matching. *(static)*
    interactions : Optional[NodeInteractionList]
        Far-field interaction list; retained only when the runtime was asked to.
    dual_tree_result : Optional[DualTreeWalkResult]
        Full traversal result; likewise opt-in.
    retry_events : Tuple[DualTreeRetryEvent, ...]
        Traversal retries that occurred while preparing. *(static)*, and
        diagnostic only.
    nearfield_interop : Optional[NearfieldInteropData]
        Prebuilt near-field leaf/node view.
    nearfield_target_leaf_ids : Optional[Array]
        Precomputed per-edge target leaf ids.
    nearfield_source_leaf_ids : Optional[Array]
        Precomputed per-edge source leaf ids.
    nearfield_valid_pairs : Optional[Array]
        Precomputed per-edge validity mask.
    nearfield_chunk_sort_indices : Optional[Array]
        Precomputed chunk scatter sort permutation.
    nearfield_chunk_group_ids : Optional[Array]
        Precomputed chunk scatter group ids.
    nearfield_chunk_unique_indices : Optional[Array]
        Precomputed chunk unique-target indices.
    force_scale_nodes : Optional[Array]
        Per-node force scale for the adaptive acceptance test.
    execution_backend : str
        ``"radix"`` or ``"octree"``. *(static)*; selects which of the two
        artifact families evaluation reads.
    octree : Optional[OctreeExecutionData]
        Octree view of ``tree``; ``None`` on the radix backend.
    octree_upward : Optional[OctreeSolidFMMComplexMultipoles]
        Octree-native upward artifacts. Dropped under minimum-memory on the same
        terms as ``upward``.
    octree_downward : Optional[OctreeSolidFMMDownwardPlan]
        Octree-native downward plan.
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
        """Canonical sorted particle positions owned by ``tree``.

        Returns
        -------
        Array
            Positions ``[N, 3]`` in Morton order. Read from ``tree`` rather than
            stored here, so the two cannot drift apart.

        Raises
        ------
        ValueError
            If ``tree`` carries none -- a tree built without FMM topology.
        """
        value = getattr(self.tree, "positions_sorted", None)
        if value is None:
            raise ValueError("prepared tree is missing positions_sorted")
        return jnp.asarray(value)

    @property
    def masses_sorted(self: "FMMPreparedState") -> Array:
        """Canonical sorted particle masses owned by ``tree``.

        Returns
        -------
        Array
            Masses ``[N]`` in Morton order, from ``tree``.

        Raises
        ------
        ValueError
            If ``tree`` carries none.
        """
        value = getattr(self.tree, "masses_sorted", None)
        if value is None:
            raise ValueError("prepared tree is missing masses_sorted")
        return jnp.asarray(value)

    @property
    def inverse_permutation(self: "FMMPreparedState") -> Array:
        """Canonical inverse permutation owned by ``tree``.

        Returns
        -------
        Array
            Index map from Morton order back to the caller's original particle
            order, cast to the index dtype.

        Raises
        ------
        ValueError
            If ``tree`` carries none.
        """
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
    """Tree/upward artifacts produced during prepare_state orchestration.

    The first phase's output, handed to the dual-tree phase. Internal to
    ``prepare_state``; nothing here survives into :class:`FMMPreparedState`
    unchanged.

    Attributes
    ----------
    tree_mode : str
        Build mode actually used, which can differ from the one requested when a
        fallback was taken.
    tree : Tree
        The built tree.
    positions_sorted : Array
        Particle positions ``[N, 3]`` in Morton order.
    masses_sorted : Array
        Particle masses ``[N]`` in the same order.
    inverse_permutation : Array
        Maps sorted back to original particle order.
    leaf_cap : int
        Widest leaf occupancy produced.
    leaf_parameter : int
        Leaf-size request the build was made with; the cache-key half of the
        pair, as in :class:`_TreeBuildArtifacts`.
    topology_key : Optional[str]
        Topology signature, when one was computed.
    upward : TreeUpwardData
        Upward-sweep result.
    locals_template : Optional[LocalExpansionData]
        Zero-filled locals matching the tree's shape, kept so the downward sweep
        can allocate without re-deriving the layout.
    """

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
    """Dual-tree and downward artifacts produced during prepare_state.

    The second phase's output, consuming
    :class:`_PrepareStateTreeUpwardArtifacts`.

    Attributes
    ----------
    interactions : Optional[NodeInteractionList]
        Far-field interaction list; ``None`` when the traversal produced none or
        it was not retained.
    neighbor_list : NodeNeighborList
        Near-field neighbour lists. Always present -- the near field runs
        regardless of what the far field found.
    traversal_result : Optional[DualTreeWalkResult]
        Full traversal result; retained only on request.
    compact_far_pairs : Optional[CompactTaggedFarPairs]
        Far pairs in compact tagged form, when the traversal emitted them.
    downward : TreeDownwardData
        Downward-sweep result carrying the local expansions.
    cache_entry : Optional[_InteractionCacheEntry]
        Interaction-cache entry this phase used or created; ``None`` when the
        cache is off or missed.
    """

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
    """Build octree-native upward artifacts when the execution tree exposes them.

    Parameters
    ----------
    octree : Optional[OctreeExecutionData]
        Octree view of the tree; ``None`` short-circuits to ``None``.
    positions_sorted : Array
        Particle positions ``[N, 3]`` in Morton order.
    masses_sorted : Array
        Particle masses ``[N]`` in the same order.
    expansion_basis : ExpansionBasis
        Only ``"solidfmm"`` has an octree-native upward pass; anything else
        short-circuits.
    max_order : int
        Expansion order ``p``.

    Returns
    -------
    Optional[OctreeSolidFMMComplexMultipoles]
        The octree multipoles, or ``None`` when either precondition fails.
    """

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

    Parameters
    ----------
    upward : TreeUpwardData
        The upward bundle produced by the sweep.
    memory_objective : str
        Only ``"minimum_memory"`` drops the payload; every other value keeps it.

    Returns
    -------
    Optional[TreeUpwardData]
        ``upward``, or ``None`` under minimum memory.
    """

    if str(memory_objective).strip().lower() == "minimum_memory":
        return None
    return upward


def _prepared_state_octree_upward_payload(
    *,
    octree_upward: Optional[OctreeSolidFMMComplexMultipoles],
    memory_objective: str,
) -> Optional[OctreeSolidFMMComplexMultipoles]:
    """Return the octree-upward payload to retain in prepared state.

    The octree counterpart of :func:`_prepared_state_upward_payload`, dropping on
    the same condition and for the same reason.

    Parameters
    ----------
    octree_upward : Optional[OctreeSolidFMMComplexMultipoles]
        The octree upward bundle, if one was built.
    memory_objective : str
        Only ``"minimum_memory"`` drops the payload.

    Returns
    -------
    Optional[OctreeSolidFMMComplexMultipoles]
        ``octree_upward``, or ``None`` under minimum memory.
    """

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
    """Build octree-native downward scaffolding when prepared octree data exists.

    Builds the interaction plan from whichever source is available: native far
    pairs on the octree backend, otherwise the radix interaction list translated
    into octree space. With neither, there is nothing to plan and this returns
    ``None``.

    Parameters
    ----------
    octree : Optional[OctreeExecutionData]
        Octree view of the tree.
    octree_upward : Optional[OctreeSolidFMMComplexMultipoles]
        Octree multipoles; required, since the plan is built against them.
    interactions : Optional[NodeInteractionList]
        Radix interaction list, used when native pairs are unavailable.
    native_far_pairs : Optional[CompactTaggedOctreeFarPairs]
        Octree-native far pairs; preferred, but only on the octree backend.
    execution_backend : str
        Which backend is active; gates the native-pairs branch.

    Returns
    -------
    Optional[OctreeSolidFMMDownwardPlan]
        The downward plan, or ``None`` when the octree data or a pair source is
        missing.
    """

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
    """Run octree-native M2L/L2L when the narrow octree backend is active.

    "Narrow" because every one of five conditions must hold; any miss returns
    ``octree_downward`` untouched rather than raising, so this is safe to call
    unconditionally.

    Parameters
    ----------
    octree : Optional[OctreeExecutionData]
        Octree view of the tree.
    octree_upward : Optional[OctreeSolidFMMComplexMultipoles]
        Octree multipoles, the M2L sources.
    octree_downward : Optional[OctreeSolidFMMDownwardPlan]
        The plan to run, and the value returned unchanged when a condition
        fails.
    expansion_basis : ExpansionBasis
        Must be ``"solidfmm"``.
    execution_backend : str
        Must be ``"octree"``.
    m2l_chunk_size : Optional[int]
        Pairs per M2L chunk; ``None`` takes 4096.

    Returns
    -------
    Optional[OctreeSolidFMMDownwardPlan]
        The plan with M2L accumulated and L2L propagated, or the input plan
        unchanged.
    """

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


def _octree_farfield_eval_inputs(
    state: Any,
) -> tuple[Optional[LocalExpansionData], Optional[Array], Optional[Array]]:
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

    Parameters
    ----------
    state : Any
        A prepared state. Typed ``Any`` because one of the two call sites passes
        ``PreparedStateLike``, so this must accept a ``LargeNPreparedState`` as
        well as an :class:`FMMPreparedState` -- and that union is defined in the
        engine module, which this one must not import (ARCHITECTURE §1). The
        ``getattr`` defaults below are what actually make both safe: a state
        lacking the octree attributes takes the ``(None, None, None)`` branch
        rather than raising.

    Returns
    -------
    tuple[Optional[LocalExpansionData], Optional[Array], Optional[Array]]
        ``(farfield_local_data, farfield_leaf_nodes, farfield_node_ranges)``,
        all three in octree node-id space, or ``(None, None, None)``. All three
        are ``None`` together -- callers may test any one of them.
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
    """Far-pair payloads prepared for the downward sweep.

    Attributes
    ----------
    far_pairs_by_gear : Optional[tuple[tuple[Array, Array], ...]]
        Far pairs bucketed by expansion order, one ``(sources, targets)`` pair
        per gear and aligned with ``p_gears_for_downward``. ``None`` when the
        sweep runs at a single order.
    far_pairs_coo : Optional[_FarPairCOO]
        The same pairs in coordinate form, when that layout was built.
    adaptive_order_for_downward : bool
        Whether the downward sweep should vary its order per pair.
    p_gears_for_downward : tuple[int, ...]
        The gears themselves, in the order ``far_pairs_by_gear`` follows.
    recent_far_pairs_by_gear_counts : tuple[int, ...]
        Pair count per gear, for diagnostics.
    """

    far_pairs_by_gear: Optional[tuple[tuple[Array, Array], ...]]
    far_pairs_coo: Optional[_FarPairCOO]
    adaptive_order_for_downward: bool
    p_gears_for_downward: tuple[int, ...]
    recent_far_pairs_by_gear_counts: tuple[int, ...]


def _empty_interaction_storage_like(
    interactions: Optional[NodeInteractionList],
) -> NodeInteractionList:
    """Return zero-pair interaction storage while preserving node-shaped metadata.

    Zeroes the node-shaped arrays in place of dropping them, and empties only the
    per-pair ones. Keeping the node-shaped shapes is the point: downstream code
    indexes them by node, so a genuinely empty list would change shapes and
    force a recompile where an all-zero one does not.

    Parameters
    ----------
    interactions : Optional[NodeInteractionList]
        The list to derive shapes and dtypes from. Not optional in practice --
        see Raises; the annotation only spares callers a narrowing check.

    Returns
    -------
    NodeInteractionList
        Same node-shaped arrays, zeroed, with zero-length pair arrays that keep
        their original dtypes.

    Raises
    ------
    ValueError
        If ``interactions`` is ``None``, since there is then nothing to take the
        shapes from.
    """

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

    Every failure to split is a graceful one: an inverted order range, missing
    level offsets, too few levels, or offsets that will not transfer to the host
    all return the single-gear form instead of raising. Mixed order is an
    optimisation, so falling back to one order everywhere is always correct.

    Parameters
    ----------
    interactions : NodeInteractionList
        Supplies ``level_offsets``, which is where the split point comes from.
    src_far : Array
        Far-pair source node ids.
    tgt_far : Array
        Far-pair target node ids, aligned with ``src_far``.
    max_order : int
        Order for the coarser levels.
    min_order : int
        Order for the deeper levels. Must be below ``max_order`` for a split to
        happen at all.

    Returns
    -------
    tuple[tuple[int, ...], tuple[tuple[Array, Array], ...]]
        ``(gears, pairs_by_gear)``, aligned elementwise. On a split this is
        ``((min_order, max_order), (deep_pairs, coarse_pairs))`` -- **deeper
        levels first**, matching the ascending gear order rather than the
        coarse-to-fine reading of the summary. Otherwise the single-gear form
        ``((max_order,), ((src_far, tgt_far),))``.
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
