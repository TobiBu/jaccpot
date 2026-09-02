"""
Fast Multipole Method (FMM) for computing gravitational accelerations.

This implementation uses multipole and local expansions to compute
gravitational forces in O(N) time instead of O(N^2) for direct summation.

RE-EXPORTS. Eleven of the imports below are unused *here* and are not dead. Each
carries ``# noqa: F401`` so an ``--select=F401`` sweep leaves it alone:

    NearfieldInteropData                     _bucket_far_pairs_by_level_split
    _PrepareStateTreeUpwardArtifacts         _build_nearfield_interop_data
    _build_tree_with_config                  _prepare_solidfmm_downward_sweep
    _evaluate_local_expansions_for_particles build_interactions_and_neighbors
    adaptive_pair_policy                     sh_size
    enforce_conjugate_symmetry_batch

They are reached in THREE ways, and a sweep has to account for all three -- this
cleanup removed ``_build_tree_with_config`` on the strength of the first two and
the suite caught it:

1. ``from ._fmm_impl import X``            -- greppable
2. ``_fmm_impl.X`` on a module alias       -- greppable
3. ``mock.patch.object(alias, "X", ...)``  -- the name is a STRING, and the call
   is usually split across lines, so no single-line grep finds it. Only an AST
   walk over ``patch.object``/``setattr`` calls does.

``__all__`` would be the tidier mechanism and is DELIBERATELY NOT USED here.
``runtime/fmm/__init__.py`` does ``from .._fmm_impl import *``; with no
``__all__`` that pulls every public name, and adding one narrows the star to
whatever the list happens to contain. Tried during this cleanup: it broke
``jaccpot.runtime.fmm.FMMPreparedState`` and, through it, ``import jaccpot``.
Any future ``__all__`` here has to enumerate the whole star surface, not just
the re-exports.

The ``noqa`` markers are less precise than they look, and that is a known cost
rather than an oversight: isort hoists a trailing comment from the name onto the
``from ... import (`` line and re-merges duplicate imports from one module, so
three of the ten end up suppressing F401 for their whole import group. A name
that later goes unused inside one of those groups will not be flagged. Splitting
them apart does not survive isort -- it re-merges on the next run.

Reaching internals through this module is a habit worth breaking rather than
extending -- ``examples/benchmark_gpu_radix_worker.py`` did it for thirteen
symbols and silently broke on six of them when ``kernels/core.py`` was split,
because an attribute lookup fails at call time rather than at import. New
callers should import from the module that defines the symbol.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Literal, Optional, Union

import jax
from beartype import beartype
from beartype.typing import Callable, Tuple
from jaxtyping import Array, DTypeLike, jaxtyped
from yggdrax.interactions import (  # noqa: F401
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    build_interactions_and_neighbors,
)
from yggdrax.tree import Tree, TreeType, available_tree_types

from jaccpot.config import (
    FarFieldConfig,
    FMMPreset,
    MACTypeInput,
    MemoryObjective,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)
from jaccpot.downward.local_expansions import LocalExpansionData
from jaccpot.operators.complex_ops import (  # noqa: F401
    enforce_conjugate_symmetry_batch,
)
from jaccpot.operators.real_harmonics import sh_size  # noqa: F401

from ._adaptive_policy import adaptive_pair_policy  # noqa: F401
from ._interaction_cache import _InteractionCacheEntry, _RefreshDualPlannerHint
from ._large_n_types import LargeNPreparedState
from .fmm_autotune import AutotuneMixin
from .fmm_caches import (
    _clear_global_runtime_caches,
    _m2l_autotune_payload,
    _restore_m2l_autotune_payload,
)
from .fmm_constants import (
    _GROUPED_SCHEDULE_BUDGET_DEFAULT,
    _LARGE_N_GPU_UPWARD_LEAF_BATCH_SIZE,
)
from .fmm_derivatives import DerivativesMixin
from .fmm_diagnostics import DiagnosticsMixin
from .fmm_evaluate import EvaluateMixin
from .fmm_overrides import (
    OverridesMixin,
    normalize_traversal_config_request,
    warn_full_traversal_config_replacement,
)
from .fmm_policy import PolicyMixin
from .fmm_prepare import PrepareMixin
from .fmm_presets import FMMPresetConfig, get_preset_config
from .fmm_state import (  # noqa: F401
    FMMPreparedState,
    FMMResolvedConfig,
    _bucket_far_pairs_by_level_split,
    _build_tree_with_config,
    _GeometryReuseEntry,
    _normalize_strict_refresh_diag_mode,
    _PrepareStateTreeUpwardArtifacts,
    _resolve_fmm_config,
    _strict_refresh_diag_stage_flags,
    _TopologyReuseEntry,
)
from .fmm_strict_cap_profile import StrictCapProfileMixin
from .fmm_strict_run import StrictRunMixin
from .fmm_sweeps import SweepsMixin
from .kernels.core import (  # noqa: F401
    ExpansionBasis,
    NearfieldInteropData,
    _build_nearfield_interop_data,
    _evaluate_local_expansions_for_particles,
    _normalize_strict_refresh_detail_diag_mode,
    _prepare_solidfmm_downward_sweep,
)
from .reference import compute_gravitational_potential as reference_compute_potential

# RE-EXPORTS. The names below are imported for other modules to reach through
# this one, and are unused *here*. Holding references makes that a fact the
# interpreter can see: `pyflakes` counts them as used, so whatever it still
# reports for this module is a real dead import rather than noise. A tuple of
# STRINGS would not do it, and neither does `# noqa` -- pyflakes ignores noqa,
# and isort hoists a trailing comment onto the whole import group, so a name that
# later goes unused inside that group is never flagged. Verified, 2026-08-18.
_REEXPORTS = (
    NearfieldInteropData,
    _PrepareStateTreeUpwardArtifacts,
    _bucket_far_pairs_by_level_split,
    _build_nearfield_interop_data,
    _build_tree_with_config,
    _evaluate_local_expansions_for_particles,
    _prepare_solidfmm_downward_sweep,
    adaptive_pair_policy,
    build_interactions_and_neighbors,
    enforce_conjugate_symmetry_batch,
    sh_size,
)

FarFieldMode = Literal["auto", "pair_grouped", "class_major"]
NearFieldMode = Literal["auto", "baseline", "bucketed"]
JerkMode = Literal["fast_approx", "accurate"]
PreparedStateLike = Union["FMMPreparedState", LargeNPreparedState]


def derive_split_build_default(
    *,
    memory_objective: str,
    backend: str,
    tree_type: str,
    expansion_basis: str,
    streamed_far_pairs: bool,
) -> bool:
    """Whether the low-peak split dual-tree build is the default for this config.

    Pulled out of :meth:`FMMEngine._resolve_derived_lane_flags` so it can be
    evaluated twice against different field values, and so the five conjuncts are
    testable without a GPU. ``backend`` is a parameter rather than a call to
    :func:`jax.default_backend` for exactly that second reason.

    **This is evaluated twice on purpose.** The first evaluation happens while the
    caller's configuration is still in force; the second, in
    :meth:`FMMEngine._apply_large_n_gpu_production_contract`, after that contract
    has coerced ``memory_objective`` and ``streamed_far_pairs`` to the values the
    ``large_n_gpu`` preset requires. Without the second, a caller who passes an
    ``advanced=`` config alongside the preset gets the predicate computed from
    their config's *defaults* -- ``streamed_far_pairs=None`` becoming ``False`` --
    and the preset silently runs the monolithic dual-tree build it exists to
    avoid. Measured consequence: an N=1e7 census OOMed on a 4.77 GiB allocation
    inside ``_dual_tree_build_raw``.

    Parameters
    ----------
    memory_objective : str
        Resolved memory objective; only ``"minimum_memory"`` selects the split build.
    backend : str
        JAX default backend name. The split build is a GPU-only default.
    tree_type : str
        Resolved tree type; the split build exists for ``"radix"``.
    expansion_basis : str
        Resolved expansion basis; paired with ``"solidfmm"`` on this lane.
    streamed_far_pairs : bool
        Whether far pairs are streamed rather than materialised.

    Returns
    -------
    bool
        ``True`` when every conjunct holds, i.e. the split build is the default.
    """

    return bool(
        memory_objective == "minimum_memory"
        and backend == "gpu"
        and tree_type == "radix"
        and expansion_basis == "solidfmm"
        and bool(streamed_far_pairs)
    )


class FMMEngine(
    PrepareMixin,
    EvaluateMixin,
    StrictRunMixin,
    SweepsMixin,
    OverridesMixin,
    AutotuneMixin,
    PolicyMixin,
    DerivativesMixin,
    StrictCapProfileMixin,
    DiagnosticsMixin,
):
    """Fast Multipole Method engine for gravitational N-body simulations.

    The concrete engine every mixin in ``runtime/`` is mixed into, and the object
    :class:`~jaccpot.solver.FMMSolver` constructs. Almost every keyword below is a
    lane or sizing knob rather than a physics choice: the ones that change the
    computed numbers are ``theta``, ``G``, ``softening``, ``working_dtype``,
    ``expansion_basis``, the ``mac_*``/``adaptive_*`` family and the order pins.
    The rest bound memory or pick between implementations that compute the same
    thing.

    Constructor arguments are documented here rather than on ``__init__`` because
    pydoclint runs with ``allow-init-docstring`` off.

    **Resolution order is load-bearing.** ``__init__`` delegates to a sequence of
    ``_resolve_*`` / ``_init_*`` helpers that mutate ``self`` in place, and each
    was extracted verbatim and left in its original position for that reason
    (commits b462e45, dee46d6). Reordering them changes what the engine resolves
    to. A later contract can also override what a caller asked for:
    :meth:`_apply_large_n_gpu_production_contract` forces ``runtime_path`` to
    ``"large_n"`` and clears ``precompute_nearfield_scatter_schedules`` and
    ``mixed_order_farfield``.

    Parameters
    ----------
    theta : float
        Opening angle for the multipole acceptance criterion; typically 0.5-1.0.
    G : float
        Gravitational constant.
    softening : float
        Plummer softening length, which bounds the near-field pair force at short
        separation.
    working_dtype : Optional[DTypeLike]
        Dtype the sweeps compute in. ``None`` resolves against the device later
        rather than here.
    expansion_basis : ExpansionBasis
        Expansion algebra. ``"cartesian"`` or ``"solidfmm"``; ``"complex"`` is
        accepted as an alias and normalised to ``"solidfmm"``.
    basis_impl : Optional[Any]
        The basis object, when one exists. ``None`` for bases the runtime implements
        internally rather than through the interface.
    m2l_impl : Optional[str]
        M2L translation implementation. ``None`` means "let the basis decide", which
        selects ``"rot_scale"`` for the real basis.
    adaptive_order : bool
        Choose the expansion order per interaction from ``p_gears`` instead of using
        ``max_order`` everywhere.
    p_gears : Optional[tuple[int, ...]]
        Candidate orders for ``adaptive_order``, smallest passing one wins. Ignored
        when ``adaptive_order`` is ``False``.
    use_pallas : Optional[bool]
        Near-field kernel selection; ``None`` auto-detects from the device.
    reuse_topology : bool
        Reuse the tree across calls instead of rebuilding every time. Trades
        accuracy for speed as the particles drift from the tree they were binned
        into.
    rebuild_every : int
        With ``reuse_topology``, rebuild after this many calls. Must be positive.
    mac_force_scale_mode : str
        How the per-node force scale entering Dehnen's eq (16a) is obtained:
        ``"prev"`` (default) reuses the previous full evaluation's accelerations,
        ``"paper"``/``"paper_fb"`` run a prepass, and the ``*_cached`` variants
        require a cached scale rather than computing one.
    mac_force_scale_prepass_theta : Optional[float]
        Opening angle for the force-scale prepass's own traversal. ``None`` takes
        the default. Only consulted by the prepass modes.
    mac_force_scale_fb_inflation : float
        Inflates the far-field source-to-target distance so the eq (16b) scale stays
        a strict lower bound.
    adaptive_error_model : str
        Error estimator for adaptive acceptance: ``"tail_proxy"`` (default),
        ``"dehnen_degree"``, or ``"dehnen_paper"`` -- the last being the eq (15)
        estimator, which also switches the node force-scale reduction from max to
        min.
    adaptive_eps : Optional[float]
        Relative force-accuracy target of eq (16a). ``None`` takes the default,
        which a paper-style MAC rejects -- see ``Raises``.
    dehnen_geometry_mode : str
        How node centres and radii entering the Dehnen MAC are measured: ``"com"``
        (default), ``"exact"``, ``"tree"``, ``"tree_approx"`` or ``"runtime"``. Some
        modes run a host loop over nodes and warn.
    mac_theta_max : float
        Upper clamp on the opening angle the adaptive MAC may choose.
    mac_type : MACTypeInput
        Multipole acceptance criterion, e.g. ``"bh"`` or ``"dehnen"``. Lives here
        rather than in a group because it straddles traversal and accuracy.
    dehnen_radius_scale : float
        Scale applied to node radii in the Dehnen MAC.
    interaction_retry_logger : Optional[Callable[[DualTreeRetryEvent], None]]
        Called once per dual-tree retry, when the traversal overflows its pair
        capacity and re-runs with a larger one. Purely observational -- use it to
        detect that ``traversal_config`` is undersized.
    use_dense_interactions : Optional[bool]
        Materialize the interaction list densely rather than as a compact list --
        faster for small trees, quadratic in memory for large ones.
    runtime_path : Literal['auto', 'large_n']
        ``"auto"`` or ``"large_n"``. Forcing ``"large_n"`` selects the memory-lean
        lane regardless of particle count.
    preset : Optional[Union[str, FMMPreset]]
        Named configuration bundle applied before the individual keywords above. An
        enum member or its string value.
    fixed_order : Optional[int]
        Pin the expansion order, bypassing preset and adaptive selection.
    fixed_max_leaf_size : Optional[int]
        Pin the maximum leaf size, bypassing preset selection.

    farfield : Optional[FarFieldConfig]
        The far-field group: grouping mode, rotation, the M2L/L2L chunk sizes,
        streamed far pairs, mixed-order settings and gradient retention.
        ``None`` means ``FarFieldConfig()``.

        Three names differ (``mode``, ``rotation``, ``mixed_order``) and one
        **default** does: ``FarFieldConfig.rotation`` is ``None`` where this
        constructor defaulted to ``"solidfmm"``. ``None`` means "not
        overridden", so it is resolved at the unpack -- the same resolution the
        facade already performs at ``solver.py:445-447``. This is the second of
        the four groups where the field-by-field default check caught a
        difference the names alone would not have shown.
    tree : Optional[TreeConfig]
        Tree construction as one group: type, build mode, leaf target, and the
        local-refinement knobs. ``None`` means ``TreeConfig()``.

        Two names differ (``mode`` for ``tree_build_mode``, ``leaf_target`` for
        ``target_leaf_particles``) and, uniquely among the groups so far, one
        **default** differs: ``TreeConfig.tree_type`` is ``None`` where this
        constructor defaulted to ``"radix"``. ``None`` there means "not
        overridden", so the default is resolved at the unpack -- the same
        ``or "radix"`` the facade already applies. Without that, every caller
        omitting the group would silently get ``tree_type=None``. The field-by-
        field default check across the mapping is what caught it.
    nearfield : Optional[NearFieldConfig]
        The near-field trio as one group: mode, edge chunk size and whether the
        scatter schedules are precomputed. ``None`` means ``NearFieldConfig()``,
        whose three defaults were checked against the flat parameters this
        replaced before the swap. Note the names differ -- the class drops the
        redundant ``nearfield_`` prefix -- so this is a rename mapping rather
        than the straight pass-through ``runtime_policy`` gets; the mapping is
        stated in full where the group is unpacked. Fields are documented on
        :class:`jaccpot.config.NearFieldConfig`.
    runtime_policy : Optional[RuntimePolicyConfig]
        The seventeen execution-policy knobs, as one frozen group: backend and
        host-refine mode, ``fail_fast``, the memory objective and budget, the
        traversal capacities and overrides, the cache/retention flags, the
        autotune switch and the schedule budgets. ``None`` means
        ``RuntimePolicyConfig()``, whose defaults were checked field by field
        against the flat parameters this replaced -- all seventeen matched, so
        omitting it is exactly the old behaviour. Each field is documented on
        :class:`jaccpot.config.RuntimePolicyConfig`, which is the public
        vocabulary the facade already builds; this signature used to flatten it
        back out into seventeen keywords, and that duplication is what audit F09
        called unreviewable.

    Raises
    ------
    ValueError
        If any option above is outside its documented domain. The checks are
        spread across the ``_resolve_*`` helpers, so the message names the
        offending knob rather than pointing here. ``__init__`` itself now
        contains no ``raise`` at all -- every validation lives in a helper --
        which is why the ``def`` line below carries ``noqa: DOC502``. The
        section stays because it is accurate for a caller: constructing with a
        bad value does raise. The suppression is targeted at this one class
        rather than turning ``--skip-checking-raises`` on repo-wide, which
        would stop the check everywhere to fix it in one place.
    """

    def __init__(  # noqa: DOC502
        self: "FMMEngine",
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
        mac_force_scale_prepass_theta: Optional[float] = None,
        mac_force_scale_fb_inflation: float = 1.0,
        adaptive_error_model: str = "tail_proxy",
        adaptive_eps: Optional[float] = None,
        dehnen_geometry_mode: str = "com",
        mac_theta_max: float = 1.0,
        # `MACTypeInput`, not yggdrax's `MACType`: this constructor accepts a
        # fourth value, "dehnen_error", which `_base_mac_type()` maps to "dehnen"
        # before the traversal sees it. See the alias.
        mac_type: MACTypeInput = "bh",
        dehnen_radius_scale: float = 1.0,
        # DualTreeTraversalConfig (replace all four capacities), or a
        # TraversalOverrides / mapping of named capacities (merge onto the
        # preset's resolved sizing). See normalize_traversal_config_request.
        interaction_retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
        use_dense_interactions: Optional[bool] = None,
        runtime_path: Literal["auto", "large_n"] = "auto",
        preset: Optional[Union[str, FMMPreset]] = None,
        fixed_order: Optional[int] = None,
        fixed_max_leaf_size: Optional[int] = None,
        farfield: Optional[FarFieldConfig] = None,
        tree: Optional[TreeConfig] = None,
        nearfield: Optional[NearFieldConfig] = None,
        runtime_policy: Optional[RuntimePolicyConfig] = None,
    ):
        # The seventeen execution-policy knobs arrive as one frozen group
        # (audit F09). Unpacked into the same locals the flat parameters used,
        # so every line below this point is untouched and the resolution order
        # is unchanged. RuntimePolicyConfig's defaults were verified field by
        # field against the flat defaults before the swap -- all seventeen
        # matched, so a caller who omits the group gets exactly what it got.
        _policy = RuntimePolicyConfig() if runtime_policy is None else runtime_policy
        # The near-field trio arrives as one group too (audit F09). Unlike the
        # policy group above, the names differ: NearFieldConfig drops the
        # redundant `nearfield_` prefix inside a class already called that. The
        # mapping below is the whole of it, and all three defaults were checked
        # against the flat ones before the swap.
        # The tree group. Two of its names differ (`mode` for `tree_build_mode`,
        # `leaf_target` for `target_leaf_particles`) and one DEFAULT differs, which
        # is the first time the field-by-field check across a mapping has caught
        # something: TreeConfig.tree_type defaults to None where this constructor
        # defaulted to "radix". None there means "not overridden", so the default
        # is resolved here -- the same `or "radix"` the facade already applies at
        # `solver.py`'s construction call. Without this line every caller who omits
        # the group would silently get tree_type=None instead of "radix".
        # The far-field group, last of the four (audit F09). Three names differ
        # (`mode`, `rotation`, `mixed_order`) and, like the tree group, one
        # DEFAULT differs: FarFieldConfig.rotation is None where this constructor
        # defaulted to "solidfmm". Resolved here exactly as the facade already
        # resolves it at `solver.py:445-447` -- None means "not overridden".
        _ff = FarFieldConfig() if farfield is None else farfield
        grouped_interactions = _ff.grouped_interactions
        farfield_mode = _ff.mode
        complex_rotation = "solidfmm" if _ff.rotation is None else _ff.rotation
        m2l_chunk_size = _ff.m2l_chunk_size
        l2l_chunk_size = _ff.l2l_chunk_size
        streamed_far_pairs = _ff.streamed_far_pairs
        mixed_order_farfield = _ff.mixed_order
        mixed_order_min_order = _ff.mixed_order_min_order
        retain_far_pairs_for_grad = _ff.retain_far_pairs_for_grad
        _tree = TreeConfig() if tree is None else tree
        tree_type = "radix" if _tree.tree_type is None else _tree.tree_type
        tree_build_mode = _tree.mode
        target_leaf_particles = _tree.leaf_target
        refine_local = _tree.refine_local
        max_refine_levels = _tree.max_refine_levels
        aspect_threshold = _tree.aspect_threshold
        _nf = NearFieldConfig() if nearfield is None else nearfield
        nearfield_mode = _nf.mode
        nearfield_edge_chunk_size = _nf.edge_chunk_size
        precompute_nearfield_scatter_schedules = _nf.precompute_scatter_schedules
        execution_backend = _policy.execution_backend
        host_refine_mode = _policy.host_refine_mode
        fail_fast = _policy.fail_fast
        memory_objective = _policy.memory_objective
        memory_budget_bytes = _policy.memory_budget_bytes
        max_pair_queue = _policy.max_pair_queue
        pair_process_block = _policy.pair_process_block
        traversal_config = _policy.traversal_config
        enable_interaction_cache = _policy.enable_interaction_cache
        retain_traversal_result = _policy.retain_traversal_result
        retain_interactions = _policy.retain_interactions
        prepare_stage_memory_split_enabled = _policy.prepare_stage_memory_split_enabled
        autotune_m2l_chunk = _policy.autotune_m2l_chunk
        precompute_grouped_class_segments = _policy.precompute_grouped_class_segments
        grouped_schedule_budget_bytes = _policy.grouped_schedule_budget_bytes
        nearfield_schedule_item_cap = _policy.nearfield_schedule_item_cap
        upward_leaf_batch_size = _policy.upward_leaf_batch_size

        self._validate_expansion_family(
            adaptive_order=adaptive_order,
            basis_impl=basis_impl,
            expansion_basis=expansion_basis,
            m2l_impl=m2l_impl,
            p_gears=p_gears,
            use_pallas=use_pallas,
        )
        self._init_recent_topology_counters(
            rebuild_every=rebuild_every,
            reuse_topology=reuse_topology,
        )
        self._resolve_mac_and_adaptive_policy(
            adaptive_eps=adaptive_eps,
            adaptive_error_model=adaptive_error_model,
            complex_rotation=complex_rotation,
            dehnen_geometry_mode=dehnen_geometry_mode,
            farfield_mode=farfield_mode,
            mac_force_scale_fb_inflation=mac_force_scale_fb_inflation,
            mac_force_scale_mode=mac_force_scale_mode,
            mac_force_scale_prepass_theta=mac_force_scale_prepass_theta,
            mac_theta_max=mac_theta_max,
            mixed_order_farfield=mixed_order_farfield,
            mixed_order_min_order=mixed_order_min_order,
            streamed_far_pairs=streamed_far_pairs,
        )
        self._resolve_lane_modes(
            nearfield_mode=nearfield_mode,
            runtime_path=runtime_path,
            execution_backend=execution_backend,
            nearfield_edge_chunk_size=nearfield_edge_chunk_size,
            precompute_nearfield_scatter_schedules=(
                precompute_nearfield_scatter_schedules
            ),
        )
        self._resolve_memory_and_cache_options(
            memory_objective=memory_objective,
            memory_budget_bytes=memory_budget_bytes,
            enable_interaction_cache=enable_interaction_cache,
            retain_traversal_result=retain_traversal_result,
            retain_interactions=retain_interactions,
            prepare_stage_memory_split_enabled=prepare_stage_memory_split_enabled,
            fail_fast=fail_fast,
            autotune_m2l_chunk=autotune_m2l_chunk,
        )
        self._resolve_schedule_budgets(
            grouped_schedule_budget_bytes=grouped_schedule_budget_bytes,
            nearfield_schedule_item_cap=nearfield_schedule_item_cap,
            precompute_grouped_class_segments=precompute_grouped_class_segments,
            upward_leaf_batch_size=upward_leaf_batch_size,
        )
        self._resolve_tree_options(
            dehnen_radius_scale=dehnen_radius_scale,
            host_refine_mode=host_refine_mode,
            tree_type=tree_type,
        )
        preset_config = get_preset_config(preset) if preset is not None else None

        # Split the traversal request before resolution. A full
        # DualTreeTraversalConfig keeps its historical "replace everything"
        # meaning (and warns, because that also replaces the capacities the
        # caller did not intend to touch); a TraversalOverrides/mapping becomes a
        # field-by-field merge applied after the policy has sized for this N, so
        # naming one capacity cannot move the other three.
        traversal_config, traversal_field_overrides = (
            normalize_traversal_config_request(traversal_config)
        )
        self._traversal_field_overrides: dict[str, int] = traversal_field_overrides
        if traversal_config is not None:
            warn_full_traversal_config_replacement(
                supplied=traversal_config,
                preset_name=(
                    str(preset_config.name.value) if preset_config is not None else None
                ),
            )

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

        self._unpack_resolved_config(
            mac_type=mac_type,
            resolved=resolved,
        )
        self._check_dehnen_paper_requirements()
        self._resolve_runtime_defaults(
            resolved=resolved,
            preset_config=preset_config,
            interaction_retry_logger=interaction_retry_logger,
        )
        self._resolve_refresh_and_strict_modes()
        self._resolve_large_n_diag_modes()
        self._init_compiled_lane_caches(
            m2l_chunk_size=m2l_chunk_size,
            l2l_chunk_size=l2l_chunk_size,
            traversal_config=traversal_config,
            max_pair_queue=max_pair_queue,
            pair_process_block=pair_process_block,
            grouped_interactions=grouped_interactions,
            fixed_order=fixed_order,
            fixed_max_leaf_size=fixed_max_leaf_size,
        )
        self._resolve_derived_lane_flags()
        self._resolve_static_sizing_flags(
            retain_far_pairs_for_grad=retain_far_pairs_for_grad,
        )

    def _validate_expansion_family(
        self,
        *,
        adaptive_order: bool,
        basis_impl: Optional[Any],
        expansion_basis: ExpansionBasis,
        m2l_impl: Optional[str],
        p_gears: Optional[tuple[int, ...]],
        use_pallas: Optional[bool],
    ) -> None:
        """Validate the expansion family and resolve the Pallas near-field default.

        Extracted verbatim from ``__init__`` lines 418-449 (audit 2.1 step 3),
        called in the original position so the resolution order is unchanged.

        Parameters
        ----------
        adaptive_order : bool
            Passed through from ``__init__`` unchanged.
        basis_impl : Optional[Any]
            Passed through from ``__init__`` unchanged.
        expansion_basis : ExpansionBasis
            Passed through from ``__init__`` unchanged.
        m2l_impl : Optional[str]
            Passed through from ``__init__`` unchanged.
        p_gears : Optional[tuple[int, ...]]
            Passed through from ``__init__`` unchanged.
        use_pallas : Optional[bool]
            Passed through from ``__init__`` unchanged.

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.

        Raises
        ------
        ValueError
            If the expansion basis is not 'cartesian' or 'solidfmm'.
        """
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

    def _init_recent_topology_counters(
        self,
        *,
        rebuild_every: int,
        reuse_topology: bool,
    ) -> None:
        """Zero the most-recent-topology counters and the reuse/rebuild policy.

        Extracted verbatim from ``__init__`` lines 450-469 (audit 2.1 step 3),
        called in the original position so the resolution order is unchanged.

        Parameters
        ----------
        rebuild_every : int
            Passed through from ``__init__`` unchanged.
        reuse_topology : bool
            Passed through from ``__init__`` unchanged.

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.

        Raises
        ------
        ValueError
            If ``rebuild_every`` is not positive.
        """
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

    def _resolve_mac_and_adaptive_policy(
        self,
        *,
        adaptive_eps: Optional[float],
        adaptive_error_model: str,
        complex_rotation: str,
        dehnen_geometry_mode: str,
        farfield_mode: FarFieldMode,
        mac_force_scale_fb_inflation: Optional[float],
        mac_force_scale_mode: str,
        mac_force_scale_prepass_theta: Optional[float],
        mac_theta_max: float,
        mixed_order_farfield: bool,
        mixed_order_min_order: Optional[int],
        streamed_far_pairs: Optional[bool],
    ) -> None:
        """Resolve the MAC family, the adaptive-order error model and mixed order.

        Extracted verbatim from ``__init__`` lines 470-529 (audit 2.1 step 3),
        called in the original position so the resolution order is unchanged.

        Parameters
        ----------
        adaptive_eps : Optional[float]
            Passed through from ``__init__`` unchanged.
        adaptive_error_model : str
            Passed through from ``__init__`` unchanged.
        complex_rotation : str
            Passed through from ``__init__`` unchanged.
        dehnen_geometry_mode : str
            Passed through from ``__init__`` unchanged.
        farfield_mode : FarFieldMode
            Passed through from ``__init__`` unchanged.
        mac_force_scale_fb_inflation : Optional[float]
            Passed through from ``__init__`` unchanged. Inflates the far-field
            source-to-target distance so the eq (16b) scale is a strict lower
            bound; ``None`` takes the default.
        mac_force_scale_mode : str
            Passed through from ``__init__`` unchanged.
        mac_force_scale_prepass_theta : Optional[float]
            Passed through from ``__init__`` unchanged. Opening angle for the
            force-scale prepass's own traversal; ``None`` takes the default.
        mac_theta_max : float
            Passed through from ``__init__`` unchanged.
        mixed_order_farfield : bool
            Passed through from ``__init__`` unchanged.
        mixed_order_min_order : Optional[int]
            Passed through from ``__init__`` unchanged.
        streamed_far_pairs : Optional[bool]
            Passed through from ``__init__`` unchanged.

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.

        Raises
        ------
        ValueError
            If a MAC/adaptive option is outside its documented domain, or a paper-style MAC is given a non-positive ``adaptive_eps``.
        """
        force_scale_mode_norm = str(mac_force_scale_mode).strip().lower()
        if force_scale_mode_norm not in (
            "prev",
            "prepass",
            "paper",
            "paper_cached",
            "paper_fb",
            "paper_fb_cached",
        ):
            raise ValueError(
                "mac_force_scale_mode must be 'prev', 'prepass', 'paper', "
                "'paper_cached', 'paper_fb', or 'paper_fb_cached'"
            )
        self.mac_force_scale_mode = force_scale_mode_norm
        self.mac_force_scale_prepass_theta = (
            None
            if mac_force_scale_prepass_theta is None
            else float(mac_force_scale_prepass_theta)
        )
        if self.mac_force_scale_prepass_theta is not None and not (
            0.0 < self.mac_force_scale_prepass_theta <= 1.0
        ):
            raise ValueError("mac_force_scale_prepass_theta must be in (0, 1]")
        self.mac_force_scale_fb_inflation = float(mac_force_scale_fb_inflation)
        if self.mac_force_scale_fb_inflation < 0.0:
            raise ValueError("mac_force_scale_fb_inflation must be >= 0")
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
        if dehnen_geometry_mode_norm not in (
            "com",
            "exact",
            "tree",
            "tree_approx",
            "runtime",
        ):
            raise ValueError(
                "dehnen_geometry_mode must be 'com', 'exact', 'tree', "
                "'tree_approx', or 'runtime'"
            )
        self.dehnen_geometry_mode = dehnen_geometry_mode_norm
        self.mac_theta_max = float(mac_theta_max)
        if not (0.0 < self.mac_theta_max <= 1.0):
            raise ValueError("mac_theta_max must be in (0, 1]")
        self.adaptive_eps = None if adaptive_eps is None else float(adaptive_eps)
        if self.adaptive_eps is not None and self.adaptive_eps <= 0.0:
            raise ValueError("adaptive_eps must be > 0 when provided")
        self._last_force_scale_nodes: Optional[Array] = None
        #: Per-particle force scale from the most recent eq (16b) prepass, in sorted
        #: (tree) order. Diagnostic only -- the criterion consumes the node-reduced
        #: array above. It exists so the estimator can be scored against an exact
        #: O(N^2) f_b without re-running the prepass, which is how the O(N) estimate
        #: was validated (see bench/validation/fb_estimator_fidelity.py).
        self._last_force_scale_particles: Optional[Array] = None
        self._in_force_scale_prepass = False
        #: Per-node effective opening angles from the most recent prepare_state under
        #: mac_type='dehnen_theta'. Diagnostic only -- the traversal consumes them as
        #: rescaled geometry.radius, not from here.
        self._recent_effective_theta_nodes: Optional[Array] = None

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

    def _resolve_lane_modes(
        self,
        *,
        nearfield_mode: str,
        runtime_path: str,
        execution_backend: str,
        nearfield_edge_chunk_size: int,
        precompute_nearfield_scatter_schedules: bool,
    ) -> None:
        """Normalise and validate the three lane-selection strings.

        Extracted verbatim from ``__init__`` (audit **F09**) and called in the
        original position, so the resolution order is unchanged. Characterised
        first by ``tests/unit/runtime/test_engine_config_resolution.py``.

        ``_explicit_nearfield_mode`` records whether the caller named a value, so
        the policy layer can tell "left at auto" from "asked for auto". It is
        computed as ``!= "auto"``, which means naming ``"auto"`` reads as *not*
        explicit -- carried over unchanged, and pinned by that test.

        Parameters
        ----------
        nearfield_mode : str
            Passed through from ``__init__`` unchanged.
        runtime_path : str
            Passed through from ``__init__`` unchanged.
        execution_backend : str
            Passed through from ``__init__`` unchanged.
        nearfield_edge_chunk_size : int
            Passed through from ``__init__`` unchanged.
        precompute_nearfield_scatter_schedules : bool
            Passed through from ``__init__`` unchanged.

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.

        Raises
        ------
        ValueError
            If any lane string is unrecognised, or the edge chunk size is not
            positive -- same messages as before the extraction.
        """
        nearfield_mode_norm = str(nearfield_mode).strip().lower()
        if nearfield_mode_norm not in ("auto", "baseline", "bucketed"):
            raise ValueError("nearfield_mode must be 'auto', 'baseline', or 'bucketed'")
        runtime_path_norm = str(runtime_path).strip().lower()
        if runtime_path_norm not in ("auto", "large_n"):
            raise ValueError("runtime_path must be 'auto' or 'large_n'")
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

    def _resolve_memory_and_cache_options(
        self,
        *,
        memory_objective: str,
        memory_budget_bytes: Optional[int],
        enable_interaction_cache: bool,
        retain_traversal_result: bool,
        retain_interactions: bool,
        prepare_stage_memory_split_enabled: Optional[bool],
        fail_fast: bool,
        autotune_m2l_chunk: bool,
    ) -> None:
        """Resolve the memory objective, retention flags and the strict-lane pair.

        Extracted verbatim from ``__init__`` (audit **F09**), called in the
        original position.

        ``fail_fast`` and ``autotune_m2l_chunk`` are resolved **here, together,
        and in this order** on purpose: the second reads ``self.fail_fast`` set
        by the first, because a timing-driven chunk search inside the lane whose
        purpose is to fail rather than adapt would be a contradiction. Splitting
        them across two helpers, or calling them the other way round, leaves the
        autotune silently on under ``fail_fast`` and nothing else in the
        constructor notices. That is the "resolution order" sensitivity F09
        flags, and it is pinned in both directions by
        ``tests/unit/runtime/test_engine_config_resolution.py``.

        Parameters
        ----------
        memory_objective : str
            Passed through from ``__init__`` unchanged.
        memory_budget_bytes : Optional[int]
            Passed through from ``__init__`` unchanged.
        enable_interaction_cache : bool
            Passed through from ``__init__`` unchanged.
        retain_traversal_result : bool
            Passed through from ``__init__`` unchanged.
        retain_interactions : bool
            Passed through from ``__init__`` unchanged.
        prepare_stage_memory_split_enabled : Optional[bool]
            Passed through from ``__init__`` unchanged. ``None`` means "let the
            policy decide" and stays distinct from ``False``.
        fail_fast : bool
            Passed through from ``__init__`` unchanged.
        autotune_m2l_chunk : bool
            Passed through from ``__init__`` unchanged.

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.

        Raises
        ------
        ValueError
            If the objective is unrecognised, or a supplied memory budget is not
            positive -- same messages as before the extraction.
        """
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

    def _resolve_schedule_budgets(
        self,
        *,
        grouped_schedule_budget_bytes: Optional[int],
        nearfield_schedule_item_cap: Optional[int],
        precompute_grouped_class_segments: Optional[bool],
        upward_leaf_batch_size: Optional[int],
    ) -> None:
        """Resolve the grouped/near-field schedule budgets and the upward batch size.

        Extracted verbatim from ``__init__`` lines 571-597 (audit 2.1 step 3),
        called in the original position so the resolution order is unchanged.

        Parameters
        ----------
        grouped_schedule_budget_bytes : Optional[int]
            Passed through from ``__init__`` unchanged.
        nearfield_schedule_item_cap : Optional[int]
            Passed through from ``__init__`` unchanged.
        precompute_grouped_class_segments : Optional[bool]
            Passed through from ``__init__`` unchanged.
        upward_leaf_batch_size : Optional[int]
            Passed through from ``__init__`` unchanged.

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.

        Raises
        ------
        ValueError
            If a schedule budget, item cap or batch size is not positive.
        """
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

    def _resolve_tree_options(
        self,
        *,
        dehnen_radius_scale: float,
        host_refine_mode: str,
        tree_type: str,
    ) -> None:
        """Resolve the Dehnen radius scale, host refinement and tree type.

        Extracted verbatim from ``__init__`` lines 598-617 (audit 2.1 step 3),
        called in the original position so the resolution order is unchanged.

        Parameters
        ----------
        dehnen_radius_scale : float
            Passed through from ``__init__`` unchanged.
        host_refine_mode : str
            Passed through from ``__init__`` unchanged.
        tree_type : str
            Passed through from ``__init__`` unchanged.

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.

        Raises
        ------
        ValueError
            If ``host_refine_mode`` or ``tree_type`` is outside its documented domain.
        """
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

    def _unpack_resolved_config(
        self,
        *,
        mac_type: MACTypeInput,
        resolved: FMMResolvedConfig,
    ) -> None:
        """Unpack the resolved config bundle onto self and pick the force-scale mode.

        Extracted verbatim from ``__init__`` lines 657-671 (audit 2.1 step 3),
        called in the original position so the resolution order is unchanged.

        Parameters
        ----------
        mac_type : MACTypeInput
            Passed through from ``__init__`` unchanged.
        resolved : FMMResolvedConfig
            Passed through from ``__init__`` unchanged.

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.
        """
        self.config = resolved
        self.preset = resolved.preset
        self.theta = resolved.theta
        self.mac_type = mac_type
        if self._uses_per_node_effective_theta():
            warnings.warn(
                "mac_type='dehnen_theta' is a refuted experiment retained only so "
                "its negative result stays reproducible. Measured against the exact "
                "criterion at N=4096/p=8: 12-9300x worse force error at 1.35-15x "
                "more interaction work (p99.99 reached 2.3e+02 on bulge+halo). A "
                "per-node opening angle cannot carry eq (16a), whose acceptance "
                "needs a product of a source and a sink term where the traversal "
                "test is a sum. Use mac_type='dehnen_error' for the exact criterion.",
                FutureWarning,
                stacklevel=2,
            )
        if self._uses_dehnen_error_policy():
            if self.adaptive_error_model == "tail_proxy":
                self.adaptive_error_model = "dehnen_paper"
            if self.mac_force_scale_mode == "prev":
                # 'prev' means "reuse the last force scale", which in paper mode is
                # exactly 'paper_cached' -- a low-order prepass on the cold call,
                # the cached scale after that. Dehnen §5.4 licenses the reuse
                # ("only very slightly worse" than the exact a_b), and 'paper'
                # re-runs the full prepass on *every* prepare_state, which costs
                # ~3.5x steady state. Keep 'paper' for whoever asks for it.
                self.mac_force_scale_mode = "paper_cached"

    def _check_dehnen_paper_requirements(self) -> None:
        """Reject a paper-style MAC that was given no explicit accuracy target.

        Extracted verbatim from ``__init__`` lines 672-692 (audit 2.1 step 3),
        called in the original position so the resolution order is unchanged.

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.

        Raises
        ------
        ValueError
            If a paper-style MAC was requested without an explicit ``adaptive_eps``.
        """
        # Dehnen eq (16a) is parameterised by a relative force-accuracy target
        # `eps`, not by an opening angle: acceptance is gated only by the error
        # test plus eq (16a)'s own `theta < 1` convergence guard, so `theta` has
        # no effect on which pairs are accepted. Falling back to the tail-proxy
        # heuristic `theta**(p+2)` therefore silently invents an eps that has
        # nothing to do with the criterion -- at theta=0.6, p=4 it is 0.047, a
        # 4.7% per-interaction tolerance, two to three decades looser than the
        # range Dehnen works in. Require it explicitly instead.
        if (
            self._uses_paper_style_traversal_policy()
            and not self.adaptive_order
            and self.adaptive_eps is None
        ):
            raise ValueError(
                "the Dehnen paper MAC (mac_type='dehnen_error' or "
                "adaptive_error_model='dehnen_paper') requires an explicit "
                "adaptive_eps: it is the relative force-accuracy target of "
                "eq (16a). The theta-derived default is a tail_proxy heuristic "
                "and is far too loose here. Note that theta itself does not "
                "gate acceptance in this mode."
            )

    def _resolve_runtime_defaults(
        self,
        *,
        resolved: FMMResolvedConfig,
        preset_config: Optional[FMMPresetConfig],
        interaction_retry_logger: Optional[Callable[[DualTreeRetryEvent], None]],
    ) -> None:
        """Resolve G/softening/dtype, jit defaults, chunk sizes, traversal and tree build.

        Extracted verbatim from ``__init__`` lines 692-806 (audit 2.1 step 2).
        Called in the original position, so the resolution order is unchanged --
        that order is load-bearing (b462e45, dee46d6).

        Parameters
        ----------
        resolved : FMMResolvedConfig
            Passed through from ``__init__`` unchanged.
        preset_config : Optional[FMMPresetConfig]
            Passed through from ``__init__`` unchanged.
        interaction_retry_logger : Optional[Callable[[DualTreeRetryEvent], None]]
            Passed through from ``__init__`` unchanged.

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.
        """
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
        self._refresh_timing_evaluate_seconds: float = 0.0

    def _resolve_refresh_and_strict_modes(self) -> None:
        """Resolve the refresh-timing, dual-planner and strict-lane execution modes.

        Extracted verbatim from ``__init__`` lines 807-901 (audit 2.1 step 2).
        Called in the original position, so the resolution order is unchanged --
        that order is load-bearing (b462e45, dee46d6).

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.
        """
        # Whether the M2L/L2L substage timers actually ran. They cost a device
        # sync per substage, so they are conditional -- and a conditional timer
        # that reports 0.0 when it did not run is indistinguishable from a stage
        # that was free. Surfaced as refresh_substages_measured.
        self._refresh_timing_substages_measured: bool = False
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

    def _resolve_large_n_diag_modes(self) -> None:
        """Resolve the large-N fused defaults and every diagnostic-mode env switch.

        Extracted verbatim from ``__init__`` lines 902-1003 (audit 2.1 step 2).
        Called in the original position, so the resolution order is unchanged --
        that order is load-bearing (b462e45, dee46d6).

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.
        """
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

    def _init_compiled_lane_caches(
        self,
        *,
        m2l_chunk_size: Optional[int],
        l2l_chunk_size: Optional[int],
        traversal_config: Optional[DualTreeTraversalConfig],
        max_pair_queue: Optional[int],
        pair_process_block: Optional[int],
        grouped_interactions: Optional[bool],
        fixed_order: Optional[int],
        fixed_max_leaf_size: Optional[int],
    ) -> None:
        """Initialise the compiled-lane caches and record which knobs the caller set.

        Extracted verbatim from ``__init__`` lines 1004-1022 (audit 2.1 step 2).
        Called in the original position, so the resolution order is unchanged --
        that order is load-bearing (b462e45, dee46d6).

        Parameters
        ----------
        m2l_chunk_size : Optional[int]
            Passed through from ``__init__`` unchanged.
        l2l_chunk_size : Optional[int]
            Passed through from ``__init__`` unchanged.
        traversal_config : Optional[DualTreeTraversalConfig]
            Passed through from ``__init__`` unchanged.
        max_pair_queue : Optional[int]
            Passed through from ``__init__`` unchanged.
        pair_process_block : Optional[int]
            Passed through from ``__init__`` unchanged.
        grouped_interactions : Optional[bool]
            Passed through from ``__init__`` unchanged.
        fixed_order : Optional[int]
            Passed through from ``__init__`` unchanged.
        fixed_max_leaf_size : Optional[int]
            Passed through from ``__init__`` unchanged.

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.
        """
        # Compiled radix fast-lane acceleration evaluates, keyed by the
        # Python constants the traced body closes over (jax.jit keys on the
        # pytree structure and avals itself). See
        # _large_n_pipeline._large_n_fastlane_eval_fn.
        self._large_n_fastlane_eval_jit_cache: dict[tuple[Any, ...], Any] = {}
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

    def _resolve_derived_lane_flags(self) -> None:
        """Derive the two cross-cutting lane flags from the already-resolved config.

        Extracted verbatim from ``__init__`` lines 1023-1035 (audit 2.1 step 2).
        Called in the original position, so the resolution order is unchanged --
        that order is load-bearing (b462e45, dee46d6).

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.
        """
        self._streamed_minimum_memory_gpu_default_split_build: bool = (
            derive_split_build_default(
                memory_objective=self.memory_objective,
                backend=jax.default_backend(),
                tree_type=self.tree_type,
                expansion_basis=self.expansion_basis,
                streamed_far_pairs=self.streamed_far_pairs,
            )
        )
        self._large_n_gpu_production_profile_cached: bool = (
            str(self.preset).strip().lower() == "large_n_gpu"
            and str(self.tree_type).strip().lower() == "radix"
            and str(self.expansion_basis).strip().lower() == "solidfmm"
            and str(self.execution_backend).strip().lower() != "octree"
        )

    def _resolve_static_sizing_flags(
        self,
        *,
        retain_far_pairs_for_grad: bool,
    ) -> None:
        """Resolve static runtime sizing, grad far-pair retention and fast-lane centres.

        Extracted verbatim from ``__init__`` lines 1036-1069 (audit 2.1 step 2).
        Called in the original position, so the resolution order is unchanged --
        that order is load-bearing (b462e45, dee46d6).

        Parameters
        ----------
        retain_far_pairs_for_grad : bool
            Passed through from ``__init__`` unchanged.

        Returns
        -------
        None
            Mutates ``self`` in place, exactly as the inlined code did.
        """
        self._static_runtime_fixed_sizing: bool = str(
            os.environ.get("JACCPOT_STATIC_RUNTIME_FIXED_SIZING", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Retain the frozen M2L pair list (``compact_far_pairs``) on the prepared
        # state so a gradient path can re-run the downward sweep against it.
        # ``prepare_state`` builds those pairs whenever the streamed-compact policy
        # holds -- which the canonical large-N production config satisfies -- but
        # then discards them (see fmm_prepare, _PrepareStateDualDownwardArtifacts),
        # because retaining them costs ~24 B/pair and the large-N preset targets
        # ``memory_objective="minimum_memory"``. Without them a differentiable
        # large-N seam has nothing to re-run: the far field would be treated as a
        # constant, which is a SILENTLY WRONG gradient (zero mass-sensitivity
        # through P2M/M2M/M2L/L2L) rather than an error. Default OFF so the
        # production forward and its memory profile are untouched.
        self.retain_far_pairs_for_grad: bool = bool(retain_far_pairs_for_grad) or str(
            os.environ.get("JACCPOT_RETAIN_FAR_PAIRS_FOR_GRAD", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Opt-in geometric (box/aabb) centres for the real-basis large-N fast lane,
        # decoupled from grouped_interactions. This selects center_mode="aabb" in the
        # production fast lane (see _resolve_runtime_execution_overrides) for the real
        # (Dehnen) basis only, without engaging grouped_interactions (so the streamed
        # pair_grouped near/far payload is unchanged). Default OFF: the fast lane keeps
        # its data-dependent COM centres. This is enabling infrastructure -- box centres
        # quantise far-field displacements into interaction classes, a prerequisite for
        # class-cached far-field kernels. NB such caching is only effective on a REGULAR
        # cell grid (e.g. a uniform octree); the radix binary tree's box centres do not
        # grid-quantise, so class-cached grouping does NOT transfer to this lane
        # (measured: the distinct-class count explodes past any fixed capacity at
        # production N). On its own the knob is speed-neutral and ~2x looser (still
        # ~1e-3 in forces) vs COM.
        self._fastlane_geometric_centers: bool = str(
            os.environ.get("JACCPOT_LARGE_N_FASTLANE_GEOMETRIC_CENTERS", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._apply_large_n_gpu_production_contract()

    def _is_large_n_gpu_production_profile(self) -> bool:
        """Whether this solver should run the canonical large-N GPU contract.

        All four conditions must hold: the ``large_n_gpu`` preset, a radix tree,
        the solidfmm basis and a non-octree backend. A cached attribute, when
        present, overrides the live check.

        Returns
        -------
        bool
            ``True`` when the profile matches and the contract in
            :meth:`_apply_large_n_gpu_production_contract` should be applied.
        """
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
            self.upward_leaf_batch_size = _LARGE_N_GPU_UPWARD_LEAF_BATCH_SIZE

        # Re-derive the split-build default from the fields this contract has just
        # coerced. `_resolve_derived_lane_flags` ran BEFORE them, so it saw whatever
        # the caller supplied: on the bare preset that is already correct, but a
        # caller who also passes `advanced=FMMAdvancedConfig(...)` replaces the
        # preset's config with their own, whose defaults are
        # `memory_objective="balanced"` and `streamed_far_pairs=None` -> False. The
        # predicate then came out False and this preset silently ran the monolithic
        # dual-tree build it exists to avoid. Note `streamed_far_pairs` is the
        # load-bearing conjunct, not `memory_objective`: setting only the latter in
        # an advanced config left the predicate False.
        self._streamed_minimum_memory_gpu_default_split_build = (
            derive_split_build_default(
                memory_objective=self.memory_objective,
                backend=jax.default_backend(),
                tree_type=self.tree_type,
                expansion_basis=self.expansion_basis,
                streamed_far_pairs=self.streamed_far_pairs,
            )
        )

    def _resolve_execution_backend(self) -> str:
        """Resolve the active FMM execution backend without altering tree choice.

        Returns
        -------
        str
            The configured backend, with ``"auto"`` resolved to ``"radix"``.
            Deliberately does not touch ``tree_type`` -- backend and tree family
            are independent choices, and only the ``"octree"`` backend constrains
            the tree (see :meth:`_ensure_execution_backend_supported`).
        """
        if self.execution_backend == "auto":
            return "radix"
        return self.execution_backend

    def _ensure_execution_backend_supported(
        self, *, tree: Optional[Tree] = None
    ) -> str:
        """Validate execution backends that are available for the current tree.

        Only ``"octree"`` has requirements; every other backend returns
        immediately.

        Parameters
        ----------
        tree : Optional[Tree]
            Tree whose ``tree_type`` is checked. ``None`` falls back to the
            engine's configured ``tree_type``.

        Returns
        -------
        str
            The validated backend name.

        Raises
        ------
        ValueError
            If the ``"octree"`` backend is paired with a non-octree tree.
        NotImplementedError
            If the ``"octree"`` backend is paired with a basis other than
            ``"solidfmm"``.
        """
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
        self: "FMMEngine",
    ) -> Tuple[DualTreeRetryEvent, ...]:
        """Return retry telemetry collected during the latest build."""

        return self._recent_retry_events

    @property
    def recent_topology_reused(self: "FMMEngine") -> bool:
        """Whether the most recent prepare/evaluate path reused cached topology."""

        return bool(self._recent_topology_reused)

    def clear_prepared_state_cache(self: "FMMEngine") -> None:
        """Clear cached prepared-state payloads used by reuse mode."""

        self._prepared_state_cache_key = None
        self._prepared_state_cache_value = None
        self._prepared_state_cache_positions = None
        self._prepared_state_cache_masses = None
        self._topology_reuse_entry = None
        self._recent_topology_reused = False

    def clear_runtime_caches(
        self: "FMMEngine", *, clear_jax_compilation: bool = False
    ) -> None:
        """Release solver/runtime caches to reduce memory pressure.

        Drops every retained cache, template and diagnostic counter, so the next
        call re-prepares from scratch. Numerically inert -- it frees memory and
        costs recompute, and cannot change what a subsequent evaluation returns.

        Parameters
        ----------
        clear_jax_compilation : bool
            Also clear JAX's own compilation cache. Off by default because that
            cache is process-global and shared with everything else in the
            process, so clearing it makes unrelated code recompile too.

        Returns
        -------
        None
            Mutates ``self`` in place.
        """

        self.clear_prepared_state_cache()
        self._locals_template = None
        self._interaction_cache = None
        self._interaction_cache_hits = 0
        self._interaction_cache_misses = 0
        self._tree_workspace = None
        self._last_force_scale_nodes = None
        self._last_force_scale_particles = None
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
        self._refresh_timing_evaluate_seconds = 0.0
        self._refresh_timing_substages_measured = False
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

    def export_m2l_autotune_cache(self: "FMMEngine") -> list[dict[str, Any]]:
        """Return a JSON-serializable snapshot of global M2L autotune results.

        The autotune cache is **process-global**, not per-engine: this reads the
        same table every engine in the process writes to, so the method is on the
        engine for discoverability only.

        Returns
        -------
        list[dict[str, Any]]
            One entry per autotuned configuration, ready for ``json.dump``.
        """

        return _m2l_autotune_payload()

    def import_m2l_autotune_cache(
        self: "FMMEngine",
        payload: list[dict[str, Any]],
        *,
        merge: bool = True,
    ) -> int:
        """Restore global M2L autotune results from serialized payload.

        Writes to the process-global table, so this affects every engine in the
        process, not just this one.

        Parameters
        ----------
        payload : list[dict[str, Any]]
            Entries as produced by :meth:`export_m2l_autotune_cache`.
        merge : bool
            Merge into the existing table rather than replacing it.

        Returns
        -------
        int
            Number of entries restored.
        """

        return _restore_m2l_autotune_payload(payload, merge=bool(merge))

    def save_m2l_autotune_cache(self: "FMMEngine", path: str) -> int:
        """Write global M2L autotune cache to a JSON file.

        Parameters
        ----------
        path : str
            Destination path, overwritten if it exists.

        Returns
        -------
        int
            Number of entries written.
        """

        payload = self.export_m2l_autotune_cache()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return int(len(payload))

    def load_m2l_autotune_cache(
        self: "FMMEngine",
        path: str,
        *,
        merge: bool = True,
    ) -> int:
        """Load global M2L autotune cache from a JSON file.

        Parameters
        ----------
        path : str
            Path to a file written by :meth:`save_m2l_autotune_cache`.
        merge : bool
            Merge into the existing table rather than replacing it.

        Returns
        -------
        int
            Number of entries restored.

        Raises
        ------
        ValueError
            If the file's top-level JSON value is not a list.
        """

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("autotune cache JSON must contain a list payload")
        return self.import_m2l_autotune_cache(payload, merge=bool(merge))

    # ------------------------------------------------------------------
    # Expansion construction up to a given order
    # order=0: monopole, order=1: +dipole, order=2: +quadrupole
    # order=3: +octupole, order=4: +hexadecapole
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Expansion evaluation up to a given order
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Direct summation fallback (for validation / small N)
    # ------------------------------------------------------------------


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
    """Compute gravitational accelerations via the Fast Multipole Method.

    A one-shot convenience wrapper: it builds a fresh :class:`FMMEngine` per call
    and therefore reuses nothing between calls. Construct an engine directly if
    you are evaluating more than once.

    Parameters
    ----------
    positions : Array
        Source and target particle positions.
    masses : Array
        Particle masses aligned with ``positions``.
    theta : float
        Opening angle for the acceptance criterion.
    G : Union[float, Array]
        Gravitational constant.
    softening : Union[float, Array]
        Plummer softening length.
    bounds : Optional[Tuple[Array, Array]]
        Optional explicit domain bounds used during tree construction.
    leaf_size : int
        Target maximum particle count per leaf.
    max_order : int
        Multipole/local expansion order.
    return_potential : bool
        When ``True``, return ``(accelerations, potentials)``.

    Returns
    -------
    Union[Array, Tuple[Array, Array]]
        Accelerations ``[N, 3]``, or ``(accelerations, potentials)`` when
        ``return_potential`` is set.
    """

    fmm = FMMEngine(
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
    """Compute gravitational potential at evaluation points.

    Delegates to the **direct-sum reference** implementation, not to the FMM. It
    is exact and quadratic in the particle count, so it is a correctness oracle
    rather than a fast path -- unlike
    :func:`compute_gravitational_acceleration`, which does run the FMM.

    Parameters
    ----------
    positions : Array
        Source particle positions.
    masses : Array
        Particle masses aligned with ``positions``.
    eval_points : Array
        Points at which to evaluate the potential.
    G : Union[float, Array]
        Gravitational constant.
    softening : Union[float, Array]
        Plummer softening length.

    Returns
    -------
    Array
        Potential at each entry of ``eval_points``.
    """

    return reference_compute_potential(
        positions,
        masses,
        eval_points,
        G=G,
        softening=softening,
    )
