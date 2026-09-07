"""Preset-first configuration model for Jaccpot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

BASIS_DOC = (
    "Preferred production basis is 'real' (Dehnen no-sqrt2 harmonics): the radix "
    "large-N fast lane runs pure-real end to end with no complex<->real "
    "conversion. 'solidfmm'/'complex' are retained for cross-checking only."
)
FARFIELD_MODE_DOC = (
    "For large_n_gpu production, far-field execution is canonicalized to "
    "'pair_grouped'."
)
NEARFIELD_MODE_DOC = (
    "For large_n_gpu production, near-field execution is canonicalized to "
    "'bucketed'."
)
MEMORY_OBJECTIVE_DOC = (
    "For large_n_gpu production, memory objective is canonicalized to "
    "'minimum_memory'."
)

Basis = Literal["cartesian", "solidfmm", "complex", "real"]
FarFieldMode = Literal["auto", "pair_grouped", "class_major"]
NearFieldMode = Literal["auto", "baseline", "bucketed"]
GradNearFieldLane = Literal["auto", "bucketed", "fast_lane"]
#: How the runtime should trade peak memory against throughput when it resolves
#: chunk sizes, streaming and schedule precomputation. ``"balanced"`` is the default;
#: ``"minimum_memory"`` is what ``FMMPreset.LARGE_N_GPU`` canonicalizes to, and is
#: what makes galaxy-scale runs fit. Public: exported from ``jaccpot``.
MemoryObjective = Literal["balanced", "throughput", "minimum_memory"]
FMMExecutionBackend = Literal["auto", "radix", "octree"]

#: Multipole acceptance criteria a **caller** may ask for -- a strict superset of
#: yggdrax's ``MACType``.
#:
#: WHY THIS EXISTS RATHER THAN REUSING ``MACType``. yggdrax declares
#: ``MACType = Literal["bh", "engblom", "dehnen"]``, and the traversal it owns
#: accepts exactly those three. jaccpot adds two more, ``"dehnen_error"`` and
#: ``"dehnen_theta"``: the Dehnen (2014) §5 mass-dependent MAC, which is a
#: *jaccpot-level policy* built on top of the geometric ``"dehnen"`` test. The two
#: differ only in how the criterion is applied -- ``dehnen_error`` installs a
#: solver-owned pair policy, ``dehnen_theta`` folds it into one opening angle per
#: node -- and both reduce to ``"dehnen"`` for the traversal.
#:
#: ``dehnen_theta`` is listed here because it must be *accepted and translated*,
#: not because it is recommended: it is a refuted experiment kept so its negative
#: result stays reproducible, and constructing a solver with it raises a
#: ``FutureWarning`` saying so. Use ``"dehnen_error"`` for the exact criterion.
#: :meth:`~jaccpot.runtime.fmm_policy.PolicyMixin._mac_type_for_traversal`
#: translates them before yggdrax ever sees them, and ``_uses_dehnen_error_policy``
#: is what switches the extra machinery on.
#:
#: So the constructor accepts five values and hands three downstream. Annotating it
#: with yggdrax's narrower ``MACType`` understated the accepted set on a public
#: signature: ``mac_type="dehnen_error"`` is documented in two of the package's own
#: error messages, yet a runtime type check rejected it 68 times (F40). Use this
#: alias for anything a caller passes in, and yggdrax's ``MACType`` for the
#: resolved value going out.
MACTypeInput = Literal["bh", "engblom", "dehnen", "dehnen_error", "dehnen_theta"]


class FMMPreset(str, Enum):
    """User-facing quality/speed presets.

    A ``str`` enum, so ``preset="fast"`` and ``preset=FMMPreset.FAST`` are
    interchangeable everywhere. The four members are frozen by
    ``tests/unit/test_public_api_surface.py``.

    Presets resolve by two different routes, which matters when tracing why a knob
    took the value it did: ``FAST`` and ``LARGE_N_GPU`` resolve through first-class
    bundles in :func:`jaccpot.runtime.fmm_presets.get_preset_config`, while
    ``BALANCED`` and ``ACCURATE`` resolve through advanced-config defaults in
    ``solver._default_advanced_for_preset`` and never reach ``get_preset_config``
    at all. See ARCHITECTURE.md section 6.

    Attributes
    ----------
    FAST :
        ``"fast"``. Lowest accuracy, lowest cost. The package default, and the
        first of the two members that resolve through ``get_preset_config``.
    BALANCED :
        ``"balanced"``. Middle ground; resolves through advanced-config defaults.
    ACCURATE :
        ``"accurate"``. Tightest accuracy settings. What the golden
        characterization suite uses.
    LARGE_N_GPU :
        ``"large_n_gpu"``. Galaxy-scale GPU profile, canonicalized to the
        low-memory streamed fast path (see :class:`RuntimePolicyConfig`).
        Measured at 1M particles: forward 2.5 s, forward+backward 69 s, 11 GB
        peak. It is not "more accurate than ``ACCURATE``": its bundle carries no
        accuracy knob at all, and it shares ``FAST``'s tree settings exactly
        (``lbvh``, 64 particles per leaf, no host-side refinement), differing
        only in traversal capacities and in forcing ``jit_tree``.
    """

    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    LARGE_N_GPU = "large_n_gpu"


@dataclass(frozen=True)
class TreeConfig:
    """Tree-construction overrides for advanced runtime tuning.

    Frozen. Every field defaults to ``None`` meaning "leave the preset's choice
    alone" -- so an instance with one field set overrides exactly that one thing.

    Attributes
    ----------
    tree_type : Optional[str]
        Yggdrax tree family, e.g. ``"radix"`` (the production default) or
        ``"kdtree"``.
    mode : Optional[str]
        Builder selector, ``"lbvh"`` or ``"fixed_depth"``.
    leaf_target : Optional[int]
        Desired particles per leaf for fixed-depth builds. Note this is a *target*;
        the achieved occupancy is padded to a static bound at prepare time.
    refine_local : Optional[bool]
        Enable the host-side refinement pass that splits elongated leaves.
    max_refine_levels : Optional[int]
        Depth cap for that refinement pass.
    aspect_threshold : Optional[float]
        Leaf aspect ratio above which refinement splits a leaf.
    """

    tree_type: Optional[str] = None
    mode: Optional[str] = None
    leaf_target: Optional[int] = None
    refine_local: Optional[bool] = None
    max_refine_levels: Optional[int] = None
    aspect_threshold: Optional[float] = None


@dataclass(frozen=True)
class FarFieldConfig:
    """Far-field interaction and translation-kernel overrides.

    Frozen. ``None`` means "leave the preset's choice alone"; the non-``None``
    defaults below are the actual defaults, not placeholders.

    Attributes
    ----------
    grouped_interactions : Optional[bool]
        Group M2L pairs into displacement classes so one rotation block serves a
        whole class. Requires geometric (not centre-of-mass) expansion centres,
        because the classification quantises pair displacements onto a lattice and
        applies one representative displacement per class.
    mode : FarFieldMode
        ``"auto"``, ``"pair_grouped"`` or ``"class_major"``. Must not be left at
        ``"auto"`` by the time the grouped M2L runs -- an unresolved ``"auto"``
        used to reach the kernel and raise.

        ``"pair_grouped"`` and ``"class_major"`` are two batchings of one
        computation and agree to reassociation; the choice between them is a
        throughput one, not an accuracy one. Both are less accurate than the
        ungrouped default, because grouping rotates by one representative lattice
        displacement per class rather than by each pair's own direction. That
        residual does not shrink with expansion order. Measured relative L2 versus
        a direct sum (``preset="accurate"``, solidfmm, leaf 8, theta 0.5, uniform
        N=256), orders 2 / 4 / 6:

        ==============  =========  =========  =========
        mode            p=2        p=4        p=6
        ==============  =========  =========  =========
        default         7.230e-04  8.148e-05  1.128e-05
        grouped         8.927e-04  2.049e-04  1.887e-04
        ==============  =========  =========  =========
    rotation : Optional[str]
        M2L rotation implementation, e.g. ``"solidfmm"`` or ``"cached"``.
    m2l_chunk_size : Optional[int]
        Pairs per chunk in the chunked M2L ``lax.scan``. A memory/throughput knob;
        it bounds peak memory rather than changing the result.
    l2l_chunk_size : Optional[int]
        The same for the L2L cascade.
    streamed_far_pairs : Optional[bool]
        Stream the far-pair list instead of materialising it, trading recompute
        for peak memory.
    mixed_order : bool
        Allow per-pair expansion orders in the far field.
    mixed_order_min_order : Optional[int]
        Floor on the per-pair order when ``mixed_order`` is on.
    retain_far_pairs_for_grad : bool
        Keep the frozen M2L pair list on the prepared state so a gradient path can
        re-run the downward sweep against it. Costs ~24 B/pair of steady-state
        memory, so it is off by default (the large-N preset targets minimum
        memory); **required** to differentiate the large-N path.
    """

    grouped_interactions: Optional[bool] = None
    mode: FarFieldMode = "auto"
    rotation: Optional[str] = None
    m2l_chunk_size: Optional[int] = None
    l2l_chunk_size: Optional[int] = None
    streamed_far_pairs: Optional[bool] = None
    mixed_order: bool = False
    mixed_order_min_order: Optional[int] = None
    # Keep the frozen M2L pair list on the prepared state so a gradient path can
    # re-run the downward sweep against it. Costs ~24 B/pair of steady state
    # memory, so it is off by default (the large-N preset targets minimum memory);
    # required for differentiating the large-N path.
    retain_far_pairs_for_grad: bool = False


@dataclass(frozen=True)
class NearFieldConfig:
    """Near-field direct-interaction strategy overrides.

    Frozen.

    Attributes
    ----------
    mode : NearFieldMode
        ``"auto"``, ``"baseline"`` or ``"bucketed"``. The two concrete modes agree
        **to round-off, not bit-exactly** -- they differ in edge order, hence in
        accumulation order. See
        :func:`jaccpot.nearfield.near_field.compute_leaf_p2p_accelerations`.
    edge_chunk_size : int
        Chunk width for the bucketed edge scan. A performance knob, not a
        numerical one.
    precompute_scatter_schedules : bool
        Build the bucketed scatter schedules at prepare time instead of per
        evaluation. Falls back silently and safely when the schedule would exceed
        its cap or overflow int32, which is a normal production state rather than
        an error.
    """

    mode: NearFieldMode = "auto"
    edge_chunk_size: int = 256
    precompute_scatter_schedules: bool = True


#: The four capacities a traversal config carries, in ``DualTreeTraversalConfig``
#: declaration order. Kept here so ``TraversalOverrides`` and the runtime
#: normalizer cannot drift apart, and so an unknown key can be rejected by name.
TRAVERSAL_OVERRIDE_FIELDS = (
    "max_pair_queue",
    "process_block",
    "max_interactions_per_node",
    "max_neighbors_per_leaf",
)


@dataclass(frozen=True)
class TraversalOverrides:
    """Field-by-field traversal-capacity overrides, merged onto the preset's own.

    Prefer this over passing a bare ``DualTreeTraversalConfig`` as
    ``RuntimePolicyConfig.traversal_config``.

    ``DualTreeTraversalConfig`` has no defaults for three of its four fields, so
    a caller who wants to raise ``max_pair_queue`` alone has to invent values for
    ``process_block`` and the two caps -- and supplying the object at all makes
    the runtime treat every capacity as caller-owned, which switches off the
    preset's N-dependent traversal sizing entirely. Measured on ``large_n_gpu``
    at N=65536: passing an explicit config whose ``max_pair_queue`` equalled the
    value already in force took per-step time from 1085 ms to ~3200 ms, purely
    because ``process_block`` and the interaction/neighbour caps reverted to
    whatever the caller had typed.

    Every field here defaults to ``None``, meaning "leave this one to the
    preset". Only the fields set explicitly are applied, on top of the sizing the
    runtime resolved for the current particle count -- so overriding one
    capacity cannot silently move the others.

    Examples
    --------
    Raise the pair queue and change nothing else::

        FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(
                traversal_config=TraversalOverrides(max_pair_queue=1 << 19)
            )
        )

    A plain ``dict`` with the same keys is accepted and means the same thing.

    Attributes
    ----------
    max_pair_queue : Optional[int]
        Capacity of the dual-tree pair queue.
    process_block : Optional[int]
        Pair-queue block size processed per traversal step.
    max_interactions_per_node : Optional[int]
        Cap on far-field (M2L) interactions recorded per node.
    max_neighbors_per_leaf : Optional[int]
        Cap on near-field neighbour leaves recorded per leaf.
    """

    max_pair_queue: Optional[int] = None
    process_block: Optional[int] = None
    max_interactions_per_node: Optional[int] = None
    max_neighbors_per_leaf: Optional[int] = None

    def __post_init__(self) -> None:
        for name in TRAVERSAL_OVERRIDE_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if int(value) < 1:
                raise ValueError(
                    f"TraversalOverrides.{name} must be >= 1 when set; got {value!r}"
                )

    def as_dict(self: "TraversalOverrides") -> dict[str, int]:
        """Return only the fields that were set, as ``{name: int}``.

        Returns
        -------
        dict[str, int]
            One entry per non-``None`` field, coerced to ``int``. Fields left at
            ``None`` are **absent**, not present-and-``None``: downstream code
            merges this over the preset's traversal config, so an omitted key is
            what "leave the preset's choice alone" is expressed as. An
            all-defaults instance therefore yields ``{}``.
        """

        return {
            name: int(getattr(self, name))
            for name in TRAVERSAL_OVERRIDE_FIELDS
            if getattr(self, name) is not None
        }


@dataclass(frozen=True)
class RuntimePolicyConfig:
    """Execution-policy overrides for tree build, traversal and caching.

    Frozen. This is the largest of the override groups and the one most likely to
    change *how* a run executes rather than what it computes.

    Notes
    -----
    For ``preset="large_n_gpu"``, runtime policy is canonicalized to the production
    low-memory fast path (``minimum_memory`` + streamed ``pair_grouped`` + bucketed
    near field), so overrides that contradict that profile are overridden back.

    ``traversal_config`` accepts a :class:`TraversalOverrides` (or a ``dict`` with
    the same keys) for a field-by-field merge onto the preset's resolved capacities,
    or a full ``DualTreeTraversalConfig`` for the legacy replace-everything
    behaviour. The latter warns, because replacing the object also replaces the
    capacities you did not mean to change -- measured at N=65536, an override
    intended as a no-op made a run 3x slower.

    Attributes
    ----------
    execution_backend : FMMExecutionBackend
        ``"auto"``, ``"radix"`` or ``"octree"``. ``"auto"`` may choose; an explicit
        request is honoured or fails loudly, never silently substituted.
    host_refine_mode : str
        Whether leaf refinement runs on the host, on device, or by policy.
    fail_fast : bool
        Raise instead of falling back when a requested configuration cannot run.
    jit_tree : Optional[bool]
        Compile the tree build. ``None`` leaves the policy decision to the runtime.
    jit_traversal : Optional[bool]
        Compile the dual-tree traversal.
    memory_objective : MemoryObjective
        ``"balanced"``, ``"throughput"`` or ``"minimum_memory"``; steers the
        chunk-size and streaming defaults.
    memory_budget_bytes : Optional[int]
        Advisory ceiling used when resolving those defaults.
    max_pair_queue : Optional[int]
        Legacy single-capacity override for the traversal pair queue. Prefer
        ``traversal_config`` with :class:`TraversalOverrides`.
    pair_process_block : Optional[int]
        Legacy override for the traversal process-block width.
    traversal_config : Optional[Any]
        Traversal capacities -- see the note above for the two accepted forms and
        why they behave differently.
    enable_interaction_cache : bool
        Reuse the process-level interaction-list cache across calls.
    retain_traversal_result : bool
        Keep the dual-tree walk result on the prepared state.
    retain_interactions : bool
        Keep the resolved M2L interaction list on the prepared state.
    prepare_stage_memory_split_enabled : Optional[bool]
        Split the prepare stage to lower peak memory at some throughput cost.
    autotune_m2l_chunk : bool
        Measure and pick the M2L chunk size at prepare time. Off by default, and
        consequently the autotune path is thinly covered.
    precompute_grouped_class_segments : Optional[bool]
        Build grouped-class segment tables at prepare time.
    grouped_schedule_budget_bytes : Optional[int]
        Byte budget above which grouped-class precomputation is skipped.
    nearfield_schedule_item_cap : Optional[int]
        Item cap above which the near-field scatter schedules fall back to being
        recomputed per evaluation.
    upward_leaf_batch_size : Optional[int]
        Leaves per batch in the P2M sweep.
    fixed_order : Optional[int]
        Pins the expansion order used by the evaluate stage instead of inferring
        it from the local-expansion coefficient count. Distinct from the per-call
        ``max_order``, which sizes the upward sweep. Legacy name: ``fixed_order``.
    fixed_max_leaf_size : Optional[int]
        Concrete leaf-size cap for the evaluate stage. Serves two purposes: it
        supplies ``max_leaf_size`` where a compiled path needs it concrete rather
        than traced, and it is the bound whose violation raises
        ``"fixed_max_leaf_size too small for prepared tree"``. Legacy name:
        ``fixed_max_leaf_size``.
    """

    execution_backend: FMMExecutionBackend = "auto"
    host_refine_mode: str = "auto"
    fail_fast: bool = False
    jit_tree: Optional[bool] = None
    jit_traversal: Optional[bool] = None
    memory_objective: MemoryObjective = "balanced"
    memory_budget_bytes: Optional[int] = None
    max_pair_queue: Optional[int] = None
    pair_process_block: Optional[int] = None
    traversal_config: Optional[Any] = None
    enable_interaction_cache: bool = True
    retain_traversal_result: bool = True
    retain_interactions: bool = True
    prepare_stage_memory_split_enabled: Optional[bool] = None
    autotune_m2l_chunk: bool = False
    precompute_grouped_class_segments: Optional[bool] = None
    grouped_schedule_budget_bytes: Optional[int] = None
    nearfield_schedule_item_cap: Optional[int] = None
    upward_leaf_batch_size: Optional[int] = None
    fixed_order: Optional[int] = None
    fixed_max_leaf_size: Optional[int] = None


@dataclass(frozen=True)
class GradConfig:
    """Execution options for :meth:`~jaccpot.FastMultipoleMethod.differentiable_accelerations`.

    Every field defaults to ``None``/``"auto"``, meaning "use the measured
    default, or the matching ``JACCPOT_*`` environment variable if it is set".
    Setting a field explicitly always wins over the environment, so this is a
    strict superset of the env-var interface it replaces -- existing scripts
    that export the variables keep working.

    Prefer this over the environment variables. Env vars are process-global and
    cannot differ between two solvers in one program, which matters as soon as
    Jaccpot is embedded in someone else's simulation loop.

    Attributes
    ----------
    nearfield_lane : GradNearFieldLane
        Which near-field traversal the gradient path takes. The near field is
        ~83% of the forward and ~91% of the reverse, so this is the choice that
        matters.

        * ``"auto"`` (default) -- ``"bucketed"`` below
          ``nearfield_fast_lane_min_particles``, ``"fast_lane"`` at or above it.
          This exists because the bucketed reverse **OOMs** at galaxy scale
          (30 GB peak at N=200000 against the fast lane's 6.8 GB), and requiring
          a user to know that in advance is a trap.
        * ``"bucketed"`` -- the edge-list kernel. Best at small N.
        * ``"fast_lane"`` -- leaf-major traversal. Same edge set, same force, a
          different traversal. What the suite asserts is relative L2 below 1e-13
          on the forward and below 1e-12 on both the position and mass gradients
          (``tests/unit/test_nearfield_fastlane_grad_path.py``, fp64) -- not
          bit-equality. The bit-identical *force checksums* reported in
          ``docs/differentiable_fmm.md`` come from the benchmark harness, which is
          not part of the test suite; do not read this as an asserted invariant.
          With ``use_pallas`` on an Ampere+ GPU the reverse is the analytic O(N)
          leaf-pair rule, which is what makes 200k--1M gradients feasible.
    nearfield_fast_lane_min_particles : int
        Crossover for ``nearfield_lane="auto"``. The default 100000 comes from
        the measured 200k OOM; below ~10^4 the two lanes are within +-20% with
        no consistent winner.
    fused_m2l_pallas : Optional[bool]
        Opt into the differentiable fused-Pallas M2L (Ampere+; falls back to
        pure-JAX elsewhere). ``None`` defers to
        ``JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS``. Roughly halves the forward
        at small N; the reverse is near-field-bound, so the end-to-end win is
        ~1.3-1.5x.
    analytic_p2p_vjp : Optional[bool]
        Analytic reverse rule for the near-field P2P, on by default. Turning it
        off restores plain autodiff and is for A/B measurement, not production.
    analytic_l2p_vjp : Optional[bool]
        Analytic reverse rule for the real-basis L2P, on by default. Same
        caveat as ``analytic_p2p_vjp``.
    reverse_tiers : Optional[int]
        Maximum occupancy tiers for the analytic leaf-pair reverse (default 4).
        The prepacked payload is padded to the global maximum neighbour count,
        so most of it is padding (fill 45% at N=200000, 14.5% at N=1000000);
        tiering lets low-occupancy leaves read a narrower slot window.
    reverse_tier_min_gain : Optional[float]
        Predicted slot-visit reduction below which tiering is declined (default
        3.0). Tiering costs throughput, so it only pays when the saving is
        large: measured 4.3x *slower* at N=200000, 1.91x *faster* at N=1000000.
        Calibrated on two points on one A100 -- re-measure on other hardware.
    reverse_skip_empty_tiles : Optional[bool]
        Skip reverse tiles carrying no valid source slot (default on).
        Semantics-preserving; worth ~1.24x at N=1000000 and nothing at 200000.
    reverse_leaf_batch : Optional[int]
        Reverse-pass leaf batch size (default 8). Deliberately independent of
        the forward's: the backward materialises per-tile pair tensors, so small
        tiles are what keep its memory bounded.
    reverse_block_tile : Optional[int]
        Reverse-pass block tile size (default 8). Same reasoning as
        ``reverse_leaf_batch``.
    """

    nearfield_lane: GradNearFieldLane = "auto"
    nearfield_fast_lane_min_particles: int = 100_000
    fused_m2l_pallas: Optional[bool] = None
    analytic_p2p_vjp: Optional[bool] = None
    analytic_l2p_vjp: Optional[bool] = None
    reverse_tiers: Optional[int] = None
    reverse_tier_min_gain: Optional[float] = None
    reverse_skip_empty_tiles: Optional[bool] = None
    reverse_leaf_batch: Optional[int] = None
    reverse_block_tile: Optional[int] = None

    def __post_init__(self) -> None:
        lane = str(self.nearfield_lane).strip().lower()
        if lane not in ("auto", "bucketed", "fast_lane"):
            raise ValueError(
                "GradConfig.nearfield_lane must be 'auto', 'bucketed' or "
                f"'fast_lane'; got {self.nearfield_lane!r}"
            )
        if int(self.nearfield_fast_lane_min_particles) < 0:
            raise ValueError(
                "GradConfig.nearfield_fast_lane_min_particles must be >= 0; got "
                f"{self.nearfield_fast_lane_min_particles!r}"
            )


@dataclass(frozen=True)
class FMMAdvancedConfig:
    """Aggregate container for all advanced FMM override groups.

    Frozen, and the single ``advanced=`` argument of
    :class:`jaccpot.FastMultipoleMethod`. Each group defaults to its own
    all-defaults instance, so overriding one field of one group leaves everything
    else at the preset's resolution:

    >>> FMMAdvancedConfig(farfield=FarFieldConfig(retain_far_pairs_for_grad=True))

    ``jaccpot.runtime.fmm_state._resolve_fmm_config`` normalises this, the preset,
    and the constructor arguments into a single validated ``FMMResolvedConfig``.

    Attributes
    ----------
    tree : TreeConfig
        Tree-construction overrides.
    farfield : FarFieldConfig
        Far-field interaction and translation-kernel overrides.
    nearfield : NearFieldConfig
        Near-field strategy overrides.
    runtime : RuntimePolicyConfig
        Execution-policy overrides.
    mac_type : Optional[str]
        Multipole acceptance criterion, e.g. ``"bh"`` or ``"dehnen"``. Lives here
        rather than in a group because it straddles traversal and accuracy.
    dehnen_radius_scale : float
        Scale applied to node radii in the Dehnen MAC.
    """

    tree: TreeConfig = TreeConfig()
    farfield: FarFieldConfig = FarFieldConfig()
    nearfield: NearFieldConfig = NearFieldConfig()
    runtime: RuntimePolicyConfig = RuntimePolicyConfig()
    mac_type: Optional[str] = None
    dehnen_radius_scale: float = 1.0


__all__ = [
    "Basis",
    "FMMAdvancedConfig",
    "FMMExecutionBackend",
    "FMMPreset",
    "FarFieldConfig",
    "FarFieldMode",
    "GradConfig",
    "GradNearFieldLane",
    "MACTypeInput",
    "MemoryObjective",
    "NearFieldConfig",
    "NearFieldMode",
    "RuntimePolicyConfig",
    "TraversalOverrides",
    "TreeConfig",
]
