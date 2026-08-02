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
MemoryObjective = Literal["balanced", "throughput", "minimum_memory"]
FMMExecutionBackend = Literal["auto", "radix", "octree"]


class FMMPreset(str, Enum):
    """User-facing quality/speed presets."""

    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    LARGE_N_GPU = "large_n_gpu"


@dataclass(frozen=True)
class TreeConfig:
    """Tree-construction overrides for advanced runtime tuning."""

    tree_type: Optional[str] = None
    mode: Optional[str] = None
    leaf_target: Optional[int] = None
    refine_local: Optional[bool] = None
    max_refine_levels: Optional[int] = None
    aspect_threshold: Optional[float] = None


@dataclass(frozen=True)
class FarFieldConfig:
    """Far-field interaction and translation-kernel overrides."""

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
    """Near-field direct-interaction strategy overrides."""

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
        """Return only the fields that were set, as ``{name: int}``."""

        return {
            name: int(getattr(self, name))
            for name in TRAVERSAL_OVERRIDE_FIELDS
            if getattr(self, name) is not None
        }


@dataclass(frozen=True)
class RuntimePolicyConfig:
    """Execution-policy overrides for tree build and traversal.

    Notes:
    - For `preset='large_n_gpu'`, runtime policy is canonicalized to the
      production low-memory fast path (minimum_memory + streamed pair_grouped
      + bucketed nearfield).
    - ``traversal_config`` accepts a :class:`TraversalOverrides` (or a ``dict``
      with the same keys) for a field-by-field merge onto the preset's resolved
      capacities, or a full ``DualTreeTraversalConfig`` for the legacy
      replace-everything behaviour. The latter warns, because replacing the
      object also replaces the capacities you did not mean to change.
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
        * ``"fast_lane"`` -- leaf-major traversal. Same edge set and the same
          force (bit-identical checksums), different traversal. With
          ``use_pallas`` on an Ampere+ GPU its reverse is the analytic O(N)
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
    analytic_p2p_vjp, analytic_l2p_vjp : Optional[bool]
        Analytic reverse rules for the near-field P2P and the real-basis L2P,
        both on by default. Turning them off restores plain autodiff and is for
        A/B measurement, not production.
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
    reverse_leaf_batch, reverse_block_tile : Optional[int]
        Reverse-pass tiling (defaults 8 and 8). Deliberately independent of the
        forward's: the backward materialises per-tile pair tensors, so small
        tiles are what keep its memory bounded.
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
    """Aggregate container for all advanced FMM override groups."""

    tree: TreeConfig = TreeConfig()
    farfield: FarFieldConfig = FarFieldConfig()
    nearfield: NearFieldConfig = NearFieldConfig()
    runtime: RuntimePolicyConfig = RuntimePolicyConfig()
    mac_type: Optional[str] = None
    dehnen_radius_scale: float = 1.0
