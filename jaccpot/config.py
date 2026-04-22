"""Preset-first configuration model for Jaccpot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

BASIS_DOC = "Preferred production basis is 'solidfmm'."
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


@dataclass(frozen=True)
class NearFieldConfig:
    """Near-field direct-interaction strategy overrides."""

    mode: NearFieldMode = "auto"
    edge_chunk_size: int = 256
    precompute_scatter_schedules: bool = True


@dataclass(frozen=True)
class RuntimePolicyConfig:
    """Execution-policy overrides for tree build and traversal.

    Notes:
    - `runtime_path='legacy'` is deprecated and will be removed.
    - For `preset='large_n_gpu'`, runtime policy is canonicalized to the
      production low-memory fast path (minimum_memory + streamed pair_grouped
      + bucketed nearfield).
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
    autotune_m2l_chunk: bool = False
    precompute_grouped_class_segments: Optional[bool] = None
    grouped_schedule_budget_bytes: Optional[int] = None
    nearfield_schedule_item_cap: Optional[int] = None
    upward_leaf_batch_size: Optional[int] = None


@dataclass(frozen=True)
class FMMAdvancedConfig:
    """Aggregate container for all advanced FMM override groups."""

    tree: TreeConfig = TreeConfig()
    farfield: FarFieldConfig = FarFieldConfig()
    nearfield: NearFieldConfig = NearFieldConfig()
    runtime: RuntimePolicyConfig = RuntimePolicyConfig()
    mac_type: Optional[str] = None
    dehnen_radius_scale: float = 1.0
