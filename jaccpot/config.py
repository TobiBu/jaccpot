"""Preset-first configuration model for Jaccpot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

Basis = Literal["cartesian", "solidfmm"]
FarFieldMode = Literal["auto", "pair_grouped", "class_major"]
NearFieldMode = Literal["auto", "baseline", "bucketed"]


class FMMPreset(str, Enum):
    """User-facing quality/speed presets."""

    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


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


@dataclass(frozen=True)
class NearFieldConfig:
    """Near-field direct-interaction strategy overrides."""

    mode: NearFieldMode = "auto"
    edge_chunk_size: int = 256
    precompute_scatter_schedules: bool = True


@dataclass(frozen=True)
class RuntimePolicyConfig:
    """Execution-policy overrides for tree build and traversal."""

    host_refine_mode: str = "auto"
    jit_tree: Optional[bool] = None
    jit_traversal: Optional[bool] = None
    max_pair_queue: Optional[int] = None
    pair_process_block: Optional[int] = None
    traversal_config: Optional[Any] = None


@dataclass(frozen=True)
class FMMAdvancedConfig:
    """Aggregate container for all advanced FMM override groups."""

    tree: TreeConfig = TreeConfig()
    farfield: FarFieldConfig = FarFieldConfig()
    nearfield: NearFieldConfig = NearFieldConfig()
    runtime: RuntimePolicyConfig = RuntimePolicyConfig()
    mac_type: Optional[str] = None
    dehnen_radius_scale: float = 1.0
