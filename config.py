"""Preset-first configuration model for Jaccpot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

from jaxtyping import DTypeLike

Basis = Literal["cartesian", "spherical", "solidfmm"]
FarFieldMode = Literal["auto", "pair_grouped", "class_major"]
NearFieldMode = Literal["auto", "baseline", "bucketed"]


class FMMPreset(str, Enum):
    """User-facing quality/speed presets."""

    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


@dataclass(frozen=True)
class TreeConfig:
    mode: Optional[str] = None
    leaf_target: Optional[int] = None
    refine_local: Optional[bool] = None
    max_refine_levels: Optional[int] = None
    aspect_threshold: Optional[float] = None


@dataclass(frozen=True)
class FarFieldConfig:
    grouped_interactions: Optional[bool] = None
    mode: FarFieldMode = "auto"
    rotation: Optional[str] = None
    m2l_chunk_size: Optional[int] = None
    l2l_chunk_size: Optional[int] = None


@dataclass(frozen=True)
class NearFieldConfig:
    mode: NearFieldMode = "auto"
    edge_chunk_size: int = 256
    precompute_scatter_schedules: bool = True


@dataclass(frozen=True)
class RuntimePolicyConfig:
    host_refine_mode: str = "auto"
    jit_tree: Optional[bool] = None
    jit_traversal: Optional[bool] = None
    max_pair_queue: Optional[int] = None
    pair_process_block: Optional[int] = None
    traversal_config: Optional[Any] = None


@dataclass(frozen=True)
class FMMAdvancedConfig:
    tree: TreeConfig = TreeConfig()
    farfield: FarFieldConfig = FarFieldConfig()
    nearfield: NearFieldConfig = NearFieldConfig()
    runtime: RuntimePolicyConfig = RuntimePolicyConfig()
    mac_type: Optional[str] = None
    dehnen_radius_scale: float = 1.0


@dataclass(frozen=True)
class SolverConfig:
    preset: FMMPreset
    basis: Basis
    theta: float
    softening: float
    working_dtype: Optional[DTypeLike]
    advanced: Optional[FMMAdvancedConfig]
