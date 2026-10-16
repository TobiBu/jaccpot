"""Jaccpot: high-level FMM solver APIs built on Yggdrasil artifacts."""

from __future__ import annotations

from ._typecheck import enable_runtime_typecheck

enable_runtime_typecheck()

from .autodiff import direct_sum_gravitational_acceleration
from .basis import RealSHBasis
from .config import (
    FarFieldConfig,
    FMMAdvancedConfig,
    FMMPreset,
    GradConfig,
    MemoryObjective,
    NearFieldConfig,
    RuntimePolicyConfig,
    TraversalOverrides,
    TreeConfig,
)
from .nornax_adapter import BlockStepFMM, DistributedBlockStepFMM
from .odisseo import OdisseoFMMCoupler
from .runtime.fused_provider import FusedForceProvider
from .solver import FastMultipoleMethod

__all__ = [
    "BlockStepFMM",
    "DistributedBlockStepFMM",
    "FMMAdvancedConfig",
    "FMMPreset",
    "FarFieldConfig",
    "GradConfig",
    "FastMultipoleMethod",
    "FusedForceProvider",
    "MemoryObjective",
    "RealSHBasis",
    "NearFieldConfig",
    "OdisseoFMMCoupler",
    "RuntimePolicyConfig",
    "TraversalOverrides",
    "TreeConfig",
    "direct_sum_gravitational_acceleration",
]
