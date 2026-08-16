"""Jaccpot: high-level FMM solver APIs built on Yggdrasil artifacts."""

from __future__ import annotations

from ._typecheck import enable_runtime_typecheck

enable_runtime_typecheck()

from .autodiff import direct_sum_gravitational_acceleration
from .basis import ComplexSHBasis, RealSHBasis
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
from .nornax_adapter import BlockStepFMM
from .odisseo import OdisseoFMMCoupler
from .solver import FastMultipoleMethod

__all__ = [
    "BlockStepFMM",
    "FMMAdvancedConfig",
    "FMMPreset",
    "FarFieldConfig",
    "GradConfig",
    "FastMultipoleMethod",
    "ComplexSHBasis",
    "MemoryObjective",
    "RealSHBasis",
    "NearFieldConfig",
    "OdisseoFMMCoupler",
    "RuntimePolicyConfig",
    "TraversalOverrides",
    "TreeConfig",
    "direct_sum_gravitational_acceleration",
]
