"""Jaccpot: high-level FMM solver APIs built on Yggdrasil artifacts."""

from ._typecheck import enable_runtime_typecheck

enable_runtime_typecheck()

from .autodiff import differentiable_gravitational_acceleration
from .basis import ComplexSHBasis, RealSHBasis
from .config import (
    FarFieldConfig,
    FMMAdvancedConfig,
    FMMPreset,
    MemoryObjective,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)
from .odisseo import OdisseoFMMCoupler
from .solver import FastMultipoleMethod

__all__ = [
    "FMMAdvancedConfig",
    "FMMPreset",
    "FarFieldConfig",
    "FastMultipoleMethod",
    "ComplexSHBasis",
    "MemoryObjective",
    "RealSHBasis",
    "NearFieldConfig",
    "OdisseoFMMCoupler",
    "RuntimePolicyConfig",
    "TreeConfig",
    "differentiable_gravitational_acceleration",
]
