"""Jaccpot: high-level FMM solver APIs built on Yggdrasil artifacts."""

from ._typecheck import enable_runtime_typecheck

enable_runtime_typecheck()

from .config import (
    FMMAdvancedConfig,
    FMMPreset,
    FarFieldConfig,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)
from .autodiff import differentiable_gravitational_acceleration
from .solver import FastMultipoleMethod

__all__ = [
    "FMMAdvancedConfig",
    "FMMPreset",
    "FarFieldConfig",
    "FastMultipoleMethod",
    "NearFieldConfig",
    "RuntimePolicyConfig",
    "TreeConfig",
    "differentiable_gravitational_acceleration",
]
