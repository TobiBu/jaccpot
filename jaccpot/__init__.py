"""Jaccpot: high-level FMM solver APIs built on Yggdrasil artifacts."""

from __future__ import annotations

from ._typecheck import enable_runtime_typecheck

enable_runtime_typecheck()

from .autodiff import direct_sum_gravitational_acceleration
from .basis import RealSHBasis
from .config import (
    _COMPLEX_BASIS_REMOVED,
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
from .solver import FastMultipoleMethod

__all__ = [
    "BlockStepFMM",
    "DistributedBlockStepFMM",
    "FMMAdvancedConfig",
    "FMMPreset",
    "FarFieldConfig",
    "GradConfig",
    "FastMultipoleMethod",
    "MemoryObjective",
    "RealSHBasis",
    "NearFieldConfig",
    "OdisseoFMMCoupler",
    "RuntimePolicyConfig",
    "TraversalOverrides",
    "TreeConfig",
    "direct_sum_gravitational_acceleration",
]


def __getattr__(name: str) -> object:
    """Refuse the removed ``ComplexSHBasis`` with its migration message.

    ``from jaccpot import ComplexSHBasis`` would otherwise fail with a bare "cannot
    import name". An ``ImportError`` (not an ``AttributeError``) is what carries the
    message through a ``from ... import``; the complex basis was removed in the
    2026-10 cleanup (C).

    Parameters
    ----------
    name : str
        The attribute looked up on the package.

    Returns
    -------
    object
        Never returns: every name that reaches here is missing.

    Raises
    ------
    ImportError
        For ``ComplexSHBasis``, with the migration message.
    AttributeError
        For any other missing name, as for a module without ``__getattr__``.
    """
    if name == "ComplexSHBasis":
        raise ImportError(_COMPLEX_BASIS_REMOVED, name=__name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
