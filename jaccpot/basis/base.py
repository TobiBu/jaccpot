"""Shared basis interface for FMM coefficient transforms and M2L kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jaxtyping import Array


@runtime_checkable
class BasisInterface(Protocol):
    """Minimal basis contract used by solver/runtime orchestration.

    Implementations define coefficient layout metadata and batched transforms
    needed by M2L operators.
    """

    name: str
    p_max: int
    coefficient_ordering: str
    runtime_expansion_basis: str

    def n_coeffs(self, p: int) -> int:
        """Return number of packed coefficients for expansion order ``p``."""

    def pack_coeffs(self, coeffs: Array, *, order: int) -> Array:
        """Pack basis coefficients into the runtime 1D layout."""

    def unpack_coeffs(self, packed: Array, *, order: int) -> Array:
        """Unpack runtime 1D coefficient layout into structured basis form."""

    def rotate_to_z(self, coeffs: Array, directions: Array, *, order: int) -> Array:
        """Rotate batched coefficients into a frame where direction maps to ``+z``."""

    def rotate_from_z(self, coeffs: Array, directions: Array, *, order: int) -> Array:
        """Rotate batched coefficients back from the ``+z`` frame."""

    def m2l_rot_scale(self, sources: Array, deltas: Array, *, order: int) -> Array:
        """Evaluate batched M2L translation in the basis' native convention."""


@dataclass(frozen=True)
class BasisMetadata:
    """Static metadata helper for basis implementations."""

    name: str
    p_max: int
    coefficient_ordering: str
    runtime_expansion_basis: str
