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

    A ``runtime_checkable`` Protocol, so ``isinstance`` checks only that the
    four methods exist -- it does not check their signatures, and it does not
    check the attributes below at all.

    Attributes
    ----------
    name : str
        Basis identifier.
    p_max : int
        Largest expansion order this basis' metadata was written for.
        Documentation of the intended range, not a runtime bound.
    coefficient_ordering : str
        Human-readable statement of the packing rule.
    runtime_expansion_basis : str
        Which runtime operator family backs this basis. This is the field that
        decides which kernels actually run, so two bases with different ``name``
        can share one implementation.
    """

    name: str
    p_max: int
    coefficient_ordering: str
    runtime_expansion_basis: str

    def n_coeffs(self: "BasisInterface", p: int) -> int:
        """Return number of packed coefficients for expansion order ``p``.

        Parameters
        ----------
        p : int
            Expansion order.

        Returns
        -------
        int
            Packed coefficient count in this basis' layout.
        """

    def pack_coeffs(self: "BasisInterface", coeffs: Array, *, order: int) -> Array:
        """Pack basis coefficients into the runtime 1D layout.

        Parameters
        ----------
        coeffs : Array
            Coefficients in the basis' structured form.
        order : int
            Expansion order ``p``.

        Returns
        -------
        Array
            Packed 1D coefficients, last axis of length ``n_coeffs(order)``.
        """

    def unpack_coeffs(self: "BasisInterface", packed: Array, *, order: int) -> Array:
        """Unpack runtime 1D coefficient layout into structured basis form.

        The inverse of :meth:`pack_coeffs`. A basis whose structured form already
        *is* the packed layout may implement this as a validating pass-through.

        Parameters
        ----------
        packed : Array
            Packed 1D coefficients.
        order : int
            Expansion order ``p``.

        Returns
        -------
        Array
            Coefficients in the basis' structured form.
        """

    def m2l_rot_scale(
        self: "BasisInterface", sources: Array, deltas: Array, *, order: int
    ) -> Array:
        """Evaluate batched M2L translation in the basis' native convention.

        Optional in practice: an implementation whose production M2L is reached
        through the runtime's ``basis_mode`` seam rather than through this object
        may raise :class:`NotImplementedError` here.
        :class:`~jaccpot.RealSHBasis` does exactly that, while
        :class:`~jaccpot.basis.complex_sh.ComplexSHBasis` implements it.

        Parameters
        ----------
        sources : Array
            Packed source multipoles, one row per pair.
        deltas : Array
            ``[N, 3]`` displacements, ``target centre - source centre``. See
            ``docs/operator_conventions.md`` section 1 -- the same-type
            translations use the opposite sign.
        order : int
            Expansion order ``p``.

        Returns
        -------
        Array
            Packed local coefficients, one row per pair.
        """


@dataclass(frozen=True)
class BasisMetadata:
    """Static metadata helper for basis implementations.

    Metadata only: it carries no coefficient data, and none of these fields is
    enforced at runtime.

    Attributes
    ----------
    name : str
        Basis identifier.
    p_max : int
        Largest expansion order this metadata was written for. Documentation of
        the intended range, **not** a runtime bound -- nothing checks an order
        against it.
    coefficient_ordering : str
        Human-readable statement of the packing rule.
    runtime_expansion_basis : str
        Which runtime operator family backs this basis.
    """

    name: str
    p_max: int
    coefficient_ordering: str
    runtime_expansion_basis: str


__all__ = [
    "BasisInterface",
    "BasisMetadata",
]
