"""Basis abstractions for FMM coefficient representations."""

from __future__ import annotations

from jaccpot.config import _COMPLEX_BASIS_REMOVED

from .base import BasisInterface, BasisMetadata
from .real_sh import RealSHBasis

__all__ = [
    "BasisInterface",
    "BasisMetadata",
    "RealSHBasis",
]


def __getattr__(name: str) -> object:
    """Refuse the removed ``ComplexSHBasis`` with its migration message.

    The complex basis, its ``ComplexSHBasis`` adapter and the
    ``complex_to_real_coeffs`` / ``real_to_complex_coeffs`` helpers were removed in
    the 2026-10 cleanup (C). An ``ImportError`` (not an ``AttributeError``) is what
    carries the message through a ``from ... import``.

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
