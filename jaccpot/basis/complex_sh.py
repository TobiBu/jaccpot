"""Complex spherical-harmonic basis adapter for existing solidfmm kernels."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array

from jaccpot.operators.complex_ops import m2l_complex_reference_batch
from jaccpot.operators.real_harmonics import sh_size


@dataclass(frozen=True)
class ComplexSHBasis:
    """Adapter exposing the solidfmm complex basis through the common basis API.

    The complex counterpart of :class:`~jaccpot.RealSHBasis`, and like it, layout
    metadata only -- it carries no coefficient data. ``basis="complex"`` and
    ``basis="solidfmm"`` both resolve to this same object: they are two spellings of
    one basis, not two bases, so they produce bit-identical forces (measured at
    N=2048, p=4, theta=0.5: max difference exactly 0.0).

    Coefficients use the packed solidfmm layout -- ell-major blocks with
    ``m = -ell..+ell`` for each degree, so degree ``ell`` occupies
    ``[ell**2, (ell+1)**2)`` and an order-``p`` expansion has ``(p+1)**2`` entries.
    The independent cross-check worth running is this basis against ``"real"``, not
    against ``"solidfmm"``.

    Attributes
    ----------
    p_max : int
        Largest expansion order this metadata is valid for.
    name : str
        Basis identifier.
    coefficient_ordering : str
        Human-readable statement of the packing rule above.
    runtime_expansion_basis : str
        Which runtime operator family backs this basis.
    """

    p_max: int = 32
    name: str = "complex"
    coefficient_ordering: str = "ell-major, m=-ell..+ell (packed complex SH)"
    runtime_expansion_basis: str = "solidfmm"

    def n_coeffs(self: "ComplexSHBasis", p: int) -> int:
        """Return packed coefficient count for order ``p``.

        Parameters
        ----------
        p : int
            Expansion order. Not validated against ``p_max`` -- that field
            documents the range this metadata was written for, it is not a
            runtime bound.

        Returns
        -------
        int
            ``(p + 1) ** 2``.
        """
        return int(sh_size(int(p)))

    def pack_coeffs(self: "ComplexSHBasis", coeffs: Array, *, order: int) -> Array:
        """Validate and return packed solidfmm complex coefficients.

        Validation only -- the complex basis' structured form already *is* the
        packed layout, so there is nothing to rearrange. Only the last axis is
        checked, so any leading batch shape passes through.

        Parameters
        ----------
        coeffs : Array
            Packed complex coefficients; last axis of length ``n_coeffs(order)``.
        order : int
            Expansion order ``p``.

        Returns
        -------
        Array
            ``coeffs`` as an array, unchanged.

        Raises
        ------
        ValueError
            If the last axis is not ``n_coeffs(order)``.
        """
        coeffs_arr = jnp.asarray(coeffs)
        expected = self.n_coeffs(order)
        if int(coeffs_arr.shape[-1]) != expected:
            raise ValueError(
                f"expected last dimension {expected} for order={int(order)}, got {coeffs_arr.shape[-1]}"
            )
        return coeffs_arr

    def unpack_coeffs(self: "ComplexSHBasis", packed: Array, *, order: int) -> Array:
        """Return packed coefficients (complex basis already uses packed layout).

        Because pack and unpack are the same operation here, this delegates to
        :meth:`pack_coeffs` -- so it validates rather than merely returning, and
        a wrong-length input raises from here too.

        Parameters
        ----------
        packed : Array
            Packed complex coefficients.
        order : int
            Expansion order ``p``.

        Returns
        -------
        Array
            ``packed`` as an array, unchanged.
        """
        return self.pack_coeffs(packed, order=order)

    def m2l_rot_scale(
        self: "ComplexSHBasis", sources: Array, deltas: Array, *, order: int
    ) -> Array:
        """Run existing batched complex M2L kernel for source multipoles.

        Implemented here, unlike :meth:`~jaccpot.RealSHBasis.m2l_rot_scale`,
        which always raises because the production real M2L is reached through
        the runtime's ``basis_mode`` seam instead. The two sibling adapters are
        deliberately asymmetric in that respect.

        Parameters
        ----------
        sources : Array
            Packed complex multipoles, one row per pair.
        deltas : Array
            ``(batch, 3)`` displacements, ``target centre - source centre``. See
            ``docs/operator_conventions.md`` section 1 -- the same-type
            translations use the opposite sign.
        order : int
            Expansion order ``p``.

        Returns
        -------
        Array
            Packed local coefficients, one row per pair.

        Raises
        ------
        ValueError
            If ``sources`` has the wrong packed length, or ``deltas`` is not
            2-D with three columns.
        """
        src = self.pack_coeffs(sources, order=order)
        delta_arr = jnp.asarray(deltas)
        if delta_arr.ndim != 2 or int(delta_arr.shape[1]) != 3:
            raise ValueError("deltas must have shape (batch, 3)")
        return m2l_complex_reference_batch(src, delta_arr, order=int(order))


__all__ = [
    "ComplexSHBasis",
]
