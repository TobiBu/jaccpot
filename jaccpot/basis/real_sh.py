"""Real spherical-harmonic basis layout.

The complex <-> real conversion helpers (``complex_to_real_coeffs``,
``real_to_complex_coeffs``) went with the complex basis in the 2026-10 cleanup
(C).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array


def n_real_sh_coeffs(order: int) -> int:
    """Number of packed real SH coefficients for ``order``.

    Parameters
    ----------
    order : int
        Expansion order ``p``. Must be non-negative.

    Returns
    -------
    int
        ``(p + 1) ** 2`` -- the real packing holds one entry per ``(n, m)`` with
        ``m`` running the full ``-n..n``, unlike a complex packing that stores
        only ``m >= 0`` and recovers the rest by conjugation.

    Raises
    ------
    ValueError
        If ``order`` is negative.
    """
    p = int(order)
    if p < 0:
        raise ValueError("order must be non-negative")
    return (p + 1) * (p + 1)


@dataclass(frozen=True)
class RealSHBasis:
    """Packed real spherical-harmonic basis metadata and layout helpers.

    The production default basis. Frozen, and carries no coefficient data -- it
    describes the layout so callers can size buffers and index coefficients without
    duplicating the packing rule.

    Coefficients are Dehnen (2014) solid harmonics in the **no-sqrt2 real basis**,
    packed ell-major with ``m = -ell..+ell`` inside each degree, so degree ``ell``
    occupies ``[ell**2, (ell+1)**2)`` and an order-``p`` expansion has ``(p+1)**2``
    entries. Positive ``m`` holds the cosine channel and negative ``m`` the sine
    channel.

    Attributes
    ----------
    p_max : int
        Largest expansion order this metadata is valid for.
    name : str
        Basis identifier, ``"real"``.
    coefficient_ordering : str
        Human-readable statement of the packing rule above.
    runtime_expansion_basis : str
        Which runtime operator family backs this basis. It is ``"solidfmm"`` rather
        than ``"real"`` because the runtime's basis seam names the *operator*
        family, whose only basis is the real one.
    """

    p_max: int = 32
    name: str = "real"
    coefficient_ordering: str = "ell-major, m=-ell..+ell (packed real SH)"
    runtime_expansion_basis: str = "solidfmm"

    def n_coeffs(self: "RealSHBasis", p: int) -> int:
        """Return packed coefficient count for expansion order ``p``.

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
        return n_real_sh_coeffs(int(p))

    def pack_coeffs(self: "RealSHBasis", coeffs: Array, *, order: int) -> Array:
        """Validate and return packed real SH coefficients.

        A shape check, not a repacking: the real basis already stores what the
        runtime consumes, so "packing" is the identity once the width is
        confirmed.

        Parameters
        ----------
        coeffs : Array
            Coefficients to check, ``(..., (p+1)**2)``.
        order : int
            Expansion order ``p``.

        Returns
        -------
        Array
            ``coeffs`` as an array, unchanged in value and layout.

        Raises
        ------
        ValueError
            If the trailing dimension is not ``(p+1)**2``.
        """
        arr = jnp.asarray(coeffs)
        expected = self.n_coeffs(order)
        if int(arr.shape[-1]) != expected:
            raise ValueError(
                f"expected last dimension {expected} for order={int(order)}, got {arr.shape[-1]}"
            )
        return arr

    def unpack_coeffs(self: "RealSHBasis", packed: Array, *, order: int) -> Array:
        """Return packed coefficients (real SH uses the packed runtime layout).

        The identity, deliberately: pack and unpack are the same operation for
        this basis, and the method exists so callers can be written against the
        :class:`~jaccpot.BasisInterface` protocol without special-casing it.

        Parameters
        ----------
        packed : Array
            Packed coefficients, ``(..., (p+1)**2)``.
        order : int
            Expansion order ``p``.

        Returns
        -------
        Array
            The same coefficients, shape-checked by
            :meth:`pack_coeffs` and so subject to its ``ValueError``.
        """
        return self.pack_coeffs(packed, order=order)

    def rotate_to_z(
        self: "RealSHBasis", coeffs: Array, directions: Array, *, order: int
    ) -> Array:
        """Rotate real SH coefficients into a z-aligned frame (not yet implemented).

        Present to satisfy the basis protocol, not to be called. The real
        rotations that production uses are not routed through this class at all:
        they live in :mod:`jaccpot.operators.m2l_real_rot_scale`, which the
        runtime reaches directly.

        Parameters
        ----------
        coeffs : Array
            Packed real coefficients that would be rotated.
        directions : Array
            Target directions to align to z.
        order : int
            Expansion order ``p``.

        Returns
        -------
        Array
            Never returns; the annotation states the intended contract.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("real SH rotations are implemented in Stage J3")

    def rotate_from_z(
        self: "RealSHBasis", coeffs: Array, directions: Array, *, order: int
    ) -> Array:
        """Rotate real SH coefficients back from a z-aligned frame.

        The inverse of :meth:`rotate_to_z`, and unimplemented on the same terms.

        Parameters
        ----------
        coeffs : Array
            Packed real coefficients in the z-aligned frame.
        directions : Array
            Directions the frame was aligned to.
        order : int
            Expansion order ``p``.

        Returns
        -------
        Array
            Never returns; the annotation states the intended contract.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("real SH rotations are implemented in Stage J3")

    def m2l_rot_scale(
        self: "RealSHBasis", sources: Array, deltas: Array, *, order: int
    ) -> Array:
        """Translate real SH multipoles to locals via rotate/scale M2L path.

        Unimplemented here for the same reason as the two rotations above: the
        production real rot-scale M2L is
        :func:`~jaccpot.operators.m2l_real_rot_scale.m2l_rot_scale_real_batch`,
        reached by the runtime directly rather than through this object.

        Parameters
        ----------
        sources : Array
            Packed real multipole coefficients, one row per pair.
        deltas : Array
            Target-minus-source centre displacements ``[N, 3]``.
        order : int
            Expansion order ``p``.

        Returns
        -------
        Array
            Never returns; the annotation states the intended contract.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("real SH rotate+scale M2L is implemented in Stage J3")


__all__ = [
    "RealSHBasis",
    "n_real_sh_coeffs",
]
