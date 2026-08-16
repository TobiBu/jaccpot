"""Real spherical-harmonic basis layout and complex/real conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array


def _idx_nm(n: int, m: int) -> int:
    """Packed index for degree ``n`` and order ``m`` in ``m=-n..n`` layout.

    Degree ``n`` occupies ``[n**2, (n+1)**2)``, so the offset within that block is
    ``m + n`` and the whole packing is ``n*n + m + n``. Host arithmetic on Python
    ints, not a traced computation -- callers use it to build static indices.

    Parameters
    ----------
    n : int
        Degree.
    m : int
        Order, in ``-n..n``.

    Returns
    -------
    int
        Index into the packed ``(p+1)**2`` layout.

    Raises
    ------
    ValueError
        If ``abs(m) > n``, which would index into a neighbouring degree's block
        rather than fail on its own.
    """
    if abs(int(m)) > int(n):
        raise ValueError("|m| must be <= n")
    return int(n) * int(n) + (int(m) + int(n))


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


def complex_to_real_coeffs(complex_coeffs: Array, *, order: int) -> Array:
    """Convert packed complex SH coefficients to packed real SH coefficients.

    This conversion assumes Condon-Shortley style paired complex coefficients
    where each ``(n, m>0)`` pair is transformed into two real coefficients
    ``(n, +m)`` and ``(n, -m)``.

    .. warning::

       This produces the **unitary sqrt(2) tesseral basis**, which is NOT the
       basis :class:`RealSHBasis` below describes and NOT what the real
       M2L/M2M/L2L/L2P operators consume. Those use Dehnen's **no-sqrt2** real
       solid harmonics, and the two normalizations are incompatible -- feeding
       this function's output to them is silently wrong, not an error. The
       matching conversion for that path is
       :func:`~jaccpot.operators.real_dehnen_q.complex_to_dehnen_real_coeffs`,
       which is pinned against ``p2m_real_direct`` for a point source.

       Living in the same module as ``RealSHBasis`` is what makes this a trap;
       the name does not distinguish them.

    Parameters
    ----------
    complex_coeffs : Array
        Packed complex coefficients, ``(..., (p+1)**2)``.
    order : int
        Expansion order ``p``.

    Returns
    -------
    Array
        Packed real coefficients of the same shape, in the input's real dtype.
        Positive ``m`` carries the cosine channel, negative ``m`` the sine.

    Raises
    ------
    ValueError
        If the trailing dimension is not ``(p+1)**2``.
    """

    coeffs = jnp.asarray(complex_coeffs)
    expected = n_real_sh_coeffs(order)
    if int(coeffs.shape[-1]) != expected:
        raise ValueError(
            f"expected last dimension {expected} for order={int(order)}, got {coeffs.shape[-1]}"
        )

    out = jnp.zeros(coeffs.shape, dtype=coeffs.real.dtype)
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, dtype=coeffs.real.dtype))

    for n in range(int(order) + 1):
        idx0 = _idx_nm(n, 0)
        out = out.at[..., idx0].set(jnp.real(coeffs[..., idx0]))
        for m in range(1, n + 1):
            idx_p = _idx_nm(n, m)
            idx_n = _idx_nm(n, -m)
            sign = -1.0 if (m % 2) else 1.0
            c_p = coeffs[..., idx_p]
            c_n = coeffs[..., idx_n]
            r_p = (sign * c_p + c_n) / sqrt2
            r_n = (sign * c_p - c_n) / (1j * sqrt2)
            out = out.at[..., idx_p].set(jnp.real(r_p))
            out = out.at[..., idx_n].set(jnp.real(r_n))

    return out


def real_to_complex_coeffs(real_coeffs: Array, *, order: int) -> Array:
    """Convert packed real SH coefficients to packed complex SH coefficients.

    The inverse of :func:`complex_to_real_coeffs`, and it inherits that
    function's convention: it expects the **unitary sqrt(2) tesseral basis**, not
    Dehnen's no-sqrt2 real harmonics. Round-tripping through the pair is exact to
    round-off; round-tripping through one of these and one of the Dehnen
    converters is not a round trip at all.

    Parameters
    ----------
    real_coeffs : Array
        Packed real coefficients, ``(..., (p+1)**2)``.
    order : int
        Expansion order ``p``.

    Returns
    -------
    Array
        Packed complex coefficients of the same shape, promoted to at least
        ``complex64``.

    Raises
    ------
    ValueError
        If the trailing dimension is not ``(p+1)**2``.
    """

    coeffs = jnp.asarray(real_coeffs)
    expected = n_real_sh_coeffs(order)
    if int(coeffs.shape[-1]) != expected:
        raise ValueError(
            f"expected last dimension {expected} for order={int(order)}, got {coeffs.shape[-1]}"
        )

    out = jnp.zeros(coeffs.shape, dtype=jnp.result_type(coeffs.dtype, jnp.complex64))
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, dtype=coeffs.dtype))

    for n in range(int(order) + 1):
        idx0 = _idx_nm(n, 0)
        out = out.at[..., idx0].set(coeffs[..., idx0].astype(out.dtype))
        for m in range(1, n + 1):
            idx_p = _idx_nm(n, m)
            idx_n = _idx_nm(n, -m)
            sign = -1.0 if (m % 2) else 1.0
            r_p = coeffs[..., idx_p]
            r_n = coeffs[..., idx_n]
            c_p = sign * (r_p + 1j * r_n) / sqrt2
            c_n = (r_p - 1j * r_n) / sqrt2
            out = out.at[..., idx_p].set(c_p.astype(out.dtype))
            out = out.at[..., idx_n].set(c_n.astype(out.dtype))

    return out


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
        family, and the real operators are reached through it via a static
        ``basis_mode`` discriminator.
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
        runtime reaches directly via its static ``basis_mode`` discriminator.

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
        reached through the runtime's ``basis_mode`` seam rather than through
        this object.

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
    "complex_to_real_coeffs",
    "real_to_complex_coeffs",
    "n_real_sh_coeffs",
]
