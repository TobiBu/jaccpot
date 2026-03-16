"""Real spherical-harmonic basis layout and complex/real conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array


def _idx_nm(n: int, m: int) -> int:
    """Packed index for degree ``n`` and order ``m`` in ``m=-n..n`` layout."""
    if abs(int(m)) > int(n):
        raise ValueError("|m| must be <= n")
    return int(n) * int(n) + (int(m) + int(n))


def n_real_sh_coeffs(order: int) -> int:
    """Number of packed real SH coefficients for ``order``."""
    p = int(order)
    if p < 0:
        raise ValueError("order must be non-negative")
    return (p + 1) * (p + 1)


def complex_to_real_coeffs(complex_coeffs: Array, *, order: int) -> Array:
    """Convert packed complex SH coefficients to packed real SH coefficients.

    This conversion assumes Condon-Shortley style paired complex coefficients
    where each ``(n, m>0)`` pair is transformed into two real coefficients
    ``(n, +m)`` and ``(n, -m)``.
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
    """Convert packed real SH coefficients to packed complex SH coefficients."""

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
    """Packed real spherical-harmonic basis metadata and layout helpers."""

    p_max: int = 32
    name: str = "real"
    coefficient_ordering: str = "ell-major, m=-ell..+ell (packed real SH)"
    runtime_expansion_basis: str = "solidfmm"

    def n_coeffs(self: "RealSHBasis", p: int) -> int:
        """Return packed coefficient count for expansion order ``p``."""
        return n_real_sh_coeffs(int(p))

    def pack_coeffs(self: "RealSHBasis", coeffs: Array, *, order: int) -> Array:
        """Validate and return packed real SH coefficients."""
        arr = jnp.asarray(coeffs)
        expected = self.n_coeffs(order)
        if int(arr.shape[-1]) != expected:
            raise ValueError(
                f"expected last dimension {expected} for order={int(order)}, got {arr.shape[-1]}"
            )
        return arr

    def unpack_coeffs(self: "RealSHBasis", packed: Array, *, order: int) -> Array:
        """Return packed coefficients (real SH uses the packed runtime layout)."""
        return self.pack_coeffs(packed, order=order)

    def rotate_to_z(
        self: "RealSHBasis", coeffs: Array, directions: Array, *, order: int
    ) -> Array:
        """Rotate real SH coefficients into a z-aligned frame (not yet implemented)."""
        raise NotImplementedError("real SH rotations are implemented in Stage J3")

    def rotate_from_z(
        self: "RealSHBasis", coeffs: Array, directions: Array, *, order: int
    ) -> Array:
        """Rotate real SH coefficients back from a z-aligned frame."""
        raise NotImplementedError("real SH rotations are implemented in Stage J3")

    def m2l_rot_scale(
        self: "RealSHBasis", sources: Array, deltas: Array, *, order: int
    ) -> Array:
        """Translate real SH multipoles to locals via rotate/scale M2L path."""
        raise NotImplementedError("real SH rotate+scale M2L is implemented in Stage J3")


__all__ = [
    "RealSHBasis",
    "complex_to_real_coeffs",
    "real_to_complex_coeffs",
    "n_real_sh_coeffs",
]
