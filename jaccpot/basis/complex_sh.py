"""Complex spherical-harmonic basis adapter for existing solidfmm kernels."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array

from jaccpot.operators.complex_ops import m2l_complex_reference_batch
from jaccpot.operators.real_harmonics import sh_size


@dataclass(frozen=True)
class ComplexSHBasis:
    """Adapter exposing the current solidfmm complex basis through a common API.

    Coefficients use the existing packed solidfmm layout (ell-major blocks with
    ``m=-ell..+ell`` for each degree).
    """

    p_max: int = 32
    name: str = "complex"
    coefficient_ordering: str = "ell-major, m=-ell..+ell (packed complex SH)"
    runtime_expansion_basis: str = "solidfmm"

    def n_coeffs(self: "ComplexSHBasis", p: int) -> int:
        """Return packed coefficient count for order ``p``."""
        return int(sh_size(int(p)))

    def pack_coeffs(self: "ComplexSHBasis", coeffs: Array, *, order: int) -> Array:
        """Validate and return packed solidfmm complex coefficients."""
        coeffs_arr = jnp.asarray(coeffs)
        expected = self.n_coeffs(order)
        if int(coeffs_arr.shape[-1]) != expected:
            raise ValueError(
                f"expected last dimension {expected} for order={int(order)}, got {coeffs_arr.shape[-1]}"
            )
        return coeffs_arr

    def unpack_coeffs(self: "ComplexSHBasis", packed: Array, *, order: int) -> Array:
        """Return packed coefficients (complex basis already uses packed layout)."""
        return self.pack_coeffs(packed, order=order)

    def m2l_rot_scale(
        self: "ComplexSHBasis", sources: Array, deltas: Array, *, order: int
    ) -> Array:
        """Run existing batched complex M2L kernel for source multipoles."""
        src = self.pack_coeffs(sources, order=order)
        delta_arr = jnp.asarray(deltas)
        if delta_arr.ndim != 2 or int(delta_arr.shape[1]) != 3:
            raise ValueError("deltas must have shape (batch, 3)")
        return m2l_complex_reference_batch(src, delta_arr, order=int(order))
