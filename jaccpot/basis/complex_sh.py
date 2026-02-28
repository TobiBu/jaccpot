"""Complex spherical-harmonic basis adapter for existing solidfmm kernels."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array

from jaccpot.operators.complex_ops import (
    complex_rotation_blocks_from_z_solidfmm_batch,
    complex_rotation_blocks_to_z_solidfmm_batch,
    m2l_complex_reference_batch,
)
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

    def rotate_to_z(
        self: "ComplexSHBasis", coeffs: Array, directions: Array, *, order: int
    ) -> Array:
        """Rotate batched multipoles into a frame aligned to ``+z``."""
        coeffs_arr = self.pack_coeffs(coeffs, order=order)
        x, y, z = self._split_xyz(directions)
        to_blocks = complex_rotation_blocks_to_z_solidfmm_batch(
            x,
            y,
            z,
            order=int(order),
            dtype=coeffs_arr.dtype,
        )
        return self._apply_rotation_blocks(coeffs_arr, to_blocks)

    def rotate_from_z(
        self: "ComplexSHBasis", coeffs: Array, directions: Array, *, order: int
    ) -> Array:
        """Rotate batched multipoles from the ``+z`` frame back to world frame."""
        coeffs_arr = self.pack_coeffs(coeffs, order=order)
        x, y, z = self._split_xyz(directions)
        from_blocks = complex_rotation_blocks_from_z_solidfmm_batch(
            x,
            y,
            z,
            order=int(order),
            dtype=coeffs_arr.dtype,
        )
        return self._apply_rotation_blocks(coeffs_arr, from_blocks)

    def m2l_rot_scale(
        self: "ComplexSHBasis", sources: Array, deltas: Array, *, order: int
    ) -> Array:
        """Run existing batched complex M2L kernel for source multipoles."""
        src = self.pack_coeffs(sources, order=order)
        delta_arr = jnp.asarray(deltas)
        if delta_arr.ndim != 2 or int(delta_arr.shape[1]) != 3:
            raise ValueError("deltas must have shape (batch, 3)")
        return m2l_complex_reference_batch(src, delta_arr, order=int(order))

    @staticmethod
    def _split_xyz(directions: Array) -> tuple[Array, Array, Array]:
        directions_arr = jnp.asarray(directions)
        if directions_arr.ndim != 2 or int(directions_arr.shape[1]) != 3:
            raise ValueError("directions must have shape (batch, 3)")
        return directions_arr[:, 0], directions_arr[:, 1], directions_arr[:, 2]

    @staticmethod
    def _apply_rotation_blocks(coeffs: Array, blocks: list[Array]) -> Array:
        offsets = [0]
        for ell, block in enumerate(blocks):
            width = int(block.shape[-1])
            if width != 2 * ell + 1:
                raise ValueError("rotation block width does not match ell")
            offsets.append(offsets[-1] + width)

        def rotate_one(row: Array, row_blocks: list[Array]) -> Array:
            parts = []
            for ell, block in enumerate(row_blocks):
                start = offsets[ell]
                end = offsets[ell + 1]
                parts.append(block @ row[start:end])
            return jnp.concatenate(parts, axis=0)

        return jax.vmap(rotate_one)(coeffs, blocks)
