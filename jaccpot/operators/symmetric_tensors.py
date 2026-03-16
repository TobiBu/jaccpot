"""Utilities for packed symmetric tensors in 3D.

These helpers provide static metadata and contraction routines used by
derivative-tower kernels. The representation keeps only unique symmetric
components, indexed by Cartesian exponent tuples ``(nx, ny, nz)`` with
``nx + ny + nz = order``.
"""

from __future__ import annotations

from functools import lru_cache, partial
from math import comb

import jax
import jax.numpy as jnp
from jaxtyping import Array


def symmetric_component_count(order: int, *, dim: int = 3) -> int:
    """Return number of unique symmetric tensor components."""
    if order < 0:
        raise ValueError("order must be non-negative")
    if dim <= 0:
        raise ValueError("dim must be positive")
    return comb(order + dim - 1, dim - 1)


@lru_cache(maxsize=None)
def symmetric_multi_indices_3d(order: int) -> tuple[tuple[int, int, int], ...]:
    """List 3D exponent tuples in deterministic packed order."""
    if order < 0:
        raise ValueError("order must be non-negative")
    combos: list[tuple[int, int, int]] = []
    for nx in range(order, -1, -1):
        for ny in range(order - nx, -1, -1):
            nz = order - nx - ny
            combos.append((nx, ny, nz))
    return tuple(combos)


def symmetric_order_offsets_3d(max_order: int) -> tuple[int, ...]:
    """Offsets into a packed derivative tower for orders ``0..max_order``."""
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    offsets = [0]
    total = 0
    for order in range(max_order + 1):
        total += symmetric_component_count(order, dim=3)
        offsets.append(total)
    return tuple(offsets)


@lru_cache(maxsize=None)
def _contraction_index_map_3d(order: int) -> tuple[tuple[int, int, int], ...]:
    """Map packed order-``order`` components to order-``order-1`` + axis."""
    if order <= 0:
        raise ValueError("order must be positive")
    high = symmetric_multi_indices_3d(order)
    low = symmetric_multi_indices_3d(order - 1)
    index = {combo: idx for idx, combo in enumerate(high)}

    out: list[tuple[int, int, int]] = []
    for nx, ny, nz in low:
        out.append(
            (
                index[(nx + 1, ny, nz)],
                index[(nx, ny + 1, nz)],
                index[(nx, ny, nz + 1)],
            )
        )
    return tuple(out)


def component_lift_index_map_3d(order: int) -> tuple[tuple[int, int, int], ...]:
    """Map packed order-``order`` components to axis-lifted order-``order+1``.

    For each packed exponent triple ``beta`` of total degree ``order``, returns
    the three packed indices corresponding to ``beta + e_x``, ``beta + e_y``,
    and ``beta + e_z`` at total degree ``order + 1``.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    return _contraction_index_map_3d(order + 1)


@partial(jax.jit, static_argnames=("order",))
def contract_symmetric_one_axis_3d(
    packed: Array,
    vector: Array,
    *,
    order: int,
) -> Array:
    """Contract one tensor axis with ``vector``.

    Parameters
    ----------
    packed:
        Symmetric order-``order`` tensor in packed representation.
    vector:
        Shape ``(3,)`` contraction vector.
    order:
        Tensor order. Must be positive.
    """
    if order <= 0:
        raise ValueError("order must be positive")
    gather = jnp.asarray(_contraction_index_map_3d(int(order)), dtype=jnp.int32)
    gathered = packed[gather]
    return jnp.sum(gathered * vector[None, :], axis=1)


__all__ = [
    "component_lift_index_map_3d",
    "contract_symmetric_one_axis_3d",
    "symmetric_component_count",
    "symmetric_multi_indices_3d",
    "symmetric_order_offsets_3d",
]
