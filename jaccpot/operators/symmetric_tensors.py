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
    """Return number of unique symmetric tensor components.

    ``C(order + dim - 1, dim - 1)`` -- the multiset count, i.e. how many
    exponent tuples of that total degree exist.

    Parameters
    ----------
    order : int
        Tensor order ``n``. Must be non-negative.
    dim : int
        Spatial dimension. Must be positive; the rest of this module assumes 3.

    Returns
    -------
    int
        Packed component count.

    Raises
    ------
    ValueError
        If ``order`` is negative or ``dim`` is not positive.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    if dim <= 0:
        raise ValueError("dim must be positive")
    return comb(order + dim - 1, dim - 1)


@lru_cache(maxsize=None)
def symmetric_multi_indices_3d(order: int) -> tuple[tuple[int, int, int], ...]:
    """List 3D exponent tuples in deterministic packed order.

    This ordering **defines** the packed layout for every function here, and it
    descends in ``nx`` then ``ny``. It is the exact *reverse* of
    :func:`~jaccpot.operators.multipole_utils.multi_index_tuples`, which ascends
    (checked for orders 0-8), so the two packed representations in this package
    are not interchangeable and indices from one must never be used against the
    other.

    Memoized with an unbounded ``lru_cache``: one tuple per order, and orders are
    small and few.

    Parameters
    ----------
    order : int
        Tensor order ``n``. Must be non-negative.

    Returns
    -------
    tuple[tuple[int, int, int], ...]
        The ``(nx, ny, nz)`` triples with ``nx + ny + nz == order``, in packed
        order.

    Raises
    ------
    ValueError
        If ``order`` is negative.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    combos: list[tuple[int, int, int]] = []
    for nx in range(order, -1, -1):
        for ny in range(order - nx, -1, -1):
            nz = order - nx - ny
            combos.append((nx, ny, nz))
    return tuple(combos)


def symmetric_order_offsets_3d(max_order: int) -> tuple[int, ...]:
    """Offsets into a packed derivative tower for orders ``0..max_order``.

    Parameters
    ----------
    max_order : int
        Highest order in the tower. Must be non-negative.

    Returns
    -------
    tuple[int, ...]
        ``max_order + 2`` cumulative offsets: entry ``k`` is where order ``k``
        starts, and the last entry is the total length. Having the end sentinel
        is what lets a caller slice order ``k`` as ``offsets[k]:offsets[k + 1]``.

    Raises
    ------
    ValueError
        If ``max_order`` is negative.
    """
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
    """Map packed order-``order`` components to order-``order-1`` + axis.

    Row ``i`` holds the order-``order`` packed indices of ``beta + e_x``,
    ``beta + e_y`` and ``beta + e_z``, where ``beta`` is the ``i``-th order-``order
    - 1`` triple. Read one way this is a contraction map; read the other it is a
    lift, which is why :func:`component_lift_index_map_3d` is a thin alias for it
    at ``order + 1``.

    Memoized with an unbounded ``lru_cache``.

    Parameters
    ----------
    order : int
        Tensor order ``n``. Must be positive.

    Returns
    -------
    tuple[tuple[int, int, int], ...]
        One index triple per order-``n-1`` component.

    Raises
    ------
    ValueError
        If ``order`` is not positive.
    """
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

    Parameters
    ----------
    order : int
        Tensor order ``n``. Must be non-negative -- unlike
        :func:`_contraction_index_map_3d`, which it defers to at ``order + 1``,
        this accepts ``0``.

    Returns
    -------
    tuple[tuple[int, int, int], ...]
        One index triple per order-``order`` component.

    Raises
    ------
    ValueError
        If ``order`` is negative.
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
    packed : Array
        Symmetric order-``n`` tensor in the packed representation,
        ``[symmetric_component_count(n)]``, ordered by
        :func:`symmetric_multi_indices_3d`.
    vector : Array
        Contraction vector ``[3]``, in the same Cartesian frame as the tensor.
    order : int
        Tensor order ``n``, which must be positive. Static under ``jit``
        (declared in ``static_argnames``) -- the gather index map is built from
        it in Python.

    Returns
    -------
    Array
        The contracted symmetric tensor of order ``n-1``, packed,
        ``[symmetric_component_count(n - 1)]``. Dtype follows the
        ``packed * vector`` promotion.

    Raises
    ------
    ValueError
        If ``order <= 0``. Order 0 has no axis to contract; because ``order`` is
        static this fires at trace time.

    Notes
    -----
    Differentiable in ``packed`` and ``vector`` -- a gather followed by a
    multiply and a single ``sum`` over the length-3 axis, with no degenerate
    branch. The reduction is that one ``sum`` over three terms; its order is
    part of the numerics and should not be rewritten.
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
