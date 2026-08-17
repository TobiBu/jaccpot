"""Utilities for handling symmetric multipole tensors.

This module provides helpers for working with the packed triangular
representation discussed in recent FMM optimisation work.  The packed
layout stores all Cartesian symmetric tensor components of order ``l``
contiguously using a triangular indexing scheme, which avoids redundant
entries while remaining friendly to vectorised JAX code.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, jaxtyped

from .dtypes import INDEX_DTYPE

MAX_MULTIPOLE_ORDER = 4


def multi_index_tuples(level: int) -> tuple[tuple[int, int, int], ...]:
    """Return tuples ``(i, j, k)`` satisfying ``i + j + k = level``.

    Parameters
    ----------
    level : int
        Tensor order ``l``. Must be non-negative.

    Returns
    -------
    tuple[tuple[int, int, int], ...]
        All multi-indices of that order, in the same sequence as
        :func:`triangular_indices` produces them, so the two are
        interchangeable as a packed-layout ordering.

    Raises
    ------
    ValueError
        If ``level`` is negative.
    """

    lvl = int(level)
    if lvl < 0:
        raise ValueError("level must be >= 0")
    combos = []
    for i in range(lvl + 1):
        for j in range(lvl + 1 - i):
            combos.append((i, j, lvl - i - j))
    return tuple(combos)


def multi_index_factorial(combo: tuple[int, int, int]) -> int:
    """Return ``i! * j! * k!`` for a multi-index tuple.

    This is the symmetry weight of a Cartesian multipole component: the number
    of distinct index permutations the packed layout collapses into one slot.

    Parameters
    ----------
    combo : tuple[int, int, int]
        Multi-index ``(i, j, k)``.

    Returns
    -------
    int
        The product of the three factorials.
    """

    i, j, k = combo
    return math.factorial(i) * math.factorial(j) * math.factorial(k)


@jaxtyped(typechecker=beartype)
def multi_power(vec: Array, combo: tuple[int, int, int]) -> Array:
    """Return ``vec[0]^i * vec[1]^j * vec[2]^k`` for ``combo = (i, j, k)``.

    Zero exponents are skipped rather than raised to the power of zero, which
    keeps the result exact at ``vec == 0`` instead of relying on ``0 ** 0``.

    Parameters
    ----------
    vec : Array
        Length-3 vector. Traced values are fine; ``combo`` drives Python-level
        branching, ``vec`` does not.
    combo : tuple[int, int, int]
        Exponents, static under ``jit``.

    Returns
    -------
    Array
        Scalar monomial, same dtype as ``vec``.
    """

    value = jnp.array(1.0, dtype=vec.dtype)
    if combo[0]:
        value = value * vec[0] ** combo[0]
    if combo[1]:
        value = value * vec[1] ** combo[1]
    if combo[2]:
        value = value * vec[2] ** combo[2]
    return value


@jaxtyped(typechecker=beartype)
def level_size(level: int) -> int:
    """Return coefficient count for a symmetric tensor of order ``level``.

    Parameters
    ----------
    level : int
        Tensor order ``l``.

    Returns
    -------
    int
        ``(l + 1)(l + 2) / 2``, the size of one packed level.
    """

    level_int = int(level)
    return (level_int + 1) * (level_int + 2) // 2


@jaxtyped(typechecker=beartype)
def level_offset(level: int) -> int:
    """Return the packed offset for order ``level``.

    Offsets accumulate contributions from lower orders using the closed
    form ``level(level+1)(level+2)/6``.

    Parameters
    ----------
    level : int
        Tensor order ``l``.

    Returns
    -------
    int
        Index of the first slot of level ``l`` in the concatenated layout.
    """

    level_int = int(level)
    return (level_int * (level_int + 1) * (level_int + 2)) // 6


@jaxtyped(typechecker=beartype)
def total_coefficients(max_order: int) -> int:
    """Return total packed length for orders ``0..max_order`` inclusive.

    Equal to ``level_offset(max_order + 1)`` -- the two are the same closed
    form read from opposite ends.

    Parameters
    ----------
    max_order : int
        Highest order included.

    Returns
    -------
    int
        ``(p + 1)(p + 2)(p + 3) / 6`` for ``p = max_order``.
    """

    order = int(max_order)
    return (order + 1) * (order + 2) * (order + 3) // 6


@jaxtyped(typechecker=beartype)
def triangular_index(level: int, i: int, j: int) -> int:
    """Map Cartesian indices to the packed triangular index.

    The mapping assumes ``i >= 0``, ``j >= 0`` and ``i + j <= level``.  The
    remaining index is ``k = level - i - j``.

    Parameters
    ----------
    level : int
        Tensor order ``l``.
    i : int
        First Cartesian index.
    j : int
        Second Cartesian index.

    Returns
    -------
    int
        Slot within level ``l``, i.e. relative to :func:`level_offset`, not an
        index into the whole concatenated buffer.

    Raises
    ------
    ValueError
        If the indices are negative or ``i + j > level``.
    """

    lvl = int(level)
    ii = int(i)
    jj = int(j)
    if ii < 0 or jj < 0 or ii + jj > lvl:
        raise ValueError("Invalid triangular indices for given level")
    prefix = ii * (lvl + 1) - (ii * (ii - 1)) // 2
    return prefix + jj


@jaxtyped(typechecker=beartype)
def triangular_indices(level: int) -> Array:
    """Enumerate all ``(i, j, k)`` tuples for a given tensor order.

    The array form of :func:`multi_index_tuples`, in the same order, and it is
    this ordering that defines the packed layout used by :func:`pack_tensor`.

    Parameters
    ----------
    level : int
        Tensor order ``l``. Static under ``jit``: it fixes the output length.

    Returns
    -------
    Array
        ``[level_size(level), 3]`` of ``INDEX_DTYPE`` multi-indices.
    """

    lvl = int(level)
    grid_i = jnp.arange(lvl + 1, dtype=INDEX_DTYPE)
    grid_j = jnp.arange(lvl + 1, dtype=INDEX_DTYPE)
    ii, jj = jnp.meshgrid(grid_i, grid_j, indexing="ij")
    mask = ii + jj <= lvl
    i_vals = ii[mask]
    j_vals = jj[mask]
    k_vals = lvl - i_vals - j_vals
    return jnp.stack([i_vals, j_vals, k_vals], axis=1)


@jaxtyped(typechecker=beartype)
def pack_tensor(level: int, tensor: Array) -> Array:
    """Pack a symmetric Cartesian tensor of order ``level``.

    Parameters
    ----------
    level : int
        Tensor order ``l``. Static under ``jit``: it fixes the output length and
        is read with ``int()``.
    tensor : Array
        Cartesian components ``[l+1, l+1, l+1]``. Only the entries with
        ``i + j + k == l`` are read; everything off that simplex is ignored
        rather than checked, so a caller who fills the wrong slots gets zeros
        with no error.

    Returns
    -------
    Array
        1-D packed representation, length ``level_size(level)``, ordered by
        :func:`triangular_indices`. Same dtype as ``tensor``.

    Raises
    ------
    ValueError
        If ``tensor`` is not exactly ``[l+1, l+1, l+1]``. A static-shape check,
        so it fires at trace time and never inside a compiled step.

    Notes
    -----
    A pure gather, so differentiable in ``tensor`` (the VJP is the matching
    scatter, which is what :func:`unpack_tensor` computes) and independent of
    ``level`` as a differentiable quantity.
    """

    lvl = int(level)
    if tensor.shape != (lvl + 1, lvl + 1, lvl + 1):
        raise ValueError("tensor must have shape (level+1, level+1, level+1)")
    idx = triangular_indices(lvl)
    i_vals, j_vals, k_vals = idx[:, 0], idx[:, 1], idx[:, 2]
    return tensor[i_vals, j_vals, k_vals]


@jaxtyped(typechecker=beartype)
def unpack_tensor(level: int, data: Array) -> Array:
    """Unpack a packed triangular buffer back into Cartesian components.

    The inverse of :func:`pack_tensor` on the simplex, and its VJP: a scatter
    where packing is a gather. Round-tripping is exact, but only on the
    ``i + j + k == level`` entries -- everything off the simplex comes back
    zero, since packing never read it.

    Parameters
    ----------
    level : int
        Tensor order ``l``. Static under ``jit``: it fixes the output shape.
    data : Array
        Packed buffer whose last axis has length ``level_size(level)``.

    Returns
    -------
    Array
        Cartesian components ``[l+1, l+1, l+1]``, same dtype as ``data``, zero
        off the simplex.

    Raises
    ------
    ValueError
        If the last axis of ``data`` is not ``level_size(level)``. A static-shape
        check, so it fires at trace time and never inside a compiled step.
    """

    lvl = int(level)
    expected = level_size(lvl)
    if data.shape[-1] != expected:
        raise ValueError(f"Packed data length {data.shape[-1]} != expected {expected}")
    tensor = jnp.zeros((lvl + 1, lvl + 1, lvl + 1), dtype=data.dtype)
    idx = triangular_indices(lvl)
    i_vals, j_vals, k_vals = idx[:, 0], idx[:, 1], idx[:, 2]
    tensor = tensor.at[i_vals, j_vals, k_vals].set(data)
    return tensor


# Cached ``(i, j, k)`` tuples by tensor order for hot-path reuse.
LOCAL_LEVEL_COMBOS: dict[int, tuple[tuple[int, int, int], ...]] = {
    level: multi_index_tuples(level) for level in range(MAX_MULTIPOLE_ORDER + 1)
}


# Cached inverse multi-index factorials used in local-expansion recurrences.
LOCAL_COMBO_INV_FACTORIAL: dict[tuple[int, int, int], float] = {
    combo: 1.0 / multi_index_factorial(combo)
    for combos in LOCAL_LEVEL_COMBOS.values()
    for combo in combos
}


__all__ = [
    "MAX_MULTIPOLE_ORDER",
    "LOCAL_COMBO_INV_FACTORIAL",
    "LOCAL_LEVEL_COMBOS",
    "multi_index_tuples",
    "multi_index_factorial",
    "multi_power",
    "level_size",
    "level_offset",
    "total_coefficients",
    "triangular_index",
    "triangular_indices",
    "pack_tensor",
    "unpack_tensor",
]
