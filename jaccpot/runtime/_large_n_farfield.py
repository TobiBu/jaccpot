"""Far-field helpers for the slim large-N runtime path."""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array

from ._large_n_types import LargeNPreparedState
from .dtypes import INDEX_DTYPE
from .kernels._shared import PackedAccelerationDerivatives


def evaluate_large_n_farfield(
    state: LargeNPreparedState,
    *,
    return_potential: bool,
) -> tuple[Array, Optional[Array], Optional[PackedAccelerationDerivatives]]:
    """Evaluate leaf-local expansions for every particle in sorted order.

    A thin adapter: it unpacks ``state`` and defers to
    :func:`~jaccpot.runtime.kernels._evaluate._evaluate_local_expansions_for_particles`,
    pinning ``expansion_basis="solidfmm"`` and ``max_acc_derivative_order=0``.
    Neither is a parameter here, so this wrapper cannot reach the real basis or
    the derivative outputs -- which is why the third return slot is always
    ``None``.

    Far field only -- the near field is a separate additive term.

    Parameters
    ----------
    state : LargeNPreparedState
        Prepared large-N state. Read for the local expansions, the sorted
        positions, the neighbour list's leaf ids, the tree's node ranges, and the
        leaf width and order -- so the expansions must already have been built
        into it; this does no downward sweep of its own.
    return_potential : bool
        Whether to compute per-particle potentials. Static under ``jit``: it
        selects a branch inside the delegate that changes what is returned.

    Returns
    -------
    Array
        ``[N, 3]`` field in **sorted** (tree) order, not the caller's input
        order. As in the delegate, this is the gradient, not the acceleration.
    Optional[Array]
        Potentials ``[N]``, or ``None`` when ``return_potential`` is false.
    Optional[PackedAccelerationDerivatives]
        Packed acceleration derivatives -- **always** ``None`` here, because
        ``max_acc_derivative_order`` is pinned to 0 above. The hint stays as wide
        as the delegate's own so the two signatures line up.
    """
    # Import lazily to avoid a circular dependency during module import.
    from .kernels.core import _evaluate_local_expansions_for_particles

    return _evaluate_local_expansions_for_particles(
        state.local_data,
        state.positions_sorted,
        leaf_nodes=jnp.asarray(state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE),
        node_ranges=jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE),
        max_leaf_size=int(state.max_leaf_size),
        order=int(state.local_data.order),
        expansion_basis="solidfmm",
        return_potential=bool(return_potential),
        max_acc_derivative_order=0,
    )
