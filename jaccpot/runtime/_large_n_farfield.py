"""Far-field helpers for the slim large-N runtime path."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from ._large_n_types import LargeNPreparedState
from .dtypes import INDEX_DTYPE


def evaluate_large_n_farfield(
    state: LargeNPreparedState,
    *,
    return_potential: bool,
) -> tuple[Array, Array, Array]:
    """Evaluate leaf-local expansions for every particle in sorted order."""
    # Import lazily to avoid a circular dependency during module import.
    from ._fmm_impl import _evaluate_local_expansions_for_particles

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
