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
    """Evaluate leaf-local expansions for every particle in sorted order.

    A thin adapter: it unpacks ``state`` and defers to
    :func:`~jaccpot.runtime.kernels._evaluate._evaluate_local_expansions_for_particles`,
    pinning ``expansion_basis="solidfmm"`` and ``max_acc_derivative_order=0``.
    Neither is a parameter here, so this wrapper cannot reach the real basis or
    the derivative outputs.

    Far field only -- the near field is a separate additive term.

    .. warning::
       **The return annotation is wrong, and the ``Returns`` section below
       matches the annotation rather than reality**, because pydoclint compares
       the two textually. The delegate is annotated
       ``tuple[Array, Optional[Array], Optional[PackedAccelerationDerivatives]]``
       and genuinely returns ``None`` in the last two slots: the potential is
       ``None`` unless ``return_potential``, and the derivative tuple is *always*
       ``None`` here because ``max_acc_derivative_order`` is pinned to 0.
       Unpacking this as three arrays will fail. The function currently has no
       callers anywhere in ``jaccpot/``, ``tests/`` or ``bench/``, which is why
       nothing has tripped over it; correcting the annotation needs its own
       change with its own test.

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
        ``[N, 3]`` accelerations in **sorted** (tree) order, not the caller's
        input order. This one is genuinely an ``Array``.
    Array
        Potentials ``[N]`` -- but ``None`` when ``return_potential`` is false.
        See the warning above.
    Array
        Packed acceleration derivatives -- always ``None`` here. See the warning
        above.
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
