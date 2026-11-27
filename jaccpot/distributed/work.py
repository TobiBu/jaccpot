"""Per-particle work of the fused mesh force, for work-balanced migration cuts (M4).

The count-balanced cuts (:func:`jaccpot.distributed.migration.balanced_splitters_shard`)
give every device the same number of particles. A device holding a dense core then
does more near-field work than one holding a halo: the force's time is the slowest
device's. :func:`row_work` estimates each particle's share of its device's force from
the force's own topology, inside the force's ``shard_map`` (the ``targets`` hook of
:meth:`~jaccpot.distributed.template_force.FusedMeshForce.shard`):

* **near**: the particle-particle interactions of its leaf -- the live particles of
  every leaf in its leaf's near list (the topology's ``neighbor_list``) and of every
  peer row the cross exchange pairs with it (the cross near sink);
* **far**: the device's local far pairs (the walk's ``far_needed``) times
  ``far_weight`` (one M2L in units of one particle-particle interaction), spread
  evenly over its live particles -- a far pair costs the same wherever it is, and
  the far lists are not kept per leaf.

:func:`jaccpot.distributed.migration.work_balanced_splitters_shard` then cuts the
Morton curve at equal cumulative work. ``far_weight`` is a guess (an M2L at p 4-6 is
some tens of pair interactions); HoreKa's per-stage profile calibrates it (M5).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

__all__ = [
    "DEFAULT_FAR_WEIGHT",
    "row_work",
    "work_far_weight",
]

#: one far pair (M2L) in units of one particle-particle interaction
DEFAULT_FAR_WEIGHT = 16.0


def work_far_weight() -> float:
    """``JACCPOT_MESH_WORK_FAR_WEIGHT`` (default :data:`DEFAULT_FAR_WEIGHT`).

    Returns
    -------
    float
        The weight of one far pair.
    """
    from jaccpot._env import env_float

    return float(env_float("JACCPOT_MESH_WORK_FAR_WEIGHT", DEFAULT_FAR_WEIGHT))


def _live_counts(tree: Any) -> Array:
    """Live particles under every node (a padded shard's ranges reach its dead rows).

    Parameters
    ----------
    tree : Any
        The topology's tree (``node_ranges``, ``left_child``).

    Returns
    -------
    Array
        ``(total_nodes,)`` int32.
    """
    from jaccpot.distributed.cross import _live_node_ends

    nr = jnp.asarray(tree.node_ranges)
    ends = _live_node_ends(tree)
    return jnp.maximum(ends - nr[:, 0] + 1, 0).astype(jnp.int32)


def row_work(
    topology: Any,
    near_sink: Any,
    live: Array,
    *,
    cap: int,
    far_weight: float = DEFAULT_FAR_WEIGHT,
) -> Array:
    """Each input row's work in the force that built ``topology`` (inside ``shard_map``).

    Parameters
    ----------
    topology : Any
        The device's :class:`~jaccpot.runtime.fused_topology.FusedTopology` (its
        ``prepared.tree``, ``prepared.neighbor_list`` and ``walk_needs``).
    near_sink : Any
        The cross hook's near sink of the same trace (``target_node``,
        ``source_row``, ``row_count``); ``None`` on one device.
    live : Array
        Live rows (leading).
    cap : int
        Rows of the shard. Static.
    far_weight : float
        One far pair in particle-particle interactions. Static.

    Returns
    -------
    Array
        ``(cap,)`` float32, zero on the padding.
    """
    from jaccpot.distributed.cross import _row_of_slot

    idx = jnp.int32
    prepared = topology.prepared
    tree = prepared.tree
    nl = prepared.neighbor_list
    n_node = _live_counts(tree)
    total = int(n_node.shape[0])
    num_internal = int(jnp.asarray(tree.left_child).shape[0])

    # near: per leaf, the live particles of its near leaves
    leaf_nodes = jnp.asarray(nl.leaf_indices).astype(idx)
    counts = jnp.asarray(nl.counts).astype(idx)
    neighbors = jnp.asarray(nl.neighbors).astype(idx)
    owner, _, live_e, _ = _row_of_slot(counts, int(neighbors.shape[0]))
    src = jnp.where(live_e, n_node[jnp.clip(neighbors, 0, total - 1)], 0)
    near_slot = jax.ops.segment_sum(src, owner, num_segments=int(counts.shape[0]))
    per_node = (
        jnp.zeros((total,), jnp.float32)
        .at[jnp.clip(leaf_nodes, 0, total - 1)]
        .add(jnp.where(leaf_nodes >= 0, near_slot, 0).astype(jnp.float32))
    )
    if near_sink is not None and "target_node" in near_sink:
        # the cross near pairs: (my leaf, received row), each row's particle count
        tgt = jnp.asarray(near_sink["target_node"]).astype(idx)
        row = jnp.asarray(near_sink["source_row"]).astype(idx)
        rc = jnp.asarray(near_sink["row_count"]).astype(idx)
        ok = (tgt >= 0) & (tgt < total) & (row >= 0) & (row < rc.shape[0])
        add = jnp.where(ok, rc[jnp.clip(row, 0, rc.shape[0] - 1)], 0)
        per_node = per_node.at[jnp.where(ok, tgt, total)].add(
            add.astype(jnp.float32), mode="drop"
        )

    # each sorted particle's leaf (the leaves are in Morton order)
    nr = jnp.asarray(tree.node_ranges)
    starts = nr[num_internal:, 0]
    n_live = jnp.minimum(jnp.asarray(live).astype(idx), cap)
    j = jnp.arange(cap, dtype=idx)
    from jaccpot._searchsorted import searchsorted_method

    leaf = (
        num_internal
        + jnp.searchsorted(
            starts, j, side="right", method=searchsorted_method()
        ).astype(idx)
        - 1
    )
    leaf = jnp.clip(leaf, num_internal, total - 1)
    work_sorted = jnp.where(j < n_live, per_node[leaf], 0.0)
    walk_needs = getattr(topology, "walk_needs", None)
    if walk_needs is not None and float(far_weight) > 0.0:
        far = jnp.asarray(walk_needs)[0].astype(jnp.float32)
        per = far * float(far_weight) / jnp.maximum(n_live, 1).astype(jnp.float32)
        work_sorted = work_sorted + jnp.where(j < n_live, per, 0.0)
    rows = jnp.asarray(tree.particle_indices).astype(idx)
    return (
        jnp.zeros((cap,), jnp.float32)
        .at[jnp.where(j < n_live, rows, cap)]
        .set(work_sorted, mode="drop")
    )
