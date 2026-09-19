"""The cross-domain hook: what a device imports, assembled between the sweeps.

Plan ``~/.claude/plans/phase-c-interleaved-cross-field.md``. This is the callable
handed to :func:`~jaccpot.distributed.fused.fused_force_step` as ``cross_hook``. It
runs at the one point where the exchange belongs -- the multipoles exist and the
downward sweep has not consumed them -- and returns the imported far sources for the
M2L to concatenate, so the L2L cascade still runs once.

Every piece it calls is built and tested in yggdrax: ``occupancy_cut`` (the summary),
``export_walk`` (what this device owes everyone, in one walk), ``build_send_buffers``
(grouped and deduplicated), ``exchange_export_list`` (two ragged rounds), and
``receiver_interaction_lists`` (the imported per-cell lists expanded onto local
targets).

**The index convention is load-bearing.** Imported sources are returned to sit at
``[n_local, n_local + n_import)`` in the concatenated multipole array. Every local
index is then strictly below every imported one, which is what makes the walk's
``(min, max)`` canonicalisation an exact (local target, imported source) ordering.
Reversed, the M2L expands the wrong way round with a plausible-looking result.

**At ndev = 1 this returns nothing**, because a device never exports to itself -- which
is what makes it testable: the whole pipeline runs and the force must not move.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from yggdrax.distributed.comm import AXIS_NAME
from yggdrax.distributed.export import build_send_buffers, export_walk
from yggdrax.distributed.import_cells import (
    exchange_export_list,
    receiver_interaction_lists,
)
from yggdrax.distributed.summary import occupancy_cut

__all__ = ["CrossCapacities", "make_cross_hook"]


class CrossCapacities:
    """Static capacities for one cross exchange. Every one is an overflow risk.

    Attributes are read at trace time and fix every buffer shape, so they cannot be
    derived from what actually arrives. Over-allocate and read the flags.
    """

    def __init__(
        self,
        *,
        max_cells: int = 1024,
        max_leaves_per_cell: int = 4,
        export_far_cap: int = 1 << 16,
        export_near_cap: int = 1 << 16,
        send_node_cap: int = 1 << 13,
        send_csr_cap: int = 1 << 16,
        recv_node_cap: int = 1 << 13,
        recv_csr_cap: int = 1 << 16,
        walk_queue: int = 1 << 16,
        recv_far_cap: int = 1 << 17,
        recv_near_cap: int = 1 << 17,
    ) -> None:
        self.max_cells = int(max_cells)
        self.max_leaves_per_cell = int(max_leaves_per_cell)
        self.export_far_cap = int(export_far_cap)
        self.export_near_cap = int(export_near_cap)
        self.send_node_cap = int(send_node_cap)
        self.send_csr_cap = int(send_csr_cap)
        self.recv_node_cap = int(recv_node_cap)
        self.recv_csr_cap = int(recv_csr_cap)
        self.walk_queue = int(walk_queue)
        self.recv_far_cap = int(recv_far_cap)
        self.recv_near_cap = int(recv_near_cap)


def make_cross_hook(
    *,
    ndev: int,
    theta: float,
    caps: Optional[CrossCapacities] = None,
    mac_type: str = "dehnen",
    axis_name: str = AXIS_NAME,
    record: Optional[dict] = None,
) -> Callable[[Any], Optional[tuple]]:
    """Build the ``cross_hook`` for a mesh of ``ndev`` devices.

    Parameters
    ----------
    ndev:
        Mesh size. At 1 the hook still runs everything and returns an empty import.
    theta:
        MAC parameter -- must match what the local lane evaluates with, since the
        sender's export decisions and the receiver's expansion both use it.
    caps:
        Static capacities; see :class:`CrossCapacities`.
    mac_type:
        MAC variant for the export and receiver walks. Static.
    axis_name:
        Mesh axis; must match the enclosing ``shard_map``.
    record:
        Optional dict the hook writes diagnostics into (call count, import sizes,
        overflow flags). For probes -- it is host state, so it holds tracers under
        jit and concrete values only when the hook runs eagerly.

    Returns
    -------
    Callable
        ``hook(tree_artifacts) -> (multipoles, centers, src, tgt) | None``, with the
        imported sources indexed from ``n_local``.
    """
    cap = caps if caps is not None else CrossCapacities()

    def hook(tree_artifacts: Any) -> Optional[tuple]:
        tree = tree_artifacts.tree
        upward = tree_artifacts.upward
        geom = upward.geometry
        mp = upward.multipoles

        parent = jnp.asarray(tree.parent)
        n_local = int(jnp.asarray(mp.packed).shape[0])
        num_internal = int(jnp.asarray(tree.left_child).shape[0])

        summary = occupancy_cut(
            parent,
            jnp.asarray(tree.node_ranges),
            num_internal,
            max_leaves=cap.max_leaves_per_cell,
            capacity=cap.max_cells,
        )
        cells = summary.cells
        live = jnp.arange(cap.max_cells) < summary.num_cells
        safe = jnp.where(live, cells, 0)
        my_cen = jnp.where(live[:, None], jnp.asarray(geom.center)[safe], 0.0)
        my_rad = jnp.where(live, jnp.asarray(geom.radius)[safe], 0.0)

        # every device's summary, so a sender can decide unilaterally
        all_cen = jax.lax.all_gather(my_cen, axis_name, tiled=False)
        all_rad = jax.lax.all_gather(my_rad, axis_name, tiled=False)
        all_act = jax.lax.all_gather(live, axis_name, tiled=False)
        me = jax.lax.axis_index(axis_name)

        idx = parent.dtype
        leaf_fill = jnp.full((n_local - num_internal,), -1, idx)
        left = jnp.concatenate([jnp.asarray(tree.left_child, idx), leaf_fill])
        right = jnp.concatenate([jnp.asarray(tree.right_child, idx), leaf_fill])

        ex = export_walk(
            left,
            right,
            jnp.asarray(geom.center),
            jnp.asarray(geom.radius),
            jnp.argmin(parent).astype(idx),
            all_cen,
            all_rad,
            all_act,
            float(theta),
            me,
            max_pair_queue=cap.walk_queue,
            far_cap=cap.export_far_cap,
            near_cap=cap.export_near_cap,
            mac_type=mac_type,
        )

        sb = build_send_buffers(
            ex.far_cell,
            ex.far_node,
            ex.far_count,
            ndev=ndev,
            max_cells=cap.max_cells,
            num_nodes=n_local,
            node_capacity=cap.send_node_cap,
            csr_capacity=cap.send_csr_cap,
        )
        rows = jnp.clip(sb.node_rows, 0, n_local - 1)
        alive = (sb.node_rows >= 0)[:, None]
        payload = jnp.concatenate(
            [
                jnp.where(alive, jnp.asarray(mp.packed)[rows], 0.0),
                jnp.where(alive, jnp.asarray(mp.centers)[rows], 0.0),
            ],
            axis=1,
        )

        got = exchange_export_list(
            payload,
            sb.node_sizes,
            sb.csr_cell,
            sb.csr_row,
            sb.csr_sizes,
            payload_capacity=cap.recv_node_cap,
            csr_capacity=cap.recv_csr_cap,
            ndev=ndev,
            axis_name=axis_name,
        )

        n_coeff = int(jnp.asarray(mp.packed).shape[1])
        imp_mp = got.payload[:, :n_coeff]
        imp_cen = got.payload[:, n_coeff:]

        combined_left = jnp.concatenate([left, jnp.full((cap.recv_node_cap,), -1, idx)])
        combined_right = jnp.concatenate(
            [right, jnp.full((cap.recv_node_cap,), -1, idx)]
        )
        combined_cen = jnp.concatenate([jnp.asarray(geom.center), imp_cen])
        combined_rad = jnp.concatenate(
            [
                jnp.asarray(geom.radius),
                jnp.zeros((cap.recv_node_cap,), geom.radius.dtype),
            ]
        )

        rl = receiver_interaction_lists(
            combined_left,
            combined_right,
            combined_cen,
            combined_rad,
            n_local,
            cells,
            got.csr_cell,
            got.csr_row,
            got.num_csr,
            float(theta),
            max_pair_queue=cap.walk_queue,
            far_cap=cap.recv_far_cap,
            near_cap=cap.recv_near_cap,
            mac_type=mac_type,
        )

        if record is not None:
            record["export_far"] = ex.far_count
            record["send_nodes"] = jnp.sum(sb.node_sizes)
            record["recv_nodes"] = got.num_payload
            record["recv_csr"] = got.num_csr
            record["far_pairs"] = rl.far_count
            record["overflow"] = (
                ex.far_overflow
                | ex.near_overflow
                | ex.queue_overflow
                | sb.node_overflow
                | sb.csr_overflow
                | rl.far_overflow
                | rl.queue_overflow
            )

        # -1 padding on the pair list is dropped by the CSR build; the imported
        # sources are rebased to sit ABOVE every local index, which is what the
        # (min, max) canonicalisation downstream depends on
        live_pair = jnp.arange(rl.far_target.shape[0]) < rl.far_count
        src = jnp.where(live_pair, jnp.asarray(n_local) + rl.far_source, -1)
        tgt = jnp.where(live_pair, rl.far_target, -1)
        return imp_mp, imp_cen, src, tgt

    return hook
