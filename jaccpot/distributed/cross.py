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

**The FAR half is interleaved; the NEAR half is added afterwards, and that asymmetry is
deliberate.** A far contribution arrives as a local EXPANSION, so adding it after the
force would need a second L2L cascade down to the leaves -- which is the whole reason
the far exchange sits between the sweeps. A near contribution is a direct sum evaluated
at the particles, with no cascade behind it, so computing it as its own small kernel and
adding the result costs one launch over a pool that Phase 3.1 measured at 1.4-4.3 % of
the local near field. Gravity is linear in the sources, so the sum is exact either way;
only the cost differs, and for the near half it differs the other way.
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


_CROSS_MAC_GEOMETRY_ENV = "JACCPOT_CROSS_MAC_GEOMETRY"


def _cross_mac_geometry_mode() -> str:
    """``"com"`` (default: the geometry the lane's own walk uses) or ``"aabb"`` (the
    box geometry this exchange used before Task 1; kept as a control)."""
    import os

    raw = os.environ.get(_CROSS_MAC_GEOMETRY_ENV, "com").strip().lower()
    if raw not in ("com", "aabb"):
        raise ValueError(f"{_CROSS_MAC_GEOMETRY_ENV} must be 'com' or 'aabb', got {raw!r}")
    return raw


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
        leaf_width: int = 64,
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
        self.leaf_width = int(leaf_width)
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
    near_sink: Optional[dict] = None,
    near_theta: Optional[float] = None,
    export_theta: Optional[float] = None,
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
    near_sink:
        Optional dict the hook writes the imported NEAR payload into -- the leaf
        pool and the (local target leaf, imported leaf) pairs. The caller evaluates
        that term after the force and adds it; see the module docstring for why the
        near half is added rather than interleaved. Holds tracers under jit, so it
        must be consumed inside the same trace.
    near_theta:
        MAC parameter of the receiver's walk over the NEAR import; ``None`` uses
        ``theta``. ``0.0`` makes every near-imported pair bottom out as a direct
        sum, which serves the pairs the sender exported as near and the receiver
        would otherwise call far (unservable: no multipole travels with a near
        leaf). Correct, and a measurement of that hole rather than the fix for it.
    export_theta:
        MAC parameter of the SENDER's export walk alone; ``None`` uses ``theta``.
        A BISECTION knob, not a setting: ``0.0`` makes every exported pair bottom
        out at a sender leaf, so the far list is empty and the whole cross field
        travels as particles and is summed directly (with ``near_theta=0``). If
        the error then matches the single-GPU lane, the residual lives in the
        multipole path (import, M2L, cascade); if it does not, it lives elsewhere.

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
        mp = upward.multipoles
        # THE GEOMETRY THE MAC IS TESTED ON MUST BE THE ONE THE EXPANSIONS USE.
        # The lane's real-basis sweep expands about the COM (`center_mode='com'`
        # only) and its own walk tests the MAC about those centres with the exact
        # particle radii about them (`resolve_walk_geometry`, default "com" in the
        # strict fused lane -- plan sub-10ms Phase 1.2). This exchange used
        # `upward.geometry` -- AABB centres, box radii -- so the sender accepted
        # pairs that are admissible about the box centre and DIVERGENT about the
        # COM the receiver's M2L actually expands from. Divergent pairs do not
        # improve with order: measured as a floor near 1.4e-03 the reference lane
        # passes through (record, Task 1), while routing the whole cross field
        # through direct sums removed it (6.0e-04 at p6). Same defect class as
        # memory `mac-geometry-inconsistent-with-com-centres`, one level up.
        # `JACCPOT_CROSS_MAC_GEOMETRY=aabb` keeps the old behaviour as a control.
        box_geom = upward.geometry
        if _cross_mac_geometry_mode() == "com":
            from jaccpot.runtime._mac_geometry import resolve_walk_geometry

            geom, _ = resolve_walk_geometry(
                tree,
                tree_artifacts.positions_sorted,
                box_geom,
                getattr(mp, "centers", None),
                leaf_cap=int(tree_artifacts.leaf_cap),
                default_mode="com",
            )
            if geom is None:
                geom = box_geom
        else:
            geom = box_geom

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
            float(theta if export_theta is None else export_theta),
            me,
            max_pair_queue=cap.walk_queue,
            far_cap=cap.export_far_cap,
            near_cap=cap.export_near_cap,
            mac_type=mac_type,
        )

        if record is not None:
            # Recompute the REAL MAC on the pairs the sender emitted, using the
            # sender's own arrays. This is the control that separates "export_walk
            # emits inadmissible pairs" from "the geometry changes in transit":
            # the receiver runs the identical check on the identical pairs below.
            from yggdrax._interactions_impl import _compute_mac_ok

            mc = int(cap.max_cells)
            fc = jnp.asarray(ex.far_cell)
            fn = jnp.asarray(ex.far_node)
            f_live = (jnp.arange(fc.shape[0]) < ex.far_count) & (fc >= 0)
            f_dev = jnp.where(f_live, fc // mc, 0)
            f_slot = jnp.where(f_live, fc % mc, 0)
            f_safe = jnp.where(f_live, fn, 0)
            d_s = all_cen[f_dev, f_slot] - jnp.asarray(geom.center)[f_safe]
            d2_s = jnp.sum(d_s * d_s, axis=-1)
            ok_s = _compute_mac_ok(
                mac_type=mac_type,
                theta_sq=jnp.asarray(float(theta) ** 2, d2_s.dtype),
                dist_sq=d2_s,
                extent_target=all_rad[f_dev, f_slot],
                extent_source=jnp.asarray(geom.radius)[f_safe],
                valid_pairs=f_live,
                different_nodes=jnp.ones_like(f_live),
            )
            # Walk centres vs expansion centres. Under the COM geometry these are
            # the SAME array and both numbers must read 0 -- that is the check the
            # geometry fix is in place. Under `aabb` they differed on 11116/16383
            # nodes by up to 180 units, which is why both centres travel.
            _gc = jnp.asarray(geom.center)
            _mc_ = jnp.asarray(mp.centers)
            record["center_mismatch"] = jnp.sum(jnp.any(_gc != _mc_, axis=-1))
            record["center_max_delta"] = jnp.max(jnp.abs(_gc - _mc_))
            record["export_far_live"] = jnp.sum(f_live)
            record["export_mac_fail"] = jnp.sum(f_live & ~ok_s)

            # CONTROL. The near list is what BOTTOMED OUT -- those pairs failed the
            # MAC, which is why they are near. If this recomputation is meaningful
            # it must reject most of them while accepting the far list. If both
            # come out the same the recomputation is wrong, not the walk.
            nc = jnp.asarray(ex.near_cell)
            nn = jnp.asarray(ex.near_node)
            n_live = (jnp.arange(nc.shape[0]) < ex.near_count) & (nc >= 0)
            n_dev = jnp.where(n_live, nc // mc, 0)
            n_slot = jnp.where(n_live, nc % mc, 0)
            n_safe = jnp.where(n_live, nn, 0)
            d_n = all_cen[n_dev, n_slot] - jnp.asarray(geom.center)[n_safe]
            d2_n = jnp.sum(d_n * d_n, axis=-1)
            ok_n = _compute_mac_ok(
                mac_type=mac_type,
                theta_sq=jnp.asarray(float(theta) ** 2, d2_n.dtype),
                dist_sq=d2_n,
                extent_target=all_rad[n_dev, n_slot],
                extent_source=jnp.asarray(geom.radius)[n_safe],
                valid_pairs=n_live,
                different_nodes=jnp.ones_like(n_live),
            )
            record["export_near_live"] = jnp.sum(n_live)
            record["export_near_mac_fail"] = jnp.sum(n_live & ~ok_n)

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
        # The RADIUS travels with the multipole. Without it the receiver has to
        # invent one, and the only available default -- zero -- makes every imported
        # node pass the MAC at any distance: multipoles get used at close range and
        # the near list is starved of exactly the pairs that carry the largest
        # forces. The symptom is a cross field that reaches the force and still
        # leaves it badly wrong.
        payload = jnp.concatenate(
            [
                jnp.where(alive, jnp.asarray(mp.packed)[rows], 0.0),
                jnp.where(alive, jnp.asarray(mp.centers)[rows], 0.0),
                # BOTH centres travel, because they are different points and the
                # two consumers need different ones. `mp.centers` is the expansion
                # centre and belongs to the M2L; `geom.center` is the geometric
                # centre the MAC was computed with. Shipping only the first made
                # the receiver re-test the MAC at a point the sender never used,
                # so pairs the sender had accepted came back inadmissible -- and
                # `combined_cen` was then geometric for local nodes and expansion
                # for imported ones, inside the one array the walk reads.
                jnp.where(alive, jnp.asarray(geom.center)[rows], 0.0),
                jnp.where(alive, jnp.asarray(geom.radius)[rows][:, None], 0.0),
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
        imp_cen = got.payload[:, n_coeff : n_coeff + 3]  # expansion -> M2L
        imp_gcen = got.payload[:, n_coeff + 3 : n_coeff + 6]  # geometric -> MAC
        imp_rad = got.payload[:, n_coeff + 6]

        combined_left = jnp.concatenate([left, jnp.full((cap.recv_node_cap,), -1, idx)])
        combined_right = jnp.concatenate(
            [right, jnp.full((cap.recv_node_cap,), -1, idx)]
        )
        # the WALK's geometry: geometric centres on both halves
        combined_cen = jnp.concatenate([jnp.asarray(geom.center), imp_gcen])
        combined_rad = jnp.concatenate([jnp.asarray(geom.radius), imp_rad])

        if record is not None:
            from yggdrax._interactions_impl import _compute_mac_ok as _mac_ok

            sc = jnp.asarray(got.csr_cell)
            sr = jnp.asarray(got.csr_row)
            s_live = (jnp.arange(sc.shape[0]) < got.num_csr) & (sc >= 0)
            sa = jnp.where(s_live, cells[jnp.where(s_live, sc, 0)], 0)
            sb_ = jnp.where(s_live, n_local + sr, 0)
            d_r = combined_cen[sa] - combined_cen[sb_]
            d2_r = jnp.sum(d_r * d_r, axis=-1)
            ok_r = _mac_ok(
                mac_type=mac_type,
                theta_sq=jnp.asarray(float(theta) ** 2, d2_r.dtype),
                dist_sq=d2_r,
                extent_target=combined_rad[sa],
                extent_source=combined_rad[sb_],
                valid_pairs=s_live,
                different_nodes=jnp.ones_like(s_live),
            )
            record["seed_live"] = jnp.sum(s_live)
            record["seed_mac_fail"] = jnp.sum(s_live & ~ok_r)

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

        # ---- the NEAR half: ship the exported leaves' PARTICLES -------------
        if near_sink is not None:
            W = int(cap.leaf_width)
            nr = jnp.asarray(tree.node_ranges)
            pos_sorted = jnp.asarray(tree.positions_sorted)
            mass_sorted = jnp.asarray(tree.masses_sorted)

            sb_n = build_send_buffers(
                ex.near_cell,
                ex.near_node,
                ex.near_count,
                ndev=ndev,
                max_cells=cap.max_cells,
                num_nodes=n_local,
                node_capacity=cap.send_node_cap,
                csr_capacity=cap.send_csr_cap,
            )
            # a leaf's particles are a contiguous run [start, end]; gather them into
            # a W-wide tile, which is the layout the leafpair kernel's pool wants
            lrow = jnp.clip(sb_n.node_rows, 0, n_local - 1)
            starts = nr[lrow, 0]
            ends = nr[lrow, 1]
            slot = jnp.arange(W, dtype=starts.dtype)[None, :]
            idx_p = jnp.clip(starts[:, None] + slot, 0, pos_sorted.shape[0] - 1)
            okrow = (sb_n.node_rows >= 0)[:, None]
            valid = okrow & (starts[:, None] + slot <= ends[:, None])
            tile_pos = jnp.where(valid[..., None], pos_sorted[idx_p], 0.0)
            tile_mass = jnp.where(valid, mass_sorted[idx_p], 0.0)
            # A leaf holding more than W particles would lose the excess here with
            # no other trace of it -- a MISSING force that every invariant passes.
            # W comes from a capacity, so this is a real possibility, not a
            # theoretical one, and it has to surface as an overflow.
            tile_truncated = ((sb_n.node_rows >= 0) & (ends - starts + 1 > W)).any()
            # The near import ships its OWN geometry. The far and near halves export
            # DIFFERENT node sets -- far sends internal nodes, near sends leaves --
            # so payload row k means a different node in each. Scoring the near walk
            # with the far import's centres, as this first did, pairs every imported
            # leaf's particles with an unrelated node's geometry: a wrong force that
            # looks entirely plausible.
            near_payload = jnp.concatenate(
                [
                    tile_pos.reshape(tile_pos.shape[0], -1),
                    tile_mass,
                    jnp.where(okrow, jnp.asarray(geom.center)[lrow], 0.0),
                    jnp.where(okrow, jnp.asarray(geom.radius)[lrow][:, None], 0.0),
                ],
                axis=1,
            )

            got_n = exchange_export_list(
                near_payload,
                sb_n.node_sizes,
                sb_n.csr_cell,
                sb_n.csr_row,
                sb_n.csr_sizes,
                payload_capacity=cap.recv_node_cap,
                csr_capacity=cap.recv_csr_cap,
                ndev=ndev,
                axis_name=axis_name,
            )
            imp_pos = got_n.payload[:, : 3 * W].reshape(-1, W, 3)
            imp_mass = got_n.payload[:, 3 * W : 4 * W]
            imp_cen_n = got_n.payload[:, 4 * W : 4 * W + 3]
            imp_rad_n = got_n.payload[:, 4 * W + 3]

            combined_cen_n = jnp.concatenate([jnp.asarray(geom.center), imp_cen_n])
            combined_rad_n = jnp.concatenate([jnp.asarray(geom.radius), imp_rad_n])

            rl_n = receiver_interaction_lists(
                combined_left,
                combined_right,
                combined_cen_n,
                combined_rad_n,
                n_local,
                cells,
                got_n.csr_cell,
                got_n.csr_row,
                got_n.num_csr,
                # The near import ships PARTICLES, so any pair this walk calls far
                # is unservable and gets dropped -- 89568/93005 of them, which is
                # the largest known hole left. Setting this to 0 makes every pair
                # bottom out as near, which costs direct sums but drops nothing.
                # It is an EXPERIMENT to size that hole, not the fix: the fix is to
                # ship multipoles for the near-exported leaves as well.
                float(theta if near_theta is None else near_theta),
                max_pair_queue=cap.walk_queue,
                far_cap=cap.recv_far_cap,
                near_cap=cap.recv_near_cap,
                mac_type=mac_type,
            )
            near_sink["positions"] = imp_pos
            near_sink["masses"] = imp_mass
            near_sink["mask"] = imp_mass != 0.0
            near_sink["target_node"] = rl_n.near_target
            near_sink["source_row"] = rl_n.near_source
            near_sink["count"] = rl_n.near_count
            if record is not None:
                record["near_recv_nodes"] = got_n.num_payload
                record["near_list_pairs"] = rl_n.near_count
                # pairs the NEAR walk classified as far are wasted: the near import
                # ships particles, not multipoles, so nothing can consume them
                record["near_walk_far_pairs"] = rl_n.far_count
            near_sink["overflow"] = (
                summary.overflow
                | sb_n.node_overflow
                | sb_n.csr_overflow
                | rl_n.near_overflow
                | rl_n.queue_overflow
                | tile_truncated
            )

        if record is not None:
            record["export_far"] = ex.far_count
            record["send_nodes"] = jnp.sum(sb.node_sizes)
            record["recv_nodes"] = got.num_payload
            record["recv_csr"] = got.num_csr
            record["far_pairs"] = rl.far_count
            record["near_pairs"] = rl.near_count
            # A zero imported radius is the failure this ships the radius to avoid,
            # and it is INVISIBLE in the force: it does not crash, it does not
            # overflow, it just moves close pairs into the far list. Count them.
            live_rows = jnp.arange(imp_rad.shape[0]) < got.num_payload
            record["imported_zero_radius"] = jnp.sum(
                jnp.where(live_rows, imp_rad <= 0.0, False)
            )
            record["imported_rows"] = got.num_payload
            record["summary_cells"] = summary.num_cells
            record["summary_leaves"] = jnp.sum(summary.leaves_per_cell)
            record["overflow"] = (
                # `summary.overflow` was missing from this list, and it is the one
                # flag whose absence is silent: a truncated cut drops part of the
                # RECEIVER from the exchange, so those particles get no cross field
                # at all. It loses force, not accuracy, so no invariant moves.
                summary.overflow
                | ex.far_overflow
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


def cross_near_acceleration(
    near: dict,
    local_leaf_positions: Array,
    local_leaf_masses: Array,
    local_leaf_mask: Array,
    leaf_particle_indices: Array,
    n_particles: int,
    num_internal: int,
    *,
    softening_sq: Array,
    G: Array,
    chunk: int = 64,
    interpret: bool = False,
    accum: Optional[str] = None,
) -> Array:
    """Acceleration on local particles from the IMPORTED leaves, as its own term.

    Added to the force rather than interleaved, because a near contribution is a
    direct sum with no cascade behind it -- see the module docstring. Correct by
    linearity of gravity in the sources.

    The pool is ``[local leaves ; imported leaves]`` and ``num_target_leaves`` is the
    local count, so the kernel emits local rows only (Phase 4.3). The CSR carries the
    cross pairs alone: the local near field is already in the force this is added to,
    and including it here would double it.

    Parameters
    ----------
    near:
        The ``near_sink`` a :func:`make_cross_hook` filled.
    local_leaf_positions, local_leaf_masses, local_leaf_mask:
        This device's leaf-major pool, ``(L, W, 3)`` and ``(L, W)``.
    leaf_particle_indices:
        ``(L, W)`` particle index per padded leaf slot, for scattering the result
        back into particle order.
    n_particles:
        Rows in the output.
    num_internal:
        Internal-node count of the local tree. A target arrives as a NODE id and the
        pool is indexed by LEAF ROW, and the two differ by exactly this -- deriving
        it from the pool's own length instead only works for a balanced structure,
        which is not something to assume here.
    softening_sq, G:
        Scalars, matching the local lane's.
    chunk:
        Entries per chunk for the leafpair table. Static.
    interpret:
        Pallas interpret mode.
    accum:
        Accumulator width of the leafpair kernel, ``"input"`` or ``"wide"`` (a
        float32 partial per source leaf, float64 across leaves -- the two-level
        accumulator of the local lane's near field). ``None`` reads the SAME
        env choice the local lane reads, ``JACCPOT_NEARFIELD_ACCUM``, so the two
        near terms widen together. Before this knob existed the cross term was
        pinned to ``"input"`` whatever the local lane did, which made it the one
        fp32-only reduction in an otherwise widened force.

    Returns
    -------
    Array
        ``(n_particles, 3)`` acceleration to ADD to the local force.
    """
    from jaccpot.nearfield.near_field import _env_choice
    from jaccpot.pallas.nearfield_leafpair_csr import (
        build_leafpair_chunk_table,
        leafpair_chunk_capacity,
        nearfield_leafpair_csr_pallas,
    )

    if accum is None:
        accum = _env_choice("JACCPOT_NEARFIELD_ACCUM", "input", ("input", "wide"))

    loc_pos = jnp.asarray(local_leaf_positions)
    loc_mass = jnp.asarray(local_leaf_masses)
    loc_mask = jnp.asarray(local_leaf_mask)
    idx = jnp.asarray(leaf_particle_indices)
    imp_pos = jnp.asarray(near["positions"])
    imp_mass = jnp.asarray(near["masses"])
    imp_mask = jnp.asarray(near["mask"])

    L = int(loc_pos.shape[0])

    # The imported tiles are W wide because a CAPACITY said so; the local pool is
    # as wide as the tree made it. Concatenating requires they agree, and the two
    # are set in different places, so pad the narrower up rather than trusting a
    # caller to keep them in step. Padding is mask-false, so it changes nothing.
    W = max(int(loc_pos.shape[1]), int(imp_pos.shape[1]))

    def _widen(pos, mass, mask, extra_idx=None):
        w = int(pos.shape[1])
        if w == W:
            return pos, mass, mask, extra_idx
        pad = W - w
        pos = jnp.pad(pos, ((0, 0), (0, pad), (0, 0)))
        mass = jnp.pad(mass, ((0, 0), (0, pad)))
        mask = jnp.pad(mask, ((0, 0), (0, pad)))
        if extra_idx is not None:
            extra_idx = jnp.pad(extra_idx, ((0, 0), (0, pad)))
        return pos, mass, mask, extra_idx

    loc_pos, loc_mass, loc_mask, idx = _widen(loc_pos, loc_mass, loc_mask, idx)
    imp_pos, imp_mass, imp_mask, _ = _widen(imp_pos, imp_mass, imp_mask)

    pool_pos = jnp.concatenate([loc_pos, imp_pos])
    pool_mass = jnp.concatenate([loc_mass, imp_mass])
    pool_mask = jnp.concatenate([loc_mask, imp_mask])

    # the cross pairs, as a CSR over LOCAL target leaf rows
    tgt_node = jnp.asarray(near["target_node"])
    src_row = jnp.asarray(near["source_row"])
    n_pair = jnp.asarray(near["count"])
    P = int(tgt_node.shape[0])
    # A target arrives as a leaf NODE id; the pool is indexed by leaf ROW. The
    # upper bound is part of `live` rather than a clip: clipping an out-of-range
    # target would silently ADD its sources to row 0 or L-1, which is a wrong force,
    # whereas dropping it is a missing one. Neither is acceptable, but only the
    # second is detectable downstream, and the test asserts nothing is dropped.
    lrow = tgt_node - int(num_internal)
    live = (jnp.arange(P) < n_pair) & (lrow >= 0) & (lrow < L)
    row = jnp.where(live, lrow, L)
    order = jnp.argsort(jnp.where(live, row, L), stable=True)
    row_s = row[order]
    src_s = jnp.where(live[order], L + src_row[order], -1)
    counts = jnp.bincount(jnp.where(live[order], row_s, L), length=L + 1)[:L].astype(
        jnp.int32
    )
    offsets = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)]
    )
    cap_chunks = leafpair_chunk_capacity(P, L, int(chunk))
    table = build_leafpair_chunk_table(
        offsets, counts, chunk=int(chunk), capacity=cap_chunks
    )

    out = nearfield_leafpair_csr_pallas(
        pool_pos,
        pool_mass,
        pool_mask,
        src_s.astype(jnp.int32),
        table,
        softening_sq=softening_sq,
        G=G,
        chunk=int(chunk),
        interpret=bool(interpret),
        include_self=False,  # the local self term is already in the force
        num_target_leaves=L,
        accum=str(accum),
    )
    acc = out[..., :3]
    # The scatter below is a PERMUTATION, not a reduction: every live particle sits
    # in exactly one leaf slot, so each output row receives one addend and the
    # dead slots all land on the discarded row n_particles. Nothing is summed in
    # the input dtype here; the only reductions are inside the kernel, under
    # `accum`.
    flat_idx = jnp.where(loc_mask, idx, n_particles).reshape(-1)
    return (
        jnp.zeros((n_particles + 1, 3), acc.dtype)
        .at[flat_idx]
        .add(acc.reshape(-1, 3))[:n_particles]
    )
