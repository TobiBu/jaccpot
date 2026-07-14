"""Reusable multi-GPU distributed FMM force driver.

This is a faithful extraction of the per-device ``shard_map`` body validated in
``tests/test_distributed_solidfmm_far_shardmap.py``.  The pipeline, per device,
inside one ``shard_map``:

    global bounds + sanitize padding
    local radix tree  (yggdrax ``Tree.from_particles``)
    solidfmm upward sweep (P2M / M2M)         -> per-node complex multipoles
    self dual-tree walk (M2L list + near CSR)
    remote coarse tree  (frontier of leaf COMs, own domain excluded)
      all_gather local multipoles -> seed coarse leaves with the REAL order-p
      multipole of the remote leaf they stand for -> M2M up the coarse tree
    cross-walk local vs coarse tree at ``theta_cross`` -> far (M2L) + near
    two-round ragged halo import of the near remote particles
    far = self M2L (local list) + remote M2L (cross list, separate src/tgt
          centres) -> level-by-level L2L cascade -> L2P            (far = -G*grad)
    near = ONE combined P2P over ``[local ; halo]`` (explicit override CSR path)
    total accel = far + near

Correctness notes carried over from the validated test (do not "simplify"):
  * the near P2P MUST be driven through ``leaf_particle_indices_override`` + the
    ``neighbor_*_override`` CSR -- jaccpot's default node_ranges-gather P2P is
    ~100x wrong under ``shard_map``;
  * the ``shard_map`` mesh axis MUST be named ``"gpus"`` (every internal
    ``all_gather`` / ragged-all-to-all in yggdrax.distributed hard-codes it);
  * the driver keeps only the *full* (self + remote) force path -- the test's
    separate ``_self`` outputs existed only to isolate local-vs-remote error.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

from yggdrax import build_interactions_and_neighbors, compute_tree_geometry
from yggdrax.distributed import device_count, make_mesh
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl
from yggdrax.distributed.let import (
    build_coarse_frontier,
    build_remote_coarse_tree,
    import_near_halo,
)
from yggdrax.distributed.local_tree import sanitize_padding
from yggdrax.distributed.partition import global_bounds
from yggdrax.dtypes import INDEX_DTYPE, complex_dtype_for_real
from yggdrax.interactions import DualTreeTraversalConfig
from yggdrax.morton import morton_encode_impl
from yggdrax.tree import (
    Tree,
    get_level_offsets,
    get_node_levels,
    get_nodes_by_level,
)
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.downward.local_expansions import LocalExpansionData
from jaccpot.nearfield.near_field import compute_leaf_p2p_accelerations
from jaccpot.operators.complex_ops import (
    enforce_conjugate_symmetry_batch,
    m2l_complex_reference_batch,
)
from jaccpot.operators.real_harmonics import complex_to_dehnen_real_coeffs, sh_size
from jaccpot.runtime._fmm_impl import (
    _accumulate_real_m2l_fullbatch,
    _accumulate_solidfmm_m2l_fullbatch,
    _apply_real_m2l,
    _evaluate_local_expansions_for_particles,
    _propagate_solidfmm_locals_by_level,
)
from jaccpot.upward.solidfmm_complex_tree_expansions import (
    _aggregate_m2m_complex_by_level,
    prepare_solidfmm_complex_upward_sweep,
)

# Order of the per-device diagnostic vector returned alongside the forces.
DIAG_FIELDS = (
    "cross_far_pairs",
    "cross_near_pairs",
    "cross_queue_overflow",
    "cross_far_overflow",
    "cross_near_overflow",
    "self_far_pairs",
    "self_near_pairs",
    "self_queue_overflow",
    "self_far_overflow",
    "self_near_overflow",
)


@dataclass(frozen=True)
class DistributedFMMConfig:
    """Static knobs for the distributed FMM force evaluation.

    ``order`` (multipole order ``p``), ``theta`` (local self MAC), ``theta_cross``
    (cross-domain far-field MAC), ``leaf_size``, ``softening`` and ``G`` are the
    physics/accuracy knobs.  The ``*_cap`` fields set fixed traversal buffer
    shapes and therefore must be large enough to hold the interaction/neighbour
    lists without truncation -- grow them (see the capacity calibrator in the
    benchmark harness) for large N or strongly clustered distributions.  An
    overflow shows up in the returned diagnostics as a nonzero
    ``*_overflow`` flag.
    """

    order: int = 3
    theta: float = 0.4
    theta_cross: float = 0.1
    leaf_size: int = 8
    softening: float = 0.02
    G: float = 1.0
    rotation: str = "solidfmm"
    mac_type: str = "bh"
    # Far-field expansion basis: "solidfmm" (complex) or "real" (Dehnen no-sqrt2).
    # "real" converges the per-device far field onto the single-GPU fast-lane path
    # (memory-lighter, and unlocks the fused real M2L Pallas kernel when
    # JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS is set). Upward sweep + coarse M2M stay
    # complex; multipoles are converted to Dehnen real coeffs at the M2L boundary.
    basis: str = "solidfmm"
    # self dual-tree walk capacities
    max_interactions_per_node: int = 512
    max_neighbors_per_leaf: int = 128
    max_pair_queue: int = 1 << 15
    process_block: int = 64
    # cross-walk (local vs remote coarse tree) capacities
    cross_max_interactions_per_node: int = 512
    cross_max_neighbors_per_leaf: int = 128
    cross_max_pair_queue: int = 1 << 15

    def with_scaled_caps(self, factor: float) -> "DistributedFMMConfig":
        """Return a copy with every capacity multiplied by ``factor`` (rounded up).

        Handy for the overflow-retry loop in capacity calibration.
        """

        def g(v: int) -> int:
            return int(np.ceil(v * factor))

        return dataclasses.replace(
            self,
            max_interactions_per_node=g(self.max_interactions_per_node),
            max_neighbors_per_leaf=g(self.max_neighbors_per_leaf),
            max_pair_queue=g(self.max_pair_queue),
            cross_max_interactions_per_node=g(self.cross_max_interactions_per_node),
            cross_max_neighbors_per_leaf=g(self.cross_max_neighbors_per_leaf),
            cross_max_pair_queue=g(self.cross_max_pair_queue),
        )


@dataclass
class DistributedFMMResult:
    """Result of :func:`distributed_fmm_accelerations`.

    ``accelerations`` are in the *original* input particle order.  ``diagnostics``
    is a dict of per-device diagnostic arrays (see :data:`DIAG_FIELDS`), plus the
    reduced boolean ``overflow`` (True if any device truncated any list).
    """

    accelerations: np.ndarray
    diagnostics: dict[str, Any]
    ndev: int
    cap: int
    config: DistributedFMMConfig

    @property
    def overflow(self) -> bool:
        return bool(self.diagnostics.get("overflow", False))


def partition_for_devices(
    positions: np.ndarray,
    masses: np.ndarray,
    ndev: int,
    *,
    leaf_size: int,
    bounds: tuple | None = None,
) -> dict:
    """Morton-sort particles and split into ``ndev`` contiguous SFC domains.

    Each domain is padded up to a common per-device capacity ``cap`` (a multiple
    of ``leaf_size``); padding rows carry mass 0, are placed at a real particle
    position (so geometry is not inflated), and get global id ``-1``.  A global
    id array is threaded through so distributed forces can be scattered back to
    the input order.

    Returns a dict with the flat ``(ndev*cap, ...)`` arrays ready for
    ``shard_map`` plus ``cap``, ``counts`` and ``bounds``.
    """

    positions = np.asarray(positions)
    masses = np.asarray(masses)
    n = positions.shape[0]
    if n < ndev:
        raise ValueError(f"need at least ndev={ndev} particles, got {n}")

    base, rem = divmod(n, ndev)
    counts = np.full(ndev, base, np.int64)
    counts[:rem] += 1
    cap = int(np.ceil(counts.max() / leaf_size) * leaf_size)

    if bounds is None:
        lo, hi = positions.min(0), positions.max(0)
        span = np.where(hi > lo, hi - lo, 1.0)
        bounds = (jnp.asarray(lo - span * 1e-6), jnp.asarray(hi + span * 1e-6))

    codes = np.asarray(morton_encode_impl(jnp.asarray(positions), bounds))
    order = np.argsort(codes)

    pos_g = np.zeros((ndev, cap, 3), positions.dtype)
    mass_g = np.zeros((ndev, cap), masses.dtype)
    gid_g = np.full((ndev, cap), -1, np.int64)
    start = 0
    for d in range(ndev):
        c = int(counts[d])
        chunk = order[start : start + c]
        pos_g[d, :c] = positions[chunk]
        mass_g[d, :c] = masses[chunk]
        gid_g[d, :c] = chunk
        pos_g[d, c:] = positions[chunk[0]]  # pad at a real point
        start += c

    return {
        "pos_flat": pos_g.reshape(ndev * cap, 3),
        "mass_flat": mass_g.reshape(ndev * cap),
        "gid_flat": gid_g.reshape(ndev * cap),
        "counts": counts,
        "cap": cap,
        "bounds": bounds,
        "n": n,
        "ndev": ndev,
    }


def _make_fn(config: DistributedFMMConfig, ndev: int, cap: int) -> Callable:
    """Build the per-device ``shard_map`` body closing over static shapes.

    This is a faithful copy of ``fn`` / ``_combined_neighbors`` from
    ``tests/test_distributed_solidfmm_far_shardmap.py`` with the module-level
    constants promoted to ``config`` fields and only the *full* force path kept.
    """

    p = config.order
    leaf = config.leaf_size
    G = config.G
    soft = config.softening
    rot = config.rotation
    mac = config.mac_type
    theta = config.theta
    theta_cross = config.theta_cross
    is_real = str(config.basis).strip().lower() == "real"

    C = sh_size(p)
    if cap % leaf != 0:
        raise ValueError(f"cap={cap} must be a multiple of leaf_size={leaf}")
    Lloc = cap // leaf
    max_req = (ndev - 1) * Lloc
    max_recv = (ndev - 1) * Lloc

    cfg = DualTreeTraversalConfig(
        max_interactions_per_node=config.max_interactions_per_node,
        max_neighbors_per_leaf=config.max_neighbors_per_leaf,
        max_pair_queue=config.max_pair_queue,
        process_block=config.process_block,
    )
    KC = config.cross_max_interactions_per_node
    KN = config.cross_max_neighbors_per_leaf
    x_queue = config.cross_max_pair_queue

    def _combined_neighbors(tree, nbr, cross, rct, halo, n_halo_rows, local_only=False):
        """Vectorised unified P2P neighbour CSR over [local leaves ; halo blocks]."""
        leaf_nodes = jnp.asarray(nbr.leaf_indices, INDEX_DTYPE)
        n_lloc = leaf_nodes.shape[0]
        total_nodes = jnp.asarray(tree.node_ranges).shape[0]
        node_to_row = (
            jnp.full((total_nodes + 1,), -1, INDEX_DTYPE)
            .at[leaf_nodes]
            .set(jnp.arange(n_lloc, dtype=INDEX_DTYPE))
        )
        u_leaves = n_lloc + n_halo_rows

        lc = jnp.asarray(nbr.counts, INDEX_DTYPE)
        l_src_node = jnp.asarray(nbr.neighbors, INDEX_DTYPE)
        Ne_loc = l_src_node.shape[0]
        l_cum = jnp.cumsum(lc)
        e_loc = jnp.arange(Ne_loc, dtype=INDEX_DTYPE)
        l_tgt = jnp.searchsorted(l_cum, e_loc, side="right")
        l_valid = e_loc < l_cum[-1]
        l_src_row = node_to_row[jnp.clip(l_src_node, 0, total_nodes)]
        l_valid = l_valid & (l_src_row >= 0)

        cc = jnp.asarray(cross.neighbor_counts, INDEX_DTYPE)
        r_src_cnode = jnp.asarray(cross.neighbor_indices, INDEX_DTYPE)
        Ne_rem = r_src_cnode.shape[0]
        c_cum = jnp.cumsum(cc)
        e_rem = jnp.arange(Ne_rem, dtype=INDEX_DTYPE)
        c_row = jnp.searchsorted(c_cum, e_rem, side="right")
        r_valid = e_rem < c_cum[-1]
        cross_leaf = jnp.asarray(cross.leaf_indices, INDEX_DTYPE)
        c_tgt_node = cross_leaf[jnp.clip(c_row, 0, cross_leaf.shape[0] - 1)]
        r_tgt = node_to_row[jnp.clip(c_tgt_node, 0, total_nodes)]
        coarse_ranges = jnp.asarray(rct.tree.node_ranges, INDEX_DTYPE)
        coarse_pos = coarse_ranges[
            jnp.clip(r_src_cnode, 0, coarse_ranges.shape[0] - 1), 0
        ]
        block = halo.coarse_to_halo[
            jnp.clip(coarse_pos, 0, halo.coarse_to_halo.shape[0] - 1)
        ]
        r_src_row = n_lloc + block
        r_valid = r_valid & (r_tgt >= 0) & (block >= 0)

        if local_only:
            tgt, src, valid = l_tgt, l_src_row, l_valid
        else:
            tgt = jnp.concatenate([l_tgt, r_tgt])
            src = jnp.concatenate([l_src_row, r_src_row])
            valid = jnp.concatenate([l_valid, r_valid])
        sentinel = jnp.asarray(u_leaves, INDEX_DTYPE)
        tgt = jnp.where(valid, tgt, sentinel)
        src = jnp.where(valid, src, sentinel)

        srt = jnp.argsort(tgt)
        src_s = src[srt]
        counts = jnp.bincount(tgt, length=u_leaves).astype(INDEX_DTYPE)
        offsets = jnp.concatenate([jnp.zeros((1,), INDEX_DTYPE), jnp.cumsum(counts)])
        return offsets, src_s, counts, u_leaves

    def fn(pos, mass, gid, count):
        bounds = global_bounds(pos)
        pos_s, mass_s = sanitize_padding(pos, mass, count)
        tree = Tree.from_particles(
            pos_s,
            mass_s,
            tree_type="radix",
            bounds=bounds,
            return_reordered=True,
            leaf_size=leaf,
        )
        lp = tree.positions_sorted
        lm = tree.masses_sorted
        gid_sorted = gid[jnp.asarray(tree.particle_indices, INDEX_DTYPE)]
        geom = compute_tree_geometry(tree, lp, max_leaf_size=leaf)
        total_nodes = jnp.asarray(tree.node_ranges).shape[0]
        cdtype = complex_dtype_for_real(lp.dtype)

        # local solidfmm upward
        up = prepare_solidfmm_complex_upward_sweep(
            tree, lp, lm, max_order=p, max_leaf_size=leaf, rotation=rot
        )
        centers = up.multipoles.centers
        packed = up.multipoles.packed
        # Far-field coefficient dtype + source multipoles. For the real basis the
        # upward sweep stays complex (validated), and we convert to Dehnen no-sqrt2
        # real coeffs at the M2L boundary; the L2L/L2P then auto-select the real
        # path by dtype. solidfmm keeps the complex packed multipoles unchanged.
        coeff_dtype = lp.dtype if is_real else cdtype
        packed_use = (
            complex_to_dehnen_real_coeffs(packed, order=p) if is_real else packed
        )

        inter, nbr, self_res = build_interactions_and_neighbors(
            tree,
            geom,
            theta=theta,
            traversal_config=cfg,
            mac_type=mac,
            return_result=True,
        )

        # remote coarse tree over frontier (leaf COM + mass)
        mm = compute_tree_mass_moments(tree, lp, lm)
        fr = build_coarse_frontier(tree, mm.mass, mm.center_of_mass)
        rct = build_remote_coarse_tree(fr, ndev, bounds=bounds)

        # coarse solidfmm centres (COM) + level structure
        upc = prepare_solidfmm_complex_upward_sweep(
            rct.tree,
            rct.positions_sorted,
            rct.masses_sorted,
            max_order=p,
            max_leaf_size=1,
            rotation=rot,
        )
        c_centers = upc.multipoles.centers
        c_lc = jnp.asarray(rct.tree.left_child, INDEX_DTYPE)
        c_rc = jnp.asarray(rct.tree.right_child, INDEX_DTYPE)
        c_total = c_centers.shape[0]
        c_nint = int(c_lc.shape[0])
        c_nr = jnp.asarray(rct.tree.node_ranges, INDEX_DTYPE)
        c_leaves = jnp.arange(c_nint, c_total, dtype=INDEX_DTYPE)

        # seed coarse leaves with remote leaves' REAL solidfmm multipoles
        gpacked = jax.lax.all_gather(packed, "gpus", tiled=False)  # [ndev,N,C]
        spos = c_nr[c_leaves, 0]
        dom = rct.tag_domain[spos]
        nod = rct.tag_node_id[spos]
        okm = nod >= 0
        leafp = gpacked[jnp.where(okm, dom, 0), jnp.where(okm, nod, 0)]
        leafp = jnp.where(okm[:, None], leafp, 0.0)
        seed = (
            jnp.zeros((c_total, C), dtype=cdtype).at[c_leaves].set(leafp.astype(cdtype))
        )
        c_nbl = get_nodes_by_level(rct.tree)
        c_loff = get_level_offsets(rct.tree)
        c_numlev = int(c_loff.shape[0] - 1)
        coarse_packed = _aggregate_m2m_complex_by_level(
            seed,
            c_centers,
            c_lc,
            c_rc,
            jnp.asarray(c_nbl, INDEX_DTYPE),
            jnp.asarray(c_loff, INDEX_DTYPE),
            order=p,
            num_internal=c_nint,
            num_levels=c_numlev,
            level_batch_width=max(c_nint, 1),
            rotation=rot,
        )
        # coarse M2M stays complex (validated); convert to real at the M2L boundary.
        coarse_packed_use = (
            complex_to_dehnen_real_coeffs(coarse_packed, order=p)
            if is_real
            else coarse_packed
        )

        cross = dual_tree_walk_cross_impl(
            tree,
            geom,
            rct.tree,
            rct.geometry,
            theta_cross,
            mac_type=mac,
            max_interactions_per_node=KC,
            max_neighbors_per_leaf=KN,
            max_pair_queue=x_queue,
        )
        halo = import_near_halo(
            rct,
            cross,
            lp,
            lm,
            ndev,
            leaf_size=leaf,
            max_req_leaves=max_req,
            max_recv_leaves=max_recv,
        )

        leaf_nodes = jnp.asarray(nbr.leaf_indices, INDEX_DTYPE)
        nr = jnp.asarray(tree.node_ranges, INDEX_DTYPE)
        node_levels = get_node_levels(tree)
        lc_full = jnp.asarray(tree.left_child, INDEX_DTYPE)
        rc_full = jnp.asarray(tree.right_child, INDEX_DTYPE)

        def _l2p(loc_coeffs):
            # Real coeffs carry no conjugate symmetry -> skip the complex-only fixup.
            if not is_real:
                loc_coeffs = enforce_conjugate_symmetry_batch(loc_coeffs, order=p)
            loc_coeffs = _propagate_solidfmm_locals_by_level(
                loc_coeffs,
                centers,
                lc_full,
                rc_full,
                node_levels,
                order=p,
                rotation=rot,
                total_nodes=int(total_nodes),
                basis_mode="real" if is_real else "complex",
            )
            ld = LocalExpansionData(order=p, centers=centers, coefficients=loc_coeffs)
            # L2P auto-selects the real path from the (real-typed) coefficients;
            # expansion_basis stays "solidfmm" for both.
            g = _evaluate_local_expansions_for_particles(
                ld,
                lp,
                leaf_nodes=leaf_nodes,
                node_ranges=nr,
                max_leaf_size=leaf,
                order=p,
                expansion_basis="solidfmm",
                return_potential=False,
            )[0]
            return -G * g

        # far: self M2L (local list) + remote M2L (cross list). Both bases share the
        # tgt-minus-src delta convention; the real path routes through the same
        # Pallas-aware _apply_real_m2l kernel used by the single-GPU fast lane.
        zeros = jnp.zeros((int(total_nodes), C), dtype=coeff_dtype)
        s_src = jnp.asarray(inter.sources, INDEX_DTYPE)
        s_tgt = jnp.asarray(inter.targets, INDEX_DTYPE)
        s_active = jnp.sum((s_tgt >= 0).astype(INDEX_DTYPE))
        if is_real:
            loc_self = _accumulate_real_m2l_fullbatch(
                zeros,
                packed_use,
                centers,
                s_src,
                s_tgt,
                s_active,
                order=p,
                m2l_impl="rot_scale",
                total_nodes=int(total_nodes),
            )
        else:
            loc_self = _accumulate_solidfmm_m2l_fullbatch(
                zeros,
                packed_use,
                centers,
                s_src,
                s_tgt,
                s_active,
                order=p,
                rotation=rot,
                total_nodes=int(total_nodes),
            )

        # cross M2L: separate source (coarse) / target (local) centres
        x_src = jnp.asarray(cross.interaction_sources, INDEX_DTYPE)
        x_tgt = jnp.asarray(cross.interaction_targets, INDEX_DTYPE)
        x_valid = x_tgt >= 0
        xs = jnp.where(x_valid, x_src, 0)
        xt = jnp.where(x_valid, x_tgt, 0)
        x_deltas = centers[xt] - c_centers[xs]
        if is_real:
            x_contribs = _apply_real_m2l(
                coarse_packed_use[xs], x_deltas, order=p, m2l_impl="rot_scale"
            ).astype(coeff_dtype)
        else:
            x_contribs = m2l_complex_reference_batch(
                coarse_packed_use[xs], x_deltas, order=p, rotation=rot
            ).astype(cdtype)
        x_contribs = jnp.where(x_valid[:, None], x_contribs, 0)
        loc_full = loc_self + jax.ops.segment_sum(x_contribs, xt, int(total_nodes))
        far_full = _l2p(loc_full)

        # near: leaf particle-index groups into [local ; halo] buffer
        lranges = nr[leaf_nodes]
        kk = jnp.arange(leaf, dtype=INDEX_DTYPE)
        loc_idx = jnp.clip(lranges[:, 0][:, None] + kk[None, :], 0, cap - 1)
        loc_mask = kk[None, :] < (lranges[:, 1] - lranges[:, 0] + 1)[:, None]
        halo_idx = (cap + jnp.arange(max_req * leaf, dtype=INDEX_DTYPE)).reshape(
            max_req, leaf
        )
        halo_mask = halo.valid.reshape(max_req, leaf)
        concat_pos = jnp.concatenate([lp, halo.positions], axis=0)
        concat_mass = jnp.concatenate([lm, halo.masses], axis=0)

        offsets, src_s, counts, u_leaves = _combined_neighbors(
            tree, nbr, cross, rct, halo, max_req
        )
        lp_idx = jnp.concatenate([loc_idx, halo_idx], axis=0)
        lp_mask = jnp.concatenate([loc_mask, halo_mask], axis=0)
        near_full = compute_leaf_p2p_accelerations(
            tree,
            nbr,
            concat_pos,
            concat_mass,
            G=G,
            softening=soft,
            nearfield_mode="baseline",
            node_ranges_override=jnp.zeros((u_leaves + 1, 2), INDEX_DTYPE),
            leaf_nodes_override=jnp.arange(u_leaves, dtype=INDEX_DTYPE),
            neighbor_offsets_override=offsets,
            neighbor_indices_override=src_s,
            neighbor_counts_override=counts,
            leaf_particle_indices_override=lp_idx,
            leaf_particle_mask_override=lp_mask,
        )

        # far_full is (cap,3) already; near_full is (cap + halo, 3) -> keep the
        # local rows.  (The validated test slices BOTH to [:cap] on return.)
        accel = far_full[:cap] + near_full[:cap]

        diag = jnp.array(
            [
                cross.far_pair_count.astype(jnp.float64),
                cross.near_pair_count.astype(jnp.float64),
                cross.queue_overflow.astype(jnp.float64),
                cross.far_overflow.astype(jnp.float64),
                cross.near_overflow.astype(jnp.float64),
                self_res.far_pair_count.astype(jnp.float64),
                self_res.near_pair_count.astype(jnp.float64),
                self_res.queue_overflow.astype(jnp.float64),
                self_res.far_overflow.astype(jnp.float64),
                self_res.near_overflow.astype(jnp.float64),
            ]
        )
        return accel, gid_sorted[:cap, None].astype(jnp.int64), diag[None, :]

    return fn


def make_force_evaluator(
    config: DistributedFMMConfig,
    ndev: int,
    cap: int,
    mesh,
    *,
    jit: bool = True,
) -> Callable:
    """Build a callable ``(pos_flat, mass_flat, gid_flat, counts) -> (accel, gid, diag)``.

    The returned callable runs the per-device pipeline under ``shard_map`` over
    ``mesh`` (axis ``"gpus"``).  Wrapped in ``jax.jit`` by default so that, after
    a warmup call, repeated invocations measure steady-state device time
    (the natural per-force-evaluation metric).  Pass ``jit=False`` to run the
    eager ``shard_map`` (used by the correctness path, matching the test).
    """

    fn = _make_fn(config, ndev, cap)

    def evaluate(pos_flat, mass_flat, gid_flat, counts):
        return shard_map(
            fn,
            mesh=mesh,
            in_specs=(P("gpus"), P("gpus"), P("gpus"), P("gpus")),
            out_specs=(P("gpus"), P("gpus"), P("gpus")),
            check_vma=False,
        )(pos_flat, mass_flat, gid_flat, counts)

    return jax.jit(evaluate) if jit else evaluate


def distributed_fmm_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    config: DistributedFMMConfig | None = None,
    mesh=None,
    ndev: int | None = None,
    jit: bool = False,
) -> DistributedFMMResult:
    """Evaluate distributed FMM accelerations for all particles.

    Handles the host-side SFC decomposition + padding + global-id tracking,
    runs the ``shard_map`` force pipeline, and scatters the per-device forces
    back into the original input order.

    Returns a :class:`DistributedFMMResult`; ``.accelerations`` has shape
    ``(N, 3)`` in input order, ``.diagnostics`` includes the per-device pair
    counts / overflow flags and a reduced ``overflow`` bool.
    """

    if config is None:
        config = DistributedFMMConfig()
    if mesh is None:
        if ndev is None:
            ndev = device_count()
        mesh = make_mesh(ndev)
    else:
        ndev = int(np.prod(list(mesh.shape.values())))

    part = partition_for_devices(
        positions, masses, ndev, leaf_size=config.leaf_size
    )
    cap = part["cap"]
    counts_dev = jnp.asarray(part["counts"], INDEX_DTYPE)

    evaluate = make_force_evaluator(config, ndev, cap, mesh, jit=jit)
    accel_o, gid_o, diag_o = evaluate(
        jnp.asarray(part["pos_flat"]),
        jnp.asarray(part["mass_flat"]),
        jnp.asarray(part["gid_flat"]),
        counts_dev,
    )
    accel_o = np.asarray(accel_o)
    gid_o = np.asarray(gid_o).reshape(-1).astype(np.int64)
    diag_o = np.asarray(diag_o)

    n = part["n"]
    accel = np.zeros((n, 3), np.float64)
    seen = np.zeros(n, bool)
    for row in range(accel_o.shape[0]):
        g = gid_o[row]
        if g >= 0:
            accel[g] = accel_o[row]
            seen[g] = True
    if not seen.all():
        raise RuntimeError(
            f"{int((~seen).sum())} particles missing from the distributed result "
            "(padding/capacity bug)"
        )

    diag = {name: diag_o[:, i] for i, name in enumerate(DIAG_FIELDS)}
    overflow = bool(
        np.any(
            diag_o[
                :,
                [
                    DIAG_FIELDS.index(k)
                    for k in (
                        "cross_queue_overflow",
                        "cross_far_overflow",
                        "cross_near_overflow",
                        "self_queue_overflow",
                        "self_far_overflow",
                        "self_near_overflow",
                    )
                ],
            ]
            > 0
        )
    )
    diag["overflow"] = overflow

    return DistributedFMMResult(
        accelerations=accel,
        diagnostics=diag,
        ndev=ndev,
        cap=cap,
        config=config,
    )
