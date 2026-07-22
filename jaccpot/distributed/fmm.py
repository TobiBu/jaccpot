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
from typing import Any, Callable, NamedTuple, Optional

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
from jaccpot.nearfield.near_field import (
    _compute_leaf_p2p_prepared_large_n_self_only_impl,
    _prepare_leaf_data_from_groups,
    _radix_fast_lane_prepacked_pallas,
    _radix_fast_lane_prepacked_pallas_decoupled,
    compute_leaf_p2p_accelerations,
)
from jaccpot.operators.complex_ops import (
    enforce_conjugate_symmetry_batch,
    m2l_complex_reference_batch,
)
from jaccpot.operators.real_harmonics import sh_size
from jaccpot.runtime._interaction_cache import (
    _build_treecode_artifacts_strict_streamed,
)
from jaccpot.runtime.kernels.core import (
    _accumulate_m2l_fullbatch,
    _apply_real_m2l,
    _evaluate_local_expansions_for_particles,
    _propagate_solidfmm_locals_by_level,
)
from jaccpot.upward.real_tree_expansions import (
    aggregate_m2m_real_by_level,
    prepare_real_upward_sweep,
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


class _TreecodeWalkDiag(NamedTuple):
    """Minimal self-walk diagnostic shim for the treecode local walk.

    The treecode walk emits no ``DualTreeWalkResult``; it auto-sizes per-leaf caps and
    exposes the far/near counts we surface in the per-device diagnostic vector. There is
    no transient pair-queue (``queue_overflow`` is always 0 -- that is the point of the
    swap), but the flat far/near buffers can still overflow, so ``far_overflow`` /
    ``near_overflow`` are set from the true counts vs the (right-sized) caps and drive
    ``auto_scale_caps`` -- the eager-only builder guard is skipped under the trace.
    """

    far_pair_count: Any
    near_pair_count: Any
    queue_overflow: Any
    far_overflow: Any
    near_overflow: Any


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
    # dehnen bounding-SPHERE MAC extents (the correct multipole-radius bound, matching
    # the single-GPU fast lane and required for stable multi-step use). "bh" (box) is
    # cheaper but under-bounds the source radius; keep it only for single-shot use.
    mac_type: str = "dehnen"
    # Far-field expansion basis: "real" (Dehnen no-sqrt2, DEFAULT) or "solidfmm" (complex).
    # "real" is the per-device far field converged onto the single-GPU fast-lane path
    # (memory-lighter, and unlocks the fused real M2L Pallas kernel when
    # JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS is set). Upward sweep + coarse M2M stay
    # complex; multipoles are converted to Dehnen real coeffs at the M2L boundary.
    basis: str = "real"
    # Near-field backend: "baseline" (pure-JAX combined [local;halo] P2P) or
    # "pallas" (fused leafpair Pallas kernel, sm_80+, the single-GPU fast-lane
    # near-field). "auto" (DEFAULT) picks pallas on Ampere+, baseline elsewhere.
    # Numerically equivalent to baseline (validated), only the cross-leaf P2P is fused.
    nearfield_backend: str = "auto"
    # Local self-interaction walk: "dual_tree" (yggdrax dual-tree walk, DEFAULT) or
    # "treecode" (the single-GPU fast-lane device-resident treecode walk). The dual-tree
    # walk's transient pair-queue caps per-GPU N (self_queue_overflow); the treecode walk
    # streams far/near with no such queue, so per-GPU N scales like the single-GPU lane.
    # Parity with dual_tree at mac_type="dehnen" (accuracy-profile parity, leaf-only far
    # targets -> L2L no-op, self-excluded near CSR). Cross-domain LET is unchanged.
    local_walk: str = "dual_tree"
    # Sphere-radius scale for the treecode dehnen MAC (matches the dual-tree extents).
    dehnen_radius_scale: float = 1.0
    # Treecode local-walk flat buffer sizes (only used when local_walk="treecode").
    # The builder's own default is a fixed 1<<21 (2M) near-edge buffer, which makes the
    # combined-P2P neighbour build chew a 2M-edge array per device AND can SILENTLY
    # truncate the near list at ~1M/GPU (the treecode overflow guard is eager-only, so
    # it is skipped under the shard_map trace -> wrong forces with no diagnostic). When
    # these are None the driver right-sizes them from the local tree: the near buffer to
    # ``max_neighbors_per_leaf * num_leaves`` (the same per-leaf near budget the dual-tree
    # walk uses, and already grown by ``with_scaled_caps`` on the auto-scale retry) and
    # the far buffer to ``treecode_far_cap`` or 131072. The true per-device far/near
    # counts and an accurate overflow flag are surfaced in the ``self_*`` diagnostics so
    # ``auto_scale_caps`` grows them on overflow exactly like the dual-tree caps.
    treecode_near_cap: Optional[int] = None
    treecode_far_cap: Optional[int] = None
    # self dual-tree walk capacities
    max_interactions_per_node: int = 512
    max_neighbors_per_leaf: int = 128
    max_pair_queue: int = 1 << 15
    process_block: int = 64
    # cross-walk (local vs remote coarse tree) capacities
    cross_max_interactions_per_node: int = 512
    cross_max_neighbors_per_leaf: int = 128
    cross_max_pair_queue: int = 1 << 15
    # Static length of the cross-far M2L input. The cross walk sizes its far buffer at
    # t_total * cross_max_interactions_per_node but PACKS the valid interactions into the
    # front; feeding the whole (mostly-invalid) buffer to the fused real-M2L pads it to a
    # power of two and OOMs (~6 GiB at N>=400k/GPU). When None the driver right-sizes the
    # M2L input to cross_max_interactions_per_node * num_target_leaves (~2x tighter than
    # the buffer, and >= the far volume on the ICs measured), overflowing into
    # cross_far_overflow so auto_scale grows it. Set an explicit value to override.
    cross_far_cap: Optional[int] = None
    # Run the far-field M2L (self + cross) in fp32. The per-pair rotation blocks
    # [N, Bp, mdp, mdp] dominate M2L peak memory; fp32 halves it (≈2x more particles/GPU)
    # at negligible accuracy cost (fp32 ~1e-7 << the order-p FMM truncation error). The
    # accumulation into the local expansions stays coeff_dtype (fp64). Opt-in; the
    # single-GPU fast lane is unaffected (this field is distributed-only).
    far_m2l_fp32: bool = False
    # Chunk the far-field M2L (both self and cross) over pairs in blocks of this many
    # pairs (None = one full batch). The fused M2L builds per-pair rotation blocks
    # [Npairs, Bp, mdp, mdp]; doing a whole far list at once is the M2L OOM (cross-far
    # >200k/GPU; self-far >1.2M/GPU). Scanning fixed-size blocks bounds peak M2L memory to
    # ~block pairs regardless of the total, removing the M2L per-GPU ceiling at some
    # throughput cost. Numerics-identical (associative masked segment-sum accumulation).
    m2l_chunk: Optional[int] = None
    # Chunk the fused-Pallas NEAR field over blocks of this many TARGET leaves (None = one
    # full batch). The near field densifies the combined CSR into [u_leaves, S_near] tables
    # + [num_edges] temporaries and runs the leafpair P2P over all u_leaves at once -- the
    # ~40GB aggregate wall at >1.2M/GPU. Scanning fixed-size target-leaf blocks bounds both
    # the densification and the P2P peak to ~block*S_near (source pool stays resident),
    # removing the near-field per-GPU ceiling at some throughput cost. Numerics-identical
    # (disjoint target blocks + scatter-add). Applies ONLY when the pallas near-field is
    # active (nearfield_backend "pallas", or "auto" on Ampere+); ignored for baseline.
    nearfield_chunk: Optional[int] = None

    def with_scaled_caps(
        self: "DistributedFMMConfig", factor: float
    ) -> "DistributedFMMConfig":
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
            cross_far_cap=(
                None if self.cross_far_cap is None else g(self.cross_far_cap)
            ),
            # Grow the treecode buffers on retry too. When these are None the driver
            # sizes the near buffer off ``max_neighbors_per_leaf`` (scaled above), so the
            # retry already grows the effective near cap; scale any explicit values here.
            treecode_near_cap=(
                None if self.treecode_near_cap is None else g(self.treecode_near_cap)
            ),
            treecode_far_cap=(
                None if self.treecode_far_cap is None else g(self.treecode_far_cap)
            ),
        )

    def with_selective_scaled_caps(
        self: "DistributedFMMConfig", diag_o: Any, factor: float
    ) -> "DistributedFMMConfig":
        """Return a copy scaling ONLY the caps whose overflow flag actually fired.

        ``diag_o`` is the per-device diagnostic matrix (shape ``(ndev, len(DIAG_FIELDS))``)
        from the last evaluation; a cap is grown by ``factor`` iff its overflow flag is
        nonzero on any device.

        Unlike :meth:`with_scaled_caps` (which grows *every* cap uniformly), this grows
        only the buffer a device actually overflowed. Uniform scaling couples the
        self/cross x far/near/queue caps, so a single overflowing buffer drags up an
        unrelated, already-oversized cap. Concretely: on a connected IC only the
        cross-near buffer overflows, but uniform scaling also doubles
        ``cross_max_interactions_per_node`` (the cross-*far* cap) each retry -- and the
        cross-far list feeds the fused real-M2L, whose Pallas buffer pads to a power of
        two and OOMs (~6 GiB at N>=200k/2GPU) even though the true cross-far volume is
        ~10^3 pairs. Selective scaling grows just the buffer that overflowed, lifting the
        connected-IC ceiling without inflating the M2L.
        """
        diag = np.asarray(diag_o)

        def flagged(name: str) -> bool:
            return bool(np.any(diag[:, DIAG_FIELDS.index(name)] > 0))

        def g(v: int) -> int:
            return int(np.ceil(v * factor))

        def maybe(name: str, value: int) -> int:
            return g(value) if flagged(name) else value

        # Under the treecode local walk the self near/far caps derive from
        # max_neighbors_per_leaf / max_interactions_per_node (times num_leaves) when the
        # explicit treecode_*_cap overrides are None, so scaling those knobs grows the
        # effective self buffers for both the dual-tree and treecode walks.
        grow_self_far = flagged("self_far_overflow")
        grow_self_near = flagged("self_near_overflow")
        grow_cross_far = flagged("cross_far_overflow")
        return dataclasses.replace(
            self,
            max_interactions_per_node=maybe(
                "self_far_overflow", self.max_interactions_per_node
            ),
            max_neighbors_per_leaf=maybe(
                "self_near_overflow", self.max_neighbors_per_leaf
            ),
            max_pair_queue=maybe("self_queue_overflow", self.max_pair_queue),
            cross_max_interactions_per_node=maybe(
                "cross_far_overflow", self.cross_max_interactions_per_node
            ),
            cross_max_neighbors_per_leaf=maybe(
                "cross_near_overflow", self.cross_max_neighbors_per_leaf
            ),
            cross_max_pair_queue=maybe(
                "cross_queue_overflow", self.cross_max_pair_queue
            ),
            # cross_far_overflow also covers the M2L-input right-size (far volume >
            # cross_far_cap); grow an explicit cap so the retry widens the M2L slice. When
            # None the slice derives from cross_max_interactions_per_node (grown above).
            cross_far_cap=(
                g(self.cross_far_cap)
                if (grow_cross_far and self.cross_far_cap is not None)
                else self.cross_far_cap
            ),
            treecode_near_cap=(
                g(self.treecode_near_cap)
                if (grow_self_near and self.treecode_near_cap is not None)
                else self.treecode_near_cap
            ),
            treecode_far_cap=(
                g(self.treecode_far_cap)
                if (grow_self_far and self.treecode_far_cap is not None)
                else self.treecode_far_cap
            ),
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
    def overflow(self: "DistributedFMMResult") -> bool:
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


def _chunked_real_m2l_accumulate(
    loc_init: jax.Array,
    multip: jax.Array,
    src_idx: jax.Array,
    tgt_idx: jax.Array,
    src_centers: jax.Array,
    tgt_centers: jax.Array,
    valid: jax.Array,
    *,
    order: int,
    block: int,
    total_nodes: int,
    m2l_dtype: Any,
) -> jax.Array:
    """Accumulate the real-basis M2L over pairs in fixed-size blocks (bounded peak mem).

    The fused M2L builds per-pair rotation blocks ``[Npairs, Bp, mdp, mdp]``; running a whole
    far list in one batch is the M2L OOM. Scan over ``ceil(Npairs/block)`` blocks so peak M2L
    memory is ~``block`` pairs, independent of the total. The block M2Ls are masked and
    segment-summed into ``loc_init`` -- associative, so numerics-identical to the full batch.

    Index-based (not pre-gathered): only the compact ``multip`` (``[total_nodes, C]``), the
    index arrays, and the centre arrays are resident; the per-pair source multipoles and deltas
    are gathered ``[block, *]`` INSIDE the scan. This avoids materialising the full
    ``[Npairs, C]`` gather (itself a >1GB peak at high N), so the only large intermediates are
    the per-block rotation blocks. Deltas are ``tgt - src`` in the source dtype then cast.
    """
    npairs = src_idx.shape[0]
    nblk = -(-npairs // block)  # ceil
    pad_n = nblk * block - npairs
    out_dtype = loc_init.dtype
    si = jnp.pad(src_idx, (0, pad_n)).reshape(nblk, block)
    ti = jnp.pad(tgt_idx, (0, pad_n)).reshape(nblk, block)
    vd = jnp.pad(valid, (0, pad_n), constant_values=False).reshape(nblk, block)

    def body(
        loc: jax.Array,
        blk: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, None]:
        sib, tib, vdb = blk
        smb = multip[sib].astype(m2l_dtype)
        dlb = (tgt_centers[tib] - src_centers[sib]).astype(m2l_dtype)
        contribs = _apply_real_m2l(smb, dlb, order=order, m2l_impl="rot_scale").astype(
            out_dtype
        )
        contribs = jnp.where(vdb[:, None], contribs, 0)
        loc = loc + jax.ops.segment_sum(contribs, jnp.where(vdb, tib, 0), total_nodes)
        return loc, None

    loc, _ = jax.lax.scan(body, loc_init, (si, ti, vd))
    return loc


def _chunked_pallas_nearfield_accumulate(
    leaf_positions: jax.Array,
    leaf_masses: jax.Array,
    leaf_mask_g: jax.Array,
    safe_idx: jax.Array,
    concat_pos: jax.Array,
    offsets: jax.Array,
    counts: jax.Array,
    src_s: jax.Array,
    *,
    n_lloc: int,
    S_near: int,
    block: int,
    G: Any,
    softening_sq: jax.Array,
) -> jax.Array:
    """Fused-Pallas near field over blocks of TARGET leaves (bounded peak memory).

    Full-batch near field densifies the combined CSR into ``[u_leaves, S_near]`` tables (+
    ``[num_edges]`` temporaries) and runs the leafpair P2P over all ``u_leaves`` at once -- the
    ~40GB aggregate wall at >1.2M/GPU. This scans ``ceil(n_lloc/block)`` target-leaf blocks,
    building each block's ``[block, S_near]`` densification + running self + pairs on the block,
    so peak drops from ``u_leaves*S_near`` to ``block*S_near``. The SOURCE gather pool
    (``leaf_positions``/``leaf_masses``/``leaf_mask_g``) stays fully resident (sources span all
    ``u_leaves`` rows). Only target rows ``[0, n_lloc)`` produce kept output; per-block partials
    scatter-add into a ``[cap+halo, 3]`` accumulator, so it is numerics-identical to the full batch
    (disjoint target blocks -> exactly identical, not merely associative).

    The per-block densification is built DIRECTLY from the CSR (``offsets``/``counts``/``src_s``):
    ``sids_blk[i, j] = src_s[offsets[b0+i] + j]`` for ``j < min(counts[b0+i], S_near)`` -- the first
    (up to) ``S_near`` sources of each target leaf, exactly what the full densification keeps (rank
    ``< S_near``). This is ``[block, S_near]`` throughout (no ``[num_edges]`` temporaries) and is
    bit-identical to the full batch: same ``src_s`` order, same per-leaf truncation, so raw per-leaf
    counts exceeding ``S_near`` (offsets uncapped, counts from bincount) are truncated identically.
    """
    nblk = -(-n_lloc // block)  # ceil
    num_edges = int(src_s.shape[0])
    near0 = jnp.zeros_like(concat_pos)
    cols = jnp.arange(S_near, dtype=INDEX_DTYPE)  # [S_near]
    src_zero = jnp.asarray(0, src_s.dtype)

    def _body(near: jax.Array, k: jax.Array) -> tuple[jax.Array, None]:
        b0 = k * block
        rows = b0 + jnp.arange(block, dtype=INDEX_DTYPE)
        valid_t = rows < n_lloc
        safe_rows = jnp.where(valid_t, rows, 0)
        tgt_pos = leaf_positions[safe_rows]  # [block, W, 3]
        tgt_mass = leaf_masses[safe_rows]  # [block, W]
        tgt_mask = leaf_mask_g[safe_rows] & valid_t[:, None]  # [block, W]
        tgt_idx = safe_idx[safe_rows]  # [block, W] global scatter targets

        # self term (block): pure-JAX per-leaf; scatters to global-order [cap+halo, 3].
        self_blk = _compute_leaf_p2p_prepared_large_n_self_only_impl(
            concat_pos, tgt_pos, tgt_mass, tgt_mask, tgt_idx, G=G, softening_sq=softening_sq
        )

        # per-block densification built DIRECTLY from the CSR (no [num_edges] window):
        # sids_blk[i, j] = src_s[offsets[b0+i] + j] for j < min(counts[b0+i], S_near).
        leaf_off = offsets[safe_rows]  # [block] each target leaf's edge start
        leaf_cnt = counts[safe_rows]  # [block] raw near count per target leaf
        keep_col = (cols[None, :] < jnp.minimum(leaf_cnt, S_near)[:, None]) & valid_t[
            :, None
        ]  # [block, S_near]
        edge_idx = leaf_off[:, None] + cols[None, :]  # [block, S_near]
        safe_edge = jnp.clip(edge_idx, 0, num_edges - 1)
        sids_blk = jnp.where(keep_col, src_s[safe_edge], src_zero)  # [block, S_near]
        svalid_blk = keep_col

        # pair term (block): full source pool resident, target restricted to the block.
        pair_blk = _radix_fast_lane_prepacked_pallas_decoupled(
            sids_blk[:, :, None],
            svalid_blk[:, :, None],
            tgt_pos,
            tgt_mask,
            tgt_idx,
            leaf_positions,
            leaf_masses,
            leaf_mask_g,
            concat_pos,
            G=G,
            softening_sq=softening_sq,
            compute_potential=False,
        )
        return near + self_blk + pair_blk, None

    near, _ = jax.lax.scan(_body, near0, jnp.arange(nblk, dtype=INDEX_DTYPE))
    return near


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

    # Local self-walk selection: dual-tree (default) or the fast-lane treecode walk.
    lw = str(config.local_walk).strip().lower()
    if lw not in {"dual_tree", "treecode"}:
        raise ValueError(
            f"local_walk must be 'dual_tree' or 'treecode'; got {config.local_walk!r}"
        )
    use_treecode_local = lw == "treecode"
    dehnen_radius_scale = float(config.dehnen_radius_scale)

    # Near-field backend resolution (trace-time; sm_80+ gate mirrors the single-GPU
    # lane). "auto" -> fused leafpair Pallas on Ampere+, pure-JAX baseline elsewhere.
    nf_backend = str(config.nearfield_backend).strip().lower()
    if nf_backend not in {"auto", "pallas", "baseline"}:
        raise ValueError(
            "nearfield_backend must be 'auto', 'pallas' or 'baseline'; "
            f"got {config.nearfield_backend!r}"
        )
    if nf_backend == "auto":
        from jaccpot.pallas.nearfield_fused_leaf import pallas_nearfield_fused_supported

        use_pallas_near = bool(pallas_nearfield_fused_supported())
    else:
        use_pallas_near = nf_backend == "pallas"
    # Padded source-leaf-id width for the fused leafpair kernel: per target leaf the
    # combined CSR holds at most (local self-near + cross-near) neighbour leaves.
    S_near = int(config.max_neighbors_per_leaf) + int(
        config.cross_max_neighbors_per_leaf
    )
    soft2 = float(soft) ** 2

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
        # tgt holds target leaf-row ids (<= u_leaves << 2^31), so int32 halves the argsort
        # value array and the bincount scatter (the 1.27GiB near-field alloc at high N).
        tgt = jnp.where(valid, tgt, sentinel).astype(jnp.int32)
        src = jnp.where(valid, src, sentinel)

        srt = jnp.argsort(tgt)
        src_s = src[srt]
        counts = jnp.bincount(tgt, length=u_leaves).astype(INDEX_DTYPE)
        offsets = jnp.concatenate([jnp.zeros((1,), INDEX_DTYPE), jnp.cumsum(counts)])
        return offsets, src_s, counts, u_leaves

    def fn(
        pos: jax.Array, mass: jax.Array, gid: jax.Array, count: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
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

        # local upward. Real basis runs a NATIVE-real P2M + M2M (multipoles are
        # real from the start -> the coarse-tree all_gather below ships half the
        # bytes); solidfmm runs the complex sweep. Both produce per-node multipoles
        # + COM centers with the same API.
        if is_real:
            up = prepare_real_upward_sweep(
                tree, lp, lm, max_order=p, max_leaf_size=leaf
            )
        else:
            up = prepare_solidfmm_complex_upward_sweep(
                tree, lp, lm, max_order=p, max_leaf_size=leaf, rotation=rot
            )
        centers = up.multipoles.centers
        packed = up.multipoles.packed  # real when is_real, complex otherwise
        coeff_dtype = lp.dtype if is_real else cdtype
        packed_use = packed

        if use_treecode_local:
            # Fast-lane device-resident treecode walk in place of the dual-tree walk.
            # Drop-in: compact_far_pairs (leaf-only far targets -> L2L no-op) feed the
            # same real M2L; the self-excluded near CSR feeds the same combined P2P.
            #
            # Right-size the flat far/near buffers from the (static) local tree instead
            # of the builder's fixed 1<<21 near default: the near buffer is what the
            # combined-P2P neighbour build iterates over, so an oversized buffer is pure
            # overhead, and at ~1M/GPU the fixed 2M could be EXCEEDED (silent truncation:
            # the treecode overflow guard is eager-only, skipped under this trace). The
            # near budget mirrors the dual-tree walk's per-leaf bound (already grown by
            # with_scaled_caps on the auto-scale retry).
            num_internal = int(tree.topology.left_child.shape[0])
            num_leaves = int(total_nodes) - num_internal
            if config.treecode_near_cap is not None:
                tc_near_cap = int(config.treecode_near_cap)
            else:
                tc_near_cap = max(
                    1 << 14, int(config.max_neighbors_per_leaf) * num_leaves
                )
            # Far mirrors near: the compact far list holds <= (interaction-list size) far
            # pairs per leaf, so budget max_interactions_per_node * num_leaves. Keyed off
            # a with_scaled_caps-scaled field so the auto-scale retry grows it too, and
            # num_leaves-proportional so it does not under-size at ~1M/GPU (the fixed
            # 131072 could be exceeded by a spread IC -> silent far truncation).
            if config.treecode_far_cap is not None:
                tc_far_cap = int(config.treecode_far_cap)
            else:
                tc_far_cap = max(
                    1 << 14, int(config.max_interactions_per_node) * num_leaves
                )
            _art = _build_treecode_artifacts_strict_streamed(
                tree=tree,
                geometry=geom,
                theta=theta,
                mac_type=mac,
                dehnen_radius_scale=dehnen_radius_scale,
                compact_far_pair_capacity=tc_far_cap,
                near_cap=tc_near_cap,
            )
            inter = _art.compact_far_pairs
            nbr = _art.neighbor_list
            _z = jnp.zeros((), jnp.int64)
            # near_counts are UNCLAMPED true counts (see _compact_near), so > cap is exact
            # truncation; far_pair_count is clamped to far_cap, so == cap flags a possible
            # overflow. Surfaced so _reduce_overflow/auto_scale_caps grow the caps.
            far_cnt = jnp.asarray(inter.far_pair_count)
            near_cnt = jnp.sum(jnp.asarray(nbr.counts))
            self_res = _TreecodeWalkDiag(
                far_pair_count=far_cnt,
                near_pair_count=near_cnt,
                queue_overflow=_z,
                far_overflow=(far_cnt >= tc_far_cap).astype(jnp.int64),
                near_overflow=(near_cnt > tc_near_cap).astype(jnp.int64),
            )
        else:
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

        # coarse centres (COM) + level structure (same basis as the local sweep)
        if is_real:
            upc = prepare_real_upward_sweep(
                rct.tree,
                rct.positions_sorted,
                rct.masses_sorted,
                max_order=p,
                max_leaf_size=1,
            )
        else:
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

        # seed coarse leaves with the remote leaves' own multipoles. The gathered
        # array is REAL in the real basis (half the bytes of the complex packed).
        gpacked = jax.lax.all_gather(packed, "gpus", tiled=False)  # [ndev,N,C]
        spos = c_nr[c_leaves, 0]
        dom = rct.tag_domain[spos]
        nod = rct.tag_node_id[spos]
        okm = nod >= 0
        leafp = gpacked[jnp.where(okm, dom, 0), jnp.where(okm, nod, 0)]
        leafp = jnp.where(okm[:, None], leafp, 0.0)
        seed = (
            jnp.zeros((c_total, C), dtype=coeff_dtype)
            .at[c_leaves]
            .set(leafp.astype(coeff_dtype))
        )
        c_nbl = get_nodes_by_level(rct.tree)
        c_loff = get_level_offsets(rct.tree)
        c_numlev = int(c_loff.shape[0] - 1)
        if is_real:
            coarse_packed = aggregate_m2m_real_by_level(
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
            )
        else:
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
        coarse_packed_use = coarse_packed  # real when is_real, complex otherwise

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
        # Far-M2L compute dtype: fp32 halves the per-pair rotation-block memory (the M2L
        # peak) at negligible accuracy cost; accumulation stays coeff_dtype (fp64).
        m2l_dtype = jnp.float32 if config.far_m2l_fp32 else coeff_dtype
        s_src = jnp.asarray(inter.sources, INDEX_DTYPE)
        s_tgt = jnp.asarray(inter.targets, INDEX_DTYPE)
        # The treecode compact far pairs are 0-padded (not -1) with the true count in
        # far_pair_count; the dual-tree list is trimmed, so a >=0 test recovers its count.
        if use_treecode_local:
            s_active = jnp.asarray(inter.far_pair_count, INDEX_DTYPE)
        else:
            s_active = jnp.sum((s_tgt >= 0).astype(INDEX_DTYPE))
        if is_real and config.m2l_chunk:
            # Blockwise self-far M2L (same bound as the cross-far path): the per-pair
            # rotation blocks are the >1.2M/GPU OOM here too. Reproduce the fullbatch
            # gather (valid mask, safe src/tgt, tgt-minus-src deltas) then scan in blocks.
            s_idx = jnp.arange(s_src.shape[0], dtype=INDEX_DTYPE)
            s_valid = (s_idx < s_active) & (s_src >= 0) & (s_tgt >= 0)
            s_safe_src = jnp.where(s_valid, s_src, 0)
            s_safe_tgt = jnp.where(s_valid, s_tgt, 0)
            # Index-based: gather the source multipoles + deltas inside the scan (avoids the
            # full [Npairs, C] gather that otherwise peaks >1GB at high N). Self tree = single
            # centre array for both src and tgt.
            loc_self = _chunked_real_m2l_accumulate(
                zeros,
                packed_use,
                s_safe_src,
                s_safe_tgt,
                centers,
                centers,
                s_valid,
                order=p,
                block=int(config.m2l_chunk),
                total_nodes=int(total_nodes),
                m2l_dtype=m2l_dtype,
            )
        elif is_real:
            loc_self = _accumulate_m2l_fullbatch(
                zeros,
                packed_use.astype(m2l_dtype),
                centers.astype(m2l_dtype),
                s_src,
                s_tgt,
                s_active,
                order=p,
                basis_mode="real",
                m2l_impl="rot_scale",
                total_nodes=int(total_nodes),
            )
        else:
            loc_self = _accumulate_m2l_fullbatch(
                zeros,
                packed_use,
                centers,
                s_src,
                s_tgt,
                s_active,
                order=p,
                basis_mode="complex",
                rotation=rot,
                total_nodes=int(total_nodes),
            )

        # cross M2L: separate source (coarse) / target (local) centres.
        # The cross walk sizes interaction_sources at t_total * KC but PACKS the valid far
        # interactions into the front [0, far_pair_count); the tail is -1. Feeding the
        # whole buffer to the fused real-M2L pads that length to a power of two and OOMs
        # (~6 GiB at N>=400k/GPU). Slice to a right-sized cross-far cap so the M2L only
        # spans the actual far volume. KC * num_target_leaves mirrors the self treecode
        # far cap and is ~2x tighter than the t_total*KC buffer; cross_far_cap overrides.
        # A far volume above the cap is surfaced (cross_far_overflow, below) so auto_scale
        # widens it -- so the slice never silently drops interactions.
        max_far = int(jnp.asarray(cross.interaction_sources).shape[0])
        num_tgt_leaves = int(jnp.asarray(nbr.leaf_indices).shape[0])
        if config.cross_far_cap is not None:
            xfar_cap = int(config.cross_far_cap)
        else:
            xfar_cap = max(1 << 14, KC * num_tgt_leaves)
        xfar_cap = min(xfar_cap, max_far)
        x_src = jnp.asarray(cross.interaction_sources, INDEX_DTYPE)[:xfar_cap]
        x_tgt = jnp.asarray(cross.interaction_targets, INDEX_DTYPE)[:xfar_cap]
        x_valid = x_tgt >= 0
        xs = jnp.where(x_valid, x_src, 0)
        xt = jnp.where(x_valid, x_tgt, 0)
        if is_real and config.m2l_chunk:
            # Blockwise cross-far M2L, index-based (gathers source multipoles + deltas per
            # block, so no full [Npairs, C] materialisation). Bounds peak memory; numerics-
            # identical to the full batch (associative accumulate). src=coarse tree/centres.
            loc_full = _chunked_real_m2l_accumulate(
                loc_self,
                coarse_packed_use,
                xs,
                xt,
                c_centers,
                centers,
                x_valid,
                order=p,
                block=int(config.m2l_chunk),
                total_nodes=int(total_nodes),
                m2l_dtype=m2l_dtype,
            )
        else:
            x_deltas = centers[xt] - c_centers[xs]
            if is_real:
                x_contribs = _apply_real_m2l(
                    coarse_packed_use[xs].astype(m2l_dtype),
                    x_deltas.astype(m2l_dtype),
                    order=p,
                    m2l_impl="rot_scale",
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
        if use_pallas_near:
            # Fused leafpair Pallas near-field (single-GPU fast-lane kernel): the
            # intra-leaf self block is computed separately and the cross-leaf
            # neighbour pairs run through the leafpair kernel (sources gathered by
            # id in-kernel). Mirrors compute_leaf_p2p_accelerations_radix_fast_lane;
            # numerically equal to the baseline combined P2P (CPU-validated 2e-16).
            soft2_a = jnp.asarray(soft2, concat_pos.dtype)
            (
                leaf_positions,
                leaf_masses,
                leaf_mask_g,
                safe_idx,
            ) = _prepare_leaf_data_from_groups(lp_idx, lp_mask, concat_pos, concat_mass)
            if config.nearfield_chunk:
                # Bounded-memory path: scan target-leaf blocks (densification + leafpair
                # P2P per block) instead of materialising the full [u_leaves, S_near] tables
                # and running all u_leaves at once. Numerics-identical (disjoint target
                # blocks + scatter-add). See _chunked_pallas_nearfield_accumulate.
                near_full = _chunked_pallas_nearfield_accumulate(
                    leaf_positions,
                    leaf_masses,
                    leaf_mask_g,
                    safe_idx,
                    concat_pos,
                    offsets,
                    counts,
                    src_s,
                    n_lloc=int(loc_idx.shape[0]),
                    S_near=S_near,
                    block=int(config.nearfield_chunk),
                    G=G,
                    softening_sq=soft2_a,
                )
            else:
                self_acc = _compute_leaf_p2p_prepared_large_n_self_only_impl(
                    concat_pos,
                    leaf_positions,
                    leaf_masses,
                    leaf_mask_g,
                    safe_idx,
                    G=G,
                    softening_sq=soft2_a,
                )
                # Densify the combined CSR (offsets/src_s/counts, sorted by target) into a
                # padded [u_leaves, S_near] source-leaf-id table + validity mask.
                # int32 index math: e/t_of_e/rank/flat are pure [num_edges] index temporaries
                # (values < u_leaves*S_near << 2^31 at any feasible per-GPU N), so int32 halves
                # the near-field densification peak (the aggregate wall at >1.2M/GPU). Kernel-
                # facing arrays (offsets/src_s/counts/sids) stay INDEX_DTYPE.
                _i32 = jnp.int32
                e = jnp.arange(src_s.shape[0], dtype=_i32)
                t_of_e = (jnp.searchsorted(offsets, e, side="right") - 1).astype(_i32)
                rank = (e - offsets[jnp.clip(t_of_e, 0, u_leaves)].astype(_i32)).astype(
                    _i32
                )
                in_range = (t_of_e >= 0) & (t_of_e < u_leaves) & (rank < S_near)
                flat = jnp.where(in_range, t_of_e * _i32(S_near) + rank, _i32(-1))
                sids = (
                    jnp.zeros((u_leaves * S_near,), INDEX_DTYPE)
                    .at[flat]
                    .set(src_s, mode="drop")
                    .reshape(u_leaves, S_near)
                )
                svalid = (
                    jnp.zeros((u_leaves * S_near,), bool)
                    .at[flat]
                    .set(jnp.ones_like(flat, bool), mode="drop")
                    .reshape(u_leaves, S_near)
                )
                pair_acc = _radix_fast_lane_prepacked_pallas(
                    sids.reshape(u_leaves, S_near, 1),
                    svalid.reshape(u_leaves, S_near, 1),
                    leaf_positions,
                    leaf_masses,
                    leaf_mask_g,
                    safe_idx,
                    concat_pos,
                    G=G,
                    softening_sq=soft2_a,
                    compute_potential=False,
                )
                near_full = self_acc + pair_acc
        else:
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

        # cross-far overflow = per-node far-buffer overflow (KC) OR the M2L-input slice
        # dropping interactions (far volume > xfar_cap); either way auto_scale grows it.
        cross_far_ovf = jnp.maximum(
            cross.far_overflow.astype(jnp.float64),
            (cross.far_pair_count > xfar_cap).astype(jnp.float64),
        )
        diag = jnp.array(
            [
                cross.far_pair_count.astype(jnp.float64),
                cross.near_pair_count.astype(jnp.float64),
                cross.queue_overflow.astype(jnp.float64),
                cross_far_ovf,
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
    mesh: Any,
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

    def evaluate(
        pos_flat: jax.Array,
        mass_flat: jax.Array,
        gid_flat: jax.Array,
        counts: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return shard_map(
            fn,
            mesh=mesh,
            in_specs=(P("gpus"), P("gpus"), P("gpus"), P("gpus")),
            out_specs=(P("gpus"), P("gpus"), P("gpus")),
            check_vma=False,
        )(pos_flat, mass_flat, gid_flat, counts)

    return jax.jit(evaluate) if jit else evaluate


_OVERFLOW_FIELDS = (
    "cross_queue_overflow",
    "cross_far_overflow",
    "cross_near_overflow",
    "self_queue_overflow",
    "self_far_overflow",
    "self_near_overflow",
)


def _reduce_overflow(diag_o: np.ndarray) -> bool:
    """True if any device flagged a traversal-buffer overflow."""
    idx = [DIAG_FIELDS.index(k) for k in _OVERFLOW_FIELDS]
    return bool(np.any(diag_o[:, idx] > 0))


def distributed_fmm_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    config: DistributedFMMConfig | None = None,
    mesh: Any = None,
    ndev: int | None = None,
    jit: bool = False,
    auto_scale_caps: bool = False,
    cap_scale_factor: float = 2.0,
    max_cap_retries: int = 4,
    cap_presets_path: str | None = None,
) -> DistributedFMMResult:
    """Evaluate distributed FMM accelerations for all particles.

    Handles the host-side SFC decomposition + padding + global-id tracking,
    runs the ``shard_map`` force pipeline, and scatters the per-device forces
    back into the original input order.

    When ``auto_scale_caps`` is set, a traversal-buffer overflow (which grows
    with device count as the cross-domain LET expands) triggers a retry with the
    capacities scaled by ``cap_scale_factor`` (up to ``max_cap_retries`` times),
    rebuilding the evaluator each time. The ``config`` on the returned result is
    the one that actually produced the (non-overflowing) forces.

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

    part = partition_for_devices(positions, masses, ndev, leaf_size=config.leaf_size)
    cap = part["cap"]
    counts_dev = jnp.asarray(part["counts"], INDEX_DTYPE)
    pos_f = jnp.asarray(part["pos_flat"])
    mass_f = jnp.asarray(part["mass_flat"])
    gid_f = jnp.asarray(part["gid_flat"])

    # Seed the caps from a saved preset for this (per-GPU N, ndev) so a repeat run at a
    # known size starts already-sized and skips the auto_scale recompiles. auto_scale
    # stays on as the safety net; the converged caps refresh the preset below.
    total_n = int(part["n"])
    per_gpu_n = -(-total_n // ndev)  # ceil
    presets = {}
    if cap_presets_path is not None:
        from . import cap_presets as _cp

        presets = _cp.load_presets(cap_presets_path)
        seed = _cp.lookup(presets, per_gpu_n, ndev)
        if seed is not None:
            config = _cp.apply_caps(config, seed)

    attempt = 0
    while True:
        evaluate = make_force_evaluator(config, ndev, cap, mesh, jit=jit)
        accel_o, gid_o, diag_o = evaluate(pos_f, mass_f, gid_f, counts_dev)
        diag_o = np.asarray(diag_o)
        overflow = _reduce_overflow(diag_o)
        if not overflow or not auto_scale_caps or attempt >= max_cap_retries:
            break
        # Grow only the buffers that actually overflowed. Uniform scaling couples the
        # cross-far cap (which feeds the fused-M2L Pallas buffer) to unrelated overflows
        # and OOMs on connected ICs; see with_selective_scaled_caps.
        config = config.with_selective_scaled_caps(diag_o, cap_scale_factor)
        attempt += 1
    accel_o = np.asarray(accel_o)
    gid_o = np.asarray(gid_o).reshape(-1).astype(np.int64)

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
    diag["overflow"] = overflow
    diag["cap_retries"] = attempt

    # Persist the converged caps as the preset for this size (only when overflow-free, so
    # we never cache a truncated/wrong config). Next run at this (per-GPU N, ndev) reuses
    # them and skips the retries.
    if cap_presets_path is not None and not overflow:
        from . import cap_presets as _cp

        _cp.record(presets, per_gpu_n, ndev, total_n, _cp.caps_of(config))
        _cp.save_presets(cap_presets_path, presets)

    return DistributedFMMResult(
        accelerations=accel,
        diagnostics=diag,
        ndev=ndev,
        cap=cap,
        config=config,
    )
