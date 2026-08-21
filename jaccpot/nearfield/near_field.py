"""Near-field evaluation helpers for the Fast Multipole Method."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jax import lax
from jaxtyping import Array, Bool, Float, Int, jaxtyped
from yggdrax.dtypes import INDEX_DTYPE, as_index
from yggdrax.interactions import NodeNeighborList
from yggdrax.tree import Tree

from jaccpot._env import env_choice, env_flag, env_int

# Several of these are unused *in this module* -- some were already, and four more
# became so when the radix fast lane moved to `_fast_lane.py` (Tier 1.4). They are
# kept because they are a re-export surface, not dead code (F16): consumers reach
# them both by `from ...near_field import X` (`tests/unit/test_custom_vjp_parity.py`,
# `bench/audit_nearfield_padding.py`) and as attributes of the module
# (`nf._leafpair_accel_analytic_vjp` in `tests/unit/test_nearfield_fastlane_grad_path.py`).
# Removing them keeps every import in the package working and breaks callers at
# attribute-access time instead, which is how F16 was originally mis-measured. Do
# not let a pyflakes sweep take them.
from ._kernels import (
    _bucketed_chunk_pair_accels_remat,
    _pair_contributions,
    _pair_contributions_batched,
    _self_contributions,
)

# `compute_leaf_p2p_accelerations_target_block_pairs_only` and the
# `prepare_*` schedule builders below are unused *in this module* but are part of
# its public surface: `runtime/_large_n_nearfield.py`, `runtime/_nearfield_fastlane.py`
# and the tests import them from here. Keeping them re-exported is safe -- none of
# the four new sibling modules imports `near_field`, so there is no cycle -- and it
# means the seam split changes no call site outside `nearfield/` (F16).
from ._large_n_blocks import (  # noqa: F401
    _compute_leaf_p2p_prepared_large_n_accel_only_impl,
    _compute_leaf_p2p_prepared_large_n_accel_only_target_blocks_impl,
    _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl,
    _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl,
    _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_tiled_impl,
    _compute_leaf_p2p_prepared_large_n_self_only_impl,
    compute_leaf_p2p_accelerations_target_block_pairs_only,
)
from ._scatter import (
    _scatter_contributions,
    _scatter_scalar_contributions,
    _scatter_scalars_with_schedule,
    _scatter_vectors_with_schedule,
)
from ._schedules import (  # noqa: F401  -- public re-export surface, see below
    _prepare_leaf_data,
    _prepare_leaf_data_from_groups,
    prepare_bucketed_scatter_schedules,
    prepare_bucketed_scatter_schedules_from_groups,
    prepare_leaf_neighbor_pairs,
)
from .grad import (
    _leafpair_accel_analytic_vjp,
    _pair_accel_cvjp,
    _pair_accel_masked_accels,
    build_leafpair_reverse_tiers,
)

__all__ = [
    "RadixFastLanePerfCounters",
    "build_leafpair_reverse_tiers",
    "collect_radix_fast_lane_counters",
    "compute_leaf_p2p_accelerations",
    "compute_leaf_p2p_accelerations_large_n_accel_only",
    "compute_leaf_p2p_accelerations_target_block_pairs_only",
    "prepare_bucketed_scatter_schedules",
    "prepare_bucketed_scatter_schedules_from_groups",
]

# RE-EXPORTS. The names below are imported for other modules to reach through
# this one, and are unused *here*. Holding references makes that a fact the
# interpreter can see: `pyflakes` counts them as used, so whatever it still
# reports for this module is a real dead import rather than noise. A tuple of
# STRINGS would not do it, and neither does `# noqa` -- pyflakes ignores noqa,
# and isort hoists a trailing comment onto the whole import group, so a name that
# later goes unused inside that group is never flagged. Verified, 2026-08-18.
_REEXPORTS = (
    _leafpair_accel_analytic_vjp,
    _pair_accel_cvjp,
    _pair_accel_masked_accels,
)

_LARGE_N_NEARFIELD_DIAG_MODES = frozenset(("full", "self_only", "pairs_only", "zero"))


def _large_n_nearfield_diag_mode() -> str:
    return env_choice(
        "JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE", "full", _LARGE_N_NEARFIELD_DIAG_MODES
    )


# Unclamped by design: several near-field knobs use 0 as "unset, pick a default"
# (JACCPOT_NEARFIELD_PALLAS_NUM_WARPS, ..._TARGET_SUBTILE), so a minimum of 1
# would turn "auto" into a real, wrong value.
_env_flag = env_flag
_env_int = env_int


@dataclass(frozen=True)
class RadixFastLanePerfCounters:
    """Static-shape payload counters for radix fast-lane nearfield diagnostics.

    Derived from the payload's *shapes*, not from a run: every field counts the
    padded slot capacity, so masked-off slots are included. That is the point --
    these are what the lane will move regardless of occupancy -- but it means a
    sparsely occupied payload reports the same traffic as a full one, and none of
    these numbers is a measurement.

    Attributes
    ----------
    gather_bytes : int
        Bytes read gathering target and source positions and masses.
    scatter_bytes : int
        Bytes written scattering accelerations back to particle order.
    scatter_ops : int
        Number of scattered target slots, i.e. ``scatter_bytes`` before the
        per-element width is applied.
    target_batches : int
        Target-leaf batches the lane will scan, ``ceil`` of the leaf count over
        the payload's ``batch_tile_t``.
    source_slot_tiles : int
        Source-slot tiles per target batch, ``ceil`` over ``batch_tile_s``.
    """

    gather_bytes: int
    scatter_bytes: int
    scatter_ops: int
    target_batches: int
    source_slot_tiles: int


def collect_radix_fast_lane_counters(
    *,
    payload: Any,
    positions_dtype: jnp.dtype,
    masses_dtype: jnp.dtype,
    accelerations_dtype: Optional[jnp.dtype] = None,
) -> RadixFastLanePerfCounters:
    """Estimate deterministic payload gather/scatter costs for one evaluation.

    Host-side and shape-only: it calls ``int()`` on array sizes, so it must not be
    used on tracers and cannot be called from inside a jitted function. The
    payload arrays themselves are never read, only measured.

    Parameters
    ----------
    payload : Any
        A ``RadixFastNearfieldPayload``. Untyped to keep the runtime layer out of
        this module's imports. ``target_particle_ids`` and
        ``source_particle_ids`` are required; ``batch_tile_t`` and
        ``batch_tile_s`` are read through ``getattr`` defaulting to 1, so a
        payload lacking them yields one batch per leaf rather than an error.
    positions_dtype : jnp.dtype
        Dtype the positions will be gathered in; both target and source sides are
        assumed to share it.
    masses_dtype : jnp.dtype
        Dtype the masses will be gathered in.
    accelerations_dtype : Optional[jnp.dtype]
        Dtype accelerations will be scattered in. ``None`` (the default) reuses
        ``positions_dtype``, which is the usual case; pass it explicitly for a
        mixed-precision lane.

    Returns
    -------
    RadixFastLanePerfCounters
        Padded-capacity byte and tile counts. See the class docstring for why
        these are an upper bound rather than an observation.
    """

    if accelerations_dtype is None:
        accelerations_dtype = positions_dtype

    target_particle_ids = jnp.asarray(payload.target_particle_ids, dtype=INDEX_DTYPE)
    source_particle_ids = jnp.asarray(payload.source_particle_ids, dtype=INDEX_DTYPE)

    target_slot_count = int(target_particle_ids.size)
    source_slot_count = int(source_particle_ids.size)

    pos_itemsize = int(jnp.dtype(positions_dtype).itemsize)
    mass_itemsize = int(jnp.dtype(masses_dtype).itemsize)
    accel_itemsize = int(jnp.dtype(accelerations_dtype).itemsize)

    gather_bytes = int(
        target_slot_count * (3 * pos_itemsize + mass_itemsize)
        + source_slot_count * (3 * pos_itemsize + mass_itemsize)
    )
    scatter_bytes = int(target_slot_count * 3 * accel_itemsize)
    scatter_ops = int(target_slot_count)

    num_target_leaves = (
        int(target_particle_ids.shape[0]) if target_particle_ids.ndim >= 1 else 0
    )
    num_source_slots = (
        int(source_particle_ids.shape[1]) if source_particle_ids.ndim >= 2 else 0
    )
    target_batch_size = max(1, int(getattr(payload, "batch_tile_t", 1)))
    source_slot_tile_size = max(1, int(getattr(payload, "batch_tile_s", 1)))
    target_batches = (num_target_leaves + target_batch_size - 1) // target_batch_size
    source_slot_tiles = (
        num_source_slots + source_slot_tile_size - 1
    ) // source_slot_tile_size

    return RadixFastLanePerfCounters(
        gather_bytes=gather_bytes,
        scatter_bytes=scatter_bytes,
        scatter_ops=scatter_ops,
        target_batches=int(target_batches),
        source_slot_tiles=int(source_slot_tiles),
    )


@partial(
    jax.jit,
    static_argnums=(12,),
    static_argnames=(
        "return_potential",
        "collect_neighbor_pairs",
        "nearfield_mode",
        "edge_chunk_size",
        "use_precomputed_scatter",
    ),
)
def _compute_leaf_p2p_impl(
    node_ranges: Array,
    leaf_nodes: Array,
    offsets: Array,
    neighbors: Array,
    positions: Array,
    masses: Array,
    target_leaf_ids: Array,
    source_leaf_ids: Array,
    valid_pairs: Array,
    precomputed_chunk_sort_indices: Array,
    precomputed_chunk_group_ids: Array,
    precomputed_chunk_unique_indices: Array,
    max_leaf_size: int,
    *,
    G: Union[float, Array],
    softening_sq: Array,
    return_potential: bool,
    collect_neighbor_pairs: bool,
    nearfield_mode: str,
    edge_chunk_size: int,
    use_precomputed_scatter: bool,
) -> Union[
    Array,
    Tuple[Array, Array],
    Tuple[Array, Array, Array],
    Tuple[Array, Array, Array, Array],
]:
    """JIT near-field kernel over leaf-neighbor particle interactions.

    Pads the per-leaf particle data with :func:`_prepare_leaf_data` and then
    defers to :func:`_compute_leaf_p2p_from_prepared_leaf_data_impl`, which is
    the same kernel taking that padded data as arguments. The two used to be
    separate 460-line copies that were 94.4% line-identical and had already
    drifted apart; this follows the delegation precedent
    :func:`_compute_leaf_p2p_prepared_large_n_accel_only_impl` already sets in
    this module.

    ``max_leaf_size`` is recovered inside the callee as
    ``leaf_particle_idx.shape[1]``, which :func:`_prepare_leaf_data` sets to
    exactly this argument -- so it stays a compile-time constant on both sides
    of the call and neither side gains a traced shape.

    Parameters
    ----------
    node_ranges : Array
        Per-node ``[start, end]`` particle spans, ``[num_nodes, 2]``.
    leaf_nodes : Array
        Node indices of the leaves, ``[num_leaves]``.
    offsets : Array
        CSR row offsets into ``neighbors``, ``[num_leaves + 1]``.
    neighbors : Array
        Flat leaf-neighbour edge list, ``[num_edges]``.
    positions : Array
        Particle positions ``[N, 3]`` in Morton order.
    masses : Array
        Particle masses ``[N]`` in the same order.
    target_leaf_ids : Array
        Per-edge target leaf index, ``[num_edges]``.
    source_leaf_ids : Array
        Per-edge source leaf index, ``[num_edges]``.
    valid_pairs : Array
        Per-edge validity mask, ``[num_edges]``; padded edges are ``False``.
    precomputed_chunk_sort_indices : Array
        Scatter-schedule sort permutation, or an empty array when
        ``use_precomputed_scatter`` is ``False``.
    precomputed_chunk_group_ids : Array
        Scatter-schedule group ids, same convention.
    precomputed_chunk_unique_indices : Array
        Scatter-schedule unique-target indices, same convention.
    max_leaf_size : int
        Padded leaf width. Static under ``jit`` (``static_argnums=(12,)``).
    G : Union[float, Array]
        Gravitational constant.
    softening_sq : Array
        Squared Plummer softening length.
    return_potential : bool
        Also return per-particle potentials. Static under ``jit``.
    collect_neighbor_pairs : bool
        Also return the realised ``(target, source)`` leaf pairs and their
        count. Static under ``jit``.
    nearfield_mode : str
        ``"baseline"`` or ``"bucketed"``. Static under ``jit``.
    edge_chunk_size : int
        Edge-chunk width for ``"bucketed"``. Static under ``jit``.
    use_precomputed_scatter : bool
        Use the precomputed scatter schedules rather than deriving them.
        Static under ``jit``.

    Returns
    -------
    Union[Array, Tuple[Array, Array], Tuple[Array, Array, Array], Tuple[Array, Array, Array, Array]]
        Accelerations ``[N, 3]``, followed by potentials ``[N]`` when
        ``return_potential``, followed by the neighbour-pair array and count
        when ``collect_neighbor_pairs``. A one-element result is unwrapped.
    """
    (
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
    ) = _prepare_leaf_data(
        node_ranges,
        leaf_nodes,
        positions,
        masses,
        max_leaf_size=max_leaf_size,
    )
    return _compute_leaf_p2p_from_prepared_leaf_data_impl(
        offsets,
        neighbors,
        positions,
        target_leaf_ids,
        source_leaf_ids,
        valid_pairs,
        precomputed_chunk_sort_indices,
        precomputed_chunk_group_ids,
        precomputed_chunk_unique_indices,
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
        G=G,
        softening_sq=softening_sq,
        return_potential=return_potential,
        collect_neighbor_pairs=collect_neighbor_pairs,
        nearfield_mode=nearfield_mode,
        edge_chunk_size=edge_chunk_size,
        use_precomputed_scatter=use_precomputed_scatter,
    )


@partial(
    jax.jit,
    static_argnames=(
        "return_potential",
        "collect_neighbor_pairs",
        "nearfield_mode",
        "edge_chunk_size",
        "use_precomputed_scatter",
    ),
)
def _compute_leaf_p2p_from_prepared_leaf_data_impl(
    offsets: Array,
    neighbors: Array,
    positions: Array,
    target_leaf_ids: Array,
    source_leaf_ids: Array,
    valid_pairs: Array,
    precomputed_chunk_sort_indices: Array,
    precomputed_chunk_group_ids: Array,
    precomputed_chunk_unique_indices: Array,
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    *,
    G: Union[float, Array],
    softening_sq: Array,
    return_potential: bool,
    collect_neighbor_pairs: bool,
    nearfield_mode: str,
    edge_chunk_size: int,
    use_precomputed_scatter: bool,
) -> Union[
    Array,
    Tuple[Array, Array],
    Tuple[Array, Array, Array],
    Tuple[Array, Array, Array, Array],
]:
    """JIT near-field kernel over explicit per-leaf particle groups.

    The near-field edge-list kernel proper. :func:`_compute_leaf_p2p_impl` is a
    thin wrapper that derives the padded leaf data and delegates here (Tier 1.2 --
    the two used to be 94.4% line-identical copies), so this is the single
    implementation of both the ``"baseline"`` and ``"bucketed"`` traversals.

    ``max_leaf_size`` is **not** a parameter: it is recovered as
    ``leaf_particle_idx.shape[1]``, which keeps it a compile-time constant without
    a second static argument to keep in sync.

    Differentiable in ``positions``, ``leaf_positions`` and ``leaf_masses``.

    Parameters
    ----------
    offsets : Array
        CSR row offsets into ``neighbors``, ``[num_leaves + 1]``.
    neighbors : Array
        Flat leaf-neighbour edge list, ``[num_edges]``. Its length alone selects
        whether any pair work happens.
    positions : Array
        Particle positions ``[N, 3]``; also fixes the output shape.
    target_leaf_ids : Array
        Per-edge target leaf index, ``[num_edges]``.
    source_leaf_ids : Array
        Per-edge source leaf index, ``[num_edges]``.
    valid_pairs : Array
        Per-edge validity ``[num_edges]``; padded edges contribute exactly zero.
    precomputed_chunk_sort_indices : Array
        Scatter-schedule sort permutation, or an empty array when
        ``use_precomputed_scatter`` is ``False``.
    precomputed_chunk_group_ids : Array
        Scatter-schedule group ids, same convention.
    precomputed_chunk_unique_indices : Array
        Scatter-schedule unique-target indices, same convention.
    leaf_positions : Array
        Padded per-leaf positions ``[num_leaves, W, 3]``.
    leaf_masses : Array
        Padded per-leaf masses ``[num_leaves, W]``.
    leaf_mask : Array
        Padded per-leaf validity ``[num_leaves, W]``.
    leaf_particle_idx : Array
        Particle index per padded slot ``[num_leaves, W]``; its second axis is
        where ``max_leaf_size`` comes from.
    G : Union[float, Array]
        Gravitational constant.
    softening_sq : Array
        Squared Plummer softening length.
    return_potential : bool
        Also return per-particle potentials. Static under ``jit``.
    collect_neighbor_pairs : bool
        Also return the realised ``(target, source)`` leaf pairs and their count.
        Static under ``jit``.
    nearfield_mode : str
        ``"baseline"`` (per-pair scan) or ``"bucketed"`` (edge-chunked). Static
        under ``jit``. The two visit the same pairs, so they agree to
        reassociation -- measured 4.2e-16 apart, one ulp.
    edge_chunk_size : int
        Edge-chunk width for ``"bucketed"``. Static under ``jit``; batching only.
    use_precomputed_scatter : bool
        Consume the precomputed scatter schedules instead of deriving them. Static
        under ``jit``.

    Returns
    -------
    Union[Array, Tuple[Array, Array], Tuple[Array, Array, Array], Tuple[Array, Array, Array, Array]]
        Accelerations ``[N, 3]``, followed by potentials ``[N]`` when
        ``return_potential``, followed by the neighbour-pair array and count when
        ``collect_neighbor_pairs``. A one-element result is unwrapped.

    Raises
    ------
    ValueError
        If ``nearfield_mode`` is not ``"baseline"`` or ``"bucketed"``, or
        ``edge_chunk_size`` is not positive.
    """
    dtype = positions.dtype
    g_const = jnp.asarray(G, dtype=dtype)
    soft_sq = softening_sq
    max_leaf_size = int(leaf_particle_idx.shape[1])

    accelerations = jnp.zeros_like(positions)
    if return_potential:
        potentials = jnp.zeros((positions.shape[0],), dtype=dtype)
    else:
        potentials = None

    self_accel, self_potential = _self_contributions(
        leaf_positions,
        leaf_masses,
        leaf_mask,
        softening_sq=soft_sq,
        G=g_const,
        compute_potential=return_potential,
    )
    accelerations = _scatter_contributions(
        accelerations,
        leaf_particle_idx,
        self_accel,
        leaf_mask,
    )
    if return_potential and self_potential is not None and potentials is not None:
        potentials = _scatter_scalar_contributions(
            potentials,
            leaf_particle_idx,
            self_potential,
            leaf_mask,
        )

    inputs = (target_leaf_ids, source_leaf_ids, valid_pairs)

    if neighbors.shape[0] > 0:
        mode = str(nearfield_mode).strip().lower()
        if mode not in ("baseline", "bucketed"):
            raise ValueError("nearfield_mode must be 'baseline' or 'bucketed'")
        if mode == "bucketed":
            chunk = int(edge_chunk_size)
            if chunk <= 0:
                raise ValueError("edge_chunk_size must be positive")
            starts = jnp.arange(0, neighbors.shape[0], chunk, dtype=INDEX_DTYPE)
            chunk_offsets = jnp.arange(chunk, dtype=INDEX_DTYPE)
            chunk_flat_size = int(chunk * max_leaf_size)

            if return_potential and potentials is not None:
                if use_precomputed_scatter:

                    def _chunk_body(carry, data):
                        acc, pot = carry
                        start, sort_idx, group_ids, unique_indices = data
                        edge_idx = start + chunk_offsets
                        in_range = edge_idx < neighbors.shape[0]
                        safe_edge_idx = jnp.where(in_range, edge_idx, 0)
                        valid_edge = in_range & valid_pairs[safe_edge_idx]

                        def _compute(args):
                            acc_in, pot_in = args
                            tgt_leaf = target_leaf_ids[safe_edge_idx]
                            src_leaf = source_leaf_ids[safe_edge_idx]
                            tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                            src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                            tgt_pos = leaf_positions[tgt_leaf_local]
                            tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                            src_pos = leaf_positions[src_leaf_local]
                            src_mass = leaf_masses[src_leaf_local]
                            src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                            pair_acc, pair_pot = _pair_contributions_batched(
                                tgt_pos,
                                tgt_mask,
                                src_pos,
                                src_mass,
                                src_mask,
                                softening_sq=soft_sq,
                                G=g_const,
                                compute_potential=True,
                            )
                            acc_out = _scatter_vectors_with_schedule(
                                acc_in,
                                pair_acc,
                                tgt_mask,
                                sort_idx,
                                group_ids,
                                unique_indices,
                            )
                            pot_out = _scatter_scalars_with_schedule(
                                pot_in,
                                pair_pot,
                                tgt_mask,
                                sort_idx,
                                group_ids,
                                unique_indices,
                            )
                            return acc_out, pot_out

                        return (
                            lax.cond(
                                jnp.any(valid_edge),
                                _compute,
                                lambda args: args,
                                (acc, pot),
                            ),
                            None,
                        )

                    (accelerations, potentials), _ = lax.scan(
                        _chunk_body,
                        (accelerations, potentials),
                        (
                            starts,
                            precomputed_chunk_sort_indices[:, :chunk_flat_size],
                            precomputed_chunk_group_ids[:, :chunk_flat_size],
                            precomputed_chunk_unique_indices[:, :chunk_flat_size],
                        ),
                    )
                else:

                    def _chunk_body(carry, start):
                        acc, pot = carry
                        edge_idx = start + chunk_offsets
                        in_range = edge_idx < neighbors.shape[0]
                        safe_edge_idx = jnp.where(in_range, edge_idx, 0)
                        valid_edge = in_range & valid_pairs[safe_edge_idx]

                        def _compute(args):
                            acc_in, pot_in = args
                            tgt_leaf = target_leaf_ids[safe_edge_idx]
                            src_leaf = source_leaf_ids[safe_edge_idx]
                            tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                            src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                            tgt_pos = leaf_positions[tgt_leaf_local]
                            tgt_mask = leaf_mask[tgt_leaf_local] & valid_edge[:, None]
                            tgt_ids = leaf_particle_idx[tgt_leaf_local]
                            src_pos = leaf_positions[src_leaf_local]
                            src_mass = leaf_masses[src_leaf_local]
                            src_mask = leaf_mask[src_leaf_local] & valid_edge[:, None]

                            pair_acc, pair_pot = _pair_contributions_batched(
                                tgt_pos,
                                tgt_mask,
                                src_pos,
                                src_mass,
                                src_mask,
                                softening_sq=soft_sq,
                                G=g_const,
                                compute_potential=True,
                            )
                            acc_out = _scatter_contributions(
                                acc_in,
                                tgt_ids,
                                pair_acc,
                                tgt_mask,
                            )
                            pot_out = _scatter_scalar_contributions(
                                pot_in,
                                tgt_ids,
                                pair_pot,
                                tgt_mask,
                            )
                            return acc_out, pot_out

                        return (
                            lax.cond(
                                jnp.any(valid_edge),
                                _compute,
                                lambda args: args,
                                (acc, pot),
                            ),
                            None,
                        )

                    (accelerations, potentials), _ = lax.scan(
                        _chunk_body,
                        (accelerations, potentials),
                        starts,
                    )
            else:
                if use_precomputed_scatter:

                    def _chunk_body(acc, data):
                        start, sort_idx, group_ids, unique_indices = data
                        edge_idx = start + chunk_offsets
                        in_range = edge_idx < neighbors.shape[0]
                        safe_edge_idx = jnp.where(in_range, edge_idx, 0)
                        valid_edge = in_range & valid_pairs[safe_edge_idx]

                        def _compute(acc_in):
                            tgt_leaf = target_leaf_ids[safe_edge_idx]
                            src_leaf = source_leaf_ids[safe_edge_idx]
                            tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                            src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                            # Gather + pair evaluation, rematerialized: the
                            # composite is the dominant reverse-pass residual at
                            # galaxy N (see ``_bucketed_chunk_pair_accels``).
                            pair_acc, tgt_mask = _bucketed_chunk_pair_accels_remat(
                                leaf_positions,
                                leaf_masses,
                                leaf_mask,
                                tgt_leaf_local,
                                src_leaf_local,
                                valid_edge,
                                soft_sq,
                                g_const,
                            )
                            return _scatter_vectors_with_schedule(
                                acc_in,
                                pair_acc,
                                tgt_mask,
                                sort_idx,
                                group_ids,
                                unique_indices,
                            )

                        return (
                            lax.cond(
                                jnp.any(valid_edge),
                                _compute,
                                lambda acc_in: acc_in,
                                acc,
                            ),
                            None,
                        )

                    accelerations, _ = lax.scan(
                        _chunk_body,
                        accelerations,
                        (
                            starts,
                            precomputed_chunk_sort_indices[:, :chunk_flat_size],
                            precomputed_chunk_group_ids[:, :chunk_flat_size],
                            precomputed_chunk_unique_indices[:, :chunk_flat_size],
                        ),
                    )
                else:

                    def _chunk_body(acc, start):
                        edge_idx = start + chunk_offsets
                        in_range = edge_idx < neighbors.shape[0]
                        safe_edge_idx = jnp.where(in_range, edge_idx, 0)
                        valid_edge = in_range & valid_pairs[safe_edge_idx]

                        def _compute(acc_in):
                            tgt_leaf = target_leaf_ids[safe_edge_idx]
                            src_leaf = source_leaf_ids[safe_edge_idx]
                            tgt_leaf_local = jnp.where(valid_edge, tgt_leaf, 0)
                            src_leaf_local = jnp.where(valid_edge, src_leaf, 0)

                            tgt_ids = leaf_particle_idx[tgt_leaf_local]
                            # Gather + pair evaluation, rematerialized: the
                            # composite is the dominant reverse-pass residual at
                            # galaxy N (see ``_bucketed_chunk_pair_accels``).
                            pair_acc, tgt_mask = _bucketed_chunk_pair_accels_remat(
                                leaf_positions,
                                leaf_masses,
                                leaf_mask,
                                tgt_leaf_local,
                                src_leaf_local,
                                valid_edge,
                                soft_sq,
                                g_const,
                            )
                            return _scatter_contributions(
                                acc_in,
                                tgt_ids,
                                pair_acc,
                                tgt_mask,
                            )

                        return (
                            lax.cond(
                                jnp.any(valid_edge),
                                _compute,
                                lambda acc_in: acc_in,
                                acc,
                            ),
                            None,
                        )

                    accelerations, _ = lax.scan(
                        _chunk_body,
                        accelerations,
                        starts,
                    )
        elif return_potential and potentials is not None:

            def _edge_body(carry, data):
                acc, pot = carry
                tgt_idx, src_idx, is_valid = data

                def true_branch(
                    args: tuple[Array, Array, Array, Array],
                ) -> tuple[Array, Array]:
                    acc_state, pot_state, tgt, src = args
                    target_pos = leaf_positions[tgt]
                    target_mask = leaf_mask[tgt]
                    target_ids = leaf_particle_idx[tgt]

                    source_pos = leaf_positions[src]
                    source_mass = leaf_masses[src]
                    source_mask = leaf_mask[src]

                    pair_accel, pair_pot = _pair_contributions(
                        target_pos,
                        target_mask,
                        source_pos,
                        source_mass,
                        source_mask,
                        softening_sq=soft_sq,
                        G=g_const,
                        compute_potential=True,
                    )

                    masked_acc = jnp.where(target_mask[:, None], pair_accel, 0.0)
                    masked_pot = jnp.where(target_mask, pair_pot, 0.0)

                    acc_state = acc_state.at[target_ids].add(masked_acc)
                    pot_state = pot_state.at[target_ids].add(masked_pot)
                    return acc_state, pot_state

                def false_branch(
                    args: tuple[Array, Array, Array, Array],
                ) -> tuple[Array, Array]:
                    acc_state, pot_state, *_ = args
                    return acc_state, pot_state

                updated = lax.cond(
                    is_valid,
                    true_branch,
                    false_branch,
                    (acc, pot, tgt_idx, src_idx),
                )
                return updated, None

            (accelerations, potentials), _ = lax.scan(
                _edge_body,
                (accelerations, potentials),
                inputs,
            )
        else:

            def _edge_body(acc, data):
                tgt_idx, src_idx, is_valid = data

                def true_branch(args: tuple[Array, Array, Array]) -> Array:
                    acc_state, tgt, src = args
                    target_pos = leaf_positions[tgt]
                    target_mask = leaf_mask[tgt]
                    target_ids = leaf_particle_idx[tgt]

                    source_pos = leaf_positions[src]
                    source_mass = leaf_masses[src]
                    source_mask = leaf_mask[src]

                    pair_accel, _ = _pair_contributions(
                        target_pos,
                        target_mask,
                        source_pos,
                        source_mass,
                        source_mask,
                        softening_sq=soft_sq,
                        G=g_const,
                        compute_potential=False,
                    )

                    masked_acc = jnp.where(target_mask[:, None], pair_accel, 0.0)
                    acc_state = acc_state.at[target_ids].add(masked_acc)
                    return acc_state

                def false_branch(args: tuple[Array, Array, Array]) -> Array:
                    acc_state, *_ = args
                    return acc_state

                updated_acc = lax.cond(
                    is_valid,
                    true_branch,
                    false_branch,
                    (acc, tgt_idx, src_idx),
                )
                return updated_acc, None

            accelerations, _ = lax.scan(
                _edge_body,
                accelerations,
                inputs,
            )

    neighbor_pairs = jnp.zeros((0, 2), dtype=INDEX_DTYPE)
    pair_count = as_index(0)
    if collect_neighbor_pairs:
        max_pairs = neighbors.shape[0]
        pair_buffer = jnp.zeros((max_pairs, 2), dtype=INDEX_DTYPE)

        def _pair_body(idx, state):
            ptr, buf = state

            def _add_pair(args):
                ptr_val, buf_val = args
                pair = jnp.stack([target_leaf_ids[idx], source_leaf_ids[idx]], axis=0)
                buf_val = buf_val.at[ptr_val].set(pair)
                return ptr_val + as_index(1), buf_val

            return lax.cond(
                valid_pairs[idx],
                _add_pair,
                lambda args: args,
                (ptr, buf),
            )

        pair_count, pair_buffer = lax.fori_loop(
            0,
            max_pairs,
            _pair_body,
            (as_index(0), pair_buffer),
        )
        neighbor_pairs = pair_buffer

    outputs = (accelerations,)
    if return_potential and potentials is not None:
        outputs += (potentials,)
    if collect_neighbor_pairs:
        outputs += (neighbor_pairs, pair_count)

    if len(outputs) == 1:
        return outputs[0]
    return outputs


@jaxtyped(typechecker=beartype)
def compute_leaf_p2p_accelerations(
    tree: Tree,
    neighbor_list: NodeNeighborList,
    positions_sorted: Float[Array, "n 3"],
    masses_sorted: Float[Array, "n"],
    *,
    G: Union[float, Array] = 1.0,
    softening: float = 0.0,
    max_leaf_size: Optional[int] = None,
    return_potential: bool = False,
    collect_neighbor_pairs: bool = False,
    nearfield_mode: str = "baseline",
    edge_chunk_size: int = 256,
    precomputed_target_leaf_ids: Optional[Int[Array, "pairs"]] = None,
    precomputed_source_leaf_ids: Optional[Int[Array, "pairs"]] = None,
    precomputed_valid_pairs: Optional[Bool[Array, "pairs"]] = None,
    precomputed_chunk_sort_indices: Optional[Int[Array, "chunks chunkflat"]] = None,
    precomputed_chunk_group_ids: Optional[Int[Array, "chunks chunkflat"]] = None,
    precomputed_chunk_unique_indices: Optional[Int[Array, "chunks chunkflat"]] = None,
    # NOT `Int[Array, "nodes 2"]`, and that is a finding rather than caution: the
    # leading axis is CALLER-DEPENDENT. The single-GPU path passes
    # `tree.node_ranges`, one row per tree node (5, 7, ... 187 observed across 63
    # captured calls); `distributed/fmm.py:1565` passes `jnp.zeros((u_leaves+1, 2))`.
    # Naming that axis would bind it to whichever caller ran first and break the
    # other -- and the distributed lane cannot be exercised here, because every
    # test in tests/distributed/ skips below 2 devices. Only the trailing `2` is
    # invariant across callers, so only the trailing `2` is asserted.
    node_ranges_override: Optional[Int[Array, "_ 2"]] = None,
    leaf_nodes_override: Optional[Int[Array, "leaves"]] = None,
    neighbor_offsets_override: Optional[Int[Array, "leaves+1"]] = None,
    neighbor_indices_override: Optional[Int[Array, "edges"]] = None,
    neighbor_counts_override: Optional[Int[Array, "leaves"]] = None,
    leaf_particle_indices_override: Optional[Int[Array, "leaves w"]] = None,
    leaf_particle_mask_override: Optional[Bool[Array, "leaves w"]] = None,
) -> Union[
    Array,
    Tuple[Array, Array],
    Tuple[Array, Array, Array],
    Tuple[Array, Array, Array, Array],
]:
    """Compute near-field contributions for all leaf particle pairs.

    Parameters
    ----------
    tree : Tree
        Radix tree whose ``node_ranges`` map leaves to particle spans.
    neighbor_list : NodeNeighborList
        Precomputed leaf-neighbour CSR metadata (``leaf_indices``, ``offsets``,
        ``neighbors``, ``counts``). Each ``*_override`` below replaces exactly
        one of these fields.
    positions_sorted : Float[Array, 'n 3']
        Particle positions ``[N, 3]`` in Morton order.
    masses_sorted : Float[Array, 'n']
        Particle masses ``[N]`` in Morton order.
    G : Union[float, Array]
        Gravitational constant, applied as a plain multiplier.
    softening : float
        Plummer softening length. Squared host-side via ``float(softening)``, so
        it must be a concrete Python float, not a tracer.
    max_leaf_size : Optional[int]
        Static bound on per-leaf particle count. **Required under ``jit``**: when
        it is ``None`` the code reads the true maximum with ``.item()``, which
        only works on concrete values (see ``Raises``).
    return_potential : bool
        Also accumulate potentials. Static under ``jit`` -- it changes the number
        of returned arrays.
    collect_neighbor_pairs : bool
        Also return the processed ``(target, source)`` leaf-pair buffer. Static
        under ``jit``, and it additionally flips the internal edge ordering
        (``sort_by_source``), so enabling it is not purely additive.
    nearfield_mode : str
        Either ``"baseline"`` or ``"bucketed"``; static under ``jit``.
        ``"bucketed"`` is the minimum-memory large-N GPU path and preserves
        target-local edge order for scatter locality. The two modes are meant to
        produce the same accelerations.
    edge_chunk_size : int
        Chunk width for the bucketed edge scan. Static under ``jit``; a
        performance knob, not a numerical one.
    precomputed_target_leaf_ids : Optional[Int[Array, 'pairs']]
        Leaf-pair schedule buffer: target leaf id per edge. Supplied together
        with the other ``precomputed_*`` arrays by prepared state to skip
        re-deriving the schedule. Must match the neighbour-list edge order the
        rest of the schedule was built against.
    precomputed_source_leaf_ids : Optional[Int[Array, 'pairs']]
        Source leaf id per edge, same ordering contract.
    precomputed_valid_pairs : Optional[Bool[Array, 'pairs']]
        Boolean mask marking which padded edge slots are real.
    precomputed_chunk_sort_indices : Optional[Int[Array, 'chunks chunkflat']]
        Scatter-schedule permutation for the chunked accumulation.
    precomputed_chunk_group_ids : Optional[Int[Array, 'chunks chunkflat']]
        Chunk group id per sorted edge.
    precomputed_chunk_unique_indices : Optional[Int[Array, 'chunks chunkflat']]
        Unique-chunk boundary indices. This and the previous two are consumed as
        a set: the precomputed scatter path only engages when all three are given.
    node_ranges_override : Optional[Int[Array, '_ 2']]
        Replaces ``tree.node_ranges``.
    leaf_nodes_override : Optional[Int[Array, 'leaves']]
        Replaces ``neighbor_list.leaf_indices``.
    neighbor_offsets_override : Optional[Int[Array, 'leaves+1']]
        Replaces ``neighbor_list.offsets``.
    neighbor_indices_override : Optional[Int[Array, 'edges']]
        Replaces ``neighbor_list.neighbors``.
    neighbor_counts_override : Optional[Int[Array, 'leaves']]
        Replaces ``neighbor_list.counts``.
    leaf_particle_indices_override : Optional[Int[Array, 'leaves w']]
        Explicit per-leaf particle index table ``[num_leaves, max_leaf_size]``.
        Supplying it *sets* ``max_leaf_size`` from its second axis, overriding
        the argument.
    leaf_particle_mask_override : Optional[Bool[Array, 'leaves w']]
        Validity mask matching ``leaf_particle_indices_override``.

    Returns
    -------
    Union[Array, Tuple[Array, Array], Tuple[Array, Array, Array], Tuple[Array, Array, Array, Array]]
        Accelerations ``[N, 3]`` in Morton order, returned bare when neither flag
        is set. ``return_potential`` appends potentials ``[N]``;
        ``collect_neighbor_pairs`` appends the full ``(target, source)`` pair
        buffer and then the scalar count of valid entries (slice with
        ``neighbor_pairs[:neighbor_pair_count]``). With both flags the order is
        ``(accelerations, potentials, neighbor_pairs, pair_count)``.

    Raises
    ------
    ValueError
        If ``max_leaf_size`` is ``None`` while tracing. The bound is otherwise
        read from the data with ``.item()``; under a tracer that raises
        ``TypeError``, which is re-raised as this ``ValueError``. It is the one
        place in this function where a host sync is attempted deliberately, and
        it is the reason ``max_leaf_size`` must be passed by every jitted caller.

    Notes
    -----
    Differentiable in ``positions_sorted``, ``masses_sorted``, and ``G``. Not
    differentiable in ``softening`` (pulled to the host) or in any of the
    integer index/topology arrays.

    Every extent here is padded to a static bound -- leaf occupancy via
    ``max_leaf_size``, the edge list via the valid-pair mask. Padded slots must
    contribute exactly zero and must never form ``0 * inf``; see
    ``bench/audit_nearfield_padding.py``.

    An empty leaf set short-circuits to zeros of the right shape. Self-pairs and
    coincident particles are handled by the ``softening`` term; with
    ``softening == 0`` a coincident pair is a genuine singularity that this
    function does not guard.

    ``nearfield_mode="baseline"`` and ``"bucketed"`` agree **to a tolerance, not
    bit-exactly**, and that is the correct contract rather than a gap: the two
    deliberately differ in edge order (``sort_by_source``), which changes the
    order of the floating-point accumulation, so bit-equality is not expected.
    Asserted by
    ``tests/integration/test_fmm.py::test_nearfield_bucketed_matches_baseline``,
    parametrised over N in {96, 256} x {float32, float64} x
    ``edge_chunk_size`` in {64, 128}, at ``1e-6`` (fp32) and ``1e-13`` (fp64).
    Those bounds are derived from measured round-off (~1 eps in each dtype), not
    assumed, and the fp32/chunk-128 cases run in the CI smoke leg.

    The **float64 cases are the sharp instrument**: perturbing the bucketed path's
    softening by 3e-6 relative induces a ~1e-10 divergence, which fp64 catches
    against its 1e-13 bound and fp32 structurally cannot, since fp32 round-off is
    already ~1e-7.
    **The ``precomputed_*`` contract is shape-encoded, and partial sets are
    supported by design.** An earlier note here called this unvalidated; that was
    wrong. The mechanism: a ``None`` is converted to a zero-size sentinel by the
    caller in :mod:`jaccpot.runtime.fmm_evaluate`, and
    :mod:`jaccpot.runtime.kernels.core` then admits an array only if its leading
    dimension equals ``neighbor_list.neighbors.shape[0]`` (the scatter schedules
    must match ``(chunk_count, chunk_flat_size)``). Anything else -- including the
    sentinel -- is ignored and recomputed.

    Three groups fall back **independently**: the target/valid pair vectors, the
    source leaf ids, and the scatter schedules. That is what makes
    ``_prepare_bucketed_scatter_schedules_safe`` in
    :mod:`jaccpot.runtime.fmm_prepare` sound when it returns ``(None, None, None)``
    on int32 overflow, on exceeding the schedule cap, or on any exception, while
    the pair vectors stay populated. A 3-of-6 set is a normal production state,
    not an error.

    **The invariant that is *not* checked is edge order.** Precomputed vectors
    must be positionally aligned with ``neighbor_list.neighbors`` -- i.e. built
    with ``sort_by_source=False``. When ``precomputed_source_leaf_ids`` is absent
    the source ids are re-derived positionally as ``leaf_lookup[neighbors]``, so a
    source-sorted ``target_leaf_ids`` would be silently paired against unsorted
    sources. Both orderings have the same length, so no shape check can catch it,
    and the result is wrong forces with no error and no NaN. All in-repo producers
    pass ``sort_by_source=False``; see :func:`prepare_leaf_neighbor_pairs`, whose
    default is the *unsafe* value for this contract.
    """

    positions = jnp.asarray(positions_sorted)
    masses = jnp.asarray(masses_sorted)
    node_ranges = (
        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
        if node_ranges_override is None
        else jnp.asarray(node_ranges_override, dtype=INDEX_DTYPE)
    )

    leaf_nodes = (
        jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
        if leaf_nodes_override is None
        else jnp.asarray(leaf_nodes_override, dtype=INDEX_DTYPE)
    )
    offsets = (
        jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE)
        if neighbor_offsets_override is None
        else jnp.asarray(neighbor_offsets_override, dtype=INDEX_DTYPE)
    )
    neighbors = (
        jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE)
        if neighbor_indices_override is None
        else jnp.asarray(neighbor_indices_override, dtype=INDEX_DTYPE)
    )
    neighbor_counts = (
        jnp.asarray(neighbor_list.counts, dtype=INDEX_DTYPE)
        if neighbor_counts_override is None
        else jnp.asarray(neighbor_counts_override, dtype=INDEX_DTYPE)
    )

    if leaf_nodes.size == 0:
        zeros = jnp.zeros_like(positions)
        if return_potential:
            pot_zeros = jnp.zeros((positions.shape[0],), dtype=zeros.dtype)
            return zeros, pot_zeros
        return zeros

    if leaf_particle_indices_override is not None:
        explicit_leaf_particle_indices = jnp.asarray(
            leaf_particle_indices_override,
            dtype=INDEX_DTYPE,
        )
        explicit_leaf_particle_mask = (
            jnp.asarray(leaf_particle_mask_override, dtype=bool)
            if leaf_particle_mask_override is not None
            else jnp.ones_like(explicit_leaf_particle_indices, dtype=bool)
        )
        max_leaf_size = int(explicit_leaf_particle_indices.shape[1])
    else:
        explicit_leaf_particle_indices = None
        explicit_leaf_particle_mask = None
        if max_leaf_size is None:
            leaf_ranges = node_ranges[leaf_nodes]
            counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
            try:
                max_leaf_size = int(jnp.max(counts).item())
            except TypeError as exc:
                raise ValueError(
                    "max_leaf_size must be provided when tracing or JIT-compiling"
                ) from exc

    softening_sq = jnp.asarray(float(softening) ** 2, dtype=positions.dtype)

    use_precomputed_scatter = (
        precomputed_chunk_sort_indices is not None
        and precomputed_chunk_group_ids is not None
        and precomputed_chunk_unique_indices is not None
    )

    if precomputed_target_leaf_ids is None or precomputed_valid_pairs is None:
        # Precomputed scatter schedules are built against the neighbor-list edge
        # order used by prepared state. Re-derive leaf-pair vectors in that same
        # order so bucketed scans stay aligned with the schedule buffers.
        sort_by_source = not bool(collect_neighbor_pairs)
        if str(nearfield_mode).strip().lower() == "bucketed" and not bool(
            use_precomputed_scatter
        ):
            # The minimum-memory large-N GPU path uses direct scatter in the
            # bucketed loop, so preserving target-local edge order improves
            # output-update locality more than source-sorted gather locality.
            sort_by_source = False
        if use_precomputed_scatter:
            sort_by_source = False
        target_leaf_ids, source_leaf_ids, valid_pairs = prepare_leaf_neighbor_pairs(
            node_ranges,
            leaf_nodes,
            offsets,
            neighbors,
            sort_by_source=sort_by_source,
        )
    else:
        target_leaf_ids = jnp.asarray(precomputed_target_leaf_ids, dtype=INDEX_DTYPE)
        valid_pairs = jnp.asarray(precomputed_valid_pairs, dtype=bool)
        if precomputed_source_leaf_ids is None:
            # Compact prepared-state mode: derive source leaf ids directly from
            # neighbor edges while reusing precomputed target/valid buffers.
            total_nodes = node_ranges.shape[0]
            leaf_lookup = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
            leaf_lookup = leaf_lookup.at[leaf_nodes].set(
                jnp.arange(leaf_nodes.shape[0], dtype=INDEX_DTYPE)
            )
            source_leaf_ids = leaf_lookup[neighbors]
            valid_pairs = valid_pairs & (source_leaf_ids >= 0)
        else:
            source_leaf_ids = jnp.asarray(
                precomputed_source_leaf_ids,
                dtype=INDEX_DTYPE,
            )

    if use_precomputed_scatter:
        chunk_sort_indices = jnp.asarray(
            precomputed_chunk_sort_indices, dtype=INDEX_DTYPE
        )
        chunk_group_ids = jnp.asarray(precomputed_chunk_group_ids, dtype=INDEX_DTYPE)
        chunk_unique_indices = jnp.asarray(
            precomputed_chunk_unique_indices,
            dtype=INDEX_DTYPE,
        )
    else:
        chunk_sort_indices = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        chunk_group_ids = jnp.zeros((0, 0), dtype=INDEX_DTYPE)
        chunk_unique_indices = jnp.zeros((0, 0), dtype=INDEX_DTYPE)

    if explicit_leaf_particle_indices is None:
        return _compute_leaf_p2p_impl(
            node_ranges,
            leaf_nodes,
            offsets,
            neighbors,
            positions,
            masses,
            target_leaf_ids,
            source_leaf_ids,
            valid_pairs,
            chunk_sort_indices,
            chunk_group_ids,
            chunk_unique_indices,
            int(max_leaf_size),
            G=G,
            softening_sq=softening_sq,
            return_potential=return_potential,
            collect_neighbor_pairs=collect_neighbor_pairs,
            nearfield_mode=nearfield_mode,
            edge_chunk_size=int(edge_chunk_size),
            use_precomputed_scatter=use_precomputed_scatter,
        )

    (
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
    ) = _prepare_leaf_data_from_groups(
        explicit_leaf_particle_indices,
        explicit_leaf_particle_mask,
        positions,
        masses,
    )
    return _compute_leaf_p2p_from_prepared_leaf_data_impl(
        offsets,
        neighbors,
        positions,
        target_leaf_ids,
        source_leaf_ids,
        valid_pairs,
        chunk_sort_indices,
        chunk_group_ids,
        chunk_unique_indices,
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
        G=G,
        softening_sq=softening_sq,
        return_potential=return_potential,
        collect_neighbor_pairs=collect_neighbor_pairs,
        nearfield_mode=nearfield_mode,
        edge_chunk_size=int(edge_chunk_size),
        use_precomputed_scatter=use_precomputed_scatter,
    )


def compute_leaf_p2p_accelerations_large_n_accel_only(
    tree: Tree,
    neighbor_list: NodeNeighborList,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    G: Union[float, Array] = 1.0,
    softening: float = 0.0,
    edge_chunk_size: int = 256,
    precomputed_target_leaf_ids: Optional[Array] = None,
    precomputed_source_leaf_ids: Optional[Array] = None,
    precomputed_valid_pairs: Optional[Array] = None,
    leaf_particle_indices: Array,
    leaf_particle_mask: Optional[Array] = None,
    precomputed_target_block_leaf_ids: Optional[Array] = None,
    precomputed_target_block_source_leaf_ids: Optional[Array] = None,
    precomputed_target_block_valid_mask: Optional[Array] = None,
    precomputed_target_block_offsets: Optional[Array] = None,
    precomputed_target_block_source_leaf_ids_padded: Optional[Array] = None,
    precomputed_target_block_valid_mask_padded: Optional[Array] = None,
    delayed_scatter_chunks_per_superchunk: Optional[int] = None,
    chunk_scan_batch_size: Optional[int] = None,
    chunk_scan_unroll: Optional[int] = None,
    superchunk_scan_unroll: Optional[int] = None,
    sorted_scatter_hint: Optional[bool] = None,
    grouped_sorted_scatter: Optional[bool] = None,
    superchunk_target_reduce: Optional[bool] = None,
    disable_chunk_cond: Optional[bool] = None,
    target_leaf_batch_size: Optional[int] = None,
    target_block_tile_size: Optional[int] = None,
    target_block_tile_scan_unroll: Optional[int] = None,
    target_block_batch_scan_unroll: Optional[int] = None,
    target_block_overflow_fast_max_blocks: Optional[int] = None,
) -> Array:
    """Specialized accel-only bucketed near-field path for large-N prepared data.

    The large-N entry point: self block plus cross-leaf pairs, accelerations only.
    It selects among the kernels in :mod:`jaccpot.nearfield._large_n_blocks` from
    which ``precomputed_*`` artefacts the caller supplies, and **the choice is
    shape-encoded rather than validated** -- passing the padded target-block
    tensors selects the prepacked path, passing the run-length ones selects the
    offsets path, and passing neither falls back to the edge list. That is why
    there are 31 parameters and most are ``Optional``.

    Every ``None`` tuning knob means "resolve the default", not "off".

    Differentiable in ``positions_sorted`` and ``masses_sorted``.

    Parameters
    ----------
    tree : Tree
        Radix tree; supplies ``node_ranges`` and the leaf list.
    neighbor_list : NodeNeighborList
        Leaf-neighbour CSR metadata.
    positions_sorted : Array
        Particle positions ``[N, 3]`` in Morton order.
    masses_sorted : Array
        Particle masses ``[N]`` in the same order.
    G : Union[float, Array]
        Gravitational constant. Default ``1.0``.
    softening : float
        Plummer softening **length** (squared internally). Must be a concrete
        Python float, not a tracer. Default ``0.0``.
    edge_chunk_size : int
        Edge-chunk width for the edge-list path. Default ``256``.
    precomputed_target_leaf_ids : Optional[Array]
        Per-edge target leaf ids; derived from ``neighbor_list`` when ``None``.
    precomputed_source_leaf_ids : Optional[Array]
        Per-edge source leaf ids. **Must be positionally aligned with**
        ``neighbors`` -- see the warning on
        :func:`~jaccpot.nearfield._schedules.prepare_leaf_neighbor_pairs`: a
        source-sorted vector has the identical shape and produces wrong forces
        silently.
    precomputed_valid_pairs : Optional[Array]
        Per-edge validity, same convention.
    leaf_particle_indices : Array
        Explicit per-leaf particle membership ``[num_leaves, W]``. Required.
    leaf_particle_mask : Optional[Array]
        Validity for that table; derived when ``None``.
    precomputed_target_block_leaf_ids : Optional[Array]
        Target leaf id per block, for the target-owned paths.
    precomputed_target_block_source_leaf_ids : Optional[Array]
        Source leaf ids per block.
    precomputed_target_block_valid_mask : Optional[Array]
        Per-lane validity for those blocks.
    precomputed_target_block_offsets : Optional[Array]
        Run-length offsets per target leaf; selects the offsets kernel.
    precomputed_target_block_source_leaf_ids_padded : Optional[Array]
        Rectangular ``[leaf, block, lane]`` source ids; selects the prepacked
        kernel, which is the production large-N gradient path.
    precomputed_target_block_valid_mask_padded : Optional[Array]
        Per-lane validity for the padded layout.
    delayed_scatter_chunks_per_superchunk : Optional[int]
        Chunks per superchunk before the target reduction.
    chunk_scan_batch_size : Optional[int]
        Chunks per scan step.
    chunk_scan_unroll : Optional[int]
        Unroll factor for the chunk scan.
    superchunk_scan_unroll : Optional[int]
        Unroll factor for the superchunk scan.
    sorted_scatter_hint : Optional[bool]
        Promise that the scatter indices are sorted. A *promise*: it must match
        the data.
    grouped_sorted_scatter : Optional[bool]
        Use the segment-grouped scatter; same caveat.
    superchunk_target_reduce : Optional[bool]
        Reduce per target leaf within a superchunk before scattering. Changes the
        summation grouping, so it can move the last digits.
    disable_chunk_cond : Optional[bool]
        Skip the per-chunk ``lax.cond`` early-out.
    target_leaf_batch_size : Optional[int]
        Target leaves per scan step on the target-owned paths.
    target_block_tile_size : Optional[int]
        Source lanes per tile.
    target_block_tile_scan_unroll : Optional[int]
        Unroll factor for the tile scan.
    target_block_batch_scan_unroll : Optional[int]
        Unroll factor for the target-batch scan.
    target_block_overflow_fast_max_blocks : Optional[int]
        Cap above which the bounded tiled overflow kernel replaces the prepacked
        fast path. See
        :data:`~jaccpot.runtime.fmm_constants._NEARFIELD_TARGET_BLOCK_OVERFLOW_FAST_MAX_BLOCKS`.

    Returns
    -------
    Array
        Per-particle accelerations ``[N, 3]``. No potentials -- that is the point
        of the accel-only specialisation.
    """
    positions = jnp.asarray(positions_sorted)
    masses = jnp.asarray(masses_sorted)
    node_ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    offsets = jnp.asarray(neighbor_list.offsets, dtype=INDEX_DTYPE)
    neighbors = jnp.asarray(neighbor_list.neighbors, dtype=INDEX_DTYPE)

    if leaf_nodes.size == 0:
        return jnp.zeros_like(positions)

    explicit_leaf_particle_indices = jnp.asarray(
        leaf_particle_indices,
        dtype=INDEX_DTYPE,
    )
    explicit_leaf_particle_mask = (
        jnp.asarray(leaf_particle_mask, dtype=bool)
        if leaf_particle_mask is not None
        else jnp.ones_like(explicit_leaf_particle_indices, dtype=bool)
    )

    if precomputed_target_leaf_ids is None or precomputed_valid_pairs is None:
        target_leaf_ids, source_leaf_ids, valid_pairs = prepare_leaf_neighbor_pairs(
            node_ranges,
            leaf_nodes,
            offsets,
            neighbors,
            sort_by_source=False,
        )
    else:
        target_leaf_ids = jnp.asarray(precomputed_target_leaf_ids, dtype=INDEX_DTYPE)
        valid_pairs = jnp.asarray(precomputed_valid_pairs, dtype=bool)
        if precomputed_source_leaf_ids is None:
            total_nodes = node_ranges.shape[0]
            leaf_lookup = jnp.full((total_nodes,), -1, dtype=INDEX_DTYPE)
            leaf_lookup = leaf_lookup.at[leaf_nodes].set(
                jnp.arange(leaf_nodes.shape[0], dtype=INDEX_DTYPE)
            )
            source_leaf_ids = leaf_lookup[neighbors]
            valid_pairs = valid_pairs & (source_leaf_ids >= 0)
        else:
            source_leaf_ids = jnp.asarray(
                precomputed_source_leaf_ids,
                dtype=INDEX_DTYPE,
            )

    (
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
    ) = _prepare_leaf_data_from_groups(
        explicit_leaf_particle_indices,
        explicit_leaf_particle_mask,
        positions,
        masses,
    )

    softening_sq = jnp.asarray(float(softening) ** 2, dtype=positions.dtype)
    delayed_scatter_chunks_per_superchunk = int(
        1
        if delayed_scatter_chunks_per_superchunk is None
        else delayed_scatter_chunks_per_superchunk
    )
    chunk_scan_batch_size = int(
        1 if chunk_scan_batch_size is None else chunk_scan_batch_size
    )
    chunk_scan_unroll = int(1 if chunk_scan_unroll is None else chunk_scan_unroll)
    superchunk_scan_unroll = int(
        1 if superchunk_scan_unroll is None else superchunk_scan_unroll
    )
    sorted_scatter_hint = bool(
        False if sorted_scatter_hint is None else sorted_scatter_hint
    )
    grouped_sorted_scatter = bool(
        False if grouped_sorted_scatter is None else grouped_sorted_scatter
    )
    superchunk_target_reduce = bool(
        False if superchunk_target_reduce is None else superchunk_target_reduce
    )
    disable_chunk_cond = bool(
        True if disable_chunk_cond is None else disable_chunk_cond
    )
    target_leaf_batch_size = int(
        32 if target_leaf_batch_size is None else target_leaf_batch_size
    )
    target_block_tile_size = int(
        8 if target_block_tile_size is None else target_block_tile_size
    )
    target_block_tile_scan_unroll = int(
        1 if target_block_tile_scan_unroll is None else target_block_tile_scan_unroll
    )
    target_block_batch_scan_unroll = int(
        1 if target_block_batch_scan_unroll is None else target_block_batch_scan_unroll
    )
    target_block_overflow_fast_max_blocks = int(
        65536
        if target_block_overflow_fast_max_blocks is None
        else target_block_overflow_fast_max_blocks
    )
    use_target_blocks = (
        precomputed_target_block_offsets is not None
        and precomputed_target_block_leaf_ids is not None
        and precomputed_target_block_source_leaf_ids is not None
        and precomputed_target_block_valid_mask is not None
    )
    use_target_blocks_prepacked = (
        precomputed_target_block_source_leaf_ids_padded is not None
        and precomputed_target_block_valid_mask_padded is not None
    )

    if use_target_blocks_prepacked:
        self_acc = _compute_leaf_p2p_prepared_large_n_self_only_impl(
            positions,
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=G,
            softening_sq=softening_sq,
        )
        pair_acc = (
            _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl(
                positions,
                jnp.asarray(
                    precomputed_target_block_source_leaf_ids_padded,
                    dtype=INDEX_DTYPE,
                ),
                jnp.asarray(precomputed_target_block_valid_mask_padded, dtype=bool),
                leaf_positions,
                leaf_masses,
                leaf_mask,
                leaf_particle_idx,
                G=G,
                softening_sq=softening_sq,
                target_leaf_batch_size=target_leaf_batch_size,
                target_block_tile_size=target_block_tile_size,
                target_block_tile_scan_unroll=target_block_tile_scan_unroll,
                target_block_batch_scan_unroll=target_block_batch_scan_unroll,
            )
        )
        if use_target_blocks:
            overflow_block_count = int(
                precomputed_target_block_source_leaf_ids.shape[0]
            )
            overflow_pair_kernel = (
                _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_tiled_impl
                if overflow_block_count <= target_block_overflow_fast_max_blocks
                else _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl
            )
            overflow_pair_acc = overflow_pair_kernel(
                positions,
                jnp.asarray(precomputed_target_block_offsets, dtype=INDEX_DTYPE),
                jnp.asarray(precomputed_target_block_leaf_ids, dtype=INDEX_DTYPE),
                jnp.asarray(
                    precomputed_target_block_source_leaf_ids,
                    dtype=INDEX_DTYPE,
                ),
                jnp.asarray(precomputed_target_block_valid_mask, dtype=bool),
                leaf_positions,
                leaf_masses,
                leaf_mask,
                leaf_particle_idx,
                G=G,
                softening_sq=softening_sq,
                target_leaf_batch_size=target_leaf_batch_size,
                target_block_tile_size=target_block_tile_size,
                target_block_tile_scan_unroll=target_block_tile_scan_unroll,
                target_block_batch_scan_unroll=target_block_batch_scan_unroll,
            )
            pair_acc = pair_acc + overflow_pair_acc
        return self_acc + pair_acc

    if use_target_blocks:
        return _compute_leaf_p2p_prepared_large_n_accel_only_target_blocks_impl(
            positions,
            jnp.asarray(precomputed_target_block_offsets, dtype=INDEX_DTYPE),
            jnp.asarray(precomputed_target_block_leaf_ids, dtype=INDEX_DTYPE),
            jnp.asarray(precomputed_target_block_source_leaf_ids, dtype=INDEX_DTYPE),
            jnp.asarray(precomputed_target_block_valid_mask, dtype=bool),
            leaf_positions,
            leaf_masses,
            leaf_mask,
            leaf_particle_idx,
            G=G,
            softening_sq=softening_sq,
            target_leaf_batch_size=target_leaf_batch_size,
            target_block_tile_size=target_block_tile_size,
            target_block_tile_scan_unroll=target_block_tile_scan_unroll,
            target_block_batch_scan_unroll=target_block_batch_scan_unroll,
        )

    return _compute_leaf_p2p_prepared_large_n_accel_only_impl(
        positions,
        target_leaf_ids,
        source_leaf_ids,
        valid_pairs,
        leaf_positions,
        leaf_masses,
        leaf_mask,
        leaf_particle_idx,
        G=G,
        softening_sq=softening_sq,
        edge_chunk_size=int(edge_chunk_size),
        chunks_per_superchunk=delayed_scatter_chunks_per_superchunk,
        chunk_scan_batch_size=chunk_scan_batch_size,
        chunk_scan_unroll=chunk_scan_unroll,
        superchunk_scan_unroll=superchunk_scan_unroll,
        sorted_scatter_hint=sorted_scatter_hint,
        grouped_sorted_scatter=grouped_sorted_scatter,
        superchunk_target_reduce=superchunk_target_reduce,
        disable_chunk_cond=disable_chunk_cond,
    )
