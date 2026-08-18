"""Slim prepared-state contracts for the large-N runtime path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jax.typing import ArrayLike
from jaxtyping import Array
from yggdrax.interactions import (
    CompactTaggedFarPairs,
    DualTreeRetryEvent,
    NodeNeighborList,
)
from yggdrax.tree import Tree

from jaccpot.downward.local_expansions import LocalExpansionData

from .dtypes import INDEX_DTYPE
from .fmm_constants import _NEARFIELD_TARGET_BLOCK_OVERFLOW_FAST_MAX_BLOCKS

__all__ = [
    "LargeNCompiledState",
    "LargeNExecutionConfig",
    "LargeNPreparedState",
    "RadixFastNearfieldPayload",
    "large_n_as_prepared_state",
    "large_n_to_compiled_state",
]


class _CompatInteractionStorage(NamedTuple):
    sources: Array
    targets: Array
    level_offsets: Optional[Array]


class _CompatDownwardView(NamedTuple):
    interactions: _CompatInteractionStorage


@dataclass(frozen=True)
class LargeNExecutionConfig:
    """Resolved policy for the narrow large-N GPU runtime path.

    *Resolved*, not requested: by the time one of these exists every ``"auto"``
    has been decided, so nothing here is a hint. Frozen, and not a pytree -- it
    is host-side policy consumed while building a prepared state, never carried
    into a trace.

    Attributes
    ----------
    nearfield_mode : str
        Which near-field schedule to build (``"bucketed"`` and the target-block
        modes). Selects which of the optional arrays on
        :class:`LargeNPreparedState` get populated.
    nearfield_edge_chunk_size : int
        Pairs per chunk in the bucketed near-field schedule.
    retain_leaf_groups : bool
        Keep the per-leaf grouping arrays on the prepared state. Memory against
        the ability to re-derive schedules without rebuilding.
    retain_pair_vectors : bool
        Keep the materialised pair vectors. The expensive one -- these are the
        O(pairs) arrays, so this is the main footprint knob here.
    precompute_scatter : bool
        Build the scatter schedules during prepare rather than at evaluation.
    target_owned_block_size : int
        Target particles per owned block in the target-block near-field modes.
    speed_prepared_layout : bool
        Use the speed-oriented prepared layout. Trades footprint for evaluation
        throughput; it does not change what is computed.
    radix_fast_lane : bool
        Build the radix fast-lane payload. This is what makes
        ``radix_fast_payload`` non-``None`` downstream, and so what selects the
        lane with the analytic O(N) reverse.
    """

    nearfield_mode: str
    nearfield_edge_chunk_size: int
    retain_leaf_groups: bool
    retain_pair_vectors: bool
    precompute_scatter: bool
    target_owned_block_size: int
    speed_prepared_layout: bool
    radix_fast_lane: bool


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RadixFastNearfieldPayload:
    """Canonical payload contract for radix fast-lane nearfield evaluation.

    THE ARRAY FIELDS ARE ``ArrayLike``, NOT ``Array``, AND THAT IS DELIBERATE.
    The production builder (``runtime/_nearfield_fastlane._payload``) fills them
    with **NumPy**, and its comment says why: this payload is memoized and reused
    across traces, so ``jnp`` arrays minted inside the first trace would be that
    trace's tracers and leak into the next one as an ``UnexpectedTracerError``.
    Host constants re-enter each trace cleanly, and the lane's own ``jnp.asarray``
    converts them at the point of use.

    So a ``jax.Array`` hint here was simply wrong -- it described neither what the
    builder produces nor what the consumers want, and it made every construction
    fail under ``JACCPOT_RUNTIME_TYPECHECK=1`` (F40). Do **not** "fix" this by
    converting the builder to ``jnp``: that reintroduces the tracer leak.

    Registered as a pytree, and the split is the thing to know before adding a
    field: the seven array fields are **children** (traced), every ``int`` is
    **aux** (static, so changing one recompiles). ``tree_unflatten`` accepts a
    2-entry aux as well as the current 7 for backward compatibility, defaulting
    the five it cannot find -- so an old flattened payload silently reconstructs
    with default tiling rather than failing.

    Attributes
    ----------
    target_leaf_ids : ArrayLike
        Leaf id per target batch row.
    target_particle_ids : ArrayLike
        Target particle index per (leaf, slot). Padded; gate with
        ``target_particle_mask``.
    target_particle_mask : ArrayLike
        Occupancy for ``target_particle_ids``, same shape.
    source_leaf_ids : ArrayLike
        Source leaf id per (target leaf, slot tile).
    source_leaf_valid_mask : ArrayLike
        Which entries of ``source_leaf_ids`` are real.
    source_particle_ids : ArrayLike
        Source particle index per (target leaf, slot, lane). Left **empty** on
        purpose by the prepacked builder: an empty pair of source-particle arrays
        is what selects the prepacked source-leaf-id layout over the materialised
        pairs layout, whose reverse is not wired.
    source_particle_mask : ArrayLike
        Occupancy for ``source_particle_ids``, same shape and same convention.
    batch_tile_t : int
        Target leaves per batch. Static.
    batch_tile_s : int
        Source slots per tile. Static.
    source_slot_scan_unroll : int
        Unroll for the source-slot scan; defaults to 1. Static, tuning only.
    target_batch_scan_unroll : int
        Unroll for the target-batch scan; defaults to 1. Static, tuning only.
    fallback_block_tile_size : int
        Block tile size for the fallback lane; defaults to 8.
    fallback_tile_scan_unroll : int
        Tile-scan unroll for the fallback lane; defaults to 1.
    fallback_batch_scan_unroll : int
        Batch-scan unroll for the fallback lane; defaults to 1.
    """

    target_leaf_ids: ArrayLike
    target_particle_ids: ArrayLike
    target_particle_mask: ArrayLike
    source_leaf_ids: ArrayLike
    source_leaf_valid_mask: ArrayLike
    source_particle_ids: ArrayLike
    source_particle_mask: ArrayLike
    batch_tile_t: int
    batch_tile_s: int
    source_slot_scan_unroll: int = 1
    target_batch_scan_unroll: int = 1
    fallback_block_tile_size: int = 8
    fallback_tile_scan_unroll: int = 1
    fallback_batch_scan_unroll: int = 1

    def tree_flatten(
        self: "RadixFastNearfieldPayload",
    ) -> tuple[tuple[Any, ...], tuple[int, int, int, int, int, int, int]]:
        children = (
            self.target_leaf_ids,
            self.target_particle_ids,
            self.target_particle_mask,
            self.source_leaf_ids,
            self.source_leaf_valid_mask,
            self.source_particle_ids,
            self.source_particle_mask,
        )
        aux = (
            int(self.batch_tile_t),
            int(self.batch_tile_s),
            int(self.source_slot_scan_unroll),
            int(self.target_batch_scan_unroll),
            int(self.fallback_block_tile_size),
            int(self.fallback_tile_scan_unroll),
            int(self.fallback_batch_scan_unroll),
        )
        return children, aux

    @classmethod
    def tree_unflatten(
        cls: type["RadixFastNearfieldPayload"],
        aux: tuple[Any, ...],
        children: tuple[Any, ...],
    ) -> "RadixFastNearfieldPayload":
        if len(aux) == 2:
            batch_tile_t, batch_tile_s = aux
            source_slot_scan_unroll = 1
            target_batch_scan_unroll = 1
            fallback_block_tile_size = 8
            fallback_tile_scan_unroll = 1
            fallback_batch_scan_unroll = 1
        else:
            (
                batch_tile_t,
                batch_tile_s,
                source_slot_scan_unroll,
                target_batch_scan_unroll,
                fallback_block_tile_size,
                fallback_tile_scan_unroll,
                fallback_batch_scan_unroll,
            ) = aux
        (
            target_leaf_ids,
            target_particle_ids,
            target_particle_mask,
            source_leaf_ids,
            source_leaf_valid_mask,
            source_particle_ids,
            source_particle_mask,
        ) = children
        return cls(
            target_leaf_ids=target_leaf_ids,
            target_particle_ids=target_particle_ids,
            target_particle_mask=target_particle_mask,
            source_leaf_ids=source_leaf_ids,
            source_leaf_valid_mask=source_leaf_valid_mask,
            source_particle_ids=source_particle_ids,
            source_particle_mask=source_particle_mask,
            batch_tile_t=int(batch_tile_t),
            batch_tile_s=int(batch_tile_s),
            source_slot_scan_unroll=int(source_slot_scan_unroll),
            target_batch_scan_unroll=int(target_batch_scan_unroll),
            fallback_block_tile_size=int(fallback_block_tile_size),
            fallback_tile_scan_unroll=int(fallback_tile_scan_unroll),
            fallback_batch_scan_unroll=int(fallback_batch_scan_unroll),
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LargeNCompiledState:
    """JAX-carryable large-N runtime state used by compiled refresh loops.

    A thin carrier around :class:`LargeNPreparedState`, existing so a compiled
    refresh loop can carry the state as a pytree with the hot arrays hoisted to
    the top level instead of reached through properties on every iteration.

    It does not replace the prepared state: ``prepared`` is still a child, so the
    whole thing is carried either way, and :meth:`to_prepared` hands the original
    back unchanged. Constructing one via :meth:`from_prepared` calls
    ``jnp.asarray`` and ``int()`` on the source's properties, so build it outside
    the loop, not inside.

    Attributes
    ----------
    prepared : LargeNPreparedState
        The underlying prepared state, carried as a pytree child.
    positions_sorted : Array
        ``[N, 3]`` positions in tree order, hoisted from ``prepared.tree``.
    masses_sorted : Array
        ``[N]`` masses in tree order, hoisted the same way.
    inverse_permutation : Array
        ``[N]`` map from tree order back to input order, in ``INDEX_DTYPE``.
    topology_key : Optional[str]
        Identity of the tree topology, or ``None``. **Aux**, so a change to it
        triggers a retrace -- which is the intent: a different topology is a
        different compiled program.
    max_leaf_size : int
        Padded leaf width. Aux, static.
    local_order : int
        Local-expansion order. Aux, static. Falls back to
        ``prepared.local_data.order`` when the source does not carry it.
    """

    prepared: LargeNPreparedState
    positions_sorted: Array
    masses_sorted: Array
    inverse_permutation: Array
    topology_key: Optional[str]
    max_leaf_size: int
    local_order: int

    @classmethod
    def from_prepared(
        cls: type["LargeNCompiledState"], prepared: "LargeNPreparedState"
    ) -> "LargeNCompiledState":
        return cls(
            prepared=prepared,
            positions_sorted=jnp.asarray(prepared.positions_sorted),
            masses_sorted=jnp.asarray(prepared.masses_sorted),
            inverse_permutation=jnp.asarray(
                prepared.inverse_permutation, dtype=INDEX_DTYPE
            ),
            topology_key=getattr(prepared, "topology_key", None),
            max_leaf_size=int(prepared.max_leaf_size),
            local_order=int(
                getattr(prepared, "local_order", prepared.local_data.order)
            ),
        )

    def to_prepared(self: "LargeNCompiledState") -> "LargeNPreparedState":
        return self.prepared

    def tree_flatten(
        self: "LargeNCompiledState",
    ) -> tuple[tuple[Any, ...], tuple[Optional[str], int, int]]:
        children = (
            self.prepared,
            self.positions_sorted,
            self.masses_sorted,
            self.inverse_permutation,
        )
        aux = (self.topology_key, int(self.max_leaf_size), int(self.local_order))
        return children, aux

    @classmethod
    def tree_unflatten(
        cls: type["LargeNCompiledState"],
        aux: tuple[Optional[str], int, int],
        children: tuple[Any, ...],
    ) -> "LargeNCompiledState":
        topology_key, max_leaf_size, local_order = aux
        prepared, positions_sorted, masses_sorted, inverse_permutation = children
        return cls(
            prepared=prepared,
            positions_sorted=positions_sorted,
            masses_sorted=masses_sorted,
            inverse_permutation=inverse_permutation,
            topology_key=topology_key,
            max_leaf_size=int(max_leaf_size),
            local_order=int(local_order),
        )


def large_n_as_prepared_state(
    state: Union["LargeNPreparedState", LargeNCompiledState],
) -> "LargeNPreparedState":
    if isinstance(state, LargeNCompiledState):
        return state.prepared
    return state


def large_n_to_compiled_state(
    state: Union["LargeNPreparedState", LargeNCompiledState],
) -> LargeNCompiledState:
    if isinstance(state, LargeNCompiledState):
        return state
    return LargeNCompiledState.from_prepared(state)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LargeNPreparedState:
    """Prepared-state payload for the dedicated large-N full-evaluation path.

    Everything the large-N lane needs to evaluate without rebuilding: the tree,
    the local expansions, the neighbour list, and one near-field schedule.

    **Which optional arrays are populated is decided by ``nearfield_mode`` and
    ``radix_fast_lane``, and the unused ones are ``None``.** The three schedules
    are alternatives, not layers: the bucketed chunk arrays, the target-block
    arrays, and the radix payload. Reading a field belonging to a mode that is
    not active gets ``None``, so consumers branch rather than assume.

    Registered as a pytree, and the children/aux split is the thing to check
    before adding a field: arrays and sub-payloads are **children** (traced);
    every scalar, dtype, string and the retry-event tuple is **aux** (static, so
    changing one forces a recompile). ``theta`` is aux -- sweeping it recompiles.
    The dtypes round-trip through ``str(jnp.dtype(...))`` because aux must be
    hashable.

    ``tree_unflatten`` accepts four historical aux lengths (13, 14, 29, and the
    current 30) and defaults whatever is missing, so an older flattened state
    reconstructs silently with default tuning rather than raising. Only a length
    below 13 is rejected. The children tuple is likewise back-compatible, keying
    off how many trail after ``radix_fast_payload``.

    Attributes
    ----------
    tree : Tree
        The yggdrax tree. Also the source of ``positions_sorted``,
        ``masses_sorted`` and ``inverse_permutation``, which are properties here
        rather than fields and raise ``ValueError`` if the tree lacks them.
    local_data : LocalExpansionData
        Local expansions per node, from the downward sweep.
    neighbor_list : NodeNeighborList
        Leaf-neighbour lists backing the near field.
    nearfield_leaf_particle_indices : Array
        ``[num_leaves, W]`` particle index per padded leaf slot.
    nearfield_leaf_particle_mask : Array
        ``[num_leaves, W]`` occupancy for the above.
    nearfield_target_leaf_ids : Optional[Array]
        Target leaf id per near-field pair; ``None`` when the active mode does
        not materialise a pair list.
    nearfield_source_leaf_ids : Optional[Array]
        Source leaf id per near-field pair, paired positionally with the above.
    nearfield_valid_pairs : Optional[Array]
        Validity mask over the pair list.
    nearfield_chunk_sort_indices : Optional[Array]
        Bucketed mode: permutation sorting pairs into chunks.
    nearfield_chunk_group_ids : Optional[Array]
        Bucketed mode: chunk id per sorted pair.
    nearfield_chunk_unique_indices : Optional[Array]
        Bucketed mode: first index of each chunk group.
    nearfield_target_block_leaf_ids : Optional[Array]
        Target-block mode: leaf id per target block.
    nearfield_target_block_source_leaf_ids : Optional[Array]
        Target-block mode: source leaf ids per block.
    nearfield_target_block_valid_mask : Optional[Array]
        Target-block mode: validity for the above.
    nearfield_target_block_offsets : Optional[Array]
        Target-block mode: CSR offsets into the block source lists.
    nearfield_target_block_source_leaf_ids_padded : Optional[Array]
        Target-block mode: the fixed-shape padded twin of the source leaf ids,
        which is what the compiled lane consumes.
    nearfield_target_block_valid_mask_padded : Optional[Array]
        Target-block mode: validity for the padded twin.
    nearfield_target_block_size : int
        Target particles per block. Aux.
    max_leaf_size : int
        Padded leaf width ``W``. Aux.
    local_order : int
        Local-expansion order. Aux.
    input_dtype : jnp.dtype
        Dtype the caller supplied. Aux; round-tripped as a string.
    working_dtype : jnp.dtype
        Dtype the evaluation runs in, which may differ from ``input_dtype``. Aux.
    theta : float
        Opening angle the interaction lists were built at. Aux, so it is baked
        into the compiled program -- changing it recompiles, and it is recorded
        here precisely because the lists are only valid for this value.
    topology_key : Optional[str]
        Identity of the tree topology for refresh reuse, or ``None``. Aux.
    retry_events : Tuple[DualTreeRetryEvent, ...]
        Traversal capacity retries hit while building. Aux. Non-empty means the
        build succeeded only after growing a buffer -- diagnostic, not an error.
    force_scale_nodes : Optional[Array]
        Per-node force scale for the acceptance criterion, or ``None``. ``None``
        is not neutral: downstream it becomes a unit scale, which is a different
        criterion. See
        :meth:`~jaccpot.runtime.fmm_prepare.PrepareMixin._resolve_force_scale_nodes_for_prepare`.
    execution_backend : str
        Backend tag; defaults to ``"large_n"``. Aux.
    expansion_basis : str
        Basis the expansions are in; defaults to ``"solidfmm"``. Aux.
    nearfield_mode : str
        Active near-field schedule; defaults to ``"bucketed"``. Aux. This is the
        field that decides which optional arrays above are populated.
    nearfield_edge_chunk_size : int
        Bucketed mode: pairs per chunk; defaults to 256. Aux.
    nearfield_delayed_scatter_chunks_per_superchunk : int
        Bucketed mode: chunks per delayed-scatter superchunk; defaults to 1. Aux.
    nearfield_chunk_scan_batch_size : int
        Bucketed mode: chunks per scan step; defaults to 1. Aux.
    nearfield_chunk_scan_unroll : int
        Bucketed mode: chunk-scan unroll; defaults to 1. Aux.
    nearfield_superchunk_scan_unroll : int
        Bucketed mode: superchunk-scan unroll; defaults to 1. Aux.
    nearfield_sorted_scatter_hint : bool
        Assert the scatter indices are sorted, enabling the cheaper scatter.
        Defaults to ``False``. A **promise, not a request**: asserting it falsely
        gives wrong sums, not a slow path.
    nearfield_grouped_sorted_scatter : bool
        Use the grouped sorted scatter; defaults to ``False``. Aux.
    nearfield_superchunk_target_reduce : bool
        Reduce per target inside a superchunk; defaults to ``False``. Aux.
    nearfield_disable_chunk_cond : bool
        Skip the per-chunk emptiness test; defaults to ``True``. Aux. Disabling
        the test is the default because the branch costs more than the work it
        saves at this size.
    nearfield_target_leaf_batch_size : int
        Target leaves per batch; defaults to 32. Aux.
    nearfield_target_block_tile_size : int
        Source tile width in target-block mode; defaults to 8. Aux.
    nearfield_target_block_tile_scan_unroll : int
        Tile-scan unroll; defaults to 1. Aux.
    nearfield_target_block_batch_scan_unroll : int
        Batch-scan unroll; defaults to 1. Aux.
    nearfield_target_block_overflow_fast_max_blocks : int
        Block cap below which the fast overflow path is used.
    nearfield_target_block_overflow_profile_capacity : int
        Overflow capacity the compiled profile was built for; defaults to 0.
    nearfield_target_block_overflow_active_blocks : int
        Overflow blocks actually in use; defaults to 0. Compared against the
        capacity above to decide whether a compiled program can be reused.
    speed_prepared_layout : bool
        Speed-oriented layout was used; defaults to ``False``. Aux.
    radix_fast_lane : bool
        Radix fast lane is active; defaults to ``False``. Aux.
    disable_specialized_large_n_nearfield : bool
        Fall back to the generic near-field kernels; defaults to ``False``. Aux.
    radix_fast_payload : Optional[RadixFastNearfieldPayload]
        The fast-lane payload, or ``None`` when the lane is off. A pytree child,
        so it is carried through traces with the rest of the state.
    radix_overflow_payload : Optional[RadixFastNearfieldPayload]
        Second payload for leaves that overflowed the fast lane's fixed shapes,
        or ``None``. When present its contribution is **additive** -- dropping it
        silently loses the overflowing leaves' near field.
    compact_far_pairs : Optional[CompactTaggedFarPairs]
        Compact far-pair list when the streamed far field retains one, else
        ``None``.
    """

    tree: Tree
    local_data: LocalExpansionData
    neighbor_list: NodeNeighborList
    nearfield_leaf_particle_indices: Array
    nearfield_leaf_particle_mask: Array
    nearfield_target_leaf_ids: Optional[Array]
    nearfield_source_leaf_ids: Optional[Array]
    nearfield_valid_pairs: Optional[Array]
    nearfield_chunk_sort_indices: Optional[Array]
    nearfield_chunk_group_ids: Optional[Array]
    nearfield_chunk_unique_indices: Optional[Array]
    nearfield_target_block_leaf_ids: Optional[Array]
    nearfield_target_block_source_leaf_ids: Optional[Array]
    nearfield_target_block_valid_mask: Optional[Array]
    nearfield_target_block_offsets: Optional[Array]
    nearfield_target_block_source_leaf_ids_padded: Optional[Array]
    nearfield_target_block_valid_mask_padded: Optional[Array]
    nearfield_target_block_size: int
    max_leaf_size: int
    local_order: int
    input_dtype: jnp.dtype
    working_dtype: jnp.dtype
    theta: float
    topology_key: Optional[str]
    retry_events: Tuple[DualTreeRetryEvent, ...]
    force_scale_nodes: Optional[Array] = None
    execution_backend: str = "large_n"
    expansion_basis: str = "solidfmm"
    nearfield_mode: str = "bucketed"
    nearfield_edge_chunk_size: int = 256
    nearfield_delayed_scatter_chunks_per_superchunk: int = 1
    nearfield_chunk_scan_batch_size: int = 1
    nearfield_chunk_scan_unroll: int = 1
    nearfield_superchunk_scan_unroll: int = 1
    nearfield_sorted_scatter_hint: bool = False
    nearfield_grouped_sorted_scatter: bool = False
    nearfield_superchunk_target_reduce: bool = False
    nearfield_disable_chunk_cond: bool = True
    nearfield_target_leaf_batch_size: int = 32
    nearfield_target_block_tile_size: int = 8
    nearfield_target_block_tile_scan_unroll: int = 1
    nearfield_target_block_batch_scan_unroll: int = 1
    nearfield_target_block_overflow_fast_max_blocks: int = (
        _NEARFIELD_TARGET_BLOCK_OVERFLOW_FAST_MAX_BLOCKS
    )
    nearfield_target_block_overflow_profile_capacity: int = 0
    nearfield_target_block_overflow_active_blocks: int = 0
    speed_prepared_layout: bool = False
    radix_fast_lane: bool = False
    disable_specialized_large_n_nearfield: bool = False
    radix_fast_payload: Optional[RadixFastNearfieldPayload] = None
    radix_overflow_payload: Optional[RadixFastNearfieldPayload] = None
    compact_far_pairs: Optional[CompactTaggedFarPairs] = None

    @property
    def positions_sorted(self: "LargeNPreparedState") -> Array:
        value = getattr(self.tree, "positions_sorted", None)
        if value is None:
            raise ValueError("prepared tree is missing positions_sorted")
        return jnp.asarray(value)

    @property
    def masses_sorted(self: "LargeNPreparedState") -> Array:
        value = getattr(self.tree, "masses_sorted", None)
        if value is None:
            raise ValueError("prepared tree is missing masses_sorted")
        return jnp.asarray(value)

    @property
    def inverse_permutation(self: "LargeNPreparedState") -> Array:
        value = getattr(self.tree, "inverse_permutation", None)
        if value is None:
            raise ValueError("prepared tree is missing inverse_permutation")
        return jnp.asarray(value, dtype=INDEX_DTYPE)

    @property
    def interactions(self: "LargeNPreparedState") -> None:
        """Compatibility view: large-N state does not retain far interactions."""
        return None

    @property
    def downward(self: "LargeNPreparedState") -> _CompatDownwardView:
        """Compatibility view exposing empty downward interactions."""
        empty = _CompatInteractionStorage(
            sources=jnp.zeros((0,), dtype=INDEX_DTYPE),
            targets=jnp.zeros((0,), dtype=INDEX_DTYPE),
            level_offsets=None,
        )
        return _CompatDownwardView(interactions=empty)

    def tree_flatten(
        self: "LargeNPreparedState",
    ) -> tuple[
        tuple[Any, ...],
        tuple[Any, ...],
    ]:
        children = (
            self.tree,
            self.local_data,
            self.neighbor_list,
            self.nearfield_leaf_particle_indices,
            self.nearfield_leaf_particle_mask,
            self.nearfield_target_leaf_ids,
            self.nearfield_source_leaf_ids,
            self.nearfield_valid_pairs,
            self.nearfield_chunk_sort_indices,
            self.nearfield_chunk_group_ids,
            self.nearfield_chunk_unique_indices,
            self.nearfield_target_block_leaf_ids,
            self.nearfield_target_block_source_leaf_ids,
            self.nearfield_target_block_valid_mask,
            self.nearfield_target_block_offsets,
            self.nearfield_target_block_source_leaf_ids_padded,
            self.nearfield_target_block_valid_mask_padded,
            self.radix_fast_payload,
            self.radix_overflow_payload,
            self.compact_far_pairs,
            self.force_scale_nodes,
        )
        aux = (
            int(self.nearfield_target_block_size),
            int(self.max_leaf_size),
            int(self.local_order),
            str(jnp.dtype(self.input_dtype)),
            str(jnp.dtype(self.working_dtype)),
            float(self.theta),
            self.topology_key,
            self.retry_events,
            str(self.execution_backend),
            str(self.expansion_basis),
            str(self.nearfield_mode),
            int(self.nearfield_edge_chunk_size),
            int(self.nearfield_delayed_scatter_chunks_per_superchunk),
            int(self.nearfield_chunk_scan_batch_size),
            int(self.nearfield_chunk_scan_unroll),
            int(self.nearfield_superchunk_scan_unroll),
            bool(self.nearfield_sorted_scatter_hint),
            bool(self.nearfield_grouped_sorted_scatter),
            bool(self.nearfield_superchunk_target_reduce),
            bool(self.nearfield_disable_chunk_cond),
            int(self.nearfield_target_leaf_batch_size),
            int(self.nearfield_target_block_tile_size),
            int(self.nearfield_target_block_tile_scan_unroll),
            int(self.nearfield_target_block_batch_scan_unroll),
            int(self.nearfield_target_block_overflow_fast_max_blocks),
            int(self.nearfield_target_block_overflow_profile_capacity),
            int(self.nearfield_target_block_overflow_active_blocks),
            bool(self.speed_prepared_layout),
            bool(self.radix_fast_lane),
            bool(self.disable_specialized_large_n_nearfield),
        )
        return children, aux

    @classmethod
    def tree_unflatten(
        cls: type["LargeNPreparedState"],
        aux: tuple[Any, ...],
        children: tuple[Any, ...],
    ) -> "LargeNPreparedState":
        if len(aux) < 13:
            raise ValueError("LargeNPreparedState aux payload is malformed")
        local_order: Optional[int] = None
        if len(aux) == 13:
            (
                nearfield_target_block_size,
                max_leaf_size,
                input_dtype_name,
                working_dtype_name,
                theta,
                topology_key,
                retry_events,
                execution_backend,
                expansion_basis,
                nearfield_mode,
                nearfield_edge_chunk_size,
                speed_prepared_layout,
                radix_fast_lane,
            ) = aux
            nearfield_delayed_scatter_chunks_per_superchunk = 1
            nearfield_chunk_scan_batch_size = 1
            nearfield_chunk_scan_unroll = 1
            nearfield_superchunk_scan_unroll = 1
            nearfield_sorted_scatter_hint = False
            nearfield_grouped_sorted_scatter = False
            nearfield_superchunk_target_reduce = False
            nearfield_disable_chunk_cond = True
            nearfield_target_leaf_batch_size = 32
            nearfield_target_block_tile_size = 8
            nearfield_target_block_tile_scan_unroll = 1
            nearfield_target_block_batch_scan_unroll = 1
            nearfield_target_block_overflow_fast_max_blocks = (
                _NEARFIELD_TARGET_BLOCK_OVERFLOW_FAST_MAX_BLOCKS
            )
            nearfield_target_block_overflow_profile_capacity = 0
            nearfield_target_block_overflow_active_blocks = 0
            disable_specialized_large_n_nearfield = False
        elif len(aux) == 14:
            (
                nearfield_target_block_size,
                max_leaf_size,
                input_dtype_name,
                working_dtype_name,
                theta,
                topology_key,
                retry_events,
                execution_backend,
                expansion_basis,
                nearfield_mode,
                nearfield_edge_chunk_size,
                speed_prepared_layout,
                radix_fast_lane,
                disable_specialized_large_n_nearfield,
            ) = aux
            nearfield_delayed_scatter_chunks_per_superchunk = 1
            nearfield_chunk_scan_batch_size = 1
            nearfield_chunk_scan_unroll = 1
            nearfield_superchunk_scan_unroll = 1
            nearfield_sorted_scatter_hint = False
            nearfield_grouped_sorted_scatter = False
            nearfield_superchunk_target_reduce = False
            nearfield_disable_chunk_cond = True
            nearfield_target_leaf_batch_size = 32
            nearfield_target_block_tile_size = 8
            nearfield_target_block_tile_scan_unroll = 1
            nearfield_target_block_batch_scan_unroll = 1
            nearfield_target_block_overflow_fast_max_blocks = (
                _NEARFIELD_TARGET_BLOCK_OVERFLOW_FAST_MAX_BLOCKS
            )
            nearfield_target_block_overflow_profile_capacity = 0
            nearfield_target_block_overflow_active_blocks = 0
        elif len(aux) == 29:
            (
                nearfield_target_block_size,
                max_leaf_size,
                input_dtype_name,
                working_dtype_name,
                theta,
                topology_key,
                retry_events,
                execution_backend,
                expansion_basis,
                nearfield_mode,
                nearfield_edge_chunk_size,
                nearfield_delayed_scatter_chunks_per_superchunk,
                nearfield_chunk_scan_batch_size,
                nearfield_chunk_scan_unroll,
                nearfield_superchunk_scan_unroll,
                nearfield_sorted_scatter_hint,
                nearfield_grouped_sorted_scatter,
                nearfield_superchunk_target_reduce,
                nearfield_disable_chunk_cond,
                nearfield_target_leaf_batch_size,
                nearfield_target_block_tile_size,
                nearfield_target_block_tile_scan_unroll,
                nearfield_target_block_batch_scan_unroll,
                nearfield_target_block_overflow_fast_max_blocks,
                nearfield_target_block_overflow_profile_capacity,
                nearfield_target_block_overflow_active_blocks,
                speed_prepared_layout,
                radix_fast_lane,
                disable_specialized_large_n_nearfield,
            ) = aux
        else:
            (
                nearfield_target_block_size,
                max_leaf_size,
                local_order,
                input_dtype_name,
                working_dtype_name,
                theta,
                topology_key,
                retry_events,
                execution_backend,
                expansion_basis,
                nearfield_mode,
                nearfield_edge_chunk_size,
                nearfield_delayed_scatter_chunks_per_superchunk,
                nearfield_chunk_scan_batch_size,
                nearfield_chunk_scan_unroll,
                nearfield_superchunk_scan_unroll,
                nearfield_sorted_scatter_hint,
                nearfield_grouped_sorted_scatter,
                nearfield_superchunk_target_reduce,
                nearfield_disable_chunk_cond,
                nearfield_target_leaf_batch_size,
                nearfield_target_block_tile_size,
                nearfield_target_block_tile_scan_unroll,
                nearfield_target_block_batch_scan_unroll,
                nearfield_target_block_overflow_fast_max_blocks,
                nearfield_target_block_overflow_profile_capacity,
                nearfield_target_block_overflow_active_blocks,
                speed_prepared_layout,
                radix_fast_lane,
                disable_specialized_large_n_nearfield,
            ) = aux
        (
            tree,
            local_data,
            neighbor_list,
            nearfield_leaf_particle_indices,
            nearfield_leaf_particle_mask,
            nearfield_target_leaf_ids,
            nearfield_source_leaf_ids,
            nearfield_valid_pairs,
            nearfield_chunk_sort_indices,
            nearfield_chunk_group_ids,
            nearfield_chunk_unique_indices,
            nearfield_target_block_leaf_ids,
            nearfield_target_block_source_leaf_ids,
            nearfield_target_block_valid_mask,
            nearfield_target_block_offsets,
            nearfield_target_block_source_leaf_ids_padded,
            nearfield_target_block_valid_mask_padded,
            radix_fast_payload,
            *remaining_children,
        ) = children
        if len(remaining_children) == 1:
            radix_overflow_payload = None
            compact_far_pairs = None
            (force_scale_nodes,) = remaining_children
        elif len(remaining_children) == 2:
            radix_overflow_payload, force_scale_nodes = remaining_children
            compact_far_pairs = None
        else:
            radix_overflow_payload, compact_far_pairs, force_scale_nodes = (
                remaining_children
            )
        if local_order is None:
            local_order = int(getattr(local_data, "order", 0))
        return cls(
            tree=tree,
            local_data=local_data,
            neighbor_list=neighbor_list,
            nearfield_leaf_particle_indices=nearfield_leaf_particle_indices,
            nearfield_leaf_particle_mask=nearfield_leaf_particle_mask,
            nearfield_target_leaf_ids=nearfield_target_leaf_ids,
            nearfield_source_leaf_ids=nearfield_source_leaf_ids,
            nearfield_valid_pairs=nearfield_valid_pairs,
            nearfield_chunk_sort_indices=nearfield_chunk_sort_indices,
            nearfield_chunk_group_ids=nearfield_chunk_group_ids,
            nearfield_chunk_unique_indices=nearfield_chunk_unique_indices,
            nearfield_target_block_leaf_ids=nearfield_target_block_leaf_ids,
            nearfield_target_block_source_leaf_ids=nearfield_target_block_source_leaf_ids,
            nearfield_target_block_valid_mask=nearfield_target_block_valid_mask,
            nearfield_target_block_offsets=nearfield_target_block_offsets,
            nearfield_target_block_source_leaf_ids_padded=(
                nearfield_target_block_source_leaf_ids_padded
            ),
            nearfield_target_block_valid_mask_padded=(
                nearfield_target_block_valid_mask_padded
            ),
            nearfield_target_block_size=int(nearfield_target_block_size),
            max_leaf_size=int(max_leaf_size),
            local_order=int(local_order),
            input_dtype=jnp.dtype(input_dtype_name),
            working_dtype=jnp.dtype(working_dtype_name),
            theta=float(theta),
            topology_key=topology_key,
            retry_events=retry_events,
            force_scale_nodes=force_scale_nodes,
            execution_backend=str(execution_backend),
            expansion_basis=str(expansion_basis),
            nearfield_mode=str(nearfield_mode),
            nearfield_edge_chunk_size=int(nearfield_edge_chunk_size),
            nearfield_delayed_scatter_chunks_per_superchunk=int(
                nearfield_delayed_scatter_chunks_per_superchunk
            ),
            nearfield_chunk_scan_batch_size=int(nearfield_chunk_scan_batch_size),
            nearfield_chunk_scan_unroll=int(nearfield_chunk_scan_unroll),
            nearfield_superchunk_scan_unroll=int(nearfield_superchunk_scan_unroll),
            nearfield_sorted_scatter_hint=bool(nearfield_sorted_scatter_hint),
            nearfield_grouped_sorted_scatter=bool(nearfield_grouped_sorted_scatter),
            nearfield_superchunk_target_reduce=bool(nearfield_superchunk_target_reduce),
            nearfield_disable_chunk_cond=bool(nearfield_disable_chunk_cond),
            nearfield_target_leaf_batch_size=int(nearfield_target_leaf_batch_size),
            nearfield_target_block_tile_size=int(nearfield_target_block_tile_size),
            nearfield_target_block_tile_scan_unroll=int(
                nearfield_target_block_tile_scan_unroll
            ),
            nearfield_target_block_batch_scan_unroll=int(
                nearfield_target_block_batch_scan_unroll
            ),
            nearfield_target_block_overflow_fast_max_blocks=int(
                nearfield_target_block_overflow_fast_max_blocks
            ),
            nearfield_target_block_overflow_profile_capacity=int(
                nearfield_target_block_overflow_profile_capacity
            ),
            nearfield_target_block_overflow_active_blocks=int(
                nearfield_target_block_overflow_active_blocks
            ),
            speed_prepared_layout=bool(speed_prepared_layout),
            radix_fast_lane=bool(radix_fast_lane),
            disable_specialized_large_n_nearfield=bool(
                disable_specialized_large_n_nearfield
            ),
            radix_fast_payload=radix_fast_payload,
            radix_overflow_payload=radix_overflow_payload,
            compact_far_pairs=compact_far_pairs,
        )
