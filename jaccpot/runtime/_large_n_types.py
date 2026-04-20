"""Slim prepared-state contracts for the large-N runtime path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jaxtyping import Array
from yggdrax.interactions import DualTreeRetryEvent, NodeNeighborList
from yggdrax.tree import Tree

from jaccpot.downward.local_expansions import LocalExpansionData

from .dtypes import INDEX_DTYPE


@dataclass(frozen=True)
class LargeNExecutionConfig:
    """Resolved policy for the narrow large-N GPU runtime path."""

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
    """Canonical payload contract for radix fast-lane nearfield evaluation."""

    target_leaf_ids: Array
    target_particle_ids: Array
    target_particle_mask: Array
    source_leaf_ids: Array
    source_leaf_valid_mask: Array
    source_particle_ids: Array
    source_particle_mask: Array
    batch_tile_t: int
    batch_tile_s: int
    source_slot_scan_unroll: int = 1
    target_batch_scan_unroll: int = 1
    fallback_block_tile_size: int = 8
    fallback_tile_scan_unroll: int = 1
    fallback_batch_scan_unroll: int = 1

    def tree_flatten(
        self,
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
        cls,
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
class LargeNPreparedState:
    """Prepared-state payload for the dedicated large-N full-evaluation path."""

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
    input_dtype: jnp.dtype
    working_dtype: jnp.dtype
    theta: float
    topology_key: Optional[str]
    retry_events: Tuple[DualTreeRetryEvent, ...]
    force_scale_nodes: Optional[Array] = None
    execution_backend: str = "large_n"
    expansion_basis: str = "solidfmm"
    nearfield_mode: str = "baseline"
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
    nearfield_target_block_overflow_fast_max_blocks: int = 65536
    speed_prepared_layout: bool = False
    radix_fast_lane: bool = False
    disable_specialized_large_n_nearfield: bool = False
    radix_fast_payload: Optional[RadixFastNearfieldPayload] = None

    @property
    def positions_sorted(self) -> Array:
        value = getattr(self.tree, "positions_sorted", None)
        if value is None:
            raise ValueError("prepared tree is missing positions_sorted")
        return jnp.asarray(value)

    @property
    def masses_sorted(self) -> Array:
        value = getattr(self.tree, "masses_sorted", None)
        if value is None:
            raise ValueError("prepared tree is missing masses_sorted")
        return jnp.asarray(value)

    @property
    def inverse_permutation(self) -> Array:
        value = getattr(self.tree, "inverse_permutation", None)
        if value is None:
            raise ValueError("prepared tree is missing inverse_permutation")
        return jnp.asarray(value, dtype=INDEX_DTYPE)

    def tree_flatten(
        self,
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
            self.force_scale_nodes,
        )
        aux = (
            int(self.nearfield_target_block_size),
            int(self.max_leaf_size),
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
            bool(self.speed_prepared_layout),
            bool(self.radix_fast_lane),
            bool(self.disable_specialized_large_n_nearfield),
        )
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[Any, ...],
        children: tuple[Any, ...],
    ) -> "LargeNPreparedState":
        if len(aux) < 13:
            raise ValueError("LargeNPreparedState aux payload is malformed")
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
            nearfield_target_block_overflow_fast_max_blocks = 65536
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
            nearfield_target_block_overflow_fast_max_blocks = 65536
        else:
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
            force_scale_nodes,
        ) = children
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
            speed_prepared_layout=bool(speed_prepared_layout),
            radix_fast_lane=bool(radix_fast_lane),
            disable_specialized_large_n_nearfield=bool(
                disable_specialized_large_n_nearfield
            ),
            radix_fast_payload=radix_fast_payload,
        )
