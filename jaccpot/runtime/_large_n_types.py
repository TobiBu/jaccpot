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
    retain_pair_vectors: bool
    precompute_scatter: bool


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
        tuple[
            int,
            str,
            str,
            float,
            Optional[str],
            Tuple[DualTreeRetryEvent, ...],
            str,
            str,
            str,
            int,
        ],
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
            self.force_scale_nodes,
        )
        aux = (
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
        )
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[Any, ...],
        children: tuple[Any, ...],
    ) -> "LargeNPreparedState":
        (
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
        )
