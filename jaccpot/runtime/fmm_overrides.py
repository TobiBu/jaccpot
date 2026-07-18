"""OverridesMixin: fmm_overrides methods extracted from the FastMultipoleMethod
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array
from yggdrax.interactions import DualTreeTraversalConfig

from .fmm_caches import _contains_tracer
from .fmm_constants import (
    _CLASS_MAJOR_CPU_PARTICLE_THRESHOLD,
    _GPU_LARGE_PARTICLE_THRESHOLD,
    _GPU_MAX_INTERACTIONS_PER_NODE,
    _GPU_MAX_NEIGHBORS_PER_LEAF,
    _GPU_MIN_INTERACTIONS_PER_NODE,
    _GPU_MIN_NEIGHBORS_PER_LEAF,
    _GPU_MIN_PAIR_QUEUE_LARGE,
    _GPU_MIN_PAIR_QUEUE_MEDIUM,
    _GPU_MIN_PAIR_QUEUE_XL,
    _GPU_MINIMUM_MEMORY_INTERACTIONS_PER_NODE,
    _GPU_MINIMUM_MEMORY_NEIGHBORS_PER_LEAF,
    _GPU_MINIMUM_MEMORY_PAIR_QUEUE,
    _GPU_MINIMUM_MEMORY_PROCESS_BLOCK,
    _KDTREE_DEFAULT_TRAVERSAL_CONFIG,
    _LARGE_CPU_M2L_CHUNK_SIZE,
    _LARGE_CPU_PARTICLE_THRESHOLD,
    _LARGE_CPU_TRAVERSAL_CONFIG,
    _MINIMUM_MEMORY_CPU_M2L_CHUNK_SIZE,
    _MINIMUM_MEMORY_GPU_M2L_CHUNK_SIZE,
    _NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_LARGE,
    _NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_MEDIUM,
    _NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_XL,
    _NEARFIELD_BUCKETED_CPU_PARTICLE_THRESHOLD,
    _TRACING_MAX_INTERACTIONS_PER_NODE,
    _TRACING_MAX_NEIGHBORS_PER_LEAF,
    _TRACING_MAX_PAIR_QUEUE,
    _TRACING_MAX_PROCESS_BLOCK,
    _minimum_memory_streamed_gpu_traversal_ceiling,
    _minimum_memory_streamed_gpu_traversal_seed,
)
from .fmm_state import _RuntimeExecutionOverrides


class OverridesMixin:
    def _prepared_state_cache_lookup(
        self,
        *,
        key: tuple[Any, ...],
        positions: Array,
        masses: Array,
    ) -> Optional[PreparedStateLike]:
        """Return cached prepared state when key and inputs exactly match."""
        cached_key = self._prepared_state_cache_key
        cached_value = self._prepared_state_cache_value
        cached_positions = self._prepared_state_cache_positions
        cached_masses = self._prepared_state_cache_masses
        if cached_key is None or cached_value is None:
            return None
        if cached_positions is None or cached_masses is None:
            return None
        if cached_key != key:
            return None
        if (
            cached_positions.shape != positions.shape
            or cached_positions.dtype != positions.dtype
            or cached_masses.shape != masses.shape
            or cached_masses.dtype != masses.dtype
        ):
            return None
        if not bool(jnp.array_equal(positions, cached_positions)):
            return None
        if not bool(jnp.array_equal(masses, cached_masses)):
            return None
        return cached_value

    def _prepared_state_cache_store(
        self,
        *,
        key: tuple[Any, ...],
        positions: Array,
        masses: Array,
        state: PreparedStateLike,
    ) -> None:
        """Store prepared-state payload and the exact input arrays used."""
        if _contains_tracer((positions, masses, state)):
            return
        self._prepared_state_cache_key = key
        self._prepared_state_cache_value = state
        self._prepared_state_cache_positions = positions
        self._prepared_state_cache_masses = masses

    def _resolve_jit_tree_flag(
        self,
        positions: Array,
        *,
        jit_tree_override: Optional[bool],
    ) -> bool:
        """Resolve tree-build JIT mode with a CPU-friendly auto heuristic."""

        if self.tree_type != "radix":
            return False

        if jit_tree_override is not None:
            return bool(jit_tree_override)

        default_mode = self._jit_tree_default
        if default_mode != "auto":
            return bool(default_mode)

        backend = jax.default_backend()
        num_particles = int(jnp.asarray(positions).shape[0])
        # CPU tree build often performs better without JIT for small/medium N.
        if backend == "cpu" and num_particles <= 8192:
            return False
        return True

    def _resolve_runtime_execution_overrides(
        self,
        *,
        num_particles: int,
        backend: Optional[str] = None,
    ) -> _RuntimeExecutionOverrides:
        """Resolve adaptive runtime traversal/chunk settings."""

        traversal_config = self.traversal_config
        m2l_chunk_size = self.m2l_chunk_size
        l2l_chunk_size = self.l2l_chunk_size
        grouped_interactions = (
            False
            if self.grouped_interactions is None
            else bool(self.grouped_interactions)
        )
        farfield_mode = self.farfield_mode
        center_mode = "com"
        refine_local_override: Optional[bool] = None
        adaptive_applied = False

        backend_name = jax.default_backend() if backend is None else str(backend)
        n_particles = int(num_particles)
        production_large_n = self._is_large_n_gpu_production_profile()
        static_runtime_fixed_sizing = bool(
            getattr(self, "_static_runtime_fixed_sizing", True)
        )
        minimum_memory = self.memory_objective == "minimum_memory" or production_large_n
        large_cpu = (
            backend_name == "cpu" and n_particles >= _LARGE_CPU_PARTICLE_THRESHOLD
        )
        class_major_cpu = (
            backend_name == "cpu" and n_particles >= _CLASS_MAJOR_CPU_PARTICLE_THRESHOLD
        )
        class_major_gpu = (
            backend_name == "gpu" and n_particles >= _GPU_LARGE_PARTICLE_THRESHOLD
        )

        if self.host_refine_mode == "off":
            refine_local_override = False
        elif self.host_refine_mode == "on":
            refine_local_override = True
        elif (
            large_cpu
            and self.tree_type == "radix"
            and self.preset == "fast"
            and self.expansion_basis == "solidfmm"
            and self.mac_type == "dehnen"
        ):
            refine_local_override = False

        if (
            self.tree_type == "kdtree"
            and not self._explicit_traversal_config
            and not self._explicit_max_pair_queue
            and not self._explicit_pair_process_block
        ):
            traversal_config = _KDTREE_DEFAULT_TRAVERSAL_CONFIG

        if (
            not self._explicit_grouped_interactions
            and self.preset == "fast"
            and self.expansion_basis == "solidfmm"
            and self.mac_type == "dehnen"
            and self.tree_type == "radix"
            and large_cpu
            and not minimum_memory
        ):
            grouped_interactions = True
        if (
            not self._explicit_grouped_interactions
            and self.preset in ("fast", "large_n_gpu")
            and self.expansion_basis == "solidfmm"
            and self.mac_type == "dehnen"
            and self.tree_type == "radix"
            and backend_name == "gpu"
            and n_particles >= _GPU_LARGE_PARTICLE_THRESHOLD
            and not minimum_memory
        ):
            grouped_interactions = True

        if production_large_n:
            grouped_interactions = False
            farfield_mode = "pair_grouped"

        if static_runtime_fixed_sizing:
            # Static sizing mode: keep traversal/chunk execution knobs fixed to
            # constructor/global-input values and skip adaptive runtime rewrites.
            if self.streamed_far_pairs and grouped_interactions:
                grouped_interactions = False
                farfield_mode = "pair_grouped"
            if not grouped_interactions:
                farfield_mode = "pair_grouped"
            return _RuntimeExecutionOverrides(
                traversal_config=traversal_config,
                m2l_chunk_size=m2l_chunk_size,
                l2l_chunk_size=l2l_chunk_size,
                grouped_interactions=grouped_interactions,
                farfield_mode=farfield_mode,
                center_mode=center_mode,
                refine_local_override=refine_local_override,
                adaptive_applied=False,
            )

        if self.streamed_far_pairs and grouped_interactions:
            # Streamed far-pair execution and grouped/class-major M2L are
            # competing strategies. The grouped path overrides streaming in the
            # downward sweep, so keeping both enabled only pays the grouped
            # traversal/materialization cost while defeating the user's request
            # for streamed execution.
            grouped_interactions = False
            farfield_mode = "pair_grouped"

        if (
            self.preset == "fast"
            and self.expansion_basis == "solidfmm"
            and self.mac_type == "dehnen"
            and self.tree_type == "radix"
            and large_cpu
            and not self._explicit_traversal_config
            and not self._explicit_max_pair_queue
            and not self._explicit_pair_process_block
        ):
            traversal_config = _LARGE_CPU_TRAVERSAL_CONFIG
            adaptive_applied = True

            if not self._explicit_m2l_chunk_size:
                m2l_chunk_size = _LARGE_CPU_M2L_CHUNK_SIZE
            if not self._explicit_l2l_chunk_size:
                l2l_chunk_size = self.l2l_chunk_size
        if (
            backend_name == "gpu"
            and self.tree_type == "radix"
            and traversal_config is not None
            and not self._explicit_traversal_config
            and n_particles >= _GPU_LARGE_PARTICLE_THRESHOLD
        ):
            current_queue = int(traversal_config.max_pair_queue)
            current_block = int(traversal_config.process_block)
            current_interactions = int(traversal_config.max_interactions_per_node)
            current_neighbors = int(traversal_config.max_neighbors_per_leaf)

            if minimum_memory:
                target_queue = _GPU_MINIMUM_MEMORY_PAIR_QUEUE
                target_block = _GPU_MINIMUM_MEMORY_PROCESS_BLOCK
                target_interactions = _GPU_MINIMUM_MEMORY_INTERACTIONS_PER_NODE
                target_neighbors = _GPU_MINIMUM_MEMORY_NEIGHBORS_PER_LEAF
            elif n_particles >= 4_194_304:
                target_queue = _GPU_MIN_PAIR_QUEUE_XL
                target_block = current_block
                target_interactions = _GPU_MIN_INTERACTIONS_PER_NODE
                target_neighbors = _GPU_MIN_NEIGHBORS_PER_LEAF
            elif n_particles >= 1_048_576:
                target_queue = _GPU_MIN_PAIR_QUEUE_LARGE
                target_block = current_block
                target_interactions = _GPU_MIN_INTERACTIONS_PER_NODE
                target_neighbors = _GPU_MIN_NEIGHBORS_PER_LEAF
            else:
                target_queue = _GPU_MIN_PAIR_QUEUE_MEDIUM
                target_block = current_block
                target_interactions = _GPU_MIN_INTERACTIONS_PER_NODE
                target_neighbors = _GPU_MIN_NEIGHBORS_PER_LEAF

            if minimum_memory:
                next_queue = min(current_queue, int(target_queue))
                next_block = min(current_block, int(target_block))
                next_interactions = min(current_interactions, int(target_interactions))
                next_neighbors = min(current_neighbors, int(target_neighbors))
            else:
                next_queue = max(current_queue, int(target_queue))
                next_block = current_block
                next_interactions = min(
                    max(current_interactions, int(target_interactions)),
                    int(_GPU_MAX_INTERACTIONS_PER_NODE),
                )
                next_neighbors = min(
                    max(current_neighbors, int(target_neighbors)),
                    int(_GPU_MAX_NEIGHBORS_PER_LEAF),
                )
            if (
                next_queue != current_queue
                or next_block != current_block
                or next_interactions != current_interactions
                or next_neighbors != current_neighbors
            ):
                traversal_config = DualTreeTraversalConfig(
                    max_pair_queue=int(next_queue),
                    process_block=int(next_block),
                    max_interactions_per_node=int(next_interactions),
                    max_neighbors_per_leaf=int(next_neighbors),
                )
        if (
            minimum_memory
            and backend_name == "gpu"
            and self.tree_type == "radix"
            and not self._explicit_traversal_config
            and not self._explicit_max_pair_queue
            and not self._explicit_pair_process_block
        ):
            # The Yggdrax count-pass auto-sizing path is still too expensive on
            # large GPU radix trees. Keep the large-N minimum-memory route on a
            # bounded explicit traversal config so host-side retry can grow from
            # a safe baseline without compiling the count-pass kernel.
            traversal_config = _minimum_memory_streamed_gpu_traversal_seed(
                num_particles=n_particles
            )
        if (
            production_large_n
            and backend_name == "gpu"
            and traversal_config is not None
        ):
            # Production large-N radix path should not allow oversized explicit
            # traversal seeds to inflate memory footprint. Keep user overrides
            # only within the bounded streamed minimum-memory ceiling.
            explicit_ceiling = _minimum_memory_streamed_gpu_traversal_ceiling(
                num_particles=n_particles
            )
            traversal_config = DualTreeTraversalConfig(
                max_pair_queue=int(
                    min(
                        int(traversal_config.max_pair_queue),
                        int(explicit_ceiling.max_pair_queue),
                    )
                ),
                process_block=int(
                    min(
                        int(traversal_config.process_block),
                        int(explicit_ceiling.process_block),
                    )
                ),
                max_interactions_per_node=int(
                    min(
                        int(traversal_config.max_interactions_per_node),
                        int(explicit_ceiling.max_interactions_per_node),
                    )
                ),
                max_neighbors_per_leaf=int(
                    min(
                        int(traversal_config.max_neighbors_per_leaf),
                        int(explicit_ceiling.max_neighbors_per_leaf),
                    )
                ),
            )
        if (
            minimum_memory
            and backend_name == "gpu"
            and self.tree_type == "radix"
            and self.expansion_basis == "solidfmm"
            and bool(self.streamed_far_pairs)
            and not bool(grouped_interactions)
            and bool(self.fail_fast)
            and not self._explicit_traversal_config
            and not self._explicit_max_pair_queue
            and not self._explicit_pair_process_block
            and traversal_config is not None
            and n_particles >= 1_048_576
        ):
            explicit_ceiling = _minimum_memory_streamed_gpu_traversal_ceiling(
                num_particles=n_particles
            )
            capped_queue = min(
                int(traversal_config.max_pair_queue),
                int(explicit_ceiling.max_pair_queue),
            )
            capped_block = min(
                int(traversal_config.process_block),
                int(explicit_ceiling.process_block),
            )
            capped_interactions = min(
                int(traversal_config.max_interactions_per_node),
                int(explicit_ceiling.max_interactions_per_node),
            )
            capped_neighbors = min(
                int(traversal_config.max_neighbors_per_leaf),
                int(explicit_ceiling.max_neighbors_per_leaf),
            )
            traversal_config = DualTreeTraversalConfig(
                max_pair_queue=int(capped_queue),
                process_block=int(capped_block),
                max_interactions_per_node=int(capped_interactions),
                max_neighbors_per_leaf=int(capped_neighbors),
            )
        if grouped_interactions:
            center_mode = "aabb"
            if farfield_mode == "auto":
                if minimum_memory:
                    farfield_mode = "pair_grouped"
                else:
                    farfield_mode = (
                        "class_major"
                        if (class_major_cpu or class_major_gpu)
                        else "pair_grouped"
                    )
        else:
            farfield_mode = "pair_grouped"

        if minimum_memory and not self._explicit_m2l_chunk_size:
            m2l_chunk_size = (
                _MINIMUM_MEMORY_GPU_M2L_CHUNK_SIZE
                if backend_name == "gpu"
                else _MINIMUM_MEMORY_CPU_M2L_CHUNK_SIZE
            )

        return _RuntimeExecutionOverrides(
            traversal_config=traversal_config,
            m2l_chunk_size=m2l_chunk_size,
            l2l_chunk_size=l2l_chunk_size,
            grouped_interactions=grouped_interactions,
            farfield_mode=farfield_mode,
            center_mode=center_mode,
            refine_local_override=refine_local_override,
            adaptive_applied=adaptive_applied,
        )

    def _resolve_tracing_traversal_config(
        self,
        *,
        traversal_config: Optional[DualTreeTraversalConfig],
    ) -> Optional[DualTreeTraversalConfig]:
        """Clamp traced traversal capacities to avoid pathological padding.

        Applies only when prepare_state runs under tracing and the user did not
        explicitly provide traversal_config overrides.
        """

        if traversal_config is None or self._explicit_traversal_config:
            return traversal_config

        current_queue = int(traversal_config.max_pair_queue)
        capped_queue = min(current_queue, _TRACING_MAX_PAIR_QUEUE)
        current_block = int(traversal_config.process_block)
        capped_block = min(current_block, _TRACING_MAX_PROCESS_BLOCK)
        current_neighbors = int(traversal_config.max_neighbors_per_leaf)
        capped_neighbors = min(current_neighbors, _TRACING_MAX_NEIGHBORS_PER_LEAF)
        current_interactions = int(traversal_config.max_interactions_per_node)
        capped_interactions = min(
            current_interactions, _TRACING_MAX_INTERACTIONS_PER_NODE
        )
        if (
            capped_queue == current_queue
            and capped_block == current_block
            and capped_neighbors == current_neighbors
            and capped_interactions == current_interactions
        ):
            return traversal_config

        return DualTreeTraversalConfig(
            max_pair_queue=int(capped_queue),
            process_block=int(capped_block),
            max_interactions_per_node=int(capped_interactions),
            max_neighbors_per_leaf=int(capped_neighbors),
        )

    def _resolve_nearfield_mode(self, *, num_particles: int) -> str:
        """Resolve near-field execution mode from configured policy."""
        if self._is_large_n_gpu_production_profile():
            if (
                not bool(self._explicit_nearfield_mode)
                and jax.default_backend() == "gpu"
                and int(num_particles) < 262_144
            ):
                return "baseline"
            return "bucketed"
        mode = str(self.nearfield_mode).strip().lower()
        if mode != "auto":
            return mode
        backend = jax.default_backend()
        large_gpu = (
            backend == "gpu"
            and int(num_particles) >= 262_144
            and str(self.preset).strip().lower() == "large_n_gpu"
            and str(self.expansion_basis).strip().lower() == "solidfmm"
        )
        large_cpu = (
            backend == "cpu"
            and int(num_particles) >= _NEARFIELD_BUCKETED_CPU_PARTICLE_THRESHOLD
        )
        if large_gpu:
            return "bucketed"
        if (
            large_cpu
            and self.preset == "fast"
            and self.expansion_basis == "solidfmm"
            and self.mac_type == "dehnen"
        ):
            return "bucketed"
        return "baseline"

    def _resolve_nearfield_edge_chunk_size(
        self,
        *,
        num_particles: int,
        nearfield_mode: str,
    ) -> int:
        """Resolve near-field edge chunk size with large-N auto policy."""
        base_chunk = int(self.nearfield_edge_chunk_size)
        if base_chunk <= 0:
            raise ValueError("nearfield_edge_chunk_size must be positive")
        mode = str(self.nearfield_mode).strip().lower()
        auto_policy_enabled = mode == "auto" or (
            self._is_large_n_gpu_production_profile()
            and not bool(self._explicit_nearfield_mode)
        )
        if (not auto_policy_enabled) or str(
            nearfield_mode
        ).strip().lower() != "bucketed":
            return base_chunk

        n = int(num_particles)
        if jax.default_backend() == "gpu":
            if (
                str(self.preset).strip().lower() == "large_n_gpu"
                and str(self.expansion_basis).strip().lower() == "solidfmm"
            ):
                if n >= 262_144:
                    return max(base_chunk, 256)
            return base_chunk

        if n >= 2_000_000:
            return max(base_chunk, _NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_XL)
        if n >= 1_000_000:
            return max(base_chunk, _NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_LARGE)
        if n >= _NEARFIELD_BUCKETED_CPU_PARTICLE_THRESHOLD:
            return max(base_chunk, _NEARFIELD_BUCKETED_CPU_EDGE_CHUNK_MEDIUM)
        return base_chunk
