"""OverridesMixin: fmm_overrides methods extracted from the FastMultipoleMethod
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot.config import TRAVERSAL_OVERRIDE_FIELDS, TraversalOverrides

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

# Merge base for a field-by-field override that arrives on a route with no
# resolved config at all (no preset, no policy default). Only the fields the
# caller did not name are read from here.
_TRAVERSAL_MERGE_FALLBACK = {
    "max_pair_queue": int(_KDTREE_DEFAULT_TRAVERSAL_CONFIG.max_pair_queue),
    "process_block": int(_KDTREE_DEFAULT_TRAVERSAL_CONFIG.process_block),
    "max_interactions_per_node": int(
        _KDTREE_DEFAULT_TRAVERSAL_CONFIG.max_interactions_per_node
    ),
    "max_neighbors_per_leaf": int(
        _KDTREE_DEFAULT_TRAVERSAL_CONFIG.max_neighbors_per_leaf
    ),
}


def normalize_traversal_config_request(
    traversal_config: Any,
) -> tuple[Optional[DualTreeTraversalConfig], dict[str, int]]:
    """Split a ``traversal_config`` request into (full replacement, field merge).

    Returns ``(full, fields)`` where at most one is non-empty:

    * a ``DualTreeTraversalConfig`` instance is a **full replacement** -- the
      historical behaviour, kept so existing callers are unaffected. It comes
      back as ``full`` with an empty ``fields``, and the caller warns.
    * a :class:`~jaccpot.config.TraversalOverrides` or a ``Mapping`` of the same
      keys is a **field merge**: it comes back as ``fields``, with ``full`` None,
      to be applied on top of whatever capacities the runtime resolved for the
      current particle count.
    * ``None`` (or an all-``None`` ``TraversalOverrides``) is neither.

    Rejecting an unknown mapping key by name matters more than it looks: a typo
    in a dict key would otherwise be a silently ignored tuning request, which is
    the exact failure this whole seam exists to prevent.
    """

    if traversal_config is None:
        return None, {}
    if isinstance(traversal_config, DualTreeTraversalConfig):
        return traversal_config, {}
    if isinstance(traversal_config, TraversalOverrides):
        return None, traversal_config.as_dict()
    if isinstance(traversal_config, Mapping):
        unknown = sorted(
            set(map(str, traversal_config.keys())) - set(TRAVERSAL_OVERRIDE_FIELDS)
        )
        if unknown:
            known = ", ".join(TRAVERSAL_OVERRIDE_FIELDS)
            raise ValueError(
                f"traversal_config has unknown field(s) {unknown}; "
                f"valid fields are: {known}"
            )
        fields: dict[str, int] = {}
        for name in TRAVERSAL_OVERRIDE_FIELDS:
            value = traversal_config.get(name)
            if value is None:
                continue
            if int(value) < 1:
                raise ValueError(
                    f"traversal_config['{name}'] must be >= 1 when set; got {value!r}"
                )
            fields[name] = int(value)
        return None, fields
    raise TypeError(
        "traversal_config must be a DualTreeTraversalConfig (full replacement), a "
        "jaccpot.config.TraversalOverrides, a mapping of "
        f"{{{', '.join(TRAVERSAL_OVERRIDE_FIELDS)}}}, or None; got "
        f"{type(traversal_config).__name__}"
    )


def warn_full_traversal_config_replacement(
    *,
    supplied: DualTreeTraversalConfig,
    preset_name: Optional[str],
) -> None:
    """Warn that a full ``DualTreeTraversalConfig`` replaces the preset's sizing.

    Not a deprecation: a full config remains the right tool when a caller has
    measured all four capacities together. But the failure mode it caused
    (measured 3x slower at N=65536 from an override intended to be a no-op) was
    silent, and the fix for silence is to say something.
    """

    warnings.warn(
        "traversal_config was given as a full DualTreeTraversalConfig"
        + (f" alongside preset={preset_name!r}" if preset_name else "")
        + ", which replaces ALL FOUR traversal capacities "
        f"(max_pair_queue={int(supplied.max_pair_queue)}, "
        f"process_block={int(supplied.process_block)}, "
        f"max_interactions_per_node={int(supplied.max_interactions_per_node)}, "
        f"max_neighbors_per_leaf={int(supplied.max_neighbors_per_leaf)}) and "
        "switches off the particle-count-dependent sizing the preset would have "
        "applied. To change one capacity and leave the rest to the preset, pass "
        "jaccpot.config.TraversalOverrides(...) instead.",
        UserWarning,
        stacklevel=3,
    )


class OverridesMixin:
    def _apply_traversal_field_overrides(
        self,
        traversal_config: Optional[DualTreeTraversalConfig],
    ) -> Optional[DualTreeTraversalConfig]:
        """Merge the caller's named capacities onto a runtime-resolved config.

        Applied last, after every policy clamp, so a field the caller named wins
        and a field they did not name keeps the value the preset resolved for
        this N. Idempotent, so calling it on more than one route is safe.
        """

        fields = getattr(self, "_traversal_field_overrides", None)
        if not fields:
            return traversal_config
        if traversal_config is None:
            # No base to merge onto (no preset config, and no policy default on
            # this route). Fall back to the yggdrax defaults for what the caller
            # did not name; a partial request must not become a shape error.
            base = dict(_TRAVERSAL_MERGE_FALLBACK)
        else:
            base = {
                name: int(getattr(traversal_config, name))
                for name in TRAVERSAL_OVERRIDE_FIELDS
            }
        base.update(fields)
        if traversal_config is not None and all(
            base[name] == int(getattr(traversal_config, name))
            for name in TRAVERSAL_OVERRIDE_FIELDS
        ):
            return traversal_config
        return DualTreeTraversalConfig(**base)

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

    def _clamp_gpu_traversal_config_for_memory(
        self,
        *,
        traversal_config: Optional[DualTreeTraversalConfig],
        backend_name: str,
        n_particles: int,
        minimum_memory: bool,
        production_large_n: bool,
        grouped_interactions: bool,
        honor_explicit_traversal: bool = False,
    ) -> Optional[DualTreeTraversalConfig]:
        """Apply the deterministic GPU traversal memory-safety caps.

        These caps are closed-form in ``num_particles`` (no count-pass kernel):
        they bound the streamed GPU traversal buffers on the minimum-memory /
        large-N production lane. They are *memory-safety* clamps, not adaptive
        performance rewrites, so they must apply even under static fixed sizing
        -- otherwise an oversized preset/explicit seed (e.g. the large_n_gpu
        262144 pair-queue default) reaches the GPU traversal build unclamped and
        inflates device-memory footprint (a minimum-memory OOM regression).
        """

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
            and not (self._explicit_traversal_config and honor_explicit_traversal)
            and traversal_config is not None
        ):
            # Bound the production large-N radix traversal to the streamed
            # minimum-memory ceiling so an oversized *preset* seed cannot inflate the
            # device footprint. But when the caps were supplied EXPLICITLY and the
            # caller is in static fixed-sizing mode (``honor_explicit_traversal``),
            # pass them through unclamped: static sizing means "use the sizes I gave
            # you", and this ceiling is a fixed constant (524288 pair-queue) far below
            # the frontier a concentrated multi-million-particle disk needs -- clamping
            # an explicit cap there turned a deliberate, memory-fitting override into a
            # fail-fast traversal overflow at N >= ~2M on >=40 GB GPUs. Auto-sized
            # (non-explicit) preset seeds are still bounded here, and the adaptive path
            # (static sizing off) still clamps explicit caps for adaptive memory mgmt.
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
        return traversal_config

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
            # Opt-in geometric (box/aabb) centres for the real-basis fast lane,
            # decoupled from grouped_interactions (which stays False here to keep
            # the streamed pair_grouped near/far payload). Centres flow
            # upward->M2L->L2L->L2P via upward.multipoles.centers. Default OFF.
            if (
                bool(getattr(self, "_fastlane_geometric_centers", False))
                and self._solidfmm_basis_mode() == "real"
            ):
                center_mode = "aabb"

        if static_runtime_fixed_sizing:
            # Static sizing mode: keep traversal/chunk execution knobs fixed to
            # constructor/global-input values and skip adaptive runtime rewrites.
            if not self._explicit_grouped_interactions:
                # Auto-enabling the grouped/class-major M2L above is itself an
                # adaptive rewrite, so static sizing must not inherit it. It used
                # to leak through at n >= _GPU_LARGE_PARTICLE_THRESHOLD and left
                # two invariants broken: ``farfield_mode`` stayed at "auto" (the
                # grouped branch of _solidfmm_downward_accumulate_from_multipoles
                # rejects that, so prepare_state/compute_accelerations raised), and
                # it was decoupled from the geometric centres the grouped
                # classification requires -- that path quantises pair
                # displacements onto a lattice and applies ONE representative
                # displacement per class, which is only valid with
                # ``center_mode="aabb"`` (see the adaptive branch below).
                grouped_interactions = False
            if self.streamed_far_pairs and grouped_interactions:
                grouped_interactions = False
            if grouped_interactions:
                center_mode = "aabb"
                if farfield_mode == "auto":
                    farfield_mode = (
                        "pair_grouped"
                        if minimum_memory
                        else (
                            "class_major"
                            if (class_major_cpu or class_major_gpu)
                            else "pair_grouped"
                        )
                    )
            else:
                farfield_mode = "pair_grouped"
            # Deterministic GPU memory-safety traversal caps are NOT adaptive
            # rewrites -- they bound the streamed GPU traversal buffers on the
            # minimum-memory / large-N production lane and must still apply here,
            # otherwise an oversized *preset* seed reaches the GPU build unclamped
            # and inflates device memory (a minimum-memory OOM regression). Static
            # fixed sizing keeps EXPLICIT caps as-given (honor_explicit_traversal),
            # so a caller who sized the traversal for their GPU is not clamped into
            # a fail-fast overflow at large N.
            traversal_config = self._clamp_gpu_traversal_config_for_memory(
                traversal_config=traversal_config,
                backend_name=backend_name,
                n_particles=n_particles,
                minimum_memory=minimum_memory,
                production_large_n=production_large_n,
                grouped_interactions=grouped_interactions,
                honor_explicit_traversal=True,
            )
            # Last, after every clamp: see the matching call on the adaptive
            # return below. Both exits of this method must merge, or which
            # capacities a caller's override reaches would depend on
            # JACCPOT_STATIC_RUNTIME_FIXED_SIZING.
            traversal_config = self._apply_traversal_field_overrides(traversal_config)
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
        traversal_config = self._clamp_gpu_traversal_config_for_memory(
            traversal_config=traversal_config,
            backend_name=backend_name,
            n_particles=n_particles,
            minimum_memory=minimum_memory,
            production_large_n=production_large_n,
            grouped_interactions=grouped_interactions,
        )
        # Last, after every clamp: a capacity the caller named explicitly wins,
        # and one they did not name keeps the value resolved above for this N.
        traversal_config = self._apply_traversal_field_overrides(traversal_config)
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
        if getattr(self, "_traversal_field_overrides", None):
            # A named capacity is an explicit request on this route too: capping
            # it back down to the tracing ceiling would reintroduce exactly the
            # silent substitution this seam removes. Fields the caller did not
            # name are still clamped, below.
            return self._apply_traversal_field_overrides(
                self._resolve_tracing_traversal_config_unchecked(
                    traversal_config=traversal_config
                )
            )
        return self._resolve_tracing_traversal_config_unchecked(
            traversal_config=traversal_config
        )

    def _resolve_tracing_traversal_config_unchecked(
        self,
        *,
        traversal_config: DualTreeTraversalConfig,
    ) -> DualTreeTraversalConfig:
        """Apply the traced-capacity ceilings unconditionally."""

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
        """Resolve near-field execution mode from configured policy.

        The ``"auto"`` GPU answer is ``"bucketed"`` at every N, because
        ``"baseline"`` is a ``lax.scan`` over leaf pairs *one pair at a time*
        with a ``lax.cond`` per pair. That is the readable reference traversal
        and it is competitive on a CPU, but on a GPU it serialises one launch
        per leaf pair. Measured on an idle A100 at N=4096, leaf 64, p=4, real
        basis, ``preset="accurate"``: 5.23 s baseline against 0.0103 s bucketed,
        a factor of **507**, which is what made ``accurate`` on an A100 ~30x
        slower than ``accurate`` on the CPU beside it.

        The two traversals visit the same leaf pairs, so this is a scheduling
        choice and not an accuracy one. Measured at fp64 on CPU, N=8192/leaf 64:
        both sit at 4.4075521942e-05 rel-L2 against a direct O(N^2) sum
        (agreeing to 13 significant figures), and differ from each other by
        4.2e-16 -- one ulp of reassociation.

        Deliberately GPU-only: on a CPU the per-pair scan has no launch cost to
        pay and the existing CPU crossovers below stay as they were measured.
        """

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
        if backend == "gpu":
            # See the docstring: the per-pair baseline scan cannot win on a GPU
            # at any size, so there is no crossover to tune here.
            return "bucketed"
        large_cpu = (
            backend == "cpu"
            and int(num_particles) >= _NEARFIELD_BUCKETED_CPU_PARTICLE_THRESHOLD
        )
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
