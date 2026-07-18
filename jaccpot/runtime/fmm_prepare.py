"""PrepareMixin: fmm_prepare methods extracted from the FastMultipoleMethod
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Callable, Tuple
from jaxtyping import Array, jaxtyped
from yggdrax.dense_interactions import DenseInteractionBuffers
from yggdrax.geometry import compute_tree_geometry
from yggdrax.grouped_interactions import GroupedInteractionBuffers
from yggdrax.interactions import (
    CompactTaggedFarPairs,
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    DualTreeWalkResult,
    MACType,
    NodeInteractionList,
    NodeNeighborList,
    build_octree_native_far_pairs,
    build_octree_native_neighbor_lists,
)
from yggdrax.morton import morton_encode
from yggdrax.tree import (
    RadixTree,
    Tree,
    rebuild_static_radix_tree_from_template,
    reorder_particles_by_indices,
)

from jaccpot.downward.local_expansions import (
    LocalExpansionData,
    TreeDownwardData,
    initialize_local_expansions,
)
from jaccpot.nearfield.near_field import (
    prepare_bucketed_scatter_schedules,
    prepare_bucketed_scatter_schedules_from_groups,
    prepare_leaf_neighbor_pairs,
)
from jaccpot.operators.multipole_utils import MAX_MULTIPOLE_ORDER, total_coefficients
from jaccpot.upward.tree_expansions import TreeUpwardData

from ._adaptive_policy import (
    adaptive_pair_policy,
    adaptive_policy_tolerance,
    bucket_far_pairs_by_tag,
)
from ._interaction_cache import (
    _build_dual_tree_artifacts,
    _compiled_refresh_dual_planner_route,
    _DualTreeArtifacts,
    _interaction_cache_key,
    _InteractionCacheEntry,
    _RefreshDualPlannerHint,
)
from ._large_n_pipeline import can_use_large_n_prepare_path, prepare_large_n_state
from ._nearfield_cache import (
    NearfieldPrecomputeArtifacts,
    nearfield_cache_matches,
    nearfield_from_cache,
    with_nearfield_cache_artifacts,
)
from ._octree_adapter import build_octree_execution_data_with_status
from .dtypes import INDEX_DTYPE
from .fmm_caches import _contains_tracer, _estimate_payload_nbytes, _format_nbytes
from .fmm_constants import (
    _NEARFIELD_GPU_PRECOMPUTE_MAX_PARTICLES,
    _NEARFIELD_SCATTER_SCHEDULE_INT32_ITEM_LIMIT,
    _NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP,
    _NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP_GPU,
    _cap_minimum_memory_streamed_gpu_traversal_config_for_tree,
    _prepare_diag,
)
from .fmm_state import (
    FMMPreparedState,
    TreeBuilderConfig,
    _bucket_far_pairs_by_level_split,
    _build_octree_downward_artifacts,
    _build_octree_upward_artifacts,
    _build_tree_with_config,
    _empty_interaction_storage_like,
    _finalize_octree_downward_artifacts,
    _GeometryReuseEntry,
    _prepared_state_octree_upward_payload,
    _prepared_state_upward_payload,
    _PrepareStateDualDownwardArtifacts,
    _PrepareStateFarPairPlan,
    _PrepareStateTreeUpwardArtifacts,
    _RuntimeExecutionOverrides,
    _TopologyReuseCandidate,
    _TopologyReuseEntry,
    _TreeBuildArtifacts,
)
from .kernels.core import (
    NearfieldInteropData,
    _build_nearfield_interop_data,
    _empty_interaction_storage_for_tree,
    _FarPairCOO,
    _infer_bounds,
)


class PrepareMixin:
    def _validate_prepare_state_request(
        self,
        *,
        leaf_size: int,
        max_order: int,
    ) -> None:
        """Validate order/leaf-size constraints for state preparation."""
        if leaf_size < 1:
            raise ValueError("leaf_size must be >= 1")
        if self.fixed_order is not None and int(self.fixed_order) != int(max_order):
            raise ValueError("fixed_order must match max_order")
        if max_order > MAX_MULTIPOLE_ORDER and self.expansion_basis != "solidfmm":
            raise NotImplementedError(
                "orders above 4 require expansion_basis='solidfmm'",
            )

    def _prepare_state_input_arrays(
        self,
        positions: Array,
        masses: Array,
    ) -> tuple[Array, Array, Any]:
        """Validate prepare-state inputs and cast them to the working dtype."""
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        input_dtype = positions_arr.dtype

        if positions_arr.ndim != 2 or positions_arr.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if masses_arr.shape != (positions_arr.shape[0],):
            raise ValueError("masses must have shape (N,)")
        if positions_arr.shape[0] == 0:
            raise ValueError("need at least one particle")

        working_dtype = self.working_dtype
        if working_dtype is not None and positions_arr.dtype != working_dtype:
            positions_arr = positions_arr.astype(working_dtype)
        if working_dtype is not None and masses_arr.dtype != working_dtype:
            masses_arr = masses_arr.astype(working_dtype)
        return positions_arr, masses_arr, input_dtype

    def _resolve_prepare_state_bounds(
        self,
        *,
        positions: Array,
        bounds: Optional[Tuple[Array, Array]],
    ) -> tuple[Array, Array]:
        """Return bounds converted to the working dtype or infer them."""
        if bounds is None:
            min_corner, max_corner = _infer_bounds(positions)
            return min_corner, max_corner
        min_corner, max_corner = bounds
        return (
            jnp.asarray(min_corner, dtype=positions.dtype),
            jnp.asarray(max_corner, dtype=positions.dtype),
        )

    def _topology_reuse_candidate(
        self,
        *,
        positions: Array,
        bounds: Tuple[Array, Array],
        tree_config: TreeBuilderConfig,
        leaf_size: int,
        refine_local: bool,
        max_refine_levels: int,
        aspect_threshold: float,
        allow_stateful_cache: bool,
    ) -> Optional[_TopologyReuseCandidate]:
        """Return a radix-topology reuse signature when host-side caching is safe."""

        if (
            (not self.reuse_topology and tree_config.mode != "static_radix")
            or not allow_stateful_cache
            or self.tree_type != "radix"
        ):
            return None
        try:
            morton_codes = morton_encode(positions, bounds)
            orig_idx = jnp.arange(positions.shape[0], dtype=INDEX_DTYPE)
            sorted_indices = jnp.lexsort((orig_idx, morton_codes))
            sorted_codes = morton_codes[sorted_indices]
            if tree_config.mode == "static_radix":
                num_leaves = (int(positions.shape[0]) + int(leaf_size) - 1) // int(
                    leaf_size
                )
                hasher = hashlib.sha256()
                hasher.update(b"static_radix_topology_v1")
                hasher.update(
                    np.asarray(int(positions.shape[0]), dtype=np.int64).tobytes()
                )
                hasher.update(
                    np.asarray(max(2 * num_leaves - 1, 1), dtype=np.int64).tobytes()
                )
                hasher.update(
                    np.asarray(max(num_leaves - 1, 0), dtype=np.int64).tobytes()
                )
                hasher.update(np.asarray(int(leaf_size), dtype=np.int64).tobytes())
                key = hasher.hexdigest()
            else:
                key = self._topology_reuse_key_from_sorted_codes(
                    sorted_codes=sorted_codes,
                    tree_config=tree_config,
                    leaf_size=leaf_size,
                    refine_local=refine_local,
                    max_refine_levels=max_refine_levels,
                    aspect_threshold=aspect_threshold,
                )
        except Exception:
            return None
        if key is None:
            return None
        return _TopologyReuseCandidate(
            key=key,
            sorted_indices=jnp.asarray(sorted_indices, dtype=INDEX_DTYPE),
            sorted_codes=jnp.asarray(sorted_codes),
            bounds=bounds,
        )

    def _topology_reuse_key_from_sorted_codes(
        self,
        *,
        sorted_codes: Array,
        tree_config: TreeBuilderConfig,
        leaf_size: int,
        refine_local: bool,
        max_refine_levels: int,
        aspect_threshold: float,
    ) -> Optional[str]:
        """Build a stable topology key from sorted Morton codes and tree options."""

        try:
            hasher = hashlib.sha256()
            hasher.update(
                np.asarray(jax.device_get(sorted_codes), dtype=np.uint64).tobytes()
            )
            hasher.update(str(tree_config.mode).encode("utf8"))
            leaf_param = (
                int(leaf_size)
                if tree_config.mode == "lbvh"
                else int(tree_config.target_leaf_particles)
            )
            hasher.update(np.asarray(leaf_param, dtype=np.int64).tobytes())
            hasher.update(np.asarray(int(bool(refine_local)), dtype=np.int64).tobytes())
            hasher.update(np.asarray(int(max_refine_levels), dtype=np.int64).tobytes())
            hasher.update(
                np.asarray(float(aspect_threshold), dtype=np.float64).tobytes()
            )
        except Exception:
            return None
        return hasher.hexdigest()

    def _static_radix_topology_key_from_tree(
        self,
        tree: RadixTree,
        *,
        leaf_size: int,
    ) -> Optional[str]:
        """Return a stable key for a static-radix data-structure shape."""

        try:
            hasher = hashlib.sha256()
            hasher.update(b"static_radix_topology_v1")
            hasher.update(np.asarray(int(tree.num_particles), dtype=np.int64).tobytes())
            hasher.update(
                np.asarray(int(tree.parent.shape[0]), dtype=np.int64).tobytes()
            )
            hasher.update(
                np.asarray(int(tree.num_internal_nodes), dtype=np.int64).tobytes()
            )
            hasher.update(np.asarray(int(leaf_size), dtype=np.int64).tobytes())
        except Exception:
            return None
        return hasher.hexdigest()

    def _rebuild_tree_artifacts_from_topology(
        self,
        *,
        candidate: _TopologyReuseCandidate,
        entry: _TopologyReuseEntry,
        positions: Array,
        masses: Array,
    ) -> _TreeBuildArtifacts:
        """Reorder particles and attach them to a cached radix topology."""

        positions_sorted, masses_sorted, inverse = reorder_particles_by_indices(
            positions,
            masses,
            candidate.sorted_indices,
        )
        cached_tree = entry.tree
        if not isinstance(cached_tree, RadixTree):
            raise ValueError("topology reuse currently supports radix trees only")
        topology = cached_tree.topology
        if (
            cached_tree.build_mode == "static_radix"
            and candidate.sorted_codes is not None
            and candidate.bounds is not None
        ):
            num_internal = int(cached_tree.num_internal_nodes)
            leaf_starts = jnp.asarray(
                cached_tree.node_ranges[num_internal:, 0],
                dtype=INDEX_DTYPE,
            )
            topology = topology._replace(
                particle_indices=jnp.asarray(
                    candidate.sorted_indices,
                    dtype=INDEX_DTYPE,
                ),
                morton_codes=jnp.asarray(candidate.sorted_codes),
                bounds_min=jnp.asarray(candidate.bounds[0], dtype=positions.dtype),
                bounds_max=jnp.asarray(candidate.bounds[1], dtype=positions.dtype),
                leaf_codes=jnp.asarray(candidate.sorted_codes)[leaf_starts],
                leaf_depths=jnp.full(
                    (leaf_starts.shape[0],),
                    -1,
                    dtype=INDEX_DTYPE,
                ),
                use_morton_geometry=jnp.asarray(False, dtype=jnp.bool_),
            )
        rebuilt_tree = RadixTree(
            topology=topology,
            build_mode=cached_tree.build_mode,
            positions_sorted=positions_sorted,
            masses_sorted=masses_sorted,
            inverse_permutation=inverse,
            workspace=cached_tree.workspace,
        )
        return _TreeBuildArtifacts(
            tree=rebuilt_tree,
            positions_sorted=positions_sorted,
            masses_sorted=masses_sorted,
            inverse_permutation=inverse,
            workspace=cached_tree.workspace,
            max_leaf_size=int(entry.max_leaf_size),
            cache_leaf_parameter=int(entry.cache_leaf_parameter),
        )

    def _rebuild_tree_artifacts_from_static_template(
        self,
        *,
        template_tree: RadixTree,
        positions: Array,
        masses: Array,
        bounds: Optional[Tuple[Array, Array]],
        max_leaf_size: int,
        cache_leaf_parameter: int,
    ) -> _TreeBuildArtifacts:
        """Refresh static-radix tree artifacts from a fixed template topology."""

        rebuilt_result = rebuild_static_radix_tree_from_template(
            positions,
            masses,
            template_tree,
            bounds=bounds,
            return_reordered=True,
        )
        if not isinstance(rebuilt_result, tuple) or len(rebuilt_result) != 4:
            raise RuntimeError(
                "static radix template rebuild must return tree and reordered arrays"
            )
        rebuilt_tree, positions_sorted, masses_sorted, inverse = rebuilt_result
        if not isinstance(rebuilt_tree, RadixTree):
            raise ValueError("static radix template rebuild returned non-radix tree")
        return _TreeBuildArtifacts(
            tree=rebuilt_tree,
            positions_sorted=jnp.asarray(positions_sorted, dtype=positions.dtype),
            masses_sorted=jnp.asarray(masses_sorted, dtype=masses.dtype),
            inverse_permutation=jnp.asarray(inverse, dtype=INDEX_DTYPE),
            workspace=template_tree.workspace,
            max_leaf_size=int(max_leaf_size),
            cache_leaf_parameter=int(cache_leaf_parameter),
        )

    def _build_locals_template_for_prepare_state(
        self,
        *,
        tree: Tree,
        upward: TreeUpwardData,
        max_order: int,
        pos_sorted: Array,
    ) -> Optional[LocalExpansionData]:
        """Build initial local-expansion buffers matching the active basis."""
        if self.expansion_basis == "solidfmm":
            # Solid-FMM does not reuse a cached locals template across prepare calls.
            # Let the downward pass allocate its accumulator on demand so we do not
            # hold a redundant zero buffer alive across dual-tree staging.
            return None

        template = self._locals_template
        total_nodes = int(tree.parent.shape[0])
        coeff_count = total_coefficients(max_order)
        reuse_template = (
            template is not None
            and int(template.order) == max_order
            and template.coefficients.shape == (total_nodes, coeff_count)
            and template.coefficients.dtype == pos_sorted.dtype
        )

        if reuse_template:
            return LocalExpansionData(
                order=max_order,
                centers=upward.multipoles.centers,
                coefficients=jnp.zeros_like(template.coefficients),
            )
        return initialize_local_expansions(
            tree,
            upward.multipoles.centers,
            max_order=max_order,
        )

    def _prepare_state_tree_and_upward(
        self,
        *,
        positions_arr: Array,
        masses_arr: Array,
        bounds: Optional[Tuple[Array, Array]],
        leaf_size: int,
        max_order: int,
        refine_local_val: bool,
        max_refine_levels_val: int,
        aspect_threshold_val: float,
        jit_tree_override: Optional[bool],
        upward_center_mode: str,
        allow_stateful_cache: bool,
    ) -> _PrepareStateTreeUpwardArtifacts:
        """Build tree artifacts and run upward preparation for prepare_state."""
        tree_config = self.config.tree
        if self.tree_type != "radix" and tree_config.mode in (
            "fixed_depth",
            "static_radix",
        ):
            tree_config = TreeBuilderConfig(
                mode="lbvh",
                target_leaf_particles=tree_config.target_leaf_particles,
                refine_local=tree_config.refine_local,
                max_refine_levels=tree_config.max_refine_levels,
                aspect_threshold=tree_config.aspect_threshold,
            )

        inferred_bounds = self._resolve_prepare_state_bounds(
            positions=positions_arr,
            bounds=bounds,
        )
        jit_tree_flag = self._resolve_jit_tree_flag(
            positions_arr,
            jit_tree_override=jit_tree_override,
        )

        self._recent_topology_reused = False
        cached_topology = self._topology_reuse_entry
        topology_candidate = None
        if cached_topology is not None:
            topology_candidate = self._topology_reuse_candidate(
                positions=positions_arr,
                bounds=inferred_bounds,
                tree_config=tree_config,
                leaf_size=int(leaf_size),
                refine_local=refine_local_val,
                max_refine_levels=max_refine_levels_val,
                aspect_threshold=aspect_threshold_val,
                allow_stateful_cache=allow_stateful_cache,
            )
        can_reuse_cached_topology = (
            topology_candidate is not None
            and cached_topology is not None
            and topology_candidate.key == cached_topology.key
            and cached_topology.reuse_count < (self.rebuild_every - 1)
        )
        topology_key_for_state: Optional[str] = None

        tree_build_t0 = time.perf_counter()
        if can_reuse_cached_topology:
            build_artifacts = self._rebuild_tree_artifacts_from_topology(
                candidate=topology_candidate,
                entry=cached_topology,
                positions=positions_arr,
                masses=masses_arr,
            )
            self._recent_topology_reused = True
            topology_key_for_state = topology_candidate.key
            self._topology_reuse_entry = _TopologyReuseEntry(
                key=cached_topology.key,
                tree=build_artifacts.tree,
                max_leaf_size=int(cached_topology.max_leaf_size),
                cache_leaf_parameter=int(cached_topology.cache_leaf_parameter),
                reuse_count=int(cached_topology.reuse_count) + 1,
            )
        else:
            build_artifacts = _build_tree_with_config(
                positions_arr,
                masses_arr,
                inferred_bounds,
                tree_type=self.tree_type,
                tree_config=tree_config,
                leaf_size=int(leaf_size),
                workspace=(self._tree_workspace if allow_stateful_cache else None),
                jit_tree=jit_tree_flag,
                refine_local=refine_local_val,
                max_refine_levels=max_refine_levels_val,
                aspect_threshold=aspect_threshold_val,
            )
            if allow_stateful_cache:
                self._tree_workspace = build_artifacts.workspace
                if tree_config.mode == "static_radix":
                    topology_key_for_state = self._static_radix_topology_key_from_tree(
                        build_artifacts.tree,
                        leaf_size=int(leaf_size),
                    )
                if self.reuse_topology:
                    if topology_candidate is not None:
                        topology_key_for_state = topology_candidate.key
                    elif topology_key_for_state is None:
                        topology_key_for_state = (
                            self._topology_reuse_key_from_sorted_codes(
                                sorted_codes=build_artifacts.tree.morton_codes,
                                tree_config=tree_config,
                                leaf_size=int(leaf_size),
                                refine_local=refine_local_val,
                                max_refine_levels=max_refine_levels_val,
                                aspect_threshold=aspect_threshold_val,
                            )
                        )
                if topology_key_for_state is not None:
                    self._topology_reuse_entry = _TopologyReuseEntry(
                        key=topology_key_for_state,
                        tree=build_artifacts.tree,
                        max_leaf_size=int(build_artifacts.max_leaf_size),
                        cache_leaf_parameter=int(build_artifacts.cache_leaf_parameter),
                        reuse_count=0,
                    )
                elif self.reuse_topology:
                    self._topology_reuse_entry = None
        if bool(getattr(self, "_refresh_timing_active", False)):
            self._refresh_timing_tree_build_seconds += (
                time.perf_counter() - tree_build_t0
            )

        tree = build_artifacts.tree
        pos_sorted = build_artifacts.positions_sorted
        mass_sorted = build_artifacts.masses_sorted
        total_nodes = int(tree.parent.shape[0])
        num_internal = int(jnp.asarray(tree.left_child).shape[0])
        num_leaves = int(total_nodes - num_internal)
        _prepare_diag(
            "tree built "
            f"particles={int(pos_sorted.shape[0])} leaf_size={int(leaf_size)} "
            f"total_nodes={total_nodes} num_internal={num_internal} num_leaves={num_leaves} "
            f"tree_type={self.tree_type} tree_mode={tree_config.mode} jit_tree={bool(jit_tree_flag)}"
        )
        # Keep one leaf-size contract in eager and traced paths.
        leaf_cap_hint = int(build_artifacts.max_leaf_size)
        upward_leaf_batch = (
            int(self.upward_leaf_batch_size)
            if self.upward_leaf_batch_size is not None
            else None
        )
        _prepare_diag(
            "upward start "
            f"max_order={int(max_order)} center_mode={upward_center_mode} "
            f"leaf_cap={leaf_cap_hint} upward_leaf_batch_size={upward_leaf_batch}"
        )
        geometry_cache_key: Optional[tuple[Any, ...]] = None
        cached_geometry = None
        if allow_stateful_cache:
            if topology_key_for_state is not None:
                geometry_key_base: Any = topology_key_for_state
            else:
                bounds_key = (
                    None if bounds is None else (int(id(bounds[0])), int(id(bounds[1])))
                )
                geometry_key_base = (
                    self.tree_type,
                    tree_config.mode,
                    int(leaf_size),
                    bool(refine_local_val),
                    int(max_refine_levels_val),
                    float(aspect_threshold_val),
                    bounds_key,
                )
            geometry_cache_key = (
                geometry_key_base,
                int(id(positions_arr)),
                int(positions_arr.shape[0]),
                str(positions_arr.dtype),
            )
            geometry_entry = self._geometry_reuse_entry
            if geometry_entry is not None and geometry_entry.key == geometry_cache_key:
                cached_geometry = geometry_entry.geometry

        upward_t0 = time.perf_counter()
        defer_geometry = (
            bool(getattr(self, "_refresh_timing_active", False))
            and tree_config.mode == "static_radix"
            and str(upward_center_mode).strip().lower() == "com"
            and self._interaction_cache is not None
        )
        upward = self.prepare_upward_sweep(
            tree,
            pos_sorted,
            mass_sorted,
            max_order=max_order,
            center_mode=upward_center_mode,
            max_leaf_size=leaf_cap_hint,
            precomputed_geometry=cached_geometry,
            defer_geometry=defer_geometry,
        )
        if bool(getattr(self, "_refresh_timing_active", False)):
            self._refresh_timing_upward_compute_seconds += (
                time.perf_counter() - upward_t0
            )
        if (
            allow_stateful_cache
            and geometry_cache_key is not None
            and upward.geometry is not None
        ):
            self._geometry_reuse_entry = _GeometryReuseEntry(
                key=geometry_cache_key,
                geometry=upward.geometry,
            )
        _prepare_diag(
            "upward done "
            f"multipole_order={int(upward.multipoles.order)} "
            f"multipole_shape={tuple(int(v) for v in upward.multipoles.packed.shape)}"
        )
        locals_template = self._build_locals_template_for_prepare_state(
            tree=tree,
            upward=upward,
            max_order=max_order,
            pos_sorted=pos_sorted,
        )

        return _PrepareStateTreeUpwardArtifacts(
            tree_mode=tree_config.mode,
            tree=tree,
            positions_sorted=pos_sorted,
            masses_sorted=mass_sorted,
            inverse_permutation=build_artifacts.inverse_permutation,
            leaf_cap=leaf_cap_hint,
            leaf_parameter=build_artifacts.cache_leaf_parameter,
            topology_key=topology_key_for_state,
            upward=upward,
            locals_template=locals_template,
        )

    def _prepare_state_tree_upward_and_dual_downward(
        self,
        *,
        positions_arr: Array,
        masses_arr: Array,
        bounds: Optional[Tuple[Array, Array]],
        leaf_size: int,
        max_order: int,
        refine_local_val: bool,
        max_refine_levels_val: int,
        aspect_threshold_val: float,
        jit_tree_override: Optional[bool],
        upward_center_mode: str,
        allow_stateful_cache: bool,
        theta_val: float,
        mac_type_val: MACType,
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        record_retry: Callable[[DualTreeRetryEvent], None],
    ) -> tuple[_PrepareStateTreeUpwardArtifacts, _PrepareStateDualDownwardArtifacts]:
        """Build tree/upward and dual/downward artifacts in one helper call."""

        tree_artifacts = self._prepare_state_tree_and_upward(
            positions_arr=positions_arr,
            masses_arr=masses_arr,
            bounds=bounds,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            refine_local_val=bool(refine_local_val),
            max_refine_levels_val=int(max_refine_levels_val),
            aspect_threshold_val=float(aspect_threshold_val),
            jit_tree_override=jit_tree_override,
            upward_center_mode=upward_center_mode,
            allow_stateful_cache=bool(allow_stateful_cache),
        )
        dual_downward_artifacts = self._prepare_state_dual_and_downward(
            tree_artifacts=tree_artifacts,
            force_scale_nodes=None,
            upward_center_mode=upward_center_mode,
            theta_val=float(theta_val),
            mac_type_val=mac_type_val,
            dehnen_radius_scale=self.dehnen_radius_scale,
            runtime_traversal_config=runtime_traversal_config,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            grouped_interactions=False,
            farfield_mode=self.farfield_mode,
            record_retry=record_retry,
            refine_local_val=bool(refine_local_val),
            max_refine_levels_val=int(max_refine_levels_val),
            aspect_threshold_val=float(aspect_threshold_val),
            allow_stateful_cache=bool(allow_stateful_cache),
        )
        return tree_artifacts, dual_downward_artifacts

    def _prepare_state_dual_and_downward(
        self,
        *,
        tree_artifacts: _PrepareStateTreeUpwardArtifacts,
        force_scale_nodes: Optional[Array],
        upward_center_mode: str,
        theta_val: float,
        mac_type_val: MACType,
        dehnen_radius_scale: float,
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        grouped_interactions: bool,
        farfield_mode: str,
        record_retry: Callable[[DualTreeRetryEvent], None],
        refine_local_val: bool,
        max_refine_levels_val: int,
        aspect_threshold_val: float,
        allow_stateful_cache: bool,
        suppress_host_side_effects: bool = False,
    ) -> _PrepareStateDualDownwardArtifacts:
        """Build/reuse interactions and prepare downward artifacts.

        Memory note:
        The dominant warm prepare peak is usually the far-field payload that
        exists long enough to feed M2L: raw interactions or streamed COO pairs,
        optional grouped layouts, and the downward locals buffer. Only the
        returned ``TreeDownwardData`` is meant to survive into prepared state;
        grouped schedules and other M2L feed artifacts are transient and should
        stay scoped to this helper.
        """
        suppress_host_side_effects = bool(suppress_host_side_effects)
        refresh_timing_active = bool(
            getattr(self, "_refresh_timing_active", False)
        ) and (not suppress_host_side_effects)
        dual_total_t0 = time.perf_counter() if refresh_timing_active else 0.0
        dual_stage_sum = 0.0

        def _stage_now() -> float:
            return time.perf_counter() if refresh_timing_active else 0.0

        def _record_dual_stage(attr: str, start: float) -> None:
            nonlocal dual_stage_sum
            if not refresh_timing_active:
                return
            elapsed = float(time.perf_counter() - start)
            dual_stage_sum += elapsed
            setattr(
                self,
                attr,
                float(getattr(self, attr, 0.0)) + elapsed,
            )

        def _record_dual_artifact_substage(name: str, elapsed: float) -> None:
            if not refresh_timing_active:
                return
            attr_by_name = {
                "dual_split_shared_far_pairs_leaf_neighbors": "_refresh_timing_dual_split_shared_far_near_seconds",
                "dual_split_shared_count": "_refresh_timing_dual_split_shared_count_seconds",
                "dual_split_shared_combined_fill": "_refresh_timing_dual_split_shared_combined_fill_seconds",
                "dual_split_shared_far_fill": "_refresh_timing_dual_split_shared_far_fill_seconds",
                "dual_split_shared_near_fill": "_refresh_timing_dual_split_shared_near_fill_seconds",
                "dual_split_far_pairs": "_refresh_timing_dual_split_far_pairs_seconds",
                "dual_split_leaf_neighbors": "_refresh_timing_dual_split_leaf_neighbors_seconds",
                "dual_split_interactions_and_neighbors": "_refresh_timing_dual_split_combined_seconds",
                "dual_raw_interactions_and_neighbors": "_refresh_timing_dual_raw_combined_seconds",
                "dual_split_dense_buffers": "_refresh_timing_dual_split_dense_buffers_seconds",
            }
            attr = attr_by_name.get(str(name))
            if attr is None:
                return
            setattr(self, attr, float(getattr(self, attr, 0.0)) + float(elapsed))

        stage_t0 = _stage_now()
        pair_policy = None
        policy_state = None
        cache_key = None
        use_paper_fixed_policy = (not self.adaptive_order) and (
            self._uses_paper_style_traversal_policy()
        )
        if (
            bool(suppress_host_side_effects)
            and bool(getattr(self, "_strict_fused_mode_active", False))
            and bool(getattr(self, "_strict_fused_device_only", False))
        ):
            use_paper_fixed_policy = False

        # A static-radix topology key describes the capacity-fixed tree shape,
        # not the current leaf membership, geometry, or MAC decisions. Reusing
        # cached dual-tree artifacts across evolved positions can therefore
        # attach stale neighbor/far-field payloads to freshly sorted particles.
        stateful_cache_enabled = (
            bool(allow_stateful_cache)
            and bool(self.enable_interaction_cache)
            and str(tree_artifacts.tree_mode) != "static_radix"
            and (not suppress_host_side_effects)
        )
        grouped_interactions_active = bool(grouped_interactions)
        strict_fused_device_only_hot_path = (
            bool(suppress_host_side_effects)
            and bool(getattr(self, "_strict_fused_mode_active", False))
            and bool(getattr(self, "_strict_fused_device_only", False))
        )
        adaptive_order_active = bool(self.adaptive_order) and not bool(
            strict_fused_device_only_hot_path
        )
        mixed_order_farfield_active = bool(self.mixed_order_farfield) and not bool(
            strict_fused_device_only_hot_path
        )
        retain_interactions_active = bool(self.retain_interactions) and not bool(
            strict_fused_device_only_hot_path
        )
        need_traversal_result = (
            bool(self.retain_traversal_result)
            and not bool(strict_fused_device_only_hot_path)
        ) or bool(use_paper_fixed_policy)
        traced_prepare_inputs = bool(
            _contains_tracer(
                (tree_artifacts.positions_sorted, tree_artifacts.masses_sorted)
            )
        )
        strict_fused_node_interactions_safe_path = (
            bool(strict_fused_device_only_hot_path)
            and str(
                os.environ.get(
                    "JACCPOT_STATIC_STRICT_FUSED_NODE_INTERACTIONS_SAFE_PATH",
                    "0",
                )
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            and not (
                str(
                    os.environ.get(
                        "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE",
                        "0",
                    )
                )
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            )
        )
        use_compact_streamed_pairs = (
            (bool(self.streamed_far_pairs) or bool(strict_fused_device_only_hot_path))
            and not bool(strict_fused_node_interactions_safe_path)
            and not adaptive_order_active
            and not grouped_interactions_active
            and not mixed_order_farfield_active
            and not retain_interactions_active
            and not bool(need_traversal_result)
            and (
                not bool(traced_prepare_inputs)
                or bool(strict_fused_device_only_hot_path)
            )
        )
        need_compact_far_pairs = (
            bool(adaptive_order_active) and not bool(need_traversal_result)
        ) or bool(use_compact_streamed_pairs)
        need_node_interactions = not bool(use_compact_streamed_pairs)
        use_dense_interactions_for_prepare = bool(self.use_dense_interactions) and (
            self.expansion_basis != "solidfmm"
        )
        if not suppress_host_side_effects:
            _prepare_diag(
                "dual-tree start "
                f"theta={theta_val:.3f} mac_type={mac_type_val} "
                f"streamed={bool(self.streamed_far_pairs)} grouped={grouped_interactions_active} "
                f"farfield_mode={farfield_mode} memory_objective={self.memory_objective} "
                f"traversal_config={runtime_traversal_config} "
                f"need_compact_far_pairs={bool(need_compact_far_pairs)} "
                f"need_node_interactions={bool(need_node_interactions)} "
                f"dense_buffers={bool(use_dense_interactions_for_prepare)}"
            )
        if (
            runtime_traversal_config is not None
            and self.memory_objective == "minimum_memory"
            and jax.default_backend() == "gpu"
            and self.tree_type == "radix"
            and self.expansion_basis == "solidfmm"
            and bool(self.streamed_far_pairs)
            and not grouped_interactions_active
        ):
            total_nodes = int(tree_artifacts.tree.parent.shape[0])
            num_internal = int(jnp.asarray(tree_artifacts.tree.left_child).shape[0])
            num_leaves = max(1, total_nodes - num_internal)
            sanitized_traversal_config = (
                _cap_minimum_memory_streamed_gpu_traversal_config_for_tree(
                    traversal_config=runtime_traversal_config,
                    total_nodes=total_nodes,
                    num_leaves=num_leaves,
                    num_particles=int(tree_artifacts.positions_sorted.shape[0]),
                )
            )
            if sanitized_traversal_config != runtime_traversal_config:
                far_slots_before = total_nodes * int(
                    runtime_traversal_config.max_interactions_per_node
                )
                near_slots_before = num_leaves * int(
                    runtime_traversal_config.max_neighbors_per_leaf
                )
                if not suppress_host_side_effects:
                    _prepare_diag(
                        "capped explicit traversal_config for legacy streamed GPU walk "
                        f"total_nodes={total_nodes} num_leaves={num_leaves} "
                        f"far_slots={far_slots_before} near_slots={near_slots_before} "
                        f"from={runtime_traversal_config} "
                        f"to={sanitized_traversal_config}"
                    )
                runtime_traversal_config = sanitized_traversal_config
        jit_traversal_for_prepare = bool(
            self._jit_traversal_default
        ) and not _contains_tracer(
            (tree_artifacts.positions_sorted, tree_artifacts.masses_sorted)
        )
        if self.prepare_stage_memory_split_enabled is not None:
            allow_split_build = bool(self.prepare_stage_memory_split_enabled)
        elif self._prepare_stage_memory_split_env_override is None:
            # Default to the lower-peak split traversal build in the production
            # minimum-memory streamed GPU path; keep env opt-out for debugging.
            allow_split_build = bool(
                self._streamed_minimum_memory_gpu_default_split_build
                and not grouped_interactions_active
            )
        else:
            allow_split_build = bool(self._prepare_stage_memory_split_env_override)
        if bool(strict_fused_device_only_hot_path):
            allow_split_build = True
        if not suppress_host_side_effects:
            _prepare_diag(f"allow_split_build={bool(allow_split_build)}")
        tree_mode_static_radix = (
            str(tree_artifacts.tree_mode).strip().lower() == "static_radix"
        )
        strict_mode_active = bool(
            (
                self._strict_gpu_mode_on
                or (
                    self._strict_gpu_mode_auto
                    and self._large_n_gpu_production_profile_cached
                    and tree_mode_static_radix
                )
            )
        )
        if strict_mode_active:
            strict_context_key = self._strict_cap_profile_context_key(
                tree_mode=str(tree_artifacts.tree_mode),
                leaf_parameter=int(tree_artifacts.leaf_parameter),
                particle_count=int(
                    jnp.asarray(tree_artifacts.positions_sorted).shape[0]
                ),
            )
            strict_fused_hot_path = bool(
                getattr(self, "_strict_fused_mode_active", False)
            )
            strict_profile_key_stable = str(self._strict_profiled_context_key) == str(
                strict_context_key
            )
            if not (strict_fused_hot_path and strict_profile_key_stable):
                self._maybe_load_strict_cap_profile(context_key=strict_context_key)
            if bool(self._strict_cap_require_exact_profile_match):
                if str(self._strict_profiled_context_key) != str(strict_context_key):
                    self._strict_runner_fail_fast_reject_count += 1
                    raise RuntimeError(
                        "strict static lane requires exact cap profile key match: "
                        f"requested={strict_context_key} "
                        f"resolved={self._strict_profiled_context_key or 'none'}"
                    )
            profiled_q = int(self._strict_profiled_max_pair_queue)
            profiled_b = int(self._strict_profiled_pair_process_block)
            if bool(self._strict_cap_require_exact_profile_match) and profiled_q <= 0:
                self._strict_runner_fail_fast_reject_count += 1
                raise RuntimeError(
                    "strict static lane requires non-zero profiled max_pair_queue "
                    f"for key {strict_context_key}"
                )
            if profiled_q > 0:
                if runtime_traversal_config is not None:
                    runtime_traversal_config = DualTreeTraversalConfig(
                        max_pair_queue=max(
                            int(runtime_traversal_config.max_pair_queue),
                            int(profiled_q),
                        ),
                        process_block=(
                            int(profiled_b)
                            if profiled_b > 0
                            else int(runtime_traversal_config.process_block)
                        ),
                        max_interactions_per_node=int(
                            runtime_traversal_config.max_interactions_per_node
                        ),
                        max_neighbors_per_leaf=int(
                            runtime_traversal_config.max_neighbors_per_leaf
                        ),
                    )
                else:
                    runtime_traversal_config = DualTreeTraversalConfig(
                        max_pair_queue=int(profiled_q),
                        process_block=(
                            int(profiled_b)
                            if profiled_b > 0
                            else int(self.pair_process_block or 1024)
                        ),
                        max_interactions_per_node=8192,
                        max_neighbors_per_leaf=4096,
                    )
            if bool(traced_prepare_inputs) and not bool(
                strict_fused_device_only_hot_path
            ):
                allow_split_build = False
            if not suppress_host_side_effects:
                self._refresh_strict_mode_active_count += 1
                # Strict mode contract: one-shot shared count->fill and single queue.
                if not bool(getattr(self, "_strict_shared_env_applied", False)):
                    os.environ["YGGDRAX_DUAL_TREE_SHARED_COUNT_FILL_ONE_SHOT"] = "1"
                    os.environ[
                        "YGGDRAX_DUAL_TREE_SHARED_COUNT_FILL_STEADY_SINGLE_QUEUE"
                    ] = "1"
                    self._strict_shared_env_applied = True
        strict_streamed_fast_path = bool(
            strict_mode_active
            and bool(allow_split_build)
            and bool(use_compact_streamed_pairs)
            and not grouped_interactions_active
            and not bool(need_traversal_result)
            and not adaptive_order_active
            and not mixed_order_farfield_active
            and (
                not bool(traced_prepare_inputs)
                or bool(strict_fused_device_only_hot_path)
            )
        )
        if bool(strict_fused_device_only_hot_path):
            blockers: list[str] = []
            if not bool(strict_mode_active):
                blockers.append("strict_mode_inactive")
            if not bool(allow_split_build):
                blockers.append("split_build_disabled")
            if not bool(use_compact_streamed_pairs):
                blockers.append("compact_streamed_pairs_disabled")
            if bool(grouped_interactions_active):
                blockers.append("grouped_interactions_active")
            if bool(need_traversal_result):
                blockers.append("traversal_result_required")
            if bool(adaptive_order_active):
                blockers.append("adaptive_order_active")
            if bool(mixed_order_farfield_active):
                blockers.append("mixed_order_farfield_active")
            if bool(traced_prepare_inputs) and not bool(strict_streamed_fast_path):
                blockers.append("compact_streamed_tracer_unsupported")
            if bool(getattr(self, "_strict_fused_fastlane_diag_enabled", True)):
                self._strict_fused_fastlane_attempts += 1
                if bool(strict_streamed_fast_path):
                    self._strict_fused_fastlane_hits += 1
                    self._strict_fused_fastlane_last_blockers = tuple()
                else:
                    self._strict_fused_fastlane_misses += 1
                    self._strict_fused_fastlane_last_blockers = tuple(blockers)
                    counts = dict(
                        getattr(self, "_strict_fused_fastlane_block_counts", {})
                    )
                    for key in blockers:
                        counts[str(key)] = int(counts.get(str(key), 0)) + 1
                    self._strict_fused_fastlane_block_counts = counts
            if not bool(strict_streamed_fast_path):
                # Production fused device-only hot path MUST engage the radix
                # streamed fast lane. Never silently fall back to the slower
                # generic build -- raise loudly with the blocking reasons so a
                # misconfiguration is caught instead of quietly running the
                # ~10x-slower generic path.
                self._strict_fused_fallback_count += 1
                self._strict_fused_last_fallback_reason = (
                    "fused_device_only_hot_path_fastlane_blocked:" + ",".join(blockers)
                )
                raise RuntimeError(
                    "strict fused device-only production lane could not engage the "
                    "radix streamed fast lane and must not silently fall back to a "
                    f"slower path (blockers={blockers}). This indicates a "
                    "misconfiguration of the large-N GPU production profile."
                )
        if strict_streamed_fast_path:
            if not suppress_host_side_effects:
                self._refresh_dual_planner_cache_hits += 1
                self._refresh_dual_planner_execute_count += 1
                self._refresh_dual_planner_steady_timing_bypass_count += 1
            return self._prepare_state_dual_and_downward_strict_streamed_fast(
                tree_artifacts=tree_artifacts,
                theta_val=theta_val,
                mac_type_val=mac_type_val,
                dehnen_radius_scale=dehnen_radius_scale,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                record_retry=record_retry,
                farfield_mode=farfield_mode,
                retain_interactions=bool(retain_interactions_active),
                suppress_host_side_effects=suppress_host_side_effects,
            )

        if self.adaptive_order or use_paper_fixed_policy:
            policy_orders = self.p_gears
            if use_paper_fixed_policy:
                policy_orders = (int(tree_artifacts.upward.multipoles.order),)
            if len(policy_orders) == 0:
                raise ValueError("adaptive traversal policy requires non-empty orders")
            policy_state = self._build_adaptive_policy_state(
                upward=tree_artifacts.upward,
                tree=tree_artifacts.tree,
                positions_sorted=tree_artifacts.positions_sorted,
                p_gears=policy_orders,
                force_scale_nodes=force_scale_nodes,
                eps=jnp.asarray(
                    (
                        self.adaptive_eps
                        if self.adaptive_eps is not None
                        else adaptive_policy_tolerance(
                            theta=theta_val,
                            p_gears=policy_orders,
                            dtype=tree_artifacts.upward.multipoles.packed.real.dtype,
                        )
                    ),
                    dtype=tree_artifacts.upward.multipoles.packed.real.dtype,
                ),
                theta=jnp.asarray(
                    theta_val,
                    dtype=tree_artifacts.upward.multipoles.packed.real.dtype,
                ),
                error_model_code=jnp.asarray(
                    self._traversal_policy_error_model_code(),
                    dtype=jnp.int32,
                ),
                dehnen_geometry_mode=self.dehnen_geometry_mode,
            )
            pair_policy = adaptive_pair_policy
        else:
            cache_key = _interaction_cache_key(
                tree_artifacts.tree,
                topology_key=tree_artifacts.topology_key,
                tree_mode=tree_artifacts.tree_mode,
                leaf_parameter=tree_artifacts.leaf_parameter,
                theta=theta_val,
                mac_type=mac_type_val,
                dehnen_radius_scale=dehnen_radius_scale,
                expansion_basis=self.expansion_basis,
                center_mode=upward_center_mode,
                max_pair_queue=self.max_pair_queue,
                pair_process_block=self.pair_process_block,
                traversal_config=runtime_traversal_config,
                refine_local=refine_local_val,
                max_refine_levels=max_refine_levels_val,
                aspect_threshold=aspect_threshold_val,
            )
        has_pair_policy = pair_policy is not None
        has_policy_state = policy_state is not None
        planner_enabled = bool(
            (
                self._refresh_dual_planner_mode_on
                or (
                    self._refresh_dual_planner_mode_auto
                    and self._large_n_gpu_production_profile_cached
                    and tree_mode_static_radix
                )
            )
        )
        planner_hint: Optional[_RefreshDualPlannerHint] = None
        planner_cache_hit = False
        if planner_enabled:
            strict_split_fastlane = bool(
                (
                    bool(suppress_host_side_effects)
                    and not has_pair_policy
                    and not has_policy_state
                )
                or (
                    strict_mode_active
                    and bool(allow_split_build)
                    and not grouped_interactions_active
                    and not bool(need_traversal_result)
                    and not has_pair_policy
                    and not has_policy_state
                )
            )
            if strict_split_fastlane:
                # Strict/static production lane: keep routing fully on a
                # fixed host-side decision and skip compiled route probing +
                # device_get round-trips in the refresh hot path.
                planner_hint = _RefreshDualPlannerHint(
                    use_split_build=bool(allow_split_build),
                    suppress_substage_timing=True,
                )
                planner_cache_hit = True
                if not suppress_host_side_effects:
                    self._refresh_dual_planner_cache_hits += 1
                    self._refresh_dual_planner_execute_count += 1
            else:
                traversal_key = (
                    "none"
                    if runtime_traversal_config is None
                    else (
                        f"{int(runtime_traversal_config.max_pair_queue)}:"
                        f"{int(runtime_traversal_config.process_block)}:"
                        f"{int(runtime_traversal_config.max_interactions_per_node)}:"
                        f"{int(runtime_traversal_config.max_neighbors_per_leaf)}"
                    )
                )
                planner_key = "|".join(
                    (
                        str(tree_artifacts.topology_key),
                        str(tree_artifacts.tree_mode),
                        str(int(tree_artifacts.leaf_parameter)),
                        f"{float(theta_val):.12g}",
                        str(mac_type_val),
                        str(grouped_interactions_active),
                        str(bool(need_traversal_result)),
                        str(bool(need_compact_far_pairs)),
                        str(bool(need_node_interactions)),
                        str(bool(allow_split_build)),
                        str(traversal_key),
                    )
                )
                planner_hint = self._refresh_dual_planner_cache.get(planner_key)
                if planner_hint is None:
                    if not suppress_host_side_effects:
                        self._refresh_dual_planner_cache_misses += 1
                    total_nodes_planner = int(tree_artifacts.tree.parent.shape[0])
                    internal_nodes_planner = int(
                        jnp.asarray(tree_artifacts.tree.left_child).shape[0]
                    )
                    leaf_count_planner = max(
                        0, total_nodes_planner - internal_nodes_planner
                    )
                    (
                        use_split_build_compiled,
                        _use_compact_shared_far_near_compiled,
                        suppress_substage_timing_compiled,
                    ) = _compiled_refresh_dual_planner_route(
                        allow_split_build_flag=jnp.asarray(
                            bool(allow_split_build), dtype=jnp.bool_
                        ),
                        grouped_interactions_flag=jnp.asarray(
                            grouped_interactions_active, dtype=jnp.bool_
                        ),
                        need_traversal_result_flag=jnp.asarray(
                            bool(need_traversal_result), dtype=jnp.bool_
                        ),
                        has_pair_policy_flag=jnp.asarray(
                            has_pair_policy, dtype=jnp.bool_
                        ),
                        has_policy_state_flag=jnp.asarray(
                            has_policy_state, dtype=jnp.bool_
                        ),
                        leaf_count=jnp.asarray(leaf_count_planner, dtype=jnp.int32),
                        need_node_interactions_flag=jnp.asarray(
                            bool(need_node_interactions), dtype=jnp.bool_
                        ),
                        need_compact_far_pairs_flag=jnp.asarray(
                            bool(need_compact_far_pairs), dtype=jnp.bool_
                        ),
                        use_dense_interactions_flag=jnp.asarray(
                            bool(use_dense_interactions_for_prepare), dtype=jnp.bool_
                        ),
                    )
                    if suppress_host_side_effects:
                        use_split_build_compiled_bool = bool(allow_split_build)
                        suppress_substage_timing_compiled_bool = True
                    else:
                        use_split_build_compiled_bool = bool(use_split_build_compiled)
                        suppress_substage_timing_compiled_bool = bool(
                            suppress_substage_timing_compiled
                        )
                    if not suppress_host_side_effects:
                        self._refresh_dual_planner_compiled_route_count += 1
                    planner_hint = _RefreshDualPlannerHint(
                        use_split_build=use_split_build_compiled_bool,
                        suppress_substage_timing=suppress_substage_timing_compiled_bool,
                    )
                    if not suppress_host_side_effects:
                        self._refresh_dual_planner_cache[planner_key] = planner_hint
                        self._refresh_dual_planner_compile_count += 1
                else:
                    planner_cache_hit = True
                    if not suppress_host_side_effects:
                        self._refresh_dual_planner_cache_hits += 1
                if not suppress_host_side_effects:
                    self._refresh_dual_planner_execute_count += 1
        planner_allow_steady_timing_bypass = bool(
            self._planner_steady_timing_bypass_enabled
        )
        dual_artifact_timing_callback = _record_dual_artifact_substage
        if (
            planner_hint is not None
            and planner_cache_hit
            and bool(getattr(planner_hint, "suppress_substage_timing", False))
            and planner_allow_steady_timing_bypass
        ):
            dual_artifact_timing_callback = None
            if not suppress_host_side_effects:
                self._refresh_dual_planner_steady_timing_bypass_count += 1
        _record_dual_stage("_refresh_timing_dual_setup_seconds", stage_t0)

        stage_t0 = time.perf_counter()
        geometry_factory = (
            None
            if tree_artifacts.upward.geometry is not None
            else lambda: compute_tree_geometry(
                tree_artifacts.tree,
                tree_artifacts.positions_sorted,
                max_leaf_size=int(tree_artifacts.leaf_cap),
            )
        )
        dual_artifacts, cache_entry = _build_dual_tree_artifacts(
            tree_artifacts.tree,
            tree_artifacts.upward.geometry,
            geometry_factory=geometry_factory,
            theta=theta_val,
            mac_type=mac_type_val,
            dehnen_radius_scale=dehnen_radius_scale,
            cache_key=cache_key,
            cache_entry=(self._interaction_cache if stateful_cache_enabled else None),
            max_pair_queue=self.max_pair_queue,
            pair_process_block=self.pair_process_block,
            traversal_config=runtime_traversal_config,
            retry_logger=(
                None
                if strict_mode_active
                else (
                    record_retry
                    if bool(getattr(self, "_strict_cap_record_enabled", True))
                    else (None if jit_traversal_for_prepare else record_retry)
                )
            ),
            fail_fast=(self.fail_fast or strict_mode_active),
            use_dense_interactions=use_dense_interactions_for_prepare,
            grouped_interactions=grouped_interactions,
            grouped_chunk_size=runtime_m2l_chunk_size,
            need_traversal_result=need_traversal_result,
            need_compact_far_pairs=need_compact_far_pairs,
            need_node_interactions=need_node_interactions,
            precompute_grouped_class_segments=self._should_precompute_grouped_class_segments(
                grouped_chunk_size=runtime_m2l_chunk_size,
                farfield_mode=farfield_mode,
            ),
            grouped_schedule_budget_bytes=self._grouped_schedule_item_budget(),
            allow_split_build=allow_split_build,
            pair_policy=pair_policy,
            policy_state=policy_state,
            jit_traversal=jit_traversal_for_prepare,
            timing_callback=dual_artifact_timing_callback,
            planner_hint=planner_hint,
        )
        if stateful_cache_enabled:
            if bool(getattr(dual_artifacts, "cache_hit", False)):
                if not suppress_host_side_effects:
                    self._interaction_cache_hits += 1
            else:
                if not suppress_host_side_effects:
                    self._interaction_cache_misses += 1
        _record_dual_stage("_refresh_timing_dual_artifact_build_seconds", stage_t0)
        if stateful_cache_enabled:
            self._interaction_cache = cache_entry

        stage_t0 = _stage_now()
        (
            interactions,
            neighbor_list,
            traversal_result,
            compact_far_pairs,
            dense_buffers,
            grouped_buffers,
            grouped_segment_starts,
            grouped_segment_lengths,
            grouped_segment_class_ids,
            grouped_segment_sort_permutation,
            grouped_segment_group_ids,
            grouped_segment_unique_targets,
        ) = self._unpack_dual_tree_artifacts(dual_artifacts)
        if not suppress_host_side_effects:
            far_pair_count_diag = None
            if compact_far_pairs is not None:
                far_pair_count_diag = int(compact_far_pairs.sources.shape[0])
            elif interactions is not None:
                far_pair_count_diag = int(interactions.sources.shape[0])
            total_nodes_diag = int(tree_artifacts.tree.parent.shape[0])
            internal_nodes_diag = int(
                jnp.asarray(tree_artifacts.tree.left_child).shape[0]
            )
            leaf_count_diag = max(0, total_nodes_diag - internal_nodes_diag)
            self._recent_dual_node_count = int(total_nodes_diag)
            self._recent_dual_leaf_count = int(leaf_count_diag)
            self._recent_dual_neighbor_count = int(neighbor_list.neighbors.shape[0])
            self._recent_dual_far_pair_count = (
                0 if far_pair_count_diag is None else int(far_pair_count_diag)
            )
            _prepare_diag(
                "dual-tree done "
                f"neighbor_count={int(neighbor_list.neighbors.shape[0])} "
                f"far_pair_count={far_pair_count_diag} "
                f"compact_far_pairs={compact_far_pairs is not None} "
                f"interactions_present={interactions is not None}"
            )
            _prepare_diag(
                "dual-tree bytes "
                f"neighbors={_format_nbytes(_estimate_payload_nbytes(neighbor_list))} "
                f"compact_far_pairs={_format_nbytes(_estimate_payload_nbytes(compact_far_pairs))} "
                f"interactions={_format_nbytes(_estimate_payload_nbytes(interactions))} "
                f"dense_buffers={_format_nbytes(_estimate_payload_nbytes(dense_buffers))} "
                f"grouped_buffers={_format_nbytes(_estimate_payload_nbytes(grouped_buffers))}"
            )

        strict_streamed_direct_far_pairs = bool(
            strict_mode_active
            and bool(use_compact_streamed_pairs)
            and compact_far_pairs is not None
            and not bool(adaptive_order_active)
            and not bool(mixed_order_farfield_active)
        )
        if strict_streamed_direct_far_pairs:
            src_far = jnp.asarray(compact_far_pairs.sources, dtype=INDEX_DTYPE)
            tgt_far = jnp.asarray(compact_far_pairs.targets, dtype=INDEX_DTYPE)
            far_pairs_coo = _FarPairCOO(
                sources=src_far,
                targets=tgt_far,
                active_count=getattr(compact_far_pairs, "far_pair_count", None),
            )
            far_pairs_by_gear = ((src_far, tgt_far),)
            adaptive_order_for_downward = True
            p_gears_for_downward = (int(tree_artifacts.upward.multipoles.order),)
            if not suppress_host_side_effects:
                self._recent_far_pairs_by_gear_counts = (int(src_far.shape[0]),)
        else:
            far_pair_plan = self._prepare_state_plan_far_pairs_for_downward(
                interactions=interactions,
                traversal_result=traversal_result,
                compact_far_pairs=compact_far_pairs,
                upward=tree_artifacts.upward,
            )
            far_pairs_by_gear = far_pair_plan.far_pairs_by_gear
            far_pairs_coo = far_pair_plan.far_pairs_coo
            adaptive_order_for_downward = far_pair_plan.adaptive_order_for_downward
            p_gears_for_downward = far_pair_plan.p_gears_for_downward
            if not suppress_host_side_effects:
                self._recent_far_pairs_by_gear_counts = (
                    far_pair_plan.recent_far_pairs_by_gear_counts
                )
        _record_dual_stage("_refresh_timing_dual_far_pair_plan_seconds", stage_t0)

        stage_t0 = _stage_now()
        if suppress_host_side_effects and bool(
            getattr(self, "_static_runtime_fixed_sizing", True)
        ):
            runtime_m2l_chunk_size = runtime_m2l_chunk_size
        else:
            runtime_m2l_chunk_size = self._prepare_state_autotune_downward_chunk_size(
                upward=tree_artifacts.upward,
                far_pairs_by_gear=far_pairs_by_gear,
                p_gears_for_downward=p_gears_for_downward,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            )
        if not suppress_host_side_effects:
            self._recent_dual_m2l_chunk_size = (
                0 if runtime_m2l_chunk_size is None else int(runtime_m2l_chunk_size)
            )
        _record_dual_stage("_refresh_timing_dual_m2l_autotune_seconds", stage_t0)

        stage_t0 = _stage_now()
        interactions_for_downward = (
            None
            if strict_streamed_direct_far_pairs and not bool(retain_interactions_active)
            else self._prepare_state_select_interactions_for_downward(
                interactions=interactions,
                far_pairs_coo=far_pairs_coo,
            )
        )
        _record_dual_stage(
            "_refresh_timing_dual_select_interactions_seconds",
            stage_t0,
        )

        stage_t0 = _stage_now()
        downward = self._prepare_downward_with_artifacts(
            tree=tree_artifacts.tree,
            upward=tree_artifacts.upward,
            theta_val=theta_val,
            locals_template=tree_artifacts.locals_template,
            interactions=interactions_for_downward,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            runtime_traversal_config=runtime_traversal_config,
            record_retry=record_retry,
            dense_buffers=dense_buffers,
            grouped_interactions=grouped_interactions,
            grouped_buffers=grouped_buffers,
            grouped_segment_starts=grouped_segment_starts,
            grouped_segment_lengths=grouped_segment_lengths,
            grouped_segment_class_ids=grouped_segment_class_ids,
            grouped_segment_sort_permutation=grouped_segment_sort_permutation,
            grouped_segment_group_ids=grouped_segment_group_ids,
            grouped_segment_unique_targets=grouped_segment_unique_targets,
            farfield_mode=farfield_mode,
            far_pairs_coo=far_pairs_coo,
            far_pairs_by_gear=far_pairs_by_gear,
            adaptive_order=adaptive_order_for_downward,
            p_gears=p_gears_for_downward,
        )
        _record_dual_stage("_refresh_timing_dual_downward_compute_seconds", stage_t0)

        stage_t0 = _stage_now()
        if not suppress_host_side_effects:
            _prepare_diag(
                "downward done "
                f"locals_shape={tuple(int(v) for v in downward.locals.coefficients.shape)} "
                f"interactions_shape={tuple(int(v) for v in downward.interactions.sources.shape)}"
            )
            _prepare_diag(
                "downward bytes "
                f"locals={_format_nbytes(_estimate_payload_nbytes(downward.locals))} "
                f"stored_interactions={_format_nbytes(_estimate_payload_nbytes(downward.interactions))}"
            )
        if not bool(retain_interactions_active):
            # Prepared state only needs the locals; keep a shape-compatible
            # placeholder so downstream code does not accidentally pin the full
            # far-field pair payload after prepare_state completes.
            downward = downward._replace(
                interactions=(
                    _empty_interaction_storage_like(interactions)
                    if interactions is not None
                    else _empty_interaction_storage_for_tree(tree_artifacts.tree)
                )
            )
        interactions_out: Optional[NodeInteractionList]
        if bool(retain_interactions_active):
            interactions_out = interactions
        else:
            interactions_out = None
        _record_dual_stage("_refresh_timing_dual_finalize_seconds", stage_t0)
        if refresh_timing_active:
            residual = max(
                0.0,
                float(time.perf_counter() - dual_total_t0) - float(dual_stage_sum),
            )
            self._refresh_timing_dual_residual_seconds += residual
        return _PrepareStateDualDownwardArtifacts(
            interactions=interactions_out,
            neighbor_list=neighbor_list,
            traversal_result=(
                traversal_result if bool(need_traversal_result) else None
            ),
            compact_far_pairs=(
                compact_far_pairs
                if (
                    bool(adaptive_order_active)
                    or bool(strict_streamed_direct_far_pairs)
                )
                else None
            ),
            downward=downward,
            cache_entry=cache_entry,
        )

    def _prepare_state_extract_adaptive_far_pairs(
        self,
        *,
        traversal_result: Optional[DualTreeWalkResult],
        compact_far_pairs: Optional[CompactTaggedFarPairs],
    ) -> tuple[Array, Array, Array]:
        """Extract tagged far pairs for adaptive-order downward planning."""

        if len(self.p_gears) == 0:
            raise ValueError("adaptive_order=True requires non-empty p_gears")
        if traversal_result is not None:
            far_total = int(traversal_result.far_pair_count)
            far_sources = jnp.asarray(
                traversal_result.interaction_sources[:far_total], dtype=INDEX_DTYPE
            )
            far_targets = jnp.asarray(
                traversal_result.interaction_targets[:far_total], dtype=INDEX_DTYPE
            )
            far_tags = jnp.asarray(
                traversal_result.interaction_tags[:far_total], dtype=INDEX_DTYPE
            )
            return far_sources, far_targets, far_tags
        if compact_far_pairs is not None:
            return (
                jnp.asarray(compact_far_pairs.sources, dtype=INDEX_DTYPE),
                jnp.asarray(compact_far_pairs.targets, dtype=INDEX_DTYPE),
                jnp.asarray(compact_far_pairs.tags, dtype=INDEX_DTYPE),
            )
        raise RuntimeError("adaptive-order traversal requires tagged far-pair payload")

    def _prepare_state_build_streamed_far_pair_plan(
        self,
        *,
        interactions: Optional[NodeInteractionList],
        compact_far_pairs: Optional[CompactTaggedFarPairs],
        upward: TreeUpwardData,
    ) -> tuple[_FarPairCOO, tuple[tuple[Array, Array], ...], tuple[int, ...]]:
        """Build streamed far-pair payloads and per-gear buckets."""

        if compact_far_pairs is not None:
            src_far = jnp.asarray(compact_far_pairs.sources, dtype=INDEX_DTYPE)
            tgt_far = jnp.asarray(compact_far_pairs.targets, dtype=INDEX_DTYPE)
        else:
            if interactions is None:
                raise RuntimeError(
                    "streamed far-pair execution requires interactions or compact far pairs"
                )
            src_far = jnp.asarray(interactions.sources, dtype=INDEX_DTYPE)
            tgt_far = jnp.asarray(interactions.targets, dtype=INDEX_DTYPE)
        active_count = (
            getattr(compact_far_pairs, "far_pair_count", None)
            if compact_far_pairs is not None
            else None
        )
        far_pairs_coo = _FarPairCOO(
            sources=src_far,
            targets=tgt_far,
            active_count=active_count,
        )
        max_order_int = int(upward.multipoles.order)
        strict_fused_device_only_active = bool(
            getattr(self, "_strict_fused_mode_active", False)
        ) and bool(getattr(self, "_strict_fused_device_only", False))
        if (
            self.mixed_order_farfield
            and max_order_int >= 1
            and (not strict_fused_device_only_active)
        ):
            if interactions is None:
                raise RuntimeError(
                    "mixed-order streamed farfield requires node interaction lists"
                )
            min_order_candidate = (
                max_order_int - 1
                if self.mixed_order_min_order is None
                else int(self.mixed_order_min_order)
            )
            min_order_candidate = max(0, min(min_order_candidate, max_order_int))
            p_gears_for_downward, far_pairs_by_gear = _bucket_far_pairs_by_level_split(
                interactions=interactions,
                src_far=src_far,
                tgt_far=tgt_far,
                max_order=max_order_int,
                min_order=min_order_candidate,
            )
        else:
            p_gears_for_downward = (max_order_int,)
            far_pairs_by_gear = ((src_far, tgt_far),)
        return far_pairs_coo, far_pairs_by_gear, p_gears_for_downward

    def _prepare_state_plan_far_pairs_for_downward(
        self,
        *,
        interactions: Optional[NodeInteractionList],
        traversal_result: Optional[DualTreeWalkResult],
        compact_far_pairs: Optional[CompactTaggedFarPairs],
        upward: TreeUpwardData,
    ) -> _PrepareStateFarPairPlan:
        """Prepare far-pair payloads consumed by the downward sweep."""

        far_pairs_by_gear = None
        far_pairs_coo: Optional[_FarPairCOO] = None
        adaptive_order_for_downward = bool(self.adaptive_order)
        p_gears_for_downward = self.p_gears
        recent_counts: tuple[int, ...] = tuple()

        if self.adaptive_order:
            far_sources, far_targets, far_tags = (
                self._prepare_state_extract_adaptive_far_pairs(
                    traversal_result=traversal_result,
                    compact_far_pairs=compact_far_pairs,
                )
            )
            far_pairs_by_gear = bucket_far_pairs_by_tag(
                far_sources,
                far_targets,
                far_tags,
                num_tags=len(self.p_gears),
            )
            recent_counts = tuple(
                int(bucket_src.shape[0]) for bucket_src, _ in far_pairs_by_gear
            )
        elif self.streamed_far_pairs:
            (
                far_pairs_coo,
                far_pairs_by_gear,
                p_gears_for_downward,
            ) = self._prepare_state_build_streamed_far_pair_plan(
                interactions=interactions,
                compact_far_pairs=compact_far_pairs,
                upward=upward,
            )
            adaptive_order_for_downward = True
            recent_counts = tuple(
                int(bucket_src.shape[0]) for bucket_src, _ in far_pairs_by_gear
            )

        return _PrepareStateFarPairPlan(
            far_pairs_by_gear=far_pairs_by_gear,
            far_pairs_coo=far_pairs_coo,
            adaptive_order_for_downward=adaptive_order_for_downward,
            p_gears_for_downward=p_gears_for_downward,
            recent_far_pairs_by_gear_counts=recent_counts,
        )

    def _prepare_state_autotune_downward_chunk_size(
        self,
        *,
        upward: TreeUpwardData,
        far_pairs_by_gear: Optional[tuple[tuple[Array, Array], ...]],
        p_gears_for_downward: tuple[int, ...],
        runtime_m2l_chunk_size: Optional[int],
    ) -> Optional[int]:
        """Choose runtime M2L chunk size for the downward pass."""

        static_runtime_fixed_sizing = bool(
            getattr(self, "_static_runtime_fixed_sizing", True)
        )
        if static_runtime_fixed_sizing:
            return runtime_m2l_chunk_size

        if bool(getattr(self, "_strict_fused_mode_active", False)):
            return runtime_m2l_chunk_size

        if (
            runtime_m2l_chunk_size is not None
            or not bool(self.autotune_m2l_chunk)
            or self.expansion_basis != "solidfmm"
            or jax.default_backend() != "gpu"
            or far_pairs_by_gear is None
            or len(far_pairs_by_gear) == 0
        ):
            return runtime_m2l_chunk_size

        tune_idx = 0
        tune_pair_count = -1
        for idx, (src_bucket, _) in enumerate(far_pairs_by_gear):
            count_i = int(src_bucket.shape[0])
            if count_i > tune_pair_count:
                tune_idx = idx
                tune_pair_count = count_i
        if tune_pair_count <= 0:
            return runtime_m2l_chunk_size

        tune_src, tune_tgt = far_pairs_by_gear[tune_idx]
        tune_order = int(
            upward.multipoles.order
            if tune_idx >= len(p_gears_for_downward)
            else p_gears_for_downward[tune_idx]
        )
        tuned_chunk = self._autotune_runtime_m2l_chunk_size(
            upward=upward,
            src=tune_src,
            tgt=tune_tgt,
            order=tune_order,
            pair_count=tune_pair_count,
        )
        if tuned_chunk is not None:
            return int(tuned_chunk)
        return runtime_m2l_chunk_size

    def _prepare_state_select_interactions_for_downward(
        self,
        *,
        interactions: Optional[NodeInteractionList],
        far_pairs_coo: Optional[_FarPairCOO],
    ) -> Optional[NodeInteractionList]:
        """Choose whether downward should consume node interactions or COO pairs."""

        if (
            self.streamed_far_pairs
            and far_pairs_coo is not None
            and not bool(self.retain_interactions)
        ):
            # Streamed far-pair execution can feed the downward pass directly
            # from COO pairs, so avoid keeping a second node-structured
            # interaction list alive unless the caller asked to retain it.
            return None
        return interactions

    def _prepare_state_nearfield_artifacts(
        self,
        *,
        neighbor_list: NodeNeighborList,
        nearfield_interop: NearfieldInteropData,
        leaf_cap: int,
        num_particles: int,
        cache_entry: Optional[_InteractionCacheEntry],
        allow_stateful_cache: bool,
    ) -> NearfieldPrecomputeArtifacts:
        """Build/reuse near-field precompute artifacts for prepare_state."""
        effective_leaf_cap = (
            int(nearfield_interop.leaf_particle_indices.shape[1])
            if nearfield_interop.leaf_particle_indices is not None
            else int(leaf_cap)
        )
        nearfield_mode_resolved = self._resolve_nearfield_mode(
            num_particles=num_particles
        )
        retain_pair_vectors = (
            nearfield_mode_resolved == "bucketed"
            and self._should_retain_nearfield_pair_vectors(
                num_particles=int(num_particles)
            )
        )
        nearfield_edge_chunk_size_resolved = self._resolve_nearfield_edge_chunk_size(
            num_particles=num_particles,
            nearfield_mode=nearfield_mode_resolved,
        )
        if nearfield_cache_matches(
            cache_entry,
            nearfield_mode=nearfield_mode_resolved,
            nearfield_edge_chunk_size=nearfield_edge_chunk_size_resolved,
            leaf_cap=int(leaf_cap),
            require_pair_vectors=bool(retain_pair_vectors),
        ):
            return nearfield_from_cache(cache_entry)

        nearfield_artifacts = self._prepare_nearfield_precompute_artifacts(
            neighbor_list=neighbor_list,
            nearfield_interop=nearfield_interop,
            leaf_cap=effective_leaf_cap,
            num_particles=num_particles,
            nearfield_mode=nearfield_mode_resolved,
            nearfield_edge_chunk_size=nearfield_edge_chunk_size_resolved,
            retain_pair_vectors=retain_pair_vectors,
        )
        if allow_stateful_cache and cache_entry is not None:
            self._interaction_cache = with_nearfield_cache_artifacts(
                cache_entry,
                artifacts=nearfield_artifacts,
                nearfield_mode=nearfield_mode_resolved,
                nearfield_edge_chunk_size=nearfield_edge_chunk_size_resolved,
                leaf_cap=int(leaf_cap),
            )
        return nearfield_artifacts

    def _update_locals_template_cache_after_prepare(
        self,
        *,
        locals_template: Optional[LocalExpansionData],
        upward: TreeUpwardData,
        max_order: int,
    ) -> None:
        """Update reusable cartesian local-expansion template after prepare_state."""
        if locals_template is not None and self.expansion_basis == "cartesian":
            self._locals_template = LocalExpansionData(
                order=max_order,
                centers=upward.multipoles.centers,
                coefficients=jnp.zeros_like(locals_template.coefficients),
            )
            return
        self._locals_template = None

    def _prepare_nearfield_precompute_artifacts(
        self,
        *,
        neighbor_list: NodeNeighborList,
        nearfield_interop: NearfieldInteropData,
        leaf_cap: int,
        num_particles: int,
        nearfield_mode: Optional[str] = None,
        nearfield_edge_chunk_size: Optional[int] = None,
        retain_pair_vectors: Optional[bool] = None,
    ) -> NearfieldPrecomputeArtifacts:
        """Best-effort precompute of nearfield leaf-pair and scatter artifacts."""
        nearfield_chunk_sort_indices = None
        nearfield_chunk_group_ids = None
        nearfield_chunk_unique_indices = None

        resolved_nearfield_mode = (
            self._resolve_nearfield_mode(num_particles=num_particles)
            if nearfield_mode is None
            else str(nearfield_mode).strip().lower()
        )
        resolved_nearfield_edge_chunk_size = (
            self._resolve_nearfield_edge_chunk_size(
                num_particles=num_particles,
                nearfield_mode=resolved_nearfield_mode,
            )
            if nearfield_edge_chunk_size is None
            else int(nearfield_edge_chunk_size)
        )
        retain_pair_vectors_resolved = (
            self._should_retain_nearfield_pair_vectors(num_particles=int(num_particles))
            if retain_pair_vectors is None
            else bool(retain_pair_vectors)
        )

        should_precompute_scatter = self._should_precompute_nearfield_scatter_schedules(
            num_particles=int(num_particles)
        )
        if not bool(retain_pair_vectors_resolved) and not bool(
            should_precompute_scatter
        ):
            # In large-N minimum-memory GPU runs, prepared evaluation can derive
            # near-field pair vectors on demand from the neighbor list. Avoid
            # materializing enormous edge-index buffers during prepare_state
            # when we are neither retaining them nor building scatter schedules.
            return NearfieldPrecomputeArtifacts(
                target_leaf_ids=None,
                source_leaf_ids=None,
                valid_pairs=None,
                chunk_sort_indices=None,
                chunk_group_ids=None,
                chunk_unique_indices=None,
            )
        if resolved_nearfield_mode != "bucketed":
            return NearfieldPrecomputeArtifacts(
                target_leaf_ids=None,
                source_leaf_ids=None,
                valid_pairs=None,
                chunk_sort_indices=None,
                chunk_group_ids=None,
                chunk_unique_indices=None,
            )

        nearfield_target_leaf_ids = None
        nearfield_valid_pairs = None

        (
            nearfield_target_leaf_ids,
            nearfield_source_leaf_ids,
            nearfield_valid_pairs,
        ) = self._prepare_leaf_neighbor_pairs_safe(
            nearfield_interop=nearfield_interop,
        )

        traced_nearfield_pairs = False
        if nearfield_target_leaf_ids is not None and nearfield_valid_pairs is not None:
            traced_nearfield_pairs = _contains_tracer(
                (nearfield_target_leaf_ids, nearfield_valid_pairs)
            )

        if (
            nearfield_target_leaf_ids is not None
            and nearfield_valid_pairs is not None
            and not traced_nearfield_pairs
            and bool(should_precompute_scatter)
        ):
            (
                nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices,
            ) = self._prepare_bucketed_scatter_schedules_safe(
                nearfield_interop=nearfield_interop,
                target_leaf_ids=nearfield_target_leaf_ids,
                valid_pairs=nearfield_valid_pairs,
                leaf_cap=int(leaf_cap),
                edge_chunk_size=resolved_nearfield_edge_chunk_size,
            )

        return NearfieldPrecomputeArtifacts(
            target_leaf_ids=(
                nearfield_target_leaf_ids if retain_pair_vectors_resolved else None
            ),
            source_leaf_ids=(
                nearfield_source_leaf_ids if retain_pair_vectors_resolved else None
            ),
            valid_pairs=(
                nearfield_valid_pairs if retain_pair_vectors_resolved else None
            ),
            chunk_sort_indices=nearfield_chunk_sort_indices,
            chunk_group_ids=nearfield_chunk_group_ids,
            chunk_unique_indices=nearfield_chunk_unique_indices,
        )

    def _should_retain_nearfield_pair_vectors(
        self,
        *,
        num_particles: int,
    ) -> bool:
        """Decide whether prepared-state near-field pair vectors should be retained."""
        if self.memory_objective == "minimum_memory":
            return False
        if jax.default_backend() != "gpu":
            return True
        return int(num_particles) <= int(_NEARFIELD_GPU_PRECOMPUTE_MAX_PARTICLES)

    def _should_precompute_nearfield_scatter_schedules(
        self,
        *,
        num_particles: int,
    ) -> bool:
        """Decide whether to materialize near-field scatter schedules."""
        if not bool(self.precompute_nearfield_scatter_schedules):
            return False
        if jax.default_backend() != "gpu":
            return True
        # For large GPU runs, schedule materialization is often memory-dominant;
        # keep execution streamed/chunked instead.
        return int(num_particles) <= int(_NEARFIELD_GPU_PRECOMPUTE_MAX_PARTICLES)

    def _prepare_leaf_neighbor_pairs_safe(
        self,
        *,
        nearfield_interop: NearfieldInteropData,
    ) -> tuple[Optional[Array], Optional[Array], Optional[Array]]:
        """Best-effort leaf neighbor pair generation."""
        try:
            return prepare_leaf_neighbor_pairs(
                jnp.asarray(nearfield_interop.node_ranges, dtype=INDEX_DTYPE),
                jnp.asarray(nearfield_interop.leaf_nodes, dtype=INDEX_DTYPE),
                jnp.asarray(nearfield_interop.offsets, dtype=INDEX_DTYPE),
                jnp.asarray(nearfield_interop.neighbors, dtype=INDEX_DTYPE),
                # Keep edge order aligned with neighbor_list so source leaf ids can
                # be derived on demand without storing a second index vector.
                sort_by_source=False,
            )
        except Exception:
            return None, None, None

    def _prepare_bucketed_scatter_schedules_safe(
        self,
        *,
        nearfield_interop: NearfieldInteropData,
        target_leaf_ids: Array,
        valid_pairs: Array,
        leaf_cap: int,
        edge_chunk_size: int,
    ) -> tuple[Optional[Array], Optional[Array], Optional[Array]]:
        """Best-effort bucketed scatter schedule generation."""
        try:
            edge_count = int(target_leaf_ids.shape[0])
            chunk = int(edge_chunk_size)
            chunk_count = (edge_count + chunk - 1) // chunk if edge_count > 0 else 0
            schedule_items = int(chunk_count * chunk * int(leaf_cap))
            if schedule_items > int(_NEARFIELD_SCATTER_SCHEDULE_INT32_ITEM_LIMIT):
                return None, None, None
            schedule_item_cap = self._resolve_nearfield_schedule_item_cap(
                edge_count=edge_count,
                leaf_cap=int(leaf_cap),
                edge_chunk_size=chunk,
            )
            if schedule_items > int(schedule_item_cap):
                return None, None, None
            if nearfield_interop.leaf_particle_indices is not None:
                return prepare_bucketed_scatter_schedules_from_groups(
                    jnp.asarray(
                        nearfield_interop.leaf_particle_indices,
                        dtype=INDEX_DTYPE,
                    ),
                    jnp.asarray(nearfield_interop.leaf_particle_mask, dtype=bool),
                    target_leaf_ids,
                    valid_pairs,
                    edge_chunk_size=chunk,
                )
            return prepare_bucketed_scatter_schedules(
                jnp.asarray(nearfield_interop.node_ranges, dtype=INDEX_DTYPE),
                jnp.asarray(nearfield_interop.leaf_nodes, dtype=INDEX_DTYPE),
                target_leaf_ids,
                valid_pairs,
                max_leaf_size=int(leaf_cap),
                edge_chunk_size=chunk,
            )
        except Exception:
            return None, None, None

    def _resolve_nearfield_schedule_item_cap(
        self,
        *,
        edge_count: int,
        leaf_cap: int,
        edge_chunk_size: int,
    ) -> int:
        """Return the max near-field schedule items allowed for this workload."""
        del edge_count, leaf_cap, edge_chunk_size
        if self.nearfield_schedule_item_cap is not None:
            return int(self.nearfield_schedule_item_cap)

        backend_name = jax.default_backend()
        base_cap = (
            _NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP_GPU
            if backend_name == "gpu"
            else _NEARFIELD_SCATTER_SCHEDULE_ITEM_CAP
        )
        if self.memory_objective == "minimum_memory":
            base_cap = max(1, base_cap // 4)
        elif self.memory_objective == "throughput":
            base_cap = base_cap * 2

        if self.memory_budget_bytes is not None:
            bytes_per_item = 3 * np.dtype(np.int32).itemsize
            budget_limited_cap = max(1, int(self.memory_budget_bytes) // bytes_per_item)
            base_cap = min(base_cap, budget_limited_cap)
        return int(base_cap)

    def _should_precompute_grouped_class_segments(
        self,
        *,
        grouped_chunk_size: Optional[int],
        farfield_mode: str,
    ) -> bool:
        """Decide whether grouped class-major schedules should be materialized."""
        if grouped_chunk_size is None:
            return False
        if str(farfield_mode).strip().lower() != "class_major":
            return False
        if self.precompute_grouped_class_segments is not None:
            return bool(self.precompute_grouped_class_segments)
        return self.memory_objective != "minimum_memory"

    def _grouped_schedule_item_budget(self) -> int:
        """Return max bytes allowed for cached grouped schedule matrices."""
        return int(self.grouped_schedule_budget_bytes)

    def _resolve_target_indices(
        self,
        *,
        target_indices: Optional[Array],
        num_particles: int,
    ) -> Optional[Array]:
        """Validate and normalize optional target particle indices."""
        if target_indices is None:
            return None
        indices = jnp.asarray(target_indices)
        if indices.ndim != 1:
            raise ValueError("target_indices must be a 1D array")
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise ValueError("target_indices must contain integer values")
        if indices.shape[0] == 0:
            return indices.astype(INDEX_DTYPE)
        # Under JAX tracing we cannot materialize min/max as Python ints.
        if isinstance(indices, jax.core.Tracer):
            return indices.astype(INDEX_DTYPE)
        min_idx = int(jnp.min(indices))
        max_idx = int(jnp.max(indices))
        if min_idx < 0 or max_idx >= int(num_particles):
            raise ValueError("target_indices contains out-of-range values")
        return indices.astype(INDEX_DTYPE)

    def _unpack_dual_tree_artifacts(
        self,
        dual_artifacts: _DualTreeArtifacts,
    ) -> tuple[
        NodeInteractionList,
        NodeNeighborList,
        Optional[DualTreeWalkResult],
        Optional[CompactTaggedFarPairs],
        Optional[DenseInteractionBuffers],
        Optional[GroupedInteractionBuffers],
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
        Optional[Array],
    ]:
        """Unpack dual-tree artifacts for downward preparation and state export."""
        return (
            dual_artifacts.interactions,
            dual_artifacts.neighbor_list,
            dual_artifacts.traversal_result,
            dual_artifacts.compact_far_pairs,
            dual_artifacts.dense_buffers,
            dual_artifacts.grouped_buffers,
            dual_artifacts.grouped_segment_starts,
            dual_artifacts.grouped_segment_lengths,
            dual_artifacts.grouped_segment_class_ids,
            dual_artifacts.grouped_segment_sort_permutation,
            dual_artifacts.grouped_segment_group_ids,
            dual_artifacts.grouped_segment_unique_targets,
        )

    def _prepare_downward_with_artifacts(
        self,
        *,
        tree: Tree,
        upward: TreeUpwardData,
        theta_val: float,
        locals_template: Optional[LocalExpansionData],
        interactions: Optional[NodeInteractionList],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        record_retry: Callable[[DualTreeRetryEvent], None],
        dense_buffers: Optional[DenseInteractionBuffers],
        grouped_interactions: bool,
        grouped_buffers: Optional[GroupedInteractionBuffers],
        grouped_segment_starts: Optional[Array],
        grouped_segment_lengths: Optional[Array],
        grouped_segment_class_ids: Optional[Array],
        grouped_segment_sort_permutation: Optional[Array],
        grouped_segment_group_ids: Optional[Array],
        grouped_segment_unique_targets: Optional[Array],
        farfield_mode: str,
        far_pairs_coo: Optional[_FarPairCOO] = None,
        far_pairs_by_gear: Optional[tuple[tuple[Array, Array], ...]] = None,
        adaptive_order: bool = False,
        p_gears: tuple[int, ...] = tuple(),
    ) -> TreeDownwardData:
        """Prepare downward sweep using precomputed interaction artifacts."""
        return self.prepare_downward_sweep(
            tree,
            upward,
            theta=theta_val,
            initial_locals=locals_template,
            interactions=interactions,
            m2l_chunk_size=runtime_m2l_chunk_size,
            l2l_chunk_size=runtime_l2l_chunk_size,
            traversal_config=runtime_traversal_config,
            retry_logger=record_retry,
            dense_buffers=dense_buffers,
            grouped_interactions=grouped_interactions,
            grouped_buffers=grouped_buffers,
            grouped_segment_starts=grouped_segment_starts,
            grouped_segment_lengths=grouped_segment_lengths,
            grouped_segment_class_ids=grouped_segment_class_ids,
            grouped_segment_sort_permutation=grouped_segment_sort_permutation,
            grouped_segment_group_ids=grouped_segment_group_ids,
            grouped_segment_unique_targets=grouped_segment_unique_targets,
            farfield_mode=farfield_mode,
            far_pairs_coo=far_pairs_coo,
            far_pairs_by_gear=far_pairs_by_gear,
            adaptive_order=adaptive_order,
            p_gears=p_gears,
        )

    def _prepare_state_dual_and_downward_strict_streamed_fast(
        self,
        *,
        tree_artifacts: _PrepareStateTreeUpwardArtifacts,
        theta_val: float,
        mac_type_val: MACType,
        dehnen_radius_scale: float,
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        record_retry: Callable[[DualTreeRetryEvent], None],
        farfield_mode: str,
        retain_interactions: bool = False,
        suppress_host_side_effects: bool = False,
    ) -> _PrepareStateDualDownwardArtifacts:
        """Strict static fast path with compact streamed far-pairs only."""

        geometry_factory = (
            None
            if tree_artifacts.upward.geometry is not None
            else lambda: compute_tree_geometry(
                tree_artifacts.tree,
                tree_artifacts.positions_sorted,
                max_leaf_size=int(tree_artifacts.leaf_cap),
            )
        )
        dual_artifacts, cache_entry = _build_dual_tree_artifacts(
            tree_artifacts.tree,
            tree_artifacts.upward.geometry,
            geometry_factory=geometry_factory,
            theta=theta_val,
            mac_type=mac_type_val,
            dehnen_radius_scale=dehnen_radius_scale,
            cache_key=None,
            cache_entry=None,
            max_pair_queue=self.max_pair_queue,
            pair_process_block=self.pair_process_block,
            traversal_config=runtime_traversal_config,
            retry_logger=None,
            fail_fast=True,
            use_dense_interactions=False,
            grouped_interactions=False,
            grouped_chunk_size=runtime_m2l_chunk_size,
            need_traversal_result=False,
            need_compact_far_pairs=True,
            need_node_interactions=False,
            precompute_grouped_class_segments=False,
            grouped_schedule_budget_bytes=self._grouped_schedule_item_budget(),
            allow_split_build=True,
            pair_policy=None,
            policy_state=None,
            jit_traversal=True,
            timing_callback=None,
            planner_hint=_RefreshDualPlannerHint(
                use_split_build=True,
                suppress_substage_timing=True,
            ),
        )
        (
            interactions,
            neighbor_list,
            traversal_result,
            compact_far_pairs,
            dense_buffers,
            grouped_buffers,
            grouped_segment_starts,
            grouped_segment_lengths,
            grouped_segment_class_ids,
            grouped_segment_sort_permutation,
            grouped_segment_group_ids,
            grouped_segment_unique_targets,
        ) = self._unpack_dual_tree_artifacts(dual_artifacts)
        del (
            interactions,
            traversal_result,
            dense_buffers,
            grouped_buffers,
            grouped_segment_starts,
            grouped_segment_lengths,
            grouped_segment_class_ids,
            grouped_segment_sort_permutation,
            grouped_segment_group_ids,
            grouped_segment_unique_targets,
        )
        if compact_far_pairs is None:
            raise RuntimeError(
                "strict streamed fast path requires compact far-pair artifacts"
            )
        if not suppress_host_side_effects:
            total_nodes_diag = int(tree_artifacts.tree.parent.shape[0])
            internal_nodes_diag = int(
                jnp.asarray(tree_artifacts.tree.left_child).shape[0]
            )
            leaf_count_diag = max(0, total_nodes_diag - internal_nodes_diag)
            self._recent_dual_node_count = int(total_nodes_diag)
            self._recent_dual_leaf_count = int(leaf_count_diag)
            self._recent_dual_neighbor_count = int(neighbor_list.neighbors.shape[0])
            self._recent_dual_far_pair_count = int(compact_far_pairs.sources.shape[0])

        src_far = jnp.asarray(compact_far_pairs.sources, dtype=INDEX_DTYPE)
        tgt_far = jnp.asarray(compact_far_pairs.targets, dtype=INDEX_DTYPE)
        far_pairs_coo = _FarPairCOO(
            sources=src_far,
            targets=tgt_far,
            active_count=getattr(compact_far_pairs, "far_pair_count", None),
        )
        far_pairs_by_gear: tuple[tuple[Array, Array], ...] = ((src_far, tgt_far),)
        p_gears_for_downward = (int(tree_artifacts.upward.multipoles.order),)
        if not suppress_host_side_effects:
            self._recent_far_pairs_by_gear_counts = (int(src_far.shape[0]),)

        if suppress_host_side_effects and bool(
            getattr(self, "_static_runtime_fixed_sizing", True)
        ):
            runtime_m2l_chunk_size = runtime_m2l_chunk_size
        else:
            runtime_m2l_chunk_size = self._prepare_state_autotune_downward_chunk_size(
                upward=tree_artifacts.upward,
                far_pairs_by_gear=far_pairs_by_gear,
                p_gears_for_downward=p_gears_for_downward,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            )
        if not suppress_host_side_effects:
            self._recent_dual_m2l_chunk_size = (
                0 if runtime_m2l_chunk_size is None else int(runtime_m2l_chunk_size)
            )
        downward = self._prepare_downward_with_artifacts(
            tree=tree_artifacts.tree,
            upward=tree_artifacts.upward,
            theta_val=theta_val,
            locals_template=tree_artifacts.locals_template,
            interactions=None,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            runtime_traversal_config=runtime_traversal_config,
            record_retry=record_retry,
            dense_buffers=None,
            grouped_interactions=False,
            grouped_buffers=None,
            grouped_segment_starts=None,
            grouped_segment_lengths=None,
            grouped_segment_class_ids=None,
            grouped_segment_sort_permutation=None,
            grouped_segment_group_ids=None,
            grouped_segment_unique_targets=None,
            farfield_mode=farfield_mode,
            far_pairs_coo=far_pairs_coo,
            far_pairs_by_gear=far_pairs_by_gear,
            adaptive_order=True,
            p_gears=p_gears_for_downward,
        )
        if not bool(retain_interactions):
            downward = downward._replace(
                interactions=_empty_interaction_storage_for_tree(tree_artifacts.tree)
            )
        return _PrepareStateDualDownwardArtifacts(
            interactions=None,
            neighbor_list=neighbor_list,
            traversal_result=None,
            compact_far_pairs=compact_far_pairs,
            downward=downward,
            cache_entry=cache_entry,
        )

    @jaxtyped(typechecker=beartype)
    def prepare_state(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        theta: Optional[float] = None,
        jit_tree: Optional[bool] = None,
        refine_local: Optional[bool] = None,
        max_refine_levels: Optional[int] = None,
        aspect_threshold: Optional[float] = None,
        runtime_overrides_override: Optional[_RuntimeExecutionOverrides] = None,
        fused_device_mode: bool = False,
    ) -> PreparedStateLike:
        """Precompute tree and interaction data for repeated evaluations.

        When ``tree_build_mode`` is ``"fixed_depth"`` the optional
        ``refine_local``, ``max_refine_levels``, and ``aspect_threshold``
        arguments control the host-side leaf refinement pass.
        """

        self._validate_prepare_state_request(
            leaf_size=int(leaf_size),
            max_order=int(max_order),
        )

        refine_local_val = (
            self.refine_local if refine_local is None else bool(refine_local)
        )
        max_refine_levels_val = (
            self.max_refine_levels
            if max_refine_levels is None
            else int(max_refine_levels)
        )
        aspect_threshold_val = (
            self.aspect_threshold
            if aspect_threshold is None
            else float(aspect_threshold)
        )

        collected_retries: list[DualTreeRetryEvent] = []

        def record_retry(event: DualTreeRetryEvent) -> None:
            collected_retries.append(event)
            if self.interaction_retry_logger is not None:
                self.interaction_retry_logger(event)

        input_t0 = time.perf_counter()
        positions_arr, masses_arr, input_dtype = self._prepare_state_input_arrays(
            positions,
            masses,
        )
        if bool(getattr(self, "_refresh_timing_active", False)):
            self._refresh_timing_input_seconds += time.perf_counter() - input_t0
        allow_stateful_cache = not _contains_tracer((positions_arr, masses_arr))

        runtime_overrides = runtime_overrides_override
        if runtime_overrides is None:
            runtime_overrides = self._resolve_runtime_execution_overrides(
                num_particles=int(positions_arr.shape[0]),
            )
        runtime_traversal_config = runtime_overrides.traversal_config
        runtime_m2l_chunk_size = runtime_overrides.m2l_chunk_size
        runtime_l2l_chunk_size = runtime_overrides.l2l_chunk_size
        grouped_interactions = runtime_overrides.grouped_interactions
        farfield_mode = runtime_overrides.farfield_mode
        upward_center_mode = runtime_overrides.center_mode
        if refine_local is None and runtime_overrides.refine_local_override is not None:
            refine_local_val = bool(runtime_overrides.refine_local_override)
        if not allow_stateful_cache:
            runtime_traversal_config = self._resolve_tracing_traversal_config(
                traversal_config=runtime_traversal_config,
            )

        theta_val = float(self.theta if theta is None else theta)
        mac_type_val = self._base_mac_type()

        if can_use_large_n_prepare_path(
            self,
            positions_arr=positions_arr,
            masses_arr=masses_arr,
            allow_stateful_cache=allow_stateful_cache,
        ):
            return prepare_large_n_state(
                self,
                positions_arr=positions_arr,
                masses_arr=masses_arr,
                input_dtype=input_dtype,
                bounds=bounds,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta_val=theta_val,
                mac_type_val=mac_type_val,
                refine_local_val=refine_local_val,
                max_refine_levels_val=max_refine_levels_val,
                aspect_threshold_val=aspect_threshold_val,
                jit_tree_override=jit_tree,
                allow_stateful_cache=allow_stateful_cache,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                upward_center_mode=upward_center_mode,
                record_retry=record_retry,
                collected_retries=collected_retries,
                fused_device_mode=bool(fused_device_mode),
            )

        tree_artifacts = self._prepare_state_tree_and_upward(
            positions_arr=positions_arr,
            masses_arr=masses_arr,
            bounds=bounds,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            refine_local_val=refine_local_val,
            max_refine_levels_val=max_refine_levels_val,
            aspect_threshold_val=aspect_threshold_val,
            jit_tree_override=jit_tree,
            upward_center_mode=upward_center_mode,
            allow_stateful_cache=allow_stateful_cache,
        )
        force_scale_nodes = None
        use_paper_force_scale = self._uses_paper_style_force_scale()
        if use_paper_force_scale:
            node_count = int(tree_artifacts.tree.parent.shape[0])
            previous_force_scale = self._last_force_scale_nodes
            reduction_mode = self._force_scale_reduction_mode()
            need_prepass = False
            policy_orders = self._policy_orders_for_prepare_state(
                max_order=int(max_order)
            )
            if self.mac_force_scale_mode == "paper":
                need_prepass = True
            elif self.mac_force_scale_mode == "prev" or self._in_force_scale_prepass:
                if (
                    previous_force_scale is not None
                    and int(previous_force_scale.shape[0]) == node_count
                ):
                    force_scale_nodes = jnp.asarray(
                        previous_force_scale,
                        dtype=positions_arr.dtype,
                    )
                elif self._uses_paper_style_traversal_policy() and (
                    not self._in_force_scale_prepass
                ):
                    need_prepass = True
                else:
                    force_scale_nodes = jnp.ones(
                        (node_count,),
                        dtype=positions_arr.dtype,
                    )
            else:
                need_prepass = True
            if need_prepass:
                if len(policy_orders) == 0:
                    raise ValueError(
                        "mac_force_scale_mode='prepass' requires non-empty orders"
                    )
                low_order = int(min(policy_orders))
                if self.mac_force_scale_mode == "paper" and (
                    self._uses_paper_style_traversal_policy()
                ):
                    low_order = 1 if int(max_order) >= 1 else 0
                if self.mac_force_scale_mode == "paper" and (
                    self._uses_paper_style_traversal_policy()
                ):
                    prepass_sorted = (
                        self._compute_force_scale_paper_prepass_from_tree_artifacts(
                            tree_artifacts=tree_artifacts,
                            low_order=low_order,
                            theta_val=theta_val,
                            upward_center_mode=upward_center_mode,
                            runtime_traversal_config=runtime_traversal_config,
                            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                            grouped_interactions=grouped_interactions,
                            farfield_mode=farfield_mode,
                            record_retry=record_retry,
                            refine_local_val=refine_local_val,
                            max_refine_levels_val=max_refine_levels_val,
                            aspect_threshold_val=aspect_threshold_val,
                        )
                    )
                else:
                    self._in_force_scale_prepass = True
                    saved_p_gears = self.p_gears
                    saved_adaptive_order = self.adaptive_order
                    saved_adaptive_error_model = self.adaptive_error_model
                    saved_mac_type = self.mac_type
                    try:
                        self.p_gears = (low_order,)
                        self.adaptive_order = bool(saved_adaptive_order)
                        prepass_acc = self.compute_accelerations(
                            positions_arr,
                            masses_arr,
                            bounds=bounds,
                            leaf_size=int(leaf_size),
                            max_order=low_order,
                            return_potential=False,
                            theta=theta_val,
                            reuse_prepared_state=False,
                            jit_tree=jit_tree,
                            jit_traversal=False,
                        )
                    finally:
                        self.p_gears = saved_p_gears
                        self.adaptive_order = saved_adaptive_order
                        self.adaptive_error_model = saved_adaptive_error_model
                        self.mac_type = saved_mac_type
                        self._in_force_scale_prepass = False
                    prepass_sorted = jnp.asarray(prepass_acc)[
                        jnp.argsort(tree_artifacts.inverse_permutation)
                    ]
                force_scale_nodes = self._compute_node_force_scale_from_sorted_acc(
                    tree=tree_artifacts.tree,
                    accelerations_sorted=prepass_sorted,
                    reduction=reduction_mode,
                ).astype(positions_arr.dtype)
                self._last_force_scale_nodes = force_scale_nodes
        dual_downward_artifacts = self._prepare_state_dual_and_downward(
            tree_artifacts=tree_artifacts,
            force_scale_nodes=force_scale_nodes,
            upward_center_mode=upward_center_mode,
            theta_val=theta_val,
            mac_type_val=mac_type_val,
            dehnen_radius_scale=self.dehnen_radius_scale,
            runtime_traversal_config=runtime_traversal_config,
            grouped_interactions=grouped_interactions,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            farfield_mode=farfield_mode,
            record_retry=record_retry,
            refine_local_val=refine_local_val,
            max_refine_levels_val=max_refine_levels_val,
            aspect_threshold_val=aspect_threshold_val,
            allow_stateful_cache=allow_stateful_cache,
        )

        if allow_stateful_cache:
            self._update_locals_template_cache_after_prepare(
                locals_template=tree_artifacts.locals_template,
                upward=tree_artifacts.upward,
                max_order=int(max_order),
            )

        retry_events_tuple = tuple(collected_retries)
        if allow_stateful_cache:
            self._recent_retry_events = retry_events_tuple
            self._record_strict_cap_profile_from_retries(
                retry_events_tuple,
                context_key=self._strict_cap_profile_context_key(
                    tree_mode=str(tree_artifacts.tree_mode),
                    leaf_parameter=int(tree_artifacts.leaf_parameter),
                    particle_count=int(
                        jnp.asarray(tree_artifacts.positions_sorted).shape[0]
                    ),
                ),
            )

        execution_backend = self._resolve_execution_backend()
        tree_type_norm = (
            str(getattr(tree_artifacts.tree, "tree_type", "")).strip().lower()
        )
        build_octree_payload = (
            execution_backend == "octree" or tree_type_norm == "octree"
        )
        if build_octree_payload:
            octree, octree_native = build_octree_execution_data_with_status(
                tree_artifacts.tree
            )
        else:
            octree, octree_native = None, False
        # Only build the native-octree interaction lists when the octree view is
        # actually native (non-degenerate). On a degenerate octree (build_octree_
        # execution_data fell back to the binary tree), the native walk would produce
        # far pairs in a node space inconsistent with `octree`/the near list -> gaps +
        # double-counts; leaving native_far_pairs=None routes far through the compat
        # interaction list on the same (fallback) tree, matching the near field.
        octree_native_neighbors = None
        if execution_backend == "octree" and octree is not None and octree_native:
            octree_native_neighbors = build_octree_native_neighbor_lists(
                tree_artifacts.tree,
                tree_artifacts.upward.geometry,
                theta=theta_val,
                mac_type=mac_type_val,
                dehnen_radius_scale=self.dehnen_radius_scale,
                max_pair_queue=self.max_pair_queue,
                process_block=self.pair_process_block,
                traversal_config=runtime_traversal_config,
            )
        nearfield_interop = _build_nearfield_interop_data(
            tree_artifacts.tree,
            dual_downward_artifacts.neighbor_list,
            octree=None,
            native_neighbors=None,
        )
        if (
            execution_backend == "octree"
            and nearfield_interop.leaf_particle_indices is None
        ):
            leaf_nodes_nf = jnp.asarray(nearfield_interop.leaf_nodes, dtype=INDEX_DTYPE)
            node_ranges_nf = jnp.asarray(
                nearfield_interop.particle_order_node_ranges,
                dtype=INDEX_DTYPE,
            )
            leaf_ranges_nf = node_ranges_nf[leaf_nodes_nf]
            counts_nf = leaf_ranges_nf[:, 1] - leaf_ranges_nf[:, 0] + 1
            width_nf = int(jnp.max(counts_nf)) if int(leaf_nodes_nf.shape[0]) > 0 else 0
            if width_nf > 0:
                offsets_nf = jnp.arange(width_nf, dtype=INDEX_DTYPE)
                leaf_particle_indices_nf = (
                    leaf_ranges_nf[:, 0][:, None] + offsets_nf[None, :]
                )
                leaf_particle_mask_nf = offsets_nf[None, :] < counts_nf[:, None]
                particle_to_leaf_position_nf = jnp.zeros(
                    (int(positions_arr.shape[0]),),
                    dtype=INDEX_DTYPE,
                )
                particle_to_leaf_position_nf = particle_to_leaf_position_nf.at[
                    leaf_particle_indices_nf[leaf_particle_mask_nf]
                ].set(
                    jnp.repeat(
                        jnp.arange(int(leaf_nodes_nf.shape[0]), dtype=INDEX_DTYPE),
                        counts_nf.astype(INDEX_DTYPE),
                    )
                )
            else:
                leaf_particle_indices_nf = jnp.zeros(
                    (int(leaf_nodes_nf.shape[0]), 0), dtype=INDEX_DTYPE
                )
                leaf_particle_mask_nf = jnp.zeros(
                    (int(leaf_nodes_nf.shape[0]), 0), dtype=bool
                )
                particle_to_leaf_position_nf = jnp.zeros(
                    (int(positions_arr.shape[0]),),
                    dtype=INDEX_DTYPE,
                )
            nearfield_interop = nearfield_interop._replace(
                leaf_particle_indices=leaf_particle_indices_nf,
                leaf_particle_mask=leaf_particle_mask_nf,
                particle_to_leaf_position=particle_to_leaf_position_nf,
            )
        nearfield_artifacts = self._prepare_state_nearfield_artifacts(
            neighbor_list=dual_downward_artifacts.neighbor_list,
            nearfield_interop=nearfield_interop,
            leaf_cap=tree_artifacts.leaf_cap,
            num_particles=int(positions_arr.shape[0]),
            cache_entry=dual_downward_artifacts.cache_entry,
            allow_stateful_cache=allow_stateful_cache,
        )
        _prepare_diag(
            "nearfield bytes "
            f"target_leaf_ids={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.target_leaf_ids))} "
            f"source_leaf_ids={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.source_leaf_ids))} "
            f"valid_pairs={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.valid_pairs))} "
            f"chunk_sort_indices={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.chunk_sort_indices))} "
            f"chunk_group_ids={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.chunk_group_ids))} "
            f"chunk_unique_indices={_format_nbytes(_estimate_payload_nbytes(nearfield_artifacts.chunk_unique_indices))}"
        )
        octree_upward = _build_octree_upward_artifacts(
            octree=octree,
            positions_sorted=tree_artifacts.positions_sorted,
            masses_sorted=tree_artifacts.masses_sorted,
            expansion_basis=self.expansion_basis,
            max_order=int(max_order),
        )
        octree_native_far_pairs = None
        if execution_backend == "octree" and octree is not None and octree_native:
            octree_native_far_pairs = build_octree_native_far_pairs(
                tree_artifacts.tree,
                tree_artifacts.upward.geometry,
                theta=theta_val,
                mac_type=mac_type_val,
                dehnen_radius_scale=self.dehnen_radius_scale,
                max_pair_queue=self.max_pair_queue,
                process_block=self.pair_process_block,
                traversal_config=runtime_traversal_config,
            )
        octree_downward = _build_octree_downward_artifacts(
            octree=octree,
            octree_upward=octree_upward,
            interactions=dual_downward_artifacts.interactions,
            native_far_pairs=octree_native_far_pairs,
            execution_backend=execution_backend,
        )

        return FMMPreparedState(
            tree=tree_artifacts.tree,
            upward=_prepared_state_upward_payload(
                upward=tree_artifacts.upward,
                memory_objective=self.memory_objective,
            ),
            downward=dual_downward_artifacts.downward,
            neighbor_list=dual_downward_artifacts.neighbor_list,
            max_leaf_size=tree_artifacts.leaf_cap,
            input_dtype=input_dtype,
            working_dtype=positions_arr.dtype,
            expansion_basis=self.expansion_basis,
            theta=theta_val,
            topology_key=tree_artifacts.topology_key,
            interactions=dual_downward_artifacts.interactions,
            dual_tree_result=dual_downward_artifacts.traversal_result,
            retry_events=retry_events_tuple,
            nearfield_interop=nearfield_interop,
            nearfield_target_leaf_ids=nearfield_artifacts.target_leaf_ids,
            nearfield_source_leaf_ids=nearfield_artifacts.source_leaf_ids,
            nearfield_valid_pairs=nearfield_artifacts.valid_pairs,
            nearfield_chunk_sort_indices=nearfield_artifacts.chunk_sort_indices,
            nearfield_chunk_group_ids=nearfield_artifacts.chunk_group_ids,
            nearfield_chunk_unique_indices=nearfield_artifacts.chunk_unique_indices,
            force_scale_nodes=force_scale_nodes,
            execution_backend=execution_backend,
            octree=octree,
            octree_upward=_prepared_state_octree_upward_payload(
                octree_upward=octree_upward,
                memory_objective=self.memory_objective,
            ),
            octree_downward=_finalize_octree_downward_artifacts(
                octree=octree,
                octree_upward=octree_upward,
                octree_downward=octree_downward,
                expansion_basis=self.expansion_basis,
                execution_backend=execution_backend,
                m2l_chunk_size=runtime_m2l_chunk_size,
            ),
        )
