"""EvaluateMixin: fmm_evaluate methods extracted from the FastMultipoleMethod
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, jaxtyped
from yggdrax.interactions import NodeNeighborList
from yggdrax.tree import Tree

from jaccpot.downward.local_expansions import LocalExpansionData, TreeDownwardData
from jaccpot.nearfield.near_field import compute_leaf_p2p_accelerations

from ._large_n_pipeline import evaluate_large_n_state
from ._large_n_types import LargeNPreparedState
from .dtypes import INDEX_DTYPE
from .fmm_caches import _contains_tracer
from .fmm_state import FMMPreparedState, _octree_farfield_eval_inputs
from .kernels.core import (
    NearfieldInteropData,
    PackedAccelerationDerivatives,
    _build_nearfield_interop_data,
    _evaluate_local_expansions_for_particles,
    _evaluate_prepared_tree,
    _evaluate_prepared_tree_targets,
    _evaluate_tree_compiled_impl,
    _infer_order_from_coeff_count,
    _prepare_tree_evaluation_inputs,
)
from .reference import direct_sum as reference_direct_sum


class EvaluateMixin:
    @jaxtyped(typechecker=beartype)
    def compute_accelerations(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        *,
        target_indices: Optional[Array] = None,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        return_potential: bool = False,
        theta: Optional[float] = None,
        jit_tree: Optional[bool] = None,
        refine_local: Optional[bool] = None,
        max_refine_levels: Optional[int] = None,
        aspect_threshold: Optional[float] = None,
        jit_traversal: Optional[bool] = None,
        reuse_prepared_state: bool = False,
        max_acc_derivative_order: int = 0,
    ) -> Union[
        Array,
        Tuple[Array, Array],
        Tuple[Array, PackedAccelerationDerivatives],
        Tuple[Array, Array, PackedAccelerationDerivatives],
    ]:
        """Run the full FMM pipeline for particle accelerations.

        Parameters
        ----------
        positions : Array
            Source and target particle positions.
        masses : Array
            Particle masses aligned with ``positions``.
        target_indices : Optional[Array]
            Optional 1D index array selecting which target-particle outputs to
            return. All particles are still used as source masses.
        bounds : Optional[Tuple[Array, Array]]
            Optional explicit domain bounds used during tree construction.
        leaf_size : int
            Target maximum particle count per leaf for the prepared tree.
        max_order : int
            Multipole/local expansion order used for the upward and downward
            passes.
        return_potential : bool
            When ``True``, return a tuple ``(accelerations, potentials)``.
        theta : Optional[float]
            Optional per-call MAC opening angle override.
        jit_tree : Optional[bool]
            When ``True``, specialise tree construction via JIT to amortise
            repeated builds for consistent tree sizes.
        refine_local : Optional[bool]
            Override the fixed-depth builder's local refinement toggle when
            ``tree_build_mode`` is ``"fixed_depth"``.
        max_refine_levels : Optional[int]
            Maximum local refinement iterations passed to the builder.
        aspect_threshold : Optional[float]
            Aspect ratio threshold that triggers additional splits in the
            refinement pass.
        jit_traversal : Optional[bool]
            When ``True``, evaluate the traversal/evaluation path with the
            compiled implementation for improved throughput.
        reuse_prepared_state : bool
            Reuse the most recent prepared state when identical array objects
            and preparation parameters are provided.

        Returns
        -------
        Union[Array, Tuple[Array, Array]]
            Accelerations for all particles or selected targets. When
            ``return_potential`` is ``True``, also returns the potential.
        """

        cache_key: Optional[tuple[Any, ...]] = None
        state: Optional[FMMPreparedState] = None
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        if _contains_tracer((positions_arr, masses_arr)):
            if bool(return_potential):
                raise NotImplementedError(
                    "traced compute_accelerations fallback does not support return_potential=True"
                )
            if int(max_acc_derivative_order) != 0:
                raise NotImplementedError(
                    "traced compute_accelerations fallback does not support acceleration derivatives"
                )
            target_positions = (
                positions_arr
                if target_indices is None
                else jnp.asarray(
                    positions_arr[jnp.asarray(target_indices, dtype=INDEX_DTYPE),],
                    dtype=positions_arr.dtype,
                )
            )
            return jax.vmap(
                lambda eval_point: reference_direct_sum(
                    positions_arr,
                    masses_arr,
                    eval_point,
                    G=self.G,
                    softening=self.softening,
                )
            )(target_positions)
        if reuse_prepared_state:
            if bounds is None:
                bounds_key: tuple[Any, ...] = ("none",)
            else:
                bounds_key = ("set", id(bounds[0]), id(bounds[1]))
            cache_key = (
                positions_arr.shape,
                str(positions_arr.dtype),
                masses_arr.shape,
                str(masses_arr.dtype),
                bounds_key,
                int(leaf_size),
                int(max_order),
                None if theta is None else float(theta),
                None if jit_tree is None else bool(jit_tree),
                None if refine_local is None else bool(refine_local),
                None if max_refine_levels is None else int(max_refine_levels),
                None if aspect_threshold is None else float(aspect_threshold),
            )
            state = self._prepared_state_cache_lookup(
                key=cache_key,
                positions=positions_arr,
                masses=masses_arr,
            )

        if state is None:
            state = self.prepare_state(
                positions,
                masses,
                bounds=bounds,
                leaf_size=leaf_size,
                max_order=max_order,
                theta=theta,
                jit_tree=jit_tree,
                refine_local=refine_local,
                max_refine_levels=max_refine_levels,
                aspect_threshold=aspect_threshold,
            )
            if reuse_prepared_state and cache_key is not None:
                self._prepared_state_cache_store(
                    key=cache_key,
                    positions=positions_arr,
                    masses=masses_arr,
                    state=state,
                )

        jit_traversal_flag = (
            self._jit_traversal_default
            if jit_traversal is None
            else bool(jit_traversal)
        )

        evaluation = self.evaluate_prepared_state(
            state,
            target_indices=target_indices,
            return_potential=return_potential,
            jit_traversal=jit_traversal_flag,
            max_acc_derivative_order=max_acc_derivative_order,
        )
        return evaluation

    @jaxtyped(typechecker=beartype)
    def evaluate_prepared_state(
        self: "FastMultipoleMethod",
        state: PreparedStateLike,
        *,
        target_indices: Optional[Array] = None,
        return_potential: bool = False,
        jit_traversal: bool = True,
        max_acc_derivative_order: int = 0,
    ) -> Union[
        Array,
        Tuple[Array, Array],
        Tuple[Array, PackedAccelerationDerivatives],
        Tuple[Array, Array, PackedAccelerationDerivatives],
    ]:
        """Evaluate accelerations/potentials for all particles or targets."""

        if isinstance(state, LargeNPreparedState):
            return evaluate_large_n_state(
                self,
                state,
                target_indices=target_indices,
                return_potential=return_potential,
                max_acc_derivative_order=max_acc_derivative_order,
            )

        resolved_target_indices = self._resolve_target_indices(
            target_indices=target_indices,
            num_particles=int(state.inverse_permutation.shape[0]),
        )
        tracing_targets = isinstance(
            state.positions_sorted, jax.core.Tracer
        ) or isinstance(resolved_target_indices, jax.core.Tracer)
        derivative_order = int(max_acc_derivative_order)
        if derivative_order < 0:
            raise ValueError("max_acc_derivative_order must be non-negative")
        if derivative_order > 0 and state.expansion_basis != "solidfmm":
            raise NotImplementedError(
                "max_acc_derivative_order > 0 currently requires expansion_basis='solidfmm'"
            )

        use_full_eval_for_targets = bool(return_potential) and (
            resolved_target_indices is not None
        )
        # Octree backend: evaluate the octree-native far-field locals (the near-field is
        # already octree-native). Only the full-particle path honours these overrides.
        (
            octree_farfield_local_data,
            octree_farfield_leaf_nodes,
            octree_farfield_node_ranges,
        ) = _octree_farfield_eval_inputs(state)
        if (
            resolved_target_indices is None
            or tracing_targets
            or use_full_eval_for_targets
        ):
            evaluation = _evaluate_prepared_tree(
                fmm=self,
                tree=state.tree,
                positions_sorted=state.positions_sorted,
                masses_sorted=state.masses_sorted,
                downward=state.downward,
                neighbor_list=state.neighbor_list,
                nearfield_interop=state.nearfield_interop,
                farfield_local_data=octree_farfield_local_data,
                farfield_leaf_nodes=octree_farfield_leaf_nodes,
                farfield_node_ranges=octree_farfield_node_ranges,
                nearfield_target_leaf_ids=state.nearfield_target_leaf_ids,
                nearfield_source_leaf_ids=state.nearfield_source_leaf_ids,
                nearfield_valid_pairs=state.nearfield_valid_pairs,
                nearfield_chunk_sort_indices=state.nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids=state.nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices=state.nearfield_chunk_unique_indices,
                max_leaf_size=state.max_leaf_size,
                return_potential=return_potential,
                jit_traversal=jit_traversal,
                max_acc_derivative_order=derivative_order,
            )
        else:
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[resolved_target_indices],
                dtype=INDEX_DTYPE,
            )
            evaluation = _evaluate_prepared_tree_targets(
                fmm=self,
                tree=state.tree,
                positions_sorted=state.positions_sorted,
                masses_sorted=state.masses_sorted,
                downward=state.downward,
                neighbor_list=state.neighbor_list,
                nearfield_interop=state.nearfield_interop,
                farfield_local_data=None,
                farfield_leaf_nodes=None,
                farfield_node_ranges=None,
                target_sorted_indices=target_sorted_indices,
                return_potential=return_potential,
                max_acc_derivative_order=derivative_order,
            )

        if jnp.issubdtype(state.input_dtype, jnp.floating):
            output_dtype = state.input_dtype
        else:
            output_dtype = state.working_dtype

        if derivative_order > 0:
            if return_potential:
                acc_sorted, pot_sorted, deriv_sorted = evaluation
            else:
                acc_sorted, deriv_sorted = evaluation
            if resolved_target_indices is None:
                accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation]
                derivatives = tuple(
                    jnp.asarray(level)[state.inverse_permutation]
                    for level in deriv_sorted
                )
                if return_potential:
                    potentials = jnp.asarray(pot_sorted)[state.inverse_permutation]
            elif tracing_targets or use_full_eval_for_targets:
                accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation][
                    resolved_target_indices
                ]
                derivatives = tuple(
                    jnp.asarray(level)[state.inverse_permutation][
                        resolved_target_indices
                    ]
                    for level in deriv_sorted
                )
                if return_potential:
                    potentials = jnp.asarray(pot_sorted)[state.inverse_permutation][
                        resolved_target_indices
                    ]
            else:
                accelerations = jnp.asarray(acc_sorted)
                derivatives = tuple(jnp.asarray(level) for level in deriv_sorted)
                if return_potential:
                    potentials = jnp.asarray(pot_sorted)
            accelerations = accelerations.astype(output_dtype)
            derivatives = tuple(level.astype(output_dtype) for level in derivatives)
            if return_potential:
                return accelerations, potentials.astype(output_dtype), derivatives
            return accelerations, derivatives

        if return_potential:
            acc_sorted, pot_sorted = evaluation
            if resolved_target_indices is None:
                accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation]
                potentials = jnp.asarray(pot_sorted)[state.inverse_permutation]
            elif tracing_targets or use_full_eval_for_targets:
                accelerations = jnp.asarray(acc_sorted)[state.inverse_permutation][
                    resolved_target_indices
                ]
                potentials = jnp.asarray(pot_sorted)[state.inverse_permutation][
                    resolved_target_indices
                ]
            else:
                accelerations = jnp.asarray(acc_sorted)
                potentials = jnp.asarray(pot_sorted)
            accelerations = accelerations.astype(output_dtype)
            potentials = potentials.astype(output_dtype)
            return accelerations, potentials

        if resolved_target_indices is None:
            accelerations = jnp.asarray(evaluation)[state.inverse_permutation]
        elif tracing_targets:
            accelerations = jnp.asarray(evaluation)[state.inverse_permutation][
                resolved_target_indices
            ]
        else:
            accelerations = jnp.asarray(evaluation)
        accelerations = accelerations.astype(output_dtype)
        return accelerations

    @jaxtyped(typechecker=beartype)
    def _evaluate_prepared_state_at_positions_sorted(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        positions_sorted: Array,
        *,
        target_indices: Optional[Array] = None,
        jit_traversal: bool = True,
    ) -> Array:
        """Evaluate accelerations for updated sorted positions on a fixed topology."""
        positions_sorted_arr = jnp.asarray(positions_sorted, dtype=state.working_dtype)
        if positions_sorted_arr.shape != state.positions_sorted.shape:
            raise ValueError(
                "positions_sorted must have shape "
                f"{tuple(state.positions_sorted.shape)}, got {tuple(positions_sorted_arr.shape)}"
            )

        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(positions_sorted_arr.shape[0]),
        )
        upward = self.prepare_upward_sweep(
            state.tree,
            positions_sorted_arr,
            state.masses_sorted,
            max_order=int(state.downward.locals.order),
            center_mode=runtime_overrides.center_mode,
            max_leaf_size=int(state.max_leaf_size),
        )
        downward = self.prepare_downward_sweep(
            state.tree,
            upward,
            theta=float(state.theta),
            mac_type=self.mac_type,
            initial_locals=None,
            interactions=state.interactions,
            m2l_chunk_size=runtime_overrides.m2l_chunk_size,
            l2l_chunk_size=runtime_overrides.l2l_chunk_size,
            grouped_interactions=runtime_overrides.grouped_interactions,
            farfield_mode=runtime_overrides.farfield_mode,
            dehnen_radius_scale=self.dehnen_radius_scale,
        )
        resolved_target_indices = self._resolve_target_indices(
            target_indices=target_indices,
            num_particles=int(state.inverse_permutation.shape[0]),
        )
        tracing_targets = isinstance(
            positions_sorted_arr, jax.core.Tracer
        ) or isinstance(resolved_target_indices, jax.core.Tracer)
        # Octree backend: evaluate octree-native far-field locals (full path only).
        (
            octree_farfield_local_data,
            octree_farfield_leaf_nodes,
            octree_farfield_node_ranges,
        ) = _octree_farfield_eval_inputs(state)
        if resolved_target_indices is None or tracing_targets:
            evaluation = _evaluate_prepared_tree(
                fmm=self,
                tree=state.tree,
                positions_sorted=positions_sorted_arr,
                masses_sorted=state.masses_sorted,
                downward=downward,
                neighbor_list=state.neighbor_list,
                nearfield_interop=state.nearfield_interop,
                farfield_local_data=octree_farfield_local_data,
                farfield_leaf_nodes=octree_farfield_leaf_nodes,
                farfield_node_ranges=octree_farfield_node_ranges,
                nearfield_target_leaf_ids=state.nearfield_target_leaf_ids,
                nearfield_source_leaf_ids=state.nearfield_source_leaf_ids,
                nearfield_valid_pairs=state.nearfield_valid_pairs,
                nearfield_chunk_sort_indices=state.nearfield_chunk_sort_indices,
                nearfield_chunk_group_ids=state.nearfield_chunk_group_ids,
                nearfield_chunk_unique_indices=state.nearfield_chunk_unique_indices,
                max_leaf_size=state.max_leaf_size,
                return_potential=False,
                jit_traversal=jit_traversal,
                max_acc_derivative_order=0,
            )
        else:
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[resolved_target_indices],
                dtype=INDEX_DTYPE,
            )
            evaluation = _evaluate_prepared_tree_targets(
                fmm=self,
                tree=state.tree,
                positions_sorted=positions_sorted_arr,
                masses_sorted=state.masses_sorted,
                downward=downward,
                neighbor_list=state.neighbor_list,
                nearfield_interop=state.nearfield_interop,
                farfield_local_data=None,
                farfield_leaf_nodes=None,
                farfield_node_ranges=None,
                target_sorted_indices=target_sorted_indices,
                return_potential=False,
                max_acc_derivative_order=0,
            )

        if jnp.issubdtype(state.input_dtype, jnp.floating):
            output_dtype = state.input_dtype
        else:
            output_dtype = state.working_dtype
        if resolved_target_indices is None:
            accelerations = jnp.asarray(evaluation)[state.inverse_permutation]
        elif tracing_targets:
            accelerations = jnp.asarray(evaluation)[state.inverse_permutation][
                resolved_target_indices
            ]
        else:
            accelerations = jnp.asarray(evaluation)
        return accelerations.astype(output_dtype)

    @jaxtyped(typechecker=beartype)
    def evaluate_tree(
        self: "FastMultipoleMethod",
        tree: Tree,
        positions_sorted: Array,
        masses_sorted: Array,
        locals_or_downward: Union[LocalExpansionData, TreeDownwardData],
        neighbor_list: NodeNeighborList,
        *,
        nearfield_interop: Optional[NearfieldInteropData] = None,
        farfield_local_data: Optional[LocalExpansionData] = None,
        farfield_leaf_nodes: Optional[Array] = None,
        farfield_node_ranges: Optional[Array] = None,
        precomputed_target_leaf_ids: Optional[Array] = None,
        precomputed_source_leaf_ids: Optional[Array] = None,
        precomputed_valid_pairs: Optional[Array] = None,
        precomputed_chunk_sort_indices: Optional[Array] = None,
        precomputed_chunk_group_ids: Optional[Array] = None,
        precomputed_chunk_unique_indices: Optional[Array] = None,
        max_leaf_size: Optional[int] = None,
        return_potential: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Combine far- and near-field effects for leaf particles."""

        setup = _prepare_tree_evaluation_inputs(
            tree,
            positions_sorted,
            masses_sorted,
            locals_or_downward,
            neighbor_list,
            farfield_local_data=farfield_local_data,
            farfield_leaf_nodes=farfield_leaf_nodes,
            farfield_node_ranges=farfield_node_ranges,
            max_leaf_size=max_leaf_size,
            return_potential=return_potential,
        )

        if setup.empty_output is not None:
            return setup.empty_output

        locals_data = setup.locals_data
        positions = setup.positions
        masses = setup.masses
        leaf_nodes = setup.leaf_nodes
        node_ranges = setup.node_ranges
        resolved_max_leaf = setup.max_leaf_size

        order = int(locals_data.order)
        nearfield_mode = self._resolve_nearfield_mode(
            num_particles=int(positions.shape[0])
        )
        nearfield_edge_chunk_size = self._resolve_nearfield_edge_chunk_size(
            num_particles=int(positions.shape[0]),
            nearfield_mode=nearfield_mode,
        )
        nearfield_view = (
            _build_nearfield_interop_data(tree, neighbor_list)
            if nearfield_interop is None
            else nearfield_interop
        )

        near = compute_leaf_p2p_accelerations(
            tree,
            neighbor_list,
            positions,
            masses,
            G=self.G,
            softening=self.softening,
            max_leaf_size=resolved_max_leaf,
            return_potential=return_potential,
            nearfield_mode=nearfield_mode,
            edge_chunk_size=nearfield_edge_chunk_size,
            precomputed_target_leaf_ids=precomputed_target_leaf_ids,
            precomputed_source_leaf_ids=precomputed_source_leaf_ids,
            precomputed_valid_pairs=precomputed_valid_pairs,
            precomputed_chunk_sort_indices=precomputed_chunk_sort_indices,
            precomputed_chunk_group_ids=precomputed_chunk_group_ids,
            precomputed_chunk_unique_indices=precomputed_chunk_unique_indices,
            node_ranges_override=nearfield_view.node_ranges,
            leaf_nodes_override=nearfield_view.leaf_nodes,
            neighbor_offsets_override=nearfield_view.offsets,
            neighbor_indices_override=nearfield_view.neighbors,
            neighbor_counts_override=nearfield_view.counts,
            leaf_particle_indices_override=nearfield_view.leaf_particle_indices,
            leaf_particle_mask_override=nearfield_view.leaf_particle_mask,
        )

        far_grad, far_potential_pre, _ = _evaluate_local_expansions_for_particles(
            locals_data,
            positions,
            leaf_nodes=leaf_nodes,
            node_ranges=node_ranges,
            max_leaf_size=resolved_max_leaf,
            order=order,
            expansion_basis=self.expansion_basis,
            return_potential=return_potential,
            max_acc_derivative_order=0,
        )

        # far_grad is d/d(delta) of +1/r with delta = center - eval_point.
        # Physical acceleration is d/d(eval_point)(+1/r) * G = -d/d(delta)(+1/r) * G.
        far_acc = -self.G * far_grad

        if return_potential:
            near_acc, near_pot = near
            far_pot = (
                -self.G * far_potential_pre
                if far_potential_pre is not None
                else jnp.zeros((positions.shape[0],), dtype=positions.dtype)
            )
            accelerations = near_acc + far_acc
            potentials = near_pot + far_pot
            return accelerations, potentials

        accelerations = near + far_acc
        return accelerations

    @jaxtyped(typechecker=beartype)
    def evaluate_tree_compiled(
        self: "FastMultipoleMethod",
        tree: Tree,
        positions_sorted: Array,
        masses_sorted: Array,
        locals_or_downward: Union[LocalExpansionData, TreeDownwardData],
        neighbor_list: NodeNeighborList,
        *,
        nearfield_interop: Optional[NearfieldInteropData] = None,
        farfield_local_data: Optional[LocalExpansionData] = None,
        farfield_leaf_nodes: Optional[Array] = None,
        farfield_node_ranges: Optional[Array] = None,
        precomputed_target_leaf_ids: Optional[Array] = None,
        precomputed_source_leaf_ids: Optional[Array] = None,
        precomputed_valid_pairs: Optional[Array] = None,
        precomputed_chunk_sort_indices: Optional[Array] = None,
        precomputed_chunk_group_ids: Optional[Array] = None,
        precomputed_chunk_unique_indices: Optional[Array] = None,
        max_leaf_size: Optional[int] = None,
        return_potential: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        """JIT-compiled variant of :meth:`evaluate_tree`."""

        resolved_max_leaf = (
            self.fixed_max_leaf_size
            if self.fixed_max_leaf_size is not None
            else max_leaf_size
        )

        setup = _prepare_tree_evaluation_inputs(
            tree,
            positions_sorted,
            masses_sorted,
            locals_or_downward,
            neighbor_list,
            farfield_local_data=farfield_local_data,
            farfield_leaf_nodes=farfield_leaf_nodes,
            farfield_node_ranges=farfield_node_ranges,
            max_leaf_size=resolved_max_leaf,
            return_potential=return_potential,
        )

        if setup.empty_output is not None:
            return setup.empty_output

        if self.fixed_order is not None:
            order = int(self.fixed_order)
        else:
            coeff_count = int(setup.locals_data.coefficients.shape[-1])
            order = _infer_order_from_coeff_count(
                coeff_count=coeff_count,
                expansion_basis=self.expansion_basis,
            )

        if self.fixed_max_leaf_size is not None and setup.max_leaf_size > int(
            self.fixed_max_leaf_size
        ):
            raise ValueError("fixed_max_leaf_size too small for prepared tree")
        nearfield_mode = self._resolve_nearfield_mode(
            num_particles=int(setup.positions.shape[0])
        )
        nearfield_edge_chunk_size = self._resolve_nearfield_edge_chunk_size(
            num_particles=int(setup.positions.shape[0]),
            nearfield_mode=nearfield_mode,
        )
        nearfield_view = (
            _build_nearfield_interop_data(tree, neighbor_list)
            if nearfield_interop is None
            else nearfield_interop
        )

        return _evaluate_tree_compiled_impl(
            tree,
            setup.positions,
            setup.masses,
            setup.locals_data,
            neighbor_list,
            jnp.asarray(nearfield_view.leaf_nodes, dtype=INDEX_DTYPE),
            jnp.asarray(nearfield_view.node_ranges, dtype=INDEX_DTYPE),
            jnp.asarray(nearfield_view.offsets, dtype=INDEX_DTYPE),
            jnp.asarray(nearfield_view.neighbors, dtype=INDEX_DTYPE),
            jnp.asarray(nearfield_view.counts, dtype=INDEX_DTYPE),
            (
                jnp.asarray(nearfield_view.leaf_particle_indices, dtype=INDEX_DTYPE)
                if nearfield_view.leaf_particle_indices is not None
                else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(nearfield_view.leaf_particle_mask, dtype=bool)
                if nearfield_view.leaf_particle_mask is not None
                else jnp.zeros((0, 0), dtype=bool)
            ),
            setup.leaf_nodes,
            setup.node_ranges,
            (
                jnp.asarray(precomputed_target_leaf_ids, dtype=INDEX_DTYPE)
                if precomputed_target_leaf_ids is not None
                else jnp.zeros((0,), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(precomputed_source_leaf_ids, dtype=INDEX_DTYPE)
                if precomputed_source_leaf_ids is not None
                else jnp.zeros((0,), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(precomputed_valid_pairs, dtype=bool)
                if precomputed_valid_pairs is not None
                else jnp.zeros((0,), dtype=bool)
            ),
            (
                jnp.asarray(precomputed_chunk_sort_indices, dtype=INDEX_DTYPE)
                if precomputed_chunk_sort_indices is not None
                else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(precomputed_chunk_group_ids, dtype=INDEX_DTYPE)
                if precomputed_chunk_group_ids is not None
                else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
            ),
            (
                jnp.asarray(precomputed_chunk_unique_indices, dtype=INDEX_DTYPE)
                if precomputed_chunk_unique_indices is not None
                else jnp.zeros((0, 0), dtype=INDEX_DTYPE)
            ),
            jnp.zeros((setup.leaf_nodes.shape[0] + 1,), dtype=INDEX_DTYPE),
            jnp.zeros((0,), dtype=INDEX_DTYPE),
            jnp.zeros((0, 0), dtype=INDEX_DTYPE),
            jnp.zeros((0, 0), dtype=bool),
            jnp.zeros((setup.leaf_nodes.shape[0], 0, 0), dtype=INDEX_DTYPE),
            jnp.zeros((setup.leaf_nodes.shape[0], 0, 0), dtype=bool),
            G=self.G,
            softening=self.softening,
            order=order,
            expansion_basis=self.expansion_basis,
            max_leaf_size=setup.max_leaf_size,
            return_potential=return_potential,
            nearfield_mode=nearfield_mode,
            nearfield_edge_chunk_size=nearfield_edge_chunk_size,
        )
