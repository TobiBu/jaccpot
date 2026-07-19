"""DerivativesMixin: fmm_derivatives methods extracted from the FastMultipoleMethod
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

from math import comb
from typing import Any, Optional

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, jaxtyped
from yggdrax.geometry import compute_tree_geometry
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.downward.local_expansions import LocalExpansionData
from jaccpot.operators.symmetric_tensors import contract_symmetric_one_axis_3d
from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_source_motion_multipoles,
)
from jaccpot.upward.tree_expansions import NodeMultipoleData, TreeUpwardData

from .dtypes import INDEX_DTYPE
from .fmm_caches import _contains_tracer
from .fmm_state import FMMPreparedState
from .kernels.core import (
    PackedAccelerationDerivatives,
    _build_nearfield_interop_data,
    _build_target_nearfield_source_index_matrix,
    _compute_targeted_nearfield,
    _evaluate_local_expansions_for_particles,
    _evaluate_local_expansions_for_target_particles,
    _map_targets_to_leaf_positions,
)


class DerivativesMixin:
    @jaxtyped(typechecker=beartype)
    def compute_accelerations_and_jerk(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        theta: Optional[float] = None,
        jit_tree: Optional[bool] = None,
        refine_local: Optional[bool] = None,
        max_refine_levels: Optional[int] = None,
        aspect_threshold: Optional[float] = None,
        jit_traversal: Optional[bool] = None,
        reuse_prepared_state: bool = False,
        jerk_mode: str = "fast_approx",
        jerk_fd_dt: float = 1e-3,
    ) -> tuple[Array, Array]:
        """Run FMM and return accelerations plus jerk estimates.

        Jerk combines:
        - exact near-field pairwise jerk from source/target velocities,
        - far-field convective term from acceleration Jacobian times target velocity.
        """
        cache_key: Optional[tuple[Any, ...]] = None
        state: Optional[FMMPreparedState] = None
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
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
        return self.evaluate_prepared_state_with_jerk(
            state,
            velocities,
            target_indices=target_indices,
            jit_traversal=jit_traversal_flag,
            jerk_mode=jerk_mode,
            jerk_fd_dt=jerk_fd_dt,
        )

    @jaxtyped(typechecker=beartype)
    def compute_accelerations_with_time_derivatives(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        theta: Optional[float] = None,
        jit_tree: Optional[bool] = None,
        refine_local: Optional[bool] = None,
        max_refine_levels: Optional[int] = None,
        aspect_threshold: Optional[float] = None,
        jit_traversal: Optional[bool] = None,
        reuse_prepared_state: bool = False,
        max_time_derivative_order: int = 1,
        mode: str = "accurate",
    ) -> tuple[Array, tuple[Array, ...]]:
        """Run FMM and return accelerations plus time derivatives up to order K."""
        cache_key: Optional[tuple[Any, ...]] = None
        state: Optional[FMMPreparedState] = None
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
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
        return self.evaluate_prepared_state_with_time_derivatives(
            state,
            velocities,
            target_indices=target_indices,
            jit_traversal=jit_traversal_flag,
            max_time_derivative_order=max_time_derivative_order,
            mode=mode,
        )
        if target_indices is None and not _contains_tracer((state, evaluation)):
            accelerations_out = evaluation[0] if return_potential else evaluation
            inv = jnp.asarray(state.inverse_permutation)
            sorted_idx = jnp.argsort(inv)
            accelerations_sorted = jnp.asarray(accelerations_out)[sorted_idx]
            self._last_force_scale_nodes = (
                self._compute_node_force_scale_from_sorted_acc(
                    tree=state.tree,
                    accelerations_sorted=accelerations_sorted,
                    reduction=self._force_scale_reduction_mode(),
                )
            )
        return evaluation

    @jaxtyped(typechecker=beartype)
    def evaluate_prepared_state_with_jerk(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
        jit_traversal: bool = True,
        jerk_mode: str = "fast_approx",
        jerk_fd_dt: float = 1e-3,
    ) -> tuple[Array, Array]:
        """Evaluate accelerations and jerk for all particles or targets."""
        vel_arr = jnp.asarray(velocities, dtype=state.working_dtype)
        if vel_arr.shape != state.positions_sorted.shape:
            raise ValueError(
                "velocities must have shape "
                f"{tuple(state.positions_sorted.shape)}, got {tuple(vel_arr.shape)}"
            )

        resolved_target_indices = self._resolve_target_indices(
            target_indices=target_indices,
            num_particles=int(state.inverse_permutation.shape[0]),
        )
        mode = str(jerk_mode).strip().lower()
        if mode not in ("fast_approx", "accurate"):
            raise ValueError("jerk_mode must be 'fast_approx' or 'accurate'")

        if mode == "accurate":
            if state.expansion_basis != "solidfmm":
                dt = float(jerk_fd_dt)
                if dt <= 0.0:
                    raise ValueError("jerk_fd_dt must be positive")

                accelerations = self.evaluate_prepared_state(
                    state,
                    target_indices=resolved_target_indices,
                    return_potential=False,
                    jit_traversal=jit_traversal,
                    max_acc_derivative_order=0,
                )
                particle_indices = jnp.asarray(
                    state.tree.particle_indices, dtype=INDEX_DTYPE
                )
                vel_sorted = vel_arr[particle_indices]
                positions_plus_sorted = (
                    jnp.asarray(state.positions_sorted) + dt * vel_sorted
                )
                positions_minus_sorted = (
                    jnp.asarray(state.positions_sorted) - dt * vel_sorted
                )

                acc_plus = self._evaluate_prepared_state_at_positions_sorted(
                    state=state,
                    positions_sorted=positions_plus_sorted,
                    target_indices=resolved_target_indices,
                    jit_traversal=jit_traversal,
                )
                acc_minus = self._evaluate_prepared_state_at_positions_sorted(
                    state=state,
                    positions_sorted=positions_minus_sorted,
                    target_indices=resolved_target_indices,
                    jit_traversal=jit_traversal,
                )
                jerk = (acc_plus - acc_minus) / (2.0 * dt)
                if jnp.issubdtype(state.input_dtype, jnp.floating):
                    output_dtype = state.input_dtype
                else:
                    output_dtype = state.working_dtype
                return accelerations.astype(output_dtype), jerk.astype(output_dtype)

            acc_eval = self.evaluate_prepared_state(
                state,
                target_indices=resolved_target_indices,
                return_potential=False,
                jit_traversal=jit_traversal,
                max_acc_derivative_order=1,
            )
            accelerations, acc_derivs = acc_eval
            acc_jac = acc_derivs[0]
            vel_targets = (
                vel_arr
                if resolved_target_indices is None
                else vel_arr[resolved_target_indices]
            )
            far_convective_jerk = jnp.einsum("nij,nj->ni", acc_jac, vel_targets)
            far_source_motion_jerk = (
                self._evaluate_farfield_time_derivative_orders(
                    state=state,
                    velocities=vel_arr,
                    target_indices=resolved_target_indices,
                    max_time_derivative_order=1,
                )[0]
                - far_convective_jerk
            )
            near_jerk = self._evaluate_target_nearfield_jerk(
                state=state,
                velocities=vel_arr,
                target_indices=resolved_target_indices,
            )
            jerk = near_jerk + far_convective_jerk + far_source_motion_jerk
            if jnp.issubdtype(state.input_dtype, jnp.floating):
                output_dtype = state.input_dtype
            else:
                output_dtype = state.working_dtype
            return accelerations.astype(output_dtype), jerk.astype(output_dtype)

        acc_eval = self.evaluate_prepared_state(
            state,
            target_indices=resolved_target_indices,
            return_potential=False,
            jit_traversal=jit_traversal,
            max_acc_derivative_order=1,
        )
        accelerations, acc_derivs = acc_eval
        acc_jac = acc_derivs[0]
        vel_targets = (
            vel_arr
            if resolved_target_indices is None
            else vel_arr[resolved_target_indices]
        )
        far_jerk = jnp.einsum("nij,nj->ni", acc_jac, vel_targets)

        near_jerk = self._evaluate_target_nearfield_jerk(
            state=state,
            velocities=vel_arr,
            target_indices=resolved_target_indices,
        )

        jerk = near_jerk + far_jerk
        if jnp.issubdtype(state.input_dtype, jnp.floating):
            output_dtype = state.input_dtype
        else:
            output_dtype = state.working_dtype
        return accelerations.astype(output_dtype), jerk.astype(output_dtype)

    @jaxtyped(typechecker=beartype)
    def evaluate_prepared_state_with_time_derivatives(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
        jit_traversal: bool = True,
        max_time_derivative_order: int = 1,
        mode: str = "accurate",
    ) -> tuple[Array, tuple[Array, ...]]:
        """Evaluate accelerations and total time derivatives up to order K.

        Returns ``(accelerations, derivatives)`` where ``derivatives[n-1]``
        corresponds to ``D_t^n a``.
        """
        k_max = int(max_time_derivative_order)
        if k_max < 1:
            raise ValueError("max_time_derivative_order must be >= 1")
        if k_max > 3:
            raise NotImplementedError(
                "max_time_derivative_order > 3 is not implemented yet"
            )
        resolved_target_indices = self._resolve_target_indices(
            target_indices=target_indices,
            num_particles=int(state.inverse_permutation.shape[0]),
        )
        mode_norm = str(mode).strip().lower()
        if mode_norm not in ("accurate",):
            raise ValueError("mode must be 'accurate'")

        vel_arr = jnp.asarray(velocities, dtype=state.working_dtype)
        if vel_arr.shape != state.positions_sorted.shape:
            raise ValueError(
                "velocities must have shape "
                f"{tuple(state.positions_sorted.shape)}, got {tuple(vel_arr.shape)}"
            )

        accelerations = self.evaluate_prepared_state(
            state,
            target_indices=resolved_target_indices,
            return_potential=False,
            jit_traversal=jit_traversal,
            max_acc_derivative_order=0,
        )
        far_terms = self._evaluate_farfield_time_derivative_orders(
            state=state,
            velocities=vel_arr,
            target_indices=resolved_target_indices,
            max_time_derivative_order=k_max,
        )
        near_terms = self._evaluate_target_nearfield_time_derivatives(
            state=state,
            velocities=vel_arr,
            target_indices=resolved_target_indices,
            max_time_derivative_order=k_max,
        )
        derivatives = tuple(n + f for n, f in zip(near_terms, far_terms))
        if jnp.issubdtype(state.input_dtype, jnp.floating):
            output_dtype = state.input_dtype
        else:
            output_dtype = state.working_dtype
        return accelerations.astype(output_dtype), tuple(
            d.astype(output_dtype) for d in derivatives
        )

    @jaxtyped(typechecker=beartype)
    def _evaluate_target_nearfield_jerk(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
    ) -> Array:
        particle_indices = jnp.asarray(state.tree.particle_indices, dtype=INDEX_DTYPE)
        vel_sorted = velocities[particle_indices]

        if target_indices is None:
            target_sorted_indices = jnp.arange(
                state.positions_sorted.shape[0], dtype=INDEX_DTYPE
            )
        else:
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[target_indices], dtype=INDEX_DTYPE
            )
        leaf_nodes = jnp.asarray(state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
        node_ranges = jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE)
        target_leaf_positions = _map_targets_to_leaf_positions(
            target_sorted_indices=target_sorted_indices,
            leaf_nodes=leaf_nodes,
            node_ranges=node_ranges,
        )
        near_source_idx, near_source_mask = _build_target_nearfield_source_index_matrix(
            target_sorted_indices=target_sorted_indices,
            target_leaf_positions=target_leaf_positions,
            nearfield_interop=(
                _build_nearfield_interop_data(state.tree, state.neighbor_list)
                if state.nearfield_interop is None
                else state.nearfield_interop
            ),
        )
        _, _, near_jerk_sorted, _, _ = _compute_targeted_nearfield(
            positions_sorted=state.positions_sorted,
            masses_sorted=state.masses_sorted,
            target_sorted_indices=target_sorted_indices,
            source_indices=near_source_idx,
            source_mask=near_source_mask,
            G=jnp.asarray(self.G, dtype=state.positions_sorted.dtype),
            softening=float(self.softening),
            return_potential=False,
            velocities_sorted=vel_sorted,
            return_jerk=True,
        )
        if near_jerk_sorted is None:
            raise RuntimeError("expected near-field jerk values")
        if target_indices is None:
            return near_jerk_sorted[state.inverse_permutation]
        return near_jerk_sorted

    @jaxtyped(typechecker=beartype)
    def _evaluate_target_nearfield_time_derivatives(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
        max_time_derivative_order: int,
    ) -> tuple[Array, ...]:
        """Evaluate near-field time derivatives up to order K (currently K<=2)."""
        k_max = int(max_time_derivative_order)
        if k_max < 1:
            return tuple()
        if k_max > 3:
            raise NotImplementedError(
                "near-field time derivatives above order 3 are not implemented"
            )
        particle_indices = jnp.asarray(state.tree.particle_indices, dtype=INDEX_DTYPE)
        vel_sorted = velocities[particle_indices]
        if target_indices is None:
            target_sorted_indices = jnp.arange(
                state.positions_sorted.shape[0], dtype=INDEX_DTYPE
            )
        else:
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[target_indices], dtype=INDEX_DTYPE
            )
        leaf_nodes = jnp.asarray(state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
        node_ranges = jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE)
        target_leaf_positions = _map_targets_to_leaf_positions(
            target_sorted_indices=target_sorted_indices,
            leaf_nodes=leaf_nodes,
            node_ranges=node_ranges,
        )
        near_source_idx, near_source_mask = _build_target_nearfield_source_index_matrix(
            target_sorted_indices=target_sorted_indices,
            target_leaf_positions=target_leaf_positions,
            nearfield_interop=(
                _build_nearfield_interop_data(state.tree, state.neighbor_list)
                if state.nearfield_interop is None
                else state.nearfield_interop
            ),
        )
        _, _, near_jerk_sorted, near_snap_sorted, near_crackle_sorted = (
            _compute_targeted_nearfield(
                positions_sorted=state.positions_sorted,
                masses_sorted=state.masses_sorted,
                target_sorted_indices=target_sorted_indices,
                source_indices=near_source_idx,
                source_mask=near_source_mask,
                G=jnp.asarray(self.G, dtype=state.positions_sorted.dtype),
                softening=float(self.softening),
                return_potential=False,
                velocities_sorted=vel_sorted,
                return_jerk=True,
                return_snap=(k_max >= 2),
                return_crackle=(k_max >= 3),
            )
        )
        if near_jerk_sorted is None:
            raise RuntimeError("expected near-field jerk values")
        if k_max >= 2 and near_snap_sorted is None:
            raise RuntimeError("expected near-field snap values")
        if k_max >= 3 and near_crackle_sorted is None:
            raise RuntimeError("expected near-field crackle values")
        if target_indices is None:
            jerk = near_jerk_sorted[state.inverse_permutation]
            if k_max >= 2:
                snap = near_snap_sorted[state.inverse_permutation]  # type: ignore[index]
            if k_max >= 3:
                crackle = near_crackle_sorted[state.inverse_permutation]  # type: ignore[index]
        else:
            jerk = near_jerk_sorted
            if k_max >= 2:
                snap = near_snap_sorted  # type: ignore[assignment]
            if k_max >= 3:
                crackle = near_crackle_sorted  # type: ignore[assignment]
        if k_max == 1:
            return (jerk,)
        if k_max == 2:
            return (jerk, snap)  # type: ignore[possibly-undefined]
        return (jerk, snap, crackle)  # type: ignore[possibly-undefined]

    @jaxtyped(typechecker=beartype)
    def _evaluate_source_motion_farfield_jerk(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
    ) -> Array:
        """Evaluate source-motion far-field jerk using analytic dM->dL contraction."""
        if state.expansion_basis != "solidfmm":
            raise NotImplementedError(
                "source-motion far-field jerk currently requires expansion_basis='solidfmm'"
            )
        particle_indices = jnp.asarray(state.tree.particle_indices, dtype=INDEX_DTYPE)
        vel_sorted = velocities[particle_indices]
        centers = jnp.asarray(state.downward.locals.centers, dtype=state.working_dtype)
        source_motion_packed = prepare_solidfmm_complex_source_motion_multipoles(
            state.tree,
            state.positions_sorted,
            state.masses_sorted,
            vel_sorted,
            max_order=int(state.downward.locals.order),
            centers=centers,
            time_derivative_order=1,
            max_leaf_size=int(state.max_leaf_size),
            rotation=self.complex_rotation,
        )
        source_motion_multipoles = NodeMultipoleData(
            order=int(state.downward.locals.order),
            centers=centers,
            moments=None,  # type: ignore[arg-type]
            packed=jnp.asarray(source_motion_packed),
            component_matrix=None,
            source_motion_packed=None,
        )
        source_motion_upward = TreeUpwardData(
            geometry=compute_tree_geometry(state.tree, state.positions_sorted),
            mass_moments=compute_tree_mass_moments(
                state.tree,
                state.positions_sorted,
                state.masses_sorted,
            ),
            multipoles=source_motion_multipoles,
        )
        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(state.positions_sorted.shape[0]),
        )
        source_motion_downward = self.prepare_downward_sweep(
            state.tree,
            source_motion_upward,
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
        tracing_targets = isinstance(
            state.positions_sorted, jax.core.Tracer
        ) or isinstance(target_indices, jax.core.Tracer)
        if target_indices is None or tracing_targets:
            far_grad_sorted, _, _ = _evaluate_local_expansions_for_particles(
                source_motion_downward.locals,
                state.positions_sorted,
                leaf_nodes=jnp.asarray(
                    state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE
                ),
                node_ranges=jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE),
                max_leaf_size=state.max_leaf_size,
                order=int(source_motion_downward.locals.order),
                expansion_basis=state.expansion_basis,
                return_potential=False,
                max_acc_derivative_order=0,
            )
            if target_indices is None:
                far_grad = jnp.asarray(far_grad_sorted)[state.inverse_permutation]
            else:
                far_grad = jnp.asarray(far_grad_sorted)[state.inverse_permutation][
                    target_indices
                ]
        else:
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[target_indices], dtype=INDEX_DTYPE
            )
            leaf_nodes = jnp.asarray(
                state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE
            )
            node_ranges = jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE)
            target_leaf_positions = _map_targets_to_leaf_positions(
                target_sorted_indices=target_sorted_indices,
                leaf_nodes=leaf_nodes,
                node_ranges=node_ranges,
            )
            far_grad, _, _ = _evaluate_local_expansions_for_target_particles(
                local_data=source_motion_downward.locals,
                positions_sorted=state.positions_sorted,
                target_sorted_indices=target_sorted_indices,
                target_leaf_positions=target_leaf_positions,
                leaf_nodes=leaf_nodes,
                order=int(source_motion_downward.locals.order),
                expansion_basis=state.expansion_basis,
                return_potential=False,
                max_acc_derivative_order=0,
            )
        return -jnp.asarray(self.G, dtype=state.positions_sorted.dtype) * far_grad

    @jaxtyped(typechecker=beartype)
    def _evaluate_farfield_time_derivative_orders(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Array,
        *,
        target_indices: Optional[Array] = None,
        max_time_derivative_order: int,
    ) -> tuple[Array, ...]:
        """Evaluate far-field total time derivatives up to order ``max_time_derivative_order``.

        Uses binomial expansion of ``(∂t + v·∇)^n a`` with analytic source-motion
        locals ``L_k = ∂t^k L`` and acceleration spatial derivatives.
        """
        if state.expansion_basis != "solidfmm":
            raise NotImplementedError(
                "far-field higher time derivatives currently require expansion_basis='solidfmm'"
            )
        k_max = int(max_time_derivative_order)
        if k_max <= 0:
            return tuple()

        def _contract_acc_tensor_with_velocity_power(
            tensor: Array,
            velocity: Array,
            *,
            order: int,
        ) -> Array:
            """Contract symmetric acceleration-derivative tensor ``order`` times."""
            if order <= 0:
                raise ValueError("order must be positive")

            def contract_row(row: Array, vrow: Array) -> Array:
                # row shape: (3, components(order))
                contracted = row
                for ord_i in range(order, 0, -1):
                    contracted = jax.vmap(
                        lambda comp: contract_symmetric_one_axis_3d(
                            comp,
                            vrow,
                            order=ord_i,
                        ),
                        in_axes=0,
                        out_axes=0,
                    )(contracted)
                return contracted[:, 0]

            return jax.vmap(contract_row, in_axes=(0, 0), out_axes=0)(tensor, velocity)

        particle_indices = jnp.asarray(state.tree.particle_indices, dtype=INDEX_DTYPE)
        vel_sorted = velocities[particle_indices]
        centers = jnp.asarray(state.downward.locals.centers, dtype=state.working_dtype)
        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(state.positions_sorted.shape[0]),
        )
        resolved_target_indices = self._resolve_target_indices(
            target_indices=target_indices,
            num_particles=int(state.inverse_permutation.shape[0]),
        )
        tracing_targets = isinstance(
            state.positions_sorted, jax.core.Tracer
        ) or isinstance(resolved_target_indices, jax.core.Tracer)
        vel_targets = (
            velocities
            if resolved_target_indices is None
            else velocities[resolved_target_indices]
        )

        # Build local coefficient streams L_k = ∂t^k L, including k=0.
        locals_by_k: list[LocalExpansionData] = [state.downward.locals]
        geometry = compute_tree_geometry(state.tree, state.positions_sorted)
        mass_moments = compute_tree_mass_moments(
            state.tree, state.positions_sorted, state.masses_sorted
        )
        for k in range(1, k_max + 1):
            source_motion_packed = prepare_solidfmm_complex_source_motion_multipoles(
                state.tree,
                state.positions_sorted,
                state.masses_sorted,
                vel_sorted,
                max_order=int(state.downward.locals.order),
                centers=centers,
                time_derivative_order=k,
                max_leaf_size=int(state.max_leaf_size),
                rotation=self.complex_rotation,
            )
            source_motion_upward = TreeUpwardData(
                geometry=geometry,
                mass_moments=mass_moments,
                multipoles=NodeMultipoleData(
                    order=int(state.downward.locals.order),
                    centers=centers,
                    moments=None,  # type: ignore[arg-type]
                    packed=jnp.asarray(source_motion_packed),
                    component_matrix=jnp.asarray(source_motion_packed),
                    source_motion_packed=None,
                ),
            )
            down_k = self.prepare_downward_sweep(
                state.tree,
                source_motion_upward,
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
            locals_by_k.append(down_k.locals)

        def _evaluate_local_stream(
            local_data: LocalExpansionData,
            *,
            max_acc_deriv_order: int,
        ) -> tuple[Array, Optional[PackedAccelerationDerivatives]]:
            if resolved_target_indices is None or tracing_targets:
                far_grad_sorted, _, far_derivs = (
                    _evaluate_local_expansions_for_particles(
                        local_data,
                        state.positions_sorted,
                        leaf_nodes=jnp.asarray(
                            state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE
                        ),
                        node_ranges=jnp.asarray(
                            state.tree.node_ranges, dtype=INDEX_DTYPE
                        ),
                        max_leaf_size=state.max_leaf_size,
                        order=int(local_data.order),
                        expansion_basis=state.expansion_basis,
                        return_potential=False,
                        max_acc_derivative_order=max_acc_deriv_order,
                    )
                )
                if resolved_target_indices is None:
                    far_grad = jnp.asarray(far_grad_sorted)[state.inverse_permutation]
                    if far_derivs is None:
                        derivs = None
                    else:
                        derivs = tuple(
                            jnp.asarray(level)[state.inverse_permutation]
                            for level in far_derivs
                        )
                else:
                    far_grad = jnp.asarray(far_grad_sorted)[state.inverse_permutation][
                        resolved_target_indices
                    ]
                    if far_derivs is None:
                        derivs = None
                    else:
                        derivs = tuple(
                            jnp.asarray(level)[state.inverse_permutation][
                                resolved_target_indices
                            ]
                            for level in far_derivs
                        )
                return far_grad, derivs
            target_sorted_indices = jnp.asarray(
                state.inverse_permutation[resolved_target_indices], dtype=INDEX_DTYPE
            )
            leaf_nodes = jnp.asarray(
                state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE
            )
            node_ranges = jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE)
            target_leaf_positions = _map_targets_to_leaf_positions(
                target_sorted_indices=target_sorted_indices,
                leaf_nodes=leaf_nodes,
                node_ranges=node_ranges,
            )
            far_grad, _, far_derivs = _evaluate_local_expansions_for_target_particles(
                local_data=local_data,
                positions_sorted=state.positions_sorted,
                target_sorted_indices=target_sorted_indices,
                target_leaf_positions=target_leaf_positions,
                leaf_nodes=leaf_nodes,
                order=int(local_data.order),
                expansion_basis=state.expansion_basis,
                return_potential=False,
                max_acc_derivative_order=max_acc_deriv_order,
            )
            return far_grad, far_derivs

        g_const = jnp.asarray(self.G, dtype=state.positions_sorted.dtype)
        outputs: list[Array] = []
        for n in range(1, k_max + 1):
            accum = jnp.zeros_like(vel_targets)
            for k in range(0, n + 1):
                m = n - k
                far_grad_k, far_derivs_k = _evaluate_local_stream(
                    locals_by_k[k],
                    max_acc_deriv_order=m,
                )
                if m == 0:
                    term_vec = -g_const * far_grad_k
                else:
                    if far_derivs_k is None:
                        raise RuntimeError(
                            "expected far-field acceleration derivatives"
                        )
                    acc_deriv_tensor = g_const * far_derivs_k[m - 1]
                    term_vec = _contract_acc_tensor_with_velocity_power(
                        acc_deriv_tensor,
                        vel_targets,
                        order=m,
                    )
                accum = accum + float(comb(n, k)) * term_vec
            outputs.append(accum)
        return tuple(outputs)
