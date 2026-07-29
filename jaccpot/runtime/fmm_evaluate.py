"""EvaluateMixin: fmm_evaluate methods extracted from the FastMultipoleMethod
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, jaxtyped
from yggdrax.interactions import NodeNeighborList
from yggdrax.tree import Tree

from jaccpot.downward.local_expansions import LocalExpansionData, TreeDownwardData
from jaccpot.nearfield.near_field import (
    compute_leaf_p2p_accelerations,
    compute_leaf_p2p_accelerations_radix_fast_lane,
)

from ._large_n_pipeline import evaluate_large_n_state
from ._large_n_types import LargeNPreparedState
from ._nearfield_fastlane import (
    leaf_major_nearfield_payload_cached,
    nearfield_topology_arrays,
)
from .dtypes import INDEX_DTYPE
from .fmm_caches import _contains_tracer
from .fmm_constants import _env_flag
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

# Opt-in: route the differentiable near field through the leaf-major fast lane
# instead of the bucketed edge-list kernel. Off by default so the shipped grad
# path is unchanged until the in-context win is proven on the target hardware.
_DIFFERENTIABLE_NEARFIELD_FAST_LANE = "JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE"


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
        masses_sorted: Optional[Array] = None,
        target_indices: Optional[Array] = None,
        jit_traversal: bool = True,
        nearfield_mode_override: Optional[str] = None,
        force_ungrouped_farfield: bool = False,
    ) -> Array:
        """Evaluate accelerations for updated sorted positions on a fixed topology.

        Re-runs the upward sweep (P2M + COM centers) and the downward sweep (M2L
        on the frozen ``state.interactions`` list, then L2L) on the live
        ``positions_sorted``, followed by L2P + pure-JAX P2P. Every discrete
        artifact -- the tree, interaction list, neighbor list, and near-field
        index buffers -- is taken from ``state`` unchanged. When ``masses_sorted``
        is provided it is threaded through P2M and the near-field in place of
        ``state.masses_sorted`` (the differentiable-w.r.t.-mass path); otherwise
        the prepared masses are reused. Both ``positions_sorted`` and
        ``masses_sorted`` are differentiable inputs, and the result never reads
        ``state.upward``/``state.downward``, so ``jax.grad`` over this method is an
        exact fixed-topology gradient.
        """
        positions_sorted_arr = jnp.asarray(positions_sorted, dtype=state.working_dtype)
        if positions_sorted_arr.shape != state.positions_sorted.shape:
            raise ValueError(
                "positions_sorted must have shape "
                f"{tuple(state.positions_sorted.shape)}, got {tuple(positions_sorted_arr.shape)}"
            )
        if masses_sorted is None:
            masses_sorted_arr = state.masses_sorted
        else:
            masses_sorted_arr = jnp.asarray(masses_sorted, dtype=state.working_dtype)
            if masses_sorted_arr.shape != state.masses_sorted.shape:
                raise ValueError(
                    "masses_sorted must have shape "
                    f"{tuple(state.masses_sorted.shape)}, got {tuple(masses_sorted_arr.shape)}"
                )

        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(positions_sorted_arr.shape[0]),
        )
        grouped_interactions = runtime_overrides.grouped_interactions
        farfield_mode = runtime_overrides.farfield_mode
        if force_ungrouped_farfield and grouped_interactions:
            # The grouped/class-major M2L builds its pair classes on the HOST
            # (yggdrax ``build_grouped_interactions_from_pairs`` does
            # ``np.asarray(jax.device_get(geometry.center))``), so it raises
            # ``TracerArrayConversionError`` as soon as the expansion centres are
            # traced -- i.e. on every reverse pass. The ungrouped M2L applies each
            # pair's exact displacement instead of one representative per lattice
            # class, so it is the MORE accurate of the two (measured ~6.6x closer to
            # the exact direct sum on a deep-tree config) -- not merely a different
            # execution strategy. NOTE the expansion centres
            # (``runtime_overrides.center_mode``) are passed through UNCHANGED: the
            # grouped classification requires geometric (aabb) centres, and rewriting
            # them here would change the force the gradient is taken of.
            grouped_interactions = False
            farfield_mode = "pair_grouped"
        upward = self.prepare_upward_sweep(
            state.tree,
            positions_sorted_arr,
            masses_sorted_arr,
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
            grouped_interactions=grouped_interactions,
            farfield_mode=farfield_mode,
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
                masses_sorted=masses_sorted_arr,
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
                nearfield_mode_override=nearfield_mode_override,
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
                masses_sorted=masses_sorted_arr,
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
    def _evaluate_prepared_state_at_positions_and_masses_sorted(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        positions_sorted: Array,
        masses_sorted: Array,
        *,
        target_indices: Optional[Array] = None,
        jit_traversal: bool = False,
        nearfield_mode_override: Optional[str] = None,
        force_ungrouped_farfield: bool = True,
    ) -> Array:
        """Fixed-topology re-eval at live sorted positions AND masses.

        Named differentiable seam used by :meth:`differentiable_accelerations`.
        Thin front for :meth:`_evaluate_prepared_state_at_positions_sorted` with
        ``masses_sorted`` threaded through P2M and the near-field, so gradients
        flow w.r.t. both positions and masses at fixed topology.

        ``force_ungrouped_farfield`` defaults to ``True`` here: the grouped M2L
        classifies pairs on the host and is therefore not traceable, so the
        differentiable seam always takes the ungrouped M2L -- which is also the more
        accurate of the two, see
        :meth:`_evaluate_prepared_state_at_positions_sorted`.
        """
        return self._evaluate_prepared_state_at_positions_sorted(
            state,
            positions_sorted,
            masses_sorted=masses_sorted,
            target_indices=target_indices,
            jit_traversal=jit_traversal,
            nearfield_mode_override=nearfield_mode_override,
            force_ungrouped_farfield=force_ungrouped_farfield,
        )

    @jaxtyped(typechecker=beartype)
    def differentiable_accelerations(
        self: "FastMultipoleMethod",
        state: Union[FMMPreparedState, LargeNPreparedState],
        positions: Array,
        masses: Array,
        *,
        target_indices: Optional[Array] = None,
        jit_traversal: bool = False,
        grad_plan: Optional[Any] = None,
    ) -> Array:
        """Exact fixed-topology gradients of the FMM force w.r.t. positions/masses.

        End-to-end differentiable single-GPU FMM acceleration. The tree topology
        (Morton order, node membership, MAC accept/reject, the M2L interaction
        list, and the near-field partition) is taken as a **constant** from the
        pre-built ``state``; the numeric pipeline (P2M, COM centers, M2M/M2L/L2L
        translations, L2P, near-field P2P) is re-evaluated on the live
        ``positions``/``masses`` so ``jax.grad``/``jax.vjp`` over this method give
        exact gradients at fixed topology (see ``docs/differentiable_fmm_design.md``).

        Because tree construction (:meth:`prepare_state`) is not traceable, build
        ``state`` once, concretely, then differentiate this method::

            state = fmm.prepare_state(pos0, mass0, max_order=p, leaf_size=...)
            g = jax.grad(lambda p, m:
                (fmm.differentiable_accelerations(state, p, m) ** 2).sum()
            )(pos, mass)

        Parameters
        ----------
        state : FMMPreparedState
            Topology + buffers built by :meth:`prepare_state`. Captured as a
            constant; must be a radix ``FMMPreparedState`` with
            ``expansion_basis == "solidfmm"``.
        positions, masses : Array
            Differentiated inputs, in the ORIGINAL (unsorted) particle order.
        target_indices : Optional[Array]
            Optional targets to return (all particles are still sources).
        jit_traversal : bool
            Kept ``False`` on the gradient path (fully pure-JAX ``evaluate_tree``).

        Notes
        -----
        Bare ``jax.grad``/``jax.vjp`` over this method give exact gradients, and are
        the recommended usage -- the inner numeric kernels are already jit-compiled.
        Wrapping the *entire* call in an outer ``jax.jit`` is supported at moderate N
        but can fail at large N (the re-run of the prepared sweeps retains host-side
        ops). See ``docs/differentiable_fmm_design.md`` ("jit limitation").

        This method forces the **ungrouped** M2L (``force_ungrouped_farfield``): the
        grouped/class-major M2L builds its pair classes on the host, so it is not
        traceable and used to break the *bare* reverse pass -- not just the outer-jit
        one -- at ``n >= 65536``, where the resolver auto-enables it. Expansion
        centres are untouched.

        Note this is **not** merely a different execution strategy: the grouped M2L
        quantises pair displacements onto a lattice and applies one representative
        displacement per class, so it is an approximation, while the ungrouped M2L
        uses each pair's exact displacement. The grad path is therefore at least as
        accurate as the grouped forward (measured ~6.6x closer to the exact direct
        sum on a deep-tree config), but a caller who explicitly requests
        ``grouped_interactions=True`` gets a forward force that differs from the
        force this gradient is taken of, at the grouped path's own accuracy.

        The M2L uses the pure-JAX path by default. Setting
        ``JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1`` opts into the fused-Pallas M2L
        fast lane on an Ampere+ (sm_80) GPU: the fused kernels now carry a
        ``custom_vjp`` (Pallas forward + autodiff-of-twin reverse), so the fast lane
        is differentiable. It falls back to the pure-JAX M2L on unsupported hardware.

        The near field uses the bucketed pure-JAX edge-list kernel by default.
        ``JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE=1`` re-expresses it leaf-major
        and routes it through the radix fast lane instead. Same edge set, same
        force -- a different traversal -- and profiling put the near field at ~83%
        of this method's forward and ~91% of its reverse, so it is where the
        remaining time is.

        Which reverse you get depends on ``use_pallas``, and the difference is
        large: with ``use_pallas=True`` on an Ampere+ GPU the lane is the fused
        Pallas kernel wrapped in a ``custom_vjp`` whose backward is the **analytic
        O(N) leaf-pair reverse**; with ``use_pallas=False`` it is the tiled pure-JAX
        prepacked kernel, differentiated by ordinary autodiff. Only the former buys
        the O(N) reverse memory. Opt-in either way: measure on your hardware before
        switching, because the leaf-major traversal also gathers and scatters and
        its in-context cost is not predicted by isolated micro-benchmarks.

        Returns
        -------
        Array
            ``(N, 3)`` accelerations in the original input order.
        """
        if isinstance(state, LargeNPreparedState):
            # Large-N production path. NOT a variant of the radix branch below: the
            # large-N state carries no interaction list and a compat ``downward``
            # view with no ``.locals``, so the frozen M2L list comes from
            # ``compact_far_pairs`` and the near field is the fused Pallas fast lane
            # driven through its ``custom_vjp``. See runtime/_large_n_grad.py.
            from ._large_n_grad import (
                evaluate_large_n_state_at_positions_and_masses_sorted,
                prepare_large_n_grad_plan,
            )

            if target_indices is not None:
                raise NotImplementedError(
                    "differentiable_accelerations does not support target_indices "
                    "on the large-N path (matching evaluate_large_n_state)."
                )
            plan = (
                grad_plan
                if grad_plan is not None
                else prepare_large_n_grad_plan(self, state)
            )
            inverse_permutation_host = np.asarray(
                jax.device_get(state.inverse_permutation)
            )
            forward_permutation = jnp.asarray(
                np.argsort(inverse_permutation_host, kind="stable"), dtype=INDEX_DTYPE
            )
            positions_sorted = jnp.asarray(positions, dtype=state.working_dtype)[
                forward_permutation
            ]
            masses_sorted = jnp.asarray(masses, dtype=state.working_dtype)[
                forward_permutation
            ]
            accelerations_sorted = (
                evaluate_large_n_state_at_positions_and_masses_sorted(
                    self,
                    state,
                    positions_sorted,
                    masses_sorted,
                    plan=plan,
                )
            )
            output_dtype = (
                state.input_dtype
                if jnp.issubdtype(state.input_dtype, jnp.floating)
                else state.working_dtype
            )
            return jnp.asarray(accelerations_sorted)[state.inverse_permutation].astype(
                output_dtype
            )
        if getattr(state, "expansion_basis", None) != "solidfmm":
            raise NotImplementedError(
                "differentiable_accelerations currently supports "
                "expansion_basis='solidfmm' (basis 'complex'/'solidfmm' or 'real') "
                f"only; got {getattr(state, 'expansion_basis', None)!r}."
            )
        # The fused Pallas M2L kernels now carry a custom_vjp reverse rule (the
        # forward is the Pallas kernel; the reverse is autodiff of the pure-jnp
        # twin -- see jaccpot/pallas/m2l_{complex,real}_fused.py), so
        # JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS is an OPT-IN fast lane on the grad
        # path rather than a hard reject. When set (and the GPU is Ampere+), the
        # M2L dispatch routes through the differentiable fused kernel; otherwise
        # the default pure-JAX M2L runs. The near-field is still forced to the
        # bucketed pure-JAX path below (the fused-Pallas near-field ROI does not
        # justify wrapping it at the N the reverse pass bounds -- see
        # docs/differentiable_fmm_design.md). Left off by default until proven.

        # Reorder live inputs into Morton/sorted order using the FROZEN integer
        # permutation. ``state.inverse_permutation`` maps sorted -> original
        # (original[i] == sorted[inverse_permutation[i]]); its inverse ``fwd``
        # maps original -> sorted, so ``positions_sorted = positions[fwd]``.
        # Resolve ``fwd`` on the HOST from the concrete prepared permutation so
        # it embeds as a compile-time constant: a traced scatter would not be
        # constant-folded under ``jax.jit`` for large N and would force
        # concretization downstream. The gather VJP is scatter-add, so
        # cotangents flow to positions/masses but never to the index array.
        inverse_permutation_host = np.asarray(jax.device_get(state.inverse_permutation))
        forward_permutation = jnp.asarray(
            np.argsort(inverse_permutation_host, kind="stable"), dtype=INDEX_DTYPE
        )
        positions_sorted = jnp.asarray(positions, dtype=state.working_dtype)[
            forward_permutation
        ]
        masses_sorted = jnp.asarray(masses, dtype=state.working_dtype)[
            forward_permutation
        ]
        # The seam already returns accelerations in the original input order.
        # Default to the vectorized "bucketed" near-field: it is bit-identical to
        # the default "baseline" scan (which is gated to large-N only) but ~600x
        # faster at moderate N and has a cheaper reverse pass. At very large N its
        # higher memory footprint may matter; the reverse pass bounds N here
        # anyway. See docs/differentiable_fmm_design.md.
        return self._evaluate_prepared_state_at_positions_and_masses_sorted(
            state,
            positions_sorted,
            masses_sorted,
            target_indices=target_indices,
            jit_traversal=jit_traversal,
            nearfield_mode_override=(
                "fast_lane"
                if _env_flag(_DIFFERENTIABLE_NEARFIELD_FAST_LANE, False)
                else "bucketed"
            ),
        )

    def _evaluate_leaf_major_nearfield(
        self: "FastMultipoleMethod",
        tree: Tree,
        neighbor_list: NodeNeighborList,
        nearfield_interop: Optional[NearfieldInteropData],
        positions: Array,
        masses: Array,
        *,
        max_leaf_size: int,
        return_potential: bool,
    ) -> Array:
        """Near field via the leaf-major radix fast lane instead of the edge list.

        The payload is a pure function of the frozen topology, so it is built
        once per neighbour list, on the host, and memoized; only the
        position/mass gathers are re-executed per call. ``differentiable=True``
        costs nothing on the forward (the ``custom_vjp`` forward *is* the same
        kernel) and is what makes the lane usable under ``jax.grad`` at all,
        since ``pallas_call`` has no autodiff rule.

        The topology is read straight off ``tree``/``neighbor_list`` when the
        state carries no interop view -- deliberately NOT through
        ``_build_nearfield_interop_data``, whose device-side index work would be
        traced (and so unusable as a static shape) under an outer ``jax.jit``.
        """
        if bool(return_potential):
            raise NotImplementedError(
                "nearfield_mode='fast_lane' is an acceleration-only lane; it is "
                "the differentiable near field, and the potential half of the "
                "leaf-pair custom_vjp is not wired. Use 'bucketed' or 'baseline' "
                "for potentials."
            )
        payload = leaf_major_nearfield_payload_cached(
            num_particles=int(positions.shape[0]),
            max_leaf_size=int(max_leaf_size),
            **nearfield_topology_arrays(tree, neighbor_list, nearfield_interop),
        )
        return compute_leaf_p2p_accelerations_radix_fast_lane(
            positions_sorted=positions,
            masses_sorted=masses,
            payload=payload,
            G=self.G,
            softening=float(self.softening),
            return_potential=False,
            use_pallas=bool(getattr(self, "use_pallas", False)),
            differentiable=True,
        )

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
        nearfield_mode_override: Optional[str] = None,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Combine far- and near-field effects for leaf particles.

        ``nearfield_mode_override`` forces the near-field execution mode instead
        of the policy resolution (used by the differentiable path to select the
        vectorized ``"bucketed"`` near-field, which is bit-identical to the
        default ``"baseline"`` scan but orders of magnitude faster and has a
        cheaper reverse pass). ``None`` keeps the resolved policy unchanged.

        The extra value ``"fast_lane"`` is not a mode of the edge-list kernel at
        all: it re-expresses the near field leaf-major and routes it through
        :func:`compute_leaf_p2p_accelerations_radix_fast_lane`, which owns the
        analytic O(N) leaf-pair reverse. Same edge set, same force; a different
        traversal. See :mod:`jaccpot.runtime._nearfield_fastlane`.
        """

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
        nearfield_mode = (
            str(nearfield_mode_override).strip().lower()
            if nearfield_mode_override is not None
            else self._resolve_nearfield_mode(num_particles=int(positions.shape[0]))
        )
        if nearfield_mode == "fast_lane":
            near = self._evaluate_leaf_major_nearfield(
                tree,
                neighbor_list,
                nearfield_interop,
                positions,
                masses,
                max_leaf_size=resolved_max_leaf,
                return_potential=return_potential,
            )
        else:
            nearfield_view = (
                _build_nearfield_interop_data(tree, neighbor_list)
                if nearfield_interop is None
                else nearfield_interop
            )
            nearfield_edge_chunk_size = self._resolve_nearfield_edge_chunk_size(
                num_particles=int(positions.shape[0]),
                nearfield_mode=nearfield_mode,
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
