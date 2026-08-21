"""EvaluateMixin: fmm_evaluate methods extracted from the FMMEngine
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

# pyright: reportPossiblyUnboundVariable=false
#
# Scoped to this ONE rule, not to type checking of this file. Every instance here
# is the same reviewed pattern: `pot_sorted` and `potentials` are bound and read
# under the same `return_potential` test, in an evaluation dispatch that tests it
# in several sibling branches. All seven were checked individually and none can be
# unbound (audit E.4 bucket D).
#
# Suppressed at file scope rather than per line because a trailing
# `# pyright: ignore` does not survive `black` here -- it wraps these subscript
# expressions and carries the comment onto the closing bracket, off the line the
# error is reported on, so the suppression silently stops working. The per-line
# alternatives are a `None` sentinel (which only moves the complaint to
# `jnp.asarray`) or a temporary per use, and neither is worth reshaping a hot
# dispatch path for. The cost of this form is real and worth naming: a genuinely
# unbound local in this file would no longer be reported by this rule.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Float, Int, jaxtyped
from yggdrax.interactions import NodeNeighborList
from yggdrax.tree import Tree

from jaccpot._jax_compat import Tracer
from jaccpot.config import GradConfig
from jaccpot.downward.local_expansions import LocalExpansionData, TreeDownwardData
from jaccpot.nearfield._fast_lane import (
    compute_leaf_p2p_accelerations_radix_fast_lane,
)
from jaccpot.nearfield.near_field import compute_leaf_p2p_accelerations

from ._large_n_pipeline import evaluate_large_n_state
from ._large_n_types import LargeNPreparedState
from ._nearfield_fastlane import (
    leaf_major_nearfield_payload_cached,
    nearfield_topology_arrays,
)
from .dtypes import INDEX_DTYPE
from .fmm_caches import _contains_tracer
from .fmm_state import FMMPreparedState, _octree_farfield_eval_inputs
from .grad_options import grad_option_overrides, resolve_grad_options
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

if TYPE_CHECKING:  # pragma: no cover - annotations only, no runtime import
    # `_fmm_impl` imports this mixin, and `_large_n_grad` is already deferred to a
    # function-local import below for compile-time reasons, so both must stay out of
    # the module-scope import graph. Inheriting `_EngineBase` makes the mixin *be* the
    # engine under a type checker, so every `self.<engine attribute>` resolves; at
    # runtime the alias is `object`, leaving the MRO exactly as it was. See E.2.
    from ._fmm_impl import FMMEngine, PreparedStateLike
    from ._large_n_grad import LargeNGradPlan

    _EngineBase = FMMEngine
else:  # pragma: no cover - annotations only, never an import at runtime
    _EngineBase = object

__all__ = [
    "EvaluateMixin",
]

_OUTER_JIT_WARNED = False


def _warn_if_traced_under_an_outer_jit(positions: Any, masses: Any) -> None:
    """Say, once, that an outer ``jax.jit`` here means a long compile.

    XLA gives no progress output, so the compile of this pipeline reads exactly
    like a hang -- and it defeats every ``try: jit / except: fall back``, because
    slowness is not an exception. Measured on an A100 at N=4096, leaf 32, p=4,
    fp64: 19.1 s to compile the forward on ``preset="accurate"``, 159.6 s for the
    gradient, and a single inner module (``jit__accumulate_m2l_fullbatch``)
    taking 3m28s on the default preset. Those are real one-time costs, not
    failures -- but a user watching a silent terminal has no way to know that,
    and the tranche-1 measurement recorded it as "did not finish".

    Warned rather than raised: an outer jit is legitimate and, on the radix path,
    worth it (measured 264x on the steady-state forward). The warning names
    ``differentiable_step_fn``, which pays the same compile at a point the caller
    chose instead of inside a timed region.

    Parameters
    ----------
    positions : Any
        Candidate positions. Typed ``Any`` because the whole question is whether
        it is a tracer -- an ``Array`` annotation would be checked before the
        function could ask.
    masses : Any
        Candidate masses, on the same terms. Either being a tracer is enough.
    """

    global _OUTER_JIT_WARNED
    if _OUTER_JIT_WARNED:
        return
    if not any(isinstance(value, Tracer) for value in (positions, masses)):
        return
    _OUTER_JIT_WARNED = True
    warnings.warn(
        "differentiable_accelerations is being traced (an outer jax.jit or "
        "jax.grad is in charge). Compiling this pipeline takes tens of seconds "
        "to minutes and XLA reports no progress, so it looks like a hang: "
        "measured on an A100 at N=4096/fp64, 19 s for the forward, 160 s for the "
        "gradient, and one inner module alone at 3m28s. It is a one-time cost. "
        "To pay it deliberately rather than inside a timed region, use "
        "FMMEngine.differentiable_step_fn(state, compile_now=(pos, "
        "mass)). This warning is issued once per process.",
        UserWarning,
        stacklevel=3,
    )


class EvaluateMixin(_EngineBase):
    @jaxtyped(typechecker=beartype)
    def compute_accelerations(
        self,
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        target_indices: Optional[Int[Array, "t"]] = None,
        bounds: Optional[Tuple[Float[Array, "3"], Float[Array, "3"]]] = None,
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
        positions : Float[Array, 'n 3']
            Source and target particle positions.
        masses : Float[Array, 'n']
            Particle masses aligned with ``positions``.
        target_indices : Optional[Int[Array, 't']]
            Optional 1D index array selecting which target-particle outputs to
            return. All particles are still used as source masses.
        bounds : Optional[Tuple[Float[Array, '3'], Float[Array, '3']]]
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
        max_acc_derivative_order : int
            Highest acceleration-derivative order to return alongside the
            accelerations. ``0`` (the default) returns none. Non-zero requires
            ``expansion_basis="solidfmm"`` and is rejected on the traced
            fallback; see ``Raises``.

        Returns
        -------
        Union[Array, Tuple[Array, Array], Tuple[Array, PackedAccelerationDerivatives], Tuple[Array, Array, PackedAccelerationDerivatives]]
            Accelerations ``[N, 3]`` for all particles, or ``[len(target_indices), 3]``
            when ``target_indices`` is given. ``return_potential`` inserts the
            potentials next, and ``max_acc_derivative_order > 0`` appends a
            :class:`PackedAccelerationDerivatives` last, so the four shapes are
            ``a``, ``(a, pot)``, ``(a, derivs)`` and ``(a, pot, derivs)``.

        Raises
        ------
        NotImplementedError
            If ``positions`` or ``masses`` is a tracer (an outer ``jit``/``grad``
            is in charge) *and* either ``return_potential`` is ``True`` or
            ``max_acc_derivative_order`` is non-zero. The traced path falls back
            to a direct sum that supports neither.

        Notes
        -----
        This is the forward-only convenience entry point. For gradients use
        :meth:`differentiable_accelerations`, or
        :meth:`differentiable_step_fn` to pay the compile at a chosen moment --
        under an outer trace this method falls back to an O(N^2) direct sum
        rather than running the FMM pipeline, which is correct but not what a
        caller timing "the FMM" is measuring.
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
        self,
        state: PreparedStateLike,
        *,
        target_indices: Optional[Int[Array, "t"]] = None,
        return_potential: bool = False,
        jit_traversal: bool = True,
        max_acc_derivative_order: int = 0,
    ) -> Union[
        Array,
        Tuple[Array, Array],
        Tuple[Array, PackedAccelerationDerivatives],
        Tuple[Array, Array, PackedAccelerationDerivatives],
    ]:
        """Evaluate accelerations/potentials for all particles or targets.

        Evaluates the expansions ``state`` already carries, so it takes no live
        positions or masses and must not be differentiated -- see
        :meth:`differentiable_accelerations` for that. Dispatches on the state's
        type: a ``LargeNPreparedState`` goes to the large-N lane, everything else
        to the radix path.

        Parameters
        ----------
        state : PreparedStateLike
            Prepared tree and interaction lists, of either kind.
        target_indices : Optional[Int[Array, 't']]
            1-D indices selecting which targets to return; all particles remain
            sources.
        return_potential : bool
            Also return the potential.
        jit_traversal : bool
            Use the compiled traversal.
        max_acc_derivative_order : int
            Packed spatial derivative tower depth; ``0`` returns no tower.

        Returns
        -------
        Union[Array, Tuple[Array, Array], Tuple[Array, PackedAccelerationDerivatives], Tuple[Array, Array, PackedAccelerationDerivatives]]
            Accelerations, with potentials inserted next under
            ``return_potential`` and the packed derivative tower appended last.

        Raises
        ------
        ValueError
            If the requested combination is not supported for this state.
        NotImplementedError
            For options the large-N lane does not implement, such as a target
            subset on that path.
        """

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
        tracing_targets = isinstance(state.positions_sorted, Tracer) or isinstance(
            resolved_target_indices, Tracer
        )
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

        self._record_force_scale_from_evaluation(
            state=state,
            evaluation=evaluation,
            full_evaluation=(resolved_target_indices is None),
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
                return (
                    accelerations,
                    potentials.astype(output_dtype),
                    derivatives,
                )
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
        self,
        state: FMMPreparedState,
        positions_sorted: Float[Array, "n 3"],
        *,
        masses_sorted: Optional[Float[Array, "n"]] = None,
        target_indices: Optional[Int[Array, "t"]] = None,
        jit_traversal: bool = True,
        nearfield_mode_override: Optional[str] = None,
        nearfield_reverse_options: Optional[Any] = None,
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

        Parameters
        ----------
        state : FMMPreparedState
            Frozen topology. Read for its discrete artifacts only; its baked
            expansions are deliberately not used.
        positions_sorted : Float[Array, 'n 3']
            Live positions ``[N, 3]`` in the state's Morton order. Differentiable.
        masses_sorted : Optional[Float[Array, 'n']]
            Live masses ``[N]``, likewise differentiable. ``None`` reuses the
            state's, which is the difference between a position-only gradient and
            one that also flows to mass.
        target_indices : Optional[Int[Array, 't']]
            Subset of targets to return; all particles remain sources.
        jit_traversal : bool
            Use the compiled traversal.
        nearfield_mode_override : Optional[str]
            Force the near-field mode; ``None`` keeps the resolved policy.
        nearfield_reverse_options : Optional[Any]
            Reverse-pass tuning for the leaf-major lane; inert elsewhere. ``Any``
            to keep the grad types out of this module.
        force_ungrouped_farfield : bool
            Take the ungrouped far-field path regardless of policy.

        Returns
        -------
        Array
            Accelerations in the state's sorted order.

        Raises
        ------
        ValueError
            If the supplied arrays do not match the state's particle count or
            ordering.
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
        tracing_targets = isinstance(positions_sorted_arr, Tracer) or isinstance(
            resolved_target_indices, Tracer
        )
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
                nearfield_reverse_options=nearfield_reverse_options,
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
    def differentiable_accelerations(
        self,
        state: Union[FMMPreparedState, LargeNPreparedState],
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        target_indices: Optional[Int[Array, "t"]] = None,
        jit_traversal: bool = False,
        grad_plan: Optional["LargeNGradPlan"] = None,
        grad_config: Optional[GradConfig] = None,
    ) -> Array:
        """Exact fixed-topology gradients of the FMM force w.r.t. positions/masses.

        End-to-end differentiable single-GPU FMM acceleration. The tree topology
        (Morton order, node membership, MAC accept/reject, the M2L interaction
        list, and the near-field partition) is taken as a **constant** from the
        pre-built ``state``; the numeric pipeline (P2M, COM centers, M2M/M2L/L2L
        translations, L2P, near-field P2P) is re-evaluated on the live
        ``positions``/``masses`` so ``jax.grad``/``jax.vjp`` over this method give
        exact gradients at fixed topology (see ``docs/differentiable_fmm.md``).

        Because tree construction (:meth:`prepare_state`) is not traceable, build
        ``state`` once, concretely, then differentiate this method::

            state = fmm.prepare_state(pos0, mass0, max_order=p, leaf_size=...)
            g = jax.grad(lambda p, m:
                (fmm.differentiable_accelerations(state, p, m) ** 2).sum()
            )(pos, mass)

        Parameters
        ----------
        state : Union[FMMPreparedState, LargeNPreparedState]
            Topology + buffers built by :meth:`prepare_state`, captured as a
            constant. A radix ``FMMPreparedState`` needs a solidfmm basis; a
            ``LargeNPreparedState`` (the ``large_n_gpu`` preset) additionally
            needs ``retain_far_pairs_for_grad=True`` so the frozen M2L pair list
            survives ``prepare_state``.
        positions : Float[Array, 'n 3']
            Differentiated source/target positions ``[N, 3]``, in the ORIGINAL
            (unsorted) particle order -- this method applies ``state``'s
            permutation itself.
        masses : Float[Array, 'n']
            Differentiated masses ``[N]``, in the same original order.
        target_indices : Optional[Int[Array, 't']]
            Optional targets to return (all particles are still sources). Not
            supported on the large-N path.
        jit_traversal : bool
            Kept ``False`` on the gradient path (fully pure-JAX ``evaluate_tree``).
        grad_plan : Optional[LargeNGradPlan]
            Large-N path only. The frozen discriminators returned by
            :func:`~jaccpot.runtime._large_n_grad.prepare_large_n_grad_plan`.
            Build it once and pass it here to hoist the validation and pair-list
            setup out of an optimisation loop; omitted, it is rebuilt per call.
        grad_config : Optional[GradConfig]
            Execution options for the gradient path -- most importantly which
            near-field traversal to take. The default ``"auto"`` lane picks the
            leaf-major fast lane at N >= 100000, because the bucketed reverse
            OOMs at galaxy scale (30 GB peak at 200k against 6.8 GB). See
            :class:`~jaccpot.config.GradConfig`; every field also has a
            ``JACCPOT_*`` environment fallback, and an explicit field wins.

        Returns
        -------
        Array
            ``(N, 3)`` accelerations in the original input order.

        Notes
        -----
        Bare ``jax.grad``/``jax.vjp`` over this method give exact gradients, and are
        the recommended usage -- the inner numeric kernels are already jit-compiled.
        Wrapping the *entire* call in an outer ``jax.jit`` is supported at moderate N
        but can fail at large N (the re-run of the prepared sweeps retains host-side
        ops). See ``docs/differentiable_fmm.md`` ("jit limitation").

        This method forces the **ungrouped** M2L: the grouped/class-major M2L
        builds its pair classes on the host, so it is not traceable and used to
        break the *bare* reverse pass -- not just the outer-jit one -- at
        ``n >= 65536``, where the resolver auto-enables it. Expansion centres are
        untouched. That is not merely a different execution strategy: the grouped
        M2L quantises pair displacements onto a lattice and applies one
        representative displacement per class, so it is an approximation, while
        the ungrouped M2L uses each pair's exact displacement. The grad path is
        therefore at least as accurate as the grouped forward (measured ~6.6x
        closer to the exact direct sum on a deep-tree config), but a caller who
        explicitly requests ``grouped_interactions=True`` gets a forward force
        that differs from the force this gradient is taken of.
        """
        _warn_if_traced_under_an_outer_jit(positions, masses)
        options = resolve_grad_options(
            grad_config,
            num_particles=int(jnp.asarray(positions).shape[0]),
            supports_fast_lane=not isinstance(state, LargeNPreparedState),
        )
        # The fused-M2L and analytic-VJP gates are read at trace time by code
        # several layers down with no argument channel; the override makes those
        # GradConfig fields effective for exactly this call.
        with grad_option_overrides(options):
            if isinstance(state, LargeNPreparedState):
                return self._differentiable_accelerations_large_n(
                    state,
                    positions,
                    masses,
                    target_indices=target_indices,
                    grad_plan=grad_plan,
                    options=options,
                )
            return self._differentiable_accelerations_radix(
                state,
                positions,
                masses,
                target_indices=target_indices,
                jit_traversal=jit_traversal,
                options=options,
            )

    def differentiable_step_fn(
        self,
        state: Union[FMMPreparedState, LargeNPreparedState],
        *,
        target_indices: Optional[Array] = None,
        grad_config: Optional[GradConfig] = None,
        jit_traversal: bool = False,
        compile_now: Optional[Tuple[Array, Array]] = None,
    ) -> Callable[[Array, Array], Array]:
        """Return a compiled ``f(positions, masses) -> accelerations``.

        The step seam for a training or inference loop: build ``state`` once with
        :meth:`prepare_state`, then call this once, then call the result per step.
        The compile is paid once and amortised; the returned function is
        differentiable, so ``jax.grad`` over it works and is compiled too.

        Why this exists rather than "just wrap it in ``jax.jit``". Eagerly,
        ``differentiable_accelerations`` re-traces its whole pipeline on every
        call, and the re-tracing dominates. Measured on an idle A100 at N=4096,
        leaf 64, p=4, real basis, evaluating a prebuilt tree:

        ==============  ===========  =============  =========  =========
        preset          eager        compile once   compiled   factor
        ==============  ===========  =============  =========  =========
        ``accurate``       6.681 s        18.88 s    0.0110 s      607x
        ``large_n_gpu``    2.843 s         6.11 s    4.036 s       0.7x
        ==============  ===========  =============  =========  =========

        Both numbers are reported because they disagree, and the disagreement is
        the useful part: on the radix path compiling is transformative, while on
        the large-N production path the fast-lane kernels are already compiled
        individually and folding them into one module is slightly *worse*. The
        reverse pass on ``large_n_gpu`` behaves the same way (eager 12.62 s,
        compiled 15.11 s after a 19.3 s compile). So this is a seam to reach for
        on the radix path, and to measure before adopting on the large-N one.

        The forward is bit-identical on ``large_n_gpu`` (max abs diff 0.0) and
        agrees to 2.3e-12 on ``accurate`` at fp64, which is reassociation.

        Parameters
        ----------
        state : Union[FMMPreparedState, LargeNPreparedState]
            A prepared state from :meth:`prepare_state`. Closed over as a
            constant, exactly as :meth:`differentiable_accelerations` treats it.
        target_indices : Optional[Array]
            Optional targets to return; all particles remain sources. Passed
            through to :meth:`differentiable_accelerations` unchanged.
        grad_config : Optional[GradConfig]
            Grad-path options, passed through unchanged but resolved **once**,
            here, rather than per call -- so a later environment change does not
            take effect on an already-returned step function.
        jit_traversal : bool
            Passed through to :meth:`differentiable_accelerations` unchanged.
        compile_now : Optional[Tuple[Array, Array]]
            ``(positions, masses)`` to compile against immediately, so the
            one-time cost lands here rather than inside the first timed step.
            Recommended when benchmarking.

        Returns
        -------
        Callable[[Array, Array], Array]
            A compiled ``f(positions, masses) -> accelerations`` closing over
            ``state``. Differentiable, so ``jax.grad`` over it works and is
            itself compiled. Positions and masses are in the original (unsorted)
            particle order, as for :meth:`differentiable_accelerations`.

        Raises
        ------
        TypeError
            If ``state`` holds tracers, i.e. it was built inside a trace. Tree
            construction is host-side and not traceable, so such a state cannot
            be a compile-time constant. Raised here rather than deep inside the
            trace, where it surfaces as a leaked-tracer error naming an internal
            cache.
        """

        from .fmm_caches import _contains_tracer

        if _contains_tracer(state):
            raise TypeError(
                "differentiable_step_fn needs a concrete prepared state, but this "
                "one holds tracers -- it was built inside a jax transform. "
                "prepare_state does host-side tree construction and is not "
                "traceable; build the state outside, then call this."
            )
        options = resolve_grad_options(
            grad_config,
            num_particles=int(
                getattr(getattr(state, "inverse_permutation", None), "shape", (0,))[0]
            ),
            supports_fast_lane=not isinstance(state, LargeNPreparedState),
        )

        def _step(positions: Array, masses: Array) -> Array:
            with grad_option_overrides(options):
                if isinstance(state, LargeNPreparedState):
                    return self._differentiable_accelerations_large_n(
                        state,
                        positions,
                        masses,
                        target_indices=target_indices,
                        grad_plan=None,
                        options=options,
                    )
                return self._differentiable_accelerations_radix(
                    state,
                    positions,
                    masses,
                    target_indices=target_indices,
                    jit_traversal=jit_traversal,
                    options=options,
                )

        compiled = jax.jit(_step)
        if compile_now is not None:
            # Warm the jit cache by CALLING it, not by returning
            # `.lower(...).compile()`: an AOT-compiled object is specialised to
            # one signature and rejects JAX transformations, so returning it
            # would make the seam undifferentiable -- the one thing it exists
            # for. (`jax.grad` over it fails with "Cannot apply JAX
            # transformations to a function lowered and compiled for a
            # particular signature".)
            jax.block_until_ready(compiled(*compile_now))
        return compiled

    def _forward_permutation(
        self,
        state: Union[FMMPreparedState, LargeNPreparedState],
    ) -> Array:
        """Original-order -> Morton-sorted gather index, memoized on the state.

        ``state.inverse_permutation`` maps sorted -> original
        (``original[i] == sorted[inverse_permutation[i]]``); its inverse ``fwd``
        maps original -> sorted, so ``positions_sorted = positions[fwd]``. The
        gather's VJP is a scatter-add, and ``fwd`` is integer, so cotangents flow
        to positions/masses but never to the index array.

        Resolved on the HOST from the concrete prepared permutation so it embeds
        as a compile-time constant: a traced ``argsort``/scatter would not be
        constant-folded under ``jax.jit`` at large N and would force
        concretization downstream.

        Memoized because it is frozen topology. Uncached this was a device-to-host
        copy plus an O(N log N) host sort on **every call** -- once per step of an
        optimisation loop, for a value that cannot change while ``state`` is
        alive. The cache is a single slot keyed by the permutation array's
        identity, and holds a strong reference to it, so a live entry pins its own
        key and no freed array's id can be reused against a stale entry.

        Parameters
        ----------
        state : Union[FMMPreparedState, LargeNPreparedState]
            Prepared state whose ``inverse_permutation`` is inverted. Its array
            identity is the cache key.

        Returns
        -------
        Array
            Integer gather index ``fwd`` with ``positions_sorted ==
            positions[fwd]``.
        """
        permutation = state.inverse_permutation
        cached = getattr(self, "_forward_permutation_memo", None)
        if cached is not None and cached[0] is permutation:
            # Never hand back a cached tracer. A memo written during a trace that
            # outlives it makes the NEXT call fail with UnexpectedTracerError
            # pointing here, which reads as a bug in the caller. Dropping the
            # entry costs one host sort and cannot be wrong.
            if not _contains_tracer(cached):
                return cached[1]
            self._forward_permutation_memo = None
        inverse_permutation_host = np.asarray(jax.device_get(permutation))
        forward_permutation = jnp.asarray(
            np.argsort(inverse_permutation_host, kind="stable"), dtype=INDEX_DTYPE
        )
        # Only memoise concrete values, for the same reason: this method is called
        # from inside `jax.jit`/`jax.grad` traces by design (see
        # differentiable_step_fn), and the memo lives on the solver, which
        # outlives any single trace.
        if not _contains_tracer((permutation, forward_permutation)):
            self._forward_permutation_memo = (permutation, forward_permutation)
        return forward_permutation

    def _sorted_inputs(
        self,
        state: Union[FMMPreparedState, LargeNPreparedState],
        positions: Array,
        masses: Array,
    ) -> Tuple[Array, Array]:
        """Gather live positions/masses into Morton order at frozen topology.

        Both are cast to the state's working dtype before the gather, so a
        caller passing fp64 into an fp32 state does not silently widen the
        pipeline.

        Parameters
        ----------
        state : Union[FMMPreparedState, LargeNPreparedState]
            Supplies the permutation and the working dtype.
        positions : Array
            Live positions ``[N, 3]`` in the caller's original order.
        masses : Array
            Live masses ``[N]``, same order.

        Returns
        -------
        Tuple[Array, Array]
            ``(positions_sorted, masses_sorted)``. A gather, so its VJP is a
            scatter-add and cotangents flow back to both inputs.
        """
        forward_permutation = self._forward_permutation(state)
        return (
            jnp.asarray(positions, dtype=state.working_dtype)[forward_permutation],
            jnp.asarray(masses, dtype=state.working_dtype)[forward_permutation],
        )

    def _grad_output_dtype(
        self,
        state: Union[FMMPreparedState, LargeNPreparedState],
    ) -> Any:
        """Return accelerations in the caller's float dtype where there is one.

        "Where there is one" is the whole condition: an integer or otherwise
        non-floating ``input_dtype`` cannot hold a gradient, so the working dtype
        is used rather than round-tripping the result through it.

        Parameters
        ----------
        state : Union[FMMPreparedState, LargeNPreparedState]
            Supplies both candidate dtypes.

        Returns
        -------
        Any
            ``state.input_dtype`` when it is a floating type, else
            ``state.working_dtype``.
        """
        if jnp.issubdtype(state.input_dtype, jnp.floating):
            return state.input_dtype
        return state.working_dtype

    def _differentiable_accelerations_large_n(
        self,
        state: LargeNPreparedState,
        positions: Array,
        masses: Array,
        *,
        target_indices: Optional[Array],
        grad_plan: Optional["LargeNGradPlan"],
        options: Any,
    ) -> Array:
        """Differentiable evaluation on the ``large_n_gpu`` production state.

        NOT a variant of the radix branch: the large-N state carries no
        interaction list and a compat ``downward`` view with no ``.locals``, so
        the frozen M2L list comes from ``compact_far_pairs`` and the near field is
        the fused Pallas fast lane driven through its ``custom_vjp``. The
        near-field lane is therefore not selectable here -- this state has exactly
        one differentiable near field. See ``runtime/_large_n_grad.py``.

        Parameters
        ----------
        state : LargeNPreparedState
            Frozen large-N topology. Carries compact far pairs instead of an
            interaction list.
        positions : Array
            Live positions ``[N, 3]``, differentiated.
        masses : Array
            Live masses ``[N]``, differentiated.
        target_indices : Optional[Array]
            Must be ``None`` here; the large-N lane has no target-subset path.
        grad_plan : Optional[LargeNGradPlan]
            Prebuilt validation and pair-list setup, so an optimisation loop can
            hoist it out. ``None`` builds it per call.
        options : Any
            Resolved :class:`~jaccpot.config.GradConfig`. Its ``nearfield_lane``
            is ignored on this state -- see above.

        Returns
        -------
        Array
            Accelerations ``(N, 3)`` in the caller's original order.

        Raises
        ------
        NotImplementedError
            For options this lane does not support, including a target subset.
        """
        from ._large_n_grad import (
            evaluate_large_n_state_at_positions_and_masses_sorted,
            prepare_large_n_grad_plan,
        )

        if target_indices is not None:
            raise NotImplementedError(
                "differentiable_accelerations does not support target_indices "
                "on the large-N path (matching evaluate_large_n_state)."
            )
        if (
            not options.nearfield_lane_was_auto
            and options.nearfield_lane != "fast_lane"
        ):
            raise NotImplementedError(
                "GradConfig(nearfield_lane="
                f"{options.nearfield_lane!r}) is not available on a "
                "LargeNPreparedState: its near field is the prepacked leaf-major "
                "Pallas lane, baked at prepare_state, and there is no bucketed "
                "edge list to fall back to. Use a radix FMMPreparedState if you "
                "need the bucketed near field."
            )
        plan = (
            grad_plan
            if grad_plan is not None
            else prepare_large_n_grad_plan(self, state)
        )
        positions_sorted, masses_sorted = self._sorted_inputs(state, positions, masses)
        accelerations_sorted = evaluate_large_n_state_at_positions_and_masses_sorted(
            self,
            state,
            positions_sorted,
            masses_sorted,
            plan=plan,
        )
        return jnp.asarray(accelerations_sorted)[state.inverse_permutation].astype(
            self._grad_output_dtype(state)
        )

    def _differentiable_accelerations_radix(
        self,
        state: FMMPreparedState,
        positions: Array,
        masses: Array,
        *,
        target_indices: Optional[Array],
        jit_traversal: bool,
        options: Any,
    ) -> Array:
        """Differentiable evaluation on a radix ``FMMPreparedState``.

        Re-runs P2M -> M2M -> M2L -> L2L -> L2P plus the near-field P2P on the
        live inputs against the frozen topology. The near-field lane comes from
        the resolved :class:`~jaccpot.config.GradConfig`:

        * ``"bucketed"`` -- the vectorized edge-list kernel. Bit-identical to the
          ``"baseline"`` per-leaf scan (which is gated to large N only) but ~600x
          faster at moderate N, with a cheaper reverse.
        * ``"fast_lane"`` -- the leaf-major traversal. Same edge set, same force;
          with ``use_pallas`` on Ampere+ its reverse is the analytic O(N)
          leaf-pair rule. This is the only lane whose reverse memory is O(N), and
          at N >= 10^5 it is the difference between a gradient and an OOM.

        The fused-Pallas M2L (``GradConfig.fused_m2l_pallas``) is a separate,
        orthogonal opt-in: the fused kernels carry their own ``custom_vjp``, so
        the flag simply routes the M2L dispatch through them, falling back to
        pure-JAX on non-Ampere hardware.

        Parameters
        ----------
        state : FMMPreparedState
            Frozen radix topology.
        positions : Array
            Live positions ``[N, 3]``, differentiated.
        masses : Array
            Live masses ``[N]``, differentiated.
        target_indices : Optional[Array]
            Subset of targets to return; all particles remain sources.
        jit_traversal : bool
            Kept ``False`` on the gradient path.
        options : Any
            Resolved :class:`~jaccpot.config.GradConfig`; ``nearfield_lane``
            selects between the two lanes described above.

        Returns
        -------
        Array
            Accelerations ``(N, 3)`` in the caller's original order.

        Raises
        ------
        NotImplementedError
            If the resolved lane or option combination has no implementation for
            this state.
        """
        if getattr(state, "expansion_basis", None) != "solidfmm":
            raise NotImplementedError(
                "differentiable_accelerations currently supports "
                "expansion_basis='solidfmm' (basis 'complex'/'solidfmm' or 'real') "
                f"only; got {getattr(state, 'expansion_basis', None)!r}."
            )
        positions_sorted, masses_sorted = self._sorted_inputs(state, positions, masses)
        # The seam already returns accelerations in the original input order, and
        # forces the ungrouped M2L (force_ungrouped_farfield defaults True there).
        return self._evaluate_prepared_state_at_positions_sorted(
            state,
            positions_sorted,
            masses_sorted=masses_sorted,
            target_indices=target_indices,
            jit_traversal=jit_traversal,
            nearfield_mode_override=options.nearfield_lane,
            nearfield_reverse_options=options.reverse,
            force_ungrouped_farfield=True,
        )

    def _evaluate_leaf_major_nearfield(
        self,
        tree: Tree,
        neighbor_list: NodeNeighborList,
        nearfield_interop: Optional[NearfieldInteropData],
        positions: Array,
        masses: Array,
        *,
        max_leaf_size: int,
        return_potential: bool,
        reverse_options: Optional[Any] = None,
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

        Parameters
        ----------
        tree : Tree
            Frozen tree, read for its topology.
        neighbor_list : NodeNeighborList
            Frozen neighbour lists; with ``tree`` this is what the memoized
            payload is keyed on.
        nearfield_interop : Optional[NearfieldInteropData]
            Prebuilt interop view; ``None`` reads the topology directly, which is
            the jit-safe route described above.
        positions : Array
            Live positions ``[N, 3]`` in sorted order.
        masses : Array
            Live masses ``[N]`` in sorted order.
        max_leaf_size : int
            Padded leaf width. Static -- it sizes the leaf-major blocks.
        return_potential : bool
            Also return the potential.
        reverse_options : Optional[Any]
            Reverse-pass tuning for this lane. ``Any`` to keep the grad types out
            of this module.

        Returns
        -------
        Array
            Near-field accelerations in sorted order, with potentials alongside
            when requested.

        Raises
        ------
        NotImplementedError
            If the lane is unavailable for this configuration -- notably when
            Pallas is required but the hardware is pre-Ampere.
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
            reverse_options=reverse_options,
        )

    @jaxtyped(typechecker=beartype)
    def evaluate_tree(
        self,
        tree: Tree,
        positions_sorted: Float[Array, "n 3"],
        masses_sorted: Float[Array, "n"],
        locals_or_downward: Union[LocalExpansionData, TreeDownwardData],
        neighbor_list: NodeNeighborList,
        *,
        nearfield_interop: Optional[NearfieldInteropData] = None,
        farfield_local_data: Optional[LocalExpansionData] = None,
        farfield_leaf_nodes: Optional[Int[Array, "leaves"]] = None,
        # `farfield_leaf_nodes` and `farfield_node_ranges` are NOT the same axis,
        # despite the shared prefix. Observed across the captured calls:
        # leaf_nodes (3,) with node_ranges (5, 2); (5,) with (9, 2); (9,) with
        # (17, 2) -- i.e. leaves and 2*leaves-1 nodes, which is what a binary
        # radix tree gives. Binding both to `leaves` would have been wrong. The
        # node axis is left anonymous for the same reason as
        # `near_field.py::node_ranges_override`: only the trailing 2 is invariant.
        farfield_node_ranges: Optional[Int[Array, "_ 2"]] = None,
        precomputed_target_leaf_ids: Optional[Int[Array, "pairs"]] = None,
        precomputed_source_leaf_ids: Optional[Int[Array, "pairs"]] = None,
        precomputed_valid_pairs: Optional[Bool[Array, "pairs"]] = None,
        precomputed_chunk_sort_indices: Optional[Int[Array, "chunks chunkflat"]] = None,
        precomputed_chunk_group_ids: Optional[Int[Array, "chunks chunkflat"]] = None,
        precomputed_chunk_unique_indices: Optional[
            Int[Array, "chunks chunkflat"]
        ] = None,
        max_leaf_size: Optional[int] = None,
        return_potential: bool = False,
        nearfield_mode_override: Optional[str] = None,
        nearfield_reverse_options: Optional[Any] = None,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Combine far- and near-field effects for leaf particles.

        ``nearfield_mode_override`` forces the near-field execution mode instead
        of the policy resolution (used by the differentiable path to select the
        vectorized ``"bucketed"`` near-field, which is orders of magnitude faster
        than the default ``"baseline"`` scan and has a cheaper reverse pass).
        ``None`` keeps the resolved policy unchanged.

        ``"bucketed"`` agrees with ``"baseline"`` **to a tolerance, not
        bit-exactly** -- the two differ in edge order, hence in accumulation
        order. Asserted by
        ``tests/integration/test_fmm.py::test_nearfield_bucketed_matches_baseline``
        over N x dtype x chunk size, at ``1e-6`` (fp32) and ``1e-13`` (fp64) --
        both derived from measured round-off. See
        :func:`~jaccpot.nearfield.near_field.compute_leaf_p2p_accelerations` for
        why the fp64 cases are the ones that can actually fail.

        The extra value ``"fast_lane"`` is not a mode of the edge-list kernel at
        all: it re-expresses the near field leaf-major and routes it through
        :func:`compute_leaf_p2p_accelerations_radix_fast_lane`, which owns the
        analytic O(N) leaf-pair reverse. Same edge set, same force; a different
        traversal. See :mod:`jaccpot.runtime._nearfield_fastlane`.

        Parameters
        ----------
        tree : Tree
            Built tree.
        positions_sorted : Float[Array, 'n 3']
            Morton-sorted positions ``[N, 3]``.
        masses_sorted : Float[Array, 'n']
            Morton-sorted masses ``[N]``.
        locals_or_downward : Union[LocalExpansionData, TreeDownwardData]
            Local expansions, or the downward result carrying them.
        neighbor_list : NodeNeighborList
            Leaf ordering and neighbour lists.
        nearfield_interop : Optional[NearfieldInteropData]
            Prebuilt near-field view; ``None`` builds one.
        farfield_local_data : Optional[LocalExpansionData]
            Far-field locals override; ``None`` uses those in
            ``locals_or_downward``.
        farfield_leaf_nodes : Optional[Int[Array, 'leaves']]
            Far-field leaf view override; ``None`` uses the radix view.
        farfield_node_ranges : Optional[Int[Array, '_ 2']]
            Far-field node-range override; ``None`` as above.
        precomputed_target_leaf_ids : Optional[Int[Array, 'pairs']]
            Per-edge target leaf ids. ``None`` means absent -- unlike the
            compiled kernel further down the stack, which opts in by *shape*.
        precomputed_source_leaf_ids : Optional[Int[Array, 'pairs']]
            Per-edge source leaf ids.
        precomputed_valid_pairs : Optional[Bool[Array, 'pairs']]
            Per-edge validity mask.
        precomputed_chunk_sort_indices : Optional[Int[Array, 'chunks chunkflat']]
            Chunk scatter sort permutation.
        precomputed_chunk_group_ids : Optional[Int[Array, 'chunks chunkflat']]
            Chunk scatter group ids.
        precomputed_chunk_unique_indices : Optional[Int[Array, 'chunks chunkflat']]
            Chunk unique-target indices.
        max_leaf_size : Optional[int]
            Padded leaf width; ``None`` measures it, which needs concrete values.
        return_potential : bool
            Also return the potential.
        nearfield_mode_override : Optional[str]
            Forces the near-field mode; see above. Accepted here and NOT by
            :meth:`evaluate_tree_compiled`, which resolves its own policy.
        nearfield_reverse_options : Optional[Any]
            Grad-path reverse tuning for the leaf-major lane; inert elsewhere.

        Returns
        -------
        Union[Array, Tuple[Array, Array]]
            Morton-sorted accelerations, or ``(accelerations, potentials)`` under
            ``return_potential``.
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
                reverse_options=nearfield_reverse_options,
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
        self,
        tree: Tree,
        positions_sorted: Float[Array, "n 3"],
        masses_sorted: Float[Array, "n"],
        locals_or_downward: Union[LocalExpansionData, TreeDownwardData],
        neighbor_list: NodeNeighborList,
        *,
        nearfield_interop: Optional[NearfieldInteropData] = None,
        farfield_local_data: Optional[LocalExpansionData] = None,
        farfield_leaf_nodes: Optional[Int[Array, "leaves"]] = None,
        # `farfield_leaf_nodes` and `farfield_node_ranges` are NOT the same axis,
        # despite the shared prefix. Observed across the captured calls:
        # leaf_nodes (3,) with node_ranges (5, 2); (5,) with (9, 2); (9,) with
        # (17, 2) -- i.e. leaves and 2*leaves-1 nodes, which is what a binary
        # radix tree gives. Binding both to `leaves` would have been wrong. The
        # node axis is left anonymous for the same reason as
        # `near_field.py::node_ranges_override`: only the trailing 2 is invariant.
        farfield_node_ranges: Optional[Int[Array, "_ 2"]] = None,
        precomputed_target_leaf_ids: Optional[Int[Array, "pairs"]] = None,
        precomputed_source_leaf_ids: Optional[Int[Array, "pairs"]] = None,
        precomputed_valid_pairs: Optional[Bool[Array, "pairs"]] = None,
        precomputed_chunk_sort_indices: Optional[Int[Array, "chunks chunkflat"]] = None,
        precomputed_chunk_group_ids: Optional[Int[Array, "chunks chunkflat"]] = None,
        precomputed_chunk_unique_indices: Optional[
            Int[Array, "chunks chunkflat"]
        ] = None,
        max_leaf_size: Optional[int] = None,
        return_potential: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        """JIT-compiled variant of :meth:`evaluate_tree`.

        Same evaluation, same arguments, minus the two overrides: the compiled
        traversal resolves its own near-field policy, so
        ``nearfield_mode_override`` and ``nearfield_reverse_options`` have no
        route in here. That is why the differentiable path uses
        ``jit_traversal=False`` -- it needs those overrides to reach the near
        field.

        Parameters
        ----------
        tree : Tree
            Built tree.
        positions_sorted : Float[Array, 'n 3']
            Morton-sorted positions ``[N, 3]``.
        masses_sorted : Float[Array, 'n']
            Morton-sorted masses ``[N]``.
        locals_or_downward : Union[LocalExpansionData, TreeDownwardData]
            Local expansions, or the downward result carrying them.
        neighbor_list : NodeNeighborList
            Leaf ordering and neighbour lists.
        nearfield_interop : Optional[NearfieldInteropData]
            Prebuilt near-field view; ``None`` builds one.
        farfield_local_data : Optional[LocalExpansionData]
            Far-field locals override.
        farfield_leaf_nodes : Optional[Int[Array, 'leaves']]
            Far-field leaf view override.
        farfield_node_ranges : Optional[Int[Array, '_ 2']]
            Far-field node-range override.
        precomputed_target_leaf_ids : Optional[Int[Array, 'pairs']]
            Per-edge target leaf ids.
        precomputed_source_leaf_ids : Optional[Int[Array, 'pairs']]
            Per-edge source leaf ids.
        precomputed_valid_pairs : Optional[Bool[Array, 'pairs']]
            Per-edge validity mask.
        precomputed_chunk_sort_indices : Optional[Int[Array, 'chunks chunkflat']]
            Chunk scatter sort permutation.
        precomputed_chunk_group_ids : Optional[Int[Array, 'chunks chunkflat']]
            Chunk scatter group ids.
        precomputed_chunk_unique_indices : Optional[Int[Array, 'chunks chunkflat']]
            Chunk unique-target indices.
        max_leaf_size : Optional[int]
            Padded leaf width. Required under a trace -- measuring it would need
            to concretize.
        return_potential : bool
            Also return the potential. Static: it changes the return arity.

        Returns
        -------
        Union[Array, Tuple[Array, Array]]
            Morton-sorted accelerations, or ``(accelerations, potentials)``.

        Raises
        ------
        ValueError
            If ``max_leaf_size`` is absent where the compiled path needs it
            concrete, or the supplied views do not align with the tree.
        """

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
