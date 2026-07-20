"""StrictRunMixin: fmm_strict_run methods extracted from the FastMultipoleMethod
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Callable, Tuple
from jaxtyping import Array
from yggdrax.interactions import DualTreeRetryEvent, NodeNeighborList
from yggdrax.tree import RadixTree

from ._large_n_pipeline import evaluate_large_n_state, prepare_large_n_state
from ._large_n_types import LargeNPreparedState
from .dtypes import INDEX_DTYPE
from .fmm_caches import _contains_tracer
from .fmm_state import (
    TreeBuilderConfig,
    _PrepareStateDualDownwardArtifacts,
    _PrepareStateTreeUpwardArtifacts,
    _RuntimeExecutionOverrides,
    _TopologyReuseEntry,
    _velocity_verlet_state_update,
)
from .kernels.core import _empty_interaction_storage_for_tree, _FarPairCOO


class StrictRunMixin:
    def refresh_prepared_state(
        self: "FastMultipoleMethod",
        prepared_state: PreparedStateLike,
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
        fused_device_mode: bool = False,
    ) -> PreparedStateLike:
        """Refresh prepared state under large-N/radix profile constraints."""
        if not self._is_large_n_gpu_production_profile():
            raise NotImplementedError(
                "refresh_prepared_state is currently supported only for "
                "preset='large_n_gpu', tree_type='radix', expansion_basis='solidfmm'."
            )
        if not isinstance(prepared_state, LargeNPreparedState):
            raise NotImplementedError(
                "refresh_prepared_state currently supports LargeNPreparedState only."
            )

        self._compiled_profile_refresh_calls += 1
        refresh_timing_enabled = bool(getattr(self, "_refresh_timing_enabled", False))
        if not refresh_timing_enabled:
            next_state = self._refresh_large_n_same_topology(
                prepared_state,
                positions,
                masses,
                bounds=bounds,
                leaf_size=int(
                    prepared_state.max_leaf_size if leaf_size is None else leaf_size
                ),
                max_order=(
                    int(prepared_state.local_data.order)
                    if max_order is None
                    else int(max_order)
                ),
                theta=theta,
                fused_device_mode=bool(fused_device_mode),
            )
            if next_state is None:
                next_state = self.prepare_state(
                    positions,
                    masses,
                    bounds=bounds,
                    leaf_size=int(
                        prepared_state.max_leaf_size if leaf_size is None else leaf_size
                    ),
                    max_order=(
                        int(prepared_state.local_data.order)
                        if max_order is None
                        else int(max_order)
                    ),
                    theta=theta,
                    fused_device_mode=bool(fused_device_mode),
                )
            return next_state

        refresh_t0 = time.perf_counter()
        input_before = float(getattr(self, "_refresh_timing_input_seconds", 0.0))
        tree_before = float(getattr(self, "_refresh_timing_tree_upward_seconds", 0.0))
        dual_before = float(getattr(self, "_refresh_timing_dual_downward_seconds", 0.0))
        nearfield_before = float(
            getattr(self, "_refresh_timing_nearfield_seconds", 0.0)
        )
        profile_t0 = time.perf_counter()
        prev_profile = self._compiled_profile_from_prepared_state(prepared_state)
        prev_fingerprint = self._compiled_profile_fingerprint(prev_profile)
        profile_seconds = time.perf_counter() - profile_t0

        was_refresh_timing_active = bool(getattr(self, "_refresh_timing_active", False))
        self._refresh_timing_active = True
        try:
            next_state = self._refresh_large_n_same_topology(
                prepared_state,
                positions,
                masses,
                bounds=bounds,
                leaf_size=int(
                    prepared_state.max_leaf_size if leaf_size is None else leaf_size
                ),
                max_order=(
                    int(prepared_state.local_data.order)
                    if max_order is None
                    else int(max_order)
                ),
                theta=theta,
                fused_device_mode=bool(fused_device_mode),
            )
            if next_state is None:
                next_state = self.prepare_state(
                    positions,
                    masses,
                    bounds=bounds,
                    leaf_size=int(
                        prepared_state.max_leaf_size if leaf_size is None else leaf_size
                    ),
                    max_order=(
                        int(prepared_state.local_data.order)
                        if max_order is None
                        else int(max_order)
                    ),
                    theta=theta,
                    fused_device_mode=bool(fused_device_mode),
                )
        finally:
            self._refresh_timing_active = was_refresh_timing_active
        prepare_elapsed = time.perf_counter() - refresh_t0

        profile_t0 = time.perf_counter()
        next_profile = self._compiled_profile_from_prepared_state(next_state)
        next_fingerprint = self._compiled_profile_fingerprint(next_profile)
        self._compiled_profile_record_transition(next_fingerprint)

        if next_fingerprint == prev_fingerprint:
            self._compiled_profile_refresh_reuse_tier_full += 1
        elif self._compiled_profile_capacity_compatible(prev_profile, next_profile):
            self._compiled_profile_refresh_reuse_tier_topology += 1
        else:
            self._compiled_profile_refresh_reuse_tier_overflow += 1
        profile_seconds += time.perf_counter() - profile_t0
        total_elapsed = time.perf_counter() - refresh_t0
        input_delta = (
            float(getattr(self, "_refresh_timing_input_seconds", 0.0)) - input_before
        )
        tree_delta = (
            float(getattr(self, "_refresh_timing_tree_upward_seconds", 0.0))
            - tree_before
        )
        dual_delta = (
            float(getattr(self, "_refresh_timing_dual_downward_seconds", 0.0))
            - dual_before
        )
        nearfield_delta = (
            float(getattr(self, "_refresh_timing_nearfield_seconds", 0.0))
            - nearfield_before
        )
        stage_sum = (
            input_delta
            + tree_delta
            + dual_delta
            + nearfield_delta
            + float(profile_seconds)
        )
        # prepare_large_n_state records cumulative stage timings directly on
        # the solver. Attribute the unaccounted part of this refresh to Python
        # overhead, sync, compilation, or other work outside the explicit
        # large-N stage timers.
        self._refresh_timing_profile_accounting_seconds += float(profile_seconds)
        self._refresh_timing_total_seconds += float(total_elapsed)
        self._refresh_timing_compile_or_sync_suspect_seconds += max(
            0.0,
            float(total_elapsed) - float(stage_sum),
        )
        self._refresh_timing_calls += 1
        return next_state

    def strict_prepare_refresh_and_evaluate(
        self: "FastMultipoleMethod",
        prepared_state: Optional[PreparedStateLike],
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        theta: Optional[float] = None,
        jit_traversal: Optional[bool] = True,
        runtime_overrides: Optional[_RuntimeExecutionOverrides] = None,
        fused_device_mode: Optional[bool] = None,
    ) -> tuple[PreparedStateLike, Array]:
        """Strict static-radix helper: prepare/refresh once, then evaluate."""
        if not self._is_large_n_gpu_production_profile():
            self._strict_runner_fail_fast_reject_count += 1
            raise RuntimeError(
                "strict_prepare_refresh_and_evaluate requires large_n_gpu production profile."
            )

        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        profile_key = (
            f"n={int(positions_arr.shape[0])}|"
            f"leaf={int(leaf_size)}|"
            f"order={int(max_order)}|"
            f"theta={float(self.theta if theta is None else theta):.12g}"
        )
        if profile_key in self._strict_runner_seen_profile_keys:
            self._strict_runner_profile_key_hits += 1
        else:
            self._strict_runner_profile_key_misses += 1
            self._strict_runner_compile_count += 1
            self._strict_runner_seen_profile_keys.add(profile_key)
        self._strict_runner_execute_count += 1

        if prepared_state is None:
            next_state = self.prepare_state(
                positions_arr,
                masses_arr,
                bounds=bounds,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_tree=self._jit_tree_default,
                runtime_overrides_override=runtime_overrides,
                fused_device_mode=bool(
                    self._strict_fused_mode_active
                    if fused_device_mode is None
                    else fused_device_mode
                ),
            )
        else:
            if not isinstance(prepared_state, LargeNPreparedState):
                self._strict_runner_fail_fast_reject_count += 1
                raise RuntimeError(
                    "strict_prepare_refresh_and_evaluate requires LargeNPreparedState input."
                )
            next_state_try = self._refresh_large_n_same_topology(
                prepared_state,
                positions_arr,
                masses_arr,
                bounds=bounds,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                runtime_overrides_override=runtime_overrides,
                fused_device_mode=bool(
                    self._strict_fused_mode_active
                    if fused_device_mode is None
                    else fused_device_mode
                ),
            )
            if next_state_try is None:
                self._strict_runner_fail_fast_reject_count += 1
                raise RuntimeError(
                    "strict_prepare_refresh_and_evaluate fail-fast: "
                    "refresh miss (profile/topology mismatch)."
                )
            next_state = next_state_try

        acc = self.evaluate_prepared_state(
            next_state,
            target_indices=None,
            return_potential=False,
            jit_traversal=(
                self._jit_traversal_default
                if jit_traversal is None
                else bool(jit_traversal)
            ),
            max_acc_derivative_order=0,
        )
        return next_state, jnp.asarray(acc)

    def strict_run_segmented(
        self: "FastMultipoleMethod",
        *,
        state: Any,
        masses: Array,
        num_steps: int,
        refresh_every: int,
        segment_runner: Callable[[Any, Array, int], tuple[Any, Any]],
        positions_getter: Callable[[Any], Array],
        prepared_state: Optional[PreparedStateLike] = None,
        leaf_size: int = 16,
        max_order: int = 2,
        theta: Optional[float] = None,
        jit_traversal: Optional[bool] = True,
        rematerialize_fn: Optional[Callable[[Any], Any]] = None,
        collect_history: bool = False,
    ) -> tuple[Any, PreparedStateLike, Optional[list[Any]]]:
        """Run strict refresh/evaluate cadence with caller-provided segment runner."""
        if int(num_steps) <= 0:
            raise ValueError("num_steps must be positive")
        if int(refresh_every) <= 0:
            raise ValueError("refresh_every must be positive")

        num_steps_i = int(num_steps)
        refresh_every_i = int(refresh_every)
        full_segments = num_steps_i // refresh_every_i
        tail_segment = num_steps_i % refresh_every_i

        state_curr = state
        prepared_curr = prepared_state
        history: Optional[list[Any]] = [] if collect_history else None
        runtime_overrides_cached = self._resolve_runtime_execution_overrides(
            num_particles=int(jnp.asarray(masses).shape[0]),
        )

        for _ in range(full_segments):
            positions_curr = positions_getter(state_curr)
            prepared_curr, acc_self = self.strict_prepare_refresh_and_evaluate(
                prepared_curr,
                positions_curr,
                masses,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_traversal=jit_traversal,
                runtime_overrides=runtime_overrides_cached,
                fused_device_mode=bool(self._strict_fused_mode_active),
            )
            state_curr, seg_hist = segment_runner(
                state_curr,
                jnp.asarray(acc_self),
                int(refresh_every_i),
            )
            if rematerialize_fn is not None:
                state_curr = rematerialize_fn(state_curr)
            if history is not None:
                history.append(seg_hist)

        if tail_segment > 0:
            positions_curr = positions_getter(state_curr)
            prepared_curr, acc_self = self.strict_prepare_refresh_and_evaluate(
                prepared_curr,
                positions_curr,
                masses,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_traversal=jit_traversal,
                runtime_overrides=runtime_overrides_cached,
                fused_device_mode=bool(self._strict_fused_mode_active),
            )
            state_curr, seg_hist = segment_runner(
                state_curr,
                jnp.asarray(acc_self),
                int(tail_segment),
            )
            if rematerialize_fn is not None:
                state_curr = rematerialize_fn(state_curr)
            if history is not None:
                history.append(seg_hist)

        return state_curr, prepared_curr, history

    def strict_run_v2(
        self: "FastMultipoleMethod",
        *,
        state: Array,
        masses: Array,
        dt: float,
        num_steps: int,
        refresh_every: int,
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
        prepared_state: Optional[PreparedStateLike] = None,
        initial_self_acceleration: Optional[Array] = None,
        jit_traversal: Optional[bool] = True,
        add_external: bool = False,
        external_acceleration_fn: Optional[Callable[[Array], Array]] = None,
        rematerialize_between_refresh: bool = True,
        return_history: bool = False,
        return_prepared_state: bool = True,
        step_callback: Optional[Callable[[Array, Array], None]] = None,
        step_callback_stride: int = 1,
    ) -> tuple[Array, Optional[PreparedStateLike], Optional[Array]]:
        """Run endpoint-correct velocity Verlet with strict prepared-state refresh.

        ``step_callback`` is an optional traced, side-effecting hook called inside
        the device-resident scan as ``step_callback(step_index, state)`` every
        ``step_callback_stride`` steps (``step_index`` and ``state`` are traced
        device values). It must be fire-and-forget (return nothing) and should use
        ``jax.debug.callback`` internally to ship only small, on-device-reduced
        data to the host (e.g. a projected density grid), so the GPU is not
        stalled. It does not touch the scan carry and is independent of
        ``return_history``."""
        state_arr = jnp.asarray(state)
        masses_arr = jnp.asarray(masses)
        dt_arr = jnp.asarray(float(dt), dtype=state_arr.dtype)
        num_steps_i = int(num_steps)

        if not self._is_large_n_gpu_production_profile():
            self._strict_v2_fail_fast_reject_count += 1
            raise RuntimeError("strict_run_v2 requires large_n_gpu production profile.")
        if num_steps_i <= 0:
            raise ValueError("num_steps must be positive")
        if int(refresh_every) != 1:
            self._strict_v2_fail_fast_reject_count += 1
            raise ValueError(
                "strict_run_v2 requires refresh_every=1 for endpoint-correct "
                "velocity-Verlet self gravity"
            )

        profile_key = (
            f"n={int(state_arr.shape[0])}|leaf={int(leaf_size)}|"
            f"order={int(max_order)}|refresh=1|"
            f"dt={float(dt):.12g}|external={int(bool(add_external))}|"
            f"theta={float(self.theta if theta is None else theta):.12g}"
        )
        if profile_key in self._strict_v2_seen_profile_keys:
            self._strict_v2_profile_key_hits += 1
        else:
            self._strict_v2_profile_key_misses += 1
            self._strict_v2_compile_count += 1
            self._strict_v2_seen_profile_keys.add(profile_key)
        self._strict_v2_execute_count += 1

        fused_mode_requested = bool(getattr(self, "_strict_fused_mode_enabled", False))
        fused_mode_allowed = self._strict_fused_profile_allows_n(
            int(state_arr.shape[0])
        )
        self._strict_fused_mode_active = bool(
            fused_mode_requested and fused_mode_allowed
        )
        if self._strict_fused_mode_active:
            if profile_key in self._strict_fused_seen_profile_keys:
                self._strict_fused_profile_key_hits += 1
            else:
                self._strict_fused_profile_key_misses += 1
                self._strict_fused_compile_count += 1
                self._strict_fused_seen_profile_keys.add(profile_key)
            self._strict_fused_execute_count += 1
            self._strict_fused_device_refresh_route_count += num_steps_i
            self._strict_fused_planner_bypassed_count += num_steps_i
        elif fused_mode_requested and not fused_mode_allowed:
            self._strict_fused_fallback_count += 1
            self._strict_fused_last_fallback_reason = (
                "particle_count_not_in_JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET"
            )
            # Fused mode was requested but this particle count is not in the
            # configured profile set. Refuse to silently disable the fused fast
            # lane and run a slower non-fused path -- raise so the profile set is
            # fixed (or cleared to allow all N) instead.
            raise RuntimeError(
                "strict fused mode requested but particle count "
                f"N={int(state_arr.shape[0])} is not in "
                "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET="
                f"{os.environ.get('JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET', '')!r}; "
                "refusing to silently fall back to a slower non-fused path. Add "
                "this N to the profile set, or leave it empty to allow all N."
            )
        else:
            self._strict_fused_last_fallback_reason = ""

        self._strict_velocity_verlet_acceleration_carry_active = True
        diag_mode = str(getattr(self, "_strict_refresh_diag_mode", "full"))
        eval_diag_mode = str(getattr(self, "_large_n_eval_diag_mode", "full"))
        detail_diag_mode = str(
            getattr(self, "_strict_refresh_detail_diag_mode", "full")
        )
        self_eval_active = (
            bool(getattr(self, "_strict_refresh_diag_eval_active", True))
            and detail_diag_mode == "full"
            and eval_diag_mode != "zero"
        )
        self._strict_self_force_bootstrap_evaluations = int(self_eval_active)
        self._strict_self_force_endpoint_evaluations = (
            num_steps_i if self_eval_active else 0
        )
        self._strict_external_bootstrap_evaluations = int(
            bool(add_external) and external_acceleration_fn is not None
        )
        self._strict_external_endpoint_evaluations = (
            num_steps_i
            if bool(add_external) and external_acceleration_fn is not None
            else 0
        )

        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(state_arr.shape[0])
        )
        prepared_curr = prepared_state
        if prepared_curr is None:
            prepared_curr = self.prepare_state(
                state_arr[:, 0, :],
                masses_arr,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_tree=self._jit_tree_default,
                runtime_overrides_override=runtime_overrides,
                fused_device_mode=bool(self._strict_fused_mode_active),
            )
        if self._strict_fused_mode_active and not isinstance(
            prepared_curr, LargeNPreparedState
        ):
            self._strict_runner_fail_fast_reject_count += 1
            raise RuntimeError(
                "strict fused velocity-Verlet requires LargeNPreparedState input."
            )
        if isinstance(prepared_curr, LargeNPreparedState):
            self._record_large_n_eval_shape_diagnostics(prepared_curr)

        def _evaluate_self(prepared_in: PreparedStateLike, state_in: Array) -> Array:
            if not self_eval_active:
                return jnp.zeros_like(state_in[:, 0, :])
            if eval_diag_mode == "permutation_only":
                return jnp.asarray(prepared_in.positions_sorted)[
                    prepared_in.inverse_permutation
                ] * jnp.asarray(0.0, dtype=state_in.dtype)
            return jnp.asarray(
                evaluate_large_n_state(
                    self,
                    prepared_in,
                    target_indices=None,
                    return_potential=False,
                    max_acc_derivative_order=0,
                ),
                dtype=state_in.dtype,
            )

        if not self_eval_active:
            acceleration_self_current = jnp.zeros_like(state_arr[:, 0, :])
        elif initial_self_acceleration is None:
            acceleration_self_current = _evaluate_self(prepared_curr, state_arr)
        else:
            acceleration_self_current = jnp.asarray(
                initial_self_acceleration, dtype=state_arr.dtype
            )
        if add_external and external_acceleration_fn is not None:
            acceleration_current = acceleration_self_current + jnp.asarray(
                external_acceleration_fn(state_arr), dtype=state_arr.dtype
            )
        else:
            acceleration_current = acceleration_self_current

        def _static_target_block_capacity_ok(
            prepared_in: PreparedStateLike,
        ) -> Array:
            padded = getattr(
                prepared_in,
                "nearfield_target_block_source_leaf_ids_padded",
                None,
            )
            if padded is None:
                return jnp.asarray(True)
            padded_arr = jnp.asarray(padded)
            if padded_arr.ndim != 3 or int(padded_arr.shape[1]) == 0:
                return jnp.asarray(True)
            offsets = jnp.asarray(prepared_in.neighbor_list.offsets)
            counts = offsets[1:] - offsets[:-1]
            capacity = int(padded_arr.shape[1]) * int(padded_arr.shape[2])
            return jnp.all(counts <= jnp.asarray(capacity, dtype=counts.dtype))

        def _refresh_and_evaluate_endpoint(
            prepared_in: PreparedStateLike,
            state_position: Array,
        ) -> tuple[PreparedStateLike, Array]:
            if diag_mode in {"integrator_only", "eval_only"}:
                prepared_new = prepared_in
            else:
                prepared_new = self._refresh_large_n_same_topology(
                    prepared_in,
                    state_position[:, 0, :],
                    masses_arr,
                    bounds=None,
                    leaf_size=int(leaf_size),
                    max_order=int(max_order),
                    theta=theta,
                    runtime_overrides_override=None,
                    fused_device_mode=bool(self._strict_fused_mode_active),
                )
                if prepared_new is None:
                    raise RuntimeError(
                        "strict velocity-Verlet refresh failed: topology/profile mismatch"
                    )
            return prepared_new, _evaluate_self(prepared_new, state_position)

        if self._strict_fused_mode_active:
            # Stash the concrete tree depth now, while prepared_curr is concrete,
            # so the traced refresh inside the compiled runner passes it as the
            # M2M level-loop static arg. Keyed into cache_key so a topology with a
            # different depth compiles its own runner.
            static_upward_num_levels = self._resolve_upward_num_levels(
                getattr(prepared_curr, "tree", None)
            )
            cache_key = (
                "strict_velocity_verlet",
                tuple(int(v) for v in state_arr.shape),
                str(state_arr.dtype),
                tuple(int(v) for v in masses_arr.shape),
                str(masses_arr.dtype),
                float(dt),
                num_steps_i,
                int(leaf_size),
                int(max_order),
                float(self.theta if theta is None else theta),
                bool(add_external),
                (
                    id(external_acceleration_fn)
                    if external_acceleration_fn is not None
                    else 0
                ),
                bool(rematerialize_between_refresh),
                bool(return_history),
                diag_mode,
                detail_diag_mode,
                eval_diag_mode,
                str(getattr(self, "_large_n_nearfield_diag_mode", "full")),
                static_upward_num_levels,
                id(step_callback) if step_callback is not None else 0,
                int(step_callback_stride),
            )
            jit_cache = getattr(self, "_strict_fused_jit_function_cache", {})
            compiled_runner = jit_cache.get(cache_key)
            if compiled_runner is None:

                @jax.jit
                def _compiled_runner(
                    prepared_initial: LargeNPreparedState,
                    state_initial: Array,
                    acceleration_initial: Array,
                ) -> tuple[
                    tuple[LargeNPreparedState, Array, Array, Array], Optional[Array]
                ]:
                    def _step(carry, scan_x):
                        (
                            prepared_now,
                            state_now,
                            acceleration_now,
                            capacity_ok_now,
                        ) = carry
                        position_new = (
                            state_now[:, 0]
                            + state_now[:, 1] * dt_arr
                            + 0.5 * acceleration_now * dt_arr**2
                        )
                        state_position = state_now.at[:, 0].set(position_new)
                        prepared_new, acceleration_self_new = (
                            _refresh_and_evaluate_endpoint(prepared_now, state_position)
                        )
                        if add_external and external_acceleration_fn is not None:
                            acceleration_new = acceleration_self_new + jnp.asarray(
                                external_acceleration_fn(state_position),
                                dtype=state_now.dtype,
                            )
                        else:
                            acceleration_new = acceleration_self_new
                        state_new = _velocity_verlet_state_update(
                            state_now,
                            acceleration_now,
                            acceleration_new,
                            dt_arr,
                        )
                        if rematerialize_between_refresh:
                            state_new = jnp.asarray(state_new, dtype=state_now.dtype)
                        if step_callback is not None:
                            # Fire-and-forget streaming hook (e.g. render). Gated by
                            # stride via lax.cond so it only fires + only computes its
                            # on-device reduction on emit steps. Returns a dummy int so
                            # both cond branches match; the result is discarded and the
                            # scan carry is untouched.
                            def _emit(_):
                                step_callback(scan_x, state_new)
                                return jnp.int32(0)

                            def _skip(_):
                                return jnp.int32(0)

                            jax.lax.cond(
                                (scan_x % jnp.int32(step_callback_stride))
                                == jnp.int32(0),
                                _emit,
                                _skip,
                                operand=None,
                            )
                        capacity_ok_new = capacity_ok_now & (
                            _static_target_block_capacity_ok(prepared_new)
                        )
                        return (
                            prepared_new,
                            state_new,
                            acceleration_new,
                            capacity_ok_new,
                        ), (state_new if return_history else None)

                    # Feed a per-step index only when a streaming callback needs it
                    # (keeps the no-callback path byte-for-byte unchanged).
                    scan_xs = (
                        jnp.arange(num_steps_i, dtype=jnp.int32)
                        if step_callback is not None
                        else None
                    )
                    return jax.lax.scan(
                        _step,
                        (
                            prepared_initial,
                            state_initial,
                            acceleration_initial,
                            _static_target_block_capacity_ok(prepared_initial),
                        ),
                        xs=scan_xs,
                        length=num_steps_i,
                    )

                compiled_runner = _compiled_runner
                jit_cache[cache_key] = compiled_runner
                self._strict_fused_jit_function_cache = jit_cache

            try:
                (
                    prepared_curr,
                    state_curr,
                    _,
                    capacity_ok_all,
                ), history_out = compiled_runner(
                    prepared_curr,
                    state_arr,
                    jnp.asarray(acceleration_current, dtype=state_arr.dtype),
                )
                self._strict_static_target_block_capacity_ok = bool(
                    np.asarray(jax.device_get(capacity_ok_all))
                )
                if not self._strict_static_target_block_capacity_ok:
                    max_blocks = os.environ.get(
                        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
                        "32",
                    )
                    raise RuntimeError(
                        "fused payload static target-block cap exceeded during "
                        "compiled velocity-Verlet scan: max_blocks_per_leaf="
                        f"{max_blocks}. Increase "
                        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF."
                    )
            except Exception as exc:
                if bool(
                    getattr(self, "_strict_fused_disallow_host_segment_fallback", False)
                ):
                    raise RuntimeError(
                        "strict fused velocity-Verlet scan failed while host fallback "
                        "is disallowed"
                    ) from exc
                raise
        else:
            state_curr = state_arr
            history_parts: list[Array] = []
            acceleration_now = jnp.asarray(acceleration_current, dtype=state_arr.dtype)
            for _ in range(num_steps_i):
                position_new = (
                    state_curr[:, 0]
                    + state_curr[:, 1] * dt_arr
                    + 0.5 * acceleration_now * dt_arr**2
                )
                state_position = state_curr.at[:, 0].set(position_new)
                prepared_curr, acceleration_self_new = _refresh_and_evaluate_endpoint(
                    prepared_curr, state_position
                )
                if add_external and external_acceleration_fn is not None:
                    acceleration_new = acceleration_self_new + jnp.asarray(
                        external_acceleration_fn(state_position),
                        dtype=state_curr.dtype,
                    )
                else:
                    acceleration_new = acceleration_self_new
                state_curr = _velocity_verlet_state_update(
                    state_curr, acceleration_now, acceleration_new, dt_arr
                )
                acceleration_now = acceleration_new
                if return_history:
                    history_parts.append(state_curr)
            history_out = jnp.stack(history_parts, axis=0) if return_history else None

        self._strict_runner_execute_count += num_steps_i
        if profile_key in self._strict_runner_seen_profile_keys:
            self._strict_runner_profile_key_hits += num_steps_i
        else:
            self._strict_runner_seen_profile_keys.add(profile_key)
            self._strict_runner_compile_count += 1
            self._strict_runner_profile_key_misses += 1
            self._strict_runner_profile_key_hits += max(0, num_steps_i - 1)
        prepared_out = prepared_curr if return_prepared_state else None
        return state_curr, prepared_out, history_out

    def strict_fused_prepared_eval_fn(
        self: "FastMultipoleMethod",
        *,
        positions: Array,
        masses: Array,
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
    ) -> tuple[PreparedStateLike, Callable[[PreparedStateLike], Array]]:
        """Build a fused-lane prepared state and return a jitted eval-only closure.

        Isolates the *evaluate* cost of the strict fused static-radix lane for
        apples-to-apples benchmarking against functional FMM eval APIs (e.g.
        jaxfmm ``eval_potential``): the prepared state is built eagerly with the
        fused device-mode layout (optimized flat compact far-pairs + static
        target-block near-field), exactly as ``strict_run_v2`` bootstraps it, and
        the returned closure runs the same self-force evaluation the fused step
        runs per endpoint (``evaluate_large_n_state``) with **no refresh and no
        velocity-Verlet update**.

        Returns ``(prepared_state, eval_fn)``; time ``eval_fn(prepared_state)``.
        """
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        if not self._is_large_n_gpu_production_profile():
            raise RuntimeError(
                "strict_fused_prepared_eval_fn requires large_n_gpu production profile."
            )
        fused_mode_requested = bool(getattr(self, "_strict_fused_mode_enabled", False))
        fused_mode_allowed = self._strict_fused_profile_allows_n(
            int(positions_arr.shape[0])
        )
        self._strict_fused_mode_active = bool(
            fused_mode_requested and fused_mode_allowed
        )
        if not self._strict_fused_mode_active:
            raise RuntimeError(
                "strict fused mode is not active for this particle count/config; "
                "enable JACCPOT_STATIC_STRICT_FUSED_MODE and include N in "
                "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET."
            )
        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(positions_arr.shape[0])
        )
        prepared = self.prepare_state(
            positions_arr,
            masses_arr,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            theta=theta,
            jit_tree=self._jit_tree_default,
            runtime_overrides_override=runtime_overrides,
            fused_device_mode=True,
        )
        if not isinstance(prepared, LargeNPreparedState):
            raise RuntimeError("strict fused eval-only requires a LargeNPreparedState.")
        self._record_large_n_eval_shape_diagnostics(prepared)

        @jax.jit
        def _eval(prepared_in: PreparedStateLike) -> Array:
            return jnp.asarray(
                evaluate_large_n_state(
                    self,
                    prepared_in,
                    target_indices=None,
                    return_potential=False,
                    max_acc_derivative_order=0,
                )
            )

        return prepared, _eval

    def _refresh_large_n_same_topology(
        self: "FastMultipoleMethod",
        prepared_state: LargeNPreparedState,
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]],
        leaf_size: int,
        max_order: int,
        theta: Optional[float],
        runtime_overrides_override: Optional[_RuntimeExecutionOverrides] = None,
        fused_device_mode: bool = False,
    ) -> Optional[LargeNPreparedState]:
        """Refresh large-N numeric payloads when the radix topology is unchanged."""

        self._large_n_same_topology_refresh_attempts += 1
        if not isinstance(prepared_state.tree, RadixTree):
            self._large_n_same_topology_refresh_misses += 1
            self._large_n_same_topology_refresh_miss_no_key += 1
            return None

        refresh_timing_active = bool(
            getattr(self, "_refresh_timing_active", False)
        ) and not (
            bool(fused_device_mode)
            and bool(getattr(self, "_strict_fused_disable_hot_timing", False))
        )

        input_t0 = time.perf_counter() if refresh_timing_active else 0.0
        positions_arr, masses_arr, input_dtype = self._prepare_state_input_arrays(
            positions,
            masses,
        )
        if refresh_timing_active:
            self._refresh_timing_input_seconds += time.perf_counter() - input_t0
        traced_refresh = bool(_contains_tracer((positions_arr, masses_arr)))
        allow_stateful_cache = bool(fused_device_mode) or (not traced_refresh)
        if (not allow_stateful_cache) and (not bool(fused_device_mode)):
            self._large_n_same_topology_refresh_misses += 1
            self._large_n_same_topology_refresh_miss_traced += 1
            return None

        self._validate_prepare_state_request(
            leaf_size=int(leaf_size),
            max_order=int(max_order),
        )
        runtime_overrides = runtime_overrides_override
        if runtime_overrides is None:
            runtime_overrides = self._resolve_runtime_execution_overrides(
                num_particles=int(positions_arr.shape[0]),
            )
        runtime_traversal_config = runtime_overrides.traversal_config
        runtime_m2l_chunk_size = runtime_overrides.m2l_chunk_size
        runtime_l2l_chunk_size = runtime_overrides.l2l_chunk_size
        upward_center_mode = runtime_overrides.center_mode
        refine_local_val = self.refine_local
        if runtime_overrides.refine_local_override is not None:
            refine_local_val = bool(runtime_overrides.refine_local_override)
        max_refine_levels_val = self.max_refine_levels
        aspect_threshold_val = self.aspect_threshold
        theta_val = float(self.theta if theta is None else theta)
        mac_type_val = self._base_mac_type()

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
        static_fused_refresh = bool(fused_device_mode) and (
            str(tree_config.mode).strip().lower() == "static_radix"
        )
        inferred_bounds = self._resolve_prepare_state_bounds(
            positions=positions_arr,
            bounds=bounds,
        )

        tree_t0 = time.perf_counter() if refresh_timing_active else 0.0
        refresh_topology_key = getattr(prepared_state, "topology_key", None)
        topology_candidate = None

        if static_fused_refresh:
            build_artifacts = self._rebuild_tree_artifacts_from_static_template(
                template_tree=prepared_state.tree,
                positions=positions_arr,
                masses=masses_arr,
                bounds=inferred_bounds,
                max_leaf_size=int(prepared_state.max_leaf_size),
                cache_leaf_parameter=int(leaf_size),
            )
            if refresh_topology_key is None:
                refresh_topology_key = "static_fused_template"
        else:
            previous_topology_key = refresh_topology_key
            if previous_topology_key is None:
                if tree_config.mode == "static_radix" and isinstance(
                    prepared_state.tree, RadixTree
                ):
                    previous_topology_key = self._static_radix_topology_key_from_tree(
                        prepared_state.tree,
                        leaf_size=int(leaf_size),
                    )
                else:
                    previous_codes = getattr(prepared_state.tree, "morton_codes", None)
                    if previous_codes is not None:
                        previous_topology_key = (
                            self._topology_reuse_key_from_sorted_codes(
                                sorted_codes=jnp.asarray(previous_codes),
                                tree_config=tree_config,
                                leaf_size=int(leaf_size),
                                refine_local=refine_local_val,
                                max_refine_levels=max_refine_levels_val,
                                aspect_threshold=aspect_threshold_val,
                            )
                        )
            if previous_topology_key is None:
                self._large_n_same_topology_refresh_misses += 1
                self._large_n_same_topology_refresh_miss_no_key += 1
                if tree_config.mode == "static_radix":
                    self._static_radix_refresh_misses += 1
                return None

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
            if (
                topology_candidate is None
                or topology_candidate.key != previous_topology_key
            ):
                self._large_n_same_topology_refresh_misses += 1
                self._large_n_same_topology_refresh_miss_topology += 1
                if tree_config.mode == "static_radix":
                    self._static_radix_refresh_misses += 1
                return None

            topology_entry = _TopologyReuseEntry(
                key=str(previous_topology_key),
                tree=prepared_state.tree,
                max_leaf_size=int(prepared_state.max_leaf_size),
                cache_leaf_parameter=int(leaf_size),
                reuse_count=0,
            )
            build_artifacts = self._rebuild_tree_artifacts_from_topology(
                candidate=topology_candidate,
                entry=topology_entry,
                positions=positions_arr,
                masses=masses_arr,
            )
            refresh_topology_key = topology_candidate.key

        strict_refresh_diag_mode = str(
            getattr(self, "_strict_refresh_diag_mode", "full")
        )
        strict_refresh_detail_diag_mode = str(
            getattr(self, "_strict_refresh_detail_diag_mode", "full")
        )
        strict_refresh_tree_detail_only = strict_refresh_detail_diag_mode in {
            "tree_sort_only",
            "tree_metadata_only",
        }
        strict_refresh_upward_detail_only = strict_refresh_detail_diag_mode in {
            "p2m_only",
            "m2m_only",
        }
        if bool(static_fused_refresh) and (
            strict_refresh_diag_mode == "tree_only" or strict_refresh_tree_detail_only
        ):
            self._large_n_same_topology_refresh_hits += 1
            if tree_config.mode == "static_radix":
                self._static_radix_refresh_hits += 1
            return replace(
                prepared_state,
                tree=build_artifacts.tree,
                topology_key=refresh_topology_key,
            )

        defer_geometry = False
        upward = self.prepare_upward_sweep(
            build_artifacts.tree,
            build_artifacts.positions_sorted,
            build_artifacts.masses_sorted,
            max_order=int(max_order),
            center_mode=upward_center_mode,
            max_leaf_size=int(build_artifacts.max_leaf_size),
            defer_geometry=defer_geometry,
        )
        locals_template = self._build_locals_template_for_prepare_state(
            tree=build_artifacts.tree,
            upward=upward,
            max_order=int(max_order),
            pos_sorted=build_artifacts.positions_sorted,
        )
        tree_artifacts = _PrepareStateTreeUpwardArtifacts(
            tree_mode=tree_config.mode,
            tree=build_artifacts.tree,
            positions_sorted=build_artifacts.positions_sorted,
            masses_sorted=build_artifacts.masses_sorted,
            inverse_permutation=build_artifacts.inverse_permutation,
            leaf_cap=int(build_artifacts.max_leaf_size),
            leaf_parameter=int(build_artifacts.cache_leaf_parameter),
            topology_key=refresh_topology_key,
            upward=upward,
            locals_template=locals_template,
        )
        if refresh_timing_active:
            self._refresh_timing_tree_upward_seconds += time.perf_counter() - tree_t0

        if bool(static_fused_refresh) and (
            strict_refresh_diag_mode == "upward_only"
            or strict_refresh_upward_detail_only
        ):
            dep = jnp.asarray(0.0, dtype=tree_artifacts.positions_sorted.dtype)
            multipoles = getattr(tree_artifacts.upward, "multipoles", None)
            packed = getattr(multipoles, "packed", None)
            centers = getattr(multipoles, "centers", None)
            if packed is not None:
                dep = dep + jnp.asarray(
                    jnp.real(jnp.sum(jnp.asarray(packed))),
                    dtype=dep.dtype,
                ) * jnp.asarray(0.0, dtype=dep.dtype)
            if centers is not None:
                dep = dep + jnp.asarray(
                    jnp.sum(jnp.asarray(centers)),
                    dtype=dep.dtype,
                ) * jnp.asarray(0.0, dtype=dep.dtype)
            diag_tree = replace(
                tree_artifacts.tree,
                positions_sorted=tree_artifacts.positions_sorted + dep,
            )
            self._large_n_same_topology_refresh_hits += 1
            if tree_config.mode == "static_radix":
                self._static_radix_refresh_hits += 1
            return replace(
                prepared_state,
                tree=diag_tree,
                topology_key=refresh_topology_key,
            )

        collected_retries: list[DualTreeRetryEvent] = []

        def record_retry(event: DualTreeRetryEvent) -> None:
            collected_retries.append(event)
            if self.interaction_retry_logger is not None:
                self.interaction_retry_logger(event)

        dual_t0 = time.perf_counter() if refresh_timing_active else 0.0
        strict_fused_traced_hot_path = bool(fused_device_mode) and bool(
            getattr(self, "_strict_fused_mode_active", False)
        )
        cached_compact_far_pairs = getattr(prepared_state, "compact_far_pairs", None)
        compact_far_pairs_carry_placeholder = cached_compact_far_pairs
        reuse_static_compact_pairs_enabled = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_REUSE_COMPACT_PAIRS",
                "1",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        allow_unsafe_compact_pair_reuse = str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE",
                "0",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        safe_fresh_compact_pair_rebuild = (
            bool(strict_fused_traced_hot_path)
            and str(tree_config.mode).strip().lower() == "static_radix"
            and str(
                os.environ.get(
                    "JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD",
                    "1",
                )
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            and not bool(allow_unsafe_compact_pair_reuse)
        )
        reuse_static_compact_pairs = (
            bool(strict_fused_traced_hot_path)
            and bool(cached_compact_far_pairs is not None)
            and str(tree_config.mode).strip().lower() == "static_radix"
            and bool(reuse_static_compact_pairs_enabled)
            and bool(allow_unsafe_compact_pair_reuse)
        )
        if (
            bool(strict_fused_traced_hot_path)
            and bool(cached_compact_far_pairs is not None)
            and str(tree_config.mode).strip().lower() == "static_radix"
            and bool(reuse_static_compact_pairs_enabled)
            and not bool(allow_unsafe_compact_pair_reuse)
            and not bool(safe_fresh_compact_pair_rebuild)
        ):
            raise RuntimeError(
                "strict fused compact far-pair reuse is unsafe for moved "
                "static-radix positions: cached M2L pairs can change after "
                "the drift and corrupt endpoint forces. A production fix needs "
                "fresh fixed-cap compact pairs with an active mask/count, or a "
                "proven far-pair validity key. Set "
                "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE=1 "
                "only for legacy performance experiments."
            )
        if bool(safe_fresh_compact_pair_rebuild):
            cached_compact_far_pairs = None
        if bool(strict_fused_traced_hot_path) and (
            str(tree_config.mode).strip().lower() == "static_radix"
        ):
            if reuse_static_compact_pairs:
                self._static_radix_compact_pair_reuse_hits += 1
            else:
                self._static_radix_compact_pair_reuse_misses += 1
        if reuse_static_compact_pairs:
            src_far = jnp.asarray(cached_compact_far_pairs.sources, dtype=INDEX_DTYPE)
            tgt_far = jnp.asarray(cached_compact_far_pairs.targets, dtype=INDEX_DTYPE)
            far_pairs_by_gear = ((src_far, tgt_far),)
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
                farfield_mode="pair_grouped",
                far_pairs_coo=_FarPairCOO(
                    sources=src_far,
                    targets=tgt_far,
                    active_count=getattr(
                        cached_compact_far_pairs, "far_pair_count", None
                    ),
                ),
                far_pairs_by_gear=far_pairs_by_gear,
                adaptive_order=True,
                p_gears=(int(tree_artifacts.upward.multipoles.order),),
            )
            downward = downward._replace(
                interactions=_empty_interaction_storage_for_tree(tree_artifacts.tree)
            )
            dual_downward_artifacts = _PrepareStateDualDownwardArtifacts(
                interactions=None,
                neighbor_list=prepared_state.neighbor_list,
                traversal_result=None,
                compact_far_pairs=cached_compact_far_pairs,
                downward=downward,
                cache_entry=None,
            )
        else:
            dual_downward_artifacts = self._prepare_state_dual_and_downward(
                tree_artifacts=tree_artifacts,
                force_scale_nodes=prepared_state.force_scale_nodes,
                upward_center_mode=upward_center_mode,
                theta_val=theta_val,
                mac_type_val=mac_type_val,
                dehnen_radius_scale=self.dehnen_radius_scale,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                grouped_interactions=False,
                farfield_mode="pair_grouped",
                record_retry=record_retry,
                refine_local_val=refine_local_val,
                max_refine_levels_val=max_refine_levels_val,
                aspect_threshold_val=aspect_threshold_val,
                allow_stateful_cache=True,
                suppress_host_side_effects=strict_fused_traced_hot_path,
            )
        if str(tree_config.mode).strip().lower() == "static_radix":
            tree_now = build_artifacts.tree
            leaf_codes = getattr(tree_now, "leaf_codes", None)
            parent = getattr(tree_now, "parent", None)
            left_child = getattr(tree_now, "left_child", None)
            compact_pairs = getattr(dual_downward_artifacts, "compact_far_pairs", None)
            compact_sources = (
                getattr(compact_pairs, "sources", None)
                if compact_pairs is not None
                else None
            )
            far_pair_count = (
                int(getattr(compact_sources, "shape", (0,))[0])
                if compact_sources is not None
                else int(getattr(self, "_recent_dual_far_pair_count", 0))
            )
            chunk_size = (
                4096 if runtime_m2l_chunk_size is None else int(runtime_m2l_chunk_size)
            )
            self._static_radix_tree_leaf_count = (
                int(getattr(leaf_codes, "shape", (0,))[0])
                if leaf_codes is not None
                else int(getattr(self, "_recent_dual_leaf_count", 0))
            )
            self._static_radix_tree_node_count = (
                int(getattr(parent, "shape", (0,))[0])
                if parent is not None
                else int(getattr(self, "_recent_dual_node_count", 0))
            )
            self._static_radix_far_pair_count = int(far_pair_count)
            self._static_radix_m2l_chunk_count = (
                0
                if chunk_size <= 0 or far_pair_count <= 0
                else int((far_pair_count + chunk_size - 1) // chunk_size)
            )
            self._static_radix_l2l_edge_count = (
                2 * int(getattr(left_child, "shape", (0,))[0])
                if left_child is not None
                else 0
            )

        if refresh_timing_active:
            elapsed = time.perf_counter() - dual_t0
            recorded = float(
                getattr(self, "_refresh_timing_dual_downward_seconds", 0.0)
            )
            # _prepare_state_dual_and_downward records detailed timing itself.
            # Keep this branch intentionally empty except to make the elapsed
            # value visible while avoiding double accounting.
            _ = (elapsed, recorded)

        if (
            tree_config.mode != "static_radix"
            and not self._large_n_neighbor_list_matches(
                prepared_state.neighbor_list,
                dual_downward_artifacts.neighbor_list,
            )
        ):
            self._large_n_same_topology_refresh_misses += 1
            self._large_n_same_topology_refresh_miss_neighbor += 1
            return None

        self._large_n_same_topology_refresh_hits += 1
        if tree_config.mode == "static_radix":
            self._static_radix_refresh_hits += 1

        if allow_stateful_cache and (not traced_refresh):
            self._update_locals_template_cache_after_prepare(
                locals_template=tree_artifacts.locals_template,
                upward=tree_artifacts.upward,
                max_order=int(max_order),
            )
            self._recent_retry_events = tuple(collected_retries)
            self._record_strict_cap_profile_from_retries(
                self._recent_retry_events,
                context_key=self._strict_cap_profile_context_key(
                    tree_mode=str(tree_artifacts.tree_mode),
                    leaf_parameter=int(tree_artifacts.leaf_parameter),
                    particle_count=int(
                        jnp.asarray(tree_artifacts.positions_sorted).shape[0]
                    ),
                ),
            )
            self._topology_reuse_entry = _TopologyReuseEntry(
                key=str(refresh_topology_key),
                tree=tree_artifacts.tree,
                max_leaf_size=int(tree_artifacts.leaf_cap),
                cache_leaf_parameter=int(tree_artifacts.leaf_parameter),
                reuse_count=0,
            )

        refreshed_state = prepare_large_n_state(
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
            jit_tree_override=None,
            allow_stateful_cache=allow_stateful_cache,
            runtime_traversal_config=runtime_traversal_config,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
            upward_center_mode=upward_center_mode,
            record_retry=record_retry,
            collected_retries=collected_retries,
            tree_artifacts=tree_artifacts,
            dual_downward_artifacts=dual_downward_artifacts,
            fused_device_mode=bool(fused_device_mode),
        )
        if bool(safe_fresh_compact_pair_rebuild):
            return replace(
                refreshed_state,
                compact_far_pairs=compact_far_pairs_carry_placeholder,
            )
        return refreshed_state

    def _large_n_neighbor_list_matches(
        self: "FastMultipoleMethod",
        previous: NodeNeighborList,
        current: NodeNeighborList,
    ) -> bool:
        """Return True when current active neighbor edges match previous state."""

        try:
            prev_offsets = np.asarray(jax.device_get(previous.offsets))
            cur_offsets = np.asarray(jax.device_get(current.offsets))
            prev_counts = np.asarray(jax.device_get(previous.counts))
            cur_counts = np.asarray(jax.device_get(current.counts))
            prev_leaf = np.asarray(jax.device_get(previous.leaf_indices))
            cur_leaf = np.asarray(jax.device_get(current.leaf_indices))
            if (
                prev_offsets.shape != cur_offsets.shape
                or prev_counts.shape != cur_counts.shape
                or prev_leaf.shape != cur_leaf.shape
            ):
                return False
            if (
                not np.array_equal(prev_offsets, cur_offsets)
                or not np.array_equal(prev_counts, cur_counts)
                or not np.array_equal(prev_leaf, cur_leaf)
            ):
                return False
            active_edges = int(cur_offsets[-1]) if cur_offsets.size > 0 else 0
            prev_neighbors = np.asarray(jax.device_get(previous.neighbors))
            cur_neighbors = np.asarray(jax.device_get(current.neighbors))
            if int(prev_neighbors.shape[0]) < active_edges:
                return False
            if int(cur_neighbors.shape[0]) < active_edges:
                return False
            return bool(
                np.array_equal(
                    prev_neighbors[:active_edges],
                    cur_neighbors[:active_edges],
                )
            )
        except Exception:
            return False

    def update_multipoles_only(
        self: "FastMultipoleMethod",
        prepared_state: PreparedStateLike,
        positions: Array,
        masses: Array,
        *,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
    ) -> PreparedStateLike:
        """Refresh multipole/local payloads when topology key remains unchanged."""
        if not self._is_large_n_gpu_production_profile():
            raise NotImplementedError(
                "update_multipoles_only is currently supported only for "
                "preset='large_n_gpu', tree_type='radix', expansion_basis='solidfmm'."
            )
        if not isinstance(prepared_state, LargeNPreparedState):
            raise NotImplementedError(
                "update_multipoles_only currently supports LargeNPreparedState only."
            )
        self._compiled_profile_multipoles_only_calls += 1
        refreshed = self.refresh_prepared_state(
            prepared_state,
            positions,
            masses,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
        )
        if getattr(refreshed, "topology_key", None) != getattr(
            prepared_state, "topology_key", None
        ):
            raise RuntimeError(
                "Topology changed during update_multipoles_only; "
                "use rebuild_topology_in_place for topology updates."
            )
        return refreshed

    def rebuild_topology_in_place(
        self: "FastMultipoleMethod",
        prepared_state: PreparedStateLike,
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
    ) -> PreparedStateLike:
        """Rebuild topology while attempting to remain profile-capacity compatible."""
        if not self._is_large_n_gpu_production_profile():
            raise NotImplementedError(
                "rebuild_topology_in_place is currently supported only for "
                "preset='large_n_gpu', tree_type='radix', expansion_basis='solidfmm'."
            )
        if not isinstance(prepared_state, LargeNPreparedState):
            raise NotImplementedError(
                "rebuild_topology_in_place currently supports LargeNPreparedState only."
            )
        self._compiled_profile_topology_rebuild_calls += 1
        return self.refresh_prepared_state(
            prepared_state,
            positions,
            masses,
            bounds=bounds,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
        )
