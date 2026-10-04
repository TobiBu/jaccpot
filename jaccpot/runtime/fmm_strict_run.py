"""StrictRunMixin: fmm_strict_run methods extracted from the FMMEngine
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Callable, Tuple
from jaxtyping import Array
from yggdrax.interactions import DualTreeRetryEvent, NodeNeighborList
from yggdrax.tree import RadixTree

from jaccpot._env import env_choice, env_flag

from ._large_n_pipeline import evaluate_large_n_state, prepare_large_n_state
from ._large_n_types import LargeNPreparedState, LargeNPrepareRequest
from .dtypes import INDEX_DTYPE
from .fmm_caches import _contains_tracer
from .fmm_state import (
    TreeBuilderConfig,
    _PrepareStateDualDownwardArtifacts,
    _PrepareStateTreeUpwardArtifacts,
    _RuntimeExecutionOverrides,
    _TopologyReuseEntry,
    _velocity_verlet_kick_drifted,
    _velocity_verlet_state_update,
)
from .kernels._downward_prep import _far_pair_coo_from
from .kernels.core import _empty_interaction_storage_for_tree

if TYPE_CHECKING:  # pragma: no cover - annotations only, no runtime import
    # The engine lives in `_fmm_impl`, which imports *these mixins* -- so this import
    # must stay under TYPE_CHECKING or it would form the cycle ARCHITECTURE §8
    # forbids. Inheriting `_EngineBase` makes each mixin *be* the engine under a type
    # checker, so every `self.<engine attribute>` resolves; at runtime the alias is
    # `object`, leaving the MRO exactly as it was. The audit's E.2 records why this
    # beats annotating `self`, and what it does not buy at runtime.
    from ._fmm_impl import FMMEngine, PreparedStateLike

    _EngineBase = FMMEngine
else:  # pragma: no cover - annotations only, never an import at runtime
    _EngineBase = object

__all__ = [
    "StrictRunMixin",
]


def _walk_caps_key(validated: Optional[dict]) -> tuple:
    """The part of the validated walk caps a traced refresh is sized from.

    The traced flat walk builds its lists at the validated widths and sizes its
    queue from the validated peak wavefront (``pow2(1.5 x peak)``, see
    ``_strict_fused_capacity_handoff``), both read when the runner TRACES. A
    runner cached under a key without them would keep the old sizes after a
    re-plan. The queue enters as its pow2 target, so a re-prepare whose peak moved a
    little does not recompile.

    Parameters
    ----------
    validated : Optional[dict]
        ``engine._strict_fused_validated_caps``.

    Returns
    -------
    tuple
        Hashable key fragment.
    """
    if not isinstance(validated, dict):
        return ()
    peak = validated.get("peak_wavefront")
    queue_target = (
        None if peak is None else 1 << (max(1, int(1.5 * int(peak))) - 1).bit_length()
    )
    return (
        validated.get("compact_far_pair_capacity"),
        validated.get("near_edge_capacity"),
        queue_target,
    )


def _fresh_compact_pair_rebuild_enabled() -> bool:
    """Whether the fused refresh rebuilds its far-pair list fresh on every step.

    ``JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD`` (default on) and not the
    legacy ``JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE``. Then the
    far list a state carries into the refresh is never read: the refresh builds its
    own, uses it in the same step and returns the input's as a shape placeholder.

    Returns
    -------
    bool
        The flag pair's verdict (the caller adds the lane conditions).
    """
    fresh = os.environ.get(
        "JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD", "1"
    ).strip().lower() in {"1", "true", "yes", "on"}
    unsafe = os.environ.get(
        "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    return fresh and not unsafe


def _unaliased(tree: Any) -> Any:
    """``tree`` with every array that appears under more than one leaf copied.

    XLA refuses to donate one buffer twice ("Attempt to donate the same buffer
    twice"), and a prepared state can hold the same array under two fields (the
    flat walk's neighbour list uses one leaf-node range for ``leaf_indices`` and
    ``particle_order_leaf_indices``). The repeats are small index arrays.

    Parameters
    ----------
    tree : Any
        A pytree about to be donated.

    Returns
    -------
    Any
        The same pytree, each buffer appearing once.
    """
    seen: set[int] = set()

    def _once(leaf: Any) -> Any:
        if not isinstance(leaf, jax.Array):
            return leaf
        if id(leaf) in seen:
            return jnp.copy(leaf)
        seen.add(id(leaf))
        return leaf

    return jax.tree_util.tree_map(_once, tree)


class StrictRunMixin(_EngineBase):
    def _strict_far_pairs_ride_outside_the_scan(self, prepared: Any) -> bool:
        """Whether ``strict_run_v2`` may keep the state's far list out of the scan.

        On the static-radix fused lane with the fresh far-pair rebuild (the
        default) the carried far list is a placeholder: every refresh builds its
        own and returns the input's unchanged, so it is dead inside the scan, yet
        it was an argument AND an output of the compiled runner (3 x P int32 each,
        1.1 GB at 2.5e7 on the disc+bulge IC). It is detached before the scan and
        re-attached to the returned state, which the gradient path reads.

        Parameters
        ----------
        prepared : Any
            The state about to enter the scan.

        Returns
        -------
        bool
            ``True`` when the list can ride outside.
        """
        mode = str(getattr(self.config.tree, "mode", "")).strip().lower()
        return (
            self.tree_type == "radix"
            and mode == "static_radix"
            and isinstance(prepared, LargeNPreparedState)
            and getattr(prepared, "compact_far_pairs", None) is not None
            and _fresh_compact_pair_rebuild_enabled()
        )

    def _replan_walk_caps_from_needs(self, needs: np.ndarray) -> bool:
        """Raise the validated walk caps to what a failed segment's walks needed.

        Parameters
        ----------
        needs : np.ndarray
            The segment's running maximum of
            :func:`jaccpot.runtime.capacity_guard.last_refresh_walk_needs`.

        Returns
        -------
        bool
            Whether anything was re-planned. ``False`` -- the caller raises -- when
            no walk flag fired (the failure was some other capacity), when a list
            that overflowed was NAMED by the caller (a deliberate bound, never
            widened), or when the lane is not the flat walk.
        """
        from jaccpot.runtime._interaction_cache import (
            _tight_list_capacity,
            flat_walk_cap_headroom,
        )
        from jaccpot.runtime.capacity_guard import WALK_NEEDS_FIELDS

        n = dict(zip(WALK_NEEDS_FIELDS, (int(v) for v in needs)))
        validated = dict(getattr(self, "_strict_fused_validated_caps", None) or {})
        if not validated.get("flat_walk"):
            return False
        if not (n["far_overflow"] or n["near_overflow"] or n["queue_overflow"]):
            return False
        if (n["far_overflow"] and validated.get("far_named")) or (
            n["near_overflow"] and validated.get("near_edge_named")
        ):
            return False
        headroom = flat_walk_cap_headroom()
        # the counts are lower bounds when the walk stops at its first overflow or
        # its queue overflowed (pairs never classified): a list that overflowed
        # then at least doubles
        lower = bool(n["lower_bound"] or n["queue_overflow"])
        for key, needed, ovf in (
            ("compact_far_pair_capacity", n["far_needed"], n["far_overflow"]),
            ("near_edge_capacity", n["near_needed"], n["near_overflow"]),
        ):
            cap = int(validated.get(key) or 0)
            grown = _tight_list_capacity(needed, headroom=headroom, floor=cap)
            if ovf and lower:
                grown = max(grown, 2 * cap)
            validated[key] = int(grown)
        peak = max(int(validated.get("peak_wavefront") or 0), n["peak_wavefront"])
        if n["queue_overflow"]:
            traced = getattr(self, "_strict_fused_traced_caps", None) or {}
            # one past the traced queue: the handoff's pow2(1.5 x peak) doubles it
            peak = max(peak, int(traced.get("queue_capacity") or 0) + 1)
        validated["peak_wavefront"] = int(peak)
        self._strict_fused_validated_caps = validated
        return True

    def _raise_scan_capacity_saturated(self, needs: Optional[np.ndarray]) -> None:
        """Raise the fused scan's capacity error (after any retry was spent).

        Parameters
        ----------
        needs : Optional[np.ndarray]
            The segment's walk needs, named in the message when given.

        Raises
        ------
        RuntimeError
            Always.
        """
        from jaccpot.runtime.capacity_guard import WALK_NEEDS_FIELDS

        max_blocks = os.environ.get(
            "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
            "32",
        )
        traced_caps = getattr(self, "_strict_fused_traced_caps", None) or {}
        walk = (
            ""
            if needs is None
            else " The segment's walks needed: "
            + ", ".join(f"{k}={int(v)}" for k, v in zip(WALK_NEEDS_FIELDS, needs))
            + "."
        )
        raise RuntimeError(
            "a fixed capacity saturated inside the compiled velocity-Verlet "
            "scan, so the refreshed interaction lists are truncated and the "
            "forces from that step on are wrong. Checked: static target-block "
            f"cap (max_blocks_per_leaf={max_blocks}), traced neighbour cap "
            f"({traced_caps.get('max_neighbors_per_leaf_used')} per leaf) and "
            f"compact far-pair cap ({traced_caps.get('compact_far_pair_capacity')}). "
            "Raise JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF, pass "
            "jaccpot.TraversalOverrides(max_neighbors_per_leaf=...), or raise "
            "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP."
            + (
                " On the flat-walk lane (the default; "
                "JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK=0 for the dual walk) "
                "the far-pair count saturates on ANY overflow -- far, near "
                "(JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP) or queue "
                "(TraversalOverrides(max_pair_queue=...)) -- so check all three; "
                "unnamed caps are re-planned once per segment "
                "(JACCPOT_STRICT_SEGMENT_RETRY=0 turns that off), named ones never."
                if traced_caps.get("flat_walk")
                else ""
            )
            + walk
        )

    def refresh_prepared_state(
        self,
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
        """Refresh prepared state under large-N/radix profile constraints.

        Rebinds an existing state to new particle data without a full
        ``prepare_state``. Tries the same-topology fast path first
        (:meth:`_refresh_large_n_same_topology`) and falls back to a full
        preparation when that declines.
        Supported only on the large-N production profile (``preset="large_n_gpu"``,
        radix tree, solidfmm basis); anything else raises rather than silently
        taking a slower path.

        Parameters
        ----------
        prepared_state : PreparedStateLike
            State to refresh. Must be a ``LargeNPreparedState``.
        positions : Array
            New particle positions ``[N, 3]``.
        masses : Array
            New particle masses ``[N]``.
        bounds : Optional[Tuple[Array, Array]]
            Explicit ``(lower, upper)`` domain bounds.
        leaf_size : Optional[int]
            Leaf target; ``None`` keeps the state's own.
        max_order : Optional[int]
            Expansion order; ``None`` keeps the state's own.
        theta : Optional[float]
            Opening angle; ``None`` keeps the state's own.
        fused_device_mode : bool
            Refresh into the fused device-resident layout.

        Returns
        -------
        PreparedStateLike
            A NEW state -- nothing is mutated in place, despite what the two
            wrappers around this one are called.

        Raises
        ------
        NotImplementedError
            If the profile is not large-N production, or the state is not a
            ``LargeNPreparedState``.
        """
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
        # NOTE: the prepare stage has no timer of its own here on purpose -- its
        # cost is already the sum of the `_refresh_timing_{input,tree_upward,
        # dual_downward,nearfield}_seconds` fields accumulated below, and
        # `_refresh_timing_compile_or_sync_suspect_seconds` catches whatever those
        # do not account for. A dead `prepare_elapsed = ...` used to sit here.
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
        self,
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
        """Strict static-radix helper: prepare/refresh once, then evaluate.

        Prepares when ``prepared_state`` is ``None`` and refreshes otherwise, so a
        loop can call this unconditionally and let the first iteration build.
        Returning the state alongside the accelerations is what makes that
        possible.

        Every call is counted against a profile key (particle count, leaf size,
        order, theta); a key it has not seen before counts as a compile. That is
        what the strict-runner diagnostics report, and it is why changing any of
        those four mid-loop is visible rather than merely slow.
        Supported only on the large-N production profile (``preset="large_n_gpu"``,
        radix tree, solidfmm basis); anything else raises rather than silently
        taking a slower path.

        Parameters
        ----------
        prepared_state : Optional[PreparedStateLike]
            State to refresh, or ``None`` to prepare one.
        positions : Array
            Particle positions ``[N, 3]``.
        masses : Array
            Particle masses ``[N]``.
        bounds : Optional[Tuple[Array, Array]]
            Explicit ``(lower, upper)`` domain bounds.
        leaf_size : int
            Target maximum particles per leaf.
        max_order : int
            Expansion order ``p``.
        theta : Optional[float]
            Per-call MAC opening-angle override.
        jit_traversal : Optional[bool]
            Per-call override of jitted traversal.
        runtime_overrides : Optional[_RuntimeExecutionOverrides]
            Replacement runtime overrides for this call.
        fused_device_mode : Optional[bool]
            Use the fused device-resident layout; ``None`` takes the engine's
            current setting.

        Returns
        -------
        tuple[PreparedStateLike, Array]
            ``(prepared_state, accelerations)``. Feed the state back in next
            call.

        Raises
        ------
        RuntimeError
            If the profile is not large-N production.
        """
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

        # Timed, because "per step" for a caller means refresh AND evaluate, while
        # every refresh_* counter covers only the refresh. Leaving the evaluate
        # out made it the single largest term in "unattributed" -- and an
        # unattributed remainder that is really one named stage is a gap in the
        # taxonomy, not a measurement.
        evaluate_timing_active = bool(getattr(self, "_refresh_timing_active", False))
        evaluate_t0 = time.perf_counter() if evaluate_timing_active else 0.0
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
        acc = jnp.asarray(acc)
        if evaluate_timing_active:
            # Block, or this records dispatch rather than the evaluate: the
            # counter's whole purpose is to be comparable with the refresh
            # stages, which are blocked.
            if not _contains_tracer(acc):
                jax.block_until_ready(acc)
            self._refresh_timing_evaluate_seconds += time.perf_counter() - evaluate_t0
        return next_state, acc

    def strict_run_segmented(
        self,
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
        """Run strict refresh/evaluate cadence with caller-provided segment runner.

        The integrator-agnostic runner: it owns only the refresh cadence, and the
        caller supplies the stepping. ``num_steps`` is cut into segments of
        ``refresh_every`` plus a tail, the state is refreshed at each boundary,
        and ``segment_runner`` advances the caller's own state within a segment.
        Use :meth:`strict_run_v2` instead when the integrator is velocity Verlet
        over a raw array.

        Parameters
        ----------
        state : Any
            The caller's integrator state, opaque here: it is only passed to
            ``segment_runner`` and ``positions_getter``.
        masses : Array
            Particle masses ``[N]``.
        num_steps : int
            Total steps. Must be positive.
        refresh_every : int
            Steps per segment. Must be positive.
        segment_runner : Callable[[Any, Array, int], tuple[Any, Any]]
            ``(state, accelerations, num_steps) -> (next_state, output)``. The
            output is collected only under ``collect_history``.
        positions_getter : Callable[[Any], Array]
            Extracts ``[N, 3]`` positions from the caller's state, so the refresh
            knows where the particles are.
        prepared_state : Optional[PreparedStateLike]
            Existing state to refresh, or ``None`` to prepare on the first
            segment.
        leaf_size : int
            Target maximum particles per leaf.
        max_order : int
            Expansion order ``p``.
        theta : Optional[float]
            Per-call MAC opening-angle override.
        jit_traversal : Optional[bool]
            Per-call override of jitted traversal.
        rematerialize_fn : Optional[Callable[[Any], Any]]
            Applied to the state between segments, for callers that must rebuild
            device arrays across a refresh.
        collect_history : bool
            Accumulate each segment's output. Off by default because the history
            is retained on device.

        Returns
        -------
        tuple[Any, PreparedStateLike, Optional[list[Any]]]
            ``(final_state, prepared_state, history)``; ``history`` is ``None``
            unless ``collect_history``.

        Raises
        ------
        ValueError
            If ``num_steps`` or ``refresh_every`` is not positive.
        """
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
        self,
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
        donate_prepared_state: bool = False,
        carry: Optional[str] = None,
        donate_state: bool = False,
    ) -> tuple[Array, Optional[Any], Optional[Array]]:
        """Run endpoint-correct velocity Verlet with strict prepared-state refresh.

        ``step_callback`` is an optional traced, side-effecting hook called inside
        the device-resident scan as ``step_callback(step_index, state)`` every
        ``step_callback_stride`` steps (``step_index`` and ``state`` are traced
        device values). It must be fire-and-forget (return nothing) and should use
        ``jax.debug.callback`` internally to ship only small, on-device-reduced
        data to the host (e.g. a projected density grid), so the GPU is not
        stalled. It does not touch the scan carry and is independent of
        ``return_history``.

        Endpoint-correct velocity Verlet, run device-resident. Unlike
        :meth:`strict_run_segmented` the integrator is fixed and the state is a
        raw array, which is what lets the whole loop live inside one scan.

        ``refresh_every`` must be 1: endpoint correctness needs the self-gravity
        refreshed at every step, so any other value is rejected rather than
        silently approximated.

        Parameters
        ----------
        state : Array
            Packed integrator state ``[N, 2, 3]`` -- positions and velocities
            stacked on axis 1.
        masses : Array
            Particle masses ``[N]``.
        dt : float
            Timestep.
        num_steps : int
            Total steps. Must be positive.
        refresh_every : int
            Must be 1; see above.
        leaf_size : int
            Target maximum particles per leaf.
        max_order : int
            Expansion order ``p``.
        theta : Optional[float]
            Per-call MAC opening-angle override.
        prepared_state : Optional[PreparedStateLike]
            Existing state to refresh, or ``None`` to prepare.
        initial_self_acceleration : Optional[Array]
            Self-gravity at step 0 ``[N, 3]`` if already known; ``None`` costs one
            extra evaluation to obtain it.
        jit_traversal : Optional[bool]
            Per-call override of jitted traversal.
        add_external : bool
            Add ``external_acceleration_fn`` to the self-gravity each step.
        external_acceleration_fn : Optional[Callable[[Array], Array]]
            ``positions -> accelerations``; traced into the scan, so it must be
            jittable.
        rematerialize_between_refresh : bool
            Rebuild device arrays at refresh boundaries.
        return_history : bool
            Return every step's state rather than only the last.
        return_prepared_state : bool
            Return the prepared state so the next call can reuse it.
        step_callback : Optional[Callable[[Array, Array], None]]
            Fire-and-forget streaming hook; see above.
        step_callback_stride : int
            Steps between ``step_callback`` invocations.
        donate_prepared_state : bool
            Hand ``prepared_state``'s buffers to the compiled scan, which then
            writes the returned state into them instead of allocating a second copy
            (the state is the scan's carry: arguments and outputs were each 5.7 GiB
            at 2.5e7 particles). The passed state is CONSUMED -- its arrays are
            deleted -- so pass the state this call returns to the next one, never
            the same one twice. A state this call prepares itself
            (``prepared_state=None``) is always donated: nothing else holds it.
        carry : Optional[str]
            What the fused scan carries between steps: ``"state"`` (the prepared
            state) or ``"particles"`` (positions, velocities and forces only; the
            in-scan refresh rebuilds the rest from a shape template, see
            :mod:`jaccpot.runtime.strict_carry`). ``None`` reads
            ``JACCPOT_STRICT_CARRY`` (default ``"state"``). With ``"particles"``
            the returned prepared state is a
            :class:`~jaccpot.runtime.strict_carry.StrictParticleCarry` handle that
            only the next ``strict_run_v2(carry="particles")`` call accepts; it
            carries the self-gravity at the returned positions, so that call needs
            neither a prepare nor an initial force evaluation.
        donate_state : bool
            ``carry="particles"`` only; off by default, so the input state is kept.
            When set, ``state``'s buffer is handed to the compiled scan, which
            writes the returned state into it instead of allocating a second copy
            (24 B per particle at the scan's peak). The passed ``state`` is
            CONSUMED -- its array is deleted -- so pass the returned state to the
            next call. A capacity retry re-runs the segment from its start, so the
            call keeps a host copy of the start state (one device-to-host copy
            per call, only when donating).

        Returns
        -------
        tuple[Array, Optional[Any], Optional[Array]]
            ``(final_state, prepared_state, history)``. The second is ``None``
            unless ``return_prepared_state``, the third ``None`` unless
            ``return_history``.

        Raises
        ------
        RuntimeError
            If the profile is not large-N production; if fused mode was requested
            at an unsupported particle count; if a refresh hits a
            topology/profile mismatch; or if the fused scan fails while
            ``_strict_fused_disallow_host_segment_fallback`` is set, in which case
            the original error is chained.
        ValueError
            If ``num_steps`` is not positive, or ``refresh_every`` is not 1; if
            ``carry="particles"`` is asked of a lane that cannot rebuild the state
            from its shapes, or a :class:`StrictParticleCarry` comes without it;
            if ``donate_state`` is asked without ``carry="particles"``.
        Exception
            Re-raised unchanged when the fused scan fails and the host-segment
            fallback IS allowed -- the caller sees the underlying failure rather
            than a wrapper, because the fallback path is expected to handle it.
        """
        from jaccpot.runtime.strict_carry import StrictParticleCarry

        state_arr = jnp.asarray(state)
        masses_arr = jnp.asarray(masses)
        dt_arr = jnp.asarray(float(dt), dtype=state_arr.dtype)
        num_steps_i = int(num_steps)
        carry_mode = (
            env_choice("JACCPOT_STRICT_CARRY", "state", ("state", "particles"))
            if carry is None
            else str(carry)
        )
        if carry_mode not in ("state", "particles"):
            raise ValueError(f"carry must be 'state' or 'particles', got {carry!r}")
        if donate_state and carry_mode != "particles":
            raise ValueError(
                "donate_state needs carry='particles' (the state carry donates its "
                "prepared state instead: donate_prepared_state)"
            )
        handle_in = (
            prepared_state if isinstance(prepared_state, StrictParticleCarry) else None
        )
        if handle_in is not None:
            if carry_mode != "particles":
                raise ValueError(
                    "a StrictParticleCarry handle is accepted only with "
                    "carry='particles'"
                )
            if int(handle_in.num_particles) != int(state_arr.shape[0]):
                raise ValueError(
                    f"the handle is for {handle_in.num_particles} particles, the "
                    f"state has {int(state_arr.shape[0])}"
                )
            prepared_state = None

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
        # a state built here is held by nothing else, so the scan may consume it
        donate_carry = bool(donate_prepared_state) or prepared_state is None
        particle_carry = carry_mode == "particles"
        if particle_carry and not self._strict_fused_mode_active:
            raise ValueError("carry='particles' needs the strict fused lane")
        if prepared_curr is None and handle_in is None:
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
        if (
            self._strict_fused_mode_active
            and handle_in is None
            and not isinstance(prepared_curr, LargeNPreparedState)
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
            # `evaluate_large_n_state` takes only the large-N state. The profile
            # guard at the top of `strict_run_v2` already admits nothing else --
            # verified by calling the method on a non-large-N engine, which raises
            # there and never reaches this closure -- but the parameter carries the
            # wider union, so the requirement is stated where it is relied on.
            if not isinstance(prepared_in, LargeNPreparedState):
                raise RuntimeError(
                    "strict_run_v2 reached the large-N self-evaluation with a "
                    f"{type(prepared_in).__name__}. Only the large_n_gpu "
                    "production profile gets this far, so this is an internal "
                    "invariant, not a configuration error."
                )
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

        particle_run = bool(self._strict_fused_mode_active) and particle_carry
        acceleration_self_current: Optional[Array]
        if not self_eval_active:
            acceleration_self_current = jnp.zeros_like(state_arr[:, 0, :])
        elif initial_self_acceleration is None and handle_in is not None:
            acceleration_self_current = jnp.asarray(
                handle_in.self_acceleration, dtype=state_arr.dtype
            )
        elif initial_self_acceleration is None and particle_run:
            # the particle scan evaluates it itself, as its first refresh, once
            # the prepared state is freed: bitwise the eager evaluation, whose
            # temporary block next to the whole prepared state was the first
            # allocation to fail from 1.36e8 particles on (a fragmented arena)
            acceleration_self_current = None
        elif initial_self_acceleration is None:
            acceleration_self_current = _evaluate_self(prepared_curr, state_arr)
        else:
            acceleration_self_current = jnp.asarray(
                initial_self_acceleration, dtype=state_arr.dtype
            )
        if particle_run:
            # the particle run builds its own from the self-gravity, per attempt:
            # a total built here would only sit in the arena through the scan
            acceleration_current = None
        elif add_external and external_acceleration_fn is not None:
            acceleration_current = acceleration_self_current + jnp.asarray(
                external_acceleration_fn(state_arr), dtype=state_arr.dtype
            )
        else:
            acceleration_current = acceleration_self_current

        from jaccpot.nearfield._fast_lane import _nearfield_csr_lane_enabled

        # the CSR near-field lane never reads the rectangle, so its capacity is
        # not a correctness condition there (plan sub-10ms 4.1)
        rectangle_guard_active = not bool(_nearfield_csr_lane_enabled())

        def _static_target_block_capacity_ok(
            prepared_in: PreparedStateLike,
            after_refresh: bool = False,
        ) -> Array:
            # The guard is shared with the multi-GPU lane (`capacity_guard`); the
            # traced caps are host constants recorded while the refresh traced.
            # `after_refresh`: the call directly follows a refresh IN THIS TRACE, so
            # the verdict on the lists it built (which the returned state no longer
            # carries in the fresh-rebuild mode) is live and must be folded in. The
            # initial-state call must not read it: it could be a previous trace's.
            from jaccpot.runtime.capacity_guard import (
                fused_state_capacity_ok,
                last_refresh_capacity_ok,
            )

            ok = fused_state_capacity_ok(
                prepared_in,
                traced_caps=getattr(self, "_strict_fused_traced_caps", None),
                rectangle_guard_active=rectangle_guard_active,
            )
            if after_refresh:
                ok = ok & last_refresh_capacity_ok(self)
            return ok

        def _refresh_and_evaluate_endpoint(
            prepared_in: PreparedStateLike,
            state_position: Array,
            masses_in: Array,
        ) -> tuple[PreparedStateLike, Array]:
            if diag_mode in {"integrator_only", "eval_only"}:
                prepared_new = prepared_in
                # no refresh in this trace: the capacity side channels must not
                # carry a previous trace's verdict into this one
                self._last_refresh_capacity_ok = None
                self._last_refresh_walk_needs = None
            else:
                # Same invariant as `_evaluate_self` above, restated at the
                # second place that relies on it.
                if not isinstance(prepared_in, LargeNPreparedState):
                    raise RuntimeError(
                        "strict_run_v2 reached the large-N refresh with a "
                        f"{type(prepared_in).__name__}. Only the large_n_gpu "
                        "production profile gets this far, so this is an "
                        "internal invariant, not a configuration error."
                    )
                prepared_new = self._refresh_large_n_same_topology(
                    prepared_in,
                    state_position[:, 0, :],
                    masses_in,
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

        def _emit_step(step_index: Array, state_new: Array, fire: Array) -> None:
            # Fire-and-forget streaming hook (e.g. render). Gated by stride (and by
            # ``fire``) via lax.cond so it only fires + only computes its on-device
            # reduction on emit steps. Returns a dummy int so both cond branches
            # match; the result is discarded and the scan carry is untouched.
            assert step_callback is not None

            def _emit(_):
                step_callback(step_index, state_new)
                return jnp.int32(0)

            def _skip(_):
                return jnp.int32(0)

            jax.lax.cond(
                fire & ((step_index % jnp.int32(step_callback_stride)) == jnp.int32(0)),
                _emit,
                _skip,
                operand=None,
            )

        def _advance(
            prepared_now: PreparedStateLike,
            state_now: Array,
            acceleration_now: Array,
            masses_in: Array,
            scan_x: Any,
            emit: bool = True,
        ) -> tuple[PreparedStateLike, Array, Array, Array]:
            # one velocity-Verlet step of the fused scan: drift, refresh + self
            # force at the new positions, kick; shared by both carries
            position_new = (
                state_now[:, 0]
                + state_now[:, 1] * dt_arr
                + 0.5 * acceleration_now * dt_arr**2
            )
            state_position = state_now.at[:, 0].set(position_new)
            prepared_new, acceleration_self_new = _refresh_and_evaluate_endpoint(
                prepared_now, state_position, masses_in
            )
            if add_external and external_acceleration_fn is not None:
                acceleration_new = acceleration_self_new + jnp.asarray(
                    external_acceleration_fn(state_position),
                    dtype=state_now.dtype,
                )
            else:
                acceleration_new = acceleration_self_new
            # kick the drifted state: its positions are the drift's own (the full
            # update recomputed them from the old positions, which then had to
            # outlive the in-place drift for the whole step)
            state_new = _velocity_verlet_kick_drifted(
                state_position,
                acceleration_now,
                acceleration_new,
                dt_arr,
            )
            if rematerialize_between_refresh:
                state_new = jnp.asarray(state_new, dtype=state_now.dtype)
            if emit and step_callback is not None:
                _emit_step(scan_x, state_new, jnp.asarray(True))
            return prepared_new, state_new, acceleration_new, acceleration_self_new

        if self._strict_fused_mode_active and particle_carry:
            # handed over in a box the callee empties: a local here would keep the
            # concrete state (and its far list) alive through the whole scan
            prepared_box = [prepared_curr]
            prepared_curr = None
            state_curr, prepared_curr, history_out = self._strict_particle_carry_run(
                prepared_box=prepared_box,
                handle_in=handle_in,
                state_arr=state_arr,
                masses_arr=masses_arr,
                acceleration_self_current=acceleration_self_current,
                advance=_advance,
                refresh_evaluate=_refresh_and_evaluate_endpoint,
                capacity_ok=_static_target_block_capacity_ok,
                evaluate_self=_evaluate_self,
                num_steps_i=num_steps_i,
                cache_parts=(
                    float(dt),
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
                    id(step_callback) if step_callback is not None else 0,
                    int(step_callback_stride),
                ),
                return_history=bool(return_history),
                step_callback=step_callback,
                emit_step=_emit_step if step_callback is not None else None,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                runtime_overrides=runtime_overrides,
                initial_self_acceleration=initial_self_acceleration,
                add_external=bool(add_external),
                external_acceleration_fn=external_acceleration_fn,
                donate_state=bool(donate_state),
            )
        elif self._strict_fused_mode_active:
            from jaccpot.runtime.capacity_guard import (
                WALK_NEEDS_FIELDS,
                last_refresh_walk_needs,
            )

            def _compiled_runner_for(prepared_ref: PreparedStateLike) -> Callable:
                # Stash the concrete tree depth now, while the state is concrete,
                # so the traced refresh inside the compiled runner passes it as the
                # M2M level-loop static arg. Keyed into cache_key so a topology with
                # a different depth compiles its own runner.
                static_upward_num_levels = self._resolve_upward_num_levels(
                    getattr(prepared_ref, "tree", None)
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
                    # the traced walk is sized from these at TRACE time; a re-plan
                    # that changes them must not reuse a runner traced before it
                    _walk_caps_key(getattr(self, "_strict_fused_validated_caps", None)),
                    donate_carry,
                )
                jit_cache = getattr(self, "_strict_fused_jit_function_cache", {})
                compiled_runner = jit_cache.get(cache_key)
                if compiled_runner is not None:
                    return compiled_runner
                # The new runner traces on its first call, and its scan checks the
                # INITIAL state before this trace's refresh records its traced caps:
                # a record left by an earlier, narrower trace (a re-planned segment)
                # would fail a state that fits. Without one, the check falls back to
                # the state's own widths, which is what the eager prepare validated.
                self._strict_fused_traced_caps = None

                # The masses are an ARGUMENT, not a closure constant: the cache key
                # holds only their shape and dtype, so a closed-over array was
                # silently reused by a later call with different masses.
                @partial(jax.jit, donate_argnums=(0,) if donate_carry else ())
                def _compiled_runner(
                    prepared_initial: LargeNPreparedState,
                    state_initial: Array,
                    acceleration_initial: Array,
                    masses_in: Array,
                ) -> tuple[
                    tuple[LargeNPreparedState, Array, Array, Array, Array],
                    Optional[Array],
                ]:
                    def _step(carry, scan_x):
                        (
                            prepared_now,
                            state_now,
                            acceleration_now,
                            capacity_ok_now,
                            walk_needs_now,
                        ) = carry
                        prepared_new, state_new, acceleration_new, _ = _advance(
                            prepared_now, state_now, acceleration_now, masses_in, scan_x
                        )
                        capacity_ok_new = capacity_ok_now & (
                            _static_target_block_capacity_ok(
                                prepared_new, after_refresh=True
                            )
                        )
                        # what the walks of this segment needed, at most: a failed
                        # segment is re-planned from it
                        walk_needs_new = jnp.maximum(
                            walk_needs_now, last_refresh_walk_needs(self)
                        )
                        return (
                            prepared_new,
                            state_new,
                            acceleration_new,
                            capacity_ok_new,
                            walk_needs_new,
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
                            jnp.zeros((len(WALK_NEEDS_FIELDS),), jnp.int32),
                        ),
                        xs=scan_xs,
                        length=num_steps_i,
                    )

                jit_cache[cache_key] = _compiled_runner
                self._strict_fused_jit_function_cache = jit_cache
                return _compiled_runner

            try:
                retried = False
                while True:
                    # the far list is dead inside the scan (fresh rebuild): keep it
                    # out of the carry and put it back on the returned state
                    far_outside = self._strict_far_pairs_ride_outside_the_scan(
                        prepared_curr
                    )
                    far_kept = prepared_curr.compact_far_pairs if far_outside else None
                    prepared_in = (
                        replace(prepared_curr, compact_far_pairs=None)
                        if far_outside
                        else prepared_curr
                    )
                    if donate_carry:
                        prepared_in = _unaliased(prepared_in)
                        # Caches that may share the state's buffers would hold
                        # deleted arrays after the call: the topology-reuse entry
                        # keeps the prepare's tree (a later prepare_state with the
                        # same key rebuilds from it), the prepared-state slot a
                        # whole state. Both only save a rebuild.
                        self._topology_reuse_entry = None
                        self._prepared_state_cache_key = None
                        self._prepared_state_cache_value = None
                        self._prepared_state_cache_positions = None
                        self._prepared_state_cache_masses = None
                    compiled_runner = _compiled_runner_for(prepared_in)
                    prepared_curr = None  # a donated carry is gone after the call
                    (
                        prepared_out,
                        state_out,
                        _,
                        capacity_ok_all,
                        walk_needs,
                    ), history_out = compiled_runner(
                        prepared_in,
                        state_arr,
                        jnp.asarray(acceleration_current, dtype=state_arr.dtype),
                        masses_arr,
                    )
                    del prepared_in
                    self._strict_static_target_block_capacity_ok = bool(
                        np.asarray(jax.device_get(capacity_ok_all))
                    )
                    if self._strict_static_target_block_capacity_ok:
                        prepared_curr = (
                            replace(prepared_out, compact_far_pairs=far_kept)
                            if far_outside
                            else prepared_out
                        )
                        state_curr = state_out
                        break
                    needs = np.asarray(jax.device_get(walk_needs)).astype(np.int64)
                    # Segment retry: a walk list or queue of the traced refresh
                    # outgrew the caps the eager prepare sized. Re-plan them from
                    # what the segment's walks needed, re-prepare from the
                    # segment's START, recompile and run the segment once more.
                    # Anything else (a named cap, a leaf capacity, a second failure)
                    # raises below.
                    if (
                        retried
                        or not env_flag("JACCPOT_STRICT_SEGMENT_RETRY", True)
                        or not self._replan_walk_caps_from_needs(needs)
                    ):
                        self._raise_scan_capacity_saturated(needs)
                    retried = True
                    del prepared_out, state_out, history_out, far_kept
                    replanned_peak = int(
                        (self._strict_fused_validated_caps or {}).get(
                            "peak_wavefront", 0
                        )
                    )
                    prepared_curr = self.prepare_state(
                        state_arr[:, 0, :],
                        masses_arr,
                        leaf_size=int(leaf_size),
                        max_order=int(max_order),
                        theta=theta,
                        jit_tree=self._jit_tree_default,
                        runtime_overrides_override=runtime_overrides,
                        fused_device_mode=True,
                    )
                    # the eager walk saw the segment's START; keep the wavefront the
                    # segment needed, so the retraced walk's queue covers it
                    validated = dict(self._strict_fused_validated_caps or {})
                    validated["peak_wavefront"] = max(
                        int(validated.get("peak_wavefront") or 0), replanned_peak
                    )
                    self._strict_fused_validated_caps = validated
                    # the starting force from the state the segment now starts from
                    # (the same field when the caller's state matched its positions)
                    if initial_self_acceleration is None:
                        acceleration_current = _evaluate_self(prepared_curr, state_arr)
                        if add_external and external_acceleration_fn is not None:
                            acceleration_current = acceleration_current + jnp.asarray(
                                external_acceleration_fn(state_arr),
                                dtype=state_arr.dtype,
                            )
                    self._strict_fused_fallback_count += 1
                    self._strict_fused_last_fallback_reason = "capacity_segment_retry"
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
                    prepared_curr, state_position, masses_arr
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

    def _strict_particle_carry_run(
        self,
        *,
        prepared_box: list,
        handle_in: Optional[Any],
        state_arr: Array,
        masses_arr: Array,
        acceleration_self_current: Optional[Array],
        advance: Callable[..., Any],
        refresh_evaluate: Callable[..., Any],
        capacity_ok: Callable[..., Array],
        evaluate_self: Callable[..., Array],
        num_steps_i: int,
        cache_parts: tuple,
        return_history: bool,
        step_callback: Optional[Callable[[Array, Array], None]],
        emit_step: Optional[Callable[[Array, Array, Array], None]],
        leaf_size: int,
        max_order: int,
        theta: Optional[float],
        runtime_overrides: Any,
        initial_self_acceleration: Optional[Array],
        add_external: bool,
        external_acceleration_fn: Optional[Callable[[Array], Array]],
        donate_state: bool,
    ) -> tuple[Array, Any, Optional[Array]]:
        """The fused scan of ``strict_run_v2(carry="particles")``.

        The scan carries ``(state, acceleration, self acceleration, ok, walk
        needs, steps that fitted)``; each step materialises the prepared state
        from its shape template (:mod:`jaccpot.runtime.strict_carry`) and
        refreshes it from the drifted positions, as the state carry does. The concrete prepared state
        (if any) is dropped before the scan, so it and the far list are freed.

        A segment whose refresh overflows a list is re-run once from its start
        with re-planned caps (the stream stops at the failed step, so it never
        sees a failed step's state). ``donate_state`` hands the start state to the
        scan, so a host copy of it is kept for that re-run.

        Parameters
        ----------
        prepared_box : list
            ``[prepared]``: the concrete prepared state at ``state_arr``'s
            positions, or ``[None]`` when ``handle_in`` brings the template. The
            box is emptied, so nothing here keeps the state alive in the scan.
        handle_in : Optional[Any]
            The previous call's :class:`StrictParticleCarry`, or ``None``.
        state_arr : Array
            ``[N, 2, 3]`` start state.
        masses_arr : Array
            ``[N]`` masses.
        acceleration_self_current : Optional[Array]
            Self-gravity at the start; the total acceleration is built from it
            (plus the external field) per attempt, and donated to the scan when
            this call owns it. ``None``: the scan evaluates it itself, as a
            refresh at the start positions before its first step.
        advance : Callable[..., Any]
            ``strict_run_v2``'s step body.
        refresh_evaluate : Callable[..., Any]
            ``strict_run_v2``'s refresh + self-force at given positions (the
            step's own), for the scan's start force.
        capacity_ok : Callable[..., Array]
            ``strict_run_v2``'s capacity verdict on a (refreshed) state.
        evaluate_self : Callable[..., Array]
            ``strict_run_v2``'s self-force evaluation.
        num_steps_i : int
            Steps.
        cache_parts : tuple
            The call's static configuration, part of the compile-cache key.
        return_history : bool
            Stack every step's state.
        step_callback : Optional[Callable[[Array, Array], None]]
            Streaming hook (feeds the step index).
        emit_step : Optional[Callable[[Array, Array, Array], None]]
            ``strict_run_v2``'s stride-gated emitter ``(step, state, fire)``: the
            scan fires it only for steps whose refresh fitted.
        leaf_size : int
            Leaf target (re-prepare on a segment retry).
        max_order : int
            Expansion order (same).
        theta : Optional[float]
            Opening angle (same).
        runtime_overrides : Any
            Runtime overrides (same).
        initial_self_acceleration : Optional[Array]
            The caller's starting self-gravity, if given.
        add_external : bool
            Whether an external field is added.
        external_acceleration_fn : Optional[Callable[[Array], Array]]
            The external field.
        donate_state : bool
            Hand ``state_arr``'s buffer to the scan (``strict_run_v2``'s
            ``donate_state``).

        Returns
        -------
        tuple[Array, Any, Optional[Array]]
            ``(final_state, StrictParticleCarry, history)``. A lane that would read
            the carried state (far list not rebuilt fresh in the scan) raises
            ``ValueError``; a failed scan under
            ``_strict_fused_disallow_host_segment_fallback`` a chained
            ``RuntimeError``, as the state carry does.
        """
        from jaccpot.runtime.capacity_guard import (
            WALK_NEEDS_FIELDS,
            last_refresh_walk_needs,
        )
        from jaccpot.runtime.strict_carry import (
            StrictParticleCarry,
            materialize_template,
            shape_template,
        )

        def _template_of(prepared: Any) -> Any:
            if not self._strict_far_pairs_ride_outside_the_scan(prepared):
                raise ValueError(
                    "carry='particles' needs the fresh far-pair rebuild inside the "
                    "scan (JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD): "
                    "otherwise a step reads the carried far list"
                )
            return shape_template(replace(prepared, compact_far_pairs=None))

        def _runner_for(
            template: Any,
            n_steps: int,
            donate_state_i: bool,
            donate_acc_i: bool,
            start_force: bool,
        ) -> Callable:
            # the cells lane's level bound is stashed by the eager prepare
            static_upward_num_levels = self._resolve_upward_num_levels(None)
            cache_key = (
                "strict_velocity_verlet_particles",
                tuple(int(v) for v in state_arr.shape),
                str(state_arr.dtype),
                tuple(int(v) for v in masses_arr.shape),
                str(masses_arr.dtype),
                int(n_steps),
                *cache_parts,
                static_upward_num_levels,
                _walk_caps_key(getattr(self, "_strict_fused_validated_caps", None)),
                template.key(),
                bool(donate_state_i),
                bool(donate_acc_i),
                bool(start_force),
            )
            jit_cache = getattr(self, "_strict_fused_jit_function_cache", {})
            runner = jit_cache.get(cache_key)
            if runner is not None:
                return runner
            self._strict_fused_traced_caps = None

            # named like the state carry's runner: dumps and the stage analyser
            # select ``*_compiled_runner*``. The initial acceleration is donated
            # when this call built it: its buffer becomes the self-gravity output,
            # and the total acceleration (which the caller never reads) is no
            # output at all -- two (N, 3) arrays fewer next to the scan's block.
            # The state is donated on request (``donate_state``) or when it is
            # this call's own copy (a retry's): its buffer becomes the result.
            def _scan(
                state_initial: Array,
                acceleration_initial: Array,
                acc_self_initial: Array,
                masses_in: Array,
                ok_initial: Array,
                needs_initial: Array,
            ) -> tuple[tuple[Array, Array, Array, Array, Array], Optional[Array]]:
                def _step(carry, scan_x):
                    state_now, acc_now, _, ok_now, needs_now, done = carry
                    prepared_new, state_new, acc_new, acc_self_new = advance(
                        materialize_template(template),
                        state_now,
                        acc_now,
                        masses_in,
                        scan_x,
                        emit=False,
                    )
                    ok_new = ok_now & capacity_ok(prepared_new, after_refresh=True)
                    needs_new = jnp.maximum(needs_now, last_refresh_walk_needs(self))
                    # a segment that overflowed is re-run from its start, so the
                    # carry is NOT frozen (keeping the step's old state for a
                    # where() held 36 B per particle through every step); only
                    # the stream stops, so it never sees a failed step's state
                    if emit_step is not None:
                        emit_step(scan_x, state_new, ok_new)
                    return (
                        state_new,
                        acc_new,
                        acc_self_new,
                        ok_new,
                        needs_new,
                        done + ok_new.astype(jnp.int32),
                    ), (state_new if return_history else None)

                scan_xs = (
                    jnp.arange(n_steps, dtype=jnp.int32)
                    if emit_step is not None
                    else None
                )
                (state_f, _, acc_self_f, ok_f, needs_f, done_f), history = jax.lax.scan(
                    _step,
                    (
                        state_initial,
                        acceleration_initial,
                        acc_self_initial,
                        ok_initial,
                        needs_initial,
                        jnp.zeros((), jnp.int32),
                    ),
                    xs=scan_xs,
                    length=int(n_steps),
                )
                return (state_f, acc_self_f, ok_f, needs_f, done_f), history

            no_needs = jnp.zeros((len(WALK_NEEDS_FIELDS),), jnp.int32)
            if start_force:

                @partial(jax.jit, donate_argnums=(0,) if donate_state_i else ())
                def _compiled_runner(
                    state_initial: Array,
                    masses_in: Array,
                    ok_initial: Array,
                ) -> tuple[tuple[Array, Array, Array, Array, Array], Optional[Array]]:
                    # the start force, as the steps' own refresh at the start
                    # positions (bitwise the eager prepare + evaluation)
                    prepared0, acc_self0 = refresh_evaluate(
                        materialize_template(template), state_initial, masses_in
                    )
                    ok0 = ok_initial & capacity_ok(prepared0, after_refresh=True)
                    needs0 = jnp.maximum(no_needs, last_refresh_walk_needs(self))
                    return _scan(
                        state_initial,
                        _initial_acceleration(acc_self0, state_initial),
                        jnp.zeros_like(acc_self0),
                        masses_in,
                        ok0,
                        needs0,
                    )

            else:
                donate = ((0,) if donate_state_i else ()) + (
                    (1,) if donate_acc_i else ()
                )

                @partial(jax.jit, donate_argnums=donate)
                def _compiled_runner(
                    state_initial: Array,
                    acceleration_initial: Array,
                    masses_in: Array,
                    ok_initial: Array,
                ) -> tuple[tuple[Array, Array, Array, Array, Array], Optional[Array]]:
                    # the self-gravity slot is written by every step (num_steps >= 1)
                    return _scan(
                        state_initial,
                        acceleration_initial,
                        jnp.zeros_like(acceleration_initial),
                        masses_in,
                        ok_initial,
                        no_needs,
                    )

            jit_cache[cache_key] = _compiled_runner
            self._strict_fused_jit_function_cache = jit_cache
            return _compiled_runner

        external_active = bool(add_external) and external_acceleration_fn is not None
        # the self-gravity is this call's own (evaluated here, or zeros) unless it
        # came from the caller or from the handle: only then may it be donated
        self_owned = handle_in is None and initial_self_acceleration is None

        def _initial_acceleration(acc_self: Array, state_now: Array) -> Array:
            acc_self = jnp.asarray(acc_self, dtype=state_now.dtype)
            if not external_active:
                return acc_self
            assert external_acceleration_fn is not None
            return acc_self + jnp.asarray(
                external_acceleration_fn(state_now), dtype=state_now.dtype
            )

        prepared_curr = prepared_box.pop() if prepared_box else None
        if handle_in is not None:
            template = handle_in.template
            ok_initial = jnp.asarray(True)  # the segment that made it fitted
        else:
            template = _template_of(prepared_curr)
            # the scan's initial verdict, on the concrete state before it is freed,
            # against the state's own widths (what the eager prepare validated),
            # not a traced-caps record an earlier trace may have left
            saved_caps = getattr(self, "_strict_fused_traced_caps", None)
            self._strict_fused_traced_caps = None
            ok_initial = jnp.asarray(capacity_ok(prepared_curr))
            self._strict_fused_traced_caps = saved_caps
        prepared_curr = None
        # caches that would keep the prepare's buffers alive through the scan
        self._topology_reuse_entry = None
        self._prepared_state_cache_key = None
        self._prepared_state_cache_value = None
        self._prepared_state_cache_positions = None
        self._prepared_state_cache_masses = None
        retry_enabled = env_flag("JACCPOT_STRICT_SEGMENT_RETRY", True)
        start_host: Optional[np.ndarray] = None
        if donate_state and retry_enabled:
            # the scan consumes the start state, and a capacity retry re-runs the
            # segment from it: keep it on the host (one copy per call, only when
            # donating -- the scan itself stays as lean as it can be)
            start_host = np.asarray(jax.device_get(state_arr))
        try:
            retried = False
            state_now = state_arr
            acc_self_now: Optional[Array] = acceleration_self_current
            del acceleration_self_current
            acc_self_owned = self_owned
            in_scan_force = acc_self_now is None
            donate_state_now = bool(donate_state)
            while True:
                runner = _runner_for(
                    template,
                    num_steps_i,
                    donate_state_now,
                    external_active or acc_self_owned,
                    in_scan_force,
                )
                if in_scan_force:
                    (state_out, acc_self_out, ok_all, walk_needs, done), history_out = (
                        runner(state_now, masses_arr, ok_initial)
                    )
                else:
                    assert acc_self_now is not None
                    acc0 = _initial_acceleration(acc_self_now, state_now)
                    if acc_self_owned and not external_active:
                        acc_self_now = None  # it is acc0, which the call consumes
                    (state_out, acc_self_out, ok_all, walk_needs, done), history_out = (
                        runner(state_now, acc0, masses_arr, ok_initial)
                    )
                    del acc0
                self._strict_static_target_block_capacity_ok = bool(
                    np.asarray(jax.device_get(ok_all))
                )
                if self._strict_static_target_block_capacity_ok:
                    handle = StrictParticleCarry(
                        template=template,
                        self_acceleration=acc_self_out,
                        num_particles=int(state_arr.shape[0]),
                    )
                    return state_out, handle, history_out
                needs = np.asarray(jax.device_get(walk_needs)).astype(np.int64)
                # diagnostics: the steps the failed segment completed
                self._strict_particle_failed_step = int(
                    np.asarray(jax.device_get(done))
                )
                # segment retry: re-plan the walk caps from what the segment
                # needed, re-prepare at the segment's START, run it once more
                if (
                    retried
                    or not retry_enabled
                    or not self._replan_walk_caps_from_needs(needs)
                ):
                    self._raise_scan_capacity_saturated(needs)
                retried = True
                del state_out, history_out, acc_self_out
                if donate_state_now:
                    assert start_host is not None
                    state_now = jax.device_put(start_host, masses_arr.sharding)
                    donate_state_now = True  # our own copy
                replanned_peak = int(
                    (self._strict_fused_validated_caps or {}).get("peak_wavefront", 0)
                )
                prepared = self.prepare_state(
                    state_now[:, 0, :],
                    masses_arr,
                    leaf_size=int(leaf_size),
                    max_order=int(max_order),
                    theta=theta,
                    jit_tree=self._jit_tree_default,
                    runtime_overrides_override=runtime_overrides,
                    fused_device_mode=True,
                )
                validated = dict(self._strict_fused_validated_caps or {})
                validated["peak_wavefront"] = max(
                    int(validated.get("peak_wavefront") or 0), replanned_peak
                )
                self._strict_fused_validated_caps = validated
                if acc_self_now is None and not in_scan_force:
                    # consumed by the failed segment: the same self-gravity
                    # again, from the same positions
                    acc_self_now = evaluate_self(prepared, state_now)
                template = _template_of(prepared)
                self._strict_fused_traced_caps = None
                ok_initial = jnp.asarray(capacity_ok(prepared))
                del prepared
                self._topology_reuse_entry = None
                self._prepared_state_cache_value = None
                self._strict_fused_fallback_count += 1
                self._strict_fused_last_fallback_reason = "capacity_segment_retry"
        except Exception as exc:
            if bool(
                getattr(self, "_strict_fused_disallow_host_segment_fallback", False)
            ):
                raise RuntimeError(
                    "strict fused velocity-Verlet scan failed while host fallback "
                    "is disallowed"
                ) from exc
            raise

    def strict_fused_prepared_eval_fn(
        self,
        *,
        positions: Array,
        masses: Array,
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
        bounds: Optional[tuple[Array, Array]] = None,
        donate_prepared: bool = False,
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

        Parameters
        ----------
        positions : Array
            Particle positions ``[N, 3]``.
        masses : Array
            Particle masses ``[N]``.
        leaf_size : int
            Target maximum particles per leaf.
        max_order : int
            Expansion order ``p``.
        theta : Optional[float]
            Per-call MAC opening-angle override.
        bounds : Optional[tuple[Array, Array]]
            Morton box for the tree; ``None`` infers it from ``positions``. A mesh
            device passes the GLOBAL box its traced force builds in, so the eager
            tree -- whose leaf count, level widths and walk caps size the static
            shapes -- is the tree that force walks, not one cut in the shard's own,
            smaller box (finer cells: 33,388 leaves against 26,302 on one 1e6 shard).
        donate_prepared : bool
            Jit ``eval_fn`` with its argument donated, so the evaluation may reuse
            the state's buffers for its temporaries. ``eval_fn`` then CONSUMES the
            state: one call per state. Off by default -- the seam exists to time
            repeated calls on one state.

        Returns
        -------
        tuple[PreparedStateLike, Callable[[PreparedStateLike], Array]]
            ``(prepared_state, eval_fn)``. A benchmarking seam, not a simulation
            entry point -- ``eval_fn`` deliberately omits the refresh and the
            Verlet update that ``strict_run_v2`` bundles with the same
            evaluation.

        Raises
        ------
        RuntimeError
            If the profile is not large-N production.
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
            **({} if bounds is None else {"bounds": bounds}),
        )
        if not isinstance(prepared, LargeNPreparedState):
            raise RuntimeError("strict fused eval-only requires a LargeNPreparedState.")
        self._record_large_n_eval_shape_diagnostics(prepared)

        if donate_prepared:
            # see strict_run_v2: caches sharing the state's buffers must not
            # outlive a donating call
            self._topology_reuse_entry = None
            self._prepared_state_cache_value = None
            self._prepared_state_cache_key = None

        @partial(jax.jit, donate_argnums=(0,) if donate_prepared else ())
        def _eval(prepared_in: LargeNPreparedState) -> Array:
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
        self,
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
        num_valid: Optional[Array] = None,
        cross_hook: Optional[Callable[[Any], None]] = None,
    ) -> Optional[LargeNPreparedState]:
        """Refresh large-N numeric payloads when the radix topology is unchanged.

        The fast path behind :meth:`refresh_prepared_state`: when the Morton
        topology is unchanged, only the numeric payloads need rebuilding, not the
        tree or the interaction lists.

        **``None`` is a miss, not a failure.** Every early return increments a
        named counter (no radix tree, traced inputs, neighbour list changed, ...)
        and hands the decision back to the caller, which then does a full
        preparation. Nothing here raises to signal "could not reuse" -- that is
        the protocol, and it is why the diagnostics carry a miss breakdown rather
        than a single hit rate.

        Parameters
        ----------
        prepared_state : LargeNPreparedState
            State whose payloads are refreshed.
        positions : Array
            New particle positions ``[N, 3]``.
        masses : Array
            New particle masses ``[N]``.
        bounds : Optional[Tuple[Array, Array]]
            Explicit ``(lower, upper)`` domain bounds.
        leaf_size : int
            Leaf target.
        max_order : int
            Expansion order ``p``.
        theta : Optional[float]
            Opening angle; ``None`` keeps the state's own.
        runtime_overrides_override : Optional[_RuntimeExecutionOverrides]
            Replacement runtime overrides for this refresh.
        fused_device_mode : bool
            Refresh into the fused device-resident layout. Also relaxes the
            traced-input guard, since the fused lane is designed to be traced.
        num_valid : Optional[Array]
            Live row count of a capacity-padded shard (the distributed fused
            lane); ``None`` treats every row as live. Requires cell leaves.
        cross_hook : Optional[Callable[[Any], None]]
            Called once per refresh between the upward and downward sweeps with the
            tree artifacts (`jaccpot.distributed.cross.make_cross_hook`). It returns
            ``(multipoles, centers, src, tgt)`` for the cross-domain far field, which the
            downward sweep concatenates behind the local nodes, or ``None``. ``None``
            (default) is the single-domain lane, bit-identical.

        Returns
        -------
        Optional[LargeNPreparedState]
            The refreshed state, or ``None`` when the fast path declined -- see
            above.

        Raises
        ------
        RuntimeError
            Only for genuine inconsistencies, not for a declined reuse.
        """
        # Side channels for the traced capacity guard and the segment retry; see
        # the end of this method and `capacity_guard.last_refresh_walk_needs`.
        self._last_refresh_capacity_ok = None
        self._last_refresh_walk_needs = None

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
                num_valid=num_valid,
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

        # as the prepare: the COM walk never reads the box geometry
        defer_geometry = self._defers_box_geometry(tree_config.mode, upward_center_mode)
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
            and _fresh_compact_pair_rebuild_enabled()
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
        # The one point where a cross-domain exchange belongs: the multipoles exist
        # here and the downward sweep -- either branch below -- has not consumed them.
        # It goes in the REFRESH because that is what a per-step force runs; the
        # prepare path builds the state once, and `LargeNPreparedState.upward` is None
        # (measured), so no caller holding a state can reach the multipoles at all.
        # Doing the cross field afterwards instead would need a SECOND L2L cascade,
        # and Phase 3.4 measured the far half as the bigger one.
        #
        # ABOVE the branch, not inside one: the compact-pair reuse path and the
        # general path both build a downward sweep, and a hook in only one of them
        # silently never fires on the other -- which is exactly what happened first.
        #
        # Phase C1: called, result discarded, force bit-identical either way.
        cross_far = None
        if cross_hook is not None:
            # ONE exact-radius COM geometry for the hook and the local walk below:
            # each used to resolve its own, 9.7 ms per force at 1e6 particles per
            # A100. Shared only where the two would compute the same thing -- the
            # strict fused lane (COM by default) and no folded per-node radius scale
            # (`dehnen_theta` folds one the hook does not apply).
            if (
                getattr(self, "_strict_fused_mode_active", False)
                and self._folded_criterion_radius_scale() is None
            ):
                tree_artifacts = tree_artifacts._replace(
                    walk_geometry=self._strict_walk_geometry(tree_artifacts)
                )
            cross_far = cross_hook(tree_artifacts)
        if reuse_static_compact_pairs:
            from jaccpot.runtime.fmm_prepare import _gear_pairs_for_autotune

            src_far = jnp.asarray(cached_compact_far_pairs.sources, dtype=INDEX_DTYPE)
            tgt_far = jnp.asarray(cached_compact_far_pairs.targets, dtype=INDEX_DTYPE)
            far_pairs_by_gear = _gear_pairs_for_autotune(
                cached_compact_far_pairs, src_far, tgt_far
            )
            downward = self._prepare_downward_with_artifacts(
                cross_far=cross_far,
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
                far_pairs_coo=_far_pair_coo_from(
                    cached_compact_far_pairs, src_far, tgt_far
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
                cross_far=cross_far,
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
            if reuse_static_compact_pairs:
                # The compact-far-pair reuse branch above does NOT go through
                # _prepare_state_dual_and_downward, so nothing else records this
                # stage -- and this is the steady-state route once topology is
                # frozen, which is precisely when a per-step breakdown is being
                # read. Leaving it unrecorded made the entire downward pass
                # (~30% of per-step time at N=65536) land in "unattributed", and
                # made every dual_* counter read as a hard zero.
                self._refresh_timing_dual_downward_seconds += elapsed
            # Otherwise _prepare_state_dual_and_downward has already recorded
            # this stage and its children; adding elapsed here would double-count.

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
            request=LargeNPrepareRequest(
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
            ),
            tree_artifacts=tree_artifacts,
            dual_downward_artifacts=dual_downward_artifacts,
            fused_device_mode=bool(fused_device_mode),
        )
        # The capacity verdict of the lists THIS refresh built. Under trace a walk
        # overflow (far / near / queue) or a leaf-capacity overflow surfaces only as
        # a saturated `compact_far_pairs.far_pair_count`, and in the fresh-rebuild
        # mode below that freshly built list is swapped for the cached placeholder
        # before the state is returned (carry shapes must not change) -- so a guard
        # reading the RETURNED state sees the prepare's count, never this one. It is
        # left here, as a tracer, for the caller to read inside the same trace
        # (`capacity_guard.last_refresh_capacity_ok`). It must not be read anywhere
        # else: it is cleared at the start of every refresh.
        from jaccpot.runtime.capacity_guard import fused_state_capacity_ok

        self._last_refresh_capacity_ok = fused_state_capacity_ok(refreshed_state)
        if bool(safe_fresh_compact_pair_rebuild):
            return replace(
                refreshed_state,
                compact_far_pairs=compact_far_pairs_carry_placeholder,
            )
        return refreshed_state

    def _large_n_neighbor_list_matches(
        self,
        previous: NodeNeighborList,
        current: NodeNeighborList,
    ) -> bool:
        """Return True when current active neighbor edges match previous state.

        Compares only the **active** prefix ``neighbors[:active_edges]``, where
        the active count comes from the CSR offsets. The arrays are
        fixed-capacity and padded, so comparing them whole would report a
        difference whenever the padding differs -- which says nothing about the
        edges.

        Conservative by construction: any exception is caught and reported as
        "changed". A false negative costs a full rebuild; a false positive would
        reuse a stale topology, which is a wrong force.

        Parameters
        ----------
        previous : NodeNeighborList
            Neighbour list from the prepared state.
        current : NodeNeighborList
            Newly built neighbour list.

        Returns
        -------
        bool
            ``True`` only when offsets, counts, leaf indices and the active edge
            prefix all match exactly.
        """

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
        self,
        prepared_state: PreparedStateLike,
        positions: Array,
        masses: Array,
        *,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
    ) -> PreparedStateLike:
        """Refresh multipole/local payloads when topology key remains unchanged.

        A :meth:`refresh_prepared_state` under a different name and its own
        diagnostic counter, for the case where the caller knows the tree mapping
        still holds. It takes no ``bounds``, since changing the domain would
        change that mapping. Large-N production profile only.

        Parameters
        ----------
        prepared_state : PreparedStateLike
            State whose payloads are refreshed.
        positions : Array
            New particle positions ``[N, 3]``.
        masses : Array
            New particle masses ``[N]``.
        leaf_size : Optional[int]
            Leaf target; ``None`` keeps the state's own.
        max_order : Optional[int]
            Expansion order; ``None`` keeps the state's own.
        theta : Optional[float]
            Opening angle; ``None`` keeps the state's own.

        Returns
        -------
        PreparedStateLike
            The refreshed state.

        Raises
        ------
        NotImplementedError
            If the profile is not large-N production, or the state is not a
            ``LargeNPreparedState``.
        RuntimeError
            If the refresh could not preserve the topology mapping the caller
            asserted was unchanged.
        """
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
        self,
        prepared_state: PreparedStateLike,
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
    ) -> PreparedStateLike:
        """Rebuild topology while attempting to remain profile-capacity compatible.

        The third face of :meth:`refresh_prepared_state`: same call, its own
        counter, and ``bounds`` forwarded because a rebuild may legitimately
        change the domain.

        "In place" names the intent to stay within the compiled profile's
        capacities so the existing executables keep applying -- NOT mutation. A
        new state is returned and the input is untouched. Large-N production
        profile only.

        Parameters
        ----------
        prepared_state : PreparedStateLike
            State whose topology is rebuilt.
        positions : Array
            New particle positions ``[N, 3]``.
        masses : Array
            New particle masses ``[N]``.
        bounds : Optional[Tuple[Array, Array]]
            Explicit ``(lower, upper)`` domain bounds.
        leaf_size : Optional[int]
            Leaf target; ``None`` keeps the state's own.
        max_order : Optional[int]
            Expansion order; ``None`` keeps the state's own.
        theta : Optional[float]
            Opening angle; ``None`` keeps the state's own.

        Returns
        -------
        PreparedStateLike
            The rebuilt state.

        Raises
        ------
        NotImplementedError
            If the profile is not large-N production, or the state is not a
            ``LargeNPreparedState``.
        """
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
