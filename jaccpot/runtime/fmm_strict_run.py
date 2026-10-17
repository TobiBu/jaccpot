"""StrictRunMixin: fmm_strict_run methods extracted from the FMMEngine
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Callable, Tuple
from jaxtyping import Array
from yggdrax.interactions import DualTreeRetryEvent, NodeNeighborList
from yggdrax.tree import RadixTree, get_level_offsets

from jaccpot._env import env_choice, env_flag

from ._large_n_pipeline import prepare_large_n_state
from ._large_n_types import LargeNPreparedState, LargeNPrepareRequest
from ._mac_geometry import walk_geometry_default
from .capacity_plan import auto_leaf_capacity
from .dtypes import INDEX_DTYPE
from .fmm_caches import _contains_tracer
from .fmm_state import (
    TreeBuilderConfig,
    _PrepareStateTreeUpwardArtifacts,
    _RuntimeExecutionOverrides,
    _TopologyReuseEntry,
    _velocity_verlet_kick_drifted,
)
from .fused_provider import FusedForceProvider, require_cell_leaves
from .fused_topology import ForceParams, FusedTopology, StepParams, _is_static

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

#: The attribute of a state carry's returned ``LargeNPreparedState`` that holds the
#: scan's last eq (16b) ``f_b`` under ``mac_type='dehnen_error'`` (input order).
_CARRIED_FORCE_SCALE = "_strict_force_scale"


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
        """Whether ``strict_run_v2`` keeps the state's far list out of the scan.

        On the static-radix fused lane the fresh far-pair rebuild (its only mode
        since the 2026-10 cleanup, X6) makes the carried far list a placeholder:
        every refresh builds its own and returns the input's unchanged, so it is
        dead inside the scan, yet it was an argument AND an output of the
        compiled runner (3 x P int32 each, 1.1 GB at 2.5e7 on the disc+bulge IC).
        It is detached before the scan and NOT put back: it belongs to the
        positions the call started from, while the returned tree and positions
        are the end's, so the gradient path read the wrong far field from it
        (rel-L2 7.5e-3 against 7.9e-5 at N=768, the 2026-10-13 review). The
        returned state carries ``compact_far_pairs=None``, which
        ``differentiable_accelerations`` refuses with a clear error.

        A state without a far list (one this method returned) qualifies too: the
        lane, not the list, is what makes a step read none of it.

        Parameters
        ----------
        prepared : Any
            The state about to enter the scan.

        Returns
        -------
        bool
            ``True`` on the lane whose scan rebuilds the far list every step.
        """
        mode = str(getattr(self.config.tree, "mode", "")).strip().lower()
        return (
            self.tree_type == "radix"
            and mode == "static_radix"
            and isinstance(prepared, LargeNPreparedState)
        )

    def _replan_walk_caps_from_needs(self, needs: np.ndarray) -> bool:
        """Raise the validated walk caps to what a failed segment's walks needed.

        Parameters
        ----------
        needs : np.ndarray
            The segment's running maximum of
            :meth:`jaccpot.runtime.fused_topology.FusedTopology.needs` (its walk
            entries; :meth:`_replan_tree_from_needs` reads the tree's).

        Returns
        -------
        bool
            Whether anything was re-planned. ``False`` -- the caller raises -- when
            no walk flag fired (the failure was some other capacity), when a list
            that overflowed was NAMED by the caller (a deliberate bound, never
            widened), or when the lane is not the flat walk.

        Raises
        ------
        RuntimeError
            When a list that overflowed needs more than the index range can build
            (:func:`jaccpot.runtime._interaction_cache.flat_walk_list_limit`):
            here, before the re-prepare, since no re-plan fits it.
        """
        from jaccpot.runtime._interaction_cache import (
            _tight_list_capacity,
            flat_walk_cap_headroom,
            flat_walk_list_limit,
            flat_walk_list_limit_message,
            strict_walk_backend,
        )
        from jaccpot.runtime.capacity_guard import WALK_NEEDS_FIELDS

        n = dict(zip(WALK_NEEDS_FIELDS, (int(v) for v in needs)))
        validated = dict(getattr(self, "_strict_fused_validated_caps", None) or {})
        if not validated.get("flat_walk"):
            return False
        if not (n["far_overflow"] or n["near_overflow"] or n["queue_overflow"]):
            return False
        if self._walk_overflowed_a_named_cap(needs):
            return False
        headroom = flat_walk_cap_headroom()
        # the counts are lower bounds when the walk stops at its first overflow or
        # its queue overflowed (pairs never classified): a list that overflowed
        # then at least doubles
        lower = bool(n["lower_bound"] or n["queue_overflow"])
        # Neither the headroom nor the doubling may carry a list past the index
        # range (2/3 of int32 for the Pallas walk's CSRs): the re-prepare would
        # fail deep in its list build. A list clamped there still holds what it
        # needed; one that needs more fits no width, so the retry stops here.
        limit = flat_walk_list_limit(INDEX_DTYPE, csr=strict_walk_backend() == "pallas")
        for key, kind, needed, ovf in (
            ("compact_far_pair_capacity", "far", n["far_needed"], n["far_overflow"]),
            ("near_edge_capacity", "near", n["near_needed"], n["near_overflow"]),
        ):
            cap = int(validated.get(key) or 0)
            if ovf and needed > limit:
                raise RuntimeError(
                    flat_walk_list_limit_message(
                        kind, f"needs {needed} directed entries", limit=limit
                    )
                )
            if ovf and (needed < 0 or (lower and cap >= limit)):
                # a count past the int32 range wraps negative in the traced needs
                raise RuntimeError(
                    flat_walk_list_limit_message(
                        kind, f"overflowed {cap} directed entries", limit=limit
                    )
                )
            grown = _tight_list_capacity(
                needed, headroom=headroom, floor=cap, limit=limit
            )
            if ovf and lower:
                grown = max(grown, min(2 * cap, limit))
            validated[key] = int(grown)
        peak = max(int(validated.get("peak_wavefront") or 0), n["peak_wavefront"])
        if n["queue_overflow"]:
            traced = getattr(self, "_strict_fused_traced_caps", None) or {}
            # one past the traced queue: the handoff's pow2(1.5 x peak) doubles it
            peak = max(peak, int(traced.get("queue_capacity") or 0) + 1)
        validated["peak_wavefront"] = int(peak)
        self._strict_fused_validated_caps = validated
        return True

    def _walk_overflowed_a_named_cap(self, needs: np.ndarray) -> bool:
        """Whether a list a failed segment's walk overflowed was NAMED by the caller.

        A named cap is a deliberate memory bound and never widened, so no re-plan
        makes such a segment fit: the retry declines at once rather than re-prepare
        and recompile for a re-run that must fail too.

        Parameters
        ----------
        needs : np.ndarray
            The segment's running maximum of
            :meth:`jaccpot.runtime.fused_topology.FusedTopology.needs`.

        Returns
        -------
        bool
            ``True`` when the far list overflowed a named far cap or the near list
            a named near cap.
        """
        from jaccpot.runtime.capacity_guard import WALK_NEEDS_FIELDS

        n = dict(zip(WALK_NEEDS_FIELDS, (int(v) for v in needs)))
        validated = getattr(self, "_strict_fused_validated_caps", None) or {}
        return bool(
            (n["far_overflow"] and validated.get("far_named"))
            or (n["near_overflow"] and validated.get("near_edge_named"))
        )

    def _replan_tree_from_needs(self, needs: np.ndarray) -> bool:
        """Grow the cell-leaf tree's bounds to what a failed segment's trees needed.

        The TREE half of the segment retry (the walk's is
        :meth:`_replan_walk_caps_from_needs`). A rebuild's tree overflows on one of
        three bounds; each grows from the segment's running maximum of
        :meth:`jaccpot.runtime.fused_topology.FusedTopology.needs`:

        * the leaf capacity to ``auto_leaf_capacity(leaf_needed)`` when the
          positions needed more cell leaves than it holds (sticky: later prepares
          on this engine use it);
        * the level-loop depth bound to ``max(bound, depth_needed + 8)``;
        * the level loops' batch width -- registered per tree shape, so the caller
          widens it after the re-prepare, for the shape that prepare built.

        Every bound only grows. The topology-reuse entry and the prepared-state
        slot are cleared (they hold a tree of the old capacity), and so are the
        engine's compiled runners: a program reads the level loops' batch width
        from the registry when it traces, and that width is not in its key.

        Parameters
        ----------
        needs : np.ndarray
            The segment's running maximum of ``FusedTopology.needs``
            (``fused_topology.NEEDS_FIELDS``).

        Returns
        -------
        bool
            Whether a bound grew. ``False`` -- the caller raises -- when no tree
            overflowed, the leaves are buckets, an installed capacity plan fixes
            the bounds (multi-GPU), or every bound already covers the needs.
        """
        from jaccpot.runtime._level_shapes import registered_level_batch_width
        from jaccpot.runtime.capacity_plan import fused_capacity_plan
        from jaccpot.runtime.fmm_sweeps import _CELLS_DEPTH_HEADROOM
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS

        n = dict(zip(NEEDS_FIELDS, (int(v) for v in needs)))
        if not n.get("tree_overflow"):
            return False
        if getattr(self, "_tree_leaf_partition", "buckets") != "cells":
            return False
        if fused_capacity_plan() is not None:
            return False
        capacity = int(getattr(self, "_tree_leaf_capacity", None) or 0)
        bound = getattr(self, "_cells_upward_num_levels", None)
        # the bounds the failed programs checked: the template's capacity (a cell
        # tree of C leaves has 2C - 1 nodes), the stashed depth bound, and the
        # batch width registered for that shape
        width = registered_level_batch_width(
            total_nodes=2 * capacity - 1, num_internal=capacity - 1
        )
        leaf_over = n["leaf_needed"] > capacity
        depth_over = bound is not None and n["depth_needed"] > int(bound)
        width_over = width is not None and n["widest_needed"] > int(width)
        if not (leaf_over or depth_over or width_over):
            return False
        if leaf_over:
            # the configured headroom (TreeConfig.leaf_headroom under
            # leaf_capacity="auto"), the bench's 1.15 for an explicit capacity
            headroom = getattr(self, "_tree_leaf_capacity_auto", None) or 1.15
            self._tree_leaf_capacity = max(
                capacity, auto_leaf_capacity(n["leaf_needed"], float(headroom))
            )
        # the depth bound covers the deepest tree too, whichever bound overflowed:
        # the re-prepare only raises it to the START's depth + 8
        self._cells_upward_num_levels = max(
            int(bound or 0), min(64, n["depth_needed"] + _CELLS_DEPTH_HEADROOM)
        )
        self.clear_prepared_state_cache()
        self._strict_fused_jit_function_cache = {}
        return True

    def _segment_retry_mode(
        self, needs: np.ndarray, retries: int, tree_first: bool
    ) -> Optional[str]:
        """How a failed segment may be retried, or ``None`` when it must raise.

        At most two retries per segment. The first re-plans whatever overflowed
        (walk caps and/or tree bounds). A second is allowed only after a first
        that re-planned the TREE, and only for the walk: the walk's needs in a
        segment whose tree overflowed were measured on that broken tree (a
        truncated leaf set, or level loops that dropped nodes) and can fall short,
        while the re-run's, on a tree that fits, are the walk's real needs.

        Parameters
        ----------
        needs : np.ndarray
            The failed run's running maximum of
            :meth:`jaccpot.runtime.fused_topology.FusedTopology.needs`.
        retries : int
            Retries this segment has run so far.
        tree_first : bool
            The first failure was a tree overflow (so the first retry re-planned
            the tree).

        Returns
        -------
        Optional[str]
            ``"full"`` (walk and tree), ``"walk"`` (walk only), or ``None``.
        """
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS

        if not env_flag("JACCPOT_STRICT_SEGMENT_RETRY", True):
            return None
        if retries == 0:
            return "full"
        tree_now = int(needs[NEEDS_FIELDS.index("tree_overflow")])
        if retries == 1 and tree_first and not tree_now:
            return "walk"
        return None

    def _raise_scan_capacity_saturated(
        self, needs: Optional[np.ndarray], *, retries: int = 0
    ) -> None:
        """Raise the fused scan's capacity error (after any retry was spent).

        Parameters
        ----------
        needs : Optional[np.ndarray]
            The segment's needs (``fused_topology.NEEDS_FIELDS``), named in the
            message when given; a cell-leaf tree overflow among them is named
            first, with what overflowed and its remedy, and the walk's saturation
            after it.
        retries : int
            Retries the segment ran before this failure (each a re-plan and a
            re-run from its start); the bounds named are then the grown ones.

        Raises
        ------
        RuntimeError
            Always.
        """
        from jaccpot.runtime._level_shapes import registered_level_batch_width
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS

        max_blocks = os.environ.get(
            "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
            "32",
        )
        traced_caps = getattr(self, "_strict_fused_traced_caps", None) or {}
        walk = (
            ""
            if needs is None
            else " The segment's walks needed: "
            + ", ".join(f"{k}={int(v)}" for k, v in zip(NEEDS_FIELDS, needs))
            + "."
        )
        named = (
            dict(zip(NEEDS_FIELDS, (int(v) for v in needs)))
            if needs is not None
            else {}
        )
        ran = (
            f" Segment retries run: {int(retries)} (at most two per segment, "
            "the second only for the walk after a tree re-plan; "
            "JACCPOT_STRICT_SEGMENT_RETRY=0 turns them off)."
        )
        saturated = (
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
                "unnamed caps are re-planned once per segment (twice after a tree "
                "re-plan; JACCPOT_STRICT_SEGMENT_RETRY=0 turns that off), named "
                "ones never."
                if traced_caps.get("flat_walk")
                else ""
            )
        )
        if not named.get("tree_overflow"):
            raise RuntimeError(saturated + ran + walk)
        # the cell-leaf tree first -- what overflowed, against which bound -- then
        # the walk's saturation when its lists saturated in the same segment
        walk_saturated = any(
            named.get(k) for k in ("far_overflow", "near_overflow", "queue_overflow")
        )
        capacity = getattr(self, "_tree_leaf_capacity", None)
        bound = getattr(self, "_cells_upward_num_levels", None)
        width = (
            None
            if capacity is None
            else registered_level_batch_width(
                total_nodes=2 * int(capacity) - 1, num_internal=int(capacity) - 1
            )
        )
        failed = []
        if named.get("leaf_needed", 0) > int(capacity or 0):
            failed.append(
                f"{named['leaf_needed']} cell leaves, more than "
                f"TreeConfig(leaf_capacity={capacity}) holds"
            )
        if bound is not None and named.get("depth_needed", 0) > int(bound):
            failed.append(
                f"a tree {named['depth_needed']} levels deep, past the level-loop "
                f"bound ({bound})"
            )
        if width is not None and named.get("widest_needed", 0) > int(width):
            failed.append(
                f"a level of {named['widest_needed']} nodes, wider than the level "
                f"loops' batch ({width})"
            )
        if not failed:
            failed.append(
                "more than its bounds hold (leaf capacity "
                f"{capacity}, level-loop bound {bound}, level batch {width}; an "
                "installed capacity plan sets them on the multi-GPU lane)"
            )
        if retries:
            replan = (
                " The segment was re-planned at its start (the leaf capacity, depth "
                "bound and level batch grown to what its trees needed) and the "
                "re-run overflowed again: the bounds named are the grown ones."
            )
        elif not env_flag("JACCPOT_STRICT_SEGMENT_RETRY", True):
            replan = (
                " The segment retry is off (JACCPOT_STRICT_SEGMENT_RETRY=0), which "
                "would re-plan the tree once at the segment's start."
            )
        else:
            replan = " It could not be re-planned."
        raise RuntimeError(
            "the cell-leaf tree overflowed inside the compiled velocity-Verlet "
            "scan: a step's rebuild needed "
            + "; ".join(failed)
            + ", so the forces from that step on are wrong."
            + replan
            + " Raise TreeConfig(leaf_capacity=...), with headroom over the live "
            "cell-leaf count of every configuration the run reaches (it grows as "
            "the particles cluster), or run shorter segments."
            + (" In the same segment, " + saturated if walk_saturated else "")
            + ran
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
            Use the fused device-resident layout; ``None`` means ``False``. (It
            used to take the engine's ``_strict_fused_mode_active``, so the same
            call built a different layout after any ``strict_run_v2``.)

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
                fused_device_mode=bool(fused_device_mode),
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
                fused_device_mode=bool(fused_device_mode),
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

    def strict_run_v2(
        self,
        *,
        state: Array,
        masses: Array,
        dt: Union[float, Array],
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
        force_params: Optional[ForceParams] = None,
        external_params: Any = None,
        carry_total_acceleration: bool = False,
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

        Endpoint-correct velocity Verlet, run device-resident. The integrator is
        fixed and the state is a raw array, which is what lets the whole loop
        live inside one scan.

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
        dt : Union[float, Array]
            Timestep. A Python number is a compile-time literal of the scan (the
            default); an array is TRACED: an argument of the compiled scan, cast to
            the state's dtype, so one program serves every value.
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
            ``state -> accelerations``: called with the packed state ``[N, 2, 3]``
            (positions ``state[:, 0]``), not with the positions alone, and returns
            ``[N, 3]``. Traced into the scan, so it must be jittable. It should
            depend on the positions only: at a step's endpoint the velocity half
            still holds the step's starting velocities. The rows come in the
            caller's order, except under ``JACCPOT_STRICT_CARRY_ORDER=tree`` (see
            ``carry``).
        rematerialize_between_refresh : bool
            Rebuild device arrays at refresh boundaries.
        return_history : bool
            Return every step's state rather than only the last.
        return_prepared_state : bool
            Return the prepared state so the next call can reuse it. On the
            static-radix lane it carries no far-pair list
            (``compact_far_pairs=None``): the scan builds one per step and keeps
            none, and the one the call started from belongs to the start
            positions. To differentiate at the returned positions,
            ``prepare_state`` there first; ``differentiable_accelerations``
            refuses the returned state. Under ``mac_type='dehnen_error'`` it also
            holds the scan's last f_b, and a call continuing from it starts its
            thresholds there (as the particle carry's handle does), so calls of
            1 and 2 steps are one call of 3.
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
            :mod:`jaccpot.runtime.strict_carry`; cell leaves only, see
            :func:`~jaccpot.runtime.fused_provider.require_cell_leaves`). ``None``
            reads ``JACCPOT_STRICT_CARRY`` (default ``"state"``). With ``"particles"``
            the returned prepared state is a
            :class:`~jaccpot.runtime.strict_carry.StrictParticleCarry` handle that
            only the next ``strict_run_v2(carry="particles")`` call accepts; it
            carries the self-gravity at the returned positions, so that call needs
            neither a prepare nor an initial force evaluation. A call without one
            prepares caps-only (``FusedForceProvider.prepare(caps_only=True)``: the
            template, the walk's caps, the verdict and the f_b seed, no list and no
            downward pass; its start program evaluates the force), as does its
            segment retry; ``JACCPOT_STRICT_CAPS_ONLY_PREPARE=0`` prepares in full.
            ``JACCPOT_STRICT_CARRY_ORDER=tree`` (particles only; default
            ``input``) carries the rows in the previous step's Morton order: no
            per-step gather of the force back into input order and no inverse
            permutation, and the tree's gathers read an almost sorted array; the
            state is put back into input order once per call. The same values
            (bitwise unless two particles share a Morton code). It needs an
            external field that acts row by row, and it is not used with
            ``return_history`` or a ``step_callback``.
        donate_state : bool
            ``carry="particles"`` only; off by default, so the input state is kept.
            When set, ``state``'s buffer is handed to the compiled scan, which
            writes the returned state into it instead of allocating a second copy
            (24 B per particle at the scan's peak). The passed ``state`` is
            CONSUMED -- its array is deleted -- so pass the returned state to the
            next call. A capacity retry re-runs the segment from its start, so the
            call keeps a host copy of the start state (one device-to-host copy
            per call, only when donating).
        force_params : Optional[ForceParams]
            ``G`` and the softening squared of the self-gravity
            (:class:`~jaccpot.runtime.fused_topology.ForceParams`); ``None`` takes
            the solver's. Python floats are compile-time literals (the solver's own
            values give the default program); arrays are TRACED, cast to the
            working dtype and passed to the compiled scan after its other
            arguments: one compile for every value. The walk is the solver's
            either way: under a compact softening kernel an ``eps2`` beyond the
            softening its separation floor covers is refused (``OVERFLOW_PARAMS``,
            :meth:`FusedForceProvider.params_overflow`), and
            ``mac_type='dehnen_error'`` takes the solver's own only.
        external_params : Any
            Parameters of the external field (any pytree). Given, the field is
            called as ``external_acceleration_fn(state, external_params)`` -- the
            arrays among them traced like ``dt``, the Python numbers literals --
            instead of ``external_acceleration_fn(state)``. Needs ``add_external``
            and ``external_acceleration_fn``.
        carry_total_acceleration : bool
            ``carry="particles"`` only; off by default. The returned handle also
            carries the scan's TOTAL acceleration at the returned state
            (``StrictParticleCarry.acceleration``: the self-gravity plus the
            external field, as the last step computed it) and the field's identity
            (:func:`~jaccpot.runtime.strict_carry.external_field_key`). A call
            continuing from such a handle with the same field, field parameters
            and force parameters starts its scan from that total and CONSUMES it
            (its buffer is donated to the scan, as a start force this call
            evaluated itself is); otherwise the call evaluates the field again at
            its start positions, eagerly, outside the scan -- which can differ
            from the scan's fused evaluation in the last bit (CPU, N = 512, the
            pure-JAX near field: 1 of 3072 components, 1.9e-6, 3 x 2 chained
            steps against 6). Chained calls with it are one call of all the steps,
            bit for bit, and compile one scan program. The production runner
            (:class:`jaccpot.production.ProductionRunner`) passes it.

        Returns
        -------
        tuple[Array, Optional[Any], Optional[Array]]
            ``(final_state, prepared_state, history)``. The second is ``None``
            unless ``return_prepared_state``, the third ``None`` unless
            ``return_history``.

        Raises
        ------
        RuntimeError
            If the profile is not large-N production; if the particle count is not
            in ``JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET``; or if a refresh hits a
            topology/profile mismatch. A failed fused scan raises its own error,
            unwrapped.
        ValueError
            If ``num_steps`` is not positive, or ``refresh_every`` is not 1; if
            ``carry="particles"`` is asked of a lane that cannot rebuild the state
            from its shapes (bucket leaves, or a far list read from the carry), or
            a :class:`StrictParticleCarry` comes without it; if
            ``donate_state`` is asked without ``carry="particles"``; if
            ``force_params`` are refused (``mac_type='dehnen_error'``) or beyond
            the walk's softening; or if ``external_params`` come without an
            external field.
        """
        from jaccpot.runtime.strict_carry import (
            StrictParticleCarry,
            external_field_key,
        )

        state_arr = jnp.asarray(state)
        masses_arr = jnp.asarray(masses)
        # A Python timestep is a closure constant of the scan (a literal of its
        # program); an array is one of its arguments (``step_params`` below), cast
        # to the state's dtype and never closed over
        dt_static = _is_static(dt)
        dt_arr = jnp.asarray(float(dt), dtype=state_arr.dtype) if dt_static else None
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
        if carry_total_acceleration and carry_mode != "particles":
            raise ValueError(
                "carry_total_acceleration needs carry='particles' (its handle "
                "carries the total acceleration)"
            )
        if external_params is not None and not (
            bool(add_external) and external_acceleration_fn is not None
        ):
            raise ValueError(
                "external_params needs add_external=True and an "
                "external_acceleration_fn(state, external_params)"
            )
        if carry_mode == "particles":
            # before any work: the particle carry rebuilds from shapes alone
            require_cell_leaves(self)
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
            f"dt={f'{float(dt):.12g}' if dt_static else 'traced'}|"
            f"external={int(bool(add_external))}|"
            f"theta={float(self.theta if theta is None else theta):.12g}"
        )
        if profile_key in self._strict_v2_seen_profile_keys:
            self._strict_v2_profile_key_hits += 1
        else:
            self._strict_v2_profile_key_misses += 1
            self._strict_v2_compile_count += 1
            self._strict_v2_seen_profile_keys.add(profile_key)
        self._strict_v2_execute_count += 1

        # The fused scan is the strict lane's only loop: the host-driven one
        # (`JACCPOT_STATIC_STRICT_FUSED_MODE=off`) went in the 2026-10 cleanup (X6).
        # The profile set only restricts it to the particle counts it names.
        # The flag is a DIAGNOSTIC ("a strict call engaged the fused lane"): no
        # later call reads it to choose numerics. The MAC geometry is the tree's
        # (`walk_geometry_default`), and the refreshes below name their lane.
        self._strict_fused_mode_active = self._strict_fused_profile_allows_n(
            int(state_arr.shape[0])
        )
        if not self._strict_fused_mode_active:
            self._strict_fused_fallback_count += 1
            self._strict_fused_last_fallback_reason = (
                "particle_count_not_in_JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET"
            )
            # Refuse rather than run anything else, so the profile set is fixed
            # (or cleared to allow all N).
            raise RuntimeError(
                "strict_run_v2: particle count "
                f"N={int(state_arr.shape[0])} is not in "
                "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET="
                f"{os.environ.get('JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET', '')!r}, "
                "and the fused scan is the strict lane's only loop. Add this N to "
                "the profile set, or leave it empty to allow all N."
            )
        if profile_key in self._strict_fused_seen_profile_keys:
            self._strict_fused_profile_key_hits += 1
        else:
            self._strict_fused_profile_key_misses += 1
            self._strict_fused_compile_count += 1
            self._strict_fused_seen_profile_keys.add(profile_key)
        self._strict_fused_execute_count += 1
        self._strict_fused_device_refresh_route_count += num_steps_i

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
        # mac_type='dehnen_error' on the fused lane: each step's walk takes eq
        # (16a)'s thresholds from eq (16b)'s f_b of the previous step's evaluation,
        # which the scan carries (input order). ``None`` in every other run, so the
        # carries hold no extra array.
        scale_carry = bool(self._flat_walk_criterion_active())
        if scale_carry and (not self_eval_active or eval_diag_mode != "full"):
            raise ValueError(
                "mac_type='dehnen_error' on the fused lane needs the full self-force "
                "evaluation: its thresholds come from that evaluation's force scale"
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

        # The step's parameters. ``None`` (Python dt, the solver's own G and
        # softening, a one-argument external field) keeps every program as it
        # was: literals, no extra argument. Otherwise they ride in one
        # ``StepParams`` argument after the others -- Python numbers as aux data
        # (literals, part of the compile key), arrays as its traced leaves.
        force_resolved: Optional[ForceParams] = None
        if force_params is not None:
            # checked before any work: refused under the criterion, or beyond the
            # softening the walk is built for
            force_resolved = FusedForceProvider(
                self, leaf_size=int(leaf_size), max_order=int(max_order), theta=theta
            ).resolve_params(force_params)
        step_params: Optional[StepParams] = None
        if force_resolved is not None or not dt_static or external_params is not None:
            step_params = StepParams(
                force=(
                    ForceParams(G=float(self.G), eps2=float(self.softening) ** 2)
                    if force_resolved is None
                    else force_resolved
                ),
                dt=float(dt) if dt_static else jnp.asarray(dt, state_arr.dtype),
                external=external_params,
            )
        # the integrator's part of the compile keys: the literal dt, or the
        # parameters' structure (their literals, not their traced values)
        dt_key: Any = (
            float(dt)
            if step_params is None
            else (
                jax.tree_util.tree_structure(step_params),
                tuple(
                    (jnp.shape(x), jnp.result_type(x).name)
                    for x in jax.tree_util.tree_leaves(step_params)
                ),
            )
        )
        params_args: tuple = () if step_params is None else (step_params,)
        two_arg_external = external_params is not None
        # what a particle carry's handle records its self-gravity was computed with
        force_key: Optional[tuple] = (
            None
            if force_resolved is None
            else (
                float(np.asarray(force_resolved.G)),
                float(np.asarray(force_resolved.eps2)),
            )
        )

        def _dt(params: Optional[StepParams]) -> Array:
            # the step's timestep: the closure literal, or the traced argument
            if params is None or _is_static(params.dt):
                assert dt_arr is not None
                return dt_arr
            return params.dt

        def _external(state_rows: Array, params: Optional[StepParams]) -> Array:
            assert external_acceleration_fn is not None
            if two_arg_external:
                assert params is not None
                return external_acceleration_fn(state_rows, params.external)
            return external_acceleration_fn(state_rows)

        def _force(params: Optional[StepParams]) -> Optional[ForceParams]:
            return None if params is None else params.force

        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(state_arr.shape[0])
        )
        prepared_curr = prepared_state
        # a state built here is held by nothing else, so the scan may consume it
        donate_carry = bool(donate_prepared_state) or prepared_state is None
        particle_carry = carry_mode == "particles"
        # The step's force: every refresh, evaluation, verdict, compile key and
        # re-plan below goes through the provider. Its template comes from the
        # prepared state (or the handle's); the state carry rebuilds from the
        # state it carries instead.
        prov: FusedForceProvider
        # The particle carry's bootstrap and segment retry prepare caps-only: the
        # template's shapes, the trace-time engine state, the verdict and the f_b
        # seed, no list and no downward pass (its start program evaluates the
        # force). The same template and programs as the full prepare.
        caps_only = particle_carry and env_flag(
            "JACCPOT_STRICT_CAPS_ONLY_PREPARE", True
        )
        if prepared_curr is None and handle_in is None:
            prov, topology_start = FusedForceProvider.prepare(
                self,
                state_arr[:, 0, :],
                masses_arr,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                runtime_overrides=runtime_overrides,
                caps_only=caps_only,
            )
            prepared_curr = topology_start.prepared
            del topology_start
        elif handle_in is not None:
            # this call's configuration over the template the handle carries
            prov = FusedForceProvider(
                self,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                runtime_overrides=runtime_overrides,
                template=handle_in.template,
            )
        else:
            if not isinstance(prepared_curr, LargeNPreparedState):
                self._strict_runner_fail_fast_reject_count += 1
                raise RuntimeError(
                    "strict fused velocity-Verlet requires LargeNPreparedState input."
                )
            self._record_large_n_eval_shape_diagnostics(prepared_curr)
            prov = FusedForceProvider.freeze_template(
                self,
                prepared_curr,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                runtime_overrides=runtime_overrides,
            )

        if force_resolved is not None and int(prov.params_overflow(force_resolved)):
            raise ValueError(
                f"force_params eps2={force_resolved.eps2} is beyond the softening the "
                f"walk is built for (eps2 <= {prov.walk_eps2()}, the "
                f"{getattr(self, 'softening_kernel', 'plummer')} kernel's support at "
                "the solver's separation floor): the far field would take pairs "
                "inside the kernel's support unsoftened (OVERFLOW_PARAMS). Build the "
                "solver with the largest softening the run uses."
            )

        def _evaluate_self(
            prepared_in: PreparedStateLike,
            state_in: Array,
            params: Optional[StepParams] = None,
        ) -> Array:
            if not self_eval_active:
                return jnp.zeros_like(state_in[:, 0, :])
            if eval_diag_mode == "permutation_only":
                return jnp.asarray(prepared_in.positions_sorted)[
                    prepared_in.inverse_permutation
                ] * jnp.asarray(0.0, dtype=state_in.dtype)
            # the provider states the large-N requirement where it relies on it
            return jnp.asarray(
                prov.accelerations(
                    FusedTopology.from_prepared(prepared_in), _force(params)
                ),
                dtype=state_in.dtype,
            )

        def _evaluate_self_scaled(
            prepared_in: PreparedStateLike,
            state_in: Array,
            params: Optional[StepParams] = None,
        ) -> tuple[Array, Optional[Array]]:
            # `_evaluate_self`, plus eq (16b)'s f_b (input order) when the scan
            # carries it; the same single evaluation, the force bitwise unchanged
            if not scale_carry:
                return _evaluate_self(prepared_in, state_in, params), None
            acc, scale = prov.accelerations(
                FusedTopology.from_prepared(prepared_in),
                _force(params),
                force_scale=True,
            )
            return (
                jnp.asarray(acc, dtype=state_in.dtype),
                jnp.asarray(scale, dtype=state_in.dtype),
            )

        def _force_scale_seed(prepared_in: Any) -> Optional[Array]:
            # the first refresh's f_b: the eager prepare's eq (16b) prepass (its own
            # tree's sorted order) back in input order, else one evaluation
            if not scale_carry or prepared_in is None:
                return None
            fb_sorted = getattr(self, "_last_force_scale_particles", None)
            perm = jnp.asarray(prepared_in.tree.particle_indices, dtype=INDEX_DTYPE)
            if fb_sorted is not None and int(fb_sorted.shape[0]) == int(perm.shape[0]):
                fb_sorted = jnp.asarray(fb_sorted, dtype=state_arr.dtype)
                return jnp.zeros_like(fb_sorted).at[perm].set(fb_sorted)
            return _evaluate_self_scaled(prepared_in, state_arr)[1]

        acceleration_self_current: Optional[Array]
        if not self_eval_active:
            acceleration_self_current = jnp.zeros_like(state_arr[:, 0, :])
        elif (
            initial_self_acceleration is None
            and handle_in is not None
            and handle_in.force_key == force_key
        ):
            acceleration_self_current = jnp.asarray(
                handle_in.self_acceleration, dtype=state_arr.dtype
            )
        elif initial_self_acceleration is None and particle_carry:
            # the particle scan evaluates it itself, as its first refresh, once
            # the prepared state is freed: bitwise the eager evaluation, whose
            # temporary block next to the whole prepared state was the first
            # allocation to fail from 1.36e8 particles on (a fragmented arena)
            acceleration_self_current = None
        elif initial_self_acceleration is None:
            acceleration_self_current = _evaluate_self(
                prepared_curr, state_arr, step_params
            )
        else:
            acceleration_self_current = jnp.asarray(
                initial_self_acceleration, dtype=state_arr.dtype
            )
        if particle_carry:
            # the particle run builds its own from the self-gravity, per attempt:
            # a total built here would only sit in the arena through the scan
            acceleration_current = None
        elif add_external and external_acceleration_fn is not None:
            acceleration_current = acceleration_self_current + jnp.asarray(
                _external(state_arr, step_params), dtype=state_arr.dtype
            )
        else:
            acceleration_current = acceleration_self_current

        def _refresh_and_evaluate_endpoint(
            prepared_in: PreparedStateLike,
            state_position: Array,
            masses_in: Array,
            scale_in: Optional[Array] = None,
            params: Optional[StepParams] = None,
        ) -> tuple[FusedTopology, Array, Optional[Array]]:
            # the step's topology at the drifted positions (rebuilt from
            # ``prepared_in``'s shapes; the state carry passes its carry, the
            # particle carry a materialised template) and its self-force
            if diag_mode in {"integrator_only", "eval_only"}:
                # no refresh in this trace: the capacity side channels must not
                # carry a previous trace's verdict into this one, and the verdict
                # folds an empty one (``fresh`` without ``refresh_ok``)
                self._last_refresh_capacity_ok = None
                self._last_refresh_walk_needs = None
                topology = FusedTopology(prepared=prepared_in, fresh=True)
            else:
                topology = prov._rebuild_like(
                    prepared_in,
                    state_position[:, 0, :],
                    masses_in,
                    force_scale=scale_in if scale_carry else None,
                )
            acc_self, scale_new = _evaluate_self_scaled(
                topology.prepared, state_position, params
            )
            return topology, acc_self, scale_new

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
            scale_now: Optional[Array] = None,
            params: Optional[StepParams] = None,
        ) -> tuple[FusedTopology, Array, Array, Array, Optional[Array]]:
            # one velocity-Verlet step of the fused scan: drift, refresh + self
            # force at the new positions, kick; shared by both carries. The
            # step's verdict is the caller's, from the returned topology, after
            # the kick (where the scan has always folded it)
            dt_now = _dt(params)
            position_new = (
                state_now[:, 0]
                + state_now[:, 1] * dt_now
                + 0.5 * acceleration_now * dt_now**2
            )
            state_position = state_now.at[:, 0].set(position_new)
            topology_new, acceleration_self_new, scale_new = (
                _refresh_and_evaluate_endpoint(
                    prepared_now, state_position, masses_in, scale_now, params
                )
            )
            if add_external and external_acceleration_fn is not None:
                acceleration_new = acceleration_self_new + jnp.asarray(
                    _external(state_position, params),
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
                dt_now,
            )
            if rematerialize_between_refresh:
                state_new = jnp.asarray(state_new, dtype=state_now.dtype)
            if emit and step_callback is not None:
                _emit_step(scan_x, state_new, jnp.asarray(True))
            return (
                topology_new,
                state_new,
                acceleration_new,
                acceleration_self_new,
                scale_new,
            )

        # The particle carry in TREE order (``JACCPOT_STRICT_CARRY_ORDER=tree``): the
        # scan carries the particles in the previous step's Morton order, with their
        # masses and input indices, so the force is never gathered back into input
        # order and the tree's gathers read an almost sorted array (the permutation
        # from one step's order to the next is close to the identity). The state
        # goes back to input order once, when the call returns. Only where nothing
        # sees a step's state from outside (no history, no step callback); the
        # external field must act row by row (it sees the rows in tree order).
        carry_order = env_choice(
            "JACCPOT_STRICT_CARRY_ORDER", "input", ("input", "tree")
        )
        tree_order = (
            particle_carry
            and carry_order == "tree"
            and self_eval_active
            and diag_mode not in {"integrator_only", "eval_only"}
            and eval_diag_mode == "full"
            and not return_history
            and step_callback is None
            # the force-scale carry rides the input order only
            and not scale_carry
        )

        def _advance_tree_order(
            prepared_now: PreparedStateLike,
            state_now: Array,
            acceleration_now: Array,
            masses_now: Array,
            ids_now: Array,
            params: Optional[StepParams] = None,
        ) -> tuple[FusedTopology, Array, Array, Array, Array, Array]:
            # `_advance` with the rows in the carry's order; returns the state, the
            # accelerations, masses and input indices in the NEW tree order
            dt_now = _dt(params)
            position_new = (
                state_now[:, 0]
                + state_now[:, 1] * dt_now
                + 0.5 * acceleration_now * dt_now**2
            )
            state_position = state_now.at[:, 0].set(position_new)
            topology_new = prov._rebuild_like(
                prepared_now, state_position[:, 0, :], masses_now
            )
            prepared_new = topology_new.prepared
            # row j of the tree is row perm[j] of the carry
            perm = jnp.asarray(
                prepared_new.tree.topology.particle_indices, dtype=INDEX_DTYPE
            )
            acceleration_self_new = jnp.asarray(
                prov.accelerations(topology_new, _force(params), order="tree"),
                dtype=state_now.dtype,
            )
            # The tree's own sorted positions and masses are this very gather
            # (``x[perm]``), already alive through the near field: reused, they
            # cost no second copy. Only in the force's dtype, though: a state or
            # masses in another one (a float64 state under the float32 force) are
            # gathered from the carry, or every step would round them to the
            # force's dtype. Velocities, the old acceleration and the input
            # indices follow the permutation here.
            positions_sorted = prepared_new.positions_sorted
            if positions_sorted.dtype != state_now.dtype:
                positions_sorted = state_position[:, 0][perm]
            masses_sorted = prepared_new.masses_sorted
            if masses_sorted.dtype != masses_now.dtype:
                masses_sorted = masses_now[perm]
            state_sorted = jnp.stack(
                [
                    jnp.asarray(positions_sorted, state_now.dtype),
                    state_position[:, 1][perm],
                ],
                axis=1,
            )
            acceleration_prev = acceleration_now[perm]
            masses_new = jnp.asarray(masses_sorted, masses_now.dtype)
            ids_new = ids_now[perm]
            if add_external and external_acceleration_fn is not None:
                acceleration_new = acceleration_self_new + jnp.asarray(
                    _external(state_sorted, params),
                    dtype=state_now.dtype,
                )
            else:
                acceleration_new = acceleration_self_new
            state_new = _velocity_verlet_kick_drifted(
                state_sorted,
                acceleration_prev,
                acceleration_new,
                dt_now,
            )
            if rematerialize_between_refresh:
                state_new = jnp.asarray(state_new, dtype=state_now.dtype)
            return (
                topology_new,
                state_new,
                acceleration_new,
                acceleration_self_new,
                masses_new,
                ids_new,
            )

        if particle_carry:
            # carry_total_acceleration: the field's identity, for the returned
            # handle and to check the one passed in; and the scan's own total at
            # the start positions when that handle carries one of this very field,
            # its parameters and this call's force (see the parameter)
            external_key: Optional[tuple] = None
            start_total: Optional[Array] = None
            handle_total = getattr(handle_in, "acceleration", None)
            if carry_total_acceleration or handle_total is not None:
                external_on = (
                    bool(add_external) and external_acceleration_fn is not None
                )
                external_key = external_field_key(
                    external_acceleration_fn if external_on else None,
                    external_params if external_on else None,
                )
            if (
                handle_total is not None
                and initial_self_acceleration is None
                and self_eval_active
                and handle_in.force_key == force_key
                and handle_in.external_key == external_key
            ):
                start_total = handle_total
            # handed over in a box the callee empties: a local here would keep the
            # concrete state (and its far list) alive through the whole scan
            scale_seed = _force_scale_seed(prepared_curr) if handle_in is None else None
            prepared_box = [prepared_curr]
            prepared_curr = None
            state_curr, prepared_curr, history_out = self._strict_particle_carry_run(
                scale_seed=scale_seed,
                scale_carry=scale_carry,
                force_scale_seed=_force_scale_seed,
                prepared_box=prepared_box,
                handle_in=handle_in,
                state_arr=state_arr,
                masses_arr=masses_arr,
                acceleration_self_current=acceleration_self_current,
                advance=_advance,
                refresh_evaluate=_refresh_and_evaluate_endpoint,
                provider=prov,
                evaluate_self=_evaluate_self,
                advance_tree=_advance_tree_order if tree_order else None,
                num_steps_i=num_steps_i,
                # the integrator's part of the compile keys; the provider's
                # key() holds the force's
                cache_parts=(
                    dt_key,
                    bool(add_external),
                    (
                        id(external_acceleration_fn)
                        if external_acceleration_fn is not None
                        else 0
                    ),
                    bool(rematerialize_between_refresh),
                    bool(return_history),
                    id(step_callback) if step_callback is not None else 0,
                    int(step_callback_stride),
                ),
                return_history=bool(return_history),
                step_callback=step_callback,
                emit_step=_emit_step if step_callback is not None else None,
                initial_self_acceleration=initial_self_acceleration,
                add_external=bool(add_external),
                external_acceleration_fn=(
                    None
                    if external_acceleration_fn is None
                    else partial(_external, params=step_params)
                ),
                donate_state=bool(donate_state),
                step_params=step_params,
                force_key=force_key,
                start_total=start_total,
                return_total=bool(carry_total_acceleration),
                external_key=external_key,
                caps_only=caps_only,
            )
        else:
            from jaccpot.runtime.fused_topology import NEEDS_FIELDS

            def _compiled_runner_for(prepared_ref: PreparedStateLike) -> Callable:
                # The integrator's part of the key here, the force's from the
                # provider: leaf, order, theta, criterion, diagnostic modes, the
                # level-loop bound and the validated walk caps (the traced walk is
                # sized from them at TRACE time, so a re-plan must not reuse a
                # runner traced before it). ``like`` stashes the concrete tree
                # depth now, while the state is concrete, for the traced refresh's
                # M2M level loop, and keys it.
                cache_key = (
                    "strict_velocity_verlet",
                    tuple(int(v) for v in state_arr.shape),
                    str(state_arr.dtype),
                    tuple(int(v) for v in masses_arr.shape),
                    str(masses_arr.dtype),
                    dt_key,
                    num_steps_i,
                    bool(add_external),
                    (
                        id(external_acceleration_fn)
                        if external_acceleration_fn is not None
                        else 0
                    ),
                    bool(rematerialize_between_refresh),
                    bool(return_history),
                    id(step_callback) if step_callback is not None else 0,
                    int(step_callback_stride),
                    donate_carry,
                    *prov.key(like=prepared_ref),
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
                    scale_initial: Optional[Array] = None,
                    params_in: Optional[StepParams] = None,
                ) -> tuple[
                    tuple[
                        LargeNPreparedState, Array, Array, Array, Array, Optional[Array]
                    ],
                    Optional[Array],
                ]:
                    def _step(carry, scan_x):
                        (
                            prepared_now,
                            state_now,
                            acceleration_now,
                            capacity_ok_now,
                            walk_needs_now,
                            scale_now,
                        ) = carry
                        topology_new, state_new, acceleration_new, _, scale_new = (
                            _advance(
                                prepared_now,
                                state_now,
                                acceleration_now,
                                masses_in,
                                scan_x,
                                scale_now=scale_now,
                                params=params_in,
                            )
                        )
                        capacity_ok_new = capacity_ok_now & (
                            prov.capacity_ok(topology_new)
                        )
                        # what the walks of this segment needed, at most: a failed
                        # segment is re-planned from it
                        walk_needs_new = jnp.maximum(
                            walk_needs_now, topology_new.needs()
                        )
                        return (
                            topology_new.prepared,
                            state_new,
                            acceleration_new,
                            capacity_ok_new,
                            walk_needs_new,
                            scale_new,
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
                            prov.capacity_ok(
                                FusedTopology.from_prepared(prepared_initial)
                            ),
                            jnp.zeros((len(NEEDS_FIELDS),), jnp.int32),
                            scale_initial,
                        ),
                        xs=scan_xs,
                        length=num_steps_i,
                    )

                jit_cache[cache_key] = _compiled_runner
                self._strict_fused_jit_function_cache = jit_cache
                return _compiled_runner

            retries, tree_first = 0, False
            # A state an earlier call's scan returned holds that scan's own f_b
            # (set below): the next refresh's thresholds continue from it, as the
            # particle carry's handle does. The eager prepare's prepass is at the
            # EARLIER call's start positions, in its tree's order -- scattered
            # through this state's permutation it was another step's f_b of the
            # wrong particles, and 1 + 2 chained steps differed from 3 in one
            # call (1,349 of 3,072 state elements at N = 512, max 5.7e-5).
            carried_scale = (
                getattr(prepared_state, _CARRIED_FORCE_SCALE, None)
                if scale_carry
                else None
            )
            if carried_scale is not None and int(carried_scale.shape[0]) == int(
                state_arr.shape[0]
            ):
                scale_curr = jnp.asarray(carried_scale, dtype=state_arr.dtype)
            else:
                scale_curr = _force_scale_seed(prepared_curr)
            del carried_scale
            while True:
                # the far list is dead inside the scan (fresh rebuild): keep it
                # out of the carry, and off the returned state too -- it is the
                # START positions' list, and the gradient path would read it
                prepared_in = (
                    replace(prepared_curr, compact_far_pairs=None)
                    if self._strict_far_pairs_ride_outside_the_scan(prepared_curr)
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
                    _scale_out,
                ), history_out = compiled_runner(
                    prepared_in,
                    state_arr,
                    jnp.asarray(acceleration_current, dtype=state_arr.dtype),
                    masses_arr,
                    scale_curr,
                    *params_args,
                )
                del prepared_in
                self._strict_static_target_block_capacity_ok = bool(
                    np.asarray(jax.device_get(capacity_ok_all))
                )
                if self._strict_static_target_block_capacity_ok:
                    prepared_curr = prepared_out
                    state_curr = state_out
                    if scale_carry:
                        # the scan's f_b at the returned positions (input order),
                        # for a call that continues from this state; an attribute
                        # of this object, not a pytree field, so it lives and dies
                        # with the returned state and never enters a carry
                        object.__setattr__(
                            prepared_curr, _CARRIED_FORCE_SCALE, _scale_out
                        )
                    break
                needs = np.asarray(jax.device_get(walk_needs)).astype(np.int64)
                # Segment retry: a walk list or queue of the traced refresh
                # outgrew the caps the eager prepare sized, or the cell-leaf tree
                # its bounds. Re-plan them from what the segment needed,
                # re-prepare from the segment's START
                # (`FusedForceProvider.replanned`), recompile and run the segment
                # once more -- after a tree re-plan, a second time for the walk
                # (`_segment_retry_mode`). Anything else raises below.
                mode = self._segment_retry_mode(needs, retries, tree_first)
                if mode is None:
                    self._raise_scan_capacity_saturated(needs, retries=retries)
                if retries == 0:
                    tree_first = bool(needs[NEEDS_FIELDS.index("tree_overflow")])
                retries += 1
                del prepared_out, state_out, history_out
                replanned = prov.replanned(
                    needs, state_arr[:, 0, :], masses_arr, walk_only=mode == "walk"
                )
                if replanned is None:
                    self._raise_scan_capacity_saturated(needs, retries=retries - 1)
                prov, topology_start = replanned
                prepared_curr = topology_start.prepared
                del replanned, topology_start
                scale_curr = _force_scale_seed(prepared_curr)
                # the starting force from the state the segment now starts from
                # (the same field when the caller's state matched its positions)
                if initial_self_acceleration is None:
                    acceleration_current = _evaluate_self(
                        prepared_curr, state_arr, step_params
                    )
                    if add_external and external_acceleration_fn is not None:
                        acceleration_current = acceleration_current + jnp.asarray(
                            _external(state_arr, step_params),
                            dtype=state_arr.dtype,
                        )
                self._strict_fused_fallback_count += 1
                self._strict_fused_last_fallback_reason = "capacity_segment_retry"

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
        scale_seed: Optional[Array] = None,
        scale_carry: bool = False,
        force_scale_seed: Optional[Callable[[Any], Optional[Array]]] = None,
        prepared_box: list,
        handle_in: Optional[Any],
        state_arr: Array,
        masses_arr: Array,
        acceleration_self_current: Optional[Array],
        advance: Callable[..., Any],
        refresh_evaluate: Callable[..., Any],
        provider: FusedForceProvider,
        evaluate_self: Callable[..., Array],
        num_steps_i: int,
        cache_parts: tuple,
        return_history: bool,
        step_callback: Optional[Callable[[Array, Array], None]],
        emit_step: Optional[Callable[[Array, Array, Array], None]],
        initial_self_acceleration: Optional[Array],
        add_external: bool,
        external_acceleration_fn: Optional[Callable[[Array], Array]],
        donate_state: bool,
        advance_tree: Optional[Callable[..., Any]] = None,
        step_params: Optional[StepParams] = None,
        force_key: Optional[tuple] = None,
        start_total: Optional[Array] = None,
        return_total: bool = False,
        external_key: Optional[tuple] = None,
        caps_only: bool = False,
    ) -> tuple[Array, Any, Optional[Array]]:
        """The fused scan of ``strict_run_v2(carry="particles")``.

        The scan carries ``(state, acceleration, self acceleration, ok, walk
        needs, steps that fitted)``; each step materialises the prepared state
        from the provider's shape template (:mod:`jaccpot.runtime.strict_carry`)
        and refreshes it from the drifted positions, as the state carry does. The
        concrete prepared state (if any) is dropped before the scan, so it and the
        far list are freed.

        A segment whose refresh overflows a list is re-run once from its start
        with re-planned caps (``FusedForceProvider.replanned``; the stream stops at
        the failed step, so it never sees a failed step's state). ``donate_state``
        hands the start state to the scan, so a host copy of it is kept for that
        re-run.

        Parameters
        ----------
        scale_seed : Optional[Array]
            ``mac_type='dehnen_error'``: eq (16b)'s ``f_b`` per particle (input
            order) for the first refresh's thresholds -- the eager prepare's
            prepass; ``None`` takes the handle's.
        scale_carry : bool
            Carry ``f_b`` through the scan (the criterion is on). Static.
        force_scale_seed : Optional[Callable[[Any], Optional[Array]]]
            Builds a seed from a concrete prepared state (a segment retry's
            re-prepare).
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
            ``strict_run_v2``'s step body; returns the step's topology first.
        refresh_evaluate : Callable[..., Any]
            ``strict_run_v2``'s refresh + self-force at given positions (the
            step's own), for the scan's start force; returns the topology first.
        provider : FusedForceProvider
            The call's force: the template the steps materialise, the capacity
            verdict, the force's part of the compile keys, the segment re-plan. A
            re-plan replaces it with the re-prepared one, which the returned handle
            carries.
        evaluate_self : Callable[..., Array]
            ``strict_run_v2``'s self-force evaluation.
        num_steps_i : int
            Steps.
        cache_parts : tuple
            The integrator's static configuration (``dt``, the external field,
            history, the callback), part of the compile-cache key; the provider's
            ``key()`` holds the force's.
        return_history : bool
            Stack every step's state.
        step_callback : Optional[Callable[[Array, Array], None]]
            Streaming hook (feeds the step index).
        emit_step : Optional[Callable[[Array, Array, Array], None]]
            ``strict_run_v2``'s stride-gated emitter ``(step, state, fire)``: the
            scan fires it only for steps whose refresh fitted.
        initial_self_acceleration : Optional[Array]
            The caller's starting self-gravity, if given.
        add_external : bool
            Whether an external field is added.
        external_acceleration_fn : Optional[Callable[[Array], Array]]
            The external field at the start state (its parameters bound).
        donate_state : bool
            Hand ``state_arr``'s buffer to the scan (``strict_run_v2``'s
            ``donate_state``).
        advance_tree : Optional[Callable[..., Any]]
            ``strict_run_v2``'s tree-order step, or ``None``. Given, the scan
            carries the rows in tree order with their masses and input indices
            (``advance`` is then unused), and the final state and self-gravity
            are put back into input order once, after the scan.
        step_params : Optional[StepParams]
            ``strict_run_v2``'s traced step parameters: the last argument of the
            start program and of the scan, after the donated ones; ``None`` adds
            none.
        force_key : Optional[tuple]
            The call's force parameters as host floats (``None``: the solver's
            own), recorded on the returned handle.
        start_total : Optional[Array]
            The handle's total acceleration at the start state
            (``carry_total_acceleration``), already matched to this call's field
            and force: the first attempt's scan starts from it instead of the
            self-gravity plus an eager field evaluation, and consumes it. A retry
            builds the total again.
        return_total : bool
            ``carry_total_acceleration``: the returned handle carries the total
            acceleration at the final state (with a field, an extra output of the
            scan; without, the self-gravity) and ``external_key``.
        external_key : Optional[tuple]
            The field's identity (``external_field_key``), recorded on the handle
            when ``return_total``.
        caps_only : bool
            A segment retry re-prepares caps-only (``FusedForceProvider.replanned``):
            the start force a handle's call consumed comes from the start program
            then, as a first call's does, with the re-prepare's f_b seed kept.

        Returns
        -------
        tuple[Array, Any, Optional[Array]]
            ``(final_state, StrictParticleCarry, history)``. A lane that would read
            the carried state (far list not rebuilt fresh in the scan) raises
            ``ValueError``; a failed scan raises its own error, unwrapped, as the
            state carry does.

        Raises
        ------
        RuntimeError
            If the criterion is on and no force scale came from the eager prepare
            or the handle.
        """
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS
        from jaccpot.runtime.strict_carry import (
            PreparedCaps,
            StrictParticleCarry,
            host_copy,
        )

        external_active = bool(add_external) and external_acceleration_fn is not None
        # the scan's total acceleration as an output of its own: with a field only
        # (without one it is the self-gravity, which is returned anyway)
        total_out = bool(return_total) and external_active

        def _runner_for(
            prov_i: FusedForceProvider,
            n_steps: int,
            donate_state_i: bool,
            donate_acc_i: bool,
        ) -> Callable:
            # the provider's key holds the template's shapes and the level bound
            # (the cells lane's, stashed by the eager prepare)
            cache_key = (
                "strict_velocity_verlet_particles",
                tuple(int(v) for v in state_arr.shape),
                str(state_arr.dtype),
                tuple(int(v) for v in masses_arr.shape),
                str(masses_arr.dtype),
                int(n_steps),
                *cache_parts,
                bool(donate_state_i),
                bool(donate_acc_i),
                *prov_i.key(),
                # the tree-order flag stays last (tests read key[-1])
                total_out,
                advance_tree is not None,
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
                scale_initial: Optional[Array] = None,
                params_in: Optional[StepParams] = None,
            ) -> tuple[
                # (state, self-gravity, ok, needs, done, f_b[, total acceleration])
                tuple[Optional[Array], ...],
                Optional[Array],
            ]:
                def _step(carry, scan_x):
                    state_now, acc_now, _, ok_now, needs_now, done, scale_now = carry[
                        :7
                    ]
                    if advance_tree is not None:
                        (
                            topology_new,
                            state_new,
                            acc_new,
                            acc_self_new,
                            masses_new,
                            ids_new,
                        ) = advance_tree(
                            prov_i.like(),
                            state_now,
                            acc_now,
                            carry[7],
                            carry[8],
                            params=params_in,
                        )
                        tree_rows: tuple = (masses_new, ids_new)
                        scale_new = scale_now  # never both (see tree_order)
                    else:
                        topology_new, state_new, acc_new, acc_self_new, scale_new = (
                            advance(
                                prov_i.like(),
                                state_now,
                                acc_now,
                                masses_in,
                                scan_x,
                                emit=False,
                                scale_now=scale_now,
                                params=params_in,
                            )
                        )
                        tree_rows = ()
                    ok_new = ok_now & prov_i.capacity_ok(topology_new)
                    needs_new = jnp.maximum(needs_now, topology_new.needs())
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
                        scale_new,
                        *tree_rows,
                    ), (state_new if return_history else None)

                scan_xs = (
                    jnp.arange(n_steps, dtype=jnp.int32)
                    if emit_step is not None
                    else None
                )
                tree_rows0: tuple = ()
                if advance_tree is not None:
                    # the carry starts in input order: row i is particle i
                    tree_rows0 = (
                        masses_in,
                        jnp.arange(masses_in.shape[0], dtype=INDEX_DTYPE),
                    )
                carry_f, history = jax.lax.scan(
                    _step,
                    (
                        state_initial,
                        acceleration_initial,
                        acc_self_initial,
                        ok_initial,
                        needs_initial,
                        jnp.zeros((), jnp.int32),
                        scale_initial,
                        *tree_rows0,
                    ),
                    xs=scan_xs,
                    length=int(n_steps),
                )
                state_f, acc_f, acc_self_f, ok_f, needs_f, done_f, scale_f = carry_f[:7]
                if advance_tree is not None:
                    # back to input order, once: row j is particle ids[j]
                    ids_f = carry_f[8]
                    state_f = (
                        jnp.zeros_like(state_f)
                        .at[ids_f]
                        .set(state_f, unique_indices=True)
                    )
                    acc_self_f = (
                        jnp.zeros_like(acc_self_f)
                        .at[ids_f]
                        .set(acc_self_f, unique_indices=True)
                    )
                    if total_out:
                        acc_f = (
                            jnp.zeros_like(acc_f)
                            .at[ids_f]
                            .set(acc_f, unique_indices=True)
                        )
                outputs = (state_f, acc_self_f, ok_f, needs_f, done_f, scale_f)
                return outputs + ((acc_f,) if total_out else ()), history

            donate = ((0,) if donate_state_i else ()) + ((1,) if donate_acc_i else ())

            @partial(jax.jit, donate_argnums=donate)
            def _compiled_runner(
                state_initial: Array,
                acceleration_initial: Array,
                masses_in: Array,
                ok_initial: Array,
                scale_initial: Optional[Array] = None,
                params_in: Optional[StepParams] = None,
            ) -> tuple[
                # (state, self-gravity, ok, needs, done, f_b[, total acceleration])
                tuple[Optional[Array], ...],
                Optional[Array],
            ]:
                # the self-gravity slot is written by every step (num_steps >= 1)
                return _scan(
                    state_initial,
                    acceleration_initial,
                    jnp.zeros_like(acceleration_initial),
                    masses_in,
                    ok_initial,
                    jnp.zeros((len(NEEDS_FIELDS),), jnp.int32),
                    scale_initial,
                    params_in,
                )

            jit_cache[cache_key] = _compiled_runner
            self._strict_fused_jit_function_cache = jit_cache
            return _compiled_runner

        def _start_force_for(prov_i: FusedForceProvider) -> Callable:
            # The start positions' self-gravity, as the steps' own refresh: its
            # OWN small program, run after the prepared state is freed and before
            # the scan. Inside the scan's program XLA overlapped it with the first
            # step (a one-step scan is inlined), doubling the temporary block.
            cache_key = (
                "strict_particle_start_force",
                tuple(int(v) for v in state_arr.shape),
                str(state_arr.dtype),
                *cache_parts,
                *prov_i.key(),
            )
            jit_cache = getattr(self, "_strict_fused_jit_function_cache", {})
            fn = jit_cache.get(cache_key)
            if fn is not None:
                return fn
            self._strict_fused_traced_caps = None

            # named so the dumps' ``*_compiled_runner*`` selection includes it
            @jax.jit
            def _compiled_runner_start(
                state_initial: Array,
                masses_in: Array,
                ok_initial: Array,
                scale_initial: Optional[Array] = None,
                params_in: Optional[StepParams] = None,
            ) -> tuple[Array, Array, Array, Optional[Array]]:
                topology0, acc_self0, scale0 = refresh_evaluate(
                    prov_i.like(),
                    state_initial,
                    masses_in,
                    scale_initial,
                    params_in,
                )
                ok0 = ok_initial & prov_i.capacity_ok(topology0)
                needs0 = jnp.maximum(
                    jnp.zeros((len(NEEDS_FIELDS),), jnp.int32),
                    topology0.needs(),
                )
                return acc_self0, ok0, needs0, scale0

            jit_cache[cache_key] = _compiled_runner_start
            self._strict_fused_jit_function_cache = jit_cache
            return _compiled_runner_start

        params_args: tuple = () if step_params is None else (step_params,)
        # the self-gravity is this call's own (evaluated here, or zeros) unless it
        # came from the caller or from the handle: only then may it be donated.
        # A handle's total acceleration is the call's to consume (without a field
        # it IS the handle's self-gravity, then donated like an owned one)
        self_owned = (
            handle_in is None and initial_self_acceleration is None
        ) or start_total is not None

        def _initial_acceleration(acc_self: Array, state_now: Array) -> Array:
            acc_self = jnp.asarray(acc_self, dtype=state_now.dtype)
            if not external_active:
                return acc_self
            assert external_acceleration_fn is not None
            return acc_self + jnp.asarray(
                external_acceleration_fn(state_now), dtype=state_now.dtype
            )

        prepared_curr = prepared_box.pop() if prepared_box else None
        prov_i = provider
        # a lane whose steps would read the carried far list has no template
        prov_i.template_or_raise()
        if handle_in is not None:
            ok_initial = jnp.asarray(True)  # the segment that made it fitted
        else:
            # the scan's initial verdict, on the concrete state before it is freed,
            # against the state's own widths (what the eager prepare validated),
            # not a traced-caps record an earlier trace may have left
            saved_caps = getattr(self, "_strict_fused_traced_caps", None)
            self._strict_fused_traced_caps = None
            ok_initial = jnp.asarray(
                prov_i.capacity_ok(FusedTopology.from_prepared(prepared_curr))
            )
            self._strict_fused_traced_caps = saved_caps
        prepared_curr = None
        # caches that would keep the prepare's buffers alive through the scan
        self._topology_reuse_entry = None
        self._prepared_state_cache_key = None
        self._prepared_state_cache_value = None
        self._prepared_state_cache_positions = None
        self._prepared_state_cache_masses = None
        retry_enabled = env_flag("JACCPOT_STRICT_SEGMENT_RETRY", True)
        start_host: Optional[Any] = None
        if donate_state and retry_enabled:
            # the scan consumes the start state, and a capacity retry re-runs the
            # segment from it: keep it on the host (one copy per call, only when
            # donating -- the scan itself stays as lean as it can be; on a GPU
            # asynchronous, into pinned memory)
            start_host = host_copy(state_arr)
        if handle_in is not None:
            scale_seed = getattr(handle_in, "force_scale", None)
        if scale_carry and scale_seed is None:
            raise RuntimeError(
                "mac_type='dehnen_error' on the fused lane needs a force scale to "
                "start from: none came from the eager prepare or the handle (a handle "
                "from a run without the criterion cannot continue one with it)"
            )
        retries, tree_first = 0, False
        state_now = state_arr
        scale_now: Optional[Array] = scale_seed
        acc_self_now: Optional[Array] = acceleration_self_current
        del acceleration_self_current
        acc_self_owned = self_owned
        # evaluated by the start program (the first call's start, owned)
        in_scan_force = acc_self_now is None
        donate_state_now = bool(donate_state)
        while True:
            ran_scan = True
            if in_scan_force:
                acc_self_now, ok_initial, walk_needs, scale_now = _start_force_for(
                    prov_i
                )(state_now, masses_arr, ok_initial, scale_now, *params_args)
                acc_self_owned = True
                if not bool(np.asarray(jax.device_get(ok_initial))):
                    # the start itself overflows: no scan, straight to a retry
                    ran_scan = False
                    ok_all, done = ok_initial, jnp.zeros((), jnp.int32)
                    state_out = acc_self_out = history_out = scale_out = None
                    acc_total_out = None
            if ran_scan:
                runner = _runner_for(
                    prov_i,
                    num_steps_i,
                    donate_state_now,
                    external_active or acc_self_owned,
                )
                assert acc_self_now is not None
                if start_total is not None:
                    # the previous scan's own total at these positions (first
                    # attempt only: a retry builds it again)
                    acc0, start_total = start_total, None
                else:
                    acc0 = _initial_acceleration(acc_self_now, state_now)
                if acc_self_owned and not external_active:
                    acc_self_now = None  # it is acc0, which the call consumes
                outputs, history_out = runner(
                    state_now, acc0, masses_arr, ok_initial, scale_now, *params_args
                )
                (
                    state_out,
                    acc_self_out,
                    ok_all,
                    walk_needs,
                    done,
                    scale_out,
                ) = outputs[:6]
                acc_total_out = outputs[6] if total_out else acc_self_out
                del acc0, outputs
            self._strict_static_target_block_capacity_ok = bool(
                np.asarray(jax.device_get(ok_all))
            )
            if self._strict_static_target_block_capacity_ok:
                handle = StrictParticleCarry(
                    template=prov_i.template,
                    self_acceleration=acc_self_out,
                    num_particles=int(state_arr.shape[0]),
                    force_scale=scale_out,
                    provider=prov_i,
                    force_key=force_key,
                    acceleration=acc_total_out if return_total else None,
                    external_key=external_key if return_total else None,
                )
                return state_out, handle, history_out
            needs = np.asarray(jax.device_get(walk_needs)).astype(np.int64)
            # diagnostics: the steps the failed segment completed
            self._strict_particle_failed_step = int(np.asarray(jax.device_get(done)))
            # segment retry: re-plan the walk caps and the tree from what the
            # segment needed, re-prepare at the segment's START, run it once more
            # (after a tree re-plan, a second time for the walk)
            mode = (
                self._segment_retry_mode(needs, retries, tree_first)
                if retry_enabled
                else None
            )
            if mode is None:
                self._raise_scan_capacity_saturated(needs, retries=retries)
            if retries == 0:
                tree_first = bool(needs[NEEDS_FIELDS.index("tree_overflow")])
            retries += 1
            del state_out, history_out, acc_self_out, acc_total_out
            if donate_state_now and ran_scan:
                assert start_host is not None
                state_now = jax.device_put(start_host, masses_arr.sharding)
                donate_state_now = True  # our own copy
            replanned = prov_i.replanned(
                needs,
                state_now[:, 0, :],
                masses_arr,
                walk_only=mode == "walk",
                caps_only=caps_only,
            )
            if replanned is None:
                self._raise_scan_capacity_saturated(needs, retries=retries - 1)
            prov_i, topology_start = replanned
            prepared = topology_start.prepared
            del replanned, topology_start
            # a caps-only re-prepare holds no list to evaluate on
            start_program_force = acc_self_now is None and isinstance(
                prepared, PreparedCaps
            )
            if in_scan_force:
                acc_self_now = None  # the start program evaluates it again
            elif acc_self_now is None and not start_program_force:
                # consumed by the failed segment: the same self-gravity
                # again, from the same positions
                acc_self_now = evaluate_self(prepared, state_now, step_params)
            prov_i.template_or_raise()
            if scale_carry and force_scale_seed is not None:
                # the re-prepare's own eq (16b) prepass at the segment's start
                scale_now = force_scale_seed(prepared)
            self._strict_fused_traced_caps = None
            ok_initial = jnp.asarray(
                prov_i.capacity_ok(FusedTopology.from_prepared(prepared))
            )
            del prepared
            if start_program_force and not in_scan_force:
                # the same self-gravity as the steps' own refresh at the segment's
                # start, in its own program; the scan's first thresholds stay the
                # re-prepare's seed, as after a full re-prepare
                acc_self_now, ok_start, _, _ = _start_force_for(prov_i)(
                    state_now, masses_arr, ok_initial, scale_now, *params_args
                )
                if bool(np.asarray(jax.device_get(ok_start))):
                    ok_initial = ok_start
                else:
                    # the start itself overflows: the loop's start program runs
                    # it again and the re-plan sizes from its needs
                    acc_self_now, in_scan_force = None, True
            self._topology_reuse_entry = None
            self._prepared_state_cache_value = None
            self._strict_fused_fallback_count += 1
            self._strict_fused_last_fallback_reason = "capacity_segment_retry"

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
        force_params: Optional[ForceParams] = None,
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
        force_params : Optional[ForceParams]
            As in :meth:`strict_run_v2`: ``None`` (or the solver's own floats) is
            the default program; arrays are arguments of ``eval_fn``'s program,
            bound to the values given here.

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
            If the profile is not large-N production, or the particle count is not
            in ``JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET``.
        ValueError
            If ``force_params`` are refused or beyond the walk's softening (as in
            :meth:`strict_run_v2`).
        """
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        if not self._is_large_n_gpu_production_profile():
            raise RuntimeError(
                "strict_fused_prepared_eval_fn requires large_n_gpu production profile."
            )
        # diagnostic only, as in `strict_run_v2`
        self._strict_fused_mode_active = self._strict_fused_profile_allows_n(
            int(positions_arr.shape[0])
        )
        if not self._strict_fused_mode_active:
            raise RuntimeError(
                "strict fused mode is not active for this particle count: "
                f"N={int(positions_arr.shape[0])} is not in "
                "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET="
                f"{os.environ.get('JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET', '')!r}. "
                "Add this N to the profile set, or leave it empty to allow all N."
            )
        runtime_overrides = self._resolve_runtime_execution_overrides(
            num_particles=int(positions_arr.shape[0])
        )
        prov, topology = FusedForceProvider.prepare(
            self,
            positions_arr,
            masses_arr,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            theta=theta,
            bounds=bounds,
            runtime_overrides=runtime_overrides,
        )
        prepared = topology.prepared
        del topology

        if donate_prepared:
            # see strict_run_v2: caches sharing the state's buffers must not
            # outlive a donating call
            self._topology_reuse_entry = None
            self._prepared_state_cache_value = None
            self._prepared_state_cache_key = None

        force = prov.resolve_params(force_params)
        if force is not None and int(prov.params_overflow(force)):
            raise ValueError(
                f"force_params eps2={force.eps2} is beyond the softening the walk "
                f"is built for (eps2 <= {prov.walk_eps2()}; OVERFLOW_PARAMS)"
            )
        if force is not None:
            # the parameters are an argument of the program, bound to this call's
            # values: one compile for any of them
            @partial(jax.jit, donate_argnums=(0,) if donate_prepared else ())
            def _eval_params(
                prepared_in: LargeNPreparedState, params_in: ForceParams
            ) -> Array:
                return jnp.asarray(
                    prov.accelerations(
                        FusedTopology.from_prepared(prepared_in), params_in
                    )
                )

            return prepared, partial(_eval_params, params_in=force)

        # the provider's evaluation of the state as it stands: no rebuild
        @partial(jax.jit, donate_argnums=(0,) if donate_prepared else ())
        def _eval(prepared_in: LargeNPreparedState) -> Array:
            return jnp.asarray(
                prov.accelerations(FusedTopology.from_prepared(prepared_in))
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
        force_scale_particles: Optional[Array] = None,
        strict_fused_lane: bool = False,
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
        force_scale_particles : Optional[Array]
            ``mac_type='dehnen_error'``: eq (16b)'s ``f_b`` per particle in input
            order, carried from the previous step's evaluation. Reduced to the
            per-node ``min_b f_b`` on this step's tree, it sets eq (16a)'s
            thresholds for the walk. ``None`` (default) keeps the state's own.
        strict_fused_lane : bool
            This is the strict fused lane's per-step refresh (``strict_run_v2``'s
            scan, the multi-GPU ``fused_force_step``): with ``fused_device_mode``
            it takes the traced hot path -- the streamed fast lane or a raise, no
            host side effects, the eager pass's caps widened, a fresh far list
            returned as the input's placeholder. ``False`` (default, every public
            refresh) never does. The caller states it: it used to be read off the
            engine's ``_strict_fused_mode_active``, which the first strict call
            left set, so ``refresh_prepared_state`` changed path after one.

        Returns
        -------
        Optional[LargeNPreparedState]
            The refreshed state, or ``None`` when the fast path declined -- see
            above. What it calls raises only for genuine inconsistencies, never
            for a declined reuse.
        """
        # Side channels for the traced capacity guard and the segment retry; see
        # the end of this method and `capacity_guard.last_refresh_walk_needs`.
        self._last_refresh_capacity_ok = None
        self._last_refresh_walk_needs = None
        self._last_refresh_tree_overflow = None
        self._last_refresh_tree_needs = None

        self._large_n_same_topology_refresh_attempts += 1
        if not isinstance(prepared_state.tree, RadixTree):
            self._large_n_same_topology_refresh_misses += 1
            self._large_n_same_topology_refresh_miss_no_key += 1
            return None

        # no host stage timers inside the fused lane: they would sync the device
        # (`JACCPOT_STATIC_STRICT_FUSED_DISABLE_HOT_TIMING=0` went in X6)
        refresh_timing_active = bool(
            getattr(self, "_refresh_timing_active", False)
        ) and not bool(fused_device_mode)

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
        if self.tree_type != "radix" and tree_config.mode == "static_radix":
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
            # the cell-leaf rebuild's own capacity flag (leaf cap, depth, level
            # width; a tracer under trace), for `FusedTopology.overflow_bits`,
            # and what it needed of each, for the segment's re-plan
            self._last_refresh_tree_overflow = getattr(
                build_artifacts, "leaf_capacity_overflow", None
            )
            self._last_refresh_tree_needs = getattr(build_artifacts, "tree_needs", None)
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

        strict_fused_traced_hot_path = bool(fused_device_mode) and bool(
            strict_fused_lane
        )
        # The fused refresh builds its far-pair list fresh every step and returns
        # the input's as a shape placeholder: the list a state carries in is never
        # read (the unsafe reuse of it went in the 2026-10 cleanup, X6).
        compact_far_pairs_carry_placeholder = getattr(
            prepared_state, "compact_far_pairs", None
        )
        safe_fresh_compact_pair_rebuild = (
            bool(strict_fused_traced_hot_path)
            and str(tree_config.mode).strip().lower() == "static_radix"
        )
        # The one point where a cross-domain exchange belongs: the multipoles exist
        # here and the downward sweep below has not consumed them.
        # It goes in the REFRESH because that is what a per-step force runs; the
        # prepare path builds the state once, and `LargeNPreparedState.upward` is None
        # (measured), so no caller holding a state can reach the multipoles at all.
        # Doing the cross field afterwards instead would need a SECOND L2L cascade,
        # and Phase 3.4 measured the far half as the bigger one.
        #
        # ABOVE the dual/downward build, not inside it: there used to be a second
        # route (the unsafe compact-pair reuse, removed in the 2026-10 cleanup,
        # X6), and a hook in only one of them silently never fired on the other.
        #
        # Phase C1: called, result discarded, force bit-identical either way.
        cross_far = None
        if cross_hook is not None:
            # ONE exact-radius COM geometry for the hook and the local walk below:
            # each used to resolve its own, 9.7 ms per force at 1e6 particles per
            # A100. Shared only where the two would compute the same thing -- a
            # tree whose walk defaults to COM (static radix, the fused lane's) and
            # no folded per-node radius scale (`dehnen_theta` folds one the hook
            # does not apply).
            if (
                walk_geometry_default(tree_artifacts.tree_mode, upward_center_mode)
                == "com"
                and self._folded_criterion_radius_scale() is None
            ):
                tree_artifacts = tree_artifacts._replace(
                    walk_geometry=self._walk_geometry(tree_artifacts)
                )
            cross_far = cross_hook(tree_artifacts)
        force_scale_nodes_now = prepared_state.force_scale_nodes
        if force_scale_particles is not None:
            from jaccpot.runtime._force_scale_levels import node_force_scale_min

            levels_fb = self._resolve_upward_num_levels(tree_artifacts.tree)
            if levels_fb is None:
                levels_fb = int(get_level_offsets(tree_artifacts.tree).shape[0] - 1)
            force_scale_nodes_now = node_force_scale_min(
                tree=tree_artifacts.tree,
                force_scale_particles=force_scale_particles,
                num_levels=int(levels_fb),
            )
        dual_downward_artifacts = self._prepare_state_dual_and_downward(
            cross_far=cross_far,
            tree_artifacts=tree_artifacts,
            force_scale_nodes=force_scale_nodes_now,
            upward_center_mode=upward_center_mode,
            theta_val=theta_val,
            mac_type_val=mac_type_val,
            dehnen_radius_scale=self.dehnen_radius_scale,
            runtime_traversal_config=runtime_traversal_config,
            runtime_m2l_chunk_size=runtime_m2l_chunk_size,
            runtime_l2l_chunk_size=runtime_l2l_chunk_size,
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
            self._recent_retry_events = tuple(collected_retries)
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
