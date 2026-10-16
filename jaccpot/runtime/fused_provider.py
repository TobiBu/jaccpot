"""``FusedForceProvider``: the fused lane's per-step force, as an object.

The fused ``strict_run_v2`` scan computes one force per step in two halves, and
this module names them:

``rebuild``
    the topology at new positions: the static-radix tree rebuilt from its
    template, P2M and M2M, the flat walk with its fresh far and near lists, M2L
    and L2L (``_refresh_large_n_same_topology`` on the strict lane). The result is
    a :class:`FusedTopology`: the refreshed prepared state plus what the walk and
    the tree reported about their fixed capacities (as tracers, under a trace).
``accelerations``
    the evaluation on that topology: near-field P2P, L2P and the scatter back to
    the input order (``evaluate_large_n_state``). It takes no positions or masses:
    the rebuild already ran the far field at the positions it was given.

:class:`FusedForceProvider` holds the static configuration both halves are traced
under (the engine, leaf size, order, opening angle, runtime overrides) and the
SHAPE TEMPLATE of the lane's prepared state (:mod:`jaccpot.runtime.strict_carry`),
which is all a rebuild reads of a prepared state: no carried leaf value reaches a
step's outputs (``tests/integration/strict/test_strict_run_v2.py``'s dead-code
test), so a caller carries particles and a template, never a stale far list.

The provider is eager to make (``prepare`` / ``freeze_template``) and traced to
use: the caller jits ``rebuild`` / ``accelerations`` / ``step`` inside its own
program. ``key()`` is what a compile cache keyed on the provider must hold.

Its inputs and outputs -- :class:`~jaccpot.runtime.fused_topology.FusedTopology`
and the force parameters (:class:`~jaccpot.runtime.fused_topology.ForceParams`,
:class:`~jaccpot.runtime.fused_topology.StepParams`) -- live in
:mod:`jaccpot.runtime.fused_topology`. Only the engine's own ``G`` and softening
are accepted for now; traced values come later (S2 stage C4).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Union

import jax.numpy as jnp
from beartype.typing import Callable, Tuple
from jaxtyping import Array

from ._large_n_pipeline import evaluate_large_n_state
from ._large_n_types import LargeNPreparedState
from .fused_topology import ForceParams, FusedTopology, StepParams
from .strict_carry import PreparedShapeTemplate, materialize_template, shape_template

__all__ = [
    "FusedForceProvider",
    "require_cell_leaves",
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


def require_cell_leaves(solver: Any) -> None:
    """Refuse a rebuild from a shape template on bucket leaves.

    On cell leaves (``TreeConfig(leaf_partition="cells")``) the static-radix
    rebuild builds the cells and the radix structure over them from the new
    positions: it reads only the template's shapes. On bucket leaves it keeps
    the eager tree's structure -- its node ranges and children
    (``yggdrax`` ``rebuild_static_radix_tree_from_template``) -- so it reads
    those VALUES, which a shape template does not hold: the particle carry
    materialises them as zeros and the walk overflows its queue (a peak
    wavefront of 3^11 = 177,147 at N = 512) on every step.

    Parameters
    ----------
    solver : Any
        ``FastMultipoleMethod`` or its engine.

    Raises
    ------
    ValueError
        If the solver's leaves are buckets.
    """
    engine = _engine_of(solver)
    if getattr(engine, "_tree_leaf_partition", "buckets") != "cells":
        raise ValueError(
            "carry='particles' (a rebuild from a shape template) needs cell "
            "leaves: TreeConfig(leaf_partition='cells', leaf_capacity=...). On "
            "bucket leaves the rebuild keeps the eager tree's node ranges and "
            "children, values a shape template does not hold; carry='state' "
            "runs on bucket leaves."
        )


def _engine_of(solver: Any) -> Any:
    """The runtime engine behind a facade (or the engine itself).

    Parameters
    ----------
    solver : Any
        ``FastMultipoleMethod`` or its engine.

    Returns
    -------
    Any
        The engine.
    """
    return getattr(solver, "_impl", solver)


# --------------------------------------------------------------------------
# The provider.
# --------------------------------------------------------------------------


class FusedForceProvider:
    """The fused lane's force: ``rebuild`` a topology, evaluate ``accelerations``.

    Built eagerly -- :meth:`prepare` (a prepare at given positions) or
    :meth:`freeze_template` (from a prepared state) -- and used inside the caller's
    jitted program. It holds no device buffer: the engine, the static
    configuration and the prepared state's shape template.

    Parameters
    ----------
    engine : Any
        The runtime engine (``FastMultipoleMethod._impl``) or the facade.
    leaf_size : int
        Leaf-size request (the refresh's cache leaf parameter).
    max_order : int
        Expansion order ``p``.
    theta : Optional[float]
        Opening-angle override; ``None`` takes the engine's at trace time.
    runtime_overrides : Any
        Runtime execution overrides for the eager prepares (a re-plan's); the
        traced rebuild resolves its own, as the strict scan always has.
    template : Optional[PreparedShapeTemplate]
        Shapes of the lane's prepared state (far list detached), which
        :meth:`rebuild` materialises; ``None`` on a lane that reads its carried
        far list (not the static-radix fused lane), where only
        :meth:`_rebuild_like` with a real state works.
    """

    def __init__(
        self,
        engine: Any,
        *,
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
        runtime_overrides: Any = None,
        template: Optional[PreparedShapeTemplate] = None,
    ) -> None:
        self.engine = _engine_of(engine)
        self.leaf_size = int(leaf_size)
        self.max_order = int(max_order)
        self.theta = theta
        self.runtime_overrides = runtime_overrides
        self.template = template

    # ----------------------------------------------------------------------
    # Eager: prepare, freeze, re-plan.
    # ----------------------------------------------------------------------

    @classmethod
    def prepare(
        cls,
        solver: Any,
        positions: Array,
        masses: Array,
        *,
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
        bounds: Optional[Tuple[Array, Array]] = None,
        runtime_overrides: Any = None,
    ) -> tuple["FusedForceProvider", FusedTopology]:
        """Prepare the fused lane at ``positions`` and freeze its template.

        The eager prepare ``strict_run_v2`` bootstraps its scan with: the fused
        device-resident layout, validated walk caps recorded on the engine (the
        traced rebuild is sized from them) and, under ``mac_type='dehnen_error'``,
        the eq (16b) prepass.

        Parameters
        ----------
        solver : Any
            ``FastMultipoleMethod`` or its engine, on the ``large_n_gpu`` profile.
        positions : Array
            ``[N, 3]`` positions.
        masses : Array
            ``[N]`` masses.
        leaf_size : int
            Leaf-size request.
        max_order : int
            Expansion order ``p``.
        theta : Optional[float]
            Opening-angle override.
        bounds : Optional[Tuple[Array, Array]]
            Morton box; ``None`` infers it from ``positions``.
        runtime_overrides : Any
            Already-resolved runtime overrides; ``None`` resolves them for ``N``.

        Returns
        -------
        tuple[FusedForceProvider, FusedTopology]
            The provider and the topology at ``positions`` (``from_prepared``).

        Raises
        ------
        RuntimeError
            If the profile is not large-N production, the particle count is not in
            ``JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET``, or the prepare did not
            return a ``LargeNPreparedState``.
        """
        engine = _engine_of(solver)
        positions_arr = jnp.asarray(positions)
        masses_arr = jnp.asarray(masses)
        n = int(positions_arr.shape[0])
        if not engine._is_large_n_gpu_production_profile():
            raise RuntimeError(
                "FusedForceProvider requires the large_n_gpu production profile."
            )
        if not engine._strict_fused_profile_allows_n(n):
            raise RuntimeError(
                f"FusedForceProvider: N={n} is not in "
                "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET; add it, or leave the set "
                "empty to allow all N."
            )
        if runtime_overrides is None:
            runtime_overrides = engine._resolve_runtime_execution_overrides(
                num_particles=n
            )
        prepared = engine.prepare_state(
            positions_arr,
            masses_arr,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            theta=theta,
            jit_tree=engine._jit_tree_default,
            runtime_overrides_override=runtime_overrides,
            fused_device_mode=True,
            **({} if bounds is None else {"bounds": bounds}),
        )
        if not isinstance(prepared, LargeNPreparedState):
            raise RuntimeError(
                "the fused lane's prepare returned a "
                f"{type(prepared).__name__}, not a LargeNPreparedState"
            )
        engine._record_large_n_eval_shape_diagnostics(prepared)
        provider = cls.freeze_template(
            engine,
            prepared,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
            runtime_overrides=runtime_overrides,
        )
        return provider, FusedTopology.from_prepared(prepared)

    @classmethod
    def freeze_template(
        cls,
        solver: Any,
        prepared: Any,
        *,
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
        runtime_overrides: Any = None,
    ) -> "FusedForceProvider":
        """A provider for an existing prepared state (its shapes, not its buffers).

        Parameters
        ----------
        solver : Any
            ``FastMultipoleMethod`` or its engine.
        prepared : Any
            The lane's prepared state.
        leaf_size : int
            Leaf-size request.
        max_order : int
            Expansion order ``p``.
        theta : Optional[float]
            Opening-angle override.
        runtime_overrides : Any
            Runtime overrides for a re-plan's eager prepare.

        Returns
        -------
        FusedForceProvider
            With the template when the lane rebuilds its far list fresh every step
            (static-radix fused lane), else without one.
        """
        engine = _engine_of(solver)
        template = None
        if engine._strict_far_pairs_ride_outside_the_scan(prepared):
            template = shape_template(replace(prepared, compact_far_pairs=None))
        return cls(
            engine,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
            runtime_overrides=runtime_overrides,
            template=template,
        )

    def replanned(
        self,
        needs: Any,
        positions: Array,
        masses: Array,
    ) -> Optional[tuple["FusedForceProvider", FusedTopology]]:
        """Re-plan the walk caps from what a failed segment needed and re-prepare.

        The segment retry: the unnamed flat-walk caps grow to cover ``needs``
        (``engine._replan_walk_caps_from_needs``), the lane is prepared again at
        the segment's START, and the peak wavefront the segment needed is kept so
        the retraced walk's queue covers it.

        Parameters
        ----------
        needs : Any
            The segment's running maximum of :meth:`FusedTopology.needs` (host).
        positions : Array
            ``[N, 3]`` positions the segment started from.
        masses : Array
            ``[N]`` masses.

        Returns
        -------
        Optional[tuple[FusedForceProvider, FusedTopology]]
            The provider and topology for the re-run, or ``None`` when nothing can
            be re-planned (no walk flag fired, a named cap overflowed, not the
            flat walk): the caller raises.
        """
        engine = self.engine
        if not engine._replan_walk_caps_from_needs(needs):
            return None
        replanned_peak = int(
            (engine._strict_fused_validated_caps or {}).get("peak_wavefront", 0)
        )
        prepared = engine.prepare_state(
            positions,
            masses,
            leaf_size=int(self.leaf_size),
            max_order=int(self.max_order),
            theta=self.theta,
            jit_tree=engine._jit_tree_default,
            runtime_overrides_override=self.runtime_overrides,
            fused_device_mode=True,
        )
        # the eager walk saw the segment's START; keep the wavefront the segment
        # needed, so the retraced walk's queue covers it
        validated = dict(engine._strict_fused_validated_caps or {})
        validated["peak_wavefront"] = max(
            int(validated.get("peak_wavefront") or 0), replanned_peak
        )
        engine._strict_fused_validated_caps = validated
        provider = type(self).freeze_template(
            engine,
            prepared,
            leaf_size=self.leaf_size,
            max_order=self.max_order,
            theta=self.theta,
            runtime_overrides=self.runtime_overrides,
        )
        return provider, FusedTopology.from_prepared(prepared)

    # ----------------------------------------------------------------------
    # Static configuration and the compile key.
    # ----------------------------------------------------------------------

    @property
    def scale_carry(self) -> bool:
        """``mac_type='dehnen_error'`` on the flat walk: f_b rides from step to step.

        Returns
        -------
        bool
            Whether :meth:`accelerations` must return the force scale and the next
            :meth:`rebuild` take it.
        """
        return bool(self.engine._flat_walk_criterion_active())

    @property
    def rectangle_guard_active(self) -> bool:
        """Whether the rectangle near-field capacity is a correctness condition.

        Returns
        -------
        bool
            ``False`` on the CSR near-field lane, which never reads the rectangle.
        """
        from jaccpot.nearfield._fast_lane import _nearfield_csr_lane_active

        return not _nearfield_csr_lane_active(
            bool(getattr(self.engine, "use_pallas", False))
        )

    @property
    def theta_value(self) -> float:
        """The opening angle the rebuild walks with.

        Returns
        -------
        float
            The override, else the engine's.
        """
        return float(self.engine.theta if self.theta is None else self.theta)

    def params(self) -> ForceParams:
        """The engine's own force parameters, static.

        Returns
        -------
        ForceParams
            ``ForceParams(G, softening**2)`` as Python floats.
        """
        return ForceParams(
            G=float(self.engine.G), eps2=float(self.engine.softening) ** 2
        )

    def key(self, like: Any = None) -> tuple:
        """What a compile cache keyed on this provider must hold.

        Everything the traced rebuild and evaluation read at TRACE time, apart from
        what the engine fixes for its lifetime (``G``, softening, MAC, kernels --
        the engine's own cache is per engine): the leaf, order and opening angle,
        whether f_b is carried, the diagnostic modes, the level-loop bound and the
        validated walk caps the traced walk is sized from, and the template's
        shapes when the rebuild materialises it.

        The level bound is resolved now (``engine._resolve_upward_num_levels``):
        from ``like``'s tree when it is a concrete prepared state -- which stashes
        that tree's depth for the trace that follows, as the state carry has always
        done -- else the engine's stash.

        Parameters
        ----------
        like : Any
            The prepared state a caller will rebuild from (the state carry's carry),
            or ``None`` when the rebuild materialises the template.

        Returns
        -------
        tuple
            Hashable.
        """
        engine = self.engine
        levels = engine._resolve_upward_num_levels(getattr(like, "tree", None))
        parts: tuple = (
            int(self.leaf_size),
            int(self.max_order),
            self.theta_value,
            self.scale_carry,
            str(getattr(engine, "_strict_refresh_diag_mode", "full")),
            str(getattr(engine, "_strict_refresh_detail_diag_mode", "full")),
            str(getattr(engine, "_large_n_eval_diag_mode", "full")),
            str(getattr(engine, "_large_n_nearfield_diag_mode", "full")),
            levels,
            _walk_caps_key(getattr(engine, "_strict_fused_validated_caps", None)),
        )
        if like is None and self.template is not None:
            parts = parts + (self.template.key(),)
        return parts

    # ----------------------------------------------------------------------
    # Traced: rebuild, evaluate, verdict, step.
    # ----------------------------------------------------------------------

    def template_or_raise(self) -> PreparedShapeTemplate:
        """The shape template, or the reason this lane has none.

        Returns
        -------
        PreparedShapeTemplate
            The template :meth:`rebuild` materialises.

        Raises
        ------
        ValueError
            If the lane reads its carried far list (not the static-radix fused
            lane's fresh far-pair rebuild): a rebuild from shapes alone would read
            a list that is not there; or if the leaves are buckets
            (:func:`require_cell_leaves`).
        """
        if self.template is None:
            raise ValueError(
                "carry='particles' needs the fused static-radix lane's fresh "
                "far-pair rebuild inside the scan (a LargeNPreparedState on a "
                "static_radix tree): otherwise a step reads the carried far "
                "list"
            )
        require_cell_leaves(self.engine)
        return self.template

    def like(self) -> Any:
        """The template as a prepared state of broadcasts (inside a trace).

        Returns
        -------
        Any
            :func:`~jaccpot.runtime.strict_carry.materialize_template` of the
            template; nothing reads its values.
        """
        return materialize_template(self.template_or_raise())

    def rebuild(
        self,
        positions: Array,
        masses: Array,
        *,
        force_scale: Optional[Array] = None,
        bounds: Optional[Tuple[Array, Array]] = None,
        num_valid: Optional[Array] = None,
        cross_hook: Optional[Callable[[Any], Any]] = None,
    ) -> FusedTopology:
        """The topology at ``positions``: tree, P2M/M2M, walk, M2L, L2L.

        Traced (call it inside the caller's jit). Reads only the template's shapes.

        Parameters
        ----------
        positions : Array
            ``[N, 3]`` positions.
        masses : Array
            ``[N]`` masses.
        force_scale : Optional[Array]
            ``mac_type='dehnen_error'``: eq (16b)'s f_b per particle (input order)
            from the previous evaluation, for eq (16a)'s thresholds. Required then.
        bounds : Optional[Tuple[Array, Array]]
            Morton box; ``None`` infers it from ``positions``.
        num_valid : Optional[Array]
            Live row count of a capacity-padded shard (cell leaves only).
        cross_hook : Optional[Callable[[Any], Any]]
            The multi-GPU cross-domain hook (see ``_refresh_large_n_same_topology``).

        Returns
        -------
        FusedTopology
            ``fresh=True``, with the rebuild's capacity verdict and walk needs.

        Raises
        ------
        ValueError
            If the criterion needs a force scale and none was given (the template
            holds no thresholds of its own), or the provider has no template
            (:meth:`template_or_raise`).
        """
        if force_scale is None and self.scale_carry:
            raise ValueError(
                "mac_type='dehnen_error': the rebuild needs the previous "
                "evaluation's force scale (accelerations(..., force_scale=True))"
            )
        return self._rebuild_like(
            self.like(),
            positions,
            masses,
            force_scale=force_scale,
            bounds=bounds,
            num_valid=num_valid,
            cross_hook=cross_hook,
        )

    def _rebuild_like(
        self,
        like: Any,
        positions: Array,
        masses: Array,
        *,
        force_scale: Optional[Array] = None,
        bounds: Optional[Tuple[Array, Array]] = None,
        num_valid: Optional[Array] = None,
        cross_hook: Optional[Callable[[Any], Any]] = None,
    ) -> FusedTopology:
        """:meth:`rebuild` from a given prepared state instead of the template.

        The state carry rebuilds from the state it carries (its shapes and static
        fields, no values): materialising the template there would add broadcasts
        the carry does not have.

        Parameters
        ----------
        like : Any
            A prepared state of the lane's shapes (concrete, traced or broadcast).
        positions : Array
            ``[N, 3]`` positions.
        masses : Array
            ``[N]`` masses.
        force_scale : Optional[Array]
            As in :meth:`rebuild`; ignored unless :attr:`scale_carry`.
        bounds : Optional[Tuple[Array, Array]]
            As in :meth:`rebuild`.
        num_valid : Optional[Array]
            As in :meth:`rebuild`.
        cross_hook : Optional[Callable[[Any], Any]]
            As in :meth:`rebuild`.

        Returns
        -------
        FusedTopology
            As in :meth:`rebuild`.

        Raises
        ------
        RuntimeError
            If ``like`` is not a ``LargeNPreparedState`` or the refresh declined
            (topology/profile mismatch).
        """
        engine = self.engine
        if not isinstance(like, LargeNPreparedState):
            raise RuntimeError(
                "strict_run_v2 reached the large-N refresh with a "
                f"{type(like).__name__}. Only the large_n_gpu "
                "production profile gets this far, so this is an "
                "internal invariant, not a configuration error."
            )
        scale_carry = self.scale_carry
        prepared_new = engine._refresh_large_n_same_topology(
            like,
            positions,
            masses,
            bounds=bounds,
            leaf_size=int(self.leaf_size),
            max_order=int(self.max_order),
            theta=self.theta,
            runtime_overrides_override=None,
            fused_device_mode=True,
            num_valid=num_valid,
            cross_hook=cross_hook,
            force_scale_particles=force_scale if scale_carry else None,
            strict_fused_lane=True,
        )
        if prepared_new is None:
            raise RuntimeError(
                "strict velocity-Verlet refresh failed: topology/profile mismatch"
            )
        traced_caps = getattr(engine, "_strict_fused_traced_caps", None) or {}
        topo = FusedTopology(
            prepared=prepared_new,
            refresh_ok=getattr(engine, "_last_refresh_capacity_ok", None),
            walk_needs=getattr(engine, "_last_refresh_walk_needs", None),
            tree_overflow=getattr(engine, "_last_refresh_tree_overflow", None),
            fresh=True,
            caps=(
                traced_caps.get("max_neighbors_per_leaf_used"),
                traced_caps.get("compact_far_pair_capacity"),
                bool(self.rectangle_guard_active),
            ),
        )
        # the side channels are moved, not copied: they hold this trace's tracers
        engine._last_refresh_capacity_ok = None
        engine._last_refresh_walk_needs = None
        engine._last_refresh_tree_overflow = None
        return topo

    def accelerations(
        self,
        topology: FusedTopology,
        params: Optional[Union[ForceParams, StepParams]] = None,
        *,
        order: str = "input",
        force_scale: bool = False,
    ) -> Any:
        """The self-gravity on ``topology``: near P2P, L2P, scatter.

        Parameters
        ----------
        topology : FusedTopology
            From :meth:`rebuild` (or :meth:`FusedTopology.from_prepared`).
        params : Optional[Union[ForceParams, StepParams]]
            ``None`` or the engine's own (:meth:`params`); other values arrive with
            traced parameters (S2 stage C4).
        order : str
            ``"input"`` (the caller's row order) or ``"tree"`` (the topology's
            Morton order: row ``j`` is particle ``tree.particle_indices[j]``; no
            gather back).
        force_scale : bool
            Also return eq (16b)'s f_b per particle, in the same order.

        Returns
        -------
        Any
            ``[N, 3]`` accelerations in the state's input dtype, or ``(acc, f_b)``.

        Raises
        ------
        ValueError
            On an unknown ``order``.
        RuntimeError
            If the topology does not hold a ``LargeNPreparedState``.
        """
        self._check_params(params)
        if order not in ("input", "tree"):
            raise ValueError(f"order must be 'input' or 'tree', got {order!r}")
        prepared = topology.prepared
        if not isinstance(prepared, LargeNPreparedState):
            raise RuntimeError(
                "strict_run_v2 reached the large-N self-evaluation with a "
                f"{type(prepared).__name__}. Only the large_n_gpu "
                "production profile gets this far, so this is an internal "
                "invariant, not a configuration error."
            )
        extra: dict = {}
        if order == "tree":
            extra["sorted_output"] = True
        if force_scale:
            extra["return_force_scale"] = True
        return evaluate_large_n_state(
            self.engine,
            prepared,
            target_indices=None,
            return_potential=False,
            max_acc_derivative_order=0,
            **extra,
        )

    def capacity_ok(self, topology: FusedTopology) -> Array:
        """The step's capacity verdict, as the fused scan folds it into its carry.

        Every fixed capacity of the returned state (``fused_state_capacity_ok``
        with the engine's traced caps, read NOW -- inside the trace that recorded
        them) and, for a rebuilt topology, the far list the rebuild built. Equal to
        ``overflow_ok(topology.overflow_bits())`` but for the tree's own overflow
        (``OVERFLOW_TREE``), which it misses on the cell-leaf rebuild.

        Parameters
        ----------
        topology : FusedTopology
            A rebuilt topology, or an eager one (``fresh=False``: its own widths).

        Returns
        -------
        Array
            Boolean scalar.
        """
        from jaccpot.runtime.capacity_guard import fused_state_capacity_ok

        ok = fused_state_capacity_ok(
            topology.prepared,
            traced_caps=getattr(self.engine, "_strict_fused_traced_caps", None),
            rectangle_guard_active=self.rectangle_guard_active,
        )
        if topology.fresh:
            ok = ok & (
                jnp.asarray(True)
                if topology.refresh_ok is None
                else jnp.asarray(topology.refresh_ok, jnp.bool_)
            )
        return ok

    def step(
        self,
        positions: Array,
        masses: Array,
        params: Optional[Union[ForceParams, StepParams]] = None,
        aux: Optional[tuple] = None,
    ) -> tuple[Any, tuple]:
        """One force: :meth:`rebuild` then :meth:`accelerations`, with the verdict.

        Parameters
        ----------
        positions : Array
            ``[N, 3]`` positions.
        masses : Array
            ``[N]`` masses.
        params : Optional[Union[ForceParams, StepParams]]
            As in :meth:`accelerations`.
        aux : Optional[tuple]
            The previous step's ``(f_b, bits, needs)``, or ``None`` on the first.
            Under ``mac_type='dehnen_error'`` its f_b sets this rebuild's
            thresholds, so the first step needs one too (``(f_b, None, None)``).

        Returns
        -------
        tuple[Any, tuple]
            ``(acc, (f_b, bits, needs))``: f_b for the next step (``None`` without
            the criterion), the overflow bits OR-ed and the walk needs max-ed over
            the steps ``aux`` has seen.
        """
        f_b_in, bits_in, needs_in = (None, None, None) if aux is None else aux
        topology = self.rebuild(positions, masses, force_scale=f_b_in)
        if self.scale_carry:
            acc, f_b = self.accelerations(topology, params, force_scale=True)
        else:
            acc, f_b = self.accelerations(topology, params), None
        bits = topology.overflow_bits()
        needs = topology.needs()
        if bits_in is not None:
            bits = bits | jnp.asarray(bits_in, jnp.uint32)
        if needs_in is not None:
            needs = jnp.maximum(jnp.asarray(needs_in, jnp.int32), needs)
        return acc, (f_b, bits, needs)

    # ----------------------------------------------------------------------
    # Helpers.
    # ----------------------------------------------------------------------

    def _check_params(self, params: Optional[Union[ForceParams, StepParams]]) -> None:
        """Refuse parameters the lane cannot honour yet.

        Parameters
        ----------
        params : Optional[Union[ForceParams, StepParams]]
            As in :meth:`accelerations`.

        Raises
        ------
        NotImplementedError
            For traced values, or values other than the engine's own.
        """
        if params is None:
            return
        force = params.force if isinstance(params, StepParams) else params
        if not force.is_static():
            raise NotImplementedError(
                "traced force parameters are not supported yet: pass Python floats"
            )
        own = self.params()
        if float(force.G) != own.G or float(force.eps2) != own.eps2:
            raise NotImplementedError(
                "force parameters other than the solver's own G and softening are "
                f"not supported yet (got G={force.G}, eps2={force.eps2}; the solver "
                f"has G={own.G}, eps2={own.eps2}): pass provider.params() or None"
            )
