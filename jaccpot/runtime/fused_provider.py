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
``potential``
    the same topology's potential per particle, O(N) (``evaluate_large_n_potential``:
    the near field's fourth lane, the far field's L2P value): the energy diagnostic's,
    in a program of its own.

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
:mod:`jaccpot.runtime.fused_topology`.

**Force parameters.** ``accelerations`` / ``step`` take ``G`` and the softening
squared from ``params``: ``None`` (or the engine's own values as Python floats) is
the engine's, compile-time literals in the program exactly as before; other Python
floats are literals too; arrays are TRACED -- cast to the working dtype here, one
program for every value. The rebuild does not read them: the walk is built for
the engine's softening, so a compact kernel's separation floor stays static and an
``eps2`` above the one it covers sets ``OVERFLOW_PARAMS`` (:meth:`params_overflow`).
``mac_type='dehnen_error'`` takes the engine's own only (its thresholds and far
f_b are built with them).

**A caller's integrator.** ``step`` inside the caller's ``lax.scan``, with the
drift and kick around it, is the force ``strict_run_v2(carry="particles")`` steps
with (``tests/integration/strict/test_strict_runners_on_provider.py``). User guide:
``docs/fused_provider.md``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Callable, Tuple
from jaxtyping import Array

from ..softening import support_factor
from ._lane_platform import on_lane_platform, on_lane_platform_of_solver
from ._large_n_pipeline import evaluate_large_n_potential, evaluate_large_n_state
from ._large_n_types import LargeNPreparedState
from .fused_topology import (
    OVERFLOW_GRAD_LISTS,
    OVERFLOW_PARAMS,
    ForceParams,
    FusedTopology,
    StepParams,
    _is_static,
    overflow_ok,
)
from .kernels._evaluate import _cubic_bounds_active
from .strict_carry import (
    PreparedCaps,
    PreparedShapeTemplate,
    materialize_template,
    shape_template,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only
    # deferred, as ``fmm_evaluate`` defers it: the gradient lane's modules stay out
    # of the forward lane's import graph
    from ._large_n_grad import LargeNGradPlan, LargeNGradStatics

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


#: what a traced rebuild leaves on the engine for the trace that runs it (its walk
#: caps and the side channels its verdict and needs are moved from): saved and
#: restored around the caps-only template's ``eval_shape``, which must leave none
_TRACE_STATE = (
    "_strict_fused_traced_caps",
    "_last_refresh_capacity_ok",
    "_last_refresh_walk_needs",
    "_last_refresh_tree_overflow",
    "_last_refresh_tree_needs",
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


def _fold_verdict(
    bits: Array, needs: Array, bits_in: Any, needs_in: Any
) -> tuple[Array, Array]:
    """Fold one step's bits and needs into the ones ``aux`` carried.

    A step counts only when it started on the trajectory: every step ``aux`` has
    seen fitted. After a failure the caller's scan steps on a truncated force, and
    a retry re-plans from these needs (nornax's ``retry_segment``), as
    ``strict_run_v2``'s scans fold them. :meth:`FusedForceProvider.step`'s fold,
    shared with :meth:`FusedForceProvider.differentiable_step` (the same ops).

    Parameters
    ----------
    bits : Array
        This step's ``uint32`` overflow bits.
    needs : Array
        This step's ``int32`` needs.
    bits_in : Any
        The bits ``aux`` carried, or ``None``.
    needs_in : Any
        The needs ``aux`` carried, or ``None``.

    Returns
    -------
    tuple[Array, Array]
        ``(bits, needs)`` folded.
    """
    on_trajectory = None
    if bits_in is not None:
        bits_in = jnp.asarray(bits_in, jnp.uint32)
        on_trajectory = overflow_ok(bits_in)
        bits = jnp.where(on_trajectory, bits, jnp.uint32(0)) | bits_in
    if needs_in is not None:
        needs_in = jnp.asarray(needs_in, jnp.int32)
        needs = jnp.maximum(needs_in, needs)
        if on_trajectory is not None:
            needs = jnp.where(on_trajectory, needs, needs_in)
    return bits, needs


@jax.custom_vjp
def _gradient_gate(acc: Array, ok: Array) -> Array:
    """``acc`` itself; its cotangent becomes NaN when ``ok`` is ``False``.

    The differentiable step's refusal, in the program: on lists its reverses
    cannot read (``ok`` the plan's symmetry verdict) the gradient is NaN, never a
    plausible wrong number, whether or not the caller reads the bits.

    Parameters
    ----------
    acc : Array
        The accelerations.
    ok : Array
        Boolean scalar.

    Returns
    -------
    Array
        ``acc``.
    """
    del ok
    return acc


def _gradient_gate_fwd(acc: Array, ok: Array) -> tuple[Array, Array]:
    """Forward of :func:`_gradient_gate`: the identity, ``ok`` kept.

    Parameters
    ----------
    acc : Array
        The accelerations.
    ok : Array
        Boolean scalar.

    Returns
    -------
    tuple[Array, Array]
        ``(acc, ok)``.
    """
    return acc, ok


def _gradient_gate_bwd(ok: Array, cotangent: Array) -> tuple[Array, None]:
    """Reverse of :func:`_gradient_gate`: the cotangent, or NaN.

    Parameters
    ----------
    ok : Array
        Boolean scalar from the forward.
    cotangent : Array
        The accelerations' cotangent.

    Returns
    -------
    tuple[Array, None]
        The cotangent where ``ok``, NaN otherwise; none for ``ok``.
    """
    nan = jnp.asarray(jnp.nan, cotangent.dtype)
    return jnp.where(ok, cotangent, nan), None


_gradient_gate.defvjp(_gradient_gate_fwd, _gradient_gate_bwd)


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
    grad_statics : Optional[LargeNGradStatics]
        The gradient plan's static fields (:meth:`with_grad_plan`), which
        :meth:`differentiable_step` needs; ``None`` (default) on a provider made
        for the forward only.
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
        grad_statics: Optional[LargeNGradStatics] = None,
    ) -> None:
        self.engine = _engine_of(engine)
        self.leaf_size = int(leaf_size)
        self.max_order = int(max_order)
        self.theta = theta
        self.runtime_overrides = runtime_overrides
        self.template = template
        self.grad_statics = grad_statics

    # ----------------------------------------------------------------------
    # Eager: prepare, freeze, re-plan.
    # ----------------------------------------------------------------------

    @classmethod
    @on_lane_platform_of_solver
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
        caps_only: bool = False,
        differentiable: bool = False,
    ) -> tuple["FusedForceProvider", FusedTopology]:
        """Prepare the fused lane at ``positions`` and freeze its template.

        The eager prepare ``strict_run_v2`` bootstraps its scan with: the fused
        device-resident layout, validated walk caps recorded on the engine (the
        traced rebuild is sized from them) and, under ``mac_type='dehnen_error'``,
        the eq (16b) prepass.

        ``caps_only``: what the particle carry reads of that prepare, and no more
        (``PrepareMixin.prepare_state_caps_only``: the tree, the upward pass, the
        eager walk's caps, the seed; no list, no downward pass, no near-field
        payload). The template is then the SHAPES of a rebuild of the eager tree
        under ``jax.eval_shape`` -- what the scan's steps trace -- and the topology
        holds a :class:`~jaccpot.runtime.strict_carry.PreparedCaps` (the eager
        tree, the verdict) instead of a prepared state: nothing can be evaluated
        on it. The same template key, provider key and engine trace-time state as
        the full prepare (``tests/integration/strict/test_caps_only_prepare.py``).
        A lane without the flat walk gets the full prepare.

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
        caps_only : bool
            The caps-only prepare (above).
        differentiable : bool
            Also freeze the gradient plan's static fields on the eager topology
            (:meth:`with_grad_plan`), for :meth:`differentiable_step`. Needs the
            full prepare (its far list): not with ``caps_only``.

        Returns
        -------
        tuple[FusedForceProvider, FusedTopology]
            The provider and the topology at ``positions`` (``from_prepared``; a
            ``PreparedCaps`` under ``caps_only``).

        Raises
        ------
        RuntimeError
            If the profile is not large-N production, the particle count is not in
            ``JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET``, or the prepare did not
            return a ``LargeNPreparedState``.
        ValueError
            If ``differentiable`` comes with ``caps_only``.
        """
        if differentiable and caps_only:
            raise ValueError(
                "differentiable=True freezes the gradient plan on the eager "
                "topology's far list, which a caps-only prepare does not build: "
                "prepare in full (caps_only=False)"
            )
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
        prepared: Any = None
        if caps_only:
            prepared = engine.prepare_state_caps_only(
                positions_arr,
                masses_arr,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=theta,
                jit_tree=engine._jit_tree_default,
                runtime_overrides_override=runtime_overrides,
                **({} if bounds is None else {"bounds": bounds}),
            )
            if isinstance(prepared, PreparedCaps):
                provider = cls(
                    engine,
                    leaf_size=leaf_size,
                    max_order=max_order,
                    theta=theta,
                    runtime_overrides=runtime_overrides,
                )
                provider, prepared = provider._with_caps_template(
                    prepared, positions_arr, masses_arr
                )
                engine._record_large_n_eval_shape_diagnostics(prepared.shapes)
                return provider, FusedTopology.from_prepared(prepared)
        if prepared is None:
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
            hint = ""
            if jax.default_backend() == "cpu" and not bool(
                getattr(engine, "cpu_fused_lane", False)
            ):
                hint = (
                    ": the lane declines a CPU backend unless the solver takes its "
                    "CPU route, FMMAdvancedConfig(runtime=RuntimePolicyConfig("
                    "cpu_fused_lane=True)) (jaccpot.runtime._lane_platform)"
                )
            raise RuntimeError(
                "the fused lane's prepare returned a "
                f"{type(prepared).__name__}, not a LargeNPreparedState" + hint
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
        topology = FusedTopology.from_prepared(prepared)
        if differentiable:
            provider = provider.with_grad_plan(topology)
        return provider, topology

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

    @on_lane_platform
    def replanned(
        self,
        needs: Any,
        positions: Array,
        masses: Array,
        *,
        walk_only: bool = False,
        caps_only: bool = False,
        bounds: Optional[Tuple[Array, Array]] = None,
    ) -> Optional[tuple["FusedForceProvider", FusedTopology]]:
        """Re-plan from what a failed segment needed and re-prepare at its start.

        The segment retry. The WALK: the unnamed flat-walk caps grow to cover
        ``needs`` (``engine._replan_walk_caps_from_needs``; a far list the walk
        filled exactly grows by at least one pair), and the peak wavefront
        the segment needed is kept so the retraced walk's queue covers it. The
        TREE (cell leaves): the leaf capacity, the level-loop depth bound and the
        level loops' batch width grow to what the segment's trees needed
        (``engine._replan_tree_from_needs``; the width is registered for the shape
        the re-prepare builds). Either or both -- or, when neither overflowed, on
        the pure-JAX near field the rectangle layout's static target-block cap
        doubles (``engine._replan_rectangle_blocks``) --; then the lane is
        prepared again at the segment's START. A grown leaf capacity is a new
        template shape, so the re-run's programs have new keys and compile.

        Parameters
        ----------
        needs : Any
            The segment's running maximum of :meth:`FusedTopology.needs` (host).
        positions : Array
            ``[N, 3]`` positions the segment started from.
        masses : Array
            ``[N]`` masses.
        walk_only : bool
            Re-plan the walk caps only: the second retry after a tree re-plan,
            whose re-run measured the walk on a tree that fits.
        caps_only : bool
            Re-prepare caps-only (:meth:`prepare`): the particle carry's retry,
            whose re-run starts from the start program. The topology then holds
            a ``PreparedCaps``, on which nothing can be evaluated.
        bounds : Optional[Tuple[Array, Array]]
            The re-prepare's Morton box; ``None`` (the default) infers it from
            ``positions``. The escaper split passes its tree rows' box
            (:func:`jaccpot.escapers.live_bounds`).

        Returns
        -------
        Optional[tuple[FusedForceProvider, FusedTopology]]
            The provider and topology for the re-run, or ``None`` when nothing can
            be re-planned (neither a walk flag, a far list filled exactly, nor a
            tree overflow a bound can fix, and on the pure-JAX near field the
            rectangle layout's cap at its last rung; a named cap overflowed --
            then not even the tree --; not the flat walk): the caller raises.
        """
        from ._level_shapes import widen_registered_levels
        from .fused_topology import NEEDS_FIELDS

        engine = self.engine
        if engine._walk_overflowed_a_named_cap(needs):
            # a deliberate bound the re-run would overflow again, tree or not
            return None
        # both: a segment can outgrow the walk caps and the tree at once
        walk = engine._replan_walk_caps_from_needs(needs)
        tree = False if walk_only else engine._replan_tree_from_needs(needs)
        # the pure-JAX near field's rectangle layout, which no need records: when
        # neither the walk nor the tree overflowed, its rows are what failed
        # (`OVERFLOW_RECTANGLE`; the CSR lane never reads it)
        rectangle = (
            not (walk or tree)
            and self.rectangle_guard_active
            and engine._replan_rectangle_blocks()
        )
        if not (walk or tree or rectangle):
            return None
        replanned_peak = int(
            (engine._strict_fused_validated_caps or {}).get("peak_wavefront", 0)
        )
        prepare_kw: dict = dict(
            leaf_size=int(self.leaf_size),
            max_order=int(self.max_order),
            theta=self.theta,
            jit_tree=engine._jit_tree_default,
            runtime_overrides_override=self.runtime_overrides,
        )
        if bounds is not None:
            prepare_kw["bounds"] = bounds
        prepared: Any = (
            engine.prepare_state_caps_only(positions, masses, **prepare_kw)
            if caps_only
            else None
        )
        if prepared is None:
            prepared = engine.prepare_state(
                positions, masses, fused_device_mode=True, **prepare_kw
            )
        # the eager walk saw the segment's START; keep the wavefront the segment
        # needed, so the retraced walk's queue covers it
        validated = dict(engine._strict_fused_validated_caps or {})
        validated["peak_wavefront"] = max(
            int(validated.get("peak_wavefront") or 0), replanned_peak
        )
        engine._strict_fused_validated_caps = validated
        if tree:
            # the prepare registered the START tree's widest level and depth for
            # its shape; the re-run's rebuilds reach the segment's
            tree_now = prepared.tree
            widen_registered_levels(
                total_nodes=int(tree_now.parent.shape[0]),
                num_internal=int(tree_now.left_child.shape[0]),
                widest=int(needs[NEEDS_FIELDS.index("widest_needed")]),
                num_levels=min(
                    int(tree_now.level_offsets.shape[0]) - 1,
                    int(engine._cells_upward_num_levels),
                ),
            )
        if isinstance(prepared, PreparedCaps):
            provider, prepared = type(self)(
                engine,
                leaf_size=self.leaf_size,
                max_order=self.max_order,
                theta=self.theta,
                runtime_overrides=self.runtime_overrides,
            )._with_caps_template(prepared, positions, masses)
            return provider, FusedTopology.from_prepared(prepared)
        provider = type(self).freeze_template(
            engine,
            prepared,
            leaf_size=self.leaf_size,
            max_order=self.max_order,
            theta=self.theta,
            runtime_overrides=self.runtime_overrides,
        )
        return provider, FusedTopology.from_prepared(prepared)

    def retry_mode(self, needs: Any, retries: int, tree_first: bool) -> Optional[str]:
        """How a failed segment may be retried: ``strict_run_v2``'s policy.

        For a caller that drives its own segments over :meth:`step` (nornax) and
        retries them with :meth:`replanned`: the answer the lane's runners act on
        (``engine._segment_retry_mode``), so the policy has one source. At most
        two retries per segment: the first re-plans whatever overflowed, a second
        only the walk, and only after a first that re-planned the tree (the walk's
        needs on an overflowed tree can fall short).
        ``JACCPOT_STRICT_SEGMENT_RETRY=0`` turns it off.

        Parameters
        ----------
        needs : Any
            The failed run's running maximum of :meth:`FusedTopology.needs`
            (host).
        retries : int
            Retries this segment has run so far.
        tree_first : bool
            The segment's first failure was a tree overflow.

        Returns
        -------
        Optional[str]
            ``"full"`` (:meth:`replanned` with ``walk_only=False``), ``"walk"``
            (``walk_only=True``), or ``None``: the caller raises.
        """
        return self.engine._segment_retry_mode(
            np.asarray(needs).astype(np.int64), int(retries), bool(tree_first)
        )

    def _with_caps_template(
        self, caps: PreparedCaps, positions: Array, masses: Array
    ) -> tuple["FusedForceProvider", PreparedCaps]:
        """The template and verdict the full prepare would give, from a caps-only one.

        The rebuild of ``caps.stub`` (the eager tree and the state's static
        fields) at abstract positions under ``jax.eval_shape`` is the trace the
        particle carry's steps run, so its output's SHAPES are the prepared
        state's: the template the full prepare freezes, the far list detached.
        Tracing it leaves the engine's trace-time side channels (the traced walk
        caps, the verdict and needs a rebuild moves off the engine) as they were.

        The verdict is ``fused_state_capacity_ok`` of the state the full prepare
        would have built (no traced caps): its far list not full, and on the
        rectangle near-field route every row within the static target blocks --
        from the eager walk's counts and the template's block shape.

        Parameters
        ----------
        caps : PreparedCaps
            From ``prepare_state_caps_only``.
        positions : Array
            ``[N, 3]`` positions of the prepare (their shape and dtype).
        masses : Array
            ``[N]`` masses (their shape and dtype).

        Returns
        -------
        tuple[FusedForceProvider, PreparedCaps]
            This configuration with the template, and ``caps`` with the verdict
            and the shapes.
        """
        engine = self.engine
        n = int(positions.shape[0])
        x = jax.ShapeDtypeStruct(tuple(positions.shape), positions.dtype)
        m = jax.ShapeDtypeStruct(tuple(masses.shape), masses.dtype)
        # the scan's f_b carry: the state's dtype, input order
        fb = jax.ShapeDtypeStruct((n,), positions.dtype) if self.scale_carry else None

        # The state's leaves that are not arrays (a tree's particle count, an
        # expansion order: Python numbers the rebuild reads statically) stay what
        # they are, as the full prepare's template keeps them; eval_shape would
        # make them scalars. So only the arrays leave the trace.
        static: dict = {}
        array = object()  # marks an array leaf in ``static["leaves"]``

        def _shapes(like: Any, x_in: Any, m_in: Any, fb_in: Any) -> list:
            topology = self._rebuild_like(like, x_in, m_in, force_scale=fb_in)
            leaves, static["treedef"] = jax.tree_util.tree_flatten(
                replace(topology.prepared, compact_far_pairs=None)
            )
            is_array = [isinstance(leaf, (jax.Array, np.ndarray)) for leaf in leaves]
            static["leaves"] = [
                array if flag else leaf for leaf, flag in zip(leaves, is_array)
            ]
            return [leaf for leaf, flag in zip(leaves, is_array) if flag]

        saved = {name: getattr(engine, name, None) for name in _TRACE_STATE}
        try:
            arrays = iter(jax.eval_shape(_shapes, caps.stub, x, m, fb))
        finally:
            for name, value in saved.items():
                setattr(engine, name, value)
        # the array leaves as an eager array's template records them (no weak types)
        leaves = []
        for leaf in static["leaves"]:
            if leaf is array:
                sds = next(arrays)
                leaf = jax.ShapeDtypeStruct(tuple(sds.shape), sds.dtype)
            leaves.append(leaf)
        template = PreparedShapeTemplate(
            treedef=static["treedef"], leaves=tuple(leaves)
        )
        shapes = jax.tree_util.tree_unflatten(template.treedef, leaves)
        walk = caps.walk
        ok = int(walk.compact_far_pair_capacity) <= 0 or int(walk.far_pair_count) < int(
            walk.compact_far_pair_capacity
        )
        padded = getattr(shapes, "nearfield_target_block_source_leaf_ids_padded", None)
        if (
            ok
            and padded is not None
            and self.rectangle_guard_active
            and len(padded.shape) == 3
            and int(padded.shape[1]) > 0
        ):
            ok = int(walk.max_leaf_neighbors) <= int(padded.shape[1]) * int(
                padded.shape[2]
            )
        provider = type(self)(
            engine,
            leaf_size=self.leaf_size,
            max_order=self.max_order,
            theta=self.theta,
            runtime_overrides=self.runtime_overrides,
            template=template,
        )
        return provider, replace(caps, capacity_ok=bool(ok), shapes=shapes)

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

        from ._lane_platform import lane_platform_scope

        # the CPU route's interpret-mode CSR switches (a property: not an entry
        # point the decorator scopes, and read by callers' own verdicts)
        with lane_platform_scope(self.engine):
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
        validated walk caps the traced walk is sized from, the template's shapes
        when the rebuild materialises it, and a cubic Morton box (appended only
        when the box is a cube, so a per-axis key is the one it always was:
        ``JACCPOT_CUBIC_BOUNDS`` can flip the box between two calls of one
        engine).

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
        if _cubic_bounds_active(getattr(engine, "_tree_cubic_bounds", False)):
            parts = parts + ("cubic_bounds",)
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

    @on_lane_platform
    def rebuild(
        self,
        positions: Array,
        masses: Array,
        *,
        force_scale: Optional[Array] = None,
        bounds: Optional[Tuple[Array, Array]] = None,
        num_valid: Optional[Array] = None,
        cross_hook: Optional[Callable[[Any], Any]] = None,
        keep_far_list: bool = False,
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
        keep_far_list : bool
            Keep the far list the walk built in the topology
            (``prepared.compact_far_pairs``) instead of the template's placeholder:
            the gradient plan (:meth:`grad_plan`) reads it. ``False`` (default):
            the program as always.

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
            keep_far_list=keep_far_list,
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
        keep_far_list: bool = False,
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
        keep_far_list : bool
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
            keep_far_list=bool(keep_far_list),
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
            tree_needs=getattr(engine, "_last_refresh_tree_needs", None),
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
        engine._last_refresh_tree_needs = None
        return topo

    @on_lane_platform
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
            ``G`` and the softening squared (a :class:`StepParams`' force part).
            ``None`` or the engine's own as Python floats: the engine's literals,
            the program unchanged. Other floats are literals; arrays are traced
            (cast to the working dtype). See :meth:`params_overflow` for an
            ``eps2`` the walk was not built for.
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
            On an unknown ``order``, or parameters other than the engine's own
            under ``mac_type='dehnen_error'``.
        RuntimeError
            If the topology does not hold a ``LargeNPreparedState``.
        """
        force = self.resolve_params(params)
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
        if force is not None:
            extra["force_params"] = force
        return evaluate_large_n_state(
            self.engine,
            prepared,
            target_indices=None,
            return_potential=False,
            max_acc_derivative_order=0,
            **extra,
        )

    @on_lane_platform
    def potential(
        self,
        topology: FusedTopology,
        params: Optional[Union[ForceParams, StepParams]] = None,
        *,
        order: str = "input",
    ) -> Array:
        """The self-gravity potential on ``topology``: near lane 4, far L2P, scatter.

        ``phi_i = -G sum_{j != i} m_j psi(r_ij)`` with the force's softening kernel
        (Newtonian beyond its support), the far field from the same local
        expansions the accelerations read (``evaluate_large_n_potential``). O(N), in
        a program of its own: no acceleration program changes. The self-gravity's
        energy is ``W = 1/2 sum_i m_i phi_i``.

        Parameters
        ----------
        topology : FusedTopology
            From :meth:`rebuild` (or :meth:`FusedTopology.from_prepared`).
        params : Optional[Union[ForceParams, StepParams]]
            As in :meth:`accelerations`. Arrays reach the near field only on the
            fused Pallas lane (the CSR direct kernel); the generic lane takes Python
            floats.
        order : str
            ``"input"`` or ``"tree"``, as in :meth:`accelerations`.

        Returns
        -------
        Array
            ``[N]`` in the state's input dtype.

        Raises
        ------
        ValueError
            On an unknown ``order``, or parameters other than the engine's own
            under ``mac_type='dehnen_error'``.
        RuntimeError
            If the topology does not hold a ``LargeNPreparedState``.
        """
        force = self.resolve_params(params)
        if order not in ("input", "tree"):
            raise ValueError(f"order must be 'input' or 'tree', got {order!r}")
        prepared = topology.prepared
        if not isinstance(prepared, LargeNPreparedState):
            raise RuntimeError(
                "the fused lane's potential needs a LargeNPreparedState, got "
                f"{type(prepared).__name__}"
            )
        return evaluate_large_n_potential(
            self.engine,
            prepared,
            sorted_output=order == "tree",
            force_params=force,
        )

    @on_lane_platform
    def capacity_ok(self, topology: FusedTopology) -> Array:
        """The step's capacity verdict, as the fused scan folds it into its carry.

        Every fixed capacity of the returned state (``fused_state_capacity_ok``
        with the engine's traced caps, read NOW -- inside the trace that recorded
        them) and, for a rebuilt topology, the far list the rebuild built and the
        cell-leaf tree it rebuilt (leaf capacity, depth bound, level width: the
        walk never sees that overflow, so it is folded here). Equal to
        ``overflow_ok(topology.overflow_bits())``. A caps-only prepare's topology
        (a ``PreparedCaps``) has its eager verdict.

        Parameters
        ----------
        topology : FusedTopology
            A rebuilt topology, or an eager one (``fresh=False``: its own widths,
            or a caps-only prepare's verdict).

        Returns
        -------
        Array
            Boolean scalar.
        """
        from jaccpot.runtime.capacity_guard import fused_state_capacity_ok

        if isinstance(topology.prepared, PreparedCaps):
            # a caps-only prepare's: the verdict the full prepare's state gets
            return jnp.asarray(bool(topology.prepared.capacity_ok))
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
            if topology.tree_overflow is not None:
                ok = ok & ~jnp.asarray(topology.tree_overflow, jnp.bool_)
        return ok

    @on_lane_platform
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
            the criterion), the overflow bits (``OVERFLOW_PARAMS`` included) OR-ed
            and the walk needs max-ed over the steps ``aux`` has seen, up to and
            including the first whose verdict failed.
        """
        f_b_in, bits_in, needs_in = (None, None, None) if aux is None else aux
        topology = self.rebuild(positions, masses, force_scale=f_b_in)
        if self.scale_carry:
            acc, f_b = self.accelerations(topology, params, force_scale=True)
        else:
            acc, f_b = self.accelerations(topology, params), None
        bits = topology.overflow_bits(
            extra=0 if params is None else self.params_overflow(params)
        )
        bits, needs = _fold_verdict(bits, topology.needs(), bits_in, needs_in)
        return acc, (f_b, bits, needs)

    # ----------------------------------------------------------------------
    # Differentiable: the gradient plan and the differentiable step (S9).
    # ----------------------------------------------------------------------

    @on_lane_platform
    def with_grad_plan(self, topology: FusedTopology) -> "FusedForceProvider":
        """This provider with the gradient plan's static fields frozen.

        Eager, once: S8's plan checks (``prepare_large_n_grad_plan``: the radix fast
        lane, the solidfmm basis, the near field's prepacked layout with no overflow
        or target-block payload, the near and far lists symmetric) on an eager
        topology of THIS configuration -- :meth:`prepare`'s, which keeps its far
        list -- and the plan's static fields (order, leaf capacity, centre mode,
        chunk sizes, the far list's capacity and layout) kept. Every
        :meth:`grad_plan` checks a traced topology against them. A re-planned
        provider (:meth:`replanned`) has none: freeze them again on its eager
        topology.

        Parameters
        ----------
        topology : FusedTopology
            An eager topology holding a ``LargeNPreparedState`` with its far list
            (:meth:`prepare`, not ``caps_only``).

        Returns
        -------
        FusedForceProvider
            The same configuration and template, with ``grad_statics``.

        Raises
        ------
        ValueError
            If the topology holds no prepared state with a far list (a caps-only
            prepare's, or a traced rebuild's).
        """
        from ._large_n_grad import freeze_large_n_grad_statics

        prepared = topology.prepared
        if (
            not isinstance(prepared, LargeNPreparedState)
            or getattr(prepared, "compact_far_pairs", None) is None
        ):
            raise ValueError(
                "with_grad_plan needs an eager topology with its far list "
                "(FusedForceProvider.prepare, caps_only=False); got a "
                f"{type(prepared).__name__} without one"
            )
        provider = type(self)(
            self.engine,
            leaf_size=self.leaf_size,
            max_order=self.max_order,
            theta=self.theta,
            runtime_overrides=self.runtime_overrides,
            template=self.template,
            grad_statics=freeze_large_n_grad_statics(self.engine, prepared),
        )
        return provider

    @on_lane_platform
    def grad_plan(self, topology: FusedTopology) -> tuple["LargeNGradPlan", Array]:
        """The gradient plan of a traced topology, and its lists' symmetry verdict.

        Traced. The static fields are the frozen ones (:meth:`with_grad_plan`),
        checked against this topology's (a mismatch raises at trace time); the
        data are the far list its rebuild kept (``rebuild(keep_far_list=True)``).
        The symmetry the reverses read each pair from both ends by -- the near
        list always, the far list on the sort-free M2L reverse -- is checked on the
        device (S8 checked it on the host), so a scan can fold it into its bits.

        Parameters
        ----------
        topology : FusedTopology
            From ``rebuild(..., keep_far_list=True)``.

        Returns
        -------
        tuple[LargeNGradPlan, Array]
            The plan and a boolean scalar (``False``: the gradient on these lists
            would be wrong; :meth:`differentiable_step` sets
            ``OVERFLOW_GRAD_LISTS`` and makes it NaN). A provider without frozen
            statics raises (:meth:`_grad_statics_or_raise`).
        """
        from ._large_n_grad import traced_large_n_grad_plan

        return traced_large_n_grad_plan(
            self.engine, topology.prepared, self._grad_statics_or_raise()
        )

    def _grad_statics_or_raise(self) -> "LargeNGradStatics":
        """The frozen gradient-plan statics, or the reason there are none.

        Returns
        -------
        LargeNGradStatics
            :attr:`grad_statics`.

        Raises
        ------
        ValueError
            If the provider was made for the forward only.
        """
        if self.grad_statics is None:
            raise ValueError(
                "this provider has no gradient plan: make it with "
                "FusedForceProvider.prepare(..., differentiable=True) or "
                "provider.with_grad_plan(eager_topology)"
            )
        return self.grad_statics

    @on_lane_platform
    def differentiable_accelerations(
        self,
        topology: FusedTopology,
        positions: Array,
        masses: Array,
        params: Optional[Union[ForceParams, StepParams]] = None,
        *,
        plan: "LargeNGradPlan",
        force_scale: bool = False,
        cross: Any = None,
    ) -> Any:
        """The self-gravity on ``topology`` at LIVE positions and masses.

        S8's differentiable evaluation (``_differentiable_accelerations_large_n``):
        P2M, M2M, M2L, L2L and L2P recomputed from ``positions`` and ``masses`` on
        the topology's lists (``plan``), the near field's direct kernel through its
        ``custom_vjp`` -- the production kernels, each with its reverse. At the
        positions and masses the topology was rebuilt from it is
        :meth:`accelerations` bitwise. Unlike :meth:`accelerations` it reads the
        positions: the gradient reaches the far field's source side.

        Parameters
        ----------
        topology : FusedTopology
            From ``rebuild(..., keep_far_list=True)``.
        positions : Array
            ``[N, 3]`` positions, input order (differentiated).
        masses : Array
            ``[N]`` masses, input order (differentiated).
        params : Optional[Union[ForceParams, StepParams]]
            As in :meth:`accelerations`; arrays are differentiated.
        plan : LargeNGradPlan
            :meth:`grad_plan`'s plan of this topology (fold its verdict too).
        force_scale : bool
            Also return eq (16b)'s f_b per particle (input order), as
            :meth:`accelerations` does: the near kernel's force-scale lane plus the
            rebuild's far half; not differentiated.
        cross : Any
            On a multi-GPU device: the cross far sources its rebuild's hook
            imported (:class:`~jaccpot.runtime._large_n_grad.LargeNCrossFar`), for
            the far chain -- the cross near term is the caller's
            (:func:`jaccpot.distributed.grad.mesh_differentiable_force_shard`).
            ``None`` (default): one domain.

        Returns
        -------
        Any
            ``[N, 3]`` accelerations in the state's input dtype, or ``(acc, f_b)``.

        Raises
        ------
        RuntimeError
            If the topology does not hold a ``LargeNPreparedState``.
        """
        from .grad_options import grad_option_overrides, resolve_grad_options

        force = self.resolve_params(params)
        prepared = topology.prepared
        if not isinstance(prepared, LargeNPreparedState):
            raise RuntimeError(
                "the differentiable fused force needs a LargeNPreparedState, got "
                f"{type(prepared).__name__}"
            )
        options = resolve_grad_options(
            None,
            num_particles=int(prepared.positions_sorted.shape[0]),
            supports_fast_lane=False,
        )
        with grad_option_overrides(options):
            return self.engine._differentiable_accelerations_large_n(
                prepared,
                positions,
                masses,
                target_indices=None,
                grad_plan=plan,
                options=options,
                force_params=force,
                return_force_scale=bool(force_scale),
                cross=cross,
            )

    @on_lane_platform
    def differentiable_step(
        self,
        positions: Array,
        masses: Array,
        params: Optional[Union[ForceParams, StepParams]] = None,
        aux: Optional[tuple] = None,
    ) -> tuple[Any, tuple]:
        """:meth:`step`, differentiable in ``positions``, ``masses`` and ``params``.

        The same signature, forward and aux. The topology stage runs on the
        INPUTS' stopped gradients -- ``rebuild(stop_gradient(x), stop_gradient(m),
        force_scale=stop_gradient(f_b))`` -- so nothing of the tree, the walk or
        the level loops is linearized (their while loops have no reverse); the
        rebuild keeps its far list, and the force is S8's differentiable
        evaluation on that topology at the LIVE ``positions`` and ``masses``
        (:meth:`differentiable_accelerations`). Differentiating
        ``accelerations(rebuild(stop_gradient(x)))`` instead gives a gradient of
        exactly zero (the rebuild ran the far field; ``accelerations`` reads no
        positions): this API never offers that form.

        * the forward is :meth:`step`'s bitwise: the same rebuild, and S8's
          evaluation is the production one at the rebuild's own inputs (the far
          chain runs twice, once in the rebuild and once here: v1, by decision);
        * ``mac_type='dehnen_error'``: f_b is the production one (the near kernel's
          force-scale lane in the same call, the rebuild's far half), its
          gradient stopped -- it sets the next walk's thresholds;
        * the bits and needs fold as :meth:`step`'s, plus ``OVERFLOW_GRAD_LISTS``
          when this step's lists are not symmetric (:meth:`grad_plan`'s device
          verdict). That bit is outside ``overflow_ok`` (the force is right) and
          inside ``grad_ok``: check ``grad_ok(bits)`` before using a gradient.
          On such lists the gradient is also made NaN, so it is never a plausible
          wrong number.

        Needs the gradient plan's statics (:meth:`with_grad_plan`, or
        ``prepare(..., differentiable=True)``).

        Parameters
        ----------
        positions : Array
            ``[N, 3]`` positions.
        masses : Array
            ``[N]`` masses.
        params : Optional[Union[ForceParams, StepParams]]
            As in :meth:`step`; array ``G`` and ``eps2`` are differentiated.
        aux : Optional[tuple]
            As in :meth:`step`.

        Returns
        -------
        tuple[Any, tuple]
            ``(acc, (f_b, bits, needs))`` as :meth:`step` returns them.
        """
        from .grad_options import forward_mode_enabled

        self._grad_statics_or_raise()  # before the rebuild's trace, not after
        f_b_in, bits_in, needs_in = (None, None, None) if aux is None else aux
        stop = jax.lax.stop_gradient
        topology = self.rebuild(
            stop(positions),
            stop(masses),
            force_scale=None if f_b_in is None else stop(f_b_in),
            keep_far_list=True,
        )
        plan, symmetric = self.grad_plan(topology)
        if self.scale_carry:
            acc, f_b = self.differentiable_accelerations(
                topology, positions, masses, params, plan=plan, force_scale=True
            )
            f_b = stop(f_b)
        else:
            acc = self.differentiable_accelerations(
                topology, positions, masses, params, plan=plan
            )
            f_b = None
        if forward_mode_enabled():
            return self._forward_gated(acc, f_b, topology, params, bits_in, needs_in)
        acc = _gradient_gate(acc, symmetric)
        # the verdict as `step` computes it: on the template's placeholder far list
        # (the kept one would add checks of its own to `overflow_bits`)
        bits = replace(
            topology, prepared=replace(topology.prepared, compact_far_pairs=None)
        ).overflow_bits(extra=0 if params is None else self.params_overflow(params))
        bits = bits | jnp.where(
            symmetric, jnp.uint32(0), jnp.uint32(OVERFLOW_GRAD_LISTS)
        )
        bits, needs = _fold_verdict(bits, topology.needs(), bits_in, needs_in)
        return acc, (f_b, bits, needs)

    def _forward_gated(
        self,
        acc: Array,
        f_b: Any,
        topology: FusedTopology,
        params: Optional[Union[ForceParams, StepParams]],
        bits_in: Any,
        needs_in: Any,
    ) -> tuple[Any, tuple]:
        """:meth:`differentiable_step`'s verdict and gate in forward mode.

        ``ad_mode="forward"``: the tangent kernels are target-driven and read no
        list from both ends, so the symmetry verdict (``OVERFLOW_GRAD_LISTS``) is
        neither needed nor set -- the bits are :meth:`step`'s. The tangent is made
        NaN when THIS step's capacity verdict fails
        (:func:`~jaccpot.runtime._forward_seams.tangent_gate`): a truncated list's
        derivative is never a plausible wrong number.

        Parameters
        ----------
        acc : Array
            The step's accelerations.
        f_b : Any
            Its f_b (gradient stopped), or ``None``.
        topology : FusedTopology
            The step's rebuilt topology.
        params : Optional[Union[ForceParams, StepParams]]
            As in :meth:`step`.
        bits_in : Any
            The bits ``aux`` carried, or ``None``.
        needs_in : Any
            The needs ``aux`` carried, or ``None``.

        Returns
        -------
        tuple[Any, tuple]
            ``(acc, (f_b, bits, needs))``.
        """
        from ._forward_seams import tangent_gate

        bits = replace(
            topology, prepared=replace(topology.prepared, compact_far_pairs=None)
        ).overflow_bits(extra=0 if params is None else self.params_overflow(params))
        acc = tangent_gate(acc, overflow_ok(bits))
        bits, needs = _fold_verdict(bits, topology.needs(), bits_in, needs_in)
        return acc, (f_b, bits, needs)

    # ----------------------------------------------------------------------
    # Helpers.
    # ----------------------------------------------------------------------

    def walk_eps2(self) -> Optional[float]:
        """The largest softening squared the rebuild's walk is correct for.

        A compact kernel is exactly Newtonian beyond its support ``h = k eps``,
        and the walk keeps every pair closer than its separation floor (the
        engine's ``softening_floor`` x softening; ``k`` x softening unset) out of
        the far field: so it covers ``eps <= floor / k``, and at least the
        engine's own softening (a floor set below ``k eps`` is the caller's
        choice). Plummer has no support and no floor.

        Returns
        -------
        Optional[float]
            ``max((floor / k)^2, softening^2)``; ``None`` for a kernel without
            compact support.
        """
        engine = self.engine
        k = float(support_factor(getattr(engine, "softening_kernel", "plummer")))
        if k <= 0.0:
            return None
        floor = float(engine._walk_separation_floor())
        return max((floor / k) ** 2, float(engine.softening) ** 2)

    def params_overflow(
        self, params: Optional[Union[ForceParams, StepParams]]
    ) -> Array:
        """``OVERFLOW_PARAMS`` when ``params``' softening is beyond the walk's.

        The walk is built for the engine's softening (its separation floor is
        static); a compact kernel's support grows with ``eps2``, so a larger one
        reaches pairs the far field took unsoftened.

        Parameters
        ----------
        params : Optional[Union[ForceParams, StepParams]]
            As in :meth:`accelerations`.

        Returns
        -------
        Array
            ``uint32`` scalar: ``OVERFLOW_PARAMS`` or ``0`` (a tracer for a traced
            ``eps2``).
        """
        force = self.resolve_params(params)
        limit = self.walk_eps2()
        if force is None or limit is None:
            return jnp.uint32(0)
        if _is_static(force.eps2):
            return jnp.uint32(OVERFLOW_PARAMS if float(force.eps2) > limit else 0)
        eps2 = jnp.asarray(force.eps2)
        return jnp.where(
            eps2 > jnp.asarray(limit, eps2.dtype),
            jnp.uint32(OVERFLOW_PARAMS),
            jnp.uint32(0),
        )

    def resolve_params(
        self, params: Optional[Union[ForceParams, StepParams]]
    ) -> Optional[ForceParams]:
        """The force parameters to evaluate with; ``None`` for the engine's own.

        Needs no template: ``strict_run_v2`` checks a call's parameters with a
        provider of the call's configuration before it prepares anything.

        Parameters
        ----------
        params : Optional[Union[ForceParams, StepParams]]
            As in :meth:`accelerations`.

        Returns
        -------
        Optional[ForceParams]
            ``None`` when nothing differs from the engine's literals (the default
            program), else the parameters with their arrays in the working dtype.

        Raises
        ------
        ValueError
            For anything but the engine's own under ``mac_type='dehnen_error'``.
        """
        if params is None:
            return None
        force = params.force if isinstance(params, StepParams) else params
        own = self.params()
        if (
            force.is_static()
            and float(force.G) == own.G
            and float(force.eps2) == own.eps2
        ):
            return None
        if self.scale_carry:
            raise ValueError(
                "mac_type='dehnen_error' takes the solver's own G and softening "
                f"only (G={own.G}, eps2={own.eps2}): eq (16a)'s thresholds and the "
                "far half of its force scale are built with them in the rebuild. "
                "Traced or other force parameters need another MAC."
            )
        dtype = self.engine.working_dtype

        def cast(value: Any) -> Any:
            return value if _is_static(value) else jnp.asarray(value, dtype)

        return ForceParams(G=cast(force.G), eps2=cast(force.eps2))
