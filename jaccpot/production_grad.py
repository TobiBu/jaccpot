"""The production runner's gradient mode (S9 session 2).

:meth:`ProductionRunner.gradient_plan <jaccpot.production.ProductionRunner.gradient_plan>`
differentiates a multi-segment production rollout in two passes:

1. **The forward pre-pass** is :meth:`ProductionRunner.run
   <jaccpot.production.ProductionRunner.run>` itself -- its segments, its segment
   retries and re-plans, its callbacks and restart files -- with a recorder that
   keeps, per segment, what the segment's scan ran with: the plan (leaf capacity,
   walk caps, level registry, depth bound), the provider (its shape template), and
   how its start acceleration and first f_b were made (``strict_run_v2``'s
   ``_strict_particle_last_start``: carried over from the previous segment, or
   evaluated again after a re-plan, from which seed). Each segment's provider gets
   the gradient plan's statics frozen (:meth:`FusedForceProvider.with_grad_plan
   <jaccpot.FusedForceProvider.with_grad_plan>`) on a rebuild at the segment's start
   under that segment's plan. The result is a :class:`GradientPlan`: one
   :class:`GradSegment` per segment.
2. **The gradient run** (:meth:`GradientPlan.rollout`) is a chain of
   :func:`~jaccpot.runtime.grad_rollout.differentiable_rollout` calls over each
   segment's ``differentiable_step``. Between segments only STATICS change -- the
   provider, and the engine's trace-time plan, installed while that segment traces
   -- so the chain is one pure function of ``(positions, velocities, masses, dt,
   force_params, external_params)``, and :meth:`GradientPlan.value_and_grad`
   differentiates it in ONE jitted program.

THE BOUNDARIES are the runner's, so the gradient run's forward is the runner's
trajectory bit for bit (``tests/integration/strict/test_runner_gradient_mode.py``):

* ``"carried"`` (no retry): the previous segment's total acceleration and aux
  (``carry_total_acceleration``; the rollout's ``acceleration=`` / ``aux=``);
* ``"start"`` (a fresh run's first segment, or a retry whose start overflowed): the
  start force at the start positions from the recorded seed, its f_b feeding the
  first step;
* ``"seed_kept"`` (a retry of a segment with a handle, no field; the caps-only
  re-prepare): the start force from the re-prepare's seed, and the SAME seed for
  the first step's thresholds (the runner discards the start force's f_b);
* ``"self"`` (a retry with a field): the previous segment's last SELF-gravity -- the
  chain carries it out of that segment in the aux -- plus the field at the start
  state, and the re-prepare's seed for the first step.
* ``"eager_self"`` (``JACCPOT_STRICT_CAPS_ONLY_PREPARE=0``: a full re-prepare's
  evaluation) is chained as ``"seed_kept"`` and not gated bitwise.

The chain evaluates every start force itself (under ``jax.checkpoint``, outside
the rollouts' hand-written reverse) and adds the field as the runner does, EAGERLY: behind an
``optimization_barrier``, so XLA cannot contract the field's last multiply into the
add (an FMA). That makes a one-multiply field (the tests' harmonic pull) bitwise at
a ``"self"`` boundary. A field with more arithmetic is not always bitwise there or
at step 0: XLA contracts multiply-adds INSIDE the field in a program and not op by
op, so the runner's eager field and any jitted one differ by an ulp (the NFW halo,
CPU float64, N = 512: 264 of 1536 components at step 0; after 4-6 steps 9-26 of
3072 state elements, 5.6e-17). Inside the scans the field is jitted on both sides.

A failed verdict inside the gradient run -- the pre-pass's caps did not hold, e.g.
for inputs moved away from the pre-pass's -- is refused
(:class:`~jaccpot.runtime.grad_rollout.RolloutVerdictError`), never returned.

RESTARTS. A plan made with ``restart_from`` (a restart file or a
:class:`~jaccpot.production.RunResult`) starts from that state, and its first
segment from the restored handle's total acceleration, self-gravity and f_b. Those
are data, not functions of the restart state, so the start acceleration's VALUE is
the file's (the forward stays the runner's) while its TANGENT is the force
recomputed at the restart positions (:func:`_keep_value`; one more force in the
program, checkpointed). The gradient with respect to the restart state is
therefore the physical one -- what the uninterrupted run's reverse pass carries
across that edge. Measured (CPU, N = 512, float64, a restart after the first of two
2-step segments, pulled back through the first segment): with the geometric MAC
1.8e-16 of the largest component (round-off: the reverse sums are grouped
otherwise); under ``dehnen_error`` 3.2e-8 (positions), 6.1e-8 (masses), 1.8e-10
(velocities), because the recompute's thresholds come from the f_b the restart
carries -- the force's own output at that state -- not from the step before's,
which no restart file holds: at the restart positions the two pick other pairs
(the near list, and 2944 against 2946 far pairs), so the edge's force is
linearised on neighbouring lists (an FMM-accuracy difference, not round-off).

THE ENGINE'S TRACE-TIME STATE. The traced rebuild reads the walk caps, the leaf
capacity, the depth bound and the level registry when it TRACES. The chain installs
each segment's plan before that segment's trace and leaves the engine as it found
it (:func:`jaccpot.production._trace_state_kept`). Reverse rules are traced after the
forward, with the LAST segment's plan installed (:meth:`GradientPlan.value_and_grad`):
the plans only grow over a run, and no reverse kernel reads the walk caps.

Single card only (``mesh=None``). The module imports jax at import time; the
runner imports it lazily.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Callable

from jaccpot.runtime.fused_topology import NEEDS_FIELDS
from jaccpot.runtime.grad_rollout import (
    RolloutResult,
    check_rollout,
    differentiable_rollout,
)

__all__ = [
    "GRADIENT_INPUTS",
    "START_KINDS",
    "GradSegment",
    "GradientPlan",
    "GradientResult",
]

#: what :meth:`GradientPlan.value_and_grad` can differentiate, by name
GRADIENT_INPUTS = (
    "positions",
    "velocities",
    "masses",
    "dt",
    "force_params",
    "external_params",
)

#: how a segment's start acceleration and first f_b are made (module docstring)
START_KINDS = ("start", "carried", "seed_kept", "self", "eager_self")


# --------------------------------------------------------------------------
# The plan a segment traced with.
# --------------------------------------------------------------------------


def _plan_snapshot(engine: Any) -> dict:
    """The engine's trace-time plan now, as objects (not JSON).

    What :func:`jaccpot.production._plan_record` records for a restart file, kept
    exactly: the leaf capacity, a copy of the validated walk caps, the level
    registry and the depth bound.

    Parameters
    ----------
    engine : Any
        The runtime engine.

    Returns
    -------
    dict
        ``leaf_capacity``, ``walk_caps``, ``level_widths``, ``level_counts``,
        ``upward_num_levels``.
    """
    from jaccpot.runtime import _level_shapes

    caps = getattr(engine, "_strict_fused_validated_caps", None)
    return {
        "leaf_capacity": getattr(engine, "_tree_leaf_capacity", None),
        "walk_caps": None if caps is None else copy.deepcopy(dict(caps)),
        "level_widths": dict(_level_shapes._WIDTHS),
        "level_counts": dict(_level_shapes._LEVELS),
        "upward_num_levels": getattr(engine, "_cells_upward_num_levels", None),
    }


def _install_plan(engine: Any, plan: dict) -> None:
    """Make ``plan`` (:func:`_plan_snapshot`) the engine's trace-time plan.

    The level registry's entries for the plan's shapes are set to the plan's
    values; entries of other shapes stay (no rebuild of this plan reads them).

    Parameters
    ----------
    engine : Any
        The runtime engine.
    plan : dict
        A snapshot.
    """
    from jaccpot.runtime import _level_shapes

    if plan["leaf_capacity"] is not None:
        engine._tree_leaf_capacity = int(plan["leaf_capacity"])
    if plan["walk_caps"] is not None:
        engine._strict_fused_validated_caps = copy.deepcopy(plan["walk_caps"])
    _level_shapes._WIDTHS.update(plan["level_widths"])
    _level_shapes._LEVELS.update(plan["level_counts"])
    if plan["upward_num_levels"] is not None:
        engine._cells_upward_num_levels = int(plan["upward_num_levels"])


def _host(array: Any) -> Optional[np.ndarray]:
    """A host copy of a device array (``None`` stays ``None``).

    Parameters
    ----------
    array : Any
        A device array or ``None``.

    Returns
    -------
    Optional[np.ndarray]
        The copy.
    """
    return None if array is None else np.array(jax.device_get(array))


# --------------------------------------------------------------------------
# The plan's parts.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GradSegment:
    """One segment of a :class:`GradientPlan`: what its scan ran with.

    Attributes
    ----------
    step : int
        The step the segment starts at (counted from the run's ``t0``).
    steps : int
        Its length.
    kind : str
        How its start acceleration and first f_b were made, one of
        :data:`START_KINDS` (module docstring).
    seed : Optional[np.ndarray]
        ``[N]`` f_b (input order, host) the segment's start force or first step
        used, for every kind but ``"carried"``; ``None`` without
        ``mac_type='dehnen_error'``.
    retries : int
        The segment retries the pre-pass needed for it (``0``: its plan is the
        previous segment's).
    provider : Any
        The :class:`~jaccpot.FusedForceProvider` its scan used, with the gradient
        plan's statics frozen.
    plan : dict
        The engine's trace-time plan its scan traced with (:func:`_plan_snapshot`).
    """

    step: int
    steps: int
    kind: str
    seed: Optional[np.ndarray]
    retries: int
    provider: Any
    plan: dict


class GradientResult(NamedTuple):
    """What :meth:`GradientPlan.value_and_grad` returns.

    Attributes
    ----------
    value : Any
        The loss.
    grads : dict
        Its gradient with respect to each input named in ``wrt``, by name
        (:data:`GRADIENT_INPUTS`); a pytree input (``force_params``,
        ``external_params``) gets a gradient of its structure.
    result : RolloutResult
        The gradient run's final state; ``aux`` is ``(f_b, bits, needs)`` with the
        bits OR-ed and the needs max-ed over every segment.
    plan : Any
        The :class:`GradientPlan` it ran (reuse it: its programs are compiled).
    """

    value: Any
    grads: dict
    result: RolloutResult
    plan: Any


# --------------------------------------------------------------------------
# The chain's pieces.
# --------------------------------------------------------------------------


@jax.custom_vjp
def _keep_value(value: Any, recomputed: Any) -> Any:
    """``value``, differentiated as if it were ``recomputed``.

    A restart's start acceleration: the file's VALUE (so the forward is the
    runner's), with the TANGENT of the force recomputed at the restart positions
    (so the gradient reaches the restart state through it). ``value`` gets no
    cotangent.

    Parameters
    ----------
    value : Any
        What the forward returns.
    recomputed : Any
        Same shape and dtype; what the reverse pass differentiates.

    Returns
    -------
    Any
        ``value``.
    """
    del recomputed
    return value


def _keep_value_fwd(value: Any, recomputed: Any) -> tuple:
    """Forward of :func:`_keep_value`.

    Parameters
    ----------
    value : Any
        Returned.
    recomputed : Any
        Unused.

    Returns
    -------
    tuple
        ``(value, None)``.
    """
    del recomputed
    return value, None


def _keep_value_bwd(residual: Any, cotangent: Any) -> tuple:
    """Reverse of :func:`_keep_value`: the cotangent goes to ``recomputed``.

    Parameters
    ----------
    residual : Any
        ``None``: nothing is saved.
    cotangent : Any
        The output's cotangent.

    Returns
    -------
    tuple
        ``(None, cotangent)``: a SYMBOLIC zero for ``value``, so a value read
        from a rollout's aux (a carried self-gravity) gives that rollout's aux no
        cotangent and its guard (``grad_rollout._refuse_aux_cotangent``) stays
        quiet; zeros as an array would read as a gradient through the aux.
    """
    del residual
    return None, cotangent


_keep_value.defvjp(_keep_value_fwd, _keep_value_bwd)


def _carrying_self(step_fn: Callable[..., tuple]) -> Callable[..., tuple]:
    """``step_fn`` with its acceleration and its input aux also carried in the aux.

    For the segment BEFORE a ``"self"`` boundary: the runner's retry there starts
    from the handle's self-gravity, the last step's, which the rollout's result
    (the total acceleration) does not hold. The aux becomes ``(aux, acc,
    aux_in)``; the values are ``step_fn``'s. The rollout does not differentiate
    its aux under ``"step"`` / ``"nested"`` (``grad_rollout._adjoint_rollout``),
    so there the carried acceleration is a VALUE only: its tangent is the force
    recomputed from the last step's inputs, the end positions and ``aux_in``
    (:meth:`GradientPlan.rollout`).

    Parameters
    ----------
    step_fn : Callable[..., tuple]
        A provider's ``differentiable_step``.

    Returns
    -------
    Callable[..., tuple]
        ``step(x, m, params, (aux, acc_prev, aux_prev)) -> (acc, (aux_out, acc,
        aux))``.
    """

    def step(x: Any, m: Any, params: Any, aux: Any) -> tuple:
        acc, inner = step_fn(x, m, params, aux[0])
        return acc, (inner, acc, aux[0])

    return step


def _under_plan(
    engine: Any, plan: dict, step_fn: Callable[..., tuple], in_chain: list
) -> Callable[..., tuple]:
    """``step_fn`` that traces under its segment's ``plan`` wherever it is traced.

    The chain traces each segment's forward with that segment's plan installed
    (:meth:`GradientPlan.rollout`). The rollout's reverse under ``"step"`` /
    ``"nested"`` is hand-written (``grad_rollout._adjoint_rollout``, a
    ``custom_vjp``) and traces the step AGAIN when the reverse pass is built: after
    the chain's loop, when the last segment's plan -- or, past the loop's
    ``_trace_state_kept``, the caller's -- is the engine's. A rebuild traced there
    takes another plan's caps and the step's frozen grad-plan statics refuse it.
    Outside the loop (``in_chain[0]`` false) the step therefore installs its plan
    for the call and leaves the engine as it found it; inside, it traces exactly
    as before.

    Parameters
    ----------
    engine : Any
        The runtime engine.
    plan : dict
        The segment's plan (:func:`_plan_snapshot`).
    step_fn : Callable[..., tuple]
        The segment's step.
    in_chain : list
        One flag: ``True`` while the chain's forward loop traces.

    Returns
    -------
    Callable[..., tuple]
        ``step_fn`` with the same signature.
    """

    def step(x: Any, m: Any, params: Any, aux: Any) -> tuple:
        if in_chain[0]:
            return step_fn(x, m, params, aux)
        from jaccpot.production import _trace_state_kept

        with _trace_state_kept(engine):
            _install_plan(engine, plan)
            return step_fn(x, m, params, aux)

    return step


def _field_at(
    field: Optional[Callable[..., Any]], args: Any, x: Any, v: Any
) -> Optional[Any]:
    """The external field at ``(x, v)``, as ``strict_run_v2``'s start total adds it.

    Parameters
    ----------
    field : Optional[Callable[..., Any]]
        ``field(rows)`` or ``field(rows, args)``.
    args : Any
        Its parameters, or ``None``.
    x : Any
        ``[N, 3]``.
    v : Any
        ``[N, 3]``.

    Returns
    -------
    Optional[Any]
        ``[N, 3]`` in ``x``'s dtype, or ``None``.
    """
    if field is None:
        return None
    rows = jnp.stack([x, v], axis=1)
    out = field(rows) if args is None else field(rows, args)
    return jnp.asarray(out, dtype=x.dtype)


# --------------------------------------------------------------------------
# The plan.
# --------------------------------------------------------------------------


@dataclass
class GradientPlan:
    """A production rollout's settled segments, and its gradient run.

    Made by :meth:`ProductionRunner.gradient_plan
    <jaccpot.production.ProductionRunner.gradient_plan>` (the forward pre-pass);
    module docstring for the design.

    Attributes
    ----------
    engine : Any
        The runtime engine.
    segments : tuple
        One :class:`GradSegment` per segment of the pre-pass, in order.
    start : dict
        The pre-pass's inputs, host copies: ``positions``, ``velocities``,
        ``masses``.
    params : dict
        ``dt``, ``force_params``, ``external_params`` as the runner passed them to
        its segments (arrays with ``SegmentConfig.traced_params``).
    boundary : Optional[dict]
        For a plan from ``restart_from``: the restored handle's ``acceleration``
        (total), ``self_acceleration`` and ``force_scale`` (host); ``None`` for a
        fresh run.
    external_fn : Optional[Callable[..., Any]]
        The runner's external field.
    forward : Any
        The pre-pass's :class:`~jaccpot.production.RunResult`.
    """

    engine: Any
    segments: tuple
    start: dict
    params: dict
    boundary: Optional[dict]
    external_fn: Optional[Callable[..., Any]]
    forward: Any
    _programs: dict = dataclasses.field(default_factory=dict, repr=False)

    def inputs(self) -> dict:
        """The pre-pass's inputs, on device: what the gradient run starts from.

        Returns
        -------
        dict
            :data:`GRADIENT_INPUTS` -> value.
        """
        return {
            "positions": jnp.asarray(self.start["positions"]),
            "velocities": jnp.asarray(self.start["velocities"]),
            "masses": jnp.asarray(self.start["masses"]),
            **self.params,
        }

    def rollout(
        self,
        inputs: dict,
        *,
        checkpoint: Optional[str] = "step",
        inner_steps: Optional[int] = None,
    ) -> RolloutResult:
        """The gradient run: the segments chained, each on its settled plan. Traced.

        Pure in ``inputs``; call it inside ``jax.jit`` (with the masses an
        argument, as :func:`~jaccpot.runtime.grad_rollout.differentiable_rollout`
        asks). Its forward is the pre-pass's trajectory, bitwise, at the
        pre-pass's inputs (module docstring).

        Parameters
        ----------
        inputs : dict
            :data:`GRADIENT_INPUTS` -> value (:meth:`inputs` for the pre-pass's).
        checkpoint : Optional[str]
            Each segment's ``differentiable_rollout(checkpoint=...)``.
        inner_steps : Optional[int]
            Each segment's ``inner_steps`` (``"nested"`` only).

        Returns
        -------
        RolloutResult
            The final positions, velocities and total acceleration; ``aux`` is
            ``(f_b, bits, needs)`` folded over every segment. Check it
            (``check_rollout(result, gradient=True)``) before using a gradient.

        Raises
        ------
        ValueError
            If the first segment continues a handle (a plan from ``restart_from``)
            and the plan holds no restored arrays, or a segment's start kind is
            unknown.
        """
        from jaccpot.production import _trace_state_kept

        x = jnp.asarray(inputs["positions"])
        v = jnp.asarray(inputs["velocities"])
        m = jnp.asarray(inputs["masses"])
        dt = inputs["dt"]
        params = inputs["force_params"]
        field = self.external_fn
        args = inputs["external_params"] if field is not None else None
        n, dtype = int(x.shape[0]), x.dtype
        segments = self.segments
        rollout_kw = dict(
            params=params,
            external_fn=field,
            external_args=args,
            checkpoint=checkpoint,
            inner_steps=inner_steps,
        )

        def array(host: Any) -> Any:
            return None if host is None else jnp.asarray(host)

        def verdict(bits_in: Any, needs_in: Any) -> tuple:
            # the folded verdict so far, as arrays (a scan carries them)
            return (
                jnp.zeros((), jnp.uint32) if bits_in is None else bits_in,
                (
                    jnp.zeros((len(NEEDS_FIELDS),), jnp.int32)
                    if needs_in is None
                    else needs_in
                ),
            )

        def plus_field(acc: Any, x_in: Any, v_in: Any) -> Any:
            # A segment's start total as the runner builds it: the self-gravity
            # plus the field at the start state, added EAGERLY, one op at a time.
            # The barrier keeps XLA from contracting the field's last multiply into
            # this add (an FMA): the NFW field itself is bitwise eager or jitted,
            # but 3530 of 12288 totals moved by an ulp without it (CPU, float64)
            acc = jnp.asarray(acc, dtype)
            extra = _field_at(field, args, x_in, v_in)
            return acc if extra is None else acc + jax.lax.optimization_barrier(extra)

        prev: Optional[RolloutResult] = None
        prev_self: Any = None
        bits: Any = None
        needs: Any = None
        f_b: Any = None
        # true while the loop below traces (`_under_plan`)
        in_chain = [True]
        with _trace_state_kept(self.engine):
            for k, seg in enumerate(segments):
                _install_plan(self.engine, seg.plan)
                base = seg.provider.differentiable_step

                def self_at(
                    x_in: Any, m_in: Any, aux_in: Any, _base: Callable = base
                ) -> tuple:
                    # one force outside the rollouts (a segment's start)
                    acc, inner = _base(x_in, m_in, params, aux_in)
                    return jnp.asarray(acc, dtype), inner

                # Rematerialised in the reverse pass, so its residuals (one whole
                # force's) are not held through the chain's; outside any loop, so
                # `prevent_cse` keeps XLA from folding the recompute into the
                # forward. (The rollout's own start force is a branch of its
                # hand-written reverse loop instead, which also keeps its zero /
                # iota fills apart: `grad_rollout._adjoint_rollout`.)
                self_force = (
                    self_at
                    if checkpoint is None
                    else jax.checkpoint(self_at, prevent_cse=True)
                )
                carry_self = k + 1 < len(segments) and segments[k + 1].kind == "self"
                step_fn = _under_plan(
                    self.engine,
                    seg.plan,
                    _carrying_self(base) if carry_self else base,
                    in_chain,
                )
                seed = array(seg.seed)
                restart = self.boundary if k == 0 else None
                if prev is None and seg.kind in ("carried", "self"):
                    if restart is None:
                        raise ValueError(
                            f"a {seg.kind!r} first segment continues a handle: "
                            "the plan needs the restored handle's arrays "
                            "(GradientPlan.boundary)"
                        )
                    # the restored arrays' VALUES, the TANGENT of the force
                    # recomputed at the restart positions (thresholds from the f_b
                    # the restart carries)
                    f_b_r = array(restart["force_scale"])
                    acc_r, inner_r = self_force(x, m, (f_b_r, None, None))
                    bits, needs = inner_r[1], inner_r[2]
                acc0: Any
                if seg.kind == "carried":
                    if prev is not None:
                        acc0, aux0 = prev.acceleration, (f_b, bits, needs)
                    else:
                        acc0 = _keep_value(
                            jnp.asarray(restart["acceleration"], dtype),
                            plus_field(acc_r, x, v),
                        )
                        aux0 = (f_b_r, bits, needs)
                elif seg.kind == "start":
                    # the start force's f_b feeds the first step
                    acc_self, aux0 = self_force(x, m, (seed, bits, needs))
                    acc0 = plus_field(acc_self, x, v)
                elif seg.kind in ("seed_kept", "eager_self"):
                    acc_self, inner = self_force(x, m, (seed, bits, needs))
                    acc0 = plus_field(acc_self, x, v)
                    # the seed again for the first step (the runner discards the
                    # start force's f_b), with the start force's verdict
                    aux0 = (seed, inner[1], inner[2])
                elif seg.kind == "self":
                    acc_self = (
                        prev_self
                        if prev is not None
                        else _keep_value(
                            jnp.asarray(restart["self_acceleration"], dtype), acc_r
                        )
                    )
                    acc0 = plus_field(acc_self, x, v)
                    aux0 = (seed, *verdict(bits, needs))
                else:
                    raise ValueError(f"unknown segment start kind {seg.kind!r}")
                if carry_self:
                    aux0 = (aux0, jnp.zeros((n, 3), dtype), aux0)
                out = differentiable_rollout(
                    step_fn,
                    x,
                    v,
                    m,
                    dt,
                    seg.steps,
                    aux=aux0,
                    acceleration=acc0,
                    **rollout_kw,
                )
                inner = out.aux
                if carry_self:
                    inner, last_self, last_in = out.aux
                    # The last step's self-gravity. A plain scan differentiates
                    # the aux; "step" / "nested" do not (their reverse is
                    # hand-written), so there its VALUE comes from the aux (the
                    # handle's, bitwise) and its TANGENT from the same force
                    # recomputed on this segment's plan at the same inputs -- the
                    # end positions are the last step's drifted ones, `last_in` its
                    # aux -- so on the same lists.
                    prev_self = (
                        last_self
                        if checkpoint is None
                        else _keep_value(
                            jnp.asarray(last_self, dtype),
                            self_force(out.positions, m, last_in)[0],
                        )
                    )
                f_b, bits, needs = inner
                x, v, prev = out.positions, out.velocities, out
        in_chain[0] = False
        assert prev is not None
        return RolloutResult(x, v, prev.acceleration, (f_b, bits, needs))

    def value_and_grad(
        self,
        loss: Callable[[RolloutResult, dict], Any],
        *,
        wrt: Sequence[str] = ("positions", "velocities", "masses"),
        inputs: Optional[dict] = None,
        checkpoint: Optional[str] = "step",
        inner_steps: Optional[int] = None,
    ) -> GradientResult:
        """The loss of the gradient run and its gradient; a failed verdict raises.

        ONE jitted ``jax.value_and_grad`` of ``loss(rollout(inputs), inputs)``,
        compiled once per ``(loss, wrt, checkpoint, inner_steps)`` and the static
        inputs (a Python ``dt`` stays a literal, as in the runner's segments
        without ``traced_params``). Every array input is an argument of the
        program, the masses included.

        Parameters
        ----------
        loss : Callable[[RolloutResult, dict], Any]
            ``loss(result, inputs) -> scalar``: ``result`` the final state
            (:meth:`rollout`), ``inputs`` the (traced) inputs.
        wrt : Sequence[str]
            The inputs to differentiate, of :data:`GRADIENT_INPUTS`. ``dt`` must
            then be an array.
        inputs : Optional[dict]
            Inputs that replace the pre-pass's (:meth:`inputs`), e.g. an
            optimiser's next iterate; the pre-pass's plans must still hold there,
            or the verdict refuses the gradient.
        checkpoint : Optional[str]
            As :meth:`rollout`.
        inner_steps : Optional[int]
            As :meth:`rollout`.

        Returns
        -------
        GradientResult
            ``(value, grads, result, plan)``. A gradient run with a step that
            overflowed a capacity or whose gradient lists failed is REFUSED:
            ``check_rollout(result, gradient=True)`` raises
            :class:`~jaccpot.runtime.grad_rollout.RolloutVerdictError` (re-plan
            with a new pre-pass at these inputs).

        Raises
        ------
        ValueError
            For an unknown input name in ``wrt`` or ``inputs``.
        """
        from jaccpot.production import _trace_state_kept

        wrt = tuple(wrt)
        given = {} if inputs is None else dict(inputs)
        unknown = sorted((set(wrt) | set(given)) - set(GRADIENT_INPUTS))
        if unknown:
            raise ValueError(
                f"unknown gradient inputs {unknown}; known: {list(GRADIENT_INPUTS)}"
            )
        merged = {**self.inputs(), **given}
        diff = {name: merged[name] for name in wrt}

        def is_static(value: Any) -> bool:
            leaves = jax.tree_util.tree_leaves(value)
            return all(isinstance(leaf, (bool, int, float)) for leaf in leaves)

        rest = {k: val for k, val in merged.items() if k not in diff}
        static = {k: val for k, val in rest.items() if is_static(val)}
        dynamic = {k: val for k, val in rest.items() if k not in static}
        key = (loss, wrt, checkpoint, inner_steps, repr(sorted(static.items())))
        program = self._programs.get(key)
        if program is None:

            def objective(diff_in: dict, dynamic_in: dict) -> tuple:
                every = {**static, **dynamic_in, **diff_in}
                result = self.rollout(
                    every, checkpoint=checkpoint, inner_steps=inner_steps
                )
                return loss(result, every), result

            program = jax.jit(jax.value_and_grad(objective, has_aux=True))
            self._programs[key] = program
        # the reverse rules trace after the forward: under the last plan
        with _trace_state_kept(self.engine):
            _install_plan(self.engine, self.segments[-1].plan)
            (value, result), grads = program(diff, dynamic)
        check_rollout(result, gradient=True)
        return GradientResult(value, grads, result, self)


# --------------------------------------------------------------------------
# The recorder: the runner's forward pre-pass.
# --------------------------------------------------------------------------


class _Recorder:
    """What :meth:`ProductionRunner.run` hands its segments, kept per segment.

    Installed on the runner by :meth:`ProductionRunner.gradient_plan` for the
    pre-pass only (``runner._grad_recorder``); every hook is host-side.

    Parameters
    ----------
    runner : Any
        The :class:`~jaccpot.production.ProductionRunner`.
    freeze_cache : Optional[dict]
        The freeze step's compiled rebuild programs, by the provider's compile key,
        shared with other pre-passes of this runner (``gradient_plan(...,
        freeze_cache=)``); ``None``: this pre-pass's own (compiled again per call).
    """

    def __init__(self, runner: Any, freeze_cache: Optional[dict] = None) -> None:
        self.runner = runner
        self.engine = runner.engine
        self.segments: list = []
        self.start: Optional[dict] = None
        self.params: Optional[dict] = None
        self.boundary: Optional[dict] = None
        self._masses: Any = None
        self._step = 0
        self._start_state: Optional[np.ndarray] = None
        self._start_f_b: Optional[np.ndarray] = None
        #: the gradient statics frozen so far, by provider compile key
        self._frozen: dict = {}
        #: the freeze step's rebuild programs, by provider compile key
        self._rebuild_programs: dict = {} if freeze_cache is None else freeze_cache

    def begin(
        self,
        state: Any,
        masses: Any,
        handle: Any,
        step: int,
        dt: Any,
        force_params: Any,
        external_params: Any,
    ) -> None:
        """Before the first segment: the inputs, and a restored handle's arrays.

        Parameters
        ----------
        state : Any
            ``[N, 2, 3]`` the run starts from.
        masses : Any
            ``[N]``.
        handle : Any
            The handle it continues (``None``: a fresh run).
        step : int
            Its step.
        dt : Any
            The segments' ``dt``.
        force_params : Any
            The segments' force parameters.
        external_params : Any
            The segments' field parameters.
        """
        host = _host(state)
        assert host is not None
        self.start = {
            "positions": host[:, 0].copy(),
            "velocities": host[:, 1].copy(),
            "masses": _host(masses),
        }
        self.params = {
            "dt": dt,
            "force_params": force_params,
            "external_params": external_params,
        }
        self._masses = masses
        self._step = int(step)
        if handle is not None:
            total = handle.acceleration
            self.boundary = {
                "acceleration": _host(
                    handle.self_acceleration if total is None else total
                ),
                "self_acceleration": _host(handle.self_acceleration),
                "force_scale": _host(handle.force_scale),
            }

    def before_segment(self, state: Any, handle: Any) -> None:
        """Keep the segment's start (it is donated to the segment).

        Parameters
        ----------
        state : Any
            ``[N, 2, 3]`` the segment starts from.
        handle : Any
            The handle it continues, or ``None``.
        """
        self._start_state = _host(state)
        self._start_f_b = None if handle is None else _host(handle.force_scale)

    def after_segment(self, steps: int, handle: Any, retries: int) -> None:
        """Record the segment that just ran, with its plan still installed.

        Parameters
        ----------
        steps : int
            Its length.
        handle : Any
            Its returned handle (its provider is the scan's).
        retries : int
            The retries it needed.

        Raises
        ------
        RuntimeError
            If ``strict_run_v2`` recorded no start for the segment.
        """
        record = self.engine.__dict__.pop("_strict_particle_last_start", None)
        if record is None:
            raise RuntimeError(
                "the pre-pass's segment left no start record: the runner's "
                "strict_run_v2 is not the particle carry this mode reads"
            )
        kind = str(record["kind"])
        seed = None if kind == "carried" else _host(record["seed"])
        provider = self._freeze(
            handle.provider, self._start_f_b if kind == "carried" else seed
        )
        self.segments.append(
            GradSegment(
                step=self._step,
                steps=int(steps),
                kind=kind,
                seed=seed,
                retries=int(retries),
                provider=provider,
                plan=_plan_snapshot(self.engine),
            )
        )
        self._step += int(steps)

    def _freeze(self, provider: Any, f_b: Optional[np.ndarray]) -> Any:
        """``provider`` with its gradient statics, frozen on a rebuild at the start.

        The rebuild runs under the segment's plan (still installed) at the
        segment's start positions, with its start thresholds; its trace leaves the
        engine as it found it. ``strict_run_v2`` makes a new provider object every
        call, so the statics are kept by the provider's compile key (the template,
        the walk caps, the depth bound: what they depend on), and a segment whose
        key an earlier one froze rebuilds and compiles nothing. The rebuild's
        program is kept by the same key and engine (``freeze_cache``: across
        pre-passes too), so a pre-pass whose key an earlier one compiled freezes
        the statics on this start's own eager topology without compiling (the
        freeze's trace and compile were 35 s, most of a warm pre-pass; its run
        0.1 s: CPU interpret, N = 512, ``dehnen_error``, 2026-11-01).

        Parameters
        ----------
        provider : Any
            The segment's forward provider.
        f_b : Optional[np.ndarray]
            The thresholds' f_b (``None`` without ``dehnen_error``).

        Returns
        -------
        Any
            The provider with ``grad_statics`` (:meth:`FusedForceProvider.with_grad_plan
            <jaccpot.FusedForceProvider.with_grad_plan>`).
        """
        from jaccpot.production import _trace_state_kept

        key = provider.key()
        statics = self._frozen.get(key)
        if statics is None:
            assert self._start_state is not None
            state = jnp.asarray(self._start_state)
            f_b_in = None if f_b is None else jnp.asarray(f_b)
            with _trace_state_kept(self.engine):
                topology = self._rebuild_program(key, provider)(
                    state[:, 0], self._masses, f_b_in
                )
                statics = provider.with_grad_plan(topology).grad_statics
            self._frozen[key] = statics
        return type(provider)(
            provider.engine,
            leaf_size=provider.leaf_size,
            max_order=provider.max_order,
            theta=provider.theta,
            runtime_overrides=provider.runtime_overrides,
            template=provider.template,
            grad_statics=statics,
        )

    def _rebuild_program(self, key: tuple, provider: Any) -> Callable[..., Any]:
        """The jitted ``provider.rebuild(..., keep_far_list=True)`` of ``key``.

        Cached by the provider's compile key, which holds everything the traced
        rebuild reads at trace time apart from the engine (``provider.key()``), and
        by the engine itself: an entry of another engine is not reused.

        Parameters
        ----------
        key : tuple
            ``provider.key()``.
        provider : Any
            The provider to trace it with when no program is cached.

        Returns
        -------
        Callable[..., Any]
            ``program(positions, masses, force_scale) -> FusedTopology``.
        """
        entry = self._rebuild_programs.get(key)
        if entry is not None and entry[0] is self.engine:
            return entry[1]
        program = jax.jit(
            lambda x, m, s: provider.rebuild(x, m, force_scale=s, keep_far_list=True)
        )
        self._rebuild_programs[key] = (self.engine, program)
        return program

    def plan(self, forward: Any) -> GradientPlan:
        """The :class:`GradientPlan` of the pre-pass.

        Parameters
        ----------
        forward : Any
            The pre-pass's :class:`~jaccpot.production.RunResult`.

        Returns
        -------
        GradientPlan
            Its segments.

        Raises
        ------
        ValueError
            If the pre-pass ran no segment.
        """
        if not self.segments:
            raise ValueError(
                "the pre-pass ran no segment (num_steps reached, or a callback "
                "stopped it at step 0): nothing to differentiate"
            )
        assert self.start is not None and self.params is not None
        return GradientPlan(
            engine=self.engine,
            segments=tuple(self.segments),
            start=self.start,
            params=self.params,
            boundary=self.boundary,
            external_fn=self.runner.external_acceleration_fn,
            forward=forward,
        )
