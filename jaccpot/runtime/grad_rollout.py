"""Differentiable, checkpointed velocity-Verlet rollouts over a force step (S9).

:func:`differentiable_rollout` runs ``num_steps`` steps of the PRODUCTION velocity
Verlet -- the expressions and their association exactly as
``strict_run_v2(carry="particles")`` writes them -- over any force with the
contract of :meth:`FusedForceProvider.step <jaccpot.FusedForceProvider.step>`::

    step_fn(positions, masses, params, aux) -> (acceleration, aux)

and is a pure function of its arrays, so ``jax.grad`` goes through it. The force
is a protocol, not a provider: the fused lane's step, its differentiable twin, or a
direct sum (:func:`direct_sum_step_fn`, the oracle the tests develop on) run in the
same scan. The step's ``aux`` is opaque here: it is threaded from one step to the
next and returned, so a step that folds a capacity verdict into it (the provider's
``(f_b, bits, needs)``) returns the verdict of the whole rollout. Only the verdict
helpers at the end of this module read it.

WHAT A REVERSE PASS STORES. ``checkpoint`` decides it, and nothing else does:

* ``None``: a plain ``lax.scan``. Reverse mode keeps every residual of every step
  -- for the direct sum the ``[N, N]`` pair arrays, for the fused lane the whole
  linearised force -- so it is for small ``N`` and short horizons (and the
  reference the other two are bitwise against).
* ``"step"``: each step's force and kick are recomputed in the reverse pass from
  the DRIFTED positions (track C's placement): the drift and the kick are linear
  in ``(x, v, a)`` at a fixed ``dt``, so their reverse needs no primal value and
  the forward saves per step only what the force's recompute reads, the drifted
  positions and the aux leaves -- 12 B per particle in float32 (24 in float64),
  plus the fused lane's f_b (4 B under ``dehnen_error``). Measured
  (``bench/grad_rollout_memory.py``, CPU direct sum, N = 4096): 12.00 B/particle
  per step in float32, against 36.00 with the checkpoint around the whole step
  (which stored ``(x, v, a)``). A DIFFERENTIATED ``dt`` makes the drift
  bilinear (``v dt``, ``a dt^2``): its cotangent needs ``v`` and ``a`` of every
  step, 36 B/particle per step then. The masses, the force and field parameters
  and ``dt`` enter every step unchanged and are stored once
  (``tests/unit/runtime/test_grad_rollout.py`` reads the residuals off
  ``print_saved_residuals`` and the temporaries off ``memory_analysis``). The
  reverse pass recomputes each force once. The reverse is written by hand
  (:func:`_adjoint_rollout`, a ``custom_vjp``), not ``jax.checkpoint`` in a
  ``lax.scan``: reverse mode of a scan hoists every loop-invariant fill of the
  force (zeros, iotas) out of the forward loop, and on the fused lane that
  held ~650 B/particle beside the forward step (session 2, track D).
* ``"nested"``: two levels. The forward stores one carry ``(x, v, a, aux)`` per
  segment of ``inner_steps`` steps; the reverse pass recomputes one segment,
  storing the per-step residual above, and reverses it. About
  ``num_steps / inner_steps`` carries plus ``inner_steps`` step residuals, for
  one more forward per step than ``"step"``.

The three compute the same forward bit for bit (the same expressions in the
same order). Their gradients agree to round-off, NOT bitwise: a checkpoint
regroups the cotangent sums of a value used on both sides of it, ``"nested"`` sums
a scan constant's cotangent per segment and then over segments, and XLA fuses
the recomputed forward with the reverse differently (on CPU it contracts a
multiply and an add into one FMA within a fusion). Measured on the direct sum
(CPU, N = 64, 4-6 steps): 1.5e-15 of the largest component in float64, 2.7e-7 in
float32. The start force (``acceleration=None``) is checkpointed with them, so
its residuals -- one whole force's -- are not held through the reverse pass
either.

A ``num_steps`` the inner length does not divide gets a SHORTER LAST SEGMENT
(``num_steps % inner_steps`` steps, checkpointed the same way) rather than a
refusal: the horizon is the physics' choice, not the checkpointing's. The cost is
a second trace of the step body (the tail's scan), which on the fused lane's
interpret-mode CPU path is minutes of compile; pick an ``inner_steps`` that
divides ``num_steps`` to avoid it.

THE FORWARD OF A GRADIENT PROGRAM is the forward-only rollout's to round-off,
not always bitwise: a plain scan's forward also writes every residual, and XLA
fuses it differently (direct sum, CPU: up to 62 of 192 elements of one array
moved by an ulp). A bitwise comparison with ``strict_run_v2`` is a claim about
the jitted forward, not about the primal of a ``value_and_grad``.

HOST OFFLOAD (session 2) changes WHERE the outer carries live, not what is stored:
under ``"nested"`` they are the residual ``starts`` of :func:`_adjoint_rollout`'s
forward (``rollout_fwd``), the one place to put them in host memory.

THE MASSES are an argument, and must stay one up to the caller's ``jax.jit``: a
closed-over array is a constant of the compiled program, which XLA compiles
differently (``tests/integration/strict/test_strict_runners_on_provider.py``
records one float32 rounding moved at the third step that way).
:func:`value_and_grad_checked` jits with every array an argument.

THE VERDICT. A gradient through a step whose capacity verdict failed is the
gradient of a truncated force: a plausible wrong number. The rollout cannot raise
inside a trace, so it returns the folded aux and the caller checks it on the host
(:func:`rollout_ok`, :func:`check_rollout`), or lets
:func:`value_and_grad_checked` refuse the gradient for it.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Callable
from jaxtyping import Array

from .fused_topology import (
    GRAD_FAIL_MASK,
    OVERFLOW_FAIL_MASK,
    ForceParams,
    StepParams,
    grad_ok,
    overflow_ok,
)
from .reference import direct_sum

__all__ = [
    "CHECKPOINT_MODES",
    "RolloutResult",
    "RolloutVerdictError",
    "check_rollout",
    "differentiable_rollout",
    "direct_sum_step_fn",
    "rollout_ok",
    "value_and_grad_checked",
]

#: what ``differentiable_rollout(checkpoint=...)`` accepts
CHECKPOINT_MODES = (None, "step", "nested")


class RolloutResult(NamedTuple):
    """The state after a rollout, and the folded step aux.

    Attributes
    ----------
    positions : Array
        ``[N, 3]`` final positions.
    velocities : Array
        ``[N, 3]`` final velocities.
    acceleration : Array
        ``[N, 3]`` TOTAL acceleration at the final positions (the step's force
        plus the external field): what the next rollout starts from
        (``acceleration=``) to continue this one bitwise.
    aux : Any
        The last step's aux. For the fused provider ``(f_b, bits, needs)``, the
        overflow bits OR-ed over every step (:func:`rollout_ok`); ``None`` for a
        step without one (the direct sum). Passed with ``acceleration`` it
        continues the rollout.
    """

    positions: Array
    velocities: Array
    acceleration: Array
    aux: Any


class RolloutVerdictError(RuntimeError):
    """A gradient (or state) computed on a rollout whose capacity verdict failed."""


# --------------------------------------------------------------------------
# The rematerialisation policy: one place.
# --------------------------------------------------------------------------


def _checkpointed(fn: Callable, level: str) -> Callable:
    """``fn`` under the rollout's checkpoint policy for ``level``.

    Every level saves nothing but the inputs its recompute reads
    (``jax.checkpoint``'s default policy): a step's force and kick read the
    drifted positions and the aux (its old velocity and acceleration enter the
    kick linearly, so they are not saved), a segment's the carry at its start.
    Session 2's host offload gives the ``"segment"`` level an offloading policy
    here (the outer carries to pinned host memory) and changes nothing else.

    Parameters
    ----------
    fn : Callable
        The function to checkpoint.
    level : str
        ``"step"`` or ``"segment"`` (inside a ``lax.scan`` body), or ``"edge"``
        (outside any scan: the start force, a tail segment).

    Returns
    -------
    Callable
        The checkpointed function.

    Raises
    ------
    ValueError
        If ``level`` is none of the three.
    """
    if level not in ("step", "segment", "edge"):
        raise ValueError(f"unknown checkpoint level {level!r}")
    # Inside a scan the reverse pass's recompute sits in another loop than the
    # forward it repeats, so XLA cannot merge the two and the optimisation
    # barriers ``prevent_cse`` adds would only constrain fusion. Outside one, the
    # recompute and the forward are in one computation: without the barrier XLA's
    # CSE would fold them and keep the forward's residuals alive after all.
    return jax.checkpoint(fn, prevent_cse=level == "edge")


# --------------------------------------------------------------------------
# The rollout.
# --------------------------------------------------------------------------


def _field_reads_velocities(
    external_fn: Callable[..., Array],
    external_args: Any,
    positions: Array,
) -> bool:
    """Whether the field may read the velocity rows it is handed.

    Traces ``external_fn`` once more (``jax.make_jaxpr``) on abstract
    ``[N, 2, 3]`` rows and looks at every equation that consumes them: a
    ``slice`` confined to the position row (``state[:, 0]``, the halo fields'
    pattern) reads no velocity; anything else -- another slice, a gather, a
    nested call that takes the rows whole -- counts as reading them. Conservative:
    a field it cannot see into is said to read them.

    WHY. ``strict_run_v2``'s field takes ``[N, 2, 3]`` rows of position and
    velocity. Inside a checkpointed step the reverse pass's recompute rebuilds
    those rows, and rebuilding them needs the step's velocities, so a field that
    reads only positions -- every galaxy halo -- still made the velocities a
    per-step residual (+12 B/particle in float32; measured with a harmonic field
    whose constant is differentiated). A field that reads no velocity gets zero
    velocity rows instead: the same output, bit for bit, and nothing saved for
    them.

    Parameters
    ----------
    external_fn : Callable[..., Array]
        The field.
    external_args : Any
        Its parameters, or ``None``.
    positions : Array
        ``[N, 3]`` positions: the rows' shape and dtype.

    Returns
    -------
    bool
        ``False`` when every use of the rows is a slice of the position row.
    """
    n = positions.shape[0]
    rows = jax.ShapeDtypeStruct((n, 2, 3), positions.dtype)

    def field_of(rows_in: Array) -> Array:
        if external_args is None:
            return external_fn(rows_in)
        return external_fn(rows_in, external_args)

    jaxpr = jax.make_jaxpr(field_of)(rows).jaxpr
    rows_var = jaxpr.invars[0]
    if any(v is rows_var for v in jaxpr.outvars):
        return True
    for eqn in jaxpr.eqns:
        if not any(v is rows_var for v in eqn.invars):
            continue
        if eqn.primitive.name != "slice":
            return True
        start, limit = eqn.params["start_indices"], eqn.params["limit_indices"]
        if not (int(start[1]) == 0 and int(limit[1]) <= 1):
            return True
    return False


def _external_acceleration(
    external_fn: Optional[Callable[..., Array]],
    external_args: Any,
    positions: Array,
    velocities: Array,
) -> Optional[Array]:
    """The external field at ``positions``, as ``strict_run_v2`` calls it.

    Parameters
    ----------
    external_fn : Optional[Callable[..., Array]]
        ``external_fn(state)`` or, with ``external_args``,
        ``external_fn(state, external_args)``; ``state`` is ``[N, 2, 3]`` (rows
        of position and velocity). ``None``: no field.
    external_args : Any
        The field's parameters (a pytree; arrays are differentiable), or ``None``.
    positions : Array
        ``[N, 3]`` positions (the step's drifted ones).
    velocities : Array
        ``[N, 3]`` velocities (the step's starting ones: the field sees the
        state ``strict_run_v2`` hands it, drifted positions beside unkicked
        velocities), or zeros for a field that does not read them
        (:func:`_field_reads_velocities`).

    Returns
    -------
    Optional[Array]
        ``[N, 3]`` in the positions' dtype, or ``None`` without a field.
    """
    if external_fn is None:
        return None
    state_rows = jnp.stack([positions, velocities], axis=1)
    field = (
        external_fn(state_rows)
        if external_args is None
        else external_fn(state_rows, external_args)
    )
    return jnp.asarray(field, dtype=positions.dtype)


class _StepConsts:
    """The step's constants, split into the differentiated and the static.

    A step reads the masses, ``dt``, the force's ``params`` and the field's
    ``external_args``. The adjoint (:func:`_adjoint_rollout`) is a ``custom_vjp``,
    which differentiates its explicit inputs only, so the arrays are passed to it
    (:attr:`dynamic`) and the Python numbers are kept static: a literal of the
    program, as before (the provider's static-parameter program).

    Parameters
    ----------
    static : tuple
        ``(masses, dt, leaves, treedef, traced)``: the masses, ``dt`` (an array,
        or ``(value, dtype)`` for a Python number), the flattened
        ``(params, external_args)`` and which leaves are arrays.
    """

    def __init__(self, static: tuple) -> None:
        self._masses, self._dt, self._leaves, self._treedef, self._traced = static

    @classmethod
    def split(
        cls, masses: Array, dt: Any, dtype: Any, params: Any, external_args: Any
    ) -> "_StepConsts":
        """Split the step's constants.

        Parameters
        ----------
        masses : Array
            ``[N]`` masses (always passed: they are differentiated).
        dt : Any
            The timestep as given: a Python number stays a literal, an array is
            passed (in the state's dtype).
        dtype : Any
            The state's dtype.
        params : Any
            The force's parameters (a pytree, or ``None``).
        external_args : Any
            The field's parameters (a pytree, or ``None``).

        Returns
        -------
        _StepConsts
            The split.
        """
        leaves, treedef = jax.tree_util.tree_flatten((params, external_args))
        traced = tuple(not isinstance(leaf, Real) for leaf in leaves)
        dt_static = isinstance(dt, Real)
        return cls(
            (
                masses,
                (float(dt), dtype) if dt_static else jnp.asarray(dt, dtype=dtype),
                leaves,
                treedef,
                traced,
            )
        )

    @property
    def dt_traced(self) -> bool:
        """Whether ``dt`` is an array (its cotangent needs ``v`` and ``a``)."""
        return not isinstance(self._dt, tuple)

    @property
    def dynamic(self) -> tuple:
        """``(masses, dt or None, array leaves)``: the differentiated inputs."""
        return (
            self._masses,
            self._dt if self.dt_traced else None,
            tuple(leaf for leaf, t in zip(self._leaves, self._traced) if t),
        )

    def join(self, dynamic: tuple) -> tuple:
        """``(masses, dt, params, external_args)`` from :attr:`dynamic`'s values.

        Parameters
        ----------
        dynamic : tuple
            A tuple of :attr:`dynamic`'s structure (the custom_vjp's tracers).

        Returns
        -------
        tuple
            The constants a step reads.
        """
        masses, dt, traced_leaves = dynamic
        it = iter(traced_leaves)
        leaves = [
            next(it) if t else leaf for leaf, t in zip(self._leaves, self._traced)
        ]
        params, external_args = jax.tree_util.tree_unflatten(self._treedef, leaves)
        if dt is None:
            # a Python dt: a literal of the program, cast as strict_run_v2 casts it
            dt = jnp.asarray(self._dt[0], dtype=self._dt[1])
        return masses, dt, params, external_args


def _adjoint_rollout(
    drift: Callable[..., Array],
    force_and_kick: Callable[..., tuple],
    carry: tuple,
    consts: tuple,
    num_steps: int,
    inner: Optional[int],
    keep_va: bool,
) -> tuple:
    """``num_steps`` steps with a hand-written reverse pass (``"step"``, ``"nested"``).

    The forward is a plain ``lax.scan`` of ``drift`` then ``force_and_kick``. The
    reverse is a ``custom_vjp``: a reverse scan whose body recomputes one step's
    force at its stored drifted positions under ``jax.vjp`` and pulls the carry's
    cotangent through it, the drift transposed by hand (it is linear in
    ``(x, v, a)``). What is stored is what the checkpointed scan stored -- per
    step the drifted positions and the aux (plus ``v`` and ``a`` with
    ``keep_va``); under ``inner`` (``"nested"``) one carry per segment, the
    segment recomputed in the reverse pass storing its steps.

    WHY NOT ``jax.checkpoint`` IN A SCAN. Reverse mode of a ``lax.scan`` hoists
    the loop-invariant part of the forward body out of the loop
    (``_scan_partial_eval``'s "hoist loop-invariant residuals"): everything in a
    force that depends on no carry -- every ``jnp.zeros`` / ``full`` / ``iota``
    fill of the tree, the walk, the lists and the expansion tables -- is
    computed once before the loop and stays alive through the forward scan. On
    the fused lane at 2e6 (``preset="galaxy"``, float32) that was ~650 B/particle
    held beside the forward step's own temporaries, the peak of the whole
    gradient (999 B/particle of W). Here the forward scan is bound with every
    input known, so nothing is partially evaluated and nothing hoisted.

    The aux is threaded and returned, not differentiated (the provider's f_b is
    gradient-stopped, its bits and needs are integers). A step function must not
    close over values being differentiated (pass them as ``params``): a
    ``custom_vjp`` rejects that.

    Parameters
    ----------
    drift : Callable[..., Array]
        ``drift(x, v, a, consts) -> x``.
    force_and_kick : Callable[..., tuple]
        ``force_and_kick(x, v, a, aux, consts) -> (v, a, aux)`` at drifted ``x``.
    carry : tuple
        ``(x, v, a, aux)`` at the start.
    consts : tuple
        The differentiated constants (:attr:`_StepConsts.dynamic`).
    num_steps : int
        Steps, ``>= 1``.
    inner : Optional[int]
        ``None`` (``"step"``) or the segment length (``"nested"``).
    keep_va : bool
        Store ``v`` and ``a`` per step (a traced ``dt``, a field that reads the
        velocities).

    Returns
    -------
    tuple
        ``(x, v, a, aux)`` after ``num_steps`` steps.
    """

    def one(carry_in: tuple, c: tuple) -> tuple:
        x, v, a, aux_in = carry_in
        x = drift(x, v, a, c)
        v_new, a_new, aux_out = force_and_kick(x, v, a, aux_in, c)
        saved = (x, aux_in) + ((v, a) if keep_va else ())
        return (x, v_new, a_new, aux_out), saved

    def run(carry_in: tuple, c: tuple, length: int) -> tuple:
        return jax.lax.scan(
            lambda cr, _: (one(cr, c)[0], None), carry_in, None, length
        )[0]

    def run_saving(carry_in: tuple, c: tuple, length: int) -> tuple:
        return jax.lax.scan(lambda cr, _: one(cr, c), carry_in, None, length)

    def back(ct: tuple, saved: tuple, c: tuple, length: int) -> tuple:
        # the reverse of `length` steps: ct = (x_bar, v_bar, a_bar) after them,
        # returned before them, with the constants' cotangent summed over them
        def body(state: tuple, saved_k: tuple) -> tuple:
            (x_bar, v_bar, a_bar), c_bar = state
            x, aux_in = saved_k[:2]
            v, a = saved_k[2:] if keep_va else (jnp.zeros_like(x), jnp.zeros_like(x))

            def kick(x_: Array, v_: Array, a_: Array, c_: tuple) -> tuple:
                v_new, a_new, _ = force_and_kick(x_, v_, a_, aux_in, c_)
                return v_new, a_new

            _, pull = jax.vjp(kick, x, v, a, c)
            gx, gv, ga, gc = pull((v_bar, a_bar))
            x_bar = x_bar + gx
            # the drift's transpose: linear in (x, v, a), so x passes through and
            # (v, a) need no value unless dt is traced (then they are stored)
            _, pull_drift = jax.vjp(lambda v_, a_, c_: drift(x, v_, a_, c_), v, a, c)
            dv, da, dc = pull_drift(x_bar)
            c_bar = jax.tree_util.tree_map(jnp.add, c_bar, gc)
            c_bar = jax.tree_util.tree_map(jnp.add, c_bar, dc)
            return ((x_bar, gv + dv, ga + da), c_bar), None

        zero_c = jax.tree_util.tree_map(jnp.zeros_like, c)
        (ct, c_bar), _ = jax.lax.scan(
            body, (ct, zero_c), saved, length=length, reverse=True
        )
        return ct, c_bar

    outer, tail = (0, 0) if inner is None else divmod(num_steps, inner)

    @jax.custom_vjp
    def rollout(carry_in: tuple, c: tuple) -> tuple:
        return run(carry_in, c, num_steps)

    def rollout_fwd(carry_in: tuple, c: tuple) -> tuple:
        if inner is None:
            out, saved = run_saving(carry_in, c, num_steps)
            return out, (saved, c)

        def segment(cr: tuple, _: Any) -> tuple:
            return run(cr, c, inner), cr

        out, starts = jax.lax.scan(segment, carry_in, None, outer)
        tail_start = None
        if tail:
            # the remainder segment's start: one more stored carry
            tail_start = out
            out = run(out, c, tail)
        return out, (starts, tail_start, c)

    def rollout_bwd(res: tuple, ct_out: tuple) -> tuple:
        ct = tuple(ct_out[:3])
        if inner is None:
            saved, c = res
            ct, c_bar = back(ct, saved, c, num_steps)
        else:
            starts, tail_start, c = res
            c_bar = jax.tree_util.tree_map(jnp.zeros_like, c)
            if tail:
                _, saved = run_saving(tail_start, c, tail)
                ct, c_bar = back(ct, saved, c, tail)

            def segment_back(state: tuple, start: tuple) -> tuple:
                ct_s, c_bar_s = state
                _, saved_s = run_saving(start, c, inner)
                ct_s, gc = back(ct_s, saved_s, c, inner)
                return (ct_s, jax.tree_util.tree_map(jnp.add, c_bar_s, gc)), None

            (ct, c_bar), _ = jax.lax.scan(
                segment_back, (ct, c_bar), starts, length=outer, reverse=True
            )
        # the aux carries no gradient
        return (ct + (None,), c_bar)

    rollout.defvjp(rollout_fwd, rollout_bwd)
    return rollout(carry, consts)


def differentiable_rollout(
    step_fn: Callable[..., tuple[Any, Any]],
    positions: Array,
    velocities: Array,
    masses: Array,
    dt: Union[float, Array],
    num_steps: int,
    *,
    params: Optional[Union[ForceParams, StepParams, Any]] = None,
    aux: Any = None,
    acceleration: Optional[Array] = None,
    external_fn: Optional[Callable[..., Array]] = None,
    external_args: Any = None,
    checkpoint: Optional[str] = "step",
    inner_steps: Optional[int] = None,
) -> RolloutResult:
    """``num_steps`` velocity-Verlet steps over ``step_fn``, differentiably.

    Each step is the production particle carry's, expression for expression::

        x = x + v * dt + 0.5 * a * dt**2
        a_new, aux = step_fn(x, masses, params, aux)
        a_new = a_new + external_fn(stack([x, v]), external_args)   # if a field
        v = v + 0.5 * (a + a_new) * dt

    so over :meth:`FusedForceProvider.step <jaccpot.FusedForceProvider.step>`,
    under ``jax.jit`` with the masses an argument, the trajectory is
    ``strict_run_v2(carry="particles")``'s bit for bit (same start acceleration;
    ``tests/integration/strict/test_grad_rollout_on_provider.py``). The rollout is
    pure -- no host sync, no donation, no retry -- so it is differentiable in
    ``positions``, ``velocities``, ``masses``, ``dt``, ``params``,
    ``external_args`` and ``acceleration`` as far as ``step_fn`` and the field are.

    Parameters
    ----------
    step_fn : Callable[..., tuple[Any, Any]]
        ``step_fn(positions [N, 3], masses [N], params, aux) -> (acc [N, 3],
        aux)``. Called with ``aux`` as given for the first force, then with the
        aux it returned; the aux's pytree structure must not change after the
        first call (a scan carries it). The provider's ``step`` returns
        ``(f_b, bits, needs)``; a direct sum ``None``.
    positions : Array
        ``[N, 3]`` start positions.
    velocities : Array
        ``[N, 3]`` start velocities, same dtype.
    masses : Array
        ``[N]`` masses. A scan constant: stored once, never per step.
    dt : Union[float, Array]
        Timestep. A Python float is a literal of the program, an array a traced
        (differentiable) argument; either is cast to the positions' dtype, as
        ``strict_run_v2`` casts it.
    num_steps : int
        Steps, ``>= 1``. Static.
    params : Optional[Union[ForceParams, StepParams, Any]]
        Passed to every ``step_fn`` call unchanged (the provider's
        ``ForceParams``: traced ``G`` and ``eps2``). ``None``: the force's own.
    aux : Any
        The aux of the FIRST force call. With ``acceleration=None`` that is the
        start force's input: ``None``, or the provider's ``(f_b, None, None)``
        seed under ``mac_type='dehnen_error'``. With ``acceleration`` given it is
        the aux that came with it (a previous rollout's ``RolloutResult.aux``),
        in the structure ``step_fn`` returns.
    acceleration : Optional[Array]
        ``[N, 3]`` TOTAL acceleration at the start positions (force plus field).
        ``None``: computed here, ``step_fn`` at ``positions`` plus the field, so
        it is part of the differentiated function. Given, it is an input: the
        gradient flows into it, not through how it was computed.
    external_fn : Optional[Callable[..., Array]]
        ``strict_run_v2``'s external field: ``external_fn(state)`` or, with
        ``external_args``, ``external_fn(state, external_args)``, with ``state``
        ``[N, 2, 3]`` (the drifted positions beside the unkicked velocities),
        returning ``[N, 3]``; added to the step's acceleration. ``None``: none.
    external_args : Any
        The field's parameters (any pytree; arrays differentiable).
    checkpoint : Optional[str]
        ``None`` (plain scan), ``"step"`` (default) or ``"nested"``: what the
        reverse pass stores (module docstring). Static.
    inner_steps : Optional[int]
        ``"nested"`` only: steps per segment, ``>= 1``; ``None`` takes
        ``round(sqrt(num_steps))``. Clamped to ``num_steps``. A remainder runs
        as a shorter last segment. Static.

    Returns
    -------
    RolloutResult
        ``(positions, velocities, acceleration, aux)`` after ``num_steps``
        steps. Check ``aux`` on the host (:func:`check_rollout`) before trusting
        the state or a gradient of it.

    Raises
    ------
    ValueError
        If ``num_steps`` or ``inner_steps`` is not a positive int, ``checkpoint``
        is not one of :data:`CHECKPOINT_MODES`, ``inner_steps`` comes without
        ``checkpoint="nested"``, ``external_args`` without ``external_fn``, or the
        arrays' shapes or the state's dtypes disagree.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, (int, np.integer)):
        raise ValueError(f"num_steps must be a Python int, got {num_steps!r}")
    num_steps = int(num_steps)
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if checkpoint not in CHECKPOINT_MODES:
        raise ValueError(
            f"checkpoint must be one of {CHECKPOINT_MODES}, got {checkpoint!r}"
        )
    if inner_steps is not None:
        if checkpoint != "nested":
            raise ValueError("inner_steps needs checkpoint='nested'")
        if isinstance(inner_steps, bool) or not isinstance(
            inner_steps, (int, np.integer)
        ):
            raise ValueError(f"inner_steps must be a Python int, got {inner_steps!r}")
        if int(inner_steps) < 1:
            raise ValueError(f"inner_steps must be >= 1, got {inner_steps}")
    if external_args is not None and external_fn is None:
        raise ValueError("external_args needs an external_fn(state, external_args)")
    positions = jnp.asarray(positions)
    velocities = jnp.asarray(velocities)
    masses = jnp.asarray(masses)
    n = int(positions.shape[0]) if positions.ndim == 2 else -1
    if positions.shape != (n, 3) or velocities.shape != (n, 3):
        raise ValueError(
            "positions and velocities must both be [N, 3], got "
            f"{tuple(positions.shape)} and {tuple(velocities.shape)}"
        )
    if velocities.dtype != positions.dtype:
        raise ValueError(
            f"velocities are {velocities.dtype}, positions {positions.dtype}: the "
            "state has one dtype"
        )
    if masses.shape != (n,):
        raise ValueError(f"masses must be [{n}], got {tuple(masses.shape)}")
    if acceleration is not None and jnp.shape(acceleration) != (n, 3):
        raise ValueError(
            f"acceleration must be [{n}, 3], got {tuple(jnp.shape(acceleration))}"
        )

    dtype = positions.dtype
    # a field that ignores the velocity rows is handed zeros there: the same
    # output, and the checkpointed step's recompute then needs no velocities
    field_velocities = external_fn is not None and _field_reads_velocities(
        external_fn, external_args, positions
    )

    # What the steps are differentiated in, besides the carry: the masses, an
    # array dt, and the array leaves of the force and field parameters. Python
    # numbers stay literals of the program (dt cast to the state's dtype as
    # strict_run_v2 casts it); the provider tells a static parameter from a
    # traced one.
    consts = _StepConsts.split(masses, dt, dtype, params, external_args)

    def force(x: Array, v: Array, aux_in: Any, c: tuple) -> tuple[Array, Any]:
        # the step's total acceleration at x, in the state's dtype, as the
        # particle carry builds it: the force, plus the field at (x, v)
        masses_c, _, params_c, field_args_c = consts.join(c)
        acc, aux_out = step_fn(x, masses_c, params_c, aux_in)
        acc = jnp.asarray(acc, dtype=dtype)
        rows_v = v if field_velocities else jnp.zeros_like(v)
        field = _external_acceleration(external_fn, field_args_c, x, rows_v)
        return (acc if field is None else acc + field), aux_out

    def force_and_kick(x: Array, v: Array, a: Array, aux_in: Any, c: tuple) -> tuple:
        # the force at the drifted positions and the kick with the mean of the
        # old and new accelerations: what a step's reverse recomputes. Its input
        # is the DRIFTED x; v and a enter linearly (the kick, and the field's rows
        # only if the field reads velocities), so the reverse pass needs x and
        # the aux, not the whole carry
        a_new, aux_out = force(x, v, aux_in, c)
        return v + 0.5 * (a + a_new) * consts.join(c)[1], a_new, aux_out

    def drift(x: Array, v: Array, a: Array, c: tuple) -> Array:
        # strict_run_v2's `_advance`: the same expression, in the same order
        dt_c = consts.join(c)[1]
        return x + v * dt_c + 0.5 * a * dt_c**2

    c0 = consts.dynamic
    if acceleration is None:
        # the start force: checkpointed, so its residuals (one whole force's) are
        # not held through the reverse pass
        start = force if checkpoint is None else _checkpointed(force, "edge")
        acceleration, aux = start(positions, velocities, aux, c0)
    else:
        acceleration = jnp.asarray(acceleration, dtype=dtype)
    carry = (positions, velocities, acceleration, aux)

    if checkpoint is None:

        def step(carry_in: tuple, _: Any) -> tuple:
            # one velocity-Verlet step (strict_run_v2's `_advance` and
            # `_velocity_verlet_kick_drifted`): drift, force, kick
            x, v, a, aux_in = carry_in
            x = drift(x, v, a, c0)
            v, a_new, aux_out = force_and_kick(x, v, a, aux_in, c0)
            return (x, v, a_new, aux_out), None

        carry = jax.lax.scan(step, carry, None, num_steps)[0]
    else:
        inner = None
        if checkpoint == "nested":
            inner = (
                max(1, round(math.sqrt(num_steps)))
                if inner_steps is None
                else int(inner_steps)
            )
            inner = min(inner, num_steps)
        # the velocities and accelerations of every step are kept only where the
        # reverse reads them: a traced dt (the drift and the kick are bilinear in
        # it) or a field that reads the velocity rows
        keep_va = consts.dt_traced or field_velocities
        carry = _adjoint_rollout(
            drift, force_and_kick, carry, c0, num_steps, inner, keep_va
        )
    return RolloutResult(*carry)


# --------------------------------------------------------------------------
# A direct-sum step: the oracle force with the provider's step contract.
# --------------------------------------------------------------------------


def direct_sum_step_fn(
    *,
    G: Union[float, Array] = 1.0,
    softening: Union[float, Array] = 0.0,
    softening_kernel: Optional[str] = None,
) -> Callable[..., tuple[Array, None]]:
    """The softened O(N^2) direct sum as a ``step_fn``.

    :func:`jaccpot.runtime.reference.direct_sum` -- the oracle the FMM paths are
    checked against (``fmm_evaluate``'s traced fallback runs it the same way) --
    at every particle, behind the step contract, so the rollout (and anything
    else written against :meth:`FusedForceProvider.step
    <jaccpot.FusedForceProvider.step>`) runs on the exact force: the gate the
    fused lane's rollout gradient is checked against (plan S9, G3), and the force
    this module is developed on. It takes ``G`` and the softening as arrays
    (``direct_sum_gravitational_acceleration`` declares them floats).

    Each particle's own pair has a zero separation vector and contributes zero
    (Plummer: needs a softening above zero, else ``0 / 0``). ``O(N^2)`` time and
    ``[N, N, 3]`` temporaries: for gradient work at modest ``N``.

    Parameters
    ----------
    G : Union[float, Array]
        Gravitational constant when the step gets no ``params``.
    softening : Union[float, Array]
        Plummer-equivalent softening LENGTH when the step gets no ``params``.
    softening_kernel : Optional[str]
        The pair kernel (:mod:`jaccpot.softening`); ``None`` gives the default.
        Static.

    Returns
    -------
    Callable[..., tuple[Array, None]]
        ``step(positions, masses, params=None, aux=None) -> (acc, None)``. A
        :class:`~jaccpot.ForceParams` ``params`` (or a ``StepParams``' force)
        replaces ``G`` and the softening (``eps2``, squared, as the provider takes
        it), so both are differentiable when traced. The aux is ignored and
        returned as ``None``: there is no capacity to overflow.
    """

    def step(
        positions: Array, masses: Array, params: Any = None, aux: Any = None
    ) -> tuple[Array, None]:
        del aux  # nothing to fold: the direct sum has no capacity
        g, eps = G, softening
        force = params.force if isinstance(params, StepParams) else params
        if force is not None:
            g, eps = force.G, jnp.sqrt(jnp.asarray(force.eps2))
        acc = jax.vmap(
            lambda point: direct_sum(
                positions,
                masses,
                point,
                G=g,
                softening=eps,
                softening_kernel=softening_kernel,
            )
        )(positions)
        return acc, None

    return step


# --------------------------------------------------------------------------
# The verdict: host-side, after the rollout.
# --------------------------------------------------------------------------


def _verdict_bits(aux: Any) -> Optional[Any]:
    """The overflow bits in a step aux, or ``None`` for a step that has none.

    Parameters
    ----------
    aux : Any
        A :class:`RolloutResult`, or a step aux: ``None`` (no verdict) or the
        provider's ``(f_b, bits, ...)`` (bits second).

    Returns
    -------
    Optional[Any]
        The ``uint32`` bits, or ``None``.

    Raises
    ------
    TypeError
        If the aux is neither ``None`` nor a tuple with the bits second.
    """
    if isinstance(aux, RolloutResult):
        aux = aux.aux
    if aux is None:
        return None
    if isinstance(aux, (tuple, list)) and len(aux) >= 2 and aux[1] is not None:
        return aux[1]
    raise TypeError(
        "no capacity verdict in this aux: expected None (a step without one) or "
        f"the provider's (f_b, bits, needs), got {type(aux).__name__}; pass "
        "verdict_of= to read another step's"
    )


def rollout_ok(aux: Any, *, gradient: bool = False) -> bool:
    """Whether every step of a rollout fitted its capacities. Host sync.

    Parameters
    ----------
    aux : Any
        A :class:`RolloutResult` or its ``aux``.
    gradient : bool
        The gradient's verdict, :func:`~jaccpot.runtime.fused_topology.grad_ok`
        (``GRAD_FAIL_MASK``: the force's bits and the differentiable step's
        ``OVERFLOW_GRAD_LISTS``), instead of the state's,
        :func:`~jaccpot.runtime.fused_topology.overflow_ok`
        (``OVERFLOW_FAIL_MASK``).

    Returns
    -------
    bool
        ``True`` for a step without a verdict (``aux`` ``None``); else the
        verdict of the folded bits.
    """
    bits = _verdict_bits(aux)
    if bits is None:
        return True
    verdict = grad_ok if gradient else overflow_ok
    return bool(np.asarray(verdict(bits)))


def check_rollout(aux: Any, *, gradient: bool = False) -> None:
    """Raise when a rollout's capacity verdict failed. Host sync.

    Parameters
    ----------
    aux : Any
        A :class:`RolloutResult` or its ``aux``.
    gradient : bool
        Check the gradient's verdict (:func:`rollout_ok`).

    Raises
    ------
    RolloutVerdictError
        If a verdict bit is set: some step's force was truncated (or, for a
        gradient, its gradient lists failed), so the state or the gradient is
        wrong. The message names the bits; re-plan the caps on a forward pre-pass
        (the provider's ``replanned``) and run again.
    """
    if rollout_ok(aux, gradient=gradient):
        return
    from . import fused_topology

    bits = int(np.asarray(_verdict_bits(aux)))
    names = sorted(
        name
        for name in fused_topology.__all__
        if name.startswith("OVERFLOW_")
        and not name.endswith("_MASK")
        and bits & int(getattr(fused_topology, name))
    )
    what = "gradient" if gradient else "capacity"
    raise RolloutVerdictError(
        f"the rollout's {what} verdict failed (bits {bits:#x}, verdict bits "
        f"{bits & (GRAD_FAIL_MASK if gradient else OVERFLOW_FAIL_MASK):#x}: "
        f"{', '.join(names)}): a step ran on a "
        "truncated force or gradient, so its state and gradient are refused. "
        "Re-plan the caps on a forward pre-pass and run again."
    )


def value_and_grad_checked(
    fun: Callable[..., tuple[Any, Any]],
    argnums: Union[int, tuple[int, ...]] = 0,
    *,
    static_argnums: Union[int, tuple[int, ...]] = (),
    verdict_of: Optional[Callable[[Any], Any]] = None,
) -> Callable[..., tuple[tuple[Any, Any], Any]]:
    """``jax.jit(jax.value_and_grad(fun, has_aux=True))`` that refuses a failed verdict.

    ``fun(*args) -> (loss, aux)``, with ``aux`` the :class:`RolloutResult` (or its
    step aux) of the rollout the loss is computed on. The returned function runs
    the jitted value-and-gradient, then checks the GRADIENT's verdict on the host
    (:func:`check_rollout` with ``gradient=True``) and raises instead of returning
    a gradient of a truncated force or of unsymmetric gradient lists. Every array the loss
    reads must come in through ``args`` (masses included): closed over, it would
    be a constant of the compiled program, which XLA compiles differently.

    Parameters
    ----------
    fun : Callable[..., tuple[Any, Any]]
        The loss, returning ``(scalar, aux)``.
    argnums : Union[int, tuple[int, ...]]
        Arguments to differentiate, as in ``jax.value_and_grad``.
    static_argnums : Union[int, tuple[int, ...]]
        Arguments that are static under ``jax.jit`` (a step count, a checkpoint
        mode), as in ``jax.jit``.
    verdict_of : Optional[Callable[[Any], Any]]
        Maps ``fun``'s aux to what :func:`check_rollout` reads (a
        ``RolloutResult`` or a step aux); ``None``: the aux itself.

    Returns
    -------
    Callable[..., tuple[tuple[Any, Any], Any]]
        ``run(*args) -> ((loss, aux), grads)``; raises
        :class:`RolloutVerdictError` (from :func:`check_rollout`) on a failed
        verdict.
    """
    value_and_grad = jax.jit(
        jax.value_and_grad(fun, argnums=argnums, has_aux=True),
        static_argnums=static_argnums,
    )

    def run(*args: Any) -> tuple[tuple[Any, Any], Any]:
        (value, aux), grads = value_and_grad(*args)
        check_rollout(aux if verdict_of is None else verdict_of(aux), gradient=True)
        return (value, aux), grads

    return run
