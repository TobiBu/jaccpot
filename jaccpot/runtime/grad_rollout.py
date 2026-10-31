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
* ``"step"``: ``jax.checkpoint`` around the WHOLE step, drift and kick included.
  The scan then saves exactly each step's carry ``(x, v, a, aux)``: 36 B per
  particle in float32 (72 in float64) plus the aux leaves (the fused lane's f_b,
  4 B, under ``dehnen_error``). The masses, the force and external parameters and
  ``dt`` enter every step unchanged: the scan's partial evaluation hoists them out
  as loop invariants, so they are stored once, not per step
  (``tests/unit/runtime/test_grad_rollout.py`` reads the residuals off
  ``print_saved_residuals`` and the compiled temporaries off ``memory_analysis``).
  The reverse pass recomputes each step once.
* ``"nested"``: two levels. An outer scan over segments of ``inner_steps`` steps,
  each segment checkpointed whole and each step inside it too: the outer scan
  stores one carry per segment, and the reverse pass of one segment recomputes it
  storing one carry per step. About ``num_steps / inner_steps + inner_steps``
  carries (``2 sqrt(T)`` at the default ``inner_steps = round(sqrt(T))``) for one
  more forward per step than ``"step"``.

All three compute the same numbers: the forward is the same expressions, and the
gradients agree bitwise (float32 and float64, direct sum, CPU). The start force
(``acceleration=None``) is checkpointed with them, so its residuals -- one whole
force's -- are not held through the reverse pass either.

A ``num_steps`` the inner length does not divide gets a SHORTER LAST SEGMENT
(``num_steps % inner_steps`` steps, checkpointed the same way) rather than a
refusal: the horizon is the physics' choice, not the checkpointing's. The cost is
a second trace of the step body (the tail's scan), which on the fused lane's
interpret-mode CPU path is minutes of compile; pick an ``inner_steps`` that
divides ``num_steps`` to avoid it.

HOST OFFLOAD (session 2) changes WHERE the outer carries live, not what is stored:
:func:`_checkpointed` is the one place a rematerialisation policy is chosen, so an
offloading policy for the ``"segment"`` level replaces one line.

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
from typing import Any, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Callable
from jaxtyping import Array

from .fused_topology import OVERFLOW_FAIL_MASK, ForceParams, StepParams
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

    Every level saves nothing but its inputs (``jax.checkpoint``'s default
    policy): a step's inputs are its carry, a segment's are the carry at its
    start. Session 2's host offload gives the ``"segment"`` level an offloading
    policy here (the outer carries to pinned host memory) and changes nothing
    else.

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
        velocities).

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
    # a Python dt stays a literal of the program, an array a traced argument;
    # strict_run_v2 casts both to the state's dtype the same way
    dt_arr = jnp.asarray(dt, dtype=dtype)

    def force(x: Array, v: Array, aux_in: Any) -> tuple[Array, Any]:
        # the step's total acceleration at x, in the state's dtype, as the
        # particle carry builds it: the force, plus the field at (x, v)
        acc, aux_out = step_fn(x, masses, params, aux_in)
        acc = jnp.asarray(acc, dtype=dtype)
        field = _external_acceleration(external_fn, external_args, x, v)
        return (acc if field is None else acc + field), aux_out

    def step(carry: tuple) -> tuple:
        # one velocity-Verlet step (strict_run_v2's `_advance` and
        # `_velocity_verlet_kick_drifted`): drift, force at the new positions,
        # kick with the mean of the old and new accelerations
        x, v, a, aux_in = carry
        x = x + v * dt_arr + 0.5 * a * dt_arr**2
        a_new, aux_out = force(x, v, aux_in)
        v = v + 0.5 * (a + a_new) * dt_arr
        return x, v, a_new, aux_out

    if acceleration is None:
        # the start force: checkpointed with the steps, so its residuals (one
        # whole force's) are not held through the reverse pass
        start = force if checkpoint is None else _checkpointed(force, "edge")
        acceleration, aux = start(positions, velocities, aux)
    else:
        acceleration = jnp.asarray(acceleration, dtype=dtype)
    carry = (positions, velocities, acceleration, aux)

    def scan_steps(carry_in: tuple, length: int, body: Callable) -> tuple:
        return jax.lax.scan(lambda c, _: (body(c), None), carry_in, None, length)[0]

    if checkpoint is None:
        carry = scan_steps(carry, num_steps, step)
    elif checkpoint == "step":
        carry = scan_steps(carry, num_steps, _checkpointed(step, "step"))
    else:
        inner = (
            max(1, round(math.sqrt(num_steps)))
            if inner_steps is None
            else int(inner_steps)
        )
        inner = min(inner, num_steps)
        outer, tail = divmod(num_steps, inner)
        step_c = _checkpointed(step, "step")

        def segment(length: int) -> Callable:
            return lambda c: scan_steps(c, length, step_c)

        carry = scan_steps(carry, outer, _checkpointed(segment(inner), "segment"))
        if tail:
            carry = _checkpointed(segment(tail), "edge")(carry)
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


def _fail_mask(gradient: bool) -> int:
    """The verdict bits a state (or a gradient) is refused on.

    Parameters
    ----------
    gradient : bool
        The gradient's verdict: the force's bits and, where the provider defines
        it, ``GRAD_FAIL_MASK``'s (the differentiable step's gradient lists,
        ``OVERFLOW_GRAD_LISTS``, which the forward does not read).

    Returns
    -------
    int
        The mask: ``OVERFLOW_FAIL_MASK``, or ``GRAD_FAIL_MASK`` for a gradient.
    """
    from . import fused_topology

    if gradient:
        return int(getattr(fused_topology, "GRAD_FAIL_MASK", OVERFLOW_FAIL_MASK))
    return int(OVERFLOW_FAIL_MASK)


def rollout_ok(aux: Any, *, gradient: bool = False) -> bool:
    """Whether every step of a rollout fitted its capacities. Host sync.

    Parameters
    ----------
    aux : Any
        A :class:`RolloutResult` or its ``aux``.
    gradient : bool
        Check the gradient's verdict (``GRAD_FAIL_MASK``: the force's bits and the
        differentiable step's gradient-list bit) instead of the state's
        (``OVERFLOW_FAIL_MASK``: ``fused_topology.overflow_ok``).

    Returns
    -------
    bool
        ``True`` for a step without a verdict (``aux`` ``None``); else whether
        no bit of the mask is set in the folded bits.
    """
    bits = _verdict_bits(aux)
    if bits is None:
        return True
    return (int(np.asarray(bits, np.uint32)) & _fail_mask(gradient)) == 0


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
        and name != "OVERFLOW_FAIL_MASK"
        and bits & int(getattr(fused_topology, name))
    )
    what = "gradient" if gradient else "capacity"
    raise RolloutVerdictError(
        f"the rollout's {what} verdict failed (bits {bits:#x}, verdict bits "
        f"{bits & _fail_mask(gradient):#x}: {', '.join(names)}): a step ran on a "
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
