"""Position precision of the float32 particle state (multi-node plan A3, U2).

In a float32 state the production drift ``x + v dt + 0.5 a dt**2`` is evaluated
left to right: ``x + v dt`` is rounded to ``x``'s float32 grid, and then the
second-order term ``0.5 a dt**2`` -- smaller than ``x``'s last bit for most
particles at ``dt`` 5e-4 -- is rounded against ``x`` again, on its own. That loss is
systematic (the same sign along an orbit), so the velocity-Verlet step degrades
towards first order in the positions and the energy drifts linearly in time,
independent of N: the merger probe of 2026-10-31 (``docs/production_2026-11.md``,
A3) drifted 7x more in energy and 24x more in angular momentum than the same run
with a float64 state, whose force was the very same float32 force.

Two opt-in schemes here keep the float32 force (it reads float32 positions, as
every lane does) and fix the state, through the runner's integration-scheme seam
(:class:`jaccpot.scheme.SplitStepScheme`; ``ProductionRunner(scheme=...)``,
``strict_run_v2(scheme=...)``):

:class:`SummedDriftKDK`
    the float32 ``[N, 2, 3]`` state, with the drift's increment summed first:
    ``x + (v dt + 0.5 a dt**2)``. One rounding of ``x`` per step, unbiased, and no
    byte more than the default.
:class:`CompensatedKDK`
    the positions (and, optionally, the velocities) as an unevaluated sum of two
    float32 numbers, ``x = hi + lo`` with ``|lo| <= ulp(hi) / 2`` (a "double-float";
    the anchor is ``hi`` itself, a per-particle float32 anchor, the offset ``lo``).
    The state is ``[N, 3, 3]`` (rows: ``hi``, ``v``, ``lo``) or ``[N, 4, 3]``
    (``x_hi``, ``v_hi``, ``x_lo``, ``v_lo``): rows 0 and 1 stay the float32
    positions and velocities every reader of a state reads (the force, the external
    field, the diagnostics), and ``hi`` is ``x`` correctly rounded to float32 -- what
    a float64 state's cast hands the force. The updates are Knuth's TwoSum plus one
    renormalisation: about 48 bits of position, 12 (24) B per particle more.

Both are rowwise (any row order: the tree-order carry, a mesh shard).

:func:`compensated_state` packs a float64 (or float32) ``[N, 2, 3]`` state into a
compensated one, :func:`state_float64` unpacks either form to float64 ``[N, 2, 3]``.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

__all__ = [
    "CompensatedKDK",
    "SummedDriftKDK",
    "compensated_state",
    "state_float64",
]


def _two_sum(a: Array, b: Array) -> tuple[Array, Array]:
    """Knuth's TwoSum: ``s = fl(a + b)`` and its exact rounding error ``e``.

    Parameters
    ----------
    a : Array
        First addend.
    b : Array
        Second addend.

    Returns
    -------
    tuple[Array, Array]
        ``(s, e)`` with ``s + e == a + b`` exactly.
    """
    s = a + b
    b_virtual = s - a
    return s, (a - (s - b_virtual)) + (b - b_virtual)


def _compensated_add(hi: Array, lo: Array, delta: Array) -> tuple[Array, Array]:
    """``(hi + lo) + delta`` as a renormalised pair (``hi`` the rounded sum).

    Parameters
    ----------
    hi : Array
        The pair's leading part.
    lo : Array
        Its residual.
    delta : Array
        The increment.

    Returns
    -------
    tuple[Array, Array]
        The new ``(hi, lo)``, ``|lo| <= ulp(hi) / 2``.
    """
    s, e = _two_sum(hi, delta)
    lo_sum = lo + e
    hi_new = s + lo_sum
    # Dekker's FastTwoSum (|s| >= |lo_sum|): the part of lo_sum hi_new dropped
    return hi_new, lo_sum - (hi_new - s)


def _drift_increment(velocity: Array, acceleration: Array, dt: Array) -> Array:
    # ``v dt + 0.5 a dt**2``: the built-in drift's terms, summed among themselves
    return velocity * dt + 0.5 * acceleration * dt**2


def _kick_increment(acceleration: Array, acceleration_new: Array, dt: Array) -> Array:
    # the built-in kick's increment (``_velocity_verlet_kick_drifted``)
    return 0.5 * (acceleration + acceleration_new) * dt


class SummedDriftKDK:
    """Velocity Verlet with the drift's increment summed before it meets ``x``.

    ``open_step``: ``x' = x + (v dt + 0.5 a dt**2)``; ``close_step``: the built-in
    kick ``v' = v + 0.5 (a + a') dt``. The state is the default ``[N, 2, 3]``.

    Attributes
    ----------
    rowwise : bool
        ``True``.
    """

    rowwise: bool = True

    def key(self) -> Hashable:
        """The compile-key part.

        Returns
        -------
        Hashable
            ``("jaccpot.SummedDriftKDK", 1)``.
        """
        return ("jaccpot.SummedDriftKDK", 1)

    def open_step(self, state: Array, acceleration: Array, dt: Array) -> Array:
        """The drift, increment first.

        Parameters
        ----------
        state : Array
            ``[N, 2, 3]``.
        acceleration : Array
            ``[N, 3]`` total acceleration at the state's positions.
        dt : Array
            The step.

        Returns
        -------
        Array
            ``[N, 2, 3]``: the new positions, the starting velocities.
        """
        acceleration = jnp.asarray(acceleration, state.dtype)
        return state.at[:, 0].set(
            state[:, 0] + _drift_increment(state[:, 1], acceleration, dt)
        )

    def close_step(
        self, state_open: Array, acceleration: Array, acceleration_new: Array, dt: Array
    ) -> Array:
        """The kick.

        Parameters
        ----------
        state_open : Array
            :meth:`open_step`'s result.
        acceleration : Array
            ``[N, 3]`` at the step's start.
        acceleration_new : Array
            ``[N, 3]`` at the opened positions.
        dt : Array
            The step.

        Returns
        -------
        Array
            ``[N, 2, 3]``.
        """
        dtype = state_open.dtype
        return state_open.at[:, 1].set(
            state_open[:, 1]
            + _kick_increment(
                jnp.asarray(acceleration, dtype),
                jnp.asarray(acceleration_new, dtype),
                dt,
            )
        )


class CompensatedKDK:
    """Velocity Verlet on a compensated (double-float32) position state.

    Parameters
    ----------
    velocities : bool
        Compensate the velocities too (``[N, 4, 3]`` state). ``False``
        (default): ``[N, 3, 3]``, float32 velocities.

    Attributes
    ----------
    rowwise : bool
        ``True``.
    """

    rowwise: bool = True

    def __init__(self, velocities: bool = False) -> None:
        self.velocities = bool(velocities)
        #: the state's rows: 3, or 4 with compensated velocities
        self.rows = 4 if self.velocities else 3

    def key(self) -> Hashable:
        """The compile-key part.

        Returns
        -------
        Hashable
            ``("jaccpot.CompensatedKDK", 1, velocities)``.
        """
        return ("jaccpot.CompensatedKDK", 1, self.velocities)

    def _check(self, state: Array) -> None:
        if state.ndim != 3 or int(state.shape[1]) != self.rows:
            raise ValueError(
                f"CompensatedKDK(velocities={self.velocities}) steps an "
                f"[N, {self.rows}, 3] state (compensated_state()), got "
                f"{tuple(state.shape)}"
            )

    def open_step(self, state: Array, acceleration: Array, dt: Array) -> Array:
        """The drift, compensated: ``(hi, lo) += v dt + 0.5 a dt**2``.

        Parameters
        ----------
        state : Array
            ``[N, rows, 3]``.
        acceleration : Array
            ``[N, 3]`` total acceleration at ``hi``.
        dt : Array
            The step.

        Returns
        -------
        Array
            ``[N, rows, 3]``: row 0 the new positions rounded to float32 (the
            force's), row 2 their residual.
        """
        self._check(state)
        delta = _drift_increment(
            state[:, 1], jnp.asarray(acceleration, state.dtype), dt
        )
        if self.velocities:
            # the velocity's own residual enters the drift (one float32 product)
            delta = delta + state[:, 3] * dt
        hi, lo = _compensated_add(state[:, 0], state[:, 2], delta)
        return state.at[:, 0].set(hi).at[:, 2].set(lo)

    def close_step(
        self, state_open: Array, acceleration: Array, acceleration_new: Array, dt: Array
    ) -> Array:
        """The kick (compensated when :attr:`velocities`).

        Parameters
        ----------
        state_open : Array
            :meth:`open_step`'s result.
        acceleration : Array
            ``[N, 3]`` at the step's start.
        acceleration_new : Array
            ``[N, 3]`` at the opened positions.
        dt : Array
            The step.

        Returns
        -------
        Array
            ``[N, rows, 3]``.
        """
        self._check(state_open)
        dtype = state_open.dtype
        delta = _kick_increment(
            jnp.asarray(acceleration, dtype), jnp.asarray(acceleration_new, dtype), dt
        )
        if not self.velocities:
            return state_open.at[:, 1].set(state_open[:, 1] + delta)
        hi, lo = _compensated_add(state_open[:, 1], state_open[:, 3], delta)
        return state_open.at[:, 1].set(hi).at[:, 3].set(lo)


def compensated_state(state: Any, velocities: bool = False) -> np.ndarray:
    """Pack a ``[N, 2, 3]`` state into :class:`CompensatedKDK`'s float32 rows.

    Parameters
    ----------
    state : Any
        ``[N, 2, 3]``, float64 (the residuals keep its bits) or float32 (zero
        residuals). Host or device; converted on the host.
    velocities : bool
        Also split the velocities (``[N, 4, 3]``).

    Returns
    -------
    np.ndarray
        float32 ``[N, 3, 3]`` (``hi``, ``v``, ``lo``) or ``[N, 4, 3]``.

    Raises
    ------
    ValueError
        If ``state`` is not ``[N, 2, 3]``.
    """
    s64 = np.asarray(state, np.float64)
    if s64.ndim != 3 or s64.shape[1:] != (2, 3):
        raise ValueError(f"compensated_state takes an [N, 2, 3] state, got {s64.shape}")
    hi = s64.astype(np.float32)
    lo = (s64 - hi.astype(np.float64)).astype(np.float32)
    rows = [hi[:, 0], hi[:, 1], lo[:, 0]]
    if velocities:
        rows.append(lo[:, 1])
    return np.stack(rows, axis=1)


def state_float64(state: Any) -> np.ndarray:
    """Either state form as float64 ``[N, 2, 3]`` (``hi + lo`` where compensated).

    Parameters
    ----------
    state : Any
        ``[N, 2, 3]`` (any float dtype) or a :class:`CompensatedKDK` state
        ``[N, 3, 3]`` / ``[N, 4, 3]``.

    Returns
    -------
    np.ndarray
        float64 ``[N, 2, 3]``.

    Raises
    ------
    ValueError
        If ``state`` is none of these shapes.
    """
    s = np.asarray(state)
    if s.ndim != 3 or s.shape[2] != 3 or s.shape[1] not in (2, 3, 4):
        raise ValueError(f"not a particle state: {s.shape}")
    out = s[:, :2].astype(np.float64)
    if s.shape[1] >= 3:
        out[:, 0] += s[:, 2].astype(np.float64)
    if s.shape[1] == 4:
        out[:, 1] += s[:, 3].astype(np.float64)
    return out
