"""The integration-scheme seam of the production runner (multi-node plan I1, U1).

nornax owns the integration schemes, jaccpot the force and the run's
orchestration. :class:`~jaccpot.production.ProductionRunner` (and
``strict_run_v2(carry="particles")`` under it) takes a scheme through the small
structural protocol defined here, :class:`SplitStepScheme`, which nornax
implements (``nornax.schemes.LeapfrogKDK``) without jaccpot importing nornax: the
dependency stays ``nornax -> jaccpot``.

A step of the runner's scan is cut at its ONE force evaluation::

    state_open = scheme.open_step(state, acceleration, dt)
    acceleration_new = self-gravity + external field at state_open[:, 0]   # jaccpot
    state_new = scheme.close_step(state_open, acceleration, acceleration_new, dt)

and everything around the force -- the tree refresh, the capacity verdict, the
walk needs, the ``dehnen_error`` force scale, the tree-order permutation, the
segment retry, restarts, diagnostics and donation -- stays jaccpot's. The
carry's acceleration is always the total field at the carried positions (first
same as last), which the segment handle, the restart file and the next
segment's start force rely on. ``scheme=None`` (the default) is the runner's
built-in velocity-Verlet expressions, program for program.

``docs/production.md`` ("Integration schemes") has the contract and the
I1 record (``docs/production_2026-11.md``, I1) its measurements.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Protocol, runtime_checkable

from jaxtyping import Array

__all__ = ["SplitStepScheme", "check_scheme", "scheme_record"]


@runtime_checkable
class SplitStepScheme(Protocol):
    """One step of the runner's scan, split at its single force evaluation.

    Attributes
    ----------
    rowwise : bool
        Each row's update reads only that row (and ``dt``). The tree-order carry
        (``JACCPOT_STRICT_CARRY_ORDER=tree``) hands the scheme its rows in the
        previous step's tree order and accepts rowwise schemes only.

    Notes
    -----
    Both methods are traced inside the runner's jitted scan: pure functions of
    their arguments, jittable, returning arrays of the input's shape and dtype.
    ``state`` is the packed ``[N, 2, 3]`` carry (positions, velocities),
    ``acceleration`` the TOTAL acceleration ``[N, 3]`` at its positions (self plus
    external), ``dt`` the step's timestep (a literal or a traced scalar, in the
    state's dtype). The runner evaluates the force at ``state_open[:, 0]``
    between the two calls, exactly once per step.
    """

    rowwise: bool

    def key(self) -> Hashable:
        """The scheme's static configuration, part of the runner's compile keys.

        Returns
        -------
        Hashable
            Equal keys must mean the same traced arithmetic.
        """
        ...

    def open_step(self, state: Array, acceleration: Array, dt: Array) -> Array:
        """Everything before the step's force evaluation.

        Parameters
        ----------
        state : Array
            ``[N, 2, 3]`` at the step's start.
        acceleration : Array
            ``[N, 3]`` total acceleration at ``state``'s positions.
        dt : Array
            The step's timestep.

        Returns
        -------
        Array
            ``[N, 2, 3]``: its positions are where the force is evaluated.
        """
        ...

    def close_step(
        self,
        state_open: Array,
        acceleration: Array,
        acceleration_new: Array,
        dt: Array,
    ) -> Array:
        """Everything after it.

        Parameters
        ----------
        state_open : Array
            :meth:`open_step`'s result (rows permuted by the tree-order carry).
        acceleration : Array
            The step's starting total acceleration, in ``state_open``'s rows.
        acceleration_new : Array
            The total acceleration at ``state_open``'s positions.
        dt : Array
            The step's timestep.

        Returns
        -------
        Array
            ``[N, 2, 3]`` at the step's end; its positions must be
            ``state_open``'s (the force was evaluated there).
        """
        ...


def check_scheme(scheme: Any) -> Any:
    """Validate a scheme for the runner; ``None`` passes through.

    Parameters
    ----------
    scheme : Any
        ``None`` (the built-in velocity Verlet) or a :class:`SplitStepScheme`.

    Returns
    -------
    Any
        ``scheme``.

    Raises
    ------
    TypeError
        If it lacks the protocol's members or its key is not hashable.
    """
    if scheme is None:
        return None
    missing = [
        name
        for name in ("rowwise", "key", "open_step", "close_step")
        if not hasattr(scheme, name)
    ]
    if missing:
        raise TypeError(
            f"{type(scheme).__name__} is not a jaccpot.scheme.SplitStepScheme: "
            f"missing {', '.join(missing)} (nornax.schemes.LeapfrogKDK is one)"
        )
    try:
        hash(scheme.key())
    except TypeError as exc:
        raise TypeError(
            f"{type(scheme).__name__}.key() must be hashable (it is part of the "
            f"runner's compile keys): {exc}"
        ) from exc
    return scheme


def scheme_record(scheme: Any) -> dict | None:
    """What a restart file and a run record say about the scheme.

    Parameters
    ----------
    scheme : Any
        ``None`` or a :class:`SplitStepScheme`.

    Returns
    -------
    dict | None
        ``{"qualname": ..., "key": repr(key)}``, or ``None`` for the built-in.
    """
    if scheme is None:
        return None
    cls = type(scheme)
    return {
        "qualname": f"{cls.__module__}.{cls.__qualname__}",
        "key": repr(scheme.key()),
    }
