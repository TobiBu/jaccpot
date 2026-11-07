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

**On the mesh** (multi-node plan I2: ``ProductionRunner(mesh=..., scheme=...)``)
the same two calls run inside the segment's ``shard_map``, on ONE device's
``(cap, 2, 3)`` rows (padding rows included: zero mass, zero velocity, zero
acceleration). A ``rowwise`` scheme runs there unchanged. A scheme whose update
needs a mesh-wide quantity (a centre of mass, a global timestep) is not rowwise
and declares it by implementing :class:`ShardedSplitStepScheme`: its
``open_step_shard`` / ``close_step_shard`` get the live-row mask and the mesh
axis and reduce through collectives (``nornax.sharded.mesh_sum``: mesh order,
the same bits on any process layout). A non-rowwise scheme without them is
refused on the mesh (:func:`check_mesh_scheme`).

``docs/production.md`` ("Integration schemes") has the contract and the
I1/I2 record (``docs/production_2026-11.md``, I1 and I2) its measurements.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Protocol, runtime_checkable

from jaxtyping import Array

__all__ = [
    "ShardedSplitStepScheme",
    "SplitStepScheme",
    "check_mesh_scheme",
    "check_scheme",
    "scheme_close_shard",
    "scheme_open_shard",
    "scheme_record",
]


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


@runtime_checkable
class ShardedSplitStepScheme(SplitStepScheme, Protocol):
    """A scheme with a mesh form: its step reads mesh-wide quantities (I2).

    The mesh runner calls these instead of ``open_step`` / ``close_step`` when
    a scheme has them (rowwise or not); everything else is
    :class:`SplitStepScheme`'s contract on one device's rows. The reductions
    must be collectives over ``axis_name`` whose result does not depend on the
    reduction order of the transport (``nornax.sharded.mesh_sum``: an
    ``all_gather`` folded in mesh order), or one process with N devices and N
    processes with one device each stop being the same run.
    """

    def open_step_shard(
        self,
        state: Array,
        acceleration: Array,
        dt: Array,
        *,
        live_rows: Array,
        axis_name: str,
    ) -> Array:
        """:meth:`SplitStepScheme.open_step` on one device's rows.

        Parameters
        ----------
        state : Array
            ``(cap, 2, 3)`` this device's rows.
        acceleration : Array
            ``(cap, 3)`` their total acceleration.
        dt : Array
            The step.
        live_rows : Array
            ``(cap,)`` boolean: the rows that hold particles.
        axis_name : str
            The mesh axis to reduce over.

        Returns
        -------
        Array
            ``(cap, 2, 3)``.
        """
        ...

    def close_step_shard(
        self,
        state_open: Array,
        acceleration: Array,
        acceleration_new: Array,
        dt: Array,
        *,
        live_rows: Array,
        axis_name: str,
    ) -> Array:
        """:meth:`SplitStepScheme.close_step` on one device's rows.

        Parameters
        ----------
        state_open : Array
            ``(cap, 2, 3)``: :meth:`open_step_shard`'s rows (after a migration:
            the rows this device owns now).
        acceleration : Array
            ``(cap, 3)`` the step's starting total acceleration, same rows.
        acceleration_new : Array
            ``(cap, 3)`` the total acceleration at ``state_open``'s positions.
        dt : Array
            The step.
        live_rows : Array
            ``(cap,)`` boolean.
        axis_name : str
            The mesh axis.

        Returns
        -------
        Array
            ``(cap, 2, 3)``.
        """
        ...


def _has_shard_form(scheme: Any) -> bool:
    return callable(getattr(scheme, "open_step_shard", None)) and callable(
        getattr(scheme, "close_step_shard", None)
    )


def check_mesh_scheme(scheme: Any) -> Any:
    """:func:`check_scheme`, plus what the mesh needs; ``None`` passes through.

    Parameters
    ----------
    scheme : Any
        ``None`` or a :class:`SplitStepScheme`.

    Returns
    -------
    Any
        ``scheme``.

    Raises
    ------
    ValueError
        For a scheme that is not rowwise and has no mesh form: its step would
        read only one device's rows as if they were the system. (What
        :func:`check_scheme` refuses is a ``TypeError`` first.)
    """
    scheme = check_scheme(scheme)
    if scheme is None or bool(scheme.rowwise) or _has_shard_form(scheme):
        return scheme
    raise ValueError(
        f"{type(scheme).__name__} is not rowwise: on the mesh each device holds "
        "only its own rows, so a step that reads other rows (a centre of mass, a "
        "global timestep) must say how it reduces them. Implement "
        "jaccpot.scheme.ShardedSplitStepScheme (open_step_shard / "
        "close_step_shard, reducing with nornax.sharded.mesh_sum), or run it on "
        "one card (mesh=None)"
    )


def scheme_open_shard(
    scheme: Any,
    state: Array,
    acceleration: Array,
    dt: Array,
    live_rows: Array,
    axis_name: str,
) -> Array:
    """The scheme's opening half on one device's rows (inside ``shard_map``).

    Parameters
    ----------
    scheme : Any
        A scheme that passed :func:`check_mesh_scheme`.
    state : Array
        ``(cap, 2, 3)`` one device's rows.
    acceleration : Array
        ``(cap, 3)`` their total acceleration.
    dt : Array
        The step.
    live_rows : Array
        ``(cap,)`` boolean.
    axis_name : str
        The mesh axis.

    Returns
    -------
    Array
        ``(cap, 2, 3)``.
    """
    if _has_shard_form(scheme):
        return scheme.open_step_shard(
            state, acceleration, dt, live_rows=live_rows, axis_name=axis_name
        )
    return scheme.open_step(state, acceleration, dt)


def scheme_close_shard(
    scheme: Any,
    state_open: Array,
    acceleration: Array,
    acceleration_new: Array,
    dt: Array,
    live_rows: Array,
    axis_name: str,
) -> Array:
    """The scheme's closing half on one device's rows (inside ``shard_map``).

    Parameters
    ----------
    scheme : Any
        A scheme that passed :func:`check_mesh_scheme`.
    state_open : Array
        ``(cap, 2, 3)`` the opened rows.
    acceleration : Array
        ``(cap, 3)`` the step's starting total acceleration.
    acceleration_new : Array
        ``(cap, 3)`` the total acceleration at the opened positions.
    dt : Array
        The step.
    live_rows : Array
        ``(cap,)`` boolean.
    axis_name : str
        The mesh axis.

    Returns
    -------
    Array
        ``(cap, 2, 3)``.
    """
    if _has_shard_form(scheme):
        return scheme.close_step_shard(
            state_open,
            acceleration,
            acceleration_new,
            dt,
            live_rows=live_rows,
            axis_name=axis_name,
        )
    return scheme.close_step(state_open, acceleration, acceleration_new, dt)


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
