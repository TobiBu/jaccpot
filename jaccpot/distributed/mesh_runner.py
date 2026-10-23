"""The production runner's multi-GPU mode: :class:`~jaccpot.production.ProductionRunner` on the mesh.

``ProductionRunner(solver, ..., mesh=...)`` (a 1-D ``jax.sharding.Mesh`` or a
:class:`~jaccpot.production.MeshRunConfig`) dispatches :meth:`~jaccpot.production.ProductionRunner.run`
and :meth:`~jaccpot.production.ProductionRunner.save_restart` here. The single
card's loop -- segments, edges, callbacks (a truthy return stops the run),
diagnostics, an in-memory continuation (restart files: not yet) -- runs on the fused multi-GPU
lane's :class:`~jaccpot.distributed.segment.MeshSegmentRunner`, whose segments are
one ``jit(shard_map(scan))`` program each, with its own segment retry and its
repartition between segments. ``mesh=None`` never reaches this module: the single
card's programs are untouched.

**Input (id) order on the API, the mesh layout inside.** A fresh run decomposes the
caller's ``[N, 2, 3]`` state over the mesh (``rollout.decompose``: Morton domains,
``cap`` rows per device, ids ``0..N-1`` in input order); :attr:`RunResult.state
<jaccpot.production.RunResult.state>` is the final state gathered back by id (one
device-to-host copy at the end) and :attr:`RunResult.handle
<jaccpot.production.RunResult.handle>` the segment runner, so a continuation from
the result runs on in memory, bitwise. An edge gathers nothing unless asked
(:meth:`~jaccpot.production.MeshSegmentEdge.state_host`).

**Parameters.** ``dt`` and the external field's parameters are program arguments;
``G`` and the softening are the force's own (the fused mesh force is static in
them, as the single card under ``mac_type='dehnen_error'``): ``force_params`` must
be ``None`` or equal them. The external field and potential see one device's
``(cap, 2, 3)`` rows -- the single card's ``field(state, params)`` contract
restricted to rows, so a per-particle field works unchanged.

**Diagnostics.** The moments are the single card's fp64 reductions over the flat
sharded arrays: a padding row has zero mass and zero velocity, so every moment is
exact without a mask. The energy is ``K + W_self + W_ext`` from
:meth:`MeshSegmentRunner.potential_energy
<jaccpot.distributed.segment.MeshSegmentRunner.potential_energy>` (the force's
``potential_shard``; a force without one reports the energy fields as ``None`` and
``energy_unavailable``, the reason). The mesh force is prepared before the first
segment, so step 0's ``E0`` is evaluated AT step 0 (one card waits for the first
segment's provider).

"""

from __future__ import annotations

import dataclasses
import time
import warnings
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["MeshBackend", "gather_by_id"]


def _host(arr: Any) -> np.ndarray:
    """A device array on the host (all shards).

    Parameters
    ----------
    arr : Any
        A (sharded) device array.

    Returns
    -------
    np.ndarray
        The host copy.
    """
    return np.asarray(jax.device_get(arr))


def _live_rows(count: np.ndarray, cap: int) -> np.ndarray:
    """Flat row indices of every device's live rows, in mesh order.

    Parameters
    ----------
    count : np.ndarray
        ``(ndev,)`` live rows per device.
    cap : int
        Rows per device.

    Returns
    -------
    np.ndarray
        ``(N,)`` row indices.
    """
    return np.concatenate(
        [np.arange(d * cap, d * cap + int(c)) for d, c in enumerate(count)]
    ).astype(np.int64)


def _by_id(flat: dict, ids_flat: np.ndarray, count: np.ndarray) -> dict:
    """Host layout arrays in id order, with the identity checked.

    Parameters
    ----------
    flat : dict
        ``name -> (ndev * cap, ...)`` host arrays.
    ids_flat : np.ndarray
        ``(ndev * cap,)`` global ids (``-1`` on padding).
    count : np.ndarray
        ``(ndev,)`` live rows per device.

    Returns
    -------
    dict
        ``name -> (N, ...)`` in id order, and ``owner`` (device per id).

    Raises
    ------
    ValueError
        If the live ids are not exactly ``0..N-1`` once each.
    """
    ndev = int(count.shape[0])
    cap = int(ids_flat.shape[0]) // ndev
    rows = _live_rows(count, cap)
    ids = ids_flat[rows].astype(np.int64)
    n = int(rows.shape[0])
    seen = np.zeros(n, bool)
    if n:
        if ids.min() < 0 or ids.max() >= n:
            raise ValueError("particle identity broken: an id outside 0..N-1")
        seen[ids] = True
    if not seen.all():
        raise ValueError("particle identity broken: ids are not exactly 0..N-1 once")
    out = {}
    for name, arr in flat.items():
        full = np.empty((n,) + arr.shape[1:], arr.dtype)
        full[ids] = arr[rows]
        out[name] = full
    owner = np.empty(n, np.int32)
    owner[ids] = rows // cap
    out["owner"] = owner
    return out


def gather_by_id(
    layout: Any, names: Sequence = ("state", "masses", "acceleration")
) -> dict:
    """A :class:`~jaccpot.distributed.segment.MeshState`'s arrays on the host, by id.

    One device-to-host copy of each named array and of the ids and counts; the
    live rows of every device placed at their global ids. What
    :meth:`~jaccpot.production.MeshSegmentEdge.state_host` and the runner's
    :attr:`~jaccpot.production.RunResult.state` read.

    Parameters
    ----------
    layout : Any
        The mesh state.
    names : Sequence
        Of its array fields (``state``, ``masses``, ``acceleration``,
        ``force_scale``); one that is ``None`` (not computed yet) is left out.

    Returns
    -------
    dict
        ``name -> (N, ...)`` host arrays in id order, and ``owner`` (the device
        that holds each id).
    """
    flat = {
        name: _host(getattr(layout, name))
        for name in names
        if getattr(layout, name) is not None
    }
    return _by_id(flat, _host(layout.ids), _host(layout.count))


class MeshBackend:
    """:class:`~jaccpot.production.ProductionRunner`'s multi-GPU mode.

    Built by the runner (``ProductionRunner(..., mesh=...)``), which keeps its
    public surface: :meth:`run` and :meth:`save_restart` are the runner's, read
    the runner's ``segment``, ``callbacks``, fields and settings at every call
    (a caller may change them between runs, as Odisseo's thin client does) and
    keep the runner's ``_current`` edge record, which its restart cadence reads.

    Parameters
    ----------
    runner : Any
        The :class:`~jaccpot.production.ProductionRunner`; its ``mesh`` is the
        :class:`~jaccpot.production.MeshRunConfig`.
    """

    def __init__(self, runner: Any) -> None:
        from jaccpot.distributed.rollout import _mesh_devices

        self.runner = runner
        self.config = runner.mesh
        self.mesh = self.config.mesh
        self.axis_name = str(self.mesh.axis_names[0])
        self.devices = _mesh_devices(self.mesh)
        self.ndev = len(self.devices)
        #: whether this backend warned that its force has no potential
        self._energy_warned = False

    # ------------------------------------------------------------------
    # The loop.
    # ------------------------------------------------------------------

    def run(
        self,
        state: Any,
        masses: Any,
        *,
        dt: Any,
        num_steps: int,
        t0: float,
        force_params: Any,
        external_params: Any,
        restart_from: Any,
    ) -> Any:
        """:meth:`ProductionRunner.run <jaccpot.production.ProductionRunner.run>` on the mesh.

        Parameters
        ----------
        state : Any
            ``[N, 2, 3]`` in input (id) order; ``None`` with ``restart_from``.
        masses : Any
            ``[N]``; ``None`` with ``restart_from``.
        dt : Any
            Timestep (a program argument: one compile for every value).
        num_steps : int
            The run's total length in steps from ``t0``.
        t0 : float
            Start time (a continued run keeps its own).
        force_params : Any
            ``None``, or a ``ForceParams`` equal to the force's own ``G`` and
            softening squared.
        external_params : Any
            The external field's parameters, or ``None``.
        restart_from : Any
            ``None`` or a mesh :class:`~jaccpot.production.RunResult`.

        Returns
        -------
        Any
            A :class:`~jaccpot.production.RunResult` with ``mode="mesh"``:
            ``state`` and ``masses`` gathered by id on the host, ``handle`` the
            :class:`~jaccpot.distributed.segment.MeshSegmentRunner`.

        Raises
        ------
        ValueError
            Without ``state`` and ``masses`` for a fresh run, for force
            parameters other than the force's own, a ``step_callback``, a
            single-card result, or a decomposition that does not fit ``cap``.
        TypeError
            For a ``restart_from`` that is none of the above.
        RuntimeError
            When a segment fails after the segment runner's own re-plan.
        """
        from jaccpot._env import env_text
        from jaccpot.production import (
            MeshSegmentEdge,
            RunResult,
            _baselines,
            _diagnostics_row,
            _moments,
            _warn_fusion_autotuner,
            segment_lengths,
        )

        runner = self.runner
        if runner.step_callback is not None:
            raise ValueError(
                "the multi-GPU mode has no in-scan step_callback; use a callback "
                "at the segment edges"
            )
        started = time.perf_counter()
        dt_value = float(np.asarray(dt))
        restored: Optional[dict] = None
        if restart_from is None:
            if state is None or masses is None:
                raise ValueError("a fresh run needs state and masses")
            self._check_force_params(force_params, self.config.force)
            host_state = _host(state)
            host_masses = _host(masses)
            external_arg = self._external_arg(external_params, host_state.dtype)
            msr = self._fresh(host_state, host_masses)
            del host_state, host_masses
            step, segment_done = 0, 0
            t0_run = float(t0)
            anchor: tuple = (t0_run, 0)
            baselines: Optional[dict] = None
        elif isinstance(restart_from, RunResult):
            res = restart_from
            if res.mode != "mesh" or res.handle is None:
                raise ValueError(
                    "a multi-GPU run continues a multi-GPU RunResult whose handle "
                    f"is still there (mode {res.mode!r}, handle "
                    f"{'consumed' if res.handle is None else 'kept'})"
                )
            msr = res.handle
            self._check_force_params(force_params, msr.force)
            step, segment_done, t0_run = res.step, res.segment, res.t0
            anchor = res.time_anchor if dt_value == res.dt else (res.time, res.step)
            baselines = res.baselines
            external_arg = self._external_arg(external_params, msr.state.state.dtype)
            restored = {"from": "RunResult", "step": step, "bitwise": True}
            res.state = res.handle = None  # consumed
        else:
            raise TypeError(
                "restart_from must be a multi-GPU RunResult, got "
                f"{type(restart_from).__name__}"
            )
        _warn_fusion_autotuner(self.devices[0].platform, env_text("XLA_FLAGS"))

        def time_at(step_now: int) -> float:
            return anchor[0] + (step_now - anchor[1]) * dt_value

        config = runner.segment
        names = tuple(config.diagnostics)
        every = int(config.diagnostics_every)
        every_energy = int(config.diagnostics_every_energy)
        energy_on = every > 0 and every_energy > 0 and "energy" in names
        edges: list = []
        restarts: list = []
        stopped = False

        def diagnostics_at(step_now: int, due: bool) -> Optional[dict]:
            nonlocal baselines
            if every == 0 or not due:
                return None
            layout = msr.state
            moments = _moments(layout.state, layout.masses)
            if baselines is None:
                baselines = dict(_baselines(moments, time_at(step_now)), step=step_now)
            energy_due = every_energy > 0 and segment_done % every_energy == 0
            potential, reason = None, None
            if energy_on and energy_due:
                potential, reason = self._potential(msr, external_arg)
                if (
                    potential is not None
                    and potential["ok"]
                    and baselines.get("energy") is None
                    and step_now == baselines.get("step")
                ):
                    # E0 at step 0 itself: the mesh force is prepared already
                    w0 = potential["self"] + potential["external"]
                    baselines["potential"] = w0
                    baselines["energy"] = float(baselines["kinetic"]) + w0
            row = _diagnostics_row(
                moments, baselines, time_at(step_now), names, energy_due, potential
            )
            if reason is not None:
                row["energy_unavailable"] = reason
            return row

        def edge_at(retries: int, wall_s: float, due: bool, report: Any) -> Any:
            layout = msr.state
            return MeshSegmentEdge(
                step=step,
                time=time_at(step),
                segment=segment_done,
                retries=retries,
                wall_s=wall_s,
                state=layout.state,
                masses=layout.masses,
                self_acceleration=None,
                diagnostics=diagnostics_at(step, due),
                plan=self._plan_record(msr, report),
                layout=layout,
            )

        def at_edge(edge: Any) -> bool:
            # the position save_restart writes, the restart if due, the callbacks
            runner._current = dict(
                handle=msr,
                step=step,
                time=edge.time,
                dt=dt_value,
                t0=t0_run,
                segment=segment_done,
                num_steps=int(num_steps),
                baselines=baselines,
                anchor=anchor,
                external_params=external_arg,
                plan=edge.plan,
            )
            restart = runner._restart_due(segment_done)
            if restart is not None:
                restarts.append(restart)
            stop = False
            for callback in runner.callbacks:
                result = callback(edge)
                if result is not None and bool(result):
                    stop = True
            edges.append(
                dataclasses.replace(
                    edge, state=None, masses=None, self_acceleration=None, layout=None
                )
            )
            return stop

        if restart_from is None:
            stopped = at_edge(edge_at(0, 0.0, True, None))
        missing = max(0, int(num_steps) - step)
        lengths = () if stopped else segment_lengths(missing, config.steps)
        runner._warn_remainder(lengths)
        for index, steps in enumerate(lengths):
            begun = time.perf_counter()
            try:
                report = msr.run(int(steps), dt=dt_value, external_params=external_arg)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"ProductionRunner (mesh, {self.ndev} devices): segment "
                    f"{segment_done + 1} (steps {step} to {step + int(steps)}) "
                    "failed after the mesh segment runner's own re-plan. Last "
                    f"restart: {runner._last_restart or 'none written'}. {exc}"
                ) from exc
            jax.block_until_ready(msr.state.state)
            wall_s = time.perf_counter() - begun
            step += int(steps)
            segment_done += 1
            last = index == len(lengths) - 1
            due = every > 0 and (segment_done % every == 0 or last)
            if at_edge(edge_at(int(report.retries), wall_s, due, report)):
                stopped = True
                break
        gathered = gather_by_id(msr.state, ("state", "masses"))
        return RunResult(
            state=gathered["state"],
            masses=gathered["masses"],
            handle=msr,
            step=step,
            time=time_at(step),
            dt=dt_value,
            t0=t0_run,
            segment=segment_done,
            edges=edges,
            baselines=baselines,
            stopped=stopped,
            restarts=restarts,
            restored=restored,
            wall_s=time.perf_counter() - started,
            time_anchor=anchor,
            state_owned=True,
            mode="mesh",
        )

    # ------------------------------------------------------------------
    # Building the segment runner.
    # ------------------------------------------------------------------

    def _force(self, *, cross_caps: Any = None, level_floor: Any = None) -> Any:
        """The run's force: the configuration's object, or a new fused mesh force.

        Parameters
        ----------
        cross_caps : Any
            Cross capacities for the fused force (a restore's); ``None`` takes the
            configuration's.
        level_floor : Any
            The fused force's level floor (a restore's), or ``None``.

        Returns
        -------
        Any
            The force (:class:`~jaccpot.distributed.segment.MeshSegmentRunner`'s
            protocol).
        """
        if self.config.force is not None:
            return self.config.force
        from jaccpot.distributed.template_force import FusedMeshForce

        settings = self.runner.settings
        return FusedMeshForce(
            self.runner.solver,
            self.mesh,
            leaf_size=int(settings.leaf_size),
            max_order=int(settings.max_order),
            cross_caps=self.config.cross_caps if cross_caps is None else cross_caps,
            level_floor=level_floor,
            axis_name=self.axis_name,
        )

    def _segment_runner(self, layout: Any, force: Any) -> Any:
        """A :class:`~jaccpot.distributed.segment.MeshSegmentRunner` at ``layout``.

        Parameters
        ----------
        layout : Any
            The :class:`~jaccpot.distributed.segment.MeshState`.
        force : Any
            The force; prepared here (and the cross capacities sized, under
            ``cross_caps="auto"``).

        Returns
        -------
        Any
            The segment runner, the state donated to its segments.
        """
        from jaccpot.distributed.segment import MeshSegmentRunner

        return MeshSegmentRunner(
            self.mesh,
            layout,
            force,
            external_acceleration_fn=self.runner.external_acceleration_fn,
            repartition_every=int(self.config.repartition_every),
            num_samples=int(self.config.num_samples),
            donate=True,
            axis_name=self.axis_name,
        )

    def _cap(self, num_particles: int) -> int:
        """Rows per device: the configuration's, or the library default.

        Parameters
        ----------
        num_particles : int
            N.

        Returns
        -------
        int
            ``cap``.
        """
        from jaccpot.production import mesh_shard_capacity

        if self.config.cap is not None:
            return int(self.config.cap)
        return mesh_shard_capacity(
            num_particles, self.ndev, float(self.config.cap_headroom)
        )

    def _decomposed(self, state: np.ndarray, masses: np.ndarray) -> Any:
        """The host by-id particles decomposed over the mesh.

        Parameters
        ----------
        state : np.ndarray
            ``[N, 2, 3]`` in id order.
        masses : np.ndarray
            ``[N]``.

        Returns
        -------
        Any
            The :class:`~jaccpot.distributed.segment.MeshState` (ids ``0..N-1``,
            no acceleration yet).
        """
        from jaccpot.distributed.rollout import decompose
        from jaccpot.distributed.segment import MeshState

        shard, _ = decompose(
            self.mesh,
            state[:, 0],
            state[:, 1],
            masses,
            cap=self._cap(int(state.shape[0])),
            num_samples=int(self.config.num_samples),
            axis_name=self.axis_name,
        )
        return MeshState.from_shard_state(shard, self.mesh, axis_name=self.axis_name)

    def _fresh(self, state: np.ndarray, masses: np.ndarray) -> Any:
        """A fresh run's segment runner: decomposed, its force prepared.

        Parameters
        ----------
        state : np.ndarray
            ``[N, 2, 3]`` host state in input order.
        masses : np.ndarray
            ``[N]`` host masses.

        Returns
        -------
        Any
            The segment runner.
        """
        return self._segment_runner(self._decomposed(state, masses), self._force())

    # ------------------------------------------------------------------
    # Parameters, diagnostics, the plan record.
    # ------------------------------------------------------------------

    def _own_force_params(self, force: Any) -> tuple:
        """``(G, eps^2)`` the force is static in.

        Parameters
        ----------
        force : Any
            The run's force, or ``None`` (the fused force: the solver's).

        Returns
        -------
        tuple
            Python floats.
        """
        if force is not None and hasattr(force, "G") and hasattr(force, "softening"):
            return float(force.G), float(force.softening) ** 2
        engine = self.runner.engine
        return float(engine.G), float(engine.softening) ** 2

    def _check_force_params(self, force_params: Any, force: Any) -> None:
        """Refuse force parameters the mesh force cannot take.

        Parameters
        ----------
        force_params : Any
            The caller's ``ForceParams``, or ``None``.
        force : Any
            The run's force (``None``: the fused force).

        Raises
        ------
        ValueError
            For any value but the force's own ``G`` and softening squared.
        """
        if force_params is None:
            return
        own = self._own_force_params(force)
        try:
            given = (
                float(np.asarray(force_params.G)),
                float(np.asarray(force_params.eps2)),
            )
        except (TypeError, jax.errors.ConcretizationTypeError):
            given = None
        if given != own:
            raise ValueError(
                "the multi-GPU mode's force is static in G and the softening: its "
                "programs are traced with the force's own (the fused mesh force's "
                "are the solver's, as the single card's under "
                "mac_type='dehnen_error'), so force_params must be None or equal "
                f"them (G={own[0]}, eps2={own[1]}); got {force_params!r}. Another "
                "G or softening is another solver."
            )

    def _external_arg(self, external_params: Any, dtype: Any) -> Any:
        """The field's parameters as the segments pass them: floats as arrays.

        Parameters
        ----------
        external_params : Any
            The caller's parameters, or ``None``.
        dtype : Any
            The state's dtype.

        Returns
        -------
        Any
            The same pytree, Python floats as arrays in ``dtype`` (program
            arguments, the single card's ``traced_params``).
        """
        if external_params is None:
            return None
        return jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype) if isinstance(x, float) else x,
            external_params,
        )

    def _potential(self, msr: Any, external_arg: Any) -> tuple:
        """The edge's potential energies, or why there are none.

        Parameters
        ----------
        msr : Any
            The segment runner.
        external_arg : Any
            The field's parameters.

        Returns
        -------
        tuple
            ``(potential, None)`` (:meth:`MeshSegmentRunner.potential_energy
            <jaccpot.distributed.segment.MeshSegmentRunner.potential_energy>`) or
            ``(None, reason)`` when the force has no mesh potential (warned once).
        """
        try:
            return (
                msr.potential_energy(
                    external_potential_fn=self.runner.external_potential_fn,
                    external_params=external_arg,
                ),
                None,
            )
        except NotImplementedError as exc:
            reason = str(exc)
            if not self._energy_warned:
                self._energy_warned = True
                warnings.warn(
                    f"the multi-GPU mode's energy diagnostic is unavailable: {reason}. "
                    "Its fields are None (energy_unavailable says why).",
                    stacklevel=4,
                )
            return None, reason

    def _plan_record(self, msr: Any, report: Any) -> dict:
        """What the segment that ended at this edge ran with (host values, JSON-able).

        Parameters
        ----------
        msr : Any
            The segment runner.
        report : Any
            Its :class:`~jaccpot.distributed.segment.SegmentReport`, or ``None``
            at step 0.

        Returns
        -------
        dict
            ``leaf_capacity``, ``walk_caps``, ``level_plan`` (the merged
            ``FusedCapacityPlan`` fields), ``level_floor``, ``upward_num_levels``,
            ``cross_caps``, ``cross_needs_measured`` (the fused force's; ``None``
            for a force without them), ``mesh`` (``ndev``, ``cap``, ``counts``),
            ``repartition`` (``repartitioned``, ``declined``, ``sent_off_device``
            of the repartition before the segment; ``None`` at step 0) and
            ``bits``.
        """
        from jaccpot.production import _jsonable

        force = msr.force
        setup = getattr(force, "setup", None)
        engine = getattr(force, "engine", None)
        caps = getattr(engine, "_strict_fused_validated_caps", None)
        cross = getattr(force, "cross_caps", None)
        floor = getattr(force, "level_floor", None)
        measured = getattr(force, "cross_needs_measured", None)
        counts = (
            [int(c) for c in _host(msr.state.count)]
            if report is None
            else list(report.counts)
        )
        return {
            "leaf_capacity": None if setup is None else int(setup.leaf_capacity),
            "walk_caps": None if caps is None else _jsonable(dict(caps)),
            "level_plan": (
                None if setup is None else [int(v) for v in setup.plan.fingerprint()]
            ),
            "level_floor": None if floor is None else [int(v) for v in floor],
            "upward_num_levels": getattr(engine, "_cells_upward_num_levels", None),
            "cross_caps": None if cross is None else _jsonable(dict(vars(cross))),
            "cross_needs_measured": (
                None
                if measured is None
                else [int(v) for v in np.asarray(measured).reshape(-1)]
            ),
            "mesh": {"ndev": int(msr.ndev), "cap": int(msr.cap), "counts": counts},
            "repartition": (
                None
                if report is None
                else {
                    "repartitioned": bool(report.repartitioned),
                    "declined": bool(report.declined),
                    "sent_off_device": int(report.sent_off_device),
                }
            ),
            "bits": None if report is None else int(report.bits),
        }

    # ------------------------------------------------------------------
    # Restart files.
    # ------------------------------------------------------------------

    def save_restart(self, path: Any) -> str:
        """Not yet on the mesh.

        Parameters
        ----------
        path : Any
            Output file.

        Returns
        -------
        str
            Never.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(f"multi-GPU restart files ({path}): not yet")
