"""The production runner's multi-GPU mode: :class:`~jaccpot.production.ProductionRunner` on the mesh.

``ProductionRunner(solver, ..., mesh=...)`` (a 1-D ``jax.sharding.Mesh`` or a
:class:`~jaccpot.production.MeshRunConfig`) dispatches :meth:`~jaccpot.production.ProductionRunner.run`
and :meth:`~jaccpot.production.ProductionRunner.save_restart` here. The single
card's loop -- segments, edges, callbacks (a truthy return stops the run),
diagnostics, restart files and their continuation -- runs on the fused multi-GPU
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

**Restart files** hold the state, masses, total acceleration and (under the
criterion) f_b BY ID, the layout (ids per row, the padding rows' positions) and the
force's plan. On the mesh that wrote it, a restore rebuilds the identical layout
and plan -- leaf capacity, walk caps, the level plan and floor, the depth bound,
the cross capacities, the template's shapes -- and checks the digest of the force's
compile key: the continuation is bitwise the uninterrupted run. On another mesh the
by-id state is decomposed again and the force prepared afresh there:
``restored["bitwise"]`` is ``False``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import warnings
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

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
            ``None``, a restart file (:meth:`save_restart`) or a mesh
            :class:`~jaccpot.production.RunResult`.

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
            single-card result or restart file, or a decomposition that does not
            fit ``cap``.
        TypeError
            For a ``restart_from`` that is none of the above.
        RuntimeError
            When a segment fails after the segment runner's own re-plan; the
            message names the last restart file.
        """
        from jaccpot._env import env_text
        from jaccpot.production import (
            MeshSegmentEdge,
            RunResult,
            _baselines,
            _diagnostics_row,
            _moments,
            _warn_fusion_autotuner,
            load_restart,
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
        elif isinstance(restart_from, (str, os.PathLike)):
            arrays, meta = load_restart(restart_from)
            counters = meta["counters"]
            step, segment_done = int(counters["step"]), int(counters["segment"])
            t0_run = float(counters["t0"])
            anchor = (
                tuple(counters["anchor"])
                if dt_value == float(counters["dt"])
                else (float(counters["time"]), step)
            )
            baselines = meta["baselines"]
            self._check_force_params(force_params, self.config.force)
            external_arg = self._external_arg(external_params, arrays["state"].dtype)
            msr, restored = self._restore(
                str(restart_from),
                arrays,
                meta,
                dt_value=dt_value,
                external_arg=external_arg,
            )
            del arrays
        else:
            raise TypeError(
                "restart_from must be a RunResult or a restart file, got "
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

    def _sharded(self, arr: Optional[np.ndarray]) -> Any:
        """A flat host array placed ``P(axis)`` on the mesh.

        Parameters
        ----------
        arr : Optional[np.ndarray]
            ``(ndev * cap, ...)``, or ``None``.

        Returns
        -------
        Any
            The sharded array, or ``None``.
        """
        if arr is None:
            return None
        return jax.device_put(arr, NamedSharding(self.mesh, P(self.axis_name)))

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
        """:meth:`ProductionRunner.save_restart <jaccpot.production.ProductionRunner.save_restart>` on the mesh.

        One ``.npz``, written to a temporary file and renamed: ``state``,
        ``masses``, ``acceleration`` (the TOTAL) and, under
        ``mac_type='dehnen_error'``, ``force_scale``, all BY ID; the layout as
        ``mesh_ids`` (the id of every row, ``-1`` on padding) and ``mesh_padding``
        (the padding rows' positions -- they stay where the last repartition put
        them; everything else on a padding row is zero); and ``meta``: the
        single card's counters, baselines, field record and fingerprint, plus
        ``mesh`` (ndev, cap, counts, the force's class) and ``plan`` (the edge's
        plan record, :meth:`_plan_record`), ``template_shapes`` and the digest of
        the force's compile key. One gather of the layout's arrays.

        Parameters
        ----------
        path : Any
            Output file.

        Returns
        -------
        str
            ``path``.

        Raises
        ------
        RuntimeError
            Before the first segment (step 0 is the run's own input).
        """
        from jaccpot.production import (
            RESTART_FORMAT,
            _atomic_savez,
            _key_digest,
            _qualname,
            _template_shapes,
        )
        from jaccpot.runtime.strict_carry import external_field_key

        runner = self.runner
        current = runner._current
        msr = None if current is None else current["handle"]
        if msr is None or msr.state.acceleration is None:
            raise RuntimeError(
                "save_restart needs a run that has done a segment: the state at "
                "step 0 is the run's input"
            )
        layout = msr.state
        flat = {
            name: _host(getattr(layout, name))
            for name in ("state", "masses", "acceleration", "force_scale")
            if getattr(layout, name) is not None
        }
        ids_flat = _host(layout.ids).astype(np.int32)
        count = _host(layout.count)
        by_id = _by_id(flat, ids_flat, count)
        arrays = {name: by_id[name] for name in flat}
        arrays["mesh_ids"] = ids_flat
        arrays["mesh_padding"] = flat["state"][ids_flat < 0, 0]
        force = msr.force
        field = runner.external_acceleration_fn
        external_key = external_field_key(field, current["external_params"])
        setup = getattr(force, "setup", None)
        template = None if setup is None else setup.provider.template
        meta = {
            "format": RESTART_FORMAT,
            "counters": {
                "step": int(current["step"]),
                "time": float(current["time"]),
                "dt": float(current["dt"]),
                "t0": float(current["t0"]),
                "segment": int(current["segment"]),
                "num_steps": int(current["num_steps"]),
                "anchor": [float(current["anchor"][0]), int(current["anchor"][1])],
            },
            "baselines": current["baselines"],
            "mesh": {
                "ndev": int(msr.ndev),
                "cap": int(msr.cap),
                "axis_name": str(msr.axis_name),
                "counts": [int(c) for c in count],
                "num_samples": int(msr.num_samples),
                "repartition_every": int(msr.repartition_every),
                "force": _qualname(type(force)),
            },
            "plan": current["plan"],
            "template_shapes": None if template is None else _template_shapes(template),
            "digests": {"force_key": _key_digest(force.key())},
            "external": {
                "field": _qualname(field),
                "params_sha256": None if external_key is None else external_key[1],
            },
            "fingerprint": runner._fingerprint(by_id["state"], by_id["masses"]),
        }
        # UTF-8 bytes, as the single card's
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)
        _atomic_savez(str(path), **arrays)
        runner._last_restart = str(path)
        return str(path)

    def _restore(
        self,
        path: str,
        arrays: dict,
        meta: dict,
        *,
        dt_value: float,
        external_arg: Any,
    ) -> tuple:
        """The segment runner a restart file continues from, rebuilt in this process.

        On the mesh that wrote it (the same device count and ``cap``): the
        identical layout from the by-id arrays, ``mesh_ids`` and
        ``mesh_padding``, and the force's plan restored around its prepare at the
        restart's positions (:meth:`_restored_force`, :meth:`_install_plan`);
        then the digest of the force's compile key is checked -- equal, the next
        segments trace the run's programs. Elsewhere the by-id
        state is decomposed on this mesh, its carried acceleration and f_b placed
        by id, and the force prepared afresh: correct, not bitwise
        (``mesh_layout`` in the mismatches).

        Parameters
        ----------
        path : str
            The file (for messages).
        arrays : dict
            Its arrays.
        meta : dict
            Its ``meta``.
        dt_value : float
            This run's timestep.
        external_arg : Any
            The field parameters the segments pass.

        Returns
        -------
        tuple
            ``(MeshSegmentRunner, restored)``; ``restored`` holds ``from``,
            ``step``, ``bitwise``, ``mismatches``, ``digests`` and ``layout``
            (``"file"`` or ``"decomposed"``).

        Raises
        ------
        ValueError
            For a single-card file.
        """
        from jaccpot.production import _key_digest, _qualname
        from jaccpot.runtime.strict_carry import external_field_key

        saved = meta.get("mesh")
        if saved is None:
            raise ValueError(
                f"{path} was written on one card (no layout in it): continue it "
                "with ProductionRunner(...) without mesh="
            )
        runner = self.runner
        written = meta["fingerprint"]
        now = runner._fingerprint(arrays["state"], arrays["masses"])
        mismatches = [
            key
            for key in sorted(set(written) | set(now))
            if written.get(key) != now.get(key)
        ]
        if dt_value != float(meta["counters"]["dt"]):
            mismatches.append("dt")
        field = runner.external_acceleration_fn
        external = meta["external"]
        key = external_field_key(field, external_arg)
        field_matches = external["field"] == _qualname(field) and external[
            "params_sha256"
        ] == (None if key is None else key[1])
        if not field_matches:
            # the total of another field (or of none): the next segment computes
            # this one's in a start force
            mismatches.append("external_field")
        acceleration = arrays["acceleration"] if field_matches else None
        force_scale = arrays.get("force_scale")
        # the layout is the mesh's and the cap; whether the programs are the run's
        # is the force key's digest (below), whatever class built the force
        same_layout = int(saved["ndev"]) == self.ndev and (
            self.config.cap is None or int(self.config.cap) == int(saved["cap"])
        )
        if same_layout:
            layout = self._layout_from_file(arrays, int(saved["cap"]), acceleration)
            msr = self._segment_runner(layout, self._restored_force(meta["plan"]))
            self._install_plan(msr.force, meta)
        else:
            mismatches.append("mesh_layout")
            layout = self._decomposed(arrays["state"], arrays["masses"])
            ids = _host(layout.ids)
            live = ids >= 0
            for name, by_id in (
                ("acceleration", acceleration),
                ("force_scale", force_scale),
            ):
                if by_id is None:
                    continue
                flat = np.zeros((ids.shape[0],) + by_id.shape[1:], by_id.dtype)
                flat[live] = by_id[ids[live]]
                setattr(layout, name, self._sharded(flat))
            msr = self._segment_runner(layout, self._force())
        digests = {"force_key": _key_digest(msr.force.key())}
        if same_layout and digests["force_key"] != meta["digests"].get("force_key"):
            mismatches.append("force_key")
        msr.resume(
            segment_index=int(meta["counters"]["segment"]),
            steps_done=int(meta["counters"]["step"]),
            external_params=external_arg,
        )
        if mismatches:
            warnings.warn(
                f"restart {path}: {mismatches} differ from the run that wrote it; "
                "the continuation is correct but not bitwise the uninterrupted run",
                stacklevel=4,
            )
        restored = {
            "from": path,
            "step": int(meta["counters"]["step"]),
            "bitwise": not mismatches,
            "mismatches": mismatches,
            "digests": digests,
            "layout": "file" if same_layout else "decomposed",
        }
        return msr, restored

    def _layout_from_file(
        self, arrays: dict, cap: int, acceleration: Optional[np.ndarray]
    ) -> Any:
        """The written run's :class:`~jaccpot.distributed.segment.MeshState`, bit for bit.

        Live rows from the by-id arrays through ``mesh_ids``; padding rows at
        ``mesh_padding`` with zero velocity, mass, acceleration and f_b -- what the
        segment program and the repartition leave on them.

        Parameters
        ----------
        arrays : dict
            The file's arrays.
        cap : int
            Its rows per device.
        acceleration : Optional[np.ndarray]
            ``[N, 3]`` the carried total, or ``None`` (a start force recomputes it).

        Returns
        -------
        Any
            The mesh state, placed on this mesh.

        Raises
        ------
        ValueError
            If the layout's rows are not ``ndev * cap`` or its live rows do not
            lead every device's rows.
        """
        from jaccpot.distributed.segment import MeshState

        rows = self.ndev * int(cap)
        ids = np.asarray(arrays["mesh_ids"], np.int32)
        if ids.shape != (rows,):
            raise ValueError(f"the file's layout has {ids.shape} ids, not ({rows},)")
        live = ids >= 0
        count = live.reshape(self.ndev, cap).sum(axis=1).astype(np.int32)
        leading = np.arange(cap)[None, :] < count[:, None]
        if not np.array_equal(live.reshape(self.ndev, cap), leading):
            raise ValueError("the file's layout: live rows do not lead every device")
        state = arrays["state"]
        flat_state = np.zeros((rows,) + state.shape[1:], state.dtype)
        flat_state[live] = state[ids[live]]
        flat_state[~live, 0] = arrays["mesh_padding"]

        def flat(by_id: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if by_id is None:
                return None
            out = np.zeros((rows,) + by_id.shape[1:], by_id.dtype)
            out[live] = by_id[ids[live]]
            return out

        return MeshState(
            state=self._sharded(flat_state),
            masses=self._sharded(flat(arrays["masses"])),
            ids=self._sharded(ids),
            count=self._sharded(count),
            acceleration=self._sharded(flat(acceleration)),
            force_scale=self._sharded(flat(arrays.get("force_scale"))),
        )

    def _restored_force(self, plan: dict) -> Any:
        """The force of a same-mesh restore, before its prepare.

        The run's leaf capacity, walk caps and depth bound installed on the
        solver's engine (the prepare at the restart's positions starts from
        them), and the library's fused force built with the run's cross
        capacities (explicit, so nothing is measured again) and level floor. A
        force object of the configuration is used as given: the engine's half of
        the plan is installed, its own capacities are what it holds (the digest
        check says whether they are the run's).

        Parameters
        ----------
        plan : dict
            The file's plan record.

        Returns
        -------
        Any
            The force.
        """
        from jaccpot.distributed.cross import CrossCapacities
        from jaccpot.runtime.capacity_plan import install_walk_caps

        engine = self.runner.engine
        if plan.get("leaf_capacity") is not None:
            engine._tree_leaf_capacity = int(plan["leaf_capacity"])
        install_walk_caps(engine, plan.get("walk_caps"))
        if plan.get("upward_num_levels") is not None:
            engine._cells_upward_num_levels = int(plan["upward_num_levels"])
        if self.config.force is not None:
            return self.config.force
        cross = plan.get("cross_caps")
        floor = plan.get("level_floor")
        return self._force(
            cross_caps=None if cross is None else CrossCapacities(**cross),
            level_floor=None if floor is None else tuple(int(v) for v in floor),
        )

    def _install_plan(self, force: Any, meta: dict) -> None:
        """Put back the plan the run's programs were traced with, after the prepare.

        The single card's restore, on the mesh force: the prepare at the
        restart's positions re-measured the walk caps, the merged level plan and
        the template's shapes there; the run's own replace them -- the walk caps
        and depth bound on the engine, the level plan in the setup, the template's
        leaf shapes (``production._template_with_shapes``: the structure stays the
        prepare's and must match). A force without a setup (the direct force)
        has nothing to restore.

        Parameters
        ----------
        force : Any
            The prepared force.
        meta : dict
            The file's ``meta``.
        """
        from jaccpot.production import _template_with_shapes

        setup = getattr(force, "setup", None)
        if setup is None:
            return
        from jaccpot.runtime.capacity_plan import FusedCapacityPlan, install_walk_caps
        from jaccpot.runtime.fused_provider import FusedForceProvider

        plan = meta["plan"]
        engine = force.engine
        install_walk_caps(engine, plan.get("walk_caps"))
        if plan.get("upward_num_levels") is not None:
            engine._cells_upward_num_levels = int(plan["upward_num_levels"])
        provider = setup.provider
        template = _template_with_shapes(
            provider.template_or_raise(), meta.get("template_shapes")
        )
        if template is not None:
            provider = FusedForceProvider(
                engine,
                leaf_size=provider.leaf_size,
                max_order=provider.max_order,
                theta=provider.theta,
                runtime_overrides=provider.runtime_overrides,
                template=template,
            )
        level_plan = setup.plan
        if plan.get("level_plan") is not None:
            level_plan = FusedCapacityPlan(*(int(v) for v in plan["level_plan"]))
        force.setup = dataclasses.replace(
            setup,
            provider=provider,
            plan=level_plan,
            walk_caps=(
                setup.walk_caps
                if plan.get("walk_caps") is None
                else dict(plan["walk_caps"])
            ),
        )
