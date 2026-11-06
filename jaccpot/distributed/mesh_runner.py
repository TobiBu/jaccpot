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

**Several processes** (``jax.distributed``, one process per device; M2,
:mod:`jaccpot.distributed.multiprocess`): the same loop and programs on a mesh
over every process's devices (``multiprocess.global_mesh``). Each process may
pass only its id range (``ProcessLocalParticles``); host reads of sharded arrays
are gathered over the processes, float reductions folded in mesh order, so the
run is bitwise the single process's on the same devices. Callbacks run on every
process (an edge's gathers are collectives: every process must make them);
``SnapshotWriter`` and ``DiagnosticsLog`` write their files on process 0 only
(``all_processes=True`` to write on each). Restart files are refused (one ``.npz`` through one host; sharded
checkpoints are track M3); ``restart_from=<RunResult>`` continues in memory.

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

#: why a multi-process run refuses restart files until M3 (user decision U5)
_MULTIPROCESS_RESTART = (
    "the restart file is one .npz gathered through one host. Sharded per-rank "
    "checkpoints (orbax, plan track M3) replace it; until then continue a "
    "multi-process run in memory (restart_from=<its RunResult>)"
)


def _host(arr: Any) -> np.ndarray:
    """A device array on the host (all shards).

    Parameters
    ----------
    arr : Any
        A (sharded) device array; under several processes (``jax.distributed``)
        every process gets the global value
        (:func:`~jaccpot.distributed.multiprocess.host_value`).

    Returns
    -------
    np.ndarray
        The host copy.
    """
    from jaccpot.distributed.multiprocess import host_value

    return host_value(arr)


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


def _by_id(
    flat: dict, ids_flat: np.ndarray, count: np.ndarray, tail: Optional[tuple] = None
) -> dict:
    """Host layout arrays in id order, with the identity checked.

    Parameters
    ----------
    flat : dict
        ``name -> (ndev * cap, ...)`` host arrays.
    ids_flat : np.ndarray
        ``(ndev * cap,)`` global ids (``-1`` on padding).
    count : np.ndarray
        ``(ndev,)`` live rows per device.
    tail : Optional[tuple]
        The escaper tail's ``(flat, ids_flat, count)`` (MESC; its rows by device
        too), or ``None``. A name ``flat`` has and the tail has not is zero on
        its rows.

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
    owner_rows = rows // cap
    if tail is not None:
        t_flat, t_ids_flat, t_count = tail
        t_cap = int(t_ids_flat.shape[0]) // ndev
        t_rows = _live_rows(t_count, t_cap)
        flat = {
            name: (
                arr[rows],
                (
                    t_flat[name][t_rows]
                    if name in t_flat
                    else np.zeros((t_rows.shape[0],) + arr.shape[1:], arr.dtype)
                ),
            )
            for name, arr in flat.items()
        }
        ids = np.concatenate([ids, t_ids_flat[t_rows].astype(np.int64)])
        owner_rows = np.concatenate([owner_rows, t_rows // t_cap])
        rows = None
    n = int(ids.shape[0])
    seen = np.zeros(n, bool)
    if n:
        if ids.min() < 0 or ids.max() >= n:
            raise ValueError("particle identity broken: an id outside 0..N-1")
        seen[ids] = True
    if not seen.all():
        raise ValueError("particle identity broken: ids are not exactly 0..N-1 once")
    out = {}
    for name, arr in flat.items():
        if rows is None:
            bound, escaped = arr
            full = np.empty((n,) + bound.shape[1:], bound.dtype)
            full[ids] = np.concatenate([bound, escaped])
        else:
            full = np.empty((n,) + arr.shape[1:], arr.dtype)
            full[ids] = arr[rows]
        out[name] = full
    owner = np.empty(n, np.int32)
    owner[ids] = owner_rows
    out["owner"] = owner
    return out


def gather_by_id(
    layout: Any,
    names: Sequence = ("state", "masses", "acceleration"),
    tail: Any = None,
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
    tail : Any
        The escaper tail (:class:`~jaccpot.distributed.escapers.EscaperTail`,
        MESC): its escapers are gathered too (``force_scale`` zero on them);
        ``None`` or an empty tail: the layout alone.

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
    tail_part = None
    if tail is not None and int(tail.total) > 0:
        tail_part = (
            {
                name: _host(getattr(tail, name))
                for name in ("state", "masses", "acceleration")
                if name in flat and getattr(tail, name) is not None
            },
            _host(tail.ids),
            _host(tail.count),
        )
    return _by_id(flat, _host(layout.ids), _host(layout.count), tail_part)


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
            ``[N, 2, 3]`` in input (id) order; ``None`` with ``restart_from``. Under
            several processes also a
            :class:`~jaccpot.distributed.multiprocess.ProcessLocalParticles` (this
            process's ids, ``masses=None``).
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
        from jaccpot.distributed.multiprocess import (
            ProcessLocalParticles,
            is_multiprocess,
            mesh_moments,
            refuse_in_multiprocess,
        )
        from jaccpot.production import (
            MeshSegmentEdge,
            RunResult,
            _baselines,
            _diagnostics_row,
            _escaper_fields,
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
        if int(runner.segment.restart_every) > 0 or isinstance(
            restart_from, (str, os.PathLike)
        ):
            refuse_in_multiprocess(
                "a restart file (SegmentConfig.restart_every, restart_from=<file>)",
                _MULTIPROCESS_RESTART,
            )
        started = time.perf_counter()
        dt_value = float(np.asarray(dt))
        restored: Optional[dict] = None
        # the escaper path (escapers=, MESC): the split's record -- the bound
        # rows' count, the ledger, the transfers and the tail's capacity; the
        # escapers themselves are the segment runner's tail. ``None``: off.
        policy = runner.escapers
        split: Optional[dict] = None
        if restart_from is None:
            if state is None or (
                masses is None and not isinstance(state, ProcessLocalParticles)
            ):
                raise ValueError("a fresh run needs state and masses")
            self._check_force_params(force_params, self.config.force)
            if isinstance(state, ProcessLocalParticles):
                # multi-process input: this process's id range only
                host_state = np.asarray(state.state)
                host_masses = np.asarray(state.masses)
                external_arg = self._external_arg(external_params, host_state.dtype)
                num_all = int(state.num_particles)
                local_n: Optional[int] = num_all
            else:
                host_state = _host(state)
                host_masses = _host(masses)
                external_arg = self._external_arg(external_params, host_state.dtype)
                num_all = int(host_state.shape[0])
                local_n = None
            if policy is None:
                msr = self._fresh(host_state, host_masses, num_particles=local_n)
            else:
                # the escaper path: the bound rows decomposed, the escapers in
                # their tail (MESC)
                msr, split = self._fresh_split(
                    host_state, host_masses, num_all, local_n, policy
                )
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
            split = self._split_in(res.escapers, policy, msr)
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
            split = self._split_in(meta.get("escapers"), policy, msr)
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
        # the escaper path: particles over the mesh, and this edge's re-split
        # (its transfers, jump and the new split's potential, which the edge's
        # row reuses)
        n_all = 0
        if split is not None:
            n_all = int(np.sum(_host(msr.state.count))) + int(msr.tail.total)
        edge_split: dict = {}

        def moments_of(arrays: Any) -> dict:
            return (
                mesh_moments(arrays.state, arrays.masses, self.mesh, self.axis_name)
                if is_multiprocess()
                else _moments(arrays.state, arrays.masses)
            )

        def diagnostics_at(step_now: int, due: bool) -> Optional[dict]:
            nonlocal baselines
            if every == 0 or not due:
                return None
            layout = msr.state
            moments = moments_of(layout)
            if split is not None and msr.tail_on:
                # the escapers' moments (their padding rows: zero mass and speed)
                escaped = moments_of(msr.tail)
                moments = {k: moments[k] + escaped[k] for k in moments}
            if baselines is None:
                baselines = dict(_baselines(moments, time_at(step_now)), step=step_now)
            energy_due = every_energy > 0 and segment_done % every_energy == 0
            potential, reason = None, None
            if energy_on and energy_due:
                potential = edge_split.get("potential")
                if potential is None:
                    potential, reason = self._potential(msr, external_arg)
                if (
                    potential is not None
                    and potential["ok"]
                    and baselines.get("energy") is None
                    and step_now == baselines.get("step")
                ):
                    # E0 at step 0 itself: the mesh force is prepared already
                    w0 = (
                        potential["self"]
                        + potential["external"]
                        + potential.get("couple", 0.0)
                    )
                    baselines["potential"] = w0
                    baselines["energy"] = float(baselines["kinetic"]) + w0
            row = _diagnostics_row(
                moments, baselines, time_at(step_now), names, energy_due, potential
            )
            if reason is not None:
                row["energy_unavailable"] = reason
            if split is not None:
                row.update(
                    _escaper_fields(
                        split, edge_split, row, baselines, n_all, potential, energy_on
                    )
                )
            return row

        def edge_at(retries: int, wall_s: float, due: bool, report: Any) -> Any:
            layout = msr.state
            edge = MeshSegmentEdge(
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
                escapers=None if split is None else msr.tail,
            )
            edge_split.clear()
            return edge

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
                escapers=None if split is None else dict(split),
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
                    edge,
                    state=None,
                    masses=None,
                    self_acceleration=None,
                    layout=None,
                    escapers=None,
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
            if split is not None and segment_done % int(policy.every) == 0:
                # the membership at this edge; a change moves the rows and
                # records the ledger's jump
                edge_split.update(
                    self._resplit(
                        msr, split, policy, n_all, energy=energy_on,
                        external_arg=external_arg,
                    )
                )
            due = every > 0 and (segment_done % every == 0 or last)
            if at_edge(edge_at(int(report.retries), wall_s, due, report)):
                stopped = True
                break
        gathered = gather_by_id(
            msr.state, ("state", "masses"), tail=None if split is None else msr.tail
        )
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
            escapers=None if split is None else dict(split),
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

    def _segment_runner(
        self, layout: Any, force: Any, migration: Optional[dict] = None
    ) -> Any:
        """A :class:`~jaccpot.distributed.segment.MeshSegmentRunner` at ``layout``.

        Parameters
        ----------
        layout : Any
            The :class:`~jaccpot.distributed.segment.MeshState`.
        force : Any
            The force; prepared here (and the cross capacities sized, under
            ``cross_caps="auto"``).
        migration : Optional[dict]
            ``migrate=True``: the ``splitters`` (the cuts of ``layout``'s ownership:
            a same-mesh restore's file, or :meth:`_decomposed`'s) and, on a
            restore, ``migration_cap``; ``None``: the configuration's buffer, the
            cuts chosen by the runner's repartition.

        Returns
        -------
        Any
            The segment runner, the state donated to its segments.
        """
        from jaccpot.distributed.segment import MeshSegmentRunner

        migration = migration or {}
        cap = migration.get("migration_cap")
        return MeshSegmentRunner(
            self.mesh,
            layout,
            force,
            external_acceleration_fn=self.runner.external_acceleration_fn,
            repartition_every=int(self.config.repartition_every),
            num_samples=int(self.config.num_samples),
            donate=True,
            axis_name=self.axis_name,
            migrate=bool(self.config.migrate),
            migration_cap=self.config.migration_cap if cap is None else int(cap),
            splitters=migration.get("splitters"),
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

    def _decomposed(
        self,
        state: np.ndarray,
        masses: np.ndarray,
        force: Any,
        num_particles: Optional[int] = None,
    ) -> tuple[Any, Optional[dict]]:
        """The host by-id particles decomposed over the mesh.

        Under ``migrate=True`` the Morton ranges are cut in the FORCE's box (the
        box the migration's cuts live in, ``TreeConfig.cubic_bounds``) and the cuts
        are read off them (``migration.ownership_splitters``), so the segment
        runner starts on them instead of repartitioning into that box again: one
        cap-sized exchange and its buffers less in the setup.

        Parameters
        ----------
        state : np.ndarray
            ``[N, 2, 3]`` in id order.
        masses : np.ndarray
            ``[N]``.
        force : Any
            The run's force (its box shape).
        num_particles : Optional[int]
            ``None``: ``state`` is the whole set. Else N over all processes, and
            ``state`` this process's ids
            (:class:`~jaccpot.distributed.multiprocess.ProcessLocalParticles`,
            :func:`~jaccpot.distributed.rollout.decompose_local`).

        Returns
        -------
        tuple[Any, Optional[dict]]
            The :class:`~jaccpot.distributed.segment.MeshState` (ids ``0..N-1``,
            no acceleration yet) and, under ``migrate=True``, ``{"splitters":
            <its cuts>}`` for :meth:`_segment_runner` (``None`` otherwise).
        """
        from jaccpot.distributed.migration import ownership_splitters
        from jaccpot.distributed.rollout import decompose, decompose_local
        from jaccpot.distributed.segment import MeshState, _force_box_cubic

        migrate = bool(self.config.migrate)
        cubic = _force_box_cubic(force) if migrate else None
        if num_particles is None:
            shard, _ = decompose(
                self.mesh,
                state[:, 0],
                state[:, 1],
                masses,
                cap=self._cap(int(state.shape[0])),
                num_samples=int(self.config.num_samples),
                axis_name=self.axis_name,
                cubic=cubic,
            )
        else:
            shard, _ = decompose_local(
                self.mesh,
                state[:, 0],
                state[:, 1],
                masses,
                num_particles=int(num_particles),
                cap=self._cap(int(num_particles)),
                num_samples=int(self.config.num_samples),
                axis_name=self.axis_name,
                cubic=cubic,
            )
        migration = None
        if migrate:
            migration = {
                "splitters": ownership_splitters(
                    self.mesh,
                    shard.positions,
                    shard.count,
                    cubic=bool(cubic),
                    axis_name=self.axis_name,
                )
            }
        layout = MeshState.from_shard_state(shard, self.mesh, axis_name=self.axis_name)
        return layout, migration

    def _fresh(
        self,
        state: np.ndarray,
        masses: np.ndarray,
        num_particles: Optional[int] = None,
    ) -> Any:
        """A fresh run's segment runner: decomposed, its force prepared.

        Parameters
        ----------
        state : np.ndarray
            ``[N, 2, 3]`` host state in input order (or this process's ids, with
            ``num_particles``).
        masses : np.ndarray
            ``[N]`` host masses.
        num_particles : Optional[int]
            As :meth:`_decomposed`.

        Returns
        -------
        Any
            The segment runner.
        """
        force = self._force()
        layout, migration = self._decomposed(state, masses, force, num_particles)
        return self._segment_runner(layout, force, migration)

    # ------------------------------------------------------------------
    # The escaper path (escapers=, MESC; jaccpot.distributed.escapers).
    # ------------------------------------------------------------------

    def _coupling_G(self, force: Any) -> float:
        """The coupling's ``G``: the force's own (it is static in it).

        Parameters
        ----------
        force : Any
            The run's force.

        Returns
        -------
        float
            ``G``.
        """
        return self._own_force_params(force)[0]

    def _fresh_split(
        self,
        state: np.ndarray,
        masses: np.ndarray,
        num_all: int,
        num_particles: Optional[int],
        policy: Any,
    ) -> tuple:
        """A fresh run's segment runner under the escaper policy, and its split.

        The membership of the initial state (:func:`~jaccpot.distributed.escapers.initial_membership`);
        nobody beyond ``r_escape``: the run's default decomposition with an empty
        tail (the run IS the run without escapers until someone leaves); else the
        bound rows decomposed and the escapers placed in their tail
        (:func:`~jaccpot.distributed.escapers.place_split`), so the cuts, the box
        and every capacity are the bound rows'.

        Parameters
        ----------
        state : np.ndarray
            ``[N, 2, 3]`` (or this process's ids, with ``num_particles``).
        masses : np.ndarray
            ``[N]``.
        num_all : int
            N over the run.
        num_particles : Optional[int]
            As :meth:`_decomposed`.
        policy : Any
            The :class:`~jaccpot.escapers.EscaperPolicy`.

        Returns
        -------
        tuple
            ``(MeshSegmentRunner, split)``; ``split`` holds ``n_bound``,
            ``ledger``, ``transfers``, ``tail_cap``, ``tail_grown`` (the tail's
            growths) and ``recuts`` (``migrate=True``: the repartitions after a
            changed split).
        """
        from jaccpot.distributed.escapers import (
            empty_tail,
            initial_membership,
            place_split,
        )
        from jaccpot.distributed.migration import ownership_splitters
        from jaccpot.distributed.multiprocess import local_id_range
        from jaccpot.distributed.segment import MeshState, _force_box_cubic

        lo = 0 if num_particles is None else local_id_range(num_all, self.mesh)[0]
        escaper, transfers = initial_membership(
            state[:, 0],
            masses,
            lo=lo,
            num_particles=num_all,
            mesh=self.mesh,
            policy=policy,
        )
        force = self._force()
        if transfers == 0:
            layout, migration = self._decomposed(state, masses, force, num_particles)
            tail = empty_tail(self.mesh, self.axis_name, state.dtype)
        else:
            migrate = bool(self.config.migrate)
            cubic = _force_box_cubic(force) if migrate else None
            shard, tail = place_split(
                self.mesh,
                state,
                masses,
                escaper,
                lo=lo,
                num_particles=num_all,
                cap=self._cap(num_all),
                num_samples=int(self.config.num_samples),
                axis_name=self.axis_name,
                cubic=cubic,
            )
            migration = None
            if migrate:
                migration = {
                    "splitters": ownership_splitters(
                        self.mesh,
                        shard.positions,
                        shard.count,
                        cubic=bool(cubic),
                        axis_name=self.axis_name,
                    )
                }
            layout = MeshState.from_shard_state(
                shard, self.mesh, axis_name=self.axis_name
            )
        msr = self._segment_runner(layout, force, migration)
        msr.attach_escapers(tail, self._coupling_G(force))
        split = {
            "n_bound": int(num_all) - int(tail.total),
            "ledger": 0.0,
            "transfers": int(transfers),
            "tail_cap": int(tail.cap),
            "tail_grown": 0,
            "recuts": 0,
        }
        return msr, split

    def _split_in(self, split_in: Optional[dict], policy: Any, msr: Any) -> Optional[dict]:
        """A continued run's split (a ``RunResult``'s or a restart file's).

        Parameters
        ----------
        split_in : Optional[dict]
            The record it carries (``None``: it ran without escapers).
        policy : Any
            This runner's policy (``None``: off).
        msr : Any
            The continued segment runner (its tail is the record's).

        Returns
        -------
        Optional[dict]
            The split, or ``None``.

        Raises
        ------
        ValueError
            If the run continues in another escaper mode.
        """
        if (split_in is None) != (policy is None):
            raise ValueError(
                "a run continues in its own escaper mode: "
                + (
                    "this one ran with escapers= (pass the runner the policy)"
                    if policy is None
                    else "this one ran without escapers (escapers=None)"
                )
            )
        if split_in is None:
            return None
        if msr.tail is None:
            raise ValueError("the continued run has no escaper tail")
        return {
            "n_bound": int(split_in["n_bound"]),
            "ledger": split_in["ledger"],
            "transfers": int(split_in["transfers"]),
            "tail_cap": int(msr.tail.cap),
            "tail_grown": int(split_in.get("tail_grown", 0)),
            "recuts": int(split_in.get("recuts", 0)),
        }

    def _resplit(
        self,
        msr: Any,
        split: dict,
        policy: Any,
        n_all: int,
        *,
        energy: bool,
        external_arg: Any,
    ) -> dict:
        """A segment edge's membership check; on a change, the rows moved and the jump.

        :func:`jaccpot.escapers.resplit` on the mesh: the per-device counts of
        rows leaving and returning (one small program); on a change the energy of
        the SAME state in its old and its new representation (the ledger's jump;
        ``split["ledger"]`` becomes ``None`` once a jump is unknown), the tail
        grown first if a device's escapers outgrow it (a new tail capacity: its
        programs compile again), and the device-local move
        (:func:`~jaccpot.distributed.escapers.resplit`). The carried
        accelerations are dropped: the next segment's start force evaluates the
        new split.

        Parameters
        ----------
        msr : Any
            The segment runner.
        split : dict
            The run's split, updated in place.
        policy : Any
            The radii.
        n_all : int
            Particles over the mesh.
        energy : bool
            Evaluate the jump (the run's energy diagnostic is on).
        external_arg : Any
            The external potential's parameters.

        Returns
        -------
        dict
            ``transfers`` and, after a change, ``jump`` and ``potential`` (the new
            split's, which the edge's diagnostics row reuses).

        Raises
        ------
        RuntimeError
            If no particle would stay in the tree, or a device's bound rows would
            outgrow its shard (``cap``).
        """
        from jaccpot.distributed.escapers import (
            grown_tail,
            membership,
            resplit,
            tail_capacity,
        )

        tail = msr.tail
        out, back = membership(self.mesh, self.axis_name, msr.state, tail, policy)
        changed = int(out.sum() + back.sum())
        if changed == 0:
            return {"transfers": 0}
        bound_now = _host(msr.state.count).astype(np.int64)
        tail_now = _host(tail.count).astype(np.int64)
        bound_next = bound_now - out + back
        tail_next = tail_now - back + out
        if int(bound_next.sum()) <= 0:
            raise RuntimeError(
                f"the escaper policy (r_escape {policy.r_escape}) leaves no "
                "particle in the tree"
            )
        if int(bound_next.max()) > int(msr.cap):
            raise RuntimeError(
                f"returning escapers would leave device {int(bound_next.argmax())} "
                f"with {int(bound_next.max())} bound rows > its shard's cap="
                f"{msr.cap}: give the shards more headroom (MeshRunConfig.cap)"
            )

        def total(potential: Optional[dict]) -> Optional[float]:
            if potential is None or not potential["ok"]:
                return None
            return (
                float(potential["self"])
                + float(potential["external"])
                + float(potential.get("couple", 0.0))
            )

        before = total(self._potential(msr, external_arg)[0]) if energy else None
        if int(tail_next.max()) > int(tail.cap):
            tail = grown_tail(
                self.mesh, self.axis_name, tail, tail_capacity(int(tail_next.max()))
            )
            split["tail_cap"] = int(tail.cap)
            split["tail_grown"] = int(split.get("tail_grown", 0)) + 1
        layout, tail = resplit(
            self.mesh,
            self.axis_name,
            msr.state,
            tail,
            policy,
            float(msr.coupling_G),
            donate=bool(msr.donate),
        )
        tail.total = int(tail_next.sum())
        msr.state = layout
        msr.tail = tail
        split["n_bound"] = int(n_all) - int(tail.total)
        split["transfers"] = int(split["transfers"]) + changed
        edge: dict = {"transfers": changed, "jump": None}
        if msr.migrate:
            # The bound rows' box is another one now (a far row left it, or a
            # returner widened it), and the migration's fixed cuts are Morton
            # codes of the OLD box: in the new one they cut anywhere (a 4-device
            # mesh overflowed a shard at the next step's migration). Re-cut in
            # the new box as the start does: a full repartition, its cuts kept.
            msr.splitters = None
            msr.state, diag = msr._repartition(msr.state)
            if bool(diag["declined"][0]):
                raise RuntimeError(
                    "the repartition after a changed escaper split was declined: "
                    "some device would receive "
                    f"{int(np.asarray(diag['recv_counts']).max())} rows > "
                    f"cap={msr.cap}"
                )
            split["recuts"] = int(split.get("recuts", 0)) + 1
        if energy:
            after, _ = self._potential(msr, external_arg)
            edge["potential"] = after
            after_w = total(after)
            if before is not None and after_w is not None:
                edge["jump"] = after_w - before
        if edge["jump"] is None or split["ledger"] is None:
            split["ledger"] = None
        else:
            split["ledger"] = float(split["ledger"]) + float(edge["jump"])
        return edge

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
            potential = msr.potential_energy(
                external_potential_fn=self.runner.external_potential_fn,
                external_params=external_arg,
            )
            if msr.tail_on:
                # the escaper split: the coupling W_c, the escapers' K and W_ext
                from jaccpot.distributed.escapers import tail_energy

                extra = tail_energy(
                    self.mesh,
                    self.axis_name,
                    msr.state,
                    msr.tail,
                    float(msr.coupling_G),
                    external_potential_fn=self.runner.external_potential_fn,
                    external_params=external_arg,
                )
                potential["external"] = potential["external"] + extra[
                    "external_escapers"
                ]
                potential["couple"] = extra["couple"]
                potential["kinetic_escapers"] = extra["kinetic_escapers"]
            return potential, None
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
            of the repartition before the segment; ``None`` at step 0),
            ``migration`` (``migrate=True``: the buffer's ``cap``, the segment's
            ``migrated`` rows and per-step ``need``; ``None`` otherwise) and
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
            "migration": (
                None
                if not getattr(msr, "migrate", False)
                else {
                    "cap": msr.migration_cap,
                    "migrated": None if report is None else int(report.migrated),
                    "need": None if report is None else int(report.migration_need),
                }
            ),
            "bits": None if report is None else int(report.bits),
            # the escaper path (MESC) only: the tail's capacity and counts
            **(
                {}
                if getattr(msr, "tail", None) is None
                else {
                    "escapers": {
                        "cap": int(msr.tail.cap),
                        "counts": [int(c) for c in _host(msr.tail.count)],
                    }
                }
            ),
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
        from jaccpot.distributed.multiprocess import refuse_in_multiprocess
        from jaccpot.production import (
            RESTART_FORMAT,
            _atomic_savez,
            _key_digest,
            _qualname,
            _template_shapes,
        )
        from jaccpot.runtime.strict_carry import external_field_key

        refuse_in_multiprocess("save_restart", _MULTIPROCESS_RESTART)
        runner = self.runner
        current = runner._current
        msr = None if current is None else current["handle"]
        split = None if current is None else current.get("escapers")
        # a changed escaper split drops the carried accelerations (the
        # continuation's start force evaluates them), so its edges hold none
        resplit_edge = split is not None and int(current["segment"]) > 0
        if msr is None or (msr.state.acceleration is None and not resplit_edge):
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
        tail = None if split is None else msr.tail
        tail_part = None
        if tail is not None:
            tail_part = (
                {
                    name: _host(getattr(tail, name))
                    for name in ("state", "masses", "acceleration")
                    if getattr(tail, name) is not None
                },
                _host(tail.ids).astype(np.int32),
                _host(tail.count),
            )
        by_id = _by_id(flat, ids_flat, count, tail_part)
        arrays = {name: by_id[name] for name in flat}
        arrays["mesh_ids"] = ids_flat
        arrays["mesh_padding"] = flat["state"][ids_flat < 0, 0]
        if tail_part is not None:
            # the escaper tail's layout: the id of every row (-1 on padding); its
            # padding rows are zero
            arrays["escaper_ids"] = tail_part[1]
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
                # migrate=True: the cuts of this layout's ownership and the buffer,
                # which a same-mesh restore keeps (the cuts are uint64 codes)
                "migration": (
                    None
                    if not getattr(msr, "migrate", False)
                    else {
                        "splitters": [int(v) for v in _host(msr.splitters)],
                        "migration_cap": msr.migration_cap,
                    }
                ),
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
        if split is not None and runner.escapers is not None:
            # only with the escaper path: the policy, the split and the ledger
            meta["escapers"] = {
                **runner.escapers.record(),
                "n_bound": int(split["n_bound"]),
                "ledger": split["ledger"],
                "transfers": int(split["transfers"]),
                "tail_cap": int(split["tail_cap"]),
                "tail_grown": int(split.get("tail_grown", 0)),
                "recuts": int(split.get("recuts", 0)),
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
        acceleration = arrays.get("acceleration") if field_matches else None
        force_scale = arrays.get("force_scale")
        # the escaper path (MESC): the tail's rows, by id
        tail_ids = arrays.get("escaper_ids")
        if (tail_ids is None) != (runner.escapers is None):
            raise ValueError(
                "a run continues in its own escaper mode: "
                + (
                    f"{path} ran with escapers= (pass the runner the policy)"
                    if runner.escapers is None
                    else f"{path} ran without escapers (escapers=None)"
                )
            )
        # the layout is the mesh's and the cap; whether the programs are the run's
        # is the force key's digest (below), whatever class built the force
        same_layout = int(saved["ndev"]) == self.ndev and (
            self.config.cap is None or int(self.config.cap) == int(saved["cap"])
        )
        tail = None
        if same_layout:
            layout = self._layout_from_file(arrays, int(saved["cap"]), acceleration)
            if tail_ids is not None:
                tail = self._tail_from_file(arrays, tail_ids, acceleration)
            migration = saved.get("migration")
            if bool(migration) != bool(self.config.migrate):
                # the cuts (or their absence) are part of the trajectory
                mismatches.append("migration")
            msr = self._segment_runner(
                layout,
                self._restored_force(meta["plan"]),
                migration if self.config.migrate else None,
            )
            self._install_plan(msr.force, meta)
        elif tail_ids is not None:
            # another mesh: the bound rows decomposed there, the escapers placed
            # in their tail (as a fresh run with this membership)
            mismatches.append("mesh_layout")
            force = self._force()
            layout, migration, tail = self._decomposed_split(
                arrays, tail_ids, force, int(meta["escapers"]["tail_cap"])
            )
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
            msr = self._segment_runner(layout, force, migration)
        else:
            mismatches.append("mesh_layout")
            force = self._force()
            layout, migration = self._decomposed(
                arrays["state"], arrays["masses"], force
            )
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
            msr = self._segment_runner(layout, force, migration)
        if tail is not None:
            msr.attach_escapers(tail, self._coupling_G(msr.force))
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

    def _tail_from_file(
        self,
        arrays: dict,
        tail_ids: np.ndarray,
        acceleration: Optional[np.ndarray],
    ) -> Any:
        """The written run's escaper tail, bit for bit (MESC).

        Parameters
        ----------
        arrays : dict
            The file's arrays (by id).
        tail_ids : np.ndarray
            ``escaper_ids``: the id of every tail row (``-1`` on padding).
        acceleration : Optional[np.ndarray]
            ``[N, 3]`` the carried total, or ``None``.

        Returns
        -------
        Any
            The :class:`~jaccpot.distributed.escapers.EscaperTail`, placed.

        Raises
        ------
        ValueError
            If the tail's rows do not split over the mesh or its live rows do not
            lead every device.
        """
        from jaccpot.distributed.escapers import EscaperTail

        ids = np.asarray(tail_ids, np.int32)
        if ids.shape[0] % self.ndev:
            raise ValueError(f"the file's escaper tail has {ids.shape} ids")
        cap_e = ids.shape[0] // self.ndev
        live = ids >= 0
        count = live.reshape(self.ndev, cap_e).sum(axis=1).astype(np.int32)
        leading = np.arange(cap_e)[None, :] < count[:, None]
        if not np.array_equal(live.reshape(self.ndev, cap_e), leading):
            raise ValueError(
                "the file's escaper tail: live rows do not lead every device"
            )

        def flat(by_id: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if by_id is None:
                return None
            out = np.zeros((ids.shape[0],) + by_id.shape[1:], by_id.dtype)
            out[live] = by_id[ids[live]]
            return out

        return EscaperTail(
            state=self._sharded(flat(arrays["state"])),
            masses=self._sharded(flat(arrays["masses"])),
            ids=self._sharded(ids),
            count=self._sharded(count),
            cap=int(cap_e),
            total=int(live.sum()),
            acceleration=self._sharded(flat(acceleration)),
        )

    def _decomposed_split(
        self, arrays: dict, tail_ids: np.ndarray, force: Any, tail_cap: int
    ) -> tuple:
        """A restart's by-id split on another mesh: the bound rows decomposed, the
        escapers in their tail (:func:`~jaccpot.distributed.escapers.place_split`).

        Parameters
        ----------
        arrays : dict
            The file's arrays (by id).
        tail_ids : np.ndarray
            The written tail's ids (who is an escaper).
        force : Any
            The run's force (its box shape).
        tail_cap : int
            The written tail's capacity (kept, if it holds this mesh's chunks).

        Returns
        -------
        tuple
            ``(MeshState, migration, EscaperTail)``.
        """
        from jaccpot.distributed.escapers import place_split, tail_capacity
        from jaccpot.distributed.migration import ownership_splitters
        from jaccpot.distributed.multiprocess import _device_chunks
        from jaccpot.distributed.segment import MeshState, _force_box_cubic

        state = arrays["state"]
        n = int(state.shape[0])
        escaper = np.zeros(n, bool)
        escaper[tail_ids[tail_ids >= 0]] = True
        most = max(
            int(escaper[a:b].sum()) for a, b in _device_chunks(n, self.ndev)
        )
        migrate = bool(self.config.migrate)
        cubic = _force_box_cubic(force) if migrate else None
        shard, tail = place_split(
            self.mesh,
            state,
            arrays["masses"],
            escaper,
            lo=0,
            num_particles=n,
            cap=self._cap(n),
            num_samples=int(self.config.num_samples),
            axis_name=self.axis_name,
            cubic=cubic,
            tail_cap=max(int(tail_cap), tail_capacity(most)),
        )
        migration = None
        if migrate:
            migration = {
                "splitters": ownership_splitters(
                    self.mesh,
                    shard.positions,
                    shard.count,
                    cubic=bool(cubic),
                    axis_name=self.axis_name,
                )
            }
        layout = MeshState.from_shard_state(shard, self.mesh, axis_name=self.axis_name)
        return layout, migration, tail

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
