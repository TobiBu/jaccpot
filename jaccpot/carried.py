"""The runner's CARRIED-scheme path: a scheme that owns its carry and its segment program.

Hybrid Hermite session 3 (multi-node plan U6, track HH3b; ``docs/production_2026-11.md``,
``## HH3b``). :class:`~jaccpot.scheme.SplitStepScheme` (I1) cuts ONE step of the
runner's own scan at its force evaluation, so the carry stays the runner's ``[N, 2,
3]`` state. The direct-sum clusters of ``nornax.hybrid_clusters`` need a second
carried block (the ``[K, M]`` clusters, the live-row count, the ids, the dwell
counters, the energy ledger, the slow force's verdicts) and a second force (the
galaxy's field at the stars through ``target_points``); their segment is nornax's
``cluster_rollout``. :class:`~jaccpot.scheme.CarriedScheme` is the seam one level up:
the scheme owns its carry and its programs, and :class:`ProductionRunner` keeps THE
LOOP -- the segment cadence, the retry / re-plan loop (its reasons now include the
target plans and the reserve next to the walk and tree caps), the edges (membership
checks and transfers, through the scheme), the diagnostics rows, the callbacks, the
restart files (``.npz``, and on one card also an orbax checkpoint directory), the
fingerprint and the bitwise check of a continuation.

``ProductionRunner(scheme=<a carried scheme>)`` dispatches :meth:`run` and
:meth:`save_restart` here (:func:`run_carried`, :func:`save_carried`); without one
nothing here runs, so every default program is unchanged. jaccpot does not import
nornax: the dependency stays ``nornax -> jaccpot``.

This module also holds the jaccpot pieces a carried scheme needs to restart its own
providers bitwise in a fresh process (:func:`trace_state`, :func:`pinned_trace_state`,
:func:`provider_record`, :func:`provider_with_record`): the trace-time state of the
engines its programs are traced against, and each provider's template shapes and
compile-key digest -- what ``ProductionRunner._restore`` restores for its own
provider (leaf capacity, validated walk caps, level registry, depth bound, template
shapes, digests).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
import warnings
from typing import Any, Iterator, Optional, Sequence

import numpy as np

__all__ = [
    "key_digest",
    "pinned_trace_state",
    "provider_record",
    "provider_with_record",
    "run_carried",
    "save_carried",
    "trace_state",
]


# --------------------------------------------------------------------------
# Trace-time state: what a provider's programs read from the engine when they trace.
# --------------------------------------------------------------------------


def key_digest(key: Any) -> str:
    """sha256 of a compile key's repr (no object addresses in it: process-free).

    Parameters
    ----------
    key : Any
        A provider's or a template's compile key.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha256(repr(key).encode()).hexdigest()


def trace_state(engines: Sequence[Any]) -> dict:
    """The trace-time state of ``engines`` and the level registry (JSON-able).

    What the restart record of :class:`~jaccpot.production.ProductionRunner` holds
    for its one engine (``_plan_record``): per engine the validated walk caps the
    traced walk is sized from, the leaf capacity and the depth bound; once the
    level registry (the level loops' batch widths and depth counts, process-wide).

    Parameters
    ----------
    engines : Sequence[Any]
        Runtime engines (or ``FastMultipoleMethod`` facades), in a fixed order.

    Returns
    -------
    dict
        ``engines`` (one dict per engine) and ``level_widths`` / ``level_counts``.
    """
    from jaccpot.production import _jsonable
    from jaccpot.runtime import _level_shapes

    out = []
    for engine in engines:
        impl = getattr(engine, "_impl", engine)
        caps = getattr(impl, "_strict_fused_validated_caps", None)
        out.append(
            {
                "walk_caps": None if caps is None else _jsonable(dict(caps)),
                "leaf_capacity": getattr(impl, "_tree_leaf_capacity", None),
                "upward_num_levels": getattr(impl, "_cells_upward_num_levels", None),
            }
        )
    return {
        "engines": out,
        "level_widths": [
            [int(a), int(b), int(w)] for (a, b), w in _level_shapes._WIDTHS.items()
        ],
        "level_counts": [
            [int(a), int(b), int(n)] for (a, b), n in _level_shapes._LEVELS.items()
        ],
    }


def _install(engines: Sequence[Any], record: dict) -> None:
    from jaccpot.runtime import _level_shapes
    from jaccpot.runtime.capacity_plan import install_walk_caps

    for engine, rec in zip(engines, record["engines"]):
        impl = getattr(engine, "_impl", engine)
        if rec["walk_caps"] is not None:
            install_walk_caps(impl, rec["walk_caps"])
        if rec["leaf_capacity"] is not None:
            impl._tree_leaf_capacity = int(rec["leaf_capacity"])
        if rec["upward_num_levels"] is not None:
            impl._cells_upward_num_levels = int(rec["upward_num_levels"])
    _level_shapes._WIDTHS.clear()
    _level_shapes._WIDTHS.update(
        {(int(a), int(b)): int(w) for a, b, w in record["level_widths"]}
    )
    _level_shapes._LEVELS.clear()
    _level_shapes._LEVELS.update(
        {(int(a), int(b)): int(n) for a, b, n in record["level_counts"]}
    )


@contextlib.contextmanager
def pinned_trace_state(
    engines: Sequence[Any], record: Optional[dict]
) -> Iterator[None]:
    """Run a block with ``record`` installed on ``engines``, and leave them as found.

    Every program of a carried scheme's force is called inside this block with the
    record taken when the force was made: a program traces against exactly that
    state whenever it first runs -- in the run that wrote a restart and in the
    process that continues it -- and its trace leaves nothing behind for the next
    one (as ``_trace_state_kept`` does for the runner's energy programs). Calls of a
    compiled program read nothing of it. ``record=None``: a no-op.

    Parameters
    ----------
    engines : Sequence[Any]
        The engines (the order of :func:`trace_state`).
    record : Optional[dict]
        :func:`trace_state`'s record.

    Yields
    ------
    None
        Inside the block.
    """
    if record is None or not engines:
        yield
        return
    from jaccpot.production import _trace_state_kept

    with contextlib.ExitStack() as stack:
        seen: set = set()
        for engine in engines:
            impl = getattr(engine, "_impl", engine)
            if id(impl) not in seen:
                seen.add(id(impl))
                stack.enter_context(_trace_state_kept(impl))
        _install(engines, record)
        yield


def provider_record(provider: Any) -> dict:
    """A provider's template shapes and compile-key digests (JSON-able).

    Call it inside :func:`pinned_trace_state` with the force's record (the key
    reads the engine's caps and depth bound).

    Parameters
    ----------
    provider : Any
        A ``FusedForceProvider``.

    Returns
    -------
    dict
        ``template_shapes`` (or ``None`` without a template), ``provider_key``
        and ``template_key`` digests.
    """
    from jaccpot.production import _template_shapes

    template = getattr(provider, "template", None)
    return {
        "template_shapes": None if template is None else _template_shapes(template),
        "template_key": None if template is None else key_digest(template.key()),
        "provider_key": key_digest(provider.key()),
    }


def provider_with_record(provider: Any, record: dict) -> tuple:
    """``provider`` with a recorded template's shapes (a restore), and whether it fits.

    The provider comes from a prepare at the restart's positions: its STRUCTURE
    (treedef, static leaves) is the run's, its array shapes are this prepare's. The
    run's shapes are taken from ``record`` (as ``ProductionRunner._restore`` does).

    Parameters
    ----------
    provider : Any
        The restore prepare's ``FusedForceProvider``.
    record : dict
        :func:`provider_record` of the run's provider.

    Returns
    -------
    tuple
        ``(provider, ok)``; ``ok`` False when the structure does not fit (the
        prepare's own provider is returned).
    """
    from jaccpot.production import _template_with_shapes
    from jaccpot.runtime.fused_provider import FusedForceProvider

    template = getattr(provider, "template", None)
    if template is None or record.get("template_shapes") is None:
        return provider, template is None and record.get("template_shapes") is None
    fitted = _template_with_shapes(template, record["template_shapes"])
    if fitted is None:
        return provider, False
    return (
        FusedForceProvider(
            provider.engine,
            leaf_size=provider.leaf_size,
            max_order=provider.max_order,
            theta=provider.theta,
            runtime_overrides=provider.runtime_overrides,
            template=fitted,
            grad_statics=provider.grad_statics,
        ),
        True,
    )


# --------------------------------------------------------------------------
# The loop.
# --------------------------------------------------------------------------


def _is_orbax_path(path: str) -> bool:
    """Whether a carried run writes an orbax checkpoint to ``path``.

    Parameters
    ----------
    path : str
        The restart path.

    Returns
    -------
    bool
        ``False`` for ``*.npz`` (one file), else ``True`` (a directory).
    """
    return not str(path).endswith(".npz")


def _potential_of(energy: dict, escapers_on: bool) -> dict:
    """The runner's ``potential`` dict (``_energy_fields``) from a scheme's energy terms.

    Parameters
    ----------
    energy : dict
        :meth:`CarriedScheme.energy`'s terms.
    escapers_on : bool
        Whether the run has the escaper path (``couple`` is then ``W_c``).

    Returns
    -------
    dict
        ``self`` (every internal potential term), ``external`` (0), ``ok``,
        ``replans``, ``bits`` and, with escapers, ``couple`` and
        ``kinetic_escapers``.
    """
    w = (
        float(energy["W_gg"])
        + float(energy["W_gc"])
        + float(energy["W_cc_inter"])
        + float(energy["W_cc"])
    )
    out = {
        "self": w,
        "external": 0.0,
        "ok": bool(energy.get("ok", True)),
        "replans": 0,
        "bits": None,
    }
    if escapers_on:
        out["couple"] = float(energy.get("W_c", 0.0))
        out["kinetic_escapers"] = float(energy.get("K_escapers", 0.0))
    return out


def _cluster_fields(
    energy: Optional[dict],
    info: dict,
    ledgers: dict,
    base: dict,
    row: dict,
    escapers_on: bool = False,
) -> dict:
    """The clusters' diagnostics columns (only on a carried run).

    Parameters
    ----------
    energy : Optional[dict]
        The scheme's energy terms at the edge (``None`` when not due).
    info : dict
        The edge's report (:meth:`CarriedScheme.edge` / ``describe``).
    ledgers : dict
        ``strip`` (the stripping ledger, exact) and ``escapers`` (``None`` without
        the escaper path or once a jump was unknown).
    base : dict
        Step 0's baselines (``energy``).
    row : dict
        The row so far (``energy``).
    escapers_on : bool
        Whether the run has the escaper path (its ledger then enters).

    Returns
    -------
    dict
        ``n_live``, ``n_stripped``, ``bound_mass`` (``[K]``), ``strip_transfers``,
        ``strip_ledger``, ``strip_jump`` (the ledger's exact jump at this edge),
        ``strip_jump_measured`` (the full measured one, with the scheme's
        ``measure_jumps``), ``dissolved``, ``reserve`` and, with the
        energy, ``W_gg``, ``W_gc``, ``W_cc``, ``W_cc_inter``, ``energy_ledgered``
        (``E - strip_ledger - escaper ledger``) and its drift against ``E0``.
    """
    fields: dict = {
        "n_live": int(info["n_live"]),
        "n_stripped": int(info["n_stripped"]),
        "bound_mass": [float(v) for v in info["bound_mass"]],
        "strip_transfers": int(info.get("n_take", 0)),
        "strip_jump": info.get("jump"),
        "strip_jump_measured": info.get("strip_jump_measured"),
        "strip_ledger": float(ledgers["strip"]),
        "dissolved": int(info.get("dissolved", 0)),
        "reserve": int(info["reserve"]),
    }
    if energy is not None and row.get("energy") is not None:
        for name in ("W_gg", "W_gc", "W_cc", "W_cc_inter"):
            fields[name] = float(energy[name])
        esc = ledgers.get("escapers") if escapers_on else 0.0
        if esc is not None:
            ledgered = float(row["energy"]) - float(ledgers["strip"]) - float(esc)
            e0 = base.get("energy")
            fields["energy_ledgered"] = ledgered
            fields["energy_ledgered_drift"] = (
                (ledgered - float(e0)) / abs(float(e0)) if e0 else None
            )
    return fields


def run_carried(
    runner: Any,
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
    """:meth:`ProductionRunner.run` under a carried scheme (the clusters).

    The parameters and the result are :meth:`ProductionRunner.run`'s. The
    differences: ``state``/``masses`` are the GALAXY's (the scheme holds its own
    second block, e.g. the clusters); the result's ``carried`` is the scheme's
    carry (``handle`` is ``None``); the edges' ``carried`` too, valid during the
    callback. A continuation (``restart_from`` a :class:`RunResult` or a file) is
    bitwise the uninterrupted run when ``restored["bitwise"]``.

    Parameters
    ----------
    runner : Any
        The :class:`~jaccpot.production.ProductionRunner`.
    state : Any
        ``[N, 2, 3]`` galaxy positions and velocities (``None`` with
        ``restart_from``).
    masses : Any
        ``[N]``.
    dt : Any
        Timestep.
    num_steps : int
        The run's total length from ``t0``.
    t0 : float
        Start time.
    force_params : Any
        Must be ``None`` (the solver's own ``G`` and softening).
    external_params : Any
        Must be ``None`` (no external field on this path).
    restart_from : Any
        ``None``, a :class:`RunResult` of this runner, or a restart file /
        checkpoint directory.

    Returns
    -------
    Any
        A :class:`~jaccpot.production.RunResult`.

    Raises
    ------
    ValueError
        For force or field parameters, a fresh run without state, or a restart
        of another kind of run.
    TypeError
        For a ``restart_from`` that is neither a result nor a path.
    RuntimeError
        When a segment still fails its verdict after the scheme's re-plans.
    """
    from jaccpot.checkpoint import is_checkpoint
    from jaccpot.production import (
        RunResult,
        SegmentEdge,
        _baselines,
        _diagnostics_row,
        _escaper_fields,
        load_restart,
        segment_lengths,
    )

    if force_params is not None or external_params is not None:
        raise ValueError(
            "a carried scheme (the clusters) takes the solver's own G and softening "
            "and no external field: pass force_params=None, external_params=None"
        )
    scheme = runner.scheme
    started = time.perf_counter()
    dt_value = float(np.asarray(dt))
    policy = runner.escapers
    restored: Optional[dict] = None
    if restart_from is None:
        if state is None or masses is None:
            raise ValueError("a fresh run needs state and masses")
        carry = scheme.start(
            runner.solver,
            runner.settings,
            state,
            masses,
            dt=dt_value,
            t0=float(t0),
            escapers=policy,
        )
        step, segment_done, t0_run = 0, 0, float(t0)
        anchor: tuple = (t0_run, 0)
        baselines: Optional[dict] = None
        ledgers = {"strip": 0.0, "escapers": 0.0 if policy is not None else None}
    elif isinstance(restart_from, RunResult):
        res = restart_from
        if res.carried is None:
            raise ValueError(
                "this RunResult is not a carried scheme's: continue it with the "
                "runner that made it"
            )
        carry = res.carried
        step, segment_done, t0_run = res.step, res.segment, res.t0
        anchor = res.time_anchor if dt_value == res.dt else (res.time, res.step)
        baselines = res.baselines
        ledgers = dict(res.escapers or {}).get("ledgers") or {
            "strip": 0.0,
            "escapers": 0.0 if policy is not None else None,
        }
        restored = {"from": "RunResult", "step": step, "bitwise": True}
        res.carried = None  # consumed
    elif isinstance(restart_from, (str, os.PathLike)):
        path = str(restart_from)
        if is_checkpoint(path):
            from jaccpot.checkpoint import read_meta, restore_arrays, verify_shards

            meta = read_meta(path)
            names = list(meta["manifest"]["arrays"])
            arrays = restore_arrays(path, {n: (None, None) for n in names})
            verify_shards(arrays, meta["manifest"])
        else:
            arrays, meta = load_restart(path)
        if meta.get("carried") is None:
            raise ValueError(
                f"{path} was not written by a carried scheme's run: continue it "
                "with ProductionRunner(scheme=None or a SplitStepScheme)"
            )
        if (meta.get("escapers") is None) != (policy is None):
            raise ValueError(
                "a run continues in its own escaper mode: "
                + (
                    "this one ran with escapers= (pass the runner the policy)"
                    if policy is None
                    else "this one ran without escapers (escapers=None)"
                )
            )
        carry, mismatches = scheme.restore(
            arrays,
            meta["carried"],
            solver=runner.solver,
            settings=runner.settings,
            dt=dt_value,
            escapers=policy,
        )
        del arrays
        counters = meta["counters"]
        step, segment_done = int(counters["step"]), int(counters["segment"])
        t0_run = float(counters["t0"])
        anchor = (
            tuple(counters["anchor"])
            if dt_value == float(counters["dt"])
            else (float(counters["time"]), step)
        )
        baselines = meta["baselines"]
        ledgers = dict(meta["ledgers"])
        written = meta["fingerprint"]
        now = _fingerprint(runner, carry)
        mismatches = list(mismatches) + [
            key
            for key in sorted(set(written) | set(now))
            if written.get(key) != now.get(key)
        ]
        if dt_value != float(counters["dt"]):
            mismatches.append("dt")
        if mismatches:
            warnings.warn(
                f"restart {path}: {mismatches} differ from the run that wrote it; "
                "the continuation is correct but not bitwise the uninterrupted run",
                stacklevel=3,
            )
        restored = {
            "from": path,
            "step": step,
            "bitwise": not mismatches,
            "mismatches": mismatches,
        }
    else:
        raise TypeError(
            "restart_from must be a RunResult or a restart file, got "
            f"{type(restart_from).__name__}"
        )
    seg = runner.segment
    names = tuple(seg.diagnostics)
    every = int(seg.diagnostics_every)
    every_energy = int(seg.diagnostics_every_energy)
    energy_on = every > 0 and every_energy > 0 and "energy" in names
    if policy is not None and not energy_on:
        ledgers["escapers"] = None  # an escaper jump needs the energy
    edges: list = []
    restarts: list = []
    stopped = False

    def time_at(step_now: int) -> float:
        return anchor[0] + (step_now - anchor[1]) * dt_value

    if baselines is None and every > 0:
        baselines = _baselines(scheme.moments(carry), time_at(step))
        if energy_on:
            e0 = scheme.energy(carry)
            pot = _potential_of(e0, policy is not None)
            baselines["energy"] = (
                float(scheme.moments(carry)["kinetic"])
                + pot["self"]
                + pot.get("couple", 0.0)
            )
            baselines["potential"] = pot["self"] + pot.get("couple", 0.0)

    def edge_row(
        info: dict, energy_due: bool, diag_due: bool, escaper_edge: dict
    ) -> Optional[dict]:
        if not diag_due or baselines is None:
            return None
        energy = scheme.energy(carry) if energy_due else None
        potential = (
            None if energy is None else _potential_of(energy, policy is not None)
        )
        row = _diagnostics_row(
            scheme.moments(carry),
            baselines,
            time_at(step),
            names,
            energy_due,
            potential,
        )
        row.update(
            _cluster_fields(energy, info, ledgers, baselines, row, policy is not None)
        )
        if policy is not None:
            esc = _escaper_fields(
                {
                    "n_bound": int(info["n_galaxy_rows"]) - int(info["n_escapers"]),
                    "ledger": ledgers["escapers"],
                },
                escaper_edge,
                row,
                baselines,
                int(info["n_galaxy_rows"]),
                potential,
                energy_on,
            )
            # the ledgered energy is the clusters' (both ledgers): keep ours
            esc.pop("energy_ledgered", None)
            esc.pop("energy_ledgered_drift", None)
            row.update(esc)
        return row

    def emit(
        retries: int, wall: float, info: dict, escaper_edge: dict, final: bool
    ) -> bool:
        nonlocal stopped
        diag_due = every > 0 and (segment_done % every == 0 or final)
        energy_due = (
            energy_on
            and diag_due
            and (segment_done == 0 or segment_done % every_energy == 0 or final)
        )
        row = edge_row(info, energy_due, diag_due, escaper_edge)
        runner._current = {
            "carried": carry,
            "step": step,
            "time": time_at(step),
            "dt": dt_value,
            "t0": t0_run,
            "segment": segment_done,
            "num_steps": int(num_steps),
            "anchor": anchor,
            "baselines": baselines,
            "ledgers": dict(ledgers),
        }
        written = None
        every_restart = int(seg.restart_every)
        if every_restart and segment_done and segment_done % every_restart == 0:
            path = str(seg.restart_path).format(step=step, segment=segment_done)
            written = save_carried(runner, path)
            restarts.append(written)
        view = None
        if runner.callbacks:
            view = scheme.galaxy(carry)
        edge = SegmentEdge(
            step=step,
            time=time_at(step),
            segment=segment_done,
            retries=retries,
            wall_s=wall,
            state=None if view is None else view[0],
            masses=None if view is None else view[1],
            self_acceleration=None if view is None else view[2],
            diagnostics=row,
            plan=dict(info.get("plan", {}), replans=info.get("replans", [])),
            carried=carry,
        )
        stop = False
        for cb in runner.callbacks:
            if cb(edge):
                stop = True
        edges.append(
            SegmentEdge(
                step=step,
                time=time_at(step),
                segment=segment_done,
                retries=retries,
                wall_s=wall,
                state=None,
                masses=None,
                self_acceleration=None,
                diagnostics=row,
                plan=edge.plan,
            )
        )
        del written
        return stop

    if restart_from is None:
        stopped = emit(0, 0.0, scheme.describe(carry), {}, False)
    lengths = segment_lengths(int(num_steps) - step, int(seg.steps))
    if int(num_steps) - step > 0:
        runner._warn_remainder(lengths)
    max_retries = int(getattr(scheme, "max_retries", 2))
    # Donation (CarriedScheme.keep_start/restore_start): the segment consumes its start and
    # the retry restores it from a host copy, so one galaxy state is on the device.
    # The first segment's carry may hold the caller's arrays (a fresh start):
    # donated only with donate_input or after a restart from a file.
    donates = callable(getattr(scheme, "keep_start", None)) and callable(
        getattr(scheme, "restore_start", None)
    )
    owned_first = bool(getattr(seg, "donate_input", False)) or isinstance(
        restart_from, (str, os.PathLike)
    )

    def run_segment(start: Any, length: int, donate: bool) -> Any:
        if donate:
            return scheme.segment(start, dt_value, length, donate=True)
        return scheme.segment(start, dt_value, length)

    for index, length in enumerate(lengths):
        if stopped:
            break
        wall0 = time.perf_counter()
        donate = donates and (index > 0 or owned_first)
        start = carry
        kept = scheme.keep_start(start) if donate else None
        carry = run_segment(start, int(length), donate)
        if donate:
            del start  # consumed by the segment
        retries = 0
        reasons_all: list = []
        while True:
            reasons = scheme.verdict(carry)
            if not reasons:
                break
            reasons_all.append(list(reasons))
            if retries >= max_retries:
                raise RuntimeError(
                    f"segment {segment_done + 1}: the carried scheme's verdict failed "
                    f"after {retries} re-plans ({reasons}); last restart: "
                    f"{runner._last_restart}"
                )
            if donate:
                start = scheme.restore_start(kept)
            carry = run_segment(scheme.replanned(start, carry), int(length), donate)
            if donate:
                del start
            retries += 1
        if not donate:
            del start
        del kept
        step += int(length)
        segment_done += 1
        escapers_due = policy is not None and segment_done % int(policy.every) == 0
        carry, info = scheme.edge(
            carry,
            index=segment_done,
            escapers_due=escapers_due,
            energy=energy_on,
        )
        ledgers["strip"] = float(info["strip_ledger"])
        escaper_edge: dict = {}
        if policy is not None and info.get("escaper_transfers"):
            jump = info.get("escaper_jump")
            escaper_edge = {"transfers": int(info["escaper_transfers"]), "jump": jump}
            if jump is None or ledgers["escapers"] is None:
                ledgers["escapers"] = None
            else:
                ledgers["escapers"] = float(ledgers["escapers"]) + float(jump)
        if reasons_all:
            info = dict(info, replans=reasons_all + list(info.get("replans", [])))
        final = index == len(lengths) - 1
        stopped = emit(retries, time.perf_counter() - wall0, info, escaper_edge, final)
    view = scheme.galaxy(carry)
    return RunResult(
        state=view[0],
        masses=view[1],
        handle=None,
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
        mode="single",
        escapers=(
            None
            if policy is None
            else {
                "ledger": ledgers["escapers"],
                "ledgers": dict(ledgers),
                **scheme.escaper_split(carry),
            }
        ),
        carried=carry,
    )


def _fingerprint(runner: Any, carry: Any) -> dict:
    """The runner's fingerprint for a carried run (its galaxy rows) and the scheme's record.

    Parameters
    ----------
    runner : Any
        The runner.
    carry : Any
        The carry.

    Returns
    -------
    dict
        ``ProductionRunner._fingerprint`` plus ``carried``.
    """
    x, m = runner.scheme.fingerprint_arrays(carry)
    out = runner._fingerprint(x, m)
    out["carried"] = runner.scheme.record()
    return out


def save_carried(runner: Any, path: Any) -> str:
    """:meth:`ProductionRunner.save_restart` for a carried run.

    ``path`` ending in ``.npz``: one ``.npz`` (the runner's format: arrays and the
    ``meta`` JSON), written to a temporary name and renamed. Anything else: an orbax
    checkpoint directory (:func:`jaccpot.checkpoint.write_checkpoint`; the
    ``jaccpot[io]`` extra) holding the same arrays and ``meta`` plus the manifest
    (sha256 per array). Either continues bitwise in a fresh process.

    Parameters
    ----------
    runner : Any
        The runner.
    path : Any
        The file or checkpoint directory.

    Returns
    -------
    str
        ``path``.

    Raises
    ------
    RuntimeError
        Before :meth:`ProductionRunner.run` reached an edge.
    """
    import json

    import jax

    from jaccpot.production import RESTART_FORMAT, _atomic_savez
    from jaccpot.scheme import scheme_record

    current = runner._current
    if current is None or current.get("carried") is None:
        raise RuntimeError("save_restart needs a carried run that reached an edge")
    carry = current["carried"]
    arrays, carried_meta = runner.scheme.checkpoint(carry)
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
        "ledgers": current["ledgers"],
        "scheme": scheme_record(runner.scheme),
        "carried": carried_meta,
        "fingerprint": _fingerprint(runner, carry),
    }
    if runner.escapers is not None:
        meta["escapers"] = {
            **runner.escapers.record(),
            "ledger": current["ledgers"].get("escapers"),
        }
    path = str(path)
    if _is_orbax_path(path):
        import jax.numpy as jnp

        from jaccpot.checkpoint import write_checkpoint

        write_checkpoint(path, {k: jnp.asarray(v) for k, v in arrays.items()}, meta)
    else:
        host = {k: np.asarray(jax.device_get(v)) for k, v in arrays.items()}
        host["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)
        _atomic_savez(path, **host)
    runner._last_restart = path
    return path
