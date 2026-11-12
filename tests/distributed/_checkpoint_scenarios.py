"""Scenarios for ``test_checkpoint_multiprocess.py`` (M3): run in child processes.

Each function takes the device count ``n`` and returns host arrays and JSON-able
values; ``_multiprocess_harness`` runs it as ONE process with ``n`` forced CPU
devices (``sp``) or as ``n`` processes with one device each (``mp``). The
checkpoints go under ``M3_ROOT/<side>`` (``M3_SIDE``: the harness's mode unless
set), so a later scenario in other processes -- of another count -- can read them
(``M3_FROM``).

The system is ESC's (``test_production_runner_mesh_escapers._system``: a Plummer
sphere and eight fliers leaving ``r_escape``), 6 segments of 10 steps, the energy at
every edge, migrating. ``M3_FORCE``: ``direct`` (``DirectMeshForce``, the default)
or ``fused`` (the CPU fused lane, ``M3_MAC`` ``dehnen`` / ``dehnen_error``);
``M3_ESC``: ``1`` escapers on (the default), ``0`` off.

Not a test module (no ``test_`` prefix): the children import it after
``jax.distributed`` is initialised.
"""

from __future__ import annotations

import json
import os
import warnings

import jax
import numpy as np

from jaccpot.distributed import multiprocess as mp

AXIS = "gpus"
#: the checkpoint after which the restarts continue
MID = 3


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _side() -> str:
    return _env("M3_SIDE", "mp" if mp.is_multiprocess() else "sp")


def _input(state: np.ndarray, masses: np.ndarray, mesh) -> tuple:
    """The whole set (one process) or this process's ids."""
    if not mp.is_multiprocess():
        return state, masses
    lo, hi = mp.local_id_range(int(state.shape[0]), mesh)
    return (
        mp.ProcessLocalParticles(state[lo:hi], masses[lo:hi], int(state.shape[0])),
        None,
    )


def _runner(n: int, **segment):
    """The scenario's runner on an ``n``-device mesh (``n == 0``: one card)."""
    from jaccpot import FastMultipoleMethod
    from jaccpot.distributed.segment import DirectMeshForce
    from jaccpot.escapers import EscaperPolicy
    from jaccpot.production import MeshRunConfig, ProductionRunner, SegmentConfig
    from tests.distributed import test_production_runner_mesh_escapers as tme

    fused = _env("M3_FORCE", "direct") == "fused"
    mac = _env("M3_MAC", "dehnen")
    escapers = _env("M3_ESC", "1") == "1"
    if fused:
        from tests.distributed._multiprocess_scenarios import _cpu_fused_lane
        from tests.integration.strict import test_escapers as te

        _cpu_fused_lane(mac == "dehnen_error")
        solver, settings, force = te._solver(mac), te._settings(mac), None
    else:
        solver = FastMultipoleMethod(preset="large_n_gpu", G=tme.G, softening=tme.SOFT)
        settings = None
        force = DirectMeshForce(G=tme.G, softening=tme.SOFT, axis_name=AXIS)
    mesh = None
    if n:
        mesh = MeshRunConfig(
            mp.global_mesh(AXIS),
            cap=int(np.ceil(1.5 * (tme.N + tme.K + 1) / n)),
            num_samples=64,
            migrate=True,
            force=force,
            checkpoint="orbax",
        )
    return ProductionRunner(
        solver,
        **({} if settings is None else {"settings": settings}),
        segment=SegmentConfig(steps=tme.STEPS, diagnostics_every_energy=1, **segment),
        escapers=EscaperPolicy(tme.R_ESCAPE, tme.R_RETURN) if escapers else None,
        mesh=mesh,
    )


def _total_steps() -> int:
    from tests.distributed import test_production_runner_mesh_escapers as tme

    return tme.STEPS * tme.SEGMENTS


def _rows(res) -> list:
    """Each edge's diagnostics (JSON-able; ``wall_s`` left out)."""
    return [
        {
            k: (None if v is None else np.asarray(v).tolist())
            for k, v in (edge.diagnostics or {}).items()
        }
        for edge in res.edges
    ]


def _result(res, tag: str) -> dict:
    """A run's final state, carried arrays, edges and escaper record."""
    out = {f"{tag}_state": np.asarray(res.state), f"{tag}_edges": _rows(res)}
    if res.mode == "mesh":
        got = res.handle.gather()
        for name in ("acceleration", "force_scale"):
            if name in got:
                out[f"{tag}_{name}"] = got[name]
    if res.escapers is not None:
        out[f"{tag}_split"] = {
            k: res.escapers[k] for k in ("n_bound", "ledger", "transfers")
        }
    if res.restored is not None:
        out[f"{tag}_restored"] = {
            k: res.restored.get(k) for k in ("bitwise", "mismatches", "layout")
        }
    return out


def write_and_restart(n: int) -> dict:
    """Gate 1 on ``n`` devices: the run in one go writing a checkpoint at every
    edge; a fresh runner restarted from the middle one; a chain of three runs
    each restarted from the previous one's last checkpoint. The checkpoints stay
    under ``M3_ROOT/<side>`` for the cross-count scenarios."""
    from tests.distributed import test_production_runner_mesh_escapers as tme

    root = os.path.join(_env("M3_ROOT", "."), _side())
    os.makedirs(root, exist_ok=True)
    total = _total_steps()
    state, masses = tme._system()
    s, m = _input(state, masses, mp.global_mesh(AXIS))
    once = _runner(
        n, restart_every=1, restart_path=os.path.join(root, "once_{segment}")
    ).run(s, m, dt=tme.DT, num_steps=total)
    out = _result(once, "once")
    out["restarts"] = [os.path.basename(p) for p in once.restarts]
    resumed = _runner(n).run(
        dt=tme.DT, num_steps=total, restart_from=os.path.join(root, f"once_{MID}")
    )
    out.update(_result(resumed, "resumed"))
    # the chain: 2 segments, restart; 2 more, restart; the rest
    chain = os.path.join(root, "chain_{segment}")
    s, m = _input(state, masses, mp.global_mesh(AXIS))
    _runner(n, restart_every=2, restart_path=chain).run(
        s, m, dt=tme.DT, num_steps=2 * tme.STEPS
    )
    _runner(n, restart_every=2, restart_path=chain).run(
        dt=tme.DT, num_steps=4 * tme.STEPS, restart_from=chain.format(segment=2)
    )
    last = _runner(n).run(
        dt=tme.DT, num_steps=total, restart_from=chain.format(segment=4)
    )
    out.update(_result(last, "chained"))
    return out


def restore_from(n: int) -> dict:
    """Continue ``M3_FROM`` (a checkpoint written elsewhere) on ``n`` devices
    (``M3_SINGLE=1``: on one card, no mesh) to the end."""
    from tests.distributed import test_production_runner_mesh_escapers as tme

    single = _env("M3_SINGLE", "0") == "1"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _runner(0 if single else n).run(
            dt=tme.DT, num_steps=_total_steps(), restart_from=_env("M3_FROM", "")
        )
    out = _result(res, "restored")
    out["warned"] = [
        str(w.message)[:300] for w in caught if "restart" in str(w.message)
    ]
    return out


def snapshots(n: int) -> dict:
    """``SnapshotWriter(sharded=True)`` on the mesh: every edge's snapshot read
    back by id equals the edge's gathered state."""
    from jaccpot.checkpoint import load_checkpoint
    from jaccpot.production import SnapshotWriter
    from tests.distributed import test_production_runner_mesh_escapers as tme

    root = os.path.join(_env("M3_ROOT", "."), _side(), "snapshots")
    writer = SnapshotWriter(root, every=2, sharded=True)
    seen: dict = {}

    def keep(edge):
        if edge.segment % 2 == 0:
            seen[edge.step] = edge.state_host()

    runner = _runner(n)
    runner.callbacks = (writer, keep)
    state, masses = tme._system()
    s, m = _input(state, masses, mp.global_mesh(AXIS))
    runner.run(s, m, dt=tme.DT, num_steps=_total_steps())
    out = {"paths": [os.path.basename(p) for p in writer.paths]}
    same = []
    for path in writer.paths:
        got, meta = load_checkpoint(path, ["state"])
        same.append(bool(np.array_equal(got["state"], seen[meta["step"]])))
    out["same"] = same
    out["snap_last"] = got["state"]
    return out


def dump(result: dict, path: str) -> None:
    """For debugging: a scenario's JSON part."""
    with open(path, "w") as fh:
        json.dump(
            {k: v for k, v in result.items() if not isinstance(v, np.ndarray)},
            fh,
            default=str,
        )


__all__ = ["restore_from", "snapshots", "write_and_restart"]

# keep jax imported for the children (the harness imports this after init)
_ = jax
