"""M3 on HoreKa: sharded checkpoint I/O per rank, elastic restores, walltime chaining.

One process per GPU (``srun``, ``jaccpot.distributed.multiprocess.initialize()``
from the SLURM step) or one process driving every local GPU (``--arm sp``). Each
process writes ``result_<cmd>_p<rank>.json`` under ``--out``.

``write``
    A Plummer sphere of ``--n`` particles (each process draws only its id range, in
    fixed id blocks: the same IC on any process count), one segment of ``--steps``
    steps on the global mesh, then ``--reps`` sharded checkpoints of that edge to
    ``--ckpt`` (the second and later REPLACE the first: the atomic swap is in the
    timing). Per process: write seconds, the bytes of its own shards, the bytes on
    disk of its OCDBT database, peak host RSS and device peak.
``restore``
    The restore of ``--ckpt`` on THIS mesh (the same device count: each process
    reads its shards; another: the elastic re-sharding restore), timed per process
    (the restore program's compile included, and a second restore without it),
    then ``--steps`` more steps as one segment (the continuation runs).
``chain``
    The walltime-chaining pattern (multi-node plan M5): continue from the newest
    committed checkpoint matching ``--pattern`` (or start fresh), run toward
    ``--target-steps`` writing a checkpoint every ``--every`` segments, and when
    the job's ``--deadline`` (epoch seconds) is within ``--margin`` seconds -- a
    decision every process takes together -- write one more checkpoint and stop.
    ``STATUS`` in ``--out`` says ``done`` or ``continue`` (the sbatch resubmits).

The by-id gather a ``RunResult`` ends with is ``O(N)`` host memory on EVERY
process (M2's ``host_value``); at ``--n`` above ``--lazy-result-above`` (default
2e8) this bench replaces it with a stub (the bench reads nothing from it). An open
M5 item.

    srun -n 4 python bench/m3_checkpoint_io.py write --n 1e8 --ckpt $W/ck --out $W/w4
    srun -n 2 python bench/m3_checkpoint_io.py restore --ckpt $W/ck --out $W/r2
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import socket
import sys
import time

_FUSION = "--xla_gpu_experimental_enable_fusion_autotuner=false"
#: ids per independently seeded draw of the IC (process-count independent)
_BLOCK = 1 << 20


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("write", "restore", "chain"):
        q = sub.add_parser(name)
        q.add_argument("--arm", choices=("sp", "mp"), default="mp")
        q.add_argument("--out", required=True)
        q.add_argument("--steps", type=int, default=2, help="steps per segment")
        q.add_argument("--dt", type=float, default=2e-3)
        q.add_argument("--softening", type=float, default=1e-3)
        q.add_argument(
            "--mac", choices=("dehnen_error", "dehnen"), default="dehnen_error"
        )
        q.add_argument("--lazy-result-above", type=float, default=2e8)
    w = sub.choices["write"]
    w.add_argument("--n", type=float, required=True)
    w.add_argument("--ckpt", required=True)
    w.add_argument("--reps", type=int, default=2)
    w.add_argument("--seed", type=int, default=11)
    r = sub.choices["restore"]
    r.add_argument("--ckpt", required=True)
    r.add_argument("--reps", type=int, default=2)
    c = sub.choices["chain"]
    c.add_argument("--n", type=float, required=True)
    c.add_argument(
        "--pattern", required=True, help="restart_path, e.g. $W/ck_{step:08d}"
    )
    c.add_argument("--target-steps", type=int, required=True)
    c.add_argument("--every", type=int, default=1, help="segments per checkpoint")
    c.add_argument("--deadline", type=float, default=0.0, help="epoch s; 0: none")
    c.add_argument("--margin", type=float, default=600.0)
    c.add_argument("--seed", type=int, default=11)
    return p.parse_args()


ARGS = _args()
_flags = os.environ.get("XLA_FLAGS", "")
if _FUSION.split("=")[0] not in _flags:
    os.environ["XLA_FLAGS"] = f"{_flags} {_FUSION}".strip()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def _join() -> None:
    if ARGS.arm == "mp":
        from jaccpot.distributed import multiprocess as mp

        mp.initialize(initialization_timeout=900)


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _du(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _plummer_range(lo: int, hi: int, seed: int) -> tuple:
    """Ids ``lo..hi-1`` of a Plummer sphere (unit mass, G = 1), float32.

    Drawn in blocks of ``_BLOCK`` ids, block ``b`` from ``default_rng([seed, b])``:
    every process count draws the same particles.
    """
    states, masses = [], []
    for b in range(lo // _BLOCK, (hi - 1) // _BLOCK + 1 if hi > lo else 0):
        rng = np.random.default_rng([seed, b])
        k = _BLOCK
        x = rng.uniform(1e-6, 1.0, k)
        r = np.minimum(1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0), 30.0)
        d = rng.normal(size=(k, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        v_esc = np.sqrt(2.0) * (1.0 + r * r) ** -0.25
        u = rng.normal(size=(k, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        speed = 0.5 * v_esc * rng.uniform(0.0, 1.0, k) ** (1.0 / 3.0)
        a, z = b * _BLOCK, (b + 1) * _BLOCK
        take = slice(max(lo, a) - a, min(hi, z) - a)
        state = np.stack([r[:, None] * d, speed[:, None] * u], axis=1)[take]
        states.append(state.astype(np.float32))
        masses.append(np.full(state.shape[0], 1.0, np.float32))
    return np.concatenate(states), np.concatenate(masses)


def _runner(mesh, **segment):
    import dataclasses

    from jaccpot import FastMultipoleMethod
    from jaccpot.production import (
        GALAXY_SETTINGS,
        GalaxySettings,
        MeshRunConfig,
        ProductionRunner,
        SegmentConfig,
    )

    if ARGS.mac == "dehnen_error":
        solver = FastMultipoleMethod(preset="galaxy", softening=ARGS.softening)
        settings = GALAXY_SETTINGS
    else:
        base = FastMultipoleMethod(preset="galaxy").advanced
        solver = FastMultipoleMethod(
            preset="galaxy",
            softening=ARGS.softening,
            advanced=dataclasses.replace(base, mac_type="dehnen"),
        )
        settings = GalaxySettings(mac_type="dehnen", adaptive_eps=None)
    return ProductionRunner(
        solver,
        settings=settings,
        segment=SegmentConfig(steps=ARGS.steps, diagnostics_every_energy=0, **segment),
        mesh=MeshRunConfig(mesh, checkpoint="orbax"),
    )


def _input(num: int, mesh) -> tuple:
    from jaccpot.distributed import multiprocess as mp

    # masses 1/N: total mass 1
    if mp.is_multiprocess():
        lo, hi = mp.local_id_range(num, mesh)
        s, m = _plummer_range(lo, hi, ARGS.seed)
        return mp.ProcessLocalParticles(s, m / np.float32(num), num), None
    s, m = _plummer_range(0, num, ARGS.seed)
    return s, m / np.float32(num)


def _lazy_result(num: int) -> None:
    """Above ``--lazy-result-above``: the run's final by-id gather becomes a stub."""
    if num <= ARGS.lazy_result_above:
        return
    import jaccpot.distributed.mesh_runner as mr

    mr.gather_by_id = lambda *a, **k: {"state": None, "masses": None}


def _record() -> dict:
    import jax

    return {
        "cmd": ARGS.cmd,
        "arm": ARGS.arm,
        "process": int(jax.process_index()),
        "processes": int(jax.process_count()),
        "devices": int(jax.device_count()),
        "local_devices": [str(d) for d in jax.local_devices()],
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
    }


def _device_peaks() -> list:
    import jax

    return [
        int((d.memory_stats() or {}).get("peak_bytes_in_use", 0))
        for d in jax.local_devices()
    ]


def _finish(record: dict) -> None:
    import jax

    record["peak_rss_mb"] = _rss_mb()
    record["device_peak_bytes"] = _device_peaks()
    os.makedirs(ARGS.out, exist_ok=True)
    name = f"result_{ARGS.cmd}_p{int(jax.process_index())}.json"
    with open(os.path.join(ARGS.out, name), "w") as fh:
        json.dump(record, fh, indent=1, default=str)
    print(
        f"[p{record['process']}] {ARGS.cmd}: {json.dumps(record.get('summary'))}",
        flush=True,
    )
    if ARGS.arm == "mp":
        jax.distributed.shutdown()


def write() -> None:
    _join()
    import jax

    from jaccpot.distributed import multiprocess as mp

    mesh = mp.global_mesh()
    num = int(ARGS.n)
    _lazy_result(num)
    record = _record()
    t0 = time.perf_counter()
    s, m = _input(num, mesh)
    record["ic_s"] = time.perf_counter() - t0
    runner = _runner(mesh)
    t0 = time.perf_counter()
    runner.run(s, m, dt=ARGS.dt, num_steps=ARGS.steps)
    record["segment_s_incl_compile"] = time.perf_counter() - t0
    del s, m
    rows = []
    for rep in range(int(ARGS.reps)):
        t0 = time.perf_counter()
        path = runner.save_restart(ARGS.ckpt)
        wall = time.perf_counter() - t0
        stats = dict(runner._last_checkpoint_stats)
        rows.append(
            {
                "rep": rep,
                "wall_s": wall,
                "seconds": stats["seconds"],
                "bytes_local": stats["bytes_local"],
                "bytes_total": stats["bytes"],
                "replace": rep > 0,
            }
        )
    me = int(jax.process_index())
    db = os.path.join(path, "arrays", f"ocdbt.process_{me}")
    record.update(
        n=num,
        ckpt=path,
        writes=rows,
        disk_bytes_this_process=_du(db),
        disk_bytes_total=_du(path) if me == 0 else None,
        summary={
            "n": num,
            "write_s": [round(r["wall_s"], 3) for r in rows],
            "bytes_local": rows[-1]["bytes_local"],
        },
    )
    _finish(record)


def restore() -> None:
    _join()
    import warnings

    import jax

    from jaccpot.checkpoint import read_meta
    from jaccpot.distributed import multiprocess as mp

    mesh = mp.global_mesh()
    meta = read_meta(ARGS.ckpt)
    num = int(meta["checkpoint"]["num_particles"])
    _lazy_result(num)
    record = _record()
    record.update(
        n=num,
        written_on=meta["checkpoint"]["processes"],
        written_ndev=meta["mesh"]["ndev"],
    )
    runner = _runner(mesh)
    backend = runner._mesh_backend
    rows = []
    msr = restored = None
    for rep in range(int(ARGS.reps)):
        del msr
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            msr, restored = backend._restore_checkpoint(
                ARGS.ckpt, meta, dt_value=ARGS.dt, external_arg=None
            )
        jax.block_until_ready(msr.state.state)
        rows.append({"rep": rep, "wall_s": time.perf_counter() - t0})
    record["restores"] = rows
    record["restored"] = {k: restored[k] for k in ("bitwise", "mismatches", "layout")}
    t0 = time.perf_counter()
    report = msr.run(int(ARGS.steps), dt=ARGS.dt)
    jax.block_until_ready(msr.state.state)
    record["continuation_s_incl_compile"] = time.perf_counter() - t0
    record["continuation_retries"] = int(report.retries)
    record["summary"] = {
        "n": num,
        "from_ndev": meta["mesh"]["ndev"],
        "to_ndev": int(np.asarray(mesh.devices).size),
        "restore_s": [round(r["wall_s"], 3) for r in rows],
        "layout": restored["layout"],
    }
    _finish(record)


class _Walltime:
    """A callback: stop (after a checkpoint) once the deadline is near, decided together."""

    def __init__(self, runner, deadline: float, margin: float) -> None:
        self.runner, self.deadline, self.margin = runner, deadline, margin
        self.stopped_at = None
        self.last_segment_s = 0.0
        self._t = time.time()

    def __call__(self, edge) -> bool:
        from jaccpot.distributed import multiprocess as mp

        now = time.time()
        if edge.segment:
            self.last_segment_s = now - self._t
        self._t = now
        if self.deadline <= 0 or edge.segment == 0:
            return False  # (step 0 is the run's input: nothing to checkpoint)
        # the next segment and a checkpoint must fit before the deadline
        need = self.margin + 2.0 * self.last_segment_s
        late = now + need >= self.deadline
        if mp.is_multiprocess():
            late = any(bool(v) for v in mp.allgather_object(bool(late)))
        if late:
            self.stopped_at = edge.step
            path = str(self.runner.segment.restart_path).format(
                step=edge.step, segment=edge.segment
            )
            # the cadence may have written this edge's checkpoint already
            if os.path.abspath(path) != os.path.abspath(
                str(self.runner._last_restart or "")
            ):
                self.runner.save_restart(path)
        return late


def chain() -> None:
    _join()
    import jax

    from jaccpot.checkpoint import latest_checkpoint
    from jaccpot.distributed import multiprocess as mp

    mesh = mp.global_mesh()
    num = int(ARGS.n)
    _lazy_result(num)
    record = _record()
    runner = _runner(mesh, restart_every=int(ARGS.every), restart_path=ARGS.pattern)
    guard = _Walltime(runner, ARGS.deadline, ARGS.margin)
    runner.callbacks = (guard,)
    latest = latest_checkpoint(ARGS.pattern)
    t0 = time.perf_counter()
    if latest is None:
        s, m = _input(num, mesh)
        res = runner.run(s, m, dt=ARGS.dt, num_steps=int(ARGS.target_steps))
        del s, m
    else:
        res = runner.run(
            dt=ARGS.dt, num_steps=int(ARGS.target_steps), restart_from=latest
        )
    status = "done" if res.step >= int(ARGS.target_steps) else "continue"
    record.update(
        n=num,
        resumed_from=latest,
        step=int(res.step),
        wall_s=time.perf_counter() - t0,
        restarts=list(res.restarts),
        stopped_at=guard.stopped_at,
        status=status,
        summary={"from": latest, "step": int(res.step), "status": status},
    )
    if jax.process_index() == 0:
        os.makedirs(ARGS.out, exist_ok=True)
        with open(os.path.join(ARGS.out, "STATUS"), "w") as fh:
            fh.write(status + "\n")
    _finish(record)


if __name__ == "__main__":
    {"write": write, "restore": restore, "chain": chain}[ARGS.cmd]()
