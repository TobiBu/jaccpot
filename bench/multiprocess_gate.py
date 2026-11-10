"""M2 on GPUs: the multi-GPU runner as N processes x 1 GPU against 1 process x N GPUs.

One entry point for both arms, launched by SLURM (``bench/slurm/m2g/``) or by hand:

* ``run --arm sp``: ONE process drives every visible GPU (``srun --ntasks=1``);
* ``run --arm mp``: one process per GPU, joined by
  ``jaccpot.distributed.multiprocess.initialize()`` with NO arguments: jax's SLURM
  detection (``SLURM_NTASKS``, ``SLURM_PROCID``, ``SLURM_LOCALID``, the first node
  of ``SLURM_STEP_NODELIST`` as the coordinator). A task that sees all of the
  node's cards takes card ``SLURM_LOCALID`` (jax's rule); a task SLURM bound to
  one card takes that card (``multiprocess.slurm_bound_device_ids``). Each
  process passes only its id range (``ProcessLocalParticles``).

Both build the same Plummer sphere (``--n-per-gpu`` per card, seeded; or, with
``--ic disc``, a prefix of the 25M disc + bulge IC ``--disc-ic`` at ``--n`` particles,
its masses rescaled, as ``bench/multigpu_segment_gate.py`` takes it), the
production preset (``galaxy``: leaf 64, p6, ``dehnen_error`` at 1e-4; ``--mac
dehnen`` for the other MAC), migration on, a repartition before every segment
after the first, the energy at every edge, and run it ``--repeats`` times in the
process (the first includes the compiles; the others are the warm timing).
Process 0 saves the final state, acceleration and owner by id of the first
repeat (``state.npy`` ...); every process writes ``result_p<rank>.json``
(walls per run and per edge, peak bytes of its cards, the SLURM and CUDA
environment it saw, the edges' diagnostics).

``compare A B`` prints, per array, how many elements differ bit for bit between
two arms' outputs, and whether the edges' diagnostics are equal. Run every arm
twice (A-vs-A: the GPU lane's production flags are not deterministic run to run
under ``dehnen_error``; ``--deterministic`` adds ``--xla_gpu_deterministic_ops=
true`` for the bitwise gate).

``collectives --arm sp|mp``: the mesh runner's collectives (psum f32/f64/int,
pmax, pmin, all_gather, all_to_all, ppermute, the native ragged_all_to_all,
three calls each) against host references, and the float psum against the
mesh-order fold (``tests/distributed/_multiprocess_scenarios.collectives``).

    srun --ntasks=4 --ntasks-per-node=4 python bench/multiprocess_gate.py run \\
        --arm mp --out OUT/mp_a
    srun --ntasks=1 python bench/multiprocess_gate.py run --arm sp --out OUT/sp_a
    python bench/multiprocess_gate.py compare OUT/sp_a OUT/mp_a
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ragged_exchange_flags import add_ragged_exchange_flags  # noqa: E402

_FUSION = "--xla_gpu_experimental_enable_fusion_autotuner=false"


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--arm", choices=("sp", "mp"), required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--n-per-gpu", type=float, default=2e6)
    run.add_argument("--ic", choices=("plummer", "disc"), default="plummer")
    run.add_argument(
        "--n",
        type=float,
        default=None,
        help="total particles (default: --n-per-gpu times the devices); --ic disc needs it",
    )
    run.add_argument(
        "--disc-ic",
        default=os.environ.get(
            "DISC_IC", "/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz"
        ),
        help="--ic disc: the 25M disc + bulge .npz (state0, mass, G_code, rdisk_code)",
    )
    run.add_argument(
        "--softening",
        type=float,
        default=None,
        help="default: 1e-3 (Plummer), the disc formula 0.5 rdisk / sqrt(N / 1e5) (disc)",
    )
    run.add_argument(
        "--mac", choices=("dehnen_error", "dehnen"), default="dehnen_error"
    )
    run.add_argument("--segments", type=int, default=3)
    run.add_argument("--steps", type=int, default=4, help="steps per segment")
    run.add_argument(
        "--dt", type=float, default=None, help="default 2e-3 (Plummer), 5e-4 (disc)"
    )
    run.add_argument("--repeats", type=int, default=2)
    run.add_argument("--no-migrate", action="store_true")
    run.add_argument("--deterministic", action="store_true")
    col = sub.add_parser("collectives")
    col.add_argument("--arm", choices=("sp", "mp"), required=True)
    col.add_argument("--out", required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("a")
    cmp.add_argument("b")
    cmp.add_argument("--json", default=None)
    return p.parse_args()


ARGS = _args()
if ARGS.cmd != "compare":
    flags = os.environ.get("XLA_FLAGS", "")
    if _FUSION.split("=")[0] not in flags:
        flags = f"{flags} {_FUSION}".strip()
    if getattr(ARGS, "deterministic", False) and "deterministic_ops" not in flags:
        flags = f"{flags} --xla_gpu_deterministic_ops=true"
    os.environ["XLA_FLAGS"] = flags
    add_ragged_exchange_flags()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

_SLURM_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_STEP_ID",
    "SLURM_NTASKS",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_STEP_NODELIST",
    "SLURM_STEP_GPUS",
    "SLURM_JOB_GPUS",
    "CUDA_VISIBLE_DEVICES",
)


def _join() -> None:
    """``initialize()`` with no arguments on the mp arm, before any backend starts."""
    if ARGS.arm == "mp":
        from jaccpot.distributed import multiprocess as mp

        mp.initialize(initialization_timeout=600)


def _smi() -> str:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception as exc:  # diagnostics only
        return f"nvidia-smi failed: {exc!r}"


def _plummer(num: int, seed: int) -> tuple:
    """The test scenarios' Plummer sphere (``_multiprocess_scenarios._plummer``)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests.distributed._multiprocess_scenarios import _plummer as draw

    return draw(num, seed)


def _disc(num: int) -> tuple:
    """``(state, mass, G, softening)``: a prefix of the shuffled disc + bulge IC.

    As ``bench/multigpu_segment_gate.py`` takes it: the first ``num`` rows, their
    masses times ``N_all / num`` (the same galaxy), ``G_code``, and the softening
    formula ``0.5 rdisk / sqrt(num / 1e5)`` unless ``--softening`` is given.
    """
    with np.load(ARGS.disc_ic) as ic:
        n_all = int(ic["state0"].shape[0])
        if num > n_all:
            raise SystemExit(f"--n {num} > the disc IC's {n_all} rows")
        state = np.asarray(ic["state0"][:num], np.float32)
        mass = np.asarray(ic["mass"][:num], np.float64) * (n_all / num)
        g = float(ic["G_code"]) if "G_code" in ic.files else 1.0
        soft = float(0.5 * float(ic["rdisk_code"]) / np.sqrt(num / 1e5))
    if ARGS.softening is not None:
        soft = ARGS.softening
    return state, mass.astype(np.float32), g, soft


def _runner(mesh, g: float = 1.0, softening: float = 1e-3):
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
        solver = FastMultipoleMethod(preset="galaxy", G=g, softening=softening)
        settings = GALAXY_SETTINGS
    else:
        base = FastMultipoleMethod(preset="galaxy").advanced
        solver = FastMultipoleMethod(
            preset="galaxy",
            G=g,
            softening=softening,
            advanced=dataclasses.replace(base, mac_type="dehnen"),
        )
        settings = GalaxySettings(mac_type="dehnen", adaptive_eps=None)
    return ProductionRunner(
        solver,
        settings=settings,
        segment=SegmentConfig(steps=ARGS.steps, diagnostics_every_energy=1),
        mesh=MeshRunConfig(mesh, migrate=not ARGS.no_migrate),
    )


def _write(name: str, value: dict) -> None:
    os.makedirs(ARGS.out, exist_ok=True)
    with open(os.path.join(ARGS.out, name), "w") as fh:
        json.dump(value, fh, indent=1, default=str)


def run() -> None:
    _join()
    import jax

    jax.config.update("jax_enable_x64", True)
    from jaccpot.distributed import multiprocess as mp

    me = jax.process_index()
    mesh = mp.global_mesh()
    ndev = int(np.asarray(mesh.devices).size)
    num = int(ARGS.n) if ARGS.n is not None else int(ARGS.n_per_gpu) * ndev
    if ARGS.ic == "disc" and ARGS.n is None:
        raise SystemExit("--ic disc needs --n (the total particle count)")
    dt = ARGS.dt if ARGS.dt is not None else (5e-4 if ARGS.ic == "disc" else 2e-3)
    record: dict = {
        "arm": ARGS.arm,
        "process": me,
        "processes": jax.process_count(),
        "devices": [str(d) for d in np.asarray(mesh.devices).reshape(-1)],
        "local_devices": [str(d) for d in jax.local_devices()],
        "host": socket.gethostname(),
        "env": {k: os.environ.get(k) for k in _SLURM_KEYS},
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "num_particles": num,
        "ic": ARGS.ic,
        "mac": ARGS.mac,
        "segments": ARGS.segments,
        "steps": ARGS.steps,
        "dt": dt,
        "smi_before": _smi(),
        "runs": [],
    }
    t0 = time.perf_counter()
    if ARGS.ic == "disc":
        state, mass, g, soft = _disc(num)
    else:
        state, mass = _plummer(num, seed=11)
        g, soft = 1.0, (1e-3 if ARGS.softening is None else ARGS.softening)
    record.update(G=g, softening=soft)
    if mp.is_multiprocess():
        lo, hi = mp.local_id_range(num, mesh)
        s, m = mp.ProcessLocalParticles(state[lo:hi], mass[lo:hi], num), None
    else:
        s, m = state, mass
    record["ic_s"] = time.perf_counter() - t0
    del state, mass
    runner = _runner(mesh, g, soft)
    for repeat in range(ARGS.repeats):
        begun = time.perf_counter()
        res = runner.run(s, m, dt=dt, num_steps=ARGS.steps * ARGS.segments)
        wall = time.perf_counter() - begun
        record["runs"].append(
            {
                "wall_s": wall,
                "edges": [
                    {
                        "step": e.step,
                        "segment": e.segment,
                        "retries": e.retries,
                        "wall_s": e.wall_s,
                        "diagnostics": e.diagnostics,
                    }
                    for e in res.edges
                ],
                "retries": [int(r.retries) for r in res.handle.reports],
            }
        )
        if repeat == 0:
            got = dict(res.handle.gather(), state=res.state)
            if me == 0:
                os.makedirs(ARGS.out, exist_ok=True)
                for key in ("state", "acceleration", "owner"):
                    np.save(os.path.join(ARGS.out, f"{key}.npy"), got[key])
            del got
        del res
        print(f"[p{me}] repeat {repeat}: {wall:.2f} s", flush=True)
    record["peak_bytes"] = [
        int((d.memory_stats() or {}).get("peak_bytes_in_use", 0))
        for d in jax.local_devices()
    ]
    record["smi_after"] = _smi()
    _write(f"result_p{me}.json", record)
    if mp.is_multiprocess():
        jax.distributed.shutdown()


def collectives() -> None:
    _join()
    import jax

    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests.distributed._multiprocess_scenarios import collectives as probe

    result = probe(int(jax.device_count()))
    if jax.process_index() == 0:
        arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        os.makedirs(ARGS.out, exist_ok=True)
        np.savez(os.path.join(ARGS.out, "collectives.npz"), **arrays)
        _write(
            "collectives.json",
            {k: v for k, v in result.items() if not isinstance(v, np.ndarray)},
        )
    if jax.process_count() > 1:
        jax.distributed.shutdown()


def compare() -> None:
    out: dict = {"a": ARGS.a, "b": ARGS.b, "arrays": {}}
    for key in ("state", "acceleration", "owner"):
        pa, pb = (os.path.join(d, f"{key}.npy") for d in (ARGS.a, ARGS.b))
        if not (os.path.exists(pa) and os.path.exists(pb)):
            out["arrays"][key] = "missing"
            continue
        x, y = np.load(pa), np.load(pb)
        if x.shape != y.shape or x.dtype != y.dtype:
            out["arrays"][key] = f"shape/dtype {x.shape}{x.dtype} vs {y.shape}{y.dtype}"
            continue
        width = x.dtype.itemsize
        xb = np.ascontiguousarray(x).reshape(-1).view(np.uint8).reshape(-1, width)
        yb = np.ascontiguousarray(y).reshape(-1).view(np.uint8).reshape(-1, width)
        out["arrays"][key] = {
            "differ": int(np.any(xb != yb, axis=1).sum()),
            "of": int(x.size),
            "max_abs": (
                float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                if x.size
                else 0.0
            ),
        }
    edges = []
    for d in (ARGS.a, ARGS.b):
        path = os.path.join(d, "result_p0.json")
        with open(path) as fh:
            first = json.load(fh)["runs"][0]
        edges.append(
            [(e["step"], e["retries"], e["diagnostics"]) for e in first["edges"]]
        )
    out["edges_equal"] = edges[0] == edges[1]
    out["bitwise"] = out["edges_equal"] and all(
        isinstance(v, dict) and v["differ"] == 0 for v in out["arrays"].values()
    )
    text = json.dumps(out, indent=1, default=str)
    print(text)
    if ARGS.json:
        with open(ARGS.json, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    {"run": run, "collectives": collectives, "compare": compare}[ARGS.cmd]()
