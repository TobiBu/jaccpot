"""The escaper path on the mesh (wave 3, MESC): step time and peak memory per card.

The FULL unclipped 1e8 Sgr IC (100,000,001 rows, a 45.7 Mpc box) through
``ProductionRunner(mesh=MeshRunConfig(...), escapers=EscaperPolicy(r_escape))`` as
N processes x 1 GPU (``--arm mp``, launched by ``srun``; each process reads the IC
and keeps its id range, ``ProcessLocalParticles``) or one process driving every
visible GPU (``--arm sp``). Rows:

``esc``   escapers ON (``r_escape`` 80 = 800 kpc): ~1.8e5 escapers in the tail;
``off``   escapers OFF on the IC with every particle beyond ``--clip`` pulled
          radially onto it (same N): the control.

``--segments`` segments of ``--steps`` steps at dt 1e-3, softening 1.5e-3, the
energy every ``--energy-every`` edges, migration on (the mesh default). The first
segment holds the compiles: ``step_s`` is the median of the later segments'
``wall_s / steps``; ``edge_s`` the wall time outside the segments (diagnostics,
energies, re-splits and their re-cuts) per edge after the first. Every process
writes ``result_p<rank>.json`` (its cards' peak bytes, the walls, the edges'
diagnostics with the escaper fields, the split).

    srun --ntasks=4 --ntasks-per-node=4 python bench/mesh_escaper_timing.py \\
        --arm mp --row esc --ic sgr_light_1e8_noclip.npz --out OUT/esc_dehnen
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ragged_exchange_flags import add_ragged_exchange_flags  # noqa: E402

_FUSION = "--xla_gpu_experimental_enable_fusion_autotuner=false"


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", choices=("sp", "mp"), required=True)
    p.add_argument("--row", choices=("esc", "off"), required=True)
    p.add_argument("--ic", required=True, help="an .npz with state0 [N,2,3] and mass")
    p.add_argument("--out", required=True)
    p.add_argument("--mac", choices=("dehnen", "dehnen_error"), default="dehnen")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--segments", type=int, default=6)
    p.add_argument("--r-escape", type=float, default=80.0, help="code units (10 kpc)")
    p.add_argument("--clip", type=float, default=80.0)
    p.add_argument("--energy-every", type=int, default=1)
    return p.parse_args()


ARGS = _args()
flags = os.environ.get("XLA_FLAGS", "")
if _FUSION.split("=")[0] not in flags:
    flags = f"{flags} {_FUSION}".strip()
os.environ["XLA_FLAGS"] = flags
add_ragged_exchange_flags()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402


def main() -> None:
    a = ARGS
    if a.arm == "mp":
        from jaccpot.distributed import multiprocess as mp

        mp.initialize(initialization_timeout=600)
    import jax

    from jaccpot import FastMultipoleMethod
    from jaccpot.distributed import multiprocess as mp
    from jaccpot.escapers import EscaperPolicy
    from jaccpot.production import (
        GalaxySettings,
        MeshRunConfig,
        ProductionRunner,
        SegmentConfig,
    )

    me = int(jax.process_index())
    os.makedirs(a.out, exist_ok=True)
    t_load = time.perf_counter()
    z = np.load(a.ic)
    state = np.asarray(z["state0"], np.float32)
    mass = np.asarray(z["mass"], np.float32)
    del z
    num = int(state.shape[0])
    record: dict = {"args": vars(a), "n": num, "process": me}
    if a.row == "off":
        r = np.linalg.norm(state[:, 0].astype(np.float64), axis=1)
        far = r > a.clip
        state[far, 0] = (state[far, 0] * (a.clip / r[far])[:, None]).astype(np.float32)
        record["n_clipped"] = int(far.sum())
    mesh = mp.global_mesh("gpus")
    if mp.is_multiprocess():
        lo, hi = mp.local_id_range(num, mesh)
        given: tuple = (mp.ProcessLocalParticles(state[lo:hi], mass[lo:hi], num), None)
    else:
        given = (state, mass)
    del state, mass
    record["load_s"] = time.perf_counter() - t_load
    base = FastMultipoleMethod(preset="galaxy").advanced
    if a.mac == "dehnen":
        settings = GalaxySettings(mac_type="dehnen", adaptive_eps=None, theta=0.7)
        solver = FastMultipoleMethod(
            preset="galaxy",
            softening=1.5e-3,
            theta=0.7,
            advanced=dataclasses.replace(base, mac_type="dehnen"),
        )
    else:
        settings = GalaxySettings()
        solver = FastMultipoleMethod(preset="galaxy", softening=1.5e-3)
    runner = ProductionRunner(
        solver,
        settings=settings,
        segment=SegmentConfig(steps=a.steps, diagnostics_every_energy=a.energy_every),
        escapers=EscaperPolicy(r_escape=a.r_escape) if a.row == "esc" else None,
        mesh=MeshRunConfig(mesh),
    )
    t0 = time.perf_counter()
    res = runner.run(*given, dt=1e-3, num_steps=a.steps * a.segments)
    record["wall_s"] = time.perf_counter() - t0
    seg = [e.wall_s for e in res.edges[1:]]
    record["segment_wall_s"] = seg
    record["step_s"] = statistics.median(seg[1:]) / a.steps if len(seg) > 1 else None
    record["edge_s"] = (record["wall_s"] - sum(seg)) / max(1, len(seg)) if seg else None
    record["edges"] = [
        {
            "step": e.step,
            "retries": e.retries,
            "wall_s": e.wall_s,
            "counts": e.plan["mesh"]["counts"],
            "escapers": e.plan.get("escapers"),
            "diag": e.diagnostics,
        }
        for e in res.edges
    ]
    record["escapers"] = res.escapers
    record["ndev"] = int(np.asarray(mesh.devices).size)
    record["peak_bytes"] = {
        str(d.id): int((d.memory_stats() or {}).get("peak_bytes_in_use", 0))
        for d in jax.local_devices()
    }
    per_card = num / record["ndev"]
    record["peak_bytes_per_particle"] = {
        k: v / per_card for k, v in record["peak_bytes"].items()
    }
    record["device"] = str(jax.devices()[0].device_kind)
    record["env"] = {
        k: os.environ.get(k)
        for k in (
            "SLURM_JOB_ID",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "CUDA_VISIBLE_DEVICES",
        )
    }
    with open(os.path.join(a.out, f"result_p{me}.json"), "w") as fh:
        json.dump(record, fh, indent=1, default=str)
    print(
        f"[done] p{me} {a.row}/{a.mac}: step {record['step_s']} s, edge "
        f"{record['edge_s']} s, peak B/p per card {record['peak_bytes_per_particle']}",
        flush=True,
    )
    if mp.is_multiprocess():
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
