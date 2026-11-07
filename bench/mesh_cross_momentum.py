"""M1 on HoreKa: momentum of the multi-GPU runner at scale, split into its local and cross parts.

The local half of the record (``docs/production_2026-11.md`` ``## M1``): every cross
pair is evaluated on both of its devices from bitwise-identical shipped data, as one
card evaluates its own pairs once per direction, so the cross part of ``sum m a``
cancels as well as the local part does (fp64 on forced CPU devices: both at 1e-17
relative; fp32 on two A100s: both ~4e-9). This job checks that at production scale
(the ``galaxy`` preset, 1e7-1e8 particles, 4 and 8 GPUs as N processes x 1 GPU, one
process per GPU over NCCL), where the cross share of the force is smaller than in the
local toy runs and the ranks sit on different nodes.

Per run: the ``ProductionRunner`` mesh mode (migration on, a repartition before every
segment after the first), ``--segments`` x ``--steps`` steps at ``--dt``, momentum
and angular momentum at every edge (``momentum_drift = |P - P0| / sum m |v|_0``),
and at the start and at the end the force with and without the cross hook on the
same state: ``|sum m a| / sum m |a|`` of the total, of the local part and of the
cross part alone, and the cross part's share of ``sum m |a|``. Every number is a
mesh-wide fold (each device's fp64 row all-gathered, folded in mesh order), so
every process reads the same values. Process 0 writes ``result.json``; walls per
edge are in it (a dedicated node: they count).

Launch as ``bench/multiprocess_gate.py`` (``bench/slurm/m1/``):

    srun --ntasks=4 --ntasks-per-node=4 python bench/mesh_cross_momentum.py \\
        --arm mp --n-per-gpu 2.5e6 --out OUT/mp4
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ragged_exchange_flags import add_ragged_exchange_flags  # noqa: E402

_FUSION = "--xla_gpu_experimental_enable_fusion_autotuner=false"


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", choices=("sp", "mp"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-per-gpu", type=float, default=2.5e6)
    p.add_argument("--mac", choices=("dehnen_error", "dehnen"), default="dehnen_error")
    p.add_argument("--ic", choices=("plummer", "disc"), default="disc")
    p.add_argument("--segments", type=int, default=10)
    p.add_argument("--steps", type=int, default=100, help="steps per segment")
    p.add_argument("--dt", type=float, default=1e-3)
    return p.parse_args()


ARGS = _args()
_flags = os.environ.get("XLA_FLAGS", "")
if _FUSION.split("=")[0] not in _flags:
    os.environ["XLA_FLAGS"] = f"{_flags} {_FUSION}".strip()
add_ragged_exchange_flags()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402


def _ic(num: int, seed: int) -> tuple:
    """``[N, 2, 3]`` state and masses: the test scenarios' Plummer sphere, or a
    Plummer bulge (half the mass, a = 0.3) and an exponential disc (R_d 1, z_d 0.1,
    circular speeds of the enclosed mass), centred, total mass 1."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    if ARGS.ic == "plummer":
        from tests.distributed._multiprocess_scenarios import _plummer

        return _plummer(num, seed)
    rng = np.random.default_rng(seed)
    nb, a = num // 2, 0.3
    nd = num - nb
    x = rng.uniform(0.0, 1.0, nb)
    r = np.minimum(a / np.sqrt(x ** (-2.0 / 3.0) - 1.0), 10 * a)
    d = rng.normal(size=(nb, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    sig = np.sqrt(0.5 / 6.0 / np.sqrt(r**2 + a * a))
    R = np.minimum(rng.gamma(2.0, 1.0, nd), 6.0)
    ph = rng.uniform(0, 2 * np.pi, nd)
    menc = 0.5 * R**3 / (R**2 + a * a) ** 1.5 + 0.5 * (1 - np.exp(-R) * (1 + R))
    vc = np.sqrt(menc / np.sqrt(R * R + 0.01))
    pos = np.concatenate(
        [
            r[:, None] * d,
            np.stack([R * np.cos(ph), R * np.sin(ph), 0.1 * rng.normal(size=nd)], 1),
        ]
    )
    vel = np.concatenate(
        [
            sig[:, None] * rng.normal(size=(nb, 3)),
            np.stack([-vc * np.sin(ph), vc * np.cos(ph), np.zeros(nd)], 1)
            + 0.05 * rng.normal(size=(nd, 3)),
        ]
    )
    m = np.full(num, 1.0 / num)
    pos -= np.sum(m[:, None] * pos, 0)
    vel -= np.sum(m[:, None] * vel, 0)
    return np.stack([pos, vel], 1), m


def _runner(mesh):
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
        solver = FastMultipoleMethod(preset="galaxy", G=1.0, softening=1e-3)
        settings = GALAXY_SETTINGS
    else:
        base = FastMultipoleMethod(preset="galaxy").advanced
        solver = FastMultipoleMethod(
            preset="galaxy",
            G=1.0,
            softening=1e-3,
            advanced=dataclasses.replace(base, mac_type="dehnen"),
        )
        settings = GalaxySettings(mac_type="dehnen", adaptive_eps=None)
    return ProductionRunner(
        solver,
        settings=settings,
        segment=SegmentConfig(steps=ARGS.steps, diagnostics_every=ARGS.steps),
        mesh=MeshRunConfig(mesh, migrate=True),
    )


def split(handle) -> dict:
    """``|sum m a| / sum m |a|`` of the force with and without the cross hook at the
    handle's state: total, local part, cross part alone; the cross share."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    force, st = handle.force, handle.state
    axis = handle.axis_name
    carry = st.force_scale is not None
    keep = force.hook, force.near_sink

    def program(with_cross: bool):
        def body(state, masses, count, ids, *fb):
            cap = int(state.shape[0])
            live = jnp.minimum(count[0], cap)
            if not with_cross:
                force.hook, force.near_sink = None, None
            try:
                out = force.shard(state[:, 0, :], masses, live, ids, *fb)
            finally:
                force.hook, force.near_sink = keep
            rows = (jnp.arange(cap) < live)[:, None]
            acc = jnp.where(rows, out[0].astype(jnp.float64), 0.0)
            return acc

        return body

    def rows_of(acc_t, acc_l, masses):
        m = masses.astype(jnp.float64)[:, None]
        acc_c = acc_t - acc_l
        row = jnp.concatenate(
            [
                jnp.sum(m * acc_t, 0),
                jnp.sum(m * acc_l, 0),
                jnp.sum(m * acc_c, 0),
                jnp.stack(
                    [
                        jnp.sum(m[:, 0] * jnp.linalg.norm(a, axis=1))
                        for a in (acc_t, acc_l, acc_c)
                    ]
                ),
            ]
        )
        return jax.lax.all_gather(row, axis)

    def body(state, masses, count, ids, *fb):
        acc_t = program(True)(state, masses, count, ids, *fb)
        force.before_trace()
        acc_l = program(False)(state, masses, count, ids, *fb)
        return rows_of(acc_t, acc_l, masses)

    spec = P(axis)
    args = (st.state, st.masses, st.count, st.ids) + (
        (st.force_scale,) if carry else ()
    )
    fn = jax.jit(
        jax.shard_map(
            body,
            mesh=handle.mesh,
            in_specs=(spec,) * len(args),
            out_specs=P(),
            check_vma=False,
        )
    )
    force.before_trace()
    with force.context():
        rows = np.asarray(fn(*args), np.float64)  # (ndev, 12), replicated
    folded = rows[0].copy()
    for r in rows[1:]:
        folded = folded + r  # mesh order
    tot, loc, cro = folded[0:3], folded[3:6], folded[6:9]
    s_t, s_l, s_c = folded[9:12]
    return {
        "total": float(np.linalg.norm(tot) / s_t),
        "local_part": float(np.linalg.norm(loc) / s_l),
        "cross_part": float(np.linalg.norm(cro) / max(s_c, 1e-300)),
        "cross_share": float(s_c / s_t),
        "per_device_cross": rows[:, 6:9].tolist(),
    }


def main() -> None:
    if ARGS.arm == "mp":
        from jaccpot.distributed import multiprocess as mp

        mp.initialize(initialization_timeout=600)
    import jax

    jax.config.update("jax_enable_x64", True)
    from jaccpot.distributed import multiprocess as mp

    me = jax.process_index()
    mesh = mp.global_mesh()
    ndev = int(np.asarray(mesh.devices).size)
    num = int(ARGS.n_per_gpu) * ndev
    rec: dict = {
        "arm": ARGS.arm,
        "processes": jax.process_count(),
        "devices": ndev,
        "host": socket.gethostname(),
        "num_particles": num,
        "mac": ARGS.mac,
        "ic": ARGS.ic,
        "segments": ARGS.segments,
        "steps": ARGS.steps,
        "dt": ARGS.dt,
        "xla_flags": os.environ.get("XLA_FLAGS"),
    }
    state, mass = _ic(num, seed=7)
    if mp.is_multiprocess():
        lo, hi = mp.local_id_range(num, mesh)
        s, m = mp.ProcessLocalParticles(state[lo:hi], mass[lo:hi], num), None
    else:
        s, m = state, mass
    del state, mass
    runner = _runner(mesh)
    t0 = time.perf_counter()
    first = runner.run(s, m, dt=ARGS.dt, num_steps=ARGS.steps)
    rec["split_start"] = split(first.handle)
    total = ARGS.steps * ARGS.segments
    res = runner.run(dt=ARGS.dt, num_steps=total, restart_from=first)
    rec["wall_s"] = time.perf_counter() - t0
    rec["split_end"] = split(res.handle)
    rec["edges"] = [
        {
            "step": e.step,
            "wall_s": e.wall_s,
            "retries": e.retries,
            **{
                k: (e.diagnostics or {}).get(k)
                for k in ("momentum_drift", "angular_momentum_drift", "com_drift")
            },
        }
        for e in (*first.edges, *res.edges)
    ]
    rec["peak_bytes"] = [
        int((d.memory_stats() or {}).get("peak_bytes_in_use", 0))
        for d in jax.local_devices()
    ]
    if me == 0:
        os.makedirs(ARGS.out, exist_ok=True)
        with open(os.path.join(ARGS.out, "result.json"), "w") as fh:
            json.dump(rec, fh, indent=1, default=str)
    print(
        json.dumps({k: rec[k] for k in ("split_start", "split_end")})[:600], flush=True
    )
    if mp.is_multiprocess():
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
