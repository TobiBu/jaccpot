"""M4 on HoreKa: the LET exchange vs all_gather and work cuts vs count cuts, at scale.

M4 (``docs/production_2026-11.md`` ``## M4``) made two opt-in switches: the LET
exchange of the cross summaries (``JACCPOT_CROSS_EXCHANGE=let``: each peer gets only
the part of a device's summary tree its cross walk opens; bitwise to all_gather) and
work-balanced rebalance cuts (``JACCPOT_MESH_CUTS=work``: equal near + far work per
device instead of equal counts). What the local cards could not measure: walls and
bytes on dedicated A100s with NVLink and InfiniBand, at 4 and 8 GPUs.

Per run (one arm = one exchange x one cut mode): the ``ProductionRunner`` mesh mode
(migration on, a repartition before every segment after the first), ``--segments``
x ``--steps`` steps; per segment edge its wall, the live rows per device, the work
per device before and after the rebalance (work cuts: the runner's estimate), the
LET's send / receive rows (the needs, max over the mesh) next to the all_gather's
``(ndev - 1) x max_summary_nodes``, the overflow bits and retries; momentum split
(total / local / cross) at the start and the end, as ``mesh_cross_momentum.py``;
each card's peak. Process 0 writes ``result.json``.

    srun --ntasks=4 --ntasks-per-node=4 python bench/m4_exchange_cuts.py \\
        --arm mp --exchange let --cuts work --n-per-gpu 2.5e6 --out OUT/let_work
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
    p.add_argument("--exchange", choices=("all_gather", "let"), default="all_gather")
    p.add_argument("--cuts", choices=("count", "work"), default="count")
    return p.parse_args()


ARGS = _args()
# the two switches, read when the runner builds its hook and its rebalance
os.environ["JACCPOT_CROSS_EXCHANGE"] = ARGS.exchange
os.environ["JACCPOT_MESH_CUTS"] = ARGS.cuts
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


def _reports(handle) -> list:
    """Per segment: counts, work before / after the rebalance, the LET rows (needs,
    max over the mesh), the all_gather's rows, bits and retries."""
    from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS

    if handle is None:
        return []
    caps = getattr(handle.force, "cross_caps", None)
    ndev = int(handle.ndev)
    gathered_rows = None if caps is None else (ndev - 1) * int(caps.max_summary_nodes)
    out = []
    for r in getattr(handle, "reports", []):
        cross = dict(zip(CROSS_NEEDS_FIELDS, list(r.needs)[-len(CROSS_NEEDS_FIELDS) :]))
        work = getattr(r, "work", []) or []
        out.append(
            {
                "segment": r.segment,
                "steps": r.steps,
                "retries": r.retries,
                "bits": r.bits,
                "counts": r.counts,
                "work_before": work[0] if work else None,
                "work_after": work[1] if work else None,
                "imbalance_before": (
                    max(work[0]) / (sum(work[0]) / len(work[0])) if work else None
                ),
                "imbalance_after": (
                    max(work[1]) / (sum(work[1]) / len(work[1])) if work else None
                ),
                "let_send_rows_max": cross.get("let_send_cap"),
                "let_recv_rows_max": cross.get("let_recv_cap"),
                "all_gather_rows": gathered_rows,
                "summary_nodes_max": cross.get("max_summary_nodes"),
            }
        )
    return out


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
        "exchange": ARGS.exchange,
        "cuts": ARGS.cuts,
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
    # the continued run's runner holds every segment's report (the first run's
    # handle moved into it)
    rec["segments_report"] = _reports(res.handle or first.handle)
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
