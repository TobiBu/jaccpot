"""Hybrid Hermite session 4 on a node: the clusters' internal sum and the clusters on the mesh.

One row per process (HoreKa jobs "hh4a" and "hh4c"; ``bench/slurm/hh/hybrid_clusters_runs.sbatch``).
The system is the one HH3b's gate 5, HH4a's item 4 and HH4c's GPU gate used: a Hernquist
galaxy of ``N`` particles (``a = 1``, ``M = 1``, isotropic velocities from the local circular
speed) and ``K`` Plummer clusters of ``M`` stars (``r_c = 0.012``, ``M_c = 1e-4``, Aarseth-Henon-
Henon velocities) on circular orbits at radii 0.3-1.5 in random planes; galaxy softening 0.005,
the stars' ``r_c / 20``; the galaxy ``nornax.adapters.jaccpot_hybrid.galaxy_solver`` (``dehnen_error``;
``--mac geo``: geometric theta 0.6), the star tree geometric theta 0.5; Hermite-4 internal steps
through a blocked direct sum; dt 2e-3.

* ``share`` (HH4a item 4, nornax ``docs/hybrid_hermite.md`` section 15): the O(K M^2) internal
  sum's share of a slow step through ``FusedClustersForce`` on ONE card. Times (median of
  ``--reps`` after a warm-up of at least 1.5 s: idle A100 clocks) the slow force evaluation, the
  internal derivatives, the fixed ``n_sub = 16`` flow, the adaptive flow (``AarsethSteps(eta=0.02,
  max_rung=16, target_block=64)``) and one whole step of each scheme; counts the internal pairs;
  ``share_*`` = the internal part of a step. ``--coupling none`` attributes the slow force's
  cluster-cluster part.
* ``mesh`` (HH4c, nornax ``docs/hybrid_hermite.md`` section 16): the clusters' runner on the first
  ``--cards`` devices of the process. ``--arm card``: ``ClustersRunnerScheme`` on one card (the
  single-card reference); ``--arm mesh``: ``MeshClustersRunnerScheme`` through
  ``ProductionRunner(mesh=)`` (one process, a 1-D mesh over the cards). ``--segments`` x ``--steps``
  steps with stripping (``StrippingPolicy()`` defaults), cross + coupling fmm, ``--reserve`` free
  galaxy rows (per device on the mesh). Records every edge (wall since the start, stripped stars,
  retries, re-plans, owners, rows per device, energy and momentum drift), the start's wall (the
  force's prepare and the start force), the first segment's wall (its compiles), the later segments'
  wall per step (the steady step) and every device's peak.

Shared cards give no meaningful timings: this is for dedicated nodes.

usage: hybrid_clusters_runs.py share --n 1e6 --k 1 --m 1e5 --coupling fmm --out row.json
       hybrid_clusters_runs.py mesh --arm mesh --cards 4 --n 4e7 --k 10 --m 1e4 --out row.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ragged_exchange_flags import add_ragged_exchange_flags  # noqa: E402


def _allow_py_spy() -> None:
    """Let py-spy read this process's stack (``ptrace_scope`` 1): read a stuck rank before killing it."""
    try:
        import ctypes

        ctypes.CDLL(None).prctl(
            0x59616D61, ctypes.c_ulong(-1), 0, 0, 0
        )  # PR_SET_PTRACER, ANY
    except Exception:  # noqa: BLE001 - not Linux, or no prctl: nothing to allow
        pass


_allow_py_spy()


MC, RC, EPS_G, DT = 1e-4, 0.012, 0.005, 2e-3
EPS_C = RC / 20


def _hernquist(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = rng.uniform(1e-5, 1.0, n)
    r = np.minimum(np.sqrt(u) / (1 - np.sqrt(u)), 20.0)
    d = rng.normal(size=(n, 3))
    x = r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    s = np.sqrt(r) / (1 + r) / np.sqrt(3.0)
    return x, rng.normal(size=(n, 3)) * s[:, None]


def _plummer(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = rng.uniform(1e-4, 0.99, n)
    r = RC / np.sqrt(u ** (-2 / 3) - 1)
    d = rng.normal(size=(n, 3))
    x = r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    q = np.empty(0)
    while q.size < n:  # Aarseth-Henon-Henon: q^2 (1 - q^2)^3.5 by rejection
        x1, x2 = rng.uniform(0, 1, (2, 2 * n))
        q = np.concatenate([q, x1[x2 * 0.1 < x1 * x1 * (1 - x1 * x1) ** 3.5]])
    vm = q[:n] * np.sqrt(2 * MC / np.sqrt(r * r + RC * RC))
    d = rng.normal(size=(n, 3))
    v = vm[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    return x - x.mean(0), v - v.mean(0)


def _system(n: int, k: int, m: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    xg, vg = _hernquist(n, rng)
    mg = np.full(n, 1.0 / n)
    xc, vc = np.zeros((k, m, 3)), np.zeros((k, m, 3))
    mc = np.full((k, m), MC / m)
    for i in range(k):
        x, v = _plummer(m, rng)
        r = rng.uniform(0.3, 1.5)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        t = np.cross(d, rng.normal(size=3))
        t /= np.linalg.norm(t)
        xc[i] = x + r * d
        vc[i] = v + np.sqrt(r / (1 + r) ** 2) * t
    return xg, vg, mg, xc, vc, mc


def _solvers(a: argparse.Namespace) -> tuple:
    from nornax.adapters.jaccpot_hybrid import galaxy_solver

    from jaccpot import FastMultipoleMethod

    def adv():  # the preset's advanced config at this leaf size and expansion order
        base = FastMultipoleMethod(preset="galaxy").advanced
        return dataclasses.replace(
            base,
            tree=dataclasses.replace(base.tree, leaf_target=a.leaf),
            runtime=dataclasses.replace(base.runtime, fixed_order=a.order),
        )

    crit = {} if a.mac == "err" else dict(mac_type="dehnen", theta=0.6)
    gal = galaxy_solver(
        softening=EPS_G, G=1.0, precision="fp32", advanced=adv(), **crit
    )
    clu = galaxy_solver(
        mac_type="dehnen", theta=0.5, softening=EPS_C, G=1.0, precision="fp32",
        advanced=adv(),
    )  # fmt: skip
    return gal, clu


def _peaks(devices) -> list:
    out = []
    for d in devices:
        try:
            out.append((d.memory_stats() or {}).get("peak_bytes_in_use"))
        except Exception as exc:  # noqa: BLE001 - the CPU has no stats
            out.append(repr(exc))
    return out


def _timed(fn, args, reps: int) -> tuple[float, object]:
    import jax

    out = jax.block_until_ready(fn(*args))
    t_end = time.time() + 1.5  # warm the clocks (an idle A100 runs at 765 MHz)
    while time.time() < t_end:
        jax.block_until_ready(fn(*args))
    walls = []
    for _ in range(reps):
        t = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        walls.append(time.perf_counter() - t)
    return float(np.median(walls)), out


def share(a: argparse.Namespace, row: dict) -> None:
    """HH4a item 4: the internal sum's share of a slow step (one card)."""
    import jax
    import jax.numpy as jnp
    from nornax.adapters.jaccpot_hybrid import FusedClustersForce
    from nornax.forces.direct import DirectSumGravity
    from nornax.hybrid import MembershipSplit
    from nornax.hybrid_clusters import ClusterHybridHermite
    from nornax.hybrid_steps import AarsethSteps

    n, k, m = int(a.n), a.k, int(a.m)
    xg, vg, mg, xc, vc, mc = _system(n, k, m, a.seed)
    f32 = jnp.float32
    xg, vg, mg = (jnp.asarray(v, f32) for v in (xg, vg, mg))
    xc, vc, mc = (jnp.asarray(v) for v in (xc, vc, mc))
    gal, clu = _solvers(a)
    t0 = time.perf_counter()
    force, aux0 = FusedClustersForce.prepare(
        gal, clu, xg, mg, xc, mc, leaf_size=a.leaf, max_order=a.order,
        cluster_leaf_size=a.cluster_leaf, reserve=a.reserve, cross="fmm",
        coupling=a.coupling,
    )  # fmt: skip
    row["prepare_s"] = time.perf_counter() - t0
    split = MembershipSplit(
        DirectSumGravity(G=1.0, softening=EPS_C, block_size=min(m, 1024))
    )
    fixed = ClusterHybridHermite(n_sub=16, order=4, split=split, cluster_batch=1)
    steps = AarsethSteps(eta=0.02, max_rung=16, target_block=64)
    ad = ClusterHybridHermite(order=4, split=split, cluster_batch=1, steps=steps)
    start = jax.jit(
        lambda au: fixed.initial_state(
            force, galaxy=(xg, vg, mg), clusters=(xc, vc, mc), aux=au, reserve=a.reserve
        )
    )(aux0)
    jax.block_until_ready(start)
    progs = {
        "slow": (
            jax.jit(
                lambda s: fixed._slow(
                    force, s.galaxy_positions, s.galaxy_masses, s.n_live, s.clusters,
                    s.aux,
                )[:2]
            ),
            start,
        ),  # fmt: skip
        "derivs": (jax.jit(fixed.internal_derivatives), start.clusters),
        "flow_fixed16": (jax.jit(lambda c: fixed.fast_flow(c, DT)), start.clusters),
        "flow_adaptive": (
            jax.jit(lambda c: ad.fast_flow_stats(c, DT)),
            start.clusters,
        ),
        "step_fixed16": (jax.jit(lambda s: fixed.step(s, DT, force)), start),
        "step_adaptive": (jax.jit(lambda s: ad.step(s, DT, force)), start),
    }
    stats = None
    for name, (fn, arg) in progs.items():
        row[f"t_{name}_s"], out = _timed(fn, (arg,), a.reps)
        if name == "flow_adaptive":
            stats = out[1]
        print(f"  {name}: {row[f't_{name}_s']:.4f} s", flush=True)
    row["adaptive"] = {
        key: np.asarray(v).tolist() for key, v in stats._asdict().items()
    }
    # 16 substeps + the derivatives after the opening kick; the adaptive start evaluates through crackle
    row["pairs_fixed16_per_step"] = int(k * 17 * m * m)
    row["pairs_adaptive_per_step"] = int(np.sum(np.asarray(stats.pairs))) + k * m * m
    row["share_fixed16"] = (row["t_flow_fixed16_s"] + row["t_derivs_s"]) / row[
        "t_step_fixed16_s"
    ]
    row["share_adaptive"] = row["t_flow_adaptive_s"] / row["t_step_adaptive_s"]
    row["peak_bytes"] = _peaks(jax.devices()[:1])


def mesh(a: argparse.Namespace, row: dict) -> None:
    """HH4c: the clusters' runner on one card or on a mesh of ``--cards`` cards."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from nornax import hybrid_mesh as hm
    from nornax.adapters.jaccpot_mesh import MeshClustersRunnerScheme
    from nornax.adapters.jaccpot_runner import ClustersRunnerScheme
    from nornax.forces.direct import DirectSumGravity
    from nornax.hybrid import MembershipSplit
    from nornax.hybrid_clusters import ClusterHybridHermite, StrippingPolicy

    from jaccpot.production import GALAXY_SETTINGS, ProductionRunner, SegmentConfig

    devices = jax.devices()
    if a.cards > len(devices):
        raise SystemExit(f"--cards {a.cards}: only {len(devices)} devices visible")
    devices = devices[: a.cards]
    if a.arm == "card" and a.cards != 1:
        raise SystemExit("--arm card runs on one card (--cards 1)")
    n, k, m = int(a.n), a.k, int(a.m)
    xg, vg, mg, xc, vc, mc = _system(n, k, m, a.seed)
    gal, clu = _solvers(a)
    scheme = ClusterHybridHermite(
        n_sub=16,
        order=4,
        split=MembershipSplit(
            DirectSumGravity(G=1.0, softening=EPS_C, block_size=min(m, 1024))
        ),
        cluster_batch=1,
    )
    cl = (jnp.asarray(xc), jnp.asarray(vc), jnp.asarray(mc))
    opts = dict(cross="fmm", coupling=a.coupling, cluster_leaf_size=a.cluster_leaf)
    settings = dataclasses.replace(GALAXY_SETTINGS, leaf_size=a.leaf, max_order=a.order)
    edges: list[dict] = []
    t0 = time.perf_counter()

    def on_edge(e) -> None:
        d = e.diagnostics or {}
        r = {
            key: d.get(key)
            for key in (
                "n_live", "n_stripped", "strip_transfers", "energy_drift",
                "energy_ledgered_drift", "momentum_drift",
            )
        }  # fmt: skip
        r.update(
            segment=e.segment,
            step=e.step,
            retries=e.retries,
            replans=e.plan.get("replans"),
            owners=e.plan.get("owners"),
            n_live_per_device=e.plan.get("n_live_per_device"),
            wall_s=time.perf_counter() - t0,
        )
        edges.append(r)
        print(json.dumps(r, default=float), flush=True)

    seg = SegmentConfig(
        steps=a.steps,
        diagnostics_every=1,
        diagnostics_every_energy=1,
        domain_guard="off",
    )
    policy = StrippingPolicy()
    if a.arm == "card":
        sch = ClustersRunnerScheme(
            scheme,
            cl,
            reserve=a.reserve,
            policy=policy,
            cluster_solver=clu,
            options=opts,
        )
        runner = ProductionRunner(
            gal, settings=settings, segment=seg, scheme=sch, callbacks=[on_edge]
        )
    else:
        slots = a.slots or None
        sch = MeshClustersRunnerScheme(
            scheme, cl, reserve=a.reserve, policy=policy, cluster_solver=clu,
            options=opts, slots=slots,
        )  # fmt: skip
        runner = ProductionRunner(
            gal,
            settings=settings,
            segment=seg,
            scheme=sch,
            callbacks=[on_edge],
            mesh=Mesh(np.array(devices), (hm.AXIS_NAME,)),
        )
    f32 = jnp.float32
    state = jnp.stack([jnp.asarray(xg, f32), jnp.asarray(vg, f32)], 1)
    del xg, vg
    res = runner.run(state, jnp.asarray(mg, f32), dt=DT, num_steps=a.segments * a.steps)
    jax.block_until_ready(res.state)
    row["wall_s"] = time.perf_counter() - t0
    row["edges"] = edges
    row["mode"] = getattr(res, "mode", None)
    # the edge at step 0 is the start (the force's prepare, the start force); then one edge per segment
    start = [e["wall_s"] for e in edges if e["step"] == 0]
    walls = [e["wall_s"] for e in edges if e["step"] > 0]
    row["start_wall_s"] = start[0] if start else None
    seg_walls = [b - w for w, b in zip(([start[0]] if start else [0.0]) + walls, walls)]
    row["segment_wall_s"] = seg_walls
    # the first segment compiles its programs; the later ones are the steady step
    row["first_segment_wall_s"] = seg_walls[0] if seg_walls else None
    later = seg_walls[1:]
    row["steady_step_s"] = float(np.median(later)) / a.steps if later else None
    row["steady_step_s_min"] = float(min(later)) / a.steps if later else None
    row["n_stripped_end"] = edges[-1]["n_stripped"] if edges else None
    row["peak_bytes"] = _peaks(devices)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["share", "mesh"])
    ap.add_argument("--n", type=float, default=1e6, help="galaxy particles (all cards)")
    ap.add_argument("--k", type=int, default=1, help="clusters")
    ap.add_argument("--m", type=float, default=1e3, help="stars per cluster")
    ap.add_argument("--coupling", choices=["fmm", "direct", "none"], default="fmm")
    ap.add_argument("--mac", choices=["err", "geo"], default="err")
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--cluster-leaf", type=int, default=64)
    ap.add_argument(
        "--reserve",
        type=int,
        default=None,
        help="free galaxy rows for stripped stars (share: 1024; mesh: 4096 per device)",
    )
    ap.add_argument(
        "--reps", type=int, default=3, help="share: timed calls per program"
    )
    ap.add_argument("--arm", choices=["card", "mesh"], default="mesh")
    ap.add_argument(
        "--cards", type=int, default=1, help="mesh: the first CARDS devices"
    )
    ap.add_argument(
        "--slots", type=int, default=0, help="mesh: cluster slots per device (0: K)"
    )
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5, help="mesh: steps per segment")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.reserve is None:
        a.reserve = 1024 if a.what == "share" else 4096
    add_ragged_exchange_flags()  # before jax starts: the mesh's ragged exchanges on NCCL

    import jax

    jax.config.update("jax_enable_x64", True)
    row = dict(vars(a), jax=jax.__version__, devices=[str(d) for d in jax.devices()])
    row["cuda_visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    row["xla_flags"] = os.environ.get("XLA_FLAGS")
    t0 = time.perf_counter()
    (share if a.what == "share" else mesh)(a, row)
    row["total_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(row, fh, indent=1, default=lambda o: np.asarray(o).tolist())
    keys = (
        ("share_fixed16", "share_adaptive", "t_step_fixed16_s", "t_slow_s")
        if a.what == "share"
        else ("first_segment_wall_s", "steady_step_s", "n_stripped_end")
    )
    print(
        json.dumps(
            {
                key: row.get(key)
                for key in ("what", "arm", "cards", "n", "k", "m") + keys
            }
        )
    )
    print("peak bytes", row.get("peak_bytes"), flush=True)


if __name__ == "__main__":
    main()
