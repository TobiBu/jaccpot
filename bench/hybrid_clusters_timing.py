"""The hybrid clusters' slow step against the galaxy step, one row per process (HoreKa job "hh3").

nornax's ``FusedClustersForce`` (hybrid Hermite session 3: K direct-sum clusters in a
live galaxy, the stars' field at the galaxy from jaccpot's grouped back-reaction) on
ONE card: a Hernquist galaxy of ``N`` particles (``a = 1``, ``M = 1``) and ``K``
Plummer clusters of ``M`` stars (``r_c = 0.012``, ``M_c = 1e-4`` each) at log-uniform
radii 0.3-10 in random directions -- HH3a's gate-4 IC:

* ``--mode galaxy``: the galaxy's own force step through the same force object
  (``cluster_slow_accelerations`` without clusters), the base;
* ``--mode slow``: the whole slow evaluation -- the galaxy step, the galaxy's field
  at all ``K x M`` stars (one target plan), the stars' field at every galaxy row
  (``plan_back_reaction`` with ``SourceGroups``), and the cluster-cluster coupling
  (``--coupling fmm|direct|none``);
* ``--mode back``: the back-reaction alone (the star tree's rebuild + the grouped
  evaluation at all ``N`` galaxy rows);
* ``--mode coupling``: the coupling alone (``coupling="fmm"``: the star tree's field
  at the stars minus each cluster's own).

Each program is compiled once, run ``--warm`` times, then timed ``--reps`` times (min
and median, ms); the row records the plans, the verdicts, the back-reaction's stats
(queue peak, MAC tests, inner points) and the device peak. Shared cards give no
meaningful timings: this is for dedicated nodes.

usage: hybrid_clusters_timing.py --n 1e8 --k 10 --m 1e4 --mode slow --coupling fmm --out row.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import numpy as np


def _hernquist(n: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(1e-5, 1.0, n)
    r = np.minimum(np.sqrt(u) / (1 - np.sqrt(u)), 20.0)
    d = rng.normal(size=(n, 3))
    return (r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]).astype(np.float32)


def _plummer(n: int, a: float, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(1e-4, 0.99, n)
    r = a / np.sqrt(u ** (-2 / 3) - 1)
    d = rng.normal(size=(n, 3))
    x = r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    return x - x.mean(0)


def _ic(n: int, k: int, m: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    xg = _hernquist(n, rng)
    mg = np.full(n, 1.0 / n, np.float32)
    r = np.exp(rng.uniform(np.log(0.3), np.log(10.0), k))
    d = rng.normal(size=(k, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    xc = np.stack([_plummer(m, 0.012, rng) for _ in range(k)]) + (r[:, None] * d)[
        :, None, :
    ]
    mc = np.full((k, m), 1e-4 / m)
    return xg, mg, xc, mc


def _solver(mac: str, *, eps: float, theta: float, leaf: int, order: int, cpu: bool):
    from jaccpot import FastMultipoleMethod

    base = FastMultipoleMethod(preset="galaxy").advanced
    adv = dataclasses.replace(
        base,
        tree=dataclasses.replace(base.tree, leaf_target=leaf),
        runtime=dataclasses.replace(
            base.runtime, fixed_order=order, cpu_fused_lane=cpu
        ),
        mac_type="dehnen_error" if mac == "err" else "dehnen",
    )
    crit = (
        dict(
            adaptive_eps=1e-4,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if mac == "err"
        else {}
    )
    return FastMultipoleMethod(
        preset="galaxy",
        theta=theta,
        G=1.0,
        softening=eps,
        precision="fp32",
        use_pallas=True,
        advanced=adv,
        **crit,
    )


def _peak() -> object:
    import jax

    try:
        return jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    except Exception as exc:  # noqa: BLE001 - the CPU has no stats
        return repr(exc)


def _time(fn, args, warm: int, reps: int) -> tuple[list, object]:
    import jax

    for _ in range(warm):
        jax.block_until_ready(fn(*args))
    walls = []
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        walls.append(1e3 * (time.perf_counter() - t))
    return walls, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=float, default=1e6)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=float, default=1e4)
    ap.add_argument("--mode", choices=["galaxy", "slow", "back", "coupling"])
    ap.add_argument("--coupling", choices=["fmm", "direct", "none"], default="fmm")
    ap.add_argument("--mac", choices=["geo", "err"], default="err")
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--reserve", type=int, default=0)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from nornax.adapters.jaccpot_hybrid import FusedClustersForce

    cpu = jax.default_backend() == "cpu"
    n, k, m = int(a.n), int(a.k), int(a.m)
    eps_g, eps_c = 0.005, 0.012 / 20
    row = dict(vars(a), backend=jax.default_backend(), jax=jax.__version__)
    row["device"] = str(jax.devices()[0])
    xg, mg, xc, mc = _ic(n, k, m, a.seed)
    xg_d, mg_d = jnp.asarray(xg), jnp.asarray(mg)
    xc_d, mc_d = jnp.asarray(xc), jnp.asarray(mc)
    coupling = a.coupling if a.mode != "coupling" else "fmm"
    t0 = time.perf_counter()
    force, aux = FusedClustersForce.prepare(
        _solver(a.mac, eps=eps_g, theta=a.theta, leaf=a.leaf, order=a.order, cpu=cpu),
        _solver("geo", eps=eps_c, theta=0.5, leaf=a.leaf, order=a.order, cpu=cpu),
        xg_d,
        mg_d,
        xc_d,
        mc_d,
        leaf_size=a.leaf,
        max_order=a.order,
        reserve=a.reserve,
        cross="fmm",
        coupling=coupling,
    )
    row["prepare_plan_s"] = time.perf_counter() - t0
    row["plans"] = {
        name: str(getattr(force, name))
        for name in ("stars_plan", "galaxy_plan", "coupling_plan")
    }
    rows = force.rows
    if a.reserve:
        xg_d = jnp.concatenate([xg_d, jnp.zeros((a.reserve, 3), xg_d.dtype)])
        mg_d = jnp.concatenate([mg_d, jnp.zeros((a.reserve,), mg_d.dtype)])
    n_live = jnp.asarray(n, jnp.int32)
    assert int(xg_d.shape[0]) == rows
    if a.mode == "galaxy":

        def run(x, w, nl, au):
            return force.cluster_slow_accelerations(x, w, nl, None, None, aux=au)[0]

        fn = jax.jit(run)
        args = (xg_d, mg_d, n_live, aux)
    elif a.mode == "slow":

        def run(x, w, nl, s, sm, au):
            a_g, a_c, au = force.cluster_slow_accelerations(x, w, nl, s, sm, aux=au)
            return a_g, a_c, au[2], au[3]

        fn = jax.jit(run)
        args = (xg_d, mg_d, n_live, xc_d, mc_d, aux)
    elif a.mode == "back":

        def run(x, nl, s, sm):
            xs, ms = force._star_slots(s, sm, x.dtype)
            ctopo = force.clusters.rebuild(xs, ms)
            return force.clusters.target_field(
                ctopo,
                x,
                force.galaxy_plan,
                softening=force.back_softening,
                **force._back_extra(x, nl, s, sm),
            )

        fn = jax.jit(run)
        args = (xg_d, n_live, xc_d, mc_d)
    else:

        def run(s, sm):
            xs, ms = force._star_slots(s, sm, jnp.float32)
            ctopo = force.clusters.rebuild(xs, ms)
            acc, _, ok, needs = force._coupling_field(xs, ms, ctopo)
            return acc, ok, needs

        fn = jax.jit(run)
        args = (xc_d, mc_d)
    try:
        temp = fn.lower(*args).compile().memory_analysis().temp_size_in_bytes
        row["temp_bytes"] = int(temp)
    except Exception as exc:  # noqa: BLE001 - not every backend reports it
        row["temp_bytes"] = repr(exc)
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    row["first_call_s"] = time.perf_counter() - t0
    walls, out = _time(fn, args, a.warm, a.reps)
    row["ms"] = walls
    row["ms_min"] = min(walls)
    row["ms_median"] = float(np.median(walls))
    if a.mode == "slow":
        row["ok"] = np.asarray(out[2]).tolist()
        row["needs"] = np.asarray(out[3]).tolist()
    elif a.mode == "back":
        row["ok"] = bool(out.ok)
        row["needs"] = np.asarray(out.needs).tolist()
        row["stats"] = np.asarray(out.stats).tolist()
    elif a.mode == "coupling":
        row["ok"] = bool(out[1])
        row["needs"] = np.asarray(out[2]).tolist()
    row["peak_bytes"] = _peak()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(row, fh, indent=1, default=str)
    print(
        json.dumps(
            {
                key: row[key]
                for key in ("mode", "n", "k", "m", "coupling", "ms_min", "peak_bytes")
            }
        )
    )


if __name__ == "__main__":
    main()
