"""``target_points`` cost against the galaxy step, one row per process (HoreKa job "hh").

The hybrid Hermite scheme's two slow-force evaluations next to the galaxy's own
force step, on ONE card, a Hernquist galaxy of ``N`` particles (``a = 1``,
``M = 1``) with a Plummer cluster of ``M`` stars on a circular orbit at ``R = 1``
(``r_c = 0.005``, ``M_c = 1e-4``) -- the IC of the HH GPU runs:

* ``--mode step``: the galaxy provider's ``step`` (rebuild + force), the base;
* ``--mode targets``: ``step_with_targets`` with the ``M`` stars as target points
  (the galaxy -> stars field; ``--far m2p|local``, ``--walk frontier|dense``);
* ``--mode back``: the cluster's field at all ``N`` galaxy points through
  :func:`~jaccpot.runtime.target_back_reaction.evaluate_back_reaction` (the root
  multipole outside ``r_in``, the frontier walk inside), including the cluster's
  rebuild;
* ``--mode back_full``: the same through ``target_points`` over all ``N`` points
  (session 2's comparison; expect it to be the memory-heavy one).

Each program is compiled once, run ``--warm`` times, then timed ``--reps`` times
(min and median, ms); the row records the plan, the walk's stats (queue peak, MAC
tests, inner points), and the device peak. No GPU timing is meaningful on a shared
card: this is for dedicated nodes.

usage: target_points_timing.py --n 1e8 --m 1e5 --mode targets --mac err --out row.json
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


def _ic(n: int, m: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    xg = _hernquist(n, rng)
    mg = np.full(n, 1.0 / n, np.float32)
    xc = (_plummer(m, 0.005, rng) + np.array([1.0, 0.0, 0.0])).astype(np.float32)
    mc = np.full(m, 1e-4 / m, np.float32)
    return xg, mg, xc, mc


def _solver(mac: str, *, eps: float, theta: float, leaf: int, order: int, cpu: bool):
    from jaccpot import FastMultipoleMethod

    base = FastMultipoleMethod(preset="galaxy").advanced
    adv = dataclasses.replace(
        base,
        runtime=dataclasses.replace(base.runtime, fixed_order=order, cpu_fused_lane=cpu),
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
    ap.add_argument("--m", type=float, default=1e4)
    ap.add_argument("--mode", choices=["step", "targets", "back", "back_full"])
    ap.add_argument("--mac", choices=["geo", "err"], default="err")
    ap.add_argument("--far", choices=["m2p", "local"], default="m2p")
    ap.add_argument("--walk", choices=["frontier", "dense"], default="frontier")
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from jaccpot import FusedForceProvider

    cpu = jax.default_backend() == "cpu"
    n, m = int(a.n), int(a.m)
    eps_g, eps_c = 0.005, 0.005 / 20
    row = dict(vars(a), backend=jax.default_backend(), jax=jax.__version__)
    row["device"] = str(jax.devices()[0])
    xg, mg, xc, mc = _ic(n, m, a.seed)
    xg_d, mg_d = jnp.asarray(xg), jnp.asarray(mg)
    xc_d, mc_d = jnp.asarray(xc), jnp.asarray(mc)
    t0 = time.perf_counter()
    if a.mode in ("step", "targets"):
        prov, topo = FusedForceProvider.prepare(
            _solver(a.mac, eps=eps_g, theta=a.theta, leaf=a.leaf, order=a.order, cpu=cpu),
            xg_d,
            mg_d,
            leaf_size=a.leaf,
            max_order=a.order,
        )
        aux = None
        if prov.scale_carry:
            aux = (prov.accelerations(topo, force_scale=True)[1], None, None)
        if a.mode == "step":
            fn = jax.jit(lambda x, w, au: prov.step(x, w, aux=au)[0])
            args = (xg_d, mg_d, aux)
        else:
            from jaccpot.runtime.target_points import (
                bootstrap_target_force_scale,
                plan_target_points,
            )

            plan = plan_target_points(prov, topo, xc_d, far=a.far, walk=a.walk)
            fb = None
            if prov.scale_carry:
                fb, _ = bootstrap_target_force_scale(prov, topo, xc_d, plan=plan)
            row["plan"] = dataclasses.asdict(plan)

            def run(x, w, p, au, f):
                acc, _, field = prov.step_with_targets(
                    x, w, p, plan, aux=au, target_force_scale=f
                )
                return acc, field

            fn = jax.jit(run)
            args = (xg_d, mg_d, xc_d, aux, fb)
        del topo
    else:
        clu, ctopo = FusedForceProvider.prepare(
            _solver("geo", eps=eps_c, theta=0.5, leaf=a.leaf, order=a.order, cpu=cpu),
            xc_d,
            mc_d,
            leaf_size=a.leaf,
            max_order=a.order,
        )
        if a.mode == "back":
            from jaccpot.runtime.target_back_reaction import plan_back_reaction

            plan = plan_back_reaction(clu, ctopo, xg_d, softening=eps_g, far=a.far)
            row["plan"] = dict(
                inner=dataclasses.asdict(plan.inner),
                theta_mp=plan.theta_mp,
                mp_order=plan.mp_order,
                slice_size=plan.slice_size,
            )
        else:
            plan = clu.plan_targets(ctopo, xg_d, softening=eps_g)
            row["plan"] = dataclasses.asdict(plan)
        del ctopo

        def run(xs, ws, p):
            t = clu.rebuild(xs, ws)
            return clu.target_field(t, p, plan, softening=eps_g)

        fn = jax.jit(run)
        args = (xc_d, mc_d, xg_d)
    row["prepare_plan_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    row["first_call_s"] = time.perf_counter() - t0
    walls, out = _time(fn, args, a.warm, a.reps)
    row["ms"] = walls
    row["ms_min"] = min(walls)
    row["ms_median"] = float(np.median(walls))
    field = out[1] if a.mode == "targets" else (None if a.mode == "step" else out)
    if field is not None:
        row["ok"] = bool(field.ok)
        row["needs"] = np.asarray(field.needs).tolist()
        row["stats"] = np.asarray(field.stats).tolist()
    row["peak_bytes"] = _peak()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(row, fh, indent=1, default=str)
    print(json.dumps({k: row[k] for k in ("mode", "n", "m", "mac", "far", "ms_min", "peak_bytes")}))


if __name__ == "__main__":
    main()
