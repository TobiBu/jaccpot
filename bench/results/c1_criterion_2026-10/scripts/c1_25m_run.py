"""C1 follow-up: the 25M production configuration at one shared dt, to t_end 1.5, with the C1 probes.

Adapted from the S3 session 2 production gate (``s32prod/prodrun.py``). The IC, the
environment, the XLA flags and the analytic NFW halo are ``bench/fused_memory_budget.py``'s
(imported with the L3 pin's argv: ``--ic disc --n 25e6 --softening 1.5e-3
--softening-kernel ferrers3 --caps unnamed``, ``JACCPOT_STRICT_CARRY=particles``), with
``--prealloc 0`` (a shared card) and ``--dt``. The solver is
``FastMultipoleMethod(preset="galaxy", softening=1.5e-3)`` as is (dehnen_error 1e-4, the
cubic box), the run ``ProductionRunner`` with segments of 0.025 time units (so the three
runs' edges fall at the same times), the moments every edge and the energy every 2nd edge
(0.05), no restarts, SnapshotWriter at t = 0, 0.5, 1.0, 1.5 and a DiagnosticsLog.

The C1 probe (a callback; it reads ``runner._current["handle"]`` as the runner's own
energy diagnostic does, and leaves the engine's trace state as it found it):
* every energy edge: A2 of the disc (R < 3 R_d, and 0-1 / 1-2 / 2-3 R_d), the Lagrangian
  radii of all baryons, the disc and the bulge (0.5 % .. 90 %, equal masses), L;
* the per-particle energy E_i = v^2 / 2 + phi_i + Phi_halo at t = 0 and t_end, phi_i from
  the provider's potential on a topology rebuilt at those positions (the energy
  diagnostic's program, per particle), saved as float32 for the diffusion score;
* at the first edge, the median particle displacement against t = 0 (the state keeps the
  input order: a permutation would show O(1)).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--dt", type=float, required=True)
ap.add_argument("--n", type=int, default=25_000_000)
ap.add_argument("--t-end", type=float, default=1.5)
ap.add_argument("--seg-time", type=float, default=0.025)
ap.add_argument("--energy-every", type=int, default=2)
ap.add_argument("--snap-every", type=int, default=20)
ap.add_argument("--scratch", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--jsonl", required=True)
MY = ap.parse_args()

BENCH = os.path.join(os.environ["S1_JACCPOT"], "bench")
sys.argv = [
    "fused_memory_budget.py",
    "--n",
    str(MY.n),
    "--caps",
    "unnamed",
    "--prealloc",
    "0",
    "--no-analysis",
    "--ic",
    "disc",
    "--softening",
    "1.5e-3",
    "--softening-kernel",
    "ferrers3",
    "--skip-eval",
    "--dt",
    repr(MY.dt),
    "--out",
    MY.out + ".bench.json",
    "--env",
    "JACCPOT_STRICT_CARRY=particles",
]
sys.path.insert(0, BENCH)
import fused_memory_budget as fmb  # noqa: E402  (module-level setup = the pin's)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jaccpot  # noqa: E402
from jaccpot import FastMultipoleMethod  # noqa: E402
from jaccpot.production import (  # noqa: E402
    DIAGNOSTICS,
    DiagnosticsLog,
    ProductionRunner,
    SegmentConfig,
    SnapshotWriter,
    _trace_state_kept,
)
from jaccpot.runtime.fused_topology import ForceParams  # noqa: E402

DEV = jax.devices()[0]
assert DEV.platform == "gpu", jax.devices()
print(
    "jaccpot",
    jaccpot.__file__,
    "device",
    DEV,
    os.environ.get("CUDA_VISIBLE_DEVICES"),
    flush=True,
)

_IC = np.load(fmb._DISC_IC, mmap_mode="r")
_G = float(_IC["G_code"]) if "G_code" in _IC.files else 1.0
_HALO_M, _HALO_RS = float(_IC["halo_mass_code"]), float(_IC["halo_rs_code"])
_RD = float(_IC["rdisk_code"])
COMP = np.asarray(_IC["component"][: MY.n])
LAGR = (0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)


def nfw_potential(state_in):
    """AGAMA's NFW potential per unit mass (the S3.2 driver's)."""
    pos = state_in[:, 0, :]
    r = jnp.sqrt(jnp.maximum(jnp.sum(pos * pos, axis=1), 1e-30))
    x = r / _HALO_RS
    log1p_over_x = jnp.where(
        x < 1e-3, 1.0 - 0.5 * x + x * x / 3.0, jnp.log1p(x) / jnp.maximum(x, 1e-30)
    )
    return -(_G * _HALO_M / _HALO_RS) * log1p_over_x


class C1Probe:
    def __init__(self, engine, x0_host, num_steps):
        self.engine = engine
        self.runner = None
        self.x0 = x0_host  # [N, 2, 3] float32, host
        self.num_steps = num_steps
        self.rows = []
        self.progs = {}
        self.e0 = None
        self.e_end = None
        self.disc = jnp.asarray(np.flatnonzero(COMP == 0).astype(np.int32))
        self.bulge = jnp.asarray(np.flatnonzero(COMP == 1).astype(np.int32))
        self.disc_mask = jnp.asarray(COMP == 0)
        self.order_check = None
        self.energy_s = []

    def particle_energy(self, state, masses):
        handle = self.runner._current["handle"]
        prov = handle.provider
        fk = handle.force_key
        params = None if fk is None else ForceParams(G=float(fk[0]), eps2=float(fk[1]))
        key = (prov.key(), fk)
        prog = self.progs.get(key)
        if prog is None:

            def fn(state_in, m, scale):
                topo = prov.rebuild(state_in[:, 0], m, force_scale=scale)
                phi = prov.potential(topo, params)
                v = state_in[:, 1]
                e = 0.5 * jnp.sum(v * v, 1) + phi + nfw_potential(state_in)
                return e.astype(jnp.float32), prov.capacity_ok(topo)

            prog = jax.jit(fn)
            self.progs[key] = prog
        scale = handle.force_scale if prov.scale_carry else None
        state = jax.device_put(state, masses.sharding)
        t0 = time.perf_counter()
        with _trace_state_kept(self.engine):
            e, ok = prog(state, masses, scale)
        e = np.asarray(jax.device_get(e))
        self.energy_s.append(time.perf_counter() - t0)
        return e, bool(jax.device_get(ok))

    def moments(self, state):
        x, v = state[:, 0], state[:, 1]
        r = jnp.sqrt(jnp.sum(x * x, 1))
        rc = jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        ph = jnp.arctan2(x[:, 1], x[:, 0])
        a2 = {}
        for lo, hi in ((0, 1), (1, 2), (2, 3), (0, 3)):
            sel = self.disc_mask & (rc >= lo * _RD) & (rc < hi * _RD)
            c = jnp.sum(jnp.where(sel, jnp.cos(2 * ph), 0.0))
            s = jnp.sum(jnp.where(sel, jnp.sin(2 * ph), 0.0))
            a2[f"{lo}-{hi}"] = float(jnp.hypot(c, s) / jnp.maximum(jnp.sum(sel), 1))
        lagr = {}
        for name, rr in (("all", r), ("disc", r[self.disc]), ("bulge", r[self.bulge])):
            srt = jnp.sort(rr)
            idx = jnp.asarray(
                [min(int(f * srt.shape[0]), srt.shape[0] - 1) for f in LAGR]
            )
            lagr[name] = np.asarray(srt[idx]).astype(float).tolist()
        ang = jnp.sum(jnp.cross(x.astype(jnp.float64), v.astype(jnp.float64)), 0)
        return dict(A2=a2, lagrange=lagr, L=np.asarray(ang).tolist())

    def __call__(self, edge):
        if edge.segment == 1 and self.order_check is None:
            idx = np.arange(0, MY.n, 250)
            xs = np.asarray(jax.device_get(edge.state[idx, 0]))
            d = np.linalg.norm(xs - self.x0[idx, 0], axis=1)
            self.order_check = dict(
                median=float(np.median(d)), p99=float(np.percentile(d, 99))
            )
            print(
                "order check (|x - x0| after one segment)", self.order_check, flush=True
            )
        handle_ok = (
            self.runner is not None and self.runner._current.get("handle") is not None
        )
        if self.e0 is None and handle_ok:
            self.e0, ok0 = self.particle_energy(jnp.asarray(self.x0), edge.masses)
            print(f"E_i(0): ok {ok0}, mean {self.e0.mean():.6g}", flush=True)
        if edge.segment % MY.energy_every == 0:
            row = dict(step=edge.step, time=edge.time, diagnostics=edge.diagnostics)
            row.update(self.moments(edge.state))
            self.rows.append(row)
            d = edge.diagnostics or {}
            print(
                f"t {edge.time:.4f} step {edge.step}: dE {d.get('energy_drift')} dL "
                f"{d.get('angular_momentum_drift')} A2 {row['A2']['0-3']:.4g} r1% "
                f"{row['lagrange']['all'][1]:.5g} wall {edge.wall_s:.2f} s",
                flush=True,
            )
        if edge.step == self.num_steps:
            self.e_end, ok1 = self.particle_energy(edge.state, edge.masses)
            print(f"E_i(end): ok {ok1}, mean {self.e_end.mean():.6g}", flush=True)


def main():
    t_main = time.perf_counter()
    os.makedirs(MY.scratch, exist_ok=True)
    pos, vel, mass, soft, dt, ext = fmb._initial_conditions()
    mass = fmb._in_g_units(mass)
    assert abs(dt - MY.dt) < 1e-15 and ext is not None
    seg_steps = int(round(MY.seg_time / dt))
    num_steps = int(round(MY.t_end / dt))
    assert num_steps % seg_steps == 0, (num_steps, seg_steps)
    x0_host = np.stack([np.asarray(pos, np.float32), np.asarray(vel, np.float32)], 1)
    state = jnp.asarray(x0_host)
    masses = jnp.asarray(mass, jnp.float32)
    del pos, vel
    solver = FastMultipoleMethod(preset="galaxy", softening=soft)
    engine = solver._impl
    seg = SegmentConfig(
        steps=seg_steps,
        traced_params=True,
        donate_input=True,
        diagnostics_every=1,
        diagnostics=DIAGNOSTICS,
        diagnostics_every_energy=MY.energy_every,
        restart_every=0,
    )
    probe = C1Probe(engine, x0_host, num_steps)
    snaps = SnapshotWriter(
        os.path.join(MY.scratch, "snapshots"), every=MY.snap_every, initial=True
    )
    if os.path.exists(MY.jsonl):
        os.remove(MY.jsonl)
    runner = ProductionRunner(
        solver,
        segment=seg,
        external_acceleration_fn=ext,
        external_potential_fn=nfw_potential,
        callbacks=(snaps, DiagnosticsLog(MY.jsonl), probe),
    )
    probe.runner = runner
    t0 = time.perf_counter()
    res = runner.run(state, masses, dt=dt, num_steps=num_steps)
    del state
    jax.block_until_ready(res.state)
    wall = time.perf_counter() - t0
    np.savez(
        os.path.join(MY.scratch, "particle_energy.npz"),
        e0=probe.e0,
        e_end=probe.e_end,
        comp=COMP,
    )
    mem = fmb._mem()
    out = dict(
        dt=dt,
        n=MY.n,
        num_steps=num_steps,
        segment_steps=seg_steps,
        t_end=res.time,
        softening=soft,
        jaccpot=jaccpot.__file__,
        device=[str(DEV), os.environ.get("CUDA_VISIBLE_DEVICES")],
        xla_flags=os.environ.get("XLA_FLAGS"),
        env={k: v for k, v in os.environ.items() if k.startswith("JACCPOT_")},
        solver=dict(
            theta=float(engine.theta),
            mac_type=str(getattr(engine, "mac_type", None)),
            softening_kernel=str(getattr(engine, "softening_kernel", None)),
            working_dtype=str(jnp.dtype(engine.working_dtype)),
        ),
        run_wall_s=wall,
        steps_per_s_hint=num_steps / wall,
        mem_peak_gib=mem["peak"] / float(1 << 30),
        order_check=probe.order_check,
        particle_energy_s=probe.energy_s,
        rows=probe.rows,
        snapshots=list(snaps.paths),
        baselines=res.baselines,
        total_wall_s=time.perf_counter() - t_main,
    )
    with open(MY.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(
        f"done: {num_steps} steps in {wall:.1f} s ({num_steps / wall:.2f} steps/s), peak {out['mem_peak_gib']:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
