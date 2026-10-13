"""Time Odisseo's public integrate() on the fused lane (clipped Plummer, as jaccpot's
bench/fused_memory_budget.py --ic plummer_clipped: seed 0, rmax 20, velocities rng 7).

Three calls: A (warm-up, `--steps`), B (same length again: does it recompile?), C (3x the
steps). Per-step cost = (T_C - T_B) / (2 * steps) when B and C both pay the same
fixed cost (prepare + compile); also reported: device peak bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8_000_000)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--leaf", type=int, default=None)
ap.add_argument("--order", type=int, default=None)
ap.add_argument("--theta", type=float, default=None)
ap.add_argument("--softening", type=float, default=1e-7)
ap.add_argument("--kernel", default=None)
ap.add_argument("--state-dtype", default="float64", choices=("float32", "float64"))
ap.add_argument("--history", action="store_true", help="return_snapshots=True")
ap.add_argument("--dt", type=float, default=1e-2)
ap.add_argument("--prealloc", type=float, default=0.88)
ap.add_argument("--skip-c", action="store_true")
ap.add_argument("--out", required=True)
ARGS = ap.parse_args()

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(ARGS.prealloc)
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + " --xla_gpu_enable_allocator_spatial_partitioning=false"
    + " --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL"
    + " --xla_gpu_graph_min_graph_size=2"
)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from odisseo.integration_api import integrate, resolve_lane  # noqa: E402
from odisseo.option_classes import FMM_ACC, SimulationConfig, SimulationParams  # noqa: E402


def plummer_clipped(n: int, seed: int = 0, rmax: float = 20.0):
    rng = np.random.default_rng(seed)
    x_max = rmax**3 / (rmax**2 + 1.0) ** 1.5
    x = rng.uniform(0.0, x_max, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1).astype(np.float32)
    rng = np.random.default_rng(7)
    rr = np.linalg.norm(pos.astype(np.float64), axis=1)
    q = np.empty(n)
    todo = np.arange(n)
    while todo.size:
        xx = rng.uniform(0.0, 1.0, todo.size)
        y = rng.uniform(0.0, 0.1, todo.size)
        ok = y < xx * xx * (1.0 - xx * xx) ** 3.5
        q[todo[ok]] = xx[ok]
        todo = todo[~ok]
    speed = q * np.sqrt(2.0) * (1.0 + rr * rr) ** -0.25
    mu = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    st = np.sqrt(1.0 - mu * mu)
    vel = (speed[:, None] * np.stack([st * np.cos(phi), st * np.sin(phi), mu], 1)).astype(
        np.float32
    )
    return pos, vel, np.full(n, 1.0 / n, np.float32)


def peak_gib() -> float:
    st = jax.devices()[0].memory_stats() or {}
    return float(st.get("peak_bytes_in_use", 0)) / 2**30


pos, vel, mass = plummer_clipped(ARGS.n)
sdt = jnp.dtype(ARGS.state_dtype)
state = jnp.asarray(np.stack([pos, vel], 1), dtype=sdt)
mass_j = jnp.asarray(mass, dtype=sdt)

kw = dict(
    N_particles=ARGS.n,
    acceleration_scheme=FMM_ACC,
    num_timesteps=ARGS.steps,
    softening=ARGS.softening,
    return_snapshots=bool(ARGS.history),
    num_snapshots=ARGS.steps,
)
if ARGS.kernel:
    kw["softening_kernel"] = ARGS.kernel
if ARGS.leaf:
    kw["fmm_leaf_size"] = ARGS.leaf
if ARGS.order:
    kw["fmm_max_order"] = ARGS.order
    kw["fmm_fixed_order"] = ARGS.order
if ARGS.theta:
    kw["fmm_theta"] = ARGS.theta
config = SimulationConfig(**kw)
lane = resolve_lane(config)
print(f"[odisseo_time] lane={lane} config={ {k: v for k, v in kw.items()} }", flush=True)


def run(steps: int):
    cfg = config._replace(num_timesteps=steps, num_snapshots=steps)
    params = SimulationParams(G=1.0, t_end=ARGS.dt * steps)
    t0 = time.perf_counter()
    out = integrate(state, mass_j, cfg, params)
    # return_snapshots=True returns odisseo's SnapshotData; time and report its states
    out = getattr(out, "states", out)
    out = jax.block_until_ready(out)
    return time.perf_counter() - t0, out


row = {"n": ARGS.n, "args": vars(ARGS), "lane": lane, "config": {k: str(v) for k, v in kw.items()}}
ta, out = run(ARGS.steps)
row["t_a_s"] = ta
row["peak_a_gib"] = peak_gib()
print(f"[odisseo_time] A {ta:.2f} s peak {row['peak_a_gib']:.2f} GiB out {out.shape} {out.dtype}", flush=True)
del out
tb, out = run(ARGS.steps)
row["t_b_s"] = tb
print(f"[odisseo_time] B {tb:.2f} s", flush=True)
del out
if not ARGS.skip_c:
    tc, out = run(3 * ARGS.steps)
    row["t_c_s"] = tc
    row["step_ms_est"] = 1e3 * (tc - tb) / (2 * ARGS.steps)
    print(f"[odisseo_time] C {tc:.2f} s -> per step {row['step_ms_est']:.1f} ms", flush=True)
    fin = np.asarray(out, np.float64)
    row["finite"] = bool(np.isfinite(fin).all())
row["peak_gib"] = peak_gib()
row["peak_b_per_particle"] = row["peak_gib"] * 2**30 / ARGS.n
print(f"[odisseo_time] peak {row['peak_gib']:.2f} GiB ({row['peak_b_per_particle']:.0f} B/p)", flush=True)
json.dump(row, open(ARGS.out, "w"), indent=1)
