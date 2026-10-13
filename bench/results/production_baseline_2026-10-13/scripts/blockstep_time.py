"""Time Odisseo's block-step lane (integrate_blockstep_jaccpot, device topology -> jitted
lane) on a clipped Plummer sphere in fp64. Calls: A n_base=1 (compile), B n_base=1 again,
C n_base=1+extra. Per base step = (T_C - T_B) / extra. Device peak bytes reported."""

import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100_000)
ap.add_argument("--k-max", type=int, default=3)
ap.add_argument("--extra", type=int, default=3)
ap.add_argument("--dt-max", type=float, default=2e-2)
ap.add_argument("--softening", type=float, default=1e-2)
ap.add_argument("--leaf", type=int, default=64)
ap.add_argument("--theta", type=float, default=0.6)
ap.add_argument("--out", required=True)
A = ap.parse_args()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from odisseo.blockstep_coupling import BlockStepOptions, integrate_blockstep_jaccpot
from odisseo.option_classes import SimulationConfig, SimulationParams

rng = np.random.default_rng(0)
n = A.n
x_max = 20.0**3 / (20.0**2 + 1.0) ** 1.5
u = rng.uniform(0, x_max, n)
r = 1 / np.sqrt(u ** (-2 / 3) - 1)
d = rng.normal(size=(n, 3))
d /= np.linalg.norm(d, axis=1, keepdims=True)
pos = r[:, None] * d
sig = np.sqrt(1.0 / (6.0 * np.sqrt(1.0 + r * r)))
vel = rng.normal(size=(n, 3)) * sig[:, None]
state = jnp.asarray(np.stack([pos, vel], 1), jnp.float64)
mass = jnp.full((n,), 1.0 / n, jnp.float64)
config = SimulationConfig(N_particles=n, softening=A.softening)
params = SimulationParams(G=1.0)
opts = BlockStepOptions(dt_max=A.dt_max, k_max=A.k_max, theta=A.theta, leaf_size=A.leaf)


def run(nb):
    t0 = time.perf_counter()
    res = integrate_blockstep_jaccpot(
        state, mass, config, params, options=opts, n_base=nb, track_energy=False
    )
    jax.block_until_ready(res.state)
    return time.perf_counter() - t0, res


row = {"args": vars(A)}
ta, res = run(1)
row["t_a_s"] = ta
print(
    f"[bs] A {ta:.1f} s rungs {np.asarray(res.rung_histogram).tolist()} far {res.num_far_pairs} near {res.num_near_pairs}",
    flush=True,
)
tb, res = run(1)
row["t_b_s"] = tb
print(f"[bs] B {tb:.1f} s", flush=True)
tc, res = run(1 + A.extra)
row["t_c_s"] = tc
row["base_step_s_est"] = (tc - tb) / A.extra
row["seconds_per_base_step_reported"] = res.seconds_per_base_step
row["rungs"] = np.asarray(res.rung_histogram).tolist()
row["mom_drift_max"] = float(jnp.max(res.momentum_drift))
st = jax.devices()[0].memory_stats() or {}
row["peak_gib"] = st.get("peak_bytes_in_use", 0) / 2**30
row["peak_b_per_particle"] = st.get("peak_bytes_in_use", 0) / n
print(
    f"[bs] C {tc:.1f} s -> base step {row['base_step_s_est']:.2f} s (reported {res.seconds_per_base_step:.2f}); "
    f"peak {row['peak_gib']:.2f} GiB ({row['peak_b_per_particle']:.0f} B/p) mom {row['mom_drift_max']:.1e}",
    flush=True,
)
json.dump(row, open(A.out, "w"), indent=1)
