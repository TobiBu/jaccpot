"""How widely a timestep criterion spreads the 25M disc+bulge's particles over rungs.

usage: python bench/blockstep_timestep_spread.py [--n 30000] [--softening 1.5e-3] [--kmax 6]

A block-timestep scheme can save at most ``N 2**k_max / sum_i 2**rung_i`` over a
shared step at the fastest particle's ``dt_min``, so what it buys is set by the spread
of the wanted timesteps, i.e. by the criterion. This takes a random prefix of the IC
(masses rescaled), its fp64 direct-sum self-gravity (Plummer-softened) plus the NFW
halo, and reports for two criteria -- the acceleration criterion ``sqrt(eps / |a|)``
(nornax, GADGET) and an orbit-time one ``sqrt(r / |a|)`` -- the percentiles of
``dt / dt_min``, the rung histogram, the bound, and both against cylindrical radius.
``eta`` cancels in every ratio. Recorded in ``docs/production_2026-10.md`` (S0, D1).
"""

from __future__ import annotations

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30_000)
ap.add_argument("--softening", type=float, default=1.5e-3)
ap.add_argument("--kmax", type=int, default=6)
ARGS = ap.parse_args()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ic = np.load("/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz")
n_all = int(ic["state0"].shape[0])
st = np.asarray(ic["state0"][: ARGS.n], np.float64)
m = np.asarray(ic["mass"][: ARGS.n], np.float64) * (n_all / ARGS.n)
G = float(ic["G_code"]) if "G_code" in ic.files else 1.0
HM, RS = float(ic["halo_mass_code"]), float(ic["halo_rs_code"])
RD = float(ic["rdisk_code"])
eps = ARGS.softening
x = st[:, 0]
X, M = jnp.asarray(x), jnp.asarray(m)


@jax.jit
def _block(xt):
    d = X[None] - xt[:, None]
    r2 = jnp.sum(d * d, -1) + eps * eps
    return jnp.einsum("ij,ijk->ik", G * M[None] * r2**-1.5, d)


a_self = np.concatenate(
    [np.asarray(_block(X[i : i + 1000])) for i in range(0, ARGS.n, 1000)]
)
r = np.linalg.norm(x, axis=1)
R = np.linalg.norm(x[:, :2], axis=1)
q = r / RS
ext = (
    -(G * HM)
    * np.where(
        q < 1e-3,
        (1 / (2 * RS**3)) * (1 - 4 / 3 * q),
        np.log1p(q) / r**3 - 1 / (r * r * (RS + r)),
    )[:, None]
    * x
)
a = np.linalg.norm(a_self + ext, axis=1)
crit = {
    "acceleration sqrt(eps/|a|)": np.sqrt(eps / a),
    "orbit time sqrt(r/|a|)": np.sqrt(r / a),
}
k = ARGS.kmax
for name, dt in crit.items():
    ratio = dt / dt.min()
    rung = np.clip(np.ceil(k - np.log2(ratio)), 0, k).astype(int)
    hist = np.bincount(rung, minlength=k + 1)
    bound = ARGS.n * 2**k / float(np.sum(2.0**rung))
    pct = np.round(np.percentile(ratio, [1, 10, 50, 90, 99]), 2).tolist()
    print(
        f"{name}: dt/dt_min p1/p10/p50/p90/p99 {pct}; rungs (k_max {k}) {hist.tolist()}; bound {bound:.2f}x"
    )
edges = np.array([0, 0.25, 0.5, 1, 2, 3, 4, 6, 10, np.inf]) * RD
print("R/R_d bin | fraction | median |a| | median dt/dt_min, acceleration | orbit time")
dt_a, dt_o = crit.values()
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (R >= lo) & (R < hi)
    if s.sum() < 20:
        continue
    print(
        f"{lo / RD:5.2f}-{min(hi / RD, 99):5.2f} | {s.mean():.3f} | {np.median(a[s]):.3g} | "
        f"{np.median(dt_a[s] / dt_a.min()):.2f} | {np.median(dt_o[s] / dt_o.min()):.1f}"
    )
fit = R > 0.3 * RD
slope = np.polyfit(np.log(R[fit]), np.log(dt_a[fit]), 1)[0]
print(
    f"acceleration criterion: dt ~ R^{slope:.2f} beyond 0.3 R_d; |a| max/median {a.max() / np.median(a):.1f}"
)
