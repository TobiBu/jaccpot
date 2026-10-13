"""Accuracy and cost of two momentum-exact block-step maps, by direct summation.

usage: python bench/blockstep_hold_vs_max_ab.py OUT.json [--n 20000] [--kmax 3,4]
           [--n-base 32] [--eta 0.1] [--softening-rdisk 0.02]

Both schemes give each interaction ONE level and kick both partners with it, so both
conserve momentum to round-off; they differ in which level, and so in what an FMM
would have to evaluate at a fine boundary:

* MAX (the mutual lane, nornax ``block_kdk_*``): level ``max(rung_i, rung_j)``,
  every particle drifts every ``dt_min``. A fast particle's partners are kicked at
  its rate; at a fine boundary every particle near or far-paired with a fast one is
  kicked, and every particle drifts (the O(N) floor of a tree code).
* HOLD (Pelupessy et al. 2012; GADGET-4's hierarchical gravity): level
  ``min(rung_i, rung_j)``. Level ``l`` kicks the pairs whose slower end is on rung
  ``l``, drifts the rung-``l`` particles, and evolves the faster set recursively,
  so a fine boundary touches only the particles on the fine rungs. The price:
  forces between fast and slow particles act on the slow step.

The IC is a random prefix of the 25M disc+bulge (masses rescaled) in its NFW halo
(external field kicked on each particle's own rung), Plummer-softened pair forces in
fp64. Rungs are assigned at each base step from ``eta * sqrt(eps / |a|)`` with the
fastest particle on the finest rung at t=0. Each scheme is scored at the end against
a shared-step reference at ``dt_min / 4``: energy error ``|E - E0| / |W0|``, the
position error distribution, momentum, and the pair evaluations relative to a shared
``dt_min`` run (the cost a pruned FMM would track). Total momentum is not conserved
in the halo, so the momentum check is the pair impulse balance ``sum_i m_i a_i``
of each map's level-weighted pair force.
"""

from __future__ import annotations

import argparse
import json
import time

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=20_000)
ap.add_argument("--kmax", default="3,4")
ap.add_argument("--n-base", type=int, default=32)
ap.add_argument("--eta", type=float, default=0.1)
ap.add_argument("--softening-rdisk", type=float, default=0.02)
ap.add_argument("--block", type=int, default=1024)
ARGS = ap.parse_args()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ic = np.load("/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz")
n_all = int(ic["state0"].shape[0])
st = np.asarray(ic["state0"][: ARGS.n], np.float64)
M = jnp.asarray(np.asarray(ic["mass"][: ARGS.n], np.float64) * (n_all / ARGS.n))
G = float(ic["G_code"]) if "G_code" in ic.files else 1.0
HM, RS = float(ic["halo_mass_code"]), float(ic["halo_rs_code"])
EPS = ARGS.softening_rdisk * float(ic["rdisk_code"])
X0, V0 = jnp.asarray(st[:, 0]), jnp.asarray(st[:, 1])
N = ARGS.n
NB = -(-N // ARGS.block)
PAD = NB * ARGS.block


def ext_acc(x):
    r2 = jnp.sum(x * x, 1)
    r = jnp.sqrt(jnp.maximum(r2, 1e-30))
    q = r / RS
    big = jnp.log1p(q) / jnp.maximum(r2 * r, 1e-30) - 1.0 / jnp.maximum(
        r2 * (RS + r), 1e-30
    )
    ser = (1.0 / (2.0 * RS**3)) * (1.0 - (4.0 / 3.0) * q)
    return -(G * HM) * jnp.where(q < 1e-3, ser, big)[:, None] * x


def ext_pot(x):
    r = jnp.sqrt(jnp.maximum(jnp.sum(x * x, 1), 1e-30))
    return -G * HM * jnp.log1p(r / RS) / r


def pair_acc(x, rung, weight_of):
    """``a_i = sum_j w(rung_i, rung_j) G m_j (x_j - x_i) / (r^2 + eps^2)^1.5``, target blocks."""
    xp = jnp.zeros((PAD, 3)).at[:N].set(x)
    rp = jnp.zeros((PAD,), jnp.int32).at[:N].set(rung)
    ids = jnp.arange(PAD)

    def blk(b):
        sl = jax.lax.dynamic_slice_in_dim
        xt, rt, it = (
            sl(xp, b * ARGS.block, ARGS.block),
            sl(rp, b * ARGS.block, ARGS.block),
            sl(ids, b * ARGS.block, ARGS.block),
        )
        d = x[None, :, :] - xt[:, None, :]
        r2 = jnp.sum(d * d, -1) + EPS * EPS
        c = G * M[None, :] * r2**-1.5
        w = weight_of(rt[:, None], rung[None, :]) * (
            it[:, None] != jnp.arange(N)[None, :]
        )
        return jnp.einsum("ij,ijk->ik", c * w, d)

    out = jax.lax.map(blk, jnp.arange(NB))
    return out.reshape(PAD, 3)[:N]


def energy(x, v):
    def blk(b):
        sl = jax.lax.dynamic_slice_in_dim
        xp = jnp.zeros((PAD, 3)).at[:N].set(x)
        mp = jnp.zeros((PAD,)).at[:N].set(M)
        xt, mt = sl(xp, b * ARGS.block, ARGS.block), sl(mp, b * ARGS.block, ARGS.block)
        it = b * ARGS.block + jnp.arange(ARGS.block)
        d = x[None, :, :] - xt[:, None, :]
        r = jnp.sqrt(jnp.sum(d * d, -1) + EPS * EPS)
        w = it[:, None] != jnp.arange(N)[None, :]
        return jnp.sum(mt[:, None] * M[None, :] * w / r)

    w_self = -0.5 * G * jnp.sum(jax.lax.map(blk, jnp.arange(NB)))
    return 0.5 * jnp.sum(M[:, None] * v * v) + w_self + jnp.sum(M * ext_pot(x)), w_self


ALL = lambda ri, rj: jnp.ones(jnp.broadcast_shapes(ri.shape, rj.shape))  # noqa: E731


@jax.jit
def total_acc(x):
    return pair_acc(x, jnp.zeros(N, jnp.int32), ALL) + ext_acc(x)


def rungs_for(a, dt_max, k):
    dt = ARGS.eta * jnp.sqrt(EPS / (jnp.linalg.norm(a, axis=1) + 1e-30))
    return jnp.clip(jnp.ceil(jnp.log2(dt_max / dt)), 0, k).astype(jnp.int32)


def shared(x, v, dt, steps):
    def body(c, _):
        x, v, a = c
        v = v + 0.5 * dt * a
        x = x + dt * v
        a = total_acc(x)
        return (x, v + 0.5 * dt * a, a), None

    (x, v, _), _ = jax.lax.scan(body, (x, v, total_acc(x)), None, length=steps)
    return x, v


def max_base_step(x, v, k, dt_max):
    """One base step of the MAX map: global drift by dt_min, boundary weights per level."""
    dtl = dt_max / 2.0 ** jnp.arange(k + 1)
    rung = rungs_for(total_acc(x), dt_max, k)
    n_sub = 1 << k
    pairs = 0.0
    for s in range(n_sub + 1):
        if s in (0, n_sub):
            wl = 0.5 * dtl
        else:
            wl = jnp.where(
                jnp.asarray([s % (1 << (k - l)) == 0 for l in range(k + 1)]), dtl, 0.0
            )
        a = pair_acc(x, rung, lambda ri, rj: wl[jnp.maximum(ri, rj)]) + wl[rung][
            :, None
        ] * ext_acc(x)
        if s > 0:  # s = 0 merges with the previous base step's last boundary
            floor = 0 if s == n_sub else k - ((s & -s).bit_length() - 1)
            n_lt = jnp.sum(rung < floor).astype(jnp.float64)
            pairs = (
                pairs + float(N) ** 2 - n_lt**2
            )  # ordered pairs with max level >= floor
        v = v + a
        if s < n_sub:
            x = x + (dt_max / n_sub) * v
    return x, v, rung, pairs


def hold_base_step(x, v, k, dt_max):
    """One base step of the HOLD map (min level), unrolled: K(1/2) D_S [F twice] K(1/2)."""
    rung = rungs_for(total_acc(x), dt_max, k)
    pairs = [0.0]

    def kick(x, v, l, h):
        w = lambda ri, rj: (jnp.minimum(ri, rj) == l).astype(jnp.float64)  # noqa: E731
        a = pair_acc(x, rung, w) + (rung == l)[:, None] * ext_acc(x)
        n_ge = jnp.sum(rung >= l).astype(jnp.float64)
        n_gt = jnp.sum(rung > l).astype(jnp.float64)
        # ordered pairs with min level == l; half per kick, since the closing kick
        # of one level-l step merges with the opening kick of the next
        pairs[0] = pairs[0] + 0.5 * (n_ge**2 - n_gt**2)
        return v + h * a

    def evolve(x, v, l, dt):
        v = kick(x, v, l, 0.5 * dt)
        slow = (rung == l)[:, None]
        if l == k:
            x = x + dt * v * slow
        else:
            x = x + dt * v * slow  # drift S; commutes with the fast evolution
            x, v = evolve(x, v, l + 1, 0.5 * dt)
            x, v = evolve(x, v, l + 1, 0.5 * dt)
        v = kick(x, v, l, 0.5 * dt)
        return x, v

    x, v = evolve(x, v, 0, dt_max)
    return x, v, rung, pairs[0]


def momentum(v):
    return np.asarray(jnp.sum(M[:, None] * v, 0))


report = {"args": vars(ARGS), "n": N, "softening": EPS, "rows": {}}
E0, W0 = energy(X0, V0)
p0 = momentum(V0)
a0 = total_acc(X0)
dt_req0 = ARGS.eta * jnp.sqrt(EPS / (jnp.linalg.norm(a0, axis=1) + 1e-30))
dt_min = float(jnp.min(dt_req0))
for k in [int(s) for s in ARGS.kmax.split(",")]:
    dt_max = dt_min * (1 << k)
    steps = ARGS.n_base * (1 << k)
    t0 = time.perf_counter()
    xr, vr = jax.jit(shared, static_argnums=(3,))(X0, V0, dt_min / 4, 4 * steps)
    xs, vs = jax.jit(shared, static_argnums=(3,))(X0, V0, dt_min, steps)
    rows = {"reference_dt_min_over_4": None}
    for name, fn in (("max", max_base_step), ("hold", hold_base_step)):
        step = jax.jit(lambda x, v, fn=fn: fn(x, v, k, dt_max))
        x, v, pairs, hist = X0, V0, 0.0, np.zeros(k + 1)
        for _ in range(ARGS.n_base):
            x, v, rung, p = step(x, v)
            pairs += float(p)
            hist += np.bincount(np.asarray(rung), minlength=k + 1)
        rows[name] = (x, v, pairs, hist / ARGS.n_base)
    shared_pairs = float(steps) * N * N  # one force per dt_min step (KDK reuses it)
    out = {"dt_min": dt_min, "dt_max": dt_max, "steps": steps}
    for name, (x, v) in (("shared_dt_min", (xs, vs)),) + tuple(
        (nm, rows[nm][:2]) for nm in ("max", "hold")
    ):
        e, _ = energy(x, v)
        dx = np.linalg.norm(np.asarray(x - xr), axis=1)
        out[name] = dict(
            dE_over_W0=float(abs(e - E0) / abs(W0)),
            dx_median_over_eps=float(np.median(dx) / EPS),
            dx_p99_over_eps=float(np.percentile(dx, 99) / EPS),
        )
    for nm in ("max", "hold"):
        out[nm]["pair_evaluations_vs_shared"] = rows[nm][2] / shared_pairs
        out[nm]["mean_rung_hist"] = rows[nm][3].tolist()
    # the halo changes the total momentum, so check the pair forces themselves: a
    # symmetric level map must give a zero net pair impulse (here, at t = 0)
    r0 = rungs_for(a0, dt_max, k)
    for nm, w in (
        ("max", lambda ri, rj: jnp.maximum(ri, rj).astype(jnp.float64) + 1.0),
        ("hold", lambda ri, rj: jnp.minimum(ri, rj).astype(jnp.float64) + 1.0),
    ):
        ap_ = pair_acc(X0, r0, w)
        out[nm]["pair_impulse_imbalance"] = float(
            np.linalg.norm(momentum(ap_))
            / float(jnp.sum(M * jnp.linalg.norm(ap_, axis=1)))
        )
    out["seconds"] = time.perf_counter() - t0
    report["rows"][k] = out
    print(
        f"[ab] k_max {k} dt_min {dt_min:.3e} ({steps} fine steps): {json.dumps(out)}",
        flush=True,
    )
    with open(ARGS.out, "w") as fh:
        json.dump(report, fh, indent=1)
