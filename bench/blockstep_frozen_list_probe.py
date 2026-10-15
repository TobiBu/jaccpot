"""The force error of interaction lists frozen over one block-step base step.

usage: python bench/blockstep_frozen_list_probe.py OUT.json [--n 1000000] [--kmax 3,4,5,6]
           [--dt-min 5e-4] [--theta 0.8] [--order 6] [--leaf 64] [--targets 4096]
           [--margin-safety 1.01]

A block-step base step drifts every particle for ``dt_max = 2**k_max * dt_min`` while
the mutual lane keeps the pair lists it built at the start (nornax hands one
topology to every boundary). This measures what that costs on the 25M disc+bulge IC
(a random prefix, masses rescaled; ferrers3 at the production eps 1.5e-3): the
mutual lane's device topology is built at ``x0``, the particles drift straight to
``x1 = x0 + v0 dt_max``, and the force at ``x1`` is evaluated

* with the lists frozen at ``x0`` (what a base step's last boundary sees),
* with lists rebuilt at ``x1`` (what a per-step rebuild sees),

both scored against an fp64 direct sum (same kernel) on random targets. Momentum
checks cannot see this error: a stale list is still a symmetric partition.

``--margin-safety S`` adds the drift-margin arm (S1): lists built at ``x0`` with
``drift_margin = S * max|v0| dt_max`` (``suggest_drift_margin``), scored at ``x1`` like
the frozen arm, with their far/near pair counts against the plain lists' and whether
``check_drift`` flags the drift. ``S`` a hair above 1 keeps the straight drift inside
the margin through round-off.
"""

from __future__ import annotations

import argparse
import json
import time

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=1_000_000)
ap.add_argument("--kmax", default="3,4,5,6")
ap.add_argument(
    "--dt-min", type=float, default=5e-4, help="the disc's production shared dt"
)
ap.add_argument("--theta", type=float, default=0.8)
ap.add_argument("--order", type=int, default=6)
ap.add_argument("--leaf", type=int, default=64)
ap.add_argument("--softening", type=float, default=1.5e-3)
ap.add_argument("--kernel", default="ferrers3")
ap.add_argument("--targets", type=int, default=4096)
ap.add_argument(
    "--margin-safety",
    type=float,
    default=0.0,
    help="add the drift-margin arm at this safety factor (0: off)",
)
ARGS = ap.parse_args()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jaccpot import BlockStepFMM  # noqa: E402
from jaccpot.nornax_adapter import suggest_drift_margin  # noqa: E402
from jaccpot.softening import pair_factors, softening_params  # noqa: E402

ic = np.load("/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz")
n_all = int(ic["state0"].shape[0])
state = np.asarray(
    ic["state0"][: ARGS.n], np.float64
)  # shuffled: a prefix is a subsample
mass = np.asarray(ic["mass"][: ARGS.n], np.float64) * (n_all / ARGS.n)
g = float(ic["G_code"]) if "G_code" in ic.files else 1.0
x0, v0 = state[:, 0], state[:, 1]
idx = np.sort(np.random.default_rng(12345).choice(ARGS.n, ARGS.targets, replace=False))


def direct(x: np.ndarray) -> np.ndarray:
    """fp64 direct sum at ``idx`` with the same softening kernel, self excluded."""
    p, m = jnp.asarray(x), jnp.asarray(mass)
    params = softening_params(ARGS.kernel, ARGS.softening, jnp.float64)
    chunk = 1 << 18
    npad = -(-ARGS.n // chunk) * chunk
    sp = jnp.zeros((npad, 3)).at[: ARGS.n].set(p).reshape(-1, chunk, 3)
    sm = jnp.zeros((npad,)).at[: ARGS.n].set(m).reshape(-1, chunk)
    sid = jnp.arange(npad).reshape(-1, chunk)

    @jax.jit
    def block(tp, tid):
        def body(c, acc):
            d = tp[:, None, :] - sp[c][None, :, :]
            r2 = jnp.sum(d * d, axis=-1)
            f = pair_factors(jnp.where(r2 > 0, r2, 1.0), params, ARGS.kernel)[0]
            w = jnp.where(tid[:, None] == sid[c][None, :], 0.0, f * sm[c][None, :])
            return acc - g * jnp.einsum("ij,ijk->ik", w, d)

        return jax.lax.fori_loop(0, sp.shape[0], body, jnp.zeros_like(tp))

    out = [
        np.asarray(
            block(p[jnp.asarray(idx[b : b + 256])], jnp.asarray(idx[b : b + 256]))
        )
        for b in range(0, len(idx), 256)
    ]
    return np.concatenate(out)


def score(a: np.ndarray, ref: np.ndarray) -> dict:
    got = a[idx]
    e = np.linalg.norm(got - ref, axis=1) / np.maximum(
        np.linalg.norm(ref, axis=1), 1e-300
    )
    return dict(
        rel_l2=float(np.linalg.norm(got - ref) / np.linalg.norm(ref)),
        median=float(np.median(e)),
        p99=float(np.percentile(e, 99)),
        max=float(e.max()),
    )


report = {"args": vars(ARGS), "rows": {}}
for k in [int(s) for s in ARGS.kmax.split(",")]:
    t0 = time.perf_counter()
    dt_max = (1 << k) * ARGS.dt_min
    lane = dict(
        softening=ARGS.softening,
        softening_kernel=ARGS.kernel,
        k_max=k,
        theta=ARGS.theta,
        max_order=ARGS.order,
        G=g,
        leaf_size=ARGS.leaf,
        backend="jax",
        static_shapes=True,
        topology_backend="device",
    )
    force = BlockStepFMM(**lane)
    X0, M = jnp.asarray(x0), jnp.asarray(mass)
    force.prepare(X0, M)
    topo0 = force.rebuild_state(X0, M)
    x1 = x0 + v0 * dt_max
    X1 = jnp.asarray(x1)
    frozen = np.asarray(force.weighted_accelerations(topo0, X1, M))
    topo1 = force.rebuild_state(X1, M)
    fresh = np.asarray(force.weighted_accelerations(topo1, X1, M))
    ref = direct(x1)
    disp = np.linalg.norm(v0 * dt_max, axis=1)
    row = dict(
        dt_max=dt_max,
        displacement_median=float(np.median(disp)),
        displacement_p99=float(np.percentile(disp, 99)),
        displacement_over_softening_p99=float(np.percentile(disp, 99) / ARGS.softening),
        frozen=score(frozen, ref),
        fresh=score(fresh, ref),
        overflow_frozen=bool(np.asarray(getattr(topo0, "topology_overflow", False))),
        overflow_fresh=bool(np.asarray(getattr(topo1, "topology_overflow", False))),
        far_pairs=int(topo0.num_far_pairs),
        near_pairs=int(topo0.num_near_pairs),
        seconds=time.perf_counter() - t0,
    )
    if ARGS.margin_safety > 0.0:
        margin = suggest_drift_margin(v0, dt_max, safety=ARGS.margin_safety)
        guarded = BlockStepFMM(**lane, drift_margin=margin)
        guarded.prepare(X0, M)
        topo_m = guarded.rebuild_state(X0, M)
        row["margin"] = dict(
            drift_margin=margin,
            far_pairs=int(topo_m.num_far_pairs),
            near_pairs=int(topo_m.num_near_pairs),
            frozen=score(
                np.asarray(guarded.weighted_accelerations(topo_m, X1, M)), ref
            ),
            guard_fired=bool(guarded.check_drift(topo_m, X1).topology_overflow),
            overflow=bool(np.asarray(topo_m.topology_overflow)),
            seconds=time.perf_counter() - t0 - row["seconds"],
        )
    report["rows"][k] = row
    print(
        f"[frozen] k_max {k} dt_max {dt_max:.2e}: displacement p99 {row['displacement_p99']:.3e} "
        f"({row['displacement_over_softening_p99']:.1f} eps) | frozen rel-L2 {row['frozen']['rel_l2']:.3e} "
        f"max {row['frozen']['max']:.2e} | fresh rel-L2 {row['fresh']['rel_l2']:.3e} "
        f"max {row['fresh']['max']:.2e} | {row['seconds']:.0f} s",
        flush=True,
    )
    if "margin" in row:
        arm = row["margin"]
        print(
            f"[margin] k_max {k}: drift_margin {arm['drift_margin']:.3e} | far "
            f"{row['far_pairs']} -> {arm['far_pairs']}, near {row['near_pairs']} -> "
            f"{arm['near_pairs']} | frozen rel-L2 {arm['frozen']['rel_l2']:.3e} max "
            f"{arm['frozen']['max']:.2e} | guard {arm['guard_fired']}",
            flush=True,
        )
    with open(ARGS.out, "w") as fh:
        json.dump(report, fh, indent=1)
