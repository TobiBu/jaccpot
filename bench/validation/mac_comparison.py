"""Figure 03 -- force error vs WALL CLOCK on the fused lane: ``dehnen_error`` vs geometric.

REBASE_PLAN replaces the September figure (geometric vs mass-dependent criterion at
equal pair work on ``preset="accurate"``, via ``mac_error_distribution.py``) with
the trade production actually makes: on ``bench/scaling/_galaxy.py``'s solver
(``preset="galaxy"``: the fused lane, cell leaves of 64, p6, ferrers3, fp32), at one
fixed N, sweep

* ``dehnen_error`` -- Dehnen (2014) eq (16a), evaluated per pair in the flat walk,
  over ``--eps`` (``adaptive_eps``; the preset's is 1e-4), and
* ``dehnen`` -- the geometric criterion, over ``--theta`` (the preset's 0.8),

and put every point at ``(rel-L2 of the force against an fp64 direct sum,
per-step wall clock)``. Only the MAC (and its knob) moves off the preset;
``config.points[*].galaxy.off_preset`` records it per point.

What each point is
------------------
* **error**: the step's acceleration at ``--accuracy-targets`` seeded targets
  against an fp64 direct sum with the same softening kernel, computed once per
  N (``_timing.direct_reference``): aggregate rel-L2, and the per-target
  percentiles (p50/p90/p99/max of ``|da|/|a|``).
* **cost**: ``FusedForceProvider.step`` jitted (rebuild + evaluate: what a
  simulation pays per step), minimum over ``--repeats``, with the steady-state f_b
  (``_timing.warm_fused_step``); evaluation only is recorded beside it. Not timed:
  the eager prepare (setup; ``setup.prepare_s``, one sample, compiles included).
* **lists**: the walk's directed far and near pair counts from the step's own aux
  (``needs``), so the old pair-count axis is available, but it is not the x-axis.
* A point whose capacity verdict fails is ``valid: false`` (the force is wrong),
  is not plotted and enters no comparison.

Comparisons (``data.comparisons``)
----------------------------------
For each valid ``dehnen_error`` point: the geometric arm's step time log-
interpolated at the same rel-L2 (``cost_ratio`` = mass / geometric at matched
error; < 1 favours ``dehnen_error``), and the geometric arm's rel-L2 interpolated at
the same step time (``error_ratio`` = geometric / mass at matched cost; > 1 favours
``dehnen_error``). Outside the geometric arm's range a ratio is null, never
extrapolated. ``data.preset_pair`` is the two presets' own points (``eps`` 1e-4 vs
``theta`` 0.8) when both were swept: their cost and error ratios directly.

Reference to check against, not to copy: ``docs/production.md`` -- on the 25M
disc + bulge, ``dehnen_error`` 1e-4 cost 1.12x the geometric theta-0.8 step for
2.8x lower rel-L2 (260.5 vs 233.6 ms; 1.49e-4 vs 4.18e-4). The September output's
"median 78x" at equal interaction count is NOT produced by this script: it was a
pair-work statement on another lane and is retired unless remeasured.

Initial conditions: a ``bench.scaling._timing.distribution`` (default the
production rows' clipped Plummer), or ``--ic-npz`` -- the 25M disc + bulge IC the
reference row used (``state0``, ``mass``, ``rdisk_code``, ``G_code``; a prefix is
a random subsample of the shuffled file, masses rescaled so it is the same
galaxy, softening ``0.5 rdisk / sqrt(N / 1e5)`` unless ``--softening`` is given,
as ``bench/fused_memory_budget.py --ic disc``). The disc's analytic halo is an
external field and is not part of the force scored here.

The adaptive_eps sweep
----------------------
A point off the preset's 1e-4 is ``_galaxy.galaxy_solver(adaptive_eps=...)``, and
the built solver's own ``adaptive_eps`` is checked against the one asked for.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu python -m bench.validation.mac_comparison --n 4096 \\
        --eps 1e-3,1e-4 --theta 0.6,0.8 --repeats 2 --warmup 1 \\
        --accuracy-targets 512 --gpu-select none --json-out /tmp/smoke.json

Paper run (one free A100; 25M, as the reference row)::

    python -m bench.validation.mac_comparison --n 25000000 \\
        --ic-npz /export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "validation/mac_comparison.json"


def _floats(text: str) -> list[float]:
    return [float(v) for v in str(text).split(",") if v.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", default="25e6", help="Particle count (1e6 notation allowed)")
    p.add_argument(
        "--eps",
        default="1e-3,3e-4,1e-4,3e-5,1e-5",
        help="dehnen_error's adaptive_eps sweep (the preset's is 1e-4)",
    )
    p.add_argument(
        "--theta",
        default="0.5,0.6,0.7,0.8,0.9",
        help="The geometric criterion's theta sweep (the preset's is 0.8)",
    )
    p.add_argument("--distribution", default="plummer_clipped")
    p.add_argument("--rmax", type=float, default=20.0)
    p.add_argument(
        "--ic-npz",
        default=None,
        help="Disc + bulge IC (.npz with state0, mass, rdisk_code[, G_code])",
    )
    p.add_argument(
        "--softening",
        type=float,
        default=None,
        help="Default: 1e-7 for a _timing distribution; the disc formula for --ic-npz",
    )
    p.add_argument(
        "--G", dest="g", type=float, default=None, help="Default 1 or G_code"
    )
    p.add_argument("--accuracy-targets", type=int, default=4096)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    runmeta.add_common_args(p)
    p.set_defaults(dtype="float32")
    return p.parse_args()


def _solver(
    GX: Any,
    *,
    softening: float,
    G: float,
    mac_type: str,
    theta: Optional[float],
    eps: Optional[float],
) -> Any:
    """``galaxy_solver`` at one sweep point; only the MAC and its knob move."""
    kw: dict[str, Any] = {"softening": softening, "G": G}
    if mac_type != "dehnen_error":
        kw.update(mac_type=mac_type, theta=theta)
    preset_eps = float(GX.GALAXY.adaptive_eps)
    if eps is None or math.isclose(eps, preset_eps, rel_tol=1e-12):
        return GX.galaxy_solver(**kw)
    solver = GX.galaxy_solver(adaptive_eps=float(eps), **kw)
    got = getattr(getattr(solver, "_impl", solver), "adaptive_eps", None)
    if got is None or not math.isclose(float(got), float(eps), rel_tol=1e-12):
        raise RuntimeError(f"asked for adaptive_eps={eps}, the solver has {got}")
    return solver


def _initial_conditions(args: argparse.Namespace, n: int) -> tuple[Any, Any, dict]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from bench.scaling import _timing as T

    if args.ic_npz:
        ic = np.load(args.ic_npz)
        n_all = int(ic["state0"].shape[0])
        if n > n_all:
            raise SystemExit(f"{args.ic_npz} has {n_all} particles, asked for {n}")
        pos = np.asarray(ic["state0"][:n, 0], np.float32)  # state0 is [N, 2, 3]
        mass = (np.asarray(ic["mass"][:n], np.float64) * (n_all / n)).astype(np.float32)
        g = float(ic["G_code"]) if "G_code" in ic.files else 1.0
        rdisk = float(ic["rdisk_code"])
        soft = (
            float(0.5 * rdisk / np.sqrt(n / 1e5))
            if args.softening is None
            else float(args.softening)
        )
        info = {
            "ic": str(args.ic_npz),
            "ic_particles": n_all,
            "subsample": "prefix of the shuffled IC, masses x N_all / N",
            "softening_rule": (
                "0.5 rdisk / sqrt(N / 1e5)" if args.softening is None else "--softening"
            ),
            "external_field": "the IC's NFW halo is not part of the scored force",
        }
        return (
            jnp.asarray(pos),
            jnp.asarray(mass),
            {
                "softening": soft,
                "G": g if args.g is None else float(args.g),
                **info,
            },
        )
    key = jax.random.PRNGKey(runmeta.seed_sequence(args.seed, args.distribution, n))
    x, m = T.distribution(key, n, args.distribution, jnp.float32, rmax=args.rmax)
    return (
        x,
        m,
        {
            "softening": 1e-7 if args.softening is None else float(args.softening),
            "G": 1.0 if args.g is None else float(args.g),
            "ic": args.distribution,
            "rmax": (
                float(args.rmax) if args.distribution == "plummer_clipped" else None
            ),
        },
    )


def _loglog_at(xs: list[float], ys: list[float], x: float) -> Optional[float]:
    """``y(x)`` by log-log interpolation inside the measured range, else None."""
    import numpy as np

    pts = sorted((a, b) for a, b in zip(xs, ys) if a and b and a > 0 and b > 0)
    if len(pts) < 2 or not x or x <= 0:
        return None
    lx = np.log([a for a, _ in pts])
    ly = np.log([b for _, b in pts])
    if np.log(x) < lx[0] or np.log(x) > lx[-1]:
        return None
    # a non-monotone arm (noise) still interpolates between its sorted neighbours
    return float(np.exp(np.interp(np.log(x), lx, ly)))


def main() -> int:
    args = _parse_args()
    if args.dtype != "float32":
        print(
            "[fig03] preset='galaxy' is fp32 (D3); --dtype float64 is not this figure"
        )
        return 2
    runmeta.select_gpu(args.gpu_select)
    from bench.scaling import _timing as T  # noqa: E402  (imports no jax)

    env = T.launch_env()

    import jax  # noqa: E402
    import numpy as np  # noqa: E402

    from bench.scaling import _galaxy as GX  # noqa: E402
    from jaccpot.runtime.fused_topology import NEEDS_FIELDS  # noqa: E402

    n = int(float(args.n))
    x, m, ic = _initial_conditions(args, n)
    x, m = jax.block_until_ready((x, m))
    soft, G = float(ic["softening"]), float(ic["G"])
    kernel = str(GX.GALAXY.softening_kernel)

    targets = T.sample_targets(n, int(args.accuracy_targets), args.seed)
    print(f"[fig03] fp64 direct reference at {len(targets)} targets, N={n} ...")
    ref = T.direct_reference(
        np.asarray(x), np.asarray(m), targets, softening=soft, kernel=kernel, G=G
    )

    points = [("dehnen_error", None, e) for e in _floats(args.eps)]
    points += [("dehnen", t, None) for t in _floats(args.theta)]
    far_i = NEEDS_FIELDS.index("far_needed")
    near_i = NEEDS_FIELDS.index("near_needed")
    records: list[dict[str, Any]] = []
    for mac_type, theta, eps in points:
        label = f"{mac_type} " + (
            f"eps={eps:g}" if eps is not None else f"theta={theta:g}"
        )
        rec: dict[str, Any] = {
            "arm": "dehnen_error" if mac_type == "dehnen_error" else "geometric",
            "mac_type": mac_type,
            "theta": float(theta if theta is not None else GX.GALAXY.theta),
            "adaptive_eps": eps,
            "n": n,
            "distribution": ic["ic"],
            "card_load_before": GX.card_load(),
        }
        try:
            solver = _solver(
                GX, softening=soft, G=G, mac_type=mac_type, theta=theta, eps=eps
            )
            galaxy = GX.describe(
                solver,
                mac_type=None if mac_type == "dehnen_error" else mac_type,
                theta=theta,
            )
            if eps is not None:
                galaxy["adaptive_eps"] = float(eps)
                if not math.isclose(eps, float(GX.GALAXY.adaptive_eps)):
                    galaxy["off_preset"]["adaptive_eps"] = float(eps)
            rec["galaxy"] = galaxy
            fs, setup = T.warm_fused_step(solver, x, m)
            reps = dict(warmup=int(args.warmup), repeats=int(args.repeats))
            s = T.time_min_repeat(lambda: fs.step(x, m), **reps)
            e = T.time_min_repeat(lambda: fs.evaluate(), **reps)
            acc, aux = fs.step(x, m)
            acc_t = np.asarray(jax.device_get(acc), np.float64)[targets]
            ok = bool(fs.ok(aux)) and bool(setup["first_step_ok"])
            needs = np.asarray(aux[2])
            rec.update(
                {
                    "valid": ok,
                    "overflow_bits": int(aux[1]),
                    "step": {"min_s": s[0], "mean_s": s[1], "std_s": s[2]},
                    "evaluate": {"min_s": e[0], "mean_s": e[1], "std_s": e[2]},
                    "rel_l2": T.aggregate_rel_l2(acc_t, ref),
                    "percentiles": T.per_particle_errors(acc_t, ref),
                    "far_pairs": int(needs[far_i]),
                    "near_pairs": int(needs[near_i]),
                    "setup": setup,
                }
            )
            fs = acc = None  # free the provider before the next point
        except Exception as exc:
            rec.update({"valid": False, "error": str(exc)[:400]})
            print(f"  {label}: FAILED {str(exc)[:200]}")
        rec["card_load_after"] = GX.card_load()
        if rec.get("valid") and not GX.card_free(
            rec.get("card_load_before"), rec["card_load_after"]
        ):
            rec["valid"] = False
            rec["invalid_reason"] = "contended card"
        records.append(rec)
        if rec.get("step"):
            print(
                f"  {label:<24s} valid={rec['valid']} step={rec['step']['min_s'] * 1e3:9.2f} ms "
                f"rel-L2={rec['rel_l2']:.3e} p99={rec['percentiles']['p99']:.3e} "
                f"far={rec['far_pairs']} near={rec['near_pairs']}",
                flush=True,
            )

    valid = [r for r in records if r.get("valid")]
    geo = [r for r in valid if r["arm"] == "geometric"]
    geo_err = [r["rel_l2"] for r in geo]
    geo_t = [r["step"]["min_s"] for r in geo]
    comparisons: list[dict[str, Any]] = []
    for r in (r for r in valid if r["arm"] == "dehnen_error"):
        t_geo = _loglog_at(geo_err, geo_t, r["rel_l2"])
        e_geo = _loglog_at(geo_t, geo_err, r["step"]["min_s"])
        comparisons.append(
            {
                "adaptive_eps": r["adaptive_eps"],
                "rel_l2": r["rel_l2"],
                "step_s": r["step"]["min_s"],
                "geometric_step_s_at_matched_error": t_geo,
                "cost_ratio": (r["step"]["min_s"] / t_geo) if t_geo else None,
                "geometric_rel_l2_at_matched_cost": e_geo,
                "error_ratio": (e_geo / r["rel_l2"]) if e_geo else None,
            }
        )

    preset_pair = None
    mass_p = [
        r
        for r in valid
        if r["arm"] == "dehnen_error"
        and math.isclose(r["adaptive_eps"], float(GX.GALAXY.adaptive_eps))
    ]
    geo_p = [r for r in geo if math.isclose(r["theta"], float(GX.GALAXY.theta))]
    if mass_p and geo_p:
        a, b = mass_p[0], geo_p[0]
        preset_pair = {
            "dehnen_error_eps": a["adaptive_eps"],
            "geometric_theta": b["theta"],
            "step_s": {
                "dehnen_error": a["step"]["min_s"],
                "geometric": b["step"]["min_s"],
            },
            "rel_l2": {"dehnen_error": a["rel_l2"], "geometric": b["rel_l2"]},
            "cost_ratio": a["step"]["min_s"] / b["step"]["min_s"],
            "error_ratio": b["rel_l2"] / a["rel_l2"],
            "reference_25M_disc": {"cost_ratio": 1.12, "error_ratio": 2.8},
        }
        print(
            f"\npreset pair: dehnen_error 1e-4 costs {preset_pair['cost_ratio']:.2f}x "
            f"geometric theta 0.8 for {preset_pair['error_ratio']:.2f}x lower rel-L2 "
            f"(reference, 25M disc: 1.12x for 2.8x)"
        )
    for c in comparisons:
        cr, er = c["cost_ratio"], c["error_ratio"]
        print(
            f"  eps={c['adaptive_eps']:g}: cost ratio at matched error "
            + (f"{cr:.2f}" if cr else "n/a (outside the geometric range)")
            + ", error ratio at matched cost "
            + (f"{er:.2f}" if er else "n/a (outside the geometric range)")
        )

    config = {
        "n": n,
        # theta is the geometric arm's knob and eps the other's: both grids
        "theta": _floats(args.theta),
        "eps": _floats(args.eps),
        "order": int(GX.GALAXY.max_order),
        "basis": "real",
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": "float32",
        "leaf_size": int(GX.GALAXY.leaf_size),
        "preset": "galaxy",
        "distribution": ic["ic"],
        "ic": ic,
        "softening": soft,
        "softening_kernel": kernel,
        "G": G,
        "accuracy_targets": int(len(targets)),
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "launch_env": env,
        "arms": {
            "dehnen_error": "mac_type='dehnen_error' (the preset's), eq (16a) in the "
            "flat walk, swept over adaptive_eps",
            "geometric": "mac_type='dehnen' (geometric), swept over theta",
        },
        "error": "aggregate rel-L2 of the step's acceleration against an fp64 "
        "direct sum (same softening kernel) at the sampled targets",
        "cost": "FusedForceProvider.step jitted (rebuild + evaluate), min over "
        "repeats, steady-state f_b",
        "retired": "the September 'median 78x at equal interaction count' is not "
        "produced here (another lane, a pair-work statement)",
    }
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config=config,
        meta=runmeta.run_meta({"argv": sys.argv[1:]}),
        data={
            "records": records,
            "comparisons": comparisons,
            "preset_pair": preset_pair,
        },
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
