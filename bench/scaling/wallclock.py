"""Figure 04 -- wall clock per step vs N on the fused lane (``preset="galaxy"``).

D3: sections 4-7 time what users run, ``FastMultipoleMethod(preset="galaxy")`` --
the fused single-card lane, cell leaves of 64, p6, ``dehnen_error`` at
``adaptive_eps`` 1e-4, ferrers3, fp32 -- built by ``bench/scaling/_galaxy.py`` so
every single-card figure is on one configuration. The September version timed
``preset="large_n_gpu"`` on ``prepare_state``/``evaluate_prepared_state`` with the
jaxFMM paper's p/theta sets; none of that is the production path any more.

Timing protocol (``bench/scaling/_timing.py``): warm up, then the **minimum** over
repeats with the result blocked on device; the mean and spread are recorded so a
noisy point is visible. ``nvidia-smi``'s compute processes are recorded before and
after each N (``card_load_before`` / ``card_load_after``): a row measured beside
another job says so. The launch environment is production's (fusion autotuner off).

What is timed, and what is not
------------------------------
* ``jaccpot_step`` -- **the headline**. ``FusedForceProvider.step`` jitted: the
  topology rebuilt at the positions (tree, P2M, M2M, flat walk, M2L, L2L), then
  the evaluation (P2P, L2P, scatter), with the capacity verdict. That is what a
  simulation pays per step. Its f_b comes from the previous step's evaluation (the
  steady state; ``_timing.warm_fused_step``).
* ``jaccpot_evaluate`` -- evaluation only, on the prepare's topology (no rebuild):
  the secondary series, so the rebuild's share is readable off the figure.
* ``jaccpot_potential`` -- the potential on the prepare's topology (its own O(N)
  program): only the like-for-like partner of jaxFMM's potential below.
* ``direct_acceleration`` -- the O(N^2) sum with the same pair law (ferrers3) in
  fp32, jitted over every target, up to ``--direct-max-n``.
* NOT timed: the eager ``prepare`` (setup, paid once per run and per segment
  retry, and dominated by compiles). Its wall time is recorded per row as
  ``setup.prepare_s`` -- one sample, compiles included -- and is not a series.

Every input is a jit ARGUMENT, never a closed-over constant. A row whose capacity
verdict fails (``FusedStep.ok`` False: the force is WRONG, not slow) is recorded
with ``valid: false`` and is not fitted; the notebook must not plot it.

Accuracy
--------
Up to ``--accuracy-max-n`` every row scores the step's acceleration (and the
potential) against an fp64 direct sum with the solver's own softening kernel on
``--accuracy-targets`` seeded targets (all of them below that count): aggregate
rel-L2 and the per-target percentiles.

jaxFMM, at matched accuracy only
--------------------------------
jaxFMM evaluates a **potential**, unsoftened, with the Laplace kernel 1/(4 pi r).
A wall clock at mismatched accuracy is not a comparison, so at every N up to
``--accuracy-max-n`` this script runs each ``(p, theta)`` of ``--jaxfmm-grid``,
scores its potential (x 4 pi, against an fp64 unsoftened direct potential) and
picks the cheapest candidate whose error is at most jaccpot's potential error x
(1 + ``--jaxfmm-tolerance``). Every candidate's time and error is recorded; the
matched one is the ``jaxfmm_potential`` / ``jaxfmm_potential_full`` series
(prebuilt hierarchy / hierarchy + evaluation). Above ``--accuracy-max-n`` the last
matched ``(p, theta)`` is reused and the row says at which N it was matched. If no
candidate reaches jaccpot's accuracy the most accurate one is timed and the row
says ``matched: false``. Pairs: ``jaccpot_potential`` vs ``jaxfmm_potential``
(evaluation on a prebuilt structure), ``jaccpot_step`` vs ``jaxfmm_potential_full``
(rebuild + evaluate; jaccpot's computes the acceleration and f_b, more work).
jaxFMM is not a dependency: without it the arm is skipped and the config says so.

jz-fmm (D5)
-----------
jz-fmm is citable (arXiv:2609.09307) but there is no importable runner for it in
this repository: the head-to-head of 2026-10 ran it from its own virtualenv through
the Odisseo bench harness (``bench/results/fused_memory/round5/compare_jzfmm/``,
``codes/jzfmm_force_eval.py`` there). This script has no jz-fmm arm; a paper
comparison must come from that harness at the same IC, N and accuracy targets.

Small N
-------
``preset="galaxy"`` fixes ``cell_min_level=8`` (256^3 Morton cells), tuned for
N ~ 1e7-1e8. Below a few million particles most cell leaves hold one or a few
particles and nearly every pair is far (measured on CPU, a unit normal draw:
8168 live leaves at N = 8192 and 20 directed near pairs). The small-N end of the
ladder times that degenerate tree; read the figure's left end as such.

Usage
-----
CPU smoke (the CPU route; interpreted kernels, timings mean nothing)::

    JAX_PLATFORMS=cpu python -m bench.scaling.wallclock --n 2048,4096 \\
        --repeats 2 --warmup 1 --direct-max-n 4096 --accuracy-max-n 4096 \\
        --accuracy-targets 512 --fit-min-n 0 --gpu-select none \\
        --json-out /tmp/smoke.json

Paper run (one free A100; the default ladder reaches 1.9e8)::

    python -m bench.scaling.wallclock
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

DEFAULT_OUT = "scaling/wallclock_vs_n.json"
#: 1.9e8 is the one-card ceiling row of the production tables.
DEFAULT_N = "1e4,3e4,1e5,3e5,1e6,3e6,1e7,3e7,1e8,1.9e8"
#: jaxFMM candidates, "p:theta": its paper's sets (p3/p6 at theta 0.77, p3 at 0.5)
#: plus the higher orders an fp32 galaxy force may need to be matched at all.
DEFAULT_JAXFMM_GRID = "3:0.77,3:0.5,4:0.6,6:0.77,6:0.5,8:0.5,10:0.4"
SET_NAME = "galaxy"


def _parse_n(text: str) -> list[int]:
    return [int(float(v)) for v in str(text).split(",") if v.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n",
        default=None,
        help=f"Comma-separated N, 1e6 notation allowed (default: {DEFAULT_N})",
    )
    p.add_argument(
        "--n-min-exp",
        type=int,
        default=None,
        help="Instead of --n: a log-even ladder from 2**n-min-exp ...",
    )
    p.add_argument(
        "--n-max-exp", type=int, default=None, help="... to 2**n-max-exp ..."
    )
    p.add_argument("--n-steps", type=int, default=None, help="... in this many steps")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--distribution",
        default="plummer_clipped",
        help="bench.scaling._timing.distribution (default: the production rows' IC)",
    )
    p.add_argument("--rmax", type=float, default=20.0, help="plummer_clipped cut")
    p.add_argument(
        "--softening",
        type=float,
        default=1e-7,
        help="ferrers3 softening (default 1e-7, the clipped-Plummer production rows')",
    )
    p.add_argument("--G", dest="g", type=float, default=1.0)
    p.add_argument(
        "--direct-max-n",
        type=int,
        default=1 << 17,
        help="Skip the O(N^2) direct-sum timing above this N (default: 131072)",
    )
    p.add_argument(
        "--accuracy-max-n",
        type=int,
        default=1 << 24,
        help="Skip the fp64 accuracy score and the jaxFMM matching above this N",
    )
    p.add_argument(
        "--accuracy-targets",
        type=int,
        default=4096,
        help="Seeded targets the fp64 direct sum is evaluated at (default: 4096)",
    )
    p.add_argument(
        "--fit-min-n",
        type=int,
        default=1_000_000,
        help=(
            "Exclude N below this from the fitted exponents: below it a GPU is "
            "launch-bound and the galaxy tree degenerate (docstring, 'Small N')"
        ),
    )
    p.add_argument("--skip-jaxfmm", action="store_true")
    p.add_argument("--jaxfmm-grid", default=DEFAULT_JAXFMM_GRID)
    p.add_argument("--jaxfmm-max-n", type=int, default=1 << 25)
    p.add_argument("--jaxfmm-tolerance", type=float, default=0.10)
    runmeta.add_common_args(p)
    p.set_defaults(dtype="float32")
    return p.parse_args()


def _ladder(args: argparse.Namespace) -> list[int]:
    from bench.scaling import _timing as T

    if args.n is not None:
        return _parse_n(args.n)
    if args.n_min_exp is not None or args.n_max_exp is not None:
        lo = args.n_min_exp if args.n_min_exp is not None else args.n_max_exp
        hi = args.n_max_exp if args.n_max_exp is not None else lo
        return T.n_values(lo, hi, args.n_steps or (hi - lo + 1))
    return _parse_n(DEFAULT_N)


def _import_jaxfmm() -> tuple[Any, Any]:
    try:
        from jaxfmm import eval_potential, gen_hierarchy
    except ImportError:  # jaxFMM < 0.3 kept these in submodules
        from jaxfmm.fmm import eval_potential
        from jaxfmm.hierarchy import gen_hierarchy
    return gen_hierarchy, eval_potential


def main() -> int:
    args = _parse_args()
    if args.dtype != "float32":
        print(
            "[fig04] preset='galaxy' is fp32 (D3); --dtype float64 is not this figure"
        )
        return 2
    runmeta.select_gpu(args.gpu_select)
    from bench.scaling import _timing as T  # noqa: E402  (imports no jax)

    env = T.launch_env()

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402

    from bench.scaling import _galaxy as GX  # noqa: E402

    have_jaxfmm = False
    jaxfmm_note = "skipped (--skip-jaxfmm)"
    if not args.skip_jaxfmm:
        try:
            gen_hierarchy, eval_potential = _import_jaxfmm()
            have_jaxfmm = True
            jaxfmm_note = "available"
        except Exception as exc:
            jaxfmm_note = f"unavailable: {str(exc)[:160]}"
            print(f"[fig04] jaxFMM {jaxfmm_note}; its arm is skipped")
    grid = [
        (int(c.split(":")[0]), float(c.split(":")[1]))
        for c in str(args.jaxfmm_grid).split(",")
        if c.strip()
    ]

    ns = _ladder(args)
    kernel = str(GX.GALAXY.softening_kernel)
    records: list[dict[str, Any]] = []
    matched_params: Optional[tuple[int, float]] = None
    matched_at_n: Optional[int] = None
    galaxy: dict[str, Any] = {}

    for n in ns:
        key = jax.random.PRNGKey(runmeta.seed_sequence(args.seed, args.distribution, n))
        x, m = T.distribution(key, n, args.distribution, jnp.float32, rmax=args.rmax)
        x, m = jax.block_until_ready((x, m))
        row: dict[str, Any] = {
            "n": int(n),
            "param_set": SET_NAME,
            "p": int(GX.GALAXY.max_order),
            "theta": float(GX.GALAXY.theta),
            "timings": {},
            "accuracy": {},
            "card_load_before": GX.card_load(),
        }

        def _time(label: str, fn, *, store: Optional[dict] = None) -> Optional[float]:
            target = row["timings"] if store is None else store
            try:
                tmin, tmean, tstd = T.time_min_repeat(
                    fn, warmup=int(args.warmup), repeats=int(args.repeats)
                )
            except Exception as exc:
                print(f"  [{label}] N={n} failed: {str(exc)[:160]}")
                target[label] = {
                    "min_s": None,
                    "mean_s": None,
                    "std_s": None,
                    "error": str(exc)[:300],
                }
                return None
            target[label] = {"min_s": tmin, "mean_s": tmean, "std_s": tstd}
            return tmin

        # -- jaccpot, the fused lane ------------------------------------------
        try:
            solver = GX.galaxy_solver(softening=float(args.softening), G=float(args.g))
            galaxy = galaxy or GX.describe(solver)
            fs, setup = T.warm_fused_step(solver, x, m)
            row["setup"] = setup
        except Exception as exc:
            print(f"N={n:<10d} prepare FAILED: {str(exc)[:200]}")
            row["valid"] = False
            row["error"] = str(exc)[:400]
            row["card_load_after"] = GX.card_load()
            records.append(row)
            continue

        _time("jaccpot_step", lambda: fs.step(x, m))
        _time("jaccpot_evaluate", lambda: fs.evaluate())
        # the verdict of the step the series timed (same inputs, same program)
        acc, aux = fs.step(x, m)
        acc = jax.block_until_ready(acc)
        row["valid"] = bool(fs.ok(aux)) and bool(setup["first_step_ok"])
        row["overflow_bits"] = int(aux[1])
        if not row["valid"]:
            print(f"N={n:<10d} capacity verdict FAILED (bits {int(aux[1]):#x})")

        want_accuracy = n <= int(args.accuracy_max_n)
        jaxfmm_here = have_jaxfmm and n <= int(args.jaxfmm_max_n)
        if want_accuracy or jaxfmm_here:
            _time("jaccpot_potential", lambda: fs.potential())

        targets = None
        acc_t = pot_t = None
        if want_accuracy:
            targets = T.sample_targets(n, int(args.accuracy_targets), args.seed)
            acc_t = np.asarray(acc, np.float64)[targets]
            pot_t = np.asarray(fs.potential(), np.float64)[targets]
        del acc

        # -- direct sum, O(N^2) ------------------------------------------------
        if n <= int(args.direct_max_n):
            direct = T.direct_sum_fn(
                softening=float(args.softening), kernel=kernel, G=float(args.g), n=n
            )
            _time("direct_acceleration", lambda: direct(x, m))

        # -- accuracy against the fp64 direct sum -----------------------------
        if want_accuracy:
            x_host = np.asarray(x)
            m_host = np.asarray(m)
            ref_acc, ref_pot = T.direct_reference(
                x_host,
                m_host,
                targets,
                softening=float(args.softening),
                kernel=kernel,
                G=float(args.g),
                potential=True,
            )
            row["accuracy"] = {
                "targets": int(len(targets)),
                "jaccpot_acceleration_rel_l2": T.aggregate_rel_l2(acc_t, ref_acc),
                "jaccpot_acceleration_percentiles": T.per_particle_errors(
                    acc_t, ref_acc
                ),
                "jaccpot_potential_rel_l2": T.aggregate_rel_l2(pot_t, ref_pot),
                "reference": f"fp64 direct sum, {kernel}, softening {args.softening:g}",
            }

        # -- jaxFMM at matched accuracy ---------------------------------------
        if jaxfmm_here:
            jrow: dict[str, Any] = {"candidates": []}
            ref_newton = None
            if want_accuracy:
                # jaxFMM is unsoftened: an unsoftened fp64 reference for its arm
                _, pot_newton = T.direct_reference(
                    x_host,
                    m_host,
                    targets,
                    softening=0.0,
                    kernel="plummer",
                    G=1.0,
                    potential=True,
                )
                ref_newton = -pot_newton  # sum_j m_j / r_ij
            if want_accuracy:
                candidates = grid
            else:
                candidates = [matched_params] if matched_params else []
            for p_j, th_j in candidates:
                cand: dict[str, Any] = {"p": p_j, "theta": th_j, "timings": {}}
                try:
                    hier = gen_hierarchy(x, p=p_j, theta=th_j, s=3, N_max=128)
                    _time(
                        "potential",
                        lambda: eval_potential(m, **hier),
                        store=cand["timings"],
                    )
                    _time(
                        "potential_full",
                        lambda: eval_potential(
                            m, **gen_hierarchy(x, p=p_j, theta=th_j, s=3, N_max=128)
                        ),
                        store=cand["timings"],
                    )
                    if ref_newton is not None:
                        pot = np.asarray(eval_potential(m, **hier), np.float64)
                        # jaxFMM's Green's function is 1/(4 pi r): undo the factor
                        cand["potential_rel_l2"] = T.aggregate_rel_l2(
                            4.0 * math.pi * pot[targets], ref_newton
                        )
                    del hier
                except Exception as exc:
                    cand["error"] = str(exc)[:300]
                jrow["candidates"].append(cand)

            best = None
            if want_accuracy:
                goal = row["accuracy"]["jaccpot_potential_rel_l2"] * (
                    1.0 + float(args.jaxfmm_tolerance)
                )
                jrow["goal_potential_rel_l2"] = goal
                scored = [
                    c
                    for c in jrow["candidates"]
                    if c.get("potential_rel_l2") is not None
                    and c["timings"].get("potential", {}).get("min_s")
                ]
                # The score assumes eval_potential returns input order (never
                # exercised here: jaxFMM is not installed on the dev host). If it
                # returned its own sorted order every candidate would score ~1;
                # flag that rather than time a "best" that is not a match.
                if scored and min(c["potential_rel_l2"] for c in scored) > 0.5:
                    jrow["order_suspect"] = True
                within = [c for c in scored if c["potential_rel_l2"] <= goal]
                if within:
                    best = min(within, key=lambda c: c["timings"]["potential"]["min_s"])
                    jrow["matched"] = True
                elif scored:
                    best = min(scored, key=lambda c: c["potential_rel_l2"])
                    jrow["matched"] = False
                if best is not None:
                    matched_params = (best["p"], best["theta"])
                    matched_at_n = int(n)
            elif jrow["candidates"]:
                best = jrow["candidates"][0]
                jrow["matched"] = None  # matched at matched_at_n, not here
            if best is not None:
                jrow["p"], jrow["theta"] = best["p"], best["theta"]
                jrow["matched_at_n"] = matched_at_n
                jrow["potential_rel_l2"] = best.get("potential_rel_l2")
                for src, dst in (
                    ("potential", "jaxfmm_potential"),
                    ("potential_full", "jaxfmm_potential_full"),
                ):
                    if src in best["timings"]:
                        row["timings"][dst] = best["timings"][src]
            row["jaxfmm"] = jrow

        row["card_load_after"] = GX.card_load()
        if row.get("valid") and not GX.card_free(
            row.get("card_load_before"), row["card_load_after"]
        ):
            row["valid"] = False
            row["invalid_reason"] = "contended card"
        records.append(row)
        fs = None  # free the provider before the next N
        times = " ".join(
            f"{k}={v['min_s'] * 1e3:.2f}ms" if v.get("min_s") else f"{k}=ERR"
            for k, v in row["timings"].items()
        )
        err = row["accuracy"].get("jaccpot_acceleration_rel_l2")
        err_s = f" rel-L2={err:.3e}" if err is not None else ""
        print(f"N={n:<10d} valid={row['valid']} {times}{err_s}", flush=True)

    # -- fitted exponents, over valid rows only ------------------------------
    fits: dict[str, dict[str, Any]] = {}
    valid = [r for r in records if r.get("valid")]
    for label in sorted({k for r in valid for k in r["timings"]}):
        pts = [
            (r["n"], r["timings"][label]["min_s"])
            for r in valid
            if r["timings"].get(label, {}).get("min_s")
            # a jaxFMM time is a comparison only at matched accuracy, and only at
            # an N where the match was measured (not reused from a smaller N)
            and (
                not label.startswith("jaxfmm")
                or (
                    (r.get("jaxfmm") or {}).get("matched") is True
                    and (r.get("jaxfmm") or {}).get("matched_at_n") == r["n"]
                )
            )
        ]
        if len(pts) >= 2:
            fits[f"{SET_NAME}:{label}"] = T.fit_log_log_exponent(
                [a for a, _ in pts], [b for _, b in pts], min_n=int(args.fit_min_n)
            )

    print("\n=== fitted exponents (log t = alpha log N + c) ===")
    for name, fit in fits.items():
        print(
            f"  {name:<40s} alpha={fit['exponent']:.3f} "
            f"R^2={fit['r_squared']:.4f} over {fit['n_points']} pts "
            f"N={fit.get('fit_min_n')}..{fit.get('fit_max_n')}"
        )

    config = {
        "n": ns,
        "theta": float(GX.GALAXY.theta),
        "order": int(GX.GALAXY.max_order),
        "basis": "real",
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": "float32",
        "leaf_size": int(GX.GALAXY.leaf_size),
        "preset": "galaxy",
        "galaxy": galaxy,
        "distribution": args.distribution,
        "rmax": float(args.rmax) if args.distribution == "plummer_clipped" else None,
        "softening": float(args.softening),
        "softening_kernel": kernel,
        "G": float(args.g),
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "direct_max_n": int(args.direct_max_n),
        "accuracy_max_n": int(args.accuracy_max_n),
        "accuracy_targets": int(args.accuracy_targets),
        "fit_min_n": int(args.fit_min_n),
        "launch_env": env,
        "timed_region": {
            "jaccpot_step": "FusedForceProvider.step, jitted: rebuild + evaluate "
            "(the per-step cost); f_b from the previous step",
            "jaccpot_evaluate": "evaluation only on the prepare's topology",
            "jaccpot_potential": "potential only on the prepare's topology",
            "direct_acceleration": "fp32 O(N^2) sum, same pair law, every target",
            "jaxfmm_potential": "matched candidate, eval_potential on a prebuilt "
            "hierarchy",
            "jaxfmm_potential_full": "matched candidate, gen_hierarchy + "
            "eval_potential",
            "not_timed": "the eager prepare (setup; setup.prepare_s per row, one "
            "sample, compiles included)",
        },
        "jaxfmm": {
            "status": jaxfmm_note,
            "grid": [{"p": p_, "theta": t_} for p_, t_ in grid],
            "tolerance": float(args.jaxfmm_tolerance),
            "max_n": int(args.jaxfmm_max_n),
            "kernel": "1/(4 pi r), unsoftened; x 4 pi against an unsoftened fp64 "
            "direct potential",
            "s": 3,
            "N_max": 128,
        },
        "jzfmm": "no runner in this repository (D5); see the module docstring",
        "valid_rule": "rows with valid=false failed the capacity verdict: the "
        "force is wrong; never plot or fit them",
    }
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config=config,
        meta=runmeta.run_meta({"argv": sys.argv[1:]}),
        data={"records": records, "fits": fits},
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
