"""Figure 04 -- wall-clock vs N (log-log), jaccpot vs direct sum vs jaxFMM.

Sweeps a log-even N ladder and times each runner with the shared protocol from
``bench/scaling/_timing.py`` (the same one ``bench/bench_jaxfmm_paper_compare.py``
uses): warm up, then take the minimum over repeats with the result blocked on
device. The minimum because this is a shared GPU, where a mean measures other
users' jobs; the spread is recorded so a noisy point is visible.

Comparing like with like
-----------------------
jaxFMM evaluates a **potential**; jaccpot's production output is an
**acceleration**, which costs strictly more (three components, from the gradient
of the expansion). Timing one against the other and calling it a head-to-head
would be meaningless, so this script measures both jaccpot paths separately:

* ``jaccpot_potential`` vs ``jaxfmm_potential`` -- the fair head-to-head.
* ``jaccpot_acceleration`` vs ``direct_acceleration`` -- the O(N log N) against
  O(N^2) crossover, which is the figure's other story.

What is timed, and what is not
------------------------------
The timed region is **evaluation on an already-built tree** for jaccpot and
jaxFMM alike (``evaluate_prepared_state`` / ``eval_potential`` on a hierarchy
built outside the loop), which is what a simulation pays per step when topology
is reused. Because that excludes setup, ``jaccpot_acceleration_full`` also
records ``compute_accelerations`` end to end -- build plus evaluate -- so the
figure cannot be read as claiming the per-step number covers tree construction.

A speed comparison without an accuracy comparison is not a result, so at every N
up to ``--accuracy-max-n`` the achieved error against the direct sum is recorded
for both FMM runners. Above that the O(N^2) reference is unaffordable and the
error fields are null.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu python -m bench.scaling.wallclock \\
        --n-min-exp 10 --n-max-exp 11 --n-steps 2 --repeats 3 \\
        --gpu-select none --json-out /tmp/smoke.json

Paper run::

    python -m bench.scaling.wallclock --n-min-exp 11 --n-max-exp 22 --n-steps 12
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

# The jaxFMM paper's parameter sets, as encoded in bench_jaxfmm_paper_compare.py.
# Set 4 varies the split parameter s, which has no direct jaccpot control, so it
# is excluded here rather than compared unfairly.
PARAM_SETS = {
    "1_default": {"p": 3, "theta": 0.77, "s": 3},
    "2_high_p": {"p": 6, "theta": 0.77, "s": 3},
    "3_low_theta": {"p": 3, "theta": 0.50, "s": 3},
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n-min-exp", type=int, default=11)
    p.add_argument("--n-max-exp", type=int, default=22)
    p.add_argument("--n-steps", type=int, default=12)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--param-set", default="1_default", choices=tuple(PARAM_SETS) + ("all",)
    )
    p.add_argument("--basis", default="real")
    # `large_n_gpu`, not `accurate`. Measured on an A100 at N=16384, leaf=64, p=4,
    # evaluating a prebuilt tree: accurate 27.3 s vs large_n_gpu 197 ms, a factor
    # of 139, in steady state rather than compilation. `accurate` does not take
    # the radix fast lane, so timing it on a GPU would publish a preset artefact
    # as a scaling result. The accuracy figures (01-03) stay on `accurate`.
    p.add_argument("--preset", default="large_n_gpu")
    p.add_argument(
        "--leaf-size",
        type=int,
        default=128,
        help="jaccpot leaf size; 128 matches jaxFMM's N_max=128 (default: 128)",
    )
    p.add_argument("--distribution", default="uniform_cube")
    p.add_argument(
        "--full-repeats",
        type=int,
        default=3,
        help=(
            "Repeats for the build+evaluate series only. It is 40x more expensive "
            "per call than the evaluate-only series (measured 8.5 s vs 0.19 s at "
            "N=2048), because it rebuilds the tree every call, and at --repeats it "
            "dominates the entire sweep's wall time. It is context for the "
            "per-step numbers rather than the headline, so it gets a smaller "
            "sample -- recorded in the config (default: 3)"
        ),
    )
    p.add_argument(
        "--direct-max-n",
        type=int,
        default=1 << 15,
        help="Skip the O(N^2) direct-sum timing above this N (default: 32768)",
    )
    p.add_argument(
        "--full-max-n",
        type=int,
        default=1 << 17,
        help="Skip the build+evaluate series above this N (default: 131072)",
    )
    p.add_argument(
        "--accuracy-max-n",
        type=int,
        default=1 << 14,
        help="Skip the accuracy cross-check above this N (default: 16384)",
    )
    p.add_argument(
        "--fit-min-n",
        type=int,
        default=1 << 14,
        help=(
            "Exclude N below this from the fitted exponent. Below it a GPU is "
            "launch-overhead bound, which biases the exponent low (default: 16384)"
        ),
    )
    p.add_argument("--softening", type=float, default=1e-3)
    p.add_argument("--G", dest="g", type=float, default=1.0)
    p.add_argument("--skip-jaxfmm", action="store_true")
    runmeta.add_common_args(p)
    p.set_defaults(dtype="float32")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    runmeta.select_gpu(args.gpu_select)
    runmeta.enable_x64(args.dtype)

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402

    from bench.scaling import _timing as T  # noqa: E402
    from bench.validation import _harness as H  # noqa: E402
    from jaccpot import FastMultipoleMethod  # noqa: E402

    have_jaxfmm = False
    if not args.skip_jaxfmm:
        try:
            try:
                from jaxfmm import eval_potential, gen_hierarchy
            except ImportError:  # jaxFMM < 0.3 kept these in submodules
                from jaxfmm.fmm import eval_potential
                from jaxfmm.hierarchy import gen_hierarchy
            have_jaxfmm = True
        except Exception as exc:
            print(f"[fig04] jaxFMM unavailable ({exc}); its arm will be skipped")

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    ns = T.n_values(args.n_min_exp, args.n_max_exp, args.n_steps)
    sets = tuple(PARAM_SETS) if args.param_set == "all" else (args.param_set,)

    records: list[dict[str, Any]] = []
    for set_name in sets:
        ps = PARAM_SETS[set_name]
        for n in ns:
            key = jax.random.PRNGKey(runmeta.seed_sequence(args.seed, set_name, n))
            points, charges = T.distribution(key, n, args.distribution, dtype)
            points = jax.block_until_ready(points)

            row: dict[str, Any] = {
                "n": int(n),
                "param_set": set_name,
                "p": ps["p"],
                "theta": ps["theta"],
                "s": ps["s"],
                "timings": {},
                "accuracy": {},
            }

            solver = FastMultipoleMethod(
                preset=args.preset,
                basis=args.basis,
                theta=float(ps["theta"]),
                softening=float(args.softening),
                G=float(args.g),
            )

            def _time(
                label: str,
                fn,
                repeats: Optional[int] = None,
                warmup: Optional[int] = None,
            ) -> Optional[float]:
                reps = int(args.repeats) if repeats is None else int(repeats)
                warm = int(args.warmup) if warmup is None else int(warmup)
                try:
                    tmin, tmean, tstd = T.time_min_repeat(fn, warmup=warm, repeats=reps)
                except Exception as exc:
                    print(f"  [{label}] N={n} failed: {str(exc)[:120]}")
                    row["timings"][label] = {
                        "min_s": None,
                        "mean_s": None,
                        "std_s": None,
                        "error": str(exc)[:300],
                    }
                    return None
                row["timings"][label] = {
                    "min_s": tmin,
                    "mean_s": tmean,
                    "std_s": tstd,
                }
                return tmin

            # -- jaccpot, evaluation on a prebuilt tree -----------------------
            prepared = None
            try:
                prepared = solver.prepare_state(
                    points,
                    charges,
                    leaf_size=int(args.leaf_size),
                    max_order=int(ps["p"]),
                    theta=float(ps["theta"]),
                )
            except Exception as exc:
                print(f"  [jaccpot prepare] N={n} failed: {str(exc)[:120]}")

            if prepared is not None:
                _time(
                    "jaccpot_potential",
                    lambda: solver.evaluate_prepared_state(
                        prepared, return_potential=True
                    )[1],
                )
                _time(
                    "jaccpot_acceleration",
                    lambda: solver.evaluate_prepared_state(prepared),
                )

            # Build + evaluate, so the figure cannot be read as claiming the
            # per-step number includes tree construction. Fewer repeats and its
            # own N cap: it rebuilds the tree on every call, which makes it ~40x
            # the cost of the evaluate-only series and would otherwise set the
            # wall time of the whole sweep.
            if n <= int(args.full_max_n):
                _time(
                    "jaccpot_acceleration_full",
                    lambda: solver.compute_accelerations(
                        points,
                        charges,
                        leaf_size=int(args.leaf_size),
                        max_order=int(ps["p"]),
                    ),
                    repeats=int(args.full_repeats),
                    warmup=1,
                )

            # -- jaxFMM, evaluation on a prebuilt hierarchy -------------------
            if have_jaxfmm:
                try:
                    hier = gen_hierarchy(
                        points,
                        p=int(ps["p"]),
                        theta=float(ps["theta"]),
                        s=int(ps["s"]),
                        N_max=128,
                    )
                    _time("jaxfmm_potential", lambda: eval_potential(charges, **hier))
                except Exception as exc:
                    print(f"  [jaxfmm] N={n} failed: {str(exc)[:120]}")
                    row["timings"]["jaxfmm_potential"] = {
                        "min_s": None,
                        "mean_s": None,
                        "std_s": None,
                        "error": str(exc)[:300],
                    }

            # -- direct sum, O(N^2) ------------------------------------------
            if n <= int(args.direct_max_n):
                _time(
                    "direct_acceleration",
                    lambda: H.chunked_direct_accelerations(
                        points, charges, softening=args.softening, G=args.g
                    ),
                )

            # -- accuracy cross-check ----------------------------------------
            if n <= int(args.accuracy_max_n):
                reference = jax.block_until_ready(
                    H.chunked_direct_accelerations(
                        points, charges, softening=args.softening, G=args.g
                    )
                )
                if prepared is not None:
                    accel = jax.block_until_ready(
                        solver.evaluate_prepared_state(prepared)
                    )
                    row["accuracy"]["jaccpot_acceleration_rel_l2"] = H.rel_l2(
                        accel, reference
                    )
                # jaxFMM returns a potential, so its accuracy is checked against
                # a direct-sum potential rather than against the acceleration.
                #
                # Two convention mismatches have to be undone first, or jaxFMM
                # reads as catastrophically wrong when it is not:
                #  * it evaluates the Laplace Green's function 1/(4 pi r), not
                #    sum q/r. Unscaled, the rel-L2 comes out at exactly
                #    1 - 1/(4 pi) = 0.9204 regardless of p or theta -- the
                #    signature of a constant factor, not of an inaccurate solver.
                #  * it has no softening, so the reference for *its* arm uses a
                #    negligible epsilon rather than the solver's.
                # With both applied it lands at 5.6e-7 at p=6/theta=0.5.
                if have_jaxfmm:
                    try:
                        pot_ref = _direct_potential(
                            points, charges, softening=1e-8, G=args.g
                        )
                        hier = gen_hierarchy(
                            points,
                            p=int(ps["p"]),
                            theta=float(ps["theta"]),
                            s=int(ps["s"]),
                            N_max=128,
                        )
                        pot = jax.block_until_ready(eval_potential(charges, **hier))
                        pot_scaled = np.asarray(pot, dtype=np.float64) * (4.0 * math.pi)
                        num = float(np.linalg.norm(pot_scaled - pot_ref))
                        den = float(np.linalg.norm(pot_ref))
                        row["accuracy"]["jaxfmm_potential_rel_l2"] = num / max(
                            den, 1e-300
                        )
                    except Exception as exc:
                        row["accuracy"]["jaxfmm_potential_rel_l2"] = None
                        row["accuracy"]["jaxfmm_error"] = str(exc)[:200]

            records.append(row)
            times = " ".join(
                f"{k}={v['min_s']*1e3:.2f}ms" if v.get("min_s") else f"{k}=ERR"
                for k, v in row["timings"].items()
            )
            print(f"N={n:<9d} set={set_name:<11s} {times}", flush=True)

    # -- fitted exponents ---------------------------------------------------
    fits: dict[str, dict[str, Any]] = {}
    labels = sorted({k for r in records for k in r["timings"]})
    for set_name in sets:
        subset = [r for r in records if r["param_set"] == set_name]
        for label in labels:
            xs = [r["n"] for r in subset if r["timings"].get(label, {}).get("min_s")]
            ys = [
                r["timings"][label]["min_s"]
                for r in subset
                if r["timings"].get(label, {}).get("min_s")
            ]
            if len(xs) < 2:
                continue
            fits[f"{set_name}:{label}"] = T.fit_log_log_exponent(
                xs, ys, min_n=int(args.fit_min_n)
            )

    print("\n=== fitted exponents (log t = alpha log N + c) ===")
    for name, fit in fits.items():
        print(
            f"  {name:<45s} alpha={fit['exponent']:.3f} "
            f"R^2={fit['r_squared']:.4f} over {fit['n_points']} pts "
            f"N={fit.get('fit_min_n')}..{fit.get('fit_max_n')}"
        )

    config = {
        "n": ns,
        "theta": [PARAM_SETS[s]["theta"] for s in sets],
        "order": [PARAM_SETS[s]["p"] for s in sets],
        "basis": args.basis,
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": args.dtype,
        "leaf_size": int(args.leaf_size),
        "preset": args.preset,
        "distribution": args.distribution,
        "param_sets": {s: PARAM_SETS[s] for s in sets},
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "full_repeats": int(args.full_repeats),
        "full_max_n": int(args.full_max_n),
        "softening": float(args.softening),
        "G": float(args.g),
        "timed_region": (
            "evaluation on a prebuilt tree/hierarchy for jaccpot_* and "
            "jaxfmm_potential; jaccpot_acceleration_full additionally includes "
            "tree construction"
        ),
        "jaxfmm_available": have_jaxfmm,
        "jaxfmm_kernel": "1/(4 pi r), unsoftened; rescaled by 4 pi for the accuracy check",
        "softening_applies_to": "jaccpot and direct-sum arms only",
        "fit_min_n": int(args.fit_min_n),
    }
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config=config,
        meta=runmeta.run_meta({"argv": sys.argv[1:]}),
        data={"records": records, "fits": fits},
    )
    print(f"\nwrote {out}")
    return 0


def _direct_potential(positions, masses, *, softening: float, G: float):
    """Direct O(N^2) potential, the reference for jaxFMM's output."""

    import jax.numpy as jnp
    import numpy as np

    pos = np.asarray(positions, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    n = pos.shape[0]
    out = np.zeros(n, dtype=np.float64)
    eps_sq = float(softening) ** 2
    # Blocked rather than one (N, N) allocation: at N = 16384 the dense form is
    # 2 GB in float64.
    block = 1024
    for start in range(0, n, block):
        stop = min(start + block, n)
        delta = pos[start:stop, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=2) + eps_sq)
        inv = 1.0 / dist
        rows = np.arange(start, stop)
        inv[np.arange(stop - start), rows] = 0.0
        out[start:stop] = G * (inv * mass[None, :]).sum(axis=1)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
