"""Figures 17 and 18 data -- the full fits, and what the degeneracy does to them.

One script, one sweep, and the axes are the ones section 2 says are results in
their own right:

* **parameterisation** -- ``parametric`` (7 or 11 parameters) against
  ``positions`` (3N). The contrast is the section.
* **N** -- and therefore free-parameter count.
* **initial guess** -- all four modes. Section 2 expects this to be the
  section's most interesting number, because the loss is non-convex in the
  positions and a structurally wrong smooth start is the honest test of whether
  the fit can find structure it was not handed.
* **noise** -- including a zero-noise run. The residual is reported against the
  noise floor, because a fit that has reached the floor has extracted
  everything the data holds and driving it lower is fitting the realisation.
* **regularisation** -- including a deliberately **unregularised** run. That run
  is fig17(d): showing what the degeneracy does is an informative panel, not an
  embarrassment, and it is why the regularisers are evaluated even at weight
  zero.
* **softening** -- two values, so the sensitivity hard decision 6 asks for is
  measured rather than asserted.
* **perturber** -- with and without the LMC-like overdensity, which is the
  localised non-axisymmetric feature fig17(b) shows the residual map around.

What the records are built to show
----------------------------------
fig18's content is a **divergence**: the field residual keeps falling while the
density agreement saturates. That is the degeneracy stated quantitatively, so
every iteration row carries both -- the field residual (primary) and, at a
coarser stride because it is not free, the density agreement (secondary). A
figure that plotted only the loss could not show it.

The per-particle position error is recorded too, at the end only, carrying its
own ``degenerate: True`` flag and caveat string from
:mod:`~jaccpot.applications.density_reconstruction.diagnostics`. It is not a
score and the record says so in the same breath as the number.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu python -m bench.payoff_static.reconstruction_runs \\
        --n 192 --tracers 64 --iterations 6 --cases smoke \\
        --gpu-select none --json-out /tmp/smoke.json

Paper run::

    python -m bench.payoff_static.reconstruction_runs
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "density_reconstruction/reconstruction_runs.json"

#: The four initial-guess modes of section 2, in increasing wrongness.
INITIAL_GUESSES = (
    "perturbed_truth",
    "isotropized_truth",
    "smooth_wrong",
    "uniform_sphere",
)


def _parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", default="16384,131072", help="Source counts N")
    p.add_argument("--tracers", type=int, default=8192, help="Tracer count M")
    p.add_argument("--iterations", type=int, default=200, help="Gradient steps")
    p.add_argument("--learning-rate", type=float, default=2.0e-3, help="Step size")
    p.add_argument(
        "--parametric-learning-rate",
        type=float,
        default=2.0e-2,
        help="Step size for the parametric arm, whose parameters are log scales",
    )
    p.add_argument("--order", type=int, default=4, help="Expansion order")
    p.add_argument("--theta", type=float, default=0.5, help="MAC parameter")
    p.add_argument("--leaf-size", type=int, default=64, help="Leaf size")
    p.add_argument(
        "--softenings",
        default="1e-2,3e-2",
        help="Softening lengths; two values give the sensitivity run",
    )
    p.add_argument(
        "--noise-fractions", default="0.0,0.05", help="Fractional acceleration noise"
    )
    p.add_argument("--rebuild-cadence", type=int, default=1, help="Rebuild every k")
    p.add_argument(
        "--diagnostics-every",
        type=int,
        default=20,
        help="Stride for the density-agreement diagnostic, which is not free",
    )
    p.add_argument(
        "--cases",
        choices=("full", "smoke"),
        default="full",
        help="'smoke' runs one tiny case per parameterisation, for CI",
    )
    p.add_argument(
        "--perturbers", default="lmc_like,none", help="Perturber modes to sweep"
    )
    runmeta.add_common_args(p)
    return p.parse_args()


def _case_list(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build the sweep's case list.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line.

    Returns
    -------
    List[Dict[str, Any]]
        One dict per case. The full sweep is deliberately not a full Cartesian
        product: the initial-guess axis is swept at the *reference* setting and
        the other axes each vary against that reference, so the cost is linear
        in the number of axes rather than exponential, and every point still
        differs from the reference in exactly one respect.
    """
    n_list = [int(v) for v in str(args.n).split(",") if v]
    softenings = [float(v) for v in str(args.softenings).split(",") if v]
    noises = [float(v) for v in str(args.noise_fractions).split(",") if v]
    perturbers = [v for v in str(args.perturbers).split(",") if v]

    if args.cases == "smoke":
        return [
            {
                "parameterization": kind,
                "N": n_list[0],
                "initial_guess": "perturbed_truth",
                "noise_fraction": noises[0],
                "softening": softenings[0],
                "perturber": perturbers[0],
                "regularized": True,
            }
            for kind in ("parametric", "positions")
        ]

    reference = {
        "N": n_list[0],
        "initial_guess": "perturbed_truth",
        "noise_fraction": noises[0],
        "softening": softenings[0],
        "perturber": perturbers[0],
        "regularized": True,
    }
    cases: List[Dict[str, Any]] = []
    seen = set()

    def add(**overrides: Any) -> None:
        case = {**reference, **overrides}
        key = tuple(sorted((k, str(v)) for k, v in case.items()))
        if key not in seen:
            seen.add(key)
            cases.append(case)

    for kind in ("parametric", "positions"):
        # The reference point, then one axis at a time off it.
        add(parameterization=kind)
        for guess in INITIAL_GUESSES:
            add(parameterization=kind, initial_guess=guess)
        for n in n_list:
            add(parameterization=kind, N=n)
        for noise in noises:
            add(parameterization=kind, noise_fraction=noise)
        for softening in softenings:
            add(parameterization=kind, softening=softening)
        for perturber in perturbers:
            add(parameterization=kind, perturber=perturber)
    # The unregularised run exists only for the high-dimensional arm: with 7
    # parameters there is no clumping degeneracy to expose.
    add(parameterization="positions", regularized=False)
    add(parameterization="positions", regularized=False, initial_guess="smooth_wrong")
    return cases


def _run_case(case: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run one fit and assemble its record.

    Parameters
    ----------
    case : Dict[str, Any]
        One entry from :func:`_case_list`.
    args : argparse.Namespace
        Parsed command line.

    Returns
    -------
    Dict[str, Any]
        The record for this case.
    """
    import numpy as np

    from jaccpot.applications.density_reconstruction.diagnostics import (
        density_agreement,
        enclosed_mass_profile,
        field_residual,
        moment_drift,
        position_error,
        radial_bins,
    )
    from jaccpot.applications.density_reconstruction.fit import FitConfig, run_fit
    from jaccpot.applications.density_reconstruction.forward import (
        make_forward_operator,
    )
    from jaccpot.applications.density_reconstruction.loss import Regularization
    from jaccpot.applications.density_reconstruction.parameterize import (
        initial_positions,
        make_parameterization,
    )
    from jaccpot.applications.density_reconstruction.truth import (
        TruthConfig,
        make_ground_truth,
    )

    n = int(case["N"])
    config = TruthConfig(
        num_particles=n,
        num_tracers=int(args.tracers),
        seed=int(args.seed),
        softening=float(case["softening"]),
        noise_fraction=float(case["noise_fraction"]),
        perturber=str(case["perturber"]),
        generating_order=max(int(args.order) + 2, 6),
        generating_theta=min(float(args.theta), 0.4),
        generating_leaf_size=int(args.leaf_size),
    )
    truth = make_ground_truth(config)
    operator = make_forward_operator(
        tracer_positions=truth.tracer_positions,
        source_mass=truth.source_mass,
        num_sources=n,
        softening=float(case["softening"]),
        order=int(args.order),
        theta=float(args.theta),
        leaf_size=int(args.leaf_size),
    )
    kind = str(case["parameterization"])
    parameterization = make_parameterization(kind, config=config)

    if kind == "positions":
        start_positions = initial_positions(
            truth.source_positions,
            mode=str(case["initial_guess"]),
            seed=int(args.seed) + 5,
            perturbation=0.05,
        )
        initial_params = parameterization.pack(start_positions)
        learning_rate = float(args.learning_rate)
    else:
        # The parametric arm's "initial guess" is an offset in parameter space;
        # the four position modes have no parametric analogue, so the axis is
        # expressed as increasing offsets and labelled with the same names.
        offsets = {
            "perturbed_truth": 0.05,
            "isotropized_truth": 0.20,
            "smooth_wrong": 0.50,
            "uniform_sphere": 1.00,
        }
        offset = offsets[str(case["initial_guess"])]
        true_values = parameterization.true_params(config)
        initial_params = parameterization.pack(
            {name: float(value) + offset for name, value in true_values.items()}
        )
        start_positions = np.asarray(
            parameterization.to_positions(initial_params), dtype=np.float64
        )
        learning_rate = float(args.parametric_learning_rate)

    regularization = Regularization() if case["regularized"] else Regularization.none()

    # A heartbeat, so a long GPU run has a progress trace rather than a silent
    # hour. The per-iteration data itself comes from result.history below.
    stride = max(int(args.diagnostics_every), 1)

    def progress(row: Dict[str, Any]) -> None:
        if row["iteration"] % stride == 0:
            print(
                f"      iter {row['iteration']:>5} loss {row['loss']:.6e} "
                f"|g|/param {row['gradient']['rms_per_parameter']:.3e}",
                flush=True,
            )

    result = run_fit(
        operator=operator,
        observed=truth.observed_accelerations,
        parameterization=parameterization,
        initial_params=initial_params,
        config=FitConfig(
            num_iterations=int(args.iterations),
            learning_rate=learning_rate,
            rebuild_cadence=int(args.rebuild_cadence),
            regularization=regularization,
            history_every=1,
            seed=int(args.seed),
        ),
        progress=progress,
    )

    before_field = field_residual(
        operator.evaluate(start_positions),
        truth.observed_accelerations,
        clean=truth.clean_accelerations,
    )
    after_field = field_residual(
        operator.evaluate(result.positions),
        truth.observed_accelerations,
        clean=truth.clean_accelerations,
    )
    before_density = density_agreement(
        start_positions, truth.source_positions, source_mass=truth.source_mass
    )
    after_density = density_agreement(
        result.positions, truth.source_positions, source_mass=truth.source_mass
    )
    edges = radial_bins(truth.source_positions)

    record = {
        **case,
        "num_free_parameters": int(parameterization.num_free),
        "M": int(args.tracers),
        "learning_rate": learning_rate,
        "iterations": int(args.iterations),
        "rebuild_cadence": int(args.rebuild_cadence),
        "regularization": regularization.as_record(),
        "generating_config": truth.record(),
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "field_residual_before": before_field,
        "field_residual_after": after_field,
        # Only the scalar summaries of the density comparison go in the record;
        # the full profiles would multiply the file size by the case count.
        "density_before": {
            k: v for k, v in before_density.items() if not k.startswith("profile_r")
        },
        "density_after": {
            k: v for k, v in after_density.items() if not k.startswith("profile_r")
        },
        "profile_truth": enclosed_mass_profile(
            truth.source_positions, source_mass=truth.source_mass, edges=edges
        ),
        "profile_reconstructed": enclosed_mass_profile(
            result.positions, source_mass=truth.source_mass, edges=edges
        ),
        "profile_initial": enclosed_mass_profile(
            start_positions, source_mass=truth.source_mass, edges=edges
        ),
        "position_error": position_error(result.positions, truth.source_positions),
        "moment_drift": moment_drift(
            result.positions, truth.source_positions, source_mass=truth.source_mass
        ).as_record(),
        "loss_trace": [
            {
                "iteration": row["iteration"],
                "loss": row["loss"],
                "components": row["components"],
                "gradient_rms_per_parameter": row["gradient"]["rms_per_parameter"],
                "rebuilt": row["rebuilt"],
            }
            for row in result.history
        ],
        "switch_summary": result.switch_summary,
        "timing": result.timing,
    }
    print(
        f"  {kind:>10} N={n:>8} guess={case['initial_guess']:<18} "
        f"noise={case['noise_fraction']:<5} soft={case['softening']:<6} "
        f"pert={case['perturber']:<9} reg={int(bool(case['regularized']))} | "
        f"field {before_field['rel_l2']:.3e}->{after_field['rel_l2']:.3e} | "
        f"density {before_density['grid_rel_l2']:.3e}->"
        f"{after_density['grid_rel_l2']:.3e}",
        flush=True,
    )
    return record


def main() -> int:
    """Run the sweep and write the results JSON.

    Returns
    -------
    int
        Process exit status.
    """
    args = _parse_args()
    runmeta.select_gpu(args.gpu_select)
    runmeta.enable_x64(args.dtype)

    cases = _case_list(args)
    print(f"reconstruction_runs: {len(cases)} cases", flush=True)

    out = args.json_out or DEFAULT_OUT
    config_record = {
        "n": [int(v) for v in str(args.n).split(",") if v],
        "theta": float(args.theta),
        "order": int(args.order),
        "basis": "solidfmm",
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": str(args.dtype),
        "M": int(args.tracers),
        "leaf_size": int(args.leaf_size),
        "iterations": int(args.iterations),
        "rebuild_cadence": int(args.rebuild_cadence),
        "softenings": [float(v) for v in str(args.softenings).split(",") if v],
        "noise_fractions": [
            float(v) for v in str(args.noise_fractions).split(",") if v
        ],
        "initial_guesses": list(INITIAL_GUESSES),
        "cases": str(args.cases),
    }
    records: List[Dict[str, Any]] = []

    def flush() -> Any:
        """Write the results JSON as it currently stands.

        Returns
        -------
        Any
            The path written.

        Notes
        -----
        Called after every case. This sweep is 18 gradient-descent fits and
        runs for hours; a run stopped or killed partway used to leave nothing
        behind. ``jsonio.write_result`` is documented idempotent, so the cost is
        rewriting one JSON per case.
        """
        return jsonio.write_result(
            out,
            config=config_record,
            meta=runmeta.run_meta(),
            data={"records": records},
        )

    for case in cases:
        try:
            records.append(_run_case(case, args))
        except Exception as exc:  # pragma: no cover - OOM or divergence
            print(f"  {case}: FAILED ({type(exc).__name__}: {exc})", flush=True)
            records.append(
                {
                    **case,
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
        flush()

    print(f"wrote {flush()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
