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

Two settings that are measurements, and one diagnosis that was wrong
--------------------------------------------------------------------
``--iterations`` defaults to 240 and the parametric arm's step size to 1e-2,
both changed from a first sweep at 120 and 2e-2. Checked at N=512 on CPU before
spending GPU time: lr 1e-2 over 240 iterations reaches a field residual of
2.096e-03 where lr 2e-2 over 120 reaches 1.288e-02 at softening 0.01, and
1.404e-03 against 6.891e-03 at softening 0.03. Strictly better in both, so these
are the better settings on their own merits.

**But they were chosen for the wrong reason, and the record should say so.** The
first sweep produced two cells where the field residual got WORSE -- the
parametric arm at softening 0.03 (5.298e-02 -> 1.089e-01) and at N=131072
(5.536e-02 -> 6.756e-02) -- and I attributed that to the step size being too
large. It was not. The cause was the REGULARISER DOMINATING THE OBJECTIVE. With
absolute weights, the penalty-to-data ratio at iteration 0 ranged from 0.00 to
32.7 across a single sweep, because the data misfit is normalised by the
observed field's rms while the floor penalty grows with both particle density
and softening length. At N=131072 the ratio was 15:1, and the trace shows the
optimiser doing exactly what it was told: the total fell 5.164e-02 -> 4.775e-02
while the data term ROSE 3.065e-03 -> 3.914e-03 and the floor penalty fell
4.584e-02 -> 4.114e-02. Halving the step size moved that cell from 6.756e-02 to
6.249e-02 and left it worse than its 5.536e-02 start, which is what a fix
addressing the wrong cause looks like.

The actual fix is in
:class:`~jaccpot.applications.density_reconstruction.loss.Regularization`:
weights are now fractions of the initial data misfit, resolved once at
iteration 0, so the same number means the same thing in every cell. Every run
records the requested fractions, the resolved absolute weights and the initial
misfit, so the balance is inspectable rather than inferred.

The other bad cell was real and remains the reason for 240 iterations:
``uniform_sphere`` with free positions moved only 9.985e-01 -> 9.913e-01 in 120
iterations, which cannot be distinguished from not having run long enough and so
says nothing about the initial-guess sensitivity section 2 asks for.

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
    p.add_argument(
        "--merge-from",
        default="",
        help=(
            "Comma-separated result JSONs to concatenate into --json-out and "
            "exit, measuring nothing. These sweeps are host-bound and "
            "single-threaded, so wall-clock is cut by running disjoint slices "
            "as separate processes and stitching them here"
        ),
    )
    p.add_argument("--n", default="16384,131072", help="Source counts N")
    p.add_argument("--tracers", type=int, default=8192, help="Tracer count M")
    p.add_argument("--iterations", type=int, default=240, help="Gradient steps")
    p.add_argument("--learning-rate", type=float, default=2.0e-3, help="Step size")
    p.add_argument(
        "--parametric-learning-rate",
        type=float,
        default=1.0e-2,
        help=(
            "Step size for the parametric arm, whose parameters are log "
            "scales. Lowered from 2e-2, which diverged at softening 0.03 and "
            "at N=131072; see the module docstring"
        ),
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


def _slim_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Drop per-iteration bulk that no figure reads.

    The committed artifact is a figure's input, and
    ``bench/results/.gitignore`` is explicit that only the small summary a
    figure actually reads belongs in the index -- "if a file is large enough
    that you hesitate, it is a bulk array". The first version of this sweep
    ignored that: 5.2 MB for 18 cases, against 2.5 MB for all 183 result JSONs
    the repository already tracks and 185 kB for the largest single one.

    Three things go, none of them read by fig17 or fig18:

    * ``switch_summary``'s per-comparison churn records and event list -- 82 kB
      per case, half the file, and this sweep is not the switching figure;
      fig19 has its own artifact. The churn *means* stay, since they are small
      and say whether a fit's topology was moving.
    * the per-iteration loss ``components`` and gradient norms. fig18 plots
      iteration against loss; the components matter while diagnosing a fit, not
      afterwards.
    * ``profile_truth`` inside ``density_before``/``density_after``, which is a
      verbatim copy of the top-level ``profile_truth``.

    Parameters
    ----------
    record : Dict[str, Any]
        One case's record.

    Returns
    -------
    Dict[str, Any]
        A copy with the bulk removed. Applied both when a case finishes and
        when slices are merged, so a single-shot run and a sharded one produce
        the same artifact.
    """
    slim = dict(record)
    summary = slim.get("switch_summary")
    if isinstance(summary, dict):
        extensive = {
            k: v for k, v in (summary.get("extensive") or {}).items() if k != "events"
        }
        intensive = {
            k: v
            for k, v in (summary.get("intensive") or {}).items()
            if k != "per_comparison"
        }
        slim["switch_summary"] = {
            "switch_metric": summary.get("switch_metric"),
            "content_key": summary.get("content_key"),
            "extensive": extensive,
            "intensive": intensive,
        }
    slim["loss_trace"] = [
        {
            "iteration": row["iteration"],
            "loss": row["loss"],
            "rebuilt": row.get("rebuilt"),
        }
        for row in slim.get("loss_trace", [])
    ]
    for key in ("density_before", "density_after"):
        block = slim.get(key)
        if isinstance(block, dict):
            slim[key] = {k: v for k, v in block.items() if not k.startswith("profile_")}
    return slim


def _merge_runs(args: argparse.Namespace) -> int:
    """Concatenate sharded result files into one artifact.

    These sweeps are host-bound and single-threaded -- measured at load average
    22 on 64 cores with the GPU at 0-19% -- so wall-clock is cut by running
    disjoint slices of the sweep as separate PROCESSES rather than by any
    change to the numerics. Each slice writes its own JSON; this stitches them
    back into the single artifact a figure reads, and measures nothing.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line; ``merge_from`` names the inputs and ``json_out``
        the destination.

    Returns
    -------
    int
        Process exit status.

    Raises
    ------
    SystemExit
        If an input is missing, or if the inputs disagree on a configuration
        axis that would make them incomparable. Stitching slices measured at
        different accuracy settings produces a curve over two problems.
    """
    sources = [v.strip() for v in str(args.merge_from).split(",") if v.strip()]
    records: List[Dict[str, Any]] = []
    merged_config: Optional[Dict[str, Any]] = None
    metas: List[Dict[str, Any]] = []
    pinned = ("theta", "order", "basis", "leaf_size", "softening", "M")

    for source in sources:
        path = source if source.startswith("/") else str(jsonio.results_path(source))
        try:
            artifact = jsonio.read_result(path)
        except FileNotFoundError as exc:
            raise SystemExit(f"merge input not found: {path}") from exc
        config = dict(artifact["config"])
        if merged_config is None:
            merged_config = config
        else:
            clashes = {
                key: (merged_config.get(key), config.get(key))
                for key in pinned
                if merged_config.get(key) != config.get(key)
            }
            if clashes:
                raise SystemExit(
                    f"refusing to merge {path}: inputs disagree on {clashes}"
                )
        records.extend(_slim_record(r) for r in artifact["data"].get("records", []))
        metas.append(dict(artifact.get("meta", {})))

    if merged_config is None:
        raise SystemExit("--merge-from named no readable inputs")
    merged_config["n"] = sorted({r["N"] for r in records if "N" in r})
    merged_config["merged_from"] = sources

    out = args.json_out or DEFAULT_OUT
    written = jsonio.write_result(
        out,
        config=merged_config,
        # Each slice's provenance is kept: they are separate processes, and one
        # sha over them would claim more than is true.
        meta={**runmeta.run_meta(), "merged_source_meta": metas},
        data={"records": records},
    )
    print(f"merged {len(sources)} slice(s) -> {written}")
    return 0


def main() -> int:
    """Run the sweep and write the results JSON.

    Returns
    -------
    int
        Process exit status.
    """
    args = _parse_args()
    if str(args.merge_from).strip():
        return _merge_runs(args)
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
            records.append(_slim_record(_run_case(case, args)))
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
