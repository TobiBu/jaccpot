"""Figure 19 data -- what a per-iteration rebuild actually does to the gradient.

fig19 is the section's **second substantive result**, not a supporting panel: it
is the first high-dimensional evidence that per-iteration tree rebuilds do not
obstruct convergence for an *inference* objective through the *FMM*. Both
qualifiers matter, because the Yggdrax precedent (D-015, D-016) closes the
parameter-count gap and neither of the other two: its objective is geometric
(mean nearest-neighbour spacing) and its operator is a radix tree's Morton
ordering, not an FMM with a MAC.

So this script measures four things, and each answers a question the Yggdrax run
cannot.

**1. Switch rates against rebuild cadence and step size, intensively.**
The rates are fractions of particles or of interacting pairs -- never an "any
change?" counter. D-016 is explicit about why: the extensive counter saturates
at 1.000 from ``N`` alone and reads as an uninformative result when it is really
a metric artefact. The extensive count is recorded alongside, labelled, so the
contrast is visible in the artifact. And ``near_pair_churn`` / ``far_pair_churn``
are the *interaction-list* rates -- every Yggdrax rate, its
``leaf_change_fraction`` included, is permutation-derived.

**2. Loss continuity across a switch.** A switch boundary is where the
piecewise-smooth loss can jump. The measurement takes the loss at the same
positions under the old topology and the new one; the difference is the jump,
and it is reported relative to the step's own loss decrease. A jump far smaller
than the progress made per step is the statement that the switching does not
obstruct the descent.

**3. FD-vs-autodiff within a topology epoch, and how far the gradient moves
across a rebuild.** Within an epoch the two must agree to FD precision -- the
section 2 contract, measured. **The pinned arm pins the interaction-list
selection as well as the tree** (D-016): pinning only the tree leaves a residual
that reads as a gradient bug and is not one. Passing one prepared state to both
arms pins both, since the state carries the tree, the M2L list and the
near-field CSR together. Across a rebuild the question is asked of the
*gradient* -- ``|grad_B - grad_A| / |grad_A|`` at the same positions, which is
directly comparable to the Yggdrax exactness result -- and **not** of a finite
difference whose two evaluations straddle the switch. That last quantity was
what the first version of this script reported, and it was meaningless: see
:func:`_fd_agreement` for the measurement that shows it tracks ``eps`` rather
than the pipeline.

**4. The effect of cadence k on the final residual.** Whether ``k > 1`` is
usable is an open question that only the interaction-list rate can answer, and
it is answered here rather than assumed. It is explicitly **not** justified by
appeal to ordering stability: the Yggdrax run shows the Morton permutation
changing on every step of a converging descent at these ``N``.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu python -m bench.payoff_static.topology_switching \\
        --n 192 --tracers 64 --iterations 6 --cadences 1,2 \\
        --gpu-select none --json-out /tmp/smoke.json

Paper run::

    python -m bench.payoff_static.topology_switching
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict, List

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "density_reconstruction/topology_switching.json"


def _parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", default="4096,32768,262144", help="Source counts N")
    p.add_argument("--tracers", type=int, default=2048, help="Tracer count M")
    p.add_argument("--iterations", type=int, default=60, help="Gradient steps per fit")
    p.add_argument("--cadences", default="1,2,4,8,16", help="Rebuild cadences k")
    p.add_argument(
        "--learning-rates",
        default="5e-4,2e-3,8e-3",
        help="Step sizes; the switch rate is expected to depend on this",
    )
    p.add_argument("--order", type=int, default=4, help="Expansion order")
    p.add_argument("--theta", type=float, default=0.5, help="MAC parameter")
    p.add_argument("--leaf-size", type=int, default=64, help="Leaf size")
    p.add_argument("--softening", type=float, default=1.0e-2, help="Plummer softening")
    p.add_argument("--fd-samples", type=int, default=6, help="FD directions to sample")
    p.add_argument("--fd-eps", type=float, default=1.0e-6, help="FD step")
    p.add_argument(
        "--near-set-sample",
        type=int,
        default=128,
        help=(
            "Particles to retain near-field source sets for; feeds the "
            "non-saturating near_set_churn rate and dominates the "
            "instrumentation cost (0 disables it)"
        ),
    )
    p.add_argument(
        "--intensive-every",
        type=int,
        default=1,
        help="Sampling stride for the intensive churn rates",
    )
    runmeta.add_common_args(p)
    return p.parse_args()


def _setup(n: int, args: argparse.Namespace) -> Any:
    """Build the truth, operator and parameterisation for one ``N``.

    Parameters
    ----------
    n : int
        Source count.
    args : argparse.Namespace
        Parsed command line.

    Returns
    -------
    Any
        ``(config, truth, operator, parameterization)``.
    """
    from jaccpot.applications.density_reconstruction.forward import (
        make_forward_operator,
    )
    from jaccpot.applications.density_reconstruction.parameterize import (
        make_parameterization,
    )
    from jaccpot.applications.density_reconstruction.truth import (
        TruthConfig,
        make_ground_truth,
    )

    config = TruthConfig(
        num_particles=int(n),
        num_tracers=int(args.tracers),
        seed=int(args.seed),
        softening=float(args.softening),
        generating_order=max(int(args.order) + 2, 6),
        generating_theta=min(float(args.theta), 0.4),
        generating_leaf_size=int(args.leaf_size),
    )
    truth = make_ground_truth(config)
    operator = make_forward_operator(
        tracer_positions=truth.tracer_positions,
        source_mass=truth.source_mass,
        num_sources=int(n),
        softening=float(args.softening),
        order=int(args.order),
        theta=float(args.theta),
        leaf_size=int(args.leaf_size),
    )
    parameterization = make_parameterization("positions", config=config)
    return config, truth, operator, parameterization


def _fd_agreement(
    *,
    operator: Any,
    observed: Any,
    positions: Any,
    state: Any,
    other_state: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Autodiff against finite differences, and against a rebuilt topology.

    Two questions, and the second one replaces a measurement that did not mean
    anything.

    **Within one topology epoch**, a directional finite difference of the
    frozen-topology map must agree with ``jax.grad`` of the same map. That is
    the fixed-topology contract of section 2, measured. Both arms use ONE
    prepared state, which pins the tree **and** the M2L/near-field interaction
    lists together -- D-016 is explicit that pinning only the tree leaves a
    residual that reads as a gradient bug.

    **Across a switch**, the useful question is how much the *gradient* moves
    when the topology is rebuilt at the same positions:
    ``|grad_B - grad_A| / |grad_A|``. This is directly comparable to the
    Yggdrax exactness result -- there, differentiating through the rebuild was
    bit-identical to the frozen-ordering gradient -- and it is bounded and
    interpretable.

    What this deliberately does **not** report is a finite difference whose two
    evaluations straddle the switch. The first version of this script did, and
    the number was nonsense: at N=32768 it came out at a relative error of
    5.0e3, which a reader would read as autodiff being wrong by a factor of
    5000. It is not a derivative at all. The two topologies assign slightly
    different losses to the same configuration -- a relative offset of ~1.3e-3,
    which the loss-continuity arm measures properly -- and dividing that offset
    by ``2 * eps`` with ``eps = 1e-6`` inflates it by 5e5. The quantity
    therefore diverges as ``eps -> 0`` and measures the step size, not the
    pipeline. ``crossed_eps_scaling`` below demonstrates that rather than
    asserting it: the same quantity is evaluated at ``eps`` and ``10 * eps``,
    and a ~10x ratio is the signature of an offset divided by a step.

    Parameters
    ----------
    operator : Any
        The observation operator.
    observed : Any
        ``(M, 3)`` observed accelerations.
    positions : Any
        ``(N, 3)`` positions to differentiate at.
    state : Any
        The epoch's prepared state; pins tree and interaction lists together.
    other_state : Any
        A state rebuilt from later positions -- the far side of a switch.
    args : argparse.Namespace
        Supplies ``fd_samples`` and ``fd_eps``.

    Returns
    -------
    Dict[str, Any]
        The pinned FD-vs-autodiff agreement, the across-rebuild gradient
        difference, and the eps-scaling demonstration.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    obs = jnp.asarray(observed)

    def loss_at(current_state: Any, p: Any) -> Any:
        predicted = operator.evaluate_at_topology(current_state, p)
        return jnp.mean(jnp.sum((predicted - obs) ** 2, axis=-1))

    x = jnp.asarray(positions)
    gradient = jax.grad(lambda p: loss_at(state, p))(x)
    gradient_other = jax.grad(lambda p: loss_at(other_state, p))(x)

    # The across-rebuild question, asked of the gradient itself.
    norm = float(jnp.linalg.norm(gradient))
    delta = float(jnp.linalg.norm(gradient_other - gradient))
    cosine = float(
        jnp.sum(gradient * gradient_other)
        / (jnp.linalg.norm(gradient) * jnp.linalg.norm(gradient_other) + 1.0e-300)
    )

    rng = np.random.default_rng(int(args.seed) + 17)
    eps = float(args.fd_eps)
    pinned: List[float] = []
    crossed_small: List[float] = []
    crossed_large: List[float] = []
    for _ in range(int(args.fd_samples)):
        raw = rng.standard_normal(np.shape(positions))
        direction = jnp.asarray(raw / np.linalg.norm(raw))
        analytic = float(jnp.sum(gradient * direction))
        scale = abs(analytic) + 1.0e-300

        # Pinned: both evaluations at the SAME state, so the finite difference
        # perturbs the very function autodiff differentiated.
        plus = float(loss_at(state, x + eps * direction))
        minus = float(loss_at(state, x - eps * direction))
        pinned.append(abs((plus - minus) / (2.0 * eps) - analytic) / scale)

        # The straddling quantity, at two step sizes, purely to show that it
        # scales like 1/eps and is therefore not a derivative.
        for step, sink in ((eps, crossed_small), (10.0 * eps, crossed_large)):
            ahead = float(loss_at(other_state, x + step * direction))
            behind = float(loss_at(state, x - step * direction))
            sink.append(abs((ahead - behind) / (2.0 * step) - analytic) / scale)

    small = float(np.median(crossed_small))
    large = float(np.median(crossed_large))
    return {
        "pinned_worst_rel": float(np.max(pinned)),
        "pinned_median_rel": float(np.median(pinned)),
        "fd_eps": eps,
        "fd_samples": int(args.fd_samples),
        "pinned_note": (
            "Both FD evaluations and the autodiff gradient use one prepared "
            "state, which pins the tree AND the M2L/near-field interaction "
            "lists together (D-016). Pinning only the tree leaves a residual "
            "that reads as a gradient bug and is not one."
        ),
        # The meaningful across-rebuild numbers.
        "gradient_norm": norm,
        "gradient_delta_across_rebuild": delta,
        "gradient_delta_rel_across_rebuild": delta / (norm + 1.0e-300),
        "gradient_cosine_across_rebuild": cosine,
        "crossed_eps_scaling": {
            "median_rel_at_eps": small,
            "median_rel_at_10eps": large,
            "ratio": small / (large + 1.0e-300),
            "note": (
                "A finite difference straddling a switch is NOT reported as an "
                "error, because it is not a derivative: it divides the "
                "topology-induced loss offset by 2*eps and so diverges as eps "
                "-> 0. A ratio near 10 here is that signature -- the quantity "
                "tracks the step size, not the pipeline. Use "
                "gradient_delta_rel_across_rebuild for the across-rebuild "
                "question and the loss_continuity arm for the offset itself."
            ),
        },
    }


def _loss_jump_across_switches(
    *, operator: Any, observed: Any, history_positions: List[Any], states: List[Any]
) -> Dict[str, Any]:
    """Measure the loss discontinuity at each rebuild boundary.

    Parameters
    ----------
    operator : Any
        The observation operator.
    observed : Any
        ``(M, 3)`` observed accelerations.
    history_positions : List[Any]
        Positions at each rebuild, in order.
    states : List[Any]
        The prepared state built at each of those positions.

    Returns
    -------
    Dict[str, Any]
        Per-boundary absolute and relative jumps, and their summary. The jump
        is the loss at **the same positions** under the outgoing and incoming
        topologies, so it isolates the topology change from the optimiser's
        step.
    """
    import jax.numpy as jnp
    import numpy as np

    obs = jnp.asarray(observed)

    def loss_at(state: Any, p: Any) -> float:
        predicted = operator.evaluate_at_topology(state, jnp.asarray(p))
        return float(jnp.mean(jnp.sum((predicted - obs) ** 2, axis=-1)))

    jumps: List[Dict[str, Any]] = []
    for index in range(1, len(states)):
        positions = history_positions[index]
        before = loss_at(states[index - 1], positions)
        after = loss_at(states[index], positions)
        jumps.append(
            {
                "boundary": index,
                "loss_outgoing_topology": before,
                "loss_incoming_topology": after,
                "absolute_jump": after - before,
                "relative_jump": abs(after - before) / (abs(before) + 1.0e-300),
            }
        )
    relative = [j["relative_jump"] for j in jumps]
    return {
        "boundaries": jumps,
        "max_relative_jump": float(np.max(relative)) if relative else None,
        "median_relative_jump": float(np.median(relative)) if relative else None,
    }


def _run_cadence(
    *, n: int, cadence: int, learning_rate: float, args: argparse.Namespace
) -> Dict[str, Any]:
    """One fit at one cadence and step size, fully instrumented.

    Parameters
    ----------
    n : int
        Source count.
    cadence : int
        Rebuild cadence ``k``.
    learning_rate : float
        Optimiser step size.
    args : argparse.Namespace
        Parsed command line.

    Returns
    -------
    Dict[str, Any]
        The record for this point.
    """
    import numpy as np

    from jaccpot.applications.density_reconstruction.diagnostics import field_residual
    from jaccpot.applications.density_reconstruction.fit import FitConfig, run_fit
    from jaccpot.applications.density_reconstruction.loss import Regularization
    from jaccpot.applications.density_reconstruction.parameterize import (
        initial_positions,
    )

    config, truth, operator, parameterization = _setup(n, args)
    start = initial_positions(
        truth.source_positions,
        mode="perturbed_truth",
        seed=int(args.seed) + 5,
        perturbation=0.05,
    )

    # Capture the positions and state at every rebuild, for the loss-jump arm.
    captured_positions: List[Any] = []
    captured_states: List[Any] = []
    original_prepare = operator.prepare

    def capturing_prepare(positions: Any) -> Any:
        state = original_prepare(positions)
        captured_positions.append(np.array(positions, dtype=np.float64))
        captured_states.append(state)
        return state

    object.__setattr__(operator, "prepare", capturing_prepare)
    try:
        result = run_fit(
            operator=operator,
            observed=truth.observed_accelerations,
            parameterization=parameterization,
            initial_params=parameterization.pack(start),
            config=FitConfig(
                num_iterations=int(args.iterations),
                learning_rate=float(learning_rate),
                rebuild_cadence=int(cadence),
                regularization=Regularization.none(),
                history_every=1,
                intensive_every=int(args.intensive_every),
                near_set_sample=int(args.near_set_sample),
                seed=int(args.seed),
            ),
        )
    finally:
        object.__setattr__(operator, "prepare", original_prepare)

    before = field_residual(
        operator.evaluate(start),
        truth.observed_accelerations,
        clean=truth.clean_accelerations,
    )
    after = field_residual(
        operator.evaluate(result.positions),
        truth.observed_accelerations,
        clean=truth.clean_accelerations,
    )

    record: Dict[str, Any] = {
        "N": int(n),
        "M": int(args.tracers),
        "num_free_parameters": int(parameterization.num_free),
        "rebuild_cadence": int(cadence),
        "learning_rate": float(learning_rate),
        "iterations": int(args.iterations),
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "loss_ratio": (
            result.final_loss / result.initial_loss if result.initial_loss else None
        ),
        "field_residual_before": before,
        "field_residual_after": after,
        "switch_summary": result.switch_summary,
        "timing": result.timing,
        "loss_trace": [
            {
                "iteration": row["iteration"],
                "loss": row["loss"],
                "rebuilt": row["rebuilt"],
                "gradient_rms_per_parameter": row["gradient"]["rms_per_parameter"],
            }
            for row in result.history
        ],
    }

    if len(captured_states) >= 2:
        record["loss_continuity"] = _loss_jump_across_switches(
            operator=operator,
            observed=truth.observed_accelerations,
            history_positions=captured_positions,
            states=captured_states,
        )
        # Relate the jump to the progress a step makes: a discontinuity far
        # below the per-step improvement is the statement that switching does
        # not obstruct the descent.
        losses = [row["loss"] for row in result.history]
        improvements = [abs(losses[i - 1] - losses[i]) for i in range(1, len(losses))]
        median_improvement = float(np.median(improvements)) if improvements else None
        record["loss_continuity"]["median_step_improvement"] = median_improvement
        median_jump = record["loss_continuity"]["median_relative_jump"]
        if median_improvement and median_jump is not None and losses:
            record["loss_continuity"]["jump_over_step_improvement"] = float(
                median_jump * abs(losses[0]) / median_improvement
            )

    if len(captured_states) >= 2:
        record["fd_agreement"] = _fd_agreement(
            operator=operator,
            observed=truth.observed_accelerations,
            positions=captured_positions[0],
            state=captured_states[0],
            other_state=captured_states[1],
            args=args,
        )

    intensive = (result.switch_summary or {}).get("intensive", {}).get("mean", {})
    extensive = (result.switch_summary or {}).get("extensive", {})
    continuity = record.get("loss_continuity") or {}
    agreement = record.get("fd_agreement") or {}
    near_set = intensive.get("near_set_churn")
    print(
        f"  N={n:>8} k={cadence:>3} lr={learning_rate:<8g} "
        f"loss {result.initial_loss:.3e}->{result.final_loss:.3e} | "
        f"EXT {extensive.get('switch_rate')} | "
        f"near_SET {'n/a' if near_set is None else round(near_set, 5)} "
        f"nearpair {round(intensive.get('near_pair_churn') or 0.0, 4)} "
        f"leaf {round(intensive.get('leaf_churn') or 0.0, 4)} | "
        f"jump/step {continuity.get('jump_over_step_improvement')} | "
        f"grad delta {agreement.get('gradient_delta_rel_across_rebuild')}",
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

    n_list = [int(v) for v in str(args.n).split(",") if v]
    cadences = [int(v) for v in str(args.cadences).split(",") if v]
    rates = [float(v) for v in str(args.learning_rates).split(",") if v]

    print(
        f"topology_switching: N in {n_list}, cadences {cadences}, "
        f"learning rates {rates}, {args.iterations} iterations",
        flush=True,
    )

    out = args.json_out or DEFAULT_OUT
    config_record = {
        "n": n_list,
        "theta": float(args.theta),
        "order": int(args.order),
        "basis": "solidfmm",
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": str(args.dtype),
        "M": int(args.tracers),
        "leaf_size": int(args.leaf_size),
        "softening": float(args.softening),
        "iterations": int(args.iterations),
        "cadences": cadences,
        "learning_rates": rates,
        "intensive_every": int(args.intensive_every),
        "near_set_sample": int(args.near_set_sample),
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
        Called after EVERY point rather than once at the end. This sweep runs
        for hours -- 28 fits, on the leaf-64 configuration it needs for a
        non-empty far field, which is ~10x slower per evaluation than fig16's
        leaf 256 -- and a run stopped, pre-empted, or killed by an
        out-of-memory partway used to leave nothing behind at all. Overwriting
        the same path is already the contract (``jsonio.write_result`` is
        documented idempotent), so the cost is rewriting a small JSON per point
        and the benefit is that a partial sweep is still a usable artifact.
        """
        return jsonio.write_result(
            out,
            config=config_record,
            meta=runmeta.run_meta(),
            data={"records": records},
        )

    for n in n_list:
        for cadence in cadences:
            # The step-size axis only needs sweeping at one cadence: it is a
            # separate question (how far does a step move the ordering) from the
            # cadence question (how stale may the interaction list be).
            for learning_rate in rates if cadence == 1 else rates[:1]:
                try:
                    records.append(
                        _run_cadence(
                            n=n,
                            cadence=cadence,
                            learning_rate=learning_rate,
                            args=args,
                        )
                    )
                except Exception as exc:  # pragma: no cover - OOM at large N
                    print(
                        f"  N={n} k={cadence} lr={learning_rate}: FAILED "
                        f"({type(exc).__name__}: {exc})",
                        flush=True,
                    )
                    records.append(
                        {
                            "N": int(n),
                            "rebuild_cadence": int(cadence),
                            "learning_rate": float(learning_rate),
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
