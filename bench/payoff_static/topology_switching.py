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

**3. FD-vs-autodiff agreement within a topology epoch against across a switch.**
Within an epoch the two must agree to FD precision -- that is the section 2
contract. Across a switch they legitimately disagree, and quantifying that
disagreement is the honest form of the claim. **The pinned arm pins the
interaction-list selection as well as the tree** (D-016): pinning only the tree
leaves a residual that reads as a gradient bug and is not one. Passing one
prepared state to both arms pins both, because the state carries the tree, the
M2L list and the near-field CSR together.

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
    """Directional FD-vs-autodiff agreement, within one epoch and across a switch.

    Parameters
    ----------
    operator : Any
        The observation operator.
    observed : Any
        ``(M, 3)`` observed accelerations.
    positions : Any
        ``(N, 3)`` positions to differentiate at.
    state : Any
        The epoch's prepared state. Both FD arms and the autodiff arm use this
        one object, which pins the tree **and** the interaction-list selection
        together -- D-016's requirement.
    other_state : Any
        A state rebuilt from perturbed positions, standing in for the topology
        on the far side of a switch.
    args : argparse.Namespace
        Supplies ``fd_samples`` and ``fd_eps``.

    Returns
    -------
    Dict[str, Any]
        Worst and median relative disagreement for the pinned arm and the
        crossed arm, and the ratio between them.
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

    rng = np.random.default_rng(int(args.seed) + 17)
    eps = float(args.fd_eps)
    pinned: List[float] = []
    crossed: List[float] = []
    for _ in range(int(args.fd_samples)):
        raw = rng.standard_normal(np.shape(positions))
        direction = jnp.asarray(raw / np.linalg.norm(raw))
        analytic = float(jnp.sum(gradient * direction))

        # Pinned: both evaluations at the SAME state, so the finite difference
        # perturbs the very function autodiff differentiated.
        plus = float(loss_at(state, x + eps * direction))
        minus = float(loss_at(state, x - eps * direction))
        fd_pinned = (plus - minus) / (2.0 * eps)

        # Crossed: the two sides use different topologies, which is what an FD
        # step that straddles a switch boundary actually samples.
        plus_other = float(loss_at(other_state, x + eps * direction))
        fd_crossed = (plus_other - minus) / (2.0 * eps)

        scale = abs(analytic) + 1.0e-300
        pinned.append(abs(fd_pinned - analytic) / scale)
        crossed.append(abs(fd_crossed - analytic) / scale)

    return {
        "pinned_worst_rel": float(np.max(pinned)),
        "pinned_median_rel": float(np.median(pinned)),
        "crossed_worst_rel": float(np.max(crossed)),
        "crossed_median_rel": float(np.median(crossed)),
        "crossed_over_pinned_median": float(
            np.median(crossed) / (np.median(pinned) + 1.0e-300)
        ),
        "fd_eps": eps,
        "fd_samples": int(args.fd_samples),
        "pinned_note": (
            "Both FD evaluations and the autodiff gradient use one prepared "
            "state, which pins the tree AND the M2L/near-field interaction "
            "lists together (D-016). Pinning only the tree leaves a residual "
            "that reads as a gradient bug and is not one."
        ),
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
    print(
        f"  N={n:>8} k={cadence:>3} lr={learning_rate:<8g} "
        f"loss {result.initial_loss:.3e}->{result.final_loss:.3e} | "
        f"EXT switch_rate {extensive.get('switch_rate')} | "
        f"INT slot {intensive.get('slot_churn')} leaf {intensive.get('leaf_churn')} "
        f"nearpair {intensive.get('near_pair_churn')}",
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

    records: List[Dict[str, Any]] = []
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

    out = args.json_out or DEFAULT_OUT
    written = jsonio.write_result(
        out,
        config={
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
        },
        meta=runmeta.run_meta(),
        data={"records": records},
    )
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
