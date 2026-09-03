"""The optimisation driver: gradient descent through a per-iteration rebuild.

One loop, two parameterisations, and the rebuild cadence as a first-class knob.

The topology epoch
------------------
Hard decision 5: the topology is rebuilt from the current positions and the MAC
test is frozen. This driver makes that structure explicit. A **topology epoch**
is a stretch of ``rebuild_cadence`` iterations sharing one prepared state:

* Within an epoch the gradient is the *exact* gradient of the fixed-topology,
  fixed-interaction-list forward map. That is the contract section 2 states, and
  nothing here approximates it.
* Across epochs the loss is piecewise smooth, with switching boundaries at the
  rebuilds. :class:`~jaccpot.applications.density_reconstruction.topology.SwitchLog`
  instruments those boundaries rather than hoping they do not matter.

``rebuild_cadence = k > 1`` smooths the loss over longer stretches at the cost
of a stale interaction list. **The trade-off is a measurement, not a guess**, and
it is not justified by ordering stability: the Yggdrax scale-up shows the Morton
permutation changing on every step of a converging descent at these ``N``.
Whether the *interaction list* is stable enough for ``k > 1`` is what
``near_pair_churn`` answers, and that is the content of fig19's cadence panel.

Eager or compiled, and why eager is the default
-----------------------------------------------
``differentiable_step_fn`` returns ``jax.jit`` over a closure that captures the
prepared state as a **constant**, so a rebuilt state is a fresh compile. Measured
on an A100 at N=4096, leaf 64, p=4, fp64:

===========================  ==================  ====================
path                         first iteration     later iterations
===========================  ==================  ====================
eager (the default)          20.3 s prepare,     1.3 s prepare,
                             39.3 s gradient     0.62 s gradient
compiled, state as constant  ~39 s compile       0.018 s gradient
===========================  ==================  ====================

The eager path does **not** recompile across rebuilds -- the prepared state's
array *shapes* were measured identical over six consecutive rebuilds at fixed
``N``, so JAX's own cache hits and only Python-level dispatch is paid. That
makes eager the right default: it is flat in the number of rebuilds, where the
compiled path pays ~39 s at every one. The compiled path wins only when an epoch
is long enough to amortise its compile, which at N=4096 is around
``k >= 65``; at larger ``N`` the per-step work grows while the compile does not,
so the crossover falls. ``jit_within_epoch`` selects it, and the results JSON
records which was used -- a wall-clock number is not comparable across the two.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.applications.density_reconstruction.diagnostics import gradient_norm
from jaccpot.applications.density_reconstruction.forward import (
    ForwardOperator,
    assert_masses_frozen_and_equal,
)
from jaccpot.applications.density_reconstruction.loss import (
    LeafBlocks,
    Regularization,
    leaf_blocks_from_state,
    regularization_terms,
    total_loss,
)
from jaccpot.applications.density_reconstruction.topology import SwitchLog

__all__ = ["FitConfig", "FitResult", "make_optimizer", "run_fit"]

OptimizerName = Literal["adam", "adamw", "sgd", "rmsprop"]


def make_optimizer(name: OptimizerName, learning_rate: float, **kwargs: Any) -> Any:
    """Build an optax optimiser by name.

    Parameters
    ----------
    name : OptimizerName
        ``"adam"``, ``"adamw"``, ``"sgd"`` or ``"rmsprop"``.
    learning_rate : float
        Step size, or any optax schedule.
    **kwargs : Any
        Forwarded to the optax constructor.

    Returns
    -------
    Any
        An ``optax.GradientTransformation``.

    Raises
    ------
    ValueError
        If ``name`` is not one of the four.
    ImportError
        If optax is not installed. It is declared under the ``applications``
        extra rather than as a core dependency -- ``pip install
        'jaccpot[applications]'``.
    """
    try:
        import optax
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "fit.py needs optax, which jaccpot declares under the "
            "'applications' extra rather than as a core dependency: "
            "pip install 'jaccpot[applications]'"
        ) from exc
    builders = {
        "adam": optax.adam,
        "adamw": optax.adamw,
        "sgd": optax.sgd,
        "rmsprop": optax.rmsprop,
    }
    if name not in builders:
        raise ValueError(
            f"unknown optimizer {name!r}; expected one of {sorted(builders)}"
        )
    return builders[name](learning_rate, **kwargs)


@dataclass(frozen=True)
class FitConfig:
    """Everything that makes one fit reproducible. Recorded verbatim in the JSON.

    Attributes
    ----------
    num_iterations : int
        Gradient steps to take.
    learning_rate : float
        Optimiser step size.
    optimizer : OptimizerName
        Which optax optimiser.
    rebuild_cadence : int
        Rebuild the topology every ``k`` iterations, reusing the interaction
        list in between. ``1`` rebuilds every iteration.
    regularization : Regularization
        Term weights. Terms are evaluated even at weight zero.
    loss_scale : Optional[float]
        Divides the acceleration residual so the loss is O(1). ``None`` derives
        it from the observed field's rms, which is what makes a loss value
        comparable across ``N`` and across noise levels.
    jit_within_epoch : bool
        Compile the step for each topology epoch. See the module docstring for
        when this is the faster choice; the default of ``False`` is right
        whenever the cadence is short.
    track_switches : bool
        Instrument the rebuilds. Off only when host memory is the constraint.
    intensive_every : int
        Sampling stride for the intensive churn rates.
    near_set_sample : int
        Particles to retain near-field source sets for, feeding
        ``near_set_churn``. That one rate dominates the instrumentation cost --
        45 s per comparison at N=32768 with the original 4096-particle default
        -- so it is exposed here rather than buried. ``0`` disables it.
    history_every : int
        Record a history row every ``n`` iterations. The final iteration is
        always recorded.
    seed : int
        Recorded for provenance; the driver itself draws nothing.
    """

    num_iterations: int = 200
    learning_rate: float = 1.0e-3
    optimizer: OptimizerName = "adam"
    rebuild_cadence: int = 1
    regularization: Regularization = field(default_factory=Regularization)
    loss_scale: Optional[float] = None
    jit_within_epoch: bool = False
    track_switches: bool = True
    intensive_every: int = 1
    near_set_sample: int = 128
    history_every: int = 1
    seed: int = 0

    def as_record(self: "FitConfig") -> Dict[str, Any]:
        """Return a JSON-safe copy.

        Returns
        -------
        Dict[str, Any]
            Every field, with the nested regularisation expanded.
        """
        record = asdict(self)
        record["regularization"] = self.regularization.as_record()
        return record


@dataclass
class FitResult:
    """One completed fit: the answer, the path it took, and what moved under it.

    Attributes
    ----------
    params : Any
        Final parameter pytree.
    positions : np.ndarray
        ``(N, 3)`` final source positions.
    history : List[Dict[str, Any]]
        Per-recorded-iteration rows: loss and its components, gradient norms,
        whether the topology was rebuilt, and what changed when it was.
    switch_summary : Optional[Dict[str, Any]]
        :meth:`SwitchLog.summary`, or ``None`` when tracking was off.
    config : Dict[str, Any]
        The :class:`FitConfig` record.
    timing : Dict[str, Any]
        Wall-clock totals, split into prepare / gradient / update, plus the
        first iteration on its own -- it carries the one-time compile and
        averaging it into the rest would misreport both.
    """

    params: Any
    positions: np.ndarray
    history: List[Dict[str, Any]]
    switch_summary: Optional[Dict[str, Any]]
    config: Dict[str, Any]
    timing: Dict[str, Any]

    @property
    def final_loss(self: "FitResult") -> float:
        """Return the last recorded total loss.

        Returns
        -------
        float
            The final row's ``loss``, or ``inf`` for an empty history.
        """
        return float(self.history[-1]["loss"]) if self.history else float("inf")

    @property
    def initial_loss(self: "FitResult") -> float:
        """Return the first recorded total loss.

        Returns
        -------
        float
            The first row's ``loss``, or ``inf`` for an empty history.
        """
        return float(self.history[0]["loss"]) if self.history else float("inf")

    def as_record(self: "FitResult") -> Dict[str, Any]:
        """Return a JSON-safe summary, without the final positions.

        Returns
        -------
        Dict[str, Any]
            Config, history, switch summary, timing and the loss endpoints.
            Positions are omitted deliberately: at ``N = 1e7`` they are 240 MB
            and do not belong in a results JSON.
        """
        return {
            "config": self.config,
            "history": self.history,
            "switch_summary": self.switch_summary,
            "timing": self.timing,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "loss_ratio": (
                self.final_loss / self.initial_loss
                if self.initial_loss not in (0.0, float("inf"))
                else float("nan")
            ),
        }


def _resolve_loss_scale(observed: np.ndarray, configured: Optional[float]) -> float:
    """Pick the residual scale that makes the loss O(1).

    Parameters
    ----------
    observed : np.ndarray
        ``(M, 3)`` observed accelerations.
    configured : Optional[float]
        An explicit scale, or ``None`` to derive one.

    Returns
    -------
    float
        The configured value, or the rms acceleration magnitude. Never zero.
    """
    if configured is not None:
        return float(configured)
    rms = float(np.sqrt(np.mean(np.sum(np.asarray(observed) ** 2, axis=-1))))
    return rms if rms > 0.0 else 1.0


def _shard_parameters(params: Any, devices: Sequence[Any]) -> Any:
    """Place a parameter pytree across a 1-D device mesh.

    Parameters
    ----------
    params : Any
        Parameter pytree of JAX arrays.
    devices : Sequence[Any]
        Two or more devices to shard the leading axis over.

    Returns
    -------
    Any
        The pytree with every leaf whose leading axis divides the device count
        sharded across the mesh, and every other leaf replicated. A 0-d leaf --
        each of the parametric model's scalars -- has no axis to split and is
        replicated; that is correct rather than a fallback, since splitting 7
        numbers across 4 devices would be nonsense.

    Notes
    -----
    Uses ``yggdrax.distributed.make_mesh`` with ``NamedSharding``, which is what
    ``jaccpot/distributed/fmm.py`` and ``jaccpot/mutual/distributed.py`` already
    use, so this section's mesh is built the same way as section 5's.

    The first version of this called ``jax.sharding.PositionalSharding``, which
    **does not exist in jax 0.10.2** -- it was removed, and pyproject's pin is
    load-bearing, so it is not coming back. That went unnoticed because a
    single-device run skips this branch entirely: the multi-GPU path was never
    executed until fig20 first ran on more than one device, and then failed at
    every device count with an ``AttributeError``. It is now exercised by
    ``tests/applications`` on four forced host devices, which needs no GPU.
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    from yggdrax.distributed import make_mesh

    count = len(devices)
    mesh = make_mesh(count, devices=list(devices))
    sharded = NamedSharding(mesh, P(mesh.axis_names[0]))
    replicated = NamedSharding(mesh, P())

    def place(leaf: Any) -> Any:
        array = jnp.asarray(leaf)
        divisible = array.ndim >= 1 and array.shape[0] % count == 0
        return jax.device_put(array, sharded if divisible else replicated)

    return jax.tree_util.tree_map(place, params)


def run_fit(
    *,
    operator: ForwardOperator,
    observed: Any,
    parameterization: Any,
    initial_params: Any,
    config: FitConfig,
    devices: Optional[Sequence[Any]] = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> FitResult:
    """Run the reconstruction fit.

    Parameters
    ----------
    operator : ForwardOperator
        The observation operator. Its ``prepare`` builds each epoch's topology
        and its ``evaluate_at_topology`` is the differentiated map.
    observed : Any
        ``(M, 3)`` observed accelerations at the tracers.
    parameterization : Any
        A :class:`~jaccpot.applications.density_reconstruction.parameterize.PositionsParameterization`
        or
        :class:`~jaccpot.applications.density_reconstruction.parameterize.ParametricParameterization`.
    initial_params : Any
        Starting parameter pytree, from the parameterisation's ``pack``.
    config : FitConfig
        Iterations, step size, cadence, regularisation.
    devices : Optional[Sequence[Any]]
        Devices to shard the parameters over. With more than one, the parameter
        arrays are placed under a single-axis sharding, which is what lets the
        parameter count exceed one device's memory -- the secondary claim of
        section 7. ``None`` leaves placement to JAX.
    progress : Optional[Callable[[Dict[str, Any]], None]]
        Called with each history row as it is recorded. For a long GPU run this
        is the difference between a progress trace and a silent hour.

    Returns
    -------
    FitResult
        The final parameters, the path, and the switch instrumentation.

    Raises
    ------
    ValueError
        If ``config.rebuild_cadence`` is below 1, or if the parameter pytree
        carries a mass leaf -- hard decision 1, checked here as well as at
        construction because this is the function that hands the pytree to
        ``jax.grad``.
    """
    if int(config.rebuild_cadence) < 1:
        raise ValueError(
            f"rebuild_cadence must be >= 1, got {config.rebuild_cadence!r}"
        )

    observed_host = np.asarray(observed, dtype=np.float64)
    scale = _resolve_loss_scale(observed_host, config.loss_scale)
    observed_device = jnp.asarray(observed_host)

    source_masses = np.full(
        (operator.num_sources,), operator.source_mass, dtype=np.float64
    )
    # The invariant, re-checked at the seam that actually differentiates.
    assert_masses_frozen_and_equal(initial_params, source_masses)

    params = jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf), initial_params)
    if devices is not None and len(devices) > 1:
        params = _shard_parameters(params, devices)

    optimizer = make_optimizer(config.optimizer, config.learning_rate)
    opt_state = optimizer.init(params)
    # Imported here rather than at module scope: optax lives under the
    # 'applications' extra, and importing this module must not require it.
    from optax import apply_updates

    switch_log = (
        SwitchLog(
            intensive_every=int(config.intensive_every),
            near_set_sample=int(config.near_set_sample),
        )
        if config.track_switches
        else None
    )

    history: List[Dict[str, Any]] = []
    totals = {"prepare": 0.0, "gradient": 0.0, "update": 0.0}
    first_iteration: Dict[str, float] = {}

    state: Any = None
    blocks: Optional[LeafBlocks] = None
    epoch_step: Optional[Callable[[Any], Tuple[Any, Any]]] = None
    epoch_index = -1

    def value_and_grad_for(current_state: Any, current_blocks: LeafBlocks) -> Any:
        """Build the value-and-grad function for one topology epoch.

        Parameters
        ----------
        current_state : Any
            The epoch's prepared state, a frozen constant.
        current_blocks : LeafBlocks
            The epoch's leaf partition, a frozen constant.

        Returns
        -------
        Any
            ``f(params) -> ((total, components), grads)``.
        """

        def objective(p: Any) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            positions = parameterization.to_positions(p)
            predicted = operator.evaluate_at_topology(current_state, positions)
            extra = regularization_terms(
                positions,
                current_blocks,
                softening=operator.softening,
                regularization=config.regularization,
            )
            return total_loss(
                predicted,
                observed_device,
                weights=config.regularization.weights(),
                scale=scale,
                extra_terms=extra,
            )

        return jax.value_and_grad(objective, has_aux=True)

    started = time.perf_counter()
    for iteration in range(int(config.num_iterations)):
        rebuilt = iteration % int(config.rebuild_cadence) == 0
        changed: Tuple[str, ...] = ()
        if rebuilt:
            t0 = time.perf_counter()
            positions_host = np.asarray(
                jax.device_get(parameterization.to_positions(params)), dtype=np.float64
            )
            state = operator.prepare(positions_host)
            blocks = leaf_blocks_from_state(state, num_sources=operator.num_sources)
            elapsed = time.perf_counter() - t0
            totals["prepare"] += elapsed
            if iteration == 0:
                first_iteration["prepare"] = elapsed
            if switch_log is not None:
                changed = switch_log.observe(state, iteration=iteration)
            epoch_index += 1
            grad_fn = value_and_grad_for(state, blocks)
            epoch_step = jax.jit(grad_fn) if config.jit_within_epoch else grad_fn

        assert epoch_step is not None  # set on the first iteration by construction
        t0 = time.perf_counter()
        (loss_value, components), grads = epoch_step(params)
        loss_value = jax.block_until_ready(loss_value)
        elapsed = time.perf_counter() - t0
        totals["gradient"] += elapsed
        if iteration == 0:
            first_iteration["gradient"] = elapsed

        t0 = time.perf_counter()
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = apply_updates(params, updates)
        elapsed = time.perf_counter() - t0
        totals["update"] += elapsed
        if iteration == 0:
            first_iteration["update"] = elapsed

        last = iteration == int(config.num_iterations) - 1
        if last or iteration % int(config.history_every) == 0:
            row: Dict[str, Any] = {
                "iteration": iteration,
                "epoch": epoch_index,
                "rebuilt": bool(rebuilt),
                "loss": float(loss_value),
                "components": {k: float(v) for k, v in components.items()},
                "gradient": gradient_norm(grads),
            }
            if rebuilt and switch_log is not None:
                row["topology_changed"] = list(changed)
                row["interaction_lists_changed"] = bool(
                    "far_pairs" in changed or "near_pairs" in changed
                )
            history.append(row)
            if progress is not None:
                progress(row)

    wall = time.perf_counter() - started
    final_positions = np.asarray(
        jax.device_get(parameterization.to_positions(params)), dtype=np.float64
    )

    iterations = max(int(config.num_iterations), 1)
    later = max(iterations - 1, 1)
    timing = {
        "wall_seconds": wall,
        "totals_seconds": dict(totals),
        "first_iteration_seconds": dict(first_iteration),
        # The steady-state cost, with the compile-bearing first iteration
        # excluded rather than averaged in.
        "mean_later_gradient_seconds": (
            (totals["gradient"] - first_iteration.get("gradient", 0.0)) / later
        ),
        "mean_later_prepare_seconds": (
            (totals["prepare"] - first_iteration.get("prepare", 0.0))
            / max(len(range(0, iterations, int(config.rebuild_cadence))) - 1, 1)
        ),
        "path": "jit_within_epoch" if config.jit_within_epoch else "eager",
        "num_devices": len(devices) if devices is not None else 1,
        "loss_scale": scale,
    }

    return FitResult(
        params=params,
        positions=final_positions,
        history=history,
        switch_summary=None if switch_log is None else switch_log.summary(),
        config={
            **config.as_record(),
            **parameterization.record(),
            **operator.record(),
        },
        timing=timing,
    )
