"""ODISSEO coupling helpers built on top of the Jaccpot solver API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array

from .config import GradConfig
from .runtime.fmm_caches import _contains_tracer
from .solver import FastMultipoleMethod, FMMPreparedState


def _extract_positions_from_state(state: Array) -> Array:
    """Extract an ``(N, 3)`` position array from ODISSEO primitive state."""
    state_arr = jnp.asarray(state)
    if state_arr.ndim != 3 or state_arr.shape[1:] != (2, 3):
        raise ValueError("state must have shape (N, 2, 3)")
    return state_arr[:, 0, :]


@dataclass
class OdisseoFMMCoupler:
    """Cache-oriented adapter for coupling ODISSEO and Jaccpot FMM."""

    solver: FastMultipoleMethod
    leaf_size: int = 16
    max_order: int = 4
    _prepared_state: Optional[FMMPreparedState] = None
    _masses: Optional[Array] = None

    def clear(self: "OdisseoFMMCoupler") -> None:
        """Drop the cached prepared-state payload."""
        self._prepared_state = None
        self._masses = None

    def prepare(
        self: "OdisseoFMMCoupler",
        state: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
    ) -> FMMPreparedState:
        """Prepare source tree/interactions from an ODISSEO primitive state."""
        positions = _extract_positions_from_state(state)
        state_prepared = self.solver.prepare_state(
            positions,
            masses,
            bounds=bounds,
            leaf_size=self.leaf_size if leaf_size is None else int(leaf_size),
            max_order=self.max_order if max_order is None else int(max_order),
        )
        self._prepared_state = state_prepared
        self._masses = jnp.asarray(masses)
        return state_prepared

    def accelerations(
        self: "OdisseoFMMCoupler",
        state: Array,
        masses: Optional[Array] = None,
        *,
        active_indices: Optional[Array] = None,
        return_potential: bool = False,
        rebuild_sources: bool = False,
        bounds: Optional[Tuple[Array, Array]] = None,
        differentiable: bool = False,
        grad_config: Optional[GradConfig] = None,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Evaluate accelerations for all particles or active targets only.

        When ``rebuild_sources=False`` this reuses the cached source tree and
        evaluates only the requested targets for fast active-particle substeps.

        Parameters
        ----------
        state : Array
            ODISSEO primitive state, shape ``(N, 2, 3)``.
        masses : Optional[Array]
            Particle masses; required on the first call, cached thereafter.
        active_indices : Optional[Array]
            Evaluate only these targets (all particles remain sources).
        return_potential : bool
            Also return the potential. Not available on the differentiable path.
        rebuild_sources : bool
            Rebuild the tree from ``state`` before evaluating.
        differentiable : bool
            Route through :meth:`~jaccpot.FastMultipoleMethod.differentiable_accelerations`
            so ``jax.grad`` over this call gives exact fixed-topology gradients
            w.r.t. the positions carried by ``state`` and w.r.t. ``masses``. The
            default forward path evaluates the **prebaked** expansions and takes
            no live inputs, so its gradient is identically zero -- see below.
        grad_config : Optional[GradConfig]
            Gradient-path options, forwarded when ``differentiable=True``.

        Returns
        -------
        Union[Array, Tuple[Array, Array]]
            ``(N, 3)`` accelerations, or ``(accelerations, potential)``.

        Raises
        ------
        NotImplementedError
            If traced inputs reach the forward path. That path routes to
            :meth:`~jaccpot.FastMultipoleMethod.evaluate_prepared_state`, which
            reads the frozen ``state.downward`` locals and never touches the live
            positions or masses. Differentiating it therefore returns **exactly
            zero** rather than failing -- the worst outcome available, so it is
            rejected explicitly instead. Pass ``differentiable=True``.
        """
        traced = _contains_tracer(state) or _contains_tracer(masses)
        if traced and not differentiable:
            raise NotImplementedError(
                "OdisseoFMMCoupler.accelerations received traced positions or "
                "masses on the forward path. That path evaluates the PREBAKED "
                "expansions from the cached prepared state and never reads the "
                "live inputs, so jax.grad over it would return exactly zero "
                "instead of raising. Pass differentiable=True to route through "
                "FastMultipoleMethod.differentiable_accelerations, which re-runs "
                "the numeric pipeline on the live inputs at frozen topology."
            )

        if differentiable:
            if rebuild_sources or self._prepared_state is None:
                # prepare_state is not traceable, so the topology cannot be built
                # inside the differentiated function. This is the fixed-topology
                # contract, not a limitation of the coupler.
                if traced:
                    raise NotImplementedError(
                        "the source tree must be built OUTSIDE the differentiated "
                        "function: prepare_state is not traceable. Call "
                        "coupler.prepare(state0, masses) once with concrete "
                        "inputs, then differentiate accelerations(..., "
                        "differentiable=True)."
                    )
                masses_arr = self._masses if masses is None else masses
                if masses_arr is None:
                    raise ValueError(
                        "masses must be provided on first prepare/evaluation"
                    )
                self.prepare(state, masses_arr, bounds=bounds)
            prepared_state = self._prepared_state
            if prepared_state is None:
                raise RuntimeError("prepared state is unexpectedly missing")
            if return_potential:
                raise NotImplementedError(
                    "differentiable=True is an acceleration-only path; the "
                    "potential half of the near-field custom_vjp is not wired."
                )
            masses_arr = self._masses if masses is None else masses
            if masses_arr is None:
                raise ValueError("masses must be provided on first prepare/evaluation")
            return self.solver.differentiable_accelerations(
                prepared_state,
                _extract_positions_from_state(state),
                jnp.asarray(masses_arr),
                target_indices=active_indices,
                grad_config=grad_config,
            )

        if rebuild_sources or self._prepared_state is None:
            masses_arr = self._masses if masses is None else masses
            if masses_arr is None:
                raise ValueError("masses must be provided on first prepare/evaluation")
            self.prepare(
                state,
                masses_arr,
                bounds=bounds,
            )
        prepared_state = self._prepared_state
        if prepared_state is None:
            raise RuntimeError("prepared state is unexpectedly missing")
        return self.solver.evaluate_prepared_state(
            prepared_state,
            target_indices=active_indices,
            return_potential=return_potential,
        )
