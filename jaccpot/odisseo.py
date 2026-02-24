"""ODISSEO coupling helpers built on top of the Jaccpot solver API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array

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

    def clear(self) -> None:
        """Drop the cached prepared-state payload."""
        self._prepared_state = None
        self._masses = None

    def prepare(
        self,
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
        self,
        state: Array,
        masses: Optional[Array] = None,
        *,
        active_indices: Optional[Array] = None,
        return_potential: bool = False,
        rebuild_sources: bool = False,
        bounds: Optional[Tuple[Array, Array]] = None,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Evaluate accelerations for all particles or active targets only.

        When ``rebuild_sources=False`` this reuses the cached source tree and
        evaluates only the requested targets for fast active-particle substeps.
        """
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
