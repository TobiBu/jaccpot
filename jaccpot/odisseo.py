"""ODISSEO coupling helpers built on top of the Jaccpot solver API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .config import GradConfig
from .runtime._fmm_impl import PreparedStateLike
from .runtime.fmm_caches import _contains_tracer
from .solver import FastMultipoleMethod


def _extract_positions_from_state(state: Array) -> Array:
    """Extract an ``(N, 3)`` position array from ODISSEO primitive state.

    Parameters
    ----------
    state : Array
        ODISSEO primitive state ``[N, 2, 3]``: slot 0 of the middle axis is
        position, slot 1 is velocity. The shape is checked so a transposed or
        already-flattened array fails loudly here rather than producing a tree
        built on velocities.

    Returns
    -------
    Array
        ``[N, 3]`` positions -- a view-like slice of ``state``, so it stays
        differentiable with respect to it.

    Raises
    ------
    ValueError
        If ``state`` is not three-dimensional with trailing shape ``(2, 3)``.
        A host-side check on the static shape, so it fires at trace time and is
        safe under ``jit``.
    """
    state_arr = jnp.asarray(state)
    if state_arr.ndim != 3 or state_arr.shape[1:] != (2, 3):
        raise ValueError("state must have shape (N, 2, 3)")
    return state_arr[:, 0, :]


@dataclass
class OdisseoFMMCoupler:
    """Adapter that lets ODISSEO reuse one prepared FMM state across steps.

    ODISSEO integrates with a fixed particle set whose positions move each step, so
    the expensive part -- the tree build and interaction lists -- can be built once
    and reused while only the numerics are re-evaluated. This holds that state and
    the masses it was built against.

    Not frozen, because :meth:`prepare` and :meth:`clear` mutate the cache. Not
    thread-safe, and not a pytree: the cached state is a host-side object, so a
    coupler instance must not be closed over by a jitted function.

    **The cache is not self-invalidating**, and that is the thing to know before
    using this in a long run. :meth:`accelerations` reuses the cached tree unless
    you pass ``rebuild_sources=True``; it does not measure drift and will not warn.
    Once the particles have moved far enough that the cached partition is wrong, the
    forces come back computed against the stale partition, with no error and no NaN.
    The caller owns the rebuild cadence. :meth:`clear` drops the cache outright.

    Two private fields hold the cache -- ``_prepared_state`` (``None`` before the
    first :meth:`prepare`) and ``_masses`` (the masses that state was built
    against), which is why ``masses`` is required on the first
    :meth:`accelerations` call and optional afterwards.

    Attributes
    ----------
    solver : FastMultipoleMethod
        The solver used to build and evaluate state.
    leaf_size : int
        Default leaf size for :meth:`prepare`, overridable per call.
    max_order : int
        Default expansion order for :meth:`prepare`, overridable per call.
    """

    solver: FastMultipoleMethod
    leaf_size: int = 16
    max_order: int = 4
    # `PreparedStateLike`, not `FMMPreparedState`: `solver.prepare_state` returns
    # the union and the large-N path returns a `LargeNPreparedState`, which the
    # narrower annotation denied. Widening rather than narrowing, so nothing that
    # worked stops working (audit E.5).
    _prepared_state: Optional[PreparedStateLike] = None
    _masses: Optional[Array] = None

    def clear(self: "OdisseoFMMCoupler") -> None:
        """Drop the cached prepared-state payload."""
        self._prepared_state = None
        self._masses = None

    def prepare(
        self: "OdisseoFMMCoupler",
        state: Float[Array, "n 2 3"],
        masses: Float[Array, "n"],
        *,
        bounds: Optional[Tuple[Float[Array, "3"], Float[Array, "3"]]] = None,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
    ) -> PreparedStateLike:
        """Prepare source tree/interactions from an ODISSEO primitive state.

        Overwrites the cache unconditionally -- both ``_prepared_state`` and the
        ``_masses`` it was built against. Calling this is the explicit rebuild
        that :meth:`accelerations` will not do for you.

        Parameters
        ----------
        state : Float[Array, 'n 2 3']
            ODISSEO primitive state ``[N, 2, 3]``; only the position slot is
            read. Velocities do not enter the tree.
        masses : Float[Array, 'n']
            ``[N]`` particle masses. Retained on the instance, which is what
            makes ``masses`` optional on later :meth:`accelerations` calls.
        bounds : Optional[Tuple[Float[Array, '3'], Float[Array, '3']]]
            ``(lower, upper)`` root-box corners, or ``None`` (the default) to
            derive them from the positions. Supplying fixed bounds across steps
            is what keeps the tree topology -- and so the compiled shapes --
            stable in a run.
        leaf_size : Optional[int]
            Particles per leaf; ``None`` (the default) uses the instance's
            ``leaf_size``.
        max_order : Optional[int]
            Expansion order; ``None`` (the default) uses the instance's
            ``max_order``.

        Returns
        -------
        PreparedStateLike
            The prepared state, also stored on the instance. Returned so a
            caller can hold it directly; the two are the same object, so
            :meth:`clear` does not invalidate a reference already taken.
        """
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
        state: Float[Array, "n 2 3"],
        masses: Optional[Float[Array, "n"]] = None,
        *,
        active_indices: Optional[Int[Array, "t"]] = None,
        return_potential: bool = False,
        rebuild_sources: bool = False,
        bounds: Optional[Tuple[Float[Array, "3"], Float[Array, "3"]]] = None,
        differentiable: bool = False,
        grad_config: Optional[GradConfig] = None,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Evaluate accelerations for all particles or active targets only.

        When ``rebuild_sources=False`` this reuses the cached source tree and
        evaluates only the requested targets for fast active-particle substeps.

        Parameters
        ----------
        state : Float[Array, 'n 2 3']
            ODISSEO primitive state, shape ``(N, 2, 3)``.
        masses : Optional[Float[Array, 'n']]
            Particle masses; required on the first call, cached thereafter.
        active_indices : Optional[Int[Array, 't']]
            Evaluate only these targets (all particles remain sources).
        return_potential : bool
            Also return the potential. Not available on the differentiable path.
        rebuild_sources : bool
            Rebuild the tree from ``state`` before evaluating.
        bounds : Optional[Tuple[Float[Array, '3'], Float[Array, '3']]]
            Explicit ``(lower, upper)`` domain bounds for tree construction,
            forwarded to :meth:`prepare`. Used only when a rebuild actually
            happens -- i.e. when ``rebuild_sources=True`` or no state is cached
            yet; otherwise it is silently ignored.
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

            Also raised on the differentiable path when a rebuild is needed while
            tracing (``prepare_state`` is not traceable, so the source tree must
            be built outside the differentiated function), and when
            ``return_potential=True`` is combined with ``differentiable=True``
            (the potential half of the near-field ``custom_vjp`` is not wired).
        ValueError
            If ``masses`` is ``None`` on a call that needs to build or rebuild
            the tree and nothing has been cached by a previous
            :meth:`prepare`/evaluation.
        RuntimeError
            If the prepared state is missing after a successful prepare. A
            defensive invariant check, not a reachable user error.
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


__all__ = [
    "OdisseoFMMCoupler",
]
