"""Differentiable re-evaluation seam for the large-N (``preset="large_n_gpu"``) path.

``evaluate_large_n_state`` is forward-only by construction: it takes no positions
or masses and evaluates L2P against the **prebaked** ``state.local_data``. Simply
lifting the reject in ``differentiable_accelerations`` would therefore not raise --
it would return a plausible but **wrong** gradient, with the whole
P2M -> M2M -> M2L -> L2L source-side chain treated as a constant and
``d(accel)/d(mass)`` through the far field exactly zero.

This module supplies the missing seam: re-run the numeric pipeline on live
positions/masses with every discrete artifact (Morton order, node membership, MAC
decisions, the M2L pair list, the near-field partition, leaf packing) frozen from
the prepared state, mirroring the radix seam in
``fmm_evaluate._evaluate_prepared_state_at_positions_sorted``.

Two structural differences from the radix seam:

* ``LargeNPreparedState.interactions`` is unconditionally ``None`` and ``.downward``
  is a compat view with no ``.locals``, so the frozen M2L list has to come from
  ``state.compact_far_pairs`` (retained only when ``retain_far_pairs_for_grad`` is
  set) fed through ``far_pairs_coo``, and the expansion order from
  ``state.local_order``.
* the near field is the fused Pallas radix fast lane rather than the bucketed
  pure-JAX path, so it is driven with ``differentiable=True`` to route through the
  lanes' ``custom_vjp`` wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._large_n_nearfield import evaluate_large_n_nearfield_fast_lane
from ._large_n_types import LargeNPreparedState
from .dtypes import INDEX_DTYPE
from .kernels.core import _evaluate_local_expansions_for_particles, _FarPairCOO


@dataclass(frozen=True)
class LargeNGradPlan:
    """Frozen-topology discriminators for a differentiable large-N evaluation.

    Built once, concretely, outside any trace; captured as a constant by the
    differentiated function. Hoist it out of an optimisation loop to avoid
    rebuilding it per step.
    """

    far_pair_sources: Array
    far_pair_targets: Array
    far_pair_active_count: Optional[Array]
    order: int
    max_leaf_size: int
    center_mode: str
    farfield_mode: str
    m2l_chunk_size: Optional[int]
    l2l_chunk_size: Optional[int]
    num_far_pairs: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(message)


def _engine(fmm: Any) -> Any:
    """Accept either the public ``FMMEngine`` facade or the runtime engine.

    The sweeps and the override resolver live on the runtime engine; the public
    solver holds it as ``_impl``. Callers reach this module both ways -- from
    ``differentiable_accelerations`` (already the engine) and from user code
    holding the facade -- so normalise rather than fail on an attribute lookup.
    """
    return getattr(fmm, "_impl", fmm)


def prepare_large_n_grad_plan(
    fmm: Any,
    state: LargeNPreparedState,
) -> LargeNGradPlan:
    """Validate a large-N state for differentiation and freeze its discriminators.

    Every rejection here is loud on purpose. A partially-wired large-N seam does
    not fail -- it returns an incomplete gradient -- so anything unsupported must
    raise rather than degrade.
    """
    _require(
        isinstance(state, LargeNPreparedState),
        "prepare_large_n_grad_plan requires a LargeNPreparedState.",
    )
    _require(
        bool(getattr(state, "radix_fast_lane", False)),
        "differentiable large-N evaluation requires the radix fast lane "
        "(state.radix_fast_lane is False).",
    )
    _require(
        getattr(state, "radix_fast_payload", None) is not None,
        "differentiable large-N evaluation requires state.radix_fast_payload.",
    )
    _require(
        str(getattr(state, "expansion_basis", "")).strip().lower() == "solidfmm",
        "differentiable large-N evaluation supports expansion_basis='solidfmm' "
        f"only; got {getattr(state, 'expansion_basis', None)!r}.",
    )

    # Near-field lanes that exist but are not wired for autodiff must REJECT, not
    # be silently dropped -- omitting a lane would differentiate a different force
    # than the forward computes. The active layout is N-dependent (measured: the
    # per-particle "pairs" layout at N=65536, the prepacked layout at N=200000), so
    # this is not hypothetical.
    overflow = getattr(state, "radix_overflow_payload", None)
    _require(
        overflow is None,
        "differentiable large-N evaluation does not yet cover the radix OVERFLOW "
        "near-field payload; it is populated for this state, and ignoring it would "
        "silently differentiate a different force than the forward computes.",
    )
    target_block_ids = getattr(state, "nearfield_target_block_source_leaf_ids", None)
    _require(
        target_block_ids is None or int(target_block_ids.size) == 0,
        "differentiable large-N evaluation does not yet cover the target-block "
        "near-field payload; it is populated for this state (see above).",
    )
    payload = state.radix_fast_payload
    _require(
        int(getattr(payload, "source_particle_ids").size) == 0,
        "differentiable large-N evaluation currently covers the PREPACKED "
        "source-leaf-id near-field layout only, but this state uses the "
        "materialized per-particle 'pairs' layout (payload.source_particle_ids is "
        "non-empty). The pairs lane needs its own analytic reverse; a larger "
        "leaf_target or particle count selects the prepacked layout.",
    )

    compact = getattr(state, "compact_far_pairs", None)
    if compact is None:
        raise RuntimeError(
            "LargeNPreparedState.compact_far_pairs is None: the frozen M2L pair "
            "list was discarded during prepare_state (it is retained only for "
            "adaptive_order or the strict static_radix lane). Without it the far "
            "field would have to be treated as a CONSTANT, which yields a silently "
            "wrong gradient -- zero mass-sensitivity through P2M/M2M/M2L/L2L -- "
            "rather than an error. Rebuild the state with "
            "retain_far_pairs_for_grad=True (FarFieldConfig field, constructor "
            "kwarg, or JACCPOT_RETAIN_FAR_PAIRS_FOR_GRAD=1)."
        )

    positions = state.positions_sorted
    if isinstance(positions, jax.core.Tracer):
        raise RuntimeError(
            "prepare_large_n_grad_plan must be called OUTSIDE any trace: the plan "
            "freezes topology as concrete constants. Build the state and the plan "
            "first, then differentiate the evaluation."
        )

    engine = _engine(fmm)
    overrides = engine._resolve_runtime_execution_overrides(
        num_particles=int(positions.shape[0])
    )
    _require(
        not bool(overrides.grouped_interactions),
        "differentiable large-N evaluation requires ungrouped far-field execution; "
        "the grouped M2L classifies pairs on the host and is not traceable.",
    )

    sources = jnp.asarray(compact.sources, dtype=INDEX_DTYPE)
    targets = jnp.asarray(compact.targets, dtype=INDEX_DTYPE)
    return LargeNGradPlan(
        far_pair_sources=sources,
        far_pair_targets=targets,
        far_pair_active_count=getattr(compact, "far_pair_count", None),
        order=int(state.local_order),
        max_leaf_size=int(state.max_leaf_size),
        center_mode=str(overrides.center_mode),
        farfield_mode=str(overrides.farfield_mode),
        m2l_chunk_size=overrides.m2l_chunk_size,
        l2l_chunk_size=overrides.l2l_chunk_size,
        num_far_pairs=int(sources.shape[0]),
    )


def large_n_farfield_locals_at(
    fmm: Any,
    state: LargeNPreparedState,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    plan: LargeNGradPlan,
) -> Any:
    """Re-run P2M -> M2M -> M2L -> L2L on live inputs at frozen topology.

    This is the piece that makes the far-field gradient real: it returns FRESH
    locals instead of the prebaked ``state.local_data``, so cotangents reach the
    source-side chain and ``d/d(mass)`` through the far field is nonzero.

    The M2L runs against the frozen pair list via ``far_pairs_coo``. Note
    ``active_count`` is consumed as a MASK rather than a loop bound, which is what
    keeps it reverse-mode safe.
    """
    engine = _engine(fmm)
    upward = engine.prepare_upward_sweep(
        state.tree,
        positions_sorted,
        masses_sorted,
        max_order=plan.order,
        center_mode=plan.center_mode,
        max_leaf_size=plan.max_leaf_size,
    )
    downward = engine.prepare_downward_sweep(
        state.tree,
        upward,
        theta=float(state.theta),
        mac_type=engine.mac_type,
        initial_locals=None,
        interactions=None,
        m2l_chunk_size=plan.m2l_chunk_size,
        l2l_chunk_size=plan.l2l_chunk_size,
        grouped_interactions=False,
        farfield_mode=plan.farfield_mode,
        dehnen_radius_scale=engine.dehnen_radius_scale,
        far_pairs_coo=_FarPairCOO(
            plan.far_pair_sources,
            plan.far_pair_targets,
            plan.far_pair_active_count,
        ),
        far_pairs_by_gear=((plan.far_pair_sources, plan.far_pair_targets),),
        adaptive_order=True,
        p_gears=(plan.order,),
    )
    return downward.locals


def evaluate_large_n_state_at_positions_and_masses_sorted(
    fmm: Any,
    state: LargeNPreparedState,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    plan: LargeNGradPlan,
    include_near: bool = True,
    include_far: bool = True,
) -> Array:
    """Fixed-topology large-N re-evaluation at live inputs, in SORTED order.

    Mirrors the production ``evaluate_large_n_state`` accel fast lane expression
    for expression, with one deliberate difference: L2P is evaluated against the
    locals recomputed by :func:`large_n_farfield_locals_at` rather than the frozen
    ``state.local_data``.

    ``include_near`` / ``include_far`` exist for the diagnostic split that the
    correctness gate relies on: with the near field disabled, a mass gradient that
    comes out exactly zero proves the far-field chain is not wired.
    """
    positions = jnp.asarray(positions_sorted, dtype=state.working_dtype)
    masses = jnp.asarray(masses_sorted, dtype=state.working_dtype)
    if positions.shape != state.positions_sorted.shape:
        raise ValueError(
            "positions_sorted must have shape "
            f"{tuple(state.positions_sorted.shape)}, got {tuple(positions.shape)}"
        )
    if masses.shape != state.masses_sorted.shape:
        raise ValueError(
            "masses_sorted must have shape "
            f"{tuple(state.masses_sorted.shape)}, got {tuple(masses.shape)}"
        )

    engine = _engine(fmm)
    total = jnp.zeros_like(positions)

    if include_far:
        locals_live = large_n_farfield_locals_at(
            engine, state, positions, masses, plan=plan
        )
        far_grad, _, _ = _evaluate_local_expansions_for_particles(
            locals_live,
            positions,
            leaf_nodes=jnp.asarray(state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE),
            node_ranges=jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE),
            max_leaf_size=plan.max_leaf_size,
            order=plan.order,
            expansion_basis="solidfmm",
            return_potential=False,
            max_acc_derivative_order=0,
        )
        total = total + (-float(getattr(engine, "G")) * far_grad)

    if include_near:
        near_acc = evaluate_large_n_nearfield_fast_lane(
            engine,
            state,
            return_potential=False,
            positions_sorted=positions,
            masses_sorted=masses,
            differentiable=True,
        )
        total = total + near_acc

    return total
