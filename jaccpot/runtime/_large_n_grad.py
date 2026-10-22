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
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaccpot._jax_compat import Tracer

from ._large_n_nearfield import evaluate_large_n_nearfield_fast_lane
from ._large_n_types import LargeNPreparedState
from .dtypes import INDEX_DTYPE
from .kernels._downward_prep import (
    _SymmetricTargetSortedFarPairCOO,
    _TargetSortedFarPairCOO,
)
from .kernels.core import (
    _evaluate_local_expansions_for_particles,
    _evaluate_local_expansions_particle_major,
    _FarPairCOO,
)

__all__ = [
    "LargeNGradPlan",
    "evaluate_large_n_state_at_positions_and_masses_sorted",
    "large_n_farfield_l2p_at",
    "large_n_farfield_locals_at",
    "prepare_large_n_grad_plan",
    "resolve_grad_force_params",
]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["far_pair_sources", "far_pair_targets", "far_pair_active_count"],
    meta_fields=[
        "order",
        "max_leaf_size",
        "center_mode",
        "m2l_chunk_size",
        "l2l_chunk_size",
        "num_far_pairs",
        "far_pairs_target_sorted",
        "far_pairs_symmetric",
    ],
)
@dataclass(frozen=True)
class LargeNGradPlan:
    """Frozen-topology discriminators for a differentiable large-N evaluation.

    Built once, concretely, outside any trace. Hoist it out of an optimisation loop
    to avoid rebuilding it per step. A pytree (the pair arrays are its leaves, the
    rest static), so a jitted gradient can take it -- and the prepared state -- as
    ARGUMENTS: closed over, ``jax.jit`` embeds every array of the topology in the
    program as a constant (a second copy of the far and near lists, and XLA
    constant-folds through them at compile time).

    Attributes
    ----------
    far_pair_sources : Array
        Source node of each frozen far pair.
    far_pair_targets : Array
        Target node of each frozen far pair; with ``far_pairs_target_sorted``, the
        list's CSR row offsets instead (``[total_nodes + 1]``).
    far_pair_active_count : Optional[Array]
        Live pair count when the lists are padded. Consumed as a MASK, not as a
        loop bound -- that is what keeps the M2L reverse-mode safe.
    order : int
        Expansion order the plan was frozen at.
    max_leaf_size : int
        Leaf slot capacity.
    center_mode : str
        How expansion centres are chosen.
    m2l_chunk_size : Optional[int]
        M2L chunk size; ``None`` leaves it to the runtime.
    l2l_chunk_size : Optional[int]
        L2L chunk size; ``None`` leaves it to the runtime.
    num_far_pairs : int
        Allocated pair capacity, as a Python int so it can size a scan.
    far_pairs_target_sorted : bool
        The frozen list is the walk's ``TargetSortedFarPairs`` (CSR order,
        ``far_pair_targets`` its row offsets): the M2L reads it as the production
        refresh hands it over, without expanding a target per pair or sorting.
    far_pairs_symmetric : bool
        The target-sorted list holds every directed pair both ways (checked at
        plan time, :func:`_csr_pairs_symmetric`): the M2L reverse reads the
        forward's CSR in both passes and sorts nothing.
    """

    far_pair_sources: Array
    far_pair_targets: Array
    far_pair_active_count: Optional[Array]
    order: int
    max_leaf_size: int
    center_mode: str
    m2l_chunk_size: Optional[int]
    l2l_chunk_size: Optional[int]
    num_far_pairs: int
    far_pairs_target_sorted: bool = False
    far_pairs_symmetric: bool = False


def _require(condition: bool, message: str) -> None:
    """Raise ``NotImplementedError`` unless ``condition`` holds.

    NotImplementedError rather than ValueError on purpose: every rejection in
    this module means a large-N seam is not wired for differentiation, which is a
    gap in the implementation and not bad user input.

    Parameters
    ----------
    condition : bool
        The requirement being checked.
    message : str
        What is unsupported, phrased for the user who hit it.

    Returns
    -------
    None
        Returns normally when ``condition`` holds.

    Raises
    ------
    NotImplementedError
        If ``condition`` is false.
    """
    if not condition:
        raise NotImplementedError(message)


def _engine(fmm: Any) -> Any:
    """Accept either the public ``FMMEngine`` facade or the runtime engine.

    The sweeps and the override resolver live on the runtime engine; the public
    solver holds it as ``_impl``. Callers reach this module both ways -- from
    ``differentiable_accelerations`` (already the engine) and from user code
    holding the facade -- so normalise rather than fail on an attribute lookup.

    Parameters
    ----------
    fmm : Any
        Either the public facade or the runtime engine.

    Returns
    -------
    Any
        The runtime engine -- ``fmm._impl`` when given the facade, otherwise
        ``fmm`` unchanged.
    """
    return getattr(fmm, "_impl", fmm)


def _pair_hash_sums(a: Array, b: Array, live: Array) -> Array:
    """Order-free sums of two hashes of every live directed pair, both ways round.

    Parameters
    ----------
    a : Array
        ``[E]`` one end of each pair.
    b : Array
        ``[E]`` the other end, aligned.
    live : Array
        ``[E]`` which entries are pairs.

    Returns
    -------
    Array
        ``uint64[2, 2]``: ``[k] = (sum h_k(a, b), sum h_k(b, a))`` over the live
        entries, ``h_k`` the splitmix64 finalizer of a seeded combination (the sums
        wrap; integer addition is exact in any order).
    """

    def mix(x: Array) -> Array:
        x = (x ^ (x >> 30)) * jnp.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * jnp.uint64(0x94D049BB133111EB)
        return x ^ (x >> 31)

    a64, b64 = a.astype(jnp.uint64), b.astype(jnp.uint64)
    zero = jnp.zeros((), jnp.uint64)
    rows = []
    for seed in (0x9E3779B97F4A7C15, 0xD1B54A32D192ED03):
        k = jnp.uint64(seed)

        def h(x: Array, y: Array, k: Array = k) -> Array:
            return mix(mix(x ^ k) + y * jnp.uint64(0xC2B2AE3D27D4EB4F))

        rows.append(
            jnp.stack(
                [
                    jnp.sum(jnp.where(live, h(a64, b64), zero)),
                    jnp.sum(jnp.where(live, h(b64, a64), zero)),
                ]
            )
        )
    return jnp.stack(rows)


#: entries hashed per launch: the check reads the lists in chunks, so it needs no
#: array of the lists' length (at 1e8 the far list's capacity is 542M entries)
_SYMMETRY_CHUNK = 1 << 24


@partial(jax.jit, static_argnames=("size",))
def _csr_pair_hash_sums(
    entries: Array,
    row_starts: Array,
    counts: Array,
    base: Array,
    shift: Array,
    *,
    size: int,
) -> Array:
    """:func:`_pair_hash_sums` of the CSR pairs ``(row, entry)`` in one chunk.

    Parameters
    ----------
    entries : Array
        ``[E]`` the partner of each entry.
    row_starts : Array
        ``[R]`` non-decreasing row starts.
    counts : Array
        ``[R]`` live entries per row (rows may be padded).
    base : Array
        First entry of the chunk.
    shift : Array
        Subtracted from each entry (node ids to row indices).
    size : int
        Entries per chunk. Static.

    Returns
    -------
    Array
        As :func:`_pair_hash_sums`, over entries ``[base, base + size)``.
    """
    num = entries.shape[0]
    pos = base + jnp.arange(size, dtype=INDEX_DTYPE)
    row = jnp.searchsorted(row_starts, pos, side="right").astype(INDEX_DTYPE) - 1
    safe = jnp.maximum(row, 0)
    live = (pos < num) & (row >= 0) & (pos - row_starts[safe] < counts[safe])
    partner = entries[jnp.minimum(pos, num - 1)] - shift
    return _pair_hash_sums(safe, partner, live)


def _csr_pairs_symmetric(
    entries: Array, row_starts: Array, counts: Array, shift: Any = 0
) -> bool:
    """Whether a CSR's pairs ``(row, entry)`` are, as a multiset, their reverses.

    A probabilistic multiset test (two independent 64-bit hashes summed over the
    pairs, no sort): a list that is not symmetric passes with probability about
    ``2^-64``.

    Parameters
    ----------
    entries : Array
        ``[E]`` the partner of each entry (same index space as the rows).
    row_starts : Array
        ``[R]`` (or ``[R + 1]``) non-decreasing row starts.
    counts : Array
        ``[R]`` live entries per row.
    shift : Any
        Subtracted from each entry first (a neighbour list's leaf node offset).

    Returns
    -------
    bool
        ``True`` when both hash sums agree both ways round.
    """
    entries = jnp.asarray(entries, INDEX_DTYPE)
    shift = jnp.asarray(shift, INDEX_DTYPE)
    cnt = jnp.asarray(counts, INDEX_DTYPE)
    # a CSR's offsets may carry the end as an R + 1-th entry: the rows are R
    starts = jnp.asarray(row_starts, INDEX_DTYPE)[: int(cnt.shape[0])]
    num = int(entries.shape[0])
    size = min(_SYMMETRY_CHUNK, 1 << max(0, (num - 1).bit_length()))
    total = np.zeros((2, 2), np.uint64)
    for base in range(0, num, size):
        part = _csr_pair_hash_sums(
            entries,
            starts,
            cnt,
            jnp.asarray(base, INDEX_DTYPE),
            shift,
            size=size,
        )
        total = total + np.asarray(part, np.uint64)  # wraps, as the device sums do
    return bool(np.all(total[:, 0] == total[:, 1]))


def _directed_pairs_symmetric(a: Array, b: Array, live: Array) -> bool:
    """:func:`_csr_pairs_symmetric` for pairs given end by end (small lists).

    Parameters
    ----------
    a : Array
        ``[E]`` one end of each pair.
    b : Array
        ``[E]`` the other end.
    live : Array
        ``[E]`` which entries are pairs.

    Returns
    -------
    bool
        ``True`` when both hash sums agree both ways round.
    """
    sums = np.asarray(jax.jit(_pair_hash_sums)(a, b, live))
    return bool(np.all(sums[:, 0] == sums[:, 1]))


def _near_list_symmetric(neighbor_list: Any) -> bool:
    """Whether the leaf neighbour CSR holds every (leaf, leaf) pair both ways.

    Parameters
    ----------
    neighbor_list : Any
        The state's ``neighbor_list`` (row starts, counts, neighbour node ids,
        leaf node ids).

    Returns
    -------
    bool
        As :func:`_csr_pairs_symmetric`.
    """
    leaf_nodes = jnp.asarray(neighbor_list.leaf_indices, INDEX_DTYPE)
    return _csr_pairs_symmetric(
        neighbor_list.neighbors,
        neighbor_list.offsets,
        neighbor_list.counts,
        shift=leaf_nodes[0],
    )


def _far_list_symmetric(sources: Array, row_offsets: Array, total_nodes: int) -> bool:
    """Whether a target-sorted far list holds every (source, target) pair both ways.

    Parameters
    ----------
    sources : Array
        ``[P]`` sources in CSR order (padding past the live prefix).
    row_offsets : Array
        ``[>= total_nodes + 1]`` row offsets by target.
    total_nodes : int
        Rows. Static.

    Returns
    -------
    bool
        As :func:`_csr_pairs_symmetric`.
    """
    offsets = jnp.asarray(row_offsets, INDEX_DTYPE)[: total_nodes + 1]
    return _csr_pairs_symmetric(sources, offsets[:-1], offsets[1:] - offsets[:-1])


def prepare_large_n_grad_plan(
    fmm: Any,
    state: LargeNPreparedState,
) -> LargeNGradPlan:
    """Validate a large-N state for differentiation and freeze its discriminators.

    Every rejection here is loud on purpose. A partially-wired large-N seam does
    not fail -- it returns an incomplete gradient -- so anything unsupported must
    raise rather than degrade.

    Parameters
    ----------
    fmm : Any
        Facade or runtime engine; normalised by :func:`_engine`.
    state : LargeNPreparedState
        Prepared state to freeze. Must carry a retained far-pair list --
        ``retain_far_pairs_for_grad=True`` at construction -- since the M2L
        recomputation runs against it.

    Returns
    -------
    LargeNGradPlan
        The frozen discriminators, safe to capture as a constant.

    Raises
    ------
    RuntimeError
        If the state is missing the retained far pairs, or its configuration is
        one this seam cannot differentiate. Loud rather than degraded: a
        partially-wired seam returns an incomplete gradient, which nothing
        downstream would notice.
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
            "LargeNPreparedState.compact_far_pairs is None: this state has no "
            "frozen M2L pair list. Without it the far field would have to be "
            "treated as a CONSTANT, which yields a silently wrong gradient -- zero "
            "mass-sensitivity through P2M/M2M/M2L/L2L -- rather than an error. "
            "Either prepare_state discarded it (it is retained only for "
            "adaptive_order or the strict static_radix lane): rebuild the state "
            "with retain_far_pairs_for_grad=True (FarFieldConfig field, "
            "constructor kwarg, or JACCPOT_RETAIN_FAR_PAIRS_FOR_GRAD=1). Or the "
            "state was returned by strict_run_v2, whose scan rebuilds the far list "
            "every step and keeps none (the one it started with belongs to the "
            "start positions): call prepare_state at the returned positions and "
            "differentiate that state."
        )

    positions = state.positions_sorted
    if isinstance(positions, Tracer):
        raise RuntimeError(
            "prepare_large_n_grad_plan must be called OUTSIDE any trace: the plan "
            "freezes topology as concrete constants. Build the state and the plan "
            "first, then differentiate the evaluation."
        )

    engine = _engine(fmm)
    overrides = engine._resolve_runtime_execution_overrides(
        num_particles=int(positions.shape[0])
    )
    from ._interaction_cache import TargetSortedFarPairs

    # The direct near layout's reverse takes each source-side cotangent from the
    # other end of the pair (nearfield_leafpair_csr_direct_vjp): on a list that is
    # not symmetric it would be wrong, silently, so refuse one. The walk's lists
    # are symmetric by construction (one canonical walk emits (A, B) and (B, A)).
    _require(
        state.neighbor_list is None or _near_list_symmetric(state.neighbor_list),
        "differentiable large-N evaluation needs a SYMMETRIC near list (its "
        "reverse reads each pair from both ends); this state's leaf neighbour CSR "
        "is not.",
    )
    sources = jnp.asarray(compact.sources, dtype=INDEX_DTYPE)
    # the list as the walk built it: a target-sorted list keeps its row offsets
    # (the production refresh's M2L operands), anything else one target per pair
    target_sorted = isinstance(compact, TargetSortedFarPairs)
    targets = jnp.asarray(compact.targets, dtype=INDEX_DTYPE)
    num_nodes = int(jnp.asarray(state.tree.node_ranges).shape[0])
    far_symmetric = bool(target_sorted) and _far_list_symmetric(
        sources, targets, num_nodes
    )
    return LargeNGradPlan(
        far_pair_sources=sources,
        far_pair_targets=targets,
        far_pair_active_count=getattr(compact, "far_pair_count", None),
        order=int(state.local_order),
        max_leaf_size=int(state.max_leaf_size),
        center_mode=str(overrides.center_mode),
        m2l_chunk_size=overrides.m2l_chunk_size,
        l2l_chunk_size=overrides.l2l_chunk_size,
        num_far_pairs=int(sources.shape[0]),
        far_pairs_target_sorted=bool(target_sorted),
        far_pairs_symmetric=far_symmetric,
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

    The chain is the production refresh's (``_refresh_large_n_same_topology``):
    the same upward sweep (P2M and M2M through their Pallas ``custom_vjp`` seams;
    the box geometry deferred where the walk tests the COM MAC, as there), and the
    same downward sweep on the same operands -- a target-sorted walk list goes to
    the CSR M2L lanes as its row offsets (no target expanded per pair, no sort),
    and the L2L cascade follows. On the plan's frozen lists the locals are the
    production prepare's bitwise (measured on an A100 at 2e5, the G1 point, and
    under ``mac_type='dehnen_error'``). The force-scale outputs of the
    ``dehnen_error`` sweep (eq 16b's far ``f_b``, a topology input of the next
    walk) are not computed here: the locals do not depend on them.

    Parameters
    ----------
    fmm : Any
        Facade or runtime engine.
    state : LargeNPreparedState
        Prepared state supplying the frozen topology.
    positions_sorted : Array
        Live positions in the state's sorted order.
    masses_sorted : Array
        Live masses, same order.
    plan : LargeNGradPlan
        Frozen discriminators from :func:`prepare_large_n_grad_plan`.

    Returns
    -------
    Any
        Freshly computed local expansions -- NOT ``state.local_data``, which is
        prebaked and would cut the source-side chain.
    """
    from jaccpot.upward._mass_moments_vjp import gather_reverse_mass_moments

    engine = _engine(fmm)
    # the COMs' reverse by the tree's ancestor sums: gathers only (no f64[N + 1]
    # scatter, the same sums in every run), the forward yggdrax's
    with gather_reverse_mass_moments():
        upward = engine.prepare_upward_sweep(
            state.tree,
            positions_sorted,
            masses_sorted,
            max_order=plan.order,
            center_mode=plan.center_mode,
            max_leaf_size=plan.max_leaf_size,
            defer_geometry=engine._defers_box_geometry(
                str(engine.config.tree.mode), plan.center_mode
            ),
        )
    coo = _FarPairCOO
    if plan.far_pairs_target_sorted:
        coo = (
            _SymmetricTargetSortedFarPairCOO
            if plan.far_pairs_symmetric
            else _TargetSortedFarPairCOO
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
        dehnen_radius_scale=engine.dehnen_radius_scale,
        far_pairs_coo=coo(
            plan.far_pair_sources,
            plan.far_pair_targets,
            plan.far_pair_active_count,
        ),
        adaptive_order=True,
        p_gears=(plan.order,),
    )
    return downward.locals


def large_n_farfield_l2p_at(
    state: LargeNPreparedState,
    local_data: Any,
    positions_sorted: Array,
) -> Array:
    """The far-field gradient at live positions, evaluated as production evaluates it.

    The L2P of :func:`~jaccpot.runtime._large_n_pipeline.evaluate_large_n_state`'s
    fast lane, with its layout, chunk and kernel
    (:func:`~jaccpot.runtime._large_n_pipeline.resolve_large_n_l2p`) and its
    operands: on the default particle layout with the Pallas kernel, the
    particle-major L2P through its ``custom_vjp``
    (:func:`~jaccpot.pallas.l2p_real.l2p_real_particles_pallas_cvjp`), whose
    reverse reduces the coefficient and centre cotangents per leaf in the kernel
    -- no per-particle coefficient cotangent exists. With the locals of
    :func:`large_n_farfield_locals_at` at the state's own positions and masses this
    is the production eval's far half bitwise (A100, 2e5).

    The caller applies ``-G`` (production: ``far_acc = -G * far_grad``), so ``G``
    may be a traced parameter.

    Parameters
    ----------
    state : LargeNPreparedState
        Prepared state supplying the frozen topology (leaves, node ranges).
    local_data : Any
        Local expansions, e.g. :func:`large_n_farfield_locals_at`'s.
    positions_sorted : Array
        Live positions in the state's sorted order.

    Returns
    -------
    Array
        ``[N, 3]`` gradient of the far-field potential with respect to
        ``centre - position`` (the acceleration is ``-G`` times it), SORTED order.
    """
    from ._large_n_pipeline import resolve_large_n_l2p

    layout, chunk, kernel = resolve_large_n_l2p()
    leaf_nodes = jnp.asarray(state.neighbor_list.leaf_indices, dtype=INDEX_DTYPE)
    node_ranges = jnp.asarray(state.tree.node_ranges, dtype=INDEX_DTYPE)
    if layout == "particle":
        return _evaluate_local_expansions_particle_major(
            local_data,
            positions_sorted,
            leaf_nodes=leaf_nodes,
            node_ranges=node_ranges,
            order=int(state.local_order),
            chunk=chunk,
            kernel=kernel,
        )
    far_grad, _, _ = _evaluate_local_expansions_for_particles(
        local_data,
        positions_sorted,
        leaf_nodes=leaf_nodes,
        node_ranges=node_ranges,
        max_leaf_size=int(state.max_leaf_size),
        order=int(state.local_order),
        expansion_basis="solidfmm",
        return_potential=False,
        max_acc_derivative_order=0,
    )
    return far_grad


def resolve_grad_force_params(fmm: Any, force_params: Any) -> Any:
    """The force parameters a differentiable large-N evaluation runs with.

    As :meth:`~jaccpot.runtime.fused_provider.FusedForceProvider.resolve_params`
    does for the fused lane: ``None``, or the engine's own values as Python
    floats, is the engine's (``None``: its literals, the production program);
    other Python floats are literals; arrays are cast to the working dtype and
    may be traced -- ``jax.grad`` in ``G`` and ``eps2`` at the frozen topology.

    Parameters
    ----------
    fmm : Any
        Facade or runtime engine.
    force_params : Any
        A :class:`~jaccpot.runtime.fused_topology.ForceParams`, or ``None``.

    Returns
    -------
    Any
        ``None`` for the engine's own parameters, else a ``ForceParams``.

    Raises
    ------
    ValueError
        For anything but the engine's own under ``mac_type='dehnen_error'``: the
        frozen lists' eq (16a) thresholds were built with them (the fused lane's
        refusal, user decision S2).
    """
    from .fused_topology import ForceParams, _is_static

    if force_params is None:
        return None
    engine = _engine(fmm)
    own_g, own_eps2 = float(engine.G), float(engine.softening) ** 2
    if (
        force_params.is_static()
        and float(force_params.G) == own_g
        and float(force_params.eps2) == own_eps2
    ):
        return None
    if str(getattr(engine, "mac_type", "")) == "dehnen_error":
        raise ValueError(
            "mac_type='dehnen_error' takes the solver's own G and softening only "
            f"(G={own_g}, eps2={own_eps2}): the frozen lists' eq (16a) thresholds "
            "were built with them. Gradients in the positions and masses at those "
            "lists need no force_params; G and eps2 need another MAC."
        )
    dtype = engine.working_dtype

    def cast(value: Any) -> Any:
        return value if _is_static(value) else jnp.asarray(value, dtype)

    return ForceParams(G=cast(force_params.G), eps2=cast(force_params.eps2))


def evaluate_large_n_state_at_positions_and_masses_sorted(
    fmm: Any,
    state: LargeNPreparedState,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    plan: LargeNGradPlan,
    include_near: bool = True,
    include_far: bool = True,
    force_params: Any = None,
) -> Array:
    """Fixed-topology large-N re-evaluation at live inputs, in SORTED order.

    The production ``evaluate_large_n_state`` fast lane, expression for
    expression and kernel for kernel (plan S8): the near field is the CSR lane's
    direct layout through its ``custom_vjp``
    (:func:`~jaccpot.pallas.nearfield_leafpair_csr_direct_vjp.
    nearfield_leafpair_csr_sorted_direct_pallas_cvjp`, whose forward is the
    production call), the far field is :func:`large_n_farfield_l2p_at` on the locals
    :func:`large_n_farfield_locals_at` recomputes, and the two are summed as
    production sums them. At the state's own positions and masses this is the
    production evaluation bitwise; the one deliberate difference is that the L2P
    reads recomputed locals rather than the frozen ``state.local_data``, so
    cotangents reach the source-side chain.

    What production adds that this does not: the caller's external field
    (``strict_run_v2``'s ``external_acceleration_fn``, outside the self-gravity;
    differentiate it in the caller), and the near field's overflow / target-block
    additions of the pure-JAX rectangle route, which the CSR lane never populates
    and :func:`prepare_large_n_grad_plan` refuses.

    ``include_near`` / ``include_far`` exist for the diagnostic split that the
    correctness gate relies on: with the near field disabled, a mass gradient that
    comes out exactly zero proves the far-field chain is not wired.

    Parameters
    ----------
    fmm : Any
        Facade or runtime engine.
    state : LargeNPreparedState
        Prepared state supplying the frozen topology.
    positions_sorted : Array
        Live positions in the state's sorted order.
    masses_sorted : Array
        Live masses, same order.
    plan : LargeNGradPlan
        Frozen discriminators from :func:`prepare_large_n_grad_plan`.
    include_near : bool
        Include the near-field contribution.
    include_far : bool
        Include the far-field contribution.
    force_params : Any
        A :class:`~jaccpot.runtime.fused_topology.ForceParams` (``G`` and the
        softening squared) resolved by :func:`resolve_grad_force_params`; ``None``
        takes the engine's literals. Arrays may be traced: the gradient in them
        is the derivative of the force at the frozen lists (for a compact
        softening kernel, valid up to the softening the walk was built for,
        ``FusedForceProvider.walk_eps2``).

    Returns
    -------
    Array
        Accelerations in SORTED order -- the caller applies the inverse
        permutation. Returning sorted is deliberate: it keeps this function
        expression-for-expression comparable with the production fast lane.

    Raises
    ------
    ValueError
        If ``positions_sorted`` or ``masses_sorted`` does not match the state's
        shapes.
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
    # the engine's G and softening (literals), or the caller's -- as production
    g_far = float(getattr(engine, "G")) if force_params is None else force_params.G
    near_kw = (
        {}
        if force_params is None
        else {"G": force_params.G, "softening_sq": force_params.eps2}
    )
    near_acc = jnp.zeros_like(positions)
    far_acc = jnp.zeros_like(positions)
    if include_near:
        near_acc = evaluate_large_n_nearfield_fast_lane(
            engine,
            state,
            return_potential=False,
            positions_sorted=positions,
            masses_sorted=masses,
            differentiable=True,
            **near_kw,
        )
    if include_far:
        locals_live = large_n_farfield_locals_at(
            engine, state, positions, masses, plan=plan
        )
        far_acc = -g_far * large_n_farfield_l2p_at(state, locals_live, positions)
    return near_acc + far_acc
