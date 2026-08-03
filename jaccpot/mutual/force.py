"""Assembly of the mutual FMM force, with block-step level awareness.

This module turns the frozen :class:`~jaccpot.mutual.topology.MutualTopology`
plus the two mutual kernels into the force a block-power-of-two KDK leapfrog
consumes, and adds the rung vocabulary the integrator needs.

Levels
------
A pair of particles belongs to interaction level ``k = max(rung_i, rung_j)`` --
the level of its *finer* endpoint. Both kernels take a ``level_weights`` vector
and multiply every pair by ``level_weights[k]``, so a single traversal evaluates
*any* linear combination of levels:

* one-hot at ``k`` -> ``level_accelerations(level=k)``;
* the boundary's kick weights -> the whole sub-step boundary in one pass.

That second form is the point. The per-level interface costs one full tree
traversal per active level, which over a base step is ``sum_s (active levels at
s)`` traversals -- about 19 for ``k_max = 3``. Weighting instead of masking makes
it ``n_sub + 1`` (9 for ``k_max = 3``): the cost scales with *boundaries*, not
boundaries x levels, which is what preserves the individual-timestep advantage
once the force is an FMM rather than a masked direct sum.

Exact vs. cell-level splitting
------------------------------
The near field applies the **exact per-particle** predicate: every particle pair
in a leaf block is weighted by its own ``max(rung_i, rung_j)``. The far field
assigns each *cell* the rung of its most active particle and weights a cell pair
by ``max(rung_A, rung_B)`` -- the falcON activity-gating approximation (strategy
B2 of the design). It over-refines: a coarse particle sharing a cell with a fine
one is treated at the fine level.

Both are genuine *partitions* of the interaction set, so their per-level
contributions still sum to the total force and each level still conserves
momentum exactly. What the cell-level split does **not** reproduce is a
direct-sum oracle's per-level decomposition, so cross-checks against
``MutualDirectSumGravity`` are made on the total force, momentum and energy
rather than level by level. Keeping one multipole set per cell is what avoids the
``(k_max + 1)x`` multipole memory an exact per-rung-multipole scheme would need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from jaccpot.mutual.farfield import MutualTreeArrays, mutual_far_field_forces
from jaccpot.mutual.nearfield import mutual_near_field_forces
from jaccpot.mutual.topology import MutualTopology

__all__ = [
    "MutualFMMState",
    "MutualForceResult",
    "build_mutual_state",
    "mutual_accelerations",
    "mutual_level_accelerations",
    "mutual_weighted_accelerations",
    "boundary_level_weights",
    "boundary_weight_table",
    "level_weights_from_floor",
    "n_sub",
    "active_level_floor",
    "is_sync_boundary",
]


class MutualForceResult(NamedTuple):
    """Accelerations plus the near/far force split that produced them."""

    accelerations: Array
    near_forces: Array
    far_forces: Array


@dataclass(frozen=True)
class MutualFMMState:
    """Device-resident frozen state for repeated mutual FMM evaluations.

    Built once per topology refresh (per base step in a block-step run) and then
    reused across every boundary of that step. Holding it fixed is what makes
    ``jax.grad`` over an evaluation an exact fixed-topology gradient.
    """

    topology: MutualTopology
    tree: MutualTreeArrays
    leaf_particles: Array
    leaf_particle_valid: Array
    near_a: Array
    near_b: Array
    near_valid: Array
    self_leaves: Array
    far_a: Array
    far_b: Array
    forward_permutation: Array
    inverse_permutation: Array
    softening: float
    G: float
    order: int
    use_pallas: bool
    near_chunk_size: Optional[int] = None
    pallas_interpret: bool = False

    @property
    def num_particles(self: "MutualFMMState") -> int:
        """Number of particles this state was built for."""
        return int(self.topology.num_particles)


def build_mutual_state(
    topology: MutualTopology,
    *,
    softening: float,
    G: float = 1.0,
    use_pallas: bool = False,
    near_chunk_size: Optional[int] = None,
    pallas_interpret: bool = False,
    index_dtype: Any = jnp.int32,
) -> MutualFMMState:
    """Move a host-built topology onto the device as compile-time constants."""
    topo = topology
    node_to_leaf = jnp.zeros((topo.num_nodes,), dtype=index_dtype)
    node_to_leaf = node_to_leaf.at[jnp.asarray(topo.leaf_nodes)].set(
        jnp.arange(topo.num_leaves, dtype=index_dtype)
    )
    near_a = node_to_leaf[jnp.asarray(topo.near_a)]
    near_b = node_to_leaf[jnp.asarray(topo.near_b)]

    far_a = jnp.asarray(topo.far_a, dtype=index_dtype)
    far_b = jnp.asarray(topo.far_b, dtype=index_dtype)
    # Both directions of every canonical pair, in one batch, so a single kernel
    # invocation with one rounding regime produces both halves.
    far_source = jnp.concatenate([far_b, far_a])
    far_target = jnp.concatenate([far_a, far_b])
    far_valid = jnp.ones((int(far_source.shape[0]),), dtype=bool)

    tree = MutualTreeArrays(
        num_nodes=int(topo.num_nodes),
        order=int(topo.order),
        leaf_nodes=jnp.asarray(topo.leaf_nodes, dtype=index_dtype),
        leaf_particles=jnp.asarray(topo.leaf_particles, dtype=index_dtype),
        leaf_particle_valid=jnp.asarray(topo.leaf_particle_valid),
        level_nodes=tuple(
            jnp.asarray(level, dtype=index_dtype) for level in topo.level_nodes
        ),
        level_parents=tuple(
            jnp.asarray(level, dtype=index_dtype)
            for level in topo.parent_of_level_nodes
        ),
        far_source=far_source,
        far_target=far_target,
        far_valid=far_valid,
    )
    return MutualFMMState(
        topology=topo,
        tree=tree,
        leaf_particles=tree.leaf_particles,
        leaf_particle_valid=tree.leaf_particle_valid,
        near_a=near_a,
        near_b=near_b,
        near_valid=jnp.ones((int(near_a.shape[0]),), dtype=bool),
        self_leaves=jnp.arange(topo.num_leaves, dtype=index_dtype),
        far_a=far_a,
        far_b=far_b,
        forward_permutation=jnp.asarray(topo.forward_permutation, dtype=index_dtype),
        inverse_permutation=jnp.asarray(topo.inverse_permutation, dtype=index_dtype),
        softening=float(softening),
        G=float(G),
        order=int(topo.order),
        use_pallas=bool(use_pallas),
        near_chunk_size=near_chunk_size,
        pallas_interpret=bool(pallas_interpret),
    )


# ---------------------------------------------------------------------------
# Block-step schedule. Duplicated from nornax's ``blockstep.schedule`` on purpose:
# the adapter must not import nornax (the dependency graph is Jaccpot -> Yggdrax,
# Nornax standalone, ODISSEO -> both). These are pure integer identities, checked
# against nornax's own implementation in the cross-repo tests.
# ---------------------------------------------------------------------------


def n_sub(k_max: int) -> int:
    """Return the number of smallest sub-steps per base step, ``2**k_max``."""
    return 1 << int(k_max)


def is_sync_boundary(s: int, k_max: int) -> bool:
    """Return whether boundary ``s`` is synchronized across every rung."""
    return int(s) == 0 or int(s) == (1 << int(k_max))


def active_level_floor(s: int, k_max: int) -> int:
    """Return the smallest level kicked at boundary ``s``."""
    s, k_max = int(s), int(k_max)
    if is_sync_boundary(s, k_max):
        return 0
    return k_max - ((s & -s).bit_length() - 1)


def level_weights_from_floor(
    active_floor: int,
    k_max: int,
    dt_max: float,
    *,
    half: float = 1.0,
    dtype: Any = jnp.float64,
) -> Array:
    """Return kick weights for the levels at or above ``active_floor``.

    Level ``k`` is kicked with ``half * dt_max / 2**k``; levels below the floor
    are not active at this boundary and get zero.

    ``active_floor``, ``half`` and ``dt_max`` may be **tracers**. That is what lets
    a caller drive the boundaries with ``lax.scan`` instead of unrolling them: with
    static-only weights an integrator has to emit one traced boundary kick per
    boundary, so its graph grows like ``2**k_max``. The concrete path is kept
    separate and unchanged rather than folded into the traced one, so existing
    callers keep bit-identical weights (the traced form evaluates the product in
    ``dtype`` where the Python form evaluates it in double and then casts, which can
    differ by an ulp in float32).
    """
    k_max = int(k_max)
    if all(
        not isinstance(jnp.asarray(v), jax.core.Tracer)
        for v in (active_floor, half, dt_max)
    ):
        weights = [
            (
                (float(half) * float(dt_max) / float(1 << k))
                if k >= int(active_floor)
                else 0.0
            )
            for k in range(k_max + 1)
        ]
        return jnp.asarray(weights, dtype=dtype)

    # Traced form. `1 / 2**k` is exact in binary floating point, so scaling by it
    # is the same operation as dividing by `2**k` -- no accuracy is given up for
    # traceability.
    levels = jnp.arange(k_max + 1)
    inverse = jnp.asarray([1.0 / float(1 << k) for k in range(k_max + 1)], dtype=dtype)
    scale = jnp.asarray(half, dtype=dtype) * jnp.asarray(dt_max, dtype=dtype)
    return jnp.where(levels >= jnp.asarray(active_floor), scale * inverse, 0.0)


def boundary_weight_table(
    k_max: int, dt_max: Any, *, dtype: Any = jnp.float64
) -> Array:
    """Return the ``(n_sub + 1, k_max + 1)`` table of every boundary's kick weights.

    Row ``s`` is :func:`boundary_level_weights` for boundary ``s``. The point of
    materialising it is that an integrator can then index it with a **traced**
    boundary index -- ``table[s]`` inside a ``lax.scan`` -- and hand the row
    straight to ``boundary_kick(..., level_weights=...)``. That is what lifts a
    fused base step from ``2**k_max`` unrolled boundary kicks in the traced graph
    to a single one, without giving up the runtime win of one traversal per
    boundary.

    The schedule is data-independent, so the table is a compile-time constant; it
    is ``(2**k_max + 1) x (k_max + 1)`` floats, i.e. 72 values at ``k_max = 3``.
    """
    return jnp.stack(
        [
            boundary_level_weights(s, int(k_max), dt_max, dtype=dtype)
            for s in range(n_sub(int(k_max)) + 1)
        ]
    )


def boundary_level_weights(
    s: int, k_max: int, dt_max: float, *, dtype: Any = jnp.float64
) -> Array:
    """Return the per-level kick weights applied at sub-step boundary ``s``.

    ``half`` is ``0.5`` at the synchronized ends of the base step and ``1.0``
    inside. Feeding this straight into the kernels' ``level_weights`` is the
    fused-boundary primitive: one traversal per boundary instead of one per
    active level.
    """
    return level_weights_from_floor(
        active_level_floor(s, k_max),
        k_max,
        dt_max,
        half=0.5 if is_sync_boundary(s, k_max) else 1.0,
        dtype=dtype,
    )


def _cell_rungs(state: MutualFMMState, rung_sorted: Array) -> Array:
    """Assign each node the rung of its most active (finest) particle.

    Leaves reduce their padded block with a ``-1`` fill so empty slots cannot win
    the maximum; internal nodes then take the running maximum up the same level
    schedule the multipole sweep uses.
    """
    tree = state.tree
    valid = tree.leaf_particle_valid
    per_slot = jnp.where(valid, rung_sorted[tree.leaf_particles], -1)
    node_rung = jnp.full((tree.num_nodes,), -1, dtype=per_slot.dtype)
    node_rung = node_rung.at[tree.leaf_nodes].max(jnp.max(per_slot, axis=1))
    for nodes, parents in zip(reversed(tree.level_nodes), reversed(tree.level_parents)):
        node_rung = node_rung.at[parents].max(node_rung[nodes])
    return node_rung


def _far_pair_weights(
    state: MutualFMMState,
    rung_sorted: Optional[Array],
    level_weights: Optional[Array],
) -> Optional[Array]:
    """Cell-level weights for each directed far entry (both directions equal)."""
    if rung_sorted is None or level_weights is None:
        return None
    node_rung = _cell_rungs(state, rung_sorted)
    pair_level = jnp.maximum(node_rung[state.far_a], node_rung[state.far_b])
    pair_level = jnp.clip(pair_level, 0, int(level_weights.shape[0]) - 1)
    weights = jnp.take(level_weights, pair_level, axis=0, mode="clip")
    # The canonical pair list was doubled into (b->a, a->b); the same weight must
    # ride both halves or the +F/-F cancellation breaks.
    return jnp.concatenate([weights, weights])


def mutual_weighted_accelerations(
    state: MutualFMMState,
    positions: Array,
    masses: Array,
    *,
    rung: Optional[Array] = None,
    level_weights: Optional[Array] = None,
    return_parts: bool = False,
) -> Array | MutualForceResult:
    """Evaluate ``sum_k level_weights[k] * a_k`` in a single mutual traversal.

    With ``level_weights=None`` this is simply the full acceleration. Inputs and
    outputs are in the caller's original particle order; the Morton permutation
    is applied internally and is a frozen integer constant, so cotangents flow to
    positions and masses but never to the index arrays.
    """
    positions = jnp.asarray(positions)
    masses = jnp.asarray(masses)
    fwd = state.forward_permutation
    pos_sorted = positions[fwd]
    mass_sorted = masses[fwd]
    rung_sorted = None if rung is None else jnp.asarray(rung)[fwd]
    if level_weights is not None:
        level_weights = jnp.asarray(level_weights, dtype=pos_sorted.dtype)

    near = mutual_near_field_forces(
        pos_sorted,
        mass_sorted,
        leaf_particles=state.leaf_particles,
        leaf_particle_valid=state.leaf_particle_valid,
        near_a=state.near_a,
        near_b=state.near_b,
        near_valid=state.near_valid,
        self_leaves=state.self_leaves,
        softening=state.softening,
        G=state.G,
        rung=rung_sorted,
        level_weights=level_weights,
        chunk_size=state.near_chunk_size,
        use_pallas=state.use_pallas,
        interpret=state.pallas_interpret,
    )
    far = mutual_far_field_forces(
        pos_sorted,
        mass_sorted,
        state.tree,
        G=state.G,
        far_weights=_far_pair_weights(state, rung_sorted, level_weights),
        use_pallas=state.use_pallas,
        interpret=state.pallas_interpret,
    )
    # Divide by mass once, at the very end: the accumulated quantity that cancels
    # structurally is the force, not the acceleration.
    total = near + far
    inv = state.inverse_permutation
    safe_mass = jnp.where(mass_sorted > 0, mass_sorted, jnp.ones_like(mass_sorted))
    acc = jnp.where(mass_sorted[:, None] > 0, total / safe_mass[:, None], 0.0)
    if return_parts:
        return MutualForceResult(
            accelerations=acc[inv], near_forces=near[inv], far_forces=far[inv]
        )
    return acc[inv]


def mutual_accelerations(
    state: MutualFMMState, positions: Array, masses: Array, **kwargs: Any
) -> Array | MutualForceResult:
    """Return the full mutual FMM acceleration (all levels, unit weights)."""
    return mutual_weighted_accelerations(state, positions, masses, **kwargs)


def mutual_level_accelerations(
    state: MutualFMMState,
    positions: Array,
    masses: Array,
    *,
    rung: Array,
    level: int,
    k_max: int,
) -> Array:
    """Return the level-``k`` antisymmetric acceleration for every particle.

    Implements the ``MutualForceModel`` contract structurally: only interactions
    assigned to ``level`` contribute, they are applied antisymmetrically, and
    summing over ``k = 0 .. k_max`` reproduces the full acceleration.
    """
    weights = jnp.zeros((int(k_max) + 1,), dtype=jnp.asarray(positions).dtype)
    weights = weights.at[int(level)].set(1.0)
    return mutual_weighted_accelerations(
        state, positions, masses, rung=rung, level_weights=weights
    )
