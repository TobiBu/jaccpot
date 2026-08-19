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

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from jaccpot.mutual.farfield import (
    MutualTreeArrays,
    _scan_levels,
    dense_level_schedule,
    mutual_far_field_forces,
    snap_capacity,
)
from jaccpot.mutual.nearfield import mutual_near_field_forces
from jaccpot.mutual.topology import MutualTopology

__all__ = [
    "MutualCapacities",
    "MutualCapacityOverflow",
    "MutualFMMState",
    "MutualForceResult",
    "build_mutual_state",
    "resolve_mutual_capacities",
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


class MutualCapacityOverflow(RuntimeError):
    """A topology did not fit the capacities its state was built for.

    Raised rather than truncating. A truncated interaction list still conserves
    momentum -- dropping a pair drops both its halves -- so the error would not
    show up in the diagnostic this lane is judged on, only in a silently wrong
    force.
    """


class MutualCapacities(NamedTuple):
    """Fixed capacities that make a prepared state's shapes distribution-free.

    Note the consequence for anything that read ``state.far_a.shape[0]`` as a
    *count*: once padded, that is the **capacity**. A configuration with no far
    pairs at all still reports a nonzero capacity, so a vacuity check written
    against the shape silently stops working. Use ``state.num_far_pairs`` /
    ``state.num_near_pairs``.

    ``near`` and ``far`` bound the canonical pair lists; ``depth`` and ``width``
    bound the level schedule; ``queue`` bounds the device traversal's wavefront.
    Everything else about a mutual state is already fixed by
    ``(N, leaf_size, order)``: ``num_leaves == ceil(N / leaf_size)`` and
    ``max_leaf_size == leaf_size``, because the tree builder slices the Morton
    order into fixed-width buckets.

    ``queue`` cannot be derived from a *finished* host topology -- it bounds the
    widest intermediate pair front, which the host traversal does not record --
    so it is resolved by trial in
    :meth:`jaccpot.BlockStepFMM.freeze_template` and defaults to 0, meaning "not
    resolved yet, use the builder's own default". Getting it wrong is severe and
    quiet: the device walk's loop *terminates* on queue overflow, so the pair
    lists come back drastically short. Measured at N = 100 000, leaf 32: a 65 536
    front yielded 11 022 far pairs against the correct 341 504, a **90% force
    error** -- with momentum still conserved to 7e-22, because dropping a
    canonical pair drops both of its halves.
    """

    near: int
    far: int
    depth: int
    width: int
    queue: int = 0


def resolve_mutual_capacities(
    topology: MutualTopology,
    *,
    relative: float = 0.10,
    absolute: int = 256,
) -> MutualCapacities:
    """Size capacities from one built topology, with drift headroom.

    Every value is snapped onto :data:`~jaccpot.mutual.farfield.CAPACITY_LADDER`
    with the additive-plus-relative headroom
    :func:`~jaccpot.mutual.farfield.snap_capacity` documents, so nearby
    topologies resolve to the *same* capacity and share one compiled program.

    Depth and width use a small absolute margin instead: they are counted in
    levels and nodes-per-level, where measured drift is a couple of units
    (depth 22 -> 24, widest level 20 -> 24), and where over-provisioning a level
    row costs real work in the cascades.
    """
    widths = [int(np.asarray(level).shape[0]) for level in topology.level_nodes] or [0]
    num_leaves = int(np.asarray(topology.leaf_nodes).shape[0])
    return MutualCapacities(
        near=snap_capacity(
            int(np.asarray(topology.near_a).shape[0]),
            relative=relative,
            absolute=absolute,
        ),
        far=snap_capacity(
            int(np.asarray(topology.far_a).shape[0]),
            relative=relative,
            absolute=absolute,
        ),
        # Depth and width need *generous* headroom, not the small absolute
        # margin the pair lists get. Measured drift on a two-clump N = 4096
        # rollout was a couple of levels (22 -> 24), which is what the earlier
        # +25%/+4 policy was sized for -- and it is not representative. On a
        # Hernquist cusp the LBVH tree depth went 16 -> 30 mid-rollout, nearly
        # doubling, because tree depth follows the Morton-code distribution and a
        # central cusp concentrates it hard. Over-provisioning a level row costs
        # one all-invalid (no-op) iteration of the cascade scan; under-
        # provisioning raises and ends the run.
        # Neither is free: `depth` is the number of cascade-scan iterations and
        # `width` is the work inside each, so over-provisioning depth is not a
        # no-op the way an invalid *slot* is. They get different policies because
        # they drift differently. Depth is the volatile one -- an LBVH tree over a
        # Hernquist cusp went 16 -> 30 mid-rollout, nearly doubling, because tree
        # depth follows the Morton-code distribution and a central cusp
        # concentrates it hard -- and the failure mode is a hard raise that ends
        # the run, with no option to re-resolve (that would change the compiled
        # shape). Width drifted only ~20% in the same measurements.
        depth=snap_capacity(
            max(
                len(topology.level_nodes),
                # A *bound*, not a measurement. A measured depth cap with any
                # fixed headroom is fragile for LBVH: over a Hernquist cusp the
                # depth was seen to go 16 -> 30 at N = 2e4 and 12 -> 34 at
                # N = 1e5, because tree depth follows the Morton-code
                # distribution and a central cusp concentrates it hard. Four
                # times the balanced depth of the same leaf count covers those
                # and is still O(log N).
                4 * max(1, int(np.ceil(np.log2(max(2, int(num_leaves)))))),
            ),
            relative=1.0,
            absolute=8,
        ),
        width=snap_capacity(max(widths), relative=0.5, absolute=32),
        # Left unresolved: see MutualCapacities.queue. A host topology does not
        # record the widest pair front it passed through.
        queue=0,
    )
# A pytree child needs a dtype, so the default must be an array -- but a
# dataclass rejects an array default as "mutable", hence the factory.
def _no_overflow() -> Array:
    return jnp.asarray(False)


def _no_cause() -> Array:
    return jnp.asarray(0, dtype=jnp.int32)


#: Bit positions in ``MutualFMMState.overflow_causes``, in the order a report
#: should read them.
OVERFLOW_CAUSES = ("far", "near", "pair_queue", "level_width", "tree_depth")



@dataclass(frozen=True)
class MutualFMMState:
    """Device-resident frozen state for repeated mutual FMM evaluations.

    Built once per topology refresh (per base step in a block-step run) and then
    reused across every boundary of that step. Holding it fixed is what makes
    ``jax.grad`` over an evaluation an exact fixed-topology gradient.

    Registered as a **pytree**, so it can be a traced ``jax.jit`` argument rather
    than a closed-over constant. That is the difference between one compiled
    program per *rebuild* and one per *capacity profile*; combined with the
    padding in :func:`build_mutual_state` it is one program for a whole run. The
    array fields are the children and the scalars are aux data, which is why aux
    has to stay hashable -- the host-side ``MutualTopology`` used to live here and
    was removed for exactly that reason (it holds NumPy arrays, so it is
    unhashable, and nothing but ``num_particles`` ever read it).
    """

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
    num_particles_: int = 0
    # Occupancy counters. These are 0-d *arrays*, i.e. pytree children, not aux
    # data -- deliberately. Aux data is part of the treedef and therefore part of
    # the jit cache key, so a counter that changes every rebuild would re-key the
    # cache and undo the whole point of the padding. As children they are simply
    # traced (and unused) inside the compiled force, and readable on the host.
    num_near_pairs: Array = 0
    num_far_pairs: Array = 0
    topology_overflow: Array = field(default_factory=_no_overflow)
    overflow_causes: Array = field(default_factory=_no_cause)
    """True when a capacity was exceeded while building this state on device.

    A 0-d device bool, not an exception: the device builder runs under trace,
    where raising is not available. It must be *read* -- overflow silently drops
    interactions, and a mutual list that has lost a canonical pair has lost both
    its halves, so it still conserves momentum exactly and the diagnostic this
    lane is usually judged on will not notice. The host builder leaves it False.
    """

    @property
    def num_particles(self: "MutualFMMState") -> int:
        """Number of particles this state was built for."""
        return int(self.num_particles_)

    @property
    def near_capacity(self: "MutualFMMState") -> int:
        """Allocated slots for canonical near pairs -- ``>= num_near_pairs``."""
        return int(self.near_a.shape[0])

    @property
    def far_capacity(self: "MutualFMMState") -> int:
        """Allocated slots for canonical far pairs -- ``>= num_far_pairs``."""
        return int(self.far_a.shape[0])


def _pad_pair_list(values, cap: int, *, index_dtype) -> Array:
    """Pad a 1-D index list up to ``cap`` with zeros (masked by its valid array)."""
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.shape[0] > cap:
        raise ValueError(f"pair list of {arr.shape[0]} exceeds capacity {cap}")
    out = np.zeros((cap,), dtype=np.int64)
    out[: arr.shape[0]] = arr
    return jnp.asarray(out, dtype=index_dtype)


def build_mutual_state(
    topology: MutualTopology,
    *,
    softening: float,
    G: float = 1.0,
    use_pallas: bool = False,
    near_chunk_size: Optional[int] = None,
    pallas_interpret: bool = False,
    index_dtype: Any = jnp.int32,
    caps: Optional["MutualCapacities"] = None,
) -> MutualFMMState:
    """Move a host-built topology onto the device.

    With ``caps=None`` every array is sized exactly to the topology, which is the
    historical behaviour and the right choice for a one-shot evaluation. Pass
    ``caps`` (see :func:`resolve_mutual_capacities`) to pad the pair lists and the
    level schedule to fixed capacities instead: the arrays then depend on the
    *capacities* rather than on the particle distribution, so one compiled
    program serves every rebuild. That is the difference between paying jaccpot's
    ~200 s mutual-force compile once and paying it per base step.
    """
    topo = topology
    node_to_leaf = jnp.zeros((topo.num_nodes,), dtype=index_dtype)
    node_to_leaf = node_to_leaf.at[jnp.asarray(topo.leaf_nodes)].set(
        jnp.arange(topo.num_leaves, dtype=index_dtype)
    )

    n_near = int(np.asarray(topo.near_a).shape[0])
    n_far = int(np.asarray(topo.far_a).shape[0])
    near_cap = n_near if caps is None else int(caps.near)
    far_cap = n_far if caps is None else int(caps.far)
    if n_near > near_cap or n_far > far_cap:
        raise MutualCapacityOverflow(
            f"topology overflows its capacities: near {n_near}/{near_cap}, "
            f"far {n_far}/{far_cap}. Re-resolve the capacities with more headroom; "
            "silently truncating the interaction list would give a wrong force."
        )

    near_a = node_to_leaf[_pad_pair_list(topo.near_a, near_cap, index_dtype=index_dtype)]
    near_b = node_to_leaf[_pad_pair_list(topo.near_b, near_cap, index_dtype=index_dtype)]
    near_valid = jnp.arange(near_cap) < n_near

    far_a = _pad_pair_list(topo.far_a, far_cap, index_dtype=index_dtype)
    far_b = _pad_pair_list(topo.far_b, far_cap, index_dtype=index_dtype)
    # Both directions of every canonical pair, in one batch, so a single kernel
    # invocation with one rounding regime produces both halves.
    far_source = jnp.concatenate([far_b, far_a])
    far_target = jnp.concatenate([far_a, far_b])
    one_dir_valid = jnp.arange(far_cap) < n_far
    far_valid = jnp.concatenate([one_dir_valid, one_dir_valid])

    level_nodes, level_parents, level_valid, _, _ = dense_level_schedule(
        topo.level_nodes,
        topo.parent_of_level_nodes,
        depth_cap=None if caps is None else int(caps.depth),
        width_cap=None if caps is None else int(caps.width),
        index_dtype=index_dtype,
    )

    tree = MutualTreeArrays(
        num_nodes=int(topo.num_nodes),
        order=int(topo.order),
        leaf_nodes=jnp.asarray(topo.leaf_nodes, dtype=index_dtype),
        leaf_particles=jnp.asarray(topo.leaf_particles, dtype=index_dtype),
        leaf_particle_valid=jnp.asarray(topo.leaf_particle_valid),
        level_nodes=level_nodes,
        level_parents=level_parents,
        level_valid=level_valid,
        far_source=far_source,
        far_target=far_target,
        far_valid=far_valid,
    )
    return MutualFMMState(
        num_particles_=int(topo.num_particles),
        num_near_pairs=jnp.asarray(n_near, dtype=jnp.int32),
        num_far_pairs=jnp.asarray(n_far, dtype=jnp.int32),
        # A pytree child must be an array. The host traversal never overflows --
        # it allocates to the exact counts -- but the leaf still has to have a
        # dtype, or every consumer that walks the state's leaves trips over it.
        topology_overflow=jnp.asarray(False),
        overflow_causes=jnp.asarray(0, dtype=jnp.int32),
        tree=tree,
        leaf_particles=tree.leaf_particles,
        leaf_particle_valid=tree.leaf_particle_valid,
        near_a=near_a,
        near_b=near_b,
        near_valid=near_valid,
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
# pytree registration
# ---------------------------------------------------------------------------
#
# Both types are registered so a prepared state can be a *traced* jit argument.
# The split is the same in each: array leaves become children, scalars become
# aux data. Aux data must be hashable and comparable -- jax uses it in the
# treedef, which is part of the jit cache key -- so nothing holding a NumPy array
# may live there.
#
# `MutualTreeArrays` needs an explicit registration even though it is a
# NamedTuple (and so already a pytree): as a NamedTuple its `num_nodes` and
# `order` ints are *children*, which means they would arrive traced and break
# every `int(tree.order)` the operators do. Registering it moves them to aux.

_STATE_CHILDREN = (
    "tree",
    "leaf_particles",
    "leaf_particle_valid",
    "near_a",
    "near_b",
    "near_valid",
    "self_leaves",
    "far_a",
    "far_b",
    "forward_permutation",
    "inverse_permutation",
    "num_near_pairs",
    "num_far_pairs",
    "topology_overflow",
    "overflow_causes",
)
_STATE_AUX = (
    "softening",
    "G",
    "order",
    "use_pallas",
    "near_chunk_size",
    "pallas_interpret",
    "num_particles_",
)
_TREE_CHILDREN = (
    "leaf_nodes",
    "leaf_particles",
    "leaf_particle_valid",
    "level_nodes",
    "level_parents",
    "level_valid",
    "far_source",
    "far_target",
    "far_valid",
)
_TREE_AUX = ("num_nodes", "order")


def _flatten_tree(tree: MutualTreeArrays):
    return (
        tuple(getattr(tree, name) for name in _TREE_CHILDREN),
        tuple(int(getattr(tree, name)) for name in _TREE_AUX),
    )


def _unflatten_tree(aux, children) -> MutualTreeArrays:
    return MutualTreeArrays(
        **dict(zip(_TREE_AUX, aux)), **dict(zip(_TREE_CHILDREN, children))
    )


def _flatten_state(state: MutualFMMState):
    return (
        tuple(getattr(state, name) for name in _STATE_CHILDREN),
        tuple(getattr(state, name) for name in _STATE_AUX),
    )


def _unflatten_state(aux, children) -> MutualFMMState:
    return MutualFMMState(
        **dict(zip(_STATE_CHILDREN, children)), **dict(zip(_STATE_AUX, aux))
    )


jax.tree_util.register_pytree_node(MutualTreeArrays, _flatten_tree, _unflatten_tree)
jax.tree_util.register_pytree_node(MutualFMMState, _flatten_state, _unflatten_state)




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

    def _propagate(acc, nodes, parents, valid):
        child = jnp.where(valid, nodes, 0)
        parent = jnp.where(valid, parents, 0)
        # -1 is the identity for this max: every real rung is >= 0 and `acc`
        # starts at -1, so a padded slot's `.max(-1)` on node 0 cannot change it.
        return acc.at[parent].max(jnp.where(valid, acc[child], -1))

    return _scan_levels(_propagate, node_rung, tree, deepest_first=True)


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
