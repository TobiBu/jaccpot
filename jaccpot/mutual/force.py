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
    """Accelerations plus the near/far force split that produced them.

    Attributes
    ----------
    accelerations : Array
        ``(n, 3)`` accelerations in the caller's original particle order.
    near_forces : Array
        ``(n, 3)`` direct P2P contribution to the force, same order.
    far_forces : Array
        ``(n, 3)`` multipole contribution to the force, same order.
    """

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

    Attributes
    ----------
    near : int
        Capacity of the canonical near (leaf-pair) list.
    far : int
        Capacity of the canonical far (node-pair) list.
    depth : int
        Rows in the dense level schedule, i.e. the deepest tree level covered.
        Also the cascade scan's iteration count, so over-provisioning costs work.
    width : int
        Slots per level row, i.e. the widest level covered.
    queue : int
        Wavefront capacity for the device traversal.
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
    drift_headroom: bool = False,
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

    Parameters
    ----------
    topology : MutualTopology
        One built topology to size the capacities from.
    relative : float
        Fractional headroom on the pair lists. Default ``0.10``.
    absolute : int
        Additive headroom on the pair lists. Default ``256``.
    drift_headroom : bool
        Size ``depth``/``width`` for a *rollout* that rebuilds the tree, rather
        than for this one topology. Off by default: it costs 3.6-5.3x the level
        schedule at test scale, and the cascade walks that schedule on every
        evaluation, so only a caller that will actually rebuild should pay it. A
        static-radix template does not need it either -- its linkage is frozen,
        so its depth is invariant across rebuilds.

    Returns
    -------
    MutualCapacities
        Capacities snapped onto :data:`~jaccpot.mutual.farfield.CAPACITY_LADDER`,
        with ``queue`` left at 0 for the caller to resolve by trial.
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
                # Only with `drift_headroom`. A measured depth cap with any fixed
                # margin is fragile for LBVH over a *rollout*: on a Hernquist cusp
                # the depth was seen to go 16 -> 30 at N = 2e4 and 12 -> 34 at
                # N = 1e5, because tree depth follows the Morton-code distribution
                # and a central cusp concentrates it hard. Four times the balanced
                # depth of the leaf count covers those and is still O(log N).
                #
                # It is NOT the default, because `depth` is the cascade scan's
                # iteration count and `width` the work inside each, so this is
                # paid on every evaluation by every caller -- including one-shot
                # builds that never rebuild and so cannot drift. Measured
                # inflation of the level schedule at test scale: 4.0x at N = 512,
                # 5.3x at 2048, 3.6x at 4096. That is what OOM-killed a 16 GB CI
                # runner.
                (
                    4 * max(1, int(np.ceil(np.log2(max(2, int(num_leaves))))))
                    if drift_headroom
                    else 0
                ),
            ),
            relative=1.0 if drift_headroom else 0.25,
            absolute=8 if drift_headroom else 4,
        ),
        width=snap_capacity(
            max(widths),
            relative=0.5 if drift_headroom else 0.25,
            absolute=32 if drift_headroom else 16,
        ),
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

    Every index array below is a frozen integer constant: cotangents flow to
    positions and masses, never to the topology.

    Attributes
    ----------
    tree : MutualTreeArrays
        Device-resident tree the far-field sweep walks.
    leaf_particles : Array
        ``(num_leaves, leaf_capacity)`` particle indices, Morton-sorted order.
    leaf_particle_valid : Array
        Boolean mask of the same shape marking occupied slots in the padding.
    near_a : Array
        Leaf index of the first endpoint of each canonical near pair.
    near_b : Array
        Leaf index of the second endpoint of each canonical near pair.
    near_valid : Array
        Boolean mask over the near pair list, which is padded to its capacity.
    self_leaves : Array
        Leaf indices whose intra-leaf self-interaction is evaluated.
    far_a : Array
        Node index of the first endpoint of each canonical far pair.
    far_b : Array
        Node index of the second endpoint of each canonical far pair.
    forward_permutation : Array
        Maps original particle order to Morton-sorted order.
    inverse_permutation : Array
        Maps Morton-sorted order back to the caller's original order.
    softening : float
        Plummer softening length used by the near-field kernel.
    G : float
        Gravitational constant.
    order : int
        Expansion order.
    use_pallas : bool
        Whether to dispatch the fused Pallas kernels instead of pure JAX.
    near_chunk_size : Optional[int]
        Pair-chunk size for the near kernel; ``None`` leaves it unchunked.
    pallas_interpret : bool
        Run the Pallas kernels in interpret mode (CPU debugging).
    num_particles_ : int
        Particle count, as aux data. Trailing underscore because
        ``num_particles`` is the property that reads it.
    num_near_pairs : Array
        Live entries in the near pair list. Read this, **not**
        ``near_a.shape[0]``, which is the capacity once padded.
    num_far_pairs : Array
        Live entries in the canonical far pair list, same caveat.
    topology_overflow : Array
        True when a capacity was exceeded while building this state on device.
    overflow_causes : Array
        Bitmask over :data:`OVERFLOW_CAUSES` naming which capacities overflowed.
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

    @property
    def num_particles(self: "MutualFMMState") -> int:
        """Number of particles this state was built for.

        Returns
        -------
        int
            Particle count, read from aux data rather than from a host-side
            topology object -- that object was removed from the state so aux
            could stay hashable.
        """
        return int(self.num_particles_)

    @property
    def near_capacity(self: "MutualFMMState") -> int:
        """Allocated slots for canonical near pairs.

        Returns
        -------
        int
            The capacity, which is ``>= num_near_pairs``. This is what
            ``near_a.shape[0]`` reports once the list is padded, which is why the
            occupancy has its own field.
        """
        return int(self.near_a.shape[0])

    @property
    def far_capacity(self: "MutualFMMState") -> int:
        """Allocated slots for canonical far pairs.

        Returns
        -------
        int
            The capacity, which is ``>= num_far_pairs``; same caveat as
            :attr:`near_capacity`.
        """
        return int(self.far_a.shape[0])


def _pad_pair_list(values: Any, cap: int, *, index_dtype: Any) -> Array:
    """Pad a 1-D index list up to ``cap`` with zeros.

    Parameters
    ----------
    values : Any
        The live entries, in order.
    cap : int
        Target capacity.
    index_dtype : Any
        Integer dtype of the returned array.

    Returns
    -------
    Array
        A ``(cap,)`` array whose leading entries are ``values``. The padding is
        index 0, not -1, because every consumer gathers by it before masking; the
        companion ``*_valid`` array is what removes the contribution.

    Raises
    ------
    ValueError
        If ``values`` is longer than ``cap``.
    """
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
    """Move a host-built topology onto the device as compile-time constants.

    The canonical far pair list is doubled here, into ``(b -> a, a -> b)``, so a
    single kernel invocation with one rounding regime produces both halves of
    every pair. That is what makes the ``+F / -F`` cancellation exact rather
    than approximate.

    With ``caps=None`` every array is sized exactly to the topology, which is the
    historical behaviour and the right choice for a one-shot evaluation. Pass
    ``caps`` to pad the pair lists and the level schedule to fixed capacities
    instead: the arrays then depend on the *capacities* rather than on the
    particle distribution, so one compiled program serves every rebuild. That is
    the difference between paying jaccpot's ~200 s mutual-force compile once and
    paying it per base step.

    Parameters
    ----------
    topology : MutualTopology
        Frozen host-side topology: leaves, pair lists and the Morton permutation.
    softening : float
        Plummer softening length handed to the near-field kernel.
    G : float
        Gravitational constant. Default ``1.0``.
    use_pallas : bool
        Dispatch the fused Pallas kernels instead of the pure-JAX lanes.
    near_chunk_size : Optional[int]
        Pair-chunk size for the near kernel; ``None`` leaves it unchunked.
    pallas_interpret : bool
        Run the Pallas kernels in interpret mode, for CPU debugging.
    index_dtype : Any
        Integer dtype for every index array. Default ``jnp.int32``.
    caps : Optional[MutualCapacities]
        Fixed capacities to pad to; see :func:`resolve_mutual_capacities`.
        ``None`` sizes every array exactly to this topology.

    Returns
    -------
    MutualFMMState
        Device-resident state ready for repeated evaluation.

    Raises
    ------
    MutualCapacityOverflow
        If ``caps`` is given and this topology does not fit it. Raised rather than
        truncated: a dropped canonical pair loses both of its halves, so momentum
        stays exactly conserved and nothing else would reveal the loss.
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

    near_a = node_to_leaf[
        _pad_pair_list(topo.near_a, near_cap, index_dtype=index_dtype)
    ]
    near_b = node_to_leaf[
        _pad_pair_list(topo.near_b, near_cap, index_dtype=index_dtype)
    ]
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
    """Return the number of smallest sub-steps per base step, ``2**k_max``.

    Parameters
    ----------
    k_max : int
        Finest rung in the hierarchy.

    Returns
    -------
    int
        Number of sub-steps, ``2**k_max``.
    """
    return 1 << int(k_max)


def is_sync_boundary(s: int, k_max: int) -> bool:
    """Return whether boundary ``s`` is synchronized across every rung.

    The two ends of a base step, ``s = 0`` and ``s = 2**k_max``, are the only
    boundaries where every level is kicked together.

    Parameters
    ----------
    s : int
        Sub-step boundary index, ``0 .. 2**k_max``.
    k_max : int
        Finest rung in the hierarchy.

    Returns
    -------
    bool
        ``True`` at a synchronized end of the base step.
    """
    return int(s) == 0 or int(s) == (1 << int(k_max))


def active_level_floor(s: int, k_max: int) -> int:
    """Return the smallest level kicked at boundary ``s``.

    Levels at or above the floor are active; coarser ones are not due a kick
    yet. The floor follows the trailing-zero count of ``s``, which is the usual
    block-step identity: the more factors of two divide the boundary index, the
    coarser the levels that reach it.

    Parameters
    ----------
    s : int
        Sub-step boundary index, ``0 .. 2**k_max``.
    k_max : int
        Finest rung in the hierarchy.

    Returns
    -------
    int
        Smallest active level, ``0`` at a synchronized boundary.
    """
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

    Parameters
    ----------
    active_floor : int
        Smallest level to kick; may be a tracer.
    k_max : int
        Finest rung in the hierarchy. Always static -- it sets the output shape.
    dt_max : float
        Base-step size, the time step of level ``0``. May be a tracer.
    half : float
        Overall scale, ``0.5`` at a synchronized boundary and ``1.0`` inside.
        May be a tracer. Default ``1.0``.
    dtype : Any
        Output dtype. Default ``jnp.float64``.

    Returns
    -------
    Array
        ``(k_max + 1,)`` kick weights, zero below ``active_floor``.
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

    Parameters
    ----------
    k_max : int
        Finest rung in the hierarchy. Static -- it sets the table shape.
    dt_max : Any
        Base-step size, the time step of level ``0``. May be a tracer.
    dtype : Any
        Output dtype. Default ``jnp.float64``.

    Returns
    -------
    Array
        ``(2**k_max + 1, k_max + 1)`` table, row ``s`` being boundary ``s``.
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

    Parameters
    ----------
    s : int
        Sub-step boundary index, ``0 .. 2**k_max``.
    k_max : int
        Finest rung in the hierarchy.
    dt_max : float
        Base-step size, the time step of level ``0``.
    dtype : Any
        Output dtype. Default ``jnp.float64``.

    Returns
    -------
    Array
        ``(k_max + 1,)`` kick weights, zero on the levels this boundary skips.
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

    Parameters
    ----------
    state : MutualFMMState
        Frozen state supplying the tree and its leaf blocks.
    rung_sorted : Array
        ``(n,)`` per-particle rung, already in Morton-sorted order.

    Returns
    -------
    Array
        ``(num_nodes,)`` node rung; ``-1`` for nodes holding no particles.
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
    """Cell-level weights for each directed far entry (both directions equal).

    A far pair sits at level ``max(rung_a, rung_b)``, the level of its finer
    endpoint, exactly as a particle pair does.

    Parameters
    ----------
    state : MutualFMMState
        Frozen state supplying the canonical far pair list.
    rung_sorted : Optional[Array]
        ``(n,)`` per-particle rung in Morton-sorted order, or ``None``.
    level_weights : Optional[Array]
        ``(k_max + 1,)`` kick weights, or ``None``.

    Returns
    -------
    Optional[Array]
        Weight per directed far entry, or ``None`` when either input is
        ``None`` -- the unweighted full-force case.
    """
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

    Parameters
    ----------
    state : MutualFMMState
        Frozen state from :func:`build_mutual_state`.
    positions : Array
        ``(n, 3)`` positions in the caller's original order.
    masses : Array
        ``(n,)`` masses in the caller's original order.
    rung : Optional[Array]
        ``(n,)`` per-particle rung. Required whenever ``level_weights`` is given.
    level_weights : Optional[Array]
        ``(k_max + 1,)`` weight per interaction level. ``None`` weights every
        level by one, giving the full acceleration.
    return_parts : bool
        Return the near/far split alongside the acceleration.

    Returns
    -------
    Array | MutualForceResult
        ``(n, 3)`` accelerations, or a :class:`MutualForceResult` when
        ``return_parts`` is set. Zero-mass particles get zero acceleration.
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
    """Return the full mutual FMM acceleration (all levels, unit weights).

    Parameters
    ----------
    state : MutualFMMState
        Frozen state from :func:`build_mutual_state`.
    positions : Array
        ``(n, 3)`` positions in the caller's original order.
    masses : Array
        ``(n,)`` masses in the caller's original order.
    **kwargs : Any
        Forwarded verbatim to :func:`mutual_weighted_accelerations`.

    Returns
    -------
    Array | MutualForceResult
        ``(n, 3)`` accelerations, or a :class:`MutualForceResult` if
        ``return_parts=True`` was forwarded.
    """
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

    Parameters
    ----------
    state : MutualFMMState
        Frozen state from :func:`build_mutual_state`.
    positions : Array
        ``(n, 3)`` positions in the caller's original order.
    masses : Array
        ``(n,)`` masses in the caller's original order.
    rung : Array
        ``(n,)`` per-particle rung, in the caller's original order.
    level : int
        Interaction level to isolate.
    k_max : int
        Finest rung in the hierarchy; sets the length of the weight vector.

    Returns
    -------
    Array
        ``(n, 3)`` accelerations from level ``level`` alone.
    """
    weights = jnp.zeros((int(k_max) + 1,), dtype=jnp.asarray(positions).dtype)
    weights = weights.at[int(level)].set(1.0)
    return mutual_weighted_accelerations(
        state, positions, masses, rung=rung, level_weights=weights
    )
