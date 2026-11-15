"""The field at points that are not sources: ``target_points`` (issue #302).

The fused lane evaluates its force at its own sources. A caller that needs the
field somewhere else -- the galaxy's tidal field at the stars of a globular
cluster that is integrated by direct summation (hybrid Hermite, multi-node plan
U6), an observation operator's tracers -- had to append the points as zero-mass
particles, which breaks the mass-dependent criterion, takes tree capacity and
moves the topology. This module evaluates the field of a :class:`FusedTopology`'s
sources at ``M`` massless points instead, with the topology untouched:

1. **Upward pass.** The topology keeps its local expansions, not its multipoles,
   so the sources' P2M/M2M is run again on the topology's tree (the rebuild's own
   real sweep, ``prepare_real_upward_sweep``) about the COMs the locals were
   expanded about (``local_data.centers``, reused: no mass-moment pass), with the
   walk's exact COM radii (:func:`~jaccpot.runtime._mac_geometry.com_mac_geometry`);
   a node's mass is its monopole.
2. **Target groups.** The points are sorted along a Morton curve of their own
   bounding box and cut into groups of ``group_size``; a group is a sphere (its
   box centre, the largest member distance). The groups are not a tree and do not
   enter the source tree: they exist only for the walk below.
3. **One-sided walk** (:attr:`TargetPlan.walk`). The criterion is the source
   walk's: the geometric opening angle ``(r_n + r_g)^2 <= theta^2 d^2`` with the
   separation floor (a compact kernel's support), or eq (16a) of Dehnen (2014)
   for ``mac_type='dehnen_error'`` -- the SOURCE node's multipole error against
   the GROUP's threshold ``adaptive_eps * min_b f_b`` over the group's own points
   (their f_b, never the sources'). A node is used if it is accepted and every
   ancestor was rejected; a leaf whose ancestors were all rejected and that is
   rejected itself is a near leaf.

   * ``"frontier"`` (the default, session 2): each group descends the source
     tree from the root, breadth first, through ONE ring queue of pending
     (group, node) pairs (``queue_capacity``), ``walk_chunk`` pairs per
     ``while_loop`` step, new groups' (group, root) seeds topping each chunk up:
     a pair is tested once; accepted -> far, rejected leaf -> near, rejected
     internal node -> its two children are queued. The MAC is tested on
     ``O(groups x accepted x 2)`` pairs instead of every node (at 1e5 sources
     x 1e3 points 26,268 tests vs the dense 1,507,296; at 1e8 the dense block did
     not fit an A100). The accepted SET is the dense walk's, pair for pair.
   * ``"dense"`` (session 1, kept for comparisons): every group tests every
     node (pointer doubling over the parent array for the ancestors), in blocks
     sized to ``_WALK_BUDGET_BYTES``.

   Either walk compacts its pairs into FLAT (group, node) far and (group, leaf)
   near lists whose lengths are the plan's TOTAL capacities.
4. **Far** (:attr:`TargetPlan.far`). ``"m2p"`` (default): M2P per point -- M2L
   of the used nodes onto the point itself, L2P at zero offset (the lane's own
   real-basis operators); eq (15) is taken with ``rho_B = 0`` at the group's
   closest member. ``"local"``: one M2L per (group, node) into a local
   expansion about the group's centre and one L2P per point, with the group
   radius in the criterion (eq 16a two-sided; the geometric angle has it
   already) -- ``group_size`` times fewer M2L, at a slightly larger error at
   equal criterion (``docs/production_2026-11.md``, HH2a).
   **Near:** the softened pair kernel of the engine
   (:func:`~jaccpot.softening.masked_pair_factors`) over the near leaves'
   particles; a point on top of a source gets the kernel's finite value. Both
   run ``plan.pair_chunk`` pairs at a time and scatter-add into the points.

A ``point_mask`` evaluates a subset of a static-length point array (the dead
points sort last, seed no group and get zeros);
:mod:`jaccpot.runtime.target_back_reaction` uses it for the inner points of a
compact source set's field at very many points (the hybrid back-reaction).

Everything is traced and static-shaped: the :class:`TargetPlan` (group size,
total far/near capacities, chunk) is the only static input, so moving points
recompile nothing. An overflow of either capacity is reported (``ok``,
``needs``; the frontier queue's peak in ``stats``), never silent;
:func:`plan_target_points` sizes the capacities eagerly.

**Memory** is bounded by the flat lists (8 bytes per capacity entry), the
frontier queue (8 bytes per entry) or one dense walk block
(``_WALK_BUDGET_BYTES``), and one chunk of evaluation temporaries
(``pair_chunk x group_size`` M2P pairs of ~7 KB at p6, and as many near pairs x
``leaf_cap`` slots of ~30 B) -- not by groups x the WORST group's needs. The first
version gave every group the worst group's capacity: galaxy points next to a
cluster needed ~1e3 cluster nodes, and 31,250 galaxy groups x 32 points x 1e3
nodes x 7 KB asked for 14.9 GiB on an A100 (2026-11-11). The scatter-add makes the
GPU sums order-nondeterministic at round-off (the CPU's are deterministic).

Accuracy: the far field is M2P, so the error per accepted node is the
multipole's own truncation, which the criterion bounds (eq 16a) or the opening
angle limits. Not implemented here (left
on #302): gradients through the walk's lists (the arithmetic is plain ``jnp``,
but no seam freezes the lists), the ``dehnen_theta`` criterion, the
``mac_force_scale_floor`` on targets, the solver facade's ``target_points=`` and
the mesh.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

__all__ = [
    "TargetField",
    "TargetPlan",
    "TargetSources",
    "bootstrap_target_force_scale",
    "evaluate_target_points",
    "plan_target_points",
    "plan_target_points_from_sources",
    "target_mac_type",
]

#: the criteria the target walk implements
_GEOMETRIC_MACS = ("dehnen", "bh")
#: the target walks (:attr:`TargetPlan.walk`)
_WALKS = ("frontier", "dense")
#: the far-field forms (:attr:`TargetPlan.far`)
_FARS = ("m2p", "local")
_TINY = 1e-24


@dataclasses.dataclass(frozen=True)
class TargetPlan:
    """Static shapes of a target evaluation: a compile key, never traced.

    The capacities are TOTALS over all groups -- the walk's accepted (group,
    node) far pairs and (group, leaf) near pairs, compacted into flat lists --
    so a few groups that need many pairs (points inside a dense source region)
    do not make every group pay for them. Memory is bounded by
    ``far_capacity + near_capacity`` index pairs, the walk's
    ``group_block x nodes`` masks and one ``pair_chunk`` of evaluation
    temporaries (:func:`target_memory_estimate`).

    Attributes
    ----------
    num_points : int
        ``M``, the number of target points.
    group_size : int
        Points per target group.
    far_capacity : int
        Total far (M2P) (group, node) pairs.
    near_capacity : int
        Total near (P2P) (group, leaf) pairs.
    group_block : int
        Upper bound on the groups walked together; the evaluation lowers it so
        the walk's masks stay within ``_WALK_BUDGET_BYTES`` (a large tree walks
        one group at a time).
    pair_chunk : int
        Pairs evaluated together.
    walk : str
        ``"frontier"`` (default): each group descends the source tree from the
        root, breadth first, through a ring queue of (group, node) pairs of
        ``queue_capacity`` entries, ``walk_chunk`` pairs a step -- the MAC is
        tested only on the children of opened nodes. ``"dense"``: every group
        tests every node (session 1's walk; ``groups x nodes`` tests, kept for
        comparisons).
    queue_capacity : int
        The frontier walk's ring queue (pending (group, node) pairs).
    walk_chunk : int
        Pairs the frontier walk tests per step (also the seeds' granularity).
    far : str
        The far field: ``"m2p"`` (default) -- each accepted node's multipole
        at every point of the group (M2L onto the point, L2P at zero offset);
        ``"local"`` -- one M2L per (group, node) into a local expansion about
        the group's centre, then one L2P per point, with the group's radius in
        the criterion (eq 16a two-sided, ``rho_B`` = the group radius; the
        geometric angle already has it). For COMPACT target groups only (a
        cluster's stars: the same accuracy for ~1/32 of the M2L work): on
        sparse wide groups its truncation, ``((r_n + r_g)/d)^(p+1)`` against
        M2P's ``(r_n/d)^(p+1)``, costs orders of magnitude
        (``docs/production_2026-11.md``, HH2a).
    """

    num_points: int
    group_size: int = 32
    far_capacity: int = 4096
    near_capacity: int = 1024
    group_block: int = 64
    pair_chunk: int = 128
    walk: str = "frontier"
    queue_capacity: int = 8192
    walk_chunk: int = 2048
    far: str = "m2p"

    def __post_init__(self) -> None:
        for name in (
            "num_points",
            "group_size",
            "far_capacity",
            "near_capacity",
            "group_block",
            "pair_chunk",
            "queue_capacity",
            "walk_chunk",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"TargetPlan.{name} must be >= 1")
        if self.far not in _FARS:
            raise ValueError(f"TargetPlan.far must be one of {_FARS}, got {self.far!r}")
        if self.walk not in _WALKS:
            raise ValueError(
                f"TargetPlan.walk must be one of {_WALKS}, got {self.walk!r}"
            )

    @property
    def num_groups(self) -> int:
        """Groups after padding ``M`` up to a multiple of ``group_size``."""
        return -(-int(self.num_points) // int(self.group_size))

    def grown(
        self, needs: Any, headroom: float = 1.5, stats: Optional[Any] = None
    ) -> "TargetPlan":
        """A plan whose capacities cover ``needs`` with ``headroom``.

        Parameters
        ----------
        needs : Any
            ``(far, near)`` total pair counts an evaluation needed.
        headroom : float
            Multiplier on the needs.
        stats : Optional[Any]
            The evaluation's :attr:`TargetField.stats`: its first entry (the
            frontier queue's peak) sizes ``queue_capacity``; an overflowed queue
            (peak above the capacity) at least doubles it, since a dropped
            subtree hides its own pending pairs.

        Returns
        -------
        TargetPlan
            The same plan with capacities ``max(old, need * headroom)``, rounded
            up to a multiple of ``pair_chunk`` (the queue: of ``walk_chunk``).
        """
        far, near = (int(v) for v in np.asarray(needs).reshape(-1)[:2])
        chunk = int(self.pair_chunk)
        queue = int(self.queue_capacity)
        if stats is not None:
            peak = int(np.asarray(stats).reshape(-1)[0])
            if peak > queue:
                queue = max(2 * queue, _ceil_to(peak * 2, int(self.walk_chunk)))
            queue = max(queue, _ceil_to(peak * headroom, int(self.walk_chunk)))
        return dataclasses.replace(
            self,
            far_capacity=max(int(self.far_capacity), _ceil_to(far * headroom, chunk)),
            near_capacity=max(
                int(self.near_capacity), _ceil_to(near * headroom, chunk)
            ),
            queue_capacity=queue,
        )


class TargetField(NamedTuple):
    """What a target evaluation returns (in the points' order and dtype).

    ``needs`` are TOTALS over the groups (see :class:`TargetPlan`).

    Attributes
    ----------
    acc : Array
        ``[M, 3]`` acceleration of the sources' field at the points.
    potential : Optional[Array]
        ``[M]`` potential ``-G sum m psi`` (``None`` unless asked).
    force_scale : Optional[Array]
        ``[M]`` eq (16b)'s ``f_b = sum G m / (r^2 + eps^2)`` at the points (far
        nodes as monopoles): the next evaluation's ``force_scale`` under
        ``dehnen_error`` (``None`` unless asked).
    needs : Array
        ``int32 [2]``: the total far / near pair counts the walk accepted.
    ok : Array
        Boolean scalar: both counts (and the frontier queue) fit the plan (the
        result is then complete).
    stats : Optional[Array]
        ``int [3]``: the frontier queue's peak of pending pairs (``0`` for the
        dense walk), the MAC tests the walk ran, and the points the walk served
        (the live points; a back-reaction's inner points).
    """

    acc: Array
    potential: Optional[Array]
    force_scale: Optional[Array]
    needs: Array
    ok: Array
    stats: Optional[Array] = None


def _ceil_to(value: Union[int, float], multiple: int) -> int:
    """``value`` rounded up to a multiple of ``multiple``, at least one multiple.

    Parameters
    ----------
    value : Union[int, float]
        A count.
    multiple : int
        The granularity.

    Returns
    -------
    int
        The rounded count.
    """
    m = max(1, int(multiple))
    return max(m, int(m * math.ceil(float(value) / m)))


#: bytes the walk's per-block masks may take (``group_block x nodes`` rows of a
#: few words each): a large tree walks fewer groups at a time
_WALK_BUDGET_BYTES = 256 * 2**20
#: measured temporaries per (group, node) entry of the walk (``mem_new.py``)
_WALK_BYTES_PER_ENTRY = 64


def _walk_block(plan: "TargetPlan", nodes: int) -> int:
    """Groups walked together: ``plan.group_block`` capped by the walk budget.

    Parameters
    ----------
    plan : TargetPlan
        The plan.
    nodes : int
        Source tree nodes.

    Returns
    -------
    int
        A power of two in ``[1, plan.group_block]``.
    """
    fit = max(1, _WALK_BUDGET_BYTES // max(1, int(nodes) * _WALK_BYTES_PER_ENTRY))
    block = 1
    while block * 2 <= min(int(plan.group_block), fit):
        block *= 2
    return block


def target_mac_type(engine: Any, mac: Optional[str] = None) -> str:
    """The criterion a target walk on ``engine``'s topology uses.

    Parameters
    ----------
    engine : Any
        The provider's engine.
    mac : Optional[str]
        Override: ``"dehnen"`` (geometric) or ``"dehnen_error"``; ``None`` takes
        the engine's.

    Returns
    -------
    str
        ``"dehnen"`` / ``"bh"`` (geometric) or ``"dehnen_error"``.

    Raises
    ------
    NotImplementedError
        For a criterion the target walk does not implement (``dehnen_theta``,
        ``engblom``).
    """
    if mac is None:
        if bool(engine._flat_walk_criterion_active()):
            name = "dehnen_error"
        elif bool(engine._uses_per_node_effective_theta()):
            name = "dehnen_theta"
        else:
            name = str(engine._base_mac_type())
    else:
        name = str(mac)
    if name in _GEOMETRIC_MACS or name == "dehnen_error":
        return name
    raise NotImplementedError(
        f"target_points: mac_type {name!r} is not implemented (geometric "
        "'dehnen'/'bh' and 'dehnen_error' are; see issue #302)"
    )


def _morton_order(points: Array, live: Optional[Array] = None) -> Array:
    """Argsort of the points along a 10-bit-per-axis Morton curve of their box.

    Parameters
    ----------
    points : Array
        ``[M, 3]``.
    live : Optional[Array]
        ``[M]`` bool: only these span the box, and the others sort last.

    Returns
    -------
    Array
        ``[M]`` the stable argsort of the Morton codes.
    """
    if live is None:
        lo = jnp.min(points, axis=0)
        hi = jnp.max(points, axis=0)
    else:
        lo = jnp.min(jnp.where(live[:, None], points, jnp.inf), axis=0)
        hi = jnp.max(jnp.where(live[:, None], points, -jnp.inf), axis=0)
        anyl = jnp.any(live)
        lo = jnp.where(anyl, lo, 0.0)
        hi = jnp.where(anyl, hi, 0.0)
        points = jnp.where(live[:, None], points, lo)
    span = jnp.maximum(jnp.max(hi - lo), jnp.asarray(_TINY, points.dtype))
    q = jnp.clip(((points - lo) / span) * 1023.0, 0.0, 1023.0).astype(jnp.uint32)

    def spread(v: Array) -> Array:
        v = (v | (v << 16)) & jnp.uint32(0x030000FF)
        v = (v | (v << 8)) & jnp.uint32(0x0300F00F)
        v = (v | (v << 4)) & jnp.uint32(0x030C30C3)
        v = (v | (v << 2)) & jnp.uint32(0x09249249)
        return v

    code = spread(q[:, 0]) | (spread(q[:, 1]) << 1) | (spread(q[:, 2]) << 2)
    if live is not None:
        # 30-bit codes: the dead points' key sorts after every live one
        code = jnp.where(live, code, jnp.uint32(0xFFFFFFFF))
    return jnp.argsort(code, stable=True)


class _SourceTables(NamedTuple):
    """Per-node arrays of the source tree the walk and the far field read.

    Attributes
    ----------
    centers : Array
        ``[nodes + 1, 3]`` expansion centres (COM); row ``nodes`` is a zero
        sentinel.
    radius : Array
        ``[nodes]`` the walk's exact COM radii.
    multipoles : Array
        ``[nodes, (p+1)^2]`` packed real multipoles. No sentinel row (a copy of
        ~2.4 GB at 1e8 sources): padding pairs read it clipped and are masked.
    mass : Array
        ``[nodes + 1]`` node masses (sentinel zero).
    nonempty : Array
        ``[nodes]`` nodes with particles and mass.
    is_leaf : Array
        ``[nodes]``.
    parent : Array
        ``[nodes + 1]``; the root's and the sentinel's parent is the sentinel.
    ranges : Array
        ``[nodes + 1, 2]`` inclusive particle ranges (the sentinel's empty).
    table : Optional[Array]
        ``[nodes, W]`` eq (16a)'s table (``dehnen_error`` only).
    positions : Array
        ``[n, 3]`` sorted source positions.
    masses : Array
        ``[n]`` sorted source masses.
    children : Array
        ``[nodes + 1, 2]`` an internal node's two children (the frontier walk's
        descent); leaves and the sentinel have the sentinel ``nodes``.
    root : Array
        The root node (``int32`` scalar).
    """

    centers: Array
    radius: Array
    multipoles: Array
    mass: Array
    nonempty: Array
    is_leaf: Array
    parent: Array
    ranges: Array
    table: Optional[Array]
    positions: Array
    masses: Array
    children: Array
    root: Array


def _source_tables(
    engine: Any, topology: Any, *, order: int, mac: str
) -> _SourceTables:
    """The upward pass and the walk geometry of a topology's source tree.

    Parameters
    ----------
    engine : Any
        The provider's engine.
    topology : Any
        A ``FusedTopology`` with a full prepared state.
    order : int
        Expansion order.
    mac : str
        The target walk's criterion (:func:`target_mac_type`).

    Returns
    -------
    _SourceTables
        The per-node tables.
    """
    from yggdrax.interactions import WALK_TABLE_THRESHOLD

    from jaccpot.upward.real_tree_expansions import prepare_real_upward_sweep

    from ._mac_geometry import com_mac_geometry
    from ._walk_criterion import dehnen_walk_table

    prepared = topology.prepared
    tree = prepared.tree
    pos = jnp.asarray(tree.positions_sorted)
    mas = jnp.asarray(tree.masses_sorted, pos.dtype)
    leaf_cap = int(prepared.max_leaf_size)
    num_levels = engine._resolve_upward_num_levels(tree)
    nodes = int(jnp.asarray(tree.node_ranges).shape[0])
    # the topology's locals are expanded about the same COMs the rebuild's
    # sweep computed (bitwise): reusing them skips the mass-moment pass, whose
    # float64 prefix sums over every particle were this program's largest
    # buffers at 1e8 (a 7.7 GiB request on an A100 next to the galaxy's state)
    local = getattr(prepared, "local_data", None)
    known = None if local is None else getattr(local, "centers", None)
    if known is not None and tuple(jnp.shape(known)) != (nodes, 3):
        known = None
    upward = prepare_real_upward_sweep(
        tree,
        pos,
        mas,
        max_order=int(order),
        max_leaf_size=leaf_cap,
        leaf_batch_size=getattr(engine, "upward_leaf_batch_size", None),
        static_num_levels=num_levels,
        centers=known,
    )
    centers = jnp.asarray(upward.multipoles.centers, pos.dtype)
    packed = jnp.asarray(upward.multipoles.packed)
    radius = com_mac_geometry(
        tree, pos, centers, leaf_cap=leaf_cap, num_levels=num_levels
    ).radius
    radius = jnp.asarray(radius, pos.dtype) * jnp.asarray(
        float(getattr(engine, "dehnen_radius_scale", 1.0) or 1.0), pos.dtype
    )
    # the monopole is the node's mass (summed in the working dtype: within
    # ~1e-5 of the float64 prefix sums, which the criterion does not notice)
    mass = jnp.asarray(packed[:, 0], pos.dtype)
    ranges = jnp.asarray(tree.node_ranges, jnp.int32)
    internal = int(jnp.asarray(tree.left_child).shape[0])
    counts = ranges[:, 1] - ranges[:, 0] + 1
    raw_parent = jnp.asarray(tree.parent, jnp.int32)
    parent = jnp.where((raw_parent < 0) | (raw_parent >= nodes), nodes, raw_parent)
    root = jnp.argmax(raw_parent < 0).astype(jnp.int32)
    kids = jnp.stack(
        [
            jnp.asarray(tree.left_child, jnp.int32).reshape(-1),
            jnp.asarray(tree.right_child, jnp.int32).reshape(-1),
        ],
        axis=1,
    )
    kids = jnp.where((kids < 0) | (kids >= nodes), nodes, kids)
    children = jnp.concatenate(
        [kids, jnp.full((nodes + 1 - internal, 2), nodes, jnp.int32)]
    )
    table = None
    if mac == "dehnen_error":
        # the source rows of eq (16a); the reverse direction (targets source
        # nothing) is made to pass with an infinite source threshold
        table = dehnen_walk_table(
            multipole_packed=packed,
            mass=mass,
            radius=radius,
            threshold=jnp.full((nodes,), jnp.inf, pos.dtype),
            gravitational_constant=float(engine.G),
            order=int(order),
        ).astype(pos.dtype)
        table = table.at[:, WALK_TABLE_THRESHOLD].set(jnp.inf)
    return _SourceTables(
        centers=jnp.concatenate([centers, jnp.zeros((1, 3), pos.dtype)]),
        radius=radius,
        multipoles=packed,
        mass=jnp.concatenate([mass, jnp.zeros((1,), pos.dtype)]),
        nonempty=(counts > 0) & (mass != 0.0),
        is_leaf=jnp.arange(nodes) >= internal,
        parent=jnp.concatenate([parent, jnp.asarray([nodes], jnp.int32)]),
        ranges=jnp.concatenate([ranges, jnp.asarray([[0, -1]], jnp.int32)]),
        table=table,
        positions=pos,
        masses=mas,
        children=children,
        root=root,
    )


def _pair_accept(
    src: _SourceTables,
    node: Array,
    centre: Array,
    rad: Array,
    threshold: Optional[Array],
    *,
    mac: str,
    theta: float,
    theta_max: float,
    floor: float,
    order: int,
    far: str = "m2p",
) -> Array:
    """The criterion on ``K`` (group, node) pairs, elementwise.

    Both walks call this: the dense one with every node against one group, the
    frontier one with a chunk of queued pairs, so the two accept the same pairs
    with the same arithmetic.

    Parameters
    ----------
    src : _SourceTables
        The source tree.
    node : Array
        ``[K]`` source nodes (``< nodes``).
    centre : Array
        ``[K, 3]`` (or ``[3]``) the groups' centres.
    rad : Array
        ``[K]`` (or scalar) the groups' radii.
    threshold : Optional[Array]
        ``[K]`` (or scalar) the groups' eq (16a) thresholds (``dehnen_error``).
    mac : str
        The criterion.
    theta : float
        The geometric opening angle.
    theta_max : float
        Eq (16a)'s convergence bound.
    floor : float
        The walk's separation floor.
    order : int
        Expansion order.
    far : str
        :attr:`TargetPlan.far`: under ``dehnen_error``, ``"m2p"`` takes eq (15)
        with ``rho_B = 0`` at the group's closest member, ``"local"`` the
        two-sided form with ``rho_B`` = the group radius at the centre distance.

    Returns
    -------
    Array
        ``[K]`` bool: the node's multipole may serve the whole group (it is
        non-empty, off the group's centre, outside the floor and accepted).
    """
    from yggdrax.interactions import WALK_TABLE_POWER0, dehnen_pair_accept

    dtype = src.radius.dtype
    nradius = src.radius[node]
    diff = src.centers[node] - centre
    dist_sq = jnp.sum(diff * diff, axis=-1)
    rad = jnp.broadcast_to(jnp.asarray(rad, dtype), dist_sq.shape)
    valid = src.nonempty[node] & (dist_sq > 0.0)
    if float(floor) > 0.0:
        reach = nradius + rad + jnp.asarray(floor, dtype)
        valid = valid & (reach * reach <= dist_sq)
    if mac == "dehnen_error":
        # the far field is M2P (no local expansion about the group), so eq (15)
        # is taken with rho_B = 0 at the distance of the group's closest possible
        # member, d - r_g: an upper bound for every point of the group
        cols = WALK_TABLE_POWER0 + int(order)
        rows = src.table[node]
        row_a = [rows[..., k] for k in range(cols)]
        zero = jnp.zeros(dist_sq.shape, dtype)
        row_b = [zero, jnp.broadcast_to(threshold, dist_sq.shape)] + [
            zero for _ in range(cols - 2)
        ]
        if far == "local":
            # M2L into the group's local expansion: the error of the pair at the
            # centre distance with both radii (eq 16a; the group sources nothing)
            radius_b, pair_d2 = rad, dist_sq
        else:
            closest = jnp.maximum(jnp.sqrt(dist_sq) - rad, 0.0)
            radius_b, pair_d2 = zero, closest * closest
        return valid & dehnen_pair_accept(
            row_a=row_a,
            row_b=row_b,
            radius_a=nradius,
            radius_b=radius_b,
            dist_sq=pair_d2,
            order=int(order),
            theta_max=jnp.asarray(theta_max, dtype),
        )
    extent = nradius + rad
    return valid & (extent * extent <= jnp.asarray(theta * theta, dtype) * dist_sq)


def _walk_masks(
    src: _SourceTables,
    centre: Array,
    rad: Array,
    threshold: Optional[Array],
    *,
    mac: str,
    theta: float,
    theta_max: float,
    floor: float,
    order: int,
    far: str = "m2p",
) -> tuple[Array, Array]:
    """``(far, near)`` node masks of ONE group against every source node (dense).

    Parameters
    ----------
    src : _SourceTables
        The source tree.
    centre : Array
        ``[3]`` the group's centre.
    rad : Array
        The group's radius.
    threshold : Optional[Array]
        The group's eq (16a) threshold (``dehnen_error``).
    mac : str
        The criterion.
    theta : float
        The geometric opening angle.
    theta_max : float
        Eq (16a)'s convergence bound.
    floor : float
        The walk's separation floor.
    order : int
        Expansion order.
    far : str
        :attr:`TargetPlan.far` (the criterion's form).

    Returns
    -------
    tuple[Array, Array]
        ``[nodes]`` far (M2P) and near (P2P leaf) masks.
    """
    nodes = int(src.radius.shape[0])
    accept = _pair_accept(
        src,
        jnp.arange(nodes, dtype=jnp.int32),
        centre[None, :],
        rad,
        threshold,
        mac=mac,
        theta=theta,
        theta_max=theta_max,
        floor=floor,
        order=order,
        far=far,
    )
    # visited: every strict ancestor rejected (pointer doubling; the sentinel
    # `nodes` is its own parent and counts as rejected)
    rejected = jnp.concatenate([~accept, jnp.ones((1,), bool)])
    up = src.parent
    chain = rejected[up]
    for _ in range(int(math.ceil(math.log2(max(nodes, 2)))) + 1):
        chain = chain & chain[up]
        up = up[up]
    visited = chain[:nodes]
    far = visited & accept
    near = visited & ~accept & src.is_leaf & src.nonempty
    return far, near


def _far_chunk(
    src: _SourceTables,
    groups: Array,
    gvalid: Array,
    grp: Array,
    node: Array,
    *,
    order: int,
    G: float,
    eps2_plummer: float,
    want_scale: bool,
) -> tuple[Array, Array, Array]:
    """M2P of one chunk of (group, node) far pairs at the groups' points.

    Parameters
    ----------
    src : _SourceTables
        The source tree.
    groups : Array
        ``[groups, G, 3]`` the points.
    gvalid : Array
        ``[groups, G]`` live points.
    grp : Array
        ``[C]`` group per pair.
    node : Array
        ``[C]`` source node per pair (``nodes``: padding).
    order : int
        Expansion order.
    G : float
        Gravitational constant.
    eps2_plummer : float
        The Plummer-equivalent softening squared (eq 16b's monopole term).
    want_scale : bool
        Also eq (16b)'s force scale.

    Returns
    -------
    tuple[Array, Array, Array]
        ``[C, G, 3]`` acceleration, ``[C, G]`` potential, ``[C, G]`` force scale.
    """
    from jaccpot.operators.m2l_real_rot_scale import m2l_rot_scale_real_batch
    from jaccpot.operators.real_p2m_l2p import evaluate_local_real_with_grad

    dtype = src.radius.dtype
    nodes = int(src.radius.shape[0])
    chunk = int(grp.shape[0])
    gsize = int(groups.shape[1])
    pts = groups[grp]  # [C, G, 3]
    live = (node < nodes)[:, None] & gvalid[grp]  # [C, G]
    centres = src.centers[node]  # [C, 3]
    deltas = pts - centres[:, None, :]
    deltas = jnp.where(live[..., None], deltas, jnp.ones_like(deltas))
    mult = jnp.broadcast_to(
        jnp.take(src.multipoles, node, axis=0, mode="clip")[:, None, :],
        (chunk, gsize, src.multipoles.shape[1]),
    )
    locals_ = m2l_rot_scale_real_batch(
        mult.reshape(chunk * gsize, -1), deltas.reshape(chunk * gsize, 3), order=order
    )
    zero = jnp.zeros((3,), dtype)
    grad, phi = jax.vmap(
        lambda loc: evaluate_local_real_with_grad(loc, zero, order=int(order))
    )(locals_)
    gconst = jnp.asarray(G, dtype)
    acc = jnp.where(live[..., None], -gconst * grad.reshape(chunk, gsize, 3), 0.0)
    pot = jnp.where(live, -gconst * phi.reshape(chunk, gsize), 0.0)
    scale = jnp.zeros((chunk, gsize), dtype)
    if want_scale:
        r2 = jnp.sum((pts - centres[:, None, :]) ** 2, axis=-1) + jnp.asarray(
            eps2_plummer, dtype
        )
        scale = jnp.where(live, gconst * src.mass[node][:, None] / r2, 0.0)
    return acc, pot, scale


def _far_local_chunk(
    src: _SourceTables,
    groups: Array,
    gvalid: Array,
    centres: Array,
    grp: Array,
    node: Array,
    *,
    order: int,
    G: float,
    eps2_plummer: float,
    want_scale: bool,
) -> tuple[Array, Array]:
    """M2L of one chunk of (group, node) far pairs into the groups' local expansions.

    Parameters
    ----------
    src : _SourceTables
        The source tree.
    groups : Array
        ``[groups, G, 3]`` the points.
    gvalid : Array
        ``[groups, G]`` live points.
    centres : Array
        ``[groups, 3]`` the groups' expansion centres.
    grp : Array
        ``[C]`` group per pair.
    node : Array
        ``[C]`` source node per pair (``nodes``: padding).
    order : int
        Expansion order.
    G : float
        Gravitational constant.
    eps2_plummer : float
        The Plummer-equivalent softening squared (eq 16b's monopole term).
    want_scale : bool
        Also eq (16b)'s force scale at the points.

    Returns
    -------
    tuple[Array, Array]
        ``[C, (p+1)^2]`` local contributions (zero for padding) and ``[C, G]``
        the force scale.
    """
    from jaccpot.operators.m2l_real_rot_scale import m2l_rot_scale_real_batch

    dtype = src.radius.dtype
    nodes = int(src.radius.shape[0])
    live = node < nodes
    gc = centres[grp]
    deltas = gc - src.centers[node]
    deltas = jnp.where(live[:, None], deltas, jnp.ones_like(deltas))
    loc = m2l_rot_scale_real_batch(
        jnp.take(src.multipoles, node, axis=0, mode="clip"), deltas, order=int(order)
    )
    loc = jnp.where(live[:, None], loc, 0.0)
    scale = jnp.zeros(gvalid[grp].shape, dtype)
    if want_scale:
        pts = groups[grp]
        r2 = jnp.sum((pts - src.centers[node][:, None, :]) ** 2, axis=-1) + (
            jnp.asarray(eps2_plummer, dtype)
        )
        alive = live[:, None] & gvalid[grp]
        scale = jnp.where(
            alive, jnp.asarray(G, dtype) * src.mass[node][:, None] / r2, 0.0
        )
    return loc, scale


def _l2p_groups(
    locals_: Array,
    groups: Array,
    gvalid: Array,
    centres: Array,
    *,
    order: int,
    G: float,
    batch: int,
) -> tuple[Array, Array]:
    """L2P of every group's local expansion at its points.

    Parameters
    ----------
    locals_ : Array
        ``[groups, (p+1)^2]``.
    groups : Array
        ``[groups, G, 3]``.
    gvalid : Array
        ``[groups, G]``.
    centres : Array
        ``[groups, 3]``.
    order : int
        Expansion order.
    G : float
        Gravitational constant.
    batch : int
        Groups per ``lax.map`` step.

    Returns
    -------
    tuple[Array, Array]
        ``[groups, G, 3]`` acceleration, ``[groups, G]`` potential.
    """
    from jaccpot.operators.real_p2m_l2p import evaluate_local_real_with_grad

    gconst = jnp.asarray(G, locals_.dtype)

    def one(xs: tuple) -> tuple[Array, Array]:
        loc, pts, ok, c = xs
        grad, phi = jax.vmap(
            lambda d: evaluate_local_real_with_grad(loc, d, order=int(order))
        )(c[None, :] - pts)
        acc = jnp.where(ok[:, None], -gconst * grad, 0.0)
        return acc, jnp.where(ok, -gconst * phi, 0.0)

    return jax.lax.map(
        one, (locals_, groups, gvalid, centres), batch_size=max(1, int(batch))
    )


def _near_chunk(
    src: _SourceTables,
    groups: Array,
    gvalid: Array,
    grp: Array,
    leaf: Array,
    *,
    leaf_cap: int,
    G: float,
    soft_params: Array,
    kernel: str,
    eps2_plummer: float,
    want_potential: bool,
    want_scale: bool,
) -> tuple[Array, Array, Array]:
    """Softened P2P of one chunk of (group, leaf) near pairs.

    Parameters
    ----------
    src : _SourceTables
        The source tree.
    groups : Array
        ``[groups, G, 3]`` the points.
    gvalid : Array
        ``[groups, G]`` live points.
    grp : Array
        ``[C]`` group per pair.
    leaf : Array
        ``[C]`` source leaf (node id) per pair (``nodes``: padding, empty).
    leaf_cap : int
        Particles per leaf at most.
    G : float
        Gravitational constant.
    soft_params : Array
        :func:`~jaccpot.softening.softening_params` of the engine.
    kernel : str
        The softening kernel.
    eps2_plummer : float
        The Plummer-equivalent softening squared (eq 16b).
    want_potential : bool
        Accumulate the potential.
    want_scale : bool
        Accumulate eq (16b)'s force scale.

    Returns
    -------
    tuple[Array, Array, Array]
        ``[C, G, 3]`` acceleration, ``[C, G]`` potential, ``[C, G]`` force scale.
    """
    from jaccpot.softening import masked_pair_factors

    dtype = src.radius.dtype
    pts = groups[grp]  # [C, G, 3]
    start = src.ranges[leaf, 0]
    count = src.ranges[leaf, 1] - start + 1
    slot = jnp.arange(int(leaf_cap), dtype=jnp.int32)
    sidx = start[:, None] + slot[None, :]  # [C, L]
    slive = slot[None, :] < count[:, None]
    sidx = jnp.where(slive, sidx, 0)
    spos = src.positions[sidx]  # [C, L, 3]
    smas = jnp.where(slive, src.masses[sidx], 0.0)
    dx = pts[:, :, None, :] - spos[:, None, :, :]  # [C, G, L, 3]
    r2 = jnp.sum(dx * dx, axis=-1)
    active = slive[:, None, :] & gvalid[grp][:, :, None]
    g, psi, _ = masked_pair_factors(
        r2, active, soft_params, kernel, potential=want_potential
    )
    gconst = jnp.asarray(G, dtype)
    w = smas[:, None, :] * g
    acc = -gconst * jnp.sum(w[..., None] * dx, axis=2)
    pot = jnp.zeros(r2.shape[:2], dtype)
    if want_potential:
        pot = -gconst * jnp.sum(smas[:, None, :] * psi, axis=2)
    scale = jnp.zeros(r2.shape[:2], dtype)
    if want_scale:
        r2s = jnp.where(active, r2 + jnp.asarray(eps2_plummer, dtype), 1.0)
        scale = gconst * jnp.sum(jnp.where(active, smas[:, None, :] / r2s, 0.0), axis=2)
    return acc, pot, scale


def _compact_pairs(
    src: _SourceTables,
    centres: Array,
    radii: Array,
    thresholds: Array,
    live_groups: Array,
    *,
    block: int,
    far_cap: int,
    near_cap: int,
    walk: dict,
) -> tuple[Array, Array, Array, Array, Array]:
    """The walk over every group, compacted into flat (group, node) pair lists.

    One sequential scan over blocks of ``block`` groups: each block's masks
    (``[block, nodes]``) are ranked and scattered into the flat lists at the
    running offset, so memory is the lists plus one block's masks.

    Parameters
    ----------
    src : _SourceTables
        The source tree.
    centres : Array
        ``[groups, 3]`` group centres (groups padded to a multiple of ``block``).
    radii : Array
        ``[groups]`` group radii.
    thresholds : Array
        ``[groups]`` eq (16a) thresholds (unused by the geometric criterion).
    live_groups : Array
        ``[groups]`` groups with a live point.
    block : int
        Groups per scan step.
    far_cap : int
        Far list length.
    near_cap : int
        Near list length.
    walk : dict
        :func:`_walk_masks`'s static keywords.

    Returns
    -------
    tuple[Array, Array, Array, Array, Array]
        Far groups, far nodes, near groups, near leaves (padding: group 0,
        node ``nodes``), and ``int32 [2]`` the total counts.
    """
    nodes = int(src.radius.shape[0])
    ngroups = int(centres.shape[0])
    nblocks = ngroups // int(block)
    node_ids = jnp.arange(nodes, dtype=jnp.int32)

    def masks(c: Array, r: Array, t: Array, ok: Array) -> tuple[Array, Array]:
        far, near = _walk_masks(src, c, r, t, **walk)
        return far & ok, near & ok

    def body(carry: tuple, xs: tuple) -> tuple[tuple, None]:
        fg, fn, ng, nn, off = carry
        gid, c, r, t, ok = xs
        far, near = jax.vmap(masks)(c, r, t, ok)  # [B, nodes]

        def place(
            mask: Array, start: Array, cap: int, lst_g: Array, lst_n: Array
        ) -> tuple[Array, Array, Array]:
            cnt = jnp.sum(mask, axis=1, dtype=jnp.int32)
            first = start + jnp.cumsum(cnt) - cnt  # [B]
            rank = jnp.cumsum(mask, axis=1, dtype=jnp.int32) - 1
            pos = jnp.where(mask, first[:, None] + rank, cap).reshape(-1)
            lst_g = lst_g.at[pos].set(
                jnp.broadcast_to(gid[:, None], mask.shape).reshape(-1), mode="drop"
            )
            lst_n = lst_n.at[pos].set(
                jnp.broadcast_to(node_ids[None, :], mask.shape).reshape(-1),
                mode="drop",
            )
            return lst_g, lst_n, start + jnp.sum(cnt, dtype=jnp.int32)

        fg, fn, f_end = place(far, off[0], far_cap, fg, fn)
        ng, nn, n_end = place(near, off[1], near_cap, ng, nn)
        return (fg, fn, ng, nn, jnp.stack([f_end, n_end])), None

    gids = jnp.arange(ngroups, dtype=jnp.int32).reshape(nblocks, block)
    init = (
        jnp.zeros((far_cap,), jnp.int32),
        jnp.full((far_cap,), nodes, jnp.int32),
        jnp.zeros((near_cap,), jnp.int32),
        jnp.full((near_cap,), nodes, jnp.int32),
        jnp.zeros((2,), jnp.int32),
    )
    xs = (
        gids,
        centres.reshape(nblocks, block, 3),
        radii.reshape(nblocks, block),
        thresholds.reshape(nblocks, block),
        live_groups.reshape(nblocks, block),
    )
    (fg, fn, ng, nn, totals), _ = jax.lax.scan(body, init, xs)
    return fg, fn, ng, nn, totals


def _count_dtype() -> Any:
    """The MAC-test counter's dtype.

    Returns
    -------
    Any
        ``int64`` under x64 (the default here), else ``int32``.
    """
    return jnp.int64 if bool(jax.config.read("jax_enable_x64")) else jnp.int32


def _frontier_pairs(
    src: _SourceTables,
    centres: Array,
    radii: Array,
    thresholds: Array,
    n_live: Array,
    *,
    chunk: int,
    queue_cap: int,
    far_cap: int,
    near_cap: int,
    walk: dict,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """The frontier walk: every group descends the source tree from the root.

    A ring queue of ``queue_cap`` pending (group, node) pairs, processed
    ``chunk`` at a time in a ``while_loop`` (breadth first, first in first out).
    A step pops up to ``chunk`` pairs and tops the chunk up with new groups'
    (group, root) seeds, so only the groups in flight hold queue entries. Each
    pair is tested once (:func:`_pair_accept`): accepted -> the far list;
    rejected leaf -> the near list; rejected internal node -> its two children
    are queued. The accepted SET is the dense walk's (a node is used iff it is
    accepted and every ancestor was rejected); only the lists' order differs.
    Shapes are static; a full queue drops the pushes that do not fit and says so
    (the peak it would have needed is reported).

    Parameters
    ----------
    src : _SourceTables
        The source tree.
    centres : Array
        ``[groups, 3]`` group centres.
    radii : Array
        ``[groups]`` group radii.
    thresholds : Array
        ``[groups]`` eq (16a) thresholds (unused by the geometric criterion).
    n_live : Array
        The live groups (a prefix of the groups): ``int32`` scalar.
    chunk : int
        Pairs per step.
    queue_cap : int
        Ring queue entries.
    far_cap : int
        Far list length.
    near_cap : int
        Near list length.
    walk : dict
        :func:`_pair_accept`'s static keywords.

    Returns
    -------
    tuple[Array, Array, Array, Array, Array, Array]
        Far groups, far nodes, near groups, near leaves (padding: group 0, node
        ``nodes``), ``int32 [2]`` the total counts, and ``[2]`` (queue peak,
        MAC tests).
    """
    nodes = int(src.radius.shape[0])
    width = int(chunk)
    qcap = int(queue_cap)
    cdt = _count_dtype()
    lane = jnp.arange(width, dtype=jnp.int32)
    is_leaf = jnp.concatenate([src.is_leaf, jnp.ones((1,), bool)])
    near_ok = jnp.concatenate([src.is_leaf & src.nonempty, jnp.zeros((1,), bool)])
    has_any = (src.ranges[:, 1] - src.ranges[:, 0] + 1) > 0
    n_live = jnp.asarray(n_live, jnp.int32)

    def emit(
        mask: Array,
        start: Array,
        cap: int,
        lst_g: Array,
        lst_n: Array,
        g: Array,
        n: Array,
    ) -> tuple[Array, Array, Array]:
        rank = jnp.cumsum(mask, dtype=jnp.int32) - 1
        pos = jnp.where(mask, start + rank, cap)
        lst_g = lst_g.at[pos].set(g, mode="drop")
        lst_n = lst_n.at[pos].set(n, mode="drop")
        return lst_g, lst_n, start + jnp.sum(mask, dtype=jnp.int32)

    def cond(st: tuple) -> Array:
        return (st[3] > 0) | (st[4] < n_live)

    def body(st: tuple) -> tuple:
        qg, qn, head, pend, seed, fg, fn, ng, nn, cnt, peak, tests = st
        take = jnp.minimum(pend, width)
        from_q = lane < take
        qidx = (head + lane) % qcap
        sid = seed + lane - take
        from_s = (~from_q) & (sid < n_live)
        live = from_q | from_s
        g = jnp.where(from_q, qg[qidx], jnp.where(from_s, sid, 0))
        n = jnp.where(from_q, qn[qidx], jnp.where(from_s, src.root, nodes))
        n_in = jnp.where(live, jnp.minimum(n, nodes - 1), 0)
        accept = live & _pair_accept(
            src, n_in, centres[g], radii[g], thresholds[g], **walk
        )
        rejected = live & ~accept
        leaf = is_leaf[n]
        near = rejected & leaf & near_ok[n]
        opened = rejected & ~leaf & has_any[n]
        fg, fn, f_end = emit(accept, cnt[0], far_cap, fg, fn, g, n)
        ng, nn, n_end = emit(near, cnt[1], near_cap, ng, nn, g, n)
        head = (head + take) % qcap
        pend = pend - take
        rank = jnp.cumsum(opened, dtype=jnp.int32) - 1
        n_open = jnp.sum(opened, dtype=jnp.int32)
        kids = src.children[n]
        for k in range(2):
            slot = pend + 2 * rank + k
            fits = opened & (slot < qcap)
            at = jnp.where(fits, (head + slot) % qcap, qcap)
            qg = qg.at[at].set(g, mode="drop")
            qn = qn.at[at].set(kids[:, k], mode="drop")
        want = pend + 2 * n_open
        peak = jnp.maximum(peak, want)
        pend = jnp.minimum(want, qcap)
        seed = seed + jnp.clip(width - take, 0, jnp.maximum(n_live - seed, 0))
        tests = tests + jnp.sum(live, dtype=cdt)
        cnt = jnp.stack([f_end, n_end])
        return qg, qn, head, pend, seed, fg, fn, ng, nn, cnt, peak, tests

    zero = jnp.zeros((), jnp.int32)
    init = (
        jnp.zeros((qcap,), jnp.int32),
        jnp.full((qcap,), nodes, jnp.int32),
        zero,
        zero,
        zero,
        jnp.zeros((far_cap,), jnp.int32),
        jnp.full((far_cap,), nodes, jnp.int32),
        jnp.zeros((near_cap,), jnp.int32),
        jnp.full((near_cap,), nodes, jnp.int32),
        jnp.zeros((2,), jnp.int32),
        zero,
        jnp.zeros((), cdt),
    )
    out = jax.lax.while_loop(cond, body, init)
    fg, fn, ng, nn, cnt, peak, tests = out[5:]
    return fg, fn, ng, nn, cnt, jnp.stack([peak.astype(cdt), tests])


class _TargetWalk(NamedTuple):
    """The walk's outcome before the far/near passes (also for tests: pair sets).

    Attributes
    ----------
    src : _SourceTables
        The source tree.
    perm : Array
        ``[M]`` the points' Morton order (dead points last).
    groups : Array
        ``[groups, G, 3]`` the sorted points.
    gvalid : Array
        ``[groups, G]`` live points.
    centres : Array
        ``[groups, 3]`` the groups' centres.
    lists : tuple
        ``(far groups, far nodes, near groups, near leaves)``.
    totals : Array
        ``int32 [2]`` the far / near pair counts.
    stats : Array
        ``[3]`` queue peak, MAC tests, live points.
    ok : Array
        Every count fits the plan.
    eps : float
        The pair-kernel softening.
    kernel : str
        The softening kernel.
    order : int
        Expansion order.
    """

    src: _SourceTables
    perm: Array
    groups: Array
    gvalid: Array
    centres: Array
    lists: tuple
    totals: Array
    stats: Array
    ok: Array
    eps: float
    kernel: str
    order: int


def _target_walk(
    provider: Any,
    topology: Any,
    pts_in: Array,
    *,
    plan: TargetPlan,
    force_scale: Optional[Array],
    mac: Optional[str],
    softening: Optional[float],
    point_mask: Optional[Array],
    tables: Optional[_SourceTables] = None,
) -> _TargetWalk:
    """Groups and the walk of :func:`evaluate_target_points` (its arguments).

    Parameters
    ----------
    provider : Any
        The provider.
    topology : Any
        Its topology.
    pts_in : Array
        ``[M, 3]``.
    plan : TargetPlan
        Static shapes.
    force_scale : Optional[Array]
        ``[M]`` (``dehnen_error``).
    mac : Optional[str]
        Criterion override.
    softening : Optional[float]
        The points' softening override.
    point_mask : Optional[Array]
        ``[M]`` live points.
    tables : Optional[_SourceTables]
        The topology's :func:`_source_tables` when the caller has them.

    Returns
    -------
    _TargetWalk
        The groups and the pair lists.

    Raises
    ------
    ValueError
        If ``softening`` is negative.
    """
    from jaccpot.softening import resolve_softening_kernel, support_factor

    engine = provider.engine
    mac_name = target_mac_type(engine, mac)
    kernel = resolve_softening_kernel(getattr(engine, "softening_kernel", None))
    if softening is None:
        eps = float(engine.softening)
        floor = float(engine._walk_separation_floor())
    else:
        eps = float(softening)
        if eps < 0.0:
            raise ValueError(f"softening must be >= 0, got {eps}")
        factor = getattr(engine, "softening_floor", None)
        floor = float(support_factor(kernel) if factor is None else factor) * eps
    order = int(provider.max_order)
    src = (
        _source_tables(engine, topology, order=order, mac=mac_name)
        if tables is None
        else tables
    )
    dtype = src.radius.dtype
    nodes = int(src.radius.shape[0])
    dense = plan.walk == "dense"
    block = _walk_block(plan, nodes) if dense else 1
    pts = pts_in.astype(dtype)
    m = int(plan.num_points)
    gsize = int(plan.group_size)
    ngroups = -(-plan.num_groups // block) * block
    pad = ngroups * gsize - m
    live_in = None if point_mask is None else jnp.asarray(point_mask, bool)
    perm = _morton_order(pts, live_in)
    sorted_pts = pts[perm]
    valid = jnp.arange(ngroups * gsize) < m
    if live_in is not None:
        valid = valid & jnp.concatenate([live_in[perm], jnp.zeros((pad,), bool)])
    if pad:
        sorted_pts = jnp.concatenate(
            [sorted_pts, jnp.broadcast_to(sorted_pts[-1:], (pad, 3))]
        )
    groups = sorted_pts.reshape(ngroups, gsize, 3)
    gvalid = valid.reshape(ngroups, gsize)
    live_groups = jnp.any(gvalid, axis=1)
    lo = jnp.min(jnp.where(gvalid[..., None], groups, jnp.inf), axis=1)
    hi = jnp.max(jnp.where(gvalid[..., None], groups, -jnp.inf), axis=1)
    centres = jnp.where(live_groups[:, None], 0.5 * (lo + hi), 0.0)
    radii = jnp.sqrt(
        jnp.max(
            jnp.where(gvalid, jnp.sum((groups - centres[:, None, :]) ** 2, -1), 0.0),
            axis=1,
        )
    )
    if mac_name == "dehnen_error":
        fb = jnp.asarray(force_scale, dtype)[perm]
        if pad:
            fb = jnp.concatenate([fb, jnp.full((pad,), jnp.inf, dtype)])
        fb = jnp.where(valid, fb, jnp.inf).reshape(ngroups, gsize)
        thr = jnp.maximum(
            jnp.asarray(float(engine.adaptive_eps), dtype) * jnp.min(fb, axis=1),
            jnp.asarray(_TINY, dtype),
        )
    else:
        thr = jnp.zeros((ngroups,), dtype)
    chunk = int(plan.pair_chunk)
    far_cap = _ceil_to(int(plan.far_capacity), chunk)
    near_cap = _ceil_to(int(plan.near_capacity), chunk)
    walk = dict(
        mac=mac_name,
        theta=float(provider.theta_value),
        theta_max=float(getattr(engine, "mac_theta_max", 1.0)),
        floor=floor,
        order=order,
        far=str(plan.far),
    )
    n_points = jnp.sum(valid, dtype=_count_dtype())
    if dense:
        fg, fn, ng, nn, totals = _compact_pairs(
            src,
            centres,
            radii,
            thr,
            live_groups,
            block=block,
            far_cap=far_cap,
            near_cap=near_cap,
            walk=walk,
        )
        tests = jnp.sum(live_groups, dtype=_count_dtype()) * nodes
        stats = jnp.stack([jnp.zeros((), _count_dtype()), tests, n_points])
        ok = (totals[0] <= far_cap) & (totals[1] <= near_cap)
    else:
        fg, fn, ng, nn, totals, wstats = _frontier_pairs(
            src,
            centres,
            radii,
            thr,
            jnp.sum(live_groups, dtype=jnp.int32),
            chunk=int(plan.walk_chunk),
            queue_cap=int(plan.queue_capacity),
            far_cap=far_cap,
            near_cap=near_cap,
            walk=walk,
        )
        stats = jnp.concatenate([wstats, n_points[None]])
        ok = (
            (totals[0] <= far_cap)
            & (totals[1] <= near_cap)
            & (wstats[0] <= int(plan.queue_capacity))
        )
    return _TargetWalk(
        src=src,
        perm=perm,
        groups=groups,
        gvalid=gvalid,
        centres=centres,
        lists=(fg, fn, ng, nn),
        totals=totals,
        stats=stats,
        ok=ok,
        eps=eps,
        kernel=kernel,
        order=order,
    )


def evaluate_target_points(
    provider: Any,
    topology: Any,
    points: Array,
    *,
    plan: TargetPlan,
    force_scale: Optional[Array] = None,
    potential: bool = False,
    return_force_scale: bool = False,
    mac: Optional[str] = None,
    softening: Optional[float] = None,
    point_mask: Optional[Array] = None,
) -> TargetField:
    """The field of ``topology``'s sources at ``points``; traced or eager.

    Three passes, every one memory-bounded: the walk (:attr:`TargetPlan.walk`:
    the frontier walk by default, or the dense sweep), compacted into flat
    (group, node) far and (group, leaf) near lists of the plan's total
    capacities; then the far pairs (M2P) and the near pairs (softened P2P),
    ``plan.pair_chunk`` at a time, scatter-added into the groups' points.

    Parameters
    ----------
    provider : Any
        The :class:`~jaccpot.runtime.fused_provider.FusedForceProvider` the
        topology was rebuilt by (its engine's ``G``, softening, criterion).
    topology : Any
        A :class:`~jaccpot.runtime.fused_topology.FusedTopology` with a full
        prepared state (a rebuilt one, or an eager prepare's).
    points : Array
        ``[M, 3]`` target points; ``M == plan.num_points``.
    plan : TargetPlan
        Static shapes (:func:`plan_target_points`).
    force_scale : Optional[Array]
        ``[M]`` the points' eq (16b) ``f_b`` from a previous evaluation: required
        under ``mac_type='dehnen_error'`` (the groups' thresholds), ignored
        otherwise.
    potential : bool
        Also return the potential.
    return_force_scale : bool
        Also return the points' ``f_b`` for the next evaluation.
    mac : Optional[str]
        Criterion override (:func:`target_mac_type`).
    softening : Optional[float]
        The points' pair-kernel softening (Plummer-equivalent, the engine's
        kernel), with the target walk's separation floor following it
        (``softening_floor``, or the kernel's support factor, times it);
        ``None`` takes the engine's. A source tree built with a SMALL softening
        (its own walk stays O(n log n)) can then give the field of a larger one
        -- a compact cluster's back-reaction on heavy galaxy particles.
    point_mask : Optional[Array]
        ``[M]`` bool: only these points are evaluated (the others get zeros and
        cost nothing in the walk). A capacity-padded subset (the back-reaction's
        inner points) keeps a static ``M`` this way.

    Returns
    -------
    TargetField
        In the points' order and dtype.

    Raises
    ------
    ValueError
        If ``points`` is not ``[plan.num_points, 3]``, or ``dehnen_error`` has no
        ``force_scale``, or ``softening`` is negative.
    """
    engine = provider.engine
    mac_name = target_mac_type(engine, mac)
    pts_in = jnp.asarray(points)
    if pts_in.ndim != 2 or tuple(pts_in.shape) != (int(plan.num_points), 3):
        raise ValueError(
            f"points must be [{int(plan.num_points)}, 3] (the plan's), got "
            f"{tuple(pts_in.shape)}"
        )
    if mac_name == "dehnen_error" and force_scale is None:
        raise ValueError(
            "mac_type='dehnen_error': the target walk needs the points' force "
            "scale (a previous evaluation's return_force_scale=True, or "
            "bootstrap_target_force_scale)"
        )
    return _evaluate(
        provider,
        topology,
        pts_in,
        plan=plan,
        force_scale=force_scale,
        potential=potential,
        return_force_scale=return_force_scale,
        mac=mac,
        softening=softening,
        point_mask=point_mask,
        tables=None,
    )


def _evaluate(
    provider: Any,
    topology: Any,
    pts_in: Array,
    *,
    plan: TargetPlan,
    force_scale: Optional[Array],
    potential: bool,
    return_force_scale: bool,
    mac: Optional[str],
    softening: Optional[float],
    point_mask: Optional[Array],
    tables: Optional[_SourceTables],
) -> TargetField:
    """:func:`evaluate_target_points` after its checks, with optional tables.

    Parameters
    ----------
    provider : Any
        The provider.
    topology : Any
        Its topology.
    pts_in : Array
        ``[M, 3]`` checked points.
    plan : TargetPlan
        Static shapes.
    force_scale : Optional[Array]
        As in :func:`evaluate_target_points`.
    potential : bool
        As in :func:`evaluate_target_points`.
    return_force_scale : bool
        As in :func:`evaluate_target_points`.
    mac : Optional[str]
        As in :func:`evaluate_target_points`.
    softening : Optional[float]
        As in :func:`evaluate_target_points`.
    point_mask : Optional[Array]
        As in :func:`evaluate_target_points`.
    tables : Optional[_SourceTables]
        Precomputed :func:`_source_tables` (``None``: computed here).

    Returns
    -------
    TargetField
        As :func:`evaluate_target_points`.
    """
    from jaccpot.softening import softening_params

    engine = provider.engine
    tw = _target_walk(
        provider,
        topology,
        pts_in,
        plan=plan,
        force_scale=force_scale,
        mac=mac,
        softening=softening,
        point_mask=point_mask,
        tables=tables,
    )
    src, groups, gvalid, order, eps, kernel = (
        tw.src,
        tw.groups,
        tw.gvalid,
        tw.order,
        tw.eps,
        tw.kernel,
    )
    fg, fn, ng, nn = tw.lists
    dtype = src.radius.dtype
    ngroups, gsize = int(groups.shape[0]), int(groups.shape[1])
    m = int(plan.num_points)
    chunk = int(plan.pair_chunk)
    want_scale = bool(return_force_scale)
    want_pot = bool(potential)
    zeros3 = jnp.zeros((ngroups, gsize, 3), dtype)
    zeros1 = jnp.zeros((ngroups, gsize), dtype)

    def accumulate(carry: tuple, grp: Array, out: tuple) -> tuple:
        a, p, f = carry
        da, dp, df = out
        a = a.at[grp].add(da)
        p = p.at[grp].add(dp) if want_pot else p
        f = f.at[grp].add(df) if want_scale else f
        return a, p, f

    def far_body(carry: tuple, xs: tuple) -> tuple[tuple, None]:
        grp, node = xs
        out = _far_chunk(
            src,
            groups,
            gvalid,
            grp,
            node,
            order=order,
            G=float(engine.G),
            eps2_plummer=eps * eps,
            want_scale=want_scale,
        )
        return accumulate(carry, grp, out), None

    def near_body(carry: tuple, xs: tuple) -> tuple[tuple, None]:
        grp, leaf = xs
        out = _near_chunk(
            src,
            groups,
            gvalid,
            grp,
            leaf,
            leaf_cap=int(topology.prepared.max_leaf_size),
            G=float(engine.G),
            soft_params=softening_params(kernel, eps, dtype),
            kernel=kernel,
            eps2_plummer=eps * eps,
            want_potential=want_pot,
            want_scale=want_scale,
        )
        return accumulate(carry, grp, out), None

    def far_local_body(carry: tuple, xs: tuple) -> tuple[tuple, None]:
        loc, f = carry
        grp, node = xs
        dloc, df = _far_local_chunk(
            src,
            groups,
            gvalid,
            tw.centres,
            grp,
            node,
            order=order,
            G=float(engine.G),
            eps2_plummer=eps * eps,
            want_scale=want_scale,
        )
        loc = loc.at[grp].add(dloc)
        f = f.at[grp].add(df) if want_scale else f
        return (loc, f), None

    carry = (zeros3, zeros1, zeros1)
    if plan.far == "local":
        ncoef = int(src.multipoles.shape[1])
        (loc, scale0), _ = jax.lax.scan(
            far_local_body,
            (jnp.zeros((ngroups, ncoef), src.multipoles.dtype), zeros1),
            (fg.reshape(-1, chunk), fn.reshape(-1, chunk)),
        )
        acc0, pot0 = _l2p_groups(
            loc,
            groups,
            gvalid,
            tw.centres,
            order=order,
            G=float(engine.G),
            batch=max(1, chunk // gsize),
        )
        carry = (acc0.astype(dtype), pot0.astype(dtype), scale0)
    else:
        carry, _ = jax.lax.scan(
            far_body, carry, (fg.reshape(-1, chunk), fn.reshape(-1, chunk))
        )
    (acc, pot, scale), _ = jax.lax.scan(
        near_body, carry, (ng.reshape(-1, chunk), nn.reshape(-1, chunk))
    )
    out_dtype = pts_in.dtype

    def back(v: Array, tail: tuple) -> Array:
        # scatter by the permutation (not a second sort)
        rows = v.reshape((ngroups * gsize,) + tail)[:m].astype(out_dtype)
        return jnp.zeros((m,) + tail, out_dtype).at[tw.perm].set(rows)

    return TargetField(
        acc=back(acc, (3,)),
        potential=back(pot, ()) if potential else None,
        force_scale=back(scale, ()) if return_force_scale else None,
        needs=tw.totals,
        ok=tw.ok,
        stats=tw.stats,
    )


class TargetSources(NamedTuple):
    """Sources to rebuild a topology from INSIDE a planning program.

    Accepted wherever :func:`plan_target_points` and
    :func:`bootstrap_target_force_scale` take a topology: each planning round
    then rebuilds (``provider.rebuild``), as a step does, so no eager prepared
    state has to stay resident beside the target program (at 1e8 sources an
    eager topology is ~25 GB of an A100's 40).

    Attributes
    ----------
    positions : Array
        ``[N, 3]`` source positions.
    masses : Array
        ``[N]`` source masses.
    force_scale : Optional[Array]
        ``[N]`` the sources' f_b for a ``dehnen_error`` rebuild (``None``
        otherwise).
    """

    positions: Array
    masses: Array
    force_scale: Optional[Array]


def _jitted(provider: Any, plan: TargetPlan, **static: Any) -> Any:
    """``evaluate_target_points`` as one compiled program (the eager planner's).

    Parameters
    ----------
    provider : Any
        The provider.
    plan : TargetPlan
        Static capacities.
    **static : Any
        Further static keywords of :func:`evaluate_target_points`.

    Returns
    -------
    Any
        ``jit(run)(topology, points, force_scale, point_mask=None) -> TargetField``.
    """

    def run(
        topology: Any,
        points: Array,
        force_scale: Optional[Array],
        point_mask: Optional[Array] = None,
    ) -> TargetField:
        if isinstance(topology, TargetSources):
            topology = provider.rebuild(
                topology.positions, topology.masses, force_scale=topology.force_scale
            )
        return evaluate_target_points(
            provider,
            topology,
            points,
            plan=plan,
            force_scale=force_scale,
            point_mask=point_mask,
            **static,
        )

    return jax.jit(run)


def bootstrap_target_force_scale(
    provider: Any,
    topology: Any,
    points: Array,
    *,
    plan: TargetPlan,
    softening: Optional[float] = None,
    point_mask: Optional[Array] = None,
) -> tuple[Array, TargetField]:
    """The points' first ``f_b`` under ``dehnen_error``: one geometric pass.

    Parameters
    ----------
    provider : Any
        The provider.
    topology : Any
        Its topology.
    points : Array
        ``[M, 3]``.
    plan : TargetPlan
        Static shapes.
    softening : Optional[float]
        As in :func:`evaluate_target_points`.
    point_mask : Optional[Array]
        As in :func:`evaluate_target_points`.

    Returns
    -------
    tuple[Array, TargetField]
        ``(f_b, field)`` of a pass with the geometric criterion at the engine's
        theta.
    """
    field = _jitted(
        provider, plan, return_force_scale=True, mac="dehnen", softening=softening
    )(topology, points, None, point_mask)
    return field.force_scale, field


def plan_target_points(
    provider: Any,
    topology: Any,
    points: Array,
    *,
    group_size: int = 32,
    headroom: float = 1.5,
    force_scale: Optional[Array] = None,
    group_block: int = 64,
    pair_chunk: int = 128,
    max_rounds: int = 8,
    softening: Optional[float] = None,
    walk: str = "frontier",
    walk_chunk: int = 2048,
    point_mask: Optional[Array] = None,
    far: str = "m2p",
) -> TargetPlan:
    """Size a :class:`TargetPlan` eagerly from the points' current needs.

    Starts from small capacities and grows them from the evaluation's ``needs``
    (and the frontier queue's peak, :attr:`TargetField.stats`) until it fits,
    then adds ``headroom`` (moving points stay inside it; a step that does not
    reports ``ok=False``).

    Parameters
    ----------
    provider : Any
        The provider.
    topology : Any
        A topology at the sources' current positions, or :class:`TargetSources`
        (each round rebuilds it in-program).
    points : Array
        ``[M, 3]`` the points now.
    group_size : int
        Points per group.
    headroom : float
        Capacity multiplier over the measured needs.
    force_scale : Optional[Array]
        As in :func:`evaluate_target_points` (``dehnen_error``); ``None`` there
        bootstraps it with a geometric pass.
    group_block : int
        Upper bound on the groups walked together (the dense walk).
    pair_chunk : int
        Pairs evaluated together.
    max_rounds : int
        Re-plans before giving up.
    softening : Optional[float]
        The points' pair-kernel softening the evaluations will use
        (:func:`evaluate_target_points`).
    walk : str
        :attr:`TargetPlan.walk`.
    walk_chunk : int
        :attr:`TargetPlan.walk_chunk`.
    point_mask : Optional[Array]
        As in :func:`evaluate_target_points`.
    far : str
        :attr:`TargetPlan.far`.

    Returns
    -------
    TargetPlan
        A plan that fitted ``points`` with ``headroom``.

    Raises
    ------
    RuntimeError
        If no plan fitted within ``max_rounds``.
    """
    plan = TargetPlan(
        num_points=int(jnp.shape(points)[0]),
        group_size=int(group_size),
        group_block=int(group_block),
        pair_chunk=int(pair_chunk),
        walk=str(walk),
        walk_chunk=int(walk_chunk),
        queue_capacity=4 * int(walk_chunk),
        far=str(far),
    )
    if target_mac_type(provider.engine) == "dehnen_error" and force_scale is None:
        for _ in range(int(max_rounds)):
            fb, field = bootstrap_target_force_scale(
                provider,
                topology,
                points,
                plan=plan,
                softening=softening,
                point_mask=point_mask,
            )
            if bool(field.ok):
                break
            plan = plan.grown(field.needs, 1.0, field.stats)
        force_scale = fb
    for _ in range(int(max_rounds)):
        field = _jitted(provider, plan, softening=softening)(
            topology, points, force_scale, point_mask
        )
        if bool(field.ok):
            return plan.grown(field.needs, float(headroom), field.stats)
        plan = plan.grown(field.needs, 1.0, field.stats)
    raise RuntimeError(
        f"plan_target_points: no plan fitted in {max_rounds} rounds (last needs "
        f"{np.asarray(field.needs).tolist()}, stats "
        f"{np.asarray(field.stats).tolist()})"
    )


def plan_target_points_from_sources(
    provider: Any,
    positions: Array,
    masses: Array,
    points: Array,
    *,
    source_force_scale: Optional[Array] = None,
    **kwargs: Any,
) -> TargetPlan:
    """:func:`plan_target_points` without a resident topology.

    Each planning round's program rebuilds the sources' topology itself (as
    ``step`` / ``step_with_targets`` do), so the caller can drop an eager
    prepare's state first: the galaxy -> stars plan at 1e8 sources ran out of
    memory on an A100 next to the eager topology (HH2a).

    Parameters
    ----------
    provider : Any
        The provider.
    positions : Array
        ``[N, 3]`` the sources now.
    masses : Array
        ``[N]`` their masses.
    points : Array
        ``[M, 3]`` the points now.
    source_force_scale : Optional[Array]
        ``[N]`` the sources' f_b (a ``dehnen_error`` provider's rebuild needs
        it: ``accelerations(topology, force_scale=True)``).
    **kwargs : Any
        :func:`plan_target_points`'s keywords.

    Returns
    -------
    TargetPlan
        As :func:`plan_target_points`.
    """
    sources = TargetSources(
        positions=jnp.asarray(positions),
        masses=jnp.asarray(masses),
        force_scale=(
            None if source_force_scale is None else jnp.asarray(source_force_scale)
        ),
    )
    return plan_target_points(provider, sources, points, **kwargs)
